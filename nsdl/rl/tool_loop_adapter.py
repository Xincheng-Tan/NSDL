"""Engine-agnostic adapter that bridges verl's rollout engine to the L-NSDL
multi-turn tool loop (``nsdl.tool_runtime.run_agentic_tool_loop``).

Why this exists (docs/plan.md §P5.3.3): verl's vLLM rollout is single-turn. The
L-NSDL policy is agentic — it emits ``[TOOL_CALL]`` blocks, the deterministic
``tool_runtime`` runs the tool, the observation is appended as a ``role=tool``
turn, and the model continues. RL must (a) drive that loop inside rollout, (b)
return a per-token ``loss_mask`` that masks the tool-observation tokens (the
policy did not generate them), and (c) surface the per-trajectory ``trace`` so
the lnsdl reward reads real ``stopped_reason`` / ``emitted`` / ``executed``.

Design (correctness-first):
  * We REUSE ``run_agentic_tool_loop`` verbatim — it already encodes the
    stop-on-second-[TOOL_CALL] suppress, json-split recovery, and stop logic
    (final_plan / no_tool_call / max_tool_rounds). Reimplementing it inside verl
    would risk the train/infer divergence this project has repeatedly hit.
  * The loop only needs a backend with ``generate(messages, **kwargs) -> str``.
    ``RecordingBackend`` wraps an engine generate-callable and, on every call,
    records the exact ``(prompt_token_ids, generated_token_ids)`` the engine
    saw/produced. The response token stream and ``loss_mask`` are then assembled
    DIRECTLY from those recorded ids — no token-boundary guessing, exact by
    construction (assistant tokens = generated = mask 1; everything the template
    inserted between turns, i.e. tool observations + role headers = mask 0).

This module is engine-agnostic and import-light so it is unit-testable with a
fake engine + fake tokenizer (see tests/test_tool_loop_adapter.py). The verl
rollout class (verl/.../vllm_multiturn_rollout.py) supplies the real vLLM engine
+ Qwen2.5-VL processor.
"""
from __future__ import annotations

import re
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

# tool_runtime is the single source of truth for the loop + tool-turn rendering.
from nsdl.tool_runtime import (
    build_initial_messages,
    run_agentic_tool_loop,
)


# ---------------------------------------------------------------------------
# Recording backend: wraps an engine generate-callable into the backend
# interface run_agentic_tool_loop expects, recording exact token ids per call.
# ---------------------------------------------------------------------------
@dataclass
class _RoundRecord:
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    text: str


# render_messages_fn(messages, image) -> vl_messages, or (vl_messages, images)
# (role=tool -> [TOOL_RESULT] user mapping + scene image into the first user
# turn + any per-turn crop images, P5.10.1). Default uses the SFT/inference
# renderer so rollout == training byte-for-byte. Both return shapes are accepted
# (see _normalize_render); the pair form is required for multi-image rollouts.
RenderFn = Callable[[Sequence[Mapping[str, Any]], Any], Any]
# tokenize_fn(vl_messages, images) -> (prompt_token_ids, multi_modal_data).
# ``images`` is the ORDERED list of every image referenced by vl_messages (scene
# first, then crops in appearance order) -- not a single image. The order is what
# binds each <|image_pad|> placeholder to its picture.
# CONTRACT (required for exact loss-mask assembly): tokenize_fn MUST render with
# add_generation_prompt=True (the assistant header is part of the PROMPT), and
# engine_generate_fn MUST return ONLY the newly-generated tokens (not the
# prompt). The chat template must be append-only (true for Qwen2.5). Under this
# contract round-(r+1) prompt = round-r prompt + round-r generated + tool turn +
# next assistant header, so the per-round diff is exactly the masked glue.
TokenizeFn = Callable[[List[dict], Any], tuple]
# engine_generate_fn(prompt_token_ids, multi_modal_data, sampling) -> token_ids
EngineGenerateFn = Callable[[List[int], Any, Any], List[int]]
# decode_fn(token_ids) -> str
DecodeFn = Callable[[List[int]], str]


def _default_render(messages: Sequence[Mapping[str, Any]], image: Any) -> tuple:
    """Reuse the inference/SFT message->vl_messages mapping for exact parity.

    Returns ``(vl_messages, images)`` -- the pair form, because a detect turn can
    carry crop images (P5.10.1) that must reach the tokenizer alongside the
    scene image.
    """
    from nsdl.slm_backends.local_vl import LocalVisionTextSLMBackend
    return LocalVisionTextSLMBackend._build_messages(messages, image)


def _normalize_render(rendered: Any, image: Any) -> tuple:
    """-> ``(vl_messages, images)`` from either render-fn return shape.

    ``_build_messages`` returns the pair. A legacy/text-only render fn returns a
    bare message list; then the only image in play is the scene image, so the
    list is ``[image]`` (or empty). Accepting both keeps custom render fns (and
    the fake-engine unit tests) working while the real path stays multi-image.
    """
    if isinstance(rendered, tuple) and len(rendered) == 2:
        vl_messages, images = rendered
        return list(vl_messages), [im for im in list(images) if im is not None]
    return list(rendered), ([image] if image is not None else [])


def render_with_images(
    messages: Sequence[Mapping[str, Any]],
    image: Any,
    render_messages_fn: Optional[RenderFn] = None,
) -> tuple:
    """Public seam: render a conversation to ``(vl_messages, images)``.

    The rollout uses this both to build each round's prompt and, at assembly
    time, to recover the trajectory's full ordered image list (scene + every
    crop observation) for the multimodal training inputs.
    """
    fn = render_messages_fn or _default_render
    return _normalize_render(fn(messages, image), image)


class RecordingBackend:
    """Backend for ``run_agentic_tool_loop`` backed by an external engine.

    Each ``generate`` call renders the running conversation to a prompt, asks the
    engine for completion token ids, decodes them to text (returned to the loop
    for parsing), and records ``(prompt_token_ids, generated_token_ids)``.
    """

    def __init__(
        self,
        *,
        render_messages_fn: RenderFn,
        tokenize_fn: TokenizeFn,
        engine_generate_fn: EngineGenerateFn,
        decode_fn: DecodeFn,
        image: Any = None,
        sampling: Any = None,
    ) -> None:
        self._render = render_messages_fn
        self._tokenize = tokenize_fn
        self._engine_generate = engine_generate_fn
        self._decode = decode_fn
        self._image = image
        self._sampling = sampling
        self.records: List[_RoundRecord] = []

    def generate(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        image = kwargs.get("image", self._image)
        vl_messages, images = render_with_images(messages, image, self._render)
        prompt_token_ids, multi_modal_data = self._tokenize(vl_messages, images)
        gen_ids = list(self._engine_generate(prompt_token_ids, multi_modal_data, self._sampling))
        text = self._decode(gen_ids)
        self.records.append(_RoundRecord(list(prompt_token_ids), gen_ids, text))
        return text


# ---------------------------------------------------------------------------
# Response + loss-mask assembly (exact, from recorded engine tokens)
# ---------------------------------------------------------------------------
@dataclass
class AssembledTrajectory:
    prompt_token_ids: List[int]          # round-0 prompt (= verl prompt) ids
    response_token_ids: List[int]        # everything after the prompt
    loss_mask: List[int]                 # 1 = assistant (generated) token, 0 = obs/glue
    stopped_reason: str
    trace: dict
    # Crop images injected by tool turns DURING the trajectory (P5.10.1), in
    # appearance order and excluding the scene image (which lives in the prompt
    # and already has pixel values from the dataset). The rollout turns these
    # into the extra `multi_modal_inputs` rows the actor forward needs, because
    # their <|image_pad|> placeholders sit inside `response_token_ids`.
    response_images: List[Any] = field(default_factory=list)


def assemble_from_records(records: Sequence[_RoundRecord]) -> tuple:
    """Return (prompt_ids, response_ids, loss_mask) from per-round records.

    The chat template is append-only (Qwen2.5): each round's prompt extends the
    previous (prompt + generated). So:
      * round-0 prompt           -> the RL prompt (excluded from response/mask).
      * round-r generated tokens -> assistant, mask 1.
      * tokens that appear in round-(r+1) prompt beyond (round-r prompt +
        round-r generated) -> template glue + tool observation, mask 0.
    """
    if not records:
        return [], [], []
    prompt0 = list(records[0].prompt_token_ids)
    response_ids: List[int] = []
    loss_mask: List[int] = []

    prev_full = list(prompt0)
    for i, rec in enumerate(records):
        # observation / glue = this round's prompt minus the previous full seq.
        if i > 0:
            glue = _suffix_after_prefix(prev_full, rec.prompt_token_ids)
            response_ids.extend(glue)
            loss_mask.extend([0] * len(glue))
            prev_full = list(rec.prompt_token_ids)
        # generated assistant tokens -> trainable.
        gen = list(rec.generated_token_ids)
        response_ids.extend(gen)
        loss_mask.extend([1] * len(gen))
        prev_full = prev_full + gen
    return prompt0, response_ids, loss_mask


def _suffix_after_prefix(prev_full: Sequence[int], cur_prompt: Sequence[int]) -> List[int]:
    """Tokens of ``cur_prompt`` beyond the ``prev_full`` prefix.

    Append-only templates guarantee ``cur_prompt`` starts with ``prev_full``; we
    fall back to a longest-common-prefix diff if a tokenizer re-segments a
    boundary (rare), so the mask never desynchronizes.
    """
    n = len(prev_full)
    if list(cur_prompt[:n]) == list(prev_full):
        return list(cur_prompt[n:])
    # robust fallback: longest common prefix
    lcp = 0
    for a, b in zip(prev_full, cur_prompt):
        if a != b:
            break
        lcp += 1
    return list(cur_prompt[lcp:])


# ---------------------------------------------------------------------------
# Image-placeholder expansion (vLLM stream -> trainable stream)
# ---------------------------------------------------------------------------
# The two consumers of a multimodal sequence disagree on how an image is spelled:
#
#   vLLM (rollout)   ONE <|image_pad|> per image; the engine expands it against
#                    multi_modal_data internally.
#   actor forward    ONE <|image_pad|> per MERGED VISION PATCH, i.e.
#                    grid.prod() // merge_size**2 copies -- this is what verl's
#                    RLHFDataset writes into the prompt `input_ids`.
#
# The response is assembled from the vLLM-side prompts, so any crop placeholder
# it carries is unexpanded. Concatenating it onto the dataset prompt (expanded)
# would hand Qwen2.5-VL fewer image tokens than image features. These helpers do
# the conversion, and keep truncation from ever slicing a placeholder run in half.
def expand_image_pad_tokens(
    token_ids: Sequence[int],
    loss_mask: Sequence[int],
    *,
    image_pad_id: int,
    pad_counts: Sequence[int],
) -> tuple:
    """Expand the i-th ``image_pad_id`` in ``token_ids`` into ``pad_counts[i]`` copies.

    Returns ``(ids, mask)``. Each expanded token inherits the placeholder's mask
    value (0 -- an observation the policy did not generate), so the loss mask
    stays aligned by construction.

    Raises ``ValueError`` when the placeholder count and ``pad_counts`` length
    disagree: that means the render and the image list have drifted apart, which
    would silently mis-bind crops to the wrong pictures. Fail loudly instead.
    """
    ids = list(token_ids)
    mask = list(loss_mask)
    if len(ids) != len(mask):
        raise ValueError(f"token_ids/loss_mask length mismatch: {len(ids)} vs {len(mask)}")
    n_placeholders = sum(1 for t in ids if t == image_pad_id)
    if n_placeholders != len(pad_counts):
        raise ValueError(
            f"response carries {n_placeholders} <|image_pad|> placeholder(s) but "
            f"{len(pad_counts)} image(s) were collected -- render/image drift"
        )
    if not pad_counts:
        return ids, mask
    out_ids: List[int] = []
    out_mask: List[int] = []
    seen = 0
    for tok, m in zip(ids, mask):
        if tok == image_pad_id:
            repeat = max(1, int(pad_counts[seen]))
            out_ids.extend([tok] * repeat)
            out_mask.extend([m] * repeat)
            seen += 1
        else:
            out_ids.append(tok)
            out_mask.append(m)
    return out_ids, out_mask


def truncate_preserving_image_runs(
    token_ids: Sequence[int],
    loss_mask: Sequence[int],
    *,
    image_pad_id: int,
    max_length: int,
) -> tuple:
    """Truncate to ``max_length`` without ever cutting inside a placeholder run.

    A partially-kept run would leave the sequence with fewer image tokens than
    the image it refers to has features, which the actor forward rejects. When
    the cut lands inside a run we retreat to the START of that run, dropping the
    whole image. Returns ``(ids, mask, n_images_kept)`` so the caller can trim
    the matching pixel-value rows.
    """
    ids = list(token_ids)
    mask = list(loss_mask)
    if max_length is not None and 0 <= max_length < len(ids):
        cut = max_length
        # inside a run iff this position and the previous one are both pads
        if cut > 0 and ids[cut - 1] == image_pad_id and cut < len(ids) and ids[cut] == image_pad_id:
            while cut > 0 and ids[cut - 1] == image_pad_id:
                cut -= 1
        ids, mask = ids[:cut], mask[:cut]
    # count surviving runs -> how many images the sequence still refers to
    kept = 0
    prev_pad = False
    for tok in ids:
        is_pad = tok == image_pad_id
        if is_pad and not prev_pad:
            kept += 1
        prev_pad = is_pad
    return ids, mask, kept


# Agentic block markers, kept in parity with slm_interface._AGENTIC_BLOCK_NAMES.
# A [TOOL_CALL] block has NO closing delimiter — it ends at the next block marker
# (or <think> / end-of-text), so "the second tool call" == the first marker that
# appears AFTER the first [TOOL_CALL].
_MARKER_RE = re.compile(r"\[(?:GOAL|ASSIGN|PLAN_SKETCH|TOOL_CALL|PLAN)\]|<think\b")


def truncate_gen_ids_to_first_tool_call(
    gen_ids: Sequence[int],
    raw_text: str,
    decode_fn: DecodeFn,
) -> List[int]:
    """Return the prefix of ``gen_ids`` ending at the FIRST tool call.

    P9 one-tool-call-per-turn protocol enforcement on the TRAINED token stream:
    when a turn stacks 2+ tool calls, drop the stacked tail so it carries no
    policy gradient (``assemble_from_records`` masks only what remains). This
    extends the existing conversation-text guard
    (``tool_runtime._truncate_assistant_to_tool_calls``) to the trained tokens.

    Contract: ``raw_text == decode_fn(gen_ids)``. The character cut is the start
    offset of the first agentic-block marker AFTER the first ``[TOOL_CALL]``. A
    clean single-call turn (0/1 marker after the first ``[TOOL_CALL]``) is
    returned unchanged.

    SAFETY: returns ``list(gen_ids)`` unchanged on ANY ambiguity — never emits an
    empty or corrupt token stream. Scope: the rarer "two JSON objects in one
    ``[TOOL_CALL]`` body" form produces only one marker and is returned unchanged
    (a small, documented residual; marker-based truncation covers the dominant
    multi-``[TOOL_CALL]``-block form).
    """
    gen_ids = list(gen_ids)
    n = len(gen_ids)
    if n == 0:
        return gen_ids

    # (a) first [TOOL_CALL], then the first marker strictly after it.
    markers = list(_MARKER_RE.finditer(raw_text))
    first_tc = next((m for m in markers if m.group(0) == "[TOOL_CALL]"), None)
    if first_tc is None:
        return gen_ids  # no tool call -> nothing to enforce
    cut_char = next((m.start() for m in markers if m.start() > first_tc.start()), None)
    if cut_char is None:
        return gen_ids  # single call, no stacked tail

    # (b) defensive: the char offsets are only meaningful if raw_text is exactly
    # the decode of gen_ids (guards a skip_special_tokens mismatch etc.).
    if decode_fn(gen_ids) != raw_text:
        return gen_ids

    # (c) largest k with len(decode(gen_ids[:k])) <= cut_char (decode length is
    # monotone in k for an append-only tokenizer) -> binary search, ~log2(n) decodes.
    lo, hi, best = 0, n, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(decode_fn(gen_ids[:mid])) <= cut_char:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # (d) reject empty / no-op / non-prefix / lost-tool-call cases.
    if not (0 < best < n):
        return gen_ids
    kept = decode_fn(gen_ids[:best])
    if not raw_text.startswith(kept) or "[TOOL_CALL]" not in kept:
        return gen_ids
    return gen_ids[:best]


# ---------------------------------------------------------------------------
# Per-sample rollout context
# ---------------------------------------------------------------------------
@dataclass
class LnsdlSampleContext:
    """Everything needed to run one agentic rollout + run its tools.

    Built by the data/rollout layer from a VIKI-L2 sample via the SAME path the
    eval harness uses (scene_parser.parse -> scene_refiner.refine ->
    build_affordance_context), so tool execution is identical to inference.
    """
    scene_facts: Any
    affordance_context: Any
    image: Any = None          # bytes (preferred) or PIL.Image
    detector: Any = None       # shared YOLO detector (process-global)

    @classmethod
    def from_sample(cls, sample: Any, *, scene_parser: Any, scene_refiner: Any,
                    detector: Any = None) -> "LnsdlSampleContext":
        from nsdl.aig_builder import build_affordance_context
        initial = scene_parser.parse(sample)
        scene_facts = scene_refiner.refine(sample, initial, yolo_detections=None)
        affordance_context = build_affordance_context(scene_facts)
        return cls(
            scene_facts=scene_facts,
            affordance_context=affordance_context,
            image=getattr(sample, "image_bytes", None),
            detector=detector,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, scene_parser: Any,
                     scene_refiner: Any, detector: Any = None) -> "LnsdlSampleContext":
        """Reconstruct from the parquet ``lnsdl_sample`` dict (scripts/prepare_rl_data.py
        --mode lnsdl), rebuilding the minimal BenchmarkSample the parser/refiner read.
        """
        import base64
        from nsdl.schema import BenchmarkSample
        img_b64 = payload.get("image_b64")
        image_bytes = base64.b64decode(img_b64) if img_b64 else None
        sample = BenchmarkSample(
            level=payload.get("level", "VIKI-L2"),
            split=payload.get("split", "train"),
            sample_index=int(payload.get("sample_index", 0)),
            task_id=payload.get("task_id"),
            task_name=payload.get("task_name"),
            instruction=payload.get("instruction", ""),
            system_prompt="",
            image_bytes=image_bytes,
            robot_inventory=dict(payload.get("robot_inventory") or {}),
            ground_truth=dict(payload.get("ground_truth") or {}),
        )
        return cls.from_sample(sample, scene_parser=scene_parser,
                               scene_refiner=scene_refiner, detector=detector)

    def initial_messages(self) -> List[dict]:
        return build_initial_messages(self.scene_facts, self.affordance_context)


# ---------------------------------------------------------------------------
# Top-level driver: run one agentic trajectory and assemble verl-ready outputs
# ---------------------------------------------------------------------------
def run_tool_loop_collect(
    ctx: LnsdlSampleContext,
    *,
    render_messages_fn: Optional[RenderFn] = None,
    tokenize_fn: TokenizeFn,
    engine_generate_fn: EngineGenerateFn,
    decode_fn: DecodeFn,
    sampling: Any = None,
    max_tool_rounds: int = 3,
    max_tool_calls_per_turn: int = 1,
    stop_after_first_tool_call: bool = False,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
) -> AssembledTrajectory:
    """Drive the agentic loop for one sample; return ids + loss_mask + trace."""
    backend = RecordingBackend(
        render_messages_fn=render_messages_fn or _default_render,
        tokenize_fn=tokenize_fn,
        engine_generate_fn=engine_generate_fn,
        decode_fn=decode_fn,
        image=ctx.image,
        sampling=sampling,
    )
    loop = run_agentic_tool_loop(
        backend,
        ctx.scene_facts,
        ctx.affordance_context,
        image=ctx.image,
        detector=ctx.detector,
        max_tool_rounds=max_tool_rounds,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        # Cluster 10 (debug_sftv3.md §8.3): behavioral truncation must match the
        # eval harness — drop the stacked [PLAN] tail BEFORE parsing so the
        # schedule executes and the next round emits the repaired plan. Without
        # this the rollout stops on the un-repaired stacked plan (R_task=0,
        # emitted≠executed -> E6 gate) while eval scores the repaired one.
        truncate_to_first_tool_call=stop_after_first_tool_call,
        generation_kwargs=dict(generation_kwargs or {}),
    )
    # P9: enforce one-tool-call-per-turn on the TRAINED token stream (drop the
    # stacked tail so it carries no gradient). Safe no-op on any ambiguity.
    if stop_after_first_tool_call:
        for rec in backend.records:
            rec.generated_token_ids = truncate_gen_ids_to_first_tool_call(
                rec.generated_token_ids, rec.text, decode_fn
            )
    prompt_ids, response_ids, loss_mask = assemble_from_records(backend.records)

    trace = build_trace(loop, protocol_enforced=stop_after_first_tool_call)
    return AssembledTrajectory(
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        loss_mask=loss_mask,
        stopped_reason=loop.stopped_reason,
        trace=trace,
        response_images=collect_response_images(
            loop.messages, ctx.image, render_messages_fn=render_messages_fn
        ),
    )


def collect_response_images(
    messages: Sequence[Mapping[str, Any]],
    scene_image: Any,
    *,
    render_messages_fn: Optional[RenderFn] = None,
) -> List[Any]:
    """Images a trajectory injected AFTER the prompt, in appearance order.

    The renderer is the single source of truth for image order (it is what binds
    placeholders to pictures), so we re-render the final conversation and drop
    the leading scene image -- that one lives in the round-0 prompt and already
    has pixel values from the dataset.
    """
    _vl, images = render_with_images(messages, scene_image, render_messages_fn)
    if scene_image is not None and images:
        return list(images[1:])
    return list(images)


def build_trace(loop: Any, protocol_enforced: bool = False) -> dict:
    """Self-contained trace with protocol fields and compact tool observations.

    The lnsdl reward reads protocol counts from this dict and, when enabled,
    checks whether final structured GOAL/PLAN blocks consume the returned tool
    observations. Keep this independent of the eval harness's richer
    ``loop_to_trace`` so RL has no dependency on scripts/.
    """
    events = loop.events or []
    emitted = sum(len(ev.get("tool_calls") or []) for ev in events)
    executed = sum(len(ev.get("tool_results") or []) for ev in events)
    stacked_turns = sum(1 for ev in events if (ev.get("suppressed_tool_calls") or []))
    # Cluster 10: with behavioral truncation ON the stacked tail is dropped
    # BEFORE parse, so it never reaches tool_calls/suppressed — the honest
    # stacking-rate diagnostic is the per-turn truncated_stacked_tail flag set
    # by run_agentic_tool_loop (0 when the guard is off or nothing stacked).
    truncated_turns = sum(1 for ev in events if ev.get("truncated_stacked_tail"))
    return {
        "stopped_reason": loop.stopped_reason,
        "rounds": loop.rounds,
        # Raw counts of what the policy ACTUALLY emitted (unaffected by P9 token
        # truncation) — this is the honest stacking-rate diagnostic the reward /
        # probe track. protocol_enforced flags that the stacked tail was masked
        # out of training so the two are distinguishable downstream.
        "emitted_tool_call_count": emitted,
        "executed_tool_call_count": executed,
        "stacked_tool_call_turns": stacked_turns,
        "truncated_stacked_turns": truncated_turns,
        "protocol_enforced": bool(protocol_enforced),
        "tool_observations": _compact_tool_observations(events),
    }


def _compact_tool_observations(events: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Return bounded, reward-facing summaries of executed tool results."""
    observations: List[dict] = []
    for ev in events:
        calls = list(ev.get("executable_tool_calls") or [])
        results = list(ev.get("tool_results") or [])
        for idx, result in enumerate(results):
            if not isinstance(result, Mapping):
                continue
            call = calls[idx] if idx < len(calls) and isinstance(calls[idx], Mapping) else {}
            obs = _compact_tool_result(call, result)
            if obs:
                observations.append(obs)
    return observations


def _compact_tool_result(call: Mapping[str, Any], result: Mapping[str, Any]) -> dict:
    name = _tool_name(call)
    obs: dict[str, Any] = {"name": name} if name else {}
    if "ok" in result:
        obs["ok"] = bool(result.get("ok"))

    entities = _collect_result_entities(result)
    relations = _dedupe_strings(_coerce_str_iter(result.get("relations")), limit=16)
    ops = _collect_result_ops(result)
    op_ids = _collect_result_op_ids(result)

    if entities:
        obs["entities"] = entities
    if relations:
        obs["relations"] = relations
    if ops:
        obs["ops"] = ops
    if op_ids:
        obs["op_ids"] = op_ids

    code = str(result.get("code") or "").strip()
    if code:
        obs["code"] = code
    return obs


def _tool_name(call: Mapping[str, Any]) -> str:
    return str(call.get("name") or call.get("tool") or call.get("tool_name") or "").strip()


def _collect_result_entities(result: Mapping[str, Any]) -> List[str]:
    entities: List[str] = []
    entities.extend(_coerce_str_iter(result.get("entities")))
    for key, value in result.items():
        if key in {"entities", "relations", "confidences", "ops", "operations", "primitives",
                   "schedule", "inserted", "reason", "ok", "code", "op_ids", "message"}:
            continue
        if isinstance(value, Mapping):
            if any(marker in value for marker in ("tags", "entity_type", "unknown")):
                entities.append(str(key))
        elif isinstance(value, AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
            entities.extend(_coerce_str_iter(value))
    return _dedupe_strings(entities, limit=16)


def _collect_result_ops(result: Mapping[str, Any]) -> List[dict]:
    raw_ops: List[Any] = []
    for key in ("ops", "operations"):
        value = result.get(key)
        if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
            raw_ops.extend(value)
    ops: List[dict] = []
    for raw in raw_ops:
        if not isinstance(raw, Mapping):
            continue
        op = _compact_op(raw)
        if op:
            ops.append(op)
    return ops[:24]


def _compact_op(raw: Mapping[str, Any]) -> dict:
    op: dict[str, Any] = {}
    for key in ("op_id", "op_type", "entity", "dest", "robot"):
        value = str(raw.get(key) or "").strip()
        if value:
            op[key] = value
    deps = _dedupe_strings(_coerce_str_iter(raw.get("deps") or raw.get("predecessor_op_ids")), limit=8)
    if deps:
        op["deps"] = deps
    return op


def _collect_result_op_ids(result: Mapping[str, Any]) -> List[str]:
    op_ids: List[str] = []
    op_ids.extend(_coerce_str_iter(result.get("op_ids")))
    reason = result.get("reason")
    if isinstance(reason, Mapping):
        op_ids.extend(_coerce_str_iter(reason.get("op_ids")))
    for op in _collect_result_ops(result):
        if op.get("op_id"):
            op_ids.append(str(op["op_id"]))
    return _dedupe_strings(op_ids, limit=24)


def _coerce_str_iter(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [str(key) for key in value.keys()]
    if isinstance(value, AbcSequence) and not isinstance(value, (bytes, bytearray)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (str, int, float, bool)):
                out.append(str(item))
        return out
    return [str(value)]


def _dedupe_strings(values: Iterable[str], *, limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break
    return out


__all__ = [
    "RecordingBackend",
    "LnsdlSampleContext",
    "AssembledTrajectory",
    "assemble_from_records",
    "build_trace",
    "collect_response_images",
    "expand_image_pad_tokens",
    "render_with_images",
    "run_tool_loop_collect",
    "truncate_preserving_image_runs",
]
