"""Multi-turn tool-calling vLLM rollout for L-NSDL agentic RL (P5.3).

Subclasses ``vLLMRollout`` (spmd) and overrides ``generate_sequences`` to drive
the L-NSDL agentic loop (``nsdl.tool_runtime.run_agentic_tool_loop`` via
``nsdl.rl.tool_loop_adapter``): generate -> parse ``[TOOL_CALL]`` -> run the
deterministic tool -> append observation -> generate, until ``[PLAN]`` or
``max_tool_rounds``.

It returns the standard verl tensors (``prompts``/``responses``/``input_ids``/
``attention_mask``/``position_ids``) PLUS a per-token ``loss_mask`` that masks
tool-observation tokens (the policy did not generate them), and attaches a
per-sample protocol ``trace`` into ``non_tensor_batch['extra_info']`` so the
``lnsdl`` reward reads real ``stopped_reason`` / ``emitted`` / ``executed``.

Selection: set ``actor_rollout_ref.rollout.name=vllm_multiturn`` (wired in
``fsdp_workers._build_rollout``) and the ``rollout.multi_turn.*`` config block.

DATA CONTRACT (produced by ``scripts/prepare_rl_data.py --mode lnsdl``): each
sample's ``non_tensor_batch['lnsdl_sample']`` is the minimal-payload dict
(instruction / prompt-derived robot_inventory / image_b64 / ground_truth.init_pos
/ ids) that ``LnsdlSampleContext.from_payload`` rebuilds into a BenchmarkSample,
then runs ``scene_parser.parse`` + NoOp refiner for scene_facts IDENTICAL to the
eval harness; ``data_source='lnsdl'`` and ``reward_model.ground_truth`` carries
``{gt_dag_json, gt_goal_json, time_steps, robots}`` for the reward.

STATUS: implemented; the masking/loop core is unit-tested with a fake engine,
and policy generates are batched round-by-round across active trajectories while
deterministic tool execution remains per trajectory.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, List, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.third_party.vllm import vllm_version
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _repeat_interleave

logger = logging.getLogger(__name__)
_MT_TRACE = os.environ.get("LNSDL_MT_TRACE", "0") == "1"

# Qwen2.5-VL vision markers. Kept as module constants because
# ``tokenize_vl_prompt`` has to reason about how many placeholders the chat
# template already emitted before deciding whether a literal ``<image>`` in the
# text is a placeholder or content (see the deficit logic there).
IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class vLLMMultiTurnRollout(vLLMRollout):
    """vLLM rollout that runs the L-NSDL multi-turn tool loop per sample."""

    def __init__(self, model_path, config, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self._model_path = model_path
        self._tokenizer = tokenizer
        self._processor = None
        self._scene_parser = None
        self._scene_refiner = None
        self._detector = None
        self._image_pad_id_cache = None
        mt = config.get("multi_turn", {}) if hasattr(config, "get") else {}
        self.mt_max_tool_rounds = int(mt.get("max_tool_rounds", 3)) if mt else 3
        self.mt_max_calls_per_turn = int(mt.get("max_tool_calls_per_turn", 1)) if mt else 1
        # P9 token-tail masking + cluster-10 behavioral truncation (drop the
        # stacked [PLAN] before parse so the schedule executes; debug_sftv3.md
        # §8.3). Default ON per plan.md P5.3′.0②′: the v3.3 init stacks ~50% of
        # ID turns (≥5% threshold), so rollouts without the guard diverge from
        # the eval harness and get zeroed by the E6 stack-gate. Set
        # multi_turn.stop_after_first_tool_call=False only for a non-stacking init.
        self.mt_stop_after_first_tool_call = (
            _as_bool(mt.get("stop_after_first_tool_call", True)) if mt else True
        )
        self.mt_batch_rounds = _as_bool(mt.get("batch_rounds", True)) if mt else True
        configured_context = config.get("max_model_len", None) if hasattr(config, "get") else None
        base_context = int(configured_context or (config.prompt_length + config.response_length))
        raw_context = mt.get("max_context_tokens", None) if mt else None
        raw_guard = mt.get("context_guard_tokens", 128) if mt else 128
        self.mt_max_context_tokens = int(raw_context or base_context)
        self.mt_context_guard_tokens = int(128 if raw_guard is None else raw_guard)
        self._lnsdl_ready = False

    # -- lazy heavy deps (processor / parser / refiner / detector) -----------
    def _ensure_lnsdl(self):
        if self._lnsdl_ready:
            return
        from transformers import AutoProcessor
        from nsdl.scene_parser import build_scene_parser
        from nsdl.scene_facts_refiner import build_scene_facts_refiner
        mt = self.config.get("multi_turn", {})
        model_path = mt.get("model_path") or self._model_path
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Defaults mirror scripts/eval_lnsdl_agentic_viki_l2.py (rule / none).
        self._scene_parser = build_scene_parser(mt.get("scene_parser_backend", "rule"))
        self._scene_refiner = build_scene_facts_refiner(mt.get("scene_facts_refiner_backend", "none"))
        # Detector optional; detect tool returns an error payload when absent.
        detector_backend = mt.get("detector_backend", "none")
        if detector_backend and detector_backend != "none":
            try:
                from nsdl.yolo_detector import build_structured_detector
                self._detector = build_structured_detector(
                    detector_backend, weights_path=mt.get("detector_weights")
                )
            except Exception:
                self._detector = None
        self._lnsdl_ready = True

    # -- per-call helpers handed to the adapter ------------------------------
    def _make_tokenize_fn(self):
        def tokenize(vl_messages, images):
            return tokenize_vl_prompt(
                vl_messages,
                images,
                processor=self._processor,
                tokenizer=self._tokenizer,
                image_pad_id=self._image_pad_id,
            )

        return tokenize

    @property
    def _image_pad_id(self) -> int:
        if getattr(self, "_image_pad_id_cache", None) is None:
            self._image_pad_id_cache = int(
                self._tokenizer.convert_tokens_to_ids("<|image_pad|>")
            )
        return self._image_pad_id_cache

    @property
    def _vision_special_ids(self) -> frozenset:
        # Tokens the POLICY must never contribute to the trained stream: an
        # <|image_pad|> it samples itself has no pixels behind it, so
        # expand_image_pad_tokens's drift guard (and, past it, the actor's
        # image-feature scatter) would reject the whole batch. Sampling at
        # temperature 1.0 makes this on-policy reachable (s42 died at step 31,
        # 2026-09-05: "2 placeholders but 1 image"). Same class as the trailing
        # <|vision_start|> device assert (debug_rl.md old §Q9).
        if getattr(self, "_vision_special_ids_cache", None) is None:
            ids = set()
            tok_obj = getattr(self, "_tokenizer", None)
            for tok in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"):
                tid = tok_obj.convert_tokens_to_ids(tok) if tok_obj is not None else None
                if tid is not None and tid != getattr(self._tokenizer, "unk_token_id", None):
                    ids.add(int(tid))
            self._vision_special_ids_cache = frozenset(ids)
        return self._vision_special_ids_cache

    def _image_pad_counts(self, images: list) -> tuple:
        """-> (pad token count per image, image_processor outputs) for ``images``.

        The count is ``grid.prod() // merge_size**2`` -- the same arithmetic
        ``RLHFDataset`` uses to expand the prompt, so the response expansion
        lands on the identical convention.
        """
        if not images:
            return [], None
        pils = [_to_pil(im) for im in images]
        inputs = self._processor.image_processor(pils, return_tensors="pt")
        merge_length = self._processor.image_processor.merge_size ** 2
        counts = [int(g.prod()) // merge_length for g in inputs["image_grid_thw"]]
        return counts, inputs

    def _expanded_prompt_len(self, prompt_len: int, images: list) -> int:
        """``prompt_len`` counted the way vLLM counts it -- images expanded.

        ``tokenize_vl_prompt`` deliberately leaves ``<|image_pad|>`` UNEXPANDED,
        one token per image, because that is the ``raw_prompt_ids`` convention
        verl hands the engine. vLLM expands each placeholder against
        ``multi_modal_data`` and only THEN validates against ``max_model_len``.
        So the raw length understates the real context by ~300 tokens for a
        scene image plus ~50 per crop, and the context guard below used to read
        that raw number: on 2026-08-07 both shards of the gate-3 variance audit
        died ~400/450 prompts in with "decoder prompt (length 9556) is longer
        than the maximum model length of 9216" -- a prompt the guard had just
        waved through. Same config as scripts/run_dapo_lnsdl_7b.sh, so this was
        reachable in P9 training too (debug_rl.md §Q7).

        The count is a pure function of ``(width, height)`` -- the processor's
        smart_resize maps the size to a patch grid -- so one probe per distinct
        size is exact, and worth caching: a multi-round tool loop re-tokenizes
        the same images on every round.
        """
        pils = [_to_pil(im) for im in (images or []) if im is not None]
        if not pils:
            return int(prompt_len)
        keys = [getattr(p, "size", None) for p in pils]
        if any(k is None for k in keys):  # not a PIL image; cannot key it, just probe
            counts, _ = self._image_pad_counts(pils)
            return int(prompt_len) + sum(int(c) - 1 for c in counts)
        cache = self.__dict__.setdefault("_image_pad_count_by_size", {})
        todo = [(k, p) for k, p in zip(keys, pils) if k not in cache]
        if todo:
            counts, _ = self._image_pad_counts([p for _, p in todo])
            for (key, _pil), count in zip(todo, counts):
                cache[key] = int(count)
        # -1 per image: the placeholder token is already counted in prompt_len.
        return int(prompt_len) + sum(cache[k] - 1 for k in keys)

    def _make_engine_generate_fn(self, sampling_override: dict):
        sampling_override = dict(sampling_override)
        sampling_override.pop("n", None)

        def engine_generate(prompt_token_ids, multi_modal_data, _sampling):
            vllm_input: dict = {"prompt_token_ids": list(prompt_token_ids)}
            if multi_modal_data is not None:
                vllm_input["multi_modal_data"] = multi_modal_data
            with self.update_sampling_params(n=1, **sampling_override):
                outputs = self.inference_engine.generate(
                    prompts=[vllm_input], sampling_params=self.sampling_params, use_tqdm=False
                )
            return list(outputs[0].outputs[0].token_ids)

        return engine_generate

    def _batch_engine_generate(
        self,
        vllm_inputs: list[dict],
        sampling_override: dict,
    ) -> list[list[int]]:
        sampling_override = dict(sampling_override)
        sampling_override.pop("n", None)
        with self.update_sampling_params(n=1, **sampling_override):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
            )
        return [list(out.outputs[0].token_ids) for out in outputs]

    # -- main entry ----------------------------------------------------------
    def _finalize_trajectory(self, traj) -> tuple:
        """-> ``(response_ids, loss_mask, kept_images)`` ready for the actor.

        Converts the rollout's vLLM-shaped response (one <|image_pad|> per crop)
        into the training shape (one pad per merged vision patch) and then
        truncates to ``response_length`` without splitting a placeholder run.
        Images whose run did not survive truncation are dropped so the pixel
        rows we attach always match the tokens that remain.
        """
        from nsdl.rl.tool_loop_adapter import (
            expand_image_pad_tokens,
            truncate_preserving_image_runs,
        )

        images = list(traj.response_images or [])
        ids, mask = list(traj.response_token_ids), list(traj.loss_mask)
        counts: list = []
        if images:
            counts, _ = self._image_pad_counts(images)
            ids, mask = expand_image_pad_tokens(
                ids, mask, image_pad_id=self._image_pad_id, pad_counts=counts
            )
        ids, mask, kept = truncate_preserving_image_runs(
            ids, mask, image_pad_id=self._image_pad_id,
            max_length=self.config.response_length,
        )
        # P5.3'.0-3 observation item: crops cost response budget, not VRAM
        # (docs/debug_rl.md §6.1). Record what this trajectory actually spent so
        # the E-run can read the crop token share instead of assuming it.
        if isinstance(getattr(traj, "trace", None), dict):
            traj.trace["response_image_count"] = len(images)
            traj.trace["response_image_kept"] = kept
            traj.trace["response_image_tokens"] = int(sum(counts[:kept]))
        return ids, mask, images[:kept]

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        from nsdl.rl.tool_loop_adapter import LnsdlSampleContext, run_tool_loop_collect

        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        self._ensure_lnsdl()

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        prompt_non_tensor_batch = prompts.non_tensor_batch
        if "lnsdl_sample" not in prompt_non_tensor_batch:
            raise RuntimeError(
                "vllm_multiturn rollout requires non_tensor_batch['lnsdl_sample']; "
                "regenerate the RL parquet with scripts/prepare_rl_data.py (lnsdl mode)."
            )
        samples = prompt_non_tensor_batch["lnsdl_sample"]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        n = 1 if (not do_sample or is_validate) else self.sampling_params.n
        sampling_override = _sampling_override(self.config, do_sample, is_validate)

        tokenize_fn = self._make_tokenize_fn()
        engine_generate_fn = self._make_engine_generate_fn(sampling_override)
        decode_fn = lambda ids: self._tokenizer.decode(ids, skip_special_tokens=True)  # noqa: E731

        response_lists: List[List[int]] = []
        loss_mask_lists: List[List[int]] = []
        traces: List[dict] = []
        response_images: List[list] = []
        if self.mt_batch_rounds:
            states = []
            for i in range(batch_size):
                ctx = LnsdlSampleContext.from_payload(
                    samples[i], scene_parser=self._scene_parser,
                    scene_refiner=self._scene_refiner, detector=self._detector,
                )
                for _ in range(n):
                    states.append(_TrajectoryState(ctx=ctx, messages=ctx.initial_messages()))
            trajectories = self._run_batched_tool_loop(
                states=states,
                tokenize_fn=tokenize_fn,
                sampling_override=sampling_override,
            )
            for traj in trajectories:
                resp, lm, imgs = self._finalize_trajectory(traj)
                response_lists.append(resp)
                loss_mask_lists.append(lm)
                response_images.append(imgs)
                traces.append(traj.trace)
        else:
            for i in range(batch_size):
                ctx = LnsdlSampleContext.from_payload(
                    samples[i], scene_parser=self._scene_parser,
                    scene_refiner=self._scene_refiner, detector=self._detector,
                )
                for _ in range(n):
                    traj = run_tool_loop_collect(
                        ctx,
                        tokenize_fn=tokenize_fn,
                        engine_generate_fn=engine_generate_fn,
                        decode_fn=decode_fn,
                        max_tool_rounds=self.mt_max_tool_rounds,
                        max_tool_calls_per_turn=self.mt_max_calls_per_turn,
                        stop_after_first_tool_call=self.mt_stop_after_first_tool_call,
                    )
                    resp, lm, imgs = self._finalize_trajectory(traj)
                    response_lists.append(resp)
                    loss_mask_lists.append(lm)
                    response_images.append(imgs)
                    traces.append(traj.trace)

        # ---- assemble verl tensors (mirror vLLMRollout, add loss_mask) ------
        response = pad_2d_list_to_length(
            response_lists, self.pad_token_id, max_length=self.config.response_length
        ).to(idx.device)
        loss_mask = pad_2d_list_to_length(
            loss_mask_lists, 0, max_length=self.config.response_length
        ).to(idx.device)
        # real (non-pad) response tokens: 1 for the first len(resp_i) positions.
        resp_real_mask = pad_2d_list_to_length(
            [[1] * len(r) for r in response_lists], 0, max_length=self.config.response_length
        ).to(idx.device)

        if n > 1:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            batch_size = batch_size * n
            if "multi_modal_inputs" in prompt_non_tensor_batch.keys():
                prompt_non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                    prompt_non_tensor_batch["multi_modal_inputs"], n
                )

        seq = torch.cat([idx, response], dim=-1)
        # attention covers ALL real tokens (incl. observations); loss_mask is the
        # strict assistant-only subset used by the policy/advantage/KL.
        attention_mask = torch.cat((attention_mask, resp_real_mask.to(attention_mask.dtype)), dim=-1)

        # P5.3′.0③: a trajectory that injected crop observations carries image
        # tokens INSIDE the response, so the actor forward needs their pixel rows
        # too, and mrope position ids must be recomputed over the whole sequence
        # (the plain arange extension below is only valid when the response has
        # no vision tokens).
        merged_mm_inputs = self._merge_multi_modal_inputs(
            prompt_non_tensor_batch.get("multi_modal_inputs"), response_images
        )
        response_length = response.size(1)
        if merged_mm_inputs is not None and any(len(im) for im in response_images):
            position_ids = self._recompute_mrope_position_ids(
                seq, attention_mask, merged_mm_inputs, fallback=position_ids
            )
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,  # (bs*n, response_length): 1=assistant, 0=obs/pad
            },
            batch_size=batch_size,
        )

        # per-sample protocol trace -> reward extra_info
        non_tensor_batch = {}
        if merged_mm_inputs is not None:
            non_tensor_batch["multi_modal_inputs"] = merged_mm_inputs
        non_tensor_batch["extra_info"] = np.array(
            [{"trace": t} for t in traces], dtype=object
        )

        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _merge_multi_modal_inputs(self, prompt_mm_inputs, response_images: list):
        """Append each trajectory's crop pixel rows to its prompt's image inputs.

        ``prompt_mm_inputs`` is verl's per-sample object array of
        ``{pixel_values, image_grid_thw}`` covering the scene image only. Crops
        appear later in the token stream, so their rows must be appended in the
        SAME order the placeholders appear -- concatenation order is what binds a
        picture to its tokens.

        Returns ``None`` when the batch has no multimodal inputs at all (the
        text-only configuration), leaving the caller's behavior unchanged.
        """
        if prompt_mm_inputs is None:
            return None
        merged = []
        for i, base in enumerate(prompt_mm_inputs):
            images = response_images[i] if i < len(response_images) else []
            if not images:
                merged.append(base)
                continue
            _counts, crop_inputs = self._image_pad_counts(images)
            entry = dict(base)
            for key in ("pixel_values", "image_grid_thw"):
                if key in entry and key in crop_inputs:
                    entry[key] = torch.cat(
                        [entry[key], crop_inputs[key].to(entry[key].dtype)], dim=0
                    )
            merged.append(entry)
        return np.array(merged, dtype=object)

    def _recompute_mrope_position_ids(self, seq, attention_mask, mm_inputs, fallback):
        """Recompute Qwen2-VL mrope position ids over prompt+response.

        Needed only when the response contains vision tokens: the arange
        extension used otherwise assumes position ids advance by one per token,
        which is false across an image span. Falls back to the caller's tensor if
        anything is off rather than emitting silently wrong positions.

        The result MUST keep ``fallback``'s rank. `get_rope_index` returns the
        full ``(3, seq)`` mrope block, but this repo's dataset stores only its
        temporal row (`rl_dataset.py:210` indexes `[0]` on a tensor whose batch
        dim `get_rope_index` had already removed), so prompt position ids arrive
        as ``(bs, seq)``. Emitting ``(bs, 3, seq)`` here made a crop-carrying gen
        batch non-concatenable with a crop-free one -- which is how the
        2026-08-04 re-run died at step 21 inside DAPO dynamic sampling
        (`Tensors must have same number of dimensions: got 2 and 3`), the first
        step whose resampling mixed the two kinds. Matching the caller keeps the
        convention consistent end-to-end; whether that convention should instead
        be full 3-D mrope is a separate, larger question (it would change
        position encoding for every multimodal RL run, not just crop ones).
        """
        try:
            from verl.models.transformers.qwen2_vl import get_rope_index
        except Exception:  # noqa: BLE001
            return _extend_position_ids(fallback, seq.size(-1) - fallback.size(-1))
        rows = []
        for i in range(seq.size(0)):
            grid = mm_inputs[i].get("image_grid_thw") if mm_inputs is not None else None
            try:
                rows.append(
                    get_rope_index(
                        self._processor,
                        input_ids=seq[i],
                        image_grid_thw=grid,
                        attention_mask=attention_mask[i],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Honour the docstring's promise for HOST-side failures only
                # (e.g. `image_grid_thw` having fewer rows than the token stream
                # has image spans -> IndexError). Those are recoverable: we lose
                # exact positions for one row instead of the whole run.
                #
                # This does NOT make the call crash-proof. A CUDA device-side
                # assert -- which is what the 2026-08-19 step-25 kill actually
                # was -- poisons the context process-wide and is not catchable
                # here; the only defence against that class is preventing the
                # bad index, done in `get_rope_index` itself.
                logger.warning(
                    "mrope recompute failed on row %d (%s: %s); falling back to "
                    "arange extension for this row -- positions across its image "
                    "spans will be approximate",
                    i,
                    type(exc).__name__,
                    exc,
                )
                return _extend_position_ids(fallback, seq.size(-1) - fallback.size(-1))
        out = torch.stack(rows, dim=0).to(seq.device)  # (bs, 3, seq)
        if fallback.dim() == 2:
            out = out[:, 0]  # temporal row only, matching the prompt convention
        return out

    def _run_batched_tool_loop(
        self,
        *,
        states: list["_TrajectoryState"],
        tokenize_fn,
        sampling_override: dict,
    ) -> list[Any]:
        """Run all active trajectories round-by-round, batching vLLM calls.

        Tool execution remains per trajectory because those calls are cheap and
        deterministic; only the expensive policy generate is batched.
        """
        from nsdl.rl.tool_loop_adapter import (
            AssembledTrajectory,
            assemble_from_records,
            build_trace,
            collect_response_images,
            truncate_gen_ids_to_first_tool_call,
        )
        from nsdl.rl.tool_loop_adapter import _RoundRecord  # same record type used by assemble_from_records
        from nsdl.slm_interface import SLMResponse, parse_slm_response
        from nsdl.tool_runtime import (
            ToolRuntimeContext,
            ToolRuntimeLoopResult,
            append_tool_result_messages,
            execute_tool_call,
            normalize_tool_call,
            slm_response_to_dict,
            tool_result_message,
            _truncate_assistant_to_tool_calls,
            _truncate_text_to_first_tool_call,
            _jsonable,
        )

        decode_fn = lambda ids: self._tokenizer.decode(ids, skip_special_tokens=True)  # noqa: E731
        max_calls = max(1, int(self.mt_max_calls_per_turn))
        active = list(range(len(states)))

        for round_index in range(self.mt_max_tool_rounds + 1):
            if not active:
                break

            vllm_inputs = []
            prompt_ids_list = []
            active_for_gen = []
            active_now = list(active)
            active = []
            for state_idx in active_now:
                state = states[state_idx]
                vl_messages, images = _default_render_messages(state.messages, state.ctx.image)
                prompt_token_ids, multi_modal_data = tokenize_fn(vl_messages, images)
                # Count what vLLM will count, not what the tokenizer returned:
                # the image placeholders are still unexpanded here.
                context_len = self._expanded_prompt_len(len(prompt_token_ids), images)
                if _context_budget_exhausted(
                    context_len,
                    self.mt_max_context_tokens,
                    self.mt_context_guard_tokens,
                ):
                    state.events.append({
                        "round": round_index,
                        "stop_after_turn": "max_context_length",
                        "context_length": int(context_len),
                        "context_length_unexpanded": int(len(prompt_token_ids)),
                        "max_context_tokens": int(self.mt_max_context_tokens),
                        "context_guard_tokens": int(self.mt_context_guard_tokens),
                        "tool_calls": [],
                        "executable_tool_calls": [],
                        "suppressed_tool_calls": [],
                        "tool_results": [],
                        "tool_messages": [],
                    })
                    # The tool observations appended after the last generated
                    # turn never reached the model: no _RoundRecord exists for
                    # this round, so the assembled response carries no
                    # <|image_pad|> for the crops they introduced, while
                    # collect_response_images below would still find the pixels.
                    # Roll the conversation back to the end of the last
                    # generated turn or _finalize_trajectory dies with
                    # "render/image drift". Every OTHER terminal path stops
                    # before execute_tool_call, which is why this only surfaced
                    # once the guard started firing (debug_rl.md §Q7/§Q8).
                    del state.messages[state.generated_message_count:]
                    # Map context exhaustion to the existing non-terminal reward
                    # channel (P_nonterm) instead of letting vLLM abort the job.
                    state.stopped_reason = "max_tool_rounds"
                    state.rounds = round_index
                    state.done = True
                    continue
                prompt_ids_list.append(list(prompt_token_ids))
                active_for_gen.append(state_idx)
                vllm_input: dict = {"prompt_token_ids": list(prompt_token_ids)}
                if multi_modal_data is not None:
                    vllm_input["multi_modal_data"] = multi_modal_data
                vllm_inputs.append(vllm_input)

            if not vllm_inputs:
                continue
            if _MT_TRACE:
                logger.warning("mt_trace round=%d generate:start n=%d prompt_lens=%s",
                               round_index, len(vllm_inputs), [len(v["prompt_token_ids"]) for v in vllm_inputs])
            generated_lists = self._batch_engine_generate(vllm_inputs, sampling_override)
            if _MT_TRACE:
                logger.warning("mt_trace round=%d generate:end gen_lens=%s",
                               round_index, [len(g) for g in generated_lists])
            for state_idx, prompt_token_ids, gen_ids in zip(active_for_gen, prompt_ids_list, generated_lists):
                state = states[state_idx]
                # Drop (not replace) policy-emitted vision markers: decode already
                # skips them, so the next-round prompt is re-tokenised without
                # them and the record stays prefix-aligned with it.
                scrub = getattr(self, "_vision_special_ids", None)
                if scrub and any(t in scrub for t in gen_ids):
                    kept = [t for t in gen_ids if t not in scrub]
                    state.scrubbed_vision_tokens += len(gen_ids) - len(kept)
                    gen_ids = kept
                raw_text = decode_fn(gen_ids)
                state.records.append(_RoundRecord(prompt_token_ids, list(gen_ids), raw_text))

                # Cluster 10 (debug_sftv3.md §8.3): behavioral truncation BEFORE
                # parse, matching the eval harness — the stacked [PLAN] tail is
                # dropped so the schedule executes and the next round emits the
                # repaired plan (token-level tail masking below stays as-is).
                parse_text = (
                    _truncate_text_to_first_tool_call(raw_text)
                    if self.mt_stop_after_first_tool_call
                    else raw_text
                )
                truncated_stacked_tail = parse_text != raw_text
                response = parse_slm_response(parse_text)
                if response.plan_sketch:
                    state.carried_plan_sketch = list(response.plan_sketch)
                executable_calls = response.tool_calls[:max_calls]
                suppressed_calls = response.tool_calls[max_calls:]
                assistant_content = (
                    _truncate_assistant_to_tool_calls(parse_text, executable_calls)
                    if (suppressed_calls or truncated_stacked_tail)
                    else parse_text
                )
                state.messages.append({"role": "assistant", "content": assistant_content})
                # Watermark: the conversation as of the end of a turn the model
                # actually generated. The context guard rewinds to it.
                state.generated_message_count = len(state.messages)
                event: dict[str, Any] = {
                    "round": round_index,
                    "assistant_raw_text": raw_text,
                    "assistant_content": assistant_content,
                    "truncated_stacked_tail": truncated_stacked_tail,
                    "parsed": slm_response_to_dict(response),
                    "tool_calls": [_jsonable(call) for call in response.tool_calls],
                    "executable_tool_calls": [_jsonable(call) for call in executable_calls],
                    "suppressed_tool_calls": [_jsonable(call) for call in suppressed_calls],
                    "max_tool_calls_per_turn": max_calls,
                    "tool_results": [],
                    "tool_messages": [],
                }

                if response.operations:
                    event["stop_after_turn"] = "final_plan"
                    state.events.append(event)
                    state.raw_text = raw_text
                    state.response = response
                    state.stopped_reason = "final_plan"
                    state.rounds = round_index
                    state.done = True
                    continue
                if not response.tool_calls:
                    event["stop_after_turn"] = "no_tool_call"
                    state.events.append(event)
                    state.raw_text = raw_text
                    state.response = response
                    state.stopped_reason = "no_tool_call"
                    state.rounds = round_index
                    state.done = True
                    continue
                if round_index >= self.mt_max_tool_rounds:
                    event["stop_after_turn"] = "max_tool_rounds"
                    event["unexecuted_tool_calls"] = [_jsonable(call) for call in response.tool_calls]
                    state.events.append(event)
                    state.raw_text = raw_text
                    state.response = response
                    state.stopped_reason = "max_tool_rounds"
                    state.rounds = round_index
                    state.done = True
                    continue

                ctx = ToolRuntimeContext(
                    scene_facts=state.ctx.scene_facts,
                    affordance_context=state.ctx.affordance_context,
                    image=state.ctx.image,
                    detector=state.ctx.detector,
                )
                event["plan_sketch_source"] = (
                    "current"
                    if response.plan_sketch
                    else ("carried" if state.carried_plan_sketch else "none")
                )
                if _MT_TRACE:
                    logger.warning("mt_trace round=%d tools:start state=%d calls=%s",
                                   round_index, state_idx, [_jsonable(c) for c in executable_calls])
                results = [
                    execute_tool_call(call, context=ctx, plan_sketch=state.carried_plan_sketch)
                    for call in executable_calls
                ]
                if _MT_TRACE:
                    logger.warning("mt_trace round=%d tools:end state=%d", round_index, state_idx)
                state.tool_results.extend(results)
                tool_messages = [
                    tool_result_message(normalize_tool_call(call).name, result)
                    for call, result in zip(executable_calls, results)
                ]
                event["tool_results"] = [_jsonable(result) for result in results]
                event["tool_messages"] = [_jsonable(message) for message in tool_messages]
                event["stop_after_turn"] = None
                state.events.append(event)
                state.messages = append_tool_result_messages(state.messages, executable_calls, results)
                state.raw_text = raw_text
                state.response = response
                state.rounds = round_index
                active.append(state_idx)

        trajectories = []
        for state in states:
            if not state.done:
                state.stopped_reason = "max_tool_rounds"
                state.rounds = self.mt_max_tool_rounds
            loop = ToolRuntimeLoopResult(
                response=state.response or SLMResponse(operations=[]),
                raw_text=state.raw_text,
                messages=state.messages,
                tool_results=state.tool_results,
                rounds=state.rounds,
                stopped_reason=state.stopped_reason,
                events=state.events,
            )
            # P9: enforce one-tool-call-per-turn on the TRAINED token stream (drop
            # the stacked tail so it carries no gradient). Safe no-op on ambiguity.
            if self.mt_stop_after_first_tool_call:
                for rec in state.records:
                    rec.generated_token_ids = truncate_gen_ids_to_first_tool_call(
                        rec.generated_token_ids, rec.text, decode_fn
                    )
            _prompt_ids, response_ids, loss_mask = assemble_from_records(state.records)
            trace = build_trace(loop, protocol_enforced=self.mt_stop_after_first_tool_call)
            if isinstance(trace, dict):
                trace["scrubbed_vision_tokens"] = int(state.scrubbed_vision_tokens)
            trajectories.append(
                AssembledTrajectory(
                    prompt_token_ids=_prompt_ids,
                    response_token_ids=response_ids,
                    loss_mask=loss_mask,
                    stopped_reason=loop.stopped_reason,
                    trace=trace,
                    response_images=collect_response_images(state.messages, state.ctx.image),
                )
            )
        return trajectories


@dataclass
class _TrajectoryState:
    ctx: Any
    messages: list[dict[str, Any]]
    records: list[Any] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    response: Any = None
    rounds: int = 0
    stopped_reason: str = "max_tool_rounds"
    scrubbed_vision_tokens: int = 0
    done: bool = False
    # F1: most-recent non-empty [PLAN_SKETCH], carried across rounds so op_ids in
    # a later-turn schedule/verify call resolve instead of failing closed.
    carried_plan_sketch: list = field(default_factory=list)
    # len(messages) at the end of the last turn the model generated. Seeded to
    # the initial prompt so a guard firing at round 0 rewinds to a no-op.
    generated_message_count: int = -1

    def __post_init__(self):
        if self.generated_message_count < 0:
            self.generated_message_count = len(self.messages)


def _default_render_messages(messages, image):
    """-> ``(vl_messages, images)``; ``images`` is scene + crops in template order."""
    from nsdl.rl.tool_loop_adapter import render_with_images
    return render_with_images(messages, image)


def tokenize_vl_prompt(vl_messages, images, *, processor, tokenizer, image_pad_id):
    """(vl_messages, ordered images) -> (prompt ids, vLLM ``multi_modal_data``).

    ``images`` is the FULL ordered list for this conversation: the scene image
    plus every crop a detect turn injected (P5.10.1). Passing only the scene
    image here was the P5.3′.0③ defect -- the extra ``<|image_pad|>``
    placeholders the template emits for crops would bind to nothing.

    Placeholders stay UNEXPANDED (one per image), which is what verl's
    ``RLHFDataset`` hands vLLM as ``raw_prompt_ids``; the engine expands them
    against ``multi_modal_data``. The trainable stream is expanded separately by
    ``expand_image_pad_tokens``.

    Module-level (not a closure) so the render/tokenize parity test can exercise
    the real function with a processor instead of a reimplementation.
    """
    text = processor.apply_chat_template(
        vl_messages, tokenize=False, add_generation_prompt=True
    )
    pils = [_to_pil(im) for im in (images or []) if im is not None]
    if not pils:
        return tokenizer.encode(text, add_special_tokens=False), None
    # Fallback for a renderer that left a literal <image> in the text (the
    # parts-list path already emits the vision tokens directly).
    #
    # It MUST be deficit-driven. `_build_messages` strips the literal `<image>`
    # only from the scene turn; anywhere else -- a tool observation, or text the
    # policy itself generated -- it survives as ordinary content. An
    # unconditional replace turns that content into a vision placeholder with no
    # picture behind it, which the guard below then (correctly) refuses. That is
    # what killed the 2026-08-04 40-step run at step 15: the policy had learned
    # to request crops heavily (crop_images/mean 0.02 -> 1.13), and one rollout
    # emitted a literal `<image>` in its own text => 4 placeholders, 3 images.
    #
    # So convert at most as many occurrences as there are still-unbound images.
    # On the real path the parts list already emitted one placeholder per image,
    # the deficit is 0, and a stray `<image>` stays text -- matching what the
    # eval-side processor does with the same conversation.
    deficit = len(pils) - text.count(IMAGE_PAD_TOKEN)
    if deficit > 0:
        text = text.replace("<image>", VISION_PLACEHOLDER, deficit)
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_placeholders = sum(1 for t in ids if t == image_pad_id)
    if n_placeholders != len(pils):
        raise RuntimeError(
            f"multimodal render mismatch: prompt has {n_placeholders} "
            f"<|image_pad|> placeholder(s) but {len(pils)} image(s) were collected. "
            "Rollout and training would bind different pictures."
        )
    return ids, {"image": pils}


def _extend_position_ids(position_ids, extra: int):
    """Arange extension of position ids (valid only for vision-free responses)."""
    if extra <= 0:
        return position_ids
    bs = position_ids.size(0)
    delta = torch.arange(1, extra + 1, device=position_ids.device)
    delta = delta.unsqueeze(0).expand(bs, -1)
    if position_ids.dim() == 3:  # qwen2vl mrope
        delta = delta.view(bs, 1, -1).expand(bs, 3, -1)
    return torch.cat([position_ids, position_ids[:, -1:] + delta], dim=-1)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _context_budget_exhausted(prompt_len: int, max_context_tokens: int, guard_tokens: int) -> bool:
    if max_context_tokens <= 0:
        return False
    return int(prompt_len) + max(0, int(guard_tokens)) >= int(max_context_tokens)


def _to_pil(image: Any):
    from PIL import Image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(image))).convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def _sampling_override(config, do_sample: bool, is_validate: bool) -> dict:
    if not do_sample:
        return {"best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
    if is_validate:
        return {
            "top_k": config.val_kwargs.top_k,
            "top_p": config.val_kwargs.top_p,
            "temperature": config.val_kwargs.temperature,
            "n": 1,
        }
    return {"n": 1}  # one completion per inner round; n trajectories looped outside
