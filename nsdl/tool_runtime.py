"""Agentic tool runtime for L-NSDL trajectories.

The runtime is deliberately small: it normalizes model-emitted tool-call JSON,
routes calls to the deterministic/perception/catalog tools, and renders tool
results into chat messages. Dataset construction and inference should both use
these helpers so tool turns are byte-stable across training and rollout.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import json
import re
import time

from nsdl.catalog_tool import lookup
from nsdl.perc_tool import detect
from nsdl.planning_tool import (
    ExpandResult,
    ScheduleResult,
    ToolReason,
    expand,
    schedule,
    verify,
    verify_goal_coverage,
)
from nsdl.schema import AffordanceContext, MicroStep, OperationStep, SceneFacts
from nsdl.slm_interface import (
    SLMResponse,
    build_slm_system_prompt,
    parse_operation_steps,
    parse_slm_response,
    serialize_slm_input,
)


TOOL_ROLE = "tool"
LOSS_MASKED_TOOL_ROLES = frozenset({TOOL_ROLE})

# The deterministic backend Phi, as defined in docs/tools.md §0.3 ("one constraint
# model, three queries"). `detect` is perception (facet A) and NOT part of Phi;
# `verify_goal_coverage` is a read-only diagnostic of the model's own [GOAL] and
# is likewise not Phi. Used by the P10.S2 Phi-off arm to ablate exactly the
# deterministic solver while leaving the perception channel untouched.
PHI_TOOLS = frozenset({"schedule", "verify", "expand"})


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolRuntimeContext:
    scene_facts: SceneFacts | None = None
    affordance_context: AffordanceContext | None = None
    image: bytes | str | Path | Mapping[str, Any] | Any | None = None
    detector: Any | None = None
    # P5.10.2 probe: >0 auto-attaches ≤k crops to detect calls that omit
    # `crop_entities` (0 = off; historical behavior, no crops).
    auto_crop_top_k: int = 0
    # P5.10.4 bracket: when set, `detect` calls are answered by this callable
    # (`(query, crop_entities) -> DetectResult`) instead of the YOLO pipeline.
    # Used for the eval-only `off`/`oracle` perception-bracket backends.
    detect_override: Any | None = None
    # P10.S1 A-axis inversion (eval-only): `(DetectResult) -> DetectResult` hook
    # applied to whatever the perception path produced, so both the YOLO and the
    # `oracle` backends share the identical evidence before the arm diverges.
    # See `nsdl.verdict_perception`; None (default) = untouched evidence-only
    # behavior, so training/rollout paths are byte-unaffected.
    detect_postprocess: Any | None = None
    # P10.S2 Phi-off arm (eval-only): tool names answered with a
    # `tool_unavailable` error payload instead of being executed. Set to
    # PHI_TOOLS to measure the Phi-offload gain with the trained front-end and
    # the perception channel held fixed. Empty (default) = every tool runs, so
    # training/rollout paths are byte-unaffected.
    disabled_tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolRuntimeLoopResult:
    response: SLMResponse
    raw_text: str
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    rounds: int
    stopped_reason: str
    events: list[dict[str, Any]] = field(default_factory=list)
    # P6.1 point 4-AR (eval-only, --force-phi-reschedule): record of the
    # harness-forced `schedule` pass applied to a direct-emit final plan.
    # None = flag off, or the episode already executed a schedule call.
    forced_phi_reschedule: dict[str, Any] | None = None


def build_initial_messages(
    scene_facts: SceneFacts,
    affordance_context: AffordanceContext,
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Render the shared initial chat messages for training and inference."""
    return [
        {"role": "system", "content": system_prompt or build_slm_system_prompt()},
        {
            "role": "user",
            "content": user_prompt or serialize_slm_input(scene_facts, affordance_context),
        },
    ]


def normalize_tool_call(raw_call: Mapping[str, Any] | ToolCall) -> ToolCall:
    """Normalize a parsed `[TOOL_CALL]` payload."""
    if isinstance(raw_call, ToolCall):
        return raw_call
    name = str(
        raw_call.get("name")
        or raw_call.get("tool")
        or raw_call.get("tool_name")
        or ""
    ).strip()
    args = raw_call.get("args")
    if args is None:
        args = raw_call.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    return ToolCall(name=name, args=dict(args))


def execute_tool_call(
    raw_call: Mapping[str, Any] | ToolCall,
    *,
    scene_facts: SceneFacts | None = None,
    affordance_context: AffordanceContext | None = None,
    image: bytes | str | Path | Mapping[str, Any] | Any | None = None,
    detector: Any | None = None,
    context: ToolRuntimeContext | None = None,
    plan_sketch: Sequence[OperationStep] | None = None,
) -> dict[str, Any]:
    """Route one tool call and return a JSON-serializable result payload.

    ``plan_sketch`` is the parsed ``[PLAN_SKETCH]`` from the same assistant turn.
    Under the B1 op_ids-reference contract, ``schedule``/``verify`` args carry
    ``op_ids`` instead of a duplicated ``ops`` array; the ops are resolved from
    ``plan_sketch``. The legacy ``ops`` form is still accepted (backward compat
    with checkpoints trained before B1).
    """
    ctx = context or ToolRuntimeContext(
        scene_facts=scene_facts,
        affordance_context=affordance_context,
        image=image,
        detector=detector,
    )
    call = normalize_tool_call(raw_call)
    name = call.name.strip().lower()
    args = call.args

    # P10.S2 Phi-off arm. Checked before dispatch so a disabled tool is never
    # executed and never partially mutates context; the model sees a normal
    # tool-error turn and must produce the plan itself.
    if name in getattr(ctx, "disabled_tools", frozenset()):
        return _error_payload(
            "tool_unavailable",
            f"Tool '{call.name}' is not available in this configuration.",
        )

    try:
        if name == "schedule":
            result = schedule(
                _ops_from_args(args, plan_sketch),
                assign=_dict_arg(args, "assign"),
                ctx=ctx.affordance_context,
                scene_facts=ctx.scene_facts,
            )
            return _schedule_payload(result)

        if name == "verify":
            result = verify(
                _ops_from_args(args, plan_sketch),
                ctx=ctx.affordance_context,
                scene_facts=ctx.scene_facts,
            )
            return _tool_reason_payload(result)

        if name == "verify_goal_coverage":
            result = verify_goal_coverage(
                _dict_arg(args, "model_goal") or _dict_arg(args, "goal"),
                args.get("observed_facts") or args.get("observations") or args.get("detect_result"),
            )
            return _tool_reason_payload(result)

        if name == "expand":
            result = expand(
                _single_op_from_args(args),
                {
                    "ctx": ctx.affordance_context,
                    "scene_facts": ctx.scene_facts,
                },
            )
            return _expand_payload(result)

        if name == "detect":
            override = getattr(ctx, "detect_override", None)
            if override is not None:
                # P5.10.4 bracket backends (off/oracle): DetectResult-shaped
                # observations without the YOLO pipeline; never any crops.
                result = override(
                    args.get("query") if "query" in args else args,
                    args.get("crop_entities"),
                )
                return _detect_payload(_apply_detect_postprocess(ctx, result))
            if ctx.image is None:
                return _error_payload(
                    "missing_image",
                    "detect requires an image or precomputed detection payload in runtime context.",
                )
            result = detect(
                ctx.image,
                args.get("query") if "query" in args else args,
                detector=ctx.detector,
                crop_entities=args.get("crop_entities"),
                auto_crop_top_k=getattr(ctx, "auto_crop_top_k", 0),
            )
            return _detect_payload(_apply_detect_postprocess(ctx, result))

        if name == "lookup":
            if "tag" in args:
                return lookup(tag=str(args.get("tag") or ""))
            return lookup(names=args.get("names") or args.get("name") or args.get("entities"))

        return _error_payload("unknown_tool", f"Unknown tool: {call.name}")
    except _OpIdResolutionError as exc:
        return _error_payload("missing_plan_sketch", str(exc))
    except Exception as exc:  # noqa: BLE001 - tool turns should fail closed.
        return _error_payload(
            "tool_runtime_error",
            f"{exc.__class__.__name__}: {exc}",
        )


def execute_response_tool_calls(
    response_or_text: SLMResponse | str,
    *,
    context: ToolRuntimeContext | None = None,
    scene_facts: SceneFacts | None = None,
    affordance_context: AffordanceContext | None = None,
    image: bytes | str | Path | Mapping[str, Any] | Any | None = None,
    detector: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute all `[TOOL_CALL]` blocks in a parsed or raw assistant turn."""
    response = (
        parse_slm_response(response_or_text)
        if isinstance(response_or_text, str)
        else response_or_text
    )
    ctx = context or ToolRuntimeContext(
        scene_facts=scene_facts,
        affordance_context=affordance_context,
        image=image,
        detector=detector,
    )
    return [
        execute_tool_call(call, context=ctx, plan_sketch=response.plan_sketch)
        for call in response.tool_calls
    ]


def run_agentic_tool_loop(
    backend: Any,
    scene_facts: SceneFacts,
    affordance_context: AffordanceContext,
    *,
    image: bytes | str | Path | Mapping[str, Any] | Any | None = None,
    detector: Any | None = None,
    max_tool_rounds: int = 3,
    max_tool_calls_per_turn: int = 1,
    generation_kwargs: Mapping[str, Any] | None = None,
    auto_crop_top_k: int = 0,
    detect_override: Any | None = None,
    detect_postprocess: Any | None = None,
    truncate_to_first_tool_call: bool = False,
    disabled_tools: frozenset[str] | Sequence[str] | None = None,
    system_prompt: str | None = None,
    force_phi_reschedule: bool = False,
    force_phi_scope: str = "direct_emit",
    force_phi_reject_feedback: bool = False,
) -> ToolRuntimeLoopResult:
    """Run a minimal multi-turn tool loop around an SLM backend.

    The backend must expose ``generate(messages=..., **kwargs) -> str``. Tool
    results are appended with the same renderer used for SFT data construction.

    ``system_prompt`` overrides the default profile prompt; it exists for the
    P10.S2 trained Phi-off arm, whose checkpoint was fine-tuned under a prompt
    that never mentions schedule / verify / expand. It is orthogonal to
    ``disabled_tools``: the prompt decides what the model is told exists, the
    disable set decides what the runtime will actually run.

    ``force_phi_reschedule`` (P6.1 point 4-AR, eval-only): when the episode ends
    with a direct-emit ``[PLAN]`` and no ``schedule`` call was ever executed,
    the harness sends the model's ``[PLAN_SKETCH]`` ops (current turn first,
    else the carried sketch, else the final plan ops themselves) through
    ``schedule`` and replaces the final operations with the solver's completed
    ops. Samples that already executed a schedule call are untouched, so this
    is the single-factor "on-demand vs always-call" arm against point 4. The
    forced call is recorded under ``forced_phi_reschedule`` (NOT in the normal
    tool accounting, so TC comparisons stay clean and the report can add it
    explicitly). A failed forced schedule falls back to the model's own plan,
    mirroring the point-3 control's honest fallback.

    ``force_phi_scope`` selects **which** samples the forcing applies to, and
    the two values answer two different questions -- do not mix them in one
    table:
      - ``"direct_emit"`` (default, the frozen P6.1 point 4-AR semantics):
        only episodes that never executed a ``schedule`` call. This is the
        "on-demand vs always-call" arm as originally registered.
      - ``"always"`` (plan.md A0f-W1): every episode, including ones that
        already called schedule mid-episode. This is what the A0f decision
        actually specifies -- the final plan is *always* Phi-expanded. On the
        7B LOCUS main row the two scopes differ by a measured 17.9% of samples
        (52.1% never called schedule; a further 17.9% called it but ended with
        a plan that is not Phi's output), so ``"direct_emit"`` reads as a
        **lower bound** on the A0f effect and must be labelled as such if the
        write-once trigger line is evaluated against it.
    ``forced_phi_reschedule.schedule_executed_before`` records which subset a
    sample fell in, so one run can be split back into both readings.

    ``force_phi_reject_feedback`` (plan.md A0g②, 2026-09-01, default OFF to
    keep the frozen 4-AR/W1 semantics byte-identical): when the forced
    schedule REJECTS the sketch, do not silently fall back to the model's own
    plan. Instead the failed forced call and its ToolReason payload are
    injected into the message stream as a real tool turn and the model gets
    further rounds to repair, until ``max_tool_rounds`` is exhausted (only
    then does the honest fallback fire, with ``fallback_after_feedback`` in
    the record). The forced call stays OUT of the normal tool accounting
    either way; per-round records live in ``events[*].forced_phi_reschedule``.
    """
    messages = build_initial_messages(
        scene_facts, affordance_context, system_prompt=system_prompt
    )
    ctx = ToolRuntimeContext(
        scene_facts=scene_facts,
        affordance_context=affordance_context,
        image=image,
        detector=detector,
        auto_crop_top_k=auto_crop_top_k,
        detect_override=detect_override,
        detect_postprocess=detect_postprocess,
        disabled_tools=frozenset(disabled_tools or ()),
    )
    kwargs = dict(generation_kwargs or {})
    if image is not None and "image_bytes" not in kwargs and isinstance(image, bytes):
        kwargs["image_bytes"] = image

    all_tool_results: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    raw_text = ""
    response = SLMResponse(operations=[])
    max_calls = max(1, int(max_tool_calls_per_turn))
    # F1: op_ids look-back. In a multi-round tool loop the `[PLAN_SKETCH]` is
    # typically emitted in an earlier turn than the `schedule`/`verify` call that
    # references its op_ids. Carry the most-recent non-empty sketch across rounds
    # so op_ids resolve against it instead of failing with `missing_plan_sketch`.
    carried_plan_sketch: list[OperationStep] = []
    # 4-AR: direct-emit means "no schedule call ever executed in this episode".
    schedule_executed = False
    # A0g②: how many reject-feedback tool turns the harness has injected.
    phi_feedback_rounds = 0

    for round_index in range(max_tool_rounds + 1):
        raw_text = backend.generate(messages=messages, **kwargs)
        # Cluster 10 (docs/debug_sftv3.md §8): 7B stacks [TOOL_CALL] + [PLAN] in
        # one turn, so the loop stops with the model's own un-repaired PLAN
        # instead of executing the schedule call. Truncating the generation to
        # the first tool call drops the stacked tail -> the schedule executes ->
        # the next round emits the deterministically-repaired plan. Mirrors the
        # RL-side truncate_gen_ids_to_first_tool_call on the trained token stream.
        parse_text = (
            _truncate_text_to_first_tool_call(raw_text)
            if truncate_to_first_tool_call
            else raw_text
        )
        truncated_stacked_tail = parse_text != raw_text
        response = parse_slm_response(parse_text)
        if response.plan_sketch:
            carried_plan_sketch = list(response.plan_sketch)
        executable_calls = response.tool_calls[:max_calls]
        suppressed_calls = response.tool_calls[max_calls:]
        assistant_content = (
            _truncate_assistant_to_tool_calls(parse_text, executable_calls)
            if (suppressed_calls or truncated_stacked_tail)
            else parse_text
        )
        messages.append({"role": "assistant", "content": assistant_content})
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
            forced_record = None
            if force_phi_reschedule and (
                force_phi_scope == "always" or not schedule_executed
            ):
                response, forced_record = _forced_phi_reschedule_final(
                    response, carried_plan_sketch, ctx,
                    schedule_executed_before=schedule_executed,
                    scope=force_phi_scope,
                    reject_feedback=(
                        force_phi_reject_feedback and round_index < max_tool_rounds
                    ),
                )
                forced_record["feedback_rounds_before"] = phi_feedback_rounds
                if forced_record.get("outcome") == "fallback_model_plan":
                    forced_record["fallback_after_feedback"] = phi_feedback_rounds > 0
                event["forced_phi_reschedule"] = forced_record
                if forced_record.get("outcome") == "rejected_feedback_injected":
                    phi_feedback_rounds += 1
                    # A0g②: surface the rejection to the model as a real tool
                    # turn (ToolReason payload) and let it repair in an extra
                    # round. Deliberately NOT added to all_tool_results: the
                    # forced call stays out of the normal TC accounting, same
                    # as the frozen 4-AR convention.
                    forced_call = {
                        "name": "schedule",
                        "args": forced_record["call_args"],
                    }
                    forced_payload = forced_record["schedule_result"]
                    event["tool_results"] = [forced_payload]
                    event["tool_messages"] = [
                        tool_result_message("schedule", forced_payload)
                    ]
                    event["stop_after_turn"] = None
                    events.append(event)
                    messages = append_tool_result_messages(
                        messages, [forced_call], [forced_payload]
                    )
                    continue
            event["stop_after_turn"] = "final_plan"
            events.append(event)
            return ToolRuntimeLoopResult(
                response=response,
                raw_text=raw_text,
                messages=messages,
                tool_results=all_tool_results,
                rounds=round_index,
                stopped_reason="final_plan",
                events=events,
                forced_phi_reschedule=forced_record,
            )
        if not response.tool_calls:
            event["stop_after_turn"] = "no_tool_call"
            events.append(event)
            return ToolRuntimeLoopResult(
                response=response,
                raw_text=raw_text,
                messages=messages,
                tool_results=all_tool_results,
                rounds=round_index,
                stopped_reason="no_tool_call",
                events=events,
            )
        if round_index >= max_tool_rounds:
            event["stop_after_turn"] = "max_tool_rounds"
            event["unexecuted_tool_calls"] = [_jsonable(call) for call in response.tool_calls]
            events.append(event)
            return ToolRuntimeLoopResult(
                response=response,
                raw_text=raw_text,
                messages=messages,
                tool_results=all_tool_results,
                rounds=round_index,
                stopped_reason="max_tool_rounds",
                events=events,
            )

        event["plan_sketch_source"] = (
            "current"
            if response.plan_sketch
            else ("carried" if carried_plan_sketch else "none")
        )
        results = [
            execute_tool_call(call, context=ctx, plan_sketch=carried_plan_sketch)
            for call in executable_calls
        ]
        if any(
            normalize_tool_call(call).name.strip().lower() == "schedule"
            for call in executable_calls
        ):
            schedule_executed = True
        all_tool_results.extend(results)
        tool_messages = [
            tool_result_message(normalize_tool_call(call).name, result)
            for call, result in zip(executable_calls, results)
        ]
        event["tool_results"] = [_jsonable(result) for result in results]
        event["tool_messages"] = [_jsonable(message) for message in tool_messages]
        event["stop_after_turn"] = None
        events.append(event)
        messages = append_tool_result_messages(messages, executable_calls, results)

    return ToolRuntimeLoopResult(
        response=response,
        raw_text=raw_text,
        messages=messages,
        tool_results=all_tool_results,
        rounds=max_tool_rounds,
        stopped_reason="max_tool_rounds",
        events=events,
    )


def _forced_phi_reschedule_final(
    response: SLMResponse,
    carried_plan_sketch: Sequence[OperationStep],
    ctx: ToolRuntimeContext,
    *,
    schedule_executed_before: bool = False,
    scope: str = "direct_emit",
    reject_feedback: bool = False,
) -> tuple[SLMResponse, dict[str, Any]]:
    """P6.1 point 4-AR: force a direct-emit final plan through ``schedule``.

    Returns the (possibly replaced) response plus a self-describing record.
    Contract choices, in order of the spec (plan.md A0d(2)):
      - ops source: current-turn ``[PLAN_SKETCH]`` first, else the carried
        sketch from an earlier turn, else the final plan ops themselves
        (disclosed as ``final_plan_ops`` -- a legacy bare-[PLAN] emission has
        no sketch, and skipping it would silently shrink the always-call arm).
      - call shape: B1 op_ids reference when every sketch op carries an id
        (the exact resolver path a trained model call takes), else an explicit
        ops array -- same fallback the runtime accepts from models.
      - the model's own ``[ASSIGN]`` rides along, as it would on a real call.
      - failure (schedule error / not ok / unparseable ops) falls back to the
        model's own plan, mirroring the point-3 control: the front-end cannot
        be asked again, so the honest output is what it already produced.
    """
    started = time.perf_counter()
    if response.plan_sketch:
        ops_source, source = list(response.plan_sketch), "current_sketch"
    elif carried_plan_sketch:
        ops_source, source = list(carried_plan_sketch), "carried_sketch"
    else:
        ops_source, source = list(response.operations), "final_plan_ops"

    record: dict[str, Any] = {
        "applied": True,
        "source": source,
        "model_plan": [_operation_payload(op) for op in response.operations],
        "sketch_op_count": len(ops_source),
        "scope": scope,
        "schedule_executed_before": bool(schedule_executed_before),
    }

    ids = [str(op.op_id) for op in ops_source if getattr(op, "op_id", None)]
    if ids and len(ids) == len(ops_source):
        args: dict[str, Any] = {"op_ids": ids}
    else:
        args = {"ops": [_operation_payload(op) for op in ops_source]}
    if isinstance(response.assign, Mapping) and response.assign:
        args["assign"] = dict(response.assign)
    record["call_shape"] = "op_ids" if "op_ids" in args else "ops"
    record["call_args"] = _jsonable(args)

    payload = execute_tool_call(
        {"name": "schedule", "args": args},
        context=ctx,
        plan_sketch=ops_source,
    )
    record["schedule_result"] = _jsonable(payload)
    record["latency_s"] = round(time.perf_counter() - started, 4)

    new_ops: list[OperationStep] | None = None
    if isinstance(payload, Mapping) and payload.get("ok") and payload.get("ops"):
        try:
            new_ops = parse_operation_steps(payload["ops"])
        except Exception:  # noqa: BLE001 - fall back to the model plan below.
            new_ops = None
    if new_ops:
        record["outcome"] = "rescheduled"
        record["forced_plan_op_count"] = len(new_ops)
        return replace(response, operations=new_ops), record

    reject_reason = (
        str(payload.get("code") or "schedule_not_ok_or_empty")
        if isinstance(payload, Mapping)
        else "schedule_payload_not_mapping"
    )
    if reject_feedback:
        # A0g②: caller injects this failed call + ToolReason as a tool turn
        # and continues the loop; the response is returned untouched.
        record["outcome"] = "rejected_feedback_injected"
        record["reject_reason"] = reject_reason
        return response, record
    record["outcome"] = "fallback_model_plan"
    record["fallback_reason"] = reject_reason
    return response, record


def forced_phi_micro_steps(record: Mapping[str, Any] | None) -> list[MicroStep] | None:
    """Return Phi's own micro-step schedule from a forced-reschedule record.

    plan.md A0k (2026-09-04, facts:E24c): the forced arm used to replace the
    model's operations with ``schedule_result.ops`` and let the eval driver
    re-expand them through ``dispatch_graph``. That re-expansion is NOT Phi's
    schedule: ``ops`` carries only the custody/precondition deps Phi derives,
    while the same-robot end-effector serialization lives in the list
    scheduler and is only visible in ``schedule_result.schedule``. On the 7B
    test_id column this turned 10/180 rows whose Phi schedule was byte-equal
    to GT into ee-capacity failures (two Grasps before a Place). The final
    plan of the forced arm must therefore be Phi's ``schedule`` itself.

    Returns None when the record is absent, not applied, not rescheduled, or
    carries no schedule -- callers then keep the driver's re-expansion (which
    for a non-forced episode is the model's own plan).
    """
    if not isinstance(record, Mapping) or not record.get("applied"):
        return None
    if record.get("outcome") != "rescheduled":
        return None
    payload = record.get("schedule_result")
    if not isinstance(payload, Mapping) or not payload.get("ok"):
        return None
    steps = payload.get("schedule")
    if not isinstance(steps, Sequence) or not steps:
        return None
    out: list[MicroStep] = []
    for raw in steps:
        if is_dataclass(raw):
            raw = asdict(raw)
        if not isinstance(raw, Mapping):
            return None
        actions = raw.get("actions")
        if not isinstance(actions, Mapping):
            return None
        out.append(
            MicroStep(
                step=int(raw.get("step", len(out) + 1)),
                actions={str(k): [str(x) for x in v] for k, v in actions.items() if v},
            )
        )
    return out


def slm_response_to_dict(response: SLMResponse) -> dict[str, Any]:
    """Return a compact parsed response payload for debug artifacts."""
    return {
        "think": list(response.think),
        "goal": _jsonable(response.goal),
        "assign": _jsonable(response.assign),
        "plan_sketch": [_operation_payload(op) for op in response.plan_sketch],
        "tool_calls": [_jsonable(call) for call in response.tool_calls],
        "plan": [_operation_payload(op) for op in response.operations],
        "block_counts": {
            name: len(bodies)
            for name, bodies in sorted(response.raw_block_lists.items())
        },
        "raw_blocks": dict(response.raw_blocks),
        "parse_diagnostics": [_jsonable(item) for item in response.parse_diagnostics],
    }


def render_tool_result(result: Any) -> str:
    """Render a tool result as stable compact JSON."""
    return json.dumps(
        _jsonable(result),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def tool_result_message(
    tool_name: str,
    result: Any,
    *,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    """Return a chat tool turn. Loss masking is keyed by ``role == "tool"``.

    When ``result`` carries crop pixels (P5.10.1, ``_crop_images`` out-of-band key
    from ``_detect_payload``), the content becomes a parts list ``[text, image…]``
    so the tool turn re-injects zoomed-in image observations. Without crops the
    content stays a plain ``str`` (historical trajectories unchanged).
    """
    crop_images = result.get("_crop_images") if isinstance(result, Mapping) else None
    rendered = render_tool_result(result)  # text-only (drops `_`-prefixed keys)
    content: Any
    if crop_images:
        content = [{"type": "text", "text": rendered}]
        for crop in crop_images:
            image = crop.get("image") if isinstance(crop, Mapping) else crop
            if image is None:
                continue
            part: dict[str, Any] = {"type": "image", "image": image}
            entity = crop.get("entity") if isinstance(crop, Mapping) else None
            if entity:
                part["entity"] = entity
            content.append(part)
    else:
        content = rendered
    message: dict[str, Any] = {
        "role": TOOL_ROLE,
        "name": str(tool_name).strip(),
        "content": content,
    }
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def append_tool_result_messages(
    messages: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any] | ToolCall],
    results: Sequence[Any],
) -> list[dict[str, Any]]:
    """Return messages plus one rendered tool turn per call/result pair."""
    rendered = [dict(message) for message in messages]
    for call, result in zip(calls, results):
        normalized = normalize_tool_call(call)
        rendered.append(tool_result_message(normalized.name, result))
    return rendered


# Cluster 10 truncation: agentic-block markers that may follow a stacked
# [TOOL_CALL] (the model should stop the turn AT the first tool call).
_STACK_MARKER_RE = re.compile(r"\[(?:GOAL|ASSIGN|PLAN_SKETCH|TOOL_CALL|PLAN)\]|<think\b")


def _truncate_text_to_first_tool_call(raw_text: str) -> str:
    """Return the prefix of ``raw_text`` ending before the first agentic-block
    marker that appears AFTER the first ``[TOOL_CALL]``.

    Text-level analogue of ``nsdl.rl.tool_loop_adapter.truncate_gen_ids_to_first_
    tool_call`` (docs/debug_sftv3.md §8). A clean single-call turn (no marker
    after the first ``[TOOL_CALL]``, e.g. ``[TOOL_CALL]{...}`` then EOS) is
    returned unchanged. Turns with no tool call are returned unchanged.
    """
    markers = list(_STACK_MARKER_RE.finditer(raw_text))
    first_tc = next((m for m in markers if m.group(0) == "[TOOL_CALL]"), None)
    if first_tc is None:
        return raw_text
    cut = next((m.start() for m in markers if m.start() > first_tc.start()), None)
    if cut is None:
        return raw_text
    truncated = raw_text[:cut].rstrip()
    # Never drop the tool call itself (defensive; cut is strictly after it).
    return truncated if "[TOOL_CALL]" in truncated else raw_text


def _truncate_assistant_to_tool_calls(raw_text: str, calls: Sequence[Mapping[str, Any]],) -> str:
    """Keep the assistant turn through the first executable tool call(s).

    This implements the P5.6 stop-on-second-[TOOL_CALL] guard for backends that
    do not support custom stopping criteria. The raw text remains in trace
    events; the conversational context only receives the executable prefix.
    """
    if not calls:
        return raw_text
    match = re.search(r"\[TOOL_CALL\]", raw_text)
    if not match:
        return raw_text
    prefix = raw_text[: match.start()].rstrip()
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    for call in calls:
        parts.extend([
            "[TOOL_CALL]",
            json.dumps(_jsonable(call), ensure_ascii=False, sort_keys=True),
        ])
    return "\n".join(parts)


def is_loss_masked_message(message: Mapping[str, Any]) -> bool:
    return str(message.get("role") or "") in LOSS_MASKED_TOOL_ROLES


class _OpIdResolutionError(Exception):
    """Schedule/verify args referenced op_ids not resolvable against the sketch."""


def _ops_from_args(
    args: Mapping[str, Any],
    plan_sketch: Sequence[OperationStep] | None = None,
) -> list[OperationStep]:
    explicit = (
        args.get("ops")
        or args.get("operations")
        or args.get("plan")
        or args.get("plan_sketch")
    )
    if explicit:
        # Legacy contract (pre-B1) or an explicit ops array: use it directly.
        return parse_operation_steps(explicit)
    op_ids = args.get("op_ids")
    if op_ids:
        # B1 op_ids-reference contract: resolve ops from the same-turn sketch.
        return _resolve_ops_by_id(op_ids, plan_sketch)
    return parse_operation_steps([])


def _resolve_ops_by_id(
    op_ids: Any,
    plan_sketch: Sequence[OperationStep] | None,
) -> list[OperationStep]:
    ids = [str(x) for x in op_ids if str(x)] if isinstance(op_ids, (list, tuple)) else []
    if not ids:
        raise _OpIdResolutionError("schedule op_ids were empty or malformed")
    sketch = list(plan_sketch or [])
    if not sketch:
        raise _OpIdResolutionError(
            "schedule referenced op_ids but no parseable [PLAN_SKETCH] was emitted "
            "in this or any prior turn"
        )
    by_id = {op.op_id: op for op in sketch if getattr(op, "op_id", None)}
    resolved: list[OperationStep] = []
    missing: list[str] = []
    for oid in ids:
        op = by_id.get(oid)
        if op is None:
            missing.append(oid)
        else:
            resolved.append(op)
    if missing:
        raise _OpIdResolutionError(f"op_ids not found in [PLAN_SKETCH]: {missing}")
    return resolved


def _single_op_from_args(args: Mapping[str, Any]) -> OperationStep:
    payload = args.get("op") or args.get("node") or args.get("operation")
    if payload is None:
        ops = _ops_from_args(args)
    else:
        ops = parse_operation_steps([payload] if isinstance(payload, Mapping) else payload)
    if not ops:
        raise ValueError("expand requires one op")
    return ops[0]


def _dict_arg(args: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = args.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _apply_detect_postprocess(ctx: ToolRuntimeContext, result: Any) -> Any:
    """Apply the eval-only P10.S1 detect hook, if one is configured.

    Placed after *both* perception paths so the A-axis inversion arm and its
    evidence-only control share byte-identical perception (see
    ``nsdl.verdict_perception``). Default (no hook) is a pass-through.
    """
    postprocess = getattr(ctx, "detect_postprocess", None)
    if postprocess is None:
        return result
    return postprocess(result)


def _detect_payload(result: Any) -> dict[str, Any]:
    """Serialize a ``DetectResult``; carry crop pixels out-of-band for the runtime.

    The JSON/text payload is ``result.to_dict()`` (crop *metadata* only). When
    crops exist, the PIL images ride under ``_crop_images`` (a `_`-prefixed key
    stripped by ``_jsonable``), consumed only by ``tool_result_message`` to build
    the tool turn's image parts (P5.10.1).
    """
    payload = result.to_dict()
    crops = getattr(result, "crops", None)
    if crops:
        payload["_crop_images"] = [
            {"entity": crop.entity, "image": crop.image} for crop in crops
        ]
    return payload


def _schedule_payload(result: ScheduleResult) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "ops": [_operation_payload(op) for op in result.ops],
        "schedule": [_jsonable(step) for step in result.schedule],
        "inserted": list(result.inserted),
        "reason": _tool_reason_payload(result.reason) if result.reason else None,
    }


def _expand_payload(result: ExpandResult) -> dict[str, Any]:
    return {
        "ok": bool(result.ok),
        "primitives": _jsonable(result.primitives),
        "end_state": list(result.end_state),
        "reason": _tool_reason_payload(result.reason) if result.reason else None,
    }


def _tool_reason_payload(reason: ToolReason | None) -> dict[str, Any]:
    if reason is None:
        return {"ok": True, "code": None, "op_ids": [], "message": "ok"}
    return {
        "ok": bool(reason.ok),
        "code": reason.code,
        "op_ids": list(reason.op_ids),
        "message": reason.message,
    }


def _operation_payload(op: OperationStep) -> dict[str, Any]:
    return {
        "op_id": op.op_id,
        "op_type": op.op_type,
        "entity": op.entity,
        "dest": op.dest,
        "deps": list(op.deps),
        "robot": op.robot,
        "params": dict(op.params),
        "gaps": list(op.gaps),
    }


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "code": code,
        "op_ids": [],
        "message": message,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, OperationStep):
        return _operation_payload(value)
    if isinstance(value, ToolReason):
        return _tool_reason_payload(value)
    if isinstance(value, ScheduleResult):
        return _schedule_payload(value)
    if isinstance(value, ExpandResult):
        return _expand_payload(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        # Drop `_`-prefixed keys (e.g. `_crop_images`): they carry in-memory
        # pixels/handles that must never reach JSON traces or text renders.
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if type(value).__module__.split(".", 1)[0] == "PIL":
        # Stable, address-free placeholder for image objects in traces.
        return "<image>"
    return str(value)
