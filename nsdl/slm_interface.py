"""SLM interface layer: prompt serialization, SLM generation, output parsing.

Supports two backends:
- ``api``: external LLM APIs (e.g. Grok / GPT) as drop-in SLM stand-ins.
- ``local``: locally loaded SFT checkpoint for direct inference.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from nsdl.catalog import (
    CONTAINER_TAG,
    OPENABLE_TAG,
    PLACE_TARGET_TAG,
    PORTABLE_ENTITY_TAG,
    WASH_STATION_TAG,
    CUTTING_SURFACE_TAG,
    SUPPORT_TAG,
    TOOL_TAG,
    PUSHABLE_CARRIER_TAG,
    entity_requires_open_before_place,
    get_entity_tags,
    get_robot_capabilities,
    normalize_entity_name,
    normalize_robot_name,
)
from nsdl.profiles.base import RuntimeMappingProxy, RuntimeSetProxy
from nsdl.schema import (
    AffordanceContext,
    AffordanceNode,
    GraphPlan,
    OperationStep,
    SceneFacts,
)
from nsdl.slm_backends import APISLMBackend, get_local_text_backend
from nsdl.slm_backends.api import (
    call_llm_with_retry as _backend_call_llm_with_retry,
    extract_response_text as _backend_extract_response_text,
)
from nsdl.slm_backends.base import SLMBackend
from nsdl.slm_backends.local_text import load_local_slm as _load_local_slm
from nsdl.slm_backends.local_vl import load_local_vlm as _load_local_vlm
from nsdl.task_family import TASK_FAMILY_PUSH_ASSISTED_TRANSPORT


def load_local_slm(weights_path: str, *, device: str = "cuda") -> None:
    """Load the default local SLM backend.

    Kept here for backward compatibility with existing eval scripts.
    """
    _load_local_slm(weights_path, device=device)


def load_local_vlm(weights_path: str, *, device: str = "cuda") -> None:
    """Load the default local vision-language SLM backend."""
    _load_local_vlm(weights_path, device=device)


def _active_profile():
    from nsdl.profiles import get_active_profile

    return get_active_profile()


# ── Operation types ───────────────────────────────────────────────────────
ALLOWED_OP_TYPES = RuntimeSetProxy(
    lambda: _active_profile().op_vocab.allowed_op_types()
)
OP_TYPE_ALIASES = RuntimeMappingProxy(
    lambda: _active_profile().op_vocab.alias_map()
)


# =========================================================================
# 1. System Prompt (fixed template, shared across all samples)
# =========================================================================

def build_slm_system_prompt() -> str:
    """Return the benchmark-profile SLM system prompt.

    v3.4 (plan.md A0g): the model-visible toolset contract is applied here so
    the SFT builder, RL parquet builder and eval paths all render byte-identical
    prompts from one env switch (LNSDL_TOOLSET_CONTRACT; default v33 is a
    byte-identical no-op).
    """
    from nsdl.toolset_contract import apply_toolset_contract

    return apply_toolset_contract(_active_profile().build_system_prompt())


SLM_SYSTEM_PROMPT = build_slm_system_prompt()


# =========================================================================
# 2. User Prompt Serialization
# =========================================================================

def serialize_slm_input(
    scene_facts: SceneFacts,
    affordance_context: AffordanceContext,
) -> str:
    """Serialize SceneFacts + AffordanceContext into the SLM user prompt.

    Current L-NSDL runtime prompt is intentionally lean:
    ``[SCENE]`` contains only the raw instruction and robot inventory, and
    ``[ROBOTS]`` contains robot capabilities/end-effectors. Entity tags and
    YOLO-derived geometry are obtained only through explicit tool calls.
    """
    lines: list[str] = []

    # ── [SCENE] block ─────────────────────────────────────────────────
    lines.append("[SCENE]")
    for contributor in _active_profile().prompt.scene_block_contributors():
        contributor_lines = contributor(scene_facts, affordance_context)
        if contributor_lines:
            lines.extend(contributor_lines)

    lines.append("")

    # ── [ROBOTS] block ────────────────────────────────────────────────
    lines.append("[ROBOTS]")
    for r in affordance_context.robot_capabilities:
        caps_str = ", ".join(r.capabilities)
        lines.append(f"{r.slot_id} ({r.robot_type}): {caps_str} | ee={r.end_effectors}")

    return "\n".join(lines)


# =========================================================================
# 3. SLM Backend Callout
# =========================================================================

def generate_operation_dag(
    scene_facts: SceneFacts,
    affordance_context: AffordanceContext,
    *,
    endpoint_index: int = 0,
    api_base_url: str | None = None,
    api_model: str | None = None,
    use_local: bool = False,
    backend: SLMBackend | None = None,
    is_grpo_ckpt: bool = False,
    temperature: float | None = None,
    image_bytes: bytes | None = None,
) -> list[OperationStep]:
    """Generate an operation DAG.

    When *use_local* is ``True`` (and a local checkpoint has been loaded via
    ``load_local_slm``), inference runs on the local SFT model.  Otherwise
    falls back to external LLM API.  A custom ``backend`` can be injected for
    cross-benchmark plan loops or tests.

    *temperature* default policy (slm-io-redesign §10.2 Q4): SFT ckpt → 0.2,
    GRPO ckpt → 0.5. Pass *is_grpo_ckpt=True* to switch automatically; or pass
    *temperature* explicitly to override.
    """
    system_prompt = build_slm_system_prompt()
    user_prompt = serialize_slm_input(scene_facts, affordance_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    slm_backend = backend or _default_slm_backend(
        endpoint_index=endpoint_index,
        api_base_url=api_base_url,
        api_model=api_model,
        use_local=use_local,
    )

    if temperature is None:
        temperature = 0.5 if is_grpo_ckpt else 0.2
    response_text = slm_backend.generate(
        messages=messages,
        **_generation_kwargs(
            slm_backend,
            temperature=temperature,
            max_tokens=1024,
            max_new_tokens=1024,
            image_bytes=image_bytes,
        ),
    )

    try:
        return parse_slm_output(response_text)
    except Exception as first_err:
        if not _supports_correction_retry(slm_backend):
            raise
        # One retry with corrective prompt
        retry_messages = [
            *messages,
            {"role": "assistant", "content": response_text},
            {
                "role": "user",
                "content": (
                    f"Invalid JSON output. Error: {first_err}. "
                    "Return exactly one JSON list of operation objects. No markdown fences."
                ),
            },
        ]
        retry_text = slm_backend.generate(
            messages=retry_messages,
            **_generation_kwargs(
                slm_backend,
                temperature=0.0,
                max_tokens=1024,
                max_new_tokens=1024,
                image_bytes=image_bytes,
            ),
        )
        return parse_slm_output(retry_text)


def generate_validated_operation_dag(
    scene_facts: SceneFacts,
    affordance_context: AffordanceContext,
    *,
    endpoint_index: int = 0,
    api_base_url: str | None = None,
    api_model: str | None = None,
    use_local: bool = False,
    backend: SLMBackend | None = None,
    max_validator_retries: int = 0,
    is_grpo_ckpt: bool = False,
    temperature: float | None = None,
    image_bytes: bytes | None = None,
):
    """Generate an op DAG and run the detection-only validator.

    Default ``max_validator_retries=0`` (2026-05-22): the corrective retry
    mechanism is currently abandoned (see docs/debug.md §6). The remaining
    error-severity detectors fire on so few real SLM samples that retry
    produces a net regression once partial credit is accounted for. The
    parameter is preserved for ablation. Validator NEVER mutates ops; it
    only diagnoses. Returns ``(ops, ValidationReport, attempts)`` where
    ``attempts`` is 1 (no retry) or 2 (one corrective retry).
    """
    system_prompt = build_slm_system_prompt()
    user_prompt = serialize_slm_input(scene_facts, affordance_context)

    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    slm_backend = backend or _default_slm_backend(
        endpoint_index=endpoint_index,
        api_base_url=api_base_url,
        api_model=api_model,
        use_local=use_local,
    )

    def _generate(messages: list[dict[str, Any]], temperature: float):
        raw_text = slm_backend.generate(
            messages=messages,
            **_generation_kwargs(
                slm_backend,
                temperature=temperature,
                max_tokens=4096,
                max_new_tokens=4096,
                image_bytes=image_bytes,
            ),
        )
        response = parse_slm_response(raw_text)
        expected_non_empty_plan = _response_expects_non_empty_plan(response)
        goal_active = _extract_goal_active(response)
        return response.operations, raw_text, expected_non_empty_plan, goal_active

    if temperature is None:
        temperature = 0.5 if is_grpo_ckpt else 0.2
    ops, raw_text, expected_non_empty_plan, goal_active = _generate(base_messages, temperature=temperature)
    report = validate_operations(
        ops,
        affordance_context,
        scene_facts,
        expected_non_empty_plan=expected_non_empty_plan,
        goal_active=goal_active,
    )
    attempts = 1
    # Keep a single corrective second pass: round 1 is normal inference, round
    # 2 is the only retry. The historical parameter name is retained for
    # compatibility with existing call sites.
    if report.ok or max_validator_retries <= 0 or not _supports_correction_retry(slm_backend):
        return ops, report, attempts

    retry_messages = [
        *base_messages,
        {"role": "assistant", "content": raw_text},
        {"role": "user", "content": format_violation_prompt(report)},
    ]
    try:
        ops, _, expected_non_empty_plan, goal_active = _generate(retry_messages, temperature=0.0)
    except Exception:
        # Retry round itself failed (e.g. JSON parse on the second try) → keep
        # the original ops so the caller can fail-closed with the diagnosis.
        return ops, report, attempts + 1
    report = validate_operations(
        ops,
        affordance_context,
        scene_facts,
        expected_non_empty_plan=expected_non_empty_plan,
        goal_active=goal_active,
    )
    return ops, report, attempts + 1


def _response_expects_non_empty_plan(response: SLMResponse) -> bool | None:
    """Return True only when the parsed `[GOAL]` says active work remains."""
    goal = response.goal
    if not isinstance(goal, dict):
        return None
    active = goal.get("active")
    if isinstance(active, list):
        return bool(active)
    return None


def _default_slm_backend(
    *,
    endpoint_index: int,
    api_base_url: str | None,
    api_model: str | None,
    use_local: bool,
) -> SLMBackend:
    if use_local:
        return get_local_text_backend()
    return APISLMBackend(
        endpoint_index=endpoint_index,
        api_base_url=api_base_url,
        api_model=api_model,
    )


def _supports_correction_retry(backend: SLMBackend) -> bool:
    return bool(getattr(backend, "supports_correction_retry", True))


def _generation_kwargs(
    backend: SLMBackend,
    *,
    temperature: float,
    max_tokens: int,
    max_new_tokens: int,
    image_bytes: bytes | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_new_tokens": max_new_tokens,
    }
    if image_bytes is not None and getattr(backend, "accepts_image_input", False):
        kwargs["image_bytes"] = image_bytes
    return kwargs


def _call_llm_with_retry(
    *,
    messages: list[dict[str, Any]],
    endpoint_index: int,
    api_base_url: str | None,
    api_model: str | None,
    temperature: float,
    max_tokens: int,
) -> Any:
    """Compatibility wrapper; implementation lives in ``slm_backends.api``."""
    return _backend_call_llm_with_retry(
        messages=messages,
        endpoint_index=endpoint_index,
        api_base_url=api_base_url,
        api_model=api_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_response_text(response: Any) -> str:
    """Compatibility wrapper; implementation lives in ``slm_backends.api``."""
    return _backend_extract_response_text(response)


# =========================================================================
# 4. Output Parsing
# =========================================================================

def _recover_truncated_json_array(text: str) -> list[Any]:
    """Best-effort recovery for a JSON array truncated by a token budget.

    Walks the text character-by-character tracking bracket depth; returns
    a list of all JSON objects that were *fully* present before the cut.
    Raises ``ValueError`` if no complete element can be recovered.
    """
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array start '[' found in truncated SLM output.")
    depth = 0
    last_object_end = -1
    i = start + 1
    in_string = False
    escape_next = False
    while i < len(text):
        ch = text[i]
        if escape_next:
            escape_next = False
        elif ch == "\\" and in_string:
            escape_next = True
        elif ch == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_object_end = i
        i += 1
    if last_object_end == -1:
        raise ValueError("No complete JSON object found in truncated SLM output.")
    recovered = text[start: last_object_end + 1] + "]"
    return json.loads(recovered)  # type: ignore[return-value]


def parse_slm_output(raw_text: str) -> list[OperationStep]:
    """Parse raw SLM output into a list of OperationStep.

    Backward-compatible wrapper. The current SLM schema is a 3-block CoT-style
    output (`[GOAL]` / `[ASSIGN]` / `[PLAN]`); use :func:`parse_slm_response`
    when you need the GOAL / ASSIGN intermediate blocks (e.g. process-level
    reward computation). This function returns only the final operation list
    extracted from the `[PLAN]` block, and falls back to legacy bare-list
    parsing when no `[PLAN]` header is present.
    """
    return parse_slm_response(raw_text).operations


# ── Agentic output schema ────────────────────────────────────────────────

_AGENTIC_BLOCK_NAMES = "GOAL|ASSIGN|PLAN_SKETCH|TOOL_CALL|PLAN"
_AGENTIC_BLOCK_PATTERN = re.compile(
    rf"\[({_AGENTIC_BLOCK_NAMES})\]\s*(.*?)(?=\s*(?:\[(?:{_AGENTIC_BLOCK_NAMES})\]|<think\b)|\Z)",
    re.DOTALL,
)
_THINK_PATTERN = re.compile(r"<think\b[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class SLMResponse:
    """Parsed view of an agentic SLM response.

    Layout::

        <think> ... </think>
        [GOAL]   <json object: targets / active / skip>
        [ASSIGN] <json object: per-robot role + items>
        [PLAN_SKETCH] <json list: intent operation sketch>
        [TOOL_CALL] <json object: {"name": ..., "args": {...}}>
        [PLAN]   <json list: final operation DAG>

    Legacy single-list outputs (no headers) are still parsed; in that case
    ``goal`` / ``assign`` are ``None`` and ``operations`` is filled from the
    bare list. Old three-block `[GOAL][ASSIGN][PLAN]` outputs remain valid.
    """

    operations: list[OperationStep]
    goal: dict[str, Any] | None = None
    assign: dict[str, Any] | None = None
    think: list[str] = field(default_factory=list)
    plan_sketch: list[OperationStep] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_blocks: dict[str, str] = field(default_factory=dict)
    raw_block_lists: dict[str, list[str]] = field(default_factory=dict)
    parse_diagnostics: list[dict[str, Any]] = field(default_factory=list)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


def _split_agentic_blocks(text: str) -> dict[str, list[str]]:
    """Extract agentic block bodies in encounter order."""
    blocks: dict[str, list[str]] = {}
    for match in _AGENTIC_BLOCK_PATTERN.finditer(text):
        name = match.group(1).upper()
        body = match.group(2).strip()
        if not body:
            continue
        blocks.setdefault(name, []).append(body)
    return blocks


def _split_cot_blocks(text: str) -> dict[str, str]:
    """Compatibility helper returning the last body per block name."""
    blocks: dict[str, str] = {}
    for name, bodies in _split_agentic_blocks(text).items():
        if bodies:
            blocks[name] = bodies[-1]
    return blocks


def _try_load_json(text: str) -> Any:
    return json.loads(_strip_markdown_fences(text))


def _json_excerpt(text: str, *, limit: int = 300) -> str:
    stripped = re.sub(r"\s+", " ", text).strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def _load_json_with_recovery(text: str) -> tuple[Any, dict[str, Any] | None]:
    """Load JSON and tolerate common tool-block recovery cases.

    Besides harmless trailing bracket noise, 7B P5.5 sometimes emitted two
    JSON objects in one ``[TOOL_CALL]`` body. Recover those as a list so the
    runtime can execute the first call and record the suppressed stacked call.
    """
    stripped = _strip_markdown_fences(text)
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError as original_exc:
        decoder = json.JSONDecoder()
        try:
            value, end_index = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            raise original_exc

        trailing = stripped[end_index:].strip()
        if trailing and not all(char in "]})" for char in trailing):
            # Trailing content after the first complete JSON object. It is either
            # (a) one or more additional stacked JSON objects — recover them as a
            # list so the runtime executes the first and records the suppressed
            # stacked call; or (b) pure decode-time degeneration garbage (stray
            # brackets, repeated fragments, hallucinated markup such as
            # ``</pre>``). In case (b) we keep the first valid object and strip the
            # garbage rather than failing the whole ``[TOOL_CALL]`` (which
            # otherwise cascades to a schedule-not-executed error). This mirrors
            # lenient JSON extraction in production function-calling stacks and
            # only fires when the first object is fully valid (``raw_decode`` above
            # already guarantees that).
            values = [value]
            index = end_index
            while index < len(stripped):
                while index < len(stripped) and stripped[index].isspace():
                    index += 1
                if index >= len(stripped):
                    break
                try:
                    next_value, next_index = decoder.raw_decode(stripped, index)
                except json.JSONDecodeError:
                    # Remaining trailing is not another JSON object: it is stray
                    # brackets or degeneration garbage. Stop and keep what parsed.
                    break
                values.append(next_value)
                index = next_index
            if len(values) > 1:
                return values, {
                    "severity": "warning",
                    "kind": "json_concatenated_values_recovered",
                    "message": str(original_exc),
                    "recovered_value_count": len(values),
                }
            return value, {
                "severity": "warning",
                "kind": "json_trailing_garbage_stripped",
                "message": str(original_exc),
                "trailing": trailing[:120],
            }
        return value, {
            "severity": "warning",
            "kind": "json_trailing_tokens_recovered",
            "message": str(original_exc),
            "trailing": trailing,
        }


def _json_parse_diagnostic(
    *,
    block: str,
    kind: str,
    exc: Exception,
    text: str,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "block": block,
        "kind": kind,
        "message": str(exc),
        "raw_excerpt": _json_excerpt(text),
    }


def parse_slm_response(raw_text: str) -> SLMResponse:
    """Parse raw SLM output into a :class:`SLMResponse`.

    Schema::

        <think>...</think>

        [GOAL]
        {"targets":[{"item":..., "destination":..., "status":...}, ...],
         "active":[...], "skip":[...]}

        [ASSIGN]
        {"R1":{"role":..., "items":[...]}, "R2":{"role":..., "items":[...]}}

        [PLAN_SKETCH]
        [<intent operation dict>, ...]

        [TOOL_CALL]
        {"name":"schedule","args":{"ops":[...], "assign":{...}}}

        [PLAN]
        [<operation dict>, ...]

    Backward compatibility: a bare JSON list (legacy) is treated as the
    `[PLAN]` block; ``goal`` / ``assign`` are then ``None``.
    """
    text = _strip_markdown_fences(raw_text)
    block_lists = _split_agentic_blocks(text)
    blocks = {name: bodies[-1] for name, bodies in block_lists.items() if bodies}

    goal_obj: dict[str, Any] | None = None
    assign_obj: dict[str, Any] | None = None
    think = _extract_think_blocks(text)
    plan_sketch: list[OperationStep] = []
    tool_calls: list[dict[str, Any]] = []
    operations: list[OperationStep] = []
    parse_diagnostics: list[dict[str, Any]] = []

    if "GOAL" in blocks:
        try:
            value = _try_load_json(blocks["GOAL"])
        except json.JSONDecodeError as exc:
            value = None
            parse_diagnostics.append(_json_parse_diagnostic(
                block="GOAL",
                kind="json_decode_error",
                exc=exc,
                text=blocks["GOAL"],
            ))
        if isinstance(value, dict):
            goal_obj = value
    if "ASSIGN" in blocks:
        try:
            value = _try_load_json(blocks["ASSIGN"])
        except json.JSONDecodeError as exc:
            value = None
            parse_diagnostics.append(_json_parse_diagnostic(
                block="ASSIGN",
                kind="json_decode_error",
                exc=exc,
                text=blocks["ASSIGN"],
            ))
        if isinstance(value, dict):
            assign_obj = value

    if "PLAN_SKETCH" in blocks:
        try:
            plan_sketch = _parse_operation_list(blocks["PLAN_SKETCH"])
        except ValueError as exc:
            plan_sketch = []
            parse_diagnostics.append(_json_parse_diagnostic(
                block="PLAN_SKETCH",
                kind="operation_parse_error",
                exc=exc,
                text=blocks["PLAN_SKETCH"],
            ))

    for tool_body in block_lists.get("TOOL_CALL", []):
        calls, diagnostics = _parse_tool_call_block_with_diagnostics(tool_body)
        tool_calls.extend(calls)
        parse_diagnostics.extend(diagnostics)

    if "PLAN" in blocks:
        try:
            operations = _parse_operation_list(blocks["PLAN"])
        except ValueError as exc:
            operations = []
            parse_diagnostics.append(_json_parse_diagnostic(
                block="PLAN",
                kind="operation_parse_error",
                exc=exc,
                text=blocks["PLAN"],
            ))
    elif not block_lists:
        # Legacy single-list output: treat the whole payload as PLAN.
        try:
            operations = _parse_operation_list(text)
        except ValueError as exc:
            operations = []
            parse_diagnostics.append(_json_parse_diagnostic(
                block="PLAN",
                kind="legacy_operation_parse_error",
                exc=exc,
                text=text,
            ))

    return SLMResponse(
        operations=operations,
        goal=goal_obj,
        assign=assign_obj,
        think=think,
        plan_sketch=plan_sketch,
        tool_calls=tool_calls,
        raw_blocks=blocks,
        raw_block_lists=block_lists,
        parse_diagnostics=parse_diagnostics,
    )


def _extract_think_blocks(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in _THINK_PATTERN.finditer(text)
        if match.group(1).strip()
    ]


def _parse_tool_call_block(block_text: str) -> list[dict[str, Any]]:
    calls, _diagnostics = _parse_tool_call_block_with_diagnostics(block_text)
    return calls


def _parse_tool_call_block_with_diagnostics(block_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    try:
        payload, recovery = _load_json_with_recovery(block_text)
    except json.JSONDecodeError as exc:
        return [], [
            _json_parse_diagnostic(
                block="TOOL_CALL",
                kind="json_decode_error",
                exc=exc,
                text=block_text,
            )
        ]
    if recovery:
        recovery.update({
            "block": "TOOL_CALL",
            "raw_excerpt": _json_excerpt(block_text),
        })
        diagnostics.append(recovery)

    raw_calls: list[Any]
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        raw_calls = payload["tool_calls"]
    elif isinstance(payload, list):
        raw_calls = payload
    else:
        raw_calls = [payload]

    calls: list[dict[str, Any]] = []
    for call_index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            diagnostics.append({
                "severity": "error",
                "block": "TOOL_CALL",
                "kind": "non_object_call",
                "call_index": call_index,
                "raw_excerpt": _json_excerpt(str(raw)),
            })
            continue
        name = str(raw.get("name") or raw.get("tool") or raw.get("tool_name") or "").strip()
        if not name:
            diagnostics.append({
                "severity": "error",
                "block": "TOOL_CALL",
                "kind": "missing_tool_name",
                "call_index": call_index,
                "raw_excerpt": _json_excerpt(json.dumps(raw, ensure_ascii=False)),
            })
            continue
        args = raw.get("args")
        if args is None:
            args = raw.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append({"name": name, "args": args})
    return calls, diagnostics


def _parse_operation_list(plan_text: str) -> list[OperationStep]:
    """Parse the [PLAN] body into a list of OperationStep (existing logic)."""
    stripped = _strip_markdown_fences(plan_text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        # Token budget may have truncated the output mid-array; try to recover
        # all fully-written elements before the cut.
        payload = _recover_truncated_json_array(stripped)

    return parse_operation_steps(payload)


def parse_operation_steps(payload: Any) -> list[OperationStep]:
    """Coerce a JSON-like operation payload into normalized OperationStep objects.

    Accepts a bare list of operation dicts, ``{"operations": [...]}``, or a JSON
    string in either shape. This is shared by response parsing and tool runtime
    argument routing so training/inference see the same normalization.
    """
    if isinstance(payload, str):
        return _parse_operation_list(payload)

    # Accept either a bare list or {"operations": [...]}
    if isinstance(payload, dict) and isinstance(payload.get("operations"), list):
        raw_ops = payload["operations"]
    elif isinstance(payload, list):
        raw_ops = payload
    else:
        raise ValueError("Expected a JSON list or object with 'operations' key.")

    if not raw_ops:
        return []

    ops: list[OperationStep] = []
    used_ids: set[str] = set()
    for idx, raw in enumerate(raw_ops, start=1):
        if not isinstance(raw, dict):
            continue
        op_type = _normalize_op_type(raw.get("op_type"))
        if op_type not in ALLOWED_OP_TYPES:
            continue

        op_id = str(raw.get("op_id", f"op{idx}")).strip() or f"op{idx}"
        suffix = 2
        base_id = op_id
        while op_id in used_ids:
            op_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(op_id)

        # Normalize deps from various field names. `depends_on` is the spelling
        # an untrained model reaches for unprompted (measured on the P6.1 point-1
        # arm: the base 7B emitted a correct acquire->deposit chain under
        # `depends_on`, it was dropped here, and the resulting empty deps chain
        # was scored as a structural error the model had not made). SFT'd
        # front-ends emit `deps`, so this alias is a no-op for them.
        deps_raw = (
            raw.get("deps")
            or raw.get("depends_on")
            or raw.get("predecessor_op_ids")
            or raw.get("predecessors")
            or []
        )
        if not isinstance(deps_raw, list):
            deps_raw = [deps_raw]
        deps = [str(d).strip() for d in deps_raw if str(d).strip()]

        entity = _normalize_nullable(raw.get("entity"))
        dest = _normalize_nullable(
            raw.get("dest") or raw.get("fixture_or_dest")
        )
        robot_raw = raw.get("robot") or raw.get("executor_hint")
        robot = str(robot_raw).strip() if robot_raw and str(robot_raw).strip().lower() not in ("none", "null", "n/a", "") else None
        params = _normalize_params(raw)
        gaps = _normalize_gaps(raw, params)
        if dest is None and op_type == "push_carrier":
            target_robot = params.get("target_robot") or params.get("target_slot")
            if isinstance(target_robot, str) and target_robot.strip():
                dest = target_robot.strip()

        ops.append(OperationStep(
            op_id=op_id,
            op_type=op_type,
            entity=entity,
            dest=dest,
            deps=deps,
            robot=robot,
            params=params,
            gaps=gaps,
        ))

    if not ops:
        raise ValueError("No valid operations found in SLM output.")
    return ops


def _normalize_op_type(value: Any) -> str:
    op_type = str(value or "").strip().lower()
    op_type = op_type.replace("-", "_").replace(" ", "_")
    return OP_TYPE_ALIASES.get(op_type, op_type)


def _normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field_name in ("params", "arguments"):
        value = raw.get(field_name)
        if isinstance(value, dict):
            params.update(value)
    for field_name in (
        "carrier",
        "tool",
        "target_robot",
        "target_slot",
        "source",
        "source_container",
        "mode",
    ):
        if field_name in raw and field_name not in params:
            params[field_name] = raw[field_name]

    normalized: dict[str, Any] = {}
    for key, value in params.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in ("none", "null", "n/a"):
                continue
            if normalized_key in {"carrier", "tool", "source", "source_container"}:
                normalized[normalized_key] = normalize_entity_name(text) or text
            elif normalized_key in {"target_robot", "target_slot"}:
                normalized[normalized_key] = _normalize_robot_slot(text)
            else:
                normalized[normalized_key] = text
            continue
        normalized[normalized_key] = value
    return normalized


def _normalize_gaps(raw: dict[str, Any], params: dict[str, Any] | None = None) -> list[str]:
    values: list[Any] = []
    for key in ("gaps", "gap", "<gap>"):
        if key in raw:
            value = raw[key]
            values.extend(value if isinstance(value, list) else [value])
    if params:
        for key in ("gaps", "gap", "<gap>"):
            if key in params:
                value = params[key]
                values.extend(value if isinstance(value, list) else [value])

    gaps: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if not text or text in {"none", "null", "n/a"}:
            continue
        text = text.removeprefix("<gap:").removesuffix(">")
        for part in re.split(r"[,|/ ]+", text):
            part = part.strip()
            if part and part not in gaps:
                gaps.append(part)
    return gaps


def _normalize_nullable(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "n/a"):
        return None
    return normalize_entity_name(text) or text


def _normalize_robot_slot(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0].lower() == "r" and text[1:].isdigit():
        return f"R{text[1:]}"
    return text


# =========================================================================
# 5. Validator (V1 + V2 + V3) — DETECTION ONLY, no DAG mutation
# =========================================================================
#
# Per docs/agent.md "Validator Boundary Rule": deterministic code MUST NOT
# patch SLM outputs. The validator only diagnoses violations; corrections
# happen via reverse-prompt retry (see ``generate_operation_dag``) so the
# training/inference signal always reflects the SLM's own planning ability.


from dataclasses import dataclass, field as _dc_field


@dataclass
class ValidationIssue:
    """A single detected DAG violation, suitable for SLM feedback."""

    code: str
    message: str
    op_ids: list[str] = _dc_field(default_factory=list)
    # F2 (2026-05-22): "warning"-level issues are emitted for diagnostics /
    # GRPO reward shaping but do NOT mark the report as failed and therefore
    # do NOT trigger a corrective retry. Default is "error" to preserve
    # legacy behavior of all existing detectors.
    severity: str = "error"


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = _dc_field(default_factory=list)

    @property
    def messages(self) -> list[str]:
        return [issue.message for issue in self.issues]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


def validate_operations(
    ops: list[OperationStep],
    affordance_context: AffordanceContext,
    scene_facts: SceneFacts,
    *,
    expected_non_empty_plan: bool | None = None,
    goal_active: list[str] | None = None,
) -> ValidationReport:
    """Detect V0/V1/V2/V3/V4 violations without mutating ops.

    V0 — Empty plan vs. non-empty active goal (only when caller asserts
         *expected_non_empty_plan=True*; otherwise an empty op list is
         treated as a legitimate "all-already-at-destination" case).
    V1 — Precondition Completeness / op-level noise / task-blind
         entity-role checks
    V2 — Capability Feasibility (robot ↔ required caps; handoff requires
         params.target_robot)
    V3 — Dependency Soundness (invalid refs, cycles)
    V4 — Plan-target coverage: every item listed in the SLM's own
         ``[GOAL].active`` must appear as the entity of at least one
         item-handling op in ``[PLAN]``. Task-blind: uses only the SLM's
         self-declared active list, never ground truth.
    """
    issues: list[ValidationIssue] = []
    if expected_non_empty_plan and not ops:
        issues.append(ValidationIssue(
            code="empty_plan",
            message=(
                "[GOAL].active is non-empty but [PLAN] is []; produce at least "
                "one acquire/deposit (or full sequence) for each active target."
            ),
            op_ids=[],
        ))
    issues.extend(_detect_missing_preconditions(ops))
    issues.extend(_detect_fixture_item_misuse(ops))
    issues.extend(_detect_invalid_destination_usage(ops))
    issues.extend(_detect_end_effector_state_violations(ops, affordance_context))
    issues.extend(_detect_invalid_open_ops(ops))
    issues.extend(_detect_duplicate_open_ops(ops))
    issues.extend(_detect_redundant_wash_activation(ops))
    issues.extend(_detect_duplicate_activations(ops))
    issues.extend(_detect_capability_violations(ops, affordance_context))
    issues.extend(_detect_dependency_violations(ops))
    # F2 (2026-05-22): V_PLAN_TARGET_COVERAGE is downgraded to a warning.
    # Empirically on the v1 1.5B ckpt it triggers misfires on
    # `bring_bowl_to_table_plate_already_there` (−16.2 viki_l2 score) and
    # produces zero net rescue on `bring_plate_and_bowl_to_table` (retry
    # only swaps which single portable gets handled — partial-credit
    # barrier). Keep emitting the issue for logging / GRPO reward shaping,
    # but never let it gate `report.ok` (i.e. never trigger retry).
    coverage_issues = _detect_plan_target_coverage(ops, goal_active)
    for issue in coverage_issues:
        issue.severity = "warning"
    issues.extend(coverage_issues)
    return ValidationReport(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def format_violation_prompt(report: ValidationReport) -> str:
    """Build the [VIOLATION] block sent back to the SLM for second-pass repair."""
    lines = [
        "[VIOLATION]",
        "Your previous response was rejected. Regenerate the entire corrected response with [GOAL], [ASSIGN], and [PLAN].",
        "Fix the following issues; do not introduce new ones:",
    ]
    # Only echo error-severity issues into the retry prompt; warnings are
    # diagnostic-only (see F2 note above).
    for issue in report.errors:
        prefix = f"- ({issue.code})"
        if issue.op_ids:
            prefix += f" {','.join(issue.op_ids)}:"
        lines.append(f"{prefix} {issue.message}")
    return "\n".join(lines)


# Backward-compat shim. Detection-only — never mutates input. Returns the
# original ops on success, None on failure (so legacy training/inference
# call sites that expect ``None`` to mean "fail-closed" keep working without
# silently masking SLM mistakes).
def validate_and_repair_operations(
    ops: list[OperationStep],
    affordance_context: AffordanceContext,
    scene_facts: SceneFacts,
) -> tuple[list[OperationStep] | None, list[str]]:
    report = validate_operations(ops, affordance_context, scene_facts)
    if not report.ok:
        return None, report.messages
    return ops, []


# Ops whose ``entity`` field represents the item being acted upon. Used by
# V_PLAN_TARGET_COVERAGE to decide whether a goal-active item is covered.
_ITEM_HANDLING_OPS: frozenset[str] = frozenset({
    "acquire", "carry_to", "deposit", "handoff",
    "load_carrier", "unload_carrier", "push_carrier",
    "process_at", "apply_tool", "inspect",
})

_CUSTODY_SOURCE_OPS: frozenset[str] = frozenset({
    "acquire",
    "local_acquire",
    "handoff",
    "unload_carrier",
    "process_at",
})
_DEPOSIT_CUSTODY_SOURCE_OPS: frozenset[str] = _CUSTODY_SOURCE_OPS | frozenset({
    "carry_to",
})
_FIXTURE_TAGS: frozenset[str] = frozenset({
    PLACE_TARGET_TAG,
    SUPPORT_TAG,
    CONTAINER_TAG,
    WASH_STATION_TAG,
    CUTTING_SURFACE_TAG,
    PUSHABLE_CARRIER_TAG,
    "area",
})
_ITEM_CAPABLE_TAGS: frozenset[str] = frozenset({
    PORTABLE_ENTITY_TAG,
    TOOL_TAG,
    "food",
    "graspable",
})
_FIXTURE_ENTITY_OPS: frozenset[str] = frozenset({
    "acquire",
    "local_acquire",
    "carry_to",
    "deposit",
    "handoff",
    "load_carrier",
    "unload_carrier",
    "process_at",
    "apply_tool",
})
_DESTINATION_REQUIRED_OPS: frozenset[str] = frozenset({
    "carry_to",
    "deposit",
    "process_at",
    "unload_carrier",
})
_HOLD_ADD_OPS: frozenset[str] = frozenset({
    "acquire",
    "local_acquire",
    "handoff",
})
_HOLD_RELEASE_OPS: frozenset[str] = frozenset({
    "deposit",
    "load_carrier",
    "process_at",
})
_HELD_ITEM_REQUIRED_OPS: frozenset[str] = frozenset({
    "carry_to",
    "deposit",
    "load_carrier",
    "process_at",
})
_FREE_EE_REQUIRED_OPS: frozenset[str] = frozenset({
    "open",
    "close",
})


def _extract_goal_active(response: SLMResponse) -> list[str] | None:
    """Pull a normalized list of active item names from the parsed GOAL block.

    Returns ``None`` when GOAL is missing or malformed (so the detector
    becomes a no-op rather than raising).
    """
    goal = response.goal
    if not isinstance(goal, dict):
        return None
    active = goal.get("active")
    if not isinstance(active, list):
        return None
    items: list[str] = []
    for entry in active:
        if isinstance(entry, str):
            name = normalize_entity_name(entry)
        elif isinstance(entry, dict):
            name = normalize_entity_name(entry.get("item") or entry.get("name") or "")
        else:
            name = ""
        if name:
            items.append(name)
    return items


def _detect_plan_target_coverage(
    ops: list[OperationStep],
    goal_active: list[str] | None,
) -> list[ValidationIssue]:
    """V_PLAN_TARGET_COVERAGE: every item in [GOAL].active must be touched
    by at least one item-handling op in [PLAN].

    Task-blind: only reads the SLM's self-declared GOAL.active and the ops it
    emitted; no ground truth, no rule-based target extraction from the
    instruction string.

    When ``goal_active`` is ``None`` (GOAL block absent/malformed) the check
    is skipped — V0 / format reward already penalises that case.
    """
    if not goal_active:
        return []
    covered: set[str] = set()
    for op in ops:
        if op.op_type not in _ITEM_HANDLING_OPS:
            continue
        name = normalize_entity_name(op.entity)
        if name:
            covered.add(name)
    issues: list[ValidationIssue] = []
    # Preserve declaration order; dedupe while keeping first occurrence.
    seen: set[str] = set()
    for item in goal_active:
        if item in seen or item in covered:
            seen.add(item)
            continue
        seen.add(item)
        issues.append(ValidationIssue(
            code="missing_plan_target_coverage",
            message=(
                f"[GOAL].active lists '{item}' but no acquire/carry_to/deposit "
                f"op in [PLAN] has entity='{item}'. Add the full handling "
                f"sequence for '{item}', or remove it from [GOAL].active and "
                f"move it to [GOAL].skip if it should not be handled."
            ),
            op_ids=[],
        ))
    return issues


# ── Detection helpers (pure: read ops, return list[ValidationIssue]) ──

def _detect_missing_preconditions(ops: list[OperationStep]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    op_by_id = {op.op_id: op for op in ops}

    # 1) deposit at openable container needs a preceding open op linked via the
    #    dependency chain.
    for op in ops:
        if op.op_type != "deposit" or not op.dest:
            continue
        if OPENABLE_TAG not in set(get_entity_tags(op.dest)):
            continue
        normalized_dest = normalize_entity_name(op.dest)
        dep_ids = _dependency_closure(op, op_by_id)
        linked_open = any(
            (dep_op := op_by_id.get(dep)) is not None
            and dep_op.op_type == "open"
            and normalize_entity_name(dep_op.dest or dep_op.entity) == normalized_dest
            for dep in dep_ids
        )
        if linked_open:
            continue
        issues.append(ValidationIssue(
            code="missing_open_before_deposit",
            message=(
                f"deposit({op.entity}, {op.dest}) targets an openable container "
                f"but has no preceding open({op.dest}) linked via deps."
            ),
            op_ids=[op.op_id],
        ))

    # 2) carry_to / deposit consume custody. The same robot must have custody of
    #    the same entity somewhere in the dependency chain.
    for op in ops:
        if op.op_type not in {"carry_to", "deposit"} or not op.entity:
            continue
        if not op.robot:
            continue
        source_ops = (
            _DEPOSIT_CUSTODY_SOURCE_OPS
            if op.op_type == "deposit"
            else _CUSTODY_SOURCE_OPS
        )
        normalized_entity = normalize_entity_name(op.entity)
        candidates = _same_entity_dep_chain_ops(
            op,
            op_by_id,
            source_ops,
        )
        same_robot = [
            candidate
            for candidate in candidates
            if _same_robot_slot(candidate.robot, op.robot)
        ]
        if same_robot:
            continue
        op_sig = f"{op.op_type}({op.entity}, {op.dest})"
        if candidates:
            holder_slots = sorted({
                str(candidate.robot)
                for candidate in candidates
                if candidate.robot
            })
            issues.append(ValidationIssue(
                code="cross_robot_custody",
                message=(
                    f"{op_sig} is assigned to {op.robot}, but custody of "
                    f"{normalized_entity} in its deps chain is held by "
                    f"{holder_slots or 'another/unspecified robot'}; use the "
                    "same robot or add an explicit handoff."
                ),
                op_ids=[op.op_id],
            ))
            continue
        issues.append(ValidationIssue(
            code=f"missing_custody_before_{op.op_type}",
            message=(
                f"{op_sig} by {op.robot} has no same-entity custody source "
                "in its deps chain; add acquire/local_acquire/handoff/carry_to "
                "or process_at for the same item before this op."
            ),
            op_ids=[op.op_id],
        ))
    return issues


def _dependency_closure(
    op: OperationStep,
    op_by_id: dict[str, OperationStep],
) -> set[str]:
    seen: set[str] = set()
    stack = list(op.deps)
    while stack:
        dep_id = stack.pop()
        if dep_id in seen:
            continue
        seen.add(dep_id)
        dep_op = op_by_id.get(dep_id)
        if dep_op is not None:
            stack.extend(dep_op.deps)
    return seen


def _same_entity_dep_chain_ops(
    op: OperationStep,
    op_by_id: dict[str, OperationStep],
    op_types: frozenset[str],
) -> list[OperationStep]:
    normalized_entity = normalize_entity_name(op.entity)
    if not normalized_entity:
        return []
    candidates: list[OperationStep] = []
    for dep_id in _dependency_closure(op, op_by_id):
        dep_op = op_by_id.get(dep_id)
        if dep_op is None or dep_op.op_type not in op_types:
            continue
        if normalize_entity_name(dep_op.entity) == normalized_entity:
            candidates.append(dep_op)
    return candidates


def _same_robot_slot(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _normalize_robot_slot(str(left)) == _normalize_robot_slot(str(right))


def _detect_fixture_item_misuse(ops: list[OperationStep]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type not in _FIXTURE_ENTITY_OPS or not op.entity:
            continue
        if not _is_fixture_only_entity(op.entity):
            continue
        issues.append(ValidationIssue(
            code="fixture_used_as_item",
            message=(
                f"{op.op_type}({op.entity}) treats a fixture/destination as the "
                "handled item; use item-handling ops only for portable/tool "
                "entities."
            ),
            op_ids=[op.op_id],
        ))
    return issues


def _detect_invalid_destination_usage(ops: list[OperationStep]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type not in _DESTINATION_REQUIRED_OPS or not op.dest:
            continue
        if _looks_like_robot_slot(op.dest):
            continue
        tags = set(get_entity_tags(op.dest))
        if not tags or tags.intersection(_FIXTURE_TAGS):
            continue
        issues.append(ValidationIssue(
            code="invalid_destination_entity",
            message=(
                f"{op.op_type}({op.entity}, {op.dest}) uses '{op.dest}' as a "
                "destination, but its catalog tags do not make it a support, "
                "container, area, station, or carrier."
            ),
            op_ids=[op.op_id],
        ))
    return issues


def _is_fixture_only_entity(entity: str | None) -> bool:
    if not entity:
        return False
    tags = set(get_entity_tags(entity))
    if not tags:
        return False
    if tags.intersection(_ITEM_CAPABLE_TAGS):
        return False
    return bool(tags.intersection(_FIXTURE_TAGS))


def _looks_like_robot_slot(value: str | None) -> bool:
    text = str(value or "").strip()
    return len(text) >= 2 and text[0].lower() == "r" and text[1:].isdigit()


def _detect_end_effector_state_violations(
    ops: list[OperationStep],
    ctx: AffordanceContext,
) -> list[ValidationIssue]:
    ee_by_slot = {
        r.slot_id: max(0, int(r.end_effectors))
        for r in ctx.robot_capabilities
    }
    op_by_id = {op.op_id: op for op in ops}
    issues: list[ValidationIssue] = []

    for op in ops:
        if not op.robot or op.robot not in ee_by_slot:
            continue
        ee = ee_by_slot[op.robot]
        held = _held_items_before_op(op, op_by_id)
        held_list = sorted(held)

        entity = normalize_entity_name(op.entity)
        if op.op_type in _HELD_ITEM_REQUIRED_OPS and entity and entity not in held:
            issues.append(ValidationIssue(
                code=f"not_holding_before_{op.op_type}",
                message=(
                    f"{op.op_type}({op.entity}, {op.dest}) is assigned to "
                    f"{op.robot}, but the deps-chain state before this op is "
                    f"holding={held_list}; add/rewire custody so {op.robot} "
                    f"is holding '{entity}' at this point."
                ),
                op_ids=[op.op_id],
            ))

        if op.op_type in {"acquire", "local_acquire"} and entity:
            if entity in held:
                issues.append(ValidationIssue(
                    code="duplicate_acquire_while_holding",
                    message=(
                        f"{op.robot} already holds '{entity}' before "
                        f"{op.op_type}({op.entity}); remove the duplicate "
                        "acquire or insert a release/deposit first."
                    ),
                    op_ids=[op.op_id],
                ))
                continue
            if len(held) >= ee:
                issues.append(ValidationIssue(
                    code="end_effector_capacity_exceeded",
                    message=(
                        f"{op.robot} has ee={ee} and is already holding "
                        f"{held_list} before {op.op_type}({op.entity}); "
                        "deposit/load/process an item first or assign a "
                        "robot with more free end-effectors."
                    ),
                    op_ids=[op.op_id],
                ))

        if op.op_type in _FREE_EE_REQUIRED_OPS and len(held) >= ee:
            target = op.dest or op.entity
            issues.append(ValidationIssue(
                code=f"no_free_end_effector_for_{op.op_type}",
                message=(
                    f"{op.robot} has ee={ee} and is holding {held_list} before "
                    f"{op.op_type}({target}); open/close needs one free "
                    "end-effector. Open/close before acquiring the item, "
                    "deposit first, or assign a robot with a free end-effector."
                ),
                op_ids=[op.op_id],
            ))
    return issues


def _held_items_before_op(
    op: OperationStep,
    op_by_id: dict[str, OperationStep],
) -> set[str]:
    held: set[str] = set()
    for dep_op in _dependency_chain_topological(op, op_by_id):
        if not _same_robot_slot(dep_op.robot, op.robot):
            continue
        _apply_holding_effect(held, dep_op)
    return held


def _dependency_chain_topological(
    op: OperationStep,
    op_by_id: dict[str, OperationStep],
) -> list[OperationStep]:
    visited: set[str] = set()
    order: list[OperationStep] = []

    def visit(dep_id: str) -> None:
        if dep_id in visited or dep_id == op.op_id:
            return
        visited.add(dep_id)
        dep_op = op_by_id.get(dep_id)
        if dep_op is None:
            return
        for child_dep in dep_op.deps:
            visit(child_dep)
        order.append(dep_op)

    for dep_id in op.deps:
        visit(dep_id)
    return order


def _apply_holding_effect(held: set[str], op: OperationStep) -> None:
    entity = normalize_entity_name(op.entity)
    if not entity:
        return
    if op.op_type in _HOLD_ADD_OPS:
        held.add(entity)
    elif op.op_type in _HOLD_RELEASE_OPS:
        held.discard(entity)


def _detect_invalid_open_ops(ops: list[OperationStep]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type != "open":
            continue
        target = op.dest or op.entity or ""
        if entity_requires_open_before_place(target):
            continue
        issues.append(ValidationIssue(
            code="open_on_non_openable",
            message=f"open({target}) targets a non-openable entity; remove this op.",
            op_ids=[op.op_id],
        ))
    return issues


def _detect_duplicate_open_ops(ops: list[OperationStep]) -> list[ValidationIssue]:
    seen: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type != "open":
            continue
        target = normalize_entity_name(op.dest or op.entity)
        if not target:
            continue
        if target not in seen:
            seen[target] = op.op_id
            continue
        issues.append(ValidationIssue(
            code="duplicate_open",
            message=(
                f"duplicate open({target}); keep only one open op per target "
                f"(first was {seen[target]})."
            ),
            op_ids=[op.op_id],
        ))
    return issues


def _detect_redundant_wash_activation(ops: list[OperationStep]) -> list[ValidationIssue]:
    process_stations = {
        normalize_entity_name(op.dest)
        for op in ops
        if op.op_type == "process_at"
    }
    process_stations.discard(None)
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type != "activate":
            continue
        target = normalize_entity_name(op.dest or op.entity)
        if target not in process_stations:
            continue
        if WASH_STATION_TAG not in set(get_entity_tags(op.dest or op.entity or "")):
            continue
        issues.append(ValidationIssue(
            code="redundant_wash_activate",
            message=(
                f"activate({target}) is redundant because process_at already "
                "drives this wash station; remove the activate op."
            ),
            op_ids=[op.op_id],
        ))
    return issues


def _detect_duplicate_activations(ops: list[OperationStep]) -> list[ValidationIssue]:
    seen: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type != "activate":
            continue
        target = normalize_entity_name(op.dest or op.entity)
        if not target:
            continue
        if target not in seen:
            seen[target] = op.op_id
            continue
        issues.append(ValidationIssue(
            code="duplicate_activate",
            message=(
                f"duplicate activate({target}); keep only one activate per target "
                f"(first was {seen[target]})."
            ),
            op_ids=[op.op_id],
        ))
    return issues


def _detect_capability_violations(
    ops: list[OperationStep],
    ctx: AffordanceContext,
) -> list[ValidationIssue]:
    robot_caps_by_slot = {
        r.slot_id: set(r.capabilities)
        for r in ctx.robot_capabilities
    }
    valid_slots = sorted(robot_caps_by_slot.keys())
    issues: list[ValidationIssue] = []
    for op in ops:
        if op.op_type in {"handoff", "push_carrier"}:
            target_robot = _op_target_robot(op)
            # push_carrier has two legal targets: a receiving robot (VIKI-L2's
            # only shape) or a destination anchor such as "sink" (LOCUS F9-M2;
            # the catalog primitive has always documented "toward a receiver or
            # destination anchor"). An anchor push carries no target_robot and
            # names the anchor in `dest`, so demanding a slot here rejected
            # valid ground truth. handoff stays strict -- it is robot-to-robot
            # by definition.
            anchor_push = (
                op.op_type == "push_carrier"
                and not target_robot
                and bool(str(op.dest or "").strip())
            )
            if anchor_push:
                pass
            elif not target_robot or not str(target_robot).strip():
                issues.append(ValidationIssue(
                    code="capability_mismatch",
                    message=(
                        f"{op.op_type} op requires params.target_robot or "
                        f"params.target_slot; pick one of {valid_slots}."
                    ),
                    op_ids=[op.op_id],
                ))
            elif str(target_robot) not in robot_caps_by_slot:
                issues.append(ValidationIssue(
                    code="capability_mismatch",
                    message=(
                        f"{op.op_type} target_robot={target_robot} is not a "
                        f"valid slot; choose one of {valid_slots}."
                    ),
                    op_ids=[op.op_id],
                ))
        required = _required_caps_for_op(op)
        if not required:
            continue
        if not op.robot:
            issues.append(ValidationIssue(
                code="missing_robot",
                message=(
                    f"{op.op_type} op has no robot slot assigned; pick from "
                    f"{valid_slots}."
                ),
                op_ids=[op.op_id],
            ))
            continue
        if op.robot not in robot_caps_by_slot:
            issues.append(ValidationIssue(
                code="unknown_robot_slot",
                message=(
                    f"robot={op.robot} is not a valid slot; choose one of "
                    f"{valid_slots}."
                ),
                op_ids=[op.op_id],
            ))
            continue
        missing = required - robot_caps_by_slot[op.robot]
        if missing:
            issues.append(ValidationIssue(
                code="capability_mismatch",
                message=(
                    f"robot={op.robot} lacks required capabilities {sorted(missing)} "
                    f"for {op.op_type}; reassign to a slot whose capabilities cover "
                    f"{sorted(required)}."
                ),
                op_ids=[op.op_id],
            ))
    return issues


def _detect_dependency_violations(ops: list[OperationStep]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    op_ids = {op.op_id for op in ops}
    for op in ops:
        invalid = [d for d in op.deps if d not in op_ids or d == op.op_id]
        if invalid:
            issues.append(ValidationIssue(
                code="invalid_dep_ref",
                message=(
                    f"deps reference unknown or self op_id(s) {sorted(set(invalid))}; "
                    "fix the dep list."
                ),
                op_ids=[op.op_id],
            ))
    cycle = _detect_cycle(ops)
    if cycle:
        issues.append(ValidationIssue(
            code="dependency_cycle",
            message=(
                f"deps form a cycle through {cycle}; rewire deps so the DAG is acyclic."
            ),
            op_ids=cycle,
        ))
    return issues


def _detect_cycle(ops: list[OperationStep]) -> list[str]:
    """Return the op_ids participating in any unresolved cycle (via Kahn)."""
    op_by_id = {op.op_id: op for op in ops}
    remaining = set(op_by_id)
    while remaining:
        ready = [
            oid for oid in remaining
            if all(d not in remaining for d in op_by_id[oid].deps)
        ]
        if not ready:
            return sorted(remaining)
        for oid in ready:
            remaining.remove(oid)
    return []

def _required_caps_for_op(op: OperationStep) -> set[str]:
    """Derive required capabilities from op_type (no AIG edges needed)."""
    t = op.op_type
    if t == "acquire":
        return {"mobility", "manipulation"}
    if t in {"local_acquire", "deposit", "load_carrier", "unload_carrier", "handoff"}:
        return {"manipulation"}
    if t == "carry_to":
        return {"mobility"}
    if t == "open":
        return {"mobility", "container_operation"}
    if t == "close":
        return {"mobility", "container_operation"}
    if t == "activate":
        return {"mobility", "activation"}
    if t == "process_at":
        return {"mobility", "manipulation", "activation"}
    if t == "apply_tool":
        return {"activation"}
    if t == "push_carrier":
        return {"mobility", "push_transfer"}
    if t == "inspect":
        return {"mobility", "activation"}
    return set()


# ── OperationStep helpers ─────────────────────────────────────────────

def _find_same_entity_dep_op(
    op: OperationStep,
    ops: list[OperationStep],
    op_types: set[str],
) -> OperationStep | None:
    normalized_entity = normalize_entity_name(op.entity)
    if not normalized_entity:
        return None
    dep_ids = set(op.deps)
    for candidate in reversed(ops):
        if candidate.op_id not in dep_ids:
            continue
        if candidate.op_type not in op_types:
            continue
        if normalize_entity_name(candidate.entity) == normalized_entity:
            return candidate
    return None


# =========================================================================
# 6. OperationStep list → GraphPlan conversion
# =========================================================================

def convert_operations_to_graph_plan(
    ops: list[OperationStep],
    affordance_context: AffordanceContext,
    scene_facts: SceneFacts,
) -> GraphPlan:
    """Convert validated OperationStep list into a GraphPlan with AffordanceNodes."""
    from nsdl.catalog import get_node_kind_spec

    nodes: list[AffordanceNode] = []
    op_to_node_id: dict[str, str] = {}
    open_nodes_by_dest: dict[str, AffordanceNode] = {}
    carry_nodes_by_dest_slot: dict[tuple[str, str | None], list[str]] = {}
    node_index = 1

    for op in ops:
        if _should_expand_cutting_process(op):
            new_nodes = _expand_cutting_process_op(
                op,
                next_node_id=f"n{node_index}",
                op_to_node_id=op_to_node_id,
                all_ops=ops,
                scene_facts=scene_facts,
            )
            if new_nodes:
                nodes.extend(new_nodes)
                node_index += len(new_nodes)
                if op.robot:
                    for node in new_nodes:
                        node.executor_slot = op.robot
                op_to_node_id[op.op_id] = new_nodes[-1].node_id
                continue

        if _should_expand_openable_deposit(op, ops):
            new_nodes = _expand_openable_deposit_op(
                op,
                next_node_id=f"n{node_index}",
                op_to_node_id=op_to_node_id,
                all_ops=ops,
                scene_facts=scene_facts,
            )
            nodes.extend(new_nodes)
            node_index += len(new_nodes)
            if op.robot:
                for node in new_nodes:
                    node.executor_slot = op.robot
            carry_node = new_nodes[0]
            carry_dest = normalize_entity_name(op.dest) or op.dest or ""
            carry_nodes_by_dest_slot.setdefault((carry_dest, op.robot), []).append(carry_node.node_id)
            op_to_node_id[op.op_id] = new_nodes[-1].node_id
            continue

        node_id = f"n{node_index}"
        node_index += 1
        predecessors = [
            op_to_node_id[dep]
            for dep in op.deps
            if dep in op_to_node_id
        ]

        node = _op_to_affordance_node(
            op,
            node_id=node_id,
            predecessors=predecessors,
            op_to_node_id=op_to_node_id,
            all_ops=ops,
            scene_facts=scene_facts,
        )

        if op.robot:
            node.executor_slot = op.robot

        if op.op_type == "open":
            container = normalize_entity_name(op.dest or op.entity)
            if container:
                open_nodes_by_dest[container] = node

        # Set required capabilities from op_type
        required = sorted(_required_caps_for_op(op))
        if required:
            node.required_affordances_hard = required

        nodes.append(node)
        op_to_node_id[op.op_id] = node_id

    for dest, open_node in open_nodes_by_dest.items():
        slot = open_node.executor_slot
        carry_candidates = carry_nodes_by_dest_slot.get((dest, slot), [])
        if not carry_candidates:
            for (candidate_dest, _candidate_slot), node_ids in carry_nodes_by_dest_slot.items():
                if candidate_dest == dest:
                    carry_candidates = node_ids
                    break
        if carry_candidates and carry_candidates[0] not in open_node.predecessors:
            open_node.predecessors.append(carry_candidates[0])

    task_family = _classify_task_family(nodes, scene_facts)

    return GraphPlan(
        task_family=task_family,
        build_mode="acc_dag",
        family_shortcut_used=False,
        compiler_notes=[
            "SLM-generated operation DAG assembled after Catalog-tag-based validation.",
        ],
        nodes=nodes,
    )


def _op_to_affordance_node(
    op: OperationStep,
    *,
    node_id: str,
    predecessors: list[str],
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
    scene_facts: SceneFacts,
) -> AffordanceNode:
    """Map a single OperationStep to an AffordanceNode."""
    entity = op.entity
    dest = op.dest

    # Find same-entity predecessor for custody/constraint propagation.
    acquire_source = _same_entity_pred_node_any(
        op,
        {"carry_to", "acquire", "local_acquire"},
        op_to_node_id,
        all_ops,
    )
    process_source = _same_entity_pred_node(op, "process_at", op_to_node_id, all_ops)

    if op.op_type == "acquire":
        constraints = {}
        if _should_prefer_new_parallel_slot(op, scene_facts):
            constraints["prefer_new_slot"] = True
        if _feeds_later_same_entity_op_type(op, {"load_carrier"}, all_ops):
            constraints["push_transfer_role"] = "source_holder"
        return _build_node(
            node_id=node_id,
            node_kind="relocate_object",
            semantic_goal=f"Acquire {entity}",
            object_name=entity,
            destination=dest,
            object_bindings={"item": entity} if entity else {},
            required_action_primitives=["move_to", "reach_object", "grasp_item"],
            required_constraints=constraints,
            predecessors=predecessors,
            dispatcher_scaffold="pickup_only",
        )

    if op.op_type == "local_acquire":
        constraints = {}
        if _feeds_later_same_entity_op_type(op, {"load_carrier"}, all_ops):
            constraints["push_transfer_role"] = "source_holder"
        return _build_node(
            node_id=node_id,
            node_kind="local_manipulation",
            semantic_goal=f"Acquire {entity} in local workspace",
            object_name=entity,
            destination=dest,
            object_bindings={"item": entity} if entity else {},
            required_action_primitives=["reach_object", "grasp_item"],
            required_constraints=constraints,
            predecessors=predecessors,
            dispatcher_scaffold="grasp_item_in_place",
        )

    if op.op_type == "carry_to":
        constraints = {"same_executor_as": acquire_source} if acquire_source else {}
        return _build_node(
            node_id=node_id,
            node_kind="relocate_object",
            semantic_goal=f"Carry {entity} to {dest}",
            object_name=entity,
            destination=dest,
            object_bindings={"item": entity, "container": dest, "target_area": dest} if entity and dest else {},
            required_action_primitives=["move_to"],
            required_constraints=constraints,
            predecessors=predecessors,
            dispatcher_scaffold="carry_to_container_anchor",
        )

    if op.op_type == "deposit":
        constraints = {}
        if acquire_source:
            constraints["same_executor_as"] = acquire_source
        elif process_source:
            constraints["same_executor_as"] = process_source
        elif _should_prefer_new_parallel_slot(op, scene_facts):
            constraints["prefer_new_slot"] = True
        scaffold = "move_held_item_then_place" if acquire_source or process_source else "pickup_and_place"
        if acquire_source and _find_pred_op(op, all_ops, {"local_acquire"}) is not None:
            # Deposit sourced from a local-workspace acquire: the executor is a
            # fixed arm whose whole task stays in reach, so the benchmark
            # expands it without a Move. VIKI deposits are never sourced from
            # local_acquire (train audit: 9275/9275 acquire), so this branch
            # only fires on LOCUS shapes.
            scaffold = "place_held_item"
        elif acquire_source:
            carry_pred = _find_pred_op(op, all_ops, {"carry_to"})
            if carry_pred is not None:
                if _same_dest(carry_pred.dest, dest):
                    scaffold = "place_held_item"
                # else: two-level destination -- the carry staged the item at
                # an area and the container is a further container-level Move
                # away, so keep move_held_item_then_place; collapsing to
                # place_held_item here is what dropped the Move on LOCUS
                # carrier rows (debug_sft.md section 4.5).
        return _build_node(
            node_id=node_id,
            node_kind="relocate_object",
            semantic_goal=f"Deposit {entity} at {dest}",
            object_name=entity,
            destination=dest,
            object_bindings={"item": entity, "target_area": dest} if entity else {},
            required_action_primitives=["move_to", "place_item"],
            required_constraints=constraints,
            predecessors=predecessors,
            dispatcher_scaffold=scaffold,
        )

    if op.op_type == "open":
        container = dest or entity
        return _build_node(
            node_id=node_id,
            node_kind="open_container",
            semantic_goal=f"Open {container}",
            object_name=container,
            destination=container,
            object_bindings={"container": container} if container else {},
            required_action_primitives=["move_to", "reach_object", "open_container"],
            predecessors=predecessors,
            dispatcher_scaffold="move_reach_open_container",
        )

    if op.op_type == "close":
        container = dest or entity
        return _build_node(
            node_id=node_id,
            node_kind="close_container",
            semantic_goal=f"Close {container}",
            object_name=container,
            destination=container,
            object_bindings={"container": container} if container else {},
            required_action_primitives=["move_to", "reach_object", "close_container"],
            predecessors=predecessors,
            dispatcher_scaffold="move_reach_close_container",
        )

    if op.op_type == "process_at":
        target_tags = set(get_entity_tags(dest or ""))
        if CUTTING_SURFACE_TAG in target_tags:
            tool = normalize_entity_name(str(op.params.get("tool", ""))) or None
            return _build_node(
                node_id=node_id,
                node_kind="tool_interaction",
                semantic_goal=f"Use {tool} at {dest}",
                object_name=tool,
                destination=dest,
                object_bindings={"target": tool} if tool else {},
                required_action_primitives=["move_to", "reach_object", "grasp_item", "interact_with"],
                required_constraints={"same_executor_as": acquire_source} if acquire_source else {},
                predecessors=predecessors,
                dispatcher_scaffold="tool_interaction_after_acquire" if acquire_source else "interact_in_place",
            )
        # Default: wash_object (or generic process)
        process = str(op.params.get("process", "wash")).strip().lower() or "wash"
        bindings = {"item": entity, "station": dest, "process": process} if entity else {"process": process}
        return _build_node(
            node_id=node_id,
            node_kind="wash_object",
            semantic_goal=f"{process.capitalize()} {entity} at {dest}",
            object_name=entity,
            destination=dest,
            object_bindings=bindings,
            required_action_primitives=["move_to", "reach_object", "grasp_item", "place_item", "interact_with"],
            required_constraints={"same_executor_as": acquire_source} if acquire_source else {},
            predecessors=predecessors,
            dispatcher_scaffold="wash_at_station",
        )

    if op.op_type == "apply_tool":
        tool = normalize_entity_name(str(op.params.get("tool", ""))) or entity
        effect = str(op.params.get("effect", "cut")).strip().lower() or "cut"
        tool_source = _same_entity_pred_node_any(
            OperationStep(op_id=op.op_id, op_type=op.op_type, entity=tool, deps=op.deps),
            {"carry_to", "acquire", "local_acquire"},
            op_to_node_id,
            all_ops,
        )
        tool_bindings = {"target": tool, "effect": effect} if tool else {"effect": effect}
        if dest:
            tool_bindings["target_area"] = dest
        return _build_node(
            node_id=node_id,
            node_kind="tool_interaction",
            semantic_goal=f"Use {tool} to {effect} at {dest}",
            object_name=tool,
            destination=dest,
            object_bindings=tool_bindings,
            required_action_primitives=["reach_object", "interact_with"],
            required_constraints={"same_executor_as": tool_source} if tool_source else {},
            predecessors=predecessors,
            dispatcher_scaffold="interact_in_place",
        )

    if op.op_type == "activate":
        target = dest or entity
        return _build_node(
            node_id=node_id,
            node_kind="activate_fixture",
            semantic_goal=f"Activate {target}",
            object_name=target,
            destination=target,
            object_bindings={"target": target} if target else {},
            required_action_primitives=["move_to", "interact_with"],
            predecessors=predecessors,
            dispatcher_scaffold="move_then_interact",
        )

    if op.op_type == "load_carrier":
        carrier = _op_carrier(op)
        constraints = {"same_executor_as": acquire_source} if acquire_source else {}
        return _build_node(
            node_id=node_id,
            node_kind="local_manipulation",
            semantic_goal=f"Load {entity} into {carrier}",
            object_name=entity,
            destination=carrier,
            object_bindings={"item": entity, "target_area": carrier} if entity and carrier else {},
            required_action_primitives=["place_item"],
            required_constraints=constraints,
            predecessors=predecessors,
            dispatcher_scaffold="place_held_item",
        )

    if op.op_type == "push_carrier":
        carrier = _op_carrier(op) or entity
        target_slot = _op_target_robot(op) or dest
        previous_push = _same_entity_pred_node_any(
            OperationStep(op_id=op.op_id, op_type=op.op_type, entity=carrier, deps=op.deps),
            {"push_carrier"},
            op_to_node_id,
            all_ops,
        ) or _prior_same_entity_node_any(
            OperationStep(op_id=op.op_id, op_type=op.op_type, entity=carrier, deps=op.deps),
            {"push_carrier"},
            op_to_node_id,
            all_ops,
        )
        constraints = {"push_transfer_role": "carrier_pusher"}
        if target_slot:
            constraints["handoff_target_slot"] = target_slot
        constraints["push_transfer_phase"] = (
            "to_receiver_holder" if previous_push else "to_source_holder"
        )
        if previous_push:
            constraints["same_executor_as"] = previous_push
        return _build_node(
            node_id=node_id,
            node_kind="push_transport",
            semantic_goal=f"Push {carrier} to {target_slot or 'handoff target'}",
            object_name=carrier,
            destination=target_slot or dest or carrier,
            object_bindings={"carrier": carrier} if carrier else {},
            required_action_primitives=["move_to", "push_carrier"],
            required_constraints=constraints,
            predecessors=predecessors,
            dispatcher_scaffold="push_carrier_to_handoff",
        )

    if op.op_type == "unload_carrier":
        carrier = _op_carrier(op)
        return _build_node(
            node_id=node_id,
            node_kind="local_manipulation",
            semantic_goal=f"Unload {entity} from {carrier} to {dest}",
            object_name=entity,
            destination=dest,
            object_bindings={"item": entity, "target_area": dest} if entity and dest else {},
            required_action_primitives=["reach_object", "grasp_item", "place_item"],
            required_constraints={"push_transfer_role": "receiver_holder"},
            predecessors=predecessors,
            dispatcher_scaffold="unload_from_carrier_to_destination",
        )

    if op.op_type == "handoff":
        return _build_node(
            node_id=node_id,
            node_kind="handoff_node",
            semantic_goal=f"Handoff {entity} at capability boundary",
            object_name=entity,
            destination=dest or "capability_boundary",
            object_bindings={"item": entity, "target_area": dest or "capability_boundary"} if entity else {},
            required_action_primitives=[],
            predecessors=predecessors,
            dispatcher_scaffold="handoff_anchor",
        )

    if op.op_type == "inspect":
        target = dest or entity or "inspection_target"
        return _build_node(
            node_id=node_id,
            node_kind="inspect_object",
            semantic_goal=f"Inspect {target}",
            object_name=target,
            destination=target,
            object_bindings={"target": target} if target else {},
            required_action_primitives=["move_to", "interact_with"],
            predecessors=predecessors,
            dispatcher_scaffold="move_then_interact",
        )

    raise ValueError(f"Unsupported op_type: {op.op_type}")


def _should_expand_openable_deposit(
    op: OperationStep,
    ops: list[OperationStep],
) -> bool:
    if op.op_type != "deposit" or not op.dest:
        return False
    dest = normalize_entity_name(op.dest)
    if not dest or not entity_requires_open_before_place(dest):
        return False
    if _find_same_entity_dep_op(op, ops, {"carry_to"}):
        return False
    acquire_ids = {
        candidate.op_id
        for candidate in ops
        if candidate.op_type == "acquire"
    }
    return any(
        candidate.op_type == "open"
        and normalize_entity_name(candidate.dest or candidate.entity) == dest
        and any(dep in acquire_ids for dep in candidate.deps)
        for candidate in ops
    )


def _should_expand_cutting_process(op: OperationStep) -> bool:
    if op.op_type != "process_at" or not op.dest:
        return False
    return CUTTING_SURFACE_TAG in get_entity_tags(op.dest)


def _expand_cutting_process_op(
    op: OperationStep,
    *,
    next_node_id: str,
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
    scene_facts: SceneFacts,
) -> list[AffordanceNode]:
    tool = normalize_entity_name(str(op.params.get("tool", ""))) or None
    if not tool:
        return []
    node_number = int(next_node_id.removeprefix("n"))
    tool_acquire_node = None
    other_predecessors: list[str] = []
    for dep in op.deps:
        dep_op = next((candidate for candidate in all_ops if candidate.op_id == dep), None)
        mapped_node_id = op_to_node_id.get(dep)
        if mapped_node_id is None:
            continue
        if dep_op and dep_op.op_type == "acquire" and normalize_entity_name(dep_op.entity) == tool:
            tool_acquire_node = mapped_node_id
        else:
            other_predecessors.append(mapped_node_id)
    if tool_acquire_node is None:
        return []

    move_node = _build_node(
        node_id=f"n{node_number}",
        node_kind="tool_interaction",
        semantic_goal=f"Move {tool} to {op.dest}",
        object_name=tool,
        destination=op.dest,
        object_bindings={"target_area": op.dest},
        required_action_primitives=["move_to"],
        required_constraints={"same_executor_as": tool_acquire_node},
        predecessors=[tool_acquire_node],
        dispatcher_scaffold="move_held_tool_to_target",
    )
    interact_node = _build_node(
        node_id=f"n{node_number + 1}",
        node_kind="tool_interaction",
        semantic_goal=f"Use {tool} at {op.dest}",
        object_name=tool,
        destination=op.dest,
        object_bindings={"target": tool},
        required_action_primitives=["interact_with"],
        required_constraints={"same_executor_as": move_node.node_id},
        predecessors=[move_node.node_id, *other_predecessors],
        dispatcher_scaffold="interact_in_place",
    )
    return [move_node, interact_node]


def _expand_openable_deposit_op(
    op: OperationStep,
    *,
    next_node_id: str,
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
    scene_facts: SceneFacts,
) -> list[AffordanceNode]:
    node_number = int(next_node_id.removeprefix("n"))
    entity = op.entity
    dest = op.dest
    open_dep_node_ids = []
    carry_predecessors = []
    for dep in op.deps:
        dep_op = next((candidate for candidate in all_ops if candidate.op_id == dep), None)
        mapped_node_id = op_to_node_id.get(dep)
        if mapped_node_id is None:
            continue
        if (
            dep_op
            and dep_op.op_type == "open"
            and normalize_entity_name(dep_op.dest or dep_op.entity) == normalize_entity_name(dest)
        ):
            open_dep_node_ids.append(mapped_node_id)
            continue
        carry_predecessors.append(mapped_node_id)

    acquire_source = _same_entity_pred_node_any(
        op,
        {"carry_to", "acquire", "local_acquire"},
        op_to_node_id,
        all_ops,
    ) or _prior_same_entity_node_any(
        op,
        {"carry_to", "acquire", "local_acquire"},
        op_to_node_id,
        all_ops,
    )
    if acquire_source and acquire_source not in carry_predecessors:
        carry_predecessors.append(acquire_source)
    carry_constraints = {"same_executor_as": acquire_source} if acquire_source else {}
    carry_node = _build_node(
        node_id=f"n{node_number}",
        node_kind="relocate_object",
        semantic_goal=f"Carry {entity} to {dest}",
        object_name=entity,
        destination=dest,
        object_bindings={"item": entity, "container": dest, "target_area": dest} if entity and dest else {},
        required_action_primitives=["move_to"],
        required_constraints=carry_constraints,
        predecessors=carry_predecessors,
        dispatcher_scaffold="carry_to_container_anchor",
    )

    place_predecessors = [carry_node.node_id, *open_dep_node_ids]
    place_node = _build_node(
        node_id=f"n{node_number + 1}",
        node_kind="relocate_object",
        semantic_goal=f"Deposit {entity} at {dest}",
        object_name=entity,
        destination=dest,
        object_bindings={"item": entity, "target_area": dest} if entity and dest else {},
        required_action_primitives=["place_item"],
        required_constraints={"same_executor_as": carry_node.node_id},
        predecessors=place_predecessors,
        dispatcher_scaffold="place_held_item",
    )
    return [carry_node, place_node]


def _should_prefer_new_parallel_slot(
    op: OperationStep,
    scene_facts: SceneFacts,
) -> bool:
    return False


def _op_carrier(op: OperationStep) -> str | None:
    carrier = op.params.get("carrier")
    if isinstance(carrier, str):
        return normalize_entity_name(carrier)
    if op.op_type in {"push_carrier"} and op.entity:
        return normalize_entity_name(op.entity)
    if op.dest and PUSHABLE_CARRIER_TAG in get_entity_tags(op.dest):
        return normalize_entity_name(op.dest)
    return "cardboardbox" if op.op_type in {"load_carrier", "unload_carrier"} else None


def _op_target_robot(op: OperationStep) -> str | None:
    for key in ("target_robot", "target_slot"):
        value = op.params.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_robot_slot(value)
    if op.dest and op.dest[:1].lower() == "r" and op.dest[1:].isdigit():
        return _normalize_robot_slot(op.dest)
    return None


def _build_node(
    *,
    node_id: str,
    node_kind: str,
    semantic_goal: str,
    object_name: str | None,
    destination: str | None,
    object_bindings: dict[str, str] | None = None,
    required_action_primitives: list[str] | None = None,
    required_constraints: dict[str, Any] | None = None,
    predecessors: list[str] | None = None,
    dispatcher_scaffold: str | None = None,
) -> AffordanceNode:
    from nsdl.catalog import get_node_kind_spec
    spec = get_node_kind_spec(node_kind)
    bindings = dict(object_bindings or {})
    constraints = dict(required_constraints or {})
    goal_predicates, start_predicates_hint, state_exports = _derive_micro_planner_contract(
        node_kind=node_kind,
        object_name=object_name,
        destination=destination,
        object_bindings=bindings,
        required_constraints=constraints,
        dispatcher_scaffold=dispatcher_scaffold,
    )
    has_item_at_goal = any(predicate.startswith("item_at(") for predicate in goal_predicates)
    return AffordanceNode(
        node_id=node_id,
        node_kind=node_kind,
        semantic_goal=semantic_goal,
        object_name=object_name,
        destination=destination,
        required_affordances_hard=list(spec.required_affordances_hard) if spec else [],
        required_affordances_soft=list(spec.required_affordances_soft) if spec else [],
        object_bindings=bindings,
        required_action_primitives=required_action_primitives or [],
        required_constraints=constraints,
        predecessors=predecessors or [],
        goal_predicates=goal_predicates,
        start_predicates_hint=start_predicates_hint,
        state_exports=state_exports,
        output_interface="object_at_destination" if has_item_at_goal else None,
        completion_check={"pos.name": destination} if has_item_at_goal and destination else {},
        dispatcher_scaffold=dispatcher_scaffold,
    )


def _derive_micro_planner_contract(
    *,
    node_kind: str,
    object_name: str | None,
    destination: str | None,
    object_bindings: dict[str, str],
    required_constraints: dict[str, Any],
    dispatcher_scaffold: str | None,
) -> tuple[list[str], list[str], list[str]]:
    """Derive the execution contract expected by the regression micro planner."""
    scaffold = dispatcher_scaffold or node_kind
    item = object_bindings.get("item", object_name)
    target_area = object_bindings.get("target_area", destination)
    container = object_bindings.get("container")
    station = object_bindings.get("station", destination)
    target = object_bindings.get("target", object_name)
    tool = object_name or object_bindings.get("item")

    goal_predicates: list[str] = []
    start_predicates_hint: list[str] = []
    state_exports: list[str] = []

    def add_goal(name: str, *args: str | None) -> None:
        predicate = _make_predicate(name, *args)
        if predicate and predicate not in goal_predicates:
            goal_predicates.append(predicate)

    def add_start(name: str, *args: str | None) -> None:
        predicate = _make_predicate(name, *args)
        if predicate and predicate not in start_predicates_hint:
            start_predicates_hint.append(predicate)

    def add_export(name: str, *args: str | None) -> None:
        predicate = _make_predicate(name, *args)
        if predicate and predicate not in state_exports:
            state_exports.append(predicate)

    if scaffold == "pickup_only":
        add_goal("holding", item or tool)
        add_export("holding", item or tool)
    elif scaffold == "pickup_and_place":
        add_goal("item_at", item, target_area)
        add_export("item_at", item, target_area)
        add_export("executor_at", target_area)
    elif scaffold == "carry_to_container_anchor":
        add_goal("holding", item)
        add_goal("executor_at", container or target_area)
        add_export("holding", item)
        add_export("executor_at", container or target_area)
    elif scaffold == "move_held_tool_to_target":
        add_goal("holding", tool)
        add_goal("executor_at", target_area)
        add_start("holding", tool)
        add_export("holding", tool)
        add_export("executor_at", target_area)
    elif scaffold == "move_held_item_then_place":
        add_goal("item_at", item, target_area)
        add_start("holding", item)
        add_export("item_at", item, target_area)
        add_export("executor_at", target_area)
    elif scaffold == "place_held_item":
        add_goal("item_at", item, target_area)
        add_start("holding", item)
        add_start("in_local_workspace", target_area)
        add_export("item_at", item, target_area)
    elif scaffold == "move_reach_open_container":
        add_goal("is_open", container or object_name)
        add_export("is_open", container or object_name)
        add_export("executor_at", container or object_name)
    elif scaffold == "move_reach_close_container":
        add_goal("is_closed", container or object_name)
        add_export("is_closed", container or object_name)
        add_export("executor_at", container or object_name)
    elif scaffold == "wash_at_station":
        add_goal("clean", item)
        add_export("clean", item)
        add_export("item_at", item, station)
        add_export("executor_at", station)
    elif scaffold == "move_then_interact":
        if node_kind in {"inspect_object", "inspect_presence"}:
            add_goal("inspected", target)
            add_export("inspected", target)
        else:
            add_goal("activated", target)
            add_export("activated", target)
        add_export("executor_at", target)
    elif scaffold == "interact_in_place":
        if node_kind == "tool_interaction":
            add_goal("tool_effect_applied", tool, destination)
            add_export("tool_effect_applied", tool, destination)
        elif node_kind in {"inspect_object", "inspect_presence"}:
            add_goal("inspected", target)
            add_export("inspected", target)
        else:
            add_goal("activated", target)
            add_export("activated", target)
    elif scaffold == "tool_interaction_after_acquire":
        add_goal("tool_effect_applied", tool, destination)
        add_start("holding", tool)
        add_start("executor_at", tool)
        add_export("tool_effect_applied", tool, destination)
        add_export("executor_at", destination)
    elif scaffold == "handoff_anchor":
        add_goal("handoff", item or object_name, target_area or destination)
        add_start("handoff", item or object_name, target_area or destination)
        add_export("handoff", item or object_name, target_area or destination)
    elif scaffold == "grasp_item_in_place":
        add_goal("holding", item)
        add_start("in_local_workspace", item)
        add_export("holding", item)
    elif scaffold == "move_carrier_only":
        carrier = object_bindings.get("carrier", object_name)
        add_goal("executor_at", carrier)
        add_export("executor_at", carrier)
    elif scaffold == "push_carrier_to_handoff":
        carrier = object_bindings.get("carrier", object_name)
        add_goal("carrier_at", carrier, "handoff_target")
        add_export("carrier_at", carrier, "handoff_target")
    elif scaffold == "unload_from_carrier_to_destination":
        add_goal("item_at", item, target_area)
        add_start("in_local_workspace", item)
        add_start("in_local_workspace", target_area)
        add_export("item_at", item, target_area)
    else:
        if node_kind == "relocate_object":
            add_goal("item_at", item or object_name, target_area or destination)
        elif node_kind == "open_container":
            add_goal("is_open", object_name)
        elif node_kind in {"inspect_object", "inspect_presence"}:
            add_goal("inspected", target or object_name)
        elif node_kind == "activate_fixture":
            add_goal("activated", target or object_name)
        elif node_kind == "wash_object":
            add_goal("clean", item or object_name)
        elif node_kind == "handoff_node":
            add_goal("handoff", item or object_name, target_area or destination)
            add_start("handoff", item or object_name, target_area or destination)

    min_end_effectors = int(required_constraints.get("min_end_effectors", 0) or 0)
    if min_end_effectors > 0:
        add_start("min_end_effectors", str(min_end_effectors))

    return goal_predicates, start_predicates_hint, state_exports


def _make_predicate(name: str, *args: str | None) -> str | None:
    values = [str(arg) for arg in args if arg is not None and str(arg).strip()]
    if not values:
        return None
    return f"{name}({', '.join(values)})"


def _same_entity_pred_node(
    op: OperationStep,
    pred_op_type: str,
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
) -> str | None:
    """Find a predecessor node that operates on the same entity."""
    return _same_entity_pred_node_any(op, {pred_op_type}, op_to_node_id, all_ops)


def _same_entity_pred_node_any(
    op: OperationStep,
    pred_op_types: set[str],
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
) -> str | None:
    """Find a predecessor node among several op types for the same entity."""
    dep_ids = set(op.deps)
    for pred in reversed(all_ops):
        if pred.op_id not in dep_ids:
            continue
        if pred.op_type not in pred_op_types:
            continue
        if normalize_entity_name(pred.entity) != normalize_entity_name(op.entity):
            continue
        node_id = op_to_node_id.get(pred.op_id)
        if node_id:
            return node_id
    return None


def _prior_same_entity_node_any(
    op: OperationStep,
    pred_op_types: set[str],
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
) -> str | None:
    """Find an already converted prior node for the same entity."""
    normalized_entity = normalize_entity_name(op.entity)
    if not normalized_entity:
        return None
    seen_current = False
    for pred in reversed(all_ops):
        if pred.op_id == op.op_id:
            seen_current = True
            continue
        if not seen_current:
            continue
        if pred.op_type not in pred_op_types:
            continue
        if normalize_entity_name(pred.entity) != normalized_entity:
            continue
        node_id = op_to_node_id.get(pred.op_id)
        if node_id:
            return node_id
    return None


def _find_node_for_op_id(
    op_to_node_id: dict[str, str],
    all_ops: list[OperationStep],
    op: OperationStep,
    pred_op_types: set[str],
) -> str | None:
    dep_ids = set(op.deps)
    for pred in reversed(all_ops):
        if pred.op_id not in dep_ids or pred.op_type not in pred_op_types:
            continue
        node_id = op_to_node_id.get(pred.op_id)
        if node_id:
            return node_id
    return None


def _find_pred_op(
    op: OperationStep,
    all_ops: list[OperationStep],
    pred_op_types: set[str],
) -> OperationStep | None:
    """Latest dependency of ``op`` whose op_type is in ``pred_op_types``.

    Same dep-based lookup as ``_find_node_for_op_id`` but returns the op
    itself: the deposit scaffold needs the predecessor's ``dest``, not just
    its node id.
    """
    dep_ids = set(op.deps)
    for pred in reversed(all_ops):
        if pred.op_id in dep_ids and pred.op_type in pred_op_types:
            return pred
    return None


def _same_dest(left: str | None, right: str | None) -> bool:
    return (left or "").strip().lower() == (right or "").strip().lower()


def _feeds_later_same_entity_op_type(
    op: OperationStep,
    later_op_types: set[str],
    all_ops: list[OperationStep],
) -> bool:
    normalized_entity = normalize_entity_name(op.entity)
    if not normalized_entity:
        return False
    seen_current = False
    for candidate in all_ops:
        if candidate.op_id == op.op_id:
            seen_current = True
            continue
        if not seen_current:
            continue
        if candidate.op_type not in later_op_types:
            continue
        if normalize_entity_name(candidate.entity) != normalized_entity:
            continue
        return True
    return False


def _classify_task_family(
    nodes: list[AffordanceNode],
    scene_facts: SceneFacts,
) -> str:
    """Post-hoc task family classification from assembled nodes."""
    node_kinds = [n.node_kind for n in nodes]
    kind_set = set(node_kinds)
    has_push_lane = any(
        "push_carrier" in (node.required_action_primitives or [])
        or node.node_kind == "push_transport"
        for node in nodes
    )
    has_unload_lane = any(
        node.dispatcher_scaffold == "unload_from_carrier_to_destination"
        for node in nodes
    )

    if kind_set.issubset({"inspect_object", "inspect_presence"}):
        return "inspect_target"
    if "wash_object" in kind_set:
        return "wash_then_relocate"
    if "tool_interaction" in kind_set:
        return "prepare_board_then_tool_interact"
    if "activate_fixture" in kind_set:
        return "place_before_activate_fixture"
    if "open_container" in kind_set:
        return "clear_to_container_with_open"
    if has_push_lane and has_unload_lane:
        return TASK_FAMILY_PUSH_ASSISTED_TRANSPORT

    relocate_nodes = [n for n in nodes if n.node_kind == "relocate_object"]
    if len(relocate_nodes) == 1:
        return "relocate_single_object"
    if len(relocate_nodes) >= 2:
        if any(n.predecessors for n in relocate_nodes):
            return "relocate_objects_sequential"
        return "relocate_objects_parallel"

    return "generalized_from_aig"
