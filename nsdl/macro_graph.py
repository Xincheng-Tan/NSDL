from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nsdl.aig_builder import build_affordance_context
from nsdl.catalog import (
    ROBOT_PRIORITY,
    entity_requires_open_before_place,
    get_end_effector_num,
    get_node_kind_spec,
    get_robot_action_primitives,
    get_robot_capabilities,
    normalize_entity_name,
    normalize_robot_name,
)
from nsdl.schema import (
    AffordanceInteraction,
    AffordanceInteractionGraph,
    AffordanceNode,
    GraphPlan,
    RobotInstance,
    SceneFacts,
)
from nsdl.slm_backends.base import SLMBackend
from nsdl.slm_interface import (
    convert_operations_to_graph_plan,
    generate_validated_operation_dag,
)
from nsdl.task_family import TASK_FAMILY_PUSH_ASSISTED_TRANSPORT


GRAPH_BUILDER_AUTO = "auto"
GRAPH_BUILDER_ACC_DAG = "acc_dag"
PROPOSAL_FAIL_CLOSED = "fail_closed"


@dataclass
class GraphProposal:
    proposal_id: str
    fit_score: float
    positive_evidence: list[str]
    blocking_reasons: list[str]
    graph_plan: GraphPlan


def build_macro_graph(
    scene_facts: SceneFacts,
    *,
    builder_mode: str = GRAPH_BUILDER_ACC_DAG,
    endpoint_index: int = 0,
    api_base_url: str | None = None,
    api_model: str | None = None,
    use_local: bool = False,
    slm_backend: SLMBackend | None = None,
    max_validator_retries: int = 0,
    image_bytes: bytes | None = None,
) -> GraphPlan:
    if builder_mode not in {
        GRAPH_BUILDER_AUTO,
        GRAPH_BUILDER_ACC_DAG,
    }:
        raise ValueError(f"Unsupported graph builder mode: {builder_mode}")

    normalized_mode = _normalize_graph_builder_mode(builder_mode)

    # ── New v2 pipeline: AffordanceContext → SLM → Validator (detect+retry) → GraphPlan ──
    validator_messages: list[str] = []
    validator_attempts = 0
    raw_op_count = 0
    validated_ops = None

    # Step 1: Build Affordance Context (pure Catalog query, no LLM)
    affordance_ctx = build_affordance_context(scene_facts)

    # Step 2: Generate operation DAG via SLM, then run detection-only validator.
    #         Default max_validator_retries=0 (no retry) since 2026-05-22 —
    #         see docs/debug.md §6: the corrective retry has been abandoned
    #         because the remaining error-severity validator gates fire on
    #         too few real SLM samples to overcome the partial-credit
    #         barrier. Set >0 only for ablation studies.
    try:
        candidate_ops, report, validator_attempts = generate_validated_operation_dag(
            scene_facts,
            affordance_ctx,
            endpoint_index=endpoint_index,
            api_base_url=api_base_url,
            api_model=api_model,
            use_local=use_local,
            backend=slm_backend,
            max_validator_retries=max_validator_retries,
            image_bytes=image_bytes,
        )
        raw_op_count = len(candidate_ops)
        validator_messages = report.messages
        if report.ok:
            validated_ops = candidate_ops
    except Exception as exc:  # noqa: BLE001
        validator_messages.append(
            f"SLM pipeline failed during generation/validation: {exc.__class__.__name__}: {exc}"
        )

    if not validated_ops:
        fail_closed = _build_fail_closed_graph_plan()
        fail_closed.build_mode = GRAPH_BUILDER_ACC_DAG
        fail_closed.builder_trace = {
            "requested_builder_mode": builder_mode,
            "effective_builder_mode": normalized_mode,
            "proposal_scores": [],
            "raw_op_count": raw_op_count,
            "validated_op_count": 0,
            "validator_messages": validator_messages,
            "validator_attempts": validator_attempts,
            "max_validator_retries": max_validator_retries,
        }
        return fail_closed

    # Step 4: Convert to GraphPlan
    try:
        graph_plan = convert_operations_to_graph_plan(
            validated_ops,
            affordance_ctx,
            scene_facts,
        )
        graph_plan.build_mode = normalized_mode
        graph_plan.family_shortcut_used = False
        if builder_mode == GRAPH_BUILDER_AUTO:
            graph_plan.compiler_notes.append(
                "Requested builder_mode=auto is treated as an ACC-DAG alias."
            )
        graph_plan.builder_trace = {
            "requested_builder_mode": builder_mode,
            "effective_builder_mode": normalized_mode,
            "proposal_scores": [],
            "raw_op_count": raw_op_count,
            "validated_op_count": len(validated_ops),
            "validator_messages": validator_messages,
            "validator_attempts": validator_attempts,
            "max_validator_retries": max_validator_retries,
        }

        # Step 5: Assign executors
        return _assign_executors_to_graph(graph_plan, scene_facts)

    except Exception as exc:  # noqa: BLE001
        fail_closed = _build_fail_closed_graph_plan()
        fail_closed.build_mode = GRAPH_BUILDER_ACC_DAG
        fail_closed.compiler_notes.append(
            f"SLM pipeline assembly/assignment failed: {exc.__class__.__name__}: {exc}"
        )
        fail_closed.builder_trace = {
            "requested_builder_mode": builder_mode,
            "effective_builder_mode": normalized_mode,
            "proposal_scores": [],
            "raw_op_count": raw_op_count,
            "validated_op_count": len(validated_ops) if validated_ops else 0,
            "validator_messages": [*validator_messages, str(exc)],
            "validator_attempts": validator_attempts,
            "max_validator_retries": max_validator_retries,
        }
        return fail_closed


def _normalize_graph_builder_mode(builder_mode: str) -> str:
    if builder_mode == GRAPH_BUILDER_AUTO:
        return GRAPH_BUILDER_ACC_DAG
    return builder_mode


def _build_operation_proposal_prompt(
    scene_facts: SceneFacts,
    aig: AffordanceInteractionGraph,
    *,
    free_dag_context: bool = False,
) -> list[dict[str, str]]:
    system_content = (
        "You are an NSDL macro operation proposer. "
        "Generate an abstract operation sequence, not benchmark micro actions. "
        "Each op_type must be in operation_vocab. "
        "Each entity and fixture_or_dest should reference an AIG node name when AIG is provided. "
        "Respect preconditions and capability constraints; use predecessor_op_ids to express the DAG. "
        "Return JSON only."
    )
    user_payload: dict[str, Any] = {
        "instruction": scene_facts.instruction,
        "refined_scene_facts": _scene_facts_prompt_payload(scene_facts),
        "operation_vocab": dict(aig.operation_vocab),
        "required_output_schema": {
            "operations": [
                {
                    "op_id": "stable unique id such as op1",
                    "op_type": "one of operation_vocab",
                    "entity": "item/source entity name or null",
                    "fixture_or_dest": "fixture/destination name or null",
                    "predecessor_op_ids": ["op ids that must complete first"],
                    "executor_hint": "optional robot type or capability hint",
                }
            ]
        },
    }
    if free_dag_context:
        user_payload["ablation_mode"] = (
            "free_dag: AIG context is intentionally withheld; infer operations from scene facts only."
        )
        user_payload["robot_capability_matrix"] = _robot_capability_matrix_payload(aig)
    else:
        user_payload["affordance_interaction_graph"] = _aig_to_prompt_payload(aig)
        user_payload["notes"] = [
            "Unknown entities have unknown_tags=true; infer cautiously from the name.",
            "Do not emit benchmark atomic actions such as Move/Reach/Grasp.",
            "Use open before deposit when an AIG containment edge requires open(container).",
            "Use process_at before final deposit when an AIG processing edge produces clean(item).",
        ]
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def _scene_facts_prompt_payload(scene_facts: SceneFacts) -> dict[str, Any]:
    payload = {
        "destination": scene_facts.destination,
        "destination_pairs": _extract_destination_pairs(scene_facts),
        "destination_candidates": scene_facts.destination_candidates,
        "source_locations": scene_facts.source_locations,
        "fallback_locations": scene_facts.fallback_locations,
        "tool_items": scene_facts.tool_items,
        "activation_targets": scene_facts.activation_targets,
        "inspection_targets": scene_facts.inspection_targets,
        "inspect_mode": _inspect_mode(scene_facts),
        "parsed_intents": scene_facts.parsed_intents,
        "mentioned_objects": scene_facts.mentioned_objects,
    }
    if scene_facts.extras:
        payload["extras"] = scene_facts.extras
    return payload


def _aig_to_prompt_payload(aig: AffordanceInteractionGraph) -> dict[str, Any]:
    return {
        "item_nodes": aig.item_nodes,
        "fixture_nodes": aig.fixture_nodes,
        "interactions": [
            _interaction_to_prompt_payload(interaction)
            for interaction in aig.interactions
        ],
        "robot_capability_matrix": _robot_capability_matrix_payload(aig),
        "operation_vocab": dict(aig.operation_vocab),
    }


def _interaction_to_prompt_payload(interaction: AffordanceInteraction) -> dict[str, Any]:
    return {
        "interaction_type": interaction.interaction_type,
        "source_entity": interaction.source_entity,
        "target_entity": interaction.target_entity,
        "triggering_tags": sorted(interaction.triggering_tags),
        "preconditions": list(interaction.preconditions),
        "postconditions": list(interaction.postconditions),
        "required_capabilities": sorted(interaction.required_capabilities),
    }


def _robot_capability_matrix_payload(aig: AffordanceInteractionGraph) -> dict[str, list[str]]:
    return {
        robot_type: sorted(capabilities)
        for robot_type, capabilities in aig.robot_capability_matrix.items()
    }


def _extract_operations_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_ops = payload
    elif isinstance(payload, dict) and isinstance(payload.get("operations"), list):
        raw_ops = payload["operations"]
    else:
        raise ValueError("Expected JSON object with operations list.")
    operations = [op for op in raw_ops if isinstance(op, dict)]
    if not operations:
        raise ValueError("Operation proposal contained no operation objects.")
    return operations


def _normalize_operation_sequence(
    ops: list[dict[str, Any]],
    aig: AffordanceInteractionGraph,
) -> list[dict[str, Any]]:
    allowed_op_types = set(aig.operation_vocab)
    normalized_ops: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_op in enumerate(ops, start=1):
        op_type = str(raw_op.get("op_type", "")).strip().lower()
        if op_type not in allowed_op_types:
            continue
        raw_op_id = str(raw_op.get("op_id", "")).strip() or f"op{index}"
        op_id = raw_op_id
        suffix = 2
        while op_id in used_ids:
            op_id = f"{raw_op_id}_{suffix}"
            suffix += 1
        used_ids.add(op_id)
        predecessor_values = (
            raw_op.get("predecessor_op_ids")
            or raw_op.get("predecessor_ops")
            or raw_op.get("predecessors")
            or []
        )
        if not isinstance(predecessor_values, list):
            predecessor_values = [predecessor_values]
        normalized_ops.append(
            {
                "op_id": op_id,
                "op_type": op_type,
                "entity": _normalize_optional_entity_ref(raw_op.get("entity")),
                "fixture_or_dest": _normalize_optional_entity_ref(raw_op.get("fixture_or_dest")),
                "predecessor_op_ids": [
                    str(value).strip()
                    for value in predecessor_values
                    if str(value).strip()
                ],
                "executor_hint": (
                    str(raw_op.get("executor_hint")).strip()
                    if raw_op.get("executor_hint") is not None
                    else None
                ),
            }
        )
    return normalized_ops


def _normalize_optional_entity_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "n/a"}:
        return None
    return normalize_entity_name(text) or text


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def _parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return json.loads(stripped)


def _build_builder_endpoint_sequence(
    *,
    endpoint_index: int,
    api_base_url: str | None,
    api_model: str | None,
) -> list[int]:
    from nsdl.llm_api import get_available_endpoint_count

    if api_base_url is not None or api_model is not None:
        return [endpoint_index]
    try:
        endpoint_count = get_available_endpoint_count()
    except Exception:
        return [endpoint_index]
    if endpoint_count <= 0:
        return [endpoint_index]
    start_index = endpoint_index % endpoint_count
    return [start_index, *[index for index in range(endpoint_count) if index != start_index]]


def _request_builder_completion_with_retry(
    *,
    messages: list[dict[str, Any]],
    endpoint_index: int,
    api_base_url: str | None,
    api_model: str | None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> tuple[Any, list[dict[str, Any]]]:
    from nsdl.llm_api import create_chat_completion

    request_trace: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for resolved_endpoint_index in _build_builder_endpoint_sequence(
        endpoint_index=endpoint_index,
        api_base_url=api_base_url,
        api_model=api_model,
    ):
        for attempt in range(2):
            try:
                response = create_chat_completion(
                    messages=messages,
                    endpoint_index=resolved_endpoint_index,
                    api_base_url=api_base_url,
                    api_model=api_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                request_trace.append(
                    {
                        "endpoint_index": resolved_endpoint_index,
                        "attempt": attempt + 1,
                        "status": "ok",
                    }
                )
                return response, request_trace
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                request_trace.append(
                    {
                        "endpoint_index": resolved_endpoint_index,
                        "attempt": attempt + 1,
                        "status": "error",
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc)[:240],
                    }
                )
    if last_error is None:
        raise RuntimeError("Macro graph builder request failed without an exception payload.")
    raise last_error


def _build_fail_closed_graph_plan() -> GraphPlan:
    return GraphPlan(
        task_family=PROPOSAL_FAIL_CLOSED,
        build_mode=GRAPH_BUILDER_ACC_DAG,
        family_shortcut_used=False,
        compiler_notes=[
            "Fail-closed proposal emits no executable nodes when no candidate topology is clearly supported.",
        ],
        nodes=[],
    )


def _effective_target_items(scene_facts: SceneFacts) -> list[str]:
    return []


def _count_mobile_manipulators(scene_facts: SceneFacts) -> int:
    count = 0
    for robot in scene_facts.robot_inventory:
        capabilities = get_robot_capabilities(robot.robot_type)
        if {"mobility", "manipulation"}.issubset(capabilities):
            count += 1
    return count


def _has_dual_ee_robot(scene_facts: SceneFacts) -> bool:
    return any(get_end_effector_num(robot.robot_type) >= 2 for robot in scene_facts.robot_inventory)


def _inspect_mode(scene_facts: SceneFacts) -> str | None:
    inspect_mode = scene_facts.inspect_mode
    if isinstance(inspect_mode, str) and inspect_mode.strip():
        return inspect_mode.strip().lower()
    parser_notes_mode = scene_facts.parser_notes.get("inspect_mode")
    if isinstance(parser_notes_mode, str) and parser_notes_mode.strip():
        return parser_notes_mode.strip().lower()
    return None


def _extract_destination_pairs(scene_facts: SceneFacts) -> list[dict[str, str]]:
    raw_pairs = scene_facts.destination_pairs or scene_facts.parser_notes.get("destination_pairs", [])
    if not isinstance(raw_pairs, list):
        return []
    normalized_pairs: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in raw_pairs:
        if not isinstance(pair, dict):
            continue
        item = normalize_entity_name(pair.get("item"))
        target = normalize_entity_name(pair.get("target"))
        if not item or not target:
            continue
        if item == target:
            continue
        pair_key = (item, target)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        normalized_pairs.append({"item": item, "target": target})
    return normalized_pairs


def _resolve_open_before_place_container(scene_facts: SceneFacts) -> str | None:
    pair_targets = {
        pair["target"]
        for pair in _extract_destination_pairs(scene_facts)
        if entity_requires_open_before_place(pair["target"])
    }
    if len(pair_targets) == 1:
        return next(iter(pair_targets))
    if scene_facts.destination and entity_requires_open_before_place(scene_facts.destination):
        return scene_facts.destination
    return None


def _same_destination_pairs(scene_facts: SceneFacts) -> list[dict[str, str]]:
    pairs = _extract_destination_pairs(scene_facts)
    if len(pairs) < 2:
        return []
    unique_targets = {pair["target"] for pair in pairs}
    if len(unique_targets) != 1:
        return []
    return pairs


def _nested_destination_pairs(scene_facts: SceneFacts) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    pairs = _extract_destination_pairs(scene_facts)
    if len(pairs) < 2:
        return None, None
    pair_items = {pair["item"] for pair in pairs}
    base_pair: dict[str, str] | None = None
    fill_pair: dict[str, str] | None = None
    for pair in pairs:
        if pair["target"] in pair_items:
            fill_pair = pair
        else:
            base_pair = pair
    return base_pair, fill_pair


def _activation_pair_layout(
    scene_facts: SceneFacts,
) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    activation_targets = set(scene_facts.activation_targets)
    fixture_pair: dict[str, str] | None = None
    side_pairs: list[dict[str, str]] = []
    for pair in _extract_destination_pairs(scene_facts):
        if pair["target"] in activation_targets and fixture_pair is None:
            fixture_pair = pair
            continue
        side_pairs.append(pair)
    return fixture_pair, side_pairs


def _is_pure_concrete_inspection(scene_facts: SceneFacts) -> bool:
    inspect_mode = _inspect_mode(scene_facts)
    if inspect_mode is not None:
        return inspect_mode == "executable" and bool(
            scene_facts.inspection_targets and scene_facts.destination
        )

    parsed_intents = set(scene_facts.parsed_intents)
    if "inspect" not in parsed_intents:
        return False
    blocking_intents = {
        "relocate",
        "retrieve",
        "conditional_container_check",
        "wash",
        "tool_interaction",
        "activate",
        "push_transfer",
    }
    if parsed_intents.intersection(blocking_intents):
        return False
    if _extract_destination_pairs(scene_facts):
        return False
    return bool(scene_facts.inspection_targets and scene_facts.destination)


def _storage_cue_allowed(scene_facts: SceneFacts) -> bool:
    provenance = scene_facts.parser_notes.get("storage_cue_provenance", {})
    if isinstance(provenance, dict) and provenance:
        return bool(provenance.get("allowed"))
    return False


def _has_explicit_storage_cue(scene_facts: SceneFacts) -> bool:
    if scene_facts.parser_notes.get("storage_cue_provenance"):
        return _storage_cue_allowed(scene_facts)
    storage_fields = (
        scene_facts.source_locations,
        scene_facts.fallback_locations,
        scene_facts.destination_candidates,
        scene_facts.mentioned_objects,
        scene_facts.inspection_targets,
    )
    return any("cabinet" in field_values for field_values in storage_fields)


def _extract_cabinet_backed_items(
    scene_facts: SceneFacts,
    target_items: list[str],
) -> list[str]:
    cabinet_items: list[str] = []
    for item in target_items:
        if item in scene_facts.targets_in_cabinet:
            cabinet_items.append(item)
    return cabinet_items


def _build_node(
    *,
    node_id: str,
    node_kind: str,
    semantic_goal: str,
    object_name: str | None,
    destination: str | None,
    object_bindings: dict[str, str] | None = None,
    required_action_primitives: list[str] | None = None,
    required_constraints: dict[str, object] | None = None,
    predecessors: list[str] | None = None,
    dispatcher_scaffold: str | None = None,
) -> AffordanceNode:
    spec = get_node_kind_spec(node_kind)
    goal_predicates, start_predicates_hint, state_exports = _derive_micro_planner_contract(
        node_kind=node_kind,
        object_name=object_name,
        destination=destination,
        object_bindings=dict(object_bindings or {}),
        required_constraints=dict(required_constraints or {}),
        dispatcher_scaffold=dispatcher_scaffold,
    )

    has_item_at_goal = any(predicate.startswith("item_at(") for predicate in goal_predicates)

    return AffordanceNode(
        node_id=node_id,
        node_kind=spec.node_kind,
        semantic_goal=semantic_goal,
        object_name=object_name,
        destination=destination,
        object_bindings=dict(object_bindings or {}),
        required_affordances_hard=list(spec.required_affordances_hard),
        required_affordances_soft=list(spec.required_affordances_soft),
        required_action_primitives=list(required_action_primitives or []),
        required_constraints=dict(required_constraints or {}),
        predecessors=list(predecessors or []),
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
    required_constraints: dict[str, object],
    dispatcher_scaffold: str | None,
) -> tuple[list[str], list[str], list[str]]:
    scaffold = dispatcher_scaffold or node_kind
    item = object_bindings.get("item", object_name)
    target_area = object_bindings.get("target_area", destination)
    container = object_bindings.get("container")
    station = object_bindings.get("station", destination)
    target = object_bindings.get("target", object_name)
    tool = object_name or object_bindings.get("item")
    item_primary = object_bindings.get("item_primary")
    item_secondary = object_bindings.get("item_secondary")

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

    if scaffold == "pickup_and_place":
        add_goal("item_at", item, target_area)
        add_export("item_at", item, target_area)
        add_export("executor_at", target_area)
    elif scaffold == "open_then_fetch":
        add_goal("item_at", item, target_area)
        add_start("inside", item, container)
        add_export("item_at", item, target_area)
        add_export("executor_at", target_area)
        add_export("is_open", container)
    elif scaffold == "carry_to_container_anchor":
        add_goal("holding", item)
        add_goal("executor_at", container)
        add_export("holding", item)
        add_export("executor_at", container)
    elif scaffold in {"open_container_in_place", "move_reach_open_container"}:
        add_goal("is_open", container or object_name)
        add_export("is_open", container or object_name)
        add_export("executor_at", container or object_name)
    elif scaffold == "place_held_item":
        add_goal("item_at", item, target_area)
        add_start("holding", item)
        add_start("in_local_workspace", target_area)
        add_export("item_at", item, target_area)
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
    elif scaffold in {"tool_grasp", "pickup_only"}:
        add_goal("holding", item or tool)
        add_export("holding", item or tool)
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
    elif scaffold == "reach_with_held_tool":
        add_goal("reaching", target_area or target)
        add_start("holding", tool)
        add_export("holding", tool)
        add_export("reaching", target_area or target)
        add_export("executor_at", target_area or target)
    elif scaffold == "tool_interaction_after_acquire":
        add_goal("tool_effect_applied", tool, destination)
        add_start("holding", tool)
        add_start("executor_at", tool)
        add_export("tool_effect_applied", tool, destination)
        add_export("executor_at", destination)
    elif scaffold == "dual_pick_then_place":
        add_goal("batched_dropoff", item_primary, item_secondary, target_area)
        add_export("item_at", item_primary, target_area)
        add_export("item_at", item_secondary, target_area)
        add_export("executor_at", target_area)
    elif scaffold == "dual_pick_objects":
        add_goal("holding_pair", item_primary, item_secondary)
        add_export("holding", item_primary)
        add_export("holding", item_secondary)
        add_export("holding_pair", item_primary, item_secondary)
    elif scaffold == "place_dual_pick_at_target":
        add_goal("batched_dropoff", item_primary, item_secondary, target_area)
        add_start("holding", item_primary)
        add_start("holding", item_secondary)
        add_start("holding_pair", item_primary, item_secondary)
        add_export("item_at", item_primary, target_area)
        add_export("item_at", item_secondary, target_area)
        add_export("executor_at", target_area)
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
    elif scaffold == "handoff_anchor":
        add_goal("handoff", item, target_area)
        add_start("handoff", item, target_area)
        add_export("handoff", item, target_area)
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


def _build_relocate_node(
    *,
    node_id: str,
    object_name: str | None,
    destination: str | None,
    semantic_goal: str,
    predecessors: list[str] | None = None,
    required_constraints: dict[str, object] | None = None,
) -> AffordanceNode:
    object_bindings = {}
    if object_name:
        object_bindings["item"] = object_name
    if destination:
        object_bindings["target_area"] = destination
    return _build_node(
        node_id=node_id,
        node_kind="relocate_object",
        semantic_goal=semantic_goal,
        object_name=object_name,
        destination=destination,
        object_bindings=object_bindings,
        required_action_primitives=["move_to", "reach_object", "grasp_item", "place_item"],
        required_constraints=required_constraints,
        predecessors=predecessors,
        dispatcher_scaffold="pickup_and_place",
    )


def _build_retrieve_from_cabinet_node(
    *,
    node_id: str,
    object_name: str | None,
    destination: str | None,
    predecessors: list[str] | None = None,
    required_constraints: dict[str, object] | None = None,
) -> AffordanceNode:
    return _build_node(
        node_id=node_id,
        node_kind="retrieve_from_container",
        semantic_goal=f"Retrieve {object_name} from cabinet and relocate it to {destination}",
        object_name=object_name,
        destination=destination,
        object_bindings={
            "container": "cabinet",
            "item": object_name,
            "target_area": destination,
        },
        required_action_primitives=[
            "move_to",
            "reach_object",
            "open_container",
            "grasp_item",
            "place_item",
        ],
        required_constraints={
            "source_container": "cabinet",
            **dict(required_constraints or {}),
        },
        predecessors=predecessors,
        dispatcher_scaffold="open_then_fetch",
    )


def _build_inspect_object_node(
    *,
    node_id: str,
    target: str,
    predecessors: list[str] | None = None,
    required_constraints: dict[str, object] | None = None,
) -> AffordanceNode:
    return _build_node(
        node_id=node_id,
        node_kind="inspect_object",
        semantic_goal=f"Inspect {target} directly at its location",
        object_name=target,
        destination=target,
        object_bindings={"target": target},
        required_action_primitives=["move_to", "interact_with"],
        required_constraints=required_constraints,
        predecessors=predecessors,
        dispatcher_scaffold="move_then_interact",
    )


def collect_selected_robot_types(graph_plan: GraphPlan) -> list[str]:
    selected_by_slot: dict[str, str] = {}
    for node in graph_plan.nodes:
        if node.executor_slot and node.executor_robot_type and node.executor_slot not in selected_by_slot:
            selected_by_slot[node.executor_slot] = node.executor_robot_type
    return [
        selected_by_slot[slot_id]
        for slot_id in sorted(selected_by_slot, key=_slot_sort_key)
    ]


def _slot_sort_key(slot_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in slot_id if ch.isdigit())
    return (int(digits) if digits else 10**9, slot_id)


def _extract_detection_instances(
    scene_facts: SceneFacts,
    *,
    label: str,
) -> list[dict[str, object]]:
    yolo_payload = scene_facts.parser_notes.get("yolo_detections", {})
    if not isinstance(yolo_payload, dict):
        return []
    grouped_detections = yolo_payload.get("grouped_detections", [])
    if not isinstance(grouped_detections, list):
        return []
    for group in grouped_detections:
        if not isinstance(group, dict):
            continue
        normalized_label = normalize_robot_name(group.get("label")) or normalize_entity_name(
            group.get("label")
        )
        if normalized_label != label:
            continue
        instances = group.get("instances", [])
        if not isinstance(instances, list):
            return []
        return [
            instance
            for instance in instances
            if isinstance(instance, dict)
        ]
    return []


def _center_xy(instance: dict[str, object]) -> tuple[float, float] | None:
    center_xy = instance.get("center_xy")
    if not isinstance(center_xy, list) or len(center_xy) != 2:
        return None
    try:
        return float(center_xy[0]), float(center_xy[1])
    except (TypeError, ValueError):
        return None


def _bbox_xyxy(instance: dict[str, object]) -> tuple[float, float, float, float] | None:
    bbox_xyxy = instance.get("bbox_xyxy")
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        return None
    try:
        return (
            float(bbox_xyxy[0]),
            float(bbox_xyxy[1]),
            float(bbox_xyxy[2]),
            float(bbox_xyxy[3]),
        )
    except (TypeError, ValueError):
        return None


def _point_in_bbox(
    center_xy: tuple[float, float] | None,
    bbox_xyxy: tuple[float, float, float, float] | None,
) -> bool:
    if center_xy is None or bbox_xyxy is None:
        return False
    x, y = center_xy
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_fully_contained(
    inner_bbox: tuple[float, float, float, float] | None,
    outer_bbox: tuple[float, float, float, float] | None,
) -> bool:
    if inner_bbox is None or outer_bbox is None:
        return False
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return ox1 <= ix1 <= ix2 <= ox2 and oy1 <= iy1 <= iy2 <= oy2


def _point_to_bbox_l1_distance(
    center_xy: tuple[float, float] | None,
    bbox_xyxy: tuple[float, float, float, float] | None,
) -> float:
    if center_xy is None or bbox_xyxy is None:
        return float("inf")
    if _point_in_bbox(center_xy, bbox_xyxy):
        return 0.0
    x, y = center_xy
    x1, y1, x2, y2 = bbox_xyxy
    dx = 0.0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0.0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    return dx + dy


def _infer_push_transfer_support_geometry(scene_facts: SceneFacts) -> dict[str, object] | None:
    table_instances = _extract_detection_instances(scene_facts, label="table")
    sink_instances = _extract_detection_instances(scene_facts, label="sink")
    panda_instances = _extract_detection_instances(scene_facts, label="panda")
    if len(table_instances) != 1 or len(sink_instances) != 1 or len(panda_instances) != 2:
        return None

    table_bbox = _bbox_xyxy(table_instances[0])
    sink_bbox = _bbox_xyxy(sink_instances[0])
    sink_center = _center_xy(sink_instances[0])
    if table_bbox is None or sink_bbox is None or sink_center is None:
        return None

    sink_on_table = _bbox_fully_contained(sink_bbox, table_bbox) or _point_in_bbox(
        sink_center,
        table_bbox,
    )
    ranked_pandas: list[dict[str, object]] = []
    for panda_instance in panda_instances:
        panda_center = _center_xy(panda_instance)
        ranked_pandas.append(
            {
                "rank_within_label": panda_instance.get("rank_within_label", 1),
                "center_xy": list(panda_instance.get("center_xy", [])),
                "table_distance_l1": round(
                    _point_to_bbox_l1_distance(panda_center, table_bbox),
                    3,
                ),
            }
        )
    ranked_pandas.sort(
        key=lambda item: (
            float(item.get("table_distance_l1", float("inf"))),
            int(item.get("rank_within_label", 1)),
        )
    )
    return {
        "sink_support_relation": "on_table" if sink_on_table else "off_table",
        "panda_table_distance_ranking": ranked_pandas,
    }


def _graph_has_push_transfer_roles(graph_plan: GraphPlan) -> bool:
    return any(
        node.required_constraints.get("push_transfer_role")
        in {"source_holder", "receiver_holder", "carrier_pusher"}
        for node in graph_plan.nodes
    )


def _resolve_push_transfer_slots(
    graph_plan: GraphPlan,
    scene_facts: SceneFacts,
) -> tuple[str, str, dict[str, object] | None] | None:
    source_node = next(
        (
            node
            for node in graph_plan.nodes
            if node.required_constraints.get("push_transfer_role") == "source_holder"
        ),
        None,
    )
    receiver_node = next(
        (
            node
            for node in graph_plan.nodes
            if node.required_constraints.get("push_transfer_role") == "receiver_holder"
        ),
        None,
    )
    pusher_node = next(
        (
            node
            for node in graph_plan.nodes
            if node.required_constraints.get("push_transfer_role") == "carrier_pusher"
        ),
        None,
    )
    if source_node is None or receiver_node is None or pusher_node is None:
        return None

    def _resolve_robot_type(node: AffordanceNode) -> str | None:
        if isinstance(node.executor_robot_type, str) and node.executor_robot_type:
            return node.executor_robot_type
        candidate_type = node.required_constraints.get("executor_robot_type")
        if isinstance(candidate_type, str) and candidate_type:
            return candidate_type
        if isinstance(node.executor_slot, str) and node.executor_slot:
            robot = next(
                (
                    item
                    for item in scene_facts.robot_inventory
                    if item.slot_id == node.executor_slot
                ),
                None,
            )
            if robot is not None:
                return robot.robot_type
        return None

    source_robot_type = _resolve_robot_type(source_node)
    if not isinstance(source_robot_type, str) or not source_robot_type:
        return None

    same_type_slots = sorted(
        [
            robot.slot_id
            for robot in scene_facts.robot_inventory
            if robot.robot_type == source_robot_type
        ],
        key=_slot_sort_key,
    )
    if len(same_type_slots) != 2:
        return None

    anchor_status = {
        str(slot_id).strip(): str(status).strip()
        for slot_id, status in scene_facts.robot_anchor_status.items()
        if str(slot_id).strip() and str(status).strip()
    }
    if anchor_status:
        on_sink_slots = [
            slot_id
            for slot_id in same_type_slots
            if anchor_status.get(slot_id) == "on_sink_support_surface"
        ]
        off_sink_slots = [
            slot_id
            for slot_id in same_type_slots
            if anchor_status.get(slot_id) == "off_sink_support_surface"
        ]
        if len(on_sink_slots) == 1 and len(off_sink_slots) == 1:
            geometry_note = (
                scene_facts.parser_notes.get("robot_anchor_geometry")
                if isinstance(scene_facts.parser_notes.get("robot_anchor_geometry"), dict)
                else None
            )
            return off_sink_slots[0], on_sink_slots[0], geometry_note

    pusher_robot_type = _resolve_robot_type(pusher_node)
    if not isinstance(pusher_robot_type, str) or not pusher_robot_type:
        return None
    pusher_slot = next(
        (
            robot.slot_id
            for robot in scene_facts.robot_inventory
            if robot.robot_type == pusher_robot_type
        ),
        None,
    )
    if not pusher_slot:
        return None

    ordered_slots = sorted(
        [robot.slot_id for robot in scene_facts.robot_inventory],
        key=_slot_sort_key,
    )
    if pusher_slot not in ordered_slots:
        return None
    anchor_index = ordered_slots.index(pusher_slot)
    rotated_slots = ordered_slots[anchor_index + 1 :] + ordered_slots[: anchor_index + 1]
    ranked_slots = [slot_id for slot_id in rotated_slots if slot_id in same_type_slots]
    if len(ranked_slots) != 2:
        return None

    source_slot = ranked_slots[0]
    receiver_slot = ranked_slots[1]
    support_geometry = None
    if source_robot_type == "panda":
        support_geometry = _infer_push_transfer_support_geometry(scene_facts)
        if support_geometry:
            table_side_slot = ranked_slots[0]
            off_table_slot = ranked_slots[1]
            sink_support_relation = str(
                support_geometry.get("sink_support_relation", "off_table")
            ).strip()
            if sink_support_relation == "on_table":
                source_slot = off_table_slot
                receiver_slot = table_side_slot
            else:
                source_slot = table_side_slot
                receiver_slot = off_table_slot
    return source_slot, receiver_slot, support_geometry


def _preassign_push_transfer_roles(
    graph_plan: GraphPlan,
    scene_facts: SceneFacts,
) -> list[str]:
    if (
        graph_plan.task_family != TASK_FAMILY_PUSH_ASSISTED_TRANSPORT
        and not _graph_has_push_transfer_roles(graph_plan)
    ):
        return []

    source_node = next(
        (
            node
            for node in graph_plan.nodes
            if node.required_constraints.get("push_transfer_role") == "source_holder"
        ),
        None,
    )
    receiver_node = next(
        (
            node
            for node in graph_plan.nodes
            if node.required_constraints.get("push_transfer_role") == "receiver_holder"
        ),
        None,
    )
    pusher_nodes = [
        node
        for node in graph_plan.nodes
        if node.required_constraints.get("push_transfer_role") == "carrier_pusher"
    ]
    if source_node is None or receiver_node is None or not pusher_nodes:
        return []

    slot_resolution = _resolve_push_transfer_slots(graph_plan, scene_facts)
    if slot_resolution is None:
        return []
    source_slot, receiver_slot, support_geometry = slot_resolution

    source_node.executor_slot = source_slot
    receiver_node.executor_slot = receiver_slot
    for pusher_node in pusher_nodes:
        handoff_slot = None
        phase = str(pusher_node.required_constraints.get("push_transfer_phase", "")).strip()
        if phase == "to_source_holder":
            handoff_slot = source_slot
        elif phase == "to_receiver_holder":
            handoff_slot = receiver_slot
        if handoff_slot:
            pusher_node.required_constraints["handoff_target_slot"] = handoff_slot
            if pusher_node.destination:
                pusher_node.destination = handoff_slot
            carrier = pusher_node.object_bindings.get("carrier", pusher_node.object_name)
            if carrier:
                pusher_node.semantic_goal = f"Push {carrier} to {handoff_slot}"

    if support_geometry:
        return [
            _format_push_transfer_support_geometry_note(
                support_geometry,
                source_slot=source_slot,
                receiver_slot=receiver_slot,
            ),
        ]
    return [
        "Preassigned push-transfer source/receiver slots from slot-level robot anchor status or inventory order: "
        f"source_holder={source_slot}, receiver_holder={receiver_slot}.",
    ]


def _format_push_transfer_support_geometry_note(
    support_geometry: dict[str, object],
    *,
    source_slot: str,
    receiver_slot: str,
) -> str:
    details: list[str] = []

    slot_grounding_source = support_geometry.get("slot_grounding_source")
    if slot_grounding_source:
        details.append(f"slot_grounding_source={slot_grounding_source}")

    sink_support_relation = support_geometry.get("sink_support_relation")
    if sink_support_relation:
        details.append(f"sink_support_relation={sink_support_relation}")

    sink_holder_slot = support_geometry.get("sink_holder_slot")
    if sink_holder_slot:
        details.append(f"sink_holder_slot={sink_holder_slot}")

    details.extend([
        f"source_holder={source_slot}",
        f"receiver_holder={receiver_slot}",
    ])

    panda_ranking = support_geometry.get("panda_table_distance_ranking")
    if panda_ranking is not None:
        details.append(f"panda_table_distance_ranking={panda_ranking}")

    return (
        "Preassigned push-transfer source/receiver slots with support geometry "
        f"or anchor status: {', '.join(details)}."
    )


def _assign_executors_to_graph(graph_plan: GraphPlan, scene_facts: SceneFacts) -> GraphPlan:
    graph_plan.compiler_notes.extend(_preassign_push_transfer_roles(graph_plan, scene_facts))
    active_slots: list[str] = []
    assigned_by_node_id: dict[str, RobotInstance] = {}
    assignment_notes: list[str] = []

    for node in graph_plan.nodes:
        if "executor_robot_type" in node.required_constraints and not node.executor_robot_type:
            node.executor_robot_type = str(node.required_constraints["executor_robot_type"])
        robot, assignment_origin = _resolve_executor_for_node(
            node=node,
            robot_inventory=scene_facts.robot_inventory,
            active_slots=active_slots,
            assigned_by_node_id=assigned_by_node_id,
        )
        node.executor_slot = robot.slot_id
        node.executor_robot_type = robot.robot_type
        assigned_by_node_id[node.node_id] = robot
        if robot.slot_id not in active_slots:
            active_slots.append(robot.slot_id)
        assignment_notes.append(
            f"Assigned {node.node_id} to {robot.slot_id}:{robot.robot_type} ({assignment_origin})."
        )

    graph_plan.compiler_notes.extend(assignment_notes)
    return graph_plan


def _resolve_executor_for_node(
    *,
    node: AffordanceNode,
    robot_inventory: list[RobotInstance],
    active_slots: list[str],
    assigned_by_node_id: dict[str, RobotInstance],
) -> tuple[RobotInstance, str]:
    same_executor_as = node.required_constraints.get("same_executor_as")
    if same_executor_as:
        if same_executor_as not in assigned_by_node_id:
            raise ValueError(
                f"Node {node.node_id} requested same_executor_as={same_executor_as}, "
                "but that node has not been assigned yet."
            )
        robot = assigned_by_node_id[same_executor_as]
        _validate_robot_for_node(robot, node)
        return robot, f"same_executor_as {same_executor_as}"

    if node.executor_slot:
        robot = next((item for item in robot_inventory if item.slot_id == node.executor_slot), None)
        if robot is None:
            raise ValueError(
                f"Node {node.node_id} requested unknown executor_slot={node.executor_slot}."
            )
        _validate_robot_for_node(robot, node)
        if node.executor_robot_type and node.executor_robot_type != robot.robot_type:
            raise ValueError(
                f"Node {node.node_id} requested executor_robot_type={node.executor_robot_type}, "
                f"but slot {robot.slot_id} is {robot.robot_type}."
            )
        return robot, "preassigned slot"

    candidate_pool = list(robot_inventory)
    if node.executor_robot_type:
        candidate_pool = [
            robot for robot in candidate_pool if robot.robot_type == node.executor_robot_type
        ]
        if not candidate_pool:
            raise ValueError(
                f"Node {node.node_id} requested unavailable robot_type={node.executor_robot_type}."
            )

    candidates = []
    prefer_new_slot = bool(node.required_constraints.get("prefer_new_slot", False))
    for robot in candidate_pool:
        if not _robot_satisfies_node(robot, node):
            continue
        capabilities = get_robot_capabilities(robot.robot_type)
        soft_match_score = len(set(node.required_affordances_soft).intersection(capabilities))
        if prefer_new_slot:
            reuse_rank = 0 if robot.slot_id not in active_slots else 1
        else:
            reuse_rank = 0 if robot.slot_id in active_slots else 1
        candidates.append((reuse_rank, soft_match_score, robot))

    if not candidates:
        raise ValueError(
            f"No available robot can satisfy node={node.node_id} ({node.semantic_goal})."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1],
            ROBOT_PRIORITY.get(item[2].robot_type, 999),
            item[2].slot_id,
        )
    )
    return candidates[0][2], "catalog-checked assignment"


def _validate_robot_for_node(robot: RobotInstance, node: AffordanceNode) -> None:
    if not _robot_satisfies_node(robot, node):
        raise ValueError(
            f"Assigned robot {robot.slot_id}:{robot.robot_type} cannot execute node={node.node_id}."
        )


def _robot_satisfies_node(robot: RobotInstance, node: AffordanceNode) -> bool:
    capabilities = get_robot_capabilities(robot.robot_type)
    if not set(node.required_affordances_hard).issubset(capabilities):
        return False

    action_primitives = set(get_robot_action_primitives(robot.robot_type))
    if not set(node.required_action_primitives).issubset(action_primitives):
        return False

    min_end_effectors = int(node.required_constraints.get("min_end_effectors", 0))
    if get_end_effector_num(robot.robot_type) < min_end_effectors:
        return False

    return True
