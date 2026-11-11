"""Deterministic planning-tool facade for L-NSDL.

This module exposes the tool-shaped Φ queries used by the new agentic path.
It intentionally leaves the legacy ``dispatcher.dispatch_graph`` entry point
untouched so existing deterministic-pipeline ablations keep their old behavior.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import os
import re
from typing import Any

from nsdl.aig_builder import build_affordance_context
from nsdl.catalog import (
    CAP_MOBILITY,
    CAP_PUSH_TRANSFER,
    PUSHABLE_CARRIER_TAG,
    ROBOT_PRIORITY,
    entity_can_be_destination,
    entity_requires_open_before_place,
    normalize_entity_name,
)
from nsdl.micro_planner import plan_node_micro_program
from nsdl.schema import (
    AffordanceContext,
    AffordanceNode,
    GraphPlan,
    MicroStep,
    OperationStep,
    RobotInstance,
    SceneFacts,
)
from nsdl.slm_interface import convert_operations_to_graph_plan, validate_operations


@dataclass
class ToolReason:
    ok: bool
    code: str | None = None
    op_ids: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class ExpandResult:
    primitives: list[list[str]] = field(default_factory=list)
    end_state: list[str] = field(default_factory=list)
    ok: bool = True
    reason: ToolReason | None = None


@dataclass
class ScheduleResult:
    ops: list[OperationStep] = field(default_factory=list)
    schedule: list[MicroStep] = field(default_factory=list)
    inserted: list[str] = field(default_factory=list)
    ok: bool = True
    reason: ToolReason | None = None
    graph_plan: GraphPlan | None = None


class _PlanningToolError(Exception):
    def __init__(self, reason: ToolReason) -> None:
        super().__init__(reason.message or reason.code or "planning_tool_error")
        self.reason = reason


_CUSTODY_SOURCE_OPS = frozenset({
    "acquire",
    "local_acquire",
    "handoff",
    "unload_carrier",
    "process_at",
})
_DEPOSIT_SOURCE_OPS = _CUSTODY_SOURCE_OPS | frozenset({"carry_to"})
_STATE_CHANGE_NODE_KINDS = frozenset({
    "open_container",
    "close_container",
    "push_transport",
    "handoff_node",
})


def schedule(
    ops: list[OperationStep],
    assign: dict[str, Any] | None = None,
    ctx: AffordanceContext | dict[str, Any] | None = None,
    scene_facts: SceneFacts | None = None,
) -> ScheduleResult:
    """Complete an intent sketch and run deterministic resource scheduling."""
    try:
        scene = _coerce_scene_facts(scene_facts, ctx)
        affordance_context = _coerce_affordance_context(ctx, scene)
        completed_ops, inserted = _complete_operations(
            ops,
            assign=assign or {},
            ctx=affordance_context,
        )
        validation = validate_operations(completed_ops, affordance_context, scene)
        if not validation.ok:
            return ScheduleResult(
                ops=completed_ops,
                inserted=inserted,
                ok=False,
                reason=_validation_reason(validation.errors[0]),
            )

        ordered_ops = _topological_order_ops(completed_ops)
        graph_plan = convert_operations_to_graph_plan(
            ordered_ops,
            affordance_context,
            scene,
        )
        _attach_executor_types(graph_plan, affordance_context)
        node_to_op = _node_to_op_ids(graph_plan, ordered_ops)
        node_programs, node_end_states = _expand_graph_nodes(graph_plan, node_to_op)
        micro_steps = _select_list_scheduler()(
            graph_plan=graph_plan,
            node_programs=node_programs,
            node_to_op=node_to_op,
            op_by_id={op.op_id: op for op in ordered_ops},
            ctx=affordance_context,
        )
        graph_plan.builder_trace = {
            **(graph_plan.builder_trace or {}),
            "planning_tool": "schedule",
            "node_end_states": {
                node_id: sorted(state)
                for node_id, state in node_end_states.items()
            },
        }
        return ScheduleResult(
            ops=ordered_ops,
            schedule=micro_steps,
            inserted=inserted,
            ok=True,
            graph_plan=graph_plan,
        )
    except _PlanningToolError as exc:
        return ScheduleResult(
            ops=list(ops),
            inserted=[],
            ok=False,
            reason=exc.reason,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed with structured reason.
        return ScheduleResult(
            ops=list(ops),
            inserted=[],
            ok=False,
            reason=ToolReason(
                ok=False,
                code="unschedulable",
                message=f"{exc.__class__.__name__}: {exc}",
            ),
        )


def verify(
    ops: list[OperationStep],
    ctx: AffordanceContext | dict[str, Any] | None = None,
    scene_facts: SceneFacts | None = None,
) -> ToolReason:
    """Return a Φ satisfiability diagnosis for an operation sketch or DAG."""
    result = schedule(ops, assign={}, ctx=ctx, scene_facts=scene_facts)
    if result.ok:
        return ToolReason(ok=True, message="ok")
    return result.reason or ToolReason(
        ok=False,
        code="unschedulable",
        message="schedule failed without a structured reason",
    )


def verify_goal_coverage(
    model_goal: dict[str, Any],
    observed_facts: dict[str, Any] | list[str] | tuple[str, ...] | str | None,
) -> ToolReason:
    """Diagnose a model-emitted GOAL block against observed relations only.

    This tool never reads GT or instruction parser state. The destination role
    comes solely from ``model_goal``; ``observed_facts`` is expected to be a
    detect-style payload such as ``{"relations": ["plate on table"]}``.
    It reports contradictions but never rewrites the goal.
    """
    if not isinstance(model_goal, dict):
        return ToolReason(
            ok=False,
            code="invalid_goal",
            message="model_goal must be a JSON object emitted by the model.",
        )

    active = {_canonical_name(item) for item in _list_values(model_goal.get("active"))}
    skip = {_canonical_name(item) for item in _list_values(model_goal.get("skip"))}
    active.discard(None)
    skip.discard(None)
    observed_relations = _observed_relations(observed_facts)
    observed_pairs = {
        (_canonical_name(entity), _canonical_name(fixture), verb, relation_text)
        for relation_text, entity, verb, fixture in observed_relations
        if verb in {"on", "in", "at"}
    }

    target_items: set[str] = set()
    conflicts: list[str] = []
    for target in model_goal.get("targets") or []:
        if not isinstance(target, dict):
            continue
        item = _canonical_name(target.get("item"))
        dest = _canonical_name(target.get("destination"))
        status = str(target.get("status") or "").strip().lower()
        if not item:
            continue
        target_items.add(item)
        if status == "active" and item not in active:
            conflicts.append(f"{item}: status active but item is absent from active list")
        if status == "skip" and item not in skip:
            conflicts.append(f"{item}: status skip but item is absent from skip list")
        if item in active and item in skip:
            conflicts.append(f"{item}: item appears in both active and skip lists")
        if item in active and dest:
            matched = [
                relation_text
                for entity, fixture, _verb, relation_text in observed_pairs
                if entity == item and fixture == dest
            ]
            if matched:
                conflicts.append(
                    f"{item}: active target conflicts with observed coverage "
                    f"for declared destination {dest}: {matched[0]}"
                )

    unknown_active = sorted(item for item in active if item and item not in target_items)
    unknown_skip = sorted(item for item in skip if item and item not in target_items)
    if unknown_active:
        conflicts.append(f"active list contains item(s) missing from targets: {unknown_active}")
    if unknown_skip:
        conflicts.append(f"skip list contains item(s) missing from targets: {unknown_skip}")

    if conflicts:
        return ToolReason(
            ok=False,
            code="goal_coverage_conflict",
            message="; ".join(conflicts),
        )
    return ToolReason(
        ok=True,
        code=None,
        message="goal_coverage_ok: no active target is contradicted by the observed relations.",
    )


def expand(
    op_or_node: OperationStep | AffordanceNode,
    ctx_state: dict[str, Any] | None = None,
) -> ExpandResult:
    """Lower one op/node into primitive benchmark actions via micro_planner."""
    state = ctx_state or {}
    try:
        if isinstance(op_or_node, AffordanceNode):
            node = op_or_node
            node_map = dict(state.get("node_map") or {node.node_id: node})
            predecessor_end_states = dict(state.get("predecessor_end_states") or {})
            if not node.executor_robot_type:
                ctx = state.get("ctx") or state.get("affordance_context")
                if isinstance(ctx, AffordanceContext):
                    _attach_executor_types(
                        GraphPlan(
                            task_family=None,
                            build_mode="planning_tool_expand",
                            family_shortcut_used=False,
                            nodes=[node],
                        ),
                        ctx,
                    )
        else:
            scene = _coerce_scene_facts(state.get("scene_facts"), state.get("ctx"))
            ctx = _coerce_affordance_context(state.get("ctx"), scene)
            graph_plan = convert_operations_to_graph_plan([op_or_node], ctx, scene)
            _attach_executor_types(graph_plan, ctx)
            node = graph_plan.nodes[0]
            node_map = {candidate.node_id: candidate for candidate in graph_plan.nodes}
            predecessor_end_states = {}

        planned = plan_node_micro_program(
            node=node,
            node_map=node_map,
            predecessor_end_states=predecessor_end_states,
        )
        return ExpandResult(
            primitives=[list(action) for action in planned.actions],
            end_state=sorted(planned.final_state),
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        return ExpandResult(
            ok=False,
            reason=ToolReason(
                ok=False,
                code="goal_unreachable",
                message=f"{exc.__class__.__name__}: {exc}",
            ),
        )


_RELATION_RE = re.compile(r"^(.+?)\s+(on|in|at|near)\s+(.+)$")


def _observed_relations(
    observed_facts: dict[str, Any] | list[str] | tuple[str, ...] | str | None,
) -> list[tuple[str, str, str, str]]:
    if observed_facts is None:
        raw_relations: list[Any] = []
    elif isinstance(observed_facts, dict):
        raw_relations = list(observed_facts.get("relations") or [])
    elif isinstance(observed_facts, str):
        raw_relations = [observed_facts]
    else:
        raw_relations = list(observed_facts)
    parsed: list[tuple[str, str, str, str]] = []
    for relation in raw_relations:
        text = str(relation or "").strip().lower()
        match = _RELATION_RE.match(text)
        if not match:
            continue
        parsed.append((text, match.group(1), match.group(2), match.group(3)))
    return parsed


def _list_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if value is None:
        return []
    return [value]


def _canonical_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_entity_name(str(value))
    return normalized.strip().lower() if normalized else None


def _coerce_scene_facts(
    scene_facts: SceneFacts | None,
    ctx: AffordanceContext | dict[str, Any] | None,
) -> SceneFacts:
    if scene_facts is not None:
        return scene_facts
    if isinstance(ctx, dict) and isinstance(ctx.get("scene_facts"), SceneFacts):
        return ctx["scene_facts"]
    if isinstance(ctx, dict) and isinstance(ctx.get("affordance_context"), AffordanceContext):
        ctx = ctx["affordance_context"]
    if isinstance(ctx, AffordanceContext):
        return SceneFacts(
            parser_backend="planning_tool",
            instruction="",
            image_available=False,
            robot_inventory=[
                RobotInstance(slot_id=entry.slot_id, robot_type=entry.robot_type)
                for entry in ctx.robot_capabilities
            ],
            primary_object=None,
            destination=None,
        )
    raise _PlanningToolError(ToolReason(
        ok=False,
        code="invalid_input",
        message="planning_tool requires scene_facts or an AffordanceContext with robots.",
    ))


def _coerce_affordance_context(
    ctx: AffordanceContext | dict[str, Any] | None,
    scene_facts: SceneFacts,
) -> AffordanceContext:
    if isinstance(ctx, AffordanceContext):
        return ctx
    if isinstance(ctx, dict):
        value = ctx.get("affordance_context") or ctx.get("ctx")
        if isinstance(value, AffordanceContext):
            return value
    return build_affordance_context(scene_facts)


def _complete_operations(
    ops: list[OperationStep],
    *,
    assign: dict[str, Any],
    ctx: AffordanceContext,
) -> tuple[list[OperationStep], list[str]]:
    inserted: list[str] = []
    base_ops = [
        _clone_op(op, robot=_op_robot(op, assign))
        for op in ops
    ]
    base_ops, open_insertions = _insert_missing_open_ops(base_ops, ctx)
    inserted.extend(open_insertions)

    _reject_cross_robot_acquire_conflicts(base_ops)

    result: list[OperationStep] = []
    used_ids: set[str] = {op.op_id for op in base_ops}
    for op in base_ops:
        completed_op = _clone_op(op)
        deps = list(completed_op.deps)

        if completed_op.op_type in {"acquire", "local_acquire"}:
            prior_same_entity = _latest_same_entity_op(completed_op, result)
            if prior_same_entity:
                _append_unique(deps, prior_same_entity.op_id)
                inserted.append(f"deps {prior_same_entity.op_id}->{completed_op.op_id}")

        if completed_op.op_type in {"carry_to", "load_carrier", "process_at"}:
            source_op = _find_custody_source(
                completed_op,
                result,
                same_robot=True,
                op_types=_DEPOSIT_SOURCE_OPS,
            )
            if source_op:
                _append_unique(deps, source_op.op_id)
                inserted.append(f"deps {source_op.op_id}->{completed_op.op_id}")

        if completed_op.op_type == "apply_tool":
            tool_source = _find_custody_source(
                completed_op,
                result,
                same_robot=True,
                op_types=_DEPOSIT_SOURCE_OPS,
            )
            if tool_source:
                tool_anchor = _find_same_entity_dest_op(
                    completed_op,
                    result,
                    op_type="carry_to",
                )
                if tool_anchor is None and completed_op.dest:
                    carry_id = _unique_id(f"auto_carry_{completed_op.op_id}", used_ids)
                    tool_anchor = OperationStep(
                        op_id=carry_id,
                        op_type="carry_to",
                        entity=completed_op.entity,
                        dest=completed_op.dest,
                        deps=[tool_source.op_id],
                        robot=completed_op.robot,
                        gaps=["precond"],
                    )
                    result.append(tool_anchor)
                    used_ids.add(carry_id)
                    inserted.append(
                        f"carry_to {carry_id}: {completed_op.entity}->{completed_op.dest}"
                    )
                _append_unique(deps, tool_anchor.op_id if tool_anchor else tool_source.op_id)
            for target_ready in _prior_target_ready_ops(completed_op, result):
                _append_unique(deps, target_ready.op_id)

        if completed_op.op_type == "activate":
            for target_ready in _prior_target_ready_ops(completed_op, result):
                _append_unique(deps, target_ready.op_id)

        if completed_op.op_type == "push_carrier":
            load_op = _latest_carrier_op(completed_op, result, op_type="load_carrier")
            if load_op:
                _append_unique(deps, load_op.op_id)
                inserted.append(f"deps {load_op.op_id}->{completed_op.op_id}")

        if completed_op.op_type == "unload_carrier":
            push_op = _latest_push_to_robot(completed_op, result)
            if push_op:
                _append_unique(deps, push_op.op_id)
                inserted.append(f"deps {push_op.op_id}->{completed_op.op_id}")
            load_op = _latest_carrier_op(completed_op, result, op_type="load_carrier")
            if load_op:
                _append_unique(deps, load_op.op_id)

        if completed_op.op_type == "deposit":
            open_op = _find_open_for_dest(completed_op, result)
            if open_op:
                _prepend_unique(deps, open_op.op_id)

            source_op = _find_custody_source(
                completed_op,
                result,
                same_robot=True,
                op_types=_DEPOSIT_SOURCE_OPS,
            )
            if source_op is None:
                other_source = _find_custody_source(
                    completed_op,
                    result,
                    same_robot=False,
                    op_types=_DEPOSIT_SOURCE_OPS,
                )
                if other_source and completed_op.robot:
                    carrier_route = _build_carrier_transfer_ops(
                        deposit_op=completed_op,
                        source_op=other_source,
                        used_ids=used_ids,
                        ctx=ctx,
                    )
                    if carrier_route:
                        result.extend(carrier_route)
                        used_ids.update(op.op_id for op in carrier_route)
                        inserted.append(
                            "carrier-route "
                            f"{completed_op.entity}: {other_source.robot}->{completed_op.robot} "
                            f"via {_op_carrier_name(carrier_route[1])}"
                        )
                        continue

                    handoff_id = _unique_id(f"auto_handoff_{completed_op.op_id}", used_ids)
                    handoff = OperationStep(
                        op_id=handoff_id,
                        op_type="handoff",
                        entity=completed_op.entity,
                        dest=completed_op.robot,
                        deps=[other_source.op_id],
                        robot=completed_op.robot,
                        params={
                            "target_robot": completed_op.robot,
                            "source_robot": other_source.robot,
                        },
                    )
                    result.append(handoff)
                    used_ids.add(handoff_id)
                    inserted.append(
                        f"handoff {handoff_id}: {completed_op.entity} "
                        f"{other_source.robot}->{completed_op.robot}"
                    )
                    source_op = handoff

            if source_op:
                _append_unique(deps, source_op.op_id)
                if completed_op.dest:
                    inserted.append(
                        f"deps {source_op.op_id}->{completed_op.op_id}; "
                        f"carry_to({completed_op.entity},{completed_op.dest}) is an implicit deposit prefix"
                    )

        completed_op.deps = deps
        result.append(completed_op)

    _reject_carrier_route_conflicts(result)
    return result, inserted


def _reject_cross_robot_acquire_conflicts(ops: list[OperationStep]) -> None:
    # Custody is single-owner: transfers between robots are expressed via
    # handoff/carrier ops (model-written or solver-inserted), never by a second
    # acquire. Two acquire-class ops on one entity from different robots is a
    # conflicting custody claim the downstream per-robot holding simulation
    # cannot see, so it is rejected here.
    acquirers: dict[str, set[str]] = {}
    for op in ops:
        if op.op_type in {"acquire", "local_acquire"}:
            slot = _normalize_robot_slot(op.robot)
            entity = _normalize_entity(op.entity)
            if slot and entity:
                acquirers.setdefault(entity, set()).add(slot)
    for entity, slots in sorted(acquirers.items()):
        if len(slots) > 1:
            raise _PlanningToolError(ToolReason(
                ok=False,
                code="missing_precondition",
                message=(
                    f"custody conflict: entity '{entity}' has acquire-class ops "
                    f"assigned to multiple robots {sorted(slots)}; one robot must "
                    "acquire and transfers are expressed via handoff or carrier ops"
                ),
            ))


def _reject_carrier_route_conflicts(ops: list[OperationStep]) -> None:
    # The expansion has no carrier-location fluent, so a carrier can serve at
    # most one relay route per plan: at most two Push segments (to the loading
    # robot, then to the receiving robot) and a single loading robot. Plans
    # that route a second relay over the same carrier are rejected regardless
    # of whether the routes were model-written or solver-inserted.
    pushes: dict[str, int] = {}
    load_robots: dict[str, set[str]] = {}
    for op in ops:
        if op.op_type == "push_carrier":
            carrier = _normalize_entity(op.entity)
            if carrier:
                pushes[carrier] = pushes.get(carrier, 0) + 1
        elif op.op_type == "load_carrier":
            carrier = _normalize_entity(
                (op.params or {}).get("carrier") or op.dest
            )
            slot = _normalize_robot_slot(op.robot)
            if carrier and slot:
                load_robots.setdefault(carrier, set()).add(slot)
    for carrier in sorted(set(pushes) | set(load_robots)):
        if pushes.get(carrier, 0) > 2 or len(load_robots.get(carrier, ())) > 1:
            raise _PlanningToolError(ToolReason(
                ok=False,
                code="unschedulable",
                message=(
                    f"carrier conflict: '{carrier}' is routed by more than one "
                    "relay chain in this plan (push segments "
                    f"{pushes.get(carrier, 0)}, loading robots "
                    f"{sorted(load_robots.get(carrier, ()))}); one carrier serves "
                    "one relay route"
                ),
            ))


def _build_carrier_transfer_ops(
    *,
    deposit_op: OperationStep,
    source_op: OperationStep,
    used_ids: set[str],
    ctx: AffordanceContext,
) -> list[OperationStep]:
    source_robot = _normalize_robot_slot(source_op.robot)
    receiver_robot = _normalize_robot_slot(deposit_op.robot)
    item = _normalize_entity(deposit_op.entity)
    destination = _normalize_entity(deposit_op.dest)
    if not source_robot or not receiver_robot or source_robot == receiver_robot:
        return []
    if not item or not destination:
        return []
    if _robot_has_capability(ctx, source_robot, CAP_MOBILITY) and _robot_has_capability(
        ctx,
        receiver_robot,
        CAP_MOBILITY,
    ):
        return []

    carrier = _choose_pushable_carrier(ctx)
    pusher_robot = _choose_push_transfer_robot(ctx, avoid={source_robot, receiver_robot})
    if not carrier or not pusher_robot:
        return []

    push_source_id = _unique_id(f"auto_push_source_{deposit_op.op_id}", used_ids)
    used_with_source = {*used_ids, push_source_id}
    load_id = _unique_id(f"auto_load_{deposit_op.op_id}", used_with_source)
    used_with_load = {*used_with_source, load_id}
    push_receiver_id = _unique_id(f"auto_push_receiver_{deposit_op.op_id}", used_with_load)

    return [
        OperationStep(
            op_id=push_source_id,
            op_type="push_carrier",
            entity=carrier,
            dest=source_robot,
            deps=[],
            robot=pusher_robot,
            params={"target_robot": source_robot, "carrier": carrier},
            gaps=["precond"],
        ),
        OperationStep(
            op_id=load_id,
            op_type="load_carrier",
            entity=deposit_op.entity,
            dest=carrier,
            deps=[source_op.op_id, push_source_id],
            robot=source_robot,
            params={"carrier": carrier},
            gaps=["deps", "precond"],
        ),
        OperationStep(
            op_id=push_receiver_id,
            op_type="push_carrier",
            entity=carrier,
            dest=receiver_robot,
            deps=[load_id],
            robot=pusher_robot,
            params={"target_robot": receiver_robot, "carrier": carrier},
            gaps=["deps", "precond"],
        ),
        OperationStep(
            op_id=deposit_op.op_id,
            op_type="unload_carrier",
            entity=deposit_op.entity,
            dest=deposit_op.dest,
            deps=[push_receiver_id, load_id],
            robot=receiver_robot,
            params={"carrier": carrier},
            gaps=list(deposit_op.gaps or []),
        ),
    ]


def _robot_has_capability(ctx: AffordanceContext, slot_id: str, capability: str) -> bool:
    for entry in ctx.robot_capabilities:
        if entry.slot_id == slot_id:
            return capability in set(entry.capabilities)
    return False


def _choose_pushable_carrier(ctx: AffordanceContext) -> str | None:
    candidates = [
        name
        for name, profile in ctx.entity_profiles.items()
        if PUSHABLE_CARRIER_TAG in set(profile.tags)
    ]
    return sorted(candidates)[0] if candidates else None


def _choose_push_transfer_robot(
    ctx: AffordanceContext,
    *,
    avoid: set[str],
) -> str | None:
    candidates = [
        entry
        for entry in ctx.robot_capabilities
        if CAP_PUSH_TRANSFER in set(entry.capabilities)
    ]
    if not candidates:
        return None
    preferred = [entry for entry in candidates if entry.slot_id not in avoid]
    pool = preferred or candidates
    pool.sort(key=lambda entry: (ROBOT_PRIORITY.get(entry.robot_type, 99), entry.slot_id))
    return pool[0].slot_id


def _prior_target_ready_ops(
    op: OperationStep,
    prior_ops: list[OperationStep],
) -> list[OperationStep]:
    target = _normalize_entity(op.dest or op.entity)
    if not target:
        return []
    return [
        candidate
        for candidate in prior_ops
        if candidate.op_type in {"deposit", "process_at"}
        and _normalize_entity(candidate.dest) == target
    ]


def _latest_same_entity_op(
    op: OperationStep,
    prior_ops: list[OperationStep],
) -> OperationStep | None:
    entity = _normalize_entity(op.entity)
    if not entity:
        return None
    for candidate in reversed(prior_ops):
        if _normalize_entity(candidate.entity) == entity:
            return candidate
    return None


def _find_same_entity_dest_op(
    op: OperationStep,
    prior_ops: list[OperationStep],
    *,
    op_type: str,
) -> OperationStep | None:
    entity = _normalize_entity(op.entity)
    dest = _normalize_entity(op.dest)
    for candidate in reversed(prior_ops):
        if candidate.op_type != op_type:
            continue
        if _normalize_entity(candidate.entity) != entity:
            continue
        if dest and _normalize_entity(candidate.dest) != dest:
            continue
        if candidate.robot == op.robot:
            return candidate
    return None


def _latest_carrier_op(
    op: OperationStep,
    prior_ops: list[OperationStep],
    *,
    op_type: str,
) -> OperationStep | None:
    carrier = _op_carrier_name(op)
    for candidate in reversed(prior_ops):
        if candidate.op_type != op_type:
            continue
        if carrier and _op_carrier_name(candidate) != carrier:
            continue
        return candidate
    return None


def _latest_push_to_robot(
    op: OperationStep,
    prior_ops: list[OperationStep],
) -> OperationStep | None:
    target_robot = op.robot
    carrier = _op_carrier_name(op)
    for candidate in reversed(prior_ops):
        if candidate.op_type != "push_carrier":
            continue
        if carrier and _op_carrier_name(candidate) != carrier:
            continue
        candidate_target = _normalize_robot_slot(
            str(candidate.params.get("target_robot") or candidate.dest or "")
        )
        if target_robot and candidate_target == target_robot:
            return candidate
    return None


def _op_carrier_name(op: OperationStep) -> str | None:
    carrier = op.params.get("carrier")
    if isinstance(carrier, str) and carrier.strip():
        return _normalize_entity(carrier)
    if op.op_type == "push_carrier":
        return _normalize_entity(op.entity)
    if op.op_type == "load_carrier":
        return _normalize_entity(op.dest)
    return None


# debug_rl.md §7.8 P5 (2026-09-07): where Phi puts the `open` it inserts for a
# deposit into an openable container, and who does it.
#   early (default, every recorded bucket): the depositing robot opens BEFORE
#     fetching its own item (inserted at that robot's first op on the entity).
#     One list step longer than the LOCUS template GT on every F8 m3 row
#     (simulator passes, length gate scores 0: 36/40 rows, both RL arms).
#   late_freehand: if a container-capable robot with >=2 end effectors is on
#     the team, that robot opens at the container while holding its item
#     (inserted right before the deposit); otherwise fall back to early. Offline
#     re-judge on 7B/8B/RL LOCUS buckets: +30..33 rows, 0 losses, no other
#     cell touched (scratch/rl_axis_analysis/p5_variant_c_bucket.py).
PHI_OPEN_PLACEMENT_ENV = "LNSDL_PHI_OPEN_PLACEMENT"
PHI_OPEN_PLACEMENT_DEFAULT = "late_freehand"


def phi_open_placement_mode() -> str:
    mode = (os.environ.get(PHI_OPEN_PLACEMENT_ENV) or PHI_OPEN_PLACEMENT_DEFAULT).strip().lower()
    if mode not in ("early", "late_freehand"):
        raise ValueError(f"{PHI_OPEN_PLACEMENT_ENV} must be early or late_freehand, got {mode!r}")
    return mode


def _choose_free_hand_opener(preferred_slot: str | None, ctx: AffordanceContext):
    """Container-capable robot that can open while holding (ee >= 2), the
    depositing robot first; None when the team has no such robot."""
    by_slot = {entry.slot_id: entry for entry in ctx.robot_capabilities}
    capable = [
        entry for entry in ctx.robot_capabilities
        if "container_operation" in set(entry.capabilities) and entry.end_effectors >= 2
    ]
    preferred = by_slot.get(preferred_slot or "")
    if preferred is not None and preferred in capable:
        return preferred.slot_id
    if not capable:
        return None
    capable.sort(key=lambda entry: (ROBOT_PRIORITY.get(entry.robot_type, 99), entry.slot_id))
    return capable[0].slot_id


def _end_effectors(ctx: AffordanceContext, slot: str | None) -> int:
    for entry in ctx.robot_capabilities:
        if entry.slot_id == slot:
            return int(entry.end_effectors)
    return 1


def _held_count_before(ops: list[OperationStep], deposit: OperationStep, slot: str) -> int:
    """Distinct items `slot` is still holding when it reaches `deposit`
    (custody sources minus deposits earlier in list order)."""
    held: set[str] = set()
    for cand in ops:
        if cand is deposit:
            break
        if _normalize_robot_slot(str(cand.robot or "")) != slot:
            continue
        entity = _normalize_entity(cand.entity)
        if not entity:
            continue
        if cand.op_type in _CUSTODY_SOURCE_OPS:
            held.add(entity)
        elif cand.op_type in ("deposit", "load_carrier", "handoff", "unload_carrier"):
            held.discard(entity)
    return len(held)


def _insert_missing_open_ops(
    ops: list[OperationStep],
    ctx: AffordanceContext,
) -> tuple[list[OperationStep], list[str]]:
    placement = phi_open_placement_mode()
    explicit_open_dests = {
        _normalize_entity(op.dest or op.entity)
        for op in ops
        if op.op_type == "open"
    }
    explicit_open_dests.discard(None)

    insertions_by_index: dict[int, list[OperationStep]] = defaultdict(list)
    inserted: list[str] = []
    used_ids = {op.op_id for op in ops}
    for index, op in enumerate(ops):
        dest = _normalize_entity(op.dest)
        if op.op_type != "deposit" or not dest:
            continue
        if not entity_requires_open_before_place(dest) or dest in explicit_open_dests:
            continue
        open_id = _unique_id(f"auto_open_{dest.replace(' ', '_')}", used_ids)
        used_ids.add(open_id)
        explicit_open_dests.add(dest)
        open_robot = None
        insertion_index = None
        open_deps: list[str] = []
        if placement == "late_freehand":
            open_robot = _choose_free_hand_opener(op.robot, ctx)
            if open_robot is not None:
                # Keep the open before the FIRST deposit into this container in
                # list order (every later deposit wires onto it), and make it
                # depend on the opener's own fetch of its item so the opener
                # opens on arrival, holding, instead of before fetching.
                insertion_index = index
                opener_slot = _normalize_robot_slot(str(open_robot))
                opener_deposit = next(
                    (cand for cand in ops[index:]
                     if cand.op_type == "deposit" and _normalize_entity(cand.dest) == dest
                     and _normalize_robot_slot(str(cand.robot or "")) == opener_slot),
                    None,
                )
                if opener_deposit is not None and _held_count_before(ops, opener_deposit, opener_slot) >= _end_effectors(ctx, open_robot):
                    # Opener arrives with every hand full (e.g. ee=2 carrying
                    # two items): it cannot open while holding -> early rule.
                    open_robot = None
                elif opener_deposit is not None:
                    source = next(
                        (cand for cand in ops
                         if cand.op_type in _DEPOSIT_SOURCE_OPS
                         and _normalize_entity(cand.entity) == _normalize_entity(opener_deposit.entity)
                         and _normalize_robot_slot(str(cand.robot or "")) == opener_slot),
                        None,
                    )
                    if source is not None:
                        open_deps = [source.op_id]
        if open_robot is None:
            open_robot = _choose_open_robot(op.robot, ctx)
            insertion_index = _earliest_related_index(ops, index, op.entity)
        insertions_by_index[insertion_index].append(OperationStep(
            op_id=open_id,
            op_type="open",
            dest=dest,
            deps=list(open_deps),
            robot=open_robot,
            gaps=["precond"],
        ))
        inserted.append(
            f"open-before-deposit {open_id}: open({dest})@{open_robot}"
            + (" (late_freehand)" if open_deps else "")
        )

    if not insertions_by_index:
        return ops, inserted

    new_ops: list[OperationStep] = []
    for index, op in enumerate(ops):
        new_ops.extend(insertions_by_index.get(index, []))
        new_ops.append(op)
    new_ops.extend(insertions_by_index.get(len(ops), []))
    return new_ops, inserted


def _choose_open_robot(preferred_slot: str | None, ctx: AffordanceContext) -> str | None:
    by_slot = {entry.slot_id: entry for entry in ctx.robot_capabilities}
    preferred = by_slot.get(preferred_slot or "")
    if preferred and "container_operation" in set(preferred.capabilities):
        return preferred.slot_id
    capable = [
        entry for entry in ctx.robot_capabilities
        if "container_operation" in set(entry.capabilities) and entry.end_effectors > 0
    ]
    if not capable:
        return preferred_slot
    capable.sort(key=lambda entry: (ROBOT_PRIORITY.get(entry.robot_type, 99), entry.slot_id))
    return capable[0].slot_id


def _earliest_related_index(
    ops: list[OperationStep],
    deposit_index: int,
    entity: str | None,
) -> int:
    normalized = _normalize_entity(entity)
    if not normalized:
        return deposit_index
    for index, candidate in enumerate(ops[:deposit_index]):
        if _normalize_entity(candidate.entity) == normalized and candidate.op_type in _DEPOSIT_SOURCE_OPS:
            return index
    return deposit_index


def _find_open_for_dest(op: OperationStep, prior_ops: list[OperationStep]) -> OperationStep | None:
    dest = _normalize_entity(op.dest)
    if not dest:
        return None
    for candidate in reversed(prior_ops):
        if candidate.op_type != "open":
            continue
        if _normalize_entity(candidate.dest or candidate.entity) == dest:
            return candidate
    return None


def _find_custody_source(
    op: OperationStep,
    prior_ops: list[OperationStep],
    *,
    same_robot: bool,
    op_types: frozenset[str] = _DEPOSIT_SOURCE_OPS,
) -> OperationStep | None:
    entity = _normalize_entity(op.entity)
    if not entity:
        return None
    for candidate in reversed(prior_ops):
        if candidate.op_type not in op_types:
            continue
        if _normalize_entity(candidate.entity) != entity:
            continue
        if same_robot and candidate.robot == op.robot:
            return candidate
        if not same_robot and candidate.robot and candidate.robot != op.robot:
            return candidate
    return None


def _clone_op(op: OperationStep, *, robot: str | None = None) -> OperationStep:
    return OperationStep(
        op_id=str(op.op_id),
        op_type=str(op.op_type),
        entity=op.entity,
        dest=op.dest,
        deps=list(op.deps or []),
        robot=robot if robot is not None else op.robot,
        params=dict(op.params or {}),
        gaps=list(getattr(op, "gaps", []) or []),
    )


def _op_robot(op: OperationStep, assign: dict[str, Any]) -> str | None:
    if op.robot:
        return _normalize_robot_slot(op.robot)
    entity = _normalize_entity(op.entity)
    if not assign:
        return None
    if entity and entity in assign and isinstance(assign[entity], str):
        return _normalize_robot_slot(assign[entity])
    candidate_slots: list[str] = []
    for slot, payload in assign.items():
        slot_name = _normalize_robot_slot(str(slot))
        if isinstance(payload, dict):
            items = payload.get("items", [])
            if entity and isinstance(items, list) and entity in {_normalize_entity(item) for item in items}:
                return slot_name
            if payload.get("robot") and entity == _normalize_entity(payload.get("entity")):
                return _normalize_robot_slot(str(payload["robot"]))
        elif isinstance(payload, list):
            if entity and entity in {_normalize_entity(item) for item in payload}:
                return slot_name
        candidate_slots.append(slot_name)
    return candidate_slots[0] if len(candidate_slots) == 1 else None


def _validation_reason(issue: Any) -> ToolReason:
    code = str(getattr(issue, "code", "") or "unschedulable")
    canonical = code
    if code in {"invalid_dep_ref"}:
        canonical = "invalid_dep"
    elif code == "dependency_cycle":
        canonical = "dep_cycle"
    elif code.startswith("no_free_end_effector") or code == "end_effector_capacity_exceeded":
        canonical = "ee_capacity"
    elif code in {"missing_robot", "unknown_robot_slot", "capability_mismatch"}:
        canonical = "capability_mismatch"
    elif (
        code.startswith("missing_")
        or code.startswith("not_holding_")
        or code == "cross_robot_custody"
    ):
        canonical = "missing_precondition"
    return ToolReason(
        ok=False,
        code=canonical,
        op_ids=list(getattr(issue, "op_ids", []) or []),
        message=str(getattr(issue, "message", "") or code),
    )


def _topological_order_ops(ops: list[OperationStep]) -> list[OperationStep]:
    by_id = {op.op_id: op for op in ops}
    original_index = {op.op_id: index for index, op in enumerate(ops)}
    remaining = set(by_id)
    ordered: list[OperationStep] = []
    while remaining:
        ready = [
            op_id for op_id in remaining
            if all(dep not in remaining for dep in by_id[op_id].deps if dep in by_id)
        ]
        if not ready:
            raise _PlanningToolError(ToolReason(
                ok=False,
                code="dep_cycle",
                op_ids=sorted(remaining),
                message=f"deps form a cycle through {sorted(remaining)}",
            ))
        ready.sort(key=lambda op_id: original_index[op_id])
        for op_id in ready:
            ordered.append(by_id[op_id])
            remaining.remove(op_id)
    return ordered


def _attach_executor_types(graph_plan: GraphPlan, ctx: AffordanceContext) -> None:
    robot_type_by_slot = {
        entry.slot_id: entry.robot_type
        for entry in ctx.robot_capabilities
    }
    for node in graph_plan.nodes:
        if not node.executor_slot:
            raise _PlanningToolError(ToolReason(
                ok=False,
                code="capability_mismatch",
                op_ids=[node.node_id],
                message=f"Node {node.node_id} is missing executor_slot.",
            ))
        robot_type = robot_type_by_slot.get(node.executor_slot)
        if not robot_type:
            raise _PlanningToolError(ToolReason(
                ok=False,
                code="capability_mismatch",
                op_ids=[node.node_id],
                message=f"Unknown robot slot {node.executor_slot}.",
            ))
        node.executor_robot_type = robot_type


def _node_to_op_ids(graph_plan: GraphPlan, ops: list[OperationStep]) -> dict[str, str]:
    if len(graph_plan.nodes) == len(ops):
        return {
            node.node_id: op.op_id
            for node, op in zip(graph_plan.nodes, ops)
        }
    return {node.node_id: node.node_id for node in graph_plan.nodes}


def _expand_graph_nodes(
    graph_plan: GraphPlan,
    node_to_op: dict[str, str],
) -> tuple[dict[str, list[list[str]]], dict[str, frozenset[str]]]:
    node_map = {node.node_id: node for node in graph_plan.nodes}
    node_programs: dict[str, list[list[str]]] = {}
    node_end_states: dict[str, frozenset[str]] = {}
    planned_nodes: set[str] = set()

    while len(planned_nodes) < len(graph_plan.nodes):
        progressed = False
        for node in graph_plan.nodes:
            if node.node_id in planned_nodes:
                continue
            if any(predecessor not in planned_nodes for predecessor in node.predecessors):
                continue
            same_executor_as = node.required_constraints.get("same_executor_as")
            if same_executor_as and str(same_executor_as) not in planned_nodes:
                continue
            try:
                planned = plan_node_micro_program(
                    node=node,
                    node_map=node_map,
                    predecessor_end_states=node_end_states,
                )
            except Exception as exc:  # noqa: BLE001
                raise _PlanningToolError(ToolReason(
                    ok=False,
                    code="goal_unreachable",
                    op_ids=[node_to_op.get(node.node_id, node.node_id)],
                    message=f"{exc.__class__.__name__}: {exc}",
                )) from exc
            node_programs[node.node_id] = [list(action) for action in planned.actions]
            node_end_states[node.node_id] = planned.final_state
            planned_nodes.add(node.node_id)
            progressed = True
        if not progressed:
            unresolved = [
                node_to_op.get(node.node_id, node.node_id)
                for node in graph_plan.nodes
                if node.node_id not in planned_nodes
            ]
            raise _PlanningToolError(ToolReason(
                ok=False,
                code="unschedulable",
                op_ids=unresolved,
                message=f"Micro planner could not expand all nodes: {unresolved}",
            ))
    return node_programs, node_end_states


def _list_schedule(
    *,
    graph_plan: GraphPlan,
    node_programs: dict[str, list[list[str]]],
    node_to_op: dict[str, str],
    op_by_id: dict[str, OperationStep],
    ctx: AffordanceContext,
) -> list[MicroStep]:
    nodes = list(graph_plan.nodes)
    robot_order = _robot_order(ctx)
    ee_by_slot = {
        entry.slot_id: int(entry.end_effectors)
        for entry in ctx.robot_capabilities
    }
    offsets = {node.node_id: 0 for node in nodes}
    completed: set[str] = set()
    completed_step: dict[str, int] = {}
    held_by_robot: dict[str, set[str]] = defaultdict(set)
    anchored_target_by_robot: dict[str, str] = {}
    micro_steps: list[MicroStep] = []
    current_step = 1

    while len(completed) < len(nodes):
        step_actions: dict[str, list[str]] = {}
        active_nodes: dict[str, AffordanceNode] = {}
        progressed_without_action = False
        delayed_zero_action_completion = False

        for robot in robot_order:
            while robot not in step_actions:
                node = _next_ready_node(
                    nodes=nodes,
                    robot=robot,
                    completed=completed,
                    completed_step=completed_step,
                    current_step=current_step,
                    node_programs=node_programs,
                    offsets=offsets,
                    free_ee=(
                        ee_by_slot.get(robot, 0) - len(held_by_robot[robot])
                    ),
                )
                if node is None:
                    break

                program = node_programs[node.node_id]
                offset = offsets[node.node_id]
                while offset < len(program) and _is_redundant_move_action(
                    robot=robot,
                    action=program[offset],
                    anchored_target_by_robot=anchored_target_by_robot,
                ):
                    offset += 1
                    offsets[node.node_id] = offset
                    progressed_without_action = True

                if offset >= len(program):
                    delayed_zero_action_completion = (
                        _complete_node_without_action(
                            node=node,
                            completed=completed,
                            completed_step=completed_step,
                            current_step=current_step,
                            node_to_op=node_to_op,
                            op_by_id=op_by_id,
                            held_by_robot=held_by_robot,
                        )
                        or delayed_zero_action_completion
                    )
                    progressed_without_action = True
                    continue

                action = program[offset]
                reason = _check_action_feasible(
                    node=node,
                    action=action,
                    node_to_op=node_to_op,
                    held_by_robot=held_by_robot,
                    ee_by_slot=ee_by_slot,
                )
                if reason is not None:
                    raise _PlanningToolError(reason)

                step_actions[robot] = action
                active_nodes[robot] = node
                offsets[node.node_id] = offset + 1
                break

        if step_actions:
            for robot, action in step_actions.items():
                node = active_nodes[robot]
                _apply_action_effects(
                    robot=robot,
                    node=node,
                    action=action,
                    held_by_robot=held_by_robot,
                    anchored_target_by_robot=anchored_target_by_robot,
                )
                if offsets[node.node_id] >= len(node_programs[node.node_id]):
                    completed.add(node.node_id)
                    completed_step[node.node_id] = current_step
                    _apply_node_completion_effects(
                        node=node,
                        node_to_op=node_to_op,
                        op_by_id=op_by_id,
                        held_by_robot=held_by_robot,
                    )
            micro_steps.append(MicroStep(
                step=len(micro_steps) + 1,
                actions=step_actions,
            ))
            current_step += 1
            continue

        if progressed_without_action:
            if delayed_zero_action_completion:
                current_step += 1
            continue

        unresolved = [
            node_to_op.get(node.node_id, node.node_id)
            for node in nodes
            if node.node_id not in completed
        ]
        raise _PlanningToolError(ToolReason(
            ok=False,
            code="unschedulable",
            op_ids=unresolved,
            message=f"List scheduler could not make progress: {unresolved}",
        ))

    return micro_steps


# plan.md A0p 附注 / facts M5c (2026-09-06, user decision (a)): which list
# scheduler `schedule` runs. v2 is the main configuration since 2026-09-06.
#   v1 (frozen, opt-in via LNSDL_PHI_SCHEDULER=v1): node-granular deps, a
#      predecessor is visible only from the step AFTER it completes. Every
#      bucket recorded before 2026-09-06 ran it, and the LOCUS build-time Phi
#      reconstruction gate (locus_validate / audit_locus_schedule_gt_reconstruct)
#      was scored against it -- those keep v1 pinned so their numbers stay
#      comparable. LOCUS ground truth itself is template-zipped per-robot tracks
#      (locus_task_pool), not a scheduler output, and is unaffected.
#   v2 (default): the runtime dispatcher's visibility rules -- a non-state-change predecessor
#      that completes in the current step is visible to robots scheduled later in
#      the same step (fixpoint over robots), and an open/close-door predecessor
#      gates only the contact action, not the approach `Move`. Same-robot ee
#      capacity and every feasibility check are unchanged. On VIKI-L2 the v1
#      schedule is exactly one step longer than the official GT on ~29% of test
#      rows (length gate -> 0); v2 removes that without touching LOCUS readings.
PHI_SCHEDULER_ENV = "LNSDL_PHI_SCHEDULER"
PHI_SCHEDULER_DEFAULT = "v2"


def phi_scheduler_mode() -> str:
    """The scheduler `schedule` will run (recorded into eval artifacts)."""
    return (os.environ.get(PHI_SCHEDULER_ENV) or PHI_SCHEDULER_DEFAULT).strip().lower()


def _select_list_scheduler():
    mode = (os.environ.get(PHI_SCHEDULER_ENV) or PHI_SCHEDULER_DEFAULT).strip().lower()
    if mode == "v2":
        return _list_schedule_v2
    if mode != "v1":
        raise ValueError(f"{PHI_SCHEDULER_ENV} must be v1 or v2, got {mode!r}")
    return _list_schedule


def _list_schedule_v2(
    *,
    graph_plan: GraphPlan,
    node_programs: dict[str, list[list[str]]],
    node_to_op: dict[str, str],
    op_by_id: dict[str, OperationStep],
    ctx: AffordanceContext,
) -> list[MicroStep]:
    nodes = list(graph_plan.nodes)
    kind_by_id = {node.node_id: node.node_kind for node in nodes}
    # a node feeding a state-change successor (load_carrier -> push_transport)
    # must not become visible in the step it finishes either (dispatcher rule)
    feeds_state_change = {
        node.node_id: any(
            kind_by_id.get(other.node_id) in _STATE_CHANGE_NODE_KINDS
            for other in nodes if node.node_id in other.predecessors
        )
        for node in nodes
    }
    door_kinds = frozenset({"open_container", "close_container"})
    robot_order = _robot_order(ctx)
    ee_by_slot = {entry.slot_id: int(entry.end_effectors) for entry in ctx.robot_capabilities}
    offsets = {node.node_id: 0 for node in nodes}
    completed: set[str] = set()
    completed_step: dict[str, int] = {}
    held_by_robot: dict[str, set[str]] = defaultdict(set)
    anchored_target_by_robot: dict[str, str] = {}
    micro_steps: list[MicroStep] = []
    current_step = 1

    def visible(deps: list[str]) -> bool:
        for dep in deps:
            if dep not in completed:
                return False
            done_at = completed_step.get(dep, 10**9)
            if done_at < current_step:
                continue
            if (done_at == current_step and kind_by_id.get(dep) not in _STATE_CHANGE_NODE_KINDS
                    and not feeds_state_change.get(dep)):
                continue
            return False
        return True

    def approach_allowed(node: AffordanceNode) -> bool:
        pending = [dep for dep in node.predecessors if not visible([dep])]
        if not pending or any(kind_by_id.get(dep) not in door_kinds for dep in pending):
            return False
        program = node_programs[node.node_id]
        offset = offsets[node.node_id]
        return offset < len(program) and bool(program[offset]) and program[offset][0] == "Move"

    def next_ready(robot: str, free_ee: int) -> AffordanceNode | None:
        fallback: AffordanceNode | None = None
        for node in nodes:
            if node.node_id in completed or node.executor_slot != robot:
                continue
            if not visible(node.predecessors) and not approach_allowed(node):
                continue
            same_executor_as = node.required_constraints.get("same_executor_as")
            if same_executor_as and not visible([str(same_executor_as)]):
                continue
            if free_ee > 0:
                return node
            if fallback is None:
                fallback = node
            if not _node_needs_free_ee(program=node_programs.get(node.node_id, []),
                                       offset=int(offsets.get(node.node_id, 0))):
                return node
        return fallback

    while len(completed) < len(nodes):
        step_actions: dict[str, list[str]] = {}
        progressed_without_action = False
        delayed_zero_action_completion = False
        changed = True
        while changed:
            changed = False
            for robot in robot_order:
                while robot not in step_actions:
                    node = next_ready(robot, ee_by_slot.get(robot, 0) - len(held_by_robot[robot]))
                    if node is None:
                        break
                    program = node_programs[node.node_id]
                    offset = offsets[node.node_id]
                    while offset < len(program) and _is_redundant_move_action(
                        robot=robot, action=program[offset],
                        anchored_target_by_robot=anchored_target_by_robot,
                    ):
                        offset += 1
                        offsets[node.node_id] = offset
                        progressed_without_action = True
                    if offset >= len(program):
                        delayed_zero_action_completion = (
                            _complete_node_without_action(
                                node=node, completed=completed, completed_step=completed_step,
                                current_step=current_step, node_to_op=node_to_op,
                                op_by_id=op_by_id, held_by_robot=held_by_robot,
                            ) or delayed_zero_action_completion
                        )
                        progressed_without_action = True
                        changed = True
                        continue
                    action = program[offset]
                    reason = _check_action_feasible(
                        node=node, action=action, node_to_op=node_to_op,
                        held_by_robot=held_by_robot, ee_by_slot=ee_by_slot,
                    )
                    if reason is not None:
                        raise _PlanningToolError(reason)
                    step_actions[robot] = action
                    offsets[node.node_id] = offset + 1
                    # effects and completion are applied immediately so that a
                    # robot later in the order (or an earlier one on the next
                    # fixpoint pass) can see the finished predecessor this step
                    _apply_action_effects(
                        robot=robot, node=node, action=action,
                        held_by_robot=held_by_robot,
                        anchored_target_by_robot=anchored_target_by_robot,
                    )
                    if offsets[node.node_id] >= len(program):
                        completed.add(node.node_id)
                        completed_step[node.node_id] = current_step
                        _apply_node_completion_effects(
                            node=node, node_to_op=node_to_op, op_by_id=op_by_id,
                            held_by_robot=held_by_robot,
                        )
                        changed = True
                    break

        if step_actions:
            micro_steps.append(MicroStep(step=len(micro_steps) + 1, actions=step_actions))
            current_step += 1
            continue
        if progressed_without_action:
            if delayed_zero_action_completion:
                current_step += 1
            continue
        unresolved = [node_to_op.get(node.node_id, node.node_id) for node in nodes if node.node_id not in completed]
        raise _PlanningToolError(ToolReason(
            ok=False, code="unschedulable", op_ids=unresolved,
            message=f"List scheduler could not make progress: {unresolved}",
        ))
    return micro_steps


def _robot_order(ctx: AffordanceContext) -> list[str]:
    entries = list(ctx.robot_capabilities)
    entries.sort(key=lambda entry: (ROBOT_PRIORITY.get(entry.robot_type, 99), entry.slot_id))
    return [entry.slot_id for entry in entries]


def _next_ready_node(
    *,
    nodes: list[AffordanceNode],
    robot: str,
    completed: set[str],
    completed_step: dict[str, int],
    current_step: int,
    node_programs: dict[str, list[list[str]]] | None = None,
    offsets: dict[str, int] | None = None,
    free_ee: int | None = None,
) -> AffordanceNode | None:
    """First dependency-ready node for ``robot``, end-effector aware.

    An intent sketch carries no deps, so ``_topological_order_ops`` lists every
    dep-free op first: a two-object carry becomes acquire/acquire/deposit/deposit.
    With two or more end-effectors that order is fine. With ee=1 it is not a
    plan that fails, it is the wrong ORDER -- the deposit that frees the gripper
    is dependency-ready and merely sits later in the same list. LOCUS M1 is
    built on exactly this shape (docs/benchmark.md M1: "double-object sequential
    carry, ee=1 forces serialisation"); VIKI-L2 never exercises it, which is why
    all 7196 VIKI rows schedule and 27/600 LOCUS m1 rows did not.

    So when the robot has no free end-effector, prefer a ready node whose
    remaining program never needs a fresh one (i.e. a deposit/close that
    RELEASES). Only if no such node is ready do we fall back to plain list
    order, where ``_check_action_feasible`` reports the capacity violation with
    the same message as before. Robots with a free end-effector, and every plan
    that scheduled before this change, take the unchanged first-ready branch.
    """
    fallback: AffordanceNode | None = None
    for node in nodes:
        if node.node_id in completed or node.executor_slot != robot:
            continue
        if not _deps_visible(node.predecessors, completed, completed_step, current_step):
            continue
        same_executor_as = node.required_constraints.get("same_executor_as")
        if same_executor_as and not _deps_visible(
            [str(same_executor_as)],
            completed,
            completed_step,
            current_step,
        ):
            continue
        if free_ee is None or free_ee > 0 or node_programs is None or offsets is None:
            return node
        if fallback is None:
            fallback = node
        if not _node_needs_free_ee(
            program=node_programs.get(node.node_id, []),
            offset=int(offsets.get(node.node_id, 0)),
        ):
            return node
    return fallback


def _node_needs_free_ee(*, program: list[list[str]], offset: int) -> bool:
    """True if any action still pending in ``program`` claims an end-effector."""
    return any(
        action and action[0] in _EE_CLAIMING_ACTIONS
        for action in program[max(0, offset):]
    )


def _deps_visible(
    deps: list[str],
    completed: set[str],
    completed_step: dict[str, int],
    current_step: int,
) -> bool:
    return all(
        dep in completed and completed_step.get(dep, 10**9) < current_step
        for dep in deps
    )


# Actions that need a free end-effector to start. Kept as one constant so the
# feasibility check and the ee-aware node ordering can never drift apart.
_EE_CLAIMING_ACTIONS = frozenset({"Grasp", "Open", "Close"})


def _check_action_feasible(
    *,
    node: AffordanceNode,
    action: list[str],
    node_to_op: dict[str, str],
    held_by_robot: dict[str, set[str]],
    ee_by_slot: dict[str, int],
) -> ToolReason | None:
    robot = node.executor_slot or ""
    action_type = action[0] if action else ""
    held = held_by_robot[robot]
    ee = ee_by_slot.get(robot, 0)
    free_ee = ee - len(held)
    op_id = node_to_op.get(node.node_id, node.node_id)

    if action_type in _EE_CLAIMING_ACTIONS and free_ee <= 0:
        return ToolReason(
            ok=False,
            code="ee_capacity",
            op_ids=[op_id],
            message=(
                f"{robot} has ee={ee} and is holding {sorted(held)} before "
                f"{action_type}; no free end-effector is available."
            ),
        )
    if action_type == "Place":
        item = _node_item(node)
        if item and item != "held_items" and item not in held:
            return ToolReason(
                ok=False,
                code="missing_precondition",
                op_ids=[op_id],
                message=(
                    f"{robot} is not holding {item} before Place; "
                    f"holding={sorted(held)}."
                ),
            )
    return None


def _apply_action_effects(
    *,
    robot: str,
    node: AffordanceNode,
    action: list[str],
    held_by_robot: dict[str, set[str]],
    anchored_target_by_robot: dict[str, str],
) -> None:
    if not action:
        return
    action_type = action[0]
    target = action[1] if len(action) >= 2 else None

    if action_type == "Grasp" and target:
        held_by_robot[robot].add(target)
    elif action_type == "Place":
        item = _node_item(node)
        if item == "held_items":
            held_by_robot[robot].clear()
        elif item:
            held_by_robot[robot].discard(item)

    updated_anchor = _updated_anchor_target(
        action=action,
        previous_anchor=anchored_target_by_robot.get(robot),
    )
    if updated_anchor:
        anchored_target_by_robot[robot] = updated_anchor


def _complete_node_without_action(
    *,
    node: AffordanceNode,
    completed: set[str],
    completed_step: dict[str, int],
    current_step: int,
    node_to_op: dict[str, str],
    op_by_id: dict[str, OperationStep],
    held_by_robot: dict[str, set[str]],
) -> bool:
    completed.add(node.node_id)
    delayed = node.node_kind in _STATE_CHANGE_NODE_KINDS
    completed_step[node.node_id] = current_step if delayed else current_step - 1
    _apply_node_completion_effects(
        node=node,
        node_to_op=node_to_op,
        op_by_id=op_by_id,
        held_by_robot=held_by_robot,
    )
    return delayed


def _apply_node_completion_effects(
    *,
    node: AffordanceNode,
    node_to_op: dict[str, str],
    op_by_id: dict[str, OperationStep],
    held_by_robot: dict[str, set[str]],
) -> None:
    op = op_by_id.get(node_to_op.get(node.node_id, ""))
    if not op or op.op_type != "handoff" or not op.entity or not op.robot:
        return
    source_robot = op.params.get("source_robot")
    if isinstance(source_robot, str) and source_robot:
        held_by_robot[source_robot].discard(op.entity)
    held_by_robot[op.robot].add(op.entity)


def _is_redundant_move_action(
    *,
    robot: str,
    action: list[str],
    anchored_target_by_robot: dict[str, str],
) -> bool:
    return len(action) >= 2 and action[0] == "Move" and anchored_target_by_robot.get(robot) == action[1]


def _updated_anchor_target(
    *,
    action: list[str],
    previous_anchor: str | None,
) -> str | None:
    if not action:
        return previous_anchor
    action_type = action[0]
    target = action[1] if len(action) >= 2 else None
    if not target:
        return previous_anchor
    if action_type in {"Move", "Place"}:
        return target
    if action_type in {"Open", "Interact"} and entity_can_be_destination(target):
        return target
    return previous_anchor


def _node_item(node: AffordanceNode) -> str | None:
    item = node.object_bindings.get("item")
    if item:
        return item
    return node.object_name


def _prepend_unique(values: list[str], value: str) -> None:
    if value in values:
        values.remove(value)
    values.insert(0, value)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _unique_id(prefix: str, used_ids: set[str]) -> str:
    candidate = prefix
    suffix = 2
    while candidate in used_ids:
        candidate = f"{prefix}_{suffix}"
        suffix += 1
    return candidate


def _normalize_entity(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "n/a"}:
        return None
    return text


def _normalize_robot_slot(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0].lower() == "r" and text[1:].isdigit():
        return f"R{text[1:]}"
    return text
