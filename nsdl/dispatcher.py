from __future__ import annotations

from nsdl.catalog import entity_can_be_destination, get_action_primitive
from nsdl.micro_planner import plan_node_micro_program
from nsdl.schema import AffordanceNode, GraphPlan, MicroStep

# Node kinds whose completion implies a physical state change that
# downstream nodes depend on (e.g. door open, carrier arrived).
# These must NOT be propagated to completed_available in the same step
# they finish — downstream nodes can only start in the *next* step.
_STATE_CHANGE_NODE_KINDS = frozenset({
    "open_container",   # Place(item, cabinet) needs door already open
    "close_container",  # downstream nodes assume the door is shut
    "push_transport",   # Reach(item in box) needs box at destination
    "handoff_node",     # Custody transfer must complete before receiver acts
})


def dispatch_graph(graph_plan: GraphPlan) -> list[MicroStep]:
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
            planned = plan_node_micro_program(
                node=node,
                node_map=node_map,
                predecessor_end_states=node_end_states,
            )
            node_programs[node.node_id] = [
                list(action)
                for action in planned.actions
            ]
            node_end_states[node.node_id] = planned.final_state
            planned_nodes.add(node.node_id)
            progressed = True

        if not progressed:
            unresolved_nodes = [
                node.node_id
                for node in graph_plan.nodes
                if node.node_id not in planned_nodes
            ]
            raise ValueError(
                "Micro planner could not plan all graph nodes. "
                f"Unresolved nodes: {unresolved_nodes}"
            )

    node_offsets = {node.node_id: 0 for node in graph_plan.nodes}
    completed: set[str] = set()
    micro_steps: list[MicroStep] = []
    anchored_target_by_robot: dict[str, str] = {}

    while len(completed) < len(graph_plan.nodes):
        step_actions: dict[str, list[str]] = {}
        completed_this_step: list[str] = []
        progressed = False
        completed_available = set(completed)

        for node in graph_plan.nodes:
            if node.node_id in completed:
                continue
            if any(predecessor not in completed_available for predecessor in node.predecessors):
                continue
            if not node.executor_slot:
                raise ValueError(f"Node {node.node_id} is missing executor_slot for dispatch.")
            if node.executor_slot in step_actions:
                continue
            same_executor_as = node.required_constraints.get("same_executor_as")
            if same_executor_as and str(same_executor_as) not in completed_available:
                continue

            program = node_programs[node.node_id]
            offset = node_offsets[node.node_id]
            while offset < len(program) and _is_redundant_move_action(
                robot=node.executor_slot,
                action=program[offset],
                anchored_target_by_robot=anchored_target_by_robot,
            ):
                offset += 1
                node_offsets[node.node_id] = offset
                progressed = True

            if offset >= len(program):
                if node.node_id not in completed_this_step:
                    completed_this_step.append(node.node_id)
                    completed_available.add(node.node_id)
                progressed = True
                continue

            action = program[offset]
            step_actions[node.executor_slot] = action
            node_offsets[node.node_id] = offset + 1
            progressed = True

            updated_anchor = _updated_anchor_target(
                action=action,
                previous_anchor=anchored_target_by_robot.get(node.executor_slot),
            )
            if updated_anchor:
                anchored_target_by_robot[node.executor_slot] = updated_anchor

            if node_offsets[node.node_id] >= len(program):
                completed_this_step.append(node.node_id)
                if (
                    node.node_kind not in _STATE_CHANGE_NODE_KINDS
                    and not _has_state_change_successor(node.node_id, graph_plan.nodes)
                ):
                    completed_available.add(node.node_id)

        if step_actions:
            micro_steps.append(
                MicroStep(
                    step=len(micro_steps) + 1,
                    actions=step_actions,
                )
            )
        elif not progressed:
            unresolved_nodes = [
                node.node_id
                for node in graph_plan.nodes
                if node.node_id not in completed
            ]
            raise ValueError(
                "Dispatcher could not make progress on the graph plan. "
                f"Unresolved nodes: {unresolved_nodes}"
            )

        completed.update(completed_this_step)

    return _compact_redundant_anchor_moves(micro_steps)


def _is_redundant_move_action(
    *,
    robot: str,
    action: list[str],
    anchored_target_by_robot: dict[str, str],
) -> bool:
    return len(action) >= 2 and action[0] == "Move" and anchored_target_by_robot.get(robot) == action[1]


def _compact_redundant_anchor_moves(micro_steps: list[MicroStep]) -> list[MicroStep]:
    """Drop `Move(x)` when the same robot is already anchored at `x`.

    This keeps execution-layer compaction inside Dispatcher instead of rewriting
    the macro graph. It mainly targets cases such as:
    `... Move(cabinet) -> Reach(cabinet) -> Open(cabinet) -> Move(cabinet)`.
    """
    compacted: list[MicroStep] = []
    anchored_target_by_robot: dict[str, str] = {}

    for step in micro_steps:
        kept_actions: dict[str, list[str]] = {}
        for robot, action in step.actions.items():
            if not action:
                continue
            action_type = action[0]
            target = action[1] if len(action) >= 2 else None

            if (
                action_type == "Move"
                and target
                and anchored_target_by_robot.get(robot) == target
            ):
                continue

            kept_actions[robot] = action
            updated_anchor = _updated_anchor_target(
                action=action,
                previous_anchor=anchored_target_by_robot.get(robot),
            )
            if updated_anchor:
                anchored_target_by_robot[robot] = updated_anchor

        if kept_actions:
            compacted.append(
                MicroStep(
                    step=len(compacted) + 1,
                    actions=kept_actions,
                )
            )

    return compacted


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


def _has_state_change_successor(
    node_id: str,
    nodes: list[AffordanceNode],
) -> bool:
    """Return True if any direct successor of *node_id* is a state-change node."""
    return any(
        node_id in node.predecessors and node.node_kind in _STATE_CHANGE_NODE_KINDS
        for node in nodes
    )


def _resolve_handoff_target(
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
) -> str:
    handoff_target_node = node.required_constraints.get("handoff_target_node")
    if not handoff_target_node:
        raise ValueError(f"Node {node.node_id} is missing handoff_target_node.")
    target_node = node_map.get(str(handoff_target_node))
    if target_node is None or not target_node.executor_slot:
        raise ValueError(
            f"Node {node.node_id} references unresolved handoff_target_node={handoff_target_node}."
        )
    return target_node.executor_slot


def _dispatch_pickup_and_place(
    *,
    item: str | None,
    target_area: str | None,
) -> list[list[str]]:
    if not item or not target_area:
        raise ValueError("pickup_and_place scaffold requires both `item` and `target_area`.")
    primitive_calls = [
        ("move_to", {"target": item}),
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
        ("move_to", {"target": target_area}),
        ("place_item", {"item": item, "target_area": target_area}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_open_then_fetch(
    *,
    container: str | None,
    item: str | None,
    target_area: str | None,
) -> list[list[str]]:
    if not container or not item or not target_area:
        raise ValueError(
            "open_then_fetch scaffold requires `container`, `item`, and `target_area`."
        )
    primitive_calls = [
        ("move_to", {"target": container}),
        ("reach_object", {"item": container}),
        ("open_container", {"container": container}),
        ("move_to", {"target": item}),
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
        ("move_to", {"target": target_area}),
        ("place_item", {"item": item, "target_area": target_area}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_carry_to_container_anchor(
    *,
    item: str | None,
    container: str | None,
) -> list[list[str]]:
    if not item or not container:
        raise ValueError("carry_to_container_anchor requires `item` and `container`.")
    primitive_calls = [
        ("move_to", {"target": item}),
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
        ("move_to", {"target": container}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_open_container_in_place(
    *,
    container: str | None,
) -> list[list[str]]:
    if not container:
        raise ValueError("open_container_in_place requires `container`.")
    primitive_calls = [
        ("reach_object", {"item": container}),
        ("open_container", {"container": container}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_move_reach_open_container(
    *,
    container: str | None,
) -> list[list[str]]:
    if not container:
        raise ValueError("move_reach_open_container requires `container`.")
    primitive_calls = [
        ("move_to", {"target": container}),
        ("reach_object", {"item": container}),
        ("open_container", {"container": container}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_place_held_item(
    *,
    item: str | None,
    target_area: str | None,
) -> list[list[str]]:
    if not item or not target_area:
        raise ValueError("place_held_item requires `item` and `target_area`.")
    return [_primitive_call_to_action("place_item", {"item": item, "target_area": target_area})]


def _dispatch_wash_at_station(
    *,
    item: str | None,
    station: str | None,
) -> list[list[str]]:
    if not item or not station:
        raise ValueError("wash_at_station requires `item` and `station`.")
    primitive_calls = [
        ("move_to", {"target": item}),
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
        ("move_to", {"target": station}),
        ("place_item", {"item": item, "target_area": station}),
        ("interact_with", {"target": station}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_move_then_interact(
    *,
    target: str | None,
) -> list[list[str]]:
    if not target:
        raise ValueError("move_then_interact requires `target`.")
    primitive_calls = [
        ("move_to", {"target": target}),
        ("interact_with", {"target": target}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_interact_in_place(
    *,
    target: str | None,
) -> list[list[str]]:
    if not target:
        raise ValueError("interact_in_place requires `target`.")
    return [_primitive_call_to_action("interact_with", {"target": target})]


def _dispatch_tool_grasp(
    *,
    tool: str | None,
) -> list[list[str]]:
    if not tool:
        raise ValueError("tool_grasp requires `tool`.")
    primitive_calls = [
        ("move_to", {"target": tool}),
        ("reach_object", {"item": tool}),
        ("grasp_item", {"item": tool}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_pickup_only(
    *,
    item: str | None,
) -> list[list[str]]:
    if not item:
        raise ValueError("pickup_only requires `item`.")
    primitive_calls = [
        ("move_to", {"target": item}),
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_move_held_tool_to_target(
    *,
    target_area: str | None,
) -> list[list[str]]:
    if not target_area:
        raise ValueError("move_held_tool_to_target requires `target_area`.")
    return [_primitive_call_to_action("move_to", {"target": target_area})]


def _dispatch_move_held_item_then_place(
    *,
    item: str | None,
    target_area: str | None,
) -> list[list[str]]:
    if not item or not target_area:
        raise ValueError("move_held_item_then_place requires `item` and `target_area`.")
    primitive_calls = [
        ("move_to", {"target": target_area}),
        ("place_item", {"item": item, "target_area": target_area}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_reach_with_held_tool(
    *,
    target: str | None,
) -> list[list[str]]:
    if not target:
        raise ValueError("reach_with_held_tool requires `target`.")
    return [_primitive_call_to_action("reach_object", {"item": target})]


def _dispatch_dual_pick_then_place(
    *,
    item_primary: str | None,
    item_secondary: str | None,
    target_area: str | None,
) -> list[list[str]]:
    if not item_primary or not item_secondary or not target_area:
        raise ValueError(
            "dual_pick_then_place requires `item_primary`, `item_secondary`, and `target_area`."
        )
    primitive_calls = [
        ("move_to", {"target": item_primary}),
        ("reach_object", {"item": item_primary}),
        ("grasp_item", {"item": item_primary}),
        ("move_to", {"target": item_secondary}),
        ("reach_object", {"item": item_secondary}),
        ("grasp_item", {"item": item_secondary}),
        ("move_to", {"target": target_area}),
        ("place_item", {"item": item_primary, "target_area": target_area}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_dual_pick_objects(
    *,
    item_primary: str | None,
    item_secondary: str | None,
) -> list[list[str]]:
    if not item_primary or not item_secondary:
        raise ValueError("dual_pick_objects requires `item_primary` and `item_secondary`.")
    primitive_calls = [
        ("move_to", {"target": item_primary}),
        ("reach_object", {"item": item_primary}),
        ("grasp_item", {"item": item_primary}),
        ("move_to", {"target": item_secondary}),
        ("reach_object", {"item": item_secondary}),
        ("grasp_item", {"item": item_secondary}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_place_dual_pick_at_target(
    *,
    target_area: str | None,
) -> list[list[str]]:
    if not target_area:
        raise ValueError("place_dual_pick_at_target requires `target_area`.")
    primitive_calls = [
        ("move_to", {"target": target_area}),
        ("place_item", {"item": "held_items", "target_area": target_area}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_grasp_item_in_place(
    *,
    item: str | None,
) -> list[list[str]]:
    if not item:
        raise ValueError("grasp_item_in_place requires `item`.")
    primitive_calls = [
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _dispatch_move_carrier_only(
    *,
    carrier: str | None,
) -> list[list[str]]:
    if not carrier:
        raise ValueError("move_carrier_only requires `carrier`.")
    return [_primitive_call_to_action("move_to", {"target": carrier})]


def _dispatch_push_carrier_to_handoff(
    *,
    carrier: str | None,
    handoff_target: str | None,
) -> list[list[str]]:
    if not carrier or not handoff_target:
        raise ValueError("push_carrier_to_handoff requires `carrier` and `handoff_target`.")
    return [
        _primitive_call_to_action(
            "push_carrier",
            {"carrier": carrier, "handoff_target": handoff_target},
        )
    ]


def _dispatch_unload_from_carrier_to_destination(
    *,
    item: str | None,
    target_area: str | None,
) -> list[list[str]]:
    if not item or not target_area:
        raise ValueError(
            "unload_from_carrier_to_destination requires `item` and `target_area`."
        )
    primitive_calls = [
        ("reach_object", {"item": item}),
        ("grasp_item", {"item": item}),
        ("place_item", {"item": item, "target_area": target_area}),
    ]
    return [_primitive_call_to_action(primitive_name, bindings) for primitive_name, bindings in primitive_calls]


def _primitive_call_to_action(primitive_name: str, bindings: dict[str, str]) -> list[str]:
    primitive = get_action_primitive(primitive_name)
    if primitive_name == "move_to":
        return [primitive.benchmark_action, bindings["target"]]
    if primitive_name == "reach_object":
        return [primitive.benchmark_action, bindings["item"]]
    if primitive_name == "grasp_item":
        return [primitive.benchmark_action, bindings["item"]]
    if primitive_name == "place_item":
        return [primitive.benchmark_action, bindings["target_area"]]
    if primitive_name == "open_container":
        return [primitive.benchmark_action, bindings["container"]]
    if primitive_name == "interact_with":
        return [primitive.benchmark_action, bindings["target"]]
    if primitive_name == "push_carrier":
        return [primitive.benchmark_action, bindings["carrier"], bindings["handoff_target"]]
    raise ValueError(f"Unsupported primitive call for dispatcher: {primitive_name}")
