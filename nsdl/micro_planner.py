from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools

from nsdl.catalog import (
    CLOSE,
    INTERACT,
    MOVE,
    OPEN,
    PLACE,
    PUSH,
    REACH,
    GRASP,
    entity_can_be_destination,
    entity_requires_open_before_place,
    get_end_effector_num,
    get_entity_profile,
)
from nsdl.schema import AffordanceNode


LOCAL_PREDICATES = {
    "executor_at",
    "reaching",
    "holding",
    "holding_pair",
    "free_end_effectors",
    "in_local_workspace",
}

# Single-valued (functional) fluents: an executor is at exactly one place and
# reaching at most one target. `_has_goal_conflict` already encodes this for the
# regression search (key `(name, None)`); the same invariant has to hold for the
# concrete state we hand to the *next* node. Ground operators cannot enforce it
# on their own because their `del_effects` can only name entities visible to the
# node that built them (`_collect_move_targets`), so an `executor_at` set by an
# earlier node survives into a later node whose target set happens not to mention
# that entity -- and the later node then skips the MOVE it actually needs.
FUNCTIONAL_FLUENTS = frozenset({"executor_at", "reaching"})


@dataclass(frozen=True)
class GroundMicroOperator:
    name: str
    actions: tuple[tuple[str, ...], ...]
    preconditions: tuple[str, ...]
    add_effects: tuple[str, ...]
    del_effects: tuple[str, ...]
    cost: int = 0

    def __post_init__(self) -> None:
        if self.cost <= 0:
            object.__setattr__(self, "cost", max(len(self.actions), 1))


@dataclass(frozen=True)
class PlannedNodeProgram:
    actions: tuple[tuple[str, ...], ...]
    final_state: frozenset[str]


def plan_node_micro_program(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
    predecessor_end_states: dict[str, frozenset[str]],
) -> PlannedNodeProgram:
    if not node.executor_robot_type:
        raise ValueError(f"Node {node.node_id} is missing executor_robot_type for micro planning.")

    resolved_goals = _resolve_goal_predicates(node=node, node_map=node_map)
    if not resolved_goals:
        raise ValueError(
            f"Node {node.node_id} has no goal_predicates and cannot be planned by regression."
        )

    start_state = _build_start_state(
        node=node,
        node_map=node_map,
        predecessor_end_states=predecessor_end_states,
    )
    operators = _build_ground_operators(
        node=node,
        node_map=node_map,
        start_state=start_state,
    )
    operator_plan = _search_regression_plan(
        start_state=start_state,
        goals=tuple(resolved_goals),
        operators=operators,
    )
    final_state = _apply_operator_plan(start_state, operator_plan)
    return PlannedNodeProgram(
        actions=tuple(
            action
            for operator in operator_plan
            for action in operator.actions
        ),
        final_state=frozenset(final_state),
    )


def _build_start_state(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
    predecessor_end_states: dict[str, frozenset[str]],
) -> frozenset[str]:
    state: set[str] = set()

    max_end_effectors = get_end_effector_num(node.executor_robot_type or "")
    state.add(_make_predicate("free_end_effectors", str(max_end_effectors)))
    state.update(node.start_predicates_hint)

    for predecessor_id in node.predecessors:
        predecessor_state = predecessor_end_states.get(predecessor_id, frozenset())
        predecessor = node_map.get(predecessor_id)
        if predecessor is None:
            continue
        if predecessor.executor_slot == node.executor_slot:
            state = {
                predicate
                for predicate in state
                if _predicate_name(predicate) not in LOCAL_PREDICATES
            }
            state.update(predecessor_state)
            continue
        state.update(
            predicate
            for predicate in predecessor_state
            if _predicate_name(predicate) not in LOCAL_PREDICATES
        )

    same_executor_as = node.required_constraints.get("same_executor_as")
    if same_executor_as:
        same_executor_state = predecessor_end_states.get(str(same_executor_as), frozenset())
        state.update(same_executor_state)

    state.update(node.start_predicates_hint)
    state.update(_build_static_facts(node=node, node_map=node_map))
    state = _drop_inaccessible_item_access(state)
    state = _normalize_local_state(state)
    return frozenset(state)


def _resolve_goal_predicates(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
) -> list[str]:
    goals: list[str] = []
    handoff_target = None
    if (
        node.required_constraints.get("handoff_target_node")
        or node.required_constraints.get("handoff_target_slot")
    ):
        handoff_target = _resolve_handoff_target(node=node, node_map=node_map)

    for predicate in node.goal_predicates:
        if "handoff_target" in predicate and handoff_target:
            predicate = predicate.replace("handoff_target", handoff_target)
        if predicate not in goals:
            goals.append(predicate)
    return goals


def _resolve_handoff_target(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
) -> str:
    handoff_target_slot = node.required_constraints.get("handoff_target_slot")
    if isinstance(handoff_target_slot, str) and handoff_target_slot.strip():
        return handoff_target_slot.strip()

    handoff_target_node = node.required_constraints.get("handoff_target_node")
    if not handoff_target_node:
        raise ValueError(f"Node {node.node_id} is missing handoff_target_node.")
    target_node = node_map.get(str(handoff_target_node))
    if target_node is None or not target_node.executor_slot:
        raise ValueError(
            f"Node {node.node_id} references unresolved handoff_target_node={handoff_target_node}."
        )
    return target_node.executor_slot


def _build_static_facts(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
) -> set[str]:
    state: set[str] = set()

    relevant_entities = _collect_relevant_entities(node=node, node_map=node_map)
    movable_items = _collect_movable_items(node)
    container_contents = _collect_container_contents(node)

    for entity in relevant_entities:
        if not entity:
            continue
        state.add(_make_predicate("navigable", entity))
        state.add(_make_predicate("can_access", entity))
        profile = get_entity_profile(entity)
        tags = set(profile.tags) if profile else set()
        if _is_movable_entity_for_node(node=node, entity=entity):
            state.add(_make_predicate("graspable", entity))
        if "container" in tags or entity in container_contents.values():
            state.add(_make_predicate("is_container", entity))
            if "openable" in tags:
                state.add(_make_predicate("is_closed", entity))
        if "pushable_carrier" in tags:
            state.add(_make_predicate("is_pushable", entity))
        if tags.intersection({"activatable", "inspectable", "wash_station", "interactable_tool"}):
            state.add(_make_predicate("can_interact", entity))
        if "wash_station" in tags:
            state.add(_make_predicate("wash_station", entity))

    for item in movable_items:
        for target in relevant_entities:
            if target and entity_can_be_destination(target):
                state.add(_make_predicate("can_place", item, target))

    for item, container in container_contents.items():
        state.add(_make_predicate("inside", item, container))

    if node.required_action_primitives and "push_carrier" in node.required_action_primitives:
        handoff_target = _resolve_handoff_target(node=node, node_map=node_map)
        state.add(_make_predicate("reachable_handoff", handoff_target))

    return state


def _drop_inaccessible_item_access(state: set[str]) -> set[str]:
    inaccessible_items = {
        _predicate_args(predicate)[0]
        for predicate in state
        if _predicate_name(predicate) == "inside" and len(_predicate_args(predicate)) == 2
    }
    return {
        predicate
        for predicate in state
        if not (
            _predicate_name(predicate) == "can_access"
            and _predicate_args(predicate)
            and _predicate_args(predicate)[0] in inaccessible_items
        )
    }


def _normalize_local_state(state: set[str]) -> set[str]:
    free_effectors = [
        predicate
        for predicate in state
        if _predicate_name(predicate) == "free_end_effectors" and _predicate_args(predicate)
    ]
    if len(free_effectors) > 1:
        state.difference_update(free_effectors)
        state.add(
            min(
                free_effectors,
                key=lambda predicate: int(_predicate_args(predicate)[0]),
            )
        )
    return state


def _build_ground_operators(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
    start_state: frozenset[str],
) -> list[GroundMicroOperator]:
    operators: list[GroundMicroOperator] = []
    allowed_primitives = set(node.required_action_primitives)
    max_end_effectors = get_end_effector_num(node.executor_robot_type or "")

    # Fallback: when the goal requires holding an item but start_state does not
    # provide it (e.g. incomplete predecessor chain), auto-extend the primitive
    # set so that transport/grasp operators are available to the planner.
    has_holding = any(_predicate_name(p) == "holding" for p in start_state)
    if not has_holding and not {"reach_object", "grasp_item"}.issubset(allowed_primitives):
        needs_holding = any(
            _predicate_name(g) in {"item_at", "tool_effect_applied", "holding"}
            for g in node.goal_predicates
        )
        if needs_holding:
            allowed_primitives |= {"move_to", "reach_object", "grasp_item"}

    reachable_targets = sorted(_collect_reachable_targets(node))

    # Tool-effect goals (e.g. cut/slice a fruit on a board) require the executor
    # to navigate to and reach the destination surface before INTERACT, even
    # when it already holds the tool from a predecessor acquire. In that case
    # `holding` is already present so the holding auto-extend above does not
    # fire, and `_collect_reachable_targets` omits `node.destination`, so no
    # `reach_at:<destination>` operator is generated and the regression search
    # cannot derive `reaching(destination)`. Make the movement primitives
    # available and treat the tool destination as a reachable target so the
    # `tool_interaction` operator can fire. This is grounded in execution
    # semantics, not task-name specialization.
    tool_destination = node.destination
    if node.node_kind == "tool_interaction" and tool_destination:
        allowed_primitives |= {"move_to", "reach_object"}
        if tool_destination not in reachable_targets:
            reachable_targets = sorted(set(reachable_targets) | {tool_destination})

    movable_items = sorted(_collect_movable_items(node))
    place_pairs = sorted(_collect_place_pairs(node))
    container_contents = _collect_container_contents(node)

    if "move_to" in allowed_primitives:
        for target in sorted(_collect_move_targets(node)):
            operators.append(
                GroundMicroOperator(
                    name=f"move_to:{target}",
                    actions=((MOVE, target),),
                    preconditions=(_make_predicate("navigable", target),),
                    add_effects=(_make_predicate("executor_at", target),),
                    del_effects=tuple(
                        itertools.chain(
                            (
                                _make_predicate("executor_at", other)
                                for other in _collect_move_targets(node)
                                if other != target
                            ),
                            (
                                _make_predicate("reaching", other)
                                for other in reachable_targets
                            ),
                        )
                    ),
                )
            )

    if "reach_object" in allowed_primitives:
        for target in reachable_targets:
            operators.extend(
                [
                    GroundMicroOperator(
                        name=f"reach_at:{target}",
                        actions=((REACH, target),),
                        preconditions=(
                            _make_predicate("executor_at", target),
                            _make_predicate("can_access", target),
                        ),
                        add_effects=(_make_predicate("reaching", target),),
                        del_effects=tuple(
                            _make_predicate("reaching", other)
                            for other in reachable_targets
                            if other != target
                        ),
                    ),
                    GroundMicroOperator(
                        name=f"reach_local:{target}",
                        actions=((REACH, target),),
                        preconditions=(
                            _make_predicate("in_local_workspace", target),
                            _make_predicate("can_access", target),
                        ),
                        add_effects=(_make_predicate("reaching", target),),
                        del_effects=tuple(
                            _make_predicate("reaching", other)
                            for other in reachable_targets
                            if other != target
                        ),
                    ),
                ]
            )

    if "grasp_item" in allowed_primitives:
        for item in movable_items:
            for free_count in range(1, max_end_effectors + 1):
                operators.append(
                    GroundMicroOperator(
                        name=f"grasp_item:{item}:{free_count}",
                        actions=((GRASP, item),),
                        preconditions=(
                            _make_predicate("reaching", item),
                            _make_predicate("graspable", item),
                            _make_predicate("free_end_effectors", str(free_count)),
                        ),
                        add_effects=(
                            _make_predicate("holding", item),
                            _make_predicate("free_end_effectors", str(free_count - 1)),
                        ),
                        del_effects=(
                            _make_predicate("reaching", item),
                            _make_predicate("free_end_effectors", str(free_count)),
                        ),
                    )
                )

        dual_items = _collect_dual_items(node)
        holding_pair_goal = None
        if len(dual_items) == 2:
            holding_pair_goal = _make_predicate("holding_pair", dual_items[0], dual_items[1])
        if (
            holding_pair_goal
            and (
                holding_pair_goal in node.goal_predicates
                or any(_predicate_name(predicate) == "batched_dropoff" for predicate in node.goal_predicates)
            )
            and max_end_effectors >= len(dual_items)
            and "move_to" in allowed_primitives
            and "reach_object" in allowed_primitives
        ):
            remaining_free = max_end_effectors - len(dual_items)
            item_primary, item_secondary = dual_items
            operators.append(
                GroundMicroOperator(
                    name=f"collect_dual_items:{item_primary}:{item_secondary}",
                    actions=(
                        (MOVE, item_primary),
                        (REACH, item_primary),
                        (GRASP, item_primary),
                        (MOVE, item_secondary),
                        (REACH, item_secondary),
                        (GRASP, item_secondary),
                    ),
                    preconditions=(
                        _make_predicate("navigable", item_primary),
                        _make_predicate("can_access", item_primary),
                        _make_predicate("graspable", item_primary),
                        _make_predicate("navigable", item_secondary),
                        _make_predicate("can_access", item_secondary),
                        _make_predicate("graspable", item_secondary),
                        _make_predicate("free_end_effectors", str(max_end_effectors)),
                    ),
                    add_effects=(
                        _make_predicate("holding", item_primary),
                        _make_predicate("holding", item_secondary),
                        holding_pair_goal,
                        _make_predicate("executor_at", item_secondary),
                        _make_predicate("free_end_effectors", str(remaining_free)),
                    ),
                    del_effects=tuple(
                        [
                            _make_predicate("free_end_effectors", str(max_end_effectors)),
                        ]
                        + [
                            _make_predicate("executor_at", other)
                            for other in _collect_move_targets(node)
                            if other != item_secondary
                        ]
                        + [
                            _make_predicate("reaching", item_primary),
                            _make_predicate("reaching", item_secondary),
                        ]
                    ),
                )
            )

    if "place_item" in allowed_primitives:
        if {"move_to", "reach_object", "grasp_item"}.issubset(allowed_primitives):
            for item, target in place_pairs:
                transport_preconditions = [
                    _make_predicate("navigable", item),
                    _make_predicate("can_access", item),
                    _make_predicate("graspable", item),
                    _make_predicate("can_place", item, target),
                    _make_predicate("navigable", target),
                    _make_predicate("free_end_effectors", str(max_end_effectors)),
                ]
                if entity_requires_open_before_place(target):
                    transport_preconditions.append(_make_predicate("is_open", target))
                operators.append(
                    GroundMicroOperator(
                        name=f"transport_item_to_target:{item}:{target}",
                        actions=(
                            (MOVE, item),
                            (REACH, item),
                            (GRASP, item),
                            (MOVE, target),
                            (PLACE, target),
                        ),
                        preconditions=tuple(transport_preconditions),
                        add_effects=(
                            _make_predicate("item_at", item, target),
                            _make_predicate("executor_at", target),
                        ),
                        del_effects=tuple(
                            [_make_predicate("holding", item)]
                            + [
                                _make_predicate("executor_at", other)
                                for other in _collect_move_targets(node)
                                if other != target
                            ]
                            + [_make_predicate("reaching", item)]
                        ),
                    )
                )

        for item, target in place_pairs:
            preconditions = [
                _make_predicate("holding", item),
                _make_predicate("can_place", item, target),
            ]
            if entity_requires_open_before_place(target):
                preconditions.append(_make_predicate("is_open", target))
            for free_count in range(0, max_end_effectors):
                operators.extend(
                    [
                        GroundMicroOperator(
                            name=f"place_item:{item}:{target}:{free_count}",
                            actions=((PLACE, target),),
                            preconditions=tuple(
                                preconditions + [_make_predicate("executor_at", target), _make_predicate("free_end_effectors", str(free_count))]
                            ),
                            add_effects=(
                                _make_predicate("item_at", item, target),
                                _make_predicate("executor_at", target),
                                _make_predicate("free_end_effectors", str(free_count + 1)),
                            ),
                            del_effects=(
                                _make_predicate("holding", item),
                                _make_predicate("free_end_effectors", str(free_count)),
                            ),
                        ),
                        GroundMicroOperator(
                            name=f"place_item_local:{item}:{target}:{free_count}",
                            actions=((PLACE, target),),
                            preconditions=tuple(
                                preconditions
                                + [
                                    _make_predicate("in_local_workspace", target),
                                    _make_predicate("free_end_effectors", str(free_count)),
                                ]
                            ),
                            add_effects=(
                                _make_predicate("item_at", item, target),
                                _make_predicate("executor_at", target),
                                _make_predicate("free_end_effectors", str(free_count + 1)),
                            ),
                            del_effects=(
                                _make_predicate("holding", item),
                                _make_predicate("free_end_effectors", str(free_count)),
                            ),
                        ),
                    ]
                )

        dual_items = _collect_dual_items(node)
        target_area = node.object_bindings.get("target_area", node.destination)
        batched_dropoff_goal = None
        if len(dual_items) == 2 and target_area:
            batched_dropoff_goal = _make_predicate(
                "batched_dropoff",
                dual_items[0],
                dual_items[1],
                target_area,
            )
        if (
            dual_items
            and target_area
            and batched_dropoff_goal
            and batched_dropoff_goal in node.goal_predicates
            and max_end_effectors >= len(dual_items)
            and "move_to" in allowed_primitives
        ):
            remaining_free = max_end_effectors - len(dual_items)
            extra_preconditions = [
                _make_predicate("holding_pair", dual_items[0], dual_items[1]),
                _make_predicate("free_end_effectors", str(remaining_free)),
                _make_predicate("navigable", target_area),
            ] + [
                _make_predicate("can_place", item, target_area)
                for item in dual_items
            ]
            if entity_requires_open_before_place(target_area):
                extra_preconditions.append(_make_predicate("is_open", target_area))
            operators.append(
                GroundMicroOperator(
                    name=f"place_dual_items:{target_area}",
                    actions=(
                        (MOVE, target_area),
                        (PLACE, target_area),
                    ),
                    preconditions=tuple(extra_preconditions),
                    add_effects=tuple(
                        [_make_predicate("item_at", item, target_area) for item in dual_items]
                        + [
                            batched_dropoff_goal,
                            _make_predicate("executor_at", target_area),
                            _make_predicate("free_end_effectors", str(max_end_effectors)),
                        ]
                    ),
                    del_effects=tuple(
                        [_make_predicate("holding", item) for item in dual_items]
                        + [
                            _make_predicate("holding_pair", dual_items[0], dual_items[1]),
                            _make_predicate("free_end_effectors", str(remaining_free)),
                        ]
                        + [
                            _make_predicate("executor_at", other)
                            for other in _collect_move_targets(node)
                            if other != target_area
                        ]
                    ),
                )
            )

    if "open_container" in allowed_primitives:
        for container in sorted(_collect_container_targets(node)):
            contained_items = [
                item
                for item, candidate_container in container_contents.items()
                if candidate_container == container
            ]
            add_effects = [
                _make_predicate("is_open", container),
                *[_make_predicate("can_access", item) for item in contained_items],
            ]
            del_effects = (_make_predicate("is_closed", container),)
            for free_count in range(1, max_end_effectors + 1):
                operators.append(
                    GroundMicroOperator(
                        name=f"open_container:{container}:{free_count}",
                        actions=((OPEN, container),),
                        preconditions=(
                            _make_predicate("reaching", container),
                            _make_predicate("is_container", container),
                            _make_predicate("is_closed", container),
                            _make_predicate("free_end_effectors", str(free_count)),
                        ),
                        add_effects=tuple(add_effects),
                        del_effects=del_effects,
                    )
                )

    if "close_container" in allowed_primitives:
        for container in sorted(_collect_container_targets(node)):
            for free_count in range(1, max_end_effectors + 1):
                operators.append(
                    GroundMicroOperator(
                        name=f"close_container:{container}:{free_count}",
                        actions=((CLOSE, container),),
                        preconditions=(
                            _make_predicate("reaching", container),
                            _make_predicate("is_container", container),
                            _make_predicate("is_open", container),
                            _make_predicate("free_end_effectors", str(free_count)),
                        ),
                        add_effects=(_make_predicate("is_closed", container),),
                        del_effects=(_make_predicate("is_open", container),),
                    )
                )

    if "interact_with" in allowed_primitives:
        operators.extend(_build_interaction_operators(node=node))

    if "push_carrier" in allowed_primitives:
        carrier = node.object_bindings.get("carrier", node.object_name)
        handoff_target = _resolve_handoff_target(node=node, node_map=node_map)
        if carrier and handoff_target:
            operators.append(
                GroundMicroOperator(
                    name=f"push_carrier:{carrier}:{handoff_target}",
                    actions=((PUSH, carrier, handoff_target),),
                    preconditions=(
                        _make_predicate("executor_at", carrier),
                        _make_predicate("is_pushable", carrier),
                        _make_predicate("reachable_handoff", handoff_target),
                    ),
                    add_effects=(
                        _make_predicate("carrier_at", carrier, handoff_target),
                        _make_predicate("executor_at", carrier),
                    ),
                    del_effects=tuple(
                        _make_predicate("carrier_at", carrier, other)
                        for other in _collect_move_targets(node)
                        if other != handoff_target
                    ),
                )
            )

    return operators


def _build_interaction_operators(node: AffordanceNode) -> list[GroundMicroOperator]:
    operators: list[GroundMicroOperator] = []
    target = node.object_bindings.get("target", node.object_name)
    tool = node.object_name or node.object_bindings.get("item")
    destination = node.destination

    if node.node_kind in {"inspect_object", "inspect_presence"} and target:
        operators.extend(
            [
                GroundMicroOperator(
                    name=f"inspect_target:{target}",
                    actions=((INTERACT, target),),
                    preconditions=(
                        _make_predicate("executor_at", target),
                        _make_predicate("can_interact", target),
                    ),
                    add_effects=(_make_predicate("inspected", target),),
                    del_effects=(),
                ),
                GroundMicroOperator(
                    name=f"inspect_target_local:{target}",
                    actions=((INTERACT, target),),
                    preconditions=(
                        _make_predicate("in_local_workspace", target),
                        _make_predicate("can_interact", target),
                    ),
                    add_effects=(_make_predicate("inspected", target),),
                    del_effects=(),
                ),
            ]
        )

    if node.node_kind == "activate_fixture" and target:
        operators.extend(
            [
                GroundMicroOperator(
                    name=f"activate_target:{target}",
                    actions=((INTERACT, target),),
                    preconditions=(
                        _make_predicate("executor_at", target),
                        _make_predicate("can_interact", target),
                    ),
                    add_effects=(_make_predicate("activated", target),),
                    del_effects=(),
                ),
                GroundMicroOperator(
                    name=f"activate_target_local:{target}",
                    actions=((INTERACT, target),),
                    preconditions=(
                        _make_predicate("in_local_workspace", target),
                        _make_predicate("can_interact", target),
                    ),
                    add_effects=(_make_predicate("activated", target),),
                    del_effects=(),
                ),
            ]
        )

    if node.node_kind == "wash_object":
        item = node.object_bindings.get("item", node.object_name)
        station = node.object_bindings.get("station", node.destination)
        if item and station:
            operators.append(
                GroundMicroOperator(
                    name=f"wash_item_at_station:{item}:{station}",
                    actions=((INTERACT, station),),
                    preconditions=(
                        _make_predicate("item_at", item, station),
                        _make_predicate("executor_at", station),
                        _make_predicate("wash_station", station),
                        _make_predicate("can_interact", station),
                    ),
                    add_effects=(_make_predicate("clean", item),),
                    del_effects=(),
                )
            )

    if node.node_kind == "tool_interaction" and tool and destination:
        operators.extend(
            [
                GroundMicroOperator(
                    name=f"use_tool_on_target_reaching:{tool}:{destination}",
                    actions=((INTERACT, tool),),
                    preconditions=(
                        _make_predicate("holding", tool),
                        _make_predicate("reaching", destination),
                    ),
                    add_effects=(_make_predicate("tool_effect_applied", tool, destination),),
                    del_effects=(),
                ),
            ]
        )

    return operators


def _search_regression_plan(
    *,
    start_state: frozenset[str],
    goals: tuple[str, ...],
    operators: list[GroundMicroOperator],
) -> list[GroundMicroOperator]:
    goal_state = frozenset(goal for goal in goals if goal)
    if goal_state.issubset(start_state):
        return []

    achievers: dict[str, list[GroundMicroOperator]] = {}
    for operator in operators:
        for effect in operator.add_effects:
            achievers.setdefault(effect, []).append(operator)

    counter = itertools.count()
    queue: list[tuple[int, int, int, frozenset[str], tuple[GroundMicroOperator, ...]]] = []
    heapq.heappush(
        queue,
        (
            _goal_distance(goal_state, start_state),
            0,
            next(counter),
            goal_state,
            (),
        ),
    )
    best_cost: dict[frozenset[str], int] = {goal_state: 0}

    while queue:
        _priority, cost, _order, current_goals, reversed_plan = heapq.heappop(queue)
        if current_goals.issubset(start_state):
            return list(reversed(reversed_plan))

        unsatisfied = [goal for goal in current_goals if goal not in start_state]
        target_goal = _select_goal_with_min_achievers(unsatisfied, achievers)
        candidate_ops = achievers.get(target_goal, [])
        if not candidate_ops:
            continue

        for operator in candidate_ops:
            if any(
                deleted in current_goals and deleted not in operator.add_effects
                for deleted in operator.del_effects
            ):
                continue
            regressed_goals = frozenset(
                (current_goals - set(operator.add_effects)).union(operator.preconditions)
            )
            if _has_goal_conflict(regressed_goals):
                continue
            next_cost = cost + operator.cost
            if next_cost >= best_cost.get(regressed_goals, 10**9):
                continue
            best_cost[regressed_goals] = next_cost
            heapq.heappush(
                queue,
                (
                    next_cost + _goal_distance(regressed_goals, start_state),
                    next_cost,
                    next(counter),
                    regressed_goals,
                    reversed_plan + (operator,),
                ),
            )

    raise ValueError(
        "Regression micro planner could not derive a valid atomic program. "
        f"goal_state={sorted(goal_state)} start_state={sorted(start_state)}"
    )


def _select_goal_with_min_achievers(
    goals: list[str],
    achievers: dict[str, list[GroundMicroOperator]],
) -> str:
    return min(
        goals,
        key=lambda goal: (len(achievers.get(goal, [])), goal),
    )


def _goal_distance(goals: frozenset[str], start_state: frozenset[str]) -> int:
    return len([goal for goal in goals if goal not in start_state])


def _has_goal_conflict(goals: frozenset[str]) -> bool:
    unique_values: dict[tuple[str, str | None], str] = {}
    for goal in goals:
        name = _predicate_name(goal)
        args = _predicate_args(goal)
        if name == "free_end_effectors":
            key = (name, None)
            value = args[0] if args else ""
        elif name in {"executor_at", "reaching"}:
            key = (name, None)
            value = args[0] if args else ""
        elif name in {"item_at", "carrier_at", "inside"}:
            key = (name, args[0] if args else None)
            value = args[1] if len(args) > 1 else ""
        else:
            continue
        if key in unique_values and unique_values[key] != value:
            return True
        unique_values[key] = value
    return False


def _apply_operator_plan(
    start_state: frozenset[str],
    operator_plan: list[GroundMicroOperator],
) -> set[str]:
    state = set(start_state)
    for operator in operator_plan:
        missing = [predicate for predicate in operator.preconditions if predicate not in state]
        if missing:
            raise ValueError(
                f"Operator {operator.name} is not applicable; missing preconditions={missing}."
            )
        state.difference_update(operator.del_effects)
        state.update(operator.add_effects)
        _enforce_functional_fluents(state, operator)
    return state


def _enforce_functional_fluents(
    state: set[str],
    operator: GroundMicroOperator,
) -> None:
    """Drop stale single-valued fluents superseded by this operator's add effects.

    Within one node this is a no-op: the operator's own `del_effects` already
    cover every entity the node knows about. It only bites across node
    boundaries, where a predecessor's `executor_at`/`reaching` would otherwise
    linger and silently satisfy a successor's precondition.
    """
    for name in FUNCTIONAL_FLUENTS:
        added = {
            predicate
            for predicate in operator.add_effects
            if _predicate_name(predicate) == name
        }
        if not added:
            continue
        state.difference_update(
            {
                predicate
                for predicate in state
                if _predicate_name(predicate) == name and predicate not in added
            }
        )


def _collect_relevant_entities(
    *,
    node: AffordanceNode,
    node_map: dict[str, AffordanceNode],
) -> set[str]:
    values = {
        node.object_name,
        node.destination,
        *node.object_bindings.values(),
    }
    if node.required_constraints.get("handoff_target_node"):
        values.add(_resolve_handoff_target(node=node, node_map=node_map))
    return {
        value
        for value in values
        if value is not None and str(value).strip()
    }


def _collect_move_targets(node: AffordanceNode) -> set[str]:
    return {
        entity
        for entity in {
            node.object_name,
            node.destination,
            node.object_bindings.get("item"),
            node.object_bindings.get("item_primary"),
            node.object_bindings.get("item_secondary"),
            node.object_bindings.get("container"),
            node.object_bindings.get("target"),
            node.object_bindings.get("target_area"),
            node.object_bindings.get("station"),
            node.object_bindings.get("carrier"),
        }
        if entity is not None and str(entity).strip()
    }


def _collect_reachable_targets(node: AffordanceNode) -> set[str]:
    return {
        entity
        for entity in {
            node.object_bindings.get("item"),
            node.object_bindings.get("item_primary"),
            node.object_bindings.get("item_secondary"),
            node.object_bindings.get("container"),
            node.object_bindings.get("target"),
            node.object_bindings.get("target_area"),
            node.object_bindings.get("station"),
            node.object_name,
        }
        if entity is not None and str(entity).strip()
    }


def _collect_movable_items(node: AffordanceNode) -> set[str]:
    return {
        entity
        for entity in {
            node.object_bindings.get("item"),
            node.object_bindings.get("item_primary"),
            node.object_bindings.get("item_secondary"),
            node.object_name,
        }
        if entity is not None and str(entity).strip() and _is_movable_entity_for_node(node=node, entity=entity)
    }


def _collect_place_pairs(node: AffordanceNode) -> set[tuple[str, str]]:
    target_area = node.object_bindings.get("target_area", node.destination)
    pairs: set[tuple[str, str]] = set()
    if target_area:
        for item in _collect_movable_items(node):
            pairs.add((item, target_area))
    return pairs


def _collect_container_targets(node: AffordanceNode) -> set[str]:
    return {
        entity
        for entity in {
            node.object_bindings.get("container"),
            node.object_name if _is_openable_container(node.object_name) else None,
        }
        if entity is not None and str(entity).strip()
    }


def _collect_container_contents(node: AffordanceNode) -> dict[str, str]:
    contents: dict[str, str] = {}
    for predicate in node.start_predicates_hint:
        if _predicate_name(predicate) != "inside":
            continue
        args = _predicate_args(predicate)
        if len(args) != 2:
            continue
        contents[args[0]] = args[1]
    return contents


def _collect_dual_items(node: AffordanceNode) -> tuple[str, ...]:
    items = [
        node.object_bindings.get("item_primary"),
        node.object_bindings.get("item_secondary"),
    ]
    return tuple(
        item
        for item in items
        if item is not None and str(item).strip()
    )


def _is_graspable_entity(entity: str | None) -> bool:
    if not entity:
        return False
    profile = get_entity_profile(entity)
    if profile is None:
        return False
    tags = set(profile.tags)
    return "portable" in tags or "graspable" in tags


def _is_movable_entity_for_node(
    *,
    node: AffordanceNode,
    entity: str | None,
) -> bool:
    if not entity:
        return False
    if _is_graspable_entity(entity):
        return True
    manipulation_item_fields = {
        node.object_bindings.get("item"),
        node.object_bindings.get("item_primary"),
        node.object_bindings.get("item_secondary"),
        node.object_name,
    }
    uses_manipulation = bool({"grasp_item", "place_item"} & set(node.required_action_primitives))
    return uses_manipulation and entity in manipulation_item_fields


def _is_openable_container(entity: str | None) -> bool:
    if not entity:
        return False
    profile = get_entity_profile(entity)
    if profile is None:
        return False
    tags = set(profile.tags)
    return "container" in tags and "openable" in tags


def _make_predicate(name: str, *args: str | None) -> str:
    values = [str(arg) for arg in args if arg is not None and str(arg).strip()]
    return f"{name}({', '.join(values)})"


def _predicate_name(predicate: str) -> str:
    return predicate.split("(", 1)[0].strip()


def _predicate_args(predicate: str) -> tuple[str, ...]:
    if "(" not in predicate or not predicate.endswith(")"):
        return ()
    payload = predicate.split("(", 1)[1][:-1].strip()
    if not payload:
        return ()
    return tuple(part.strip() for part in payload.split(","))
