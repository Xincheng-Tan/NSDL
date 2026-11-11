from __future__ import annotations

from nsdl.affordance_rules import (
    OBLIGATION_PROCESS_AT_WASH_STATION,
    compute_constraint_obligations,
)
from nsdl.catalog import (
    ACTIVATABLE_TAG,
    CAP_ACTIVATION,
    CAP_MANIPULATION,
    CAP_MOBILITY,
    CAP_PUSH_TRANSFER,
    CONTAINER_TAG,
    CUTTING_SURFACE_TAG,
    OPENABLE_TAG,
    PLACE_TARGET_TAG,
    PUSHABLE_CARRIER_TAG,
    SUPPORT_TAG,
    WASH_STATION_TAG,
    get_entity_profile,
    get_entity_tags,
    get_robot_capabilities,
    normalize_entity_name,
    normalize_robot_name,
)
from nsdl.schema import (
    INTERACTION_TYPE_ACTIVATION,
    INTERACTION_TYPE_CONTAINMENT,
    INTERACTION_TYPE_CUTTING,
    INTERACTION_TYPE_INSPECTION,
    INTERACTION_TYPE_PROCESSING,
    INTERACTION_TYPE_PUSH_TRANSFER,
    AffordanceContext,
    AffordanceInteraction,
    AffordanceInteractionGraph,
    EntityProfileEntry,
    RobotCapabilityEntry,
    SceneFacts,
)
from nsdl.catalog import (
    get_end_effector_num,
)
from nsdl.profiles.base import RuntimeMappingProxy


def _operation_vocab() -> dict[str, str]:
    from nsdl.profiles import get_active_profile

    return get_active_profile().op_vocab.as_dict()


OPERATION_VOCAB = RuntimeMappingProxy(_operation_vocab)


# ---------------------------------------------------------------------------
# New v2: Affordance Context builder (no interaction edges)
# ---------------------------------------------------------------------------


def build_affordance_context(scene_facts: SceneFacts) -> AffordanceContext:
    """Build lightweight Affordance Context from SceneFacts + Catalog.

    Only queries entity tags and robot capabilities — NO interaction edges,
    NO precondition/postcondition reasoning. Pure static knowledge lookup.

    Entity-set sourcing policy (2026-05-08, see docs/slm-io-redesign.md §2.3):
      Allowed sources are observation- or vocabulary-grounded ONLY.
        (a) ``mentioned_objects`` — catalog-vocabulary lookup over instruction
            text + scene refiner additions; this is a pure name match, NOT a
            planning decision.
        (b) ``extras["scene_entities"]`` — benchmark-specific observation feed
            (e.g. MAT-THOR symbolic init dump, VIKI-L2 detector result projected
            by the encoder when available).
      Forbidden sources (would leak the parser's instruction-understanding,
      which is exactly the SLM's job to learn):
        ``target_items``, ``destination_pairs``, ``primary_object``,
        ``destination``, ``tool_items``, ``activation_targets``,
        ``fallback_locations``, ``inspection_targets``, ``source_locations``.
      Those fields stay in SceneFacts because deterministic downstream layers
      (Validator / Micro Planner / Dispatcher) still consume them; they MUST
      NOT seed anything visible to the SLM.
    """
    all_entity_names: list[str] = []

    # (a) Catalog-vocabulary mentions surfaced by the parser/refiner
    for obj in scene_facts.mentioned_objects:
        nm = normalize_entity_name(obj)
        if nm:
            _append_unique(all_entity_names, nm)

    # (b) Benchmark-specific observation feed from extras
    extras_scene = (scene_facts.extras or {}).get("scene_entities")
    if isinstance(extras_scene, list):
        for obj in extras_scene:
            nm = normalize_entity_name(obj if isinstance(obj, str) else obj.get("name"))
            if nm:
                _append_unique(all_entity_names, nm)

    # (c) Last-line vocabulary fallback: catalog mentions found in the raw
    #     instruction string. This is just a name match (same as the SLM could
    #     do from the prompt) and ensures every instruction-named entity gets
    #     its tag annotation in [ENTITIES] without leaking semantic role.
    if scene_facts.instruction:
        from nsdl.catalog import find_entity_mentions  # local import: avoids cycles

        for mention in find_entity_mentions(scene_facts.instruction):
            _append_unique(all_entity_names, mention.canonical_name)

    # Build entity profiles (just tags, no interactions)
    entity_profiles: dict[str, EntityProfileEntry] = {}
    for name in all_entity_names:
        if not name:
            continue
        profile = get_entity_profile(name)
        if profile is not None:
            entity_profiles[name] = EntityProfileEntry(
                canonical_name=name,
                entity_type=profile.entity_type,
                tags=sorted(profile.tags),
            )
        else:
            entity_profiles[name] = EntityProfileEntry(
                canonical_name=name,
                entity_type="unknown",
                tags=[],
                unknown_tags=True,
            )

    # Build robot capability entries
    robot_capabilities: list[RobotCapabilityEntry] = []
    for robot in scene_facts.robot_inventory:
        normalized_type = normalize_robot_name(robot.robot_type) or robot.robot_type
        caps = get_robot_capabilities(normalized_type)
        ee_num = get_end_effector_num(normalized_type)
        robot_capabilities.append(RobotCapabilityEntry(
            slot_id=robot.slot_id,
            robot_type=normalized_type,
            capabilities=sorted(caps),
            end_effectors=ee_num,
        ))

    return AffordanceContext(
        entity_profiles=entity_profiles,
        robot_capabilities=robot_capabilities,
    )


# ---------------------------------------------------------------------------
# Old v1: AIG builder (retained for backward compatibility)
# ---------------------------------------------------------------------------


def build_affordance_interaction_graph(scene_facts: SceneFacts) -> AffordanceInteractionGraph:
    """Compile deterministic affordance context from scene facts.

    The builder performs no LLM calls.  It keeps unknown entities in the graph
    with an ``unknown_tags`` marker so the VLM can still reason from names while
    the validator remains conservative about capability checks.
    """

    destination_pairs = _extract_destination_pairs(scene_facts)
    goal_by_item = _goal_location_by_item(scene_facts, destination_pairs)
    target_items = _effective_target_items(scene_facts)
    for pair in destination_pairs:
        _append_unique(target_items, pair["item"])

    item_nodes: dict[str, dict] = {}
    for item in target_items:
        if item:
            item_nodes[item] = _entity_node_payload(
                item,
                current_location=_infer_current_location(scene_facts, item),
                goal_location=goal_by_item.get(item),
            )
    for tool in scene_facts.tool_items:
        tool_name = normalize_entity_name(tool)
        if tool_name and tool_name not in item_nodes:
            item_nodes[tool_name] = _entity_node_payload(
                tool_name,
                current_location=_infer_current_location(scene_facts, tool_name),
                goal_location=None,
                role="tool",
            )

    fixture_nodes: dict[str, dict] = {}
    fixture_names: list[str] = []
    for value in scene_facts.activation_targets:
        _append_unique(fixture_names, normalize_entity_name(value))
    if scene_facts.destination:
        _append_unique(fixture_names, normalize_entity_name(scene_facts.destination))
    for pair in destination_pairs:
        _append_unique(fixture_names, pair["target"])
    for value in scene_facts.fallback_locations:
        _append_unique(fixture_names, normalize_entity_name(value))
    for value in scene_facts.inspection_targets:
        _append_unique(fixture_names, normalize_entity_name(value))

    for fixture in fixture_names:
        if fixture:
            tags = get_entity_tags(fixture)
            fixture_nodes[fixture] = _entity_node_payload(
                fixture,
                required_caps=sorted(_capabilities_for_fixture_tags(tags)),
            )

    obligations = compute_constraint_obligations(scene_facts)
    wash_required_items = _wash_required_items(scene_facts, obligations, target_items)
    interactions: list[AffordanceInteraction] = []

    for item, target in goal_by_item.items():
        if not item or not target:
            continue
        tags = get_entity_tags(target)
        if not tags or tags.intersection({CONTAINER_TAG, SUPPORT_TAG, PLACE_TARGET_TAG, OPENABLE_TAG}):
            preconditions = [f"holding({item})"]
            triggering_tags = set(tags).intersection(
                {CONTAINER_TAG, SUPPORT_TAG, PLACE_TARGET_TAG, OPENABLE_TAG}
            )
            if OPENABLE_TAG in tags and CONTAINER_TAG in tags:
                preconditions.append(f"open({target})")
            if item in wash_required_items and WASH_STATION_TAG not in tags:
                preconditions.append(f"clean({item})")
            _append_interaction(
                interactions,
                AffordanceInteraction(
                    interaction_type=INTERACTION_TYPE_CONTAINMENT,
                    source_entity=item,
                    target_entity=target,
                    triggering_tags=frozenset(triggering_tags or {"unknown_tags"}),
                    preconditions=preconditions,
                    postconditions=[f"item_at({item}, {target})"],
                    required_capabilities=frozenset({CAP_MOBILITY, CAP_MANIPULATION}),
                ),
            )

    for fixture in scene_facts.activation_targets:
        station = normalize_entity_name(fixture)
        if not station:
            continue
        tags = get_entity_tags(station)
        if WASH_STATION_TAG in tags:
            for item in target_items:
                if not item:
                    continue
                _append_interaction(
                    interactions,
                    AffordanceInteraction(
                        interaction_type=INTERACTION_TYPE_PROCESSING,
                        source_entity=item,
                        target_entity=station,
                        triggering_tags=frozenset({WASH_STATION_TAG}),
                        preconditions=[f"holding({item})", f"at({station})"],
                        postconditions=[f"clean({item})"],
                        required_capabilities=frozenset(
                            {CAP_MOBILITY, CAP_MANIPULATION, CAP_ACTIVATION}
                        ),
                    ),
                )
        elif ACTIVATABLE_TAG in tags:
            source_items = [
                item
                for item, goal in goal_by_item.items()
                if goal == station
            ] or target_items or [station]
            for item in source_items:
                if not item:
                    continue
                _append_interaction(
                    interactions,
                    AffordanceInteraction(
                        interaction_type=INTERACTION_TYPE_ACTIVATION,
                        source_entity=item,
                        target_entity=station,
                        triggering_tags=frozenset({ACTIVATABLE_TAG}),
                        preconditions=[f"item_at({item}, {station})"],
                        postconditions=[f"activated({station})"],
                        required_capabilities=frozenset({CAP_MOBILITY, CAP_ACTIVATION}),
                    ),
                )

    for fixture in fixture_names:
        if not fixture:
            continue
        tags = get_entity_tags(fixture)
        if CUTTING_SURFACE_TAG not in tags or not scene_facts.tool_items:
            continue
        for item in [
            candidate
            for candidate, goal in goal_by_item.items()
            if goal == fixture and candidate not in scene_facts.tool_items
        ] or [item for item in target_items if item not in scene_facts.tool_items]:
            if not item:
                continue
            _append_interaction(
                interactions,
                AffordanceInteraction(
                    interaction_type=INTERACTION_TYPE_CUTTING,
                    source_entity=item,
                    target_entity=fixture,
                    triggering_tags=frozenset({CUTTING_SURFACE_TAG}),
                    preconditions=[
                        f"item_at({item}, {fixture})",
                        f"holding({normalize_entity_name(scene_facts.tool_items[0])})",
                    ],
                    postconditions=[
                        f"tool_effect_applied({normalize_entity_name(scene_facts.tool_items[0])}, {fixture})",
                    ],
                    required_capabilities=frozenset(
                        {CAP_MOBILITY, CAP_MANIPULATION, CAP_ACTIVATION}
                    ),
                ),
            )

    for target in scene_facts.inspection_targets:
        inspection_target = normalize_entity_name(target)
        if not inspection_target:
            continue
        _append_interaction(
            interactions,
            AffordanceInteraction(
                interaction_type=INTERACTION_TYPE_INSPECTION,
                source_entity=inspection_target,
                target_entity=inspection_target,
                triggering_tags=frozenset({"inspectable"}),
                preconditions=[f"at({inspection_target})"],
                postconditions=[f"inspected({inspection_target})"],
                required_capabilities=frozenset({CAP_MOBILITY, CAP_ACTIVATION}),
            ),
        )

    return AffordanceInteractionGraph(
        item_nodes=item_nodes,
        fixture_nodes=fixture_nodes,
        interactions=interactions,
        robot_capability_matrix={
            robot.robot_type: frozenset(get_robot_capabilities(robot.robot_type))
            for robot in scene_facts.robot_inventory
        },
        operation_vocab=dict(OPERATION_VOCAB),
    )


def _extract_destination_pairs(scene_facts: SceneFacts) -> list[dict[str, str]]:
    raw_pairs = scene_facts.destination_pairs or scene_facts.parser_notes.get("destination_pairs", [])
    if not isinstance(raw_pairs, list):
        return []
    normalized_pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in raw_pairs:
        if not isinstance(pair, dict):
            continue
        item = normalize_entity_name(pair.get("item"))
        target = normalize_entity_name(pair.get("target"))
        if not item or not target or item == target:
            continue
        key = (item, target)
        if key in seen:
            continue
        seen.add(key)
        normalized_pairs.append({"item": item, "target": target})
    return normalized_pairs


def _effective_target_items(scene_facts: SceneFacts) -> list[str]:
    return []


def _goal_location_by_item(
    scene_facts: SceneFacts,
    destination_pairs: list[dict[str, str]],
) -> dict[str, str]:
    if destination_pairs:
        return {pair["item"]: pair["target"] for pair in destination_pairs}
    destination = normalize_entity_name(scene_facts.destination)
    if not destination:
        return {}
    return {
        item: destination
        for item in _effective_target_items(scene_facts)
        if item
    }


def _infer_current_location(scene_facts: SceneFacts, item: str) -> str | None:
    if item in {normalize_entity_name(value) for value in scene_facts.targets_in_cabinet}:
        return "cabinet"
    if len(scene_facts.source_locations) == 1:
        return normalize_entity_name(scene_facts.source_locations[0])
    return None


def _entity_node_payload(
    entity: str,
    *,
    current_location: str | None = None,
    goal_location: str | None = None,
    required_caps: list[str] | None = None,
    role: str | None = None,
) -> dict:
    tags = sorted(get_entity_tags(entity))
    payload: dict = {
        "tags": tags,
        "current_location": current_location,
        "goal_location": goal_location,
    }
    if role:
        payload["role"] = role
    if required_caps is not None:
        payload["required_caps"] = list(required_caps)
    if get_entity_profile(entity) is None:
        payload["unknown_tags"] = True
    return payload


def _capabilities_for_fixture_tags(tags: set[str]) -> set[str]:
    capabilities: set[str] = set()
    if tags.intersection({CONTAINER_TAG, SUPPORT_TAG, PLACE_TARGET_TAG, OPENABLE_TAG}):
        capabilities.update({CAP_MOBILITY, CAP_MANIPULATION})
    if OPENABLE_TAG in tags and CONTAINER_TAG in tags:
        capabilities.add(CAP_MANIPULATION)
    if WASH_STATION_TAG in tags or CUTTING_SURFACE_TAG in tags:
        capabilities.update({CAP_MOBILITY, CAP_MANIPULATION, CAP_ACTIVATION})
    elif ACTIVATABLE_TAG in tags:
        capabilities.update({CAP_MOBILITY, CAP_ACTIVATION})
    if PUSHABLE_CARRIER_TAG in tags:
        capabilities.update({CAP_MOBILITY, CAP_PUSH_TRANSFER})
    return capabilities


def _wash_required_items(
    scene_facts: SceneFacts,
    obligations: list,
    target_items: list[str],
) -> set[str]:
    if not any(ob.obligation_type == OBLIGATION_PROCESS_AT_WASH_STATION for ob in obligations):
        return set()
    explicit_items = {
        normalize_entity_name(ob.entity)
        for ob in obligations
        if ob.obligation_type == OBLIGATION_PROCESS_AT_WASH_STATION and ob.entity
    }
    return {item for item in explicit_items if item} or set(target_items)


def _append_interaction(
    interactions: list[AffordanceInteraction],
    interaction: AffordanceInteraction,
) -> None:
    key = (
        interaction.interaction_type,
        interaction.source_entity,
        interaction.target_entity,
        tuple(interaction.preconditions),
    )
    for existing in interactions:
        existing_key = (
            existing.interaction_type,
            existing.source_entity,
            existing.target_entity,
            tuple(existing.preconditions),
        )
        if existing_key == key:
            return
    interactions.append(interaction)


def _append_unique(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)
