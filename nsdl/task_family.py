from __future__ import annotations

from nsdl.catalog import (
    CAP_MANIPULATION,
    CAP_MOBILITY,
    entity_can_be_destination,
    get_robot_capabilities,
    normalize_entity_name,
)
from nsdl.schema import SceneFacts
from nsdl.task_split_guard import assert_split_safe_macro_motif_names


MACRO_MOTIF_SINGLE_ITEM_RELOCATE = "single_item_relocate"
MACRO_MOTIF_MULTI_ITEM_RELOCATE_PARALLEL = "multi_item_relocate_parallel"
MACRO_MOTIF_MULTI_ITEM_RELOCATE_CHAIN = "multi_item_relocate_chain"
MACRO_MOTIF_RETRIEVE_THEN_RELOCATE = "retrieve_then_relocate"
MACRO_MOTIF_CLEAR_SURFACE_TO_CONTAINER = "clear_surface_to_container"
MACRO_MOTIF_PLACE_THEN_ACTIVATE = "place_then_activate"
MACRO_MOTIF_WASH_THEN_PLACE = "wash_then_place"
MACRO_MOTIF_PREPARE_SURFACE_THEN_TOOL_INTERACT = "prepare_surface_then_tool_interact"
MACRO_MOTIF_SETUP_CONTAINER_THEN_FILL = "setup_container_then_fill"
MACRO_MOTIF_INSPECT_THEN_RESTORE = "inspect_then_restore"
MACRO_MOTIF_INSPECT_AREA_ONLY = "inspect_area_only"
MACRO_MOTIF_CARRIER_PUSH_HANDOFF_TRANSPORT = "carrier_push_handoff_transport"

# Backward-compatible aliases: the runtime still uses the task_family field name in
# GraphPlan/result artifacts, but the values are now split-safe macro motifs.
TASK_FAMILY_RELOCATE_SINGLE_OBJECT = MACRO_MOTIF_SINGLE_ITEM_RELOCATE
TASK_FAMILY_RELOCATE_OBJECTS_PARALLEL = MACRO_MOTIF_MULTI_ITEM_RELOCATE_PARALLEL
TASK_FAMILY_RELOCATE_OBJECTS_SEQUENTIAL = MACRO_MOTIF_MULTI_ITEM_RELOCATE_CHAIN
TASK_FAMILY_RETRIEVE_FROM_CONTAINER_THEN_RELOCATE = MACRO_MOTIF_RETRIEVE_THEN_RELOCATE
TASK_FAMILY_CLEAR_TO_CONTAINER_WITH_OPEN = MACRO_MOTIF_CLEAR_SURFACE_TO_CONTAINER
TASK_FAMILY_PLACE_BEFORE_ACTIVATE_FIXTURE = MACRO_MOTIF_PLACE_THEN_ACTIVATE
TASK_FAMILY_WASH_THEN_RELOCATE = MACRO_MOTIF_WASH_THEN_PLACE
TASK_FAMILY_PREPARE_BOARD_THEN_TOOL_INTERACT = MACRO_MOTIF_PREPARE_SURFACE_THEN_TOOL_INTERACT
TASK_FAMILY_SETUP_CONTAINER_THEN_FILL = MACRO_MOTIF_SETUP_CONTAINER_THEN_FILL
TASK_FAMILY_VERIFY_TARGET_SET_THEN_RESTORE_MISSING = MACRO_MOTIF_INSPECT_THEN_RESTORE
TASK_FAMILY_INSPECT_TARGET = MACRO_MOTIF_INSPECT_AREA_ONLY
TASK_FAMILY_PUSH_ASSISTED_TRANSPORT = MACRO_MOTIF_CARRIER_PUSH_HANDOFF_TRANSPORT

SUPPORTED_MACRO_MOTIF_NAMES = {
    TASK_FAMILY_RELOCATE_SINGLE_OBJECT,
    TASK_FAMILY_RELOCATE_OBJECTS_PARALLEL,
    TASK_FAMILY_RELOCATE_OBJECTS_SEQUENTIAL,
    TASK_FAMILY_RETRIEVE_FROM_CONTAINER_THEN_RELOCATE,
    TASK_FAMILY_CLEAR_TO_CONTAINER_WITH_OPEN,
    TASK_FAMILY_PLACE_BEFORE_ACTIVATE_FIXTURE,
    TASK_FAMILY_WASH_THEN_RELOCATE,
    TASK_FAMILY_PREPARE_BOARD_THEN_TOOL_INTERACT,
    TASK_FAMILY_SETUP_CONTAINER_THEN_FILL,
    TASK_FAMILY_VERIFY_TARGET_SET_THEN_RESTORE_MISSING,
    TASK_FAMILY_INSPECT_TARGET,
    TASK_FAMILY_PUSH_ASSISTED_TRANSPORT,
}

_CLOSED_SET_VERIFICATION_PATTERNS = {
    "closed_set_named_targets",
    "inspect_missing_named_fruits",
}


def _effective_target_items(scene_facts: SceneFacts) -> list[str]:
    return []


def _mobile_manipulator_count(scene_facts: SceneFacts) -> int:
    count = 0
    for robot in scene_facts.robot_inventory:
        capabilities = get_robot_capabilities(robot.robot_type)
        if CAP_MOBILITY in capabilities and CAP_MANIPULATION in capabilities:
            count += 1
    return count


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
        pair_key = (item, target)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        normalized_pairs.append({"item": item, "target": target})
    return normalized_pairs


def _pairs_share_same_destination(scene_facts: SceneFacts) -> bool:
    pairs = _extract_destination_pairs(scene_facts)
    if len(pairs) < 2:
        return False
    return len({pair["target"] for pair in pairs}) == 1


def _has_nested_destination_pair(scene_facts: SceneFacts) -> bool:
    pairs = _extract_destination_pairs(scene_facts)
    if len(pairs) < 2:
        return False
    pair_items = {pair["item"] for pair in pairs}
    return any(pair["target"] in pair_items for pair in pairs)


def _has_activation_destination_pair(scene_facts: SceneFacts) -> bool:
    activation_targets = set(scene_facts.activation_targets)
    if not activation_targets:
        return False
    return any(pair["target"] in activation_targets for pair in _extract_destination_pairs(scene_facts))


def _is_pure_concrete_inspection_task(scene_facts: SceneFacts) -> bool:
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


def has_closed_set_verification_signal(scene_facts: SceneFacts) -> bool:
    task_specific_hints = scene_facts.parser_notes.get("task_specific_hints", {})
    if not isinstance(task_specific_hints, dict):
        task_specific_hints = {}
    pattern = task_specific_hints.get("pattern")

    if isinstance(pattern, str) and pattern.strip() in _CLOSED_SET_VERIFICATION_PATTERNS:
        return True
    return isinstance(task_specific_hints.get("closed_set_verification"), dict)


def has_closed_set_verification_state(scene_facts: SceneFacts) -> bool:
    return False


def get_supported_macro_motif_names() -> set[str]:
    return set(SUPPORTED_MACRO_MOTIF_NAMES)


def classify_macro_motif(scene_facts: SceneFacts) -> str | None:
    destination_pairs = _extract_destination_pairs(scene_facts)
    target_items = _effective_target_items(scene_facts)
    for pair in destination_pairs:
        if pair["item"] not in target_items:
            target_items.append(pair["item"])
    parsed_intents = set(scene_facts.parsed_intents)
    mobile_manipulator_count = _mobile_manipulator_count(scene_facts)
    has_single_target = len(target_items) == 1 and scene_facts.primary_object is not None
    has_multiple_targets = len(target_items) >= 2
    has_closed_set_signal = has_closed_set_verification_signal(scene_facts)
    has_closed_set_state = has_closed_set_verification_state(scene_facts)
    has_same_destination_pairs = _pairs_share_same_destination(scene_facts)
    has_nested_destination_pair = _has_nested_destination_pair(scene_facts)
    has_activation_destination_pair = _has_activation_destination_pair(scene_facts)

    if _is_pure_concrete_inspection_task(scene_facts):
        return TASK_FAMILY_INSPECT_TARGET

    if has_closed_set_state:
        if scene_facts.destination and target_items:
            return TASK_FAMILY_VERIFY_TARGET_SET_THEN_RESTORE_MISSING
        return None

    if "push_transfer" in parsed_intents and has_single_target and scene_facts.destination:
        return TASK_FAMILY_PUSH_ASSISTED_TRANSPORT

    if (
        "wash" in parsed_intents
        and has_single_target
        and scene_facts.destination
        and "sink" in scene_facts.activation_targets
    ):
        return TASK_FAMILY_WASH_THEN_RELOCATE

    if (
        "tool_interaction" in parsed_intents
        and (
            scene_facts.destination == "wooden cutting board"
            or any(pair["target"] == "wooden cutting board" for pair in destination_pairs)
        )
        and scene_facts.tool_items
        and has_single_target
    ):
        return TASK_FAMILY_PREPARE_BOARD_THEN_TOOL_INTERACT

    if (
        "tool_interaction" in parsed_intents
        and (
            scene_facts.destination == "wooden cutting board"
            or any(pair["target"] == "wooden cutting board" for pair in destination_pairs)
        )
        and scene_facts.tool_items
        and has_multiple_targets
    ):
        return TASK_FAMILY_PREPARE_BOARD_THEN_TOOL_INTERACT

    if (
        "activate" in parsed_intents
        and "wash" not in parsed_intents
        and "tool_interaction" not in parsed_intents
        and has_multiple_targets
        and scene_facts.activation_targets
        and (
            scene_facts.destination in scene_facts.activation_targets
            or has_activation_destination_pair
        )
    ):
        return TASK_FAMILY_PLACE_BEFORE_ACTIVATE_FIXTURE

    if (
        has_multiple_targets
        and has_nested_destination_pair
    ):
        return TASK_FAMILY_SETUP_CONTAINER_THEN_FILL

    if has_multiple_targets and (
        scene_facts.destination == "cabinet"
        or (
            destination_pairs
            and all(pair["target"] == "cabinet" for pair in destination_pairs)
        )
    ):
        return TASK_FAMILY_CLEAR_TO_CONTAINER_WITH_OPEN

    if (
        has_single_target
        and scene_facts.destination
        and (
            scene_facts.primary_object in scene_facts.targets_in_cabinet
            or (
                "retrieve" in parsed_intents
                and "counter" not in scene_facts.mentioned_objects
                and "cabinet" in scene_facts.mentioned_objects
            )
        )
        and not has_closed_set_signal
    ):
        return TASK_FAMILY_RETRIEVE_FROM_CONTAINER_THEN_RELOCATE

    if (
        has_multiple_targets
        and (scene_facts.destination or has_same_destination_pairs)
        and mobile_manipulator_count >= 2
        and "parallelize" in parsed_intents
        and (
            has_same_destination_pairs
            or (scene_facts.destination and not has_closed_set_signal)
        )
    ):
        return TASK_FAMILY_RELOCATE_OBJECTS_PARALLEL

    if has_multiple_targets and (
        has_same_destination_pairs
        or (scene_facts.destination and not has_closed_set_signal)
    ):
        return TASK_FAMILY_RELOCATE_OBJECTS_SEQUENTIAL

    simple_intents = {"relocate", "conditional_container_check", "retrieve", "inspect"}
    if (
        has_single_target
        and scene_facts.destination
        and parsed_intents
        and parsed_intents.issubset(simple_intents)
    ):
        return TASK_FAMILY_RELOCATE_SINGLE_OBJECT

    return None


def classify_task_family(scene_facts: SceneFacts) -> str | None:
    return classify_macro_motif(scene_facts)


assert_split_safe_macro_motif_names(get_supported_macro_motif_names())
