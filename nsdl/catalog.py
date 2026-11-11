from __future__ import annotations

from dataclasses import dataclass

from nsdl.profiles.base import (
    LayeredCatalogProvider,
    RuntimeMappingProxy,
    RuntimeSequenceProxy,
    StaticCatalogProvider,
)


MOVE = "Move"
REACH = "Reach"
GRASP = "Grasp"
PLACE = "Place"
OPEN = "Open"
CLOSE = "Close"
PUSH = "Push"
INTERACT = "Interact"

ACTION_VOCAB = [MOVE, REACH, GRASP, PLACE, OPEN, CLOSE, PUSH, INTERACT]

CAP_MOBILITY = "mobility"
CAP_MANIPULATION = "manipulation"
CAP_CONTAINER_OPERATION = "container_operation"
CAP_ACTIVATION = "activation"
CAP_PUSH_TRANSFER = "push_transfer"
CAP_DUAL_EE = "dual_ee"


@dataclass(frozen=True)
class EntityProfile:
    canonical_name: str
    entity_type: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class EntityMention:
    canonical_name: str
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class ActionPrimitive:
    name: str
    benchmark_action: str
    params: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    preconditions: tuple[str, ...]
    add_effects: tuple[str, ...]
    del_effects: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class RobotProfile:
    robot_type: str
    capabilities: tuple[str, ...]
    benchmark_actions: tuple[str, ...]
    action_primitives: tuple[str, ...]
    end_effector_num: int
    embodiment_summary: str
    hard_limits: tuple[str, ...]
    planning_notes: tuple[str, ...]


@dataclass(frozen=True)
class NodeKindSpec:
    node_kind: str
    description: str
    required_affordances_hard: tuple[str, ...]
    required_affordances_soft: tuple[str, ...]


PORTABLE_ENTITY_TAG = "portable"
PLACE_TARGET_TAG = "place_target"
CONTAINER_TAG = "container"
SUPPORT_TAG = "support"
TOOL_TAG = "tool"
ACTIVATABLE_TAG = "activatable"
INSPECTABLE_TAG = "inspectable"
OPENABLE_TAG = "openable"
WASH_STATION_TAG = "wash_station"
CUTTING_SURFACE_TAG = "cutting_surface"
PUSHABLE_CARRIER_TAG = "pushable_carrier"

ACTION_PRIMITIVES = {
    "move_to": ActionPrimitive(
        name="move_to",
        benchmark_action=MOVE,
        params=("target",),
        required_capabilities=(CAP_MOBILITY,),
        preconditions=("navigable({target})",),
        add_effects=("executor_at({target})",),
        del_effects=("executor_at({from})", "reaching({item})"),
        description="Move the assigned robot to an object, fixture, or destination area.",
    ),
    "reach_object": ActionPrimitive(
        name="reach_object",
        benchmark_action=REACH,
        params=("item",),
        required_capabilities=(CAP_MANIPULATION,),
        preconditions=("executor_at({item})", "graspable({item})", "has_free_end_effector()"),
        add_effects=("reaching({item})",),
        del_effects=(),
        description="Extend the executor to make the target item or container reachable.",
    ),
    "grasp_item": ActionPrimitive(
        name="grasp_item",
        benchmark_action=GRASP,
        params=("item",),
        required_capabilities=(CAP_MANIPULATION,),
        preconditions=("reaching({item})", "graspable({item})", "has_free_end_effector()"),
        add_effects=("holding({item})",),
        del_effects=("reaching({item})", "has_free_end_effector()"),
        description="Grasp a portable item after it has been reached.",
    ),
    "place_item": ActionPrimitive(
        name="place_item",
        benchmark_action=PLACE,
        params=("item", "target_area"),
        required_capabilities=(CAP_MANIPULATION,),
        preconditions=("holding({item})", "executor_at({target_area})", "can_place({item}, {target_area})"),
        add_effects=("item_at({item}, {target_area})", "has_free_end_effector()"),
        del_effects=("holding({item})",),
        description="Place a held item into or onto the specified target area.",
    ),
    "open_container": ActionPrimitive(
        name="open_container",
        benchmark_action=OPEN,
        params=("container",),
        required_capabilities=(CAP_CONTAINER_OPERATION,),
        preconditions=(
            "executor_at({container})",
            "reaching({container})",
            "is_container({container})",
            "is_closed({container})",
            "has_free_end_effector()",
        ),
        add_effects=("is_open({container})", "can_access_contents({container})"),
        del_effects=("is_closed({container})",),
        description="Open a container so its contents become reachable.",
    ),
    "close_container": ActionPrimitive(
        name="close_container",
        benchmark_action=CLOSE,
        params=("container",),
        required_capabilities=(CAP_CONTAINER_OPERATION,),
        preconditions=(
            "executor_at({container})",
            "reaching({container})",
            "is_container({container})",
            "is_open({container})",
            "has_free_end_effector()",
        ),
        add_effects=("is_closed({container})",),
        del_effects=("is_open({container})", "can_access_contents({container})"),
        description="Close a container after interaction is complete.",
    ),
    "interact_with": ActionPrimitive(
        name="interact_with",
        benchmark_action=INTERACT,
        params=("target",),
        required_capabilities=(CAP_ACTIVATION,),
        preconditions=("executor_at({target})", "can_interact({target})"),
        add_effects=("is_activated({target})",),
        del_effects=(),
        description="Perform a generic activation, washing, inspection, or tool-use interaction.",
    ),
    "push_carrier": ActionPrimitive(
        name="push_carrier",
        benchmark_action=PUSH,
        params=("carrier", "handoff_target"),
        required_capabilities=(CAP_MOBILITY, CAP_PUSH_TRANSFER),
        preconditions=("executor_at({carrier})", "is_pushable({carrier})", "reachable_handoff({handoff_target})"),
        add_effects=("carrier_at({carrier}, {handoff_target})", "executor_at({carrier})"),
        del_effects=("carrier_at({carrier}, {source})",),
        description="Push a carrier such as a cardboardbox toward a receiver or destination anchor.",
    ),
}

ACTION_PRIMITIVE_VOCAB = list(ACTION_PRIMITIVES.keys())

NODE_KIND_SPECS = {
    "relocate_object": NodeKindSpec(
        node_kind="relocate_object",
        description="Move one portable item to a support or container target.",
        required_affordances_hard=(CAP_MOBILITY, CAP_MANIPULATION),
        required_affordances_soft=(),
    ),
    "retrieve_from_container": NodeKindSpec(
        node_kind="retrieve_from_container",
        description="Retrieve a portable item from an openable container and relocate it.",
        required_affordances_hard=(CAP_MOBILITY, CAP_MANIPULATION, CAP_CONTAINER_OPERATION),
        required_affordances_soft=(),
    ),
    "open_container": NodeKindSpec(
        node_kind="open_container",
        description="Open a cabinet-like container before retrieval or placement.",
        required_affordances_hard=(CAP_MOBILITY, CAP_CONTAINER_OPERATION),
        required_affordances_soft=(CAP_MANIPULATION,),
    ),
    "close_container": NodeKindSpec(
        node_kind="close_container",
        description="Close a cabinet-like container after interaction is complete.",
        required_affordances_hard=(CAP_MOBILITY, CAP_CONTAINER_OPERATION),
        required_affordances_soft=(CAP_MANIPULATION,),
    ),
    "activate_fixture": NodeKindSpec(
        node_kind="activate_fixture",
        description="Activate or inspect a fixture such as toaster, sink, oven, drawer, or fridge.",
        required_affordances_hard=(CAP_MOBILITY, CAP_ACTIVATION),
        required_affordances_soft=(),
    ),
    "tool_interaction": NodeKindSpec(
        node_kind="tool_interaction",
        description="Use an interacted tool after prerequisite object placement has been satisfied.",
        required_affordances_hard=(CAP_MOBILITY, CAP_MANIPULATION, CAP_ACTIVATION),
        required_affordances_soft=(CAP_DUAL_EE,),
    ),
    "push_transport": NodeKindSpec(
        node_kind="push_transport",
        description="Use a push-capable robot to move a carrier for downstream pickup.",
        required_affordances_hard=(CAP_MOBILITY, CAP_PUSH_TRANSFER),
        required_affordances_soft=(),
    ),
    "inspect_presence": NodeKindSpec(
        node_kind="inspect_presence",
        description="Inspect whether an object or area satisfies a presence constraint.",
        required_affordances_hard=(CAP_MOBILITY, CAP_ACTIVATION),
        required_affordances_soft=(),
    ),
    "inspect_object": NodeKindSpec(
        node_kind="inspect_object",
        description="Move to and directly inspect a concrete object or fixture.",
        required_affordances_hard=(CAP_MOBILITY, CAP_ACTIVATION),
        required_affordances_soft=(),
    ),
    "local_manipulation": NodeKindSpec(
        node_kind="local_manipulation",
        description="Manipulate an item within the current local workspace without navigation.",
        required_affordances_hard=(CAP_MANIPULATION,),
        required_affordances_soft=(),
    ),
    "handoff_node": NodeKindSpec(
        node_kind="handoff_node",
        description="A communication anchor for custody transfer across a capability boundary.",
        required_affordances_hard=(),
        required_affordances_soft=(),
    ),
    "wash_object": NodeKindSpec(
        node_kind="wash_object",
        description="Wash an object at a fixture before downstream relocation.",
        required_affordances_hard=(CAP_MOBILITY, CAP_MANIPULATION, CAP_ACTIVATION),
        required_affordances_soft=(),
    ),
}


def _profile() -> object:
    from nsdl.profiles import get_active_profile

    return get_active_profile()


def _catalog_provider():
    return _profile().catalog


def _static_catalog() -> StaticCatalogProvider:
    provider = _catalog_provider()
    if isinstance(provider, StaticCatalogProvider):
        return provider
    if isinstance(provider, LayeredCatalogProvider):
        return provider._merged_static
    return StaticCatalogProvider(
        capability_vocab=tuple(provider.capability_vocab),
        entity_profiles=dict(provider.entity_profiles),
        robot_profiles=dict(provider.robot_profiles),
        action_primitives=dict(provider.action_primitives),
        node_kind_specs=dict(provider.node_kind_specs),
        robot_priority=dict(provider.robot_priority),
        robot_alias_groups=dict(provider.robot_alias_groups),
    )


CAPABILITY_VOCAB = RuntimeSequenceProxy(
    lambda: tuple(_catalog_provider().capability_vocab)
)
ENTITY_PROFILES = RuntimeMappingProxy(
    lambda: _catalog_provider().entity_profiles
)
ROBOT_PROFILES = RuntimeMappingProxy(
    lambda: _catalog_provider().robot_profiles
)
ENTITY_ALIAS_PAIRS = RuntimeSequenceProxy(
    lambda: _static_catalog().entity_alias_pairs
)
ENTITY_TAGS = RuntimeMappingProxy(
    lambda: _static_catalog().entity_tags
)
AGENT_AVAIL_ACTIONS = RuntimeMappingProxy(
    lambda: _static_catalog().agent_available_actions
)
AGENT_END_EFFECTOR_NUM = RuntimeMappingProxy(
    lambda: _static_catalog().agent_end_effector_num
)
ROBOT_CAPABILITIES = RuntimeMappingProxy(
    lambda: _static_catalog().robot_capabilities
)
ROBOT_PRIORITY = RuntimeMappingProxy(
    lambda: _catalog_provider().robot_priority
)
ROBOT_ALIAS_GROUPS = RuntimeMappingProxy(
    lambda: _catalog_provider().robot_alias_groups
)
ROBOT_ALIAS_TO_CANONICAL = RuntimeMappingProxy(
    lambda: _static_catalog().robot_alias_to_canonical
)


def normalize_catalog_text(text: str) -> str:
    return _catalog_provider().normalize_catalog_text(text)


def normalize_entity_name(text: str | None) -> str | None:
    return _catalog_provider().normalize_entity_name(text)


def normalize_robot_name(text: str | None) -> str | None:
    return _catalog_provider().normalize_robot_name(text)


def find_entity_mentions(text: str) -> list[EntityMention]:
    return _catalog_provider().find_entity_mentions(text)


def get_entity_profile(entity_name: str) -> EntityProfile | None:
    return _catalog_provider().get_entity_profile(entity_name)


def entity_has_tag(entity_name: str, tag: str) -> bool:
    canonical = normalize_entity_name(entity_name)
    return canonical in ENTITY_TAGS and tag in ENTITY_TAGS[canonical]


def entity_has_any_tag(entity_name: str, tags: set[str]) -> bool:
    canonical = normalize_entity_name(entity_name)
    return canonical in ENTITY_TAGS and bool(ENTITY_TAGS[canonical].intersection(tags))


def get_entity_tags(entity_name: str) -> set[str]:
    canonical = normalize_entity_name(entity_name)
    return set(ENTITY_TAGS.get(canonical, set()))


def entity_can_be_destination(entity_name: str) -> bool:
    return entity_has_any_tag(entity_name, {PLACE_TARGET_TAG, SUPPORT_TAG, CONTAINER_TAG})


def entity_requires_open_before_place(entity_name: str) -> bool:
    return entity_has_tag(entity_name, CONTAINER_TAG) and entity_has_tag(entity_name, OPENABLE_TAG)


def is_portable_entity(entity_name: str) -> bool:
    return entity_has_tag(entity_name, PORTABLE_ENTITY_TAG)


def get_robot_capabilities(robot_type: str) -> set[str]:
    return set(ROBOT_CAPABILITIES.get(robot_type, set()))


def get_robot_actions(robot_type: str) -> list[str]:
    return list(AGENT_AVAIL_ACTIONS.get(robot_type, []))


def get_robot_profile(robot_type: str) -> RobotProfile | None:
    return _catalog_provider().get_robot_profile(robot_type)


def get_robot_action_primitives(robot_type: str) -> list[str]:
    profile = get_robot_profile(robot_type)
    return list(profile.action_primitives) if profile else []


def get_end_effector_num(robot_type: str) -> int:
    return int(AGENT_END_EFFECTOR_NUM.get(robot_type, 0))


def get_action_primitive(primitive_name: str) -> ActionPrimitive:
    return ACTION_PRIMITIVES[primitive_name]


def get_node_kind_spec(node_kind: str) -> NodeKindSpec:
    return NODE_KIND_SPECS[node_kind]


def format_action_primitive_summary() -> list[dict[str, object]]:
    return [
        {
            "name": primitive.name,
            "benchmark_action": primitive.benchmark_action,
            "params": list(primitive.params),
            "required_capabilities": list(primitive.required_capabilities),
            "pre": list(primitive.preconditions),
            "add": list(primitive.add_effects),
            "del": list(primitive.del_effects),
            "description": primitive.description,
        }
        for primitive in ACTION_PRIMITIVES.values()
    ]


def infer_supported_node_kinds(robot_type: str) -> list[str]:
    capabilities = get_robot_capabilities(robot_type)
    return [
        spec.node_kind
        for spec in NODE_KIND_SPECS.values()
        if set(spec.required_affordances_hard).issubset(capabilities)
    ]


def format_robot_affordance_catalog(robot_inventory: list[object] | None = None) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if robot_inventory is None:
        inventory_items = [
            {"slot_id": None, "robot_type": robot_type}
            for robot_type in ROBOT_PROFILES
        ]
    else:
        inventory_items = [
            {
                "slot_id": getattr(robot, "slot_id", None),
                "robot_type": getattr(robot, "robot_type", None),
            }
            for robot in robot_inventory
        ]

    for item in inventory_items:
        robot_type = item["robot_type"]
        if not robot_type or robot_type not in ROBOT_PROFILES:
            continue
        profile = get_robot_profile(robot_type)
        if profile is None:
            continue
        entries.append(
            {
                "slot_id": item["slot_id"],
                "robot_type": robot_type,
                "capabilities": list(profile.capabilities),
                "benchmark_actions": list(profile.benchmark_actions),
                "action_primitives": list(profile.action_primitives),
                "supported_node_kinds": infer_supported_node_kinds(robot_type),
                "end_effector_num": profile.end_effector_num,
                "embodiment_summary": profile.embodiment_summary,
                "hard_limits": list(profile.hard_limits),
                "planning_notes": list(profile.planning_notes),
            }
        )
    return entries


def format_node_kind_summary() -> list[dict[str, object]]:
    return [
        {
            "node_kind": spec.node_kind,
            "description": spec.description,
            "required_affordances_hard": list(spec.required_affordances_hard),
            "required_affordances_soft": list(spec.required_affordances_soft),
        }
        for spec in NODE_KIND_SPECS.values()
    ]
