from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nsdl.catalog import (
    ACTIVATABLE_TAG,
    CAP_ACTIVATION,
    CAP_CONTAINER_OPERATION,
    CAP_MANIPULATION,
    CAP_MOBILITY,
    CAP_PUSH_TRANSFER,
    CLOSE,
    CONTAINER_TAG,
    CUTTING_SURFACE_TAG,
    GRASP,
    INSPECTABLE_TAG,
    INTERACT,
    MOVE,
    OPEN,
    OPENABLE_TAG,
    PLACE,
    PLACE_TARGET_TAG,
    PORTABLE_ENTITY_TAG,
    PUSH,
    REACH,
    SUPPORT_TAG,
    TOOL_TAG,
    WASH_STATION_TAG,
    EntityProfile,
    RobotProfile,
)
from nsdl.profiles.base import StaticCatalogProvider


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOTS_PY = REPO_ROOT / "lamma-p" / "resources" / "robots.py"
DEFAULT_SCENE_META_DIR = REPO_ROOT / "lamma-p" / "scene_metadata"

CAPABILITY_VOCAB = (
    CAP_MOBILITY,
    CAP_MANIPULATION,
    CAP_CONTAINER_OPERATION,
    CAP_ACTIVATION,
    CAP_PUSH_TRANSFER,
)

_SKILL_TO_CAPS = {
    "GoToObject": {CAP_MOBILITY},
    "ObjectNavExpertAction": {CAP_MOBILITY},
    "PickupObject": {CAP_MANIPULATION},
    "PickUpObject": {CAP_MANIPULATION},
    "PutObject": {CAP_MANIPULATION},
    "DropHandObject": {CAP_MANIPULATION},
    "ThrowObject": {CAP_MANIPULATION, CAP_ACTIVATION},
    "OpenObject": {CAP_CONTAINER_OPERATION},
    "CloseObject": {CAP_CONTAINER_OPERATION},
    "SwitchOn": {CAP_ACTIVATION},
    "SwitchOff": {CAP_ACTIVATION},
    "ToggleObjectOn": {CAP_ACTIVATION},
    "ToggleObjectOff": {CAP_ACTIVATION},
    "SliceObject": {CAP_ACTIVATION},
    "BreakObject": {CAP_ACTIVATION},
    "CookObject": {CAP_ACTIVATION},
    "CleanObject": {CAP_ACTIVATION},
    "PushObject": {CAP_PUSH_TRANSFER},
    "PullObject": {CAP_PUSH_TRANSFER},
}

_MANIPULATION_SKILLS = {
    "PickupObject",
    "PickUpObject",
    "PutObject",
    "DropHandObject",
    "ThrowObject",
}
_CONTAINER_SKILLS = {"OpenObject", "CloseObject"}
_ACTIVATION_SKILLS = {
    "SwitchOn",
    "SwitchOff",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "BreakObject",
    "CookObject",
    "CleanObject",
}
_PUSH_SKILLS = {"PushObject", "PullObject"}

_FOOD_TYPES = {
    "Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato",
}
_TOOL_TYPES = {
    "Knife", "ButterKnife", "Fork", "Spoon", "Spatula", "ScrubBrush",
    "Plunger", "BaseballBat", "TennisRacket",
}
_CUTTING_SURFACE_TYPES = {"CounterTop", "DiningTable", "CoffeeTable", "Desk"}
_WASH_STATION_TYPES = {"Sink", "SinkBasin", "Faucet", "Bathtub", "BathtubBasin"}
_HEAT_OR_TOGGLE_TYPES = {
    "CoffeeMachine", "DeskLamp", "Faucet", "FloorLamp", "LightSwitch",
    "Microwave", "StoveBurner", "StoveKnob", "Television", "Toaster",
}
_SUPPORT_TYPES = {
    "Bed", "Chair", "CoffeeTable", "CounterTop", "Desk", "DiningTable",
    "Shelf", "ShelvingUnit", "SideTable", "Sofa", "Stool", "TVStand",
}
_CONTAINER_TYPES = {
    "Box", "Bowl", "Cabinet", "Cup", "Drawer", "Fridge", "GarbageCan",
    "Microwave", "Mug", "Pan", "Plate", "Pot", "Toaster",
}
_AREA_TYPES = {"Floor"}

_EXTRA_ALIASES = {
    "cell phone": ("cellphone", "phone"),
    "coffee machine": ("coffeemachine",),
    "counter top": ("countertop", "counter"),
    "desk lamp": ("desklamp", "lamp"),
    "floor lamp": ("floorlamp", "lamp"),
    "garbage can": ("garbagecan", "trash can", "trash"),
    "light switch": ("lightswitch", "switch"),
    "remote control": ("remotecontrol", "remote"),
    "side table": ("sidetable", "table"),
    "coffee table": ("coffeetable", "table"),
    "dining table": ("diningtable", "table"),
    "television": ("tv",),
    "fridge": ("refrigerator",),
    "microwave": ("heating appliance",),
    "watering can": ("wateringcan",),
    "wine bottle": ("winebottle", "wine"),
}


@dataclass(frozen=True)
class ResolvedThorObject:
    canonical_name: str
    object_type: str
    object_id: str
    raw: Mapping[str, Any]


def load_robot_database(robots_py: Path | str = DEFAULT_ROBOTS_PY) -> dict[int, dict[str, Any]]:
    path = Path(robots_py)
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    robots: dict[int, dict[str, Any]] = {}
    for name, payload in namespace.items():
        if not isinstance(payload, dict):
            continue
        match = re.fullmatch(r"robot(\d+)", name)
        if match is None:
            continue
        robots[int(match.group(1))] = dict(payload)
    return robots


def build_robot_profiles(
    robots_db: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, RobotProfile], dict[str, tuple[str, ...]], dict[str, int]]:
    profiles: dict[str, RobotProfile] = {}
    aliases: dict[str, tuple[str, ...]] = {}
    priority: dict[str, int] = {}
    for robot_id in sorted(robots_db):
        raw = robots_db[robot_id]
        robot_type = f"robot{robot_id}"
        skills = tuple(str(skill) for skill in raw.get("skills", []) if str(skill).strip())
        caps = tuple(sorted(skills_to_capabilities(skills)))
        primitives = tuple(_skills_to_action_primitives(skills))
        benchmark_actions = tuple(_skills_to_micro_actions(skills))
        ee = _infer_end_effectors(skills)
        mass_limit = raw.get("mass_capacity", raw.get("mass"))
        hard_limits = []
        if mass_limit is not None:
            hard_limits.append(f"Mass capacity/value: {mass_limit}.")
        profiles[robot_type] = RobotProfile(
            robot_type=robot_type,
            capabilities=caps,
            benchmark_actions=benchmark_actions,
            action_primitives=primitives,
            end_effector_num=ee,
            embodiment_summary=(
                f"MAT-THOR robot{robot_id} with THOR skills: {', '.join(skills)}."
            ),
            hard_limits=tuple(hard_limits),
            planning_notes=(
                "Capabilities are projected from per-episode MAT-THOR skill lists.",
            ),
        )
        aliases[robot_type] = (robot_type, f"robot {robot_id}", f"agent {robot_id}")
        priority[robot_type] = robot_id
    return profiles, aliases, priority


def skills_to_capabilities(skills: list[str] | tuple[str, ...]) -> set[str]:
    caps: set[str] = set()
    for skill in skills:
        caps.update(_SKILL_TO_CAPS.get(str(skill), set()))
    return caps


def summarize_robot(
    robot_id: int,
    robot_payload: Mapping[str, Any],
    *,
    slot_id: str | None = None,
) -> dict[str, Any]:
    skills = [str(skill) for skill in robot_payload.get("skills", [])]
    return {
        "slot_id": slot_id or f"R{robot_id}",
        "robot_id": robot_id,
        "robot_type": f"robot{robot_id}",
        "skills": skills,
        "capabilities": sorted(skills_to_capabilities(tuple(skills))),
        "mass": robot_payload.get("mass"),
        "mass_capacity": robot_payload.get("mass_capacity"),
    }


def build_entity_profiles_from_scene_metadata(
    scene_meta_dir: Path | str = DEFAULT_SCENE_META_DIR,
) -> dict[str, EntityProfile]:
    profiles: dict[str, EntityProfile] = {}
    for path in sorted(Path(scene_meta_dir).glob("FloorPlan*.json")):
        try:
            import json

            scene_meta = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for obj in scene_meta.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            profile = entity_profile_from_thor_object(obj)
            existing = profiles.get(profile.canonical_name)
            if existing is None:
                profiles[profile.canonical_name] = profile
                continue
            merged_tags = tuple(sorted(set(existing.tags).union(profile.tags)))
            merged_aliases = tuple(dict.fromkeys([*existing.aliases, *profile.aliases]))
            profiles[profile.canonical_name] = EntityProfile(
                canonical_name=existing.canonical_name,
                entity_type=_merge_entity_type(existing.entity_type, profile.entity_type),
                tags=merged_tags,
                aliases=merged_aliases,
            )
    return profiles


def entity_profile_from_thor_object(obj: Mapping[str, Any]) -> EntityProfile:
    object_type = str(obj.get("objectType") or obj.get("type") or "").strip()
    canonical = canonicalize_object_type(object_type)
    tags = sorted(tags_for_thor_object(obj))
    aliases = aliases_for_object_type(object_type)
    return EntityProfile(
        canonical_name=canonical,
        entity_type=entity_type_for_thor_object(obj),
        tags=tuple(tags or ["scene_object"]),
        aliases=aliases,
    )


def summarize_thor_object(obj: Mapping[str, Any]) -> dict[str, Any]:
    object_type = str(obj.get("objectType") or "").strip()
    return {
        "objectId": obj.get("objectId"),
        "objectType": object_type,
        "canonical_name": canonicalize_object_type(object_type),
        "tags": sorted(tags_for_thor_object(obj)),
        "visible": bool(obj.get("visible", False)),
        "pickupable": bool(obj.get("pickupable", False)),
        "receptacle": bool(obj.get("receptacle", False)),
        "openable": bool(obj.get("openable", False)),
        "isOpen": bool(obj.get("isOpen", False)),
        "toggleable": bool(obj.get("toggleable", False)),
        "isToggled": bool(obj.get("isToggled", False)),
        "sliceable": bool(obj.get("sliceable", False)),
        "isSliced": bool(obj.get("isSliced", False)),
        "cookable": bool(obj.get("cookable", False)),
        "isCooked": bool(obj.get("isCooked", False)),
        "dirtyable": bool(obj.get("dirtyable", False)),
        "isDirty": bool(obj.get("isDirty", False)),
        "breakable": bool(obj.get("breakable", False)),
        "isBroken": bool(obj.get("isBroken", False)),
        "mass": obj.get("mass"),
        "parentReceptacles": list(obj.get("parentReceptacles") or []),
        "receptacleObjectIds": list(obj.get("receptacleObjectIds") or []),
    }


def tags_for_thor_object(obj: Mapping[str, Any]) -> set[str]:
    object_type = str(obj.get("objectType") or "").strip()
    tags: set[str] = set()
    if bool(obj.get("pickupable")):
        tags.update({PORTABLE_ENTITY_TAG, "graspable"})
    if bool(obj.get("receptacle")) or object_type in _CONTAINER_TYPES:
        tags.update({CONTAINER_TAG, SUPPORT_TAG, PLACE_TARGET_TAG})
    if bool(obj.get("openable")):
        tags.update({OPENABLE_TAG, CONTAINER_TAG, PLACE_TARGET_TAG})
    if bool(obj.get("toggleable")) or object_type in _HEAT_OR_TOGGLE_TYPES:
        tags.update({ACTIVATABLE_TAG, "toggleable"})
    if bool(obj.get("sliceable")):
        tags.add("sliceable")
    if bool(obj.get("cookable")):
        tags.add("cookable")
    if bool(obj.get("dirtyable")):
        tags.add("dirtyable")
    if bool(obj.get("isDirty")):
        tags.add("dirty")
    if bool(obj.get("breakable")):
        tags.add("breakable")
    if bool(obj.get("moveable")):
        tags.add("moveable")
    if object_type in _FOOD_TYPES:
        tags.update({"food", "ingredient"})
    if object_type in _TOOL_TYPES:
        tags.update({TOOL_TAG, "interactable_tool", "graspable"})
    if object_type in _CUTTING_SURFACE_TYPES:
        tags.update({SUPPORT_TAG, PLACE_TARGET_TAG, CUTTING_SURFACE_TAG})
    if object_type in _WASH_STATION_TYPES:
        tags.update({SUPPORT_TAG, PLACE_TARGET_TAG, WASH_STATION_TAG, ACTIVATABLE_TAG})
    if object_type in _SUPPORT_TYPES:
        tags.update({SUPPORT_TAG, PLACE_TARGET_TAG})
    if object_type in _AREA_TYPES:
        tags.update({"area", SUPPORT_TAG, PLACE_TARGET_TAG})
    if object_type in {"Blinds", "Curtains", "Window", "Mirror", "Painting", "Poster"}:
        tags.add(INSPECTABLE_TAG)
    if not tags:
        tags.add("scene_object")
    return tags


def entity_type_for_thor_object(obj: Mapping[str, Any]) -> str:
    object_type = str(obj.get("objectType") or "").strip()
    if object_type in _AREA_TYPES:
        return "area"
    if bool(obj.get("pickupable")):
        return "asset"
    if (
        bool(obj.get("receptacle"))
        or bool(obj.get("openable"))
        or bool(obj.get("toggleable"))
        or object_type in _SUPPORT_TYPES
        or object_type in _WASH_STATION_TYPES
        or object_type in _HEAT_OR_TOGGLE_TYPES
    ):
        return "fixture"
    return "asset"


def build_catalog_overlay(
    *,
    scene_meta_dir: Path | str = DEFAULT_SCENE_META_DIR,
    robots_py: Path | str = DEFAULT_ROBOTS_PY,
) -> StaticCatalogProvider:
    robots = load_robot_database(robots_py)
    robot_profiles, aliases, priority = build_robot_profiles(robots)
    return StaticCatalogProvider(
        capability_vocab=CAPABILITY_VOCAB,
        entity_profiles=build_entity_profiles_from_scene_metadata(scene_meta_dir),
        robot_profiles=robot_profiles,
        robot_alias_groups=aliases,
        robot_priority=priority,
    )


def resolve_thor_object(
    name: str | None,
    thor_objects: list[Mapping[str, Any]],
) -> ResolvedThorObject | None:
    if not name:
        return None
    raw_name = str(name).strip()
    if not raw_name:
        return None
    for obj in thor_objects:
        if raw_name == str(obj.get("objectId") or ""):
            object_type = str(obj.get("objectType") or "").strip()
            return ResolvedThorObject(
                canonical_name=canonicalize_object_type(object_type),
                object_type=object_type,
                object_id=raw_name,
                raw=obj,
            )

    canonical = canonicalize_object_ref(raw_name)
    candidates = [
        obj for obj in thor_objects
        if canonicalize_object_type(str(obj.get("objectType") or "")) == canonical
        or canonicalize_object_ref(str(obj.get("canonical_name") or "")) == canonical
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda obj: (
            not bool(obj.get("visible", False)),
            str(obj.get("objectId") or obj.get("objectType") or ""),
        )
    )
    chosen = candidates[0]
    object_type = str(chosen.get("objectType") or "").strip()
    object_id = str(chosen.get("objectId") or object_type).strip()
    return ResolvedThorObject(
        canonical_name=canonicalize_object_type(object_type),
        object_type=object_type,
        object_id=object_id,
        raw=chosen,
    )


def canonicalize_object_ref(text: str | None) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    if "|" in raw:
        raw = raw.split("|", 1)[0]
    return canonicalize_object_type(raw)


def canonicalize_object_type(object_type: str | None) -> str:
    text = str(object_type or "").strip()
    if not text:
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def aliases_for_object_type(object_type: str | None) -> tuple[str, ...]:
    canonical = canonicalize_object_type(object_type)
    compact = canonical.replace(" ", "")
    aliases = [canonical]
    if compact and compact != canonical:
        aliases.append(compact)
    raw = str(object_type or "").strip()
    if raw:
        aliases.append(raw.lower())
    aliases.extend(_EXTRA_ALIASES.get(canonical, ()))
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _skills_to_action_primitives(skills: tuple[str, ...]) -> list[str]:
    primitives: list[str] = []
    skill_set = set(skills)
    if "GoToObject" in skill_set or "ObjectNavExpertAction" in skill_set:
        primitives.append("move_to")
    if skill_set.intersection(_MANIPULATION_SKILLS):
        primitives.extend(["reach_object", "grasp_item", "place_item"])
    if skill_set.intersection(_CONTAINER_SKILLS):
        primitives.extend(["reach_object", "open_container", "close_container"])
    if skill_set.intersection(_ACTIVATION_SKILLS):
        primitives.append("interact_with")
    if skill_set.intersection(_PUSH_SKILLS):
        primitives.append("push_carrier")
    return _dedupe(primitives)


def _skills_to_micro_actions(skills: tuple[str, ...]) -> list[str]:
    actions: list[str] = []
    skill_set = set(skills)
    if "GoToObject" in skill_set or "ObjectNavExpertAction" in skill_set:
        actions.append(MOVE)
    if skill_set.intersection(_MANIPULATION_SKILLS):
        actions.extend([REACH, GRASP, PLACE])
    if skill_set.intersection(_CONTAINER_SKILLS):
        actions.extend([REACH, OPEN, CLOSE])
    if skill_set.intersection(_ACTIVATION_SKILLS):
        actions.append(INTERACT)
    if skill_set.intersection(_PUSH_SKILLS):
        actions.append(PUSH)
    return _dedupe(actions)


def _infer_end_effectors(skills: tuple[str, ...]) -> int:
    skill_set = set(skills)
    if skill_set.intersection(_MANIPULATION_SKILLS | _CONTAINER_SKILLS | {"SliceObject", "BreakObject"}):
        return 1
    return 0


def _merge_entity_type(left: str, right: str) -> str:
    if left == right:
        return left
    order = {"asset": 0, "fixture": 1, "area": 2}
    return max((left, right), key=lambda item: order.get(item, -1))


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


__all__ = [
    "DEFAULT_ROBOTS_PY",
    "DEFAULT_SCENE_META_DIR",
    "ResolvedThorObject",
    "aliases_for_object_type",
    "build_catalog_overlay",
    "build_entity_profiles_from_scene_metadata",
    "canonicalize_object_ref",
    "canonicalize_object_type",
    "entity_profile_from_thor_object",
    "load_robot_database",
    "resolve_thor_object",
    "skills_to_capabilities",
    "summarize_robot",
    "summarize_thor_object",
    "tags_for_thor_object",
]

