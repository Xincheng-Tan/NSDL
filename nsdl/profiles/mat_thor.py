from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from nsdl.adapters.mat_thor.catalog_map import build_catalog_overlay, canonicalize_object_ref
from nsdl.adapters.mat_thor.plan_emitter import MAT_THOR_PLAN_EMITTER
from nsdl.profiles import register_benchmark_profile
from nsdl.profiles.base import BenchmarkProfile, LayeredCatalogProvider
from nsdl.profiles.viki_l2 import (
    VIKI_L2_CATALOG,
    VIKI_L2_OP_VOCAB,
    VIKI_L2_PROMPT_PROVIDER,
)


MAT_THOR_CATALOG_OVERLAY = build_catalog_overlay()
MAT_THOR_CATALOG = LayeredCatalogProvider(
    base=VIKI_L2_CATALOG,
    overlay=MAT_THOR_CATALOG_OVERLAY,
)


def _mat_thor_scene_lines(scene_facts: Any, affordance_context: Any) -> list[str]:
    del affordance_context
    extras = scene_facts.extras or {}
    floor_plan = extras.get("floor_plan")
    lines = [
        f"Instruction: {scene_facts.instruction}",
        "Benchmark: MAT-THOR",
    ]
    if floor_plan is not None:
        lines.append(f"Floor plan: FloorPlan{floor_plan}")
    complexity = extras.get("complexity")
    if complexity:
        lines.append(f"Task complexity: {complexity}")
    object_counts = _object_type_counts(extras.get("thor_objects", []))
    if object_counts:
        object_parts = [
            f"{name} x{count}" if count > 1 else name
            for name, count in object_counts
        ]
        # Cross-benchmark unified label: matches VIKI-L2 [SCENE] `Observed entities:`.
        lines.append(f"Observed entities: {', '.join(object_parts)}")
    containment = _containment_lines(extras.get("thor_objects", []))
    if containment:
        lines.append("Entity locations: " + "; ".join(containment[:24]))
    return lines


def _mat_thor_robot_lines(scene_facts: Any, affordance_context: Any) -> list[str] | None:
    del affordance_context
    robots = [
        robot for robot in (scene_facts.extras or {}).get("robots", []) or []
        if isinstance(robot, dict)
    ]
    if not robots:
        return None
    parts = []
    for robot in robots:
        slot_id = robot.get("slot_id")
        robot_type = robot.get("robot_type")
        caps = ", ".join(robot.get("capabilities", []) or [])
        skills = ", ".join(robot.get("skills", []) or [])
        parts.append(f"{slot_id}={robot_type} caps[{caps}] skills[{skills}]")
    # Keep label aligned with VIKI-L2 (`Robots:`) but expose THOR skill list which
    # is benchmark-grounded observation, not instruction parsing.
    return [f"Robots: {'; '.join(parts)}"]


MAT_THOR_PROMPT_PROVIDER = replace(
    VIKI_L2_PROMPT_PROVIDER,
    contributors=(
        _mat_thor_scene_lines,
        _mat_thor_robot_lines,
    ),
)


MAT_THOR_PROFILE = BenchmarkProfile(
    name="mat_thor",
    op_vocab=VIKI_L2_OP_VOCAB,
    catalog=MAT_THOR_CATALOG,
    prompt=MAT_THOR_PROMPT_PROVIDER,
    dispatcher_emitter=MAT_THOR_PLAN_EMITTER,
    scene_extras_keys=(
        "floor_plan",
        "robots",
        "thor_objects",
        "complexity",
        "task_id",
    ),
)

register_benchmark_profile("mat_thor", MAT_THOR_PROFILE)


def _object_type_counts(objects: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter = Counter(
        str(obj.get("canonical_name") or canonicalize_object_ref(obj.get("objectType")))
        for obj in objects
        if isinstance(obj, dict) and (obj.get("canonical_name") or obj.get("objectType"))
    )
    return sorted(counter.items())


def _containment_lines(objects: list[dict[str, Any]]) -> list[str]:
    by_id = {
        str(obj.get("objectId")): str(obj.get("canonical_name") or canonicalize_object_ref(obj.get("objectType")))
        for obj in objects
        if isinstance(obj, dict) and obj.get("objectId")
    }
    lines: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if not obj.get("pickupable"):
            continue
        parents = obj.get("parentReceptacles") or []
        if not parents:
            continue
        parent = by_id.get(str(parents[0])) or canonicalize_object_ref(str(parents[0]).split("|", 1)[0])
        child = str(obj.get("canonical_name") or canonicalize_object_ref(obj.get("objectType")))
        if child and parent:
            lines.append(f"{child} in {parent}")
    return lines


__all__ = [
    "MAT_THOR_CATALOG",
    "MAT_THOR_CATALOG_OVERLAY",
    "MAT_THOR_PROFILE",
]

