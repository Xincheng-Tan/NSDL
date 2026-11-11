from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from nsdl.adapters.mat_thor.catalog_map import (
    DEFAULT_ROBOTS_PY,
    DEFAULT_SCENE_META_DIR,
    canonicalize_object_ref,
    load_robot_database,
    summarize_robot,
    summarize_thor_object,
)
from nsdl.adapters.mat_thor.dataset_index import MatThorTask
from nsdl.schema import BenchmarkSample, RobotInstance, SceneFacts


GOAL_SIDE_FIELDS = frozenset({
    "object_states",
    "trans",
    "max_trans",
    "min_trans",
    "min_tans",
})

_PUT_HINT = re.compile(r"\b(?:put|place|store|move|bring|set|archive|organize|stage)\b")
_ACTIVATE_HINT = re.compile(r"\b(?:switch|turn|toggle|activate|deactivate|power|secure)\b")
_CUT_HINT = re.compile(r"\b(?:slice|cut|chop|decomposition|prepare ingredient)\b")
_CLEAN_HINT = re.compile(r"\b(?:clean|wash|rinse|deep clean)\b")
_COOK_HINT = re.compile(r"\b(?:cook|microwave|heat|thermal|caffeine)\b")
_BREAK_HINT = re.compile(r"\b(?:break|broken|trash|waste|dispose|disposal)\b")


def encode_task(
    task: MatThorTask,
    *,
    scene_meta_dir: Path | str = DEFAULT_SCENE_META_DIR,
    robots_py: Path | str = DEFAULT_ROBOTS_PY,
) -> tuple[BenchmarkSample, SceneFacts]:
    scene_meta = load_scene_metadata(task.floor_plan, scene_meta_dir=scene_meta_dir)
    robots_db = load_robot_database(robots_py)
    robots = [
        summarize_robot(robot_id, robots_db[robot_id], slot_id=f"R{index + 1}")
        for index, robot_id in enumerate(task.robot_ids)
        if robot_id in robots_db
    ]
    thor_objects = [
        summarize_thor_object(obj)
        for obj in scene_meta.get("objects", []) or []
        if isinstance(obj, dict)
    ]
    raw_observation = {
        "benchmark": "mat_thor",
        "task_id": task.task_id,
        "task": task.instruction,
        "complexity": task.complexity,
        "floor_plan": task.floor_plan,
        "scene": scene_meta.get("scene", f"FloorPlan{task.floor_plan}"),
        "robots": robots,
        "thor_objects": thor_objects,
        "source_file": str(task.source_file),
        "line_number": task.line_number,
        "parse_status": task.parse_status,
        "parse_notes": list(task.parse_notes),
    }
    sample = BenchmarkSample(
        level="mat_thor",
        split=task.complexity.lower(),
        sample_index=task.floor_plan_record_index,
        task_id=task.task_id,
        task_name=task.instruction,
        instruction=task.instruction,
        system_prompt="",
        image_bytes=None,
        robot_inventory={
            robot["slot_id"]: robot["robot_type"]
            for robot in robots
        },
        robot_inventory_scope="available",
        image_path=None,
        ground_truth=_ground_truth_payload(task.record),
        raw_sample=_public_record(task.record),
        raw_observation=raw_observation,
        benchmark="mat_thor",
    )
    scene_facts = scene_facts_from_observation(sample)
    assert_no_goal_leak(scene_facts)
    return sample, scene_facts


def load_scene_metadata(
    floor_plan: int,
    *,
    scene_meta_dir: Path | str = DEFAULT_SCENE_META_DIR,
) -> dict[str, Any]:
    path = Path(scene_meta_dir) / f"FloorPlan{floor_plan}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def scene_facts_from_observation(sample: BenchmarkSample) -> SceneFacts:
    observation = sample.raw_observation or {}
    robots = [
        RobotInstance(
            slot_id=str(robot.get("slot_id")),
            robot_type=str(robot.get("robot_type")),
        )
        for robot in observation.get("robots", []) or []
        if isinstance(robot, dict) and robot.get("slot_id") and robot.get("robot_type")
    ]
    thor_objects = [
        obj for obj in observation.get("thor_objects", []) or []
        if isinstance(obj, dict)
    ]
    scene_object_names = _dedupe(
        str(obj.get("canonical_name"))
        for obj in thor_objects
        if obj.get("canonical_name")
    )
    instruction_mentions = _extract_instruction_mentions(
        sample.instruction,
        thor_objects,
    )
    tags_by_name = {
        str(obj.get("canonical_name")): set(obj.get("tags", []) or [])
        for obj in thor_objects
        if obj.get("canonical_name")
    }

    target_items = [
        name for name in instruction_mentions
        if _is_target_item(name, tags_by_name.get(name, set()))
    ]
    destination_candidates = [
        name for name in instruction_mentions
        if name not in target_items and _is_destination(name, tags_by_name.get(name, set()))
    ]
    activation_targets = [
        name for name in instruction_mentions
        if "activatable" in tags_by_name.get(name, set())
        or "toggleable" in tags_by_name.get(name, set())
    ]
    tool_items = [
        name for name in instruction_mentions
        if "tool" in tags_by_name.get(name, set())
    ]
    if _CUT_HINT.search(sample.instruction.lower()) and "knife" in scene_object_names:
        tool_items = _dedupe([*tool_items, "knife"])

    if not activation_targets and _ACTIVATE_HINT.search(sample.instruction.lower()):
        activation_targets = [
            name for name in destination_candidates
            if "activatable" in tags_by_name.get(name, set())
        ]

    destination = destination_candidates[0] if destination_candidates else None
    destination_pairs = []
    if destination:
        destination_pairs = [
            {"item": item, "target": destination}
            for item in target_items
            if item != destination
        ]

    parsed_intents = _infer_intents(
        sample.instruction,
        target_items=target_items,
        destination=destination,
        activation_targets=activation_targets,
        tool_items=tool_items,
    )

    extras = {
        "benchmark": "mat_thor",
        "task_id": observation.get("task_id") or sample.task_id,
        "complexity": observation.get("complexity"),
        "floor_plan": observation.get("floor_plan"),
        "scene": observation.get("scene"),
        "robots": observation.get("robots", []),
        "thor_objects": thor_objects,
        "parse_status": observation.get("parse_status"),
        "parse_notes": observation.get("parse_notes", []),
    }

    return SceneFacts(
        parser_backend="mat_thor_scene_encoder",
        instruction=sample.instruction,
        image_available=False,
        robot_inventory=robots,
        primary_object=target_items[0] if target_items else None,
        destination=destination,
        destination_pairs=destination_pairs,
        secondary_objects=target_items[1:],
        target_items=target_items,
        destination_candidates=destination_candidates,
        source_locations=[],
        fallback_locations=[],
        targets_in_cabinet=[],
        tool_items=tool_items,
        activation_targets=activation_targets,
        inspection_targets=[],
        inspect_mode=None,
        parsed_intents=parsed_intents,
        mentioned_objects=scene_object_names,
        robot_anchor_status={},
        field_sources={
            "instruction": "MAT-THOR jsonl task field",
            "robot_inventory": "MAT-THOR robot list plus resources/robots.py",
            "mentioned_objects": "AI2-THOR reset metadata objectType list",
            "extras": "AI2-THOR reset metadata summary without goal-side fields",
        },
        parser_notes={
            "stage": "mat_thor_scene_encoder",
            "instruction_mentions": instruction_mentions,
        },
        extras=extras,
    )


class MatThorSceneParser:
    def parse(self, sample: BenchmarkSample) -> SceneFacts:
        if not sample.raw_observation:
            raise ValueError("MAT-THOR sample is missing raw_observation.")
        scene_facts = scene_facts_from_observation(sample)
        assert_no_goal_leak(scene_facts)
        return scene_facts


def assert_no_goal_leak(scene_facts: SceneFacts) -> None:
    for field_name in GOAL_SIDE_FIELDS:
        if _contains_key(scene_facts.extras, field_name) or _contains_key(scene_facts.parser_notes, field_name):
            raise ValueError(f"Goal-side field leaked into SceneFacts: {field_name}")


def _ground_truth_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in GOAL_SIDE_FIELDS
        if key in record
    }


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in GOAL_SIDE_FIELDS
    }


def _extract_instruction_mentions(
    instruction: str,
    thor_objects: list[Mapping[str, Any]],
) -> list[str]:
    normalized_instruction = _normalize_text(instruction)
    matches: list[tuple[int, int, str]] = []
    seen_aliases: set[tuple[str, str]] = set()
    for obj in thor_objects:
        canonical = str(obj.get("canonical_name") or "")
        if not canonical:
            continue
        aliases = {
            canonical,
            canonical.replace(" ", ""),
            canonicalize_object_ref(str(obj.get("objectType") or "")),
            str(obj.get("objectType") or "").lower(),
        }
        for alias in aliases:
            normalized_alias = _normalize_text(alias)
            if not normalized_alias or (canonical, normalized_alias) in seen_aliases:
                continue
            seen_aliases.add((canonical, normalized_alias))
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])")
            for match in pattern.finditer(normalized_instruction):
                matches.append((match.start(), -len(normalized_alias), canonical))
    matches.sort()
    return _dedupe(name for _, _, name in matches)


def _infer_intents(
    instruction: str,
    *,
    target_items: list[str],
    destination: str | None,
    activation_targets: list[str],
    tool_items: list[str],
) -> list[str]:
    text = instruction.lower()
    intents: list[str] = []
    if target_items and (destination or _PUT_HINT.search(text)):
        intents.append("relocate")
    if _CLEAN_HINT.search(text):
        intents.append("wash")
    if _COOK_HINT.search(text):
        intents.append("activate")
    if _CUT_HINT.search(text) or tool_items:
        intents.append("tool_interaction")
    if _ACTIVATE_HINT.search(text) or activation_targets:
        intents.append("activate")
    if _BREAK_HINT.search(text):
        intents.append("tool_interaction")
    return _dedupe(intents)


def _is_target_item(name: str, tags: set[str]) -> bool:
    del name
    return bool(tags.intersection({"portable", "graspable", "food", "ingredient", "tool"}))


def _is_destination(name: str, tags: set[str]) -> bool:
    del name
    return bool(tags.intersection({"place_target", "support", "container", "activatable", "wash_station"}))


def _normalize_text(text: str) -> str:
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _dedupe(items) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _contains_key(payload: Any, key: str) -> bool:
    if isinstance(payload, dict):
        return any(
            str(raw_key) == key or _contains_key(value, key)
            for raw_key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_contains_key(value, key) for value in payload)
    return False


__all__ = [
    "GOAL_SIDE_FIELDS",
    "MatThorSceneParser",
    "assert_no_goal_leak",
    "encode_task",
    "load_scene_metadata",
    "scene_facts_from_observation",
]
