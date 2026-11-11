from __future__ import annotations

import base64
import hashlib
import imghdr
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from nsdl.catalog import (
    entity_has_tag,
    find_entity_mentions,
    is_portable_entity,
    normalize_entity_name,
)
from nsdl.schema import BenchmarkSample, SceneFacts, StructuredDetections

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_FACTS_REFINER_CACHE_DIR = REPO_ROOT / "outputs" / "scene_facts_refiner_cache"
SCENE_FACTS_REFINER_CACHE_VERSION = 2
_REFINER_MAX_ATTEMPTS_PER_ENDPOINT = 2
_REFINER_INITIAL_BACKOFF_SECONDS = 1.0
_STRONG_STORAGE_DETECTION_MIN_CONFIDENCE = 70

_AREA_REGION_LABELS = [
    "table",
    "kitchen work area",
    "kitchen island area",
    "cabinet",
]

_DEFAULT_DETECTOR_GROUNDED_LABELS = [
    "fetch",
    "stompy",
    "unitree_h1",
    "panda",
    "unitree_go2",
    "anymal_c",
    "apple",
    "banana",
    "bowl",
    "bread",
    "peach",
    "pear",
    "plate",
    "tomato",
    "knife",
    "toaster",
    "wooden cutting board",
    "sink",
]

_PREP_SURFACE_ANCHOR_LABELS = [
    "sink",
    "toaster",
    "knife",
    "wooden cutting board",
]

_SERVING_SURFACE_ANCHOR_LABELS = [
    "bowl",
    "plate",
    "apple",
    "banana",
    "peach",
    "pear",
    "tomato",
]

_STORAGE_CUE_FIELDS = (
    "source_locations",
    "fallback_locations",
    "destination_candidates",
    "inspection_targets",
    "mentioned_objects",
)

_AMBIGUOUS_TRI_FRUIT_LABELS = ("apple", "peach", "tomato")
_TARGET_LOCATION_PREDICATES = ("on_table", "in_cabinet")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _build_refiner_cache_stem(sample: BenchmarkSample) -> str:
    prefix = f"{sample.sample_index:05d}"
    if sample.task_id:
        return f"{prefix}_{sample.task_id}"
    return prefix


def _json_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_entity_list(items: list[str] | None) -> list[str]:
    if items is None:
        return []
    normalized_items = [normalize_entity_name(item) for item in items]
    return _dedupe_preserve_order([item for item in normalized_items if item])


def _normalize_destination_pairs(payload_value: Any) -> list[dict[str, str]]:
    if not isinstance(payload_value, list):
        return []
    normalized_pairs: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in payload_value:
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


def _extract_instruction_named_fruits(instruction: str) -> list[str]:
    named_fruits: list[str] = []
    for mention in find_entity_mentions(instruction):
        if not entity_has_tag(mention.canonical_name, "fruit"):
            continue
        if mention.canonical_name not in named_fruits:
            named_fruits.append(mention.canonical_name)
    return named_fruits


def _scene_facts_payload(scene_facts: SceneFacts) -> dict[str, Any]:
    payload = {
        "destination": scene_facts.destination,
        "destination_pairs": list(scene_facts.destination_pairs),
        "destination_candidates": list(scene_facts.destination_candidates),
        "source_locations": list(scene_facts.source_locations),
        "fallback_locations": list(scene_facts.fallback_locations),
        "tool_items": list(scene_facts.tool_items),
        "activation_targets": list(scene_facts.activation_targets),
        "inspection_targets": list(scene_facts.inspection_targets),
        "inspect_mode": scene_facts.inspect_mode,
        "parsed_intents": list(scene_facts.parsed_intents),
        "mentioned_objects": list(scene_facts.mentioned_objects),
        "robot_anchor_status": dict(scene_facts.robot_anchor_status),
    }
    if scene_facts.extras:
        payload["extras"] = dict(scene_facts.extras)
    return payload


def _normalize_yolo_payload(
    yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any] | list[dict[str, Any]]:
    if yolo_detections is None:
        return []
    if isinstance(yolo_detections, StructuredDetections):
        return dict(yolo_detections.prompt_payload)
    if isinstance(yolo_detections, dict):
        return dict(yolo_detections)
    return list(yolo_detections)


def _extract_supported_detector_labels(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
) -> list[str]:
    if isinstance(yolo_payload, dict):
        supported_labels = yolo_payload.get("supported_labels", [])
        if isinstance(supported_labels, list):
            normalized = [
                str(label).strip()
                for label in supported_labels
                if str(label).strip()
            ]
            if normalized:
                return _dedupe_preserve_order(normalized)
    return list(_DEFAULT_DETECTOR_GROUNDED_LABELS)


def _needs_region_reasoning(initial_scene_facts: SceneFacts) -> bool:
    candidate_fields: list[str] = []
    if initial_scene_facts.destination:
        candidate_fields.append(initial_scene_facts.destination)
    candidate_fields.extend(initial_scene_facts.destination_candidates)
    candidate_fields.extend(initial_scene_facts.source_locations)
    candidate_fields.extend(initial_scene_facts.fallback_locations)
    candidate_fields.extend(initial_scene_facts.inspection_targets)
    candidate_fields.extend(initial_scene_facts.mentioned_objects)
    if any(field in _AREA_REGION_LABELS for field in candidate_fields):
        return True
    instruction = initial_scene_facts.instruction.lower()
    return any(region in instruction for region in _AREA_REGION_LABELS)


def _collect_relevant_regions(initial_scene_facts: SceneFacts) -> list[str]:
    relevant_regions: list[str] = []
    instruction = initial_scene_facts.instruction.lower()
    candidate_fields: list[str] = []
    if initial_scene_facts.destination:
        candidate_fields.append(initial_scene_facts.destination)
    candidate_fields.extend(initial_scene_facts.destination_candidates)
    candidate_fields.extend(initial_scene_facts.source_locations)
    candidate_fields.extend(initial_scene_facts.fallback_locations)
    candidate_fields.extend(initial_scene_facts.inspection_targets)
    candidate_fields.extend(initial_scene_facts.mentioned_objects)

    for region in _AREA_REGION_LABELS:
        if region in candidate_fields or region in instruction:
            relevant_regions.append(region)

    if "table" in relevant_regions:
        for prep_region in ("kitchen work area", "kitchen island area"):
            if prep_region not in relevant_regions:
                relevant_regions.append(prep_region)

    if any(region in relevant_regions for region in ("kitchen work area", "kitchen island area")):
        for prep_region in ("kitchen work area", "kitchen island area"):
            if prep_region not in relevant_regions:
                relevant_regions.append(prep_region)

    if _has_explicit_storage_cue(initial_scene_facts) and "cabinet" not in relevant_regions:
        relevant_regions.append("cabinet")

    return relevant_regions


def _build_region_reasoning_hints(
    initial_scene_facts: SceneFacts,
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _needs_region_reasoning(initial_scene_facts):
        return None

    relevant_regions = _collect_relevant_regions(initial_scene_facts)
    detector_grounded_labels = _extract_supported_detector_labels(yolo_payload)
    prep_surface_anchors = [
        label for label in detector_grounded_labels if label in _PREP_SURFACE_ANCHOR_LABELS
    ]
    serving_surface_anchors = [
        label for label in detector_grounded_labels if label in _SERVING_SURFACE_ANCHOR_LABELS
    ]
    reasoning_rules = [
        (
            "Do not treat missing YOLO detections for "
            + ", ".join(relevant_regions)
            + " as evidence that those regions are absent."
        ),
        (
            "Infer "
            + ", ".join(relevant_regions)
            + " from the whole image plus instruction, using detector-grounded labels only as semantic anchors."
        ),
    ]
    if "table" in relevant_regions:
        reasoning_rules.extend(
            [
                "In most scenes, instruction-level table refers to a single freestanding tabletop near the center of the image; wall-adjacent counters usually do not count as table unless the instruction gives strong contrary evidence.",
                "Assume at most one instruction-level table per scene. If YOLO already provides one reliable table bbox, use that table bbox as the primary geometric anchor instead of remapping nearby wall counters or cabinets to table.",
                "table and kitchen island area are not mutually exclusive labels; the same surface can satisfy both.",
                "Do not turn table-versus-kitchen-island overlap into negative evidence by itself.",
                "If yolo_detections.table_item_overlap is present, use its precomputed overlap ratios and support_like_contained flags as the primary geometric summary for table-support reasoning.",
                "Use bbox containment and center-in-bbox from yolo_detections.grouped_detections[*].instances as generic support-relation evidence only.",
                "For instruction-level table reasoning, require an exposed serving or dining-like surface in the image; a generic prep counter with dishes or fruit is not automatically enough unless the overlap evidence is strong.",
                "Do not convert table support evidence into per-target already_at_destination or visible_elsewhere status.",
            ]
        )
    if any(region in relevant_regions for region in ("kitchen work area", "kitchen island area")):
        reasoning_rules.append(
            "Treat kitchen work area and kitchen island area as prep-surface semantics. If wall-counter versus island is unclear, preserve the initial parse and do not turn that ambiguity into missing-item or cabinet evidence."
        )
    if "cabinet" in relevant_regions:
        reasoning_rules.extend(
            [
                "Cabinet is detector-grounded in the current YOLO model, but likely_in_cabinet still requires explicit storage evidence and stronger support than visible_elsewhere on an open surface.",
                "Do not convert cabinet proximity into per-target likely_in_cabinet status.",
            ]
        )

    return {
        "detector_grounded_labels": detector_grounded_labels,
        "detector_not_supported_for_regions": relevant_regions,
        "detector_capability_note": (
            "YOLO directly grounds detector_grounded_labels. "
            + "It can directly detect table and cabinet in the current model, "
            + "but kitchen work area and kitchen island area still require whole-image semantic reasoning."
        ),
        "anchor_notes": [
            (
                "Use detector-grounded prep anchors only as indirect cues: "
                + ", ".join(prep_surface_anchors)
            )
            if prep_surface_anchors
            else "Use detector-grounded prep anchors only as indirect cues when they are visible.",
            (
                "Use detector-grounded serving anchors only as indirect cues: "
                + ", ".join(serving_surface_anchors)
            )
            if serving_surface_anchors
            else "Use detector-grounded serving anchors only as indirect cues when they are visible.",
            "Area labels still require instruction + full-image semantic reasoning.",
        ],
        "reasoning_rules": reasoning_rules,
    }


def _extract_visible_yolo_fruits(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    instruction_named_targets: list[str] | None = None,
) -> list[dict[str, Any]]:
    fruit_groups: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    def add_group(label: Any, count: Any = 1, regions: Any = None) -> None:
        canonical_label = normalize_entity_name(str(label)) if label is not None else None
        if not canonical_label or canonical_label in seen_labels:
            return
        if not entity_has_tag(canonical_label, "fruit"):
            return
        seen_labels.add(canonical_label)
        normalized_regions = []
        if isinstance(regions, list):
            normalized_regions = [str(region) for region in regions if str(region).strip()]
        fruit_groups.append(
            {
                "label": canonical_label,
                "count": int(count) if str(count).isdigit() else count,
                "regions": normalized_regions,
            }
        )

    corrected_groups = _extract_corrected_detection_groups(
        yolo_payload,
        instruction_named_targets=instruction_named_targets or [],
    )
    for group in corrected_groups:
        add_group(
            group.get("label"),
            count=group.get("count", 1),
            regions=group.get("regions", []),
        )
    return fruit_groups


def _has_explicit_storage_cue(scene_facts: SceneFacts) -> bool:
    storage_fields = (
        scene_facts.source_locations,
        scene_facts.fallback_locations,
        scene_facts.destination_candidates,
        scene_facts.mentioned_objects,
        scene_facts.inspection_targets,
    )
    return any("cabinet" in field_values for field_values in storage_fields)


def _coerce_confidence(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    value_str = str(value).strip()
    if value_str.isdigit():
        return int(value_str)
    return 0


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).strip()
    try:
        return float(value_str)
    except ValueError:
        return 0.0


def _slot_sort_key(slot_id: str) -> tuple[str, int, str]:
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(slot_id).strip())
    if match:
        return (match.group(1), int(match.group(2)), str(slot_id))
    return (str(slot_id), 0, str(slot_id))


def _iter_detection_groups(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(yolo_payload, dict):
        for field_name in ("visible_entities", "grouped_detections"):
            if field_name not in yolo_payload:
                continue
            grouped_entities = yolo_payload.get(field_name)
            if isinstance(grouped_entities, list):
                return [group for group in grouped_entities if isinstance(group, dict)]
        return []
    return [group for group in yolo_payload if isinstance(group, dict)]


def _bbox_area(bbox_xyxy: list[int]) -> int:
    if len(bbox_xyxy) != 4:
        return 0
    x1, y1, x2, y2 = bbox_xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection_area(bbox_a: list[int], bbox_b: list[int]) -> int:
    if len(bbox_a) != 4 or len(bbox_b) != 4:
        return 0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)


def _bbox_iou(bbox_a: list[int], bbox_b: list[int]) -> float:
    intersection_area = _bbox_intersection_area(bbox_a, bbox_b)
    if intersection_area <= 0:
        return 0.0
    union_area = _bbox_area(bbox_a) + _bbox_area(bbox_b) - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _point_in_bbox(center_xy: list[int], bbox_xyxy: list[int]) -> bool:
    if len(center_xy) != 2 or len(bbox_xyxy) != 4:
        return False
    x, y = center_xy
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def _point_to_bbox_l1_distance(center_xy: list[int], bbox_xyxy: list[int]) -> float:
    if len(center_xy) != 2 or len(bbox_xyxy) != 4:
        return float("inf")
    x, y = center_xy
    x1, y1, x2, y2 = bbox_xyxy
    dx = 0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    return float(dx + dy)


def _bbox_fully_contained(inner_bbox: list[int], outer_bbox: list[int]) -> bool:
    if len(inner_bbox) != 4 or len(outer_bbox) != 4:
        return False
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return ox1 <= ix1 <= ix2 <= ox2 and oy1 <= iy1 <= iy2 <= oy2


def _extract_detection_instances_by_label(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    normalized_label = normalize_entity_name(label)
    grouped_candidates = []
    if isinstance(yolo_payload, dict):
        raw_grouped = yolo_payload.get("grouped_detections")
        if isinstance(raw_grouped, list):
            grouped_candidates = [group for group in raw_grouped if isinstance(group, dict)]
    for group in grouped_candidates or _iter_detection_groups(yolo_payload):
        group_label = normalize_entity_name(group.get("label"))
        if group_label != normalized_label:
            continue
        raw_instances = group.get("instances")
        if not isinstance(raw_instances, list):
            continue
        for instance in raw_instances:
            if isinstance(instance, dict):
                instances.append(instance)
    return instances


def _extract_sink_holder_slot_from_init_pos(
    init_pos: dict[str, Any] | None,
    panda_slots: list[str],
) -> str | None:
    """Return the panda slot listed by init_pos as the holder of any sink_*.

    init_pos entries shaped like ``{"sink_2": ["R1"]}`` directly identify
    which robot slot is co-located with the sink in the scene's start state.
    This is the symbolic equivalent of what a VLM derives from the image.
    """
    if not isinstance(init_pos, dict):
        return None
    panda_slot_set = set(panda_slots)
    for entity, holder in init_pos.items():
        if not str(entity).startswith("sink"):
            continue
        if not isinstance(holder, list) or not holder:
            continue
        slot = str(holder[0])
        if slot in panda_slot_set:
            return slot
    return None


_ROBOT_ANCHOR_FROM_INIT_POS_ENV = "LNSDL_ROBOT_ANCHOR_FROM_INIT_POS"


def robot_anchor_from_init_pos_enabled() -> bool:
    """Opt-in switch for the legacy init_pos-derived panda anchor (Strategy A).

    Default off: eval derives ``robot_anchor_status`` from detector geometry
    only (and, because the agentic eval refines with ``yolo_detections=None``,
    emits no anchor at all -- byte-consistent with the SFT corpora). Set the
    env var to ``1`` only to reproduce buckets recorded before 2026-09-05.
    """
    return os.environ.get(_ROBOT_ANCHOR_FROM_INIT_POS_ENV, "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _infer_push_transfer_robot_anchor_status(
    scene_facts: SceneFacts,
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    init_pos: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if "push_transfer" not in scene_facts.parsed_intents:
        return {}, None

    panda_slots = sorted(
        [
            robot.slot_id
            for robot in scene_facts.robot_inventory
            if robot.robot_type == "panda"
        ],
        key=_slot_sort_key,
    )
    if len(panda_slots) != 2:
        return {}, None

    # Strategy A: prefer init_pos sink-holder for slot↔surface binding.
    # init_pos[sink_*]=[<slot>] is dataset-level scene state (not GT plan)
    # and is the symbolic counterpart of what a VLM grounds from the image.
    sink_holder_slot = _extract_sink_holder_slot_from_init_pos(init_pos, panda_slots)
    if sink_holder_slot is not None:
        on_sink_slot = sink_holder_slot
        off_sink_slot = next(s for s in panda_slots if s != sink_holder_slot)
        robot_anchor_status = {
            on_sink_slot: "on_sink_support_surface",
            off_sink_slot: "off_sink_support_surface",
        }
        geometry_note = {
            "slot_grounding_source": "init_pos.sink_holder",
            "sink_holder_slot": sink_holder_slot,
            "slot_grounding_rule": (
                "init_pos[sink_*]=[<slot>] directly identifies the panda slot "
                "co-located with the sink; the other panda slot is on the source surface."
            ),
            "robot_anchor_status": dict(robot_anchor_status),
        }
        return robot_anchor_status, geometry_note

    # Strategy B (fallback): YOLO bbox geometry. Used when init_pos lacks an
    # explicit sink-holder mapping (e.g., upstream pipelines that don't provide
    # ground_truth). Note: lex-order slot binding is known-imperfect; use only
    # when no direct symbolic signal is available.
    table_instances = _extract_detection_instances_by_label(yolo_payload, label="table")
    sink_instances = _extract_detection_instances_by_label(yolo_payload, label="sink")
    panda_instances = _extract_detection_instances_by_label(yolo_payload, label="panda")
    if len(table_instances) != 1 or len(sink_instances) != 1 or len(panda_instances) != 2:
        return {}, None

    table_bbox = list(table_instances[0].get("bbox_xyxy", []))
    sink_bbox = list(sink_instances[0].get("bbox_xyxy", []))
    sink_center = list(sink_instances[0].get("center_xy", []))
    if len(table_bbox) != 4 or len(sink_bbox) != 4 or len(sink_center) != 2:
        return {}, None

    sink_on_table = _bbox_fully_contained(sink_bbox, table_bbox) or _point_in_bbox(
        sink_center,
        table_bbox,
    )
    ranked_pandas: list[dict[str, Any]] = []
    for panda_instance in panda_instances:
        center_xy = list(panda_instance.get("center_xy", []))
        ranked_pandas.append(
            {
                "rank_within_label": int(panda_instance.get("rank_within_label", 1)),
                "center_xy": center_xy,
                "table_distance_l1": round(
                    _point_to_bbox_l1_distance(center_xy, table_bbox),
                    3,
                ),
            }
        )
    ranked_pandas.sort(
        key=lambda item: (
            _coerce_float(item.get("table_distance_l1")),
            int(item.get("rank_within_label", 1)),
        )
    )

    nearest_table_slot = panda_slots[0]
    farther_table_slot = panda_slots[1]
    if sink_on_table:
        robot_anchor_status = {
            nearest_table_slot: "on_sink_support_surface",
            farther_table_slot: "off_sink_support_surface",
        }
    else:
        robot_anchor_status = {
            nearest_table_slot: "off_sink_support_surface",
            farther_table_slot: "on_sink_support_surface",
        }

    geometry_note = {
        "sink_support_relation": "on_table" if sink_on_table else "off_table",
        "slot_grounding_rule": (
            "For same-type panda slots, stable slot order is aligned against the "
            "YOLO support-geometry ranking to derive slot-level anchor status."
        ),
        "panda_table_distance_ranking": ranked_pandas,
        "robot_anchor_status": dict(robot_anchor_status),
    }
    return robot_anchor_status, geometry_note


def _build_yolo_label_corrections(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
    instruction_named_targets: list[str],
) -> dict[str, str]:
    if not instruction_named_targets:
        return {}
    named_target_set = set(instruction_named_targets)
    visible_labels = {
        normalize_entity_name(group.get("label"))
        for group in _iter_detection_groups(yolo_payload)
        if normalize_entity_name(group.get("label"))
    }
    tri_fruit_set = set(_AMBIGUOUS_TRI_FRUIT_LABELS)
    instruction_tri_fruits = tri_fruit_set.intersection(named_target_set)
    visible_tri_fruits = tri_fruit_set.intersection(visible_labels)
    if not instruction_tri_fruits or not visible_tri_fruits:
        return {}

    unexpected_tri_fruits = visible_tri_fruits.difference(instruction_tri_fruits)
    missing_instruction_tri_fruits = instruction_tri_fruits.difference(visible_tri_fruits)
    corrections: dict[str, str] = {}
    for source_label in sorted(unexpected_tri_fruits):
        target_label: str | None = None
        if len(instruction_tri_fruits) >= 2 and len(missing_instruction_tri_fruits) == 1:
            target_label = next(iter(missing_instruction_tri_fruits))
        elif len(instruction_tri_fruits) == 1:
            target_label = next(iter(instruction_tri_fruits))
        elif len(missing_instruction_tri_fruits) == 1:
            target_label = next(iter(missing_instruction_tri_fruits))
        if target_label and target_label != source_label:
            corrections[source_label] = target_label
    return corrections


def _merge_corrected_detection_groups(
    corrected_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_by_label: dict[str, dict[str, Any]] = {}

    for group in corrected_groups:
        label = normalize_entity_name(group.get("label"))
        if not label:
            continue
        target_group = merged_by_label.setdefault(
            label,
            {
                "label": label,
                "category": group.get("category"),
                "regions": [],
                "instances": [],
            },
        )
        if not target_group.get("category") and group.get("category"):
            target_group["category"] = group.get("category")
        for region in group.get("regions", []):
            normalized_region = str(region).strip()
            if normalized_region and normalized_region not in target_group["regions"]:
                target_group["regions"].append(normalized_region)
        for instance in group.get("instances", []):
            if not isinstance(instance, dict):
                continue
            target_group["instances"].append(
                {
                    "label": label,
                    "category": group.get("category"),
                    "confidence": _coerce_confidence(instance.get("confidence")),
                    "bbox_xyxy": list(instance.get("bbox_xyxy", [])),
                    "center_xy": list(instance.get("center_xy", [])),
                    "region": str(instance.get("region", "")).strip(),
                }
            )

    normalized_groups: list[dict[str, Any]] = []
    for group in merged_by_label.values():
        ranked_instances = sorted(
            group["instances"],
            key=lambda instance: -instance["confidence"],
        )
        for rank, instance in enumerate(ranked_instances, start=1):
            instance["rank_within_label"] = rank
        if not group["regions"]:
            group["regions"] = _dedupe_preserve_order(
                [instance["region"] for instance in ranked_instances if instance["region"]]
            )
        normalized_groups.append(
            {
                "label": group["label"],
                "category": group.get("category"),
                "count": len(ranked_instances),
                "best_confidence": max(
                    (instance["confidence"] for instance in ranked_instances),
                    default=0,
                ),
                "regions": group["regions"],
                "instances": ranked_instances,
            }
        )
    return normalized_groups


def _extract_corrected_detection_groups(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    instruction_named_targets: list[str],
) -> list[dict[str, Any]]:
    label_corrections = _build_yolo_label_corrections(
        yolo_payload,
        instruction_named_targets,
    )
    corrected_groups: list[dict[str, Any]] = []
    for group in _iter_detection_groups(yolo_payload):
        original_label = normalize_entity_name(group.get("label"))
        corrected_label = label_corrections.get(original_label, original_label)
        if not corrected_label:
            continue
        corrected_instances: list[dict[str, Any]] = []
        for instance in group.get("instances", []):
            if not isinstance(instance, dict):
                continue
            corrected_instances.append(
                {
                    "label": corrected_label,
                    "confidence": _coerce_confidence(instance.get("confidence")),
                    "bbox_xyxy": list(instance.get("bbox_xyxy", [])),
                    "center_xy": list(instance.get("center_xy", [])),
                    "region": str(instance.get("region", "")).strip(),
                }
            )
        corrected_groups.append(
            {
                "label": corrected_label,
                "category": group.get("category"),
                "count": group.get("count", len(corrected_instances)),
                "best_confidence": _coerce_confidence(group.get("best_confidence")),
                "regions": list(group.get("regions", [])),
                "instances": corrected_instances,
            }
        )
    return _merge_corrected_detection_groups(corrected_groups)


def _has_hard_visual_storage_evidence(
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
) -> bool:
    visible_labels: list[str] = []
    if isinstance(yolo_payload, dict):
        raw_visible_labels = yolo_payload.get("visible_labels", [])
        if isinstance(raw_visible_labels, list):
            visible_labels = [
                normalize_entity_name(label)
                for label in raw_visible_labels
                if normalize_entity_name(label)
            ]
    if "cabinet" in visible_labels:
        return True

    for group in _iter_detection_groups(yolo_payload):
        if normalize_entity_name(group.get("label")) != "cabinet":
            continue
        if _coerce_confidence(group.get("best_confidence")) >= _STRONG_STORAGE_DETECTION_MIN_CONFIDENCE:
            return True
    return False


def _build_storage_cue_provenance(
    initial_scene_facts: SceneFacts,
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    instruction_or_initial = _has_explicit_storage_cue(initial_scene_facts)
    hard_visual_evidence = _has_hard_visual_storage_evidence(yolo_payload)
    return {
        "instruction_or_initial": instruction_or_initial,
        "hard_visual_evidence": hard_visual_evidence,
        "allowed": instruction_or_initial or hard_visual_evidence,
    }


def _filter_cabinet_cue_values(values: list[str], *, allow_cabinet: bool) -> list[str]:
    if allow_cabinet:
        return list(values)
    return [value for value in values if value != "cabinet"]


def _build_refiner_cache_input(
    sample: BenchmarkSample,
    initial_scene_facts: SceneFacts,
    *,
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cache_version": SCENE_FACTS_REFINER_CACHE_VERSION,
        "sample": {
            "level": sample.level,
            "split": sample.split,
            "sample_index": sample.sample_index,
            "task_id": sample.task_id,
            "task_name": sample.task_name,
            "instruction": sample.instruction,
            "image_path": sample.image_path,
            "image_byte_size": len(sample.image_bytes or b""),
        },
        "initial_scene_facts": _scene_facts_payload(initial_scene_facts),
        "robot_inventory": [
            {
                "slot_id": robot.slot_id,
                "robot_type": robot.robot_type,
            }
            for robot in initial_scene_facts.robot_inventory
        ],
        "yolo_detections": yolo_payload,
    }


def _build_refiner_cache_path(sample: BenchmarkSample) -> Path:
    return DEFAULT_SCENE_FACTS_REFINER_CACHE_DIR / f"{_build_refiner_cache_stem(sample)}.json"


def _load_cached_refinement(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cached_payload = payload.get("payload")
    if not isinstance(cached_payload, dict):
        return None
    return payload


def _write_cached_refinement(
    cache_path: Path,
    *,
    sample: BenchmarkSample,
    input_fingerprint: str,
    payload: dict[str, Any],
    response_text: str,
    request_trace: list[dict[str, Any]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_record = {
        "cache_version": SCENE_FACTS_REFINER_CACHE_VERSION,
        "sample_stem": _build_refiner_cache_stem(sample),
        "input_fingerprint": input_fingerprint,
        "payload": payload,
        "response_text": response_text,
        "request_trace": request_trace,
        "updated_at_unix": round(time.time(), 3),
    }
    cache_path.write_text(
        json.dumps(cache_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_named_fruit_inspection_task(
    initial_scene_facts: SceneFacts,
    instruction_named_fruits: list[str],
) -> bool:
    parsed_intents = set(initial_scene_facts.parsed_intents)
    if len(instruction_named_fruits) < 2:
        return False
    if "inspect_missing" in parsed_intents:
        return True
    return (
        "inspect" in parsed_intents
        and "retrieve" in parsed_intents
        and initial_scene_facts.destination == "table"
    )


def _build_task_specific_hints(
    sample: BenchmarkSample,
    initial_scene_facts: SceneFacts,
    *,
    yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    yolo_payload = _normalize_yolo_payload(yolo_detections)
    instruction_named_fruits = _extract_instruction_named_fruits(sample.instruction)
    yolo_visible_fruits = _extract_visible_yolo_fruits(
        yolo_payload,
        instruction_named_targets=[],
    )
    task_hints: dict[str, Any] = {
        "target_semantic_state_disabled": True,
        "instruction_named_fruits": instruction_named_fruits,
        "yolo_visible_fruits": yolo_visible_fruits,
        "location_predicates": list(_TARGET_LOCATION_PREDICATES),
    }

    if _is_named_fruit_inspection_task(initial_scene_facts, instruction_named_fruits):
        task_hints.setdefault("reasoning_rules", []).extend(
            [
                "For fruit verification tasks, use the fruit names explicitly mentioned in the instruction as the highest-value semantic anchor.",
                "If a named fruit bbox is fully contained in a cabinet bbox, prefer likely_in_cabinet for that fruit.",
                "If a named fruit is detector-visible and not inside cabinet by overlap, prefer visible_elsewhere over likely_in_cabinet.",
            ]
        )
    return task_hints


def _merge_scalar(
    payload: dict[str, Any],
    field_name: str,
    fallback: str | None,
) -> str | None:
    if field_name not in payload:
        return fallback
    return normalize_entity_name(payload.get(field_name))


def _merge_list(
    payload: dict[str, Any],
    field_name: str,
    fallback: list[str],
) -> list[str]:
    if field_name not in payload:
        return list(fallback)
    return _normalize_entity_list(payload.get(field_name))


def _merge_destination_pairs(
    payload: dict[str, Any],
    field_name: str,
    fallback: list[dict[str, str]],
) -> list[dict[str, str]]:
    if field_name not in payload:
        return _normalize_destination_pairs(fallback)
    return _normalize_destination_pairs(payload.get(field_name))


def _parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return json.loads(stripped)


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


def _build_format_only_retry_messages(
    base_messages: list[dict[str, Any]],
    invalid_response_text: str,
    *,
    parse_error: Exception,
) -> list[dict[str, Any]]:
    return [
        *base_messages,
        {
            "role": "assistant",
            "content": invalid_response_text,
        },
        {
            "role": "user",
            "content": (
                "Your previous answer was not valid JSON. "
                f"Parse error: {parse_error}. "
                "Retry once in format-only mode: keep the same scene-facts judgment, "
                "but return exactly one JSON object matching required_output_schema. "
                "Do not include markdown fences, comments, or any extra text."
            ),
        },
    ]


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    image_type = imghdr.what(None, h=image_bytes) or "png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{image_type};base64,{encoded}"


def build_scene_facts_refinement_prompt(
    sample: BenchmarkSample,
    initial_scene_facts: SceneFacts,
    *,
    yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    yolo_payload = _normalize_yolo_payload(yolo_detections)
    task_specific_hints = _build_task_specific_hints(
        sample,
        initial_scene_facts,
        yolo_detections=yolo_detections,
    )
    region_reasoning_hints = _build_region_reasoning_hints(
        initial_scene_facts,
        yolo_payload,
    )
    user_payload = {
        "instruction": sample.instruction,
        "initial_scene_facts": _scene_facts_payload(initial_scene_facts),
        "robot_inventory": [
            {"slot_id": robot.slot_id, "robot_type": robot.robot_type}
            for robot in initial_scene_facts.robot_inventory
        ],
        "yolo_detections": yolo_payload,
        "task_specific_hints": task_specific_hints,
        "region_reasoning_hints": region_reasoning_hints,
        "refinement_goal": (
            "Refine the initial scene facts using the image and YOLO detections. "
            "Only keep task-relevant scene information needed for later macro-graph decomposition."
        ),
        "rules": [
            "Do not generate a plan or macro graph.",
            "Do not invent primitive-level action requirements or robot assignments.",
            "Keep the output concise and limited to the scene_facts schema fields.",
            "Use benchmark entity names such as 'wooden cutting board' and 'kitchen work area'.",
            "When the image or YOLO detections do not add reliable evidence, preserve the initial parsed value.",
            "When YOLO detections are empty, still use the image if it helps, otherwise keep the initial parse.",
            "If YOLO detections conflict with the instruction, trust the instruction and correct or drop the conflicting YOLO evidence in the refined scene facts.",
            "YOLO directly grounds only the labels listed in yolo_detections.supported_labels when that field is present.",
            "The current detector can directly detect table and cabinet, but kitchen work area and kitchen island area still require instruction plus whole-image semantic reasoning.",
            "In most scenes, instruction-level table means the single freestanding tabletop in the central scene region; wall-adjacent counters usually do not count as table.",
            "Assume at most one table per scene. If yolo_detections.grouped_detections already contains one reliable table bbox, use that bbox as the primary table anchor.",
            "table and kitchen island area are not mutually exclusive labels. The same surface or object placement may satisfy both.",
            "Do not treat table-versus-kitchen-island overlap as a contradiction by itself.",
            "Do not output target_items, primary_object, secondary_objects, targets_in_cabinet, or target_status; target decomposition and skip/active decisions belong to the SLM [GOAL] output.",
            "Use bbox containment and center-in-bbox only as generic spatial observation evidence, not as per-target already_at_destination inference.",
            "Only include cabinet in source_locations, fallback_locations, inspection_targets, destination_candidates, or mentioned_objects when cabinet is explicit in the instruction or initial parse, or when detector-grounded cabinet evidence is strong.",
            "Do not introduce cabinet as a fallback cue solely because retrieval might be possible.",
        ],
        "required_output_schema": {
            "destination": "benchmark entity or null",
            "destination_pairs": [{"item": "movable benchmark entity", "target": "benchmark destination"}],
            "destination_candidates": ["benchmark entities"],
            "source_locations": ["benchmark entities"],
            "fallback_locations": ["benchmark entities"],
            "tool_items": ["benchmark entities"],
            "activation_targets": ["benchmark entities"],
            "inspection_targets": ["benchmark entities"],
            "parsed_intents": ["short intent strings"],
            "mentioned_objects": ["benchmark entities"],
        },
    }
    if region_reasoning_hints:
        user_payload["rules"].extend(region_reasoning_hints["reasoning_rules"])

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
        }
    ]
    if sample.image_bytes:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_bytes_to_data_url(sample.image_bytes)},
            }
        )

    return [
        {
            "role": "system",
            "content": (
                "You are an NSDL scene-facts refiner. "
                "Your job is to refine task-relevant scene facts for later macro-graph generation. "
                "Use the instruction as the anchor, and use the image plus YOLO detections only to confirm or supplement scene facts. "
                "Instruction semantics override conflicting YOLO detections. "
                "The current detector can directly localize table and cabinet, but kitchen work area and kitchen island area still require whole-image semantic reasoning. "
                "In most scenes, instruction-level table means the single freestanding tabletop near the center of the image, while wall-adjacent counters usually are not table. "
                "Assume at most one table per scene; if YOLO already gives one reliable table bbox, use that bbox as the primary table anchor. "
                "table and kitchen island area are not mutually exclusive labels, so do not treat overlap between them as a contradiction. "
                "Do not infer or output target_items, primary_object, secondary_objects, targets_in_cabinet, or target_status. "
                "Use bbox containment and center-in-bbox as generic support-relation evidence only. "
                "Return JSON only. Keep the scene facts minimal and task-oriented."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


class BaseSceneFactsRefiner(ABC):
    @abstractmethod
    def refine(
        self,
        sample: BenchmarkSample,
        initial_scene_facts: SceneFacts,
        *,
        yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> SceneFacts:
        raise NotImplementedError


def apply_deterministic_postprocessing(
    refined_scene_facts: SceneFacts,
    initial_scene_facts: SceneFacts,
    task_specific_hints: dict[str, Any],
    yolo_payload: dict[str, Any] | list[dict[str, Any]],
    *,
    target_status_seed: dict[str, str] | None = None,
    init_pos: dict[str, Any] | None = None,
) -> SceneFacts:
    """Apply all deterministic post-processing to *refined_scene_facts*.

    This centralises deterministic, non-planning scene cleanup. Target
    decomposition fields are intentionally cleared here so parser/refiner output
    cannot act as a hidden target-set channel.

    Parameters
    ----------
    refined_scene_facts:
        The SceneFacts to post-process (may already incorporate LLM edits, or
        may be a plain copy of *initial_scene_facts* in the NoOp path).
    initial_scene_facts:
        The SceneFacts as returned by the rule-based parser **before** any
        refinement.  Used for storage-cue provenance checks.
    task_specific_hints:
        Dict computed by ``_build_task_specific_hints()``.
    yolo_payload:
        Normalised YOLO detections (dict or list).
    target_status_seed:
        Ignored. Kept only for call-site compatibility with old cached refiner
        payloads.
    """
    # ── 1. Target semantic state is disabled ───────────────────────
    refined_scene_facts.target_items = []
    refined_scene_facts.primary_object = None
    refined_scene_facts.secondary_objects = []
    refined_scene_facts.targets_in_cabinet = []
    refined_scene_facts.parser_notes.pop("target_status", None)
    refined_scene_facts.parser_notes.pop("instruction_target_preservation", None)
    refined_scene_facts.parser_notes.pop("location_predicate_guard", None)
    refined_scene_facts.parser_notes["target_semantic_state_disabled"] = True
    refined_scene_facts.field_sources["target_items"] = (
        "disabled: target decomposition is reserved for SLM output"
    )
    refined_scene_facts.field_sources["primary_object"] = (
        "disabled: target decomposition is reserved for SLM output"
    )
    refined_scene_facts.field_sources["secondary_objects"] = (
        "disabled: target decomposition is reserved for SLM output"
    )
    refined_scene_facts.field_sources["targets_in_cabinet"] = (
        "disabled: target status inference is reserved for SLM output"
    )

    if (
        refined_scene_facts.destination
        and refined_scene_facts.destination
        not in refined_scene_facts.destination_candidates
    ):
        refined_scene_facts.destination_candidates = [
            refined_scene_facts.destination,
            *refined_scene_facts.destination_candidates,
        ]

    # ── 2. Storage-cue provenance check ───────────────────────────
    storage_cue_provenance = _build_storage_cue_provenance(
        initial_scene_facts,
        yolo_payload,
    )
    refined_scene_facts.parser_notes["storage_cue_provenance"] = storage_cue_provenance
    storage_sanitization: dict[str, Any] = {}
    if not storage_cue_provenance["allowed"]:
        for field_name in _STORAGE_CUE_FIELDS:
            field_values = list(getattr(refined_scene_facts, field_name))
            filtered_values = _filter_cabinet_cue_values(
                field_values,
                allow_cabinet=False,
            )
            if filtered_values != field_values:
                storage_sanitization[field_name] = {
                    "removed": [v for v in field_values if v not in filtered_values],
                }
                setattr(refined_scene_facts, field_name, filtered_values)
        refined_scene_facts.targets_in_cabinet = []
    if storage_sanitization:
        refined_scene_facts.parser_notes["storage_cue_sanitization"] = storage_sanitization

    # ── 3. robot_anchor_status (YOLO support geometry only, by default)
    # The init_pos sink-holder route reads dataset scene metadata that the
    # model never sees in training (0/13496 SFT rows carry an anchor because
    # the corpus builder runs no refiner) and that the official prompt/judge do
    # not expose either. Keep it strictly opt-in so old buckets stay
    # reproducible; eval never sets it (2026-09-05 ruling: geometry only).
    robot_anchor_status, robot_anchor_geometry = _infer_push_transfer_robot_anchor_status(
        refined_scene_facts,
        yolo_payload,
        init_pos=init_pos if robot_anchor_from_init_pos_enabled() else None,
    )
    if robot_anchor_status:
        refined_scene_facts.robot_anchor_status = robot_anchor_status
        refined_scene_facts.field_sources["robot_anchor_status"] = (
            "YOLO support-geometry grounding over sink/table/panda detections"
        )
        refined_scene_facts.parser_notes["robot_anchor_geometry"] = robot_anchor_geometry

    # ── 4. scene_relations (task-blind YOLO bbox geometry → "<entity> on/in/at <fixture>")
    #     Kept as an artifact/cache for diagnostics and on-demand perception
    #     tooling; the P3 lean SLM prompt no longer pre-injects these lines.
    #     Source = pure geometric containment / center-in-bbox; NO instruction
    #     parsing or role inference (Task-Blindness Rule, agent.md).
    try:
        scene_relations = _build_scene_relations(yolo_payload, refined_scene_facts)
    except Exception:  # noqa: BLE001 — never block refine on a formatting bug
        scene_relations = []
    if isinstance(yolo_payload, dict):
        if refined_scene_facts.extras is None:
            refined_scene_facts.extras = {}
        refined_scene_facts.extras["scene_relations"] = scene_relations or ["none observed"]

    return refined_scene_facts


# P5.10.0 #6 — support-geometry fuzzy band for generic bbox containment (part b).
# Containment ratio = intersection(item, fixture) / item_area. A ratio ≥ ON means
# a confident "on"; a ratio in [FUZZY_LOW, ON) is ambiguous (item straddles the
# fixture edge / low-pixel scene), so we emit BOTH "on" and "near" as competing
# hypotheses instead of hard-judging one. Bounds are heuristic geometry thresholds
# (task-blind: no instruction/GT input), tuned to only flag genuine edge cases.
_SUPPORT_ON_RATIO = 0.7
_SUPPORT_FUZZY_LOW = 0.3


def _containment_ratio(
    item_box: tuple[float, float, float, float],
    fixture_box: tuple[float, float, float, float],
) -> float:
    """Fraction of the item bbox area covered by the fixture bbox (0..1)."""
    ix1, iy1, ix2, iy2 = item_box
    fx1, fy1, fx2, fy2 = fixture_box
    item_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if item_area <= 0.0:
        return 0.0
    inter_w = max(0.0, min(ix2, fx2) - max(ix1, fx1))
    inter_h = max(0.0, min(iy2, fy2) - max(iy1, fy1))
    return (inter_w * inter_h) / item_area


def _build_scene_relations(
    yolo_payload: Any,
    scene_facts: SceneFacts | None = None,
) -> list[str]:
    """Derive task-blind ``<entity> on|in|at <fixture>`` strings from YOLO geometry.

    Pure observation: uses `table_item_overlap.support_like_contained` and
    `grouped_detections[*].instances` bbox containment / center-in-bbox.
    Never reads instruction text, parser-derived destination, or role tags.
    ``scene_facts`` is only consulted for robot slot labels / the panda-carrier
    robot relation (legacy refiner path); ``detect`` passes ``None`` so the
    model-facing tool never sees instruction-derived input.
    """
    if not isinstance(yolo_payload, dict):
        return []

    relations: list[str] = []
    seen: set[str] = set()

    def _emit(entity: str, verb: str, fixture: str) -> None:
        ent = (entity or "").strip().lower()
        fix = (fixture or "").strip().lower()
        if not ent or not fix or ent == fix:
            return
        rel = f"{ent} {verb} {fix}"
        if rel in seen:
            return
        seen.add(rel)
        relations.append(rel)

    # Map robot-type detections (panda, fetch, sawyer, ...) → "{type} {slot_id}"
    # using scene_facts.robot_inventory order. Robot observations are emitted
    # only for the cargo-carrier panda/table grounding case below.
    robot_slot_map: dict[str, list[str]] = {}
    inv = getattr(scene_facts, "robot_inventory", None) or []
    for ri in inv:
        rtype = (getattr(ri, "robot_type", "") or "").strip().lower()
        slot = (getattr(ri, "slot_id", "") or "").strip()
        if rtype and slot:
            robot_slot_map.setdefault(rtype, []).append(slot)

    robot_seen_count: dict[str, int] = {}

    def _resolve_robot_label(label: str) -> str:
        slots = robot_slot_map.get(label)
        if not slots:
            return label
        idx = robot_seen_count.get(label, 0)
        robot_seen_count[label] = idx + 1
        if idx < len(slots):
            return f"{label} {slots[idx]}"
        return label

    # (a) Precomputed table-item overlap (highest confidence).
    overlap = yolo_payload.get("table_item_overlap")
    if isinstance(overlap, dict):
        items = overlap.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                # `support_like_contained` lives inside best_table_match
                # (see yolo_detector._build_table_item_overlap_summary).
                best = item.get("best_table_match")
                if not isinstance(best, dict) or not best.get("support_like_contained"):
                    continue
                label = item.get("label") or item.get("name")
                if isinstance(label, str):
                    normalized_label = normalize_entity_name(label) or label.strip().lower()
                    if normalized_label in robot_slot_map:
                        continue
                    if is_portable_entity(normalized_label):
                        _emit(normalized_label, "on", "table")

    # (b) Generic bbox containment from grouped_detections (centers inside fixtures).
    grouped = yolo_payload.get("grouped_detections")
    fixture_labels = {"table", "cabinet", "fridge", "microwave", "drawer", "shelf",
                      "counter", "sink", "stove", "oven", "sofa", "chair", "toaster"}
    fixture_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    item_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    if isinstance(grouped, list):
        for group in grouped:
            if not isinstance(group, dict):
                continue
            label = (group.get("label") or group.get("name") or "").strip().lower()
            if not label:
                continue
            for inst in group.get("instances", []) or []:
                if not isinstance(inst, dict):
                    continue
                # Detector emits `bbox_xyxy`; tolerate `bbox`/`xyxy` legacy.
                bbox = inst.get("bbox_xyxy") or inst.get("bbox") or inst.get("xyxy")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                try:
                    x1, y1, x2, y2 = (float(v) for v in bbox)
                except (TypeError, ValueError):
                    continue
                box = (x1, y1, x2, y2)
                if label in fixture_labels:
                    fixture_boxes.append((label, box))
                else:
                    item_boxes.append((label, box))

    for item_label, (ix1, iy1, ix2, iy2) in item_boxes:
        cx = 0.5 * (ix1 + ix2)
        cy = 0.5 * (iy1 + iy2)
        is_robot = item_label in robot_slot_map
        if is_robot and (scene_facts is None or not _should_emit_robot_table_relation(scene_facts, item_label)):
            continue
        if not is_robot:
            canonical_label = normalize_entity_name(item_label) or item_label
            if not is_portable_entity(canonical_label) and not entity_has_tag(canonical_label, "pushable_carrier"):
                continue
            display_label = canonical_label
        else:
            display_label = _resolve_robot_label(item_label)
        for fix_label, (fx1, fy1, fx2, fy2) in fixture_boxes:
            if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                if is_robot and fix_label != "table":
                    continue
                # `in` for openable, else `on`
                openable = fix_label in {"cabinet", "fridge", "microwave", "drawer", "oven"}
                verb = "in" if openable else "on"
                _emit(display_label, verb, fix_label)
                # P5.10.0 #6: for surface support ("on"), if the item only
                # partially overlaps the fixture (ambiguous band), also surface a
                # competing "near" hypothesis rather than hard-committing to "on".
                if not openable and not is_robot:
                    ratio = _containment_ratio(
                        (ix1, iy1, ix2, iy2), (fx1, fy1, fx2, fy2)
                    )
                    if _SUPPORT_FUZZY_LOW <= ratio < _SUPPORT_ON_RATIO:
                        _emit(display_label, "near", fix_label)
                break

    return relations[:32]


def _should_emit_robot_table_relation(scene_facts: SceneFacts, robot_type: str) -> bool:
    """Restrict robot/table observations to the panda carrier grounding case."""
    if robot_type != "panda":
        return False
    candidate_objects = set()
    for values in (
        scene_facts.mentioned_objects,
        scene_facts.source_locations,
        scene_facts.destination_candidates,
        scene_facts.tool_items,
    ):
        candidate_objects.update(normalize_entity_name(v) or str(v).strip().lower() for v in values or [])
    if scene_facts.destination:
        candidate_objects.add(normalize_entity_name(scene_facts.destination) or scene_facts.destination)

    for obj in candidate_objects:
        try:
            if obj and entity_has_tag(obj, "pushable_carrier"):
                return True
        except Exception:
            continue
    return False


class NoOpSceneFactsRefiner(BaseSceneFactsRefiner):
    def refine(
        self,
        sample: BenchmarkSample,
        initial_scene_facts: SceneFacts,
        *,
        yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> SceneFacts:
        yolo_payload = _normalize_yolo_payload(yolo_detections)
        task_specific_hints = _build_task_specific_hints(
            sample,
            initial_scene_facts,
            yolo_detections=yolo_detections,
        )
        refined = replace(
            initial_scene_facts,
            parser_notes={
                **initial_scene_facts.parser_notes,
                "refiner_backend": "none",
                "yolo_detections": yolo_payload,
                "task_specific_hints": task_specific_hints,
            },
        )
        init_pos = (sample.ground_truth or {}).get("init_pos") if sample.ground_truth else None
        return apply_deterministic_postprocessing(
            refined,
            initial_scene_facts,
            task_specific_hints,
            yolo_payload,
            init_pos=init_pos if isinstance(init_pos, dict) else None,
        )


class LLMSceneFactsRefiner(BaseSceneFactsRefiner):
    def __init__(
        self,
        *,
        endpoint_index: int = 0,
        api_base_url: str | None = None,
        api_model: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.endpoint_index = endpoint_index
        self.api_base_url = api_base_url
        self.api_model = api_model
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_SCENE_FACTS_REFINER_CACHE_DIR

    def _build_endpoint_sequence(self) -> list[int]:
        from nsdl.llm_api import get_available_endpoint_count

        if self.api_base_url is not None or self.api_model is not None:
            return [self.endpoint_index]
        try:
            endpoint_count = get_available_endpoint_count()
        except Exception:
            return [self.endpoint_index]
        if endpoint_count <= 0:
            return [self.endpoint_index]
        start_index = self.endpoint_index % endpoint_count
        return [start_index, *[index for index in range(endpoint_count) if index != start_index]]

    def _request_completion_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, list[dict[str, Any]]]:
        from nsdl.llm_api import create_chat_completion

        request_trace: list[dict[str, Any]] = []
        last_error: Exception | None = None
        endpoint_sequence = self._build_endpoint_sequence()

        for endpoint_index in endpoint_sequence:
            for attempt in range(_REFINER_MAX_ATTEMPTS_PER_ENDPOINT):
                try:
                    response = create_chat_completion(
                        messages=messages,
                        endpoint_index=endpoint_index,
                        api_base_url=self.api_base_url,
                        api_model=self.api_model,
                        **request_kwargs,
                    )
                    request_trace.append(
                        {
                            "endpoint_index": endpoint_index,
                            "attempt": attempt + 1,
                            "status": "ok",
                        }
                    )
                    return response, request_trace
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    request_trace.append(
                        {
                            "endpoint_index": endpoint_index,
                            "attempt": attempt + 1,
                            "status": "error",
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc)[:240],
                        }
                    )
                    if attempt + 1 < _REFINER_MAX_ATTEMPTS_PER_ENDPOINT:
                        time.sleep(_REFINER_INITIAL_BACKOFF_SECONDS * (2**attempt))
        if last_error is None:
            raise RuntimeError("Scene facts refiner request failed without an exception payload.")
        raise last_error

    def refine(
        self,
        sample: BenchmarkSample,
        initial_scene_facts: SceneFacts,
        *,
        yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> SceneFacts:
        yolo_payload = _normalize_yolo_payload(yolo_detections)
        task_specific_hints = _build_task_specific_hints(
            sample,
            initial_scene_facts,
            yolo_detections=yolo_detections,
        )
        messages = build_scene_facts_refinement_prompt(
            sample,
            initial_scene_facts,
            yolo_detections=yolo_detections,
        )
        cache_path = self.cache_dir / f"{_build_refiner_cache_stem(sample)}.json"
        cache_input = _build_refiner_cache_input(
            sample,
            initial_scene_facts,
            yolo_payload=yolo_payload,
        )
        input_fingerprint = _json_fingerprint(cache_input)
        cached_record = _load_cached_refinement(cache_path)
        cache_note: dict[str, Any] = {
            "path": str(cache_path),
            "input_fingerprint": input_fingerprint,
            "hit": False,
            "stale_fallback": False,
        }
        request_trace: list[dict[str, Any]] = []
        request_kwargs = {
            "temperature": 0.0,
            "max_tokens": 768,
        }
        format_retry_note: dict[str, Any] | None = None
        response_text = ""
        payload: dict[str, Any] | None = None

        if cached_record and cached_record.get("input_fingerprint") == input_fingerprint:
            payload = dict(cached_record.get("payload", {}))
            response_text = str(cached_record.get("response_text", ""))
            cache_note["hit"] = True
            cache_note["matched_cached_input"] = True
        else:
            try:
                response, initial_trace = self._request_completion_with_retry(
                    messages=messages,
                    request_kwargs=request_kwargs,
                )
                request_trace.extend(initial_trace)
                response_text = _message_content_to_text(response.choices[0].message.content)
                try:
                    payload = _parse_json_payload(response_text)
                except json.JSONDecodeError as first_error:
                    retry_messages = _build_format_only_retry_messages(
                        messages,
                        response_text,
                        parse_error=first_error,
                    )
                    retry_response, retry_trace = self._request_completion_with_retry(
                        messages=retry_messages,
                        request_kwargs=request_kwargs,
                    )
                    request_trace.extend(retry_trace)
                    retry_text = _message_content_to_text(retry_response.choices[0].message.content)
                    try:
                        payload = _parse_json_payload(retry_text)
                    except json.JSONDecodeError as retry_error:
                        raise ValueError(
                            "Scene facts refiner format-only retry failed. "
                            f"initial_parse_error={first_error}; retry_parse_error={retry_error}"
                        ) from retry_error
                    response_text = retry_text
                    format_retry_note = {
                        "used": True,
                        "initial_parse_error": str(first_error),
                    }
                try:
                    _write_cached_refinement(
                        cache_path,
                        sample=sample,
                        input_fingerprint=input_fingerprint,
                        payload=payload,
                        response_text=response_text,
                        request_trace=request_trace,
                    )
                except OSError as cache_error:
                    cache_note["write_error"] = str(cache_error)
            except Exception:  # noqa: BLE001
                if cached_record is None:
                    raise
                payload = dict(cached_record.get("payload", {}))
                response_text = str(cached_record.get("response_text", ""))
                cache_note["stale_fallback"] = True
                cache_note["matched_cached_input"] = (
                    cached_record.get("input_fingerprint") == input_fingerprint
                )
                cache_note["fallback_reason"] = (
                    request_trace[-1]["error_type"]
                    if request_trace
                    else "request_failed_before_response"
                )

        if payload is None:
            raise RuntimeError("Scene facts refiner did not produce a payload.")

        refined_scene_facts = SceneFacts(
            parser_backend=f"{initial_scene_facts.parser_backend}+llm_refiner",
            instruction=initial_scene_facts.instruction,
            image_available=initial_scene_facts.image_available,
            robot_inventory=list(initial_scene_facts.robot_inventory),
            primary_object=None,
            destination=_merge_scalar(
                payload,
                "destination",
                initial_scene_facts.destination,
            ),
            destination_pairs=_merge_destination_pairs(
                payload,
                "destination_pairs",
                initial_scene_facts.destination_pairs,
            ),
            secondary_objects=[],
            target_items=[],
            destination_candidates=_merge_list(
                payload,
                "destination_candidates",
                initial_scene_facts.destination_candidates,
            ),
            source_locations=_merge_list(
                payload,
                "source_locations",
                initial_scene_facts.source_locations,
            ),
            fallback_locations=_merge_list(
                payload,
                "fallback_locations",
                initial_scene_facts.fallback_locations,
            ),
            targets_in_cabinet=[],
            tool_items=_merge_list(
                payload,
                "tool_items",
                initial_scene_facts.tool_items,
            ),
            activation_targets=_merge_list(
                payload,
                "activation_targets",
                initial_scene_facts.activation_targets,
            ),
            inspection_targets=_merge_list(
                payload,
                "inspection_targets",
                initial_scene_facts.inspection_targets,
            ),
            inspect_mode=initial_scene_facts.inspect_mode,
            parsed_intents=[
                str(intent).strip()
                for intent in payload.get(
                    "parsed_intents",
                    initial_scene_facts.parsed_intents,
                )
                if str(intent).strip()
            ],
            mentioned_objects=_merge_list(
                payload,
                "mentioned_objects",
                initial_scene_facts.mentioned_objects,
            ),
            robot_anchor_status=dict(initial_scene_facts.robot_anchor_status),
            field_sources={
                **initial_scene_facts.field_sources,
                "primary_object": "disabled: target decomposition is reserved for SLM output",
                "destination": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "destination" in payload
                    else initial_scene_facts.field_sources.get("destination", "")
                ),
                "destination_pairs": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "destination_pairs" in payload
                    else initial_scene_facts.field_sources.get("destination_pairs", "")
                ),
                "secondary_objects": "disabled: target decomposition is reserved for SLM output",
                "target_items": "disabled: target decomposition is reserved for SLM output",
                "destination_candidates": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "destination_candidates" in payload
                    else initial_scene_facts.field_sources.get("destination_candidates", "")
                ),
                "source_locations": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "source_locations" in payload
                    else initial_scene_facts.field_sources.get("source_locations", "")
                ),
                "fallback_locations": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "fallback_locations" in payload
                    else initial_scene_facts.field_sources.get("fallback_locations", "")
                ),
                "targets_in_cabinet": "disabled: target status inference is reserved for SLM output",
                "tool_items": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "tool_items" in payload
                    else initial_scene_facts.field_sources.get("tool_items", "")
                ),
                "activation_targets": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "activation_targets" in payload
                    else initial_scene_facts.field_sources.get("activation_targets", "")
                ),
                "inspection_targets": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "inspection_targets" in payload
                    else initial_scene_facts.field_sources.get("inspection_targets", "")
                ),
                "inspect_mode": initial_scene_facts.field_sources.get("inspect_mode", ""),
                "parsed_intents": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "parsed_intents" in payload
                    else initial_scene_facts.field_sources.get("parsed_intents", "")
                ),
                "mentioned_objects": (
                    "VLM scene-facts refinement conditioned on instruction, image, and yolo detections"
                    if "mentioned_objects" in payload
                    else initial_scene_facts.field_sources.get("mentioned_objects", "")
                ),
                "robot_anchor_status": initial_scene_facts.field_sources.get("robot_anchor_status", ""),
            },
            parser_notes={
                **initial_scene_facts.parser_notes,
                "refiner_backend": "llm",
                "image_used": bool(sample.image_bytes),
                "yolo_detections": yolo_payload,
                "initial_scene_facts": _scene_facts_payload(initial_scene_facts),
                "task_specific_hints": task_specific_hints,
                "inspect_mode": initial_scene_facts.inspect_mode,
                "format_retry": format_retry_note,
                "refiner_cache": cache_note,
                "llm_request_trace": request_trace,
                "llm_refinement_response": payload,
            },
            extras=dict(initial_scene_facts.extras),
        )

        init_pos = (sample.ground_truth or {}).get("init_pos") if sample.ground_truth else None
        return apply_deterministic_postprocessing(
            refined_scene_facts,
            initial_scene_facts,
            task_specific_hints,
            yolo_payload,
            init_pos=init_pos if isinstance(init_pos, dict) else None,
        )


class AutoSceneFactsRefiner(BaseSceneFactsRefiner):
    def __init__(
        self,
        *,
        endpoint_index: int = 0,
        api_base_url: str | None = None,
        api_model: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.llm_refiner = LLMSceneFactsRefiner(
            endpoint_index=endpoint_index,
            api_base_url=api_base_url,
            api_model=api_model,
            cache_dir=cache_dir,
        )

    def refine(
        self,
        sample: BenchmarkSample,
        initial_scene_facts: SceneFacts,
        *,
        yolo_detections: StructuredDetections | dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> SceneFacts:
        return self.llm_refiner.refine(
            sample,
            initial_scene_facts,
            yolo_detections=yolo_detections,
        )


def build_scene_facts_refiner(
    backend: str,
    *,
    endpoint_index: int = 0,
    api_base_url: str | None = None,
    api_model: str | None = None,
    cache_dir: Path | str | None = None,
) -> BaseSceneFactsRefiner:
    if backend == "none":
        return NoOpSceneFactsRefiner()
    if backend == "llm":
        return LLMSceneFactsRefiner(
            endpoint_index=endpoint_index,
            api_base_url=api_base_url,
            api_model=api_model,
            cache_dir=cache_dir,
        )
    if backend == "auto":
        return AutoSceneFactsRefiner(
            endpoint_index=endpoint_index,
            api_base_url=api_base_url,
            api_model=api_model,
            cache_dir=cache_dir,
        )
    raise ValueError(
        f"Unsupported scene-facts refiner backend: {backend}. "
        "Expected one of: none, llm, auto."
    )
