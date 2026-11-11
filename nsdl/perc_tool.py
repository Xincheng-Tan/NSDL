"""On-demand perception query tool for L-NSDL.

``detect`` wraps the structured YOLO backend and the existing task-blind scene
relation builder. It returns compact observations only: canonical entity names,
geometric relations, and confidences. It never returns bboxes or semantic target
status such as ``already_at_destination``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from nsdl.catalog import (
    ENTITY_PROFILES,
    entity_has_tag,
    find_entity_mentions,
    get_entity_profile,
    is_portable_entity,
    normalize_entity_name,
)
from nsdl.scene_facts_refiner import _build_scene_relations
from nsdl.schema import StructuredDetections
from nsdl.yolo_detector import BaseStructuredDetector, UltralyticsYoloDetector


# P5.10.0 crop-and-look geometry constants (deterministic, unit-testable).
_CROP_MAX = 2                 # token/image budget guard: ≤2 crops per detect call
_CROP_EXPAND_RATIO = 0.25     # expand bbox 25% on each side (context around the item)
_CROP_MIN_BOX = 112           # if expanded box < 112px on a side, grow from center to 112
_CROP_MIN_SHORT_SIDE = 224    # upsample so the crop's short side ≥ 224 (better pixels for VLM)


@dataclass
class CropObservation:
    """A zoomed-in image crop of one queried entity, for tool-turn re-injection.

    ``image`` is a ``PIL.Image.Image`` (pixels never enter JSON/trace text; only
    ``entity``/``confidence`` are serialized by ``DetectResult.to_dict``).
    """

    entity: str
    image: Any
    confidence: float = 0.0


def strip_crops_postprocess(result: "DetectResult | None") -> "DetectResult | None":
    """Eval-only ``detect_postprocess`` hook for the crop-off ablation arm.

    Returns the same entities/relations/confidences with every crop removed and
    no crop diagnostics, so the tool-result JSON reads as if no crop had been
    requested. ``crop_missing`` is deliberately NOT populated: in the SFT corpus
    that field co-occurs with the entity being undetected, so using it here
    would inject a spurious absence signal instead of just removing pixels.
    """
    if result is None:
        return result
    return DetectResult(
        entities=list(result.entities),
        relations=list(result.relations),
        confidences=dict(result.confidences),
    )


@dataclass
class DetectResult:
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    confidences: dict[str, float] = field(default_factory=dict)
    # P5.10.0 crop-and-look: pixels stay out of `to_dict` (text-only invariant);
    # diagnostics record why a requested crop was not produced.
    crops: list[CropObservation] = field(default_factory=list)
    crops_truncated: bool = False
    crop_missing: list[str] = field(default_factory=list)
    crop_unavailable_no_image: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entities": list(self.entities),
            "relations": list(self.relations),
            "confidences": dict(self.confidences),
        }
        # Crop metadata only — NEVER bbox coords or pixels (P5.10.0 invariant).
        if self.crops:
            payload["crops"] = [
                {"entity": crop.entity, "confidence": crop.confidence}
                for crop in self.crops
            ]
        if self.crop_missing:
            payload["crop_missing"] = list(self.crop_missing)
        if self.crops_truncated:
            payload["crops_truncated"] = True
        if self.crop_unavailable_no_image:
            payload["crop_unavailable_no_image"] = True
        return payload


_DEFAULT_DETECTOR: BaseStructuredDetector | None = None
_RELATION_RE = re.compile(r"^(.+?)\s+(on|in|at|near)\s+(.+)$")


def detect(
    image: bytes | str | Path | StructuredDetections | dict[str, Any],
    query: str | dict[str, Any] | Iterable[str] | None = None,
    *,
    detector: BaseStructuredDetector | None = None,
    crop_entities: Iterable[str] | None = None,
    auto_crop_top_k: int = 0,
) -> DetectResult:
    """Run query-scoped perception over an image or precomputed detections.

    ``crop_entities`` (P5.10.0, opt-in via ``[TOOL_CALL] args``): request zoomed-in
    crops of the named entities as a follow-up image observation. Default ``None``
    = current behavior (no crops), so historical data/calls are byte-unaffected.
    The crop only *locates + supplies better pixels*; state judgment stays in the
    model (Task-Blindness: crops carry no instruction/GT-derived semantics).
    """
    payload = _coerce_yolo_payload(image, detector=detector)
    visible_confidences = _collect_visible_entity_confidences(payload)
    raw_relations = _build_scene_relations(payload, None)
    parsed_relations = [
        parsed
        for relation in raw_relations
        if (parsed := _parse_relation(relation)) is not None
    ]

    scope = _build_query_scope(query)
    selected_relations: list[str] = []
    selected_entities: list[str] = []
    selected_confidences: dict[str, float] = {}

    def include_entity(entity: str) -> None:
        if entity not in selected_entities:
            selected_entities.append(entity)
        if entity in visible_confidences:
            selected_confidences[entity] = visible_confidences[entity]

    for relation_text, entity, _verb, fixture in parsed_relations:
        if not _relation_matches_scope(entity, fixture, scope):
            continue
        if relation_text not in selected_relations:
            selected_relations.append(relation_text)
        include_entity(entity)
        include_entity(fixture)
        relation_conf = _relation_confidence(entity, fixture, visible_confidences)
        if relation_conf is not None:
            selected_confidences[relation_text] = relation_conf

    for entity, confidence in visible_confidences.items():
        if _entity_matches_scope(entity, scope):
            include_entity(entity)
            selected_confidences[entity] = confidence

    # P5.10.2 probe: when the model did not request crops, auto-pick the ≤k
    # least-confident portable hits (most uncertain = most worth a zoomed look).
    if crop_entities is None and auto_crop_top_k > 0:
        crop_entities = _auto_crop_entities(
            selected_entities, selected_confidences, auto_crop_top_k
        )

    crops: list[CropObservation] = []
    crops_truncated = False
    crop_missing: list[str] = []
    crop_unavailable = False
    if crop_entities:
        crops, crops_truncated, crop_missing, crop_unavailable = _build_crops(
            image, payload, crop_entities
        )
        for crop in crops:
            include_entity(crop.entity)
            selected_confidences[crop.entity] = crop.confidence
        for entity in crop_missing:
            # Undetected requested entity: no crop, confidence pinned to 0.0.
            selected_confidences.setdefault(entity, 0.0)

    return DetectResult(
        entities=selected_entities,
        relations=selected_relations,
        confidences=selected_confidences,
        crops=crops,
        crops_truncated=crops_truncated,
        crop_missing=crop_missing,
        crop_unavailable_no_image=crop_unavailable,
    )


def _auto_crop_entities(
    entities: list[str],
    confidences: dict[str, float],
    top_k: int,
) -> list[str]:
    """Pick the ≤k least-confident portable entities among the detect hits."""
    portable = [entity for entity in entities if is_portable_entity(entity)]
    portable.sort(key=lambda entity: (confidences.get(entity, 0.0), entity))
    return portable[: max(0, int(top_k))]


def _build_crops(
    image: bytes | str | Path | StructuredDetections | dict[str, Any],
    payload: dict[str, Any],
    crop_entities: Iterable[str],
) -> tuple[list[CropObservation], bool, list[str], bool]:
    """Produce ≤2 deterministic entity crops from the original image pixels.

    Returns ``(crops, crops_truncated, crop_missing, crop_unavailable_no_image)``.
    """
    requested: list[str] = []
    for raw in crop_entities:
        name = normalize_entity_name(str(raw)) or str(raw).strip().lower()
        if name and name not in requested:
            requested.append(name)
    crops_truncated = len(requested) > _CROP_MAX
    requested = requested[:_CROP_MAX]
    if not requested:
        return [], crops_truncated, [], False

    pil_image = _load_pil_image(image)
    if pil_image is None:
        # Caller passed a cached payload (no pixels): degrade, do not raise.
        return [], crops_truncated, [], True

    best_boxes = _best_instance_bboxes(payload)
    crops: list[CropObservation] = []
    crop_missing: list[str] = []
    for name in requested:
        info = best_boxes.get(name)
        if info is None:
            crop_missing.append(name)
            continue
        bbox, confidence = info
        crop_image = _expand_and_crop(pil_image, bbox)
        crops.append(CropObservation(entity=name, image=crop_image, confidence=confidence))
    return crops, crops_truncated, crop_missing, False


def _load_pil_image(
    image: bytes | str | Path | StructuredDetections | dict[str, Any],
) -> Any | None:
    """Load original RGB pixels if available; ``None`` for payload-only inputs."""
    from PIL import Image

    try:
        if isinstance(image, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(image))).convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(Path(image).expanduser()).convert("RGB")
    except Exception:
        return None
    return None  # StructuredDetections / dict payload carries no pixels.


def _best_instance_bboxes(payload: dict[str, Any]) -> dict[str, tuple[list[int], float]]:
    """Map canonical label -> (highest-confidence instance bbox, confidence)."""
    best: dict[str, tuple[list[int], float]] = {}
    for group in _iter_detection_groups(payload):
        label = _canonical_entity(group.get("label") or group.get("name"))
        if not label:
            continue
        group_conf = _coerce_confidence(group.get("best_confidence"))
        chosen_bbox: list[int] | None = None
        chosen_conf = -1.0
        for inst in group.get("instances") or []:
            if not isinstance(inst, dict):
                continue
            raw_bbox = inst.get("bbox_xyxy") or inst.get("bbox") or inst.get("xyxy")
            if not (isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4):
                continue
            conf = _coerce_confidence(inst.get("confidence"))
            if conf > chosen_conf:
                chosen_conf = conf
                chosen_bbox = [int(round(float(v))) for v in raw_bbox]
        if chosen_bbox is None:
            continue
        if chosen_conf < 0:
            chosen_conf = group_conf
        best[label] = (chosen_bbox, max(0.0, chosen_conf))
    return best


def _expand_and_crop(pil_image: Any, bbox: list[int]) -> Any:
    """Deterministic crop: 25% pad -> min 112px box -> clamp -> upsample short side≥224."""
    from PIL import Image

    width, height = pil_image.size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    box_w = x2 - x1
    box_h = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    # 25% padding each side, then floor each side to at least 112px (pre-clamp).
    new_w = max(box_w * (1.0 + 2.0 * _CROP_EXPAND_RATIO), float(_CROP_MIN_BOX))
    new_h = max(box_h * (1.0 + 2.0 * _CROP_EXPAND_RATIO), float(_CROP_MIN_BOX))
    nx1 = int(round(max(0.0, min(cx - new_w / 2.0, float(width)))))
    ny1 = int(round(max(0.0, min(cy - new_h / 2.0, float(height)))))
    nx2 = int(round(max(0.0, min(cx + new_w / 2.0, float(width)))))
    ny2 = int(round(max(0.0, min(cy + new_h / 2.0, float(height)))))
    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)
    crop = pil_image.crop((nx1, ny1, nx2, ny2))
    crop_w, crop_h = crop.size
    short_side = min(crop_w, crop_h)
    if 0 < short_side < _CROP_MIN_SHORT_SIDE:
        scale = _CROP_MIN_SHORT_SIDE / short_side
        crop = crop.resize(
            (max(1, round(crop_w * scale)), max(1, round(crop_h * scale))),
            Image.LANCZOS,
        )
    return crop


def _get_default_detector() -> BaseStructuredDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = UltralyticsYoloDetector()
    return _DEFAULT_DETECTOR


def _coerce_yolo_payload(
    image: bytes | str | Path | StructuredDetections | dict[str, Any],
    *,
    detector: BaseStructuredDetector | None,
) -> dict[str, Any]:
    if isinstance(image, StructuredDetections):
        return dict(image.prompt_payload)
    if isinstance(image, dict):
        return dict(image)

    image_bytes = _read_image_bytes(image)
    active_detector = detector or _get_default_detector()
    detections = active_detector.detect(image_bytes)
    return dict(detections.prompt_payload)


def _read_image_bytes(image: bytes | str | Path) -> bytes:
    if isinstance(image, bytes):
        return image
    path = Path(image).expanduser()
    return path.read_bytes()



def _collect_visible_entity_confidences(payload: dict[str, Any]) -> dict[str, float]:
    confidences: dict[str, float] = {}
    for group in _iter_detection_groups(payload):
        label = _canonical_entity(group.get("label") or group.get("name"))
        if not label:
            continue
        confidence = _coerce_confidence(group.get("best_confidence"))
        prior = confidences.get(label)
        if prior is None or confidence > prior:
            confidences[label] = confidence
    return confidences


def _iter_detection_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped = payload.get("grouped_detections")
    if isinstance(grouped, list):
        return [group for group in grouped if isinstance(group, dict)]
    visible_entities = payload.get("visible_entities")
    if isinstance(visible_entities, list):
        return [group for group in visible_entities if isinstance(group, dict)]
    return []


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        raw = float(value)
    else:
        try:
            raw = float(str(value).strip())
        except ValueError:
            return 0.0
    if raw > 1.0:
        raw = raw / 100.0
    return round(max(0.0, min(raw, 1.0)), 4)


def _parse_relation(relation: str) -> tuple[str, str, str, str] | None:
    match = _RELATION_RE.match(str(relation).strip().lower())
    if not match:
        return None
    entity = _canonical_entity(match.group(1))
    fixture = _canonical_entity(match.group(3))
    if not entity or not fixture:
        return None
    verb = match.group(2)
    return f"{entity} {verb} {fixture}", entity, verb, fixture


def _canonical_entity(value: Any) -> str | None:
    normalized = normalize_entity_name(str(value)) if value is not None else None
    if not normalized:
        return None
    return normalized if get_entity_profile(normalized) is not None else None


def _build_query_scope(query: str | dict[str, Any] | Iterable[str] | None) -> dict[str, set[str]]:
    text_parts: list[str] = []
    names: set[str] = set()
    tags: set[str] = set()

    if isinstance(query, dict):
        for key in ("query", "text", "relation"):
            value = query.get(key)
            if isinstance(value, str):
                text_parts.append(value)
        for key in ("name", "entity", "label"):
            value = query.get(key)
            if isinstance(value, str):
                names.add(value)
        for key in ("names", "entities", "labels"):
            value = query.get(key)
            if isinstance(value, (list, tuple, set)):
                names.update(str(item) for item in value)
            elif isinstance(value, str):
                names.add(value)
        tag_value = query.get("tag")
        if isinstance(tag_value, str):
            tags.add(tag_value)
        tag_values = query.get("tags")
        if isinstance(tag_values, (list, tuple, set)):
            tags.update(str(item) for item in tag_values)
    elif isinstance(query, str):
        text_parts.append(query)
    elif query is not None:
        names.update(str(item) for item in query)

    text = " ".join(text_parts)
    for mention in find_entity_mentions(text):
        names.add(mention.canonical_name)
    tags.update(_extract_tags(text))

    normalized_names = {
        normalized
        for name in names
        if (normalized := normalize_entity_name(name))
    }
    normalized_tags = {_normalize_tag(tag) for tag in tags if _normalize_tag(tag)}
    for tag in list(normalized_tags):
        normalized_names.update(
            name
            for name in ENTITY_PROFILES
            if entity_has_tag(name, tag)
        )

    return {
        "entities": normalized_names,
        "tags": normalized_tags,
        "unscoped": set() if (normalized_names or normalized_tags) else {"true"},
    }


def _extract_tags(text: str) -> set[str]:
    tokens = {
        token.strip().lower().replace(" ", "_")
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_ -]*", text or "")
    }
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", (text or "").lower()))
    candidates = tokens | words | {word.rstrip("s") for word in words if word.endswith("s")}
    known_tags = {
        tag
        for profile in ENTITY_PROFILES.values()
        for tag in profile.tags
    }
    return {candidate for candidate in candidates if candidate in known_tags}


def _normalize_tag(tag: str) -> str:
    normalized = str(tag).strip().lower().replace(" ", "_")
    if normalized.endswith("s"):
        singular = normalized[:-1]
        known_tags = {
            tag_value
            for profile in ENTITY_PROFILES.values()
            for tag_value in profile.tags
        }
        if singular in known_tags:
            return singular
    return normalized


def _entity_matches_scope(entity: str, scope: dict[str, set[str]]) -> bool:
    if scope.get("unscoped"):
        return True
    if entity in scope["entities"]:
        return True
    return any(entity_has_tag(entity, tag) for tag in scope["tags"])


def _relation_matches_scope(
    entity: str,
    fixture: str,
    scope: dict[str, set[str]],
) -> bool:
    if scope.get("unscoped"):
        return True
    return _entity_matches_scope(entity, scope) or _entity_matches_scope(fixture, scope)


def _relation_confidence(
    entity: str,
    fixture: str,
    confidences: dict[str, float],
) -> float | None:
    values = [
        confidences[name]
        for name in (entity, fixture)
        if name in confidences
    ]
    if not values:
        return None
    return round(min(values), 4)
