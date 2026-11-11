from __future__ import annotations

import os
from contextlib import contextmanager
from abc import ABC, abstractmethod
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from nsdl.catalog import (
    CONTAINER_TAG,
    PORTABLE_ENTITY_TAG,
    ROBOT_PROFILES,
    SUPPORT_TAG,
    TOOL_TAG,
    entity_has_tag,
    get_entity_profile,
    normalize_entity_name,
    normalize_robot_name,
)
from nsdl.schema import DetectionGroup, DetectionInstance, StructuredDetections


REPO_ROOT = Path(__file__).resolve().parents[1]

# The frozen LOCUS full34 detector (YOLO.md §7.5.0). Switched in on 2026-08-07
# once §7.2's ten report items were all delivered and the §7.5.3 VIKI bracket
# came back [-0.011, +0.042] -- containing 0, i.e. no measurable loss of the old
# capability. Before that the default was the full20 `checkpoints/best.pt`,
# which cannot see 14 of the 34 LOCUS classes at all, so every LOCUS run that
# forgot to pass weights was silently perception-crippled.
DEFAULT_YOLO26_RUN_DIR = (
    REPO_ROOT / "checkpoints" / "yolo26l_locus_full34_v1.pt"
)
DEFAULT_YOLO26_WEIGHTS = DEFAULT_YOLO26_RUN_DIR
# The pre-switch full20 detector. It is NOT a fallback -- it is the fixed
# caliber of the legacy cross-version VIKI protocol (dataset_and_train.md L317),
# whose recorded SFT numbers are only comparable against this exact file. Pass
# it explicitly when reproducing those; never let it be reached by inheritance.
LEGACY_VIKI_FULL20_WEIGHTS = REPO_ROOT / "checkpoints" / "best.pt"
# Per-benchmark detector caliber (2026-09-05 evening, user ruling (a) after
# plan.md 序 25 步 0): the 34-class detector has a table-recall regression on
# VIKI images (test 1314 -> 980 of 1800 detected; 9.6% of images lose an
# "x on table" relation; facts:D3b), so each benchmark keeps the detector its
# corpus and every recorded bucket were produced with. This is EXPLICIT, not a
# hidden default: entry points resolve the caliber from --level and refuse a
# cross-benchmark file unless told so.
VIKI_DETECTOR_WEIGHTS = LEGACY_VIKI_FULL20_WEIGHTS          # VIKI-L1 / VIKI-L2: 20 classes
LOCUS_DETECTOR_WEIGHTS = DEFAULT_YOLO26_WEIGHTS               # LOCUS: 34 classes
DETECTOR_WEIGHTS_BY_LEVEL = {
    "VIKI-L1": VIKI_DETECTOR_WEIGHTS,
    "VIKI-L2": VIKI_DETECTOR_WEIGHTS,
    "LOCUS": LOCUS_DETECTOR_WEIGHTS,
}
CALIBER_MISMATCH_OPT_IN_ENV = "LNSDL_ALLOW_DETECTOR_CALIBER_MISMATCH"
DEFAULT_DETECTOR_BACKEND = "ultralytics_yolo26"


@contextmanager
def _ultralytics_import_without_dist_barrier():
    """Hide RANK/LOCAL_RANK from ultralytics while it is first imported.

    ultralytics/utils/__init__.py reads LOCAL_RANK at import and wraps its
    SettingsManager init in torch_distributed_zero_first(), which calls
    torch.distributed.barrier() on the *default* process group whenever a
    group is initialized and LOCAL_RANK not in {-1, 0}. Inside an RL rollout
    worker that group is the FSDP world, and the import happens lazily on the
    first detect call -- so the first rank whose batch contains a detect call
    blocks on a barrier no other rank joins (LOCUS 3B-RL 2026-09-10: rank 3
    spun at 100% util / 75 W for 30 min; the 7B runs only survived because
    ~96% of their trajectories call detect, so every rank imported it in
    lockstep). Predict-only use never needs those constants.
    """
    saved = {k: os.environ.pop(k) for k in ("RANK", "LOCAL_RANK") if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def detector_weights_for_level(level: str) -> Path:
    """The detector caliber that belongs to a benchmark level (facts:D3a)."""
    try:
        return DETECTOR_WEIGHTS_BY_LEVEL[level]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark level {level!r}; expected one of {sorted(DETECTOR_WEIGHTS_BY_LEVEL)}") from exc


def _same_file(a: str | Path | None, b: Path) -> bool:
    if a is None:
        return False
    try:
        return Path(a).expanduser().resolve() == b.resolve()
    except OSError:
        return Path(a).name == b.name


def is_legacy_full20_weights(weights_path: str | Path | None) -> bool:
    return _same_file(weights_path, LEGACY_VIKI_FULL20_WEIGHTS)


def caliber_mismatch_allowed() -> bool:
    return os.environ.get(CALIBER_MISMATCH_OPT_IN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def assert_detector_caliber_for_level(weights_path: str | Path | None, level: str, *, caller: str) -> None:
    """Refuse a detector file that belongs to the OTHER benchmark's caliber.

    Every VIKI-L2 v3.4.1 bucket before 2026-09-05 silently ran on whatever the
    launcher's private default was, and no bucket recorded which file; the two
    calibers are not interchangeable (facts:D3b). A deliberate cross-caliber
    run (e.g. measuring the 20->34 effect) sets the opt-in env var.
    """
    if weights_path is None or caliber_mismatch_allowed():
        return
    expected = detector_weights_for_level(level)
    if _same_file(weights_path, expected):
        return
    other = {v.resolve() for v in DETECTOR_WEIGHTS_BY_LEVEL.values()} - {expected.resolve()}
    if any(_same_file(weights_path, o) for o in other):
        raise SystemExit(
            f"[{caller}] detector weights {weights_path} belong to the other benchmark's "
            f"caliber; {level} uses {expected} (facts:D3a/D3b — the two are not "
            f"interchangeable). Set {CALIBER_MISMATCH_OPT_IN_ENV}=1 only for a deliberate "
            f"cross-caliber measurement, and label the bucket."
        )


def describe_detector_weights(weights_path: str | Path | None) -> dict[str, Any]:
    """Provenance block for result artifacts: path, sha256, class count.

    One file hash per path (cached), independent of whether the model has
    been loaded, so it can be written into every sample record.
    """
    if weights_path is None:
        return {"detector_weights": None}
    path = Path(weights_path).expanduser()
    info: dict[str, Any] = {
        "detector_weights": str(path),
        "detector_weights_sha256": None,
        "detector_num_classes": None,
        "detector_is_legacy_full20": is_legacy_full20_weights(path),
    }
    if not path.is_file():
        return info
    sha, n = _describe_cached(path)
    info["detector_weights_sha256"] = sha
    info["detector_num_classes"] = n
    return info


_DESCRIBE_CACHE: dict[str, tuple[str, int | None]] = {}


def _describe_cached(path: Path) -> tuple[str, int | None]:
    key = str(path.resolve())
    if key not in _DESCRIBE_CACHE:
        import hashlib
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _DESCRIBE_CACHE[key] = (h.hexdigest(), _count_classes_in_checkpoint(path))
    return _DESCRIBE_CACHE[key]


def _count_classes_in_checkpoint(path: Path) -> int | None:
    """Read ``names`` straight from the checkpoint; None if it cannot be read."""
    try:
        import torch
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 -- provenance must never break an eval
        return None
    for key in ("ema", "model"):
        model = ck.get(key) if isinstance(ck, dict) else None
        names = getattr(model, "names", None)
        if names:
            return len(names)
    return None
# D2（感知通用性 bracket）：现成开放词表检测器，零样本换装，不训练。
OPENVOCAB_DETECTOR_BACKEND = "yoloworld_openvocab"

_CATEGORY_PRIORITY = {
    "robot": 0,
    "tool": 1,
    "portable_object": 2,
    "container": 3,
    "support_area": 4,
    "fixture": 5,
    "entity": 6,
    "unknown": 7,
}

_PER_LABEL_MAX_CONFIDENCE_THRESHOLD = {
    "bowl": 0.2,
    "plate": 0.2,
}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_relative = (REPO_ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_supported_labels(names_payload: Any) -> list[str]:
    if isinstance(names_payload, dict):
        ordered_items = sorted(names_payload.items(), key=lambda item: int(item[0]))
        labels = [str(label) for _, label in ordered_items]
    elif isinstance(names_payload, list):
        labels = [str(label) for label in names_payload]
    else:
        labels = []
    return _dedupe_preserve_order([_canonicalize_label(label) for label in labels if label])


def _canonicalize_label(label: str) -> str:
    stripped = str(label).strip()
    robot_label = normalize_robot_name(stripped)
    if robot_label:
        return robot_label
    normalized = normalize_entity_name(stripped)
    if normalized:
        return normalized
    return stripped.replace("_", " ")


def _infer_category(label: str) -> str:
    if label in ROBOT_PROFILES:
        return "robot"
    if entity_has_tag(label, TOOL_TAG):
        return "tool"
    if entity_has_tag(label, CONTAINER_TAG):
        return "container"
    if entity_has_tag(label, SUPPORT_TAG):
        return "support_area"
    if entity_has_tag(label, PORTABLE_ENTITY_TAG):
        return "portable_object"
    entity_profile = get_entity_profile(label)
    if entity_profile and entity_profile.entity_type == "fixture":
        return "fixture"
    if entity_profile:
        return "entity"
    return "unknown"


def _clamp_int(value: float, lower: int, upper: int) -> int:
    rounded = int(round(float(value)))
    return max(lower, min(rounded, upper))


def _coarse_region(center_x: int, center_y: int, width: int, height: int) -> str:
    def bucket(value: int, size: int, left: str, mid: str, right: str) -> str:
        if size <= 0:
            return mid
        ratio = value / size
        if ratio < 1 / 3:
            return left
        if ratio > 2 / 3:
            return right
        return mid

    horizontal = bucket(center_x, width, "left", "center", "right")
    vertical = bucket(center_y, height, "top", "middle", "bottom")
    if horizontal == "center" and vertical == "middle":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"


def _instance_to_payload(instance: DetectionInstance) -> dict[str, Any]:
    return {
        "rank": instance.rank_within_label,
        "confidence": instance.confidence,
        "region": instance.region,
        "center_xy": list(instance.center_xy),
        "bbox_xyxy": list(instance.bbox_xyxy),
    }


def _group_to_payload(group: DetectionGroup) -> dict[str, Any]:
    return {
        "label": group.label,
        "category": group.category,
        "count": group.count,
        "best_confidence": group.best_confidence,
        "regions": list(group.regions),
        "instances": [_instance_to_payload(instance) for instance in group.instances],
    }


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


def _point_in_bbox(center_xy: list[int], bbox_xyxy: list[int]) -> bool:
    if len(center_xy) != 2 or len(bbox_xyxy) != 4:
        return False
    x, y = center_xy
    x1, y1, x2, y2 = bbox_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def _is_table_support_probe_label(label: str) -> bool:
    return label in {"plate", "bowl"} or entity_has_tag(label, "fruit")


def _build_table_item_overlap_summary(
    grouped_detections: list[DetectionGroup],
) -> dict[str, Any] | None:
    table_instances: list[DetectionInstance] = []
    candidate_instances: list[DetectionInstance] = []

    for group in grouped_detections:
        if group.label == "table":
            table_instances.extend(group.instances)
        elif _is_table_support_probe_label(group.label):
            candidate_instances.extend(group.instances)

    if not table_instances and not candidate_instances:
        return None

    ranked_tables = sorted(
        table_instances,
        key=lambda instance: (-instance.confidence, instance.rank_within_label),
    )

    summary_items: list[dict[str, Any]] = []
    for candidate in candidate_instances:
        item_area = _bbox_area(candidate.bbox_xyxy)
        table_matches: list[dict[str, Any]] = []
        for table_rank, table_instance in enumerate(ranked_tables, start=1):
            intersection_area = _bbox_intersection_area(
                candidate.bbox_xyxy,
                table_instance.bbox_xyxy,
            )
            overlap_ratio = (
                round(intersection_area / item_area, 3)
                if item_area > 0
                else 0.0
            )
            center_in_table = _point_in_bbox(
                candidate.center_xy,
                table_instance.bbox_xyxy,
            )
            fully_contained = item_area > 0 and intersection_area >= item_area
            support_like_contained = center_in_table or overlap_ratio >= 0.5
            table_matches.append(
                {
                    "table_rank": table_rank,
                    "table_confidence": table_instance.confidence,
                    "table_region": table_instance.region,
                    "table_bbox_xyxy": list(table_instance.bbox_xyxy),
                    "intersection_area": intersection_area,
                    "item_overlap_ratio": overlap_ratio,
                    "center_in_table": center_in_table,
                    "fully_contained": fully_contained,
                    "support_like_contained": support_like_contained,
                }
            )

        table_matches.sort(
            key=lambda match: (
                -float(match.get("item_overlap_ratio", 0.0)),
                not bool(match.get("center_in_table")),
                -int(match.get("table_confidence", 0)),
            )
        )
        summary_items.append(
            {
                "label": candidate.label,
                "rank_within_label": candidate.rank_within_label,
                "confidence": candidate.confidence,
                "region": candidate.region,
                "center_xy": list(candidate.center_xy),
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "table_matches": table_matches,
                "best_table_match": table_matches[0] if table_matches else None,
            }
        )

    summary_items.sort(
        key=lambda item: (
            item.get("label", ""),
            int(item.get("rank_within_label", 0)),
        )
    )

    return {
        "support_like_containment_rule": (
            "support_like_contained = center_in_table or "
            "(intersection_area(item_bbox, table_bbox) / item_bbox_area >= 0.5)"
        ),
        "ranking_rule": (
            "table_matches are sorted by item_overlap_ratio desc, then center_in_table, "
            "then table confidence."
        ),
        "table_instances": [
            {
                "table_rank": table_rank,
                "confidence": table_instance.confidence,
                "region": table_instance.region,
                "center_xy": list(table_instance.center_xy),
                "bbox_xyxy": list(table_instance.bbox_xyxy),
            }
            for table_rank, table_instance in enumerate(ranked_tables, start=1)
        ],
        "items": summary_items,
    }


class BaseStructuredDetector(ABC):
    @abstractmethod
    def detect(self, image_bytes: bytes) -> StructuredDetections:
        raise NotImplementedError

    def warmup(self) -> None:
        """Resolve everything the first ``detect`` would need, and raise if it can't.

        Callers that requested a detector on purpose use this to turn a lazy
        mid-run failure into a startup failure. Backends with nothing to load
        (oracle, off) inherit the no-op.
        """
        return None


class UltralyticsYoloDetector(BaseStructuredDetector):
    def __init__(
        self,
        *,
        run_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
        data_yaml_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        max_det: int = 64,
        device: str | None = None,
        predict_imgsz: int | None = None,
    ) -> None:
        resolved_weights_path = _resolve_repo_path(weights_path)
        resolved_run_dir = _resolve_repo_path(run_dir)
        if resolved_run_dir is None:
            # Derive the run dir from explicit weights rather than falling back
            # to the module default, or a caller pinning `weights_path` to the
            # legacy full20 checkpoint would silently pick up the full34 default
            # run dir -- and with it that run's `args.yaml` -> data.yaml -> the
            # WRONG class names for the weights actually loaded.
            resolved_run_dir = (resolved_weights_path.parent
                                if resolved_weights_path is not None
                                else DEFAULT_YOLO26_RUN_DIR)
        if resolved_weights_path is None and resolved_run_dir.suffix == ".pt":
            resolved_weights_path = resolved_run_dir
            resolved_run_dir = resolved_run_dir.parent
        self.run_dir = resolved_run_dir
        self.weights_path = resolved_weights_path or (self.run_dir / "weights" / "best.pt")
        self.data_yaml_path = _resolve_repo_path(data_yaml_path) or self._infer_data_yaml_path()
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.device = device
        # None => 不传 imgsz，由 ultralytics 从 ckpt overrides 继承（full34 是 960）。
        # 保持 None 时 predict_kwargs 与本改动前逐键相同。只有换装检测器需要显式钉住，
        # 因为它自带的 overrides 是 640，按各自默认跑会把分辨率差读成能力差。
        self.predict_imgsz = int(predict_imgsz) if predict_imgsz else None
        self._model = None
        self._supported_labels = self._load_supported_labels()

    def _effective_confidence_threshold_for_label(self, label: str) -> float:
        override_threshold = _PER_LABEL_MAX_CONFIDENCE_THRESHOLD.get(label)
        if override_threshold is None:
            return self.confidence_threshold
        return min(self.confidence_threshold, float(override_threshold))

    def _predict_confidence_threshold(self) -> float:
        thresholds = [self.confidence_threshold]
        thresholds.extend(
            min(self.confidence_threshold, float(override_threshold))
            for override_threshold in _PER_LABEL_MAX_CONFIDENCE_THRESHOLD.values()
        )
        return min(thresholds)

    def _build_threshold_payload(self) -> dict[str, Any]:
        threshold_payload: dict[str, Any] = {
            "confidence": int(round(self.confidence_threshold * 100)),
            "iou": int(round(self.iou_threshold * 100)),
        }
        per_label_confidence = {
            label: int(round(self._effective_confidence_threshold_for_label(label) * 100))
            for label in sorted(_PER_LABEL_MAX_CONFIDENCE_THRESHOLD)
            if self._effective_confidence_threshold_for_label(label) < self.confidence_threshold
        }
        predict_confidence = self._predict_confidence_threshold()
        if predict_confidence < self.confidence_threshold:
            threshold_payload["model_predict_confidence"] = int(round(predict_confidence * 100))
        if per_label_confidence:
            threshold_payload["per_label_confidence"] = per_label_confidence
        return threshold_payload

    def _infer_data_yaml_path(self) -> Path | None:
        args_path = self.run_dir / "args.yaml"
        if not args_path.exists():
            return None
        args_payload = _load_yaml(args_path)
        data_path = args_payload.get("data")
        return _resolve_repo_path(data_path)

    def _load_supported_labels(self) -> list[str]:
        if self.data_yaml_path and self.data_yaml_path.exists():
            data_payload = _load_yaml(self.data_yaml_path)
            supported_labels = _normalize_supported_labels(data_payload.get("names"))
            if supported_labels:
                return supported_labels
        return []

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        try:
            with _ultralytics_import_without_dist_barrier():
                from ultralytics import YOLO
        except ImportError as exc:
            # Keep the underlying import error in the *message*, not just the
            # __cause__: this is raised inside the agentic tool loop, whose
            # result records only str(exc). On the secondary machine the real
            # cause was `libGL.so.1: cannot open shared object file` (cv2 pulled
            # in by ultralytics, no system OpenGL in the container) -- with the
            # generic text alone that reads as "someone forgot to pip install",
            # and the run degrades to a silently perception-free eval.
            raise RuntimeError(
                "ultralytics is required for the YOLO detector backend. "
                "Please install the YOLO runtime environment before enabling "
                f"scene-facts refinement with detector support. Underlying import error: {exc!r}"
            ) from exc
        self._model = YOLO(str(self.weights_path))
        return self._model

    def warmup(self) -> None:
        self._load_model()

    def detect(self, image_bytes: bytes) -> StructuredDetections:
        if not image_bytes:
            raise ValueError("YOLO detector requires non-empty image bytes.")

        model = self._load_model()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = image.size
        predict_confidence_threshold = self._predict_confidence_threshold()

        predict_kwargs: dict[str, Any] = {
            "source": image,
            "verbose": False,
            "conf": predict_confidence_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_det,
        }
        if self.device:
            predict_kwargs["device"] = self.device
        if self.predict_imgsz:
            predict_kwargs["imgsz"] = self.predict_imgsz

        predictions = model.predict(**predict_kwargs)
        if not predictions:
            return self._build_empty_result(image_width=image_width, image_height=image_height)

        prediction = predictions[0]
        names_payload = (
            getattr(prediction, "names", None)
            or getattr(model, "names", None)
            or {}
        )
        supported_labels = self._supported_labels or _normalize_supported_labels(names_payload)

        boxes = getattr(prediction, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._build_empty_result(
                image_width=image_width,
                image_height=image_height,
                supported_labels=supported_labels,
            )

        xyxy_values = boxes.xyxy.tolist()
        confidence_values = boxes.conf.tolist()
        class_values = boxes.cls.tolist()

        flat_instances: list[DetectionInstance] = []
        for xyxy, confidence, class_index in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):
            raw_label = names_payload[int(class_index)]
            label = _canonicalize_label(raw_label)
            if float(confidence) < self._effective_confidence_threshold_for_label(label):
                continue
            x1 = _clamp_int(xyxy[0], 0, image_width)
            y1 = _clamp_int(xyxy[1], 0, image_height)
            x2 = _clamp_int(xyxy[2], 0, image_width)
            y2 = _clamp_int(xyxy[3], 0, image_height)
            center_x = _clamp_int((x1 + x2) / 2, 0, image_width)
            center_y = _clamp_int((y1 + y2) / 2, 0, image_height)
            flat_instances.append(
                DetectionInstance(
                    label=label,
                    category=_infer_category(label),
                    confidence=int(round(float(confidence) * 100)),
                    bbox_xyxy=[x1, y1, x2, y2],
                    center_xy=[center_x, center_y],
                    region=_coarse_region(center_x, center_y, image_width, image_height),
                )
            )

        flat_instances.sort(
            key=lambda instance: (
                _CATEGORY_PRIORITY.get(instance.category, 99),
                instance.label,
                -instance.confidence,
                instance.center_xy[1],
                instance.center_xy[0],
            )
        )

        grouped_instances: OrderedDict[tuple[str, str], list[DetectionInstance]] = OrderedDict()
        for instance in flat_instances:
            group_key = (instance.category, instance.label)
            grouped_instances.setdefault(group_key, [])
            instance.rank_within_label = len(grouped_instances[group_key]) + 1
            grouped_instances[group_key].append(instance)

        grouped_detections: list[DetectionGroup] = []
        for (category, label), instances in grouped_instances.items():
            grouped_detections.append(
                DetectionGroup(
                    label=label,
                    category=category,
                    count=len(instances),
                    best_confidence=max(instance.confidence for instance in instances),
                    regions=_dedupe_preserve_order([instance.region for instance in instances]),
                    instances=list(instances),
                )
            )

        visible_robots = [
            group for group in grouped_detections if group.category == "robot"
        ]
        visible_entities = [
            group for group in grouped_detections if group.category != "robot"
        ]

        return StructuredDetections(
            detector_backend=DEFAULT_DETECTOR_BACKEND,
            model_path=str(self.weights_path),
            run_name=self.run_dir.name if self.run_dir else None,
            image_width=image_width,
            image_height=image_height,
            confidence_threshold=int(round(self.confidence_threshold * 100)),
            iou_threshold=int(round(self.iou_threshold * 100)),
            supported_labels=supported_labels,
            visible_labels=[group.label for group in grouped_detections],
            grouped_detections=grouped_detections,
            visible_robots=visible_robots,
            visible_entities=visible_entities,
            prompt_payload=self._build_prompt_payload(
                image_width=image_width,
                image_height=image_height,
                supported_labels=supported_labels,
                grouped_detections=grouped_detections,
                visible_robots=visible_robots,
                visible_entities=visible_entities,
            ),
        )

    def _build_empty_result(
        self,
        *,
        image_width: int,
        image_height: int,
        supported_labels: list[str] | None = None,
    ) -> StructuredDetections:
        supported = list(supported_labels or self._supported_labels)
        return StructuredDetections(
            detector_backend=DEFAULT_DETECTOR_BACKEND,
            model_path=str(self.weights_path),
            run_name=self.run_dir.name if self.run_dir else None,
            image_width=image_width,
            image_height=image_height,
            confidence_threshold=int(round(self.confidence_threshold * 100)),
            iou_threshold=int(round(self.iou_threshold * 100)),
            supported_labels=supported,
            visible_labels=[],
            grouped_detections=[],
            visible_robots=[],
            visible_entities=[],
            prompt_payload=self._build_prompt_payload(
                image_width=image_width,
                image_height=image_height,
                supported_labels=supported,
                grouped_detections=[],
                visible_robots=[],
                visible_entities=[],
            ),
        )

    def _build_prompt_payload(
        self,
        *,
        image_width: int,
        image_height: int,
        supported_labels: list[str],
        grouped_detections: list[DetectionGroup],
        visible_robots: list[DetectionGroup],
        visible_entities: list[DetectionGroup],
    ) -> dict[str, Any]:
        table_item_overlap = _build_table_item_overlap_summary(grouped_detections)
        return {
            "detector_backend": DEFAULT_DETECTOR_BACKEND,
            "model_run": self.run_dir.name if self.run_dir else None,
            "image_size": {
                "width": image_width,
                "height": image_height,
            },
            "thresholds": {
                **self._build_threshold_payload(),
            },
            "supported_labels": list(supported_labels),
            "visible_labels": [group.label for group in grouped_detections],
            "coverage_note": (
                "Only supported_labels are detector-grounded. Missing detections are not proof of absence."
            ),
            "label_semantics": {
                "table": (
                    "Detector label for an independent tabletop in the central scene region. "
                    "Use it as the primary visual anchor for table-related geometric reasoning."
                ),
                "cabinet": (
                    "Detector label for cabinet-like enclosed storage surfaces or doors."
                ),
            },
            "region_overlap_notes": [
                (
                    "table and kitchen island area are not mutually exclusive semantic labels; "
                    "the same surface or object placement can satisfy both."
                ),
            ],
            "table_item_overlap": table_item_overlap,
            "grouped_detections": [_group_to_payload(group) for group in grouped_detections],
            "visible_robots": [_group_to_payload(group) for group in visible_robots],
            "visible_entities": [_group_to_payload(group) for group in visible_entities],
        }


class OpenVocabYoloWorldDetector(UltralyticsYoloDetector):
    """D2：现成开放词表检测器（YOLO-World），零样本换装，不做任何微调。

    刻意继承而非重写：分组、关系结构化、region 归类、table 重叠摘要全部逐字节
    共用父类实现，于是 D2 与主行之间**唯一**的变量是检测器权重本身。

    两处必须显式钉住，否则读数会被当成能力差：

    1. **imgsz。** 生产侧 `detect()` 不传 imgsz，ultralytics 从 ckpt overrides 继承，
       full34 训练档是 960；而 yolov8x-worldv2 自带的 overrides 是 640。按各自默认跑，
       YOLO-World 在实测样图上出 **0 框**、960 上出 4 框（cabinet .460 / bowl .428）。
       "通用检测器全面崩溃"这个结论完全会是分辨率假象——且它与既有先例（OVMM 换通用
       检测器 14.8→0.4）同号，很容易被顺理成章地当作复现。故本 backend 默认 960。
    2. **词表非空。** 父类的 `_infer_data_yaml_path` 从 `run_dir/args.yaml` 推 data.yaml，
       而开放词表权重目录里没有 args.yaml ⇒ 词表为空 ⇒ `set_classes([])` ⇒ 检测器什么都
       不发 ⇒ D2 静默塌回已跑完的 D0，而且从产物上完全看不出来。故这里硬失败。

    词表用**资产名去下划线**的自然语言形式（`wooden_cutting_board` → `wooden cutting board`），
    与 D2 前置审计（outputs/d2_preaudit/preaudit.json）逐词一致；`model.names` 在
    `set_classes` 之后会持有自然语言形式，这里再还原成下游 NSDL 代码认的规范下划线名。

    已登记的阴性结果：6 个机器人类零样本召回**全为 0.0**。换措辞（robot / humanoid robot /
    quadruped robot dog / robotic arm / mobile robot with arm）在同一词表内仍全 0；唯一
    能定位到它们的词是 "person"，但那会把 6 种机器人塌成一个标签，正好抹掉 LOCUS 调度
    依赖的机器人身份信息。故**保留原始资产名词表，没有任何事后调优进入最终配置**。
    """

    def __init__(self, *args, predict_imgsz: int | None = 960, **kwargs) -> None:
        super().__init__(*args, predict_imgsz=predict_imgsz, **kwargs)
        if not self._supported_labels:
            raise ValueError(
                "OpenVocabYoloWorldDetector 需要显式词表，但解析到空 names。"
                f"（data_yaml_path={self.data_yaml_path}，run_dir={self.run_dir}）"
                " 开放词表权重目录没有 args.yaml，必须显式传 --detector-data-yaml；"
                " 否则 set_classes([]) 会让本 arm 静默等价于 detect-off(D0)。"
            )

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO-World weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics(YOLOWorld) is required for the open-vocabulary detector "
                "backend. Note it additionally needs `clip` and `ftfy` at set_classes "
                f"time. Underlying import error: {exc!r}"
            ) from exc
        model = YOLOWorld(str(self.weights_path))
        canonical = list(self._supported_labels)
        model.set_classes([label.replace("_", " ") for label in canonical])
        # set_classes 之后名字是自然语言形式；下游一律按规范下划线名匹配，这里还原回去。
        # 顺序即类索引顺序，与 data.yaml 一致。注意必须写**底层 nn.Module** 的 names：
        # ultralytics 的 Model.names 是只读 property（写它抛 AttributeError），
        # 而 set_classes 本身也是写的 self.model.names。
        canonical_names = {index: label for index, label in enumerate(canonical)}
        model.model.names = canonical_names
        if getattr(model, "predictor", None) is not None:
            model.predictor.model.names = canonical_names
        self._model = model
        return self._model


def build_structured_detector(
    backend: str,
    *,
    run_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.70,
    max_det: int = 64,
    device: str | None = None,
) -> BaseStructuredDetector | None:
    if backend == "none":
        return None
    if backend == DEFAULT_DETECTOR_BACKEND:
        return UltralyticsYoloDetector(
            run_dir=run_dir,
            weights_path=weights_path,
            data_yaml_path=data_yaml_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            device=device,
        )
    if backend == OPENVOCAB_DETECTOR_BACKEND:
        return OpenVocabYoloWorldDetector(
            run_dir=run_dir,
            weights_path=weights_path,
            data_yaml_path=data_yaml_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            device=device,
        )
    raise ValueError(
        f"Unsupported detector backend: {backend}. "
        f"Expected one of: none, {DEFAULT_DETECTOR_BACKEND}, {OPENVOCAB_DETECTOR_BACKEND}."
    )
