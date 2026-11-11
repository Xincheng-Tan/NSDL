from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


_GATE_OPS = {
    "<=": lambda value, threshold: value <= threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    ">": lambda value, threshold: value > threshold,
}


@dataclass(frozen=True)
class AttributeGate:
    """One ``--drop-if`` predicate over an X-AnyLabeling per-shape attribute.

    The renderer GT exporter (``locus_mask_export.make_shape``) writes
    ``visible_pixels``/``visible_frac``/``visible_over_amodal``/``view_cos``/
    ``planarity``/``depth_m`` into each shape's ``attributes``, and they survive a
    human review round-trip. Gating on them HERE rather than only in the exporter
    is what makes the floors adjustable: the exporter needs the segmentation and
    position buffers, which are never written to disk, so moving a floor there
    means re-rendering every image just to re-decide a threshold.

    Two floors of the same shape are NOT interchangeable: ``visible_pixels``
    catches boxes that are small, ``visible_over_amodal`` catches boxes that are
    large but mostly occluded (a drawer front hidden behind a robot). Both cluster
    types showed up in the P7.5b human spot-check, hence a general predicate list
    instead of a single ``--min-visible-pixels``.
    """

    raw: str
    labels: frozenset[str] | None
    attribute: str
    op: str
    threshold: float

    def applies_to(self, label: str) -> bool:
        return self.labels is None or label in self.labels

    def drops(self, attributes: dict) -> bool | None:
        """``True``/``False`` to drop/keep; ``None`` when the attribute is absent.

        Absent is a distinct outcome because it is the normal case for a whole
        source: the 900 hand-drawn VIKI-train shapes carry an empty ``attributes``
        dict, so every pixel gate is silent on them and the caller has to decide
        that policy explicitly rather than inherit it from a KeyError.
        """
        if not isinstance(attributes, dict) or self.attribute not in attributes:
            return None
        try:
            value = float(attributes[self.attribute])
        except (TypeError, ValueError):
            return None
        if math.isnan(value):
            return None
        return _GATE_OPS[self.op](value, self.threshold)


def parse_attribute_gate(spec: str) -> AttributeGate:
    """``[<label>[,<label>...]:]<attribute><op><value>``, e.g.

    ``visible_pixels<24``                       -- every class
    ``cabinet,drawer:visible_pixels<128``       -- fixture floor only
    ``visible_over_amodal<0.05``                -- occluded-beyond-recognition
    """
    text = spec.strip()
    labels: frozenset[str] | None = None
    if ":" in text:
        label_part, _, text = text.partition(":")
        names = [name.strip() for name in label_part.split(",") if name.strip()]
        if not names:
            raise ValueError(f"--drop-if '{spec}' has an empty class list")
        labels = frozenset(names)
        text = text.strip()
    for op in ("<=", ">=", "<", ">"):
        attribute, sep, value = text.partition(op)
        if sep:
            attribute = attribute.strip()
            if not attribute:
                raise ValueError(f"--drop-if '{spec}' has an empty attribute name")
            try:
                threshold = float(value.strip())
            except ValueError:
                raise ValueError(
                    f"--drop-if '{spec}' has a non-numeric threshold '{value.strip()}'"
                ) from None
            return AttributeGate(
                raw=spec.strip(),
                labels=labels,
                attribute=attribute,
                op=op,
                threshold=threshold,
            )
    raise ValueError(
        f"--drop-if '{spec}' has no comparison operator (expected one of <, <=, >, >=)"
    )


def gate_verdict(
    label: str,
    attributes: dict,
    gates: list[AttributeGate],
    drop_when_attribute_missing: bool,
) -> tuple[str | None, bool]:
    """``(raw spec that dropped this box or None, saw_a_missing_attribute)``.

    First matching gate wins, so the reported reason is deterministic in the order
    the flags were passed.
    """
    missing = False
    for gate in gates:
        if not gate.applies_to(label):
            continue
        verdict = gate.drops(attributes)
        if verdict is None:
            missing = True
            if drop_when_attribute_missing:
                return gate.raw, missing
            continue
        if verdict:
            return gate.raw, missing
    return None, missing


@dataclass
class ImageRecord:
    stem: str
    image_path: Path
    xlabel_path: Path
    group: str
    split_group: str
    detections: tuple[tuple[str, tuple[float, float, float, float]], ...]
    lines: tuple[str, ...]
    label_counts: Counter
    shape_type_counts: Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert X-AnyLabeling detection annotations into an Ultralytics "
            "YOLO dataset with deterministic train/val splits."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root directory containing images, searched recursively.",
    )
    parser.add_argument(
        "--xlabels-dir",
        type=Path,
        required=True,
        help="Directory containing X-AnyLabeling JSON files.",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        required=True,
        help="Ordered class list. Unused classes are dropped by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the YOLO dataset.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio. Defaults to 0.10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Deterministic split seed. Defaults to 3407.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize images in the output dataset.",
    )
    parser.add_argument(
        "--keep-unused-classes",
        action="store_true",
        help="Keep classes from --classes-file even if they have zero boxes.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Allow images without matching JSON labels and export empty txt files.",
    )
    parser.add_argument(
        "--split-groups-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON/JSONL manifest used to keep related images in the same "
            "train/val split. Records are matched by image stem."
        ),
    )
    parser.add_argument(
        "--split-group-key",
        default="counterfactual_pair_id",
        help=(
            "Manifest field used as the indivisible split group. Nested meta/locus_meta "
            "fields are supported. Missing/null values fall back to the image stem."
        ),
    )
    parser.add_argument(
        "--drop-if",
        dest="drop_if",
        action="append",
        default=[],
        metavar="[CLASSES:]ATTR<OP>VALUE",
        help=(
            "Drop shapes whose X-AnyLabeling attribute satisfies the predicate. "
            "Repeatable; first match wins. Examples: 'visible_pixels<24', "
            "'cabinet,drawer,fridge,oven:visible_pixels<128', "
            "'visible_over_amodal<0.05'. Operators: <, <=, >, >=."
        ),
    )
    parser.add_argument(
        "--drop-if-attribute-missing",
        action="store_true",
        help=(
            "Treat a shape whose attribute is absent as dropped. Default is to keep "
            "it: hand-drawn sources (VIKI-train) carry no attributes at all, so the "
            "default makes --drop-if a LOCUS-only filter."
        ),
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_images(images_root: Path) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    for path in sorted(images_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in image_paths:
            raise ValueError(
                f"Duplicate image stem '{stem}' found in both "
                f"'{image_paths[stem]}' and '{path}'."
            )
        image_paths[stem] = path
    if not image_paths:
        raise ValueError(f"No images found under {images_root}")
    return image_paths


def discover_xlabels(xlabels_dir: Path) -> dict[str, Path]:
    label_paths: dict[str, Path] = {}
    for path in sorted(xlabels_dir.glob("*.json")):
        stem = path.stem
        if stem in label_paths:
            raise ValueError(f"Duplicate label stem '{stem}' found in {xlabels_dir}")
        label_paths[stem] = path
    if not label_paths:
        raise ValueError(f"No JSON labels found under {xlabels_dir}")
    return label_paths


def _manifest_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("samples", "records", "images"):
            records = payload.get(key)
            if isinstance(records, list):
                return [record for record in records if isinstance(record, dict)]
        if all(isinstance(value, str) for value in payload.values()):
            return [
                {"image": stem, "split_group": split_group}
                for stem, split_group in payload.items()
            ]
    raise ValueError("Split-groups manifest must be a record list or stem->group mapping.")


def _record_field(record: dict, key: str) -> object:
    if key in record:
        return record.get(key)
    for container_key in ("meta", "locus_meta"):
        nested = record.get(container_key)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)
    return None


def load_split_groups(path: Path | None, group_key: str) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Split-groups manifest not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = [
            record
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for record in [json.loads(line)]
            if isinstance(record, dict)
        ]
    else:
        records = _manifest_records(json.loads(path.read_text(encoding="utf-8")))

    split_groups: dict[str, str] = {}
    for record in records:
        image_token = (
            record.get("image")
            or record.get("image_path")
            or record.get("output_path")
            or record.get("filename")
        )
        if not image_token:
            continue
        stem = Path(str(image_token)).stem
        group_value = _record_field(record, group_key)
        if group_value is None:
            group_value = record.get("split_group")
        if group_value is None or str(group_value).strip() == "":
            continue
        split_group = str(group_value)
        prior = split_groups.get(stem)
        if prior is not None and prior != split_group:
            raise ValueError(
                f"Conflicting split groups for image stem {stem!r}: "
                f"{prior!r} != {split_group!r}"
            )
        split_groups[stem] = split_group
    return split_groups


def infer_group(images_root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(images_root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return "__root__"


def stable_key(seed: int, group: str, stem: str) -> str:
    payload = f"{seed}:{group}:{stem}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def normalize_bbox(points: list[list[float]], image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin = clamp(min(xs), 0.0, float(image_width))
    xmax = clamp(max(xs), 0.0, float(image_width))
    ymin = clamp(min(ys), 0.0, float(image_height))
    ymax = clamp(max(ys), 0.0, float(image_height))
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        return None
    x_center = (xmin + xmax) / 2.0 / float(image_width)
    y_center = (ymin + ymax) / 2.0 / float(image_height)
    box_width = width / float(image_width)
    box_height = height / float(image_height)
    return x_center, y_center, box_width, box_height


def build_records(
    images_root: Path,
    image_paths: dict[str, Path],
    xlabel_paths: dict[str, Path],
    declared_classes: list[str],
    allow_missing_labels: bool,
    split_groups: dict[str, str] | None = None,
    gates: list[AttributeGate] | None = None,
    drop_when_attribute_missing: bool = False,
) -> tuple[list[ImageRecord], list[str], Counter, Counter, dict]:
    declared_class_set = set(declared_classes)
    unknown_labels: list[str] = []
    total_label_counts: Counter = Counter()
    total_shape_type_counts: Counter = Counter()
    records: list[ImageRecord] = []
    split_groups = split_groups or {}
    gates = gates or []
    gate_drops: dict[str, Counter] = defaultdict(Counter)
    gate_stats = {
        "shapes_with_attributes": 0,
        "shapes_missing_gated_attribute": 0,
        "images_emptied": 0,
    }

    for stem, image_path in sorted(image_paths.items()):
        xlabel_path = xlabel_paths.get(stem)
        if xlabel_path is None:
            if not allow_missing_labels:
                raise FileNotFoundError(
                    f"Missing X-AnyLabeling label for image '{image_path}'."
                )
            records.append(
                ImageRecord(
                    stem=stem,
                    image_path=image_path,
                    xlabel_path=Path(),
                    group=infer_group(images_root, image_path),
                    split_group=split_groups.get(stem, stem),
                    detections=tuple(),
                    lines=tuple(),
                    label_counts=Counter(),
                    shape_type_counts=Counter(),
                )
            )
            continue

        data = json.loads(xlabel_path.read_text(encoding="utf-8"))
        image_width = int(data["imageWidth"])
        image_height = int(data["imageHeight"])
        label_counts: Counter = Counter()
        shape_type_counts: Counter = Counter()
        boxes: list[tuple[str, tuple[float, float, float, float]]] = []

        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).strip() or "unknown"
            shape_type_counts[shape_type] += 1
            total_shape_type_counts[shape_type] += 1

            if not label:
                continue
            if label not in declared_class_set:
                unknown_labels.append(label)
                continue

            points = shape.get("points") or []
            bbox = normalize_bbox(points, image_width=image_width, image_height=image_height)
            if bbox is None:
                continue

            attributes = shape.get("attributes") or {}
            if attributes:
                gate_stats["shapes_with_attributes"] += 1
            if gates:
                dropped_by, saw_missing = gate_verdict(
                    label=label,
                    attributes=attributes,
                    gates=gates,
                    drop_when_attribute_missing=drop_when_attribute_missing,
                )
                if saw_missing:
                    gate_stats["shapes_missing_gated_attribute"] += 1
                if dropped_by is not None:
                    gate_drops[dropped_by][label] += 1
                    continue

            boxes.append((label, bbox))
            label_counts[label] += 1
            total_label_counts[label] += 1

        if gates and not boxes and any(
            str(shape.get("label", "")).strip() in declared_class_set
            for shape in data.get("shapes", [])
        ):
            gate_stats["images_emptied"] += 1

        records.append(
            ImageRecord(
                stem=stem,
                image_path=image_path,
                xlabel_path=xlabel_path,
                group=infer_group(images_root, image_path),
                split_group=split_groups.get(stem, stem),
                detections=tuple(boxes),
                lines=tuple(
                    f"{declared_classes.index(label)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    for label, bbox in boxes
                ),
                label_counts=label_counts,
                shape_type_counts=shape_type_counts,
            )
        )

    gate_report = {
        "specs": [gate.raw for gate in gates],
        "drop_when_attribute_missing": bool(drop_when_attribute_missing),
        "dropped_total": sum(sum(counter.values()) for counter in gate_drops.values()),
        "dropped_by_spec": {
            raw: dict(sorted(counter.items())) for raw, counter in sorted(gate_drops.items())
        },
        **gate_stats,
    }
    return (
        records,
        sorted(set(unknown_labels)),
        total_label_counts,
        total_shape_type_counts,
        gate_report,
    )


def resolve_final_classes(
    declared_classes: list[str],
    records: list[ImageRecord],
    keep_unused_classes: bool,
) -> list[str]:
    observed_labels = {label for record in records for label in record.label_counts}
    if keep_unused_classes:
        final_classes = list(declared_classes)
    else:
        final_classes = [label for label in declared_classes if label in observed_labels]
    if not final_classes:
        raise ValueError("No labeled classes remain after filtering.")
    return final_classes


def remap_record_lines(records: list[ImageRecord], final_classes: list[str]) -> list[ImageRecord]:
    class_to_index = {label: idx for idx, label in enumerate(final_classes)}
    remapped_records: list[ImageRecord] = []
    for record in records:
        kept_label_counts: Counter = Counter()
        for label, count in record.label_counts.items():
            if label in class_to_index:
                kept_label_counts[label] = count

        kept_detections = tuple(
            (label, bbox)
            for label, bbox in record.detections
            if label in class_to_index
        )
        lines = tuple(
            f"{class_to_index[label]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            for label, bbox in kept_detections
        )

        remapped_records.append(
            ImageRecord(
                stem=record.stem,
                image_path=record.image_path,
                xlabel_path=record.xlabel_path,
                group=record.group,
                split_group=record.split_group,
                detections=kept_detections,
                lines=lines,
                label_counts=kept_label_counts,
                shape_type_counts=record.shape_type_counts,
            )
        )
    return remapped_records


def compute_val_counts(grouped_records: dict[str, list[ImageRecord]], val_ratio: float) -> dict[str, int]:
    target_total = round(sum(len(items) for items in grouped_records.values()) * val_ratio)
    ideal_counts: dict[str, float] = {
        group: len(items) * val_ratio for group, items in grouped_records.items()
    }
    val_counts: dict[str, int] = {group: math.floor(value) for group, value in ideal_counts.items()}
    assigned = sum(val_counts.values())
    remainder = max(0, target_total - assigned)
    ranked_groups = sorted(
        grouped_records,
        key=lambda group: (ideal_counts[group] - math.floor(ideal_counts[group]), group),
        reverse=True,
    )
    for group in ranked_groups[:remainder]:
        val_counts[group] += 1
    return val_counts


def split_records(
    records: list[ImageRecord],
    val_ratio: float,
    seed: int,
    *,
    keep_split_groups: bool = False,
) -> dict[str, list[ImageRecord]]:
    if keep_split_groups:
        return split_records_by_group(records, val_ratio=val_ratio, seed=seed)

    grouped_records: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.group].append(record)

    val_counts = compute_val_counts(grouped_records, val_ratio)
    split_map = {"train": [], "val": []}

    for group in sorted(grouped_records):
        ordered = sorted(
            grouped_records[group],
            key=lambda record: stable_key(seed=seed, group=group, stem=record.stem),
        )
        val_count = min(val_counts[group], len(ordered))
        split_map["val"].extend(ordered[:val_count])
        split_map["train"].extend(ordered[val_count:])

    return split_map


def split_records_by_group(
    records: list[ImageRecord],
    val_ratio: float,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    """Split indivisible related-image bundles without train/val leakage."""
    bundles: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        bundles[record.split_group].append(record)

    ordered_bundles = sorted(
        bundles.items(),
        key=lambda item: stable_key(
            seed=seed,
            group="split_group",
            stem=item[0],
        ),
    )
    target_val = round(len(records) * val_ratio)
    val_groups: set[str] = set()
    val_count = 0
    for split_group, bundle in ordered_bundles:
        bundle_size = len(bundle)
        current_error = abs(target_val - val_count)
        candidate_error = abs(target_val - (val_count + bundle_size))
        if candidate_error < current_error:
            val_groups.add(split_group)
            val_count += bundle_size

    if target_val > 0 and not val_groups and ordered_bundles:
        smallest_group, _ = min(
            ordered_bundles,
            key=lambda item: (len(item[1]), item[0]),
        )
        val_groups.add(smallest_group)

    split_map = {"train": [], "val": []}
    for record in records:
        split_name = "val" if record.split_group in val_groups else "train"
        split_map[split_name].append(record)
    return split_map


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def write_split(
    output_dir: Path,
    split_name: str,
    records: list[ImageRecord],
    image_mode: str,
) -> Counter:
    split_label_counts: Counter = Counter()
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        target_image = image_dir / record.image_path.name
        target_label = label_dir / f"{record.stem}.txt"
        materialize_image(src=record.image_path, dst=target_image, mode=image_mode)
        target_label.write_text("\n".join(record.lines) + ("\n" if record.lines else ""), encoding="utf-8")
        split_label_counts.update(record.label_counts)

    return split_label_counts


def write_data_yaml(output_dir: Path, final_classes: list[str]) -> Path:
    data_yaml = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(final_classes)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(final_classes))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def write_manifest(
    output_dir: Path,
    declared_classes: list[str],
    final_classes: list[str],
    split_map: dict[str, list[ImageRecord]],
    total_label_counts: Counter,
    split_label_counts: dict[str, Counter],
    total_shape_type_counts: Counter,
    ignored_xlabels: list[str],
    unknown_labels: list[str],
    split_groups_file: Path | None,
    split_group_key: str,
    gate_report: dict | None = None,
) -> Path:
    train_groups = {record.split_group for record in split_map["train"]}
    val_groups = {record.split_group for record in split_map["val"]}
    manifest = {
        "images_total": sum(len(items) for items in split_map.values()),
        "split_counts": {split: len(items) for split, items in split_map.items()},
        "declared_classes": declared_classes,
        "used_classes": final_classes,
        "dropped_classes": [label for label in declared_classes if label not in final_classes],
        "label_counts_total": dict(sorted(total_label_counts.items())),
        "label_counts_by_split": {
            split: dict(sorted(counter.items())) for split, counter in split_label_counts.items()
        },
        "shape_type_counts_total": dict(sorted(total_shape_type_counts.items())),
        "images_by_group": {
            group: {
                split: sum(1 for record in split_map[split] if record.group == group)
                for split in ("train", "val")
            }
            for group in sorted({record.group for records in split_map.values() for record in records})
        },
        "split_grouping": {
            "enabled": split_groups_file is not None,
            "source": str(split_groups_file.resolve()) if split_groups_file else None,
            "key": split_group_key if split_groups_file else None,
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "leaked_groups": sorted(train_groups & val_groups),
            "ok": not bool(train_groups & val_groups),
        },
        "ignored_xlabels": ignored_xlabels,
        "unknown_labels_ignored": unknown_labels,
        # The gate is a dataset-defining decision, not a run detail: `label_counts_*`
        # above are POST-gate, so the section 6.2 box gates are read off numbers that
        # only these specs explain. Recording them here is what keeps the exporter-side
        # floor and this prep-side floor from silently drifting apart.
        "attribute_gates": gate_report or {"specs": [], "dropped_total": 0},
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()

    declared_classes = read_lines(args.classes_file)
    gates = [parse_attribute_gate(spec) for spec in args.drop_if]
    unknown_gate_labels = sorted(
        {
            label
            for gate in gates
            if gate.labels
            for label in gate.labels
            if label not in set(declared_classes)
        }
    )
    if unknown_gate_labels:
        raise ValueError(
            "--drop-if names classes absent from --classes-file: "
            + ", ".join(unknown_gate_labels)
        )
    image_paths = discover_images(args.images_root)
    xlabel_paths = discover_xlabels(args.xlabels_dir)
    split_groups = load_split_groups(args.split_groups_file, args.split_group_key)

    ignored_xlabels = sorted(
        str(path.name)
        for stem, path in xlabel_paths.items()
        if stem not in image_paths
    )

    (
        records,
        unknown_labels,
        total_label_counts,
        total_shape_type_counts,
        gate_report,
    ) = build_records(
        images_root=args.images_root,
        image_paths=image_paths,
        xlabel_paths=xlabel_paths,
        declared_classes=declared_classes,
        allow_missing_labels=args.allow_missing_labels,
        split_groups=split_groups,
        gates=gates,
        drop_when_attribute_missing=args.drop_if_attribute_missing,
    )
    final_classes = resolve_final_classes(
        declared_classes=declared_classes,
        records=records,
        keep_unused_classes=args.keep_unused_classes,
    )
    records = remap_record_lines(records=records, final_classes=final_classes)
    split_map = split_records(
        records=records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        keep_split_groups=args.split_groups_file is not None,
    )

    ensure_clean_dir(args.output_dir / "images")
    ensure_clean_dir(args.output_dir / "labels")

    split_label_counts = {
        split: write_split(
            output_dir=args.output_dir,
            split_name=split,
            records=sorted(items, key=lambda record: record.stem),
            image_mode=args.image_mode,
        )
        for split, items in split_map.items()
    }

    data_yaml = write_data_yaml(output_dir=args.output_dir, final_classes=final_classes)
    manifest_path = write_manifest(
        output_dir=args.output_dir,
        declared_classes=declared_classes,
        final_classes=final_classes,
        split_map=split_map,
        total_label_counts=total_label_counts,
        split_label_counts=split_label_counts,
        total_shape_type_counts=total_shape_type_counts,
        ignored_xlabels=ignored_xlabels,
        unknown_labels=unknown_labels,
        split_groups_file=args.split_groups_file,
        split_group_key=args.split_group_key,
        gate_report=gate_report,
    )
    classes_path = args.output_dir / "classes.txt"
    classes_path.write_text("\n".join(final_classes) + "\n", encoding="utf-8")

    print(f"Prepared YOLO dataset at: {args.output_dir.resolve()}")
    print(f"data.yaml: {data_yaml.resolve()}")
    print(f"classes.txt: {classes_path.resolve()}")
    print(f"dataset_manifest.json: {manifest_path.resolve()}")
    print(f"train images: {len(split_map['train'])}")
    print(f"val images: {len(split_map['val'])}")
    print(f"used classes ({len(final_classes)}): {', '.join(final_classes)}")
    dropped = [label for label in declared_classes if label not in final_classes]
    print(f"dropped unused classes ({len(dropped)}): {', '.join(dropped) if dropped else '(none)'}")
    if ignored_xlabels:
        print(f"ignored JSON files without matching images ({len(ignored_xlabels)}): {', '.join(ignored_xlabels)}")
    if unknown_labels:
        print(f"ignored unknown labels ({len(unknown_labels)} unique): {', '.join(unknown_labels)}")
    if gates:
        print(
            f"attribute gates ({len(gates)}): {', '.join(gate.raw for gate in gates)}"
            f" [missing attribute -> {'drop' if args.drop_if_attribute_missing else 'keep'}]"
        )
        print(f"boxes dropped by gates: {gate_report['dropped_total']}")
        for raw, counts in gate_report["dropped_by_spec"].items():
            per_class = ", ".join(f"{label}={n}" for label, n in counts.items())
            print(f"  {raw}: {sum(counts.values())} ({per_class})")
        print(
            f"shapes with no gated attribute: {gate_report['shapes_missing_gated_attribute']}"
            f"; images left with zero boxes: {gate_report['images_emptied']}"
        )


if __name__ == "__main__":
    main()
