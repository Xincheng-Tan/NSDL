from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_yolo26_finetune_dataset.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_class_weighted_loss import (  # noqa: E402
    DEFAULT_CLIP,
    WEIGHT_MODES,
    compute_class_weights,
    install_class_weighted_loss,
)

# Off by default: the weighting changes the objective, so an existing run command
# must keep producing the same run unless the caller opts in.
DEFAULT_CLASS_WEIGHT_MODE = "none"

DEFAULT_MODEL_PATH = REPO_ROOT / "checkpoints" / "yolo26l.pt"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "outputs" / "yolo_pilot" / "VIKI-L2-train-pilot"
DEFAULT_XLABELS_DIR = REPO_ROOT / "outputs" / "yolo_pilot" / "VIKI-L2-train-pilot_xlabels_full"
DEFAULT_CLASSES_FILE = REPO_ROOT / "outputs" / "yolo_pilot" / "VIKI-L2-train-pilot" / "full_class_list.txt"
DEFAULT_DATASET_DIR = REPO_ROOT / "outputs" / "yolo_pilot" / "yolo26_viki_l2_pilot_full20_det"
DEFAULT_DATA_YAML = DEFAULT_DATASET_DIR / "data.yaml"
DEFAULT_PROJECT_DIR = REPO_ROOT / "outputs" / "yolo_runs"
DEFAULT_RUN_NAME = "yolo26l_viki_l2_pilot_full20_latest"
DEFAULT_COPY_BEST_TO = REPO_ROOT / "checkpoints" / "best.pt"

DEFAULT_DATASET_VAL_RATIO = 0.10
DEFAULT_DATASET_IMAGE_MODE = "symlink"
DEFAULT_DATASET_KEEP_UNUSED_CLASSES = False
DEFAULT_DATASET_ALLOW_MISSING_LABELS = False
DEFAULT_REQUIRED_LABELS = ("cabinet", "table")
DEFAULT_SEED = 3407


def _read_data_yaml_names(data_yaml_path: Path) -> list[str]:
    names: list[str] = []
    inside_names = False
    for line in data_yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            inside_names = True
            continue
        if inside_names:
            if not line.startswith(" "):
                break
            _, _, value = stripped.partition(":")
            name = value.strip()
            if name:
                names.append(name)
    return names


def _validate_dataset(
    data_yaml_path: Path,
    required_labels: tuple[str, ...],
    *,
    require_split_grouping: bool = False,
    split_group_key: str | None = None,
) -> tuple[list[str], dict[str, int]]:
    class_names = _read_data_yaml_names(data_yaml_path)
    if not class_names:
        raise RuntimeError(f"Failed to parse class names from data yaml: {data_yaml_path}")

    missing_from_names = [label for label in required_labels if label not in class_names]
    if missing_from_names:
        raise RuntimeError(
            "Dataset is missing required labels in data.yaml: "
            + ", ".join(missing_from_names)
            + f". data_yaml={data_yaml_path}"
        )

    manifest_path = data_yaml_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        if require_split_grouping:
            raise RuntimeError(
                "dataset_manifest.json is required when --split-groups-file is used: "
                f"{manifest_path}"
            )
        print(
            "warning: dataset_manifest.json not found; "
            "validated required labels from data.yaml only."
        )
        return class_names, {}, {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_counts = {
        str(label): int(count)
        for label, count in (manifest.get("label_counts_total") or {}).items()
    }

    missing_from_counts = [label for label in required_labels if label_counts.get(label, 0) <= 0]
    if missing_from_counts:
        raise RuntimeError(
            "Dataset manifest shows zero boxes for required labels: "
            + ", ".join(missing_from_counts)
            + f". manifest={manifest_path}"
        )

    train_label_counts = {
        str(label): int(count)
        for label, count in (
            (manifest.get("label_counts_by_split") or {}).get("train") or {}
        ).items()
    }
    missing_from_train = [
        label for label in required_labels if train_label_counts.get(label, 0) <= 0
    ]
    if missing_from_train:
        raise RuntimeError(
            "Dataset manifest shows zero train boxes for required labels: "
            + ", ".join(missing_from_train)
            + f". manifest={manifest_path}"
        )

    if require_split_grouping:
        split_grouping = manifest.get("split_grouping") or {}
        leaked_groups = split_grouping.get("leaked_groups") or []
        if (
            split_grouping.get("enabled") is not True
            or split_grouping.get("ok") is not True
            or leaked_groups
        ):
            raise RuntimeError(
                "Dataset split-group audit failed: "
                f"split_grouping={split_grouping!r}, manifest={manifest_path}"
            )
        if split_group_key and split_grouping.get("key") != split_group_key:
            raise RuntimeError(
                "Dataset split-group key does not match the requested key: "
                f"expected={split_group_key!r}, "
                f"actual={split_grouping.get('key')!r}, manifest={manifest_path}"
            )

    return class_names, label_counts, train_label_counts


def _prepare_dataset(
    *,
    images_root: Path,
    xlabels_dir: Path,
    classes_file: Path,
    data_yaml_path: Path,
    val_ratio: float,
    seed: int,
    image_mode: str,
    keep_unused_classes: bool,
    allow_missing_labels: bool,
    split_groups_file: Path | None,
    split_group_key: str,
    drop_if: list[str],
    drop_if_attribute_missing: bool,
) -> None:
    if data_yaml_path.name != "data.yaml":
        raise ValueError(
            "--data must point to a data.yaml file when --prepare-data is enabled. "
            f"Received: {data_yaml_path}"
        )

    command = [
        sys.executable,
        str(PREPARE_SCRIPT_PATH),
        "--images-root",
        str(images_root),
        "--xlabels-dir",
        str(xlabels_dir),
        "--classes-file",
        str(classes_file),
        "--output-dir",
        str(data_yaml_path.parent),
        "--val-ratio",
        str(val_ratio),
        "--seed",
        str(seed),
        "--image-mode",
        image_mode,
    ]
    if keep_unused_classes:
        command.append("--keep-unused-classes")
    if allow_missing_labels:
        command.append("--allow-missing-labels")
    for spec in drop_if:
        command.extend(["--drop-if", spec])
    if drop_if_attribute_missing:
        command.append("--drop-if-attribute-missing")
    if split_groups_file is not None:
        command.extend(
            [
                "--split-groups-file",
                str(split_groups_file),
                "--split-group-key",
                split_group_key,
            ]
        )

    print("Preparing YOLO dataset with command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def _ensure_training_inputs_exist(
    *,
    model_path: Path,
    images_root: Path,
    xlabels_dir: Path,
    classes_file: Path,
    split_groups_file: Path | None,
    prepare_data: bool,
) -> None:
    if not model_path.is_file():
        raise SystemExit(f"Model weights not found: {model_path}")
    if not prepare_data:
        return
    if not images_root.is_dir():
        raise SystemExit(f"Images root not found: {images_root}")
    if not xlabels_dir.is_dir():
        raise SystemExit(f"X-AnyLabeling directory not found: {xlabels_dir}")
    if not classes_file.is_file():
        raise SystemExit(f"Classes file not found: {classes_file}")
    if split_groups_file is not None and not split_groups_file.is_file():
        raise SystemExit(f"Split-groups manifest not found: {split_groups_file}")


def _print_dataset_summary(
    data_yaml_path: Path,
    class_names: list[str],
    label_counts: dict[str, int],
    required_labels: tuple[str, ...],
) -> None:
    print(f"Using dataset yaml: {data_yaml_path}")
    print(f"Using {len(class_names)} classes: {', '.join(class_names)}")
    if label_counts and required_labels:
        required_summary = ", ".join(
            f"{label}={label_counts.get(label, 0)}" for label in required_labels
        )
        print(f"Required label counts: {required_summary}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the NSDL YOLO26 detector and copy the best checkpoint "
            "to --copy-best-to."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument(
        "--prepare-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild the YOLO dataset from X-AnyLabeling annotations before training.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="Root directory containing the raw training images.",
    )
    parser.add_argument(
        "--xlabels-dir",
        type=Path,
        default=DEFAULT_XLABELS_DIR,
        help="Directory containing X-AnyLabeling JSON annotations.",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=DEFAULT_CLASSES_FILE,
        help="Ordered class list passed to prepare_yolo26_finetune_dataset.py.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_DATASET_VAL_RATIO,
        help="Validation split ratio used when rebuilding the dataset.",
    )
    parser.add_argument(
        "--prepare-image-mode",
        choices=("symlink", "hardlink", "copy"),
        default=DEFAULT_DATASET_IMAGE_MODE,
        help="How to materialize images when rebuilding the dataset.",
    )
    parser.add_argument(
        "--keep-unused-classes",
        action="store_true",
        default=DEFAULT_DATASET_KEEP_UNUSED_CLASSES,
        help="Keep zero-count classes from --classes-file during dataset preparation.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        default=DEFAULT_DATASET_ALLOW_MISSING_LABELS,
        help="Allow images without matching JSON files during dataset preparation.",
    )
    parser.add_argument(
        "--split-groups-file",
        type=Path,
        default=None,
        help=(
            "Optional render JSON/JSONL manifest used to keep counterfactual/base-scene "
            "groups in one detector split."
        ),
    )
    parser.add_argument(
        "--split-group-key",
        default="counterfactual_pair_id",
        help="Manifest field used as the indivisible detector split group.",
    )
    parser.add_argument(
        "--drop-if",
        action="append",
        metavar="[CLASSES:]ATTR<OP>VALUE",
        help=(
            "Forwarded verbatim to prepare_yolo26_finetune_dataset.py, repeatable. "
            "Must be passed here rather than pre-preparing the dataset separately, "
            "because --class-weight-mode computes weights from the prepared "
            "manifest's TRAIN counts -- gating afterwards would weight classes by "
            "boxes that never reach the loss. The VIKI reconstruction source "
            "(docs/YOLO.md §6.1/§6.7) requires 'robot_bbox_overlap>=0.5'; without "
            "it, boxes sitting behind the annotator's own robots train as positives."
        ),
    )
    parser.add_argument(
        "--drop-if-attribute-missing",
        action="store_true",
        help=(
            "Forwarded to the preparer: treat a missing attribute as a drop. Off by "
            "default because human shapes carry no exporter attributes at all, so "
            "turning it on deletes every hand-drawn box."
        ),
    )
    parser.add_argument(
        "--required-labels",
        nargs="*",
        default=list(DEFAULT_REQUIRED_LABELS),
        help=(
            "Labels that must be present in data.yaml and have positive counts in "
            "dataset_manifest.json before training continues."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", default="ram")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--copy-best-to", type=Path, default=DEFAULT_COPY_BEST_TO)
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether Ultralytics may reuse an existing run directory.",
    )
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cosine LR schedule.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=WEIGHT_MODES,
        default=DEFAULT_CLASS_WEIGHT_MODE,
        help=(
            "Per-class classification-loss weighting, computed from the TRAIN box "
            "counts of the prepared dataset. The LOCUS pilot is 78%% drawer+cabinet "
            "by box count for structural reasons (both are fixtures present in "
            "every image at ~15 / ~11 instances, every object class at ~1), which "
            "aggregate mAP hides. 'none' disables the hook."
        ),
    )
    parser.add_argument(
        "--class-weight",
        action="append",
        default=None,
        metavar="CLASS=W",
        help=(
            "Manual per-class weight override, repeatable. Wins over the mode's "
            "computed value for that class."
        ),
    )
    parser.add_argument(
        "--class-weight-clip",
        type=float,
        nargs=2,
        default=list(DEFAULT_CLIP),
        metavar=("LO", "HI"),
        help=(
            "Bounds the weight RATIO (HI/LO). Raw inverse frequency spans ~700x "
            "between drawer and wine, which would destabilise the head."
        ),
    )
    return parser


def _parse_manual_class_weights(entries: list[str] | None) -> dict[str, float]:
    manual: dict[str, float] = {}
    for entry in entries or []:
        name, sep, value = str(entry).partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"--class-weight expects CLASS=W, got {entry!r}")
        try:
            manual[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--class-weight value is not a number: {entry!r}") from None
    return manual


def main() -> None:
    args = build_parser().parse_args()

    model_path = args.model.expanduser().resolve()
    data_yaml_path = args.data.expanduser().resolve()
    project_dir = args.project.expanduser().resolve()
    copy_best_to = args.copy_best_to.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    xlabels_dir = args.xlabels_dir.expanduser().resolve()
    classes_file = args.classes_file.expanduser().resolve()
    split_groups_file = (
        args.split_groups_file.expanduser().resolve()
        if args.split_groups_file is not None
        else None
    )
    required_labels = tuple(
        dict.fromkeys(
            str(label).strip()
            for label in args.required_labels
            if str(label).strip()
        )
    )

    _ensure_training_inputs_exist(
        model_path=model_path,
        images_root=images_root,
        xlabels_dir=xlabels_dir,
        classes_file=classes_file,
        split_groups_file=split_groups_file,
        prepare_data=args.prepare_data,
    )

    if args.prepare_data:
        _prepare_dataset(
            images_root=images_root,
            xlabels_dir=xlabels_dir,
            classes_file=classes_file,
            data_yaml_path=data_yaml_path,
            val_ratio=args.val_ratio,
            seed=args.seed,
            image_mode=args.prepare_image_mode,
            keep_unused_classes=args.keep_unused_classes,
            allow_missing_labels=args.allow_missing_labels,
            split_groups_file=split_groups_file,
            split_group_key=args.split_group_key,
            drop_if=list(args.drop_if or []),
            drop_if_attribute_missing=args.drop_if_attribute_missing,
        )

    if not data_yaml_path.is_file():
        raise SystemExit(f"Data yaml not found: {data_yaml_path}")

    class_names, label_counts, train_label_counts = _validate_dataset(
        data_yaml_path=data_yaml_path,
        required_labels=required_labels,
        require_split_grouping=split_groups_file is not None,
        split_group_key=args.split_group_key if split_groups_file is not None else None,
    )
    _print_dataset_summary(
        data_yaml_path=data_yaml_path,
        class_names=class_names,
        label_counts=label_counts,
        required_labels=required_labels,
    )

    # Weighting is derived from the label counts _validate_dataset just read off
    # the prepared dataset, so the weights always match the data actually being
    # trained on (no hand-maintained table to drift).
    #
    # TRAIN-ONLY counts on purpose: compute_class_weights documents its argument as
    # the per-class train box count, and a training hyperparameter must not be a
    # function of the val split. Feeding `label_counts` (= label_counts_total,
    # train+val) leaked val statistics into the weight table; with val_ratio=0.10 it
    # also inflated every count by ~1/0.9, which the frequency-weighted
    # renormalisation mostly absorbs but does not cancel for the clipped tail.
    weight_counts = train_label_counts or label_counts
    class_weights = None
    if args.class_weight_mode != "none" or args.class_weight:
        class_weights = compute_class_weights(
            class_names,
            weight_counts,
            mode=args.class_weight_mode,
            manual=_parse_manual_class_weights(args.class_weight),
            clip=(float(args.class_weight_clip[0]), float(args.class_weight_clip[1])),
        )
        print(f"class_weight_mode: {args.class_weight_mode}")
        for name, weight in sorted(zip(class_names, class_weights),
                                   key=lambda item: item[1]):
            print(f"  {name:24s} train_boxes={weight_counts.get(name, 0):>6d} "
                  f"weight={weight:.3f}")

    from ultralytics import YOLO

    if class_weights is not None:
        # Must happen before YOLO() builds the model: BaseModel.loss creates the
        # criterion lazily from DetectionModel.init_criterion, which this patches.
        installed = install_class_weighted_loss(class_weights)
        print(f"installed_loss: {installed['loss_fn']}")

    detector = YOLO(str(model_path))
    detector.train(
        data=str(data_yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        seed=args.seed,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    trainer = getattr(detector, "trainer", None)
    if trainer is None or getattr(trainer, "save_dir", None) is None:
        raise RuntimeError("Ultralytics trainer did not expose save_dir after training.")

    save_dir = Path(trainer.save_dir).resolve()
    best_weights_path = save_dir / "weights" / "best.pt"
    if not best_weights_path.is_file():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_weights_path}")

    if class_weights is not None:
        # The objective is not recoverable from best.pt (the criterion is not
        # serialised), so a run trained with class weights must record them next
        # to the weights or its numbers cannot be reproduced or compared.
        (save_dir / "class_weights.json").write_text(json.dumps(
            {"mode": args.class_weight_mode,
             "clip": [float(args.class_weight_clip[0]),
                      float(args.class_weight_clip[1])],
             "manual": _parse_manual_class_weights(args.class_weight),
             "weights": dict(zip(class_names, class_weights)),
             # The counts the weights were actually derived from (train split only),
             # plus train+val totals for auditing. These were the same key before,
             # which made a train-only weight table indistinguishable from a
             # train+val one after the fact.
             "train_box_counts": {n: weight_counts.get(n, 0) for n in class_names},
             "total_box_counts": {n: label_counts.get(n, 0) for n in class_names}},
            indent=2), encoding="utf-8")
        print(f"class_weights: {save_dir / 'class_weights.json'}")

    copy_best_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights_path, copy_best_to)

    print(f"save_dir: {save_dir}")
    print(f"best_weights: {best_weights_path}")
    print(f"copied_best_to: {copy_best_to}")


if __name__ == "__main__":
    main()
