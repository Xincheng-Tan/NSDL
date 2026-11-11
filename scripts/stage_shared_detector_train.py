"""Stage the shared full34 detector's training inputs (docs/YOLO.md §6.1).

§6.1 forbids handing a VIKI or LOCUS root to a recursive scan: the two benchmarks
share a detector, so anything that leaks a `val`/`test_id`/`ood_*` frame into the
training pool contaminates every downstream comparison at once. The defence is an
explicit staging step -- this script -- that copies exactly the two permitted
sources into one tree and writes down what it copied.

Sources (both are *train*-only by construction):

* VIKI-L2 train, 900 frames. Labels come from `outputs/viki_recon_labels/`, NOT
  the human `_xlabels_full`: `cabinet`/`drawer`/`fridge`/`oven` were rebuilt to
  the LOCUS caliber in §6.7. Downstream must still carry
  `--drop-if 'robot_bbox_overlap>=0.5'`.
* LOCUS pilot train, 600 frames, renderer GT after the glass-cabinet human review
  (`data/lnsdl_bench/pilot/train/gt_labels/`, post `merge_locus_glass_review.py`).

Five ways a staging step can look successful and be wrong, each checked here and
recorded in the manifest rather than assumed:

1. **Stem collision.** Images are paired to labels by stem, so one shared stem
   silently gives a frame the other source's boxes. VIKI stems carry a hash
   suffix and LOCUS stems are five digits, so today there is no collision -- but
   LOCUS stems are only unique *within* a split, so this stays a hard failure.
2. **Image with no label.** YOLO reads an unlabelled frame as all-background, so
   a missed pairing is reverse supervision over the whole image, not a no-op.
3. **Dimension drift.** The label JSON's `imageWidth/imageHeight` is what the
   box coordinates are expressed in. If it disagrees with the actual PNG every
   box is silently rescaled; nothing downstream re-checks this.
4. **Out-of-vocabulary class names.** A name outside full34 is dropped by the
   dataset preparer with a count, which is easy to skim past; here it fails.
5. **Source drift after staging.** Images are hard-linked by default (1.6 GB, one
   filesystem), so an in-place edit of a source would silently change the staged
   tree. The SHA256SUMS sidecars turn that into a detectable `sha256sum -c`
   failure instead of a silent one.

Run:

    python scripts/stage_shared_detector_train.py --clean

Verify later (standard tool, no script needed):

    cd outputs/yolo_locus && sha256sum -c shared_detector_train/SHA256SUMS.viki_train.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# docs/YOLO.md §4.1. The behavioural source of truth is the file named by
# --classes-file (configs/yolo/locus_full34_class_list.txt); this tuple exists so
# `test_the_canonical_class_file_still_matches_docs_section_4_1` can catch that
# file drifting away from the doc. Neither is
# outputs/yolo_pilot/.../full_training_class_list.txt -- that is a DIFFERENT
# 35-entry list (has cooling_fan/green_peas, lacks table) and is not full34.
FULL34 = (
    "fetch", "stompy", "unitree_h1", "panda", "unitree_go2", "anymal_c",
    "apple", "banana", "bowl", "bread", "cabinet", "knife", "peach", "pear",
    "plate", "sink", "table", "toaster", "tomato", "wooden_cutting_board",
    "bottle", "cardboardbox", "cup", "drawer", "fork", "fridge", "kettle",
    "meat", "oven", "pizza", "pumpkin", "scissors", "spoon", "wine",
)

CANONICAL_CLASSES_FILE = REPO_ROOT / "configs/yolo/locus_full34_class_list.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULTS = {
    "viki_images": "outputs/yolo_pilot/VIKI-L2-train-pilot",
    "viki_labels": "outputs/viki_recon_labels",
    "locus_images": "data/lnsdl_bench/pilot/train/images",
    "locus_labels": "data/lnsdl_bench/pilot/train/gt_labels",
    "out_images": "outputs/yolo_locus/shared_detector_train",
    "out_labels": "outputs/yolo_locus/xlabels_reviewed_full34",
}


class StagingError(RuntimeError):
    """A check that must stop the run rather than be recorded and skipped."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(root: Path) -> dict[str, Path]:
    """Recursive, because VIKI nests its 900 frames under layout_0..9/."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in found:
            raise StagingError(
                f"duplicate image stem {path.stem!r} inside one source: "
                f"{found[path.stem]} and {path}"
            )
        found[path.stem] = path
    if not found:
        raise StagingError(f"no images under {root}")
    return found


def discover_labels(root: Path) -> tuple[dict[str, Path], list[str]]:
    """Split label JSONs from the bookkeeping JSONs that live beside them.

    `outputs/viki_recon_labels/reconstruction_manifest.json` and the merge
    summaries are detected structurally (no `shapes` key) rather than by an
    allow-list of names, so a new sidecar does not have to be anticipated.
    """
    labels: dict[str, Path] = {}
    bookkeeping: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StagingError(f"{path} is not readable JSON: {exc}") from exc
        if not isinstance(payload, dict) or "shapes" not in payload:
            bookkeeping.append(path.name)
            continue
        labels[path.stem] = path
    if not labels:
        raise StagingError(f"no label JSONs under {root}")
    return labels, bookkeeping


def inspect_source(name: str, images_root: Path, labels_root: Path,
                   vocab: tuple[str, ...]) -> dict:
    """Pair one source and run checks 2-4. Returns the pairing plus its stats."""
    images = discover_images(images_root)
    labels, bookkeeping = discover_labels(labels_root)

    unlabelled = sorted(set(images) - set(labels))
    if unlabelled:
        raise StagingError(
            f"[{name}] {len(unlabelled)} image(s) have no label JSON, which YOLO "
            f"would train as all-background: {unlabelled[:5]}"
        )
    orphans = sorted(set(labels) - set(images))
    if orphans:
        raise StagingError(
            f"[{name}] {len(orphans)} label JSON(s) have no image and are not "
            f"bookkeeping files: {orphans[:5]}"
        )

    classes: Counter[str] = Counter()
    dims: Counter[str] = Counter()
    for stem in sorted(images):
        payload = json.loads(labels[stem].read_text(encoding="utf-8"))
        with Image.open(images[stem]) as handle:
            actual = handle.size  # header-only read
        declared = (int(payload.get("imageWidth", -1)), int(payload.get("imageHeight", -1)))
        if declared != actual:
            raise StagingError(
                f"[{name}] {stem}: label says {declared} but the image is "
                f"{actual}; every box in this file is scaled wrong"
            )
        dims[f"{actual[0]}x{actual[1]}"] += 1
        for shape in payload.get("shapes", []):
            classes[str(shape.get("label"))] += 1

    unknown = sorted(set(classes) - set(vocab))
    if unknown:
        raise StagingError(
            f"[{name}] label names outside full34 (§4.1): "
            f"{ {k: classes[k] for k in unknown} }"
        )

    return {
        "name": name,
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "images": images,
        "labels": labels,
        "stats": {
            "n_images": len(images),
            "n_labels": len(labels),
            "n_boxes": sum(classes.values()),
            "image_sizes": dict(dims),
            "bookkeeping_json_ignored": bookkeeping,
            "class_counts": dict(sorted(classes.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
    }


def place(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-device or unsupported: fall through to a real copy
    if link_mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def require_writable(path: Path, clean: bool) -> None:
    if not path.exists():
        return
    entries = [p for p in path.iterdir() if not p.name.startswith(".")]
    if not entries:
        return
    if not clean:
        raise StagingError(
            f"{path} already holds {len(entries)} entrie(s). Re-staging into a "
            f"dirty tree would leave files from a previous source mixed in; "
            f"pass --clean to wipe it first."
        )
    shutil.rmtree(path)


def write_checksums(path: Path, root: Path, files: list[Path]) -> dict:
    """`sha256sum -c`-compatible sidecar, paths relative to `root`."""
    lines = []
    digests = []
    total_bytes = 0
    for staged in files:
        digest = sha256_of(staged)
        digests.append(digest)
        total_bytes += staged.stat().st_size
        lines.append(f"{digest}  {os.path.relpath(staged, root)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # One number that changes if any member changed, so drift is a single diff.
    rollup = hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()
    return {
        "file": os.path.relpath(path, root),
        "n_files": len(files),
        "total_bytes": total_bytes,
        "rollup_sha256": rollup,
        "self_sha256": sha256_of(path),
    }


def load_vocab(path: Path) -> tuple[str, ...]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    duplicates = [n for n, c in Counter(names).items() if c > 1]
    if duplicates:
        raise StagingError(f"{path} lists {duplicates} more than once")
    if len(names) != 34:
        raise StagingError(f"{path} has {len(names)} classes, not 34 (§4.1)")
    return tuple(names)


def run(args: argparse.Namespace) -> dict:
    classes_file = Path(args.classes_file)
    vocab = load_vocab(classes_file)
    sources = [
        inspect_source("viki_train", Path(args.viki_images), Path(args.viki_labels), vocab),
        inspect_source("locus_train", Path(args.locus_images), Path(args.locus_labels), vocab),
    ]

    a, b = (set(s["images"]) for s in sources)
    collisions = sorted(a & b)
    if collisions:
        raise StagingError(
            f"{len(collisions)} stem(s) exist in both sources, so staging would "
            f"pair a frame with the other benchmark's boxes: {collisions[:5]}"
        )

    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)
    require_writable(out_images, args.clean)
    require_writable(out_labels, args.clean)

    checksum_root = Path(os.path.commonpath([out_images.resolve(), out_labels.resolve()]))

    manifest_sources = []
    for source in sources:
        subdir = out_images / source["name"]
        staged: list[Path] = []
        for stem, image in sorted(source["images"].items()):
            image_dst = subdir / f"{stem}{image.suffix}"
            label_dst = out_labels / f"{stem}.json"
            place(image, image_dst, args.link_mode)
            place(source["labels"][stem], label_dst, args.link_mode)
            staged.extend([image_dst, label_dst])
        sums = write_checksums(
            out_images / f"SHA256SUMS.{source['name']}.txt", checksum_root, staged
        )
        manifest_sources.append({
            **{k: v for k, v in source.items() if k not in {"images", "labels", "stats"}},
            **source["stats"],
            "staged_image_dir": os.path.relpath(subdir, checksum_root),
            "checksums": sums,
        })

    union: Counter[str] = Counter()
    for entry in manifest_sources:
        union.update(entry["class_counts"])
    missing = [name for name in vocab if name not in union]

    manifest = {
        "schema": "shared_detector_train_staging/1",
        "spec": "docs/YOLO.md §6.1",
        "link_mode": args.link_mode,
        "checksum_root": str(checksum_root),
        "shared_detector_train_image_dir": str(out_images),
        "xlabels_reviewed_full34_dir": str(out_labels),
        "classes_file": str(classes_file),
        "classes_file_sha256": sha256_of(classes_file),
        "n_images": sum(e["n_images"] for e in manifest_sources),
        "n_boxes": sum(e["n_boxes"] for e in manifest_sources),
        "sources": manifest_sources,
        "full34_class_counts": {name: union.get(name, 0) for name in vocab},
        "full34_classes_with_zero_boxes": missing,
        "checks": {
            "stem_collisions_across_sources": 0,
            "images_without_labels": 0,
            "label_image_dimension_mismatches": 0,
            "labels_outside_full34": 0,
        },
        "downstream_required_flags": ["--drop-if", "robot_bbox_overlap>=0.5"],
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n",
                             encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=str(REPO_ROOT / value))
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "outputs/yolo_locus/shared_detector_train"
                    / "detector_data_manifest.json"),
    )
    parser.add_argument(
        "--classes-file", default=str(CANONICAL_CLASSES_FILE),
        help="ordered full34 list (§4.1); the same file must be passed to "
             "prepare_yolo26_finetune_dataset.py so class ids agree",
    )
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"],
                        default="hardlink",
                        help="hardlink (default) costs no disk and keeps content "
                             "identical to the source; SHA256SUMS makes any later "
                             "in-place source edit detectable.")
    parser.add_argument("--clean", action="store_true",
                        help="wipe the output dirs first; required if they are non-empty")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except StagingError as exc:
        print(f"staging refused: {exc}", file=sys.stderr)
        return 2
    print(f"staged {manifest['n_images']} images / {manifest['n_boxes']} boxes "
          f"({args.link_mode})")
    for entry in manifest["sources"]:
        print(f"  {entry['name']:12s} {entry['n_images']:4d} images "
              f"{entry['n_boxes']:6d} boxes  sizes={entry['image_sizes']}")
        if entry["bookkeeping_json_ignored"]:
            print(f"               ignored non-label JSON: "
                  f"{entry['bookkeeping_json_ignored']}")
    if manifest["full34_classes_with_zero_boxes"]:
        print(f"  full34 classes with 0 boxes: "
              f"{manifest['full34_classes_with_zero_boxes']}")
    print(f"  manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
