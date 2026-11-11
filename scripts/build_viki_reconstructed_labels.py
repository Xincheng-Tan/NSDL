#!/usr/bin/env python
"""Build reconstructed full34 GT for the VIKI-900 training images.

Context: docs/YOLO.md §6.5 (the caliber divergence) and §6.6 (the probe that
proved this is possible). `locus_viki_reconstruct_probe.py` established that a
VIKI image's fixture geometry is recoverable by matching it against a rendered
reference for its (layout, style, camera) -- camera recovery is decisive
(margin min 0.420 over all 900) and fixture boxes are nearly style-invariant.
This script consumes that match and emits merged label files.

WHAT COMES FROM WHERE, AND WHY IT IS NOT A BLANKET REPLACEMENT
--------------------------------------------------------------
Only four classes are taken from the reconstruction:

    cabinet  drawer  fridge  oven

Those are the four with a real gap. `cabinet` is the caliber conflict (the
annotator deliberately labelled only the upper transparent cabinets, ~2.1/img,
because in VIKI the fetchable fruit only ever appears there; LOCUS labels every
link group, 11.7/img). `drawer`/`fridge`/`oven` are layout fixtures that are on
screen in every image and in nobody's label file -- ~11 200 / 900 / 290 boxes of
pure reverse supervision, since YOLO reads an unlabelled fixture as background.

`sink`, `table` and `toaster` are deliberately KEPT FROM THE HUMAN, even though
the reconstruction can produce them. They are not a caliber conflict -- one
visual unit is one link group -- and the measurement says the human is the more
correct source where the two differ:

    class    ref boxes  human  images ref-only  mean robot-overlap of those
    table          384    384                0   --      (exact agreement)
    sink           900    891                9   0.784
    toaster        816    656              168   0.710

The reference render contains the *reference's* agent, not the target image's
robots (robot placement goes through the generator's unseeded `random`, so it is
not recoverable). Its boxes are therefore amodal with respect to the robots that
are actually in the VIKI frame. The table above shows the human's "missing"
sinks and toasters are overwhelmingly the ones standing behind a robot -- the
annotator was right and the reconstruction would be wrong. Replacing them would
inject exactly that error. For the four classes we do take, the same overlap on
the disagreeing images is low (0.088-0.252), i.e. they are genuinely visible and
genuinely unlabelled.

`table` is a further independent check rather than a decision: the reference
emits it on layouts 1/3/6/9 only -- exactly `locus_task_pool.TABLE_LAYOUTS` --
and 0 on the G-shaped layouts 7/8, reproducing the caliber that
`strip_viki_breakfast_bar_table.py` had to enforce on the human side by hand.

THE ROBOT-OCCLUSION RESIDUAL IS ANNOTATED, NOT SILENTLY DROPPED
---------------------------------------------------------------
Each reconstructed box gets a `robot_bbox_overlap` attribute: the fraction of
its area covered by the annotator's own robot boxes. This script does not drop
on it. The drop belongs to `prepare_yolo26_finetune_dataset.py --drop-if`
(docs/YOLO.md §6.4), so that every box-level filter in the pipeline goes through
one mechanism with one accounting table:

    --drop-if 'robot_bbox_overlap>=0.5'

Human shapes never carry the attribute, so that gate is silent on them, which is
the correct asymmetry: the human already accounted for occlusion by eye.

The attribute is an upper bound on the real occlusion -- it uses the robot's
bounding box, not its silhouette -- so a threshold near 0.5 is deliberately
conservative.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

# The four classes whose VIKI labels are wrong or absent. See the module
# docstring for why sink/table/toaster are not in this list.
DEFAULT_FROM_RECONSTRUCTION = ("cabinet", "drawer", "fridge", "oven")

# Agents. Present in every reference render because MultiAgent.__init__ reads
# agents[0].scene, so an agent-free reference config cannot be rendered at all.
# The reference's agent is not the target's, so these must never be emitted.
AGENT_LABELS = frozenset({
    "fetch", "panda", "unitree_go2", "unitree_h1", "stompy", "anymal_c",
    "xarm6", "widowx", "google_robot",
})

OVERLAP_ATTR = "robot_bbox_overlap"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def shape_bbox(shape: dict) -> tuple[float, float, float, float]:
    xs = [p[0] for p in shape["points"]]
    ys = [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_points(bbox: tuple[float, float, float, float]) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def scale_bbox(bbox, sx: float, sy: float, width: int, height: int):
    """Map a reference-resolution bbox onto the target image.

    Separate sx and sy on purpose. VIKI's 600x337 is a full-frame resize of the
    layout camera's 1920x1080 (verified by sweeping a synthetic horizontal and
    vertical crop factor against the matched reference: NCC peaks sharply at
    1.00 in both axes, so there is no field-of-view difference to correct for).
    But 337 is the *floor* of 1080/3.2 = 337.5, so the vertical factor is
    337/1080, not 1/3.2 -- a 0.15% difference that a single scalar would smear
    into every box.
    """
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, min(float(width), x0 * sx)),
        max(0.0, min(float(height), y0 * sy)),
        max(0.0, min(float(width), x1 * sx)),
        max(0.0, min(float(height), y1 * sy)),
    )


def robot_bbox_overlap(bbox, robot_bboxes) -> float:
    """Fraction of ``bbox`` covered by robot boxes, as an upper bound.

    Intersections are summed rather than unioned. Robot boxes rarely overlap
    each other (they are distinct agents standing apart), and erring high is the
    safe direction for a gate that removes suspect supervision.
    """
    area = max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0)
    if area <= 0.0:
        return 0.0
    total = 0.0
    for rb in robot_bboxes:
        ix = max(0.0, min(bbox[2], rb[2]) - max(bbox[0], rb[0]))
        iy = max(0.0, min(bbox[3], rb[3]) - max(bbox[1], rb[1]))
        total += ix * iy
    return min(total / area, 1.0)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def load_label(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "shapes" not in data:
        return None
    return data


def index_human_labels(root: Path) -> dict[str, Path]:
    """stem -> label path, ignoring the merge bookkeeping file."""
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "merge_summary.json":
            continue
        index[path.stem] = path
    return index


def ref_label_path(refs_dir: Path, layout_id: int, style_idx: int, camera_idx: int) -> Path:
    return refs_dir / "labels" / f"L{layout_id}_s{style_idx:02d}_c{camera_idx}.json"


# ---------------------------------------------------------------------------
# per-image build
# ---------------------------------------------------------------------------
def build_one(row: dict, human: dict, ref: dict, from_reconstruction: frozenset[str],
              min_box_px: float) -> tuple[dict, dict]:
    """Merged label document plus per-image accounting."""
    width = int(human.get("imageWidth") or 0)
    height = int(human.get("imageHeight") or 0)
    ref_w = int(ref.get("imageWidth") or 0)
    ref_h = int(ref.get("imageHeight") or 0)
    if not (width and height and ref_w and ref_h):
        raise ValueError("missing image dimensions")
    sx, sy = width / ref_w, height / ref_h

    stats: dict = {
        "kept_human": collections.Counter(),
        "replaced_human": collections.Counter(),
        "reconstructed": collections.Counter(),
        "degenerate": collections.Counter(),
        "overlap_ge_50": collections.Counter(),
        "overlap_ge_90": collections.Counter(),
    }

    robot_bboxes = [
        shape_bbox(s) for s in human["shapes"]
        if str(s.get("label", "")).strip() in AGENT_LABELS
    ]

    shapes: list[dict] = []
    for shape in human["shapes"]:
        label = str(shape.get("label", "")).strip()
        if label in from_reconstruction:
            stats["replaced_human"][label] += 1
            continue
        out = json.loads(json.dumps(shape))  # deep copy, verbatim passthrough
        attrs = out.get("attributes")
        out["attributes"] = {**attrs, "source": "human"} if isinstance(attrs, dict) else {"source": "human"}
        shapes.append(out)
        stats["kept_human"][label] += 1

    ref_stem = f"L{row['layout_id']}_s{row['pred']['style_idx']:02d}_c{row['pred']['camera_idx']}"
    for shape in ref["shapes"]:
        label = str(shape.get("label", "")).strip()
        if label not in from_reconstruction:
            continue
        if label in AGENT_LABELS:  # cannot happen via the flag; assert the invariant
            raise ValueError(f"agent class {label!r} requested from reconstruction")
        bbox = scale_bbox(shape_bbox(shape), sx, sy, width, height)
        if (bbox[2] - bbox[0]) < min_box_px or (bbox[3] - bbox[1]) < min_box_px:
            stats["degenerate"][label] += 1
            continue
        overlap = robot_bbox_overlap(bbox, robot_bboxes)
        if overlap >= 0.5:
            stats["overlap_ge_50"][label] += 1
        if overlap >= 0.9:
            stats["overlap_ge_90"][label] += 1
        attrs = dict(shape.get("attributes") or {})
        attrs.update({
            "source": "reconstruction",
            OVERLAP_ATTR: round(overlap, 4),
            "ref_stem": ref_stem,
            "match_score": round(float(row["score"]), 4),
            "margin_camera": round(float(row["margin_camera"]), 4),
            "margin_style": round(float(row["margin_style"]), 4),
        })
        shapes.append({
            "label": label,
            "score": shape.get("score"),
            "points": bbox_to_points(bbox),
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": attrs,
            "kie_linking": [],
        })
        stats["reconstructed"][label] += 1

    doc = {
        "version": human.get("version", "4.0.0-beta.3"),
        "flags": human.get("flags", {}),
        "shapes": shapes,
        "imagePath": human.get("imagePath", f"{row['stem']}.png"),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": human.get("description", ""),
    }
    return doc, stats


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(args) -> dict:
    rows = json.loads(Path(args.match_report).read_text(encoding="utf-8"))["rows"]
    refs_dir = Path(args.refs)
    human_index = index_human_labels(Path(args.human_labels))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from_reconstruction = frozenset(
        c.strip() for c in args.from_reconstruction.split(",") if c.strip()
    )
    bad = from_reconstruction & AGENT_LABELS
    if bad:
        raise SystemExit(f"refusing to reconstruct agent classes: {sorted(bad)}")

    ref_cache: dict[Path, dict | None] = {}
    totals = {k: collections.Counter() for k in
              ("kept_human", "replaced_human", "reconstructed", "degenerate",
               "overlap_ge_50", "overlap_ge_90")}
    overlap_hist = collections.Counter()
    skipped: dict[str, list[str]] = collections.defaultdict(list)
    quarantined: list[dict] = []
    written = 0

    for row in rows:
        stem = row["stem"]
        if float(row["score"]) < args.min_score:
            quarantined.append({"stem": stem, "reason": "score", "score": row["score"]})
            continue
        if float(row["margin_camera"]) < args.min_margin_camera:
            quarantined.append({"stem": stem, "reason": "margin_camera",
                                "margin_camera": row["margin_camera"]})
            continue
        if float(row["margin_style"]) < args.min_margin_style:
            quarantined.append({"stem": stem, "reason": "margin_style",
                                "margin_style": row["margin_style"]})
            continue

        human_path = human_index.get(stem)
        if human_path is None:
            skipped["no_human_label"].append(stem)
            continue
        human = load_label(human_path)
        if human is None:
            skipped["unreadable_human_label"].append(stem)
            continue

        rp = ref_label_path(refs_dir, row["layout_id"],
                            row["pred"]["style_idx"], row["pred"]["camera_idx"])
        if rp not in ref_cache:
            ref_cache[rp] = load_label(rp)
        ref = ref_cache[rp]
        if ref is None:
            skipped["missing_reference_label"].append(stem)
            continue

        doc, stats = build_one(row, human, ref, from_reconstruction, args.min_box_px)
        for key, counter in stats.items():
            totals[key].update(counter)
        for shape in doc["shapes"]:
            value = (shape.get("attributes") or {}).get(OVERLAP_ATTR)
            if value is not None:
                overlap_hist[min(int(float(value) * 10), 9)] += 1

        rel = human_path.relative_to(Path(args.human_labels))
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    manifest = {
        "source_match_report": str(Path(args.match_report)),
        "source_refs": str(refs_dir),
        "source_human_labels": str(Path(args.human_labels)),
        "from_reconstruction": sorted(from_reconstruction),
        "overlap_attribute": OVERLAP_ATTR,
        "gates": {"min_score": args.min_score,
                  "min_margin_camera": args.min_margin_camera,
                  "min_margin_style": args.min_margin_style,
                  "min_box_px": args.min_box_px},
        "images_in_match_report": len(rows),
        "images_written": written,
        "images_quarantined": len(quarantined),
        "quarantined": quarantined[:200],
        "skipped": {k: {"n": len(v), "stems": v[:50]} for k, v in skipped.items()},
        "boxes_kept_from_human": dict(sorted(totals["kept_human"].items())),
        "boxes_replaced_from_human": dict(sorted(totals["replaced_human"].items())),
        "boxes_reconstructed": dict(sorted(totals["reconstructed"].items())),
        "boxes_dropped_degenerate": dict(sorted(totals["degenerate"].items())),
        "reconstructed_overlap_ge_50": dict(sorted(totals["overlap_ge_50"].items())),
        "reconstructed_overlap_ge_90": dict(sorted(totals["overlap_ge_90"].items())),
        "reconstructed_overlap_decile_hist": {f"{i/10:.1f}-{(i+1)/10:.1f}": overlap_hist[i]
                                              for i in range(10)},
        "note": (
            "robot_bbox_overlap is an UPPER bound (robot bbox, not silhouette) and is "
            "written, not enforced. Drop it downstream with "
            "prepare_yolo26_finetune_dataset.py --drop-if 'robot_bbox_overlap>=0.5'."
        ),
    }
    (out_dir / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match-report", required=True,
                   help="viki_match_report.json from locus_viki_reconstruct_probe.py match")
    p.add_argument("--refs", required=True, help="probe output dir (expects labels/ inside)")
    p.add_argument("--human-labels", required=True, help="reviewed X-AnyLabeling dir")
    p.add_argument("--out", required=True)
    p.add_argument("--from-reconstruction", default=",".join(DEFAULT_FROM_RECONSTRUCTION),
                   help="comma-separated classes taken from the reconstruction and "
                        "dropped from the human labels (default: %(default)s)")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-margin-camera", type=float, default=0.0)
    p.add_argument("--min-margin-style", type=float, default=0.0)
    p.add_argument("--min-box-px", type=float, default=2.0,
                   help="drop reconstructed boxes thinner than this after downscaling")
    return p.parse_args(argv)


def main(argv=None) -> int:
    manifest = run(parse_args(argv))
    keys = ("images_written", "images_quarantined", "boxes_reconstructed",
            "boxes_replaced_from_human", "reconstructed_overlap_ge_50",
            "boxes_dropped_degenerate", "skipped")
    print(json.dumps({k: manifest[k] for k in keys}, ensure_ascii=False, indent=1))
    kept = manifest["boxes_kept_from_human"]
    print(f"[recon] kept-from-human classes: {len(kept)}, boxes {sum(kept.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
