"""Unit tests for the VIKI reconstruction pipeline (docs/YOLO.md §6.6).

Each test targets a way the pipeline could produce a plausible-looking label file
that is quietly wrong:

* a single scalar scale would smear the 337-vs-337.5 vertical floor into every box
* forgetting to drop the human's `cabinet` boxes would ship BOTH calibers at once,
  which is the exact disease being cured
* emitting the reference's own agent would label a robot that is not in the frame
* stamping `robot_bbox_overlap` onto human shapes would make the downstream gate
  delete boxes the annotator got right
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_viki_reconstructed_labels import (  # noqa: E402
    AGENT_LABELS,
    DEFAULT_FROM_RECONSTRUCTION,
    OVERLAP_ATTR,
    build_one,
    robot_bbox_overlap,
    run,
    scale_bbox,
    shape_bbox,
)
from scripts.prepare_yolo26_finetune_dataset import (  # noqa: E402
    gate_verdict,
    parse_attribute_gate,
)

REF_W, REF_H = 1920, 1080
TGT_W, TGT_H = 600, 337


def _shape(label: str, bbox, attributes: dict | None = None) -> dict:
    x0, y0, x1, y1 = bbox
    return {
        "label": label,
        "score": None,
        "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": dict(attributes or {}),
        "kie_linking": [],
    }


def _doc(width: int, height: int, shapes: list[dict], stem: str = "0001_deadbeef") -> dict:
    return {
        "version": "4.0.0-beta.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{stem}.png",
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }


def _row(stem: str = "0001_deadbeef", layout: int = 0, style: int = 4, camera: int = 2) -> dict:
    return {
        "stem": stem,
        "layout_id": layout,
        "pred": {"style_idx": style, "camera_idx": camera},
        "score": 0.9481,
        "margin_camera": 0.6499,
        "margin_style": 0.2750,
    }


FROM_RECON = frozenset(DEFAULT_FROM_RECONSTRUCTION)


def test_vertical_scale_is_not_the_horizontal_one() -> None:
    """337 is floor(1080/3.2) = floor(337.5), so a single 1/3.2 factor overshoots
    the frame by half a pixel on every box and needs clipping to hide it."""
    assert TGT_W / REF_W == pytest.approx(0.3125)
    assert TGT_H / REF_H == pytest.approx(0.31204, abs=1e-5)

    full = scale_bbox((0.0, 0.0, REF_W, REF_H), TGT_W / REF_W, TGT_H / REF_H, TGT_W, TGT_H)
    assert full == pytest.approx((0.0, 0.0, 600.0, 337.0))

    # A single scalar would put the bottom edge at 337.5 -- outside the image.
    assert REF_H * (TGT_W / REF_W) == pytest.approx(337.5)

    mid = scale_bbox((960.0, 540.0, 1920.0, 1080.0), TGT_W / REF_W, TGT_H / REF_H, TGT_W, TGT_H)
    assert mid == pytest.approx((300.0, 168.5, 600.0, 337.0), abs=1e-3)
    # The same edge under one scalar: 168.75, a quarter pixel lower.
    assert 540.0 * (TGT_W / REF_W) == pytest.approx(168.75)

    # Clipping is applied, not assumed away.
    assert scale_bbox((-50.0, -50.0, 5000.0, 5000.0), 1.0, 1.0, TGT_W, TGT_H) == (
        0.0, 0.0, float(TGT_W), float(TGT_H))


def test_robot_overlap_is_a_clamped_upper_bound() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert robot_bbox_overlap(box, []) == 0.0
    assert robot_bbox_overlap(box, [(20.0, 20.0, 30.0, 30.0)]) == 0.0
    assert robot_bbox_overlap(box, [(0.0, 0.0, 10.0, 5.0)]) == pytest.approx(0.5)
    assert robot_bbox_overlap(box, [(-5.0, -5.0, 15.0, 15.0)]) == pytest.approx(1.0)
    # Two robot boxes covering the same half sum to 1.0, not 0.5: intersections
    # are summed, so the value is an upper bound. It must still clamp at 1.0.
    twice = [(0.0, 0.0, 10.0, 6.0), (0.0, 4.0, 10.0, 10.0)]
    assert robot_bbox_overlap(box, twice) == pytest.approx(1.0)
    # A degenerate box must not divide by zero.
    assert robot_bbox_overlap((5.0, 5.0, 5.0, 5.0), twice) == 0.0


def test_build_one_replaces_only_the_four_and_keeps_the_rest_verbatim() -> None:
    human = _doc(TGT_W, TGT_H, [
        # The annotator's upper-cabinet caliber: must be REMOVED, not merged.
        _shape("cabinet", (10, 10, 50, 40)),
        # Equally-calibrated fixtures and movables: kept, attributes untouched.
        _shape("sink", (100, 200, 140, 230)),
        _shape("toaster", (300, 100, 320, 115)),
        _shape("apple", (400, 150, 410, 160)),
        # The target's robot: kept, and it is what the overlap is measured against.
        _shape("fetch", (0, 168, 300, 337)),
    ])
    ref = _doc(REF_W, REF_H, [
        _shape("cabinet", (0, 0, 320, 160), {"visible_pixels": 5000, "view_cos": 0.8}),
        _shape("drawer", (0, 540, 320, 1080), {"visible_pixels": 4000}),   # behind the robot
        _shape("fridge", (1600, 300, 1900, 900), {"visible_pixels": 9000}),
        _shape("sink", (400, 700, 500, 800)),      # not requested -> ignored
        _shape("fetch", (500, 500, 900, 800)),     # reference's own agent -> never emitted
    ])

    doc, stats = build_one(_row(), human, ref, FROM_RECON, min_box_px=2.0)

    labels = [s["label"] for s in doc["shapes"]]
    assert labels.count("cabinet") == 1                      # the reconstructed one only
    assert stats["replaced_human"]["cabinet"] == 1
    assert stats["reconstructed"] == {"cabinet": 1, "drawer": 1, "fridge": 1}
    assert labels.count("sink") == 1 and labels.count("toaster") == 1
    assert labels.count("fetch") == 1                        # the human's, not the reference's

    by_label = {s["label"]: s for s in doc["shapes"]}
    # The human's fetch box survived unchanged apart from the source stamp.
    assert shape_bbox(by_label["fetch"]) == (0.0, 168.0, 300.0, 337.0)
    assert by_label["fetch"]["attributes"] == {"source": "human"}
    assert by_label["apple"]["attributes"]["source"] == "human"

    cab = by_label["cabinet"]["attributes"]
    assert cab["source"] == "reconstruction"
    # Exporter telemetry survives, so the P0-4/P0-6 pixel gates still apply.
    assert cab["visible_pixels"] == 5000 and cab["view_cos"] == 0.8
    assert cab["ref_stem"] == "L0_s04_c2"
    assert cab["match_score"] == 0.9481 and cab["margin_camera"] == 0.6499
    assert cab[OVERLAP_ATTR] == 0.0                          # top-left, clear of the robot

    # The drawer sits in the robot's quadrant, so it must be flagged.
    assert by_label["drawer"]["attributes"][OVERLAP_ATTR] == pytest.approx(1.0)
    assert stats["overlap_ge_50"]["drawer"] == 1 and stats["overlap_ge_90"]["drawer"] == 1

    assert doc["imageWidth"] == TGT_W and doc["imageHeight"] == TGT_H
    assert doc["imageData"] is None
    assert doc["imagePath"] == "0001_deadbeef.png"


def test_reconstructed_boxes_land_inside_the_target_frame() -> None:
    human = _doc(TGT_W, TGT_H, [])
    ref = _doc(REF_W, REF_H, [_shape("cabinet", (0, 0, REF_W, REF_H))])
    doc, _ = build_one(_row(), human, ref, FROM_RECON, min_box_px=2.0)
    x0, y0, x1, y1 = shape_bbox(doc["shapes"][0])
    assert (x0, y0) == (0.0, 0.0) and (x1, y1) == (600.0, 337.0)


def test_boxes_too_thin_after_downscaling_are_dropped_and_counted() -> None:
    """A 3 px reference sliver is under 1 px at 600x337; emitting it would be a
    box with no content, and silently emitting it would inflate the class count."""
    human = _doc(TGT_W, TGT_H, [])
    ref = _doc(REF_W, REF_H, [
        _shape("drawer", (100, 100, 103, 400)),   # 3 px wide -> 0.94 px
        _shape("drawer", (100, 100, 200, 400)),   # survives
    ])
    _doc_out, stats = build_one(_row(), human, ref, FROM_RECON, min_box_px=2.0)
    assert stats["degenerate"]["drawer"] == 1
    assert stats["reconstructed"]["drawer"] == 1


def test_agent_classes_cannot_be_reconstructed() -> None:
    human = _doc(TGT_W, TGT_H, [])
    ref = _doc(REF_W, REF_H, [_shape("fetch", (0, 0, 100, 100))])
    with pytest.raises(ValueError, match="agent class"):
        build_one(_row(), human, ref, frozenset({"fetch"}), min_box_px=2.0)
    assert AGENT_LABELS & frozenset(DEFAULT_FROM_RECONSTRUCTION) == frozenset()


def test_downstream_robot_gate_hits_reconstruction_and_spares_the_human() -> None:
    """The two sources must react differently to the SAME --drop-if spec: the
    annotator already accounted for occlusion by eye, so a robot-overlap gate has
    to be silent on human boxes."""
    human = _doc(TGT_W, TGT_H, [
        _shape("sink", (0, 200, 100, 300)),      # fully behind the robot below
        _shape("fetch", (0, 168, 300, 337)),
    ])
    ref = _doc(REF_W, REF_H, [
        _shape("drawer", (0, 540, 320, 1080)),   # same region, reconstructed
        _shape("cabinet", (1600, 0, 1900, 200)),
    ])
    doc, _ = build_one(_row(), human, ref, FROM_RECON, min_box_px=2.0)
    gates = [parse_attribute_gate(f"{OVERLAP_ATTR}>=0.5")]

    verdicts = {
        s["label"]: gate_verdict(s["label"], s["attributes"], gates, False)
        for s in doc["shapes"]
    }
    assert verdicts["drawer"][0] == f"{OVERLAP_ATTR}>=0.5"       # dropped
    assert verdicts["sink"] == (None, True)                      # kept, attribute absent
    assert verdicts["cabinet"][0] is None                        # kept, overlap 0


def test_run_writes_labels_and_a_manifest_that_explains_the_deltas(tmp_path: Path) -> None:
    refs = tmp_path / "refs" / "labels"
    refs.mkdir(parents=True)
    (refs / "L0_s04_c2.json").write_text(json.dumps(_doc(REF_W, REF_H, [
        _shape("cabinet", (0, 0, 320, 160), {"visible_pixels": 5000}),
        _shape("drawer", (0, 540, 320, 1080), {"visible_pixels": 4000}),
    ])), encoding="utf-8")

    human_dir = tmp_path / "human"
    human_dir.mkdir()
    (human_dir / "0001_deadbeef.json").write_text(json.dumps(_doc(TGT_W, TGT_H, [
        _shape("cabinet", (10, 10, 50, 40)),
        _shape("sink", (100, 200, 140, 230)),
        _shape("fetch", (0, 168, 300, 337)),
    ])), encoding="utf-8")
    # Bookkeeping files must not be mistaken for labels.
    (human_dir / "merge_summary.json").write_text('{"merged": 1}', encoding="utf-8")

    report = tmp_path / "match.json"
    report.write_text(json.dumps({"n": 3, "rows": [
        _row("0001_deadbeef"),
        _row("0002_nolabel"),                                   # no human label
        dict(_row("0003_lowmargin"), margin_camera=0.01),       # quarantined
    ]}), encoding="utf-8")

    out = tmp_path / "out"
    manifest = run(type("A", (), {
        "match_report": str(report), "refs": str(tmp_path / "refs"),
        "human_labels": str(human_dir), "out": str(out),
        "from_reconstruction": ",".join(DEFAULT_FROM_RECONSTRUCTION),
        "min_score": 0.0, "min_margin_camera": 0.1, "min_margin_style": 0.0,
        "min_box_px": 2.0,
    })())

    assert manifest["images_written"] == 1
    assert manifest["images_quarantined"] == 1
    assert manifest["quarantined"][0]["reason"] == "margin_camera"
    assert manifest["skipped"]["no_human_label"]["n"] == 1
    assert manifest["boxes_reconstructed"] == {"cabinet": 1, "drawer": 1}
    assert manifest["boxes_replaced_from_human"] == {"cabinet": 1}
    assert manifest["boxes_kept_from_human"] == {"fetch": 1, "sink": 1}
    assert manifest["reconstructed_overlap_ge_50"] == {"drawer": 1}
    assert manifest["reconstructed_overlap_decile_hist"]["0.9-1.0"] == 1
    assert manifest["reconstructed_overlap_decile_hist"]["0.0-0.1"] == 1

    written = json.loads((out / "0001_deadbeef.json").read_text())
    assert sorted(s["label"] for s in written["shapes"]) == ["cabinet", "drawer", "fetch", "sink"]
    assert (out / "reconstruction_manifest.json").is_file()
