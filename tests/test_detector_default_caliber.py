"""Detector caliber: which weights a run actually loaded must be unambiguous.

The regression these pin (docs/YOLO.md §7.5.0, dataset_and_train.md L317). Two
detector calibers coexist in this repo and are NOT interchangeable:

  * ``checkpoints/yolo26l_locus_full34_v1.pt`` -- the frozen 34-class LOCUS
    detector, the headline caliber, and (since 2026-08-07) the module default;
  * ``checkpoints/best.pt`` -- the 20-class VIKI detector, the fixed caliber of
    the legacy cross-version protocol whose recorded SFT numbers are only
    comparable against that exact file.

Before the switch, a LOCUS run that forgot to pass weights silently got the
full20 model, which cannot emit 14 of the 34 classes at all. After the switch
the mirror-image hazard appears: a caller pinning ``weights_path`` to the legacy
checkpoint used to inherit ``run_dir`` from the module default, so it would read
the *full34* run's ``args.yaml -> data.yaml`` and describe 20-class weights with
34 class names. Both directions are silent, hence these tests.
"""
from __future__ import annotations

from pathlib import Path

from nsdl.yolo_detector import (
    DEFAULT_YOLO26_RUN_DIR,
    DEFAULT_YOLO26_WEIGHTS,
    LEGACY_VIKI_FULL20_WEIGHTS,
    UltralyticsYoloDetector,
)

REPO = Path(__file__).resolve().parents[1]


def test_module_default_is_the_frozen_full34_checkpoint():
    """§7.5.0: the default is the versioned LOCUS detector, not full20."""
    assert DEFAULT_YOLO26_WEIGHTS.name == "yolo26l_locus_full34_v1.pt"
    assert DEFAULT_YOLO26_WEIGHTS == DEFAULT_YOLO26_RUN_DIR
    assert UltralyticsYoloDetector().weights_path == DEFAULT_YOLO26_WEIGHTS


def test_legacy_full20_is_named_not_reachable_by_inheritance():
    """The old default stays available, but only by asking for it."""
    assert LEGACY_VIKI_FULL20_WEIGHTS.name == "best.pt"
    assert LEGACY_VIKI_FULL20_WEIGHTS != DEFAULT_YOLO26_WEIGHTS


def test_explicit_weights_do_not_inherit_the_default_run_dir():
    """Pinning weights must pin the class-name source with them.

    Otherwise the legacy full20 weights get described by the full34 run's
    data.yaml -- 20-class predictions annotated with 34 class names.
    """
    d = UltralyticsYoloDetector(weights_path=str(LEGACY_VIKI_FULL20_WEIGHTS))

    assert d.weights_path == LEGACY_VIKI_FULL20_WEIGHTS
    assert d.run_dir == LEGACY_VIKI_FULL20_WEIGHTS.parent
    # best.pt has no sibling args.yaml, so no class list is claimed at all --
    # which is honest. The failure mode is claiming the *wrong* one.
    assert d._supported_labels == []


def test_run_dir_alone_still_resolves_weights_and_labels_together():
    run_dir = REPO / "outputs" / "yolo_runs" / "yolo26l_locus_full34_v1"
    if not (run_dir / "args.yaml").exists():
        return  # training artifacts not present in this checkout

    d = UltralyticsYoloDetector(run_dir=str(run_dir))

    assert d.weights_path == run_dir / "weights" / "best.pt"
    assert len(d._supported_labels) == 34


def test_explicit_weights_win_over_an_explicit_run_dir():
    """The VIKI eval scripts pass both; weights must decide what is loaded."""
    d = UltralyticsYoloDetector(
        run_dir=str(REPO / "outputs" / "yolo_runs" / "yolo26l_locus_full34_v1"),
        weights_path=str(LEGACY_VIKI_FULL20_WEIGHTS),
    )

    assert d.weights_path == LEGACY_VIKI_FULL20_WEIGHTS
