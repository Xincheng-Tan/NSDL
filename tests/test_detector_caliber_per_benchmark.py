"""Detector caliber is explicit per benchmark (2026-09-05, facts:D3a/D3b).

History, so nobody "fixes" this back the wrong way:
  * before 2026-08-07 both benchmarks used the 20-class `checkpoints/best.pt`;
  * 2026-08-07 the module default moved to the 34-class LOCUS file, but the
    VIKI eval script kept a private 20-class pin, so VIKI-L2 v3.4.1 buckets ran
    on full20 and LOCUS on full34 without any bucket recording which;
  * 2026-09-05 a one-detector attempt (full34 everywhere) was measured
    (plan.md 序 25 步 0) and REJECTED: full34 detects `table` in only 980/1800
    VIKI test images vs 1314 for full20 and 9.6% of images lose an
    "x on table" relation. Ruling (a): each benchmark keeps the caliber its
    corpus and buckets were produced with, resolved from --level, guarded, and
    recorded in every artifact.

These tests pin that contract:
  1. the level -> weights map, and every entry point deferring to it;
  2. a cross-caliber file is refused unless LNSDL_ALLOW_DETECTOR_CALIBER_MISMATCH=1;
  3. provenance fields (path / sha256 / class count) are produced;
  4. no script re-spells `checkpoints/best.pt` as a literal default -- the
     caliber must come from the module's named constants (allowlist: the
     detector trainer's copy target and the annotation accelerator).
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nsdl.yolo_detector import (  # noqa: E402
    CALIBER_MISMATCH_OPT_IN_ENV,
    DEFAULT_YOLO26_WEIGHTS,
    LEGACY_VIKI_FULL20_WEIGHTS,
    LOCUS_DETECTOR_WEIGHTS,
    VIKI_DETECTOR_WEIGHTS,
    assert_detector_caliber_for_level,
    describe_detector_weights,
    detector_weights_for_level,
    is_legacy_full20_weights,
)


def test_level_map_is_the_documented_caliber() -> None:
    assert VIKI_DETECTOR_WEIGHTS == LEGACY_VIKI_FULL20_WEIGHTS
    assert VIKI_DETECTOR_WEIGHTS.name == "best.pt"
    assert LOCUS_DETECTOR_WEIGHTS == DEFAULT_YOLO26_WEIGHTS
    assert LOCUS_DETECTOR_WEIGHTS.name == "yolo26l_locus_full34_v1.pt"
    assert detector_weights_for_level("VIKI-L2") == VIKI_DETECTOR_WEIGHTS
    assert detector_weights_for_level("VIKI-L1") == VIKI_DETECTOR_WEIGHTS
    assert detector_weights_for_level("LOCUS") == LOCUS_DETECTOR_WEIGHTS
    with pytest.raises(ValueError):
        detector_weights_for_level("nope")


@pytest.mark.parametrize("module_name,attr", [
    ("eval_lnsdl_agentic_viki_l2", "DEFAULT_DETECTOR_WEIGHTS"),
])
def test_agentic_eval_defers_to_level(module_name: str, attr: str) -> None:
    mod = importlib.import_module(module_name)
    # None = "resolve from --level in main()"; a literal path here would be a re-pin.
    assert getattr(mod, attr) is None
    parser = mod.build_parser()
    assert parser.get_default("detector_weights") is None


def test_constant_offload_defers_to_level() -> None:
    mod = importlib.import_module("eval_constant_offload")
    assert mod.build_parser().get_default("detector_weights") is None


@pytest.mark.parametrize("module_name,attr,expected", [
    ("build_lnsdl_sft_data", "VIKI_DETECTOR_WEIGHTS", VIKI_DETECTOR_WEIGHTS),
    ("build_lnsdl_sft_data", "LOCUS_DETECTOR_WEIGHTS", LOCUS_DETECTOR_WEIGHTS),
    ("eval_viki_l1", "DEFAULT_EVAL_DETECTOR_WEIGHTS", VIKI_DETECTOR_WEIGHTS),
    ("eval_viki_l2", "DEFAULT_EVAL_DETECTOR_WEIGHTS", VIKI_DETECTOR_WEIGHTS),
])
def test_entry_point_constants_match_their_benchmark(module_name, attr, expected) -> None:
    mod = importlib.import_module(module_name)
    assert Path(getattr(mod, attr)) == expected


def test_cross_caliber_refused_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(CALIBER_MISMATCH_OPT_IN_ENV, raising=False)
    # matching caliber passes
    assert_detector_caliber_for_level(VIKI_DETECTOR_WEIGHTS, "VIKI-L2", caller="t")
    assert_detector_caliber_for_level(LOCUS_DETECTOR_WEIGHTS, "LOCUS", caller="t")
    assert_detector_caliber_for_level(None, "LOCUS", caller="t")
    # the other benchmark's file is refused both ways
    with pytest.raises(SystemExit):
        assert_detector_caliber_for_level(LOCUS_DETECTOR_WEIGHTS, "VIKI-L2", caller="t")
    with pytest.raises(SystemExit):
        assert_detector_caliber_for_level(VIKI_DETECTOR_WEIGHTS, "LOCUS", caller="t")
    # an unrelated third file (e.g. an open-vocab or v2 candidate) is not blocked here
    assert_detector_caliber_for_level(REPO / "checkpoints" / "some_other.pt", "VIKI-L2", caller="t")


def test_cross_caliber_allowed_with_opt_in(monkeypatch) -> None:
    monkeypatch.setenv(CALIBER_MISMATCH_OPT_IN_ENV, "1")
    assert_detector_caliber_for_level(LOCUS_DETECTOR_WEIGHTS, "VIKI-L2", caller="t")
    assert_detector_caliber_for_level(VIKI_DETECTOR_WEIGHTS, "LOCUS", caller="t")


def test_describe_reports_path_sha_and_class_count() -> None:
    d = describe_detector_weights(LOCUS_DETECTOR_WEIGHTS)
    assert d["detector_weights"] == str(LOCUS_DETECTOR_WEIGHTS)
    assert d["detector_is_legacy_full20"] is False
    if LOCUS_DETECTOR_WEIGHTS.is_file():
        assert d["detector_num_classes"] == 34
        assert len(d["detector_weights_sha256"]) == 64
    v = describe_detector_weights(VIKI_DETECTOR_WEIGHTS)
    assert v["detector_is_legacy_full20"] is True and is_legacy_full20_weights(VIKI_DETECTOR_WEIGHTS)
    if VIKI_DETECTOR_WEIGHTS.is_file():
        assert v["detector_num_classes"] == 20
    assert describe_detector_weights(None) == {"detector_weights": None}


_BEST_PT_ALLOWLIST = {
    "scripts/train_yolo26_detector.py",   # --copy-best-to target
    "scripts/auto_label_with_yolo.py",    # X-AnyLabeling pre-annotation accelerator
}
_PIN_RE = re.compile(r'(?:REPO|REPO_ROOT|_REPO_ROOT)\s*/\s*"checkpoints"\s*/\s*"best\.pt"')


def test_no_script_spells_the_legacy_path_as_a_default() -> None:
    offenders = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in _BEST_PT_ALLOWLIST:
            continue
        if _PIN_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, (
        "spell the caliber through nsdl.yolo_detector.VIKI_DETECTOR_WEIGHTS / "
        f"LOCUS_DETECTOR_WEIGHTS, not as a literal path: {offenders}"
    )
