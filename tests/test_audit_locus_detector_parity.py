"""Tests for scripts/audit_locus_detector_parity.py.

The audit answers one question no field in the result JSON records: did this
LOCUS bucket run the 34-class detector, or silently fall back to the 20-class
VIKI-legacy one? The only artifact-side evidence is which classes the detect
tool ever returned, so the tests pin that extraction and the pass criterion.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_locus_detector_parity.py"

spec = importlib.util.spec_from_file_location("audit_locus_detector_parity", SCRIPT)
PARITY = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(PARITY)

EXCLUSIVE = {"pumpkin", "scissors", "wine", "drawer"}


def _write_bucket(root: Path, bucket: str, samples: list[list[dict]]) -> Path:
    """samples = per-file list of tool result dicts, paired with tool_names."""
    bucket_dir = root / bucket / "ood_m"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    for i, results in enumerate(samples):
        payload = {
            "agentic_eval": {
                "trace": {
                    "tool_names": [r.pop("_tool", "detect") for r in results],
                    "tool_results": results,
                }
            }
        }
        (bucket_dir / f"{i:05d}_F4_s2m3.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / bucket


def test_full20_fallback_is_detected(tmp_path):
    _write_bucket(tmp_path, "old", [[{"entities": ["apple", "bowl"]}]] * 3)
    report = PARITY.audit(tmp_path, ["old"], EXCLUSIVE)
    (row,) = report["rows"]
    assert row["exclusive_hits"] == 0
    assert row["verdict"] == "full20-fallback-or-no-detect"
    assert row["files"] == 3


def test_full34_is_detected(tmp_path):
    _write_bucket(tmp_path, "new", [[{"entities": ["apple", "pumpkin"]}], [{"entities": ["wine"]}]])
    report = PARITY.audit(tmp_path, ["new"], EXCLUSIVE)
    (row,) = report["rows"]
    assert row["exclusive_hits"] == 2
    assert row["exclusive_classes_seen"] == ["pumpkin", "wine"]
    assert row["verdict"] == "full34"


def test_non_detect_tool_results_are_ignored(tmp_path):
    # A schedule result carrying an entity named like a class must not be
    # miscounted as perception evidence.
    _write_bucket(
        tmp_path,
        "mixed",
        [[{"_tool": "schedule", "entities": ["pumpkin"]}, {"entities": ["apple"]}]],
    )
    report = PARITY.audit(tmp_path, ["mixed"], EXCLUSIVE)
    (row,) = report["rows"]
    assert row["exclusive_hits"] == 0
    assert row["classes"] == ["apple"]


def test_bucket_with_no_detect_calls_reads_as_no_evidence(tmp_path):
    _write_bucket(tmp_path, "empty", [[{"_tool": "schedule", "ok": True}]] * 2)
    report = PARITY.audit(tmp_path, ["empty"], EXCLUSIVE)
    (row,) = report["rows"]
    assert row["files"] == 2 and row["distinct_classes"] == 0
    assert row["verdict"] == "full20-fallback-or-no-detect"


def test_missing_bucket_is_a_hard_error(tmp_path):
    with pytest.raises(SystemExit, match="missing bucket"):
        PARITY.audit(tmp_path, ["nope"], EXCLUSIVE)


def test_require_exclusive_gates(tmp_path, monkeypatch, capsys):
    _write_bucket(tmp_path, "old", [[{"entities": ["apple"]}]])
    _write_bucket(tmp_path, "new", [[{"entities": ["pumpkin"]}]])
    monkeypatch.setattr(PARITY, "load_class_names", lambda p: EXCLUSIVE if "34" in str(p) else set())
    base = ["--root", str(tmp_path), "--full20-weights", "w20", "--full34-weights", "w34"]
    assert PARITY.main([*base, "--buckets", "new", "--require-exclusive"]) == 0
    assert PARITY.main([*base, "--buckets", "old,new", "--require-exclusive"]) == 1
    assert PARITY.main([*base, "--buckets", "old,new"]) == 0  # diagnostic mode
    assert "WARNING" in capsys.readouterr().out


def test_json_out_is_written(tmp_path, monkeypatch):
    _write_bucket(tmp_path, "new", [[{"entities": ["pumpkin"]}]])
    monkeypatch.setattr(PARITY, "load_class_names", lambda p: EXCLUSIVE if "34" in str(p) else set())
    out = tmp_path / "nested" / "parity.json"
    rc = PARITY.main(
        [
            "--root", str(tmp_path), "--buckets", "new",
            "--full20-weights", "w20", "--full34-weights", "w34",
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["rows"][0]["verdict"] == "full34"
