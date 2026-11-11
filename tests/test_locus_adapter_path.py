"""LOCUS-Bench routed through the shared benchmark adapter (P10.S2).

S2's y-axis is "at what M depth does the Phi-offload gain turn positive", which
needs LOCUS rows binned by their M level -- VIKI-L2 has no M axis at all. LOCUS
rows carry the *same* column schema as VIKI, so the agentic eval loop is shared
verbatim and only the on-disk path differs. These tests pin the three things
that would silently corrupt the S2 table if they drifted:

  * the LOCUS split layout (``<split>/data.parquet``) stays distinct from the
    VIKI layout (``<level>/<split>.parquet``);
  * the M depth is read from ``locus_meta.cell[1]`` and agrees with the
    ``sNmM`` task-name encoding -- binning on the wrong field would relabel
    every point on the y-axis while still producing a plausible-looking curve;
  * the LOCUS ruler and the official judge disagree **only** on null-plan rows.

The dataset-backed tests skip when the frozen bench is absent, so a checkout
without the 15 GB parquet still runs the suite.
"""
from __future__ import annotations

import re

import pytest

from nsdl.prompt_adapter import (
    DATA_ROOT,
    LOCUS_ROOT,
    _split_path,
    load_benchmark_samples,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SPLITS = ("test_id", "ood_s", "ood_m", "ood_x")


def test_split_path_layouts_are_distinct() -> None:
    """LOCUS is a directory per split; VIKI is a file per split."""
    assert _split_path("LOCUS", "ood_m") == LOCUS_ROOT / "ood_m" / "data.parquet"
    assert _split_path("VIKI-L2", "val") == DATA_ROOT / "VIKI-L2" / "val.parquet"


def _require(split: str):
    if not (LOCUS_ROOT / split / "data.parquet").exists():
        pytest.skip(f"frozen LOCUS split not present: {split}")
    return load_benchmark_samples(level="LOCUS", split=split)


@pytest.mark.parametrize("split", SPLITS)
def test_locus_rows_survive_the_shared_adapter(split: str) -> None:
    """Every row must yield a usable sample -- a parse failure here would be
    reported as a model error rather than as a loader bug."""
    samples = _require(split)
    assert samples
    for sample in samples:
        assert sample.image_bytes, f"{split}: missing render"
        assert sample.instruction.strip()
        # LOCUS prompts use VIKI's "Available robot set" form; if that ever
        # changed, extract_available_robot_set would raise, not return {}.
        assert sample.robot_inventory
        assert sample.robot_inventory_scope == "available"


@pytest.mark.parametrize("split", SPLITS)
def test_m_depth_matches_the_task_name_encoding(split: str) -> None:
    """``locus_meta.cell`` is (S, M) and must agree with the ``sNmM`` name."""
    for sample in _require(split):
        meta = sample.ground_truth["locus_meta"]
        cell = meta["cell"]
        match = re.search(r"s(\d+)m(\d+)", sample.task_name or "")
        assert match, f"unparsable task_name: {sample.task_name}"
        assert (int(match.group(1)), int(match.group(2))) == (cell[0], cell[1])
        assert meta["base_mechanical_signature"]["m_level"] == cell[1]


def test_m_axis_spans_the_full_depth_range() -> None:
    """The eval splits must cover M0-M4, else the y-axis has nothing to say at
    the deep end (M3/M4 live only in ood_m and ood_x)."""
    depths = {
        sample.ground_truth["locus_meta"]["cell"][1]
        for split in SPLITS
        for sample in _require(split)
    }
    assert depths == {0, 1, 2, 3, 4}


def test_the_two_rulers_disagree_only_on_null_plan_rows() -> None:
    """Pinned against the live splits: reading LOCUS off the official judge
    would zero every correct "nothing to do"."""
    from nsdl.benchmark_scoring import compute_locus_l2_acc, compute_viki_l2_score

    samples = _require("test_id")
    null_rows = [s for s in samples if not s.ground_truth.get("time_steps")]
    assert null_rows, "test_id is expected to contain null-plan rows"
    for sample in null_rows:
        assert compute_locus_l2_acc([], sample.ground_truth) == 1.0
        assert compute_viki_l2_score([], sample.ground_truth) < 1.0
