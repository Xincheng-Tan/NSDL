"""A0b G1: the two defects that made the first LOCUS SFT build exit 2.

Both were silent-in-VIKI, fatal-in-LOCUS, and both are pinned here because
neither raises on regression -- they produce a plausible dataset that teaches
the wrong thing.

  1. TEACHER CACHE SCOPE. `teacher_cache_key` keys on `task_id`. On VIKI that
     is a row identity; on LOCUS it is the grid CELL, shared by 210 rows. With
     `--teacher-include-image`, a cache hit therefore hands a row a think that
     was generated while LOOKING AT A DIFFERENT RENDER. The first build showed
     it plainly: 6300 rows, 4647 "cache" hits, 2136 distinct keys.

  2. SCHEDULE SELF-VALIDATION. The intent sketch drops MECHANICAL_OPS and the
     schedule tool re-inserts them. LOCUS uses two-level destinations
     (`carry_to(x -> area)` then `deposit(x -> container)`), which the tool
     cannot reconstruct: it infers the carry destination from the deposit
     target. Emitting that call would put a Phi result in the trajectory that
     contradicts the `final_plan` the same trajectory is trained to emit.
     471/4236 scheduled LOCUS rows are affected; VIKI has 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_lnsdl_sft_data import (  # noqa: E402
    DropSample,
    load_locus_frame,
    load_locus_image_index,
    prepare_trajectory,
    teacher_cache_key,
)
from scripts.build_slm_training_data import build_slm_system_prompt  # noqa: E402

LOCUS_TRAIN_PARQUET = REPO / "data" / "lnsdl_bench" / "ramp63k" / "train" / "data.parquet"
LOCUS_ROOT_DIR = REPO / "data" / "lnsdl_bench" / "ramp63k"

# Pinned from the real corpus. 3361/3362 are two DISTINCT rows of the SAME grid
# cell F4_s2m2 -- i.e. exactly the collision shape that defect 1 produces --
# and both are rows whose schedule cannot reconstruct GT (defect 2).
GATED_ROWS = (3361, 3362)
CLEAN_ROW = 0


class _StubItem:
    """Minimal surface `teacher_cache_key` reads."""

    def __init__(self, cache_row_scope=None):
        self.task_id = "F4_s2m2"
        self.pre_goal_tool_count = 0
        self.goal = {"targets": []}
        self.assign = {}
        self.plan_sketch = []
        self.tool_calls = []
        self.coverage_repair = None
        self.repair = None
        self.final_plan = []
        self.audit = {}
        if cache_row_scope is not None:
            self.cache_row_scope = cache_row_scope


def _key(item):
    return teacher_cache_key(
        item, "gpt-5.6-terra", include_image=True, post_include_image=False, think_mode="blind"
    )


def test_cache_key_separates_rows_of_the_same_locus_cell():
    a = _key(_StubItem("train#3361"))
    b = _key(_StubItem("train#3362"))
    assert a != b, "two rows of one LOCUS cell must not share a teacher cache entry"


def test_cache_key_unchanged_when_no_row_scope():
    """VIKI must stay byte-identical so the v1-v3.3 cache remains reusable."""
    without_attr = _StubItem()
    assert not hasattr(without_attr, "cache_row_scope")
    explicit_none = _StubItem()
    explicit_none.cache_row_scope = None
    assert _key(without_attr) == _key(explicit_none)
    # ...and adding a scope must actually move the key, or the guard is inert.
    assert _key(without_attr) != _key(_StubItem("train#0"))


@pytest.fixture(scope="module")
def locus_ctx():
    if not LOCUS_TRAIN_PARQUET.exists():
        pytest.skip("LOCUS-Bench train split not present")
    return (
        load_locus_frame(LOCUS_TRAIN_PARQUET),
        load_locus_image_index(LOCUS_ROOT_DIR, "train"),
        build_slm_system_prompt(),
    )


def _prepare(locus_ctx, row_index, *, require):
    df, image_index, system_prompt = locus_ctx
    return prepare_trajectory(
        df.iloc[row_index],
        row_index=row_index,
        image_index=image_index,
        system_prompt=system_prompt,
        sketch_variant="intent",
        detect_policy="none",
        detector=None,
        goal_coverage_policy="auto",
        status_curriculum=True,
        pre_goal_tools=True,
        schedule_policy="balanced",
        schedule_boundary_ratio=0.5,
        repair_rate=0.03,
        seed=20260614,
        goal_coverage_direct_rate=0.85,
        crop_demo_policy="none",
        query_source="instruction",
        source="locus",
        locus_split="train",
        require_schedule_gt_match=require,
    )


@pytest.mark.parametrize("row_index", GATED_ROWS)
def test_unreconstructable_schedule_is_dropped_not_emitted(locus_ctx, row_index):
    ungated = _prepare(locus_ctx, row_index, require=False)
    assert ungated.audit["schedule_used"] is True
    assert ungated.audit["schedule_action_set_match_gt"] is False, (
        "fixture drifted: row %d no longer reproduces the carrier defect" % row_index
    )

    gated = _prepare(locus_ctx, row_index, require=True)
    assert gated.audit["schedule_gt_reconstruct_failed"] is True
    assert gated.audit["schedule_used"] is False
    assert "schedule" not in gated.audit["tool_names"]
    # The fallback must be an ordinary no-schedule trajectory, not a stub: the
    # row still trains the final plan, it just loses its Phi demonstration.
    assert gated.final_plan == ungated.final_plan
    # And the audit must leave no key the dataset auditor would read as a pass.
    assert "schedule_action_set_match_gt" not in gated.audit


def test_gate_is_inert_where_the_tool_reconstructs_gt(locus_ctx):
    try:
        gated = _prepare(locus_ctx, CLEAN_ROW, require=True)
        ungated = _prepare(locus_ctx, CLEAN_ROW, require=False)
    except DropSample as exc:  # pragma: no cover - fixture drift
        pytest.skip(f"row {CLEAN_ROW} dropped: {exc.code}")
    assert gated.audit["schedule_gt_reconstruct_failed"] is False
    assert gated.audit == ungated.audit
    assert gated.tool_calls == ungated.tool_calls
