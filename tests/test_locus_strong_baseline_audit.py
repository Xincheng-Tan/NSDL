"""§7-7 / §7-8 backfill: missing coverage must never masquerade as a zero.

The regression these pin (docs/benchmark.md §7-7, §7.7). §7-7 sends a cell that
is *saturated or all-zero for every method* back for difficulty repair. The
obvious implementation -- "mean the items I have for this cell" -- turns a
collection hole into a finding, because ``mean([]) -> 0`` and a cell nobody
answered reads exactly like a cell everybody failed. §7.7 measured the damage
on the real collection: without coverage filtering, 0 of 49 family-cells are
hit instead of the true 2.

So the tests below pin, in order: the coverage gate on a single (row, cell)
reading, the row-count gate that makes a thin cell ``undetermined`` rather than
a pass, the max/min asymmetry that keeps a partly-solved cell out of the
all-zero bucket, family-cell granularity (§7.7 gates at ``F8_s0m3``, not at
``F8``), and the construction-evidence adjudication that decides whether an
all-zero cell is a defect or genuine difficulty.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_locus_strong_baseline import (  # noqa: E402
    CELL_MIN_N,
    MIN_ROWS,
    build,
    cell_levels,
    cell_name,
    cell_verdicts,
    construction_evidence,
    eval_fairness,
    summarize_row,
)

# The three construction gates as ramp63k actually reports them, all clean.
CLEAN_AUDIT = {
    "dual_validator": {"simenv_pass_rate": 1.0, "phi_agreement": 1.0,
                       "gate": 0.995, "ok": True},
    "goal_text_recoverability": {"goals_checked": 11372, "n_violations": 0,
                                 "ok": True},
}


def make_row(name: str, cells: dict, *, model=None, condition="image",
             coverage=None) -> dict:
    """A summarized row: ``cells`` maps cell -> (acc, n)."""
    per = {c: {"acc": a, "n": n} for c, (a, n) in cells.items()}
    return {
        "model": model or name,
        "condition": condition,
        "api_source": "relay",
        "coverage": coverage or {"collected": 560, "total": 560, "missing": 0,
                                 "stale_text": 0, "frac": 1.0,
                                 "per_split": {}},
        "acc": 0.3, "acc_official": 0.25, "format": 1.0,
        "outcomes": {}, "prompt_text": {},
        "per_split_acc": {},
        "per_cell_acc": per,
        "per_family_cell_acc": per,
    }


def rows_all(cells: dict, n_rows: int = MIN_ROWS) -> list[dict]:
    return [make_row(f"m{i}", cells) for i in range(n_rows)]


# --------------------------------------------------------------------------
# 1. cell naming (the crash that started this)
# --------------------------------------------------------------------------

def test_cell_name_normalizes_the_scorers_list_form():
    """`locus_meta` stores the cell as a 2-list, which cannot key a dict."""
    assert cell_name([2, 3]) == "s2m3"
    assert cell_name("s2m3") == "s2m3"          # idempotent


def test_cell_levels_round_trips():
    assert cell_levels("s0m4") == (0, 4)
    assert cell_levels(cell_name([3, 1])) == (3, 1)


# --------------------------------------------------------------------------
# 2. the coverage trap
# --------------------------------------------------------------------------

def test_thin_reading_is_dropped_not_counted_as_zero():
    """A cell with 1 collected item per row is unknown, not all-zero.

    This is the trap in miniature: every row reads 0.0, but each reading rests
    on a single item, so the honest verdict is that we do not know.
    """
    res = cell_verdicts(rows_all({"s0m4": (0.0, 1)}))

    assert res["cells"]["s0m4"]["verdict"] == "undetermined"
    assert res["cells"]["s0m4"]["n_rows_qualifying"] == 0


def test_undetermined_is_not_a_pass():
    """§7-7 has no "unknown ⇒ fine" branch; a thin cell blocks the gate."""
    rows = rows_all({"s0m4": (0.0, 1)})
    grid = cell_verdicts(rows)
    assert grid["counts"]["undetermined"] == 1
    assert build_from_rows(rows, CLEAN_AUDIT)["summary"]["ok"] is False


def test_empty_collection_does_not_pass_by_vacuity():
    """No cells means no verdicts, which must not read as "nothing wrong"."""
    out = build([], audit=CLEAN_AUDIT)["strong_baseline_prerun"]

    assert out["summary"]["ok"] is False
    assert "need >= " in out["summary"]["blocking_reason"]


def test_cell_needs_enough_rows_not_just_enough_items():
    """Two well-covered rows do not settle a cell the other six missed.

    A relay blackout hits whole rows, so "the rows that answered all scored 0"
    is precisely the reading that must not be trusted.
    """
    rows = ([make_row(f"m{i}", {"s0m4": (0.0, 20)}) for i in range(2)]
            + [make_row(f"m{i}", {}) for i in range(2, 8)])

    res = cell_verdicts(rows)

    assert res["cells"]["s0m4"]["verdict"] == "undetermined"
    assert res["cells"]["s0m4"]["n_rows_qualifying"] == 2


def test_all_zero_needs_every_qualifying_row_at_zero():
    """max(), not mean(): one method scoring above 0 means real difficulty.

    A mean would let seven zeros drown a single 0.35 and send a perfectly
    discriminative cell back for repair.
    """
    cells = {"s0m4": (0.0, 20)}
    rows = rows_all(cells)
    rows[3]["per_cell_acc"]["s0m4"] = {"acc": 0.35, "n": 20}
    rows[3]["per_family_cell_acc"] = rows[3]["per_cell_acc"]

    res = cell_verdicts(rows)

    assert res["cells"]["s0m4"]["verdict"] == "discriminative"
    assert res["cells"]["s0m4"]["best"] == 0.35


def test_saturated_needs_every_qualifying_row_at_ceiling():
    """min(): one method below the ceiling means the cell still separates."""
    rows = rows_all({"s0m0": (1.0, 20)})
    assert cell_verdicts(rows)["cells"]["s0m0"]["verdict"] == "saturated"

    rows[2]["per_cell_acc"]["s0m0"] = {"acc": 0.60, "n": 20}
    rows[2]["per_family_cell_acc"] = rows[2]["per_cell_acc"]
    assert cell_verdicts(rows)["cells"]["s0m0"]["verdict"] == "discriminative"


def test_cell_min_n_boundary_is_inclusive():
    at = cell_verdicts(rows_all({"s1m1": (0.0, CELL_MIN_N)}))
    below = cell_verdicts(rows_all({"s1m1": (0.0, CELL_MIN_N - 1)}))

    assert at["cells"]["s1m1"]["verdict"] == "all_zero"
    assert below["cells"]["s1m1"]["verdict"] == "undetermined"


# --------------------------------------------------------------------------
# 3. granularity: §7.7 gates at the family-cell
# --------------------------------------------------------------------------

def test_family_cell_granularity_does_not_average_away_the_cell():
    """F9's easy m2 bridge must not rescue its all-zero m4 relay.

    Grouping by family ("F9") is the coarse mistake: the m2 mean lifts the
    aggregate off zero and the exact cell under review vanishes.
    """
    rows = []
    for i in range(MIN_ROWS):
        r = make_row(f"m{i}", {})
        r["per_family_cell_acc"] = {"F9_s0m2": {"acc": 0.62, "n": 20},
                                    "F9_s0m4": {"acc": 0.0, "n": 20}}
        r["per_cell_acc"] = r["per_family_cell_acc"]
        rows.append(r)

    fam = cell_verdicts(rows, "per_family_cell_acc")

    assert fam["cells"]["F9_s0m4"]["verdict"] == "all_zero"
    assert fam["cells"]["F9_s0m2"]["verdict"] == "discriminative"


def test_summarize_row_groups_family_cell_by_task_name():
    res = {
        "model": "m", "condition": "image", "api_source": "relay",
        "coverage": {"test_id": {"collected": 2, "total": 3, "missing": 1,
                                 "stale_text": 0}},
        "outcomes": {}, "prompt_text": {},
        "per_item": {
            "test_id:0": {"split": "test_id", "acc": 1.0, "acc_official": 1.0,
                          "format": 1.0, "cell": [0, 4], "family": "F9",
                          "task_name": "F9_s0m4"},
            "test_id:1": {"split": "test_id", "acc": 0.0, "acc_official": 0.0,
                          "format": 1.0, "cell": [0, 2], "family": "F9",
                          "task_name": "F9_s0m2"},
        },
    }

    out = summarize_row(res)

    assert set(out["per_family_cell_acc"]) == {"F9_s0m4", "F9_s0m2"}
    assert out["per_family_acc"]["F9"]["n"] == 2       # family collapses both
    assert out["per_cell_acc"]["s0m4"]["acc"] == 1.0   # list form survived
    # three denominators, carried separately (§7.5/§7.6)
    assert out["coverage"]["collected"] == 2
    assert out["coverage"]["missing"] == 1
    assert out["coverage"]["stale_text"] == 0


# --------------------------------------------------------------------------
# 4. all-zero adjudication: defect or genuine difficulty
# --------------------------------------------------------------------------

def test_clean_construction_gates_make_all_zero_a_difficulty_finding():
    """§7.7's verdict, recomputed: GT reachable + instruction complete ⇒ hard.

    The cell stays disclosed in the summary either way -- what changes is
    whether it blocks the freeze.
    """
    rows = rows_all({"s0m4": (0.0, 20)})

    out = build_from_rows(rows, CLEAN_AUDIT)
    s = out["summary"]

    assert s["grid"]["all_zero"] == ["s0m4"]           # still disclosed
    assert s["all_zero_adjudication"]["verdict"] == "real_difficulty"
    assert s["ok"] is True


def test_goal_text_violation_makes_the_same_cell_blocking():
    """The §7.6 gate-4 case: an instruction that never named the destination.

    Two cells once read all-zero for exactly this reason, and repairing the
    template took them to 0.90. So a nonzero violation count has to flip the
    verdict back to construction-suspect.
    """
    dirty = {**CLEAN_AUDIT,
             "goal_text_recoverability": {"n_violations": 3, "ok": False}}

    out = build_from_rows(rows_all({"s0m4": (0.0, 20)}), dirty)
    s = out["summary"]

    assert s["all_zero_adjudication"]["verdict"] == "construction_suspect"
    assert s["ok"] is False
    assert "suspect construction" in s["blocking_reason"]


def test_unreachable_gt_makes_the_cell_blocking():
    dirty = {**CLEAN_AUDIT,
             "dual_validator": {"simenv_pass_rate": 0.98, "phi_agreement": 1.0,
                                "gate": 0.995}}

    out = build_from_rows(rows_all({"s0m4": (0.0, 20)}), dirty)

    assert out["summary"]["ok"] is False


def test_missing_audit_is_treated_as_unproven_not_as_clean():
    """No evidence is not evidence of soundness."""
    ev = construction_evidence({})
    assert ev["ok"] is False
    assert all(c["ok"] is False for c in ev["checks"].values())

    out = build_from_rows(rows_all({"s0m4": (0.0, 20)}), None)
    assert out["summary"]["ok"] is False


def test_saturation_is_never_excused_by_construction_evidence():
    """A saturated cell cannot discriminate; clean GT does not change that."""
    out = build_from_rows(rows_all({"s0m0": (1.0, 20)}), CLEAN_AUDIT)
    s = out["summary"]

    assert s["grid"]["saturated"] == ["s0m0"]
    assert s["ok"] is False
    assert "saturated" in s["blocking_reason"]


def test_undetermined_is_never_excused_by_construction_evidence():
    out = build_from_rows(rows_all({"s0m4": (0.0, 1)}), CLEAN_AUDIT)
    s = out["summary"]

    assert s["ok"] is False
    assert "NOT a pass" in s["blocking_reason"]


# --------------------------------------------------------------------------
# 5. §7-8 eval fairness
# --------------------------------------------------------------------------

def test_eval_fairness_requires_both_channels_per_model():
    rows = [make_row("a", {}, model="a", condition="image"),
            make_row("a", {}, model="a", condition="oracle_text"),
            make_row("b", {}, model="b", condition="image")]

    f = eval_fairness(rows)

    assert f["per_model"]["a"]["both_channels"] is True
    assert f["per_model"]["b"]["missing"] == ["oracle_text"]
    assert f["ok"] is False


def test_eval_fairness_flags_the_absent_official_api_row():
    """relay and official are separate rows; neither may stand in for the other."""
    rows = [make_row("a", {}, model="a", condition=c)
            for c in ("image", "oracle_text")]

    f = eval_fairness(rows)

    assert f["api_sources_present"] == ["relay"]
    assert f["official_api_rerun_present"] is False
    assert f["ok"] is True          # §7-8 itself is about channels, not sources


def test_zero_coverage_split_is_surfaced_not_scored():
    """The concrete hole: claude-fable-5 / image had ood_x = 0/40.

    Left unreported it would sink every ood_x-only cell to all-zero.
    """
    cov = {"collected": 250, "total": 560, "missing": 310, "stale_text": 0,
           "frac": 250 / 560,
           "per_split": {"test_id": {"collected": 180, "total": 180,
                                     "missing": 0, "stale_text": 0},
                         "ood_x": {"collected": 0, "total": 40,
                                   "missing": 40, "stale_text": 0}}}
    rows = rows_all({"s0m0": (0.4, 20)})
    rows[0]["coverage"] = cov

    out = build_from_rows(rows, CLEAN_AUDIT)

    assert out["zero_coverage_splits"] == {"m0__image": ["ood_x"]}
    assert "m0__image" in out["coverage_incomplete_rows"]


# --------------------------------------------------------------------------
# helper: build() takes scorer output, these tests hold summarized rows
# --------------------------------------------------------------------------

def build_from_rows(rows: list[dict], audit) -> dict:
    """Run build()'s gate arithmetic on already-summarized rows.

    ``build`` calls ``summarize_row`` first; monkeypatching that is fussier than
    it is worth, so re-enter through the same public pieces build() composes.
    """
    import audit_locus_strong_baseline as A

    orig = A.summarize_row
    A.summarize_row = lambda r: r
    try:
        return A.build(rows, audit)["strong_baseline_prerun"]
    finally:
        A.summarize_row = orig
