"""`build_error_artifact` must emit both score columns.

This path had no coverage, and its omission of `locus_l2_acc` is what put ~640
keyless rows into otherwise-current buckets -- indistinguishable, on disk, from
the rows written before the column existed at all. That ambiguity is the bug
worth pinning: "key absent" has to mean exactly one thing.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nsdl.evaluation import build_error_artifact  # noqa: E402
from nsdl.schema import BenchmarkSample  # noqa: E402

# No `time_steps` => a null-plan item, where an empty plan is the CORRECT answer
# and `compute_locus_l2_acc` would return 1.0.
NULL_PLAN_GT = {"description": "goal already satisfied", "robots": {"R1": "drone"}}


def _sample(ground_truth):
    return BenchmarkSample(
        level="VIKI-L2", split="val", sample_index=0, task_id="t", task_name="n",
        instruction="i", system_prompt="s", ground_truth=ground_truth,
    )


def _scores(ground_truth):
    return build_error_artifact(_sample(ground_truth), RuntimeError("boom"))["benchmark_scores"]


def test_both_columns_are_emitted_on_an_errored_row():
    scores = _scores(NULL_PLAN_GT)
    assert "locus_l2_acc" in scores, "absent key must mean 'predates the column', nothing else"
    assert "viki_l2_plan_score" in scores


def test_a_crash_on_a_null_plan_item_scores_zero_not_one():
    # The whole point. This row has no plan, and on a null-plan item "no plan"
    # is the right answer -- so anything that routes a crash through the LOCUS
    # scorer awards it 1.0, and does so exactly on the state-judgment rows.
    assert _scores(NULL_PLAN_GT)["locus_l2_acc"] == 0.0


def test_both_columns_agree_on_a_crash():
    scores = _scores({"time_steps": [{"step": 1}]})
    assert scores["locus_l2_acc"] == scores["viki_l2_plan_score"] == 0.0


def test_no_dict_ground_truth_yields_none_under_the_same_guard():
    scores = _scores(None)
    assert scores["locus_l2_acc"] is None
    assert scores["viki_l2_plan_score"] is None
