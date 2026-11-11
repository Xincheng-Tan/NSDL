"""`selected_robots_exact_match` must be order-insensitive.

VIKI-L1's official judge scores an unordered SET of robot names. This flag used
to compare the two lists directly, so a multi-robot answer whose slot order
differed from the ground truth's was recorded as a miss even though the official
per-row score in `benchmark_scores` called it a hit. On the v3.3 run that was
47 / 2043 rows -- all of them |A|=2 -- a 2.3pt gap between two columns of the
same artifact. The disagreement is the bug worth pinning: these two columns must
never be able to diverge again.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nsdl.evaluation as ev  # noqa: E402
from nsdl.schema import BenchmarkSample  # noqa: E402

SAMPLE = BenchmarkSample(
    level="VIKI-L1", split="test", sample_index=0, task_id=None, task_name=None,
    instruction="put the bowl on the table", system_prompt="s",
    ground_truth="['fetch', 'stompy']",
)


class _Result:
    """Stand-in for `PipelineResult`; the real one would drag in a model."""

    sample = SAMPLE


def _artifact(monkeypatch, selected, gt):
    monkeypatch.setattr(ev, "to_debug_dict", lambda r: {"sample": {}})
    monkeypatch.setattr(ev, "to_l1_answer", lambda r: selected)
    monkeypatch.setattr(ev, "to_dispatcher_output", lambda r: [])
    monkeypatch.setattr(ev, "get_l1_ground_truth_robot_types", lambda s: gt)
    return ev.build_result_artifact(_Result(), validate_l2=False)


def test_same_set_different_order_is_a_hit(monkeypatch):
    art = _artifact(monkeypatch, ["stompy", "fetch"], ["fetch", "stompy"])
    assert art["selected_robots_exact_match"] is True


def test_identical_order_is_still_a_hit(monkeypatch):
    art = _artifact(monkeypatch, ["fetch", "stompy"], ["fetch", "stompy"])
    assert art["selected_robots_exact_match"] is True


def test_a_genuinely_different_set_is_still_a_miss(monkeypatch):
    art = _artifact(monkeypatch, ["stompy"], ["fetch", "stompy"])
    assert art["selected_robots_exact_match"] is False


def test_empty_answer_is_a_miss_not_a_crash(monkeypatch):
    art = _artifact(monkeypatch, [], ["fetch", "stompy"])
    assert art["selected_robots_exact_match"] is False


def test_absent_ground_truth_stays_none(monkeypatch):
    art = _artifact(monkeypatch, ["stompy"], None)
    assert art["selected_robots_exact_match"] is None


def test_it_agrees_with_the_official_per_row_score(monkeypatch):
    """The column that was already right is the reference, not the other way."""
    art = _artifact(monkeypatch, ["stompy", "fetch"], ["fetch", "stompy"])
    official = art["benchmark_scores"]["viki_l1_robot_activation"]
    assert official == 1.0
    assert art["selected_robots_exact_match"] is (official == 1.0)
