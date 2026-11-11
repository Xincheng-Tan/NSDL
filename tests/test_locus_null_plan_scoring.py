"""LOCUS null-plan scoring rule (docs/benchmark.md §7-7).

73/560 items in the frozen LOCUS test splits have ``time_steps == []``: the goal
already holds at the initial state, so the correct answer is *no actions*. The
official ``viki_2.acc_reward`` early-returns 0 whenever the transformed plan is
empty, which makes those items unscorable for every model. These tests pin the
LOCUS rule that fixes it and, crucially, pin that it changes **nothing** on the
other items -- a rule that drifted there would silently rewrite every VIKI and
LOCUS number we report.
"""
from __future__ import annotations

import pytest

from nsdl.benchmark_scoring import compute_locus_l2_acc
from verl.utils.reward_score import viki_2


ACTING_PLAN = [
    {"step": 1, "actions": {"R1": ["Move", "fork"]}},
    {"step": 2, "actions": {"R1": ["Reach", "fork"]}},
    {"step": 3, "actions": {"R1": ["Grasp", "fork"]}},
    {"step": 4, "actions": {"R1": ["Move", "table"]}},
    {"step": 5, "actions": {"R1": ["Place", "table"]}},
]


def _gt(*, goal_pos: str, time_steps: list) -> dict:
    """fork starts at the kitchen work area; the goal asks for ``goal_pos``."""
    return {
        "robots": {"R1": "unitree_h1"},
        "idle_robots": [],
        "init_pos": {"fork_0": ["kitchen work area"]},
        "goal_constraints": [[{"is_satisfied": True, "name": "fork", "type": "asset",
                               "status": {"pos.name": goal_pos}}]],
        "temporal_constraints": [],
        "time_steps": time_steps,
    }


@pytest.fixture()
def acting_gt() -> dict:
    return _gt(goal_pos="table", time_steps=ACTING_PLAN)


@pytest.fixture()
def null_gt() -> dict:
    # Goal is where the fork already is -> nothing to do.
    return _gt(goal_pos="kitchen work area", time_steps=[])


# -- the fix ---------------------------------------------------------------

def test_null_task_scores_empty_plan_correct(null_gt):
    assert compute_locus_l2_acc([], null_gt) == 1.0


def test_null_task_scores_acting_plan_wrong(null_gt):
    assert compute_locus_l2_acc(ACTING_PLAN, null_gt) == 0.0


def test_official_judge_cannot_score_null_tasks(null_gt):
    """The dead zone this rule exists for -- if this ever passes, drop the rule."""
    assert viki_2.acc_reward("<answer>[]</answer>", null_gt) == 0.0


# -- and it must not move anything else ------------------------------------

def test_acting_task_matches_official_judge(acting_gt):
    for plan in (ACTING_PLAN, [{"step": 1, "actions": {"R1": ["Move", "fork"]}}], []):
        official = viki_2.compute_score(
            f"<think>t</think><answer>{plan!r}</answer>", acting_gt)
        # compute_score is 0.1*format + 0.9*acc, and format is 1.0 here.
        assert compute_locus_l2_acc(plan, acting_gt) == pytest.approx(
            (official - 0.1) / 0.9), plan


def test_acting_task_correct_and_wrong_are_separated(acting_gt):
    assert compute_locus_l2_acc(ACTING_PLAN, acting_gt) == 1.0
    assert compute_locus_l2_acc([{"step": 1, "actions": {"R1": ["Move", "fork"]}}],
                                acting_gt) == 0.0
    # An empty plan is still wrong when the task genuinely requires action.
    assert compute_locus_l2_acc([], acting_gt) == 0.0
