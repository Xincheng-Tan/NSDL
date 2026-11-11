"""Unit tests for the L-NSDL P5.3 scoped-RL reward (verl/.../reward_score/lnsdl.py).

Validates the reward SHAPE and the relative ordering of the documented failure
modes (docs/dataset_and_train.md §E.1.2). It does not assert absolute R_task,
which depends on the executor/eval pipeline + real GT; the synthetic GT here
exercises the dense components (R_goal / R_faithful / R_struct) and penalties.

Two recipes live here since P5.3′.1 (2026-08-03):

* everything below the ``_patch_stack_on`` fixture runs with
  ``LNSDL_REWARD_PATCH_STACK`` forced ON — that is the pre-2026-08-03 recipe,
  now an ablation path. These tests are its regression suite: the patches must
  keep behaving exactly as recorded in docs/RL_debug.md so the E5–E8 runs stay
  reproducible.
* ``TestSimplifiedRecipe`` opts back out and pins the new default: four positive
  axes, no negative term, no gate, no cap.

Run: PYTHONPATH=verl:. python -m pytest tests/test_lnsdl_reward.py -q
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(_REPO, "verl"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from verl.utils.reward_score import lnsdl  # noqa: E402


GT = {
    "gt_goal_json": {
        "active": ["plate"],
        "targets": [{"item": "bowl", "status": "skip"}, {"item": "plate", "status": "active"}],
    },
    "time_steps": [{}, {}, {}],
    "robots": {"R1": "stompy"},
    "gt_dag_json": [{"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"}],
}


@pytest.fixture(autouse=True)
def _patch_stack_on(monkeypatch):
    """Legacy recipe for every test in this module unless explicitly overridden.

    P5.3′.1 turned the patch stack off by default; these tests were written
    against it and remain valuable as the ablation path's regression suite, so
    they keep asserting the patched behaviour rather than being deleted.
    """
    monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", True)


def test_returns_dict_with_required_keys():
    out = lnsdl.compute_score("[GOAL] {\"active\":[\"plate\"]} [PLAN] []", GT)
    for k in ("score", "acc", "r_task", "r_goal", "r_faithful", "r_struct",
              "r_tool_use", "tool_use_reward", "tool_use_eligible",
              "tool_use_clean", "tool_use_consumed", "tool_use_observed",
              "p_format", "p_nonterm", "p_cost", "n_tool_calls", "stacked", "nonterm"):
        assert k in out, f"missing key {k}"
    assert isinstance(out["score"], float)


def test_correct_active_set_scores_full_r_goal():
    traj = '[GOAL] {"active":["plate"],"skip":["bowl"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
    out = lnsdl.compute_score(traj, GT)
    assert out["r_goal"] == 1.0
    assert out["r_faithful"] == 1.0  # grasp {plate} == active {plate}


def test_task_failure_dense_reward_is_downscaled():
    # A trajectory can look plausible in GOAL/PLAN consistency while still not
    # exact-match the executable GT. It must not be rewarded like a near-success.
    traj = '[GOAL] {"active":["plate"],"skip":["bowl"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
    out = lnsdl.compute_score(traj, GT)
    if out["r_task"] == 0.0:
        assert out["task_fail_dense_scale"] == 0.70
        assert out["score"] <= 0.45


def test_failure_band_scales_struct_only_keeps_r_goal_whole(monkeypatch):
    # docs/RL_debug.md §3.2: in the failure band the anti-hacking scale must hit
    # ONLY R_struct; R_goal (and R_faithful) keep full budget. Force a failure
    # (r_task=0) with all dense components = 1 and compare the dense contribution
    # against the old uniform-scale behavior.
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 0.0)
    monkeypatch.setattr(lnsdl, "_r_struct", lambda goal, plan, gt: 1.0)
    traj = '[GOAL] {"active":["plate"],"skip":["bowl"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'

    monkeypatch.setattr(lnsdl, "TASK_FAIL_SCALE_STRUCT_ONLY", True)
    out = lnsdl.compute_score(traj, GT)
    assert out["r_task"] == 0.0 and out["r_goal"] == 1.0 and out["r_struct"] == 1.0
    assert out["dense_scale_struct_only"] == 1.0
    # struct-only: W_GOAL*1 + W_FAITHFUL*1 + W_STRUCT*1*0.70 = .30 + .15 + .105
    ds = lnsdl.TASK_FAIL_DENSE_SCALE
    expected_struct_only = lnsdl.W_GOAL + lnsdl.W_FAITHFUL + lnsdl.W_STRUCT * ds
    # score is pre-cap here (0.555 > 0.45 cap) -> assert on the dense math via cap
    assert out["score"] == min(expected_struct_only, lnsdl.TASK_FAIL_SCORE_CAP)

    monkeypatch.setattr(lnsdl, "TASK_FAIL_SCALE_STRUCT_ONLY", False)
    out_uniform = lnsdl.compute_score(traj, GT)
    # uniform: (W_GOAL+W_FAITHFUL+W_STRUCT)*0.70 = .42 -> R_goal squeezed to .21
    expected_uniform = (lnsdl.W_GOAL + lnsdl.W_FAITHFUL + lnsdl.W_STRUCT) * ds
    assert abs(out_uniform["score"] - expected_uniform) < 1e-9
    # struct-only gives R_goal strictly more budget than uniform in the failure band
    assert expected_struct_only > expected_uniform


def test_shortcut_wrong_item_zeros_r_goal():
    traj = '[GOAL] {"active":["bowl"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"bowl","robot":"R1"}]'
    out = lnsdl.compute_score(traj, GT)
    assert out["r_goal"] == 0.0  # active {bowl} disjoint from GT {plate}


def test_unfaithful_plan_partial_r_faithful():
    # goal corrected to {plate} but PLAN grasps bowl too -> symdiff 1 / union 2
    traj = '[GOAL] {"active":["plate"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"},{"op_id":"o2","op_type":"acquire","entity":"bowl","robot":"R1"}]'
    out = lnsdl.compute_score(traj, GT)
    assert abs(out["r_faithful"] - 0.5) < 1e-9


def test_last_goal_wins_across_turns():
    # first GOAL wrong (active bowl+plate), corrected GOAL right (plate) -> use last
    traj = (
        '[GOAL] {"active":["bowl","plate"]} '
        '[TOOL_CALL] {"name":"verify_goal_coverage","args":{}} '
        '[GOAL] {"active":["plate"],"skip":["bowl"]} '
        '[PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
    )
    out = lnsdl.compute_score(traj, GT)
    assert out["r_goal"] == 1.0


def test_stacked_and_nonterm_penalized_via_trace():
    traj = '[TOOL_CALL] {"name":"detect","args":{}} [TOOL_CALL] {"name":"verify","args":{}}'
    ei = {"trace": {"emitted_tool_call_count": 6, "executed_tool_call_count": 3,
                    "stopped_reason": "max_tool_rounds", "stacked_tool_call_turns": 2}}
    out = lnsdl.compute_score(traj, GT, ei)
    assert out["p_format"] == -0.40        # capped (stacked + mismatch + ...)
    assert out["p_nonterm"] == -0.30
    assert out["p_cost"] == -0.12          # 6 * -0.02
    assert out["nonterm"] == 1.0
    assert out["score"] < 0.0              # strongly negative -> kills P5.2.4 mode


def test_emitted_without_execution_triggers_format_penalty():
    traj = '[TOOL_CALL] {"name":"detect","args":{}}'
    out = lnsdl.compute_score(
        traj,
        GT,
        {"trace": {"emitted_tool_call_count": 1, "executed_tool_call_count": 0,
                   "stopped_reason": "max_tool_rounds", "stacked_tool_call_turns": 0}},
    )
    assert out["p_format"] == -0.20


def test_tool_cost_nudges_against_needless_tools():
    # Backward-compat: GT has no trivial_structure flag -> cost is always charged.
    base = '[GOAL] {"active":["plate"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
    no_tool = lnsdl.compute_score(base, GT)
    three_tools = lnsdl.compute_score(
        '[TOOL_CALL] {"a":1} x [TOOL_CALL] y [TOOL_CALL] z ' + base, GT,
        {"trace": {"emitted_tool_call_count": 3, "executed_tool_call_count": 3,
                   "stopped_reason": "final_plan"}},
    )
    assert three_tools["p_cost"] < no_tool["p_cost"]  # more tools -> larger penalty


# --- T1: conditional P_cost (docs/RL_debug.md §8) -------------------------
_THREE_TOOLS = '[TOOL_CALL] {"a":1} x [TOOL_CALL] y [TOOL_CALL] z '
_BASE = '[GOAL] {"active":["plate"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
_TRACE3 = {"trace": {"emitted_tool_call_count": 3, "executed_tool_call_count": 3,
                     "stopped_reason": "final_plan"}}


def test_cond_pcost_not_charged_on_failed_sample():
    # trivial_structure=True but the synthetic GT is not exact-matched (r_task=0)
    # -> no cost gradient (tools are not "wasteful" on a sample we did not solve).
    gt = dict(GT); gt["trivial_structure"] = True
    out = lnsdl.compute_score(_THREE_TOOLS + _BASE, gt, _TRACE3)
    assert out["r_task"] == 0.0
    assert out["p_cost"] == 0.0
    assert out["cost_charged"] == 0.0


def test_cond_pcost_charges_only_trivial_solved(monkeypatch):
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)  # force solved
    gt_triv = dict(GT); gt_triv["trivial_structure"] = True
    out_triv = lnsdl.compute_score(_THREE_TOOLS + _BASE, gt_triv, _TRACE3)
    assert out_triv["p_cost"] < 0.0            # trivial + solved -> wasteful tools taxed
    assert out_triv["cost_charged"] == 1.0

    gt_non = dict(GT); gt_non["trivial_structure"] = False
    out_non = lnsdl.compute_score(_THREE_TOOLS + _BASE, gt_non, _TRACE3)
    assert out_non["p_cost"] == 0.0            # non-trivial -> tools are the means, no tax
    assert out_non["cost_charged"] == 0.0


def test_cond_pcost_does_not_charge_skip_decision_samples(monkeypatch):
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)  # force solved
    gt = dict(GT)
    gt["trivial_structure"] = True
    gt["has_skip_decision"] = True
    out = lnsdl.compute_score(_THREE_TOOLS + _BASE, gt, _TRACE3)
    assert out["p_cost"] == 0.0
    assert out["cost_charged"] == 0.0
    assert out["skip_sample"] == 1.0


def test_skip_sample_flag_surfaced():
    gt = dict(GT); gt["has_skip_decision"] = True
    out = lnsdl.compute_score(_BASE, gt)
    assert out["skip_sample"] == 1.0
    # absent flag -> -1.0 sentinel (old parquet)
    assert lnsdl.compute_score(_BASE, GT)["skip_sample"] == -1.0


# --- E3/T4: minimal positive R_tool_use shaping --------------------------
_GOAL_TABLE = '{"active":["plate"],"targets":[{"item":"plate","destination":"table","status":"active"}]}'
_PLAN_TABLE = (
    '[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"},'
    '{"op_id":"o2","op_type":"deposit","entity":"plate","dest":"table","robot":"R1"}]'
)
_TABLE_TRAJ = f'[GOAL] {_GOAL_TABLE} [PLAN] {_PLAN_TABLE}'


def test_tool_use_rewards_consumed_detect_observation(monkeypatch):
    monkeypatch.setattr(lnsdl, "R_TOOL_USE", 0.05)
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
    trace = {
        "trace": {
            "emitted_tool_call_count": 1,
            "executed_tool_call_count": 1,
            "stopped_reason": "final_plan",
            "stacked_tool_call_turns": 0,
            "tool_observations": [{
                "name": "detect",
                "entities": ["plate", "table"],
                "relations": ["plate on table"],
            }],
        }
    }
    out = lnsdl.compute_score('[TOOL_CALL] {"name":"detect","args":{}} ' + _TABLE_TRAJ, GT, trace)
    assert out["r_tool_use"] == 1.0
    assert abs(out["tool_use_reward"] - 0.05) < 1e-9
    assert out["tool_use_eligible"] == 1.0
    assert out["tool_use_clean"] == 1.0
    assert out["tool_use_consumed"] == 1.0
    assert out["tool_use_observed"] == 1.0


def test_tool_use_does_not_reward_unconsumed_observation(monkeypatch):
    monkeypatch.setattr(lnsdl, "R_TOOL_USE", 0.05)
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
    trace = {
        "trace": {
            "emitted_tool_call_count": 1,
            "executed_tool_call_count": 1,
            "stopped_reason": "final_plan",
            "stacked_tool_call_turns": 0,
            "tool_observations": [{
                "name": "detect",
                "entities": ["cup", "sink"],
                "relations": ["cup in sink"],
            }],
        }
    }
    out = lnsdl.compute_score('[TOOL_CALL] {"name":"detect","args":{}} ' + _TABLE_TRAJ, GT, trace)
    assert out["r_tool_use"] == 0.0
    assert out["tool_use_reward"] == 0.0
    assert out["tool_use_consumed"] == 0.0


def test_tool_use_rewards_verify_op_id_consumed_by_plan(monkeypatch):
    monkeypatch.setattr(lnsdl, "R_TOOL_USE", 0.05)
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
    trace = {
        "trace": {
            "emitted_tool_call_count": 1,
            "executed_tool_call_count": 1,
            "stopped_reason": "final_plan",
            "stacked_tool_call_turns": 0,
            "tool_observations": [{"name": "verify", "ok": False, "code": "missing_dep", "op_ids": ["o2"]}],
        }
    }
    out = lnsdl.compute_score('[TOOL_CALL] {"name":"verify","args":{}} ' + _TABLE_TRAJ, GT, trace)
    assert out["r_tool_use"] == 1.0
    assert out["tool_use_consumed"] == 1.0


def test_tool_use_requires_clean_protocol_even_when_consumed(monkeypatch):
    monkeypatch.setattr(lnsdl, "R_TOOL_USE", 0.05)
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
    trace = {
        "trace": {
            "emitted_tool_call_count": 2,
            "executed_tool_call_count": 1,
            "stopped_reason": "final_plan",
            "stacked_tool_call_turns": 1,
            "tool_observations": [{
                "name": "detect",
                "entities": ["plate", "table"],
                "relations": ["plate on table"],
            }],
        }
    }
    out = lnsdl.compute_score(
        '[TOOL_CALL] {"name":"detect","args":{}} [TOOL_CALL] {"name":"verify","args":{}} '
        + _TABLE_TRAJ,
        GT,
        trace,
    )
    assert out["tool_use_eligible"] == 1.0
    assert out["tool_use_clean"] == 0.0
    assert out["r_tool_use"] == 0.0
    assert out["tool_use_reward"] == 0.0


# --- E6: stack gate on task credit (docs/RL_debug.md §3(3)) --------------
# A "stacked-but-recovered" winner (the P5.6 inference-side suppress patch
# salvages the first [TOOL_CALL], so R_task=1) must NOT out-score a clean
# winner. The gate drops W_TASK from `score` while keeping the TRUE reported
# acc/r_task, so the headline plan_acc + acc-variance dynamic sampling are
# untouched and only the group-relative advantage is re-ordered.
_CLEAN_WIN_TRACE = {"trace": {"emitted_tool_call_count": 1, "executed_tool_call_count": 1,
                              "stopped_reason": "final_plan", "stacked_tool_call_turns": 0}}
_STACKED_WIN_TRACE = {"trace": {"emitted_tool_call_count": 2, "executed_tool_call_count": 1,
                                "stopped_reason": "final_plan", "stacked_tool_call_turns": 1}}
# emitted != executed with NO stacked turn (pure suppress-recovery signature).
_MISMATCH_WIN_TRACE = {"trace": {"emitted_tool_call_count": 2, "executed_tool_call_count": 1,
                                 "stopped_reason": "final_plan", "stacked_tool_call_turns": 0}}
# Every protocol fault at once: stacked turn, emitted != executed, never emitted
# a final [PLAN], and six tool calls. Under the legacy stack this is the reward
# floor (-0.40 format, -0.30 nonterm, -0.12 cost); under P5.3′.1 it must cost 0.
_DIRTY_TRACE = {"trace": {"emitted_tool_call_count": 6, "executed_tool_call_count": 3,
                          "stopped_reason": "max_tool_rounds", "stacked_tool_call_turns": 2}}


def test_stack_gate_drops_task_credit_but_keeps_true_acc(monkeypatch):
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)  # force solved
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", True)
    out = lnsdl.compute_score(_TABLE_TRAJ, GT, _STACKED_WIN_TRACE)
    # reported task-success stays TRUE (headline + acc-variance sampling intact)
    assert out["acc"] == 1.0 and out["r_task"] == 1.0
    assert out["task_gated"] == 1.0
    # the W_TASK credit is gone from score: turning the gate off adds back exactly W_TASK
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", False)
    out_off = lnsdl.compute_score(_TABLE_TRAJ, GT, _STACKED_WIN_TRACE)
    assert out_off["task_gated"] == 0.0
    assert abs((out_off["score"] - out["score"]) - lnsdl.W_TASK) < 1e-9


def test_stack_gate_orders_clean_winner_above_stacked_winner(monkeypatch):
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)  # both solved the task
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", True)
    clean = lnsdl.compute_score(_TABLE_TRAJ, GT, _CLEAN_WIN_TRACE)
    stacked = lnsdl.compute_score(_TABLE_TRAJ, GT, _STACKED_WIN_TRACE)
    assert clean["task_gated"] == 0.0 and stacked["task_gated"] == 1.0
    # clean gets +W_TASK plus a smaller format/cost penalty -> gap is at least W_TASK.
    # This is the whole point: stacking can no longer win.
    assert clean["score"] - stacked["score"] >= lnsdl.W_TASK - 1e-9


def test_stack_gate_fires_on_emitted_executed_mismatch_without_stacked_turn(monkeypatch):
    # The suppress-recovery signature is emitted>executed even when the harness
    # counts 0 stacked turns; the gate must still fire.
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", True)
    out = lnsdl.compute_score(_TABLE_TRAJ, GT, _MISMATCH_WIN_TRACE)
    assert out["stacked"] == 0.0 and out["task_gated"] == 1.0
    assert out["acc"] == 1.0


def test_stack_gate_leaves_clean_winner_untouched(monkeypatch):
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", True)
    clean = lnsdl.compute_score(_TABLE_TRAJ, GT, _CLEAN_WIN_TRACE)
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", False)
    clean_off = lnsdl.compute_score(_TABLE_TRAJ, GT, _CLEAN_WIN_TRACE)
    assert clean["task_gated"] == 0.0
    assert abs(clean["score"] - clean_off["score"]) < 1e-9  # gate is a no-op on clean protocol


def test_stack_gate_preserves_semantic_gradient(monkeypatch):
    # The gate removes ONLY the task term, so the R_goal (取件) gradient must stay
    # alive across gated trajectories: a stacked trajectory with the correct
    # active set still scores strictly above a stacked one with the wrong item.
    # (Otherwise the gate would flatten all stacked rollouts and kill learning.)
    monkeypatch.setattr(lnsdl, "STACK_GATE_TASK", True)
    monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)  # hold task fixed
    right = lnsdl.compute_score(
        '[GOAL] {"active":["plate"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]',
        GT, _STACKED_WIN_TRACE)
    wrong = lnsdl.compute_score(
        '[GOAL] {"active":["bowl"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"bowl","robot":"R1"}]',
        GT, _STACKED_WIN_TRACE)
    assert right["task_gated"] == 1.0 and wrong["task_gated"] == 1.0
    assert right["r_goal"] == 1.0 and wrong["r_goal"] == 0.0
    assert right["score"] > wrong["score"]


class TestSimplifiedRecipe:
    """P5.3′.1 default: four positive axes, no negative term (docs/plan.md)."""

    @pytest.fixture(autouse=True)
    def _patch_stack_off(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)

    def test_default_is_simplified(self):
        # The module default -- not just what the fixture forces -- must be off.
        assert lnsdl.REWARD_PATCH_STACK is False
        out = lnsdl.compute_score(_TABLE_TRAJ, GT)
        assert out["reward_patch_stack"] == 0.0

    def test_score_is_exactly_the_weighted_positive_sum(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 0.0)
        monkeypatch.setattr(lnsdl, "_r_struct", lambda goal, plan, gt: 1.0)
        out = lnsdl.compute_score(_TABLE_TRAJ, GT, _STACKED_WIN_TRACE)
        expected = (lnsdl.W_TASK * out["r_task"]
                    + lnsdl.W_GOAL * out["r_goal"]
                    + lnsdl.W_FAITHFUL * out["r_faithful"]
                    + lnsdl.W_STRUCT * out["r_struct"])
        assert abs(out["score"] - expected) < 1e-12

    def test_no_negative_term_survives_the_worst_protocol(self):
        # Stacked turns, emitted != executed, non-termination and six tool calls
        # at once: under the old recipe this floor was -0.40-0.30-0.12; now the
        # protocol cannot subtract anything at all.
        out = lnsdl.compute_score(_TABLE_TRAJ, GT, _DIRTY_TRACE)
        assert out["p_format"] == 0.0 and out["p_nonterm"] == 0.0 and out["p_cost"] == 0.0
        assert out["tool_use_reward"] == 0.0
        assert out["score"] >= 0.0

    def test_protocol_observations_are_still_reported(self):
        # Deleting the price must not delete the diagnostic: `nonterm` in
        # particular feeds the CFA axis-A guard, which replaces P_nonterm.
        out = lnsdl.compute_score(_TABLE_TRAJ, GT, _DIRTY_TRACE)
        assert out["nonterm"] == 1.0
        assert out["stacked"] == 1.0
        assert out["n_tool_calls"] > 0

    def test_failure_band_is_neither_scaled_nor_capped(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 0.0)
        monkeypatch.setattr(lnsdl, "_r_struct", lambda goal, plan, gt: 1.0)
        out = lnsdl.compute_score(_TABLE_TRAJ, GT)
        assert out["task_fail_dense_scale"] == 1.0
        assert out["r_task"] == 0.0
        # W_GOAL + W_FAITHFUL + W_STRUCT = 0.60 > the old 0.45 cap.
        assert out["score"] > lnsdl.TASK_FAIL_SCORE_CAP

    def test_stack_gate_does_not_fire(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
        stacked = lnsdl.compute_score(_TABLE_TRAJ, GT, _STACKED_WIN_TRACE)
        clean = lnsdl.compute_score(_TABLE_TRAJ, GT, _CLEAN_WIN_TRACE)
        assert stacked["task_gated"] == 0.0
        # Protocol hygiene moved to the rollout-side truncation guardrail, so the
        # reward must be blind to it -- same axes in, same score out.
        assert abs(stacked["score"] - clean["score"]) < 1e-12

    def test_goal_gradient_survives(self):
        right = lnsdl.compute_score(
            '[GOAL] {"active":["plate"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]',
            GT)
        wrong = lnsdl.compute_score(
            '[GOAL] {"active":["bowl"]} [PLAN] [{"op_id":"o1","op_type":"acquire","entity":"bowl","robot":"R1"}]',
            GT)
        assert right["r_goal"] == 1.0 and wrong["r_goal"] == 0.0
        assert right["score"] > wrong["score"]

    def test_per_axis_fields_present_for_cfa(self):
        # lnsdl_advantage.assemble_axis_rewards reads exactly these off
        # non_tensor_batch; losing one silently zeroes an axis.
        out = lnsdl.compute_score(_TABLE_TRAJ, GT)
        for k in ("r_goal", "r_faithful", "r_struct", "r_task", "nonterm"):
            assert k in out

    def test_patch_stack_env_switch_restores_legacy(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "_r_task", lambda plan, gt: 1.0)
        simplified = lnsdl.compute_score(_TABLE_TRAJ, GT, _DIRTY_TRACE)
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", True)
        legacy = lnsdl.compute_score(_TABLE_TRAJ, GT, _DIRTY_TRACE)
        assert legacy["reward_patch_stack"] == 1.0
        assert legacy["p_format"] < 0.0 and legacy["p_nonterm"] < 0.0
        assert legacy["score"] < simplified["score"]


class TestNullPlanBranch:
    """LOCUS semantics for prompts whose ground truth prescribes no steps.

    136/900 ``ramp63k`` prompts have empty ``time_steps`` -- the correct answer
    is to emit nothing. Before this branch ``_r_task`` was wrong at *both* ends:
    ``not plan`` returned 0 for the right answer, and the ``len_gt == 0``
    shortcut returned 1.0 for any plan that survived ``_simulate``.
    """

    NULL_GT = dict(GT, time_steps=[])

    def test_empty_plan_is_correct_when_the_gt_prescribes_no_steps(self):
        assert lnsdl._r_task([], self.NULL_GT) == 1.0
        assert lnsdl._r_task(None, self.NULL_GT) == 1.0

    def test_any_plan_is_wrong_when_the_gt_prescribes_no_steps(self):
        plan = [{"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"}]
        assert lnsdl._r_task(plan, self.NULL_GT) == 0.0

    def test_missing_and_none_time_steps_take_the_same_branch(self):
        for gt in ({k: v for k, v in GT.items() if k != "time_steps"},
                   dict(GT, time_steps=None)):
            assert lnsdl._r_task([], gt) == 1.0
            assert lnsdl._r_task([{"op_id": "o1", "op_type": "acquire",
                                   "entity": "plate", "robot": "R1"}], gt) == 0.0

    def test_non_null_gt_is_untouched(self):
        """The branch must not perturb the 84.9% of prompts that have steps."""
        assert lnsdl._r_task(None, GT) == 0.0
        assert lnsdl._r_task([], GT) == 0.0

    def test_acc_follows_r_task_through_compute_score(self):
        out = lnsdl.compute_score('[GOAL] {"active":[],"skip":["bowl","plate"]} [PLAN] []',
                                  self.NULL_GT)
        assert out["r_task"] == 1.0
        assert out["acc"] == 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# 2026-08-31 LOCUS R_struct judge upgrade (debug_sft.md section 4.5.1 follow-up)
# LOCUS rows (gt carries locus_meta) additionally require the plan to survive
# the scoring chain's execution path; VIKI rows keep the validator-only judge.


LOCUS_GT = {
    "locus_meta": {"cell": [0, 1]},
    "description": "put the plate on the kitchen island area",
    "robots": {"R1": "stompy"},
    "time_steps": [{}, {}],
}

VALID_LOCUS_PLAN = [
    {"op_id": "op1", "op_type": "acquire", "entity": "plate", "dest": None,
     "deps": [], "robot": "R1"},
    {"op_id": "op2", "op_type": "deposit", "entity": "plate",
     "dest": "kitchen island area", "deps": ["op1"], "robot": "R1"},
]


def test_locus_r_struct_zeroes_when_expansion_fails(monkeypatch):
    goal = {"active": ["plate"]}
    assert lnsdl._r_struct(goal, VALID_LOCUS_PLAN, LOCUS_GT) == 1.0
    monkeypatch.setattr(lnsdl._eg, "plan_expansion_ok", lambda plan, gt: False)
    assert lnsdl._r_struct(goal, VALID_LOCUS_PLAN, LOCUS_GT) == 0.0


def test_viki_r_struct_never_consults_expansion(monkeypatch):
    def _boom(plan, gt):  # pragma: no cover - the assertion IS the test
        raise AssertionError("plan_expansion_ok must not run on VIKI rows")

    monkeypatch.setattr(lnsdl._eg, "plan_expansion_ok", _boom)
    viki_gt = {k: v for k, v in LOCUS_GT.items() if k != "locus_meta"}
    goal = {"active": ["plate"]}
    assert lnsdl._r_struct(goal, VALID_LOCUS_PLAN, viki_gt) == 1.0


def test_plan_expansion_ok_accepts_executable_plan():
    from verl.utils.reward_score import enactgraph as eg

    assert eg.plan_expansion_ok(VALID_LOCUS_PLAN, LOCUS_GT) is True
    assert eg.plan_expansion_ok([], LOCUS_GT) is False


def test_plan_expansion_ok_rejects_missing_open_dep():
    from verl.utils.reward_score import enactgraph as eg

    plan = [
        {"op_id": "op1", "op_type": "acquire", "entity": "plate", "dest": None,
         "deps": [], "robot": "R1"},
        {"op_id": "op2", "op_type": "deposit", "entity": "plate",
         "dest": "cabinet", "deps": ["op1"], "robot": "R1"},
    ]
    assert eg.plan_expansion_ok(plan, LOCUS_GT) is False


class TestNullPlanDeadzone:
    """FIX-1 (debug_rl §6.1): axis B must agree with axis T on empty plans."""

    def test_empty_gt_empty_plan_scores_struct_one(self):
        gt = {"time_steps": []}
        assert lnsdl._r_task(None, gt) == 1.0
        assert lnsdl._r_struct(None, None, gt) == 1.0
        assert lnsdl._r_struct({"active": []}, [], gt) == 1.0

    def test_nonempty_gt_empty_plan_scores_struct_zero(self):
        assert lnsdl._r_task(None, GT) == 0.0
        assert lnsdl._r_struct(None, None, GT) == 0.0
        assert lnsdl._r_struct({"active": ["plate"]}, [], GT) == 0.0

    def test_viki_gt_never_has_empty_time_steps_so_branch_is_noop(self):
        # A VIKI-shaped GT always carries steps; the new branch cannot fire.
        gt = dict(GT, time_steps=[{"actions": {"R1": ["Move", "plate"]}}])
        assert lnsdl._r_struct(None, [], gt) == 0.0


class TestLocusGateDetection:
    """FIX-2 (debug_rl §6.1(c)): the RL parquet carries locus_cell/locus_split,
    never locus_meta; the plan-expansion gate must fire on either shape."""

    def test_bench_json_shape(self):
        assert lnsdl._is_locus_gt({"locus_meta": {"cell": "F4_s2m3"}}) is True

    def test_rl_parquet_shape(self):
        assert lnsdl._is_locus_gt({"locus_cell": "F4_s2m3", "locus_split": "train"}) is True
        assert lnsdl._is_locus_gt({"locus_split": "train"}) is True

    def test_viki_shape_is_not_locus(self):
        assert lnsdl._is_locus_gt(GT) is False
        assert lnsdl._is_locus_gt({"locus_meta": None, "locus_cell": None}) is False
        assert lnsdl._is_locus_gt(None) is False

    def test_gate_calls_plan_expansion_on_rl_parquet_rows(self, monkeypatch):
        calls = []
        monkeypatch.setattr(lnsdl._eg, "validator_pass_reward", lambda s, gt: 1.0)
        monkeypatch.setattr(lnsdl._eg, "plan_expansion_ok", lambda plan, gt: calls.append(1) or False)
        plan = [{"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"}]
        assert lnsdl._r_struct(None, plan, dict(GT, locus_cell="F4_s2m3")) == 0.0
        assert calls == [1]
        # VIKI row: validator-only, gate untouched
        assert lnsdl._r_struct(None, plan, GT) == 1.0
        assert calls == [1]


class TestGoalPlanCoverage:
    """r_goal_plan = Jaccard(plan grasp set, GT active); default weight 0."""

    def _score(self, monkeypatch, text, **env):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        for k, v in env.items():
            monkeypatch.setattr(lnsdl, k, v)
        return lnsdl.compute_score(text, GT)

    def test_field_present_and_default_weight_zero(self, monkeypatch):
        plan_right = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
        plan_empty = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[]'
        a = self._score(monkeypatch, plan_right)
        b = self._score(monkeypatch, plan_empty)
        assert a["r_goal_plan"] == 1.0 and b["r_goal_plan"] == 0.0
        # weight 0 -> score does not move with r_goal_plan
        assert a["r_goal"] == b["r_goal"] == 1.0
        assert abs((a["score"] - b["score"]) - (lnsdl.W_TASK * (a["r_task"] - b["r_task"])
                                                + lnsdl.W_FAITHFUL * (a["r_faithful"] - b["r_faithful"])
                                                + lnsdl.W_STRUCT * (a["r_struct"] - b["r_struct"]))) < 1e-9

    def test_wrong_declaration_right_plan_keeps_full_goal_plan(self, monkeypatch):
        # declared bowl+plate (bowl is a GT skip), but the plan only grasps plate
        text = '[GOAL]\n{"active": ["bowl", "plate"]}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
        r = self._score(monkeypatch, text, W_GOAL_PLAN=0.15)
        assert r["r_goal"] == 0.5          # declaration penalised
        assert r["r_goal_plan"] == 1.0     # execution is right
        assert r["r_faithful"] == 0.5      # consistency term punishes the right plan

    def test_consistent_wrong_goal_gets_no_goal_plan_credit(self, monkeypatch):
        text = '[GOAL]\n{"active": ["bowl"]}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"bowl","robot":"R1"}]'
        r = self._score(monkeypatch, text, W_GOAL_PLAN=0.15)
        assert r["r_faithful"] == 1.0      # self-consistent
        assert r["r_goal"] == 0.0 and r["r_goal_plan"] == 0.0


class TestStructPhi:
    """r_struct_phi: final [PLAN] must be Phi's accepted op multiset."""
    PHI_OPS = [
        {"op_id": "auto_open_cabinet", "op_type": "open", "dest": "cabinet", "robot": "R1"},
        {"op_id": "op1", "op_type": "acquire", "entity": "wine", "robot": "R1"},
        {"op_id": "op2", "op_type": "deposit", "entity": "wine", "dest": "cabinet", "robot": "R1", "deps": ["auto_open_cabinet", "op1"]},
    ]

    def _trace(self, ok=True, ops=None):
        return {"tool_observations": [{"name": "detect", "entities": ["wine"]},
                                      {"name": "schedule", "ok": ok, "ops": self.PHI_OPS if ops is None else ops}]}

    def test_faithful_transcription_scores_one(self):
        plan = [dict(o) for o in self.PHI_OPS]
        assert lnsdl._r_struct_phi(plan, GT, self._trace()) == 1.0

    def test_renumbered_ids_and_reordered_ops_still_score_one(self):
        plan = [{"op_id": "b", "op_type": "acquire", "entity": "wine", "robot": "R1"},
                {"op_id": "a", "op_type": "open", "dest": "cabinet", "robot": "R1"},
                {"op_id": "c", "op_type": "deposit", "entity": "wine", "dest": "cabinet", "robot": "R1", "deps": ["a", "b"]}]
        assert lnsdl._r_struct_phi(plan, GT, self._trace()) == 1.0

    def test_dropped_phi_op_scores_zero(self):
        plan = [dict(o) for o in self.PHI_OPS if o["op_id"] != "op1"]   # the s30/G1 failure signature
        assert lnsdl._r_struct_phi(plan, GT, self._trace()) == 0.0

    def test_rule_validator_is_blind_to_a_legal_drop_but_phi_axis_is_not(self, monkeypatch):
        # Dropping the auto_open leaves a plan the rule validator may still
        # accept (custody intact); the Phi axis rejects it regardless.
        plan = [dict(o) for o in self.PHI_OPS if o["op_id"] != "auto_open_cabinet"]
        monkeypatch.setattr(lnsdl._eg, "validator_pass_reward", lambda s, gt: 1.0)
        assert lnsdl._r_struct(None, plan, GT) == 1.0
        assert lnsdl._r_struct_phi(plan, GT, self._trace()) == 0.0

    def test_self_produced_structure_without_phi_scores_zero(self):
        plan = [dict(o) for o in self.PHI_OPS]
        assert lnsdl._r_struct_phi(plan, GT, {"tool_observations": [{"name": "detect"}]}) == 0.0
        assert lnsdl._r_struct_phi(plan, GT, self._trace(ok=False)) == 0.0

    def test_null_gt_empty_plan_is_one_with_or_without_phi(self):
        assert lnsdl._r_struct_phi([], {"time_steps": []}, None) == 1.0
        assert lnsdl._r_struct_phi([], {"time_steps": []}, self._trace(ops=[])) == 1.0
        assert lnsdl._r_struct_phi([], GT, None) == 0.0

    def test_eval_harness_trace_shape_is_read(self):
        trace = {"tool_names": ["detect", "schedule"], "tool_results": [{"entities": []}, {"ok": True, "ops": self.PHI_OPS}]}
        assert lnsdl._r_struct_phi([dict(o) for o in self.PHI_OPS], GT, trace) == 1.0

    def test_compute_score_exposes_field_and_default_weight_zero(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        text = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
        tr = {"tool_observations": [{"name": "schedule", "ok": True, "ops": [{"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"}]}]}
        a = lnsdl.compute_score(text, GT, {"trace": tr})
        b = lnsdl.compute_score(text, GT, {"trace": {"tool_observations": [{"name": "schedule", "ok": True, "ops": []}]}})
        assert a["r_struct_phi"] == 1.0 and b["r_struct_phi"] == 0.0
        assert a["score"] == b["score"]
        monkeypatch.setattr(lnsdl, "W_STRUCT_PHI", 0.15)
        a2 = lnsdl.compute_score(text, GT, {"trace": tr})
        assert abs(a2["score"] - a["score"] - 0.15) < 1e-9


class TestStructSketch:
    """r_struct_sketch (plan.md A0p / facts TR5b): Phi accepted the LAST sketch."""
    OPS = [{"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"}]

    def _trace(self, *oks):
        obs = [{"name": "detect", "entities": ["plate"]}]
        obs += [{"name": "schedule", "ok": ok, "ops": self.OPS if ok else []} for ok in oks]
        return {"tool_observations": obs}

    def test_accepted_last_call_scores_one_rejected_scores_zero(self):
        assert lnsdl._r_struct_sketch(self.OPS, GT, self._trace(True)) == 1.0
        assert lnsdl._r_struct_sketch(self.OPS, GT, self._trace(False)) == 0.0

    def test_only_the_last_call_counts(self):
        # rejected -> repaired -> accepted is the intended repair loop: pays 1.
        assert lnsdl._r_struct_sketch(self.OPS, GT, self._trace(False, True)) == 1.0
        # accepted earlier, then a later sketch rejected: the submitted plan
        # comes from the rejected one -> pays 0 (r_struct_phi would still see
        # the earlier accepted ops).
        assert lnsdl._r_struct_sketch(self.OPS, GT, self._trace(True, False)) == 0.0
        assert lnsdl._r_struct_phi(self.OPS, GT, self._trace(True, False)) == 1.0

    def test_never_calling_schedule_scores_zero_except_null_gt_empty_plan(self):
        assert lnsdl._r_struct_sketch(self.OPS, GT, {"tool_observations": [{"name": "detect"}]}) == 0.0
        assert lnsdl._r_struct_sketch(self.OPS, GT, None) == 0.0
        assert lnsdl._r_struct_sketch([], {"time_steps": []}, None) == 1.0
        assert lnsdl._r_struct_sketch([], GT, None) == 0.0

    def test_eval_harness_trace_shape_is_read(self):
        trace = {"tool_names": ["schedule", "schedule"],
                 "tool_results": [{"ok": True, "ops": self.OPS}, {"ok": False, "code": "E_DEP"}]}
        assert lnsdl._r_struct_sketch(self.OPS, GT, trace) == 0.0

    def test_compute_score_field_default_weight_zero_and_a0p_recipe(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        text = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
        a = lnsdl.compute_score(text, GT, {"trace": self._trace(True)})
        b = lnsdl.compute_score(text, GT, {"trace": self._trace(False)})
        assert a["r_struct_sketch"] == 1.0 and b["r_struct_sketch"] == 0.0
        assert a["score"] == b["score"]                      # default weight 0: recorded runs unchanged
        assert a["r_task_phi"] == a["r_task"] and b["r_task_phi"] == -1.0
        for k, v in dict(W_TASK=0.40, W_GOAL=0.30, W_GOAL_PLAN=0.15, W_STRUCT_SKETCH=0.15,
                         W_FAITHFUL=0.0, W_STRUCT=0.0, W_STRUCT_PHI=0.0).items():
            monkeypatch.setattr(lnsdl, k, v)
        a2 = lnsdl.compute_score(text, GT, {"trace": self._trace(True)})
        b2 = lnsdl.compute_score(text, GT, {"trace": self._trace(False)})
        assert abs(a2["score"] - b2["score"] - 0.15) < 1e-9
        # transcription fidelity no longer moves the score under the A0p recipe
        dropped = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[]'
        c2 = lnsdl.compute_score(dropped, GT, {"trace": self._trace(True)})
        assert c2["r_struct_phi"] == 0.0 and c2["r_struct_sketch"] == 1.0


class TestP1SketchExecAndProtocolPrices:
    """debug_rl.md §7.8 P1 (2026-09-07): three gated fixes, default off."""
    OPS = [{"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"}]
    NULL_GT = {"time_steps": [], "robots": {"R1": "stompy"}}
    TEXT = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'

    def _trace(self, *calls):
        """calls: (ok, ops) tuples in order."""
        obs = [{"name": "detect", "entities": ["plate"]}]
        for ok, ops in calls:
            o = {"name": "schedule", "ok": ok}
            if ops:
                o["ops"] = ops
            obs.append(o)
        return {"tool_observations": obs}

    # --- STRUCT_SKETCH_EXEC ---------------------------------------------
    def test_default_off_keeps_empty_accepted_sketch_paying_one(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "STRUCT_SKETCH_EXEC", False)
        assert lnsdl._r_struct_sketch([], GT, self._trace((True, []))) == 1.0

    def test_exec_empty_accepted_sketch_pays_only_on_null_gt(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "STRUCT_SKETCH_EXEC", True)
        assert lnsdl._r_struct_sketch([], GT, self._trace((True, []))) == 0.0
        assert lnsdl._r_struct_sketch([], self.NULL_GT, self._trace((True, []))) == 1.0
        # rejected last call still 0; never called unchanged
        assert lnsdl._r_struct_sketch(self.OPS, GT, self._trace((True, self.OPS), (False, []))) == 0.0
        assert lnsdl._r_struct_sketch([], self.NULL_GT, None) == 1.0

    def test_exec_locus_row_requires_expansion(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "STRUCT_SKETCH_EXEC", True)
        calls = []
        monkeypatch.setattr(lnsdl._eg, "plan_expansion_ok", lambda plan, gt: (calls.append(plan), False)[1])
        assert lnsdl._r_struct_sketch(self.OPS, LOCUS_GT, self._trace((True, self.OPS))) == 0.0
        assert calls and calls[0][0]["op_id"] == "o1"          # judged on Phi's ops, not the [PLAN]
        monkeypatch.setattr(lnsdl._eg, "plan_expansion_ok", lambda plan, gt: True)
        assert lnsdl._r_struct_sketch(self.OPS, LOCUS_GT, self._trace((True, self.OPS))) == 1.0

    def test_exec_viki_row_never_consults_expansion(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "STRUCT_SKETCH_EXEC", True)
        def _boom(plan, gt):
            raise AssertionError("plan_expansion_ok must not run on VIKI rows")
        monkeypatch.setattr(lnsdl._eg, "plan_expansion_ok", _boom)
        assert lnsdl._r_struct_sketch(self.OPS, GT, self._trace((True, self.OPS))) == 1.0

    # --- P_FORMAT_V2 --------------------------------------------------------
    def test_format_hits_v2_counts_each_shape(self):
        clean = "<think>x</think>\n[TOOL_CALL]\n{\"name\":\"schedule\"}\n[TOOL_RESULT schedule]\n{}\n<think>y</think>\n[PLAN]\n[]"
        assert sum(lnsdl._format_hits_v2(clean).values()) == 0
        loop = "[TOOL_RESULT schedule]\n{}\n<think>y</think>\n[PLAN]\n[]]\n[THINK]y</think>\n[PLAN]\n[]]\n[THINK]y</think>\n[PLAN]\n[]]"
        h = lnsdl._format_hits_v2(loop)
        assert h["multi_plan"] == 2 and h["stray_bracket"] == 3 and h["think_pseudo"] == 2
        same_turn = "<think>y</think>\n[PLAN]\n[]\n[ASSIGN]\n{}\n[TOOL_CALL]\n{\"name\":\"schedule\"}"
        assert lnsdl._format_hits_v2(same_turn)["plan_with_tool_call"] == 1
        # a legitimate nested list is not a stray bracket
        assert lnsdl._format_hits_v2('[PLAN]\n[{"deps":["a"]}]')["stray_bracket"] == 0

    def test_p_format_v2_default_off_then_capped(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        plan_block = self.TEXT.split("[PLAN]\n")[1]
        # same final [PLAN] (so r_task etc. are unchanged), wrapped in the s46 loop shape
        loop_text = self.TEXT + "]" + "".join(f"\n[THINK]z</think>\n[PLAN]\n{plan_block}]" for _ in range(3))
        a = lnsdl.compute_score(self.TEXT, GT, {"trace": self._trace((True, self.OPS))})
        b = lnsdl.compute_score(loop_text, GT, {"trace": self._trace((True, self.OPS))})
        assert b["format_hits_v2"] > 0 and a["format_hits_v2"] == 0
        assert a["p_format"] == b["p_format"] == 0.0            # default off
        monkeypatch.setattr(lnsdl, "P_FORMAT_V2", 0.05)
        monkeypatch.setattr(lnsdl, "P_FORMAT_V2_CAP", 0.30)
        a2 = lnsdl.compute_score(self.TEXT, GT, {"trace": self._trace((True, self.OPS))})
        b2 = lnsdl.compute_score(loop_text, GT, {"trace": self._trace((True, self.OPS))})
        assert a2["score"] == a["score"]
        assert b2["p_format"] == min(0.05 * b2["format_hits_v2"], 0.30) and b2["p_format"] > 0
        assert abs((a2["score"] - b2["score"]) - b2["p_format"]) < 1e-9

    # --- P_SCHED_REPEAT -----------------------------------------------------
    def test_schedule_repeat_counts_only_calls_after_first_accepted(self):
        assert lnsdl._schedule_repeat_count(self._trace((False, []), (True, self.OPS))) == 0   # repair loop is free
        assert lnsdl._schedule_repeat_count(self._trace((True, []), (True, []), (True, []))) == 2
        assert lnsdl._schedule_repeat_count(self._trace((True, self.OPS), (False, []))) == 1
        harness = {"tool_names": ["schedule", "schedule"], "tool_results": [{"ok": True, "ops": self.OPS}, {"ok": True, "ops": self.OPS}]}
        assert lnsdl._schedule_repeat_count(harness) == 1

    def test_p_sched_repeat_default_off_then_priced(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        once = {"trace": self._trace((True, self.OPS))}
        thrice = {"trace": self._trace((True, self.OPS), (True, self.OPS), (True, self.OPS))}
        a = lnsdl.compute_score(self.TEXT, GT, once); b = lnsdl.compute_score(self.TEXT, GT, thrice)
        assert a["score"] == b["score"] and b["sched_repeat"] == 2 and b["p_cost"] == 0.0
        monkeypatch.setattr(lnsdl, "P_SCHED_REPEAT", 0.05)
        b2 = lnsdl.compute_score(self.TEXT, GT, thrice)
        assert abs(b2["p_cost"] - 0.10) < 1e-9 and abs(a["score"] - b2["score"] - 0.10) < 1e-9

    # --- P1d R_TASK_SOURCE --------------------------------------------------
    def test_r_task_source_phi_schedule_scores_phi_ops_and_falls_back(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        # null-GT row: Phi accepted the empty sketch (the right answer) but the
        # model transcribed a non-empty [PLAN]; the main configuration submits
        # Phi's schedule, so phi_schedule scoring pays R_task, plan scoring does not.
        wrong_transcription = '[GOAL]\n{"active": []}\n[PLAN]\n[{"op_id":"o1","op_type":"acquire","entity":"plate","robot":"R1"}]'
        monkeypatch.setattr(lnsdl, "R_TASK_SOURCE", "plan")   # recorded s46/s55/s58 behaviour
        a = lnsdl.compute_score(wrong_transcription, self.NULL_GT, {"trace": self._trace((True, []))})
        assert a["r_task"] == 0.0 and a["r_task_phi"] == 1.0 and a["r_task_scored"] == 0.0 and a["acc"] == 0.0
        assert a["r_task_source_phi"] == 0.0
        monkeypatch.setattr(lnsdl, "R_TASK_SOURCE", "phi_schedule")
        b = lnsdl.compute_score(wrong_transcription, self.NULL_GT, {"trace": self._trace((True, []))})
        assert b["r_task_scored"] == 1.0 and b["acc"] == 1.0 and abs(b["score"] - a["score"] - lnsdl.W_TASK) < 1e-9
        assert b["r_task_source_phi"] == 1.0
        c = lnsdl.compute_score(wrong_transcription, self.NULL_GT, {"trace": self._trace((False, []))})   # nothing accepted -> [PLAN]
        assert c["r_task_phi"] == -1.0 and c["r_task_scored"] == 0.0

    def test_default_sources_are_phi_schedule(self):
        # 2026-09-11: the default reward reads what the main configuration submits.
        assert lnsdl._plan_source("LNSDL_NO_SUCH_SOURCE_ENV") == "phi_schedule"
        assert lnsdl.R_TASK_SOURCE == "phi_schedule" and lnsdl.R_PLAN_SOURCE == "phi_schedule"

    # --- R_PLAN_SOURCE ---------------------------------------------------------
    def test_r_plan_source_phi_schedule_reads_phi_ops_and_falls_back(self, monkeypatch):
        monkeypatch.setattr(lnsdl, "REWARD_PATCH_STACK", False)
        monkeypatch.setattr(lnsdl, "W_GOAL_PLAN", 0.15)
        # Phi accepted a sketch grasping plate (GT active), the transcribed
        # [PLAN] dropped the acquire (the 17% transcription-loss case).
        phi_ops = [{"op_id": "op1", "op_type": "acquire", "entity": "plate", "robot": "R1"},
                   {"op_id": "auto_carry", "op_type": "carry_to", "entity": "plate", "dest": "table", "robot": "R1"}]
        dropped = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[]'
        trace = self._trace((True, phi_ops))
        monkeypatch.setattr(lnsdl, "R_PLAN_SOURCE", "plan")
        a = lnsdl.compute_score(dropped, GT, {"trace": trace})
        assert a["r_goal_plan_transcribed"] == 0.0 and a["r_goal_plan_phi"] == 1.0 and a["r_goal_plan"] == 0.0
        monkeypatch.setattr(lnsdl, "R_PLAN_SOURCE", "phi_schedule")
        b = lnsdl.compute_score(dropped, GT, {"trace": trace})
        assert b["r_goal_plan"] == 1.0 and b["r_plan_source_phi"] == 1.0
        assert abs(b["score"] - a["score"] - 0.15) < 1e-9
        # Phi's mechanical additions do not change the grasp set of a faithful transcription
        faithful = '[GOAL]\n{"active": ["plate"]}\n[PLAN]\n[{"op_id":"op1","op_type":"acquire","entity":"plate","robot":"R1"}]'
        f = lnsdl.compute_score(faithful, GT, {"trace": trace})
        assert f["r_goal_plan"] == f["r_goal_plan_transcribed"] == f["r_goal_plan_phi"] == 1.0
        # nothing accepted -> transcribed [PLAN]
        c = lnsdl.compute_score(faithful, GT, {"trace": self._trace((False, []))})
        assert c["r_goal_plan_phi"] == -1.0 and c["r_goal_plan"] == 1.0
