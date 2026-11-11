"""Plumbing tests for the lnsdl_axis advantage estimator inside verl.

Covers what tests/test_cfa_advantage.py cannot: that ``compute_advantage`` picks
up the per-axis reward keys the DAPO reward manager leaves in
``non_tensor_batch``, broadcasts the scalar advantage over the response tokens
with the multi-turn ``loss_mask``, and keeps ``mode=scalar`` numerically equal to
the stock ``grpo`` estimator end to end.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "verl")):
    if p not in sys.path:
        sys.path.insert(0, p)

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage  # noqa: E402

RESP_LEN = 6

# Two prompt groups of 3 rollouts. Group g0 has axis-A variance and a saturated
# axis B (the VIKI situation); g1 is fully saturated on every axis.
REWARDS = {
    "r_goal":     [0.0, 1.0, 0.5,  1.0, 1.0, 1.0],
    "r_faithful": [0.0, 1.0, 1.0,  1.0, 1.0, 1.0],
    "r_struct":   [1.0, 1.0, 1.0,  1.0, 1.0, 1.0],
    "r_task":     [0.0, 1.0, 0.0,  1.0, 1.0, 1.0],
    "nonterm":    [0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
    "score":      [0.20, 1.00, 0.45, 1.00, 1.00, 1.00],
}
UIDS = np.array(["g0", "g0", "g0", "g1", "g1", "g1"], dtype=object)


def _make_batch(loss_mask=None):
    bsz = len(UIDS)
    token_level_rewards = torch.zeros(bsz, RESP_LEN)
    token_level_rewards[:, -1] = torch.tensor(REWARDS["score"])
    tensors = {
        "responses": torch.ones(bsz, RESP_LEN, dtype=torch.long),
        "attention_mask": torch.ones(bsz, 2 * RESP_LEN, dtype=torch.long),
        "token_level_rewards": token_level_rewards,
    }
    if loss_mask is not None:
        tensors["loss_mask"] = loss_mask
    non_tensors = {k: np.asarray(v, dtype=np.float64) for k, v in REWARDS.items()}
    non_tensors["uid"] = UIDS
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _run(mode, loss_mask=None, **env):
    old = {}
    env = {"LNSDL_ADV_MODE": mode, **env}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        data = compute_advantage(_make_batch(loss_mask), AdvantageEstimator.LNSDL_AXIS)
        return data
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_estimator_is_registered_and_critic_free():
    assert AdvantageEstimator.LNSDL_AXIS.value == "lnsdl_axis"


def test_cfa_masks_saturated_group_end_to_end():
    data = _run("cfa")
    adv = data.batch["advantages"]
    assert adv.shape == (6, RESP_LEN)
    # g1 is fully saturated -> zero gradient for all three of its rollouts.
    assert torch.allclose(adv[3:], torch.zeros(3, RESP_LEN))
    # g0 has axis-A variance -> nonzero, and ordered by axis A.
    assert float(adv[1, 0]) > float(adv[0, 0])
    diag = data.meta_info["lnsdl_adv_diagnostics"]
    assert diag["mode"] == "cfa"
    assert diag["acr"] == pytest.approx(0.5)          # 1 of 2 groups dead
    assert diag["acr_per_axis"]["B"] == 1.0           # axis B saturated in both
    assert diag["active_group_frac"]["A"] == pytest.approx(0.5)
    assert data.batch["returns"].shape == adv.shape


def test_gdpo_does_not_mask_the_saturated_group():
    cfa = _run("cfa").batch["advantages"]
    gdpo = _run("gdpo").batch["advantages"]
    # The mechanism difference survives the full trainer path, not just the
    # pure function: gdpo's eps smoothing keeps the saturated group alive.
    assert torch.allclose(cfa[3:], torch.zeros(3, RESP_LEN))
    assert float(torch.max(torch.abs(gdpo[3:]))) > 0.0


def test_scalar_mode_matches_stock_grpo_estimator():
    ours = _run("scalar").batch["advantages"]
    ref = compute_advantage(_make_batch(), AdvantageEstimator.GRPO).batch["advantages"]
    assert torch.allclose(ours.float(), ref.float(), atol=1e-6)


def test_signadv_gives_gradient_in_the_saturated_group():
    adv = _run("signadv").batch["advantages"]
    # r_task = 1 for all of g1 -> A = +1 everywhere (no group statistics).
    assert torch.allclose(adv[3:], torch.ones(3, RESP_LEN))
    assert float(adv[0, 0]) == pytest.approx(-1.0)


def test_advantage_respects_multi_turn_loss_mask():
    # Tool-observation tokens are masked out by the multi-turn rollout; the
    # advantage broadcast must honour that mask, not the attention mask.
    loss_mask = torch.ones(6, RESP_LEN)
    loss_mask[:, :2] = 0.0
    adv = _run("cfa", loss_mask=loss_mask).batch["advantages"]
    assert torch.allclose(adv[:, :2], torch.zeros(6, 2))
    assert float(torch.max(torch.abs(adv[:, 2:]))) > 0.0


def test_collapse_log_is_written_when_configured(tmp_path):
    import json

    path = tmp_path / "collapse.jsonl"
    _run("cfa", LNSDL_COLLAPSE_LOG=str(path))
    rec = json.loads(path.read_text().strip().split("\n")[-1])
    assert rec["mode"] == "cfa"
    assert "acr_per_axis" in rec and "axis_grad_share" in rec


def test_axis_weights_are_honoured():
    # Restrict to axis A so the weight effect is isolated (axis T also has
    # variance in g0, so the full-axis total is not proportional to w_A alone).
    base = _run("cfa", LNSDL_CFA_AXES="A").batch["advantages"]
    doubled = _run("cfa", LNSDL_CFA_AXES="A", LNSDL_CFA_W_A="2.0").batch["advantages"]
    assert torch.allclose(doubled, 2.0 * base, atol=1e-6)


def test_axis_subset_env_switch_changes_the_gradient():
    full = _run("cfa").batch["advantages"]
    a_only = _run("cfa", LNSDL_CFA_AXES="A").batch["advantages"]
    # Dropping the task axis changes the composition but not which groups live.
    assert not torch.allclose(full, a_only)
    assert torch.allclose(a_only[3:], torch.zeros(3, RESP_LEN))


def test_nonterm_guard_zeroes_axis_a_in_the_trainer_path():
    # A non-terminating rollout must not gain axis-A credit for a well-declared
    # GOAL (the definitional replacement for the deleted P_nonterm).
    global REWARDS
    saved = REWARDS
    REWARDS = {**saved, "r_goal": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               "r_faithful": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               "nonterm": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
    try:
        adv = _run("cfa", LNSDL_CFA_AXES="A").batch["advantages"]
        # rollouts 0,1 terminated with r_A=1; rollout 2 did not -> r_A=0.
        assert float(adv[2, 0]) < 0 < float(adv[0, 0])
        off = _run("cfa", LNSDL_CFA_AXES="A",
                   LNSDL_CFA_NONTERM_GUARD="0").batch["advantages"]
        assert torch.allclose(off[:3], torch.zeros(3, RESP_LEN))
    finally:
        REWARDS = saved


# ---------------------------------------------------------------------------
# Protocol E / B (docs/plan.md §P5.3'.2b)
# ---------------------------------------------------------------------------
def test_adv_dump_is_mode_independent_and_replayable(tmp_path):
    """The dumped batch must support replaying *every* mode, not the one that ran.

    That is the whole point of protocol E: one GPU run, all estimators read off
    it offline. If the dump were taken inside a mode branch it would silently
    become mode specific and the contrast would compare a mode against itself.
    """
    import json

    path = tmp_path / "adv_batch.jsonl"
    for mode in ("scalar", "cfa"):
        _run(mode, LNSDL_ADV_DUMP=str(path))
    lines = [json.loads(l) for l in path.read_text().strip().split("\n")]
    assert len(lines) == 2
    assert [l["run_mode"] for l in lines] == ["scalar", "cfa"]
    # Same rewards in, same dump out -- the run_mode tag is the only difference.
    assert lines[0]["fields"] == lines[1]["fields"]
    assert lines[0]["uid"] == list(UIDS)
    assert [l["stage"] for l in lines] == ["post_filter", "post_filter"]

    from verl.trainer.ppo import lnsdl_advantage as la

    fields = lines[0]["fields"]
    for name in ("r_goal", "r_faithful", "r_struct", "r_task", "score", "nonterm"):
        assert name in fields, name
    axes = la.assemble_axis_rewards(fields, config=la.AxisAdvantageConfig(mode="cfa"))
    for mode in la.ADV_MODES:
        per_axis = {"score": fields["score"]} if mode == "scalar" else axes
        weights = {"score": 1.0} if mode == "scalar" else None
        adv, _ = la.compute_axis_advantage(per_axis, lines[0]["uid"], mode=mode, weights=weights)
        assert adv.shape == (len(UIDS),)
    # Replayed scalar must reproduce what the trainer computed for that mode.
    replayed, _ = la.compute_axis_advantage(
        {"score": fields["score"]}, lines[0]["uid"], mode="scalar", weights={"score": 1.0})
    live = _run("scalar").batch["advantages"][:, 0]
    assert torch.allclose(replayed.float(), live.float(), atol=1e-6)


def test_adv_dump_is_off_by_default(tmp_path):
    data = _run("cfa")
    assert "lnsdl_adv_diagnostics" in data.meta_info  # the run itself still works
    assert not list(tmp_path.iterdir())


def test_dump_stages_are_tagged_and_counted_separately(tmp_path):
    """Pre- and post-filter batches must stay distinguishable in one file.

    DAPO dynamic sampling keeps only prompt groups that *have* accuracy variance,
    so an advantage-collapse rate read off the post-filter batch is near zero by
    construction -- blind to exactly the collapse CFA is supposed to handle. The
    two sites append to the same jsonl, so the ``stage`` tag (and a per-stage
    step counter, so the two do not renumber each other) is what keeps them from
    being silently pooled.
    """
    import json

    from verl.trainer.ppo import lnsdl_advantage as la

    path = tmp_path / "adv_batch.jsonl"
    fields = {"acc": [0.0, 1.0], "score": [0.1, 0.9], "r_task": [0.0, 1.0]}
    saved = dict(la._DUMP_STEP)
    la._DUMP_STEP.clear()
    try:
        for _ in range(2):
            for stage in la.DUMP_STAGES:
                la.log_axis_batch(fields, ["g0", "g0"], path=str(path), stage=stage)
    finally:
        la._DUMP_STEP.clear()
        la._DUMP_STEP.update(saved)
    recs = [json.loads(l) for l in path.read_text().strip().split("\n")]
    assert [r["stage"] for r in recs] == ["pre_filter", "post_filter"] * 2
    for stage in la.DUMP_STAGES:
        assert [r["step"] for r in recs if r["stage"] == stage] == [1, 2]


def test_launcher_command_is_identical_across_modes():
    """Protocol B (common random numbers) rests on this.

    The three compared arms must differ ONLY in the advantage composition. Every
    LNSDL_* knob is an env var, and ``adv_estimator`` collapses every non-``off``
    mode to ``lnsdl_axis``, so the rendered hydra command should be byte
    identical -- which is what makes the step-1 rollout shared across arms. A
    future knob that leaks into the command line would silently break the
    pairing (and, per docs/debug_rl.md, a single differing ``max_model_len``
    alone is worth 15.6 accuracy points).
    """
    import hashlib
    import shutil
    import subprocess

    script = os.path.join(REPO_ROOT, "scripts", "run_dapo_lnsdl_7b.sh")
    if not os.path.exists(script):
        pytest.skip("launcher not present")
    stub = os.path.join(REPO_ROOT, "tests", "_echo_argv.sh")
    with open(stub, "w") as fh:
        fh.write('#!/usr/bin/env bash\nfor a in "$@"; do echo "$a"; done\n')
    os.chmod(stub, 0o755)
    exp = "pytest-mode-parity"
    try:
        digests = {}
        for mode in ("scalar", "gdpo", "cfa", "cfa_batchnorm", "signadv"):
            env = {**os.environ, "PYTHON_BIN": stub, "LNSDL_ADV_MODE": mode,
                   "EXP_NAME": exp}
            out = subprocess.run(["bash", script], env=env, capture_output=True,
                                 text=True, timeout=300).stdout
            argv = [l for l in out.split("\n") if not l.startswith("[")]
            digests[mode] = hashlib.sha256("\n".join(argv).encode()).hexdigest()
        assert len(set(digests.values())) == 1, digests
    finally:
        os.remove(stub)
        # The launcher mkdir -p's the collapse/dump log dir before exec.
        shutil.rmtree(os.path.join(REPO_ROOT, "results", "lnsdl", "rl", exp),
                      ignore_errors=True)
