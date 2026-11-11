"""Tests for the L-NSDL per-axis advantage entry (docs/plan.md §P5.3'.2/.2b).

Case list mirrors the plan:
  (1) saturated axis -> A_k = 0 under cfa
  (2) single axis with variance decides A_total
  (3) two varying axes compose linearly
  (4) task-axis-only variance == baseline scalar advantage (degeneracy)
  (5) mode=gdpo gives NONZERO advantage in a saturated group, cfa gives 0
  (6) mode=scalar matches core_algos.compute_grpo_outcome_advantage value-by-value
  (7) signadv is independent of group composition
  (8) acr == 1.0 on a fully saturated batch
Plus: the non-terminating axis-A guard, axis assembly modes, diagnostics keys.
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

from verl.trainer.ppo import lnsdl_advantage as la  # noqa: E402

TAU = la.DEFAULT_TAU


def _cfg(mode, **kw):
    return la.AxisAdvantageConfig(mode=mode, **kw)


# ---------------------------------------------------------------------------
# (1) saturated axis contributes nothing -- with OR without the mask
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mask", [False, True])
def test_saturated_axis_gets_zero_advantage(mask):
    # Axis B identical inside both groups (mechanically co-saturated, the VIKI
    # situation); axis A varies. std_B is **exactly** 0, so (v-mean)/(0+eps)=0
    # regardless of the mask -- this is the measurement behind A0g8's "the mask
    # is a no-op at this granularity". Parametrised so a future change that
    # makes the two paths disagree on exact saturation fails loudly.
    per_axis = {
        "A": [0.0, 1.0, 0.2, 0.9],
        "B": [1.0, 1.0, 1.0, 1.0],
    }
    groups = ["g0", "g0", "g1", "g1"]
    cfg = _cfg("cfa", cfa_mask=mask)
    adv, diag = la.compute_axis_advantage(per_axis, groups, config=cfg)

    # Axis B contributes nothing: A_total must equal the axis-A-only advantage.
    adv_a_only, _ = la.compute_axis_advantage({"A": per_axis["A"]}, groups, config=cfg)
    assert torch.allclose(adv, adv_a_only)
    # Reward-side readings are mode/knob independent (defined against tau).
    assert diag["acr_per_axis"]["B"] == 1.0
    assert diag["active_group_frac"]["B"] == 0.0
    assert diag["active_group_frac"]["A"] == 1.0
    # Advantage-side masking is exactly what the knob controls.
    assert diag["masked_group_frac"]["B"] == (1.0 if mask else 0.0)
    # The gradient is entirely driven by the axis that has variance.
    assert diag["axis_grad_share"]["A"] == pytest.approx(1.0)
    assert diag["axis_grad_share"]["B"] == pytest.approx(0.0)


def test_cfa_mask_defaults_off():
    """A0g8: default is mask OFF; env LNSDL_CFA_MASK=1 restores frozen-arm behaviour."""
    assert la.AxisAdvantageConfig().cfa_mask is False
    prev = os.environ.get("LNSDL_CFA_MASK")
    try:
        os.environ.pop("LNSDL_CFA_MASK", None)
        assert la.config_from_env().cfa_mask is False
        os.environ["LNSDL_CFA_MASK"] = "1"
        assert la.config_from_env().cfa_mask is True
    finally:
        if prev is None:
            os.environ.pop("LNSDL_CFA_MASK", None)
        else:
            os.environ["LNSDL_CFA_MASK"] = prev


# ---------------------------------------------------------------------------
# (2) a single varying axis determines A_total
# ---------------------------------------------------------------------------
def test_single_varying_axis_determines_total():
    per_axis = {
        "A": [0.0, 1.0],
        "B": [0.5, 0.5],
        "T": [1.0, 1.0],
    }
    groups = ["g0", "g0"]
    adv, diag = la.compute_axis_advantage(per_axis, groups, config=_cfg("cfa"))
    # r_A = [0, 1] -> mean .5, unbiased std = 1/sqrt(2) ~= .7071
    std = float(np.std([0.0, 1.0], ddof=1))
    expected = torch.tensor([(0.0 - 0.5) / (std + la.DEFAULT_EPS),
                             (1.0 - 0.5) / (std + la.DEFAULT_EPS)], dtype=torch.float64)
    assert torch.allclose(adv, expected, atol=1e-9)
    # Sign is correct: the better trajectory on the bottleneck axis is preferred.
    assert adv[1] > 0 > adv[0]
    assert diag["acr"] == 0.0


# ---------------------------------------------------------------------------
# (3) linear composition across axes
# ---------------------------------------------------------------------------
def test_two_varying_axes_compose_linearly():
    per_axis = {"A": [0.0, 1.0], "B": [1.0, 0.0]}
    groups = ["g0", "g0"]
    both, _ = la.compute_axis_advantage(per_axis, groups, config=_cfg("cfa"))
    only_a, _ = la.compute_axis_advantage({"A": per_axis["A"]}, groups, config=_cfg("cfa"))
    only_b, _ = la.compute_axis_advantage({"B": per_axis["B"]}, groups, config=_cfg("cfa"))
    assert torch.allclose(both, only_a + only_b, atol=1e-9)
    # Opposed axes with equal weights cancel.
    assert torch.allclose(both, torch.zeros_like(both), atol=1e-9)

    # Weights scale each contribution.
    weighted, _ = la.compute_axis_advantage(
        per_axis, groups, config=_cfg("cfa", weights={"A": 2.0, "B": 0.5})
    )
    assert torch.allclose(weighted, 2.0 * only_a + 0.5 * only_b, atol=1e-9)


# ---------------------------------------------------------------------------
# (4) degeneracy: only the task axis varies -> baseline scalar advantage
# ---------------------------------------------------------------------------
def test_task_axis_only_matches_scalar_baseline():
    per_axis = {
        "A": [0.5, 0.5, 0.5, 0.5],
        "B": [1.0, 1.0, 1.0, 1.0],
        "T": [0.0, 1.0, 1.0, 0.0],
    }
    groups = ["g0", "g0", "g1", "g1"]
    cfa_adv, _ = la.compute_axis_advantage(per_axis, groups, config=_cfg("cfa"))
    # Scalar baseline computed on the task axis alone (the only live signal).
    scalar_adv, _ = la.compute_axis_advantage(
        {"T": per_axis["T"]}, groups, config=_cfg("scalar", weights={"T": 1.0})
    )
    assert torch.allclose(cfa_adv, scalar_adv, atol=1e-9)


# ---------------------------------------------------------------------------
# (5) gdpo vs cfa: the mechanism difference is pinned down
# ---------------------------------------------------------------------------
def test_gdpo_keeps_gradient_where_cfa_masks():
    # Near-saturated axis: std nonzero but far below tau. gdpo divides by
    # std + eps and amplifies the residual; cfa **with the mask on** exits the
    # axis. Since 2026-09-01 the mask is off by default, so this test pins the
    # mechanism by asking for it explicitly (cfa_mask=True) -- that is the
    # configuration every arm frozen before that date ran under.
    tiny = TAU / 100.0
    per_axis = {"A": [0.5, 0.5 + tiny], "B": [1.0, 1.0]}
    groups = ["g0", "g0"]

    cfa_adv, cfa_diag = la.compute_axis_advantage(
        per_axis, groups, config=_cfg("cfa", cfa_mask=True))
    gdpo_adv, gdpo_diag = la.compute_axis_advantage(per_axis, groups, config=_cfg("gdpo"))

    assert torch.allclose(cfa_adv, torch.zeros_like(cfa_adv), atol=1e-12)
    assert cfa_diag["acr"] == 1.0
    assert float(torch.max(torch.abs(gdpo_adv))) > 1e-3
    assert gdpo_diag["acr"] == 0.0
    # Reward-side collapse reading agrees across modes (both see std < tau).
    assert cfa_diag["acr_per_axis"]["A"] == gdpo_diag["acr_per_axis"]["A"] == 1.0
    # Only cfa (mask on) reports the axis as masked.
    assert cfa_diag["masked_group_frac"]["A"] == 1.0
    assert gdpo_diag["masked_group_frac"]["A"] == 0.0


def test_default_cfa_differs_from_masked_cfa_only_in_the_near_saturated_band():
    """The band (0, tau) is the whole footprint of the knob.

    Off-default cfa keeps the amplified residual there (like gdpo, minus the
    batch norm); mask-on cfa zeroes it. Everywhere else the two are identical.
    """
    tiny = TAU / 100.0
    near = {"A": [0.5, 0.5 + tiny], "B": [1.0, 1.0]}
    groups = ["g0", "g0"]
    off, _ = la.compute_axis_advantage(near, groups, config=_cfg("cfa"))
    on, _ = la.compute_axis_advantage(near, groups, config=_cfg("cfa", cfa_mask=True))
    assert float(torch.max(torch.abs(off))) > 1e-3
    assert torch.allclose(on, torch.zeros_like(on), atol=1e-12)

    # Well above tau: knob is invisible.
    wide = {"A": [0.0, 1.0, 0.2, 0.9], "B": [0.0, 1.0, 1.0, 0.0]}
    g4 = ["g0", "g0", "g1", "g1"]
    a_off, _ = la.compute_axis_advantage(wide, g4, config=_cfg("cfa"))
    a_on, _ = la.compute_axis_advantage(wide, g4, config=_cfg("cfa", cfa_mask=True))
    assert torch.allclose(a_off, a_on, atol=1e-12)


def test_cfa_batchnorm_isolates_the_batchnorm_factor():
    per_axis = {"A": [0.0, 1.0, 0.3, 0.4], "B": [1.0, 1.0, 1.0, 1.0]}
    groups = ["g0", "g0", "g1", "g1"]
    plain, _ = la.compute_axis_advantage(per_axis, groups, config=_cfg("cfa"))
    bn, _ = la.compute_axis_advantage(per_axis, groups, config=_cfg("cfa_batchnorm"))
    # Same masking decision, different scale -> ordering preserved, values not.
    assert torch.equal(torch.argsort(plain), torch.argsort(bn))
    assert not torch.allclose(plain, bn)
    assert float(torch.mean(bn)) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (6) regression: mode=scalar == stock GRPO advantage
# ---------------------------------------------------------------------------
def test_scalar_mode_matches_core_algos_grpo():
    from verl.trainer.ppo import core_algos

    scores = [0.10, 0.85, 0.40, 0.40, 0.95, 0.05]
    groups = np.array(["g0", "g0", "g1", "g1", "g1", "g2"], dtype=object)

    resp_len = 4
    token_level_rewards = torch.zeros(len(scores), resp_len, dtype=torch.float64)
    token_level_rewards[:, -1] = torch.tensor(scores, dtype=torch.float64)
    response_mask = torch.ones(len(scores), resp_len, dtype=torch.float64)

    ref, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards.clone(),
        response_mask=response_mask,
        index=groups,
    )
    ours, _ = la.compute_axis_advantage(
        {"score": scores}, groups, config=_cfg("scalar", weights={"score": 1.0})
    )
    # core_algos broadcasts over tokens; compare the per-trajectory scalar.
    assert torch.allclose(ref[:, 0].double(), ours, atol=1e-9)
    # Includes GRPO's singleton-group convention (mean 0 / std 1, so the lone
    # trajectory keeps its raw score instead of being zeroed).
    assert ours[-1] == pytest.approx(scores[-1], abs=1e-6)


def test_per_axis_modes_treat_singleton_group_as_zero_variance():
    # The per-axis modes must NOT inherit GRPO's singleton convention: a group
    # of one has no within-group variance, so cfa masks it (and gdpo's eps
    # smoothing leaves it at ~0 too), rather than passing the raw reward through
    # as if it were a normalised advantage.
    cfa_adv, cfa_diag = la.compute_axis_advantage(
        {"A": [0.7], "B": [1.0]}, ["g0"], config=_cfg("cfa"))
    assert torch.allclose(cfa_adv, torch.zeros(1, dtype=torch.float64))
    assert cfa_diag["acr"] == 1.0
    gdpo_adv, _ = la.compute_axis_advantage(
        {"A": [0.7], "B": [1.0]}, ["g0"], config=_cfg("gdpo"))
    assert torch.allclose(gdpo_adv, torch.zeros(1, dtype=torch.float64), atol=1e-9)


# ---------------------------------------------------------------------------
# (7) signadv ignores group statistics
# ---------------------------------------------------------------------------
def test_signadv_is_independent_of_group_composition():
    rewards = [1.0, 0.0, 1.0, 1.0]
    adv_a, _ = la.compute_axis_advantage(
        {"T": rewards}, ["g0", "g0", "g1", "g1"], config=_cfg("signadv")
    )
    adv_b, _ = la.compute_axis_advantage(
        {"T": rewards}, ["g0", "g1", "g0", "g1"], config=_cfg("signadv")
    )
    assert torch.allclose(adv_a, adv_b)
    assert torch.allclose(adv_a, torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.float64))
    # Fully saturated group still yields a gradient -- that is the point of the
    # collapse baseline.
    adv_sat, diag = la.compute_axis_advantage(
        {"T": [1.0, 1.0]}, ["g0", "g0"], config=_cfg("signadv")
    )
    assert torch.allclose(adv_sat, torch.ones(2, dtype=torch.float64))
    assert diag["acr"] == 0.0


def test_signadv_requires_task_axis():
    with pytest.raises(ValueError):
        la.compute_axis_advantage({"A": [1.0, 0.0]}, ["g0", "g0"], config=_cfg("signadv"))


# ---------------------------------------------------------------------------
# (8) ACR on a fully saturated batch
# ---------------------------------------------------------------------------
def test_acr_is_one_on_fully_saturated_batch():
    per_axis = {
        "A": [1.0, 1.0, 0.5, 0.5],
        "B": [1.0, 1.0, 1.0, 1.0],
        "T": [1.0, 1.0, 1.0, 1.0],
    }
    groups = ["g0", "g0", "g1", "g1"]
    adv, diag = la.compute_axis_advantage(per_axis, groups, config=_cfg("cfa"))
    assert torch.allclose(adv, torch.zeros_like(adv))
    assert diag["acr"] == 1.0
    assert all(v == 1.0 for v in diag["acr_per_axis"].values())
    assert all(v == 0.0 for v in diag["active_group_frac"].values())
    # No axis claims gradient share when there is no gradient at all.
    assert all(v == 0.0 for v in diag["axis_grad_share"].values())


# ---------------------------------------------------------------------------
# axis assembly + non-terminating guard
# ---------------------------------------------------------------------------
def _fields(**kw):
    base = {
        "r_goal": [1.0, 0.0],
        "r_faithful": [0.0, 1.0],
        "r_struct": [1.0, 1.0],
        "r_task": [1.0, 0.0],
    }
    base.update(kw)
    return base


def test_axis_assembly_merge_mode():
    axes = la.assemble_axis_rewards(_fields(), config=_cfg("cfa"))
    assert set(axes) == {"A", "B", "T"}
    assert torch.allclose(axes["A"], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_axis_assembly_ablation_modes():
    goal_only = la.assemble_axis_rewards(_fields(), config=_cfg("cfa", axis_a_mode="goal_only"))
    assert torch.allclose(goal_only["A"], torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert "F" not in goal_only

    split = la.assemble_axis_rewards(_fields(), config=_cfg("cfa", axis_a_mode="split"))
    assert set(split) == {"A", "F", "B", "T"}
    assert torch.allclose(split["F"], torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_axis_subset_selection():
    axes = la.assemble_axis_rewards(_fields(), config=_cfg("cfa", axes=["A", "T"]))
    assert list(axes) == ["A", "T"]
    with pytest.raises(ValueError):
        la.assemble_axis_rewards(_fields(), config=_cfg("cfa", axes=["A", "Z"]))


def test_nonterm_guard_zeroes_axis_a():
    fields = _fields(r_goal=[1.0, 1.0], r_faithful=[1.0, 1.0], nonterm=[0.0, 1.0])
    axes = la.assemble_axis_rewards(fields, config=_cfg("cfa"))
    assert torch.allclose(axes["A"], torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_nonterm_trajectory_gets_no_positive_axis_a_advantage():
    # A high-goal trajectory that never terminates must not out-rank the
    # terminating one on axis A (the definitional replacement for P_nonterm).
    fields = _fields(
        r_goal=[0.5, 1.0], r_faithful=[0.5, 1.0], r_struct=[1.0, 1.0],
        r_task=[1.0, 0.0], nonterm=[0.0, 1.0],
    )
    axes = la.assemble_axis_rewards(fields, config=_cfg("cfa"))
    adv, _ = la.compute_axis_advantage(
        {"A": axes["A"]}, ["g0", "g0"], config=_cfg("cfa")
    )
    assert adv[1] < 0 < adv[0]


def test_guard_can_be_disabled_for_ablation():
    fields = _fields(r_goal=[1.0, 1.0], r_faithful=[1.0, 1.0], nonterm=[0.0, 1.0])
    axes = la.assemble_axis_rewards(
        fields, config=_cfg("cfa", nonterm_zeroes_axis_a=False)
    )
    assert torch.allclose(axes["A"], torch.ones(2, dtype=torch.float64))


def test_assembly_rejects_missing_reward_field():
    fields = _fields()
    fields.pop("r_struct")
    with pytest.raises(KeyError):
        la.assemble_axis_rewards(fields, config=_cfg("cfa"))


# ---------------------------------------------------------------------------
# config / env plumbing
# ---------------------------------------------------------------------------
def test_config_from_env_reads_single_switch(monkeypatch):
    monkeypatch.setenv("LNSDL_ADV_MODE", "gdpo")
    monkeypatch.setenv("LNSDL_CFA_W_A", "2.5")
    monkeypatch.setenv("LNSDL_CFA_TAU", "0.01")
    monkeypatch.setenv("LNSDL_CFA_AXES", "A,B")
    cfg = la.config_from_env()
    assert cfg.mode == "gdpo"
    assert cfg.weight("A") == 2.5
    assert cfg.weight("B") == 1.0
    assert cfg.tau == 0.01
    assert cfg.axes == ("A", "B")


def test_config_from_env_defaults_to_cfa(monkeypatch):
    for k in ("LNSDL_ADV_MODE", "LNSDL_CFA_AXES", "LNSDL_CFA_TAU"):
        monkeypatch.delenv(k, raising=False)
    assert la.config_from_env().mode == "cfa"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        la.AxisAdvantageConfig(mode="ours_new_thing")
    with pytest.raises(ValueError):
        la.compute_axis_advantage({"T": [1.0]}, ["g0"], mode="bogus")


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        la.compute_axis_advantage({"A": [1.0, 0.0], "B": [1.0]}, ["g0", "g0"], config=_cfg("cfa"))
    with pytest.raises(ValueError):
        la.compute_axis_advantage({"A": [1.0, 0.0]}, ["g0"], config=_cfg("cfa"))


# ---------------------------------------------------------------------------
# diagnostics surface
# ---------------------------------------------------------------------------
def test_diagnostics_keys_and_flattening():
    per_axis = {"A": [0.0, 1.0], "B": [1.0, 1.0], "T": [0.0, 1.0]}
    _, diag = la.compute_axis_advantage(per_axis, ["g0", "g0"], config=_cfg("cfa"))
    for key in ("acr", "acr_per_axis", "active_group_frac", "within_group_std",
                "axis_grad_share", "masked_group_frac", "mode", "n_groups", "tau"):
        assert key in diag
    assert set(diag["within_group_std"]["A"]) == {"mean", "p10", "p90"}

    flat = la.flat_metrics(diag)
    assert flat["lnsdl_adv/acr"] == diag["acr"]
    assert flat["lnsdl_adv/acr_per_axis/B"] == 1.0
    assert "lnsdl_adv/within_group_std/A/p90" in flat
    assert all(isinstance(v, float) for v in flat.values())


def test_collapse_log_appends_jsonl(tmp_path):
    import json

    path = tmp_path / "collapse.jsonl"
    per_axis = {"A": [0.0, 1.0], "B": [1.0, 1.0]}
    for _ in range(2):
        _, diag = la.compute_axis_advantage(per_axis, ["g0", "g0"], config=_cfg("cfa"))
        assert la.log_collapse_diagnostics(diag, path=str(path)) == str(path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]
    assert recs[0]["step"] < recs[1]["step"]
    assert recs[0]["mode"] == "cfa"


def test_collapse_log_off_by_default(monkeypatch):
    monkeypatch.delenv("LNSDL_COLLAPSE_LOG", raising=False)
    _, diag = la.compute_axis_advantage({"A": [0.0, 1.0]}, ["g0", "g0"], config=_cfg("cfa"))
    assert la.log_collapse_diagnostics(diag) is None


# ---------------------------------------------------------------------------
# naming discipline (docs/plan.md §2-14)
# ---------------------------------------------------------------------------
def test_module_has_no_priority_claim_wording():
    with open(la.__file__) as f:
        src = f.read().lower()
    for banned in ("first_", "novel_advantage", "our_new_estimator", "first to ", "first ever"):
        assert banned not in src, f"banned wording {banned!r} in lnsdl_advantage.py"


def test_all_modes_are_runnable():
    per_axis = {"A": [0.0, 1.0, 0.3, 0.7], "B": [1.0, 1.0, 0.5, 1.0], "T": [0.0, 1.0, 1.0, 0.0]}
    groups = ["g0", "g0", "g1", "g1"]
    for mode in la.ADV_MODES:
        cfg = _cfg(mode)
        if mode == "scalar":
            adv, diag = la.compute_axis_advantage(
                {"score": [0.1, 0.9, 0.5, 0.2]}, groups,
                config=cfg, weights={"score": 1.0})
        else:
            adv, diag = la.compute_axis_advantage(per_axis, groups, config=cfg)
        assert adv.shape == (4,)
        assert torch.isfinite(adv).all()
        assert diag["mode"] == mode


# ---------------------------------------------------------------------------
# M1 · DAPO overlong filtering (docs/debug_rl.md §3.5, 2026-09-04)
# ---------------------------------------------------------------------------
def test_overlong_filter_off_by_default_and_records_share(monkeypatch):
    monkeypatch.delenv("LNSDL_OVERLONG_FILTER", raising=False)
    from verl.trainer.ppo import lnsdl_advantage as LA
    adv = torch.tensor([1.5, -2.0, 0.7, -0.3], dtype=torch.float64)
    lens = torch.tensor([2048, 900, 2048, 1200])
    diag = {}
    out = LA.apply_overlong_filter(adv, lens, 2048, diag)
    assert torch.equal(out, adv)                      # byte-identical when off
    assert diag["overlong_filter_on"] == 0.0
    assert diag["overlong_masked_n"] == 0
    assert abs(diag["overlong_truncated_frac"] - 0.5) < 1e-12


def test_overlong_filter_zeroes_only_capped_trajectories(monkeypatch):
    monkeypatch.setenv("LNSDL_OVERLONG_FILTER", "1")
    from verl.trainer.ppo import lnsdl_advantage as LA
    adv = torch.tensor([1.5, -2.0, 0.7, -0.3], dtype=torch.float64)
    lens = torch.tensor([2048, 900, 2048, 1200])
    diag = {}
    out = LA.apply_overlong_filter(adv, lens, 2048, diag)
    assert out.tolist() == [0.0, -2.0, 0.0, -0.3]
    assert adv.tolist() == [1.5, -2.0, 0.7, -0.3]     # input not mutated
    assert diag["overlong_masked_n"] == 2 and diag["overlong_filter_on"] == 1.0
    flat = LA.flat_metrics(diag)
    assert flat["lnsdl_adv/overlong_masked_n"] == 2.0
    assert flat["lnsdl_adv/overlong_truncated_frac"] == 0.5
