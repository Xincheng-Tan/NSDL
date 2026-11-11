"""L-NSDL per-axis advantage composition (docs/plan.md §P5.3'.2 / §P5.3'.2b).

Single entry point for every advantage variant the L-NSDL experiments compare.
``compute_axis_advantage`` takes the per-axis trajectory rewards returned by
``verl/verl/utils/reward_score/lnsdl.py`` and produces one scalar advantage per
trajectory plus the collapse diagnostics. The trainer then broadcasts that
scalar over the response tokens exactly as the stock GRPO path does.

Design discipline (docs/plan.md §2-13): the four compared variants MUST share
this one code path, one rollout, one reward, one seed set, and differ ONLY by
the ``mode`` argument (env ``LNSDL_ADV_MODE``). Writing a second pipeline for
any of them destroys the controls the two cuts rest on.

Modes
-----
``scalar``
    Weighted sum of the axes first, then a single group-relative
    normalisation. Passing ``{"score": <total reward>}`` with weight 1.0 makes
    this bit-identical to ``core_algos.compute_grpo_outcome_advantage`` (the
    stock DAPO/GRPO control), including its singleton-group convention.
``gdpo``
    Per-component group normalisation, then sum, then a batch-level
    normalisation. Aligned with GDPO (arXiv 2601.05242, ICML'26; NeMo-RL
    ``GDPOAdvantageEstimator``): every component is normalised with ``std + eps``
    smoothing and **no** zero-variance masking. This is the key control for the
    CFA claim -- it holds the formula itself fixed, so the only remaining
    difference is (i) axes factorised by the sound solver Phi and (ii) the
    explicit zero-variance exit below.
``cfa``
    ``gdpo``'s per-component normalisation, **without** batch-level
    normalisation. The zero-variance exit (``std_g < tau => A_k := 0`` for that
    (group, axis)) is available but **off by default since 2026-09-01**
    (docs/plan.md A0g8): at VIKI/LOCUS granularity ``R_task``/``R_struct`` are
    0/1, so a within-group std strictly inside ``(0, tau)`` essentially never
    occurs and the mask was measured to be a no-op (debug_rl 5.5.19). Set
    ``LNSDL_CFA_MASK=1`` (or pass ``cfa_mask=True``) to restore it -- required
    to reproduce every arm frozen before that date. ``tau`` stays live either
    way: the reward-side diagnostics ``acr_per_axis`` / ``active_group_frac``
    are defined against it and are mode independent. Use ``cfa_batchnorm`` to
    control the batch-norm factor separately rather than letting it ride along
    inside a "masking works" conclusion.
``cfa_batchnorm``
    ``cfa`` + the same batch-level normalisation ``gdpo`` applies. Only exists
    so the batch-norm factor can be isolated.
``signadv``
    Sign advantage ``A = 2*r - 1`` on the task axis (arXiv 2605.07689). Fixed
    reference point, no group statistics at all -- the collapse baseline that
    keeps a gradient in fully saturated groups. CFA has to beat this on the
    state axis, otherwise the honest claim is only "CFA is one workable fix".

EDGE-GRPO (arXiv 2507.21848) is deliberately absent: its correction needs
per-token entropy, which this trajectory-level entry does not receive. It is
the designated cuttable item in docs/plan.md §7.

Naming (docs/plan.md §2-14): CFA keys are ``cfa_*``, the GDPO control is
``gdpo_*``, the collapse baseline is ``signadv_*``. No "first"/"novel" wording
anywhere in this module or its log keys.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "ADV_MODES",
    "AxisAdvantageConfig",
    "assemble_axis_rewards",
    "apply_overlong_filter",
    "compute_axis_advantage",
    "config_from_env",
    "log_axis_batch",
    "log_collapse_diagnostics",
    "overlong_filter_enabled",
]

ADV_MODES = ("scalar", "gdpo", "cfa", "cfa_batchnorm", "signadv")

# Axis identifiers. A = state judgement (facet A), B = mechanical structure
# (facet B, factorised by Phi), T = terminal task outcome, F = faithfulness when
# split out of axis A as an ablation.
AXIS_A = "A"
AXIS_B = "B"
AXIS_T = "T"
AXIS_F = "F"

DEFAULT_TAU = 1e-3
DEFAULT_EPS = 1e-6
# |w_k * A_k| below this counts as "no gradient from this axis" for the
# advantage-side collapse rate.
ZERO_ADV_ATOL = 1e-8


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envb(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class AxisAdvantageConfig:
    """Resolved knobs for one run. Built from env by :func:`config_from_env`."""

    def __init__(
        self,
        mode: str = "cfa",
        weights: Optional[Mapping[str, float]] = None,
        axes: Optional[Sequence[str]] = None,
        tau: float = DEFAULT_TAU,
        epsilon: float = DEFAULT_EPS,
        axis_a_mode: str = "merge",
        nonterm_zeroes_axis_a: bool = True,
        cfa_mask: bool = False,
    ):
        mode = str(mode).strip().lower()
        if mode not in ADV_MODES:
            raise ValueError(f"unknown advantage mode {mode!r}; expected one of {ADV_MODES}")
        self.mode = mode
        self.weights: Dict[str, float] = dict(weights or {})
        self.axes: Optional[Tuple[str, ...]] = tuple(axes) if axes else None
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        if axis_a_mode not in ("merge", "goal_only", "split"):
            raise ValueError(f"unknown axis_a_mode {axis_a_mode!r}")
        self.axis_a_mode = axis_a_mode
        self.nonterm_zeroes_axis_a = bool(nonterm_zeroes_axis_a)
        # 2026-09-01 (plan.md A0g8): default OFF. Frozen arms need cfa_mask=True.
        self.cfa_mask = bool(cfa_mask)

    def weight(self, axis: str) -> float:
        return float(self.weights.get(axis, 1.0))

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "axes": list(self.axes) if self.axes else None,
            "weights": dict(self.weights),
            "tau": self.tau,
            "epsilon": self.epsilon,
            "axis_a_mode": self.axis_a_mode,
            "nonterm_zeroes_axis_a": self.nonterm_zeroes_axis_a,
            "cfa_mask": self.cfa_mask,
        }


def config_from_env() -> AxisAdvantageConfig:
    """Read the run configuration from ``LNSDL_*`` env vars.

    ``LNSDL_ADV_MODE``            one of :data:`ADV_MODES` (default ``cfa``)
    ``LNSDL_CFA_AXES``            comma list, e.g. ``A,B,T`` (default: all built)
    ``LNSDL_CFA_W_{A,B,T,F}``     per-axis weight (default 1.0)
    ``LNSDL_CFA_TAU``             zero-variance threshold (default 1e-3)
    ``LNSDL_CFA_EPS``             std smoothing, shared by gdpo and cfa
    ``LNSDL_CFA_AXIS_A_MODE``     ``merge`` | ``goal_only`` | ``split``
    ``LNSDL_CFA_NONTERM_GUARD``   non-terminating rollout gets ``r_A := 0``
    ``LNSDL_CFA_MASK``            zero-variance exit for ``cfa``/``cfa_batchnorm``
                                  (**default off** since 2026-09-01; set to 1 to
                                  reproduce arms frozen before that date)
    """
    axes_raw = os.environ.get("LNSDL_CFA_AXES")
    axes = [a.strip() for a in axes_raw.split(",") if a.strip()] if axes_raw else None
    weights = {a: _envf(f"LNSDL_CFA_W_{a}", 1.0) for a in (AXIS_A, AXIS_B, AXIS_T, AXIS_F)}
    return AxisAdvantageConfig(
        mode=os.environ.get("LNSDL_ADV_MODE", "cfa"),
        weights=weights,
        axes=axes,
        tau=_envf("LNSDL_CFA_TAU", DEFAULT_TAU),
        epsilon=_envf("LNSDL_CFA_EPS", DEFAULT_EPS),
        axis_a_mode=os.environ.get("LNSDL_CFA_AXIS_A_MODE", "merge").strip().lower(),
        nonterm_zeroes_axis_a=_envb("LNSDL_CFA_NONTERM_GUARD", True),
        cfa_mask=_envb("LNSDL_CFA_MASK", False),
    )


# ---------------------------------------------------------------------------
# Axis assembly (reward dict -> per-axis trajectory rewards)
# ---------------------------------------------------------------------------
def assemble_axis_rewards(
    reward_fields: Mapping[str, Sequence[float]],
    config: Optional[AxisAdvantageConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Build the per-axis reward vectors from the lnsdl reward breakdown.

    ``reward_fields`` maps reward-component name -> per-trajectory values, i.e.
    the ``reward_extra_info`` dict the DAPO reward manager already puts into
    ``batch.non_tensor_batch`` before ``compute_advantage`` runs. Required keys:
    ``r_goal``, ``r_faithful``, ``r_struct``, ``r_task``; ``nonterm`` is used by
    the guard when present.

    Axis A (state judgement) merges ``R_goal`` and ``R_faithful`` because
    ``R_goal`` only reads the *declared* ``[GOAL]``: an over-grasping trajectory
    keeps ``R_goal == 1`` and only ``R_faithful`` (and the sparse ``R_task``) can
    see it. ``axis_a_mode`` exposes the two ablations (goal-only / faithful as
    its own axis).

    Non-terminating guard (docs/plan.md §P5.3'.2, the definitional replacement
    for the deleted ``P_nonterm``): a rollout that never emits a final ``[PLAN]``
    gets ``r_A := 0``. Without it CFA would hand a positive axis-A advantage to
    "declares well, never finishes" trajectories and non-termination becomes
    profitable again. This is a definition, not a penalty term.
    """
    cfg = config or config_from_env()

    def _vec(name: str) -> torch.Tensor:
        if name not in reward_fields:
            raise KeyError(
                f"axis assembly needs reward field {name!r}; got {sorted(reward_fields)}"
            )
        return torch.as_tensor(np.asarray(reward_fields[name], dtype=np.float64))

    r_goal = _vec("r_goal")
    r_faithful = _vec("r_faithful")
    r_struct = _vec("r_struct")
    r_task = _vec("r_task")

    axes: Dict[str, torch.Tensor] = {}
    if cfg.axis_a_mode == "merge":
        axes[AXIS_A] = (r_goal + r_faithful) / 2.0
    elif cfg.axis_a_mode == "goal_only":
        axes[AXIS_A] = r_goal.clone()
    else:  # split
        axes[AXIS_A] = r_goal.clone()
        axes[AXIS_F] = r_faithful.clone()
    axes[AXIS_B] = r_struct.clone()
    axes[AXIS_T] = r_task.clone()

    if cfg.nonterm_zeroes_axis_a and "nonterm" in reward_fields:
        nonterm = torch.as_tensor(np.asarray(reward_fields["nonterm"], dtype=np.float64)) > 0
        for key in (AXIS_A, AXIS_F):
            if key in axes:
                axes[key] = torch.where(nonterm, torch.zeros_like(axes[key]), axes[key])

    if cfg.axes:
        missing = [a for a in cfg.axes if a not in axes]
        if missing:
            raise ValueError(f"LNSDL_CFA_AXES requests unknown axes {missing}")
        axes = {a: axes[a] for a in cfg.axes}
    return axes


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------
def _group_slots(group_index: Sequence[Any]) -> Tuple[List[Any], Dict[Any, List[int]]]:
    slots: Dict[Any, List[int]] = defaultdict(list)
    for i, gid in enumerate(group_index):
        slots[gid].append(i)
    order = list(slots.keys())
    return order, slots


def _group_mean_std(
    values: torch.Tensor,
    slots: Mapping[Any, Sequence[int]],
    legacy_singleton: bool,
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """Per-group mean and (unbiased) std of ``values``.

    ``legacy_singleton`` reproduces ``compute_grpo_outcome_advantage``'s
    convention for a group of one (mean 0.0, std 1.0), which is what keeps
    ``mode=scalar`` bit-identical to the stock path. The per-axis modes use the
    honest statistics instead (mean = the single value, std = 0), so a singleton
    group is correctly treated as zero-variance.
    """
    means: Dict[Any, float] = {}
    stds: Dict[Any, float] = {}
    for gid, idx in slots.items():
        vals = values[list(idx)]
        if vals.numel() == 1:
            if legacy_singleton:
                means[gid] = 0.0
                stds[gid] = 1.0
            else:
                means[gid] = float(vals[0])
                stds[gid] = 0.0
        else:
            means[gid] = float(torch.mean(vals))
            stds[gid] = float(torch.std(vals))  # unbiased, matches core_algos
    return means, stds


def _normalize_by_group(
    values: torch.Tensor,
    slots: Mapping[Any, Sequence[int]],
    epsilon: float,
    tau: Optional[float],
    legacy_singleton: bool,
) -> Tuple[torch.Tensor, Dict[Any, float], Dict[Any, bool]]:
    """Group-relative normalisation. ``tau`` not None enables the CFA exit."""
    means, stds = _group_mean_std(values, slots, legacy_singleton)
    out = torch.zeros_like(values)
    masked: Dict[Any, bool] = {}
    for gid, idx in slots.items():
        std = stds[gid]
        if tau is not None and std < tau:
            masked[gid] = True
            continue  # A_k := 0 for this (group, axis)
        masked[gid] = False
        for i in idx:
            out[i] = (values[i] - means[gid]) / (std + epsilon)
    return out, stds, masked


def _percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def compute_axis_advantage(
    per_axis_rewards: Mapping[str, Sequence[float]],
    group_index: Sequence[Any],
    mode: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
    tau: Optional[float] = None,
    epsilon: Optional[float] = None,
    config: Optional[AxisAdvantageConfig] = None,
    cfa_mask: Optional[bool] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compose one scalar advantage per trajectory from its per-axis rewards.

    Parameters
    ----------
    per_axis_rewards
        axis name -> per-trajectory reward, all of length ``batch_size``.
    group_index
        per-trajectory prompt-group id (verl's ``uid``).
    mode / weights / tau / epsilon
        Explicit overrides; anything left as ``None`` comes from ``config``
        (default: :func:`config_from_env`).

    Returns
    -------
    (advantages, diagnostics)
        ``advantages`` is a float64 tensor of shape ``(batch_size,)``.
        ``diagnostics`` carries the collapse metrics described in
        docs/plan.md §P5.3'.2b: ``acr``, ``acr_per_axis``, ``active_group_frac``,
        ``within_group_std``, ``axis_grad_share``.
    """
    cfg = config or config_from_env()
    mode = (mode or cfg.mode).strip().lower()
    if mode not in ADV_MODES:
        raise ValueError(f"unknown advantage mode {mode!r}; expected one of {ADV_MODES}")
    tau = cfg.tau if tau is None else float(tau)
    epsilon = cfg.epsilon if epsilon is None else float(epsilon)
    cfa_mask = cfg.cfa_mask if cfa_mask is None else bool(cfa_mask)

    def _weight(axis: str) -> float:
        if weights is not None and axis in weights:
            return float(weights[axis])
        return cfg.weight(axis)

    if not per_axis_rewards:
        raise ValueError("per_axis_rewards is empty")
    axes = {
        k: torch.as_tensor(np.asarray(v, dtype=np.float64)).reshape(-1)
        for k, v in per_axis_rewards.items()
    }
    sizes = {k: int(v.numel()) for k, v in axes.items()}
    bsz = next(iter(sizes.values()))
    if any(s != bsz for s in sizes.values()):
        raise ValueError(f"per-axis reward lengths disagree: {sizes}")
    if len(group_index) != bsz:
        raise ValueError(f"group_index has {len(group_index)} entries, expected {bsz}")

    order, slots = _group_slots(group_index)
    n_groups = len(order)
    axis_names = list(axes.keys())

    # Reward-side per-axis within-group std is mode independent: it is the
    # collapse reading the literature compares (ACR, arXiv 2605.21125).
    reward_stds: Dict[str, Dict[Any, float]] = {}
    for name in axis_names:
        _, stds = _group_mean_std(axes[name], slots, legacy_singleton=False)
        reward_stds[name] = stds

    per_axis_adv: Dict[str, torch.Tensor] = {}
    masked_groups: Dict[str, Dict[Any, bool]] = {}

    if mode == "scalar":
        # Stock DAPO/GRPO control: weighted sum FIRST, one normalisation after.
        total = torch.zeros(bsz, dtype=torch.float64)
        for name in axis_names:
            total = total + _weight(name) * axes[name]
        adv, _, _ = _normalize_by_group(
            total, slots, epsilon=epsilon, tau=None, legacy_singleton=True
        )
        # Kept as a single pseudo-axis so the diagnostics schema is uniform.
        per_axis_adv["scalar"] = adv
        masked_groups["scalar"] = {gid: False for gid in order}
    elif mode == "signadv":
        # Sign advantage (arXiv 2605.07689): fixed reference, no group stats.
        if AXIS_T not in axes:
            raise ValueError("mode=signadv needs the task axis 'T' in per_axis_rewards")
        adv = 2.0 * axes[AXIS_T] - 1.0
        per_axis_adv[AXIS_T] = adv
        masked_groups[AXIS_T] = {gid: False for gid in order}
    else:
        # gdpo / cfa / cfa_batchnorm: normalise each component within its group.
        # A0g8: the mask is a knob now, not a mode property. Keeping the mode
        # test as well means gdpo can never accidentally acquire it.
        use_mask = cfa_mask and mode in ("cfa", "cfa_batchnorm")
        for name in axis_names:
            adv_k, _, masked = _normalize_by_group(
                axes[name],
                slots,
                epsilon=epsilon,
                tau=tau if use_mask else None,
                legacy_singleton=False,
            )
            per_axis_adv[name] = adv_k
            masked_groups[name] = masked

    advantages = torch.zeros(bsz, dtype=torch.float64)
    for name, adv_k in per_axis_adv.items():
        w = 1.0 if name == "scalar" else _weight(name)
        advantages = advantages + w * adv_k

    if mode in ("gdpo", "cfa_batchnorm"):
        # GDPO applies a batch-level normalisation after summing components.
        # cfa_batchnorm exists only to isolate this factor from the mask.
        batch_std = float(torch.std(advantages)) if bsz > 1 else 0.0
        advantages = (advantages - float(torch.mean(advantages))) / (batch_std + epsilon)

    diagnostics = _collapse_diagnostics(
        mode=mode,
        axis_names=axis_names,
        per_axis_adv=per_axis_adv,
        weight_fn=_weight,
        reward_stds=reward_stds,
        masked_groups=masked_groups,
        slots=slots,
        order=order,
        n_groups=n_groups,
        tau=tau,
        batch_size=bsz,
    )
    return advantages, diagnostics


def _collapse_diagnostics(
    mode: str,
    axis_names: Sequence[str],
    per_axis_adv: Mapping[str, torch.Tensor],
    weight_fn,
    reward_stds: Mapping[str, Mapping[Any, float]],
    masked_groups: Mapping[str, Mapping[Any, bool]],
    slots: Mapping[Any, Sequence[int]],
    order: Sequence[Any],
    n_groups: int,
    tau: float,
    batch_size: int,
) -> Dict[str, Any]:
    """Collapse metrics for docs/plan.md §P5.3'.2b (written to collapse.jsonl).

    ``acr`` is advantage-side: the fraction of prompt groups where every axis
    contributes ``|w_k * A_k| < ZERO_ADV_ATOL`` for every member, i.e. groups
    that produce no gradient at all. This is where ``cfa`` and ``gdpo`` visibly
    part company on a *near*-saturated group (std small but nonzero): ``gdpo``
    divides by ``std + eps`` and amplifies the residual noise, ``cfa`` masks it.

    ``acr_per_axis[k]`` is reward-side: the fraction of groups whose axis-``k``
    within-group std is below ``tau`` (= ``1 - active_group_frac[k]``), so it is
    comparable across modes and against the literature's ACR.
    """
    acr_per_axis: Dict[str, float] = {}
    active_group_frac: Dict[str, float] = {}
    within_group_std: Dict[str, Dict[str, float]] = {}
    masked_frac: Dict[str, float] = {}
    for name in axis_names:
        stds = [reward_stds[name][gid] for gid in order]
        below = sum(1 for s in stds if s < tau)
        acr_per_axis[name] = below / n_groups if n_groups else 0.0
        active_group_frac[name] = 1.0 - acr_per_axis[name]
        within_group_std[name] = {
            "mean": float(np.mean(stds)) if stds else 0.0,
            "p10": _percentile(stds, 10),
            "p90": _percentile(stds, 90),
        }
        mg = masked_groups.get(name, {})
        masked_frac[name] = (
            sum(1 for gid in order if mg.get(gid)) / n_groups if n_groups else 0.0
        )

    # axis_grad_share: which axis actually drives |A_total|.
    abs_share: Dict[str, float] = {}
    total_abs = 0.0
    for name, adv_k in per_axis_adv.items():
        w = 1.0 if name == "scalar" else weight_fn(name)
        s = float(torch.sum(torch.abs(w * adv_k)))
        abs_share[name] = s
        total_abs += s
    axis_grad_share = {
        name: (s / total_abs if total_abs > ZERO_ADV_ATOL else 0.0)
        for name, s in abs_share.items()
    }

    collapsed = 0
    for gid in order:
        idx = list(slots[gid])
        any_grad = False
        for name, adv_k in per_axis_adv.items():
            w = 1.0 if name == "scalar" else weight_fn(name)
            if float(torch.max(torch.abs(w * adv_k[idx]))) > ZERO_ADV_ATOL:
                any_grad = True
                break
        if not any_grad:
            collapsed += 1

    return {
        "mode": mode,
        "n_groups": int(n_groups),
        "batch_size": int(batch_size),
        "tau": float(tau),
        "acr": collapsed / n_groups if n_groups else 0.0,
        "acr_per_axis": acr_per_axis,
        "active_group_frac": active_group_frac,
        "within_group_std": within_group_std,
        "axis_grad_share": axis_grad_share,
        "masked_group_frac": masked_frac,
    }


# ---------------------------------------------------------------------------
# collapse.jsonl
# ---------------------------------------------------------------------------
_COLLAPSE_STEP = {"n": 0}


def log_collapse_diagnostics(
    diagnostics: Mapping[str, Any],
    path: Optional[str] = None,
    step: Optional[int] = None,
) -> Optional[str]:
    """Append one diagnostics line to ``collapse.jsonl``.

    Path comes from ``LNSDL_COLLAPSE_LOG`` when not passed; returns the path
    written to, or ``None`` when logging is off. Failures are swallowed -- a
    diagnostics sink must never take down a training run.
    """
    path = path or os.environ.get("LNSDL_COLLAPSE_LOG")
    if not path:
        return None
    if step is None:
        _COLLAPSE_STEP["n"] += 1
        step = _COLLAPSE_STEP["n"]
    record = {"step": int(step)}
    record.update(dict(diagnostics))
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        return None
    return path


# ---------------------------------------------------------------------------
# adv_batch.jsonl -- raw per-trajectory rewards for the offline estimator contrast
# ---------------------------------------------------------------------------
# Fields dumped per trajectory. ``acc`` is the headline plan_acc (= r_task before
# any shaping) and is what the offline contrast correlates the advantages
# against; the rest are exactly the inputs ``assemble_axis_rewards`` consumes,
# so every mode in ADV_MODES can be replayed on the dumped batch. The protocol
# observations (n_tool_calls / stacked / nonterm) ride along because they are
# what the per-mode readings have to be stratified by.
DUMP_FIELDS = (
    "acc",
    "score",
    "r_task",
    "r_goal",
    "r_faithful",
    "r_struct",
    "r_dense_raw",
    "nonterm",
    "n_tool_calls",
    "stacked",
    "task_gated",
    "p_format",
    "p_nonterm",
    "p_cost",
    # Stratifiers. ``skip_sample`` in particular: the skip stratum is 2% of the
    # official train distribution and behaves nothing like the rest
    # (docs/debug_rl.md), so any per-mode reading pooled over both is unreadable.
    "skip_sample",
    "trivial_structure",
    "reward_patch_stack",
)

_DUMP_STEP: Dict[str, int] = {}

#: Dump sites. ``pre_filter`` is the whole generation batch as it comes out of
#: rollout+reward; ``post_filter`` is what survives DAPO dynamic sampling and is
#: what the optimizer actually sees. They answer different questions and must not
#: be pooled -- see :func:`log_axis_batch`.
DUMP_STAGES = ("pre_filter", "post_filter")


def log_axis_batch(
    reward_fields: Mapping[str, Sequence[float]],
    group_index: Sequence[Any],
    path: Optional[str] = None,
    step: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
    stage: str = "post_filter",
) -> Optional[str]:
    """Append one rollout batch's raw per-trajectory rewards to ``adv_batch.jsonl``.

    docs/plan.md §P5.3'.2b protocol E: the three-mode contrast is far cheaper to
    read offline than on the GPU, but only if a *fixed* rollout batch is on disk
    with the reward breakdown intact -- ``collapse.jsonl`` records aggregates for
    the one mode that actually ran, which cannot be replayed. This dump is that
    fixed batch. It is mode independent on purpose: the same file supports every
    entry in :data:`ADV_MODES`.

    ``stage`` matters more than it looks. DAPO dynamic sampling keeps only the
    prompt groups that *have* accuracy variance, so a ``post_filter`` batch is
    conditioned on exactly the thing an advantage-collapse rate measures: ACR on
    it is near zero on the task axis by construction, and the CFA-vs-GDPO mask
    difference is invisible precisely where it would matter. Read collapse
    metrics on ``pre_filter``; read "what did the optimizer actually get" on
    ``post_filter``. Each stage keeps its own step counter so the two line up.

    Path comes from ``LNSDL_ADV_DUMP`` when not passed; returns the path written
    to, or ``None`` when the dump is off. Failures are swallowed -- like
    :func:`log_collapse_diagnostics`, a diagnostics sink must never take down a
    training run.
    """
    path = path or os.environ.get("LNSDL_ADV_DUMP")
    if not path:
        return None
    if step is None:
        step = _DUMP_STEP.get(stage, 0) + 1
        _DUMP_STEP[stage] = step

    fields: Dict[str, List[float]] = {}
    for name in DUMP_FIELDS:
        if name in reward_fields:
            fields[name] = [float(v) for v in np.asarray(reward_fields[name]).reshape(-1)]
    record: Dict[str, Any] = {
        "step": int(step),
        "stage": str(stage),
        "uid": [str(g) for g in group_index],
        "fields": fields,
    }
    if extra:
        record.update(dict(extra))
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        return None
    return path


def overlong_filter_enabled() -> bool:
    """DAPO "overlong filtering" switch (env ``LNSDL_OVERLONG_FILTER``, default off).

    The DAPO recipe masks the loss of trajectories that hit the response cap:
    their reward reflects the truncation, not the content, so the gradient is
    noise. This repo only carried DAPO's *soft* overlong penalty, and that one
    is added to ``token_level_rewards``, which the per-axis modes never read
    (docs/debug_rl.md §1.8) -- so until 2026-09-04 truncated trajectories were
    trained on at full weight. In s24 (VIKI-RL) the truncated share climbed
    from 0.4% to 20-25% by step 20 and every truncated sample was a repetition
    loop, so here truncation == runaway. Off by default: the s24/s30 recipes
    are frozen and must stay reproducible.
    """
    return _envb("LNSDL_OVERLONG_FILTER", False)


def apply_overlong_filter(
    advantages: torch.Tensor,
    response_real_len: torch.Tensor,
    response_cap: int,
    diagnostics: Dict[str, Any],
) -> torch.Tensor:
    """Zero the scalar advantage of every trajectory that hit ``response_cap``.

    Only the advantage is zeroed: the tokens stay in the PPO/KL denominators, so
    the KL anchor keeps (gently) pushing the looped tokens back toward the
    reference. Records ``overlong_masked_frac`` / ``overlong_masked_n`` in
    ``diagnostics`` (always, also when the switch is off, so the share of
    truncated trajectories is visible in every run).
    """
    truncated = response_real_len >= int(response_cap)
    n = int(truncated.sum())
    diagnostics["overlong_masked_n"] = n if overlong_filter_enabled() else 0
    diagnostics["overlong_truncated_frac"] = float(n) / max(1, int(truncated.numel()))
    diagnostics["overlong_filter_on"] = 1.0 if overlong_filter_enabled() else 0.0
    if not overlong_filter_enabled() or n == 0:
        return advantages
    out = advantages.clone()
    out[truncated.to(out.device)] = 0.0
    return out


def flat_metrics(diagnostics: Mapping[str, Any], prefix: str = "lnsdl_adv") -> Dict[str, float]:
    """Flatten diagnostics into scalar metrics for the trainer's logger."""
    out: Dict[str, float] = {
        f"{prefix}/acr": float(diagnostics.get("acr", 0.0)),
        f"{prefix}/n_groups": float(diagnostics.get("n_groups", 0)),
    }
    for key in ("overlong_masked_n", "overlong_truncated_frac", "overlong_filter_on"):
        if key in diagnostics:
            out[f"{prefix}/{key}"] = float(diagnostics[key])
    for key in ("acr_per_axis", "active_group_frac", "axis_grad_share", "masked_group_frac"):
        for axis, val in (diagnostics.get(key) or {}).items():
            out[f"{prefix}/{key}/{axis}"] = float(val)
    for axis, stats in (diagnostics.get("within_group_std") or {}).items():
        for stat, val in stats.items():
            out[f"{prefix}/within_group_std/{axis}/{stat}"] = float(val)
    return out
