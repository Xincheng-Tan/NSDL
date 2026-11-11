"""Reward function for L-NSDL VIKI-L2-train *scoped* RL (P5.3, DAPO).

This is the deterministic, trajectory-level reward for the agentic + tool
L-NSDL policy, specified in ``docs/dataset_and_train.md §E.1.2`` and
``docs/plan.md §P5.3``. It is the [NOW] scoped version (VIKI-L2 official
train only, no ManiSkill) — distinct from the [EXT] P9 general reward in
``enactgraph.py`` / ``dataset_and_train.md §E``.

DEFAULT RECIPE since 2026-08-03 (P5.3′.1, ``docs/plan.md §P5.3′.1``) — the
patch stack is OFF and the reward has **no negative term at all**:

    R_total = 0.40·R_task + 0.30·R_goal + 0.15·R_faithful + 0.15·R_struct

Why the whole penalty stack went away rather than being retuned: every patch in
it (P_format, P_nonterm, P_cost, the failure-band dense scale, the score cap,
the E6 stack gate) was a symptom patch for one root cause — protocol dirt in the
rollout channel — and the E5/E6/E8 arc showed each of them fixed its own
mechanism while none moved reportable accuracy (docs/RL_debug.md; the recorded
79–86% protocol share of failure-band score variance). Protocol hygiene now
lives in the rollout-side first-tool-call truncation guardrail (default ON), so
the reward does not need to price it. With no negative term, it is
*structurally impossible* for protocol penalties to dominate failure-band
variance again — that is the property the paper's method section claims, and it
is a consequence of the composition above, not of a tuning choice.

The per-axis fields are the advantage source (``lnsdl_advantage.py``:
axis A = (R_goal+R_faithful)/2, axis B = R_struct, axis T = R_task); ``score``
remains as the scalar-mode fallback and for logging, not as the sole signal.

Set ``LNSDL_REWARD_PATCH_STACK=1`` to restore the full pre-2026-08-03 recipe
verbatim (ablation / reproduction of the recorded E5–E8 runs). Everything from
here to the "Deliberately EXCLUDED" paragraph describes that legacy mode:

    R_total = 0.40·R_task + 0.30·R_goal + 0.15·R_faithful + 0.15·R_struct
              + LNSDL_R_TOOL_USE·R_tool_use
              + P_format + P_nonterm + P_cost          (penalties are <= 0)

    R_task    : 1 iff final [PLAN] exact-matches GT DAG (headline plan_acc).
    R_goal    : Jaccard(declared [GOAL].active, GT active) — DENSE, OOD lever.
    R_faithful: 1 - (|grasp \\ active| + |active \\ grasp|)/|grasp ∪ active|,
                active = the model's CORRECTED (last) [GOAL].active.
    R_struct  : 1 iff opval / custody / schedule feasibility pass.

    P_format  : -0.20 per hit (stacked >1 [TOOL_CALL]/turn, bad tool JSON,
                emitted != executed), capped at -0.40. On-policy root-cause for
                the stacking that the P5.6 inference-side suppress patch only
                masks (docs/debug.md §5). E6 (docs/RL_debug.md §3(3)): now a
                TIE-BREAKER (recommended 0.05/cap 0.10), because the additive
                penalty alone can never out-weigh W_TASK — see the stack gate.
    P_nonterm : -0.30 if the trajectory never emits a final [PLAN]
                (stopped_reason == max_tool_rounds) — kills the P5.2.4 tool
                explosion mode (docs/debug.md §9).
    P_cost    : -0.02 per tool call, capped at -0.20 (selective > frequent).
    R_tool_use: optional, default-off positive shaping for cleanly consumed tool
                observations. It is task-blind, success-gated, protocol-clean
                gated, and reads only final structured GOAL/PLAN fields.

    Anti-hacking guard (added after the 2026-06-30 75-step scoped run; scope
    corrected 2026-07-05 per docs/RL_debug.md §3.2): if R_task=0, the mechanical
    axis R_struct is softly down-scaled and the final score is capped. R_struct
    is a strict relaxation of R_task that co-saturates to 1 in the failure band,
    so {R_task=0, R_struct=1} is the reward-hacking surface the scale targets.
    The bottleneck SEMANTIC axes R_goal (取件) and R_faithful stay at FULL budget
    in the failure band — the prior uniform scaling squeezed R_goal from 0.30 to
    0.21 while penalties stayed unscaled, driving its failure-band covariance
    negative (anti-goal advantage; AUDIT-B). Set
    LNSDL_TASK_FAIL_SCALE_STRUCT_ONLY=0 to restore the old uniform scaling.

    Stack gate (E6, added 2026-07-06 per docs/RL_debug.md §3(3)): a protocol-dirty
    tool-call channel (stacked turn OR emitted != executed) HARD-GATES the task
    credit — the W_TASK·R_task term is dropped from ``score`` for that trajectory.
    Rationale: with an additive P_format capped at -0.40 < W_TASK=0.40, a
    "stacked-but-recovered" trajectory (the P5.6 suppress patch salvages the first
    [TOOL_CALL], so R_task still =1) nets 0.40+dense-0.40 > a clean loser, so the
    group-relative advantage FAVORS stacking and RL amplifies it (E5: skip
    stacked 0.333→1.000). Gating the SCORE term flips the ordering — a clean
    terminating winner scores strictly above a stacked winner by ~W_TASK — while
    the reported ``acc``/``r_task`` stay TRUE (headline plan_acc and the
    acc-variance dynamic sampling are untouched; only the gradient is re-ordered).
    Set LNSDL_STACK_GATE_TASK=0 to disable (pre-E6 behavior).

Deliberately EXCLUDED here (only in [EXT] P9, see §E.1.3): ToolRL-style
per-tool-chain matching reward (this was the P5.2.4 crash root cause), raw
tool-count bonuses, free-CoT semantic matching, and any hard validator gate.

Return value is a dict so the DAPO reward manager can store the breakdown and
so ``algorithm.filter_groups.metric=acc`` (dynamic sampling) can filter on the
binary task-success signal ``acc`` (= R_task). ``score`` is R_total.

Integration contract (multi-turn rollout):
    ``solution_str`` is the model's response text. For a multi-turn agentic
    rollout it is the concatenation of all assistant turns (think / segments /
    [TOOL_CALL] / [PLAN]); the LAST [GOAL] and [PLAN] win, which is exactly the
    *corrected* goal R_faithful needs. Structured per-turn signals
    (stopped_reason, emitted/executed tool-call counts, stacked turns) are read
    from ``extra_info['trace']`` when the rollout harness (tool_runtime adapter)
    populates it; otherwise they are recovered heuristically from the text.

Env overrides (ablation): ``LNSDL_REWARD_PATCH_STACK`` (default 0 = simplified
P5.3′.1 recipe, no negative term; 1 = full pre-2026-08-03 patch stack),
``LNSDL_W_TASK/GOAL/FAITHFUL/STRUCT/GOAL_PLAN/STRUCT_PHI/STRUCT_SKETCH``
(2026-09-06 A0p recipe: TASK .40 / GOAL .30 / GOAL_PLAN .15 / STRUCT_SKETCH .15,
FAITHFUL, STRUCT and STRUCT_PHI at 0 and logged only).
The remaining switches only take effect when the patch stack is ON:
``LNSDL_R_TOOL_USE``, ``LNSDL_P_FORMAT/NONTERM/COST``,
``LNSDL_P_FORMAT_CAP``, ``LNSDL_P_COST_CAP``,
``LNSDL_TASK_FAIL_DENSE_SCALE``, ``LNSDL_TASK_FAIL_SCORE_CAP``,
``LNSDL_TASK_FAIL_SCALE_STRUCT_ONLY`` (down-scale only R_struct in the failure
band, keeping R_goal/R_faithful whole; default on),
``LNSDL_STACK_GATE_TASK`` (E6 stack gate),
``LNSDL_COND_PCOST`` (T1: charge tool cost only on trivial+solved samples
without a skip-decision structure; default on, requires the
``trivial_structure`` ground_truth flag).

Protocol *observations* (``n_tool_calls``, ``stacked``, ``nonterm``,
``emitted != executed``) are reported in both modes — they are diagnostics and,
in the case of ``nonterm``, an input to the CFA axis-A guard. They stop being
*priced* when the patch stack is off; the penalty keys are still returned as
exactly 0.0 so ``scripts/analyze_lnsdl_score_variance.py`` keeps parsing and
reads the protocol share of failure-band variance as a literal 0.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

# Reuse the audited deterministic pipeline (eval_single simulation + validator)
# from the enactgraph reward; do NOT re-implement the executor here.
from . import enactgraph as _eg


# ---------------------------------------------------------------------------
# Weights (env-overridable for ablation; defaults = docs/dataset_and_train §E.1.2)
# ---------------------------------------------------------------------------
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


W_TASK = _envf("LNSDL_W_TASK", 0.40)
W_GOAL = _envf("LNSDL_W_GOAL", 0.30)
W_FAITHFUL = _envf("LNSDL_W_FAITHFUL", 0.15)
W_STRUCT = _envf("LNSDL_W_STRUCT", 0.15)
# Plan-side goal coverage: Jaccard(entities the PLAN grasps, GT active). It is
# R_goal read off the executed plan instead of the declared [GOAL], so it is
# oriented against GT like every other axis. R_faithful (declared-vs-plan
# consistency) is not: on v3.4.1 LOCUS eval it pays ~0.97 to "consistently
# executed a wrong goal" (186/560 rows) and ~0.14 to "wrong declaration, right
# plan" (11 rows, acc .727) -- debug_rl §6.5. Default weight 0 keeps every
# recorded run's score byte-identical; the term is always logged.
W_GOAL_PLAN = _envf("LNSDL_W_GOAL_PLAN", 0.0)
# Structure axis judged by Phi (debug_rl §6.7): 1 iff the final [PLAN] is the
# op multiset Phi returned from the last accepted `schedule` call. The residual
# structural failure of the main configuration is the model dropping a
# Phi-inserted op while transcribing (v3.4.1 LOCUS eval: 95/560 rows, all in
# ood_m/ood_x, acc 0/95); the rule validator behind R_struct is blind to it
# whenever the surviving plan is still mechanically legal. Default weight 0.
W_STRUCT_PHI = _envf("LNSDL_W_STRUCT_PHI", 0.0)
# Structure axis on the sketch (plan.md A0p / facts TR5b, 2026-09-06): 1 iff
# Phi ACCEPTED the model's last `schedule` call. Once the main configuration
# submits Phi's own schedule as the final plan, the [PLAN] transcription no
# longer carries weight (R_struct_phi is kept for logging only) and the one
# residual structural failure is "sketch outside Phi's domain -> rejected ->
# fall back to the self-produced plan". This term prices exactly that.
# Default weight 0 keeps every recorded run byte-identical; s42's successor
# runs set 0.15 through the launcher.
W_STRUCT_SKETCH = _envf("LNSDL_W_STRUCT_SKETCH", 0.0)
# debug_rl.md §7.8 P1 (2026-09-07). s46 showed R_struct(sketch) paying 1.0 to
# (a) the empty sketch on a non-null-GT row (71/560 eval rows: `schedule` with
# op_ids=[] is trivially accepted) and (b) Phi-accepted sketches the LOCUS
# simulator rejects at step 2 (33 ood_m rows). With no cost term either, the
# policy learned to call `schedule` again after an accepted call and to loop
# `[PLAN] []]` (43 rows). Three gated fixes, all default-off:
#   STRUCT_SKETCH_EXEC: empty accepted sketch pays 1 only on a null-GT row;
#     non-empty accepted sketch must also survive plan_expansion_ok on LOCUS rows.
#   P_FORMAT_V2: per-hit price for [PLAN]/[TOOL_CALL] in one assistant turn,
#     >1 [PLAN] block in one turn, a stray bracket after a [PLAN] JSON, or a
#     `[THINK]` pseudo-block; capped.
#   P_SCHED_REPEAT: per extra `schedule` call issued after an accepted one; capped.
STRUCT_SKETCH_EXEC = _envb("LNSDL_STRUCT_SKETCH_EXEC", False)
P_FORMAT_V2 = _envf("LNSDL_P_FORMAT_V2", 0.0)
P_FORMAT_V2_CAP = _envf("LNSDL_P_FORMAT_V2_CAP", 0.30)
P_SCHED_REPEAT = _envf("LNSDL_P_SCHED_REPEAT", 0.0)
P_SCHED_REPEAT_CAP = _envf("LNSDL_P_SCHED_REPEAT_CAP", 0.15)
# Reward source for the plan-reading terms (R_task, R_plan). `phi_schedule`
# (default since 2026-09-11) scores what the A0p main configuration submits:
# the ops of Phi's last accepted `schedule`, falling back to the model's
# transcribed [PLAN] only when no schedule was accepted (the eval fallback).
# `plan` is the recorded s46/s55/s58 behaviour (both terms on the transcribed
# [PLAN]) and is kept for reproducing those runs.
def _plan_source(env_name: str) -> str:
    val = (os.environ.get(env_name) or "phi_schedule").strip().lower()
    if val not in ("plan", "phi_schedule"):
        raise ValueError(f"{env_name} must be plan or phi_schedule, got {val!r}")
    return val


R_TASK_SOURCE = _plan_source("LNSDL_R_TASK_SOURCE")
R_PLAN_SOURCE = _plan_source("LNSDL_R_PLAN_SOURCE")

# P5.3′.1 (docs/plan.md §P5.3′.1, 2026-08-03). OFF by default: the reward is the
# four positive axes and nothing else. Everything below this line is the legacy
# patch stack, kept verbatim so the recorded E5–E8 runs stay reproducible, and
# reachable only via LNSDL_REWARD_PATCH_STACK=1.
#
# This is a deletion, not a retune. Each patch fixed the mechanism it targeted
# and none moved reportable accuracy (E6 stopped stacking saturation, E8 lowered
# stacking behaviour, neither raised nonskip acc), so the stack was treating
# symptoms of rollout protocol dirt. That is now handled by the first-tool-call
# truncation guardrail on the rollout side, which is a behavioural fix rather
# than a price signal. Removing every negative term also removes, by
# construction rather than by tuning, the failure-band variance regime where
# protocol penalties owned 79–86% of the score variance.
REWARD_PATCH_STACK = _envb("LNSDL_REWARD_PATCH_STACK", False)

R_TOOL_USE = _envf("LNSDL_R_TOOL_USE", 0.0)

P_FORMAT = _envf("LNSDL_P_FORMAT", 0.20)        # per-hit magnitude (subtracted)
P_FORMAT_CAP = _envf("LNSDL_P_FORMAT_CAP", 0.40)
P_NONTERM = _envf("LNSDL_P_NONTERM", 0.30)
P_COST = _envf("LNSDL_P_COST", 0.02)            # per tool call
P_COST_CAP = _envf("LNSDL_P_COST_CAP", 0.20)
TASK_FAIL_DENSE_SCALE = _envf("LNSDL_TASK_FAIL_DENSE_SCALE", 0.70)
TASK_FAIL_SCORE_CAP = _envf("LNSDL_TASK_FAIL_SCORE_CAP", 0.45)

# docs/RL_debug.md §3.2 (2026-07-05 AUDIT-B): the failure-band down-scale must
# hit ONLY the mechanical relaxation R_struct (co-saturated to 1, the hacking
# surface), not the bottleneck semantic axis R_goal. When on (default), R_goal
# and R_faithful keep full budget in the failure band and only R_struct is
# scaled by TASK_FAIL_DENSE_SCALE. When off, all dense terms scale uniformly
# (the pre-2026-07-05 behavior).
TASK_FAIL_SCALE_STRUCT_ONLY = _envb("LNSDL_TASK_FAIL_SCALE_STRUCT_ONLY", True)

# T1 (docs/RL_debug.md §8): conditional P_cost. On the majority of solved,
# non-trivial samples, tool calls are the MEANS to the correct plan, so taxing
# them creates a systematic "use fewer tools" advantage (the recorded
# corr(score,tools) < 0). When enabled, tool cost is charged ONLY on samples
# that are structurally trivial, have no skip-decision structure, and are
# already solved (R_task==1), where extra tools are genuinely wasteful.
# Requires the task-blind `trivial_structure` flag in ground_truth
# (scripts/prepare_rl_data.py); when the flag is absent or this is disabled,
# falls back to always charging (old behavior).
COND_PCOST = _envb("LNSDL_COND_PCOST", True)

# E6 (docs/RL_debug.md §3(3)): hard-gate the task credit on a protocol-dirty
# tool-call channel. When on (default), a trajectory with a stacked turn OR
# emitted != executed loses its W_TASK·R_task contribution to `score` (but keeps
# the TRUE reported acc/r_task). This is the root fix the additive P_format
# cannot be: since P_FORMAT_CAP (0.40) <= W_TASK (0.40), a stacked-but-suppress-
# recovered winner would otherwise out-score a clean loser and RL would amplify
# stacking (E5: skip stacked 0.333->1.000). =0 restores the pre-E6 behavior.
STACK_GATE_TASK = _envb("LNSDL_STACK_GATE_TASK", True)


# ---------------------------------------------------------------------------
# Robust block extraction for the agentic format
# ---------------------------------------------------------------------------
# enactgraph._parse_cot_blocks requires a newline after [GOAL]/[PLAN] and only
# handles the single-turn three-block format. The agentic trajectory writes
# `[GOAL] {...}` on the same line and may repeat blocks across turns, so we use
# a brace/bracket matcher that takes the LAST occurrence (= corrected output).

_norm = _eg._norm_str


def _balanced_json_after(text: str, start: int, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the balanced {...} / [...] substring beginning at/after ``start``."""
    i = text.find(open_ch, start)
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def _extract_last_block(text: str, tag: str, open_ch: str, close_ch: str):
    """Parse the JSON value following the LAST ``[tag]`` marker. None on failure."""
    marker = f"[{tag}]"
    pos = text.rfind(marker)
    if pos < 0:
        return None
    blob = _balanced_json_after(text, pos + len(marker), open_ch, close_ch)
    if blob is None:
        return None
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_goal(text: str) -> Optional[dict]:
    g = _extract_last_block(text, "GOAL", "{", "}")
    return g if isinstance(g, dict) else None


def _extract_plan(text: str) -> Optional[List[dict]]:
    p = _extract_last_block(text, "PLAN", "[", "]")
    if not isinstance(p, list):
        return None
    return [op for op in p if isinstance(op, dict)]


def _goal_active_set(goal: Optional[dict]) -> Set[str]:
    if not isinstance(goal, dict):
        return set()
    items: Set[str] = set()
    active = goal.get("active")
    if isinstance(active, list):
        for x in active:
            if isinstance(x, str):
                items.add(_norm(x))
            elif isinstance(x, dict):
                items.add(_norm(x.get("item") or x.get("name")))
    # Fall back to per-target status when 'active' is absent.
    if not items:
        for t in goal.get("targets") or []:
            if isinstance(t, dict) and _norm(t.get("status")) == "active":
                items.add(_norm(t.get("item")))
    items.discard("")
    return items


_ACQUIRE_OPS = {"acquire", "local_acquire", "load_carrier"}


def _plan_grasp_set(plan: Optional[List[dict]]) -> Set[str]:
    """Entities the PLAN actually grasps (acquire-family ops)."""
    if not plan:
        return set()
    g = {_norm(op.get("entity")) for op in plan if _norm(op.get("op_type")) in _ACQUIRE_OPS}
    g.discard("")
    return g


def _gt_active_set(gt: Dict[str, Any]) -> Set[str]:
    """GT active items from gt_goal_json (mirrors enactgraph r_goal_reward_v3).

    Falls back to the entities the GT schedule grasps when gt_goal_json is
    absent (bench-side JSON, offline re-scoring). On the RL probe the two sets
    are identical on 512/512 rows, so the fallback is exact, not a proxy.
    """
    raw = gt.get("gt_goal_json")
    if not raw:
        out = set()
        for ts in gt.get("time_steps") or []:
            actions = ts.get("actions") if isinstance(ts, dict) else None
            for act in (actions or {}).values():
                if isinstance(act, (list, tuple)) and len(act) >= 2 and str(act[0]) == "Grasp":
                    out.add(_norm(re.sub(r"_\d+$", "", str(act[1]))))
        out.discard("")
        return out
    try:
        goal = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, ValueError):
        return set()
    if not isinstance(goal, dict):
        return set()
    out = {_norm(x) for x in (goal.get("active") or [])}
    if not out:
        for t in goal.get("targets") or []:
            if isinstance(t, dict) and _norm(t.get("status")) == "active":
                out.add(_norm(t.get("item")))
    out.discard("")
    return out


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# ---------------------------------------------------------------------------
# Positive components
# ---------------------------------------------------------------------------
def _r_task(plan: Optional[List[dict]], gt: Dict[str, Any]) -> float:
    """1 iff final [PLAN] passes eval_single with step-ratio >= 0.99 (= plan_acc).

    The null-GT branch mirrors LOCUS ``compute_locus_l2_acc``: when the ground
    truth prescribes no steps at all, the correct answer *is* the empty plan, so
    emitting nothing scores 1 and emitting anything scores 0. Without it both
    ends are inverted -- ``not plan`` returned 0 for the right answer, and the
    ``len_gt == 0`` shortcut below handed 1.0 to any plan that happened to
    survive ``_simulate``. On ``ramp63k`` that is 136/900 prompts (15.1%),
    concentrated in the s1/s2 rows, and it distorts per-cell difficulty by up to
    17 points. VIKI-L2 has no such rows (0 empty ``time_steps`` in all 10,214
    train/val/test rows), so this branch is a verified no-op there and needs no
    ``data_source`` split.
    """
    if not (gt.get("time_steps") or []):
        return 0.0 if plan else 1.0
    if not plan:
        return 0.0
    try:
        passed, len_pred, len_gt = _eg._simulate(plan, gt)
    except Exception:
        return 0.0
    if not passed:
        return 0.0
    if len_gt == 0 or len_pred == 0:
        return 1.0
    return 1.0 if (len_gt / len_pred) >= 0.99 else 0.0


def _r_struct(goal: Optional[dict], plan: Optional[List[dict]], gt: Dict[str, Any]) -> float:
    """1 iff the detection-only validator finds zero opval/custody issues.

    Reuses the audited validator path in enactgraph by feeding it a normalized
    `[GOAL]\\n...\\n[PLAN]\\n...` string (its regex needs the newline form).
    """
    if not plan:
        # Mirrors the null-GT branch of _r_task / compute_locus_l2_acc: when the
        # GT prescribes no steps, the empty plan is the correct, structurally
        # sound answer. Returning 0.0 here made axis B anti-ordered with axis T
        # on every LOCUS null-plan group (15.4% of the train pool; 100% of the
        # observed reversals in s30). VIKI has no empty time_steps, so this is
        # a byte-for-byte no-op there.
        return 1.0 if not (gt.get("time_steps") or []) else 0.0
    parts: List[str] = []
    if isinstance(goal, dict):
        parts.append("[GOAL]\n" + json.dumps(goal))
    parts.append("[PLAN]\n" + json.dumps(plan))
    synthetic = "\n".join(parts)
    try:
        validator = float(_eg.validator_pass_reward(synthetic, gt))
    except Exception:
        return 0.0
    # LOCUS rows (identified by locus_meta / locus_cell, GT fields VIKI never carries)
    # additionally require the plan to survive the scoring chain's execution
    # path (graph expansion + dispatch). The rule-detector validator is a
    # strict relaxation calibrated on VIKI shapes; on LOCUS depth it stays
    # silent on exactly the skip-step plans the benchmark simulator rejects
    # (open-before-deposit deps, carrier custody), so validator-only R_struct
    # would keep paying 1.0 on plans the judge scores as error rows. VIKI
    # rows keep the validator-only judge byte-for-byte.
    if validator > 0.0 and _is_locus_gt(gt):
        return validator if _eg.plan_expansion_ok(plan, gt) else 0.0
    return validator


def _is_locus_gt(gt: Any) -> bool:
    """LOCUS rows carry ``locus_meta`` (bench JSON) or ``locus_cell``/``locus_split``
    (the RL parquet written by prepare_rl_data.py, which never copies
    ``locus_meta``). VIKI rows carry neither."""
    if not isinstance(gt, dict):
        return False
    return any(gt.get(k) is not None for k in ("locus_meta", "locus_cell", "locus_split"))


def _phi_ops_from_trace(trace: Optional[dict]) -> Optional[List[dict]]:
    """Ops of the last accepted (ok=True) `schedule` result, or None if the
    trajectory never got one. Reads the RL trace (``tool_observations``) and,
    for offline use, the eval-harness trace (``tool_names``/``tool_results``)."""
    if not isinstance(trace, AbcMapping):
        return None
    last: Optional[Mapping[str, Any]] = None
    for obs in _trace_tool_observations(trace):
        if str(obs.get("name") or "").strip().lower() == "schedule" and obs.get("ok") is True:
            last = obs
    if last is None:
        names = trace.get("tool_names") or []
        results = trace.get("tool_results") or []
        for name, res in zip(names, results):
            if str(name).strip().lower() == "schedule" and isinstance(res, AbcMapping) and res.get("ok") is True:
                last = res
    if last is None:
        return None
    ops = last.get("ops") or []
    return [op for op in ops if isinstance(op, AbcMapping)]


def _op_key(op: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (_norm(op.get("op_type")), _norm(op.get("entity")), _norm(op.get("dest")), _norm(op.get("robot")))


def _r_struct_phi(plan: Optional[List[dict]], gt: Dict[str, Any], trace: Optional[dict]) -> float:
    """1 iff the emitted [PLAN] transcribes Phi's expansion op-for-op.

    Op identity is (op_type, entity, dest, robot) as a multiset; op_ids and
    deps are deliberately not compared (the model may renumber, and on the
    v3.4.1 eval the deps-aware variant only added 2 false zeros on correct
    plans). Without an accepted Phi result the structure was self-produced, so
    the axis pays 0 -- except for the empty plan on a null-GT row, which is the
    correct answer and mirrors _r_task / _r_struct.
    """
    phi_ops = _phi_ops_from_trace(trace)
    if phi_ops is None:
        return 1.0 if (not plan and not (gt.get("time_steps") or [])) else 0.0
    want = Counter(_op_key(op) for op in phi_ops)
    have = Counter(_op_key(op) for op in (plan or []) if isinstance(op, Mapping))
    return 1.0 if want == have else 0.0


def _last_schedule_ok(trace: Optional[dict]) -> Optional[bool]:
    """``ok`` of the LAST `schedule` call in the trajectory (accepted or not),
    None when schedule was never called. Reads the RL trace first and the
    eval-harness trace second, like ``_phi_ops_from_trace``."""
    last: Optional[bool] = None
    seen = False
    for obs in _trace_tool_observations(trace):
        if str(obs.get("name") or "").strip().lower() == "schedule":
            seen = True
            last = bool(obs.get("ok"))
    if not seen and isinstance(trace, AbcMapping):
        names = trace.get("tool_names") or []
        results = trace.get("tool_results") or []
        for name, res in zip(names, results):
            if str(name).strip().lower() == "schedule" and isinstance(res, AbcMapping):
                seen = True
                last = bool(res.get("ok"))
    return last if seen else None


def _r_struct_sketch(plan: Optional[List[dict]], gt: Dict[str, Any], trace: Optional[dict]) -> float:
    """1 iff Phi accepted the model's last sketch (last `schedule` call ok).

    A rejected last call pays 0 even if an earlier call was accepted: the
    plan the model ends up submitting is the one built from the last sketch.
    Never calling schedule pays 0 -- except the empty plan on a null-GT row,
    which is the correct answer and mirrors _r_task / _r_struct_phi.
    """
    ok = _last_schedule_ok(trace)
    null_gt = not (gt.get("time_steps") or [])
    if ok is None:
        return 1.0 if (not plan and null_gt) else 0.0
    if not ok:
        return 0.0
    if not STRUCT_SKETCH_EXEC:
        return 1.0
    phi_ops = _phi_ops_from_trace(trace) or []
    if not phi_ops:
        # op_ids=[] is always accepted; it is only the right sketch when the
        # goal already holds.
        return 1.0 if null_gt else 0.0
    if _is_locus_gt(gt):
        return 1.0 if _eg.plan_expansion_ok([dict(op) for op in phi_ops], gt) else 0.0
    return 1.0


def _r_task_phi(gt: Dict[str, Any], trace: Optional[dict]) -> float:
    """Diagnostic only (weight-free): R_task judged on Phi's own ops from the
    last accepted `schedule`, i.e. what the A0p main configuration would
    submit. -1.0 when the trajectory never got an accepted schedule."""
    phi_ops = _phi_ops_from_trace(trace)
    if phi_ops is None:
        return -1.0
    return _r_task([dict(op) for op in phi_ops], gt)


def _r_faithful(grasp: Set[str], active: Set[str]) -> float:
    """1 - symmetric-difference / union of grasp vs CORRECTED active set."""
    if not grasp and not active:
        return 1.0
    union = grasp | active
    if not union:
        return 1.0
    symdiff = len(grasp - active) + len(active - grasp)
    return max(0.0, 1.0 - symdiff / len(union))


# ---------------------------------------------------------------------------
# Protocol signals (penalties). Prefer extra_info['trace']; else parse text.
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(r"\[TOOL_CALL\]")
# Boundary markers a tool_runtime render may insert between assistant turns.
_TOOL_RESULT_RE = re.compile(r"\[TOOL_RESULT|tool_result|<\|im_start\|>tool|\"role\"\s*:\s*\"tool\"")


def _stacked_from_text(text: str) -> int:
    """Heuristic: count assistant segments with >=2 [TOOL_CALL] between tool turns."""
    segments = _TOOL_RESULT_RE.split(text)
    stacked = 0
    for seg in segments:
        if len(_TOOL_CALL_RE.findall(seg)) >= 2:
            stacked += 1
    return stacked


def _bad_tool_json_from_text(text: str) -> int:
    """Count [TOOL_CALL] blocks whose following JSON object does not parse."""
    bad = 0
    for m in _TOOL_CALL_RE.finditer(text):
        blob = _balanced_json_after(text, m.end(), "{", "}")
        if blob is None:
            bad += 1
            continue
        try:
            json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            bad += 1
    return bad


def _protocol_signals(
    text: str, trace: Optional[dict], plan: Optional[List[dict]]
) -> Tuple[int, int, bool, int, int]:
    """Return (n_tool, stacked_segments, nonterm, emit_exec_mismatch, bad_json).

    trace fields (authoritative when present): emitted_tool_call_count,
    executed_tool_call_count, stopped_reason, stacked_tool_call_turns.
    """
    if isinstance(trace, dict):
        emitted = int(trace.get("emitted_tool_call_count") or 0)
        executed = int(trace.get("executed_tool_call_count") or 0)
        stopped = str(trace.get("stopped_reason") or "")
        stacked = int(
            trace.get("stacked_tool_call_turns")
            or trace.get("stacked_block_count")
            or 0
        )
        if stacked == 0:  # backfill from text if harness did not count it
            stacked = _stacked_from_text(text)
        n_tool = emitted or len(_TOOL_CALL_RE.findall(text))
        mismatch = 1 if (emitted != executed) else 0
        nonterm = stopped == "max_tool_rounds"
        bad_json = _bad_tool_json_from_text(text)
        return n_tool, stacked, nonterm, mismatch, bad_json

    # Text-only fallback (no structured trace yet).
    n_tool = len(_TOOL_CALL_RE.findall(text))
    stacked = _stacked_from_text(text)
    bad_json = _bad_tool_json_from_text(text)
    # Proxy: tools were emitted but no final [PLAN] was produced -> never closed.
    nonterm = bool(n_tool) and not plan
    return n_tool, stacked, nonterm, 0, bad_json


_PLAN_TAG_RE = re.compile(r"\[PLAN\]")
_THINK_PSEUDO_RE = re.compile(r"\[THINK\]")


def _format_hits_v2(text: str) -> Dict[str, int]:
    """Task-blind protocol-shape violations, counted per assistant segment
    (segments are split at the tool-result markers the rollout renders)."""
    hits = {"plan_with_tool_call": 0, "multi_plan": 0, "stray_bracket": 0, "think_pseudo": 0}
    for seg in _TOOL_RESULT_RE.split(text or ""):
        n_plan = len(_PLAN_TAG_RE.findall(seg))
        if n_plan and _TOOL_CALL_RE.search(seg):
            hits["plan_with_tool_call"] += 1
        if n_plan > 1:
            hits["multi_plan"] += n_plan - 1
        pos = 0
        while True:
            m = _PLAN_TAG_RE.search(seg, pos)
            if not m:
                break
            pos = m.end()
            blob = _balanced_json_after(seg, pos, "[", "]")
            if blob is None:
                continue
            tail = seg[seg.find(blob, pos) + len(blob):]
            if tail.lstrip()[:1] in ("]", "}"):
                hits["stray_bracket"] += 1
        hits["think_pseudo"] += len(_THINK_PSEUDO_RE.findall(seg))
    return hits


def _schedule_repeat_count(trace: Optional[dict]) -> int:
    """`schedule` calls issued AFTER the first accepted one (each is a wasted
    round: the accepted schedule is what the main configuration submits)."""
    seen_ok = False
    extra = 0
    for obs in _trace_tool_observations(trace):
        if str(obs.get("name") or "").strip().lower() != "schedule":
            continue
        if seen_ok:
            extra += 1
        elif obs.get("ok") is True:
            seen_ok = True
    if not seen_ok and isinstance(trace, AbcMapping):
        names = trace.get("tool_names") or []
        results = trace.get("tool_results") or []
        for name, res in zip(names, results):
            if str(name).strip().lower() != "schedule" or not isinstance(res, AbcMapping):
                continue
            if seen_ok:
                extra += 1
            elif res.get("ok") is True:
                seen_ok = True
    return extra


# ---------------------------------------------------------------------------
# Optional R_tool_use shaping. Task-blind and success/protocol-clean gated.
# ---------------------------------------------------------------------------
_RELATION_OBS_RE = re.compile(r"^\s*(.+?)\s+(?:on|in|at|near)\s+(.+?)\s*$", re.I)


def _trace_tool_observations(trace: Optional[dict]) -> List[Mapping[str, Any]]:
    if not isinstance(trace, AbcMapping):
        return []
    observations = trace.get("tool_observations")
    if not isinstance(observations, AbcSequence) or isinstance(observations, (str, bytes, bytearray)):
        return []
    return [obs for obs in observations if isinstance(obs, AbcMapping)]


def _iter_struct_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, AbcMapping):
        for item in value.values():
            yield from _iter_struct_strings(item)
        return
    if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_struct_strings(item)


def _struct_mentions(goal: Optional[dict], plan: Optional[List[dict]]) -> Set[str]:
    mentions = {_norm(value) for value in _iter_struct_strings(goal)}
    mentions.update(_norm(value) for value in _iter_struct_strings(plan))
    mentions.discard("")
    return mentions


def _op_signature(op: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    if not isinstance(op, AbcMapping):
        return None
    sig = (
        _norm(op.get("op_type")),
        _norm(op.get("entity")),
        _norm(op.get("dest")),
    )
    return sig if any(sig) else None


def _plan_op_signatures(plan: Optional[List[dict]]) -> Set[Tuple[str, str, str]]:
    sigs: Set[Tuple[str, str, str]] = set()
    for op in plan or []:
        sig = _op_signature(op)
        if sig:
            sigs.add(sig)
    return sigs


def _plan_op_ids(plan: Optional[List[dict]]) -> Set[str]:
    op_ids = {_norm(op.get("op_id")) for op in plan or [] if isinstance(op, AbcMapping)}
    op_ids.discard("")
    return op_ids


def _obs_strings(obs: Mapping[str, Any], key: str) -> List[str]:
    value = obs.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, AbcSequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _obs_op_signatures(obs: Mapping[str, Any]) -> Set[Tuple[str, str, str]]:
    ops = obs.get("ops")
    if not isinstance(ops, AbcSequence) or isinstance(ops, (str, bytes, bytearray)):
        return set()
    sigs: Set[Tuple[str, str, str]] = set()
    for op in ops:
        if not isinstance(op, AbcMapping):
            continue
        sig = _op_signature(op)
        if sig:
            sigs.add(sig)
    return sigs


def _op_sigs_consumed(obs_sigs: Set[Tuple[str, str, str]],
                      plan_sigs: Set[Tuple[str, str, str]]) -> bool:
    if not obs_sigs or not plan_sigs:
        return False
    matched = len(obs_sigs & plan_sigs)
    if len(obs_sigs) <= 2:
        return matched == len(obs_sigs)
    return matched * 5 >= len(obs_sigs) * 4


def _relation_consumed(relation: str, mentions: Set[str]) -> bool:
    match = _RELATION_OBS_RE.match(str(relation))
    if match:
        return _norm(match.group(1)) in mentions and _norm(match.group(2)) in mentions
    return _norm(relation) in mentions


def _obs_consumed(
    obs: Mapping[str, Any],
    *,
    mentions: Set[str],
    plan_sigs: Set[Tuple[str, str, str]],
    plan_ids: Set[str],
) -> bool:
    obs_sigs = _obs_op_signatures(obs)
    if obs_sigs:
        return _op_sigs_consumed(obs_sigs, plan_sigs)

    relations = _obs_strings(obs, "relations")
    if relations:
        return any(_relation_consumed(relation, mentions) for relation in relations)

    op_ids = {_norm(op_id) for op_id in _obs_strings(obs, "op_ids")}
    op_ids.discard("")
    if op_ids:
        return bool(op_ids & plan_ids)

    entities = {_norm(entity) for entity in _obs_strings(obs, "entities")}
    entities.discard("")
    if entities:
        return bool(entities & mentions)

    return False


def _r_tool_use(
    trace: Optional[dict],
    goal: Optional[dict],
    plan: Optional[List[dict]],
    *,
    n_tool: int,
    stacked: int,
    nonterm: bool,
    mismatch: int,
    bad_json: int,
    r_task: float,
) -> Tuple[float, int, int, int, int]:
    """Return (r_tool_use, eligible, clean, consumed_count, observed_count)."""
    eligible = int(r_task >= 1.0 and n_tool > 0)
    clean = int(
        bool(eligible)
        and stacked <= 0
        and not nonterm
        and mismatch == 0
        and bad_json == 0
    )
    observations = _trace_tool_observations(trace)
    if not eligible or not clean:
        return 0.0, eligible, clean, 0, len(observations)
    if not observations:
        return 0.0, eligible, clean, 0, 0

    mentions = _struct_mentions(goal, plan)
    plan_sigs = _plan_op_signatures(plan)
    plan_ids = _plan_op_ids(plan)
    consumed = sum(
        1
        for obs in observations
        if _obs_consumed(obs, mentions=mentions, plan_sigs=plan_sigs, plan_ids=plan_ids)
    )
    denom = max(int(n_tool), len(observations), 1)
    return max(0.0, min(1.0, consumed / denom)), eligible, clean, consumed, len(observations)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _legacy_patch_stack_score(
    *,
    gt: Dict[str, Any],
    trace: Optional[dict],
    goal: Optional[dict],
    plan: Optional[List[dict]],
    r_task: float,
    r_goal: float,
    r_faithful: float,
    r_struct: float,
    n_tool: int,
    stacked: int,
    nonterm: bool,
    mismatch: int,
    bad_json: int,
):
    """The pre-2026-08-03 recipe, verbatim, behind ``LNSDL_REWARD_PATCH_STACK=1``.

    Kept so the recorded E5–E8 runs and docs/RL_debug.md remain reproducible and
    so the paper can report "penalties removed" as a measured ablation rather
    than an assertion. Not reachable in the default P5.3′.1 configuration.
    """
    p_format = -min(P_FORMAT * (int(stacked > 0) + bad_json + mismatch), P_FORMAT_CAP)
    p_nonterm = -P_NONTERM if nonterm else 0.0
    (r_tool_use, tool_use_eligible, tool_use_clean,
     tool_use_consumed, tool_use_observed) = _r_tool_use(
        trace,
        goal,
        plan,
        n_tool=n_tool,
        stacked=stacked,
        nonterm=nonterm,
        mismatch=mismatch,
        bad_json=bad_json,
        r_task=r_task,
    )
    tool_use_reward = R_TOOL_USE * r_tool_use

    # T1: conditional tool cost. Charge only when structurally trivial, without
    # a GT skip-decision structure, AND already solved. For skip-decision
    # samples, tools may be the means for state checking even when the DAG is
    # mechanically trivial, so they should not receive a "fewer tools" gradient.
    # Backward-compat: if the flag is missing or COND_PCOST is off, charge as
    # before.
    trivial = gt.get("trivial_structure")
    if COND_PCOST and trivial is not None:
        has_skip_decision = bool(gt.get("has_skip_decision"))
        charge_cost = bool(trivial) and (not has_skip_decision) and (r_task >= 1.0)
    else:
        charge_cost = True
    p_cost = -min(P_COST * n_tool, P_COST_CAP) if charge_cost else 0.0

    dense_pos_raw = W_GOAL * r_goal + W_FAITHFUL * r_faithful + W_STRUCT * r_struct
    dense_scale = 1.0
    if r_task < 1.0 and TASK_FAIL_DENSE_SCALE >= 0:
        dense_scale = TASK_FAIL_DENSE_SCALE
    if TASK_FAIL_SCALE_STRUCT_ONLY:
        # Failure band: compress only the co-saturated mechanical axis R_struct;
        # keep the semantic bottleneck axes R_goal/R_faithful at full budget so
        # the group-relative advantage can order failures by 取件 semantics
        # instead of by the {R_task=0, R_struct=1} hacking surface (§3.2).
        dense_pos = W_GOAL * r_goal + W_FAITHFUL * r_faithful + W_STRUCT * r_struct * dense_scale
    else:
        dense_pos = dense_pos_raw * dense_scale
    # E6 (docs/RL_debug.md §3(3)): hard-gate the task credit when the tool-call
    # channel is protocol-dirty (stacked turn OR emitted != executed). The
    # trajectory keeps its dense semantic reward (so it stays above a clean total
    # failure) but forfeits W_TASK, so a stacked winner can no longer out-score a
    # clean one. The dense-scale/cap below stay keyed on the TRUE r_task, so this
    # is a surgical removal of exactly the task term, nothing else.
    stack_dirty = bool(STACK_GATE_TASK and (stacked > 0 or mismatch > 0))
    r_task_scored = 0.0 if stack_dirty else r_task
    pos = W_TASK * r_task_scored + dense_pos
    score = pos + tool_use_reward + p_format + p_nonterm + p_cost
    if r_task < 1.0 and TASK_FAIL_SCORE_CAP > 0:
        score = min(score, TASK_FAIL_SCORE_CAP)

    return (score, dense_pos_raw, dense_scale, stack_dirty, r_task_scored,
            p_format, p_nonterm, p_cost, charge_cost, trivial,
            r_tool_use, tool_use_reward, tool_use_eligible, tool_use_clean,
            tool_use_consumed, tool_use_observed)


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[dict] = None,
) -> Dict[str, float]:
    """Compute the P5.3 scoped-RL reward. Returns a dict (see module docstring)."""
    text = solution_str or ""
    gt = _eg._numpy_to_python(ground_truth or {})
    trace = extra_info.get("trace") if isinstance(extra_info, dict) else None

    goal = _extract_goal(text)
    plan = _extract_plan(text)
    pred_active = _goal_active_set(goal)
    grasp = _plan_grasp_set(plan)
    gt_active = _gt_active_set(gt)

    r_task = _r_task(plan, gt)
    r_goal = _jaccard(pred_active, gt_active)
    r_faithful = _r_faithful(grasp, pred_active)
    r_goal_plan_transcribed = _jaccard(grasp, gt_active)
    r_struct = _r_struct(goal, plan, gt)
    r_struct_phi = _r_struct_phi(plan, gt, trace)
    r_struct_sketch = _r_struct_sketch(plan, gt, trace)
    r_task_phi = _r_task_phi(gt, trace)
    # R_plan on the submitted ops: Phi only adds mechanical ops, so its
    # acquire-family entity set is the sketch's; -1 when nothing was accepted.
    phi_ops = _phi_ops_from_trace(trace)
    r_goal_plan_phi = -1.0 if phi_ops is None else _jaccard(_plan_grasp_set(phi_ops), gt_active)
    r_goal_plan = r_goal_plan_transcribed
    if R_PLAN_SOURCE == "phi_schedule" and r_goal_plan_phi >= 0.0:
        r_goal_plan = r_goal_plan_phi

    n_tool, stacked, nonterm, mismatch, bad_json = _protocol_signals(text, trace, plan)
    format_hits = _format_hits_v2(text)
    sched_repeat = _schedule_repeat_count(trace)

    if not REWARD_PATCH_STACK:
        # ---- P5.3′.1 default: four positive axes, no negative term. ----------
        # The protocol signals above are still computed and reported; they are
        # diagnostics (and `nonterm` feeds the CFA axis-A guard), not prices.
        p_format = p_nonterm = p_cost = 0.0
        r_tool_use = tool_use_reward = 0.0
        tool_use_eligible = tool_use_clean = tool_use_consumed = tool_use_observed = 0.0
        charge_cost = False
        trivial = gt.get("trivial_structure")
        dense_scale = 1.0
        stack_dirty = False
        r_task_scored = r_task
        if R_TASK_SOURCE == "phi_schedule" and r_task_phi >= 0.0:
            r_task_scored = r_task_phi
        dense_pos_raw = (W_GOAL * r_goal + W_FAITHFUL * r_faithful + W_STRUCT * r_struct
                         + W_GOAL_PLAN * r_goal_plan + W_STRUCT_PHI * r_struct_phi
                         + W_STRUCT_SKETCH * r_struct_sketch)
        dense_pos = dense_pos_raw
        if P_FORMAT_V2 > 0.0:
            p_format = min(P_FORMAT_V2 * sum(format_hits.values()), P_FORMAT_V2_CAP)
        if P_SCHED_REPEAT > 0.0:
            p_cost = min(P_SCHED_REPEAT * sched_repeat, P_SCHED_REPEAT_CAP)
        score = W_TASK * r_task_scored + dense_pos - p_format - p_cost
    else:
        score, dense_pos_raw, dense_scale, stack_dirty, r_task_scored, \
            p_format, p_nonterm, p_cost, charge_cost, trivial, \
            r_tool_use, tool_use_reward, tool_use_eligible, tool_use_clean, \
            tool_use_consumed, tool_use_observed = _legacy_patch_stack_score(
                gt=gt, trace=trace, goal=goal, plan=plan,
                r_task=r_task, r_goal=r_goal, r_faithful=r_faithful,
                r_struct=r_struct, n_tool=n_tool, stacked=stacked,
                nonterm=nonterm, mismatch=mismatch, bad_json=bad_json,
            )

    return {
        "score": float(score),
        # 1.0 = legacy patch stack, 0.0 = P5.3′.1 simplified recipe. Runs mixing
        # the two are not comparable, so this rides along with every sample.
        "reward_patch_stack": float(REWARD_PATCH_STACK),
        # filter_groups.metric=acc -> dynamic sampling on task-success variance.
        # filter_groups.metric=acc -> dynamic sampling on task-success variance;
        # follows the R_task source (P1d) and is otherwise the legacy r_task.
        "acc": float(r_task_phi if (R_TASK_SOURCE == "phi_schedule" and r_task_phi >= 0.0) else r_task),
        "r_task": float(r_task),
        "r_task_scored": float(r_task_scored),
        "format_hits_v2": float(sum(format_hits.values())),
        "format_plan_with_tool_call": float(format_hits["plan_with_tool_call"]),
        "format_multi_plan": float(format_hits["multi_plan"]),
        "format_stray_bracket": float(format_hits["stray_bracket"]),
        "format_think_pseudo": float(format_hits["think_pseudo"]),
        "sched_repeat": float(sched_repeat),
        "r_goal": float(r_goal),
        "r_faithful": float(r_faithful),
        "r_goal_plan": float(r_goal_plan),
        "r_goal_plan_transcribed": float(r_goal_plan_transcribed),
        "r_goal_plan_phi": float(r_goal_plan_phi),
        "r_task_source_phi": float(R_TASK_SOURCE == "phi_schedule"),
        "r_plan_source_phi": float(R_PLAN_SOURCE == "phi_schedule"),
        "r_struct_phi": float(r_struct_phi),
        "r_struct_sketch": float(r_struct_sketch),
        "r_task_phi": float(r_task_phi),
        "r_struct": float(r_struct),
        "r_tool_use": float(r_tool_use),
        "r_dense_raw": float(dense_pos_raw),
        "tool_use_reward": float(tool_use_reward),
        "tool_use_eligible": float(tool_use_eligible),
        "tool_use_clean": float(tool_use_clean),
        "tool_use_consumed": float(tool_use_consumed),
        "tool_use_observed": float(tool_use_observed),
        "task_fail_dense_scale": float(dense_scale),
        "dense_scale_struct_only": float(TASK_FAIL_SCALE_STRUCT_ONLY),
        "p_format": float(p_format),
        "p_nonterm": float(p_nonterm),
        "p_cost": float(p_cost),
        "n_tool_calls": float(n_tool),
        "stacked": float(stacked > 0),
        "nonterm": float(nonterm),
        # P5.3'.0-3: detect crops are cheap in VRAM but spend response budget
        # (docs/debug_rl.md §6.1). Reported so the crop token share is a reading
        # rather than an assumption. -1.0 when the rollout predates the fields.
        "crop_images": (
            float(trace.get("response_image_kept"))
            if isinstance(trace, dict) and trace.get("response_image_kept") is not None
            else -1.0
        ),
        "crop_tokens": (
            float(trace.get("response_image_tokens"))
            if isinstance(trace, dict) and trace.get("response_image_tokens") is not None
            else -1.0
        ),
        # E6 stack gate (docs/RL_debug.md §3(3)): 1.0 when the W_TASK credit was
        # dropped for a protocol-dirty (stacked / emitted!=executed) trajectory.
        "task_gated": float(stack_dirty),
        # T1/T2 monitoring (task-blind flags, -1.0 when the parquet predates them):
        "cost_charged": float(charge_cost),
        "trivial_structure": float(bool(trivial)) if trivial is not None else -1.0,
        "skip_sample": (
            float(bool(gt.get("has_skip_decision")))
            if gt.get("has_skip_decision") is not None else -1.0
        ),
    }
