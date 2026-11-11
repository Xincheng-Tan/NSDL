"""
LOCUS-Bench rendering & packaging (P7.4, docs/benchmark.md §4.1/§4.2-2/§5.2/§9).

Four deliverables, official files untouched:

  1. ``build_render_config`` — per-instance ManiSkill scene YAML built with
     the SAME logic as the official ``generate_tasks_from_json.py`` main
     loop, plus the S4 hook: the official code FORCES transparent cabinet
     styles {4, 11} whenever an asset starts inside a cabinet
     (generate_tasks_from_json.py:130-136); for ``meta.opaque_container``
     instances we invert that — style is sampled from the OPAQUE set so
     "not visible" is a real perceptual state. All randomness is seeded per
     instance/pair (``style_seed``) instead of the official global random.
  2. ``render_batch`` — split-aware render loop: writes per-instance scene
     configs and invokes the official ``script/generate_planning_data.py``
     subprocess per config, collecting each snapshot into
     ``<out>/<split>/images/``. GPU-optional: with ``dry_run=True`` only
     the configs and the manifest are produced (CPU-only acceptance path).
  3. ``build_split_plan`` + ``assign_split`` — the §5.1 grid split (train /
     test_id / ood_s / ood_m / ood_x) with the §5.2 robustness slices:
     paraphrase holdout (ID-test rows sampled from held-out paraphrase ids,
     flagged ``meta.paraphrase_holdout``) and layout holdout (layout 9 is
     excluded from train sampling and only appears in test splits, flagged
     ``meta.layout_holdout``).
  4. ``pack_parquet`` — VIKI-L2 column-parity parquet writer:
     ``images / data_source / prompt / ability / reward_model`` with the
     byte-exact official system-prompt prefix (``locus_prompt_prefix.txt``,
     lifted verbatim from ``data/viki/VIKI-L2/train.parquet``) + the same
     per-robot suffix (robot set / characteristics / operation APIs), user
     content ``"<image>" + description``, and ``reward_model.ground_truth``
     in the official gt dict shape (R1..R3 padded ``robots``,
     ``time_steps`` with per-slot ``None``). LOCUS meta rides in
     ``reward_model.ground_truth.locus_meta`` — additive only, so
     ``prepare_vl_sft_data`` / eval harness / ``prepare_rl_data`` consume
     these files unchanged.

  5. ``decidability_check`` — §6.3-3 spot check on rendered images, from GT
     placement (no detector): S1–S3 judgment targets must sit in a visible
     position area; S4 hidden items must be inside the (opaque) cabinet and
     the style must come from the opaque set.
"""

import hashlib
import io
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml

from locus_task_pool import iter_cells
from locus_generate import (
    BridgeVariantSampler, MaskVocabSampler, RealizationSampler,
    instantiate_counterfactual_pair,
    instantiate_locus_task,
    stratified_mask_keys,
)
from locus_validate import phi_gate, simenv_replay

__all__ = [
    "TRANSPARENT_STYLES",
    "OPAQUE_STYLES",
    "SPLIT_OF_CELL",
    "HOLDOUT_LAYOUT",
    "build_split_plan",
    "generate_split_instances",
    "build_render_config",
    "mesh_aabb",
    "rotated_min_z",
    "rest_offset_z",
    "render_batch",
    "pack_parquet",
    "decidability_check",
]

# official transparent-cabinet styles (generate_tasks_from_json.py:130-136)
# relaunches per chunk; each pass either completes jobs or burns one of a
# job's 3 `.inprogress` attempts, so this only bounds pathological churn.
_MAX_CHUNK_ATTEMPTS = 8

TRANSPARENT_STYLES = [4, 11]
OPAQUE_STYLES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
# styles avoiding the wooden floor a local proxy with cardboardbox (official logic)
NON_WOODEN_STYLES = [0, 1, 3, 5, 7, 8, 9, 10, 11]

HOLDOUT_LAYOUT = 9          # only appears in test splits (§5.2)
PARAPHRASE_HOLDOUT_FRAC = 0.3

VISIBLE_AREAS = {"kitchen work area", "kitchen island area", "table", "rack",
                 "counter", "kitchen counter", "groundA", "ground"}

ROBOT_DESCRIPTION = {
    'stompy': 'A bipedal robot designed for dynamic walking and stomping tasks, featuring articulated arms. Color: Light blue body with yellow and orange accents.',
    'fetch': 'A wheeled robot with a flexible arm for object manipulation, designed for mobility and dexterity. Color: White with blue and black accents.',
    'unitree_h1': 'A humanoid robot with arms and legs designed for human-like movements and tasks. Color: Black.',
    'panda': 'A fixed robotic arm designed for precise and delicate manipulation tasks. Color: White with black accents.',
    'anymal_c': 'A quadrupedal robot built for navigating rough terrains and performing complex tasks with four articulated legs. Color: Red and black with some accents.',
    'unitree_go2': 'A compact quadrupedal robot optimized for agile movement and stability with four legs for efficient locomotion. Color: White.',
}
AGENT_AVAIL_ACTIONS = {
    'panda':      ['Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'fetch':      ['Move', 'Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'unitree_go2': ['Move', 'Push', 'Interact'],
    'unitree_h1': ['Move', 'Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'stompy':     ['Move', 'Reach', 'Grasp', 'Place', 'Open', 'Close', 'Interact'],
    'anymal_c':   ['Move', 'Push', 'Interact'],
}

with open(os.path.join(_SCRIPT_DIR, "locus_prompt_prefix.txt")) as _f:
    VIKI_SYSTEM_PREFIX = _f.read()


# ---------------------------------------------------------------------------
# split plan (§5.1 grid + §5.2 slices)
# ---------------------------------------------------------------------------

def _cell_split(s, m):
    if s <= 2 and m <= 2:
        return "train"          # train + test_id share these cells
    if s >= 3 and m <= 2:
        return "ood_s"
    if s <= 2 and m >= 3:
        return "ood_m"
    return "ood_x"


SPLIT_OF_CELL = {(s, m): _cell_split(s, m)
                 for fam, s, m in iter_cells()}


def build_split_plan(n_train=0, n_test_id=0, n_ood=0):
    """Rows of (split, family, s, m, count). ID splits draw from the 9
    train cells; OOD splits from their own cells."""
    plan = []
    for fam, s, m in iter_cells():
        split = _cell_split(s, m)
        if split == "train":
            if n_train:
                plan.append(("train", fam, s, m, n_train))
            if n_test_id:
                plan.append(("test_id", fam, s, m, n_test_id))
        elif n_ood:
            plan.append((split, fam, s, m, n_ood))
    return plan


def _row_rng(seed, split, fam, s, m):
    """Independent rng sub-stream for one plan row.

    Why this exists: with a single shared `random.Random(seed)` walked across
    the whole plan, every row's draws depend on how many draws the PRECEDING
    rows made. Raising `n_train` for the ramp therefore reshuffles every later
    row -- measured: `n_train` 20 -> 21 leaves 0/200 `ood_m` configs intact, so
    a train-only ramp silently invalidates the already-rendered test/OOD
    splits. Deriving the stream from (seed, split, cell) makes each row's
    sampling a function of its own identity, so growing one row leaves the
    others byte-identical and `--skip-existing` can actually reuse them.

    Default since P7.5c. Switching stream derivation renumbers every existing
    config once, so it had to land together with a re-render -- which the ramp
    forces anyway: under the shared stream a 20 -> 21 train bump already leaves
    0/180 test_id, 0/35 ood_s, 1/50 ood_m and 0/10 ood_x configs intact, so the
    P7.5b renders are discarded either way. Flipping therefore costs nothing at
    this transition and buys additivity for every later ramp (docs/YOLO.md §6.2
    prefers bumping N to the 7.2k tier when a val gate misses; positional reuse
    would re-render the whole 6.3k tier for that).
    """
    raw = f"{seed}:{split}:{fam}:{s}:{m}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8],
                                       "big"))


def generate_split_instances(plan, seed=0, validate=True, max_retries=20,
                             per_row_rng=True, stratify_small_vocab=True):
    """Instantiate a split plan with all P7.2/P7.3 gates + §5.2 holdouts.

    Split hygiene applied here:
      * train never uses the holdout layout; test splits allow all layouts
        and flag rows whose sampled layout is the holdout one;
      * ~30% of test_id rows use held-out paraphrase ids (the top 30% of
        each cell's paraphrase-id range, never sampled for train).
    Balanced realization sampling is per (split, cell) so every split keeps
    the §4.3 balance property on its own.

    `per_row_rng=True` (default since P7.5c) gives each plan row its own rng
    sub-stream (see `_row_rng`) so that changing one row's count leaves the
    other rows' configs byte-identical. `per_row_rng=False` reproduces the
    P7.5b shared stream and is kept only for re-deriving that pilot.

    `stratify_small_vocab=True` (default since P7.5c) draws the §7-3
    audit-relevant masks of a too-small plan row round-robin instead of
    i.i.d., so the vocabulary cross-coverage audit is not decided by a
    20-draw lottery on the OOD cells (see `MaskVocabSampler`). This is the
    production path for that behaviour; `generate_locus_cell` keeps its
    i.i.d. draw unless the caller passes a sampler explicitly.
    """
    rng = random.Random(seed)
    out = defaultdict(list)
    stats = Counter()
    for split, fam, s, m, count in plan:
        if per_row_rng:
            rng = _row_rng(seed, split, fam, s, m)
        sampler = RealizationSampler()
        variant_sampler = BridgeVariantSampler()
        mask_sampler = MaskVocabSampler()
        stratify_keys = (stratified_mask_keys(fam, s, m, count)
                         if stratify_small_vocab else frozenset())
        if stratify_keys and fam == "F4" and s == 4:
            # §7.10 requires both halves of a counterfactual pair to draw the
            # same masks, which a min-count sampler cannot do (see
            # instantiate_counterfactual_pair). Pair invariance is the harder
            # constraint, so those rows keep the i.i.d. draw. The production
            # plan never lands here (its F4-S4 rows are large enough); small
            # ad-hoc plans do.
            stats[f"stratify_skipped_s4_pairing:{split}"] += count
            stratify_keys = frozenset()
        if stratify_keys:
            stats[f"stratified_masks:{split}"] += count
        pending_pair = []
        for k in range(count):
            realization = None
            probe_n = instantiate_locus_task(
                fam, s, m, random.Random(0), realization=realization,
                inject_distractors=False)["meta"]["n_paraphrases"]
            n_hold = max(1, int(round(probe_n * PARAPHRASE_HOLDOUT_FRAC)))
            train_ids = list(range(probe_n - n_hold))
            hold_ids = list(range(probe_n - n_hold, probe_n))

            paraphrase_holdout = (split == "test_id"
                                  and k % 10 < 10 * PARAPHRASE_HOLDOUT_FRAC)
            if split == "train":
                par_ids = train_ids
            elif paraphrase_holdout:
                par_ids = hold_ids
            else:
                par_ids = train_ids

            allowed_layouts = None
            if split == "train":
                allowed_layouts = [l for l in range(10) if l != HOLDOUT_LAYOUT]

            inst = None
            if fam == "F4" and s == 4:
                if not pending_pair:
                    for attempt in range(max_retries):
                        pair = instantiate_counterfactual_pair(
                            fam, s, m, rng,
                            allowed_paraphrase_ids=par_ids,
                            allowed_layouts=allowed_layouts,
                            mask_sampler=mask_sampler,
                            stratify_keys=stratify_keys)
                        if not validate:
                            pending_pair.extend(pair)
                            break
                        pair_results = [
                            simenv_replay(
                                cand, rng=random.Random(
                                    cand["meta"]["instantiate_seed"]))
                            for cand in pair
                        ]
                        if all(ok for ok, _ in pair_results):
                            pending_pair.extend(pair)
                            break
                        stats["simenv_rejected"] += sum(
                            not ok for ok, _ in pair_results)
                    if not pending_pair:
                        raise RuntimeError(
                            f"{split} {fam} s{s}m{m}: no valid "
                            f"counterfactual pair in {max_retries} tries")
                inst = pending_pair.pop(0)
            else:
                realization = sampler.next(fam, s, m, rng)
                bridge_variant = variant_sampler.next(fam, s, m, rng)
                for attempt in range(max_retries):
                    cand = instantiate_locus_task(
                        fam, s, m, rng, realization=realization,
                        bridge_variant=bridge_variant,
                        allowed_paraphrase_ids=par_ids,
                        allowed_layouts=allowed_layouts,
                        mask_sampler=mask_sampler,
                        stratify_keys=stratify_keys)
                    if not validate:
                        inst = cand
                        mask_sampler.commit()
                        break
                    ok, err = simenv_replay(cand, rng=random.Random(
                        cand["meta"]["instantiate_seed"]))
                    if ok:
                        inst = cand
                        # Only a kept instance spends vocab budget; a rejected
                        # attempt must not tilt the round robin (MaskVocabSampler).
                        mask_sampler.commit()
                        break
                    mask_sampler.rollback()
                    stats["simenv_rejected"] += 1
            if inst is None:
                raise RuntimeError(f"{split} {fam} s{s}m{m}: no valid "
                                   f"instance in {max_retries} tries ({err})")
            if validate:
                phi_ok, phi_code = phi_gate(inst)
                inst["meta"]["phi_ok"] = phi_ok
                inst["meta"]["phi_code"] = phi_code
                stats["phi_ok" if phi_ok else "phi_fail"] += 1
            inst["meta"]["split"] = split
            inst["meta"]["paraphrase_holdout"] = bool(paraphrase_holdout)
            inst["meta"]["layout_holdout"] = (inst["layout_id"]
                                              == HOLDOUT_LAYOUT)
            inst["meta"]["sample_index"] = len(out[split])
            out[split].append(inst)
            stats[f"n:{split}"] += 1
    return dict(out), dict(stats)


# ---------------------------------------------------------------------------
# placement feasibility: a model of the official scene_builder loop
# ---------------------------------------------------------------------------
# utils/scenes/scene_builder.py:128-231 places agents first (0.6m mutual
# separation) and then objects, appending EVERY pose — agents included, and
# objects without `randp_scale` included — to one shared `asset_pposes` list.
# An object carrying `randp_scale` redraws `ppos + randp_scale * rand(3)`
# until it is ≥0.15m from every EARLIER pose, and gives up after 100 tries
# with `exit(0)` (not a raise), which segfaults sapien and truncates the rest
# of the render chunk. Three properties of that check drive everything below:
#   * it is CHEBYSHEV and XY-ONLY (`np.max(delta_pos[:2]) < 0.15`), so a pair
#     is safe as soon as ONE of x/y is ≥0.15 apart, whatever the other does;
#   * `randp_scale` is a ONE-SIDED offset (rand() is [0,1)), so an item's
#     reach along an axis is the interval between p and p+rp, not p±rp;
#   * each item only avoids items EARLIER in config order.
_OFFICIAL_SEP = 0.15        # official reject threshold, exclusive
_SLOT_MIN_SEP = 0.16        # what we separate at: strictly above threshold
_OFFICIAL_MAX_TRIES = 101   # `while not available_pos` with `count > 100`
_MAX_FAILURE_PROB = 1e-4    # above this a group is treated as infeasible
_MAX_CONDITIONAL_FAILURE = 1e-5   # same, for the worst-realization bound
_GATE_TRIALS = 100          # see _infeasible_items: 3x100 saturates
_GATE_SEEDS = 3             # ...and one seed alone is a coin flip


def _xy_reach_box(pos_cfg):
    """[(x_lo, x_hi), (y_lo, y_hi)] over every position the draw can yield."""
    p = pos_cfg['ppos']['p']
    rp = pos_cfg.get('randp_scale')
    box = []
    for ax in (0, 1):
        a = float(p[ax])
        b = a + (float(rp[ax]) if rp else 0.0)
        box.append((min(a, b), max(a, b)))
    return box


def _reach_boxes_separable(b1, b2, sep=_OFFICIAL_SEP):
    """True iff NO draw from b1 can ever conflict with a draw from b2."""
    for (lo1, hi1), (lo2, hi2) in zip(b1, b2):
        if lo1 - hi2 >= sep or lo2 - hi1 >= sep:
            return True
    return False


def _hits(xy, placed, sep=_OFFICIAL_SEP):
    x, y = xy
    for px, py in placed:
        if abs(px - x) < sep and abs(py - y) < sep:
            return True
    return False


def _placement_estimate(boxes, seed=0, trials=48, resamples=64,
                        sep=_OFFICIAL_SEP, max_tries=_OFFICIAL_MAX_TRIES,
                        looping=None):
    """(failure probability, per-item worst per-draw conflict rate).

    Plain rejection sampling would need ~1e5 trials to resolve the failure
    rates that matter here, so the estimator is variance-reduced: it walks
    the items in config order exactly like the official loop, but instead of
    counting whole-loop failures it measures each item's per-draw conflict
    probability q against the accepted earlier poses (``resamples`` draws)
    and folds in the analytic loop survival ``1 - q**max_tries``. Failure
    needs q≈1 (q=0.9 gives 2e-5 over 101 tries), which is exactly where a
    ratio estimator is sharpest. The item is then placed at one of its own
    non-conflicting draws, so later items see a faithful conditional
    distribution. Deterministic given ``seed``.

    The second return value is each item's max q over all trials. Its max is
    the cheap escalation signal: q is estimated from ``resamples`` draws so
    it converges fast, whereas the probability itself is driven by rare
    earlier-placement realizations and needs many trials to resolve. An item
    whose worst q stays low cannot fail however long you sample it. Keeping
    it per item (not one scalar) is what lets the caller say WHICH item to
    re-slot instead of re-slotting the whole scene.

    ``looping`` marks which boxes run the object collision loop. Agent poses
    are obstacles that precede every object in ``asset_pposes`` but they are
    drawn against a different, far looser criterion (0.6m, 5000 tries), so
    they belong in the model as boxes with ``looping=False``: they occupy
    space but cannot themselves fail here.
    """
    rng = random.Random(seed)
    total = 0.0
    q_item = [0.0] * len(boxes)
    for _ in range(trials):
        placed, survival = [], 1.0
        for idx, box in enumerate(boxes):
            (x_lo, x_hi), (y_lo, y_hi) = box
            if x_hi - x_lo < 1e-9 and y_hi - y_lo < 1e-9:
                placed.append((x_lo, y_lo))     # no randp_scale: no loop
                continue
            if looping is not None and not looping[idx]:
                placed.append((x_lo + (x_hi - x_lo) * rng.random(),
                               y_lo + (y_hi - y_lo) * rng.random()))
                continue
            free, hit = [], 0
            for _k in range(resamples):
                xy = (x_lo + (x_hi - x_lo) * rng.random(),
                      y_lo + (y_hi - y_lo) * rng.random())
                if _hits(xy, placed, sep):
                    hit += 1
                else:
                    free.append(xy)
            q = hit / resamples
            q_item[idx] = max(q_item[idx], q)
            survival *= 1.0 - q ** max_tries
            # the official loop keeps the LAST draw either way; when nothing
            # was free it has already exit(0)ed, so later items are moot.
            placed.append(free[rng.randrange(len(free))] if free else xy)
        total += 1.0 - survival
    return total / trials, q_item


def placement_failure_probability(boxes, **kw):
    """P(the official loop exit(0)s) for one ordered list of reach boxes."""
    return _placement_estimate(boxes, **kw)[0]


def _coverage_probability(box_i, box_j, sep=_OFFICIAL_SEP):
    """P(one draw of j leaves NO feasible position anywhere in i's box).

    Exact, and the reason the gate is not pure Monte-Carlo. When a single
    earlier item can cover all of a later item's reach with its 0.30×0.30
    exclusion square, that later item fails with certainty — but the cover
    itself can be rare (pilot ood_s/00120: a 1.6m rock sweep covers a 20×4cm
    clothes iron in 0.4% of draws), and a 48-trial estimator sees it as a
    coin flip. The geometry, unlike the estimate, has no variance: j covers
    i on an axis iff it draws into ``(hi_i - sep, lo_i + sep)``, an interval
    that is non-empty exactly when i is narrower than ``2*sep``.
    """
    p = 1.0
    for (lo_i, hi_i), (lo_j, hi_j) in zip(box_i, box_j):
        if hi_i - lo_i >= 2 * sep - 1e-9:
            return 0.0                  # too wide for one exclusion square
        lo_c, hi_c = hi_i - sep, lo_i + sep
        span = hi_j - lo_j
        if span < 1e-9:
            # j is fixed (no randp_scale). It covers with certainty or never;
            # going through `overlap` would score every fixed blocker 0.
            if not lo_c - 1e-9 <= lo_j <= hi_c + 1e-9:
                return 0.0
            continue
        overlap = min(hi_j, hi_c) - max(lo_j, lo_c)
        if overlap <= 1e-9:
            return 0.0                  # j can never draw into the cover
        p *= min(1.0, overlap / span)
    return p


def _infeasible_items(boxes, seed, looping=None, thresh=_MAX_FAILURE_PROB,
                      cond_thresh=_MAX_CONDITIONAL_FAILURE):
    """Gate: which items make the official loop fail this ordering often
    enough to care? Returns the set of offending indices, empty if none.

    Three criteria, because the marginal failure probability alone is not
    estimable at build-path cost. Failure needs BOTH a tight item AND an
    unlucky placement of the items before it, and the second factor can sit
    at 1/3000 (ood_s/00116: 6e-4 overall, still reading 4e-7 after 400
    trials, only resolving at 2000). Sampling that rare realization per
    config is hopeless; recognising the tight item is not.

      * exact single-blocker coverage — a closed form, no variance at all,
        and a rigorous LOWER bound on the failure probability;
      * ``q**101`` per item, the failure probability CONDITIONAL on the worst
        realization observed. It converges in a handful of trials because q
        is measured from ``resamples`` draws, not from trials, and gets a
        stricter bar than the marginal rate precisely because it is a
        conditional: q=0.89 means an item was ~90% blocked once, which is
        worth a slot even if the marginal rate is small;
      * the marginal estimate itself, which catches whatever combination of
        partial blockers the other two miss. It is a whole-scene quantity,
        so when only it trips, the blame goes to the worst-q item.

    Sampling is ``_GATE_SEEDS`` x ``_GATE_TRIALS`` = 300 realizations, and
    the split matters. What needs sampling is not the failure but the bad
    EARLIER realization that enables it; at the estimator's default 48 that
    is a coin flip (it left 8 pilot configs at 2e-4..1e-3) and ~200 saturates
    it. Splitting those across seeds on top is what makes the verdict stable:
    the per-item q is a MAX over trials, so a scene sitting near the bar
    (pilot ood_s/00115: bread_3 at q=0.953 under one seed, 0.875 under the
    next, bar 0.892) flips with the stream. Unioning the seeds means the
    build path is at least as strict as any single-seed audit of its output,
    which is the direction the asymmetry has to point. ~3 minutes across the
    whole 6860-instance ramp, against a ~15h render.
    """
    bad = set()
    for k in range(len(boxes)):
        if looping is not None and not looping[k]:
            continue                    # agents cannot fail this loop
        (x_lo, x_hi), (y_lo, y_hi) = boxes[k]
        if x_hi - x_lo < 1e-9 and y_hi - y_lo < 1e-9:
            continue                    # no randp_scale: appended, never drawn
        if any(_coverage_probability(boxes[k], boxes[j]) > thresh
               for j in range(k)):
            bad.add(k)
    for s in range(_GATE_SEEDS):
        p, q_item = _placement_estimate(boxes, seed=seed + s, looping=looping,
                                        trials=_GATE_TRIALS)
        for k, q in enumerate(q_item):
            if q ** _OFFICIAL_MAX_TRIES > cond_thresh:
                bad.add(k)
        if p > thresh and not bad:
            bad.add(max(range(len(q_item)), key=lambda k: q_item[k]))
    return bad


def _is_infeasible(boxes, seed, **kw):
    """``_infeasible_items`` as a predicate."""
    return bool(_infeasible_items(boxes, seed, **kw))


def _slot_bound(bounds, member, ax):
    """Allowed ``ppos.p[ax]`` interval for one member, or ``(-inf, +inf)``.

    ``bounds`` is keyed by ``id()`` because a config dict is unhashable and
    names are only unique within a scene; every caller builds it in the same
    frame that owns the dicts, so the ids cannot be recycled underneath it.
    """
    b = (bounds or {}).get(id(member))
    return b[ax] if b else (float('-inf'), float('inf'))


def _reslot_merged_group(members, min_sep=_SLOT_MIN_SEP, bounds=None):
    """Give every jittering member of one merged group its own slot.

    Case (b) can assume a shared anchor and split the declared span evenly.
    A cross-anchor group has neither a common anchor nor a common span, so
    this schedules intervals along the axis with the most room instead:
    members that cannot move (zero jitter on that axis) occupy their point
    first, then each mobile member takes the LOWEST position ≥ its own reach
    that clears every occupied point by ``min_sep``. Because the official
    check is Chebyshev, separating on one axis alone is sufficient — so the
    other axis keeps its jitter untouched.

    A second pass hands back jitter ALONG the slot axis too: once the slots
    are known, each may grow up to ``min_sep`` short of the next slot above
    it (and no further than its own declared reach). Without it a 2m counter
    sweep sharing a group with one tight item would collapse to a point,
    which would cost far more scene diversity than the conflict was worth.

    ``bounds`` (see ``fit_in_cabinet``) caps a member's slot: a cabinet
    member may be moved anywhere its body still clears the carcass and no
    further, INCLUDING in the ``max(occupied) + min_sep`` fallback, which is
    otherwise unbounded and is what pushed pilot ood_x/00023's tomato 16cm
    out through a closed door. Bounded members also decide the slot axis and
    schedule along the WHOLE carcass rather than their own inset box —
    without both, the clamp alone turns three of the pilot's leaks into two
    unsolvable scenes (coverage 1.0), which is a worse failure than the leak.
    A member the shelf genuinely cannot separate ends up at the bound edge:
    one conflict for the official loop to draw around, never a leak.
    """
    boxes = [_xy_reach_box(m['pos']) for m in members]
    extent = [max(b[ax][1] for b in boxes) - min(b[ax][0] for b in boxes)
              for ax in (0, 1)]
    lim = [[_slot_bound(bounds, m, k) for m in members] for k in (0, 1)]
    # Axis choice: normally "the one with the most room". A bounded member
    # changes what room MEANS — its carcass is 0.94m along one axis and 0.34m
    # (one door depth) along the other, so slotting along the short one packs
    # everyone onto the same point and hands the official loop a scene it
    # cannot solve. Bounded members therefore pick the axis first, by the
    # tightest bound each axis leaves them; the extent rule breaks ties and
    # still decides every group that has no bounded member at all.
    room = [min((b[1] - b[0] for b in lim[k] if b[1] < float('inf')),
                default=float('inf')) for k in (0, 1)]
    if abs(room[0] - room[1]) > 1e-9 and min(room) < float('inf'):
        ax = 0 if room[0] > room[1] else 1
    else:
        ax = 0 if extent[0] >= extent[1] else 1
    limit = lim[ax]
    mobile = [i for i, b in enumerate(boxes)
              if b[ax][1] - b[ax][0] > 1e-9 or limit[i][1] < float('inf')]
    fixed = [i for i in range(len(boxes)) if i not in set(mobile)]
    start = {i: boxes[i][ax][0] for i in fixed}
    occupied = list(start.values())
    # lowest-reach-first, config order breaking ties: stable and byte-repro
    for i in sorted(mobile, key=lambda i: (boxes[i][ax][0], i)):
        lo, hi = boxes[i][ax]
        width = hi - lo
        if limit[i][1] < float('inf'):
            # the carcass, not the already-inset box, is what a cabinet
            # member may travel within: the fit pass shrank that box to
            # whatever the DECLARED anchor allowed, which is a fraction of
            # the shelf. Scheduling from the shelf's low end instead packs
            # the group along the 0.94m span — starting from each member's
            # own inset box only ever moves items UP, and runs out of shelf
            # (pilot ood_x/00023: two bodies 0.13m apart with 0.7m of empty
            # shelf behind them, which the official loop cannot solve).
            lo, hi = limit[i]
        cands = sorted({lo} | {o + min_sep for o in occupied
                               if o + min_sep > lo})
        s = next((c for c in cands if c <= hi + 1e-9
                  and all(abs(c - o) >= min_sep - 1e-9 for o in occupied)),
                 None)
        if s is None:
            # the declared reach cannot hold this member at all; push past
            # everything (same fallback-widening spirit as case (b)) — but
            # never past a bound, since leaving a container is not a
            # placement the official loop would have produced either.
            s = min(max(occupied) + min_sep, limit[i][1])
            s = max(s, limit[i][0])
        start[i] = (s, width)
        occupied.append(s)
    occupied.sort()
    for i in mobile:
        s, width = start[i]
        above = [o for o in occupied if o > s + 1e-9]
        ceiling = (above[0] - min_sep) if above else float('inf')
        # a bounded member keeps its width and never widens (the fit pass's
        # invariant); an unbounded one keeps the declared absolute reach.
        reach = (min(s + width, limit[i][1]) if limit[i][1] < float('inf')
                 else boxes[i][ax][1])
        jitter = max(0.0, min(reach, ceiling) - s)
        pos = members[i]['pos']
        pos['ppos']['p'][ax] = s
        # a zero-jitter bounded member is mobile here (it may be TRANSLATED
        # inside its carcass even though it has no reach to give), so the
        # key may be absent rather than merely all-zero.
        rp = list(pos.get('randp_scale') or [0.0, 0.0, 0.0])
        rp[ax] = jitter
        if all(abs(v) < 1e-9 for v in rp):
            pos.pop('randp_scale', None)    # exact placement: skip the loop
        else:
            pos['randp_scale'] = rp


def resolve_cross_anchor_conflicts(object_cfgs, agent_cfgs=(), max_passes=4,
                                   bounds=None):
    """Case (c): re-slot items whose reach boxes overlap ACROSS anchors.

    Case (b) groups by exact anchor and therefore cannot see two items that
    approach each other from different anchors (a long counter sweep passing
    under a rack column). Here the grouping is the transitive closure of
    "these two boxes can conflict at all", which is the exact complement of
    the official check's guarantee, so a member of one group provably cannot
    fight a member of another.

    GATED on measured infeasibility, like case (b) and for the same reason:
    group membership is realization-dependent, so touching a group that the
    official loop already handles would risk perturbing a non-target asset
    and breaking the §7.10 counterfactual-pair invariance. Overlap is common
    (~10% of pilot configs) while real infeasibility is rare (27/1160), and
    only the latter is re-slotted. Agents are obstacles but never movable —
    their poses are task semantics — so they only enter the model.

    The gate runs over the WHOLE ordered scene, not group by group. Failure
    is a property of the official loop's single pass over ``asset_pposes``,
    and a per-group gate is a different question: it reseeds, and it drops
    the agents and objects the group cannot fight. Those are sound omissions
    for conflicts but not for the RNG stream, so the two gates disagreed on
    borderline scenes (pilot ood_s/00115, train/00363 — flagged by an audit
    over the full scene, invisible to the per-group gate). Scoring the scene
    the loop actually runs makes build path and audit the same measurement;
    grouping is then only used to decide who gets re-slotted alongside the
    item the gate blamed.

    Returns the number of groups re-slotted.
    """
    agent_boxes = [_xy_reach_box(a['pos']) for a in agent_cfgs
                   if isinstance(a.get('pos'), dict)]
    n_agents = len(agent_boxes)
    touched = 0
    for _ in range(max_passes):
        boxes = [_xy_reach_box(c['pos']) for c in object_cfgs]
        ordered = agent_boxes + boxes
        looping = [False] * n_agents + [True] * len(boxes)
        seed = int.from_bytes(hashlib.sha256(
            repr(ordered).encode()).digest()[:8], 'big')
        bad = {k - n_agents for k in _infeasible_items(
            ordered, seed, looping=looping) if k >= n_agents}
        if not bad:
            break

        parent = list(range(len(object_cfgs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _reach_boxes_separable(boxes[i], boxes[j]):
                    parent[find(i)] = find(j)
        groups = defaultdict(list)
        for i in range(len(boxes)):
            groups[find(i)].append(i)

        flagged = [m for root, m in sorted(groups.items())
                   if any(i in bad for i in m)
                   and any('randp_scale' in object_cfgs[i]['pos'] for i in m)]
        if not flagged:
            break                       # blamed item has no jitter to give
        for members in flagged:
            # re-slotting can shift a member outside its original box, so the
            # merge is re-run until it reaches a fixed point.
            _reslot_merged_group([object_cfgs[i] for i in members],
                                 bounds=bounds)
            touched += 1
    return touched


# ---------------------------------------------------------------------------
# case (d): rest the body on its anchor plane (docs/benchmark.md §9)
# ---------------------------------------------------------------------------
# The official placement loop positions an object's ORIGIN inside the declared
# position area and never looks at the object's own extent
# (utils/scenes/scene_builder.py:203-245: `Pose.create_from_pq(temp_ppos,
# qpos)`), and nothing steps physics afterwards — `locus_render_masked.render_one`
# is `env.reset()` + 11 `render()` calls, and `render()` does not settle. The
# .obj assets are authored with the origin at the AABB BASE (measured over all
# 22 distinct meshes: xy centred to 1e-4, z in [0, h]), so an unrotated object
# rests exactly on the plane — but `pos.qpos` rotates about that same base
# origin, which swings part of the body BELOW it. The layouts compensate by
# hand on 4 of 29 classes (`ppos.p[2]` = 0.05/0.08/0.15) and not at all on the
# rest, so 19/29 classes sink: indoor_plant 20.0cm, bottle 16.0, clothes_iron
# 14.4, bread 8.7, cup 6.2 (and `plate`'s 0.08 lift over-corrects into a 3.6cm
# hover). That is the mechanism behind the visible 穿模 of cups/bottles/forks
# into the counter.
#
# It is NOT the mechanism behind the S4 leak, though the two correlate: a
# 30-scene same-RNG A/B left `opaque_container_leaks` at 9/30 before and after
# the lift, because the leaking bodies poke out HORIZONTALLY through a closed
# door, not downward through the shelf. The horizontal axis is case (e) below.
#
# So each object is lifted by however much of it hangs below the plane. The
# correction is ONE-SIDED (sinking only): a hovering object is geometrically
# odd but does not destroy a silhouette or breach a container, and lowering it
# would perturb scenes no measurement flags. `.glb` assets (hammer, shoe) are
# origin-CENTRED rather than base-aligned, so they get the same treatment from
# their real AABB rather than an assumed convention.
_MESH_AABB_CACHE = {}


def _obj_aabb(path):
    """AABB of a Wavefront .obj, stdlib only. Read as bytes: several assets
    carry non-UTF8 bytes in comments/material names."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with open(path, "rb") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) < 4 or parts[0] != b"v":
                continue
            for ax in range(3):
                v = float(parts[ax + 1])
                lo[ax] = min(lo[ax], v)
                hi[ax] = max(hi[ax], v)
    if lo[0] == float("inf"):
        raise ValueError(f"no vertices in {path}")
    return lo, hi


def _glb_aabb(path):
    """AABB of a binary glTF, stdlib only.

    glTF requires min/max on every POSITION accessor, so the AABB needs no
    vertex decoding — only the node transforms applied to each primitive's
    accessor box. The box's 8 corners are transformed and re-bounded, which is
    a bound rather than the exact hull under rotation; both LOCUS .glb assets
    have identity node transforms, so it is exact there.
    """
    import struct

    blob = open(path, "rb").read()
    if blob[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    off, doc = 12, None
    while off < len(blob):
        length, kind = struct.unpack_from("<II", blob, off)
        off += 8
        if kind == 0x4E4F534A:                     # 'JSON'
            doc = json.loads(blob[off:off + length])
            break
        off += length
    if doc is None:
        raise ValueError(f"no JSON chunk in {path}")

    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
                for i in range(4)]

    def local(node):
        m = node.get("matrix")
        if m:                                      # glTF matrices are column-major
            return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
        t = node.get("translation") or [0.0, 0.0, 0.0]
        x, y, z, w = node.get("rotation") or [0.0, 0.0, 0.0, 1.0]
        s = node.get("scale") or [1.0, 1.0, 1.0]
        rot = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        return [[rot[r][c] * s[c] for c in range(3)] + [t[r]] for r in range(3)] \
            + [[0.0, 0.0, 0.0, 1.0]]

    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    nodes = doc.get("nodes") or []
    eye = [[float(r == c) for c in range(4)] for r in range(4)]
    stack = [(i, eye) for i in range(len(nodes))
             if not any(i in (n.get("children") or []) for n in nodes)]
    while stack:
        idx, parent = stack.pop()
        node = nodes[idx]
        world = mul(parent, local(node))
        for child in node.get("children") or []:
            stack.append((child, world))
        if "mesh" not in node:
            continue
        for prim in doc["meshes"][node["mesh"]].get("primitives") or []:
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            if "min" not in acc or "max" not in acc:
                raise ValueError(f"POSITION accessor without min/max in {path}")
            for bits in range(8):
                pt = [acc["max" if bits >> k & 1 else "min"][k] for k in range(3)]
                for ax in range(3):
                    v = sum(world[ax][k] * pt[k] for k in range(3)) + world[ax][3]
                    lo[ax] = min(lo[ax], v)
                    hi[ax] = max(hi[ax], v)
    if lo[0] == float("inf"):
        raise ValueError(f"no POSITION accessor in {path}")
    return lo, hi


def mesh_aabb(path):
    """Cached (min3, max3) of a mesh file in its own units."""
    key = os.path.abspath(path)
    if key not in _MESH_AABB_CACHE:
        reader = _glb_aabb if key.lower().endswith((".glb", ".gltf")) else _obj_aabb
        _MESH_AABB_CACHE[key] = reader(key)
    return _MESH_AABB_CACHE[key]


def rotated_min_z(mesh_path, scale, qpos):
    """Lowest point of the scaled, rotated body relative to its own origin.

    ``qpos`` is sapien's wxyz. Under rotation the extremum is attained at a
    vertex, and the vertex set is not recoverable from the AABB alone, so this
    transforms the AABB's 8 corners: an upper bound on min z, i.e. it can
    under-lift a concave mesh but never over-lift one. Exact for the
    axis-aligned 90° rotations that dominate the layouts.
    """
    lo, hi = mesh_aabb(mesh_path)
    sx, sy, sz = (scale if isinstance(scale, (list, tuple)) else [scale] * 3)[:3]
    lo = [lo[0] * sx, lo[1] * sy, lo[2] * sz]
    hi = [hi[0] * sx, hi[1] * sy, hi[2] * sz]
    w, x, y, z = qpos
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    row = (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))
    return min(sum(row[k] * (hi[k] if bits >> k & 1 else lo[k]) for k in range(3))
               for bits in range(8))


def rest_offset_z(item_cfg, lift_z, assets_root=None):
    """How far to lift ``item_cfg`` so its body rests on its anchor plane.

    ``lift_z`` is the height the config already places the origin above the
    plane (the layout's own hand-tuned ``ppos.p[2]``). Returns 0.0 when the
    body already clears the plane, and 0.0 with no complaint when the object
    has no visual mesh — a primitive has no file to measure, and a missing
    asset is the renderer's error to raise, not this pass's.
    """
    visual = item_cfg.get("visual") or {}
    filename = visual.get("filename")
    if not filename:
        return 0.0
    path = os.path.join(assets_root or _ROBOFACTORY, filename)
    if not os.path.isfile(path):
        return 0.0
    min_z = rotated_min_z(path, visual.get("scale", 1.0),
                          item_cfg["pos"]["qpos"])
    return max(0.0, -(min_z + float(lift_z)))


# ---------------------------------------------------------------------------
# case (e): keep a cabinet-anchored body inside the carcass (docs/benchmark.md)
# ---------------------------------------------------------------------------
# Same root cause as case (d) on the horizontal axes: the official loop places
# the ORIGIN inside the declared `cabinet` position area and never checks the
# body's own extent, so an object longer than the area's clearance pokes out
# through the closed door (or off the end of the run) and the S4 premise —
# "an opaque container must be OPENED to observe its contents" — silently
# fails. 9 of the pilot's 30 in-cabinet opaque targets leaked this way, and
# re-rendering those same 30 configs with this pass applied takes them to 0
# (`outputs/locus_cabfix_pilot/`, 2026-07-30: 8 leaks before -- the 9th is a
# 9-pixel graze that today's `MIN_VISIBLE_PIXELS_MOVABLE` floor keeps out of
# the labels -- and 30/30 targets end up correctly hidden behind solid doors).
#
# Two facts make it fixable. First, the reference for containment is the
# CARCASS, not the declared area: the layouts' `cabinet` areas were authored
# for small default assets and are far tighter than the cabinet (layout 5
# declares a 2cm-deep band inside a 34cm-deep cabinet, and part of that band
# is not even inside the carcass), so insetting within the declared band alone
# still leaves a bread 2cm proud of the door — measured, not assumed. Second,
# a rotated body's origin is NOT its XY centre: bread's `qpos` maps its 53cm
# local z-axis onto world x, so the origin sits at one END of the loaf
# (rel-x [-0.537, +0.012]). The inset therefore has to be SIGNED per side,
# exactly like `rotated_min_z` is for z.
#
# `_CABINET_INTERIOR` is measured, not derived: robocasa builds fixtures
# through an align_to/stack chain that only the scene builder resolves, so
# `locus_cabinet_probe.py` builds one env per layout and dumps
# `Fixture.get_int_sites(all_points=True, relative=False)` for every CLOSED
# cabinet class whose interior meets the layout's declared `cabinet` band.
# The result is a property of the layout blueprint (sizes come from the
# robocasa layout yaml, styles only swap materials), so it is a constant here
# and re-derivable by rerunning the probe. Layout 0 is absent because
# `locus_task_pool.ALL_LAYOUTS` never samples it.
_CABINET_INTERIOR = {
    # layout_id: ((x_lo, x_hi), (y_lo, y_hi)) of the closed-cabinet interior
    1: ((1.280, 2.220), (-0.370, -0.030)),   # cab_main            HingeCabinet
    2: ((2.580, 3.520), (-0.370, -0.030)),   # cab_2               HingeCabinet
    3: ((0.730, 1.670), (-0.370, -0.030)),   # cab_main            HingeCabinet
    4: ((2.830, 3.170), (-3.718, -2.778)),   # cab_2               HingeCabinet
    5: ((0.030, 0.370), (-3.470, -2.530)),   # cab_1_left          HingeCabinet
    6: ((6.030, 6.370), (-1.670, -0.730)),   # cab_1_right         HingeCabinet
    7: ((0.680, 1.620), (-0.370, -0.030)),   # cab_main            HingeCabinet
    # layout 8's band spans a 3-cabinet run; the union is the containment
    # reference because the dividers between them are behind the same doors.
    8: ((0.680, 3.120), (-0.370, -0.030)),   # cab_1+cab_main+cab_2
    9: ((1.680, 2.620), (-0.370, -0.030)),   # cab_1               HingeCabinet
}

# Quarter-turn yaws tried when the body does not fit as authored. A yaw about
# WORLD z is the one reorientation that is free here: it cannot change which
# way is up, and it leaves `rotated_min_z` invariant, so case (d)'s lift stays
# correct whichever order the two passes run in.
_CABINET_YAWS = (0, 1, 3, 2)          # preference order: identity first


def _quat_mul(a, b):
    """Hamilton product of two sapien wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _yaw_quat(k):
    """wxyz for k quarter turns about world z."""
    return list(_YAW_TABLE[k % 4])


_YAW_TABLE = (
    (1.0, 0.0, 0.0, 0.0),                                  # 0
    (0.70710678118654752, 0.0, 0.0, 0.70710678118654752),  # +90
    (0.0, 0.0, 0.0, 1.0),                                  # 180
    (0.70710678118654752, 0.0, 0.0, -0.70710678118654752),  # -90
)


def rotated_xy_box(mesh_path, scale, qpos):
    """Signed XY extent of the scaled, rotated body about its own origin.

    Returns ``((x_lo, x_hi), (y_lo, y_hi))`` — offsets from the origin, so
    ``x_lo`` is normally negative. Asymmetric by construction: the layouts'
    quaternions routinely park the origin at one end of the body. Like
    ``rotated_min_z`` this transforms the AABB's 8 corners, which bounds the
    true hull from outside — it can over-inset a concave mesh, never
    under-inset one, so containment is never claimed falsely.
    """
    lo, hi = mesh_aabb(mesh_path)
    sx, sy, sz = (scale if isinstance(scale, (list, tuple)) else [scale] * 3)[:3]
    lo = [lo[0] * sx, lo[1] * sy, lo[2] * sz]
    hi = [hi[0] * sx, hi[1] * sy, hi[2] * sz]
    w, x, y, z = qpos
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    rows = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)))
    out = []
    for row in rows:
        vals = [sum(row[k] * (hi[k] if bits >> k & 1 else lo[k])
                    for k in range(3)) for bits in range(8)]
        out.append((min(vals), max(vals)))
    return tuple(out)


def _clamp_interval(lo, hi, flo, fhi):
    """Move ``[lo, hi]`` into ``[flo, fhi]`` with the smallest change.

    Shrinks where the two overlap; translates to the nearest feasible point
    when they are disjoint; collapses onto the feasible midpoint when the
    feasible interval is itself empty (``flo > fhi``: the body is wider than
    the cabinet on this axis and no placement contains it).
    """
    if flo > fhi:
        mid = 0.5 * (flo + fhi)
        return mid, mid, False
    lo2, hi2 = max(lo, flo), min(hi, fhi)
    if lo2 > hi2:                       # disjoint: nearest feasible point
        p = flo if hi < flo else fhi
        return p, p, True
    return lo2, hi2, True


def fit_in_cabinet(item_cfg, interior, assets_root=None):
    """Pull one cabinet-anchored item's reach box inside the carcass.

    Rewrites ``pos.ppos.p[:2]`` / ``pos.randp_scale[:2]`` so that EVERY point
    the official loop can draw keeps the body inside ``interior``, yawing the
    item by a quarter turn first if that is what makes the body fit. Returns
    a report dict; ``contained`` is False only when no yaw fits the body on
    some axis, in which case the item is centred to minimise the overhang.

    The reach box is only ever shrunk or translated, never widened, so this
    cannot introduce a placement conflict the official loop did not already
    have — but it CAN shrink a span to a point, which is why it runs before
    cases (b)/(c) rather than after: those two own separation, and they should
    see the boxes this pass leaves behind.
    """
    pos = item_cfg["pos"]
    visual = item_cfg.get("visual") or {}
    filename = visual.get("filename")
    report = {"name": item_cfg.get("name"), "yaw": 0, "contained": True,
              "shift": 0.0, "bounds": None}
    if not filename:
        return report                    # primitive: no mesh, nothing to bound
    path = os.path.join(assets_root or _ROBOFACTORY, filename)
    if not os.path.isfile(path):
        return report
    scale = visual.get("scale", 1.0)
    rp = pos.get("randp_scale") or [0.0, 0.0, 0.0]
    band = [(min(pos["ppos"]["p"][ax], pos["ppos"]["p"][ax] + rp[ax]),
             max(pos["ppos"]["p"][ax], pos["ppos"]["p"][ax] + rp[ax]))
            for ax in (0, 1)]

    best = None
    for k in _CABINET_YAWS:
        qpos = _quat_mul(_YAW_TABLE[k], pos["qpos"]) if k else list(pos["qpos"])
        rel = rotated_xy_box(path, scale, qpos)
        feas = [(interior[ax][0] - rel[ax][0], interior[ax][1] - rel[ax][1])
                for ax in (0, 1)]
        fits = all(f[0] <= f[1] + 1e-12 for f in feas)
        out = [_clamp_interval(band[ax][0], band[ax][1], *feas[ax])
               for ax in (0, 1)]
        shift = sum(max(0.0, band[ax][0] - out[ax][0])
                    + max(0.0, out[ax][1] - band[ax][1]) for ax in (0, 1))
        score = (fits, -shift)
        if best is None or score > best[0]:
            best = (score, k, qpos, out, fits, feas)
    _score, k, qpos, out, fits, feas = best

    if k:
        pos["qpos"] = list(qpos)
        report["yaw"] = k * 90
    for ax in (0, 1):
        lo, hi = out[ax][0], out[ax][1]
        pos["ppos"]["p"][ax] = lo
        if "randp_scale" in pos:
            new_rp = list(pos["randp_scale"])
            new_rp[ax] = hi - lo
            pos["randp_scale"] = new_rp
    report["contained"] = bool(fits)
    report["shift"] = sum(max(0.0, band[ax][0] - out[ax][0])
                          + max(0.0, out[ax][1] - band[ax][1]) for ax in (0, 1))
    # The hard bound cases (b)/(c) have to honour: the interval of ppos values
    # that keeps the WHOLE body inside the carcass on this yaw. Handed on
    # rather than re-derived there, so a member can be re-slotted (which is
    # what keeps the official loop safe) without being re-slotted out through
    # a closed door (which is what the S4 premise forbids). None when no yaw
    # fits: an unbounded member is one the audit already reports uncontained.
    report["bounds"] = [tuple(feas[ax]) for ax in (0, 1)] if fits else None
    return report


# ---------------------------------------------------------------------------
# case (f): keep a body from poking through the room's walls
# ---------------------------------------------------------------------------
# The lateral twin of case (d). `rest_offset_z` stops a body sinking through
# the plane it rests ON; nothing stopped it sticking through the planes it
# stands NEXT TO. The official loop places the ORIGIN inside a declared
# `position_areas` band and never checks the body's extent, and those bands
# are authored to the COUNTER, which butts against the wall -- so the reach
# box legitimately runs right up to the wall plane and any body whose origin
# is not its centre hangs through it.
#
# Measured on the shipped scissors (`assets/objects/scissors/new.obj`, the
# worst case because its qpos maps the 48cm blade axis onto world +y and
# parks the origin at the pivot): rel-y is [-0.002, +0.477], i.e. the body is
# essentially ALL on the +y side of its own origin. Layout 6/8 counters give
# origin y in [-0.51, -0.27] against a back wall at y=0.000, so the far end
# of the reach box puts 0.21m of scissors outside the room. Spot-check rate
# was 20% of scissors instances (docs/YOLO.md), and the criterion there was
# depth, not silhouette area -- the blade is thin, so it reads as a sliver in
# the image and as a hard depth violation in the buffer.
#
# `_WALL_PLANES` is measured, not derived, for the same reason
# `_CABINET_INTERIOR` is: `locus_wall_probe.py` builds one env per layout and
# reads `Wall.pos` / `Wall.size` / `Wall.wall_side` back off the built scene.
# Walls are FINITE rectangles, not half-space boundaries -- layout 9 is
# L-shaped and carries a second `left` wall at x=+2.075 that spans only
# y in [-4.54, -3.41] -- so each entry keeps its lateral span and a wall
# constrains a body only where the two overlap. Reducing a layout to one
# plane per side instead would push layout 9's counter bodies through an
# alcove wall that is not in front of them. All nine layouts' walls span
# z in [0, 3], so there is no height gate: anything standing on a counter is
# within every wall's vertical extent.
#
# Entries: (side, plane coord on the normal axis, lateral lo, lateral hi).
_WALL_PLANES = {
    1: (('back',  +0.000, -1.040, +5.340),
        ('left',  -1.000, -5.040, +0.040),
        ('right', +5.300, -5.040, +0.040)),
    2: (('back',  +0.000, -0.040, +5.240),
        ('right', +5.200, -5.240, +0.040)),
    3: (('back',  +0.000, -0.040, +5.240),
        ('front', -5.900, -0.040, +5.240),
        ('left',  +0.000, -5.940, +0.040),
        ('right', +5.200, -5.940, +0.040)),
    4: (('back',  +0.000, -0.040, +3.240),
        ('left',  +0.000, -5.640, +0.040),
        ('right', +3.200, -5.640, +0.040)),
    5: (('back',  +0.000, -0.040, +4.440),
        ('left',  +0.000, -4.540, +0.040),
        ('right', +4.400, -4.540, +0.040)),
    6: (('back',  +0.000, -0.040, +6.440),
        ('left',  +0.000, -5.040, +0.040),
        ('right', +6.400, -5.040, +0.040)),
    7: (('back',  +0.000, -0.040, +4.440),
        ('left',  +0.000, -5.040, +0.040),
        ('right', +4.400, -5.040, +0.040)),
    8: (('back',  +0.000, -0.040, +6.120),
        ('left',  +0.000, -6.040, +0.040),
        ('right', +6.080, -6.040, +0.040)),
    9: (('back',  +0.000, -0.340, +5.940),
        ('front', -3.400, -0.340, +2.040),
        ('left',  -0.300, -3.440, +0.040),
        ('left',  +2.075, -4.540, -3.410),   # L-shaped: alcove only
        ('right', +5.900, -4.540, +0.040)),
}

# side -> (normal axis, sign of the room interior), i.e. the room satisfies
# `inward * (coord - plane) >= 0`. Cross-checked per layout against
# `_CABINET_INTERIOR`, which was measured through an unrelated code path
# (`get_int_sites`), by `locus_wall_probe._check_layout`.
_WALL_SIDES = {"back": (1, -1), "front": (1, +1),
               "left": (0, +1), "right": (0, -1)}

_MESH_VERTS_CACHE = {}


def _obj_verts(path):
    """Cached vertex list of a Wavefront .obj, stdlib only.

    Same byte-level parse as ``_obj_aabb`` (several assets carry non-UTF8
    bytes in comments and material names) but keeps the points instead of
    reducing them, so an extent can be taken along an arbitrary direction.
    """
    key = os.path.abspath(path)
    if key not in _MESH_VERTS_CACHE:
        verts = []
        with open(key, "rb") as f:
            for raw in f:
                parts = raw.split()
                if len(parts) < 4 or parts[0] != b"v":
                    continue
                verts.append((float(parts[1]), float(parts[2]),
                              float(parts[3])))
        if not verts:
            raise ValueError(f"no vertices in {key}")
        _MESH_VERTS_CACHE[key] = verts
    return _MESH_VERTS_CACHE[key]


def hull_xy_box(mesh_path, scale, qpos):
    """Exact XY extent of the scaled, rotated body about its own origin.

    Same contract as ``rotated_xy_box`` -- ``((x_lo, x_hi), (y_lo, y_hi))``
    as offsets from the origin -- but projects every vertex instead of the
    AABB's 8 corners, so it is the extent itself rather than a bound on it.

    Case (e) deliberately keeps the looser ``rotated_xy_box``: over-insetting
    inside a cabinet is harmless (the body is behind a door, only containment
    is observable) and its pilot verification was recorded against that
    function. Out on a counter the difference is not harmless -- the AABB
    bound over-states banana's y extent by 26% and bread's by 16% under the
    layouts' quaternions, and every over-stated centimetre pulls a body that
    was never through the wall, which puts an image into the re-render set
    for nothing. The mesh this reads is the VISUAL mesh, which is exactly
    the geometry that either does or does not appear to cross the wall.

    Falls back to the AABB bound for .glb, whose vertex buffers this module
    does not decode (``_glb_aabb`` reads only the accessors' min/max).
    """
    if mesh_path.lower().endswith((".glb", ".gltf")):
        return rotated_xy_box(mesh_path, scale, qpos)
    sx, sy, sz = (scale if isinstance(scale, (list, tuple)) else [scale] * 3)[:3]
    w, x, y, z = qpos
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    rows = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)))
    out = [[float("inf"), float("-inf")], [float("inf"), float("-inf")]]
    for vx, vy, vz in _obj_verts(mesh_path):
        vx, vy, vz = vx * sx, vy * sy, vz * sz
        for i, r in enumerate(rows):
            d = r[0] * vx + r[1] * vy + r[2] * vz
            if d < out[i][0]:
                out[i][0] = d
            if d > out[i][1]:
                out[i][1] = d
    return (tuple(out[0]), tuple(out[1]))



def clear_of_walls(item_cfg, walls, assets_root=None):
    """Pull one item's reach box off the room's walls.

    Rewrites ``pos.ppos.p[:2]`` / ``pos.randp_scale[:2]`` so that EVERY point
    the official loop can draw keeps the body on the room side of every wall
    in front of it. Translation only -- unlike ``fit_in_cabinet`` this never
    yaws the item. A yaw is free inside a cabinet, where the body is hidden
    behind a door and only containment is observable; out on a counter the
    orientation is part of the scene the model is asked to read, so changing
    it would alter instances that are not defective.

    Deliberately a **no-op unless the body can actually reach a wall**, which
    is what keeps the re-render targeted: a config in which every item is
    already clear comes out byte-identical, so it never enters the delta set.

    Returns a report dict; ``shift`` is the total change to the reach box
    across both axes and is 0.0 exactly when the item was already clear.
    """
    pos = item_cfg["pos"]
    visual = item_cfg.get("visual") or {}
    filename = visual.get("filename")
    report = {"name": item_cfg.get("name"), "shift": 0.0,
              "walls": [], "feasible": True}
    if not filename:
        return report                    # primitive: no mesh, nothing to bound
    path = os.path.join(assets_root or _ROBOFACTORY, filename)
    if not os.path.isfile(path):
        return report
    rel = hull_xy_box(path, visual.get("scale", 1.0), pos["qpos"])
    rp = pos.get("randp_scale") or [0.0, 0.0, 0.0]
    band = [(min(pos["ppos"]["p"][ax], pos["ppos"]["p"][ax] + rp[ax]),
             max(pos["ppos"]["p"][ax], pos["ppos"]["p"][ax] + rp[ax]))
            for ax in (0, 1)]

    feas = [[float("-inf"), float("inf")], [float("-inf"), float("inf")]]
    for side, coord, lat_lo, lat_hi in walls:
        axis, inward = _WALL_SIDES[side]
        lat = 1 - axis
        # Does any drawable placement put the body in front of this wall?
        if (band[lat][1] + rel[lat][1] < lat_lo
                or band[lat][0] + rel[lat][0] > lat_hi):
            continue
        # Guard against teleporting a body across a wall it is BEHIND. On an
        # L-shaped layout the lateral test can match a partition that this
        # item sits on the far side of; constraining then would shove it
        # bodily into the other half of the room. If the reach box's centre
        # is already outside, this wall is not ours to enforce.
        mid = 0.5 * (band[axis][0] + band[axis][1])
        if inward * (mid - coord) < 0:
            continue
        if inward < 0:
            feas[axis][1] = min(feas[axis][1], coord - rel[axis][1])
        else:
            feas[axis][0] = max(feas[axis][0], coord - rel[axis][0])
        report["walls"].append(side)

    # Reported so an audit can weight a config by how much of the reach box is
    # actually illegal: the pass fires when ANY drawable point protrudes, but
    # the shipped image drew exactly one of them, so "config moves" overstates
    # "image is defective" whenever the legal part of the band is large.
    report["band"] = [list(b) for b in band]
    report["feas"] = [list(f) for f in feas]

    for ax in (0, 1):
        lo, hi, ok = _clamp_interval(band[ax][0], band[ax][1], *feas[ax])
        report["feasible"] &= bool(ok)
        report["shift"] += abs(lo - band[ax][0]) + abs(hi - band[ax][1])
        if abs(lo - band[ax][0]) < 1e-12 and abs(hi - band[ax][1]) < 1e-12:
            continue                     # untouched: leave the config alone
        pos["ppos"]["p"][ax] = lo
        if "randp_scale" in pos:
            new_rp = list(pos["randp_scale"])
            new_rp[ax] = hi - lo
            pos["randp_scale"] = new_rp
    return report


def cabinet_containment_report(item_cfg, interior, assets_root=None):
    """Does this item's CURRENT reach box keep the body inside ``interior``?

    Read-only twin of ``fit_in_cabinet``, used to audit the boxes AFTER cases
    (b)/(c) have run: both of those may push a member past its declared reach
    (``_reslot_merged_group``'s ``max(occupied) + min_sep`` fallback), so
    "fitted before" does not by itself imply "contained after".
    """
    visual = item_cfg.get("visual") or {}
    filename = visual.get("filename")
    if not filename:
        return None
    path = os.path.join(assets_root or _ROBOFACTORY, filename)
    if not os.path.isfile(path):
        return None
    rel = rotated_xy_box(path, visual.get("scale", 1.0),
                         item_cfg["pos"]["qpos"])
    rp = item_cfg["pos"].get("randp_scale") or [0.0, 0.0, 0.0]
    p = item_cfg["pos"]["ppos"]["p"]
    worst = 0.0
    for ax in (0, 1):
        lo = min(p[ax], p[ax] + rp[ax]) + rel[ax][0]
        hi = max(p[ax], p[ax] + rp[ax]) + rel[ax][1]
        worst = max(worst, interior[ax][0] - lo, hi - interior[ax][1])
    return {"name": item_cfg.get("name"), "overhang": max(0.0, worst)}


# ---------------------------------------------------------------------------
# render config with the S4 opaque hook
# ---------------------------------------------------------------------------

def render_stem(inst):
    """``<split>_<sample_index>`` — the on-disk identity of one render."""
    return f'{inst["meta"]["split"]}_{inst["meta"]["sample_index"]:05d}'


def build_render_config(inst, layouts_dir=None, rng=None,
                        wall_clear_scope=None):
    """Scene YAML dict for one instance — official main-loop logic with
    seeded randomness and the S4 opaque-style inversion.

    ``wall_clear_scope`` restricts case (f) to a set of ``render_stem``s.
    ``None`` (default) applies it everywhere. A scope exists because the pass
    invalidates the render of every config it touches, and docs/YOLO.md §6.2.5
    settled on repairing a 1111-config subset rather than all 1955: an
    out-of-scope config must then reproduce BYTE-IDENTICALLY so `render_batch`
    keeps its existing snapshot instead of deleting it as stale. Scoping the
    pass (rather than scoping the re-render) is what keeps config and image
    describing the same scene for every sample in the pack.
    """
    if rng is None:
        render_seed = int(inst["meta"].get(
            "style_seed", inst["meta"]["instantiate_seed"] ^ 0x5EED))
    else:
        render_seed = rng.getrandbits(64)

    def _render_rng(label):
        raw = f"{render_seed}:{label}".encode()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        return random.Random(seed)

    agent_rng = _render_rng("agents")
    layouts_dir = layouts_dir or os.path.join(_ROBOFACTORY, "configs/layouts")
    with open(os.path.join(layouts_dir,
                           f'layout_{inst["layout_id"]}.yaml')) as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    objects_dict, agents_dict, pos_dict = {}, {}, {}
    for agent_cfg in cfg['agents']:
        agents_dict[agent_cfg['robot_uid'].rsplit('-', 1)[0]] = agent_cfg
    for object_cfg in cfg['objects']:
        name = object_cfg['name'].rsplit('_', 1)[0] \
            .replace(' ', '_').replace('-', '_')
        objects_dict[name] = object_cfg
    for pos_area in cfg['position_areas']:
        for pos_name in pos_area['names']:
            pos_dict[pos_name] = pos_area['pos']

    new_agent_cfgs = []
    in_cabinet = False
    scene_has_cabinet = any(
        name.rsplit('_', 1)[0] in ("cabinet", "kitchen cabinet")
        for name in inst["init_pos"]
        if not (name.startswith('R') and name[1:].isdigit()))
    avoid_wooden = False
    c = 0

    def _add_agent(robot_id, robot_type):
        nonlocal c
        if robot_type.startswith('panda') and robot_id in inst['init_pos']:
            robot_init_pos = agent_rng.choice(inst['init_pos'][robot_id])
            base = dict(agents_dict[robot_init_pos])
            base['robot_uid'] = f'{robot_id}-{robot_init_pos}-{c}'
        else:
            base = dict(agents_dict[robot_type])
            base['robot_uid'] = f'{robot_type}-{c}'
        # Case (a) for AGENTS (pilot train/560): a fixed-mount arm with
        # randp_scale=[0,0,0] redraws the SAME point against the official
        # ≥0.6m agent-separation loop — if an earlier mobile robot landed
        # close, all 5000 draws fail identically → exit(0) → sapien SIGSEGV.
        # No key → official code skips the loop (exact placement, intended
        # for fixed mounts). agents_dict entries are shared: replace `pos`
        # with a filtered copy, never mutate in place.
        rp = (base.get('pos') or {}).get('randp_scale')
        if rp is not None and all(abs(v) < 1e-9 for v in rp):
            base['pos'] = {k: v for k, v in base['pos'].items()
                           if k != 'randp_scale'}
        new_agent_cfgs.append(base)
        c += 1

    for robot_id, robot_type in inst['robots'].items():
        _add_agent(robot_id, robot_type)
    for idle_robot in inst.get('idle_robots', []):
        _add_agent(idle_robot, idle_robot)

    new_object_cfgs = []
    cabinet_items = []
    init_pos = inst['init_pos']

    def _resolve_area(area, object_rng, hops=0):
        """LOCUS extension of the official position lookup: a 'present'
        realization can align an item to a container ASSET (plate/bowl/
        sink); officially only areas and agent anchors are resolvable, so
        chase the container's own init position (seeded choice) until an
        area or agent anchor is reached."""
        if hops > 4:
            raise ValueError(f'position chain too deep at {area}')
        if area in pos_dict or (area.startswith('R') and area[1:].isdigit()):
            return area
        for holder_name, holder_pos in init_pos.items():
            if holder_name.rsplit('_', 1)[0] == area and holder_pos:
                return _resolve_area(
                    object_rng.choice(list(holder_pos)), object_rng, hops + 1)
        raise KeyError(f'unresolvable position area: {area}')

    for item_name, item_pos in init_pos.items():
        if item_name.startswith('R') and item_name[1:].isdigit():
            continue
        item_type = item_name.rsplit('_', 1)[0] \
            .replace(' ', '_').replace('-', '_')
        if item_type not in objects_dict:
            continue
        object_rng = _render_rng(f"object:{item_name}")
        item_cfg = json.loads(json.dumps(objects_dict[item_type]))
        item_cfg['name'] = item_name
        # the layout's hand-tuned lift above whatever plane it lands on; both
        # branches below add the plane's own z, so capture it first (case (d))
        lift_z = float(item_cfg['pos']['ppos']['p'][2])
        if item_type == 'cardboardbox':
            avoid_wooden = True
        position_area = _resolve_area(
            object_rng.choice(list(item_pos)), object_rng)
        if position_area.startswith('R') and position_area[1:].isdigit():
            base = next((a for a in new_agent_cfgs
                         if a['robot_uid'].startswith(position_area)), None)
            if base is None:
                raise ValueError(f'no agent config for {position_area}')
            asset_base = base['pos']['ppos']['p']
            delta_area = base['robot_uid'].split('-')[1]
            delta = pos_dict[f'{delta_area}_asset']
            for ax in range(len(item_cfg['pos']['ppos']['p'])):
                item_cfg['pos']['ppos']['p'][ax] = \
                    asset_base[ax] + delta['ppos']['p'][ax]
            item_cfg['pos']['randp_scale'] = delta['randp_scale']
            lift_z = float(delta['ppos']['p'][2])   # the layout lift is gone
        else:
            if position_area in ('cabinet', 'kitchen cabinet'):
                in_cabinet = True
                cabinet_items.append(item_cfg)
            new_pos = pos_dict[position_area]
            for ax in range(len(item_cfg['pos']['ppos']['p'])):
                item_cfg['pos']['ppos']['p'][ax] += new_pos['ppos']['p'][ax]
            item_cfg['pos']['randp_scale'] = new_pos['randp_scale']
        item_cfg['pos']['ppos']['p'][2] += rest_offset_z(item_cfg, lift_z)
        new_object_cfgs.append(item_cfg)

    # ---- case (e): pull cabinet-anchored bodies inside the carcass -------
    # Before cases (a)/(b)/(c): this pass only ever shrinks or translates a
    # reach box, and those three own separation, so they must see the boxes
    # it leaves behind rather than the declared ones.
    interior = _CABINET_INTERIOR.get(int(inst["layout_id"]))
    cabinet_fits = []
    if cabinet_items and interior is None:
        raise KeyError(
            f'layout {inst["layout_id"]} anchors an item in a cabinet but has '
            f'no _CABINET_INTERIOR entry (rerun locus_cabinet_probe.py)')
    for item_cfg in cabinet_items:
        cabinet_fits.append(fit_in_cabinet(item_cfg, interior))
    # id() -> allowed ppos interval per XY axis, honoured by cases (b)/(c).
    cabinet_bounds = {id(it): rep["bounds"]
                      for it, rep in zip(cabinet_items, cabinet_fits)
                      if rep.get("bounds")}

    # ---- case (f): keep bodies from poking through the room's walls ------
    # Also before (a)/(b)/(c), and for the same reason as (e): it only ever
    # shrinks or translates a reach box, so separation must see what it
    # leaves behind. Cabinet-anchored items are skipped -- (e) has already
    # bounded them to the carcass, which is inside the room, and re-clamping
    # them here would silently overwrite the `cabinet_bounds` that (b)/(c)
    # are about to honour.
    walls = _WALL_PLANES.get(int(inst["layout_id"]))
    if walls is None:
        raise KeyError(
            f'layout {inst["layout_id"]} has no _WALL_PLANES entry '
            f'(rerun locus_wall_probe.py)')
    cabinet_ids = {id(it) for it in cabinet_items}
    if wall_clear_scope is None or render_stem(inst) in wall_clear_scope:
        wall_clears = [clear_of_walls(it, walls)
                       for it in new_object_cfgs if id(it) not in cabinet_ids]
    else:
        wall_clears = []

    # ---- placement feasibility (pilot-verified fixes, 2026-07-27) --------
    # The official placement loop (utils/scenes/scene_builder.py) exit(0)s —
    # which segfaults sapien mid-build — when an object with `randp_scale`
    # cannot draw a position ≥0.15 XY from all earlier objects in 100 tries.
    # Two structural cases surfaced by the P7.5 pilot:
    # (a) all-zero jitter (F9 carrier load: item and cardboardbox share the
    #     exact anchor): co-location is INTENDED, so drop the key — official
    #     code only runs the collision loop when `randp_scale` is present;
    # (b) ≥2 items resolving to the same small-jitter anchor (F5 rack:
    #     already-at-dest candidates / same-tag extras / clutter): a 5–10cm
    #     box cannot hold two items 15cm apart, so each item gets its own
    #     disjoint slot of the declared span (see the SLOT note below).
    #     Note `randp_scale` is a ONE-SIDED offset, not ± jitter: official
    #     code draws `ppos + randp_scale * np.random.rand(3)` and rand() is
    #     [0,1), so an item reaches [p, p+rp] along each axis. Grouping is
    #     XY-ONLY: the official check ignores z (`delta_pos[:2]`), so rack
    #     shelves stacked at the same XY column fight each other too
    #     (pilot sample train/348: candidate on the upper shelf left zero
    #     feasible measure for two clutter items on the lower shelf).
    anchor_groups = defaultdict(list)
    for item_cfg in new_object_cfgs:
        rp = item_cfg['pos'].get('randp_scale')
        if rp is None:
            continue
        if all(abs(v) < 1e-9 for v in rp):
            del item_cfg['pos']['randp_scale']          # case (a)
            continue
        anchor = tuple(round(v, 4) for v in item_cfg['pos']['ppos']['p'][:2])
        anchor_groups[anchor].append(item_cfg)
    #     Widening alone is not enough, because the official loop draws
    #     uniformly and rejects: it needs a large feasible *measure*, not a
    #     feasible point. Monte-Carlo of that exact loop (100 tries/item,
    #     0.15m Chebyshev-XY separation, 0.05m minor axis) over a length-L
    #     span: n=3 still fails 3.7% of scenes at L=0.54 (= the previous
    #     `0.18·n` bound) and n=4 fails 13.8%, n=5 24.0%. That residual killed pilot sample
    #     train/00341 (3 items on one rack anchor), and since the official
    #     failure path is `exit(0)` rather than a raise, it also truncated the
    #     rest of that render chunk. Widening further would push items off the
    #     shelf (0.30·n = 0.90m vs the 0.54m span the layout declares), so
    #     instead each item gets its own disjoint SLOT of the declared span:
    #     slot = L/n, jitter = slot - 0.16, anchor offset = i·slot. Adjacent
    #     draws are then ≥0.16m apart by construction, the first draw always
    #     succeeds, and every item stays inside the official position area.
    #     Widening is kept only as the fallback when the declared span cannot
    #     hold n items at all (slot < 0.16m); order is the stable config order,
    #     so the result is still byte-reproducible.
    #     The rule is GATED on measured infeasibility: a group whose declared
    #     span already makes the official loop safe is left byte-untouched.
    #     Span at which the Monte-Carlo failure rate drops below ~1e-3:
    #     n=2 0.32m, n=3 0.62m, n=4 0.80m, n=5 1.10m -> the slightly
    #     conservative closed form 0.30·(n-1)+0.02 covers all of them. Gating
    #     matters beyond saving work: group membership is realization-
    #     dependent (an S4 pair's target sits on the counter anchor in one
    #     realization and in the cabinet in the other), so every touch risks
    #     perturbing a NON-target asset and breaking §7.10 pair invariance.
    # (c) items whose one-sided reach boxes overlap in both X and Y while
    #     sitting at DIFFERENT anchors (a long counter sweep such as
    #     rp=[2.35, 0.42] passing under a rack column). Anchor slotting is
    #     blind to these — they are separate groups by construction — so
    #     they are merged by reach-box overlap instead and re-slotted by
    #     interval scheduling; see ``resolve_cross_anchor_conflicts``.
    #     Measured over the 1160 P7.5b pilot configs: 23 sat above the 1e-4
    #     failure bar (worst ood_x/00032 at 0.28, ood_s/00114 at 0.11 —
    #     the one that actually failed during the P7.5b re-render), and
    #     after this pass, none do. It touches 27 configs (2.3%), not the
    #     ~121 the deferred note feared, because it is gated on measured
    #     infeasibility rather than on overlap. Retries did mask these
    #     (`_job_np_seed(config, attempt)` makes each attempt independent),
    #     but only at 3× the render cost of the loser; landed for the P7.5c
    #     ramp because re-slotting invalidates renders, which is free only
    #     when everything is re-rendered anyway.
    _MIN_SEP = _SLOT_MIN_SEP              # official check is 0.15, exclusive

    def _required_span(n):
        return 0.30 * (n - 1) + 0.02

    for anchor, group in anchor_groups.items():
        if len(group) < 2:
            continue
        n = len(group)
        rp0 = group[0]['pos']['randp_scale']
        ax = 0 if abs(rp0[0]) >= abs(rp0[1]) else 1
        span = max(abs(item['pos']['randp_scale'][ax]) for item in group)
        if span >= _required_span(n):
            continue                       # already feasible: do not touch
        sign = 1 if rp0[ax] >= 0 else -1
        span = max(span, _MIN_SEP * n)     # fallback widening if too narrow
        slot = span / n
        jitter = slot - _MIN_SEP
        for i, item_cfg in enumerate(group):
            lim = _slot_bound(cabinet_bounds, item_cfg, ax)
            p = item_cfg['pos']['ppos']['p'][ax] + sign * i * slot
            rp = list(item_cfg['pos']['randp_scale'])
            rp[ax] = sign * jitter
            # a cabinet member's slot may not leave the carcass (see
            # ``fit_in_cabinet``): clamp the whole reach box back in, exactly
            # as the fit pass would have, rather than the anchor alone.
            lo, hi, _ok = _clamp_interval(min(p, p + rp[ax]),
                                          max(p, p + rp[ax]), *lim)
            if lim[1] < float('inf'):
                p, rp[ax] = lo, hi - lo
            item_cfg['pos']['ppos']['p'][ax] = p
            if all(abs(v) < 1e-9 for v in rp):
                # no jitter left on any axis: exact slot placement. Dropping
                # the key makes the official loop skip collision sampling
                # entirely (see case (a)), which is exactly what we want here.
                del item_cfg['pos']['randp_scale']
            else:
                item_cfg['pos']['randp_scale'] = rp

    resolve_cross_anchor_conflicts(new_object_cfgs, new_agent_cfgs,
                                   bounds=cabinet_bounds)   # case (c)

    # Cases (b)/(c) may push a member past its declared reach (the
    # `max(occupied) + min_sep` fallback), so re-audit rather than assume.
    # Audit-only on purpose: re-clamping HERE would undo a separation those
    # passes just established and can end in the official loop's exit(0),
    # which segfaults sapien mid-build — the bound is handed to them instead
    # (`fit_in_cabinet`'s `bounds`) so they can respect it while they still
    # have the freedom to schedule around it. Measured over the 1160-config
    # pilot: before that plumbing this fired on 3/128 cabinet scenes
    # (ood_x/00023 tomato 16cm out through a closed door, /00027, /00029),
    # after it on 0. What remains is `plate` (20/128), which no carcass can
    # hold on any yaw — see `fit_in_cabinet` and docs/benchmark.md §9.
    cabinet_escapes = [r for r in (cabinet_containment_report(it, interior)
                                   for it in cabinet_items) if r
                       and r["overhang"] > 1e-6]

    # ---- style selection: the S4 hook lives here -------------------------
    # official: in_cabinet -> transparent {4, 11} so hidden items stay
    # visible. S4 instead keys opacity on the cabinet fixture itself, not
    # on target placement: counter/cabinet counterfactuals therefore share
    # an opaque style. S1-S3 keep the official transparent control.
    opaque = bool(inst.get("meta", {}).get("opaque_container"))
    style_rng = _render_rng("style")
    if scene_has_cabinet and opaque:
        pool = [st for st in OPAQUE_STYLES
                if not avoid_wooden or st in NON_WOODEN_STYLES]
        style_idx = style_rng.choice(pool)
    elif in_cabinet:
        style_idx = style_rng.choice(TRANSPARENT_STYLES)
    elif avoid_wooden:
        style_idx = style_rng.choice(NON_WOODEN_STYLES)
    else:
        style_idx = style_rng.randint(0, 11)
    cfg['scene']['env']['style_idx'] = style_idx

    cfg['agents'] = new_agent_cfgs
    cfg['objects'] = new_object_cfgs
    cfg['cameras']['human_render'] = [
        _render_rng("camera").choice(cfg['cameras']['human_render'])]

    gt = {k: v for k, v in inst.items() if k != "meta"}
    cfg['gt'] = gt
    if cabinet_fits:
        cfg['locus_cabinet_fit'] = {
            "items": [r["name"] for r in cabinet_fits],
            "yawed": [r["name"] for r in cabinet_fits if r["yaw"]],
            "uncontained": [r["name"] for r in cabinet_fits
                            if not r["contained"]],
            "escaped": [r["name"] for r in cabinet_escapes],
        }
    moved = [r for r in wall_clears if r["shift"] > 0]
    if moved:
        cfg['locus_wall_clear'] = {
            "items": [r["name"] for r in moved],
            "walls": sorted({s for r in moved for s in r["walls"]}),
            "max_shift": round(max(r["shift"] for r in moved), 6),
            "infeasible": [r["name"] for r in moved if not r["feasible"]],
        }

    for idx, agent_cfg in enumerate(new_agent_cfgs):
        uid = agent_cfg['robot_uid']
        if uid.startswith('R') and uid.split('-')[0][1:].isdigit() \
                and 'panda' in uid:
            new_agent_cfgs[idx]['robot_uid'] = f'panda-{c}'
            c += 1
    return cfg, style_idx


def _relocate_reusable_renders(split_dir, config_dir, split, planned):
    """Move already-rendered artifacts to the index their config now occupies.

    `sample_index` is a flat position within a split, so adding samples to an
    earlier plan row shifts every later sample's index even when its config is
    byte-identical -- `per_row_rng` keeps each row's *content* stable, not its
    position. Renders are addressed by index, so without this pass the ramp
    deletes (as stale) and re-renders configs it already has on disk: bumping
    the train tier from 6.3k to 7.2k, which docs/YOLO.md §6.2 prefers when a
    detector-val gate misses, would redo all 6.3k images for ~900 new ones.

    Matching is on config text, which `render_batch` already treats as the
    render's identity, so this reuses exactly what the stale-check would have
    kept had the index not moved. Returns the number of relocated samples.
    """
    if not os.path.isdir(config_dir):
        return 0
    existing = {}
    for fn in sorted(os.listdir(config_dir)):
        stem, ext = os.path.splitext(fn)
        if ext != ".yaml" or not stem.startswith(f"{split}_"):
            continue
        tail = stem[len(split) + 1:]
        if not tail.isdigit():
            continue
        with open(os.path.join(config_dir, fn)) as f:
            # First index wins if two samples share a config: one render can
            # only be moved to one destination, and the loser just re-renders.
            existing.setdefault(f.read(), int(tail))
    if not existing:
        return 0

    def _artifacts(idx):
        img = os.path.join(split_dir, "images", f"{idx:05d}.png")
        return [img, os.path.join(split_dir, "gt_labels", f"{idx:05d}.json"),
                img + ".inprogress"]

    moves, claimed = [], set()
    for idx, cfg_text in planned:
        old = existing.get(cfg_text)
        if old is None or old == idx or old in claimed:
            continue
        if not os.path.isfile(_artifacts(old)[0]):
            continue            # nothing rendered there yet; let it render
        claimed.add(old)
        moves.append((old, idx))
    if not moves:
        return 0

    # Two phases: destinations may still be occupied by renders that are
    # themselves sources of a later move (a ramp permutes indices, so moves
    # form chains and cycles). Park every source aside first, then land them.
    for old, idx in moves:
        for path in _artifacts(old):
            if os.path.isfile(path):
                os.replace(path, path + ".relocating")
    for old, idx in moves:
        for src, dst in zip(_artifacts(old), _artifacts(idx)):
            if os.path.isfile(src + ".relocating"):
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                os.replace(src + ".relocating", dst)
    print(f"[render_batch] {split}: reused {len(moves)} render(s) whose "
          f"sample_index moved", flush=True)
    return len(moves)


def render_batch(instances, out_dir, split, dry_run=False,
                 config_dir=None, export_masks=False, on_missing="raise",
                 wall_clear_scope=None):
    """Write scene configs (and optionally render) for one split.

    Rendering shells out to the official ``script/generate_planning_data.py``
    per config (the official ``solve`` hard-exits after one snapshot, so
    one subprocess per instance is the official cadence too). Each snapshot
    is moved to ``<out_dir>/<split>/images/<sample_index>.png``. With
    ``dry_run=True`` only configs + manifest are written.

    With ``export_masks=True`` rendering instead goes through
    ``locus_render_masked.py`` (chunked subprocesses, official env
    construction replicated in-process): each snapshot additionally gets
    a renderer-GT label JSON in ``<out_dir>/<split>/gt_labels/`` captured
    from the SAME camera and frame (docs/YOLO.md §6.3). The interpreter is
    ``$LOCUS_RENDER_PYTHON`` if set (the dedicated ``locus-render`` env),
    else ``sys.executable``. ``on_missing="mark"`` downgrades per-sample
    render failures from an exception to a manifest ``rendered: False`` row
    plus a warning (long pilot runs report all failures at the end).
    ``wall_clear_scope`` is forwarded to ``build_render_config``.
    """
    # Absolute before anything derives from it: the mask worker and the
    # official generator are both spawned with ``cwd=_ROBOFACTORY``, so a
    # relative ``out_dir`` makes every path handed to them resolve under
    # RoboFactory/ instead. That failure is not fast -- it surfaces as
    # FileNotFoundError on the jobs JSON once per chunk, after the config
    # pre-pass, i.e. ~17 min into a 15 h run for the 6.3k ramp.
    out_dir = os.path.abspath(out_dir)
    split_dir = os.path.join(out_dir, split)
    images_dir = os.path.join(split_dir, "images")
    config_dir = os.path.abspath(config_dir) if config_dir \
        else os.path.join(split_dir, "configs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    manifest = []
    mask_jobs = []
    demo_dir = os.path.join(_ROBOFACTORY, "demos", "VikiBenchTask")
    # Pre-pass: configs are deterministic, so build them all before touching
    # disk. `_relocate_reusable_renders` needs the whole planned set to tell a
    # moved sample from a genuinely new one, and it must run before the loop's
    # stale-check deletes a render that merely changed index.
    planned = []
    for inst in instances:
        cfg, style_idx = build_render_config(
            inst, wall_clear_scope=wall_clear_scope)
        inst["meta"]["style_idx"] = style_idx
        planned.append((inst, cfg, style_idx, yaml.dump(cfg)))
    if not dry_run:
        _relocate_reusable_renders(
            split_dir, config_dir, split,
            [(i["meta"]["sample_index"], text) for i, _, _, text in planned])

    for inst, cfg, style_idx, cfg_text in planned:
        idx = inst["meta"]["sample_index"]
        cfg_path = os.path.join(config_dir, f"{split}_{idx:05d}.yaml")
        image_rel = f"images/{idx:05d}.png"
        # Config is the render's identity: if the (deterministic) generation
        # now produces a different config than the one on disk — e.g. after a
        # placement-feasibility fix — any existing snapshot/labels for this
        # sample are stale and must not survive skip-existing resume.
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                if f.read() != cfg_text:
                    for stale in (os.path.join(split_dir, image_rel),
                                  os.path.join(split_dir, "gt_labels",
                                               f"{idx:05d}.json"),
                                  os.path.join(split_dir, image_rel)
                                  + ".inprogress"):
                        if os.path.isfile(stale):
                            os.remove(stale)
        with open(cfg_path, "w") as f:
            f.write(cfg_text)
        entry = {
            "sample_index": idx,
            "task_id": inst["task_id"],
            "cell": inst["meta"]["cell"],
            "bench_family": inst["meta"]["bench_family"],
            "style_idx": style_idx,
            "opaque_container": inst["meta"]["opaque_container"],
            "layout_id": inst["layout_id"],
            "counterfactual_pair_id": inst["meta"].get(
                "counterfactual_pair_id"),
            "style_seed": inst["meta"].get("style_seed"),
            "state_realization": inst["meta"].get("state_realization"),
            "config": os.path.relpath(cfg_path, split_dir),
            "image": image_rel,
            "rendered": False,
        }
        if not dry_run and export_masks:
            entry["mask_json"] = f"gt_labels/{idx:05d}.json"
            entry["label_source"] = "renderer_gt"
            mask_jobs.append({
                "config": os.path.abspath(cfg_path),
                "image": os.path.join(split_dir, image_rel),
                "json_dir": os.path.join(split_dir, "gt_labels"),
            })
        elif not dry_run:
            before = set(os.listdir(os.path.join(demo_dir, "images"))) \
                if os.path.isdir(os.path.join(demo_dir, "images")) else set()
            subprocess.run(
                [sys.executable, "script/generate_planning_data.py",
                 cfg_path],
                check=True, cwd=_ROBOFACTORY,
            )
            after = set(os.listdir(os.path.join(demo_dir, "images")))
            new = sorted(after - before)
            if len(new) != 1:
                raise RuntimeError(
                    f"expected exactly one new snapshot, got {new}")
            os.replace(os.path.join(demo_dir, "images", new[0]),
                       os.path.join(split_dir, image_rel))
            entry["rendered"] = True
        manifest.append(entry)

    if mask_jobs:
        render_python = os.environ.get("LOCUS_RENDER_PYTHON", sys.executable)
        # Chunked subprocesses: bounds per-process lifetime (GPU allocator
        # fragmentation over hundreds of env create/close cycles) while still
        # amortizing the render env's import cost over many configs.
        chunk = max(1, int(os.environ.get("LOCUS_RENDER_CHUNK", "150")))
        # One coherent summary set per run: stale jobs/summary files from a
        # previous run's different chunk size would double-count in
        # aggregation (resume state lives in the product files, not here).
        for fn in os.listdir(split_dir):
            if fn.startswith(("mask_export_jobs_", "mask_export_summary_")):
                os.remove(os.path.join(split_dir, fn))
        for start in range(0, len(mask_jobs), chunk):
            part = mask_jobs[start:start + chunk]
            jobs_path = os.path.join(
                split_dir, f"mask_export_jobs_{start:05d}.json")
            with open(jobs_path, "w") as f:
                json.dump(part, f, indent=1)
            # check=False + retry: the worker dies mid-chunk on two distinct
            # paths — sapien SIGSEGV after many env create/close cycles, and
            # the official placement loop's `exit(0)` (scene_builder.py:213)
            # when 100 random draws fail the 0.15m XY separation check.
            # The latter exits with status *0*, so the return code cannot be
            # used to detect an aborted chunk (that mistake silently dropped
            # 108 of chunk 300's jobs in the first P7.5b run). Progress on the
            # product files is the only ground truth: keep relaunching while
            # the worker still advances. --skip-existing resumes from the
            # crash point, and its per-job `.inprogress` attempt markers make
            # a hopeless job give up (SKIP-CRASHED) after 3 reseeded tries, so
            # the loop terminates: every pass either finishes jobs or burns a
            # marker attempt.
            summary_path = os.path.join(
                split_dir, f"mask_export_summary_{start:05d}.json")

            def _pending():
                out = []
                for job in part:
                    stem = os.path.splitext(os.path.basename(job["image"]))[0]
                    label = os.path.join(job["json_dir"], stem + ".json")
                    if not (os.path.isfile(job["image"])
                            and os.path.isfile(label)):
                        out.append(job["image"])
                return out

            prev_pending = None
            for attempt in range(_MAX_CHUNK_ATTEMPTS):
                proc = subprocess.run(
                    [render_python,
                     os.path.join(_SCRIPT_DIR, "locus_render_masked.py"),
                     "--jobs", jobs_path, "--skip-existing",
                     "--summary", summary_path],
                    check=False, cwd=_ROBOFACTORY,
                )
                pending = _pending()
                if not pending:
                    break
                # No progress AND nothing left to retry (every pending job has
                # exhausted its markers, i.e. the worker itself gave up and
                # recorded the error) -> stop; the failure list below reports.
                stalled = pending == prev_pending and not any(
                    os.path.isfile(img + ".inprogress") for img in pending)
                print(f"[render_batch] WARNING: mask worker chunk {start} "
                      f"exited {proc.returncode} with {len(pending)} job(s) "
                      f"unfinished (attempt {attempt + 1}/"
                      f"{_MAX_CHUNK_ATTEMPTS}); "
                      f"{'giving up' if stalled else 'resuming chunk'}",
                      flush=True)
                if stalled:
                    break
                prev_pending = pending
        failures = []
        for entry in manifest:
            if entry.get("label_source") != "renderer_gt":
                continue
            image_path = os.path.join(split_dir, entry["image"])
            json_path = os.path.join(split_dir, entry["mask_json"])
            if os.path.isfile(image_path) and os.path.isfile(json_path):
                entry["rendered"] = True
            else:
                failures.append(entry["sample_index"])
        if failures:
            msg = (f"mask-export render incomplete for {len(failures)} "
                   f"sample(s) in split {split!r}: {failures[:20]}"
                   f"{'...' if len(failures) > 20 else ''} "
                   f"(see mask_export_summary_*.json for per-job errors)")
            if on_missing == "raise":
                raise RuntimeError(msg)
            print(f"[render_batch] WARNING: {msg}", flush=True)

    with open(os.path.join(split_dir, "render_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# parquet packing (VIKI-L2 column parity)
# ---------------------------------------------------------------------------

def _system_prompt(robots):
    robots_clean = {slot: r for slot, r in robots.items() if r}
    chars = {r: ROBOT_DESCRIPTION[r] for r in dict.fromkeys(
        robots_clean.values())}
    apis = {r: AGENT_AVAIL_ACTIONS[r] for r in dict.fromkeys(
        robots_clean.values())}
    return (VIKI_SYSTEM_PREFIX
            + f"Available robot set: {robots_clean}\n"
            + f"Robot characteristics: {chars}\n"
            + f"Their available operation APIs: {apis}\n")


def _to_official_gt(inst):
    """Instance → the official parquet gt dict shape (R1..R3 padding,
    per-slot None in time_steps), with LOCUS meta riding along."""
    robots = {"R1": None, "R2": None, "R3": None}
    robots.update(inst["robots"])
    time_steps = []
    for i, step in enumerate(inst["ground_truth"]):
        actions = {"R1": None, "R2": None, "R3": None}
        for r, op in step.items():
            actions[r] = list(op)
        time_steps.append({"step": i + 1, "actions": actions})
    return {
        "description": inst["description"],
        "goal_constraints": inst["goal_constraints"],
        "idle_robots": list(inst.get("idle_robots", [])),
        "init_pos": {k: (list(v) if v is not None else None)
                     for k, v in inst["init_pos"].items()},
        "layout_id": inst["layout_id"],
        "robots": robots,
        "task_id": inst["task_id"],
        "task_name": inst["task_name"],
        "temporal_constraints": inst.get("temporal_constraints", []),
        "time_steps": time_steps,
        "locus_meta": inst["meta"],
    }


def pack_parquet(instances, split_dir, out_path=None,
                 require_images=True):
    """Pack one split into a VIKI-L2 column-parity parquet.

    Columns byte-compatible with ``data/viki/VIKI-L2/*.parquet``:
    images (list of {bytes, path}), data_source, prompt (system+user),
    ability, reward_model ({ground_truth, style}). ``data_source`` is
    ``locus_bench`` so downstream reward routing can distinguish it.
    """
    import pandas as pd

    rows = []
    for inst in instances:
        idx = inst["meta"]["sample_index"]
        image_path = os.path.join(split_dir, "images", f"{idx:05d}.png")
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        elif require_images:
            raise FileNotFoundError(image_path)
        else:
            image_bytes = b""
        rows.append({
            "images": [{"bytes": image_bytes, "path": None}],
            "data_source": "locus_bench",
            "prompt": [
                {"content": _system_prompt(inst["robots"]), "role": "system"},
                {"content": "<image>" + inst["description"], "role": "user"},
            ],
            "ability": "Task Planning",
            "reward_model": {
                "ground_truth": _to_official_gt(inst),
                "style": "rule",
            },
        })
    df = pd.DataFrame(rows,
                      columns=["images", "data_source", "prompt",
                               "ability", "reward_model"])
    out_path = out_path or os.path.join(split_dir, "data.parquet")
    df.to_parquet(out_path)
    return out_path


# ---------------------------------------------------------------------------
# decidability spot check (§6.3-3, GT-placement level)
# ---------------------------------------------------------------------------

def decidability_check(instances):
    """From GT placement (no detector): every S1–S3 judgment candidate must
    resolve to a visible position area; every S4 hidden item must sit inside
    the cabinet with an opaque style. Returns (ok, violations)."""
    violations = []
    for inst in instances:
        meta = inst["meta"]
        s = meta["cell"][0]
        if s == 0:
            continue
        tpl_positions = {}
        for name, pos in inst["init_pos"].items():
            if name.startswith('R') and name[1:].isdigit():
                continue
            tpl_positions[name.rsplit('_', 1)[0]] = list(pos or [])
        # find each candidate's realized position via mask_map
        spec_realization = meta["state_realization"]
        for mask_key, state in spec_realization.items():
            value = meta["mask_map"].get(mask_key)
            if value is None or value not in tpl_positions:
                continue
            positions = tpl_positions[value]
            if s == 4 and state == "cabinet":
                if positions != ["cabinet"]:
                    violations.append((inst["task_name"], value,
                                       "hidden item not in cabinet",
                                       positions))
                if meta.get("style_idx") is not None \
                        and meta["style_idx"] in TRANSPARENT_STYLES:
                    violations.append((inst["task_name"], value,
                                       "S4 rendered with transparent style",
                                       meta["style_idx"]))
            else:
                bad = [p for p in positions if p not in VISIBLE_AREAS
                       and p != "cabinet"
                       and not (p.startswith('R') and p[1:].isdigit())
                       # containers on visible surfaces count as visible
                       and p not in tpl_positions]
                if bad:
                    violations.append((inst["task_name"], value,
                                       "candidate in non-visible area", bad))
                if s < 4 and positions == ["cabinet"] \
                        and not meta["opaque_container"]:
                    # visible-cabinet (transparent style) is official
                    # behavior; fine. Flag only if the style went opaque.
                    if meta.get("style_idx") in OPAQUE_STYLES:
                        violations.append((inst["task_name"], value,
                                           "S<4 cabinet item with opaque "
                                           "style", meta["style_idx"]))
    return not violations, violations


if __name__ == "__main__":
    plan = build_split_plan(n_train=2, n_test_id=2, n_ood=2)
    splits, stats = generate_split_instances(plan, seed=0)
    out_dir = os.path.join(_ROBOFACTORY, "script_general", "locus_pack_smoke")
    total = 0
    for split, instances in splits.items():
        manifest = render_batch(instances, out_dir, split, dry_run=True)
        pq = pack_parquet(instances, os.path.join(out_dir, split),
                          require_images=False)
        ok, viol = decidability_check(instances)
        total += len(instances)
        print(f"{split}: {len(instances)} inst, parquet={os.path.basename(pq)}, "
              f"decidability={'ok' if ok else viol[:2]}")
    print(f"total {total} | stats {stats}")
