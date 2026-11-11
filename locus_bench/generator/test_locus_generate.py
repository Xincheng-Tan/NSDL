"""P7.2 acceptance test (docs/benchmark.md §9 / §4.2 / §4.3):

  1. Balanced realization sampling: per-(family, cell) max realization share
     within the §4.3 tolerance (min-count sampler ⇒ counts differ by ≤1)
  2. GT still replays success in the official SimEnv WITH distractors
     injected (distractors are scene-legal: GT never touches them)
  3. Meta accounting complete (bench_family/cell/state_realization/
     paraphrase_id/mask_map/distractors/empty_plan) and consistent with the
     instance; realization never leaks into description/task_name
  4. Distractor discipline: only S2/S3, 1..3 per scene, same-tag extras only
     at the destination, F5's cleared surface never hosts clutter
  5. paraphrase holdout: allowed_paraphrase_ids restricts sampling
  6. Same seed ⇒ byte-identical batch (reproducibility)

Run:  cd RoboFactory && python script_general/test_locus_generate.py
"""

import json
import os
import random
import sys
from collections import Counter

import pytest

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROBOFACTORY)
sys.path.insert(0, os.path.join(_ROBOFACTORY, "script_general"))

from locus_generate import (  # noqa: E402
    NEUTRAL_CLUTTER, MaskVocabSampler, RealizationSampler,
    STRATIFY_COUNT_FACTOR, audit_mask_keys, generate_locus_batch,
    instantiate_counterfactual_pair, instantiate_locus_task,
    realization_balance_report, stratified_mask_keys,
)
from locus_task_pool import (  # noqa: E402
    ALL_LAYOUTS, FIXTURE_MASK_LAYOUTS, MASK_VOCAB_WEIGHTS,
    enumerate_realizations, iter_cells, layouts_for_masks, make_template,
    weighted_mask_choice,
)
from test_locus_task_pool import _replay  # noqa: E402

BATCH = generate_locus_batch(n_per_cell=8, seed=0)


def test_realization_balance():
    report = realization_balance_report(BATCH)
    assert len(report) == len(list(iter_cells()))
    bad = {k: v for k, v in report.items() if not v["ok"]}
    assert not bad, f"balance violations: {bad}"
    # min-count sampler: counts within a cell differ by at most 1
    for key, entry in report.items():
        counts = list(entry["counts"].values())
        assert max(counts) - min(counts) <= 1, (key, counts)


def test_gt_replays_with_distractors():
    rng = random.Random(123)
    failures = []
    n_with_distractors = 0
    for inst in BATCH:
        if inst["meta"]["distractors"]:
            n_with_distractors += 1
        ok, err = _replay(inst, rng)
        if not ok:
            failures.append((inst["task_name"], inst["meta"]["cell"],
                             inst["meta"]["state_realization"],
                             inst["meta"]["distractors"], err))
    assert not failures, f"{len(failures)} failures; first 3: {failures[:3]}"
    assert n_with_distractors > 0, "no instance received distractors"


def test_meta_accounting():
    for inst in BATCH:
        meta = inst["meta"]
        for key in ("bench_family", "cell", "state_realization",
                    "paraphrase_id", "mask_map", "distractors",
                    "instantiate_seed", "style_seed", "empty_plan",
                    "opaque_container", "state_entailed_open",
                    "base_mechanical_signature", "bridge_kind",
                    "bridge_variant", "counterfactual_pair_id",
                    "observation_tool"):
            assert key in meta, (inst["task_name"], key)
        s, m = meta["cell"]
        assert inst["task_name"] == f'{meta["bench_family"]}_s{s}m{m}'
        assert meta["empty_plan"] == (not inst["ground_truth"])
        assert "present" not in inst["task_name"]
        # distractor assets really are in init_pos and never in GT targets
        gt_targets = {t for step in inst["ground_truth"]
                      for op in step.values() for t in op[1:]}
        for d in meta["distractors"]:
            assert any(k.rsplit('_', 1)[0] == d["name"]
                       for k in inst["init_pos"]), d
            assert d["name"] not in gt_targets, (inst["task_name"], d)


def test_distractor_discipline():
    for inst in BATCH:
        meta = inst["meta"]
        s = meta["cell"][0]
        ds = meta["distractors"]
        if s not in (2, 3):
            assert not ds, (inst["task_name"], ds)
            continue
        assert 1 <= len(ds) <= 3, (inst["task_name"], len(ds))
        for d in ds:
            if d["kind"] == "clutter":
                assert d["name"] in NEUTRAL_CLUTTER, d
            else:
                assert d["kind"] == "extra_present_same_tag", d
                assert meta["bench_family"] in ("F3", "F5"), d
                assert d["pos"] in ("table", "rack"), d
            if meta["bench_family"] == "F5":
                assert d["pos"] != "kitchen work area", (inst["task_name"], d)


def test_paraphrase_holdout():
    rng = random.Random(9)
    sampler = RealizationSampler()
    for _ in range(20):
        inst = instantiate_locus_task(
            "F2", 2, 1, rng, realization=sampler.next("F2", 2, 1, rng),
            allowed_paraphrase_ids=[0, 1, 2])
        assert inst["meta"]["paraphrase_id"] in (0, 1, 2)
    inst = instantiate_locus_task("F2", 2, 1, rng, paraphrase_id=5)
    assert inst["meta"]["paraphrase_id"] == 5


def test_no_realization_leak_in_instances():
    """Same (cell, paraphrase, masks) with opposite realizations must yield
    the identical instruction string — the answer is only in the scene."""
    for fam, s, m in iter_cells():
        reals = enumerate_realizations(fam, s, m)
        if len(reals) < 2:
            continue
        descs = set()
        for realization in (reals[0], reals[-1]):
            rng = random.Random(77)   # same rng ⇒ same masks/paraphrase
            inst = instantiate_locus_task(
                fam, s, m, rng, realization=realization,
                inject_distractors=False)
            descs.add(inst["description"])
        assert len(descs) == 1, (fam, s, m, descs)


def test_reproducibility():
    a = generate_locus_batch(n_per_cell=2, seed=42)
    b = generate_locus_batch(n_per_cell=2, seed=42)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_oven_samples_only_land_on_oven_bearing_layouts():
    """Groundability: only the robocasa "large" layouts declare `type: oven`,
    and layout 3 loses it again to customized_layout_3.yaml — so an F10 sample
    whose mask1 is `oven` MUST render on 1/6/8. Pilot P7.5a shipped three
    un-groundable ones (train 00585 on L3, 00586 on L2, 00590 on L4): the
    instruction named an oven the scene never built."""
    rng = random.Random(3)
    seen = set()
    for _ in range(120):
        inst = instantiate_locus_task("F10", 0, 0, rng)
        target = inst["meta"]["mask_map"]["mask1"]
        seen.add(target)
        if target == "oven":
            assert inst["layout_id"] in FIXTURE_MASK_LAYOUTS["oven"]
            assert inst["meta"]["fixture_layout_masks"] == ["oven"]
        else:
            # a non-restricted target must keep the full pool, so the layout
            # distribution of every other sample is untouched by this rule
            assert inst["meta"]["fixture_layout_masks"] == []
    assert "oven" in seen and len(seen) > 1


def test_layouts_for_masks_reports_and_never_widens():
    # unrestricted masks pass the pool through untouched
    assert layouts_for_masks(["apple", "kitchen work area"]) == (ALL_LAYOUTS, [])
    # the restricting mask is named, so meta can record WHY
    assert layouts_for_masks(["oven", "apple"]) == ([1, 6, 8], ["oven"])
    # intersection with an incompatible caller pool comes back EMPTY rather
    # than silently falling back to the full pool: the caller must reject
    assert layouts_for_masks(["oven"], [2, 4]) == ([], ["oven"])


def test_ungroundable_mask_layout_combination_raises():
    """A mask/pool combination with no valid layout must fail loudly. Silent
    widening is what shipped the three bad pilot samples."""
    raised = 0
    for seed in range(60):
        try:
            instantiate_locus_task("F10", 0, 0, random.Random(seed),
                                   allowed_layouts=[2, 4, 5])
        except ValueError as exc:
            assert "oven" in str(exc) and "groundabl" in str(exc)
            raised += 1
    # F10's mask1 is one of drawer/fridge/oven, so over 60 seeds the oven draw
    # (and hence the raise) must occur — a silently-passing loop would mean the
    # guard never fires.
    assert raised > 0


def test_empty_plan_realizations_reachable():
    """Balanced sampling must actually surface the all-present (empty plan)
    realizations at benchmark scale."""
    n_empty = sum(1 for inst in BATCH if inst["meta"]["empty_plan"])
    assert n_empty > 0


def test_weighted_mask_choice_is_uniform_path_when_weights_tie():
    """RNG-consumption contract: an unweighted list must consume the rng
    exactly like `rng.choice`, so cells with no weighted vocab keep their
    already-rendered configs valid (docs/YOLO.md 6.2 blast radius)."""
    values = ["apple", "pear", "peach", "banana"]
    for seed in range(50):
        a = weighted_mask_choice(random.Random(seed), values)
        b = random.Random(seed).choice(values)
        assert a == b, (seed, a, b)
        # and the rng must be left in the same state
        r1, r2 = random.Random(seed), random.Random(seed)
        weighted_mask_choice(r1, values)
        r2.choice(values)
        assert r1.getstate() == r2.getstate(), seed


def test_weighted_mask_choice_skews_toward_weighted_values():
    """The three single-list detector classes must actually get lifted."""
    values = ["pizza", "wine", "kettle", "scissors", "pumpkin"]
    rng = random.Random(0)
    counts = Counter(weighted_mask_choice(rng, values) for _ in range(20000))
    w = MASK_VOCAB_WEIGHTS
    total = sum(w.get(v, 1.0) for v in values)
    for value in values:
        expected = w.get(value, 1.0) / total
        observed = counts[value] / 20000
        assert abs(observed - expected) < 0.02, (value, observed, expected)
    # the lifted pair must outdraw the unweighted mates
    assert counts["wine"] > counts["kettle"]
    assert counts["pizza"] > counts["pumpkin"]


def test_weighted_mask_choice_keeps_full_vocab_support():
    """Weighting may reshape the distribution but must never zero a value out,
    or the 7-3 vocabulary cross-coverage audit ('0 holes') would break."""
    values = ["pizza", "wine", "kettle", "scissors", "pumpkin"]
    rng = random.Random(1)
    seen = {weighted_mask_choice(rng, values) for _ in range(5000)}
    assert seen == set(values), sorted(set(values) - seen)
    # an all-equal table (including all-zero) is uniform by definition
    assert weighted_mask_choice(random.Random(0), ["a", "b"],
                                weights={"a": 0.0, "b": 0.0}) in ("a", "b")
    # a negative weight is a mis-specified table, not a distribution
    with pytest.raises(ValueError):
        weighted_mask_choice(random.Random(0), ["a", "b"],
                             weights={"a": -1.0, "b": 1.0})


def test_weighted_vocab_lifts_starved_classes_in_batch():
    """End-to-end: at batch scale the lifted classes must beat the uniform
    rate inside the very list that starved them.

    Scoped to F1's mask2 (`_F1_ITEMS_B`) on purpose -- `scissors`/`pumpkin`
    are also drawn from F5/F8 lists, so pooling every mask_map value would
    compare draws that never competed with `wine`/`pizza`.
    """
    f1b = {"pizza", "wine", "kettle", "scissors", "pumpkin"}
    # own batch, not the shared 8/cell BATCH: only 4 cells carry F1 mask2, so
    # BATCH yields 32 draws and the 6/9 share is inside sampling noise there.
    drawn = Counter()
    for inst in generate_locus_batch(n_per_cell=64, seed=0):
        if inst["meta"]["bench_family"] != "F1":
            continue
        value = inst["meta"]["mask_map"].get("mask2")
        if value in f1b:
            drawn[value] += 1
    total = sum(drawn.values())
    assert total > 200, f"too few F1 mask2 draws to assert a share: {total}"
    lifted = sum(n for c, n in drawn.items() if c in MASK_VOCAB_WEIGHTS)
    share = lifted / total
    # 2 lifted at w=3 vs 3 plain at w=1 -> expected share 6/9 = 0.667, well
    # clear of the uniform 2/5 = 0.4 this fix exists to move away from
    assert 0.58 < share < 0.75, (dict(drawn), share)


# --- P7.5c: stratified mask sampling for small plan rows (§7-3) ------------

def test_stratified_mask_keys_only_covers_audit_relevant_small_vocabs():
    # F9-s1m4 is the cell that produced the 4 documented holes: OOD, n=20,
    # 9-value candidate vocab -> must be selected.
    assert stratified_mask_keys("F9", 1, 4, 20) == {"mask1"}
    # the same cell with a large enough row is left alone
    assert stratified_mask_keys("F9", 1, 4, 3 * 9) == set()
    # F9-s0m4 has no candidates at all, so nothing is audit-relevant
    assert stratified_mask_keys("F9", 0, 4, 20) == set()
    # a single-value vocab is never stratified (nothing to rotate, and the
    # rng-consumption contract must stay identical)
    for fam, s, m in iter_cells():
        tpl = make_template(fam, s, m, random.Random(0))
        for key in stratified_mask_keys(fam, s, m, 1):
            assert len(tpl[key]) > 1, (fam, s, m, key)
            assert key in audit_mask_keys(tpl), (fam, s, m, key)


def test_mask_vocab_sampler_covers_vocab_and_stays_balanced():
    values = list("abcdefghi")
    sampler = MaskVocabSampler()
    rng = random.Random(0)
    drawn = Counter()
    for _ in range(len(values)):
        v = sampler.next("F9", 1, 4, "mask1", values, rng)
        sampler.commit()
        drawn[v] += 1
    # one full pass covers the vocab exactly once -- this is the property the
    # i.i.d. draw could not deliver at n=20 (docs/benchmark.md §9 P7.5c)
    assert set(drawn) == set(values), drawn
    assert max(drawn.values()) == 1
    for _ in range(len(values) + 4):
        sampler.next("F9", 1, 4, "mask1", values, rng)
        sampler.commit()
    counts = sampler.counts()[("F9", 1, 4, "mask1")]
    assert max(counts) - min(counts) <= 1, counts


def test_mask_vocab_sampler_rollback_does_not_spend_budget():
    """A SimEnv-rejected attempt must not tilt the round robin."""
    values = list("abc")
    sampler = MaskVocabSampler()
    rng = random.Random(0)
    first = sampler.next("F9", 1, 4, "mask1", values, rng)
    sampler.rollback()
    assert sampler.counts()[("F9", 1, 4, "mask1")] == [0, 0, 0]
    # after rollback every value is still equally eligible, so a full pass
    # still covers the whole vocab including the rolled-back one
    seen = set()
    for _ in range(len(values)):
        seen.add(sampler.next("F9", 1, 4, "mask1", values, rng))
        sampler.commit()
    assert seen == set(values), (first, seen)
    # staged draws within one instance don't collide either
    sampler2 = MaskVocabSampler()
    a = sampler2.next("F9", 1, 4, "mask1", values, rng)
    b = sampler2.next("F9", 1, 4, "mask1", values, rng)
    assert a != b, (a, b)


def test_stratification_is_incompatible_with_s4_pairing_and_never_hit():
    """§7.10 mask invariance: refuse rather than desync a counterfactual pair."""
    with pytest.raises(ValueError, match="counterfactual pairing"):
        instantiate_counterfactual_pair(
            "F4", 4, 1, random.Random(0),
            mask_sampler=MaskVocabSampler(),
            stratify_keys={"mask1"})
    # and the production plan never asks for it
    import locus_render_pack as pack
    plan = pack.build_split_plan(n_train=210, n_test_id=6, n_ood=20)
    for split, fam, s, m, count in plan:
        if fam == "F4" and s == 4:
            assert not stratified_mask_keys(fam, s, m, count), (split, count)


def test_stratified_draw_consumes_rng_like_the_iid_draw():
    """One draw per key either way, so unstratified keys stay byte-identical."""
    values = ["a", "b", "c"]
    r1, r2 = random.Random(7), random.Random(7)
    weighted_mask_choice(r1, values)
    MaskVocabSampler().next("F9", 1, 4, "mask1", values, r2)
    assert r1.getstate() == r2.getstate()


if __name__ == "__main__":
    for fn in [test_realization_balance,
               test_gt_replays_with_distractors,
               test_meta_accounting,
               test_distractor_discipline,
               test_paraphrase_holdout,
               test_no_realization_leak_in_instances,
               test_reproducibility,
               test_oven_samples_only_land_on_oven_bearing_layouts,
               test_layouts_for_masks_reports_and_never_widens,
               test_ungroundable_mask_layout_combination_raises,
               test_empty_plan_realizations_reachable,
               test_weighted_mask_choice_is_uniform_path_when_weights_tie,
               test_weighted_mask_choice_skews_toward_weighted_values,
               test_weighted_mask_choice_keeps_full_vocab_support,
               test_weighted_vocab_lifts_starved_classes_in_batch,
               test_stratified_mask_keys_only_covers_audit_relevant_small_vocabs,
               test_mask_vocab_sampler_covers_vocab_and_stays_balanced,
               test_mask_vocab_sampler_rollback_does_not_spend_budget,
               test_stratification_is_incompatible_with_s4_pairing_and_never_hit,
               test_stratified_draw_consumes_rng_like_the_iid_draw]:
        fn()
        print(f"{fn.__name__}: OK")
