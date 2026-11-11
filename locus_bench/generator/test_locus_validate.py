"""P7.3 acceptance test (docs/benchmark.md §6.2/§6.3/§7):

  1. Inlined SimEnv gate: validated batch generation completes; every
     emitted instance replays success; realization balance unaffected by
     reject-and-resample
  2. Φ reconstruction gate: agreement ≥ 99.5% on the pilot batch (§6.1 /
     §7-6 dual-validator key)
  3. audit_report.json carries every §7 key 1–6 computed (7–8 pending
     markers), and the computed ones all pass
  4. Empty-plan scoring semantics (§6.2):
       a. GT prediction → success  (all realizations)
       b. empty prediction on all-present realization → success
       c. empty prediction when actives remain → FAILED_GOAL_CONSTRAINT
       d. over-grasp (relocating a pre-satisfied item away) → failure
  5. Report is JSON-serializable and reproducible for a fixed seed

Run:  cd RoboFactory && python script_general/test_locus_validate.py
"""

import json
import os
import random
import sys

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROBOFACTORY)
sys.path.insert(0, os.path.join(_ROBOFACTORY, "script_general"))

from locus_generate import instantiate_locus_task  # noqa: E402
from locus_validate import (  # noqa: E402
    _vocab_coverage, build_audit_report, eval_single_locus,
    generate_validated_batch, phi_gate, simenv_replay,
)

BATCH, STATS = generate_validated_batch(n_per_cell=4, seed=0)
REPORT = build_audit_report(BATCH, STATS)


def test_validated_batch_gate():
    assert len(BATCH) == 196  # 49 explicit family-cells x 4
    rng = random.Random(1)
    for inst in BATCH:
        ok, err = simenv_replay(inst, rng=rng)
        assert ok, (inst["task_name"], err)
    # balance survives reject-and-resample
    assert REPORT["realization_balance"]["ok"], \
        REPORT["realization_balance"]["worst_max_share"]


def test_phi_agreement_gate():
    dv = REPORT["dual_validator"]
    assert dv["phi_checked"] == len(BATCH)
    assert dv["phi_agreement"] is not None
    assert dv["phi_agreement"] >= 0.995, dv["phi_failures"]
    assert dv["ok"] is True


def test_audit_report_keys():
    for key in ("official_base", "realization_balance", "vocab_coverage",
                "realization_leak_paired_check", "axis_orthogonality",
                "dual_validator", "strong_baseline_prerun", "eval_fairness"):
        assert key in REPORT, key
    for key in ("family_train_coverage", "operator_train_support",
                "composition_holdout", "s4_counterfactual_pairing"):
        assert key in REPORT, key
    assert REPORT["official_base"]["official_files_modified"] == []
    assert all(REPORT["realization_leak_paired_check"].values())
    assert REPORT["axis_orthogonality"]["ok"], REPORT["axis_orthogonality"]
    assert REPORT["strong_baseline_prerun"].startswith("pending")
    assert REPORT["family_train_coverage"]["ok"]
    assert REPORT["operator_train_support"]["lookup_count"] == 0
    assert REPORT["operator_train_support"]["structural_ok"]
    # Four samples/cell prove structural support but intentionally do not
    # pretend to satisfy the ≥100 production-scale freeze threshold.
    assert not REPORT["operator_train_support"]["scale_gate_ok"]
    assert REPORT["composition_holdout"]["ok"]
    assert REPORT["s4_counterfactual_pairing"]["ok"]
    json.dumps(REPORT)  # serializable


def test_empty_plan_scoring():
    # F2 s2m1: two candidates, realizations span all-present -> all-absent
    rng = random.Random(42)
    all_present = {"mask1": "present", "mask2": "present"}
    one_absent = {"mask1": "absent", "mask2": "present"}

    inst_ap = instantiate_locus_task("F2", 2, 1, rng,
                                     realization=all_present,
                                     inject_distractors=False)
    inst_oa = instantiate_locus_task("F2", 2, 1, rng,
                                     realization=one_absent,
                                     inject_distractors=False)

    # a. GT replays success on both
    for inst in (inst_ap, inst_oa):
        r = eval_single_locus(inst["ground_truth"], inst)
        assert r["success"], (inst["meta"]["state_realization"], r)

    # b. empty pred on all-present -> success (and GT is indeed empty)
    assert inst_ap["meta"]["empty_plan"]
    r = eval_single_locus([], inst_ap)
    assert r["success"] and r["empty_pred"] and r["empty_gt"]

    # c. empty pred with an active remaining -> goal failure
    assert not inst_oa["meta"]["empty_plan"]
    r = eval_single_locus([], inst_oa)
    assert not r["success"]
    assert r["error"] == "failed goal constraint."

    # d. over-grasp: move the pre-satisfied item AWAY from the table
    present_item = inst_oa["meta"]["mask_map"]["mask2"]
    over = [
        {"R1": ["Move", present_item]},
        {"R1": ["Reach", present_item]},
        {"R1": ["Grasp", present_item]},
        {"R1": ["Move", "kitchen work area"]},
        {"R1": ["Place", "kitchen work area"]},
    ]
    r = eval_single_locus(over, inst_oa)
    assert not r["success"], "over-grasp must fail the goal check"


def test_phi_empty_plan_trivial_pass():
    rng = random.Random(3)
    inst = instantiate_locus_task(
        "F2", 2, 1, rng,
        realization={"mask1": "present", "mask2": "present"},
        inject_distractors=False)
    ok, code = phi_gate(inst)
    assert ok and code is None


def test_report_reproducible():
    batch2, stats2 = generate_validated_batch(n_per_cell=4, seed=0)
    report2 = build_audit_report(batch2, stats2)
    assert json.dumps(REPORT, sort_keys=True) == \
        json.dumps(report2, sort_keys=True)


def test_vocab_coverage_has_no_holes_under_stratified_plan():
    """§7-3 "0 holes" on the production plan (P7.5c).

    The i.i.d. path is asserted to still show the four documented F9 holes,
    so this test fails loudly if the fix silently stops being the reason
    coverage is full.
    """
    import locus_render_pack as pack
    plan = pack.build_split_plan(n_train=210, n_test_id=6, n_ood=20)

    def coverage(stratify):
        splits, _ = pack.generate_split_instances(
            plan, seed=0, validate=False, stratify_small_vocab=stratify)
        instances = [i for v in splits.values() for i in v]
        return _vocab_coverage(instances)

    off = coverage(False)
    assert off["F9"]["coverage"] == 0.7778, off["F9"]
    assert [h[0] for h in off["F9"]["holes"]] == \
        ["banana", "peach", "pumpkin", "spoon"], off["F9"]["holes"]

    on = coverage(True)
    bad = {f: c for f, c in on.items() if c["coverage"] < 1.0}
    assert not bad, f"§7-3 vocabulary cross-coverage holes remain: {bad}"


def test_stratification_leaves_train_byte_identical():
    """The fix must not invalidate the 6300 rendered train images."""
    import locus_render_pack as pack
    plan = pack.build_split_plan(n_train=210, n_test_id=6, n_ood=20)

    def configs(stratify):
        splits, _ = pack.generate_split_instances(
            plan, seed=0, validate=False, stratify_small_vocab=stratify)
        return {(sp, i["meta"]["sample_index"]):
                json.dumps(i, sort_keys=True, default=str)
                for sp, insts in splits.items() for i in insts}

    off, on = configs(False), configs(True)
    assert set(off) == set(on), "sample index set changed"
    changed = {k for k in off if off[k] != on[k]}
    assert not [k for k in changed if k[0] == "train"], \
        "train configs changed -> 6300 renders invalidated"
    # only eval-split rows that are actually stratified may move
    assert len(changed) == 108, len(changed)


if __name__ == "__main__":
    for fn in [test_validated_batch_gate,
               test_phi_agreement_gate,
               test_audit_report_keys,
               test_empty_plan_scoring,
               test_phi_empty_plan_trivial_pass,
               test_report_reproducible,
               test_vocab_coverage_has_no_holes_under_stratified_plan,
               test_stratification_leaves_train_byte_identical]:
        fn()
        print(f"{fn.__name__}: OK")
