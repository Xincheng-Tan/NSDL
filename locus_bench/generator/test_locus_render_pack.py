"""P7.4 acceptance test (docs/benchmark.md §4.2-2/§5.1/§5.2/§6.3-3/§9):

  1. Split plan: grid cells map to the §5.1 splits; per-split instance
     generation keeps gates + balance; empty-plan realizations survive
  2. Layout holdout: train rows never use the holdout layout; test rows
     that draw it are flagged
  3. Paraphrase holdout: train never samples held-out paraphrase ids;
     ~30% of test_id rows do and are flagged
  4. S4 opaque hook: in-cabinet + opaque_container → style from the opaque
     set; S1–S3 in-cabinet keeps the official transparent {4, 11};
     cardboardbox scenes avoid wooden styles; config build is seed-stable
  5. Parquet parity: byte-identical column set + system-prompt prefix vs
     data/viki/VIKI-L2/train.parquet; official gt keys preserved (plus
     locus_meta); roundtrip read works
  6. Decidability check: passes on generated batches, and actually rejects
     a corrupted instance (negative control)
  7. (already exercised on GPU, artifacts committed under
     locus_render_real_smoke/: real render of S4 F4 → opaque style,
     S1 F8 control → transparent style, both snapshots produced; this test
     re-validates the artifacts when present)

Run:  cd RoboFactory && python script_general/test_locus_render_pack.py
"""

import copy
import hashlib
import inspect
import json
import os
import random
import sys
import tempfile

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROBOFACTORY)
sys.path.insert(0, os.path.join(_ROBOFACTORY, "script_general"))

from locus_generate import realization_balance_report  # noqa: E402
from locus_render_pack import (  # noqa: E402
    HOLDOUT_LAYOUT, OPAQUE_STYLES, TRANSPARENT_STYLES, VIKI_SYSTEM_PREFIX,
    _CABINET_INTERIOR, _clamp_interval, _coverage_probability, _is_infeasible,
    _reach_boxes_separable, _relocate_reusable_renders, _xy_reach_box,
    build_render_config, build_split_plan, cabinet_containment_report,
    decidability_check, fit_in_cabinet,
    generate_split_instances, mesh_aabb, pack_parquet,
    placement_failure_probability, render_batch,
    resolve_cross_anchor_conflicts, rest_offset_z, rotated_min_z,
    rotated_xy_box,
)
from locus_render_pack import (  # noqa: E402
    _WALL_PLANES, _WALL_SIDES, clear_of_walls, hull_xy_box, render_stem,
)
from locus_task_pool import ALL_LAYOUTS, iter_cells  # noqa: E402

PLAN = build_split_plan(n_train=3, n_test_id=3, n_ood=3)
SPLITS, STATS = generate_split_instances(PLAN, seed=0)


def test_split_plan_grid():
    got = {split: sorted({tuple(i["meta"]["cell"]) for i in insts})
           for split, insts in SPLITS.items()}
    train_cells = {(s, m) for _, s, m in iter_cells() if s <= 2 and m <= 2}
    assert set(got["train"]) == train_cells
    assert set(got["test_id"]) == train_cells
    assert all(s >= 3 and m <= 2 for s, m in got["ood_s"])
    assert all(s <= 2 and m >= 3 for s, m in got["ood_m"])
    assert all(s >= 3 and m >= 3 for s, m in got["ood_x"])
    # gates ran on everything; min-count balance (counts differ by ≤1)
    # holds per split — the §4.3 share tolerance is a batch-scale audit
    # and is checked in test_locus_generate at n_per_cell=8
    assert STATS.get("phi_fail", 0) == 0
    for split, insts in SPLITS.items():
        report = realization_balance_report(insts)
        for key, entry in report.items():
            counts = list(entry["counts"].values())
            assert max(counts) - min(counts) <= 1, (split, key, counts)
    assert any(i["meta"]["empty_plan"] for insts in SPLITS.values()
               for i in insts)


def test_layout_holdout():
    assert all(i["layout_id"] != HOLDOUT_LAYOUT for i in SPLITS["train"])
    assert all(not i["meta"]["layout_holdout"] for i in SPLITS["train"])
    for split in ("test_id", "ood_s", "ood_m", "ood_x"):
        for inst in SPLITS[split]:
            assert inst["meta"]["layout_holdout"] == \
                (inst["layout_id"] == HOLDOUT_LAYOUT)


def test_paraphrase_holdout():
    for inst in SPLITS["train"]:
        n_hold = max(1, int(round(inst["meta"]["n_paraphrases"] * 0.3)))
        assert inst["meta"]["paraphrase_id"] < \
            inst["meta"]["n_paraphrases"] - n_hold
        assert not inst["meta"]["paraphrase_holdout"]
    flagged = [i for i in SPLITS["test_id"] if i["meta"]["paraphrase_holdout"]]
    assert flagged, "no paraphrase-holdout rows in test_id"
    for inst in flagged:
        n_hold = max(1, int(round(inst["meta"]["n_paraphrases"] * 0.3)))
        assert inst["meta"]["paraphrase_id"] >= \
            inst["meta"]["n_paraphrases"] - n_hold


def test_style_hook():
    seen_opaque = seen_transparent = 0
    for insts in SPLITS.values():
        for inst in insts:
            cfg, style_idx = build_render_config(inst)
            in_cabinet = any(
                list(pos or []) == ["cabinet"]
                for name, pos in inst["init_pos"].items()
                if not (name.startswith('R') and name[1:].isdigit()))
            has_box = any(name.rsplit('_', 1)[0] == "cardboardbox"
                          for name in inst["init_pos"])
            scene_has_cabinet = any(
                name.rsplit('_', 1)[0] in ("cabinet", "kitchen cabinet")
                for name in inst["init_pos"])
            if scene_has_cabinet and inst["meta"]["opaque_container"]:
                assert style_idx in OPAQUE_STYLES, (inst["task_name"], style_idx)
                seen_opaque += 1
            elif in_cabinet:
                assert style_idx in TRANSPARENT_STYLES, \
                    (inst["task_name"], style_idx)
                seen_transparent += 1
            if has_box and not in_cabinet:
                assert style_idx != 2 and style_idx != 4 and style_idx != 6, \
                    (inst["task_name"], style_idx)
            # seed-stability
            cfg2, style2 = build_render_config(inst)
            assert style2 == style_idx
            assert json.dumps(cfg, sort_keys=True, default=str) == \
                json.dumps(cfg2, sort_keys=True, default=str)
    assert seen_opaque > 0 and seen_transparent > 0


def test_parquet_parity(tmp_dir="script_general/locus_pack_test"):
    import pandas as pd
    insts = SPLITS["ood_s"][:6]
    out = os.path.join(_ROBOFACTORY, tmp_dir, "ood_s")
    render_batch(insts, os.path.join(_ROBOFACTORY, tmp_dir), "ood_s",
                 dry_run=True)
    pq = pack_parquet(insts, out, require_images=False)
    df = pd.read_parquet(pq)
    ref = pd.read_parquet(os.path.join(
        _ROBOFACTORY, "..", "data", "viki", "VIKI-L2", "train.parquet"))
    assert df.columns.tolist() == ref.columns.tolist()
    row, ref_row = df.iloc[0], ref.iloc[0]
    assert row["ability"] == ref_row["ability"]
    assert row["data_source"] == "locus_bench"
    assert row["reward_model"]["style"] == ref_row["reward_model"]["style"]
    # system prompt: identical prefix, identical suffix line structure
    sp = row["prompt"][0]["content"]
    assert sp.startswith(VIKI_SYSTEM_PREFIX)
    assert "Available robot set:" in sp and \
        "Their available operation APIs:" in sp
    assert row["prompt"][1]["content"].startswith("<image>")
    # gt dict: official key set preserved, locus_meta additive
    gt, ref_gt = row["reward_model"]["ground_truth"], \
        ref_row["reward_model"]["ground_truth"]
    assert set(ref_gt.keys()) <= set(gt.keys())
    assert set(gt.keys()) - set(ref_gt.keys()) == {"locus_meta"}
    assert set(gt["robots"].keys()) == {"R1", "R2", "R3"}
    if len(gt["time_steps"]):
        assert set(gt["time_steps"][0]["actions"].keys()) == \
            {"R1", "R2", "R3"}
    assert gt["locus_meta"]["cell"] is not None


def test_decidability_check():
    for insts in SPLITS.values():
        ok, violations = decidability_check(insts)
        assert ok, violations
    # negative control: S4 hidden item relocated out of the cabinet
    s4 = next(i for i in SPLITS["ood_s"] if i["meta"]["cell"][0] == 4
              and i["meta"]["state_realization"].get("mask1") == "cabinet")
    broken = copy.deepcopy(s4)
    item = broken["meta"]["mask_map"]["mask1"]
    for name in broken["init_pos"]:
        if name.rsplit('_', 1)[0] == item:
            broken["init_pos"][name] = ["kitchen work area"]
    ok, violations = decidability_check([broken])
    assert not ok and violations


def test_real_render_artifacts_if_present():
    smoke = os.path.join(_ROBOFACTORY, "script_general",
                         "locus_render_real_smoke", "smoke")
    manifest_path = os.path.join(smoke, "render_manifest.json")
    if not os.path.exists(manifest_path):
        print("  (real-render artifacts absent; GPU smoke skipped)")
        return
    manifest = json.load(open(manifest_path))
    by_task = {e["task_id"]: e for e in manifest}
    s4 = next(e for e in by_task.values() if e["opaque_container"])
    ctrl = next(e for e in by_task.values() if not e["opaque_container"])
    assert s4["rendered"] and ctrl["rendered"]
    assert s4["style_idx"] in OPAQUE_STYLES
    assert ctrl["style_idx"] in TRANSPARENT_STYLES
    for e in manifest:
        img = os.path.join(smoke, e["image"])
        assert os.path.exists(img) and os.path.getsize(img) > 100_000, img


def _cell_sigs(splits, split):
    """(family, cell) -> [instance signature], order preserved."""
    out = {}
    for inst in splits[split]:
        meta = inst["meta"]
        key = (meta["bench_family"], tuple(meta["cell"]))
        out.setdefault(key, []).append(
            (tuple(sorted(meta["mask_map"].items())), inst["layout_id"],
             meta["instantiate_seed"]))
    return out


def test_shared_rng_couples_plan_rows():
    """Documents the LEGACY (`--shared-rng`, P7.5b) stream's coupling, which is
    why per_row_rng became the default: growing n_train reshuffles the OTHER
    splits' configs too, so a train-only ramp cannot reuse the already-rendered
    test/OOD images under that stream (docs/YOLO.md §6.2)."""
    a, _ = generate_split_instances(
        build_split_plan(n_train=2, n_test_id=2, n_ood=2), seed=0,
        validate=False, per_row_rng=False)
    b, _ = generate_split_instances(
        build_split_plan(n_train=5, n_test_id=2, n_ood=2), seed=0,
        validate=False, per_row_rng=False)
    ood_a, ood_b = _cell_sigs(a, "ood_m"), _cell_sigs(b, "ood_m")
    assert ood_a and ood_a.keys() == ood_b.keys()
    intact = sum(1 for k in ood_a if ood_b[k][:len(ood_a[k])] == ood_a[k])
    assert intact < len(ood_a), (
        "shared rng unexpectedly decoupled -- if this now holds, the "
        "per_row_rng workaround may no longer be needed")


def test_per_row_rng_is_the_default():
    """The ramp-additive stream must be what a plain call gets: every caller
    that omits the flag (locus_pilot, ad-hoc scripts) has to land on the same
    stream as the P7.5c renders, or its configs silently disagree with disk."""
    a, _ = generate_split_instances(
        build_split_plan(n_train=2, n_test_id=2, n_ood=2), seed=0,
        validate=False)
    b, _ = generate_split_instances(
        build_split_plan(n_train=2, n_test_id=2, n_ood=2), seed=0,
        validate=False, per_row_rng=True)
    assert _cell_sigs(a, "train") == _cell_sigs(b, "train")
    assert _cell_sigs(a, "ood_m") == _cell_sigs(b, "ood_m")


def test_pilot_driver_defaults_to_the_production_identity_knobs():
    """The driver that renders the benchmark must reach disk-matching configs
    with no flags. Both knobs ARE render identity — per_row_rng renumbers every
    config, and mask stratification moved 108 of the ramp63k eval configs — so
    a default flip on either silently disagrees with what is on disk."""
    import locus_pilot
    ns = locus_pilot.build_parser().parse_args([])
    assert ns.shared_rng is False, "pilot would walk the retired P7.5b stream"
    assert ns.no_stratify_masks is False, "pilot would reopen the F9 holes"


def test_scoped_rerender_keeps_the_whole_benchmark_in_the_report():
    """A --splits run re-renders PART of an existing benchmark (the 108
    stratification-changed eval configs). Two things must survive it:

      * the untouched splits stay in pilot_report.json -- otherwise a scoped
        re-render replaces a complete v1.0 record with a partial one;
      * the §7 audit still runs over the whole PLAN, since those gates
        (vocabulary coverage, family/operator support, orthogonality) are
        properties of the plan, not of whichever splits were re-rendered.
        Auditing only the re-rendered subset would report holes that do not
        exist.
    """
    import locus_pilot
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "pilot_report.json"), "w") as f:
            json.dump({"config": {"seed": 0},
                       "splits": {"train": {"n_instances": 6300,
                                            "n_rendered": 6300,
                                            "parquet": "train/data.parquet"},
                                  "test_id": {"n_instances": 180,
                                              "n_rendered": 180}}}, f)
        report = locus_pilot.main([
            "--n-train", "1", "--n-test-id", "1", "--n-ood", "1",
            "--out", tmp, "--dry-run", "--splits", "test_id"])

    plan = build_split_plan(n_train=1, n_test_id=1, n_ood=1)
    n_test_id = sum(r[-1] for r in plan if r[0] == "test_id")

    # carried forward untouched, re-rendered split replaced
    assert report["splits"]["train"]["n_rendered"] == 6300
    assert report["carried_forward_splits"] == ["train"]
    assert report["splits"]["test_id"]["n_instances"] == n_test_id
    assert n_test_id != 6300, "the re-rendered split must not be the stale entry"

    # the audit saw every split's instances, not just test_id's
    audit = report["audit_report"]
    n_planned = sum(r[-1] for r in plan)
    assert audit["n_instances"] == n_planned, (
        f"audit saw {audit['n_instances']} of {n_planned} planned instances "
        f"-> a scoped re-render would report plan-level gates as failing")
    assert audit["vocab_coverage"], "audit ran on an empty instance set"


def test_pilot_report_records_both_identity_knobs():
    """Whatever the knobs are set to, the report must say so: the same seed
    yields different configs per setting, so a report omitting them cannot be
    reproduced (docs/benchmark.md §9 P7.5c)."""
    import locus_pilot
    ns = locus_pilot.build_parser().parse_args(
        ["--shared-rng", "--no-stratify-masks"])
    assert ns.shared_rng is True and ns.no_stratify_masks is True
    src = inspect.getsource(locus_pilot.main)
    for key in ('"per_row_rng"', '"stratify_small_vocab"'):
        assert key in src, f"{key} missing from pilot_report config block"


def test_per_row_rng_makes_a_ramp_additive():
    """With per_row_rng, growing one plan row leaves every other row
    byte-identical and each cell keeps its prefix, so already-rendered
    configs stay reusable instead of being renumbered."""
    a, _ = generate_split_instances(
        build_split_plan(n_train=2, n_test_id=2, n_ood=2), seed=0,
        validate=False, per_row_rng=True)
    b, _ = generate_split_instances(
        build_split_plan(n_train=5, n_test_id=2, n_ood=2), seed=0,
        validate=False, per_row_rng=True)
    for split in ("ood_m", "ood_s", "ood_x", "test_id"):
        sa, sb = _cell_sigs(a, split), _cell_sigs(b, split)
        assert sa, split
        assert sa == sb, f"{split} changed under a train-only ramp"
    ta, tb = _cell_sigs(a, "train"), _cell_sigs(b, "train")
    for key, sigs in ta.items():
        assert tb[key][:len(sigs)] == sigs, f"train cell {key} lost its prefix"
        assert len(tb[key]) == 5


def test_per_row_rng_keeps_realization_balance():
    """Switching stream derivation must not break the §4.3 balance property."""
    splits, _ = generate_split_instances(
        build_split_plan(n_train=4, n_test_id=0, n_ood=0), seed=0,
        validate=False, per_row_rng=True)
    report = realization_balance_report(splits["train"])
    bad = {k: v for k, v in report.items() if not v["ok"]}
    assert not bad, f"per_row_rng broke §4.3 balance: {bad}"


def _seed_render_dir(root, split, configs):
    """Lay out a rendered split: {idx: config_text} -> config/image/label."""
    cfg_dir = os.path.join(root, split, "configs")
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "gt_labels")
    for d in (cfg_dir, img_dir, lbl_dir):
        os.makedirs(d, exist_ok=True)
    for idx, text in configs.items():
        with open(os.path.join(cfg_dir, f"{split}_{idx:05d}.yaml"), "w") as f:
            f.write(text)
        with open(os.path.join(img_dir, f"{idx:05d}.png"), "w") as f:
            f.write(f"PNG:{text}")
        with open(os.path.join(lbl_dir, f"{idx:05d}.json"), "w") as f:
            f.write(f"LBL:{text}")
    return cfg_dir, img_dir, lbl_dir


def _rendered(img_dir):
    out = {}
    for fn in os.listdir(img_dir):
        if fn.endswith(".png"):
            with open(os.path.join(img_dir, fn)) as f:
                out[int(fn[:-4])] = f.read()
    return out


def test_render_batch_absolutises_out_dir():
    """A relative --out silently mis-targets every path handed to the mask
    worker, which runs with cwd=RoboFactory. It cost a 6.3k ramp launch: the
    config pre-pass succeeded, then all 42 chunks died on FileNotFoundError for
    the jobs JSON ~17 min in. Fail-proof it at the one place that spawns with a
    foreign cwd, so no caller has to remember."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        cwd = os.getcwd()
        os.chdir(root)
        try:
            insts = copy.deepcopy(SPLITS["ood_s"][:2])
            manifest = render_batch(insts, "rel_out", "ood_s", dry_run=True)
            cfg = os.path.join(root, "rel_out", "ood_s", "configs")
            assert os.path.isdir(cfg), "configs must land under the given path"
            assert len(manifest) == 2
            # Nothing may have been created under RoboFactory/rel_out.
            assert not os.path.exists(os.path.join(_ROBOFACTORY, "rel_out"))
        finally:
            os.chdir(cwd)


def test_relocate_reuses_renders_whose_index_shifted():
    """A ramp shifts flat sample_index even for byte-identical configs, so
    renders must be matched by config content, not by position -- otherwise
    the 6.3k -> 7.2k bump (docs/YOLO.md §6.2) re-renders everything."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        split = "train"
        cfg_dir, img_dir, _ = _seed_render_dir(
            root, split, {0: "cfgA", 1: "cfgB", 2: "cfgC"})
        # Chain: a new sample takes index 1, pushing B and C up by one. Every
        # destination is occupied by a render that is itself a source.
        planned = [(0, "cfgA"), (1, "cfgNEW"), (2, "cfgB"), (3, "cfgC")]
        n = _relocate_reusable_renders(
            os.path.join(root, split), cfg_dir, split, planned)
        assert n == 2, n
        assert _rendered(img_dir) == {
            0: "PNG:cfgA", 2: "PNG:cfgB", 3: "PNG:cfgC"}, _rendered(img_dir)
        assert not os.path.exists(os.path.join(img_dir, "00001.png")), (
            "the new sample's slot must be empty so it actually renders")
        for idx, want in ((0, "cfgA"), (2, "cfgB"), (3, "cfgC")):
            with open(os.path.join(root, split, "gt_labels",
                                   f"{idx:05d}.json")) as f:
                assert f.read() == f"LBL:{want}", idx


def test_relocate_handles_a_pure_permutation():
    """Sources and destinations can form a cycle (swap), which a naive
    one-phase move would clobber."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        split = "ood_m"
        cfg_dir, img_dir, _ = _seed_render_dir(
            root, split, {0: "cfgA", 1: "cfgB"})
        n = _relocate_reusable_renders(
            os.path.join(root, split), cfg_dir, split,
            [(0, "cfgB"), (1, "cfgA")])
        assert n == 2, n
        assert _rendered(img_dir) == {0: "PNG:cfgB", 1: "PNG:cfgA"}
        assert not [f for f in os.listdir(img_dir)
                    if f.endswith(".relocating")], "temp files left behind"


def test_relocate_is_a_noop_when_nothing_moved():
    """Steady state must not touch disk: reruns of an unchanged plan are the
    common case and must stay free."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        split = "test_id"
        cfg_dir, img_dir, _ = _seed_render_dir(
            root, split, {0: "cfgA", 1: "cfgB"})
        before = _rendered(img_dir)
        n = _relocate_reusable_renders(
            os.path.join(root, split), cfg_dir, split,
            [(0, "cfgA"), (1, "cfgB"), (2, "cfgNEW")])
        assert n == 0, n
        assert _rendered(img_dir) == before


def test_relocate_skips_configs_with_no_render_on_disk():
    """A config present but never rendered (crashed job) must not be
    'relocated' into someone else's slot and counted as reusable."""
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        split = "train"
        cfg_dir, img_dir, _ = _seed_render_dir(root, split, {0: "cfgA"})
        os.remove(os.path.join(img_dir, "00000.png"))
        n = _relocate_reusable_renders(
            os.path.join(root, split), cfg_dir, split, [(5, "cfgA")])
        assert n == 0, n
        assert _rendered(img_dir) == {}


def _obj(name, p, rp=None):
    pos = {"ppos": {"p": list(p)}}
    if rp is not None:
        pos["randp_scale"] = list(rp)
    return {"name": name, "pos": pos}


def _boxes(cfgs):
    return [_xy_reach_box(c["pos"]) for c in cfgs]


def test_reach_box_is_one_sided():
    """`ppos + randp_scale * rand(3)` with rand() in [0,1): the reach is the
    interval between p and p+rp, NOT p±rp. Getting this wrong doubles every
    box and would make the case (c) merge fire on scenes that are fine."""
    assert _xy_reach_box(_obj("a", [1.0, 2.0, 0], [0.5, 0.25, 0])["pos"]) == \
        [(1.0, 1.5), (2.0, 2.25)]
    assert _xy_reach_box(_obj("a", [1.0, 2.0, 0], [-0.5, 0, 0])["pos"]) == \
        [(0.5, 1.0), (2.0, 2.0)]
    assert _xy_reach_box(_obj("a", [1.0, 2.0, 0])["pos"]) == \
        [(1.0, 1.0), (2.0, 2.0)]


def test_chebyshev_separation_is_per_axis():
    """`np.max(delta_pos[:2]) < 0.15` — one axis ≥0.15 apart makes the pair
    safe no matter how far the other axis overlaps. This is what lets the
    re-slot zero jitter on ONE axis and keep the other."""
    far_x = _xy_reach_box(_obj("a", [0, 0, 0], [0.1, 5.0, 0])["pos"])
    near = _xy_reach_box(_obj("b", [0.30, 0, 0], [0.1, 5.0, 0])["pos"])
    assert _reach_boxes_separable(far_x, near)          # 0.30-0.10 = 0.20
    closer = _xy_reach_box(_obj("b", [0.20, 0, 0], [0.1, 5.0, 0])["pos"])
    assert not _reach_boxes_separable(far_x, closer)    # 0.20-0.10 = 0.10


def test_placement_model_matches_an_analytic_geometry():
    """A 2m sweep followed by a 10cm item at x=1.0: the item is fully
    blocked exactly when the sweep lands in (0.95, 1.15), so the official
    loop must exit(0) with probability 0.20/2.0 = 0.10."""
    boxes = _boxes([_obj("sweep", [0, 0, 0], [2.0, 0, 0]),
                    _obj("tight", [1.0, 0, 0], [0.10, 0, 0])])
    p = placement_failure_probability(boxes, seed=7, trials=400)
    assert 0.06 < p < 0.14, p


def test_cross_anchor_conflict_is_resolved():
    """Case (c): two DIFFERENT anchors, so anchor slotting is blind to them
    and only the reach-box merge can see the conflict."""
    cfgs = [_obj("sweep", [0, 0, 0], [2.0, 0, 0]),
            _obj("tight", [1.0, 0, 0], [0.10, 0, 0])]
    assert resolve_cross_anchor_conflicts(cfgs) == 1
    p = placement_failure_probability(_boxes(cfgs), seed=7, trials=400)
    assert p < 1e-4, p


def test_cross_anchor_reslot_keeps_what_jitter_it_can():
    """Neither item may collapse to a point. The sweep is capped 0.16m short
    of the slot above it (1.0 - 0.16 = 0.84); the tight item has nothing
    above it at all, so it keeps its declared 0.10 in full."""
    cfgs = [_obj("sweep", [0, 0, 0], [2.0, 0, 0]),
            _obj("tight", [1.0, 0, 0], [0.10, 0, 0])]
    resolve_cross_anchor_conflicts(cfgs)
    assert cfgs[0]["pos"]["ppos"]["p"][0] == 0
    assert abs(cfgs[0]["pos"]["randp_scale"][0] - 0.84) < 1e-9, cfgs[0]
    assert cfgs[1]["pos"]["ppos"]["p"][0] == 1.0
    assert abs(cfgs[1]["pos"]["randp_scale"][0] - 0.10) < 1e-9, cfgs[1]


def test_coverage_probability_is_exact():
    """The no-variance half of the gate. A 1.6m sweep whose 0.30 exclusion
    square can swallow a 20x4cm item blocks it with certainty — but only for
    the fraction of draws that land on the cover, which is what makes it
    invisible to a 48-trial estimator (pilot ood_s/00120)."""
    tiny = _xy_reach_box(_obj("iron", [0.0, 0.0, 0], [0.20, 0.04, 0])["pos"])
    sweep = _xy_reach_box(_obj("rock", [0.10, -0.80, 0], [0.0, 1.60, 0])["pos"])
    # To cover ALL of i, j must sit in (hi_i - 0.15, lo_i + 0.15).
    # x: that window is (0.05, 0.15); the sweep is fixed at 0.10, inside -> 1
    # y: the window is (0.04-0.15, 0.0+0.15) = (-0.11, 0.15), and the sweep
    #    spans [-0.80, 0.80], so 0.26/1.60 of its draws land there.
    assert abs(_coverage_probability(tiny, sweep) - 0.26 / 1.60) < 1e-9
    # a fixed blocker just outside the window covers nothing (x=0.0 < 0.05)
    off = _xy_reach_box(_obj("rock", [0.0, -0.80, 0], [0.0, 1.60, 0])["pos"])
    assert _coverage_probability(tiny, off) == 0.0
    # an item wider than 2*sep can never be covered by a single blocker
    wide = _xy_reach_box(_obj("board", [0, 0, 0], [0.40, 0.04, 0])["pos"])
    assert _coverage_probability(wide, sweep) == 0.0


def test_cross_anchor_gate_leaves_roomy_scenes_untouched():
    """Overlap is common (~10% of pilot configs), real infeasibility is not
    (11/1160). The gate matters beyond saved work: group membership is
    realization-dependent, so touching a scene the official loop already
    handles risks perturbing a non-target asset and breaking the §7.10
    counterfactual-pair invariance."""
    cfgs = [_obj("sweep", [0, 0, 0], [3.0, 0, 0]),
            _obj("wide", [1.0, 0, 0], [1.5, 0, 0])]
    assert not _reach_boxes_separable(*_boxes(cfgs))    # they DO overlap
    before = copy.deepcopy(cfgs)
    assert resolve_cross_anchor_conflicts(cfgs) == 0
    assert cfgs == before


def test_cross_anchor_reslot_is_idempotent():
    """Re-slotting can shift a member outside its original box, so the merge
    re-runs to a fixed point — a second call must then be a no-op."""
    cfgs = [_obj("sweep", [0, 0, 0], [2.0, 0, 0]),
            _obj("tight", [1.0, 0, 0], [0.10, 0, 0]),
            _obj("tight2", [1.05, 0, 0], [0.10, 0, 0])]
    resolve_cross_anchor_conflicts(cfgs)
    settled = copy.deepcopy(cfgs)
    assert resolve_cross_anchor_conflicts(cfgs) == 0
    assert cfgs == settled


def test_cross_anchor_never_moves_agents():
    """Agent poses are task semantics (and are drawn against the 0.6m agent
    criterion, not this one): they are obstacles, never movable."""
    agents = [_obj("R0", [1.0, 0, 0], [0.05, 0, 0])]
    before = copy.deepcopy(agents)
    cfgs = [_obj("sweep", [0, 0, 0], [2.0, 0, 0]),
            _obj("tight", [1.0, 0, 0], [0.10, 0, 0])]
    resolve_cross_anchor_conflicts(cfgs, agents)
    assert agents == before


def test_pilot_configs_are_placement_feasible():
    """Regression on the real thing: every P7.5b pilot config, rebuilt through
    the current placement code, must leave the official loop safe.

    Two bars, neither of them ``_is_infeasible``. The gate is deliberately
    more conservative than this audit — it thresholds a MAX over sampled
    realizations, so on a scene near the bar it fires or not depending on the
    rng stream, and an audit demanding its verdict reproduce under a fresh
    seed would be demanding that noise reproduce. (It is still deterministic
    per config, seeded from the scene itself, so renders stay reproducible.)
    Nor can a tight per-config bar be asserted: failure needs a rare EARLIER
    realization, so a config whose true rate is ~1e-6 reads anywhere in
    1e-12..1e-3 depending on whether 200 trials happened to hit it
    (ood_s/00121, a same-anchor trio with 2.2m of sweep that case (b) rightly
    passed — force-slotting it would zero the risk and delete two items'
    y-jitter to buy 1e-9).

    So: (1) no single config may be materially risky — the bar is 1e-2, two
    orders under the pre-fix worst (ood_x/00032 at 0.28) and an order above
    the post-fix residual, wide enough that the estimator's spread at these
    rates cannot cross it; (2) the SUM of per-config rates, i.e. the expected
    number of ``exit(0)`` events over the whole set, must stay negligible —
    an average over 1160 configs, and therefore the stable statistic. Pre-fix
    that sum was >1.0 (23 configs above 1e-4, six of them above 0.04, and
    ood_s/00114 at 0.11 actually did fail during the P7.5b re-render);
    post-fix it is ~1e-3. The variance-free coverage term must be exactly 0.
    """
    import glob
    import yaml
    root = os.path.join(os.path.dirname(_ROBOFACTORY),
                        "data/lnsdl_bench/pilot")
    files = sorted(glob.glob(os.path.join(root, "*/configs/*.yaml")))
    if not files:
        print("  (no pilot render dir; skipped)")
        return
    worst, covered, total = (0.0, None), [], 0.0
    for path in files:
        with open(path) as fh:
            cfg = yaml.load(fh.read(), Loader=yaml.FullLoader)
        agents = [a for a in cfg.get("agents", [])
                  if isinstance(a.get("pos"), dict)]
        objects = cfg.get("objects", [])
        resolve_cross_anchor_conflicts(objects, agents)
        boxes = _boxes(agents) + _boxes(objects)
        looping = [False] * len(agents) + [True] * len(objects)
        # seed off the path, not hash(): PYTHONHASHSEED makes hash() differ
        # per process, and a flaky feasibility gate is worse than none.
        seed = int.from_bytes(
            hashlib.sha256(path.encode()).digest()[:8], "big")
        for k in range(len(agents), len(boxes)):
            (x_lo, x_hi), (y_lo, y_hi) = boxes[k]
            if x_hi - x_lo < 1e-9 and y_hi - y_lo < 1e-9:
                continue                # no randp_scale: never enters the loop
            if any(_coverage_probability(boxes[k], boxes[j]) > 1e-4
                   for j in range(k)):
                covered.append((os.path.basename(path), k))
        p = max(placement_failure_probability(
            boxes, seed=seed + s, looping=looping, trials=200)
            for s in range(3))
        total += p
        if p > worst[0]:
            worst = (p, os.path.basename(path))
    assert not covered, covered[:8]
    assert worst[0] < 1e-2, worst
    assert total < 0.05, (total, worst)


# ---------------------------------------------------------------------------
# case (d): rest the body on its anchor plane
# ---------------------------------------------------------------------------

def _write_obj(path, verts):
    with open(path, "w") as f:
        f.write("# unit test mesh\n")
        for v in verts:
            f.write("v %f %f %f\n" % tuple(v))
        f.write("vn 0.0 0.0 1.0\nf 1 2 3\n")


def test_mesh_aabb_reads_obj_and_glb_without_a_mesh_library():
    import struct
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        obj = os.path.join(d, "m.obj")
        # "vn"/"vt" lines must not be mistaken for vertices, and a non-UTF8
        # byte anywhere in the file must not kill the read (real assets have
        # them in comments and material names).
        with open(obj, "wb") as f:
            f.write(b"# \xa8 latin-1 comment\nv -1 -2 0\nv 3 4 5\n"
                    b"vn 0 0 -9\nvt 0 0\n")
        assert mesh_aabb(obj) == ([-1.0, -2.0, 0.0], [3.0, 4.0, 5.0])

        doc = {"nodes": [{"children": [1]},
                         {"mesh": 0, "translation": [0.0, 0.0, 10.0]}],
               "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
               "accessors": [{"min": [-1.0, -1.0, -2.0], "max": [1.0, 1.0, 2.0]}]}
        blob = json.dumps(doc).encode()
        blob += b" " * (-len(blob) % 4)
        glb = os.path.join(d, "m.glb")
        with open(glb, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(blob)))
            f.write(struct.pack("<II", len(blob), 0x4E4F534A) + blob)
        # the node transform must be applied: min z is -2 in mesh space, +8 here
        assert mesh_aabb(glb) == ([-1.0, -1.0, 8.0], [1.0, 1.0, 12.0])


def test_rest_offset_lifts_exactly_what_hangs_below_the_plane():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # the .obj convention all 22 LOCUS meshes follow: xy centred, z in [0,1]
        _write_obj(os.path.join(d, "m.obj"),
                   [(-0.5, -0.5, 0.0), (0.5, 0.5, 1.0), (0.0, 0.0, 0.5)])
        item = {"visual": {"filename": "m.obj", "scale": [0.4, 0.4, 0.4]},
                "pos": {"ppos": {"p": [1.0, 2.0, 0.0]}, "qpos": [1, 0, 0, 0]}}
        # upright: already resting, nothing to do
        assert rest_offset_z(item, 0.0, assets_root=d) == 0.0
        # 90° about x (the layouts' most common qpos, [0.707, 0.707, 0, 0]):
        # the body now spans z in [-0.2, 0.2] scaled → lift by 0.2
        item["pos"]["qpos"] = [0.70710678, 0.70710678, 0, 0]
        assert abs(rest_offset_z(item, 0.0, assets_root=d) - 0.2) < 1e-6
        # a layout that already lifts by MORE than it sinks needs no help, and
        # the correction is one-sided: a hovering object is left hovering.
        assert rest_offset_z(item, 0.25, assets_root=d) == 0.0
        # partial hand-tuned lift → only the remainder
        assert abs(rest_offset_z(item, 0.05, assets_root=d) - 0.15) < 1e-6
        # no mesh to measure (primitive) / missing file → no opinion
        assert rest_offset_z({"pos": {"qpos": [1, 0, 0, 0]}}, 0.0) == 0.0
        assert rest_offset_z(
            {"visual": {"filename": "nope.obj"},
             "pos": {"qpos": [1, 0, 0, 0]}}, 0.0, assets_root=d) == 0.0


def test_every_layout_object_now_rests_on_its_plane():
    """The measured defect, closed on the real assets: 19 of 29 object classes
    sank below their anchor plane after the layouts' own hand-tuned lifts
    (indoor_plant 20.0cm, bottle 16.0, clothes_iron 14.4, bread 8.7, cup 6.2),
    which is the mechanism behind both the visible 穿模 and the S4 leak."""
    import glob
    import yaml
    sank = {}
    for path in sorted(glob.glob(os.path.join(_ROBOFACTORY,
                                              "configs/layouts/layout_*.yaml"))):
        with open(path) as f:
            layout = yaml.safe_load(f)
        for obj in layout.get("objects") or []:
            if not (obj.get("visual") or {}).get("filename"):
                continue                       # primitive: nothing to measure
            lift = float(obj["pos"]["ppos"]["p"][2])
            before = -rotated_min_z(
                os.path.join(_ROBOFACTORY, obj["visual"]["filename"]),
                obj["visual"].get("scale", 1.0), obj["pos"]["qpos"]) - lift
            after = before - rest_offset_z(obj, lift)
            assert after <= 1e-9, (path, obj["name"], after)
            if before > 5e-3:
                sank[obj["name"].rsplit("_", 1)[0]] = round(before, 4)
    assert len(sank) >= 19, sank
    assert sank.get("indoor_plant", 0) > 0.19 and sank.get("bread", 0) > 0.08


def test_rest_offset_only_moves_z_and_only_downward_offenders():
    """Guard the invariant the case (b)/(c) gates depend on: the pass must not
    touch x/y (their reach-box separability is computed in XY) and must never
    lower an object, so a scene the feasibility gates cleared stays cleared."""
    inst = SPLITS["train"][0]
    cfg, _ = build_render_config(inst)
    import locus_render_pack as pack
    orig = pack.rest_offset_z
    try:
        pack.rest_offset_z = lambda *a, **k: 0.0
        plain, _ = build_render_config(inst)
    finally:
        pack.rest_offset_z = orig
    by_name = {o["name"]: o for o in plain["objects"]}
    assert set(by_name) == {o["name"] for o in cfg["objects"]}
    lifted = 0
    for obj in cfg["objects"]:
        ref = by_name[obj["name"]]
        assert obj["pos"]["ppos"]["p"][:2] == ref["pos"]["ppos"]["p"][:2]
        assert obj["pos"].get("randp_scale") == ref["pos"].get("randp_scale")
        delta = obj["pos"]["ppos"]["p"][2] - ref["pos"]["ppos"]["p"][2]
        assert delta >= -1e-12, (obj["name"], delta)
        lifted += delta > 1e-9
    assert lifted, "no object was lifted: the pass is not wired in"


# ---------------------------------------------------------------------------
# case (e): keep a cabinet-anchored body inside the carcass
# ---------------------------------------------------------------------------

def _cabinet_item(d, name="loaf", span=(0.0, 0.9), qpos=(1, 0, 0, 0),
                  p=(1.0, 2.0, 1.5), rp=(0.1, 0.0, 0.0)):
    """One config item whose mesh is a bar of length ``span`` along local x.

    The origin sits at one END of the bar when ``span`` starts at 0, which is
    the real bread/banana case: the layouts' quaternions routinely park the
    origin off-centre, so an inset that assumed symmetry would be wrong.
    """
    _write_obj(os.path.join(d, "bar.obj"),
               [(span[0], -0.05, 0.0), (span[1], 0.05, 0.1), (0.0, 0.0, 0.05)])
    return {"name": name,
            "visual": {"filename": "bar.obj", "scale": 1.0},
            "pos": {"ppos": {"p": list(p)}, "randp_scale": list(rp),
                    "qpos": list(qpos)}}


def test_rotated_xy_box_is_signed_and_survives_yaw():
    """The extent has to be measured per SIDE, not as a half-width: the origin
    is not the XY centre once the layout's quaternion is applied."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        item = _cabinet_item(d)
        path = os.path.join(d, "bar.obj")
        (xlo, xhi), (ylo, yhi) = rotated_xy_box(path, 1.0, [1, 0, 0, 0])
        assert (abs(xlo) < 1e-9 and abs(xhi - 0.9) < 1e-9), (xlo, xhi)
        assert abs(ylo + 0.05) < 1e-9 and abs(yhi - 0.05) < 1e-9
        # +90° about z sends local +x onto world +y: the long axis swaps.
        (xlo, xhi), (ylo, yhi) = rotated_xy_box(
            path, 1.0, [0.70710678118654752, 0, 0, 0.70710678118654752])
        assert abs(xhi - xlo - 0.1) < 1e-6, (xlo, xhi)
        assert abs(yhi - ylo - 0.9) < 1e-6, (ylo, yhi)
        # scale is applied before rotation, per axis
        (xlo, xhi), _ = rotated_xy_box(path, [2.0, 1.0, 1.0], [1, 0, 0, 0])
        assert abs(xhi - 1.8) < 1e-9
        assert item["pos"]["qpos"] == [1, 0, 0, 0]     # read-only on the item


def test_clamp_interval_shrinks_translates_and_collapses():
    # overlap → shrink to the intersection, keeping whatever jitter survives
    assert _clamp_interval(0.0, 1.0, 0.5, 2.0) == (0.5, 1.0, True)
    # already inside → untouched
    assert _clamp_interval(0.2, 0.3, 0.0, 1.0) == (0.2, 0.3, True)
    # disjoint → translate to the nearest feasible point, span collapses
    assert _clamp_interval(0.0, 0.1, 0.5, 2.0) == (0.5, 0.5, True)
    assert _clamp_interval(3.0, 4.0, 0.5, 2.0) == (2.0, 2.0, True)
    # feasible set empty (body wider than the cabinet) → centre, report False
    lo, hi, ok = _clamp_interval(0.0, 1.0, 0.6, 0.4)
    assert (lo, hi) == (0.5, 0.5) and ok is False


def test_fit_in_cabinet_insets_within_the_carcass():
    """The whole reach box must land inside the interior, not just its centre:
    the official loop draws a uniform point from the box, so containment has to
    hold for EVERY draw or the leak is merely made rarer."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        interior = ((1.0, 2.0), (1.9, 2.1))
        # a box already inside the feasible set must come back untouched --
        # every yaw contains it, so the tie has to break towards identity.
        item = _cabinet_item(d, p=(1.02, 2.0, 1.5), rp=(0.05, 0.0, 0.0))
        rep = fit_in_cabinet(item, interior, assets_root=d)
        assert rep["contained"] and rep["yaw"] == 0 and rep["shift"] == 0.0
        assert item["pos"]["ppos"]["p"][0] == 1.02
        assert abs(item["pos"]["randp_scale"][0] - 0.05) < 1e-12

        # a box straddling the far wall is SHRUNK to the part that fits, which
        # keeps some jitter rather than collapsing to a point
        item = _cabinet_item(d, p=(1.05, 2.0, 1.5), rp=(0.3, 0.0, 0.0))
        rep = fit_in_cabinet(item, interior, assets_root=d)
        assert rep["contained"] and rep["yaw"] == 0, rep
        p, rp = item["pos"]["ppos"]["p"], item["pos"]["randp_scale"]
        # bar runs +0.9 from the origin → the origin may not exceed 2.0-0.9
        assert p[0] >= 1.0 - 1e-9 and p[0] + rp[0] <= 1.1 + 1e-9, (p, rp)
        assert rp[0] > 0.0, "shrank a still-feasible span to a point"
        assert cabinet_containment_report(
            item, interior, assets_root=d)["overhang"] <= 1e-9
        # z and the untouched axis are this pass's business to leave alone
        assert p[2] == 1.5 and rp[1] == 0.0


def test_fit_in_cabinet_prefers_the_smaller_disturbance():
    """A yaw and a translation are both legal repairs, so the choice is a
    policy: keep the object where the layout author put it, and reorient
    instead. Measured cost of the other order on the real layouts: the L1/L2
    bread would be dragged ~0.5m along the run to avoid a 180° turn."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        interior = ((1.0, 2.0), (1.9, 2.1))
        # origin-at-one-end bar: identity needs a 0.5m drag, 180° needs none
        item = _cabinet_item(d, p=(1.6, 2.0, 1.5), rp=(0.3, 0.0, 0.0))
        rep = fit_in_cabinet(item, interior, assets_root=d)
        assert rep["contained"] and rep["yaw"] == 180, rep
        assert rep["shift"] == 0.0, rep
        assert cabinet_containment_report(
            item, interior, assets_root=d)["overhang"] <= 1e-9


def test_fit_in_cabinet_yaws_a_body_that_cannot_fit_as_authored():
    """The measured L4/L6 bread: a 0.549m loaf in a 0.34m-deep cabinet. No
    translation can contain it, so the pass must be allowed to yaw."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        interior = ((1.0, 1.3), (1.0, 2.0))        # deep in y, shallow in x
        item = _cabinet_item(d, p=(1.1, 1.4, 1.5), rp=(0.1, 0.1, 0.0))
        rep = fit_in_cabinet(item, interior, assets_root=d)
        assert rep["contained"], rep
        assert rep["yaw"] in (90, 270), rep       # long axis rotated onto y
        assert cabinet_containment_report(
            item, interior, assets_root=d)["overhang"] <= 1e-9
        # a body that fits on NO axis is centred and reported, never claimed
        tight = ((1.0, 1.3), (1.0, 1.3))
        item2 = _cabinet_item(d, p=(1.1, 1.1, 1.5), rp=(0.1, 0.1, 0.0))
        rep2 = fit_in_cabinet(item2, tight, assets_root=d)
        assert rep2["contained"] is False, rep2
        # primitives and missing assets have no measurable extent → no opinion
        bare = {"name": "cube", "pos": {"ppos": {"p": [1.0, 1.0, 1.0]},
                                        "qpos": [1, 0, 0, 0]}}
        assert fit_in_cabinet(bare, interior)["contained"] is True
        assert cabinet_containment_report(bare, interior) is None


def test_cabinet_interior_table_matches_the_layouts_it_claims():
    """`_CABINET_INTERIOR` is measured by `locus_cabinet_probe.py`, so guard the
    two properties the fit pass relies on: one entry per sampled layout, and
    each entry actually overlaps the layout's own declared `cabinet` band (a
    table keyed to the wrong fixture would silently move objects out of the
    cabinet they were meant to be in)."""
    import yaml
    assert sorted(_CABINET_INTERIOR) == sorted(ALL_LAYOUTS)
    for layout_id, interior in sorted(_CABINET_INTERIOR.items()):
        with open(os.path.join(_ROBOFACTORY,
                               f"configs/layouts/layout_{layout_id}.yaml")) as f:
            layout = yaml.safe_load(f)
        pos = next(a["pos"] for a in layout["position_areas"]
                   if any("cabinet" in str(n).lower() for n in a["names"]))
        p, rp = pos["ppos"]["p"], pos["randp_scale"]
        for ax in (0, 1):
            lo, hi = min(p[ax], p[ax] + rp[ax]), max(p[ax], p[ax] + rp[ax])
            assert not (hi < interior[ax][0] or lo > interior[ax][1]), \
                (layout_id, ax, (lo, hi), interior[ax])
            assert interior[ax][1] > interior[ax][0], (layout_id, ax)


def test_reslot_keeps_a_bounded_member_inside_its_carcass():
    """Cases (b)/(c) separate; case (e) contains. Where they disagree, the
    scheduler has to find a slot that does both, and the pilot showed it
    could not do that from the inset box alone (ood_x/00023: two bodies
    0.13m apart at the shelf's far end, 0.7m of shelf empty behind them).

    Three bodies on one 0.9m shelf, all pinned at the SAME point with no
    jitter — separable only by using the whole carcass.
    """
    import tempfile
    import locus_render_pack as pack
    with tempfile.TemporaryDirectory() as d:
        # short bar (0.1m) so three of them fit at 0.16m spacing
        members = [_cabinet_item(d, name=f"b{i}", span=(0.0, 0.1),
                                 p=(1.75, 2.0, 1.5), rp=(0.02, 0.0, 0.0))
                   for i in range(3)]
        bound = {id(m): [(1.0, 1.8), (1.98, 2.02)] for m in members}
        pack._reslot_merged_group(members, bounds=bound)
        xs = sorted(m["pos"]["ppos"]["p"][0] for m in members)
        assert xs[0] >= 1.0 - 1e-9 and xs[-1] <= 1.8 + 1e-9, xs
        for a, b in zip(xs, xs[1:]):
            assert b - a >= 0.16 - 1e-9, xs
        for m in members:                      # never widened past its reach
            assert (m["pos"].get("randp_scale") or [0.0])[0] <= 0.02 + 1e-9
        # and the axis is chosen by where the carcass has room, not by the
        # group's own extent: a deep-in-y bound must not slot along x.
        members = [_cabinet_item(d, name=f"c{i}", span=(0.0, 0.1),
                                 p=(1.2, 2.0, 1.5), rp=(0.3, 0.0, 0.0))
                   for i in range(2)]
        bound = {id(m): [(1.15, 1.25), (2.0, 2.9)] for m in members}
        pack._reslot_merged_group(members, bounds=bound)
        ys = sorted(m["pos"]["ppos"]["p"][1] for m in members)
        assert ys[1] - ys[0] >= 0.16 - 1e-9, ys
        assert all(1.15 - 1e-9 <= m["pos"]["ppos"]["p"][0] <= 1.25 + 1e-9
                   for m in members)
        # unbounded groups keep the old behaviour exactly
        free = [_cabinet_item(d, name=f"f{i}", span=(0.0, 0.1),
                              p=(1.0, 2.0, 1.5), rp=(0.05, 0.0, 0.0))
                for i in range(2)]
        pack._reslot_merged_group(free)
        assert [m["pos"]["ppos"]["p"][0] for m in free] == [1.0, 1.16]


def test_plate_is_the_only_body_no_cabinet_can_hold():
    """The residual, stated as the measurement that produces it.

    Ten movable classes can be anchored in a cabinet (`init_pos` lists them,
    always as a SINGLE-element position list, so there is nowhere to re-draw
    them to). Nine fit; `plate` does not, in any layout, under any yaw about z:
    it is authored lying flat (`qpos` = a quarter turn about x) at 0.400x0.400m
    against a 0.34m-deep carcass. This is an asset-vs-fixture fact, not a bug in
    the fit pass, so it is pinned here rather than asserted away -- and it is
    what `test_every_cabinet_anchored_object_...` is allowed to see uncontained.
    """
    import yaml
    import locus_render_pack as pack
    classes = ["bread", "apple", "banana", "plate", "pumpkin", "tomato",
               "scissors", "fork", "spoon", "peach"]
    uncontainable = set()
    for layout_id in ALL_LAYOUTS:
        (xl, xh), (yl, yh) = _CABINET_INTERIOR[layout_id]
        span = (xh - xl, yh - yl)
        assert abs(min(span) - 0.34) < 1e-9, (layout_id, span)  # one door depth
        with open(os.path.join(_ROBOFACTORY,
                               f"configs/layouts/layout_{layout_id}.yaml")) as f:
            layout = yaml.safe_load(f)
        objs = {o["name"].rsplit("_", 1)[0].replace(" ", "_").replace("-", "_"): o
                for o in layout["objects"]}
        for cls in classes:
            obj = objs.get(cls)
            visual = (obj or {}).get("visual") or {}
            if not visual.get("filename"):
                continue
            path = os.path.join(_ROBOFACTORY, visual["filename"])
            q = obj["pos"].get("qpos", [1, 0, 0, 0])
            fits = False
            for k in pack._CABINET_YAWS:
                qq = pack._quat_mul(pack._YAW_TABLE[k], q) if k else list(q)
                (x0, x1), (y0, y1) = rotated_xy_box(
                    path, visual.get("scale", [1, 1, 1]), qq)
                fits = fits or (x1 - x0 <= span[0] + 1e-9
                                and y1 - y0 <= span[1] + 1e-9)
            if not fits:
                uncontainable.add(cls)
    assert uncontainable == {"plate"}, uncontainable


def test_every_containable_cabinet_object_ends_up_inside_the_cabinet():
    """The measured defect, closed on the real assets and the real layouts.

    Also pins the two orderings that make the pass sound: it runs BEFORE cases
    (b)/(c) (which own separation and may only see already-inset boxes), and
    the post-(b)/(c) audit is what proves those two did not push a member back
    out through the door.
    """
    import locus_render_pack as pack
    plan = build_split_plan(n_train=40, n_test_id=12, n_ood=28)
    insts, _ = generate_split_instances(plan, seed=7)
    seen = leaked_before = 0
    for rows in insts.values():
        for inst in rows:
            cfg, _ = build_render_config(inst)
            fit = cfg.get("locus_cabinet_fit")
            if not fit:
                continue
            seen += 1
            # `plate` is the one body no carcass can hold (see the test above);
            # everything else must come back contained, and must still be
            # contained after (b)/(c) have moved things around.
            residual = {n for n in fit["uncontained"]
                        if n.rsplit("_", 1)[0] == "plate"}
            assert set(fit["uncontained"]) == residual, (inst["meta"]["cell"], fit)
            assert set(fit["escaped"]) <= residual, (inst["meta"]["cell"], fit)
            # same instance with the fit pass stubbed out: how bad was it?
            orig = pack.fit_in_cabinet
            try:
                pack.fit_in_cabinet = lambda it, *a, **k: {
                    "name": it.get("name"), "yaw": 0, "contained": True,
                    "shift": 0.0}
                plain, _ = build_render_config(inst)
            finally:
                pack.fit_in_cabinet = orig
            interior = pack._CABINET_INTERIOR[int(inst["layout_id"])]
            leaked_before += any(
                (pack.cabinet_containment_report(o, interior) or
                 {"overhang": 0.0})["overhang"] > 5e-3
                for o in plain["objects"]
                if o["name"] in fit["items"] and o["name"] not in residual)
    assert seen >= 5, f"only {seen} cabinet-anchored scenes sampled"
    assert leaked_before, "no scene leaked before the fix: nothing was proven"


# ---------------------------------------------------------------------------
# case (f): keep a body from poking through the room's walls
# ---------------------------------------------------------------------------

def _wall_item(d, name="blade", span=(0.0, 0.5), p=(2.0, -0.2, 0.95),
               rp=(0.0, 0.3, 0.0), qpos=(1, 0, 0, 0)):
    """A bar of length ``span`` along local x with its origin at one END.

    That asymmetry is the whole defect: scissors' quaternion maps its 48cm
    axis onto world +y and parks the origin at the pivot, so the body is
    almost entirely on one side of the point the placement loop bounds.
    """
    _write_obj(os.path.join(d, "bar.obj"),
               [(span[0], -0.02, 0.0), (span[1], 0.02, 0.05),
                (0.0, 0.0, 0.02)])
    return {"name": name,
            "visual": {"filename": "bar.obj", "scale": 1.0},
            "pos": {"ppos": {"p": list(p)}, "randp_scale": list(rp),
                    "qpos": list(qpos)}}


def test_clear_of_walls_trims_only_the_infeasible_end_of_the_reach_box():
    """The band must lose exactly the part that put the body through the wall
    and keep the rest -- a pass that recentred or collapsed the box would move
    placements that were already legal."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # bar runs local +x; yaw +90 maps it onto world +y, origin at the end
        yaw90 = (0.70710678118654752, 0, 0, 0.70710678118654752)
        # band y in [-0.8, -0.3]; body spans [origin, origin+0.5]; wall at
        # y=0 => origin <= -0.5, so the band is PARTLY legal and must be cut
        # rather than moved.
        item = _wall_item(d, p=(2.0, -0.8, 0.95), rp=(0.0, 0.5, 0.0),
                          qpos=yaw90)
        walls = (("back", 0.0, -10.0, 10.0),)
        rep = clear_of_walls(item, walls, d)
        assert rep["shift"] > 0 and rep["walls"] == ["back"]
        assert rep["feasible"] is True
        lo = item["pos"]["ppos"]["p"][1]
        hi = lo + item["pos"]["randp_scale"][1]
        assert abs(lo - (-0.8)) < 1e-9, lo         # legal end untouched
        assert abs(hi - (-0.5)) < 1e-9, hi         # infeasible end cut back
        assert item["pos"]["ppos"]["p"][0] == 2.0  # x never touched

        # and when NO part of the band is legal, translate to the nearest
        # feasible point rather than leaving the body through the wall
        item = _wall_item(d, p=(2.0, -0.2, 0.95), rp=(0.0, 0.3, 0.0),
                          qpos=yaw90)
        assert clear_of_walls(item, walls, d)["shift"] > 0
        lo = item["pos"]["ppos"]["p"][1]
        assert abs(lo - (-0.5)) < 1e-9, lo
        assert abs(item["pos"]["randp_scale"][1]) < 1e-9


def test_clear_of_walls_is_a_noop_for_a_body_that_already_clears():
    """The property the targeted re-render depends on: a config the pass does
    not move must come out byte-identical, or it lands in the delta set and
    costs a render for nothing."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        item = _wall_item(d, p=(2.0, -3.0, 0.95), rp=(0.0, 0.3, 0.0))
        before = copy.deepcopy(item)
        rep = clear_of_walls(item, (("back", 0.0, -10.0, 10.0),), d)
        assert rep["shift"] == 0.0
        assert item == before


def test_clear_of_walls_ignores_a_wall_the_body_is_not_in_front_of():
    """Walls are finite rectangles, not half-space boundaries: layout 9 is
    L-shaped and carries a `left` wall spanning only y in [-4.54, -3.41]. A
    body elsewhere along y must not be constrained by it."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        item = _wall_item(d, p=(1.0, -1.0, 0.95), rp=(0.0, 0.2, 0.0))
        before = copy.deepcopy(item)
        # a `left` wall at x=2.075 would otherwise shove this body to x>=2.075
        rep = clear_of_walls(item, (("left", 2.075, -4.54, -3.41),), d)
        assert rep["walls"] == [] and rep["shift"] == 0.0
        assert item == before


def test_clear_of_walls_never_teleports_a_body_across_a_wall():
    """If the reach box's centre is already on the far side, the wall belongs
    to another part of the floor plan and enforcing it would fling the body
    bodily into the other half of the room."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        item = _wall_item(d, p=(1.0, -3.8, 0.95), rp=(0.0, 0.2, 0.0))
        before = copy.deepcopy(item)
        rep = clear_of_walls(item, (("left", 2.075, -10.0, 10.0),), d)
        assert rep["walls"] == [] and rep["shift"] == 0.0
        assert item == before


def test_hull_xy_box_is_tighter_than_the_aabb_bound_and_never_wider():
    """`rotated_xy_box` transforms the AABB's 8 corners, which bounds the hull
    from OUTSIDE; under a non-axis-aligned rotation that slack is real (26% of
    banana's y extent) and every slack centimetre drags a clean image into the
    re-render set. Case (e) keeps the bound on purpose, so the two must stay
    distinct functions -- this pins that they do."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        _write_obj(os.path.join(d, "bar.obj"),
                   [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        path = os.path.join(d, "bar.obj")
        q = [0.9238795325112867, 0.0, 0.0, 0.3826834323650898]   # 45 deg yaw
        bound = rotated_xy_box(path, 1.0, q)
        hull = hull_xy_box(path, 1.0, q)
        for ax in (0, 1):
            assert bound[ax][0] <= hull[ax][0] + 1e-12
            assert bound[ax][1] >= hull[ax][1] - 1e-12
        # the triangle is not its own AABB, so the bound is strictly looser
        assert (bound[1][1] - bound[1][0]) > (hull[1][1] - hull[1][0]) + 1e-9
        # identity rotation: nothing to be loose about
        assert hull_xy_box(path, 1.0, [1, 0, 0, 0]) == \
            rotated_xy_box(path, 1.0, [1, 0, 0, 0])


def test_wall_plane_table_brackets_every_cabinet_interior():
    """`_WALL_PLANES` and `_CABINET_INTERIOR` were measured through unrelated
    code paths (`Wall.pos` vs `get_int_sites`). Every cabinet is indoors, so
    the two tables have to agree -- and if the interior sign in `_WALL_SIDES`
    were inverted, the clearance pass would push bodies OUT of the room and
    this is what would catch it."""
    for layout, interior in _CABINET_INTERIOR.items():
        for side, coord, lat_lo, lat_hi in _WALL_PLANES[layout]:
            axis, inward = _WALL_SIDES[side]
            lat = 1 - axis
            # only walls that actually span the cabinet can bound it
            if interior[lat][1] < lat_lo or interior[lat][0] > lat_hi:
                continue
            for end in interior[axis]:
                assert inward * (end - coord) >= -1e-6, (
                    f"layout {layout}: cabinet interior {end:.3f} is outside "
                    f"the {side} wall at {coord:.3f}")


def test_every_layout_wall_is_measured_and_bounded():
    """A missing layout would raise at config-build time rather than silently
    skipping the pass, so the table has to cover all nine."""
    assert set(_WALL_PLANES) == set(ALL_LAYOUTS), sorted(_WALL_PLANES)
    for layout, walls in _WALL_PLANES.items():
        assert walls, layout
        for side, coord, lat_lo, lat_hi in walls:
            assert side in _WALL_SIDES, (layout, side)
            assert lat_lo < lat_hi, (layout, side)


def test_clear_of_walls_leaves_primitives_and_missing_assets_alone():
    """Same contract as `rest_offset_z`: a primitive has no file to measure
    and a missing asset is the renderer's error to raise, not this pass's."""
    prim = {"name": "cube", "pos": {"ppos": {"p": [1.0, -0.1, 0.9]},
                                    "qpos": [1, 0, 0, 0]}}
    before = copy.deepcopy(prim)
    assert clear_of_walls(prim, (("back", 0.0, -10.0, 10.0),))["shift"] == 0.0
    assert prim == before
    ghost = {"name": "ghost", "visual": {"filename": "nope/missing.obj"},
             "pos": {"ppos": {"p": [1.0, -0.1, 0.9]}, "qpos": [1, 0, 0, 0]}}
    before = copy.deepcopy(ghost)
    assert clear_of_walls(ghost, (("back", 0.0, -10.0, 10.0),))["shift"] == 0.0
    assert ghost == before


def _wall_touched_stems(insts):
    """Stems whose config case (f) actually moves something in."""
    return {render_stem(i) for rows in insts.values() for i in rows
            if "locus_wall_clear" in build_render_config(i)[0]}


def test_wall_clear_scope_restricts_the_pass_to_listed_stems():
    """docs/YOLO.md §6.2.5 repairs a subset, so the scope has to be exact in
    both directions: in-scope configs keep the fix, out-of-scope ones must
    come back byte-identical to the unscoped-OFF build or `render_batch`
    deletes a snapshot that is still valid."""
    import yaml
    plan = build_split_plan(n_train=40, n_test_id=12, n_ood=28)
    insts, _ = generate_split_instances(plan, seed=7)
    touched = sorted(_wall_touched_stems(insts))
    assert len(touched) >= 4, f"only {len(touched)} wall-moved configs sampled"
    scope = set(touched[:2])

    for rows in insts.values():
        for inst in rows:
            stem = render_stem(inst)
            scoped = yaml.dump(build_render_config(
                inst, wall_clear_scope=scope)[0])
            everywhere = yaml.dump(build_render_config(inst)[0])
            off = yaml.dump(build_render_config(
                inst, wall_clear_scope=set())[0])
            if stem in scope:
                assert scoped == everywhere, stem
                assert scoped != off, f"{stem} in scope but pass was a no-op"
            else:
                assert scoped == off, stem
                if stem in touched:
                    assert scoped != everywhere, stem


def test_wall_clear_scope_none_is_the_unrestricted_default():
    """The scope is an opt-in for one repair round; absent it, every config
    gets the fix (a fresh pack must never inherit the subset)."""
    import yaml
    for rows in SPLITS.values():
        for inst in rows:
            assert (yaml.dump(build_render_config(inst)[0])
                    == yaml.dump(build_render_config(
                        inst, wall_clear_scope=None)[0]))


def test_wall_clear_scope_rejects_stems_from_a_different_plan():
    """A scope is derived by auditing an existing pack. If the plan since
    changed shape the same stem names a different scene, so the loader must
    fail rather than silently repair the wrong configs."""
    import locus_pilot
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scope.txt")
        good = render_stem(next(iter(SPLITS["train"])))
        with open(path, "w") as f:
            f.write(f"{good}\ntrain_99999\n")
        try:
            locus_pilot._load_wall_clear_scope(path, SPLITS)
        except SystemExit as exc:
            assert "train_99999" in str(exc), exc
        else:
            raise AssertionError("stale scope was accepted")
        with open(path, "w") as f:
            f.write(f"{good}\n")
        stems, digest = locus_pilot._load_wall_clear_scope(path, SPLITS)
        assert stems == {good} and len(digest) == 64


def test_render_batch_keeps_out_of_scope_renders_and_drops_in_scope_ones():
    """The end-to-end contract of a scoped repair round: a config the scope
    excludes must survive the stale-check with its snapshot intact, and an
    included one must have its snapshot deleted so it re-renders."""
    insts = [copy.deepcopy(i) for i in SPLITS["train"]]
    touched = {render_stem(i) for i in insts
               if "locus_wall_clear" in build_render_config(i)[0]}
    if not touched:
        return              # tiny plan drew no wall-adjacent scene: nothing to prove
    scope = {sorted(touched)[0]}
    with tempfile.TemporaryDirectory() as d:
        # lay down the PRE-FIX pack, then stage a snapshot for every sample
        render_batch(insts, d, "train", dry_run=True, wall_clear_scope=set())
        split_dir = os.path.join(d, "train")
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "gt_labels"), exist_ok=True)
        for inst in insts:
            idx = inst["meta"]["sample_index"]
            for p in (os.path.join(split_dir, "images", f"{idx:05d}.png"),
                      os.path.join(split_dir, "gt_labels", f"{idx:05d}.json")):
                with open(p, "w") as f:
                    f.write("stub")
        render_batch(insts, d, "train", dry_run=True, wall_clear_scope=scope)
        for inst in insts:
            idx = inst["meta"]["sample_index"]
            alive = os.path.isfile(
                os.path.join(split_dir, "images", f"{idx:05d}.png"))
            assert alive == (render_stem(inst) not in scope), render_stem(inst)


if __name__ == "__main__":
    for fn in [test_split_plan_grid,
               test_layout_holdout,
               test_paraphrase_holdout,
               test_style_hook,
               test_parquet_parity,
               test_decidability_check,
               test_real_render_artifacts_if_present,
               test_shared_rng_couples_plan_rows,
               test_per_row_rng_is_the_default,
               test_pilot_driver_defaults_to_the_production_identity_knobs,
               test_scoped_rerender_keeps_the_whole_benchmark_in_the_report,
               test_pilot_report_records_both_identity_knobs,
               test_relocate_reuses_renders_whose_index_shifted,
               test_relocate_handles_a_pure_permutation,
               test_relocate_is_a_noop_when_nothing_moved,
               test_relocate_skips_configs_with_no_render_on_disk,
               test_per_row_rng_makes_a_ramp_additive,
               test_per_row_rng_keeps_realization_balance,
               test_reach_box_is_one_sided,
               test_chebyshev_separation_is_per_axis,
               test_placement_model_matches_an_analytic_geometry,
               test_cross_anchor_conflict_is_resolved,
               test_cross_anchor_reslot_keeps_what_jitter_it_can,
               test_coverage_probability_is_exact,
               test_cross_anchor_gate_leaves_roomy_scenes_untouched,
               test_cross_anchor_reslot_is_idempotent,
               test_cross_anchor_never_moves_agents,
               test_pilot_configs_are_placement_feasible,
               test_mesh_aabb_reads_obj_and_glb_without_a_mesh_library,
               test_rest_offset_lifts_exactly_what_hangs_below_the_plane,
               test_every_layout_object_now_rests_on_its_plane,
               test_rest_offset_only_moves_z_and_only_downward_offenders,
               test_rotated_xy_box_is_signed_and_survives_yaw,
               test_clamp_interval_shrinks_translates_and_collapses,
               test_fit_in_cabinet_insets_within_the_carcass,
               test_fit_in_cabinet_prefers_the_smaller_disturbance,
               test_fit_in_cabinet_yaws_a_body_that_cannot_fit_as_authored,
               test_cabinet_interior_table_matches_the_layouts_it_claims,
               test_reslot_keeps_a_bounded_member_inside_its_carcass,
               test_plate_is_the_only_body_no_cabinet_can_hold,
               test_every_containable_cabinet_object_ends_up_inside_the_cabinet,
               test_clear_of_walls_trims_only_the_infeasible_end_of_the_reach_box,
               test_clear_of_walls_is_a_noop_for_a_body_that_already_clears,
               test_clear_of_walls_ignores_a_wall_the_body_is_not_in_front_of,
               test_clear_of_walls_never_teleports_a_body_across_a_wall,
               test_hull_xy_box_is_tighter_than_the_aabb_bound_and_never_wider,
               test_wall_plane_table_brackets_every_cabinet_interior,
               test_every_layout_wall_is_measured_and_bounded,
               test_clear_of_walls_leaves_primitives_and_missing_assets_alone,
               test_wall_clear_scope_restricts_the_pass_to_listed_stems,
               test_wall_clear_scope_none_is_the_unrestricted_default,
               test_wall_clear_scope_rejects_stems_from_a_different_plan,
               test_render_batch_absolutises_out_dir,
               test_render_batch_keeps_out_of_scope_renders_and_drops_in_scope_ones]:
        fn()
        print(f"{fn.__name__}: OK")
