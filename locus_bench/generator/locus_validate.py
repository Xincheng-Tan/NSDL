"""
LOCUS-Bench validation & audit layer (P7.3, docs/benchmark.md §6.3/§7/§9).

Four deliverables, all reusing official/existing validators (no semantic
changes to official files):

  1. ``generate_validated_batch`` — generation loop with the official-SimEnv
     replay gate INLINED (reject-and-resample, same realization kept on
     retry so balance accounting is unaffected; benchmark.md §6.3-1).
  2. ``phi_gate`` — Φ reconstruction gate (benchmark.md §6.3-2/§6.1): the
     instance GT is converted to VIKI-parquet shape, reverse-mapped to an
     intent sketch (mechanical details erased) and rebuilt by
     ``nsdl.planning_tool.schedule`` + ``verify``. Empty plans pass
     trivially (nothing to reconstruct).
  3. ``build_audit_report`` — ``audit_report.json`` with benchmark.md §7
     keys 1–6 computed, 7–8 marked pending (P7.5).
  4. ``eval_single_locus`` — empty-plan-aware scoring wrapper around the
     official SimEnv evaluation (benchmark.md §6.2 "空 plan 语义"):
     empty prediction is a legal input evaluated against goal constraints
     on the untouched initial state (⇒ success iff realization is
     all-present); a non-empty prediction that moves a pre-satisfied item
     away fails the goal check (over-grasp) — no official file modified.

Determinism: SimEnv metadata construction pins each asset's initial
position with a seeded rng (the official ``eval_gt.py`` uses
``random.choice``); for LOCUS this choice never affects the verdict
(open-surface alternatives are goal-equivalent) but pinning keeps runs
byte-reproducible.
"""

import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_ROBOFACTORY)
for _p in (_PROJECT_ROOT, _ROBOFACTORY,
           os.path.join(_ROBOFACTORY, "script_general")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from locus_task_pool import enumerate_realizations, iter_cells, make_template
from locus_generate import (
    BridgeVariantSampler, RealizationSampler, instantiate_counterfactual_pair,
    instantiate_locus_task, realization_balance_report,
)
from utils.eval.eval import Eval

__all__ = [
    "simenv_replay",
    "phi_gate",
    "generate_validated_batch",
    "build_audit_report",
    "eval_single_locus",
]

CONTAINER_ASSETS = ['plate', 'cabinet', 'drawer', 'bowl', 'sink', 'toaster',
                    'tray', 'cardboardbox']


# ---------------------------------------------------------------------------
# 1. official SimEnv replay (canonical implementation; eval_gt.py semantics)
# ---------------------------------------------------------------------------

def _build_metadata(inst, rng):
    metadata = {"agents": {}, "assets": {}}
    for robot_id, robot_type in inst["robots"].items():
        metadata["agents"][robot_id] = {"type": robot_type,
                                        "pos": {"name": robot_id}}
    for asset_name, asset_pos in inst["init_pos"].items():
        if asset_name.startswith('R') and asset_name[1:].isdigit():
            continue
        asset_type = asset_name.rsplit('_', 1)[0]
        metadata["assets"][asset_type] = {
            "pos": {"name": rng.choice(asset_pos)}}
        if asset_type in CONTAINER_ASSETS:
            metadata["assets"][asset_type]['params'] = {
                "is_container": True,
                "position_kwargs": {
                    "name": asset_type,
                    "isolated": asset_type in ['cabinet'],
                },
            }
    metadata['goal_constraints'] = inst['goal_constraints']
    metadata['temporal_constraints'] = inst['temporal_constraints']
    return metadata


def _format_answer(steps):
    return [{r: f'<{",".join(a)}>' for r, a in step.items()} for step in steps]


def simenv_replay(inst, steps=None, rng=None):
    """Replay ``steps`` (default: the instance GT) in the official SimEnv.

    Returns (ok, error_desc).
    """
    rng = rng or random.Random(0)
    judger = Eval()
    judger.set_env(_build_metadata(inst, rng))
    with redirect_stdout(io.StringIO()):
        ok = judger.eval(_format_answer(
            inst["ground_truth"] if steps is None else steps))
    return ok, (None if ok else judger.get_error_desc())


# ---------------------------------------------------------------------------
# 2. Φ reconstruction gate
# ---------------------------------------------------------------------------

def to_viki_sample_data(inst):
    """LOCUS instance → the sample_data dict shape consumed by the Φ audit
    chain (scripts.build_slm_training_data / audit_planning_tool_p1)."""
    robots_raw = {"R1": None, "R2": None, "R3": None}
    robots_raw.update(inst["robots"])
    time_steps = []
    for i, step in enumerate(inst["ground_truth"]):
        actions = {"R1": None, "R2": None, "R3": None}
        for robot, op in step.items():
            actions[robot] = list(op)
        time_steps.append({"step": i + 1, "actions": actions})
    robot_inventory = [{"slot": slot, "type": rtype}
                       for slot, rtype in sorted(robots_raw.items())
                       if rtype is not None]
    return {
        "instruction": inst["description"],
        "description": inst["description"],
        "robot_inventory": robot_inventory,
        "time_steps": time_steps,
        "task_name": inst["task_name"],
        "task_id": inst["task_id"],
        "goal_constraints": inst["goal_constraints"],
        "robots_raw": robots_raw,
    }


def make_locus_sketch(gt_ops):
    """LOCUS Φ-gate sketch: erase mechanical ops and cross-robot deps, KEEP
    each robot's own sequential chain.

    Deviation from the P1 ``make_sketch`` variants (which erase all deps):
    LOCUS M1 cells contain a pattern absent from VIKI train — an ee=1 robot
    serially relocating several independent items. Φ synthesizes
    same-entity/custody deps but not ee-forced serialization between
    different items, so a fully deps-erased sketch hits ``ee_capacity``.
    The per-robot op order is the GT's own intent serialization (not a
    mechanical synthesis product), so the sketch keeps it; open/close/
    carrier ops and cross-robot deps stay erased for Φ to rebuild.
    """
    from nsdl.schema import OperationStep
    from scripts.audit_planning_tool_p1 import MECHANICAL_OPS

    sketch = []
    last_by_robot = {}
    for op in gt_ops:
        if op.op_type == "unload_carrier":
            new_op = OperationStep(
                op_id=op.op_id, op_type="deposit", entity=op.entity,
                dest=op.dest, deps=[], robot=op.robot, params={},
                gaps=["deps", "precond"],
            )
        elif op.op_type in MECHANICAL_OPS or op.op_type == "open":
            continue
        else:
            new_op = OperationStep(
                op_id=op.op_id, op_type=op.op_type, entity=op.entity,
                dest=op.dest, deps=[], robot=op.robot,
                params=dict(op.params or {}),
                gaps=["deps", "precond"] if op.op_type == "deposit" else ["deps"],
            )
        prev = last_by_robot.get(op.robot)
        if prev is not None:
            new_op.deps = [prev]
        last_by_robot[op.robot] = new_op.op_id
        sketch.append(new_op)
    return sketch


def phi_gate(inst, variant="intent"):
    """sketch → schedule → verify reconstruction gate (benchmark.md §6.1).

    Returns (ok, code) where code is None on success. Empty plans pass
    trivially (nothing to reconstruct)."""
    if not inst["ground_truth"]:
        return True, None
    import os
    # The gate's recorded pass rates (benchmark.md §6.1, phi_sketch_opschain .966)
    # were scored against Phi's v1 list scheduler; keep it pinned so re-audits
    # stay comparable even though the eval main configuration moved to v2
    # (plan.md A0p, 2026-09-06). Ground truth itself is template-built above
    # and does not depend on the scheduler.
    os.environ.setdefault("LNSDL_PHI_SCHEDULER", "v1")
    from nsdl.aig_builder import build_affordance_context
    from nsdl.planning_tool import schedule, verify
    from scripts.build_slm_training_data import (
        build_scene_facts_from_gt, reverse_map_time_steps,
    )

    sample_data = to_viki_sample_data(inst)
    try:
        scene_facts = build_scene_facts_from_gt(sample_data, idx=0)
        ctx = build_affordance_context(scene_facts)
        gt_ops = reverse_map_time_steps(sample_data)
        sketch = make_locus_sketch(gt_ops)
        result = schedule(sketch, assign={}, ctx=ctx, scene_facts=scene_facts)
        if not result.ok:
            code = result.reason.code if result.reason else "schedule_failed"
            return False, code
        reason = verify(result.ops, ctx, scene_facts)
        if not reason.ok:
            return False, reason.code or "verify_failed"
        return True, None
    except Exception as exc:  # noqa: BLE001 - audit must not abort the batch
        return False, f"exception:{exc.__class__.__name__}"


# ---------------------------------------------------------------------------
# 3. generation loop with inlined gates (reject-and-resample)
# ---------------------------------------------------------------------------

def generate_validated_batch(n_per_cell, seed=0, families=None,
                             max_retries=20, run_phi=True, **kwargs):
    """Balanced, gate-validated batch.

    Every emitted instance passed the SimEnv replay gate; retries keep the
    sampler-assigned realization so balance accounting is unaffected.
    Returns (instances, stats). Φ results are recorded per instance in
    ``meta.phi_ok``/``meta.phi_code`` (diagnostic; SimEnv is the blocking
    gate, Φ agreement is audited batch-level in the report, §7-6)."""
    rng = random.Random(seed)
    fixed_variant = kwargs.pop("bridge_variant", None)
    sampler = RealizationSampler()
    variant_sampler = BridgeVariantSampler()
    instances, stats = [], Counter()
    for fam, s, m in iter_cells():
        if families and fam not in families:
            continue
        pending_pair = []
        for _ in range(n_per_cell):
            inst = None
            realization = None
            if fam == "F4" and s == 4:
                if not pending_pair:
                    for attempt in range(max_retries):
                        pair = instantiate_counterfactual_pair(
                            fam, s, m, rng, **kwargs)
                        pair_results = []
                        for cand in pair:
                            result = simenv_replay(
                                cand, rng=random.Random(
                                    cand["meta"]["instantiate_seed"]))
                            pair_results.append(result)
                            stats["simenv_checked"] += 1
                        if all(ok for ok, _ in pair_results):
                            pending_pair.extend(pair)
                            if attempt:
                                stats["resampled"] += attempt
                            break
                        stats["simenv_rejected"] += sum(
                            not ok for ok, _ in pair_results)
                        for ok, err in pair_results:
                            if not ok:
                                stats[f"reject:{err}"] += 1
                    if not pending_pair:
                        raise RuntimeError(
                            f"{fam} s{s}m{m}: no valid counterfactual pair "
                            f"in {max_retries} attempts")
                inst = pending_pair.pop(0)
            else:
                realization = sampler.next(fam, s, m, rng)
                bridge_variant = (
                    fixed_variant or variant_sampler.next(fam, s, m, rng))
                for attempt in range(max_retries):
                    cand = instantiate_locus_task(
                        fam, s, m, rng, realization=realization,
                        bridge_variant=bridge_variant, **kwargs)
                    ok, err = simenv_replay(cand, rng=random.Random(
                        cand["meta"]["instantiate_seed"]))
                    stats["simenv_checked"] += 1
                    if ok:
                        inst = cand
                        if attempt:
                            stats["resampled"] += attempt
                        break
                    stats["simenv_rejected"] += 1
                    stats[f"reject:{err}"] += 1
            if inst is None:
                raise RuntimeError(
                    f"{fam} s{s}m{m} {realization}: no valid instance in "
                    f"{max_retries} attempts (last error: {err})")
            if run_phi:
                phi_ok, phi_code = phi_gate(inst)
                inst["meta"]["phi_ok"] = phi_ok
                inst["meta"]["phi_code"] = phi_code
                stats["phi_ok" if phi_ok else "phi_fail"] += 1
                if not phi_ok:
                    stats[f"phi:{phi_code}"] += 1
            inst["meta"]["sample_index"] = len(instances)
            instances.append(inst)
    return instances, dict(stats)


# ---------------------------------------------------------------------------
# 4. audit report (benchmark.md §7 keys)
# ---------------------------------------------------------------------------

def _vocab_coverage(instances):
    """§7-3: candidate×destination coverage per family over the batch."""
    seen = defaultdict(set)
    for inst in instances:
        meta = inst["meta"]
        spec_cands = {}
        tpl = make_template(meta["bench_family"], *meta["cell"],
                            random.Random(0))
        for cand in tpl["state_realization_spec"]["candidates"]:
            spec_cands[cand["mask_key"]] = cand.get("dest_mask")
        for key, dest_key in spec_cands.items():
            value = meta["mask_map"].get(key)
            dest = meta["mask_map"].get(dest_key, dest_key)
            if value:
                seen[meta["bench_family"]].add((value, dest))
    full = defaultdict(set)
    for fam, s, m in iter_cells():
        tpl = make_template(fam, s, m, random.Random(0))
        for cand in tpl["state_realization_spec"]["candidates"]:
            dests = tpl.get(cand.get("dest_mask"), [cand.get("dest_mask")])
            for v in tpl[cand["mask_key"]]:
                for d in dests:
                    full[fam].add((v, d))
    report = {}
    for fam in sorted(full):
        holes = sorted(full[fam] - seen.get(fam, set()))
        report[fam] = {
            "pairs_total": len(full[fam]),
            "pairs_covered": len(seen.get(fam, set())),
            "coverage": round(len(seen.get(fam, set())) / len(full[fam]), 4)
            if full[fam] else 1.0,
            "holes": [list(h) for h in holes[:20]],
        }
    return report


def _leak_audit():
    """§7-4: paired-realization instruction identity per cell (instance-level
    anti-shortcut; the answer lives only in the scene)."""
    results = {}
    for fam, s, m in iter_cells():
        reals = enumerate_realizations(fam, s, m)
        if len(reals) < 2:
            continue
        descs = set()
        for realization in (reals[0], reals[-1]):
            inst = instantiate_locus_task(
                fam, s, m, random.Random(99), realization=realization,
                inject_distractors=False)
            descs.add(inst["description"])
        results[f"{fam}_s{s}m{m}"] = len(descs) == 1
    return results


def _orthogonality_audit(instances):
    """§7-5 / §2.3: structure stats move only with m; state stats only with s.

    Key: at fixed (family, m), the SET of realization-conditioned structural
    signatures {(n_robots, has_open)} must be identical across s levels
    (within-cell variance across realizations is S-semantics — e.g. S4
    hidden ⇒ open branch — and is expected). Dually, at fixed (family, s),
    the realization-class count must be identical across m levels.
    """
    struct = defaultdict(lambda: defaultdict(set))  # (fam,m) -> s -> sigs
    state = defaultdict(lambda: defaultdict(set))   # (fam,s) -> m -> n_real
    for inst in instances:
        meta = inst["meta"]
        fam, (s, m) = meta["bench_family"], meta["cell"]
        if inst["ground_truth"]:  # empty plans carry no structure signal
            has_open = any(op[0] == "Open" for step in inst["ground_truth"]
                           for op in step.values())
            struct[(fam, m)][s].add((len(inst["robots"]), has_open))
        state[(fam, s)][m].add(meta["n_realizations"])
    struct_bad = {
        f"{fam}_m{m}": {f"s{s}": sorted(v) for s, v in by_s.items()}
        for (fam, m), by_s in struct.items()
        if len({frozenset(v) for v in by_s.values()}) > 1
    }
    state_bad = {
        f"{fam}_s{s}": {f"m{m}": sorted(v) for m, v in by_m.items()}
        for (fam, s), by_m in state.items()
        if len({frozenset(v) for v in by_m.values()}) > 1
    }
    return {
        "structure_varies_with_s_at_fixed_m": struct_bad,  # should be empty
        "state_varies_with_m_at_fixed_s": state_bad,       # should be empty
        "ok": not struct_bad and not state_bad,
    }


def _split_name(s, m):
    if s <= 2 and m <= 2:
        return "train"
    if s >= 3 and m <= 2:
        return "ood_s"
    if s <= 2 and m >= 3:
        return "ood_m"
    return "ood_x"


def _count_lookup(value):
    if isinstance(value, dict):
        return sum(int(str(k).lower() == "lookup") + _count_lookup(v)
                   for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_count_lookup(v) for v in value)
    return int(isinstance(value, str) and value.strip().lower() == "lookup")


def _bridge_audit(instances):
    """§7-9: family/operator support and exact-composition holdout."""
    train = [i for i in instances if _split_name(*i["meta"]["cell"]) == "train"]
    ood = [i for i in instances if _split_name(*i["meta"]["cell"]) != "train"]
    ood_families = sorted({i["meta"]["bench_family"] for i in ood})
    train_family_counts = Counter(i["meta"]["bench_family"] for i in train)
    family_rows = {
        fam: {
            "train_count": train_family_counts[fam],
            "ood_count": sum(i["meta"]["bench_family"] == fam for i in ood),
            "ok": train_family_counts[fam] > 0,
        }
        for fam in ood_families
    }

    support_specs = {
        "tag_quantifier_detect_tag": lambda i: (
            i["meta"]["bridge_kind"] == "tag_quantifier_k2"
            and i["meta"]["observation_tool"] == {
                "name": "detect",
                "args": {"query": {"tag": "fruit"}},
            }),
        "transparent_fallback": lambda i: (
            i["meta"]["bridge_kind"] == "transparent_fallback"),
        "bounded_clear_set": lambda i: (
            i["meta"]["bridge_kind"] == "bounded_clear_set_k2"),
        "independent_deposit_twin": lambda i: (
            i["meta"]["bridge_kind"] == "independent_deposit_twin"),
        "carrier_load_component": lambda i: (
            i["meta"]["bridge_kind"] == "carrier_components_two_robot"
            and i["meta"].get("bridge_variant") == "load"),
        "carrier_push_component": lambda i: (
            i["meta"]["bridge_kind"] == "carrier_components_two_robot"
            and i["meta"].get("bridge_variant") == "push"),
        "carrier_unload_component": lambda i: (
            i["meta"]["bridge_kind"] == "carrier_components_two_robot"
            and i["meta"].get("bridge_variant") == "unload"),
    }
    support = {}
    for name, predicate in support_specs.items():
        count = sum(bool(predicate(i)) for i in train)
        support[name] = {
            "train_count": count,
            "required": 100,
            "structural_support": count > 0,
            "scale_gate": count >= 100,
        }
    f4_binary = Counter(
        i["meta"]["state_realization"].get("mask1")
        for i in train
        if i["meta"]["bench_family"] == "F4"
        and i["meta"]["cell"][0] == 2)
    support["fallback_binary_realizations"] = {
        "counts": dict(sorted(f4_binary.items())),
        "required_each": 20,
        "structural_support": all(f4_binary.get(k, 0) > 0
                                  for k in ("cabinet", "counter")),
        "scale_gate": all(f4_binary.get(k, 0) >= 20
                          for k in ("cabinet", "counter")),
    }

    high_level_train = [
        i["meta"]["sample_index"] for i in train
        if i["meta"]["cell"][0] >= 3 or i["meta"]["cell"][1] >= 3
    ]
    train_signatures = {
        json.dumps({
            "family": i["meta"]["bench_family"],
            "s_level": i["meta"]["cell"][0],
            "mechanical": i["meta"]["base_mechanical_signature"],
            "opaque": i["meta"]["opaque_container"],
        }, sort_keys=True)
        for i in train
    }
    ood_signatures = {
        json.dumps({
            "family": i["meta"]["bench_family"],
            "s_level": i["meta"]["cell"][0],
            "mechanical": i["meta"]["base_mechanical_signature"],
            "opaque": i["meta"]["opaque_container"],
        }, sort_keys=True)
        for i in ood
    }
    leaked_signatures = sorted(train_signatures & ood_signatures)
    lookup_count = sum(_count_lookup(i) for i in instances)
    return {
        "family_train_coverage": {
            "families": family_rows,
            "ok": all(row["ok"] for row in family_rows.values()),
        },
        "operator_train_support": {
            "operators": support,
            "lookup_count": lookup_count,
            "structural_ok": (
                all(row["structural_support"] for row in support.values())
                and lookup_count == 0),
            "scale_gate_ok": (
                all(row["scale_gate"] for row in support.values())
                and lookup_count == 0),
        },
        "composition_holdout": {
            "high_level_train_rows": high_level_train,
            "exact_ood_signatures_in_train": leaked_signatures,
            "ok": not high_level_train and not leaked_signatures,
        },
    }


def _s4_pair_audit(instances):
    """§7-10: paired static-scene semantics, without active-perception claims."""
    groups = defaultdict(list)
    for inst in instances:
        pair_id = inst["meta"].get("counterfactual_pair_id")
        if pair_id:
            groups[pair_id].append(inst)
    checked = failures = incomplete = 0
    for pair_id, pair in groups.items():
        if len(pair) != 2:
            incomplete += 1
            continue
        checked += 1
        by_state = {
            i["meta"]["state_realization"].get("mask1"): i for i in pair
        }
        if set(by_state) != {"cabinet", "counter"}:
            failures += 1
            continue
        hidden, counter = by_state["cabinet"], by_state["counter"]
        target = hidden["meta"]["mask_map"]["mask1"]
        hidden_pos = next(
            list(v) for k, v in hidden["init_pos"].items()
            if k.rsplit("_", 1)[0] == target)
        counter_pos = next(
            list(v) for k, v in counter["init_pos"].items()
            if k.rsplit("_", 1)[0] == target)
        invariant = (
            hidden["description"] == counter["description"]
            and hidden["layout_id"] == counter["layout_id"]
            and hidden["meta"]["style_seed"] == counter["meta"]["style_seed"]
            and hidden["meta"]["mask_map"] == counter["meta"]["mask_map"]
            and hidden["meta"]["opaque_container"]
            and counter["meta"]["opaque_container"])
        semantic = (
            hidden_pos == ["cabinet"]
            and counter_pos != ["cabinet"]
            and hidden["meta"]["state_entailed_open"]
            and not counter["meta"]["state_entailed_open"])
        if not invariant or not semantic:
            failures += 1
    return {
        "pairs_checked": checked,
        "incomplete_pairs": incomplete,
        "failures": failures,
        "post_open_observation_claims": 0,
        "ok": checked > 0 and failures == 0 and incomplete == 0,
    }


def _init_areas(init_pos):
    """asset base name -> the position areas it occupies at init."""
    areas = defaultdict(set)
    for key, pos in init_pos.items():
        if key.startswith("R") and key[1:].isdigit():
            continue
        for p in (pos or []):
            areas[key.rsplit("_", 1)[0]].add(p)
    return areas


def goal_text_recoverability(instances, max_examples=20):
    """§6.3-4: every goal the scene does not already satisfy must be *sayable*
    from the instruction.

    The three §6.3 gates in place before this one (SimEnv replay, Φ
    reconstruction, decidability) all reason from the goal downward, so none
    of them can see a goal the *text* never mentions -- Φ is handed the goal
    and reconstructs a plan for it, and it happily reports ``phi_ok`` on a
    sample whose instruction is missing that goal entirely. That is exactly
    how two families shipped un-answerable rows into the frozen splits
    (`_F9_DESC_M3` naming no destination, `_F4_DESC` not covering M2's second
    constraint; docs/benchmark.md §7.6).

    The check is deliberately narrow -- destination *positions*, not asset
    names. Quantified families (F3 "all fruits", F5 "everything on the
    surface") name their targets by predicate on purpose, so requiring every
    goal asset to appear verbatim would flag them by design. A goal already
    satisfied at init is exempt: "nothing to do" is recoverable from the scene
    and is precisely what the S axis asks the model to judge.
    """
    checked = 0
    violations = []
    for inst in instances:
        text = inst["description"].lower()
        areas = _init_areas(inst["init_pos"])
        for group in inst.get("goal_constraints", []):
            for cond in group:
                pos = (cond.get("status") or {}).get("pos.name")
                if not isinstance(pos, str):
                    continue          # is_activated etc. -- no destination
                checked += 1
                if pos in areas.get(cond.get("name"), ()):
                    continue          # already true at init; see docstring
                if pos.lower() in text:
                    continue
                violations.append({
                    "task_name": inst["task_name"],
                    "cell": list(inst["meta"]["cell"]),
                    "paraphrase_id": inst["meta"]["paraphrase_id"],
                    "asset": cond.get("name"),
                    "unstated_destination": pos,
                    "description": inst["description"],
                })
    return {
        "goals_checked": checked,
        "n_violations": len(violations),
        "violations": violations[:max_examples],
        "ok": not violations,
    }


def build_audit_report(instances, stats=None):
    per_cell = Counter(
        f'{i["meta"]["bench_family"]}_s{i["meta"]["cell"][0]}'
        f'm{i["meta"]["cell"][1]}' for i in instances)
    phi_checked = [i for i in instances if "phi_ok" in i["meta"]]
    phi_ok = sum(1 for i in phi_checked if i["meta"]["phi_ok"])
    balance = realization_balance_report(instances)
    bridge = _bridge_audit(instances)
    report = {
        # 生成器永远产 v0-pilot：一批**新生成**的实例按定义不是被冻结的那份 bench。
        # v1.0 冻结章由 scripts/freeze_locus_bench.py 在四门复验通过后事后写入
        # （config + audit_report 两处）。重跑本驱动会把冻结报告覆盖回 v0-pilot，
        # 这是正确语义——重跑意味着数据变了，须重新过门再盖章。
        "bench_version": "v0-pilot",
        "n_instances": len(instances),
        "per_cell_counts": dict(sorted(per_cell.items())),
        "generation_stats": stats or {},
        # §7-1 official-base declaration
        "official_base": {
            "generator": "script_general/locus_task_pool.py + locus_generate.py",
            "official_files_modified": [],
            "reused": ["generate_prompt.instantiate_task",
                       "utils/eval SimEnv + Checker + Eval",
                       "configs/layouts/*.yaml", "VIKI-Assets objects"],
        },
        # §7-2 realization balance
        "realization_balance": {
            "ok": all(v["ok"] for v in balance.values()),
            "worst_max_share": max(v["max_share"] for v in balance.values()),
            "per_cell": balance,
        },
        # §7-3 vocab cross coverage
        "vocab_coverage": _vocab_coverage(instances),
        # §7-4 no realization leak (instance-level paired check)
        "realization_leak_paired_check": _leak_audit(),
        # §7-5 axis orthogonality
        "axis_orthogonality": _orthogonality_audit(instances),
        # §7-6 dual-validator agreement (SimEnv gate passed by construction)
        "dual_validator": {
            "simenv_pass_rate": 1.0,
            "phi_checked": len(phi_checked),
            "phi_ok": phi_ok,
            "phi_agreement": round(phi_ok / len(phi_checked), 4)
            if phi_checked else None,
            "gate": 0.995,
            "ok": (phi_ok / len(phi_checked) >= 0.995) if phi_checked else None,
            "phi_failures": Counter(
                i["meta"]["phi_code"] for i in phi_checked
                if not i["meta"]["phi_ok"]),
        },
        # §7-7 / §7-8 pending (P7.5)
        "strong_baseline_prerun": "pending:P7.5",
        "eval_fairness": "pending:P7.5",
        # §7-9 family/operator bridges + lookup blacklist
        **bridge,
        # §7-10 S4 counterfactual style/semantics boundary
        "s4_counterfactual_pairing": _s4_pair_audit(instances),
        # §6.3-4 instruction↔goal recoverability (added 2026-08-03, §7.6)
        "goal_text_recoverability": goal_text_recoverability(instances),
    }
    report["dual_validator"]["phi_failures"] = dict(
        report["dual_validator"]["phi_failures"])
    return report


# ---------------------------------------------------------------------------
# 5. empty-plan-aware scoring wrapper (benchmark.md §6.2)
# ---------------------------------------------------------------------------

def eval_single_locus(pred_steps, inst, rng=None):
    """Score a predicted plan against a LOCUS instance.

    ``pred_steps``: list of {robot: [Op, arg, ...]} steps (may be empty).
    Success = official SimEnv replay of the prediction (feasibility +
    temporal + goal constraints on the instance's realized initial state).
    The empty prediction is a first-class input: it succeeds iff the
    realized initial state already satisfies every goal (all-present
    realization) — and any prediction that relocates a pre-satisfied item
    away from its destination fails the goal check (over-grasp).
    """
    rng = rng or random.Random(inst["meta"]["instantiate_seed"]
                               if "meta" in inst else 0)
    pred_steps = pred_steps or []
    ok, err = simenv_replay(inst, steps=pred_steps, rng=rng)
    return {
        "success": bool(ok),
        "error": err,
        "empty_gt": not inst["ground_truth"],
        "empty_pred": not pred_steps,
    }


if __name__ == "__main__":
    instances, stats = generate_validated_batch(n_per_cell=4, seed=0)
    report = build_audit_report(instances, stats)
    out = os.path.join(_ROBOFACTORY, "script_general",
                       "locus_audit_report_pilot.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    dv = report["dual_validator"]
    print(f'{report["n_instances"]} instances | gen stats: {stats} | '
          f'phi agreement: {dv["phi_agreement"]} (gate {dv["gate"]}) | '
          f'balance ok: {report["realization_balance"]["ok"]} | '
          f'orthogonality ok: {report["axis_orthogonality"]["ok"]}')
    print(f"report -> {out}")
