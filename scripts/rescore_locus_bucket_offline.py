#!/usr/bin/env python3
"""Offline re-score of frozen LOCUS eval buckets under the current Phi code.

Why this exists (2026-08-31, debug_sft.md section 4.5.1): the two-level
destination fix changed how a stored final [PLAN] expands into micro steps for
two op patterns (deposit dep-linked to a carry_to with a different dest;
deposit sourced from local_acquire). `locus_l2_acc` scores the EXPANSION, so
frozen buckets scored pre-fix and new buckets scored post-fix are two rulers.
Generation is untouched -- the model's final [PLAN] is its answer -- so the
ruler can be re-applied offline, no GPU: this script re-runs the deterministic
post-loop chain of eval_lnsdl_agentic_viki_l2.py (operation_validation ->
convert_operations_to_graph_plan -> dispatch_graph -> emit -> scores) on each
stored trace's final plan.

Faithfulness is self-checked, not assumed: on rows the fix provably cannot
touch (neither pattern present) the re-scored values must equal the stored
ones row by row; any clean-row divergence is reported as `parity_broken` and
means the reconstruction is wrong, not the readings.

The original result JSONs are never modified (they are run evidence). Output
is a side-car report per bucket under results/lnsdl/audits/rescore_newphi/.

Usage:
  python scripts/rescore_locus_bucket_offline.py \
      --buckets 'results/lnsdl/v3_3/locus/locus_qwen25_7b_v33_seed42_a1_full34_cap524288'
  python scripts/rescore_locus_bucket_offline.py --buckets 'results/lnsdl/v3_3/locus/*'
  python scripts/rescore_locus_bucket_offline.py --level viki \
      --buckets 'results/lnsdl/v3_3/viki_l2/*'

VIKI buckets are in scope even though the GT-side audit shows the fix never
fires on VIKI training rows (7196/7196): that audit covers GT-DERIVED plans,
while a bucket holds MODEL-EMITTED ones, and a model can emit the
local_acquire -> deposit pattern the GT never contains (up to 42/3018 rows
per bucket; see tmp/phi_fix_probe/exposure_viki.log).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.benchmark_scoring import (  # noqa: E402
    compute_locus_l2_acc,
    compute_viki_l2_score,
)
from nsdl.dispatcher import dispatch_graph  # noqa: E402
from nsdl.macro_graph import _assign_executors_to_graph  # noqa: E402
from nsdl.profiles import VIKI_L2_PROFILE, use_benchmark_profile  # noqa: E402
from nsdl.prompt_adapter import LOCUS_LEVEL, load_benchmark_samples  # noqa: E402
from nsdl.scene_facts_refiner import build_scene_facts_refiner  # noqa: E402
from nsdl.scene_parser import build_scene_parser  # noqa: E402
from nsdl.schema import DispatcherPayload  # noqa: E402
from nsdl.serializer import to_l2_answer  # noqa: E402
from nsdl.slm_interface import (  # noqa: E402
    convert_operations_to_graph_plan,
    parse_operation_steps,
    validate_operations,
)

LOCUS_SPLITS = ("test_id", "ood_s", "ood_m", "ood_x")
VIKI_SPLITS = ("test", "val")
# Which stored column the parity gate and the aggregates are read on. VIKI's
# reportable number is the official plan score; LOCUS reports its null-plan
# aware variant (see memory: missing-column-is-not-zero).
LEVELS = {
    "locus": {"level": LOCUS_LEVEL, "splits": LOCUS_SPLITS, "key": "locus_l2_acc"},
    "viki": {"level": "VIKI-L2", "splits": VIKI_SPLITS, "key": "viki_l2_plan_score"},
}


def load_split_samples(split: str, level: str = LOCUS_LEVEL) -> dict[int, Any]:
    """sample_index -> BenchmarkSample via the eval driver's own loader.

    Must be `load_benchmark_samples(level=LOCUS)` and nothing else: the eval
    reads `data/lnsdl_bench/ramp63k/<split>` (prompt_adapter.LOCUS_ROOT). A
    hand-rolled reader pointed at the older pilot/ tree loads different
    realizations with the same filenames -- caught by this script's clean-row
    parity gate on its first run (260/560 rows diverged).
    """
    return {s.sample_index: s for s in load_benchmark_samples(level=level, split=split)}


def pattern_exposed(ops: list[dict]) -> str | None:
    """Same fires-condition as the scaffold fix: which rows CAN change."""
    by_id = {o.get("op_id"): o for o in ops if isinstance(o, dict)}
    for i, o in enumerate(ops):
        if not isinstance(o, dict) or o.get("op_type") != "deposit":
            continue
        preds = [by_id.get(x) for x in (o.get("deps") or [])]
        for prev in reversed(ops[:i]):
            if (
                isinstance(prev, dict)
                and prev.get("entity") == o.get("entity")
                and prev.get("robot") == o.get("robot")
                and prev.get("op_type")
                in ("acquire", "local_acquire", "carry_to", "handoff", "unload_carrier")
            ):
                preds.append(prev)
                break
        for p in preds:
            if not isinstance(p, dict):
                continue
            if p.get("op_type") == "local_acquire":
                return "B"
            if p.get("op_type") == "carry_to" and (
                (p.get("dest") or "").strip().lower()
                != (o.get("dest") or "").strip().lower()
            ):
                return "A"
    return None


def rescore_row(sample, plan_ops_raw: list[dict], goal: Any, scene_parser, scene_refiner) -> dict[str, Any]:
    """Mirror of the eval driver's post-loop chain (stages named identically)."""
    stage = "parse_ops"
    try:
        with use_benchmark_profile(VIKI_L2_PROFILE):
            ops = parse_operation_steps(plan_ops_raw)
            stage = "scene_parser"
            initial = scene_parser.parse(sample)
            stage = "scene_facts_refiner"
            facts = scene_refiner.refine(sample, initial, yolo_detections=None)
            ctx = build_affordance_context(facts)

            stage = "operation_validation"
            goal_active = None
            if isinstance(goal, dict) and isinstance(goal.get("active"), list):
                goal_active = [str(x) for x in goal["active"] if str(x).strip()]
            report = validate_operations(
                ops,
                ctx,
                facts,
                expected_non_empty_plan=bool(goal_active) if goal_active is not None else None,
                goal_active=goal_active,
            )
            if not report.ok:
                raise RuntimeError("; ".join(report.messages) or "operation validation failed")

            stage = "graph_plan"
            graph_plan = convert_operations_to_graph_plan(ops, ctx, facts)
            graph_plan.build_mode = "lnsdl_agentic_tool_loop"
            graph_plan.family_shortcut_used = False
            graph_plan = _assign_executors_to_graph(graph_plan, facts)

            stage = "micro_planner"
            micro_steps = dispatch_graph(graph_plan)

            stage = "dispatcher_emitter"
            dispatcher_output = VIKI_L2_PROFILE.dispatcher_emitter.emit(
                DispatcherPayload(
                    sample=sample,
                    initial_scene_facts=initial,
                    scene_facts=facts,
                    graph_plan=graph_plan,
                    micro_steps=micro_steps,
                )
            )
            plan = dispatcher_output if dispatcher_output is not None else to_l2_answer(micro_steps)
    except Exception as exc:  # noqa: BLE001 -- err rows are a scoring outcome
        return {
            "result_status": "error",
            "error_stage": stage,
            "error": f"{exc.__class__.__name__}: {exc}",
            "locus_l2_acc": None,
            "viki_l2_plan_score": None,
        }
    gt = sample.ground_truth if isinstance(sample.ground_truth, dict) else {}
    return {
        "result_status": "ok",
        "locus_l2_acc": compute_locus_l2_acc(plan, gt),
        "viki_l2_plan_score": compute_viki_l2_score(plan, gt),
    }


def rescore_bucket(bucket: str, samples_by_split: dict[str, dict[int, Any]],
                   scene_parser, scene_refiner, spec: dict) -> dict[str, Any]:
    splits, score_key = spec["splits"], spec["key"]
    rows: list[dict[str, Any]] = []
    counters: collections.Counter[str] = collections.Counter()
    agg_old: dict[str, list[float]] = collections.defaultdict(list)
    agg_new: dict[str, list[float]] = collections.defaultdict(list)
    for split in splits:
        for fp in sorted(glob.glob(os.path.join(bucket, split, "*.json"))):
            d = json.load(open(fp))
            sample_index = d.get("sample", {}).get("sample_index")
            sample = samples_by_split.get(split, {}).get(sample_index)
            trace = (d.get("agentic_eval") or {}).get("trace") or {}
            fr = trace.get("final_response") or {}
            plan_ops = fr.get("plan")
            stored_status = d.get("result_status")
            stored = (d.get("benchmark_scores") or {}).get(score_key)
            stored_score = float(stored) if stored is not None else 0.0
            agg_old[split].append(stored_score)

            if sample is None or not isinstance(plan_ops, list):
                # No reconstructable answer (e.g. crash before a final
                # response): the stored outcome stands.
                counters["no_final_plan_kept_stored"] += 1
                agg_new[split].append(stored_score)
                continue

            exposed = pattern_exposed(plan_ops)
            new = rescore_row(sample, plan_ops, fr.get("goal"), scene_parser, scene_refiner)
            new_score = new[score_key] if new[score_key] is not None else 0.0
            agg_new[split].append(float(new_score))

            same_status = (new["result_status"] == "ok") == (stored_status == "ok")
            same_score = abs(float(new_score) - stored_score) < 1e-9
            if exposed is None:
                counters["clean_parity_ok" if (same_status and same_score) else "parity_broken"] += 1
                if not (same_status and same_score):
                    rows.append({
                        "file": os.path.relpath(fp, bucket), "kind": "parity_broken",
                        "stored": {"status": stored_status, score_key: stored},
                        "rescored": new,
                    })
            else:
                counters[f"pattern_{exposed}"] += 1
                if not (same_status and same_score):
                    counters[f"pattern_{exposed}_changed"] += 1
                    rows.append({
                        "file": os.path.relpath(fp, bucket), "kind": f"pattern_{exposed}_changed",
                        "stored": {"status": stored_status, score_key: stored},
                        "rescored": new,
                    })
    per_split = {
        split: {
            "n": len(agg_old[split]),
            f"{score_key}_stored": round(sum(agg_old[split]) / len(agg_old[split]), 4) if agg_old[split] else None,
            f"{score_key}_rescored": round(sum(agg_new[split]) / len(agg_new[split]), 4) if agg_new[split] else None,
        }
        for split in splits
        if agg_old[split]
    }
    return {
        "bucket": bucket,
        "score_key": score_key,
        "counters": dict(counters),
        "per_split": per_split,
        "changed_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", required=True, help="glob of bucket dirs")
    parser.add_argument("--out-dir", default="results/lnsdl/audits/rescore_newphi")
    parser.add_argument("--level", choices=sorted(LEVELS), default="locus")
    parser.add_argument("--scene-parser-backend", default="rule")
    parser.add_argument("--scene-facts-refiner-backend", default="none")
    args = parser.parse_args()

    scene_parser = build_scene_parser(args.scene_parser_backend)
    scene_refiner = build_scene_facts_refiner(args.scene_facts_refiner_backend)
    spec = LEVELS[args.level]
    samples_by_split = {s: load_split_samples(s, spec["level"]) for s in spec["splits"]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = sorted(b for b in glob.glob(args.buckets) if os.path.isdir(b))
    for bucket in buckets:
        if not any(os.path.isdir(os.path.join(bucket, s)) for s in spec["splits"]):
            continue
        report = rescore_bucket(bucket, samples_by_split, scene_parser, scene_refiner, spec)
        name = os.path.basename(bucket.rstrip("/"))
        (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        broken = report["counters"].get("parity_broken", 0)
        print(
            f"{name}: {report['counters']}"
            + ("  !! PARITY BROKEN -- do not trust this rescoring" if broken else "")
        )
        key = report["score_key"]
        for split, agg in report["per_split"].items():
            if agg[f"{key}_stored"] != agg[f"{key}_rescored"]:
                print(f"   {split}: {agg[f'{key}_stored']} -> {agg[f'{key}_rescored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
