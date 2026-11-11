#!/usr/bin/env python
"""Audit P1 planning_tool reconstruction on VIKI-L2 train GT.

The audit reverse-maps GT time_steps into OperationStep DAGs, erases selected
deterministic details to form an intent sketch, then checks whether
planning_tool.schedule + verify can recover a feasible plan.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.planning_tool import schedule, verify  # noqa: E402
from nsdl.schema import OperationStep  # noqa: E402
from scripts.build_slm_training_data import (  # noqa: E402
    build_scene_facts_from_gt,
    extract_sample_data,
    reverse_map_time_steps,
)


MECHANICAL_OPS = {
    "carry_to",
    "handoff",
    "load_carrier",
    "push_carrier",
    "unload_carrier",
    "close",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/viki/VIKI-L2/train.parquet")
    parser.add_argument("--output", default="tmp/p1_planning_tool_audit.json")
    parser.add_argument("--variant", choices=["deps_only", "intent", "intent_no_open"], default="intent")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--failures", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.limit > 0:
        df = df.head(args.limit)

    counters: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        counters["total"] += 1
        try:
            sample_data = extract_sample_data(row)
            task_name = sample_data.get("task_name") or "unknown"
            scene_facts = build_scene_facts_from_gt(sample_data, idx=int(idx))
            ctx = build_affordance_context(scene_facts)
            gt_ops = reverse_map_time_steps(sample_data)
            sketch = make_sketch(gt_ops, args.variant)
            result = schedule(sketch, assign={}, ctx=ctx, scene_facts=scene_facts)
            if result.ok:
                reason = verify(result.ops, ctx, scene_facts)
                if reason.ok:
                    counters["ok"] += 1
                    by_task[task_name]["ok"] += 1
                    continue
                code = reason.code or "verify_failed"
                message = reason.message
            else:
                code = result.reason.code if result.reason else "schedule_failed"
                message = result.reason.message if result.reason else ""

            counters[f"fail:{code}"] += 1
            by_task[task_name][f"fail:{code}"] += 1
            if len(failures) < args.failures:
                failures.append({
                    "row_index": int(idx),
                    "task_id": sample_data.get("task_id"),
                    "task_name": task_name,
                    "code": code,
                    "message": message,
                    "gt_ops": [_op_to_dict(op) for op in gt_ops],
                    "sketch": [_op_to_dict(op) for op in sketch],
                    "scheduled_ops": [_op_to_dict(op) for op in (result.ops if result else [])],
                    "inserted": list(result.inserted if result else []),
                })
        except Exception as exc:  # noqa: BLE001 - audit should continue.
            task_name = "unknown"
            try:
                task_name = extract_sample_data(row).get("task_name") or "unknown"
            except Exception:
                pass
            counters[f"exception:{exc.__class__.__name__}"] += 1
            by_task[task_name][f"exception:{exc.__class__.__name__}"] += 1
            if len(failures) < args.failures:
                failures.append({
                    "row_index": int(idx),
                    "task_name": task_name,
                    "code": f"exception:{exc.__class__.__name__}",
                    "message": str(exc),
                })

    total = counters["total"]
    ok = counters["ok"]
    summary = {
        "parquet": args.parquet,
        "variant": args.variant,
        "total": total,
        "ok": ok,
        "pass_rate": ok / total if total else 0.0,
        "counters": dict(counters),
        "by_task": {
            task: dict(counter)
            for task, counter in sorted(by_task.items())
        },
        "failures": failures,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({
        "variant": args.variant,
        "total": total,
        "ok": ok,
        "pass_rate": round(summary["pass_rate"], 6),
        "top_failures": counters.most_common(12),
        "output": str(output),
    }, indent=2, ensure_ascii=False))
    return 0 if ok == total else 1


def make_sketch(ops: list[OperationStep], variant: str) -> list[OperationStep]:
    """Compress GT ops into a model-facing sketch.

    Variants:
      deps_only        -- keep all op types, strip deps (legacy audit variant).
      intent           -- drop MECHANICAL_OPS, rewrite unload_carrier->deposit,
                          strip deps (VIKI v1..v3.3 build contract; unchanged).
      intent_no_open   -- intent, additionally dropping open.
      ops              -- keep all op types, strip deps.
      ops_chain        -- keep all op types + per-robot sequential chain deps
                          (LOCUS build contract, 2026-08-31): mechanical ops
                          like carry_to carry intent (two-level destinations)
                          that schedule() provably cannot re-infer, and the
                          intra-robot order chain carries temporal intent
                          (e.g. activate-before-second-fetch). Cross-robot
                          deps are still left to Phi. GT-reconstruction on the
                          LOCUS eval splits is exact under this contract
                          (560/560; see tmp/phi_fix_probe).
    """
    kept_ids: set[str] = set()
    sketch: list[OperationStep] = []
    for op in ops:
        if variant in {"intent", "intent_no_open"} and op.op_type == "unload_carrier":
            kept_ids.add(op.op_id)
            sketch.append(OperationStep(
                op_id=op.op_id,
                op_type="deposit",
                entity=op.entity,
                dest=op.dest,
                deps=[],
                robot=op.robot,
                params={},
                gaps=["deps", "precond"],
            ))
            continue
        if variant in {"intent", "intent_no_open"} and op.op_type in MECHANICAL_OPS:
            continue
        if variant == "intent_no_open" and op.op_type == "open":
            continue
        kept_ids.add(op.op_id)
        sketch.append(OperationStep(
            op_id=op.op_id,
            op_type=op.op_type,
            entity=op.entity,
            dest=op.dest,
            deps=[],
            robot=op.robot,
            params=dict(op.params or {}),
            gaps=["deps", "precond"] if op.op_type == "deposit" else ["deps"],
        ))

    # Keep only dependencies still expressible in this variant for non-deps-only
    # structural ops that should remain explicitly ordered, e.g. process/activate.
    if variant == "deps_only":
        for source, target in zip(ops, sketch):
            target.deps = []
            target.gaps = ["deps"]
    else:
        for op in sketch:
            op.deps = [dep for dep in op.deps if dep in kept_ids]
    if variant == "ops_chain":
        last_by_robot: dict[str | None, str] = {}
        for op in sketch:
            if op.robot in last_by_robot:
                op.deps = [last_by_robot[op.robot]]
            last_by_robot[op.robot] = op.op_id
    return sketch


def _op_to_dict(op: OperationStep) -> dict[str, Any]:
    if is_dataclass(op):
        return asdict(op)
    return dict(op)


if __name__ == "__main__":
    raise SystemExit(main())
