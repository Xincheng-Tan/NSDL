#!/usr/bin/env python3
"""Offline re-score of frozen forced-Phi buckets with Phi's own schedule as the answer.

Why (plan.md A0k, facts:E24c, 2026-09-04): the forced arm
(``--force-phi-reschedule``) used to replace the model's operations with
``schedule_result.ops`` and let the eval driver re-expand that op-DAG through
``dispatch_graph``. The op-DAG carries only the custody/precondition deps Phi
derives; the same-robot end-effector serialization lives in Phi's list
scheduler and is only visible in ``schedule_result.schedule``. The re-expansion
therefore is NOT Phi's plan: on the 7B test_id column 10/180 rows whose Phi
schedule was byte-equal to GT became two-Grasps-before-Place failures.

Every forced record already stores ``schedule_result.schedule`` in full, so the
fix (eval driver ``--force-phi-final-plan phi_schedule``, now the default) can
be replayed on frozen buckets with zero GPU: for each row whose forced call was
applied and rescheduled, the final plan becomes Phi's schedule and the row is
re-validated and re-scored. Rows the forcing never touched keep their stored
outcome untouched.

Self-check, not assumption: on rows where the stored plan already equals Phi's
schedule (the re-expansion happened to coincide) the re-scored values must equal
the stored ones row by row; a divergence is reported as ``parity_broken`` and
means the reconstruction is wrong, not the readings.

Non-forced buckets (plan.md 序 26 / A0p, 2026-09-06): the main configuration
now submits Phi's own schedule as the final plan, and every eval trace stores
the full return of every ``schedule`` call. So for a row of an ordinary bucket
the final plan becomes the ``schedule`` of the LAST ACCEPTED (``ok``) call;
rows whose sketch Phi never accepted (or that never called schedule) keep the
stored outcome = the honest fallback to the self-produced plan, counted as
``fallback_*``. An accepted call whose schedule is empty is a real Phi answer
(plan = []). The same parity self-check applies: rows whose stored plan already
equals Phi's schedule must re-score identically.

Original result JSONs are never modified. Output = a side-car report per bucket
under results/lnsdl/audits/rescore_forcedphi/ and, with --write-derived-bucket,
a sibling bucket ``<bucket>__phisched`` holding updated copies (plan /
validation / benchmark_scores / trace.forced_phi_reschedule.final_plan_source)
so the existing paired reports (report_p61_forced_phi.py) run unchanged on it.

Usage:
  python scripts/rescore_forced_phi_bucket_offline.py \
      --buckets 'results/lnsdl/v3_3/locus/*forcedphi*' --write-derived-bucket
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.benchmark_scoring import (  # noqa: E402
    compute_locus_l2_acc,
    compute_viki_l2_score,
)
from nsdl.schema import MicroStep, OperationStep  # noqa: E402
from nsdl.serializer import to_l2_answer  # noqa: E402
from nsdl.tool_runtime import forced_phi_micro_steps  # noqa: E402
from nsdl.validator import validate_l2_plan  # noqa: E402

# --rescheduler v2 (plan.md A0p 附注 / facts M5c): instead of taking the stored
# schedule, re-run Phi's `schedule` on the stored last-accepted ops + assign with
# LNSDL_PHI_SCHEDULER=v2. The sample/context is rebuilt exactly as the eval
# driver does (rule parser, refiner "none", VIKI_L2 profile); with v1 this
# reproduces the stored schedule byte-for-byte (self-checked: 3017/3017 VIKI,
# 527/527 LOCUS on 2026-09-06), which is what licenses the offline path.
_RESCHED: dict[str, Any] = {"mode": None, "samples": {}, "parser": None, "refiner": None}


def _resched_setup(mode: str, level: str) -> None:
    from nsdl.prompt_adapter import build_sample_stem, load_benchmark_samples
    from nsdl.scene_facts_refiner import build_scene_facts_refiner
    from nsdl.scene_parser import build_scene_parser
    _RESCHED["mode"] = mode
    _RESCHED["parser"] = build_scene_parser("rule")
    _RESCHED["refiner"] = build_scene_facts_refiner("none")
    level_name = "LOCUS" if level == "locus" else "VIKI-L2"
    for split in LEVELS[level]["splits"]:
        for sample in load_benchmark_samples(level=level_name, split=split):
            _RESCHED["samples"][(split, build_sample_stem(sample))] = sample


def _rescheduled_steps(d: dict, split: str, stem: str) -> tuple[list[MicroStep] | None, str]:
    """Re-run Phi on the stored last-accepted ops. Returns (steps, tag)."""
    from nsdl import planning_tool as PT
    from nsdl.aig_builder import build_affordance_context
    from nsdl.profiles import VIKI_L2_PROFILE, use_benchmark_profile
    calls = _schedule_calls_with_args(d)
    oks = [(c, r) for c, r in calls if r.get("ok")]
    if not oks:
        return None, "fallback_all_rejected" if calls else "fallback_never_called"
    args, res = oks[-1]
    sample = _RESCHED["samples"].get((split, stem))
    if sample is None:
        return None, "resched_sample_missing"
    # The stored ops carry the auto_open Phi inserted at record time (and deps
    # onto it). ALWAYS strip both so Phi re-inserts them under the placement
    # rule this derivation runs with (plan 序 30, 2026-09-11: the rule is no
    # longer inferred from "≠ default", which silently stopped stripping once
    # the default itself moved to late_freehand). Under the same rule as the
    # record this is byte-identical to not stripping (verified on the 7B main
    # row, 560/560 rows); model-written `open` ops keep their ids and are
    # untouched.
    def _is_auto(x: object) -> bool:
        return str(x).startswith("auto_open")
    ops = [OperationStep(op_id=str(o.get("op_id")), op_type=str(o.get("op_type")), entity=o.get("entity"),
                         dest=o.get("dest"), deps=[x for x in (o.get("deps") or []) if not _is_auto(x)], robot=o.get("robot"),
                         params=dict(o.get("params") or {}), gaps=list(o.get("gaps") or []))
           for o in (res.get("ops") or []) if isinstance(o, dict) and not _is_auto(o.get("op_id"))]
    assign = args.get("assign") if isinstance(args, dict) else None
    prev = os.environ.get(PT.PHI_SCHEDULER_ENV)
    os.environ[PT.PHI_SCHEDULER_ENV] = _RESCHED["mode"]
    try:
        with use_benchmark_profile(VIKI_L2_PROFILE):
            sf = _RESCHED["refiner"].refine(sample, _RESCHED["parser"].parse(sample), yolo_detections=None)
            out = PT.schedule(ops, assign, build_affordance_context(sf), sf)
    except Exception:  # noqa: BLE001
        return None, "resched_exception"
    finally:
        if prev is None:
            os.environ.pop(PT.PHI_SCHEDULER_ENV, None)
        else:
            os.environ[PT.PHI_SCHEDULER_ENV] = prev
    if not out.ok:
        return None, "resched_rejected"
    return list(out.schedule), f"resched_{_RESCHED['mode']}"

DERIVED_SUFFIX = "__phisched"
LOCUS_SPLITS = ("test_id", "ood_s", "ood_m", "ood_x")
VIKI_SPLITS = ("test", "val")
LEVELS = {
    "locus": {"splits": LOCUS_SPLITS, "key": "locus_l2_acc"},
    "viki": {"splits": VIKI_SPLITS, "key": "viki_l2_plan_score"},
}


def _norm_plan(plan: Any) -> list[dict[str, Any]] | None:
    if not isinstance(plan, list):
        return None
    out = []
    for i, step in enumerate(plan):
        if not isinstance(step, dict) or not isinstance(step.get("actions"), dict):
            return None
        out.append({
            "step": i + 1,
            "actions": {str(k): [str(x) for x in v] for k, v in step["actions"].items() if v},
        })
    return out


def _forced_record(d: dict) -> dict | None:
    trace = ((d.get("agentic_eval") or {}).get("trace") or {})
    rec = trace.get("forced_phi_reschedule")
    return rec if isinstance(rec, dict) else None


def _schedule_calls_with_args(d: dict) -> list[tuple[dict, dict]]:
    """[(call_args, result)] of every `schedule` call, for both artifact shapes:
    the agentic eval trace (tool_names/tool_results/executed_tool_calls) and the
    constant-offload harness (序 21 arms: constant_offload.trace.tool_events)."""
    trace = ((d.get("agentic_eval") or {}).get("trace") or {})
    if trace.get("tool_names"):
        calls = trace.get("executed_tool_calls") or []
        out = []
        for i, (n, r) in enumerate(zip(trace.get("tool_names") or [], trace.get("tool_results") or [])):
            if str(n).strip().lower() == "schedule" and isinstance(r, dict):
                c = calls[i] if i < len(calls) and isinstance(calls[i], dict) else {}
                out.append((c.get("args") or {}, r))
        return out
    co_trace = ((d.get("constant_offload") or {}).get("trace") or {})
    return [((ev.get("args") or {}), ev.get("result"))
            for ev in (co_trace.get("tool_events") or [])
            if isinstance(ev, dict) and str(ev.get("name") or "").lower() == "schedule" and isinstance(ev.get("result"), dict)]


def _schedule_calls(d: dict) -> list[dict]:
    return [r for _, r in _schedule_calls_with_args(d)]


def last_ok_schedule_steps(d: dict) -> tuple[list[MicroStep] | None, str]:
    """Phi's schedule from the last accepted `schedule` call of an ordinary
    (non-forced) episode, plus the row's source tag:
      last_ok / earlier_ok_last_rejected  -> steps (may be [] = Phi said nothing to do)
      fallback_all_rejected / fallback_never_called / fallback_no_schedule_field -> None
    """
    calls = _schedule_calls(d)
    if not calls:
        return None, "fallback_never_called"
    oks = [r for r in calls if r.get("ok")]
    if not oks:
        return None, "fallback_all_rejected"
    last = oks[-1]
    raw = last.get("schedule")
    if not isinstance(raw, list):
        return None, "fallback_no_schedule_field"
    steps: list[MicroStep] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("actions"), dict):
            return None, "fallback_no_schedule_field"
        steps.append(MicroStep(
            step=int(item.get("step", len(steps) + 1)),
            actions={str(k): [str(x) for x in v] for k, v in item["actions"].items() if v},
        ))
    return steps, ("last_ok" if last is calls[-1] else "earlier_ok_last_rejected")


def rescore_row(d: dict, score_key: str, split: str | None = None, stem: str | None = None) -> dict[str, Any] | None:
    """Return the re-scored outcome for a row whose final plan becomes Phi's
    schedule -- from the forced record when the bucket is a forced arm, else
    from the last accepted `schedule` call -- or None to keep the stored row."""
    rec = _forced_record(d)
    steps = forced_phi_micro_steps(rec)
    source = "forced_record"
    if steps is None:
        if rec and rec.get("applied"):
            return None          # forced arm, forcing fell back: keep stored (already the fallback)
        steps, source = last_ok_schedule_steps(d)
        if steps is None:
            return None
    if _RESCHED["mode"] and split is not None and stem is not None:
        # forced records store the same ops/assign shape only in `schedule_result`;
        # for both kinds the trace's last accepted call is the ops Phi produced.
        re_steps, tag = _rescheduled_steps(d, split, stem)
        if re_steps is None:
            return None if tag.startswith("fallback") else {"__skip__": tag}
        steps, source = re_steps, tag
    plan = to_l2_answer(steps)
    gt = d.get("ground_truth") if isinstance(d.get("ground_truth"), dict) else {}
    validation = validate_l2_plan(plan, gt)
    return {
        "plan": plan,
        "source": source,
        "validation": {
            "success": bool(validation.success),
            "error_desc": validation.error_desc,
            "eval_commands": validation.eval_commands,
        },
        "benchmark_scores": {
            **(d.get("benchmark_scores") or {}),
            "locus_l2_acc": compute_locus_l2_acc(plan, gt),
            "viki_l2_plan_score": compute_viki_l2_score(plan, gt),
        },
    }


def rescore_bucket(bucket: str, spec: dict, *, write_derived: bool) -> dict[str, Any]:
    splits, score_key = spec["splits"], spec["key"]
    derived_root = bucket.rstrip("/") + DERIVED_SUFFIX
    if write_derived:
        if os.path.isdir(derived_root):
            shutil.rmtree(derived_root)
        os.makedirs(derived_root)
    counters: collections.Counter[str] = collections.Counter()
    agg_old: dict[str, list[float]] = collections.defaultdict(list)
    agg_new: dict[str, list[float]] = collections.defaultdict(list)
    err_old: dict[str, int] = collections.Counter()
    err_new: dict[str, int] = collections.Counter()
    changed: list[dict[str, Any]] = []
    for split in splits:
        files = sorted(glob.glob(os.path.join(bucket, split, "*.json")))
        if write_derived and files:
            os.makedirs(os.path.join(derived_root, split), exist_ok=True)
        for fp in files:
            d = json.load(open(fp))
            rel = os.path.relpath(fp, bucket)
            stored_status = d.get("result_status")
            stored = (d.get("benchmark_scores") or {}).get(score_key)
            stored_score = float(stored) if stored is not None else 0.0
            agg_old[split].append(stored_score)
            err_old[split] += int(stored_status == "error")

            new = rescore_row(d, score_key, split, os.path.basename(fp)[:-5])
            if isinstance(new, dict) and "__skip__" in new:
                counters[new["__skip__"]] += 1
                new = None
            if new is None:
                counters["untouched_kept_stored"] += 1
                rec0 = _forced_record(d)
                if rec0 and rec0.get("applied"):
                    counters["forced_fallback_kept_stored"] += 1
                else:
                    _, why = last_ok_schedule_steps(d)
                    counters[why] += 1
                agg_new[split].append(stored_score)
                err_new[split] += int(stored_status == "error")
                if write_derived:
                    shutil.copyfile(fp, os.path.join(derived_root, rel))
                continue

            new_score = float(new["benchmark_scores"][score_key] or 0.0)
            # A row that reached the forced call has a final response, so its
            # status is "ok" (validation failures are acc=0, not err) -- same
            # semantics as the live driver.
            agg_new[split].append(new_score)
            rec = _forced_record(d) or {}
            stored_plan = _norm_plan(d.get("plan"))
            same_plan = stored_plan is not None and stored_plan == _norm_plan(new["plan"])
            same_score = abs(new_score - stored_score) < 1e-9 and stored_status == "ok"
            counters[f"source:{new['source']}"] += 1
            if same_plan and stored_status == "ok":
                counters["reexpand_equals_phi_schedule"] += 1
                if not same_score:
                    counters["parity_broken"] += 1
                    changed.append({"file": rel, "kind": "parity_broken",
                                    "stored": stored, "rescored": new_score})
            elif same_plan:
                # an `error` row (runtime fail-closed on the model's own [PLAN]) whose
                # Phi schedule equals the stored (usually empty) plan: the row leaves
                # the err channel and is scored on Phi's answer -- a real change, not
                # a parity question.
                counters["error_row_rescored_same_plan"] += 1
                if not same_score:
                    counters["plan_replaced_score_changed"] += 1
                    changed.append({"file": rel, "kind": "score_changed",
                                    "stored": {"status": stored_status, score_key: stored},
                                    "rescored": {score_key: new_score,
                                                 "validation_error": new["validation"]["error_desc"]},
                                    "source": new["source"]})
            else:
                counters["plan_replaced"] += 1
                if not same_score:
                    counters["plan_replaced_score_changed"] += 1
                    changed.append({
                        "file": rel, "kind": "score_changed",
                        "stored": {"status": stored_status, score_key: stored},
                        "rescored": {score_key: new_score,
                                     "validation_error": new["validation"]["error_desc"]},
                        "source": new["source"] if new["source"] != "forced_record" else rec.get("source"),
                    })
            if write_derived:
                out = dict(d)
                out["plan"] = new["plan"]
                out["validation"] = new["validation"]
                out["benchmark_scores"] = new["benchmark_scores"]
                out["result_status"] = "ok"
                ae = dict(out.get("agentic_eval") or {})
                ae["force_phi_final_plan"] = "phi_schedule"
                ae["derived_from"] = {
                    "bucket": bucket, "file": rel,
                    "tool": "scripts/rescore_forced_phi_bucket_offline.py",
                    "note": ("A0k/facts:E24c offline replay: final plan = Phi schedule"
                             if new["source"] == "forced_record" else
                             "A0p/plan.md 序 26 offline replay: final plan = last accepted Phi schedule"),
                }
                ae["phisched_source"] = new["source"]
                ae["stored_result_status"] = stored_status
                trace = dict(ae.get("trace") or {})
                if new["source"] == "forced_record":
                    rec2 = dict(rec)
                    rec2["final_plan_source"] = "phi_schedule"
                    rec2["reexpanded_micro_steps"] = stored_plan
                    rec2["reexpand_equals_phi_schedule"] = same_plan
                    trace["forced_phi_reschedule"] = rec2
                else:
                    trace["phisched_offline"] = {
                        "source": new["source"],
                        "model_plan": stored_plan,
                        "model_plan_equals_phi_schedule": same_plan,
                        "stored_result_status": stored_status,
                    }
                ae["trace"] = trace
                if stored_status == "error":
                    out.pop("error", None)
                out["agentic_eval"] = ae
                with open(os.path.join(derived_root, rel), "w") as fh:
                    json.dump(out, fh, ensure_ascii=False, indent=2)
    per_split = {
        split: {
            "n": len(agg_old[split]),
            f"{score_key}_stored": round(sum(agg_old[split]) / len(agg_old[split]), 4),
            f"{score_key}_rescored": round(sum(agg_new[split]) / len(agg_new[split]), 4),
            "err_stored": err_old[split],
            "err_rescored": err_new[split],
        }
        for split in splits
        if agg_old[split]
    }
    return {
        "bucket": bucket,
        "derived_bucket": derived_root if write_derived else None,
        "score_key": score_key,
        "rescheduler": _RESCHED.get("mode"),
        "phi_open_placement": __import__("nsdl.planning_tool", fromlist=["x"]).phi_open_placement_mode(),
        "counters": dict(counters),
        "per_split": per_split,
        "changed_rows": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", required=True, help="glob of forced-Phi bucket dirs")
    parser.add_argument("--out-dir", default="results/lnsdl/audits/rescore_forcedphi")
    parser.add_argument("--level", choices=sorted(LEVELS), default="locus")
    parser.add_argument("--write-derived-bucket", action="store_true",
                        help="write <bucket>__phisched with updated row copies")
    parser.add_argument("--rescheduler", choices=["v1", "v2"], default=None,
                        help="re-run Phi's schedule offline on the stored ops with this list scheduler "
                             "(v1 must reproduce the stored schedule; v2 = dispatcher visibility rules)")
    parser.add_argument("--derived-suffix", default=None,
                        help="suffix of the derived bucket (default __phisched, or __phisched_<rescheduler>)")
    args = parser.parse_args()
    spec = LEVELS[args.level]
    if args.rescheduler:
        _resched_setup(args.rescheduler, args.level)
    global DERIVED_SUFFIX
    DERIVED_SUFFIX = args.derived_suffix or ("__phisched" if not args.rescheduler else f"__phisched_{args.rescheduler}")
    if args.rescheduler and not args.derived_suffix:
        from nsdl import planning_tool as _PT
        # Suffix follows the RULE, not "rule ≠ default": `__phisched_v2` is the
        # early-placement caliber and `__phisched_v2_openlate` the
        # late_freehand caliber, whatever the code default is at the time.
        if _PT.phi_open_placement_mode() == "late_freehand":
            DERIVED_SUFFIX += "_openlate"
    buckets = sorted(p for p in glob.glob(args.buckets)
                     if os.path.isdir(p) and "__phisched" not in os.path.basename(p))
    if not buckets:
        print(f"no buckets match {args.buckets}", file=sys.stderr)
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    rc = 0
    for bucket in buckets:
        report = rescore_bucket(bucket, spec, write_derived=args.write_derived_bucket)
        out = os.path.join(args.out_dir, os.path.basename(bucket.rstrip("/")) + ".json")
        with open(out, "w") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        c = report["counters"]
        print(f"[{os.path.basename(bucket)}] {json.dumps(c)}")
        for split, row in report["per_split"].items():
            print(f"   {split:8s} n={row['n']:4d} acc {row[spec['key'] + '_stored']:.4f} -> "
                  f"{row[spec['key'] + '_rescored']:.4f}  err {row['err_stored']} -> {row['err_rescored']}")
        if c.get("parity_broken"):
            print("   !! parity_broken -- reconstruction is wrong, readings not usable", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
