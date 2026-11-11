"""P7.5 pilot driver (docs/benchmark.md §9): generate → render+masks → audit.

Runs the freeze-order pilot stage on the P7.4b 49-cell registry:

  1. ``build_split_plan`` / ``generate_split_instances`` — 20–30 instances per
     family-cell with all inline gates (SimEnv replay blocking, Φ recorded,
     realization min-count balance, paraphrase/layout holdouts);
  2. ``render_batch(export_masks=True)`` — real snapshots + renderer-GT label
     JSONs from the same frame (docs/YOLO.md §6.3), per-sample failures marked
     not fatal;
  3. audits — §6.3-3 decidability check (GT placement level, now with real
     ``style_idx``), §7 audit report (balance / orthogonality / SimEnv / Φ /
     bridge / S4 pairing), renderer-GT class-count aggregation against the
     §6.2 volume gate, and the transparent-cabinet human-labeling queue;
  4. ``pack_parquet`` per fully-rendered split (VIKI-L2 column parity).

Everything lands under ``--out`` (default ``data/lnsdl_bench/pilot``); the
aggregate goes to ``<out>/pilot_report.json``. Run under the NSDL env with the
render env exposed:

  LOCUS_RENDER_PYTHON=/path/to/envs/locus-render/bin/python \
  MS_ASSET_DIR=/path/to/maniskill_data \
  python script_general/locus_pilot.py

This is the PILOT (feasibility + gate rehearsal), not the v1.0 freeze: the
≥100-per-operator scale gate, detector fine-tune, strong-baseline preruns and
the §7.11 RL variance audit stay downstream (docs/benchmark.md §9 P7.5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from locus_render_pack import (  # noqa: E402
    build_split_plan,
    decidability_check,
    generate_split_instances,
    pack_parquet,
    render_batch,
    render_stem,
)
from locus_validate import build_audit_report  # noqa: E402

# §6.2 volume gate targets (docs/YOLO.md): ≥100 train boxes per NEW class.
# The pilot reports progress against them; hitting them is the放量 stage's job.
NEW_CLASSES = ("bottle", "cardboardbox", "cup", "drawer", "fork", "fridge",
               "kettle", "meat", "oven", "pizza", "pumpkin", "scissors",
               "spoon", "wine")


def aggregate_mask_summaries(split_dir):
    """Merge chunked mask_export_summary_*.json of one split."""
    merged = []
    for fn in sorted(os.listdir(split_dir)):
        if fn.startswith("mask_export_summary_") and fn.endswith(".json"):
            with open(os.path.join(split_dir, fn)) as f:
                merged.extend(json.load(f))
    counts = Counter()
    transparent_queue = []          # through CLEAR glass: §5.3-3 says LABEL
    blurred_queue = []              # through BLURRED glass: human decides
    solid_hidden = []               # opaque carcass: correctly box-free
    unresolved_hidden = []          # probe could not place the target
    occluded_dropped = []           # boxes killed by MIN_VISIBLE_FRAC
    failures = []
    slivers = 0
    for res in merged:
        if "error" in res:
            failures.append({"image": res["image"], "error": res["error"]})
            continue
        for cls, n in res["class_counts"].items():
            counts[cls] += n
        slivers += res.get("n_slivers", 0)
        if res.get("occluded_dropped"):
            occluded_dropped.append({"image": res["image"],
                                     "boxes": res["occluded_dropped"]})
        for key, queue in (("transparent_cabinet_unlabeled", transparent_queue),
                           ("glass_blurred_review", blurred_queue),
                           ("solid_hidden_ok", solid_hidden)):
            if res.get(key):
                queue.append({"image": res["image"], "entities": res[key]})
        unresolved = (res.get("hidden_target_probe") or {}).get("unresolved")
        if unresolved:
            unresolved_hidden.append({"image": res["image"],
                                      "entities": unresolved})
    return {
        "n_rendered": len(merged) - len(failures),
        "n_failed": len(failures),
        "failures": failures,
        "class_box_counts": dict(counts),
        "n_slivers": slivers,
        # Only the glass buckets are human-labeling work. `solid_hidden` is
        # reported so "no box" stays auditable rather than looking like a miss.
        "transparent_cabinet_queue": transparent_queue,
        "blurred_glass_queue": blurred_queue,
        "solid_hidden_ok": solid_hidden,
        "unresolved_hidden": unresolved_hidden,
        "occluded_dropped": occluded_dropped,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-test-id", type=int, default=6)
    parser.add_argument("--n-ood", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(_ROBOFACTORY), "data", "lnsdl_bench", "pilot"))
    parser.add_argument("--dry-run", action="store_true",
                        help="configs + manifest + audits only, no GPU render")
    parser.add_argument("--shared-rng", action="store_true",
                        help="walk one rng across the whole plan (the P7.5b "
                             "stream). Only for re-deriving that pilot: it "
                             "couples plan rows, so any count change "
                             "reshuffles every later row's configs.")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="restrict to these splits (default: all)")
    parser.add_argument("--no-stratify-masks", action="store_true",
                        help="draw the §7-3 audit masks i.i.d. instead of "
                             "round-robin on too-small plan rows. Only for "
                             "re-deriving pre-stratification configs: it "
                             "reopens the F9 vocabulary-coverage holes.")
    parser.add_argument("--wall-clear-scope",
                        help="file of `<split>_<index>` stems, one per line: "
                             "apply the wall-clearance pass (case (f)) ONLY "
                             "to these. Configs outside the scope reproduce "
                             "byte-identically and keep their existing "
                             "render. Default (absent) = apply everywhere. "
                             "Every stem must exist in the generated plan.")
    return parser


def _load_wall_clear_scope(path, splits):
    """Read the stem whitelist and verify it describes THIS plan.

    A stale scope is the failure mode worth catching: the stems were derived
    from an audit of an existing pack, so if the plan has since changed shape
    the same stem now names a different scene and the pass would be applied to
    the wrong configs (and, worse, silently skipped on the right ones).
    """
    with open(path) as f:
        stems = {ln.strip() for ln in f if ln.strip()}
    known = {render_stem(inst) for insts in splits.values() for inst in insts}
    unknown = sorted(stems - known)
    if unknown:
        raise SystemExit(
            f"[pilot] --wall-clear-scope lists {len(unknown)} stem(s) absent "
            f"from the generated plan (e.g. {unknown[:5]}); the scope was "
            f"derived from a different plan shape and must be re-derived")
    digest = hashlib.sha256(
        "\n".join(sorted(stems)).encode()).hexdigest()
    print(f"[pilot] wall-clear scope: {len(stems)} stems, sha256 "
          f"{digest[:16]}", flush=True)
    return stems, digest


def main(argv=None):
    args = build_parser().parse_args(argv)

    plan = build_split_plan(n_train=args.n_train, n_test_id=args.n_test_id,
                            n_ood=args.n_ood)
    print(f"[pilot] plan rows={len(plan)} "
          f"instances={sum(r[-1] for r in plan)}", flush=True)
    splits, stats = generate_split_instances(
        plan, seed=args.seed, per_row_rng=not args.shared_rng,
        stratify_small_vocab=not args.no_stratify_masks)
    print(f"[pilot] generated: { {k: len(v) for k, v in splits.items()} } "
          f"stats={stats}", flush=True)

    scope, scope_digest = (None, None)
    if args.wall_clear_scope:
        scope, scope_digest = _load_wall_clear_scope(
            args.wall_clear_scope, splits)

    # Match render_batch, which must abspath anyway (its workers run with a
    # different cwd), so report/parquet paths describe the same location.
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    report = {
        "config": {"n_train": args.n_train, "n_test_id": args.n_test_id,
                   "n_ood": args.n_ood, "seed": args.seed,
                   "dry_run": bool(args.dry_run),
                   # Stream derivation is part of the render identity: the same
                   # seed gives different configs under the two modes, so a
                   # report that omits it cannot be reproduced. Mask
                   # stratification is identity in the same way (it changed 108
                   # of the ramp63k configs), so it is recorded alongside.
                   "per_row_rng": not args.shared_rng,
                   "stratify_small_vocab": not args.no_stratify_masks,
                   # Same reason as the two above: scoping case (f) changes
                   # which configs the generator emits, so a report that omits
                   # it cannot be reproduced.
                   "wall_clear_scope": (
                       None if scope is None
                       else {"file": os.path.abspath(args.wall_clear_scope),
                             "n_stems": len(scope),
                             "sha256": scope_digest})},
        "generation_stats": dict(stats),
        "splits": {},
    }

    # A --splits run re-renders part of an existing benchmark (e.g. the 108
    # stratification-changed eval configs). The report it writes must still
    # describe the whole thing, so carry forward what the previous run
    # recorded for the splits we are not touching -- otherwise a scoped
    # re-render silently replaces a complete v1.0 record with a partial one.
    prior_path = os.path.join(args.out, "pilot_report.json")
    if args.splits and os.path.isfile(prior_path):
        try:
            with open(prior_path) as f:
                prior = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"[pilot] WARNING unreadable prior report ({exc}); the "
                  f"untouched splits will be absent from this report",
                  flush=True)
        else:
            for split, entry in (prior.get("splits") or {}).items():
                if split not in args.splits:
                    report["splits"][split] = entry
            report["carried_forward_splits"] = sorted(
                s for s in report["splits"] if s not in args.splits)
            print(f"[pilot] carried forward splits "
                  f"{report['carried_forward_splits']} from prior report",
                  flush=True)

    all_instances = []
    for split, instances in splits.items():
        if args.splits and split not in args.splits:
            # Still audited below: the §7 gates are properties of the PLAN,
            # not of whichever splits this invocation re-rendered.
            all_instances.extend(instances)
            continue
        print(f"[pilot] rendering split {split} ({len(instances)})", flush=True)
        manifest = render_batch(
            instances, args.out, split,
            dry_run=args.dry_run, export_masks=not args.dry_run,
            on_missing="mark", wall_clear_scope=scope)
        split_dir = os.path.join(args.out, split)
        rendered = sum(1 for e in manifest if e["rendered"])

        ok, violations = decidability_check(instances)
        mask_summary = (aggregate_mask_summaries(split_dir)
                        if not args.dry_run else None)

        parquet_path = None
        if not args.dry_run and rendered == len(instances):
            parquet_path = pack_parquet(instances, split_dir)
        elif args.dry_run:
            parquet_path = pack_parquet(instances, split_dir,
                                        require_images=False)

        report["splits"][split] = {
            "n_instances": len(instances),
            "n_rendered": rendered,
            "decidability_ok": ok,
            "decidability_violations": violations,
            "parquet": parquet_path,
            "mask_summary": mask_summary,
        }
        all_instances.extend(instances)
        with open(os.path.join(args.out, "pilot_report.json"), "w") as f:
            json.dump(report, f, indent=1, default=str)

    audit = build_audit_report(all_instances, stats)
    report["audit_report"] = audit

    # §6.2 volume-gate progress over TRAIN renderer-GT boxes only.
    train_counts = ((report["splits"].get("train") or {}).get("mask_summary")
                    or {}).get("class_box_counts") or {}
    report["volume_gate_progress"] = {
        cls: {"train_boxes": train_counts.get(cls, 0), "target": 100}
        for cls in NEW_CLASSES
    }

    out_path = os.path.join(args.out, "pilot_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[pilot] report -> {out_path}", flush=True)

    # Carried-forward splits (a scoped --splits run) only have whatever the
    # prior report recorded, so every key here has to be optional -- indexing
    # them would crash this summary AFTER the whole render had succeeded.
    n_fail = sum((s.get("mask_summary") or {}).get("n_failed", 0)
                 for s in report["splits"].values())
    n_viol = sum(len(s.get("decidability_violations") or ())
                 for s in report["splits"].values())
    print(f"[pilot] DONE renders_failed={n_fail} "
          f"decidability_violations={n_viol} "
          f"audit_ok={ {k: v.get('ok') for k, v in audit.items() if isinstance(v, dict) and 'ok' in v} }",
          flush=True)
    return report


if __name__ == "__main__":
    main()
