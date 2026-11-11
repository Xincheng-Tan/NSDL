#!/usr/bin/env python
"""Check whether a LOCUS eval bucket actually ran the full34 detector (A0b G12).

WHY THIS EXISTS. `scripts/eval_s2_locus.sh` used to never pass DETECTOR_WEIGHTS,
so it fell back to `checkpoints/best.pt` -- the VIKI-legacy 20-class detector --
while `scripts/eval_p61_locus_points.sh` (P6.1 points 2/3) passed the 34-class
LOCUS detector. The P6.1 four-point table was therefore not perception-parity:
point 4 was handicapped. Nothing in the result JSON records which weights ran
(`agentic_eval.detector_weights` is None), so the only way to tell from an
artifact is to look at what classes the detect tool ever returned.

Measured on the pre-fix buckets (2026-08-26), both 560 files:

    locus_qwen25_7b_v33_seed42_a1  -> 13 distinct classes, 0 full34-exclusive hits
    locus_qwen25_3b_v33_seed42     -> 13 distinct classes, 0 full34-exclusive hits

The 14 full34-exclusive classes (bottle, cardboardbox, cup, drawer, fork, fridge,
kettle, meat, oven, pizza, pumpkin, scissors, spoon, wine) never appear once --
that is the signature of the full20 fallback.

PASS CRITERION for the re-run buckets: full34-exclusive hits > 0. Zero hits in a
bucket that was *supposed* to use full34 means the override did not take effect;
do not read the numbers until that is resolved.

Usage:
    python scripts/audit_locus_detector_parity.py \
        --buckets locus_qwen25_7b_v33_seed42_a1_full34,locus_qwen25_3b_v33_seed42_full34 \
        --require-exclusive
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from nsdl.yolo_detector import DEFAULT_YOLO26_WEIGHTS, LEGACY_VIKI_FULL20_WEIGHTS  # noqa: E402

DEFAULT_ROOT = REPO / "results" / "lnsdl" / "v3_3" / "locus"
# This audit compares the two calibers on purpose (G12); it names the legacy
# file through its constant so the one-detector scan in tests stays honest.
FULL20_WEIGHTS = LEGACY_VIKI_FULL20_WEIGHTS
FULL34_WEIGHTS = DEFAULT_YOLO26_WEIGHTS


def load_class_names(weights: Path) -> set[str]:
    """Read a YOLO checkpoint's class-name set.

    Deliberately reads the checkpoints rather than hardcoding the lists, so this
    audit cannot drift away from whatever the eval actually loads.
    """
    import torch

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    names = getattr(model, "names", None) or (ckpt.get("names") if isinstance(ckpt, dict) else None)
    if not names:
        raise SystemExit(f"could not read class names from {weights}")
    return set(names.values()) if isinstance(names, dict) else set(names)


def bucket_detect_classes(bucket_dir: Path) -> tuple[Counter, int]:
    """Count every entity label the detect tool returned across a bucket."""
    counts: Counter = Counter()
    files = sorted(glob.glob(str(bucket_dir / "*" / "*.json")))
    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        trace = (data.get("agentic_eval") or {}).get("trace") or {}
        names = trace.get("tool_names") or []
        results = trace.get("tool_results") or []
        # tool_names and tool_results are parallel lists; a bucket run without
        # any detect call yields nothing here, which reads as "no evidence"
        # rather than "parity failed" -- the caller sees files==N, classes==0.
        for name, result in zip(names, results):
            if name == "detect" and isinstance(result, dict):
                for entity in result.get("entities") or []:
                    counts[str(entity)] += 1
    return counts, len(files)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--buckets", required=True, help="Comma-separated bucket names under --root.")
    p.add_argument("--full20-weights", default=str(FULL20_WEIGHTS))
    p.add_argument("--full34-weights", default=str(FULL34_WEIGHTS))
    p.add_argument("--out", default="")
    p.add_argument(
        "--require-exclusive",
        action="store_true",
        help="Exit 1 unless EVERY bucket shows at least one full34-exclusive class.",
    )
    return p.parse_args(argv)


def audit(root: Path, buckets: list[str], exclusive: set[str]) -> dict:
    rows = []
    for bucket in buckets:
        bucket_dir = root / bucket
        if not bucket_dir.is_dir():
            raise SystemExit(f"missing bucket: {bucket_dir}")
        counts, n_files = bucket_detect_classes(bucket_dir)
        hits = {k: v for k, v in counts.items() if k in exclusive}
        rows.append(
            {
                "bucket": bucket,
                "files": n_files,
                "distinct_classes": len(counts),
                "exclusive_hits": sum(hits.values()),
                "exclusive_classes_seen": sorted(hits),
                "classes": sorted(counts),
                "verdict": "full34" if hits else "full20-fallback-or-no-detect",
            }
        )
    return {"exclusive_classes": sorted(exclusive), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    buckets = [b for b in args.buckets.split(",") if b]
    full20 = load_class_names(Path(args.full20_weights))
    full34 = load_class_names(Path(args.full34_weights))
    exclusive = full34 - full20
    report = audit(Path(args.root), buckets, exclusive)

    print(f"full34-exclusive classes ({len(exclusive)}): {', '.join(sorted(exclusive))}\n")
    for r in report["rows"]:
        print(
            f"{r['bucket']}\n"
            f"  files={r['files']} distinct_classes={r['distinct_classes']} "
            f"exclusive_hits={r['exclusive_hits']}  -> {r['verdict']}"
        )
        if r["exclusive_classes_seen"]:
            print(f"  exclusive seen: {', '.join(r['exclusive_classes_seen'])}")
        print(f"  classes: {', '.join(r['classes'])}")
    failed = [r["bucket"] for r in report["rows"] if not r["exclusive_hits"]]
    if failed:
        print(
            f"\nWARNING: no full34-exclusive class in: {', '.join(failed)}\n"
            "  If these buckets were meant to run full34, the override did not take\n"
            "  effect -- do not read their numbers as perception-parity."
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    if args.require_exclusive and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
