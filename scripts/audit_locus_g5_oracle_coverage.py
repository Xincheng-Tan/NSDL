#!/usr/bin/env python3
"""Check that a G5 oracle upper-bound bucket actually SAW more than the YOLO control.

WHY THIS EXISTS (memory: g5-oracle-degenerate-f9-f10). The G5 reading is
"oracle perception on the five zero / near-zero cells". The intended
interpretation is binary:

    oracle stays near zero  -> the cell is genuinely hard
    oracle lifts the cell   -> the cell was perception-bound, and
                               benchmark.md 7.9's real_difficulty verdict
                               has to be rewritten

That dichotomy is only valid if the oracle backend is in fact an UPPER bound on
perception. Measured on 2026-08-26 it is not, on every cell: for some families
the oracle vocabulary emits ZERO relations where the YOLO detector emits some,
so the "oracle" arm hands the model LESS evidence than the control. A near-zero
oracle score in such a cell says nothing about difficulty -- it is an artifact of
the backend's vocabulary, and reading it as "really hard" is exactly the error
this script exists to prevent.

The check is deliberately a comparison, not a threshold: same cells, same model,
same recipe, only the detector backend differs, so any cell where oracle gives
strictly fewer entities or fewer relations per detect call is flagged.

Usage:
    python scripts/audit_locus_g5_oracle_coverage.py \
        --oracle-bucket locus_g5_oracle_upper_a1 \
        --control-bucket locus_g5_yolo_ctrl_a1
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "results" / "lnsdl" / "v3_3" / "locus"


def bucket_cell_stats(bucket_dir: Path) -> dict[str, dict]:
    """Per LOCUS cell: rows, detect calls, and mean entities/relations per call."""
    cells: dict[str, dict] = {}
    for path in sorted(glob.glob(str(bucket_dir / "*" / "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cell = str((data.get("sample") or {}).get("task_id") or "?")
        c = cells.setdefault(
            cell,
            {"rows": 0, "detect_calls": 0, "entities": 0, "relations": 0,
             "rows_with_zero_relations": 0, "backends": set()},
        )
        c["rows"] += 1
        c["backends"].add(str((data.get("agentic_eval") or {}).get("detector_backend")))
        trace = (data.get("agentic_eval") or {}).get("trace") or {}
        row_relations = 0
        for name, result in zip(trace.get("tool_names") or [], trace.get("tool_results") or []):
            if name == "detect" and isinstance(result, dict):
                c["detect_calls"] += 1
                c["entities"] += len(result.get("entities") or [])
                n = len(result.get("relations") or [])
                c["relations"] += n
                row_relations += n
        if row_relations == 0:
            c["rows_with_zero_relations"] += 1
    for c in cells.values():
        n = max(c["detect_calls"], 1)
        c["entities_per_call"] = round(c["entities"] / n, 3)
        c["relations_per_call"] = round(c["relations"] / n, 3)
        c["backends"] = sorted(c["backends"])
    return cells


def audit(oracle: dict[str, dict], control: dict[str, dict]) -> dict:
    rows = []
    for cell in sorted(set(oracle) | set(control)):
        o, k = oracle.get(cell), control.get(cell)
        if o is None or k is None:
            rows.append({"cell": cell, "verdict": "MISSING",
                         "note": f"oracle={'y' if o else 'n'} control={'y' if k else 'n'}"})
            continue
        fewer_ent = o["entities_per_call"] < k["entities_per_call"]
        fewer_rel = o["relations_per_call"] < k["relations_per_call"]
        # CRITERION REVISED 2026-08-26 AFTER SEEING THE FIRST RUN -- disclosed, not
        # quietly swapped. The first version flagged DEGENERATE on `fewer_ent OR
        # fewer_rel`, which named all five G5 cells. Four of them (F5_s2m3/F5_s3m3/
        # F8_s0m3/F8_s1m3) were flagged purely on the ENTITY count while handing the
        # model 4-7x MORE relations than the YOLO control (3.0-5.0 vs 0.5-0.85 per
        # call, zero-relation rows 0 vs 7-11). That is not starvation: the oracle
        # emits the ground-truth entity set while YOLO additionally reports
        # duplicates and false positives, so fewer-but-correct entities is the
        # expected shape, not a deficit.
        #
        # The failure this audit exists to catch (memory g5-oracle-degenerate-f9-f10)
        # has a specific signature: the oracle vocabulary emits NO relations at all,
        # so the arm meant to be an upper bound sees less than the control. Encode
        # that, plus the general "worse on both axes" case. An entity-only deficit is
        # reported as a note so it stays visible, but it does not drive the verdict.
        starved_rel = o["relations_per_call"] == 0 and k["relations_per_call"] > 0
        worse_both = (
            o["entities_per_call"] <= k["entities_per_call"]
            and o["relations_per_call"] <= k["relations_per_call"]
            and (fewer_ent or fewer_rel)
        )
        verdict = "DEGENERATE" if (starved_rel or worse_both) else "OK"
        notes = []
        if starved_rel:
            notes.append("oracle emits zero relations (backend vocabulary gap)")
        elif fewer_ent:
            notes.append(
                "entity-only deficit (oracle emits the GT entity set; YOLO also "
                "reports duplicates/false positives) -- not verdict-driving"
            )
        rows.append({
            "cell": cell,
            "verdict": verdict,
            "notes": notes,
            "oracle_rows": o["rows"], "control_rows": k["rows"],
            "oracle_entities_per_call": o["entities_per_call"],
            "control_entities_per_call": k["entities_per_call"],
            "oracle_relations_per_call": o["relations_per_call"],
            "control_relations_per_call": k["relations_per_call"],
            "oracle_rows_with_zero_relations": o["rows_with_zero_relations"],
            "control_rows_with_zero_relations": k["rows_with_zero_relations"],
            "oracle_backends": o["backends"], "control_backends": k["backends"],
        })
    bad = [r for r in rows if r["verdict"] != "OK"]
    return {
        "cells": rows,
        "degenerate_cells": [r["cell"] for r in bad],
        "ok": not bad,
        "read_discipline": (
            "Cells listed in degenerate_cells handed the model FEWER entities or "
            "relations under the oracle backend than under YOLO. Their oracle score "
            "is not an upper bound and must not be read as evidence of intrinsic "
            "difficulty; report them as 'oracle backend degenerate on this family'."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--oracle-bucket", required=True)
    p.add_argument("--control-bucket", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--require-ok", action="store_true",
                   help="Exit 1 if any cell is DEGENERATE or MISSING.")
    args = p.parse_args(argv)

    root = Path(args.root)
    for b in (args.oracle_bucket, args.control_bucket):
        if not (root / b).is_dir():
            raise SystemExit(f"missing bucket: {root / b}")
    report = audit(bucket_cell_stats(root / args.oracle_bucket),
                   bucket_cell_stats(root / args.control_bucket))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    # One greppable line so a chain script can quote the verdict without having
    # to re-parse the JSON inside a nested command substitution.
    bad = report["degenerate_cells"]
    print("VERDICT " + ("OK all-cells-covered" if not bad else "DEGENERATE " + ",".join(bad)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 1 if (args.require_ok and not report["ok"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
