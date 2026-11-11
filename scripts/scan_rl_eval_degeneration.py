#!/usr/bin/env python
"""Greedy-degeneration scan over eval buckets (results/lnsdl/**/<bucket>/{test,val}).

A sample is "degenerate" when any assistant turn contains a unit of <=8 chars
repeated >=15 times consecutively (catches `<th<th<th`, `opopop`, `the the the`).
Also reports where the loop starts (turn index) and per-family concentration.

Usage:
  python scripts/scan_rl_eval_degeneration.py results/lnsdl/rl_v3/viki_l2/rl_formal_cfa_v341_step60 [more buckets]
  python scripts/scan_rl_eval_degeneration.py --root results/lnsdl/rl_v3/viki_l2   # every bucket under root
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

LOOP = re.compile(r"(.{1,8}?)\1{15,}")


def scan(split_dir, families=True):
    fs = sorted(glob.glob(os.path.join(split_dir, "*.json")))
    if not fs:
        return None
    n = 0
    deg = 0
    acc_sum = 0.0
    err = 0
    fam = collections.Counter()
    fam_all = collections.Counter()
    turn_idx = collections.Counter()
    for f in fs:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        n += 1
        fid = str(r.get("sample", {}).get("task_id", "?")).split("_")[-1]
        fam_all[fid] += 1
        turns = r.get("agentic_eval", {}).get("trace", {}).get("assistant_turns", []) or []
        hit = None
        for i, t in enumerate(turns):
            if LOOP.search(t):
                hit = i
                break
        if hit is not None:
            deg += 1
            fam[fid] += 1
            turn_idx[hit] += 1
        sc = (r.get("benchmark_scores") or {}).get("viki_l2_plan_score")
        if sc is None:
            sc = (r.get("benchmark_scores") or {}).get("locus_l2_acc", 0.0)
        acc_sum += float(sc or 0.0)
        if r.get("result_status") != "ok":
            err += 1
    out = dict(n=n, deg=deg, deg_rate=deg / n, acc=acc_sum / n, err=err, turn_idx=dict(turn_idx))
    if families:
        out["families"] = {k: f"{fam[k]}/{fam_all[k]}" for k in sorted(fam_all) if fam[k]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buckets", nargs="*")
    ap.add_argument("--root")
    ap.add_argument("--splits", default="test,val,test_id,ood_s,ood_m,ood_x")
    a = ap.parse_args()
    buckets = list(a.buckets)
    if a.root:
        buckets += sorted(os.path.join(a.root, b) for b in os.listdir(a.root))
    for b in buckets:
        parts = []
        for s in a.splits.split(","):
            x = scan(os.path.join(b, s))
            if x is None:
                continue
            parts.append(
                f"{s}: n={x['n']} deg={x['deg']} ({x['deg_rate']:.3f}) acc={x['acc']:.4f} err={x['err']}"
                + (f" turn={x['turn_idx']} fam={x['families']}" if x["deg"] else "")
            )
        print(os.path.basename(b.rstrip("/")), " | ".join(parts) if parts else "(no splits)")


if __name__ == "__main__":
    sys.exit(main())
