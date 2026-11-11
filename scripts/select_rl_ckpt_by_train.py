#!/usr/bin/env python3
"""debug_rl.md §7.8 P2 (2026-09-07): pick the RL checkpoint to evaluate from
TRAINING-SIDE signals only (pre-registered rule; never from test readings).

Rule: among the saved steps, take the one maximising the trailing-window mean
of ``critic/score/mean`` (window = the saved step and the two steps before it)
subject to the degeneration red lines on the same window:
  nonterm  (reward/nonterm/mean)          <= NONTERM_MAX  (default .05)
  stacked  (reward/stacked/mean)          <= STACKED_MAX  (default .05)
  clip     (response_length/clip_ratio)   <= CLIP_MAX     (default .05)
  entropy  (actor/entropy)  <= ENTROPY_X * mean entropy of steps 1-3 (default 2.5x)
Among passing steps whose windowed score is within SCORE_TOL (default .02) of
the best passing score, the LATEST step wins (the batch score is noisy at the
0.01 level and later non-degenerate steps have seen more data).
Falls back to the LAST saved step if no saved step passes (and says so).
s46 reference (SAVE_FREQ=10): score peaks at steps 22-24, red lines cross at
25-27; on the s46 log with saved=5..30 this rule picks 25 (30 fails all lines).

Usage: select_rl_ckpt_by_train.py LOG --saved 5,10,15,20,25,30 [--json]
"""
from __future__ import annotations
import argparse, json, re, sys

KEYS = {"score": "critic/score/mean", "nonterm": "reward/nonterm/mean",
        "stacked": "reward/stacked/mean", "clip": "response_length/clip_ratio",
        "entropy": "actor/entropy"}


def parse(log: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in open(log, errors="replace"):
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        d: dict[str, float] = {}
        for kv in m.group(2).split(" - "):
            if ":" in kv:
                k, v = kv.rsplit(":", 1)
                try:
                    d[k.strip()] = float(v)
                except ValueError:
                    pass
        if d:
            rows[int(m.group(1))] = d
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--saved", default="5,10,15,20,25,30")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--nonterm-max", type=float, default=0.05)
    ap.add_argument("--stacked-max", type=float, default=0.05)
    ap.add_argument("--clip-max", type=float, default=0.05)
    ap.add_argument("--entropy-x", type=float, default=2.5)
    ap.add_argument("--score-tol", type=float, default=0.02)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = parse(a.log)
    saved = [int(x) for x in a.saved.split(",") if x.strip()]
    base = [rows[t].get(KEYS["entropy"]) for t in (1, 2, 3) if t in rows and rows[t].get(KEYS["entropy"]) is not None]
    entropy_max = (sum(base) / len(base)) * a.entropy_x if base else float("inf")
    table = []
    for s in saved:
        win = [rows[t] for t in range(s - a.window + 1, s + 1) if t in rows]
        if not win:
            table.append({"step": s, "missing": True}); continue
        agg = {k: sum(r.get(key, 0.0) for r in win) / len(win) for k, key in KEYS.items()}
        agg["step"] = s; agg["n"] = len(win)
        agg["entropy_max"] = entropy_max
        agg["pass"] = (agg["nonterm"] <= a.nonterm_max and agg["stacked"] <= a.stacked_max
                       and agg["clip"] <= a.clip_max and agg["entropy"] <= entropy_max)
        table.append(agg)
    ok = [t for t in table if t.get("pass")]
    if ok:
        best = max(t["score"] for t in ok)
        chosen = max(t["step"] for t in ok if t["score"] >= best - a.score_tol)
        why = f"latest step passing red lines with windowed score within {a.score_tol} of best ({best:.3f})"
    else:
        chosen = [t for t in table if not t.get("missing")][-1]["step"] if any(not t.get("missing") for t in table) else saved[-1]
        why = "NO saved step passed the red lines -> last saved step (report this)"
    if a.json:
        print(json.dumps({"chosen": chosen, "why": why, "table": table}, indent=1))
    else:
        for t in table:
            if t.get("missing"):
                print(f"step {t['step']:3d}  (no rows)"); continue
            print(f"step {t['step']:3d}  score {t['score']:.3f}  nonterm {t['nonterm']:.3f}  stacked {t['stacked']:.3f}  clip {t['clip']:.3f}  entropy {t['entropy']:.3f}(<= {t['entropy_max']:.3f})  {'PASS' if t['pass'] else 'fail'}")
        print(f"CHOSEN {chosen}  ({why})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
