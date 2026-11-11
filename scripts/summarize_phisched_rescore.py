#!/usr/bin/env python3
"""Markdown summary of results/lnsdl/audits/rescore_phisched/*.json (plan.md 序 26)."""
import glob, json, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "results/lnsdl/audits/rescore_phisched"
rows = []
for fp in sorted(glob.glob(os.path.join(root, "*.json"))):
    r = json.load(open(fp)); key = r["score_key"]; c = r["counters"]; ps = r["per_split"]
    n = sum(v["n"] for v in ps.values())
    fallback = sum(v for k, v in c.items() if k.startswith("fallback_")) + c.get("forced_fallback_kept_stored", 0)
    replaced = c.get("plan_replaced", 0)
    cells = []
    for sp, v in ps.items():
        cells.append(f"{sp} {v[key+'_stored']:.4f}→{v[key+'_rescored']:.4f} (err {v['err_stored']}→{v['err_rescored']})")
    tot_old = sum(v[key+'_stored'] * v['n'] for v in ps.values()) / n; tot_new = sum(v[key+'_rescored'] * v['n'] for v in ps.values()) / n
    err_old = sum(v['err_stored'] for v in ps.values()); err_new = sum(v['err_rescored'] for v in ps.values())
    rows.append((os.path.basename(r["bucket"]), n, tot_old, tot_new, err_old, err_new, replaced, fallback, c.get("parity_broken", 0), " · ".join(cells)))
print("| bucket | n | ALL stored→phisched | err stored→phisched | plan replaced | fallback(自产) | parity_broken | per split |")
print("|---|---|---|---|---|---|---|---|")
for b, n, o, nw, eo, en, rep, fb, pb, cells in rows:
    print(f"| `{b}` | {n} | {o:.4f}→{nw:.4f} ({(nw-o)*100:+.2f}) | {eo}→{en} | {rep} | {fb} | {pb} | {cells} |")
