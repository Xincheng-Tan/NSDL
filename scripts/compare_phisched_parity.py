#!/usr/bin/env python3
"""plan.md 序 26 parity self-check: the offline-derived `<mainrow>__phisched`
bucket vs the live forced-Phi arm (`<mainrow>_forcedphi_always__phisched`).
Both runs decode at temperature 0 from the same checkpoint, so up to the final
submission their trajectories should coincide; on rows where the derived side
took Phi's last accepted schedule (source last_ok) and the forced side
rescheduled, the final plans and scores must agree row by row. Rows where they
differ are listed with the reason so a mismatch is read, not assumed.

Usage: python scripts/compare_phisched_parity.py DERIVED_BUCKET FORCED_PHISCHED_BUCKET [--level locus|viki]
"""
import argparse, collections, glob, json, os, sys

def norm(plan):
    if not isinstance(plan, list): return None
    return [(i + 1, tuple(sorted((str(k), tuple(str(x) for x in v)) for k, v in (st.get("actions") or {}).items() if v)))
            for i, st in enumerate(plan) if isinstance(st, dict)]

def load(bucket):
    out = {}
    for fp in glob.glob(os.path.join(bucket, "*", "*.json")):
        d = json.load(open(fp)); rel = os.path.relpath(fp, bucket)
        ae = d.get("agentic_eval") or {}; tr = ae.get("trace") or {}
        rec = tr.get("forced_phi_reschedule") if isinstance(tr.get("forced_phi_reschedule"), dict) else None
        out[rel] = dict(plan=norm(d.get("plan")), status=d.get("result_status"), bs=d.get("benchmark_scores") or {},
                        turns=[t for t in (tr.get("assistant_turns") or []) if isinstance(t, str)],
                        src=ae.get("phisched_source") or ((rec or {}).get("final_plan_source") and f"forced:{(rec or {}).get('outcome')}") or "stored",
                        forced_outcome=(rec or {}).get("outcome"))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("derived"); ap.add_argument("forced"); ap.add_argument("--level", default="locus")
    a = ap.parse_args(); key = "locus_l2_acc" if a.level == "locus" else "viki_l2_plan_score"
    D, F = load(a.derived), load(a.forced)
    common = sorted(set(D) & set(F)); c = collections.Counter(); diffs = []; hard = []
    for rel in common:
        d, f = D[rel], F[rel]
        cell = (d["src"], f["forced_outcome"])
        c[f"pair:{cell[0]}|forced={cell[1]}"] += 1
        same_plan = d["plan"] == f["plan"]; same_score = abs(float(d["bs"].get(key) or 0) - float(f["bs"].get(key) or 0)) < 1e-9
        # the forced arm only differs in the final submission, so the trajectory
        # up to the last model turn should be byte-identical when decoding is
        # deterministic; a plan mismatch on a row whose turns already diverged is
        # decoding noise between two runs, not a reconstruction error.
        same_traj = d["turns"][:-1] == f["turns"][:-1] and bool(d["turns"]) and bool(f["turns"])
        c["trajectory_identical_before_final" if same_traj else "trajectory_diverged"] += 1
        if d["src"] == "last_ok" and f["forced_outcome"] == "rescheduled":
            c["comparable"] += 1
            if same_plan and same_score: c["comparable_identical"] += 1
            else:
                kind = "plan" if not same_plan else "score"
                c["comparable_DIFF_traj_diverged" if not same_traj else "comparable_DIFF_traj_identical"] += 1
                diffs.append((rel, kind + ("" if same_traj else " [traj diverged]"), d["bs"].get(key), f["bs"].get(key)))
                if same_traj: hard.append(rel)
        elif same_plan and same_score: c["other_identical"] += 1
        else: c["other_diff"] += 1
    print(f"derived={a.derived}\nforced ={a.forced}\ncommon rows={len(common)} (derived-only {len(set(D)-set(F))}, forced-only {len(set(F)-set(D))})")
    for k, v in sorted(c.items()): print(f"  {v:4d} {k}")
    for rel, kind, sd, sf in diffs[:20]: print(f"  DIFF {rel} {kind}: derived={sd} forced={sf}")
    print(f"verdict: {'PARITY OK' if not hard else 'PARITY BROKEN'} -- {len(hard)} comparable rows differ with identical trajectories")
    return 0 if not hard else 1

if __name__ == "__main__": sys.exit(main())
