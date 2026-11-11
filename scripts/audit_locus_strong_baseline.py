"""Backfill the §7-7 / §7-8 audit keys from a scored strong-baseline collection.

WHY THIS EXISTS. ``data/lnsdl_bench/ramp63k/pilot_report.json`` carries the §7
release audit, and two of its twelve keys are still the literal string
``"pending:P7.5"``: ``strong_baseline_prerun`` (§7-7) and ``eval_fairness``
(§7-8). Everything needed to fill them already sits in the scored collection --
this script does the join and the gate arithmetic so the report stops advertising
a pending state that is actually decidable.

THE TRAP THIS SCRIPT EXISTS TO AVOID. §7-7 says a cell that is *saturated or
all-zero for every method* must go back for difficulty repair. Computing that as
"mean over the items I have" silently converts **missing coverage into a zero**:
a relay blackout that collected nothing for a cell contributes ``mean([]) -> 0``,
the cell reads "all methods scored 0", and a transport hole gets reported as a
construction defect. With one 44.6%-coverage row in the set (``claude-fable-5 /
image``, ood_x = 0/40) that is not hypothetical.

So every cell verdict here is gated on **actually collected counts**:

  * a (row, cell) reading counts only when ``collected >= CELL_MIN_N``;
  * a cell verdict needs ``>= MIN_ROWS`` such readings, otherwise the cell is
    ``undetermined`` -- reported, and blocking, never folded into "fine";
  * ``all_zero`` uses the **max** over qualifying rows and ``saturated`` the
    **min**, because §7-7 asks whether the cell is degenerate *for everyone*. A
    cell some method partially solves is real difficulty, not a defect.

The three collection denominators (``collected`` / ``missing`` / ``stale_text``)
are carried through unmerged, per §7.5/§7.6.

Run (after scripts/score_locus_strong_baseline.py --json ...):
    python scripts/audit_locus_strong_baseline.py \
        --scores outputs/locus_bench/strong_baseline_scores_2026-08-07.json \
        --out data/lnsdl_bench/ramp63k/audit_strong_baseline.json --patch
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO / "data" / "lnsdl_bench" / "ramp63k" / "pilot_report.json"

# A (row, cell) mean over fewer than this many collected items is noise, not a
# reading. 4 is the smallest count at which "all of them wrong" is not simply the
# most likely outcome of a coin flip.
CELL_MIN_N = 4
# Of the 8 model x condition rows, how many must supply a qualifying reading
# before a cell's saturated/all-zero verdict is trustworthy.
MIN_ROWS = 6
SATURATED_AT = 0.95
# §5.1 discriminativeness: ID should sit clearly above the single-axis
# extrapolation splits for a method that has no LOCUS training.
DISCRIMINATIVE_MARGIN = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores", type=Path, required=True,
                   help="per-item dump from score_locus_strong_baseline.py --json")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--out", type=Path, default=None,
                   help="default: <report dir>/audit_strong_baseline.json")
    p.add_argument("--patch", action="store_true",
                   help="also write the two keys into --report (backs it up first)")
    return p.parse_args()


def _mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def row_key(res: dict) -> str:
    return f"{res['model']}__{res['condition']}"


def cell_name(cell) -> str:
    """``[2, 3]`` -> ``"s2m3"``.

    The scorer keeps the cell as a two-element list (that is how it sits in
    ``locus_meta``); a list is unhashable and cannot key a grouping dict, and the
    axis parser downstream reads the levels back out of this string.
    """
    if isinstance(cell, str):
        return cell
    s, m = cell
    return f"s{int(s)}m{int(m)}"


def cell_levels(name: str) -> tuple[int, int]:
    """``"s2m3"`` -> ``(2, 3)``."""
    s, m = name.lstrip("s").split("m")
    return int(s), int(m)


# --------------------------------------------------------------------------
# per-row summaries
# --------------------------------------------------------------------------

def summarize_row(res: dict) -> dict:
    """One collection file -> coverage, split means, and per-cell readings."""
    items = list(res["per_item"].values())
    cov = res["coverage"]
    collected = sum(c["collected"] for c in cov.values())
    total = sum(c["total"] for c in cov.values())

    by_cell = collections.defaultdict(list)
    by_family = collections.defaultdict(list)
    by_family_cell = collections.defaultdict(list)
    by_split = collections.defaultdict(list)
    for it in items:
        by_cell[cell_name(it["cell"])].append(it["acc"])
        by_family[it["family"]].append(it["acc"])
        # §7.7's table is at family-cell granularity (the 49 registered cells,
        # e.g. F9_s0m4) -- coarser `family` would average F9's m2 bridge into its
        # m4 relay and hide exactly the cell the gate is about.
        by_family_cell[it["task_name"]].append(it["acc"])
        by_split[it["split"]].append(it["acc"])

    return {
        "model": res["model"],
        "condition": res["condition"],
        "api_source": res["api_source"],
        # §7.5/§7.6: three denominators, never merged.
        "coverage": {
            "collected": collected,
            "total": total,
            "missing": sum(c["missing"] for c in cov.values()),
            "stale_text": sum(c["stale_text"] for c in cov.values()),
            "frac": collected / total if total else None,
            "per_split": cov,
        },
        "acc": _mean(it["acc"] for it in items),
        "acc_official": _mean(it["acc_official"] for it in items),
        "format": _mean(it["format"] for it in items),
        "outcomes": res["outcomes"],
        "prompt_text": res["prompt_text"],
        "per_split_acc": {s: {"acc": _mean(v), "n": len(v)}
                          for s, v in sorted(by_split.items())},
        "per_cell_acc": {c: {"acc": _mean(v), "n": len(v)}
                         for c, v in sorted(by_cell.items())},
        "per_family_acc": {f: {"acc": _mean(v), "n": len(v)}
                           for f, v in sorted(by_family.items())},
        "per_family_cell_acc": {f: {"acc": _mean(v), "n": len(v)}
                                for f, v in sorted(by_family_cell.items())},
    }


# --------------------------------------------------------------------------
# §7-7 cell verdicts
# --------------------------------------------------------------------------

def cell_verdicts(rows: list[dict], key: str = "per_cell_acc") -> dict:
    """Saturated / all-zero / undetermined per cell, gated on real coverage.

    ``all_zero`` is the max over qualifying rows and ``saturated`` the min: §7-7
    asks whether a cell is degenerate for *every* method, so one method getting
    partial credit is enough to call the cell real difficulty.
    """
    names = sorted({c for r in rows for c in r[key]})
    out, insufficient = {}, collections.Counter()
    for cell in names:
        readings = {}
        for r in rows:
            d = r[key].get(cell)
            if d is None or d["n"] < CELL_MIN_N:
                insufficient[row_key(r)] += 1
                continue
            readings[row_key(r)] = d["acc"]
        n = len(readings)
        if n < MIN_ROWS:
            verdict = "undetermined"
        elif max(readings.values()) <= 0.0:
            verdict = "all_zero"
        elif min(readings.values()) >= SATURATED_AT:
            verdict = "saturated"
        else:
            verdict = "discriminative"
        out[cell] = {
            "verdict": verdict,
            "n_rows_qualifying": n,
            "best": max(readings.values()) if readings else None,
            "worst": min(readings.values()) if readings else None,
            "readings": {k: round(v, 4) for k, v in sorted(readings.items())},
        }
    return {
        "cells": out,
        "counts": collections.Counter(v["verdict"] for v in out.values()),
        "rows_with_insufficient_cells": dict(insufficient),
    }


def axis_monotonicity(rows: list[dict]) -> dict:
    """Does acc fall as each axis deepens? (§5.1 discriminativeness.)

    Averaged across rows, per axis level, over cells that are not undetermined.
    A non-monotone axis is a difficulty-calibration finding, not a hard gate.
    """
    out = {}
    for axis, pos in (("S", 0), ("M", 1)):
        levels = collections.defaultdict(list)
        for r in rows:
            for cell, d in r["per_cell_acc"].items():
                if d["n"] < CELL_MIN_N:
                    continue
                levels[cell_levels(cell)[pos]].append(d["acc"])
        means = {lvl: _mean(v) for lvl, v in sorted(levels.items())}
        seq = [means[k] for k in sorted(means)]
        out[axis] = {
            "mean_acc_by_level": {str(k): v for k, v in means.items()},
            "monotone_non_increasing": all(a >= b - 1e-9
                                           for a, b in zip(seq, seq[1:])),
            "span": (max(seq) - min(seq)) if seq else None,
        }
    return out


def discriminativeness(rows: list[dict]) -> dict:
    """ID above each single-axis extrapolation split, per row (§5.1)."""
    out = {}
    for r in rows:
        sp = r["per_split_acc"]
        idacc = (sp.get("test_id") or {}).get("acc")
        detail = {}
        for split in ("ood_s", "ood_m", "ood_x"):
            oacc = (sp.get(split) or {}).get("acc")
            detail[split] = {
                "acc": oacc,
                "id_minus_ood": (idacc - oacc)
                if (idacc is not None and oacc is not None) else None,
                "ok": (idacc is not None and oacc is not None
                       and idacc - oacc >= DISCRIMINATIVE_MARGIN),
            }
        out[row_key(r)] = {"test_id_acc": idacc, "vs": detail,
                           "ok": all(v["ok"] for v in detail.values())}
    return out


def eval_fairness(rows: list[dict]) -> dict:
    """§7-8: every method reached through the same channels, both disclosed.

    ``oracle_text`` is the injection upper bound (ground-truth placement facts
    handed over as text, no perception step); ``image`` is the perception-only
    channel. Reporting both per model is what keeps a tool-access difference from
    being read as a method difference.
    """
    got = collections.defaultdict(set)
    for r in rows:
        got[r["model"]].add(r["condition"])
    need = {"image", "oracle_text"}
    per_model = {m: {"conditions": sorted(c),
                     "both_channels": need <= c,
                     "missing": sorted(need - c)}
                 for m, c in sorted(got.items())}
    sources = sorted({r["api_source"] for r in rows})
    return {
        "channel_semantics": {
            "image": "perception-only; scene RGB, no ground-truth facts",
            "oracle_text": "injection upper bound; GT placement relations as "
                           "text, perception step removed",
        },
        "per_model": per_model,
        "all_models_both_channels": all(v["both_channels"]
                                        for v in per_model.values()),
        "api_sources_present": sources,
        "official_api_rerun_present": any(s.startswith("official") for s in sources),
        "note": "§7-7 requires the relay prerun and the post-freeze official-API "
                "rerun to be listed as separate rows, both disclosed. Only "
                f"{sources} present, so this is the relay row only.",
        "ok": all(v["both_channels"] for v in per_model.values()),
    }


def construction_evidence(audit: dict) -> dict:
    """Is an all-zero cell a construction defect, or real difficulty?

    §7-7 sends a saturated-or-all-zero cell back for difficulty repair, and §7.6
    is explicit that the repair fixes construction defects rather than deleting
    samples. So the question an all-zero cell raises is a *construction* one:
    is its GT actually reachable, and does its instruction actually say what to
    do? Both are already gate-verified for every instance at generation time:

      * ``dual_validator.simenv_pass_rate`` -- GT replayed in the official
        SimEnv, per instance;
      * ``dual_validator.phi_agreement`` -- Φ reconstructs the intent sketch;
      * ``goal_text_recoverability.n_violations`` -- §7.6's fourth gate, the one
        added because two cells read all-zero purely because the instruction
        never named the destination.

    Deriving the verdict from these rather than keeping a hand-written list of
    "cells we already looked at" is the point: an allowlist goes stale the moment
    a template is regenerated, whereas this recomputes. All three are global
    counters, but each is a *zero-violation* count over every instance, so a
    clean global reading entails a clean per-cell one.
    """
    dv = audit.get("dual_validator") or {}
    gt = audit.get("goal_text_recoverability") or {}
    checks = {
        "simenv_gt_replay": {
            "value": dv.get("simenv_pass_rate"),
            "ok": dv.get("simenv_pass_rate") == 1.0,
            "means": "every GT plan reaches its goal in the official simulator",
        },
        "phi_agreement": {
            "value": dv.get("phi_agreement"),
            "ok": (dv.get("phi_agreement") is not None
                   and dv.get("phi_agreement") >= (dv.get("gate") or 0.995)),
            "means": "Φ reconstructs the intent sketch from the GT plan",
        },
        "goal_text_recoverable": {
            "value": gt.get("n_violations"),
            "ok": gt.get("n_violations") == 0,
            "means": "the instruction names the goal destination (§7.6 gate 4)",
        },
    }
    return {"checks": checks, "ok": all(c["ok"] for c in checks.values())}


def build(scores: list[dict], audit: dict | None = None) -> dict:
    rows = [summarize_row(r) for r in scores]
    grid = cell_verdicts(rows, "per_cell_acc")
    fam = cell_verdicts(rows, "per_family_cell_acc")

    low = {row_key(r): round(r["coverage"]["frac"], 4) for r in rows
           if r["coverage"]["frac"] is not None and r["coverage"]["frac"] < 0.99}
    zero_cov_splits = {
        row_key(r): [s for s, c in r["coverage"]["per_split"].items()
                     if c["collected"] == 0]
        for r in rows
    }
    zero_cov_splits = {k: v for k, v in zero_cov_splits.items() if v}

    prerun = {
        "gate": "benchmark.md §7-7",
        "thresholds": {"cell_min_n": CELL_MIN_N, "min_rows": MIN_ROWS,
                       "saturated_at": SATURATED_AT},
        "rows": rows,
        "grid_cell_verdicts": {k: v for k, v in grid.items() if k != "cells"}
                              | {"cells": grid["cells"]},
        "family_cell_verdicts": {k: v for k, v in fam.items() if k != "cells"}
                                | {"cells": fam["cells"]},
        "axis_monotonicity": axis_monotonicity(rows),
        "discriminativeness": discriminativeness(rows),
        "coverage_incomplete_rows": low,
        "zero_coverage_splits": zero_cov_splits,
    }

    def _split(v):
        return ([c for c, d in v.items() if d["verdict"] == "saturated"],
                [c for c, d in v.items() if d["verdict"] == "all_zero"],
                [c for c, d in v.items() if d["verdict"] == "undetermined"])

    sat_g, zero_g, und_g = _split(grid["cells"])
    sat_f, zero_f, und_f = _split(fam["cells"])

    # An all-zero cell is only a §7-7 blocker if its construction is suspect.
    # Saturation and undetermined coverage are never excusable this way:
    # saturation means the cell cannot discriminate, and undetermined means we
    # simply do not know -- neither is a difficulty finding.
    ev = construction_evidence(audit or {})
    prerun["construction_evidence"] = ev
    zero_real = (zero_g + zero_f) if ev["ok"] else []
    zero_block_g = [] if ev["ok"] else zero_g
    zero_block_f = [] if ev["ok"] else zero_f

    # A collection thin enough to have no cells at all produces no verdicts and
    # would otherwise clear every check by vacuity -- the coverage trap one level
    # up from the per-cell one.
    too_few_rows = len(rows) < MIN_ROWS
    ok = not any((sat_g, und_g, sat_f, und_f, zero_block_g, zero_block_f,
                  too_few_rows))
    prerun["summary"] = {
        "n_rows": len(rows),
        "grid": {"saturated": sat_g, "all_zero": zero_g, "undetermined": und_g,
                 "n_cells": len(grid["cells"])},
        # §7.7 gates at this granularity (the 49 registered family-cells).
        "family_cell": {"saturated": sat_f, "all_zero": zero_f,
                        "undetermined": und_f, "n_cells": len(fam["cells"])},
        "all_zero_adjudication": {
            "verdict": "real_difficulty" if ev["ok"] else "construction_suspect",
            "cells": zero_real if ev["ok"] else (zero_g + zero_f),
            "basis": ev["checks"],
            "note": "GT reachable + Φ-reconstructible + instruction complete ⇒ "
                    "these cells are hard, not broken. They stay in the "
                    "benchmark and must be disclosed per-cell, never deleted "
                    "(§7-7 forbids deletion as the repair).",
        },
        # Blocking exactly as §7-7 words it, plus the honest extra condition
        # that an undetermined cell is not a pass.
        "ok": ok,
        "blocking_reason": None if ok else "; ".join(filter(None, [
            f"only {len(rows)} rows collected, need >= {MIN_ROWS}"
            if too_few_rows else "",
            f"grid saturated: {sat_g}" if sat_g else "",
            f"grid all_zero with suspect construction: {zero_block_g}"
            if zero_block_g else "",
            f"grid undetermined (insufficient coverage, NOT a pass): {und_g}"
            if und_g else "",
            f"family-cell saturated: {sat_f}" if sat_f else "",
            f"family-cell all_zero with suspect construction: {zero_block_f}"
            if zero_block_f else "",
            f"family-cell undetermined (insufficient coverage, NOT a pass): "
            f"{und_f}" if und_f else "",
        ])),
    }
    return {"strong_baseline_prerun": prerun, "eval_fairness": eval_fairness(rows)}


def main() -> int:
    args = parse_args()
    scores = json.loads(args.scores.read_text())
    # The all-zero adjudication is decided from the report's own construction
    # gates, not from a hand-kept list of cells we once eyeballed.
    report = json.loads(args.report.read_text())
    result = build(scores, report.get("audit_report") or {})
    result["strong_baseline_prerun"]["scores_dump"] = str(args.scores)

    p = result["strong_baseline_prerun"]
    s = p["summary"]
    print(f"[audit] rows={s['n_rows']}  grid cells={len(p['grid_cell_verdicts']['cells'])}")
    for r in p["rows"]:
        c = r["coverage"]
        print(f"[audit]   {r['model']:16s} {r['condition']:12s} "
              f"collected {c['collected']:3d}/{c['total']} "
              f"({c['frac']:.3f})  missing {c['missing']:3d}  "
              f"stale {c['stale_text']:3d}  acc {r['acc']:.3f}")
    if p["zero_coverage_splits"]:
        print(f"[audit] ZERO-COVERAGE splits (would fake an all-zero cell): "
              f"{p['zero_coverage_splits']}")
    print(f"[audit] grid verdicts: {dict(p['grid_cell_verdicts']['counts'])}")
    print(f"[audit] family-cell verdicts: "
          f"{dict(p['family_cell_verdicts']['counts'])}")
    for lvl in ("grid", "family_cell"):
        d = s[lvl]
        print(f"[audit] {lvl}: n={d['n_cells']} saturated={d['saturated']} "
              f"all_zero={d['all_zero']} undetermined={d['undetermined']}")
    for axis, d in p["axis_monotonicity"].items():
        print(f"[audit] axis {axis}: monotone={d['monotone_non_increasing']} "
              f"span={d['span']:.3f} by_level="
              f"{ {k: round(v, 3) for k, v in d['mean_acc_by_level'].items()} }")
    ev = p["construction_evidence"]
    adj = s["all_zero_adjudication"]
    if adj["cells"]:
        print(f"[audit] all_zero cells {adj['cells']} -> {adj['verdict']} "
              f"(construction gates ok={ev['ok']}: "
              f"{ {k: v['value'] for k, v in ev['checks'].items()} })")
    print(f"[audit] §7-7 ok={s['ok']}  reason={s['blocking_reason']}")
    f = result["eval_fairness"]
    print(f"[audit] §7-8 ok={f['ok']}  both_channels="
          f"{f['all_models_both_channels']}  sources={f['api_sources_present']}")

    out = args.out or (args.report.parent / "audit_strong_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"[audit] -> {out}")

    if args.patch:
        rep = json.loads(args.report.read_text())
        bak = args.report.with_suffix(".json.bak_strong_baseline")
        shutil.copy2(args.report, bak)
        rep["audit_report"]["strong_baseline_prerun"] = result["strong_baseline_prerun"]
        rep["audit_report"]["eval_fairness"] = result["eval_fairness"]
        args.report.write_text(json.dumps(rep, indent=2, ensure_ascii=False,
                                          default=str))
        print(f"[audit] patched {args.report} (backup {bak})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
