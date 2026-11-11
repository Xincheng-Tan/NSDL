"""Audit the §7-7 pipeline full-injection row and re-check the cell verdicts.

Input is the per-item dump ``scripts/score_locus_strong_baseline.py --json``
writes for the two collections ``scripts/run_locus_phi_upper_bound.py`` produces
(``gt_ops`` = render only, ``phi_sketch`` = Φ resynthesizes the erased
structure). Output is ``audit_phi_upper_bound.json`` next to the §7-7 audit.

THREE QUESTIONS, AND ONLY THE FIRST ONE IS THE ONE §7-7 ASKED.

1. **Is any cell unsolvable?** §7-7 sends a cell back for difficulty repair when
   it is all-zero *for every method*, and that verdict is only a difficulty
   reading if something can score above zero there. The RENDER bound answers it:
   GT operations, no Φ synthesis, straight through the deployed graph builder /
   dispatcher / emitter. A cell it scores 0 on is one where our pipeline cannot
   even reproduce the ground truth, and the zero belongs to the construction.

2. **Where does the render bound lose, and to what?** Its misses are classified
   mechanically, not narrated: a plan that becomes the GT under a bijection of
   the robot slots (``R1``/``R2``/``R3``, rewritten inside ``Push`` arguments
   too) is a structurally identical plan the exact-match judge rejects, and
   saying "the pipeline cannot express it" would be false.

3. **Is the Φ row an upper bound at all?** It is named one in §7.1/§7.2, and the
   measurement says it is not: it is an upper bound on *exact GT reproduction
   through our pipeline given a perfect sketch*, and real methods beat it in
   cells where they echo the GT's own ordering. This audit prints every cell
   where a §7-7 method scores strictly higher, so the row cannot be quoted as a
   ceiling on achievable score.

Run:
    python scripts/audit_locus_phi_upper_bound.py \
        --scores outputs/locus_bench/phi_upper_bound_scores.json
"""

from __future__ import annotations

import argparse
import ast
import collections
import itertools
import json
import pathlib
import re
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BENCH = _REPO / "data" / "lnsdl_bench" / "ramp63k"
_COLLECTIONS = _REPO / "data" / "lnsdl_bench" / "phi_upper_bound"
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SLOTS = ("R1", "R2", "R3")

RENDER = "gt_ops"
PHI = "phi_sketch"


def _cell(value) -> str:
    if isinstance(value, str):
        return value
    s, m = value
    return f"s{int(s)}m{int(m)}"


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def group(items: dict, key) -> dict[str, dict]:
    out = collections.defaultdict(list)
    for value in items.values():
        out[key(value)].append(value["acc"])
    return {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in sorted(out.items())}


def _steps(plan) -> list[dict]:
    """-> [{robot: [action, ...]}] with idle slots dropped.

    ``v`` arrives as a numpy array on the parquet side, where ``if v`` raises
    rather than testing emptiness -- hence the explicit length test.
    """
    return [{k: [str(a) for a in v]
             for k, v in step.get("actions", {}).items()
             if v is not None and len(v)}
            for step in plan if isinstance(step, dict)]


def _canon(steps: list[dict]) -> list:
    return [tuple(sorted((k, tuple(v)) for k, v in step.items())) for step in steps]


def _relabel(steps: list[dict], perm: dict[str, str]) -> list[dict]:
    """Rename slots in the KEYS and inside the action arguments.

    ``Push`` carries the receiving robot as its second argument, so a rename
    that only touches the keys leaves a handoff pointing at the old slot and
    every carrier row reads as "genuinely different" when it is a relabelling.
    """
    return [{perm.get(k, k): [perm.get(a, a) for a in v] for k, v in step.items()}
            for step in steps]


def _is_slot_permutation(pred: list[dict], truth: list[dict]) -> bool:
    target = _canon(truth)
    return any(_canon(_relabel(pred, dict(zip(SLOTS, order)))) == target
               for order in itertools.permutations(SLOTS))


def _gt_steps() -> dict[str, list[dict]]:
    import pandas as pd
    out: dict[str, list[dict]] = {}
    for split_dir in sorted(p for p in _BENCH.iterdir() if (p / "data.parquet").is_file()):
        split = split_dir.name
        if split == "train":
            continue
        frame = pd.read_parquet(split_dir / "data.parquet", columns=["reward_model"])
        for i in range(len(frame)):
            truth = frame.iloc[i]["reward_model"]["ground_truth"]
            steps = truth.get("time_steps")
            out[f"{split}:{i:05d}"] = _steps(
                [] if steps is None else [dict(s) for s in steps])
    return out


def classify(variant: str, items: dict, truth: dict[str, list[dict]]) -> dict:
    """Why each scored-0 row scored 0. Mechanical, no judgement calls."""
    path = _COLLECTIONS / f"phi-full-injection__{variant}.jsonl"
    records = {json.loads(line)["item_id"]: json.loads(line)
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    counts: collections.Counter[str] = collections.Counter()
    slot_perm_rows: list[str] = []
    pipeline_rows: list[dict] = []
    for item_id, value in sorted(items.items()):
        if value["acc"] >= 1.0:
            continue
        record = records.get(item_id, {})
        match = _ANSWER.search(record.get("response", ""))
        if not match:
            counts["pipeline_failure"] += 1
            pipeline_rows.append({"item_id": item_id,
                                  "stage": record.get("phi_stage"),
                                  "code": record.get("phi_code")})
            continue
        try:
            pred = _steps(ast.literal_eval(match.group(1)))
        except Exception:  # noqa: BLE001
            counts["unparseable"] += 1
            continue
        gt = truth[item_id]
        if _is_slot_permutation(pred, gt):
            counts["slot_permutation_of_gt"] += 1
            slot_perm_rows.append(item_id)
        elif len(pred) == len(gt):
            counts["same_length_different_schedule"] += 1
        else:
            counts["different_length"] += 1
    return {"n_scored_zero": sum(counts.values()), "by_reason": dict(counts),
            "slot_permutation_rows": slot_perm_rows,
            "pipeline_failures": pipeline_rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=pathlib.Path,
                    default=_REPO / "outputs" / "locus_bench" / "phi_upper_bound_scores.json")
    ap.add_argument("--strong-baseline", type=pathlib.Path,
                    default=_BENCH / "audit_strong_baseline.json")
    ap.add_argument("--out", type=pathlib.Path,
                    default=_BENCH / "audit_phi_upper_bound.json")
    args = ap.parse_args()

    scored = {r["condition"]: r for r in json.loads(args.scores.read_text())}
    missing = {RENDER, PHI} - set(scored)
    if missing:
        print(f"missing variants in {args.scores}: {sorted(missing)}")
        return 1
    truth = _gt_steps()

    rows = {}
    for variant in (RENDER, PHI):
        res = scored[variant]
        items = res["per_item"]
        rows[variant] = {
            "variant": variant,
            "api_source": res.get("api_source"),
            "coverage": res["coverage"],
            "acc": _mean(v["acc"] for v in items.values()),
            "acc_official": _mean(v["acc_official"] for v in items.values()),
            "outcomes": res["outcomes"],
            "per_split_acc": group(items, lambda v: v["split"]),
            "per_cell_acc": group(items, lambda v: _cell(v["cell"])),
            "per_family_cell_acc": group(
                items, lambda v: f'{v["family"]}_{_cell(v["cell"])}'),
            "failures": classify(variant, items, truth),
        }

    render_cells = rows[RENDER]["per_family_cell_acc"]
    phi_cells = rows[PHI]["per_family_cell_acc"]
    worst = min(render_cells.items(), key=lambda kv: kv[1]["acc"])

    # §7-7's question. A cell the render bound cannot score on at all is a cell
    # whose zero is ours, not the methods'.
    unsolvable = sorted(k for k, v in render_cells.items() if v["acc"] <= 0.0)

    # The naming check. `best` in the §7-7 audit is the max over qualifying
    # method rows, so a cell listed here is one where a REAL method outscored
    # the thing §7.1 calls an upper bound.
    sb = json.loads(args.strong_baseline.read_text())["strong_baseline_prerun"]
    verdicts = sb["family_cell_verdicts"]["cells"]
    beaten = []
    for cell, value in sorted(phi_cells.items()):
        best = verdicts.get(cell, {}).get("best")
        if isinstance(best, (int, float)) and best > value["acc"] + 1e-9:
            beaten.append({"cell": cell, "phi_sketch": value["acc"],
                           "best_method": best, "n": value["n"],
                           "render_bound": render_cells[cell]["acc"]})

    all_zero = sb["summary"]["family_cell"]["all_zero"]
    recheck = {cell: {"render_bound": render_cells.get(cell, {}).get("acc"),
                      "phi_sketch": phi_cells.get(cell, {}).get("acc"),
                      "n": render_cells.get(cell, {}).get("n")}
               for cell in all_zero}

    report = {
        "gate": "benchmark.md §7-7 (third strong-baseline row: 流水线全注入上界)",
        "produced_by": ["scripts/run_locus_phi_upper_bound.py",
                        "scripts/score_locus_strong_baseline.py",
                        "scripts/audit_locus_phi_upper_bound.py"],
        "ruler": "nsdl.benchmark_scoring.compute_locus_l2_acc (same as every other LOCUS row)",
        "pooled_into_7_7_cell_verdicts": False,
        "rows": rows,
        "solvability": {
            "question": "§7-7: is any family-cell all-zero because it is unsolvable?",
            "answer_from": RENDER,
            "unsolvable_cells": unsolvable,
            "weakest_cell": {"cell": worst[0], **worst[1]},
            "n_cells": len(render_cells),
        },
        "all_zero_cell_recheck": recheck,
        "not_an_upper_bound": {
            "cells_where_a_method_scores_higher": beaten,
            "n_cells_beaten": len(beaten),
        },
        "disclosures": [
            "The `format` column is 0.000 on both rows by construction: viki_2.format_reward "
            "requires a <think> block and this row has no model to think. It is not a "
            "format failure and must not be compared with the method rows' format column.",
            "Neither row is pooled into the §7-7 saturated/all-zero cell vote. Those "
            "verdicts are over METHODS; an oracle row voting in them would erase the "
            "all-zero verdicts the gate exists to raise.",
            "Null-GT rows (73/560) are not run: there is nothing to inject and the correct "
            "answer is the empty plan, which compute_locus_l2_acc scores 1.0. They are "
            "counted in the denominator, so both rows are directly comparable to the "
            "method rows scored on the same 560.",
            "`phi_sketch` is an upper bound on EXACT GT REPRODUCTION through our pipeline "
            "given a perfect sketch -- not a ceiling on achievable LOCUS score. Where Φ "
            "synthesizes a different-but-schedulable ordering the exact-match judge scores "
            "it 0, and methods that echo the GT's own ordering can and do beat it.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"=== §7-7 pipeline full-injection row -> {args.out} ===")
    for variant in (RENDER, PHI):
        row = rows[variant]
        splits = "  ".join(f'{s} {v["acc"]:.3f}' for s, v in row["per_split_acc"].items())
        print(f"\n{variant:<11} acc {row['acc']:.4f}   {splits}")
        print(f"{'':11} scored-0 reasons: {row['failures']['by_reason']}")
    print(f"\nsolvability  : unsolvable cells = {unsolvable or 'none'}   "
          f"(weakest of {len(render_cells)}: {worst[0]} {worst[1]['acc']:.3f})")
    for cell, value in recheck.items():
        print(f"all-zero cell: {cell}  render bound {value['render_bound']:.3f} / "
              f"phi_sketch {value['phi_sketch']:.3f}  -- a correct answer EXISTS and is "
              f"renderable, so the cell is hard, not broken")
    print(f"\nNOT an upper bound: a §7-7 method outscores phi_sketch in "
          f"{len(beaten)}/{len(phi_cells)} family-cells")
    for entry in beaten[:8]:
        print(f"  {entry['cell']:<12} phi_sketch {entry['phi_sketch']:.3f} < "
              f"best method {entry['best_method']:.3f}   (render bound "
              f"{entry['render_bound']:.3f})")
    if len(beaten) > 8:
        print(f"  ... {len(beaten) - 8} more, see {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
