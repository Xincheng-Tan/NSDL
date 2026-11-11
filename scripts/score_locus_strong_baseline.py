"""Offline scorer for the §7-7 strong-baseline collection (no API, no GPU).

Reads the raw JSONL written by ``run_locus_strong_baseline.py`` and scores every
completed response with the **official VIKI-L2 judge**
(``verl.utils.reward_score.viki_2``) -- the same judge our own models are scored
with, so baseline and ours land on one ruler.

Two denominators are reported separately and must never be merged:

  collected   items with an ``ok`` response. A response that is empty of a plan,
              unparseable, or simply wrong scores 0 -- that is *model* failure
              and belongs in the numerator's denominator.
  missing     items with no ``ok`` response at all (relay blackout). These are
              *transport* holes; they are excluded from the mean and reported as
              coverage, because counting them as 0 would report gateway downtime
              as baseline difficulty.

The per-cell table joins ``locus_meta`` (cell, bench_family) back from the
parquet by ``(split, row)``, which is why the collector only had to record those
two fields.

The cross-condition S-axis delta is computed on the **paired intersection**
(items completed under both conditions), not on the two marginal means -- the
conditions have different coverage, so marginal means are not comparable.
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import io
import json
import pathlib
import random
import re
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_BENCH = _REPO / "data/lnsdl_bench/ramp63k"
_DEFAULT_DIR = _REPO / "data/lnsdl_bench/strong_baseline"
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Importing nsdl.benchmark_scoring is what puts the vendored verl on sys.path --
# `from verl...` does not resolve without it. Same seam our own eval uses, and
# the LOCUS success rule lives there so baseline and ours share one ruler.
from nsdl.benchmark_scoring import compute_locus_l2_acc  # noqa: E402
from verl.utils.reward_score import viki_2  # noqa: E402

sys.path.insert(0, str(_REPO / "scripts"))
from run_locus_strong_baseline import prompt_sha  # noqa: E402


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def _py(obj):
    """parquet -> plain python. The official judge type-checks list/dict, and a
    numpy ndarray passes none of those isinstance tests (it would silently score
    every plan 0)."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [_py(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(x) for x in obj]
    return obj


_GT_CACHE: dict[str, list[dict]] = {}


def _rewritten_rows() -> dict[str, set[int]]:
    """``{split: {row, ...}}`` for every instruction rewritten after freeze.

    Read from the ``description_repair_*.json`` reports the repair script
    leaves in the bench directory, so the rule generalises to any future text
    repair instead of pinning one date.
    """
    out: dict[str, set[int]] = collections.defaultdict(set)
    for p in sorted(_BENCH.glob("description_repair_*.json")):
        for split, value in json.loads(p.read_text()).items():
            if isinstance(value, dict):
                out[split] |= {int(k) for k in value.get("rewrites", {})}
    return out


def gt_table(split: str) -> list[dict]:
    """-> per row: {'gt', 'cell', 'family', 'task_name', 'prompt_sha'} for one split."""
    if split in _GT_CACHE:
        return _GT_CACHE[split]
    import pandas as pd
    df = pd.read_parquet(_BENCH / split / "data.parquet")
    rows = []
    for i in range(len(df)):
        gt = _py(df.iloc[i]["reward_model"]["ground_truth"])
        meta = gt.get("locus_meta") or {}
        cell = meta.get("cell") or [None, None]
        user = next(m["content"] for m in df.iloc[i]["prompt"]
                    if m["role"] == "user")
        rows.append({"gt": gt, "cell": (cell[0], cell[1]),
                     "family": meta.get("bench_family"),
                     "task_name": gt.get("task_name"),
                     "prompt_sha": prompt_sha(
                         user.replace("<image>", "", 1).strip()),
                     "null_task": not gt.get("time_steps")})
    _GT_CACHE[split] = rows
    return rows


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _plan_of(response: str):
    """-> (plan_list | None, reason). None means the response never produced a
    plan at all (protocol failure, not a wrong plan)."""
    m = _ANSWER.search(response)
    if not m:
        return None, "no_answer_tag"
    body = m.group(1).strip()
    if body.startswith("{{") and body.endswith("}}"):
        body = body.replace("{{", "{").replace("}}", "}")
    try:
        parsed = ast.literal_eval(body)
    except Exception:  # noqa: BLE001
        return None, "unparseable"
    if not isinstance(parsed, list):
        return None, "not_a_plan"
    return parsed, ""


def score_one(response: str, gt: dict, item_id: str) -> tuple[float, float, float, str]:
    """-> (acc_official, acc_locus, format, outcome).

    ``acc_official`` is ``viki_2.acc_reward`` verbatim, kept unmodified as the
    published-ruler column. ``acc_locus`` is the shared LOCUS rule
    (``nsdl.benchmark_scoring.compute_locus_l2_acc``), which differs only on
    null-plan items -- see that docstring. Both columns are always printed; they
    must never be silently merged.
    """
    plan, why = _plan_of(response)
    null_task = not gt.get("time_steps")

    # eval_single picks a position at random when an asset lists several; seed
    # per item so the table is reproducible across re-runs. The judge also
    # prints entity-resolution chatter to stdout -- swallow it.
    def _guarded(fn, *a):
        random.seed(item_id)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return float(fn(*a))
        except Exception:  # noqa: BLE001 -- a judge crash is a 0, not a dead run
            return 0.0

    acc_o = _guarded(viki_2.acc_reward, response, gt)
    fmt = float(viki_2.format_reward(response))
    if plan is None:
        return acc_o, 0.0, fmt, why
    acc_l = _guarded(compute_locus_l2_acc, plan, gt)

    empty = not viki_2.transform_actions(plan)
    if null_task:
        why = "correct_null" if empty else "missed_null"
    elif empty:
        why = "spurious_null"
    else:
        why = "correct" if acc_l >= 1.0 else "wrong_plan"
    return acc_o, acc_l, fmt, why


# --------------------------------------------------------------------------
# per-file pass
# --------------------------------------------------------------------------

def score_file(path: pathlib.Path) -> dict:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not recs:
        return {}
    model = recs[0]["model"]
    condition = recs[0]["condition"]
    api_source = recs[0].get("api_source", "unknown")

    ok = {}
    stale = collections.Counter()
    rewritten = _rewritten_rows()
    for r in recs:
        if r["status"] != "ok":
            continue
        item_id = r["item_id"]
        row = gt_table(r["split"])[r["row"]]
        sha = r.get("prompt_sha")
        # §7.6 rewrote 66 instructions after a collection had already answered
        # the old wording. Such a response is not a model failure and must not be
        # scored -- it answered a question that is no longer being asked. Two
        # ways to detect one: a stamp that disagrees with the current text, or no
        # stamp at all on a row the repair reports list as rewritten. Among the
        # rows that survive, a stamped one always beats an unstamped one, so the
        # choice never depends on line order.
        if sha is not None and sha != row["prompt_sha"]:
            stale[item_id] += 1
            continue
        if sha is None and r["row"] in rewritten.get(r["split"], ()):
            stale[item_id] += 1
            continue
        prev = ok.get(item_id)
        if prev is not None and prev.get("prompt_sha") is not None and sha is None:
            continue
        ok[item_id] = r
    attempted = {r["item_id"] for r in recs}
    unverified = sum(1 for r in ok.values() if r.get("prompt_sha") is None)

    per_item, outcomes = {}, collections.Counter()
    for item_id, r in sorted(ok.items()):
        row = gt_table(r["split"])[r["row"]]
        acc_o, acc_l, fmt, why = score_one(r["response"], row["gt"], item_id)
        outcomes[why] += 1
        per_item[item_id] = {"split": r["split"], "row": r["row"],
                             "acc": acc_l, "acc_official": acc_o,
                             "format": fmt, "cell": row["cell"],
                             "family": row["family"], "task_name": row["task_name"],
                             "null_task": row["null_task"], "outcome": why}

    splits = sorted({v["split"] for v in per_item.values()} | {i.split(":")[0] for i in attempted})
    coverage = {}
    for s in splits:
        total = len(gt_table(s))
        got = sum(1 for v in per_item.values() if v["split"] == s)
        stale_only = sum(1 for i in stale if i.split(":")[0] == s and i not in ok)
        coverage[s] = {"collected": got, "total": total,
                       "missing": total - got, "stale_text": stale_only}

    return {"model": model, "condition": condition, "api_source": api_source,
            "file": str(path), "per_item": per_item, "outcomes": dict(outcomes),
            "coverage": coverage,
            "prompt_text": {"stale_rows_dropped": sum(stale.values()),
                            "items_with_stale_rows": len(stale),
                            "items_scored_unstamped": unverified}}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


def report(res: dict) -> None:
    items = res["per_item"]
    cov = res["coverage"]
    tot_c = sum(c["collected"] for c in cov.values())
    tot_n = sum(c["total"] for c in cov.values())
    print(f"\n=== {res['model']} / {res['condition']}   [{res['api_source']}] ===")
    n_stale = sum(c.get("stale_text", 0) for c in cov.values())
    print(f"  coverage        : {tot_c}/{tot_n} collected"
          f"  ({tot_n - tot_c - n_stale} missing = relay holes, {n_stale} stale text"
          f"; both excluded from means)")
    pt = res.get("prompt_text") or {}
    if pt.get("stale_rows_dropped") or n_stale:
        print(f"  prompt text     : {pt.get('stale_rows_dropped', 0)} rows dropped as "
              f"stale (§7.6 rewrite); {n_stale} items now have NO valid row -- "
              f"re-collect with --only-rows")
    if not items:
        print("  (nothing scored)")
        return
    print(f"  acc  (LOCUS)    : {_mean(v['acc'] for v in items.values()):.3f}"
          f"   over {tot_c} collected")
    print(f"  acc  (official) : {_mean(v['acc_official'] for v in items.values()):.3f}"
          f"   [viki_2 verbatim; 0 by construction on null-plan items]")
    nulls = [v for v in items.values() if v["null_task"]]
    if nulls:
        print(f"  null-plan cells : {_mean(v['acc'] for v in nulls):.3f}  n={len(nulls)}"
              f"   (goal already holds at init; correct answer = empty plan)")
    print(f"  format reward   : {_mean(v['format'] for v in items.values()):.3f}")
    print(f"  outcomes        : {res['outcomes']}")

    print("  per split       :")
    for s in sorted(cov):
        sub = [v for v in items.values() if v["split"] == s]
        m = f"{_mean(v['acc'] for v in sub):.3f}" if sub else "  n/a"
        print(f"    {s:<8} acc {m}   n={len(sub)}/{cov[s]['total']}"
              + (f"  (missing {cov[s]['missing']})" if cov[s]["missing"] else ""))

    cells = collections.defaultdict(list)
    for v in items.values():
        cells[v["cell"]].append(v["acc"])
    s_vals = sorted({c[0] for c in cells if c[0] is not None})
    m_vals = sorted({c[1] for c in cells if c[1] is not None})
    if s_vals and m_vals:
        print("  per cell (rows = S depth, cols = M depth; '.' = not collected):")
        print("        " + "".join(f"  m{m:<5}" for m in m_vals))
        for s in s_vals:
            cline = f"    s{s} "
            for m in m_vals:
                vals = cells.get((s, m))
                cline += f"  {_mean(vals):.2f}({len(vals):>2})" if vals else "  .     "
            print(cline)

    fams = collections.defaultdict(list)
    for v in items.values():
        fams[v["family"]].append(v["acc"])
    print("  per family      : " + "  ".join(
        f"{f}={_mean(a):.2f}(n{len(a)})" for f, a in sorted(fams.items(), key=lambda kv: str(kv[0]))))


def report_pairs(results: list[dict]) -> None:
    """S-axis isolate: oracle_text - image, on items completed under BOTH."""
    by_model = collections.defaultdict(dict)
    for r in results:
        by_model[r["model"]][r["condition"]] = r
    printed = False
    for model, conds in sorted(by_model.items()):
        if "image" not in conds or "oracle_text" not in conds:
            continue
        img, txt = conds["image"]["per_item"], conds["oracle_text"]["per_item"]
        shared = sorted(set(img) & set(txt))
        if not shared:
            continue
        if not printed:
            print("\n=== S-axis isolate: oracle_text - image (paired) ===")
            printed = True
        a_i, a_t = _mean(img[k]["acc"] for k in shared), _mean(txt[k]["acc"] for k in shared)
        print(f"  {model:<16} image {a_i:.3f}  oracle_text {a_t:.3f}  "
              f"delta {a_t - a_i:+.3f}   paired n={len(shared)}")
        for s in sorted({img[k]["split"] for k in shared}):
            sub = [k for k in shared if img[k]["split"] == s]
            di = _mean(txt[k]["acc"] for k in sub) - _mean(img[k]["acc"] for k in sub)
            print(f"    {s:<8} delta {di:+.3f}  n={len(sub)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(_DEFAULT_DIR))
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--json", default=None, help="also dump per-item scores here")
    args = ap.parse_args()

    paths = sorted(pathlib.Path(args.dir).glob(args.glob))
    if not paths:
        print(f"no collections under {args.dir}/{args.glob}")
        return 1
    results = []
    for p in paths:
        res = score_file(p)
        if res:
            results.append(res)
            report(res)
    report_pairs(results)

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "per_item"} |
             {"per_item": {k: dict(v, cell=list(v["cell"])) for k, v in r["per_item"].items()}}
             for r in results], ensure_ascii=False, indent=1))
        print(f"\nper-item scores -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
