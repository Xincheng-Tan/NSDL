#!/usr/bin/env python3
"""Enlarge the held-in RL monitoring probe (protocol D, docs/plan.md §P5.3'.2b).

Why not just re-run ``prepare_rl_data.py --probe-size 512``: that rebuilds the
whole v3 parquet through the scene/AIG/prompt pipeline, and the RL parquet v3 is
frozen (byte-parity with the v3.3 SFT prompts was measured, not assumed). This
script derives the enlarged probe *from the frozen train.parquet* instead, so
train/test are never touched and prompt bytes are carried over verbatim.

Two problems with the current 128-row probe, both of which this fixes:

1. **n is too small to read.** nonskip n=64 at p~.5 gives a 95% CI of +-12.3
   accuracy points, while the step-to-step trajectory swing is 19-23 points --
   i.e. the probe cannot distinguish a mode difference from its own noise.

2. **The prefix rule is family-biased.** ``prepare_rl_data._emit_lnsdl_probe``
   takes ``nonskip_rows[:64]`` after sorting by ``task_id``, and task_id sorts on
   the *scene* prefix (``2361_6-1``) while the task family is the *suffix*
   (``6-1``). The two are correlated, so the prefix is not a representative
   sample and does not become one as n grows: family ``10-1`` is 14.1% of the 64
   against 4.3% of train (3.3x over), ``13-2``/``8-1``/``8-2``/``12-3`` are ~3x
   under. ``10-1`` is precisely the family that carries the v3.3 stacking
   regression (docs/debug_sft.md), so the old probe over-weights the one family
   whose behaviour is already known to be atypical.

Selection rule (deterministic, no RNG):

* skip: **all** 144 skip rows in train. The stratum is exhausted, so its reading
  carries no sampling error against train at all. (It is 2% of the official
  distribution; it stays a separate data_source and is never pooled.)
* nonskip: N rows allocated across families in proportion to the *full* train
  nonskip family distribution (largest-remainder), with the current probe's 64
  rows forced in, then filled in task_id order within each family.

The forced-in clause makes the result a strict **superset** of the current
probe, so any per-sample reading from an old run stays recoverable on the same
task_ids. It does *not* make old and new aggregate numbers comparable: batch
composition changes the bf16 reduction order in the rollout, which is worth up
to 15.6 accuracy points on its own (docs/debug_rl.md §1.3). Compare arms within
one probe file, never across two.

Writes a NEW file (default ``probe512.parquet``); ``probe.parquet`` is left
alone so in-flight runs and the frozen dataset are unaffected.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def _gt(r) -> dict:
    return (r or {}).get("ground_truth", {}) or {}


def _tid(r) -> str:
    return str(_gt(r).get("task_id") or "")


def _is_skip(r) -> bool:
    return bool(_gt(r).get("has_skip_decision"))


def _family(task_id: str) -> str:
    """``2361_6-1`` -> ``6-1``. The suffix is the task family; the prefix is the scene."""
    return task_id.rsplit("_", 1)[-1]


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Allocate ``total`` integer slots proportional to ``weights``.

    Largest-remainder rather than rounding each independently, so the parts sum
    to exactly ``total`` instead of drifting by a few rows.
    """
    exact = {k: total * w for k, w in weights.items()}
    base = {k: int(v) for k, v in exact.items()}
    left = total - sum(base.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:left]:
        base[k] += 1
    return base


def build(train_path: Path, cur_probe: Path | None, n_nonskip: int) -> tuple[pd.DataFrame, dict]:
    tr = pd.read_parquet(train_path)
    tr = tr.assign(
        _tid=tr["reward_model"].map(_tid),
        _skip=tr["reward_model"].map(_is_skip),
    )
    tr["_fam"] = tr["_tid"].map(_family)
    tr = tr.sort_values("_tid", kind="mergesort")

    skip = tr[tr._skip]
    nonskip = tr[~tr._skip]
    if n_nonskip > len(nonskip):
        raise SystemExit(f"asked for {n_nonskip} nonskip rows, train has {len(nonskip)}")

    forced: set[str] = set()
    if cur_probe is not None and cur_probe.exists():
        cp = pd.read_parquet(cur_probe)
        cp_tid = cp["reward_model"].map(_tid)
        cp_skip = cp["reward_model"].map(_is_skip)
        forced = set(cp_tid[~cp_skip])
        missing = forced - set(nonskip._tid)
        if missing:
            raise SystemExit(f"current probe has {len(missing)} nonskip rows absent from train")

    fam_p = (nonskip._fam.value_counts() / len(nonskip)).to_dict()
    target = _largest_remainder(fam_p, n_nonskip)
    have = Counter(_family(t) for t in forced)

    # max(target, have) can only overshoot the budget if the old probe already
    # over-fills a family beyond its share of N. Fail loudly rather than silently
    # returning a set that is neither proportional nor a superset.
    take = {f: max(target.get(f, 0), have.get(f, 0)) for f in set(target) | set(have)}
    over = sum(take.values()) - n_nonskip
    if over > 0:
        raise SystemExit(
            f"n_nonskip={n_nonskip} is too small to stay a superset of the current probe "
            f"(need >= {sum(take.values())}); families over target: "
            + ", ".join(f"{f}(have {have[f]} > target {target.get(f, 0)})"
                        for f in sorted(take) if have.get(f, 0) > target.get(f, 0))
        )

    picked: list[str] = []
    for fam in sorted(take):
        rows = nonskip[nonskip._fam == fam]
        keep = [t for t in rows._tid if t in forced]
        rest = [t for t in rows._tid if t not in forced]
        picked.extend((keep + rest)[: take[fam]])

    chosen = pd.concat([skip, nonskip[nonskip._tid.isin(picked)]]).sort_values("_tid", kind="mergesort")
    chosen = chosen.copy()
    # Same data_source tagging as prepare_rl_data._emit_lnsdl_probe: verl reports
    # the two strata as separate data_sources instead of pooling a 2%-prevalence
    # family into the headline number.
    chosen["data_source"] = ["lnsdl_skip" if s else "lnsdl_nonskip" for s in chosen._skip]

    stats = {
        "n_total": int(len(chosen)),
        "n_skip": int(chosen._skip.sum()),
        "n_nonskip": int((~chosen._skip).sum()),
        "n_forced_kept": int(len(forced & set(chosen._tid))),
        "n_forced": int(len(forced)),
        "family_target": {k: int(v) for k, v in sorted(target.items())},
        "family_actual": {k: int(v) for k, v in
                          sorted(Counter(_family(t) for t in chosen[~chosen._skip]._tid).items())},
        "family_train_frac": {k: round(v, 4) for k, v in sorted(fam_p.items())},
    }
    stats["family_l1_dev"] = round(sum(
        abs(stats["family_actual"].get(f, 0) / max(stats["n_nonskip"], 1) - p)
        for f, p in fam_p.items()), 4)
    task_ids = list(chosen._tid)
    return chosen.drop(columns=["_tid", "_skip", "_fam"]), {**stats, "task_ids": task_ids}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/lnsdl/rl/v3/viki_l2_train")
    ap.add_argument("--n-nonskip", type=int, default=368,
                    help="nonskip rows; all skip rows are always taken (144 in v3)")
    ap.add_argument("--out-name", default="probe512.parquet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = Path(args.data_dir)
    df, stats = build(d / "train.parquet", d / "probe.parquet", args.n_nonskip)

    old = pd.read_parquet(d / "probe.parquet")
    old_tids = set(old["reward_model"].map(_tid))
    superset = old_tids <= set(stats["task_ids"])
    print(f"rows={stats['n_total']} (skip {stats['n_skip']} / nonskip {stats['n_nonskip']})")
    print(f"superset of current probe: {superset}  "
          f"(forced {stats['n_forced_kept']}/{stats['n_forced']} nonskip + all skip)")
    print(f"family L1 deviation from train nonskip: {stats['family_l1_dev']:.3f} "
          f"(current 128-row probe: 0.469)")
    worst = sorted(stats["family_actual"].items(),
                   key=lambda kv: -abs(kv[1] / stats["n_nonskip"] - stats["family_train_frac"].get(kv[0], 0)))[:5]
    for f, c in worst:
        print(f"  {f:>5}: {c:3d} = {c / stats['n_nonskip']:.3f} vs train {stats['family_train_frac'].get(f, 0):.3f}")
    if not superset:
        raise SystemExit("refusing to write: not a superset of the current probe")
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    out = d / args.out_name
    df.to_parquet(out)
    ids = d / (Path(args.out_name).stem + "_task_ids.json")
    ids.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)\nwrote {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
