"""Pre-flight for a scoped wall-clearance re-render, before anything is deleted.

`render_batch` decides what to re-render by comparing the config it generates
against the config on disk, and a mismatch DELETES the existing snapshot. That
makes the scope's blast radius worth measuring before the run rather than
after: this rebuilds every config in the pack, in memory, and reports

  1. how many configs the scope actually invalidates (must equal the tier
     size -- a larger number means the scope is not the only thing that
     changed since the pack was rendered, e.g. an unrelated generator edit,
     and the run would silently re-render more than was approved);
  2. which in-scope items case (f) leaves with no room to jitter. The official
     placement loop `exit(0)`s -- which segfaults sapien mid-build and
     truncates the rest of the render chunk -- when an item with `randp_scale`
     cannot draw a position >=0.15m from all earlier items in 100 tries, so a
     band the pass collapses to a point is the failure mode to look at first;
  3. any item the pass could not make feasible at all.

Nothing is written and the pack is not touched.

    python script_general/locus_wall_scope_preflight.py \
        --pack /path/to/NSDL/data/lnsdl_bench/ramp63k \
        --scope configs/locus/wall_clear_scope_1111.txt \
        --n-train 210 --n-test-id 6 --n-ood 20 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml                                              # noqa: E402

from locus_render_pack import (                          # noqa: E402
    build_render_config, build_split_plan, generate_split_instances,
    render_stem)


def _jitter(item):
    rp = (item.get("pos") or {}).get("randp_scale")
    if rp is None:
        return None
    return tuple(round(float(v), 9) for v in rp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--n-train", type=int, default=210)
    ap.add_argument("--n-test-id", type=int, default=6)
    ap.add_argument("--n-ood", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="optional JSON report")
    args = ap.parse_args(argv)

    with open(args.scope) as f:
        scope = {ln.strip() for ln in f if ln.strip()}

    plan = build_split_plan(n_train=args.n_train, n_test_id=args.n_test_id,
                            n_ood=args.n_ood)
    splits, _ = generate_split_instances(plan, seed=args.seed)
    known = {render_stem(i) for rows in splits.values() for i in rows}
    missing = sorted(scope - known)
    if missing:
        raise SystemExit(f"scope lists {len(missing)} stem(s) absent from the "
                         f"plan (e.g. {missing[:5]})")

    changed, identical, absent = [], 0, []
    collapsed, infeasible = [], []
    for split, rows in splits.items():
        for i, inst in enumerate(rows):
            stem = render_stem(inst)
            cfg, _ = build_render_config(inst, wall_clear_scope=scope)
            text = yaml.dump(cfg)
            path = os.path.join(args.pack, split, "configs", f"{stem}.yaml")
            if not os.path.isfile(path):
                absent.append(stem)
            else:
                with open(path) as f:
                    if f.read() == text:
                        identical += 1
                    else:
                        changed.append(stem)
            if stem not in scope:
                continue
            rep = cfg.get("locus_wall_clear")
            if rep and rep.get("infeasible"):
                infeasible.append({"stem": stem,
                                   "items": rep["infeasible"]})
            # what the same scene looks like with the pass off: any item that
            # had jitter then and has none now lost its whole band to (f).
            off, _ = build_render_config(inst, wall_clear_scope=set())
            before = {o["name"]: _jitter(o) for o in off.get("objects") or []}
            for obj in cfg.get("objects") or []:
                was, now = before.get(obj["name"]), _jitter(obj)
                if was is None or was == now:
                    continue
                if now is None or all(abs(v) < 1e-9 for v in now):
                    collapsed.append({"stem": stem, "item": obj["name"],
                                      "was": was, "now": now})
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{len(rows)}", flush=True)
        print(f"[preflight] {split} done", flush=True)

    by_split = collections.Counter(s.rsplit("_", 1)[0] for s in changed)
    print(f"\nconfigs regenerated identical : {identical}")
    print(f"configs INVALIDATED           : {len(changed)}  "
          f"(scope size {len(scope)})")
    print(f"  by split: {dict(sorted(by_split.items()))}")
    if absent:
        print(f"configs absent from the pack  : {len(absent)} "
              f"(e.g. {absent[:3]})")
    ok = set(changed) == scope
    print(f"invalidated set == scope       : {ok}")
    if not ok:
        extra, short = sorted(set(changed) - scope), sorted(scope - set(changed))
        print(f"  unexpected extra: {len(extra)} (e.g. {extra[:5]})")
        print(f"  in scope but unchanged: {len(short)} (e.g. {short[:5]})")
    print(f"\nitems whose jitter band collapsed to a point: {len(collapsed)} "
          f"in {len({c['stem'] for c in collapsed})} config(s)")
    for c in collapsed[:10]:
        print(f"  {c['stem']} {c['item']}: {c['was']} -> {c['now']}")
    print(f"items the pass could not make feasible: {len(infeasible)}")
    for c in infeasible[:10]:
        print(f"  {c['stem']}: {c['items']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"identical": identical, "changed": sorted(changed),
                       "absent": absent, "scope_matches": ok,
                       "collapsed": collapsed, "infeasible": infeasible},
                      f, indent=1)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
