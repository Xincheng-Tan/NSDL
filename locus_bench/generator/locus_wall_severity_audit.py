"""How many of the wall-clearance delta-set configs are ACTUALLY defective?

``locus_wall_clear_audit.py`` counts configs whose reach box admits a
protruding placement. That is the right criterion for "must re-render" (the
render draws a fresh sample, so a config with an illegal corner can produce a
defective image) but it is the WRONG criterion for "is the shipped image
defective": the shipped image drew exactly one point out of that box.

So this replays the same pass and, instead of a boolean, reports the measure
of the reach box that is illegal. Official placement draws
``ppos + randp_scale * rand(3)`` with ``rand`` uniform and independent per
axis, so for one item

    P(drawn placement protrudes) = 1 - prod_ax |band_ax n feasible_ax| / |band_ax|

and a degenerate axis (``randp_scale == 0``) contributes 1 or 0. Per config
the items are drawn independently, so ``p_any = 1 - prod_items (1 - p_item)``
and ``sum_configs p_any`` is the expected number of shipped images that
actually contain a protruding body.

That number is directly comparable with the two human readings on record:
the 20-image scissors probe (y = 4/20) and the BASELINE layer (0 defects /
156 boxes, 95% upper bound 1.9%). ``--only`` restricts the run to a list of
config stems so the probe images can be scored on their own and used to
calibrate the model rather than trusting it.

Depth is reported alongside, both absolute and as a fraction of the body's
own extent along the violated axis, because a 1 cm bite out of a 30 cm loaf
is not the same defect as losing half a pair of scissors.

    python script_general/locus_wall_severity_audit.py \
        --pack /path/to/NSDL/data/lnsdl_bench/ramp63k \
        --out /tmp/wall_severity.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from collections import Counter, defaultdict

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from locus_render_pack import (          # noqa: E402
    _CABINET_INTERIOR, _WALL_PLANES, clear_of_walls, hull_xy_box,
    rotated_xy_box)
from locus_wall_clear_audit import _in_cabinet, SPLITS   # noqa: E402


def _axis_stats(band, feas):
    """(p_legal, E[protrusion], max protrusion) for one axis.

    ``protrusion(x) = max(0, f0 - x, x - f1)`` -- how far past the wall plane
    the body's near face sits when the origin is drawn at ``x``.
    """
    b0, b1 = band
    f0, f1 = feas
    lo_pen = max(0.0, f0 - b0) if f0 != float("-inf") else 0.0
    hi_pen = max(0.0, b1 - f1) if f1 != float("inf") else 0.0
    max_pen = max(lo_pen, hi_pen)
    length = b1 - b0
    if length <= 1e-12:                  # degenerate: randp_scale 0 on this axis
        pen = 0.0
        if f0 != float("-inf"):
            pen = max(pen, f0 - b0)
        if f1 != float("inf"):
            pen = max(pen, b0 - f1)
        pen = max(0.0, pen)
        return (0.0 if pen > 0 else 1.0), pen, pen
    lo = max(b0, f0)
    hi = min(b1, f1)
    legal = max(0.0, hi - lo)
    # integral of the two linear ramps outside [f0, f1], clipped to the band
    area = 0.0
    if f0 != float("-inf"):
        x = min(b1, f0)
        if x > b0:
            area += 0.5 * (f0 - b0 + f0 - x) * (x - b0)
    if f1 != float("inf"):
        x = max(b0, f1)
        if b1 > x:
            area += 0.5 * (b1 - f1 + x - f1) * (b1 - x)
    return legal / length, area / length, max_pen


def score_config(path, assets_root):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    layout = int(cfg["scene"]["env"]["layout_idx"])
    walls = _WALL_PLANES[layout]
    interior = _CABINET_INTERIOR.get(layout)
    items = []
    for item in cfg.get("objects") or []:
        if _in_cabinet(item, interior, assets_root):
            continue
        rep = clear_of_walls(copy.deepcopy(item), walls, assets_root)
        if rep["shift"] <= 0:
            continue
        visual = item.get("visual") or {}
        rel = hull_xy_box(os.path.join(assets_root, visual["filename"]),
                          visual.get("scale", 1.0), item["pos"]["qpos"])
        p_legal, e_pen, max_pen, worst_ax = 1.0, 0.0, 0.0, 0
        for ax in (0, 1):
            pl, ep, mp = _axis_stats(rep["band"][ax], rep["feas"][ax])
            p_legal *= pl
            e_pen += ep
            if mp > max_pen:
                max_pen, worst_ax = mp, ax
        extent = rel[worst_ax][1] - rel[worst_ax][0]
        items.append({
            "name": rep["name"],
            "asset": visual.get("filename"),
            "p_violate": 1.0 - p_legal,
            "e_depth": e_pen,
            "max_depth": max_pen,
            "extent": extent,
            "max_frac": (max_pen / extent) if extent > 1e-9 else 0.0,
        })
    return layout, items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--assets-root", default=_ROBOFACTORY)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--only", help="file of config stems, one per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    keep = None
    if args.only:
        with open(args.only) as f:
            keep = {ln.strip() for ln in f if ln.strip()}

    per_split, per_config = {}, {}
    by_asset_p = defaultdict(float)
    by_asset_n = Counter()
    for split in args.splits:
        paths = sorted(glob.glob(os.path.join(args.pack, split,
                                              "configs", "*.yaml")))
        exp, hits = 0.0, 0
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if keep is not None and stem not in keep:
                continue
            layout, items = score_config(path, args.assets_root)
            if not items:
                continue
            hits += 1
            p_any = 1.0
            for it in items:
                p_any *= (1.0 - it["p_violate"])
                by_asset_p[it["asset"]] += it["p_violate"]
                by_asset_n[it["asset"]] += 1
            p_any = 1.0 - p_any
            exp += p_any
            per_config[stem] = {
                "layout": layout,
                "p_any": round(p_any, 6),
                "max_depth": round(max(i["max_depth"] for i in items), 4),
                "max_frac": round(max(i["max_frac"] for i in items), 4),
                "e_depth": round(sum(i["e_depth"] for i in items), 4),
                "items": [{k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in it.items()} for it in items],
            }
        per_split[split] = {"configs": len(paths), "in_delta": hits,
                            "expected_defective": round(exp, 2)}
        print(f"{split}: delta {hits}, expected actually-protruding "
              f"{exp:.1f}", flush=True)

    out = {"per_split": per_split,
           "per_config": per_config,
           "by_asset": {a: {"configs": by_asset_n[a],
                            "expected_defective": round(p, 2)}
                        for a, p in sorted(by_asset_p.items(),
                                           key=lambda kv: -kv[1])}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    d = sum(v["in_delta"] for v in per_split.values())
    e = sum(v["expected_defective"] for v in per_split.values())
    print(f"\ndelta {d} configs -> expected {e:.1f} actually-protruding images "
          f"({100.0 * e / max(d, 1):.1f}% of the delta set)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
