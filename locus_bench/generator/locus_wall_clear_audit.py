"""Size the re-render delta set for the lateral wall-clearance pass.

``locus_render_pack.clear_of_walls`` is a no-op for any body that cannot
reach a wall, so a config in which it moves nothing re-renders byte-identical
and does not need re-rendering at all. This replays the pass over the configs
that were **actually rendered** (``data/lnsdl_bench/<pack>/<split>/configs``)
rather than over freshly generated instances, so the answer is about the
shipped images, not about a re-sampling that might not reproduce them.

The count is exact, not an estimate: if every item in a config comes back
with ``shift == 0`` the whole downstream (cases (a)/(b)/(c), the render RNG)
sees identical input and produces the identical image. The 349-image figure
in docs/YOLO.md is a different quantity -- it counted images whose depth
buffer showed a violation for the one placement that was drawn, whereas this
counts configs whose reach box admits one.

Cabinet-anchored items are excluded the way the pass excludes them, detected
here by their body lying inside ``_CABINET_INTERIOR`` (the rendered config no
longer records which position area an item resolved from).

    python script_general/locus_wall_clear_audit.py \
        --pack /path/to/NSDL/data/lnsdl_bench/ramp63k \
        --out /tmp/wall_clear_audit.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from collections import Counter

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from locus_render_pack import (          # noqa: E402
    _CABINET_INTERIOR, _WALL_PLANES, clear_of_walls, rotated_xy_box)

SPLITS = ("train", "test_id", "ood_s", "ood_m", "ood_x")


def _in_cabinet(item_cfg, interior, assets_root):
    """Does this item's whole reach box sit inside the cabinet carcass?

    Proxy for "case (e) owns this item". Conservative in the safe direction:
    a false positive only means the audit declines to count an item that the
    pass also declines to touch.
    """
    if interior is None:
        return False
    visual = item_cfg.get("visual") or {}
    filename = visual.get("filename")
    if not filename:
        return False
    path = os.path.join(assets_root, filename)
    if not os.path.isfile(path):
        return False
    rel = rotated_xy_box(path, visual.get("scale", 1.0),
                         item_cfg["pos"]["qpos"])
    p = item_cfg["pos"]["ppos"]["p"]
    rp = item_cfg["pos"].get("randp_scale") or [0.0, 0.0, 0.0]
    for ax in (0, 1):
        lo = min(p[ax], p[ax] + rp[ax]) + rel[ax][0]
        hi = max(p[ax], p[ax] + rp[ax]) + rel[ax][1]
        if lo < interior[ax][0] - 1e-9 or hi > interior[ax][1] + 1e-9:
            return False
    return True


def audit_config(path, assets_root):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    layout = int(cfg["scene"]["env"]["layout_idx"])
    walls = _WALL_PLANES[layout]
    interior = _CABINET_INTERIOR.get(layout)
    moved = []
    for item in cfg.get("objects") or []:
        if _in_cabinet(item, interior, assets_root):
            continue
        # deep-copied: the pass rewrites in place and we must not disturb
        # the shipped config, which is still the render's input of record.
        rep = clear_of_walls(copy.deepcopy(item), walls, assets_root)
        if rep["shift"] > 0:
            rep["asset"] = (item.get("visual") or {}).get("filename")
            moved.append(rep)
    return layout, moved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True,
                    help="dataset root holding <split>/configs/*.yaml")
    ap.add_argument("--assets-root", default=_ROBOFACTORY)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    per_split, delta, by_asset, by_layout, by_wall = {}, [], Counter(), Counter(), Counter()
    worst = []
    for split in args.splits:
        paths = sorted(glob.glob(os.path.join(args.pack, split,
                                              "configs", "*.yaml")))
        hits = 0
        for path in paths:
            layout, moved = audit_config(path, args.assets_root)
            if not moved:
                continue
            hits += 1
            stem = os.path.splitext(os.path.basename(path))[0]
            delta.append(stem)
            by_layout[layout] += 1
            for rep in moved:
                by_asset[rep["asset"]] += 1
                for s in rep["walls"]:
                    by_wall[s] += 1
                worst.append((rep["shift"], stem, rep["name"], rep["asset"]))
        per_split[split] = {"configs": len(paths), "needs_rerender": hits}
        print(f"{split}: {hits}/{len(paths)} configs move", flush=True)

    worst.sort(reverse=True)
    out = {"per_split": per_split,
           "delta_set": sorted(delta),
           "by_layout": dict(sorted(by_layout.items())),
           "by_asset": by_asset.most_common(),
           "by_wall": by_wall.most_common(),
           "worst_shifts": [{"shift": round(s, 4), "config": c,
                             "item": n, "asset": a}
                            for s, c, n, a in worst[:25]]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    total = sum(v["needs_rerender"] for v in per_split.values())
    n_cfg = sum(v["configs"] for v in per_split.values())
    print(f"\ntotal {total}/{n_cfg} configs need re-render "
          f"({100.0 * total / max(n_cfg, 1):.1f}%)")
    print(f"by asset: {by_asset.most_common(10)}")
    print(f"by wall:  {by_wall.most_common()}")
    print(f"by layout: {dict(sorted(by_layout.items()))}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
