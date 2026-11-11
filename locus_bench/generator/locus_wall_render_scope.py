"""Derive the re-render scope for the wall-clearance repair round.

``locus_wall_severity_audit.py`` scores every config in the delta set; this
turns that score into the stem whitelist ``locus_pilot --wall-clear-scope``
consumes. Keeping the two apart matters because the audit is expensive (it
loads every mesh) while the tier choice is a judgement call that gets revised:
docs/YOLO.md §6.2.5 compares six tiers and settles on ``>1/3 body length,
restricted to the four large-overhang assets`` = 1111 configs, because it
leaves the same human-visible residue as the 1591 tier at 30% of the render
cost and, unlike the two full-coverage tiers, does not overwrite any of the
192 manually drawn glass-cabinet boxes.

Both halves of the criterion are load-bearing and neither is a proxy for the
other. The asset filter is what keeps the intersection with the manual boxes
at zero (all 21 collisions come from small-overhang assets). The 1/3 threshold
is calibrated, not chosen: replayed against the 20 hand-judged scissors images
it predicts 3.80 defects against 4 observed, where ``>0`` predicts 10.29 and
``>1/2`` predicts 1.23.

    python script_general/locus_wall_render_scope.py \
        --severity /tmp/wall_severity.json \
        --out configs/locus/wall_clear_scope_1111.txt
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os

# The four assets whose overhang is large enough to be worth a re-render, per
# the per-asset medians in docs/YOLO.md §6.2.4. `hammer_annotated` and
# `single_shoe_annotated` are deliberately absent: their overhang medians are
# 1.5-6.9 cm and they are the assets that collide with the manual boxes.
DEFAULT_ASSETS = ("scissors", "bread", "banana", "kettle")
DEFAULT_FRAC = 1.0 / 3.0


def asset_name(path):
    """``assets/objects/bread/new.obj`` -> ``bread``.

    The basename is the mesh variant (``new`` / ``base``), never the asset, so
    matching on it silently selects nothing.
    """
    return os.path.basename(os.path.dirname(path)) if path else ""


def select(per_config, assets, min_frac):
    """Stems with at least one listed asset buried deeper than ``min_frac``.

    Selection is per CONFIG even though the criterion is per item: the pass
    runs over a whole scene, and a config that is re-rendered anyway may as
    well have its small-overhang bodies repaired too.
    """
    assets = set(assets)
    return sorted(
        stem for stem, rec in per_config.items()
        if any(asset_name(it["asset"]) in assets and it["max_frac"] > min_frac
               for it in rec["items"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--severity", required=True,
                    help="locus_wall_severity_audit.py --out JSON")
    ap.add_argument("--assets", nargs="*", default=list(DEFAULT_ASSETS))
    ap.add_argument("--min-frac", type=float, default=DEFAULT_FRAC,
                    help="burial depth as a fraction of the body's own extent")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.severity) as f:
        audit = json.load(f)
    per_config = audit["per_config"]
    stems = select(per_config, args.assets, args.min_frac)

    by_split = collections.Counter(s.rsplit("_", 1)[0] for s in stems)
    residual = sum(rec["p_any"] for stem, rec in per_config.items()
                   if stem not in set(stems))
    text = "\n".join(stems) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    digest = hashlib.sha256("\n".join(stems).encode()).hexdigest()
    print(f"delta set {len(per_config)} -> scope {len(stems)} configs")
    print(f"  by split: {dict(sorted(by_split.items()))}")
    print(f"  expected protruding bodies left unrepaired: {residual:.1f}")
    print(f"  sha256 {digest}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
