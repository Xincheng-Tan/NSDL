"""Bring stale on-disk config text back in line with the generator, no re-render.

The 2026-08-03 goal-destination repair (docs/YOLO.md 6.3, "modify the text,
keep the sample -- zero re-render") rewrote the task description in the
generator AND in each split's ``data.parquet``, but never rewrote the config
YAMLs the pack was rendered from. 276 configs therefore still carry the
pre-repair sentence while the parquet the model reads carries the repaired one.

That divergence is invisible until something regenerates a config and compares:
``render_batch`` treats ANY text mismatch as stale and DELETES the image and
``gt_labels`` before re-rendering. So a scoped wall-clearance run that should
touch 1111 configs instead invalidates 1357 -- and 100 of the extra 246 are
frames a human reviewed for the glass cabinet, 96 of them carrying
hand-added boxes that the delete would destroy.

Re-rendering them would also be pointless: the divergence is confined to
``gt.description``, which no part of the renderer reads. The geometry that
produced the existing pixels is identical. So the repair is to rewrite the
text on disk and leave the image alone.

The guard that makes this safe is the refusal below: a config is rewritten
only if a field-by-field comparison against the freshly generated one shows
``gt.description`` as the ONLY difference. Anything else means the generator
moved geometry too, the pixels really are stale, and the config must be left
for the renderer to handle rather than quietly stamped as current.

    python script_general/locus_refresh_stale_descriptions.py \
        --pack /path/to/NSDL/data/lnsdl_bench/ramp63k \
        --scope configs/locus/wall_clear_scope_1111.txt --apply
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

DESCRIPTION_PATH = ".gt.description"


def leaf_diff(a, b, path=""):
    """Every leaf path at which two loaded configs disagree."""
    if type(a) is not type(b):
        return [path or "<root>"]
    out = []
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                out += leaf_diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}[len]"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += leaf_diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--scope", help="stems the wall-clearance pass will be "
                                    "applied to; they are re-rendered anyway, "
                                    "so they are left untouched here")
    ap.add_argument("--n-train", type=int, default=210)
    ap.add_argument("--n-test-id", type=int, default=6)
    ap.add_argument("--n-ood", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true",
                    help="without this nothing is written")
    ap.add_argument("--backup", help="JSON of the replaced descriptions")
    args = ap.parse_args(argv)

    scope = set()
    if args.scope:
        with open(args.scope) as f:
            scope = {ln.strip() for ln in f if ln.strip()}

    plan = build_split_plan(n_train=args.n_train, n_test_id=args.n_test_id,
                            n_ood=args.n_ood)
    splits, _ = generate_split_instances(plan, seed=args.seed)

    refreshed, refused, backup = [], [], {}
    by_split = collections.Counter()
    for split, rows in splits.items():
        for inst in rows:
            stem = render_stem(inst)
            if stem in scope:
                continue
            path = os.path.join(args.pack, split, "configs", f"{stem}.yaml")
            if not os.path.isfile(path):
                continue
            cfg, _ = build_render_config(inst, wall_clear_scope=scope or None)
            text = yaml.dump(cfg)
            with open(path) as f:
                on_disk_text = f.read()
            if on_disk_text == text:
                continue
            on_disk = yaml.safe_load(on_disk_text)
            diff = leaf_diff(on_disk, cfg)
            if diff != [DESCRIPTION_PATH]:
                refused.append({"stem": stem, "fields": sorted(diff)[:8]})
                continue
            backup[stem] = on_disk["gt"]["description"]
            refreshed.append(stem)
            by_split[split] += 1
            if args.apply:
                with open(path, "w") as f:
                    f.write(text)
        print(f"[refresh] {split} scanned", flush=True)

    print(f"\nconfigs with description-only drift : {len(refreshed)}")
    print(f"  by split: {dict(sorted(by_split.items()))}")
    print(f"configs REFUSED (geometry moved too): {len(refused)}")
    for r in refused[:10]:
        print(f"  {r['stem']}: {r['fields']}")
    print("\n" + ("REWRITTEN" if args.apply else "DRY RUN - nothing written"))

    if args.backup and args.apply:
        with open(args.backup, "w") as f:
            json.dump(backup, f, indent=1)
        print(f"wrote {args.backup}")
    # a refusal is the signal that this script is the wrong tool for that
    # config, not that it failed; surface it as a non-zero exit either way.
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
