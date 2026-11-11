"""Measure the closed-cabinet interior box per LOCUS layout.

Source of truth for ``locus_render_pack._CABINET_INTERIOR`` (case (e)).
robocasa resolves fixture geometry through an ``align_to``/``stack`` chain
that only the scene builder walks, so the interior is not readable from the
layout yaml; this builds one env per layout and reads
``Fixture.get_int_sites(all_points=True, relative=False)`` instead.

The reported box is the union, over every CLOSED cabinet class whose interior
meets the layout's declared ``cabinet`` position area in XY and spans its z,
of that fixture's world-frame interior AABB. Open shelving is excluded on
purpose: S4 needs a container that must be OPENED, and layout 5's band happens
to sit inside an ``OpenCabinet`` as well as the hinge cabinet it means.

Run from the RoboFactory directory in the render env::

    python script_general/locus_cabinet_probe.py --out /tmp/cabinet_interiors.json

Needs one renderable config per layout; ``--configs`` takes a JSON list of
``[layout_id, config_path]`` pairs (defaults to generating them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# robocasa classes whose contents are hidden until a door/panel is opened.
CLOSED_CABINET_CLASSES = (
    "HingeCabinet", "SingleCabinet", "PanelCabinet", "HousingCabinet",
)


def declared_band(layout_id, layouts_dir=None):
    """The layout's own ``cabinet`` position area as ((lo, hi) per axis, z)."""
    import yaml

    layouts_dir = layouts_dir or os.path.join(_ROBOFACTORY, "configs/layouts")
    with open(os.path.join(layouts_dir, f"layout_{layout_id}.yaml")) as f:
        cfg = yaml.safe_load(f)
    pos = next(a["pos"] for a in cfg["position_areas"]
               if any("cabinet" in str(n).lower() for n in a["names"]))
    p, rp = pos["ppos"]["p"], pos["randp_scale"]
    return [(min(p[ax], p[ax] + rp[ax]), max(p[ax], p[ax] + rp[ax]))
            for ax in (0, 1)], float(p[2])


def probe_layout(config_path):
    """World-frame interior AABBs of every fixture in one built scene."""
    import gymnasium as gym
    import numpy as np
    import yaml
    import tasks  # noqa: F401  registers VikiBenchTask-rf

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env = gym.make(cfg["task_name"] + "-rf", config=config_path,
                   obs_mode="none", control_mode="pd_joint_pos",
                   render_mode="rgb_array", reward_mode="dense",
                   sim_backend=os.environ.get("VIKI_SIM_BACKEND", "gpu"))
    out = []
    try:
        env.reset(seed=0)
        for sd in env.unwrapped.scene_builder.scene_data:
            for role, fixture in sd["fixtures"].items():
                if not hasattr(fixture, "get_int_sites"):
                    continue
                try:
                    pts = np.array([np.asarray(s, dtype=float) for s in
                                    fixture.get_int_sites(all_points=True,
                                                          relative=False)])
                except Exception:
                    continue            # fixture without interior bounds sites
                out.append({"role": role, "cls": type(fixture).__name__,
                            "lo": pts.min(0).tolist(),
                            "hi": pts.max(0).tolist()})
    finally:
        env.close()
    return out


def interior_for_layout(layout_id, fixtures, z_pad=0.08):
    """Union interior box of the closed cabinets meeting the declared band."""
    band, z = declared_band(layout_id)
    hits = [f for f in fixtures
            if f["cls"] in CLOSED_CABINET_CLASSES
            and f["lo"][2] - z_pad <= z <= f["hi"][2] + z_pad
            and not (band[0][1] < f["lo"][0] or band[0][0] > f["hi"][0])
            and not (band[1][1] < f["lo"][1] or band[1][0] > f["hi"][1])]
    if not hits:
        raise LookupError(
            f"layout {layout_id}: no closed cabinet meets the declared band "
            f"{band} at z={z}")
    box = [(min(f["lo"][ax] for f in hits), max(f["hi"][ax] for f in hits))
           for ax in (0, 1)]
    return {"layout_id": layout_id, "interior": box, "band": band, "z": z,
            "roles": [f["role"] for f in hits],
            "classes": sorted({f["cls"] for f in hits})}


def _default_configs(out_dir):
    """One renderable config per layout in ``locus_task_pool.ALL_LAYOUTS``."""
    import yaml
    import locus_render_pack as pack
    from locus_task_pool import ALL_LAYOUTS

    os.makedirs(out_dir, exist_ok=True)
    insts, _ = pack.generate_split_instances(
        pack.build_split_plan(n_train=600, n_test_id=180, n_ood=380), seed=0)
    picked = {}
    for rows in insts.values():
        for inst in rows:
            lid = int(inst["layout_id"])
            if lid in ALL_LAYOUTS and lid not in picked:
                cfg, _style = pack.build_render_config(inst)
                path = os.path.join(out_dir, f"probe_layout_{lid}.yaml")
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f)
                picked[lid] = path
        if len(picked) == len(ALL_LAYOUTS):
            break
    missing = sorted(set(ALL_LAYOUTS) - set(picked))
    if missing:
        raise LookupError(f"no instance sampled for layouts {missing}")
    return [[lid, picked[lid]] for lid in sorted(picked)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", help="JSON list of [layout_id, config_path]")
    ap.add_argument("--config-dir", default="/tmp/locus_cabinet_probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    pairs = (json.load(open(args.configs)) if args.configs
             else _default_configs(args.config_dir))
    report = {}
    for layout_id, config_path in pairs:
        fixtures = probe_layout(config_path)
        report[str(layout_id)] = interior_for_layout(int(layout_id), fixtures)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"wrote {args.out}")
    print("_CABINET_INTERIOR = {")
    for key in sorted(report, key=int):
        box = report[key]["interior"]
        roles = "+".join(report[key]["roles"])
        print(f"    {key}: (({box[0][0]:.3f}, {box[0][1]:.3f}), "
              f"({box[1][0]:.3f}, {box[1][1]:.3f})),   # {roles}")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
