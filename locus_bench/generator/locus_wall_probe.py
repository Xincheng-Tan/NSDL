"""Measure the room's wall planes per LOCUS layout.

Source of truth for ``locus_render_pack._WALL_PLANES``, which the lateral
clearance pass needs the way ``rest_offset_z`` needs the anchor plane: that
pass answers "how far must this body move so it stops intersecting the
plane it rests on", and its x/y analogue needs the plane a body can be
pushed *through* — the room's walls.

Walls are not readable from the layout yaml: ``ManiSkillSceneBuilder``
rotates and displaces every fixture at scene-assembly time, so only a built
scene knows where a wall ended up. This builds one env per layout and reads
``Wall.pos`` / ``Wall.size`` / ``Wall.wall_side``, exactly as
``locus_cabinet_probe.py`` reads cabinet interiors.

Two things about ``Wall`` (robocasa ``fixtures/others.py``) that this relies
on, both verified against the source:

* ``__init__`` shifts ``pos`` along the side axis and ``build`` shifts it
  back by the same amount, in place — so after ``env.reset`` the surviving
  ``pos`` is both the layout position and the position the plane visual is
  actually built at. That plane is what an object appears to poke through,
  which is the defect we are measuring, so ``pos[axis]`` is the number we
  want. (The box *collision* is centred there and extends ``±size[2]``;
  irrelevant here — the images, not the physics, are what is wrong.)
* ``backing`` walls return from ``build`` before adding any visual, so they
  are skipped: nothing can be seen to intersect them.

The room-interior side of each plane is **checked, not assumed** — see
``planes_for_layout``.

Run from the RoboFactory directory in the render env::

    python script_general/locus_wall_probe.py --out /tmp/wall_planes.json

``--configs`` takes a JSON list of ``[layout_id, config_path]`` pairs and
defaults to generating them via ``locus_cabinet_probe._default_configs``.
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

# side -> (axis, sign of the room interior relative to the plane), i.e. the
# room satisfies ``inward * (coord - plane) >= 0``. Read off the outward
# shifts in ``Wall.__init__`` (back +y, front -y, left -x, right +x), then
# confirmed per layout by ``_check_layout``.
WALL_SIDES = {
    "back":  (1, -1),   # room is y <= plane
    "front": (1, +1),   # room is y >= plane
    "left":  (0, +1),   # room is x >= plane
    "right": (0, -1),   # room is x <= plane
}


def probe_layout(config_path):
    """Every non-backing wall in one built scene, in world coordinates."""
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
                side = getattr(fixture, "wall_side", None)
                if side not in WALL_SIDES:
                    continue            # not a wall, or the floor
                if getattr(fixture, "backing", False):
                    continue            # never built as a visual
                out.append({
                    "role": role,
                    "name": getattr(fixture, "name", role),
                    "side": side,
                    "pos": np.asarray(fixture.pos, dtype=float).tolist(),
                    "size": np.asarray(fixture.size, dtype=float).tolist(),
                })
    finally:
        env.close()
    return out


def _check_layout(layout_id, planes):
    """Sanity-check the interior sign against geometry we already measured.

    ``WALL_SIDES`` was derived by reading shift directions out of
    ``Wall.__init__``; getting it backwards would silently produce a
    clearance pass that shoves every object *into* the wall. Two independent
    checks, both of which fail loudly if the convention is inverted:

    1. opposing walls must bracket a non-empty room;
    2. that room must contain the layout's cabinet interior, which
       ``locus_cabinet_probe`` measured by a completely different route
       (``get_int_sites``).

    Returns a list of human-readable problems; empty means consistent.
    """
    from locus_render_pack import _CABINET_INTERIOR

    problems = []
    room = {}
    for axis, (lo_side, hi_side) in ((0, ("left", "right")),
                                     (1, ("front", "back"))):
        lo = planes.get(lo_side, {}).get("coord")
        hi = planes.get(hi_side, {}).get("coord")
        if lo is not None and hi is not None and lo >= hi:
            problems.append(
                f"{lo_side}={lo:.3f} not below {hi_side}={hi:.3f} "
                f"— axis {axis} interior sign looks inverted")
        room[axis] = (lo, hi)

    interior = _CABINET_INTERIOR.get(int(layout_id))
    if interior is not None:
        for axis in (0, 1):
            lo, hi = room[axis]
            c_lo, c_hi = interior[axis]
            if lo is not None and c_lo < lo - 1e-6:
                problems.append(f"cabinet interior axis {axis} lo {c_lo:.3f} "
                                f"outside wall {lo:.3f}")
            if hi is not None and c_hi > hi + 1e-6:
                problems.append(f"cabinet interior axis {axis} hi {c_hi:.3f} "
                                f"outside wall {hi:.3f}")
    return problems


def planes_for_layout(layout_id, walls):
    """Reduce one layout's walls to the innermost plane per side.

    Innermost = most constraining, so a body cleared against it clears every
    wall on that side. A side with no wall is *absent* rather than infinite:
    the caller must read "no entry" as "unconstrained here", which is the
    honest description of a room that is open on that side.
    """
    planes = {}
    for w in walls:
        axis, inward = WALL_SIDES[w["side"]]
        coord = float(w["pos"][axis])
        cur = planes.get(w["side"])
        # inward=+1 -> room is above the plane -> innermost is the largest
        if cur is None or (coord > cur["coord"]) == (inward > 0):
            planes[w["side"]] = {"coord": coord, "axis": axis,
                                 "inward": inward, "role": w["role"]}
    return {"layout_id": int(layout_id), "planes": planes,
            "n_walls": len(walls),
            "problems": _check_layout(layout_id, planes)}


def main(argv=None):
    from locus_cabinet_probe import _default_configs

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", help="JSON list of [layout_id, config_path]")
    ap.add_argument("--config-dir", default="/tmp/locus_wall_probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    pairs = (json.load(open(args.configs)) if args.configs
             else _default_configs(args.config_dir))
    report = {}
    for layout_id, config_path in pairs:
        walls = probe_layout(config_path)
        rec = planes_for_layout(int(layout_id), walls)
        rec["walls"] = walls
        report[str(layout_id)] = rec
        print(f"layout {layout_id}: {len(walls)} walls, "
              f"sides={sorted(rec['planes'])}, "
              f"problems={rec['problems'] or 'none'}", flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"\nwrote {args.out}")
    print("_WALL_PLANES = {   # side -> world coord of the plane")
    for key in sorted(report, key=int):
        pl = report[key]["planes"]
        items = ", ".join(f"'{s}': {pl[s]['coord']:.3f}" for s in sorted(pl))
        print(f"    {key}: {{{items}}},")
    print("}")

    bad = {k: v["problems"] for k, v in report.items() if v["problems"]}
    if bad:
        print(f"\nFAILED consistency check: {json.dumps(bad, indent=1)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
