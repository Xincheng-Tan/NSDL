"""LOCUS renderer-GT instance-mask → bbox exporter (docs/YOLO.md §6.3, P7.5).

Closes the gap YOLO.md §6.3 flags as "尚未实现": the renderer knows which pixel
belongs to which actor, so full34 boxes are GROUND TRUTH, not pseudo-labels. This
is categorically different from a text-prompted detector (GroundingDINO/SAM3),
which has to *guess* the class of a pixel the renderer *drew*.

Pipeline
--------
1. ``capture_segmentation(env)`` — grab the ``Segmentation`` texture from the SAME
   human-render camera and frame the RGB snapshot comes from. ManiSkill's
   ``Scene.get_human_render_camera_images`` hardcodes ``segmentation=False``, so we
   go one level down to the camera and ask for it ourselves. Official files stay
   untouched (docs/benchmark.md §9 discipline).
2. ``masks_to_boxes(seg, id_map)`` — per-actor visible pixels → tight bbox. The
   buffer only ever contains VISIBLE pixels, which is exactly YOLO.md §5.3-1
   ("只标可见像素，不画 amodal"): an occluded object shrinks to its visible part
   and a fully hidden one yields no box at all -- which is also §5.3-3 for the S4
   opaque-cabinet targets. No visibility heuristic is needed or wanted.
3. ``export_annotations(...)`` — X-AnyLabeling JSON, byte-compatible with what
   ``scripts/auto_label_with_yolo.py`` writes, so the existing review GUI and
   ``prepare_yolo26_finetune_dataset.py`` consume it unchanged.

Class resolution (the part that genuinely needs a human decision, made ONCE)
---------------------------------------------------------------------------
Movable objects are built with ``name=<class>_<idx>`` (robocasa
``scene_builder.py:514``), so ``meat_0 -> meat`` is mechanical. Fixtures are NOT:
they are named by ROLE (``cab_1``, ``stack_2``, ``counter_main``), so the class
has to come from the fixture TYPE. ``FIXTURE_TYPE_TO_CLASS`` is that mapping and
it encodes the §5.3-5 framing caliber:

* ``single_cabinet`` / ``hinge_cabinet`` -> ``cabinet``; ``drawer`` ->
  ``drawer``. A ``stack`` is a composite of those, so its links are attributed
  to whichever member class owns them, never boxed as one blob.
* ``open_cabinet`` (doorless wall shelf board) and ``panel_cabinet``
  (unopenable island side panel) are NOT ``cabinet``: they carry none of the
  cabinet semantics LOCUS needs and are visually a shelf / a counter side. See
  ``DROPPED_TYPES``.
* A multi-link fixture is unioned over its FRONT links only
  (``_filter_caliber_links``), not over every link -- a drawer's carcass leaks a
  handful of pixels through the gap and would drag the box "into the wall" on
  some samples but not others, which is exactly the §5.3-5 instability.
* A box whose visible pixels are a sparse scatter inside a big rectangle (a
  fixture hidden behind another fixture, e.g. the under-sink base cabinet) is
  dropped by ``MIN_VISIBLE_FRAC``: a human cannot draw it either.
* ``housing_cabinet`` is structural trim around a fridge/oven/microwave, NOT an
  interactable cabinet -> dropped, so we do not glue the surrounding carcass onto
  the appliance box.
* A WALL ``counter`` is not a detector class (§4.2 keeps semantic surfaces out,
  §5.3-5 forbids folding an adjacent counter into a fixture box). The kitchen
  ISLAND is the exception and maps to ``table``: robocasa builds it as
  ``type: counter`` with ``default_config_name: island``, but VIKI's own layout
  yamls name that very area ``names: ['table', 'kitchen island area']``
  (``configs/layouts/layout_{1,3,6,9}.yaml`` — exactly the four blueprint island
  layouts), LOCUS templates use ``table`` as a destination mask on those layouts,
  and the 900 VIKI-train hand annotations drew ``table`` on it at ~1.0/img. So
  ``table`` is one object with one agreed referent across the area vocabulary,
  the human labels and the detector. See ``is_island_counter``.
* Everything not in full34 (walls, floor, plants, hammer, shoe, ...) is dropped
  explicitly via ``DROPPED_TYPES``/``DROPPED_OBJECT_CLASSES`` rather than by a
  silent fallthrough -- an unmapped name is an ERROR (``strict=True``), because
  quietly skipping a real class would poison the training set.

What still needs a human (YOLO.md §6.3, unchanged by this module)
----------------------------------------------------------------
Spot-check transparent cabinets (glass reflections/double reads), tiny cutlery
(``fork/spoon/scissors``), thin elongated assets, and the fixture caliber itself.
This exporter changes the human's job from "draw 34 classes on every image" to
"approve a caliber, then audit"; it does not remove the audit.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.join(_ROBOFACTORY, "script_general")
for _p in (_ROBOFACTORY, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = [
    "FULL34_CLASSES",
    "FIXTURE_TYPE_TO_CLASS",
    "DROPPED_TYPES",
    "DROPPED_OBJECT_CLASSES",
    "MIN_BOX_AREA",
    "MIN_BOX_SIDE",
    "LINK_CALIBER_CLASSES",
    "LINK_MIN_FRAC",
    "LINK_MIN_PIXELS",
    "MIN_VISIBLE_FRAC",
    "KEEP_ANYWAY_VISIBLE_PIXELS",
    "MIN_VISIBLE_PIXELS_FIXTURE",
    "MIN_VISIBLE_PIXELS_MOVABLE",
    "FIXTURE_CLASSES",
    "min_visible_pixels_for",
    "capture_segmentation",
    "actor_view_probe",
    "class_of_actor_name",
    "masks_to_boxes",
    "make_shape",
    "build_annotation_payload",
    "export_annotations",
    "summarize_class_counts",
]

# Fixed full34 order lives in ONE place in the repo (docs/YOLO.md §3).
_CLASS_LIST_FILE = os.path.join(
    os.path.dirname(_ROBOFACTORY), "configs", "yolo", "locus_full34_class_list.txt"
)


def _load_full34():
    with open(_CLASS_LIST_FILE) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if len(names) != 34:
        raise ValueError(f"expected 34 classes in {_CLASS_LIST_FILE}, got {len(names)}")
    return tuple(names)


FULL34_CLASSES = _load_full34()

# ---------------------------------------------------------------------------
# class resolution
# ---------------------------------------------------------------------------
# Fixture TYPE (robocasa customized_layout_*.yaml `type:` field) -> full34 class.
# This is the §5.3-5 "框口径必须稳定" decision, made once for the whole batch.
FIXTURE_TYPE_TO_CLASS = {
    "single_cabinet": "cabinet",
    "hinge_cabinet": "cabinet",
    "drawer": "drawer",
    "fridge": "fridge",
    "oven": "oven",
    "sink": "sink",
    "toaster": "toaster",
}

# The kitchen island is a `type: counter` fixture, so its class cannot come from
# the type alone -- the same type builds every wall counter, which is NOT a
# detector box. robocasa distinguishes them with `default_config_name: island`,
# but that key is DELETED during config resolution (scene_utils.py:88-138) and
# `Counter` keeps no island flag, so the only discriminators that survive to the
# built entity are the NAME and the depth (island >= 1.0 m vs 0.65 m for every
# wall counter). The name is the one the segmentation id map hands us:
#
#   layout 1 one_wall_large   island_group/island_counter -> island_counter_island_group_0
#   layout 3 L_shaped_large   left_group/island           -> island_left_group_0
#   layout 6 U_shaped_large   island_group/island         -> island_island_group_0
#   layout 9 wraparound       island_group/island         -> island_island_group_0
#
# i.e. the role name starts with `island`, and/or the fixture lives in the
# `island_group` whose name scene_builder.py:198 appends to every member.
_ISLAND_NAME_PREFIX = "island"
_ISLAND_GROUP_TOKEN = "_island_group"


def is_island_counter(name):
    """True if a ``type: counter`` entity is the kitchen ISLAND, not a wall run.

    Caliber note (docs/YOLO.md §6.3): the G_shaped layouts 7/8 have no island —
    their `front_group/counter_1` is a 0.65 m-deep wall counter with stools (a
    breakfast bar). The 900 VIKI-train hand annotations DID draw `table` there;
    LOCUS layouts 7/8 declare no `table` position area
    (`configs/layouts/layout_*.yaml` name it only on 1/3/6/9), so no LOCUS
    instruction can refer to a table on them and this function returns False.

    That disagreement is no longer merely disclosed: because both sources train
    ONE shared full34 detector, those 192 VIKI boxes made a shallow wall counter a
    `table` positive from one source and background from the other on the same two
    layouts. They were removed by `scripts/strip_viki_breakfast_bar_table.py`, so
    this predicate is now the single caliber for both sources.
    """
    raw = str(name).split("/", 1)[0].lower()
    return raw.startswith(_ISLAND_NAME_PREFIX) or _ISLAND_GROUP_TOKEN in raw

# Fixture types that exist in the scene but must NOT become detector boxes.
# Kept explicit so an unmapped NEW type raises instead of silently vanishing.
DROPPED_TYPES = frozenset({
    # structural / semantic-region / not-in-full34 fixtures
    "wall", "floor", "framed_window", "window", "wall_accessory", "box",
    # `counter` is only dropped for WALL runs; the island resolves to `table`
    # before this set is consulted (see is_island_counter / module docstring).
    "counter",
    "housing_cabinet",  # appliance trim, not an interactable cabinet
    # `open_cabinet` = robocasa OpenCabinet, the doorless wall SHELF board
    # (`default_config_name: shelves`, textured wood_planks/wood_grain). It has
    # no door, no carcass and no interior, so it supports none of the semantics
    # `cabinet` carries in LOCUS (openable, hides a target, anchors `in`). It is
    # also the `rack` semantic region of §4.2, which is explicitly NOT a
    # detector class. Boxing it as `cabinet` taught the detector "any
    # horizontal wood board is a cabinet".
    "open_cabinet",
    # `panel_cabinet` = PanelCabinet, "a panel door but no handle ... an
    # unopenable cabinet" (robocasa docstring), declared with
    # `solid_body: true` as the blank SIDE PANEL of a kitchen island. Visually
    # a counter side, functionally not an interactable cabinet.
    "panel_cabinet",
    "stack",            # composite: members are attributed individually
    "fixture_stack",
    "hood", "stovetop", "stove", "microwave", "dishwasher", "coffee_machine",
    "stool", "knife_block", "paper_towel", "utensil_holder", "utensil_rack",
    "accessory", "plant",   # robocasa FIXTURES maps these to Accessory
    "cabinet_panel", "cabinet_shelf", "handle",
})

# Movable-object classes present in layouts but deliberately outside full34
# (docs/YOLO.md §4.2: unused by LOCUS templates / not detector classes).
DROPPED_OBJECT_CLASSES = frozenset({
    "clothes_iron", "cooling_fan", "green_peas", "hammer", "indoor_plant",
    "pot", "rock", "shoe", "t_shirt", "towel", "watering",
})

# Degenerate-sliver guard: a couple of stray pixels from a reflection or a
# one-pixel edge is not a trainable box. Reported, not silently dropped.
MIN_BOX_AREA = 16
MIN_BOX_SIDE = 2

# --- §5.3-5 framing caliber for multi-link fixtures --------------------------
# Unioning EVERY link of an articulation is what made the drawer/cabinet caliber
# unstable. robocasa drawers expose the visible front face (`inner_box`) AND the
# carcass/interior (`object`); cabinets expose door panels AND the carcass. The
# carcass usually contributes a handful of stray pixels through a gap, but those
# pixels sit far to the side/back, so the unioned box silently grows "into the
# wall" on some samples and not others. Measured on pilot train_00301:
#
#   stack_6_main_group_4  inner_box 198x166 (5667 px) + object  99x92 (229 px)
#                         -> union 279x167   (link filter: 198x166)
#   stack_4_main_group_3  inner_box  51x58  (1033 px) + object  61x22 (  7 px)
#                         -> union 108x66    (link filter:  51x58)
#   cab_2_main_group      hingerightdoor 515x395 + hingeleftdoor 299x330
#                         + object 572x432 -> union 675x437 (filter: 635x425)
#
# So: keep only links carrying a meaningful share of the articulation's visible
# pixels, ALWAYS keeping the largest one (a fully-through-glass cabinet can have
# every link below the threshold; dropping all of them would delete the box).
LINK_CALIBER_CLASSES = frozenset({"cabinet", "drawer", "fridge", "oven"})
LINK_MIN_FRAC = 0.10
LINK_MIN_PIXELS = 64

# --- viewpoint-occlusion guard ----------------------------------------------
# A box whose visible pixels are a scatter of specks inside a large rectangle is
# not human-labelable and must not be trained on: the under-sink base cabinet in
# pilot train_00302 is `stack_2_main_group_1`, box area 37128 with 109 visible
# pixels (frac 0.0029); train_00306's `island_stack_1_island_group_1` is
# 144/14720 (0.0098). Legitimate cabinet boxes in the SAME images sit at
# frac >= 0.35, so the two populations separate cleanly. MIN_BOX_AREA/-SIDE
# cannot catch this: the bounding box is huge, only its fill is not.
#
# The absolute escape hatch keeps genuinely small-but-solid instances (a fork
# seen edge-on fills its own box densely, so it never needs it; a heavily
# occluded but contiguous object does).
MIN_VISIBLE_FRAC = 0.05
KEEP_ANYWAY_VISIBLE_PIXELS = 2000

# --- absolute visible-pixel floor, two-tier ---------------------------------
# `MIN_VISIBLE_FRAC` only catches a SPARSE fill inside a large rectangle. The
# 51-image spot-check (2026-07-29) found the complementary failure: a box that
# is small AND densely filled, so its fraction is high and every existing guard
# passes it. 13 of the 24 findings are this one shape — a drawer whose only
# visible pixels are half a protruding pull ring (`stack_4_left_group_2_0`,
# `stack_3_right_group_{1,2,3}_0`), or a fixture almost entirely behind a robot
# (`stack_1_right_group_2_0` behind fetch: "像素框大，但基本被 fetch 遮挡").
# MIN_BOX_AREA=16 is three orders of magnitude too permissive for those.
#
# The floor is two-tier because the two populations mean different things at the
# same pixel count:
#
#  * a FIXTURE is a large, planar, self-similar surface. 100 visible pixels of a
#    brown drawer front carry no identity — the neighbouring drawer front is the
#    same brown, and the reviewer's complaint was precisely that the boundary is
#    undrawable ("cabinet 之间、drawer 之间的分界很模糊"). Few visible pixels on a
#    fixture therefore always means "essentially hidden", never "small".
#  * a MOVABLE object is small in world terms, so few pixels is its NORMAL
#    appearance at range: a fork seen edge-on fills ~30 px and stays perfectly
#    identifiable, and it is one of the ten classes still short of the §6.2
#    volume gate. A fixture-grade floor here would delete trainable positives.
#
# Reported into `stats["under_min_pixels"]`, never silently dropped, so the
# manifest shows the cost. Set either to 0 to disable that tier.
MIN_VISIBLE_PIXELS_FIXTURE = 128
MIN_VISIBLE_PIXELS_MOVABLE = 24

# Fixture-grade classes: everything the layout BUILDS as a fixture, i.e. the
# `FIXTURE_TYPE_TO_CLASS` range plus `table` (the island, resolved by name — see
# `is_island_counter`). Robots are neither: they are large but articulated and
# self-dissimilar, so they take the movable floor.
FIXTURE_CLASSES = frozenset(FIXTURE_TYPE_TO_CLASS.values()) | {"table"}


def min_visible_pixels_for(label, fixture=MIN_VISIBLE_PIXELS_FIXTURE,
                           movable=MIN_VISIBLE_PIXELS_MOVABLE):
    """Absolute visible-pixel floor for one full34 class."""
    return fixture if label in FIXTURE_CLASSES else movable


def class_of_actor_name(name, fixture_types=None, strict=True):
    """Map a built actor/link name to its full34 class, or ``None`` if dropped.

    ``name`` is the sapien actor name: ``"<class>_<idx>"`` for movable objects
    (robocasa builds them with ``name=object_name``), or the fixture ROLE name
    (``cab_1``, ``stack_2``) for fixtures -- which is why ``fixture_types``
    (role name -> yaml ``type``) is required to resolve those.

    ``strict=True`` raises on a name that is neither mapped nor explicitly
    dropped. That is the point: a new asset must be classified by a human, not
    absorbed by a fallthrough.
    """
    fixture_types = fixture_types or {}
    raw = str(name)

    # Fixtures resolve by declared type, not by their role name.
    ftype = fixture_types.get(raw)
    if ftype is None and "/" in raw:            # link path "cab_1/door"
        ftype = fixture_types.get(raw.split("/", 1)[0])
    if ftype is not None:
        ftype = str(ftype).strip().lower()
        if ftype in FIXTURE_TYPE_TO_CLASS:
            return FIXTURE_TYPE_TO_CLASS[ftype]
        # `counter` is the one type whose class depends on WHICH instance it is:
        # the island is `table`, every wall run is dropped (see is_island_counter).
        if ftype == "counter":
            return "table" if is_island_counter(raw) else None
        if ftype in DROPPED_TYPES:
            return None
        if strict:
            raise KeyError(
                f"fixture type {ftype!r} (actor {raw!r}) is neither in "
                f"FIXTURE_TYPE_TO_CLASS nor DROPPED_TYPES -- classify it "
                f"explicitly (docs/YOLO.md §5.3-5)"
            )
        return None

    base = raw.split("/", 1)[0]
    stem = base.rsplit("_", 1)[0] if base.rsplit("_", 1)[-1].isdigit() else base
    stem = stem.replace(" ", "_").replace("-", "_").lower()
    if stem in FULL34_CLASSES:
        return stem
    if stem in DROPPED_OBJECT_CLASSES:
        return None
    # Robot links: the agent uid is the class (fetch, panda, unitree_h1, ...).
    for robot in FULL34_CLASSES[:6]:
        if stem == robot or stem.startswith(robot + "_"):
            return robot
    if stem in DROPPED_TYPES:
        return None
    if strict:
        raise KeyError(
            f"actor {raw!r} (stem {stem!r}) maps to no full34 class and is not "
            f"in DROPPED_OBJECT_CLASSES -- classify it explicitly"
        )
    return None


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def capture_segmentation(env, camera_name=None, with_position=False):
    """Return ``(seg_hw_int, id_map)`` for one human-render camera.

    ``seg_hw_int`` is an ``(H, W)`` int array of per-scene actor/link ids;
    ``id_map`` maps id -> Actor/Link (ManiSkill ``env.segmentation_id_map``).

    With ``with_position=True`` returns ``(seg, id_map, probe)`` instead, where
    ``probe`` carries the camera-frame ``Position`` texture and the camera
    object for ``actor_view_probe``. Both textures come from the SAME
    ``capture()``, which is the only way the per-pixel geometry and the mask are
    guaranteed to describe one frame.

    Why not ``env.render()``: ManiSkill's ``get_human_render_camera_images``
    passes ``segmentation=False`` (``envs/scene.py``), so the RGB path throws the
    buffer away. We capture from the same camera object instead -- same extrinsics,
    same frame, so boxes and pixels agree by construction.

    Requires the ``default`` (or ``minimal``) shader pack: only those declare a
    ``Segmentation`` texture. Ray-tracing packs (``rt``/``rt-fast``) declare
    ``Color`` only, so mask export and rt rendering are mutually exclusive.
    """
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    cameras = base.scene.human_render_cameras
    if not cameras:
        raise RuntimeError("scene has no human_render cameras")
    if camera_name is None:
        if len(cameras) != 1:
            raise ValueError(
                f"camera_name is required when several human-render cameras exist: "
                f"{sorted(cameras)} -- it MUST be the one the RGB snapshot used"
            )
        camera_name = next(iter(cameras))
    if camera_name not in cameras:
        raise KeyError(f"unknown human-render camera {camera_name!r}; have {sorted(cameras)}")

    camera = cameras[camera_name]
    shader = getattr(camera.config.shader_config, "shader_pack", None)
    if shader not in ("default", "minimal"):
        raise RuntimeError(
            f"shader_pack={shader!r} exposes no Segmentation texture; render with "
            f"'default' (rt/rt-fast cannot produce masks)"
        )
    for obj in getattr(base, "_hidden_objects", []):
        obj.show_visual()
    base.scene.update_render()
    camera.capture()
    obs = camera.get_obs(rgb=False, depth=False, position=with_position,
                         segmentation=True)
    seg = obs["segmentation"]
    seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
    seg = np.asarray(seg).squeeze()           # (1,H,W,1) / (H,W,1) -> (H,W)
    if seg.ndim != 2:
        raise RuntimeError(f"unexpected segmentation shape {seg.shape}")
    seg = seg.astype(np.int64)
    if not with_position:
        return seg, base.segmentation_id_map
    pos = obs["position"]
    pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
    pos = np.asarray(pos).squeeze()           # (1,H,W,4) -> (H,W,4)
    if pos.ndim != 3 or pos.shape[:2] != seg.shape:
        raise RuntimeError(
            f"position texture {pos.shape} does not match segmentation "
            f"{seg.shape}")
    return seg, base.segmentation_id_map, {"position": pos, "camera": camera}


# ---------------------------------------------------------------------------
# A-2 instrumentation: why is this box small?
# ---------------------------------------------------------------------------
# The spot-check produced two complaints that look identical in the label file
# ("几乎不可辨认") but have opposite causes and opposite fixes:
#
#   * OCCLUDED -- "像素框大，但基本被 fetch 遮挡" (train_00083). The actor is
#     large on screen and something in front of it owns the pixels. Fix lives in
#     scene composition: robot spawn areas, camera pose.
#   * EDGE-ON -- "只露出抽屉面板凸出的半截拉环" (train_00020). Nothing occludes
#     it; the drawer front is nearly parallel to the view ray, so its own
#     silhouette is a sliver. Fix lives in the caliber (drop it) or the camera.
#
# `visible_pixels` alone cannot separate them, and neither can `visible_frac`
# (both cases fill their own small box densely). Two per-actor measurements do:
#
#   amodal_pixels  the actor's own image footprint with nothing in front of it,
#                  measured as the area of the convex hull of its projected
#                  render AABB clipped to the frame. This is an UPPER bound on
#                  the true silhouette (AABB >= mesh), exact only for
#                  box-shaped bodies, so `visible/amodal` is a lower bound on
#                  visibility. That direction is the safe one: it can call a
#                  fully visible round object partly occluded, never the
#                  reverse.
#   view_cos       |cos| between the view ray and the actor's dominant visible
#                  surface normal, from a plane fit of its visible 3D points.
#                  ~1 = facing the camera, ~0 = edge-on. Convention-free: both
#                  vectors come from the camera-frame Position texture, so no
#                  robocasa fixture-orientation assumption enters.
#
# This is instrumentation, not a gate: it is recorded per box so the P1 caliber
# question ("which of these drawers should have been dropped?") can be answered
# from the manifest instead of by re-rendering. `MIN_VISIBLE_PIXELS_*` is the
# gate, and it deliberately does not consult either of these numbers.

# The `default` shader pack's Position texture is NOT metres: ManiSkill's
# `default_position_texture_transform` (render/shaders.py) returns
# `(xyz_cam * 1000).to(int16)`, i.e. camera-frame MILLIMETRES with 1mm
# quantisation and a +-32.7m range. `view_cos`/`planarity` are scale-free and
# so were right either way, but `depth_m` read 1000x high until this scale
# landed (a 4.76m drawer was reported as 4764.966).
_POSITION_TO_M = 1e-3


def _render_aabb(entity):
    """World-frame AABB of an entity's render geometry, or None.

    sapien only exposes this on the RenderBodyComponent, so a collision-only or
    hidden body has no answer -- which is correct: it owns no pixels either.
    """
    try:
        import sapien.render as sr
    except Exception:                                   # pragma: no cover
        return None
    obj = getattr(entity, "_objs", None)
    obj = obj[0] if obj else entity                     # ManiSkill wrapper
    entity_ = getattr(obj, "entity", obj)
    find = getattr(entity_, "find_component_by_type", None)
    if find is None:
        return None
    body = find(sr.RenderBodyComponent)
    if body is None:
        return None
    try:
        aabb = body.compute_global_aabb_tight()
    except Exception:                                   # pragma: no cover
        return None
    return np.asarray(aabb, dtype=np.float64).reshape(2, 3)


def _hull(points):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def _clip_to_rect(poly, w, h):
    """Sutherland-Hodgman clip of a convex polygon to ``[0,w] x [0,h]``."""
    # inside(x, y) := a*x + b*y + c >= 0, i.e. x >= 0, x <= w, y >= 0, y <= h
    for a, b, c in ((1, 0, 0.0), (-1, 0, float(w)),
                    (0, 1, 0.0), (0, -1, float(h))):
        if not poly:
            return []
        out, n = [], len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            dp = a * p[0] + b * p[1] + c
            dq = a * q[0] + b * q[1] + c
            if dp >= 0:
                out.append(p)
            if (dp >= 0) != (dq >= 0):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
    return poly


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        s += x0 * y1 - x1 * y0
    return abs(s) * 0.5


def _camera_matrices(camera):
    """``(K, E)`` as ``(3,3)`` / ``(3,4)`` float arrays, whatever we were handed.

    ``scene.human_render_cameras[name]`` is ManiSkill's ``sensors.camera.Camera``
    WRAPPER, which has no projection methods of its own -- they live on the
    sapien camera it holds in ``.camera`` (it re-exports them only through
    ``get_params()``). Accept all three shapes: the wrapper, the raw camera, and
    a ``get_params()`` dict, so the probe cannot break on a ManiSkill version
    that moves them again.
    """
    for obj in (camera, getattr(camera, "camera", None)):
        if obj is not None and hasattr(obj, "get_intrinsic_matrix"):
            return obj.get_intrinsic_matrix(), obj.get_extrinsic_matrix()
    if hasattr(camera, "get_params"):
        p = camera.get_params()
        return p["intrinsic_cv"], p["extrinsic_cv"]
    raise AttributeError(
        f"{type(camera).__name__} exposes no intrinsic/extrinsic matrices")


def _project_aabb_area(aabb, camera, shape):
    """Image-space area of an AABB's projected convex hull, clipped to frame.

    Returns None when any corner is at or behind the camera plane: the
    projection is then not a convex hull of finite points and any number we
    printed would be fiction.
    """
    h, w = int(shape[0]), int(shape[1])
    K, E = _camera_matrices(camera)
    K = K.cpu().numpy() if hasattr(K, "cpu") else np.asarray(K)
    E = E.cpu().numpy() if hasattr(E, "cpu") else np.asarray(E)
    K = np.asarray(K, dtype=np.float64).reshape(-1, 3, 3)[0]
    E = np.asarray(E, dtype=np.float64).reshape(-1, 3, 4)[0]
    lo, hi = aabb
    corners = np.array([[lo[0] if b & 1 else hi[0],
                         lo[1] if b & 2 else hi[1],
                         lo[2] if b & 4 else hi[2]] for b in range(8)])
    cam = corners @ E[:, :3].T + E[:, 3]
    if np.any(cam[:, 2] <= 1e-6):
        return None
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    return _poly_area(_clip_to_rect(_hull(uv.tolist()), w, h))


def _plane_view_cos(points_cam):
    """|cos| between the view ray and the best-fit plane's normal.

    ``points_cam`` are camera-frame xyz of one actor's visible pixels. The
    normal is the smallest-variance direction of their covariance, which for a
    planar patch (a drawer front, a counter top) is the surface normal, and for
    a blob is arbitrary -- so this is reported together with the planarity
    residual and read only where the residual says the fit meant something.
    """
    if points_cam.shape[0] < 16:
        return None, None
    c = points_cam.mean(0)
    q = points_cam - c
    try:
        _u, s, vh = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:                       # pragma: no cover
        return None, None
    n = vh[-1]
    d = c / (np.linalg.norm(c) or 1.0)
    flat = float(s[-1] / (s[0] or 1.0))          # 0 = perfectly planar
    return abs(float(np.dot(n, d))), flat


def actor_view_probe(seg, id_map, probe, fixture_types=None, strict=False,
                     max_pixels_per_fit=4096):
    """Per-actor ``amodal_pixels`` / ``view_cos`` / depth for one frame.

    Keyed by the same resolved name ``masks_to_boxes`` groups links under (the
    ARTICULATION name for links), so the result merges into its boxes by
    ``actor``. Articulation entries union the render AABBs of every link, which
    over-states the amodal footprint of a fixture whose carcass is buried in the
    wall -- the same direction of error as the AABB itself.

    ``strict=False`` by default: instrumentation must never be the thing that
    fails a render. Unclassifiable actors are simply skipped.
    """
    position = probe["position"]
    camera = probe["camera"]
    hw = seg.shape
    per_actor = defaultdict(lambda: {"visible_pixels": 0, "aabbs": [],
                                     "pixels": []})
    for sid in np.unique(seg).tolist():
        if sid <= 0:
            continue
        entity = id_map.get(sid)
        if entity is None:
            continue
        art = getattr(entity, "articulation", None)
        name = getattr(entity, "name", None) or str(entity)
        resolve = (getattr(art, "name", None) or name) if art is not None else name
        cls = class_of_actor_name(resolve, fixture_types=fixture_types,
                                  strict=strict)
        if cls is None:
            continue
        mask = seg == sid
        rec = per_actor[resolve]
        rec["label"] = cls
        rec["visible_pixels"] += int(mask.sum())
        aabb = _render_aabb(entity)
        if aabb is not None:
            rec["aabbs"].append(aabb)
        pts = position[mask][:, :3]
        if pts.shape[0] > max_pixels_per_fit:           # a plane fit needs no more
            step = pts.shape[0] // max_pixels_per_fit + 1
            pts = pts[::step]
        rec["pixels"].append(pts)

    out = {}
    for resolve, rec in per_actor.items():
        amodal = None
        if rec["aabbs"]:
            stack = np.stack(rec["aabbs"])
            union = np.array([stack[:, 0, :].min(0), stack[:, 1, :].max(0)])
            amodal = _project_aabb_area(union, camera, hw)
        pts = np.concatenate(rec["pixels"]) if rec["pixels"] else None
        view_cos = flat = depth = None
        if pts is not None and pts.shape[0]:
            pts = np.asarray(pts, dtype=np.float64) * _POSITION_TO_M
            view_cos, flat = _plane_view_cos(pts)
            depth = float(np.linalg.norm(pts, axis=1).mean())
        vis = rec["visible_pixels"]
        out[resolve] = {
            "label": rec.get("label"),
            "visible_pixels": vis,
            "amodal_pixels": None if amodal is None else int(round(amodal)),
            "visible_over_amodal": (None if not amodal
                                    else round(vis / float(amodal), 4)),
            "view_cos": None if view_cos is None else round(view_cos, 4),
            "planarity": None if flat is None else round(flat, 4),
            "depth_m": None if depth is None else round(depth, 3),
        }
    return out


# ---------------------------------------------------------------------------
# masks -> boxes
# ---------------------------------------------------------------------------
def _filter_caliber_links(links, min_frac=LINK_MIN_FRAC,
                          min_pixels=LINK_MIN_PIXELS):
    """Keep the links that define a fixture's visible FRONT, drop stray carcass.

    ``links`` is ``[(visible_pixels, (x0, y0, x1, y1)), ...]`` for one
    articulation. A link survives if it carries ``>= min_frac`` of the
    articulation's visible pixels or ``>= min_pixels`` absolute; the largest
    link always survives, so the result is never empty. See
    ``LINK_CALIBER_CLASSES`` for why this exists.
    """
    if not links:
        return []
    total = sum(n for n, _ in links)
    thresh = max(min_frac * total, float(min_pixels))
    kept = [l for l in links if l[0] >= thresh]
    return kept or [max(links, key=lambda l: l[0])]


def masks_to_boxes(seg, id_map, fixture_types=None, strict=True,
                   min_area=MIN_BOX_AREA, min_side=MIN_BOX_SIDE,
                   min_visible_frac=MIN_VISIBLE_FRAC,
                   min_visible_pixels_fixture=MIN_VISIBLE_PIXELS_FIXTURE,
                   min_visible_pixels_movable=MIN_VISIBLE_PIXELS_MOVABLE,
                   link_caliber=True):
    """Visible-pixel bboxes per full34 class.

    Returns ``(boxes, stats)``. ``boxes`` is a list of
    ``{"label", "actor", "xyxy", "visible_pixels", "visible_frac"}`` in full34
    class order then position; ``stats`` records what was dropped and why, so a
    shrinking class count is visible in the manifest instead of being mistaken
    for "class absent".

    One box per ACTOR (not per connected component): a partly occluded object
    split into two visible patches is still one object, and an amodal-free tight
    box over its visible pixels is the §5.3-1/-6 caliber. Multi-link
    articulations (cabinet/drawer/fridge/oven) are unioned back to their parent
    articulation, which is what makes "the cabinet box" a single stable frame --
    but only over the links that pass ``_filter_caliber_links``, otherwise the
    frame is not stable at all (see ``LINK_CALIBER_CLASSES``).

    Two guards drop boxes a human could not draw:

    * ``min_area`` / ``min_side`` -- degenerate slivers;
    * ``min_visible_frac`` -- boxes whose visible pixels are a sparse scatter
      inside a large rectangle, i.e. a fixture hidden behind another fixture
      (``MIN_VISIBLE_FRAC``). Set to 0 to disable.
    * ``min_visible_pixels_fixture`` / ``_movable`` -- the absolute two-tier
      floor for a box that is small AND dense, which the fraction guard passes
      by construction (see ``MIN_VISIBLE_PIXELS_FIXTURE``). Set to 0 to disable.

    All three record into ``stats["slivers"]`` / ``stats["occluded"]`` /
    ``stats["under_min_pixels"]`` rather than vanishing.
    """
    ids, counts = np.unique(seg, return_counts=True)
    per_key_links = defaultdict(list)
    stats = {"unmapped_ids": [], "dropped_actors": [], "slivers": [],
             "occluded": [], "under_min_pixels": [], "dropped_pixels": {},
             "background_pixels": 0}

    for sid, npx in zip(ids.tolist(), counts.tolist()):
        if sid <= 0:                               # 0 = background / infinite
            stats["background_pixels"] += int(npx)
            continue
        entity = id_map.get(sid)
        if entity is None:
            stats["unmapped_ids"].append(int(sid))
            continue
        # ManiSkill strips the articulation prefix off Link names at wrap time
        # (structs/articulation.py: wrapped_link.name = "door", "link_0", ...),
        # so for links the class identity lives on the ARTICULATION name
        # (fixture role name for robocasa fixtures, agent uid for robots).
        art = getattr(entity, "articulation", None)
        name = getattr(entity, "name", None) or str(entity)
        resolve_name = (getattr(art, "name", None) or name) if art is not None else name
        cls = class_of_actor_name(resolve_name, fixture_types=fixture_types,
                                  strict=strict)
        if cls is None:
            key = name if art is None else f"{resolve_name}/{name}"
            stats["dropped_actors"].append(key)
            # A-2: how much of the frame goes to things we deliberately do not
            # label. train_00184's unlabeled knife block overlapped a labeled
            # `scissors`, so "is the dropped set visually significant?" is a
            # question the manifest has to be able to answer.
            stats["dropped_pixels"][key] = (
                stats["dropped_pixels"].get(key, 0) + int(npx))
            continue

        ys, xs = np.nonzero(seg == sid)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        # Group articulation links under one parent so a cabinet is one box.
        per_key_links[(cls, resolve_name)].append((int(npx), (x0, y0, x1, y1)))

    boxes = []
    for (cls, actor), links in per_key_links.items():
        kept = (_filter_caliber_links(links)
                if link_caliber and cls in LINK_CALIBER_CLASSES else links)
        npix = sum(n for n, _ in kept)
        x0 = min(b[0] for _, b in kept)
        y0 = min(b[1] for _, b in kept)
        x1 = max(b[2] for _, b in kept)
        y1 = max(b[3] for _, b in kept)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        frac = npix / float(w * h)
        entry = {"label": cls, "actor": actor,
                 "xyxy": [float(x0), float(y0), float(x1 + 1), float(y1 + 1)],
                 "visible_pixels": npix, "visible_frac": round(frac, 4),
                 "n_links": len(links), "n_links_kept": len(kept)}
        if w * h < min_area or min(w, h) < min_side:
            stats["slivers"].append({**entry, "xyxy": [x0, y0, x1, y1]})
            continue
        floor = min_visible_pixels_for(cls, fixture=min_visible_pixels_fixture,
                                       movable=min_visible_pixels_movable)
        if floor and npix < floor:
            stats["under_min_pixels"].append({**entry, "floor": int(floor)})
            continue
        if (min_visible_frac and frac < min_visible_frac
                and npix < KEEP_ANYWAY_VISIBLE_PIXELS):
            stats["occluded"].append(entry)
            continue
        boxes.append(entry)
    order = {c: i for i, c in enumerate(FULL34_CLASSES)}
    boxes.sort(key=lambda b: (order[b["label"]], b["xyxy"][0], b["xyxy"][1]))
    stats["n_boxes"] = len(boxes)
    return boxes, stats


# ---------------------------------------------------------------------------
# X-AnyLabeling JSON (format-compatible with scripts/auto_label_with_yolo.py)
# ---------------------------------------------------------------------------
def make_shape(label, xyxy, score=1.0, actor=None, visible_pixels=None,
               visible_frac=None, probe=None):
    """One rectangle shape. ``score=1.0`` because this is renderer GT, not a
    detector confidence -- the review GUI still lets a human edit or delete it.

    ``actor``/``visible_pixels``/``visible_frac`` go into ``attributes`` (an
    X-AnyLabeling-native per-shape dict that ``Shape.load_from_dict`` reads and
    ``Shape.to_dict`` writes back verbatim, so a human edit round-trip does not
    strip them). Without this the shipped JSON kept only label+points and any
    later attribution question -- "which fixture role produced this suspect
    box?", "re-filter by visible fraction" -- required re-rendering the scene,
    which render non-determinism makes non-exact.

    ``probe`` is the A-2 record for this actor (``amodal_pixels``, ``view_cos``,
    ``depth_m``, ...); its keys are merged into the same dict, so "occluded" and
    "edge-on" stay distinguishable after the fact.
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    attributes = {}
    if actor is not None:
        attributes["actor"] = str(actor)
    if visible_pixels is not None:
        attributes["visible_pixels"] = int(visible_pixels)
    if visible_frac is not None:
        attributes["visible_frac"] = float(visible_frac)
    for key in ("amodal_pixels", "visible_over_amodal", "view_cos",
                "planarity", "depth_m"):
        if probe and probe.get(key) is not None:
            attributes[key] = probe[key]
    return {
        "label": label,
        "score": float(score),
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": attributes,
        "kie_linking": [],
    }


def build_annotation_payload(image_path, image_height, image_width, shapes,
                             description=None, source="renderer_gt"):
    return {
        "version": "4.0.0-beta.3",
        "flags": {"locus_label_source": source},
        "shapes": list(shapes),
        "imagePath": str(image_path),
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
        "description": description,
    }


def export_annotations(seg, id_map, image_path, out_dir, fixture_types=None,
                       strict=True, description=None, probe=None):
    """Write one X-AnyLabeling JSON next to the reviewed-label tree.

    Returns ``(json_path, boxes, stats)``. An image with zero full34 instances
    still gets a JSON with an empty ``shapes`` list -- YOLO.md §5.3-8 requires the
    file to exist (dataset prep forbids ``--allow-missing-labels``).

    ``probe`` is ``actor_view_probe``'s per-actor A-2 record, merged onto both
    the returned boxes and the written attributes when given. The dropped boxes
    in ``stats`` get it too: the whole point of the instrumentation is to explain
    the ones that did NOT survive.
    """
    boxes, stats = masks_to_boxes(seg, id_map, fixture_types=fixture_types,
                                 strict=strict)
    if probe:
        for bucket in (boxes, stats["slivers"], stats["occluded"],
                       stats["under_min_pixels"]):
            for entry in bucket:
                rec = probe.get(entry.get("actor")) or {}
                for key in ("amodal_pixels", "visible_over_amodal", "view_cos",
                            "planarity", "depth_m"):
                    if rec.get(key) is not None:
                        entry[key] = rec[key]
    h, w = int(seg.shape[0]), int(seg.shape[1])
    payload = build_annotation_payload(
        image_path=image_path, image_height=h, image_width=w,
        shapes=[make_shape(b["label"], b["xyxy"], actor=b.get("actor"),
                           visible_pixels=b.get("visible_pixels"),
                           visible_frac=b.get("visible_frac"),
                           probe=(probe or {}).get(b.get("actor")))
                for b in boxes],
        description=description)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(str(image_path)))[0]
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return json_path, boxes, stats


def summarize_class_counts(all_boxes):
    """Per-class box counts for the §6.2 volume gate (≥100 train / 20 val)."""
    counts = {c: 0 for c in FULL34_CLASSES}
    for boxes in all_boxes:
        for b in boxes:
            counts[b["label"]] += 1
    return counts
