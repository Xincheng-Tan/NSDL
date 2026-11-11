"""LOCUS render worker with same-frame renderer-GT mask export (P7.5 hook).

This is the in-process bridge between the official render cadence and
``locus_mask_export``: the official path (``script/generate_planning_data.py``
-> ``planner.run`` -> ``solutions/viki_bench_task.solve``) snapshots RGB and
``exit(0)``s inside a subprocess, so there is no place to ask the SAME env for
its segmentation buffer. This worker replicates the official env construction
and snapshot cadence line-for-line (same ``gym.make`` kwargs, same
``reset(seed)``, same warmup-render loop, same ``res_video[0]`` frame) and then
captures the ``Segmentation`` texture from the same camera before the env is
closed -- RGB and boxes agree by construction. Official files stay untouched
(docs/benchmark.md §9 discipline); ``planner/run.py`` differences are:

* no ``RecordEpisodeMA`` wrapper (it only records trajectories/h5, which the
  snapshot path never uses; rendering is untouched by it);
* no ``exit(0)`` -- many configs are rendered per process, because this worker
  lives in a dedicated render env (``LOCUS_RENDER_PYTHON``) whose imports are
  expensive on network storage. Env is still created/closed per config, so
  state isolation matches the official one-process-per-config cadence.

Run environment: this module needs mani_skill/sapien (the ``locus-render``
env), NOT the NSDL training env. ``locus_render_pack.render_batch(...,
export_masks=True)`` shells out to it via ``LOCUS_RENDER_PYTHON``.

``build_fixture_types`` is the missing input of
``locus_mask_export.class_of_actor_name``: robocasa fixtures are named by ROLE
(``cab_1_main_group_0``), their CLASS lives in the scene builder's fixture
instances. The map is keyed by the *built* sapien entity name (articulation or
static actor), which is exactly what the segmentation id map yields.
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

from locus_mask_export import (  # noqa: E402
    FULL34_CLASSES,
    capture_segmentation,
    actor_view_probe,
    export_annotations,
)

# Official transparent cabinet styles (generate_tasks_from_json.py:130-136 /
# locus_render_pack.py S4 hook): scenes rendered with these MAY contain
# through-glass-visible objects the segmentation buffer cannot label. Kept
# only for the no-renderer resume-path heuristic (transparent_cabinet_gaps);
# the renderer path classifies every style via the owner probe instead, since
# styles 2/7/9 also declare a window panel on cab_top_small.
_TRANSPARENT_STYLES = (4, 11)

# robocasa cabinet.py panel shapes that are see-through (fixture_registry/
# cabinet.yaml: full_window_clear/full_window_blurred both set
# panel_type=full_window; divided_window has small window panes let through
# too, but at a scale no target survives recognizably, so it is excluded).
_WINDOW_PANEL_TYPES = frozenset({"full_window"})
# fixture_registry/cabinet.yaml: full_window_clear opacity=0.1, _blurred=0.4.
_BLURRED_OPACITY_MIN = 0.25


def _job_np_seed(config_path, attempt):
    """Stable per-(config, attempt) seed for the global numpy placement RNG."""
    import hashlib

    digest = hashlib.md5(
        f"{os.path.basename(str(config_path))}:{attempt}".encode()).hexdigest()
    return int(digest[:8], 16)

__all__ = [
    "build_fixture_types",
    "probe_jobs",
    "patch_jobs",
    "build_fixture_index",
    "glass_kind",
    "hidden_in_cabinet_candidates",
    "opaque_container_leaks",
    "transparent_cabinet_gaps",
    "classify_hidden_targets",
    "render_one",
    "render_jobs",
    "main",
]


def _type_of_class():
    """Reverse the robocasa ``FIXTURES`` registry: fixture class -> type string.

    First key wins on aliases (e.g. Accessory is registered under ``accessory``
    first), which keeps the reverse map inside the exporter's explicit
    FIXTURE_TYPE_TO_CLASS / DROPPED_TYPES vocabulary. Imported lazily so the
    CPU-only unit tests can exercise :func:`build_fixture_types` with an
    injected map and never touch sapien.
    """
    from utils.scenes.robocasa.scene_builder import FIXTURES

    out = {}
    for type_str, cls in FIXTURES.items():
        out.setdefault(cls, type_str)
    return out


def build_fixture_types(scene_data, type_of_class=None):
    """Map built fixture entity names -> robocasa type strings.

    ``scene_data`` is ``RobocasaSceneBuilder.scene_data`` (one dict per
    scene_idx with a ``fixtures`` name->Fixture mapping). The returned map is
    keyed by the name of the entity the fixture actually built
    (``fixture.articulation.name`` for articulated fixtures,
    ``fixture.actor.name`` for static ones) because that is the name the
    segmentation id map resolves to. The role name itself is added as a
    fallback key (harmless, helps debugging).
    """
    type_of_class = type_of_class or _type_of_class()
    types = {}
    for sd in scene_data:
        for role_name, fixture in sd["fixtures"].items():
            type_str = type_of_class.get(type(fixture))
            if type_str is None:
                # Same explicitness contract as the exporter: an unmapped
                # fixture class must be classified by a human, not skipped.
                raise KeyError(
                    f"fixture class {type(fixture).__name__!r} (role "
                    f"{role_name!r}) missing from the FIXTURES reverse map"
                )
            for attr in ("articulation", "actor"):
                ent = getattr(fixture, attr, None)
                name = getattr(ent, "name", None)
                if name:
                    types[str(name)] = type_str
            types.setdefault(str(role_name), type_str)
    return types


def _box_attr(box, key):
    """Read one box field across BOTH in-memory shapes.

    ``masks_to_boxes``/``export_annotations`` hand the live render path FLAT
    dicts (``{"label", "actor", "visible_pixels", "visible_frac", "xyxy"}``);
    only ``make_shape`` nests the last three under ``attributes`` for the label
    JSON, which is what the resume path and the tests read back. A reader that
    knows only the nested form degrades SILENTLY on the live path (every actor
    reads as absent → the S4 gate returns "no leaks" on a leaking scene, and
    :func:`_boxed_actors` falls back to label keying). Measured that way:
    the 2026-07-29 A/B rendered nine known-leaking scenes and `render_one`
    reported zero, while the same configs+labels scored 9/9 off disk.
    """
    if key in box:
        return box[key]
    return (box.get("attributes") or {}).get(key)


def _boxed_actors(boxes):
    """The set of entities that produced a box, keyed by ACTOR not label.

    Label keying silently mixes instances: an in-cabinet ``bread_1`` reads as
    "boxed" whenever any other ``bread`` is boxed on the counter, which both
    hides it from the glass/solid probe and hides it from the S4 leak gate —
    and S4's counterfactual pairs put a second instance of the same class
    outside the cabinet on purpose. Callers that only carry labels (there is no
    actor to key on) fall back to the label, preserving the old behaviour.
    """
    out = set()
    for b in boxes:
        actor = _box_attr(b, "actor")
        out.add(str(actor) if actor else str(b.get("label")))
    return out


def _in_cabinet_targets(cfg):
    """GT entities whose init position is a cabinet, in the labeled vocabulary.

    Restricted to MOVABLE objects (``cfg['objects']``): ``gt.init_pos`` also
    carries fixture entities such as ``cabinet_0`` whose "init position" is the
    cabinet itself. Those are never leaks (they own their own pixels) and, under
    actor keying, would otherwise be reported as hidden-in-cabinet candidates on
    every scene and sent to the owner probe.
    """
    movable = {str(o.get("name")) for o in (cfg.get("objects") or [])
               if isinstance(o, dict)}
    out = []
    for ent, holder in (((cfg.get("gt") or {}).get("init_pos")) or {}).items():
        if movable and str(ent) not in movable:
            continue
        holders = holder if isinstance(holder, list) else [holder]
        if not any("cabinet" in str(h).lower() for h in holders):
            continue
        base = str(ent).rsplit("_", 1)
        stem = base[0] if len(base) == 2 and base[1].isdigit() else str(ent)
        if stem in FULL34_CLASSES:
            out.append((str(ent), stem))
    return out


def hidden_in_cabinet_candidates(cfg, boxes):
    """GT entities placed IN a cabinet that produced no box, any style.

    `visible_pixels == 0` alone cannot tell "behind an opaque carcass" (a human
    cannot label it either -- YOLO.md §5.3-3 says do NOT label) from "behind a
    glass door" (the rasterizer gives the pixel to the glass, but a human sees
    it and MUST label it). This returns the candidates; the owner probe
    (:func:`classify_hidden_targets`) decides which bucket each one lands in.
    Pure function so the CPU tests can pin the caliber divergence without a
    renderer.
    """
    boxed = _boxed_actors(boxes)
    return [ent for ent, stem in _in_cabinet_targets(cfg)
            if ent not in boxed and stem not in boxed]


def opaque_container_leaks(cfg, boxes):
    """The complement gate: in-cabinet targets that DID produce pixels, in a
    scene whose cabinet style is opaque.

    S4's premise is that the container must be OPENED to observe what is in it,
    so a target inside an opaque cabinet must contribute zero pixels. The
    style test is sufficient to decide the precondition, because placement and
    style are not independent: the official hook forces a transparent style
    ({4, 11}) whenever anything starts in a cabinet
    (``generate_tasks_from_json.py:130-136``), and LOCUS only overrides it for
    ``meta.opaque_container``, which ``decidability_check`` already requires to
    be an S4 cell. So *in-cabinet target + non-transparent style* is exactly
    the S4 configuration, and any box there is a leak — no extra flag needed,
    which is what lets the resume path (:func:`_summary_from_existing`) run
    this gate from disk with no renderer.

    Returns one record per leaking entity, carrying the exporter's own pixel
    accounting so a leak can be sized (a 9-pixel graze and a 2072-pixel
    protrusion are the same defect at very different severities).
    """
    style = ((cfg.get("scene") or {}).get("env") or {}).get("style_idx")
    if style in _TRANSPARENT_STYLES:
        return []
    by_actor = {}
    for b in boxes:
        actor = _box_attr(b, "actor")
        if actor:
            by_actor[str(actor)] = b
    leaks = []
    for ent, _stem in _in_cabinet_targets(cfg):
        box = by_actor.get(ent)
        if box is None:
            continue
        leaks.append({"entity": ent, "style_idx": style,
                      "visible_pixels": _box_attr(box, "visible_pixels"),
                      "visible_frac": _box_attr(box, "visible_frac")})
    return leaks


def transparent_cabinet_gaps(cfg, boxes):
    """Style-gated subset of :func:`hidden_in_cabinet_candidates`.

    Kept because the resume path (``--skip-existing``) has no renderer and can
    only apply the style heuristic. It over-reports (a transparent-STYLE scene
    still has opaque lower cabinets, and only ``cab_top``/``cab_top_small``
    carry the window panel) and under-reports (styles 2/7/9 also declare
    ``full_window_clear`` on ``cab_top_small``), which is exactly why the
    renderer path uses the owner probe instead.
    """
    style = ((cfg.get("scene") or {}).get("env") or {}).get("style_idx")
    if style not in _TRANSPARENT_STYLES:
        return []
    return hidden_in_cabinet_candidates(cfg, boxes)


def build_fixture_index(scene_data):
    """Map built fixture entity name (and role name) -> the Fixture instance.

    Same keying as :func:`build_fixture_types` (segmentation resolves links to
    their articulation name), but keeps the object so the occlusion probe can
    read ``panel_type`` / ``panel_config`` off it.
    """
    index = {}
    for sd in scene_data:
        for role_name, fixture in sd["fixtures"].items():
            for attr in ("articulation", "actor"):
                ent = getattr(fixture, attr, None)
                name = getattr(ent, "name", None)
                if name:
                    index[str(name)] = fixture
            index.setdefault(str(role_name), fixture)
    return index


def glass_kind(fixture, link_name):
    """``"glass_clear"`` / ``"glass_blurred"`` / ``None`` for a pixel owner.

    Only a DOOR link of a window-panelled cabinet is see-through: the same
    fixture's carcass links are opaque, and in a transparent-style scene the
    lower cabinets keep solid panels (the style yaml only sets
    ``full_window_*`` on ``cab_top`` / ``cab_top_small``). Opacity comes from
    ``panel_config`` (0.1 clear / 0.4 blurred).
    """
    if fixture is None or "door" not in str(link_name).lower():
        return None
    panel_type = str(getattr(fixture, "panel_type", "") or "").lower()
    if panel_type not in _WINDOW_PANEL_TYPES:
        return None
    opacity = (getattr(fixture, "panel_config", None) or {}).get("opacity", 0.0)
    try:
        opacity = float(opacity)
    except (TypeError, ValueError):
        opacity = 0.0
    return "glass_blurred" if opacity >= _BLURRED_OPACITY_MIN else "glass_clear"


def classify_hidden_targets(candidates, owners, fixture_index):
    """Bucket each hidden in-cabinet target by WHAT owns its projected pixel.

    ``owners`` maps entity name -> ``{"link", "articulation", "pixel"}`` (or
    ``{"error": ...}`` / ``{"offscreen": True}``) as produced by the renderer
    probe. Returns ``{"glass_clear": [...], "glass_blurred": [...],
    "solid": [...], "unresolved": [...]}`` where the first two are
    human-REVIEWABLE candidates and ``solid`` is correctly box-free.

    The glass buckets are candidates, NOT assertions of visibility. This reads
    ONE pixel -- the one the target's origin projects to -- and **a transparent
    door does not imply the contents have pixels**: the cabinet's shelf boards,
    its carcass sides and the room wall sit behind the door and still occlude
    the target, while the probe stops at the glass because the door is
    frontmost (the segmentation and position buffers cannot see past it).
    Round 2 measured 21/100 queued targets unfindable by a human, 15 of them in
    ``layout_id == 5``. Deciding this here would need a second segmentation
    pass with the door links hidden; that is a re-render and is deferred
    (docs/YOLO.md §7.5.7). Until then the human adjudicates and
    ``scripts/audit_locus_glass_review.py`` folds the verdict back as a
    ``solid_hidden_ok`` correction.
    """
    out = {"glass_clear": [], "glass_blurred": [], "solid": [], "unresolved": []}
    for ent in candidates:
        info = (owners or {}).get(ent) or {}
        if info.get("error") or info.get("offscreen") or not info.get("link"):
            out["unresolved"].append({"entity": ent, **info})
            continue
        art = info.get("articulation") or info.get("link")
        kind = glass_kind(fixture_index.get(str(art)), info.get("link"))
        rec = {"entity": ent, "owner": info.get("link"),
               "owner_articulation": info.get("articulation"),
               "pixel": info.get("pixel")}
        out[kind or "solid"].append(rec)
    return out


def probe_pixel_owners(base, seg, id_map, entities):
    """Which entity owns the pixel each target's origin projects to.

    Projects the target's world origin into the SAME human-render camera the
    RGB/segmentation came from (real ``intrinsic_cv``/``extrinsic_cv``, OpenCV
    convention: camera-frame +z forward), then reads the segmentation buffer at
    that pixel. Returns ``{entity: {"link","articulation","pixel"}}``, with
    ``{"offscreen": True}`` or ``{"error": ...}`` when it cannot resolve.

    The origin is a single point, so a target whose centre is behind a mullion
    but whose body is visible reads as ``solid``; that lands it in the
    conservative bucket (routed to human review, not silently labeled).
    """
    import numpy as np

    cam = next(iter(base.scene.human_render_cameras.values()))
    params = cam.get_params()

    def _np(x):
        x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
        return np.asarray(x).squeeze()

    K, E = _np(params["intrinsic_cv"]), _np(params["extrinsic_cv"])
    H, W = seg.shape
    actors = dict(base.scene.actors)
    out = {}
    for ent in entities:
        actor = actors.get(ent)
        if actor is None:
            out[ent] = {"error": "no such actor"}
            continue
        pw = _np(actor.pose.p).reshape(-1)[:3]
        pc = E[:3, :3] @ pw + E[:3, 3]
        if pc[2] <= 0:
            out[ent] = {"offscreen": True, "reason": "behind camera"}
            continue
        uv = K @ pc
        u, v = int(round(float(uv[0] / uv[2]))), int(round(float(uv[1] / uv[2])))
        if not (0 <= u < W and 0 <= v < H):
            out[ent] = {"offscreen": True, "pixel": [u, v]}
            continue
        owner = id_map.get(int(seg[v, u]))
        art = getattr(owner, "articulation", None)
        out[ent] = {
            "link": getattr(owner, "name", None) or (str(owner) if owner else None),
            "articulation": getattr(art, "name", None),
            "pixel": [u, v],
        }
    return out


def render_one(config_path, out_image, out_json_dir, seed=0,
               sim_backend=None, strict=True, np_seed=None):
    """Render one scene config; write RGB PNG + renderer-GT label JSON.

    Mirrors ``planner/run.py:_main`` + ``solutions/viki_bench_task.solve``:
    same env kwargs, same seed semantics (official default 0), same
    11-render warmup, same first-batch frame. Returns a manifest entry dict.

    ``np_seed`` pins the GLOBAL numpy RNG the official placement loop draws
    from (``utils/scenes/scene_builder.py`` ``randp_scale`` collision
    avoidance). The official loop ``exit(0)``s after 100 failed placement
    draws; with the default fresh-process RNG state that failure is
    deterministic per config, so retries must reseed to get new placement
    rolls (see ``render_jobs`` attempt markers).
    """
    import gymnasium as gym
    import numpy as np
    import yaml
    from PIL import Image
    import tasks  # noqa: F401  registers VikiBenchTask-rf

    if np_seed is not None:
        np.random.seed(np_seed % (2 ** 32))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_id = cfg["task_name"] + "-rf"
    sim_backend = sim_backend or os.environ.get("VIKI_SIM_BACKEND", "gpu")

    env = gym.make(
        env_id,
        config=config_path,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend=sim_backend,
    )
    try:
        env.reset(seed=seed)
        base = env.unwrapped
        res_video = base.render()
        for _ in range(10):
            res_video = base.render()
        frame = res_video[0, :, :, :]
        frame = frame.cpu().numpy() if hasattr(frame, "cpu") else frame
        os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)
        Image.fromarray(frame).save(out_image)

        fixture_types = build_fixture_types(base.scene_builder.scene_data)
        seg, id_map, geom = capture_segmentation(base, with_position=True)
        # A-2 instrumentation. `strict=False` here on purpose: an actor the
        # class map rejects is already `export_annotations`'s error to raise, and
        # raising it first from the probe would blame the wrong pass.
        probe = actor_view_probe(seg, id_map, geom,
                                 fixture_types=fixture_types, strict=False)
        json_path, boxes, stats = export_annotations(
            seg, id_map, image_path=os.path.abspath(out_image),
            out_dir=out_json_dir, fixture_types=fixture_types, strict=strict,
            description=None, probe=probe)

        # In-cabinet GT target with no box: the segmentation buffer is
        # GEOMETRIC, so a glass door owns the pixels of everything behind it and
        # an opaque carcass owns the pixels of everything inside it -- both read
        # as `visible_pixels == 0`, but §5.3-3 wants the first LABELED and the
        # second left unlabeled. Only the renderer can tell them apart, by
        # asking who owns the target's projected pixel.
        candidates = hidden_in_cabinet_candidates(cfg, boxes)  # style-independent
        owners = (probe_pixel_owners(base, seg, id_map, candidates)
                  if candidates else {})
        hidden = classify_hidden_targets(
            candidates, owners, build_fixture_index(base.scene_builder.scene_data))
    finally:
        env.close()

    counts = {}
    for b in boxes:
        counts[b["label"]] = counts.get(b["label"], 0) + 1

    return {
        "config": config_path,
        "image": out_image,
        "mask_json": json_path,
        "n_boxes": stats["n_boxes"],
        "class_counts": counts,
        "n_slivers": len(stats["slivers"]),
        "n_occluded_dropped": len(stats.get("occluded") or []),
        "occluded_dropped": stats.get("occluded") or [],
        # A-1 absolute floor + A-2 instrumentation (docs/YOLO.md §5.3-9)
        "n_under_min_pixels": len(stats.get("under_min_pixels") or []),
        "under_min_pixels": stats.get("under_min_pixels") or [],
        "dropped_pixels": stats.get("dropped_pixels") or {},
        "unmapped_ids": stats["unmapped_ids"],
        "dropped_actors": sorted(set(stats["dropped_actors"])),
        # human-labelable (through glass) vs correctly box-free (solid)
        "transparent_cabinet_unlabeled": [r["entity"] for r in
                                          hidden["glass_clear"]],
        "glass_blurred_review": [r["entity"] for r in hidden["glass_blurred"]],
        "solid_hidden_ok": [r["entity"] for r in hidden["solid"]],
        "hidden_target_probe": hidden,
        "glass_verdict_source": "owner_probe",
        # P0-1 hard gate: the S4 premise is that an opaque container hides its
        # contents, so this list must be empty on every opaque-style scene.
        "opaque_container_leak": opaque_container_leaks(cfg, boxes),
    }


def _summary_from_existing(job):
    """Resume path (--skip-existing): rebuild the summary entry for a job whose
    image + label JSON already exist, WITHOUT re-rendering. Class counts come
    from the label shapes; the transparent-cabinet flag falls back to the
    style heuristic (pure function) because the glass-vs-solid owner probe needs
    a live renderer -- so a resumed entry can only over-report, never miss.
    Needed because a chunk rerun overwrites its summary file — skipped entries
    must not lose their bookkeeping."""
    import yaml

    json_path = os.path.join(
        job["json_dir"],
        os.path.splitext(os.path.basename(job["image"]))[0] + ".json")
    with open(json_path) as f:
        payload = json.load(f)
    # carry attributes, not just the label: the actor name and pixel counts are
    # what make the actor-keyed gaps and the S4 leak gate exact off disk.
    boxes = [{"label": s["label"], "attributes": s.get("attributes") or {}}
             for s in payload.get("shapes", [])]
    counts = {}
    for b in boxes:
        counts[b["label"]] = counts.get(b["label"], 0) + 1
    with open(job["config"]) as f:
        cfg = yaml.safe_load(f)
    return {
        "config": job["config"],
        "image": job["image"],
        "mask_json": json_path,
        "n_boxes": len(boxes),
        "class_counts": counts,
        "n_slivers": 0,
        "unmapped_ids": [],
        "dropped_actors": [],
        "n_occluded_dropped": 0,
        "occluded_dropped": [],
        # Both are drop buckets, and a resumed entry has only the SURVIVORS on
        # disk, so 0 here means "not measured", exactly like n_slivers above.
        "n_under_min_pixels": 0,
        "under_min_pixels": [],
        "dropped_pixels": {},
        "transparent_cabinet_unlabeled": transparent_cabinet_gaps(cfg, boxes),
        "glass_blurred_review": [],
        "solid_hidden_ok": [],
        "hidden_target_probe": {"style_heuristic_only": True},
        # Explicit so a resumed summary can never be mistaken downstream for a
        # probe-derived verdict: the style heuristic both over- and
        # under-reports (see transparent_cabinet_gaps), and a run that resumed
        # most of its jobs would otherwise look authoritative. Re-derive with
        # `--probe-only` before trusting the glass buckets.
        "glass_verdict_source": "style_heuristic_provisional",
        # unlike the glass buckets this one needs no renderer: it is decided by
        # style + GT placement + the boxes already on disk (see
        # :func:`opaque_container_leaks`), so a resumed entry carries the real
        # verdict and an existing render tree can be audited without a GPU.
        "opaque_container_leak": opaque_container_leaks(cfg, boxes),
        "skipped_existing": True,
    }


def render_jobs(jobs, seed=0, sim_backend=None, strict=True, summary_path=None,
                skip_existing=False):
    """Render a list of ``{"config","image","json_dir"[,"seed"]}`` jobs.

    Per-job failures (e.g. a strict unmapped-actor KeyError from a new
    layout/style combination) are recorded as ``{"error": ...}`` entries and
    do NOT kill the batch: a multi-hour pilot render must surface every
    failure at the end, not die at the first one. The caller decides whether
    failures are fatal (``render_batch`` marks the manifest row unrendered).

    ``skip_existing=True`` is the resume path: a job whose PNG and label JSON
    both already exist is summarized from disk instead of re-rendered
    (renders are one-frame snapshots, so an existing complete pair is final).

    The summary is flushed after every job, not once at the end: the official
    placement loop can ``exit(0)`` mid-chunk (scene_builder.py:213), which
    would otherwise discard the bookkeeping of every job that already
    succeeded in this process.
    """
    results = []

    def _flush():
        if not summary_path:
            return
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, summary_path)

    for job in jobs:
        json_path = os.path.join(
            job["json_dir"],
            os.path.splitext(os.path.basename(job["image"]))[0] + ".json")
        marker = job["image"] + ".inprogress"
        if skip_existing and os.path.isfile(job["image"]) \
                and os.path.isfile(json_path):
            if os.path.isfile(marker):
                os.remove(marker)
            res = _summary_from_existing(job)
            results.append(res)
            _flush()
            print(f"[locus_render_masked] SKIP-EXISTING {job['image']}: "
                  f"boxes={res['n_boxes']}", flush=True)
            continue
        # Attempt counter persisted across worker crashes: the official
        # placement loop exit(0)s (→ sapien teardown SIGSEGV) when 100 random
        # draws fail, and that outcome is deterministic for a given numpy RNG
        # state — so every retry reseeds np.random differently (np_seed =
        # attempt) and a job only gets skipped after 3 crashed attempts.
        attempt = 0
        if skip_existing and os.path.isfile(marker):
            try:
                with open(marker) as f:
                    attempt = int(f.read().strip() or 0)
            except ValueError:
                attempt = 1
            if attempt >= 3:
                os.remove(marker)
                res = {"config": job["config"], "image": job["image"],
                       "error": "crashed_3_attempts (placement exit(0) / "
                                "sapien SIGSEGV)"}
                print(f"[locus_render_masked] SKIP-CRASHED {job['image']}",
                      flush=True)
                results.append(res)
                _flush()
                continue
        try:
            os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
            with open(marker, "w") as f:
                f.write(str(attempt + 1))
            res = render_one(
                job["config"], job["image"], job["json_dir"],
                seed=job.get("seed", seed), sim_backend=sim_backend,
                strict=strict, np_seed=_job_np_seed(job["config"], attempt))
            os.remove(marker)
        except Exception as exc:                      # noqa: BLE001
            if os.path.isfile(marker):
                os.remove(marker)
            res = {"config": job["config"], "image": job["image"],
                   "error": f"{type(exc).__name__}: {exc}"}
            print(f"[locus_render_masked] FAILED {job['image']}: "
                  f"{res['error']}", flush=True)
            results.append(res)
            _flush()
            continue
        results.append(res)
        _flush()
        note = ""
        if res["transparent_cabinet_unlabeled"]:
            note += f" TRANSPARENT-GAP: {res['transparent_cabinet_unlabeled']}"
        if res.get("glass_blurred_review"):
            note += f" BLURRED-GLASS: {res['glass_blurred_review']}"
        if res.get("solid_hidden_ok"):
            note += f" solid-hidden-ok: {res['solid_hidden_ok']}"
        if res.get("opaque_container_leak"):
            note += (" OPAQUE-LEAK: " + ", ".join(
                f"{r['entity']}={r['visible_pixels']}px"
                for r in res["opaque_container_leak"]))
        print(f"[locus_render_masked] {res['image']}: "
              f"boxes={res['n_boxes']}{note}", flush=True)
    _flush()
    return results


def probe_jobs(jobs, seed=0, sim_backend=None, strict=True,
               summary_path=None):
    """Re-derive the glass/solid buckets for renders ALREADY on disk.

    The resume path (``--skip-existing``) has no renderer, so it falls back to
    the style heuristic, which both over-reports (transparent-style scenes keep
    solid lower cabinets) and under-reports (styles 2/7/9 also declare
    ``full_window_clear``). This mode rebuilds the scene purely to run the
    owner probe and leaves the existing PNG/label JSON untouched — RGB and
    labels go to a throwaway directory.

    Placement is reproduced, not re-rolled: ``np_seed`` is the same
    ``_job_np_seed(config, attempt)`` the original render used (``attempt``
    defaults to 0, i.e. succeeded first try), and the official draws are the
    only source of placement randomness — so the probed geometry matches the
    scene the on-disk image was rendered from. Only the rasterizer is
    non-deterministic, at a magnitude far below one target's pixel footprint.
    """
    import tempfile

    results = []

    def _flush():
        if not summary_path:
            return
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, summary_path)

    for job in jobs:
        with tempfile.TemporaryDirectory() as scratch:
            try:
                res = render_one(
                    job["config"],
                    os.path.join(scratch, os.path.basename(job["image"])),
                    scratch, seed=job.get("seed", seed),
                    sim_backend=sim_backend, strict=strict,
                    np_seed=_job_np_seed(job["config"],
                                         job.get("attempt", 0)))
            except Exception as exc:                  # noqa: BLE001
                res = {"config": job["config"],
                       "error": f"{type(exc).__name__}: {exc}"}
        res["image"] = job["image"]
        res.pop("mask_json", None)
        res["probe_only"] = True
        results.append(res)
        _flush()
        print(f"[locus_render_masked] PROBE {job['image']}: "
              f"clear={res.get('transparent_cabinet_unlabeled')} "
              f"blurred={res.get('glass_blurred_review')} "
              f"solid={res.get('solid_hidden_ok')} "
              f"leak={[r['entity'] for r in res.get('opaque_container_leak') or []]} "
              f"err={res.get('error')}", flush=True)
    _flush()
    return results


def patch_jobs(jobs, patch_classes=("table",), seed=0, sim_backend=None,
               strict=True, summary_path=None):
    """Re-export ONLY the given classes into label JSONs already on disk.

    Motivating case: the ``table`` caliber (island ``type: counter`` -> ``table``)
    landed after the 1160-image pilot was rendered. A full re-export would be
    wrong twice over — it would re-render every PNG for a class whose geometry is
    a STATIC fixture, and it would overwrite the human glass-review boxes that
    were merged into those same JSONs.

    So this mode rebuilds the scene, exports to a throwaway directory, and
    splices only the ``patch_classes`` shapes into the on-disk JSON: exporter
    shapes of those classes are replaced, every other shape (and every
    human-added shape, identified by ``attributes.label_source``) is left
    byte-identical. Placement is reproduced with the original
    ``_job_np_seed(config, attempt)`` exactly as in :func:`probe_jobs`; the
    on-disk PNG is never touched, so RGB/label alignment is preserved (a static
    fixture's box is invariant under the rasterizer's sub-pixel run-to-run
    noise, unlike an object the placement loop positions).

    A job whose scene contains none of ``patch_classes`` is still recorded, with
    ``n_added: 0`` — that is the audit evidence that e.g. wall counters never
    become ``table``.
    """
    import tempfile

    patch = {str(c) for c in patch_classes}
    results = []

    def _flush():
        if not summary_path:
            return
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, summary_path)

    for job in jobs:
        stem = os.path.splitext(os.path.basename(job["image"]))[0]
        target = os.path.join(job["json_dir"], stem + ".json")
        rec = {"config": job["config"], "image": job["image"],
               "label": target, "patch_classes": sorted(patch),
               "patch_only": True}
        if not os.path.isfile(target):
            rec["error"] = f"no label JSON at {target}"
            results.append(rec)
            _flush()
            print(f"[locus_render_masked] PATCH-SKIP {target}: missing",
                  flush=True)
            continue
        with tempfile.TemporaryDirectory() as scratch:
            try:
                res = render_one(
                    job["config"],
                    os.path.join(scratch, os.path.basename(job["image"])),
                    scratch, seed=job.get("seed", seed),
                    sim_backend=sim_backend, strict=strict,
                    np_seed=_job_np_seed(job["config"], job.get("attempt", 0)))
                with open(res["mask_json"]) as f:
                    fresh = json.load(f)
            except Exception as exc:                  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                results.append(rec)
                _flush()
                print(f"[locus_render_masked] PATCH-FAILED {target}: "
                      f"{rec['error']}", flush=True)
                continue

        new_shapes = [s for s in fresh.get("shapes", [])
                      if s.get("label") in patch]
        with open(target) as f:
            payload = json.load(f)
        old = payload.get("shapes", [])
        # Drop only EXPORTER shapes of the patched classes; a human-added shape
        # of the same class is a deliberate override and must survive.
        kept = [s for s in old
                if s.get("label") not in patch
                or (s.get("attributes") or {}).get("label_source")]
        removed = len(old) - len(kept)
        payload["shapes"] = kept + new_shapes
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, target)

        rec.update({"n_added": len(new_shapes), "n_removed": removed,
                    "n_shapes_before": len(old),
                    "n_shapes_after": len(payload["shapes"]),
                    "added": [{"label": s["label"],
                               "actor": (s.get("attributes") or {}).get("actor"),
                               "visible_frac": (s.get("attributes") or {}).get(
                                   "visible_frac")} for s in new_shapes]})
        results.append(rec)
        _flush()
        print(f"[locus_render_masked] PATCH {target}: +{len(new_shapes)} "
              f"-{removed} ({[a['actor'] for a in rec['added']]})", flush=True)
    _flush()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", help="JSON file: list of jobs "
                        '[{"config",..., "image",..., "json_dir",...}]')
    parser.add_argument("--config", help="single-shot: scene config yaml")
    parser.add_argument("--out-image", help="single-shot: output PNG path")
    parser.add_argument("--out-json-dir", help="single-shot: label JSON dir")
    parser.add_argument("--seed", type=int, default=0,
                        help="reset seed (official default 0)")
    parser.add_argument("--attempt", type=int, default=0,
                        help="single-shot: placement RNG attempt index")
    parser.add_argument("--sim-backend", default=None,
                        help="cpu|gpu (default: $VIKI_SIM_BACKEND or gpu)")
    parser.add_argument("--no-strict", action="store_true",
                        help="drop unmapped actors instead of raising")
    parser.add_argument("--skip-existing", action="store_true",
                        help="resume: jobs with PNG+JSON already on disk are "
                             "summarized from disk, not re-rendered")
    parser.add_argument("--summary", help="write per-image summary JSON here")
    parser.add_argument("--probe-only", action="store_true",
                        help="re-run the glass/solid owner probe for renders "
                             "already on disk; existing PNG/label JSON are "
                             "left untouched")
    parser.add_argument("--patch-classes", nargs="+", default=None,
                        help="re-export ONLY these classes into the label "
                             "JSONs already on disk (other shapes, incl. "
                             "human-added ones, are preserved); PNGs untouched")
    args = parser.parse_args(argv)

    strict = not args.no_strict
    if args.jobs:
        with open(args.jobs) as f:
            jobs = json.load(f)
        if args.patch_classes:
            patch_jobs(jobs, patch_classes=args.patch_classes, seed=args.seed,
                       sim_backend=args.sim_backend, strict=strict,
                       summary_path=args.summary)
        elif args.probe_only:
            probe_jobs(jobs, seed=args.seed, sim_backend=args.sim_backend,
                       strict=strict, summary_path=args.summary)
        else:
            render_jobs(jobs, seed=args.seed, sim_backend=args.sim_backend,
                        strict=strict, summary_path=args.summary,
                        skip_existing=args.skip_existing)
    else:
        if not (args.config and args.out_image and args.out_json_dir):
            parser.error("need --jobs or (--config --out-image --out-json-dir)")
        res = render_one(args.config, args.out_image, args.out_json_dir,
                         seed=args.seed, sim_backend=args.sim_backend,
                         strict=strict,
                         np_seed=_job_np_seed(args.config, args.attempt))
        print(json.dumps(res, indent=1))
    # sapien/vulkan teardown can SIGSEGV at interpreter shutdown after many
    # env create/close cycles (observed after a clean 150-render chunk). All
    # work and summaries are flushed above, so skip the destructors — same
    # spirit as the official solve()'s exit(0).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
