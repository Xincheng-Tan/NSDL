"""CPU-only unit tests for locus_mask_export + locus_render_masked hook logic.

No sapien/mani_skill import: entities and scene_data are faked with the same
attribute surface the ManiSkill structs expose (``.name``, ``.articulation``),
matching what ``BaseEnv.segmentation_id_map`` yields (Actor structs for static
actors, Link structs — with articulation-prefix-stripped bare names — for
articulation links). GPU-side capture is exercised by the real render smoke,
not here.

Run: python -m pytest script_general/test_locus_mask_export.py -q
"""

import json
import os
import sys

import numpy as np
import pytest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from locus_mask_export import (
    DROPPED_TYPES,
    FIXTURE_CLASSES,
    FULL34_CLASSES,
    MIN_VISIBLE_FRAC,
    MIN_VISIBLE_PIXELS_FIXTURE,
    MIN_VISIBLE_PIXELS_MOVABLE,
    _clip_to_rect,
    _hull,
    _plane_view_cos,
    _poly_area,
    _project_aabb_area,
    actor_view_probe,
    class_of_actor_name,
    export_annotations,
    is_island_counter,
    make_shape,
    masks_to_boxes,
    min_visible_pixels_for,
    summarize_class_counts,
)
from locus_render_masked import (
    build_fixture_types,
    classify_hidden_targets,
    glass_kind,
    hidden_in_cabinet_candidates,
    opaque_container_leaks,
    transparent_cabinet_gaps,
)


class FakeEntity:
    """Actor struct stand-in: bare ``.name``, no articulation."""

    def __init__(self, name, articulation=None):
        self.name = name
        self.articulation = articulation


class FakeArticulation:
    def __init__(self, name):
        self.name = name


def link(art, link_name):
    """Link struct stand-in: ManiSkill strips the articulation prefix, so the
    link's own name is bare ("door", "link_0", "base_link")."""
    return FakeEntity(link_name, articulation=art)


# ---------------------------------------------------------------------------
# class_of_actor_name (name-level)
# ---------------------------------------------------------------------------

def test_movable_object_names_resolve_to_full34():
    assert class_of_actor_name("banana_1") == "banana"
    assert class_of_actor_name("wooden_cutting_board_3") == "wooden_cutting_board"
    assert class_of_actor_name("cardboardbox_0") == "cardboardbox"


def test_robot_agent_uid_variants_resolve():
    # loader.name = uid or f"{uid}-agent-{idx}" (mani_skill base_agent.py)
    assert class_of_actor_name("unitree_h1-0") == "unitree_h1"
    assert class_of_actor_name("unitree_h1-agent-0") == "unitree_h1"
    assert class_of_actor_name("anymal_c-agent-1") == "anymal_c"
    assert class_of_actor_name("panda-0") == "panda"
    assert class_of_actor_name("fetch") == "fetch"


def test_fixture_role_names_need_fixture_types():
    ft = {"cab_1_main_group_0": "single_cabinet",
          "counter_main_main_group_0": "counter"}
    assert class_of_actor_name("cab_1_main_group_0", fixture_types=ft) == "cabinet"
    assert class_of_actor_name("counter_main_main_group_0", fixture_types=ft) is None


def test_island_counter_is_table_wall_counter_is_dropped():
    """The `table` LOCUS instructions name IS the island counter: VIKI's own
    layout yamls call that area ['table', 'kitchen island area'] on layouts
    1/3/6/9 (the four blueprint island layouts) and the VIKI-train hand labels
    drew `table` on it. The wall runs share the same `type: counter` and must
    stay unboxed, so the split is by entity name, not by type."""
    ft = {
        # real pilot entity names, one per island layout
        "island_counter_island_group_0": "counter",   # layout 1 one_wall_large
        "island_left_group_0": "counter",             # layout 3 L_shaped_large
        "island_island_group_0": "counter",           # layouts 6 / 9
        # wall runs in the same scenes
        "counter_main_main_group_0": "counter",
        "counter_1_front_group_0": "counter",         # G_shaped breakfast bar
        "counter_corner_left_group_0": "counter",
    }
    got = {n: class_of_actor_name(n, fixture_types=ft) for n in ft}
    assert got == {
        "island_counter_island_group_0": "table",
        "island_left_group_0": "table",
        "island_island_group_0": "table",
        "counter_main_main_group_0": None,
        "counter_1_front_group_0": None,
        "counter_corner_left_group_0": None,
    }
    assert "table" in FULL34_CLASSES
    # link paths resolve too (a counter is static, but the caliber must not
    # depend on that)
    assert class_of_actor_name("island_island_group_0/object",
                               fixture_types=ft) == "table"


def test_is_island_counter_predicate():
    for name in ("island_counter_island_group_0", "island_left_group_0",
                 "island_island_group_0", "Island_Island_Group_0"):
        assert is_island_counter(name)
    # the island's own SIDE PANEL is a `panel_cabinet`, dropped by type before
    # this predicate is ever consulted -- but the predicate must not be the
    # thing that saves it, so assert the type route still wins.
    assert class_of_actor_name("island_panel_cab_left_island_group_1_0",
                               fixture_types={
                                   "island_panel_cab_left_island_group_1_0":
                                   "panel_cabinet"}) is None
    for name in ("counter_main_main_group_0", "counter_1_front_group_0",
                 "cab_1_main_group_0", "stack_2_main_group_1_0"):
        assert not is_island_counter(name)


def test_dropped_object_classes_and_accessory_types_return_none():
    assert class_of_actor_name("hammer_2") is None
    ft = {"plant_1_room_0": "plant", "acc_1_main_group_0": "accessory"}
    assert class_of_actor_name("plant_1_room_0", fixture_types=ft) is None
    assert class_of_actor_name("acc_1_main_group_0", fixture_types=ft) is None


def test_unmapped_name_raises_in_strict_mode():
    with pytest.raises(KeyError):
        class_of_actor_name("mystery_asset_7")
    assert class_of_actor_name("mystery_asset_7", strict=False) is None


# ---------------------------------------------------------------------------
# masks_to_boxes (entity-level: links resolve via articulation name)
# ---------------------------------------------------------------------------

def _seg_and_map(entries, shape=(60, 80)):
    """entries: list of (sid, entity, [(y0,y1,x0,x1), ...] pixel patches)."""
    seg = np.zeros(shape, dtype=np.int64)
    id_map = {}
    for sid, entity, patches in entries:
        for y0, y1, x0, x1 in patches:
            seg[y0:y1, x0:x1] = sid
        id_map[sid] = entity
    return seg, id_map


def test_cabinet_links_union_to_one_box_via_articulation():
    cab = FakeArticulation("cab_1_main_group_0")
    ft = {"cab_1_main_group_0": "single_cabinet"}
    seg, id_map = _seg_and_map([
        (3, link(cab, "door"), [(10, 20, 10, 20)]),        # 100 px
        (4, link(cab, "body"), [(20, 40, 10, 30)]),        # 400 px
        (5, link(cab, "handle_link"), [(12, 14, 18, 20)]),  # 4 px, below floor
    ])
    boxes, stats = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["label"] == "cabinet"
    # union of the two substantial links; the 4-px handle sliver is inside them
    # anyway, so the frame is unchanged and the caliber filter is a no-op here.
    assert b["xyxy"] == [10.0, 10.0, 30.0, 40.0]
    assert b["n_links"] == 3 and b["n_links_kept"] == 2
    assert stats["n_boxes"] == 1


def test_stray_carcass_link_does_not_inflate_drawer_box():
    """§5.3-5 caliber: a drawer's carcass leaks a few pixels through the gap,
    far to the side of the visible front face. Unioning it in is what made the
    shipped pilot boxes extend "into the wall" on some samples but not others
    (train_00301 stack_4_main_group_3: 51x58 front + 7-px carcass -> 108x66)."""
    drw = FakeArticulation("stack_4_main_group_3_0")
    ft = {"stack_4_main_group_3_0": "drawer"}
    seg, id_map = _seg_and_map([
        (3, link(drw, "inner_box"), [(10, 30, 10, 30)]),   # 400 px front face
        (4, link(drw, "object"), [(10, 12, 60, 63)]),      # 6 px stray carcass
    ])
    boxes, _ = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert [b["xyxy"] for b in boxes] == [[10.0, 10.0, 30.0, 30.0]]
    assert boxes[0]["n_links_kept"] == 1

    # ...and with the filter off the old inflated frame comes back, so the test
    # pins the mechanism, not just the outcome.
    old, _ = masks_to_boxes(seg, id_map, fixture_types=ft, link_caliber=False)
    assert old[0]["xyxy"] == [10.0, 10.0, 63.0, 30.0]


def test_caliber_filter_always_keeps_largest_link():
    """A cabinet seen entirely through its own glass can have EVERY link below
    the threshold. Dropping all of them would delete the box (and, in an early
    draft, raised ValueError on min() of an empty sequence)."""
    cab = FakeArticulation("cab_2_main_group_0")
    ft = {"cab_2_main_group_0": "hinge_cabinet"}
    seg, id_map = _seg_and_map([
        (3, link(cab, "hingeleftdoor"), [(10, 14, 10, 14)]),   # 16 px
        (4, link(cab, "hingerightdoor"), [(10, 13, 20, 24)]),  # 12 px
    ])
    # the absolute floor is off here on purpose: this test pins the LINK
    # caliber, and 28 px of cabinet is below `MIN_VISIBLE_PIXELS_FIXTURE`, so
    # leaving it on would drop the box before the mechanism under test ran.
    boxes, _ = masks_to_boxes(seg, id_map, fixture_types=ft,
                              min_visible_pixels_fixture=0)
    assert len(boxes) == 1 and boxes[0]["n_links_kept"] == 1
    assert boxes[0]["xyxy"] == [10.0, 10.0, 14.0, 14.0]        # the larger one


def test_near_invisible_fixture_box_is_dropped_as_occluded():
    """The under-sink base cabinet: a large frame holding a scatter of specks
    (train_00302 stack_2_main_group_1 = 109 visible px over a 37128-px box,
    frac 0.0029). No human can draw it, so it must not be trained on. The
    MIN_BOX_AREA/-SIDE guard cannot catch it — the box is huge."""
    cab = FakeArticulation("stack_2_main_group_1_0")
    ft = {"stack_2_main_group_1_0": "hinge_cabinet"}
    # 4 blocks totalling 144 px spread over a 122x122 frame -> frac 0.0097,
    # i.e. the train_00306 island cabinet (144/14720). Above the absolute floor
    # on purpose, so what this test pins is the FRACTION guard: the two are
    # independent and the 109-px train_00302 case now trips the floor first.
    seg, id_map = _seg_and_map([
        (3, link(cab, "object"), [(10, 16, 10, 16), (10, 16, 126, 132),
                                  (126, 132, 10, 16), (126, 132, 126, 132)]),
    ], shape=(140, 140))
    boxes, stats = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert boxes == [] and stats["under_min_pixels"] == []
    assert len(stats["occluded"]) == 1
    dropped = stats["occluded"][0]
    assert dropped["label"] == "cabinet" and dropped["visible_frac"] < 0.01
    assert dropped["visible_pixels"] >= MIN_VISIBLE_PIXELS_FIXTURE

    # a densely-filled cabinet of the SAME frame size survives untouched
    seg2, id_map2 = _seg_and_map(
        [(3, link(cab, "object"), [(10, 132, 10, 132)])], shape=(140, 140))
    boxes2, stats2 = masks_to_boxes(seg2, id_map2, fixture_types=ft)
    assert len(boxes2) == 1 and stats2["occluded"] == []
    assert boxes2[0]["visible_frac"] > 0.9

    # and the guard is switchable off for diagnostics
    boxes3, _ = masks_to_boxes(seg, id_map, fixture_types=ft,
                               min_visible_frac=0)
    assert len(boxes3) == 1
    assert MIN_VISIBLE_FRAC > 0        # default is ON


def test_open_shelf_and_island_panel_are_not_cabinet():
    """The two "wall boards" and the "table side" reported as mislabeled
    cabinets: robocasa OpenCabinet (doorless shelf board, = the `rack` semantic
    region of §4.2) and PanelCabinet (`solid_body`, unopenable island side
    panel). Neither carries cabinet semantics, so neither is a detector box."""
    ft = {"shelves_main_group_0": "open_cabinet",
          "island_panel_cab_left_island_group_1_0": "panel_cabinet",
          "cab_1_main_group_0": "hinge_cabinet"}
    shelf = FakeArticulation("shelves_main_group_0")
    panel = FakeArticulation("island_panel_cab_left_island_group_1_0")
    cab = FakeArticulation("cab_1_main_group_0")
    seg, id_map = _seg_and_map([
        (3, link(shelf, "object"), [(2, 12, 2, 70)]),
        (4, link(panel, "object"), [(20, 45, 2, 40)]),
        (5, link(cab, "hingeleftdoor"), [(20, 45, 45, 75)]),
    ])
    boxes, stats = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert [b["label"] for b in boxes] == ["cabinet"]
    assert [b["actor"] for b in boxes] == ["cab_1_main_group_0"]
    assert sorted(stats["dropped_actors"]) == [
        "island_panel_cab_left_island_group_1_0/object",
        "shelves_main_group_0/object"]
    for t in ("open_cabinet", "panel_cabinet"):
        assert t in DROPPED_TYPES


def test_robot_links_resolve_via_agent_articulation():
    h1 = FakeArticulation("unitree_h1-agent-0")
    seg, id_map = _seg_and_map([
        (7, link(h1, "base_link"), [(0, 10, 0, 10)]),
        (8, link(h1, "torso_link"), [(10, 30, 2, 8)]),
    ])
    boxes, _ = masks_to_boxes(seg, id_map)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "unitree_h1"


def test_occluded_object_partial_patches_stay_one_tight_box():
    seg, id_map = _seg_and_map([
        (2, FakeEntity("bread_0"), [(5, 10, 5, 15), (5, 10, 25, 30)]),
    ])
    boxes, _ = masks_to_boxes(seg, id_map)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "bread"
    assert boxes[0]["xyxy"] == [5.0, 5.0, 30.0, 10.0]


def test_s4_hidden_object_absent_from_buffer_gets_no_box():
    cab = FakeArticulation("cab_1_main_group_0")
    ft = {"cab_1_main_group_0": "single_cabinet"}
    # banana exists in the scene (GT) but has zero visible pixels: only the
    # cabinet occupies the buffer. §5.3-3 caliber: no box for the hidden item.
    seg, id_map = _seg_and_map([
        (3, link(cab, "door"), [(10, 40, 10, 40)]),
        # id 9 (banana) intentionally has no pixels
    ])
    id_map[9] = FakeEntity("banana_1")
    boxes, _ = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert [b["label"] for b in boxes] == ["cabinet"]


def test_dropped_and_sliver_and_background_all_reported():
    ft = {"counter_main_main_group_0": "counter",
          "wall_main_room_0": "wall"}
    seg, id_map = _seg_and_map([
        (2, FakeEntity("counter_main_main_group_0"), [(0, 30, 0, 80)]),
        (3, FakeEntity("wall_main_room_0"), [(30, 60, 0, 80)]),
        (4, FakeEntity("apple_0"), [(5, 6, 5, 6)]),   # 1px sliver
    ])
    boxes, stats = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert boxes == []
    assert len(stats["dropped_actors"]) == 2
    assert len(stats["slivers"]) == 1
    assert stats["slivers"][0]["label"] == "apple"
    assert stats["background_pixels"] == 0    # fully covered scene


def test_unmapped_entity_raises_strict_drops_nonstrict():
    seg, id_map = _seg_and_map([(2, FakeEntity("mystery_9"), [(0, 10, 0, 10)])])
    with pytest.raises(KeyError):
        masks_to_boxes(seg, id_map)
    boxes, stats = masks_to_boxes(seg, id_map, strict=False)
    assert boxes == [] and stats["dropped_actors"] == ["mystery_9"]


# ---------------------------------------------------------------------------
# X-AnyLabeling JSON export
# ---------------------------------------------------------------------------

def test_export_annotations_payload_and_empty_json(tmp_path):
    seg, id_map = _seg_and_map([
        (2, FakeEntity("apple_0"), [(5, 25, 5, 25)]),
    ])
    json_path, boxes, _ = export_annotations(
        seg, id_map, image_path="/x/y/00042.png", out_dir=str(tmp_path))
    assert os.path.basename(json_path) == "00042.json"
    payload = json.load(open(json_path))
    # keys prepare_yolo26_finetune_dataset.py / the review GUI rely on
    assert set(payload) >= {"version", "flags", "shapes", "imagePath",
                            "imageHeight", "imageWidth"}
    assert payload["imageHeight"] == 60 and payload["imageWidth"] == 80
    assert payload["flags"]["locus_label_source"] == "renderer_gt"
    (shape,) = payload["shapes"]
    assert shape["label"] == "apple" and shape["shape_type"] == "rectangle"
    assert len(shape["points"]) == 4
    # attribution must SURVIVE onto disk: without it, tracing a suspect box back
    # to its fixture role or re-filtering by visible fraction needs a re-render,
    # which render non-determinism makes non-exact.
    assert shape["attributes"]["actor"] == "apple_0"
    assert shape["attributes"]["visible_pixels"] == 400
    assert 0 < shape["attributes"]["visible_frac"] <= 1.0

    # zero-instance image still writes a JSON (YOLO.md §5.3-8)
    seg2 = np.zeros((60, 80), dtype=np.int64)
    json_path2, boxes2, _ = export_annotations(
        seg2, {}, image_path="/x/y/00043.png", out_dir=str(tmp_path))
    assert json.load(open(json_path2))["shapes"] == []


def test_summarize_class_counts_covers_full34():
    counts = summarize_class_counts([[{"label": "apple"}], [{"label": "apple"},
                                     {"label": "cabinet"}]])
    assert counts["apple"] == 2 and counts["cabinet"] == 1
    assert set(counts) == set(FULL34_CLASSES)


# ---------------------------------------------------------------------------
# build_fixture_types (injected reverse map, no robocasa import)
# ---------------------------------------------------------------------------

class FakeFixtureA:      # articulated (cabinet-like)
    def __init__(self, name):
        self.articulation = FakeArticulation(name + "_0")


class FakeFixtureS:      # static (counter-like)
    def __init__(self, name):
        self.actor = FakeEntity(name + "_0")


def test_build_fixture_types_keys_by_built_entity_name():
    scene_data = [{"fixtures": {
        "cab_1_main_group": FakeFixtureA("cab_1_main_group"),
        "counter_main_main_group": FakeFixtureS("counter_main_main_group"),
    }}]
    tmap = {FakeFixtureA: "single_cabinet", FakeFixtureS: "counter"}
    ft = build_fixture_types(scene_data, type_of_class=tmap)
    assert ft["cab_1_main_group_0"] == "single_cabinet"
    assert ft["counter_main_main_group_0"] == "counter"
    # role-name fallback keys also present
    assert ft["cab_1_main_group"] == "single_cabinet"


def test_build_fixture_types_unmapped_class_raises():
    class Unknown:
        pass
    with pytest.raises(KeyError):
        build_fixture_types([{"fixtures": {"x": Unknown()}}],
                            type_of_class={FakeFixtureA: "single_cabinet"})


def test_dropped_types_cover_accessory_aliases():
    # regression: robocasa registers plant/paper_towel/knife_block/... as
    # Accessory; the reverse map yields "accessory" (first key), and every
    # alias that can surface must be droppable, not a strict error.
    for t in ("accessory", "plant", "paper_towel", "knife_block",
              "utensil_holder", "stool"):
        assert t in DROPPED_TYPES


# ---------------------------------------------------------------------------
# transparent-cabinet caliber divergence (verified on real smoke F8-S1 style 11)
# ---------------------------------------------------------------------------

def _cfg(style_idx, init_pos):
    return {"scene": {"env": {"style_idx": style_idx}},
            "gt": {"init_pos": init_pos}}


def test_transparent_cabinet_gap_flagged():
    # peach sits in a transparent cabinet but the glass door owns the pixels →
    # no box exported → must be flagged for human labeling (YOLO.md §5.3-3).
    cfg = _cfg(11, {"peach_1": ["cabinet"], "knife_2": ["kitchen work area"]})
    boxes = [{"label": "knife"}, {"label": "cabinet"}]
    assert transparent_cabinet_gaps(cfg, boxes) == ["peach_1"]


def test_transparent_gap_not_flagged_when_boxed_or_opaque():
    # boxed through-glass object (e.g. door open) → no gap
    cfg = _cfg(4, {"peach_1": ["cabinet"]})
    assert transparent_cabinet_gaps(cfg, [{"label": "peach"}]) == []
    # opaque style: hidden-object no-box is CORRECT S4 caliber, not a gap
    cfg = _cfg(7, {"banana_1": ["cabinet"]})
    assert transparent_cabinet_gaps(cfg, []) == []
    # transparent style but object not in a cabinet → not a gap
    cfg = _cfg(11, {"bowl_2": ["kitchen island area"]})
    assert transparent_cabinet_gaps(cfg, []) == []


def _box(label, actor, px=100, frac=0.5):
    return {"label": label,
            "attributes": {"actor": actor, "visible_pixels": px,
                           "visible_frac": frac}}


def test_gaps_are_keyed_by_actor_not_label():
    """The S4 counterfactual pair puts a SECOND instance of the target class
    outside the cabinet on purpose, so label keying made the in-cabinet
    instance disappear from the probe (pilot ood_s: bread_1 in the cabinet,
    bread_0 on the counter → `bread` boxed → bread_1 silently dropped)."""
    cfg = _cfg(11, {"bread_1": ["cabinet"], "bread_0": ["kitchen work area"]})
    assert hidden_in_cabinet_candidates(cfg, [_box("bread", "bread_0")]) \
        == ["bread_1"]
    # the in-cabinet instance itself boxed → not a gap
    assert hidden_in_cabinet_candidates(
        cfg, [_box("bread", "bread_0"), _box("bread", "bread_1")]) == []
    # a caller with labels only (no actor to key on) keeps the old behaviour
    assert hidden_in_cabinet_candidates(cfg, [{"label": "bread"}]) == []


def test_opaque_container_leak_is_flagged_with_its_pixel_size():
    """P0-1: an in-cabinet target that produced pixels under an opaque style
    breaks S4's "must open the container to observe" premise. Sized, because a
    9-pixel graze and a 2072-pixel protrusion are the same defect at very
    different severities (both measured in the pilot)."""
    cfg = _cfg(2, {"bread_1": ["cabinet"], "bread_0": ["kitchen work area"]})
    leaks = opaque_container_leaks(
        cfg, [_box("bread", "bread_0", 5000, 0.9),
              _box("bread", "bread_1", 497, 0.4844)])
    assert leaks == [{"entity": "bread_1", "style_idx": 2,
                      "visible_pixels": 497, "visible_frac": 0.4844}]


def test_opaque_container_leak_ignores_glass_and_clean_scenes():
    cfg_op = _cfg(2, {"bread_1": ["cabinet"]})
    # no box for the in-cabinet target → the container did its job
    assert opaque_container_leaks(cfg_op, [_box("cabinet", "cab_main_0")]) == []
    # transparent style: through-glass pixels are the OFFICIAL behaviour, and
    # §5.3-3 wants them labeled, so they are not leaks
    for style in (4, 11):
        cfg = _cfg(style, {"bread_1": ["cabinet"]})
        assert opaque_container_leaks(cfg, [_box("bread", "bread_1")]) == []
    # object not in a cabinet at all
    cfg = _cfg(2, {"bowl_2": ["kitchen island area"]})
    assert opaque_container_leaks(cfg, [_box("bowl", "bowl_2")]) == []


def test_opaque_container_leak_matches_the_pilot_ground_truth():
    """Reproduces the 9 leaks the pilot's on-disk labels actually carry, so the
    gate is pinned against measured data rather than against its own logic."""
    pilot = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)),
                         "data", "lnsdl_bench", "pilot")
    if not os.path.isdir(pilot):
        pytest.skip("pilot render tree not present")
    import yaml
    found = {}
    for split in ("train", "test_id", "ood_s", "ood_x"):
        man_path = os.path.join(pilot, split, "render_manifest.json")
        if not os.path.isfile(man_path):
            continue
        with open(man_path) as f:
            manifest = json.load(f)
        for rec in manifest:
            label_path = os.path.join(pilot, split, rec["mask_json"])
            if not os.path.isfile(label_path):
                continue
            with open(os.path.join(pilot, split, rec["config"])) as f:
                cfg = yaml.safe_load(f)
            with open(label_path) as f:
                boxes = json.load(f)["shapes"]
            for leak in opaque_container_leaks(cfg, boxes):
                found[f"{split}/{rec['sample_index']:05d}"] = leak["entity"]
    assert found == {"ood_s/00060": "bread_1", "ood_s/00064": "banana_1",
                     "ood_s/00068": "banana_1", "ood_s/00078": "banana_1",
                     "ood_s/00086": "bread_1", "ood_s/00094": "banana_1",
                     "ood_s/00096": "banana_1", "ood_s/00098": "bread_1",
                     "ood_x/00010": "bread_1"}


def test_opaque_container_leak_reads_the_flat_live_box_form():
    """`export_annotations` returns FLAT boxes (`{"label", "actor",
    "visible_pixels", ...}`); only `make_shape` nests them under `attributes`
    for the label JSON. Reading only the nested form fails SILENTLY on the live
    render path — measured 2026-07-29: `render_one` reported 0 leaks on the nine
    configs that score 9/9 when the same labels are read back off disk."""
    cfg = _cfg(2, {"bread_1": ["cabinet"], "bread_0": ["kitchen work area"]})
    flat = [{"label": "bread", "actor": "bread_0", "visible_pixels": 5000,
             "visible_frac": 0.9, "xyxy": [0, 0, 10, 10]},
            {"label": "bread", "actor": "bread_1", "visible_pixels": 497,
             "visible_frac": 0.4844, "xyxy": [0, 0, 5, 5]}]
    assert opaque_container_leaks(cfg, flat) == [
        {"entity": "bread_1", "style_idx": 2,
         "visible_pixels": 497, "visible_frac": 0.4844}]
    # ...and the same flat form must key the glass/solid probe by actor too
    assert hidden_in_cabinet_candidates(cfg, flat[:1]) == ["bread_1"]


def test_in_cabinet_targets_ignore_fixture_entities():
    """`gt.init_pos` also carries fixtures whose "init position" is the cabinet
    itself (`cabinet_0`). Those own their pixels and are never leaks; before the
    movable-object filter they only stayed out by accident, because label keying
    matched them against the `cabinet` boxes."""
    cfg = _cfg(2, {"cabinet_0": ["cabinet"], "bread_1": ["cabinet"]})
    cfg["objects"] = [{"name": "bread_1", "type": "bread"}]
    boxes = [_box("cabinet", "cab_1_left_group_0", 4000, 0.6)]
    assert hidden_in_cabinet_candidates(cfg, boxes) == ["bread_1"]
    assert opaque_container_leaks(cfg, boxes) == []


# ---------------------------------------------------------------------------
# glass-vs-solid occlusion routing (owner probe), verified on pilot 00301-00310
# ---------------------------------------------------------------------------
class FakeCabinetFixture:
    def __init__(self, panel_type=None, opacity=None):
        self.panel_type = panel_type
        self.panel_config = {} if opacity is None else {"opacity": opacity}


def test_hidden_candidates_are_style_independent():
    """The style gate was the bug: it flagged EVERY in-cabinet target of a
    style-{4,11} scene as human-labelable, including ones behind an opaque hood
    (pilot 00301 apple_1 → hood_main_group) — and missed styles 2/7/9, which
    also put full_window_clear on cab_top_small."""
    cfg = _cfg(7, {"banana_1": ["cabinet"], "knife_2": ["kitchen work area"]})
    assert hidden_in_cabinet_candidates(cfg, [{"label": "knife"}]) == ["banana_1"]
    assert transparent_cabinet_gaps(cfg, [{"label": "knife"}]) == []


def test_glass_kind_reads_panel_type_and_opacity():
    # fixture_registry/cabinet.yaml: clear=0.1, blurred=0.4
    clear = FakeCabinetFixture("full_window", 0.1)
    blurred = FakeCabinetFixture("full_window", 0.4)
    assert glass_kind(clear, "hingerightdoor") == "glass_clear"
    assert glass_kind(blurred, "hingeleftdoor") == "glass_blurred"
    # same fixture, non-door link → the carcass is opaque
    assert glass_kind(clear, "object") is None
    # solid-panel cabinet door → not see-through
    assert glass_kind(FakeCabinetFixture("slab"), "hingedoor") is None
    assert glass_kind(FakeCabinetFixture("raised"), "hingedoor") is None
    assert glass_kind(None, "hingedoor") is None


def test_classify_hidden_targets_splits_glass_from_solid():
    """Pilot evidence: 00306/00310 banana → hingerightdoor of a style-11
    (clear) cabinet, human CAN label; 00305/00309 → style-4 (blurred), human
    decides; 00301 apple → hood, 00302 bread → carcass, both unlabelable."""
    index = {
        "cab_main_main_group_0": FakeCabinetFixture("full_window", 0.1),
        "cab_2_main_group_0": FakeCabinetFixture("full_window", 0.4),
        "hood_main_group_0": FakeCabinetFixture(None),
        "cab_1_left_group_0": FakeCabinetFixture("slab"),
    }
    owners = {
        "banana_1": {"link": "hingerightdoor",
                     "articulation": "cab_main_main_group_0", "pixel": [1, 2]},
        "apple_3": {"link": "hingeleftdoor",
                    "articulation": "cab_2_main_group_0", "pixel": [3, 4]},
        "apple_1": {"link": "object", "articulation": "hood_main_group_0",
                    "pixel": [5, 6]},
        "bread_1": {"link": "object", "articulation": "cab_1_left_group_0",
                    "pixel": [7, 8]},
        "pear_9": {"offscreen": True, "pixel": [-4, 12]},
    }
    out = classify_hidden_targets(list(owners), owners, index)
    assert [r["entity"] for r in out["glass_clear"]] == ["banana_1"]
    assert [r["entity"] for r in out["glass_blurred"]] == ["apple_3"]
    assert sorted(r["entity"] for r in out["solid"]) == ["apple_1", "bread_1"]
    assert [r["entity"] for r in out["unresolved"]] == ["pear_9"]


def test_classify_hidden_targets_unresolved_is_conservative():
    # a candidate the probe could not place must never be silently called
    # "solid" (= no human work needed); it goes to unresolved for review.
    out = classify_hidden_targets(["apple_1"], {}, {})
    assert [r["entity"] for r in out["unresolved"]] == ["apple_1"]
    assert out["solid"] == [] and out["glass_clear"] == []


# ---------------------------------------------------------------------------
# render_jobs resume semantics (skip-existing / crash markers), no renderer
# ---------------------------------------------------------------------------

def test_render_jobs_resume_and_crash_marker(tmp_path, monkeypatch):
    import locus_render_masked as lrm
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(_cfg(7, {"apple_1": ["counter"]})))
    jdir = str(tmp_path / "gt")

    def job(stem):
        return {"config": str(cfg_path),
                "image": str(tmp_path / f"{stem}.png"), "json_dir": jdir}

    calls = []

    def fake_render_one(config, image, json_dir, **kw):
        calls.append((os.path.basename(image), kw.get("np_seed")))
        os.makedirs(json_dir, exist_ok=True)
        open(image, "wb").write(b"png")
        stem = os.path.splitext(os.path.basename(image))[0]
        json.dump({"shapes": [{"label": "apple"}]},
                  open(os.path.join(json_dir, stem + ".json"), "w"))
        return {"config": config, "image": image, "mask_json": "x",
                "n_boxes": 1, "class_counts": {"apple": 1}, "n_slivers": 0,
                "unmapped_ids": [], "dropped_actors": [],
                "transparent_cabinet_unlabeled": []}

    monkeypatch.setattr(lrm, "render_one", fake_render_one)

    # job a: complete products already on disk → summarized, not re-rendered
    fake_render_one(str(cfg_path), job("a")["image"], jdir)
    calls.clear()
    # job b: marker with 1 crashed attempt → RETRIED with a DIFFERENT np seed
    open(job("b")["image"] + ".inprogress", "w").write("1")
    # job c: marker with 3 crashed attempts → given up (skip-crashed)
    open(job("c")["image"] + ".inprogress", "w").write("3")
    # job d: fresh → rendered with attempt-0 seed

    res = lrm.render_jobs([job("a"), job("b"), job("c"), job("d")],
                          skip_existing=True)
    rendered = [c[0] for c in calls]
    assert rendered == ["b.png", "d.png"]
    assert res[0].get("skipped_existing") and res[0]["n_boxes"] == 1
    # b retried on attempt 1 → np seed differs from a fresh attempt-0 job
    seed_b = calls[0][1]
    seed_d = calls[1][1]
    assert seed_b != lrm._job_np_seed(str(cfg_path), 0) == seed_d
    assert "crashed_3_attempts" in res[2]["error"]
    for stem in ("b", "c", "d"):
        assert not os.path.exists(job(stem)["image"] + ".inprogress")

    # c has no marker and no products now → real retry on a later pass
    calls.clear()
    res2 = lrm.render_jobs([job("c")], skip_existing=True)
    assert [c[0] for c in calls] == ["c.png"] and res2[0]["n_boxes"] == 1


def test_resumed_summary_marks_glass_verdict_provisional(tmp_path, monkeypatch):
    """A --skip-existing summary has no renderer, so its glass buckets come from
    the style heuristic and MUST be labelled as such: a run that resumed most of
    its jobs would otherwise publish heuristic verdicts as authoritative (the
    bug that produced the discarded 68/70-image queues)."""
    import locus_render_masked as lrm
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(_cfg(4, {"apple_1": ["cabinet"]})))
    jdir = str(tmp_path / "gt")
    image = str(tmp_path / "a.png")
    os.makedirs(jdir, exist_ok=True)
    open(image, "wb").write(b"png")
    json.dump({"shapes": []}, open(os.path.join(jdir, "a.json"), "w"))

    res = lrm.render_jobs(
        [{"config": str(cfg_path), "image": image, "json_dir": jdir}],
        skip_existing=True)[0]

    assert res["skipped_existing"]
    assert res["glass_verdict_source"] == "style_heuristic_provisional"
    assert res["hidden_target_probe"] == {"style_heuristic_only": True}
    # style 4 is transparent, so the heuristic DOES flag it — which is exactly
    # why the provisional marker matters: the verdict looks like a real one.
    assert res["transparent_cabinet_unlabeled"] == ["apple_1"]


def test_probe_jobs_leaves_on_disk_products_untouched(tmp_path, monkeypatch):
    """--probe-only re-derives the glass buckets for renders already on disk: it
    must reproduce the ORIGINAL placement seed (not re-roll it) and must not
    touch the existing PNG / label JSON."""
    import locus_render_masked as lrm
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(_cfg(4, {"apple_1": ["cabinet"]})))
    jdir = tmp_path / "gt"
    jdir.mkdir()
    image = tmp_path / "a.png"
    image.write_bytes(b"original-png")
    (jdir / "a.json").write_text(json.dumps({"shapes": []}))

    seen = {}

    def fake_render_one(config, out_image, out_json_dir, **kw):
        seen["out_image"] = out_image
        seen["np_seed"] = kw.get("np_seed")
        open(out_image, "wb").write(b"scratch-png")   # writes to the scratch dir
        return {"config": config, "image": out_image, "mask_json": "scratch.json",
                "n_boxes": 0, "class_counts": {}, "n_slivers": 0,
                "unmapped_ids": [], "dropped_actors": [],
                "transparent_cabinet_unlabeled": ["apple_1"],
                "glass_blurred_review": [], "solid_hidden_ok": [],
                "hidden_target_probe": {"glass_clear": [{"entity": "apple_1"}]},
                "glass_verdict_source": "owner_probe"}

    monkeypatch.setattr(lrm, "render_one", fake_render_one)
    summary = tmp_path / "probe_summary.json"
    res = lrm.probe_jobs(
        [{"config": str(cfg_path), "image": str(image), "json_dir": str(jdir)}],
        summary_path=str(summary))[0]

    # placement reproduced with the attempt-0 seed the original render used
    assert seen["np_seed"] == lrm._job_np_seed(str(cfg_path), 0)
    # RGB went to a throwaway dir, never over the on-disk render
    assert seen["out_image"] != str(image)
    assert image.read_bytes() == b"original-png"
    assert json.loads((jdir / "a.json").read_text()) == {"shapes": []}
    # verdict points at the real image and carries no label-file claim
    assert res["probe_only"] and res["image"] == str(image)
    assert "mask_json" not in res
    assert res["transparent_cabinet_unlabeled"] == ["apple_1"]
    assert res["glass_verdict_source"] == "owner_probe"
    assert json.loads(summary.read_text())[0]["image"] == str(image)


def test_patch_jobs_replaces_only_patched_classes(tmp_path, monkeypatch):
    """`--patch-classes table` landed the island caliber onto 1160 already
    rendered images. It must (a) add the fresh `table` box, (b) drop a stale
    EXPORTER `table` box, (c) leave every other exporter shape byte-identical,
    (d) preserve a human-added shape even of a patched class, and (e) never
    touch the PNG (a static fixture's box is invariant to render noise)."""
    import locus_render_masked as lrm
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(_cfg(7, {"apple_1": ["table"]})))
    jdir = tmp_path / "gt"
    jdir.mkdir()
    image = tmp_path / "a.png"
    image.write_bytes(b"original-png")
    human = {"label": "apple", "points": [[9, 9], [9, 9], [9, 9], [9, 9]],
             "attributes": {"label_source": "human_glass_review"}}
    human_table = {"label": "table", "points": [[1, 1], [1, 1], [1, 1], [1, 1]],
                   "attributes": {"label_source": "human_glass_review"}}
    stale = {"label": "table", "points": [[0, 0], [5, 0], [5, 5], [0, 5]],
             "attributes": {"actor": "counter_main_main_group_0"}}
    other = {"label": "cabinet", "points": [[2, 2], [3, 2], [3, 3], [2, 3]],
             "attributes": {"actor": "cab_1_main_group_0"}}
    (jdir / "a.json").write_text(json.dumps(
        {"flags": {"locus_label_source": "renderer_gt"},
         "imagePath": str(image),
         "shapes": [other, stale, human, human_table]}))

    seen = {}

    def fake_render_one(config, out_image, out_json_dir, **kw):
        seen["out_image"] = out_image
        seen["np_seed"] = kw.get("np_seed")
        open(out_image, "wb").write(b"scratch-png")
        mask_json = os.path.join(out_json_dir, "a.json")
        json.dump({"shapes": [
            {"label": "table", "points": [[10, 10], [40, 10], [40, 30], [10, 30]],
             "attributes": {"actor": "island_island_group_0",
                            "visible_frac": 0.8}},
            {"label": "drawer", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
             "attributes": {"actor": "stack_1_main_group_0_0"}},
        ]}, open(mask_json, "w"))
        return {"config": config, "image": out_image, "mask_json": mask_json,
                "n_boxes": 2, "class_counts": {}, "n_slivers": 0,
                "unmapped_ids": [], "dropped_actors": [],
                "transparent_cabinet_unlabeled": []}

    monkeypatch.setattr(lrm, "render_one", fake_render_one)
    summary = tmp_path / "patch_summary.json"
    res = lrm.patch_jobs(
        [{"config": str(cfg_path), "image": str(image), "json_dir": str(jdir)}],
        patch_classes=("table",), summary_path=str(summary))[0]

    payload = json.loads((jdir / "a.json").read_text())
    labels = [(s["label"], (s.get("attributes") or {}).get("actor")
               or (s.get("attributes") or {}).get("label_source"))
              for s in payload["shapes"]]
    assert labels == [
        ("cabinet", "cab_1_main_group_0"),          # untouched
        ("apple", "human_glass_review"),            # untouched
        ("table", "human_glass_review"),            # human override survives
        ("table", "island_island_group_0"),         # fresh exporter box
    ]
    # the stale wall-counter `table` box is gone, and `drawer` was NOT imported
    assert not any((s.get("attributes") or {}).get("actor")
                   == "counter_main_main_group_0" for s in payload["shapes"])
    assert not any(s["label"] == "drawer" for s in payload["shapes"])
    assert payload["flags"]["locus_label_source"] == "renderer_gt"
    # RGB untouched, scratch dir used, original placement seed reproduced
    assert image.read_bytes() == b"original-png"
    assert seen["out_image"] != str(image)
    assert seen["np_seed"] == lrm._job_np_seed(str(cfg_path), 0)
    assert res["n_added"] == 1 and res["n_removed"] == 1
    assert res["patch_only"] and res["added"][0]["actor"] == "island_island_group_0"
    assert json.loads(summary.read_text())[0]["label"] == str(jdir / "a.json")


def test_patch_jobs_records_zero_and_missing_without_dying(tmp_path, monkeypatch):
    """A wall-counter-only scene must record `n_added: 0` (the audit evidence
    that wall runs never become `table`), and a job whose label JSON is absent
    must not abort the batch."""
    import locus_render_masked as lrm
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(_cfg(7, {"apple_1": ["counter"]})))
    jdir = tmp_path / "gt"
    jdir.mkdir()
    keep = {"label": "cabinet", "points": [[2, 2], [3, 2], [3, 3], [2, 3]]}
    (jdir / "b.json").write_text(json.dumps({"shapes": [keep]}))

    def fake_render_one(config, out_image, out_json_dir, **kw):
        mask_json = os.path.join(out_json_dir, "x.json")
        json.dump({"shapes": [{"label": "cabinet", "points": []}]},
                  open(mask_json, "w"))
        return {"config": config, "image": out_image, "mask_json": mask_json,
                "n_boxes": 1, "class_counts": {}, "n_slivers": 0,
                "unmapped_ids": [], "dropped_actors": [],
                "transparent_cabinet_unlabeled": []}

    monkeypatch.setattr(lrm, "render_one", fake_render_one)
    res = lrm.patch_jobs([
        {"config": str(cfg_path), "image": str(tmp_path / "missing.png"),
         "json_dir": str(jdir)},
        {"config": str(cfg_path), "image": str(tmp_path / "b.png"),
         "json_dir": str(jdir)},
    ], patch_classes=("table",))

    assert "missing" in res[0]["error"]
    assert res[1]["n_added"] == 0 and res[1]["n_removed"] == 0
    assert json.loads((jdir / "b.json").read_text())["shapes"] == [keep]


def test_probe_jobs_records_failure_without_killing_batch(tmp_path, monkeypatch):
    import locus_render_masked as lrm
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(_cfg(4, {"apple_1": ["cabinet"]})))
    calls = []

    def fake_render_one(config, out_image, out_json_dir, **kw):
        calls.append(out_image)
        if "bad" in out_image:
            raise RuntimeError("sapien boom")
        return {"config": config, "image": out_image, "n_boxes": 0,
                "class_counts": {}, "n_slivers": 0, "unmapped_ids": [],
                "dropped_actors": [], "transparent_cabinet_unlabeled": [],
                "glass_blurred_review": [], "solid_hidden_ok": ["apple_1"],
                "hidden_target_probe": {}, "glass_verdict_source": "owner_probe"}

    monkeypatch.setattr(lrm, "render_one", fake_render_one)
    res = lrm.probe_jobs([
        {"config": str(cfg_path), "image": str(tmp_path / "bad.png"),
         "json_dir": str(tmp_path)},
        {"config": str(cfg_path), "image": str(tmp_path / "good.png"),
         "json_dir": str(tmp_path)},
    ])

    assert len(calls) == 2                      # failure did not abort the batch
    assert "RuntimeError: sapien boom" in res[0]["error"]
    assert res[0]["image"].endswith("bad.png") and res[0]["probe_only"]
    assert res[1]["solid_hidden_ok"] == ["apple_1"]


# ---------------------------------------------------------------------------
# A-1: absolute two-tier visible-pixel floor
# ---------------------------------------------------------------------------
def test_min_visible_pixels_for_is_two_tier_and_disablable():
    # every class the layout builds as a fixture takes the fixture floor
    for cls in ("cabinet", "drawer", "sink", "table"):
        assert cls in FIXTURE_CLASSES
        assert min_visible_pixels_for(cls) == MIN_VISIBLE_PIXELS_FIXTURE
    # movables, and the robot (large but articulated and self-dissimilar)
    for cls in ("fork", "bottle", "robot"):
        assert cls not in FIXTURE_CLASSES
        assert min_visible_pixels_for(cls) == MIN_VISIBLE_PIXELS_MOVABLE
    assert min_visible_pixels_for("cabinet", fixture=0) == 0


def test_absolute_floor_drops_the_dense_sliver_the_frac_guard_cannot():
    """A drawer showing half a pull ring: 100 px in a 10x10 box, frac 1.0. The
    fraction guard passes it by construction (dense), MIN_BOX_AREA=16 passes it
    (100 >= 16), and no human can name it -- this is the 13/24 spot-check
    cluster (stack_4_left_group_2_0 = 62 px, stack_3_right_group_1_0 = 24 px).
    A movable at the same pixel count is a normal fork and must survive."""
    drw = FakeArticulation("stack_4_left_group_2_0")
    ft = {"stack_4_left_group_2_0": "drawer"}
    seg, id_map = _seg_and_map([
        (3, link(drw, "object"), [(10, 20, 10, 20)]),        # 100 px, frac 1.0
        (4, FakeEntity("fork_0"), [(30, 40, 30, 40)]),       # 100 px, frac 1.0
    ])
    boxes, stats = masks_to_boxes(seg, id_map, fixture_types=ft)
    assert [b["label"] for b in boxes] == ["fork"]
    assert stats["occluded"] == [] and stats["slivers"] == []
    (dropped,) = stats["under_min_pixels"]
    assert dropped["label"] == "drawer" and dropped["visible_pixels"] == 100
    assert dropped["visible_frac"] == 1.0                    # NOT a frac case
    assert dropped["floor"] == MIN_VISIBLE_PIXELS_FIXTURE

    # ... and the movable tier still has teeth below its own floor
    seg2, id_map2 = _seg_and_map([(4, FakeEntity("fork_0"), [(30, 34, 30, 34)])])
    boxes2, stats2 = masks_to_boxes(seg2, id_map2)
    assert boxes2 == []                                      # 16 px < 24
    assert stats2["under_min_pixels"][0]["floor"] == MIN_VISIBLE_PIXELS_MOVABLE

    # both tiers off -> both boxes come back, i.e. the guard is the only cause
    boxes3, _ = masks_to_boxes(seg, id_map, fixture_types=ft,
                               min_visible_pixels_fixture=0,
                               min_visible_pixels_movable=0)
    assert sorted(b["label"] for b in boxes3) == ["drawer", "fork"]


def test_dropped_actor_pixels_are_counted_per_actor():
    """A-2: "is the unlabeled set visually significant?" (train_00184's knife
    block overlapping a labeled `scissors`) must be answerable from stats."""
    seg, id_map = _seg_and_map([
        (2, FakeEntity("wall_1"), [(0, 30, 0, 40)]),          # 1200 px, dropped
        (3, FakeEntity("scissors_0"), [(10, 20, 50, 60)]),
    ])
    boxes, stats = masks_to_boxes(seg, id_map, strict=False)
    assert [b["label"] for b in boxes] == ["scissors"]
    assert stats["dropped_pixels"] == {"wall_1": 1200}
    assert stats["background_pixels"] == 60 * 80 - 1200 - 100


# ---------------------------------------------------------------------------
# A-2: amodal projection + view_cos geometry (pure, no renderer)
# ---------------------------------------------------------------------------
class FakeCamera:
    """Pinhole with the same accessor surface sapien's camera exposes (both
    return a batched array in ManiSkill, hence the leading axis)."""

    def __init__(self, K, E):
        self._K, self._E = np.asarray(K, float), np.asarray(E, float)

    def get_intrinsic_matrix(self):
        return self._K[None]

    def get_extrinsic_matrix(self):
        return self._E[None]


class FakeWrapperCamera:
    """What ``scene.human_render_cameras[name]`` actually hands us: ManiSkill's
    ``sensors.camera.Camera``, which holds the sapien camera in ``.camera`` and
    re-exports the matrices only through ``get_params()``."""

    def __init__(self, K, E):
        self.camera = FakeCamera(K, E)

    def get_params(self):
        sapien = self.camera or self._camera
        return {"intrinsic_cv": sapien.get_intrinsic_matrix(),
                "extrinsic_cv": sapien.get_extrinsic_matrix(),
                "cam2world_gl": None}


class FakeParamsOnlyCamera(FakeWrapperCamera):
    """Same, minus the ``.camera`` handle -- only ``get_params()`` is left."""

    def __init__(self, K, E):
        super().__init__(K, E)
        self._camera, self.camera = self.camera, None


_K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
_E = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])   # world == cam


def test_hull_clip_and_area_are_exact_on_a_known_polygon():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)]
    hull = _hull(square)
    assert len(hull) == 4                       # interior point discarded
    assert _poly_area(hull) == pytest.approx(100.0)
    # half the square hangs out of a 5-wide frame
    assert _poly_area(_clip_to_rect(hull, 5, 10)) == pytest.approx(50.0)
    # a zero-width frame keeps only the degenerate x=0 edge -> no area
    assert _poly_area(_clip_to_rect(hull, 0, 10)) == pytest.approx(0.0)
    assert _poly_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0      # degenerate


def test_project_aabb_area_is_exact_and_refuses_geometry_behind_the_camera():
    cam = FakeCamera(_K, _E)
    # a 2x2 m plate at 2 m: u,v = 50 +- 100*1/2 -> a 100x100 px square
    aabb = np.array([[-1.0, -1.0, 2.0], [1.0, 1.0, 2.0]])
    assert _project_aabb_area(aabb, cam, (200, 200)) == pytest.approx(1e4)
    # clipped by a smaller frame: the visible part of the same hull
    assert _project_aabb_area(aabb, cam, (60, 80)) == pytest.approx(60 * 80)
    # twice as far -> half the linear size -> a quarter the area
    far = np.array([[-1.0, -1.0, 4.0], [1.0, 1.0, 4.0]])
    assert _project_aabb_area(far, cam, (200, 200)) == pytest.approx(2.5e3)
    # straddling the camera plane: no finite hull, so no number is reported
    straddle = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 2.0]])
    assert _project_aabb_area(straddle, cam, (200, 200)) is None


def test_project_aabb_area_accepts_the_maniskill_camera_wrapper():
    """The renderer hands the probe the WRAPPER, not the sapien camera.

    Calling ``get_intrinsic_matrix`` on it raises AttributeError and aborts the
    render -- which is exactly what happened on the first real-renderer run of
    this probe -- so all three accessor shapes are pinned here.
    """
    aabb = np.array([[-1.0, -1.0, 2.0], [1.0, 1.0, 2.0]])
    for cam in (FakeCamera(_K, _E), FakeWrapperCamera(_K, _E),
                FakeParamsOnlyCamera(_K, _E)):
        assert _project_aabb_area(aabb, cam, (200, 200)) == pytest.approx(1e4)
    with pytest.raises(AttributeError):
        _project_aabb_area(aabb, object(), (200, 200))


def test_actor_view_probe_reads_the_position_texture_in_millimetres():
    """The `default` shader emits Position as int16 MILLIMETRES in camera frame
    (mani_skill/render/shaders.py), so an unscaled read reported a 2m actor as
    `depth_m` 2000 -- which is what the first real render of this probe did."""
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[2:6, 2:6] = 7                                  # 16 px of one actor
    pos = np.zeros((8, 8, 4), dtype=np.int16)
    pos[..., 0] = 500                                  # 0.5 m right ...
    pos[..., 2] = -2000                                # ... 2 m forward (-z)
    out = actor_view_probe(seg, {7: FakeEntity("apple_1")},
                           {"position": pos, "camera": FakeCamera(_K, _E)})
    rec = out["apple_1"]
    assert rec["label"] == "apple" and rec["visible_pixels"] == 16
    assert rec["depth_m"] == pytest.approx((0.5 ** 2 + 2.0 ** 2) ** 0.5, abs=1e-3)


def _grid(ax0, ax1, fixed_axis, fixed, n=5):
    """n x n planar patch: `fixed_axis` pinned, the other two swept in [-1, 1]."""
    a, b = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    pts = np.zeros((n * n, 3))
    pts[:, ax0], pts[:, ax1], pts[:, fixed_axis] = a.ravel(), b.ravel(), fixed
    return pts


def test_plane_view_cos_separates_head_on_from_edge_on():
    # a drawer front square to the camera: normal || view ray -> |cos| = 1
    cos, flat = _plane_view_cos(_grid(0, 1, 2, 2.0))
    assert cos == pytest.approx(1.0) and flat == pytest.approx(0.0, abs=1e-9)
    # the same panel seen edge-on (its plane contains the view ray) -> |cos| = 0
    cos, flat = _plane_view_cos(_grid(1, 2, 0, 0.0) + np.array([0, 0, 2.0]))
    assert cos == pytest.approx(0.0, abs=1e-9) and flat == pytest.approx(0.0, abs=1e-9)
    # 45 deg: the fit is still planar, and |cos| lands between the two
    tilt = _grid(0, 1, 2, 0.0)
    tilt[:, 2] = tilt[:, 0]                                  # z = x plane
    cos, flat = _plane_view_cos(tilt + np.array([0, 0, 2.0]))
    assert 0.5 < cos < 0.9 and flat == pytest.approx(0.0, abs=1e-9)
    # a solid blob is not a plane: the normal is arbitrary, and `planarity` says
    # so -- which is why view_cos is only read where the residual is small.
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, 3)] * 3)
    blob = np.stack([x.ravel(), y.ravel(), z.ravel() + 4.0], axis=1)
    _cos, flat = _plane_view_cos(blob)
    assert flat == pytest.approx(1.0)
    # under 16 points there is nothing to fit and nothing is claimed
    assert _plane_view_cos(np.zeros((8, 3))) == (None, None)


def test_probe_record_reaches_the_written_json_and_the_dropped_buckets(tmp_path):
    """"occluded" and "edge-on" look identical in the label file, so the record
    has to survive into it -- including for the boxes that were dropped, which
    are exactly the ones the manifest is asked to explain."""
    drw = FakeArticulation("stack_4_left_group_2_0")
    ft = {"stack_4_left_group_2_0": "drawer"}
    seg, id_map = _seg_and_map([
        (3, link(drw, "object"), [(10, 20, 10, 20)]),        # under the floor
        (4, FakeEntity("bottle_0"), [(30, 45, 30, 45)]),     # kept
    ])
    probe = {
        "bottle_0": {"label": "bottle", "visible_pixels": 225,
                     "amodal_pixels": 900, "visible_over_amodal": 0.25,
                     "view_cos": 0.31, "planarity": 0.42, "depth_m": 1.75},
        "stack_4_left_group_2_0": {"label": "drawer", "visible_pixels": 100,
                                   "amodal_pixels": 5000,
                                   "visible_over_amodal": 0.02,
                                   "view_cos": 0.98, "planarity": 0.01,
                                   "depth_m": 3.2},
    }
    json_path, boxes, stats = export_annotations(
        seg, id_map, image_path="/x/y/00007.png", out_dir=str(tmp_path),
        fixture_types=ft, probe=probe)

    (kept,) = boxes
    assert kept["actor"] == "bottle_0" and kept["visible_over_amodal"] == 0.25
    (dropped,) = stats["under_min_pixels"]
    assert dropped["view_cos"] == 0.98 and dropped["amodal_pixels"] == 5000

    (shape,) = json.load(open(json_path))["shapes"]
    attrs = shape["attributes"]
    assert attrs["actor"] == "bottle_0" and attrs["visible_pixels"] == 225
    assert attrs["amodal_pixels"] == 900 and attrs["depth_m"] == 1.75
    assert attrs["view_cos"] == 0.31 and attrs["planarity"] == 0.42
    # a probe with holes must not invent keys
    thin = make_shape("bottle", [0, 0, 1, 1],
                      probe={"amodal_pixels": None, "view_cos": 0.5})
    assert "amodal_pixels" not in thin["attributes"]
    assert thin["attributes"]["view_cos"] == 0.5
