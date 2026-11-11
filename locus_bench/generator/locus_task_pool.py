"""
LOCUS-Bench task template generator (P7.1, docs/benchmark.md §9.1).

Produces official-schema-compatible template dicts (same fields consumed by
``generate_prompt.instantiate_task`` / ``eval_gt``) parameterized by the two
controlled difficulty axes:

  * axis S (perceptual state-judgment depth, facet A): 0..4
  * axis M (mechanical-cooperation structure depth, facet B): 0..4

Extra fields on top of the official schema:

  * ``bench_family``            – family id (F1..F10)
  * ``cell``                    – [s, m]
  * ``state_realization_spec``  – candidate set + the realization this
                                  template's ground_truth/init_pos encode +
                                  the full realization enumeration (consumed
                                  by the P7.2 instantiation wrapper for
                                  balanced sampling; NEVER enters the prompt)
  * ``opaque_container``        – True only for S4 (render hook, P7.4)
  * ``bridge_kind``             – low-depth atomic-support bridge, or None

Design notes (deviations recorded in docs/benchmark.md §9):
  * GT expansion per realization stays INSIDE this generator:
    ``make_template(family, s, m, rng, realization=...)`` returns a template
    whose ground_truth / init_pos are consistent with that realization.
  * F9 uses the Push-carrier relay motif only (official ``SimEnv.sim_step``
    handover branch has a param-order bug and no official template exercises
    Handover; do not emit Handover in GT until that is fixed upstream).
  * F5's destination varies with m (rack for M1/M2, cabinet for M3) so that
    M2 stays genuinely decoupled-parallel (a cabinet destination always
    couples robots through the shared open).
"""

import itertools
import random

__all__ = [
    "FAMILY_CELLS",
    "make_template",
    "enumerate_realizations",
    "iter_family_cells",
    "iter_cells",
    "build_locus_pool",
    "FIXTURE_MASK_LAYOUTS",
    "layouts_for_masks",
    "MASK_VOCAB_WEIGHTS",
    "weighted_mask_choice",
]

# ---------------------------------------------------------------------------
# vocab (subset of the 39 official assets present in every layout yaml)
# ---------------------------------------------------------------------------

# Non-uniform mask sampling weights (docs/YOLO.md §6.2 vocab skew).
#
# Masks are pinned by uniform `rng.choice` over each family's candidate list,
# so a class's box rate is set by how many lists it appears in, not by any
# quota. Three detector classes appear in exactly one train-grid list each
# (`wine`/`pizza` in `_F1_ITEMS_B`, `meat` in `_F1_ITEMS_A` + `_F9_ITEMS`),
# giving ~0.8 draws per train cell. At the ramp target (6.3k imgs) that is
# ~168 train boxes -> ~16.8 detector-val boxes at `--val-ratio 0.10`, under
# the §6.2 ">=20 detector-val boxes per new class" gate. VIKI-train supplies 0
# boxes for all three, so LOCUS alone has to clear it.
#
# w=3.0 lifts all three past the gate with >=1.3x margin (wine/pizza 28.0,
# meat 40.5 val boxes at 6.3k) while leaving every diluted table-mate safe
# (kettle 30.3, pumpkin 35.6, tomato 26.1). w=4.0 is past the useful point:
# wine/pizza gain little and kettle starts falling back toward the gate.
#
# `knife` is deliberately NOT lifted (0.5 draws/cell, 105 train boxes):
# VIKI-train supplies 279 knife boxes, so the merged full34 count is 384.
#
# Weights only reshape the sampling distribution -- every vocab value keeps
# non-zero probability, so the §7-3 vocabulary cross-coverage audit ("0 holes")
# is preserved and in fact reaches full coverage sooner for the rare items.
MASK_VOCAB_WEIGHTS = {
    "wine": 3.0,
    "pizza": 3.0,
    "meat": 3.0,
}


def weighted_mask_choice(rng, values, weights=None):
    """Pick one mask value, honouring `MASK_VOCAB_WEIGHTS`.

    RNG-consumption contract: when every candidate carries the same weight
    this delegates to `rng.choice`, so any cell whose vocab lists contain no
    weighted class draws byte-identically to the uniform generator and its
    already-rendered configs stay valid. Only lists that actually mix weights
    take the inverse-CDF path (one `rng.random()` draw).
    """
    weights = MASK_VOCAB_WEIGHTS if weights is None else weights
    w = [float(weights.get(v, 1.0)) for v in values]
    if any(wi < 0 for wi in w):
        raise ValueError(
            f"negative mask weight in {dict(zip(values, w))!r} -- a weight "
            f"table must stay a distribution over the vocab")
    if len(set(w)) <= 1:
        # includes the all-zero table: equal weights are uniform by definition
        return rng.choice(values)
    total = sum(w)
    target = rng.random() * total
    upto = 0.0
    for value, wi in zip(values, w):
        upto += wi
        if target < upto:
            return value
    return values[-1]        # float round-off guard


ALL_LAYOUTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TABLE_LAYOUTS = [1, 3, 6, 9]          # layouts whose position areas include "table"
TABLE_LAYOUTS_12X = [1, 3, 6]         # official 12-x precedent
# layout 6 is the only layout with BOTH pandaA_asset and pandaB_asset anchor
# areas — official 13-2 declares [3, 6] but its R3-anchored sink cannot be
# placed on layout 3 (no pandaA_asset), so the relay family renders on 6 only
RELAY_LAYOUTS = [6]

# Fixture masks that only SOME layouts instantiate. A sample naming one of
# these in a layout that has no such fixture is un-groundable: the instruction
# refers to something neither the renderer draws nor the detector can see.
#
# `oven` is the only one. Derived from the robocasa blueprints
# ($MS_ASSET_DIR/scene_datasets/robocasa_dataset/assets/scenes/kitchen_layouts):
# only the four "large" layouts declare `type: oven` + `type: stovetop`
# (one_wall_large=1, L_shaped_large=3, U_shaped_large=6, G_shaped_large=8); the
# other six declare a single combined `type: stove` instead, which is a stovetop
# unit with no oven cavity (and is in the exporter's DROPPED_TYPES). Layout 3
# then loses it again: `configs/layouts/layout_3.yaml` merges
# `customized_layout_3.yaml`, whose bottom-row group comments out `- name: oven`
# / `oven_housing`, and `merge_dicts` lets the custom group replace the base one.
# Confirmed on the P7.5a pilot with no exceptions: layouts 1/6/8 have an oven box
# in 215/215, 324/324, 77/77 images; layouts 2/3/4/5/7/9 in 0 of 544.
#
# Every other fixture mask (fridge/sink/toaster/cabinet/drawer/table/rack/the two
# area masks) is declared by all of layouts 1-9, so no entry is needed.
FIXTURE_MASK_LAYOUTS = {
    "oven": [1, 6, 8],
}

def layouts_for_masks(mask_values, layouts=None):
    """Narrow ``layouts`` to those instantiating every mask in ``mask_values``.

    Returns ``(layouts, restricting_masks)``. ``restricting_masks`` names the
    masks that actually narrowed the pool, so the caller can record WHY a
    sample's layout choice was constrained. Masks absent from
    :data:`FIXTURE_MASK_LAYOUTS` (all movable objects, all areas, and every
    fixture present in layouts 1-9) never restrict anything.

    An empty intersection is returned as-is: it means the mask combination is
    un-renderable and the caller must reject the sample, not silently widen.
    """
    keep = list(layouts if layouts is not None else ALL_LAYOUTS)
    restricting = []
    for value in mask_values:
        allowed = FIXTURE_MASK_LAYOUTS.get(str(value))
        if allowed is None:
            continue
        narrowed = [l for l in keep if l in set(allowed)]
        if narrowed != keep:
            restricting.append(str(value))
        keep = narrowed
    return keep, restricting


SURFACES = ["kitchen work area", "kitchen island area"]

# ---------------------------------------------------------------------------
# small builders
# ---------------------------------------------------------------------------

def _fetch(item, dest):
    """move-reach-grasp-move-place single fetch track."""
    return [["Move", item], ["Reach", item], ["Grasp", item],
            ["Move", dest], ["Place", dest]]


def _dual_fetch(i1, i2, dest):
    """humanoid (2 ee) picks two items then places both at dest."""
    return [["Move", i1], ["Reach", i1], ["Grasp", i1],
            ["Move", i2], ["Reach", i2], ["Grasp", i2],
            ["Move", dest], ["Place", dest]]


def _open_track(cab):
    return [["Move", cab], ["Reach", cab], ["Open", cab]]


def _zip_tracks(tracks):
    """Merge per-robot tracks into shared timesteps (official 3-1/5-1 style).

    tracks: {"R1": [[op, tgt], ...], "R2": [...]}
    """
    n = max((len(t) for t in tracks.values()), default=0)
    steps = []
    for i in range(n):
        step = {r: t[i] for r, t in tracks.items() if i < len(t)}
        if step:
            steps.append(step)
    return steps


def _seq(robot, *tracks):
    steps = []
    for t in tracks:
        steps.extend({robot: op} for op in t)
    return steps


def _goal_at(name, pos):
    return [{
        "type": "asset",
        "name": name,
        "is_satisfied": True,
        "status": {"pos.name": pos},
    }]


def _goal_activated(name):
    return [{
        "type": "asset",
        "name": name,
        "is_satisfied": True,
        "status": {"is_activated": True},
    }]


def _base(family, s, m, layouts, robot_roles, idle):
    return {
        "task_id": f"{family}_s{s}m{m}",
        "task_name": f"{family}_s{s}m{m}",
        "layout_idx": list(layouts),
        "robot_roles": list(robot_roles),
        "idle_robot_roles": list(idle),
        "bench_family": family,
        "cell": [s, m],
        "opaque_container": False,
        "bridge_kind": None,
    }


def _init_absent(mask_key, pos, exclude=None):
    entry = {"name_key": mask_key, "pos": list(pos)}
    if exclude:
        entry["exclude_keys"] = list(exclude)
    return entry


def _init_present(mask_key, dest_mask_key):
    # aligned_keys pins the item exactly at the (mask-resolved) destination
    return {"name_key": mask_key, "pos": ["<placeholder>"],
            "aligned_keys": [dest_mask_key]}


def _spec(semantics, candidates, realization, realizations):
    return {
        "semantics": semantics,
        "candidates": candidates,
        "realization": dict(realization),
        "realizations": [dict(r) for r in realizations],
    }


def _binary_realizations(mask_keys, states=("absent", "present")):
    """All state combos; all-'absent' (max structure) first."""
    combos = []
    for values in itertools.product(states, repeat=len(mask_keys)):
        combos.append(dict(zip(mask_keys, values)))
    return combos


# ---------------------------------------------------------------------------
# F1 relocate · S0-S1 × M0-M2
# ---------------------------------------------------------------------------

_F1_ITEMS_A = ["meat", "cup", "bread", "bottle", "banana",
               "apple", "tomato", "spoon", "pear", "peach", "fork"]
_F1_ITEMS_B = ["pizza", "wine", "kettle", "scissors", "pumpkin"]

_F1_DESC_SINGLE = [
    "Move the <mask1> to the <mask3> so it is ready for the next step.",
    "Please place the <mask1> onto the <mask3> to continue the preparation.",
    "Transfer the <mask1> over to the <mask3> and make sure it ends up there.",
    "Put the <mask1> on the <mask3> so it can be used later.",
    "Take the <mask1> and move it to the <mask3> in an orderly manner.",
    "Relocate the <mask1> to the <mask3> as instructed.",
]

_F1_DESC_PAIR = [
    "Move the <mask1> and the <mask2> to the <mask3> so they are ready for the next step.",
    "Please get both the <mask1> and the <mask2> onto the <mask3>.",
    "Transfer the <mask1> together with the <mask2> over to the <mask3>.",
    "Put the <mask1> and the <mask2> on the <mask3> so they can be used later.",
    "Make sure the <mask1> and the <mask2> both end up on the <mask3>.",
    "Relocate the <mask1> along with the <mask2> to the <mask3> as instructed.",
]


def _build_f1(s, m, rng, realization):
    roles = {0: ["humanoid"],
             1: ["wheeled"],
             2: ["humanoid", "wheeled"]}[m]
    tpl = _base("F1", s, m, ALL_LAYOUTS, roles, ["dog"])
    tpl["mask1"] = list(_F1_ITEMS_A)
    tpl["mask3"] = ["kitchen work area", "kitchen island area", "rack"]
    two_items = m >= 1
    if two_items:
        tpl["mask2"] = list(_F1_ITEMS_B)
    tpl["description"] = list(_F1_DESC_PAIR if two_items else _F1_DESC_SINGLE)

    # candidate is always mask1, regardless of item count (axis hygiene)
    cand_key = "mask1"
    cand_keys = [cand_key] if s >= 1 else []
    realizations = _binary_realizations(cand_keys) if cand_keys else [{}]
    if realization is None:
        realization = realizations[0]
    present = realization.get(cand_key) == "present"

    init, gt = [], []
    goal = [_goal_at("<mask1>", "<mask3>")]
    if two_items:
        goal.append(_goal_at("<mask2>", "<mask3>"))

    def _item_init(key, is_present):
        if is_present:
            return _init_present(key, "mask3")
        return _init_absent(key, SURFACES, exclude=["mask3"])

    a_present = present
    b_present = False
    init.append(_item_init("mask1", a_present))
    if two_items:
        init.append(_item_init("mask2", b_present))

    if m == 0:
        if not a_present:
            gt = _seq("R1", _fetch("<mask1>", "<mask3>"))
    elif m == 1:
        tracks = [] if a_present else [_fetch("<mask1>", "<mask3>")]
        if not b_present:
            tracks.append(_fetch("<mask2>", "<mask3>"))
        gt = _seq("R1", *tracks)
    else:  # m == 2, parallel disjoint
        tracks = {}
        if not a_present:
            tracks["R1"] = _fetch("<mask1>", "<mask3>")
        if not b_present:
            tracks["R2"] = _fetch("<mask2>", "<mask3>")
        gt = _zip_tracks(tracks)

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = goal
    tpl["state_realization_spec"] = _spec(
        "presatisfied" if s >= 1 else "none",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask3"}
         for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F2 conditional_bring · S1-S2 × M0-M2
# ---------------------------------------------------------------------------

_F2_C1 = ["cup", "fork", "bowl"]
_F2_C2 = ["plate", "spoon", "bottle"]
_F2_DISTRACTOR = ["scissors", "banana", "kettle"]

_F2_DESC_S1 = [
    "The <mask1> belongs on the <mask3>; if it is not there yet, bring it over.",
    "Check whether the <mask1> is already on the <mask3>. If not, go fetch it.",
    "Have a look at the <mask3>: should the <mask1> be missing, carry it there.",
    "Make sure a <mask1> is on the <mask3>; deliver one only if it is absent.",
    "If you do not see the <mask1> on the <mask3>, place it there.",
    "Confirm the <mask3> holds the <mask1>; retrieve it if necessary.",
]

_F2_DESC_S2 = [
    "Take a look at the <mask3>. If you don't see both a <mask1> and a <mask2> there, bring over whichever is missing.",
    "Survey the <mask3> first; whichever of the <mask1> or the <mask2> is not present, carry it there.",
    "Glance at the <mask3> and confirm it holds one <mask1> and one <mask2>. Fetch anything that is still absent.",
    "Check the <mask3> with your sensors. Should it lack either the <mask1> or the <mask2>, deliver the missing one.",
    "Scan the <mask3>. Any of the two items—the <mask1> or the <mask2>—that isn't present needs to be placed there.",
    "Verify with a quick look that the <mask3> hosts both the <mask1> and the <mask2>. Add whatever is missing.",
]


def _build_f2(s, m, rng, realization):
    roles = {0: ["humanoid"], 1: ["wheeled"], 2: ["humanoid", "wheeled"]}[m]
    tpl = _base("F2", s, m, TABLE_LAYOUTS, roles, ["dog", "arm"])
    tpl["mask1"] = list(_F2_C1)
    tpl["mask3"] = ["table"]
    cand_keys = ["mask1"]
    if s >= 2:
        tpl["mask2"] = list(_F2_C2)
        tpl["mask4"] = list(_F2_DISTRACTOR)   # scene distractor, never mentioned
        cand_keys = ["mask1", "mask2"]
    tpl["description"] = list(_F2_DESC_S2 if s >= 2 else _F2_DESC_S1)

    realizations = _binary_realizations(cand_keys)
    if realization is None:
        realization = realizations[0]

    init, goal = [], []
    actives = []
    for key in cand_keys:
        goal.append(_goal_at(f"<{key}>", "<mask3>"))
        if realization[key] == "present":
            init.append(_init_present(key, "mask3"))
        else:
            init.append(_init_absent(key, ["kitchen work area"], exclude=["mask3"]))
            actives.append(key)
    if s >= 2:
        init.append(_init_absent("mask4", ["kitchen work area"]))

    if m == 0:  # humanoid, dual-carry when both missing
        if len(actives) == 2:
            gt = _seq("R1", _dual_fetch(f"<{actives[0]}>", f"<{actives[1]}>", "<mask3>"))
        elif len(actives) == 1:
            gt = _seq("R1", _fetch(f"<{actives[0]}>", "<mask3>"))
        else:
            gt = []
    elif m == 1:  # wheeled, serial trips
        gt = _seq("R1", *[_fetch(f"<{k}>", "<mask3>") for k in actives])
    else:  # m == 2, one robot per missing candidate
        tracks = {}
        for robot, key in zip(("R1", "R2"), actives):
            tracks[robot] = _fetch(f"<{key}>", "<mask3>")
        gt = _zip_tracks(tracks)

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = goal
    tpl["state_realization_spec"] = _spec(
        "conditional_missing",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask3"}
         for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F3 restock_quantified · S2 bridge / S3 × M0-M2
# ---------------------------------------------------------------------------

_F3_F1 = ["apple", "pear"]
_F3_F2 = ["peach", "banana"]
_F3_F3 = ["tomato"]
_F3_DISTRACTOR = ["fork", "cup", "scissors", "bottle"]

_F3_DESC = [
    "Make sure every fruit is on the <mask5>; fetch any fruit that isn't there yet.",
    "Check the <mask5>: all the fruits should be on it—go get whichever fruit is missing.",
    "Inspect the <mask5> and confirm all fruits are present. Retrieve any fruit that is absent.",
    "Every fruit in this kitchen belongs on the <mask5>. Bring over any that are elsewhere.",
    "Scan the <mask5> for the fruits. Any fruit not already on it must be fetched and placed there.",
    "Ensure the <mask5> holds all the fruits; collect and deliver the ones that are missing.",
]


def _build_f3(s, m, rng, realization):
    roles = {0: ["humanoid"], 1: ["wheeled"], 2: ["humanoid", "wheeled"]}[m]
    tpl = _base("F3", s, m, TABLE_LAYOUTS_12X, roles, ["dog", "arm"])
    tpl["mask1"] = list(_F3_F1)
    tpl["mask2"] = list(_F3_F2)
    if s >= 3:
        tpl["mask3"] = list(_F3_F3)
    tpl["mask4"] = list(_F3_DISTRACTOR)   # same-surface non-fruit distractor
    tpl["mask5"] = ["table"]
    tpl["description"] = list(_F3_DESC)
    if s == 2:
        tpl["bridge_kind"] = "tag_quantifier_k2"

    # S2 is the train bridge: the same all-<tag> operator at bounded k=2.
    # S3 alone expands the quantified set to three independently realized
    # candidates (and may receive same-tag extras in the wrapper).
    cand_keys = ["mask1", "mask2"] + (["mask3"] if s >= 3 else [])
    realizations = _binary_realizations(cand_keys)
    if realization is None:
        realization = realizations[0]

    init, goal, actives = [], [], []
    for key in cand_keys:
        goal.append(_goal_at(f"<{key}>", "<mask5>"))
        if realization[key] == "present":
            init.append(_init_present(key, "mask5"))
        else:
            init.append(_init_absent(key, ["kitchen work area"], exclude=["mask5"]))
            actives.append(f"<{key}>")
    init.append(_init_absent("mask4", ["kitchen work area"]))

    if m == 0:  # humanoid: dual-carry pairs, then single trip for a leftover
        tracks = []
        i = 0
        while i + 1 < len(actives):
            tracks.append(_dual_fetch(actives[i], actives[i + 1], "<mask5>"))
            i += 2
        if i < len(actives):
            tracks.append(_fetch(actives[i], "<mask5>"))
        gt = _seq("R1", *tracks)
    elif m == 1:  # wheeled serial
        gt = _seq("R1", *[_fetch(a, "<mask5>") for a in actives])
    else:  # m == 2: split actives across both robots (humanoid may dual-carry)
        tracks = {}
        if len(actives) == 3:
            tracks["R1"] = _dual_fetch(actives[0], actives[1], "<mask5>")
            tracks["R2"] = _fetch(actives[2], "<mask5>")
        elif len(actives) == 2:
            tracks["R1"] = _fetch(actives[0], "<mask5>")
            tracks["R2"] = _fetch(actives[1], "<mask5>")
        elif len(actives) == 1:
            tracks["R1"] = _fetch(actives[0], "<mask5>")
        gt = _zip_tracks(tracks)

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = goal
    tpl["state_realization_spec"] = _spec(
        "quantified_tag",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask5",
          "tag": "fruit"} for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F4 fallback_retrieve · S2 bridge / S4 × M1-M3
# ---------------------------------------------------------------------------

_F4_ITEMS = ["bread", "apple", "banana"]
_F4_DISHES = ["plate", "bowl"]
_F4_STOW = ["pumpkin", "kettle", "wine"]

_F4_DESC = [
    "Serve the <mask1> on the <mask3>; if it's not out on the counter, it should be in the <mask2>.",
    "Put the <mask1> onto the <mask3>. Should it be missing from the counter, check the <mask2> and retrieve it.",
    "Get the <mask1> onto the <mask3>; remember, it may have been stored away in the <mask2>.",
    "Move the <mask1> to the <mask3>. In case it isn't out in the open, open the <mask2> and fetch it.",
    "Deliver the <mask1> to the <mask3>. If it's nowhere on the counters, it is inside the <mask2>.",
]

# M2 adds a second goal constraint the M1 wording cannot express: R2 relocates
# the dish itself to the island before R1 serves onto it. The M1 templates name
# only `<mask1> -> <mask3>`, so "<mask3> -> kitchen island area" was a goal the
# instruction never mentioned and no model could recover (docs/benchmark.md
# §7.6). Paraphrase ids stay aligned with `_F4_DESC` one-for-one -- a frozen
# sample's `meta.paraphrase_id` indexes this list, so reordering or resizing it
# would silently repoint already-rendered rows.
_F4_DESC_M2 = [
    "Set the <mask3> down on the kitchen island area first, then serve the <mask1> on it; if the <mask1> is not out on the counter, it should be in the <mask2>.",
    "Put the <mask3> onto the kitchen island area, then put the <mask1> onto the <mask3>. Should the <mask1> be missing from the counter, check the <mask2> and retrieve it.",
    "Get the <mask3> onto the kitchen island area, then get the <mask1> onto the <mask3>; remember, the <mask1> may have been stored away in the <mask2>.",
    "Move the <mask3> to the kitchen island area before moving the <mask1> onto it. In case the <mask1> isn't out in the open, open the <mask2> and fetch it.",
    "Bring the <mask3> to the kitchen island area, then deliver the <mask1> onto it. If the <mask1> is nowhere on the counters, it is inside the <mask2>.",
]

_F4_DESC_M3 = [
    "Serve the <mask1> on the <mask3>—if it's not out on the counter, it's in the <mask2>—and put the <mask4> away inside the <mask2> as well.",
    "Get the <mask1> onto the <mask3> (check the <mask2> if it isn't out), and stow the <mask4> in the <mask2> while you're at it.",
    "Move the <mask1> to the <mask3>; it may be hidden in the <mask2>. Also store the <mask4> inside the <mask2>.",
    "Deliver the <mask1> to the <mask3>, fetching it from the <mask2> if it isn't visible, and put the <mask4> into the <mask2>.",
    "The <mask1> goes on the <mask3>—look in the <mask2> if needed—and the <mask4> goes into the <mask2> for storage.",
]


def _build_f4(s, m, rng, realization):
    roles = {1: ["humanoid"], 2: ["humanoid", "wheeled"], 3: ["humanoid", "wheeled"]}[m]
    tpl = _base("F4", s, m, ALL_LAYOUTS, roles, ["dog"])
    tpl["opaque_container"] = (s == 4)
    if s == 2:
        tpl["bridge_kind"] = "transparent_fallback"
    tpl["mask1"] = list(_F4_ITEMS)
    tpl["mask2"] = ["cabinet"]
    tpl["mask3"] = list(_F4_DISHES)
    if m == 3:
        tpl["mask4"] = list(_F4_STOW)
    tpl["description"] = list(
        _F4_DESC_M3 if m == 3 else _F4_DESC_M2 if m == 2 else _F4_DESC)

    realizations = [{"mask1": "cabinet"}, {"mask1": "counter"}]
    if realization is None:
        realization = realizations[0]
    hidden = realization["mask1"] == "cabinet"

    init = [{"name_key": "mask2", "pos": ["room_cabinet"]}]
    if hidden:
        init.append({"name_key": "mask1", "pos": ["cabinet"], "aligned_keys": ["mask2"]})
    else:
        init.append(_init_absent("mask1", SURFACES, exclude=["mask3"]))
    init.append(_init_absent("mask3", ["kitchen island area"]))
    if m == 3:
        init.append(_init_absent("mask4", SURFACES, exclude=["mask2"]))

    serve_prefix = _open_track("<mask2>") if hidden else []
    serve = serve_prefix + _fetch("<mask1>", "<mask3>")
    if m == 2 and not hidden:
        # detour via the island so R1's Place lands strictly after R2 has set
        # the dish down (temporal constraint), mirroring official 3-1 pacing
        serve = serve[:3] + [["Move", "kitchen island area"]] + serve[3:]

    goal = [_goal_at("<mask1>", "<mask3>")]
    temporal = []
    if m == 1:
        gt = _seq("R1", serve)
    elif m == 2:
        # R2 relocates the dish to the island surface first (3-1 motif);
        # dish starts on the work area so relocating it is a real op
        dish_track = _fetch("<mask3>", "kitchen island area")
        for entry in init:
            if entry.get("name_key") == "mask3":
                entry["pos"] = ["kitchen work area"]
                entry.pop("exclude_keys", None)
        gt = _zip_tracks({"R1": serve, "R2": dish_track})
        goal.append(_goal_at("<mask3>", "kitchen island area"))
        temporal = [[
            _goal_at("<mask3>", "kitchen island area"),
            _goal_at("<mask1>", "<mask3>"),
        ]]
    else:  # m == 3, shared open: R2 stows mask4 into the cabinet R1 opened
        goal.append(_goal_at("<mask4>", "<mask2>"))
        if hidden:
            stow = _fetch("<mask4>", "<mask2>")     # place lands after R1's open
            gt = _zip_tracks({"R1": serve, "R2": stow})
        else:
            # item visible: R2 must open the cabinet itself before stowing
            stow = _open_track("<mask2>") + _fetch("<mask4>", "<mask2>")
            gt = _zip_tracks({"R1": serve, "R2": stow})

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = goal
    if temporal:
        tpl["temporal_constraints"] = temporal
    tpl["state_realization_spec"] = _spec(
        "hidden_fallback",
        [{"mask_key": "mask1", "states": ["cabinet", "counter"],
          "dest_mask": "mask3", "hidden_state": "cabinet"}],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F5 clear_surface · S2 bridge / S3 × M1-M3
# ---------------------------------------------------------------------------

_F5_I1 = ["pumpkin", "bread", "apple"]
_F5_I2 = ["scissors", "spoon", "fork"]
_F5_I3 = ["tomato", "peach"]

_F5_DESC = [
    "Everything left out on the <mask5> should be put away on the <mask4>; anything already stored can stay.",
    "Tidy up: whatever is still sitting on the <mask5> goes to the <mask4>.",
    "Clear the <mask5>—move every item that is on it over to the <mask4>.",
    "The <mask5> needs to be cleared; relocate all items on it to the <mask4>.",
    "Put away everything that remains on the <mask5> into place on the <mask4>.",
    "Any item still out on the <mask5> must be moved to the <mask4> for storage.",
]


def _build_f5(s, m, rng, realization):
    roles = {1: ["wheeled"], 2: ["humanoid", "wheeled"], 3: ["humanoid", "wheeled"]}[m]
    tpl = _base("F5", s, m, ALL_LAYOUTS, roles, ["dog"])
    dest = "cabinet" if m == 3 else "rack"
    tpl["mask1"] = list(_F5_I1)
    tpl["mask2"] = list(_F5_I2)
    if s >= 3:
        tpl["mask3"] = list(_F5_I3)
    tpl["mask4"] = [dest]
    tpl["mask5"] = ["kitchen work area"]
    tpl["description"] = list(_F5_DESC)
    if s == 2:
        tpl["bridge_kind"] = "bounded_clear_set_k2"

    cand_keys = ["mask1", "mask2"] + (["mask3"] if s >= 3 else [])
    realizations = _binary_realizations(cand_keys)
    if realization is None:
        realization = realizations[0]

    init, goal, actives = [], [], []
    if m == 3:
        init.append({"name_key": "mask4", "pos": ["room_cabinet"]})
    for key in cand_keys:
        goal.append(_goal_at(f"<{key}>", "<mask4>"))
        if realization[key] == "present":
            init.append(_init_present(key, "mask4"))
        else:
            init.append(_init_absent(key, ["kitchen work area"]))
            actives.append(f"<{key}>")

    if m == 1:  # wheeled serial to the rack
        gt = _seq("R1", *[_fetch(a, "<mask4>") for a in actives])
    elif m == 2:  # decoupled parallel to the rack
        tracks = {}
        head, tail = actives[:2], actives[2:]
        if len(head) == 2:
            tracks["R1"] = _dual_fetch(head[0], head[1], "<mask4>")
        elif len(head) == 1:
            tracks["R1"] = _fetch(head[0], "<mask4>")
        if tail:
            tracks["R2"] = []
            for a in tail:
                tracks["R2"].extend(_fetch(a, "<mask4>"))
        gt = _zip_tracks(tracks)
    else:  # m == 3: humanoid opens the cabinet once, both deposit into it
        tracks = {}
        if actives:
            r1 = _open_track("<mask4>") + _fetch(actives[0], "<mask4>")
            tracks["R1"] = r1
            if actives[1:]:
                tracks["R2"] = []
                for a in actives[1:]:
                    tracks["R2"].extend(_fetch(a, "<mask4>"))
        gt = _zip_tracks(tracks)

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = goal
    tpl["state_realization_spec"] = _spec(
        "surface_clear",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask4"}
         for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F6 set_table · S1-S2 × M2-M3
# ---------------------------------------------------------------------------

_F6_DESC = [
    "Set the table: the <mask1>, the <mask2> and the <mask3> should all be on the <mask4>; add whatever is missing.",
    "Prepare the <mask4> for a meal—make sure a <mask1>, a <mask2> and a <mask3> are all set out on it.",
    "Check the <mask4>: it needs the <mask1>, the <mask2> and the <mask3>. Bring over any piece that isn't there.",
    "Finish setting the <mask4> so that the <mask1>, <mask2> and <mask3> are all in place.",
    "The <mask4> should hold the <mask1>, the <mask2> and the <mask3>; fetch the ones still missing.",
    "Look over the <mask4> and supply whichever of the <mask1>, <mask2> or <mask3> is absent.",
]


def _build_f6(s, m, rng, realization):
    tpl = _base("F6", s, m, TABLE_LAYOUTS, ["humanoid", "wheeled"], ["dog", "arm"])
    tpl["mask1"] = ["plate"]
    tpl["mask2"] = ["cup"]
    tpl["mask3"] = ["fork", "spoon"]
    tpl["mask4"] = ["table"]
    if m == 3:
        tpl["mask5"] = ["cabinet"]
    tpl["description"] = list(_F6_DESC)

    all_keys = ["mask1", "mask2", "mask3"]
    # S1: only mask1's presence is sampled; S2: all three independent
    cand_keys = ["mask1"] if s == 1 else all_keys
    realizations = _binary_realizations(cand_keys)
    if realization is None:
        realization = realizations[0]

    def _state(key):
        return realization.get(key, "absent")

    init, goal = [], []
    if m == 3:
        init.append({"name_key": "mask5", "pos": ["room_cabinet"]})
    for key in all_keys:
        goal.append(_goal_at(f"<{key}>", "<mask4>"))
        if _state(key) == "present":
            init.append(_init_present(key, "mask4"))
        elif m == 3 and key == "mask1":
            # M3 structure knob rides on mask1 (marginal absence prob = 1/2
            # at BOTH s levels, keeping open-rate s-invariant): the absent
            # plate sits inside the cabinet
            init.append({"name_key": key, "pos": ["cabinet"], "aligned_keys": ["mask5"]})
        else:
            init.append(_init_absent(key, ["kitchen work area"], exclude=["mask4"]))

    plate_active = _state("mask1") == "absent"
    cup_active = _state("mask2") == "absent"
    fork_active = _state("mask3") == "absent"

    if m == 2:
        # humanoid dual-carries cup+fork actives, wheeled brings the plate
        tracks = {}
        pair = [k for k, a in (("mask2", cup_active), ("mask3", fork_active)) if a]
        if len(pair) == 2:
            tracks["R1"] = _dual_fetch(f"<{pair[0]}>", f"<{pair[1]}>", "<mask4>")
        elif len(pair) == 1:
            tracks["R1"] = _fetch(f"<{pair[0]}>", "<mask4>")
        if plate_active:
            tracks["R2"] = _fetch("<mask1>", "<mask4>")
        gt = _zip_tracks(tracks)
    else:  # m == 3: humanoid opens cabinet for the plate, wheeled serial rest
        tracks = {}
        if plate_active:
            tracks["R1"] = _open_track("<mask5>") + _fetch("<mask1>", "<mask4>")
        r2 = []
        for key, active in (("mask2", cup_active), ("mask3", fork_active)):
            if active:
                r2.extend(_fetch(f"<{key}>", "<mask4>"))
        if r2:
            tracks["R2"] = r2
        gt = _zip_tracks(tracks)

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = goal
    tpl["state_realization_spec"] = _spec(
        "presatisfied",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask4"}
         for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F7 device_prep · S0-S1 × M1-M2
# ---------------------------------------------------------------------------

_F7_DESC_TOAST = [
    "Insert the <mask1> into the <mask2>, activate it, and place the <mask3> on the <mask4> to prepare for serving.",
    "Put the <mask1> into the <mask2>, turn it on, and set the <mask3> onto the <mask4> for the breakfast setup.",
    "Load the <mask1> into the <mask2>, start its operation, and arrange the <mask3> neatly on the <mask4>.",
    "Begin by inserting the <mask1> into the <mask2> and switching it on; the <mask3> goes on the <mask4>.",
    "Get the <mask1> toasting in the <mask2> and make sure the <mask3> is out on the <mask4>.",
]

_F7_DESC_WASH = [
    "Rinse the <mask1> in the <mask2> and set it into the <mask3> for serving.",
    "Give the <mask1> a quick wash at the <mask2> and place it inside the <mask3>.",
    "Clean the <mask1> at the <mask2>, then deliver it into the <mask3>.",
    "Make sure the <mask1> is washed at the <mask2> before resting it in the <mask3>.",
    "Run water over the <mask1> at the <mask2>, then arrange it in the <mask3>.",
]


def _build_f7(s, m, rng, realization):
    # deterministic per cell: wash covers S0×M1, toast everything else
    device = "wash" if (s == 0 and m == 1) else "toast"
    roles = {1: ["humanoid"], 2: ["humanoid", "wheeled"]}[m]
    tpl = _base("F7", s, m, ALL_LAYOUTS, roles, ["dog"])

    cand_keys = ["mask3"] if (s >= 1 and device == "toast") else []
    realizations = _binary_realizations(cand_keys) if cand_keys else [{}]
    if realization is None:
        realization = realizations[0]
    dish_present = realization.get("mask3") == "present"

    if device == "toast":
        tpl["mask1"] = ["bread"]
        tpl["mask2"] = ["toaster"]
        tpl["mask3"] = ["plate", "bowl"]
        tpl["mask4"] = ["kitchen work area", "kitchen island area"]
        tpl["description"] = list(_F7_DESC_TOAST)
        init = [
            _init_absent("mask1", SURFACES),
            {"name_key": "mask2", "pos": ["room_toaster"]},
        ]
        if dish_present:
            init.append(_init_present("mask3", "mask4"))
        else:
            init.append(_init_absent("mask3", SURFACES, exclude=["mask4"]))

        device_track = [["Move", "<mask1>"], ["Reach", "<mask1>"], ["Grasp", "<mask1>"],
                        ["Move", "<mask2>"], ["Place", "<mask2>"], ["Interact", "<mask2>"]]
        dish_track = [] if dish_present else _fetch("<mask3>", "<mask4>")
        if m == 1:
            gt = _seq("R1", device_track, dish_track)
        else:
            tracks = {"R1": device_track}
            if dish_track:
                tracks["R2"] = dish_track
            gt = _zip_tracks(tracks)
        tpl["temporal_constraints"] = [[
            _goal_at("<mask1>", "<mask2>"),
            _goal_activated("<mask2>"),
        ]]
        tpl["goal_constraints"] = [
            _goal_at("<mask3>", "<mask4>"),
            _goal_activated("<mask2>"),
        ]
    else:  # wash (S0 M1 only)
        tpl["mask1"] = ["apple", "pear", "tomato", "peach", "banana"]
        tpl["mask2"] = ["sink"]
        tpl["mask3"] = ["bowl", "plate"]
        tpl["description"] = list(_F7_DESC_WASH)
        init = [
            _init_absent("mask1", ["kitchen work area"], exclude=["mask3"]),
            {"name_key": "mask2", "pos": ["room"]},
            _init_absent("mask3", ["kitchen island area"]),
        ]
        gt = _seq("R1", [
            ["Move", "<mask1>"], ["Reach", "<mask1>"], ["Grasp", "<mask1>"],
            ["Move", "<mask2>"], ["Place", "<mask2>"], ["Interact", "<mask2>"],
            ["Move", "<mask1>"], ["Reach", "<mask1>"], ["Grasp", "<mask1>"],
            ["Move", "<mask3>"], ["Place", "<mask3>"],
        ])
        tpl["temporal_constraints"] = [[
            _goal_at("<mask1>", "<mask2>"),
            _goal_activated("<mask2>"),
        ]]
        tpl["goal_constraints"] = [_goal_at("<mask1>", "<mask3>")]

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["state_realization_spec"] = _spec(
        "presatisfied" if cand_keys else "none",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask4"}
         for k in cand_keys],
        realization, realizations)
    tpl["device_variant"] = device
    return tpl


# ---------------------------------------------------------------------------
# F8 shared_open · S0-S1 × M2 bridge / M3
# ---------------------------------------------------------------------------

_F8_I1 = ["pumpkin", "bread", "apple", "peach", "tomato"]
_F8_I2 = ["scissors", "spoon", "fork", "knife"]

_F8_DESC = [
    "Put the <mask1> and the <mask2> into the <mask3> to clean up the workspace.",
    "Carefully place both the <mask1> and the <mask2> into the <mask3> to tidy up the area.",
    "Transfer the <mask1> along with the <mask2> into the <mask3> for proper storage.",
    "Collect the <mask1> and <mask2>, and place them neatly into the <mask3>.",
    "Store the <mask1> together with the <mask2> inside the <mask3> to make the area neat.",
    "Ensure that both the <mask1> and <mask2> end up inside the <mask3>.",
]


def _build_f8(s, m, rng, realization):
    tpl = _base("F8", s, m, ALL_LAYOUTS, ["humanoid", "wheeled"], ["dog"])
    tpl["mask1"] = list(_F8_I1)
    tpl["mask2"] = list(_F8_I2)
    tpl["mask3"] = ["rack"] if m == 2 else ["cabinet"]
    tpl["description"] = list(_F8_DESC)
    if m == 2:
        tpl["bridge_kind"] = "independent_deposit_twin"

    cand_keys = ["mask1"] if s >= 1 else []
    realizations = _binary_realizations(cand_keys) if cand_keys else [{}]
    if realization is None:
        realization = realizations[0]
    i1_present = realization.get("mask1") == "present"

    init = []
    if m == 3:
        init.append({"name_key": "mask3", "pos": ["room_cabinet"]})
    if i1_present:
        if m == 3:
            # Already stored, visible through the official transparent
            # cabinet style because this is not S4.
            init.append({"name_key": "mask1", "pos": ["cabinet"],
                         "aligned_keys": ["mask3"]})
        else:
            init.append(_init_present("mask1", "mask3"))
    else:
        init.append(_init_absent("mask1", SURFACES))
    init.append(_init_absent("mask2", SURFACES))

    if m == 2:
        # Low-depth twin: same two-deposit family and robot set, but both
        # tracks are independent and there is no shared cabinet/open.
        tracks = {"R2": _fetch("<mask2>", "<mask3>")}
        if not i1_present:
            tracks["R1"] = _fetch("<mask1>", "<mask3>")
        gt = _zip_tracks(tracks)
    elif i1_present:
        # opener humanoid handles the single remaining item (5-1 R1 track shape)
        gt = _seq("R1", [
            ["Move", "<mask2>"], ["Reach", "<mask2>"], ["Grasp", "<mask2>"],
            ["Move", "<mask3>"], ["Reach", "<mask3>"], ["Open", "<mask3>"],
            ["Place", "<mask3>"],
        ])
    else:
        # verbatim 5-1 shape: shared single open, simultaneous final places
        gt = [
            {"R1": ["Move", "<mask1>"], "R2": ["Move", "<mask2>"]},
            {"R1": ["Reach", "<mask1>"], "R2": ["Reach", "<mask2>"]},
            {"R1": ["Grasp", "<mask1>"], "R2": ["Grasp", "<mask2>"]},
            {"R1": ["Move", "<mask3>"], "R2": ["Move", "<mask3>"]},
            {"R1": ["Reach", "<mask3>"]},
            {"R1": ["Open", "<mask3>"]},
            {"R1": ["Place", "<mask3>"], "R2": ["Place", "<mask3>"]},
        ]

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    tpl["goal_constraints"] = [
        _goal_at("<mask1>", "<mask3>"),
        _goal_at("<mask2>", "<mask3>"),
    ]
    tpl["state_realization_spec"] = _spec(
        "presatisfied" if cand_keys else "none",
        [{"mask_key": k, "states": ["absent", "present"], "dest_mask": "mask3"}
         for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F9 capability_relay · S0-S1 × M2 bridge / M3 subchain / M4 relay
# ---------------------------------------------------------------------------

_F9_ITEMS = ["apple", "pumpkin", "meat", "pear", "peach", "fork", "spoon",
             "banana", "tomato"]

_F9_DESC = [
    "Move the <mask1> to the <mask3>. If the destination is too far, use the <mask2> to house the <mask1>, transport the <mask2> and fetch the <mask1>.",
    "Carry the <mask1> over to the <mask3>. If it's a long way, place the <mask1> into the <mask2> first, then move the <mask2> and retrieve the <mask1>.",
    "Bring the <mask1> to the <mask3>. When it is too far to carry directly, pack the <mask1> into the <mask2>, transport the <mask2>, and take out the <mask1> at the destination.",
    "Deliver the <mask1> to the <mask3>. If distance makes direct handling difficult, load the <mask1> into the <mask2>, transport the <mask2>, and then unload the <mask1>.",
    "Get the <mask1> over to the <mask3>. In case it's too distant, put the <mask1> into the <mask2>, shift the <mask2> across, and then retrieve the <mask1>.",
]

_F9_DESC_M2 = {
    "load": [
        "Load the <mask1> into the <mask2> so it is ready for transport.",
        "Place the <mask1> inside the <mask2> to prepare the carrier.",
        "Pack the <mask1> in the <mask2> for the next transport stage.",
        "Use the waiting arm to put the <mask1> into the <mask2>.",
        "Prepare the relay by loading the <mask1> inside the <mask2>.",
    ],
    "push": [
        "Push the loaded <mask2> carrying the <mask1> to the <mask3>.",
        "Move the <mask2>, with the <mask1> inside, over to the <mask3>.",
        "Transport the packed <mask2> to the <mask3> using the pushing robot.",
        "Have the pushing robot deliver the loaded <mask2> to the <mask3>.",
        "Advance the <mask2> containing the <mask1> to the <mask3>.",
    ],
    "unload": [
        "Unload the <mask1> from the <mask2> into the <mask3>.",
        "Retrieve the <mask1> from the <mask2> and place it in the <mask3>.",
        "Take the <mask1> out of the carrier and put it into the <mask3>.",
        "Use the waiting arm to remove the <mask1> from the <mask2> and place it in the <mask3>.",
        "Finish the carrier stage by unloading the <mask1> into the <mask3>.",
    ],
}

# The destination MUST be named: M3's goal is `<mask2> -> <mask3>` (the sink),
# and an unbound role word ("the receiver", "the handoff point") leaves it
# unrecoverable from the text -- the sibling `_F9_DESC_M2["push"]` always said
# `<mask3>`, M3 did not (docs/benchmark.md §7.6). Paraphrase ids stay aligned
# with the pre-fix list one-for-one; see the note on `_F4_DESC_M2`.
_F9_DESC_M3 = [
    "Load the <mask1> into the <mask2>, then push the loaded <mask2> to the <mask3>.",
    "Pack the <mask1> inside the <mask2> and have the pushing robot transport the carrier to the <mask3>.",
    "Place the <mask1> in the <mask2> before pushing the carrier over to the <mask3>.",
    "The arm should load the <mask1> into the <mask2>; afterward, push the carrier to the <mask3>.",
    "Prepare the <mask1> for relay by loading it in the <mask2> and moving the carrier to the <mask3>.",
]


def _build_f9(s, m, rng, realization, bridge_variant=None):
    roles = {
        2: ["dog", "arm"],
        3: ["dog", "arm"],
        4: ["dog", "arm", "arm"],
    }[m]
    tpl = _base("F9", s, m, RELAY_LAYOUTS, roles, [])
    tpl["mask1"] = list(_F9_ITEMS)
    tpl["mask2"] = ["cardboardbox"]
    # sink only (official 13-2 vocab): a destination anchored at R3 (pandaA)
    # requires pandaA_asset in the layout, which layout 3 lacks — a non-sink
    # destination would crash the render-config builder there
    tpl["mask3"] = ["sink"]
    bridge_variant = bridge_variant or "push"
    tpl["description"] = list(
        _F9_DESC_M2[bridge_variant]
        if m == 2 else _F9_DESC_M3 if m == 3 else _F9_DESC)
    if m == 2:
        tpl["bridge_kind"] = "carrier_components_two_robot"
        tpl["bridge_variant"] = bridge_variant
    elif m == 3:
        tpl["bridge_kind"] = "load_push_subchain"

    cand_keys = ["mask1"] if s >= 1 else []
    realizations = _binary_realizations(cand_keys) if cand_keys else [{}]
    if realization is None:
        realization = realizations[0]
    present = realization.get("mask1") == "present"

    if m == 2:
        # Each M2 instance exposes exactly one carrier component. Across the
        # cell, load/push/unload are all seen, but their M4 composition and
        # the third robot remain held out.
        init = [{"name_key": "mask3", "pos": ["R2"]},
                {"name_key": "R2", "pos": ["pandaB"]}]
        if bridge_variant == "load":
            init.insert(0, {"name_key": "mask2", "pos": ["R2"]})
            if present:
                init.insert(0, {"name_key": "mask1", "pos": ["cardboardbox"],
                                "aligned_keys": ["mask2"]})
                gt = []
            else:
                init.insert(0, {"name_key": "mask1", "pos": ["R2"]})
                gt = [
                    {"R2": ["Reach", "<mask1>"]},
                    {"R2": ["Grasp", "<mask1>"]},
                    {"R2": ["Place", "<mask2>"]},
                ]
        elif bridge_variant == "push":
            if present:
                init.insert(0, {"name_key": "mask2", "pos": ["<placeholder>"],
                                "aligned_keys": ["mask3"]})
            else:
                init.insert(0, {"name_key": "mask2", "pos": ["groundA"]})
            init.insert(0, {"name_key": "mask1", "pos": ["cardboardbox"],
                            "aligned_keys": ["mask2"]})
            gt = [] if present else [
                {"R1": ["Move", "<mask2>"]},
                {"R1": ["Push", "<mask2>", "<mask3>"]},
            ]
        else:  # unload
            init.insert(0, {"name_key": "mask2", "pos": ["R2"]})
            if present:
                init.insert(0, {"name_key": "mask1", "pos": ["<placeholder>"],
                                "aligned_keys": ["mask3"]})
                gt = []
            else:
                init.insert(0, {"name_key": "mask1", "pos": ["cardboardbox"],
                                "aligned_keys": ["mask2"]})
                gt = [
                    {"R2": ["Reach", "<mask1>"]},
                    {"R2": ["Grasp", "<mask1>"]},
                    {"R2": ["Place", "<mask3>"]},
                ]
    elif m == 3:
        # Two-segment subchain: one arm loads and the dog pushes the loaded
        # carrier to an open receiving area. There is deliberately no
        # second arm and no unload segment.
        init = [
            ({"name_key": "mask2", "pos": ["<placeholder>"],
              "aligned_keys": ["mask3"]} if present else
             {"name_key": "mask2", "pos": ["R2"]}),
            {"name_key": "mask3", "pos": ["groundA"]},
            {"name_key": "R2", "pos": ["pandaB"]},
        ]
        if present:
            init.insert(0, {"name_key": "mask1", "pos": ["cardboardbox"],
                            "aligned_keys": ["mask2"]})
            gt = []
        else:
            init.insert(0, {"name_key": "mask1", "pos": ["R2"]})
            gt = [
                {"R2": ["Reach", "<mask1>"]},
                {"R2": ["Grasp", "<mask1>"]},
                {"R2": ["Place", "<mask2>"]},
                {"R1": ["Move", "<mask2>"]},
                {"R1": ["Push", "<mask2>", "<mask3>"]},
            ]
    else:
        init = [
            {"name_key": "mask2", "pos": ["groundA"]},
            {"name_key": "mask3", "pos": ["R3"]},
            {"name_key": "R2", "pos": ["pandaB"]},
            {"name_key": "R3", "pos": ["pandaA"]},
        ]
        if present:
            # already inside/at the destination container
            init.insert(0, {"name_key": "mask1", "pos": ["<placeholder>"],
                            "aligned_keys": ["mask3"]})
            gt = []
        else:
            init.insert(0, {"name_key": "mask1", "pos": ["R2"]})
            gt = [
                {"R1": ["Move", "<mask2>"], "R2": ["Reach", "<mask1>"]},
                {"R1": ["Push", "<mask2>", "R2"],
                 "R2": ["Grasp", "<mask1>"]},
                {"R2": ["Place", "<mask2>"]},
                {"R1": ["Push", "<mask2>", "R3"]},
                {"R3": ["Reach", "<mask1>"]},
                {"R3": ["Grasp", "<mask1>"]},
                {"R3": ["Place", "<mask3>"]},
            ]

    tpl["init_pos"] = init
    tpl["ground_truth"] = gt
    if m == 2 and bridge_variant in ("load", "push"):
        tpl["goal_constraints"] = [_goal_at("<mask1>", "<mask2>")]
        if bridge_variant == "push":
            tpl["goal_constraints"].append(_goal_at("<mask2>", "<mask3>"))
    elif m == 3:
        tpl["goal_constraints"] = [
            _goal_at("<mask1>", "<mask2>"),
            _goal_at("<mask2>", "<mask3>"),
        ]
    else:
        tpl["goal_constraints"] = [_goal_at("<mask1>", "<mask3>")]
    state_dest = (
        "mask2" if m == 3 or (m == 2 and bridge_variant in ("load", "push"))
        else "mask3")
    tpl["state_realization_spec"] = _spec(
        "presatisfied" if cand_keys else "none",
        [{"mask_key": k, "states": ["absent", "present"],
          "dest_mask": state_dest}
         for k in cand_keys],
        realization, realizations)
    return tpl


# ---------------------------------------------------------------------------
# F10 inspect_area · S0 × M0
# ---------------------------------------------------------------------------

_F10_TARGETS = ["drawer", "fridge", "oven"]

_F10_DESC = [
    "Inspect the environment of <mask1>.",
    "Take a look around <mask1> and check its surroundings.",
    "Examine the environment near <mask1>.",
    "Observe the space around <mask1>.",
    "Explore the nearby environment of <mask1>.",
    "Assess the area surrounding <mask1>.",
    "Patrol over to <mask1> and give its vicinity a check.",
    "Head to <mask1> and survey the area carefully.",
]


def _build_f10(s, m, rng, realization):
    tpl = _base("F10", s, m, ALL_LAYOUTS, ["dog"], ["wheeled"])
    tpl["mask1"] = list(_F10_TARGETS)
    tpl["description"] = list(_F10_DESC)
    tpl["init_pos"] = [{"name_key": "mask1", "pos": ["mask1"]}]
    tpl["ground_truth"] = [
        {"R1": ["Move", "<mask1>"]},
        {"R1": ["Interact", "<mask1>"]},
    ]
    tpl["goal_constraints"] = [_goal_activated("<mask1>")]
    tpl["state_realization_spec"] = _spec("none", [], {}, [{}])
    return tpl


# ---------------------------------------------------------------------------
# registry / public API
# ---------------------------------------------------------------------------

def _cross(s_values, m_values):
    return tuple((s, m) for s in s_values for m in m_values)


# Explicit cells are intentional. F4 has the non-contiguous S set {2, 4};
# using min/max ranges would silently create the forbidden F4-S3 family.
FAMILY_CELLS = {
    "F1": {"cells": _cross((0, 1), (0, 1, 2)), "builder": _build_f1},
    "F2": {"cells": _cross((1, 2), (0, 1, 2)), "builder": _build_f2},
    "F3": {"cells": _cross((2, 3), (0, 1, 2)), "builder": _build_f3},
    "F4": {"cells": _cross((2, 4), (1, 2, 3)), "builder": _build_f4},
    "F5": {"cells": _cross((2, 3), (1, 2, 3)), "builder": _build_f5},
    "F6": {"cells": _cross((1, 2), (2, 3)), "builder": _build_f6},
    "F7": {"cells": _cross((0, 1), (1, 2)), "builder": _build_f7},
    "F8": {"cells": _cross((0, 1), (2, 3)), "builder": _build_f8},
    "F9": {"cells": _cross((0, 1), (2, 3, 4)), "builder": _build_f9},
    "F10": {"cells": ((0, 0),), "builder": _build_f10},
}


def iter_family_cells(family=None):
    """Yield explicit ``(family, s, m)`` triples for valid family-cells."""
    families = [family] if family else list(FAMILY_CELLS)
    for fam in families:
        cfg = FAMILY_CELLS[fam]
        for s, m in cfg["cells"]:
            yield fam, s, m


# Backward-compatible name; callers still use the explicit registry.
iter_cells = iter_family_cells


def make_template(family, s, m, rng=None, realization=None,
                  bridge_variant=None):
    """Build one official-schema template for cell (s, m) of ``family``.

    ``realization`` selects which state realization the returned template's
    ground_truth / init_pos encode (default: realizations[0], the
    max-structure one). Must be an element of
    ``enumerate_realizations(family, s, m)``.
    """
    cfg = FAMILY_CELLS[family]
    if (s, m) not in cfg["cells"]:
        raise ValueError(f"{family} does not cover cell (s={s}, m={m})")
    rng = rng or random.Random()
    if family == "F9":
        if m == 2 and bridge_variant not in (None, "load", "push", "unload"):
            raise ValueError(f"unsupported F9-M2 bridge_variant: "
                             f"{bridge_variant}")
        tpl = cfg["builder"](s, m, rng, realization, bridge_variant)
    else:
        tpl = cfg["builder"](s, m, rng, realization)
    return tpl


def enumerate_realizations(family, s, m, rng=None):
    """Full realization enumeration for a cell (for balanced sampling, P7.2)."""
    tpl = make_template(family, s, m, rng or random.Random(0))
    return tpl["state_realization_spec"]["realizations"]


def build_locus_pool(seed=0, per_cell=1):
    """Convenience: one default-realization template per cell."""
    rng = random.Random(seed)
    pool = []
    for fam, s, m in iter_cells():
        for _ in range(per_cell):
            pool.append(make_template(fam, s, m, rng))
    return pool


if __name__ == "__main__":
    pool = build_locus_pool()
    print(f"{len(pool)} templates across {len(list(iter_cells()))} cells")
    for tpl in pool:
        n_real = len(tpl["state_realization_spec"]["realizations"])
        print(f'{tpl["task_name"]:>10}  robots={tpl["robot_roles"]}  '
              f'gt_steps={len(tpl["ground_truth"])}  realizations={n_real}')
