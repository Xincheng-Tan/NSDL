"""S4 dataset-side plan repair / filter primitives.

These helpers are used **only** during S4 GT-trajectory construction
(``scripts/build_s4_gt_trajectories.py`` and ``scripts/repair_s4_plans.py``).
They are *not* part of the framework's runtime validator/dispatcher — see
``docs/agent.md``: at inference time the validator is detect-only and never
mutates a plan. Touching this module therefore cannot pollute the
inference-time validator boundary as long as nothing under
``nsdl/{validator,dispatcher,micro_planner}.py`` imports it.

Per ``docs/debug.md`` (2026-05-10):

    Common-1  closed openable receptacle missing OpenObject
    Common-2  CleanObject on non-dirtyable object
    Common-3  hand-busy (consecutive PickupObject without intervening Put)
    Common-4  generic synonym (table/light/...) absent from scene
    A-spec-1  PutObject -> receptacle full
    A-spec-2  ToggleObjectOn -> target controlled by another sim object
    B-spec-1  SwitchOn target already toggled in scene reset state
    B-spec-2  affordance flag mismatch on synth ops
    B-spec-3  PutObject onto non-receptacle
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Iterable

# Closed-by-default openable receptacles. AI2-THOR `forceAction=True` does
# *not* bypass the "target openable receptacle is CLOSED" precondition for
# PutObject, so we must insert OpenObject ourselves.
OPENABLE_RECEPTACLES: frozenset[str] = frozenset({
    "Fridge", "Microwave", "Cabinet", "Drawer", "Safe", "Box", "Toaster",
})

# Generic synonyms that frequently appear in ALFRED instructions but are
# not direct AI2-THOR objectTypes. Recorded for diagnostic only — Common-4
# is handled by *dropping* the sample, not by alias-resolution (per user's
# instruction: "无法鲁棒修复，直接删除").
GENERIC_NOUNS: frozenset[str] = frozenset({
    "Table", "Light", "Counter", "Lamp", "Shelf",
})

# Objects whose CleanObject in AI2-THOR is rejected with "does not have
# dirtyable property". MAT-THOR's official ``washproblem.pddl`` does
# include ``(clean tomato1)`` as a goal, so we cannot drop these samples
# without losing benchmark coverage. Instead we **rewrite** the single
# ``CleanObject(o)`` op into the canonical Sink/Faucet 6-step sequence
# (Pickup -> GoTo Sink -> Put -> ToggleOn Faucet -> ToggleOff Faucet),
# leaving the object inside SinkBasin so the original tail of the plan
# (which already does ``GoTo Sink -> Pickup o``) still executes. This
# is a *macro-to-atomic execution-detail expansion*, not a planning
# repair, and therefore consistent with the dispatcher/micro-planner
# being allowed to expand a macro op into THOR primitives (see
# ``docs/agent.md`` -- the validator boundary forbids inventing new
# planning ops, but execution-layer expansion is the dispatcher's job).
NON_DIRTYABLE_FOR_CLEAN: frozenset[str] = frozenset({
    "Apple", "Knife", "ButterKnife", "Tomato", "Lettuce", "Potato",
    "Bread", "Egg", "Pen", "Pencil", "Book", "Box", "Bottle", "CD",
    "Watch", "AlarmClock", "Vase", "WineBottle", "TissueBox",
    "Spoon", "Fork", "Mirror",  # Mirror is fixed but not CleanObject-able in train scenes
})

# ALFRED A-spec-2: actions whose target is controlled by another simulator
# object. For these we drop the sample.
CONTROLLED_TARGETS_DROP: frozenset[str] = frozenset({"StoveBurner"})


# ---------------------------------------------------------------------------
# Scene-metadata loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneFlags:
    """Per-objectType affordance / initial-state lookup, derived from
    ``lamma-p/data/train_scene_metadata/FloorPlan{N}.json::raw_objects``.
    """
    object_types: frozenset[str]
    openable: frozenset[str]
    is_open_initial: frozenset[str]
    pickupable: frozenset[str]
    receptacle: frozenset[str]
    sliceable: frozenset[str]
    breakable: frozenset[str]
    dirtyable: frozenset[str]
    toggleable: frozenset[str]
    is_toggled_initial: frozenset[str]
    cookable: frozenset[str]
    # Per-instance physical state for fine-grained checks.
    receptacle_load: dict[str, int] = field(default_factory=dict)


def load_scene_flags(meta_path: Path) -> SceneFlags:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = meta.get("raw_objects") or []
    object_types = sorted({str(o.get("objectType")) for o in raw if o.get("objectType")})
    by_flag: dict[str, set[str]] = {
        k: set() for k in (
            "openable", "is_open", "pickupable", "receptacle",
            "sliceable", "breakable", "dirtyable",
            "toggleable", "is_toggled", "cookable",
        )
    }
    receptacle_load: dict[str, int] = {}
    for o in raw:
        ot = str(o.get("objectType") or "")
        if not ot:
            continue
        if o.get("openable"):
            by_flag["openable"].add(ot)
        if o.get("isOpen"):
            by_flag["is_open"].add(ot)
        if o.get("pickupable"):
            by_flag["pickupable"].add(ot)
        if o.get("receptacle"):
            by_flag["receptacle"].add(ot)
        if o.get("sliceable"):
            by_flag["sliceable"].add(ot)
        if o.get("breakable"):
            by_flag["breakable"].add(ot)
        if o.get("dirtyable"):
            by_flag["dirtyable"].add(ot)
        if o.get("toggleable"):
            by_flag["toggleable"].add(ot)
        if o.get("isToggled"):
            by_flag["is_toggled"].add(ot)
        if o.get("cookable"):
            by_flag["cookable"].add(ot)
        n_children = len(o.get("receptacleObjectIds") or []) if o.get("receptacle") else 0
        # Track per-type *max* current load (worst-case capacity check).
        receptacle_load[ot] = max(receptacle_load.get(ot, 0), n_children)
    return SceneFlags(
        object_types=frozenset(object_types),
        openable=frozenset(by_flag["openable"]),
        is_open_initial=frozenset(by_flag["is_open"]),
        pickupable=frozenset(by_flag["pickupable"]),
        receptacle=frozenset(by_flag["receptacle"]),
        sliceable=frozenset(by_flag["sliceable"]),
        breakable=frozenset(by_flag["breakable"]),
        dirtyable=frozenset(by_flag["dirtyable"]),
        toggleable=frozenset(by_flag["toggleable"]),
        is_toggled_initial=frozenset(by_flag["is_toggled"]),
        cookable=frozenset(by_flag["cookable"]),
        receptacle_load=receptacle_load,
    )


# ---------------------------------------------------------------------------
# PDDL goal -> object_states (for run_thor.py SR scoring)
# ---------------------------------------------------------------------------

# Mapping from PDDL predicate (lower-case) to either a state string or
# ``"AT_LOCATION"`` sentinel for the receptacle-containment idiom.
_PRED_TO_STATE: dict[str, str | None] = {
    "switch-on": "ON",
    "switch-off": "OFF",
    "object-open": "OPEN",
    "object-close": "CLOSED",
    "sliced": "SLICED",
    "cleaned": "CLEAN",
    "break": "BROKEN",
    "broken": "BROKEN",
    "cooked": "COOKED",
    "filled": "FILLED",
}


def _strip_digits(sym: str) -> str:
    """``apple1`` -> ``apple``."""
    return re.sub(r"\d+$", "", str(sym).lower())


def _to_camel(sym: str, lookup: dict[str, str] | None) -> str:
    base = _strip_digits(sym)
    if lookup and base in lookup:
        return lookup[base]
    return base[:1].upper() + base[1:]


def derive_object_states(
    state: dict[str, Any],
    *,
    object_type_lookup: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Translate the PDDL ``:goal`` clause from ``state.json`` into a list
    of ``{name, contains?, state?}`` records consumable by
    ``run_thor.py::_check_object_states``.

    Aggregation rule: multiple ``(at-location ?obj <loc>)`` predicates that
    share the same ``loc`` collapse into a single
    ``{name: <Loc>, contains: [<obj>, ...]}`` record so SR scoring runs in
    O(#receptacles).
    """
    goal = state.get("goal") or []
    by_loc: dict[str, list[str]] = {}
    state_specs: list[dict[str, Any]] = []
    for pred in goal:
        if isinstance(pred, str):
            tokens = re.findall(r"[a-zA-Z0-9\-]+", pred)
        elif isinstance(pred, (list, tuple)):
            tokens = [str(t) for t in pred]
        else:
            continue
        if not tokens:
            continue
        head = tokens[0].lower()
        args = tokens[1:]
        if head == "at-location" and len(args) >= 2:
            obj_sym, loc_sym = args[0], args[1]
            loc_camel = _to_camel(loc_sym, object_type_lookup)
            by_loc.setdefault(loc_camel, []).append(_to_camel(obj_sym, object_type_lookup))
        elif head in _PRED_TO_STATE:
            target_sym = args[-1] if args else None
            if not target_sym:
                continue
            state_specs.append({
                "name": _to_camel(target_sym, object_type_lookup),
                "state": _PRED_TO_STATE[head],
            })
        elif head == "holding":
            # transient -- skip for SR check
            continue
        # all other predicates are ignored (they are domain bookkeeping)
    out: list[dict[str, Any]] = []
    for loc, kids in by_loc.items():
        out.append({"name": loc, "contains": sorted(set(kids))})
    out.extend(state_specs)
    return out


# ---------------------------------------------------------------------------
# Plan-level repairs
# ---------------------------------------------------------------------------

def _collect_steps(agent_plans: dict[str, list[dict]]) -> list[tuple[str, int, dict]]:
    """Flatten ``{aid: [step, ...]}`` to ``[(aid, idx, step), ...]``."""
    flat: list[tuple[str, int, dict]] = []
    for aid in sorted(agent_plans.keys(), key=lambda x: int(x)):
        for i, step in enumerate(agent_plans[aid]):
            flat.append((aid, i, step))
    return flat


def _renumber(agent_plans: dict[str, list[dict]]) -> None:
    """Re-issue ``step_id`` after mutation. We use ``n0.<idx>`` per agent
    (matches existing emitter convention)."""
    for aid, queue in agent_plans.items():
        for i, step in enumerate(queue):
            step["step_id"] = f"n0.{i}"


@dataclass
class RepairOutcome:
    """Per-plan repair report."""
    qid: str
    drop: bool = False
    drop_reasons: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)


def insert_open_for_closed_recep(
    agent_plans: dict[str, list[dict]],
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
) -> None:
    """Common-1: before ``PutObject(... -> R)`` where R is openable and not
    initially open, ensure an ``OpenObject(R)`` was issued earlier on the
    same agent. If absent, splice one in directly before the Put."""
    for aid, queue in agent_plans.items():
        i = 0
        while i < len(queue):
            step = queue[i]
            if step.get("action") != "PutObject":
                i += 1
                continue
            recep = step.get("receptacleObjectId") or step.get("objectId")
            if recep not in flags.openable or recep in flags.is_open_initial:
                i += 1
                continue
            already = any(
                s.get("action") == "OpenObject" and s.get("objectId") == recep
                for s in queue[:i]
            )
            if already:
                i += 1
                continue
            # Splice an OpenObject(R). Also splice GoToObject(R) if the most
            # recent navigation target on this agent is not R.
            insert: list[dict] = []
            recent_goto = next(
                (s.get("objectId") for s in reversed(queue[:i])
                 if s.get("action") == "GoToObject"),
                None,
            )
            if recent_goto != recep:
                insert.append({"action": "GoToObject", "objectId": recep})
            insert.append({"action": "OpenObject", "objectId": recep,
                           "auto_inserted": True})
            queue[i:i] = insert
            outcome.changes.append(
                f"insert_open_for_{recep}@agent{aid}:+{len(insert)}"
            )
            i += 1 + len(insert)


def fix_hand_busy(
    agent_plans: dict[str, list[dict]],
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
) -> None:
    """Common-3: scan each agent. If a second ``PickupObject`` happens
    while the hand is non-empty, splice ``PutObject(prev_held, last_loc)``
    before the second pickup, where ``last_loc`` is taken from the most
    recent ``GoToObject`` argument (a local CounterTop / Sink / etc.).

    The splice is only safe if the previously-held object has *no
    downstream* ``PutObject`` of its own, otherwise that later put would
    fail (we'd have already empty-handed it). When unsafe, drop the
    sample instead."""
    for aid, queue in agent_plans.items():
        held: str | None = None
        last_loc: str | None = None
        i = 0
        while i < len(queue):
            step = queue[i]
            act = step.get("action")
            obj = step.get("objectId")
            if act == "GoToObject":
                last_loc = obj
            elif act == "PickupObject":
                if held is not None:
                    if held == obj:
                        # exact same instance -> infeasible by construction
                        outcome.drop = True
                        outcome.drop_reasons.append("hand_busy_same_object")
                        return
                    # Look ahead: does the previously-held object get
                    # PutObject'd later? If so, splicing an early Put here
                    # would break that downstream step. Drop instead.
                    has_downstream_put = any(
                        s.get("action") == "PutObject" and s.get("objectId") == held
                        for s in queue[i:]
                    )
                    if has_downstream_put:
                        outcome.drop = True
                        outcome.drop_reasons.append(
                            f"hand_busy_conflict_with_later_put:{held}")
                        return
                    drop_at = (
                        last_loc if (last_loc and last_loc in flags.receptacle)
                        else None
                    )
                    if drop_at is None:
                        outcome.drop = True
                        outcome.drop_reasons.append("hand_busy_no_drop_recep")
                        return
                    splice = [
                        {"action": "PutObject",
                         "objectId": held,
                         "receptacleObjectId": drop_at,
                         "auto_inserted": True},
                    ]
                    queue[i:i] = splice
                    outcome.changes.append(
                        f"hand_busy_put_{held}_to_{drop_at}@agent{aid}"
                    )
                    held = obj
                    i += 1 + len(splice)
                    continue
                held = obj
            elif act == "PutObject":
                if held == obj:
                    held = None
            i += 1


def affordance_drop_check(
    agent_plans: dict[str, list[dict]],
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
    is_alfred: bool,
) -> None:
    """Common-2 / Common-4 / A-spec-2 / B-spec-2 / B-spec-3: scan all
    steps and decide whether the sample should be dropped.

    Rules (any hit ⇒ drop):
        * objectId / receptacleObjectId not in scene object_types
        * CleanObject on a type not in flags.dirtyable
        * OpenObject on a type not in flags.openable
        * PickupObject on a type not in flags.pickupable
        * SliceObject on a type not in flags.sliceable
        * BreakObject on a type not in flags.breakable
        * Toggle{On,Off}/Switch{On,Off} on a type not in flags.toggleable
        * PutObject onto a non-receptacle
        * Toggle on CONTROLLED_TARGETS_DROP (A-spec-2 for ALFRED)
    """
    flat = _collect_steps(agent_plans)
    for _aid, _idx, step in flat:
        act = step.get("action")
        obj = step.get("objectId")
        recep = step.get("receptacleObjectId")
        # generic existence check (Common-4)
        for cand in (obj, recep):
            if cand and cand not in flags.object_types:
                if cand in GENERIC_NOUNS:
                    outcome.drop = True
                    outcome.drop_reasons.append(f"generic_noun:{cand}")
                    return
                outcome.drop = True
                outcome.drop_reasons.append(f"missing_in_scene:{cand}")
                return
        if act == "CleanObject":
            # Non-dirtyable CleanObject is *rewritten* later by
            # ``rewrite_clean_non_dirtyable_to_faucet``; do not drop here.
            pass
        elif act == "OpenObject" and obj not in flags.openable:
            outcome.drop = True
            outcome.drop_reasons.append(f"open_non_openable:{obj}")
            return
        elif act == "CloseObject" and obj not in flags.openable:
            outcome.drop = True
            outcome.drop_reasons.append(f"close_non_openable:{obj}")
            return
        elif act == "PickupObject" and obj not in flags.pickupable:
            outcome.drop = True
            outcome.drop_reasons.append(f"pickup_non_pickupable:{obj}")
            return
        elif act == "SliceObject" and obj not in flags.sliceable:
            outcome.drop = True
            outcome.drop_reasons.append(f"slice_non_sliceable:{obj}")
            return
        elif act == "BreakObject" and obj not in flags.breakable:
            outcome.drop = True
            outcome.drop_reasons.append(f"break_non_breakable:{obj}")
            return
        elif act in {"ToggleObjectOn", "ToggleObjectOff", "SwitchOn", "SwitchOff"}:
            if obj in CONTROLLED_TARGETS_DROP and is_alfred:
                outcome.drop = True
                outcome.drop_reasons.append(f"toggle_controlled:{obj}")
                return
            if obj not in flags.toggleable:
                outcome.drop = True
                outcome.drop_reasons.append(f"toggle_non_toggleable:{obj}")
                return
        elif act == "PutObject":
            tgt = recep or obj
            if tgt not in flags.receptacle:
                outcome.drop = True
                outcome.drop_reasons.append(f"put_non_receptacle:{tgt}")
                return


def alfred_recep_full_check(
    agent_plans: dict[str, list[dict]],
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
) -> None:
    """A-spec-1: drop ALFRED sample if any PutObject targets a receptacle
    that already has ≥3 children at scene reset (Pan/Pot/DiningTable
    physical capacity exhausted)."""
    for _aid, _idx, step in _collect_steps(agent_plans):
        if step.get("action") != "PutObject":
            continue
        tgt = step.get("receptacleObjectId") or step.get("objectId")
        if not tgt:
            continue
        load = flags.receptacle_load.get(tgt, 0)
        if load >= 3:
            outcome.drop = True
            outcome.drop_reasons.append(f"recep_full:{tgt}({load})")
            return


def rewrite_clean_non_dirtyable_to_faucet(
    agent_plans: dict[str, list[dict]],
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
) -> None:
    """Common-2 (rewrite path): replace ``CleanObject(o)`` for
    non-dirtyable ``o`` with the canonical AI2-THOR Sink/Faucet sequence:

        PickupObject(o)            # if hand is empty (else skip)
        GoToObject(Sink)
        PutObject(o -> SinkBasin)  # uses SinkBasin if available else Sink
        ToggleObjectOn(Faucet)
        ToggleObjectOff(Faucet)

    The object is *left inside the SinkBasin* (no final re-pickup), so
    the original tail of the plan -- which typically already contains
    ``GoToObject(Sink) -> PickupObject(o)`` -- still executes correctly.

    Drop the sample when:
      * the scene has no Sink+Faucet (e.g. living-room / bedroom);
      * the hand is non-empty with a *different* object at the moment
        we hit ``CleanObject``, AND that object is needed downstream
        (would lose it on the splice).
    """
    sink_recep = "SinkBasin" if "SinkBasin" in flags.receptacle else (
        "Sink" if "Sink" in flags.receptacle else None
    )
    has_faucet = "Faucet" in flags.toggleable
    nav_to_sink = "Sink" if "Sink" in flags.object_types else (
        "SinkBasin" if "SinkBasin" in flags.object_types else None
    )

    for aid, queue in agent_plans.items():
        held: str | None = None
        i = 0
        while i < len(queue):
            step = queue[i]
            act = step.get("action")
            obj = step.get("objectId")
            if act == "PickupObject":
                held = obj
                i += 1
                continue
            if act == "PutObject" and obj == held:
                held = None
                i += 1
                continue
            if act != "CleanObject":
                i += 1
                continue
            # We are at CleanObject(obj). Decide rewrite vs leave-as-is.
            if obj in flags.dirtyable and obj not in NON_DIRTYABLE_FOR_CLEAN:
                # Real dirtyable object -- single op is fine.
                i += 1
                continue
            # Need rewrite. Check scene capability.
            if sink_recep is None or not has_faucet or nav_to_sink is None:
                outcome.drop = True
                outcome.drop_reasons.append(
                    f"clean_non_dirtyable_no_sink:{obj}"
                )
                return
            # Fixed objects (Mirror, Bathtub, Sink, Toilet, CounterTop, ...)
            # cannot be picked up and dunked under a faucet -- the rewrite
            # is meaningless. Drop.
            if obj not in flags.pickupable:
                outcome.drop = True
                outcome.drop_reasons.append(
                    f"clean_non_dirtyable_fixed:{obj}"
                )
                return
            # Hand-state check.
            if held is not None and held != obj:
                # Splicing a Pickup(obj) here would conflict; drop.
                outcome.drop = True
                outcome.drop_reasons.append(
                    f"clean_non_dirtyable_hand_busy:{held}"
                )
                return
            need_pickup = held is None
            splice: list[dict] = []
            if need_pickup:
                splice.append({"action": "PickupObject", "objectId": obj,
                               "auto_inserted": True})
            splice.extend([
                {"action": "GoToObject", "objectId": nav_to_sink,
                 "auto_inserted": True},
                {"action": "PutObject", "objectId": obj,
                 "receptacleObjectId": sink_recep, "auto_inserted": True},
                {"action": "ToggleObjectOn", "objectId": "Faucet",
                 "auto_inserted": True},
                {"action": "ToggleObjectOff", "objectId": "Faucet",
                 "auto_inserted": True},
                {"action": "PickupObject", "objectId": obj,
                 "auto_inserted": True},  # re-grab from basin
            ])
            queue[i:i + 1] = splice
            outcome.changes.append(
                f"clean_to_faucet_{obj}@agent{aid}:+{len(splice) - 1}"
            )
            # Scrub any *immediately downstream* duplicate PickupObject(obj)
            # before its next PutObject -- the apple is already held now.
            j = i + len(splice)
            while j < len(queue):
                act_j = queue[j].get("action")
                obj_j = queue[j].get("objectId")
                if act_j == "PickupObject" and obj_j == obj:
                    queue.pop(j)
                    continue
                if act_j == "PutObject" and obj_j == obj:
                    break
                j += 1
            held = obj  # held after re-pickup
            i += len(splice)


def inject_faucet_for_clean_goal(
    agent_plans: dict[str, list[dict]],
    state: dict[str, Any] | None,
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
    object_type_lookup: dict[str, str] | None,
) -> None:
    """If the PDDL goal contains ``(clean X)`` for a non-dirtyable ``X``
    AND the plan does not already contain ``CleanObject(X)`` or a
    Faucet-toggle pair, inject the Sink/Faucet 6-step sequence right
    before the first ``PickupObject(X)`` or ``GoToObject(X)``.

    This is the *idempotent companion* to ``rewrite_clean_non_dirtyable_to_faucet``
    -- it covers the case where a previous repair pass already stripped
    the original CleanObject(X) op (so there is no longer a CleanObject
    step to rewrite) but the goal still semantically requires the wash.
    """
    if state is None:
        return
    goal = state.get("goal") or []
    targets: list[str] = []
    for pred in goal:
        if isinstance(pred, str):
            tokens = re.findall(r"[a-zA-Z0-9\-]+", pred)
        elif isinstance(pred, (list, tuple)):
            tokens = [str(t) for t in pred]
        else:
            continue
        if len(tokens) >= 2 and tokens[0].lower() in {"clean", "cleaned"}:
            targets.append(_to_camel(tokens[-1], object_type_lookup))
    if not targets:
        return

    sink_recep = "SinkBasin" if "SinkBasin" in flags.receptacle else (
        "Sink" if "Sink" in flags.receptacle else None
    )
    nav_to_sink = "Sink" if "Sink" in flags.object_types else (
        "SinkBasin" if "SinkBasin" in flags.object_types else None
    )
    has_faucet = "Faucet" in flags.toggleable

    for tgt in targets:
        if tgt in flags.dirtyable and tgt not in NON_DIRTYABLE_FOR_CLEAN:
            continue  # real dirtyable -- proper CleanObject elsewhere
        if tgt not in flags.pickupable:
            continue  # fixed object -- already handled / dropped
        # skip if any agent already has a CleanObject(tgt) or a Faucet-on
        # before a Faucet-off (already wash-rewritten).
        already_handled = False
        for queue in agent_plans.values():
            if any(s.get("action") == "CleanObject" and s.get("objectId") == tgt
                   for s in queue):
                already_handled = True
                break
            faucet_on = False
            for s in queue:
                a = s.get("action")
                if a == "ToggleObjectOn" and s.get("objectId") == "Faucet":
                    faucet_on = True
                elif a == "ToggleObjectOff" and s.get("objectId") == "Faucet" and faucet_on:
                    already_handled = True
                    break
            if already_handled:
                break
        if already_handled:
            continue
        if sink_recep is None or not has_faucet or nav_to_sink is None:
            outcome.drop = True
            outcome.drop_reasons.append(f"clean_goal_no_sink:{tgt}")
            return
        # Pick the first agent that touches `tgt` (PickupObject or GoToObject).
        injected = False
        for aid, queue in agent_plans.items():
            for i, s in enumerate(queue):
                if s.get("objectId") != tgt:
                    continue
                act = s.get("action")
                if act not in {"PickupObject", "GoToObject"}:
                    continue
                # Insert Faucet sequence after this step (and Pickup if it's
                # currently a GoToObject, since hand is empty).
                pos = i + 1
                splice: list[dict] = []
                # If we inserted before any Pickup, we need to Pickup first.
                if act == "GoToObject":
                    # Check: is there a PickupObject(tgt) later? if so,
                    # we still pickup here -- the later one will be
                    # scrubbed by us.
                    splice.append({"action": "PickupObject", "objectId": tgt,
                                   "auto_inserted": True})
                splice.extend([
                    {"action": "GoToObject", "objectId": nav_to_sink,
                     "auto_inserted": True},
                    {"action": "PutObject", "objectId": tgt,
                     "receptacleObjectId": sink_recep, "auto_inserted": True},
                    {"action": "ToggleObjectOn", "objectId": "Faucet",
                     "auto_inserted": True},
                    {"action": "ToggleObjectOff", "objectId": "Faucet",
                     "auto_inserted": True},
                    {"action": "PickupObject", "objectId": tgt,
                     "auto_inserted": True},
                ])
                queue[pos:pos] = splice
                # Scrub immediately downstream duplicate PickupObject(tgt)
                j = pos + len(splice)
                while j < len(queue):
                    a2 = queue[j].get("action")
                    o2 = queue[j].get("objectId")
                    if a2 == "PickupObject" and o2 == tgt:
                        queue.pop(j)
                        continue
                    if a2 == "PutObject" and o2 == tgt:
                        break
                    j += 1
                outcome.changes.append(
                    f"inject_faucet_for_clean_{tgt}@agent{aid}:+{len(splice)}"
                )
                injected = True
                break
            if injected:
                break


def light_switch_already_on_rewrite(
    agent_plans: dict[str, list[dict]],
    flags: SceneFlags,
    *,
    outcome: RepairOutcome,
) -> None:
    """B-spec-1: when a synth plan calls ``ToggleObjectOn(o)`` /
    ``SwitchOn(o)`` where ``o`` is already ``isToggled=True`` at scene
    reset, rewrite to the off-variant (most semantically equivalent
    interpretation: "operate the light" -> "switch the light off")."""
    for aid, queue in agent_plans.items():
        for step in queue:
            act = step.get("action")
            obj = step.get("objectId")
            if act in {"ToggleObjectOn", "SwitchOn"} and obj in flags.is_toggled_initial:
                step["action"] = "ToggleObjectOff" if act == "ToggleObjectOn" else "SwitchOff"
                step["auto_rewritten"] = "already_on"
                outcome.changes.append(f"already_on_flip_{obj}@agent{aid}")
            elif act in {"ToggleObjectOff", "SwitchOff"} and obj not in flags.is_toggled_initial and obj in flags.toggleable:
                # symmetric (already off) -- left as-is; not a known failure mode
                pass


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def repair_plan_in_place(
    plan: dict[str, Any],
    state: dict[str, Any] | None,
    flags: SceneFlags | None,
    *,
    qid: str,
    object_type_lookup: dict[str, str] | None = None,
) -> RepairOutcome:
    """Apply all S4-side repairs / drops to ``plan`` mutably.

    Returns a :class:`RepairOutcome`. If ``outcome.drop`` is True, callers
    must discard the sample (do **not** write the partially-mutated plan
    to disk in that case)."""
    outcome = RepairOutcome(qid=qid)
    agent_plans = plan.get("agent_plans") or {}
    is_alfred = qid.startswith("alfred_")

    if flags is not None:
        # Pre-mutation drop checks first, so we don't waste work on bad samples.
        affordance_drop_check(agent_plans, flags, outcome=outcome,
                              is_alfred=is_alfred)
        if outcome.drop:
            return outcome
        if is_alfred:
            alfred_recep_full_check(agent_plans, flags, outcome=outcome)
            if outcome.drop:
                return outcome
        # Rewrites + insertions.
        rewrite_clean_non_dirtyable_to_faucet(agent_plans, flags, outcome=outcome)
        if outcome.drop:
            return outcome
        inject_faucet_for_clean_goal(
            agent_plans, state, flags,
            outcome=outcome, object_type_lookup=object_type_lookup,
        )
        if outcome.drop:
            return outcome
        light_switch_already_on_rewrite(agent_plans, flags, outcome=outcome)
        insert_open_for_closed_recep(agent_plans, flags, outcome=outcome)
        fix_hand_busy(agent_plans, flags, outcome=outcome)
        if outcome.drop:
            return outcome
        _renumber(agent_plans)
    else:
        # No scene metadata available. Restrict to schema-only checks
        # (still try hand-busy fix, since it doesn't require flags).
        fix_hand_busy(agent_plans, SceneFlags(
            object_types=frozenset(), openable=frozenset(),
            is_open_initial=frozenset(), pickupable=frozenset(),
            receptacle=frozenset(), sliceable=frozenset(),
            breakable=frozenset(), dirtyable=frozenset(),
            toggleable=frozenset(), is_toggled_initial=frozenset(),
            cookable=frozenset(),
        ), outcome=outcome)
        if outcome.drop:
            return outcome
        _renumber(agent_plans)

    # Derive object_states (always, even if no flags) so
    # ``run_thor.py`` can do strict GT verification.
    if state is not None:
        plan["object_states"] = derive_object_states(
            state, object_type_lookup=object_type_lookup,
        )
    return outcome


__all__ = [
    "OPENABLE_RECEPTACLES",
    "NON_DIRTYABLE_FOR_CLEAN",
    "GENERIC_NOUNS",
    "CONTROLLED_TARGETS_DROP",
    "SceneFlags",
    "load_scene_flags",
    "derive_object_states",
    "RepairOutcome",
    "repair_plan_in_place",
]
