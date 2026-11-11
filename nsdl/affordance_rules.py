"""Affordance-Constrained DAG Synthesis (AC-DAG) — Tag-Triggered Constraint Rules.

This module implements Phase 1 (Constraint Closure) of EnactGraph's
capability constraint layer. Unlike semantic-intent-based motif detection
(which requires recognizing task-type keywords like "wash", "activate",
"tool_interaction" from parsed instruction text), the AC-DAG mechanism derives
mandatory topology obligations ENTIRELY from the affordance tags of the
entities involved in the goal state.

Core principle (OOD generalization):
    Entity affordance tags are physical invariants that do not change with task
    distribution. A 'sink' always has the 'wash_station' tag regardless of how
    the instruction phrases the cleaning step. This makes topology derivation
    robust to out-of-distribution task types that involve familiar entity types.

Two-phase synthesis (AC-DAG):
    Phase 1 — Constraint Closure (this module):
        For each entity in the goal state, apply affordance-tag-triggered rules
        to derive ConstraintObligation objects. These obligations encode what
        DAG structure is physically REQUIRED by the affordance profile of the
        entities in the goal state.

    Phase 2 — Capability-Feasible Assembly (macro_graph.py):
        Map the constraint obligation profile to a capability-feasible DAG
        topology. LLM/VLM is used only for disambiguating equally-scored
        candidates, NOT for generating topology from scratch.

Key differentiation from prior work:
    - vs. SPINE-HT: constraints bound topology BEFORE LLM (early binding),
      not after LLM generation (late binding / post-hoc assignment).
    - vs. EMOS: capability constraints shape the DAG topology structure itself,
      not just the robot assignment step.
    - vs. CoMuRoS: constraint reasoning is formal (tag-driven), not LLM-implicit.
    - vs. Plan-then-Assign: topology exists only within the capability-feasible
      space; infeasible topologies never enter the candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nsdl.catalog import (
    ACTIVATABLE_TAG,
    CONTAINER_TAG,
    CUTTING_SURFACE_TAG,
    OPENABLE_TAG,
    PUSHABLE_CARRIER_TAG,
    WASH_STATION_TAG,
    entity_has_tag,
    normalize_entity_name,
)

if TYPE_CHECKING:
    from nsdl.schema import SceneFacts

# ---------------------------------------------------------------------------
# Constraint Obligation type constants
# ---------------------------------------------------------------------------

OBLIGATION_OPEN_BEFORE_PLACE = "open_before_place"
OBLIGATION_PROCESS_AT_WASH_STATION = "process_at_wash_station"
OBLIGATION_PLACE_BEFORE_TOOL_INTERACT = "place_before_tool_interact"
OBLIGATION_ACTIVATE_FIXTURE_AFTER_PLACE = "activate_fixture_after_place"
OBLIGATION_PUSH_CARRIER_HANDOFF = "push_carrier_handoff"


@dataclass(frozen=True)
class ConstraintObligation:
    """A mandatory topology obligation derived from entity affordance tags.

    Each ConstraintObligation encodes a structural requirement that the
    assembled DAG MUST satisfy.  The obligation is derived purely from the
    affordance tags of entities in the goal state — NOT from semantic intent
    labels extracted from instruction text.

    Academic note (Section 4.2, AC-DAG Synthesis):
        This is the output of the Constraint Closure (CC) phase.  The CC phase
        is the core contribution of the capability constraint layer: it replaces
        task-type recognition with entity-affordance-tag-driven partial-order
        derivation.
    """

    # Structural role of this obligation
    obligation_type: str        # one of OBLIGATION_* constants above

    # Provenance: which entity and tag triggered this rule
    trigger_entity: str | None  # entity whose tag fired the rule
    trigger_tag: str            # the affordance tag that fired

    # Rule identity for traceability
    rule_name: str

    # Operands (may differ from trigger_entity in multi-entity goals)
    entity: str | None = None   # item being operated on
    fixture: str | None = None  # fixture / station involved

    # Human-readable justification (included in VLM prompt context)
    description: str = ""

    # Predicate annotations consumed by AIG validators/repair.
    precondition_predicates: list[str] = field(default_factory=list)
    postcondition_predicates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tag-Triggered Constraint Rules
#
# Each rule fires when an entity meeting the role/tag conditions appears in
# the goal context.  Firing emits a ConstraintObligation that the graph
# assembler must respect when building the DAG.
#
# Rule ordering matters only for readability; all matching rules fire.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TagConstraintRule:
    """A single tag-triggered constraint derivation rule.

    A rule captures the logic: "IF an entity with tag T appears in role R of
    the goal state, THEN topology obligation O is mandatory."
    """

    name: str
    # Structural role the triggering entity plays in the goal.
    # One of: "destination", "activation_target", "primary_object"
    entity_role: str
    # Affordance tags that must be present on the entity (AND or OR per require_all_tags)
    trigger_tags: frozenset[str]
    require_all_tags: bool
    # What obligation to emit when the rule fires
    obligation_type: str
    # Additional guard conditions
    requires_tool_items: bool = False  # only fire when tool_items is non-empty
    description: str = ""

    def tag_condition_met(self, entity_name: str) -> bool:
        if self.require_all_tags:
            return all(entity_has_tag(entity_name, t) for t in self.trigger_tags)
        return any(entity_has_tag(entity_name, t) for t in self.trigger_tags)


TAG_CONSTRAINT_RULES: list[TagConstraintRule] = [
    # ------------------------------------------------------------------
    # Rule 1: Openable container as destination → open before place
    # Physical rationale: placing inside a closed container is physically
    # impossible.  The "openable" + "container" affordance tags on the
    # destination entity entail a mandatory open(container) precondition.
    # No "cabinet" string match or "retrieve" intent label is needed.
    # ------------------------------------------------------------------
    TagConstraintRule(
        name="openable_container_before_place",
        entity_role="destination",
        trigger_tags=frozenset({OPENABLE_TAG, CONTAINER_TAG}),
        require_all_tags=True,
        obligation_type=OBLIGATION_OPEN_BEFORE_PLACE,
        description=(
            "Destination entity has 'openable' + 'container' tags → "
            "open(destination) must precede place(item, destination).  "
            "Derived from affordance profile of destination entity."
        ),
    ),

    # ------------------------------------------------------------------
    # Rule 2: Wash station in activation context → process before relocate
    # Physical rationale: the "wash_station" tag marks an entity at which
    # an Interact action constitutes a cleaning step that precedes the
    # item's final placement.  Derived from the fixture's affordance tag,
    # NOT from the presence of "wash" in the instruction text.
    # ------------------------------------------------------------------
    TagConstraintRule(
        name="wash_station_process_before_final_relocate",
        entity_role="activation_target",
        trigger_tags=frozenset({WASH_STATION_TAG}),
        require_all_tags=False,
        obligation_type=OBLIGATION_PROCESS_AT_WASH_STATION,
        description=(
            "Activation target has 'wash_station' tag → "
            "interact(item, wash_station) must precede place(item, final_dest).  "
            "Derived from entity's wash_station affordance, not from 'wash' keyword."
        ),
    ),

    # ------------------------------------------------------------------
    # Rule 3: Cutting surface as destination + tool items → place before tool
    # Physical rationale: slicing/cutting requires the food item to be
    # positioned on the cutting surface first; the tool Interact happens
    # after placement.  Derived from "cutting_surface" tag — no
    # "tool_interaction" intent label is required.
    # ------------------------------------------------------------------
    TagConstraintRule(
        name="cutting_surface_place_before_tool_interact",
        entity_role="destination",
        trigger_tags=frozenset({CUTTING_SURFACE_TAG}),
        require_all_tags=False,
        obligation_type=OBLIGATION_PLACE_BEFORE_TOOL_INTERACT,
        requires_tool_items=True,
        description=(
            "Destination has 'cutting_surface' tag AND tool_items non-empty → "
            "place(item, cutting_surface) precedes interact(tool, item).  "
            "Derived from entity affordance, not from 'cut'/'tool_interaction' intent."
        ),
    ),

    # ------------------------------------------------------------------
    # Rule 4: Activatable (non-wash) fixture in goal → activate after load
    # Physical rationale: devices like toasters and ovens are "activatable"
    # — triggering them is the final goal step after their contents are
    # loaded.  Derived from the "activatable" tag.
    # Note: "wash_station" entities are also "activatable" but are
    # handled exclusively by Rule 2, which fires first.
    # ------------------------------------------------------------------
    TagConstraintRule(
        name="activatable_fixture_activate_after_load",
        entity_role="activation_target",
        trigger_tags=frozenset({ACTIVATABLE_TAG}),
        require_all_tags=False,
        obligation_type=OBLIGATION_ACTIVATE_FIXTURE_AFTER_PLACE,
        description=(
            "Activation target has 'activatable' tag (non-wash-station) → "
            "activate(fixture) must follow place(item, fixture).  "
            "Derived from entity's activatable affordance, not from 'activate' intent label."
        ),
    ),

    # ------------------------------------------------------------------
    # Rule 5: Pushable carrier as primary object → push-transport handoff
    # Physical rationale: a "pushable_carrier" (e.g., cardboardbox) cannot
    # be grasped by robots that only have push capability (anymal_c,
    # unitree_go2).  When the primary object carries this tag, the topology
    # must include a push_transport node followed by a handoff to a
    # manipulation-capable robot — a capability gap forces a HandoffNode.
    # ------------------------------------------------------------------
    TagConstraintRule(
        name="pushable_carrier_requires_push_handoff",
        entity_role="primary_object",
        trigger_tags=frozenset({PUSHABLE_CARRIER_TAG}),
        require_all_tags=False,
        obligation_type=OBLIGATION_PUSH_CARRIER_HANDOFF,
        description=(
            "Primary object has 'pushable_carrier' tag → "
            "push_transport(carrier, handoff) required before grasp-and-place.  "
            "Derived from entity's pushable_carrier affordance and inventory capability gap."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Constraint Closure — Phase 1 of AC-DAG Synthesis
# ---------------------------------------------------------------------------

def compute_constraint_obligations(
    scene_facts: "SceneFacts",
) -> list[ConstraintObligation]:
    """Apply Tag-Triggered Constraint Rules to derive mandatory topology obligations.

    This is Phase 1 (Constraint Closure) of Affordance-Constrained DAG Synthesis.

    Replaces semantic intent detection.  The obligations are derived entirely
    from the affordance tags of entities appearing in the goal state, making
    topology derivation invariant to instruction phrasing and robust to
    out-of-distribution task types that use familiar entity types.

    Args:
        scene_facts: Structured scene representation from the Scene Parser /
                     Scene Facts Refiner pipeline stages.

    Returns:
        A list of ConstraintObligation objects encoding mandatory structural
        requirements for the assembled DAG.  The list is empty when no
        special affordance-tag constraints are triggered, corresponding to
        simple single- or multi-object relocation topologies.
    """
    from nsdl.task_family import _extract_destination_pairs as _extract_pairs  # local import to break circular

    obligations: list[ConstraintObligation] = []
    destination_pairs = _extract_pairs(scene_facts)

    # -----------------------------------------------------------------------
    # Gather destination entity set (for Rules 1, 3)
    # -----------------------------------------------------------------------
    destinations: list[str] = []
    if scene_facts.destination:
        dest_norm = normalize_entity_name(scene_facts.destination)
        if dest_norm:
            destinations.append(dest_norm)
    for pair in destination_pairs:
        target_norm = normalize_entity_name(pair.get("target"))
        if target_norm and target_norm not in destinations:
            destinations.append(target_norm)

    # -----------------------------------------------------------------------
    # Apply destination-role rules (Rules 1, 3)
    # -----------------------------------------------------------------------
    for dest in destinations:
        # Rule 1: openable + container → open before place
        if entity_has_tag(dest, OPENABLE_TAG) and entity_has_tag(dest, CONTAINER_TAG):
            obligations.append(
                ConstraintObligation(
                    obligation_type=OBLIGATION_OPEN_BEFORE_PLACE,
                    trigger_entity=dest,
                    trigger_tag=OPENABLE_TAG,
                    rule_name="openable_container_before_place",
                    fixture=dest,
                    description=f"'{dest}' is openable+container → open must precede place",
                    precondition_predicates=[f"is_open({dest})"],
                )
            )

        # Rule 3: cutting_surface + tool_items → place before tool interact
        if entity_has_tag(dest, CUTTING_SURFACE_TAG) and scene_facts.tool_items:
            obligations.append(
                ConstraintObligation(
                    obligation_type=OBLIGATION_PLACE_BEFORE_TOOL_INTERACT,
                    trigger_entity=dest,
                    trigger_tag=CUTTING_SURFACE_TAG,
                    rule_name="cutting_surface_place_before_tool_interact",
                    fixture=dest,
                    entity=(scene_facts.tool_items[0] if scene_facts.tool_items else None),
                    description=(
                        f"'{dest}' has cutting_surface tag + tool_items={scene_facts.tool_items} "
                        "→ place precedes tool interact"
                    ),
                    precondition_predicates=[f"item_at(<item>, {dest})"],
                    postcondition_predicates=[f"tool_effect_applied(<tool>, {dest})"],
                )
            )

    # -----------------------------------------------------------------------
    # Apply activation-target-role rules (Rules 2, 4)
    # wash_station entities are handled by Rule 2 ONLY (Rule 4 is skipped for them)
    # -----------------------------------------------------------------------
    for raw_at in scene_facts.activation_targets:
        at = normalize_entity_name(raw_at)
        if not at:
            continue

        if entity_has_tag(at, WASH_STATION_TAG):
            # Rule 2: wash station takes exclusive priority
            obligations.append(
                ConstraintObligation(
                    obligation_type=OBLIGATION_PROCESS_AT_WASH_STATION,
                    trigger_entity=at,
                    trigger_tag=WASH_STATION_TAG,
                    rule_name="wash_station_process_before_final_relocate",
                    fixture=at,
                    entity=scene_facts.primary_object,
                    description=(
                        f"'{at}' has wash_station tag → interact at station precedes final placement"
                    ),
                    precondition_predicates=[f"item_at(<item>, {at})"],
                    postcondition_predicates=["clean(<item>)"],
                )
            )
        elif entity_has_tag(at, ACTIVATABLE_TAG):
            # Rule 4: generic activatable fixture (toaster, oven, etc.)
            obligations.append(
                ConstraintObligation(
                    obligation_type=OBLIGATION_ACTIVATE_FIXTURE_AFTER_PLACE,
                    trigger_entity=at,
                    trigger_tag=ACTIVATABLE_TAG,
                    rule_name="activatable_fixture_activate_after_load",
                    fixture=at,
                    description=f"'{at}' has activatable tag → activate after load/place",
                    precondition_predicates=[f"loaded({at})"],
                    postcondition_predicates=[f"activated({at})"],
                )
            )

    # -----------------------------------------------------------------------
    # Apply primary-object-role rule (Rule 5)
    # -----------------------------------------------------------------------
    primary_obj = (
        normalize_entity_name(scene_facts.primary_object)
        if scene_facts.primary_object
        else None
    )
    if primary_obj and entity_has_tag(primary_obj, PUSHABLE_CARRIER_TAG):
        obligations.append(
            ConstraintObligation(
                obligation_type=OBLIGATION_PUSH_CARRIER_HANDOFF,
                trigger_entity=primary_obj,
                trigger_tag=PUSHABLE_CARRIER_TAG,
                rule_name="pushable_carrier_requires_push_handoff",
                entity=primary_obj,
                description=(
                    f"'{primary_obj}' has pushable_carrier tag → push transport + handoff topology"
                ),
                precondition_predicates=[f"is_pushable({primary_obj})"],
                postcondition_predicates=[f"carrier_at({primary_obj}, <handoff_target>)"],
            )
        )

    return obligations


# ---------------------------------------------------------------------------
# Convenience query helpers
# ---------------------------------------------------------------------------

def has_obligation(
    obligations: list[ConstraintObligation],
    obligation_type: str,
) -> bool:
    """Return True iff at least one obligation of the given type is present."""
    return any(ob.obligation_type == obligation_type for ob in obligations)


def get_obligations_of_type(
    obligations: list[ConstraintObligation],
    obligation_type: str,
) -> list[ConstraintObligation]:
    """Return all obligations of the given type."""
    return [ob for ob in obligations if ob.obligation_type == obligation_type]


def obligations_to_summary(
    obligations: list[ConstraintObligation],
) -> list[dict[str, object]]:
    """Serialise obligations for inclusion in VLM prompt context."""
    return [
        {
            "obligation_type": ob.obligation_type,
            "trigger_entity": ob.trigger_entity,
            "trigger_tag": ob.trigger_tag,
            "rule_name": ob.rule_name,
            "fixture": ob.fixture,
            "entity": ob.entity,
            "description": ob.description,
            "precondition_predicates": list(ob.precondition_predicates),
            "postcondition_predicates": list(ob.postcondition_predicates),
        }
        for ob in obligations
    ]
