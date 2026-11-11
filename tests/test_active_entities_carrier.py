"""Regression checks for the facet-A carrier metric bug (LOCUS F9).

``_active_entities`` decides which entities a plan *relocates*; the first-GOAL
and final-plan facet-A rates are set equalities against it. It used to count
``Grasp`` only, so LOCUS F9's ``Push cardboardbox sink`` -- where the box itself
carries a goal constraint -- scored as "nothing active" and the family was
counted wrong no matter what the model said.

The naive fix (count every ``Push``) is also wrong: VIKI's push family is a
handoff, ``Push cardboardbox R2``, where the box is a vehicle and the goal
constraint names only the payload. Its supervision lists the payload alone.
The discriminator is the third argument: a place means relocation, a robot slot
means handoff.

Run with:
    python tests/test_active_entities_carrier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from summarize_eval_results import _active_entities  # noqa: E402


# LOCUS F9 sub-case 2: goal_constraints are [meat@cardboardbox, cardboardbox@sink],
# so the box is itself active.
LOCUS_PUSH_TO_PLACE = [
    {"actions": {"R1": ["Grasp", "meat"], "R2": ["Reach", "cardboardbox"]}},
    {"actions": {"R1": ["Place", "meat", "cardboardbox"], "R2": ["Idle"]}},
    {"actions": {"R1": ["Idle"], "R2": ["Push", "cardboardbox", "sink"]}},
]

# VIKI dog_push_box_for_two_panda_transport: the box is handed between robots
# and only the payload is a goal item.
VIKI_PUSH_TO_ROBOT = [
    {"actions": {"R1": ["Grasp", "pear"], "R3": ["Reach", "cardboardbox"]}},
    {"actions": {"R1": ["Place", "pear", "cardboardbox"], "R3": ["Idle"]}},
    {"actions": {"R1": ["Idle"], "R3": ["Push", "cardboardbox", "R2"]}},
]


def test_push_to_place_counts_the_carrier() -> None:
    assert _active_entities(LOCUS_PUSH_TO_PLACE) == {"meat", "cardboardbox"}


def test_push_to_robot_does_not_count_the_carrier() -> None:
    assert _active_entities(VIKI_PUSH_TO_ROBOT) == {"pear"}


def test_robot_slot_is_recognised_even_when_it_never_acts() -> None:
    """R2 is the push target but has no actions of its own in these steps.

    The slot-id set alone would then miss it, which is why the naming pattern is
    checked as well.
    """
    steps = [{"actions": {"R1": ["Push", "cardboardbox", "R2"]}}]
    assert _active_entities(steps) == set()


def test_grasp_only_plans_are_unchanged() -> None:
    steps = [
        {"actions": {"R1": ["Grasp", "apple"], "R2": ["Grasp", "cup"]}},
        {"actions": {"R1": ["Place", "apple", "table"], "R2": ["Idle"]}},
    ]
    assert _active_entities(steps) == {"apple", "cup"}


def test_malformed_input_is_tolerated() -> None:
    assert _active_entities(None) == set()
    assert _active_entities([None, "junk", {"actions": None}]) == set()
    assert _active_entities([{"actions": {"R1": ["Push"], "R2": ["Push", "box"]}}]) == set()


def main() -> int:
    tests = [
        test_push_to_place_counts_the_carrier,
        test_push_to_robot_does_not_count_the_carrier,
        test_robot_slot_is_recognised_even_when_it_never_acts,
        test_grasp_only_plans_are_unchanged,
        test_malformed_input_is_tolerated,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
