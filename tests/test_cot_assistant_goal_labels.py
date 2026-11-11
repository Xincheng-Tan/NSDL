#!/usr/bin/env python3
"""Regression tests for offline [GOAL] labels.

Run:
    python tests/test_cot_assistant_goal_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.schema import OperationStep, RobotInstance, SceneFacts  # noqa: E402
from scripts.cot_assistant_serializer import serialize_cot_assistant  # noqa: E402


def _scene(destination: str | None, pairs: list[dict[str, str]]) -> SceneFacts:
    return SceneFacts(
        parser_backend="test",
        instruction="",
        image_available=False,
        robot_inventory=[RobotInstance(slot_id="R1", robot_type="fetch")],
        primary_object=None,
        destination=destination,
        destination_pairs=pairs,
        target_items=[],
        extras={"scene_relations": ["none observed"]},
    )


def _goal(scene: SceneFacts, ops: list[OperationStep | dict]) -> dict:
    text = serialize_cot_assistant(scene, ops)
    lines = text.splitlines()
    return json.loads(lines[lines.index("[GOAL]") + 1])


def test_tool_only_entity_is_not_goal_target() -> None:
    scene = _scene(
        "wooden cutting board",
        [
            {"item": "apple", "target": "wooden cutting board"},
            {"item": "knife", "target": "wooden cutting board"},
        ],
    )
    ops = [
        OperationStep("op1", "acquire", "knife", None, [], "R1"),
        OperationStep("op2", "carry_to", "knife", "wooden cutting board", ["op1"], "R1"),
        OperationStep("op3", "acquire", "apple", None, [], "R1"),
        OperationStep("op4", "deposit", "apple", "wooden cutting board", ["op3"], "R1"),
        OperationStep(
            "op5",
            "apply_tool",
            "knife",
            "wooden cutting board",
            ["op2", "op4"],
            "R1",
            {"tool": "knife", "effect": "cut"},
        ),
    ]

    assert _goal(scene, ops) == {
        "targets": [{
            "item": "apple",
            "destination": "wooden cutting board",
            "status": "active",
            "status_reason": "needs_relocation",
            "status_line": "apple: active",
        }],
        "active": ["apple"],
        "skip": [],
    }


def test_gt_active_overrides_source_container_relation() -> None:
    scene = _scene(
        "kitchen work area",
        [
            {"item": "bowl", "target": "kitchen work area"},
            {"item": "tomato", "target": "cabinet"},
        ],
    )
    scene.extras["scene_relations"] = ["tomato in cabinet"]
    ops = [
        OperationStep("op1", "acquire", "bowl", None, [], "R1"),
        OperationStep("op2", "deposit", "bowl", "kitchen work area", ["op1"], "R1"),
        OperationStep("op3", "open", None, "cabinet", [], "R1"),
        OperationStep("op4", "acquire", "tomato", None, ["op3"], "R1"),
        OperationStep("op5", "deposit", "tomato", "bowl", ["op4"], "R1"),
    ]

    assert _goal(scene, ops) == {
        "targets": [
            {
                "item": "bowl",
                "destination": "kitchen work area",
                "status": "active",
                "status_reason": "needs_relocation",
                "status_line": "bowl: active",
            },
            {
                "item": "tomato",
                "destination": "bowl",
                "status": "active",
                "status_reason": "needs_relocation",
                "status_line": "tomato: active",
            },
        ],
        "active": ["bowl", "tomato"],
        "skip": [],
    }


def test_push_carrier_is_route_machinery_not_goal_target() -> None:
    scene = _scene(
        "sink",
        [
            {"item": "pumpkin", "target": "sink"},
            {"item": "pumpkin", "target": "cardboardbox"},
            {"item": "cardboardbox", "target": "sink"},
        ],
    )
    ops = [
        OperationStep("op1", "local_acquire", "pumpkin", None, [], "R1"),
        OperationStep("op2", "push_carrier", "cardboardbox", "R1", [], "R1", {"target_robot": "R1"}),
        OperationStep("op3", "load_carrier", "pumpkin", "cardboardbox", ["op1", "op2"], "R1", {"carrier": "cardboardbox"}),
        OperationStep("op4", "push_carrier", "cardboardbox", "R2", ["op3"], "R1", {"target_robot": "R2"}),
        OperationStep("op5", "unload_carrier", "pumpkin", "sink", ["op4"], "R1", {"carrier": "cardboardbox"}),
    ]

    assert _goal(scene, ops) == {
        "targets": [{
            "item": "pumpkin",
            "destination": "sink",
            "status": "active",
            "status_reason": "needs_relocation",
            "status_line": "pumpkin: active",
        }],
        "active": ["pumpkin"],
        "skip": [],
    }


def test_unhandled_destination_pair_becomes_skip_label() -> None:
    scene = _scene(
        "table",
        [
            {"item": "bowl", "target": "table"},
            {"item": "plate", "target": "table"},
        ],
    )
    ops = [
        {"op_id": "op1", "op_type": "acquire", "entity": "bowl", "dest": None, "deps": [], "robot": "R1"},
        {"op_id": "op2", "op_type": "deposit", "entity": "bowl", "dest": "table", "deps": ["op1"], "robot": "R1"},
    ]

    assert _goal(scene, ops) == {
        "targets": [
            {
                "item": "bowl",
                "destination": "table",
                "status": "active",
                "status_reason": "needs_relocation",
                "status_line": "bowl: active",
            },
            {
                "item": "plate",
                "destination": "table",
                "status": "skip",
                "status_reason": "observed_on_table",
                "status_line": "plate: skip(observed_on_table)",
            },
        ],
        "active": ["bowl"],
        "skip": ["plate"],
    }


if __name__ == "__main__":
    test_tool_only_entity_is_not_goal_target()
    test_gt_active_overrides_source_container_relation()
    test_push_carrier_is_route_machinery_not_goal_target()
    test_unhandled_destination_pair_becomes_skip_label()
    print("ok")
