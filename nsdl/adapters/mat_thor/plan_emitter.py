from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nsdl.adapters.mat_thor.catalog_map import resolve_thor_object
from nsdl.adapters.mat_thor.op_action_map import ThorAction
from nsdl.schema import DispatcherPayload, MicroStep, SceneFacts


_OFF_HINT = re.compile(r"\b(?:off|shut off|turn down|power down|deactivate|night|secure)\b")


@dataclass(frozen=True)
class MatThorPlanEmitter:
    """Emit MAT-THOR plan.json from dispatcher micro steps."""

    def emit(self, payload: DispatcherPayload | Sequence[MicroStep]) -> dict[str, Any]:
        if isinstance(payload, DispatcherPayload):
            return self.emit_payload(payload)
        return self.emit_steps_without_context(payload)

    def emit_payload(self, payload: DispatcherPayload) -> dict[str, Any]:
        sample = payload.sample
        scene_facts = payload.scene_facts
        extras = scene_facts.extras
        robots = _plan_robots(scene_facts)
        slot_to_agent = {
            robot["slot_id"]: int(robot["agent_id"])
            for robot in robots
            if robot.get("slot_id")
        }
        agent_plans: dict[str, list[dict[str, Any]]] = {
            str(robot["agent_id"]): []
            for robot in robots
        }
        thor_objects = _thor_objects(scene_facts)
        step_local_counts: dict[int, int] = defaultdict(int)
        skipped_micro_actions: list[dict[str, Any]] = []

        for micro_step in payload.micro_steps:
            for slot_id, micro_action in micro_step.actions.items():
                agent_id = slot_to_agent.get(slot_id)
                if agent_id is None:
                    skipped_micro_actions.append({
                        "slot_id": slot_id,
                        "reason": "slot has no MAT-THOR agent id",
                        "micro_action": list(micro_action),
                    })
                    continue
                thor_actions = micro_action_to_thor_actions(
                    micro_action,
                    scene_facts=scene_facts,
                )
                if not thor_actions:
                    continue
                queue = agent_plans.setdefault(str(agent_id), [])
                for thor_action in thor_actions:
                    local_index = step_local_counts[micro_step.step]
                    step_local_counts[micro_step.step] += 1
                    queue.append(
                        thor_action.to_plan_step(
                            step_id=f"n{micro_step.step}.{local_index}",
                            thor_objects=thor_objects,
                        )
                    )

        floor_plan = int(extras.get("floor_plan") or 0)
        return {
            "task_id": sample.task_id or extras.get("task_id") or f"FloorPlan{floor_plan}#0",
            "floor_plan": floor_plan,
            "robots": [
                {"agent_id": robot["agent_id"], "robot_id": robot["robot_id"]}
                for robot in robots
            ],
            "agent_plans": agent_plans,
            "barriers": [],
            "metadata": {
                "benchmark": "mat_thor",
                "complexity": extras.get("complexity"),
                "source": "nsdl.adapters.mat_thor.plan_emitter",
                "n_micro_steps": len(payload.micro_steps),
                "n_graph_nodes": len(payload.graph_plan.nodes),
                "skipped_micro_actions": skipped_micro_actions,
                "builder_trace": payload.graph_plan.builder_trace or {},
            },
        }

    def emit_steps_without_context(self, steps: Sequence[MicroStep]) -> dict[str, Any]:
        agent_plans: dict[str, list[dict[str, Any]]] = {"0": []}
        for micro_step in steps:
            for _, micro_action in micro_step.actions.items():
                for thor_action in micro_action_to_thor_actions(micro_action):
                    agent_plans["0"].append(
                        thor_action.to_plan_step(
                            step_id=f"n{micro_step.step}.0",
                            thor_objects=[],
                        )
                    )
        return {
            "task_id": "FloorPlan0#0",
            "floor_plan": 0,
            "robots": [{"agent_id": 0, "robot_id": 1}],
            "agent_plans": agent_plans,
            "barriers": [],
            "metadata": {
                "benchmark": "mat_thor",
                "source": "nsdl.adapters.mat_thor.plan_emitter",
                "context_missing": True,
            },
        }

    def write(self, payload: DispatcherPayload, output_path: Path | str) -> dict[str, Any]:
        plan = self.emit_payload(payload)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2, sort_keys=True))
        return plan


def micro_action_to_thor_actions(
    micro_action: Sequence[str],
    *,
    scene_facts: SceneFacts | None = None,
) -> list[ThorAction]:
    if not micro_action:
        return []
    action = str(micro_action[0])
    target = str(micro_action[1]) if len(micro_action) >= 2 else None
    if action == "Move":
        return [ThorAction("GoToObject", target)]
    if action == "Reach":
        return []
    if action == "Grasp":
        return [ThorAction("PickupObject", target)]
    if action == "Place":
        return [ThorAction("PutObject", target)]
    if action == "Open":
        return [ThorAction("OpenObject", target)]
    if action == "Close":
        return [ThorAction("CloseObject", target)]
    if action == "Push":
        return [ThorAction("PushObject", target)]
    if action == "Interact":
        return [_interact_action(target, scene_facts=scene_facts)]
    return [ThorAction("Pass", target)]


def _interact_action(target: str | None, *, scene_facts: SceneFacts | None) -> ThorAction:
    thor_objects = _thor_objects(scene_facts) if scene_facts is not None else []
    resolved = resolve_thor_object(target, thor_objects)
    tags = set()
    object_type = ""
    if resolved is not None:
        tags = set(resolved.raw.get("tags", []) or [])
        object_type = resolved.object_type.lower()
    instruction = (scene_facts.instruction if scene_facts is not None else "").lower()
    if "wash_station" in tags or any(token in object_type for token in ("sink", "faucet", "basin", "bath")):
        return ThorAction("CleanObject", target)
    if "sliceable" in tags or "knife" in object_type:
        return ThorAction("SliceObject", target)
    if "toggleable" in tags or "activatable" in tags:
        return ThorAction("SwitchOff" if _OFF_HINT.search(instruction) else "SwitchOn", target)
    if "openable" in tags:
        return ThorAction("OpenObject", target)
    return ThorAction("Pass", target)


def _plan_robots(scene_facts: SceneFacts) -> list[dict[str, Any]]:
    raw_robots = [
        robot for robot in scene_facts.extras.get("robots", []) or []
        if isinstance(robot, dict)
    ]
    robots: list[dict[str, Any]] = []
    for agent_id, robot in enumerate(raw_robots):
        try:
            robot_id = int(robot.get("robot_id"))
        except (TypeError, ValueError):
            robot_id = _robot_id_from_type(str(robot.get("robot_type") or "")) or agent_id + 1
        robots.append({
            "agent_id": agent_id,
            "robot_id": robot_id,
            "slot_id": str(robot.get("slot_id") or f"R{agent_id + 1}"),
        })
    if robots:
        return robots

    for agent_id, robot in enumerate(scene_facts.robot_inventory):
        robot_id = _robot_id_from_type(robot.robot_type) or agent_id + 1
        robots.append({
            "agent_id": agent_id,
            "robot_id": robot_id,
            "slot_id": robot.slot_id,
        })
    return robots


def _robot_id_from_type(robot_type: str) -> int | None:
    match = re.fullmatch(r"robot(\d+)", robot_type.strip())
    return int(match.group(1)) if match else None


def _thor_objects(scene_facts: SceneFacts | None) -> list[Mapping[str, Any]]:
    if scene_facts is None:
        return []
    return [
        obj for obj in scene_facts.extras.get("thor_objects", []) or []
        if isinstance(obj, Mapping)
    ]


MAT_THOR_PLAN_EMITTER = MatThorPlanEmitter()


__all__ = [
    "MAT_THOR_PLAN_EMITTER",
    "MatThorPlanEmitter",
    "micro_action_to_thor_actions",
]

