"""Serialize the dataset-v2 three-block CoT assistant response.

Layout (verbatim, three blocks each followed by JSON):

    [GOAL]
    {"targets": [...], "active": [...], "skip": [...]}

[ASSIGN]
{"R1": {"items": [...]}, ...}

    [PLAN]
    [ {"op_id": "op1", ...}, ... ]

Derivation rules — see docs/dataset and train.md §A.7.

This module is the single source of truth for the current assistant text and
is imported by dataset builders and RL-data conversion scripts.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from nsdl.schema import OperationStep, RobotInstance, SceneFacts


_ACTIVE = "active"
_SKIP = "skip"
_ACTIVE_REASON = "needs_relocation"
_SKIP_DEFAULT_REASON = "observed_on_table"
_ACQUIRE_OPS = {"acquire", "local_acquire", "load_carrier"}
_FINAL_TARGET_OPS = {"deposit", "process_at", "unload_carrier"}


# ---------------------------------------------------------------------------
# [GOAL] derivation
# ---------------------------------------------------------------------------

def _append_unique(items: list[str], item: str | None) -> None:
    if item and item not in items:
        items.append(item)


def _pair_item(pair: dict[str, Any]) -> str | None:
    item = pair.get("item")
    return str(item) if item else None


def _pair_destination(pair: dict[str, Any]) -> str | None:
    dest = pair.get("target") or pair.get("destination")
    return str(dest) if dest else None


def _status_line(item: str, status: str, reason: str) -> str:
    if status == _ACTIVE:
        return f"{item}: active"
    return f"{item}: skip({reason})"


def _skip_reason(destination: str | None) -> str:
    if destination:
        normalized = str(destination).strip().lower().replace(" ", "_")
        if normalized:
            return f"observed_on_{normalized}"
    return _SKIP_DEFAULT_REASON


def _tool_use_entities(ops: list[OperationStep]) -> set[str]:
    tools: set[str] = set()
    for op in ops:
        if op.op_type != "apply_tool":
            continue
        if op.entity:
            tools.add(str(op.entity))
        tool_name = (op.params or {}).get("tool")
        if tool_name:
            tools.add(str(tool_name))
    return tools


def _carrier_use_entities(ops: list[OperationStep]) -> set[str]:
    carriers: set[str] = set()
    for op in ops:
        if op.op_type == "push_carrier" and op.entity:
            carriers.add(str(op.entity))
        carrier_name = (op.params or {}).get("carrier")
        if carrier_name:
            carriers.add(str(carrier_name))
    return carriers


def _final_destination_by_item(ops: list[OperationStep]) -> dict[str, str | None]:
    final_dest: dict[str, str | None] = {}
    for op in ops:
        if op.op_type in _FINAL_TARGET_OPS and op.entity:
            final_dest[str(op.entity)] = str(op.dest) if op.dest else None

    # A load into a carrier is a final goal only when the item is not later
    # deposited/processed/unloaded. In VIKI-L2 push-transfer traces it is
    # normally intermediate; in synthetic "put X into box" traces it can be final.
    for op in ops:
        if op.op_type == "load_carrier" and op.entity and str(op.entity) not in final_dest:
            final_dest[str(op.entity)] = str(op.dest) if op.dest else None
    return final_dest


def _is_goal_item(
    item: str,
    *,
    final_dest: dict[str, str | None],
    tool_use: set[str],
    carrier_use: set[str],
) -> bool:
    # A knife carried to a cutting board solely for apply_tool is an instrument,
    # not a [GOAL] target. If that same knife is finally deposited somewhere,
    # final_dest keeps it eligible as a real relocation target.
    if item in tool_use and item not in final_dest:
        return False
    # The pushable carrier is route machinery, not the object requested by the
    # task, unless the plan explicitly places the carrier as the final object.
    if item in carrier_use and item not in final_dest:
        return False
    return True


def _build_goal_block(scene_facts: SceneFacts, ops: list[OperationStep]) -> dict[str, Any]:
    """Derive offline [GOAL] supervision from GT ops.

    Runtime target_status is intentionally disabled. For training labels, the
    GT operation DAG is the reliable source for active/skip: if an item is
    actually handled by the GT plan, it is active; parser or perception evidence
    must not mark it skipped. Destination pairs only add non-handled targets,
    which become skip labels with an observation-grounded reason token.
    """
    dest_map: dict[str, str | None] = {}
    target_items: list[str] = []
    active_items: list[str] = []

    final_dest = _final_destination_by_item(ops)
    tool_use = _tool_use_entities(ops)
    carrier_use = _carrier_use_entities(ops)

    # destination_pairs uses key "target" not "destination" (see scene_encoder.py).
    # Keep its order as the target ordering backbone, but do not trust it over
    # GT ops for active status or final destinations.
    for pair in scene_facts.destination_pairs or []:
        if not isinstance(pair, dict):
            continue
        item_name = _pair_item(pair)
        if not item_name or not _is_goal_item(
            item_name,
            final_dest=final_dest,
            tool_use=tool_use,
            carrier_use=carrier_use,
        ):
            continue
        _append_unique(target_items, item_name)
        dest_map.setdefault(item_name, _pair_destination(pair))

    for op in ops:
        if op.op_type not in _ACQUIRE_OPS and op.op_type not in _FINAL_TARGET_OPS:
            continue
        if not op.entity:
            continue
        item_name = str(op.entity)
        if not _is_goal_item(
            item_name,
            final_dest=final_dest,
            tool_use=tool_use,
            carrier_use=carrier_use,
        ):
            continue
        if op.op_type in _ACQUIRE_OPS:
            _append_unique(active_items, item_name)
        _append_unique(target_items, item_name)

    # GT final destinations fix parser source/destination confusions such as
    # "retrieve tomato from cabinet and place it into the bowl".
    for item_name, final in final_dest.items():
        if _is_goal_item(
            item_name,
            final_dest=final_dest,
            tool_use=tool_use,
            carrier_use=carrier_use,
        ):
            dest_map[item_name] = final

    fallback_dest = scene_facts.destination
    active_set = set(active_items)

    targets: list[dict[str, Any]] = []
    active: list[str] = []
    skip: list[str] = []
    for item in target_items:
        dest = dest_map.get(item) or fallback_dest
        status = _ACTIVE if item in active_set else _SKIP
        reason = _ACTIVE_REASON if status == _ACTIVE else _skip_reason(dest)
        targets.append({
            "item": item,
            "destination": dest,
            "status": status,
            "status_reason": reason,
            "status_line": _status_line(item, status, reason),
        })
        if status == _SKIP:
            skip.append(item)
        else:
            active.append(item)

    return {"targets": targets, "active": active, "skip": skip}


def _build_assign_block(
    ops: list[OperationStep],
    robot_inventory: list[RobotInstance],
) -> dict[str, dict[str, Any]]:
    by_slot: dict[str, dict[str, Any]] = {}
    for robot in robot_inventory:
        by_slot[robot.slot_id] = {"_type": robot.robot_type, "_ops": [], "_items": []}

    for op in ops:
        slot = op.robot
        if not slot:
            continue
        # Decoy slots may exist in inventory but never be addressed; missing
        # slots in inventory still get an entry so SLM learns to emit them.
        bucket = by_slot.setdefault(slot, {"_type": None, "_ops": [], "_items": []})
        bucket["_ops"].append(op.op_type)
        if op.op_type in _ACQUIRE_OPS and op.entity:
            if op.entity not in bucket["_items"]:
                bucket["_items"].append(op.entity)

    out: dict[str, dict[str, Any]] = {}
    for slot, bucket in by_slot.items():
        items = bucket["_items"]
        # Dataset v2 drops the old role field; SLM only emits items per slot.
        out[slot] = {"items": items}
    return out


# ---------------------------------------------------------------------------
# [PLAN] serialization
# ---------------------------------------------------------------------------

def _ops_to_json(ops: list[OperationStep]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op in ops:
        record: dict[str, Any] = {
            "op_id": op.op_id,
            "op_type": op.op_type,
            "entity": op.entity,
            "dest": op.dest,
            "deps": list(op.deps or []),
            "robot": op.robot,
        }
        if op.params:
            record["params"] = dict(op.params)
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def serialize_cot_assistant(
    scene_facts: SceneFacts,
    ops: Iterable[OperationStep | dict[str, Any]],
    *,
    robot_inventory: list[RobotInstance] | None = None,
    plan_override: list[dict[str, Any]] | None = None,
) -> str:
    """Build the assistant text containing the three CoT blocks.

    ``ops`` may be a list of OperationStep dataclasses OR plain dicts (e.g.
    when loaded back from samples.jsonl). ``plan_override`` lets callers
    supply a ready-made JSON serialisation when they don't have OperationStep
    instances handy (e.g. carried over verbatim from VIKI reverse-mapping).
    """
    op_objects: list[OperationStep] = []
    for op in ops:
        if isinstance(op, OperationStep):
            op_objects.append(op)
        elif isinstance(op, dict):
            op_objects.append(
                OperationStep(
                    op_id=str(op.get("op_id")),
                    op_type=str(op.get("op_type")),
                    entity=op.get("entity"),
                    dest=op.get("dest"),
                    deps=list(op.get("deps") or []),
                    robot=op.get("robot"),
                    params=dict(op.get("params") or {}),
                )
            )
        else:
            raise TypeError(f"Unsupported op type: {type(op)!r}")

    inventory = robot_inventory if robot_inventory is not None else list(scene_facts.robot_inventory)

    goal = _build_goal_block(scene_facts, op_objects)
    assign = _build_assign_block(op_objects, inventory)
    plan = plan_override if plan_override is not None else _ops_to_json(op_objects)

    parts: list[str] = []
    parts.append("[GOAL]")
    parts.append(json.dumps(goal, ensure_ascii=False))
    parts.append("")
    parts.append("[ASSIGN]")
    parts.append(json.dumps(assign, ensure_ascii=False))
    parts.append("")
    parts.append("[PLAN]")
    parts.append(json.dumps(plan, ensure_ascii=False))
    return "\n".join(parts)


__all__ = ["serialize_cot_assistant"]
