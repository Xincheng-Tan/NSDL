#!/usr/bin/env python3
"""Build SLM training data from VIKI-L2 parquet files.

Usage:
    python scripts/build_slm_training_data.py \
        --parquet data/viki/VIKI-L2/task_coverage_debug.parquet \
        --split debug \
        --output_dir data/v2/slm_dataset \
        --max_samples 200

For each sample:
  1. Extract instruction, robots, time_steps, goal_constraints from GT
  2. Build SceneFacts (rule-based parser) + AffordanceContext (catalog query)
  3. Reverse-map GT time_steps → 14-op-type Operation DAG (v3 vocab)
  4. Serialize system_prompt + user_prompt
  5. Round-trip verify: OpDAG → GraphPlan → dispatch → compare with GT
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

# -- project imports --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nsdl.catalog import (
    get_robot_capabilities,
    get_end_effector_num,
    normalize_entity_name,
    entity_has_tag,
    get_entity_tags,
    normalize_robot_name,
)
from nsdl.schema import (
    BenchmarkSample,
    RobotInstance,
    SceneFacts,
    AffordanceContext,
    OperationStep,
    StructuredDetections,
)
from nsdl.scene_parser import build_scene_parser
from nsdl.aig_builder import build_affordance_context
from nsdl.slm_interface import (
    build_slm_system_prompt,
    serialize_slm_input,
    convert_operations_to_graph_plan,
)
from nsdl.dispatcher import dispatch_graph
from nsdl.yolo_detector import build_structured_detector
from nsdl.scene_facts_refiner import (
    build_scene_facts_refiner,
)

# =====================================================================
# Constants
# =====================================================================
MOBILE_ROBOTS = {"fetch", "stompy", "unitree_h1", "unitree_go2", "anymal_c"}
STATIONARY_ROBOTS = {"panda"}  # no mobility
DUAL_EE_ROBOTS = {"stompy", "unitree_h1"}
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v2")

# Polarity keyword detection (instruction-level, case-insensitive).
# Used both for GT polarity inference and for activate-off augmentation.
import re as _re
_POLARITY_OFF_PATTERNS = [
    (_re.compile(r"\bswitch(?:es|ed|ing)?\s+(?:it\s+|the\s+\w+\s+|them\s+)?off\b", _re.I), "switch_off"),
    (_re.compile(r"\bturn(?:s|ed|ing)?\s+(?:it\s+|the\s+\w+\s+|them\s+)?off\b", _re.I), "turn_off"),
    (_re.compile(r"\bshut(?:s|down|ting)?\s+(?:it\s+|the\s+\w+\s+|them\s+)?(?:off|down)\b", _re.I), "shut_off"),
    (_re.compile(r"\bpower(?:s|ed|ing)?\s+(?:it\s+|the\s+\w+\s+|them\s+)?(?:off|down)\b", _re.I), "power_off"),
    (_re.compile(r"\bdeactivate(?:s|d|ing)?\b", _re.I), "deactivate"),
    (_re.compile(r"\bstop(?:s|ped|ping)?\s+(?:its?\s+operation|the\s+\w+|toasting|cooking)\b", _re.I), "stop_op"),
    (_re.compile(r"\bhalt(?:s|ed|ing)?\s+(?:its?\s+operation|the\s+\w+)\b", _re.I), "halt_op"),
]


def detect_instruction_polarity(instruction: str) -> str:
    """Return 'off' if instruction contains an explicit deactivation cue, else 'on'.

    VIKI-L2 instructions are uniformly 'on' in distribution; this hook exists
    so that augmented samples (and any future MAT-THOR-style cross-benchmark
    training data) can flow through the same reverse-mapping logic.
    """
    if not instruction:
        return "on"
    for pat, _ in _POLARITY_OFF_PATTERNS:
        if pat.search(instruction):
            return "off"
    return "on"


# Instruction-level on→off paraphrase rewrites for activate-off augmentation.
# Each entry is (compiled_pattern, replacement). Patterns are ordered by
# specificity so longer phrases are tried before shorter ones. The substitution
# is purely lexical; semantics outside the activate fixture are untouched
# (placement/cooking phrasing kept identical so the GT atomic actions still
# match the dispatcher round-trip).
_POLARITY_ON_REWRITES = [
    (_re.compile(r"switch\s+it\s+on", _re.I), "switch it off"),
    (_re.compile(r"switches\s+it\s+on", _re.I), "switches it off"),
    (_re.compile(r"turn\s+it\s+on", _re.I), "turn it off"),
    (_re.compile(r"turns\s+it\s+on", _re.I), "turns it off"),
    (_re.compile(r"turn\s+on\s+the\s+device", _re.I), "turn off the device"),
    (_re.compile(r"power\s+(?:it\s+)?on", _re.I), "power it off"),
    (_re.compile(r"start\s+its\s+operation", _re.I), "halt its operation"),
    (_re.compile(r"start\s+the\s+process", _re.I), "stop the process"),
    (_re.compile(r"begin\s+(?:its?\s+operation|the\s+process)", _re.I), "stop its operation"),
    (_re.compile(r"activate\s+it\s+to\s+start\s+the\s+process", _re.I), "deactivate it to stop the process"),
    (_re.compile(r"\bactivate\s+it\b", _re.I), "deactivate it"),
    (_re.compile(r"\bactivate\s+the\s+(\w+)", _re.I), r"deactivate the \1"),
    (_re.compile(r"\bswitch\s+on\s+the\s+(\w+)", _re.I), r"switch off the \1"),
    (_re.compile(r"\bturn\s+on\s+the\s+(\w+)", _re.I), r"turn off the \1"),
]


def paraphrase_instruction_off(instruction: str) -> str | None:
    """Best-effort on→off paraphrase. Returns None if no on-cue is found.

    The rewrite is shallow on purpose: we only flip the activation verb so the
    rest of the sentence (placement targets, cooking nouns, etc.) keeps its
    original semantics and the reverse-mapped atomic GT remains valid.
    """
    if not instruction:
        return None
    out = instruction
    changed = False
    for pat, repl in _POLARITY_ON_REWRITES:
        new_out, n = pat.subn(repl, out)
        if n > 0:
            out = new_out
            changed = True
    if not changed:
        return None
    # Sanity: if our rewrite did not produce an off-cue we trust, abort.
    if detect_instruction_polarity(out) != "off":
        return None
    return out

# =====================================================================
# 1. Extract raw data from parquet row
# =====================================================================

def extract_sample_data(row) -> dict:
    """Extract structured data from a parquet row."""
    gt = row["reward_model"]["ground_truth"]
    # Extract instruction from prompt
    prompt_content = row["prompt"][1]["content"]
    # Remove <image> tag if present
    instruction = prompt_content.replace("<image>", "").strip()

    robots_raw = gt["robots"]  # {R1: type, R2: type|None, R3: type|None}
    time_steps = gt["time_steps"]  # [{step, actions: {R1:[Act,arg,...], ...}}]
    task_name = gt["task_name"]
    goal_constraints = gt.get("goal_constraints", [])
    task_id = gt.get("task_id", "")
    description = gt.get("description", instruction)

    # Build robot inventory (active robots only)
    robot_inventory = []
    for slot, rtype in sorted(robots_raw.items()):
        if rtype is not None:
            robot_inventory.append({"slot": slot, "type": rtype})

    return {
        "instruction": instruction,
        "description": description,
        "robot_inventory": robot_inventory,
        "time_steps": time_steps,
        "task_name": task_name,
        "task_id": task_id,
        "goal_constraints": goal_constraints,
        "robots_raw": robots_raw,
    }


# =====================================================================
# 2. Build SceneFacts from GT data
# =====================================================================

def _build_benchmark_sample(
    sample_data: dict,
    idx: int = 0,
    image_bytes: bytes = b"",
) -> BenchmarkSample:
    """Build BenchmarkSample from GT data + optional image bytes."""
    instruction = sample_data["instruction"]
    robot_inv_dict = {}
    for r in sample_data["robot_inventory"]:
        robot_inv_dict[r["slot"]] = r["type"]
    for slot in ["R1", "R2", "R3"]:
        if slot not in robot_inv_dict:
            robot_inv_dict[slot] = None
    return BenchmarkSample(
        level="2",
        split="train",
        sample_index=idx,
        task_id=sample_data.get("task_id", ""),
        task_name=sample_data.get("task_name", ""),
        instruction=instruction,
        system_prompt="",
        image_bytes=image_bytes,
        robot_inventory=robot_inv_dict,
    )


def build_scene_facts_from_gt(
    sample_data: dict,
    idx: int = 0,
    image_bytes: bytes = b"",
    detector=None,
    refiner=None,
) -> SceneFacts:
    """Build SceneFacts using rule-based parser, optionally enriched by YOLO + refiner.

    When a YOLO detector is provided, the refiner (NoOp by default) runs
    ``apply_deterministic_postprocessing`` which covers target_status
    derivation, robot_anchor_status inference, and storage-cue cleanup —
    all without any LLM call.
    """
    sample = _build_benchmark_sample(sample_data, idx, image_bytes)
    parser = build_scene_parser("rule")
    scene_facts = parser.parse(sample)

    if detector is None or not image_bytes:
        return scene_facts

    # ── YOLO detection ────────────────────────────────────────────
    structured_detections = detector.detect(image_bytes)

    # ── Refiner (NoOp by default) now includes all deterministic
    #    post-processing (target_status, robot_anchor_status, etc.) ──
    if refiner is not None:
        scene_facts = refiner.refine(
            sample, scene_facts, yolo_detections=structured_detections,
        )

    return scene_facts


# =====================================================================
# 3. GT Reverse Mapping: time_steps → Operation DAG
# =====================================================================

def _infer_process(station_tags: set[str]) -> str:
    """Infer process_at variant from station tags. Default: wash."""
    if "wash_station" in station_tags:
        return "wash"
    if station_tags & {"cookable_station", "oven", "microwave"}:
        return "cook"
    if "cleaning_station" in station_tags:
        return "clean"
    if "freezer" in station_tags:
        return "freeze"
    return "wash"


def _infer_effect(tool_name: str, target_tags: set[str]) -> str:
    """Infer apply_tool effect from tool / target tags. Default: cut."""
    sharp = tool_name in {"knife", "scissors"} or "sharp" in target_tags
    if sharp and "sliceable" in target_tags:
        return "slice"
    if "peelable" in target_tags:
        return "peel"
    return "cut"



def _looks_like_robot_slot(value: object) -> bool:
    """True for the "R<digits>" slot ids used in the benchmark GT."""
    text = str(value or "").strip()
    return len(text) >= 2 and text[0].lower() == "r" and text[1:].isdigit()


def reverse_map_time_steps(sample_data: dict) -> list[OperationStep]:
    """Reverse-map GT time_steps into 14-op-type Operation DAG (v3 vocab).

    Algorithm:
    1. Split time_steps into per-robot action sequences
    2. Identify operation boundaries using pattern matching
    3. Build deps from temporal ordering
    """
    time_steps = sample_data["time_steps"]
    robots_raw = sample_data["robots_raw"]
    task_name = sample_data["task_name"]

    # Get active robot slots and their types
    active_robots = {}
    for slot, rtype in sorted(robots_raw.items()):
        if rtype is not None:
            active_robots[slot] = normalize_robot_name(rtype) or rtype

    # Step 1: Extract per-robot action sequences with global step indices
    # Each entry: (global_step, action_name, arg1, arg2?)
    per_robot: dict[str, list[tuple]] = {slot: [] for slot in active_robots}
    for ts in time_steps:
        step_idx = ts["step"]
        actions = ts["actions"]
        for slot in active_robots:
            act = actions.get(slot)
            if act is not None:
                act_list = list(act)
                per_robot[slot].append((step_idx, act_list))

    # Pre-scan: identify push targets and carriers
    # push_targets[slot] = carrier_name means a Push(..., slot) exists
    push_targets: dict[str, str] = {}
    for ts in time_steps:
        for slot, act in ts["actions"].items():
            if act is not None and act[0] == "Push" and len(act) >= 3:
                carrier = act[1]
                target_slot = act[2]
                push_targets[target_slot] = carrier

    # Step 2: Identify operations for each robot
    ops: list[OperationStep] = []
    op_counter = [0]  # mutable counter

    # Track state per robot
    robot_state: dict[str, dict] = {}
    for slot in active_robots:
        robot_state[slot] = {
            "holding": [],      # items currently held
            "holding_tool": None,  # tool being held
        }

    # -- Helper to create next op_id --
    def next_op_id():
        op_counter[0] += 1
        return f"op{op_counter[0]}"

    # -- Determine if robot is stationary (panda) --
    def is_stationary(slot):
        return active_robots[slot] in STATIONARY_ROBOTS

    def is_dual_ee(slot):
        return active_robots[slot] in DUAL_EE_ROBOTS

    # -- Per-robot operation extraction --
    # This is the core reverse-mapping logic.
    # We process each robot's action sequence and identify operation boundaries.

    # Track which ops are produced per robot, for deps
    robot_ops: dict[str, list[str]] = {slot: [] for slot in active_robots}
    # Track op creation with global step for cross-robot deps
    op_step_range: dict[str, tuple[int, int]] = {}  # op_id -> (start_step, end_step)
    # Track carrier state for push pattern
    carrier_ops: dict[str, list[str]] = {}  # carrier_name -> [load_op, push_op1, ...]

    for slot in sorted(active_robots.keys()):
        actions = per_robot[slot]
        rtype = active_robots[slot]
        i = 0
        holding = []  # items currently held by this robot
        holding_tool = None

        while i < len(actions):
            step_idx, act = actions[i]
            act_name = act[0]
            act_arg = act[1] if len(act) > 1 else None
            act_arg2 = act[2] if len(act) > 2 else None

            # --- Push pattern ---
            if act_name == "Push":
                carrier = act_arg
                target_robot = act_arg2
                oid = next_op_id()
                # Push(carrier, X) has two shapes. In VIKI-L2 X is ALWAYS a
                # robot slot (368/368 pushes in train) -- the four-legged robot
                # pushes the carrier to a receiving arm, and params.target_robot
                # is what the validator checks. LOCUS F9-M2 also pushes a carrier
                # straight to a DESTINATION ANCHOR ("Push cardboardbox sink"),
                # which the catalog has always allowed ("toward a receiver or
                # destination anchor"). Writing a place name into
                # params.target_robot made the validator reject the GT itself.
                # Only tag it as a target_robot when it really is a slot; the
                # anchor case keeps it in `dest`, so VIKI stays byte-identical.
                push_params: dict = {}
                if _looks_like_robot_slot(target_robot):
                    push_params["target_robot"] = target_robot
                ops.append(OperationStep(
                    op_id=oid, op_type="push_carrier",
                    entity=carrier, dest=target_robot,
                    deps=[], robot=slot,
                    params=push_params,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, step_idx)
                carrier_ops.setdefault(carrier, []).append(oid)
                i += 1
                continue

            # --- Open pattern ---
            # Mobile: Move(C) → Reach(C) → Open(C)
            # Stationary: Reach(C) → Open(C)
            if act_name == "Move" and _peek_sequence(actions, i, ["Move", "Reach", "Open"]):
                container = act_arg
                oid = next_op_id()
                end_step = actions[i + 2][0]
                ops.append(OperationStep(
                    op_id=oid, op_type="open",
                    entity=None, dest=container,
                    deps=[], robot=slot,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, end_step)
                i += 3
                continue

            if act_name == "Reach" and _peek_sequence(actions, i, ["Reach", "Open"]):
                container = act_arg
                oid = next_op_id()
                end_step = actions[i + 1][0]
                ops.append(OperationStep(
                    op_id=oid, op_type="open",
                    entity=None, dest=container,
                    deps=[], robot=slot,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, end_step)
                i += 2
                continue

            # --- Grasp patterns (acquire / local_acquire) ---
            # Check for dual-hold: Grasp(A)→ ...→Grasp(B) before any Place
            # Mobile acquire: Move(X) → Reach(X) → Grasp(X)
            if act_name == "Move" and _peek_sequence(actions, i, ["Move", "Reach", "Grasp"]):
                item = act_arg
                end_step = actions[i + 2][0]

                # Check if item is actually a tool
                item_norm = normalize_entity_name(item) or item
                is_tool = entity_has_tag(item_norm, "tool") or entity_has_tag(item_norm, "interactable_tool")

                oid = next_op_id()
                ops.append(OperationStep(
                    op_id=oid, op_type="acquire",
                    entity=item, dest=None,
                    deps=[], robot=slot,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, end_step)
                holding.append(item)
                if is_tool:
                    holding_tool = item
                i += 3
                continue

            # Stationary acquire (panda): Reach(X) → Grasp(X)
            if act_name == "Reach" and _peek_sequence(actions, i, ["Reach", "Grasp"]):
                item = act_arg
                end_step = actions[i + 1][0]
                # Check if this is an unload_carrier pattern
                # (panda grasping item from carrier that was just pushed to it)
                is_unload = _is_unload_pattern(slot, item, carrier_ops, active_robots, time_steps, push_targets)

                if is_unload:
                    # Look ahead for the Place that follows
                    if i + 2 < len(actions) and actions[i + 2][1][0] == "Place":
                        final_dest = actions[i + 2][1][1]
                        final_dest_norm = normalize_entity_name(final_dest) or final_dest
                        final_dest_tags = get_entity_tags(final_dest_norm)
                        carrier_name = _find_carrier_for_unload(slot, carrier_ops, active_robots, time_steps, push_targets)
                        # If Place target is the carrier itself, it's load_carrier, not unload
                        if "pushable_carrier" not in final_dest_tags:
                            end_step = actions[i + 2][0]
                            oid = next_op_id()
                            ops.append(OperationStep(
                                op_id=oid, op_type="unload_carrier",
                                entity=item, dest=final_dest,
                                deps=[], robot=slot,
                                params={"carrier": carrier_name} if carrier_name else {},
                            ))
                            robot_ops[slot].append(oid)
                            op_step_range[oid] = (step_idx, end_step)
                            i += 3
                            continue

                # Normal local_acquire for panda
                item_norm = normalize_entity_name(item) or item
                is_tool = entity_has_tag(item_norm, "tool") or entity_has_tag(item_norm, "interactable_tool")
                oid = next_op_id()
                ops.append(OperationStep(
                    op_id=oid, op_type="local_acquire",
                    entity=item, dest=None,
                    deps=[], robot=slot,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, end_step)
                holding.append(item)
                if is_tool:
                    holding_tool = item
                i += 1 + 1  # Reach + Grasp
                continue

            # --- Place patterns ---
            if act_name == "Place":
                dest = act_arg
                dest_norm = normalize_entity_name(dest) or dest
                dest_tags = get_entity_tags(dest_norm)

                # Check: Place into carrier → load_carrier
                if "pushable_carrier" in dest_tags:
                    oid = next_op_id()
                    # entity = what we're placing (last held item)
                    placed_item = holding.pop() if holding else None
                    ops.append(OperationStep(
                        op_id=oid, op_type="load_carrier",
                        entity=placed_item, dest=dest,
                        deps=[], robot=slot,
                        params={"carrier": dest},
                    ))
                    robot_ops[slot].append(oid)
                    op_step_range[oid] = (step_idx, step_idx)
                    carrier_ops.setdefault(dest, []).append(oid)
                    i += 1
                    continue

                # Check: Place(sink) → Interact(sink) → process_at
                if "wash_station" in dest_tags and i + 1 < len(actions):
                    next_act = actions[i + 1][1]
                    if next_act[0] == "Interact" and (normalize_entity_name(next_act[1]) or next_act[1]) == dest_norm:
                        end_step = actions[i + 1][0]
                        placed_item = holding.pop() if holding else None
                        oid = next_op_id()
                        ops.append(OperationStep(
                            op_id=oid, op_type="process_at",
                            entity=placed_item, dest=dest,
                            deps=[], robot=slot,
                            params={"process": _infer_process(dest_tags)},
                        ))
                        robot_ops[slot].append(oid)
                        op_step_range[oid] = (step_idx, end_step)
                        i += 2
                        continue

                # Check: Place(activatable) → Interact(activatable) → deposit + activate
                if "activatable" in dest_tags and i + 1 < len(actions):
                    next_act = actions[i + 1][1]
                    if next_act[0] == "Interact" and (normalize_entity_name(next_act[1]) or next_act[1]) == dest_norm:
                        end_step_dep = step_idx
                        end_step_act = actions[i + 1][0]
                        placed_item = holding.pop() if holding else None

                        oid_dep = next_op_id()
                        ops.append(OperationStep(
                            op_id=oid_dep, op_type="deposit",
                            entity=placed_item, dest=dest,
                            deps=[], robot=slot,
                        ))
                        robot_ops[slot].append(oid_dep)
                        op_step_range[oid_dep] = (step_idx, end_step_dep)

                        oid_act = next_op_id()
                        ops.append(OperationStep(
                            op_id=oid_act, op_type="activate",
                            entity=None, dest=dest,
                            deps=[], robot=slot,
                            params={"polarity": detect_instruction_polarity(sample_data["instruction"])},
                        ))
                        robot_ops[slot].append(oid_act)
                        op_step_range[oid_act] = (end_step_act, end_step_act)
                        i += 2
                        continue

                # Normal deposit: either Move(dest)+Place(dest) or just Place(dest)
                placed_item = holding.pop() if holding else None
                if holding_tool and placed_item == holding_tool:
                    holding_tool = None
                oid = next_op_id()
                ops.append(OperationStep(
                    op_id=oid, op_type="deposit",
                    entity=placed_item, dest=dest,
                    deps=[], robot=slot,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, step_idx)
                i += 1
                continue

            # --- Move patterns (not Open, not acquire) ---
            if act_name == "Move":
                dest = act_arg
                # If holding items and next action is Place at same dest → deposit
                if holding and i + 1 < len(actions):
                    next_act = actions[i + 1][1]
                    if next_act[0] == "Place":
                        # Move(dest) + Place(dest) → deposit
                        # Will be handled by Place branch on next iteration
                        # But we need to mark this as part of deposit
                        # Actually, skip the Move - it's part of the deposit op
                        # Just advance, deposit will be caught in Place branch
                        i += 1
                        continue

                # Move while holding but not to Place → carry_to
                if holding:
                    oid = next_op_id()
                    ops.append(OperationStep(
                        op_id=oid, op_type="carry_to",
                        entity=holding[-1], dest=dest,
                        deps=[], robot=slot,
                    ))
                    robot_ops[slot].append(oid)
                    op_step_range[oid] = (step_idx, step_idx)
                    i += 1
                    continue

                # Standalone Move (carrier approach for push)
                # The Move before Push is part of push_carrier, skip it
                if i + 1 < len(actions) and actions[i + 1][1][0] == "Push":
                    i += 1
                    continue

                # Standalone Move without holding - rare, skip
                i += 1
                continue

            # --- Interact patterns ---
            if act_name == "Interact":
                target = act_arg
                target_norm = normalize_entity_name(target) or target
                target_tags = get_entity_tags(target_norm)

                # Interact with tool while holding it → apply_tool
                if holding_tool and (target_norm == (normalize_entity_name(holding_tool) or holding_tool)):
                    # Find the destination (where we carried the tool to)
                    tool_dest = None
                    # Look back for the carry_to op for this robot
                    for prev_op in reversed(ops):
                        if prev_op.robot == slot and prev_op.op_type == "carry_to":
                            tool_dest = prev_op.dest
                            break
                    tool_norm = normalize_entity_name(holding_tool) or holding_tool
                    dest_tags_for_effect = get_entity_tags(normalize_entity_name(tool_dest) or "") if tool_dest else set()
                    oid = next_op_id()
                    ops.append(OperationStep(
                        op_id=oid, op_type="apply_tool",
                        entity=holding_tool, dest=tool_dest,
                        deps=[], robot=slot,
                        params={"tool": holding_tool, "effect": _infer_effect(tool_norm, dest_tags_for_effect)},
                    ))
                    robot_ops[slot].append(oid)
                    op_step_range[oid] = (step_idx, step_idx)
                    i += 1
                    continue

                # Interact with inspectable thing (dog_check_environment)
                if "inspectable" in target_tags and not "activatable" in target_tags:
                    oid = next_op_id()
                    ops.append(OperationStep(
                        op_id=oid, op_type="inspect",
                        entity=None, dest=target,
                        deps=[], robot=slot,
                    ))
                    robot_ops[slot].append(oid)
                    op_step_range[oid] = (step_idx, step_idx)
                    i += 1
                    continue

                # Interact with activatable → activate
                if "activatable" in target_tags:
                    oid = next_op_id()
                    ops.append(OperationStep(
                        op_id=oid, op_type="activate",
                        entity=None, dest=target,
                        deps=[], robot=slot,
                        params={"polarity": detect_instruction_polarity(sample_data["instruction"])},
                    ))
                    robot_ops[slot].append(oid)
                    op_step_range[oid] = (step_idx, step_idx)
                    i += 1
                    continue

                # Fallback: inspect
                oid = next_op_id()
                ops.append(OperationStep(
                    op_id=oid, op_type="inspect",
                    entity=None, dest=target,
                    deps=[], robot=slot,
                ))
                robot_ops[slot].append(oid)
                op_step_range[oid] = (step_idx, step_idx)
                i += 1
                continue

            # Unknown action, skip
            i += 1

    # Step 3: Build deps
    _build_deps(ops, robot_ops, op_step_range, carrier_ops, active_robots, sample_data)

    # Step 4: Topological sort so that deps are always before dependents
    # (needed for convert_operations_to_graph_plan which processes ops in order)
    ops = _topo_sort_ops(ops)

    return ops


def _topo_sort_ops(ops: list[OperationStep]) -> list[OperationStep]:
    """Topologically sort ops so that deps come before dependents."""
    op_map = {op.op_id: op for op in ops}
    visited: set[str] = set()
    result: list[OperationStep] = []

    def visit(op_id: str):
        if op_id in visited:
            return
        visited.add(op_id)
        op = op_map.get(op_id)
        if op is None:
            return
        for dep in op.deps:
            visit(dep)
        result.append(op)

    for op in ops:
        visit(op.op_id)
    return result


def _peek_sequence(actions: list[tuple], start: int, pattern: list[str]) -> bool:
    """Check if action sequence starting at `start` matches the pattern."""
    if start + len(pattern) > len(actions):
        return False
    for j, expected in enumerate(pattern):
        if actions[start + j][1][0] != expected:
            return False
    # Also check that args are consistent for the matched actions
    # (e.g., Move(X), Reach(X), Grasp(X) should all target X)
    if len(pattern) >= 2:
        first_arg = actions[start][1][1] if len(actions[start][1]) > 1 else None
        for j in range(1, len(pattern)):
            j_arg = actions[start + j][1][1] if len(actions[start + j][1]) > 1 else None
            # For Open pattern, Open target should match Move/Reach target
            # For Grasp pattern, all should match
            if j_arg != first_arg:
                return False
    return True


def _is_unload_pattern(slot, item, carrier_ops, active_robots, time_steps, push_targets=None) -> bool:
    """Check if a Reach+Grasp on a panda is unloading from a carrier."""
    # If this slot was the target of a Push, then Reach+Grasp = unload
    if push_targets and slot in push_targets:
        return True
    return False


def _find_carrier_for_unload(slot, carrier_ops, active_robots, time_steps, push_targets=None) -> str | None:
    """Find the carrier name for an unload operation at this slot."""
    if push_targets and slot in push_targets:
        return push_targets[slot]
    for carrier in carrier_ops:
        return carrier
    return None


def _build_deps(
    ops: list[OperationStep],
    robot_ops: dict[str, list[str]],
    op_step_range: dict[str, tuple[int, int]],
    carrier_ops: dict[str, list[str]],
    active_robots: dict[str, str],
    sample_data: dict,
):
    """Build dependency edges for the operation DAG.

    Rules:
    1. Same robot: sequential ops depend on the previous op
    2. Carrier chain: load→push→push→unload are chained
    3. Cross-robot cutting: apply_tool depends on deposit (of fruit)
    """
    op_map = {op.op_id: op for op in ops}

    # Rule 1: Same robot sequential deps
    for slot, op_ids in robot_ops.items():
        for i in range(1, len(op_ids)):
            prev_id = op_ids[i - 1]
            curr_id = op_ids[i]
            op_map[curr_id].deps.append(prev_id)

    # Rule 2: Carrier chain deps
    # carrier_ops has all carrier-related ops. Chain them properly.
    for carrier, related_op_ids in carrier_ops.items():
        # Find all ops related to this carrier across all robots
        carrier_related = []
        for op in ops:
            if op.op_type in ("load_carrier", "push_carrier", "unload_carrier"):
                if op.entity == carrier or (op.params and op.params.get("carrier") == carrier):
                    carrier_related.append(op)
        # Order by step
        carrier_related.sort(key=lambda o: op_step_range.get(o.op_id, (999, 999))[0])
        for j in range(1, len(carrier_related)):
            prev = carrier_related[j - 1]
            curr = carrier_related[j]
            if prev.op_id not in curr.deps:
                curr.deps.append(prev.op_id)

    # Rule 3: Cross-robot deps for cutting
    # If robot A deposits fruit on cutting_board, and robot B does apply_tool,
    # then apply_tool depends on the deposit
    for op in ops:
        if op.op_type == "apply_tool":
            # Find the deposit of fruit on cutting board by another robot
            for other_op in ops:
                if other_op.op_type == "deposit" and other_op.robot != op.robot:
                    dest_norm = normalize_entity_name(other_op.dest) or ""
                    if entity_has_tag(dest_norm, "cutting_surface"):
                        if other_op.op_id not in op.deps:
                            op.deps.append(other_op.op_id)

    # Rule 4: Cross-robot deps for parallel deposit to same dest
    # If two robots deposit to the same openable container, the one that opens
    # it should be a dep for the other's deposit
    for op in ops:
        if op.op_type == "deposit":
            dest_norm = normalize_entity_name(op.dest) or ""
            if entity_has_tag(dest_norm, "openable"):
                # Find the open op for this dest
                for other_op in ops:
                    if other_op.op_type == "open" and (normalize_entity_name(other_op.dest) or "") == dest_norm:
                        if other_op.op_id not in op.deps and other_op.robot != op.robot:
                            op.deps.append(other_op.op_id)

    # Deduplicate deps and remove self-deps
    for op in ops:
        op.deps = sorted(set(d for d in op.deps if d != op.op_id))


# =====================================================================
# 4. Round-trip verification
# =====================================================================

def verify_round_trip(
    ops: list[OperationStep],
    scene_facts: SceneFacts,
    affordance_context: AffordanceContext,
    gt_time_steps: list,
    robots_raw: dict,
) -> dict:
    """Verify: OpDAG → GraphPlan → dispatch → compare with GT.

    Returns dict with match status and diagnostics.
    """
    try:
        graph_plan = convert_operations_to_graph_plan(ops, affordance_context, scene_facts)
        # Resolve executor_robot_type from slot→type mapping
        slot_to_type = {}
        for slot, rtype in robots_raw.items():
            if rtype is not None:
                slot_to_type[slot] = normalize_robot_name(rtype) or rtype
        for node in graph_plan.nodes:
            if node.executor_slot and not node.executor_robot_type:
                node.executor_robot_type = slot_to_type.get(node.executor_slot)
        micro_steps = dispatch_graph(graph_plan)
    except Exception as e:
        return {
            "match": False,
            "error": f"forward_pass_error: {type(e).__name__}: {e}",
            "reconstructed_steps": None,
            "gt_steps": _format_gt_steps(gt_time_steps),
        }

    # Convert micro_steps to comparable format
    recon = []
    for ms in micro_steps:
        step_acts = {}
        for slot, action in ms.actions.items():
            step_acts[slot] = list(action) if action else None
        recon.append(step_acts)

    gt_formatted = _format_gt_steps(gt_time_steps)

    # Compare
    match = _compare_step_sequences(recon, gt_formatted)
    action_set_match = _compare_action_sets(recon, gt_formatted)
    action_set_match_normalized = _compare_action_sets_normalized(recon, gt_formatted)

    return {
        "match": match,
        "action_set_match": action_set_match,
        "action_set_match_normalized": action_set_match_normalized,
        "error": None,
        "reconstructed_steps": recon,
        "gt_steps": gt_formatted,
        "recon_len": len(recon),
        "gt_len": len(gt_formatted),
    }


def _format_gt_steps(gt_time_steps) -> list[dict]:
    """Convert GT time_steps to comparable format."""
    result = []
    for ts in gt_time_steps:
        step_acts = {}
        for slot, act in ts["actions"].items():
            if act is not None:
                step_acts[slot] = list(act)
        result.append(step_acts)
    return result


def _compare_step_sequences(recon: list[dict], gt: list[dict]) -> bool:
    """Compare two step sequences for equality."""
    if len(recon) != len(gt):
        return False
    for r_step, g_step in zip(recon, gt):
        if set(r_step.keys()) != set(g_step.keys()):
            return False
        for slot in r_step:
            if r_step[slot] != g_step[slot]:
                return False
    return True


def _compare_action_sets(recon: list[dict], gt: list[dict]) -> bool:
    """Compare per-robot ordered action sequences (ignoring scheduling).

    Extracts each robot's actions in order and checks they match,
    regardless of which step each action is in.
    """
    def extract_per_robot(steps):
        per_robot = {}
        for step in steps:
            for slot, act in step.items():
                per_robot.setdefault(slot, []).append(tuple(act))
        return per_robot

    recon_per = extract_per_robot(recon)
    gt_per = extract_per_robot(gt)

    if set(recon_per.keys()) != set(gt_per.keys()):
        return False
    for slot in recon_per:
        if recon_per[slot] != gt_per[slot]:
            return False
    return True


def _normalize_reach_before_interact(actions: list[tuple]) -> list[tuple]:
    """Normalize: insert Reach(dest) before Interact(tool) when GT omits it.

    Rule: if Interact(X) follows Move(Y) and X != Y, insert Reach(Y).
    This handles GT inconsistency where tool interactions sometimes skip Reach.
    """
    normalized = []
    for act in actions:
        if (act[0] == "Interact" and normalized
                and normalized[-1][0] == "Move"
                and normalized[-1][1] != act[1]):
            normalized.append(("Reach", normalized[-1][1]))
        normalized.append(act)
    return normalized


def _compare_action_sets_normalized(recon: list[dict], gt: list[dict]) -> bool:
    """Like _compare_action_sets but normalizes Reach before Interact."""
    def extract_per_robot(steps):
        per_robot = {}
        for step in steps:
            for slot, act in step.items():
                per_robot.setdefault(slot, []).append(tuple(act))
        return per_robot

    recon_per = extract_per_robot(recon)
    gt_per = extract_per_robot(gt)

    if set(recon_per.keys()) != set(gt_per.keys()):
        return False
    for slot in recon_per:
        r_norm = _normalize_reach_before_interact(recon_per[slot])
        g_norm = _normalize_reach_before_interact(gt_per[slot])
        if r_norm != g_norm:
            return False
    return True


# =====================================================================
# 5. Main pipeline
# =====================================================================

def process_sample(row, idx: int, system_prompt: str, detector=None, refiner=None) -> dict:
    """Process one sample: extract → parse → reverse_map → serialize → verify.

    The returned dict may include an ``augmented`` field carrying a synthetic
    polarity-off variant when the original instruction is amenable to an
    on→off paraphrase. The augmented variant shares the GT atomic actions
    (Interact is symmetric in VIKI-L2) so round-trip is preserved.
    """
    sample_data = extract_sample_data(row)

    # Extract image bytes from parquet when YOLO is enabled
    image_bytes = b""
    if detector is not None:
        try:
            images = row["images"]
            if hasattr(images, '__len__') and len(images) > 0:
                elem = images[0]
                if isinstance(elem, dict) and "bytes" in elem:
                    image_bytes = bytes(elem["bytes"])
        except Exception:
            pass  # fall back to no image

    # Build SceneFacts
    scene_facts = build_scene_facts_from_gt(
        sample_data, idx=idx, image_bytes=image_bytes,
        detector=detector, refiner=refiner,
    )

    # Build AffordanceContext
    affordance_context = build_affordance_context(scene_facts)

    # Reverse-map GT to Operation DAG
    ops = reverse_map_time_steps(sample_data)

    # Serialize prompts
    user_prompt = serialize_slm_input(scene_facts, affordance_context)

    # Target DAG JSON
    target_dag = [
        {
            "op_id": op.op_id,
            "op_type": op.op_type,
            "entity": op.entity,
            "dest": op.dest,
            "deps": op.deps,
            "robot": op.robot,
            **({"params": op.params} if op.params else {}),
        }
        for op in ops
    ]

    # Round-trip verification
    rt_result = verify_round_trip(ops, scene_facts, affordance_context, sample_data["time_steps"], sample_data["robots_raw"])

    result = {
        "idx": idx,
        "task_name": sample_data["task_name"],
        "task_id": sample_data["task_id"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "target_dag": target_dag,
        "round_trip": rt_result,
        "instruction": sample_data["instruction"],
        "robots": sample_data["robot_inventory"],
    }

    # ── Polarity-off augmentation ────────────────────────────────
    # Synthesize a sibling sample only when (a) the DAG contains an activate
    # op and (b) the instruction can be cleanly rewritten to an "off" cue.
    has_activate = any(op.op_type == "activate" for op in ops)
    if has_activate:
        off_instruction = paraphrase_instruction_off(sample_data["instruction"])
        if off_instruction is not None:
            aug_sample_data = dict(sample_data)
            aug_sample_data["instruction"] = off_instruction
            # Rebuild SceneFacts with the rewritten instruction so the
            # serialized [SCENE] block carries the off-cue (necessary for
            # training the model on the new prompt).
            aug_scene_facts = build_scene_facts_from_gt(
                aug_sample_data, idx=idx, image_bytes=image_bytes,
                detector=detector, refiner=refiner,
            )
            aug_affordance_context = build_affordance_context(aug_scene_facts)
            aug_user_prompt = serialize_slm_input(aug_scene_facts, aug_affordance_context)
            # Flip polarity on every activate op; copy other fields verbatim.
            aug_target_dag = []
            for op_obj in target_dag:
                op_copy = dict(op_obj)
                if op_copy["op_type"] == "activate":
                    params = dict(op_copy.get("params") or {})
                    params["polarity"] = "off"
                    op_copy["params"] = params
                aug_target_dag.append(op_copy)
            result["augmented"] = {
                "instruction": off_instruction,
                "user_prompt": aug_user_prompt,
                "target_dag": aug_target_dag,
                # Round-trip is unchanged (Interact symbol identical) — record
                # it so downstream filtering treats it like a real sample.
                "round_trip": rt_result,
            }

    return result


def main():
    parser = argparse.ArgumentParser(description="Build SLM training data from VIKI-L2")
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--split", required=True, help="Split name (train/val/test/debug)")
    parser.add_argument("--output_dir", default=f"data/{DATASET_VERSION}/slm_dataset", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--yolo", action="store_true", help="Enable YOLO detection for scene_facts enrichment")
    parser.add_argument("--yolo_weights", type=str, default=None, help="YOLO weights path (default None: caller resolves per benchmark — VIKI 20-class / LOCUS 34-class, facts:D3a)")
    parser.add_argument("--yolo_device", type=str, default=None, help="YOLO device (e.g. cuda:0)")
    args = parser.parse_args()

    print(f"Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"Processing {len(df)} samples (split={args.split})")

    # Build YOLO detector and refiner if requested
    detector = None
    refiner = None
    if args.yolo:
        print("Initializing YOLO detector...")
        detector = build_structured_detector(
            "ultralytics_yolo26",
            weights_path=args.yolo_weights,
            device=args.yolo_device,
        )
        refiner = build_scene_facts_refiner("none")
        print(f"  weights: {detector.weights_path}")
        print(f"  device: {detector.device or 'auto'}")

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)

    system_prompt = build_slm_system_prompt()

    results = []
    match_count = 0
    action_set_match_count = 0
    action_set_normalized_count = 0
    fail_count = 0
    error_count = 0

    # Clean stale failure files
    for f in failures_dir.glob("sample_*.json"):
        f.unlink()

    for idx in range(len(df)):
        row = df.iloc[idx]
        try:
            result = process_sample(row, idx, system_prompt, detector=detector, refiner=refiner)
            results.append(result)

            if result["round_trip"]["match"]:
                match_count += 1
                action_set_match_count += 1
                action_set_normalized_count += 1
            elif result["round_trip"].get("action_set_match"):
                action_set_match_count += 1
                action_set_normalized_count += 1
                fail_count += 1
            elif result["round_trip"].get("action_set_match_normalized"):
                action_set_normalized_count += 1
                fail_count += 1
            else:
                fail_count += 1
                # Save failure detail
                fail_path = failures_dir / f"sample_{idx}.json"
                with open(fail_path, "w") as f:
                    json.dump({
                        "idx": idx,
                        "task_name": result["task_name"],
                        "instruction": result["instruction"],
                        "target_dag": result["target_dag"],
                        "round_trip": result["round_trip"],
                    }, f, indent=2, default=str)

        except Exception as e:
            error_count += 1
            fail_path = failures_dir / f"sample_{idx}_error.json"
            with open(fail_path, "w") as f:
                json.dump({"idx": idx, "error": f"{type(e).__name__}: {e}"}, f, indent=2)

        if (idx + 1) % 50 == 0 or idx == len(df) - 1:
            total = match_count + fail_count + error_count
            print(f"  [{idx + 1}/{len(df)}] step_match={match_count} action_match={action_set_match_count} action_norm={action_set_normalized_count} fail={fail_count} error={error_count}")

    # Write samples.jsonl
    samples_path = output_dir / "samples.jsonl"
    augmented_count = 0
    activate_sample_count = 0
    with open(samples_path, "w") as f:
        for r in results:
            target_dag = r["target_dag"]
            has_activate = any(op["op_type"] == "activate" for op in target_dag)
            if has_activate:
                activate_sample_count += 1
            record = {
                "system_prompt": r["system_prompt"],
                "user_prompt": r["user_prompt"],
                "target_dag": target_dag,
                "metadata": {
                    "idx": r["idx"],
                    "task_name": r["task_name"],
                    "task_id": r["task_id"],
                    "instruction": r["instruction"],
                    "robots": r["robots"],
                    "round_trip_match": r["round_trip"]["match"],
                    "augmented": False,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Emit polarity-off sibling sample when available.
            aug = r.get("augmented")
            if aug is not None:
                augmented_count += 1
                aug_record = {
                    "system_prompt": r["system_prompt"],
                    "user_prompt": aug["user_prompt"],
                    "target_dag": aug["target_dag"],
                    "metadata": {
                        "idx": r["idx"],
                        "task_name": r["task_name"],
                        "task_id": r["task_id"],
                        "instruction": aug["instruction"],
                        "robots": r["robots"],
                        "round_trip_match": aug["round_trip"]["match"],
                        "augmented": True,
                        "augmentation": "polarity_off",
                        "source_idx": r["idx"],
                    },
                }
                f.write(json.dumps(aug_record, ensure_ascii=False) + "\n")

    # Write report
    report = {
        "split": args.split,
        "total": len(df),
        "processed": len(results),
        "step_match": match_count,
        "action_set_match": action_set_match_count,
        "action_set_match_normalized": action_set_normalized_count,
        "fail": fail_count - (action_set_normalized_count - match_count),  # true failures
        "error": error_count,
        "step_match_rate": match_count / max(len(results), 1),
        "action_set_match_rate": action_set_match_count / max(len(results), 1),
        "action_set_normalized_rate": action_set_normalized_count / max(len(results), 1),
        "activate_samples": activate_sample_count,
        "polarity_off_augmented": augmented_count,
        "polarity_off_aug_ratio": augmented_count / max(activate_sample_count, 1),
        "fail_by_task": _count_by_task(results, match=False),
        "match_by_task": _count_by_task(results, match=True),
    }
    report_path = output_dir / "round_trip_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone. Output: {output_dir}")
    print(f"  samples.jsonl: {len(results) + augmented_count} records "
          f"(originals={len(results)}, polarity_off_aug={augmented_count})")
    print(f"  Activate samples: {activate_sample_count}; "
          f"aug coverage: {augmented_count / max(activate_sample_count, 1) * 100:.1f}%")
    print(f"  Step-exact match: {match_count}/{len(results)} ({match_count/max(len(results),1)*100:.1f}%)")
    print(f"  Action-set match: {action_set_match_count}/{len(results)} ({action_set_match_count/max(len(results),1)*100:.1f}%)")
    print(f"  Action-set normalized: {action_set_normalized_count}/{len(results)} ({action_set_normalized_count/max(len(results),1)*100:.1f}%)")
    print(f"  Report: {report_path}")


def _count_by_task(results: list[dict], match: bool) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        if r["round_trip"]["match"] == match:
            t = r["task_name"]
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
