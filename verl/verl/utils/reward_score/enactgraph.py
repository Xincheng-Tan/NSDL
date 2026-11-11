"""Reward function for EnactGraph GRPO training.

Three public reward versions are supported via ``ENACTGRAPH_REWARD_VERSION``:

* ``v1``: the official VIKI-R-style baseline
  ``0.1 * R_format + 0.9 * R_acc``. Official VIKI-R parses
  ``<answer>...</answer>`` step/action JSON and evaluates it with
  ``eval_single``; this repo also accepts the current
  ``[GOAL]/[ASSIGN]/[PLAN]`` Operation-DAG output by extracting ``[PLAN]``
  and canonicalizing it to the same eval command representation.
* ``v2``: the legacy three-block CoT reward
  ``0.05*R_format + 0.15*R_goal + 0.10*R_assign + 0.15*R_validator
  + 0.25*R_plan_topology + 0.30*R_plan_tokens``. ``R_plan_topology`` and
  ``R_plan_tokens`` are gated by ``R_validator==1``.
* ``v3`` (default, 2026-05-22): re-weight v2 to address the dominant
  ``[GOAL].active`` mis-identification failure mode (docs/debug.md §6):
  ``0.05*R_format + 0.30*R_goal_v3 + 0.10*R_assign + 0.05*R_validator
  + 0.20*R_plan_topology + 0.30*R_plan_tokens``. The hard validator gate
  on plan rewards is REMOVED (validator misses real failures and
  occasionally false-positives, so gating amplifies its noise);
  ``R_goal_v3`` adds an explicit active-set Jaccard sub-score.

Self-correction is intentionally trained through Stream-Sc SFT, not through a
fake per-sample GRPO reward; diversity is a rollout-group property and is also
not represented in this single-completion score function.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)
# Guard so a GT/eval_single field-parity bug surfaces instead of silently becoming
# R_task==0 for every sample (docs/RL_debug.md §1 S-①c). Counted per exception type,
# not a one-shot bool: a one-shot guard reports "1 log line", which reads like
# "1 occurrence" and hides the true frequency. Re-logged at 1/10/100/... so a
# high-rate failure escalates while a rare one stays quiet.
_SIMULATE_EXC_COUNTS: "dict[str, int]" = {}


def simulate_exception_counts() -> "dict[str, int]":
    """Snapshot of swallowed ``eval_single`` exceptions, keyed by ``ExcType: msg``.

    Every entry is a trajectory whose ``R_task`` was forced to 0 by an error rather
    than by the plan actually failing. Non-empty means the reward signal is
    contaminated; read it after a run to size the contamination.
    """
    return dict(_SIMULATE_EXC_COUNTS)

# Make sure the workspace root is importable so we can reach nsdl
_REPO_ROOT = Path(__file__).resolve().parents[3]  # verl/verl/utils/reward_score → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verl.utils.reward_score.utils.eval.eval_viki_2 import eval_single  # noqa: E402


# ---------------------------------------------------------------------------
# Full inference pipeline (lazily imported to keep reward-module import cheap)
# ---------------------------------------------------------------------------
_PIPELINE_READY = False
_PIPELINE_IMPORT_ERROR: Exception | None = None
_parse_slm_output = None
_validate_operations = None
_convert_operations_to_graph_plan = None
_build_affordance_context = None
_dispatch_graph = None
_to_l2_answer = None
_plan_to_eval_commands = None
_assign_executors_to_graph = None
_SceneFacts = None
_RobotInstance = None


def _ensure_pipeline_imported() -> bool:
    global _PIPELINE_READY, _PIPELINE_IMPORT_ERROR
    global _parse_slm_output, _validate_operations
    global _convert_operations_to_graph_plan, _build_affordance_context
    global _dispatch_graph, _to_l2_answer, _plan_to_eval_commands
    global _assign_executors_to_graph, _SceneFacts, _RobotInstance

    if _PIPELINE_READY:
        return True
    if _PIPELINE_IMPORT_ERROR is not None:
        return False

    try:
        from nsdl.slm_interface import (  # noqa: E402
            parse_slm_output as _ps,
            validate_operations as _vo,
            convert_operations_to_graph_plan as _cv,
        )
        from nsdl.aig_builder import build_affordance_context as _ba  # noqa: E402
        from nsdl.dispatcher import dispatch_graph as _dg  # noqa: E402
        from nsdl.serializer import to_l2_answer as _ta  # noqa: E402
        from nsdl.validator import plan_to_eval_commands as _pc  # noqa: E402
        from nsdl.macro_graph import _assign_executors_to_graph as _ae  # noqa: E402
        from nsdl.schema import SceneFacts as _SF, RobotInstance as _RI  # noqa: E402

        _parse_slm_output = _ps
        _validate_operations = _vo
        _convert_operations_to_graph_plan = _cv
        _build_affordance_context = _ba
        _dispatch_graph = _dg
        _to_l2_answer = _ta
        _plan_to_eval_commands = _pc
        _assign_executors_to_graph = _ae
        _SceneFacts = _SF
        _RobotInstance = _RI
        _PIPELINE_READY = True
        return True
    except Exception as exc:  # noqa: BLE001
        _PIPELINE_IMPORT_ERROR = exc
        return False


def _build_minimal_scene_facts(gt: Dict[str, Any]):
    """Reconstruct a minimal SceneFacts from ground_truth for pipeline reuse.

    Only ``instruction`` and ``robot_inventory`` are populated — the SLM's DAG
    already encodes all instance-level decisions, so other scene_facts fields
    (target_items, destination_pairs, ...) are left as dataclass defaults.
    """
    robots = gt.get("robots") or {}
    robot_inventory = []
    for slot, rtype in robots.items():
        if not rtype:
            continue
        slot_str = str(slot)
        robot_inventory.append(_RobotInstance(slot_id=slot_str, robot_type=str(rtype)))

    instruction = gt.get("description") or gt.get("instruction") or ""
    return _SceneFacts(
        parser_backend="rl_reward",
        instruction=str(instruction),
        image_available=False,
        robot_inventory=robot_inventory,
        primary_object=None,
        destination=None,
    )


def _dag_to_eval_steps_via_pipeline(dag: List[dict], gt: Dict[str, Any]) -> List[Dict[str, str]] | None:
    """Run the deployed inference pipeline to convert a raw DAG to eval commands.

    Returns None on any failure so the caller can fall back to the lightweight
    converter ``_dag_to_eval_steps``.
    """
    if not _ensure_pipeline_imported():
        return None
    try:
        scene_facts = _build_minimal_scene_facts(gt)
        affordance_ctx = _build_affordance_context(scene_facts)

        raw_text = json.dumps(dag)
        ops = _parse_slm_output(raw_text)
        # IMPORTANT: do NOT call any deterministic repair here. Per
        # docs/rules.md (Validator Boundary Rule) the training reward must
        # reflect raw SLM output, not validator-repaired DAGs.
        if not ops:
            return None

        graph_plan = _convert_operations_to_graph_plan(ops, affordance_ctx, scene_facts)
        graph_plan = _assign_executors_to_graph(graph_plan, scene_facts)
        micro_steps = _dispatch_graph(graph_plan)
        l2_plan = _to_l2_answer(micro_steps)
        return _plan_to_eval_commands(l2_plan)
    except Exception:
        return None


def _numpy_to_python(obj):
    """Recursively convert numpy types to native Python types.

    Parquet round-trip can leave numpy arrays / np.integer in nested dicts,
    which breaks ``filter_none_values`` (numpy ``None`` ≠ Python ``None``).
    """
    if isinstance(obj, np.ndarray):
        return [_numpy_to_python(x) for x in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Lightweight DAG → atomic-step converter
# ---------------------------------------------------------------------------
# For open/close: when entity is missing, fall back to dest (e.g. open(None, cabinet))
# For apply_tool: Interact with entity (the tool), not dest (the surface)

_OP_PRIMITIVES = {
    "acquire":       lambda e, d: [("Move", e), ("Reach", e), ("Grasp", e)],
    "local_acquire": lambda e, d: [("Reach", e), ("Grasp", e)],
    "deposit":       lambda e, d: [("Move", d), ("Place", d)],
    "carry_to":      lambda e, d: [("Move", d)],
    "open":          lambda e, d: [("Move", e or d), ("Reach", e or d), ("Open", e or d)],
    "close":         lambda e, d: [("Move", e or d), ("Reach", e or d), ("Close", e or d)],
    "process_at":    lambda e, d: [("Move", d), ("Place", d), ("Interact", d)],
    "activate":      lambda e, d: [("Move", e or d), ("Interact", e or d)],
    "push_carrier":  lambda e, d: [("Move", e), ("Push", e, d)] if d else [("Move", e), ("Push", e)],
    "inspect":       lambda e, d: [("Move", e or d), ("Interact", e or d)],
    "handoff":       lambda e, d: [("Move", e), ("Reach", e), ("Grasp", e)],
    "load_carrier":  lambda e, d: [("Place", d)],
    "unload_carrier": lambda e, d: [("Reach", e), ("Grasp", e), ("Place", d)],
    "apply_tool":    lambda e, d: [("Move", d), ("Interact", e)],
}


def _dag_to_eval_steps(dag: list[dict]) -> list[dict[str, str]]:
    """Convert an Operation DAG to eval-compatible step list.

    Returns list of dicts: ``[{"R1": "<Move,apple>"}, ...]``
    Uses a simple topological scheduler respecting deps & single-action-per-robot.
    """
    # Parse operations
    ops: dict[str, dict] = {}
    seen_carrier_pushes: set[tuple[str, str]] = set()
    for op_dict in dag:
        op_id = str(op_dict.get("op_id", ""))
        op_type = str(op_dict.get("op_type", "")).lower().strip()
        entity = op_dict.get("entity") or ""
        dest = op_dict.get("dest") or ""
        deps_raw = op_dict.get("deps") or op_dict.get("predecessor_op_ids") or []
        if not isinstance(deps_raw, list):
            deps_raw = [deps_raw]
        deps = [str(d).strip() for d in deps_raw if str(d).strip()]
        robot = str(op_dict.get("robot") or "R1").strip()

        prim_fn = _OP_PRIMITIVES.get(op_type)
        if prim_fn is None:
            continue
        primitives = prim_fn(entity, dest)
        if op_type == "push_carrier":
            push_key = (robot, str(entity))
            push_action = ("Push", entity, dest) if dest else ("Push", entity)
            primitives = [push_action]
            if push_key not in seen_carrier_pushes:
                primitives = [("Move", entity), *primitives]
            seen_carrier_pushes.add(push_key)
        # Filter out actions with empty targets
        primitives = [p for p in primitives if all(arg for arg in p)]
        if not primitives:
            continue
        ops[op_id] = {
            "robot": robot,
            "op_type": op_type,
            "entity": entity,
            "deps": [d for d in deps if d in {od.get("op_id") for od in dag}],
            "primitives": primitives,
            "offset": 0,
        }

    # Schedule: topological, one action per robot per time step
    completed: set[str] = set()
    steps: list[dict[str, str]] = []
    robot_at: dict[str, str] = {}
    max_iters = 500

    def _movement_deps_ready(op: dict, completed_snapshot: set[str]) -> bool:
        """Allow safe pre-positioning before non-movement deps finish."""
        if op["op_type"] not in {"deposit", "process_at", "carry_to", "apply_tool"}:
            return False
        if op["offset"] != 0:
            return False
        primitives = op["primitives"]
        if not primitives or primitives[0][0] != "Move":
            return False
        item = str(op.get("entity") or "")
        hard_dep_types = {"acquire", "local_acquire", "load_carrier", "process_at", "apply_tool", "carry_to"}
        for dep_id in op["deps"]:
            dep = ops.get(dep_id)
            if dep is None:
                continue
            if dep["op_type"] in hard_dep_types and str(dep.get("entity") or "") == item:
                if dep_id not in completed_snapshot:
                    return False
        return True

    def _skip_redundant_moves(op: dict, robot: str) -> bool:
        """Advance over Move(target) when the robot is already there."""
        advanced_without_action = False
        while op["offset"] < len(op["primitives"]):
            action = op["primitives"][op["offset"]]
            if action[0] != "Move" or robot_at.get(robot) != action[1]:
                break
            op["offset"] += 1
            advanced_without_action = True
        return advanced_without_action

    def _update_robot_location(robot: str, action: tuple):
        name = action[0]
        if name in {"Move", "Place", "Interact", "Open", "Close"} and len(action) > 1:
            robot_at[robot] = str(action[1])
        elif name == "Push" and len(action) > 1:
            robot_at[robot] = str(action[1])

    for _ in range(max_iters):
        if len(completed) >= len(ops):
            break
        step_actions: dict[str, str] = {}
        busy_robots: set[str] = set()
        completed_at_step_start = set(completed)
        advanced = False

        for op_id, op in ops.items():
            if op_id in completed:
                continue
            deps_ready = all(d in completed_at_step_start for d in op["deps"])
            movement_only = (not deps_ready) and _movement_deps_ready(op, completed_at_step_start)
            if not deps_ready and not movement_only:
                continue
            robot = op["robot"]
            if robot in busy_robots:
                continue
            if _skip_redundant_moves(op, robot):
                advanced = True
                if op["offset"] >= len(op["primitives"]):
                    completed.add(op_id)
                    continue
            if movement_only and (
                op["offset"] >= len(op["primitives"]) or op["primitives"][op["offset"]][0] != "Move"
            ):
                continue
            if op["offset"] < len(op["primitives"]):
                action = op["primitives"][op["offset"]]
                step_actions[robot] = f"<{','.join(action)}>"
                op["offset"] += 1
                busy_robots.add(robot)
                advanced = True
                _update_robot_location(robot, action)
                if op["offset"] >= len(op["primitives"]):
                    completed.add(op_id)

        if step_actions:
            steps.append(step_actions)
        if not advanced:
            break  # deadlock or done

    return steps


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

def format_reward(predict_str: str) -> float:
    """Check if the model output is a parseable JSON DAG.

    Returns 1.0 if the output is a valid JSON list of operation dicts
    (each with at least ``op_id`` and ``op_type``), else 0.0.
    """
    text = predict_str.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return 0.0

    if not isinstance(parsed, list) or len(parsed) == 0:
        return 0.0

    # Check at least one element has op_id and op_type
    for item in parsed:
        if isinstance(item, dict) and "op_id" in item and "op_type" in item:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Accuracy reward (shared simulation path; gate differs between v1 / v2)
# ---------------------------------------------------------------------------

def _parse_dag(predict_str: str) -> List[dict] | None:
    """Strip markdown fences and JSON-parse to a list of op dicts. None on failure."""
    text = predict_str.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        dag = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            dag = ast.literal_eval(text)
        except Exception:
            return None
    return dag if isinstance(dag, list) else None


_VIKI_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_VIKI_STRUCTURE_RE = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)


def _parse_literal_or_json(text: str) -> Any:
    """Parse JSON/Python-literal-ish model output, matching VIKI-R tolerance."""
    text = text.strip()
    if re.search(r"\{\{.*\}\}", text, re.DOTALL):
        text = text.replace("{{", "{").replace("}}", "}")
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        try:
            return json.loads(text)
        except Exception:
            return eval(text, {"__builtins__": {}}, {})  # noqa: S307 - parity with official reward


def _extract_viki_answer_obj(predict_str: str) -> Any | None:
    """Return parsed official ``<answer>`` object if present, else None."""
    match = _VIKI_ANSWER_RE.search(predict_str)
    if not match:
        return None
    try:
        parsed = _parse_literal_or_json(match.group(1).strip())
    except Exception:
        return None
    return parsed


def _transform_viki_actions(data: Any) -> list[dict[str, str]]:
    """Official VIKI-R step/action JSON -> eval command list."""
    if isinstance(data, dict) and "step" in data and "actions" in data:
        data = [data]
    if not isinstance(data, list):
        return []

    result: list[dict[str, str]] = []
    try:
        for step_info in data:
            if not isinstance(step_info, dict):
                continue
            actions = step_info.get("actions", {})
            if not isinstance(actions, dict):
                continue
            step_actions = {}
            for robot, action_list in actions.items():
                if not action_list:
                    continue
                step_actions[str(robot)] = f'<{",".join(str(x) for x in action_list)}>'
            if step_actions:
                result.append(step_actions)
    except Exception:
        return []
    return result


def _canonical_v1_eval_steps(predict_str: str, gt: Dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    """Canonicalize official VIKI or current EnactGraph output for v1 reward.

    Returns ``(eval_steps, format_ok)``. ``format_ok`` is the VIKI-R format
    component adapted to the representation: official outputs must contain
    ``<think>/<answer>`` with parseable list/dict; current outputs must expose
    a parseable Operation DAG in ``[PLAN]`` or as a bare JSON list.
    """
    official_obj = _extract_viki_answer_obj(predict_str)
    if official_obj is not None:
        format_ok = bool(_VIKI_STRUCTURE_RE.fullmatch(predict_str)) and isinstance(official_obj, (list, dict))
        return _transform_viki_actions(official_obj), bool(format_ok)

    plan_text = _cot_plan_text(predict_str)
    dag = _parse_dag(plan_text)
    format_ok = bool(dag) and any(isinstance(item, dict) and "op_id" in item and "op_type" in item for item in dag)
    if not dag:
        return [], False
    eval_steps = _dag_to_eval_steps_via_pipeline(dag, gt)
    if not eval_steps:
        eval_steps = _dag_to_eval_steps(dag)
    return eval_steps, bool(format_ok)


def format_reward_v1(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """VIKI-R format reward with a current-output adapter."""
    gt = _numpy_to_python(ground_truth)
    _, format_ok = _canonical_v1_eval_steps(predict_str, gt)
    return 1.0 if format_ok else 0.0


def acc_reward_v1(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """Official VIKI-R binary acc reward with ratio gate, after canonicalization."""
    try:
        gt = _numpy_to_python(ground_truth)
        eval_steps, _ = _canonical_v1_eval_steps(predict_str, gt)
        if not eval_steps:
            return 0.0
        try:
            passed = bool(eval_single(eval_steps, gt))
        except Exception:
            return 0.0
        if not passed:
            return 0.0
        len_gt = len(gt.get("time_steps", []) or [])
        len_pred = len(eval_steps)
        if len_gt == 0 or len_pred == 0:
            return 1.0 if passed else 0.0
        ratio = len_gt / len_pred
        return 1.0 if ratio >= 0.99 else 0.0
    except Exception:
        return 0.0


def _simulate(dag: List[dict], gt: Dict[str, Any]) -> Tuple[bool, int, int]:
    """Run dag through the deployed pipeline (or fallback) and call eval_single.

    Returns ``(passed, len_pred, len_gt)``. ``passed`` is False on any error.
    """
    eval_steps = _dag_to_eval_steps_via_pipeline(dag, gt)
    if not eval_steps:
        eval_steps = _dag_to_eval_steps(dag)
    if not eval_steps:
        return False, 0, len(gt.get("time_steps", []) or [])

    try:
        ok = bool(eval_single(eval_steps, gt))
    except Exception as exc:
        # Observability guard (docs/RL_debug.md §1 S-①c): a missing GT field
        # (e.g. KeyError('init_pos')) makes eval_single raise for EVERY sample,
        # which a bare `except` would silently turn into R_task==0 and disguise
        # as task sparsity. Count by exception identity and re-log on decade
        # boundaries so both a systemic parity bug and a rare edge case are
        # legible without flooding the log.
        key = f"{type(exc).__name__}: {exc}"
        n = _SIMULATE_EXC_COUNTS.get(key, 0) + 1
        _SIMULATE_EXC_COUNTS[key] = n
        if n == 1 or n % 10 ** int(math.log10(n)) == 0:
            logger.warning(
                "enactgraph._simulate: eval_single raised %s (occurrence #%d) -- R_task "
                "is 0 for those trajectories. If this fires for most samples, the reward "
                "ground_truth is missing a field eval_single needs (see RL_debug §1).",
                key, n,
            )
        return False, len(eval_steps), len(gt.get("time_steps", []) or [])

    return ok, len(eval_steps), len(gt.get("time_steps", []) or [])


def acc_reward(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """v1: binary acc with hard step_ratio≥0.99 gate. Kept for ``REWARD_VERSION=v1``."""
    try:
        dag = _parse_dag(predict_str)
        if dag is None:
            return 0.0
        gt = _numpy_to_python(ground_truth)
        passed, len_pred, len_gt = _simulate(dag, gt)
        if not passed:
            return 0.0
        if len_gt == 0 or len_pred == 0:
            return 1.0 if passed else 0.0
        ratio = len_gt / len_pred
        return 1.0 if ratio >= 0.99 else 0.0
    except Exception:
        return 0.0


def validator_pass_reward(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """1.0 iff the detection-only validator finds zero issues.

    Encourages SLM outputs that are self-sufficient under the executor's
    physical constraints, rather than relying on a post-hoc safety net.
    The validator never mutates the DAG, so this reward is honest.

    Also forwards the SLM's own ``[GOAL].active`` list (when present) so the
    V_PLAN_TARGET_COVERAGE detector can fire \u2014 task-blind, since the active
    list is the SLM's own output, not ground truth.
    """
    try:
        if not _ensure_pipeline_imported():
            return 0.0
        plan_text = _cot_plan_text(predict_str)
        dag = _parse_dag(plan_text)
        if dag is None:
            return 0.0
        gt = _numpy_to_python(ground_truth)
        scene_facts = _build_minimal_scene_facts(gt)
        affordance_ctx = _build_affordance_context(scene_facts)
        ops = _parse_slm_output(json.dumps(dag))
        if not ops:
            return 0.0
        goal_active = _extract_goal_active_from_predict(predict_str)
        report = _validate_operations(
            ops, affordance_ctx, scene_facts, goal_active=goal_active,
        )
        return 1.0 if report.ok else 0.0
    except Exception:
        return 0.0


def plan_expansion_ok(plan: list, ground_truth: Dict[str, Any]) -> bool:
    """True iff the op plan survives the LOCUS scoring chain's execution path.

    Mirrors the eval driver's err detection (eval_lnsdl_agentic_viki_l2.py:
    operation_validation -> convert_operations_to_graph_plan ->
    _assign_executors_to_graph -> dispatch_graph): any exception there is a
    physical-validity violation the LOCUS judge scores as an error row. This
    is the LOCUS R_struct judge (debug_sft.md section 4.5.1 follow-up): the
    rule-detector validator alone is a strict relaxation that stays silent on
    exactly the skip-step plans the benchmark's simulator rejects (e.g. a
    deposit into an openable container with no open linked via deps).

    Task-blind: consumes the GT scene (initial state), never the GT answer.
    """
    try:
        if not _ensure_pipeline_imported():
            return False
        if not plan:
            return False
        gt = _numpy_to_python(ground_truth)
        scene_facts = _build_minimal_scene_facts(gt)
        affordance_ctx = _build_affordance_context(scene_facts)
        ops = _parse_slm_output(json.dumps(plan))
        if not ops:
            return False
        report = _validate_operations(ops, affordance_ctx, scene_facts)
        if not report.ok:
            return False
        graph_plan = _convert_operations_to_graph_plan(ops, affordance_ctx, scene_facts)
        graph_plan = _assign_executors_to_graph(graph_plan, scene_facts)
        micro_steps = _dispatch_graph(graph_plan)
        return bool(micro_steps)
    except Exception:
        return False


def _extract_goal_active_from_predict(predict_str: str) -> list[str] | None:
    """Pull a normalized list of active item names out of [GOAL] for the
    GRPO-side validator. Returns None when GOAL is missing/malformed so the
    V_PLAN_TARGET_COVERAGE check becomes a no-op (parity with inference path).
    """
    try:
        goal, _, _ = _parse_cot_blocks(predict_str)
    except Exception:
        return None
    if not isinstance(goal, dict):
        return None
    active = goal.get("active")
    if not isinstance(active, list):
        return None
    items: list[str] = []
    for entry in active:
        if isinstance(entry, str):
            name = entry.strip().lower().replace(" ", "_")
        elif isinstance(entry, dict):
            raw = entry.get("item") or entry.get("name") or ""
            name = str(raw).strip().lower().replace(" ", "_")
        else:
            name = ""
        if name:
            items.append(name)
    return items


# ---------------------------------------------------------------------------
# Dataset-v2 components: three-block CoT parsing and process rewards
# ---------------------------------------------------------------------------

_COT_BLOCK_RE = re.compile(r"\[(GOAL|ASSIGN|PLAN)\]\s*\n")


def _parse_cot_blocks(predict_str: str) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, List[dict] | None]:
    """Extract (goal, assign, plan) from a three-block SLM response.

    Backward compat: a bare JSON list is treated as PLAN-only with goal/assign None.
    """
    text = predict_str.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    blocks: Dict[str, str] = {}
    matches = list(_COT_BLOCK_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[name] = text[start:end].strip()

    goal = assign = plan = None
    if blocks:
        if "GOAL" in blocks:
            try:
                v = json.loads(blocks["GOAL"])
                if isinstance(v, dict):
                    goal = v
            except Exception:
                pass
        if "ASSIGN" in blocks:
            try:
                v = json.loads(blocks["ASSIGN"])
                if isinstance(v, dict):
                    assign = v
            except Exception:
                pass
        if "PLAN" in blocks:
            try:
                v = json.loads(blocks["PLAN"])
                if isinstance(v, list):
                    plan = v
            except Exception:
                pass
    else:
        # legacy: bare JSON list
        try:
            v = json.loads(text)
            if isinstance(v, list):
                plan = v
        except Exception:
            pass
    return goal, assign, plan


def _cot_plan_text(predict_str: str) -> str:
    """Return only the [PLAN] body; fall back to original text if absent."""
    _, _, plan = _parse_cot_blocks(predict_str)
    if plan is None:
        return predict_str
    return json.dumps(plan)


def format_reward_cot(predict_str: str) -> float:
    """1.0 iff all three blocks parse and have correct types."""
    goal, assign, plan = _parse_cot_blocks(predict_str)
    if not (isinstance(goal, dict) and isinstance(assign, dict) and isinstance(plan, list)):
        return 0.0
    if not isinstance(goal.get("targets"), list):
        return 0.0
    return 1.0


def _norm_str(s: Any) -> str:
    return str(s).strip().lower() if s is not None else ""


def r_goal_reward(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """R_goal: target_items Jaccard + per-target status accuracy + active/skip consistency.

    Final = mean(target_match, status_acc, consistency).  All three subscores are in [0,1].
    """
    try:
        pred_goal, _, _ = _parse_cot_blocks(predict_str)
        if not isinstance(pred_goal, dict):
            return 0.0
        gt = _numpy_to_python(ground_truth)
        gt_goal_raw = gt.get("gt_goal_json")
        if not gt_goal_raw:
            return 0.0
        gt_goal = json.loads(gt_goal_raw) if isinstance(gt_goal_raw, str) else gt_goal_raw
        if not isinstance(gt_goal, dict):
            return 0.0

        pt = pred_goal.get("targets") or []
        gtt = gt_goal.get("targets") or []
        if not isinstance(pt, list) or not isinstance(gtt, list):
            return 0.0

        pred_items = {_norm_str(t.get("item")) for t in pt if isinstance(t, dict)}
        gt_items = {_norm_str(t.get("item")) for t in gtt if isinstance(t, dict)}
        pred_items.discard("")
        gt_items.discard("")
        if not gt_items:
            target_match = 1.0 if not pred_items else 0.0
        else:
            inter = pred_items & gt_items
            union = pred_items | gt_items
            target_match = len(inter) / len(union) if union else 0.0

        # Per-target status acc (only over GT targets that pred also lists)
        pred_status = {_norm_str(t.get("item")): _norm_str(t.get("status")) for t in pt if isinstance(t, dict)}
        gt_status = {_norm_str(t.get("item")): _norm_str(t.get("status")) for t in gtt if isinstance(t, dict)}
        if gt_status:
            hits = sum(1 for k, v in gt_status.items() if pred_status.get(k) == v)
            status_acc = hits / len(gt_status)
        else:
            status_acc = 1.0

        # active/skip consistency with status
        pred_active = {_norm_str(x) for x in (pred_goal.get("active") or [])}
        pred_skip = {_norm_str(x) for x in (pred_goal.get("skip") or [])}
        consistent = 1.0
        for it, st in pred_status.items():
            if not it:
                continue
            if st == "already_at_destination":
                if it not in pred_skip:
                    consistent -= 1.0 / max(len(pred_status), 1)
            else:
                if it not in pred_active:
                    consistent -= 1.0 / max(len(pred_status), 1)
        consistent = max(0.0, consistent)

        return float((target_match + status_acc + consistent) / 3.0)
    except Exception:
        return 0.0


def r_assign_reward(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """R_assign: slot completeness + items vs PLAN entity-set agreement.

    Final = mean(slot_match, items_match).  Both ∈ [0,1].
    """
    try:
        _, pred_assign, pred_plan = _parse_cot_blocks(predict_str)
        if not isinstance(pred_assign, dict):
            return 0.0
        gt = _numpy_to_python(ground_truth)

        # Slot universe from ground_truth.robots dict (active slots only —
        # None-typed slots are not exposed to the SLM via [ROBOTS]).
        gt_slots = {
            str(s).strip()
            for s, t in (gt.get("robots") or {}).items()
            if t is not None
        }
        pred_slots = {str(s).strip() for s in pred_assign.keys()}
        if not gt_slots:
            slot_match = 1.0
        else:
            inter = gt_slots & pred_slots
            slot_match = len(inter) / max(len(gt_slots | pred_slots), 1)

        # Items per slot consistent with PLAN entities (acquire-family ops grouped by robot)
        if pred_plan is None:
            items_match = 0.0
        else:
            plan_items: Dict[str, set] = {}
            for op in pred_plan:
                if not isinstance(op, dict):
                    continue
                if _norm_str(op.get("op_type")) in ("acquire", "local_acquire", "load_carrier"):
                    r = str(op.get("robot") or "").strip()
                    e = _norm_str(op.get("entity"))
                    if r and e:
                        plan_items.setdefault(r, set()).add(e)
            if not pred_assign:
                items_match = 0.0
            else:
                scores = []
                for slot, body in pred_assign.items():
                    if not isinstance(body, dict):
                        continue
                    declared = {_norm_str(x) for x in (body.get("items") or []) if x}
                    actual = plan_items.get(str(slot).strip(), set())
                    union = declared | actual
                    if not union:
                        scores.append(1.0)
                    else:
                        scores.append(len(declared & actual) / len(union))
                items_match = sum(scores) / len(scores) if scores else 0.0

        return float((slot_match + items_match) / 2.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public API (matches viki_2 interface)
# ---------------------------------------------------------------------------

REWARD_VERSION = os.environ.get("ENACTGRAPH_REWARD_VERSION", "v3").strip().lower()


# ---------------------------------------------------------------------------
# GRPO-v2 helpers: split plan quality into topology + token alignment.
# All helpers READ ONLY ``ground_truth`` fields written by build_dataset.py;
# they MUST NOT touch any val/test parquet.
# ---------------------------------------------------------------------------

def _gt_plan_list(ground_truth: Dict[str, Any]) -> list[dict]:
    gt = _numpy_to_python(ground_truth)
    raw = gt.get("gt_dag_json")
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [op for op in data if isinstance(op, dict)]


def _pred_plan_list(predict_str: str) -> list[dict]:
    _, _, plan = _parse_cot_blocks(predict_str)
    if not isinstance(plan, list):
        return []
    return [op for op in plan if isinstance(op, dict)]


def _op_signature(op: dict) -> tuple[str, str, str]:
    return (_norm_str(op.get("op_type")), _norm_str(op.get("entity")), _norm_str(op.get("dest")))


def _hungarian_align(pred: list[dict], gt: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy Hungarian-style alignment by op signature similarity.

    Returns list of (pred_idx, gt_idx, sim) for matched pairs (sim>0).
    """
    if not pred or not gt:
        return []
    sims: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        ps = _op_signature(p)
        for j, g in enumerate(gt):
            gs = _op_signature(g)
            score = sum(1.0 for a, b in zip(ps, gs) if a and a == b) / 3.0
            if score > 0:
                sims.append((score, i, j))
    sims.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for sc, i, j in sims:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matches.append((i, j, sc))
    return matches


def r_plan_topology(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """Hungarian-aligned typed dep-edge F1, scaled by step-count proximity."""
    try:
        pred = _pred_plan_list(predict_str)
        gt = _gt_plan_list(ground_truth)
        if not gt:
            return 1.0 if not pred else 0.0
        if not pred:
            return 0.0
        matches = _hungarian_align(pred, gt)
        if not matches:
            return 0.0
        p_to_g = {i: j for i, j, _ in matches}
        # build dep edges in aligned ID space (only count edges where BOTH
        # endpoints are aligned)
        pred_id_to_idx = {p.get("op_id"): i for i, p in enumerate(pred) if p.get("op_id")}
        gt_id_to_idx = {g.get("op_id"): j for j, g in enumerate(gt) if g.get("op_id")}
        pred_edges: set[tuple[int, int]] = set()
        for i, p in enumerate(pred):
            if i not in p_to_g:
                continue
            for d in p.get("deps") or []:
                di = pred_id_to_idx.get(d)
                if di is None or di not in p_to_g:
                    continue
                pred_edges.add((p_to_g[di], p_to_g[i]))
        gt_edges: set[tuple[int, int]] = set()
        for j, g in enumerate(gt):
            for d in g.get("deps") or []:
                dj = gt_id_to_idx.get(d)
                if dj is None:
                    continue
                gt_edges.add((dj, j))
        if not gt_edges and not pred_edges:
            edge_f1 = 1.0
        elif not gt_edges or not pred_edges:
            edge_f1 = 0.0
        else:
            tp = len(pred_edges & gt_edges)
            prec = tp / len(pred_edges)
            rec = tp / len(gt_edges)
            edge_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        # step-count proximity
        step_prox = 1.0 - abs(len(pred) - len(gt)) / max(len(gt), 1)
        step_prox = max(0.0, step_prox)
        return float(edge_f1 * step_prox)
    except Exception:
        return 0.0


def r_plan_tokens(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """Aligned op_type/entity/dest exact-match rate."""
    try:
        pred = _pred_plan_list(predict_str)
        gt = _gt_plan_list(ground_truth)
        if not gt:
            return 1.0 if not pred else 0.0
        if not pred:
            return 0.0
        matches = _hungarian_align(pred, gt)
        if not matches:
            return 0.0
        total_fields = 3 * max(len(pred), len(gt))
        hit = sum(int(sc * 3) for _, _, sc in matches)
        return float(hit / total_fields) if total_fields else 0.0
    except Exception:
        return 0.0


def format_reward_v2(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """GRPO-v2 format reward; penalize active goals with an empty PLAN.

    Empty-plan strong penalty makes V0 violations costly during GRPO rollouts.
    """
    base = format_reward_cot(predict_str)
    try:
        goal, _, plan = _parse_cot_blocks(predict_str)
        active = (goal or {}).get("active") if isinstance(goal, dict) else None
        if isinstance(active, list) and active and isinstance(plan, list) and not plan:
            return base - 0.5
    except Exception:
        pass
    return base


def r_goal_reward_v3(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """R_goal v3 (2026-05-22): emphasise active-set correctness.

    Per debug.md (Q1–Q4, §6): the dominant failure mode on the v1 1.5B
    checkpoint is mis-identifying which items belong in ``[GOAL].active``
    (either listing already-satisfied items, or omitting genuinely needed
    items). v2's R_goal averaged four signals with equal weight, diluting
    the active-set gradient. v3 keeps the same sub-scores but reweights to
    foreground the active-set Jaccard:

        target_match (0.30) + active_match (0.40) + status_acc (0.20)
        + consistency (0.10)

    target_match / status_acc / consistency are computed exactly as in
    ``r_goal_reward``; active_match is a new Jaccard over the predicted vs
    ground-truth ``active`` sets, which is the actual policy bottleneck.
    """
    try:
        pred_goal, _, _ = _parse_cot_blocks(predict_str)
        if not isinstance(pred_goal, dict):
            return 0.0
        gt = _numpy_to_python(ground_truth)
        gt_goal_raw = gt.get("gt_goal_json")
        if not gt_goal_raw:
            return 0.0
        gt_goal = json.loads(gt_goal_raw) if isinstance(gt_goal_raw, str) else gt_goal_raw
        if not isinstance(gt_goal, dict):
            return 0.0

        pt = pred_goal.get("targets") or []
        gtt = gt_goal.get("targets") or []
        if not isinstance(pt, list) or not isinstance(gtt, list):
            return 0.0

        pred_items = {_norm_str(t.get("item")) for t in pt if isinstance(t, dict)}
        gt_items = {_norm_str(t.get("item")) for t in gtt if isinstance(t, dict)}
        pred_items.discard("")
        gt_items.discard("")
        if not gt_items:
            target_match = 1.0 if not pred_items else 0.0
        else:
            inter = pred_items & gt_items
            union = pred_items | gt_items
            target_match = len(inter) / len(union) if union else 0.0

        pred_status = {_norm_str(t.get("item")): _norm_str(t.get("status")) for t in pt if isinstance(t, dict)}
        gt_status = {_norm_str(t.get("item")): _norm_str(t.get("status")) for t in gtt if isinstance(t, dict)}
        if gt_status:
            hits = sum(1 for k, v in gt_status.items() if pred_status.get(k) == v)
            status_acc = hits / len(gt_status)
        else:
            status_acc = 1.0

        # NEW: active-set Jaccard against GT — primary failure signal.
        pred_active = {_norm_str(x) for x in (pred_goal.get("active") or [])}
        pred_active.discard("")
        gt_active_raw = gt_goal.get("active") or []
        gt_active = {_norm_str(x) for x in gt_active_raw}
        gt_active.discard("")
        if not gt_active and not pred_active:
            active_match = 1.0
        elif not (gt_active | pred_active):
            active_match = 1.0
        else:
            active_match = len(gt_active & pred_active) / len(gt_active | pred_active)

        pred_skip = {_norm_str(x) for x in (pred_goal.get("skip") or [])}
        consistent = 1.0
        for it, st in pred_status.items():
            if not it:
                continue
            if st == "already_at_destination":
                if it not in pred_skip:
                    consistent -= 1.0 / max(len(pred_status), 1)
            else:
                if it not in pred_active:
                    consistent -= 1.0 / max(len(pred_status), 1)
        consistent = max(0.0, consistent)

        return float(
            0.30 * target_match
            + 0.40 * active_match
            + 0.20 * status_acc
            + 0.10 * consistent
        )
    except Exception:
        return 0.0


def compute_score(predict_str: str, ground_truth: Dict[str, Any]) -> float:
    """Dispatch to v1/v2/v3 based on ``ENACTGRAPH_REWARD_VERSION``."""
    if REWARD_VERSION == "v3":
        # v3 (2026-05-22): re-weight to address the dominant failure mode
        # identified in docs/debug.md §6 — mis-identification of
        # [GOAL].active items. Two structural changes vs v2:
        #   1. R_validator weight 0.15 → 0.05, and the hard gate on
        #      R_plan_topology / R_plan_tokens is REMOVED. Empirically the
        #      validator misses real failures (per docs/debug.md C2) and
        #      occasionally false-positives on legitimate plans
        #      (cardboardbox V1b, see §6 F1); a hard gate amplifies its
        #      noise. Plan rewards stay independent of validator.
        #   2. R_goal weight 0.15 → 0.30 and switched to
        #      ``r_goal_reward_v3`` which scores [GOAL].active Jaccard
        #      against ground truth as the dominant sub-signal.
        #
        # Composition (sums to 1.0):
        #   0.05·R_format + 0.30·R_goal_v3 + 0.10·R_assign
        #   + 0.05·R_validator + 0.20·R_plan_topology + 0.30·R_plan_tokens
        fmt = format_reward_v2(predict_str, ground_truth)
        rgoal = r_goal_reward_v3(predict_str, ground_truth)
        rassign = r_assign_reward(predict_str, ground_truth)
        rvalid = validator_pass_reward(predict_str, ground_truth)
        rtopo = r_plan_topology(predict_str, ground_truth)
        rtok = r_plan_tokens(predict_str, ground_truth)
        return (
            0.05 * fmt
            + 0.30 * rgoal
            + 0.10 * rassign
            + 0.05 * rvalid
            + 0.20 * rtopo
            + 0.30 * rtok
        )

    if REWARD_VERSION == "v2":
        fmt = format_reward_v2(predict_str, ground_truth)
        rgoal = r_goal_reward(predict_str, ground_truth)
        rassign = r_assign_reward(predict_str, ground_truth)
        # Pass the full predict_str so V_PLAN_TARGET_COVERAGE can read [GOAL].
        # validator_pass_reward extracts the plan body internally via _cot_plan_text.
        rvalid = validator_pass_reward(predict_str, ground_truth)
        if rvalid >= 1.0:
            rtopo = r_plan_topology(predict_str, ground_truth)
            rtok = r_plan_tokens(predict_str, ground_truth)
        else:
            rtopo = 0.0
            rtok = 0.0
        return (
            0.05 * fmt
            + 0.15 * rgoal
            + 0.10 * rassign
            + 0.15 * rvalid
            + 0.25 * rtopo
            + 0.30 * rtok
        )

    if REWARD_VERSION == "v1":
        fmt = format_reward_v1(predict_str, ground_truth)
        acc = acc_reward_v1(predict_str, ground_truth)
        return 0.1 * fmt + 0.9 * acc

    raise ValueError(
        f"Unsupported ENACTGRAPH_REWARD_VERSION={REWARD_VERSION!r}; expected 'v1', 'v2', or 'v3'."
    )
