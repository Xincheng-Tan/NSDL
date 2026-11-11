from __future__ import annotations

import ast
import traceback
from typing import Any

from nsdl.benchmark_scoring import (
    compute_locus_l2_acc,
    compute_viki_l1_score,
    compute_viki_l2_score,
)
from nsdl.prompt_adapter import build_sample_stem, convert_arrays_to_native
from nsdl.schema import BenchmarkSample, PipelineResult
from nsdl.serializer import (
    compact_result_artifact,
    scene_facts_to_debug_dict,
    structured_detections_to_debug_dict,
    to_debug_dict,
    to_dispatcher_output,
    to_l1_answer,
)
from nsdl.validator import ValidationResult, validate_l2_plan


def get_sample_ground_truth(sample: BenchmarkSample) -> Any:
    if sample.ground_truth is not None:
        return sample.ground_truth
    if sample.raw_sample is None:
        return None
    reward_model = sample.raw_sample.get("reward_model", {})
    return convert_arrays_to_native(reward_model.get("ground_truth"))


def get_l1_ground_truth_robot_types(sample: BenchmarkSample) -> list[str] | None:
    raw_ground_truth = get_sample_ground_truth(sample)
    if raw_ground_truth is None:
        return None
    if isinstance(raw_ground_truth, dict):
        return [
            robot_type
            for robot_type in raw_ground_truth.get("robots", {}).values()
            if robot_type
        ]
    if isinstance(raw_ground_truth, list):
        return [str(robot_type) for robot_type in raw_ground_truth if str(robot_type).strip()]
    if isinstance(raw_ground_truth, str):
        try:
            parsed = ast.literal_eval(raw_ground_truth)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, list):
            return [str(robot_type) for robot_type in parsed if str(robot_type).strip()]
    return None


def build_validation_payload(validation_result: ValidationResult | None) -> dict[str, Any] | None:
    if validation_result is None:
        return None
    return {
        "success": validation_result.success,
        "error_desc": validation_result.error_desc,
        "eval_commands": validation_result.eval_commands,
    }


def build_result_artifact(
    result: PipelineResult,
    *,
    validate_l2: bool = True,
) -> dict[str, Any]:
    artifact = to_debug_dict(result)
    artifact["sample"]["artifact_stem"] = build_sample_stem(result.sample)
    artifact["selected_robots"] = to_l1_answer(result)
    artifact["plan"] = to_dispatcher_output(result)

    raw_ground_truth = get_sample_ground_truth(result.sample)
    artifact["ground_truth"] = raw_ground_truth
    artifact["ground_truth_selected_robots"] = get_l1_ground_truth_robot_types(result.sample)
    # VIKI-L1's official judge scores an unordered SET of robot names, so this
    # flag must be order-insensitive too. It used to compare the lists directly,
    # which under-counted every multi-robot hit whose slot order differed from
    # the ground truth's -- 47 / 2043 rows on the v3.3 run, all |A|=2. The
    # official per-row score in benchmark_scores was already correct, so the two
    # disagreed and only the official one was trustworthy.
    artifact["selected_robots_exact_match"] = (
        set(artifact["ground_truth_selected_robots"]) == set(artifact["selected_robots"] or [])
        if artifact["ground_truth_selected_robots"] is not None
        else None
    )

    validation_result = None
    if validate_l2 and isinstance(raw_ground_truth, dict):
        validation_result = validate_l2_plan(artifact["plan"], raw_ground_truth)
    artifact["validation"] = build_validation_payload(validation_result)

    benchmark_scores: dict[str, float | None] = {
        "viki_l1_robot_activation": None,
        "viki_l2_plan_score": None,
        "locus_l2_acc": None,
    }
    if artifact["ground_truth_selected_robots"] is not None:
        benchmark_scores["viki_l1_robot_activation"] = compute_viki_l1_score(
            artifact["selected_robots"],
            artifact["ground_truth_selected_robots"],
        )
    if isinstance(raw_ground_truth, dict):
        benchmark_scores["viki_l2_plan_score"] = compute_viki_l2_score(
            artifact["plan"],
            raw_ground_truth,
        )
        # Both columns are always emitted so the ruler is a reporting choice
        # rather than a run-time one. LOCUS must be read off ``locus_l2_acc``:
        # the official judge early-returns 0 on null-plan items (73/560 of the
        # frozen splits), which would score a correct "nothing to do" as wrong,
        # and the strong baselines are scored the same way. On non-null plans
        # the two differ only by the official format term.
        benchmark_scores["locus_l2_acc"] = compute_locus_l2_acc(
            artifact["plan"],
            raw_ground_truth,
        )
    artifact["benchmark_scores"] = benchmark_scores
    return artifact


def build_error_artifact(
    sample: BenchmarkSample,
    exc: Exception,
) -> dict[str, Any]:
    root_exc = getattr(exc, "original_exc", exc)
    if not isinstance(root_exc, Exception):
        root_exc = exc

    runtime_sample = getattr(exc, "sample", sample)
    if not isinstance(runtime_sample, BenchmarkSample):
        runtime_sample = sample

    initial_scene_facts = scene_facts_to_debug_dict(getattr(exc, "initial_scene_facts", None))
    scene_facts = scene_facts_to_debug_dict(getattr(exc, "scene_facts", None))
    yolo_detections = getattr(exc, "yolo_detections", None)
    if isinstance(yolo_detections, dict):
        yolo_debug_payload = yolo_detections
    elif isinstance(yolo_detections, list):
        yolo_debug_payload = yolo_detections
    else:
        yolo_debug_payload = structured_detections_to_debug_dict(yolo_detections)

    raw_ground_truth = get_sample_ground_truth(runtime_sample)
    ground_truth_selected_robots = get_l1_ground_truth_robot_types(runtime_sample)
    # `locus_l2_acc` is hard-zeroed here rather than routed through
    # `compute_locus_l2_acc`, and the difference is not cosmetic. A crashed row
    # has no plan, and on a null-plan item (goal already satisfied, correct
    # answer = no actions) that function scores an empty plan **1.0** -- so
    # rescoring a crash would hand it full credit for an answer it never gave.
    # 0.0 mirrors `viki_l2_plan_score` directly above, under the same GT guard.
    # Emitting the key at all is what keeps "absent" meaning "written by a build
    # that predates this column" instead of silently also meaning "errored".
    benchmark_scores: dict[str, float | None] = {
        "viki_l1_robot_activation": 0.0 if ground_truth_selected_robots is not None else None,
        "viki_l2_plan_score": 0.0 if isinstance(raw_ground_truth, dict) else None,
        "locus_l2_acc": 0.0 if isinstance(raw_ground_truth, dict) else None,
    }
    return compact_result_artifact({
        "sample": {
            "level": runtime_sample.level,
            "split": runtime_sample.split,
            "sample_index": runtime_sample.sample_index,
            "task_id": runtime_sample.task_id,
            "task_name": runtime_sample.task_name,
            "instruction": runtime_sample.instruction,
            "robot_inventory": runtime_sample.robot_inventory,
            "robot_inventory_scope": runtime_sample.robot_inventory_scope,
            "artifact_stem": build_sample_stem(runtime_sample),
            "image_path": runtime_sample.image_path,
            "image_byte_size": len(runtime_sample.image_bytes or b""),
        },
        "initial_scene_facts": initial_scene_facts,
        "scene_facts": scene_facts,
        "yolo_detections": yolo_debug_payload,
        "graph_plan": None,
        "selected_robots": [],
        "plan": [],
        "ground_truth": raw_ground_truth,
        "ground_truth_selected_robots": ground_truth_selected_robots,
        "selected_robots_exact_match": False if ground_truth_selected_robots is not None else None,
        "validation": None,
        "benchmark_scores": benchmark_scores,
        "error": {
            "type": root_exc.__class__.__name__,
            "message": str(root_exc),
            "pipeline_stage": getattr(exc, "stage", None),
            "traceback": traceback.format_exc(),
        },
    })
