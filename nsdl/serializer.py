from __future__ import annotations

from typing import Any

from nsdl.profiles.viki_l2.emitter import VIKI_L2_DISPATCHER_EMITTER
from nsdl.schema import (
    DetectionGroup,
    DetectionInstance,
    MicroStep,
    PipelineResult,
    SceneFacts,
    StructuredDetections,
)


def to_l1_answer(result: PipelineResult) -> list[str]:
    return list(result.selected_robot_types)


def to_l2_answer(micro_steps: list[MicroStep]) -> list[dict]:
    return VIKI_L2_DISPATCHER_EMITTER.emit(micro_steps)


def to_dispatcher_output(result: PipelineResult) -> Any:
    if result.dispatcher_output is not None:
        return result.dispatcher_output
    return to_l2_answer(result.micro_steps)


def _drop_empty_values(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        compacted[key] = value
    return compacted


def _compact_task_specific_hints(task_specific_hints: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        "target_semantic_state_disabled": task_specific_hints.get("target_semantic_state_disabled"),
        "location_predicates": task_specific_hints.get("location_predicates"),
        "instruction_named_fruits": task_specific_hints.get("instruction_named_fruits"),
        "yolo_visible_fruits": task_specific_hints.get("yolo_visible_fruits"),
    }
    return _drop_empty_values(compacted)


def _compact_llm_refinement_response(parser_notes: dict[str, Any]) -> dict[str, Any] | None:
    response = parser_notes.get("llm_refinement_response")
    if not isinstance(response, dict):
        return None
    initial_scene_facts = parser_notes.get("initial_scene_facts", {})
    if not isinstance(initial_scene_facts, dict):
        initial_scene_facts = {}

    delta: dict[str, Any] = {}
    for key, value in response.items():
        if key in {"target_items", "target_status", "primary_object", "secondary_objects", "targets_in_cabinet"}:
            continue
        if initial_scene_facts.get(key) != value:
            delta[key] = value
    return delta or None


def _compact_parser_notes(parser_notes: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        "stage": parser_notes.get("stage"),
        "image_used": parser_notes.get("image_used"),
        "refiner_backend": parser_notes.get("refiner_backend"),
        "inspect_mode": parser_notes.get("inspect_mode"),
        "destination_pairs": parser_notes.get("destination_pairs"),
        "support_targets_internal": parser_notes.get("support_targets_internal"),
        "container_targets_internal": parser_notes.get("container_targets_internal"),
        "intent_hypotheses": parser_notes.get("intent_hypotheses"),
        "mentioned_entity_spans": parser_notes.get("mentioned_entity_spans"),
        "target_items_disabled": parser_notes.get("target_items_disabled"),
        "target_semantic_state_disabled": parser_notes.get("target_semantic_state_disabled"),
        "task_specific_hints": _compact_task_specific_hints(
            parser_notes.get("task_specific_hints", {})
            if isinstance(parser_notes.get("task_specific_hints"), dict)
            else {}
        ),
        "llm_refinement_delta": _compact_llm_refinement_response(parser_notes),
        "format_retry": parser_notes.get("format_retry"),
        "robot_anchor_geometry": parser_notes.get("robot_anchor_geometry"),
        "table_overlap_guard": parser_notes.get("table_overlap_guard"),
        "cabinet_overlap_guard": parser_notes.get("cabinet_overlap_guard"),
        "uncertainty_guard": parser_notes.get("uncertainty_guard"),
    }
    return _drop_empty_values(compacted)


def _scene_facts_to_debug_dict(scene_facts: SceneFacts) -> dict[str, Any]:
    return _drop_empty_values({
        "parser_backend": scene_facts.parser_backend,
        "primary_object": scene_facts.primary_object,
        "destination": scene_facts.destination,
        "destination_pairs": scene_facts.destination_pairs,
        "secondary_objects": scene_facts.secondary_objects,
        "target_items": scene_facts.target_items,
        "destination_candidates": scene_facts.destination_candidates,
        "source_locations": scene_facts.source_locations,
        "fallback_locations": scene_facts.fallback_locations,
        "targets_in_cabinet": scene_facts.targets_in_cabinet,
        "tool_items": scene_facts.tool_items,
        "activation_targets": scene_facts.activation_targets,
        "inspection_targets": scene_facts.inspection_targets,
        "inspect_mode": scene_facts.inspect_mode,
        "parsed_intents": scene_facts.parsed_intents,
        "mentioned_objects": scene_facts.mentioned_objects,
        "robot_anchor_status": scene_facts.robot_anchor_status,
        "parser_notes": _compact_parser_notes(scene_facts.parser_notes),
        "extras": scene_facts.extras,
    })


def scene_facts_to_debug_dict(scene_facts: SceneFacts | None) -> dict[str, Any] | None:
    if scene_facts is None:
        return None
    return _scene_facts_to_debug_dict(scene_facts)


def _detection_instance_to_dict(instance: DetectionInstance) -> dict:
    return {
        "label": instance.label,
        "category": instance.category,
        "confidence": instance.confidence,
        "bbox_xyxy": list(instance.bbox_xyxy),
        "center_xy": list(instance.center_xy),
        "region": instance.region,
        "rank_within_label": instance.rank_within_label,
    }


def _detection_group_to_dict(group: DetectionGroup) -> dict:
    return {
        "label": group.label,
        "category": group.category,
        "count": group.count,
        "best_confidence": group.best_confidence,
        "regions": list(group.regions),
        "instances": [
            _detection_instance_to_dict(instance)
            for instance in group.instances
        ],
    }


def _compact_detection_group_payload(group: DetectionGroup) -> dict[str, Any]:
    return {
        "label": group.label,
        "category": group.category,
        "count": group.count,
        "best_confidence": group.best_confidence,
        "regions": list(group.regions),
        "instances": [
            _detection_instance_to_dict(instance)
            for instance in group.instances
        ],
    }


def _default_label_semantics(supported_labels: list[str] | None) -> dict[str, str] | None:
    labels = set(supported_labels or [])
    semantics: dict[str, str] = {}
    if "table" in labels:
        semantics["table"] = (
            "Detector label for an independent tabletop in the central scene region. "
            "Use it as the primary visual anchor for table-related geometric reasoning."
        )
    if "cabinet" in labels:
        semantics["cabinet"] = "Detector label for cabinet-like enclosed storage surfaces or doors."
    return semantics or None


def _default_region_overlap_notes(supported_labels: list[str] | None) -> list[str] | None:
    labels = set(supported_labels or [])
    if "table" in labels:
        return [
            (
                "table and kitchen island area are not mutually exclusive semantic labels; "
                "the same surface or object placement can satisfy both."
            )
        ]
    return None


def _structured_detections_to_dict(
    detections: StructuredDetections | None,
) -> dict[str, Any] | None:
    if detections is None:
        return None
    prompt_payload = dict(detections.prompt_payload)
    supported_labels = list(detections.supported_labels)
    return _drop_empty_values({
        "detector_backend": detections.detector_backend,
        "model_run": prompt_payload.get("model_run") or detections.run_name,
        "image_size": prompt_payload.get("image_size")
        or {
            "width": detections.image_width,
            "height": detections.image_height,
        },
        "thresholds": prompt_payload.get("thresholds")
        or {
            "confidence": detections.confidence_threshold,
                "iou": detections.iou_threshold,
        },
        "supported_labels": supported_labels,
        "visible_labels": list(detections.visible_labels),
        "coverage_note": prompt_payload.get("coverage_note"),
        "label_semantics": prompt_payload.get("label_semantics")
        or _default_label_semantics(supported_labels),
        "region_overlap_notes": prompt_payload.get("region_overlap_notes")
        or _default_region_overlap_notes(supported_labels),
        "table_item_overlap": prompt_payload.get("table_item_overlap"),
        "grouped_detections": [
            _compact_detection_group_payload(group)
            for group in detections.grouped_detections
        ],
    })


def structured_detections_to_debug_dict(
    detections: StructuredDetections | None,
) -> dict[str, Any] | None:
    return _structured_detections_to_dict(detections)


def _graph_node_to_debug_dict(node: dict[str, Any]) -> dict[str, Any]:
    return _drop_empty_values({
        "node_id": node.get("node_id"),
        "node_kind": node.get("node_kind"),
        "semantic_goal": node.get("semantic_goal"),
        "executor_slot": node.get("executor_slot"),
        "executor_robot_type": node.get("executor_robot_type"),
        "object_name": node.get("object_name"),
        "destination": node.get("destination"),
        "object_bindings": node.get("object_bindings"),
        "required_affordances_hard": node.get("required_affordances_hard"),
        "required_affordances_soft": node.get("required_affordances_soft"),
        "required_action_primitives": node.get("required_action_primitives"),
        "required_constraints": node.get("required_constraints"),
        "predecessors": node.get("predecessors"),
        "goal_predicates": node.get("goal_predicates"),
        "start_predicates_hint": node.get("start_predicates_hint"),
        "state_exports": node.get("state_exports"),
        "output_interface": node.get("output_interface"),
        "completion_check": node.get("completion_check"),
        "dispatcher_scaffold": node.get("dispatcher_scaffold"),
    })


def compact_result_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(artifact)
    compacted.pop("image_path", None)

    for scene_key in ("initial_scene_facts", "scene_facts"):
        scene_payload = compacted.get(scene_key)
        if not isinstance(scene_payload, dict):
            continue
        parser_notes = scene_payload.get("parser_notes", {})
        compacted[scene_key] = _drop_empty_values({
            "parser_backend": scene_payload.get("parser_backend"),
            "primary_object": scene_payload.get("primary_object"),
            "destination": scene_payload.get("destination"),
            "secondary_objects": scene_payload.get("secondary_objects"),
            "target_items": scene_payload.get("target_items"),
            "destination_candidates": scene_payload.get("destination_candidates"),
            "source_locations": scene_payload.get("source_locations"),
            "fallback_locations": scene_payload.get("fallback_locations"),
            "targets_in_cabinet": scene_payload.get("targets_in_cabinet"),
            "tool_items": scene_payload.get("tool_items"),
            "activation_targets": scene_payload.get("activation_targets"),
            "inspection_targets": scene_payload.get("inspection_targets"),
            "inspect_mode": scene_payload.get("inspect_mode"),
            "parsed_intents": scene_payload.get("parsed_intents"),
            "mentioned_objects": scene_payload.get("mentioned_objects"),
            "robot_anchor_status": scene_payload.get("robot_anchor_status"),
            "parser_notes": _compact_parser_notes(parser_notes if isinstance(parser_notes, dict) else {}),
            "extras": scene_payload.get("extras"),
        })

    yolo_payload = compacted.get("yolo_detections")
    if isinstance(yolo_payload, dict):
        supported_labels = yolo_payload.get("supported_labels")
        if not isinstance(supported_labels, list):
            supported_labels = []
        compacted["yolo_detections"] = _drop_empty_values({
            "detector_backend": yolo_payload.get("detector_backend"),
            "model_run": yolo_payload.get("model_run") or yolo_payload.get("run_name"),
            "image_size": yolo_payload.get("image_size")
            or {
                "width": yolo_payload.get("image_width"),
                "height": yolo_payload.get("image_height"),
            },
            "thresholds": yolo_payload.get("thresholds")
            or {
                "confidence": yolo_payload.get("confidence_threshold"),
                "iou": yolo_payload.get("iou_threshold"),
            },
            "supported_labels": supported_labels,
            "visible_labels": yolo_payload.get("visible_labels"),
            "coverage_note": yolo_payload.get("coverage_note"),
            "label_semantics": yolo_payload.get("label_semantics")
            or _default_label_semantics(supported_labels),
            "region_overlap_notes": yolo_payload.get("region_overlap_notes")
            or _default_region_overlap_notes(supported_labels),
            "table_item_overlap": yolo_payload.get("table_item_overlap"),
            "grouped_detections": yolo_payload.get("grouped_detections"),
        })

    graph_plan = compacted.get("graph_plan")
    if isinstance(graph_plan, dict):
        compacted["graph_plan"] = _drop_empty_values({
            "task_family": graph_plan.get("task_family"),
            "build_mode": graph_plan.get("build_mode"),
            "family_shortcut_used": graph_plan.get("family_shortcut_used"),
            "compiler_notes": graph_plan.get("compiler_notes"),
            "builder_trace": graph_plan.get("builder_trace"),
            "nodes": [
                _graph_node_to_debug_dict(node)
                for node in graph_plan.get("nodes", [])
                if isinstance(node, dict)
            ],
        })

    return compacted


def to_debug_dict(result: PipelineResult) -> dict:
    return compact_result_artifact({
        "sample": {
            "level": result.sample.level,
            "split": result.sample.split,
            "sample_index": result.sample.sample_index,
            "task_id": result.sample.task_id,
            "task_name": result.sample.task_name,
            "instruction": result.sample.instruction,
            "robot_inventory": result.sample.robot_inventory,
            "robot_inventory_scope": result.sample.robot_inventory_scope,
            "image_path": result.sample.image_path,
            "image_byte_size": len(result.sample.image_bytes or b""),
        },
        "initial_scene_facts": _scene_facts_to_debug_dict(result.initial_scene_facts),
        "scene_facts": _scene_facts_to_debug_dict(result.scene_facts),
        "yolo_detections": _structured_detections_to_dict(result.yolo_detections),
        "graph_plan": {
            "task_family": result.graph_plan.task_family,
            "build_mode": result.graph_plan.build_mode,
            "family_shortcut_used": result.graph_plan.family_shortcut_used,
            "compiler_notes": result.graph_plan.compiler_notes,
            "builder_trace": result.graph_plan.builder_trace,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_kind": node.node_kind,
                    "semantic_goal": node.semantic_goal,
                    "executor_slot": node.executor_slot,
                    "executor_robot_type": node.executor_robot_type,
                    "object_name": node.object_name,
                    "destination": node.destination,
                    "object_bindings": node.object_bindings,
                    "required_affordances_hard": node.required_affordances_hard,
                    "required_affordances_soft": node.required_affordances_soft,
                    "required_action_primitives": node.required_action_primitives,
                    "required_constraints": node.required_constraints,
                    "predecessors": node.predecessors,
                    "goal_predicates": node.goal_predicates,
                    "start_predicates_hint": node.start_predicates_hint,
                    "state_exports": node.state_exports,
                    "output_interface": node.output_interface,
                    "completion_check": node.completion_check,
                    "dispatcher_scaffold": node.dispatcher_scaffold,
                }
                for node in result.graph_plan.nodes
            ],
        },
        "selected_robots": to_l1_answer(result),
        "plan": to_dispatcher_output(result),
    })
