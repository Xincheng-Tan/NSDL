from __future__ import annotations

from dataclasses import replace

from nsdl.dispatcher import dispatch_graph
from nsdl.catalog import normalize_robot_name
from nsdl.macro_graph import build_macro_graph, collect_selected_robot_types
from nsdl.profiles import (
    VIKI_L2_PROFILE,
    get_benchmark_profile,
    use_benchmark_profile,
)
from nsdl.profiles.base import BenchmarkProfile
from nsdl.scene_facts_refiner import build_scene_facts_refiner
from nsdl.scene_parser import build_scene_parser
from nsdl.schema import (
    BenchmarkSample,
    DispatcherPayload,
    PipelineResult,
    StructuredDetections,
    normalize_benchmark_name,
)
from nsdl.slm_backends.base import SLMBackend
from nsdl.yolo_detector import build_structured_detector


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        sample: BenchmarkSample,
        yolo_detections: StructuredDetections | dict | list[dict] | None,
        initial_scene_facts=None,
        scene_facts=None,
        original_exc: Exception,
    ) -> None:
        super().__init__(str(original_exc))
        self.stage = stage
        self.sample = sample
        self.yolo_detections = yolo_detections
        self.initial_scene_facts = initial_scene_facts
        self.scene_facts = scene_facts
        self.original_exc = original_exc


def _filter_visible_robot_inventory(
    sample: BenchmarkSample,
    detections: StructuredDetections | None,
) -> BenchmarkSample:
    if (
        sample.robot_inventory_scope != "possible"
        or detections is None
        or not detections.visible_robots
    ):
        return sample

    visible_robot_types = {
        normalize_robot_name(group.label)
        for group in detections.visible_robots
    }
    visible_robot_types.discard(None)
    if not visible_robot_types:
        return sample

    filtered_inventory = {
        slot_id: robot_type
        for slot_id, robot_type in sample.robot_inventory.items()
        if robot_type in visible_robot_types
    }
    if not filtered_inventory:
        return sample
    return replace(sample, robot_inventory=filtered_inventory)


def _resolve_sample_profile(
    sample: BenchmarkSample,
    profile_override: BenchmarkProfile | None,
) -> BenchmarkProfile:
    if profile_override is not None:
        return profile_override
    return get_benchmark_profile(normalize_benchmark_name(sample.benchmark))


class NSDLPipeline:
    def __init__(
        self,
        *,
        scene_parser_backend: str = "rule",
        scene_facts_refiner_backend: str = "none",
        graph_builder_mode: str = "acc_dag",
        detector_backend: str = "none",
        detector_run_dir: str | None = None,
        detector_weights_path: str | None = None,
        detector_data_yaml_path: str | None = None,
        detector_confidence_threshold: float = 0.25,
        detector_iou_threshold: float = 0.70,
        detector_max_det: int = 64,
        detector_device: str | None = None,
        endpoint_index: int = 0,
        api_base_url: str | None = None,
        api_model: str | None = None,
        scene_facts_refiner_cache_dir: str | None = None,
        use_local: bool = False,
        slm_backend: SLMBackend | None = None,
        max_validator_retries: int = 0,
        profile: BenchmarkProfile | None = None,
    ) -> None:
        self.graph_builder_mode = graph_builder_mode
        self.endpoint_index = endpoint_index
        self.api_base_url = api_base_url
        self.api_model = api_model
        self.use_local = use_local
        self.slm_backend = slm_backend
        self.max_validator_retries = max_validator_retries
        self.profile = profile or VIKI_L2_PROFILE
        self._profile_override = profile
        self.scene_parser = build_scene_parser(
            scene_parser_backend,
        )
        self.scene_facts_refiner = build_scene_facts_refiner(
            scene_facts_refiner_backend,
            endpoint_index=endpoint_index,
            api_base_url=api_base_url,
            api_model=api_model,
            cache_dir=scene_facts_refiner_cache_dir,
        )
        self.structured_detector = build_structured_detector(
            detector_backend,
            run_dir=detector_run_dir,
            weights_path=detector_weights_path,
            data_yaml_path=detector_data_yaml_path,
            confidence_threshold=detector_confidence_threshold,
            iou_threshold=detector_iou_threshold,
            max_det=detector_max_det,
            device=detector_device,
        )

    def run_l2(
        self,
        sample: BenchmarkSample,
        *,
        yolo_detections: StructuredDetections | dict | list[dict] | None = None,
    ) -> PipelineResult:
        structured_detections: StructuredDetections | None = None
        yolo_payload = yolo_detections
        runtime_sample = sample
        initial_scene_facts = None
        scene_facts = None
        stage = "profile"
        try:
            profile = _resolve_sample_profile(sample, self._profile_override)
            with use_benchmark_profile(profile):
                stage = "detector"
                if yolo_detections is None and self.structured_detector and sample.image_bytes:
                    structured_detections = self.structured_detector.detect(sample.image_bytes)
                    yolo_payload = structured_detections
                elif isinstance(yolo_detections, StructuredDetections):
                    structured_detections = yolo_detections
                    yolo_payload = structured_detections

                stage = "robot_inventory_filter"
                runtime_sample = _filter_visible_robot_inventory(sample, structured_detections)

                stage = "scene_parser"
                initial_scene_facts = self.scene_parser.parse(runtime_sample)

                stage = "scene_facts_refiner"
                scene_facts = self.scene_facts_refiner.refine(
                    runtime_sample,
                    initial_scene_facts,
                    yolo_detections=yolo_payload,
                )

                stage = "macro_graph_builder"
                graph_plan = build_macro_graph(
                    scene_facts,
                    builder_mode=self.graph_builder_mode,
                    endpoint_index=self.endpoint_index,
                    api_base_url=self.api_base_url,
                    api_model=self.api_model,
                    use_local=self.use_local,
                    slm_backend=self.slm_backend,
                    max_validator_retries=self.max_validator_retries,
                    image_bytes=runtime_sample.image_bytes,
                )

                stage = "micro_planner"
                micro_steps = dispatch_graph(graph_plan)

                stage = "dispatcher_emitter"
                dispatcher_output = profile.dispatcher_emitter.emit(
                    DispatcherPayload(
                        sample=runtime_sample,
                        initial_scene_facts=initial_scene_facts,
                        scene_facts=scene_facts,
                        graph_plan=graph_plan,
                        micro_steps=micro_steps,
                    )
                )

                stage = "robot_selection"
                selected_robot_types = collect_selected_robot_types(graph_plan)
                return PipelineResult(
                    sample=runtime_sample,
                    initial_scene_facts=initial_scene_facts,
                    scene_facts=scene_facts,
                    yolo_detections=structured_detections,
                    graph_plan=graph_plan,
                    selected_robot_types=selected_robot_types,
                    micro_steps=micro_steps,
                    dispatcher_output=dispatcher_output,
                )
        except Exception as exc:
            if isinstance(exc, PipelineExecutionError):
                raise
            raise PipelineExecutionError(
                stage=stage,
                sample=runtime_sample,
                yolo_detections=yolo_payload,
                initial_scene_facts=initial_scene_facts,
                scene_facts=scene_facts,
                original_exc=exc,
            ) from exc
