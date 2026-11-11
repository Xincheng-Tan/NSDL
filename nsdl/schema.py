from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_BENCHMARK = "viki_l2"


def normalize_benchmark_name(benchmark: str | None) -> str:
    normalized = str(benchmark or DEFAULT_BENCHMARK).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return normalized or DEFAULT_BENCHMARK


@dataclass
class BenchmarkSample:
    level: str
    split: str
    sample_index: int
    task_id: str | None
    task_name: str | None
    instruction: str
    system_prompt: str
    image_bytes: bytes | None = None
    robot_inventory: dict[str, str | None] = field(default_factory=dict)
    robot_inventory_scope: str = "available"
    image_path: str | None = None
    ground_truth: dict[str, Any] | None = None
    raw_sample: dict[str, Any] | None = None
    raw_observation: dict[str, Any] | None = None
    benchmark: str = DEFAULT_BENCHMARK

    def __post_init__(self) -> None:
        self.benchmark = normalize_benchmark_name(self.benchmark)
        if self.robot_inventory is None:
            self.robot_inventory = {}


@dataclass
class RobotInstance:
    slot_id: str
    robot_type: str


@dataclass
class SceneFacts:
    parser_backend: str
    instruction: str
    image_available: bool
    robot_inventory: list[RobotInstance]
    primary_object: str | None
    destination: str | None
    destination_pairs: list[dict[str, str]] | None = field(default_factory=list)
    secondary_objects: list[str] | None = field(default_factory=list)
    target_items: list[str] | None = field(default_factory=list)
    destination_candidates: list[str] | None = field(default_factory=list)
    source_locations: list[str] | None = field(default_factory=list)
    fallback_locations: list[str] | None = field(default_factory=list)
    targets_in_cabinet: list[str] | None = field(default_factory=list)
    tool_items: list[str] | None = field(default_factory=list)
    activation_targets: list[str] | None = field(default_factory=list)
    inspection_targets: list[str] | None = field(default_factory=list)
    inspect_mode: str | None = None
    parsed_intents: list[str] | None = field(default_factory=list)
    mentioned_objects: list[str] | None = field(default_factory=list)
    robot_anchor_status: dict[str, str] | None = field(default_factory=dict)
    field_sources: dict[str, str] | None = field(default_factory=dict)
    parser_notes: dict[str, Any] | None = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        list_fields = (
            "destination_pairs",
            "secondary_objects",
            "target_items",
            "destination_candidates",
            "source_locations",
            "fallback_locations",
            "targets_in_cabinet",
            "tool_items",
            "activation_targets",
            "inspection_targets",
            "parsed_intents",
            "mentioned_objects",
        )
        for field_name in list_fields:
            if getattr(self, field_name) is None:
                setattr(self, field_name, [])

        dict_fields = (
            "robot_anchor_status",
            "field_sources",
            "parser_notes",
            "extras",
        )
        for field_name in dict_fields:
            if getattr(self, field_name) is None:
                setattr(self, field_name, {})


@dataclass
class DetectionInstance:
    label: str
    category: str
    confidence: int
    bbox_xyxy: list[int]
    center_xy: list[int]
    region: str
    rank_within_label: int = 1


@dataclass
class DetectionGroup:
    label: str
    category: str
    count: int
    best_confidence: int
    regions: list[str] = field(default_factory=list)
    instances: list[DetectionInstance] = field(default_factory=list)


@dataclass
class StructuredDetections:
    detector_backend: str
    model_path: str | None
    run_name: str | None
    image_width: int
    image_height: int
    confidence_threshold: int
    iou_threshold: int
    supported_labels: list[str] = field(default_factory=list)
    visible_labels: list[str] = field(default_factory=list)
    grouped_detections: list[DetectionGroup] = field(default_factory=list)
    visible_robots: list[DetectionGroup] = field(default_factory=list)
    visible_entities: list[DetectionGroup] = field(default_factory=list)
    prompt_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AffordanceNode:
    node_id: str
    node_kind: str
    semantic_goal: str
    object_name: str | None
    destination: str | None
    required_affordances_hard: list[str]
    executor_slot: str | None = None
    executor_robot_type: str | None = None
    object_bindings: dict[str, str] = field(default_factory=dict)
    required_affordances_soft: list[str] = field(default_factory=list)
    required_action_primitives: list[str] = field(default_factory=list)
    required_constraints: dict[str, Any] = field(default_factory=dict)
    predecessors: list[str] = field(default_factory=list)
    goal_predicates: list[str] = field(default_factory=list)
    start_predicates_hint: list[str] = field(default_factory=list)
    state_exports: list[str] = field(default_factory=list)
    output_interface: str | None = None
    completion_check: dict[str, Any] = field(default_factory=dict)
    dispatcher_scaffold: str | None = None


@dataclass
class GraphPlan:
    task_family: str | None
    build_mode: str
    family_shortcut_used: bool
    nodes: list[AffordanceNode]
    compiler_notes: list[str] = field(default_factory=list)
    builder_trace: dict[str, Any] = field(default_factory=dict)


INTERACTION_TYPE_CONTAINMENT = "containment"
INTERACTION_TYPE_PROCESSING = "processing"
INTERACTION_TYPE_ACTIVATION = "activation"
INTERACTION_TYPE_CUTTING = "cutting"
INTERACTION_TYPE_PUSH_TRANSFER = "push_transfer"
INTERACTION_TYPE_INSPECTION = "inspection"


@dataclass
class AffordanceInteraction:
    interaction_type: str
    source_entity: str
    target_entity: str
    triggering_tags: frozenset[str] = field(default_factory=frozenset)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    required_capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass
class AffordanceInteractionGraph:
    """Deprecated: retained for backward compatibility. Use AffordanceContext instead."""
    item_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    fixture_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    interactions: list[AffordanceInteraction] = field(default_factory=list)
    robot_capability_matrix: dict[str, frozenset[str]] = field(default_factory=dict)
    operation_vocab: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# New v2 data structures (Affordance Context + SLM operation DAG)
# ---------------------------------------------------------------------------


@dataclass
class EntityProfileEntry:
    """Single entity's type-level profile from Catalog."""
    canonical_name: str
    entity_type: str
    tags: list[str]
    unknown_tags: bool = False


@dataclass
class RobotCapabilityEntry:
    """Single robot slot's capability profile."""
    slot_id: str
    robot_type: str
    capabilities: list[str]
    end_effectors: int


@dataclass
class AffordanceContext:
    """Lightweight physical-knowledge context (entity tags + robot caps).

    Replaces the old AffordanceInteractionGraph for the v2 pipeline.
    Contains NO interaction edges and NO precondition/postcondition reasoning.
    """
    entity_profiles: dict[str, EntityProfileEntry] = field(default_factory=dict)
    robot_capabilities: list[RobotCapabilityEntry] = field(default_factory=list)


@dataclass
class OperationStep:
    """A single operation in the SLM-generated operation DAG."""
    op_id: str
    op_type: str
    entity: str | None = None
    dest: str | None = None
    deps: list[str] = field(default_factory=list)
    robot: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


@dataclass
class MicroStep:
    step: int
    actions: dict[str, list[str]]


@dataclass
class DispatcherPayload:
    sample: BenchmarkSample
    initial_scene_facts: SceneFacts
    scene_facts: SceneFacts
    graph_plan: GraphPlan
    micro_steps: list[MicroStep]


@dataclass
class PipelineResult:
    sample: BenchmarkSample
    initial_scene_facts: SceneFacts
    scene_facts: SceneFacts
    yolo_detections: StructuredDetections | None
    graph_plan: GraphPlan
    selected_robot_types: list[str]
    micro_steps: list[MicroStep]
    dispatcher_output: Any | None = None
