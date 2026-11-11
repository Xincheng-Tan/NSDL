from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdl.evaluation import build_result_artifact  # noqa: E402
from nsdl.llm_api import API_BASE_URL_LIST, API_MODEL_LIST  # noqa: E402
from nsdl.macro_graph import (  # noqa: E402
    GRAPH_BUILDER_ACC_DAG,
    GRAPH_BUILDER_AUTO,
)
from nsdl.pipeline import NSDLPipeline  # noqa: E402
from nsdl.prompt_adapter import build_sample_stem, load_benchmark_sample  # noqa: E402
from nsdl.yolo_detector import (  # noqa: E402
    DEFAULT_DETECTOR_BACKEND,
    DEFAULT_YOLO26_RUN_DIR,
)


DEFAULT_LEVEL = "VIKI-L2"
DEFAULT_SPLIT = "test"
DEFAULT_SAMPLE_INDEX = 0
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "nsdl"

API_INDEX = 0
API_BASE_URL = API_BASE_URL_LIST[API_INDEX]
API_MODEL = API_MODEL_LIST[API_INDEX]


def resolve_local_slm_weights_path(weights_path: str | Path) -> Path:
    path = Path(weights_path)
    if (path / "config.json").exists():
        return path
    final_path = path / "final"
    if (final_path / "config.json").exists():
        return final_path
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NSDL prototype on a single sample.")
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--sample-index", type=int, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--task-id", default=None)
    parser.add_argument(
        "--scene-parser-backend",
        choices=["rule"],
        default="rule",
        help="Initial scene-facts parse backend. The current parser stage is rule-only.",
    )
    parser.add_argument(
        "--scene-facts-refiner-backend",
        choices=["none", "llm", "auto"],
        default="none",
        help="Optional VLM stage that refines the initial scene facts using image and YOLO detections.",
    )
    parser.add_argument(
        "--graph-builder-mode",
        choices=[
            GRAPH_BUILDER_ACC_DAG,
            GRAPH_BUILDER_AUTO,
        ],
        default=GRAPH_BUILDER_ACC_DAG,
        help="Macro graph builder mode. The boundary-preserving runtime only exposes acc_dag/auto.",
    )
    parser.add_argument(
        "--detector-backend",
        choices=["none", DEFAULT_DETECTOR_BACKEND],
        default="none",
        help="Optional structured detector that produces compact YOLO detections for the refiner.",
    )
    parser.add_argument(
        "--detector-run-dir",
        default=str(DEFAULT_YOLO26_RUN_DIR),
        help="YOLO training run directory that contains args.yaml and weights/best.pt.",
    )
    parser.add_argument(
        "--detector-weights",
        default=None,
        help="Optional explicit detector weights path. Defaults to <detector-run-dir>/weights/best.pt.",
    )
    parser.add_argument(
        "--detector-data-yaml",
        default=None,
        help="Optional explicit data.yaml path. Defaults to the path recorded in args.yaml.",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--detector-iou",
        type=float,
        default=0.70,
        help="Detection IoU threshold.",
    )
    parser.add_argument(
        "--detector-max-det",
        type=int,
        default=64,
        help="Maximum detections kept per image.",
    )
    parser.add_argument(
        "--detector-device",
        default=None,
        help="Optional detector device override such as 0, cuda:0, or cpu.",
    )
    parser.add_argument("--endpoint-index", type=int, default=API_INDEX)
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--model", default=API_MODEL)
    parser.add_argument(
        "--backend",
        choices=["api", "local"],
        default="api",
        help="Macro graph generation backend.",
    )
    parser.add_argument(
        "--slm-modality",
        choices=["text", "vl"],
        default="text",
        help="Local SLM modality used when --backend=local.",
    )
    parser.add_argument("--slm-weights", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if args.backend != "local" and args.slm_modality == "vl":
        raise SystemExit("--slm-modality=vl currently requires --backend=local.")
    sample = load_benchmark_sample(
        level=args.level,
        split=args.split,
        sample_index=None if args.task_id is not None else args.sample_index,
        task_id=args.task_id,
    )

    use_local = args.backend == "local"
    local_slm_backend = None
    if use_local:
        if not args.slm_weights:
            raise SystemExit("--backend=local requires --slm-weights")
        weights_path = str(resolve_local_slm_weights_path(args.slm_weights))
        if args.slm_modality == "vl":
            from nsdl.slm_backends import get_local_vl_backend
            from nsdl.slm_interface import load_local_vlm

            load_local_vlm(weights_path, device=args.device)
            local_slm_backend = get_local_vl_backend()
        else:
            from nsdl.slm_interface import load_local_slm

            load_local_slm(weights_path, device=args.device)

    pipeline = NSDLPipeline(
        scene_parser_backend=args.scene_parser_backend,
        scene_facts_refiner_backend=args.scene_facts_refiner_backend,
        graph_builder_mode=args.graph_builder_mode,
        detector_backend=args.detector_backend,
        detector_run_dir=args.detector_run_dir,
        detector_weights_path=args.detector_weights,
        detector_data_yaml_path=args.detector_data_yaml,
        detector_confidence_threshold=args.detector_conf,
        detector_iou_threshold=args.detector_iou,
        detector_max_det=args.detector_max_det,
        detector_device=args.detector_device,
        endpoint_index=args.endpoint_index,
        api_base_url=args.base_url,
        api_model=args.model,
        use_local=use_local,
        slm_backend=local_slm_backend,
    )
    result = pipeline.run_l2(sample)
    artifact = build_result_artifact(
        result,
        validate_l2=not args.skip_validate,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{build_sample_stem(result.sample)}.json"
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"task_id: {sample.task_id}")
    print(f"task_name: {sample.task_name}")
    print(f"scene_parser_backend: {args.scene_parser_backend}")
    print(f"scene_facts_refiner_backend: {args.scene_facts_refiner_backend}")
    print(f"graph_builder_mode: {args.graph_builder_mode}")
    print(f"detector_backend: {args.detector_backend}")
    print(f"slm_backend: {args.backend}/{args.slm_modality}")
    print(f"graph_build_path: {artifact['graph_plan']['build_mode']}")
    print(f"selected_robots: {artifact['selected_robots']}")
    print(f"plan_steps: {len(artifact['plan'])}")
    if artifact["yolo_detections"]:
        print(f"yolo_visible_labels: {artifact['yolo_detections']['visible_labels']}")
    if artifact["validation"]:
        print(f"validator_success: {artifact['validation']['success']}")
        if artifact["validation"]["error_desc"]:
            print(f"validator_error: {artifact['validation']['error_desc']}")
    if artifact["benchmark_scores"]:
        print(f"viki_l1_robot_activation: {artifact['benchmark_scores']['viki_l1_robot_activation']}")
        print(f"viki_l2_plan_score: {artifact['benchmark_scores']['viki_l2_plan_score']}")
    print(f"artifact_path: {output_path}")


if __name__ == "__main__":
    main()
