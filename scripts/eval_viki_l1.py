from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdl.evaluation import build_error_artifact, build_result_artifact  # noqa: E402
from nsdl.llm_api import API_BASE_URL_LIST, API_MODEL_LIST  # noqa: E402
from nsdl.macro_graph import (  # noqa: E402
    GRAPH_BUILDER_ACC_DAG,
    GRAPH_BUILDER_AUTO,
)
from nsdl.pipeline import NSDLPipeline  # noqa: E402
from nsdl.prompt_adapter import build_sample_stem, load_benchmark_samples  # noqa: E402
from nsdl.yolo_detector import (  # noqa: E402
    DEFAULT_DETECTOR_BACKEND,
    DEFAULT_YOLO26_RUN_DIR,
    VIKI_DETECTOR_WEIGHTS,
)


API_INDEX = 0
DEFAULT_EVAL_DETECTOR_WEIGHTS = VIKI_DETECTOR_WEIGHTS  # VIKI caliber = 20 classes (facts:D3a)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable full-dataset evaluation on VIKI-L1.")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--scene-parser-backend",
        choices=["rule"],
        default="rule",
    )
    parser.add_argument(
        "--scene-facts-refiner-backend",
        choices=["none", "llm", "auto"],
        default="none",
    )
    parser.add_argument(
        "--graph-builder-mode",
        choices=[
            GRAPH_BUILDER_ACC_DAG,
            GRAPH_BUILDER_AUTO,
        ],
        default=GRAPH_BUILDER_ACC_DAG,
    )
    parser.add_argument(
        "--detector-backend",
        choices=["none", DEFAULT_DETECTOR_BACKEND],
        default=DEFAULT_DETECTOR_BACKEND,
    )
    parser.add_argument("--detector-run-dir", default=str(DEFAULT_YOLO26_RUN_DIR))
    parser.add_argument(
        "--detector-weights",
        default=str(DEFAULT_EVAL_DETECTOR_WEIGHTS),
    )
    parser.add_argument("--detector-data-yaml", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.70)
    parser.add_argument("--detector-max-det", type=int, default=64)
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--endpoint-index", type=int, default=API_INDEX)
    parser.add_argument("--base-url", default=API_BASE_URL_LIST[API_INDEX])
    parser.add_argument("--model", default=API_MODEL_LIST[API_INDEX])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "viki_l1"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = load_benchmark_samples(level="VIKI-L1", split=args.split)
    samples = samples[args.start_index :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    output_dir = Path(args.output_dir)
    if output_dir.name != args.split:
        output_dir = output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

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
    )

    processed = 0
    skipped = 0
    errors = 0
    started_at = time.time()

    for local_index, sample in enumerate(samples, start=1):
        output_path = output_dir / f"{build_sample_stem(sample)}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        sample_started_at = time.time()
        try:
            result = pipeline.run_l2(sample)
            artifact = build_result_artifact(result, validate_l2=False)
            artifact["result_status"] = "ok"
        except Exception as exc:
            errors += 1
            artifact = build_error_artifact(sample, exc)
            artifact["result_status"] = "error"

        artifact["runtime_seconds"] = round(time.time() - sample_started_at, 3)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

        if processed % args.log_every == 0:
            print(
                f"[VIKI-L1/{args.split}] processed={processed} skipped={skipped} "
                f"errors={errors} last={output_path.name}"
            )

    elapsed = round(time.time() - started_at, 2)
    print(
        f"[VIKI-L1/{args.split}] done processed={processed} skipped={skipped} "
        f"errors={errors} elapsed_s={elapsed} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
