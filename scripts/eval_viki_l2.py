from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


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
from nsdl.vl_prompt_variants import (  # noqa: E402
    PROMPT_VARIANT_FULL,
    PROMPT_VARIANT_FULL_NO_ENTITIES,
    PROMPT_VARIANT_IMAGE_ONLY,
    PROMPT_VARIANT_LEAN,
    PROMPT_VARIANTS,
    apply_vl_system_prompt_variant,
    apply_vl_prompt_variant,
    normalize_prompt_variant,
)
from nsdl.yolo_detector import (  # noqa: E402
    DEFAULT_DETECTOR_BACKEND,
    DEFAULT_YOLO26_RUN_DIR,
    VIKI_DETECTOR_WEIGHTS,
)


API_INDEX = 0
DEFAULT_DATASET_VERSION = os.environ.get("DATASET_VERSION", "v1")
DEFAULT_EVAL_DETECTOR_WEIGHTS = VIKI_DETECTOR_WEIGHTS  # VIKI caliber = 20 classes (facts:D3a)
LOCAL_MODEL_SIZE_CHOICES = ["1.5b", "3b", "7b"]
SLM_MODALITY_CHOICES = ["text", "vl"]


class PromptVariantSLMBackend:
    """Apply a VL prompt ablation variant immediately before generation."""

    def __init__(self, base_backend: Any, *, prompt_variant: str) -> None:
        self.base_backend = base_backend
        self.prompt_variant = normalize_prompt_variant(prompt_variant)
        self.accepts_image_input = bool(getattr(base_backend, "accepts_image_input", False))
        self.supports_correction_retry = bool(getattr(base_backend, "supports_correction_retry", True))

    def generate(self, messages, **kwargs: Any) -> str:
        return self.base_backend.generate(
            messages=self._apply_variant(messages),
            **kwargs,
        )

    def _apply_variant(self, messages) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        system_done = False
        user_done = False
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if role == "system" and not system_done:
                content = apply_vl_system_prompt_variant(str(content), self.prompt_variant)
                system_done = True
            elif role == "user" and not user_done:
                content = apply_vl_prompt_variant(str(content), self.prompt_variant)
                user_done = True
            output.append({**message, "content": content})
        return output


def resolve_local_slm_weights_path(weights_path: str | Path) -> Path:
    """Resolve SFT output roots such as `.../slm-sft-7b` to their HF `final/` dir."""
    path = Path(weights_path)
    if (path / "config.json").exists():
        return path
    final_path = path / "final"
    if (final_path / "config.json").exists():
        return final_path
    return path


def build_local_weights_registry(dataset_version: str) -> dict[tuple[str, ...], Path]:
    ckpt_root = REPO_ROOT / "checkpoints" / dataset_version
    return {
        # All SFT outputs are saved under `<output_dir>/final/` (HF native).
        ("7b", "sft"): ckpt_root / "slm-sft-7b",
        ("3b", "sft"): ckpt_root / "slm-sft-3b" / "final",
        ("3b", "grpo"): ckpt_root / "slm-grpo-3b",
        ("3b", "grpo", "v1"): ckpt_root / "slm-grpo-3b",
        ("3b", "grpo", "v3"): ckpt_root / "slm-grpo-3b-v3",
        ("1.5b", "sft"): ckpt_root / "slm-sft-1.5b" / "final",
        ("1.5b", "grpo"): ckpt_root / "slm-grpo-1.5b",
        ("1.5b", "grpo", "v1"): ckpt_root / "slm-grpo-1.5b",
        ("1.5b", "grpo", "v2"): ckpt_root / "slm-grpo-1.5b-v2",
        ("1.5b", "grpo", "v3"): ckpt_root / "slm-grpo-1.5b-v3",
        ("3b", "vl", "sft"): ckpt_root / "slm-vl-sft-3b",
        ("7b", "vl", "sft"): ckpt_root / "slm-vl-sft-7b",
        ("3b", "vl", "sft", PROMPT_VARIANT_IMAGE_ONLY): ckpt_root / "slm-vl-sft-3b-image-only",
        ("7b", "vl", "sft", PROMPT_VARIANT_IMAGE_ONLY): ckpt_root / "slm-vl-sft-7b-image-only",
        ("3b", "vl", "sft", PROMPT_VARIANT_FULL_NO_ENTITIES): ckpt_root / "slm-vl-sft-3b-full-no-entities",
        ("7b", "vl", "sft", PROMPT_VARIANT_FULL_NO_ENTITIES): ckpt_root / "slm-vl-sft-7b-full-no-entities",
        ("3b", "vl", "sft", PROMPT_VARIANT_LEAN): ckpt_root / "slm-vl-sft-3b-lean",
        ("7b", "vl", "sft", PROMPT_VARIANT_LEAN): ckpt_root / "slm-vl-sft-7b-lean",
        ("3b", "vl", "grpo"): ckpt_root / "slm-vl-grpo-3b",
        ("3b", "vl", "grpo", "v1"): ckpt_root / "slm-vl-grpo-3b",
        ("3b", "vl", "grpo", "v3"): ckpt_root / "slm-vl-grpo-3b-v3",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable evaluation on any VIKI-L2 split.")
    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        default=DEFAULT_DATASET_VERSION,
        help="Dataset namespace for default checkpoints/results. Default: DATASET_VERSION or v1.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split name. Supports custom parquet splits such as task_coverage_debug.",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=[],
        help="Optional task_name filter. Repeat or pass a comma-separated list.",
    )
    parser.add_argument(
        "--list-task-names",
        action="store_true",
        help="Print task_name counts for the selected split and exit.",
    )
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
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--max-validator-retries",
        type=int,
        default=0,
        help=(
            "Validator-feedback corrective retry setting for macro graph generation. "
            "Default 0 (no retry) — see docs/debug.md §6: retry is currently "
            "abandoned because the validator gates available on the 1.5B/3B ckpts "
            "either misfire or get neutralized by partial credit. Set >0 only for "
            "ablation studies."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["api", "local"],
        default="local",
        help="Macro graph generation backend: 'api' for external LLM, 'local' for SFT checkpoint.",
    )
    parser.add_argument(
        "--ckpt-type",
        choices=["sft", "grpo"],
        default="sft",
        help="Which local checkpoint family to evaluate. Combined with --model-size to pick default --slm-weights and a result subdir like '<size>_<type>' (e.g. '1.5b_sft', '3b_grpo').",
    )
    parser.add_argument(
        "--grpo-version",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="GRPO reward-function version (only used when --ckpt-type=grpo). v1 = original r_format+r_acc baseline; v2 = three-block CoT reward; v3 = active-set-focused EnactGraph reward. Affects default --slm-weights and result subdir name.",
    )
    parser.add_argument(
        "--model-size",
        choices=LOCAL_MODEL_SIZE_CHOICES,
        default="1.5b",
        help="Which SLM size to evaluate. Default 1.5b.",
    )
    parser.add_argument(
        "--slm-modality",
        choices=SLM_MODALITY_CHOICES,
        default="text",
        help="Use the text SLM backend or the vision-language SLM backend.",
    )
    parser.add_argument(
        "--vl-prompt-variant",
        "--vl-input-mode",
        dest="vl_prompt_variant",
        choices=PROMPT_VARIANTS,
        default=PROMPT_VARIANT_FULL,
        help=(
            "VL-only input condition. full = image + YOLO-derived text prompt; "
            "image_only = image + prompt with pre-injected perception text removed "
            "while keeping [ENTITIES]. "
            "full_no_entities = full prompt minus [ENTITIES] only. "
            "lean = image + instruction + robots/caps only, with no pre-injected "
            "perception or [ENTITIES]. "
            "Only valid with --slm-modality=vl."
        ),
    )
    parser.add_argument(
        "--slm-weights",
        default=None,
        help="Path to local SLM checkpoint directory (used when --backend=local). If omitted, resolved from (--model-size, --ckpt-type) via LOCAL_WEIGHTS_REGISTRY.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluation output root. Default: results/<dataset-version>/viki_l2.",
    )
    parser.add_argument(
        "--result-bucket",
        default=None,
        help="Optional output bucket override, e.g. 1.5b_sft_sc when --slm-weights is explicit.",
    )
    # ---- Sharding & device (parallel eval helpers) -------------------------
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of parallel processes splitting this split. Each process loads its own SLM copy.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="This process's shard id in [0, --num-shards). Samples are split via samples[shard_index::num_shards] (interleaved, balanced).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Local SLM device: 'auto' (try cuda:0, fall back to cuda:1 on OOM), 'cuda:0', 'cuda:1', etc. Only used with --backend=local.",
    )
    return parser


def parse_task_name_filters(raw_values: list[str]) -> list[str]:
    task_names: list[str] = []
    for raw_value in raw_values:
        for item in raw_value.split(","):
            normalized = item.strip()
            if normalized:
                task_names.append(normalized)
    return sorted(set(task_names))


def summarize_task_names(samples: list) -> list[tuple[str, int]]:
    counts = Counter(sample.task_name or "<unknown>" for sample in samples)
    return sorted(counts.items())


def apply_retry_bucket_suffix(bucket: str, max_validator_retries: int) -> str:
    # Bucket suffix: retry=0 (the post-2026-05-22 default; see docs/debug.md
    # §6) and any explicit ablation value gets `_max_validator_retries{N}`
    # appended. Historical retry=1 runs live under the un-suffixed bucket
    # name and are kept stable for diff comparisons.
    if max_validator_retries == 1:
        return bucket
    return f"{bucket}_max_validator_retries{max_validator_retries}"


def main() -> None:
    args = build_parser().parse_args()
    args.vl_prompt_variant = normalize_prompt_variant(args.vl_prompt_variant)
    if args.max_validator_retries < 0:
        raise SystemExit("--max-validator-retries must be >= 0")
    if args.backend != "local" and args.slm_modality == "vl":
        raise SystemExit("--slm-modality=vl currently requires --backend=local.")
    if args.slm_modality != "vl" and args.vl_prompt_variant != PROMPT_VARIANT_FULL:
        raise SystemExit("--vl-prompt-variant is only valid with --slm-modality=vl.")
    if args.slm_modality == "vl" and args.vl_prompt_variant == PROMPT_VARIANT_IMAGE_ONLY:
        if args.detector_backend != "none":
            print(
                "[VIKI-L2] --vl-prompt-variant=image_only disables detector_backend "
                f"(was {args.detector_backend}).",
                flush=True,
            )
            args.detector_backend = "none"
    weights_registry = build_local_weights_registry(args.dataset_version)
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "results" / args.dataset_version / "viki_l2")

    # ── Resolve local SLM weights based on (model_size, ckpt_type[, grpo_version]) ──
    if args.slm_weights is None:
        if args.slm_modality == "vl":
            if args.ckpt_type == "grpo":
                key = (args.model_size, "vl", "grpo", args.grpo_version)
            elif args.vl_prompt_variant != PROMPT_VARIANT_FULL:
                key = (args.model_size, "vl", args.ckpt_type, args.vl_prompt_variant)
            else:
                key = (args.model_size, "vl", args.ckpt_type)
        elif args.ckpt_type == "grpo":
            key = (args.model_size, "grpo", args.grpo_version)
        else:
            key = (args.model_size, args.ckpt_type)
        try:
            args.slm_weights = str(weights_registry[key])
        except KeyError as exc:
            raise SystemExit(
                f"No default weights registered for dataset={args.dataset_version}, key={key}; "
                "pass --slm-weights explicitly."
            ) from exc
    args.slm_weights = str(resolve_local_slm_weights_path(args.slm_weights))

    # ── Validate sharding ──
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}); got {args.shard_index}")

    # ── Load local SLM if requested (with OOM fallback) ──
    use_local = args.backend == "local"
    local_slm_backend = None
    if use_local:
        import torch
        if args.slm_modality == "vl":
            from nsdl.slm_backends import get_local_vl_backend
            from nsdl.slm_interface import load_local_vlm as load_local_backend

            local_slm_backend = get_local_vl_backend()
        else:
            from nsdl.slm_interface import load_local_slm as load_local_backend

        ckpt_label = (
            f"{args.model_size}_{args.slm_modality}_{args.ckpt_type}"
            + (f"_{args.grpo_version}" if args.ckpt_type == "grpo" else "")
        )
        if args.device == "auto":
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpu == 0:
                raise SystemExit("--device=auto requested but no CUDA device visible.")
            tried = []
            last_exc: Exception | None = None
            for dev_idx in range(n_gpu):
                dev = f"cuda:{dev_idx}"
                tried.append(dev)
                try:
                    print(f"[VIKI-L2] Loading local SLM ({ckpt_label}) on {dev} from {args.slm_weights} ...", flush=True)
                    load_local_backend(args.slm_weights, device=dev)
                    print(f"[VIKI-L2] Local SLM loaded on {dev}.", flush=True)
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
                    msg = str(exc).lower()
                    if "out of memory" not in msg and "cuda" not in msg:
                        raise
                    print(f"[VIKI-L2] OOM/CUDA-error on {dev}: {exc}; trying next GPU.", flush=True)
                    last_exc = exc
                    torch.cuda.empty_cache()
            else:
                raise SystemExit(f"Failed to load SLM on any of {tried}: {last_exc}")
        else:
            print(f"[VIKI-L2] Loading local SLM ({ckpt_label}) on {args.device} from {args.slm_weights} ...", flush=True)
            load_local_backend(args.slm_weights, device=args.device)
            print(f"[VIKI-L2] Local SLM loaded on {args.device}.", flush=True)
        if args.slm_modality == "vl" and args.vl_prompt_variant != PROMPT_VARIANT_FULL:
            local_slm_backend = PromptVariantSLMBackend(
                local_slm_backend,
                prompt_variant=args.vl_prompt_variant,
            )

    loaded_samples = load_benchmark_samples(level="VIKI-L2", split=args.split)
    requested_task_names = parse_task_name_filters(args.task_name)
    samples = loaded_samples
    if requested_task_names:
        requested_task_name_set = set(requested_task_names)
        samples = [
            sample
            for sample in samples
            if sample.task_name in requested_task_name_set
        ]
    if args.list_task_names:
        for task_name, count in summarize_task_names(samples):
            print(f"{task_name}\t{count}")
        return
    if not samples:
        raise ValueError(
            f"No samples found for split={args.split} after task_name filter={requested_task_names or None}."
        )

    filtered_total = len(samples)
    samples = samples[args.start_index :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    # Interleaved sharding: shard k handles samples[k::N]. Balanced runtime even
    # when sample cost varies along the split. Output files are per-sample so
    # shards never collide on disk.
    if args.num_shards > 1:
        samples = samples[args.shard_index :: args.num_shards]
    if not samples:
        raise ValueError(
            f"No samples left to evaluate for split={args.split} after start_index={args.start_index}, "
            f"max_samples={args.max_samples}, shard={args.shard_index}/{args.num_shards}."
        )

    # Layout: <output-dir>/<bucket>/<split>/...
    #   bucket = '<model_size>_<ckpt_type>[_<grpo_version>]' for local backend
    #     e.g. '1.5b_sft', '3b_grpo' (=v1, back-compat), '1.5b_grpo_v2'
    #   bucket = 'api'                                       for api backend
    if args.result_bucket:
        bucket = args.result_bucket
    elif use_local:
        modality_prefix = f"{args.model_size}_vl" if args.slm_modality == "vl" else args.model_size
        if args.ckpt_type == "grpo" and args.grpo_version != "v1":
            bucket = f"{modality_prefix}_{args.ckpt_type}_{args.grpo_version}"
        else:
            bucket = f"{modality_prefix}_{args.ckpt_type}"
        if args.slm_modality == "vl" and args.vl_prompt_variant != PROMPT_VARIANT_FULL:
            bucket = f"{bucket}_{args.vl_prompt_variant}"
    else:
        bucket = "api"
    if not args.result_bucket:
        bucket = apply_retry_bucket_suffix(bucket, args.max_validator_retries)
    output_dir = Path(args.output_dir)
    if output_dir.name != args.split:
        if output_dir.name != bucket:
            output_dir = output_dir / bucket
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
        use_local=use_local,
        slm_backend=local_slm_backend,
        max_validator_retries=args.max_validator_retries,
    )

    processed = 0
    skipped = 0
    errors = 0
    started_at = time.time()
    total = len(samples)
    log_every = max(args.log_every, 1)

    existing_outputs = sum(
        1
        for sample in samples
        if (output_dir / f"{build_sample_stem(sample)}.json").exists()
    )
    print(
        f"[VIKI-L2/{args.split}] start source_total={len(loaded_samples)} filtered_total={filtered_total} "
        f"eval_total={total} start_index={args.start_index} "
        f"max_samples={args.max_samples} overwrite={args.overwrite} "
        f"max_validator_retries={args.max_validator_retries} bucket={bucket} "
        f"slm_modality={args.slm_modality} vl_prompt_variant={args.vl_prompt_variant} "
        f"detector_backend={args.detector_backend} slm_weights={args.slm_weights} "
        f"shard={args.shard_index}/{args.num_shards} "
        f"task_name_filter={requested_task_names or None} existing_outputs={existing_outputs} "
        f"output_dir={output_dir}",
        flush=True,
    )
    print(
        f"[VIKI-L2/{args.split}] task_name_counts="
        f"{', '.join(f'{task_name}:{count}' for task_name, count in summarize_task_names(samples))}",
        flush=True,
    )

    for position, sample in enumerate(samples, start=1):
        output_path = output_dir / f"{build_sample_stem(sample)}.json"
        if position == 1 or position % log_every == 0:
            print(
                f"[VIKI-L2/{args.split}] begin {position}/{total} sample={output_path.name}",
                flush=True,
            )
        if output_path.exists() and not args.overwrite:
            skipped += 1
            if position == 1 or position % log_every == 0:
                print(
                    f"[VIKI-L2/{args.split}] skip  {position}/{total} sample={output_path.name} "
                    f"reason=existing_result",
                    flush=True,
                )
            continue

        sample_started_at = time.time()
        try:
            result = pipeline.run_l2(sample)
            artifact = build_result_artifact(
                result,
                validate_l2=not args.skip_validate,
            )
            artifact["result_status"] = "ok"
        except Exception as exc:
            errors += 1
            artifact = build_error_artifact(sample, exc)
            artifact["result_status"] = "error"

        artifact["runtime_seconds"] = round(time.time() - sample_started_at, 3)
        artifact["max_validator_retries"] = args.max_validator_retries
        artifact["slm_modality"] = args.slm_modality
        artifact["vl_prompt_variant"] = args.vl_prompt_variant
        artifact["slm_weights"] = args.slm_weights
        artifact["detector_backend"] = args.detector_backend
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1

        if position == 1 or position % log_every == 0:
            print(
                f"[VIKI-L2/{args.split}] done  {position}/{total} processed={processed} skipped={skipped} "
                f"errors={errors} last={output_path.name} runtime_s={artifact['runtime_seconds']}",
                flush=True,
            )

    elapsed = round(time.time() - started_at, 2)
    print(
        f"[VIKI-L2/{args.split}] done processed={processed} skipped={skipped} "
        f"errors={errors} elapsed_s={elapsed} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
