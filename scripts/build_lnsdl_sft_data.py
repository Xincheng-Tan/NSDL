#!/usr/bin/env python3
"""Build L-NSDL POI cold-start VL SFT data from VIKI-L2 train GT.

The builder creates agentic trajectories:

    image + lean prompt
      -> assistant: <think> [GOAL] [ASSIGN] [PLAN_SKETCH] [TOOL_CALL]
      -> tool:      tool_result (loss-masked by role)
      -> assistant: <think> [PLAN]

Only VIKI-L2 train is read by default. The teacher endpoint/model is supplied
at launch time and recorded in the dataset metadata.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.catalog import entity_has_tag, normalize_entity_name  # noqa: E402
from nsdl.llm_api import create_chat_completion, get_endpoint_config  # noqa: E402
from nsdl.perc_tool import detect  # noqa: E402
from nsdl.phi_off_prompt import build_phi_off_system_prompt  # noqa: E402
from nsdl.planning_tool import schedule, verify, verify_goal_coverage  # noqa: E402
from nsdl.schema import AffordanceContext, OperationStep, SceneFacts  # noqa: E402
from nsdl.slm_interface import build_slm_system_prompt, serialize_slm_input  # noqa: E402
from nsdl.toolset_contract import resolve_toolset_contract  # noqa: E402
from nsdl.ad_checklist_think import (  # noqa: E402
    build_checklist_grounding_think,
    build_noncommittal_pre_think,
    extract_instruction_candidates,
    plan_participant_names,
)
from nsdl.teacher_think_audit import audit_noncommittal_pre_think, audit_think  # noqa: E402
from nsdl.tool_runtime import PHI_TOOLS, render_tool_result, tool_result_message  # noqa: E402
from nsdl.yolo_detector import build_structured_detector  # noqa: E402
from scripts.audit_planning_tool_p1 import make_sketch  # noqa: E402
from scripts.build_slm_training_data import (  # noqa: E402
    _compare_action_sets_normalized,
    _format_gt_steps,
    build_scene_facts_from_gt,
    extract_sample_data,
    reverse_map_time_steps,
)
from scripts.cot_assistant_serializer import serialize_cot_assistant  # noqa: E402


DEFAULT_TEACHER_MODEL = "grok-4.5"
DEFAULT_FINAL_TEACHER_MODEL = "grok-4.5"
DEFAULT_TEACHER_ENDPOINT = "https://relay.example.com/v1/chat/completions"
TEMP_TEACHER_MODEL = DEFAULT_TEACHER_MODEL
FINAL_TEACHER_MODEL = DEFAULT_FINAL_TEACHER_MODEL
TEACHER_NOTE = (
    "P5.8 builder uses the configured teacher API with image-grounded blind pre-tool "
    "think; meta.teacher_model and meta.teacher_endpoint are authoritative."
)
NO_IMAGE_TEACHER_INPUT_NOTE = (
    "The API teacher receives text, GT-derived structure, and tool results only; "
    "the image is still stored in each student SFT row. P5.8 v2 expects "
    "--teacher-include-image for blind pre-tool think."
)
DEFAULT_OUTPUT_DIR = REPO / "data" / "lnsdl" / "sft"
DEFAULT_SCHEDULE_BOUNDARY_RATIO = 0.5
DEFAULT_REPAIR_RATE = 0.03
TEACHER_PROMPT_VERSION = "p58_consistency_guard_v1"
GOAL_SCHEMA_VERSION = "p55_discrete_status_v1"
TRAJECTORY_PROTOCOL_VERSION = "p57_one_tool_call_verify_closure_v1"
STATUS_CURRICULUM_TASK_NAMES = {
    "ensure_all_fruits_on_table",
    "serve_bread_after_checking_cabinet",
    "serve_bread_from_counter",
}
OOD_VAL_TASK_NAMES = {
    "bring_plate_to_table_bowl_already_there",
    "bring_bowl_to_table_plate_already_there",
    "bring_plate_and_bowl_to_table",
}

# --- A0b G1: LOCUS-train as an SFT source -------------------------------
# The VIKI builder is the default and its code path is unchanged. LOCUS needs
# three things VIKI does not:
#   (1) the train parquet is 15 GB in a SINGLE row group and its `images`
#       column overflows Arrow's 2 GB chunk cap, so pandas/read_table segfault
#       (memory `locus-train-parquet-unreadable`). Only iter_batches with an
#       explicit column subset works -- and the builder never needs the image
#       bytes because it reads PNGs off disk via the image index.
#   (2) `task_id` is the GRID CELL (`F1_s0m0`, 30 cells x 210 rows), not a
#       per-row key, so the VIKI image index (keyed by task_id) would collapse
#       6300 rows onto 30 images. LOCUS keys by row index instead.
#   (3) the facet-A curriculum is gated on three VIKI task NAMES, which never
#       match a LOCUS cell name -- copying the recipe verbatim would silently
#       disable g2_direct routing, forced detect and crop demos for every LOCUS
#       row. The equivalent LOCUS axis is S (docs/benchmark.md: s>=1 is the
#       state-judgement load), so the gate becomes locus_meta.cell[0] >= N.
LOCUS_ROOT = REPO / "data" / "lnsdl_bench" / "ramp63k"
# Per-benchmark caliber (facts:D3a/D3b, user ruling (a) 2026-09-05): the VIKI
# corpus keeps the 20-class detector its v3.4.1 demos were produced with; the
# 34-class file has a table-recall regression on VIKI images (序 25 步 0). Both
# names are the module's, so a change there propagates here.
from nsdl.yolo_detector import LOCUS_DETECTOR_WEIGHTS, VIKI_DETECTOR_WEIGHTS  # noqa: E402
DEFAULT_LOCUS_STATUS_MIN_S = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["viki_l2", "locus"],
        default="viki_l2",
        help="A0b G1: 'locus' builds from data/lnsdl_bench/ramp63k/<split> "
        "instead of VIKI-L2. Changes three things only: streaming parquet "
        "read, row-index image index, and the S-axis curriculum gate.",
    )
    parser.add_argument("--locus-split", default="train")
    parser.add_argument(
        "--locus-status-min-s",
        type=int,
        default=DEFAULT_LOCUS_STATUS_MIN_S,
        help="LOCUS analogue of STATUS_CURRICULUM_TASK_NAMES: rows whose "
        "locus_meta.cell[0] (S level) is >= this get the facet-A curriculum. "
        "Default 1 = every state-judgement row (4620/6300 on train).",
    )
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--viki-root", default=str(REPO / "data" / "viki" / "VIKI-L2"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=1000, help="Use <=0 for full train.")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--sample-strategy", choices=["stratified_task", "head"], default="stratified_task")
    parser.add_argument(
        "--sketch-variant",
        choices=["intent", "intent_no_open", "deps_only", "ops", "ops_chain"],
        default="intent",
        help=(
            "Sketch compression contract. VIKI builds keep the historical "
            "'intent' default; LOCUS rebuilds (2026-08-31 Phi fix) must pass "
            "'ops_chain' -- it keeps carry_to (two-level destinations) and the "
            "per-robot order chain, which schedule() provably cannot re-infer."
        ),
    )
    parser.add_argument("--think-source", choices=["api", "template"], default="api")
    parser.add_argument(
        "--query-source",
        choices=["goal", "instruction"],
        default="goal",
        help=(
            "P5.10.6 v3.2 (cluster 8): 'instruction' derives the demonstrated "
            "detect query text + crop_entities (and the (a)/(d) think candidate "
            "list) from catalog-vocabulary mentions in the INSTRUCTION, so the "
            "'what to query' policy is answer-decoupled and computable at "
            "inference. Applies to detect rows outside verify_retained with "
            "portable GOAL targets; teacher post-think cache keys keep the "
            "legacy (goal-derived) call so no re-distillation is needed. "
            "'goal' = v3/v3.1 behavior."
        ),
    )
    parser.add_argument(
        "--think-style",
        choices=["teacher", "ad_checklist"],
        default="teacher",
        help=(
            "P5.10.6 v3.1: 'ad_checklist' overrides pre_tool/grounding_tool on "
            "rows with a pre-goal detect and no coverage-conflict repair — (a) "
            "question-only non-committal blind think + (d) per-target evidence "
            "checklist derived from the real detect payload. Other think fields "
            "(post_tool, repair, coverage) keep the --think-source text. "
            "'teacher' = v3 behavior."
        ),
    )
    parser.add_argument("--teacher-model", default=TEMP_TEACHER_MODEL)
    parser.add_argument("--final-teacher-model", default=FINAL_TEACHER_MODEL)
    parser.add_argument("--endpoint-index", type=int, default=1)
    parser.add_argument("--api-base-url", default=DEFAULT_TEACHER_ENDPOINT)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument(
        "--omit-teacher-temperature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not send the temperature parameter to the teacher API.",
    )
    parser.add_argument("--teacher-max-tokens", type=int, default=420)
    parser.add_argument("--teacher-workers", type=int, default=4)
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument(
        "--api-protocol",
        choices=["auto", "chat_completions", "responses"],
        default="auto",
        help="Teacher API transport. auto tries chat-completions first and falls back to responses when needed.",
    )
    parser.add_argument(
        "--teacher-api-retries",
        type=int,
        default=8,
        help="Retry budget for transient teacher API/proxy failures.",
    )
    parser.add_argument(
        "--teacher-api-retry-sleep",
        type=float,
        default=0.5,
        help="Seconds to wait between teacher API retries.",
    )
    parser.add_argument(
        "--teacher-quality-retries",
        type=int,
        default=3,
        help=(
            "Additional full teacher re-generations when a completed think fails "
            "the lexical/GOAL-consistency audit. Bad cached thoughts are never reused."
        ),
    )
    parser.add_argument("--teacher-think-mode", choices=["blind", "legacy"], default="blind")
    parser.add_argument("--teacher-include-image", action="store_true")
    parser.add_argument(
        "--teacher-post-include-image",
        action="store_true",
        help="Also send the image to post-tool teacher calls. Default sends image only to blind pre-tool calls.",
    )
    parser.add_argument("--allow-template-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--drop-bad-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop teacher thinks that fail lexical or think/GOAL consistency audit.",
    )
    parser.add_argument("--detect-policy", choices=["auto", "none"], default="auto")
    parser.add_argument(
        "--goal-coverage-policy",
        choices=["auto", "always", "none", "direct"],
        default="auto",
        help="Inject verify_goal_coverage after detect for status-grounding samples. "
        "'direct' (v3.4, plan.md A0g) = never inject verify_goal_coverage anywhere, "
        "but KEEP the g2_direct trajectory-variant routing for curriculum rows so "
        "their teacher prompts/cache keys stay byte-identical to v3.3; requires "
        "--goal-coverage-direct-rate 1.0.",
    )
    parser.add_argument(
        "--status-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force official ID status families through detect -> goal inference -> verify_goal_coverage.",
    )
    parser.add_argument(
        "--pre-goal-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render leading detect calls before [GOAL], so teacher/model infer status after observations.",
    )
    parser.add_argument(
        "--schedule-policy",
        choices=["balanced", "heuristic", "always", "none"],
        default="balanced",
        help="P4* schedule-label policy. balanced mixes boundary transfer samples with and without schedule.",
    )
    parser.add_argument(
        "--phi-off",
        action="store_true",
        help=(
            "P10.S2 ablated arm: build the trajectory with Phi (schedule/verify/expand) "
            "and then strip it, so the model must synthesize the mechanical structure "
            "itself. Also swaps in the Phi-off system prompt. Keep --schedule-policy at "
            "its default: building phi_on first is what preserves the teacher cache keys "
            "(see strip_phi_from_trajectory)."
        ),
    )
    parser.add_argument(
        "--schedule-boundary-ratio",
        type=float,
        default=DEFAULT_SCHEDULE_BOUNDARY_RATIO,
        help="Fraction of simple transfer boundary samples that receive a schedule call under --schedule-policy balanced.",
    )
    parser.add_argument(
        "--repair-rate",
        type=float,
        default=DEFAULT_REPAIR_RATE,
        help="Fraction of schedule-positive repair-eligible samples that get an injected ok=false -> repair -> ok=true round.",
    )
    # --- P5.10.3 v3 (G2 + crop-and-look); defaults preserve v2.1 behavior ---
    parser.add_argument(
        "--goal-coverage-direct-rate",
        type=float,
        default=0.0,
        help="G2: fraction of status-curriculum samples converted to the "
        "direct-judgment pattern (first [GOAL] already carries the final "
        "skip/active decision grounded in detect observations; NO "
        "verify_goal_coverage call). The remainder keep the verify-retained "
        "pattern (conflict-repair for skip samples, confirm-ok + cross-turn "
        "schedule for the rest = F4 demo). 0.0 = v2.1 behavior.",
    )
    parser.add_argument(
        "--crop-demo-policy",
        choices=["none", "curriculum"],
        default="none",
        help="P5.10.3: 'curriculum' makes the pre-goal detect call of "
        "status-curriculum samples carry crop_entities (single-call "
        "discipline; crops of instruction-target portable items).",
    )
    parser.add_argument(
        "--crop-demo-other-rate",
        type=float,
        default=0.05,
        help="Fraction of NON-curriculum detect samples that also carry "
        "crop_entities (anti-shortcut contrast so crop != skip).",
    )
    parser.add_argument("--crop-demo-max", type=int, default=2)
    parser.add_argument(
        "--allow-schedule-gt-mismatch",
        action="store_true",
        help=(
            "Emit the schedule (Phi) demo even when the tool's reconstruction does not "
            "match GT. Off by default: a mismatched demo teaches the model a plan that "
            "contradicts the final_plan it is also trained to emit."
        ),
    )
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--emit-samples", type=int, default=8)
    args = parser.parse_args()
    if bool(args.omit_teacher_temperature):
        args.teacher_temperature = None
    resolve_source_defaults(args)
    return args


def resolve_source_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill --parquet / --yolo-weights from --source when left unset.

    Kept separate from parse_args so tests can exercise it directly.
    """
    if args.source == "locus":
        if not args.parquet:
            args.parquet = str(LOCUS_ROOT / args.locus_split / "data.parquet")
        if not args.yolo_weights:
            # The LOCUS asset set is 34 classes; the VIKI-legacy checkpoint
            # only knows 20 of them (docs/YOLO.md).
            args.yolo_weights = str(LOCUS_DETECTOR_WEIGHTS)
    else:
        if not args.parquet:
            args.parquet = str(REPO / "data" / "viki" / "VIKI-L2" / "train.parquet")
        if not args.yolo_weights:
            args.yolo_weights = str(VIKI_DETECTOR_WEIGHTS)
    return args


def main() -> int:
    # P5.10.4 anti-leak assert: the oracle perception bracket backend is
    # eval-only and must never be loaded in a data-construction process.
    assert "nsdl.oracle_perception" not in sys.modules, (
        "nsdl.oracle_perception (eval-only bracket backend) must not be "
        "imported during SFT data construction."
    )
    assert os.environ.get("NSDL_ORACLE_PERCEPTION") != "eval-only", (
        "NSDL_ORACLE_PERCEPTION must not be set in a data-construction process."
    )
    # P10.S1 anti-leak assert: the A-axis inversion arm makes detect return a
    # verdict, which would poison the SFT supervision it is meant to test.
    assert "nsdl.verdict_perception" not in sys.modules, (
        "nsdl.verdict_perception (eval-only A-axis inversion arm) must not be "
        "imported during SFT data construction."
    )
    assert os.environ.get("NSDL_VERDICT_PERCEPTION") != "eval-only", (
        "NSDL_VERDICT_PERCEPTION must not be set in a data-construction process."
    )
    args = parse_args()
    # --- v3.4 toolset-contract consistency guards (plan.md A0g ①②) ---------
    # A corpus must never demo a tool its own system prompt hides (or vice
    # versa): that is a silent half-contract, not an error any audit catches.
    toolset_contract = resolve_toolset_contract()
    if args.goal_coverage_policy == "direct" and float(args.goal_coverage_direct_rate) != 1.0:
        raise SystemExit(
            "--goal-coverage-policy direct requires --goal-coverage-direct-rate 1.0: "
            "a verify_retained row without its verify_goal_coverage call is a broken "
            "pattern, not an ablation."
        )
    if toolset_contract in ("v34", "v341"):
        # plan 序 33 (2026-09-11): the Phi-off arm IS defined under the paper
        # contract v341 -- keep detect, strip the (always-invoked) schedule call
        # so the model writes the complete [PLAN] with mechanical ops itself;
        # think reuses the cache (strip_phi_from_trajectory keeps the key).
        # The system prompt comes from build_phi_off_system_prompt(), which
        # applies the contract strip first, so the arm advertises {detect}.
        if args.phi_off and toolset_contract != "v341":
            raise SystemExit(f"LNSDL_TOOLSET_CONTRACT={toolset_contract} + --phi-off is not a defined arm (only v341 is).")
        if args.goal_coverage_policy not in ("direct", "none"):
            raise SystemExit(
                "LNSDL_TOOLSET_CONTRACT=v34 hides verify_goal_coverage from the prompt; "
                "--goal-coverage-policy must be 'direct' (or 'none'), got "
                f"{args.goal_coverage_policy!r}."
            )
        if args.schedule_policy != "always":
            raise SystemExit(
                "v3.4 contract is deterministic schedule invocation (A0g②); "
                f"--schedule-policy must be 'always', got {args.schedule_policy!r}."
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = TeacherCache(output_dir / "teacher_think_cache.jsonl")

    if args.source == "locus":
        df = load_locus_frame(Path(args.parquet))
    else:
        df = pd.read_parquet(args.parquet)
    indices = select_indices(df, max_samples=args.max_samples, strategy=args.sample_strategy, seed=args.seed)
    if args.source == "locus":
        image_index = load_locus_image_index(LOCUS_ROOT, args.locus_split)
    else:
        image_index = load_train_image_index(Path(args.viki_root))
    # P10.S2: the Phi-off arm trains against a system prompt where the
    # deterministic backend does not exist at all, so the checkpoint is
    # in-distribution rather than facing a dead tool it was taught to call.
    system_prompt = (
        build_phi_off_system_prompt() if args.phi_off else build_slm_system_prompt()
    )

    detector = None
    if args.detect_policy != "none":
        detector = build_structured_detector(
            "ultralytics_yolo26",
            weights_path=args.yolo_weights,
            device=args.yolo_device,
        )

    prepared: list[PreparedTrajectory] = []
    counters: Counter[str] = Counter()
    drop_examples: list[dict[str, Any]] = []

    for row_index in indices:
        counters["seen"] += 1
        try:
            prepared_item = prepare_trajectory(
                df.iloc[row_index],
                row_index=row_index,
                image_index=image_index,
                system_prompt=system_prompt,
                sketch_variant=args.sketch_variant,
                detect_policy=args.detect_policy,
                detector=detector,
                goal_coverage_policy=args.goal_coverage_policy,
                status_curriculum=bool(args.status_curriculum),
                pre_goal_tools=bool(args.pre_goal_tools),
                schedule_policy=args.schedule_policy,
                schedule_boundary_ratio=args.schedule_boundary_ratio,
                repair_rate=args.repair_rate,
                seed=args.seed,
                goal_coverage_direct_rate=args.goal_coverage_direct_rate,
                crop_demo_policy=args.crop_demo_policy,
                crop_demo_other_rate=args.crop_demo_other_rate,
                crop_demo_max=args.crop_demo_max,
                query_source=args.query_source,
                phi_off=bool(args.phi_off),
                source=args.source,
                locus_split=args.locus_split,
                locus_status_min_s=int(args.locus_status_min_s),
                require_schedule_gt_match=not bool(args.allow_schedule_gt_mismatch),
            )
        except DropSample as exc:
            counters[f"drop:{exc.code}"] += 1
            if len(drop_examples) < 50:
                drop_examples.append({"row_index": row_index, "code": exc.code, "message": exc.message})
            continue
        except Exception as exc:  # noqa: BLE001
            counters[f"drop:exception:{exc.__class__.__name__}"] += 1
            if len(drop_examples) < 50:
                drop_examples.append({
                    "row_index": row_index,
                    "code": f"exception:{exc.__class__.__name__}",
                    "message": str(exc),
                })
            continue

        prepared.append(prepared_item)
        counters["prepared"] += 1

    if args.think_source == "api":
        fill_api_thinks(prepared, args=args, cache=cache)
    else:
        for item in prepared:
            item.teacher = template_think(item)

    if args.think_style == "ad_checklist":
        apply_ad_checklist_thinks(prepared, counters=counters)

    rows: list[dict[str, Any]] = []
    audit_samples: list[dict[str, Any]] = []
    for item in prepared:
        if item.teacher is None:
            counters["drop:no_teacher_think"] += 1
            continue
        flags = audit_think(item.teacher, item.task_name, goal=item.goal)
        if item.audit.get("ad_checklist"):
            flags = flags + audit_noncommittal_pre_think(item.teacher.pre_tool)
        item.teacher.quality_flags = flags
        if flags:
            counters["think_quality_flagged"] += 1
            for flag in flags:
                counters[f"think_flag:{flag}"] += 1
            if args.drop_bad_think:
                counters["drop:bad_think"] += 1
                continue
        row = build_output_row(item, args=args)
        rows.append(row)
        counters["written"] += 1
        if len(audit_samples) < args.emit_samples:
            audit_samples.append(sample_preview(row))

    if not rows:
        raise SystemExit("No L-NSDL SFT rows were built; see drop counters above.")

    ds = Dataset.from_list(rows)
    parquet_path = output_dir / "train.parquet"
    hf_path = output_dir / "train"
    jsonl_path = output_dir / "samples.jsonl"
    ds.to_parquet(str(parquet_path))
    ds.save_to_disk(str(hf_path))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = build_meta(rows, counters, drop_examples, args=args)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "sample_preview.json").write_text(
        json.dumps(audit_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps({
        "rows": len(rows),
        "parquet": str(parquet_path),
        "hf_dataset": str(hf_path),
        "meta": str(output_dir / "meta.json"),
        "teacher_model": args.teacher_model,
        "teacher_note": TEACHER_NOTE,
        "teacher_include_image": bool(args.teacher_include_image),
        "teacher_input_modality": teacher_input_modality(args),
        "tool_call_distribution": meta["tool_call_distribution"],
        "teacher_status": meta["teacher_status"],
    }, indent=2, ensure_ascii=False))
    return 0


class DropSample(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class TeacherCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: dict[str, TeacherThink] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    key = str(payload.get("key") or "")
                    if key:
                        self._cache[key] = TeacherThink.from_dict(payload["think"])

    def get(self, key: str) -> "TeacherThink | None":
        return self._cache.get(key)

    def put(self, key: str, think: "TeacherThink") -> None:
        with self._lock:
            self._cache[key] = think
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "think": think.to_dict()}, ensure_ascii=False) + "\n")


class TeacherThink:
    def __init__(
        self,
        *,
        pre_tool: str,
        grounding_tool: str = "",
        coverage_repair_tool: str = "",
        repair_tool: str = "",
        post_tool: str = "",
        status: str,
        model: str,
        raw: str = "",
        quality_flags: list[str] | None = None,
    ) -> None:
        self.pre_tool = normalize_think_text(pre_tool)
        self.grounding_tool = normalize_think_text(grounding_tool)
        self.coverage_repair_tool = normalize_think_text(coverage_repair_tool)
        self.repair_tool = normalize_think_text(repair_tool)
        self.post_tool = normalize_think_text(post_tool)
        self.status = status
        self.model = model
        self.raw = raw
        self.quality_flags = quality_flags or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_tool": self.pre_tool,
            "grounding_tool": self.grounding_tool,
            "coverage_repair_tool": self.coverage_repair_tool,
            "repair_tool": self.repair_tool,
            "post_tool": self.post_tool,
            "status": self.status,
            "model": self.model,
            "raw": self.raw,
            "quality_flags": list(self.quality_flags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeacherThink":
        return cls(
            pre_tool=str(payload.get("pre_tool") or ""),
            grounding_tool=str(payload.get("grounding_tool") or ""),
            coverage_repair_tool=str(payload.get("coverage_repair_tool") or ""),
            repair_tool=str(payload.get("repair_tool") or ""),
            post_tool=str(payload.get("post_tool") or ""),
            status=str(payload.get("status") or "cache"),
            model=str(payload.get("model") or ""),
            raw=str(payload.get("raw") or ""),
            quality_flags=list(payload.get("quality_flags") or []),
        )


class PreparedTrajectory:
    def __init__(
        self,
        *,
        row_index: int,
        task_id: str,
        task_name: str,
        instruction: str,
        image_path: str,
        scene_facts: SceneFacts,
        affordance_context: AffordanceContext,
        system_prompt: str,
        user_prompt: str,
        goal: dict[str, Any],
        assign: dict[str, Any],
        plan_sketch: list[dict[str, Any]],
        final_plan: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        pre_goal_tool_count: int,
        coverage_repair: dict[str, Any] | None,
        repair: dict[str, Any] | None,
        audit: dict[str, Any],
        crop_observations: list[Any] | None = None,
    ) -> None:
        self.row_index = row_index
        self.task_id = task_id
        self.task_name = task_name
        self.instruction = instruction
        self.image_path = image_path
        self.scene_facts = scene_facts
        self.affordance_context = affordance_context
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.goal = goal
        self.assign = assign
        self.plan_sketch = plan_sketch
        self.final_plan = final_plan
        self.tool_calls = tool_calls
        self.tool_results = tool_results
        self.pre_goal_tool_count = pre_goal_tool_count
        self.coverage_repair = coverage_repair
        self.repair = repair
        self.audit = audit
        self.crop_observations = list(crop_observations or [])
        self.teacher: TeacherThink | None = None
        # Row identity for the teacher cache, set ONLY when task_id is not one.
        # VIKI leaves it None so v1-v3.3 cache keys stay byte-identical; LOCUS
        # sets it because its task_id is the grid CELL, shared by 210 rows.
        self.cache_row_scope: str | None = None


def select_indices(df: pd.DataFrame, *, max_samples: int, strategy: str, seed: int) -> list[int]:
    if max_samples <= 0 or max_samples >= len(df):
        return list(range(len(df)))
    if strategy == "head":
        return list(range(max_samples))

    rng = __import__("random").Random(seed)
    by_task: dict[str, deque[int]] = defaultdict(deque)
    for idx, row in df.iterrows():
        task_name = str(row["reward_model"]["ground_truth"].get("task_name") or "unknown")
        by_task[task_name].append(int(idx))
    for queue in by_task.values():
        items = list(queue)
        rng.shuffle(items)
        queue.clear()
        queue.extend(items)

    selected: list[int] = []
    task_names = sorted(by_task)
    while len(selected) < max_samples and any(by_task.values()):
        for task_name in task_names:
            if len(selected) >= max_samples:
                break
            if by_task[task_name]:
                selected.append(by_task[task_name].popleft())
    return selected


def load_train_image_index(viki_root: Path) -> dict[str, dict[str, str]]:
    manifest = viki_root / "train" / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing train image manifest: {manifest}")
    index: dict[str, dict[str, str]] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            image_path = Path(str(row.get("image_path") or ""))
            if not image_path.is_absolute():
                image_path = viki_root / "train" / image_path
            if task_id and image_path.exists():
                index[task_id] = {"split": "train", "image_path": str(image_path.resolve())}
    return index


def load_locus_frame(parquet_path: Path) -> pd.DataFrame:
    """Read a LOCUS split without touching the `images` column.

    `data/lnsdl_bench/ramp63k/train/data.parquet` is 15 GB in a single row
    group and its image column exceeds Arrow's 2 GB chunk limit, so
    pd.read_parquet / pq.read_table / read_row_group all crash (some of them
    by segfault). iter_batches with an explicit column subset is the only
    reader that works, and the builder only needs prompt + reward_model --
    student rows get their pixels from the on-disk PNG via the image index.
    """
    import pyarrow.parquet as pq  # local import: only LOCUS needs it

    pf = pq.ParquetFile(str(parquet_path))
    records: list[dict[str, Any]] = []
    for batch in pf.iter_batches(batch_size=256, columns=["prompt", "reward_model"]):
        records.extend(batch.to_pylist())
    return pd.DataFrame.from_records(records)


def load_locus_image_index(locus_root: Path, split: str) -> dict[str, dict[str, str]]:
    """Index LOCUS renders by ROW INDEX, not task_id.

    LOCUS `task_id` is the grid cell (`F1_s0m0`), shared by 210 rows, so the
    VIKI keying would collapse the whole split onto 30 images.
    """
    split_dir = locus_root / split
    manifest = split_dir / "render_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing LOCUS render manifest: {manifest}")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    index: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not entry.get("rendered", True):
            continue
        sample_index = entry.get("sample_index")
        rel = str(entry.get("image") or "")
        if sample_index is None or not rel:
            continue
        image_path = split_dir / rel
        if image_path.exists():
            index[str(int(sample_index))] = {
                "split": split,
                "image_path": str(image_path.resolve()),
            }
    return index


def locus_s_level(row: Any) -> int | None:
    """S-axis level of a LOCUS row, or None if the row carries no locus_meta."""
    try:
        meta = row["reward_model"]["ground_truth"].get("locus_meta") or {}
        cell = meta.get("cell")
        if cell is None:
            return None
        return int(cell[0])
    except Exception:  # noqa: BLE001 - malformed rows fall back to "no curriculum"
        return None


def prepare_trajectory(
    row: Any,
    *,
    row_index: int,
    image_index: dict[str, dict[str, str]],
    system_prompt: str,
    sketch_variant: str,
    detect_policy: str,
    detector: Any | None,
    goal_coverage_policy: str,
    status_curriculum: bool,
    pre_goal_tools: bool,
    schedule_policy: str,
    schedule_boundary_ratio: float,
    repair_rate: float,
    seed: int,
    goal_coverage_direct_rate: float = 0.0,
    crop_demo_policy: str = "none",
    crop_demo_other_rate: float = 0.05,
    crop_demo_max: int = 2,
    query_source: str = "goal",
    phi_off: bool = False,
    source: str = "viki_l2",
    locus_split: str = "train",
    locus_status_min_s: int = DEFAULT_LOCUS_STATUS_MIN_S,
    require_schedule_gt_match: bool = True,
) -> PreparedTrajectory:
    sample_data = extract_sample_data(row)
    task_id = str(sample_data.get("task_id") or "")
    task_name = str(sample_data.get("task_name") or "unknown")
    if task_name in OOD_VAL_TASK_NAMES:
        raise DropSample("ood_task_name", task_name)
    if source == "locus":
        image_key = str(row_index)
        expected_split = locus_split
    else:
        image_key = task_id
        expected_split = "train"
    image_meta = image_index.get(image_key)
    if not image_meta or image_meta.get("split") != expected_split:
        raise DropSample("missing_train_image", image_key)

    scene_facts = build_scene_facts_from_gt(sample_data, idx=row_index)
    affordance_context = build_affordance_context(scene_facts)
    gt_ops = reverse_map_time_steps(sample_data)
    gt_verify = verify(gt_ops, affordance_context, scene_facts)
    if not gt_verify.ok:
        raise DropSample("gt_verify_failed", gt_verify.message)

    sketch_ops = make_sketch(gt_ops, sketch_variant)
    goal = parse_blocks_from_legacy_serializer(scene_facts, gt_ops)["goal"]
    assign = build_assign_from_sketch(sketch_ops, scene_facts)
    plan_sketch = [_op_to_dict(op, include_gaps=True) for op in sketch_ops]
    final_plan = [_op_to_dict(op, include_gaps=False) for op in gt_ops]

    plan_faithful_violations = check_plan_faithfulness(goal, final_plan)
    if plan_faithful_violations:
        raise DropSample("plan_faithful_violation", "; ".join(plan_faithful_violations))

    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    coverage_repair: dict[str, Any] | None = None
    repair: dict[str, Any] | None = None

    if source == "locus":
        # LOCUS has no VIKI task names; the equivalent semantic axis is S
        # (docs/benchmark.md: s>=1 carries the state-judgement load, which is
        # exactly what the three VIKI curriculum tasks are).
        s_level = locus_s_level(row)
        curriculum_status_sample = (
            bool(status_curriculum)
            and s_level is not None
            and s_level >= int(locus_status_min_s)
        )
    else:
        curriculum_status_sample = (
            bool(status_curriculum)
            and task_name in STATUS_CURRICULUM_TASK_NAMES
        )

    # --- P5.10.3 v3: G2 direct-judgment routing + crop-and-look demos ---
    trajectory_variant: str | None = None
    if (
        curriculum_status_sample
        and goal_coverage_policy != "none"
        and float(goal_coverage_direct_rate) > 0.0
    ):
        is_direct = (
            stable_fraction(seed, row_index, task_id, "g2_direct")
            < float(goal_coverage_direct_rate)
        )
        trajectory_variant = "g2_direct" if is_direct else "verify_retained"

    detect_needed = detect_policy != "none" and (
        needs_detect_tool(gt_ops, goal, sample_data["instruction"])
        or curriculum_status_sample
    )
    crop_demo = False
    crop_entities: list[str] = []
    if detect_needed and crop_demo_policy == "curriculum":
        if curriculum_status_sample:
            crop_demo = True
        elif (
            stable_fraction(seed, row_index, task_id, "crop_other")
            < max(0.0, float(crop_demo_other_rate))
        ):
            crop_demo = True
        if crop_demo:
            crop_entities = choose_crop_entities(goal, max_crops=crop_demo_max)
            crop_demo = bool(crop_entities)

    # --- P5.10.6 v3.2 (cluster 8): instruction-derived query plan ---
    # The v3/v3.1 demos derived 'what to query' from goal.targets (the GT
    # answer set), so the demonstrated query policy was circular at inference
    # (query the item you already believe is missing). With
    # query_source="instruction" the query text + crop_entities come from
    # catalog mentions in the instruction text instead. verify_retained rows
    # keep the legacy queries (their trajectories stay byte-identical).
    query_candidates: list[str] = []
    query_surfaces: list[str] = []
    use_instruction_query = False
    if (
        detect_needed
        and query_source == "instruction"
        and trajectory_variant != "verify_retained"
        and choose_crop_entities(goal, max_crops=crop_demo_max or 2)
    ):
        query_candidates, query_surfaces = extract_instruction_candidates(
            sample_data["instruction"]
        )
        use_instruction_query = bool(query_candidates)

    detect_result_payload: dict[str, Any] | None = None
    crop_observations: list[Any] = []
    legacy_detect_call: dict[str, Any] | None = None
    legacy_crop_entities: list[str] | None = None
    if detect_needed:
        legacy_query = build_detect_query(gt_ops, goal)
        if use_instruction_query:
            names = (query_candidates + query_surfaces)[:6]
            query = "locations and container relations for " + ", ".join(names)
            legacy_crop_entities = list(crop_entities)
            if crop_demo:
                crop_entities = query_candidates[: max(0, int(crop_demo_max))]
        else:
            query = legacy_query
        call_args: dict[str, Any] = {"query": query}
        if crop_demo:
            # Single-call discipline (plan.md P5.10.3): uncertainty is judged
            # up front and the FIRST detect already carries crop_entities.
            call_args["crop_entities"] = list(crop_entities)
        call = {"name": "detect", "args": call_args}
        if use_instruction_query:
            legacy_args: dict[str, Any] = {"query": legacy_query}
            if crop_demo:
                legacy_args["crop_entities"] = list(legacy_crop_entities or [])
            legacy_detect_call = {"name": "detect", "args": legacy_args}
        tool_calls.append(call)
        detect_result = detect(
            image_meta["image_path"],
            query,
            detector=detector,
            crop_entities=crop_entities if crop_demo else None,
        )
        crop_observations = list(detect_result.crops)
        detect_result_payload = detect_result.to_dict()
        tool_results.append(detect_result_payload)

    goal_coverage_used = False
    goal_coverage_skipped_unclosed_conflict = False
    if trajectory_variant != "g2_direct" and should_verify_goal_coverage(
        goal=goal,
        detect_result=detect_result_payload,
        policy=goal_coverage_policy,
        curriculum_status_sample=curriculum_status_sample,
    ):
        assert detect_result_payload is not None
        coverage_repair = build_goal_coverage_repair(goal, detect_result_payload)
        if coverage_repair:
            coverage_repair["tool_index"] = len(tool_calls)
            call = coverage_repair["verify_call"]
            result = coverage_repair["verify_result"]
        else:
            call = {
                "name": "verify_goal_coverage",
                "args": {
                    "model_goal": goal,
                    "observed_facts": detect_result_payload,
                },
            }
            reason = verify_goal_coverage(goal, detect_result_payload)
            result = _tool_reason_to_dict(reason)
            if reason and not reason.ok and reason.code == "goal_coverage_conflict":
                goal_coverage_skipped_unclosed_conflict = True
                call = {}
                result = {}
        if call:
            tool_calls.append(call)
            tool_results.append(result)
            goal_coverage_used = True

    schedule_audit: dict[str, Any] = {}
    schedule_gt_reconstruct_failed = False
    schedule_decision = choose_schedule_tool(
        gt_ops,
        sketch_ops,
        goal,
        policy=schedule_policy,
        boundary_ratio=schedule_boundary_ratio,
        seed=seed,
        row_index=row_index,
        task_id=task_id,
    )
    if schedule_decision["use_schedule"]:
        prefix_tool_count = len(tool_calls)
        call, result, sched = build_schedule_call_result(
            sketch_ops,
            plan_sketch,
            assign,
            affordance_context,
            scene_facts,
        )
        # Self-validating Phi demo: audit BEFORE appending. If the tool cannot
        # reconstruct GT from the intent sketch, emitting the call would teach a
        # schedule that contradicts the final_plan in the same trajectory. On
        # VIKI this never fires (v3.3 audit: 6236 match / 0 mismatch); on LOCUS
        # it fires on two-level-destination carrier rows, whose `carry_to` the
        # intent sketch drops and the tool re-infers from the deposit target.
        candidate_schedule_audit = audit_schedule_result(result, gt_ops, sample_data)
        if require_schedule_gt_match and not candidate_schedule_audit["schedule_action_set_match_gt"]:
            schedule_gt_reconstruct_failed = True
            schedule_decision = {
                **schedule_decision,
                "use_schedule": False,
                "reason": f"{schedule_decision['reason']}+gt_reconstruct_failed",
            }
    if schedule_decision["use_schedule"]:
        repair_selected = (
            should_inject_repair(
                repair_rate=repair_rate,
                seed=seed,
                row_index=row_index,
                task_id=task_id,
            )
            and is_repair_eligible(sketch_ops)
        )
        if repair_selected:
            repair = build_repair_round(
                sketch_ops,
                plan_sketch,
                assign,
                affordance_context,
                scene_facts,
                correct_call=call,
                correct_result=result,
            )
        if repair:
            tool_calls.append(repair["failed_call"])
            tool_results.append(repair["failed_result"])
            tool_calls.append(call)
            tool_results.append(result)
        else:
            tool_calls.append(call)
            tool_results.append(result)
        if repair:
            repair["prefix_tool_count"] = prefix_tool_count
        schedule_audit = candidate_schedule_audit

    user_prompt = serialize_slm_input(scene_facts, affordance_context)
    audit = {
        "task_blind_prompt": audit_user_prompt(user_prompt),
        "final_plan_exact_gt": True,
        "tool_names": [call["name"] for call in tool_calls],
        "schedule_policy": schedule_policy,
        "schedule_decision": schedule_decision["reason"],
        "schedule_boundary": bool(schedule_decision["boundary"]),
        "schedule_used": bool(schedule_decision["use_schedule"]),
        "schedule_gt_reconstruct_failed": bool(schedule_gt_reconstruct_failed),
        "repair_injected": bool(repair),
        "status_curriculum_sample": bool(curriculum_status_sample),
        "goal_coverage_policy": goal_coverage_policy,
        "goal_coverage_used": bool(goal_coverage_used),
        "goal_coverage_repair_used": bool(coverage_repair),
        "goal_coverage_skipped_unclosed_conflict": bool(goal_coverage_skipped_unclosed_conflict),
        "plan_faithful": True,
        "plan_faithful_violations": [],
        # P5.10.3 v3 (G2 + crop) audit keys.
        "trajectory_variant": trajectory_variant,
        "crop_demo": bool(crop_demo),
        "crop_entities": list(crop_entities),
        # P5.10.6 v3.2 (cluster 8) audit keys.
        "query_source": "instruction" if use_instruction_query else "goal",
        "query_candidates": list(query_candidates),
        "query_surfaces": list(query_surfaces),
        "crop_produced": len(crop_observations),
        "crop_missing": list((detect_result_payload or {}).get("crop_missing") or []),
        # F4: schedule emitted in a later turn than [PLAN_SKETCH] (cross-turn
        # op_ids reference) happens whenever a post-goal tool precedes schedule.
        "f4_cross_turn_schedule": bool(
            goal_coverage_used and schedule_decision["use_schedule"]
        ),
        **schedule_audit,
    }
    pre_goal_tool_count = (
        count_leading_pre_goal_tools(tool_calls)
        if pre_goal_tools
        else 0
    )
    audit["pre_goal_tool_count"] = pre_goal_tool_count
    prepared = PreparedTrajectory(
        row_index=row_index,
        task_id=task_id,
        task_name=task_name,
        instruction=sample_data["instruction"],
        image_path=image_meta["image_path"],
        scene_facts=scene_facts,
        affordance_context=affordance_context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        goal=goal,
        assign=assign,
        plan_sketch=plan_sketch,
        final_plan=final_plan,
        tool_calls=tool_calls,
        tool_results=tool_results,
        pre_goal_tool_count=pre_goal_tool_count,
        coverage_repair=coverage_repair,
        repair=repair,
        audit=audit,
        crop_observations=crop_observations,
    )
    # v3.2: teacher post-think cache keys keep the legacy (goal-derived) detect
    # call so the v3.1 cache stays fully reusable (pre/grounding thinks are
    # deterministically overridden anyway; post/plan narration is unaffected by
    # the query wording).
    if source == "locus":
        # LOCUS task_id is the grid cell (F1_s0m0), shared by 210 rows, so the
        # VIKI-shaped teacher cache key collapses distinct renders onto one
        # cached think -- and --teacher-include-image means that think was
        # grounded in ANOTHER row's image. Scope the key to the row.
        prepared.cache_row_scope = f"{locus_split}#{row_index}"
    if legacy_detect_call is not None:
        prepared.cache_legacy_tool_calls = [legacy_detect_call] + tool_calls[1:]
        prepared.cache_legacy_crop_entities = list(legacy_crop_entities or [])
    if phi_off:
        strip_phi_from_trajectory(prepared)
    return prepared


def strip_phi_from_trajectory(prepared: PreparedTrajectory) -> None:
    """P10.S2 `--phi-off`: remove the deterministic backend from a built trajectory.

    The Phi-off arm's supervision needs no new pipeline: `final_plan` is already
    the ground-truth complete DAG (computed before the schedule branch runs), and
    `render_initial_assistant(..., tool_calls=[])` already emits the single-turn
    `think -> [GOAL] -> [ASSIGN] -> [PLAN_SKETCH] -> [PLAN]` shape for the 413
    rows v3.3 builds with no post-goal tool. All this does is route the other
    ~6.4k rows down that same existing path.

    WHY NOT `--schedule-policy none`. That flag exists and would produce the same
    trajectories, but `teacher_cache_key` hashes `tools` and `repair`, so dropping
    schedule *before* the key is computed invalidates ~6416/7196 entries and
    drives that many calls into a relay teacher API this project has documented as
    flaky. Building phi_on first and stripping afterwards keeps the key identical,
    so the v3.3 cache hits 100%.

    That reuse is sound, not a shortcut: `teacher_pre_prompt` only ever shows the
    teacher the leading `detect` calls (`sanitize_pre_goal_tool_calls`), never
    schedule; `--think-style ad_checklist` overrides pre/grounding narration
    deterministically and locally; and the fields whose narration does mention
    schedule (`post_tool`, `repair_tool`) are consumed only by renderers this
    strip removes.
    """
    if getattr(prepared, "cache_legacy_tool_calls", None) is None:
        prepared.cache_legacy_tool_calls = list(prepared.tool_calls)
    prepared.cache_legacy_repair = prepared.repair

    keep = [
        index
        for index, call in enumerate(prepared.tool_calls)
        if str(call.get("name") or "") not in PHI_TOOLS
    ]
    removed = len(prepared.tool_calls) - len(keep)
    # Phi is always appended after the perception / coverage calls, so the removed
    # calls form a suffix and no surviving absolute index shifts. `coverage_repair`
    # and `repair` both carry absolute tool indices, so a non-suffix removal would
    # silently misalign the renderers rather than fail.
    if keep != list(range(len(keep))):
        raise AssertionError(
            f"--phi-off expected Phi calls to form a suffix; kept indices {keep} "
            f"of {prepared.audit.get('tool_names')}"
        )
    prepared.tool_calls = [prepared.tool_calls[i] for i in keep]
    prepared.tool_results = [prepared.tool_results[i] for i in keep]
    # The repair demo is a failed-schedule / re-schedule pair, so it cannot
    # survive its own tool being gone.
    prepared.repair = None
    prepared.audit["phi_off"] = True
    prepared.audit["phi_off_calls_removed"] = removed
    prepared.audit["repair_injected"] = False
    prepared.audit["f4_cross_turn_schedule"] = False
    # `schedule_used` describes the emitted trajectory, so it has to follow the
    # strip; the pre-strip value is kept because it is what selected the teacher
    # cache entry, and `schedule_*_gt` (left untouched) still certifies that the
    # GT DAG these rows supervise is the one schedule would have produced.
    prepared.audit["phi_off_source_schedule_used"] = bool(
        prepared.audit.get("schedule_used")
    )
    prepared.audit["schedule_used"] = False
    # pre_goal_tool_count counts leading `detect`s only, so Phi can never sit
    # inside it. Check rather than clamp: clamping would silently move
    # teacher_cache_key (the count is part of the key) and hide a real ordering
    # bug behind a 100%-looking cache.
    if prepared.pre_goal_tool_count > len(prepared.tool_calls):
        raise AssertionError(
            f"--phi-off found Phi inside the pre-GOAL prefix: "
            f"pre_goal_tool_count={prepared.pre_goal_tool_count} > "
            f"{len(prepared.tool_calls)} surviving calls "
            f"({prepared.audit.get('tool_names')})"
        )
    prepared.audit["tool_names"] = [call["name"] for call in prepared.tool_calls]


def parse_blocks_from_legacy_serializer(scene_facts: SceneFacts, ops: list[OperationStep]) -> dict[str, Any]:
    text = serialize_cot_assistant(scene_facts, ops, robot_inventory=list(scene_facts.robot_inventory))
    return {
        "goal": _json_block(text, "GOAL"),
        "assign": _json_block(text, "ASSIGN"),
    }


def _json_block(text: str, block: str) -> Any:
    pattern = re.compile(rf"\[{block}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"missing block {block}")
    return json.loads(match.group(1).strip())


def build_assign_from_sketch(
    sketch_ops: Iterable[OperationStep],
    scene_facts: SceneFacts,
) -> dict[str, dict[str, list[str]]]:
    assign: dict[str, dict[str, list[str]]] = {
        robot.slot_id: {"items": []}
        for robot in scene_facts.robot_inventory
    }
    for op in sketch_ops:
        if not op.robot or not op.entity:
            continue
        if op.op_type not in {"acquire", "local_acquire", "deposit", "process_at", "apply_tool"}:
            continue
        bucket = assign.setdefault(op.robot, {"items": []})["items"]
        if op.entity not in bucket:
            bucket.append(op.entity)
    return assign


def needs_detect_tool(ops: list[OperationStep], goal: dict[str, Any], instruction: str) -> bool:
    if goal.get("skip"):
        return True
    text = instruction.lower()
    if any(cue in text for cue in ("already", "still absent", "if", "counter", "cabinet")):
        return True
    if any(op.op_type == "open" for op in ops):
        return True
    return any(entity_has_tag(normalize_entity_name(op.dest) or "", "openable") for op in ops if op.dest)


def choose_crop_entities(goal: dict[str, Any], *, max_crops: int = 2) -> list[str]:
    """Deterministic crop-request policy for training demos (P5.10.3).

    Candidates are the instruction-derived portable target items (the model's
    own goal candidates — the demonstrated policy may be goal-aware; the crop
    CONTENT itself stays task-blind bbox geometry). Stable name order, <=N.
    """
    from nsdl.catalog import is_portable_entity  # local import: optional dep path

    candidates: list[str] = []
    for target in goal.get("targets") or []:
        if not isinstance(target, dict):
            continue
        name = normalize_entity_name(str(target.get("item") or ""))
        if name and name not in candidates and is_portable_entity(name):
            candidates.append(name)
    candidates.sort()
    return candidates[: max(0, int(max_crops))]


def build_detect_query(ops: list[OperationStep], goal: dict[str, Any]) -> str:
    names: list[str] = []
    for item in list(goal.get("active") or []) + list(goal.get("skip") or []):
        if item not in names:
            names.append(str(item))
    for op in ops:
        for value in (op.entity, op.dest):
            if value and value not in names and len(names) < 6:
                names.append(str(value))
    return "locations and container relations for " + ", ".join(names[:6])


def should_verify_goal_coverage(
    *,
    goal: dict[str, Any],
    detect_result: dict[str, Any] | None,
    policy: str,
    curriculum_status_sample: bool,
) -> bool:
    if policy in ("none", "direct") or detect_result is None:
        return False
    if policy == "always":
        return True
    return bool(curriculum_status_sample or goal.get("skip"))


def build_goal_coverage_repair(
    goal: dict[str, Any],
    detect_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Create a supervised wrong-GOAL -> verify conflict -> corrected-GOAL loop.

    The injected wrong goal is only a training trajectory state: it activates an
    item that the correct GT-derived GOAL skips. The tool still receives only
    the model-emitted goal plus observed detect facts, so it remains within the
    agent.md verify_goal_coverage exception.
    """
    targets = goal.get("targets") if isinstance(goal.get("targets"), list) else []
    skip_items = [str(item) for item in goal.get("skip") or [] if str(item).strip()]
    for skip_item in skip_items:
        bad_goal = json.loads(json.dumps(goal, ensure_ascii=False))
        changed = False
        for target in bad_goal.get("targets") or []:
            if not isinstance(target, dict) or str(target.get("item") or "") != skip_item:
                continue
            target["status"] = "active"
            target["status_reason"] = "needs_relocation"
            target["status_line"] = f"{skip_item}: active"
            changed = True
            break
        if not changed and targets:
            continue
        active = [str(item) for item in bad_goal.get("active") or [] if str(item).strip()]
        if skip_item not in active:
            active.append(skip_item)
        bad_goal["active"] = active
        bad_goal["skip"] = [
            str(item)
            for item in bad_goal.get("skip") or []
            if str(item) != skip_item
        ]
        reason = verify_goal_coverage(bad_goal, detect_result)
        if not reason or reason.ok or reason.code != "goal_coverage_conflict":
            continue
        call = {
            "name": "verify_goal_coverage",
            "args": {
                "model_goal": bad_goal,
                "observed_facts": detect_result,
            },
        }
        return {
            "type": "goal_coverage_conflict_repair",
            "conflict_item": skip_item,
            "conflict_goal": bad_goal,
            "corrected_goal": goal,
            "verify_call": call,
            "verify_result": _tool_reason_to_dict(reason),
        }
    return None


def check_plan_faithfulness(goal: dict[str, Any], final_plan: list[dict[str, Any]]) -> list[str]:
    target_items = {
        str(target.get("item") or "").strip()
        for target in goal.get("targets") or []
        if isinstance(target, dict) and str(target.get("item") or "").strip()
    }
    active_items = {str(item).strip() for item in goal.get("active") or [] if str(item).strip()}
    skip_items = {str(item).strip() for item in goal.get("skip") or [] if str(item).strip()}

    violations: list[str] = []
    for step in final_plan:
        if not isinstance(step, dict):
            continue
        op_type = str(step.get("op_type") or "").strip()
        entity = str(step.get("entity") or "").strip()
        if not entity or entity not in target_items:
            continue
        if op_type in {"acquire", "local_acquire", "load_carrier"} and entity not in active_items:
            violations.append(f"{op_type}:{entity}:not_in_active")
        if op_type in {"acquire", "local_acquire", "load_carrier", "deposit"} and entity in skip_items:
            violations.append(f"{op_type}:{entity}:skip_item_in_plan")
    return violations


def count_leading_pre_goal_tools(tool_calls: list[dict[str, Any]]) -> int:
    count = 0
    for call in tool_calls:
        name = str(call.get("name") or "").strip().lower()
        if name != "detect":
            break
        count += 1
    return count


def choose_schedule_tool(
    gt_ops: list[OperationStep],
    sketch_ops: list[OperationStep],
    goal: dict[str, Any],
    *,
    policy: str,
    boundary_ratio: float,
    seed: int,
    row_index: int,
    task_id: str,
) -> dict[str, Any]:
    """Return a deterministic P4* schedule label decision."""
    if policy == "none":
        return {"use_schedule": False, "reason": "policy_none", "boundary": False}
    if policy == "always":
        return {"use_schedule": True, "reason": "policy_always", "boundary": is_schedule_boundary_sample(gt_ops, sketch_ops, goal)}

    heuristic = needs_schedule_tool(gt_ops, sketch_ops, goal)
    boundary = is_schedule_boundary_sample(gt_ops, sketch_ops, goal)
    if policy == "heuristic":
        return {
            "use_schedule": heuristic,
            "reason": "heuristic_required" if heuristic else "heuristic_skipped",
            "boundary": boundary,
        }

    if heuristic:
        return {"use_schedule": True, "reason": "heuristic_required", "boundary": boundary}

    if boundary:
        ratio = min(1.0, max(0.0, float(boundary_ratio)))
        selected = stable_fraction(seed, row_index, task_id, "schedule_boundary") < ratio
        return {
            "use_schedule": selected,
            "reason": "boundary_mixed_in" if selected else "boundary_mixed_out",
            "boundary": True,
        }

    return {"use_schedule": False, "reason": "balanced_skipped", "boundary": False}


def is_schedule_boundary_sample(
    gt_ops: list[OperationStep],
    sketch_ops: list[OperationStep],
    goal: dict[str, Any],
) -> bool:
    """Simple purposeful transfer samples that should not collapse to 100% no-tool."""
    if not (goal.get("active") or []):
        return False
    active_robots = {op.robot for op in gt_ops if op.robot}
    if len(active_robots) != 1:
        return False
    allowed_ops = {"acquire", "local_acquire", "carry_to", "deposit"}
    if any(op.op_type not in allowed_ops for op in gt_ops):
        return False
    if not any(op.op_type == "deposit" and op.dest for op in gt_ops):
        return False
    sketch_types = {op.op_type for op in sketch_ops}
    return bool(sketch_types.intersection({"acquire", "local_acquire"}) and "deposit" in sketch_types)


def stable_fraction(seed: int, row_index: int, task_id: str, salt: str) -> float:
    blob = f"{seed}:{row_index}:{task_id}:{salt}".encode("utf-8")
    value = int(hashlib.sha1(blob).hexdigest()[:12], 16)
    return value / float(16 ** 12)


def should_inject_repair(*, repair_rate: float, seed: int, row_index: int, task_id: str) -> bool:
    if repair_rate <= 0:
        return False
    ratio = min(1.0, max(0.0, float(repair_rate)))
    return stable_fraction(seed, row_index, task_id, "repair_round") < ratio


def _schedule_call_args(
    plan_sketch: list[dict[str, Any]], assign: dict[str, Any]
) -> dict[str, Any]:
    """Build schedule tool-call args under the op_ids-reference contract (B1).

    Instead of re-serializing the full nested ``ops`` array (a byte-for-byte
    duplicate of the same-turn ``[PLAN_SKETCH]`` that maximizes the fragile long
    tool-call JSON surface), the model references the sketch ops by id. ``assign``
    stays in args because robot assignment is a genuine model decision (not a
    duplicate of the sketch). The runtime resolves ``op_ids`` back to ops from the
    parsed ``[PLAN_SKETCH]`` emitted in the same turn.
    """
    op_ids = [str(op["op_id"]) for op in plan_sketch if op.get("op_id")]
    return {"op_ids": op_ids, "assign": assign}


def build_schedule_call_result(
    sketch_ops: list[OperationStep],
    plan_sketch: list[dict[str, Any]],
    assign: dict[str, Any],
    affordance_context: AffordanceContext,
    scene_facts: SceneFacts,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    call = {"name": "schedule", "args": _schedule_call_args(plan_sketch, assign)}
    sched = schedule(sketch_ops, assign=assign, ctx=affordance_context, scene_facts=scene_facts)
    if not sched.ok:
        reason = sched.reason.message if sched.reason else "schedule failed"
        raise DropSample("schedule_failed", reason)
    return call, schedule_result_payload(sched), sched


def schedule_result_payload(sched: Any) -> dict[str, Any]:
    return {
        "ok": bool(sched.ok),
        "ops": [_op_to_dict(op, include_gaps=True) for op in sched.ops],
        "schedule": [_micro_step_to_dict(step) for step in sched.schedule],
        "inserted": list(sched.inserted),
        "reason": _tool_reason_to_dict(sched.reason) if sched.reason else None,
    }


def _tool_reason_to_dict(reason: Any | None) -> dict[str, Any] | None:
    if reason is None:
        return None
    return {
        "ok": bool(getattr(reason, "ok", False)),
        "code": getattr(reason, "code", None),
        "op_ids": list(getattr(reason, "op_ids", []) or []),
        "message": str(getattr(reason, "message", "") or ""),
    }


def is_repair_eligible(sketch_ops: list[OperationStep]) -> bool:
    sources = {(op.entity, op.robot) for op in sketch_ops if op.op_type in {"acquire", "local_acquire"} and op.entity}
    return any(
        op.op_type == "deposit" and op.entity and (op.entity, op.robot) in sources
        for op in sketch_ops
    )


def build_repair_round(
    sketch_ops: list[OperationStep],
    plan_sketch: list[dict[str, Any]],
    assign: dict[str, Any],
    affordance_context: AffordanceContext,
    scene_facts: SceneFacts,
    *,
    correct_call: dict[str, Any],
    correct_result: dict[str, Any],
) -> dict[str, Any] | None:
    bad_ops = corrupt_sketch_for_repair(sketch_ops)
    if not bad_ops:
        return None
    bad_plan_sketch = [_op_to_dict(op, include_gaps=True) for op in bad_ops]
    failed_call = {"name": "schedule", "args": _schedule_call_args(bad_plan_sketch, assign)}
    failed_sched = schedule(bad_ops, assign=assign, ctx=affordance_context, scene_facts=scene_facts)
    if failed_sched.ok:
        return None
    failed_result = schedule_result_payload(failed_sched)
    failed_result["ok"] = False
    failed_result.setdefault("reason", _tool_reason_to_dict(failed_sched.reason))
    return {
        "type": "missing_custody_repair",
        "failed_plan_sketch": bad_plan_sketch,
        "repaired_plan_sketch": plan_sketch,
        "failed_call": failed_call,
        "failed_result": failed_result,
        "repair_call": correct_call,
        "repair_result": correct_result,
        "failure_code": (failed_result.get("reason") or {}).get("code"),
        "failure_message": (failed_result.get("reason") or {}).get("message"),
    }


def corrupt_sketch_for_repair(sketch_ops: list[OperationStep]) -> list[OperationStep] | None:
    """Drop one same-entity acquire before deposit to create a real ok=false schedule turn."""
    deposit = next(
        (
            op
            for op in sketch_ops
            if op.op_type == "deposit"
            and op.entity
            and any(
                src.op_type in {"acquire", "local_acquire"}
                and src.entity == op.entity
                and src.robot == op.robot
                for src in sketch_ops
            )
        ),
        None,
    )
    if deposit is None:
        return None
    removed = False
    bad_ops: list[OperationStep] = []
    for op in sketch_ops:
        if (
            not removed
            and op.op_type in {"acquire", "local_acquire"}
            and op.entity == deposit.entity
            and op.robot == deposit.robot
        ):
            removed = True
            continue
        bad_ops.append(OperationStep(
            op_id=op.op_id,
            op_type=op.op_type,
            entity=op.entity,
            dest=op.dest,
            deps=[dep for dep in op.deps if dep != op.op_id],
            robot=op.robot,
            params=dict(op.params or {}),
            gaps=list(op.gaps or []),
        ))
    return bad_ops if removed else None


def needs_schedule_tool(
    gt_ops: list[OperationStep],
    sketch_ops: list[OperationStep],
    goal: dict[str, Any],
) -> bool:
    active_robot_count = len({op.robot for op in gt_ops if op.robot})
    if active_robot_count > 1:
        return True
    if len(goal.get("active") or []) > 1:
        return True
    complex_ops = {
        "open",
        "process_at",
        "apply_tool",
        "activate",
        "inspect",
        "handoff",
        "load_carrier",
        "push_carrier",
        "unload_carrier",
        "carry_to",
    }
    if any(op.op_type in complex_ops for op in gt_ops):
        return True
    if len(sketch_ops) > 2:
        return True
    return False


def audit_schedule_result(
    schedule_result: dict[str, Any],
    gt_ops: list[OperationStep],
    sample_data: dict[str, Any],
) -> dict[str, Any]:
    scheduled_ops = schedule_result.get("ops") or []
    exact = [
        _op_dict_without_gaps(op)
        for op in scheduled_ops
        if isinstance(op, dict)
    ] == [_op_to_dict(op, include_gaps=False) for op in gt_ops]
    recon = []
    for step in schedule_result.get("schedule") or []:
        actions = step.get("actions", {}) if isinstance(step, dict) else {}
        recon.append({slot: list(action) for slot, action in actions.items() if action is not None})
    gt_formatted = _format_gt_steps(sample_data["time_steps"])
    return {
        "schedule_ops_exact_gt": bool(exact),
        "schedule_action_set_match_gt": bool(_compare_action_sets_normalized(recon, gt_formatted)),
    }


def _op_dict_without_gaps(op: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(op)
    cleaned.pop("gaps", None)
    if not cleaned.get("params"):
        cleaned.pop("params", None)
    return cleaned


def audit_user_prompt(user_prompt: str) -> bool:
    forbidden = ("Observed entities:", "Entity locations:", "[ENTITIES]", "[PLAN]")
    return not any(item in user_prompt for item in forbidden)


def fill_api_thinks(
    prepared: list[PreparedTrajectory],
    *,
    args: argparse.Namespace,
    cache: TeacherCache,
) -> None:
    def one(item: PreparedTrajectory) -> tuple[PreparedTrajectory, TeacherThink]:
        key = teacher_cache_key(
            item,
            args.teacher_model,
            include_image=args.teacher_include_image,
            post_include_image=args.teacher_post_include_image,
            think_mode=args.teacher_think_mode,
        )
        # Surgical re-distillation hook (v3.4 PROVENANCE, 2026-09-01): rows whose
        # cached think was grounded in a detect payload that has since drifted at
        # the entity/relation-set level (cpu-vs-cuda detector, 3/7196 rows) must
        # NOT reuse the stale think. Row list comes from
        # scripts/audit_v34_corpus_gates.py gate6; keys are per-row here
        # (task_id unique for the listed rows), so no collateral re-think.
        _force_rows = {
            int(tok)
            for tok in os.environ.get("LNSDL_FORCE_REDISTILL_ROWS", "").split(",")
            if tok.strip().isdigit()
        }
        cached = None if getattr(item, "row_index", None) in _force_rows else cache.get(key)
        if cached is not None:
            cached = complete_teacher_think(item, cached)
            cached_flags = audit_think(cached, item.task_name, goal=item.goal)
            if not cached_flags:
                cached.status = "cache"
                return item, cached

        # The official-data-only contract requires preserving all 7196 train
        # examples. A bad teacher narration is therefore a regeneration event,
        # never a reason to silently drop an otherwise valid train sample.
        attempts = max(1, int(args.teacher_quality_retries) + 1)
        last_think: TeacherThink | None = None
        for attempt_index in range(attempts):
            try:
                think = call_teacher_api(item, args=args)
            except Exception as exc:  # noqa: BLE001
                if not args.allow_template_fallback:
                    raise
                think = template_think(item)
                think.status = f"template_fallback:{exc.__class__.__name__}"
                think.raw = str(exc)
            think = complete_teacher_think(item, think)
            flags = audit_think(think, item.task_name, goal=item.goal)
            think.quality_flags = flags
            if not flags:
                cache.put(key, think)
                return item, think
            last_think = think
            if attempt_index + 1 < attempts:
                time.sleep(max(0.0, float(args.teacher_api_retry_sleep)))

        assert last_think is not None
        return item, last_think

    workers = max(1, int(args.teacher_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in prepared]
        for future in as_completed(futures):
            item, think = future.result()
            item.teacher = think


def call_teacher_api(item: PreparedTrajectory, *, args: argparse.Namespace) -> TeacherThink:
    if args.teacher_think_mode == "legacy":
        return call_teacher_api_legacy(item, args=args)

    pre_raw, pre_parsed = call_teacher_json(
        teacher_pre_prompt(item),
        args=args,
        include_image=bool(args.teacher_include_image),
        image_path=item.image_path,
    )
    pre_tool = pre_parsed.get("pre_tool") or pre_parsed.get("think") or ""
    grounding_tool = ""
    grounding_raw = ""
    if item.pre_goal_tool_count:
        grounding_raw, grounding_parsed = call_teacher_json(
            teacher_grounding_prompt(item, pre_tool=str(pre_tool)),
            args=args,
            include_image=False,
            image_path=item.image_path,
        )
        grounding_tool = grounding_parsed.get("grounding_tool") or grounding_parsed.get("think") or ""
    repair_tool = ""
    post_tool = ""
    post_raw = ""

    if item.tool_calls:
        if item.repair:
            post_prompt = teacher_repair_post_prompt(item, pre_tool=pre_tool)
        else:
            post_prompt = teacher_post_prompt(item, pre_tool=pre_tool)
        post_raw, post_parsed = call_teacher_json(
            post_prompt,
            args=args,
            include_image=bool(args.teacher_post_include_image),
            image_path=item.image_path,
        )
        coverage_repair_tool = post_parsed.get("coverage_repair_tool") or ""
        repair_tool = post_parsed.get("repair_tool") or ""
        post_tool = post_parsed.get("post_tool") or post_parsed.get("think") or ""
    else:
        coverage_repair_tool = ""

    return TeacherThink(
        pre_tool=str(pre_tool),
        grounding_tool=str(grounding_tool),
        coverage_repair_tool=str(coverage_repair_tool),
        repair_tool=repair_tool,
        post_tool=post_tool,
        status="api_blind",
        model=args.teacher_model,
        raw=json.dumps({"pre": pre_raw, "grounding": grounding_raw, "post": post_raw}, ensure_ascii=False),
    )


def complete_teacher_think(item: PreparedTrajectory, think: TeacherThink) -> TeacherThink:
    """Fill required local narration for supervised repair states."""
    if item.coverage_repair and not think.coverage_repair_tool:
        conflict_item = str(item.coverage_repair.get("conflict_item") or "the conflicted item")
        think.coverage_repair_tool = (
            "verify_goal_coverage reports "
            f"{conflict_item} is already observed at its declared destination, "
            "so I revise the GOAL to skip that item before planning."
        )
    return think


def call_teacher_api_legacy(item: PreparedTrajectory, *, args: argparse.Namespace) -> TeacherThink:
    raw, parsed = call_teacher_json(
        teacher_prompt_legacy(item),
        args=args,
        include_image=bool(args.teacher_include_image),
        image_path=item.image_path,
    )
    return TeacherThink(
        pre_tool=parsed.get("pre_tool") or parsed.get("pre_tool_think") or "",
        post_tool=parsed.get("post_tool") or parsed.get("post_tool_think") or "",
        status="api_legacy",
        model=args.teacher_model,
        raw=raw,
    )


def call_teacher_json(
    prompt: str,
    *,
    args: argparse.Namespace,
    include_image: bool,
    image_path: str | None,
) -> tuple[str, dict[str, str]]:
    content: Any = prompt
    if include_image:
        if not image_path:
            raise ValueError("image_path is required for image teacher calls")
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    request_kwargs: dict[str, Any] = {"timeout": args.api_timeout}
    if getattr(args, "teacher_temperature", None) is not None:
        request_kwargs["temperature"] = args.teacher_temperature
    if getattr(args, "teacher_max_tokens", None) is not None:
        request_kwargs["max_tokens"] = args.teacher_max_tokens
    attempts = max(1, int(getattr(args, "teacher_api_retries", 1)))
    sleep_s = max(0.0, float(getattr(args, "teacher_api_retry_sleep", 0.0)))
    last_error: Exception | None = None
    for attempt_idx in range(attempts):
        try:
            response = create_chat_completion(
                messages=[
                    {"role": "system", "content": teacher_system_prompt()},
                    {"role": "user", "content": content},
                ],
                endpoint_index=args.endpoint_index,
                api_base_url=args.api_base_url,
                api_model=args.teacher_model,
                api_protocol=str(getattr(args, "api_protocol", "auto")),
                **request_kwargs,
            )
            response_error = getattr(response, "error", None)
            if response_error:
                raise RuntimeError(f"teacher API error: {response_error}")
            if not getattr(response, "choices", None):
                raise RuntimeError("teacher API returned no choices")
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                raise RuntimeError("teacher API returned empty assistant text")
            return raw, parse_teacher_response(raw)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt_idx + 1 >= attempts:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)
    raise RuntimeError(f"teacher API failed after {attempts} attempts: {last_error}") from last_error


def teacher_system_prompt() -> str:
    return (
        "You write concise chain-of-thought supervision for a robotics VLM dataset. "
        "Return only JSON with the requested keys. Do not mention labels, "
        "ground truth, task names, splits, held-out sets, oracle data, or dataset construction. "
        "Stay faithful to the provided structure and tool results; do not invent objects. "
        "Only describe a target or the whole goal as skipped when the available evidence directly shows "
        "that the target is already satisfied at its required destination. Missing detections, missing support "
        "surfaces, or incomplete observations imply uncertainty, not skip."
    )


def teacher_prompt_legacy(item: PreparedTrajectory) -> str:
    return (
        "Write brief English think text for this L-NSDL training trajectory.\n"
        "The text should explain scene grounding, goal decomposition, robot assignment, "
        "why the listed tools are or are not called, and how the tool result is integrated.\n"
        "Use 1-3 compact sentences per field.\n\n"
        f"Instruction:\n{item.instruction}\n\n"
        f"Lean user prompt:\n{item.user_prompt}\n\n"
        f"[GOAL]\n{json.dumps(item.goal, ensure_ascii=False)}\n\n"
        f"[ASSIGN]\n{json.dumps(item.assign, ensure_ascii=False)}\n\n"
        f"[PLAN_SKETCH]\n{json.dumps(item.plan_sketch, ensure_ascii=False)}\n\n"
        f"Tool calls:\n{json.dumps(item.tool_calls, ensure_ascii=False)}\n\n"
        f"Tool results:\n{json.dumps(item.tool_results, ensure_ascii=False)[:6000]}\n\n"
        f"Final [PLAN]:\n{json.dumps(item.final_plan, ensure_ascii=False)}\n\n"
        "Return JSON exactly: {\"pre_tool\":\"...\", \"post_tool\":\"...\"}. "
        "If there is no tool call, post_tool may be an empty string."
    )


def teacher_consistency_rules() -> str:
    return (
        "Consistency rules: use skip only for already-satisfied targets with direct evidence at the required destination. "
        "Do not say the whole goal is skipped unless every required target is already satisfied. "
        "If an observation is missing, weak, or incomplete, say the scene remains uncertain or needs confirmation; do not convert that into skip. "
        "If one target still needs motion, describe it as active rather than saying its handling, placement, or retrieval is skipped."
    )


def teacher_pre_prompt(item: PreparedTrajectory) -> str:
    visible_calls = sanitize_pre_goal_tool_calls(pre_goal_tool_calls(item))
    tool_guidance = (
        "Pre-goal tool calls are listed below; explain what uncertainty they should resolve before writing GOAL."
        if visible_calls
        else (
            "No pre-goal perception/catalog call is planned. Explain the goal-status reasoning from the image/prompt, "
            "and do not claim that detect, schedule, or any other tool is required before GOAL."
        )
    )
    return (
        "Write the PRE-TOOL think text for this L-NSDL trajectory.\n"
        "Important: you are seeing only the instruction, lean prompt, the image if provided, "
        "and any pre-goal detect calls. You cannot see the correct GOAL, ASSIGN, "
        "PLAN_SKETCH, tool results, or final PLAN. "
        "Infer which target statuses appear active or skipped from the image/prompt, "
        "but keep the text compact and do not mention labels or ground truth. "
        f"{teacher_consistency_rules()} "
        f"{tool_guidance} "
        "Use 1-3 compact English sentences.\n\n"
        f"Instruction:\n{item.instruction}\n\n"
        f"Lean user prompt:\n{item.user_prompt}\n\n"
        f"Pre-goal tool calls, without results:\n{json.dumps(visible_calls, ensure_ascii=False)}\n\n"
        "Return JSON exactly: {\"pre_tool\":\"...\"}."
    )


def teacher_grounding_prompt(item: PreparedTrajectory, *, pre_tool: str) -> str:
    # P5.10.3 v3 / G2: direct-judgment samples must commit the skip/active
    # decision in this think — the upcoming [GOAL] is final, no verify fallback.
    g2_direct = item.audit.get("trajectory_variant") == "g2_direct"
    crop_note = ""
    if item.audit.get("crop_demo"):
        crop_entities = ", ".join(item.audit.get("crop_entities") or [])
        crop_note = (
            f" The detect result also attached zoomed-in crop images of: {crop_entities}. "
            "Treat these crops as the decisive close-up visual evidence for those small items "
            "and cite what they settle (e.g. the item is clearly on the table, or clearly "
            "sitting elsewhere / absent from the table)."
        )
    judgment_rule = (
        (
            "For EACH instruction-relevant target, state the decisive observation — a returned "
            "relation like 'pear on table', a crop, or the ABSENCE of any relation placing the "
            "item at its destination — and conclude its skip/active status NOW. The upcoming "
            "[GOAL] must already carry the final judgment; do not defer the decision to any "
            "later verification or coverage check."
        )
        if g2_direct
        else "Explain how the observations support the active/skip status decision, then stop."
    )
    return (
        "Write the GROUNDING think text after pre-goal tool results and before emitting [GOAL].\n"
        "You can see only the instruction, lean prompt, previous think, and detect observations. "
        f"{judgment_rule}{crop_note} "
        "Do not mention labels, ground truth, task names, splits, or dataset construction. "
        f"{teacher_consistency_rules()} "
        "Use 1-3 compact English sentences.\n\n"
        f"Instruction:\n{item.instruction}\n\n"
        f"Lean user prompt:\n{item.user_prompt}\n\n"
        f"Previous pre-tool think:\n{pre_tool}\n\n"
        f"Pre-goal tool calls:\n{json.dumps(pre_goal_tool_calls(item), ensure_ascii=False)}\n\n"
        f"Pre-goal tool results:\n{json.dumps(pre_goal_tool_results(item), ensure_ascii=False)[:5000]}\n\n"
        "Return JSON exactly: {\"grounding_tool\":\"...\"}."
    )


def teacher_post_prompt(item: PreparedTrajectory, *, pre_tool: str) -> str:
    coverage_request = ""
    if item.coverage_repair:
        coverage_request = (
            " The trajectory includes a verify_goal_coverage conflict before the final plan; "
            "also write coverage_repair_tool explaining the contradiction and the corrected GOAL."
        )
    return (
        "Write the POST-TOOL think text for this L-NSDL trajectory.\n"
        "Now you can see the tool results and final plan. Briefly explain how the returned "
        "observations/schedule are integrated. Keep the narration consistent with the chosen GOAL/PLAN and do not say "
        "an active target or the whole goal is skipped unless the final trajectory actually leaves every target satisfied "
        "without further motion. Use 1-2 compact English sentences."
        f"{coverage_request}\n\n"
        f"Previous pre-tool think:\n{pre_tool}\n\n"
        f"Coverage repair record:\n{json.dumps(item.coverage_repair or {}, ensure_ascii=False)[:5000]}\n\n"
        f"Tool calls:\n{json.dumps(item.tool_calls, ensure_ascii=False)}\n\n"
        f"Tool results:\n{json.dumps(item.tool_results, ensure_ascii=False)[:6000]}\n\n"
        f"Final [PLAN]:\n{json.dumps(item.final_plan, ensure_ascii=False)}\n\n"
        "Return JSON exactly: {\"coverage_repair_tool\":\"...\", \"post_tool\":\"...\"}. "
        "If there is no coverage conflict, coverage_repair_tool may be an empty string."
    )


def teacher_repair_post_prompt(item: PreparedTrajectory, *, pre_tool: str) -> str:
    assert item.repair is not None
    first_results = first_post_goal_tool_results(item)
    coverage_request = ""
    if item.coverage_repair:
        coverage_request = (
            " The trajectory also includes a verify_goal_coverage conflict; coverage_repair_tool "
            "should explain how that diagnosis changes the GOAL before the schedule repair."
        )
    return (
        "Write think text for a repair trajectory after the first tool call fails.\n"
        "You can now see the failed schedule result, the repaired tool call, the successful "
        "tool result, and the final plan. The repair_tool field should explain the failure "
        "reason and the concrete sketch repair before re-calling schedule. The post_tool "
        "field should explain why the successful result is adopted. Keep the narration consistent with the final active targets; "
        "do not describe an active target, its retrieval, or the whole goal as skipped. Use compact English."
        f"{coverage_request}\n\n"
        f"Previous pre-tool think:\n{pre_tool}\n\n"
        f"Coverage repair record:\n{json.dumps(item.coverage_repair or {}, ensure_ascii=False)[:5000]}\n\n"
        f"First-turn post-GOAL tool calls:\n{json.dumps(first_post_goal_tool_calls(item), ensure_ascii=False)}\n\n"
        f"First-turn tool results:\n{json.dumps(first_results, ensure_ascii=False)[:5000]}\n\n"
        f"Repaired [PLAN_SKETCH]:\n{json.dumps(item.plan_sketch, ensure_ascii=False)}\n\n"
        f"Repair tool call:\n{json.dumps(item.repair['repair_call'], ensure_ascii=False)}\n\n"
        f"Successful tool result:\n{json.dumps(item.repair['repair_result'], ensure_ascii=False)[:5000]}\n\n"
        f"Final [PLAN]:\n{json.dumps(item.final_plan, ensure_ascii=False)}\n\n"
        "Return JSON exactly: {\"coverage_repair_tool\":\"...\", \"repair_tool\":\"...\", \"post_tool\":\"...\"}. "
        "If there is no coverage conflict, coverage_repair_tool may be an empty string."
    )


def parse_teacher_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    for candidate in extract_teacher_json_candidates(text):
        if isinstance(candidate, dict):
            return {str(k): v for k, v in candidate.items()}
    pre = ""
    post = ""
    think_blocks = re.findall(r"<think[^>]*>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_blocks:
        pre = think_blocks[0]
        post = think_blocks[1] if len(think_blocks) > 1 else ""
    else:
        pre = text
    return {"pre_tool": pre, "post_tool": post}


def extract_teacher_json_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            candidates.append(value)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates[::-1]


def apply_ad_checklist_thinks(
    prepared: list[PreparedTrajectory],
    *,
    counters: Counter[str],
) -> None:
    """P5.10.6 v3.1 (a)+(d): override pre/grounding thinks with deterministic
    non-committal question + evidence-checklist text (docs/debug_sftv3.md §5.3).

    Eligibility: the row has a pre-goal detect call and is NOT a coverage
    conflict-repair demo (those deliberately carry a wrong first [GOAL] that a
    checklist derived from evidence would contradict). post_tool / repair /
    coverage narration keeps the --think-source text so the teacher cache stays
    fully reusable.
    """
    for item in prepared:
        if item.teacher is None:
            continue
        if not item.pre_goal_tool_count or item.coverage_repair:
            item.audit["ad_checklist"] = False
            counters["ad_checklist:ineligible"] += 1
            continue
        detect_payload: dict[str, Any] | None = None
        for call, result in zip(item.tool_calls, item.tool_results):
            if call.get("name") == "detect" and isinstance(result, dict):
                detect_payload = result
                break
        candidates = (
            list(item.audit.get("query_candidates") or [])
            if item.audit.get("query_source") == "instruction"
            else None
        )
        pre = build_noncommittal_pre_think(
            item.goal,
            crop_entities=list(item.audit.get("crop_entities") or []),
            row_index=item.row_index,
            candidates=candidates,
        )
        grounding, evidence_audit = build_checklist_grounding_think(
            item.goal,
            detect_payload or {},
            row_index=item.row_index,
            candidates=candidates,
            # v3.3 (cluster 9): extra candidates the final plan actually uses
            # (carrier/tool) get observation-only lines, never an exclusionary
            # verdict that would contradict the PLAN in the same turn.
            plan_participants=plan_participant_names(item.final_plan),
        )
        if not pre or not grounding:
            item.audit["ad_checklist"] = False
            counters["ad_checklist:no_targets_keep_teacher"] += 1
            continue
        item.teacher.pre_tool = pre
        item.teacher.grounding_tool = grounding
        item.audit["ad_checklist"] = True
        item.audit["ad_checklist_evidence"] = evidence_audit
        counters["ad_checklist:applied"] += 1
        for key in (
            "evidence_relation",
            "evidence_elsewhere",
            "evidence_absence",
            "evidence_crop_missing",
            "fallback_skip_no_evidence",
            "fallback_active_conflict",
            "extra_present",
            "extra_other",
            "extra_participating",
        ):
            counters[f"ad_checklist_items:{key}"] += len(evidence_audit.get(key) or [])


def template_think(item: PreparedTrajectory) -> TeacherThink:
    tool_names = [call["name"] for call in first_turn_tool_calls(item)]
    if tool_names:
        tool_text = "The trajectory calls " + ", ".join(tool_names) + " because the sketch has nontrivial grounding or scheduling gaps."
        post = "The returned tool observations are consistent with the sketch, so the final plan adopts the verified operation DAG."
    else:
        tool_text = "The structure is direct enough to emit the final plan without a tool call."
        post = ""
    repair = ""
    if item.repair:
        repair = (
            "The first schedule result is infeasible because the sketch omits a custody source; "
            "the repair restores the missing acquire before re-calling schedule."
        )
    coverage = ""
    if item.coverage_repair:
        coverage = (
            "The coverage check flags an active target already observed at its destination, "
            "so the corrected GOAL marks that item skipped before planning."
        )
    pre = (
        "The instruction requires deciding each target's active or skipped status from the image/prompt before seeing labels or tool results. "
        f"{tool_text}"
    )
    grounding = ""
    if item.pre_goal_tool_count:
        grounding = (
            "The detect observations are checked before GOAL; active items remain those that need relocation, "
            "while skipped items are represented by explicit skip status reasons."
        )
    return TeacherThink(
        pre_tool=pre,
        grounding_tool=grounding,
        coverage_repair_tool=coverage,
        repair_tool=repair,
        post_tool=post,
        status="template",
        model="template",
    )


def teacher_cache_key(
    item: PreparedTrajectory,
    model: str,
    *,
    include_image: bool,
    post_include_image: bool,
    think_mode: str,
) -> str:
    payload = {
        "model": model,
        "include_image": include_image,
        "post_include_image": post_include_image,
        "think_mode": think_mode,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "goal_schema_version": GOAL_SCHEMA_VERSION,
        "task_id": item.task_id,
        "pre_goal_tool_count": item.pre_goal_tool_count,
        "goal": item.goal,
        "assign": item.assign,
        "sketch": item.plan_sketch,
        "tools": item.tool_calls,
        "coverage_repair": item.coverage_repair,
        "repair": item.repair,
        "plan": item.final_plan,
    }
    # LOCUS (2026-08-26): task_id above is the grid CELL, not a row identity.
    # Without this the key collapses distinct rows onto one cached think while
    # --teacher-include-image means the think describes a DIFFERENT render.
    # Measured on the first LOCUS build: 6300 rows -> 2136 distinct keys, 4647
    # of them served from cache (the other key fields vary within a cell, so
    # the collapse is ~3 rows per key on average, not the full 210).
    # Only set for LOCUS, so every VIKI key stays byte-identical.
    row_scope = getattr(item, "cache_row_scope", None)
    if row_scope is not None:
        payload["cache_row_scope"] = row_scope
    # v3.2: rows with instruction-derived queries key on the legacy call so the
    # v3.1 teacher cache keeps hitting (see prepare_trajectory).
    legacy_calls = getattr(item, "cache_legacy_tool_calls", None)
    if legacy_calls is not None:
        payload["tools"] = legacy_calls
    # P10.S2 --phi-off: the arm strips schedule (and with it the repair demo)
    # AFTER the trajectory is built, and keys on the pre-strip values so the v3.3
    # cache stays 100% reusable. Only `repair` needs an override here; `tools` is
    # already covered above, and every other key field is computed before the
    # schedule branch runs (see strip_phi_from_trajectory).
    if item.audit.get("phi_off"):
        payload["repair"] = getattr(item, "cache_legacy_repair", None)
    # v3 markers are only added when set so that unchanged v2.1 trajectories
    # keep their exact cache keys (surgical re-distillation of the changed
    # subset only; [plan.md P5.10.3] item 2).
    variant = item.audit.get("trajectory_variant")
    if variant:
        payload["trajectory_variant"] = variant
    if item.audit.get("crop_demo"):
        legacy_crops = getattr(item, "cache_legacy_crop_entities", None)
        payload["crop_entities"] = (
            legacy_crops if legacy_crops is not None else item.audit.get("crop_entities")
        )
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def build_output_row(item: PreparedTrajectory, *, args: argparse.Namespace) -> dict[str, Any]:
    teacher = item.teacher
    assert teacher is not None
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": item.system_prompt, "name": None},
        {"role": "user", "content": item.user_prompt, "name": None},
    ]

    pre_calls = pre_goal_tool_calls(item)
    pre_results = pre_goal_tool_results(item)
    for call_index, (call, result) in enumerate(zip(pre_calls, pre_results)):
        think_text = (
            teacher.pre_tool
            if call_index == 0
            else "I have one pre-GOAL observation and will request the next one before deciding target status."
        )
        messages.append({
            "role": "assistant",
            "content": render_tool_only_assistant(think_text, call),
            "name": None,
        })
        append_rendered_tool_message(messages, call, result)

    post_goal_calls = post_goal_tool_calls(item)
    post_goal_results = post_goal_tool_results(item)
    initial_think = teacher.grounding_tool if pre_calls else teacher.pre_tool
    structured_sections_emitted = False
    next_structured_think = initial_think

    if not post_goal_calls:
        messages.append({
            "role": "assistant",
            "content": render_initial_assistant(
                item,
                teacher,
                think_text=next_structured_think,
                tool_calls=[],
            ),
            "name": None,
        })
    else:
        for relative_index, (call, result) in enumerate(zip(post_goal_calls, post_goal_results)):
            absolute_index = item.pre_goal_tool_count + relative_index
            if is_coverage_repair_call(item, absolute_index):
                messages.append({
                    "role": "assistant",
                    "content": render_goal_coverage_check_assistant(
                        item,
                        teacher,
                        think_text=next_structured_think,
                    ),
                    "name": None,
                })
                append_rendered_tool_message(messages, call, result)
                next_structured_think = teacher.coverage_repair_tool or (
                    "The coverage check reports that an active item is already observed at its declared destination, so I correct the GOAL before planning."
                )
                structured_sections_emitted = False
                continue

            if is_schedule_repair_failed_call(item, absolute_index):
                content = render_initial_assistant(
                    item,
                    teacher,
                    think_text=next_structured_think,
                    plan_sketch=item.repair["failed_plan_sketch"] if item.repair else None,
                    tool_calls=[call],
                ) if not structured_sections_emitted else render_tool_only_assistant(
                    "I will test this schedule sketch and use the returned reason if it fails.",
                    call,
                )
                messages.append({"role": "assistant", "content": content, "name": None})
                append_rendered_tool_message(messages, call, result)
                structured_sections_emitted = True
                continue

            if is_schedule_repair_correct_call(item, absolute_index):
                messages.append({
                    "role": "assistant",
                    "content": render_repair_assistant(item, teacher),
                    "name": None,
                })
                append_rendered_tool_message(messages, call, result)
                structured_sections_emitted = True
                continue

            if not structured_sections_emitted:
                messages.append({
                    "role": "assistant",
                    "content": render_initial_assistant(
                        item,
                        teacher,
                        think_text=next_structured_think,
                        tool_calls=[call],
                    ),
                    "name": None,
                })
                structured_sections_emitted = True
            else:
                messages.append({
                    "role": "assistant",
                    "content": render_tool_only_assistant(tool_followup_think(call), call),
                    "name": None,
                })
            append_rendered_tool_message(messages, call, result)

        if structured_sections_emitted:
            messages.append({
                "role": "assistant",
                "content": render_final_assistant(item, teacher),
                "name": None,
            })
        else:
            messages.append({
                "role": "assistant",
                "content": render_initial_assistant(
                    item,
                    teacher,
                    think_text=next_structured_think,
                    tool_calls=[],
                ),
                "name": None,
            })

    # --- P5.10.3: persist crop pixels next to the dataset; message content
    # stays plain str (stable Arrow schema). The collator re-attaches crops to
    # the detect tool turn via crop_attachments_json at training time. ---
    crop_attachments: list[dict[str, Any]] = []
    if item.crop_observations:
        crop_dir = Path(args.output_dir) / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        detect_tool_index = 0  # crops only ever ride the (first) pre-goal detect
        # tool messages are appended in item.tool_calls order; the i-th tool
        # message in `messages` corresponds to item.tool_calls[i].
        tool_message_indices = [
            i for i, msg in enumerate(messages) if msg.get("role") == "tool"
        ]
        message_index = (
            tool_message_indices[detect_tool_index]
            if detect_tool_index < len(tool_message_indices)
            else None
        )
        for crop in item.crop_observations:
            crop_path = crop_dir / f"{item.row_index:05d}_{item.task_id}_{crop.entity}.png"
            crop.image.save(crop_path)
            crop_attachments.append({
                "tool_call_index": detect_tool_index,
                "message_index": message_index,
                "entity": crop.entity,
                "confidence": crop.confidence,
                "image_path": str(crop_path),
            })
        # Round-budget hard check ([plan.md P5.10.3]): crop trajectories must
        # fit MAX_TOOL_ROUNDS=4 -> at most 5 assistant turns incl. final [PLAN].
        n_assistant = sum(1 for msg in messages if msg.get("role") == "assistant")
        if n_assistant > 5:
            raise ValueError(
                f"crop trajectory exceeds round budget: {n_assistant} assistant "
                f"turns for task_id={item.task_id} (max 5 = 4 tool rounds + PLAN)"
            )

    return {
        "messages": messages,
        "image_path": item.image_path,
        "image_source_split": "train",
        "task_name": item.task_name,
        "task_id": item.task_id,
        "row_index": item.row_index,
        "modality": "vl",
        "prompt_variant": "lnsdl_agentic_lean",
        "tool_call_count": len(item.tool_calls),
        "tool_names": [call["name"] for call in item.tool_calls],
        "tool_calls_json": json.dumps(item.tool_calls, ensure_ascii=False, sort_keys=True),
        "tool_results_json": json.dumps(item.tool_results, ensure_ascii=False, sort_keys=True),
        "goal_json": json.dumps(item.goal, ensure_ascii=False, sort_keys=True),
        "goal_schema_version": GOAL_SCHEMA_VERSION,
        "assign_json": json.dumps(item.assign, ensure_ascii=False, sort_keys=True),
        "plan_sketch_json": json.dumps(item.plan_sketch, ensure_ascii=False, sort_keys=True),
        "plan_json": json.dumps(item.final_plan, ensure_ascii=False, sort_keys=True),
        "teacher_model": args.teacher_model,
        "teacher_endpoint": teacher_endpoint(args),
        "teacher_model_note": TEACHER_NOTE,
        "final_teacher_model": args.final_teacher_model,
        "teacher_think_mode": args.teacher_think_mode,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "teacher_include_image": bool(args.teacher_include_image),
        "teacher_post_include_image": bool(args.teacher_post_include_image),
        "teacher_input_modality": teacher_input_modality(args),
        "teacher_input_note": teacher_input_note(args),
        "teacher_status": teacher.status,
        "think_quality_flags": list(teacher.quality_flags),
        "schedule_policy": args.schedule_policy,
        "goal_coverage_policy": args.goal_coverage_policy,
        "status_curriculum": bool(args.status_curriculum),
        "pre_goal_tool_count": item.pre_goal_tool_count,
        "repair_injected": bool(item.repair),
        "goal_coverage_repair_injected": bool(item.coverage_repair),
        "coverage_repair_json": json.dumps(item.coverage_repair or {}, ensure_ascii=False, sort_keys=True),
        "repair_json": json.dumps(item.repair or {}, ensure_ascii=False, sort_keys=True),
        "audit_json": json.dumps(item.audit, ensure_ascii=False, sort_keys=True),
        # P5.10.3 v3 columns (empty/None on v2.1-compatible rows).
        "trajectory_variant": item.audit.get("trajectory_variant") or "",
        "crop_demo": bool(item.audit.get("crop_demo")),
        "crop_attachments_json": json.dumps(crop_attachments, ensure_ascii=False, sort_keys=True),
        # P5.10.6 v3.1 column (False on v3-compatible rows).
        "think_style": args.think_style,
        "ad_checklist": bool(item.audit.get("ad_checklist")),
        # P10.S2 column (False on every phi_on row).
        "phi_off": bool(item.audit.get("phi_off")),
    }


def render_initial_assistant(
    item: PreparedTrajectory,
    teacher: TeacherThink,
    *,
    think_text: str | None = None,
    plan_sketch: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    sketch_payload = plan_sketch if plan_sketch is not None else item.plan_sketch
    calls_payload = tool_calls if tool_calls is not None else item.tool_calls
    selected_think = think_text if think_text is not None else teacher.pre_tool
    parts = [
        f"<think>{selected_think}</think>",
        "[GOAL]",
        json.dumps(item.goal, ensure_ascii=False, sort_keys=True),
        "[ASSIGN]",
        json.dumps(item.assign, ensure_ascii=False, sort_keys=True),
        "[PLAN_SKETCH]",
        json.dumps(sketch_payload, ensure_ascii=False, sort_keys=True),
    ]
    for call in calls_payload:
        parts.extend(["[TOOL_CALL]", json.dumps(call, ensure_ascii=False, sort_keys=True)])
    if not calls_payload and not item.repair:
        parts.extend(["[PLAN]", json.dumps(item.final_plan, ensure_ascii=False, sort_keys=True)])
    return "\n".join(parts)


def render_tool_only_assistant(think_text: str, call: dict[str, Any]) -> str:
    return "\n".join([
        f"<think>{think_text}</think>",
        "[TOOL_CALL]",
        json.dumps(call, ensure_ascii=False, sort_keys=True),
    ])


def render_goal_coverage_check_assistant(
    item: PreparedTrajectory,
    teacher: TeacherThink,
    *,
    think_text: str,
) -> str:
    assert item.coverage_repair is not None
    call = item.coverage_repair["verify_call"]
    conflict_goal = item.coverage_repair["conflict_goal"]
    selected_think = think_text or teacher.grounding_tool or teacher.pre_tool
    return "\n".join([
        f"<think>{selected_think}</think>",
        "[GOAL]",
        json.dumps(conflict_goal, ensure_ascii=False, sort_keys=True),
        "[TOOL_CALL]",
        json.dumps(call, ensure_ascii=False, sort_keys=True),
    ])


def render_repair_assistant(item: PreparedTrajectory, teacher: TeacherThink) -> str:
    assert item.repair is not None
    think_text = teacher.repair_tool or (
        "The failed schedule indicates missing custody, so I restore the omitted acquire step and re-run schedule."
    )
    return "\n".join([
        f"<think>{think_text}</think>",
        "[ASSIGN]",
        json.dumps(item.assign, ensure_ascii=False, sort_keys=True),
        "[PLAN_SKETCH]",
        json.dumps(item.plan_sketch, ensure_ascii=False, sort_keys=True),
        "[TOOL_CALL]",
        json.dumps(item.repair["repair_call"], ensure_ascii=False, sort_keys=True),
    ])


def render_final_assistant(item: PreparedTrajectory, teacher: TeacherThink) -> str:
    think_text = teacher.post_tool or "The tool results are consistent with the sketch, so I can adopt the final operation DAG."
    return "\n".join([
        f"<think>{think_text}</think>",
        "[PLAN]",
        json.dumps(item.final_plan, ensure_ascii=False, sort_keys=True),
    ])


def append_rendered_tool_message(
    messages: list[dict[str, Any]],
    call: dict[str, Any],
    result: dict[str, Any],
) -> None:
    msg = tool_result_message(call["name"], result)
    messages.append({"role": msg["role"], "content": msg["content"], "name": msg.get("name")})


def is_coverage_repair_call(item: PreparedTrajectory, absolute_index: int) -> bool:
    return bool(
        item.coverage_repair
        and int(item.coverage_repair.get("tool_index", -1)) == absolute_index
    )


def is_schedule_repair_failed_call(item: PreparedTrajectory, absolute_index: int) -> bool:
    return bool(
        item.repair
        and int(item.repair.get("prefix_tool_count", -1)) == absolute_index
    )


def is_schedule_repair_correct_call(item: PreparedTrajectory, absolute_index: int) -> bool:
    return bool(
        item.repair
        and int(item.repair.get("prefix_tool_count", -1)) + 1 == absolute_index
    )


def tool_followup_think(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "").strip()
    if name == "schedule":
        return "The goal and sketch are now explicit; I will call schedule to complete deterministic dependencies and resource timing."
    if name == "verify_goal_coverage":
        return "I will check the declared GOAL against the observed facts before committing the plan."
    if name == "verify":
        return "I will ask for a structured feasibility diagnosis before finalizing."
    if name == "detect":
        return "I need one more scoped observation before deciding the next planning step."
    return "I will use the next tool result before finalizing."


def first_turn_tool_calls(item: PreparedTrajectory) -> list[dict[str, Any]]:
    if not item.repair:
        return item.tool_calls[:1]
    prefix = int(item.repair.get("prefix_tool_count") or 0)
    return item.tool_calls[: prefix + 1]


def first_turn_tool_results(item: PreparedTrajectory) -> list[dict[str, Any]]:
    if not item.repair:
        return item.tool_results[:1]
    prefix = int(item.repair.get("prefix_tool_count") or 0)
    return item.tool_results[: prefix + 1]


def pre_goal_tool_calls(item: PreparedTrajectory) -> list[dict[str, Any]]:
    return item.tool_calls[: item.pre_goal_tool_count]


def pre_goal_tool_results(item: PreparedTrajectory) -> list[dict[str, Any]]:
    return item.tool_results[: item.pre_goal_tool_count]


def post_goal_tool_calls(item: PreparedTrajectory) -> list[dict[str, Any]]:
    return item.tool_calls[item.pre_goal_tool_count :]


def post_goal_tool_results(item: PreparedTrajectory) -> list[dict[str, Any]]:
    return item.tool_results[item.pre_goal_tool_count :]


def first_post_goal_tool_calls(item: PreparedTrajectory) -> list[dict[str, Any]]:
    if not item.repair:
        return post_goal_tool_calls(item)[:1]
    prefix = int(item.repair.get("prefix_tool_count") or 0)
    return item.tool_calls[item.pre_goal_tool_count : prefix + 1]


def first_post_goal_tool_results(item: PreparedTrajectory) -> list[dict[str, Any]]:
    if not item.repair:
        return post_goal_tool_results(item)[:1]
    prefix = int(item.repair.get("prefix_tool_count") or 0)
    return item.tool_results[item.pre_goal_tool_count : prefix + 1]


def sanitize_pre_goal_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name == "detect":
            clean: dict[str, Any] = {"query": args.get("query", "")}
            if args.get("crop_entities"):
                # P5.10.3: crop requests are part of the demonstrated policy;
                # the teacher narrates why zoomed evidence is requested up front.
                clean["crop_entities"] = list(args["crop_entities"])
            sanitized.append({"name": name, "args": clean})
        else:
            sanitized.append({"name": name, "args": {}})
    return sanitized


def sample_preview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "task_name": row["task_name"],
        "tool_names": row["tool_names"],
        "repair_injected": row.get("repair_injected"),
        "messages": row["messages"],
        "audit": json.loads(row["audit_json"]),
    }


def _rows_system_prompt_sha256(rows: list[dict[str, Any]]) -> str | None:
    """sha256 of the single system prompt shared by every row, or None if they differ.

    Recorded so the Phi-off dataset and the Phi-off eval can be compared
    byte-for-byte: a mismatch means one side is on a stale prompt, which would look
    like a model effect rather than a caliber bug.
    """
    digests = {
        hashlib.sha256(str(row["messages"][0]["content"]).encode("utf-8")).hexdigest()
        for row in rows
        if row.get("messages")
    }
    return digests.pop() if len(digests) == 1 else None


def build_meta(
    rows: list[dict[str, Any]],
    counters: Counter[str],
    drop_examples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_task = Counter(row["task_name"] for row in rows)
    tool_name_counts: Counter[str] = Counter()
    tool_combo_counts: Counter[str] = Counter()
    teacher_status = Counter(row["teacher_status"] for row in rows)
    quality_flags: Counter[str] = Counter()
    schedule_exact = Counter()
    prompt_audit = Counter()
    schedule_decisions: Counter[str] = Counter()
    schedule_policy_counts: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    pre_goal_counts: Counter[int] = Counter()
    goal_coverage_counts: Counter[str] = Counter()
    goal_coverage_repair_counts: Counter[str] = Counter()
    goal_coverage_skipped_conflicts: Counter[str] = Counter()
    status_curriculum_counts: Counter[str] = Counter()
    assistant_tool_turn_counts: Counter[int] = Counter()
    goal_status_counts: Counter[str] = Counter()
    trajectory_variant_counts: Counter[str] = Counter()
    crop_counts: Counter[str] = Counter()
    f4_cross_turn_schedule_rows = 0
    crop_curriculum_rows = 0
    crop_other_rows = 0
    curriculum_rows = 0
    other_detect_rows = 0
    plan_faithful_violations = 0
    max_tool_calls_per_turn = 0
    by_task_tool_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        names = row["tool_names"]
        tool_name_counts.update(names)
        tool_combo_counts.update(["+".join(names) if names else "none"])
        quality_flags.update(row.get("think_quality_flags") or [])
        audit = json.loads(row["audit_json"])
        prompt_audit.update(["pass" if audit.get("task_blind_prompt") else "fail"])
        decision = str(audit.get("schedule_decision") or "unknown")
        schedule_decisions.update([decision])
        schedule_policy_counts.update([str(audit.get("schedule_policy") or row.get("schedule_policy") or "unknown")])
        repair_counts.update(["repair" if audit.get("repair_injected") else "no_repair"])
        pre_goal_counts.update([int(row.get("pre_goal_tool_count") or 0)])
        goal_coverage_counts.update(["used" if audit.get("goal_coverage_used") else "not_used"])
        goal_coverage_repair_counts.update(["repair" if audit.get("goal_coverage_repair_used") else "no_repair"])
        goal_coverage_skipped_conflicts.update([
            "skipped_unclosed_conflict" if audit.get("goal_coverage_skipped_unclosed_conflict") else "none"
        ])
        goal = json.loads(row["goal_json"])
        for target in goal.get("targets") or []:
            if isinstance(target, dict):
                goal_status_counts.update([str(target.get("status") or "")])
        plan_faithful_violations += len(audit.get("plan_faithful_violations") or [])
        status_curriculum_counts.update(["status_curriculum" if audit.get("status_curriculum_sample") else "regular"])
        trajectory_variant_counts.update([audit.get("trajectory_variant") or "none"])
        is_curriculum = bool(audit.get("status_curriculum_sample"))
        has_detect = "detect" in names
        if is_curriculum:
            curriculum_rows += 1
        elif has_detect:
            other_detect_rows += 1
        if audit.get("crop_demo"):
            crop_counts.update(["crop_demo"])
            if is_curriculum:
                crop_curriculum_rows += 1
            else:
                crop_other_rows += 1
            if audit.get("crop_missing"):
                crop_counts.update(["crop_missing_some"])
            if not audit.get("crop_produced"):
                crop_counts.update(["crop_produced_zero"])
        if audit.get("f4_cross_turn_schedule"):
            f4_cross_turn_schedule_rows += 1
        per_turn_counts = assistant_tool_call_counts(row.get("messages") or [])
        assistant_tool_turn_counts.update(per_turn_counts)
        if per_turn_counts:
            max_tool_calls_per_turn = max(max_tool_calls_per_turn, max(per_turn_counts))
        task_stats = by_task_tool_stats[row["task_name"]]
        task_stats.update(["rows"])
        if "schedule" in names:
            task_stats.update(["schedule_rows"])
        if audit.get("schedule_boundary"):
            task_stats.update(["boundary_rows"])
            if "schedule" in names:
                task_stats.update(["boundary_schedule_rows"])
        if audit.get("repair_injected"):
            task_stats.update(["repair_rows"])
        if "schedule_ops_exact_gt" in audit:
            schedule_exact.update(["exact" if audit["schedule_ops_exact_gt"] else "not_exact"])
        if audit.get("schedule_gt_reconstruct_failed"):
            schedule_exact.update(["gt_reconstruct_failed"])
        if "schedule_action_set_match_gt" in audit:
            schedule_exact.update(["action_set_match" if audit["schedule_action_set_match_gt"] else "action_set_mismatch"])
    by_task_tool_rates: dict[str, dict[str, Any]] = {}
    for task, counter in sorted(by_task_tool_stats.items()):
        rows_n = counter["rows"]
        boundary_n = counter["boundary_rows"]
        by_task_tool_rates[task] = {
            "rows": rows_n,
            "schedule_rows": counter["schedule_rows"],
            "schedule_rate": counter["schedule_rows"] / rows_n if rows_n else 0.0,
            "boundary_rows": boundary_n,
            "boundary_schedule_rows": counter["boundary_schedule_rows"],
            "boundary_schedule_rate": counter["boundary_schedule_rows"] / boundary_n if boundary_n else None,
            "repair_rows": counter["repair_rows"],
        }
    return {
        "created_by": "scripts/build_lnsdl_sft_data.py",
        "source_parquet": args.parquet,
        "rows": len(rows),
        "max_samples": args.max_samples,
        "sample_strategy": args.sample_strategy,
        "sketch_variant": args.sketch_variant,
        "teacher_model": args.teacher_model,
        "teacher_endpoint": teacher_endpoint(args),
        "teacher_model_note": TEACHER_NOTE,
        "final_teacher_model": args.final_teacher_model,
        "think_source": args.think_source,
        "teacher_think_mode": args.teacher_think_mode,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "trajectory_protocol_version": TRAJECTORY_PROTOCOL_VERSION,
        "goal_schema_version": GOAL_SCHEMA_VERSION,
        "teacher_include_image": bool(args.teacher_include_image),
        "teacher_post_include_image": bool(args.teacher_post_include_image),
        "teacher_input_modality": teacher_input_modality(args),
        "teacher_input_note": teacher_input_note(args),
        "schedule_policy": args.schedule_policy,
        "schedule_boundary_ratio": args.schedule_boundary_ratio,
        # v3.4 (plan.md A0g): which model-visible toolset the rendered system
        # prompt carries; v33 = legacy 5 tools, v34 = {detect, schedule, expand}.
        "toolset_contract": resolve_toolset_contract(),
        # P10.S2 arm identity. `phi_off_rows` should equal the row count and
        # tool_name_counts should show zero schedule/verify/expand; the sha256 is
        # what every eval artifact's `system_prompt_sha256` must match.
        "phi_off": bool(args.phi_off),
        "phi_off_rows": sum(1 for row in rows if row.get("phi_off")),
        "system_prompt_sha256": _rows_system_prompt_sha256(rows),
        "repair_rate": args.repair_rate,
        "goal_coverage_policy": args.goal_coverage_policy,
        "source": args.source,
        "locus_split": args.locus_split if args.source == "locus" else None,
        "locus_status_min_s": int(args.locus_status_min_s) if args.source == "locus" else None,
        "status_curriculum": bool(args.status_curriculum),
        "pre_goal_tools": bool(args.pre_goal_tools),
        "plan_faithful_violations": plan_faithful_violations,
        "teacher_status": dict(sorted(teacher_status.items())),
        "counters": dict(counters),
        "drop_examples": drop_examples,
        "by_task": dict(sorted(by_task.items())),
        "tool_call_distribution": dict(sorted(Counter(row["tool_call_count"] for row in rows).items())),
        "tool_name_counts": dict(sorted(tool_name_counts.items())),
        "goal_status_counts": dict(sorted(goal_status_counts.items())),
        "tool_combo_counts": dict(sorted(tool_combo_counts.items())),
        "schedule_decisions": dict(sorted(schedule_decisions.items())),
        "schedule_policy_counts": dict(sorted(schedule_policy_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "pre_goal_tool_count_distribution": dict(sorted(pre_goal_counts.items())),
        "goal_coverage_counts": dict(sorted(goal_coverage_counts.items())),
        "goal_coverage_repair_counts": dict(sorted(goal_coverage_repair_counts.items())),
        "goal_coverage_skipped_conflict_counts": dict(sorted(goal_coverage_skipped_conflicts.items())),
        "status_curriculum_counts": dict(sorted(status_curriculum_counts.items())),
        # P5.10.3 v3 (G2 + crop + F4) audit aggregates.
        "goal_coverage_direct_rate": float(getattr(args, "goal_coverage_direct_rate", 0.0)),
        "crop_demo_policy": str(getattr(args, "crop_demo_policy", "none")),
        "trajectory_variant_counts": dict(sorted(trajectory_variant_counts.items())),
        "crop_counts": dict(sorted(crop_counts.items())),
        "crop_call_rate_skip_family": (
            crop_curriculum_rows / curriculum_rows if curriculum_rows else 0.0
        ),
        "crop_call_rate_other": (
            crop_other_rows / other_detect_rows if other_detect_rows else 0.0
        ),
        "f4_cross_turn_schedule_rows": f4_cross_turn_schedule_rows,
        # P5.10.6 v3.1 (a)+(d) audit aggregates.
        "think_style": str(getattr(args, "think_style", "teacher")),
        # P5.10.6 v3.2 (cluster 8) aggregates.
        "query_source": str(getattr(args, "query_source", "goal")),
        "query_source_counts": dict(Counter(
            json.loads(row["audit_json"]).get("query_source") or "goal" for row in rows
        )),
        "ad_checklist_rows": sum(1 for row in rows if row.get("ad_checklist")),
        "ad_checklist_counters": {
            key.split(":", 1)[1]: count
            for key, count in sorted(counters.items())
            if key.startswith(("ad_checklist:", "ad_checklist_items:"))
        },
        "max_tool_calls_per_turn": max_tool_calls_per_turn,
        "assistant_tool_call_count_distribution": dict(sorted(assistant_tool_turn_counts.items())),
        "one_tool_call_per_turn": max_tool_calls_per_turn <= 1,
        "by_task_tool_rates": by_task_tool_rates,
        "think_quality_flags": dict(sorted(quality_flags.items())),
        "prompt_task_blind_audit": dict(sorted(prompt_audit.items())),
        "schedule_gt_audit": dict(sorted(schedule_exact.items())),
        "ood_val_task_names_seen": sorted(set(by_task).intersection(OOD_VAL_TASK_NAMES)),
        "notes": [
            "Only train.parquet and train/manifest.jsonl are read by this builder by default.",
            "This output directory name should encode whether the teacher API saw images; meta.teacher_include_image is authoritative.",
            "tool role messages are loss-masked by role; VL template rendering maps them to [TOOL_RESULT name] user text.",
            "Each assistant turn emits at most one [TOOL_CALL]; multiple tools are rendered as multi-turn interleaving.",
            "Final [PLAN] is the GT reverse-mapped DAG; schedule tool output is audited separately for exact-op and normalized action-set match.",
            "P5.8 v2 removes lookup from builder-injected tool calls; detect is the only pre-GOAL perception tool.",
            "P5.10.3 v3: crop pixels live under <output_dir>/crops/ and are referenced by "
            "crop_attachments_json (tool_call_index/message_index/entity/image_path); message "
            "content stays plain text for Arrow-schema stability — the VL SFT collator must "
            "re-attach crops as tool-turn image parts at training time (plan.md P5.10.3 item 4).",
            "P5.10.3 v3: trajectory_variant g2_direct = first [GOAL] carries the final skip/active "
            "judgment grounded in detect observations, no verify_goal_coverage; verify_retained = "
            "minority conflict-repair / confirm-ok demos (the latter keep the F4 cross-turn "
            "sketch->schedule(op_ids) pattern).",
        ],
    }


def assistant_tool_call_counts(messages: list[dict[str, Any]]) -> list[int]:
    return [
        str(message.get("content") or "").count("[TOOL_CALL]")
        for message in messages
        if str(message.get("role") or "") == "assistant"
    ]


def teacher_input_modality(args: argparse.Namespace) -> str:
    return "image_text" if args.teacher_include_image else "text_only_no_image"


def teacher_input_note(args: argparse.Namespace) -> str:
    if args.teacher_include_image:
        if args.teacher_think_mode == "blind" and not args.teacher_post_include_image:
            return (
                "The API teacher receives the task image during blind pre-tool think, "
                "then text-only tool results/final plan during post-tool think."
            )
        return "The API teacher receives the task image plus text, GT-derived structure, and tool results."
    return NO_IMAGE_TEACHER_INPUT_NOTE


def teacher_endpoint(args: argparse.Namespace) -> str:
    if args.api_base_url:
        return str(args.api_base_url)
    return str(get_endpoint_config(endpoint_index=args.endpoint_index).base_url or "")


def _op_to_dict(op: OperationStep, *, include_gaps: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "op_id": op.op_id,
        "op_type": op.op_type,
        "entity": op.entity,
        "dest": op.dest,
        "deps": list(op.deps or []),
        "robot": op.robot,
    }
    if op.params:
        payload["params"] = dict(op.params)
    if include_gaps and op.gaps:
        payload["gaps"] = list(op.gaps)
    return payload


def _micro_step_to_dict(step: Any) -> dict[str, Any]:
    if hasattr(step, "step") and hasattr(step, "actions"):
        return {"step": int(step.step), "actions": {k: list(v) for k, v in step.actions.items()}}
    if isinstance(step, dict):
        return dict(step)
    return asdict(step)


def normalize_think_text(text: str) -> str:
    text = re.sub(r"</?think[^>]*>", "", str(text or ""), flags=re.IGNORECASE)
    for block_name in ("GOAL", "ASSIGN", "PLAN_SKETCH", "TOOL_CALL", "PLAN"):
        text = text.replace(f"[{block_name}]", f"{block_name} block")
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    raise SystemExit(main())
