#!/usr/bin/env python3
"""Evaluate an L-NSDL agentic VL SFT checkpoint on VIKI-L2.

This runner differs from ``eval_viki_l2.py`` in one key place: the model is
allowed to emit ``[TOOL_CALL]`` blocks, tool results are executed and appended
to the chat, and only the final ``[PLAN]`` is lowered to the legacy dispatcher
artifact used by the existing VIKI-L2 scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
# G6. Imported unconditionally (it pulls nothing heavy) so the `except` clause
# in the sample loop is valid even on runs with no --api-model: a bare
# `except NameError`-shaped guard there would be a landmine.
from nsdl.api_backend import RelayTransportError as _RelayTransportError  # noqa: E402
from nsdl.benchmark_scoring import locus_m_depth_key  # noqa: E402
from nsdl.dispatcher import dispatch_graph  # noqa: E402
from nsdl.evaluation import build_error_artifact, build_result_artifact  # noqa: E402
from nsdl.image_pixel_cap import describe_image_pixel_cap  # noqa: E402
from nsdl.macro_graph import (  # noqa: E402
    _assign_executors_to_graph,
    collect_selected_robot_types,
)
from nsdl.prompt_adapter import build_sample_stem, load_benchmark_samples  # noqa: E402
from nsdl.phi_off_prompt import build_phi_off_system_prompt  # noqa: E402
from nsdl.planning_tool import phi_open_placement_mode, phi_scheduler_mode  # noqa: E402
from nsdl.profiles import VIKI_L2_PROFILE, use_benchmark_profile  # noqa: E402
from nsdl.scene_facts_refiner import build_scene_facts_refiner  # noqa: E402
from nsdl.scene_parser import build_scene_parser  # noqa: E402
from nsdl.schema import BenchmarkSample, DispatcherPayload, PipelineResult, SceneFacts  # noqa: E402
from nsdl.slm_interface import (  # noqa: E402
    SLMResponse,
    convert_operations_to_graph_plan,
    validate_operations,
)
from nsdl.toolset_contract import resolve_toolset_contract
from nsdl.tool_runtime import (  # noqa: E402
    PHI_TOOLS,
    ToolRuntimeLoopResult,
    _jsonable,
    forced_phi_micro_steps,
    run_agentic_tool_loop,
    slm_response_to_dict,
)
from nsdl.yolo_detector import (  # noqa: E402
    DEFAULT_DETECTOR_BACKEND,
    OPENVOCAB_DETECTOR_BACKEND,
    assert_detector_caliber_for_level,
    build_structured_detector,
    describe_detector_weights,
    detector_weights_for_level,
)


def backend_processor_for_provenance(backend: Any) -> Any:
    """Best-effort handle on the live processor, for cap provenance only.

    Reaches through the private ``_processor`` because that is where the local
    VL backend caches it and adding a public accessor would change the backend
    protocol that the P10.S2 cross-generation grid evaluates through. Returns
    ``None`` rather than raising for backends that hold no processor (API-backed
    arms), which ``describe_image_pixel_cap`` reports as source
    ``"unavailable"`` -- an honest "we could not tell", never a silent "full
    resolution".
    """
    return getattr(backend, "_processor", None)


DEFAULT_WEIGHTS = {
    "3b": REPO_ROOT / "checkpoints" / "lnsdl" / "sft" / "v2" / "qwen25-vl-3b-gpt56-terra-image-p58" / "final",
    "7b": REPO_ROOT / "checkpoints" / "lnsdl" / "sft" / "v2" / "qwen25-vl-7b-gpt56-terra-image-p58" / "final",
    # Qwen2-VL-2B (previous generation) exists only as the P10.S2 capacity-grid
    # cell -- there is no v2-era 2B run -- so its default points at the v3.3
    # seed42 checkpoint. In practice every launcher passes --slm-weights
    # explicitly; this entry exists mainly so "2b" is an accepted --model-size.
    "2b": REPO_ROOT / "checkpoints" / "lnsdl" / "sft" / "v3_3" / "qwen2-vl-2b-gpt56-terra-image-p513-obsbranch-seed42" / "final",
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "lnsdl" / "v2" / "viki_l2"
# Detector caliber is PER BENCHMARK and resolved from --level (facts:D3a/D3b,
# user ruling (a) 2026-09-05): VIKI-L1/L2 -> 20-class `checkpoints/best.pt`,
# LOCUS -> 34-class `yolo26l_locus_full34_v1.pt`. `None` here means "resolve
# from --level in main()"; a cross-caliber file is refused unless
# LNSDL_ALLOW_DETECTOR_CALIBER_MISMATCH=1. Never re-pin a path here.
DEFAULT_DETECTOR_WEIGHTS = None


class AgenticEvaluationError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        sample: BenchmarkSample,
        original_exc: Exception,
        initial_scene_facts: SceneFacts | None = None,
        scene_facts: SceneFacts | None = None,
        trace: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(str(original_exc))
        self.stage = stage
        self.sample = sample
        self.original_exc = original_exc
        self.initial_scene_facts = initial_scene_facts
        self.scene_facts = scene_facts
        self.yolo_detections = None
        self.trace = trace or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    # P10.S2's y-axis is the M depth at which the Phi-offload gain turns
    # positive, which needs LOCUS rows (M0-M4) rather than VIKI-L2's flat
    # families. Both benchmarks share one row schema, so the agentic loop below
    # is untouched -- only the split path differs. LOCUS splits are
    # test_id/ood_s/ood_m/ood_x (eval) and train.
    # VIKI-L1 runs the same agentic loop and is scored by the same official
    # judge (nsdl.benchmark_scoring.compute_viki_l1_score, already wired through
    # build_result_artifact). It is the facet-A-only slice: the answer is a set
    # of robots, with no schedule to get wrong, so it is the one place our
    # number sits on a published leaderboard without a scoring caveat.
    parser.add_argument("--level", choices=["VIKI-L2", "LOCUS", "VIKI-L1"],
                        default="VIKI-L2")
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--list-task-names", action="store_true")
    parser.add_argument("--model-size", choices=sorted(DEFAULT_WEIGHTS), default="3b")
    parser.add_argument(
        "--slm-weights",
        "--checkpoint-dir",
        dest="slm_weights",
        default=None,
        help=(
            "Local SFT checkpoint directory. May point either to the run directory "
            "(e.g. checkpoints/lnsdl/sft/qwen25-vl-7b-grok4-image-p4star) "
            "or directly to its final/ subdirectory. Overrides --model-size default."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--result-bucket",
        default=None,
        help="Result subdirectory name. Defaults to a sanitized name derived from --slm-weights.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--scene-parser-backend", choices=["rule"], default="rule")
    parser.add_argument("--scene-facts-refiner-backend", choices=["none", "llm", "auto"], default="none")
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    # A1 decoding hardening (opt-in; defaults preserve historical greedy decode).
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="HF repetition_penalty (e.g. 1.05) to damp greedy degeneration loops "
        "on long tool-call JSON. Default off (=greedy, unchanged).",
    )
    parser.add_argument(
        "--stop-string",
        action="append",
        default=None,
        dest="stop_strings",
        help="Stop string(s) for generation; repeatable. Recommended: '</pre>' to "
        "halt hallucinated markup after a tool call. Default off.",
    )
    parser.add_argument(
        "--detect-attach-crops",
        type=int,
        default=0,
        help="P5.10.2 probe: auto-attach ≤N crops (least-confident portable hits) "
        "to each model detect call that omits crop_entities. 0 = off (default). "
        "Harness-side protocol-enforcement (not learned), disclosed in metadata.",
    )
    parser.add_argument(
        "--detector-backend",
        choices=[
            "none",
            DEFAULT_DETECTOR_BACKEND,
            OPENVOCAB_DETECTOR_BACKEND,
            "off",
            "oracle",
        ],
        default=DEFAULT_DETECTOR_BACKEND,
        help="P5.10.4 perception bracket: 'off' answers detect with empty "
        "observations; 'oracle' answers from parquet GT initial state "
        "(eval-only, never a training/data-construction backend). "
        f"D2 generality bracket: {OPENVOCAB_DETECTOR_BACKEND!r} swaps in an "
        "off-the-shelf open-vocabulary detector (zero-shot, never fine-tuned); "
        "it needs --detector-data-yaml for the vocabulary.",
    )
    parser.add_argument(
        "--detector-data-yaml",
        default=None,
        help="Explicit data.yaml supplying the detector class vocabulary. The "
        "fine-tuned backend infers this from <run_dir>/args.yaml, but an "
        "off-the-shelf open-vocabulary checkpoint has no such file, so D2 must "
        "pass it -- an empty vocabulary would make that arm silently identical "
        "to --detector-backend off.",
    )
    parser.add_argument(
        "--truncate-first-tool-call",
        action="store_true",
        help="Cluster 10 guard (docs/debug_sftv3.md §8): truncate each generation "
        "to the first [TOOL_CALL], dropping a stacked [PLAN]/second call so the "
        "schedule executes and the next round emits the repaired plan. Harness-side "
        "protocol-enforcement (not learned), disclosed in metadata. Default off.",
    )
    parser.add_argument(
        "--force-phi-reschedule",
        action="store_true",
        help="P6.1 point 4-AR (always-call arm, plan.md A0d(2)): when an episode "
        "ends with a direct-emit [PLAN] and never executed a schedule call, force "
        "the model's [PLAN_SKETCH] ops through schedule and score the solver's "
        "completed plan instead (failed forced calls fall back to the model's own "
        "plan, mirroring the point-3 control). Samples that already called "
        "schedule are untouched, so vs the point-4 run this isolates exactly "
        "'on-demand vs always-call'. Harness-side protocol-enforcement (not "
        "learned), disclosed in metadata; the forced call is recorded separately "
        "from model tool calls so TC stays comparable. Default off.",
    )
    parser.add_argument(
        "--force-phi-reject-feedback",
        action="store_true",
        help="A0g② (2026-09-01, v3.4 eval semantics): when the forced schedule "
        "REJECTS the sketch, inject the failed call + ToolReason into the "
        "message stream as a real tool turn and give the model further rounds "
        "to repair (fallback to the model's own plan only once max_tool_rounds "
        "is exhausted, flagged fallback_after_feedback). Default off = frozen "
        "4-AR/W1 silent-fallback semantics, byte-identical. Requires "
        "--force-phi-reschedule.",
    )
    parser.add_argument(
        "--force-phi-scope",
        choices=("direct_emit", "always"),
        default="direct_emit",
        help="Which samples --force-phi-reschedule covers (plan.md A0f-W1). "
        "'direct_emit' (default) keeps the frozen point 4-AR semantics: only "
        "episodes that never executed a schedule call. 'always' forces every "
        "episode, which is what the A0f decision actually specifies (the final "
        "plan is always Phi-expanded). The two differ by a measured 17.9%% of "
        "samples on the 7B LOCUS main row -- episodes that DID call schedule "
        "mid-way but ended on a plan that is not Phi's output, and whose acc is "
        "0.0141 vs 0.6667 for the plan==Phi group. So 'direct_emit' reads as a "
        "LOWER BOUND on the A0f effect and must be labelled as such whenever "
        "the write-once trigger line is evaluated against it. Per-sample subset "
        "membership is recorded in the trace as "
        "forced_phi_reschedule.schedule_executed_before.",
    )
    parser.add_argument(
        "--force-phi-final-plan",
        choices=("phi_schedule", "reexpand_ops"),
        default="phi_schedule",
        help="plan.md A0k / facts:E24c (2026-09-04): what the forced arm scores as "
        "its final plan. 'phi_schedule' (default, the fix) = Phi's own micro-step "
        "schedule from the forced call. 'reexpand_ops' = the pre-fix behaviour: "
        "replace the model's ops with Phi's returned op-DAG and re-expand it "
        "through dispatch_graph -- which drops the same-robot end-effector "
        "serialization (it lives only in Phi's list schedule) and turned 10/180 "
        "7B test_id rows whose Phi schedule was byte-equal to GT into "
        "two-Grasps-before-Place failures. Kept only to reproduce frozen "
        "pre-fix buckets; the per-sample trace records final_plan_source.",
    )
    parser.add_argument(
        "--disable-phi",
        action="store_true",
        help="P10.S2 y-axis (eval-only): answer schedule/verify/expand with a "
        "`tool_unavailable` error so the model must produce the plan itself, "
        "while `detect` keeps working. Gives the no-offload arm of the "
        "Phi-offload gain with the trained front-end and the perception channel "
        "held fixed -- unlike --max-tool-rounds 0, which also suppresses the "
        "plan of every sample that opens with a tool call (~90%% here). "
        # `%%` is not a typo: argparse expands help through `help % vars(action)`,
        # so a bare `%` starts a conversion (`% he` -> float) and --help dies with
        # `TypeError: must be real number, not dict`.
        "Default off.",
    )
    parser.add_argument(
        "--phi-off-prompt",
        action="store_true",
        help="P10.S2 trained Phi-off arm: evaluate under the system prompt that "
        "never mentions schedule/verify/expand (nsdl/phi_off_prompt.py), which is "
        "the prompt the --phi-off SFT checkpoint was trained on. Point --slm-weights "
        "at that checkpoint and keep --disable-phi on: the tools are gone from the "
        "prompt, so the dead-tool call rate should be ~0, and that number is the "
        "evidence that the runtime-gate arm's dead-tool-robustness confound is gone. "
        "Default off.",
    )
    parser.add_argument(
        "--detect-strip-crops",
        action="store_true",
        help="Crop-off ablation arm (eval-only, Table 3 block 1): detect keeps "
        "entities/relations/confidences but every crop is dropped before the "
        "tool result is rendered, so the model gets the same symbolic evidence "
        "without the pixels it asked for. The checkpoint was trained WITH crop "
        "demonstrations, so this arm is a LOWER BOUND on crop value. Composes "
        "with --detector-backend; mutually exclusive with --detect-verdict. "
        "Default off.",
    )
    parser.add_argument(
        "--detect-verdict",
        action="store_true",
        help="P10.S1 A-axis inversion (eval-only): detect additionally returns an "
        "externalized `already_at_destination` verdict, i.e. the tool makes the "
        "state judgement instead of the model. Composes with --detector-backend "
        "{yolo,oracle} to form the matched-perception 2x2 (evidence-only vs "
        "verdict) x (YOLO vs GT perception). The verdict is derived from THIS "
        "call's own evidence plus the instruction-derived destination, so both "
        "arms see identical perception. Default off.",
    )
    parser.add_argument(
        "--detector-weights",
        default=None,
        help="Detector weights. Default: the caliber that belongs to --level "
        "(VIKI-L1/L2 -> 20-class checkpoints/best.pt; LOCUS -> 34-class "
        "yolo26l_locus_full34_v1.pt; facts:D3a). A file from the other caliber is "
        "refused unless LNSDL_ALLOW_DETECTOR_CALIBER_MISMATCH=1.",
    )
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.70)
    parser.add_argument("--detector-max-det", type=int, default=64)

    # ---- G6: closed model + our full tool stack (docs/plan.md G6) ---------
    # The row asks "hand a much stronger closed model the SAME structural parts
    # (detect + schedule + expand, through this same tool_runtime) -- where does
    # it land?". Facet B has never been given to any baseline, so without this
    # the only answer on record is P6.1 point 2 (untrained base 7B + Phi), which
    # a reviewer would rightly call a weak target.
    parser.add_argument(
        "--api-model", default=None,
        help="Run the agentic loop against a relay-hosted closed model instead "
             "of a local checkpoint (e.g. gpt-5.6-terra). Skips VLM loading; "
             "--slm-weights/--model-size are then unused.",
    )
    parser.add_argument("--api-lines", nargs="*", default=["RELAY_A", "RELAY_A2"],
                        help="api_keys.json entries to alternate between (one "
                             "relay, two quotas -- dispatch only, not two sources)")
    parser.add_argument("--api-timeout", type=int, default=120,
                        help="Per-attempt timeout. 120 s, not 60: success "
                             "latency tracks item difficulty (median 21.8 s on "
                             "test_id vs 56.3 s on ood_x), so a tight timeout "
                             "preferentially drops the HARD splits and flatters "
                             "whatever survives (benchmark.md 7.4).")
    # 6 with backoff, not 4 back-to-back: relay image faults cluster in
    # dark windows (audits/relay_image_payload_probe.json).
    parser.add_argument("--api-max-attempts", type=int, default=6)
    parser.add_argument("--api-max-tokens", type=int, default=1500)
    parser.add_argument(
        "--no-zero-shot-supplement", dest="zero_shot_supplement",
        action="store_false", default=True,
        help="Do NOT append the worked-turn / format supplement to the system "
             "prompt under --api-model. Default is to append it: benchmark.md "
             "7.12(3) discipline (1) forbids making this row a protocol-"
             "transfer straw man, and our prompt documents the schema without "
             "ever showing a turn (a fine-tuned model learned that from 6300 "
             "demonstrations). Turning it off is an ABLATION of the supplement, "
             "not the headline G6 configuration.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.detector_weights is None:
        args.detector_weights = str(detector_weights_for_level(args.level))

    if args.api_model:
        # No local checkpoint exists for this arm. Keep the two provenance
        # fields populated with the model identity rather than a path so every
        # downstream reader (bucket naming, paired reports, the aggregator)
        # keeps working without a special case -- and so a G6 row can never be
        # mistaken for a checkpoint row by looking at `slm_weights`.
        requested_weights = slm_weights = Path(f"api:{args.api_model}")
        bucket = args.result_bucket or ("api_" + sanitize_bucket_name(args.api_model))
    else:
        requested_weights = Path(args.slm_weights) if args.slm_weights else DEFAULT_WEIGHTS[args.model_size]
        slm_weights = resolve_local_slm_weights_path(requested_weights)
        bucket = args.result_bucket or default_result_bucket(requested_weights, slm_weights)
    output_dir = Path(args.output_dir) / bucket / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _select_samples(args)
    if args.list_task_names:
        for task_name, count in summarize_task_names(samples):
            print(f"{task_name}\t{count}")
        return
    if not samples:
        raise SystemExit("No samples selected for evaluation.")

    api_backend = None
    system_prompt_override = None
    if args.api_model:
        from nsdl.api_backend import (  # noqa: WPS433
            RelayChatBackend,
            build_zero_shot_system_prompt,
        )

        backend = api_backend = RelayChatBackend(
            args.api_model,
            lines=args.api_lines,
            timeout=args.api_timeout,
            max_attempts=args.api_max_attempts,
            max_tokens=args.api_max_tokens,
        )
        if args.zero_shot_supplement:
            system_prompt_override = build_zero_shot_system_prompt()
        print(f"[L-NSDL] API backend: {api_backend.describe()} "
              f"zero_shot_supplement={args.zero_shot_supplement}", flush=True)
    else:
        load_local_vlm_with_fallback(str(slm_weights), args.device)
        from nsdl.slm_backends import get_local_vl_backend  # noqa: WPS433

        backend = get_local_vl_backend()
    detector = None
    detect_override_builder = None
    detect_postprocess_builder = None
    if args.detect_verdict:
        # P10.S1 A-axis inversion (eval-only): post-process whatever the
        # configured perception backend returned, so this arm and its
        # evidence-only control share byte-identical perception.
        os.environ["NSDL_VERDICT_PERCEPTION"] = "eval-only"
        from nsdl.verdict_perception import make_verdict_postprocess  # noqa: WPS433

        def detect_postprocess_builder(sample: BenchmarkSample, scene_facts: Any):  # noqa: ARG001
            return make_verdict_postprocess(scene_facts, source=args.detector_backend)

    if args.detect_strip_crops:
        if args.detect_verdict:
            raise SystemExit("--detect-strip-crops and --detect-verdict are mutually exclusive")
        from nsdl.perc_tool import strip_crops_postprocess  # noqa: WPS433

        def detect_postprocess_builder(sample: BenchmarkSample, scene_facts: Any):  # noqa: ARG001
            return strip_crops_postprocess

    if args.detector_backend == "off":
        # P5.10.4 bracket lower bound: detect returns empty observations.
        from nsdl.perc_tool import DetectResult  # noqa: WPS433

        def detect_override_builder(sample: BenchmarkSample):  # noqa: ARG001
            return lambda query=None, crop_entities=None: DetectResult()
    elif args.detector_backend == "oracle":
        # P5.10.4 bracket upper bound: GT-derived relations, eval-only.
        os.environ["NSDL_ORACLE_PERCEPTION"] = "eval-only"
        from nsdl.evaluation import get_sample_ground_truth  # noqa: WPS433
        from nsdl.oracle_perception import (  # noqa: WPS433
            make_oracle_detect_override,
            oracle_scope_mode,
            oracle_vocab_mode,
        )

        # W3 (2026-09-07): LNSDL_ORACLE_SCOPE=query (frozen default) | full.
        # W4 (2026-09-11): LNSDL_ORACLE_VOCAB=gt (frozen default) | yolo.
        # Resolved once here so a typo fails the run instead of silently
        # producing a default-mode bucket under a "full"/"yolovocab" name.
        oracle_scope = oracle_scope_mode()
        oracle_vocab = oracle_vocab_mode()

        def detect_override_builder(sample: BenchmarkSample):
            ground_truth = get_sample_ground_truth(sample)
            ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
            scene_facts_for_scope = None
            if oracle_scope == "full":
                # Same rule parser the agentic loop runs (image_used=false,
                # deterministic), so the instruction-relevant scope is exactly
                # the entity lists that land in the artifact's `scene_facts`.
                with use_benchmark_profile(VIKI_L2_PROFILE):
                    scene_facts_for_scope = scene_parser.parse(sample)
            return make_oracle_detect_override(
                ground_truth,
                scope_mode=oracle_scope,
                instruction=getattr(sample, "instruction", None),
                scene_facts=scene_facts_for_scope,
                vocab_mode=oracle_vocab,
            )
    elif args.detector_backend != "none":
        assert_detector_caliber_for_level(args.detector_weights, args.level, caller="eval_lnsdl_agentic_viki_l2")
        detector = build_structured_detector(
            args.detector_backend,
            weights_path=args.detector_weights,
            data_yaml_path=args.detector_data_yaml,
            confidence_threshold=args.detector_conf,
            iou_threshold=args.detector_iou,
            max_det=args.detector_max_det,
            device=args.detector_device,
        )
        # Preflight. The detector loads lazily on the first `detect` tool call,
        # deep inside the agentic loop, where a failure becomes an ordinary
        # `{"ok": false}` tool result the model is expected to cope with. The run
        # then completes with result_status=ok and plausible-looking scores while
        # perception was dead the whole time -- 2026-08-28 on the secondary
        # machine that cost a full parity-gate cycle, twice (missing
        # checkpoints/best.pt, then cv2 unable to load libGL.so.1). A requested
        # detector that cannot load is a broken run, not a degraded one, so fail
        # here where the message is still legible.
        try:
            detector.warmup()
        except Exception as exc:  # noqa: BLE001 -- re-raised immediately
            raise SystemExit(
                f"[L-NSDL] detector backend {args.detector_backend!r} was requested but "
                f"cannot be loaded, so this run would score with perception disabled.\n"
                f"  weights: {args.detector_weights}\n"
                f"  cause  : {type(exc).__name__}: {exc}\n"
                f"Pass --detector-backend none if a perception-free run is what you meant."
            ) from exc

    scene_parser = build_scene_parser(args.scene_parser_backend)
    scene_refiner = build_scene_facts_refiner(args.scene_facts_refiner_backend)

    existing_outputs = sum(
        1
        for sample in samples
        if (output_dir / f"{build_sample_stem(sample)}.json").exists()
    )
    print(
        f"[L-NSDL/{args.split}] start total={len(samples)} bucket={bucket} "
        f"weights={slm_weights} output_dir={output_dir} "
        f"tool_rounds={args.max_tool_rounds} detector={args.detector_backend} "
        f"shard={args.shard_index}/{args.num_shards} existing_outputs={existing_outputs}",
        flush=True,
    )
    print(
        f"[L-NSDL/{args.split}] task_name_counts="
        f"{', '.join(f'{name}:{count}' for name, count in summarize_task_names(samples))}",
        flush=True,
    )

    processed = skipped = errors = 0
    # G6: relay transport faults are a THIRD denominator, never merged into the
    # other two (benchmark.md 7.5). Such a sample is not written at all, so a
    # later pass re-collects it -- which is exactly the multi-pass retry the
    # relay's blackout-shaped failures require: a big inline retry cap just
    # burns budget inside one dark window.
    transport_missing = 0
    started = time.time()
    log_every = max(int(args.log_every), 1)
    for position, sample in enumerate(samples, start=1):
        output_path = output_dir / f"{build_sample_stem(sample)}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if position == 1 or position % log_every == 0:
            print(f"[L-NSDL/{args.split}] begin {position}/{len(samples)} {output_path.name}", flush=True)

        sample_started = time.time()
        trace: dict[str, Any] | None = None
        # Built outside the try so its provenance attributes (oracle scope
        # mode / per-sample unobservable list) are recordable even on error.
        detect_override = detect_override_builder(sample) if detect_override_builder else None
        try:
            result, trace = run_agentic_sample(
                sample,
                backend=backend,
                detector=detector,
                scene_parser=scene_parser,
                scene_refiner=scene_refiner,
                max_tool_rounds=args.max_tool_rounds,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_strings=args.stop_strings,
                detect_attach_crops=args.detect_attach_crops,
                truncate_first_tool_call=args.truncate_first_tool_call,
                detect_override=detect_override,
                detect_postprocess_builder=detect_postprocess_builder,
                disable_phi=args.disable_phi,
                phi_off_prompt=args.phi_off_prompt,
                system_prompt_override=system_prompt_override,
                force_phi_reschedule=args.force_phi_reschedule,
                force_phi_scope=args.force_phi_scope,
                force_phi_reject_feedback=args.force_phi_reject_feedback,
                force_phi_final_plan=args.force_phi_final_plan,
            )
            artifact = build_result_artifact(result, validate_l2=not args.skip_validate)
            artifact["result_status"] = "ok"
        except _RelayTransportError as exc:
            # NOT a model failure: the prompt demonstrably never got an answer.
            # Scoring it 0 would read a dark relay window as a wrong plan.
            transport_missing += 1
            if api_backend is not None:
                api_backend.take_records()
            print(f"[L-NSDL/{args.split}] transport-missing {output_path.name}: {exc}",
                  flush=True)
            continue
        except Exception as exc:  # noqa: BLE001
            errors += 1
            wrapped = exc if isinstance(exc, AgenticEvaluationError) else AgenticEvaluationError(
                stage="unknown",
                sample=sample,
                original_exc=exc,
            )
            artifact = build_error_artifact(sample, wrapped)
            artifact["result_status"] = "error"
            trace = getattr(wrapped, "trace", None) or trace

        artifact["runtime_seconds"] = round(time.time() - sample_started, 3)
        artifact["slm_modality"] = "vl"
        artifact["agentic_eval"] = {
            "enabled": True,
            "log_schema_version": 2,
            "max_tool_rounds": args.max_tool_rounds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "stop_strings": args.stop_strings,
            "slm_weights": str(slm_weights),
            "requested_slm_weights": str(requested_weights),
            "result_bucket": bucket,
            "detector_backend": args.detector_backend,
            # Which YOLO actually ran (path / sha256 / class count). Absent from
            # every bucket before 2026-09-05, which is why the full20-vs-full34
            # mismatch had to be reverse-engineered from class names.
            **(
                describe_detector_weights(args.detector_weights)
                if args.detector_backend not in {"none", "off", "oracle"}
                else {"detector_weights": None}
            ),
            "detect_attach_crops": args.detect_attach_crops,
            # W3 oracle scope (env LNSDL_ORACLE_SCOPE; None unless the backend is
            # `oracle`). `oracle_scope_detail` is per-sample provenance for the
            # `full` mode: the instruction-relevant scope actually used and the
            # `unobservable_entities` (opaque-cabinet S4 facts a frame cannot
            # show) that the evidence-conditional audit splits out.
            "oracle_scope": getattr(detect_override, "oracle_scope", None),
            # W4 oracle vocabulary (env LNSDL_ORACLE_VOCAB; `gt` = the frozen
            # GT-location word forms, `yolo` = only forms the YOLO relation
            # builder emits). None unless the backend is `oracle`.
            "oracle_vocab": getattr(detect_override, "oracle_vocab", None),
            "oracle_scope_detail": getattr(detect_override, "oracle_scope_detail", None),
            # P10.S1 arm label: (evidence-only | verdict) x (yolo | oracle).
            "detect_verdict": args.detect_verdict,
            "detect_strip_crops": args.detect_strip_crops,
            "a_axis_arm": (
                f"verdict_{args.detector_backend}"
                if args.detect_verdict
                else f"evidence_{args.detector_backend}"
            ),
            "truncate_first_tool_call": args.truncate_first_tool_call,
            # P6.1 point 4-AR always-call arm; per-sample record lives in the
            # trace under `forced_phi_reschedule` (outcome/source/latency).
            "force_phi_reschedule": args.force_phi_reschedule,
            # A0f-W1: which samples the forcing covers. "direct_emit" = frozen
            # point 4-AR semantics (never-called-schedule only) and reads as a
            # LOWER BOUND on the A0f effect; "always" = the A0f decision itself.
            "force_phi_scope": args.force_phi_scope,
            # A0g② v3.4 eval semantics: rejected forced calls become ToolReason
            # feedback turns instead of silent fallback (default off = frozen).
            "force_phi_reject_feedback": args.force_phi_reject_feedback,
            # A0k (facts:E24c): forced arm's final plan = Phi's own schedule
            # (fix) vs the pre-fix op-DAG re-expansion (legacy, reproduction only).
            "force_phi_final_plan": args.force_phi_final_plan,
            # v3.4 (plan.md A0g①②): which model-visible toolset the system
            # prompt carried (env LNSDL_TOOLSET_CONTRACT; v33 = legacy 5 tools,
            # v34 = {detect, schedule, expand}). Paired reports must refuse to
            # compare buckets across contracts, same as the pixel-cap rule.
            "toolset_contract": resolve_toolset_contract(),
            "phi_scheduler": phi_scheduler_mode(),
            "phi_open_placement": phi_open_placement_mode(),
            # G6 arm identity. `api_source` is a required column for every
            # third-party row (benchmark.md 7-7): relay pre-runs and official
            # API re-runs are two rows, both disclosed, never cherry-picked.
            **(
                {
                    "api_backend": api_backend.describe(),
                    "api_source": api_backend.api_source,
                    "zero_shot_supplement": args.zero_shot_supplement,
                    # Retry rate must sit next to acc, not in a debug log: a row
                    # collected through 3 retries per turn is a different
                    # measurement from one collected first-try.
                    "api_transport": _summarize_api_records(api_backend.take_records()),
                }
                if api_backend is not None
                else {}
            ),
            # Input-resolution cap actually in force (A0b, 2026-08-26; the
            # recording defect below fixed 2026-08-29). The detector-parity
            # incident showed the cost of NOT recording a knob: `detector_weights`
            # is absent from every bucket, so which YOLO ran had to be
            # reverse-engineered from the class names that appeared.
            #
            # This block used to record `LNSDL_IMAGE_MAX_PIXELS` -- the request,
            # not the effect -- and read `None` as "full 1920x1080". That is
            # wrong whenever the cap rides in on the checkpoint instead of the
            # env: `save_pretrained` bakes the capped `max_pixels` into
            # `preprocessor_config.json`, so an eval that does not re-export the
            # env var still runs capped. Bucket
            # `locus_g1_mainrow_cap524288_7b_seed42` recorded `None` for all 560
            # rows while running at 646 vision tokens. We now interrogate the
            # live processor; `image_pixel_cap_schema: 2` marks buckets written
            # after the fix, because a schema-1 `null` is irrecoverably ambiguous
            # between "full resolution" and "capped and we missed it".
            **describe_image_pixel_cap(backend_processor_for_provenance(backend)),
            # P10.S2 y-axis arm label. Disclosed per-sample so the two arms can
            # never be silently pooled when a bucket is re-aggregated.
            "disable_phi": args.disable_phi,
            "phi_off_prompt": args.phi_off_prompt,
            # Three arms, not two. `phi_off` is a runtime gate over a checkpoint
            # that was trained WITH Phi, so it also measures how gracefully the
            # model handles a dead tool -- a nuisance that is itself capacity-
            # correlated, which is why it only supports a LOWER BOUND on Phi's
            # contribution. `phi_off_trained` is the in-distribution ablation.
            "b_axis_arm": (
                "phi_off_trained"
                if args.phi_off_prompt
                else ("phi_off" if args.disable_phi else "phi_on")
            ),
            "trace": trace,
        }
        artifact["agentic_eval"]["analysis"] = build_agentic_analysis(artifact, trace)
        output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, default=_json_fallback),
            encoding="utf-8",
        )
        processed += 1

        if position == 1 or position % log_every == 0:
            print(
                f"[L-NSDL/{args.split}] done {position}/{len(samples)} "
                f"processed={processed} skipped={skipped} errors={errors} "
                f"last={output_path.name} runtime_s={artifact['runtime_seconds']}",
                flush=True,
            )

    print(
        f"[L-NSDL/{args.split}] done processed={processed} skipped={skipped} "
        f"errors={errors} elapsed_s={round(time.time() - started, 2)} output_dir={output_dir}",
        flush=True,
    )
    if transport_missing:
        # Printed as its own line, never folded into `errors`: the two answer
        # different questions ("the model got it wrong" vs "the model never
        # answered"). Re-running the same command collects only these, because
        # the written samples are skipped by file existence.
        print(
            f"[L-NSDL/{args.split}] transport-missing={transport_missing} "
            f"(NOT scored, NOT written -- re-run this command in a later window "
            f"to collect them; see benchmark.md 7.5 three denominators)",
            flush=True,
        )

    report_per_family(output_dir, split=args.split, level=args.level)


def _family_key(stem_or_task_id: str | None) -> str:
    """Extract the OOD family suffix (e.g. ``11-1``) from a task_id/stem.

    VIKI-L2 OOD ``val`` only has families ``11-1/11-2/11-3``; reporting per
    family is required so a single window's accidental hits are not mistaken for
    capability ([debug.md] cluster 5 / [test_analysis.md]).
    """
    if not stem_or_task_id:
        return "<unknown>"
    return str(stem_or_task_id).split("_")[-1]


def report_per_family(output_dir: Path, *, split: str, level: str = "VIKI-L2") -> None:
    """Aggregate per-sample result JSONs by OOD family and print L1/L2 averages.

    Scans ``output_dir`` (authoritative, also covers resumed/skipped samples)
    and prints one line per family so OOD ``val`` (11-1/11-2/11-3) is always
    reported separately, not just as a single aggregate.

    For ``level="LOCUS"`` the grouping switches to the M depth and the score
    column switches to ``locus_l2_acc`` -- the official judge zeroes every
    null-plan row, so reading LOCUS off ``viki_l2_plan_score`` would understate
    it and would not match how the strong baselines were scored.

    For ``level="VIKI-L1"`` there is no OOD-family suffix in the stem (they are
    plain ``00000``-style indices), so the VIKI-L2 key would emit one family per
    sample. Group by the size of the ground-truth robot set instead: that is the
    only structural axis L1 has, and a set-valued answer gets monotonically
    harder with it. The L2 column is dropped from the print -- L1 rows have no
    schedule, so a 0.0000 there would read as a failure rather than as N/A.
    """
    is_locus = level == "LOCUS"
    is_l1 = level == "VIKI-L1"
    score_key = "locus_l2_acc" if is_locus else "viki_l2_plan_score"
    stats: dict[str, dict[str, float]] = {}
    for path in sorted(output_dir.glob("*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if is_locus:
            family = locus_m_depth_key(artifact.get("ground_truth"))
        elif is_l1:
            gt_robots = artifact.get("ground_truth_selected_robots")
            family = f"|A|={len(gt_robots)}" if isinstance(gt_robots, list) else "<no-gt>"
        else:
            family = _family_key(path.stem)
        analysis = (artifact.get("agentic_eval") or {}).get("analysis") or {}
        scores = analysis.get("scores") or {}
        l2 = scores.get(score_key)
        l1 = scores.get("viki_l1_robot_activation")
        if l2 is None:
            l2 = (artifact.get("benchmark_scores") or {}).get(score_key, 0.0)
        if l1 is None:
            l1 = (artifact.get("benchmark_scores") or {}).get("viki_l1_robot_activation", 0.0)
        bucket = stats.setdefault(family, {"n": 0.0, "l1_sum": 0.0, "l2_sum": 0.0, "err": 0.0})
        bucket["n"] += 1
        bucket["l1_sum"] += float(l1 or 0.0)
        bucket["l2_sum"] += float(l2 or 0.0)
        if artifact.get("result_status") == "error":
            bucket["err"] += 1

    if not stats:
        return

    total_n = sum(b["n"] for b in stats.values())
    total_l2 = sum(b["l2_sum"] for b in stats.values())
    total_l1 = sum(b["l1_sum"] for b in stats.values())
    print(f"[L-NSDL/{split}] per-family report (n={int(total_n)}):", flush=True)
    for family in sorted(stats):
        b = stats[family]
        n = b["n"]
        l2_col = "" if is_l1 else f"L2_avg={b['l2_sum'] / n:.4f} "
        print(
            f"[L-NSDL/{split}]   family={family} n={int(n)} "
            f"L1_avg={b['l1_sum'] / n:.4f} {l2_col}"
            f"errors={int(b['err'])}",
            flush=True,
        )
    overall_l2 = "" if is_l1 else f" L2_avg={total_l2 / total_n:.4f}"
    print(
        f"[L-NSDL/{split}] overall L1_avg={total_l1 / total_n:.4f}{overall_l2}",
        flush=True,
    )


def _summarize_api_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-sample relay transport story: turns, attempts, statuses, latency.

    ``attempts > turns`` is the retry rate for this sample. Statuses are kept
    verbatim (``canned`` / ``EMPTY`` / ``HTTPnnn`` / exception class name)
    because they say *which* fault mode the window had, and the three modes
    have different implications: ``canned`` means the prompt never reached the
    model at all, ``EMPTY`` means it was billed with no content.
    """
    if not records:
        return {}
    ok = [r for r in records if r["status"] == "ok"]
    lat = sorted(r["latency_s"] for r in records)
    return {
        "turns": len(ok),
        "attempts": len(records),
        "retries": len(records) - len(ok),
        "statuses": dict(Counter(r["status"] for r in records)),
        "lines": dict(Counter(r["line"] for r in records)),
        "latency_s_total": round(sum(r["latency_s"] for r in records), 2),
        "latency_s_median": lat[len(lat) // 2],
    }


def run_agentic_sample(
    sample: BenchmarkSample,
    *,
    backend: Any,
    detector: Any,
    scene_parser: Any,
    scene_refiner: Any,
    max_tool_rounds: int,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float | None = None,
    stop_strings: list[str] | None = None,
    detect_attach_crops: int = 0,
    truncate_first_tool_call: bool = False,
    detect_override: Any | None = None,
    detect_postprocess_builder: Any | None = None,
    disable_phi: bool = False,
    phi_off_prompt: bool = False,
    system_prompt_override: str | None = None,
    force_phi_reschedule: bool = False,
    force_phi_scope: str = "direct_emit",
    force_phi_reject_feedback: bool = False,
    force_phi_final_plan: str = "phi_schedule",
) -> tuple[PipelineResult, dict[str, Any]]:
    initial_scene_facts = None
    scene_facts = None
    trace: dict[str, Any] = {}
    stage = "profile"
    try:
        with use_benchmark_profile(VIKI_L2_PROFILE):
            stage = "scene_parser"
            initial_scene_facts = scene_parser.parse(sample)
            stage = "scene_facts_refiner"
            scene_facts = scene_refiner.refine(sample, initial_scene_facts, yolo_detections=None)
            affordance_context = build_affordance_context(scene_facts)

            stage = "agentic_tool_loop"
            loop = run_agentic_tool_loop(
                backend,
                scene_facts,
                affordance_context,
                image=sample.image_bytes,
                detector=detector,
                max_tool_rounds=max_tool_rounds,
                auto_crop_top_k=detect_attach_crops,
                detect_override=detect_override,
                # P10.S1: built here (not in main) because the verdict needs the
                # instruction-derived destination from the parsed scene facts.
                detect_postprocess=(
                    detect_postprocess_builder(sample, scene_facts)
                    if detect_postprocess_builder
                    else None
                ),
                truncate_to_first_tool_call=truncate_first_tool_call,
                force_phi_reschedule=force_phi_reschedule,
                force_phi_scope=force_phi_scope,
                force_phi_reject_feedback=force_phi_reject_feedback,
                disabled_tools=PHI_TOOLS if disable_phi else None,
                # P10.S2 trained arm: the checkpoint was fine-tuned under a
                # prompt with no Phi in it, so it must be evaluated under the
                # same one. --disable-phi stays on alongside this, which makes
                # the (expected ~0) dead-tool call rate the reportable evidence
                # that the runtime-gate arm's robustness confound is gone.
                # G6 passes the zero-shot supplement this way; the Phi-off arm
                # passes its own prompt. _validate_args refuses the combination
                # rather than letting one silently win over the other.
                system_prompt=(
                    system_prompt_override
                    or (build_phi_off_system_prompt() if phi_off_prompt else None)
                ),
                generation_kwargs={
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": repetition_penalty,
                    "stop": stop_strings,
                },
            )
            trace = loop_to_trace(loop)

            stage = "operation_validation"
            goal_active = _goal_active(loop.response)
            report = validate_operations(
                loop.response.operations,
                affordance_context,
                scene_facts,
                expected_non_empty_plan=bool(goal_active) if goal_active is not None else None,
                goal_active=goal_active,
            )
            trace["operation_validation"] = {
                "ok": report.ok,
                "messages": list(report.messages),
                "errors": [issue.message for issue in report.errors],
                "warnings": [issue.message for issue in report.warnings],
            }
            if not report.ok:
                raise RuntimeError("; ".join(report.messages) or "operation validation failed")

            stage = "graph_plan"
            graph_plan = convert_operations_to_graph_plan(
                loop.response.operations,
                affordance_context,
                scene_facts,
            )
            graph_plan.build_mode = "lnsdl_agentic_tool_loop"
            graph_plan.family_shortcut_used = False
            graph_plan.builder_trace = {
                "agentic_tool_loop": {
                    "stopped_reason": loop.stopped_reason,
                    "rounds": loop.rounds,
                    "tool_call_count": len(trace.get("tool_names", [])),
                    "tool_names": trace.get("tool_names", []),
                    "raw_op_count": len(loop.response.operations),
                },
                "operation_validation": trace["operation_validation"],
            }
            graph_plan = _assign_executors_to_graph(graph_plan, scene_facts)

            stage = "micro_planner"
            micro_steps = dispatch_graph(graph_plan)
            forced_record = loop.forced_phi_reschedule
            if isinstance(forced_record, dict) and forced_record.get("applied"):
                # A0k (facts:E24c): the forced arm's answer is Phi's schedule,
                # not a re-expansion of Phi's op-DAG (which loses the ee
                # serialization). Both are recorded so the two readings stay
                # auditable against each other row by row.
                phi_steps = forced_phi_micro_steps(forced_record)
                reexpanded = [_jsonable(step) for step in micro_steps]
                forced_record["reexpanded_micro_steps"] = reexpanded
                forced_record["reexpand_equals_phi_schedule"] = (
                    phi_steps is not None
                    and reexpanded == [_jsonable(step) for step in phi_steps]
                )
                if force_phi_final_plan == "phi_schedule" and phi_steps is not None:
                    micro_steps = phi_steps
                    forced_record["final_plan_source"] = "phi_schedule"
                else:
                    forced_record["final_plan_source"] = "reexpand_ops"
            stage = "dispatcher_emitter"
            dispatcher_output = VIKI_L2_PROFILE.dispatcher_emitter.emit(
                DispatcherPayload(
                    sample=sample,
                    initial_scene_facts=initial_scene_facts,
                    scene_facts=scene_facts,
                    graph_plan=graph_plan,
                    micro_steps=micro_steps,
                )
            )

            return (
                PipelineResult(
                    sample=sample,
                    initial_scene_facts=initial_scene_facts,
                    scene_facts=scene_facts,
                    yolo_detections=None,
                    graph_plan=graph_plan,
                    selected_robot_types=collect_selected_robot_types(graph_plan),
                    micro_steps=micro_steps,
                    dispatcher_output=dispatcher_output,
                ),
                trace,
            )
    except _RelayTransportError:
        # G6: a relay transport fault is NOT a model failure. Let it out
        # unwrapped so the caller can leave the sample un-run -- no artifact,
        # no 0 -- and pick it up on a later pass across a different blackout
        # window. Wrapping it (the default path below) writes an
        # `result_status: error` artifact that scores 0 and, worse, satisfies
        # the file-existence resume so the sample is never retried.
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgenticEvaluationError(
            stage=stage,
            sample=sample,
            original_exc=exc,
            initial_scene_facts=initial_scene_facts,
            scene_facts=scene_facts,
            trace=trace,
        ) from exc


def _json_fallback(obj: Any) -> str:
    """json.dumps safety net: render any stray image handle as a placeholder."""
    if type(obj).__module__.split(".", 1)[0] == "PIL":
        return "<image>"
    return str(obj)


def loop_to_trace(loop: ToolRuntimeLoopResult) -> dict[str, Any]:
    assistant_turns = [
        str(message.get("content") or "")
        for message in loop.messages
        if str(message.get("role")) == "assistant"
    ]
    initial_messages = loop.messages[:2]
    system_prompt = str(initial_messages[0].get("content") or "") if initial_messages else ""
    user_prompt = str(initial_messages[1].get("content") or "") if len(initial_messages) > 1 else ""
    tool_events = flatten_tool_events(loop)
    executed_tool_events = [
        event
        for event in tool_events
        if event.get("executed")
    ]
    tool_names = [str(event.get("name") or "tool") for event in executed_tool_events]
    parse_diagnostics = [
        diagnostic
        for event in loop.events
        for diagnostic in event.get("parsed", {}).get("parse_diagnostics", [])
        if isinstance(diagnostic, dict)
    ]
    return {
        "schema_version": 2,
        "stopped_reason": loop.stopped_reason,
        "rounds": loop.rounds,
        "input": {
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() if system_prompt else None,
            "system_prompt_chars": len(system_prompt),
            "system_prompt_excerpt": system_prompt[:1000],
            "user_prompt": user_prompt,
        },
        "message_roles": [str(message.get("role") or "") for message in loop.messages],
        "assistant_turn_count": len(assistant_turns),
        "assistant_turns": assistant_turns,
        "final_raw_text": loop.raw_text,
        "events": loop.events,
        "final_response": slm_response_to_dict(loop.response),
        "parse_diagnostics": parse_diagnostics,
        "emitted_tool_call_count": sum(
            len(event.get("tool_calls", []))
            for event in loop.events
            if isinstance(event, dict)
        ),
        "executed_tool_call_count": len(executed_tool_events),
        "tool_events": tool_events,
        "tool_calls": [event.get("call") for event in tool_events],
        "executed_tool_calls": [event.get("call") for event in executed_tool_events],
        "tool_names": tool_names,
        # Sanitize: raw tool results may carry out-of-band crop pixels
        # (`_crop_images`, P5.10.1); `_jsonable` drops `_`-keys + renders PIL as
        # "<image>" so the trace stays JSON-serializable.
        "tool_results": [_jsonable(result) for result in loop.tool_results],
        "final_plan_op_count": len(loop.response.operations),
        # P6.1 point 4-AR: harness-forced schedule pass on direct-emit plans.
        # None when the flag is off or the episode already called schedule.
        "forced_phi_reschedule": loop.forced_phi_reschedule,
    }


def flatten_tool_events(loop: ToolRuntimeLoopResult) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for turn in loop.events:
        if not isinstance(turn, dict):
            continue
        calls = turn.get("tool_calls", [])
        results = turn.get("tool_results", [])
        messages = turn.get("tool_messages", [])
        if not isinstance(calls, list):
            calls = []
        if not isinstance(results, list):
            results = []
        if not isinstance(messages, list):
            messages = []
        for index, call in enumerate(calls):
            result = results[index] if index < len(results) else None
            message = messages[index] if index < len(messages) else None
            name = None
            if isinstance(call, dict):
                name = call.get("name")
            events.append({
                "round": turn.get("round"),
                "index": index,
                "name": str(name or ""),
                "call": call,
                "result": result,
                "result_ok": result.get("ok") if isinstance(result, dict) else None,
                "message": message,
                "executed": result is not None,
            })
    return events


def build_agentic_analysis(artifact: dict[str, Any], trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = trace or {}
    final_response = trace.get("final_response") if isinstance(trace.get("final_response"), dict) else {}
    operation_validation = trace.get("operation_validation") if isinstance(trace.get("operation_validation"), dict) else {}
    validation = artifact.get("validation") if isinstance(artifact.get("validation"), dict) else {}
    error = artifact.get("error") if isinstance(artifact.get("error"), dict) else {}
    benchmark_scores = artifact.get("benchmark_scores") if isinstance(artifact.get("benchmark_scores"), dict) else {}
    plan = final_response.get("plan") if isinstance(final_response.get("plan"), list) else []
    sketch = final_response.get("plan_sketch") if isinstance(final_response.get("plan_sketch"), list) else []
    tool_events = trace.get("tool_events") if isinstance(trace.get("tool_events"), list) else []
    parse_diagnostics = trace.get("parse_diagnostics") if isinstance(trace.get("parse_diagnostics"), list) else []

    signals: list[str] = []
    if artifact.get("result_status") == "error":
        stage = error.get("pipeline_stage") or "unknown"
        signals.append(f"pipeline_error:{stage}")
    if trace.get("stopped_reason") and trace.get("stopped_reason") != "final_plan":
        signals.append(f"stopped:{trace.get('stopped_reason')}")
    if parse_diagnostics:
        signals.append("parse_diagnostics")
    if operation_validation and operation_validation.get("ok") is False:
        signals.append("operation_validation_failed")
    if validation and validation.get("success") is False:
        signals.append("benchmark_validation_failed")
    if not plan:
        signals.append("empty_final_plan")

    return {
        "result_status": artifact.get("result_status"),
        "pipeline_stage": error.get("pipeline_stage"),
        "stopped_reason": trace.get("stopped_reason"),
        "scores": {
            "viki_l1_robot_activation": benchmark_scores.get("viki_l1_robot_activation"),
            "viki_l2_plan_score": benchmark_scores.get("viki_l2_plan_score"),
            "locus_l2_acc": benchmark_scores.get("locus_l2_acc"),
            "benchmark_validator_success": validation.get("success"),
            "operation_validation_ok": operation_validation.get("ok"),
        },
        "blocks": {
            "has_goal": bool(final_response.get("goal")),
            "has_assign": bool(final_response.get("assign")),
            "has_plan_sketch": bool(sketch),
            "has_plan": bool(plan),
            "goal_active": (final_response.get("goal") or {}).get("active")
            if isinstance(final_response.get("goal"), dict)
            else None,
            "plan_op_count": len(plan),
            "plan_sketch_op_count": len(sketch),
            "plan_op_types": [op.get("op_type") for op in plan if isinstance(op, dict)],
            "plan_entities": sorted({
                str(op.get("entity"))
                for op in plan
                if isinstance(op, dict) and op.get("entity") is not None
            }),
        },
        "tools": {
            "emitted_count": trace.get("emitted_tool_call_count"),
            "executed_count": trace.get("executed_tool_call_count"),
            "names": trace.get("tool_names", []),
            "result_ok": [
                event.get("result_ok")
                for event in tool_events
                if isinstance(event, dict)
            ],
            # P6.1 point 4-AR compact read (full record stays in the trace).
            # None = flag off or the episode already executed schedule; the
            # 4-AR TC column = executed_count + (1 if applied else 0), computed
            # at report time so the base count stays comparable with point 4.
            "forced_phi_reschedule": (
                {
                    "outcome": trace["forced_phi_reschedule"].get("outcome"),
                    "source": trace["forced_phi_reschedule"].get("source"),
                    "latency_s": trace["forced_phi_reschedule"].get("latency_s"),
                }
                if isinstance(trace.get("forced_phi_reschedule"), dict)
                else None
            ),
        },
        "diagnostics": {
            "parse_diagnostic_count": len(parse_diagnostics),
            "operation_validation_messages": operation_validation.get("messages", []),
            "benchmark_error_desc": validation.get("error_desc"),
            "signals": signals,
        },
    }


def _goal_active(response: SLMResponse) -> list[str] | None:
    goal = response.goal
    if not isinstance(goal, dict):
        return None
    active = goal.get("active")
    if not isinstance(active, list):
        return None
    return [str(item) for item in active if str(item).strip()]


def _select_samples(args: argparse.Namespace) -> list[BenchmarkSample]:
    samples = load_benchmark_samples(level=args.level, split=args.split)
    task_names = parse_task_name_filters(args.task_name)
    if task_names:
        wanted = set(task_names)
        samples = [sample for sample in samples if sample.task_name in wanted]
    samples = samples[args.start_index :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if args.num_shards > 1:
        samples = samples[args.shard_index :: args.num_shards]
    return samples


def parse_task_name_filters(raw_values: list[str]) -> list[str]:
    task_names: list[str] = []
    for raw_value in raw_values:
        for item in raw_value.split(","):
            normalized = item.strip()
            if normalized:
                task_names.append(normalized)
    return sorted(set(task_names))


def summarize_task_names(samples: list[BenchmarkSample]) -> list[tuple[str, int]]:
    counts = Counter(sample.task_name or "<unknown>" for sample in samples)
    return sorted(counts.items())


def resolve_local_slm_weights_path(weights_path: str | Path) -> Path:
    path = Path(weights_path)
    if (path / "config.json").exists():
        return path
    final_path = path / "final"
    if (final_path / "config.json").exists():
        return final_path
    raise SystemExit(
        "Could not find checkpoint config.json under "
        f"{path} or {final_path}. Pass --slm-weights/--checkpoint-dir to a valid "
        "checkpoint run directory or final/ directory."
    )


def default_result_bucket(requested_weights: str | Path, resolved_weights: Path) -> str:
    """Derive a stable result bucket from the checkpoint path.

    Passing either ``.../qwen25-vl-7b-grok4-image-p4star`` or
    ``.../qwen25-vl-7b-grok4-image-p4star/final`` should produce the same
    bucket, avoiding stale names such as ``grok4_no_image`` when evaluating the
    new image-P4* checkpoints.
    """
    requested = Path(requested_weights)
    base = requested.name
    if base == "final":
        base = requested.parent.name
    if not base or base == ".":
        base = resolved_weights.parent.name if resolved_weights.name == "final" else resolved_weights.name
    return "agentic_sft_" + sanitize_bucket_name(base)


def sanitize_bucket_name(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "checkpoint"


def load_local_vlm_with_fallback(weights_path: str, device: str) -> None:
    import torch
    from nsdl.slm_backends import load_local_vlm

    if device != "auto":
        print(f"[L-NSDL] Loading local VLM on {device} from {weights_path} ...", flush=True)
        load_local_vlm(weights_path, device=device)
        print(f"[L-NSDL] Local VLM loaded on {device}.", flush=True)
        return

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise SystemExit("--device=auto requested but no CUDA device visible.")
    tried: list[str] = []
    last_exc: Exception | None = None
    for dev_idx in range(n_gpu):
        dev = f"cuda:{dev_idx}"
        tried.append(dev)
        try:
            print(f"[L-NSDL] Loading local VLM on {dev} from {weights_path} ...", flush=True)
            load_local_vlm(weights_path, device=dev)
            print(f"[L-NSDL] Local VLM loaded on {dev}.", flush=True)
            return
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
            msg = str(exc).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            print(f"[L-NSDL] OOM/CUDA-error on {dev}: {exc}; trying next GPU.", flush=True)
            last_exc = exc
            torch.cuda.empty_cache()
    raise SystemExit(f"Failed to load VLM on any of {tried}: {last_exc}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_tool_rounds < 0:
        raise SystemExit("--max-tool-rounds must be >= 0")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}); got {args.shard_index}")
    if args.force_phi_reject_feedback and not args.force_phi_reschedule:
        # (was parser.error(...) -- `parser` is not in this scope, so the guard
        # raised NameError instead of the intended message. Same effect class as
        # every other guard here: fail loudly, but now with the real reason.)
        raise SystemExit("--force-phi-reject-feedback requires --force-phi-reschedule")
    if args.force_phi_reschedule and (args.disable_phi or args.phi_off_prompt):
        # P6.1 point 4-AR needs a live schedule tool: under --disable-phi the
        # forced call would hit tool_unavailable and silently fall back to the
        # model plan on every sample -- an arm that looks like 4-AR but is
        # byte-identical to the Phi-off arm. Fail loudly instead.
        raise SystemExit(
            "--force-phi-reschedule cannot be combined with --disable-phi / "
            "--phi-off-prompt (the forced schedule call would be answered with "
            "tool_unavailable on every sample)."
        )
    if args.force_phi_scope != "direct_emit" and not args.force_phi_reschedule:
        # --force-phi-scope alone does nothing: the scope only selects which
        # samples the forcing covers, and the forcing itself is off. Silently
        # ignoring it would produce a bucket named/launched as the A0f arm that
        # is byte-identical to the baseline -- the exact "request != effect"
        # failure the D0 gate was built to catch. Fail loudly instead.
        raise SystemExit(
            "--force-phi-scope requires --force-phi-reschedule (the scope only "
            "selects which samples the forcing covers; on its own it is a no-op)."
        )
    if getattr(args, "api_model", None):
        if args.phi_off_prompt:
            # Both want to own the system prompt. Letting one silently win would
            # produce a bucket labelled as one arm and prompted as the other --
            # the exact "request != effect" failure class the D0 gate exists for.
            raise SystemExit(
                "--api-model cannot be combined with --phi-off-prompt (both "
                "replace the system prompt; pick one arm)."
            )
        if args.num_shards > 1:
            # Sharding normally splits GPUs; under an API backend it splits
            # relay quota instead, and quota is the thing that must not be
            # oversubscribed -- piling N processes onto one line is how a
            # blackout gets manufactured rather than survived. The second
            # relay account was bought for exactly this (~2x concurrency,
            # benchmark.md 7.3), so the rule is one shard per DISTINCT line:
            # each process must be given a single line, and there can be no
            # more shards than lines.
            lines = list(args.api_lines or [])
            if len(lines) != 1:
                raise SystemExit(
                    "--api-model with --num-shards > 1 requires --api-lines to "
                    "name exactly ONE line per shard (e.g. shard 0 -> "
                    "RELAY_A, shard 1 -> RELAY_A2); "
                    f"got {lines!r}. Sharding splits quota, not GPUs."
                )
    if getattr(args, "detect_verdict", False) and args.detector_backend in {"none", "off"}:
        # P10.S1: the verdict is derived from the detect evidence of THIS call.
        # With no perception there is no evidence to judge, so the arm would be
        # vacuous (and silently so). Fail loudly instead.
        raise SystemExit(
            "--detect-verdict requires a perception backend that returns evidence "
            f"(--detector-backend yolo or oracle); got {args.detector_backend!r}."
        )


if __name__ == "__main__":
    main()
