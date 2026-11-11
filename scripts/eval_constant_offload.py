#!/usr/bin/env python
"""P6.1b constant-offload control (docs/plan.md §P6.1b, blocking deliverable).

The project-internal equivalent of FLEET / OSDAG / IMR-LLM: the symbolic solver
is called *unconditionally* on a fixed schedule and the front-end is *not
trained*. It is the only empirical support for cut #3 ("learned on-demand gating
+ a trained front-end beats a fixed offload pipeline"), so it has to differ from
L-NSDL in exactly those two respects and in nothing else.

Held identical to the main eval (design discipline 13 -- same code path, not a
re-implementation):

  * ``nsdl.tool_runtime.execute_tool_call`` routes every tool, so Φ, the B1
    op_ids resolver, the payload renderers and the error taxonomy are shared;
  * the same scene parser / affordance context / ``validate_operations`` /
    ``convert_operations_to_graph_plan`` / ``build_result_artifact`` path, so
    ``results/.../*.json`` is diff-able against a learned-gating run sample by
    sample;
  * the same ``detect`` backends, including the eval-only ``oracle``;
  * first-judgment metrics come from ``scripts/summarize_eval_results.py``'s
    ``_first_goal_active`` / ``_active_entities`` -- the E1+ discipline functions,
    imported rather than re-derived, so the number means the same thing here.

Deliberately different:

  1. **Gating is hardwired.** Every sample runs ``detect`` (all instruction
     candidates) -> ``schedule`` -> ``verify``, in that order, with one
     deterministic ``schedule`` retry if ``verify`` rejects. The model never
     gates: any ``[TOOL_CALL]`` it emits is discarded and counted. Tool calls
     per sample are a constant and ``tool_call_rate == 1.0`` by construction --
     contrast L-NSDL, where the rate is a learned quantity.
  2. **The front-end is not trained.** Default ``--frontend base`` runs the
     untrained ``checkpoints/Qwen2.5-VL-7B-Instruct`` with a prompt-only spec of
     the op vocabulary and sketch format. ``--frontend format-sft`` is the
     documented fallback for when base format compliance is too low to compare
     against; it must be disclosed as "format-aligned, capability-untrained".
     Which one produced a run is recorded in every artifact under
     ``constant_offload.frontend``, and the compliance rate that justifies the
     choice is printed and stored in the report.

Products (metric parity with the main eval, feeding P10.1 main-table row 2 and
the P6.1 four-way comparison): per-sample artifacts plus
``results/lnsdl/baselines/constant_offload/<bucket>/{ID,OOD}_per_family.json``
carrying ``plan_acc``, first-judgment rate, OVER/UNDER decomposition and
tool-call counts per family.

Usage (7B base front-end, ID split):

    python scripts/eval_constant_offload.py --split test --frontend base

Nothing here trains, and ``--detector-backend`` defaults to the main eval's, so
this can run alongside a training job without contending for weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _phi_scheduler_mode() -> str:
    from nsdl.planning_tool import phi_scheduler_mode
    return phi_scheduler_mode()
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.benchmark_scoring import locus_m_depth_key  # noqa: E402
from nsdl.catalog import normalize_entity_name  # noqa: E402
from nsdl.dispatcher import dispatch_graph  # noqa: E402
from nsdl.evaluation import build_error_artifact, build_result_artifact  # noqa: E402
from nsdl.macro_graph import _assign_executors_to_graph, collect_selected_robot_types  # noqa: E402
from nsdl.profiles import VIKI_L2_PROFILE, use_benchmark_profile  # noqa: E402
from nsdl.prompt_adapter import build_sample_stem, load_benchmark_samples  # noqa: E402
from nsdl.scene_facts_refiner import build_scene_facts_refiner  # noqa: E402
from nsdl.scene_parser import build_scene_parser  # noqa: E402
from nsdl.schema import BenchmarkSample, DispatcherPayload, PipelineResult  # noqa: E402
from nsdl.slm_interface import (  # noqa: E402
    _required_caps_for_op,
    build_slm_system_prompt,
    convert_operations_to_graph_plan,
    parse_operation_steps,
    parse_slm_response,
    serialize_slm_input,
    validate_operations,
)
from nsdl.tool_runtime import (  # noqa: E402
    ToolRuntimeContext,
    _operation_payload,
    execute_tool_call,
    tool_result_message,
)
from nsdl.yolo_detector import (  # noqa: E402
    DEFAULT_DETECTOR_BACKEND,
    assert_detector_caliber_for_level,
    build_structured_detector,
    describe_detector_weights,
    detector_weights_for_level,
)

sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from eval_lnsdl_agentic_viki_l2 import (  # noqa: E402
    AgenticEvaluationError,
    DEFAULT_WEIGHTS,
    _family_key,
    _goal_active,
    _json_fallback,
    load_local_vlm_with_fallback,
    parse_task_name_filters,
    resolve_local_slm_weights_path,
)

# E1+ discipline: first-judgment is read from the trace's first parseable
# `[GOAL]`, never from `blocks.goal_active`. Importing the canonical functions
# instead of re-deriving them is what makes this control's facet-A number
# comparable to the main eval's (docs/debug_sft.md §1.9).
from summarize_eval_results import _first_goal_active, _active_entities  # noqa: E402

DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "lnsdl" / "baselines" / "constant_offload"

# Signatures of a CUDA context that will not recover within this process. Once
# one of these fires, every later sample raises the same thing, so the loop must
# abort rather than write a full split's worth of error artifacts (see the
# raise in the sample loop for the run this was measured on).
_CUDA_FATAL_MARKERS = (
    "unspecified launch failure",
    "CUDA error",
    "CUDA_ERROR",
    "device-side assert triggered",
    "an illegal memory access",
    "misaligned address",
    "CUDA out of memory",       # a partial allocation leaves the context unusable
    "cuBLAS",
    "cuDNN error",
)


def _is_cuda_context_fatal(exc: BaseException) -> bool:
    """True when ``exc`` (or anything it wraps) killed the CUDA context.

    Walks the ``__cause__``/``__context__`` chain because the pipeline wraps
    failures in ``AgenticEvaluationError``, so the CUDA text is usually one or
    two levels down rather than in ``str(exc)``.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        text = f"{type(node).__name__}: {node}"
        if any(marker in text for marker in _CUDA_FATAL_MARKERS):
            return True
        node = getattr(node, "original_exc", None) or \
            node.__cause__ or node.__context__
    return False


# The untrained control front-end. Not in eval_lnsdl's DEFAULT_WEIGHTS, which
# only maps SFT checkpoints -- the whole point here is the un-SFT'd base model.
BASE_FRONTEND_WEIGHTS = _REPO_ROOT / "checkpoints" / "Qwen2.5-VL-7B-Instruct"

# The fixed pipeline. Not a policy and not a learned order -- the point of the
# control is that this list is a constant of the method.
FIXED_TOOL_ORDER = ("detect", "schedule", "verify")

# Prompt-only op-vocabulary spec for the untrained front-end. It describes the
# sketch FORMAT and the op vocabulary and nothing else: no gating decision (the
# harness owns that), no state-judgment guidance (that is the capability under
# test), no worked example of when to skip.
BASE_FRONTEND_INSTRUCTIONS = """\
You are the intent front-end of a robot planning pipeline. A deterministic
solver does all scheduling, dependency ordering, robot assignment and
feasibility repair after you. Your only job is to emit the intent sketch.

Emit exactly two blocks and nothing else:

[GOAL]
{"targets": [<every object the instruction refers to>],
 "active": [<the subset that still needs to be acted on>]}

[PLAN_SKETCH]
[{"op_id": "o1", "op_type": <one of: acquire, local_acquire, deposit,
   carry_to, open, close, activate, apply_tool, process_at, inspect,
   load_carrier, push_carrier, unload_carrier, handoff>,
  "entity": <object name>, "dest": <destination name or null>}, ...]

Rules:
- Use one op per intended manipulation, in any order; the solver fixes order.
- Do not emit robot names, timings, dependencies or primitive motions.
- Do not emit any other block. Do not emit [TOOL_CALL] or [PLAN].
- Output the two blocks as raw text, no markdown fences, no commentary.
"""

# P6.1 point 1 (pure neural). Same scaffolding as the constant-offload control
# above -- same base weights, same system prompt, same two-block grammar -- with
# one factor flipped: there is no solver behind it, so the model must produce the
# executable plan itself, robot slots included. Holding the scaffolding fixed is
# what makes (pure neural vs constant offload) a single-factor contrast; running
# the two arms off different prompt stacks would confound "no tools" with "a
# different front-end harness".
#
# Robot slots: the constant-offload control forbids robot names because its
# solver owns assignment (and a model-named slot there would silently turn the
# control into "the model assigns"). Here the opposite holds -- with Phi gone,
# self-assignment IS the capability under test, so the prompt asks for it.
NO_TOOLS_FRONTEND_INSTRUCTIONS = """\
You are a robot planning front-end. There is no solver behind you: whatever you
emit is executed as-is. You must therefore produce a complete, ordered,
executable plan yourself -- including which robot performs each operation, any
preconditions (opening a closed container before taking from it), and any
transfers between robots.

Emit exactly two blocks and nothing else:

[GOAL]
{"targets": [<every object the instruction refers to>],
 "active": [<the subset that still needs to be acted on>]}

[PLAN]
[{"op_id": "o1", "op_type": <one of: acquire, local_acquire, deposit,
   carry_to, open, close, activate, apply_tool, process_at, inspect,
   load_carrier, push_carrier, unload_carrier, handoff>,
  "entity": <object name>, "dest": <destination name or null>,
  "robot": <the robot slot that performs it, e.g. "R1">,
  "deps": [<op_ids that must finish first>]}, ...]

Rules:
- Emit the ops in a valid execution order and fill every robot slot yourself.
- Insert the preconditions yourself; nothing is inserted for you.
- Do not emit any other block. Do not emit [TOOL_CALL] or [PLAN_SKETCH].
- Output the two blocks as raw text, no markdown fences, no commentary.
"""

# P6.1 point 3 (full-translation offload, ~= AHAT / LLM-as-formalizer). The
# model is demoted to an intent translator: instruction text -> intent ops, and
# nothing else. Both facets are offloaded -- facet B to Phi as in the constant
# control, and facet A to an externalized verdict derived from the same detect
# evidence (nsdl.verdict_perception, the P10.S1 arm's judge).
#
# The externalization is enforced STRUCTURALLY, not by instruction: this arm's
# model turn never receives the detect result, so it cannot judge state even if
# it wanted to. A prompt that merely *asks* the model not to judge would leave
# the arm measuring instruction-following, and a trained front-end would judge
# anyway -- exactly the failure mode that made asking-not-to-judge unusable.
# Consequently the prompt drops the `active` field: deciding it is no longer
# this model's job.
FULL_TRANSLATION_FRONTEND_INSTRUCTIONS = """\
You are the intent translator of a robot planning pipeline. A deterministic
backend does all perception, state judgement, scheduling, dependency ordering,
robot assignment and feasibility repair after you. Your only job is to restate
the instruction as intent operations.

Emit exactly two blocks and nothing else:

[GOAL]
{"targets": [<every object the instruction refers to>]}

[PLAN_SKETCH]
[{"op_id": "o1", "op_type": <one of: acquire, local_acquire, deposit,
   carry_to, open, close, activate, apply_tool, process_at, inspect,
   load_carrier, push_carrier, unload_carrier, handoff>,
  "entity": <object name>, "dest": <destination name or null>}, ...]

Rules:
- Emit one op for EVERY object the instruction refers to. Do not decide whether
  an object still needs to be acted on -- that decision is made for you.
- Use one op per intended manipulation, in any order; the solver fixes order.
- Do not emit robot names, timings, dependencies or primitive motions.
- Do not emit any other block. Do not emit [TOOL_CALL] or [PLAN].
- Output the two blocks as raw text, no markdown fences, no commentary.
"""

# P6.1 point 3b evidence source (plan 序 35 方案 A, trained state translator).
# NOT a frontend mode: this block is the system prompt of the SEPARATE state
# model that replaces the YOLO detector as the EVIDENCE source feeding
# nsdl.verdict_perception.derive_verdicts (--evidence-source trained_state).
# The sketch front-end and every other stage of the full-translation arm stay
# byte-identical. Must stay byte-identical to the corpus builder
# (scripts/build_locus_state_translator_corpus.py) which imports it.
STATE_TRANSLATOR_FRONTEND_INSTRUCTIONS = """\
You are the state translator of a robot planning pipeline. A deterministic
backend does all state judgement, scheduling, robot assignment and feasibility
repair after you. Your only job is to report where the instruction-relevant
objects currently are, as symbolic relations grounded in the image.

Emit exactly one block and nothing else:

[STATE]
{"entities": [<instruction-relevant object and fixture names present>],
 "relations": [<"<object> on|in <fixture>" for each placement you observe>]}

Rules:
- Report placements only for objects you can ground in the image; stay silent
  about placements you cannot observe or cannot express as on/in a fixture.
- Do not decide whether an object still needs to be acted on.
- Do not emit ops, robot names, or any other block. Do not emit [TOOL_CALL],
  [GOAL], [PLAN_SKETCH] or [PLAN].
- Output the block as raw text, no markdown fences, no commentary.
"""

FRONTEND_INSTRUCTIONS_BY_MODE = {
    "constant": BASE_FRONTEND_INSTRUCTIONS,
    "none": NO_TOOLS_FRONTEND_INSTRUCTIONS,
    "full-translation": FULL_TRANSLATION_FRONTEND_INSTRUCTIONS,
}


def assign_robots_by_capability(sketch: list[Any],
                                affordance_context: Any) -> dict[str, Any]:
    """Fill in each sketch op's robot slot by a fixed capability rule.

    The control's premise is that the solver owns robot assignment, so the
    front-end prompt forbids robot names and the model emits no ``[ASSIGN]``.
    ``schedule``'s own completion pass only resolves a slot when an ``assign``
    mapping is supplied (``planning_tool._op_robot`` returns None on an empty
    one), so without this the capability validator rejects every sample with
    ``missing_robot`` -- a harness gap, not a front-end failure, and it would
    have scored the control at plan_acc 0 for the wrong reason.

    The rule is deliberately fixed and GT-free: capabilities required by the
    op_type (``slm_interface._required_caps_for_op``, the same function the
    validator uses) filtered against each slot's declared capabilities, then
    round-robin over the eligible slots keyed by ENTITY. Keying on entity
    rather than op is what keeps custody coherent -- ``_complete_operations``
    chains ``carry_to``/``apply_tool``/``deposit`` back to an acquire by the
    same robot, so splitting one entity's ops across slots would make a
    schedulable sketch unschedulable. Returns the trace record of what it did.

    A robot the FRONT-END named is discarded, not honoured. The prompt forbids
    robot names, so a named slot is format non-compliance -- and measured on the
    untrained base model it is not an edge case but the norm (37/37 of the first
    ID samples), which means honouring it silently turned this control into
    "model assigns, solver schedules". That is the wrong control twice over: it
    hands the fixed-offload pipeline an extra model decision the premise says it
    does not have, and it fails samples on an invalid slot (``R1`` for an
    ``acquire`` on a fixed arm) that the capability rule would never have
    produced -- i.e. it understates the baseline. The names are counted under
    ``model_named`` so the non-compliance is disclosed rather than hidden.

    An op no single slot can satisfy is left unassigned ON PURPOSE and recorded
    under ``unassignable``. That is not a harness gap: a heterogeneous team
    (fixed arms with ``manipulation`` but no ``mobility``, plus a quadruped with
    ``mobility``/``push_transfer`` but no ``manipulation``) genuinely cannot
    ``acquire``, and the sketch has to say ``local_acquire`` + a carrier op
    instead. Choosing the op_type is the front-end's job and exactly the
    capability P6.1b measures, so the harness must not paper over it -- but it
    must also not let the resulting ``missing_robot`` be mistaken for an
    unfilled slot, hence the explicit record.
    """
    slots = [r.slot_id for r in affordance_context.robot_capabilities]
    caps = {r.slot_id: set(r.capabilities)
            for r in affordance_context.robot_capabilities}
    entity_slot: dict[str, str] = {}
    unassignable: list[dict[str, Any]] = []
    model_named: list[dict[str, Any]] = []
    cursor = 0
    for op in sketch:
        named = str(getattr(op, "robot", "") or "")
        if named:
            # the prompt forbids this; record and overwrite (see docstring)
            model_named.append({"op_id": str(getattr(op, "op_id", "") or ""),
                                "op_type": str(op.op_type), "robot": named})
            op.robot = None
        required = _required_caps_for_op(op)
        eligible = [s for s in slots if required <= caps.get(s, set())]
        if not eligible:
            # front-end chose an op_type this team cannot execute; report it
            # rather than inventing a slot the validator would reject anyway.
            unassignable.append({
                "op_id": str(getattr(op, "op_id", "") or ""),
                "op_type": str(op.op_type),
                "entity": str(op.entity or ""),
                "required_caps": sorted(required),
                "slot_caps": {s: sorted(caps.get(s, set())) for s in slots},
            })
            continue
        key = str(op.entity or f"__op:{op.op_id}")
        if key not in entity_slot:
            entity_slot[key] = eligible[cursor % len(eligible)]
            cursor += 1
        op.robot = entity_slot[key]
    # handoff / push_carrier additionally need a partner slot in params
    for op in sketch:
        if op.op_type not in {"handoff", "push_carrier"}:
            continue
        params = dict(op.params or {})
        if params.get("target_robot") or params.get("target_slot"):
            continue
        partner = next((s for s in slots if s != op.robot), None)
        if partner:
            params["target_robot"] = partner
            op.params = params
    return {"rule": "capability_filter_then_entity_round_robin",
            "slots": slots, "entity_slot": entity_slot,
            "unassignable": unassignable,
            # ops whose robot slot the front-end filled despite the prompt
            # forbidding it; overwritten by the rule above, counted here.
            "model_named": model_named}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P6.1b constant-offload control (fixed gating, untrained front-end)")
    parser.add_argument("--split", default="test",
                        help="'test' = ID, 'val' = OOD (docs/test_analysis.md)")
    parser.add_argument("--level", default="VIKI-L2")
    parser.add_argument(
        "--offload-mode", choices=["constant", "none", "full-translation"],
        default="constant",
        help="Which point of the P6.1 offload spectrum to run. 'constant' = "
             "point 2 (facet B offloaded, fixed gating, untrained front-end). "
             "'none' = point 1 (pure neural: no solver at all). "
             "'full-translation' = point 3 (facets A *and* B offloaded: the "
             "model only translates the instruction, the harness derives the "
             "state verdict from the same detect evidence). Point 4 (L-NSDL) "
             "is the main eval, not this script.")
    parser.add_argument(
        "--frontend", choices=["base", "format-sft", "trained"], default="base",
        help="'base' = untrained Qwen2.5-VL-7B-Instruct + prompt-only (the "
             "specified control). 'format-sft' = documented fallback when base "
             "format compliance is too low to compare; disclosed as "
             "format-aligned, capability-untrained. 'trained' = the v3.3 SFT "
             "front-end, used by --offload-mode full-translation so that the "
             "point-3/point-4 gap isolates the division of labour rather than "
             "front-end capability.")
    parser.add_argument(
        "--frontend-weights", default=None,
        help="Front-end weights. Defaults to the base 7B instruct model for "
             "--frontend base; --frontend format-sft requires this explicitly.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--result-bucket", default=None)
    parser.add_argument("--task-name", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--stop-string", action="append", default=None,
                        dest="stop_strings")
    parser.add_argument("--report-only", action="store_true",
                        help="Re-aggregate existing artifacts without loading a "
                             "model or running any sample.")
    parser.add_argument("--scene-parser-backend", choices=["rule"], default="rule")
    parser.add_argument("--scene-facts-refiner-backend",
                        choices=["none", "llm", "auto"], default="none")
    parser.add_argument(
        "--detector-backend",
        choices=["none", DEFAULT_DETECTOR_BACKEND, "off", "oracle"],
        default=DEFAULT_DETECTOR_BACKEND,
        help="Same perception channel as the main eval -- equal tool access is "
             "what makes the comparison attributable to 'trained + on-demand' "
             "(§7-8 eval fairness). 'oracle' is eval-only.")
    parser.add_argument("--detector-weights", default=None,
                        help="Default: the caliber of --level (VIKI-L2 20-class / LOCUS 34-class, facts:D3a).")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.70)
    parser.add_argument("--detector-max-det", type=int, default=64)
    parser.add_argument(
        "--evidence-source", choices=["detect", "trained_state"],
        default="detect",
        help="full-translation only. 'detect' (default, byte-identical to "
             "every bucket before 2026-09-13): the state verdict is derived "
             "from the constant detect result. 'trained_state': the verdict "
             "is derived from a trained state-translator model's [STATE] "
             "relations (plan \u5e8f 35 \u65b9\u6848 A), precomputed by "
             "scripts/run_state_translator_frontend.py into "
             "--state-evidence-dir. The detect result is still recorded as a "
             "counterfactual but no longer feeds the verdict.")
    parser.add_argument(
        "--state-evidence-dir", default=None,
        help="Root dir of precomputed state-translator evidence "
             "(<dir>/<split>/<stem>.json). Required with "
             "--evidence-source trained_state; every selected sample must "
             "have a file (missing evidence hard-fails, never scores as "
             "empty: docs memory 'missing-column-is-not-zero').")
    return parser


# --------------------------------------------------------------------------
# the fixed pipeline
# --------------------------------------------------------------------------

_CANDIDATE_FIELDS = (
    "target_items", "mentioned_objects", "secondary_objects",
    "destination_candidates", "tool_items", "activation_targets",
    "inspection_targets", "source_locations", "fallback_locations",
    "targets_in_cabinet",
)


def instruction_candidates(scene_facts: Any) -> list[str]:
    """Every object the instruction names -- the constant ``detect`` query.

    A learned front-end chooses what to look at; this control looks at
    everything the instruction mentions, every time. Derived from the parsed
    instruction only (never from GT), so the query carries no answer.
    """
    names: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in names:
            names.append(text)

    for field_name in _CANDIDATE_FIELDS:
        for value in (getattr(scene_facts, field_name, None) or []):
            add(value)
    add(getattr(scene_facts, "primary_object", None))
    add(getattr(scene_facts, "destination", None))
    return names


def _op_ids(sketch: list[Any]) -> list[str]:
    return [str(op.op_id) for op in sketch if getattr(op, "op_id", None)]


def _solver_args(sketch: list[Any]) -> dict[str, Any]:
    """Args for ``schedule``/``verify`` under the shared B1 op_ids contract.

    Referencing ids (resolved against ``plan_sketch`` by the runtime) rather
    than inlining an ops array keeps this control on the exact resolver path the
    trained model's calls take. A sketch whose ops carry no ``op_id`` falls back
    to an explicit ops array, which the same runtime accepts.
    """
    ids = _op_ids(sketch)
    if ids:
        return {"op_ids": ids}
    return {"ops": [_operation_payload(op) for op in sketch]}


def run_constant_offload_sample(
    sample: BenchmarkSample,
    *,
    backend: Any,
    detector: Any,
    scene_parser: Any,
    scene_refiner: Any,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float | None,
    stop_strings: list[str] | None,
    detect_override: Any | None = None,
    frontend_instructions: str = BASE_FRONTEND_INSTRUCTIONS,
    offload_mode: str = "constant",
    state_verdict: bool = False,
    state_evidence: dict[str, Any] | None = None,
) -> tuple[PipelineResult, dict[str, Any]]:
    """One sample through the fixed detect -> schedule -> verify pipeline.

    ``state_verdict`` promotes the run from point 2 (constant offload) to point
    3 (full-translation offload): facet A leaves the model too. Two things
    change together and they are two halves of one mechanism -- the detect
    result is withheld from the model turn, so it *cannot* judge state, and the
    harness instead derives ``already_at_destination`` from that same evidence
    and drops the ops it settles. Withholding without deriving would just be a
    blinded model; deriving without withholding would leave two judges and make
    the arm measure instruction-following.
    """
    initial_scene_facts = None
    scene_facts = None
    trace: dict[str, Any] = {"offload_mode": offload_mode,
                             "state_verdict": bool(state_verdict),
                             "fixed_tool_order": list(FIXED_TOOL_ORDER)}
    stage = "profile"
    try:
        with use_benchmark_profile(VIKI_L2_PROFILE):
            stage = "scene_parser"
            initial_scene_facts = scene_parser.parse(sample)
            stage = "scene_facts_refiner"
            scene_facts = scene_refiner.refine(
                sample, initial_scene_facts, yolo_detections=None)
            affordance_context = build_affordance_context(scene_facts)

            detect_postprocess = None
            if state_verdict:
                # Import here, not at module scope: nsdl.verdict_perception
                # refuses to import without NSDL_VERDICT_PERCEPTION=eval-only,
                # and the other two offload modes must stay runnable without it.
                from nsdl.verdict_perception import make_verdict_postprocess
                detect_postprocess = make_verdict_postprocess(
                    scene_facts, source="p6.1-full-translation")

            ctx = ToolRuntimeContext(
                scene_facts=scene_facts,
                affordance_context=affordance_context,
                image=sample.image_bytes,
                detector=detector,
                detect_override=detect_override,
                detect_postprocess=detect_postprocess,
            )

            # ---- stage 1: unconditional detect over all named candidates ---
            stage = "constant_detect"
            candidates = instruction_candidates(scene_facts)
            detect_args = {"query": {"names": candidates}}
            detect_result = execute_tool_call(
                {"name": "detect", "args": detect_args}, context=ctx)
            executed: list[dict[str, Any]] = [
                {"name": "detect", "args": detect_args,
                 "result_ok": bool(detect_result.get("ok", True)),
                 "result": detect_result}]

            # ---- the front-end's only turn: an intent sketch ---------------
            stage = "frontend_sketch"
            system_prompt = build_slm_system_prompt() + "\n\n" + frontend_instructions
            user_prompt = serialize_slm_input(scene_facts, affordance_context)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if not state_verdict:
                messages.append(tool_result_message("detect", detect_result))
            gen_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "stop": stop_strings,
            }
            if isinstance(sample.image_bytes, bytes):
                gen_kwargs["image_bytes"] = sample.image_bytes
            raw_text = backend.generate(messages=messages, **gen_kwargs)
            response = parse_slm_response(raw_text)
            # The harness owns gating: a tool call from the model is not run.
            # Counting them is how we evidence that the front-end had no say.
            ignored_tool_calls = list(response.tool_calls)
            sketch = list(response.plan_sketch) or list(response.operations)

            # ---- point 3 only: unblock the trained front-end's protocol -----
            # One turn is enough for the untrained base (it does not know to
            # call detect, so it sketches immediately: compliance .989). The
            # TRAINED v3.3 front-end's learned protocol is think -> call detect
            # -> judge, so it spends its only turn asking for evidence this arm
            # withholds by design, and 83/180 test_id rows never reached a
            # sketch at all (compliance .539). That measures prompt-transfer
            # failure, not facet-A externalization -- a confound on top of the
            # two this arm already carries.
            #
            # So answer the call, in the tool channel it was trained on, with a
            # notice and NO evidence. The structural guarantee is intact -- the
            # model still never sees a detect result, so it still cannot judge
            # state -- while the stall is removed. Bounded to one extra turn:
            # a model that asks twice is making a decision, and this arm's
            # premise is that it has none to make.
            translator_retry = False
            if state_verdict and not sketch and ignored_tool_calls:
                translator_retry = True
                stage = "translator_retry"
                messages = messages + [
                    {"role": "assistant", "content": raw_text},
                    tool_result_message("detect", {
                        "status": "externalized",
                        "note": ("Perception and state judgement are handled by the "
                                 "backend in this configuration. No observations "
                                 "will be returned to you. Emit [GOAL] and "
                                 "[PLAN_SKETCH] now, covering every object the "
                                 "instruction names."),
                    }),
                ]
                raw_text_retry = backend.generate(messages=messages, **gen_kwargs)
                response = parse_slm_response(raw_text_retry)
                ignored_tool_calls += list(response.tool_calls)
                sketch = list(response.plan_sketch) or list(response.operations)
                raw_text = raw_text + "\n\n<<TRANSLATOR_RETRY>>\n\n" + raw_text_retry

            # ---- point 3 only: the externalized state verdict --------------
            # The model translated every named target; the harness now decides
            # which of them still need acting on, from the detect evidence the
            # model never saw. Items the evidence does not settle are KEPT
            # (derive_verdicts omits rather than guesses), so absence of
            # evidence never silently empties the plan.
            verdict_map: dict[str, bool] = {}
            verdict_dropped: list[str] = []
            detect_verdict_map = dict(detect_result.get("already_at_destination") or {})
            if state_verdict:
                stage = "state_verdict"
                if state_evidence is not None:
                    # 序 35 方案 A: the verdict evidence is the trained state
                    # translator's relations, NOT the detect result. Same rule
                    # judge (derive_verdicts), different evidence source; the
                    # detect verdicts stay in the trace as the counterfactual.
                    from nsdl.perc_tool import DetectResult
                    from nsdl.verdict_perception import derive_verdicts
                    state_result = DetectResult(
                        entities=[str(e) for e in (state_evidence.get("entities") or [])],
                        relations=[str(r) for r in (state_evidence.get("relations") or [])],
                    )
                    verdict_map = derive_verdicts(state_result, scene_facts)
                else:
                    verdict_map = dict(detect_verdict_map)
                if verdict_map:
                    kept = []
                    for op in sketch:
                        entity = getattr(op, "entity", None) or ""
                        name = normalize_entity_name(entity) or str(entity).strip().lower()
                        if verdict_map.get(name) is True:
                            verdict_dropped.append(
                                f"{getattr(op, 'op_id', '?')}:{name}")
                        else:
                            kept.append(op)
                    sketch = kept

            # The solver owns robot assignment in this control (the prompt
            # forbids the front-end from naming robots), so the harness fills
            # the slots by a fixed capability rule before calling schedule.
            stage = "constant_assign"
            assignment = assign_robots_by_capability(sketch, affordance_context)

            # ---- stage 2/3: schedule, then verify, with one repair retry ----
            solver_args = _solver_args(sketch)
            stage = "constant_schedule"
            schedule_result = execute_tool_call(
                {"name": "schedule", "args": solver_args},
                context=ctx, plan_sketch=sketch)
            executed.append({"name": "schedule", "args": solver_args,
                             "result_ok": bool(schedule_result.get("ok")),
                             "result": schedule_result})

            stage = "constant_verify"
            verify_result = execute_tool_call(
                {"name": "verify", "args": solver_args},
                context=ctx, plan_sketch=sketch)
            executed.append({"name": "verify", "args": solver_args,
                             "result_ok": bool(verify_result.get("ok")),
                             "result": verify_result})

            retried = False
            if not verify_result.get("ok", False):
                # The fixed pipeline's one repair affordance: re-run schedule.
                # Deterministic, not a second model turn -- the control must not
                # gain an extra learned decision here.
                retried = True
                stage = "constant_schedule_retry"
                schedule_result = execute_tool_call(
                    {"name": "schedule", "args": solver_args},
                    context=ctx, plan_sketch=sketch)
                executed.append({"name": "schedule",
                                 "args": dict(solver_args, retry=True),
                                 "result_ok": bool(schedule_result.get("ok")),
                                 "result": schedule_result})

            # The solver's completed ops ARE the plan: in this control the model
            # never emits [PLAN], the pipeline produces it.
            stage = "plan_from_solver"
            ops = _ops_from_schedule_payload(schedule_result, sketch)

            stage = "operation_validation"
            goal_active = _goal_active(response)
            if state_verdict:
                # Point 3's active set is the harness's, not the model's: the
                # prompt drops `active` and the model never saw the evidence, so
                # `_goal_active` is None here by construction. What survived the
                # verdict filter IS this arm's facet-A decision, and it is what
                # the E1+ first-judgment metric must be read from -- scoring the
                # model's absent active set would report 0 for a judgement the
                # arm structurally forbids it from making.
                goal_active = sorted({
                    normalize_entity_name(getattr(op, "entity", "") or "")
                    or str(getattr(op, "entity", "") or "").strip()
                    for op in sketch
                } - {""})
            report = validate_operations(
                ops,
                affordance_context,
                scene_facts,
                expected_non_empty_plan=(
                    bool(goal_active) if goal_active is not None else None),
                goal_active=goal_active,
            )
            trace.update({
                # `assistant_turns` is the key `_first_goal_active` reads; this
                # control has exactly one model turn, so it holds one entry.
                "assistant_turns": [raw_text],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "frontend_raw_text": raw_text,
                "frontend_parsed": {
                    "goal": response.goal,
                    "plan_sketch": [_operation_payload(op) for op in sketch],
                    "has_goal": bool(response.goal),
                    "has_plan_sketch": bool(response.plan_sketch),
                },
                "parse_diagnostics": list(response.parse_diagnostics),
                "harness_robot_assignment": assignment,
                # Point 3 only. Kept in its own key rather than spliced into
                # `assistant_turns`: that field is the verbatim model record and
                # a harness-authored [GOAL] inside it would be indistinguishable
                # from something the model said.
                "harness_goal_active": (
                    list(goal_active) if state_verdict else None),
                "state_verdict_map": verdict_map,
                "verdict_dropped_ops": verdict_dropped,
                "verdict_source": (
                    "p6.1-full-translation-trained-state"
                    if (state_verdict and state_evidence is not None)
                    else ("p6.1-full-translation" if state_verdict else None)),
                # counterfactual only when the verdict came from trained state
                "detect_verdict_map": (
                    detect_verdict_map if state_verdict else None),
                "state_evidence": (
                    {"entities": list(state_evidence.get("entities") or []),
                     "relations": list(state_evidence.get("relations") or []),
                     "parse_ok": bool(state_evidence.get("parse_ok", True))}
                    if state_evidence is not None else None),
                "detect_candidates": candidates,
                "tool_events": executed,
                "tool_names": [event["name"] for event in executed],
                "emitted_tool_call_count": len(ignored_tool_calls),
                "executed_tool_call_count": len(executed),
                "ignored_model_tool_calls": ignored_tool_calls,
                # Point 3 only. True means the trained front-end spent its first
                # turn asking for perception this arm withholds by design, and
                # the harness answered with an externalization notice carrying
                # no evidence. It is a disclosure quantity, not a fix to hide:
                # the rate says how far the arm's prompt sits outside v3.3's
                # trained protocol.
                "translator_retry": translator_retry,
                "schedule_retried": retried,
                "verify_ok": bool(verify_result.get("ok", False)),
                "final_response": {
                    "goal": response.goal,
                    "plan": [_operation_payload(op) for op in ops],
                    "plan_sketch": [_operation_payload(op) for op in sketch],
                },
                "operation_validation": {
                    "ok": report.ok,
                    "messages": list(report.messages),
                    "errors": [issue.message for issue in report.errors],
                    "warnings": [issue.message for issue in report.warnings],
                },
            })
            if not report.ok:
                raise RuntimeError(
                    "; ".join(report.messages) or "operation validation failed")

            stage = "graph_plan"
            graph_plan = convert_operations_to_graph_plan(
                ops, affordance_context, scene_facts)
            graph_plan.build_mode = (
                "full_translation_offload_pipeline" if state_verdict
                else "constant_offload_pipeline")
            graph_plan.family_shortcut_used = False
            graph_plan.builder_trace = {
                "constant_offload": {
                    "offload_mode": offload_mode,
                    **describe_detector_weights(
                        getattr(detector, "weights_path", None) if detector is not None else None
                    ),
                    "fixed_tool_order": list(FIXED_TOOL_ORDER),
                    "schedule_retried": retried,
                    "tool_call_count": len(executed),
                    "ignored_model_tool_calls": len(ignored_tool_calls),
                    "raw_op_count": len(ops),
                    "verdict_dropped_op_count": len(verdict_dropped),
                },
                "operation_validation": trace["operation_validation"],
            }
            graph_plan = _assign_executors_to_graph(graph_plan, scene_facts)

            stage = "micro_planner"
            micro_steps = dispatch_graph(graph_plan)
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
    except Exception as exc:  # noqa: BLE001
        raise AgenticEvaluationError(
            stage=stage,
            sample=sample,
            original_exc=exc,
            initial_scene_facts=initial_scene_facts,
            scene_facts=scene_facts,
            trace=trace,
        ) from exc


def run_pure_neural_sample(
    sample: BenchmarkSample,
    *,
    backend: Any,
    scene_parser: Any,
    scene_refiner: Any,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float | None,
    stop_strings: list[str] | None,
    frontend_instructions: str = NO_TOOLS_FRONTEND_INSTRUCTIONS,
) -> tuple[PipelineResult, dict[str, Any]]:
    """P6.1 point 1: the same front-end with no tools behind it at all.

    No ``detect``, no ``schedule``, no ``verify`` -- the model's own ``[PLAN]``
    is the plan. Everything downstream of the plan (graph conversion, executor
    assignment, micro-planning, the dispatcher emitter) still runs: that is the
    shared *execution* layer, not Phi, and the P10.S2 ``phi_off`` arms go through
    exactly the same path. Ablating it too would mean the arm produces no
    scoreable output at all rather than a worse one.
    """
    initial_scene_facts = None
    scene_facts = None
    trace: dict[str, Any] = {"offload_mode": "none",
                             "fixed_tool_order": []}
    stage = "profile"
    try:
        with use_benchmark_profile(VIKI_L2_PROFILE):
            stage = "scene_parser"
            initial_scene_facts = scene_parser.parse(sample)
            stage = "scene_facts_refiner"
            scene_facts = scene_refiner.refine(
                sample, initial_scene_facts, yolo_detections=None)
            affordance_context = build_affordance_context(scene_facts)

            stage = "frontend_plan"
            system_prompt = build_slm_system_prompt() + "\n\n" + frontend_instructions
            user_prompt = serialize_slm_input(scene_facts, affordance_context)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            gen_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "stop": stop_strings,
            }
            if isinstance(sample.image_bytes, bytes):
                gen_kwargs["image_bytes"] = sample.image_bytes
            raw_text = backend.generate(messages=messages, **gen_kwargs)
            response = parse_slm_response(raw_text)
            # A tool call here is a format violation, not a decision: no tool
            # exists to answer it. Counted, never executed.
            ignored_tool_calls = list(response.tool_calls)
            ops = list(response.operations) or list(response.plan_sketch)

            stage = "operation_validation"
            goal_active = _goal_active(response)
            report = validate_operations(
                ops,
                affordance_context,
                scene_facts,
                expected_non_empty_plan=(
                    bool(goal_active) if goal_active is not None else None),
                goal_active=goal_active,
            )
            trace.update({
                "assistant_turns": [raw_text],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "frontend_raw_text": raw_text,
                "frontend_parsed": {
                    "goal": response.goal,
                    "plan_sketch": [_operation_payload(op)
                                    for op in response.plan_sketch],
                    "has_goal": bool(response.goal),
                    # `format_compliance` reads this key. On this arm the
                    # required block is [PLAN], not [PLAN_SKETCH] -- the prompt
                    # asks for an executable plan because there is no solver to
                    # expand a sketch. Reporting bool(plan_sketch) would score
                    # the arm 0% compliant for obeying its own prompt.
                    "has_plan_sketch": bool(ops),
                    "emitted_block": (
                        "PLAN" if response.operations
                        else ("PLAN_SKETCH" if response.plan_sketch else None)),
                },
                "parse_diagnostics": list(response.parse_diagnostics),
                # P6.1b's harness discards and rewrites `op.robot`; this arm
                # must NOT, or "pure neural" would quietly inherit a symbolic
                # assignment step (the exact contamination `model_named_robot`
                # was introduced to catch). Kept in the same key so the two arms
                # are read off one field; `harness_overwrote` is the difference.
                "harness_robot_assignment": {
                    "harness_overwrote": False,
                    "model_named": any(
                        str(getattr(op, "robot", "") or "").strip() for op in ops),
                    "model_named_op_count": sum(
                        1 for op in ops
                        if str(getattr(op, "robot", "") or "").strip()),
                    "unassignable": [],
                },
                "tool_events": [],
                "tool_names": [],
                "emitted_tool_call_count": len(ignored_tool_calls),
                "executed_tool_call_count": 0,
                "ignored_model_tool_calls": ignored_tool_calls,
                "schedule_retried": False,
                "verify_ok": None,
                "final_response": {
                    "goal": response.goal,
                    "plan": [_operation_payload(op) for op in ops],
                    "plan_sketch": [_operation_payload(op)
                                    for op in response.plan_sketch],
                },
                "operation_validation": {
                    "ok": report.ok,
                    "messages": list(report.messages),
                    "errors": [issue.message for issue in report.errors],
                    "warnings": [issue.message for issue in report.warnings],
                },
            })
            if not report.ok:
                raise RuntimeError(
                    "; ".join(report.messages) or "operation validation failed")

            stage = "graph_plan"
            graph_plan = convert_operations_to_graph_plan(
                ops, affordance_context, scene_facts)
            graph_plan.build_mode = "pure_neural_no_tools"
            graph_plan.family_shortcut_used = False
            graph_plan.builder_trace = {
                "pure_neural": {
                    "tool_call_count": 0,
                    "ignored_model_tool_calls": len(ignored_tool_calls),
                    "raw_op_count": len(ops),
                },
                "operation_validation": trace["operation_validation"],
            }
            graph_plan = _assign_executors_to_graph(graph_plan, scene_facts)

            stage = "micro_planner"
            micro_steps = dispatch_graph(graph_plan)
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
    except Exception as exc:  # noqa: BLE001
        raise AgenticEvaluationError(
            stage=stage,
            sample=sample,
            original_exc=exc,
            initial_scene_facts=initial_scene_facts,
            scene_facts=scene_facts,
            trace=trace,
        ) from exc


def _ops_from_schedule_payload(payload: dict[str, Any], fallback: list[Any]) -> list[Any]:
    """Operations the solver completed, falling back to the raw sketch.

    ``schedule`` returns its completed op list under ``ops``. When it fails, the
    control has nothing better than the model's own sketch -- the honest
    behaviour for a pipeline whose front-end cannot be asked again.
    """
    raw_ops = payload.get("ops")
    if isinstance(raw_ops, list) and raw_ops:
        try:
            return parse_operation_steps(raw_ops)
        except Exception:  # noqa: BLE001 - fall back to the sketch
            pass
    return list(fallback)


# --------------------------------------------------------------------------
# per-family reporting with metric parity
# --------------------------------------------------------------------------

def _goal_decomposition(first_active: set[str] | None,
                        gt_grasp: set[str] | None) -> str | None:
    """correct / OVER / UNDER / mixed, or None when it cannot be computed."""
    if first_active is None or gt_grasp is None:
        return None
    if first_active == gt_grasp:
        return "correct"
    if first_active - gt_grasp and not gt_grasp - first_active:
        return "OVER"
    if gt_grasp - first_active and not first_active - gt_grasp:
        return "UNDER"
    return "mixed"


def _new_bucket() -> dict[str, Any]:
    return {
        "n": 0, "errors": 0,
        "plan_acc_sum": 0.0, "l1_sum": 0.0, "l1_n": 0,
        "tool_calls_sum": 0, "ignored_model_tool_calls_sum": 0,
        "schedule_retried": 0, "verify_ok": 0,
        "format_compliant": 0, "capability_unassignable": 0,
        "model_named_robot": 0,
        "facet_a_n": 0, "first_goal_correct": 0, "unparsed_first_goal": 0,
        "goal_decomposition": Counter(),
        # Point 3 only. The externalized judge omits rather than guesses, so it
        # settles only a fraction of samples; without this the arm's facet-A
        # number looks like a judgement quality reading when it is largely a
        # coverage reading.
        "verdict_settled": 0, "verdict_dropped_ops_sum": 0,
        # Point 3 only. How often the trained front-end stalled on its
        # first turn asking for perception this arm withholds, and the
        # harness had to answer with an evidence-free notice. Disclosed:
        # a high rate means the arm is measuring prompt transfer as much
        # as it is measuring facet-A externalization.
        "translator_retry": 0,
    }


def _report_label(level: str, split: str) -> str:
    """Report filename stem: ID/OOD on VIKI-L2, the split name on LOCUS.

    VIKI-L2 has two eval splits so ID/OOD is faithful. LOCUS has four and three
    of them ('ood_s'/'ood_m'/'ood_x') would collapse onto one filename, each run
    overwriting the last -- and the survivor would be read as if it covered the
    grid. Keying by split makes an overwrite impossible.
    """
    if level == "LOCUS":
        return split
    return "ID" if split == "test" else "OOD"


def report_per_family(output_dir: Path, *, split: str,
                      metadata: dict[str, Any],
                      level: str = "VIKI-L2") -> dict[str, Any]:
    """Aggregate to the P10.1/P6.1 shape and build the JSON product.

    On ``level="LOCUS"`` the grouping switches to the M depth and the score
    column to ``locus_l2_acc`` -- identical to the L-NSDL arm's treatment, and
    via the same shared helpers, so the P10.S2 curve compares two arms measured
    with one ruler. Reading LOCUS off ``viki_l2_plan_score`` would zero every
    null-plan row (73/560) in *both* arms and shift the crossing point.
    """
    is_locus = level == "LOCUS"
    score_key = "locus_l2_acc" if is_locus else "viki_l2_plan_score"
    per_family: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name.startswith("_") or path.name.endswith("_per_family.json"):
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trace = (artifact.get("constant_offload") or {}).get("trace") or {}
        scores = artifact.get("benchmark_scores") or {}
        family = (
            locus_m_depth_key(artifact.get("ground_truth"))
            if is_locus
            else _family_key(path.stem)
        )
        bucket = per_family.setdefault(family, _new_bucket())

        bucket["n"] += 1
        bucket["plan_acc_sum"] += float(scores.get(score_key) or 0.0)
        if isinstance(scores.get("viki_l1_robot_activation"), (int, float)):
            bucket["l1_sum"] += float(scores["viki_l1_robot_activation"])
            bucket["l1_n"] += 1
        bucket["tool_calls_sum"] += int(trace.get("executed_tool_call_count") or 0)
        bucket["ignored_model_tool_calls_sum"] += int(
            trace.get("emitted_tool_call_count") or 0)
        bucket["translator_retry"] += int(bool(trace.get("translator_retry")))
        bucket["schedule_retried"] += int(bool(trace.get("schedule_retried")))
        bucket["verify_ok"] += int(bool(trace.get("verify_ok")))
        if artifact.get("result_status") == "error":
            bucket["errors"] += 1
        if (trace.get("frontend_parsed") or {}).get("has_plan_sketch"):
            bucket["format_compliant"] += 1
        # Sketches naming an op_type no single robot can execute. A real
        # front-end failure (see assign_robots_by_capability), counted so the
        # resulting `missing_robot` errors are not read as a harness gap.
        if (trace.get("harness_robot_assignment") or {}).get("unassignable"):
            bucket["capability_unassignable"] += 1
        # Prompt says "do not emit robot names"; this is how often it does
        # anyway. Part of the front-end's disclosed format compliance, not a
        # harness quantity -- the harness overwrites the name either way.
        if (trace.get("harness_robot_assignment") or {}).get("model_named"):
            bucket["model_named_robot"] += 1

        if trace.get("state_verdict"):
            bucket["verdict_settled"] += int(bool(trace.get("state_verdict_map")))
            bucket["verdict_dropped_ops_sum"] += len(
                trace.get("verdict_dropped_ops") or [])

        # E1+ facet-A: first parseable [GOAL] vs the GT's grasped entity set.
        ground_truth = artifact.get("ground_truth")
        gt_steps = ground_truth.get("time_steps") if isinstance(ground_truth, dict) else None
        if trace and isinstance(gt_steps, list):
            bucket["facet_a_n"] += 1
            gt_grasp = _active_entities(gt_steps)
            if trace.get("state_verdict"):
                # Full-translation arm: the judge is the harness. Its decision
                # lives in `harness_goal_active`; the model emits no `active`
                # field at all, so `_first_goal_active` would return None and
                # the arm would score 0 facet-A on a technicality.
                harness_active = trace.get("harness_goal_active")
                first_active = (
                    {str(x) for x in harness_active}
                    if isinstance(harness_active, list) else None)
            else:
                first_active = _first_goal_active(trace)
            if first_active is None:
                bucket["unparsed_first_goal"] += 1
            elif first_active == gt_grasp:
                bucket["first_goal_correct"] += 1
            verdict = _goal_decomposition(first_active, gt_grasp)
            if verdict:
                bucket["goal_decomposition"][verdict] += 1

    report: dict[str, Any] = {"setup": metadata, "split": split, "families": {}}
    total = _new_bucket()
    for family in sorted(per_family):
        b = per_family[family]
        n = max(b["n"], 1)
        facet_a = max(b["facet_a_n"], 1)
        report["families"][family] = {
            "n": b["n"],
            "plan_acc": round(b["plan_acc_sum"] / n, 4),
            "viki_l1_robot_activation": (
                round(b["l1_sum"] / b["l1_n"], 4) if b["l1_n"] else None),
            "errors": b["errors"],
            # constant by construction: 3 tools (+1 on the deterministic retry)
            "tool_calls_mean": round(b["tool_calls_sum"] / n, 4),
            # Also constant by construction, but not always 1.0: the pure-neural
            # arm has no tools, and hardcoding 1.0 would put a tool-call rate on
            # the one row of the four-point table that is defined by having none.
            "tool_call_rate": (
                0.0 if (metadata.get("offload_mode") == "none") else 1.0),
            "verdict_settled_rate": (
                round(b["verdict_settled"] / n, 4)
                if metadata.get("state_verdict") else None),
            "verdict_dropped_ops_mean": (
                round(b["verdict_dropped_ops_sum"] / n, 4)
                if metadata.get("state_verdict") else None),
            "schedule_retry_rate": round(b["schedule_retried"] / n, 4),
            "translator_retry_rate": (
                round(b["translator_retry"] / n, 4)
                if metadata.get("state_verdict") else None),
            "verify_ok_rate": round(b["verify_ok"] / n, 4),
            "format_compliance": round(b["format_compliant"] / n, 4),
            "capability_unassignable_rate": round(
                b["capability_unassignable"] / n, 4),
            "model_named_robot_rate": round(b["model_named_robot"] / n, 4),
            "facet_a_n": b["facet_a_n"],
            "first_goal_correct_rate": round(b["first_goal_correct"] / facet_a, 4),
            "unparsed_first_goal_count": b["unparsed_first_goal"],
            "goal_decomposition": dict(b["goal_decomposition"]),
            "ignored_model_tool_calls_mean": round(
                b["ignored_model_tool_calls_sum"] / n, 4),
        }
        for key, value in b.items():
            if key == "goal_decomposition":
                total[key].update(value)
            else:
                total[key] += value

    n_all = max(total["n"], 1)
    facet_a_all = max(total["facet_a_n"], 1)
    report["overall"] = {
        "n": total["n"],
        "plan_acc": round(total["plan_acc_sum"] / n_all, 4),
        "tool_calls_mean": round(total["tool_calls_sum"] / n_all, 4),
        "tool_call_rate": (
            0.0 if (metadata.get("offload_mode") == "none") else 1.0),
        "verdict_settled_rate": (
            round(total["verdict_settled"] / n_all, 4)
            if metadata.get("state_verdict") else None),
        "verdict_dropped_ops_mean": (
            round(total["verdict_dropped_ops_sum"] / n_all, 4)
            if metadata.get("state_verdict") else None),
        "translator_retry_rate": (
            round(total["translator_retry"] / n_all, 4)
            if metadata.get("state_verdict") else None),
        "format_compliance": round(total["format_compliant"] / n_all, 4),
        "capability_unassignable_rate": round(
            total["capability_unassignable"] / n_all, 4),
        "model_named_robot_rate": round(total["model_named_robot"] / n_all, 4),
        "facet_a_n": total["facet_a_n"],
        "first_goal_correct_rate": round(total["first_goal_correct"] / facet_a_all, 4),
        "unparsed_first_goal_count": total["unparsed_first_goal"],
        "goal_decomposition": dict(total["goal_decomposition"]),
        "errors": total["errors"],
    }
    return report


def _select_samples(args: argparse.Namespace) -> list[BenchmarkSample]:
    samples = load_benchmark_samples(level=args.level, split=args.split)
    task_names = parse_task_name_filters(args.task_name)
    if task_names:
        wanted = set(task_names)
        samples = [s for s in samples if s.task_name in wanted]
    samples = samples[args.start_index:]
    if args.num_shards > 1:
        samples = samples[args.shard_index::args.num_shards]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    return samples


def _build_detect_override_builder(args: argparse.Namespace):
    """(builder, detector) for the shared perception bracket."""
    if args.detector_backend == "off":
        from nsdl.perc_tool import DetectResult  # noqa: WPS433

        def builder(sample: BenchmarkSample):  # noqa: ARG001
            return lambda query=None, crop_entities=None: DetectResult()

        return builder, None
    if args.detector_backend == "oracle":
        # Eval-only, and loudly marked as such in the environment.
        os.environ["NSDL_ORACLE_PERCEPTION"] = "eval-only"
        from nsdl.evaluation import get_sample_ground_truth  # noqa: WPS433
        from nsdl.oracle_perception import (  # noqa: WPS433
            make_oracle_detect_override,
            oracle_scope_mode,
            oracle_vocab_mode,
        )

        # Resolved once, mirroring the main eval runner: a typo fails the run
        # instead of silently producing a default-mode bucket under a "full"
        # name (序 41 runs two scope arms out of the same script).
        oracle_scope = oracle_scope_mode()
        oracle_vocab = oracle_vocab_mode()
        scope_parser = (build_scene_parser(args.scene_parser_backend)
                        if oracle_scope == "full" else None)

        def builder(sample: BenchmarkSample):
            ground_truth = get_sample_ground_truth(sample)
            ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
            scene_facts_for_scope = None
            if scope_parser is not None:
                # Same rule parser the pipeline runs (deterministic), so the
                # instruction-relevant scope matches the main eval's full mode.
                with use_benchmark_profile(VIKI_L2_PROFILE):
                    scene_facts_for_scope = scope_parser.parse(sample)
            return make_oracle_detect_override(
                ground_truth,
                scope_mode=oracle_scope,
                instruction=getattr(sample, "instruction", None),
                scene_facts=scene_facts_for_scope,
                vocab_mode=oracle_vocab,
            )

        return builder, None
    if args.detector_backend == "none":
        return None, None
    assert_detector_caliber_for_level(args.detector_weights, args.level, caller="eval_constant_offload")
    detector = build_structured_detector(
        args.detector_backend,
        weights_path=args.detector_weights,
        confidence_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
        max_det=args.detector_max_det,
        device=args.detector_device,
    )
    return None, detector


def main() -> None:
    args = build_parser().parse_args()
    if args.detector_weights is None:
        args.detector_weights = str(detector_weights_for_level(args.level))
    if args.frontend == "format-sft" and not args.frontend_weights:
        raise SystemExit(
            "--frontend format-sft is the disclosed fallback and needs an "
            "explicit --frontend-weights (a format-only SFT ckpt).")
    if args.frontend == "trained" and not args.frontend_weights:
        raise SystemExit(
            "--frontend trained needs an explicit --frontend-weights (the v3.3 "
            "SFT checkpoint's final/ dir).")
    if args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be < --num-shards")
    if args.offload_mode == "full-translation":
        # nsdl.verdict_perception hard-refuses to import without this. Set it
        # here rather than asking the caller to remember: the arm is
        # meaningless without the verdict, so a missing env var can only ever
        # be an oversight -- and the module's guard exists to keep the verdict
        # out of training/RL, which this eval-only script is not.
        os.environ.setdefault("NSDL_VERDICT_PERCEPTION", "eval-only")
    if args.offload_mode == "none" and args.detector_backend not in {"none", "off"}:
        raise SystemExit(
            "--offload-mode none is the pure-neural point: no solver AND no "
            "perception tool. Pass --detector-backend off.")
    if args.evidence_source == "trained_state":
        if args.offload_mode != "full-translation":
            raise SystemExit(
                "--evidence-source trained_state only makes sense for "
                "--offload-mode full-translation (the only mode with an "
                "externalized state verdict).")
        if not args.state_evidence_dir:
            raise SystemExit(
                "--evidence-source trained_state needs --state-evidence-dir "
                "(precomputed by scripts/run_state_translator_frontend.py).")

    frontend_instructions = FRONTEND_INSTRUCTIONS_BY_MODE[args.offload_mode]

    requested = Path(args.frontend_weights) if args.frontend_weights \
        else BASE_FRONTEND_WEIGHTS
    weights = resolve_local_slm_weights_path(requested)
    # Default bucket for --offload-mode constant is left exactly as it was:
    # the already-landed VIKI-L2 runs live under `base_ultralytics_yolo26` and
    # resume keys off the path, so renaming it would orphan them.
    bucket = args.result_bucket or (
        f"{args.frontend}_{args.detector_backend}"
        if args.offload_mode == "constant"
        else f"{args.offload_mode}_{args.frontend}_{args.detector_backend}")
    output_dir = Path(args.output_dir) / bucket / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    _MODE_META = {
        "constant": (
            "P6.1 point 2 (constant offload)",
            "FLEET / OSDAG / IMR-LLM (fixed offload, untrained front-end)",
            "hardwired by the harness; model tool calls are ignored"),
        "none": (
            "P6.1 point 1 (pure neural)",
            "end-to-end VLM planner, no symbolic backend",
            "no tools exist; model tool calls are ignored"),
        "full-translation": (
            "P6.1 point 3 (full-translation offload)",
            "AHAT / LLM-as-formalizer (constant translation offload, no "
            "gating, no state judgement)",
            "hardwired by the harness; the state verdict is also the "
            "harness's (nsdl.verdict_perception)"),
    }
    _control, _equivalent, _gating = _MODE_META[args.offload_mode]
    oracle_scope = oracle_vocab = None
    if args.detector_backend == "oracle":
        # 序 41 traceability: record which scope/vocab arm produced the bucket
        # (same fields the main eval runner writes next to detector_backend).
        os.environ["NSDL_ORACLE_PERCEPTION"] = "eval-only"
        from nsdl.oracle_perception import (  # noqa: WPS433
            oracle_scope_mode,
            oracle_vocab_mode,
        )
        oracle_scope = oracle_scope_mode()
        oracle_vocab = oracle_vocab_mode()
    metadata = {
        "control": _control,
        "equivalent_of": _equivalent,
        "offload_mode": args.offload_mode,
        "state_verdict": args.offload_mode == "full-translation",
        "fixed_tool_order": ([] if args.offload_mode == "none"
                             else list(FIXED_TOOL_ORDER)),
        "gating": _gating,
        "frontend": args.frontend,
        "frontend_trained": (
            "v3.3-sft" if args.frontend == "trained"
            else (False if args.frontend == "base" else "format-only")),
        "frontend_disclosure": (
            "untrained base VLM, prompt-only op-vocabulary spec"
            if args.frontend == "base" else
            "v3.3 SFT front-end, control prompt (translator role)"
            if args.frontend == "trained" else
            "format-aligned but capability-untrained (documented fallback)"),
        "frontend_weights": str(weights),
        "detector_backend": args.detector_backend,
        "oracle_scope": oracle_scope,
        "oracle_vocab": oracle_vocab,
        "evidence_source": args.evidence_source,
        "state_evidence_dir": args.state_evidence_dir,
        "shared_tool_runtime": "nsdl.tool_runtime.execute_tool_call",
        "phi_scheduler": _phi_scheduler_mode(),
        "level": args.level,
        "split": args.split,
    }

    if not args.report_only:
        samples = _select_samples(args)
        if not samples:
            raise SystemExit("No samples selected for evaluation.")

        state_evidence_by_stem: dict[str, dict[str, Any]] | None = None
        if args.evidence_source == "trained_state":
            evidence_dir = Path(args.state_evidence_dir) / args.split
            state_evidence_by_stem = {}
            missing: list[str] = []
            for sample in samples:
                stem = build_sample_stem(sample)
                path = evidence_dir / f"{stem}.json"
                if not path.exists():
                    missing.append(stem)
                    continue
                state_evidence_by_stem[stem] = json.loads(
                    path.read_text(encoding="utf-8"))
            if missing:
                # Hard-fail, never score missing evidence as "no relations"
                # (memory 'missing-column-is-not-zero').
                raise SystemExit(
                    f"--evidence-source trained_state: {len(missing)} sample(s) "
                    f"have no evidence file under {evidence_dir} "
                    f"(first: {missing[:3]}). Run "
                    f"scripts/run_state_translator_frontend.py to completion "
                    f"first.")

        load_local_vlm_with_fallback(str(weights), args.device)
        from nsdl.slm_backends import get_local_vl_backend  # noqa: WPS433

        backend = get_local_vl_backend()
        detect_override_builder, detector = _build_detect_override_builder(args)
        scene_parser = build_scene_parser(args.scene_parser_backend)
        scene_refiner = build_scene_facts_refiner(args.scene_facts_refiner_backend)

        print(f"[P6.1b/{args.split}] start total={len(samples)} bucket={bucket} "
              f"frontend={args.frontend} weights={weights} "
              f"detector={args.detector_backend} out={output_dir}", flush=True)

        processed = skipped = errors = 0
        started = time.time()
        log_every = max(int(args.log_every), 1)
        for position, sample in enumerate(samples, start=1):
            output_path = output_dir / f"{build_sample_stem(sample)}.json"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue
            sample_started = time.time()
            trace: dict[str, Any] | None = None
            try:
                if args.offload_mode == "none":
                    result, trace = run_pure_neural_sample(
                        sample,
                        backend=backend,
                        scene_parser=scene_parser,
                        scene_refiner=scene_refiner,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        stop_strings=args.stop_strings,
                        frontend_instructions=frontend_instructions,
                    )
                else:
                    result, trace = run_constant_offload_sample(
                        sample,
                        backend=backend,
                        detector=detector,
                        scene_parser=scene_parser,
                        scene_refiner=scene_refiner,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        stop_strings=args.stop_strings,
                        detect_override=(detect_override_builder(sample)
                                         if detect_override_builder else None),
                        frontend_instructions=frontend_instructions,
                        offload_mode=args.offload_mode,
                        state_verdict=(args.offload_mode == "full-translation"),
                        state_evidence=(
                            state_evidence_by_stem[build_sample_stem(sample)]
                            if state_evidence_by_stem is not None else None),
                    )
                artifact = build_result_artifact(
                    result, validate_l2=not args.skip_validate)
                artifact["result_status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                errors += 1
                wrapped = exc if isinstance(exc, AgenticEvaluationError) else \
                    AgenticEvaluationError(stage="unknown", sample=sample,
                                           original_exc=exc)
                artifact = build_error_artifact(sample, wrapped)
                artifact["result_status"] = "error"
                trace = getattr(wrapped, "trace", None) or trace
                if _is_cuda_context_fatal(exc):
                    # A dead CUDA context does not recover: every remaining
                    # sample would fail identically. Measured on the 2026-08-01
                    # OOD run -- sample 20 hit "unspecified launch failure" and
                    # the loop manufactured 1198 more error artifacts, yielding
                    # plan_acc 0.0026 / format_compliance 0.0164 that read as a
                    # capability finding about the untrained front-end.
                    #
                    # Deliberately do NOT persist this artifact. The resume
                    # check is `output_path.exists()` alone -- it never looks at
                    # result_status -- so a written one is skipped by the very
                    # re-run that is meant to replace it, and the fault silently
                    # becomes a data point. Leaving it unwritten makes the
                    # sample simply absent, which the resume then re-runs.
                    raise SystemExit(
                        f"[P6.1b/{args.split}] FATAL CUDA fault at sample "
                        f"{position}/{len(samples)} ({output_path.name}): "
                        f"{exc}\n"
                        f"  The CUDA context is unrecoverable, so continuing "
                        f"would write only error artifacts.\n"
                        f"  Aborting after {processed} processed / {errors} "
                        f"errors. This sample's artifact was NOT written, so a "
                        f"plain resume re-runs it;\n"
                        f"  every earlier artifact under\n"
                        f"    {output_dir}\n"
                        f"  predates the fault and stays valid (the skip check "
                        f"tests existence, not result_status).\n"
                        f"  Known cause on this host: VRAM headroom under a "
                        f"co-tenant -- re-run with "
                        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.")

            artifact["runtime_seconds"] = round(time.time() - sample_started, 3)
            artifact["slm_modality"] = "vl"
            artifact["constant_offload"] = dict(metadata, trace=trace)
            output_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2,
                           default=_json_fallback),
                encoding="utf-8")
            processed += 1
            if position == 1 or position % log_every == 0:
                print(f"[P6.1b/{args.split}] {position}/{len(samples)} "
                      f"processed={processed} skipped={skipped} errors={errors} "
                      f"last={output_path.name}", flush=True)

        print(f"[P6.1b/{args.split}] done processed={processed} "
              f"skipped={skipped} errors={errors} "
              f"elapsed_s={round(time.time() - started, 2)}", flush=True)

    report = report_per_family(output_dir, split=args.split, metadata=metadata,
                               level=args.level)
    # VIKI-L2 has exactly two eval splits, so ID/OOD is a faithful label there.
    # LOCUS has four (test_id/ood_s/ood_m/ood_x) and three of them would all
    # map to "OOD" -- each split would silently overwrite the previous one's
    # report and the surviving file would be read as if it covered the grid.
    # Name LOCUS reports by split.
    label = _report_label(args.level, args.split)
    report_path = Path(args.output_dir) / bucket / f"{label}_per_family.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    overall = report["overall"]
    print(f"[P6.1b/{args.split}] plan_acc={overall['plan_acc']} "
          f"first_goal_correct_rate={overall['first_goal_correct_rate']} "
          f"format_compliance={overall['format_compliance']} "
          f"tool_calls_mean={overall['tool_calls_mean']} -> {report_path}",
          flush=True)
    if overall["format_compliance"] < 0.5 and args.frontend == "base":
        print("[P6.1b] WARNING: base front-end format compliance < 0.5. Per "
              "plan.md §P6.1b this is the trigger for the documented "
              "--frontend format-sft fallback; disclose it as format-aligned, "
              "capability-untrained.", flush=True)


if __name__ == "__main__":
    main()
