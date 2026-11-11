from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdl.prompt_adapter import load_benchmark_samples  # noqa: E402


# E1+ (debug_sftv2.md §E / cluster E): facet-A process metrics MUST be computed
# from the trace's assistant turns. `analysis.blocks.goal_active` comes from the
# final response only (usually a bare [PLAN] turn, goal=None) and silently
# counts as "no over-grasp" — the 99.5% skip artifact. Do not use it here.
_GOAL_BLOCK_RE = re.compile(r"\[GOAL\]\s*(\{.*?\})\s*(?=\[|<|\Z)", re.DOTALL)

# Robot slot ids ("R1", "R2", ...). Used to tell a handoff `Push x R2` from a
# relocation `Push x sink`; see `_active_entities`.
_ROBOT_SLOT_RE = re.compile(r"^R\d+$")


def _first_goal_active(trace: dict[str, Any]) -> set[str] | None:
    """Active item set from the first parseable [GOAL] across assistant turns."""
    for turn in trace.get("assistant_turns") or []:
        text = turn if isinstance(turn, str) else json.dumps(turn)
        match = _GOAL_BLOCK_RE.search(text)
        if not match:
            continue
        try:
            goal = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        items: set[str] = set()
        for item in goal.get("active") or []:
            name = item if isinstance(item, str) else (item or {}).get("item")
            if name:
                items.add(str(name))
        return items
    return None


def _robot_slot_ids(steps: Any) -> set[str]:
    """Slot ids actually used by a step list, e.g. {"R1", "R2", "R3"}."""
    slots: set[str] = set()
    for step in steps or []:
        if isinstance(step, dict):
            slots.update(str(key) for key in (step.get("actions") or {}))
    return slots


def _active_entities(steps: Any) -> set[str]:
    """Entities a benchmark-format step list relocates (plan or GT).

    ``Grasp x`` is the carried case. ``Push x <target>`` is the carrier case and
    only counts when <target> is a place, not a robot: VIKI's push family hands
    the box between robots (``Push cardboardbox R2``) where the box is a vehicle
    and the supervision lists only the payload as active, while LOCUS F9 also
    pushes the box *to a place* (``Push cardboardbox sink``) and there the box
    carries a goal constraint of its own. Counting every ``Push`` would break the
    first case; counting none silently scores the second one wrong.
    """
    slots = _robot_slot_ids(steps)
    active: set[str] = set()
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        for action in (step.get("actions") or {}).values():
            if not isinstance(action, (list, tuple)) or len(action) < 2:
                continue
            if action[0] == "Grasp":
                active.add(str(action[1]))
            elif action[0] == "Push" and len(action) >= 3:
                target = str(action[2])
                if target not in slots and not _ROBOT_SLOT_RE.match(target):
                    active.add(str(action[1]))
    return active


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize VIKI-L1/L2 evaluation results in a result folder.")
    parser.add_argument("--results-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    result_files = sorted(
        path for path in results_dir.glob("*.json")
        if not path.name.startswith("_")
    )
    if not result_files:
        raise ValueError(f"No result json files found under {results_dir}")

    total_l1 = 0.0
    total_l2 = 0.0
    count_l1 = 0
    count_l2 = 0
    ok_count = 0
    error_count = 0
    validator_success = 0
    validator_total = 0
    log_schema_versions: Counter[str] = Counter()
    stopped_reasons: Counter[str] = Counter()
    error_stages: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    tool_combo: Counter[str] = Counter()
    tool_call_count_dist: Counter[int] = Counter()
    emitted_tool_calls = 0
    executed_tool_calls = 0
    samples_emitted_gt_executed = 0
    stacked_tool_call_turns = 0
    samples_with_stacked_tool_calls = 0
    suppressed_tool_calls = 0
    parse_diagnostics = 0
    parse_diagnostic_kinds: Counter[str] = Counter()
    trace_v2_count = 0
    missing_trace_count = 0
    operation_validation_total = 0
    operation_validation_success = 0
    # E1: protocol-fail vs plan-fail split + completed-only accuracy.
    protocol_fail = 0
    plan_fail = 0
    completed_l2_sum = 0.0
    completed_l2_count = 0
    # E1+ facet-A process metrics (trace-derived; see module docstring note).
    facet_a_total = 0
    first_goal_correct = 0
    final_grasp_correct = 0
    direct_judgment = 0        # first GOAL correct without a verify_goal_coverage call
    rescued = 0                # first GOAL wrong -> final grasp set correct
    rescue_candidates = 0      # first GOAL wrong
    unparsed_first_goal = 0

    first_payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    sample_meta = first_payload.get("sample", {})
    level = sample_meta.get("level")
    split = sample_meta.get("split")
    expected_count = None
    if level and split:
        try:
            expected_count = len(load_benchmark_samples(level=level, split=split))
        except Exception:
            expected_count = None

    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("result_status") == "error":
            error_count += 1
            error_payload = payload.get("error") or {}
            if isinstance(error_payload, dict):
                error_stages.update([str(error_payload.get("pipeline_stage") or "unknown")])
        else:
            ok_count += 1

        scores = payload.get("benchmark_scores") or {}
        l1_score = scores.get("viki_l1_robot_activation")
        l2_score = scores.get("viki_l2_plan_score")
        if isinstance(l1_score, (int, float)):
            total_l1 += float(l1_score)
            count_l1 += 1
        if isinstance(l2_score, (int, float)):
            total_l2 += float(l2_score)
            count_l2 += 1

        validation = payload.get("validation")
        if isinstance(validation, dict) and validation.get("success") is not None:
            validator_total += 1
            if validation.get("success"):
                validator_success += 1

        agentic_eval = payload.get("agentic_eval") or {}
        if isinstance(agentic_eval, dict):
            log_schema_versions.update([str(agentic_eval.get("log_schema_version") or "legacy")])
            trace = agentic_eval.get("trace")
            if isinstance(trace, dict):
                if trace.get("schema_version") == 2:
                    trace_v2_count += 1
                stopped_reasons.update([str(trace.get("stopped_reason") or "unknown")])
                names = [
                    str(name)
                    for name in (trace.get("tool_names") or [])
                    if str(name).strip()
                ]
                tool_names.update(names)
                tool_combo.update(["+".join(names) if names else "none"])
                emitted_count = int(trace.get("emitted_tool_call_count") or len(names))
                executed_count = int(trace.get("executed_tool_call_count") or len(names))
                emitted_tool_calls += emitted_count
                executed_tool_calls += executed_count
                tool_call_count_dist.update([emitted_count])
                if emitted_count > executed_count:
                    samples_emitted_gt_executed += 1
                diagnostics = trace.get("parse_diagnostics") or []
                if isinstance(diagnostics, list):
                    parse_diagnostics += len(diagnostics)
                    parse_diagnostic_kinds.update(
                        str(item.get("kind") or "unknown")
                        for item in diagnostics
                        if isinstance(item, dict)
                    )
                sample_stacked = False
                for event in trace.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    event_calls = event.get("tool_calls") or []
                    if isinstance(event_calls, list) and len(event_calls) > 1:
                        stacked_tool_call_turns += 1
                        sample_stacked = True
                    event_suppressed = event.get("suppressed_tool_calls") or []
                    if isinstance(event_suppressed, list):
                        suppressed_tool_calls += len(event_suppressed)
                if sample_stacked:
                    samples_with_stacked_tool_calls += 1
                op_validation = trace.get("operation_validation")
                if isinstance(op_validation, dict) and op_validation.get("ok") is not None:
                    operation_validation_total += 1
                    if op_validation.get("ok"):
                        operation_validation_success += 1
            else:
                missing_trace_count += 1

        # --- E1: protocol-fail / plan-fail split + completed-only accuracy ---
        trace_dict = trace if isinstance(trace, dict) else {}
        is_protocol_fail = (
            payload.get("result_status") == "error"
            or (trace_dict.get("stopped_reason") not in (None, "final_plan"))
        )
        if is_protocol_fail:
            protocol_fail += 1
        elif isinstance(l2_score, (int, float)):
            completed_l2_sum += float(l2_score)
            completed_l2_count += 1
            if float(l2_score) < 0.999:
                plan_fail += 1

        # --- E1+: facet-A process metrics (trace assistant turns vs GT) ---
        ground_truth = payload.get("ground_truth") or {}
        gt_steps = ground_truth.get("time_steps") if isinstance(ground_truth, dict) else None
        if trace_dict and isinstance(gt_steps, list):
            facet_a_total += 1
            gt_grasp = _active_entities(gt_steps)
            final_grasp = _active_entities(payload.get("plan"))
            first_active = _first_goal_active(trace_dict)
            used_verify = "verify_goal_coverage" in (trace_dict.get("tool_names") or [])
            final_ok = final_grasp == gt_grasp and payload.get("result_status") != "error"
            if final_ok:
                final_grasp_correct += 1
            if first_active is None:
                unparsed_first_goal += 1
            elif first_active == gt_grasp:
                first_goal_correct += 1
                if not used_verify:
                    direct_judgment += 1
            else:
                rescue_candidates += 1
                if final_ok:
                    rescued += 1

    print(f"results_dir: {results_dir}")
    print(f"level: {level}")
    print(f"split: {split}")
    print(f"result_files: {len(result_files)}")
    if expected_count is not None:
        print(f"dataset_expected: {expected_count}")
        print(f"dataset_complete: {len(result_files) == expected_count}")
    print(f"ok_count: {ok_count}")
    print(f"error_count: {error_count}")
    if count_l1:
        print(f"viki_l1_sum: {round(total_l1, 6)}")
        print(f"viki_l1_avg: {round(total_l1 / count_l1, 6)}")
    if count_l2:
        print(f"viki_l2_sum: {round(total_l2, 6)}")
        print(f"viki_l2_avg: {round(total_l2 / count_l2, 6)}")
    if validator_total:
        print(f"validator_success_count: {validator_success}")
        print(f"validator_success_rate: {round(validator_success / validator_total, 6)}")
    if log_schema_versions:
        print(f"log_schema_versions: {dict(sorted(log_schema_versions.items()))}")
        print(f"trace_v2_count: {trace_v2_count}")
        print(f"missing_trace_count: {missing_trace_count}")
    if stopped_reasons:
        print(f"stopped_reasons: {dict(sorted(stopped_reasons.items()))}")
        max_rounds = stopped_reasons.get("max_tool_rounds", 0)
        print(f"max_tool_rounds_count: {max_rounds}")
        print(f"max_tool_rounds_rate: {round(max_rounds / len(result_files), 6)}")
    if error_stages:
        print(f"error_stages: {dict(sorted(error_stages.items()))}")
    if operation_validation_total:
        print(f"operation_validation_success_count: {operation_validation_success}")
        print(f"operation_validation_success_rate: {round(operation_validation_success / operation_validation_total, 6)}")
    # E1: protocol-fail vs plan-fail + completed-only accuracy.
    total_files = len(result_files)
    print(f"protocol_fail_count: {protocol_fail}")
    print(f"protocol_fail_rate: {round(protocol_fail / total_files, 6)}")
    print(f"plan_fail_count: {plan_fail}")
    print(f"plan_fail_rate: {round(plan_fail / total_files, 6)}")
    if completed_l2_count:
        print(f"completed_only_count: {completed_l2_count}")
        print(f"completed_only_l2_avg: {round(completed_l2_sum / completed_l2_count, 6)}")
    # E1+ facet-A process metrics (trace-derived; never from blocks.goal_active).
    if facet_a_total:
        print(f"facet_a_samples: {facet_a_total}")
        print(f"first_goal_grasp_match_rate: {round(first_goal_correct / facet_a_total, 6)}")
        print(f"final_grasp_match_rate: {round(final_grasp_correct / facet_a_total, 6)}")
        print(f"direct_judgment_rate: {round(direct_judgment / facet_a_total, 6)}")
        print(f"first_goal_wrong_count: {rescue_candidates}")
        if rescue_candidates:
            print(f"verify_rescue_rate: {round(rescued / rescue_candidates, 6)}")
        print(f"unparsed_first_goal_count: {unparsed_first_goal}")
    if log_schema_versions:
        print(f"emitted_tool_calls: {emitted_tool_calls}")
        print(f"executed_tool_calls: {executed_tool_calls}")
        print(f"emitted_executed_match_rate: {round((len(result_files) - samples_emitted_gt_executed) / len(result_files), 6)}")
        print(f"samples_emitted_gt_executed: {samples_emitted_gt_executed}")
        print(f"tool_call_count_dist: {dict(sorted(tool_call_count_dist.items()))}")
        print(f"stacked_tool_call_turns: {stacked_tool_call_turns}")
        print(f"samples_with_stacked_tool_calls: {samples_with_stacked_tool_calls}")
        print(f"suppressed_tool_calls: {suppressed_tool_calls}")
        print(f"parse_diagnostic_count: {parse_diagnostics}")
        print(f"parse_diagnostic_kinds: {dict(sorted(parse_diagnostic_kinds.items()))}")
        print(f"tool_name_counts: {dict(sorted(tool_names.items()))}")
        print(f"tool_combo_counts: {dict(sorted(tool_combo.items()))}")


if __name__ == "__main__":
    main()
