"""P10.S2 `--phi-off`: the build-side strip and its teacher-cache shim.

Two invariants carry the whole ablation and are asserted here rather than
inferred:

1. `final_plan` is byte-identical to the phi_on build. The Phi-off arm is only
   fair if both arms are graded against the same complete DAG -- if the strip
   ever leaked into the supervision target, the S2 gain column would be
   measuring two different tasks.
2. `teacher_cache_key` is unchanged by the strip. That is what keeps the v3.3
   cache at 100% and stops ~16k distillation calls from hitting a relay API this
   project has repeatedly measured as flaky.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsdl.tool_runtime import PHI_TOOLS  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "build_lnsdl_sft_data", ROOT / "scripts" / "build_lnsdl_sft_data.py"
)
bsd = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(bsd)


GOAL = {"active": ["bowl"], "skip": [], "targets": [{"item": "bowl", "destination": "table"}]}
ASSIGN = {"R1": {"items": ["bowl"]}}
SKETCH = [{"op_id": "o1", "op_type": "relocate", "entity": "bowl", "robot": "R1"}]
FINAL_PLAN = [
    {"op_id": "o1", "op_type": "pick", "entity": "bowl", "robot": "R1"},
    {"op_id": "o2", "op_type": "carry_to", "entity": "bowl", "robot": "R1", "deps": ["o1"]},
]

DETECT_CALL = {"name": "detect", "args": {"query": "bowl on the counter"}}
COVERAGE_CALL = {"name": "verify_goal_coverage", "args": {"model_goal": GOAL}}
SCHEDULE_CALL = {"name": "schedule", "args": {"ops": SKETCH, "assign": ASSIGN}}
FAILED_SCHEDULE_CALL = {"name": "schedule", "args": {"ops": [], "assign": ASSIGN}}


def make_item(
    *,
    tool_calls: list[dict],
    pre_goal_tool_count: int = 1,
    coverage_repair: dict | None = None,
    repair: dict | None = None,
) -> object:
    return bsd.PreparedTrajectory(
        row_index=0,
        task_id="t0",
        task_name="task-0",
        instruction="put the bowl on the table",
        image_path="/dev/null",
        scene_facts=None,
        affordance_context=None,
        system_prompt="SYSTEM",
        user_prompt="USER",
        goal=copy.deepcopy(GOAL),
        assign=copy.deepcopy(ASSIGN),
        plan_sketch=copy.deepcopy(SKETCH),
        final_plan=copy.deepcopy(FINAL_PLAN),
        tool_calls=copy.deepcopy(tool_calls),
        tool_results=[{"ok": True, "i": i} for i in range(len(tool_calls))],
        pre_goal_tool_count=pre_goal_tool_count,
        coverage_repair=copy.deepcopy(coverage_repair),
        repair=copy.deepcopy(repair),
        audit={
            "tool_names": [call["name"] for call in tool_calls],
            "schedule_used": any(call["name"] == "schedule" for call in tool_calls),
            "repair_injected": bool(repair),
            "f4_cross_turn_schedule": True,
            "pre_goal_tool_count": pre_goal_tool_count,
        },
    )


def cache_key(item: object) -> str:
    return bsd.teacher_cache_key(
        item, "gpt-5.6-terra", include_image=True, post_include_image=False, think_mode="blind"
    )


# ---------- the strip ----------

def test_only_phi_tools_are_removed() -> None:
    item = make_item(tool_calls=[DETECT_CALL, COVERAGE_CALL, SCHEDULE_CALL])
    bsd.strip_phi_from_trajectory(item)
    names = [call["name"] for call in item.tool_calls]
    assert names == ["detect", "verify_goal_coverage"]
    assert not PHI_TOOLS.intersection(names)
    # detect is facet-A perception and verify_goal_coverage is a read-only
    # diagnosis of the model's own [GOAL]; ablating either would stop the axis
    # from measuring Phi-offload.
    assert item.audit["tool_names"] == names
    assert item.audit["phi_off_calls_removed"] == 1


def test_results_stay_aligned_with_calls() -> None:
    item = make_item(tool_calls=[DETECT_CALL, SCHEDULE_CALL])
    bsd.strip_phi_from_trajectory(item)
    assert len(item.tool_results) == len(item.tool_calls) == 1
    assert item.tool_results[0] == {"ok": True, "i": 0}


def test_the_supervision_target_is_untouched() -> None:
    item = make_item(tool_calls=[DETECT_CALL, SCHEDULE_CALL])
    before = json.dumps(item.final_plan, sort_keys=True)
    bsd.strip_phi_from_trajectory(item)
    assert json.dumps(item.final_plan, sort_keys=True) == before
    assert item.goal == GOAL and item.assign == ASSIGN and item.plan_sketch == SKETCH


def test_the_repair_demo_cannot_outlive_its_tool() -> None:
    # The repair round is a failed-schedule / re-schedule pair, so keeping it
    # would leave dangling absolute tool indices pointing at removed calls.
    repair = {"prefix_tool_count": 1, "failed_plan_sketch": SKETCH}
    item = make_item(tool_calls=[DETECT_CALL, FAILED_SCHEDULE_CALL, SCHEDULE_CALL], repair=repair)
    bsd.strip_phi_from_trajectory(item)
    assert item.repair is None
    assert item.audit["repair_injected"] is False
    assert [call["name"] for call in item.tool_calls] == ["detect"]


def test_audit_flags_describe_the_emitted_trajectory() -> None:
    item = make_item(tool_calls=[DETECT_CALL, SCHEDULE_CALL])
    bsd.strip_phi_from_trajectory(item)
    assert item.audit["phi_off"] is True
    assert item.audit["schedule_used"] is False
    assert item.audit["phi_off_source_schedule_used"] is True
    assert item.audit["f4_cross_turn_schedule"] is False


def test_a_non_suffix_phi_call_fails_loudly() -> None:
    # coverage_repair["tool_index"] and repair["prefix_tool_count"] are ABSOLUTE
    # indices into tool_calls. Phi is always appended last today, but if that
    # ever changed the renderers would misalign silently instead of erroring.
    item = make_item(tool_calls=[SCHEDULE_CALL, DETECT_CALL])
    with pytest.raises(AssertionError, match="suffix"):
        bsd.strip_phi_from_trajectory(item)


def test_pre_goal_count_survives_untouched() -> None:
    item = make_item(tool_calls=[DETECT_CALL, DETECT_CALL, SCHEDULE_CALL], pre_goal_tool_count=2)
    bsd.strip_phi_from_trajectory(item)
    assert item.pre_goal_tool_count == 2  # Phi never sits in the leading detects
    assert item.audit["pre_goal_tool_count"] == 2


def test_phi_inside_the_pre_goal_prefix_fails_loudly() -> None:
    # Clamping instead would move teacher_cache_key (the count is a key field)
    # and hide the ordering bug behind a cache that still looks 100%.
    item = make_item(tool_calls=[SCHEDULE_CALL], pre_goal_tool_count=1)
    with pytest.raises(AssertionError, match="pre-GOAL prefix"):
        bsd.strip_phi_from_trajectory(item)


# ---------- the cache shim ----------

@pytest.mark.parametrize(
    "tool_calls,pre_goal,repair",
    [
        ([DETECT_CALL, SCHEDULE_CALL], 1, None),
        ([DETECT_CALL, COVERAGE_CALL, SCHEDULE_CALL], 1, None),
        ([DETECT_CALL, FAILED_SCHEDULE_CALL, SCHEDULE_CALL], 1, {"prefix_tool_count": 1}),
        ([SCHEDULE_CALL], 0, None),  # no pre-goal detect: the row goes to 0 tools
        ([COVERAGE_CALL, SCHEDULE_CALL], 0, None),
    ],
)
def test_the_strip_does_not_move_the_teacher_cache_key(tool_calls, pre_goal, repair) -> None:
    # This is the whole reason the arm strips after building instead of using
    # --schedule-policy none: a moved key means ~6416/7196 rows re-distill.
    phi_on = make_item(tool_calls=tool_calls, pre_goal_tool_count=pre_goal, repair=repair)
    phi_off = make_item(tool_calls=tool_calls, pre_goal_tool_count=pre_goal, repair=repair)
    expected = cache_key(phi_on)
    bsd.strip_phi_from_trajectory(phi_off)
    assert cache_key(phi_off) == expected


def test_the_shim_only_applies_to_phi_off_rows() -> None:
    # A phi_on row must not pick up the legacy-repair override, or v3.3 keys
    # would shift for the arm that is supposed to be unchanged.
    item = make_item(tool_calls=[DETECT_CALL, SCHEDULE_CALL], repair={"prefix_tool_count": 1})
    item.cache_legacy_repair = None  # would zero the key if wrongly honoured
    assert "phi_off" not in item.audit
    assert cache_key(item) == cache_key(make_item(
        tool_calls=[DETECT_CALL, SCHEDULE_CALL], repair={"prefix_tool_count": 1}
    ))


def test_the_query_source_shim_still_wins_for_tools() -> None:
    # v3.2 already overrides `tools` with the goal-derived legacy detect call;
    # the phi_off shim must not clobber it, or instruction-query rows re-distill.
    legacy = [{"name": "detect", "args": {"query": "bowl"}}, SCHEDULE_CALL]
    item = make_item(tool_calls=[DETECT_CALL, SCHEDULE_CALL])
    item.cache_legacy_tool_calls = copy.deepcopy(legacy)
    reference = cache_key(item)
    bsd.strip_phi_from_trajectory(item)
    assert item.cache_legacy_tool_calls == legacy  # not overwritten by the strip
    assert cache_key(item) == reference


# ---------- the rendered trajectory ----------

def test_a_fully_stripped_row_still_emits_the_plan() -> None:
    # render_initial_assistant only appends [PLAN] when there are no tool calls
    # AND no repair -- both hold after the strip, so the ~6.4k newly 0-tool rows
    # land on the single-turn path v3.3 already exercises with 413 rows.
    item = make_item(tool_calls=[SCHEDULE_CALL], pre_goal_tool_count=0)
    bsd.strip_phi_from_trajectory(item)
    teacher = bsd.TeacherThink(pre_tool="thinking", status="ok", model="gpt-5.6-terra")
    text = bsd.render_initial_assistant(item, teacher, tool_calls=item.tool_calls)
    for block in ("[GOAL]", "[ASSIGN]", "[PLAN_SKETCH]", "[PLAN]"):
        assert block in text
    assert "[TOOL_CALL]" not in text
    assert json.dumps(FINAL_PLAN, ensure_ascii=False, sort_keys=True) in text
