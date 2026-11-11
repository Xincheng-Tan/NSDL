"""P6.1 point 4-AR: --force-phi-reschedule (always-call arm) runtime checks.

The arm's contract (plan.md A0d(2)): the ONLY change vs point 4 is that a
direct-emit final [PLAN] -- an episode that never executed a `schedule` call --
is forced through `schedule` and scored on the solver's completed ops. These
tests pin the discriminating edges: replacement, the already-scheduled no-op,
the flag-off byte path, sketch-source fallbacks, the schedule-only condition
(a `verify` call does not count as having consulted the scheduler), the honest
fallback on a failed forced call, and the clean TC accounting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.schema import RobotInstance, SceneFacts  # noqa: E402
from nsdl.tool_runtime import run_agentic_tool_loop  # noqa: E402


def _scene() -> tuple[SceneFacts, object]:
    scene = SceneFacts(
        parser_backend="test",
        instruction="bring the apple to the table",
        image_available=True,
        robot_inventory=[RobotInstance(slot_id="R1", robot_type="fetch")],
        primary_object=None,
        destination=None,
        mentioned_objects=["apple", "table"],
    )
    return scene, build_affordance_context(scene)


_SKETCH = [
    {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1", "gaps": ["deps"]},
    {
        "op_id": "o2",
        "op_type": "deposit",
        "entity": "apple",
        "dest": "table",
        "robot": "R1",
        "gaps": ["deps"],
    },
]
_ASSIGN = {"R1": {"items": ["apple"]}}
# A direct-emit plan the model "wrote itself": missing the o2->o1 dep that
# schedule's completion pass adds, so replacement is observable.
_MODEL_PLAN = [
    {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
    {"op_id": "o2", "op_type": "deposit", "entity": "apple", "dest": "table", "robot": "R1"},
]


def _direct_emit_turn() -> str:
    return (
        "[ASSIGN]\n" + json.dumps(_ASSIGN)
        + "\n[PLAN_SKETCH]\n" + json.dumps(_SKETCH)
        + "\n[PLAN]\n" + json.dumps(_MODEL_PLAN)
    )


class _OneTurnBackend:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        return self._text


def test_direct_emit_plan_is_rescheduled() -> None:
    scene, ctx = _scene()
    result = run_agentic_tool_loop(
        _OneTurnBackend(_direct_emit_turn()),
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
    )
    assert result.stopped_reason == "final_plan"
    record = result.forced_phi_reschedule
    assert record is not None
    assert record["applied"] is True
    assert record["outcome"] == "rescheduled"
    assert record["source"] == "current_sketch"
    assert record["call_shape"] == "op_ids"
    assert record["schedule_result"]["ok"] is True
    assert record["latency_s"] >= 0
    # The model's own (dep-less) plan is preserved for the paired analysis...
    assert [op["deps"] for op in record["model_plan"]] == [[], []]
    # ...while the scored operations are the solver's completed ops.
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == ["o1"]
    # The forced call stays OUT of the normal tool accounting (TC parity).
    assert result.tool_results == []
    assert result.events[-1]["forced_phi_reschedule"] is record


def test_flag_off_is_byte_identical_no_op() -> None:
    scene, ctx = _scene()
    result = run_agentic_tool_loop(
        _OneTurnBackend(_direct_emit_turn()),
        scene,
        ctx,
        max_tool_rounds=2,
    )
    assert result.forced_phi_reschedule is None
    assert "forced_phi_reschedule" not in result.events[-1]
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == []  # the model's own plan, untouched


def test_already_scheduled_episode_is_untouched() -> None:
    scene, ctx = _scene()

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return (
                    "[PLAN_SKETCH]\n" + json.dumps(_SKETCH)
                    + "\n[TOOL_CALL]\n"
                    + json.dumps({"name": "schedule", "args": {"op_ids": ["o1", "o2"], "assign": _ASSIGN}})
                )
            assert messages[-1]["role"] == "tool"
            return "[PLAN]\n" + json.dumps(_MODEL_PLAN)

    result = run_agentic_tool_loop(
        Backend(), scene, ctx, max_tool_rounds=2, force_phi_reschedule=True
    )
    assert result.stopped_reason == "final_plan"
    # schedule executed in round 0 => not a direct-emit sample => no forcing,
    # even though the model's final plan ignored the solver output.
    assert result.forced_phi_reschedule is None
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == []


def test_verify_call_does_not_count_as_schedule() -> None:
    scene, ctx = _scene()

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return (
                    "[PLAN_SKETCH]\n" + json.dumps(_SKETCH)
                    + "\n[TOOL_CALL]\n"
                    + json.dumps({"name": "verify", "args": {"op_ids": ["o1", "o2"]}})
                )
            assert messages[-1]["role"] == "tool"
            return "[PLAN]\n" + json.dumps(_MODEL_PLAN)

    result = run_agentic_tool_loop(
        Backend(), scene, ctx, max_tool_rounds=2, force_phi_reschedule=True
    )
    record = result.forced_phi_reschedule
    assert record is not None and record["outcome"] == "rescheduled"
    # The sketch came from the earlier turn (final turn had [PLAN] only).
    assert record["source"] == "carried_sketch"
    # The model's own verify call still counts in normal tool accounting.
    assert len(result.tool_results) == 1


def test_bare_plan_without_sketch_falls_back_to_plan_ops() -> None:
    scene, ctx = _scene()
    result = run_agentic_tool_loop(
        _OneTurnBackend("[PLAN]\n" + json.dumps(_MODEL_PLAN)),
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
    )
    record = result.forced_phi_reschedule
    assert record is not None
    assert record["source"] == "final_plan_ops"
    assert record["outcome"] == "rescheduled"
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == ["o1"]


def test_failed_forced_schedule_keeps_model_plan() -> None:
    scene, ctx = _scene()
    result = run_agentic_tool_loop(
        _OneTurnBackend(_direct_emit_turn()),
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
        # Unit-level failure injection: with schedule disabled the forced call
        # is answered with tool_unavailable, exercising the honest fallback.
        # (The CLI refuses this combination; the runtime path must still be
        # deterministic because execute_tool_call can fail for other reasons.)
        disabled_tools=frozenset({"schedule"}),
    )
    record = result.forced_phi_reschedule
    assert record is not None
    assert record["outcome"] == "fallback_model_plan"
    assert record["fallback_reason"] == "tool_unavailable"
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == []  # the model's own plan survives


# --------------------------------------------------------------------------
# plan.md A0f-W1: --force-phi-scope=always
#
# The decision changed from "on-demand gating" to "deterministic call": the
# final plan is ALWAYS Phi-expanded. The frozen `direct_emit` scope only covers
# episodes that never called schedule, which on the 7B LOCUS main row is 52.1%
# of samples -- a further 17.9% called schedule mid-way and still ended on a
# plan that is not Phi's output (acc .0141 there vs .6667 where plan == Phi).
# So `direct_emit` is a LOWER BOUND on the A0f effect and `always` is the thing
# the decision actually specifies. These tests pin that the two scopes differ
# exactly on that subset, and that the subset is recoverable per-sample.
# --------------------------------------------------------------------------


def _scheduled_then_self_written_backend():
    """An episode that calls schedule, then ignores the solver's completed ops.

    This is the shape that `direct_emit` misses and `always` catches.
    """

    class Backend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return (
                    "[PLAN_SKETCH]\n" + json.dumps(_SKETCH)
                    + "\n[TOOL_CALL]\n"
                    + json.dumps(
                        {"name": "schedule", "args": {"op_ids": ["o1", "o2"], "assign": _ASSIGN}}
                    )
                )
            return "[PLAN]\n" + json.dumps(_MODEL_PLAN)

    return Backend()


def test_scope_always_forces_an_already_scheduled_episode() -> None:
    scene, ctx = _scene()
    result = run_agentic_tool_loop(
        _scheduled_then_self_written_backend(),
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
        force_phi_scope="always",
    )
    record = result.forced_phi_reschedule
    assert record is not None, "scope=always must force even a scheduled episode"
    assert record["outcome"] == "rescheduled"
    assert record["scope"] == "always"
    # The subset flag is what lets one `always` run be split back into the
    # frozen `direct_emit` reading, so it must be true here.
    assert record["schedule_executed_before"] is True
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == ["o1"], "the scored plan must be the solver's, not the model's"


def test_scope_direct_emit_is_the_frozen_semantics() -> None:
    """Same backend, default scope => untouched. Pins that the default did not
    move when `always` was added (the frozen point 4-AR bucket stays valid)."""
    scene, ctx = _scene()
    result = run_agentic_tool_loop(
        _scheduled_then_self_written_backend(),
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
    )
    assert result.forced_phi_reschedule is None
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == []


def test_scope_always_records_subset_flag_on_direct_emit_samples() -> None:
    """On a direct-emit sample both scopes force; the flag distinguishes it."""
    scene, ctx = _scene()
    both = []
    for scope in ("direct_emit", "always"):
        result = run_agentic_tool_loop(
            _OneTurnBackend(_direct_emit_turn()),
            scene,
            ctx,
            max_tool_rounds=2,
            force_phi_reschedule=True,
            force_phi_scope=scope,
        )
        record = result.forced_phi_reschedule
        assert record is not None
        assert record["schedule_executed_before"] is False
        assert record["scope"] == scope
        both.append([list(op.deps) for op in result.response.operations])
    # Never-called-schedule samples must be byte-identical under both scopes,
    # or the two readings are not nested and `direct_emit` is not a bound.
    assert both[0] == both[1]


# ---------------------------------------------------------------------------
# A0g② (2026-09-01): --force-phi-reject-feedback -- rejected forced schedule
# becomes a ToolReason feedback turn + extra repair rounds, instead of the
# frozen silent fallback. Default-off path must stay byte-identical.
# ---------------------------------------------------------------------------


class _ScriptedBackend:
    """Replays a fixed sequence of turns (repeats the last one)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.seen_messages.append([dict(m) for m in messages])
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return text


_COMPLETED_OPS = [
    {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
    {
        "op_id": "o2",
        "op_type": "deposit",
        "entity": "apple",
        "dest": "table",
        "robot": "R1",
        "deps": ["o1"],
    },
]


def test_reject_feedback_injects_tool_turn_then_repair(monkeypatch) -> None:
    """First forced call rejects -> feedback turn -> second forced call lands."""
    import nsdl.tool_runtime as tr

    payloads = iter(
        [
            {"ok": False, "code": "ee_capacity", "op_ids": ["o1"], "message": "cap"},
            {"ok": True, "ops": _COMPLETED_OPS, "schedule": []},
        ]
    )
    forced_calls: list[dict] = []

    def _fake_execute(call, *, context, plan_sketch=None):  # noqa: ANN001
        forced_calls.append(dict(call))
        return next(payloads)

    monkeypatch.setattr(tr, "execute_tool_call", _fake_execute)

    scene, ctx = _scene()
    backend = _ScriptedBackend([_direct_emit_turn(), _direct_emit_turn()])
    result = run_agentic_tool_loop(
        backend,
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
        force_phi_scope="always",
        force_phi_reject_feedback=True,
    )
    assert backend.calls == 2
    assert result.stopped_reason == "final_plan"
    # Round 0: rejection surfaced as feedback, not fallback.
    first = result.events[0]["forced_phi_reschedule"]
    assert first["outcome"] == "rejected_feedback_injected"
    assert first["reject_reason"] == "ee_capacity"
    assert first["feedback_rounds_before"] == 0
    # The rejection reached the model as a real tool turn.
    fed = backend.seen_messages[1]
    assert fed[-1]["role"] == "tool"
    assert "ee_capacity" in fed[-1]["content"]
    # Round 1: forced call succeeded, plan replaced by Phi's ops.
    final = result.forced_phi_reschedule
    assert final is not None
    assert final["outcome"] == "rescheduled"
    assert final["feedback_rounds_before"] == 1
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == ["o1"]
    # Forced calls stay OUT of the normal TC accounting.
    assert result.tool_results == []
    assert len(forced_calls) == 2 and all(c["name"] == "schedule" for c in forced_calls)


def test_reject_feedback_exhausts_rounds_then_fallback() -> None:
    """Rejection every round -> feedback until max_tool_rounds, then honest
    fallback flagged as fallback_after_feedback."""
    scene, ctx = _scene()
    backend = _ScriptedBackend([_direct_emit_turn()])
    result = run_agentic_tool_loop(
        backend,
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
        force_phi_scope="always",
        force_phi_reject_feedback=True,
        # Deterministic rejection: schedule disabled -> tool_unavailable.
        disabled_tools=frozenset({"schedule"}),
    )
    # rounds 0 and 1 inject feedback; round 2 (== max_tool_rounds) must not.
    assert backend.calls == 3
    outcomes = [
        e["forced_phi_reschedule"]["outcome"]
        for e in result.events
        if "forced_phi_reschedule" in e
    ]
    assert outcomes == [
        "rejected_feedback_injected",
        "rejected_feedback_injected",
        "fallback_model_plan",
    ]
    final = result.forced_phi_reschedule
    assert final is not None
    assert final["fallback_after_feedback"] is True
    assert final["feedback_rounds_before"] == 2
    deps = {op.op_id: list(op.deps) for op in result.response.operations}
    assert deps["o2"] == []  # the model's own plan survives


def test_reject_feedback_default_off_keeps_frozen_fallback() -> None:
    """Flag off: single turn, silent fallback, fallback_after_feedback False."""
    scene, ctx = _scene()
    backend = _ScriptedBackend([_direct_emit_turn()])
    result = run_agentic_tool_loop(
        backend,
        scene,
        ctx,
        max_tool_rounds=2,
        force_phi_reschedule=True,
        disabled_tools=frozenset({"schedule"}),
    )
    assert backend.calls == 1
    record = result.forced_phi_reschedule
    assert record is not None
    assert record["outcome"] == "fallback_model_plan"
    assert record["fallback_after_feedback"] is False
    assert record["feedback_rounds_before"] == 0


# ---------------------------------------------------------------------------
# A0k / facts:E24c (2026-09-04): the forced arm's answer is Phi's schedule, not
# a re-expansion of Phi's op-DAG. Fixture = real 7B test_id row 00006_F1_s0m1
# (fetch, ee=1, two objects): Phi's schedule is byte-equal to GT, while the
# pre-fix re-expansion grasped both objects before placing and scored 0.
# ---------------------------------------------------------------------------

_FIX = Path(__file__).parent / "fixtures" / "forced_phi_f1_00006.json"


def _fixture() -> dict:
    return json.loads(_FIX.read_text())


def test_forced_phi_micro_steps_returns_phi_schedule():
    from nsdl.tool_runtime import forced_phi_micro_steps

    fx = _fixture()
    steps = forced_phi_micro_steps(fx["forced_phi_reschedule"])
    assert steps is not None and len(steps) == 10
    got = [{"step": s.step, "actions": s.actions} for s in steps]
    assert got == fx["forced_phi_reschedule"]["schedule_result"]["schedule"]
    # Not applied / not rescheduled / no schedule -> None (caller keeps its plan).
    assert forced_phi_micro_steps(None) is None
    assert forced_phi_micro_steps({"applied": False}) is None
    rec = dict(fx["forced_phi_reschedule"], outcome="fallback_model_plan")
    assert forced_phi_micro_steps(rec) is None
    rec = json.loads(json.dumps(fx["forced_phi_reschedule"]))
    rec["schedule_result"]["schedule"] = []
    assert forced_phi_micro_steps(rec) is None


def test_phi_schedule_scores_one_where_reexpansion_scored_zero():
    from nsdl.benchmark_scoring import compute_locus_l2_acc
    from nsdl.serializer import to_l2_answer
    from nsdl.tool_runtime import forced_phi_micro_steps
    from nsdl.validator import validate_l2_plan

    fx = _fixture()
    gt = fx["ground_truth"]
    # The frozen (pre-fix) answer: two Grasps before any Place on an ee=1 robot.
    assert compute_locus_l2_acc(fx["stored_plan"], gt) == 0.0
    assert not validate_l2_plan(fx["stored_plan"], gt).success
    plan = to_l2_answer(forced_phi_micro_steps(fx["forced_phi_reschedule"]))
    assert validate_l2_plan(plan, gt).success
    assert compute_locus_l2_acc(plan, gt) == 1.0


def test_offline_rescore_replays_the_fix(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rescore_forced", REPO / "scripts" / "rescore_forced_phi_bucket_offline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fx = _fixture()
    bucket = tmp_path / "bucket"
    (bucket / "test_id").mkdir(parents=True)
    row = {
        "sample": {"sample_index": 6},
        "plan": fx["stored_plan"],
        "ground_truth": fx["ground_truth"],
        "result_status": "ok",
        "validation": {"success": False, "error_desc": "", "eval_commands": []},
        "benchmark_scores": {"locus_l2_acc": 0.0, "viki_l2_plan_score": 0.1},
        "agentic_eval": {"trace": {"forced_phi_reschedule": fx["forced_phi_reschedule"]}},
    }
    (bucket / "test_id" / "00006_F1_s0m1.json").write_text(json.dumps(row))
    # An untouched row (forcing not applied) must be kept byte-for-byte.
    row2 = dict(row, benchmark_scores={"locus_l2_acc": 1.0, "viki_l2_plan_score": 1.0},
                agentic_eval={"trace": {"forced_phi_reschedule": None}})
    (bucket / "test_id" / "00007_F1_s0m1.json").write_text(json.dumps(row2))

    report = mod.rescore_bucket(str(bucket), mod.LEVELS["locus"], write_derived=True)
    counters = report["counters"]
    assert {k: counters[k] for k in ("plan_replaced", "plan_replaced_score_changed", "untouched_kept_stored")} == {
        "plan_replaced": 1, "plan_replaced_score_changed": 1, "untouched_kept_stored": 1}
    assert counters["source:forced_record"] == 1     # 序 26: source tag per replaced row
    ps = report["per_split"]["test_id"]
    assert ps["locus_l2_acc_stored"] == 0.5 and ps["locus_l2_acc_rescored"] == 1.0
    derived = json.loads((bucket.parent / "bucket__phisched" / "test_id" / "00006_F1_s0m1.json").read_text())
    assert derived["benchmark_scores"]["locus_l2_acc"] == 1.0
    assert derived["validation"]["success"] is True
    assert derived["agentic_eval"]["trace"]["forced_phi_reschedule"]["final_plan_source"] == "phi_schedule"
    assert derived["agentic_eval"]["force_phi_final_plan"] == "phi_schedule"
    untouched = (bucket.parent / "bucket__phisched" / "test_id" / "00007_F1_s0m1.json").read_text()
    assert untouched == json.dumps(row2)
