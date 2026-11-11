"""P3 agentic I/O schema and tool-runtime checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.schema import RobotInstance, SceneFacts  # noqa: E402
from nsdl.slm_interface import (  # noqa: E402
    build_slm_system_prompt,
    parse_slm_response,
    serialize_slm_input,
)
from nsdl.tool_runtime import (  # noqa: E402
    build_initial_messages,
    execute_response_tool_calls,
    execute_tool_call,
    is_loss_masked_message,
    render_tool_result,
    run_agentic_tool_loop,
    tool_result_message,
)
from nsdl.planning_tool import verify_goal_coverage  # noqa: E402


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


def _schedule_call() -> dict:
    ops = [
        {
            "op_id": "o1",
            "op_type": "acquire",
            "entity": "apple",
            "robot": "R1",
            "gaps": ["deps"],
        },
        {
            "op_id": "o2",
            "op_type": "deposit",
            "entity": "apple",
            "dest": "table",
            "robot": "R1",
            "gaps": ["deps"],
        },
    ]
    return {
        "name": "schedule",
        "args": {
            "ops": ops,
            "assign": {"R1": {"items": ["apple"]}},
        },
    }


def test_serialize_slm_input_is_formal_lean_runtime_prompt() -> None:
    scene, ctx = _scene()
    scene.extras["scene_relations"] = ["apple on table"]

    prompt = serialize_slm_input(scene, ctx)

    assert "[SCENE]" in prompt
    assert "Instruction: bring the apple to the table" in prompt
    assert "Robots: R1=fetch" in prompt
    assert "[ROBOTS]" in prompt
    assert "R1 (fetch):" in prompt
    assert "Observed entities:" not in prompt
    assert "Entity locations:" not in prompt
    assert "[ENTITIES]" not in prompt
    assert "[PLAN]" not in prompt


def test_system_prompt_uses_agentic_tool_protocol_without_removed_blocks() -> None:
    system_prompt = build_slm_system_prompt()

    assert "## Tool Specs" in system_prompt
    assert "[PLAN_SKETCH]" in system_prompt
    assert "[TOOL_CALL]" in system_prompt
    assert "schedule" in system_prompt
    assert "detect" in system_prompt
    assert "lookup" not in system_prompt
    assert "[COORD]" not in system_prompt
    assert "Observed entities:" not in system_prompt
    assert "Entity locations:" not in system_prompt
    assert "[ENTITIES]" not in system_prompt
    assert "fast/slow" not in system_prompt


def test_parse_agentic_response_with_tool_call_and_no_final_plan() -> None:
    raw = """
<think>Need geometry, then deterministic scheduling.</think>
[GOAL]
{"targets":[{"item":"apple","destination":"table","status":"active","status_reason":"needs_relocation","status_line":"apple: active"}],"active":["apple"],"skip":[]}
[ASSIGN]
{"R1":{"items":["apple"]}}
[PLAN_SKETCH]
[{"op_id":"o1","op_type":"acquire","entity":"apple","robot":"R1","<gap>":"deps"}]
[TOOL_CALL]
{"name":"schedule","args":{"ops":[{"op_id":"o1","op_type":"acquire","entity":"apple","robot":"R1"}],"assign":{"R1":{"items":["apple"]}}}}
"""

    response = parse_slm_response(raw)

    assert response.think == ["Need geometry, then deterministic scheduling."]
    assert response.goal is not None
    assert response.assign == {"R1": {"items": ["apple"]}}
    assert len(response.plan_sketch) == 1
    assert response.plan_sketch[0].gaps == ["deps"]
    assert response.tool_calls == [
        {
            "name": "schedule",
            "args": {
                "ops": [
                    {
                        "op_id": "o1",
                        "op_type": "acquire",
                        "entity": "apple",
                        "robot": "R1",
                    }
                ],
                "assign": {"R1": {"items": ["apple"]}},
            },
        }
    ]
    assert response.operations == []


def test_parse_tool_call_with_trailing_bracket_keeps_diagnostic() -> None:
    raw = (
        "[TOOL_CALL]\n"
        '{"name":"lookup","args":{"tag":"fruit"}}]'
    )

    response = parse_slm_response(raw)

    assert response.tool_calls == [{"name": "lookup", "args": {"tag": "fruit"}}]
    assert response.parse_diagnostics
    assert response.parse_diagnostics[0]["severity"] == "warning"
    assert response.parse_diagnostics[0]["kind"] == "json_trailing_tokens_recovered"


def test_parse_legacy_three_block_and_bare_list_outputs() -> None:
    plan = [
        {"op_id": "op1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {
            "op_id": "op2",
            "op_type": "deposit",
            "entity": "apple",
            "dest": "table",
            "deps": ["op1"],
            "robot": "R1",
        },
    ]
    old_three_block = (
        "[GOAL]\n"
        + json.dumps({"targets": [], "active": [], "skip": []})
        + "\n[ASSIGN]\n"
        + json.dumps({"R1": {"items": ["apple"]}})
        + "\n[PLAN]\n"
        + json.dumps(plan)
    )

    parsed = parse_slm_response(old_three_block)
    assert len(parsed.operations) == 2
    assert parsed.plan_sketch == []
    assert parsed.tool_calls == []

    bare = parse_slm_response(json.dumps(plan))
    assert len(bare.operations) == 2
    assert bare.goal is None
    assert bare.assign is None


def test_tool_runtime_schedule_and_rendering_are_stable() -> None:
    scene, ctx = _scene()
    call = _schedule_call()

    result = execute_tool_call(call, scene_facts=scene, affordance_context=ctx)
    assert result["ok"] is True
    assert [op["op_type"] for op in result["ops"]] == ["acquire", "deposit"]

    rendered = render_tool_result(result)
    assert rendered == render_tool_result(result)

    message = tool_result_message("schedule", result)
    assert message["role"] == "tool"
    assert message["name"] == "schedule"
    assert message["content"] == rendered
    assert is_loss_masked_message(message)

    messages = build_initial_messages(scene, ctx)
    assert messages == build_initial_messages(scene, ctx)


def test_schedule_op_ids_contract_resolves_against_plan_sketch() -> None:
    """B1: schedule args reference sketch ops by id; runtime backfills ops."""
    from nsdl.slm_interface import parse_operation_steps

    scene, ctx = _scene()
    plan_sketch = parse_operation_steps(_schedule_call()["args"]["ops"])
    op_ids_call = {
        "name": "schedule",
        "args": {"op_ids": ["o1", "o2"], "assign": {"R1": {"items": ["apple"]}}},
    }

    result = execute_tool_call(
        op_ids_call, scene_facts=scene, affordance_context=ctx, plan_sketch=plan_sketch
    )
    assert result["ok"] is True
    assert [op["op_type"] for op in result["ops"]] == ["acquire", "deposit"]

    # Legacy ops form still works (backward compat with pre-B1 checkpoints).
    legacy = execute_tool_call(_schedule_call(), scene_facts=scene, affordance_context=ctx)
    assert legacy["ok"] is True


def test_schedule_op_ids_without_sketch_fails_closed() -> None:
    """B1 fallback: op_ids with no in-turn PLAN_SKETCH -> structured error, not a
    silent empty schedule."""
    scene, ctx = _scene()
    op_ids_call = {"name": "schedule", "args": {"op_ids": ["o1"], "assign": {}}}

    result = execute_tool_call(op_ids_call, scene_facts=scene, affordance_context=ctx)
    assert result["ok"] is False
    assert result["code"] == "missing_plan_sketch"


def test_execute_response_tool_calls_uses_same_runtime_path() -> None:
    scene, ctx = _scene()
    raw = "[TOOL_CALL]\n" + json.dumps(_schedule_call())

    results = execute_response_tool_calls(raw, scene_facts=scene, affordance_context=ctx)

    assert len(results) == 1
    assert results[0]["ok"] is True

    lookup_result = execute_tool_call({"name": "lookup", "args": {"tag": "fruit"}})
    assert "apple" in lookup_result["fruit"]


def test_verify_goal_coverage_diagnoses_model_goal_against_observations_only() -> None:
    observed = {"relations": ["plate on table", "bowl near table"]}
    bad_goal = {
        "targets": [
            {
                "item": "plate",
                "destination": "table",
                "status": "active",
                "status_reason": "needs_relocation",
                "status_line": "plate: active",
            }
        ],
        "active": ["plate"],
        "skip": [],
    }
    ok_goal = {
        "targets": [
            {
                "item": "plate",
                "destination": "table",
                "status": "skip",
                "status_reason": "observed_on_table",
                "status_line": "plate: skip(observed_on_table)",
            }
        ],
        "active": [],
        "skip": ["plate"],
    }

    conflict = verify_goal_coverage(bad_goal, observed)
    assert conflict.ok is False
    assert conflict.code == "goal_coverage_conflict"
    assert "already_at_destination" not in conflict.message

    direct_ok = verify_goal_coverage(ok_goal, observed)
    routed_ok = execute_tool_call({
        "name": "verify_goal_coverage",
        "args": {"model_goal": ok_goal, "observed_facts": observed},
    })
    assert direct_ok.ok is True
    assert routed_ok["ok"] is True


def test_run_agentic_tool_loop_appends_tool_turn_then_final_plan() -> None:
    scene, ctx = _scene()
    final_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {
            "op_id": "o2",
            "op_type": "deposit",
            "entity": "apple",
            "dest": "table",
            "deps": ["o1"],
            "robot": "R1",
        },
    ]

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return "[TOOL_CALL]\n" + json.dumps(_schedule_call())
            assert messages[-1]["role"] == "tool"
            return "[PLAN]\n" + json.dumps(final_plan)

    backend = FakeBackend()
    result = run_agentic_tool_loop(
        backend,
        scene,
        ctx,
        max_tool_rounds=2,
        generation_kwargs={"temperature": 0.0},
    )

    assert backend.calls == 2
    assert result.stopped_reason == "final_plan"
    assert len(result.response.operations) == 2
    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert is_loss_masked_message(result.messages[3])
    assert result.tool_results[0]["ok"] is True
    assert result.events[0]["tool_calls"] == [_schedule_call()]
    assert result.events[0]["tool_results"][0]["ok"] is True
    assert result.events[-1]["parsed"]["plan"][0]["op_type"] == "acquire"


def test_run_agentic_tool_loop_op_ids_resolve_against_prior_turn_sketch() -> None:
    """F1: sketch emitted in an earlier turn; a later-turn schedule referencing
    its op_ids must resolve via look-back, not fail with missing_plan_sketch."""
    final_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {
            "op_id": "o2",
            "op_type": "deposit",
            "entity": "apple",
            "dest": "table",
            "deps": ["o1"],
            "robot": "R1",
        },
    ]
    sketch_body = json.dumps(
        [
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
    )
    schedule_by_ids = {
        "name": "schedule",
        "args": {"op_ids": ["o1", "o2"], "assign": {"R1": {"items": ["apple"]}}},
    }
    scene, ctx = _scene()

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                # Turn 1: emit the sketch + a detect call (no schedule yet).
                return (
                    "[PLAN_SKETCH]\n" + sketch_body + "\n"
                    "[TOOL_CALL]\n" + json.dumps({"name": "detect", "args": {"query": "apple"}})
                )
            if self.calls == 2:
                # Turn 2: schedule referencing op_ids, but NO sketch this turn.
                return "[TOOL_CALL]\n" + json.dumps(schedule_by_ids)
            return "[PLAN]\n" + json.dumps(final_plan)

    backend = FakeBackend()
    result = run_agentic_tool_loop(
        backend, scene, ctx, max_tool_rounds=3, generation_kwargs={"temperature": 0.0}
    )

    schedule_event = result.events[1]
    schedule_result = schedule_event["tool_results"][0]
    assert schedule_event["plan_sketch_source"] == "carried"
    assert schedule_result["ok"] is True, schedule_result
    assert [op["op_type"] for op in schedule_result["ops"]] == ["acquire", "deposit"]
    assert result.stopped_reason == "final_plan"


def test_run_agentic_tool_loop_stops_on_second_tool_call_in_one_turn() -> None:
    scene, ctx = _scene()
    final_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {
            "op_id": "o2",
            "op_type": "deposit",
            "entity": "apple",
            "dest": "table",
            "deps": ["o1"],
            "robot": "R1",
        },
    ]

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return "\n".join([
                    "[TOOL_CALL]",
                    json.dumps({"name": "lookup", "args": {"tag": "fruit"}}),
                    "[TOOL_CALL]",
                    json.dumps(_schedule_call()) + "]",
                ])
            assert messages[-1]["role"] == "tool"
            assert messages[-1]["name"] == "lookup"
            return "[PLAN]\n" + json.dumps(final_plan)

    backend = FakeBackend()
    result = run_agentic_tool_loop(
        backend,
        scene,
        ctx,
        max_tool_rounds=2,
        generation_kwargs={"temperature": 0.0},
    )

    assert backend.calls == 2
    assert [message["role"] for message in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.messages[3]["name"] == "lookup"
    assert result.events[0]["executable_tool_calls"] == [{"name": "lookup", "args": {"tag": "fruit"}}]
    assert result.events[0]["suppressed_tool_calls"] == [_schedule_call()]
    assert [event["name"] for event in result.events[0]["tool_messages"]] == ["lookup"]
    assert len(result.events[0]["parsed"]["parse_diagnostics"]) == 1
    assert result.events[0]["parsed"]["parse_diagnostics"][0]["severity"] == "warning"
    assert result.tool_results[0]["fruit"]
    assert len(result.tool_results) == 1


def _stacked_schedule_then_plan(schedule_call, stacked_plan):
    """One turn stacking [TOOL_CALL](schedule) + [PLAN] (cluster-10 shape)."""
    return "\n".join([
        "<think>judging targets</think>",
        "[GOAL]", json.dumps({"active": ["apple"], "skip": [], "targets": [
            {"item": "apple", "destination": "table", "status": "active"}]}),
        "[ASSIGN]", json.dumps({"R1": {"items": ["apple"]}}),
        "[PLAN_SKETCH]", json.dumps([
            {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1", "gaps": ["deps"]},
            {"op_id": "o2", "op_type": "deposit", "entity": "apple", "dest": "table", "robot": "R1", "gaps": ["deps"]},
        ]),
        "[TOOL_CALL]", json.dumps(schedule_call),
        "[PLAN]", json.dumps(stacked_plan),
    ])


def test_stacked_turn_uses_own_unrepaired_plan_without_truncation() -> None:
    """Cluster 10 baseline: when the model stacks [TOOL_CALL]+[PLAN] in one turn,
    the loop stops on the model's own PLAN and never runs the schedule call."""
    scene, ctx = _scene()
    # A deliberately deps-less plan -> what the stacked model emits (no repair).
    stacked_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {"op_id": "o2", "op_type": "deposit", "entity": "apple", "dest": "table", "robot": "R1"},
    ]
    schedule_call = {"name": "schedule", "args": {"op_ids": ["o1", "o2"], "assign": {"R1": {"items": ["apple"]}}}}

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            return _stacked_schedule_then_plan(schedule_call, stacked_plan)

    result = run_agentic_tool_loop(
        FakeBackend(), scene, ctx, max_tool_rounds=3, generation_kwargs={"temperature": 0.0}
    )
    assert result.stopped_reason == "final_plan"
    # schedule never executed; the deps-less stacked plan is used verbatim.
    assert result.tool_results == []
    assert result.response.operations[1].deps == []
    assert result.events[0].get("truncated_stacked_tail") in (None, False)


def test_truncation_drops_stacked_plan_and_runs_schedule() -> None:
    """Cluster 10 fix: with truncate_to_first_tool_call the stacked [PLAN] is
    dropped, the schedule executes (deps-repaired), and the next round emits it."""
    scene, ctx = _scene()
    stacked_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {"op_id": "o2", "op_type": "deposit", "entity": "apple", "dest": "table", "robot": "R1"},
    ]
    repaired_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"},
        {"op_id": "o2", "op_type": "deposit", "entity": "apple", "dest": "table", "deps": ["o1"], "robot": "R1"},
    ]
    schedule_call = {"name": "schedule", "args": {"op_ids": ["o1", "o2"], "assign": {"R1": {"items": ["apple"]}}}}

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return _stacked_schedule_then_plan(schedule_call, stacked_plan)
            assert messages[-1]["role"] == "tool"  # schedule executed this round
            return "[PLAN]\n" + json.dumps(repaired_plan)

    backend = FakeBackend()
    result = run_agentic_tool_loop(
        backend, scene, ctx, max_tool_rounds=3, truncate_to_first_tool_call=True,
        generation_kwargs={"temperature": 0.0},
    )
    assert backend.calls == 2
    assert result.stopped_reason == "final_plan"
    assert result.events[0]["truncated_stacked_tail"] is True
    assert "[PLAN]" not in result.events[0]["assistant_content"]
    assert len(result.tool_results) == 1 and result.tool_results[0]["ok"] is True
    # final plan now carries the deps-repaired custody chain.
    assert result.response.operations[1].deps == ["o1"]


def test_truncation_leaves_clean_single_call_turn_unchanged() -> None:
    """A well-formed single-tool-call turn must be untouched by the guard."""
    scene, ctx = _scene()
    final_plan = [{"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"}]

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return "[TOOL_CALL]\n" + json.dumps(_schedule_call())
            return "[PLAN]\n" + json.dumps(final_plan)

    result = run_agentic_tool_loop(
        FakeBackend(), scene, ctx, max_tool_rounds=2, truncate_to_first_tool_call=True,
        generation_kwargs={"temperature": 0.0},
    )
    assert result.events[0]["truncated_stacked_tail"] is False
    assert len(result.tool_results) == 1 and result.tool_results[0]["ok"] is True


def test_parse_tool_call_recovers_concatenated_json_objects() -> None:
    raw = (
        "[TOOL_CALL]\n"
        '{"name":"lookup","args":{"tag":"fruit"}}'
        '{"name":"schedule","args":{"ops":[],"assign":{}}}'
    )

    response = parse_slm_response(raw)

    assert [call["name"] for call in response.tool_calls] == ["lookup", "schedule"]
    assert response.parse_diagnostics
    assert response.parse_diagnostics[0]["kind"] == "json_concatenated_values_recovered"


# ---------------------------------------------------------- P5.10.1 crops ----

def _crop_result_with_images():
    from PIL import Image

    img1 = Image.new("RGB", (8, 8), (1, 2, 3))
    img2 = Image.new("RGB", (8, 8), (4, 5, 6))
    result = {
        "entities": ["plate"],
        "relations": [],
        "confidences": {"plate": 0.9},
        "crops": [{"entity": "plate", "confidence": 0.9}],
        "_crop_images": [
            {"entity": "plate", "image": img1},
            {"entity": "bowl", "image": img2},
        ],
    }
    return result, img1, img2


def test_tool_result_message_with_crops_renders_ordered_parts() -> None:
    result, img1, img2 = _crop_result_with_images()
    msg = tool_result_message("detect", result)

    assert msg["role"] == "tool"
    assert is_loss_masked_message(msg)  # loss mask unaffected by parts content
    assert isinstance(msg["content"], list)
    assert [part["type"] for part in msg["content"]] == ["text", "image", "image"]
    # Image order preserved; the exact PIL objects are threaded through.
    assert msg["content"][1]["image"] is img1
    assert msg["content"][2]["image"] is img2
    assert msg["content"][1]["entity"] == "plate"
    # Text part carries meta only — never the out-of-band pixel key.
    text = msg["content"][0]["text"]
    assert "_crop_images" not in text
    assert "\"crops\"" in text


def test_tool_result_message_without_crops_is_plain_string() -> None:
    """Regression: no crops → content is the exact prior string form."""
    result = {"entities": ["plate"], "relations": [], "confidences": {}}
    msg = tool_result_message("detect", result)
    assert isinstance(msg["content"], str)
    assert msg["content"] == render_tool_result(result)


def test_jsonable_and_render_strip_crop_pixels() -> None:
    from PIL import Image
    from nsdl.tool_runtime import _jsonable

    payload = {
        "entities": ["plate"],
        "_crop_images": [{"image": Image.new("RGB", (4, 4))}],
        "nested": {"_hidden": 1, "keep": 2},
    }
    assert _jsonable(payload) == {"entities": ["plate"], "nested": {"keep": 2}}
    assert _jsonable(Image.new("RGB", (4, 4))) == "<image>"
    assert "_crop_images" not in render_tool_result(payload)


def test_local_vl_build_messages_threads_crop_images_in_order() -> None:
    from nsdl.slm_backends.local_vl import LocalVisionTextSLMBackend

    scene_image = object()
    crop1 = object()
    crop2 = object()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "scene <image>"},
        {"role": "assistant", "content": "[TOOL_CALL]\n{}"},
        {
            "role": "tool",
            "name": "detect",
            "content": [
                {"type": "text", "text": "{\"entities\":[\"plate\"]}"},
                {"type": "image", "image": crop1},
                {"type": "image", "image": crop2},
            ],
        },
    ]
    vl_messages, images = LocalVisionTextSLMBackend._build_messages(messages, scene_image)

    # Scene image first, then the two crops in message order.
    assert images == [scene_image, crop1, crop2]
    # Scene image attached to the first user turn.
    user_parts = vl_messages[1]["content"]
    assert any(p["type"] == "image" and p["image"] is scene_image for p in user_parts)
    # Tool turn rendered as parts carrying both crop images in order.
    tool_parts = vl_messages[-1]["content"]
    img_parts = [p["image"] for p in tool_parts if p["type"] == "image"]
    assert img_parts == [crop1, crop2]
    assert any(p["type"] == "text" and "TOOL_RESULT detect" in p["text"] for p in tool_parts)


def test_local_vl_build_messages_no_crops_matches_prior_shape() -> None:
    from nsdl.slm_backends.local_vl import LocalVisionTextSLMBackend

    scene_image = object()
    messages = [
        {"role": "user", "content": "scene <image>"},
        {"role": "tool", "name": "detect", "content": "{\"entities\":[]}"},
    ]
    vl_messages, images = LocalVisionTextSLMBackend._build_messages(messages, scene_image)
    assert images == [scene_image]  # single scene image, as before
    # String tool content stays a string turn.
    assert isinstance(vl_messages[-1]["content"], str)
    assert "TOOL_RESULT detect" in vl_messages[-1]["content"]


def test_local_text_backend_degrades_crops_to_placeholder() -> None:
    from nsdl.slm_backends.local_text import _template_message

    rendered = _template_message(
        {
            "role": "tool",
            "name": "detect",
            "content": [
                {"type": "text", "text": "obs"},
                {"type": "image", "image": object(), "entity": "plate"},
            ],
        }
    )
    assert rendered["role"] == "user"
    assert isinstance(rendered["content"], str)
    assert "[TOOL_RESULT detect]" in rendered["content"]
    assert "[CROP plate attached]" in rendered["content"]


def test_run_agentic_tool_loop_detect_crop_attaches_image_parts() -> None:
    """Integration: a detect call with crop_entities re-injects an image tool turn."""
    import io
    import types

    from PIL import Image

    scene, ctx = _scene()
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (120, 120, 120)).save(buf, "PNG")
    image_bytes = buf.getvalue()
    payload = {
        "grouped_detections": [
            {
                "label": "plate",
                "best_confidence": 90,
                "instances": [{"confidence": 90, "bbox_xyxy": [100, 100, 140, 140]}],
            }
        ]
    }

    class FakeDetector:
        def detect(self, _b):  # noqa: ANN001
            return types.SimpleNamespace(prompt_payload=payload)

    detect_call = {
        "name": "detect",
        "args": {"query": {"entities": ["plate"]}, "crop_entities": ["plate"]},
    }
    final_plan = [
        {"op_id": "o1", "op_type": "acquire", "entity": "plate", "robot": "R1"},
        {"op_id": "o2", "op_type": "deposit", "entity": "plate", "dest": "table",
         "deps": ["o1"], "robot": "R1"},
    ]

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return "[TOOL_CALL]\n" + json.dumps(detect_call)
            # Turn 2: the detect tool turn must carry an image part.
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs, "no tool turn appended"
            content = tool_msgs[-1]["content"]
            assert isinstance(content, list)
            assert any(p.get("type") == "image" for p in content)
            return "[PLAN]\n" + json.dumps(final_plan)

    backend = FakeBackend()
    result = run_agentic_tool_loop(
        backend, scene, ctx, image=image_bytes, detector=FakeDetector(), max_tool_rounds=2,
    )
    assert backend.calls == 2
    assert result.stopped_reason == "final_plan"
    # Trace stays JSON-safe: pixels rendered as a stable placeholder, no raw PIL.
    detect_event = result.events[0]
    tool_msg = detect_event["tool_messages"][0]
    image_parts = [p for p in tool_msg["content"] if p.get("type") == "image"]
    assert image_parts and image_parts[0]["image"] == "<image>"


# --- P10.S2 y-axis: the Phi-off arm ----------------------------------------
#
# The second arm of the S2 grid keeps the same trained front-end and the same
# perception channel but makes the deterministic backend Phi unavailable, so the
# model must produce the mechanical structure itself. `gain(M) = phi_on -
# phi_off` per capacity point is the main-text y-axis. Design discipline 13 says
# the two arms share one runtime and differ only in gating, which is exactly what
# `disabled_tools` is -- these tests pin that gate.


def test_phi_tools_is_exactly_the_three_deterministic_queries() -> None:
    """Phi is the constraint model's three queries -- not perception, not diagnostics.

    docs/tools.md §0.3 defines Phi as one constraint model with three queries.
    `detect` is facet-A perception and `verify_goal_coverage` is a read-only
    diagnostic of the model's own [GOAL]; folding either into the ablation would
    make the y-axis measure something other than the Phi offload.
    """
    from nsdl.tool_runtime import PHI_TOOLS

    assert PHI_TOOLS == frozenset({"schedule", "verify", "expand"})
    assert "detect" not in PHI_TOOLS
    assert "verify_goal_coverage" not in PHI_TOOLS


def test_disabled_tool_returns_tool_unavailable_without_executing() -> None:
    from nsdl.tool_runtime import PHI_TOOLS, ToolRuntimeContext

    scene, aff = _scene()
    ctx = ToolRuntimeContext(
        scene_facts=scene, affordance_context=aff, disabled_tools=PHI_TOOLS
    )

    result = execute_tool_call(_schedule_call(), context=ctx)

    assert result["ok"] is False
    assert result["code"] == "tool_unavailable"
    # A real schedule() would have returned a `schedule` key; a gated call must
    # not leak any solver output.
    assert "schedule" not in result


def test_disabled_tools_default_empty_leaves_schedule_untouched() -> None:
    """Default construction must be byte-identical to the pre-S2 behaviour.

    Training and RL rollout share this runtime; if the gate defaulted to
    anything but empty, the arm would silently contaminate them.
    """
    from nsdl.tool_runtime import ToolRuntimeContext

    scene, aff = _scene()
    ctx = ToolRuntimeContext(scene_facts=scene, affordance_context=aff)

    assert ctx.disabled_tools == frozenset()
    result = execute_tool_call(_schedule_call(), context=ctx)
    assert result["ok"] is True
    assert len(result["ops"]) == 2


def test_phi_off_keeps_detect_and_goal_coverage_live() -> None:
    """Perception and the [GOAL] diagnostic survive the Phi ablation."""
    from nsdl.perc_tool import DetectResult
    from nsdl.tool_runtime import PHI_TOOLS, ToolRuntimeContext

    scene, aff = _scene()
    ctx = ToolRuntimeContext(
        scene_facts=scene,
        affordance_context=aff,
        disabled_tools=PHI_TOOLS,
        detect_override=lambda q=None, c=None: DetectResult(),
    )

    detected = execute_tool_call({"name": "detect", "args": {"query": ["apple"]}}, context=ctx)
    assert detected.get("code") != "tool_unavailable"

    covered = execute_tool_call(
        {"name": "verify_goal_coverage", "args": {"model_goal": {}, "observed_facts": {}}},
        context=ctx,
    )
    assert covered.get("code") != "tool_unavailable"


def test_run_agentic_tool_loop_phi_off_yields_tool_error_turn_then_plan() -> None:
    """End-to-end: the loop still appends a (failed) tool turn and keeps going.

    The model has to see a normal tool-error turn -- not a truncated episode --
    otherwise phi_off would measure "no second turn" rather than "no Phi".
    """
    from nsdl.tool_runtime import PHI_TOOLS

    scene, ctx = _scene()
    final_plan = [{"op_id": "o1", "op_type": "acquire", "entity": "apple", "robot": "R1"}]

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls += 1
            if self.calls == 1:
                return "[TOOL_CALL]\n" + json.dumps(_schedule_call())
            assert messages[-1]["role"] == "tool"
            return "[PLAN]\n" + json.dumps(final_plan)

    backend = FakeBackend()
    result = run_agentic_tool_loop(
        backend, scene, ctx, max_tool_rounds=2, disabled_tools=PHI_TOOLS,
    )

    assert backend.calls == 2
    assert result.stopped_reason == "final_plan"
    assert result.tool_results[0]["ok"] is False
    assert result.tool_results[0]["code"] == "tool_unavailable"
    assert [m["role"] for m in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert len(result.response.operations) == 1


def test_run_agentic_tool_loop_accepts_a_plain_sequence_of_disabled_tools() -> None:
    """The launcher passes a list; it must be coerced, not silently ignored."""
    scene, ctx = _scene()

    class FakeBackend:
        def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
            return "[TOOL_CALL]\n" + json.dumps(_schedule_call())

    result = run_agentic_tool_loop(
        FakeBackend(), scene, ctx, max_tool_rounds=1,
        disabled_tools=["schedule", "verify", "expand"],
    )
    assert result.tool_results[0]["code"] == "tool_unavailable"
