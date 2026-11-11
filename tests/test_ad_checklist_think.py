"""P5.10.6 v3.1 (a)+(d) think builders: branch coverage + non-committal audit."""
from __future__ import annotations

from types import SimpleNamespace

from nsdl.ad_checklist_think import (
    build_checklist_grounding_think,
    build_noncommittal_pre_think,
)
from nsdl.teacher_think_audit import audit_noncommittal_pre_think, audit_think


def make_goal(targets: list[tuple[str, str | None, str]]) -> dict:
    return {
        "active": [item for item, _, status in targets if status == "active"],
        "skip": [item for item, _, status in targets if status == "skip"],
        "targets": [
            {
                "item": item,
                "destination": dest,
                "status": status,
                "status_line": f"{item}: {status}",
                "status_reason": "needs_relocation" if status == "active" else "already_at_destination",
            }
            for item, dest, status in targets
        ],
    }


# ---------- (a) non-committal pre think ----------

def test_pre_think_poses_questions_and_never_commits() -> None:
    goal = make_goal([("bowl", "table", "active"), ("plate", "table", "skip")])
    for row_index in range(6):  # every template variant
        text = build_noncommittal_pre_think(goal, crop_entities=["bowl", "plate"], row_index=row_index)
        assert text is not None
        assert "Is the bowl currently at the table?" in text
        assert "Is the plate currently at the table?" in text
        assert "crops of: bowl, plate" in text
        assert audit_noncommittal_pre_think(text) == []


def test_pre_think_handles_missing_destination_and_empty_targets() -> None:
    goal = make_goal([("box", None, "active")])
    text = build_noncommittal_pre_think(goal, row_index=1)
    assert text is not None and "about the box?" in text
    assert audit_noncommittal_pre_think(text) == []
    assert build_noncommittal_pre_think({"targets": []}) is None


def test_noncommittal_audit_catches_verdict_statements() -> None:
    assert audit_noncommittal_pre_think("The plate is already on the table.")
    assert audit_noncommittal_pre_think("Both placements remain active.")
    assert audit_noncommittal_pre_think("The bowl is missing, so it needs to be brought over.")
    assert audit_noncommittal_pre_think("Nothing needs relocation here.")
    assert audit_noncommittal_pre_think("")
    # Questions are exempt.
    assert audit_noncommittal_pre_think("Is the plate already on the table?") == []


# ---------- (d) evidence checklist grounding think ----------

def test_checklist_skip_with_relation_and_active_absence() -> None:
    goal = make_goal([("plate", "table", "skip"), ("bowl", "table", "active")])
    detect = {
        "entities": ["plate", "table", "bowl"],
        "relations": ["plate on table"],
        "confidences": {"plate": 0.98},
        "crops": [{"entity": "plate", "confidence": 0.98}],
    }
    text, audit = build_checklist_grounding_think(goal, detect)
    assert text is not None
    assert "'plate on table' shows the plate already at the table -> skip." in text
    assert "the bowl is detected but NO relation puts it at the table" in text
    assert audit["evidence_relation"] == ["plate"]
    assert audit["evidence_absence"] == ["bowl"]
    assert audit["verdict_active"] == ["bowl"] and audit["verdict_skip"] == ["plate"]
    # verdict summary must state both sets for the think<->GOAL consistency audit
    assert "active: bowl; skip: plate." in text
    assert audit_think(SimpleNamespace(
        pre_tool="x", grounding_tool=text, coverage_repair_tool="", repair_tool="", post_tool="",
    ), "some_task", goal=goal) == []


def test_checklist_elsewhere_relation_and_not_detected() -> None:
    goal = make_goal([("peach", "table", "active"), ("banana", "table", "active")])
    detect = {"entities": ["peach", "cabinet", "table"], "relations": ["peach in cabinet"]}
    text, audit = build_checklist_grounding_think(goal, detect)
    assert "'peach in cabinet' places the peach in the cabinet, not at the table -> active." in text
    assert "nothing in the relation list places the banana at the table" in text
    assert audit["evidence_elsewhere"] == ["peach"]
    assert audit["evidence_absence"] == ["banana"]


def test_checklist_crop_missing_is_used_as_absence_evidence() -> None:
    goal = make_goal([("bowl", "table", "active")])
    detect = {
        "entities": ["table", "plate"],
        "relations": ["plate on table"],
        "crops": [{"entity": "plate", "confidence": 0.9}],
        "crop_missing": ["bowl"],
    }
    text, audit = build_checklist_grounding_think(goal, detect)
    assert "the requested crop of the bowl could not be produced" in text
    assert audit["evidence_crop_missing"] == ["bowl"]


def test_checklist_conflict_and_skip_fallback_are_counted() -> None:
    goal = make_goal([("apple", "table", "active"), ("fork", "table", "skip")])
    detect = {"entities": ["apple", "table"], "relations": ["apple on table"]}
    text, audit = build_checklist_grounding_think(goal, detect)
    assert "yet the instruction still requires handling the apple" in text
    assert audit["fallback_active_conflict"] == ["apple"]
    assert audit["fallback_skip_no_evidence"] == ["fork"]


def test_checklist_compresses_many_absence_targets_under_word_budget() -> None:
    items = ["apple", "banana", "pear", "peach", "tomato", "pumpkin", "bread"]
    goal = make_goal([(item, "kitchen work area", "active") for item in items])
    detect = {"entities": ["table"], "relations": []}
    text, audit = build_checklist_grounding_think(goal, detect)
    assert text is not None
    assert len(text.split()) <= 130
    assert set(audit["verdict_active"]) == set(items)


def test_checklist_returns_none_without_targets_or_payload() -> None:
    assert build_checklist_grounding_think({"targets": []}, {"relations": []})[0] is None
    text, _ = build_checklist_grounding_think(make_goal([("a", "b", "active")]), {})
    assert text is not None  # empty payload still yields honest absence wording


# ---------- builder integration (eligibility + override) ----------

def make_item(**kw) -> SimpleNamespace:
    goal = kw.pop("goal", make_goal([("plate", "table", "skip"), ("bowl", "table", "active")]))
    detect_result = kw.pop("detect_result", {
        "entities": ["plate", "table"],
        "relations": ["plate on table"],
    })
    teacher = SimpleNamespace(
        pre_tool="OLD PRE", grounding_tool="OLD GROUNDING", coverage_repair_tool="",
        repair_tool="", post_tool="OLD POST", quality_flags=[],
    )
    return SimpleNamespace(
        teacher=teacher,
        goal=goal,
        row_index=kw.pop("row_index", 3),
        pre_goal_tool_count=kw.pop("pre_goal_tool_count", 1),
        coverage_repair=kw.pop("coverage_repair", None),
        tool_calls=kw.pop("tool_calls", [{"name": "detect", "args": {}}]),
        tool_results=kw.pop("tool_results", [detect_result]),
        final_plan=kw.pop("final_plan", []),
        audit=kw.pop("audit", {"crop_entities": []}),
        **kw,
    )


def test_apply_override_and_eligibility_rules() -> None:
    from collections import Counter

    from scripts.build_lnsdl_sft_data import apply_ad_checklist_thinks

    eligible = make_item()
    no_detect = make_item(pre_goal_tool_count=0)
    conflict = make_item(coverage_repair={"conflict_item": "plate"})
    counters: Counter[str] = Counter()
    apply_ad_checklist_thinks([eligible, no_detect, conflict], counters=counters)

    assert eligible.audit["ad_checklist"] is True
    assert eligible.teacher.pre_tool != "OLD PRE"
    assert eligible.teacher.grounding_tool != "OLD GROUNDING"
    assert eligible.teacher.post_tool == "OLD POST"
    assert audit_noncommittal_pre_think(eligible.teacher.pre_tool) == []
    assert "'plate on table'" in eligible.teacher.grounding_tool

    assert no_detect.audit["ad_checklist"] is False
    assert no_detect.teacher.pre_tool == "OLD PRE"
    assert conflict.audit["ad_checklist"] is False
    assert conflict.teacher.grounding_tool == "OLD GROUNDING"
    assert counters["ad_checklist:applied"] == 1
    assert counters["ad_checklist:ineligible"] == 2
    assert counters["ad_checklist_items:evidence_relation"] == 1
    assert counters["ad_checklist_items:evidence_absence"] == 1


# ---------- v3.2 (cluster 8): instruction-derived candidates ----------

def test_extract_instruction_candidates_orders_and_splits() -> None:
    from nsdl.ad_checklist_think import extract_instruction_candidates

    items, surfaces = extract_instruction_candidates(
        "Scan the table for any missing fruits—the peach or the banana. "
        "Place them on the table, checking cabinet if necessary."
    )
    assert items == ["peach", "banana"]
    assert "table" in surfaces and "cabinet" in surfaces
    assert extract_instruction_candidates("") == ([], [])


def test_pre_think_questions_cover_instruction_candidates() -> None:
    # 12-1-style: GT target = apple only, instruction names apple AND tomato.
    goal = make_goal([("apple", "table", "active")])
    text = build_noncommittal_pre_think(goal, row_index=0, candidates=["apple", "tomato"])
    assert "Is the apple currently at the table?" in text
    assert "Is the tomato currently at the table?" in text
    assert audit_noncommittal_pre_think(text) == []


def test_checklist_judges_extra_candidate_present() -> None:
    goal = make_goal([("apple", "table", "active")])
    detect = {"entities": ["tomato", "table"], "relations": ["tomato on table"]}
    text, audit = build_checklist_grounding_think(goal, detect, candidates=["apple", "tomato"])
    assert "'tomato on table' shows the tomato already at the table -> it is not a relocation target." in text
    assert "nothing in the relation list places the apple at the table" in text
    assert audit["extra_present"] == ["tomato"]
    assert audit["verdict_active"] == ["apple"] and audit["verdict_skip"] == []
    assert "active: apple; skip: none." in text


def test_checklist_extra_candidate_tool_entity_safe_wording() -> None:
    # cut-family: knife is grasped by the plan but is NOT a relocation target;
    # the line must not claim "no action needed".
    goal = make_goal([("pear", "board", "active")])
    detect = {"entities": ["pear", "knife", "counter"], "relations": ["knife on counter"]}
    text, audit = build_checklist_grounding_think(goal, detect, candidates=["pear", "knife"])
    assert "'knife on counter' locates the knife -> it is not a relocation target here." in text
    assert "no action" not in text.lower()
    assert audit["extra_other"] == ["knife"]


def test_checklist_coverage_guarantee_when_candidates_miss_target() -> None:
    goal = make_goal([("bread", "table", "active")])
    detect = {"entities": ["table"], "relations": []}
    text, audit = build_checklist_grounding_think(goal, detect, candidates=["plate"])
    assert "bread" in text  # GT target judged even though not in candidates
    assert audit["verdict_active"] == ["bread"]


# ---------- v3.3 (cluster 9): participation-branched extra candidates ----------

def test_plan_participant_names_covers_entity_tool_and_carrier() -> None:
    from nsdl.ad_checklist_think import plan_participant_names

    plan = [
        {"op_type": "push_carrier", "entity": "cardboardbox", "params": {"target_robot": "R3"}},
        {"op_type": "load_carrier", "entity": "meat", "params": {"carrier": "cardboardbox"}},
        {"op_type": "apply_tool", "entity": "board", "params": {"effect": "cut", "tool": "knife"}},
        "not-a-dict",
        {"op_type": "acquire"},
    ]
    assert plan_participant_names(plan) == {"cardboardbox", "meat", "board", "knife"}
    assert plan_participant_names(None) == set()
    assert plan_participant_names([]) == set()


def test_checklist_participating_extra_is_observation_only_with_relation() -> None:
    # cut-family: the plan grasps and applies the knife -> the checklist line
    # must only locate it, never emit an exclusionary verdict (13-2 lesson).
    goal = make_goal([("pear", "board", "active")])
    detect = {"entities": ["pear", "knife", "counter"], "relations": ["knife on counter"]}
    text, audit = build_checklist_grounding_think(
        goal, detect, candidates=["pear", "knife"], plan_participants={"knife", "pear"}
    )
    assert "'knife on counter' locates the knife." in text
    assert "not a relocation target" not in text
    assert audit["extra_participating"] == ["knife"]
    assert audit["extra_other"] == [] and audit["extra_present"] == []
    # verdict summary unchanged: extras never enter it
    assert "active: pear; skip: none." in text


def test_checklist_participating_extra_without_evidence_is_availability_line() -> None:
    # 13-2-style: carrier box has no relation in the detect payload.
    goal = make_goal([("pear", "sink", "active")])
    detect = {"entities": ["pear", "sink"], "relations": []}
    text, audit = build_checklist_grounding_think(
        goal, detect, candidates=["pear", "cardboardbox"], plan_participants={"cardboardbox"}
    )
    assert "the cardboardbox is available in the scene." in text
    assert "not a relocation target" not in text
    assert audit["extra_participating"] == ["cardboardbox"]


def test_checklist_nonparticipating_extra_keeps_exclusion_line() -> None:
    # 12-1-style present fruit: plan does not touch it -> keep the v3.2
    # exclusion line (the val +30 driver).
    goal = make_goal([("apple", "table", "active")])
    detect = {"entities": ["tomato", "table"], "relations": ["tomato on table"]}
    text, audit = build_checklist_grounding_think(
        goal, detect, candidates=["apple", "tomato"], plan_participants={"apple"}
    )
    assert "'tomato on table' shows the tomato already at the table -> it is not a relocation target." in text
    assert audit["extra_present"] == ["tomato"]
    assert audit["extra_participating"] == []
