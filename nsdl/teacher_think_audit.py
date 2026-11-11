"""Shared audits for teacher think text quality and GOAL consistency."""
from __future__ import annotations

import re
from typing import Any


BANNED_THINK_TERMS = (
    "ground truth",
    "oracle",
    "task_name",
    "held-out",
    "blacklist",
    "val split",
    "test split",
    "training label",
)
THINK_NONSKIP_GOAL_SKIPPED_PATTERNS = (
    re.compile(r"\bgoal\s+is\s+skipped\b", re.I),
    re.compile(r"\bgoal\s+(?:can|could|must)\s+be\s+skipped\b", re.I),
    re.compile(
        r"\b(?:fetch|placement|serve|retrieval|task|multi-step|conditional fetch)\s+goal\s+"
        r"(?:is|can|could|must(?:\s+be)?)\s+skipped\b",
        re.I,
    ),
)


def audit_think(think: Any, task_name: str, *, goal: dict[str, Any] | None = None) -> list[str]:
    """Audit structured teacher-think fields before rendering rows."""
    pre_tool = str(getattr(think, "pre_tool", "") or "")
    grounding_tool = str(getattr(think, "grounding_tool", "") or "")
    coverage_repair_tool = str(getattr(think, "coverage_repair_tool", "") or "")
    repair_tool = str(getattr(think, "repair_tool", "") or "")
    post_tool = str(getattr(think, "post_tool", "") or "")

    flags: list[str] = []
    combined = f"{pre_tool}\n{grounding_tool}\n{coverage_repair_tool}\n{repair_tool}\n{post_tool}".lower()
    goal_decision_text = "\n".join(part for part in (pre_tool, grounding_tool, post_tool) if part)
    if not pre_tool:
        flags.append("empty_pre_tool")
    if "detect" in combined and "grounding" in combined and not grounding_tool:
        flags.append("empty_grounding_tool")
    if "repair" in combined and not repair_tool and "ok=false" in combined:
        flags.append("empty_repair_tool")
    if "goal_coverage_conflict" in combined and not coverage_repair_tool:
        flags.append("empty_coverage_repair_tool")
    if any(term in combined for term in BANNED_THINK_TERMS) or re.search(r"\bgt\b", combined):
        flags.append("banned_term")
    if task_name and task_name.lower() in combined:
        flags.append("mentions_task_name")
    if len(pre_tool.split()) > 130:
        flags.append("pre_tool_too_long")
    if len(grounding_tool.split()) > 130:
        flags.append("grounding_tool_too_long")
    if len(coverage_repair_tool.split()) > 120:
        flags.append("coverage_repair_tool_too_long")
    if len(repair_tool.split()) > 120:
        flags.append("repair_tool_too_long")
    if len(post_tool.split()) > 100:
        flags.append("post_tool_too_long")
    if goal is not None:
        flags.extend(audit_think_goal_consistency(goal_decision_text, goal))
    return flags


def audit_think_goal_consistency(think_text: str, goal: dict[str, Any]) -> list[str]:
    """Flag think/GOAL mismatches that would poison supervision."""
    text = str(think_text or "")
    flags: list[str] = []
    active_items = [str(item).strip() for item in goal.get("active") or [] if str(item).strip()]
    skip_items = [str(item).strip() for item in goal.get("skip") or [] if str(item).strip()]

    if not skip_items and any(pat.search(text) for pat in THINK_NONSKIP_GOAL_SKIPPED_PATTERNS):
        flags.append("nonskip_goal_skipped_phrase")

    for item in active_items:
        item_re = re.escape(item)
        item_token_re = item_re if " " in item else rf"[\w-]*{item_re}[\w-]*"
        item_skip_patterns = (
            re.compile(
                rf"\b{item_token_re}\b(?:[\s,;:-]+\w+){{0,3}}[\s,;:-]+(?:can|could|must)\s+be\s+skipped\b",
                re.I,
            ),
            re.compile(
                rf"\b{item_token_re}\b(?:[\s,;:-]+\w+){{0,3}}[\s,;:-]+is\s+skipped\b",
                re.I,
            ),
            re.compile(
                rf"\b{item_token_re}\b(?:[\s,;:-]+\w+){{0,3}}[\s,;:-]+(?:receives?|gets?|has)\s+skip status\b",
                re.I,
            ),
            re.compile(rf"\bskip status(?:\s+for)?\s+{item_re}\b", re.I),
        )
        if any(pat.search(text) for pat in item_skip_patterns):
            flags.append(f"active_item_marked_skipped:{item}")
    return flags


PRE_THINK_COMMIT_PATTERNS = (
    # Verdict vocabulary that would let [GOAL] copy a pre-tool conclusion
    # (docs/debug_sftv3.md §5.2 shortcut). Question sentences are exempt.
    re.compile(r"\bactive\b", re.I),
    re.compile(r"\bskip(?:ped|s)?\b", re.I),
    re.compile(r"\balready\b", re.I),
    re.compile(r"\bsatisfied\b", re.I),
    re.compile(r"\b(?:is|are|remains?|stays?|looks?|appears?|seems?)\s+(?:missing|absent|present|there|visible|clear)\b", re.I),
    re.compile(r"\bneeds?\s+(?:to\s+be\s+)?(?:brought|fetched|moved|relocated|placed|handled)\b", re.I),
    re.compile(r"\bno\s+(?:\w+\s+)?(?:needs?|is|are)\b", re.I),
    re.compile(r"\bnothing\s+(?:needs?|must|should)\b", re.I),
    re.compile(r"\bmust\s+be\s+(?:brought|fetched|moved|relocated|placed)\b", re.I),
)


def audit_noncommittal_pre_think(pre_tool: str) -> list[str]:
    """(a)-mode tripwire: the pre-tool think must not state any target verdict.

    Sentences ending with '?' are questions and exempt; every assertive
    sentence is scanned for verdict vocabulary.
    """
    flags: list[str] = []
    text = str(pre_tool or "").strip()
    if not text:
        return ["noncommittal_empty_pre_tool"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?"):
            continue
        for pattern in PRE_THINK_COMMIT_PATTERNS:
            if pattern.search(sentence):
                flags.append(f"pre_think_commits_verdict:{pattern.pattern[:30]}")
                break
    return flags


def extract_think_blocks(text: str) -> list[str]:
    return [
        str(match).strip()
        for match in re.findall(r"<think[^>]*>(.*?)</think>", str(text or ""), flags=re.I | re.S)
        if str(match).strip()
    ]


def audit_rendered_think_goal_consistency(assistant_text: str, goal: dict[str, Any]) -> list[str]:
    """Audit rendered assistant think blocks against the final GOAL."""
    think_text = "\n".join(extract_think_blocks(assistant_text))
    return audit_think_goal_consistency(think_text, goal)
