"""P5.10.6 / dataset v3.1: deterministic (a)+(d) think builders.

Root cause being fixed (docs/debug_sftv3.md §5.2): in v3 the blind pre-tool
think already committed the target verdicts and the teacher was almost always
right, so the SFT label [GOAL] was predictable from the model's own prior and
the detect evidence never entered the learned computation.

(a) non-committal pre-tool think — poses per-candidate *questions* only; the
    verdict vocabulary is banned outside question sentences, so there is no
    prior-conclusion token the model could learn to copy into [GOAL].
(d) evidence-checklist grounding think — one verdict line per GOAL target,
    derived from the *actual* detect payload (relations / entities / crops /
    crop_missing). Verdicts equal the supervised GOAL status by construction;
    the audit dict reports how each verdict is evidenced (real relation,
    closed-world absence, crop-missing signal, or an image/instruction
    fallback when the detector returned nothing usable).

Both builders are pure functions of (goal, detect payload, row_index) so they
comply with the Official-Data-Only rule: same (instruction, scene) samples,
supervision-form change only, no counterfactual construction — every cited
observation string comes verbatim from the real tool result.
"""
from __future__ import annotations

from typing import Any


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def extract_instruction_candidates(instruction: str) -> tuple[list[str], list[str]]:
    """P5.10.6 v3.2 (cluster 8): instruction-derived query plan.

    Returns (portable item candidates, surface/fixture names), both in first-
    mention order. Inputs are the instruction TEXT and the global catalog
    vocabulary only — never GT targets — so the demonstrated 'what to query'
    policy is computable at inference time (answer-decoupled).
    """
    from nsdl.catalog import find_entity_mentions, is_portable_entity, normalize_entity_name

    text = str(instruction or "")
    items: list[str] = []
    surfaces: list[str] = []
    seen: set[str] = set()
    for mention in find_entity_mentions(text):  # already sorted by mention.start
        name = normalize_entity_name(str(mention.canonical_name))
        if not name or name in seen:
            continue
        seen.add(name)
        (items if is_portable_entity(name) else surfaces).append(name)
    return items, surfaces


def _goal_targets(goal: dict[str, Any]) -> list[dict[str, Any]]:
    targets = [t for t in (goal.get("targets") or []) if isinstance(t, dict) and t.get("item")]
    return targets


def plan_participant_names(final_plan: list[dict[str, Any]] | None) -> set[str]:
    """P5.10.6 v3.3 (cluster 9): normalized names of entities that participate
    in the final plan — as an op `entity` or as `params.tool`/`params.carrier`.

    Used to branch the extra-candidate checklist line: an exclusionary verdict
    ("not a relocation target") on an entity the PLAN then uses heavily is
    contradictory supervision inside one trajectory (13-2 carrier regression,
    docs/debug_sftv3.md §7.2a). Participation is read from the supervised
    final_plan itself, so this stays within the official-data-only rule.
    """
    names: set[str] = set()
    for op in final_plan or []:
        if not isinstance(op, dict):
            continue
        entity = _norm(op.get("entity"))
        if entity:
            names.add(entity)
        params = op.get("params") or {}
        if isinstance(params, dict):
            for key in ("tool", "carrier"):
                value = _norm(params.get(key))
                if value:
                    names.add(value)
    return names


def _relation_at_destination(relations: list[str], item: str, destination: str | None) -> str | None:
    """Return the relation string that places `item` at `destination`, if any."""
    if not destination:
        return None
    item_n, dest_n = _norm(item), _norm(destination)
    for rel in relations:
        rel_n = _norm(rel)
        if rel_n.startswith(item_n + " ") and rel_n.endswith(" " + dest_n):
            return str(rel)
    return None


def _relation_elsewhere(relations: list[str], item: str, destination: str | None) -> str | None:
    """Return a relation that places `item` somewhere that is NOT `destination`."""
    item_n, dest_n = _norm(item), _norm(destination)
    for rel in relations:
        rel_n = _norm(rel)
        if rel_n.startswith(item_n + " ") and not (dest_n and rel_n.endswith(" " + dest_n)):
            return str(rel)
    return None


def _location_phrase(relation: str, item: str) -> str:
    """'peach in cabinet' -> 'in the cabinet' (best effort, falls back to raw tail)."""
    tail = str(relation)[len(str(item)):].strip()
    if not tail:
        return "elsewhere"
    parts = tail.split(" ", 1)
    if len(parts) == 2:
        return f"{parts[0]} the {parts[1]}"
    return tail


_PRE_THINK_VARIANTS = (
    (
        "A quick glance is not evidence, so I will not fix any target judgment yet. "
        "{questions} I will decide each one only from the relations and crops that "
        "detect returns."
    ),
    (
        "Before deciding anything I need structured observations rather than my "
        "first impression. {questions} The returned relations and crops, including "
        "what is NOT in the relation list, will settle each judgment."
    ),
    (
        "I should not trust an unaided read of the scene for these targets. "
        "{questions} Each judgment will follow the detect evidence — a relation, a "
        "crop, or the absence of one — not this pre-check guess."
    ),
)


def _primary_destination(targets: list[dict[str, Any]]) -> str | None:
    for target in targets:
        if target.get("destination"):
            return str(target["destination"])
    return None


def _candidate_rows(
    goal: dict[str, Any],
    candidates: list[str] | None,
) -> list[tuple[str, str | None, str | None]]:
    """Ordered (item, destination, status) rows to question/judge.

    v3.1 behavior (candidates=None): exactly the GOAL targets.
    v3.2 (candidates from the instruction): candidate order, GT targets merged
    in (coverage guarantee); non-target candidates carry status=None and the
    primary target destination for the 'is it there?' framing.
    """
    targets = _goal_targets(goal)
    by_item = {_norm(t["item"]): t for t in targets}
    if not candidates:
        return [(str(t["item"]), t.get("destination"), _norm(t.get("status"))) for t in targets]
    primary_dest = _primary_destination(targets)
    rows: list[tuple[str, str | None, str | None]] = []
    seen: set[str] = set()
    for name in candidates:
        key = _norm(name)
        if not key or key in seen:
            continue
        seen.add(key)
        target = by_item.get(key)
        if target:
            rows.append((str(target["item"]), target.get("destination"), _norm(target.get("status"))))
        else:
            rows.append((str(name), primary_dest, None))
    for target in targets:  # coverage guarantee: every GOAL target is judged
        if _norm(target["item"]) not in seen:
            rows.append((str(target["item"]), target.get("destination"), _norm(target.get("status"))))
    return rows


def build_noncommittal_pre_think(
    goal: dict[str, Any],
    *,
    crop_entities: list[str] | None = None,
    row_index: int = 0,
    candidates: list[str] | None = None,
) -> str | None:
    """(a): question-only pre-tool think. Returns None when there is no target.

    v3.2: pass `candidates` (instruction-derived) to pose one question per
    instruction candidate instead of per GT target only.
    """
    if not _goal_targets(goal):
        return None
    questions: list[str] = []
    for item, destination, _status in _candidate_rows(goal, candidates):
        if destination:
            questions.append(f"Is the {item} currently at the {destination}?")
        else:
            questions.append(f"What does the scene actually show about the {item}?")
    question_text = " ".join(questions)
    template = _PRE_THINK_VARIANTS[int(row_index) % len(_PRE_THINK_VARIANTS)]
    text = template.format(questions=question_text)
    crops = [str(c) for c in (crop_entities or []) if str(c).strip()]
    if crops:
        text += f" I am also requesting close-up crops of: {', '.join(crops)}."
    return text


def build_checklist_grounding_think(
    goal: dict[str, Any],
    detect_payload: dict[str, Any],
    *,
    row_index: int = 0,
    candidates: list[str] | None = None,
    plan_participants: set[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """(d): per-candidate evidence checklist. Returns (text, audit).

    v3.1 (candidates=None): one verdict line per GOAL target.
    v3.2: one line per instruction candidate — GT targets get their skip/active
    verdict; non-target candidates get an evidence-cited "not a relocation
    target" line (they are absent from GOAL by supervision, e.g. the already-
    present dish in a conditional-fetch task, or the knife whose handling is a
    tool-use step rather than a relocation).
    v3.3 (cluster 9, docs/debug_sftv3.md §7.3): pass `plan_participants`
    (see `plan_participant_names`) — an extra candidate that the final plan
    actually uses (carrier box, knife) gets an OBSERVATION-ONLY line with zero
    verdict instead of the exclusionary wording, so the checklist never
    contradicts the PLAN supervision that follows in the same turn. Extra
    candidates the plan does not touch keep the v3.2 exclusion line (it is
    the val +30 driver and is consistent there).

    audit keys: evidence_relation / evidence_elsewhere / evidence_absence /
    evidence_crop_missing / fallback_skip_no_evidence / fallback_active_conflict /
    extra_present / extra_other / extra_participating (lists of item names),
    plus verdict_active / verdict_skip.
    """
    targets = _goal_targets(goal)
    if not targets or not isinstance(detect_payload, dict):
        return None, {}
    relations = [str(r) for r in (detect_payload.get("relations") or [])]
    entities = {_norm(e) for e in (detect_payload.get("entities") or [])}
    crop_ok = {_norm((c or {}).get("entity")) for c in (detect_payload.get("crops") or []) if isinstance(c, dict)}
    crop_missing = {_norm(c) for c in (detect_payload.get("crop_missing") or [])}

    audit: dict[str, list[str]] = {
        "evidence_relation": [],
        "evidence_elsewhere": [],
        "evidence_absence": [],
        "evidence_crop_missing": [],
        "fallback_skip_no_evidence": [],
        "fallback_active_conflict": [],
        "extra_present": [],
        "extra_other": [],
        "extra_participating": [],
    }
    participants = {(_norm(p)) for p in (plan_participants or set())}
    lines: list[str] = []
    line_items: list[str] = []
    active_items: list[str] = []
    skip_items: list[str] = []

    for item, destination, status in _candidate_rows(goal, candidates):
        dest_text = f"at the {destination}" if destination else "handled"
        rel_at = _relation_at_destination(relations, item, destination)
        rel_else = _relation_elsewhere(relations, item, destination)
        line_items.append(item)

        if status is None:
            # v3.2 extra candidate: mentioned by the instruction, not a GOAL
            # target. Cite the evidence; never claim the plan ignores it.
            if _norm(item) in participants:
                # v3.3 (cluster 9): the plan uses this entity (carrier/tool) —
                # observation only, zero verdict; whether/how to use it is the
                # PLAN's supervision, not the checklist's.
                rel = rel_at or rel_else
                if rel:
                    lines.append(f"'{rel}' locates the {item}.")
                else:
                    lines.append(f"the {item} is available in the scene.")
                audit["extra_participating"].append(item)
            elif rel_at:
                lines.append(
                    f"'{rel_at}' shows the {item} already {dest_text} -> it is not "
                    "a relocation target."
                )
                audit["extra_present"].append(item)
            elif rel_else:
                lines.append(
                    f"'{rel_else}' locates the {item} -> it is not a relocation "
                    "target here."
                )
                audit["extra_other"].append(item)
            else:
                lines.append(
                    f"no relocation evidence involves the {item} -> it is not a "
                    "relocation target here."
                )
                audit["extra_other"].append(item)
            continue

        if status == "skip":
            skip_items.append(item)
            if rel_at:
                lines.append(f"'{rel_at}' shows the {item} already {dest_text} -> skip.")
                audit["evidence_relation"].append(item)
            elif _norm(item) in crop_ok:
                lines.append(
                    f"the close-up crop confirms the {item} in place {dest_text} -> skip."
                )
                audit["fallback_skip_no_evidence"].append(item)
            else:
                lines.append(
                    f"the image clearly shows the {item} {dest_text} even though no "
                    "relation was returned for it -> skip."
                )
                audit["fallback_skip_no_evidence"].append(item)
            continue

        # status == active (anything not skip is supervised as active).
        active_items.append(item)
        if rel_at:
            lines.append(
                f"'{rel_at}' is returned, yet the instruction still requires handling "
                f"the {item} -> keep it active."
            )
            audit["fallback_active_conflict"].append(item)
        elif rel_else:
            loc = _location_phrase(rel_else, item)
            lines.append(
                f"'{rel_else}' places the {item} {loc}, not {dest_text} -> active."
            )
            audit["evidence_elsewhere"].append(item)
        elif _norm(item) in crop_missing:
            lines.append(
                f"the requested crop of the {item} could not be produced and no "
                f"relation puts it {dest_text} -> treat it as absent -> active."
            )
            audit["evidence_crop_missing"].append(item)
        elif _norm(item) in entities:
            lines.append(
                f"the {item} is detected but NO relation puts it {dest_text} -> "
                "not there yet -> active."
            )
            audit["evidence_absence"].append(item)
        else:
            lines.append(
                f"nothing in the relation list places the {item} {dest_text} -> "
                "absent from its destination -> active."
            )
            audit["evidence_absence"].append(item)

    summary_active = ", ".join(active_items) if active_items else "none"
    summary_skip = ", ".join(skip_items) if skip_items else "none"

    def _render(current_lines: list[str]) -> str:
        return (
            "Judging each target only from the detect result: "
            + " ".join(current_lines)
            + f" Verdict from evidence - active: {summary_active}; skip: {summary_skip}."
        )

    text = _render(lines)
    if len(text.split()) > 122 and len(audit["evidence_absence"]) > 1:
        # Compress the (most common) closed-world absence lines into one grouped
        # line so the grounding think stays under the 130-word audit budget.
        grouped = ", ".join(audit["evidence_absence"])
        compact = [
            line
            for line, item in zip(lines, line_items)
            if item not in audit["evidence_absence"]
        ]
        compact.append(
            f"no relation places the {grouped} at their destinations -> all active."
        )
        text = _render(compact)
    full_audit: dict[str, Any] = {k: v for k, v in audit.items()}
    full_audit["verdict_active"] = active_items
    full_audit["verdict_skip"] = skip_items
    return text, full_audit
