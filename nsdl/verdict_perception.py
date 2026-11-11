"""P10.S1 A-axis inversion arm — **eval-only**.

The ``detect`` tool normally returns *evidence only* (entities / relations /
confidences / crops) and the skip-vs-active judgement stays inside the model's
``[GOAL]`` block. That asymmetry is the operational criterion this project
claims ([idea.md §5 #1]): *uncertain -> the tool supplies evidence, the decision
stays in the model; deterministic -> the decision is handed to a sound solver*.

This module implements the **counterfactual arm** that makes that claim
falsifiable ([review.md §5.1] / [plan.md P10.S1]): ``detect`` additionally
returns an externalized verdict ``already_at_destination: {item: bool}``, i.e.
the tool performs the state judgement on the model's behalf.

Design constraints (all three are load-bearing; violating any one invalidates
the experiment — see [plan.md P10.S1]):

1. **Matched perception.** The verdict is a pure function of the *same*
   ``DetectResult`` the evidence-only arm would have returned, plus the task
   goal spec (destination) that already sits in the model's own prompt. No GT
   is read here. The two arms therefore differ in *who decides*, never in *what
   was perceived* — so a delta cannot be explained by "we handed it the answer".
   The implementation enforces this structurally: it is a **post-processor**
   over the existing detect result, not a replacement perception backend, so it
   runs downstream of both the YOLO path and the ``oracle`` bracket backend.
2. **Verdict provenance is reported, not merged.** Running this on top of
   ``--detector-backend yolo`` vs ``oracle`` gives the matched-perception 2x2;
   the two must be reported separately, otherwise "delegating to an unsound
   judge" is confounded with "delegating at all".
3. **Additive, not subtractive.** Evidence keys are preserved verbatim and the
   verdict is *added*. The model was SFT-trained on evidence-only tool returns;
   an evidence-free (verdict-only) payload would be a format OOD and any
   degradation would be unattributable. A subtractive arm is only meaningful
   with retraining (S1b), which is out of scope here.

Anti-leak guard (same discipline as ``nsdl.oracle_perception``): importing this
module requires ``NSDL_VERDICT_PERCEPTION=eval-only``. Training, data
construction and RL never set it, so an accidental import raises. This also
means [tools.md §2]'s "detect never returns inferences" invariant is suspended
*only* inside an eval process that opted in explicitly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

if os.environ.get("NSDL_VERDICT_PERCEPTION") != "eval-only":
    raise ImportError(
        "nsdl.verdict_perception is the eval-only A-axis inversion arm (P10.S1). "
        "It deliberately violates the 'detect returns evidence, never a verdict' "
        "invariant and must never be imported from training, data-construction, "
        "or RL code paths. Set NSDL_VERDICT_PERCEPTION=eval-only explicitly in "
        "the eval runner to use it."
    )

from nsdl.catalog import find_entity_mentions, is_portable_entity, normalize_entity_name  # noqa: E402
from nsdl.perc_tool import DetectResult  # noqa: E402
from nsdl.schema import SceneFacts  # noqa: E402


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


@dataclass
class VerdictDetectResult(DetectResult):
    """``DetectResult`` plus the externalized state judgement.

    ``to_dict`` keeps every evidence key byte-identical to the base class and
    appends ``already_at_destination`` (constraint 3: additive). ``verdict_source``
    records which perception backend the verdict was derived from so the artifact
    can be audited after the fact (constraint 2).
    """

    already_at_destination: dict[str, bool] = field(default_factory=dict)
    verdict_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        if self.already_at_destination:
            payload["already_at_destination"] = dict(self.already_at_destination)
        return payload


def instruction_item_candidates(instruction: str) -> list[str]:
    """Portable items named by the instruction, in first-mention order.

    Instruction text + global catalog vocabulary only — never GT targets — so
    the candidate set is computable at inference time by the model itself.
    Mirrors ``ad_checklist_think.extract_instruction_candidates``'s item half.
    """
    items: list[str] = []
    seen: set[str] = set()
    for mention in find_entity_mentions(str(instruction or "")):
        name = normalize_entity_name(str(mention.canonical_name))
        if not name or name in seen:
            continue
        seen.add(name)
        if is_portable_entity(name):
            items.append(name)
    return items


def destination_for(item: str, scene_facts: SceneFacts | None) -> str | None:
    """Destination of ``item`` per the task goal spec already in the prompt.

    Per-item ``destination_pairs`` win over the scene-level ``destination``;
    both come from the instruction parser, never from GT.
    """
    if scene_facts is None:
        return None
    item_n = _norm(item)
    pairs = scene_facts.destination_pairs or []
    if not isinstance(pairs, list):
        pairs = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        if _norm(normalize_entity_name(pair.get("item")) or pair.get("item")) == item_n:
            target = normalize_entity_name(pair.get("target")) or pair.get("target")
            if target:
                return str(target)
    return scene_facts.destination or None


def _relation_at(relations: list[str], item: str, destination: str) -> bool:
    item_n, dest_n = _norm(item), _norm(destination)
    return any(
        _norm(rel).startswith(item_n + " ") and _norm(rel).endswith(" " + dest_n)
        for rel in relations
    )


def _relation_elsewhere(relations: list[str], item: str, destination: str) -> bool:
    item_n, dest_n = _norm(item), _norm(destination)
    return any(
        _norm(rel).startswith(item_n + " ") and not _norm(rel).endswith(" " + dest_n)
        for rel in relations
    )


def derive_verdicts(
    result: DetectResult,
    scene_facts: SceneFacts | None,
) -> dict[str, bool]:
    """``{item: already_at_destination}`` from THIS call's evidence only.

    An item is reported only when the returned relations actually settle it:
    a relation placing it at the destination -> ``True``; a relation placing it
    somewhere else -> ``False``. Items the call said nothing about are **omitted**
    rather than guessed — inventing a verdict from absence would make the arm a
    different (weaker) perception source and break constraint 1.
    """
    verdicts: dict[str, bool] = {}
    relations = list(result.relations or [])
    if not relations:
        return verdicts

    candidates = instruction_item_candidates(
        scene_facts.instruction if scene_facts is not None else ""
    )
    # Items this call observed but the instruction did not name (e.g. the model
    # queried more broadly) are still judged, so the verdict covers exactly what
    # the evidence covers.
    for entity in result.entities or []:
        name = normalize_entity_name(entity) or _norm(entity)
        if name and name not in candidates and is_portable_entity(name):
            candidates.append(name)

    for item in candidates:
        destination = destination_for(item, scene_facts)
        if not destination:
            continue
        if _norm(destination) == _norm(item):
            # Degenerate comparison guard: if an item's own destination ever
            # resolves to itself, `_relation_at` can never match and
            # `_relation_elsewhere` fires on every relation, so the item would be
            # reported "not at destination" unconditionally — a verdict from the
            # comparison's shape, not from evidence, which constraint 1 forbids.
            # This is defensive, not a fix for an observed case: on LOCUS the
            # two-hop carrier family (F9: "load the fork into the cardboardbox,
            # then push the loaded cardboardbox to the sink") is already resolved
            # correctly by `destination_pairs` (cardboardbox -> sink), and this
            # branch was measured to fire on 0/1176 eval items.
            continue
        if _relation_at(relations, item, destination):
            verdicts[item] = True
        elif _relation_elsewhere(relations, item, destination):
            verdicts[item] = False
    return verdicts


def make_verdict_postprocess(scene_facts: SceneFacts | None, *, source: str = ""):
    """Per-sample ``detect_postprocess`` hook for the tool runtime.

    Wraps whatever ``DetectResult`` the configured perception backend produced
    (YOLO or the ``oracle`` bracket) and returns a ``VerdictDetectResult``
    carrying identical evidence plus the externalized judgement.
    """

    def _postprocess(result: DetectResult) -> DetectResult:
        if result is None:
            return result
        return VerdictDetectResult(
            entities=list(result.entities),
            relations=list(result.relations),
            confidences=dict(result.confidences),
            crops=list(result.crops),
            crops_truncated=bool(result.crops_truncated),
            crop_missing=list(result.crop_missing),
            crop_unavailable_no_image=bool(result.crop_unavailable_no_image),
            already_at_destination=derive_verdicts(result, scene_facts),
            verdict_source=str(source or ""),
        )

    return _postprocess
