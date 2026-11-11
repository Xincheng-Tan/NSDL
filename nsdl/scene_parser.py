from __future__ import annotations

import re
from abc import ABC, abstractmethod

from nsdl.catalog import (
    ACTIVATABLE_TAG,
    CONTAINER_TAG,
    INSPECTABLE_TAG,
    OPENABLE_TAG,
    PLACE_TARGET_TAG,
    SUPPORT_TAG,
    TOOL_TAG,
    entity_can_be_destination,
    entity_has_any_tag,
    entity_has_tag,
    find_entity_mentions,
    is_portable_entity,
    normalize_catalog_text,
)
from nsdl.schema import BenchmarkSample, RobotInstance, SceneFacts


_DESTINATION_PREPOSITION_PATTERN = re.compile(r"\b(?:onto|on|to|into|in|inside)\b")
_SOURCE_PREPOSITION_PATTERN = re.compile(r"\b(?:from|inside|in)\b")
_FALLBACK_LOCATION_HINT_PATTERN = re.compile(
    r"\b(?:including|such as|if needed|if necessary|if absent|if missing|if it isn't|if it is not|if you don't see|if you do not see|not in sight|out of sight|out of view|nowhere outside|spot|might|check(?:ing)?|look(?:ing)?|inspect(?:ing)?|peek|verify(?:ing)?|hidden|stored)\b"
)
_WASH_HINT_PATTERN = re.compile(
    r"\b(?:wash|washed|clean|rinsed?|freshen|rinse off|run water (?:over|on)|under running water)\b"
)
_CUT_HINT_PATTERN = re.compile(r"\b(?:cut|cutting|slice|sliced|slicing|chop|chopped|chopping|carve)\b")
_POST_PLACE_HINT_PATTERN = re.compile(
    r"\b(?:set|plac(?:e[ds]?|ing)|put(?:ting)?|serv(?:e[ds]?|ing)|arrang(?:e[ds]?|ing)|rest(?:s|ed|ing)?|drop(?:ped|s|ping)?|deliver(?:ed|s|ing)?|transfer(?:red|s|ring)?)\b"
)
_ACTIVATE_HINT_PATTERN = re.compile(
    r"\b(?:activate|turn it on|switch it on|start the process|start its operation|switch on)\b"
)
_INSPECT_HINT_PATTERN = re.compile(
    r"\b(?:inspect|observe|survey|scan|glance|look|check|verify|assess|explore|examine|peek)\b"
)
_SOFT_STORAGE_FALLBACK_PATTERN = re.compile(
    r"\b(?:worth checking|may have been stored|remember (?:that )?it may be (?:in|inside)|it may be (?:in|inside)|open [a-z0-9' ]* and (?:fetch|get|retrieve))\b"
)
_ABSENCE_OUTSIDE_HINT_PATTERN = re.compile(
    r"\b(?:if (?:it(?:'s| is)? )?nowhere outside|nowhere outside|not outside|outside nowhere|if absent|if missing|if you don't see|if you do not see|out of sight|out of view|not in sight)\b"
)
_RETRIEVE_HINT_PATTERN = re.compile(r"\b(?:fetch|get|retrieve|bring)\b")
_INSPECT_MISSING_HINT_PATTERN = re.compile(
    r"\b(?:missing|ensure both|either|whichever of|whichever dish|lack either|don't see both|do not see both|can't see|cannot see|not already|not present|absent|still absent|only one dish|the other|supply the other|hosts both|holds both)\b"
)
_RECOVERY_ACTION_HINT_PATTERN = re.compile(
    r"\b(?:bring|deliver|fetch|get|carry|place|set|put|restore|supply)\b"
)
_PARALLEL_HINT_PATTERN = re.compile(r"\b(?:simultaneously|at the same time|meanwhile|along with|both)\b")
_PURPOSE_RELATION_HINT_PATTERN = re.compile(
    r"\bto\s+(?:finalize|finish|complete|wrap up|prepare|organize|tidy|clear|maintain|keep)\b"
)
# "Clear the counter -- move every item on it to the rack": the movable items are
# named only by a generic quantifier, so no destination pair can be formed and the
# destination fallback would otherwise take the first place target in mention
# order -- i.e. the surface being cleared. These two patterns identify that shape.
_GENERIC_QUANTIFIER_PATTERN = re.compile(
    r"\b(?:everything|every item|every object|every thing|all items|all objects|"
    r"anything|any item|any object|whatever|the rest|the remainder)\b"
)
_CLEARANCE_SOURCE_HINT_PATTERN = re.compile(
    r"\b(?:clear(?:ed|ing|s)?|tidy(?:ing)?|declutter(?:ed|ing)?|put away|left out|"
    r"still out|out on|sitting on|remains? on|remaining on|left on)\b"
)
_PUSH_HINT_PATTERN = re.compile(
    r"\b(?:push(?:ing)? (?:the )?cardboard ?box|shift (?:the )?cardboard ?box|cardboard ?box the|use (?:a|the) cardboard ?box|into a cardboard ?box|inside a cardboard ?box|house [a-z0-9']+ in a cardboard ?box|secure it inside a cardboard ?box|extract the|fetch [a-z0-9']+ afterward|isn't feasible|long way|too distant|route is too long|too far to carry directly|distance makes direct handling difficult|push assisted)\b"
)

_INTENT_ORDER = [
    "relocate",
    "conditional_container_check",
    "retrieve",
    "wash",
    "tool_interaction",
    "activate",
    "inspect",
    "inspect_missing",
    "parallelize",
    "push_transfer",
]

_INTENT_CONFIDENCE_THRESHOLDS = {
    "relocate": 0.60,
    "conditional_container_check": 0.60,
    "retrieve": 0.60,
    "wash": 0.75,
    "tool_interaction": 0.75,
    "activate": 0.70,
    "inspect": 0.55,
    "inspect_missing": 0.55,
    "parallelize": 0.60,
    "push_transfer": 0.75,
}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _is_movable_task_item(entity_name: str) -> bool:
    return is_portable_entity(entity_name) or entity_name == "cardboardbox"


def _looks_like_fallback_relation_bridge(bridge: str) -> bool:
    normalized_bridge = bridge.strip()
    if not normalized_bridge:
        return False
    if any(punctuation in normalized_bridge for punctuation in (".", ";", "?", "!")):
        return True
    return bool(_FALLBACK_LOCATION_HINT_PATTERN.search(normalized_bridge))


def _looks_like_fallback_destination_context(
    *,
    target_name: str,
    local_context: str,
) -> bool:
    if not entity_has_tag(target_name, OPENABLE_TAG):
        return False
    return bool(
        _FALLBACK_LOCATION_HINT_PATTERN.search(local_context)
        or _INSPECT_HINT_PATTERN.search(local_context)
    )


def _looks_like_purpose_relation_bridge(bridge: str) -> bool:
    normalized_bridge = bridge.strip()
    if not normalized_bridge:
        return False
    return bool(_PURPOSE_RELATION_HINT_PATTERN.search(normalized_bridge))


def _normalize_context(text: str) -> str:
    return normalize_catalog_text(text)


def _looks_like_openable_fallback_context(context: str) -> bool:
    normalized_context = _normalize_context(context)
    if not normalized_context:
        return False
    if _FALLBACK_LOCATION_HINT_PATTERN.search(normalized_context):
        return True
    if _SOFT_STORAGE_FALLBACK_PATTERN.search(normalized_context):
        return True
    return bool(
        _ABSENCE_OUTSIDE_HINT_PATTERN.search(normalized_context)
        and _RETRIEVE_HINT_PATTERN.search(normalized_context)
    )


class BaseSceneParser(ABC):
    @abstractmethod
    def parse(self, sample: BenchmarkSample) -> SceneFacts:
        raise NotImplementedError


class RuleBasedSceneParser(BaseSceneParser):
    def parse(self, sample: BenchmarkSample) -> SceneFacts:
        instruction_text = sample.instruction
        normalized_instruction = normalize_catalog_text(instruction_text)
        mentions = find_entity_mentions(sample.instruction)
        mentioned_entities = _dedupe_preserve_order(
            [mention.canonical_name for mention in mentions]
        )
        robot_inventory = [
            RobotInstance(slot_id=slot_id, robot_type=robot_type)
            for slot_id, robot_type in sample.robot_inventory.items()
            if robot_type
        ]

        destination_pairs = self._extract_destination_pairs(instruction_text, mentions)
        source_locations = self._extract_source_locations(instruction_text, mentions)
        fallback_locations = self._extract_fallback_locations(instruction_text, mentions)
        support_targets = _dedupe_preserve_order(
            [
                mention.canonical_name
                for mention in mentions
                if entity_has_any_tag(
                    mention.canonical_name,
                    {SUPPORT_TAG, PLACE_TARGET_TAG},
                )
            ]
        )
        container_targets = _dedupe_preserve_order(
            [
                mention.canonical_name
                for mention in mentions
                if entity_has_tag(mention.canonical_name, CONTAINER_TAG)
            ]
        )
        tool_items = _dedupe_preserve_order(
            [
                mention.canonical_name
                for mention in mentions
                if entity_has_tag(mention.canonical_name, TOOL_TAG)
            ]
        )
        activation_targets = self._extract_activation_targets(instruction_text, mentions)
        inspection_targets = self._extract_inspection_targets(normalized_instruction, mentions)
        intent_hypotheses = self._build_intent_hypotheses(
            normalized_instruction,
            mentions,
            tool_items=tool_items,
            activation_targets=activation_targets,
            inspection_targets=inspection_targets,
            destination_pairs=destination_pairs,
            source_locations=source_locations,
            fallback_locations=fallback_locations,
        )
        parsed_intents = self._select_intents(intent_hypotheses)
        # Target decomposition is an SLM responsibility. Keep the legacy schema
        # field empty so parser-derived target sets cannot influence runtime
        # planning through SceneFacts.
        target_items: list[str] = []
        inspect_mode = self._infer_inspect_mode(
            normalized_instruction,
            inspection_targets=inspection_targets,
            destination_pairs=destination_pairs,
            parsed_intents=parsed_intents,
        )
        parsed_intents = self._normalize_parsed_intents(
            parsed_intents,
            inspect_mode=inspect_mode,
        )
        destination_candidates = self._extract_destination_candidates(
            destination_pairs,
            support_targets=support_targets,
            container_targets=container_targets,
            activation_targets=activation_targets,
            inspection_targets=inspection_targets,
        )
        clearance_sources = self._extract_clearance_sources(instruction_text, mentions)
        destination_candidates = [
            candidate
            for candidate in destination_candidates
            if candidate not in clearance_sources
        ]
        destination = self._select_primary_destination(
            parsed_intents=parsed_intents,
            destination_pairs=destination_pairs,
            destination_candidates=destination_candidates,
            activation_targets=activation_targets,
            inspection_targets=inspection_targets,
            inspect_mode=inspect_mode,
        )
        primary_object = None
        secondary_objects: list[str] = []

        field_sources = {
            "instruction": "benchmark user prompt text",
            "image_available": "benchmark images field presence",
            "robot_inventory": "benchmark system prompt: Available robot set",
            "primary_object": "disabled: target decomposition is reserved for SLM output",
            "destination_pairs": "rule-based relation parse from instruction",
            "secondary_objects": "disabled: target decomposition is reserved for SLM output",
            "target_items": "disabled: target decomposition is reserved for SLM output",
            "destination": "rule-based relation parse from instruction",
            "destination_candidates": "rule-based relation parse from instruction",
            "source_locations": "instruction cue parse around source prepositions",
            "fallback_locations": "instruction cue parse around conditional container mentions",
            "targets_in_cabinet": "disabled: target status inference is reserved for SLM output",
            "tool_items": "ontology-backed entity extraction from instruction",
            "activation_targets": "rule-based verb and entity cue parse",
            "inspection_targets": "rule-based verb and entity cue parse",
            "inspect_mode": "rule-based inspection-role parse from instruction",
            "parsed_intents": "structured rule-based intent classifier with confidence-scored hypotheses",
            "mentioned_objects": "ontology-backed entity extraction from instruction",
            "robot_anchor_status": "reserved for YOLO-backed refiner geometry grounding",
        }

        return SceneFacts(
            parser_backend="rule_initial_parse",
            instruction=sample.instruction,
            image_available=bool(sample.image_bytes),
            robot_inventory=robot_inventory,
            primary_object=primary_object,
            destination=destination,
            destination_pairs=destination_pairs,
            secondary_objects=secondary_objects,
            target_items=target_items,
            destination_candidates=destination_candidates,
            source_locations=source_locations,
            fallback_locations=fallback_locations,
            targets_in_cabinet=[],
            tool_items=tool_items,
            activation_targets=activation_targets,
            inspection_targets=inspection_targets,
            inspect_mode=inspect_mode,
            parsed_intents=parsed_intents,
            mentioned_objects=mentioned_entities,
            field_sources=field_sources,
            parser_notes={
                "stage": "initial_parse",
                "image_used": False,
                "inspect_mode": inspect_mode,
                "destination_pairs": destination_pairs,
                "target_items_disabled": True,
                "support_targets_internal": support_targets,
                "container_targets_internal": container_targets,
                "clearance_sources": clearance_sources,
                "intent_hypotheses": intent_hypotheses,
                "mentioned_entity_spans": [
                    {
                        "entity": mention.canonical_name,
                        "alias": mention.alias,
                        "span": [mention.start, mention.end],
                    }
                    for mention in mentions
                ],
            },
        )

    def _extract_destination_pairs(self, instruction: str, mentions: list) -> list[dict[str, str]]:
        pairs: list[dict[str, str]] = []
        for index, mention in enumerate(mentions):
            if not _is_movable_task_item(mention.canonical_name):
                continue
            first_pair_for_item: dict[str, str] | None = None
            for next_mention in mentions[index + 1 :]:
                if not entity_can_be_destination(next_mention.canonical_name):
                    continue
                bridge = _normalize_context(instruction[mention.end : next_mention.start])
                if not _DESTINATION_PREPOSITION_PATTERN.search(bridge):
                    continue
                local_context = _normalize_context(
                    instruction[
                    max(0, next_mention.start - 32) : min(len(instruction), next_mention.end + 32)
                    ]
                )
                if _looks_like_fallback_relation_bridge(bridge):
                    continue
                if _looks_like_purpose_relation_bridge(bridge):
                    continue
                if _looks_like_fallback_destination_context(
                    target_name=next_mention.canonical_name,
                    local_context=local_context,
                ):
                    continue
                if mention.canonical_name == next_mention.canonical_name:
                    continue
                candidate_pair = {
                    "item": mention.canonical_name,
                    "target": next_mention.canonical_name,
                }
                if first_pair_for_item is None:
                    pairs.append(candidate_pair)
                    first_pair_for_item = candidate_pair
                    continue
                if (
                    _WASH_HINT_PATTERN.search(instruction.lower())
                    and _POST_PLACE_HINT_PATTERN.search(local_context)
                    and candidate_pair not in pairs
                ):
                    pairs.append(candidate_pair)
            if first_pair_for_item is not None:
                continue
        return pairs

    def _extract_clearance_sources(self, instruction: str, mentions: list) -> list[str]:
        """Place targets a "clear everything off X" instruction names as the SOURCE.

        These phrasings refer to the movable items by a generic quantifier instead
        of naming them, so ``_extract_destination_pairs`` finds nothing and the
        destination falls back to mention order -- which puts the surface being
        cleared first. Excluding these from the candidate list is what keeps
        ``destination`` pointing at where the items are supposed to end up.
        """
        if not _GENERIC_QUANTIFIER_PATTERN.search(_normalize_context(instruction)):
            return []
        sources: list[str] = []
        for index, mention in enumerate(mentions):
            if not entity_can_be_destination(mention.canonical_name):
                continue
            context = _normalize_context(
                instruction[max(0, mention.start - 64) : min(len(instruction), mention.end + 48)]
            )
            if not _CLEARANCE_SOURCE_HINT_PATTERN.search(context):
                continue
            # Only a source if some *later* place target can still serve as the
            # destination; otherwise the clearance cue belongs to the destination
            # side ("...should be put away on the rack") and we would strip the
            # only candidate there is.
            if not any(
                entity_can_be_destination(later.canonical_name)
                and later.canonical_name != mention.canonical_name
                and _DESTINATION_PREPOSITION_PATTERN.search(
                    _normalize_context(instruction[mention.end : later.start])
                )
                for later in mentions[index + 1 :]
            ):
                continue
            sources.append(mention.canonical_name)
        return _dedupe_preserve_order(sources)

    def _extract_source_locations(self, instruction: str, mentions: list) -> list[str]:
        source_locations: list[str] = []
        for mention in mentions:
            if not entity_has_tag(mention.canonical_name, OPENABLE_TAG):
                continue
            context = _normalize_context(instruction[max(0, mention.start - 24) : mention.end + 8])
            if _SOURCE_PREPOSITION_PATTERN.search(context):
                source_locations.append(mention.canonical_name)
        return _dedupe_preserve_order(source_locations)

    def _extract_fallback_locations(self, instruction: str, mentions: list) -> list[str]:
        fallback_locations: list[str] = []
        for mention in mentions:
            if not entity_has_tag(mention.canonical_name, OPENABLE_TAG):
                continue
            context = instruction[
                max(0, mention.start - 72) : min(len(instruction), mention.end + 64)
            ]
            if _looks_like_openable_fallback_context(context):
                fallback_locations.append(mention.canonical_name)
        return _dedupe_preserve_order(fallback_locations)

    def _extract_activation_targets(self, instruction: str, mentions: list) -> list[str]:
        activation_targets: list[str] = []
        for mention in mentions:
            if not entity_has_any_tag(
                mention.canonical_name,
                {ACTIVATABLE_TAG, TOOL_TAG},
            ):
                continue
            context = _normalize_context(
                instruction[
                max(0, mention.start - 40) : min(len(instruction), mention.end + 40)
                ]
            )
            if (
                _WASH_HINT_PATTERN.search(context)
                or _CUT_HINT_PATTERN.search(context)
                or _ACTIVATE_HINT_PATTERN.search(context)
                or _INSPECT_HINT_PATTERN.search(context)
            ):
                activation_targets.append(mention.canonical_name)
        return _dedupe_preserve_order(activation_targets)

    def _extract_inspection_targets(self, instruction: str, mentions: list) -> list[str]:
        if not _INSPECT_HINT_PATTERN.search(instruction):
            return []
        inspection_targets = [
            mention.canonical_name
            for mention in mentions
            if entity_has_any_tag(
                mention.canonical_name,
                {INSPECTABLE_TAG, ACTIVATABLE_TAG, PLACE_TARGET_TAG},
            )
        ]
        return _dedupe_preserve_order(inspection_targets)

    def _build_intent_hypotheses(
        self,
        instruction: str,
        mentions: list,
        *,
        tool_items: list[str],
        activation_targets: list[str],
        inspection_targets: list[str],
        destination_pairs: list[dict[str, str]],
        source_locations: list[str],
        fallback_locations: list[str],
    ) -> dict[str, dict[str, object]]:
        hypotheses: dict[str, dict[str, object]] = {}
        mentioned_entities = {mention.canonical_name for mention in mentions}
        portable_mentions = _dedupe_preserve_order(
            [
                mention.canonical_name
                for mention in mentions
                if _is_movable_task_item(mention.canonical_name)
            ]
        )
        explicitly_paired_items = {
            pair["item"]
            for pair in destination_pairs
            if isinstance(pair, dict) and pair.get("item")
        }
        explicit_pair_cover = (
            len(portable_mentions) >= 2
            and set(portable_mentions).issubset(explicitly_paired_items)
        )
        has_inspect_verb = bool(_INSPECT_HINT_PATTERN.search(instruction))
        has_missing_prior = bool(_INSPECT_MISSING_HINT_PATTERN.search(instruction))
        has_recovery_action = bool(_RECOVERY_ACTION_HINT_PATTERN.search(instruction))
        destination_under_inspection = bool(
            inspection_targets
            and any(entity_can_be_destination(target) for target in inspection_targets)
        )
        pure_inspection_only = (
            has_inspect_verb
            and not portable_mentions
            and not destination_pairs
            and not has_missing_prior
            and not has_recovery_action
        )

        def add_hypothesis(intent: str, confidence: float, evidence: list[str]) -> None:
            bounded_confidence = max(0.0, min(1.0, confidence))
            if bounded_confidence <= 0.0:
                return
            hypotheses[intent] = {
                "confidence": round(bounded_confidence, 2),
                "evidence": _dedupe_preserve_order([item for item in evidence if item]),
            }

        if destination_pairs:
            add_hypothesis("relocate", 0.95, ["explicit_destination_pairs"])

        if fallback_locations and not pure_inspection_only:
            add_hypothesis("conditional_container_check", 0.90, ["fallback_storage_cue"])

        retrieve_confidence = 0.0
        retrieve_evidence: list[str] = []
        if not pure_inspection_only and any(
            entity_has_tag(location, OPENABLE_TAG)
            for location in source_locations + fallback_locations
        ):
            retrieve_confidence += 0.75
            retrieve_evidence.append("openable_source_or_fallback")
        if not pure_inspection_only and _PUSH_HINT_PATTERN.search(instruction):
            retrieve_confidence += 0.20
            retrieve_evidence.append("push_transfer_requires_fetch")
        add_hypothesis("retrieve", retrieve_confidence, retrieve_evidence)

        if _WASH_HINT_PATTERN.search(instruction) and "sink" in mentioned_entities:
            add_hypothesis("wash", 0.95, ["wash_verb", "sink_mentioned"])

        if _CUT_HINT_PATTERN.search(instruction) and tool_items:
            add_hypothesis("tool_interaction", 0.95, ["cut_verb", "tool_mentioned"])

        activate_confidence = 0.0
        activate_evidence: list[str] = []
        if _ACTIVATE_HINT_PATTERN.search(instruction):
            activate_confidence += 0.85
            activate_evidence.append("explicit_activate_verb")
        if _WASH_HINT_PATTERN.search(instruction) and "sink" in mentioned_entities:
            activate_confidence += 0.10
            activate_evidence.append("wash_station_activation")
        if _CUT_HINT_PATTERN.search(instruction) and tool_items:
            activate_confidence += 0.10
            activate_evidence.append("tool_workflow_activation")
        add_hypothesis("activate", activate_confidence, activate_evidence)

        inspect_confidence = 0.0
        inspect_evidence: list[str] = []
        if has_inspect_verb:
            inspect_confidence += 0.45
            inspect_evidence.append("inspection_verb")
        if inspection_targets:
            inspect_confidence += 0.25
            inspect_evidence.append("inspection_targets")
        if destination_under_inspection:
            inspect_confidence += 0.10
            inspect_evidence.append("destination_under_inspection")
        add_hypothesis("inspect", inspect_confidence, inspect_evidence)

        inspect_missing_confidence = 0.0
        inspect_missing_evidence: list[str] = []
        if len(portable_mentions) >= 2:
            inspect_missing_confidence += 0.20
            inspect_missing_evidence.append("multiple_named_portables")
        if has_inspect_verb:
            inspect_missing_confidence += 0.15
            inspect_missing_evidence.append("inspection_verb")
        if inspection_targets:
            inspect_missing_confidence += 0.10
            inspect_missing_evidence.append("inspection_targets")
        if destination_under_inspection:
            inspect_missing_confidence += 0.10
            inspect_missing_evidence.append("destination_under_inspection")
        if has_missing_prior:
            inspect_missing_confidence += 0.20
            inspect_missing_evidence.append("missing_semantic_prior")
        if has_recovery_action:
            inspect_missing_confidence += 0.10
            inspect_missing_evidence.append("recovery_action")
        if not destination_pairs and len(portable_mentions) >= 2 and destination_under_inspection:
            inspect_missing_confidence += 0.15
            inspect_missing_evidence.append("closed_set_without_explicit_pairs")
        if explicit_pair_cover and not has_missing_prior:
            inspect_missing_confidence -= 0.20
        add_hypothesis(
            "inspect_missing",
            inspect_missing_confidence,
            inspect_missing_evidence,
        )

        if _PARALLEL_HINT_PATTERN.search(instruction):
            add_hypothesis("parallelize", 0.90, ["parallelize_phrase"])

        if _PUSH_HINT_PATTERN.search(instruction):
            add_hypothesis("push_transfer", 0.95, ["push_transfer_phrase"])

        return hypotheses

    def _select_intents(
        self,
        intent_hypotheses: dict[str, dict[str, object]],
    ) -> list[str]:
        parsed_intents: list[str] = []
        for intent in _INTENT_ORDER:
            hypothesis = intent_hypotheses.get(intent)
            if not isinstance(hypothesis, dict):
                continue
            confidence = hypothesis.get("confidence", 0.0)
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0
            if confidence_value >= _INTENT_CONFIDENCE_THRESHOLDS.get(intent, 0.60):
                parsed_intents.append(intent)
        return _dedupe_preserve_order(parsed_intents)

    def _normalize_parsed_intents(
        self,
        parsed_intents: list[str],
        *,
        inspect_mode: str | None,
    ) -> list[str]:
        if inspect_mode == "executable":
            return _dedupe_preserve_order(parsed_intents)
        return _dedupe_preserve_order(
            [intent for intent in parsed_intents if intent != "inspect"]
        )

    def _infer_inspect_mode(
        self,
        instruction: str,
        *,
        inspection_targets: list[str],
        destination_pairs: list[dict[str, str]],
        parsed_intents: list[str],
    ) -> str | None:
        if not _INSPECT_HINT_PATTERN.search(instruction):
            return None

        manipulate_intents = {
            "relocate",
            "retrieve",
            "conditional_container_check",
            "wash",
            "tool_interaction",
            "activate",
            "push_transfer",
        }
        has_concrete_manipulate_goal = bool(
            destination_pairs
            or set(parsed_intents).intersection(manipulate_intents)
        )
        has_explicit_inspection_target = bool(inspection_targets)

        if has_explicit_inspection_target and not has_concrete_manipulate_goal:
            return "executable"
        if has_concrete_manipulate_goal:
            return "precondition"
        return "ambiguous"

    def _extract_destination_candidates(
        self,
        destination_pairs: list[dict[str, str]],
        *,
        support_targets: list[str],
        container_targets: list[str],
        activation_targets: list[str],
        inspection_targets: list[str],
    ) -> list[str]:
        candidates = [pair["target"] for pair in destination_pairs]
        candidates.extend(
            [
                target
                for target in support_targets
                + container_targets
                + activation_targets
                + inspection_targets
                if entity_can_be_destination(target)
            ]
        )
        return _dedupe_preserve_order(candidates)

    def _select_primary_destination(
        self,
        *,
        parsed_intents: list[str],
        destination_pairs: list[dict[str, str]],
        destination_candidates: list[str],
        activation_targets: list[str],
        inspection_targets: list[str],
        inspect_mode: str | None,
    ) -> str | None:
        if "push_transfer" in parsed_intents:
            for preferred in (
                "sink",
                "table",
                "kitchen work area",
                "kitchen island area",
                "rack",
            ):
                if preferred in destination_candidates:
                    return preferred
        if "wash" in parsed_intents and destination_pairs:
            return destination_pairs[-1]["target"]
        if "tool_interaction" in parsed_intents and "wooden cutting board" in destination_candidates:
            return "wooden cutting board"
        if inspect_mode == "executable" and "inspect" in parsed_intents and inspection_targets:
            return inspection_targets[0]
        if "activate" in parsed_intents and activation_targets:
            for target in activation_targets:
                if entity_can_be_destination(target):
                    return target
        if destination_pairs:
            return destination_pairs[0]["target"]
        return destination_candidates[0] if destination_candidates else None


def build_scene_parser(backend: str) -> BaseSceneParser:
    if backend == "rule":
        return RuleBasedSceneParser()
    if backend == "mat_thor":
        from nsdl.adapters.mat_thor.scene_encoder import MatThorSceneParser

        return MatThorSceneParser()
    raise ValueError(
        f"Unsupported initial scene parser backend: {backend}. "
        "Expected one of: rule, mat_thor."
    )
