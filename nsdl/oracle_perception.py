"""P5.10.4 oracle-perception bracket backend — **eval-only**.

Generates ``detect``-shaped observations directly from a sample's parquet GT
initial state (``ground_truth.init_pos``), bypassing YOLO. Used exclusively for
the ``DETECT_BACKEND ∈ {off, yolo, oracle}`` three-point ablation that bounds
how much of the reported performance could come from the perception channel
([idea.md 感知证据通道定位约定] / [plan.md P5.10.4]).

Anti-leak guards (both enforced):
1. Importing this module requires ``NSDL_ORACLE_PERCEPTION=eval-only`` in the
   environment; the eval runner sets it right before import. Training / data
   construction / RL processes never set it, so an accidental import raises.
2. ``scripts/build_lnsdl_sft_data.py`` and ``scripts/prepare_rl_data.py``
   assert this module is absent from ``sys.modules`` at build time.

Scope modes (env ``LNSDL_ORACLE_SCOPE``, paper weakness W3, 2026-09-07):

``query`` (default, byte-identical to every bucket produced before this date)
    The oracle answers only what the model asked for, with ``detect``'s own
    ``_build_query_scope`` semantics. Memory ``d3-oracle-not-an-upper-bound``:
    this returns FEWER entities than YOLO in 37/37 cells, so a low oracle
    score cannot separate "judgment is the bottleneck" from "the required
    fact never entered the context".
``full``
    The oracle returns the GT relations of EVERY instruction-relevant entity
    regardless of the model's query: entities named in the instruction, the
    parsed scene-facts lists (mentioned_objects / destination_candidates /
    activation_targets / tool_items / ...), the benchmark's own canonical
    observation query (``locus_meta.observation_tool``, e.g. S3 quantifier
    tags -> every instance of the tag), unioned with whatever the model asked
    for (so ``full`` is a strict superset of ``query`` per call). Two extra GT
    locations (``rack``, ``cardboardbox``) are mapped in this mode only, so
    that F5 "already stored" / F9 carrier facts are expressible at all. The
    per-sample ``oracle_scope_detail`` (recorded in the eval artifact) lists
    ``unobservable_entities`` = in-scope entities sealed in an opaque cabinet
    (S4 realization): facts a single frame could not show, which the W3
    analysis splits out.

Vocabulary modes (env ``LNSDL_ORACLE_VOCAB``, paper weakness W4, 2026-09-11):

``gt`` (default, byte-identical to every bucket produced before this date)
    Relations use the GT-location vocabulary above (``in sink``,
    ``at toaster``, ``at kitchen work area``, and in ``full`` mode ``on rack``
    / ``in cardboardbox``). Several of these word forms never occur in the
    training corpus, so a low oracle score confounds "wrong facts" with
    "unfamiliar interface" (memory ``w3-full-oracle-paired-reading``).
``yolo``
    Relations are restricted to the word forms the YOLO relation builder
    (``scene_facts_refiner._build_scene_relations``) can emit and the model
    was trained on: ``on table`` / ``in cabinet`` / ``on sink`` /
    ``in drawer`` / ``in fridge`` / ``on toaster`` (corpus count over
    ``gpt56_terra_locus_train_v341_schedall``: 1997 / 599 / 142 / 81 / 17 /
    14 rows; ``at …`` never occurs). GT locations YOLO cannot express
    (``kitchen work area``, ``kitchen island area``, ``rack``,
    ``cardboardbox``) yield NO relation, exactly as YOLO stays silent about
    counter-top items; the entity itself is still listed as present. Scope
    semantics are untouched, so ``query`` + ``yolo`` differs from the YOLO
    bucket in relation correctness only (the crop channel is absent in every
    oracle mode, so the matched YOLO partner is the crop-off bucket).
"""
from __future__ import annotations

import os
import re
from typing import Any

if os.environ.get("NSDL_ORACLE_PERCEPTION") != "eval-only":
    raise ImportError(
        "nsdl.oracle_perception is an eval-only bracket backend. It must never "
        "be imported from training, data-construction, or RL code paths. Set "
        "NSDL_ORACLE_PERCEPTION=eval-only explicitly in the eval runner to use it."
    )

from nsdl.catalog import normalize_entity_name  # noqa: E402
from nsdl.perc_tool import (  # noqa: E402
    DetectResult,
    _build_query_scope,
    _entity_matches_scope,
    _relation_matches_scope,
)

_INSTANCE_SUFFIX_RE = re.compile(r"_\d+$")

ORACLE_SCOPE_ENV = "LNSDL_ORACLE_SCOPE"
ORACLE_SCOPE_MODES = ("query", "full")
ORACLE_VOCAB_ENV = "LNSDL_ORACLE_VOCAB"
ORACLE_VOCAB_MODES = ("gt", "yolo")


def oracle_scope_mode() -> str:
    """``LNSDL_ORACLE_SCOPE`` -> ``query`` (default) | ``full``; rejects typos."""
    raw = (os.environ.get(ORACLE_SCOPE_ENV) or "query").strip().lower()
    if raw not in ORACLE_SCOPE_MODES:
        raise ValueError(
            f"{ORACLE_SCOPE_ENV}={raw!r} is not one of {ORACLE_SCOPE_MODES}"
        )
    return raw


def oracle_vocab_mode() -> str:
    """``LNSDL_ORACLE_VOCAB`` -> ``gt`` (default) | ``yolo``; rejects typos."""
    raw = (os.environ.get(ORACLE_VOCAB_ENV) or "gt").strip().lower()
    if raw not in ORACLE_VOCAB_MODES:
        raise ValueError(
            f"{ORACLE_VOCAB_ENV}={raw!r} is not one of {ORACLE_VOCAB_MODES}"
        )
    return raw

# GT ``init_pos`` location -> (verb, fixture) in the same observation vocabulary
# emitted by the YOLO relation builder (on/in/at + canonical fixture names).
# Robot-held items (R1..), carrier slots (groundA/mask1/pandaA/B) and the vague
# "room" location have no fixture-relation equivalent and are skipped.
_LOCATION_RELATIONS: dict[str, tuple[str, str]] = {
    "table": ("on", "table"),
    "cabinet": ("in", "cabinet"),
    "room_cabinet": ("in", "cabinet"),
    "sink": ("in", "sink"),
    "kitchen work area": ("at", "kitchen work area"),
    "kitchen island area": ("at", "kitchen island area"),
    "room_toaster": ("at", "toaster"),
}

# ``full`` scope only. ``rack`` is the F5 "already stored" placement (73/560
# eval targets sit on it) and ``cardboardbox`` the F9 carrier; neither has a
# YOLO relation equivalent, so the ``query`` bracket never mentioned them and
# keeps not mentioning them (byte-identical default).
_FULL_EXTRA_LOCATION_RELATIONS: dict[str, tuple[str, str]] = {
    "rack": ("on", "rack"),
    "cardboardbox": ("in", "cardboardbox"),
}

# ``yolo`` vocabulary: only word forms ``_build_scene_relations`` produces
# (``in`` for openable fixtures, ``on`` otherwise; never ``at``). Locations
# with no YOLO-expressible fixture are absent on purpose (no relation emitted);
# ``full`` mode adds nothing on top of this map.
_YOLO_VOCAB_LOCATION_RELATIONS: dict[str, tuple[str, str]] = {
    "table": ("on", "table"),
    "cabinet": ("in", "cabinet"),
    "room_cabinet": ("in", "cabinet"),
    "sink": ("on", "sink"),
    "room_toaster": ("on", "toaster"),
}


def location_relations_for(scope_mode: str, vocab_mode: str) -> dict[str, tuple[str, str]]:
    """Location map for a (scope, vocab) pair; ``("query", "gt")`` is the frozen default."""
    if vocab_mode == "yolo":
        return dict(_YOLO_VOCAB_LOCATION_RELATIONS)
    if scope_mode == "full":
        return {**_LOCATION_RELATIONS, **_FULL_EXTRA_LOCATION_RELATIONS}
    return dict(_LOCATION_RELATIONS)

# Scene-facts list fields that name instruction-relevant entities.
_SCENE_FACTS_ENTITY_FIELDS = (
    "mentioned_objects",
    "destination_candidates",
    "activation_targets",
    "tool_items",
    "inspection_targets",
    "source_locations",
    "fallback_locations",
    "target_items",
    "targets_in_cabinet",
    "secondary_objects",
)
_SCENE_FACTS_SCALAR_FIELDS = ("primary_object", "destination")


def _instance_base_name(instance: str) -> str | None:
    base = _INSTANCE_SUFFIX_RE.sub("", str(instance).strip())
    return normalize_entity_name(base) or (base.lower() or None)


def build_oracle_observations(
    ground_truth: dict[str, Any],
    *,
    location_relations: dict[str, tuple[str, str]] | None = None,
    skip_self_relations: bool = False,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Deterministic ``(entities, relation_triples)`` from GT initial state.

    ``location_relations`` defaults to the frozen ``query``-mode vocabulary;
    ``full`` mode passes the extended one. ``skip_self_relations`` drops
    ``cabinet in cabinet``-style triples (the YOLO builder's ``_emit`` never
    produces ``ent == fix``); the ``gt`` vocabulary keeps them for parity with
    the stored buckets.
    """
    location_map = _LOCATION_RELATIONS if location_relations is None else location_relations
    init_pos = (ground_truth or {}).get("init_pos") or {}
    entities: list[str] = []
    triples: list[tuple[str, str, str]] = []

    def _add_entity(name: str) -> None:
        if name and name not in entities:
            entities.append(name)

    for instance, locations in init_pos.items():
        entity = _instance_base_name(instance)
        if not entity:
            continue
        if locations is None:
            continue
        if hasattr(locations, "tolist"):
            locations = locations.tolist()
        if isinstance(locations, str):
            locations = [locations]
        _add_entity(entity)
        for location in locations or []:
            mapped = location_map.get(str(location).strip())
            if mapped is None:
                continue
            verb, fixture = mapped
            if skip_self_relations and fixture == entity:
                continue
            _add_entity(fixture)
            triple = (entity, verb, fixture)
            if triple not in triples:
                triples.append(triple)
    return entities, triples


def _scene_facts_field(scene_facts: Any, name: str) -> Any:
    if scene_facts is None:
        return None
    if isinstance(scene_facts, dict):
        return scene_facts.get(name)
    return getattr(scene_facts, name, None)


def _merge_scope(*scopes: dict[str, set[str]]) -> dict[str, set[str]]:
    entities: set[str] = set()
    tags: set[str] = set()
    unscoped = False
    for scope in scopes:
        if not scope:
            continue
        entities |= set(scope.get("entities") or ())
        tags |= set(scope.get("tags") or ())
        unscoped = unscoped or bool(scope.get("unscoped"))
    return {"entities": entities, "tags": tags, "unscoped": {"true"} if unscoped else set()}


def build_full_scope(
    ground_truth: dict[str, Any],
    instruction: str | None = None,
    scene_facts: Any = None,
) -> dict[str, set[str]]:
    """Instruction-relevant scope for ``full`` mode (no model query involved).

    Sources, all instruction/GT-metadata derived and independent of what the
    model asks: (1) entity mentions + quantifier tags in the instruction text
    (same extractor ``detect`` applies to a query string, so "every fruit"
    expands to every fruit entity); (2) the parsed scene-facts entity lists;
    (3) ``locus_meta.observation_tool.args.query`` -- the benchmark's own
    canonical observation for the cell (S3 ``{"tag": ...}``). The instruction
    is read from ``ground_truth.description`` when not passed explicitly.
    """
    ground_truth = ground_truth or {}
    text = instruction if instruction is not None else str(ground_truth.get("description") or "")
    parts: list[dict[str, set[str]]] = []
    if text.strip():
        parts.append(_build_query_scope(text))

    names: set[str] = set()
    for field in _SCENE_FACTS_ENTITY_FIELDS:
        values = _scene_facts_field(scene_facts, field)
        if isinstance(values, str):
            values = [values]
        for value in values or []:
            if isinstance(value, str) and value.strip():
                names.add(value)
    for field in _SCENE_FACTS_SCALAR_FIELDS:
        value = _scene_facts_field(scene_facts, field)
        if isinstance(value, str) and value.strip():
            names.add(value)
    if names:
        parts.append(_build_query_scope(sorted(names)))

    meta = ground_truth.get("locus_meta") or {}
    observation_tool = meta.get("observation_tool") if isinstance(meta, dict) else None
    if isinstance(observation_tool, dict):
        args = observation_tool.get("args") or {}
        canonical_query = args.get("query") if isinstance(args, dict) else None
        if canonical_query:
            parts.append(_build_query_scope(canonical_query))

    merged = _merge_scope(*parts)
    # A part that matched nothing (e.g. a scene-facts list of non-catalog
    # words) must not widen the scope to "everything"; only a scope that is
    # empty as a whole falls back to unscoped (= whole GT scene), so the
    # ``full`` condition stays instruction-bounded and comparable across samples.
    merged["unscoped"] = set() if (merged["entities"] or merged["tags"]) else {"true"}
    return merged


def _select_in_scope(
    all_entities: list[str],
    all_triples: list[tuple[str, str, str]],
    scope: dict[str, set[str]],
) -> DetectResult:
    selected_entities: list[str] = []
    selected_relations: list[str] = []
    confidences: dict[str, float] = {}

    def _include(entity: str) -> None:
        if entity not in selected_entities:
            selected_entities.append(entity)
        confidences[entity] = 1.0

    for entity, verb, fixture in all_triples:
        if not _relation_matches_scope(entity, fixture, scope):
            continue
        relation = f"{entity} {verb} {fixture}"
        if relation not in selected_relations:
            selected_relations.append(relation)
        _include(entity)
        _include(fixture)
        confidences[relation] = 1.0

    for entity in all_entities:
        if _entity_matches_scope(entity, scope):
            _include(entity)

    return DetectResult(
        entities=selected_entities,
        relations=selected_relations,
        confidences=confidences,
    )


def _full_observations(
    ground_truth: dict[str, Any],
    vocab_mode: str = "gt",
) -> tuple[list[str], list[tuple[str, str, str]]]:
    return build_oracle_observations(
        ground_truth,
        location_relations=location_relations_for("full", vocab_mode),
        skip_self_relations=(vocab_mode == "yolo"),
    )


def oracle_detect_full(
    ground_truth: dict[str, Any],
    query: Any = None,
    *,
    full_scope: dict[str, set[str]] | None = None,
    vocab_mode: str | None = None,
) -> DetectResult:
    """``full``-mode observation: instruction-relevant scope ∪ the model's query.

    The model's query keeps ``detect`` semantics, so an empty/unscoped query
    still means "everything" (as in ``query`` mode); the union guarantees
    ``full(q) ⊇ query(q)`` for every q.
    """
    if full_scope is None:
        full_scope = build_full_scope(ground_truth)
    vocab = vocab_mode or oracle_vocab_mode()
    all_entities, all_triples = _full_observations(ground_truth, vocab)
    scope = _merge_scope(full_scope, _build_query_scope(query))
    return _select_in_scope(all_entities, all_triples, scope)


def describe_full_scope(
    ground_truth: dict[str, Any],
    full_scope: dict[str, set[str]],
    vocab_mode: str | None = None,
) -> dict[str, Any]:
    """Per-sample provenance for the eval artifact (``oracle_scope_detail``).

    ``unobservable_entities``: in-scope entities whose GT placement is inside
    a cabinet while ``locus_meta.opaque_container`` is true -- the S4 hidden
    realization, which no single frame can show. ``hidden_in_cabinet_entities``
    is the superset without the opacity condition (glass doors are not a
    visibility guarantee either: memory ``locus-glass-queue-overflags-layout5``),
    reported so the analysis can pick the conservative or the strict split.
    """
    # Instruction-relevant part only (no model query merged in).
    vocab = vocab_mode or oracle_vocab_mode()
    result = _select_in_scope(*_full_observations(ground_truth, vocab), full_scope)
    meta = (ground_truth or {}).get("locus_meta") or {}
    opaque = bool(meta.get("opaque_container")) if isinstance(meta, dict) else False
    in_cabinet: list[str] = []
    for relation in result.relations:
        parts = relation.split(" ")
        if len(parts) >= 3 and parts[-2] == "in" and parts[-1] == "cabinet":
            entity = " ".join(parts[:-2])
            if entity != "cabinet" and entity not in in_cabinet:
                in_cabinet.append(entity)
    return {
        "scope_entities": sorted(full_scope.get("entities") or ()),
        "scope_tags": sorted(full_scope.get("tags") or ()),
        "scope_unscoped": bool(full_scope.get("unscoped")),
        "full_entities": list(result.entities),
        "full_relations": list(result.relations),
        "hidden_in_cabinet_entities": in_cabinet,
        "unobservable_entities": list(in_cabinet) if opaque else [],
        "opaque_container": opaque,
        "location_vocab_extended": (
            [] if vocab == "yolo" else sorted(_FULL_EXTRA_LOCATION_RELATIONS)
        ),
        "oracle_vocab": vocab,
    }


def oracle_detect(
    ground_truth: dict[str, Any],
    query: Any = None,
    *,
    scope_mode: str | None = None,
    full_scope: dict[str, set[str]] | None = None,
    vocab_mode: str | None = None,
) -> DetectResult:
    """Query-scoped oracle observation with the same scope semantics as detect.

    Interface-compatible with ``perc_tool.detect``'s return value; confidences
    are pinned to 1.0 and no crops are ever produced (bracket is relations-only).
    ``scope_mode`` defaults to env ``LNSDL_ORACLE_SCOPE`` (``query`` unless set);
    ``vocab_mode`` to env ``LNSDL_ORACLE_VOCAB`` (``gt`` unless set).
    """
    mode = scope_mode or oracle_scope_mode()
    vocab = vocab_mode or oracle_vocab_mode()
    if mode == "full":
        return oracle_detect_full(ground_truth, query, full_scope=full_scope, vocab_mode=vocab)
    all_entities, all_triples = build_oracle_observations(
        ground_truth,
        location_relations=location_relations_for("query", vocab),
        skip_self_relations=(vocab == "yolo"),
    )
    scope = _build_query_scope(query)

    selected_entities: list[str] = []
    selected_relations: list[str] = []
    confidences: dict[str, float] = {}

    def _include(entity: str) -> None:
        if entity not in selected_entities:
            selected_entities.append(entity)
        confidences[entity] = 1.0

    for entity, verb, fixture in all_triples:
        if not _relation_matches_scope(entity, fixture, scope):
            continue
        relation = f"{entity} {verb} {fixture}"
        if relation not in selected_relations:
            selected_relations.append(relation)
        _include(entity)
        _include(fixture)
        confidences[relation] = 1.0

    for entity in all_entities:
        if _entity_matches_scope(entity, scope):
            _include(entity)

    return DetectResult(
        entities=selected_entities,
        relations=selected_relations,
        confidences=confidences,
    )


def make_oracle_detect_override(
    ground_truth: dict[str, Any],
    *,
    scope_mode: str | None = None,
    instruction: str | None = None,
    scene_facts: Any = None,
    vocab_mode: str | None = None,
):
    """Per-sample ``detect_override`` callable for the tool runtime.

    The callable carries ``oracle_scope`` (mode), ``oracle_vocab`` and, in
    ``full`` mode, ``oracle_scope_detail`` (see ``describe_full_scope``) so the
    eval runner can record them in the artifact without re-deriving them.
    """
    mode = scope_mode or oracle_scope_mode()
    vocab = vocab_mode or oracle_vocab_mode()
    full_scope = None
    detail = None
    if mode == "full":
        full_scope = build_full_scope(ground_truth, instruction, scene_facts)
        detail = describe_full_scope(ground_truth, full_scope, vocab_mode=vocab)

    def _override(query: Any = None, crop_entities: Any = None) -> DetectResult:  # noqa: ARG001
        return oracle_detect(
            ground_truth, query, scope_mode=mode, full_scope=full_scope, vocab_mode=vocab
        )

    _override.oracle_scope = mode  # type: ignore[attr-defined]
    _override.oracle_vocab = vocab  # type: ignore[attr-defined]
    _override.oracle_scope_detail = detail  # type: ignore[attr-defined]
    return _override
