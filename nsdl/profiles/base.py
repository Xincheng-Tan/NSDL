from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from functools import cached_property
import re
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from nsdl.catalog import (
        ActionPrimitive,
        EntityMention,
        EntityProfile,
        NodeKindSpec,
        RobotProfile,
    )


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")

SceneBlockContributor = Callable[[Any, Any], list[str] | None]


@dataclass(frozen=True)
class OpSpec:
    name: str
    description: str
    required_capabilities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    system_prompt_block: str | None = None

    def prompt_block(self) -> str:
        if self.system_prompt_block:
            return self.system_prompt_block
        if not self.required_capabilities:
            return f"- {self.name}: {self.description}"
        capability_text = " + ".join(self.required_capabilities)
        return f"- {self.name}: {self.description} Requires: {capability_text}."


@dataclass(frozen=True)
class OpVocabProfile:
    ops: tuple[OpSpec, ...]
    alias_overrides: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        raw_ops: Mapping[str, str | Mapping[str, Any]],
        *,
        alias_overrides: Mapping[str, str] | None = None,
    ) -> OpVocabProfile:
        ops: list[OpSpec] = []
        for name, raw_value in raw_ops.items():
            if isinstance(raw_value, str):
                ops.append(OpSpec(name=name, description=raw_value))
                continue
            ops.append(
                OpSpec(
                    name=name,
                    description=str(raw_value.get("description", "")).strip(),
                    required_capabilities=tuple(raw_value.get("required_capabilities", ())),
                    aliases=tuple(raw_value.get("aliases", ())),
                    preconditions=tuple(raw_value.get("preconditions", ())),
                    system_prompt_block=raw_value.get("system_prompt_block"),
                )
            )
        return cls(
            ops=tuple(ops),
            alias_overrides=dict(alias_overrides or {}),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            spec.name: spec.description
            for spec in self.ops
        }

    def allowed_op_types(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.ops)

    def alias_map(self) -> dict[str, str]:
        aliases = dict(self.alias_overrides)
        for spec in self.ops:
            for alias in spec.aliases:
                aliases[str(alias).strip().lower()] = spec.name
        return aliases

    def prompt_block(self) -> str:
        return "\n".join(spec.prompt_block() for spec in self.ops)


class CatalogProvider(Protocol):
    @property
    def capability_vocab(self) -> Sequence[str]:
        ...

    @property
    def entity_profiles(self) -> Mapping[str, EntityProfile]:
        ...

    @property
    def robot_profiles(self) -> Mapping[str, RobotProfile]:
        ...

    @property
    def action_primitives(self) -> Mapping[str, ActionPrimitive]:
        ...

    @property
    def node_kind_specs(self) -> Mapping[str, NodeKindSpec]:
        ...

    @property
    def robot_priority(self) -> Mapping[str, int]:
        ...

    @property
    def robot_alias_groups(self) -> Mapping[str, tuple[str, ...]]:
        ...

    def normalize_catalog_text(self, text: str) -> str:
        ...

    def normalize_entity_name(self, text: str | None) -> str | None:
        ...

    def normalize_robot_name(self, text: str | None) -> str | None:
        ...

    def find_entity_mentions(self, text: str) -> list[EntityMention]:
        ...

    def get_entity_profile(self, entity_name: str) -> EntityProfile | None:
        ...

    def get_robot_profile(self, robot_type: str) -> RobotProfile | None:
        ...


@dataclass(frozen=True)
class StaticCatalogProvider:
    capability_vocab: tuple[str, ...]
    entity_profiles: Mapping[str, EntityProfile]
    robot_profiles: Mapping[str, RobotProfile]
    action_primitives: Mapping[str, ActionPrimitive] = field(default_factory=dict)
    node_kind_specs: Mapping[str, NodeKindSpec] = field(default_factory=dict)
    robot_priority: Mapping[str, int] = field(default_factory=dict)
    robot_alias_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @cached_property
    def entity_alias_pairs(self) -> tuple[tuple[str, str], ...]:
        alias_pairs = [
            (alias, canonical)
            for canonical, profile in self.entity_profiles.items()
            for alias in profile.aliases
        ]
        alias_pairs.sort(key=lambda item: (-len(item[0]), item[0]))
        return tuple(alias_pairs)

    @cached_property
    def entity_tags(self) -> dict[str, set[str]]:
        return {
            canonical: set(profile.tags)
            for canonical, profile in self.entity_profiles.items()
        }

    @cached_property
    def robot_alias_to_canonical(self) -> dict[str, str]:
        return {
            self.normalize_catalog_text(alias): canonical
            for canonical, aliases in self.robot_alias_groups.items()
            for alias in aliases
        }

    @cached_property
    def agent_available_actions(self) -> dict[str, list[str]]:
        return {
            robot_type: list(profile.benchmark_actions)
            for robot_type, profile in self.robot_profiles.items()
        }

    @cached_property
    def agent_end_effector_num(self) -> dict[str, int]:
        return {
            robot_type: profile.end_effector_num
            for robot_type, profile in self.robot_profiles.items()
        }

    @cached_property
    def robot_capabilities(self) -> dict[str, set[str]]:
        return {
            robot_type: set(profile.capabilities)
            for robot_type, profile in self.robot_profiles.items()
        }

    def normalize_catalog_text(self, text: str) -> str:
        normalized = text.lower()
        normalized = normalized.replace("—", " ").replace("–", " ")
        normalized = normalized.replace("’", "'")
        normalized = normalized.replace("_", " ")
        normalized = re.sub(r"[^a-z0-9\s']", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def normalize_entity_name(self, text: str | None) -> str | None:
        if text is None:
            return None
        normalized = self.normalize_catalog_text(text)
        if normalized in self.entity_profiles:
            return normalized
        for alias, canonical in self.entity_alias_pairs:
            if normalized == alias:
                return canonical
        return normalized or None

    def normalize_robot_name(self, text: str | None) -> str | None:
        if text is None:
            return None
        normalized = self.normalize_catalog_text(text)
        if normalized in self.robot_alias_to_canonical:
            return self.robot_alias_to_canonical[normalized]
        underscored = normalized.replace(" ", "_")
        if underscored in self.robot_profiles:
            return underscored
        return None

    def find_entity_mentions(self, text: str) -> list[EntityMention]:
        from nsdl.catalog import EntityMention

        normalized = self.normalize_catalog_text(text)
        occupied_spans: list[tuple[int, int]] = []
        mentions: list[EntityMention] = []
        for alias, canonical in self.entity_alias_pairs:
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
            for match in pattern.finditer(normalized):
                span = (match.start(), match.end())
                if any(not (span[1] <= start or span[0] >= end) for start, end in occupied_spans):
                    continue
                occupied_spans.append(span)
                mentions.append(
                    EntityMention(
                        canonical_name=canonical,
                        alias=alias,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        mentions.sort(key=lambda mention: mention.start)
        return mentions

    def get_entity_profile(self, entity_name: str) -> EntityProfile | None:
        canonical = self.normalize_entity_name(entity_name)
        if canonical is None:
            return None
        return self.entity_profiles.get(canonical)

    def get_robot_profile(self, robot_type: str) -> RobotProfile | None:
        canonical = self.normalize_robot_name(robot_type) or robot_type
        return self.robot_profiles.get(canonical)


def _merge_unique_values(*values: Sequence[T]) -> tuple[T, ...]:
    merged: list[T] = []
    seen: set[T] = set()
    for group in values:
        for value in group:
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return tuple(merged)


@dataclass(frozen=True)
class LayeredCatalogProvider:
    base: CatalogProvider
    overlay: CatalogProvider

    @cached_property
    def capability_vocab(self) -> tuple[str, ...]:
        return _merge_unique_values(
            tuple(self.base.capability_vocab),
            tuple(self.overlay.capability_vocab),
        )

    @cached_property
    def entity_profiles(self) -> dict[str, EntityProfile]:
        merged = dict(self.base.entity_profiles)
        merged.update(self.overlay.entity_profiles)
        return merged

    @cached_property
    def robot_profiles(self) -> dict[str, RobotProfile]:
        merged = dict(self.base.robot_profiles)
        merged.update(self.overlay.robot_profiles)
        return merged

    @cached_property
    def action_primitives(self) -> dict[str, ActionPrimitive]:
        merged = dict(self.base.action_primitives)
        merged.update(self.overlay.action_primitives)
        return merged

    @cached_property
    def node_kind_specs(self) -> dict[str, NodeKindSpec]:
        merged = dict(self.base.node_kind_specs)
        merged.update(self.overlay.node_kind_specs)
        return merged

    @cached_property
    def robot_priority(self) -> dict[str, int]:
        merged = dict(self.base.robot_priority)
        merged.update(self.overlay.robot_priority)
        return merged

    @cached_property
    def robot_alias_groups(self) -> dict[str, tuple[str, ...]]:
        merged = dict(self.base.robot_alias_groups)
        for canonical, aliases in self.overlay.robot_alias_groups.items():
            merged[canonical] = tuple(aliases)
        return merged

    @cached_property
    def _merged_static(self) -> StaticCatalogProvider:
        return StaticCatalogProvider(
            capability_vocab=self.capability_vocab,
            entity_profiles=self.entity_profiles,
            robot_profiles=self.robot_profiles,
            action_primitives=self.action_primitives,
            node_kind_specs=self.node_kind_specs,
            robot_priority=self.robot_priority,
            robot_alias_groups=self.robot_alias_groups,
        )

    def normalize_catalog_text(self, text: str) -> str:
        return self._merged_static.normalize_catalog_text(text)

    def normalize_entity_name(self, text: str | None) -> str | None:
        return self._merged_static.normalize_entity_name(text)

    def normalize_robot_name(self, text: str | None) -> str | None:
        return self._merged_static.normalize_robot_name(text)

    def find_entity_mentions(self, text: str) -> list[EntityMention]:
        return self._merged_static.find_entity_mentions(text)

    def get_entity_profile(self, entity_name: str) -> EntityProfile | None:
        return self._merged_static.get_entity_profile(entity_name)

    def get_robot_profile(self, robot_type: str) -> RobotProfile | None:
        return self._merged_static.get_robot_profile(robot_type)


@dataclass(frozen=True)
class SystemPromptProvider:
    intro_block: str
    tag_semantics_block: str
    capability_reference_block: str
    target_status_block: str
    robot_anchor_block: str
    inspection_role_block: str
    pattern_blocks: tuple[str, ...]
    combination_rules_block: str
    output_format_block: str
    contributors: tuple[SceneBlockContributor, ...] = ()

    def build_system_prompt(self, op_vocab: OpVocabProfile) -> str:
        blocks = [
            self.intro_block,
            "## Operation Vocabulary\n" + op_vocab.prompt_block(),
            self.tag_semantics_block,
            self.capability_reference_block,
            self.target_status_block,
            self.robot_anchor_block,
            self.inspection_role_block,
            *self.pattern_blocks,
            self.combination_rules_block,
            self.output_format_block,
        ]
        return "\n\n".join(block for block in blocks if block) + "\n"

    def scene_block_contributors(self) -> tuple[SceneBlockContributor, ...]:
        return self.contributors


class DispatcherEmitter(Protocol):
    def emit(self, payload: Any) -> Any:
        ...


@dataclass(frozen=True)
class IdentityDispatcherEmitter:
    def emit(self, payload: Any) -> Any:
        return payload


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    op_vocab: OpVocabProfile
    catalog: CatalogProvider
    prompt: SystemPromptProvider
    dispatcher_emitter: DispatcherEmitter = field(default_factory=IdentityDispatcherEmitter)
    scene_extras_keys: tuple[str, ...] = ()

    def build_system_prompt(self) -> str:
        return self.prompt.build_system_prompt(self.op_vocab)


class RuntimeMappingProxy(Mapping[K, V]):
    def __init__(self, resolver: Callable[[], Mapping[K, V]]) -> None:
        self._resolver = resolver

    def _mapping(self) -> Mapping[K, V]:
        return self._resolver()

    def __getitem__(self, key: K) -> V:
        return self._mapping()[key]

    def __iter__(self):
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())

    def __repr__(self) -> str:
        return repr(dict(self._mapping()))


class RuntimeSequenceProxy(Sequence[T]):
    def __init__(self, resolver: Callable[[], Sequence[T]]) -> None:
        self._resolver = resolver

    def _sequence(self) -> Sequence[T]:
        return self._resolver()

    def __getitem__(self, index):
        return self._sequence()[index]

    def __len__(self) -> int:
        return len(self._sequence())

    def __iter__(self):
        return iter(self._sequence())

    def __repr__(self) -> str:
        return repr(list(self._sequence()))


class RuntimeSetProxy(Set[T]):
    def __init__(self, resolver: Callable[[], Set[T] | frozenset[T]]) -> None:
        self._resolver = resolver

    def _set(self) -> Set[T] | frozenset[T]:
        return self._resolver()

    def __contains__(self, value: object) -> bool:
        return value in self._set()

    def __iter__(self):
        return iter(self._set())

    def __len__(self) -> int:
        return len(self._set())

    def __repr__(self) -> str:
        return repr(set(self._set()))
