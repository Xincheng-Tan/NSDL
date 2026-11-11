"""Static catalog lookup tool for L-NSDL agentic trajectories."""
from __future__ import annotations

from typing import Any, Iterable

from nsdl.catalog import ENTITY_PROFILES, get_entity_profile, normalize_entity_name


def lookup(
    *,
    names: str | Iterable[str] | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Lookup entity profiles by name, or canonical entity names by tag.

    The tool is intentionally static: it does not read images, instructions, GT,
    or scene state. Unknown names are returned as explicit unknown entries so the
    caller can decide whether to retry with another spelling.
    """
    has_names = names is not None
    has_tag = tag is not None and str(tag).strip() != ""
    if has_names == has_tag:
        raise ValueError("lookup requires exactly one of names or tag.")

    if has_tag:
        normalized_tag = _normalize_tag(tag)
        return {
            normalized_tag: [
                name
                for name, profile in ENTITY_PROFILES.items()
                if normalized_tag in set(profile.tags)
            ]
        }

    result: dict[str, Any] = {}
    for raw_name in _coerce_names(names):
        normalized_name = normalize_entity_name(raw_name)
        profile = get_entity_profile(normalized_name) if normalized_name else None
        if profile is None:
            key = str(raw_name).strip()
            result[key] = {
                "tags": [],
                "entity_type": "unknown",
                "normalized_from": raw_name,
                "unknown": True,
            }
            continue
        result[profile.canonical_name] = {
            "tags": list(profile.tags),
            "entity_type": profile.entity_type,
            "normalized_from": raw_name,
        }
    return result


def _normalize_tag(tag: str | None) -> str:
    normalized = str(tag or "").strip().lower().replace(" ", "_")
    if normalized.endswith("s"):
        singular = normalized[:-1]
        known_tags = {
            tag_value
            for profile in ENTITY_PROFILES.values()
            for tag_value in profile.tags
        }
        if singular in known_tags:
            return singular
    return normalized


def _coerce_names(names: str | Iterable[str] | None) -> list[str]:
    if names is None:
        return []
    if isinstance(names, str):
        raw_items = [names]
    else:
        raw_items = [str(name) for name in names]

    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        stripped = item.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result
