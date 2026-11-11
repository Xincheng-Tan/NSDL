from __future__ import annotations

import re

from nsdl.catalog import find_entity_mentions, normalize_entity_name


PROMPT_VARIANT_FULL = "full"
PROMPT_VARIANT_IMAGE_ONLY = "image_only"
PROMPT_VARIANT_FULL_NO_ENTITIES = "full_no_entities"
PROMPT_VARIANT_LEAN = "lean"
PROMPT_VARIANT_LNSDL_AGENTIC_LEAN = "lnsdl_agentic_lean"
PROMPT_VARIANTS = (
    PROMPT_VARIANT_FULL,
    PROMPT_VARIANT_IMAGE_ONLY,
    PROMPT_VARIANT_FULL_NO_ENTITIES,
    PROMPT_VARIANT_LEAN,
    PROMPT_VARIANT_LNSDL_AGENTIC_LEAN,
)


def normalize_prompt_variant(value: str | None) -> str:
    raw = (value or PROMPT_VARIANT_FULL).strip().lower().replace("-", "_")
    aliases = {
        "default": PROMPT_VARIANT_FULL,
        "with_yolo": PROMPT_VARIANT_FULL,
        "imageonly": PROMPT_VARIANT_IMAGE_ONLY,
        "raw_image": PROMPT_VARIANT_IMAGE_ONLY,
        "raw_image_only": PROMPT_VARIANT_IMAGE_ONLY,
        "no_yolo": PROMPT_VARIANT_IMAGE_ONLY,
        "no_entities": PROMPT_VARIANT_FULL_NO_ENTITIES,
        "full_no_entity": PROMPT_VARIANT_FULL_NO_ENTITIES,
        "full_without_entities": PROMPT_VARIANT_FULL_NO_ENTITIES,
        "without_entities": PROMPT_VARIANT_FULL_NO_ENTITIES,
        "lean_no_tools": PROMPT_VARIANT_LEAN,
        "lean_notools": PROMPT_VARIANT_LEAN,
        "lean_no_tool": PROMPT_VARIANT_LEAN,
        "no_tools": PROMPT_VARIANT_LEAN,
        "notools": PROMPT_VARIANT_LEAN,
        "agentic_lean": PROMPT_VARIANT_LNSDL_AGENTIC_LEAN,
        "lnsdl_lean": PROMPT_VARIANT_LNSDL_AGENTIC_LEAN,
        "lnsdl": PROMPT_VARIANT_LNSDL_AGENTIC_LEAN,
    }
    variant = aliases.get(raw, raw)
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {value!r}. Expected one of {PROMPT_VARIANTS}.")
    return variant


def apply_vl_prompt_variant(user_prompt: str, variant: str | None) -> str:
    """Return the VL user prompt for a data/input ablation variant.

    ``full`` preserves the existing prompt, including YOLO-derived text.
    ``image_only`` removes pre-injected perception text while keeping
    instruction, robot inventory, catalog entity tags, robot capabilities, and
    the image.
    ``full_no_entities`` preserves the full perception text and robot anchors
    while removing only the catalog entity tag block.
    ``lean`` is the no-tools baseline input: image + instruction + robot
    inventory/capabilities only, with no pre-injected perception or entity tags.
    """
    normalized = normalize_prompt_variant(variant)
    if normalized == PROMPT_VARIANT_FULL:
        return user_prompt
    if normalized == PROMPT_VARIANT_IMAGE_ONLY:
        return _to_image_only_prompt(user_prompt)
    if normalized == PROMPT_VARIANT_FULL_NO_ENTITIES:
        return _drop_entities_block(user_prompt)
    if normalized == PROMPT_VARIANT_LEAN:
        return _to_lean_prompt(user_prompt)
    if normalized == PROMPT_VARIANT_LNSDL_AGENTIC_LEAN:
        return user_prompt
    raise AssertionError(f"Unhandled prompt variant: {normalized}")


def apply_vl_system_prompt_variant(system_prompt: str, variant: str | None) -> str:
    normalized = normalize_prompt_variant(variant)
    if normalized in {PROMPT_VARIANT_FULL, PROMPT_VARIANT_LNSDL_AGENTIC_LEAN}:
        return system_prompt
    output = system_prompt
    if normalized in {PROMPT_VARIANT_IMAGE_ONLY, PROMPT_VARIANT_LEAN}:
        output = output.replace(
            (
                "`already_at_destination` (the [SCENE] `Entity locations:` line shows the\n"
                "  item is already on/in the destination"
            ),
            (
                "`already_at_destination` (the image shows the\n"
                "  item is already on/in the destination"
            ),
        )
    if normalized in {PROMPT_VARIANT_FULL_NO_ENTITIES, PROMPT_VARIANT_LEAN}:
        output = output.replace(
            (
                "- Open is restricted to openable containers: Emit `open` (and `close`) "
                "ONLY when the target entity carries the `openable` tag in [ENTITIES] "
                "(cabinets, fridges, microwaves, drawers, etc.). NEVER emit `open` on "
                "plain support surfaces such as tables, shelves, sofas, counters; for "
                "those, place items directly via `deposit` without an antecedent open."
            ),
            (
                "- Open is restricted to true openable containers: Emit `open` (and `close`) "
                "ONLY for containers such as cabinets, fridges, microwaves, and drawers. "
                "NEVER emit `open` on plain support surfaces such as tables, shelves, "
                "sofas, or counters; for those, place items directly via `deposit` without "
                "an antecedent open."
            ),
        )
    return output


def _to_image_only_prompt(user_prompt: str) -> str:
    keep_entity_norms, _ = _instruction_entity_filter(user_prompt)
    lines: list[str] = []
    current_section = ""

    for line in user_prompt.splitlines():
        stripped = line.strip()
        section_match = re.fullmatch(r"\[([A-Z_]+)\]", stripped)
        if section_match:
            current_section = section_match.group(1)
            lines.append(line)
            continue

        if current_section == "SCENE":
            if stripped.startswith("Observed entities:") or stripped.startswith("Entity locations:"):
                continue

        if current_section == "ENTITIES" and stripped and ":" in stripped:
            entity_name = stripped.split(":", 1)[0].strip()
            entity_norm = normalize_entity_name(entity_name) or entity_name
            if keep_entity_norms and entity_norm not in keep_entity_norms:
                continue

        lines.append(line)

    return "\n".join(lines).strip()


def _to_lean_prompt(user_prompt: str) -> str:
    user_prompt = _drop_entities_block(user_prompt)
    lines: list[str] = []
    current_section = ""

    for line in user_prompt.splitlines():
        stripped = line.strip()
        section_match = re.fullmatch(r"\[([A-Z_]+)\]", stripped)
        if section_match:
            current_section = section_match.group(1)
            lines.append(line)
            continue

        if current_section == "SCENE" and (
            stripped.startswith("Observed entities:")
            or stripped.startswith("Entity locations:")
        ):
            continue

        lines.append(line)

    return _collapse_blank_runs(lines).strip()


def _drop_entities_block(user_prompt: str) -> str:
    lines: list[str] = []
    skipping_section = False

    for line in user_prompt.splitlines():
        stripped = line.strip()
        section_match = re.fullmatch(r"\[([A-Z_]+)\]", stripped)
        if section_match:
            skipping_section = section_match.group(1) == "ENTITIES"
            if not skipping_section:
                lines.append(line)
            continue
        if skipping_section:
            continue
        lines.append(line)

    return _collapse_blank_runs(lines).strip()


def _collapse_blank_runs(lines: list[str]) -> str:
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return "\n".join(collapsed)


def _instruction_entity_filter(user_prompt: str) -> tuple[set[str], list[str]]:
    instruction = _extract_instruction(user_prompt)
    mentioned_norms = {
        mention.canonical_name
        for mention in find_entity_mentions(instruction)
        if mention.canonical_name
    }
    entity_names = _entity_names_from_prompt(user_prompt)
    if not entity_names:
        return set(), []

    entity_norms = [
        normalize_entity_name(name) or name
        for name in entity_names
    ]
    if mentioned_norms:
        kept = [
            (name, norm)
            for name, norm in zip(entity_names, entity_norms)
            if norm in mentioned_norms
        ]
        if kept:
            return {norm for _, norm in kept}, [name for name, _ in kept]

    return set(entity_norms), entity_names


def _extract_instruction(user_prompt: str) -> str:
    for line in user_prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Instruction:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _entity_names_from_prompt(user_prompt: str) -> list[str]:
    names: list[str] = []
    in_entities = False
    for line in user_prompt.splitlines():
        stripped = line.strip()
        if stripped == "[ENTITIES]":
            in_entities = True
            continue
        if in_entities and re.fullmatch(r"\[[A-Z_]+\]", stripped):
            break
        if not in_entities or not stripped or ":" not in stripped:
            continue
        name = stripped.split(":", 1)[0].strip()
        if name:
            names.append(name)
    return names
