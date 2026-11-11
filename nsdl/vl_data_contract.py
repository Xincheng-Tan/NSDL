from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from nsdl.path_remap import remap_path
from nsdl.vl_prompt_variants import (
    PROMPT_VARIANT_FULL,
    PROMPT_VARIANT_FULL_NO_ENTITIES,
    PROMPT_VARIANT_IMAGE_ONLY,
    PROMPT_VARIANT_LNSDL_AGENTIC_LEAN,
    PROMPT_VARIANT_LEAN,
    normalize_prompt_variant,
)


class VLSFTDataContractError(ValueError):
    """Raised when a VL SFT dataset does not match the configured P0 variant."""


def validate_vl_sft_dataset_contract(
    dataset: Any,
    *,
    data_dir: Path,
    expected_prompt_variant: str | None,
) -> dict[str, Any]:
    """Fail fast when a VL SFT dataset violates the P0 data contract.

    P0 training data must be train-only and its serialized text must match the
    prompt variant selected by the YAML config. This keeps OOD ``val`` and ID
    ``test`` splits evaluation-only and catches stale prebuilt HF datasets.
    """
    required_columns = {"messages", "image_path", "image_source_split", "prompt_variant"}
    missing_columns = sorted(required_columns.difference(set(dataset.column_names)))
    if missing_columns:
        raise VLSFTDataContractError(
            f"{data_dir} is missing required VL SFT columns: {missing_columns}"
        )

    expected_variant = (
        normalize_prompt_variant(expected_prompt_variant)
        if expected_prompt_variant
        else None
    )
    variant_counts: Counter[str] = Counter()
    image_split_counts: Counter[str] = Counter()

    for idx, row in enumerate(dataset):
        row_variant = normalize_prompt_variant(row.get("prompt_variant"))
        variant_counts[row_variant] += 1
        if expected_variant is not None and row_variant != expected_variant:
            raise VLSFTDataContractError(
                f"{data_dir} row {idx} has prompt_variant={row_variant!r}, "
                f"expected {expected_variant!r}."
            )

        image_split = str(row.get("image_source_split") or "")
        image_split_counts[image_split] += 1
        if image_split != "train":
            raise VLSFTDataContractError(
                f"{data_dir} row {idx} uses image_source_split={image_split!r}; "
                "P0 SFT data must be built from VIKI-L2 train only."
            )

        image_path = Path(remap_path(row.get("image_path")))
        if not image_path.is_file():
            raise VLSFTDataContractError(
                f"{data_dir} row {idx} image_path does not exist: {image_path}"
            )

        system_text = _message_text(row, "system")
        user_text = _message_text(row, "user")
        _validate_prompt_text(
            user_text,
            system_text,
            variant=expected_variant or row_variant,
            row_index=idx,
            data_dir=data_dir,
        )

    return {
        "rows": len(dataset),
        "prompt_variants": dict(sorted(variant_counts.items())),
        "image_source_splits": dict(sorted(image_split_counts.items())),
    }


def _message_text(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages") or []
    for message in messages:
        if str(message.get("role")) == role:
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    raise VLSFTDataContractError(f"row is missing {role!r} message")


def _validate_prompt_text(
    user_text: str,
    system_text: str,
    *,
    variant: str,
    row_index: int,
    data_dir: Path,
) -> None:
    if "[SCENE]" not in user_text or "Instruction:" not in user_text:
        _raise_prompt_error(data_dir, row_index, variant, "missing [SCENE]/Instruction")

    if variant == PROMPT_VARIANT_LNSDL_AGENTIC_LEAN:
        if "[ROBOTS]" not in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "missing [ROBOTS]")
        forbidden = ("Observed entities:", "Entity locations:", "[ENTITIES]", "[PLAN]")
        for needle in forbidden:
            if needle in user_text:
                _raise_prompt_error(data_dir, row_index, variant, f"contains {needle}")
        return

    if "[ROBOTS]" not in user_text or "[PLAN]" not in user_text:
        _raise_prompt_error(data_dir, row_index, variant, "missing [ROBOTS] or [PLAN]")

    if variant == PROMPT_VARIANT_FULL:
        if "Entity locations:" not in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "missing Entity locations")
        if "[ENTITIES]" not in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "missing [ENTITIES]")
        return

    if variant in {PROMPT_VARIANT_IMAGE_ONLY, PROMPT_VARIANT_LEAN} and "Entity locations:" in user_text:
        _raise_prompt_error(data_dir, row_index, variant, "contains Entity locations")
    if variant in {PROMPT_VARIANT_IMAGE_ONLY, PROMPT_VARIANT_LEAN} and "[SCENE] `Entity locations:`" in system_text:
        _raise_prompt_error(
            data_dir,
            row_index,
            variant,
            "system prompt still references [SCENE] Entity locations",
        )

    if variant == PROMPT_VARIANT_IMAGE_ONLY:
        if "[ENTITIES]" not in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "missing [ENTITIES]")
        if "Observed entities:" in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "contains Observed entities")
        return

    if variant == PROMPT_VARIANT_FULL_NO_ENTITIES:
        if "Observed entities:" not in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "missing Observed entities")
        if "[ENTITIES]" in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "contains [ENTITIES]")
        if "Entity locations:" not in user_text:
            _raise_prompt_error(data_dir, row_index, variant, "missing Entity locations")
        if " in [ENTITIES]" in system_text:
            _raise_prompt_error(
                data_dir,
                row_index,
                variant,
                "system prompt still requires [ENTITIES]",
            )
        return

    if variant == PROMPT_VARIANT_LEAN:
        forbidden = ("Observed entities:", "[ENTITIES]")
        for needle in forbidden:
            if needle in user_text:
                _raise_prompt_error(data_dir, row_index, variant, f"contains {needle}")
        if " in [ENTITIES]" in system_text:
            _raise_prompt_error(
                data_dir,
                row_index,
                variant,
                "system prompt still requires [ENTITIES]",
            )
        return

    _raise_prompt_error(data_dir, row_index, variant, "unknown prompt variant")

def _raise_prompt_error(
    data_dir: Path,
    row_index: int,
    variant: str,
    reason: str,
) -> None:
    raise VLSFTDataContractError(
        f"{data_dir} row {row_index} does not match prompt_variant={variant!r}: {reason}."
    )
