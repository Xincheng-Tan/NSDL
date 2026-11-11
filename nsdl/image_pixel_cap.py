"""Canonical input-resolution cap, shared by eval / SFT / RL (A0b, 2026-08-27).

Qwen2.5-VL does not resize to a fixed grid: ``smart_resize`` keeps the native
aspect ratio and only clamps total pixels into ``[min_pixels, max_pixels]``.
Every checkpoint we use ships ``max_pixels = 12,845,056`` in its
``preprocessor_config.json`` -- 6x the pixel count of a LOCUS render -- so LOCUS
images pass through at 1920x1080 = **2,691 vision tokens** against VIKI-L2's 252
at 600x337. That one number is what pushed the LOCUS SFT past 80 GiB and what
puts an RL prompt (5,854 tok) over verl's ``max_prompt_length=4096``.

Measured cap -> vision tokens for a 1920x1080 render. Note ``smart_resize``
**floors** on the way down (``floor(dim / beta / 28) * 28``), so you cannot take
the token count of an image *rendered* at the equivalent size and assume it
carries over:

    unset -> 2,691 | 921,600 -> 1,125 (~1260x700) | 524,288 -> 646 (~952x532)

UNSET IS THE DEFAULT AND MUST STAY SO. Every bucket under
``results/lnsdl/v3_3/locus/`` produced before 2026-08-27 -- the P10.S2 capacity
grid, the P6.1 four-point table, the RL step curve, the G5 oracle bracket -- ran
at full resolution. Turning this on by default would make new numbers silently
incomparable with all of them.

Empirical basis for capping at all (chain i, 2026-08-26/27): paired McNemar over
560 LOCUS samples, two ckpt families, temperature 0.0 -- three arms, all
"not significant", **all three point estimates negative** (-1.79 / -0.89 /
-1.96 pt). The pre-registered rule reads that as PASS, but PASS here means "no
detectable cost within +-3.7 pt", not "no cost": 34 discordant pairs give ~30%
power against a 2 pt effect. Cite the interval, never the p-value alone.
"""

from __future__ import annotations

import os
from typing import Any

ENV_IMAGE_MAX_PIXELS = "LNSDL_IMAGE_MAX_PIXELS"


def resolve_image_pixel_cap() -> int | None:
    """Return the configured cap, or ``None`` when unset.

    Raises rather than falling back to full resolution: a typo'd cap that
    silently evaluated at 2,691 tokens under a bucket named ``res524288`` would
    be indistinguishable from a real PASS.
    """
    raw = os.environ.get(ENV_IMAGE_MAX_PIXELS, "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_IMAGE_MAX_PIXELS}={raw!r} is not an integer") from exc
    if cap <= 0:
        raise ValueError(f"{ENV_IMAGE_MAX_PIXELS}={cap} must be positive")
    return cap


def apply_image_pixel_cap(processor: Any, *, tag: str = "cap") -> int | None:
    """Apply the cap to a processor in place. Returns the cap, or ``None``."""
    cap = resolve_image_pixel_cap()
    if cap is None:
        return None

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError(
            f"{ENV_IMAGE_MAX_PIXELS} is set but this processor exposes no "
            "image_processor -- refusing to run at full resolution under a name "
            "that claims otherwise."
        )
    min_pixels = int(getattr(image_processor, "min_pixels", 0) or 0)
    if cap < min_pixels:
        raise ValueError(
            f"{ENV_IMAGE_MAX_PIXELS}={cap} is below the processor's "
            f"min_pixels={min_pixels}; smart_resize would scale images back UP."
        )
    image_processor.max_pixels = cap
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and "longest_edge" in size:
        # transformers>=4.49 resolves the cap through ``size`` when it is present,
        # so setting only the scalar attribute leaves it at full resolution.
        size["longest_edge"] = cap
    print(f"[{tag}] {ENV_IMAGE_MAX_PIXELS}={cap} applied (min_pixels={min_pixels})", flush=True)
    return cap


def read_effective_image_pixel_cap(processor: Any) -> int | None:
    """Return the cap a processor will ACTUALLY enforce, or ``None`` if none.

    Why this exists (A0b, 2026-08-29). ``agentic_eval.image_max_pixels`` used to
    record ``LNSDL_IMAGE_MAX_PIXELS`` -- the *request*, not the *effect*. Those
    two diverge whenever the cap rides in on the checkpoint instead of the env:
    ``save_pretrained`` writes the capped ``max_pixels`` into
    ``preprocessor_config.json``, so a later eval that (correctly) does not
    re-export the env var still runs capped. The G1 main-row bucket
    ``locus_g1_mainrow_cap524288_7b_seed42`` hit exactly this: 560 rows recorded
    ``None`` while running at 646 vision tokens, and the field's own docstring
    said ``None == full 1920x1080``. Anyone reading that bucket concludes the
    opposite of the truth.

    Precedence mirrors ``apply_image_pixel_cap`` and, under it, transformers:
    when ``size["longest_edge"]`` is present it is what ``smart_resize`` honours,
    so it wins over the scalar attribute. A disagreement between the two is
    reported as the *effective* one rather than averaged or silently preferred --
    but it is a real inconsistency, so callers that care should compare against
    ``max_pixels`` themselves.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None

    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and size.get("longest_edge") is not None:
        return int(size["longest_edge"])

    max_pixels = getattr(image_processor, "max_pixels", None)
    return None if max_pixels is None else int(max_pixels)


def describe_image_pixel_cap(processor: Any) -> dict[str, Any]:
    """Build the ``agentic_eval`` cap-provenance block.

    ``schema=2`` is the marker that separates buckets written after the fix from
    the env-only ones. It is deliberately explicit: a schema-1 bucket carrying
    ``image_max_pixels: null`` is ambiguous between "ran at full resolution" and
    "ran capped via the checkpoint and we failed to notice", and no amount of
    reading the other fields can tell those apart after the fact.
    """
    env_cap = resolve_image_pixel_cap()
    effective = read_effective_image_pixel_cap(processor)

    if effective is None:
        source = "unavailable"
    elif env_cap is not None and effective == env_cap:
        source = "env"
    else:
        source = "checkpoint"

    return {
        "image_max_pixels": effective,
        "image_max_pixels_env": env_cap,
        "image_max_pixels_source": source,
        "image_pixel_cap_schema": 2,
    }
