"""Generation-agnostic resolution of the HF VL model class for a checkpoint.

P10.S2 (capacity-boundary curve) is a 2x2 grid over *generation* x *scale*:

    Qwen2-VL    2B-Instruct / 7B-Instruct
    Qwen2.5-VL  3B-Instruct / 7B-Instruct

The 2x2 (rather than a single 2B->3B->7B line) exists precisely so the capacity
axis is not confounded with generation, which means both generations must be
trained and evaluated through the *same* code path -- a hard-pinned
``Qwen2_5_VLForConditionalGeneration`` in the trainer/backend would silently
force a second, non-comparable path for the older generation (and in fact just
fails to load it: the two architectures differ in the vision tower).

Dispatch is on the checkpoint's own ``config.model_type`` rather than on a
filename or a caller-supplied flag, so a mislabeled run directory cannot cause a
silent generation mixup.
"""

from __future__ import annotations

from typing import Any

# model_type -> (model class name, FSDP decoder-layer class name). Intentionally
# an explicit allow-list rather than AutoModelForVision2Seq: an unrecognised
# checkpoint must fail loudly here (with the supported set named) instead of
# loading under some generic mapping whose preprocessing contract we have not
# verified.
#
# The decoder-layer name is stored, NOT derived. Qwen2/Qwen2.5-VL happen to obey
# `ForConditionalGeneration -> DecoderLayer`, but Qwen3-VL does not: its layer is
# `Qwen3VLTextDecoderLayer` (the text stack moved under a `*TextModel`), so the
# old string transform would have produced a name that matches nothing. FSDP does
# not error on an unmatched wrap target -- it just wraps nothing, so the run
# trains un-sharded and OOMs hours in. Keeping this as data is what makes a wrong
# name a startup failure instead.
_SUPPORTED: dict[str, tuple[str, str]] = {
    "qwen2_vl": ("Qwen2VLForConditionalGeneration", "Qwen2VLDecoderLayer"),
    "qwen2_5_vl": ("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLDecoderLayer"),
    "qwen3_vl": ("Qwen3VLForConditionalGeneration", "Qwen3VLTextDecoderLayer"),
}


def _too_old(model_path: str, detail: str) -> ValueError:
    """Turn 'this transformers is too old' into an instruction, not a stack trace.

    Loading a Qwen3-VL checkpoint under the legacy NSDL env (transformers 4.51.3)
    fails inside ``AutoConfig`` with a bare ``KeyError: 'qwen3_vl'``, which reads
    like a corrupt checkpoint rather than a wrong environment. Both cards get
    held for the seconds it takes to find that out, and the real fix -- run in
    NSDL3 -- is nowhere in the message.
    """
    import transformers

    return ValueError(
        f"{detail} (checkpoint {model_path!r}); installed transformers is "
        f"{transformers.__version__} and Qwen3-VL needs >=4.57.0 -- run in the "
        f"NSDL3 env, not NSDL"
    )


def resolve_vl_model_type(model_path: str) -> str:
    """Read ``model_type`` from a checkpoint's config."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except KeyError as exc:  # bare unknown-model_type, older transformers
        raise _too_old(model_path, f"transformers does not know model_type {exc.args[0]!r}") from exc
    except ValueError as exc:
        # 4.51.3 catches its own KeyError and re-raises this. Its advice ("pip
        # install --upgrade transformers") is the wrong fix here -- upgrading in
        # place would swap the inference stack out from under the unfinished S2
        # appendix buckets. Match narrowly so unrelated ValueErrors still surface.
        if "does not recognize this architecture" not in str(exc):
            raise
        raise _too_old(model_path, "transformers does not recognize this architecture") from exc
    return str(getattr(config, "model_type", "") or "")


def resolve_vl_model_class(model_path: str) -> Any:
    """Return the HF model class matching this checkpoint's generation."""
    model_type = resolve_vl_model_type(model_path)
    entry = _SUPPORTED.get(model_type)
    if entry is None:
        supported = ", ".join(sorted(_SUPPORTED))
        raise ValueError(
            f"Unsupported VL model_type {model_type!r} for checkpoint {model_path!r}; "
            f"supported: {supported}"
        )
    import transformers

    class_name = entry[0]
    cls = getattr(transformers, class_name, None)
    if cls is None:
        # Reached when the allow-list knows the generation but the installed
        # transformers does not. Separate from the AutoConfig guard above: a
        # version could in principle register the config and not export the
        # class.
        raise _too_old(model_path, f"transformers has no {class_name!r}")
    return cls


def fsdp_layer_cls_for_model_type(model_type: str) -> str:
    """Decoder-layer class name to hand FSDP ``transformer_layer_cls_to_wrap``.

    The 7B configs name this explicitly; this helper exists so a config that
    names the wrong generation's layer class can be caught before a multi-hour
    run rather than after.
    """
    entry = _SUPPORTED.get(model_type)
    if entry is None:
        supported = ", ".join(sorted(_SUPPORTED))
        raise ValueError(f"Unsupported VL model_type {model_type!r}; supported: {supported}")
    return entry[1]
