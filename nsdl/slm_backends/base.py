from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


ChatMessage = Mapping[str, Any]


class SLMBackend(Protocol):
    """Text-generation backend used by the SLM interface.

    Backends receive OpenAI-style chat messages and return the raw assistant
    text. Prompt construction and JSON parsing stay in ``slm_interface``.
    Vision-language backends may additionally consume ``image_bytes`` via
    ``**kwargs``; text backends simply ignore that optional context.
    """

    def generate(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> str:
        ...


def _decoding_hardening_kwargs(
    kwargs: Mapping[str, Any],
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Translate generation_kwargs into HF ``.generate`` decoding controls.

    Shared by the local HF text and VL backends (A1 inference hardening). All
    controls are opt-in; when absent the returned dict yields the historical
    greedy behavior (``do_sample=False``), so existing runs are byte-identical.

    - ``temperature`` > 0 switches to sampling; ``0``/absent stays greedy.
    - ``repetition_penalty`` (e.g. 1.05) damps greedy degeneration loops that
      corrupt long tool-call JSON.
    - ``stop`` / ``stop_strings`` halts generation on hallucinated markup
      (``</pre>``) or a second protocol block; requires a ``tokenizer`` for HF
      ``stop_strings`` support (silently skipped if none is available).
    """
    out: dict[str, Any] = {}
    temperature = kwargs.get("temperature")
    try:
        temp_val = float(temperature) if temperature is not None else 0.0
    except (TypeError, ValueError):
        temp_val = 0.0
    if temp_val and temp_val > 0.0:
        out["do_sample"] = True
        out["temperature"] = temp_val
        top_p = kwargs.get("top_p")
        if top_p is not None:
            out["top_p"] = float(top_p)
    else:
        out["do_sample"] = False

    repetition_penalty = kwargs.get("repetition_penalty")
    if repetition_penalty is not None:
        try:
            rp = float(repetition_penalty)
        except (TypeError, ValueError):
            rp = 1.0
        if rp and rp != 1.0:
            out["repetition_penalty"] = rp

    stop = kwargs.get("stop_strings") or kwargs.get("stop")
    if stop:
        stop_list = [stop] if isinstance(stop, str) else [str(s) for s in stop]
        stop_list = [s for s in stop_list if s]
        if stop_list and tokenizer is not None:
            out["stop_strings"] = stop_list
            out["tokenizer"] = tokenizer
    return out
