from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from nsdl.slm_backends.base import ChatMessage, _decoding_hardening_kwargs


# The cap itself lives in nsdl.image_pixel_cap so eval / SFT / RL share one
# implementation -- three copies of "clamp max_pixels" would drift, and a drift
# here is invisible (it produces a bucket named `res524288` that ran at full
# resolution). Re-exported under the old names because tests/test_image_pixel_cap.py
# and the call site below reference them.
from nsdl.image_pixel_cap import ENV_IMAGE_MAX_PIXELS, apply_image_pixel_cap


def _apply_image_pixel_cap(processor: Any) -> int | None:
    return apply_image_pixel_cap(processor, tag="local_vl")


def _split_content(content: Any) -> tuple[str, list[Any]]:
    """Split a chat ``content`` into (text, images).

    Images come from parts-list content (``{"type": "image", "image": ...}``),
    e.g. a ``detect`` tool turn carrying crop observations (P5.10.1). A plain
    string yields no images (historical behavior unchanged).
    """
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        texts: list[str] = []
        images: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    texts.append(str(item.get("text", "")))
                elif item_type in {"image", "image_url"}:
                    image = item.get("image")
                    if image is None:
                        image = item.get("image_url")
                    if image is not None:
                        images.append(image)
                else:
                    texts.append(str(item.get("content", "")))
            else:
                texts.append(str(item))
        return "\n".join(part for part in texts if part).strip(), images
    return (str(content) if content is not None else ""), []


def _content_to_text(content: Any) -> str:
    """Text-only view of a chat content (drops images). Used by text collators."""
    return _split_content(content)[0]


@dataclass
class LocalVisionTextSLMBackend:
    """Locally loaded Qwen2.5-VL backend for SFT/GRPO checkpoints."""

    supports_correction_retry: bool = True
    accepts_image_input: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _model_path: str | None = field(default=None, init=False)

    def load(self, weights_path: str, *, device: str = "cuda") -> None:
        """Load and cache a local Qwen-VL-style instruct checkpoint.

        Dispatches on the checkpoint's own ``model_type`` (see
        ``nsdl.vl_model_loader``) so that P10.S2's cross-generation capacity
        grid (Qwen2-VL 2B/7B vs Qwen2.5-VL 3B/7B) evaluates through this exact
        code path — pinning ``Qwen2_5_VLForConditionalGeneration`` here would
        force a second, non-comparable eval path for the older generation.
        """
        if self._model is not None and self._model_path == weights_path:
            return

        import torch
        from transformers import AutoProcessor

        from nsdl.vl_model_loader import resolve_vl_model_class

        self._processor = AutoProcessor.from_pretrained(
            weights_path,
            trust_remote_code=True,
        )
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _apply_image_pixel_cap(self._processor)

        model_cls = resolve_vl_model_class(weights_path)
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        try:
            self._model = model_cls.from_pretrained(
                weights_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception:
            self._model = model_cls.from_pretrained(
                weights_path,
                attn_implementation="sdpa",
                **model_kwargs,
            )
        self._model.eval()
        self._model_path = weights_path

    def generate(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> str:
        """Run local VL inference and return the raw assistant text."""
        import torch
        from PIL import Image

        if self._model is None or self._processor is None:
            raise RuntimeError("Local VLM not loaded. Call load_local_vlm() first.")

        image_bytes = kwargs.get("image_bytes")
        image = None
        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

        max_new_tokens = int(
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or 1024
        )
        vl_messages, images = self._build_messages(messages, image)
        text = self._processor.apply_chat_template(
            vl_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self._processor(**processor_kwargs).to(self._model.device)
        # A1 decoding hardening: honor temperature (>0 -> sampling), an optional
        # repetition_penalty (damps greedy degeneration loops on long tool-call
        # JSON), and stop strings (halt on hallucinated markup like ``</pre>`` or a
        # second protocol block). All are opt-in via generation_kwargs; defaults
        # keep the historical greedy behavior so existing runs are unchanged.
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        gen_kwargs.update(_decoding_hardening_kwargs(kwargs, tokenizer=getattr(self._processor, "tokenizer", None)))
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    @staticmethod
    def _build_messages(
        messages: Sequence[ChatMessage], image: Any
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Render chat messages for the Qwen2.5-VL processor and collect images.

        The scene image (``image``) attaches to the first user turn, exactly as
        before. Additionally, any per-message crop images (P5.10.1, from a
        parts-list tool turn) are threaded in message order, with the returned
        ``images`` list aligned to their appearance so the processor binds each
        ``<image>`` placeholder to the right picture. With no crops the output is
        byte-identical to the prior single-image behavior.
        """
        vl_messages: list[dict[str, Any]] = []
        images: list[Any] = []
        scene_image_added = False
        for message in messages:
            role = str(message.get("role", "user"))
            content_text, part_images = _split_content(message.get("content", ""))
            if role == "tool":
                tool_name = str(message.get("name") or "tool").strip() or "tool"
                role = "user"
                content_text = f"[TOOL_RESULT {tool_name}]\n{content_text}"
            attach_scene = role == "user" and image is not None and not scene_image_added
            if attach_scene or part_images:
                parts: list[dict[str, Any]] = []
                if attach_scene:
                    content_text = content_text.replace("<image>", "").strip()
                    parts.append({"type": "image", "image": image})
                    images.append(image)
                    scene_image_added = True
                for part_image in part_images:
                    parts.append({"type": "image", "image": part_image})
                    images.append(part_image)
                parts.append({"type": "text", "text": content_text})
                vl_messages.append({"role": role, "content": parts})
            else:
                vl_messages.append({"role": role, "content": content_text})
        return vl_messages, images


_DEFAULT_LOCAL_VL_BACKEND = LocalVisionTextSLMBackend()


def load_local_vlm(weights_path: str, *, device: str = "cuda") -> None:
    """Load the process-wide default local vision-language backend."""
    _DEFAULT_LOCAL_VL_BACKEND.load(weights_path, device=device)


def get_local_vl_backend() -> LocalVisionTextSLMBackend:
    return _DEFAULT_LOCAL_VL_BACKEND
