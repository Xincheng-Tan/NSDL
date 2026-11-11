from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nsdl.slm_backends.base import ChatMessage, _decoding_hardening_kwargs


@dataclass
class LocalTextSLMBackend:
    """Locally loaded causal-LM backend for SFT/GRPO checkpoints."""

    # Enable the single validator-feedback second pass for local inference.
    supports_correction_retry: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model_path: str | None = field(default=None, init=False)

    def load(self, weights_path: str, *, device: str = "cuda") -> None:
        """Load and cache a local Qwen-style instruct checkpoint."""
        if self._model is not None and self._model_path == weights_path:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            weights_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            weights_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._model_path = weights_path

    def generate(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> str:
        """Run local SLM inference and return the raw assistant text."""
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Local SLM not loaded. Call load_local_slm() first.")

        max_new_tokens = int(
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or 1024
        )
        text = self._tokenizer.apply_chat_template(
            [_template_message(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        gen_kwargs.update(_decoding_hardening_kwargs(kwargs, tokenizer=self._tokenizer))
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)


_DEFAULT_LOCAL_TEXT_BACKEND = LocalTextSLMBackend()


def load_local_slm(weights_path: str, *, device: str = "cuda") -> None:
    """Load the process-wide default local text backend."""
    _DEFAULT_LOCAL_TEXT_BACKEND.load(weights_path, device=device)


def get_local_text_backend() -> LocalTextSLMBackend:
    return _DEFAULT_LOCAL_TEXT_BACKEND


def _content_to_prompt_text(content: Any) -> str:
    """Flatten chat content to text; image parts degrade to a placeholder line.

    A pure-text backend cannot consume crop pixels (P5.10.1), so an image part
    renders as ``[CROP <entity> attached]`` instead of raising or dropping it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type in {"image", "image_url"}:
                    entity = str(item.get("entity") or "").strip()
                    parts.append(f"[CROP {entity} attached]" if entity else "[CROP attached]")
                else:
                    parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content) if content is not None else ""


def _template_message(message: ChatMessage) -> dict[str, Any]:
    rendered = dict(message)
    content = _content_to_prompt_text(rendered.get("content", ""))
    if str(rendered.get("role")) == "tool":
        tool_name = str(rendered.get("name") or "tool").strip() or "tool"
        rendered["role"] = "user"
        content = f"[TOOL_RESULT {tool_name}]\n{content}"
    rendered["content"] = content
    return rendered
