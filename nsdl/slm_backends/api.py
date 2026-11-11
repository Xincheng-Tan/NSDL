from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nsdl.slm_backends.base import ChatMessage


@dataclass(frozen=True)
class APISLMBackend:
    """OpenAI-compatible chat-completion backend."""

    endpoint_index: int = 0
    api_base_url: str | None = None
    api_model: str | None = None
    api_key_file: Path | str | None = None
    supports_correction_retry: bool = True

    def generate(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> str:
        temperature = float(kwargs.pop("temperature", 0.2))
        max_tokens_value = kwargs.pop("max_tokens", None)
        max_new_tokens_value = kwargs.pop("max_new_tokens", None)
        max_tokens = int(max_tokens_value or max_new_tokens_value or 1024)
        response = call_llm_with_retry(
            messages=[dict(message) for message in messages],
            endpoint_index=self.endpoint_index,
            api_base_url=self.api_base_url,
            api_model=self.api_model,
            api_key_file=self.api_key_file,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return extract_response_text(response)


def call_llm_with_retry(
    *,
    messages: list[dict[str, Any]],
    endpoint_index: int,
    api_base_url: str | None,
    api_model: str | None,
    temperature: float,
    max_tokens: int,
    api_key_file: Path | str | None = None,
    **kwargs: Any,
) -> Any:
    """Call an OpenAI-compatible API with automatic endpoint fallback."""
    from nsdl.llm_api import create_chat_completion, get_available_endpoint_count

    endpoint_sequence = [endpoint_index]
    if api_base_url is None and api_model is None:
        try:
            count = get_available_endpoint_count(api_key_file=api_key_file)
            start = endpoint_index % count if count > 0 else 0
            endpoint_sequence = [start, *[i for i in range(count) if i != start]]
        except Exception:
            pass

    last_error: Exception | None = None
    for ep_idx in endpoint_sequence:
        for _attempt in range(2):
            try:
                return create_chat_completion(
                    messages=messages,
                    endpoint_index=ep_idx,
                    api_key_file=api_key_file,
                    api_base_url=api_base_url,
                    api_model=api_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
    raise last_error or RuntimeError("LLM API call failed without traceable error.")


def extract_response_text(response: Any) -> str:
    """Extract text content from an OpenAI-compatible chat completion."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts).strip()
    return str(content) if content else ""
