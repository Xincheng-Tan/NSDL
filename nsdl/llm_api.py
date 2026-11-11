from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_KEY_FILE = REPO_ROOT / "api_keys.json"

API_BASE_URL_LIST = [
    "https://api.openai.com/v1/chat/completions",
    "https://api.openai.com/v1",
]

API_MODEL_LIST = [
    "grok-4-1-fast-non-reasoning",
    "gpt-4o",
]

_PROTOCOL_CACHE: dict[tuple[str | None, str], str] = {}
_PROTOCOL_CACHE_LOCK = threading.Lock()
# Some OpenAI-compatible proxies forward Responses API output only as SSE
# deltas, while their non-streaming aggregate is an empty completed response.
# Remember that capability per endpoint/model after observing it, so a long
# batch does not pay for one failed non-streaming request per sample.
_RESPONSES_STREAM_REQUIRED_CACHE: set[tuple[str | None, str]] = set()


@dataclass(frozen=True)
class LLMEndpointConfig:
    api_key: str
    base_url: str | None
    model: str


@dataclass(frozen=True)
class ChatCompletionMessage:
    content: str


@dataclass(frozen=True)
class ChatCompletionChoice:
    message: ChatCompletionMessage


@dataclass(frozen=True)
class ChatCompletionCompat:
    choices: list[ChatCompletionChoice]
    error: Any = None
    protocol: str = "chat_completions"
    raw_response: Any = None


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _parse_key_payload(payload: str) -> list[str]:
    payload = payload.strip()
    if not payload:
        return []

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return [
            line.strip().strip(",").strip("\"'")
            for line in payload.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if isinstance(data, dict):
        for field in ("OPENAI_KEY", "AGENT_KEY", "API_KEY", "keys"):
            value = data.get(field)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        return []

    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]

    return []


def load_api_keys(api_key_file: Path | str | None = None) -> list[str]:
    env_keys = os.getenv("NSDL_API_KEYS")
    if env_keys:
        return [item.strip() for item in env_keys.split(",") if item.strip()]

    key_file = Path(api_key_file) if api_key_file else DEFAULT_API_KEY_FILE
    if not key_file.exists():
        raise FileNotFoundError(f"API key file not found: {key_file}")

    keys = _parse_key_payload(key_file.read_text(encoding="utf-8"))
    if not keys:
        raise ValueError(f"No API keys found in {key_file}")
    return keys


def get_api_key_for_index(
    endpoint_index: int = 0,
    *,
    api_key_file: Path | str | None = None,
) -> str:
    keys = load_api_keys(api_key_file)
    if endpoint_index < 0:
        raise IndexError("endpoint_index must be non-negative")
    if endpoint_index >= len(keys):
        raise IndexError(
            f"endpoint_index={endpoint_index} exceeds available api keys ({len(keys)})"
        )
    return keys[endpoint_index]


def get_endpoint_config(
    endpoint_index: int = 0,
    *,
    api_key_file: Path | str | None = None,
    api_base_url_list: Sequence[str] | None = None,
    api_model_list: Sequence[str] | None = None,
) -> LLMEndpointConfig:
    keys = load_api_keys(api_key_file)
    base_urls = list(api_base_url_list or API_BASE_URL_LIST)
    models = list(api_model_list or API_MODEL_LIST)

    if endpoint_index < 0:
        raise IndexError("endpoint_index must be non-negative")
    if endpoint_index >= len(keys):
        raise IndexError(
            f"endpoint_index={endpoint_index} exceeds available api keys ({len(keys)})"
        )
    if endpoint_index >= len(base_urls):
        raise IndexError(
            f"endpoint_index={endpoint_index} exceeds API_BASE_URL_LIST ({len(base_urls)})"
        )
    if endpoint_index >= len(models):
        raise IndexError(
            f"endpoint_index={endpoint_index} exceeds API_MODEL_LIST ({len(models)})"
        )

    return LLMEndpointConfig(
        api_key=keys[endpoint_index],
        base_url=normalize_base_url(base_urls[endpoint_index]),
        model=models[endpoint_index],
    )


def get_available_endpoint_count(
    *,
    api_key_file: Path | str | None = None,
    api_base_url_list: Sequence[str] | None = None,
    api_model_list: Sequence[str] | None = None,
) -> int:
    keys = load_api_keys(api_key_file)
    base_urls = list(api_base_url_list or API_BASE_URL_LIST)
    models = list(api_model_list or API_MODEL_LIST)
    return min(len(keys), len(base_urls), len(models))


def build_openai_client(config: LLMEndpointConfig) -> OpenAI:
    if config.base_url:
        return OpenAI(api_key=config.api_key, base_url=config.base_url)
    return OpenAI(api_key=config.api_key)


def _protocol_cache_key(config: LLMEndpointConfig) -> tuple[str | None, str]:
    return (config.base_url, config.model)


def _get_cached_protocol(config: LLMEndpointConfig) -> str | None:
    with _PROTOCOL_CACHE_LOCK:
        return _PROTOCOL_CACHE.get(_protocol_cache_key(config))


def _set_cached_protocol(config: LLMEndpointConfig, protocol: str) -> None:
    with _PROTOCOL_CACHE_LOCK:
        _PROTOCOL_CACHE[_protocol_cache_key(config)] = protocol


def _responses_stream_is_required(config: LLMEndpointConfig) -> bool:
    with _PROTOCOL_CACHE_LOCK:
        return _protocol_cache_key(config) in _RESPONSES_STREAM_REQUIRED_CACHE


def _set_responses_stream_required(config: LLMEndpointConfig) -> None:
    with _PROTOCOL_CACHE_LOCK:
        _RESPONSES_STREAM_REQUIRED_CACHE.add(_protocol_cache_key(config))


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def _should_fallback_to_responses(exc: Exception) -> bool:
    text = _error_text(exc)
    return (
        "protocol_not_supported" in text
        or "does not support chat completions" in text
        or "不支持 chat completions 协议" in text
    )


def _should_retry_responses_without_temperature(exc: Exception) -> bool:
    text = _error_text(exc)
    return "unsupported parameter: temperature" in text


def _message_part_to_responses_input(part: Any) -> list[dict[str, Any]]:
    if isinstance(part, str):
        return [{"type": "input_text", "text": part}]
    if not isinstance(part, dict):
        return [{"type": "input_text", "text": str(part)}]

    part_type = str(part.get("type") or "").strip()
    if part_type in {"text", "input_text"}:
        return [{"type": "input_text", "text": str(part.get("text") or "")}]
    if part_type in {"image_url", "input_image"}:
        image_payload = part.get("image_url")
        if isinstance(image_payload, dict):
            image_url = image_payload.get("url") or image_payload.get("image_url") or ""
        else:
            image_url = image_payload or part.get("url") or ""
        detail = str(part.get("detail") or "high")
        return [{"type": "input_image", "image_url": str(image_url), "detail": detail}]
    if "text" in part:
        return [{"type": "input_text", "text": str(part.get("text") or "")}]
    return [{"type": "input_text", "text": json.dumps(part, ensure_ascii=False)}]


def _messages_to_responses_input(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                parts.extend(_message_part_to_responses_input(part))
        else:
            parts.extend(_message_part_to_responses_input(content))
        converted.append({"role": role, "content": parts})
    return converted


def _responses_kwargs_from_chat_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    translated = dict(kwargs)
    removed_temperature = False
    if "max_tokens" in translated and "max_output_tokens" not in translated:
        translated["max_output_tokens"] = translated.pop("max_tokens")
    if translated.get("temperature") is None:
        translated.pop("temperature", None)
    if "tools" not in translated and "tool_choice" not in translated:
        translated["tool_choice"] = "none"
    return translated, removed_temperature


def _extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type != "message":
            continue
        for content_item in getattr(item, "content", []) or []:
            content_type = getattr(content_item, "type", None)
            if content_type == "output_text":
                text = getattr(content_item, "text", None)
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    return "\n".join(texts).strip()


def _coerce_responses_to_chat_completion(response: Any) -> ChatCompletionCompat:
    text = _extract_responses_text(response)
    if not text:
        raise RuntimeError("responses API returned no assistant text")
    return ChatCompletionCompat(
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=text))],
        error=getattr(response, "error", None),
        protocol="responses",
        raw_response=response,
    )


def _create_responses_request(
    *,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    stream: bool,
) -> Any:
    def create() -> Any:
        if stream:
            return client.responses.create(**request_kwargs, stream=True)
        return client.responses.create(**request_kwargs)

    try:
        return create()
    except Exception as exc:
        if "temperature" not in request_kwargs or not _should_retry_responses_without_temperature(exc):
            raise
        request_kwargs.pop("temperature", None)
        return create()


def _coerce_responses_stream_to_chat_completion(stream: Any) -> ChatCompletionCompat:
    """Collect text deltas from an OpenAI Responses API SSE stream."""
    deltas: list[str] = []
    completed_response: Any = None
    for event in stream:
        event_type = getattr(event, "type", None)
        delta = getattr(event, "delta", None)
        if event_type == "response.output_text.delta" and isinstance(delta, str):
            deltas.append(delta)
        elif event_type == "response.completed":
            completed_response = getattr(event, "response", None)

    text = "".join(deltas).strip()
    if not text and completed_response is not None:
        text = _extract_responses_text(completed_response)
    if not text:
        raise RuntimeError("streaming responses API returned no assistant text")

    return ChatCompletionCompat(
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=text))],
        error=getattr(completed_response, "error", None),
        protocol="responses",
        raw_response=completed_response,
    )


def _create_responses_completion(
    *,
    client: OpenAI,
    config: LLMEndpointConfig,
    messages: Sequence[dict[str, Any]],
    kwargs: dict[str, Any],
) -> ChatCompletionCompat:
    request_kwargs, _ = _responses_kwargs_from_chat_kwargs(kwargs)
    request_kwargs["model"] = config.model
    request_kwargs["input"] = _messages_to_responses_input(messages)
    if _responses_stream_is_required(config):
        stream = _create_responses_request(
            client=client,
            request_kwargs=request_kwargs,
            stream=True,
        )
        return _coerce_responses_stream_to_chat_completion(stream)

    response = _create_responses_request(
        client=client,
        request_kwargs=request_kwargs,
        stream=False,
    )
    try:
        return _coerce_responses_to_chat_completion(response)
    except RuntimeError as exc:
        if str(exc) != "responses API returned no assistant text":
            raise
        # A completed Responses object with no output is invalid for this
        # adapter. Retry as SSE once; if deltas arrive, cache the requirement
        # for this endpoint/model and use SSE directly for the rest of the run.
        _set_responses_stream_required(config)
        stream = _create_responses_request(
            client=client,
            request_kwargs=request_kwargs,
            stream=True,
        )
        return _coerce_responses_stream_to_chat_completion(stream)


def create_chat_completion(
    *,
    messages: Sequence[dict[str, Any]],
    endpoint_index: int = 0,
    api_key_file: Path | str | None = None,
    api_base_url: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
    api_protocol: str = "auto",
    **kwargs: Any,
):
    if api_key is not None or api_base_url is not None or api_model is not None:
        fallback_base_url: str | None = None
        fallback_model = ""
        if api_base_url is None or api_model is None:
            fallback = get_endpoint_config(endpoint_index=endpoint_index, api_key_file=api_key_file)
            fallback_base_url = fallback.base_url
            fallback_model = fallback.model
        config = LLMEndpointConfig(
            api_key=(api_key or get_api_key_for_index(endpoint_index=endpoint_index, api_key_file=api_key_file)).strip(),
            base_url=normalize_base_url(api_base_url or fallback_base_url),
            model=(api_model or fallback_model).strip(),
        )
        if not config.model:
            raise ValueError("api_model must be provided when no fallback model is configured.")
    else:
        config = get_endpoint_config(endpoint_index=endpoint_index, api_key_file=api_key_file)

    client = build_openai_client(config)
    selected_protocol = api_protocol
    if selected_protocol == "auto":
        selected_protocol = _get_cached_protocol(config) or "chat_completions"

    if selected_protocol == "responses":
        response = _create_responses_completion(
            client=client,
            config=config,
            messages=messages,
            kwargs=kwargs,
        )
        _set_cached_protocol(config, "responses")
        return response

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=list(messages),
            **kwargs,
        )
    except Exception as exc:
        if api_protocol == "auto" and _should_fallback_to_responses(exc):
            response = _create_responses_completion(
                client=client,
                config=config,
                messages=messages,
                kwargs=kwargs,
            )
            _set_cached_protocol(config, "responses")
            return response
        raise

    _set_cached_protocol(config, "chat_completions")
    return response


def describe_endpoint(config: LLMEndpointConfig) -> str:
    masked_key = f"{config.api_key[:6]}...{config.api_key[-4:]}" if len(config.api_key) > 10 else "***"
    return (
        f"model={config.model}, "
        f"base_url={config.base_url or '<default>'}, "
        f"api_key={masked_key}"
    )


def call_llm(
    prompt: str,
    *,
    endpoint_index: int = 0,
    api_key_file: Path | str | None = None,
    api_base_url: str | None = None,
    api_model: str | None = None,
    api_protocol: str = "auto",
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    """Convenience wrapper: send a single user prompt, return assistant text.

    Returns empty string on API error so callers can ``if not text: continue``.
    """
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    extra: dict[str, Any] = {"temperature": temperature, **kwargs}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    try:
        resp = create_chat_completion(
            messages=msgs,
            endpoint_index=endpoint_index,
            api_key_file=api_key_file,
            api_base_url=api_base_url,
            api_model=api_model,
            api_protocol=api_protocol,
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"[llm_api.call_llm] {type(exc).__name__}: {exc}", file=sys.stderr)
        return ""
