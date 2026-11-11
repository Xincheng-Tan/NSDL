from __future__ import annotations

from types import SimpleNamespace

import nsdl.llm_api as llm_api


class _StubResponsesAPI:
    def __init__(self, outer: "_StubClient") -> None:
        self.outer = outer

    def create(self, **kwargs):
        self.outer.responses_calls.append(kwargs)
        if kwargs.get("temperature") == 0.0:
            raise RuntimeError("Unsupported parameter: temperature")
        return SimpleNamespace(output_text="OK", error=None, output=[])


class _StubChatCompletionsAPI:
    def __init__(self, outer: "_StubClient") -> None:
        self.outer = outer

    def create(self, **kwargs):
        self.outer.chat_calls.append(kwargs)
        raise RuntimeError("protocol_not_supported")


class _StubClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.responses_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_StubChatCompletionsAPI(self))
        self.responses = _StubResponsesAPI(self)


class _StreamOnlyResponsesAPI:
    def __init__(self, outer: "_StreamOnlyClient") -> None:
        self.outer = outer

    def create(self, **kwargs):
        self.outer.responses_calls.append(kwargs)
        if not kwargs.get("stream"):
            return SimpleNamespace(output_text="", error=None, output=[])
        return iter(
            [
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="O"),
                SimpleNamespace(type="response.output_text.delta", delta="K"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text="", error=None, output=[]),
                ),
            ]
        )


class _StreamOnlyClient:
    def __init__(self) -> None:
        self.responses_calls: list[dict] = []
        self.responses = _StreamOnlyResponsesAPI(self)


def test_create_chat_completion_falls_back_to_responses_and_retries_without_temperature(
    monkeypatch,
) -> None:
    stub = _StubClient()
    llm_api._PROTOCOL_CACHE.clear()
    llm_api._RESPONSES_STREAM_REQUIRED_CACHE.clear()
    monkeypatch.setattr(llm_api, "build_openai_client", lambda config: stub)

    response = llm_api.create_chat_completion(
        messages=[
            {"role": "system", "content": "Reply with OK only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at the image and reply OK only."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ],
        api_key="test-key",
        api_base_url="https://relay.example.com/v1/chat/completions",
        api_model="gpt-5.6-terra",
        temperature=0.0,
        max_tokens=16,
    )

    assert response.protocol == "responses"
    assert response.choices[0].message.content == "OK"
    assert len(stub.chat_calls) == 1
    assert len(stub.responses_calls) == 2
    assert "temperature" not in stub.responses_calls[-1]
    assert stub.responses_calls[-1]["max_output_tokens"] == 16
    assert stub.responses_calls[-1]["tool_choice"] == "none"
    assert stub.responses_calls[-1]["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "Reply with OK only."}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at the image and reply OK only."},
                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
            ],
        },
    ]

    second = llm_api.create_chat_completion(
        messages=[{"role": "user", "content": "Reply with OK only."}],
        api_key="test-key",
        api_base_url="https://relay.example.com/v1/chat/completions",
        api_model="gpt-5.6-terra",
        max_tokens=8,
    )
    assert second.protocol == "responses"
    assert len(stub.chat_calls) == 1
    assert len(stub.responses_calls) == 3


def test_create_chat_completion_recovers_empty_responses_with_sse_and_caches_it(monkeypatch) -> None:
    stub = _StreamOnlyClient()
    llm_api._PROTOCOL_CACHE.clear()
    llm_api._RESPONSES_STREAM_REQUIRED_CACHE.clear()
    monkeypatch.setattr(llm_api, "build_openai_client", lambda config: stub)

    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at the image and reply OK only."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        "api_key": "test-key",
        "api_base_url": "https://relay.example.com/v1/chat/completions",
        "api_model": "gpt-5.6-terra",
        "api_protocol": "responses",
        "max_tokens": 16,
    }
    first = llm_api.create_chat_completion(**request)

    assert first.protocol == "responses"
    assert first.choices[0].message.content == "OK"
    assert len(stub.responses_calls) == 2
    assert "stream" not in stub.responses_calls[0]
    assert stub.responses_calls[1]["stream"] is True

    second = llm_api.create_chat_completion(**request)
    assert second.choices[0].message.content == "OK"
    assert len(stub.responses_calls) == 3
    assert stub.responses_calls[-1]["stream"] is True


def test_create_chat_completion_uses_chat_when_supported(monkeypatch) -> None:
    llm_api._PROTOCOL_CACHE.clear()
    llm_api._RESPONSES_STREAM_REQUIRED_CACHE.clear()

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="CHAT_OK"))],
    )
    stub = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)),
        responses=SimpleNamespace(create=lambda **kwargs: None),
    )
    monkeypatch.setattr(llm_api, "build_openai_client", lambda config: stub)

    result = llm_api.create_chat_completion(
        messages=[{"role": "user", "content": "Reply with OK only."}],
        api_key="test-key",
        api_base_url="https://api.openai.com/v1/chat/completions",
        api_model="grok-4-1-fast-non-reasoning",
        max_tokens=8,
    )

    assert result.choices[0].message.content == "CHAT_OK"
    assert llm_api._PROTOCOL_CACHE[("https://api.openai.com/v1", "grok-4-1-fast-non-reasoning")] == (
        "chat_completions"
    )
