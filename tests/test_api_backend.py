"""G6 API-backed tool-loop backend: offline unit tests (no network).

Every test here stubs `_once`. The point is the *policy* layer -- what counts as
a fault, how retries are dispatched, what provenance survives -- because that is
what decides whether a G6 row is readable, and it is exactly the layer a live
smoke run cannot exercise on demand (you cannot ask the relay for a canned
greeting).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nsdl import api_backend as ab  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """Retry spacing is real seconds in production; tests must not pay them.

    Patched globally rather than per-test so a future retry test cannot
    silently reintroduce a two-minute suite.
    """
    monkeypatch.setattr(ab, "_BACKOFF_S", (0, 0, 0, 0, 0))


def _backend(**kw):
    return ab.RelayChatBackend(kw.pop("model", "gpt-test"), max_attempts=kw.pop("max_attempts", 3), **kw)


class _Scripted:
    """Replaces `_once`; returns the next scripted (status, text) per call."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_cfgs = []
        self.seen_payloads = []

    def __call__(self, cfg, payload):
        self.seen_cfgs.append(cfg)
        self.seen_payloads.append(payload)
        status, text = self.script.pop(0)
        return status, text, 0.5


# ---------------------------------------------------------------------------
# transport policy
# ---------------------------------------------------------------------------
def test_ok_first_try_returns_text_and_records_one_attempt():
    b = _backend()
    b._once = _Scripted([("ok", "[PLAN]\n[]")])
    assert b.generate(messages=[{"role": "user", "content": "hi"}]) == "[PLAN]\n[]"
    rec = b.take_records()
    assert [r["status"] for r in rec] == ["ok"]
    assert b.take_records() == [], "take_records must drain, or per-sample stats double-count"


@pytest.mark.parametrize("fault", ["canned", "EMPTY", "HTTP502", "URLError"])
def test_faults_are_retried_not_returned(fault):
    """A canned greeting / empty body is a 200. Returning it would score a dark
    relay window as the model's answer -- see benchmark.md 7.1."""
    b = _backend()
    b._once = _Scripted([(fault, "junk"), ("ok", "real answer")])
    assert b.generate(messages=[{"role": "user", "content": "hi"}]) == "real answer"
    assert [r["status"] for r in b.take_records()] == [fault, "ok"]


def test_exhausted_attempts_raise_rather_than_return_garbage():
    b = _backend(max_attempts=3)
    b._once = _Scripted([("canned", "x"), ("EMPTY", ""), ("HTTP500", "y")])
    with pytest.raises(ab.RelayTransportError) as exc:
        b.generate(messages=[{"role": "user", "content": "hi"}])
    assert exc.value.statuses == {"canned": 1, "EMPTY": 1, "HTTP500": 1}
    assert len(b.take_records()) == 3


def test_retries_alternate_lines():
    """Two relay lines are one relay with two quotas: alternating buys
    concurrency, and a retry on the *other* line is the only retry that can
    dodge a per-quota fault."""
    b = _backend(max_attempts=4)
    b._once = s = _Scripted([("EMPTY", ""), ("EMPTY", ""), ("ok", "z")])
    b.generate(messages=[{"role": "user", "content": "hi"}])
    assert s.seen_cfgs[0] is not s.seen_cfgs[1]
    assert s.seen_cfgs[0] is s.seen_cfgs[2]


# ---------------------------------------------------------------------------
# message conversion
# ---------------------------------------------------------------------------
def _png(w=8, h=8) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_image_rides_only_the_first_user_turn():
    """tool_runtime passes image_bytes on EVERY round; a chat API re-sends the
    whole transcript, so attaching per-round would duplicate the image N times
    in one payload."""
    b = _backend()
    b._once = s = _Scripted([("ok", "a")])
    b.generate(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "tool result"},
        ],
        image_bytes=_png(),
    )
    msgs = s.seen_payloads[0]["messages"]
    assert isinstance(msgs[1]["content"], list)
    assert [p["type"] for p in msgs[1]["content"]] == ["text", "image_url"]
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert msgs[3]["content"] == "tool result", "later user turns stay plain text"
    assert all(isinstance(m["content"], str) for m in (msgs[0], msgs[2], msgs[3]))


def test_text_only_when_no_image():
    b = _backend()
    b._once = s = _Scripted([("ok", "a")])
    b.generate(messages=[{"role": "user", "content": "u"}])
    assert s.seen_payloads[0]["messages"][0]["content"] == "u"


def test_generation_kwargs_are_allow_listed():
    """Local-backend knobs (repetition_penalty, stop_strings, image_bytes,
    max_new_tokens) must not leak into an OpenAI payload -- a 400 halfway
    through a collection is worse than a knob silently unused."""
    b = _backend()
    b._once = s = _Scripted([("ok", "a")])
    b.generate(
        messages=[{"role": "user", "content": "u"}],
        temperature=0.0,
        max_tokens=77,
        repetition_penalty=1.05,
        stop_strings=["</pre>"],
        max_new_tokens=999,
        image_bytes=None,
    )
    payload = s.seen_payloads[0]
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 77
    assert set(payload) == {"model", "max_tokens", "messages", "temperature"}


def test_max_tokens_falls_back_to_constructor():
    b = ab.RelayChatBackend("gpt-test", max_tokens=1234)
    b._once = s = _Scripted([("ok", "a")])
    b.generate(messages=[{"role": "user", "content": "u"}])
    assert s.seen_payloads[0]["max_tokens"] == 1234


# ---------------------------------------------------------------------------
# provenance / prompt
# ---------------------------------------------------------------------------
def test_describe_carries_the_api_source_column():
    d = _backend().describe()
    assert d["api_source"] == "relay"
    assert d["lines"] == ["RELAY_A", "RELAY_A2"]


def test_unknown_line_fails_loudly():
    with pytest.raises(KeyError):
        ab.RelayChatBackend("gpt-test", lines=("NO_SUCH_LINE",))


def test_zero_shot_supplement_is_additive_and_leaks_no_task_content():
    base = "BASE PROMPT"
    out = ab.build_zero_shot_system_prompt(base)
    assert out.startswith(base), "the supplement must not rewrite the protocol prompt"
    assert "Worked Example" in out
    # benchmark.md 7.12(3)(1): format help is required, task help is forbidden.
    # The exemplar scene must not name anything the splits actually contain.
    for leaked in ("cabinet", "drawer", "bowl", "plate", "scissors", "banana"):
        assert leaked not in out[len(base):], f"exemplar leaks a benchmark entity: {leaked}"


def test_retries_are_spaced_out_not_back_to_back(monkeypatch):
    """The relay's image faults cluster in dark windows (probe 2026-09-01):
    four back-to-back attempts re-sample the same window, so spacing is the
    whole point of retrying. Attempt 0 must still be immediate."""
    slept = []
    monkeypatch.setattr(ab, "_BACKOFF_S", (3, 10, 30))
    monkeypatch.setattr(ab.time, "sleep", slept.append)
    b = _backend(max_attempts=4)
    b._once = _Scripted([("URLError", "x")] * 3 + [("ok", "answer")])
    assert b.generate(messages=[{"role": "user", "content": "hi"}]) == "answer"
    assert slept == [3, 10, 30]


def test_first_attempt_never_sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(ab.time, "sleep", slept.append)
    b = _backend()
    b._once = _Scripted([("ok", "answer")])
    b.generate(messages=[{"role": "user", "content": "hi"}])
    assert slept == []
