"""G6: an API-backed backend for ``nsdl.tool_runtime.run_agentic_tool_loop``.

The G6 row asks a question no baseline in the paper currently answers: **give a
much stronger closed model the same structural parts we give our SLM (detect +
schedule + expand, through the very same ``tool_runtime``), and where does it
land?**  Everything before this file could only be collected single-turn
(``scripts/run_locus_strong_baseline.py``: ``image`` / ``oracle_text``), so
facet B -- the deterministic backend Phi -- had never been handed to any
baseline at all (docs/plan.md G6, docs/benchmark.md 7.12(3)).

Design notes, in the order they matter:

* **Same loop, swapped backend.**  This class implements exactly the contract
  ``run_agentic_tool_loop`` requires -- ``generate(messages=..., **kwargs)
  -> str`` -- so the closed model runs through the same rounds, the same tool
  dispatch, the same parser, the same guardrail and the same judge as our own
  checkpoints.  Nothing about the comparison is re-implemented, which is the
  only way the row is a like-for-like structural comparison rather than a
  reimplementation contest.

* **Transport faults are not model failures** (docs/benchmark.md 7.1-7.4).  On
  this host an HTTP 200 can still carry a canned greeting (the prompt never
  reached the model) or an empty body (billed, no content).  Both are retried
  on the *other* relay line; only after ``max_attempts`` do we raise
  ``RelayTransportError``, which the caller must record in a ``missing``
  denominator rather than scoring as 0.  Scoring a dark relay window as a wrong
  answer would flatter every method we compare against.

* **Duplicated transport, deliberately.**  ``run_locus_strong_baseline.py`` has
  a structurally similar ``call()``.  It is not imported here: that collector's
  behaviour is frozen against already-collected JSONL, and refactoring it to
  share code with a new consumer risks silently changing rows that are already
  in the paper.  The duplication is ~40 lines and buys zero regression risk.

* **The image rides on the first user turn only.**  ``tool_runtime`` hands the
  raw image through ``generation_kwargs["image_bytes"]`` on every round; a chat
  API re-sends the whole transcript each call, so attaching it once to the
  first user message is both correct and the cheap option.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Mapping, Sequence

_REPO = pathlib.Path(__file__).resolve().parents[1]
_KEYS = _REPO / "api_keys.json"

_CANNED = "How can I help you"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0.0.0 Safari/537.36")
# An ambient http_proxy is exported on this host; urllib would pick it up
# silently and route relay traffic through a local proxy. Pin an empty ProxyHandler.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_KEYNAME = {"gpt": "OPENAI_KEY", "claude": "CLAUDE_KEY"}


class RelayTransportError(RuntimeError):
    """The prompt demonstrably never got a model answer. NOT a wrong answer.

    Raised only after every attempt across every configured line failed. The
    eval layer records it under a `missing` denominator; see
    docs/benchmark.md 7.5 (three denominators are never merged).
    """

    def __init__(self, model: str, statuses: Mapping[str, int], detail: str = "") -> None:
        super().__init__(
            f"relay transport failed for {model}: {dict(statuses)}"
            + (f" :: {detail[:200]}" if detail else "")
        )
        self.model = model
        self.statuses = dict(statuses)
        self.detail = detail


def _jpeg_b64(image_bytes: bytes, quality: int = 95) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# Keys our loop passes through ``generation_kwargs`` that a chat completions
# endpoint either does not accept or would reject. Filtering by allow-list (not
# deny-list) means a future local-backend knob cannot leak into an API payload
# and turn into an HTTP 400 halfway through a collection.
_PASSTHROUGH = {"temperature", "top_p", "max_tokens", "stop"}
_EXTRA_BODY: dict = json.loads(os.environ.get("LNSDL_API_EXTRA_BODY", "{}") or "{}")

# Seconds to wait before attempt 1, 2, 3, ... (attempt 0 is immediate). Sized so
# that the default six attempts span >2 min of wall clock rather than <1 s.
_BACKOFF_S = (3, 10, 30, 60, 60)


class RelayChatBackend:
    """OpenAI-compatible chat backend over the relay.

    ``lines`` are entries of ``api_keys.json``. They are one relay with two
    quotas (memory: relay-api-silent-failures) -- alternating between them buys
    ~2x concurrency but does NOT escape a shared gateway blackout, so they are
    a dispatch dimension only and every row stays ``api_source=relay``.
    """

    api_source = "relay"

    def __init__(
        self,
        model: str,
        *,
        lines: Sequence[str] = ("RELAY_A", "RELAY_A2"),
        timeout: int = 120,
        max_attempts: int = 6,
        max_tokens: int = 1500,
        keys_path: pathlib.Path | str | None = None,
    ) -> None:
        conf = json.loads(pathlib.Path(keys_path or _KEYS).read_text())
        missing = [ln for ln in lines if ln not in conf]
        if missing:
            raise KeyError(f"api_keys.json has no line(s) {missing}")
        self.model = model
        self.cfgs = [conf[ln] for ln in lines]
        self.line_names = list(lines)
        self.timeout = int(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.max_tokens = int(max_tokens)
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._next_line = 0

    # -- provenance ------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "backend": "relay_chat",
            "model": self.model,
            "api_source": self.api_source,
            "lines": self.line_names,
            "timeout_s": self.timeout,
            "max_attempts": self.max_attempts,
            "max_tokens": self.max_tokens,
        }

    def take_records(self) -> list[dict[str, Any]]:
        """Pop and return the per-attempt records accumulated since last call.

        The loop calls ``generate`` several times per sample, so the caller
        drains this after each sample to get that sample's transport story
        (attempts, statuses, latency). Retry rate is a G6 reporting
        requirement, not a debug aid -- it must sit next to acc.
        """
        with self._lock:
            out, self._records = self._records, []
        return out

    # -- transport -------------------------------------------------------
    def _to_openai(self, messages: Sequence[Mapping[str, Any]],
                   image_bytes: bytes | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        attached = False
        for msg in messages:
            role, content = msg.get("role"), msg.get("content")
            if role == "user" and image_bytes and not attached:
                out.append({"role": "user", "content": [
                    {"type": "text", "text": str(content)},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + _jpeg_b64(image_bytes)}},
                ]})
                attached = True
            else:
                out.append({"role": str(role), "content": str(content)})
        return out

    def _once(self, cfg: Mapping[str, Any], payload: Mapping[str, Any]
              ) -> tuple[str, str, float]:
        """-> (status, text, seconds). 'ok' means the prompt demonstrably arrived."""
        keyname = _KEYNAME["claude" if self.model.startswith("claude") else "gpt"]
        req = urllib.request.Request(
            cfg["openai_base_url"] + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": _UA,
                     "Authorization": "Bearer " + cfg[keyname]},
        )
        started = time.time()
        try:
            body = json.loads(_OPENER.open(req, timeout=self.timeout).read().decode())
        except urllib.error.HTTPError as exc:
            return (f"HTTP{exc.code}", exc.read()[:200].decode(errors="replace"),
                    time.time() - started)
        except Exception as exc:  # noqa: BLE001 -- a collector reports, never raises
            return type(exc).__name__, str(exc)[:200], time.time() - started
        dt = time.time() - started
        choices = body.get("choices") or []
        txt = (choices[0].get("message") or {}).get("content") or "" if choices else ""
        if not txt.strip():
            return "EMPTY", "", dt
        if _CANNED in txt:
            return "canned", txt[:200], dt
        return "ok", txt, dt

    # -- run_agentic_tool_loop contract ----------------------------------
    def generate(self, *, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        image_bytes = kwargs.get("image_bytes")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(kwargs.get("max_tokens") or self.max_tokens),
            "messages": self._to_openai(messages, image_bytes if isinstance(image_bytes, bytes) else None),
        }
        for key in _PASSTHROUGH:
            if key in kwargs and kwargs[key] is not None and key != "max_tokens":
                payload[key] = kwargs[key]
        # LNSDL_API_EXTRA_BODY (JSON, default none): merged into every request body.
        # 2026-09-04: the local Qwen3-VL-32B serving answers our agentic system
        # prompt with an immediate <|im_end|> on a large share of calls (first-
        # token mass sits on control tokens), which the loop sees as EMPTY x6 ->
        # transport-missing. {"min_tokens":..,"bad_words":[..]} forces content;
        # it is a protocol note on that row and is off for every relay row.
        if _EXTRA_BODY:
            payload.update(_EXTRA_BODY)

        statuses: Counter[str] = Counter()
        last_detail = ""
        for attempt in range(self.max_attempts):
            # Backoff between attempts. Measured 2026-09-01
            # (results/lnsdl/audits/relay_image_payload_probe.json): image
            # requests fail in **clusters** -- one dark round took out 837 KB,
            # 267 KB and 200 KB payloads alike (~90 s hang each) while a
            # text-only request in the same round returned in 2.2 s, and the
            # very next round returned every payload in 3-5 s. Retrying
            # back-to-back therefore just re-samples the same dark window;
            # spacing the attempts is what buys an independent draw. Sleeping
            # rather than shortening the timeout is deliberate: genuine hard
            # splits have p90 latency ~102 s (benchmark.md 7.4), which overlaps
            # the ~90 s hang, so the two are not separable by deadline.
            if attempt:
                time.sleep(_BACKOFF_S[min(attempt - 1, len(_BACKOFF_S) - 1)])
            with self._lock:
                idx = self._next_line
                self._next_line = (self._next_line + 1) % len(self.cfgs)
            status, text, dt = self._once(self.cfgs[idx], payload)
            statuses[status] += 1
            with self._lock:
                self._records.append({
                    "attempt": attempt,
                    "line": self.line_names[idx],
                    "status": status,
                    "latency_s": round(dt, 2),
                })
            if status == "ok":
                return text
            last_detail = text
        raise RelayTransportError(self.model, statuses, last_detail)


# ---------------------------------------------------------------------------
# zero-shot protocol supplement
# ---------------------------------------------------------------------------
# benchmark.md 7.12(3) discipline (1): this row must not be a protocol-transfer
# straw man. Our system prompt already documents every tool in natural language,
# but it documents the output *schema* without ever showing a worked turn -- a
# fine-tuned model learned the turn structure from 6300 demonstrations, a
# zero-shot model has to read it off the prompt. The supplement below adds the
# missing half (one complete two-turn exemplar + the failure modes we know cost
# format compliance) and nothing else: no task hints, no LOCUS entities, no
# planning advice that would leak facet-A or facet-B answers. The exemplar
# scene is synthetic and does not occur in any split.
ZERO_SHOT_SUPPLEMENT = """

## Worked Example (format only -- this scene is not one of your tasks)

Suppose [ROBOTS] lists R1 (mobility, manipulation) and R2 (container_operation),
and the instruction is "put the mug into the cupboard".

Your FIRST assistant turn emits the sketch and ONE tool call, and stops:

<think>The mug must end up inside an openable container, so opening has to be
scheduled before the deposit and the two robots must be coordinated. That is
mechanical, so I let schedule complete it.</think>

[GOAL]
{"targets":[{"item":"mug","destination":"cupboard","status":"active","status_reason":"needs_relocation","status_line":"mug: active"}],
 "active":["mug"],
 "skip":[]}

[ASSIGN]
{"R1":{"items":["mug"]},"R2":{"items":[]}}

[PLAN_SKETCH]
[{"op_id":"o1","op_type":"acquire","entity":"mug","dest":null,"deps":[],"robot":"R1","gaps":[],"params":{}},
 {"op_id":"o2","op_type":"deposit","entity":"mug","dest":"cupboard","deps":["o1"],"robot":"R1","gaps":["deps"],"params":{}}]

[TOOL_CALL]
{"name":"schedule","args":{"ops":[{"op_id":"o1","op_type":"acquire","entity":"mug","dest":null,"deps":[],"robot":"R1","params":{}},{"op_id":"o2","op_type":"deposit","entity":"mug","dest":"cupboard","deps":["o1"],"robot":"R1","params":{}}],"assign":{"R1":{"items":["mug"]},"R2":{"items":[]}}}}

Then you STOP and wait. The tool result arrives as the next message. Your SECOND
assistant turn reads it and emits the final plan:

<think>schedule returned ok with the open op inserted and assigned to R2. I adopt
its complete DAG.</think>

[PLAN]
[{"op_id":"op1","op_type":"acquire","entity":"mug","dest":null,"deps":[],"robot":"R1","params":{}},
 {"op_id":"op2","op_type":"open","entity":"cupboard","dest":null,"deps":[],"robot":"R2","params":{}},
 {"op_id":"op3","op_type":"deposit","entity":"mug","dest":"cupboard","deps":["op1","op2"],"robot":"R1","params":{}}]

## Protocol Rules That Cost Format Compliance Most Often

1. NEVER write `[TOOL_CALL]` and `[PLAN]` in the same turn. A turn containing a
   tool call must end right after the tool call JSON. Writing the plan yourself
   in that turn means the tool never runs and its correction is lost.
2. At most ONE `[TOOL_CALL]` per turn.
3. Emit the section headers literally, in brackets, on their own line:
   `[GOAL]`, `[ASSIGN]`, `[PLAN_SKETCH]`, `[TOOL_CALL]`, `[PLAN]`.
4. Every section body is raw JSON. No markdown fences, no prose inside the JSON,
   no trailing commentary after the closing bracket.
5. The episode is finished only when you emit `[PLAN]`. If a tool reports
   `ok:false`, read its `reason`, repair, and call it again -- do not give up
   and hand-write a plan that the tool just rejected.
"""


def build_zero_shot_system_prompt(base: str | None = None) -> str:
    """Base protocol prompt + the worked-turn supplement (see above)."""
    if base is None:
        from nsdl.slm_interface import build_slm_system_prompt

        base = build_slm_system_prompt()
    return base + ZERO_SHOT_SUPPLEMENT
