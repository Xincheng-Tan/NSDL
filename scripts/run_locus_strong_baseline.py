"""§7-7 strong-baseline pre-run over LOCUS-Bench, via the relay.

Two input conditions, run and reported separately (their difference is what
isolates the S axis):

  image        the rendered 1920x1080 frame (JPEG q95) + the official prompt
  oracle_text  the same prompt with ``<image>`` replaced by the GT initial
               state serialised as facts -- i.e. perception handed to the model
               for free. Uses ``nsdl.oracle_perception`` (eval-only backend).

Relay realities this runner is built around (docs/benchmark.md 7.1-7.3):
  * HTTP 200 does not mean the prompt arrived -- canned greetings and empty
    bodies are rejected as transport faults, never scored as model failure.
  * Failures come in shared gateway blackouts, so a big inline retry cap just
    burns the budget inside a dark window. Retry is *multi-pass*: one attempt
    per item per pass, resume from the JSONL, later passes land in other windows.
  * The two relay lines (RELAY_A / RELAY_A2) are one relay with two
    quotas: dispatch alternates between them for ~2x throughput. They are a
    dispatch dimension only -- every row is api_source=relay.

This script only *collects*; scoring against eval_gt is a separate offline pass,
so a scoring bug never costs another API run.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_KEYS = _REPO / "api_keys.json"
_BENCH = _REPO / "data/lnsdl_bench/ramp63k"
SPLITS = ("test_id", "ood_s", "ood_m", "ood_x")

_CANNED = "How can I help you"
_DUD_CHARS = 40
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0.0.0 Safari/537.36")
# An ambient http_proxy is exported on this host; urllib picks it up silently.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_KEYNAME = {"gpt": "OPENAI_KEY", "claude": "CLAUDE_KEY"}
_EXTRA_BODY = json.loads(os.environ.get("LNSDL_SB_EXTRA_BODY", "{}") or "{}")


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------

def _jpeg_b64(png_bytes: bytes, quality: int = 95) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.open(io.BytesIO(png_bytes)).convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _oracle_facts(ground_truth: dict) -> str:
    """GT initial state as text -- the free-perception arm of the S-axis split."""
    os.environ.setdefault("NSDL_ORACLE_PERCEPTION", "eval-only")
    from nsdl.oracle_perception import build_oracle_observations
    entities, triples = build_oracle_observations(ground_truth)
    lines = ["[SCENE FACTS - ground truth, no perception needed]",
             "entities: " + ", ".join(entities) if entities else "entities: (none)"]
    lines += [f"  {e} {v} {f}" for e, v, f in triples] or ["  (no placement relations)"]
    return "\n".join(lines) + "\n"


def load_items(splits, smoke: int | None, seed_stride: bool = True):
    """-> [(item_id, split, row_idx, system, instruction, png_bytes, gt)]

    A smoke subsample is an even stride through each split (deterministic, no
    RNG), sized proportionally to the split so the sample keeps the grid mix.
    """
    import pandas as pd
    total = {}
    picked = []
    for split in splits:
        df = pd.read_parquet(_BENCH / split / "data.parquet")
        total[split] = len(df)
        picked.append((split, df, list(range(len(df)))))
    if smoke:
        grand = sum(total[s] for s in splits)
        for i, (split, df, idxs) in enumerate(picked):
            k = max(1, round(smoke * total[split] / grand))
            stride = max(1, len(idxs) // k)
            picked[i] = (split, df, idxs[::stride][:k])
    items = []
    for split, df, idxs in picked:
        for i in idxs:
            row = df.iloc[i]
            msgs = list(row["prompt"])
            system = next(m["content"] for m in msgs if m["role"] == "system")
            user = next(m["content"] for m in msgs if m["role"] == "user")
            items.append((f"{split}:{i:05d}", split, i, system,
                          user.replace("<image>", "", 1).strip(),
                          row["images"][0]["bytes"],
                          row["reward_model"]["ground_truth"]))
    return items, total


def prompt_sha(instruction: str) -> str:
    """Identity of the instruction text a response was produced against.

    The frozen splits are not immutable: §7.6 rewrote 66 instructions in place
    after a collection had already run against the old wording. Stamping the
    text a row was collected under is what lets the scorer tell a fresh answer
    from one that answered a question no longer being asked, without relying on
    file order.
    """
    return hashlib.sha256(instruction.encode()).hexdigest()[:16]


def _selected_rows(path: pathlib.Path) -> dict[str, set[int]]:
    """Parse ``--only-rows``: either ``{split: [row, ...]}`` or the report
    written by ``scripts/repair_locus_descriptions.py``."""
    blob = json.loads(path.read_text())
    out: dict[str, set[int]] = {}
    for split, value in blob.items():
        if isinstance(value, dict) and "rewrites" in value:
            out[split] = {int(k) for k in value["rewrites"]}
        elif isinstance(value, list):
            out[split] = {int(v) for v in value}
    return out


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def _payload(model, condition, system, instruction, png, gt, max_tokens):
    if condition == "image":
        user = [{"type": "text", "text": instruction},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + _jpeg_b64(png)}}]
    else:
        user = _oracle_facts(gt) + instruction
    return {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}


def call(cfg, model, payload, timeout):
    """-> (status, text, seconds). Status 'ok' means the prompt demonstrably arrived."""
    keyname = _KEYNAME["claude" if model.startswith("claude") else "gpt"]
    req = urllib.request.Request(
        cfg["openai_base_url"] + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": _UA,
                 "Authorization": "Bearer " + cfg[keyname]})
    started = time.time()
    try:
        body = json.loads(_OPENER.open(req, timeout=timeout).read().decode())
    except urllib.error.HTTPError as exc:
        return f"HTTP{exc.code}", exc.read()[:200].decode(errors="replace"), time.time() - started
    except Exception as exc:  # noqa: BLE001 -- a collector reports, never raises
        return type(exc).__name__, str(exc)[:120], time.time() - started
    dt = time.time() - started
    choices = body.get("choices") or []
    txt = (choices[0].get("message") or {}).get("content") or "" if choices else ""
    if not txt.strip():
        return "EMPTY", "", dt                     # billed, no content -- retry
    if _CANNED in txt:
        return "canned", txt[:200], dt             # prompt never reached the model
    # 2026-09-03: claude-fable-5/image had 74/559 "ok" rows whose whole body was
    # "." (relay dud, prompt never answered) and they were scored as 0. A real
    # answer -- even a truncated <think> -- is never this short; treat as transport.
    if len(txt.strip()) < _DUD_CHARS and "<answer>" not in txt:
        return "DUD", txt[:200], dt
    return "ok", txt, dt


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="image", choices=["image", "oracle_text"])
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--smoke", type=int, default=None,
                    help="subsample to ~N items, stratified by split (cost probe)")
    ap.add_argument("--lines", nargs="*", default=["RELAY_A", "RELAY_A2"])
    ap.add_argument("--workers", type=int, default=5, help="per line")
    # 120 s, not the 60 s of the §7.1 short-probe prescription. Success latency
    # tracks how hard the item is -- gpt-5.4-mini's `image` median runs 21.8 s on
    # test_id but 56.3 s on ood_x (p90 102 s, max 121 s). A tight timeout is
    # therefore not a neutral loss: it cuts the *hard* splits preferentially and
    # would flatter the score of whatever survives. See §7.4.
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-passes", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-rows", default=None,
                    help="JSON {split: [row,...]} (or a repair report) limiting "
                         "the run to those rows -- e.g. re-collecting just the "
                         "instructions §7.6 rewrote")
    ap.add_argument("--refresh", action="store_true",
                    help="re-ask the selected items even if the JSONL already "
                         "has an ok response for them (the old row stays; the "
                         "scorer prefers the one whose prompt_sha matches)")
    args = ap.parse_args()

    conf = json.loads(_KEYS.read_text())
    cfgs = [conf[ln] for ln in args.lines]
    out = pathlib.Path(args.out or
                       _REPO / f"data/lnsdl_bench/strong_baseline/"
                               f"{args.model}__{args.condition}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    items, total = load_items(args.splits, args.smoke)
    if args.only_rows:
        sel = _selected_rows(pathlib.Path(args.only_rows))
        items = [it for it in items if it[2] in sel.get(it[1], ())]
        if not items:
            print(f"--only-rows {args.only_rows} selected nothing")
            return 1
    grand_total = sum(total[s] for s in args.splits)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add(rec["item_id"])
    if args.refresh:
        done -= {it[0] for it in items}

    lock = threading.Lock()
    fh = out.open("a")
    stat = Counter()
    lat: list[float] = []
    attempts = Counter()
    t0 = time.time()

    def one(item, cfg):
        item_id, split, idx, system, instruction, png, gt = item
        payload = _payload(args.model, args.condition, system, instruction,
                           png, gt, args.max_tokens)
        # LNSDL_SB_EXTRA_BODY (JSON, default none): merged into the request body.
        # Added 2026-09-04 for the local Qwen3-VL-32B row: on this benchmark's
        # VIKI-style system prompt the model's FIRST-token distribution is all
        # control tokens (<|im_end|> .56 / <tool_call> .23 / <|im_start|> .16),
        # so 56% of calls return an empty message. {"min_tokens":16,"bad_words":
        # [...]} forces content; the accommodation is a protocol note on that row.
        if _EXTRA_BODY:
            payload.update(_EXTRA_BODY)
        st, txt, dt = call(cfg, args.model, payload, args.timeout)
        rec = {"item_id": item_id, "split": split, "row": idx,
               "model": args.model, "condition": args.condition,
               "api_source": "relay", "status": st,
               "prompt_sha": prompt_sha(instruction),
               "latency_s": round(dt, 2), "response": txt if st == "ok" else txt[:200],
               "has_answer_tag": "<answer>" in txt if st == "ok" else False}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            stat[st] += 1
            attempts[item_id] += 1
            if st == "ok":
                lat.append(dt)
                done.add(item_id)
        return st

    for p in range(args.max_passes):
        todo = [it for it in items if it[0] not in done]
        if not todo:
            break
        # Dispatch alternates lines: one relay, two independent quotas.
        pools = [ThreadPoolExecutor(max_workers=args.workers) for _ in cfgs]
        futs = [pools[i % len(cfgs)].submit(one, it, cfgs[i % len(cfgs)])
                for i, it in enumerate(todo)]
        for f in futs:
            f.result()
        for pool in pools:
            pool.shutdown()
        print(f"  pass {p}: {len(todo)} attempted, {len(done)}/{len(items)} complete "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
    fh.close()

    wall = time.time() - t0
    n = len(items)
    calls = sum(stat.values())
    # `done` is seeded from the whole JSONL, so under --only-rows it also holds
    # items this run never selected; count only the selected ones or the line
    # reads "554/66".
    ok = sum(1 for it in items if it[0] in done)
    print(f"\n=== {args.model} / {args.condition} ===")
    if args.only_rows:
        print(f"  selected subset   : {n} rows via --only-rows {args.only_rows}")
    else:
        print(f"  smoke sample      : {n} / {grand_total} images = {n / grand_total:.1%} "
              f"of the split total")
    print(f"  completed         : {ok}/{n}" + (f"  (MISSING {n - ok})" if ok < n else ""))
    print(f"  API calls spent   : {calls}  ({calls / max(1, n):.2f} per item)")
    print(f"  statuses          : {dict(stat)}")
    if lat:
        s = sorted(lat)
        print(f"  ok latency        : med {s[len(s)//2]:.1f}s  p90 {s[int(.9*len(s))]:.1f}s  "
              f"max {s[-1]:.1f}s")
    print(f"  wall clock        : {wall:.0f}s  =>  {wall / max(1, n):.1f}s per item "
          f"(workers {args.workers} x {len(cfgs)} lines)")
    # An --only-rows subset is chosen by defect, not sampled, so extrapolating
    # its cost to the full split would be wrong in both directions.
    if ok and not args.only_rows:
        print(f"  EXTRAPOLATED full {grand_total} images: "
              f"{wall / max(1, ok) * grand_total / 3600:.1f} h, "
              f"{calls / max(1, ok) * grand_total:.0f} calls")
    print(f"  raw -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
