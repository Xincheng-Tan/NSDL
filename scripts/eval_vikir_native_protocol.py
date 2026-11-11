#!/usr/bin/env python
"""VIKI-R official weights under their NATIVE protocol (plan.md 序 36, review 9.11 W3).

The stored VIKI-R rows (``results/lnsdl/baselines/constant_offload/
locus_transfer_vikir_official_{7b,3b}``) were collected through OUR op-level
protocol (``eval_constant_offload.py --offload-mode none --frontend base``), so
their err column (71%) is protocol mismatch, not planning failure. This runner
sends the byte-identical prompt of ``eval/VIKI-L2/qwen.py`` (``build_messages``:
VIKI-R's own system prompt + image + task description) to a vLLM OpenAI server
hosting the official checkpoint, with the same decoding (temperature 0,
max_tokens 2000), and stores each row as an artifact in our bucket layout
(``<out_root>/<bucket>/<split>/<stem>.json``) so the existing tooling reads it
unchanged: ``scratch/locus_pair_s45.py`` (paired acc / err),
``scripts/replay_official_judger_taxonomy.py`` (序 22 infeasible / goal_unsat /
achieved_but_long split) and the LOCUS null-plan-aware
``nsdl.benchmark_scoring.compute_locus_l2_acc``.

Scoring (both stored):
  ``benchmark_scores.viki_l2_plan_score``  official ``viki_2.compute_score``
                                          (0.1 format + 0.9 acc, exactly qwen.py)
  ``benchmark_scores.locus_l2_acc``       LOCUS ruler (null-GT: correct iff empty)
``result_status`` is ``error`` only when the model gave no parsable
``<answer>[...]</answer>`` list (type ``unparsable_answer``) or the API failed;
those rows carry an empty plan and 0 on both rulers, mirroring "the runtime
rejected the submission" in our protocol. Parsable rows are ``ok`` even when the
official judger rejects the plan (that is the taxonomy's job).

Usage (vLLM must already be up):
  vllm serve checkpoints/vikir_official/Qwen2.5VL-7B-Instruct-VIKI-R-2 \
      --served-model-name Qwen2.5-VL-7B-Instruct --port 8020
  python scripts/eval_vikir_native_protocol.py --level LOCUS \
      --splits test_id,ood_s,ood_m,ood_x --api-base http://127.0.0.1:8020/v1 \
      --model Qwen2.5-VL-7B-Instruct --result-bucket locus_vikir_native_7b
"""
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "verl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nsdl.benchmark_scoring import compute_locus_l2_acc  # noqa: E402
from nsdl.prompt_adapter import (  # noqa: E402
    build_sample_stem,
    convert_arrays_to_native,
    load_benchmark_samples,
)
from nsdl.validator import validate_l2_plan  # noqa: E402
from verl.utils.reward_score import viki_2  # noqa: E402

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
DEFAULT_OUT = REPO_ROOT / "results" / "lnsdl" / "baselines" / "native_protocol"
LOCUS_SPLITS = ("test_id", "ood_s", "ood_m", "ood_x")
VIKI_SPLITS = ("test", "val")


def _load_qwen_module():
    """eval/VIKI-L2/qwen.py (not a package); importing it builds an OpenAI client
    pointed at 0.0.0.0:8000 but performs no network call."""
    spec = importlib.util.spec_from_file_location("vikir_qwen", REPO_ROOT / "eval" / "VIKI-L2" / "qwen.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_plan(text: str | None) -> tuple[list[dict[str, Any]], bool]:
    """Mirror of ``viki_2.acc_reward``'s extraction: last <answer> block ->
    ast.literal_eval -> list of {step, actions}. Returns (plan, parsed)."""
    if not text:
        return [], False
    match = _ANSWER_RE.search(text)
    if not match:
        return [], False
    try:
        obj = ast.literal_eval(match.group(1).strip())
    except Exception:  # noqa: BLE001
        return [], False
    if isinstance(obj, dict) and "step" in obj:
        obj = [obj]
    if not isinstance(obj, list):
        return [], False
    # An explicit empty list is a parsable "nothing to do" answer (the correct
    # LOCUS null-GT submission); only a non-list / non-step payload is unparsable.
    plan = [step for step in obj if isinstance(step, dict) and isinstance(step.get("actions"), dict)]
    if obj and not plan:
        return [], False
    return plan, True


def native_task_description(sample) -> str:
    """qwen.py: ``sample['prompt'][1]['content'].replace('<image>', '').strip()``."""
    raw = sample.raw_sample or {}
    prompt = raw.get("prompt")
    if isinstance(prompt, (list, tuple)) and len(prompt) > 1 and isinstance(prompt[1], dict):
        return str(prompt[1].get("content") or "").replace("<image>", "").strip()
    return str(sample.instruction or "").replace("<image>", "").strip()


def run_one(sample, *, qwen, client, model: str, max_tokens: int, retries: int) -> dict[str, Any]:
    ground_truth = convert_arrays_to_native(sample.ground_truth or {})
    robots = ground_truth.get("robots") or {}
    b64 = base64.b64encode(sample.image_bytes or b"").decode("utf-8")
    messages = qwen.build_messages(native_task_description(sample), robots, b64)
    t0 = time.time()
    text: str | None = None
    api_error: str | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=max_tokens)
            text = resp.choices[0].message.content
            api_error = None
            break
        except Exception as exc:  # noqa: BLE001
            api_error = f"{type(exc).__name__}: {exc}"[:300]
            time.sleep(min(30, 2 ** attempt))
    plan, parsed = parse_plan(text)
    official = float(viki_2.compute_score(text or "", ground_truth))
    gt_steps = ground_truth.get("time_steps") or []
    locus_acc = compute_locus_l2_acc(plan, ground_truth) if parsed else 0.0
    # The official judger can raise on some emitted commands (e.g. Place onto a
    # non-location: `'Position' object has no attribute 'type'`); acc_reward's
    # try/except already scores those 0, so record the exception instead of
    # dropping the row (replay taxonomy reads it as JUDGER_EXCEPTION).
    validation = None
    if parsed:
        try:
            validation = validate_l2_plan(plan, ground_truth)
        except Exception as exc:  # noqa: BLE001
            from types import SimpleNamespace
            validation = SimpleNamespace(success=False, error_desc=f"JUDGER_EXCEPTION:{type(exc).__name__}: {exc}"[:300],
                                         eval_commands=[{k: f"<{','.join(map(str, v))}>" for k, v in (st.get('actions') or {}).items()} for st in plan])
    status = "ok" if (text is not None and parsed) else "error"
    error = None
    if status == "error":
        error = {"type": "api_error" if text is None else "unparsable_answer", "message": api_error or "no <answer>[...]</answer> list"}
    return {
        "sample": {
            "level": sample.level,
            "split": sample.split,
            "sample_index": sample.sample_index,
            "task_id": sample.task_id,
            "task_name": sample.task_name,
            "instruction": sample.instruction,
            "robot_inventory": dict(sample.robot_inventory or {}),
            "image_path": sample.image_path,
            "artifact_stem": build_sample_stem(sample),
        },
        "protocol": {
            "name": "vikir_native",
            "prompt_source": "eval/VIKI-L2/qwen.py:build_messages",
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "tools": None,
            "phi": None,
        },
        "native_task_description": native_task_description(sample),
        "raw_response": text,
        "answer_parsed": parsed,
        "plan": plan,
        "ground_truth": ground_truth,
        "validation": (
            {"success": validation.success, "error_desc": validation.error_desc, "eval_commands": validation.eval_commands}
            if validation is not None else None
        ),
        "benchmark_scores": {
            "viki_l2_plan_score": official,
            "viki_l2_official_acc": 1.0 if official >= 0.9 else 0.0,
            "locus_l2_acc": locus_acc,
            "null_gt": not gt_steps,
        },
        "result_status": status,
        "error": error,
        "runtime_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="LOCUS", choices=["LOCUS", "VIKI-L2"])
    ap.add_argument("--splits", default=None, help="comma list; default all splits of the level")
    ap.add_argument("--api-base", default="http://127.0.0.1:8020/v1")
    ap.add_argument("--model", default="Qwen2.5-VL-7B-Instruct", help="served-model-name of the vLLM server")
    ap.add_argument("--result-bucket", required=True)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI  # noqa: WPS433

    qwen = _load_qwen_module()
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    splits = [s.strip() for s in (args.splits or ",".join(LOCUS_SPLITS if args.level == "LOCUS" else VIKI_SPLITS)).split(",") if s.strip()]
    out_root = Path(args.out_root) / args.result_bucket
    summary: dict[str, Any] = {}
    for split in splits:
        samples = load_benchmark_samples(level=args.level, split=split)
        if args.max_samples:
            samples = samples[: args.max_samples]
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        todo = []
        for s in samples:
            path = split_dir / f"{build_sample_stem(s)}.json"
            if path.exists() and not args.overwrite:
                continue
            todo.append((s, path))
        print(f"[{args.level}/{split}] {len(samples)} samples, {len(todo)} to run -> {split_dir}", flush=True)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, s, qwen=qwen, client=client, model=args.model,
                              max_tokens=args.max_tokens, retries=args.retries): path for s, path in todo}
            for fut in concurrent.futures.as_completed(futs):
                path = futs[fut]
                try:
                    artifact = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {path.name}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
                done += 1
                if done % 50 == 0:
                    print(f"  [{split}] {done}/{len(todo)}", flush=True)
        # split summary from disk (covers previously written rows too)
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(split_dir.glob("*.json"))]
        n = len(rows)
        bs = [r.get("benchmark_scores") or {} for r in rows]
        summary[split] = {
            "n": n,
            "official_acc": sum(b.get("viki_l2_official_acc", 0.0) for b in bs) / n if n else None,
            "locus_l2_acc": sum(b.get("locus_l2_acc", 0.0) for b in bs) / n if n else None,
            "err": sum(1 for r in rows if r.get("result_status") != "ok"),
            "unparsable": sum(1 for r in rows if (r.get("error") or {}).get("type") == "unparsable_answer"),
            "null_gt": sum(1 for b in bs if b.get("null_gt")),
        }
        print(f"[{args.level}/{split}] {summary[split]}", flush=True)
    (out_root / "summary.json").write_text(json.dumps({
        "level": args.level, "model": args.model, "api_base": args.api_base, "protocol": "vikir_native (eval/VIKI-L2/qwen.py)",
        "splits": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
