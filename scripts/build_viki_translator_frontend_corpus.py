#!/usr/bin/env python
"""序 39：VIKI 角色训练翻译前端语料（序 35-B 同款换 benchmark，plan.md 09-13）.

Trains a front-end FOR the translator role of the VIKI full-translation offload
arm (`eval_constant_offload.py --level VIKI-L2 --offload-mode full-translation`),
removing the prompt-transfer confound of reusing the agentic main-row checkpoint
there (the existing p61_full_translation_viki_l2_v341 row used S18_FINAL
out-of-role). Mirror of scripts/build_locus_translator_frontend_corpus.py with
the benchmark swapped; every deviation below is the VIKI contract, not a recipe
change:

  * data source   = data/viki/VIKI-L2/train.parquet + train image manifest
                    (image key = task_id, not row_index)
  * sketch        = make_sketch(gt_ops, "intent")  -- the VIKI build contract
                    (mechanical ops dropped, unload_carrier -> deposit)
  * naive ops for goals the GT judged already satisfied (absent from gt_ops):
        skip item in a carrier-family ROW -> deposit(item -> dest)   [the
                        intent-contract sketch form of a carried item; decided
                        per row, NOT from a global carrier vocabulary -- in
                        VIKI the same fruit is a carried item in dog_push_box
                        and an ordinary target elsewhere, so the LOCUS global
                        vocabulary rule would mislabel skipped fruits. 0 rows
                        hit this branch in practice: all 184 carrier-family
                        rows have empty skip]
        other item   -> acquire(item) + deposit(item -> dest)
  * row parity    = row_index set of the v341 reference corpus (7196 rows)

Contract (must mirror the eval arm the same way the LOCUS builder did):
  system  = build_slm_system_prompt() + "\n\n" + BASE_FRONTEND_INSTRUCTIONS
  user    = serialize_slm_input(scene_facts, affordance_context)   [+ image]
  target  = "[GOAL]\n{targets, active}\n\n[PLAN_SKETCH]\n[ops]"  -- nothing else
Judgment-free: [GOAL].active == [GOAL].targets; ops carry only
{op_id, op_type, entity, dest} -- no robot / deps / gaps.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.slm_interface import build_slm_system_prompt, serialize_slm_input  # noqa: E402
from scripts.audit_planning_tool_p1 import make_sketch  # noqa: E402
from scripts.build_lnsdl_sft_data import (  # noqa: E402
    load_train_image_index,
    parse_blocks_from_legacy_serializer,
)
from scripts.build_slm_training_data import (  # noqa: E402
    build_scene_facts_from_gt,
    extract_sample_data,
    reverse_map_time_steps,
)
from scripts.eval_constant_offload import BASE_FRONTEND_INSTRUCTIONS  # noqa: E402

SKETCH_VARIANT = "intent"  # the VIKI build contract (v341 corpus meta sketch_variant)
VIKI_ROOT = REPO / "data" / "viki" / "VIKI-L2"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-corpus",
                    default=str(REPO / "data/lnsdl/sft/v3_4_1/gpt56_terra_image_p513_obsbranch_v341_full"),
                    help="row_index parity source (同训练样本) + carrier vocabulary")
    ap.add_argument("--output-dir",
                    default=str(REPO / "data/lnsdl/sft/v3_4_1/viki_translator_frontend_v341"))
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all reference rows (smoke: small N)")
    return ap.parse_args()


def reference_rows(path: Path) -> tuple[list[int], set[str]]:
    """(sorted row_index list, carrier entity set) from the v341 reference corpus."""
    indices: list[int] = []
    carriers: set[str] = set()
    with open(path / "samples.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            indices.append(int(r["row_index"]))
            for op in json.loads(r.get("plan_json") or "[]"):
                if op.get("op_type") in ("push_carrier", "load_carrier", "unload_carrier"):
                    ent = (op.get("dest") if op.get("op_type") == "load_carrier" else op.get("entity"))
                    if ent:
                        carriers.add(str(ent))
    return sorted(set(indices)), carriers


def op_payload(op: Any) -> dict[str, Any]:
    return {"op_id": str(getattr(op, "op_id", "") or ""),
            "op_type": str(getattr(op, "op_type", "") or ""),
            "entity": getattr(op, "entity", None),
            "dest": getattr(op, "dest", None)}


CARRIER_OP_TYPES = ("push_carrier", "load_carrier", "unload_carrier")


def build_target(goal: dict[str, Any], sketch_ops: list[Any], carrier_row: bool) -> tuple[str, dict[str, Any]]:
    targets = [t for t in (goal.get("targets") or []) if isinstance(t, dict)]
    items = [str(t.get("item")) for t in targets if t.get("item")]
    dest_by_item = {str(t.get("item")): t.get("destination") for t in targets if t.get("item")}
    skip_items = [str(x) for x in (goal.get("skip") or [])]

    ops = [op_payload(op) for op in sketch_ops]
    synthesized = 0
    carrier_skip = 0
    for item in skip_items:
        dest = dest_by_item.get(item)
        if carrier_row:
            # intent contract: the carried item's sketch form is deposit-only
            # (unload_carrier -> deposit rewrite; load/push are mechanical).
            ops.append({"op_id": "", "op_type": "deposit", "entity": item, "dest": dest})
            synthesized += 1
            carrier_skip += 1
        else:
            ops.append({"op_id": "", "op_type": "acquire", "entity": item, "dest": None})
            ops.append({"op_id": "", "op_type": "deposit", "entity": item, "dest": dest})
            synthesized += 2
    for i, op in enumerate(ops, start=1):  # renumber: deps are stripped, ids stay unique
        op["op_id"] = f"op{i}"

    goal_out = {"targets": items, "active": list(items)}  # judgment-free: active == targets
    text = ("[GOAL]\n" + json.dumps(goal_out, ensure_ascii=False)
            + "\n\n[PLAN_SKETCH]\n" + json.dumps(ops, ensure_ascii=False))
    covered = {str(o.get("entity")) for o in ops if o.get("entity")} >= set(items)
    audit = {"n_targets": len(items), "n_skip": len(skip_items), "n_ops": len(ops),
             "n_synthesized_ops": synthesized, "n_carrier_skip": carrier_skip, "covered": covered}
    return text, audit


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = Path(args.reference_corpus)
    indices, carriers = reference_rows(ref)
    if args.max_samples:
        indices = indices[: args.max_samples]
    print(f"reference rows: {len(indices)}  carriers: {sorted(carriers)}")

    df = pd.read_parquet(VIKI_ROOT / "train.parquet")
    image_index = load_train_image_index(VIKI_ROOT)
    system_prompt = build_slm_system_prompt() + "\n\n" + BASE_FRONTEND_INSTRUCTIONS

    fam_counter: collections.Counter[str] = collections.Counter()
    synth_counter: collections.Counter[str] = collections.Counter()
    drops: collections.Counter[str] = collections.Counter()
    n_written = 0
    n_skip_rows = 0
    n_carrier_skip_total = 0
    preview: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []

    with open(out / "samples.jsonl", "w", encoding="utf-8") as fo:
        for row_index in indices:
            if row_index >= len(df):
                drops["row_index_out_of_range"] += 1
                continue
            sample_data = extract_sample_data(df.iloc[row_index])
            task_name = str(sample_data.get("task_name") or "unknown")
            task_id = str(sample_data.get("task_id") or "")
            image_meta = image_index.get(task_id)
            if not image_meta:
                drops["missing_image"] += 1
                continue
            try:
                scene_facts = build_scene_facts_from_gt(sample_data, idx=row_index)
                affordance_context = build_affordance_context(scene_facts)
                gt_ops = reverse_map_time_steps(sample_data)
                sketch_ops = make_sketch(gt_ops, SKETCH_VARIANT)
                goal = parse_blocks_from_legacy_serializer(scene_facts, gt_ops)["goal"]
            except Exception as exc:  # noqa: BLE001 -- audit, don't die row-wise
                drops[f"build_error:{type(exc).__name__}"] += 1
                continue
            carrier_row = any(getattr(op, "op_type", "") in CARRIER_OP_TYPES for op in gt_ops)
            target_text, audit = build_target(goal, sketch_ops, carrier_row)
            if not audit["covered"]:
                drops["target_not_covered_by_ops"] += 1
                continue
            if audit["n_skip"]:
                n_skip_rows += 1
                synth_counter[task_name] += audit["n_synthesized_ops"]
            n_carrier_skip_total += audit["n_carrier_skip"]
            fam_counter[task_name] += 1
            row_out = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": serialize_slm_input(scene_facts, affordance_context)},
                    {"role": "assistant", "content": target_text},
                ],
                "image_path": image_meta["image_path"],
                "image_source_split": "train",
                "prompt_variant": "lnsdl_agentic_lean",  # vl_data_contract required column; same as reference corpus/config
                "task_name": task_name,
                "task_id": task_id or task_name,
                "row_index": row_index,
                "modality": "translator_frontend",
                "goal_json": json.dumps(goal, ensure_ascii=False),
                "n_synthesized_ops": audit["n_synthesized_ops"],
            }
            fo.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            rows_out.append(row_out)
            if len(preview) < 5:
                preview.append(row_out)
            n_written += 1

    # ---- HF dataset serialization (run_vl_sft loads <dir>/train) -----------
    from datasets import Dataset  # local import: heavy
    ds = Dataset.from_list(rows_out)
    ds.to_parquet(str(out / "train.parquet"))
    ds.save_to_disk(str(out / "train"))

    # ---- gates -------------------------------------------------------------
    leak_hits = 0
    with open(out / "samples.jsonl", encoding="utf-8") as f:
        for line in f:
            asst = json.loads(line)["messages"][2]["content"]
            if any(tok in asst for tok in ('"status"', '"skip"', '"robot"', '"deps"', '"gaps"')):
                leak_hits += 1
    meta = {
        "arm": "viki_translator_frontend_v341 (plan 序 39 = 序 35-B 同款换 benchmark)",
        "sketch_variant": SKETCH_VARIANT,
        "judgment_free": True,
        "teacher": None,
        "reference_corpus": str(ref),
        "rows_written": n_written,
        "rows_with_skip_goals": n_skip_rows,
        "carrier_skip_rows": n_carrier_skip_total,
        "synthesized_ops_by_task": dict(synth_counter),
        "tasks": dict(fam_counter),
        "drops": dict(drops),
        "judgment_leak_rows": leak_hits,
        "carrier_vocabulary_note": "per-row carrier context, not a global vocabulary (VIKI carried items double as ordinary targets elsewhere)",
        "reference_carrier_entities": sorted(carriers),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (out / "sample_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    ok = (n_written == len(indices) - sum(drops.values()) and n_written >= 7000
          and leak_hits == 0 and not any(k.startswith("build_error") for k in drops))
    if args.max_samples:
        ok = leak_hits == 0
    print("GATE_OK" if ok else "GATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
