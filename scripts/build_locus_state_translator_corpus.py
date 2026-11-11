#!/usr/bin/env python
"""序 35 方案 A：LOCUS 状态翻译前端语料（review 9.11 W3 第二半，注册版）.

Trains a front-end that ONLY translates perception into symbolic state:
GT ``init_pos`` -> location relations of the instruction-relevant entities, in
the YOLO word-form (the vocabulary ``scene_facts_refiner._build_scene_relations``
emits and 序 34 showed is the fair interface). The externalized skip/active
verdict is then derived from these relations by the SAME rule judge the
full-translation arm already uses (``nsdl.verdict_perception.derive_verdicts``),
via ``eval_constant_offload.py --evidence-source trained_state``.

Contract (must mirror the eval pre-pass byte-for-byte,
``scripts/run_state_translator_frontend.py``):
  system  = build_slm_system_prompt() + "\n\n" + STATE_TRANSLATOR_FRONTEND_INSTRUCTIONS
  user    = serialize_slm_input(scene_facts, affordance_context)   [+ image]
  target  = "[STATE]\n{\"entities\": [...], \"relations\": [...]}"  -- nothing else

Judgment-free supervision:
  * relations are raw placements ("fork on table"), NEVER skip/active/status;
  * GT locations YOLO cannot express (kitchen work area, rack, cardboardbox,
    robot-held, carrier slots, room) yield NO relation -- the model learns to
    stay silent exactly where the YOLO channel is silent (三道过滤 discipline,
    memory locus-relation-and-crop-ceiling);
  * scope = ``instruction_candidates(scene_facts)`` -- the same instruction-only
    candidate set the constant detect queries, so the supervision carries no
    GT-target leak.

Same 6300 rows / images as the main corpus (row_index parity with
``gpt56_terra_locus_train_v341_schedall``), teacher-free.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.catalog import normalize_entity_name  # noqa: E402
from nsdl.slm_interface import build_slm_system_prompt, serialize_slm_input  # noqa: E402
from scripts.build_lnsdl_sft_data import (  # noqa: E402
    LOCUS_ROOT,
    load_locus_frame,
    load_locus_image_index,
)
from scripts.build_slm_training_data import (  # noqa: E402
    build_scene_facts_from_gt,
    extract_sample_data,
)
from scripts.eval_constant_offload import (  # noqa: E402
    STATE_TRANSLATOR_FRONTEND_INSTRUCTIONS,
    instruction_candidates,
)

# YOLO-vocabulary location map: GT ``init_pos`` location -> (verb, fixture),
# restricted to word forms ``_build_scene_relations`` can emit. MUST stay equal
# to ``nsdl.oracle_perception._YOLO_VOCAB_LOCATION_RELATIONS`` -- that module is
# eval-only and hard-refuses import from data construction, so the map is
# duplicated here and pinned by tests/test_state_translator_corpus.py.
YOLO_VOCAB_LOCATION_RELATIONS: dict[str, tuple[str, str]] = {
    "table": ("on", "table"),
    "cabinet": ("in", "cabinet"),
    "room_cabinet": ("in", "cabinet"),
    "sink": ("on", "sink"),
    "room_toaster": ("on", "toaster"),
}

_INSTANCE_SUFFIX_RE = re.compile(r"_\d+$")
_RELATION_WORDFORM_RE = re.compile(r"^\S+ (on|in) \S+$")
_LEAK_TOKENS = ('"active"', '"skip"', '"status"', '"robot"', '"deps"',
                '"already', "[GOAL]", "[PLAN_SKETCH]", "[PLAN]", "[TOOL_CALL]")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-corpus",
                    default=str(REPO / "data/lnsdl/sft/v3_4_1_locus/gpt56_terra_locus_train_v341_schedall"),
                    help="row_index parity source (同训练样本)")
    ap.add_argument("--locus-split", default="train")
    ap.add_argument("--output-dir",
                    default=str(REPO / "data/lnsdl/sft/v3_4_1_locus/locus_state_translator_v341"))
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all reference rows (smoke: small N)")
    return ap.parse_args()


def reference_row_indices(path: Path) -> list[int]:
    indices: set[int] = set()
    with open(path / "samples.jsonl", encoding="utf-8") as f:
        for line in f:
            indices.add(int(json.loads(line)["row_index"]))
    return sorted(indices)


def _base_name(instance: str) -> str | None:
    base = _INSTANCE_SUFFIX_RE.sub("", str(instance).strip())
    return normalize_entity_name(base) or (base.lower() or None)


def build_state_target(
    candidates: list[str],
    init_pos: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """[STATE] block: entities + YOLO-word-form relations for the candidates."""
    wanted: list[str] = []
    for name in candidates:
        norm = normalize_entity_name(name) or str(name).strip().lower()
        if norm and norm not in wanted:
            wanted.append(norm)

    entities: list[str] = []
    relations: list[str] = []
    loc_hist: collections.Counter[str] = collections.Counter()
    unexpressible: collections.Counter[str] = collections.Counter()

    def _add_entity(name: str) -> None:
        if name and name not in entities:
            entities.append(name)

    for instance, locations in (init_pos or {}).items():
        entity = _base_name(instance)
        if not entity or entity not in wanted:
            continue
        if locations is None:
            continue
        if hasattr(locations, "tolist"):
            locations = locations.tolist()
        if isinstance(locations, str):
            locations = [locations]
        _add_entity(entity)
        for location in locations or []:
            key = str(location).strip()
            mapped = YOLO_VOCAB_LOCATION_RELATIONS.get(key)
            if mapped is None:
                unexpressible[key] += 1
                continue
            verb, fixture = mapped
            if fixture == entity:  # parity with _emit: never ent == fix
                continue
            loc_hist[key] += 1
            _add_entity(fixture)
            rel = f"{entity} {verb} {fixture}"
            if rel not in relations:
                relations.append(rel)

    text = "[STATE]\n" + json.dumps(
        {"entities": entities, "relations": relations}, ensure_ascii=False)
    audit = {
        "n_candidates": len(wanted),
        "n_entities": len(entities),
        "n_relations": len(relations),
        "loc_hist": dict(loc_hist),
        "unexpressible": dict(unexpressible),
        "wordform_violations": sum(
            0 if _RELATION_WORDFORM_RE.match(r) else 1 for r in relations),
    }
    return text, audit


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    indices = reference_row_indices(Path(args.reference_corpus))
    if args.max_samples:
        indices = indices[: args.max_samples]
    print(f"reference rows: {len(indices)}")

    df = load_locus_frame(LOCUS_ROOT / args.locus_split / "data.parquet")
    image_index = load_locus_image_index(LOCUS_ROOT, args.locus_split)
    system_prompt = (build_slm_system_prompt() + "\n\n"
                     + STATE_TRANSLATOR_FRONTEND_INSTRUCTIONS)

    fam_counter: collections.Counter[str] = collections.Counter()
    loc_total: collections.Counter[str] = collections.Counter()
    unexpressible_total: collections.Counter[str] = collections.Counter()
    drops: collections.Counter[str] = collections.Counter()
    n_written = n_zero_relation_rows = wordform_violations = 0
    rel_count_total = 0
    preview: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []

    with open(out / "samples.jsonl", "w", encoding="utf-8") as fo:
        for row_index in indices:
            if row_index >= len(df):
                drops["row_index_out_of_range"] += 1
                continue
            image_meta = image_index.get(str(row_index))
            if not image_meta:
                drops["missing_image"] += 1
                continue
            row = df.iloc[row_index]
            sample_data = extract_sample_data(row)
            task_name = str(sample_data.get("task_name") or "unknown")
            try:
                gt = row["reward_model"]["ground_truth"]
                init_pos = gt.get("init_pos") or {}
                scene_facts = build_scene_facts_from_gt(sample_data, idx=row_index)
                affordance_context = build_affordance_context(scene_facts)
                candidates = instruction_candidates(scene_facts)
            except Exception as exc:  # noqa: BLE001 -- audit, don't die row-wise
                drops[f"build_error:{type(exc).__name__}"] += 1
                continue
            if not candidates:
                drops["no_instruction_candidates"] += 1
                continue
            target_text, audit = build_state_target(candidates, init_pos)
            if audit["wordform_violations"]:
                wordform_violations += audit["wordform_violations"]
            if audit["n_relations"] == 0:
                n_zero_relation_rows += 1
            rel_count_total += audit["n_relations"]
            loc_total.update(audit["loc_hist"])
            unexpressible_total.update(audit["unexpressible"])
            fam_counter[task_name.split("_")[0]] += 1
            row_out = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": serialize_slm_input(scene_facts, affordance_context)},
                    {"role": "assistant", "content": target_text},
                ],
                "image_path": image_meta["image_path"],
                "image_source_split": args.locus_split,
                "prompt_variant": "lnsdl_agentic_lean",  # vl_data_contract required column; same as reference corpus/config
                "task_name": task_name,
                "task_id": str(sample_data.get("task_id") or task_name),
                "row_index": row_index,
                "modality": "state_translator_frontend",
                "n_relations": audit["n_relations"],
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
            if any(tok in asst for tok in _LEAK_TOKENS):
                leak_hits += 1
    meta = {
        "arm": "locus_state_translator_v341 (plan 序 35 方案 A)",
        "judgment_free": True,
        "teacher": None,
        "reference_corpus": args.reference_corpus,
        "rows_written": n_written,
        "rows_with_zero_relations": n_zero_relation_rows,
        "relations_total": rel_count_total,
        "location_hist": dict(loc_total),
        "unexpressible_location_hist": dict(unexpressible_total),
        "families": dict(fam_counter),
        "drops": dict(drops),
        "judgment_leak_rows": leak_hits,
        "relation_wordform_violations": wordform_violations,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (out / "sample_preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    ok = (leak_hits == 0 and wordform_violations == 0
          and not any(k.startswith("build_error") for k in drops))
    if not args.max_samples:
        ok = ok and n_written >= 6000 and n_written == len(indices) - sum(drops.values())
    print("GATE_OK" if ok else "GATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
