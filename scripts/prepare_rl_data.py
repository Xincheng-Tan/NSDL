#!/usr/bin/env python3
"""Convert current SFT data → verl-format parquet for GRPO training.

Two modes for the train split:
  1. --train-jsonl data/v2/slm_dataset/all/samples.jsonl  (recommended:
     reads target_dag + robots directly; covers S1/S2/S3 without VIKI-L2 lookup)
  2. (legacy) --sft-dataset data/v2/slm_sft/train  — HF Dataset + VIKI-L2 GT join
     (only S1 task_ids match; S2/S3 fall back to in-sample GT)

Output columns: prompt, reward_model, data_source.
With ``--prompt-modality vl`` or ``--include-images``, the output also carries
the VIKI-L2 ``images`` column for verl multimodal rollouts.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from nsdl.vl_prompt_variants import (  # noqa: E402
    PROMPT_VARIANTS,
    apply_vl_system_prompt_variant,
    apply_vl_prompt_variant,
    normalize_prompt_variant,
)
from nsdl.prompt_adapter import LOCUS_LEVEL  # noqa: E402

DATASET_VERSION = os.environ.get("DATASET_VERSION", "v2")
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / DATASET_VERSION


def build_gt_index(viki_parquet: Path) -> dict[str, dict]:
    """Index VIKI-L2 ground_truth by task_id (first occurrence wins)."""
    df = pd.read_parquet(viki_parquet)
    index: dict[str, dict] = {}
    for _, row in df.iterrows():
        gt = _jsonable(row["reward_model"]["ground_truth"])
        tid = gt.get("task_id")
        if tid and tid not in index:
            index[tid] = gt
    return index


def build_image_index(viki_parquet: Path) -> dict[str, list[dict[str, Any]]]:
    """Index VIKI-L2 image payloads by task_id (first occurrence wins)."""
    df = pd.read_parquet(viki_parquet, columns=["reward_model", "images"])
    index: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        gt = _jsonable(row["reward_model"]["ground_truth"])
        tid = gt.get("task_id") if isinstance(gt, dict) else None
        if tid and tid not in index:
            images = _jsonable(row["images"])
            if images:
                index[str(tid)] = images
    return index


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:
        pass
    return obj


def main():
    # P5.10.4 anti-leak assert: oracle perception bracket is eval-only.
    assert "nsdl.oracle_perception" not in sys.modules, (
        "nsdl.oracle_perception (eval-only bracket backend) must not be "
        "imported during RL data construction."
    )
    assert os.environ.get("NSDL_ORACLE_PERCEPTION") != "eval-only", (
        "NSDL_ORACLE_PERCEPTION must not be set in a data-construction process."
    )
    # P10.S1 anti-leak assert: the A-axis inversion arm is eval-only.
    assert "nsdl.verdict_perception" not in sys.modules, (
        "nsdl.verdict_perception (eval-only A-axis inversion arm) must not be "
        "imported during RL data construction."
    )
    assert os.environ.get("NSDL_VERDICT_PERCEPTION") != "eval-only", (
        "NSDL_VERDICT_PERCEPTION must not be set in a data-construction process."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["enactgraph", "lnsdl"],
        default="enactgraph",
        help="enactgraph = legacy single-turn VL GRPO data. lnsdl = agentic "
             "multi-turn tool RL (P5.3): data_source='lnsdl', lean prompt, "
             "lnsdl_sample for rollout context reconstruction, ground_truth "
             "{gt_dag_json, gt_goal_json, time_steps, robots}.",
    )
    parser.add_argument(
        "--lnsdl-output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "lnsdl" / "rl" / "v2" / "viki_l2_train",
        help="Output dir for --mode lnsdl (writes train.parquet + test.parquet).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap samples per split (smoke testing only).",
    )
    parser.add_argument(
        "--probe-size", type=int, default=128,
        help="(--mode lnsdl) Size of the fixed held-in train probe set (T5).",
    )
    parser.add_argument(
        "--no-probe", action="store_true",
        help="(--mode lnsdl) Do not emit probe.parquet.",
    )
    parser.add_argument(
        "--lnsdl-source",
        choices=["viki_l2", "locus"],
        default="viki_l2",
        help="(--mode lnsdl) Which corpus the RL prompts come from. "
             "viki_l2 = data/viki/VIKI-L2 (the v1/v2/v3 parquets; DEFAULT, byte "
             "path unchanged). locus = data/lnsdl_bench/ramp63k (A0b G1 / P9). "
             "Only the corpus differs -- prompt construction, GT extraction and "
             "the rollout contract are the same code.",
    )
    parser.add_argument(
        "--lnsdl-locus-train-split", default="train",
        help="(--lnsdl-source locus) LOCUS split used as the RL TRAIN pool.",
    )
    parser.add_argument(
        "--lnsdl-locus-test-split", default="test_id",
        help="(--lnsdl-source locus) LOCUS split used as the in-training VAL "
             "set. Must stay an ID split: ood_s/ood_m/ood_x are the reported "
             "generalisation axes and are NEVER read here (VIKI-R hygiene).",
    )
    parser.add_argument(
        "--row-group-size", type=int, default=64,
        help="(--mode lnsdl) Parquet row-group size. LOCUS renders are ~2.4 MB "
             "each, so a single row group would push the nested `images` column "
             "past Arrow's 2 GB per-chunk ceiling and make the file unreadable "
             "by anything but iter_batches (memory: locus-train-parquet-unreadable).",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=DEFAULT_DATA_ROOT / "slm_dataset" / "all" / "samples.jsonl",
        help="Merged samples.jsonl (preferred). If the file exists, this is "
             "used instead of --sft-dataset (HF Dataset mode).",
    )
    parser.add_argument(
        "--sft-dataset",
        type=Path,
        default=DEFAULT_DATA_ROOT / "slm_sft" / "train",
        help="HF Dataset dir (legacy mode; used when --train-jsonl not present).",
    )
    parser.add_argument(
        "--viki-parquet",
        type=Path,
        default=REPO_ROOT / "data" / "viki" / "VIKI-L2" / "train.parquet",
        help="Original VIKI-L2 train.parquet with ground_truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "enactgraph_rl" / "train.parquet",
        help="Output parquet",
    )
    parser.add_argument(
        "--test-viki-parquet",
        type=Path,
        default=REPO_ROOT / "data" / "viki" / "VIKI-L2" / "test.parquet",
        help="VIKI-L2 test.parquet for val split",
    )
    parser.add_argument(
        "--test-sft-dataset",
        type=Path,
        default=None,
        help="HF Dataset dir for test split (optional)",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "enactgraph_rl" / "test.parquet",
        help="Output test parquet",
    )
    parser.add_argument(
        "--prompt-modality",
        choices=["text", "vl"],
        default="text",
        help="Use text prompts or Qwen2.5-VL prompts with a leading <image> token.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Attach the VIKI-L2 images column. Enabled automatically for --prompt-modality=vl.",
    )
    parser.add_argument(
        "--prompt-variant",
        default="full",
        choices=PROMPT_VARIANTS,
        help=(
            "Text prompt variant for VL ablations. full keeps YOLO-derived text; "
            "image_only removes pre-injected perception text while keeping "
            "instruction, robots, catalog entities, and the image column; "
            "full_no_entities keeps full perception text and removes only "
            "[ENTITIES]; lean additionally removes [ENTITIES]."
        ),
    )
    args = parser.parse_args()

    if args.mode == "lnsdl":
        run_lnsdl_mode(args)
        return

    include_images = bool(args.include_images or args.prompt_modality == "vl")
    prompt_variant = normalize_prompt_variant(args.prompt_variant)

    # ------- Build train split -------
    gt_index = build_gt_index(args.viki_parquet)
    print(f"  {len(gt_index)} unique task_ids indexed from VIKI-L2")
    train_image_index = build_image_index(args.viki_parquet) if include_images else None
    if train_image_index is not None:
        print(f"  {len(train_image_index)} train images indexed from VIKI-L2")

    if args.train_jsonl.exists():
        print(f"Loading train data from jsonl: {args.train_jsonl} ...")
        rows = _convert_from_jsonl(
            args.train_jsonl,
            gt_index,
            split_name="train",
            prompt_modality=args.prompt_modality,
            prompt_variant=prompt_variant,
            image_index=train_image_index,
        )
    else:
        print(f"Loading SFT dataset (HF) from {args.sft_dataset} ...")
        sft_ds = load_from_disk(str(args.sft_dataset))
        print(f"  {len(sft_ds)} samples")
        rows = _convert_split(
            sft_ds,
            gt_index,
            split_name="train",
            prompt_modality=args.prompt_modality,
            prompt_variant=prompt_variant,
            image_index=train_image_index,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} train rows → {args.output}")

    # ------- Build test split (from original SLM samples if SFT test not available) -------
    slm_test_path = DEFAULT_DATA_ROOT / "slm_dataset" / "test" / "samples.jsonl"
    if args.test_sft_dataset and args.test_sft_dataset.exists():
        print(f"\nLoading test SFT dataset from {args.test_sft_dataset} ...")
        test_ds = load_from_disk(str(args.test_sft_dataset))
        gt_test_index = build_gt_index(args.test_viki_parquet)
        test_image_index = build_image_index(args.test_viki_parquet) if include_images else None
        test_rows = _convert_split(
            test_ds,
            gt_test_index,
            split_name="test",
            prompt_modality=args.prompt_modality,
            prompt_variant=prompt_variant,
            image_index=test_image_index,
        )
    elif slm_test_path.exists():
        print(f"\nBuilding test split from {slm_test_path} ...")
        gt_test_index = build_gt_index(args.test_viki_parquet)
        test_image_index = build_image_index(args.test_viki_parquet) if include_images else None
        test_rows = _convert_from_jsonl(
            slm_test_path,
            gt_test_index,
            split_name="test",
            prompt_modality=args.prompt_modality,
            prompt_variant=prompt_variant,
            image_index=test_image_index,
        )
    else:
        print(f"\nNo {slm_test_path}; building ID test split directly from {args.test_viki_parquet} ...")
        test_rows = _convert_from_viki_parquet(
            args.test_viki_parquet,
            split_name="test",
            prompt_modality=args.prompt_modality,
            include_images=include_images,
            prompt_variant=prompt_variant,
        )

    if test_rows:
        df_test = pd.DataFrame(test_rows)
        df_test.to_parquet(args.test_output)
        print(f"Saved {len(df_test)} test rows → {args.test_output}")


# ---------------------------------------------------------------------------
# Task-blind structural flags for the lnsdl RL parquet (docs/RL_debug.md §8).
# Both are computed ONLY from the GT DAG / GT goal structure — no task_name,
# no instruction parsing — so they stay within docs/agent.md task-blindness.
# ---------------------------------------------------------------------------
# Ops that mark NON-trivial mechanical structure (cabinet/open, cross-robot
# handoff, carrier relay, tool application, activation, inspection). These are
# exactly what schedule.complete(O,A) synthesizes (docs/dispatcher.md); a plan
# containing any of them is not a plain single-robot pick-and-place.
_NONTRIVIAL_OP_TYPES = frozenset({
    "open", "handoff", "carry_to",
    "load_carrier", "push_carrier", "unload_carrier",
    "process_at", "apply_tool", "activate", "inspect",
})
_ACQUIRE_OP_TYPES = frozenset({"acquire", "local_acquire"})


def _trivial_structure(gt_dag: list[dict] | None) -> bool:
    """True iff the GT plan is structurally trivial per docs/agent.md
    Mechanical-Synthesis Gating Rule: single robot, only pick-and-place
    primitives, a single acquire->deposit pair (no cabinet/open, interleave,
    or cross-robot custody/handoff/carrier).

    Keyed on op_types + robot count ONLY. NOT on deps: reverse_map_time_steps
    chains same-robot sequential deps even on trivial plans, so deps is not a
    valid triviality discriminator (see RL_debug §8 T1 investigation).
    """
    if not gt_dag:
        return False
    robots = {op.get("robot") for op in gt_dag if op.get("robot")}
    if len(robots) > 1:
        return False
    if any(op.get("op_type") in _NONTRIVIAL_OP_TYPES for op in gt_dag):
        return False
    n_acquire = sum(op.get("op_type") in _ACQUIRE_OP_TYPES for op in gt_dag)
    n_deposit = sum(op.get("op_type") == "deposit" for op in gt_dag)
    return n_acquire == 1 and n_deposit == 1


def _has_skip_decision(gt_goal: dict | None) -> bool:
    """True iff the GT [GOAL] deliberately skips an already-placed target
    (active set is a strict subset of the candidate targets). These are the
    ID state-families that carry the R_goal skip-decision signal that
    generalizes to OOD acquisition (RL_debug §8 T2). Task-blind: reads only the
    serialized GT goal (top-level ``skip`` list / per-target ``status``).
    """
    if not isinstance(gt_goal, dict):
        return False
    if gt_goal.get("skip"):
        return True
    return any(
        isinstance(t, dict) and t.get("status") == "skip"
        for t in (gt_goal.get("targets") or [])
    )


# ---------------------------------------------------------------------------
# --mode lnsdl: agentic multi-turn tool RL parquet (P5.3)
# ---------------------------------------------------------------------------
def run_lnsdl_mode(args) -> None:
    """Emit the lnsdl RL parquet (train + ID-test) for DAPO multi-turn rollout.

    Contract consumed by verl/.../vllm_multiturn_rollout.py + reward_score/lnsdl.py:
      * data_source = 'lnsdl'
      * prompt      = lean agentic initial messages (system + user, <image>) —
                      identical to build_initial_messages(scene_facts, ctx).
      * images      = VIKI-L2 image payload (verl multimodal).
      * lnsdl_sample= minimal BenchmarkSample surface (instruction, prompt-derived
                      robot_inventory, image bytes b64, ground_truth.init_pos,
                      ids) so the rollout reconstructs IDENTICAL scene_facts via
                      scene_parser.parse + NoOp refiner (mirrors the eval harness).
      * reward_model.ground_truth = {task_id, task_name, instruction, robots,
                      time_steps, gt_dag_json, gt_goal_json}.
    OOD val.parquet is NEVER read here (VAL = ID test.parquet, VIKI-R hygiene).
    """
    source = getattr(args, "lnsdl_source", "viki_l2")
    if source == "locus":
        level = LOCUS_LEVEL
        plan = (
            ("train", getattr(args, "lnsdl_locus_train_split", "train"), "train.parquet"),
            ("test", getattr(args, "lnsdl_locus_test_split", "test_id"), "test.parquet"),
        )
        for _, split_name, _ in plan:
            if split_name.startswith("ood"):
                raise SystemExit(
                    f"--lnsdl-source locus refuses split {split_name!r}: the OOD "
                    "grids are the reported generalisation axes and must never "
                    "enter RL training or in-training validation."
                )
    else:
        level = "VIKI-L2"
        plan = (("train", "train", "train.parquet"), ("test", "test", "test.parquet"))

    out_dir: Path = args.lnsdl_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    row_group_size = int(getattr(args, "row_group_size", 64) or 64)
    train_rows: list[dict] | None = None
    for role, split_name, fname in plan:
        rows = _convert_lnsdl(split_name, limit=args.limit, level=level, source=source)
        path = out_dir / fname
        _write_lnsdl_parquet(rows, path, row_group_size=row_group_size)
        print(f"[lnsdl] saved {len(rows)} rows ({level}/{split_name} → {role}) → {path}")
        if role == "train":
            train_rows = rows

    # T5 (docs/RL_debug.md §8): a small FIXED, stratified held-in TRAIN probe
    # for a stationary monitoring trend. Dynamic sampling makes the training
    # batch non-stationary, so early5/late5 on it is not a learning signal;
    # this probe gives score/acc/r_goal/n_tool_calls on a constant sample set.
    # It is a train subset (held-in) used ONLY for monitoring — never for
    # optimizer updates or checkpoint selection (docs/agent.md split hygiene).
    if train_rows and not args.no_probe:
        if source == "locus":
            _emit_locus_probe(train_rows, out_dir, size=args.probe_size,
                              row_group_size=row_group_size)
        else:
            _emit_lnsdl_probe(train_rows, out_dir, size=args.probe_size)


def _write_lnsdl_parquet(rows: list[dict], path: Path, *, row_group_size: int) -> None:
    """Write with a bounded row-group size.

    ``pandas.to_parquet`` defaults to ONE row group. With VIKI-L2's 600x337
    renders that is harmless; with LOCUS's 1920x1080 renders the single nested
    ``images`` column crosses Arrow's 2 GB per-chunk ceiling, and the file then
    fails to load through *any* whole-table reader (pandas / read_table /
    read_row_group all raise ArrowNotImplementedError, and the source LOCUS
    split already demonstrates this -- memory: locus-train-parquet-unreadable).
    verl reads through ``datasets.load_dataset`` which is chunk-safe either way,
    but every audit and analysis script we own goes through pandas.
    """
    df = pd.DataFrame(rows)
    df.to_parquet(path, row_group_size=row_group_size)


def _emit_lnsdl_probe(train_rows: list[dict], out_dir: Path, *, size: int = 128) -> None:
    """Write probe.parquet: a deterministic, skip-stratified TRAIN subset."""
    def _tid(r: dict) -> str:
        return str((r.get("reward_model") or {}).get("ground_truth", {}).get("task_id") or "")

    def _is_skip(r: dict) -> bool:
        return bool((r.get("reward_model") or {}).get("ground_truth", {}).get("has_skip_decision"))

    skip_rows = sorted([r for r in train_rows if _is_skip(r)], key=_tid)
    nonskip_rows = sorted([r for r in train_rows if not _is_skip(r)], key=_tid)
    n_skip = min(len(skip_rows), size // 2)
    n_nonskip = min(len(nonskip_rows), size - n_skip)
    chosen = sorted(skip_rows[:n_skip] + nonskip_rows[:n_nonskip], key=_tid)
    if not chosen:
        print("[lnsdl] probe: no rows selected, skipping")
        return
    # Tag each probe row with a skip-stratified data_source so verl's
    # process_validation_metrics reports skip / nonskip as SEPARATE
    # data_sources (val-core/lnsdl_skip/... vs val-core/lnsdl_nonskip/...;
    # the core-vs-aux prefix follows max group size -- see docs/debug_rl.md
    # §1.3b, where grouping was fixed to sample identity so every group is a
    # singleton and mean@1 moved from val-aux to val-core).
    # This makes the nonskip half a valid proxy for the official distribution
    # trend (only ~2% of official train is skip) and exposes whether probe
    # improvements come from generalisation or just the upweighted skip family.
    # See docs/RL_debug.md §4 Problem 4.
    for row in chosen:
        row["data_source"] = "lnsdl_skip" if _is_skip(row) else "lnsdl_nonskip"
    path = out_dir / "probe.parquet"
    pd.DataFrame(chosen).to_parquet(path)
    ids_path = out_dir / "probe_task_ids.json"
    ids_path.write_text(json.dumps(
        {"n_total": len(chosen), "n_skip": n_skip, "n_nonskip": n_nonskip,
         "task_ids": [_tid(r) for r in chosen]},
        indent=2,
    ))
    print(f"[lnsdl] saved {len(chosen)} probe rows ({n_skip} skip / {n_nonskip} nonskip) → {path}")
    print(f"[lnsdl] probe task_ids → {ids_path}")


def _load_r_task():
    """Import the *production* axis-T scorer, or return None.

    ``import verl`` alone succeeds as an empty namespace package (the installed
    .pth points at a deleted tree -- memory: three-gpu-migration-decision), so
    the in-repo copy must be put on the path explicitly and the import verified
    by actually resolving the symbol.
    """
    verl_dir = str(REPO_ROOT / "verl")
    if verl_dir not in sys.path:
        sys.path.insert(0, verl_dir)
    try:
        from verl.utils.reward_score.lnsdl import _r_task  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[lnsdl] WARNING: cannot import the production _r_task ({exc}); "
              f"gt_r_task_reachable will be omitted.")
        return None
    return _r_task


_LOCUS_CELL_RE = __import__("re").compile(r"^(?P<family>[A-Za-z0-9]+)_s(?P<s>\d+)m(?P<m>\d+)$")


def _locus_s_level(cell: str) -> int | None:
    """S-axis level parsed from the LOCUS cell name (`F4_s2m2` -> 2)."""
    m = _LOCUS_CELL_RE.match(str(cell or ""))
    return int(m.group("s")) if m else None


def _emit_locus_probe(
    train_rows: list[dict],
    out_dir: Path,
    *,
    size: int = 512,
    row_group_size: int = 64,
) -> None:
    """Write probe.parquet: a deterministic, CELL-stratified LOCUS TRAIN subset.

    Two deliberate differences from the VIKI probe:

    1. **Stratum = grid cell, allocated proportionally.** The first VIKI probe
       was not merely small, it was *family-biased* (10-1 over-represented 3.3x),
       and that bias does not wash out with n -- so probe numbers from two
       differently-built probe files were never comparable (memory:
       rl-probe-prefix-rule-family-biased). LOCUS's whole design is a balanced
       S x M grid; a probe that does not preserve the cell proportions throws
       away the one property that makes per-cell reading possible.
    2. **Row identity = `row_uid`, not `task_id`.** LOCUS `task_id` IS the cell,
       shared by hundreds of rows, so a task_id-keyed probe manifest would name
       ~30 distinct strings for 512 rows and silently look like a 30-row probe.

    The S-axis tag (`_s0` vs `_s1p`) mirrors the VIKI skip/nonskip tagging: it
    makes verl's ``process_validation_metrics`` report the state-judgement half
    (facet A, s>=1) separately from the pure-structure half, instead of an
    aggregate in which a facet-A collapse can hide behind facet-B saturation.
    It is a *reporting* split only -- the reward math is identical for both.
    """
    def _gt(r: dict) -> dict:
        return (r.get("reward_model") or {}).get("ground_truth", {}) or {}

    def _uid(r: dict) -> str:
        return str(_gt(r).get("row_uid") or "")

    def _cell(r: dict) -> str:
        return str(_gt(r).get("locus_cell") or "")

    by_cell: dict[str, list[dict]] = {}
    for r in train_rows:
        by_cell.setdefault(_cell(r), []).append(r)
    for rows in by_cell.values():
        rows.sort(key=_uid)

    total = len(train_rows)
    size = min(int(size), total)
    cells = sorted(by_cell)
    # Largest-remainder apportionment: deterministic, and the per-cell share is
    # the population share to within one row.
    exact = {c: size * len(by_cell[c]) / total for c in cells}
    alloc = {c: int(exact[c]) for c in cells}
    for c in sorted(cells, key=lambda c: (-(exact[c] - alloc[c]), c))[: size - sum(alloc.values())]:
        alloc[c] += 1

    chosen: list[dict] = []
    for c in cells:
        # Shallow-copy: the probe re-tags ``data_source`` below, and these dicts
        # are the very objects already handed to train.parquet. The write order
        # happens to make in-place tagging harmless today, but a probe that
        # silently re-labels 512 TRAIN rows is not a failure anyone would catch
        # by reading the log. The copy is free (the image payload is shared by
        # reference), so pay it rather than depend on call order.
        chosen.extend(dict(r) for r in by_cell[c][: alloc[c]])
    chosen.sort(key=_uid)
    if not chosen:
        print("[lnsdl] locus probe: no rows selected, skipping")
        return

    n_s0 = 0
    for row in chosen:
        s_level = _locus_s_level(_cell(row))
        tag = "lnsdl_locus_s0" if s_level == 0 else "lnsdl_locus_s1p"
        if s_level is None:
            # An unparseable cell name means the corpus changed shape. Do not
            # guess a stratum -- that is how a mis-stratified probe becomes an
            # apparent capability trend.
            raise ValueError(
                f"LOCUS probe: cell {_cell(row)!r} does not match "
                f"{_LOCUS_CELL_RE.pattern} -- refusing to guess its S level."
            )
        n_s0 += int(tag == "lnsdl_locus_s0")
        row["data_source"] = tag

    path = out_dir / "probe.parquet"
    _write_lnsdl_parquet(chosen, path, row_group_size=row_group_size)
    ids_path = out_dir / "probe_row_uids.json"
    ids_path.write_text(json.dumps(
        {
            "n_total": len(chosen),
            "n_s0": n_s0,
            "n_s1p": len(chosen) - n_s0,
            "n_cells": len(cells),
            "per_cell": {c: alloc[c] for c in cells},
            "per_cell_population": {c: len(by_cell[c]) for c in cells},
            "row_uids": [_uid(r) for r in chosen],
        },
        indent=2,
    ))
    print(f"[lnsdl] saved {len(chosen)} LOCUS probe rows over {len(cells)} cells "
          f"({n_s0} s0 / {len(chosen) - n_s0} s>=1) → {path}")
    print(f"[lnsdl] probe row_uids → {ids_path}")


def _convert_lnsdl(
    split: str,
    *,
    limit: int | None = None,
    level: str = "VIKI-L2",
    source: str = "viki_l2",
) -> list[dict]:
    from nsdl.prompt_adapter import load_benchmark_samples
    from nsdl.scene_parser import build_scene_parser
    from nsdl.scene_facts_refiner import build_scene_facts_refiner
    from nsdl.aig_builder import build_affordance_context
    from nsdl.tool_runtime import build_initial_messages
    from scripts.build_slm_training_data import extract_sample_data, reverse_map_time_steps
    from scripts.cot_assistant_serializer import serialize_cot_assistant

    samples = load_benchmark_samples(level=level, split=split)
    if limit is not None:
        samples = samples[:limit]
    is_locus = source == "locus"
    data_source = "lnsdl_locus" if is_locus else "lnsdl"
    # Axis-T reachability, measured at build time with the SAME function the
    # trainer will use. A prompt whose own GT plan scores 0 is `all_wrong` on
    # axis T no matter what the policy emits: its group variance is zero
    # *because of the judge*, not because of the model, and a blocking gate that
    # reads `all_wrong` without this flag is reading the judge (memory:
    # rl-axis-t-uses-wrong-locus-ruler). Recording it per row keeps the honest
    # denominator in the parquet instead of in a one-off audit.
    r_task_fn = _load_r_task() if is_locus else None
    n_unreachable = 0
    # Same parse + NoOp-refine path the agentic eval uses (scene_facts parity).
    parser = build_scene_parser("rule")
    refiner = build_scene_facts_refiner("none")

    rows: list[dict] = []
    skipped = 0
    for s in samples:
        try:
            scene_facts = refiner.refine(s, parser.parse(s), yolo_detections=None)
            ctx = build_affordance_context(scene_facts)
            msgs = build_initial_messages(scene_facts, ctx)  # [system, user]
            prompt = _build_prompt(msgs[0]["content"], msgs[1]["content"], "vl")

            # GT (deterministic from VIKI-L2 time_steps).
            sample_data = extract_sample_data(s.raw_sample)
            ops = reverse_map_time_steps(sample_data)
            gt_dag = [
                {
                    "op_id": op.op_id, "op_type": op.op_type, "entity": op.entity,
                    "dest": op.dest, "deps": list(op.deps), "robot": op.robot,
                    **({"params": op.params} if op.params else {}),
                }
                for op in ops
            ]
            gt_goal, _, _ = _parse_cot_assistant(serialize_cot_assistant(scene_facts, ops))
            sgt = s.ground_truth or {}
            gt = {
                "task_id": s.task_id,
                "task_name": s.task_name,
                "instruction": s.instruction,
                "robots": _jsonable(sgt.get("robots") or {}),
                "time_steps": _jsonable(sgt.get("time_steps") or []),
                "gt_dag_json": json.dumps(gt_dag),
                "gt_goal_json": json.dumps(gt_goal) if gt_goal is not None else "",
                # Fields required by enactgraph.eval_single (R_task path). Without
                # these the reward's _simulate raises KeyError('init_pos') and is
                # silently swallowed -> R_task==0 for EVERY rollout (docs/RL_debug.md
                # §1 parity probe: train r_task 0.0000 vs eval plan_acc .9679).
                "init_pos": _jsonable(sgt.get("init_pos")),
                "goal_constraints": _jsonable(sgt.get("goal_constraints")),
                "temporal_constraints": _jsonable(sgt.get("temporal_constraints")),
                "idle_robots": _jsonable(sgt.get("idle_robots")),
                # Task-blind structural flags consumed by the reward (T1) and
                # the optional skip-decision sampler (T2), docs/RL_debug.md §8.
                "trivial_structure": bool(_trivial_structure(gt_dag)),
                "has_skip_decision": bool(_has_skip_decision(gt_goal)),
            }
            if is_locus:
                # LOCUS `task_id` is the grid CELL (`F1_s0m0`), shared by up to
                # 210 rows -- it is NOT a row identity. Every audit that keyed on
                # it has been wrong (memory: locus-sft-build-two-silent-defects
                # put 74% of rows on the wrong image that way). Carry an explicit
                # per-row uid and the cell as separate, honestly-named fields.
                gt["row_uid"] = f"{split}#{s.sample_index:05d}"
                gt["locus_cell"] = str(s.task_id or "")
                gt["locus_split"] = split
                if r_task_fn is not None:
                    reachable = bool(r_task_fn(gt_dag, gt) >= 1.0)
                    gt["gt_r_task_reachable"] = reachable
                    n_unreachable += int(not reachable)

            # Minimal sample payload for rollout-side context reconstruction.
            lnsdl_sample = {
                "instruction": s.instruction,
                "robot_inventory": _jsonable(s.robot_inventory or {}),
                "image_b64": (
                    base64.b64encode(s.image_bytes).decode("ascii")
                    if s.image_bytes else None
                ),
                "ground_truth": {"init_pos": _jsonable((s.ground_truth or {}).get("init_pos"))},
                "task_id": s.task_id,
                "task_name": s.task_name,
                "level": s.level,
                "split": s.split,
                "sample_index": s.sample_index,
            }
            if is_locus:
                lnsdl_sample["row_uid"] = gt["row_uid"]

            row = {
                "prompt": prompt,
                "reward_model": {"ground_truth": gt},
                "data_source": data_source,
                "lnsdl_sample": lnsdl_sample,
            }
            if isinstance(s.raw_sample, dict) and s.raw_sample.get("images") is not None:
                row["images"] = _jsonable(s.raw_sample["images"])
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if skipped <= 5:
                print(f"  [lnsdl:{split}] skip {getattr(s,'task_id','?')}: {exc}")
            continue
    print(f"  [lnsdl:{level}/{split}] converted {len(rows)}, skipped {skipped}")
    if skipped:
        # A silent drop changes the training pool. On VIKI the count has always
        # been 0; treat any non-zero as a finding to explain, not a rounding
        # error, and make it impossible to miss in a 6300-row build log.
        print(f"  [lnsdl:{level}/{split}] ** {skipped} rows dropped -- explain "
              f"before using this parquet **")
    if r_task_fn is not None and rows:
        frac = n_unreachable / len(rows)
        print(f"  [lnsdl:{level}/{split}] axis-T GT-unreachable: {n_unreachable}/"
              f"{len(rows)} ({frac:.4f}) -- these groups are all_wrong(T) by "
              f"construction; see gt.gt_r_task_reachable")
    return rows


def _parse_cot_assistant(assistant: str) -> tuple[dict | None, dict | None, list | None]:
    """Extract [GOAL] / [ASSIGN] / [PLAN] from a three-block assistant string."""
    import re

    blocks: dict[str, str] = {}
    for m in re.finditer(r"\[(GOAL|ASSIGN|PLAN)\]\s*\n", assistant):
        name = m.group(1)
        start = m.end()
        nxt = re.search(r"\n\[(GOAL|ASSIGN|PLAN)\]", assistant[start:])
        end = start + nxt.start() if nxt else len(assistant)
        blocks[name] = assistant[start:end].strip()

    goal = assign = plan = None
    if "GOAL" in blocks:
        try:
            v = json.loads(blocks["GOAL"])
            if isinstance(v, dict):
                goal = v
        except Exception:
            pass
    if "ASSIGN" in blocks:
        try:
            v = json.loads(blocks["ASSIGN"])
            if isinstance(v, dict):
                assign = v
        except Exception:
            pass
    if "PLAN" in blocks:
        try:
            v = json.loads(blocks["PLAN"])
            if isinstance(v, list):
                plan = v
        except Exception:
            pass
    return goal, assign, plan


def _gt_from_sample(sample: dict, gt_index: dict) -> dict:
    """Build ground_truth dict for a current samples.jsonl record.

    Prefers VIKI-L2 GT (for S1 whose task_id exists in the index) so that
    ``time_steps`` and other VIKI fields are preserved.  Falls back to
    constructing a minimal GT from the sample's own metadata for S2/S3.
    """
    meta = sample["metadata"]
    tid = meta["task_id"]
    gt = gt_index.get(tid)

    # Build gt_dag_json from target_dag (always use sample's own DAG — this
    # is the GT the reward function should compare against).
    td = sample.get("target_dag", [])
    gt_dag_json = json.dumps(td) if not isinstance(td, str) else td

    # Also parse [GOAL] / [ASSIGN] from the sample's assistant for
    # R_goal / R_assign reward shaping.
    gt_goal, gt_assign, _ = _parse_cot_assistant(sample.get("assistant", ""))
    gt_goal_json = json.dumps(gt_goal) if gt_goal is not None else ""
    gt_assign_json = json.dumps(gt_assign) if gt_assign is not None else ""

    robots_dict = _robots_from_metadata(meta)

    if gt is not None:
        # S1 path: inherit all VIKI-L2 fields, override DAG with serialised form
        gt_extended = dict(gt)
        gt_extended["gt_dag_json"] = gt_dag_json
        gt_extended["gt_goal_json"] = gt_goal_json
        gt_extended["gt_assign_json"] = gt_assign_json
        gt_extended["user_prompt"] = sample["user_prompt"]
        if robots_dict:
            # Use the exact slot universe shown in the prompt.
            gt_extended["robots"] = robots_dict
        return gt_extended

    # S2 / S3 path: build minimal GT from metadata
    return {
        "task_id": tid,
        "instruction": meta.get("instruction") or "",
        "robots": robots_dict,
        "gt_dag_json": gt_dag_json,
        "gt_goal_json": gt_goal_json,
        "gt_assign_json": gt_assign_json,
        "user_prompt": sample["user_prompt"],
    }


def _robots_from_metadata(meta: dict) -> dict[str, str]:
    robots = meta.get("robots") or {}
    if isinstance(robots, dict):
        return {str(slot): str(rtype) for slot, rtype in robots.items() if rtype}
    out: dict[str, str] = {}
    if isinstance(robots, list):
        for robot in robots:
            if not isinstance(robot, dict):
                continue
            slot = robot.get("slot") or robot.get("slot_id")
            rtype = robot.get("type") or robot.get("robot_type")
            if slot and rtype:
                out[str(slot)] = str(rtype)
    return out


def _build_prompt(system_prompt: str, user_prompt: str, prompt_modality: str) -> list[dict[str, str]]:
    if prompt_modality == "vl" and not user_prompt.lstrip().startswith("<image>"):
        user_prompt = "<image>\n" + user_prompt
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _with_optional_images(
    row: dict[str, Any],
    *,
    task_id: str,
    image_index: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any] | None:
    if image_index is None:
        return row
    images = image_index.get(task_id)
    if not images:
        return None
    out = dict(row)
    out["images"] = images
    return out


def _convert_from_viki_parquet(
    parquet_path: Path,
    split_name: str,
    *,
    prompt_modality: str = "text",
    include_images: bool = False,
    prompt_variant: str = "full",
) -> list[dict]:
    """Build current RL rows directly from a VIKI parquet split.

    This is used for the GRPO validation file. It never reads the OOD
    ``val.parquet`` by default; the caller passes VIKI-L2 ``test.parquet``.
    YOLO is not run here, so the prompt keeps the schema-stable
    ``Entity locations: none observed`` fallback when no relation is present.
    """
    from nsdl.aig_builder import build_affordance_context
    from nsdl.slm_interface import build_slm_system_prompt, serialize_slm_input
    from scripts.build_slm_training_data import (
        build_scene_facts_from_gt,
        extract_sample_data,
        reverse_map_time_steps,
    )
    from scripts.cot_assistant_serializer import serialize_cot_assistant

    df = pd.read_parquet(parquet_path)
    rows: list[dict] = []
    skipped = 0
    sys_prompt = build_slm_system_prompt()
    for idx in range(len(df)):
        row = df.iloc[idx]
        try:
            sample_data = extract_sample_data(row)
            scene_facts = build_scene_facts_from_gt(sample_data, idx=idx)
            if scene_facts.extras is None:
                scene_facts.extras = {}
            relations = scene_facts.extras.get("scene_relations")
            if not (isinstance(relations, (list, tuple)) and any(str(r).strip() for r in relations)):
                scene_facts.extras["scene_relations"] = ["none observed"]
            ctx = build_affordance_context(scene_facts)
            ops = reverse_map_time_steps(sample_data)
            target_dag = [
                {
                    "op_id": op.op_id,
                    "op_type": op.op_type,
                    "entity": op.entity,
                    "dest": op.dest,
                    "deps": list(op.deps),
                    "robot": op.robot,
                    **({"params": op.params} if op.params else {}),
                }
                for op in ops
            ]
            sample = {
                "system_prompt": sys_prompt,
                "user_prompt": apply_vl_prompt_variant(
                    serialize_slm_input(scene_facts, ctx),
                    prompt_variant,
                ),
                "assistant": serialize_cot_assistant(scene_facts, ops),
                "target_dag": target_dag,
                "metadata": {
                    "task_id": sample_data.get("task_id", f"{split_name}_{idx}"),
                    "task_name": sample_data.get("task_name", ""),
                    "instruction": sample_data.get("instruction", ""),
                    "robots": sample_data.get("robot_inventory") or [],
                },
            }
            gt = _jsonable(row["reward_model"]["ground_truth"])
            gt_extended = _gt_from_sample(sample, {sample["metadata"]["task_id"]: gt})
            out_row = {
                "prompt": _build_prompt(
                    apply_vl_system_prompt_variant(sample["system_prompt"], prompt_variant),
                    sample["user_prompt"],
                    prompt_modality,
                ),
                "reward_model": {"ground_truth": gt_extended},
                "data_source": "enactgraph",
            }
            if include_images:
                out_row["images"] = _jsonable(row["images"])
            rows.append(out_row)
        except Exception:
            skipped += 1
            continue
    print(f"  [{split_name}] converted {len(rows)} from {parquet_path}, skipped {skipped}")
    return rows


def _convert_split(
    sft_ds,
    gt_index: dict,
    split_name: str,
    *,
    prompt_modality: str = "text",
    prompt_variant: str = "full",
    image_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict]:
    """Convert an HF SFT dataset to verl-format rows."""
    rows: list[dict] = []
    skipped = 0
    for sample in sft_ds:
        tid = sample["task_id"]
        gt = gt_index.get(tid)
        if gt is None:
            skipped += 1
            continue

        messages = sample["messages"]
        # messages: [system, user, assistant] — keep only system + user
        user_prompt = apply_vl_prompt_variant(messages[1]["content"], prompt_variant)
        system_prompt = apply_vl_system_prompt_variant(messages[0]["content"], prompt_variant)
        prompt = _build_prompt(system_prompt, user_prompt, prompt_modality)

        # Store user_prompt + reverse-mapped GT DAG (legacy mode).
        gt_extended = dict(gt)
        gt_extended["user_prompt"] = user_prompt
        gt_goal, gt_assign, gt_plan = _parse_cot_assistant(messages[2]["content"] if len(messages) >= 3 else "")
        gt_extended["gt_dag_json"] = json.dumps(gt_plan or [])
        gt_extended["gt_goal_json"] = json.dumps(gt_goal) if gt_goal is not None else ""
        gt_extended["gt_assign_json"] = json.dumps(gt_assign) if gt_assign is not None else ""

        out_row = {
            "prompt": prompt,
            "reward_model": {"ground_truth": gt_extended},
            "data_source": "enactgraph",
        }
        out_row = _with_optional_images(out_row, task_id=tid, image_index=image_index)
        if out_row is None:
            skipped += 1
            continue
        rows.append(out_row)

    print(f"  [{split_name}] converted {len(rows)}, skipped {skipped} (no GT match)")
    return rows


def _convert_from_jsonl(
    jsonl_path: Path,
    gt_index: dict,
    split_name: str,
    *,
    prompt_modality: str = "text",
    prompt_variant: str = "full",
    image_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict]:
    """Convert samples.jsonl to verl-format rows (covers S1/S2/S3)."""
    rows: list[dict] = []
    seen_tids: set[str] = set()

    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            tid = sample["metadata"]["task_id"]
            # Deduplicate by task_id
            if tid in seen_tids:
                continue
            seen_tids.add(tid)

            user_prompt = apply_vl_prompt_variant(sample["user_prompt"], prompt_variant)
            system_prompt = apply_vl_system_prompt_variant(sample["system_prompt"], prompt_variant)
            prompt = _build_prompt(system_prompt, user_prompt, prompt_modality)

            gt_extended = _gt_from_sample(sample, gt_index)
            gt_extended["user_prompt"] = user_prompt
            out_row = {
                "prompt": prompt,
                "reward_model": {"ground_truth": gt_extended},
                "data_source": "enactgraph",
            }
            out_row = _with_optional_images(out_row, task_id=tid, image_index=image_index)
            if out_row is None:
                continue
            rows.append(out_row)

    print(f"  [{split_name}] converted {len(rows)} (S1 use VIKI GT; S2/S3 use in-sample GT)")
    return rows


if __name__ == "__main__":
    main()
