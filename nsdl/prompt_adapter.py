from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nsdl.catalog import normalize_robot_name
from nsdl.schema import BenchmarkSample


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "viki"

# LOCUS-Bench sits outside data/viki and uses a directory-per-split layout, but
# its rows carry the *identical* column schema to VIKI
# (['ability', 'data_source', 'images', 'prompt', 'reward_model']), so only the
# on-disk path differs -- _build_benchmark_sample is shared verbatim.
LOCUS_LEVEL = "LOCUS"
LOCUS_ROOT = REPO_ROOT / "data" / "lnsdl_bench" / "ramp63k"

AVAILABLE_ROBOT_SET_PATTERN = re.compile(
    r"Available robot set:\s*(\{.*?\})\s*Robot characteristics:",
    re.DOTALL,
)
POSSIBLE_ROBOT_LINE_PATTERN = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s+.+$", re.MULTILINE)


def convert_arrays_to_native(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [convert_arrays_to_native(item) for item in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {key: convert_arrays_to_native(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_arrays_to_native(item) for item in obj]
    if hasattr(obj, "tolist"):
        return convert_arrays_to_native(obj.tolist())
    return obj


def build_sample_stem_from_parts(
    *,
    sample_index: int,
    task_id: str | None,
) -> str:
    prefix = f"{sample_index:05d}"
    if task_id:
        return f"{prefix}_{task_id}"
    return prefix


def build_sample_image_path(
    *,
    level: str,
    split: str,
    sample_index: int,
    task_id: str | None,
) -> str:
    if level == LOCUS_LEVEL:
        # LOCUS renders are named by row index alone -- the task_id suffix that
        # VIKI stems carry is not part of the on-disk name.
        return str((LOCUS_ROOT / split / "images" / f"{sample_index:05d}.png").resolve())
    stem = (
        build_sample_stem_from_parts(
            sample_index=sample_index,
            task_id=task_id,
        )
        if task_id
        else f"{sample_index:05d}_sample_{sample_index:05d}"
    )
    return str(
        (DATA_ROOT / level / split / f"{stem}.png").resolve()
    )


def _split_path(level: str, split: str) -> Path:
    if level == LOCUS_LEVEL:
        return LOCUS_ROOT / split / "data.parquet"
    return DATA_ROOT / level / f"{split}.parquet"


def _read_split(level: str, split: str) -> list[dict[str, Any]]:
    split_path = _split_path(level, split)
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")
    try:
        return pd.read_parquet(split_path).to_dict("records")
    except pa.ArrowNotImplementedError:
        # A split whose embedded images push a nested column past Arrow's 2 GB
        # per-chunk ceiling cannot be read whole: Arrow splits the column into
        # chunks and then refuses the nested->chunked conversion. LOCUS train
        # (6300 rows, 15.3 GB, written as a *single* row group) hits this, so
        # neither read_table nor read_row_group works. Batched reads keep each
        # chunk under the ceiling and are exact -- same rows, same order.
        return [
            row
            for batch in pq.ParquetFile(split_path).iter_batches(batch_size=64)
            for row in batch.to_pylist()
        ]


def extract_available_robot_set(system_prompt: str) -> dict[str, str | None]:
    match = AVAILABLE_ROBOT_SET_PATTERN.search(system_prompt)
    if match:
        robot_set = ast.literal_eval(match.group(1))
        return {str(slot): (robot if robot is not None else None) for slot, robot in robot_set.items()}

    possible_robot_inventory: dict[str, str | None] = {}
    for index, robot_name in enumerate(POSSIBLE_ROBOT_LINE_PATTERN.findall(system_prompt), start=1):
        robot_type = normalize_robot_name(robot_name)
        if not robot_type:
            continue
        possible_robot_inventory[f"R{index}"] = robot_type
    if possible_robot_inventory:
        return possible_robot_inventory

    raise ValueError("Could not parse robot inventory from system prompt.")


def infer_robot_inventory_scope(system_prompt: str) -> str:
    if AVAILABLE_ROBOT_SET_PATTERN.search(system_prompt):
        return "available"
    if POSSIBLE_ROBOT_LINE_PATTERN.search(system_prompt):
        return "possible"
    return "available"


def _build_benchmark_sample(
    *,
    level: str,
    split: str,
    sample_index: int,
    row: dict[str, Any],
) -> BenchmarkSample:
    prompt_messages = row["prompt"]
    system_prompt = prompt_messages[0]["content"]
    instruction = prompt_messages[1]["content"].replace("<image>", "").strip()
    ground_truth = convert_arrays_to_native(row["reward_model"]["ground_truth"])
    image_bytes = row["images"][0]["bytes"] if row.get("images") else None
    robot_inventory = extract_available_robot_set(system_prompt)
    robot_inventory_scope = infer_robot_inventory_scope(system_prompt)
    task_id = ground_truth.get("task_id") if isinstance(ground_truth, dict) else None
    task_name = ground_truth.get("task_name") if isinstance(ground_truth, dict) else None
    image_path = (
        build_sample_image_path(
            level=level,
            split=split,
            sample_index=sample_index,
            task_id=task_id,
        )
        if image_bytes
        else None
    )

    return BenchmarkSample(
        level=level,
        split=split,
        sample_index=sample_index,
        task_id=task_id,
        task_name=task_name,
        instruction=instruction,
        system_prompt=system_prompt,
        image_bytes=image_bytes,
        robot_inventory=robot_inventory,
        robot_inventory_scope=robot_inventory_scope,
        image_path=image_path,
        ground_truth=ground_truth if isinstance(ground_truth, dict) else None,
        raw_sample=row,
    )


def load_benchmark_samples(
    *,
    level: str = "VIKI-L2",
    split: str = "test",
) -> list[BenchmarkSample]:
    rows = _read_split(level, split)
    return [
        _build_benchmark_sample(
            level=level,
            split=split,
            sample_index=index,
            row=row,
        )
        for index, row in enumerate(rows)
    ]


def build_sample_stem(sample: BenchmarkSample) -> str:
    return build_sample_stem_from_parts(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
    )


def load_benchmark_sample(
    *,
    level: str = "VIKI-L2",
    split: str = "test",
    sample_index: int | None = None,
    task_id: str | None = None,
) -> BenchmarkSample:
    rows = _read_split(level, split)

    selected_index = None
    selected_row = None
    for index, row in enumerate(rows):
        ground_truth = convert_arrays_to_native(row["reward_model"]["ground_truth"])
        row_task_id = ground_truth.get("task_id") if isinstance(ground_truth, dict) else None

        if task_id is not None and row_task_id == task_id:
            selected_index = index
            selected_row = row
            break
        if task_id is None and sample_index is not None and index == sample_index:
            selected_index = index
            selected_row = row
            break

    if selected_row is None:
        target = f"task_id={task_id}" if task_id is not None else f"sample_index={sample_index}"
        raise ValueError(f"Could not find sample for {target} in {level}/{split}.")

    return _build_benchmark_sample(
        level=level,
        split=split,
        sample_index=selected_index,
        row=selected_row,
    )
