from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "lamma-p" / "data" / "final_test"
EXPECTED_MAT_THOR_TASK_COUNT = 70

COMPOUND = "Compound"
COMPLEX = "Complex"
VAGUE = "Vague"

_FILE_PATTERN = re.compile(r"^FloorPlan(?P<floor_plan>\d+)(?P<suffix>pddlvague|pddl|vague)?\.json$")
_SUFFIX_COMPLEXITY = {
    "": COMPOUND,
    "pddl": COMPLEX,
    "vague": VAGUE,
    "pddlvague": VAGUE,
}
_SUFFIX_ORDER = {
    "": 0,
    "pddl": 1,
    "vague": 2,
    "pddlvague": 3,
}


@dataclass(frozen=True)
class MatThorTask:
    task_id: str
    floor_plan: int
    complexity: str
    source_file: Path
    line_number: int
    file_record_index: int
    floor_plan_record_index: int
    record: dict[str, Any]
    parse_status: str = "json"
    parse_notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def instruction(self) -> str:
        return str(self.record.get("task", "")).strip()

    @property
    def robot_ids(self) -> list[int]:
        raw_ids = self.record.get("robot list") or self.record.get("robot_list") or []
        if not isinstance(raw_ids, list):
            return []
        ids: list[int] = []
        for value in raw_ids:
            try:
                ids.append(int(value))
            except (TypeError, ValueError):
                continue
        return ids


def iter_task_files(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> Iterator[Path]:
    root = Path(dataset_dir)
    files = [path for path in root.glob("*.json") if _parse_filename(path.name) is not None]
    files.sort(key=_task_file_sort_key)
    yield from files


def iter_tasks(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    *,
    include_recovered: bool = True,
    complexities: Iterable[str] | None = None,
) -> Iterator[MatThorTask]:
    allowed_complexities = {str(item) for item in complexities} if complexities else None
    floor_offsets: dict[int, int] = {}

    for path in iter_task_files(dataset_dir):
        file_info = _parse_filename(path.name)
        if file_info is None:
            continue
        floor_plan, suffix = file_info
        complexity = _SUFFIX_COMPLEXITY[suffix]
        if allowed_complexities and complexity not in allowed_complexities:
            continue

        file_record_index = 0
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record, parse_status, parse_notes = _parse_record_line(line)
            except ValueError as exc:
                if include_recovered:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                continue
            if parse_status == "recovered_public" and not include_recovered:
                continue

            floor_index = floor_offsets.get(floor_plan, 0)
            floor_offsets[floor_plan] = floor_index + 1
            yield MatThorTask(
                task_id=f"FloorPlan{floor_plan}#{floor_index}",
                floor_plan=floor_plan,
                complexity=complexity,
                source_file=path,
                line_number=line_number,
                file_record_index=file_record_index,
                floor_plan_record_index=floor_index,
                record=record,
                parse_status=parse_status,
                parse_notes=tuple(parse_notes),
            )
            file_record_index += 1


def summarize_tasks(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    *,
    include_recovered: bool = True,
) -> dict[str, Any]:
    tasks = list(iter_tasks(dataset_dir, include_recovered=include_recovered))
    return {
        "total": len(tasks),
        "expected": EXPECTED_MAT_THOR_TASK_COUNT,
        "delta_from_expected": len(tasks) - EXPECTED_MAT_THOR_TASK_COUNT,
        "by_complexity": dict(Counter(task.complexity for task in tasks)),
        "by_floor_plan": dict(Counter(task.floor_plan for task in tasks)),
        "parse_status": dict(Counter(task.parse_status for task in tasks)),
    }


def _parse_filename(name: str) -> tuple[int, str] | None:
    match = _FILE_PATTERN.match(name)
    if match is None:
        return None
    suffix = match.group("suffix") or ""
    return int(match.group("floor_plan")), suffix


def _task_file_sort_key(path: Path) -> tuple[int, int, str]:
    parsed = _parse_filename(path.name)
    if parsed is None:
        return (999999, 999999, path.name)
    floor_plan, suffix = parsed
    return (floor_plan, _SUFFIX_ORDER[suffix], path.name)


def _parse_record_line(line: str) -> tuple[dict[str, Any], str, list[str]]:
    stripped = line.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload, "json", []
    except json.JSONDecodeError:
        pass

    try:
        payload = ast.literal_eval(stripped)
        if isinstance(payload, dict):
            return payload, "python_literal", ["parsed with ast.literal_eval"]
    except (SyntaxError, ValueError):
        pass

    recovered = _recover_public_record(stripped)
    if recovered is None:
        raise ValueError("could not parse MAT-THOR jsonl record")
    return recovered, "recovered_public", [
        "malformed jsonl; recovered task and robot list only",
    ]


def _recover_public_record(line: str) -> dict[str, Any] | None:
    task_match = re.search(r'"task"\s*:\s*"(?P<task>(?:\\"|[^"])*)"', line)
    robot_match = re.search(r'"robot list"\s*:\s*(?P<robots>\[[^\]]*\])', line)
    if task_match is None or robot_match is None:
        return None
    try:
        robot_ids = ast.literal_eval(robot_match.group("robots"))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(robot_ids, list):
        return None
    return {
        "task": task_match.group("task").replace('\\"', '"'),
        "robot list": robot_ids,
        "_parse_error": "malformed_jsonl_public_fields_only",
    }


__all__ = [
    "COMPLEX",
    "COMPOUND",
    "DEFAULT_DATASET_DIR",
    "EXPECTED_MAT_THOR_TASK_COUNT",
    "MatThorTask",
    "VAGUE",
    "iter_task_files",
    "iter_tasks",
    "summarize_tasks",
]

