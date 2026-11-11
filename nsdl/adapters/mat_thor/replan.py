from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExecFailure:
    failed_step: str | None
    action: str | None
    reason: str
    observed_delta: Mapping[str, Any] = field(default_factory=dict)


def load_exec_failure(path: Path | str) -> ExecFailure | None:
    payload = json.loads(Path(path).read_text())
    if payload.get("success") is True:
        return None
    return ExecFailure(
        failed_step=_nullable_str(payload.get("failed_step")),
        action=_nullable_str(payload.get("action")),
        reason=str(payload.get("reason") or payload.get("error") or "unknown execution failure"),
        observed_delta=payload.get("observed_delta") if isinstance(payload.get("observed_delta"), dict) else {},
    )


def format_exec_failure_block(failure: ExecFailure) -> str:
    lines = ["[EXEC_FAILURE]"]
    if failure.failed_step:
        lines.append(f"failed_step: {failure.failed_step}")
    if failure.action:
        lines.append(f"action: {failure.action}")
    lines.append(f"reason: {failure.reason}")
    if failure.observed_delta:
        lines.append(
            "observed_delta: "
            + json.dumps(dict(failure.observed_delta), ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(lines)


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ExecFailure",
    "format_exec_failure_block",
    "load_exec_failure",
]

