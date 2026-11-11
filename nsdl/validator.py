from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score.utils.eval.eval_viki_2 import eval_single  # noqa: E402


@dataclass
class ValidationResult:
    success: bool
    error_desc: str
    eval_commands: list[dict[str, str]]


def plan_to_eval_commands(plan: list[dict]) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    for step in plan:
        step_commands: dict[str, str] = {}
        for robot_slot, action in step.get("actions", {}).items():
            step_commands[robot_slot] = f"<{','.join(action)}>"
        commands.append(step_commands)
    return commands


def validate_l2_plan(plan: list[dict], ground_truth: dict) -> ValidationResult:
    eval_commands = plan_to_eval_commands(plan)
    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        success = eval_single(eval_commands, ground_truth)
    error_desc = stdout_buffer.getvalue().strip()
    return ValidationResult(
        success=success,
        error_desc=error_desc,
        eval_commands=eval_commands,
    )

