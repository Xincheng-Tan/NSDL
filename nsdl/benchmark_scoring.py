from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score import viki_1, viki_2  # noqa: E402


def _wrap_answer(answer: str) -> str:
    return f"<think>nsdl</think><answer>{answer}</answer>"


def compute_viki_l1_score(selected_robot_types: list[str], ground_truth: list[str] | str) -> float:
    answer = repr(selected_robot_types)
    ground_truth_value = repr(ground_truth) if isinstance(ground_truth, list) else ground_truth
    return float(viki_1.compute_score(_wrap_answer(answer), ground_truth_value))


def compute_viki_l2_score(plan: list[dict], ground_truth: dict) -> float:
    answer = json.dumps(plan, ensure_ascii=False)
    return float(viki_2.compute_score(_wrap_answer(answer), ground_truth))


def locus_m_depth_key(ground_truth: object) -> str:
    """Group key for a LOCUS row: its M (mechanical depth) level, e.g. ``M3``.

    P10.S2's y-axis is the M depth at which the Phi-offload gain turns
    positive, so every arm -- ours and the constant-offload control -- must bin
    LOCUS rows by M rather than by the VIKI family suffix. Sharing one
    implementation is what keeps the arms on a single code path; two copies
    that drifted would move the curve without changing any measurement.

    ``locus_meta.cell`` is ``(S, M)`` and agrees with the ``sNmM`` task-name
    encoding on all 560 frozen eval rows (pinned in
    ``tests/test_locus_adapter_path.py``).
    """
    if not isinstance(ground_truth, dict):
        return "<unknown>"
    cell = (ground_truth.get("locus_meta") or {}).get("cell")
    if isinstance(cell, (list, tuple)) and len(cell) == 2 and cell[1] is not None:
        return f"M{cell[1]}"
    return "<unknown>"


def compute_locus_l2_acc(plan: list[dict], ground_truth: dict) -> float:
    """LOCUS-Bench binary task success (accuracy only, no format term).

    Identical to the official ``viki_2.acc_reward`` except on **null-plan**
    items -- LOCUS cells whose goal already holds at the initial state, so the
    correct answer is *no actions at all* (73/560 of the frozen test splits,
    concentrated in the S>=1 rows; this is the state-judgment payload).

    ``viki_2.acc_reward`` early-returns 0 whenever the transformed plan is
    empty, so on those items no model can score above 0 -- it reports a correct
    "nothing to do" and a wrong plan identically. Here they are scored
    correct-iff-empty, which is the same length rule the official judge applies
    everywhere else (it requires ``len(gt) / len(pred) >= 0.99``, i.e. no
    superfluous steps). On the other 487 items this is byte-identical to the
    official judge.

    Baselines (``scripts/score_locus_strong_baseline.py``) and our own models
    must both go through this function, or the LOCUS table mixes two rulers.
    """
    steps = ground_truth.get("time_steps") if isinstance(ground_truth, dict) else None
    if not steps:
        return 0.0 if viki_2.transform_actions(plan) else 1.0
    answer = json.dumps(plan, ensure_ascii=False)
    return float(viki_2.acc_reward(_wrap_answer(answer), ground_truth))

