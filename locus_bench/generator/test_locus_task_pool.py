"""P7.1 acceptance test (docs/benchmark.md §9.1):

For every family × cell × realization:
  1. ``make_template`` returns an official-schema-compatible dict
  2. official ``instantiate_task`` instantiates it without exception
  3. the instantiated GT plan replays successfully in the official SimEnv
     (same metadata construction as ``eval_gt.py``)
  4. empty-plan realizations pass the goal check with zero commands
Plus axis-hygiene checks (S knobs don't move robots; M knobs don't move
state semantics) and realization-enumeration sanity.

Run:  cd RoboFactory && python -m pytest script_general/test_locus_task_pool.py -q
(no external deps beyond pyyaml already required by the repo)
"""

import io
import os
import random
import sys
from contextlib import redirect_stdout

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROBOFACTORY)
sys.path.insert(0, os.path.join(_ROBOFACTORY, "script_general"))

from locus_task_pool import (  # noqa: E402
    FAMILY_CELLS, iter_cells, make_template, enumerate_realizations,
)
from generate_prompt import instantiate_task  # noqa: E402
from utils.eval.eval import Eval  # noqa: E402

CONTAINER_ASSETS = ['plate', 'cabinet', 'drawer', 'bowl', 'sink', 'toaster',
                    'tray', 'cardboardbox']

N_SAMPLES_PER_REALIZATION = 3


def _format_answer(answer):
    commands = []
    for inst in answer:
        commands.append({r: f'<{",".join(a)}>' for r, a in inst.items()})
    return commands


def _replay(inst, rng):
    """Mirror eval_gt.py metadata construction, deterministic pos choice."""
    judger = Eval()
    metadata = {"agents": {}, "assets": {}}
    for robot_id, robot_type in inst["robots"].items():
        metadata["agents"][robot_id] = {"type": robot_type,
                                        "pos": {"name": robot_id}}
    for asset_name, asset_pos in inst["init_pos"].items():
        if asset_name.startswith('R') and asset_name[1:].isdigit():
            continue
        asset_type = asset_name.rsplit('_', 1)[0]
        metadata["assets"][asset_type] = {
            "pos": {"name": rng.choice(asset_pos)}}
        if asset_type in CONTAINER_ASSETS:
            metadata["assets"][asset_type]['params'] = {
                "is_container": True,
                "position_kwargs": {
                    "name": asset_type,
                    "isolated": asset_type in ['cabinet'],
                },
            }
    metadata['goal_constraints'] = inst['goal_constraints']
    metadata['temporal_constraints'] = inst['temporal_constraints']
    judger.set_env(metadata)
    ok = judger.eval(_format_answer(inst["ground_truth"]))
    return ok, judger.get_error_desc()


def test_every_cell_every_realization_replays():
    rng = random.Random(7)
    n_ok = n_cells = 0
    failures = []
    for fam, s, m in iter_cells():
        n_cells += 1
        for realization in enumerate_realizations(fam, s, m):
            for k in range(N_SAMPLES_PER_REALIZATION):
                tpl = make_template(fam, s, m, rng,
                                    realization=realization or None)
                with redirect_stdout(io.StringIO()):
                    inst = instantiate_task(tpl)
                ok, err = _replay(inst, rng)
                if ok:
                    n_ok += 1
                else:
                    failures.append(
                        (fam, s, m, dict(realization), err,
                         inst["description"]))
    assert not failures, (
        f"{len(failures)} replay failures (of cells={n_cells}); first 5: "
        f"{failures[:5]}")
    assert n_ok > 0


def test_empty_plan_realizations_exist_and_pass():
    """All-present realizations must yield an empty GT that still satisfies
    goals (the axis-S 'empty plan' path, benchmark.md §6.2)."""
    rng = random.Random(11)
    seen_empty = 0
    for fam, s, m in iter_cells():
        for realization in enumerate_realizations(fam, s, m):
            if realization and all(v == "present" for v in realization.values()):
                tpl = make_template(fam, s, m, rng, realization=realization)
                if tpl["ground_truth"]:
                    continue
                with redirect_stdout(io.StringIO()):
                    inst = instantiate_task(tpl)
                ok, err = _replay(inst, rng)
                assert ok, (fam, s, m, realization, err)
                seen_empty += 1
    assert seen_empty > 0, "no empty-plan realization found"


def test_axis_orthogonality_knobs():
    """S only moves state semantics; M only moves robots/structure
    (benchmark.md §2.3)."""
    rng = random.Random(3)
    for fam in FAMILY_CELLS:
        cells = list(FAMILY_CELLS[fam]["cells"])
        s_values = sorted({s for s, _ in cells})
        m_values = sorted({m for _, m in cells})
        fixed_m = m_values[0]
        fixed_s = s_values[-1]
        # vary s at fixed m: robot_roles must not change
        roles = {make_template(fam, s, fixed_m, rng)["task_name"]:
                 tuple(make_template(fam, s, fixed_m, rng)["robot_roles"])
                 for s in s_values if (s, fixed_m) in cells}
        assert len(set(roles.values())) == 1, (fam, roles)
        # vary m at fixed s: candidate mask set must not change
        cands = set()
        for m in m_values:
            if (fixed_s, m) not in cells:
                continue
            tpl = make_template(fam, fixed_s, m, rng)
            cands.add(tuple(sorted(
                c["mask_key"]
                for c in tpl["state_realization_spec"]["candidates"])))
        assert len(cands) == 1, (fam, cands)


def test_no_realization_leak_into_prompt():
    """task_name / description carry no realization info
    (benchmark.md §4.3)."""
    rng = random.Random(5)
    for fam, s, m in iter_cells():
        reals = enumerate_realizations(fam, s, m)
        if len(reals) < 2:
            continue
        names, descs = set(), set()
        for realization in reals:
            tpl = make_template(fam, s, m, rng, realization=realization)
            names.add(tpl["task_name"])
            descs.add(tuple(tpl["description"]))
        assert len(names) == 1, (fam, s, m, names)
        assert len(descs) == 1, (fam, s, m)


def test_opaque_flag_only_s4():
    rng = random.Random(1)
    for fam, s, m in iter_cells():
        tpl = make_template(fam, s, m, rng)
        assert tpl["opaque_container"] == (s == 4), (fam, s, m)


def test_explicit_family_cell_registry():
    cells = list(iter_cells())
    assert len(cells) == 49
    assert ("F4", 2, 1) in cells and ("F4", 4, 1) in cells
    assert not any(fam == "F4" and s == 3 for fam, s, _ in cells)
    try:
        make_template("F4", 3, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("explicit registry admitted forbidden F4-S3")


if __name__ == "__main__":
    for fn in [test_every_cell_every_realization_replays,
               test_empty_plan_realizations_exist_and_pass,
               test_axis_orthogonality_knobs,
               test_no_realization_leak_into_prompt,
               test_opaque_flag_only_s4,
               test_explicit_family_cell_registry]:
        fn()
        print(f"{fn.__name__}: OK")
