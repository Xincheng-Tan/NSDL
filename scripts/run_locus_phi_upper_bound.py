"""§7-7 third strong-baseline row: the pipeline full-injection upper bound.

WHY THIS ROW EXISTS. §7-7 names three strong baselines -- third-party zero-shot,
a VIKI-R reproduction, and the **pipeline full-injection upper bound** -- and
§7.9's terminal table only ever carried the first (8 rows = 4 closed models x 2
conditions). §7.1 item 7 and the §7.2 selection table both already record that
this row "takes no API, CPU is enough". Its job is to separate **hard** from
**unsolvable**: a cell every method scores 0 on is a difficulty reading only if
something CAN score above 0 there. Where this bound itself fails, the cell's
zero is a property of the construction or of our own pipeline, not of the
methods being compared -- and §7-7's discriminativeness argument uses exactly
that as its denominator.

WHAT IS INJECTED. Everything neural. There is no model, no backend, no prompt:

  * the intent sketch comes from the GROUND TRUTH plan (``reverse_map_time_steps``
    then ``make_locus_sketch``), so facet A -- which objects are active, where
    they go -- is answered perfectly;
  * the robot slot on each op is the GT's, so robot selection is answered too;
  * the scene facts come from ``build_scene_facts_from_gt``, i.e. the SAME rule
    scene parser the deployed eval runs, over the deployed instruction text plus
    the GT robot inventory.

WHAT IS NOT INJECTED, and is therefore what this row measures:

  * ``variant=phi_sketch_opschain`` (the headline row from 2026-08-31) erases
    the cross-robot dependencies only, keeping every op type and each robot's
    own sequential order. This is the DEPLOYED contract after the two-level
    destination fix (debug_sft.md §4.5.1): carry_to turned out to carry intent
    -- a destination Φ provably cannot re-infer -- so it is no longer the
    solver's to rebuild. The row therefore reads as **reconstruction
    completeness under the deployed contract**: whatever it cannot rebuild,
    no model on this contract can either.
  * ``variant=phi_sketch_opschain_noopen`` is ``phi_sketch_opschain`` with every
    ``open``/``close`` op (and any ``auto_open*`` id) stripped from the sketch
    before Φ, deps onto them dropped, and the per-robot chain re-threaded over
    what remains. This is the DEPLOYED contract as actually exercised: the
    model never writes open/close (7B main row, ood_m: 2/200 traces carry a
    model-written open, 57/200 carry a Φ ``auto_open``), so Φ's own insertion
    (``nsdl.planning_tool._insert_missing_open_ops``, placement chosen by
    ``LNSDL_PHI_OPEN_PLACEMENT``) is part of what this row must reproduce.
    ``phi_sketch_opschain`` cannot see that: it hands Φ the GT's own ``open``
    and the insertion never fires. Both knobs are stamped on every row
    (``phi_open_placement`` / ``phi_scheduler``) and ``--tag`` keeps the two
    placements in separate files.
  * ``variant=phi_sketch_opschain_nodepopen`` strips only the DEPOSIT-side
    opens: an ``open(C)`` is kept when some acquired entity starts inside ``C``
    (GT ``init_pos``), because Φ only ever inserts open-before-deposit and the
    corpus does demonstrate the source-side open (F4_s2m1/m2, 105 rows each).
    It isolates the placement question from the acquire-side gap that
    ``noopen`` also exposes.
  * ``variant=phi_sketch`` is the SUPERSEDED contract (mechanical ops erased),
    kept runnable because the frozen .7125 reading in §7.9 was taken on it and
    a superseded number must stay reproducible. Do not mix the two columns:
    ops_chain injects strictly more, so it is a looser (higher) bound, and its
    failures are a strictly smaller set. Under the old contract a cell could
    fail for a reason the deployed pipeline no longer has -- which is exactly
    why the old row over-charged Φ.
  * ``variant=gt_ops`` erases NOTHING and skips Φ's synthesis entirely: the GT
    operation DAG goes straight into the deployed graph builder, dispatcher and
    emitter. A failure here is *rendering* loss -- our op vocabulary or emitter
    cannot express the GT plan at all -- and it is the floor under the other
    variant. Reading phi_sketch without gt_ops would charge Φ for it.

The two together localize a failing cell: gt_ops == 0 means unrepresentable;
gt_ops == 1 with phi_sketch == 0 means Φ could not rebuild it.

NULL-GT ROWS ARE NOT RUN. 73/560 frozen rows have an empty GT plan (the goal
already holds at the initial state). The correct answer is no actions, the
injection has nothing to inject, and ``compute_locus_l2_acc`` scores an empty
plan 1.0 there. They are emitted as ``<answer>[]</answer>`` and marked
``phi_stage=null_task`` so the headline number can be read with or without them.

A PIPELINE FAILURE IS A ZERO, NOT A HOLE. §7-7's three collection denominators
exist because a relay blackout must not be scored as difficulty. There is no
transport here: every row is attempted locally and deterministically, so a Φ or
render failure is a genuine reading of 0. Such a row is written with the failure
code and NO ``<answer>`` tag, which the shared scorer reads as acc 0
(``no_answer_tag``); the machine-readable reason is in ``phi_stage``/``phi_code``.

OUTPUT is the strong-baseline collection JSONL format, so
``scripts/score_locus_strong_baseline.py`` scores it on the same ruler
(``compute_locus_l2_acc``, plus the official column) with no new scoring code.
It is written to its OWN directory, deliberately not into
``data/lnsdl_bench/strong_baseline/``: §7-7's saturated / all-zero cell verdicts
are computed over *methods*, and pooling an oracle upper bound into that vote
would erase the very all-zero verdicts the gate exists to raise.

Run (all four frozen splits, ~minutes, no GPU):

    python scripts/run_locus_phi_upper_bound.py --variant phi_sketch_opschain
    LNSDL_PHI_OPEN_PLACEMENT=early python scripts/run_locus_phi_upper_bound.py \
        --variant phi_sketch_opschain_noopen --tag early
    LNSDL_PHI_OPEN_PLACEMENT=late_freehand python scripts/run_locus_phi_upper_bound.py \
        --variant phi_sketch_opschain_noopen --tag late_freehand
    python scripts/run_locus_phi_upper_bound.py --variant gt_ops
    python scripts/run_locus_phi_upper_bound.py --variant phi_sketch  # historical
    python scripts/score_locus_strong_baseline.py \
        --dir data/lnsdl_bench/phi_upper_bound \
        --json outputs/locus_bench/phi_upper_bound_scores.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "scripts"), str(_REPO / "RoboFactory" / "script_general")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.dispatcher import dispatch_graph  # noqa: E402
from nsdl.macro_graph import _assign_executors_to_graph  # noqa: E402
from nsdl.planning_tool import (  # noqa: E402
    phi_open_placement_mode, phi_scheduler_mode, schedule, verify,
)
from nsdl.profiles import VIKI_L2_PROFILE, use_benchmark_profile  # noqa: E402
from nsdl.prompt_adapter import load_benchmark_samples  # noqa: E402
from nsdl.schema import DispatcherPayload  # noqa: E402
from nsdl.slm_interface import convert_operations_to_graph_plan  # noqa: E402

from run_locus_strong_baseline import prompt_sha  # noqa: E402

SPLITS = ("test_id", "ood_s", "ood_m", "ood_x")
VARIANTS = ("phi_sketch", "phi_sketch_opschain", "phi_sketch_opschain_noopen",
            "phi_sketch_opschain_nodepopen", "gt_ops")
DEFAULT_OUT = _REPO / "data" / "lnsdl_bench" / "phi_upper_bound"
MODEL = "phi-full-injection"
API_SOURCE = "none(cpu)"

# What the deployed model is contracted NOT to write: Φ inserts these itself.
DOOR_OP_TYPES = frozenset({"open", "close"})
AUTO_OPEN_PREFIX = "auto_open"


def source_containers(ops: list, init_pos: dict | None) -> set[str]:
    """Containers some acquired entity starts inside, per the GT ``init_pos``
    (keys are ``<entity>_<n>``). An ``open`` on one of these is acquire-side:
    the model must write it, Φ will not insert it."""
    out: set[str] = set()
    acquired = {str(op.entity) for op in ops
                if op.op_type in ("acquire", "local_acquire") and op.entity}
    for key, positions in (init_pos or {}).items():
        if key.rsplit("_", 1)[0] in acquired and positions:
            out.update(str(pos) for pos in positions)
    return out


def strip_door_ops(ops: list, *, keep_dests: set[str] | frozenset[str] = frozenset()) -> list:
    """Drop every open/close op (and any ``auto_open*`` id) and every dep
    that pointed at one, except opens whose container is in ``keep_dests``.
    Remaining ops are returned as fresh ``OperationStep`` copies in the
    original order; the caller re-threads the chain."""
    from nsdl.schema import OperationStep  # noqa: PLC0415

    removed = {op.op_id for op in ops
               if (op.op_type in DOOR_OP_TYPES and str(op.dest or op.entity) not in keep_dests)
               or str(op.op_id).startswith(AUTO_OPEN_PREFIX)}
    return [
        OperationStep(
            op_id=op.op_id, op_type=op.op_type, entity=op.entity, dest=op.dest,
            deps=[dep for dep in (op.deps or []) if dep not in removed],
            robot=op.robot, params=dict(op.params or {}), gaps=list(op.gaps or []),
        )
        for op in ops if op.op_id not in removed
    ]


def _sample_data(sample) -> dict:
    """The GT dict shape ``reverse_map_time_steps`` / ``build_scene_facts_from_gt``
    consume, built from a frozen LOCUS eval row.

    The instruction is the DEPLOYED prompt text (``sample.instruction``), not
    ``ground_truth['description']``. §7.6 rewrote 66 instructions in place after
    freeze; the two agree on every current row, but the prompt is what every
    other arm was scored against and it is what ``prompt_sha`` stamps, so
    reading the GT copy would silently reintroduce the stale-text failure mode
    the scorer has a whole branch for.
    """
    gt = sample.ground_truth or {}
    robots_raw = dict(gt.get("robots") or {})
    return {
        "instruction": sample.instruction,
        "description": sample.instruction,
        "robot_inventory": [{"slot": slot, "type": rtype}
                            for slot, rtype in sorted(robots_raw.items())
                            if rtype is not None],
        "time_steps": gt.get("time_steps") or [],
        "task_name": gt.get("task_name"),
        "task_id": gt.get("task_id"),
        "goal_constraints": gt.get("goal_constraints"),
        "robots_raw": robots_raw,
    }


def run_one(sample, variant: str) -> tuple[list | None, str, str]:
    """-> (plan | None, stage, code). ``plan`` is the dispatcher's answer."""
    from build_slm_training_data import (  # noqa: PLC0415 - heavy import, lazily
        build_scene_facts_from_gt, reverse_map_time_steps,
    )
    from locus_validate import make_locus_sketch  # noqa: PLC0415
    from scripts.audit_planning_tool_p1 import make_sketch  # noqa: PLC0415

    gt = sample.ground_truth or {}
    if not gt.get("time_steps"):
        # Nothing to inject and nothing to solve: the correct plan is empty.
        return [], "null_task", ""

    data = _sample_data(sample)
    stage = "scene_facts"
    try:
        with use_benchmark_profile(VIKI_L2_PROFILE):
            scene_facts = build_scene_facts_from_gt(data, idx=sample.sample_index)
            stage = "affordance_context"
            ctx = build_affordance_context(scene_facts)
            stage = "reverse_map"
            gt_ops = reverse_map_time_steps(data)

            if variant == "gt_ops":
                # No erasure and no Φ synthesis: straight into the renderer.
                ops = gt_ops
            else:
                stage = "phi_schedule"
                if variant == "phi_sketch_opschain":
                    # The SAME function the deployed corpus builder calls
                    # (build_lnsdl_sft_data.py --sketch-variant ops_chain), so
                    # the bound and the models it bounds sit on one contract.
                    sketch = make_sketch(gt_ops, "ops_chain")
                elif variant == "phi_sketch_opschain_noopen":
                    # Strip BEFORE chaining so the per-robot chain is threaded
                    # over the ops the model actually writes (A -> open -> B
                    # becomes A -> B); Φ has to put the open back itself.
                    sketch = make_sketch(strip_door_ops(gt_ops), "ops_chain")
                elif variant == "phi_sketch_opschain_nodepopen":
                    keep = source_containers(gt_ops, gt.get("init_pos"))
                    sketch = make_sketch(strip_door_ops(gt_ops, keep_dests=keep), "ops_chain")
                else:
                    sketch = make_locus_sketch(gt_ops)
                result = schedule(sketch, assign={}, ctx=ctx, scene_facts=scene_facts)
                if not result.ok:
                    code = result.reason.code if result.reason else "schedule_failed"
                    return None, stage, code
                stage = "phi_verify"
                reason = verify(result.ops, ctx, scene_facts)
                if not reason.ok:
                    return None, stage, reason.code or "verify_failed"
                ops = result.ops

            stage = "graph_plan"
            graph_plan = convert_operations_to_graph_plan(ops, ctx, scene_facts)
            graph_plan.build_mode = f"phi_upper_bound_{variant}"
            graph_plan.family_shortcut_used = False
            graph_plan = _assign_executors_to_graph(graph_plan, scene_facts)
            stage = "dispatch"
            micro_steps = dispatch_graph(graph_plan)
            stage = "emit"
            plan = VIKI_L2_PROFILE.dispatcher_emitter.emit(DispatcherPayload(
                sample=sample,
                initial_scene_facts=scene_facts,
                scene_facts=scene_facts,
                graph_plan=graph_plan,
                micro_steps=micro_steps,
            ))
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the sweep
        return None, stage, f"exception:{exc.__class__.__name__}:{exc}"[:200]
    return plan, "ok", ""


def write_collection(
    variant: str,
    *,
    splits: list[str] | tuple[str, ...] = SPLITS,
    limit: int | None = None,
    out_dir: pathlib.Path = DEFAULT_OUT,
    tag: str = "",
    load_samples=None,
    quiet: bool = False,
) -> tuple[pathlib.Path, dict[str, int]]:
    """Run ``variant`` over ``splits`` and write the strong-baseline JSONL.

    ``load_samples(split) -> list[BenchmarkSample]`` defaults to the frozen
    eval loader; the deploy-contract audit passes an image-free loader for the
    train parquet (whose ``images`` column cannot be read whole). Returns the
    file path and the stage breakdown.
    """
    load_samples = load_samples or (lambda split: load_benchmark_samples(level="LOCUS", split=split))
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = variant + (f"__{tag}" if tag else "")
    path = out_dir / f"{MODEL}__{condition}.jsonl"
    # Both Φ knobs are read from the environment at call time; stamp them so a
    # file can never be misread as having been produced under another setting.
    phi_env = {"phi_open_placement": phi_open_placement_mode(),
               "phi_scheduler": phi_scheduler_mode()}
    if not quiet:
        print(f"variant={variant} tag={tag!r} {phi_env}", flush=True)
    stages: dict[str, int] = {}
    n_ok = 0
    n = 0
    t0 = time.time()
    with path.open("w", encoding="utf-8") as fh:
        for split in splits:
            samples = load_samples(split)
            if limit:
                samples = samples[:limit]
            for sample in samples:
                # The rule parser and the emitter both narrate to stdout; the
                # sweep's own progress line is the only thing worth keeping.
                with contextlib.redirect_stdout(io.StringIO()):
                    plan, stage, code = run_one(sample, variant)
                n += 1
                key = stage if not code else f"{stage}:{code.split(':')[0]}"
                stages[key] = stages.get(key, 0) + 1
                if plan is None:
                    response = (f"[phi-upper-bound] no plan produced at stage "
                                f"{stage}: {code}")
                else:
                    n_ok += 1
                    response = "<answer>" + json.dumps(plan, ensure_ascii=False) + "</answer>"
                fh.write(json.dumps({
                    "item_id": f"{split}:{sample.sample_index:05d}",
                    "split": split, "row": sample.sample_index,
                    "model": MODEL, "condition": condition,
                    "variant": variant, **phi_env,
                    "api_source": API_SOURCE, "status": "ok",
                    "prompt_sha": prompt_sha(sample.instruction),
                    "latency_s": 0.0, "response": response,
                    "has_answer_tag": plan is not None,
                    "phi_stage": stage, "phi_code": code,
                }, ensure_ascii=False) + "\n")
            if not quiet:
                print(f"  {split:<8} done  ({n} rows, {n_ok} produced a plan)", flush=True)

    if not quiet:
        print(f"\nwrote {path}  [{n} rows, {time.time() - t0:.0f}s]")
        print("stage breakdown (a non-`ok`/`null_task` stage is a genuine 0, not a hole):")
        for stage, count in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>4}  {stage}")
    return path, stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=VARIANTS, default="phi_sketch_opschain")
    ap.add_argument("--tag", default="",
                    help="suffix on the file name / condition (e.g. the "
                         "LNSDL_PHI_OPEN_PLACEMENT value) so runs under "
                         "different Φ env knobs do not overwrite each other")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows per split (smoke only -- a prefix, not a sample)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    write_collection(args.variant, splits=args.splits, limit=args.limit,
                     out_dir=args.out_dir, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
