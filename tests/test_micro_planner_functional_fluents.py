"""Single-valued fluents must not leak across macro-node boundaries.

The defect this pins: ``move_to:<X>``'s ``del_effects`` can only name entities
the *building node* knows about (``_collect_move_targets``). An ``executor_at``
established by an earlier node therefore survives into a later node whose target
set happens not to mention that entity -- so the later node's ``place_item``
precondition ``executor_at(<dest>)`` is already satisfied and the planner emits
``Place`` with no preceding ``Move``.

Observed on ``ramp63k`` as 80/900 prompts whose own ground truth the executor
could not reproduce (uniformly one step short: 9 emitted vs 10 expected). It is
not only a scoring artefact -- ``_dispatch_graph`` is the deployed executor path,
so the robot really would try to Place at a surface it has walked away from.

Run with:
    python -m pytest tests/test_micro_planner_functional_fluents.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nsdl.aig_builder import build_affordance_context  # noqa: E402
from nsdl.dispatcher import dispatch_graph  # noqa: E402
from nsdl.macro_graph import _assign_executors_to_graph  # noqa: E402
from nsdl.micro_planner import (  # noqa: E402
    FUNCTIONAL_FLUENTS,
    GroundMicroOperator,
    _apply_operator_plan,
    _predicate_name,
    plan_node_micro_program,
)
from nsdl.schema import OperationStep, RobotInstance, SceneFacts  # noqa: E402
from nsdl.serializer import to_l2_answer  # noqa: E402
from nsdl.slm_interface import convert_operations_to_graph_plan  # noqa: E402


DEST = "kitchen island area"


def _scene() -> tuple[SceneFacts, object]:
    facts = SceneFacts(
        parser_backend="test",
        instruction=f"move the pear and the pizza to the {DEST}",
        image_available=False,
        robot_inventory=[RobotInstance(slot_id="R1", robot_type="unitree_h1")],
        primary_object="pear",
        destination=DEST,
    )
    return facts, build_affordance_context(facts)


def _revisit_ops() -> list[OperationStep]:
    """Two deposits to the same destination with an acquire in between.

    The robot must walk away (to the pizza) and therefore must walk back before
    the second Place.
    """
    return [
        OperationStep(op_id="op1", op_type="acquire", entity="pear", deps=[], robot="R1"),
        OperationStep(op_id="op2", op_type="deposit", entity="pear", dest=DEST, deps=["op1"], robot="R1"),
        OperationStep(op_id="op3", op_type="acquire", entity="pizza", deps=["op2"], robot="R1"),
        OperationStep(op_id="op4", op_type="deposit", entity="pizza", dest=DEST, deps=["op3"], robot="R1"),
    ]


def _planned_graph(ops: list[OperationStep]):
    facts, ctx = _scene()
    graph = convert_operations_to_graph_plan(ops, ctx, facts)
    return _assign_executors_to_graph(graph, facts)


def _node_programs(ops: list[OperationStep]) -> list[tuple[str, tuple[tuple[str, ...], ...], frozenset[str]]]:
    graph = _planned_graph(ops)
    node_map = {node.node_id: node for node in graph.nodes}
    end_states: dict[str, frozenset[str]] = {}
    programs = []
    for node in graph.nodes:
        program = plan_node_micro_program(
            node=node,
            node_map=node_map,
            predecessor_end_states=end_states,
        )
        end_states[node.node_id] = program.final_state
        programs.append((node.node_id, program.actions, program.final_state))
    return programs


def _action_pairs(ops: list[OperationStep]) -> list[tuple[str, str]]:
    micro_steps = dispatch_graph(_planned_graph(ops))
    return [
        (action[0], action[1] if len(action) > 1 else "")
        for step in to_l2_answer(micro_steps)
        for action in step["actions"].values()
    ]


def test_revisiting_a_destination_still_emits_the_move() -> None:
    pairs = _action_pairs(_revisit_ops())
    assert pairs == [
        ("Move", "pear"),
        ("Reach", "pear"),
        ("Grasp", "pear"),
        ("Move", DEST),
        ("Place", DEST),
        ("Move", "pizza"),
        ("Reach", "pizza"),
        ("Grasp", "pizza"),
        ("Move", DEST),
        ("Place", DEST),
    ], pairs


def test_no_node_ends_with_the_executor_in_two_places() -> None:
    for node_id, _actions, final_state in _node_programs(_revisit_ops()):
        for name in FUNCTIONAL_FLUENTS:
            held = sorted(p for p in final_state if _predicate_name(p) == name)
            assert len(held) <= 1, f"{node_id} ends holding {len(held)} {name}: {held}"


def test_the_intervening_acquire_clears_the_previous_destination() -> None:
    """The precise leak: node 3 (acquire pizza) never names DEST, so its own
    ``del_effects`` cannot retract ``executor_at(DEST)``."""
    programs = {node_id: state for node_id, _actions, state in _node_programs(_revisit_ops())}
    acquire_pizza = programs["n3"]
    assert f"executor_at(pizza)" in acquire_pizza
    assert f"executor_at({DEST})" not in acquire_pizza


def test_functional_enforcement_is_a_no_op_inside_a_single_node() -> None:
    """Intra-node, the operators' own del_effects already suffice -- so the fix
    cannot silently change any single-node plan."""
    for node_id, actions, _state in _node_programs(
        [
            OperationStep(op_id="op1", op_type="acquire", entity="pear", deps=[], robot="R1"),
            OperationStep(op_id="op2", op_type="deposit", entity="pear", dest=DEST, deps=["op1"], robot="R1"),
        ]
    ):
        assert actions, node_id
    assert _action_pairs(
        [
            OperationStep(op_id="op1", op_type="acquire", entity="pear", deps=[], robot="R1"),
            OperationStep(op_id="op2", op_type="deposit", entity="pear", dest=DEST, deps=["op1"], robot="R1"),
        ]
    ) == [
        ("Move", "pear"),
        ("Reach", "pear"),
        ("Grasp", "pear"),
        ("Move", DEST),
        ("Place", DEST),
    ]


def test_enforcement_only_retracts_the_fluents_the_operator_supersedes() -> None:
    operator = GroundMicroOperator(
        name="move_to:b",
        actions=(("Move", "b"),),
        preconditions=(),
        add_effects=("executor_at(b)",),
        del_effects=(),
    )
    state = _apply_operator_plan(
        frozenset({"executor_at(a)", "reaching(a)", "holding(cup)", "free_end_effectors(1)"}),
        [operator],
    )
    assert "executor_at(a)" not in state
    assert "executor_at(b)" in state
    # `reaching` is functional too, but this operator does not supersede it, so
    # enforcement must leave it alone -- retraction is the operator's job.
    assert "reaching(a)" in state
    # Non-functional fluents are never touched.
    assert "holding(cup)" in state
    assert "free_end_effectors(1)" in state
