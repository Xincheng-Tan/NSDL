"""The Phi-off system prompt: the same front-end with the deterministic backend removed.

WHY THIS EXISTS. P10.S2's main-text y-axis is a *difference*,
``gain(M) = acc(phi_on, M) - acc(phi_off, M)``, and the reported value is the M
depth where it turns positive. The first phi_off arm was a pure runtime gate
(``--disable-phi``): the Phi tools stayed advertised in the system prompt and
answered ``tool_unavailable``. That arm has three defects no amount of seeds can
fix (plan.md, 2026-08-13), the worst being that its score is dominated by how
gracefully a checkpoint handles a dead tool -- a nuisance that is itself
capacity-correlated, so the cross-cell comparison measures the wrong thing.

The root cause is that the model was never trained to do facet B without Phi, so
the gated arm is only a LOWER BOUND and systematically overstates Phi's
necessity. This module supplies the prompt half of the fix: a system prompt in
which schedule / verify / expand simply do not exist, paired with an SFT set
built with `--phi-off` so the checkpoint is in-distribution under it.

WHAT IS AND IS NOT REMOVED. Exactly ``nsdl.tool_runtime.PHI_TOOLS`` is removed.
``detect`` is facet-A perception and stays; ``verify_goal_coverage`` is a
read-only diagnosis of the model's own ``[GOAL]`` and stays. Ablating either
would mean the axis is no longer measuring Phi-offload. The mechanical ops
(carry_to / handoff / load_carrier / push_carrier / unload_carrier) are already
in ``## Operation Vocabulary``, so the surgery only changes WHO writes them, not
what vocabulary exists.

``[PLAN_SKETCH]`` and ``gaps`` are deliberately kept. Keeping both arms'
supervision shape identical is what makes the difference interpretable, and
``sketch`` is part of ``teacher_cache_key`` -- changing it would invalidate the
v3.3 teacher cache for no experimental gain.

FAILING LOUDLY IS THE POINT. Every anchor below is asserted to actually match.
plan.md records that ``schedule`` is "硬编码贯穿" the profile prompt; if an
upstream edit reworded one of these paragraphs, a silent no-op ``.replace()``
would leave ``schedule`` advertised in what is supposed to be the ablated arm and
the whole S2 axis would quietly measure nothing.
"""

from __future__ import annotations

import re

from nsdl.profiles.viki_l2 import VIKI_L2_PROFILE

# --- (A) Tool Specs: drop the three Phi bullets -----------------------------
# Kept verbatim from nsdl/profiles/viki_l2/__init__.py so a reworded upstream
# bullet trips the assertion instead of silently surviving into the arm.
_SCHEDULE_BULLET = """- schedule: deterministic mechanical completion, resource scheduling, and feasibility.
  Args: {"ops":[intent op objects with robot fields], "assign":{"R1":{"items":[...]}, ...}}
  Returns: {"ok":bool, "ops":[complete DAG], "schedule":[micro steps], "inserted":[...], "reason":null|{...}}
  Call when the sketch has open-container, multi-robot transfer, carrier, handoff, end-effector, or dependency uncertainty.
"""

_VERIFY_BULLET = """- verify: feasibility check without asking the model to invent missing high-level targets.
  Args: {"ops":[op objects]}
  Returns: {"ok":bool, "code":null|string, "op_ids":[...], "message":string}
  Call when you want a structured diagnosis before finalizing or after a repair.
"""

_EXPAND_BULLET = """- expand: local lowering of one op/node into primitive actions.
  Args: {"op":{op object}}
  Returns: {"ok":bool, "primitives":[...], "end_state":[...], "reason":null|{...}}
  Call only for local lowering ambiguity; schedule already uses it internally.
"""

# --- (B) mechanical-op synthesis: the model now owns it ---------------------
_MECHANICAL_OLD = """Mechanical ops such as carry_to, handoff, load_carrier, push_carrier, and unload_carrier
are synthesized by `schedule` when needed. If you call `schedule` and it succeeds,
the final `[PLAN]` should adopt its returned complete DAG.
"""

_MECHANICAL_NEW = """Mechanical ops such as carry_to, handoff, load_carrier, push_carrier, and unload_carrier
are NOT synthesized for you. When the task needs them, write them into the final
`[PLAN]` yourself, together with the dependency edges that order them, using the
op types listed in the Operation Vocabulary above.
"""

# --- (C) cross-robot collaboration -----------------------------------------
_CROSS_ROBOT_OLD = (
    "- Cross-robot collaboration is expressed inside `[PLAN_SKETCH]`: if one robot "
    "acquires an item and another deposits it, that robot assignment is the "
    "collaboration intent. `schedule` derives handoff/carrier details.\n"
)

_CROSS_ROBOT_NEW = (
    "- Cross-robot collaboration is expressed inside `[PLAN_SKETCH]`: if one robot "
    "acquires an item and another deposits it, that robot assignment is the "
    "collaboration intent. You must then write the handoff / carrier ops that "
    "realize it explicitly into `[PLAN]`.\n"
)

# --- (D) gaps in Planning Rules --------------------------------------------
_GAPS_RULE_OLD = (
    "- Use `<gap>` or `gaps` to mark deterministic details left for tools: `deps` "
    "for missing dependency edges and `precond` for missing mechanical preconditions.\n"
)

_GAPS_RULE_NEW = (
    "- Use `<gap>` or `gaps` in `[PLAN_SKETCH]` to mark deterministic details you "
    "still owe: `deps` for missing dependency edges and `precond` for missing "
    "mechanical preconditions. No tool will fill them; resolve every one of them "
    "yourself in the final `[PLAN]`.\n"
)

# --- (E) the [TOOL_CALL] example -------------------------------------------
_TOOL_CALL_EXAMPLE_OLD = '{"name":"schedule","args":{"ops":[...],"assign":{...}}}\n'
_TOOL_CALL_EXAMPLE_NEW = '{"name":"detect","args":{"query":"bowl on the dining table"}}\n'

# --- (F) op_type description in [PLAN] -------------------------------------
_OP_TYPE_OLD = "- op_type: an intent op, or a complete DAG op returned by `schedule`.\n"
_OP_TYPE_NEW = (
    "- op_type: an intent op, or any mechanical op from the Operation Vocabulary "
    "(carry_to, handoff, load_carrier, push_carrier, unload_carrier).\n"
)

# --- (G) the [PLAN_SKETCH] line in the Output Protocol ----------------------
_SKETCH_LINE_OLD = (
    "- Use only intent ops listed above. Leave deterministic details to tools via `gaps`.\n"
)
_SKETCH_LINE_NEW = (
    "- Use only intent ops listed above. Mark deterministic details with `gaps`, "
    "then resolve them yourself in `[PLAN]`.\n"
)

# (old, new, label). An empty `new` deletes the block.
_SURGERY: tuple[tuple[str, str, str], ...] = (
    (_SCHEDULE_BULLET, "", "A.schedule_bullet"),
    (_VERIFY_BULLET, "", "A.verify_bullet"),
    (_EXPAND_BULLET, "", "A.expand_bullet"),
    (_MECHANICAL_OLD, _MECHANICAL_NEW, "B.mechanical_synthesis"),
    (_CROSS_ROBOT_OLD, _CROSS_ROBOT_NEW, "C.cross_robot"),
    (_GAPS_RULE_OLD, _GAPS_RULE_NEW, "D.gaps_rule"),
    (_TOOL_CALL_EXAMPLE_OLD, _TOOL_CALL_EXAMPLE_NEW, "E.tool_call_example"),
    (_OP_TYPE_OLD, _OP_TYPE_NEW, "F.op_type"),
    (_SKETCH_LINE_OLD, _SKETCH_LINE_NEW, "G.sketch_line"),
)

# Survivors, asserted after the surgery: these are NOT Phi and their removal
# would change what the ablation measures.
_MUST_SURVIVE = ("- detect:", "- verify_goal_coverage:")


class PhiOffPromptError(RuntimeError):
    """An anchor stopped matching, so the ablated prompt cannot be trusted."""


# Under the v3.4 / v3.4.1 toolset contracts the profile prompt reaches this
# module with these bullets ALREADY removed by ``apply_toolset_contract``
# (v34: verify + verify_goal_coverage; v341: + expand). Their anchors are
# therefore allowed to be absent, and only ``detect`` must survive.
_CONTRACT_PRE_REMOVED = {
    "v33": (),
    "v34": ("A.verify_bullet",),
    "v341": ("A.verify_bullet", "A.expand_bullet"),
}


def apply_phi_off_system_prompt(system_prompt: str, contract: str | None = None) -> str:
    """Return `system_prompt` with the deterministic backend Phi removed.

    Raises `PhiOffPromptError` if any anchor fails to match, or if `schedule` /
    `verify` / `expand` survive as advertised tools -- a partial surgery is worse
    than no surgery, because it looks like a valid arm.

    ``contract`` (default: env ``LNSDL_TOOLSET_CONTRACT``) names which Phi
    bullets the toolset contract has already stripped (plan 序 33, 2026-09-11:
    the v3.4.1 Phi-off arm = ``{detect}`` only); those anchors may be absent.
    """
    from nsdl.toolset_contract import resolve_toolset_contract

    resolved = resolve_toolset_contract(contract)
    pre_removed = _CONTRACT_PRE_REMOVED[resolved]
    output = system_prompt
    for old, new, label in _SURGERY:
        if old not in output:
            if label in pre_removed:
                continue
            raise PhiOffPromptError(
                f"anchor {label!r} did not match; the viki_l2 profile prompt changed "
                f"upstream and nsdl/phi_off_prompt.py must be updated to match"
            )
        output = output.replace(old, new)

    leaked = re.search(r"\bschedule\b", output)
    if leaked:
        start = max(0, leaked.start() - 80)
        raise PhiOffPromptError(
            f"`schedule` survives the Phi-off surgery near: {output[start:leaked.end() + 80]!r}"
        )
    for name in ("- verify:", "- expand:"):
        if name in output:
            raise PhiOffPromptError(f"{name!r} survives the Phi-off surgery")
    must_survive = _MUST_SURVIVE if resolved == "v33" else ("- detect:",)
    for name in must_survive:
        if name not in output:
            raise PhiOffPromptError(
                f"{name!r} was removed; only PHI_TOOLS may be ablated (detect is "
                f"facet-A perception, verify_goal_coverage is a read-only diagnosis)"
            )
    if resolved != "v33" and "- verify_goal_coverage:" in output:
        raise PhiOffPromptError(
            f"'- verify_goal_coverage:' survives under contract {resolved}, which hides it"
        )
    return output


def build_phi_off_system_prompt() -> str:
    """The VIKI-L2 system prompt with Phi removed -- the arm's single source of truth.

    Both the SFT builder and the eval call this, so the two sides are byte-identical
    by construction rather than by convention. Under ``LNSDL_TOOLSET_CONTRACT``
    v34 / v341 the contract strip is applied first (same as
    ``build_slm_system_prompt``), so the Phi-off arm of the paper contract
    advertises exactly ``{detect}``.
    """
    from nsdl.toolset_contract import apply_toolset_contract, resolve_toolset_contract

    contract = resolve_toolset_contract()
    base = apply_toolset_contract(VIKI_L2_PROFILE.build_system_prompt(), contract)
    return apply_phi_off_system_prompt(base, contract)
