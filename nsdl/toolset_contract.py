"""Model-visible toolset contracts (plan.md A0g ①② + A0h expand removal).

v3.4 collapses the model-visible toolset from 5 to 3 —
``{detect, schedule, expand}`` — by removing the ``verify`` (Phi dry-run,
functionally redundant with ``schedule`` ok:false diagnostics) and
``verify_goal_coverage`` (unused: 7B VIKI calls == 0, LOCUS 2.0%) tool-spec
bullets from the system prompt. Both backends stay implemented and callable in
``nsdl/tool_runtime.py``; only the model-visible surface changes.

v3.4.1 (2026-09-01 user decision) further removes ``expand``: zero corpus
demonstrations, 2/29891 measured trace calls, function subsumed by
``schedule`` step (c) with failures surfaced via ``goal_unreachable``.
Final paper contract = ``{detect, schedule}``.

The contract is selected via the ``LNSDL_TOOLSET_CONTRACT`` env var
(``v341`` = 2-tool, the paper contract and the default; ``v34`` = 3-tool;
``v33`` = legacy 5-tool prompt, byte-identical).
Env-based so one switch covers the SFT builder, RL parquet builder and eval
paths, which must all render byte-identical system prompts.

Frozen v3.3 buckets/corpora are interpreted under v33; do not re-render them.
"""
from __future__ import annotations

import os
import re

TOOLSET_CONTRACT_ENV = "LNSDL_TOOLSET_CONTRACT"
TOOLSET_CONTRACT_V33 = "v33"
TOOLSET_CONTRACT_V34 = "v34"
TOOLSET_CONTRACT_V341 = "v341"
TOOLSET_CONTRACTS = (TOOLSET_CONTRACT_V33, TOOLSET_CONTRACT_V34, TOOLSET_CONTRACT_V341)

# A removed bullet starts at column 0 with "- <name>: " and runs until the next
# column-0 bullet. Matching structurally (not on frozen wording) so profile
# wording edits inside the bullet do not silently break the strip.
_REMOVED_TOOLS_BY_CONTRACT = {
    TOOLSET_CONTRACT_V34: ("verify", "verify_goal_coverage"),
    TOOLSET_CONTRACT_V341: ("verify", "verify_goal_coverage", "expand"),
}


def resolve_toolset_contract(value: str | None = None) -> str:
    raw = (value if value is not None else os.environ.get(TOOLSET_CONTRACT_ENV, "")).strip().lower()
    if not raw:
        return TOOLSET_CONTRACT_V341
    if raw not in TOOLSET_CONTRACTS:
        raise ValueError(
            f"Unknown toolset contract {raw!r} (env {TOOLSET_CONTRACT_ENV}); "
            f"expected one of {TOOLSET_CONTRACTS}."
        )
    return raw


def apply_toolset_contract(system_prompt: str, contract: str | None = None) -> str:
    """Return the system prompt under the given (or env-resolved) contract.

    v33 returns the input unchanged (byte-identical). v34 removes the
    ``verify`` and ``verify_goal_coverage`` tool-spec bullets and fails loudly
    if either bullet is missing or matches more than once — a drifted prompt
    must never be silently accepted.
    """
    resolved = resolve_toolset_contract(contract)
    if resolved == TOOLSET_CONTRACT_V33:
        return system_prompt
    removed = _REMOVED_TOOLS_BY_CONTRACT[resolved]
    output = system_prompt
    for tool in removed:
        pattern = re.compile(rf"(?ms)^- {re.escape(tool)}: .*?(?=^- )")
        matches = pattern.findall(output)
        if len(matches) != 1:
            raise ValueError(
                f"toolset contract {resolved}: expected exactly 1 '- {tool}: ' spec bullet "
                f"in the system prompt, found {len(matches)}; refusing to build a "
                "drifted prompt."
            )
        output = pattern.sub("", output)
    for tool in removed:
        if re.search(rf"\b{re.escape(tool)}\b", output):
            raise ValueError(
                f"toolset contract {resolved}: residual mention of {tool!r} after strip; "
                "the profile prompt gained a new reference that must be handled "
                "explicitly."
            )
    return output
