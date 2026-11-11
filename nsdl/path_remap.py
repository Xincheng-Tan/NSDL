"""Rewrite main-machine absolute paths embedded in prebuilt SFT datasets.

The v3.3 SFT datasets store ``image_path`` (and the ``image_path`` inside
``crop_attachments_json``) as absolute paths under the main machine's project
root, e.g. ``/path/to/NSDL/data/viki/VIKI-L2/train/....png``.
That is fine on M and fine for eval (the eval splits carry image *bytes* in
their parquet), but the secondary machine B checks the repo out at
``$HOME/projects/NSDL`` as a non-root user, where ``/root`` is not even
readable -- the dataset contract validator dies with ``PermissionError`` before
step 0.

Rebuilding the datasets on B was rejected: the paths do not appear in the
``messages`` column, so they contribute no training tokens, and rewriting
7196 rows across parquet + HF arrow + jsonl to fix something that is purely a
mount-point difference trades a reversible config for an irreversible edit to
the training data itself.

So: one env var, applied at the three points that turn a stored path into a
filesystem access. Unset (the main machine) the functions here are the
identity, so M's behaviour is byte-for-byte unchanged.

    export NSDL_PATH_REMAP="/path/to/NSDL=>$HOME/projects/NSDL"

Multiple rules separate with ``;`` and apply in order, first match wins.
See docs/env_migration_4gpu.md §3.10 and §4.5.
"""

from __future__ import annotations

import os

_ENV_VAR = "NSDL_PATH_REMAP"
_RULE_SEP = ";"
_ARROW = "=>"


def _rules() -> list[tuple[str, str]]:
    spec = os.environ.get(_ENV_VAR, "").strip()
    if not spec:
        return []
    rules: list[tuple[str, str]] = []
    for chunk in spec.split(_RULE_SEP):
        chunk = chunk.strip()
        if not chunk:
            continue
        if _ARROW not in chunk:
            raise ValueError(
                f"{_ENV_VAR} rule {chunk!r} is malformed; expected '<from>{_ARROW}<to>'"
            )
        src, dst = chunk.split(_ARROW, 1)
        src, dst = src.strip(), dst.strip()
        if not src or not dst:
            raise ValueError(f"{_ENV_VAR} rule {chunk!r} has an empty side")
        rules.append((src, dst))
    return rules


def remap_path(path: str | os.PathLike[str] | None) -> str:
    """Apply the first matching prefix rule to ``path``.

    Returns the input unchanged when ``NSDL_PATH_REMAP`` is unset or no rule
    matches, so callers can wrap unconditionally.
    """
    if path is None:
        return ""
    text = str(path)
    for src, dst in _rules():
        if text.startswith(src):
            return dst + text[len(src) :]
    return text
