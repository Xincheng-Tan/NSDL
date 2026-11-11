"""The cap has to reach SFT and RL, not just eval (A0b, 2026-08-27).

The failure this guards against is silent by construction: a run that *claims* a
cap but preprocesses at 2,691 tokens produces a plausible number and no error.
chain i caught it before launch only because the smoke test grepped the worker
log for the applied-cap line.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import sys
import tempfile

import pytest

# `import verl` from the repo root is a FAKE success: the editable-install .pth
# points at /path/to/NSDL/verl, which no longer exists, so
# `verl` resolves to a namespace package over ./verl/ and `verl.utils` is absent.
# The RL launcher works because it prepends REPO_ROOT/verl to PYTHONPATH
# (scripts/run_dapo_lnsdl_7b.sh:71); the test has to do the same or it would
# "fail to import" a module the real run imports fine.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "verl"))

from nsdl.image_pixel_cap import (
    ENV_IMAGE_MAX_PIXELS,
    apply_image_pixel_cap,
    resolve_image_pixel_cap,
)

CKPT = pathlib.Path("checkpoints/Qwen2.5-VL-7B-Instruct")


def test_resolve_unset(monkeypatch):
    monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)
    assert resolve_image_pixel_cap() is None


def test_resolve_whitespace_is_unset(monkeypatch):
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "   ")
    assert resolve_image_pixel_cap() is None


@pytest.mark.parametrize("bad", ["960x540", "abc", "1e6", ""])
def test_resolve_rejects_non_integer(monkeypatch, bad):
    if bad == "":
        pytest.skip("empty string is the documented 'unset' spelling")
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, bad)
    with pytest.raises(ValueError):
        resolve_image_pixel_cap()


def test_local_vl_still_exports_old_names():
    """local_vl delegates now; the old names are load-bearing for the eval path."""
    mod = importlib.import_module("nsdl.slm_backends.local_vl")
    assert mod.ENV_IMAGE_MAX_PIXELS == ENV_IMAGE_MAX_PIXELS
    assert callable(mod._apply_image_pixel_cap)


def test_verl_hf_processor_applies_cap(monkeypatch):
    """RL's single processor funnel must honour the cap.

    Patching `process_image` instead would NOT be equivalent: that is a PIL
    pre-resize with float scaling, while SFT and eval cap at the processor where
    smart_resize floors to multiples of 28. The two disagree by a row of patches.
    """
    from verl.utils.tokenizer import hf_processor

    if not CKPT.is_dir():
        pytest.skip(f"{CKPT} not present")
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    proc = hf_processor(str(CKPT), trust_remote_code=True)
    assert proc is not None
    assert int(proc.image_processor.max_pixels) == 524288


def test_verl_hf_processor_unset_is_full_resolution(monkeypatch):
    from verl.utils.tokenizer import hf_processor

    if not CKPT.is_dir():
        pytest.skip(f"{CKPT} not present")
    monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)
    proc = hf_processor(str(CKPT), trust_remote_code=True)
    assert int(proc.image_processor.max_pixels) > 10_000_000


def test_cap_persists_through_save_pretrained(monkeypatch):
    """SFT bakes the cap into the ckpt, which is what makes RL/eval matched.

    If this ever stops holding, a capped-trained ckpt would be evaluated at full
    resolution by default and the mismatch would look like a capability result.
    """
    from transformers import AutoProcessor

    if not CKPT.is_dir():
        pytest.skip(f"{CKPT} not present")
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    proc = AutoProcessor.from_pretrained(str(CKPT), trust_remote_code=True)
    apply_image_pixel_cap(proc, tag="test")
    tmp = tempfile.mkdtemp()
    try:
        proc.save_pretrained(tmp)
        cfg = json.loads((pathlib.Path(tmp) / "preprocessor_config.json").read_text())
        assert (
            int(cfg.get("max_pixels", -1)) == 524288
            or int((cfg.get("size") or {}).get("longest_edge", -1)) == 524288
        )
        # And reloading it, with the env var gone, must still be capped.
        monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)
        reloaded = AutoProcessor.from_pretrained(tmp, trust_remote_code=True)
        assert int(reloaded.image_processor.max_pixels) == 524288
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
