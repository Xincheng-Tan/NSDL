"""LNSDL_IMAGE_MAX_PIXELS: the eval-side input-resolution cap (A0b, 2026-08-26).

The point of these tests is that the flag is *default-off and provably so*. Every
LOCUS bucket on disk was produced at full resolution; a cap that leaked on by
accident would make new numbers silently incomparable with all of them.
"""
from __future__ import annotations

import os

import pytest

from nsdl.slm_backends.local_vl import ENV_IMAGE_MAX_PIXELS, _apply_image_pixel_cap

CKPT = "checkpoints/Qwen2.5-VL-7B-Instruct"


class _FakeImageProcessor:
    def __init__(self, min_pixels: int = 3136, max_pixels: int = 12845056):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}


class _FakeProcessor:
    def __init__(self, image_processor=None):
        self.image_processor = image_processor


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)


def test_unset_is_a_noop():
    ip = _FakeImageProcessor()
    assert _apply_image_pixel_cap(_FakeProcessor(ip)) is None
    assert ip.max_pixels == 12845056
    assert ip.size["longest_edge"] == 12845056


def test_empty_string_is_also_a_noop(monkeypatch):
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "   ")
    ip = _FakeImageProcessor()
    assert _apply_image_pixel_cap(_FakeProcessor(ip)) is None
    assert ip.max_pixels == 12845056


def test_cap_sets_both_scalar_and_size_dict(monkeypatch):
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    ip = _FakeImageProcessor()
    assert _apply_image_pixel_cap(_FakeProcessor(ip)) == 524288
    assert ip.max_pixels == 524288
    assert ip.size["longest_edge"] == 524288
    assert ip.size["shortest_edge"] == 3136  # min side untouched


@pytest.mark.parametrize("bad", ["960x540", "abc", "1e6"])
def test_non_integer_raises_rather_than_falling_back(monkeypatch, bad):
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, bad)
    with pytest.raises(ValueError):
        _apply_image_pixel_cap(_FakeProcessor(_FakeImageProcessor()))


def test_cap_below_min_pixels_raises(monkeypatch):
    """Below min_pixels smart_resize scales images back UP -- never silently."""
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "1000")
    with pytest.raises(ValueError, match="min_pixels"):
        _apply_image_pixel_cap(_FakeProcessor(_FakeImageProcessor()))


def test_processor_without_image_processor_raises(monkeypatch):
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    with pytest.raises(RuntimeError):
        _apply_image_pixel_cap(_FakeProcessor(None))


@pytest.mark.skipif(not os.path.isdir(CKPT), reason="foundation checkpoint absent")
@pytest.mark.parametrize(
    "cap,expected_tokens",
    [
        (None, 2691),      # 1920x1080 untouched: the number that caused the OOM
        (524288, 646),     # 960x540
        (921600, 1125),    # ~1280x720 (smart_resize floors on the way down)
    ],
)
def test_real_processor_token_count_on_a_locus_sized_image(monkeypatch, cap, expected_tokens):
    """The cap must actually bite, measured in vision tokens, not just set a field."""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    if cap is None:
        monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)
    else:
        monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, str(cap))
    _apply_image_pixel_cap(proc)

    img = Image.new("RGB", (1920, 1080), (128, 128, 128))
    out = proc.image_processor(images=[img], return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()   # (t, h, w) in patches
    merge = proc.image_processor.merge_size
    assert grid[1] * grid[2] // (merge * merge) == expected_tokens


@pytest.mark.skipif(not os.path.isdir(CKPT), reason="foundation checkpoint absent")
def test_cap_leaves_crop_sized_images_alone(monkeypatch):
    """Crops (short side 112) sit far below any cap we would set -- unchanged."""
    from PIL import Image
    from transformers import AutoProcessor

    proc_full = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    proc_cap = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    _apply_image_pixel_cap(proc_cap)

    crop = Image.new("RGB", (168, 112), (64, 64, 64))
    a = proc_full.image_processor(images=[crop], return_tensors="pt")["image_grid_thw"].tolist()
    b = proc_cap.image_processor(images=[crop], return_tensors="pt")["image_grid_thw"].tolist()
    assert a == b
