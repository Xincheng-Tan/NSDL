"""Cap provenance must record the EFFECT, not the request (A0b, 2026-08-29).

Regression cover for the defect found while reading the G1 main row: 560 rows
recorded ``image_max_pixels: null`` -- which the field's own comment defined as
"full 1920x1080" -- while actually running capped at 524,288 (646 vision
tokens), because the cap arrived via the checkpoint's ``preprocessor_config``
rather than via ``LNSDL_IMAGE_MAX_PIXELS``.
"""

from __future__ import annotations

import pytest

from nsdl.image_pixel_cap import (
    ENV_IMAGE_MAX_PIXELS,
    describe_image_pixel_cap,
    read_effective_image_pixel_cap,
)


class _ImageProcessor:
    def __init__(self, max_pixels=None, size=None):
        if max_pixels is not None:
            self.max_pixels = max_pixels
        if size is not None:
            self.size = size


class _Processor:
    def __init__(self, image_processor):
        self.image_processor = image_processor


def _proc(max_pixels=None, size=None):
    return _Processor(_ImageProcessor(max_pixels=max_pixels, size=size))


def test_reads_scalar_max_pixels():
    assert read_effective_image_pixel_cap(_proc(max_pixels=524288)) == 524288


def test_size_longest_edge_wins_over_scalar():
    """Mirrors apply_image_pixel_cap's own precedence: transformers>=4.49
    resolves through ``size``, so a stale scalar must not be reported."""
    p = _proc(max_pixels=12845056, size={"longest_edge": 524288})
    assert read_effective_image_pixel_cap(p) == 524288


def test_no_image_processor_is_unknown_not_full_resolution():
    class _Bare:
        pass

    assert read_effective_image_pixel_cap(_Bare()) is None
    assert describe_image_pixel_cap(_Bare())["image_max_pixels_source"] == "unavailable"


def test_g1_regression_env_unset_but_checkpoint_capped(monkeypatch):
    """THE bug. Env says nothing; the checkpoint carries the cap."""
    monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)
    block = describe_image_pixel_cap(_proc(max_pixels=524288))
    assert block["image_max_pixels"] == 524288, "must record the effect"
    assert block["image_max_pixels_env"] is None
    assert block["image_max_pixels_source"] == "checkpoint"
    assert block["image_pixel_cap_schema"] == 2


def test_env_supplied_cap_is_labelled_env(monkeypatch):
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    block = describe_image_pixel_cap(_proc(max_pixels=524288))
    assert block["image_max_pixels"] == 524288
    assert block["image_max_pixels_env"] == 524288
    assert block["image_max_pixels_source"] == "env"


def test_uncapped_records_the_real_ceiling_not_none(monkeypatch):
    """Full-resolution runs now record 12,845,056 rather than ``None``.

    ``None`` therefore no longer overloads "full resolution" with "unset",
    which is the ambiguity that made schema-1 buckets unreadable.
    """
    monkeypatch.delenv(ENV_IMAGE_MAX_PIXELS, raising=False)
    block = describe_image_pixel_cap(_proc(max_pixels=12845056))
    assert block["image_max_pixels"] == 12845056
    assert block["image_max_pixels_env"] is None


def test_env_and_processor_disagree_reports_processor(monkeypatch):
    """A cap that was requested but never applied must not read as applied."""
    monkeypatch.setenv(ENV_IMAGE_MAX_PIXELS, "524288")
    block = describe_image_pixel_cap(_proc(max_pixels=12845056))
    assert block["image_max_pixels"] == 12845056
    assert block["image_max_pixels_env"] == 524288
    assert block["image_max_pixels_source"] == "checkpoint"
