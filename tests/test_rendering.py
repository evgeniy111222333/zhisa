"""Tests for chart rendering."""
from __future__ import annotations

import numpy as np
import torch

from zhisa.rendering.chart_renderer import render_chart, render_chart_array
from zhisa.rendering.augmentations import (
    additive_gaussian_noise,
    color_jitter,
    crop_and_resize,
    horizontal_mirror,
)


def test_render_basic_shape(small_market):
    img = render_chart(small_market, size=32)
    assert img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert (img >= 0.0).all() and (img <= 1.0).all()


def test_render_short_window(small_market):
    img = render_chart(small_market.iloc[:8], size=16)
    assert img.shape == (3, 16, 16)


def test_render_empty_window():
    import pandas as pd
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    img = render_chart(empty, size=16)
    assert img.shape == (3, 16, 16)


def test_fast_array_renderer_matches_dataframe_entrypoint(small_market, monkeypatch):
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    frame = small_market.iloc[:64]
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy()
    assert torch.equal(
        render_chart(frame, size=64),
        render_chart_array(values, size=64),
    )


def test_color_jitter_range():
    img = torch.full((3, 16, 16), 0.5)
    out = color_jitter(img, strength=0.05)
    assert out.shape == img.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_horizontal_mirror_changes_pixels():
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.random((3, 16, 16)).astype(np.float32))
    mirrored = horizontal_mirror(img, p=1.0)
    assert not torch.allclose(mirrored, img)
    # Twice is identity
    assert torch.allclose(horizontal_mirror(mirrored, p=1.0), img)


def test_crop_and_resize_shape():
    img = torch.rand(3, 32, 32)
    out = crop_and_resize(img, crop_frac=0.5, size=32)
    assert out.shape == (3, 32, 32)


def test_additive_gaussian_noise_clamped():
    img = torch.zeros(3, 8, 8)
    out = additive_gaussian_noise(img, std=0.5)
    assert (out >= 0.0).all() and (out <= 1.0).all()
