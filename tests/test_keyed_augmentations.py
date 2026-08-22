"""Tests for deterministic keyed augmentations."""
from __future__ import annotations

import torch

from zhisa.rendering.augmentations import (
    KeyedAugmentor,
    key_from_string,
    transform_seed,
)
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.chart_renderer import render_ohlcv
from zhisa.rendering.spec import RenderSpec

import numpy as np


def test_key_from_string_is_stable_and_spread():
    assert key_from_string("a") == key_from_string("a")
    assert key_from_string("a") != key_from_string("b")
    assert 0 < key_from_string("") < 2**64


def test_same_key_same_output_different_run():
    img = render_ohlcv(golden_fixture(96), spec=RenderSpec(size=64))
    aug = KeyedAugmentor(transforms=("mirror", "color_jitter", "crop", "gaussian_noise"))
    a = aug.apply(img, key="sample-42")
    b = aug.apply(img, key="sample-42")
    assert torch.equal(a, b)


def test_different_key_gives_different_output():
    img = render_ohlcv(golden_fixture(96), spec=RenderSpec(size=64))
    aug = KeyedAugmentor(transforms=("mirror", "color_jitter", "crop", "gaussian_noise"))
    a = aug.apply(img, key="k1")
    b = aug.apply(img, key="k2")
    # With 4 stochastic transforms, two keys colliding into identical bytes is
    # effectively impossible; assert they are not the same tensor.
    assert not torch.equal(a, b)


def test_output_bounds_kept():
    img = render_ohlcv(golden_fixture(96), spec=RenderSpec(size=64))
    aug = KeyedAugmentor(transforms=("color_jitter", "gaussian_noise"))
    out = aug.apply(img, key="bounds")
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_identity_when_no_transforms():
    img = render_ohlcv(golden_fixture(96), spec=RenderSpec(size=64))
    aug = KeyedAugmentor(transforms=())
    assert torch.equal(aug.apply(img, key="x"), img)


def test_mirror_deterministic_given_flag():
    img = torch.rand(3, 8, 8)
    from zhisa.rendering.augmentations import horizontal_mirror_det
    flipped = horizontal_mirror_det(img, do_flip=True)
    assert torch.equal(horizontal_mirror_det(flipped, do_flip=True), img)


def test_meta_roundtrip():
    aug = KeyedAugmentor(transforms=("mirror", "crop"), strength=0.1)
    restored = KeyedAugmentor.from_meta(aug.to_meta())
    assert restored.transforms == aug.transforms
    assert restored.strength == aug.strength
    img = render_ohlcv(golden_fixture(96), spec=RenderSpec(size=64))
    assert torch.equal(aug.apply(img, "meta-key"), restored.apply(img, "meta-key"))


def test_transform_seed_variants_are_independent():
    r1 = transform_seed("s", variant="stage-1")
    r2 = transform_seed("s", variant="stage-2")
    assert int(r1.integers(0, 10**9)) != int(r2.integers(0, 10**9))


def test_last_params_recorded():
    img = render_ohlcv(golden_fixture(96), spec=RenderSpec(size=64))
    aug = KeyedAugmentor(transforms=("mirror", "color_jitter"))
    _ = aug.apply(img, key="record")
    params = aug.apply_fixed_defaults("record")
    assert "mirror" in params
    assert "jitter_delta" in params
    assert len(params["jitter_delta"]) == 3
    assert all(isinstance(v, float) for v in params["jitter_delta"])


def test_invalid_transform_rejected():
    import pytest
    with pytest.raises(ValueError):
        KeyedAugmentor(transforms=("bogus",))