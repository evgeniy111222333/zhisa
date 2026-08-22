"""Tests for the canonical RenderSpec (versioning + content hashing)."""
from __future__ import annotations

import pytest

from zhisa.rendering.spec import (
    DEFAULT_OVERLAYS,
    RENDER_SPEC_VERSION,
    RenderSpec,
    default_render_spec,
)


def test_default_spec_is_valid():
    spec = RenderSpec()
    assert spec.size == 64
    assert spec.supersample >= 1
    assert 0.0 < spec.price_frac <= 1.0
    assert spec.overlays == DEFAULT_OVERLAYS


def test_constructor_validates_bad_values():
    with pytest.raises(ValueError):
        RenderSpec(size=0)
    with pytest.raises(ValueError):
        RenderSpec(supersample=0)
    with pytest.raises(ValueError):
        RenderSpec(price_frac=1.5)
    with pytest.raises(ValueError):
        RenderSpec(red=(2.0, 0.0, 0.0))


def test_content_hash_stable_across_equal_specs():
    a = RenderSpec(size=128, supersample=4)
    b = RenderSpec(size=128, supersample=4)
    assert a.content_hash() == b.content_hash()


def test_content_hash_depends_on_semantics():
    base = RenderSpec(size=128)
    assert base.content_hash() != RenderSpec(size=129).content_hash()
    assert base.content_hash() != RenderSpec(include_volume=False).content_hash()
    assert base.content_hash() != RenderSpec(include_overlays=False).content_hash()
    assert base.content_hash() != RenderSpec(supersample=2).content_hash()
    # colour palette is part of the visual identity
    assert base.content_hash() != RenderSpec(red=(0.1, 0.2, 0.3)).content_hash()


def test_content_hash_ignores_seed_and_overlay_order():
    a = RenderSpec(overlays=((10, (0.2, 0.6, 1.0)), (30, (1.0, 0.6, 0.2))))
    b = RenderSpec(overlays=((30, (1.0, 0.6, 0.2)), (10, (0.2, 0.6, 1.0))))
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() == RenderSpec(seed=999, overlays=a.overlays).content_hash()


def test_version_is_folded_into_hash():
    a = RenderSpec(version=RENDER_SPEC_VERSION)
    b = RenderSpec(version="9.9.9")
    assert a.content_hash() != b.content_hash()


def test_default_render_spec_factory():
    s = default_render_spec(size=32)
    assert s.size == 32
    s2 = default_render_spec(include_volume=False)
    assert s2.include_volume is False
    assert s2.size == 64


def test_meta_roundtrip():
    spec = RenderSpec(size=32, supersample=2, include_volume=False)
    restored = RenderSpec.from_meta(spec.to_meta())
    assert restored == spec
    assert restored.content_hash() == spec.content_hash()