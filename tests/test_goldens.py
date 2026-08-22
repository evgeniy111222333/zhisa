"""Tests for the golden-image registry (deterministic QA of the renderer)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from zhisa.rendering.goldens import (
    GoldenError,
    assert_matches_golden,
    compute_digest,
    digest_of_saved,
    golden_fixture,
    has_golden,
    store_golden,
)
from zhisa.rendering.spec import RenderSpec


def test_fixture_is_deterministic():
    a = golden_fixture(96)
    b = golden_fixture(96)
    assert np.array_equal(a, b)
    assert a.dtype == np.float64
    assert a.shape == (96, 5)


def test_compute_digest_stable():
    fx = golden_fixture(96)
    spec = RenderSpec(size=64)
    assert compute_digest(fx, spec) == compute_digest(fx, spec)


def test_store_and_verify(tmp_path):
    fx = golden_fixture(96)
    spec = RenderSpec(size=64)
    meta = store_golden(tmp_path, "btc_like", fx, spec)
    assert has_golden(tmp_path, "btc_like", spec)
    assert meta["image_sha256"] == compute_digest(fx, spec)
    # A fresh render must match bit-exactly.
    assert_matches_golden(tmp_path, "btc_like", fx, spec)


def test_store_is_idempotent(tmp_path):
    fx = golden_fixture(96)
    spec = RenderSpec(size=64)
    store_golden(tmp_path, "iod", fx, spec)
    first = (tmp_path / "iod" / _fp_dir(spec) / "image.sha256").read_text()
    store_golden(tmp_path, "iod", fx, spec)  # no force -> untouched
    second = (tmp_path / "iod" / _fp_dir(spec) / "image.sha256").read_text()
    assert first == second


def test_mismatch_raises(tmp_path):
    fx = golden_fixture(96)
    spec_stored = RenderSpec(size=64)
    store_golden(tmp_path, "mm", fx, spec_stored)
    # A different window (different length) -> different pixels -> must raise.
    other = golden_fixture(64)
    with pytest.raises(GoldenError):
        assert_matches_golden(tmp_path, "mm", other, spec_stored)
    # A different spec is a different fingerprint -> no golden yet.
    with pytest.raises(GoldenError):
        assert_matches_golden(tmp_path, "mm", fx, RenderSpec(size=32))


def test_missing_golden_raises(tmp_path):
    fx = golden_fixture(96)
    with pytest.raises(GoldenError):
        assert_matches_golden(tmp_path, "nope", fx, RenderSpec(size=64))


def test_corruption_detected(tmp_path):
    fx = golden_fixture(96)
    spec = RenderSpec(size=64)
    store_golden(tmp_path, "corrupt", fx, spec)
    img_path = tmp_path / "corrupt" / _fp_dir(spec) / "image.npy"
    arr = np.load(img_path)
    arr[3, 3] = 0.99  # corrupt one pixel on disk
    np.save(img_path, arr)
    with pytest.raises(GoldenError):
        assert_matches_golden(tmp_path, "corrupt", fx, spec)


def test_digest_of_saved_matches_meta(tmp_path):
    fx = golden_fixture(96)
    spec = RenderSpec(size=64)
    meta = store_golden(tmp_path, "digest", fx, spec)
    assert digest_of_saved(tmp_path / "digest" / _fp_dir(spec)) == meta["image_sha256"]


def _fp_dir(spec: RenderSpec) -> str:
    from zhisa.rendering.chart_renderer import render_fingerprint
    return render_fingerprint(spec)


def test_meta_is_json_and_has_renderer_version(tmp_path):
    fx = golden_fixture(96)
    spec = RenderSpec(size=64)
    store_golden(tmp_path, "v", fx, spec)
    meta_path = tmp_path / "v" / _fp_dir(spec) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "renderer_version" in meta
    assert "fingerprint" in meta
    assert meta["shape"] == [64, 64, 3]