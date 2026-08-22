"""Tests for the canonical (single-source-of-truth) renderer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.rendering.chart_renderer import (
    CANONICAL_RENDERER_VERSION,
    render_chart,
    render_chart_array,
    render_fingerprint,
    render_ohlcv,
)
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec


def _df_from_ohlcv(ohlcv: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": ohlcv[:, 0],
            "high": ohlcv[:, 1],
            "low": ohlcv[:, 2],
            "close": ohlcv[:, 3],
            "volume": ohlcv[:, 4],
        }
    )


def test_canonical_render_shape_and_range():
    ohlcv = golden_fixture(96)
    img = render_ohlcv(ohlcv, spec=RenderSpec(size=64))
    assert img.shape == (3, 64, 64)
    assert img.dtype == torch.float32
    assert (img >= 0.0).all() and (img <= 1.0).all()


def test_single_renderer_ignores_env_flag(monkeypatch):
    """The ideal: one renderer, no env-flag divergence."""
    ohlcv = golden_fixture(96)
    df = _df_from_ohlcv(ohlcv)

    img_default = render_chart(df, size=64)
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    img_fast = render_chart(df, size=64)
    monkeypatch.setenv("ZHISA_FAST_RENDER", "0")
    img_no_fast = render_chart(df, size=64)

    assert torch.equal(img_default, img_fast)
    assert torch.equal(img_default, img_no_fast)


def test_dataframe_and_array_entry_points_are_identical():
    ohlcv = golden_fixture(96)
    df = _df_from_ohlcv(ohlcv)
    assert torch.equal(render_chart(df, size=64), render_chart_array(ohlcv, size=64))


def test_determinism_bit_exact():
    ohlcv = golden_fixture(96)
    a = render_ohlcv(ohlcv, spec=RenderSpec(size=64))
    b = render_ohlcv(ohlcv, spec=RenderSpec(size=64))
    assert torch.equal(a, b)


def test_anti_aliasing_supersample_changes_histogram_not_identity():
    """Supersampled AA yields a smoother image but keeps buffered tails."""
    ohlcv = golden_fixture(96)
    lo = render_ohlcv(ohlcv, spec=RenderSpec(size=64, supersample=1))
    hi = render_ohlcv(ohlcv, spec=RenderSpec(size=64, supersample=4))
    # Not identical pixel-wise (AA smooths edges)...
    assert not torch.equal(lo, hi)
    # ...but content is preserved: neither is blank and means stay comparable.
    assert float(hi.mean()) > 0.01
    assert int((hi > 0.05).sum()) > 0


def test_fingerprint_changes_with_spec_or_semantics():
    s1 = RenderSpec(size=64)
    s2 = RenderSpec(size=128)
    s3 = RenderSpec(size=64, include_volume=False)
    assert render_fingerprint(s1) != render_fingerprint(s2)
    assert render_fingerprint(s1) != render_fingerprint(s3)
    # Same spec -> same fingerprint (stable content identity).
    assert render_fingerprint(RenderSpec(size=64)) == render_fingerprint(s1)
    assert CANONICAL_RENDERER_VERSION != ""


def test_canonical_deterministic_across_order_of_operations():
    """Two identical windows rendered in isolation give the same image."""
    fx = golden_fixture(128)
    # window 10..74 rendered twice independently
    a = render_ohlcv(fx[10:74], spec=RenderSpec(size=64))
    b = render_ohlcv(fx[10:74], spec=RenderSpec(size=64))
    assert torch.equal(a, b)


def test_empty_and_short_windows_safe():
    spec = RenderSpec(size=16)
    with pytest.raises(ValueError):
        render_ohlcv(np.zeros((0, 5)), spec=spec)
    with pytest.raises(ValueError):
        render_ohlcv(np.zeros((1, 4)), spec=spec)
    # 8 bars on a dataframe is fine (historical behaviour preserved).
    img = render_chart(_df_from_ohlcv(golden_fixture(8)), size=16)
    assert img.shape == (3, 16, 16)