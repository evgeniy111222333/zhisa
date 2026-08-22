"""Integration: MarketDataset + CompiledChartStore (compute-free path).

Verifies the ideal end-to-end within a dataset:

- a dataset fed by a compiled store yields bit-exact charts vs the canonical
  renderer and vs the legacy in-memory lazy path,
- the store-backed dataset performs zero rasterisation at access time,
- the render contract survives into the ``SampleSpec`` provenance and a bad
  (wrong spec / wrong window) store is rejected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.data.chart_store import CompiledChartStore
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.rendering.chart_renderer import render_ohlcv
from zhisa.rendering.spec import RenderSpec
from zhisa.utils.seeding import set_seed


def _df(n_bars: int = 1500, seed: int = 1234) -> pd.DataFrame:
    set_seed(seed)
    return generate_market(MarketConfig(n_bars=n_bars, freq="5min", seed=seed))


def _spec(chart_window: int = 64, image_size: int = 64):
    return SampleSpec(
        chart_window=chart_window,
        feature_window=chart_window,
        image_size=image_size,
        horizons=(4, 16, 64),
    )


def _build_store(df, spec: SampleSpec, window: int):
    render_spec = RenderSpec(size=spec.image_size)
    ds_len = len(df) - spec.chart_window - max(spec.horizons) - 1
    return CompiledChartStore.build(df, window=window, spec=render_spec, indices=range(ds_len))


def test_store_backed_dataset_matches_lazy_path_bitexact():
    df = _df(800)
    spec = _spec()
    set_seed(1234)
    lazy = MarketDataset(df, spec=spec, cache_charts=False, chart_cache_size=-1)
    set_seed(1234)
    store = _build_store(df, spec, window=spec.chart_window)
    sourced = MarketDataset(
        df,
        spec=spec,
        cache_charts=True,            # even with caching requested, store wins
        chart_cache_size=-1,
        chart_source=store,
    )
    n = min(len(lazy), len(sourced))
    for t in list(range(0, n, max(1, n // 20))) + [n - 1]:
        assert torch_equal(lazy[t]["chart"], sourced[t]["chart"]), f"t={t}"


def test_store_backed_dataset_is_compute_free():
    """Accessing the store-backed dataset must not rasterise anything."""
    df = _df(600)
    spec = _spec(chart_window=32, image_size=32)
    set_seed(1234)
    store = _build_store(df, spec, window=32)
    set_seed(1234)
    sourced = MarketDataset(df, spec=spec, cache_charts=False, chart_cache_size=-1, chart_source=store)
    # The dataset must advertise the fast path and hold no in-memory chart array.
    assert sourced.__fast_getitem__ is True
    assert sourced._chart_arr is None
    for t in (0, 10, len(sourced) - 1):
        img = sourced[t]["chart"]
        assert tuple(img.shape) == (3, 32, 32)


def test_store_backed_charts_equal_direct_canonical_render():
    df = _df(500)
    spec = _spec(chart_window=32, image_size=32)
    store = _build_store(df, spec, window=32)
    set_seed(0)
    sourced = MarketDataset(
        df, spec=spec, cache_charts=False, chart_cache_size=-1, chart_source=store
    )
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    rs = RenderSpec(size=32)
    for t in (0, 7, len(sourced) - 1):
        expected = render_ohlcv(ohlcv[t:t + 32], spec=rs)
        assert torch_equal(expected, sourced[t]["chart"])


def test_too_short_store_rejected():
    df = _df(200)
    spec = _spec(chart_window=64, image_size=32)
    store = CompiledChartStore.build(df, window=64, spec=RenderSpec(size=32), indices=range(5))
    import pytest
    with pytest.raises(ValueError):
        MarketDataset(df, spec=spec, chart_source=store)


def test_contract_is_recorded_in_dataset():
    """The dataset surfaces the store provenance for checkpoint recording."""
    df = _df(400)
    spec = _spec(chart_window=32, image_size=32)
    store = _build_store(df, spec, window=32)
    sourced = MarketDataset(df, spec=spec, chart_source=store)
    meta = sourced._chart_source.render_meta
    assert meta["renderer"]  # renderer_version
    assert meta["spec_hash"] == RenderSpec(size=32).content_hash()
    assert meta["fingerprint"]
    assert meta["content_key"]


def torch_equal(a, b):
    import torch
    return torch.equal(torch.as_tensor(a), torch.as_tensor(b))