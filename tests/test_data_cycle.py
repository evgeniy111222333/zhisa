"""Tests for the data-update cycle (stale detection, incremental refresh, block hash)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from zhisa.data.chart_store import CompiledChartStore, covered_prefix_hash
from zhisa.data.data_cycle import refresh_symbol_store, update_prepared_charts
from zhisa.data.dataset import SampleSpec
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec


def _frame(n_bars: int, seed: int = 0) -> pd.DataFrame:
    fx = golden_fixture(n_bars)
    rule = pd.date_range("2024-01-01", periods=n_bars, freq="5min")
    return pd.DataFrame(
        {
            "open": fx[:, 0], "high": fx[:, 1], "low": fx[:, 2],
            "close": fx[:, 3], "volume": fx[:, 4], "timestamp": rule,
        }
    ).set_index("timestamp")


def _grow(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return a frame that *appends* ``k`` new, distinct bars after ``df``."""
    fx = golden_fixture(max(k, 8))[:k] * 1.01  # distinct continuation values
    rule = pd.date_range(
        df.index[-1].to_pydatetime() + pd.Timedelta("5min"), periods=k, freq="5min"
    )
    tail = pd.DataFrame(
        {
            "open": fx[:, 0], "high": fx[:, 1], "low": fx[:, 2],
            "close": fx[:, 3], "volume": fx[:, 4], "timestamp": rule,
        }
    ).set_index("timestamp")
    assert not df.index.intersection(tail.index).size
    return pd.concat([df, tail])


def _spec(window: int = 32, image: int = 32):
    return SampleSpec(chart_window=window, feature_window=window, image_size=image, horizons=(4, 16, 64))


def test_covered_prefix_hash_is_append_stable():
    df1 = _frame(300)
    grown = _grow(df1, 80)
    assert len(grown) == 380
    assert covered_prefix_hash(df1, 32, 149) == covered_prefix_hash(grown, 32, 149)
    tampered = grown.copy()
    tampered.iloc[5, 2] += 1.0
    assert covered_prefix_hash(grown, 32, 149) != covered_prefix_hash(tampered, 32, 149)


def test_stale_detection_and_assert(tmp_path):
    df = _frame(300)
    spec = _spec()
    rs = RenderSpec(size=32)
    store, _, _ = refresh_symbol_store(df, spec, tmp_path, render_spec=rs)
    store.assert_fresh(df, 32, rs, n=store.meta["n_images"])

    # same frame, same n -> not stale
    assert store.stale_for(df, 32, rs, n=int(store.meta["n_images"])) is False
    # grown frame, bigger desired n -> stale (growth)
    grown = _grow(df, 60)
    assert store.stale_for(grown, 32, rs, n=int(store.meta["n_images"])) is False  # prefix ok
    bigger_n = int(store.meta["n_images"]) + 60
    assert store.stale_for(grown, 32, rs, n=bigger_n) is True
    with pytest.raises(RuntimeError):
        store.assert_fresh(grown, 32, rs, n=bigger_n)
    # tampered prefix -> stale even at same n
    tampered = df.copy()
    tampered.iloc[4, 1] *= 1.5
    assert store.stale_for(tampered, 32, rs, n=int(store.meta["n_images"])) is True


def test_refresh_incremental_reuses_prefix(tmp_path):
    df = _frame(400)
    spec = _spec()
    rs = RenderSpec(size=32)
    store, rf0, _ = refresh_symbol_store(df, spec, tmp_path, render_spec=rs)
    n0 = int(store.meta["n_images"])          # 400 - 32 - 64 - 1 = 303
    assert rf0.stale is True                  # built just now (needs-build semantics)
    assert rf0.n_images == n0

    grown = _grow(df, 100)                    # 500 bars -> n1 = 403
    new_store, rf, stats = refresh_symbol_store(grown, spec, tmp_path, render_spec=rs)
    assert stats.reused_prefix_rows == n0, "prefix should be reused, not re-rendered"
    assert stats.rendered_rows == (int(new_store.meta["n_images"]) - n0)
    assert stats.reused_artifact is False
    fresh = CompiledChartStore.build(
        grown, window=32, spec=rs, indices=range(int(new_store.meta["n_images"]))
    )
    assert new_store.render_checksum(full=True) == fresh.render_checksum(full=True)


def test_refresh_no_change_is_cache_hit(tmp_path):
    df = _frame(300)
    spec = _spec()
    rs = RenderSpec(size=32)
    _, _, _ = refresh_symbol_store(df, spec, tmp_path, render_spec=rs)
    store2, rf, stats = refresh_symbol_store(df, spec, tmp_path, render_spec=rs)
    assert stats.reused_artifact is True
    assert rf.stale is False
    assert stats.rendered_rows == 0


def test_update_prepared_charts_report(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir(parents=True)
    (root / "symbols").mkdir(parents=True)
    a = _frame(400)
    a.name = "BTC_USDT"
    a.to_parquet(root / "symbols" / "BTC_USDT.parquet")
    b = _frame(350)
    b.to_parquet(root / "symbols" / "ETH_USDT.parquet")

    charts = tmp_path / "charts"
    spec = _spec(32, 32)
    report = update_prepared_charts(root, charts, spec, workers=1, chunk=60)
    assert report.total_symbols == 2
    assert report.stale_count == 2  # both were newly built
    for r in report.refreshed:
        assert r.n_images > 0
        assert r.content_key
    # second pass -> nothing stale (exact cache hits)
    report2 = update_prepared_charts(root, charts, spec, workers=1, chunk=60)
    assert report2.stale_count == 0
    # single-symbol restrict works
    report3 = update_prepared_charts(root, charts, spec, symbols=["ETH_USDT"])
    assert report3.total_symbols == 1


def test_non_contiguous_store_freshness_not_comparable():
    df = _frame(200)
    rs = RenderSpec(size=32)
    store = CompiledChartStore.build(df, window=32, spec=rs, indices=[5, 10, 15])
    with pytest.raises(NotImplementedError):
        store.stale_for(df, 32, rs)