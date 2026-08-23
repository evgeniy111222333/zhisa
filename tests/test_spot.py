"""Tests for the dual-market (spot) channels in preparation."""
from __future__ import annotations

import pandas as pd

from zhisa.data.preparation import PrepareConfig, _join_spot_channels
from zhisa.scripts._real_data import futures_context_symbol_slug


def _frame(p0=100.0):
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": p0, "high": p0 * 1.01, "low": p0 * 0.99, "close": p0,
        "volume": 1000.0,
    }, index=idx)


def _spot_frame(sr, tmp, sym):
    root = tmp / "tsdb_spot"
    from zhisa.storage.tsdb import TimeSeriesDB
    from zhisa.storage.schema import Timeframe, SeriesKey
    db = TimeSeriesDB(root)
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": sr, "high": sr * 1.01, "low": sr * 0.99, "close": sr,
        "volume": 2000.0,
    }, index=idx)
    db.ingest(SeriesKey(sym, Timeframe.H1), df)
    return root


def _cfg(tmp, *, with_spot=True, spot_root=None):
    return PrepareConfig(tsdb_root=tmp, out_root=tmp / "o",
                         symbols=["BTC/USDT", "ETH/USDT"], timeframe="1h",
                         with_futures_context=False,
                         with_spot=with_spot,
                         spot_root=spot_root or (tmp / "tsdb_spot"))


def test_spot_channels_added_when_all_present(tmp_path):
    sr = _spot_frame(99.0, tmp_path, "BTC/USDT")
    _spot_frame(120.0, tmp_path, "ETH/USDT")
    per = {"BTC/USDT": _frame(100.0), "ETH/USDT": _frame(120.0)}
    out, info = _join_spot_channels(per, _cfg(tmp_path, spot_root=sr))
    assert not info["dropped"]
    assert "ctx_spot_basis_1h" in out["BTC/USDT"].columns
    # perp=100 vs spot=99 -> basis ~ +0.010, lagged by 1 -> first bar NaN
    b = out["BTC/USDT"]["ctx_spot_basis_1h"]
    assert pd.isna(b.iloc[0])
    assert abs(b.iloc[1] - (100.0 / 99.0 - 1.0)) < 1e-3
    assert out["BTC/USDT"].columns.equals(out["ETH/USDT"].columns)


def test_spot_uniform_drop_when_missing(tmp_path):
    sr = _spot_frame(99.0, tmp_path, "BTC/USDT")  # ETH missing
    per = {"BTC/USDT": _frame(100.0), "ETH/USDT": _frame(120.0)}
    out, info = _join_spot_channels(per, _cfg(tmp_path, spot_root=sr))
    assert info["dropped"] is True
    assert not any("ctx_spot" in c for c in out["BTC/USDT"].columns)
    assert out["BTC/USDT"].columns.equals(out["ETH/USDT"].columns)


def test_spot_disabled_noop(tmp_path):
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    out, info = _join_spot_channels(per, _cfg(tmp_path, with_spot=False))
    assert info["enabled"] is False
    assert "ctx_spot_basis_1h" not in out["BTC/USDT"].columns