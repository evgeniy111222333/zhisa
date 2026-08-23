"""Tests for the order-book depth (bookDepth) preparation channel."""
from __future__ import annotations

import pandas as pd

from zhisa.data.preparation import PrepareConfig, _join_bookdepth
from zhisa.scripts._real_data import futures_context_symbol_slug


def _frame():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame(index=idx)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = 100.0
    return df


def _bd_frame():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"bd_imb_1": [0.1] * 6, "bd_mean_depth_pos1": [4400.0] * 6},
                      index=idx)
    return df


def _cfg(tmp, *, with_bd=True, root=None):
    return PrepareConfig(tsdb_root=tmp, out_root=tmp / "o", symbols=["BTC/USDT", "ETH/USDT"],
                         timeframe="1h", with_futures_context=False,
                         with_bookdepth=with_bd, bookdepth_root=root or (tmp / "bd"))


def test_bookdepth_joins_when_all_present(tmp_path, ):
    bd_root = tmp_path / "bd"
    for sym in ("BTC/USDT", "ETH/USDT"):
        p = bd_root / f"{futures_context_symbol_slug(sym)}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        _bd_frame().to_parquet(p)
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    out, info = _join_bookdepth(per, _cfg(tmp_path, root=bd_root))
    assert not info["dropped"]
    assert "bd_imb_1" in out["BTC/USDT"].columns
    assert out["BTC/USDT"].columns.equals(out["ETH/USDT"].columns)


def test_bookdepth_dropped_for_uniform_schema_when_missing(tmp_path):
    bd_root = tmp_path / "bd"
    p = bd_root / f"{futures_context_symbol_slug('BTC/USDT')}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    _bd_frame().to_parquet(p)  # ETH missing
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    out, info = _join_bookdepth(per, _cfg(tmp_path, root=bd_root))
    assert info["dropped"] is True
    assert info["missing"] == ["ETH/USDT"]
    # uniform: no bd_* columns anywhere
    assert not any(c.startswith("bd_") for c in out["BTC/USDT"].columns)
    assert out["BTC/USDT"].columns.equals(out["ETH/USDT"].columns)


def test_bookdepth_disabled_is_noop(tmp_path):
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    out, info = _join_bookdepth(per, _cfg(tmp_path, with_bd=False))
    assert info["enabled"] is False
    assert "bd_imb_1" not in out["BTC/USDT"].columns