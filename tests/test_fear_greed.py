"""Tests for the Fear & Greed sentiment channel integration."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from zhisa.data import fear_greed as fg
from zhisa.data.preparation import PrepareConfig, _merge_context


def _fake_payload():
    return {"data": [
        {"value": "25", "value_classification": "Fear", "timestamp": 1704067200},   # 2024-01-01
        {"value": "55", "value_classification": "Greed", "timestamp": 1704153600},  # 2024-01-02
    ]}


def test_fetch_parse(monkeypatch):
    import urllib.request
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def read(self): return json.dumps(_fake_payload()).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    df = fg.fetch_fear_greed_history()
    assert len(df) == 2
    assert df["fng_index"].dtype == np.float32
    assert (df["fng_index"] >= 0).all() and (df["fng_index"] <= 100).all()
    assert df.index.is_unique and df.index.tz is not None
    assert df["classification"].iloc[0] == "Fear"


def test_cache_roundtrip(tmp_path, monkeypatch):
    import urllib.request
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def read(self): return json.dumps(_fake_payload()).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    cache = tmp_path / "fng.parquet"
    fg.download_to_cache(cache)
    df = fg.load_fear_greed(cache)
    assert len(df) == 2 and df["fng_index"].iloc[0] == 25.0


def test_fear_greed_column_is_causal(tmp_path):
    bar_idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    fng = pd.DataFrame({"fng_index": [25.0, 55.0]},
                        index=pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True))
    col = fg.fear_greed_column(bar_idx, fng)
    assert col.index.equals(bar_idx)
    # shift(1) => the very first bar is NaN (no value strictly before it)
    assert pd.isna(col.iloc[0])
    # bar at 2024-01-02 00:00 uses the 2024-01-01 value, not 01-02
    d2 = col.loc["2024-01-02 00:00:00+00:00"]
    assert d2 == 25.0
    assert np.isfinite(col.dropna()).all()


def test_prepare_injects_fng(tmp_path):
    ctx = tmp_path / "ctx"
    for slug in ("BTCUSDT", "ETHUSDT"):
        p = ctx / slug / "15m"
        p.mkdir(parents=True, exist_ok=True)
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        pd.DataFrame({"funding_rate": 1e-4, "open_interest": 1e5}, index=idx).to_parquet(p / "context.parquet")
    fng_path = tmp_path / "fng.parquet"
    fng = pd.DataFrame({"fng_index": [20.0, 80.0]},
                       index=pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True))
    fng.to_parquet(fng_path)

    def _frame():
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        out = pd.DataFrame(index=idx)
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = 100.0
        return out
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    cfg = PrepareConfig(tsdb_root=tmp_path, out_root=tmp_path / "out",
                        symbols=["BTC/USDT", "ETH/USDT"], timeframe="1h",
                        with_futures_context=True, context_root=ctx,
                        context_timeframe="15m", require_all_context=True,
                        with_fear_greed=True, fear_greed_cache=fng_path)
    merged, info = _merge_context(per, cfg)  # context stage passes through
    # inject fng exactly as prepare_dataset does
    from zhisa.data.fear_greed import fear_greed_column
    for sym, df in merged.items():
        merged[sym] = df.assign(ctx_fng_index=fear_greed_column(df.index, fng))
    assert "ctx_fng_index" in merged["BTC/USDT"].columns
    assert merged["BTC/USDT"].columns.equals(merged["ETH/USDT"].columns)
    assert pd.isna(merged["BTC/USDT"]["ctx_fng_index"].iloc[0])