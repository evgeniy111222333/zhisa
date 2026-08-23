"""Tests: strict 'all symbols must have context' guard for S1 preparation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from zhisa.data.preparation import PrepareConfig, _merge_context
from zhisa.data.context_merger import CTX_COLUMN_PREFIX


def _frame(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0
    df["high"] = 101.0
    df["low"] = 99.0
    df["close"] = 100.5
    df["volume"] = 1000.0
    return df


def _write_ctx(root: Path, slug: str) -> None:
    p = root / slug / "15m" / "context.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame({"funding_rate": 1e-4, "open_interest": 1e5}, index=idx)
    df.to_parquet(p)


def _cfg(tmp, *, require: bool, with_ctx: bool = True) -> PrepareConfig:
    return PrepareConfig(
        tsdb_root=Path(tmp) / "tsdb",
        out_root=Path(tmp) / "out",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        with_futures_context=with_ctx,
        context_root=Path(tmp) / "ctx" if with_ctx else None,
        context_timeframe="15m",
        require_all_context=require,
    )


def test_guard_raises_when_symbol_lacks_context(tmp_path):
    ctx = tmp_path / "ctx"
    _write_ctx(ctx, "BTCUSDT")  # only BTC has a context parquet
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    with pytest.raises(ValueError, match="require_all_context.*ETH/USDT"):
        _merge_context(per, _cfg(tmp_path, require=True))


def test_guard_passes_when_all_symbols_have_context(tmp_path):
    ctx = tmp_path / "ctx"
    _write_ctx(ctx, "BTCUSDT")
    _write_ctx(ctx, "ETHUSDT")
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    out, info = _merge_context(per, _cfg(tmp_path, require=True))
    for sym in ("BTC/USDT", "ETH/USDT"):
        assert CTX_COLUMN_PREFIX in str(out[sym].attrs.get("context_merge", {}).get("columns_added")) \
            or out[sym].attrs.get("context_merge", {}).get("columns_added")


def test_guard_disabled_defaults_to_lenient(tmp_path):
    ctx = tmp_path / "ctx"
    _write_ctx(ctx, "BTCUSDT")  # ETH lacks context
    per = {"BTC/USDT": _frame(), "ETH/USDT": _frame()}
    out, _ = _merge_context(per, _cfg(tmp_path, require=False))
    # ETH frame stays unchanged (no context) and no error is raised
    assert out["ETH/USDT"].equals(_frame())