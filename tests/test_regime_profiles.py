"""Tests for named regime-intelligence profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.regime import (
    RegimeIntelligence,
    build_regime_profile_config,
    get_regime_profile,
    list_regime_profiles,
    resolve_regime_profile,
)


def _ohlcv_from_close(close: np.ndarray, *, volume: float | np.ndarray = 100.0, **extra) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    if np.isscalar(volume):
        vol = np.full(close.size, float(volume))
    else:
        vol = np.asarray(volume, dtype=np.float64)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    data = {
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": vol,
    }
    data.update(extra)
    return pd.DataFrame(data, index=idx)


def test_regime_profiles_are_listed_and_resolved_without_btc_lock_in() -> None:
    names = list_regime_profiles()

    assert "default" in names
    assert "btc_intraday" in names
    assert "high_beta_alt" in names
    assert "equity_intraday" in names
    assert resolve_regime_profile(symbol="BTC/USDT").name == "btc_intraday"
    assert resolve_regime_profile(symbol="SOL/USDT").name == "high_beta_alt"
    assert resolve_regime_profile(symbol="AAPL", asset_class="equity").name == "equity_intraday"


def test_build_regime_profile_config_applies_overrides() -> None:
    cfg = build_regime_profile_config(
        "equity_intraday",
        source_timeframe="1m",
        timeframes=("1m", "5m"),
        benchmark_symbol="QQQ",
    )

    assert cfg.source_timeframe == "1m"
    assert cfg.timeframes == ("1m", "5m")
    assert cfg.context.benchmark_symbol == "QQQ"
    assert cfg.context.benchmark_aliases == ("SPY", "QQQ", "IWM")


def test_high_beta_alt_profile_is_more_conservative_than_default() -> None:
    n = 260
    close = np.linspace(100.0, 130.0, n)
    df = _ohlcv_from_close(
        close,
        bid_depth=np.r_[np.full(n - 1, 1000.0), [120.0]],
        ask_depth=np.r_[np.full(n - 1, 1000.0), [260.0]],
        taker_buy_volume=np.r_[np.full(n - 1, 100.0), [20.0]],
        taker_sell_volume=np.r_[np.full(n - 1, 100.0), [180.0]],
        spread_bps=np.r_[np.full(n - 1, 2.0), [12.0]],
    )

    default = RegimeIntelligence(build_regime_profile_config("default", timeframes=("5m", "15m"))).analyze(df)
    alt = RegimeIntelligence(build_regime_profile_config("high_beta_alt", timeframes=("5m", "15m"))).analyze(df)

    assert alt.transition_risk >= default.transition_risk
    assert alt.position_size_multiplier <= default.position_size_multiplier
    assert alt.features["scoring"]["tradeability"]["weak_book"] <= 0.0


def test_profile_lookup_rejects_unknown_names() -> None:
    try:
        get_regime_profile("unknown_profile")
    except KeyError as exc:
        assert "valid profiles" in str(exc)
    else:
        raise AssertionError("unknown profile did not raise")
