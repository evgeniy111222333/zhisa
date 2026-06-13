"""Tests for trend maturity and liquidity/value structure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.regime import (
    MarketStructureAnalyzer,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    StructureConfig,
)


def _ohlcv_from_close(close: np.ndarray, *, volume: float | np.ndarray = 100.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    if np.isscalar(volume):
        vol = np.full(close.size, float(volume))
    else:
        vol = np.asarray(volume, dtype=np.float64)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }, index=idx)


def test_structure_detects_late_or_exhausted_trend() -> None:
    n = 220
    x = np.linspace(0, 1, n)
    close = 100.0 * np.exp(0.55 * x)
    volume = np.full(n, 100.0)
    volume[-1] = 450.0
    df = _ohlcv_from_close(close, volume=volume)
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] * 1.04
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-1] * 0.995

    report = MarketStructureAnalyzer().analyze(df)

    assert report.trend.direction == "up"
    assert report.trend.phase in {"late", "exhausted"}
    assert report.trend.maturity_score > 0.6
    assert "late_trend" in report.trend.flags


def test_structure_builds_liquidity_and_value_map() -> None:
    rng = np.random.default_rng(5)
    close = 100.0 + np.sin(np.arange(180) / 8) * 2.0 + rng.normal(0, 0.1, 180)
    volume = 100.0 + 50.0 * np.exp(-((close - 100.0) ** 2) / 2.0)
    df = _ohlcv_from_close(close, volume=volume)

    report = MarketStructureAnalyzer(StructureConfig(lookback=120)).analyze(df)
    liq = report.liquidity

    assert liq.value_area_low < liq.value_area_high
    assert liq.point_of_control >= liq.value_area_low
    assert liq.point_of_control <= liq.value_area_high
    assert liq.upper_levels or liq.lower_levels
    assert liq.nearest_level is not None


def test_structure_analysis_is_causal_no_lookahead() -> None:
    close = np.linspace(100.0, 120.0, 160)
    close[-1] = 80.0
    df = _ohlcv_from_close(close)
    analyzer = MarketStructureAnalyzer()

    full_at_t = analyzer.analyze(df, t=100)
    truncated = analyzer.analyze(df.iloc[:101])

    assert full_at_t.to_dict() == truncated.to_dict()


def test_regime_intelligence_uses_trend_phase_and_liquidity_context() -> None:
    n = 260
    x = np.linspace(0, 1, n)
    close = 100.0 * np.exp(0.50 * x)
    volume = np.full(n, 100.0)
    volume[-1] = 500.0
    df = _ohlcv_from_close(close, volume=volume)
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] * 1.05
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-1] * 0.995

    report = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(df)

    assert report.trend_phase in {"late", "exhausted"}
    assert "market_structure" in report.features
    assert "pullback_to_value_only" in report.allowed_playbooks
    assert "late_trend_chase" in report.blocked_playbooks
    assert any("trend_phase" in x for x in report.explanation["why"])
