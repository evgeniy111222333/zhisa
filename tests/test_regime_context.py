"""Tests for derivatives crowding and cross-asset regime context."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from zhisa.regime import (
    CorrelationState,
    CrowdingState,
    MacroRegime,
    MarketContextAnalyzer,
    MarketContextConfig,
    MarketContextReport,
    RegimeFeatureVectorizer,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RiskMode,
)


def _ohlcv_from_close(close: np.ndarray, **extra) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    data = {
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": np.full(close.size, 100.0),
    }
    data.update(extra)
    return pd.DataFrame(data, index=idx)


def test_market_context_analyzer_detects_crowded_derivatives_state() -> None:
    n = 160
    close = np.linspace(100.0, 130.0, n)
    funding = np.r_[np.full(n - 1, 0.00005), [0.0012]]
    oi = np.r_[np.linspace(1000.0, 1050.0, n - 12), np.linspace(1050.0, 1300.0, 12)]
    long_short = np.r_[np.full(n - 1, 1.05), [1.9]]
    liquidation = np.r_[np.full(n - 1, 10.0), [80.0]]
    df = _ohlcv_from_close(
        close,
        funding=funding,
        open_interest=oi,
        long_short_ratio=long_short,
        liquidation_volume=liquidation,
    )

    report = MarketContextAnalyzer().analyze(df, symbol="BTC/USDT")

    assert report.crowding.direction == "long_crowded"
    assert report.crowding.crowding_score > 0.65
    assert "crowded_long_funding" in report.crowding.flags
    assert "open_interest_fast_change" in report.crowding.flags
    assert "liquidation_spike" in report.crowding.flags
    assert any("crowding" in w for w in report.warnings)
    json.dumps(report.to_dict())


def test_market_context_analyzer_detects_benchmark_led_cross_asset_regime() -> None:
    rng = np.random.default_rng(7)
    n = 180
    lag = 3
    btc_ret = 0.002 * np.sin(np.arange(n) / 5) + rng.normal(0.0, 0.0002, n)
    alt_ret = np.r_[np.zeros(lag), btc_ret[:-lag]] + rng.normal(0.0, 0.00015, n)
    eth_ret = np.r_[np.zeros(lag), btc_ret[:-lag]] * 0.8 + rng.normal(0.0, 0.00015, n)
    btc = 100.0 * np.exp(np.cumsum(btc_ret))
    alt = 20.0 * np.exp(np.cumsum(alt_ret))
    eth = 50.0 * np.exp(np.cumsum(eth_ret))
    assets = {
        "BTC/USDT": _ohlcv_from_close(btc),
        "ALT/USDT": _ohlcv_from_close(alt),
        "ETH/USDT": _ohlcv_from_close(eth),
    }

    report = MarketContextAnalyzer(MarketContextConfig(lead_lag_bars=lag)).analyze(
        assets["ALT/USDT"],
        symbol="ALT/USDT",
        assets=assets,
        benchmark_symbol="BTC/USDT",
    )

    assert report.correlation.regime == "benchmark_led"
    assert report.correlation.leader_symbol == "BTC/USDT"
    assert report.correlation.leader_lead_score > 0.12
    assert report.correlation.n_assets == 3


def test_market_context_analysis_is_causal_no_lookahead() -> None:
    n = 140
    close = np.linspace(100.0, 110.0, n)
    funding = np.full(n, 0.00005)
    funding[-1] = 0.005
    df = _ohlcv_from_close(close, funding=funding)
    analyzer = MarketContextAnalyzer()

    full_at_t = analyzer.analyze(df, t=90)
    truncated = analyzer.analyze(df.iloc[:91])

    assert full_at_t.to_dict() == truncated.to_dict()
    assert full_at_t.crowding.crowding_score < 0.65


def test_regime_intelligence_uses_market_context_for_risk_and_playbooks() -> None:
    n = 260
    x = np.linspace(0, 1, n)
    close = 100.0 * np.exp(0.30 * x)
    funding = np.r_[np.full(n - 1, 0.00005), [0.0012]]
    oi = np.r_[np.linspace(1000.0, 1050.0, n - 12), np.linspace(1050.0, 1300.0, 12)]
    long_short = np.r_[np.full(n - 1, 1.05), [1.9]]
    df = _ohlcv_from_close(
        close,
        funding=funding,
        open_interest=oi,
        long_short_ratio=long_short,
    )
    analyzer = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m")))

    report = analyzer.analyze(df, symbol="BTC/USDT")

    assert report.primary_regime == MacroRegime.BULL_TREND.value
    assert report.risk_mode == RiskMode.REDUCED.value
    assert report.position_size_multiplier <= 0.5
    assert "crowded_long_chase" in report.blocked_playbooks
    assert "pullback_only_long" in report.allowed_playbooks
    assert report.features["market_context"]["crowding"]["direction"] == "long_crowded"
    assert any("crowding" in x for x in report.explanation["why"])
    assert any("crowding" in x for x in report.explanation["danger"])


def test_regime_intelligence_accepts_precomputed_market_context() -> None:
    close = np.linspace(100.0, 120.0, 180)
    supplied = MarketContextReport(
        crowding=CrowdingState(crowding_score=0.8, direction="long_crowded", flags=["crowded_long_funding"]),
        correlation=CorrelationState(regime="fragmented", avg_correlation=0.1, n_assets=4),
        warnings=["synthetic context"],
    )
    report = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(
        _ohlcv_from_close(close),
        extra_context={"market_context": supplied.to_dict()},
    )

    assert report.features["market_context"]["correlation"]["regime"] == "fragmented"
    assert "market_beta_chase" in report.blocked_playbooks
    assert any("fragmented" in x for x in report.explanation["danger"])


def test_regime_vectorizer_includes_market_context_features() -> None:
    close = np.linspace(100.0, 120.0, 180)
    supplied = MarketContextReport(
        crowding=CrowdingState(
            funding=0.001,
            crowding_score=0.8,
            direction="long_crowded",
            flags=["crowded_long_funding"],
        ),
        correlation=CorrelationState(
            regime="benchmark_led",
            avg_correlation=0.55,
            leader_lead_score=0.3,
            n_assets=4,
        ),
    )
    report = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(
        _ohlcv_from_close(close),
        extra_context={"market_context": supplied},
    )
    vectorizer = RegimeFeatureVectorizer()
    vec = vectorizer.transform(report)
    names = vectorizer.feature_names

    assert vec.shape == (vectorizer.dim,)
    assert vec[names.index("context.crowding.crowding_score")] == 0.8
    assert vec[names.index("context.correlation.leader_lead_score")] == 0.3
    assert vec[names.index("crowding_direction.long_crowded")] == 1.0
    assert vec[names.index("correlation_regime.benchmark_led")] == 1.0


def test_market_context_analyzer_supports_non_btc_benchmark() -> None:
    rng = np.random.default_rng(11)
    n = 170
    lag = 2
    spy_ret = 0.0015 * np.sin(np.arange(n) / 6) + rng.normal(0.0, 0.00015, n)
    stock_ret = np.r_[np.zeros(lag), spy_ret[:-lag]] + rng.normal(0.0, 0.0001, n)
    spy = 400.0 * np.exp(np.cumsum(spy_ret))
    stock = 80.0 * np.exp(np.cumsum(stock_ret))
    assets = {
        "SPY": _ohlcv_from_close(spy),
        "AAPL": _ohlcv_from_close(stock),
    }

    report = MarketContextAnalyzer(
        MarketContextConfig(benchmark_symbol="SPY", benchmark_aliases=("SPY",), lead_lag_bars=lag)
    ).analyze(assets["AAPL"], symbol="AAPL", assets=assets, benchmark_symbol="SPY")

    assert report.correlation.regime == "benchmark_led"
    assert report.correlation.benchmark_symbol == "SPY"
    assert report.correlation.leader_symbol == "SPY"
