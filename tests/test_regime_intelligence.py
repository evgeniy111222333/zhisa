"""Tests for Market Regime Intelligence."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from zhisa.regime import (
    MacroRegime,
    MesoRegime,
    MicroRegime,
    RegimeClassificationConfig,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RiskPostureConfig,
    RiskMode,
    TradeabilityScoringConfig,
    compute_regime_features,
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


def _analyzer() -> RegimeIntelligence:
    return RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m", "1h")))


def test_bull_trend_report_allows_long_playbooks() -> None:
    x = np.linspace(0, 1, 420)
    close = 100.0 * np.exp(0.35 * x) + 0.15 * np.sin(np.arange(420) / 7)
    report = _analyzer().analyze(_ohlcv_from_close(close), symbol="BTC/USDT")

    assert report.primary_regime == MacroRegime.BULL_TREND.value
    assert "trend_pullback_long" in report.allowed_playbooks
    assert "blind_mean_reversion_short" in report.blocked_playbooks
    assert report.tradeability_score > 0.45
    assert 0.0 <= report.confidence <= 1.0
    assert "trend_score" in report.features["aggregate"]
    scoring = report.features["scoring"]
    assert scoring["transition_risk"]["score"] == report.transition_risk
    assert scoring["tradeability"]["score"] == report.tradeability_score
    assert "constructive_meso" in scoring["tradeability"]


def test_high_vol_crash_goes_defensive() -> None:
    pre = np.linspace(120.0, 125.0, 180)
    crash = np.linspace(124.0, 82.0, 60)
    chop = 82.0 + 2.0 * np.sin(np.arange(60) / 2)
    close = np.r_[pre, crash, chop]
    volume = np.r_[np.full(180, 100.0), np.full(60, 450.0), np.full(60, 300.0)]
    report = _analyzer().analyze(_ohlcv_from_close(close, volume=volume))

    assert report.primary_regime == MacroRegime.HIGH_VOL_CRASH.value
    assert report.risk_mode == RiskMode.DEFENSIVE.value
    assert report.position_size_multiplier <= 0.35
    assert "blind_dip_buy" in report.blocked_playbooks
    assert report.transition_risk > 0.2


def test_compression_blocks_large_pre_breakout_positions() -> None:
    rng = np.random.default_rng(0)
    early = 100.0 + np.cumsum(rng.normal(0.0, 0.35, 220))
    late = np.full(120, early[-1]) + 0.03 * np.sin(np.arange(120) / 3)
    close = np.r_[early, late]
    report = _analyzer().analyze(_ohlcv_from_close(close))

    assert report.secondary_regime in {MesoRegime.COMPRESSION.value, MesoRegime.CHOP.value}
    if report.secondary_regime == MesoRegime.COMPRESSION.value:
        assert "volatility_expansion_wait" in report.allowed_playbooks
        assert "large_pre_breakout_position" in report.blocked_playbooks
    assert report.position_size_multiplier <= 0.75


def test_liquidity_sweep_sets_stop_run_context() -> None:
    close = 100.0 + np.sin(np.arange(180) / 8)
    df = _ohlcv_from_close(close)
    prior_high = float(df["high"].iloc[-40:-1].max())
    df.iloc[-1, df.columns.get_loc("high")] = prior_high * 1.03
    df.iloc[-1, df.columns.get_loc("close")] = prior_high * 0.995
    report = _analyzer().analyze(df)

    assert report.micro_regime == MicroRegime.STOP_RUN.value
    assert report.secondary_regime == MesoRegime.FAILED_BREAKOUT.value
    assert "liquidity_sweep_reversal" in report.allowed_playbooks
    assert any("liquidity sweep" in x for x in report.explanation["danger"])


def test_regime_analysis_is_causal_no_lookahead() -> None:
    rng = np.random.default_rng(123)
    close = 100.0 + np.cumsum(rng.normal(0.02, 0.4, 360))
    df = _ohlcv_from_close(close)
    analyzer = _analyzer()

    full_at_t = analyzer.analyze(df, t=220)
    truncated = analyzer.analyze(df.iloc[:221])

    assert full_at_t.primary_regime == truncated.primary_regime
    assert full_at_t.secondary_regime == truncated.secondary_regime
    assert full_at_t.micro_regime == truncated.micro_regime
    assert full_at_t.features["aggregate"] == truncated.features["aggregate"]


def test_regime_feature_snapshot_matches_truncated_input() -> None:
    close = 100.0 + np.cumsum(np.sin(np.arange(260) / 12) * 0.05 + 0.03)
    df = _ohlcv_from_close(close)

    a = compute_regime_features(df, t=180)
    b = compute_regime_features(df.iloc[:181])

    assert a.to_dict() == b.to_dict()


def test_regime_report_is_json_serializable_and_complete() -> None:
    close = np.linspace(100.0, 105.0, 260)
    report = _analyzer().analyze(_ohlcv_from_close(close), extra_context={"funding": 0.001})
    payload = report.to_dict()

    for key in (
        "primary_regime", "secondary_regime", "micro_regime",
        "confidence", "uncertainty", "allowed_playbooks",
        "blocked_playbooks", "risk_mode", "position_size_multiplier",
        "explanation", "features", "probabilities",
    ):
        assert key in payload
    assert any("funding" in x for x in payload["explanation"]["danger"])
    json.dumps(payload)


def test_tradeability_scoring_config_changes_score_without_api_breakage() -> None:
    x = np.linspace(0, 1, 320)
    close = 100.0 * np.exp(0.28 * x)
    df = _ohlcv_from_close(close)

    default = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(df)
    conservative = RegimeIntelligence(
        RegimeIntelligenceConfig(
            timeframes=("5m", "15m"),
            tradeability_scoring=TradeabilityScoringConfig(base_score=0.25),
        )
    ).analyze(df)

    assert conservative.tradeability_score < default.tradeability_score
    assert conservative.features["scoring"]["tradeability"]["base"] == 0.25


def test_risk_posture_config_controls_default_position_size() -> None:
    close = np.linspace(100.0, 103.0, 260)
    df = _ohlcv_from_close(close)
    report = RegimeIntelligence(
        RegimeIntelligenceConfig(
            timeframes=("5m", "15m"),
            risk_posture=RiskPostureConfig(normal_size_multiplier=0.42),
        )
    ).analyze(df)

    assert report.risk_mode == RiskMode.NORMAL.value
    assert report.position_size_multiplier == 0.42


def test_classification_config_controls_macro_probability_surface() -> None:
    x = np.linspace(0, 1, 320)
    close = 100.0 * np.exp(0.30 * x)
    df = _ohlcv_from_close(close)

    default = RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(df)
    muted_bull = RegimeIntelligence(
        RegimeIntelligenceConfig(
            timeframes=("5m", "15m"),
            classification=RegimeClassificationConfig(
                bull_trend_weight=0.0,
                bull_efficiency_weight=0.0,
                bull_return_weight=0.0,
            ),
        )
    ).analyze(df)

    assert muted_bull.probabilities[MacroRegime.BULL_TREND.value] < default.probabilities[MacroRegime.BULL_TREND.value]
