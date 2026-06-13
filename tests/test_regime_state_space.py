"""Tests for unsupervised state-space regime modeling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.regime import (
    RegimeFeatureVectorizer,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    StateSpaceConfig,
    StateSpaceRegimeModel,
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


def _transition_market() -> pd.DataFrame:
    calm = 100.0 + 0.25 * np.sin(np.arange(170) / 7)
    trend = np.linspace(calm[-1], 138.0, 90)
    crash = np.linspace(138.0, 84.0, 54)
    close = np.r_[calm, trend, crash]
    volume = np.r_[np.full(calm.size, 100.0), np.full(trend.size, 130.0), np.full(crash.size, 520.0)]
    return _ohlcv_from_close(close, volume=volume)


def test_state_space_detects_change_point_and_valid_transition_matrix() -> None:
    df = _transition_market()
    model = StateSpaceRegimeModel(StateSpaceConfig(min_history=80, lookback=220))

    before = model.analyze(df, t=210)
    after = model.analyze(df, t=len(df) - 1)

    assert after.change_point_score >= before.change_point_score
    assert after.transition_probability > 0.25
    assert 0 <= after.current_state < 4
    assert len(after.state_probabilities) == 4
    np.testing.assert_allclose(np.asarray(after.transition_matrix).sum(axis=1), np.ones(4), atol=1e-6)
    assert after.state_label in after.state_labels


def test_state_space_analysis_is_causal_no_lookahead() -> None:
    df = _transition_market()
    model = StateSpaceRegimeModel(StateSpaceConfig(min_history=80, lookback=180))

    full_at_t = model.analyze(df, t=180)
    truncated = model.analyze(df.iloc[:181])

    assert full_at_t.to_dict() == truncated.to_dict()


def test_state_space_returns_default_with_insufficient_history() -> None:
    df = _ohlcv_from_close(np.linspace(100.0, 101.0, 40))
    report = StateSpaceRegimeModel(StateSpaceConfig(min_history=80)).analyze(df)

    assert report.state_label == "insufficient_history"
    assert report.transition_probability == 0.0
    assert report.entropy == 1.0


def test_regime_intelligence_uses_state_space_for_transition_context() -> None:
    df = _transition_market()
    report = RegimeIntelligence(
        RegimeIntelligenceConfig(timeframes=("5m", "15m"), state_space=StateSpaceConfig(min_history=80, lookback=220))
    ).analyze(df)

    assert "state_space" in report.features
    assert report.features["state_space"]["change_point_score"] > 0.5
    assert report.transition_risk > 0.35
    assert "transition_wait" in report.allowed_playbooks
    assert "regime_transition_chase" in report.blocked_playbooks
    assert any("state_space=" in x for x in report.explanation["why"])
    assert any("change-point" in x or "transition probability" in x for x in report.explanation["danger"])


def test_regime_vectorizer_includes_state_space_features() -> None:
    df = _transition_market()
    report = RegimeIntelligence(
        RegimeIntelligenceConfig(timeframes=("5m", "15m"), state_space=StateSpaceConfig(min_history=80, lookback=220))
    ).analyze(df)
    vectorizer = RegimeFeatureVectorizer()
    vec = vectorizer.transform(report)
    names = vectorizer.feature_names

    assert vec[names.index("context.state_space.change_point_score")] == report.features["state_space"]["change_point_score"]
    assert vec[names.index("context.state_space.transition_probability")] == report.features["state_space"]["transition_probability"]
