"""Tests for regime state, memory, vectorization, and trainable encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from zhisa.regime import (
    MacroRegime,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RegimeMemory,
    RegimeMemoryConfig,
    RegimeOutcome,
    RegimeStateTracker,
    RegimeStateTrackerConfig,
    RegimeFeatureVectorizer,
)
from zhisa.regime.encoder import RegimeEncoder, append_regime_context


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


def _bull_report():
    x = np.linspace(0, 1, 420)
    close = 100.0 * np.exp(0.35 * x) + 0.1 * np.sin(np.arange(420) / 7)
    return _analyzer().analyze(_ohlcv_from_close(close), symbol="BTC/USDT")


def _crash_report():
    pre = np.linspace(120.0, 125.0, 180)
    crash = np.linspace(124.0, 82.0, 72)
    chop = 82.0 + 1.2 * np.sin(np.arange(90) / 2)
    close = np.r_[pre, crash, chop]
    volume = np.r_[np.full(180, 100.0), np.full(72, 500.0), np.full(90, 350.0)]
    return _analyzer().analyze(_ohlcv_from_close(close), symbol="BTC/USDT")


def test_regime_vectorizer_is_stable_and_complete() -> None:
    report = _bull_report()
    vectorizer = RegimeFeatureVectorizer()

    a = vectorizer.transform(report)
    b = vectorizer.transform(report)

    assert a.shape == (vectorizer.dim,)
    assert len(vectorizer.feature_names) == vectorizer.dim
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert a.sum() > 0.0


def test_regime_state_tracker_detects_confirmed_transition_and_caps_history() -> None:
    bull = np.linspace(100.0, 150.0, 330)
    crash = np.linspace(150.0, 90.0, 80)
    close = np.r_[bull, crash]
    volume = np.r_[np.full(bull.size, 100.0), np.full(crash.size, 650.0)]
    df = _ohlcv_from_close(close, volume=volume)
    tracker = RegimeStateTracker(RegimeStateTrackerConfig(history_size=3, min_persistence=2))

    s1 = tracker.update(df, t=260, symbol="BTC/USDT")
    s2 = tracker.update(df, t=300, symbol="BTC/USDT")
    s3 = tracker.update(df, t=len(df) - 1, symbol="BTC/USDT")

    assert s1.current.primary_regime == MacroRegime.BULL_TREND.value
    assert s2.stable_for >= 2
    assert s3.changed is True
    assert s3.transition is not None
    assert s3.transition.from_primary == MacroRegime.BULL_TREND.value
    assert s3.transition.to_primary == MacroRegime.HIGH_VOL_CRASH.value
    assert s3.transition.confirmed is True
    assert s3.history_size == 3
    frame = tracker.as_frame()
    assert list(frame.columns)[:3] == ["timestamp", "primary_regime", "secondary_regime"]


def test_regime_memory_retrieves_similar_analogs_and_summarizes_outcomes() -> None:
    bull = _bull_report()
    crash = _crash_report()
    memory = RegimeMemory(RegimeMemoryConfig(capacity=10, min_similarity=-1.0))
    memory.add(bull, outcome=RegimeOutcome(forward_return=0.04, realized_vol=0.02, max_drawdown=-0.01))
    memory.add(crash, outcome={"forward_return": -0.08, "realized_vol": 0.09, "max_drawdown": -0.12})

    matches = memory.query(bull, k=2)
    summary = memory.summarize(matches)

    assert matches[0].item.report.primary_regime == bull.primary_regime
    assert matches[0].similarity > matches[1].similarity
    assert summary.n == 2
    assert summary.hit_rate == 0.5
    assert summary.worst_drawdown == -0.12
    assert summary.regimes[MacroRegime.BULL_TREND.value] == 1


def test_regime_memory_capacity_keeps_latest_items() -> None:
    memory = RegimeMemory(RegimeMemoryConfig(capacity=1))
    bull = _bull_report()
    crash = _crash_report()

    memory.add(bull)
    memory.add(crash)

    assert len(memory) == 1
    assert memory.items[0].report.primary_regime == crash.primary_regime


def test_regime_encoder_shapes_and_gradients() -> None:
    reports = [_bull_report(), _crash_report()]
    encoder = RegimeEncoder()

    out = encoder(reports)
    loss = (
        out["embedding"].sum()
        + out["macro_logits"].sum()
        + out["meso_logits"].sum()
        + out["risk_logits"].sum()
        + out["playbook_logits"].sum()
        + out["tradeability"].sum()
        + out["transition_risk"].sum()
    )
    loss.backward()

    assert out["embedding"].shape == (2, encoder.cfg.embed_dim)
    assert out["macro_logits"].shape[0] == 2
    assert out["playbook_logits"].shape[0] == 2
    assert out["tradeability"].shape == (2,)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())


def test_append_regime_context_supports_numpy_and_torch() -> None:
    np_ctx = np.zeros((2, 10), dtype=np.float32)
    np_regime = np.ones((2, 4), dtype=np.float32)
    torch_ctx = torch.zeros(2, 10)
    torch_regime = torch.ones(2, 4)

    np_out = append_regime_context(np_ctx, np_regime)
    torch_out = append_regime_context(torch_ctx, torch_regime)

    assert np_out.shape == (2, 14)
    assert torch_out.shape == (2, 14)
    assert np.allclose(np_out[:, -4:], 1.0)
    assert torch.allclose(torch_out[:, -4:], torch.ones(2, 4))
