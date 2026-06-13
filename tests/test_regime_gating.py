"""Tests for regime-aware action gating."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from zhisa.env.actions import DiscreteAction
from zhisa.regime import (
    MacroRegime,
    MesoRegime,
    RegimeActionGateConfig,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RiskMode,
    apply_regime_action_mask,
    regime_action_mask,
    regime_position_size_multiplier,
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


def _crash_report():
    pre = np.linspace(120.0, 125.0, 180)
    crash = np.linspace(124.0, 72.0, 70)
    close = np.r_[pre, crash]
    volume = np.r_[np.full(180, 100.0), np.full(70, 700.0)]
    return _analyzer().analyze(_ohlcv_from_close(close, volume=volume))


def _bull_report():
    x = np.linspace(0, 1, 360)
    close = 100.0 * np.exp(0.25 * x)
    return _analyzer().analyze(_ohlcv_from_close(close))


def test_crash_gate_blocks_new_longs_but_allows_derisking() -> None:
    report = _crash_report()
    mask = regime_action_mask(report, current_position=0.5)

    assert report.primary_regime == MacroRegime.HIGH_VOL_CRASH.value
    assert bool(mask[int(DiscreteAction.LONG_100)]) is False
    assert bool(mask[int(DiscreteAction.LONG_50)]) is True  # same exposure is a hold, not new risk
    assert bool(mask[int(DiscreteAction.CLOSE)]) is True
    assert bool(mask[int(DiscreteAction.PARTIAL_CLOSE)]) is True
    assert bool(mask[int(DiscreteAction.SKIP)]) is True


def test_compression_reduced_gate_blocks_full_size_entries() -> None:
    report = replace(
        _bull_report(),
        secondary_regime=MesoRegime.COMPRESSION.value,
        risk_mode=RiskMode.REDUCED.value,
        tradeability_score=0.5,
        position_size_multiplier=0.5,
    )
    mask = regime_action_mask(report, current_position=0.0)

    assert bool(mask[int(DiscreteAction.LONG_25)]) is True
    assert bool(mask[int(DiscreteAction.LONG_50)]) is True
    assert bool(mask[int(DiscreteAction.LONG_100)]) is False
    assert bool(mask[int(DiscreteAction.SHORT_100)]) is False


def test_low_tradeability_blocks_new_risk_not_close() -> None:
    report = replace(_bull_report(), tradeability_score=0.05)
    mask = regime_action_mask(report, current_position=-0.5)

    assert bool(mask[int(DiscreteAction.LONG_25)]) is False
    assert bool(mask[int(DiscreteAction.SHORT_100)]) is False
    assert bool(mask[int(DiscreteAction.CLOSE)]) is True
    assert bool(mask[int(DiscreteAction.PARTIAL_CLOSE)]) is True


def test_apply_regime_action_mask_sets_blocked_logits_to_negative_infinity() -> None:
    report = _crash_report()
    logits = torch.zeros(2, 9)
    masked = apply_regime_action_mask(logits, report, current_position=0.0)

    assert masked.shape == logits.shape
    assert masked[:, int(DiscreteAction.LONG_100)].max().item() < -1e8
    assert masked[:, int(DiscreteAction.CLOSE)].eq(0.0).all()


def test_regime_position_size_multiplier_is_bounded_by_risk_mode() -> None:
    report = replace(
        _bull_report(),
        risk_mode=RiskMode.DEFENSIVE.value,
        position_size_multiplier=1.5,
        tradeability_score=1.0,
        uncertainty=0.0,
        transition_risk=0.0,
    )
    mult = regime_position_size_multiplier(report, cfg=RegimeActionGateConfig(defensive_max_abs_target=0.2))

    assert 0.0 <= mult <= 0.2
