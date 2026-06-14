"""Tests for online regime recalibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.regime import (
    OnlineRegimeCalibrator,
    RegimeCalibrationConfig,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RegimeOutcome,
    plan_trade,
)


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    idx = pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": np.full(close.size, 100.0),
    }, index=idx)


def _bull_report():
    close = 100.0 * np.exp(0.25 * np.linspace(0, 1, 220))
    return RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m"))).analyze(_ohlcv_from_close(close))


def test_online_calibrator_does_not_change_report_before_min_samples() -> None:
    report = _bull_report()
    calibrator = OnlineRegimeCalibrator(RegimeCalibrationConfig(min_samples=4))
    calibrator.update(report, RegimeOutcome(forward_return=-0.02, max_drawdown=-0.04), playbook="trend_pullback_long")

    calibrated = calibrator.calibrate(report, playbook="trend_pullback_long")

    assert calibrated == report


def test_online_calibrator_reduces_confidence_tradeability_and_size_after_bad_outcomes() -> None:
    report = _bull_report()
    calibrator = OnlineRegimeCalibrator(RegimeCalibrationConfig(min_samples=4, window=16))
    for _ in range(5):
        calibrator.update(
            report,
            RegimeOutcome(forward_return=-0.025, realized_vol=0.02, max_drawdown=-0.055),
            playbook="trend_pullback_long",
        )

    calibrated = calibrator.calibrate(report, playbook="trend_pullback_long")
    rel = calibrated.features["online_calibration"]

    assert rel["n"] == 5
    assert rel["reliability_score"] < 0.55
    assert calibrated.confidence < report.confidence
    assert calibrated.tradeability_score < report.tradeability_score
    assert calibrated.position_size_multiplier < report.position_size_multiplier
    assert any("online calibration" in x for x in calibrated.explanation["danger"])


def test_online_calibrator_preserves_or_improves_after_good_outcomes() -> None:
    report = _bull_report()
    calibrator = OnlineRegimeCalibrator(RegimeCalibrationConfig(min_samples=4, window=16))
    for _ in range(5):
        calibrator.update(
            report,
            RegimeOutcome(forward_return=0.035, realized_vol=0.01, max_drawdown=-0.005),
            playbook="trend_pullback_long",
        )

    calibrated = calibrator.calibrate(report, playbook="trend_pullback_long")

    assert calibrated.confidence >= report.confidence * 0.95
    assert calibrated.tradeability_score >= report.tradeability_score * 0.95
    assert calibrated.features["online_calibration"]["hit_rate"] == 1.0


def test_calibrated_report_reduces_trade_plan_risk_budget() -> None:
    report = _bull_report()
    calibrator = OnlineRegimeCalibrator(RegimeCalibrationConfig(min_samples=4, window=16))
    for _ in range(5):
        calibrator.update(
            report,
            {"forward_return": -0.03, "realized_vol": 0.03, "max_drawdown": -0.06},
            playbook="trend_pullback_long",
        )

    original_plan = plan_trade(report)
    calibrated_plan = plan_trade(calibrator.calibrate(report, playbook="trend_pullback_long"))

    assert calibrated_plan.risk_budget < original_plan.risk_budget
