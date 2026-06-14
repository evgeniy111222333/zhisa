"""Tests for experience-aware playbook priors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhisa.regime import (
    PlaybookPriorConfig,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RegimeMemory,
    RegimeMemoryConfig,
    RegimeOutcome,
    RegimePlaybookScorer,
    RegimeTradePlanner,
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
    return RegimeIntelligence(RegimeIntelligenceConfig(timeframes=("5m", "15m")))


def _bull_report():
    x = np.linspace(0, 1, 260)
    close = 100.0 * np.exp(0.24 * x)
    return _analyzer().analyze(_ohlcv_from_close(close))


def _memory_with_outcomes(report, playbook: str, returns: list[float], drawdowns: list[float]) -> RegimeMemory:
    memory = RegimeMemory(RegimeMemoryConfig(capacity=100, min_similarity=-1.0))
    for ret, dd in zip(returns, drawdowns):
        memory.add(
            report,
            outcome=RegimeOutcome(
                forward_return=ret,
                realized_vol=0.015,
                max_drawdown=dd,
                label=playbook,
            ),
        )
    return memory


def test_playbook_prior_boosts_good_history() -> None:
    report = _bull_report()
    memory = _memory_with_outcomes(
        report,
        "trend_pullback_long",
        returns=[0.025, 0.030, 0.018, 0.022, 0.035, 0.015],
        drawdowns=[-0.006, -0.005, -0.009, -0.008, -0.004, -0.007],
    )

    prior = RegimePlaybookScorer(memory, PlaybookPriorConfig(min_samples=5)).prior(report, "trend_pullback_long")

    assert prior.n == 6
    assert prior.hit_rate == 1.0
    assert prior.mean_signed_return > 0.0
    assert prior.reliability > 0.6
    assert prior.score_adjustment > 0.0
    assert not prior.insufficient_samples


def test_playbook_prior_penalizes_bad_tail_risk_history() -> None:
    report = _bull_report()
    memory = _memory_with_outcomes(
        report,
        "trend_pullback_long",
        returns=[0.01, -0.02, -0.015, 0.005, -0.03, -0.01],
        drawdowns=[-0.02, -0.08, -0.06, -0.05, -0.10, -0.04],
    )

    prior = RegimePlaybookScorer(memory, PlaybookPriorConfig(min_samples=5)).prior(report, "trend_pullback_long")

    assert prior.n == 6
    assert prior.worst_drawdown <= -0.08
    assert prior.tail_risk > 1.0
    assert prior.reliability < 0.55
    assert prior.score_adjustment < 0.0


def test_playbook_prior_limits_boost_when_samples_are_insufficient() -> None:
    report = _bull_report()
    memory = _memory_with_outcomes(
        report,
        "trend_pullback_long",
        returns=[0.05, 0.06],
        drawdowns=[-0.002, -0.003],
    )

    prior = RegimePlaybookScorer(memory, PlaybookPriorConfig(min_samples=5, max_boost=0.20)).prior(report, "trend_pullback_long")

    assert prior.n == 2
    assert prior.insufficient_samples
    assert 0.0 <= prior.score_adjustment <= 0.05


def test_memory_aware_planner_changes_setup_score() -> None:
    report = _bull_report()
    base_plan = RegimeTradePlanner().plan(report)
    playbook = base_plan.setups[0].playbook
    good_memory = _memory_with_outcomes(
        report,
        playbook,
        returns=[0.03, 0.025, 0.02, 0.028, 0.018, 0.022],
        drawdowns=[-0.004, -0.006, -0.005, -0.003, -0.006, -0.004],
    )
    bad_memory = _memory_with_outcomes(
        report,
        playbook,
        returns=[-0.02, -0.015, -0.03, -0.01, -0.025, -0.02],
        drawdowns=[-0.08, -0.06, -0.09, -0.05, -0.07, -0.10],
    )

    good_plan = RegimeTradePlanner(memory=good_memory).plan(report)
    bad_plan = RegimeTradePlanner(memory=bad_memory).plan(report)

    good_setup = next(s for s in good_plan.setups if s.playbook == playbook)
    bad_setup = next((s for s in bad_plan.setups if s.playbook == playbook), None)

    assert good_setup.memory_prior["score_adjustment"] > 0.0
    if bad_setup is not None:
        assert bad_setup.memory_prior["score_adjustment"] < 0.0
        assert bad_setup.score < good_setup.score
