"""Tests for regime intelligence diagnostics."""
from __future__ import annotations

import pytest

from zhisa.regime import (
    ExpectedDuration,
    MacroRegime,
    MesoRegime,
    MicroRegime,
    RegimeDiagnosticsConfig,
    RegimeOutcome,
    RegimeReport,
    RiskMode,
    diagnose_regime_sequence,
    plan_trade,
)


def _report(
    primary: str,
    *,
    transition: float = 0.2,
    tradeability: float = 0.7,
    playbooks: list[str] | None = None,
) -> RegimeReport:
    return RegimeReport(
        primary_regime=primary,
        secondary_regime=MesoRegime.PULLBACK.value,
        micro_regime=MicroRegime.QUIET.value,
        confidence=0.75,
        uncertainty=0.2,
        expected_duration=ExpectedDuration.MEDIUM.value,
        transition_risk=transition,
        tradeability_score=tradeability,
        allowed_playbooks=playbooks or ["trend_pullback_long"],
        blocked_playbooks=[],
        risk_mode=RiskMode.NORMAL.value,
        position_size_multiplier=0.75,
        stop_style="atr_based",
        take_profit_style="partial_trailing",
        explanation={"why": [], "danger": []},
        probabilities={primary: 0.75},
    )


def test_regime_diagnostics_groups_outcomes_by_regime_and_playbook() -> None:
    reports = [
        _report(MacroRegime.BULL_TREND.value, playbooks=["trend_pullback_long"]),
        _report(MacroRegime.BULL_TREND.value, playbooks=["trend_pullback_long"]),
        _report(MacroRegime.BEAR_TREND.value, playbooks=["trend_pullback_short"]),
    ]
    outcomes = [
        RegimeOutcome(forward_return=0.03, realized_vol=0.01, max_drawdown=-0.005),
        RegimeOutcome(forward_return=-0.01, realized_vol=0.02, max_drawdown=-0.02),
        RegimeOutcome(forward_return=-0.04, realized_vol=0.03, max_drawdown=-0.01),
    ]

    report = diagnose_regime_sequence(reports, outcomes)

    assert report.coverage["n_reports"] == 3
    assert report.by_primary_regime[MacroRegime.BULL_TREND.value]["n"] == 2
    assert report.by_primary_regime[MacroRegime.BULL_TREND.value]["hit_rate"] == 0.5
    assert report.by_playbook["trend_pullback_long"]["mean_forward_return"] == pytest.approx(0.01)
    assert report.by_playbook["trend_pullback_short"]["worst_drawdown"] == -0.01


def test_regime_diagnostics_scores_no_trade_zones_with_plans() -> None:
    reports = [
        _report(MacroRegime.LOW_LIQUIDITY_CHOP.value, tradeability=0.1, playbooks=["no_trade_wait"]),
        _report(MacroRegime.BULL_TREND.value, tradeability=0.8, playbooks=["trend_pullback_long"]),
        _report(MacroRegime.HIGH_VOL_CRASH.value, tradeability=0.1, playbooks=["no_trade_wait"]),
    ]
    outcomes = [
        {"forward_return": 0.002, "realized_vol": 0.01, "max_drawdown": -0.03},
        {"forward_return": 0.025, "realized_vol": 0.01, "max_drawdown": -0.004},
        {"forward_return": -0.02, "realized_vol": 0.04, "max_drawdown": -0.06},
    ]
    plans = [plan_trade(r) for r in reports]

    diag = diagnose_regime_sequence(
        reports,
        outcomes,
        plans=plans,
        cfg=RegimeDiagnosticsConfig(opportunity_return_threshold=0.01, avoided_drawdown_threshold=0.02),
    )

    assert diag.no_trade["n"] == 2
    assert diag.no_trade["avoided_drawdown_rate"] == 1.0
    assert diag.no_trade["missed_opportunity_rate"] == 0.5
    assert diag.coverage["n_plans"] == 3


def test_regime_diagnostics_transition_prediction_metrics() -> None:
    reports = [
        _report(MacroRegime.BULL_TREND.value, transition=0.15),
        _report(MacroRegime.BULL_TREND.value, transition=0.85),
        _report(MacroRegime.BEAR_TREND.value, transition=0.75),
        _report(MacroRegime.BEAR_TREND.value, transition=0.20),
        _report(MacroRegime.BEAR_TREND.value, transition=0.10),
        _report(MacroRegime.BULL_TREND.value, transition=0.80),
        _report(MacroRegime.BULL_TREND.value, transition=0.10),
    ]
    outcomes = [RegimeOutcome(forward_return=0.0, realized_vol=0.0, max_drawdown=0.0) for _ in reports]

    diag = diagnose_regime_sequence(
        reports,
        outcomes,
        cfg=RegimeDiagnosticsConfig(transition_horizon=1, transition_threshold=0.55),
    )

    assert diag.transition["n"] == 6
    assert diag.transition["positive_rate"] > 0.0
    assert diag.transition["precision"] > 0.0
    assert diag.transition["recall"] > 0.0
    assert diag.transition["auc"] >= 0.5
    assert diag.transition["brier"] >= 0.0
