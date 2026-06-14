"""Tests for regime-aware trade planning."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from zhisa.env.actions import DiscreteAction
from zhisa.regime import (
    MacroRegime,
    RegimeIntelligence,
    RegimeIntelligenceConfig,
    RegimeTradePlanner,
    RiskMode,
    TradePlan,
    plan_trade,
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


def test_trade_planner_builds_long_plan_for_bull_trend() -> None:
    x = np.linspace(0, 1, 260)
    close = 100.0 * np.exp(0.28 * x)
    report = _analyzer().analyze(_ohlcv_from_close(close))

    plan = RegimeTradePlanner().plan(report, current_position=0.0)

    assert isinstance(plan, TradePlan)
    assert plan.status in {"tradeable", "conditional"}
    assert plan.risk_budget > 0.0
    assert plan.setups
    assert plan.setups[0].direction == "long"
    assert plan.execution.order_type in {"limit", "market_or_limit"}
    assert plan.execution.scale_in_steps >= 1
    assert plan.position_management.intent in {"add", "hold"}
    if report.trend_phase in {"late", "exhausted"}:
        assert plan.position_management.add_allowed is False
    else:
        assert plan.position_management.add_allowed is True
    assert plan.recommended_action in {
        int(DiscreteAction.LONG_25),
        int(DiscreteAction.LONG_50),
        int(DiscreteAction.LONG_100),
    }
    assert plan.to_dict()["setups"][0]["playbook"] == plan.setups[0].playbook


def test_trade_planner_late_trend_prefers_pullback_to_value() -> None:
    n = 260
    x = np.linspace(0, 1, n)
    close = 100.0 * np.exp(0.50 * x)
    volume = np.full(n, 100.0)
    volume[-1] = 500.0
    df = _ohlcv_from_close(close, volume=volume)
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] * 1.05
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-1] * 0.995
    report = _analyzer().analyze(df)

    plan = plan_trade(report)

    assert report.trend_phase in {"late", "exhausted"}
    assert "late_trend_chase" in report.blocked_playbooks
    assert any(s.playbook == "pullback_to_value_only" for s in plan.setups)
    assert plan.execution.allow_market is False
    assert plan.execution.urgency in {"passive", "wait"}
    assert plan.position_management.add_allowed is False
    assert any("late/exhausted" in note for note in plan.management_notes)


def test_trade_planner_no_trade_when_tradeability_too_low() -> None:
    x = np.linspace(0, 1, 220)
    close = 100.0 * np.exp(0.20 * x)
    report = replace(
        _analyzer().analyze(_ohlcv_from_close(close)),
        tradeability_score=0.05,
        risk_mode=RiskMode.REDUCED.value,
    )

    plan = RegimeTradePlanner().plan(report)

    assert plan.status == "no_trade"
    assert "tradeability below threshold" in plan.no_trade_reasons
    assert plan.recommended_action in {
        int(DiscreteAction.CLOSE),
        int(DiscreteAction.PARTIAL_CLOSE),
        int(DiscreteAction.SKIP),
    }


def test_trade_planner_respects_crash_long_constraints() -> None:
    pre = np.linspace(120.0, 125.0, 160)
    crash = np.linspace(124.0, 72.0, 70)
    close = np.r_[pre, crash]
    volume = np.r_[np.full(pre.size, 100.0), np.full(crash.size, 700.0)]
    report = _analyzer().analyze(_ohlcv_from_close(close, volume=volume))

    plan = plan_trade(report, current_position=0.0)

    assert report.primary_regime == MacroRegime.HIGH_VOL_CRASH.value
    assert all(s.direction != "long" for s in plan.setups)
    assert bool(plan.action_mask[int(DiscreteAction.LONG_100)]) is False
    assert "full_size_long" in report.blocked_playbooks


def test_trade_planner_uses_orderflow_execution_constraints() -> None:
    x = np.linspace(0, 1, 260)
    close = 100.0 * np.exp(0.22 * x)
    report = _analyzer().analyze(_ohlcv_from_close(close))
    report = replace(
        report,
        features={
            **report.features,
            "market_context": {
                **report.features["market_context"],
                "orderflow": {
                    "flags": ["wide_spread", "thin_depth"],
                    "spread_bps": 18.0,
                    "orderflow_score": 0.8,
                    "direction": "buy_pressure",
                },
            },
        },
    )

    plan = plan_trade(report, current_position=0.0)

    assert plan.execution.order_type == "post_only_limit"
    assert plan.execution.allow_market is False
    assert plan.execution.post_only is True
    assert plan.execution.urgency == "passive"
    assert plan.execution.max_slippage_bps <= 2.0
    assert any("poor orderflow liquidity" in note for note in plan.execution.notes)


def test_trade_planner_reduce_only_when_position_exceeds_budget() -> None:
    x = np.linspace(0, 1, 240)
    close = 100.0 * np.exp(0.18 * x)
    report = replace(
        _analyzer().analyze(_ohlcv_from_close(close)),
        risk_mode=RiskMode.DEFENSIVE.value,
        position_size_multiplier=0.2,
        tradeability_score=0.8,
        uncertainty=0.0,
        transition_risk=0.0,
    )

    plan = plan_trade(report, current_position=0.9)

    assert plan.execution.reduce_only is True
    assert plan.position_management.de_risk_required is True
    assert plan.position_management.intent == "reduce"
    assert abs(plan.position_management.reduce_to) <= plan.risk_budget + 1e-9
