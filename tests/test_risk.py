"""Tests for risk management: limits, position sizing, stops, guard."""
from __future__ import annotations

import math

import numpy as np
import pytest

from zhisa.risk.guard import RiskGuard
from zhisa.risk.limits import RiskLimits, RiskState
from zhisa.risk.position_sizing import (
    SizingConfig,
    fixed_fractional,
    kelly_fractional,
    volatility_targeted,
)
from zhisa.risk.stops import StopConfig, compute_stops, trail_stop


def test_fixed_fractional():
    size = fixed_fractional(equity=10_000, risk_per_trade=0.01, stop_distance=5.0)
    # Risk 1% of 10k = 100. Position size = 100 / 5 = 20 units.
    assert math.isclose(size, 20.0, rel_tol=1e-9)


def test_volatility_targeted_clamps():
    # realised below target -> we want to lever up; should equal target/realised
    s = volatility_targeted(equity=10_000, realised_vol=0.10, target_vol=0.20, max_leverage=3.0)
    assert math.isclose(s, 2.0, rel_tol=1e-6)
    # realised very small -> we want to lever up beyond max -> clamped to leverage cap
    s2 = volatility_targeted(equity=10_000, realised_vol=0.05, target_vol=0.20, max_leverage=3.0)
    assert math.isclose(s2, 3.0, rel_tol=1e-6)
    # realised high -> we want to lever down
    s3 = volatility_targeted(equity=10_000, realised_vol=2.0, target_vol=0.20, max_leverage=3.0)
    assert math.isclose(s3, 0.1, rel_tol=1e-6)


def test_kelly_zero_when_loss_zero():
    assert kelly_fractional(0.6, avg_win=10.0, avg_loss=0.0) == 0.0
    assert kelly_fractional(0.0, avg_win=10.0, avg_loss=5.0) == 0.0
    assert kelly_fractional(1.0, avg_win=10.0, avg_loss=5.0) == 0.0


def test_kelly_positive_when_edge():
    # Strong edge: 60% win rate, 2:1 payoff
    # Kelly f* = p/a - q = 0.6/2.0 - 0.4 = -0.1 (negative -> no edge with this math)
    # Use a slightly different setup so Kelly is positive: 60% win, 4:1 payoff
    f = kelly_fractional(0.6, avg_win=40.0, avg_loss=10.0, kelly_fraction=0.25)
    # f* = 0.6/4 - 0.4 = -0.25 -> still negative; pick a real edge
    f = kelly_fractional(0.8, avg_win=20.0, avg_loss=10.0, kelly_fraction=0.25)
    # f* = 0.8/2 - 0.2 = 0.2 -> positive
    assert f > 0


def test_compute_stops_long():
    sl, tp = compute_stops(side=+1, entry_price=100.0, atr=2.0,
                           cfg=StopConfig(sl_atr_mult=1.5, tp_atr_mult=3.0))
    assert sl == pytest.approx(97.0)
    assert tp == pytest.approx(106.0)


def test_compute_stops_short():
    sl, tp = compute_stops(side=-1, entry_price=100.0, atr=2.0)
    assert sl == pytest.approx(103.0)
    assert tp == pytest.approx(94.0)


def test_trail_stop_tightens_only():
    s0 = 90.0
    # Long: price moves up, stop should ratchet up
    s1 = trail_stop(s0, side=+1, price=110.0, atr=2.0)
    assert s1 > s0
    # Long: price moves down, stop should NOT move down
    s2 = trail_stop(s1, side=+1, price=105.0, atr=2.0)
    assert s2 == s1


def test_risk_guard_blocks_on_drawdown():
    g = RiskGuard(RiskLimits(max_drawdown=0.10))
    g.state.equity = 0.85
    g.state.peak_equity = 1.0
    d = g.check_order(requested_size_equity=0.05, instrument="BTC",
                      positions={}, current_price=100.0)
    assert not d.allowed
    assert "max_drawdown" in d.reason


def test_risk_guard_blocks_on_instrument_cap():
    g = RiskGuard(RiskLimits(max_position_per_instrument=1.0))
    g.state.equity = 1.0
    d = g.check_order(requested_size_equity=2.0, instrument="BTC",
                      positions={"BTC": 0.0}, current_price=100.0)
    # Should be clipped (suggested_size between 0 and 1) but allowed
    assert d.allowed
    assert d.suggested_size < 1.0


def test_risk_state_drawdown():
    s = RiskState(equity=0.9, peak_equity=1.0)
    assert math.isclose(s.drawdown, 0.1, rel_tol=1e-9)
    s.update_equity(0.95)
    assert s.peak_equity == 1.0
    assert math.isclose(s.drawdown, 0.05, rel_tol=1e-9)
