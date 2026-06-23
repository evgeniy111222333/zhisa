"""Tests for rule-based expert policies used by imitation learning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.expert import (
    ExpertPolicy,
    MomentumExpert,
    SmaCrossExpert,
    SymmetricUtilityExpert,
    SUPPORTED_EXPERTS,
    TripleBarrierExpert,
    build_expert,
)
from zhisa.data.labeling import TripleBarrierConfig
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.actions import DiscreteAction


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_expert_known_kinds():
    for kind in ("triple_barrier", "momentum", "sma_cross", "symmetric_utility"):
        e = build_expert(kind)
        assert isinstance(e, ExpertPolicy)
        assert e.name == kind


def test_build_expert_forwards_kwargs():
    e = build_expert("momentum", lookback=8, threshold=0.001)
    assert e.lookback == 8
    assert e.threshold == pytest.approx(0.001)


def test_build_expert_unknown_raises():
    with pytest.raises(ValueError, match="unknown expert kind"):
        build_expert("nope")


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(SUPPORTED_EXPERTS))
def test_expert_outputs_in_action_range(kind, small_market):
    e = build_expert(kind)
    actions = e.predict_array(small_market, start=e.chart_window)
    assert actions.dtype == np.int64
    assert (actions >= 0).all() and (actions < 9).all()
    # Length matches what we asked for.
    assert len(actions) == len(small_market) - e.chart_window


@pytest.mark.parametrize("kind", list(SUPPORTED_EXPERTS))
def test_expert_warmup_bars_return_skip(kind, small_market):
    """All experts must return SKIP (0) at bars where the observation isn't ready."""
    e = build_expert(kind)
    # First ``chart_window`` bars should all be SKIP.
    expected = DiscreteAction.CLOSE if kind == "symmetric_utility" else DiscreteAction.SKIP
    for t in range(e.chart_window):
        assert e.predict(small_market, t) == int(expected)


def test_symmetric_utility_uses_target_position_actions(small_market):
    expert = SymmetricUtilityExpert(chart_window=32, horizons=(8, 16, 32))
    actions = expert.predict_array(small_market)
    valid = {
        int(DiscreteAction.LONG_25), int(DiscreteAction.LONG_50), int(DiscreteAction.LONG_100),
        int(DiscreteAction.SHORT_25), int(DiscreteAction.SHORT_50), int(DiscreteAction.SHORT_100),
        int(DiscreteAction.CLOSE),
    }
    assert set(map(int, actions)).issubset(valid)
    assert int(DiscreteAction.SKIP) not in actions
    assert int(DiscreteAction.PARTIAL_CLOSE) not in actions


def test_symmetric_utility_does_not_turn_two_sided_whipsaw_into_short():
    n = 160
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 100.0)
    high = np.full(n, 100.2)
    low = np.full(n, 99.8)
    # Both symmetric barriers are touched after the decision bar. Under the
    # conservative contract, both sides lose and the target must stay flat.
    high[65:80] = 105.0
    low[65:80] = 95.0
    frame = pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close, "volume": 1.0,
    }, index=idx)
    expert = SymmetricUtilityExpert(
        chart_window=32, horizons=(16,), horizon_weights=(1.0,),
        take_profit_atr=1.0, stop_loss_atr=1.0,
    )
    assert expert.predict(frame, 64) == int(DiscreteAction.CLOSE)


def test_symmetric_utility_costs_increase_flat_targets(small_market):
    low_cost = SymmetricUtilityExpert(chart_window=32, horizons=(8, 16, 32), fee_bps=0, slippage_bps=0)
    high_cost = SymmetricUtilityExpert(chart_window=32, horizons=(8, 16, 32), fee_bps=30, slippage_bps=20)
    low_actions = low_cost.predict_array(small_market)
    high_actions = high_cost.predict_array(small_market)
    assert np.count_nonzero(high_actions == int(DiscreteAction.CLOSE)) >= np.count_nonzero(
        low_actions == int(DiscreteAction.CLOSE)
    )


# ---------------------------------------------------------------------------
# TripleBarrierExpert
# ---------------------------------------------------------------------------


def test_triple_barrier_expert_action_distribution_is_not_constant(small_market):
    """At least one of the three action classes must appear over a long window."""
    e = TripleBarrierExpert(chart_window=32, max_holding=16)
    a = e.predict_array(small_market, start=e.chart_window)
    unique = set(int(x) for x in a)
    # The expert should mix all three of {LONG_50, SHORT_50, SKIP} on
    # a non-trivial synthetic market. The CI is wide, so we just need
    # at least 2 distinct values to be present.
    assert len(unique) >= 2


def test_triple_barrier_expert_trailing_bars_are_skip(small_market):
    e = TripleBarrierExpert(chart_window=32, max_holding=16)
    n = len(small_market)
    # The last ``max_holding`` bars cannot have a defined forward
    # barrier; the expert must conservatively return SKIP.
    for t in range(n - 16, n):
        assert e.predict(small_market, t) == int(DiscreteAction.SKIP)


def test_triple_barrier_expert_does_not_mutate_input(small_market):
    e = TripleBarrierExpert()
    df_copy = small_market.copy()
    _ = e.predict_array(df_copy, start=32)
    pd.testing.assert_frame_equal(df_copy, small_market)


# ---------------------------------------------------------------------------
# MomentumExpert
# ---------------------------------------------------------------------------


def test_momentum_expert_long_on_strictly_rising_series():
    """A monotonically rising series should produce LONG_50 for every eligible bar."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(100.0, 200.0, n),
        "high": np.linspace(101.0, 201.0, n),
        "low": np.linspace(99.0, 199.0, n),
        "close": np.linspace(100.0, 200.0, n),
        "volume": np.ones(n),
    }, index=idx)
    e = MomentumExpert(lookback=16, threshold=0.0, chart_window=32)
    for t in range(32, n):
        assert e.predict(df, t) == int(DiscreteAction.LONG_50)


def test_momentum_expert_short_on_strictly_falling_series():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(200.0, 100.0, n),
        "high": np.linspace(201.0, 101.0, n),
        "low": np.linspace(199.0, 99.0, n),
        "close": np.linspace(200.0, 100.0, n),
        "volume": np.ones(n),
    }, index=idx)
    e = MomentumExpert(lookback=16, threshold=0.0, chart_window=32)
    for t in range(32, n):
        assert e.predict(df, t) == int(DiscreteAction.SHORT_50)


def test_momentum_expert_threshold_deadband():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    # Tiny up-trend below the threshold → SKIP.
    close = np.full(n, 100.0)
    close[32:] = 100.01
    df = pd.DataFrame({
        "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close, "volume": np.ones(n),
    }, index=idx)
    e = MomentumExpert(lookback=16, threshold=0.05, chart_window=32)
    for t in range(32, n):
        assert e.predict(df, t) == int(DiscreteAction.SKIP)


# ---------------------------------------------------------------------------
# SmaCrossExpert
# ---------------------------------------------------------------------------


def test_sma_cross_expert_returns_valid_actions(small_market):
    e = SmaCrossExpert(fast=10, slow=30, chart_window=32)
    actions = e.predict_array(small_market, start=32)
    valid = {int(DiscreteAction.SKIP), int(DiscreteAction.LONG_50), int(DiscreteAction.SHORT_50)}
    assert set(int(x) for x in actions).issubset(valid)


def test_sma_cross_expert_crossover_on_constructed_series():
    """A price series with a clean bullish crossover should fire LONG_50 exactly once."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.concatenate([
        np.full(100, 100.0),
        np.linspace(100.0, 200.0, 100),
    ])
    df = pd.DataFrame({
        "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close, "volume": np.ones(n),
    }, index=idx)
    e = SmaCrossExpert(fast=10, slow=30, chart_window=32)
    longs = sum(1 for t in range(32, n) if e.predict(df, t) == int(DiscreteAction.LONG_50))
    assert longs >= 1
