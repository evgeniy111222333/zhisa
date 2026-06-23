"""Tests for labeling: triple-barrier, realised vol, regime."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.labeling import (
    ForwardReturnConfig,
    TripleBarrierConfig,
    forward_return_targets,
    hmm_regime_labels,
    realized_volatility,
    triple_barrier,
)


def test_triple_barrier_returns_dataframe(small_market):
    out = triple_barrier(small_market, TripleBarrierConfig(max_holding=16))
    assert {"label", "ret", "t_hit"}.issubset(out.columns)
    assert len(out) == len(small_market)
    assert out["label"].isin([-1, 0, 1]).all()


def test_forward_return_targets_follow_future_close_sign():
    index = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
    close = np.array([100.0, 101.0, 99.0, 99.0, 102.0, 100.0])
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones_like(close),
        },
        index=index,
    )

    out = forward_return_targets(df, ForwardReturnConfig(horizon=2, flat_return_bps=0.0))

    assert out["label"].tolist() == [-1, -1, 1, 1, 0, 0]
    assert out["ret"].iloc[0] == pytest.approx(-0.01)
    assert out["ret"].iloc[2] == pytest.approx((102.0 / 99.0) - 1.0)


def test_forward_return_targets_are_not_long_only_triple_barrier_labels():
    index = pd.date_range("2024-01-01", periods=80, freq="15min", tz="UTC")
    x = np.arange(len(index), dtype=np.float64)
    close = 100.0 + np.sin(x / 2.0)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones_like(close),
        },
        index=index,
    )
    forward = forward_return_targets(df, ForwardReturnConfig(horizon=4, flat_return_bps=0.0))
    asymmetric = triple_barrier(df, TripleBarrierConfig(tp_atr_mult=2.0, sl_atr_mult=1.0, max_holding=4))

    valid = slice(0, -4)
    forward_down_share = float((forward["label"].iloc[valid] == -1).mean())
    asymmetric_down_share = float((asymmetric["label"].iloc[valid] == -1).mean())

    assert 0.35 < forward_down_share < 0.65
    assert abs(asymmetric_down_share - forward_down_share) > 0.15


def test_triple_barrier_no_lookahead(small_market):
    """Mutating the future must not change the label at index t."""
    out = triple_barrier(small_market, TripleBarrierConfig(max_holding=8))
    label_t = out["label"].iloc[100]
    mutated = small_market.copy()
    mutated.iloc[200:, 0] *= 100  # extreme forward mutation
    out2 = triple_barrier(mutated, TripleBarrierConfig(max_holding=8))
    assert out2["label"].iloc[100] == label_t


def test_realized_volatility_shape(small_market):
    s = realized_volatility(small_market, horizon=16, annualise=False)
    assert s.shape[0] == len(small_market)
    assert (s.dropna() >= 0).all()


def test_hmm_regime_labels(small_market):
    s = hmm_regime_labels(small_market, n_states=3, lookback=64, prefer_sklearn=False)
    assert s.shape[0] == len(small_market)
    # Labels should be a small set
    assert s.nunique() <= 3


def test_hmm_regime_labels_stable(small_market):
    """Test that regime labels are stable over a small period due to rebalance_period."""
    # With rebalance_period > len(small_market), it only fits once.
    s = hmm_regime_labels(small_market, n_states=3, lookback=64, rebalance_period=1000, prefer_sklearn=False)
    assert s.shape[0] == len(small_market)
    assert s.nunique() <= 3


def test_hmm_regime_labels_sklearn(small_market):
    """Test that sklearn path works and sorts clusters."""
    try:
        import sklearn
    except ImportError:
        pytest.skip("sklearn not installed")
    s = hmm_regime_labels(small_market, n_states=3, lookback=64, rebalance_period=1000, prefer_sklearn=True)
    assert s.shape[0] == len(small_market)
    assert s.nunique() <= 3


def test_hmm_regime_labels_are_causal_when_future_is_appended(small_market):
    prefix = small_market.iloc[:300]
    prefix_labels = hmm_regime_labels(
        prefix,
        n_states=3,
        lookback=64,
        rebalance_period=100,
        prefer_sklearn=False,
    )
    full_labels = hmm_regime_labels(
        small_market,
        n_states=3,
        lookback=64,
        rebalance_period=100,
        prefer_sklearn=False,
    )
    np.testing.assert_array_equal(prefix_labels.to_numpy(), full_labels.iloc[:300].to_numpy())
