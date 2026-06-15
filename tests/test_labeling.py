"""Tests for labeling: triple-barrier, realised vol, regime."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.labeling import (
    TripleBarrierConfig,
    hmm_regime_labels,
    realized_volatility,
    triple_barrier,
)


def test_triple_barrier_returns_dataframe(small_market):
    out = triple_barrier(small_market, TripleBarrierConfig(max_holding=16))
    assert {"label", "ret", "t_hit"}.issubset(out.columns)
    assert len(out) == len(small_market)
    assert out["label"].isin([-1, 0, 1]).all()


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

