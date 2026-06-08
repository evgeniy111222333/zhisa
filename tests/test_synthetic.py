"""Tests for the synthetic market generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market, _default_transition


def test_generation_shape(small_market):
    df = small_market
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1500
    for col in ("open", "high", "low", "close", "volume", "regime"):
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_generation_invariants(small_market):
    df = small_market
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] >= 0).all()


def test_determinism():
    cfg = MarketConfig(n_bars=500, seed=7)
    a = generate_market(cfg)
    b = generate_market(cfg)
    np.testing.assert_array_equal(a["close"].to_numpy(), b["close"].to_numpy())


def test_regimes_present(small_market):
    df = small_market
    assert df["regime"].nunique() >= 2


def test_default_transition_valid():
    M = _default_transition()
    assert M.shape == (5, 5)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-6)
    assert (M >= 0).all()
