"""Tests for the backtest engine, metrics, and splitters."""
from __future__ import annotations

import numpy as np
import pytest

from zhisa.backtest.engine import run_backtest, _extract_trade_returns, buy_and_hold_benchmark
from zhisa.backtest.metrics import compute_metrics
from zhisa.backtest.splitter import SplitSpec, walk_forward_splits, purged_kfold_indices
from zhisa.env.trading_env import EnvConfig


def test_metrics_basic():
    eq = np.array([1.0, 1.01, 1.02, 1.0, 0.99, 1.05])
    m = compute_metrics(eq)
    assert m.n_periods == 5
    assert m.total_return == pytest.approx(0.05, rel=1e-3)
    assert m.max_drawdown >= 0
    assert m.sharpe is not None


def test_metrics_zero_length():
    m = compute_metrics(np.array([1.0]))
    assert m.n_periods == 0


def test_buy_and_hold_shape(small_market):
    bh = buy_and_hold_benchmark(small_market)
    assert bh.shape[0] == len(small_market)
    assert bh[0] == 1.0


def test_random_policy_backtest(tiny_market):
    rng = np.random.default_rng(0)
    policy = lambda _obs: int(rng.integers(0, 9))
    result = run_backtest(tiny_market, policy, cfg=EnvConfig(seed=0, window=8, image_size=8))
    assert len(result.equity) > 1
    assert result.metrics.n_periods > 0


def test_walk_forward_splits():
    n = 1000
    spec = SplitSpec(train_size=400, test_size=100, step=100)
    folds = walk_forward_splits(n, spec)
    assert len(folds) > 0
    for f in folds:
        assert f.train[1] - f.train[0] == 400
        assert f.test[1] - f.test[0] == 100


def test_walk_forward_embargo():
    n = 2000
    spec = SplitSpec(train_size=500, test_size=200, step=200, embargo=20)
    folds = walk_forward_splits(n, spec)
    for f in folds:
        assert f.test[0] - f.train[1] >= 20


def test_purged_kfold():
    folds = purged_kfold_indices(1000, n_splits=5, embargo=5)
    assert len(folds) == 5
    for f in folds:
        assert f.test[1] > f.test[0]


def test_extract_trade_returns():
    positions = np.array([0.0, 0.5, 0.5, 0.0, 0.0, -0.5, -0.5, 0.0])
    equity = np.array([1.0, 1.0, 1.05, 1.06, 1.06, 1.06, 1.04, 1.04])
    tr = _extract_trade_returns(positions, equity)
    assert tr.size >= 1
    # First trade was long; PnL positive
    assert tr[0] > 0
