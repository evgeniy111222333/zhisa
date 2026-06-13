"""Tests for the backtest engine, metrics, and splitters."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zhisa.backtest.engine import run_backtest, _extract_trade_returns, buy_and_hold_benchmark
from zhisa.backtest.metrics import compute_metrics
from zhisa.backtest.splitter import SplitSpec, walk_forward_splits, purged_kfold_indices
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy


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


def test_backtest_script_uses_model_config_for_env(tmp_path, monkeypatch, small_market):
    from zhisa.scripts import backtest as backtest_script

    model_cfg = {
        "in_numeric_features": 32,
        "in_context_features": 10,
        "window": 8,
        "image_size": 16,
        "n_actions": 9,
        "n_regime_classes": 4,
    }
    model = build_default_policy(**model_cfg)
    ckpt = tmp_path / "model.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": model_cfg,
        # Legacy field deliberately disagrees; model_config must win.
        "config": {**model_cfg, "window": 32, "image_size": 64},
    }, ckpt)

    captured: dict = {}
    dummy_equity = np.array([1.0, 1.01, 1.02, 1.015, 1.03])

    def fake_run_backtest(df, policy, cfg, *, seed=0):
        captured["cfg"] = cfg
        return SimpleNamespace(metrics=compute_metrics(dummy_equity))

    monkeypatch.setattr(backtest_script, "generate_market", lambda cfg: small_market)
    monkeypatch.setattr(backtest_script, "run_backtest", fake_run_backtest)

    rc = backtest_script.main([
        "--checkpoint", str(ckpt),
        "--bars", "120",
        "--out", "",
    ])

    assert rc == 0
    assert captured["cfg"].window == model_cfg["window"]
    assert captured["cfg"].image_size == model_cfg["image_size"]


def test_evaluate_script_uses_model_config_for_env(tmp_path, monkeypatch, small_market):
    from zhisa.scripts import evaluate as evaluate_script

    model_cfg = {
        "in_numeric_features": 32,
        "in_context_features": 10,
        "window": 8,
        "image_size": 16,
        "n_actions": 9,
        "n_regime_classes": 4,
    }
    model = build_default_policy(**model_cfg)
    ckpt = tmp_path / "model.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": model_cfg,
        "config": {**model_cfg, "window": 32, "image_size": 64},
    }, ckpt)

    captured: dict = {}
    dummy_equity = np.array([1.0, 1.01, 1.02, 1.015, 1.03])

    def fake_run_backtest(df, policy, cfg, *, seed=0):
        captured["cfg"] = cfg
        return SimpleNamespace(metrics=compute_metrics(dummy_equity))

    monkeypatch.setattr(evaluate_script, "generate_market", lambda cfg: small_market)
    monkeypatch.setattr(evaluate_script, "run_backtest", fake_run_backtest)

    out = tmp_path / "eval.json"
    rc = evaluate_script.main([
        "--checkpoint", str(ckpt),
        "--bars", "120",
        "--out", str(out),
    ])

    assert rc == 0
    assert captured["cfg"].window == model_cfg["window"]
    assert captured["cfg"].image_size == model_cfg["image_size"]
    assert out.exists()
