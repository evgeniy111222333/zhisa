"""Tests for the backtest engine, metrics, and splitters."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zhisa.backtest.engine import run_backtest, _extract_trade_returns, buy_and_hold_benchmark
from zhisa.backtest.metrics import compute_metrics
from zhisa.backtest.regime_ab import RegimeABConfig, RegimeGatedPolicy, run_regime_ab_backtest
from zhisa.backtest.regime_walkforward import (
    RegimeProfileSelectionConfig,
    RegimeWalkForwardConfig,
    run_regime_profile_walk_forward_ab,
    run_regime_walk_forward_ab,
)
from zhisa.backtest.splitter import SplitSpec, walk_forward_splits, purged_kfold_indices
from zhisa.env.actions import DiscreteAction
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


def test_backtest_engine_supports_state_aware_policy(tiny_market):
    class StateAwarePolicy:
        def __init__(self):
            self.seen_t = []
            self.observed = 0

        def select_action(self, *, obs, env):
            self.seen_t.append(env._t)
            return int(DiscreteAction.SKIP)

        def observe_step(self, *, obs, action, reward, info, env):
            self.observed += 1

    policy = StateAwarePolicy()
    result = run_backtest(
        tiny_market,
        policy,
        cfg=EnvConfig(seed=0, window=8, image_size=8, episode_length=10),
    )

    assert policy.seen_t[0] == 8
    assert policy.observed == result.metrics.n_periods


def test_regime_ab_backtest_masks_crash_longs():
    import pandas as pd

    close = np.linspace(150.0, 70.0, 120)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(np.abs(close - open_) * 0.2, close * 0.001)
    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": np.full(close.size, 700.0),
    }, index=pd.date_range("2026-01-01", periods=close.size, freq="5min", tz="UTC"))

    class LongOnlyPolicy:
        def logits(self, _obs):
            logits = torch.zeros(9)
            logits[int(DiscreteAction.LONG_100)] = 10.0
            return logits

        def __call__(self, _obs):
            return int(DiscreteAction.LONG_100)

    ab = run_regime_ab_backtest(
        df,
        LongOnlyPolicy(),
        env_cfg=EnvConfig(seed=0, window=32, image_size=8, episode_length=40, kill_on_drawdown=False),
        cfg=RegimeABConfig(),
        seed=0,
    )

    assert ab.gated.regime_summary["n_masked_actions"] > 0
    assert ab.comparison["regime_summary"]["masked_action_rate"] > 0.0
    assert "plan_status" in ab.gated.regime_summary
    assert "recommended_playbooks" in ab.gated.regime_summary
    assert "execution_order_types" in ab.gated.regime_summary
    assert "execution_urgency" in ab.gated.regime_summary
    assert "position_intents" in ab.gated.regime_summary
    assert "reduce_only_rate" in ab.gated.regime_summary
    assert "no_market_rate" in ab.gated.regime_summary
    assert ab.baseline.result.metrics.n_periods > 0
    assert ab.gated.result.metrics.n_periods > 0


def test_regime_gated_policy_forces_planned_reduce_action():
    mask = np.ones(9, dtype=bool)
    plan = SimpleNamespace(
        status="conditional",
        position_management=SimpleNamespace(
            de_risk_required=True,
            intent="reduce",
            reduce_to=0.25,
        ),
    )

    action = RegimeGatedPolicy._planned_management_action(
        plan,
        mask,
        current_action=int(DiscreteAction.LONG_50),
        current_position=0.5,
    )

    assert action == int(DiscreteAction.LONG_25)


def test_regime_gated_policy_forces_skip_when_flat_no_trade():
    mask = np.ones(9, dtype=bool)
    plan = SimpleNamespace(
        status="no_trade",
        position_management=SimpleNamespace(
            de_risk_required=False,
            intent="hold_or_wait",
            reduce_to=0.0,
        ),
    )

    action = RegimeGatedPolicy._planned_management_action(
        plan,
        mask,
        current_action=int(DiscreteAction.LONG_25),
        current_position=0.0,
    )

    assert action == int(DiscreteAction.SKIP)


def test_regime_ab_backtest_accepts_named_analyzer_profile(tiny_market):
    class SkipPolicy:
        def __call__(self, _obs):
            return int(DiscreteAction.SKIP)

    ab = run_regime_ab_backtest(
        tiny_market,
        SkipPolicy(),
        env_cfg=EnvConfig(seed=0, window=8, image_size=8, episode_length=12),
        cfg=RegimeABConfig(
            analyzer_profile="high_beta_alt",
            symbol="SOL/USDT",
            benchmark_symbol="BTC/USDT",
        ),
        seed=0,
    )

    assert ab.gated.regime_summary["n_steps"] > 0
    assert "mean_transition_risk" in ab.gated.regime_summary
    assert "mean_tradeability" in ab.gated.regime_summary


def test_regime_ab_backtest_auto_resolves_profile_from_symbol(tiny_market):
    class SkipPolicy:
        def __call__(self, _obs):
            return int(DiscreteAction.SKIP)

    ab = run_regime_ab_backtest(
        tiny_market,
        SkipPolicy(),
        env_cfg=EnvConfig(seed=0, window=8, image_size=8, episode_length=12),
        cfg=RegimeABConfig(symbol="AAPL", asset_class="equity", benchmark_symbol="SPY"),
        seed=0,
    )

    assert ab.gated.regime_summary["n_steps"] > 0
    assert "primary_regimes" in ab.gated.regime_summary


def test_regime_walk_forward_ab_aggregates_multiple_folds(tiny_market):
    class SkipPolicy:
        def __call__(self, _obs):
            return int(DiscreteAction.SKIP)

    result = run_regime_walk_forward_ab(
        tiny_market,
        SkipPolicy(),
        cfg=RegimeWalkForwardConfig(
            split=SplitSpec(train_size=120, test_size=80, step=80, n_splits=2),
            min_test_bars=40,
        ),
        env_cfg=EnvConfig(seed=0, window=8, image_size=8, episode_length=20),
        seed=0,
    )

    assert result.summary["n_folds"] == 2
    assert "mean_delta" in result.summary
    assert "std_delta" in result.summary
    assert "gated_win_rate" in result.summary
    assert "primary_regimes" in result.summary
    assert "risk_modes" in result.summary
    assert "plan_status" in result.summary
    assert "recommended_playbooks" in result.summary
    assert "plan_status_rate" in result.summary
    assert "execution_order_types" in result.summary
    assert "execution_urgency" in result.summary
    assert "position_intents" in result.summary
    assert "mean_reduce_only_rate" in result.summary
    assert "mean_no_market_rate" in result.summary
    assert len(result.fold_results) == 2


def test_regime_profile_walk_forward_selects_profile(tiny_market):
    class SkipPolicy:
        def __call__(self, _obs):
            return int(DiscreteAction.SKIP)

    result = run_regime_profile_walk_forward_ab(
        tiny_market,
        SkipPolicy(),
        cfg=RegimeProfileSelectionConfig(
            split=SplitSpec(train_size=120, test_size=80, step=80, n_splits=1),
            profiles=("default", "high_beta_alt"),
            min_test_bars=40,
            selection_metric="delta_sharpe",
        ),
        env_cfg=EnvConfig(seed=0, window=8, image_size=8, episode_length=12),
        seed=0,
    )

    assert result.best_profile in {"default", "high_beta_alt"}
    assert set(result.profile_scores) == {"default", "high_beta_alt"}
    assert result.summary["best_profile"] == result.best_profile
    assert result.summary["n_profiles"] == 2
    report = result.calibration_report
    assert result.summary["calibration_report"] == report
    assert report["best_profile"] == result.best_profile
    assert len(report["profile_rank"]) == 2
    assert report["profile_rank"][0]["rank"] == 1
    assert "masked_action_rate" in report["profile_rank"][0]
    assert "mean_transition_risk" in report["profile_rank"][0]
    assert "mean_tradeability" in report["profile_rank"][0]
    assert "dominant_primary_regime" in report["profile_rank"][0]
    assert "dominant_risk_mode" in report["profile_rank"][0]
    assert "score_delta" in report["best_vs_default"]
    assert report["selection_reasons"]


def test_regime_profile_walk_forward_supports_lower_is_better_metric(tiny_market):
    class SkipPolicy:
        def __call__(self, _obs):
            return int(DiscreteAction.SKIP)

    result = run_regime_profile_walk_forward_ab(
        tiny_market,
        SkipPolicy(),
        cfg=RegimeProfileSelectionConfig(
            split=SplitSpec(train_size=120, test_size=80, step=80, n_splits=1),
            profiles=("default", "high_beta_alt"),
            min_test_bars=40,
            selection_metric="mean_masked_action_rate",
            higher_is_better=False,
        ),
        env_cfg=EnvConfig(seed=0, window=8, image_size=8, episode_length=12),
        seed=0,
    )

    best_score = result.profile_scores[result.best_profile]
    assert best_score == min(result.profile_scores.values())
    assert result.calibration_report["higher_is_better"] is False


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


def test_backtest_script_regime_ab_uses_model_config(tmp_path, monkeypatch, small_market):
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
    torch.save({"model": model.state_dict(), "model_config": model_cfg}, ckpt)

    captured: dict = {}
    dummy_equity = np.array([1.0, 1.01, 1.02, 1.015, 1.03])
    dummy_result = SimpleNamespace(metrics=compute_metrics(dummy_equity))
    dummy_ab = SimpleNamespace(
        baseline=SimpleNamespace(name="baseline", result=dummy_result),
        gated=SimpleNamespace(
            name="regime_gated",
            result=dummy_result,
            regime_summary={"n_steps": 4, "n_masked_actions": 1},
        ),
        comparison={"ok": True},
    )

    def fake_regime_ab(df, policy, *, env_cfg, cfg, seed=0):
        captured["cfg"] = env_cfg
        captured["policy"] = policy
        return dummy_ab

    monkeypatch.setattr(backtest_script, "generate_market", lambda cfg: small_market)
    monkeypatch.setattr(backtest_script, "run_regime_ab_backtest", fake_regime_ab)

    rc = backtest_script.main([
        "--checkpoint", str(ckpt),
        "--bars", "120",
        "--out", "",
        "--regime-ab",
    ])

    assert rc == 0
    assert captured["cfg"].window == model_cfg["window"]
    assert captured["cfg"].image_size == model_cfg["image_size"]
    assert hasattr(captured["policy"], "logits")


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
