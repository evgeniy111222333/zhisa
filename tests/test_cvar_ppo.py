"""Tests for the CVaR-Constrained PPO trainer (Lagrangian dual)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.training.cvar_ppo import (
    CVaRPPOConfig,
    CVaRPPOTrainer,
    _per_episode_returns,
)


def _make_market(n_bars: int = 500, seed: int = 0) -> pd.DataFrame:
    return generate_market(MarketConfig(n_bars=n_bars, seed=seed))


def _make_model(df: pd.DataFrame, window: int = 16, image_size: int = 32):
    spec = SampleSpec(chart_window=window, feature_window=window,
                      image_size=image_size, n_regime_states=4)
    probe = MarketDataset(df, spec=spec)
    n_feat = probe._features.shape[1]
    n_ctx = probe._time_features.shape[1]
    return build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=window, image_size=image_size, n_actions=9, n_regime_classes=4,
    )


def test_per_episode_returns_groups_correctly():
    rewards = np.array([0.1, 0.2, 0.3, 0.0, 0.5, 0.1, 0.0], dtype=np.float32)
    dones = np.array([0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
    out = _per_episode_returns(rewards, dones)
    np.testing.assert_allclose(out, [0.6, 0.6], atol=1e-5)


def test_per_episode_returns_handles_no_dones():
    rewards = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    out = _per_episode_returns(rewards, dones)
    assert len(out) == 1
    np.testing.assert_allclose(out, [0.6], atol=1e-5)


def test_per_episode_returns_empty():
    out = _per_episode_returns(np.zeros(0), np.zeros(0))
    assert out.size == 0


def test_cvar_ppo_config_validates():
    with pytest.raises(ValueError):
        CVaRPPOConfig(cvar_alpha=0.0)
    with pytest.raises(ValueError):
        CVaRPPOConfig(cvar_threshold=-0.1)
    with pytest.raises(ValueError):
        CVaRPPOConfig(cvar_lambda_lr=0.0)


def test_cvar_ppo_trainer_smoke():
    df = _make_market(n_bars=300, seed=0)
    model = _make_model(df)
    cfg = CVaRPPOConfig(
        n_iterations=1, n_episodes=1, max_steps_per_episode=20,
        cvar_alpha=0.2, cvar_threshold=0.05, n_epochs=1, minibatch_size=8,
        log_every=1, device="cpu", seed=0,
    )
    cfg.env_cfg = EnvConfig(episode_length=20, window=16, image_size=32)
    trainer = CVaRPPOTrainer(model, cfg)
    out = trainer.fit(df)
    assert "history" in out
    assert "cvar_history" in out
    assert len(trainer.cvar_history) == 1
    assert "cvar" in trainer.cvar_history[0]
    assert "lambda_cvar" in trainer.cvar_history[0]


def test_cvar_ppo_lambda_increases_on_violation():
    df = _make_market(n_bars=400, seed=0)
    model = _make_model(df)
    cfg = CVaRPPOConfig(
        n_iterations=2, n_episodes=2, max_steps_per_episode=20,
        cvar_alpha=0.2, cvar_threshold=0.5,  # tight -> violations expected
        cvar_lambda_init=0.0, cvar_lambda_lr=0.5, cvar_lambda_max=10.0,
        n_epochs=1, minibatch_size=8, log_every=1, device="cpu", seed=0,
    )
    cfg.env_cfg = EnvConfig(episode_length=20, window=16, image_size=32)
    trainer = CVaRPPOTrainer(model, cfg)
    trainer.fit(df)
    assert trainer.lambda_cvar > 0.0


def test_cvar_ppo_lambda_clamped_to_max():
    df = _make_market(n_bars=400, seed=0)
    model = _make_model(df)
    cfg = CVaRPPOConfig(
        n_iterations=2, n_episodes=2, max_steps_per_episode=20,
        cvar_alpha=0.2, cvar_threshold=0.0,
        cvar_lambda_init=0.0, cvar_lambda_lr=10.0, cvar_lambda_max=2.0,
        n_epochs=1, minibatch_size=8, log_every=1, device="cpu", seed=0,
    )
    cfg.env_cfg = EnvConfig(episode_length=20, window=16, image_size=32)
    trainer = CVaRPPOTrainer(model, cfg)
    trainer.fit(df)
    assert trainer.lambda_cvar <= cfg.cvar_lambda_max


def test_cvar_ppo_warmup_keeps_lambda_at_init():
    """When ``n_iterations == cvar_warmup_iters`` the dual is never updated."""
    df = _make_market(n_bars=400, seed=0)
    model = _make_model(df)
    cfg = CVaRPPOConfig(
        n_iterations=2, n_episodes=2, max_steps_per_episode=20,
        cvar_alpha=0.2, cvar_threshold=0.0,
        cvar_lambda_init=0.0, cvar_lambda_lr=0.5,
        cvar_warmup_iters=2,  # all iterations are warmup
        n_epochs=1, minibatch_size=8, log_every=1, device="cpu", seed=0,
    )
    cfg.env_cfg = EnvConfig(episode_length=20, window=16, image_size=32)
    trainer = CVaRPPOTrainer(model, cfg)
    trainer.fit(df)
    assert trainer.lambda_cvar == 0.0
    for entry in trainer.cvar_history:
        assert entry["lambda_cvar"] == 0.0


def test_cvar_ppo_dual_ascent_step_unit():
    """Direct check of the dual update rule: ``lambda <- clip(lambda + lr * violation, 0, max)``."""
    from zhisa.risk.cvar import cvar_constraint_violation
    returns = np.array([-1.0, -2.0, 0.5, 0.6], dtype=np.float32)
    alpha = 0.5
    threshold = 0.3
    lr = 0.5
    lam = 0.0
    violation = cvar_constraint_violation(returns, alpha=alpha, threshold=threshold)
    assert violation > 0.0
    lam = float(np.clip(lam + lr * violation, 0.0, 10.0))
    assert lam > 0.0
    # Safe case: returns all positive, no violation, lambda unchanged.
    safe_returns = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    safe_violation = cvar_constraint_violation(safe_returns, alpha=alpha, threshold=threshold)
    assert safe_violation == 0.0
    lam_after = float(np.clip(lam + lr * safe_violation, 0.0, 10.0))
    assert lam_after == lam


def test_cvar_ppo_save_load_roundtrip(tmp_path):
    df = _make_market(n_bars=300, seed=0)
    model = _make_model(df)
    cfg = CVaRPPOConfig(
        n_iterations=1, n_episodes=1, max_steps_per_episode=10,
        cvar_alpha=0.2, cvar_threshold=0.1, n_epochs=1, minibatch_size=8,
        log_every=1, device="cpu", seed=0,
        checkpoint=str(tmp_path / "cvar_ppo.pt"),
    )
    cfg.env_cfg = EnvConfig(episode_length=10, window=16, image_size=32)
    trainer = CVaRPPOTrainer(model, cfg)
    trainer.fit(df)
    payload = torch.load(str(tmp_path / "cvar_ppo.pt"), weights_only=False, map_location="cpu")
    assert "model" in payload
    assert "lambda_cvar" in payload
    assert "cvar_history" in payload


def test_cvar_ppo_history_records_cvar_metric():
    df = _make_market(n_bars=300, seed=0)
    model = _make_model(df)
    cfg = CVaRPPOConfig(
        n_iterations=2, n_episodes=2, max_steps_per_episode=10,
        cvar_alpha=0.3, cvar_threshold=0.05,
        n_epochs=1, minibatch_size=8, log_every=1, device="cpu", seed=0,
    )
    cfg.env_cfg = EnvConfig(episode_length=10, window=16, image_size=32)
    trainer = CVaRPPOTrainer(model, cfg)
    out = trainer.fit(df)
    for entry in trainer.cvar_history:
        assert "cvar" in entry
        assert "cvar_violation" in entry
        assert "lambda_cvar" in entry
        assert "cvar_penalty" in entry
        assert np.isfinite(entry["cvar"])
        assert entry["cvar_violation"] >= 0.0
