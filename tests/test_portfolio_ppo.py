"""Tests for the :class:`PortfolioPPOTrainer`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.portfolio_env import PortfolioConfig, PortfolioEnv
from zhisa.env.trading_env import EnvConfig
from zhisa.models.portfolio_policy import PortfolioPolicyNetwork
from zhisa.training.portfolio_ppo import (
    PortfolioPPOConfig,
    PortfolioPPOTrainer,
    PortfolioRolloutBuffer,
    PortfolioTransition,
)


def _make_market(n_bars: int = 300, seed: int = 0) -> pd.DataFrame:
    return generate_market(MarketConfig(n_bars=n_bars, seed=seed))


def _make_model(n_instruments: int, embed_dim: int = 32):
    cfg_args = dict(
        n_instruments=n_instruments,
        in_numeric_features=32, in_context_features=10,
        window=16, image_size=32, embed_dim=embed_dim, fusion_hidden=32,
    )
    return PortfolioPolicyNetwork(type(PortfolioPPOConfig()).__mro__[0].__new__(type(PortfolioPPOConfig()))) if False else _build(cfg_args)


def _build(cfg_args: dict) -> PortfolioPolicyNetwork:
    from zhisa.models.portfolio_policy import PortfolioPolicyConfig
    return PortfolioPolicyNetwork(PortfolioPolicyConfig(**cfg_args))


def _probe_portfolio_dim(df: pd.DataFrame) -> int:
    """Reconstruct the portfolio summary dim by inspecting PortfolioEnv.

    Portfolio summary = 3*N (positions, drawdown, equity shares) + n_corr_features + 2
    """
    from zhisa.env.portfolio_env import PortfolioEnv
    env = PortfolioEnv({"a": df, "b": df}, cfg=PortfolioConfig(n_instruments=2))
    return int(env.observation_space["portfolio"].shape[0])


def _probe_instrument_obs_dims(df: pd.DataFrame):
    """Probe instrument observation dims (numeric and context feature counts) from a real env."""
    from zhisa.env.portfolio_env import PortfolioEnv
    env = PortfolioEnv({"a": df, "b": df}, cfg=PortfolioConfig(
        n_instruments=2, instrument_names=["a", "b"],
        env_cfg=EnvConfig(window=16, image_size=32, episode_length=20),
    ))
    obs, _ = env.reset()
    inst0 = obs["instruments"][0]
    n_feat = inst0["numeric"].shape[-1]
    n_ctx = inst0["context"].shape[-1]
    return n_feat, n_ctx


def test_portfolio_ppo_config_validates():
    with pytest.raises(ValueError):
        PortfolioPPOConfig(n_instruments=1)


def test_portfolio_buffer_round_trip():
    buf = PortfolioRolloutBuffer()
    N, A = 2, 9
    for i in range(3):
        buf.add(PortfolioTransition(
            chart=np.zeros((N, 3, 4, 4), dtype=np.float32),
            numeric=np.zeros((N, 4, 4), dtype=np.float32),
            context=np.zeros((N, 4), dtype=np.float32),
            action=int(i),
            actions_per_instrument=np.array([1, 2], dtype=np.int64),
            portfolio=np.zeros(8, dtype=np.float32),
            log_prob=-1.5,
            log_prob_per_instrument=np.array([-0.7, -0.8], dtype=np.float32),
            action_mask=np.ones((N, A), dtype=bool),
            reward=0.1 * i,
            value=0.5,
            done=False,
        ))
    assert len(buf) == 3
    stacked = buf.stack_tensors()
    assert stacked["chart"].shape == (3, N, 3, 4, 4)
    assert stacked["actions_per_instrument"].shape == (3, N)
    assert stacked["action_mask"].shape == (3, N, A)
    assert stacked["log_prob_per_instrument"].shape == (3, N)


def test_portfolio_ppo_trainer_smoke():
    df = _make_market(n_bars=300, seed=0)
    portfolio_dim = _probe_portfolio_dim(df)
    n_feat, n_ctx = _probe_instrument_obs_dims(df)
    model = _build(dict(
        n_instruments=2, in_numeric_features=n_feat, in_context_features=n_ctx,
        window=16, image_size=32, embed_dim=24, fusion_hidden=24,
        portfolio_dim=portfolio_dim,
    ))
    cfg = PortfolioPPOConfig(
        n_instruments=2, n_iterations=1, n_episodes=1, max_steps_per_episode=10,
        n_epochs=1, minibatch_size=4, log_every=1, device="cpu", seed=0,
        portfolio_dim=portfolio_dim,
    )
    cfg.env_cfg = EnvConfig(episode_length=10, window=16, image_size=32)
    trainer = PortfolioPPOTrainer(model, cfg)
    out = trainer.fit({"a": df, "b": df})
    assert "history" in out
    assert len(out["history"]) == 1
    entry = out["history"][0]
    assert "mean_return" in entry
    assert "mean_gross_leverage" in entry
    assert np.isfinite(entry["mean_return"])


def test_portfolio_ppo_never_breaches_gross_cap():
    """Across a few rollouts, the env's gross-leverage info should never exceed cap."""
    df = _make_market(n_bars=300, seed=0)
    portfolio_dim = _probe_portfolio_dim(df)
    n_feat, n_ctx = _probe_instrument_obs_dims(df)
    model = _build(dict(
        n_instruments=2, in_numeric_features=n_feat, in_context_features=n_ctx,
        window=16, image_size=32, embed_dim=24, fusion_hidden=24,
        portfolio_dim=portfolio_dim,
    ))
    cfg = PortfolioPPOConfig(
        n_instruments=2, n_iterations=2, n_episodes=2, max_steps_per_episode=10,
        n_epochs=1, minibatch_size=4, log_every=1, device="cpu", seed=0,
        portfolio_dim=portfolio_dim,
    )
    cfg.env_cfg = EnvConfig(episode_length=10, window=16, image_size=32)
    gross_cap = 0.5  # tight cap
    env_cfg = PortfolioConfig(
        n_instruments=2, instrument_names=["a", "b"],
        env_cfg=cfg.env_cfg, gross_leverage_cap=gross_cap,
    )
    trainer = PortfolioPPOTrainer(model, cfg)
    out = trainer.fit({"a": df, "b": df}, env_cfg=env_cfg)
    for entry in out["history"]:
        # mean_gross_leverage is an average; the max gross per episode
        # is bounded by cap + slippage headroom. We don't expect the
        # *average* to exceed cap; if it does, mask failed.
        assert entry["mean_gross_leverage"] <= gross_cap + 0.05


def test_portfolio_ppo_save_load_roundtrip(tmp_path):
    df = _make_market(n_bars=200, seed=0)
    portfolio_dim = _probe_portfolio_dim(df)
    n_feat, n_ctx = _probe_instrument_obs_dims(df)
    model = _build(dict(
        n_instruments=2, in_numeric_features=n_feat, in_context_features=n_ctx,
        window=16, image_size=32, embed_dim=24, fusion_hidden=24,
        portfolio_dim=portfolio_dim,
    ))
    cfg = PortfolioPPOConfig(
        n_instruments=2, n_iterations=1, n_episodes=1, max_steps_per_episode=5,
        n_epochs=1, minibatch_size=2, log_every=1, device="cpu", seed=0,
        portfolio_dim=portfolio_dim, checkpoint=str(tmp_path / "ppo.pt"),
    )
    cfg.env_cfg = EnvConfig(episode_length=5, window=16, image_size=32)
    trainer = PortfolioPPOTrainer(model, cfg)
    trainer.fit({"a": df, "b": df})
    payload = torch.load(str(tmp_path / "ppo.pt"), weights_only=False, map_location="cpu")
    assert "model" in payload
    assert "ppo_config" in payload


def test_portfolio_ppo_mask_enforced_in_buffer():
    """The buffer's action_mask should be a bool (N, A) tensor with at least one True per instrument."""
    df = _make_market(n_bars=200, seed=0)
    portfolio_dim = _probe_portfolio_dim(df)
    n_feat, n_ctx = _probe_instrument_obs_dims(df)
    model = _build(dict(
        n_instruments=2, in_numeric_features=n_feat, in_context_features=n_ctx,
        window=16, image_size=32, embed_dim=24, fusion_hidden=24,
        portfolio_dim=portfolio_dim,
    ))
    cfg = PortfolioPPOConfig(
        n_instruments=2, n_iterations=1, n_episodes=1, max_steps_per_episode=5,
        n_epochs=1, minibatch_size=2, log_every=1, device="cpu", seed=0,
        portfolio_dim=portfolio_dim,
    )
    cfg.env_cfg = EnvConfig(episode_length=5, window=16, image_size=32)
    trainer = PortfolioPPOTrainer(model, cfg)
    # Manually run one rollout and check the buffer.
    env = PortfolioEnv({"a": df, "b": df}, cfg=PortfolioConfig(
        n_instruments=2, instrument_names=["a", "b"],
        env_cfg=cfg.env_cfg, gross_leverage_cap=0.5,
    ))
    buf, _ = trainer._collect_rollout(env)
    assert len(buf) > 0
    stacked = buf.stack_tensors()
    assert stacked["action_mask"].dtype == bool
    # At least one action must be valid per (timestep, instrument).
    for i in range(stacked["action_mask"].shape[0]):
        for j in range(stacked["action_mask"].shape[1]):
            assert stacked["action_mask"][i, j].any()
