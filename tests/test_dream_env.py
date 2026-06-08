"""Tests for :class:`DreamEnv` and the latent actor-critic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.env.dream_env import DreamEnv, DreamEnvConfig
from zhisa.models.latent_actor_critic import LatentActorCritic, LatentActorCriticConfig
from zhisa.models.world_model import WorldModel, WorldModelConfig


def _trainable_wm(state_dim: int = 8, n_actions: int = 5, hidden: int = 8) -> WorldModel:
    cfg = WorldModelConfig(state_dim=state_dim, n_actions=n_actions, dynamics_hidden=hidden)
    torch.manual_seed(0)
    return WorldModel(cfg)


def test_latent_actor_critic_forward_shapes():
    cfg = LatentActorCriticConfig(state_dim=16, n_actions=5, hidden_dim=8)
    ac = LatentActorCritic(cfg)
    z = torch.randn(4, 16)
    logits, value = ac(z)
    assert logits.shape == (4, 5)
    assert value.shape == (4,)


def test_latent_actor_critic_act_returns_in_range():
    cfg = LatentActorCriticConfig(state_dim=8, n_actions=3, hidden_dim=4)
    ac = LatentActorCritic(cfg)
    z = torch.randn(2, 8)
    a, lp = ac.act(z, deterministic=True)
    assert a.shape == (2,)
    assert (a >= 0).all() and (a < 3).all()
    assert torch.isfinite(lp).all()


def test_latent_actor_critic_entropy_positive():
    cfg = LatentActorCriticConfig(state_dim=4, n_actions=3, hidden_dim=4)
    ac = LatentActorCritic(cfg)
    z = torch.randn(2, 4)
    ent = ac.entropy(z)
    assert (ent >= 0).all()


def test_latent_actor_critic_config_validates():
    with pytest.raises(ValueError):
        LatentActorCriticConfig(hidden_dim=0)


def test_dream_env_reset_returns_dict_obs():
    wm = _trainable_wm()
    pool = torch.randn(4, 8)
    env = DreamEnv(wm, pool, cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, horizon=10))
    obs, info = env.reset(seed=0)
    assert "z" in obs and "h" in obs
    assert obs["z"].shape == (8,)
    assert obs["h"].shape == (1, 8)


def test_dream_env_step_returns_valid_transition():
    wm = _trainable_wm()
    pool = torch.randn(4, 8)
    env = DreamEnv(wm, pool, cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, horizon=10, seed=42))
    obs, _ = env.reset(seed=0)
    a = 2
    next_obs, reward, terminated, truncated, info = env.step(a)
    assert next_obs["z"].shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "d_prob" in info


def test_dream_env_terminates_at_horizon():
    wm = _trainable_wm()
    pool = torch.randn(2, 8)
    env = DreamEnv(wm, pool, cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, horizon=5))
    env.reset(seed=0)
    truncated_seen = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            truncated_seen = True
            break
    assert truncated_seen


def test_dream_env_deterministic_with_seed():
    wm = _trainable_wm()
    pool = torch.randn(8, 8)
    e1 = DreamEnv(wm, pool, cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, horizon=5, seed=1))
    e2 = DreamEnv(wm, pool, cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, horizon=5, seed=1))
    o1, _ = e1.reset(seed=0)
    o2, _ = e2.reset(seed=0)
    np.testing.assert_array_equal(o1["z"], o2["z"])
    _, r1, _, _, _ = e1.step(2)
    _, r2, _, _, _ = e2.step(2)
    assert r1 == r2


def test_dream_env_pools_can_be_tensors():
    wm = _trainable_wm()
    pool = torch.zeros(3, 8)
    env = DreamEnv(wm, pool, cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, horizon=5))
    obs, _ = env.reset(seed=0)
    assert obs["z"].shape == (8,)


def test_dream_env_handles_explicit_h_pool():
    wm = _trainable_wm()
    pool_z = torch.zeros(3, 8)
    pool_h = torch.zeros(3, 1, 8)
    env = DreamEnv(wm, pool_z, initial_h=pool_h,
                   cfg=DreamEnvConfig(state_dim=8, dynamics_hidden=8, dynamics_layers=1, horizon=5))
    obs, _ = env.reset(seed=0)
    assert obs["h"].shape == (1, 8)
