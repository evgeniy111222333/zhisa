from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.scripts._rl_training import build_policy_from_checkpoint, load_trading_checkpoint
from zhisa.training.optim import OptimConfig
from zhisa.training.s4_rl import (
    PPOConfig,
    PPOTrainer,
    _balanced_env_schedule,
    _env_sampling_probabilities,
)


def _tiny_model(window: int = 8, image_size: int = 8):
    return build_default_policy(
        in_numeric_features=32,
        in_context_features=10,
        window=window,
        image_size=image_size,
        embed_dim=16,
        vision_channels=(8, 16),
        fusion_layers=1,
        memory_layers=1,
        memory_max_len=4,
    )


def test_random_start_samples_valid_reproducible_offsets(monkeypatch):
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    frame = generate_market(MarketConfig(n_bars=200, freq="15min", seed=1))
    cfg = EnvConfig(window=8, image_size=8, episode_length=16, random_start=True)
    env = TradingEnv(frame, cfg=cfg)
    _, first = env.reset(seed=10)
    _, repeated = env.reset(seed=10)
    _, other = env.reset(seed=11)
    assert first["start_index"] == repeated["start_index"]
    assert cfg.window <= first["start_index"] <= len(frame) - cfg.episode_length - 1
    assert other["start_index"] != first["start_index"]


def test_rollout_stores_pre_action_memory(monkeypatch):
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    frame = generate_market(MarketConfig(n_bars=100, freq="15min", seed=2))
    frame = frame[["open", "high", "low", "close", "volume"]]
    model = _tiny_model()
    cfg = PPOConfig(
        n_iterations=1,
        n_episodes=1,
        max_steps_per_episode=2,
        minibatch_size=2,
        device="cpu",
        optim=OptimConfig(lr=1e-4, scheduler="none"),
        env_cfg=EnvConfig(window=8, image_size=8, episode_length=2, random_start=True),
    )
    trainer = PPOTrainer(model, cfg)
    buffer, stats = trainer._collect_rollout(TradingEnv(frame, cfg=cfg.env_cfg))
    first = next(iter(buffer))
    assert torch.is_tensor(first.history)
    assert torch.count_nonzero(first.history).item() == 0
    assert first.chart.dtype == np.uint8
    assert "ep_equity_returns" in stats


def test_s2_checkpoint_is_rejected_for_s4(tmp_path):
    path = tmp_path / "s2.pt"
    torch.save({
        "model": _tiny_model().state_dict(),
        "checkpoint_meta": {
            "stage": "s2_supervised",
            "trading_policy_ready": False,
            "policy_head_trained": False,
        },
    }, path)
    with pytest.raises(ValueError, match="not a trained trading policy"):
        load_trading_checkpoint(path)


def test_s2b_checkpoint_rebuilds_exact_architecture(tmp_path):
    model = _tiny_model(window=8, image_size=8)
    config = dict(model.cfg.__dict__)
    config["vision_channels"] = list(config["vision_channels"])
    payload = {
        "model": model.state_dict(),
        "model_config": config,
        "checkpoint_meta": {
            "stage": "s2b_imitation",
            "trading_policy_ready": True,
            "policy_head_trained": True,
        },
    }
    restored = build_policy_from_checkpoint(payload)
    assert restored.cfg.window == 8
    assert restored.cfg.image_size == 8
    assert set(restored.state_dict()) == set(model.state_dict())


def test_validation_is_reproducible_and_uses_requested_cvar_alpha(monkeypatch):
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    frames = [
        generate_market(MarketConfig(n_bars=100, freq="15min", seed=seed))[
            ["open", "high", "low", "close", "volume"]
        ]
        for seed in (11, 12)
    ]
    cfg = PPOConfig(
        max_steps_per_episode=4,
        device="cpu",
        env_cfg=EnvConfig(
            window=8, image_size=8, episode_length=4, random_start=True,
        ),
    )
    trainer = PPOTrainer(_tiny_model(), cfg)
    envs = [TradingEnv(frame, cfg=cfg.env_cfg) for frame in frames]
    first = trainer._evaluate_policy(envs, n_episodes=4, seed=77, cvar_alpha=0.5)
    second = trainer._evaluate_policy(envs, n_episodes=4, seed=77, cvar_alpha=0.5)
    assert first == second
    assert first["cvar_alpha"] == 0.5
    assert "cvar" in first and "cvar_10" in first


def test_rollout_sampling_weights_segments_by_valid_starts(monkeypatch):
    monkeypatch.setenv("ZHISA_FAST_RENDER", "1")
    frames = [
        generate_market(MarketConfig(n_bars=n, freq="15min", seed=n))[
            ["open", "high", "low", "close", "volume"]
        ]
        for n in (100, 200)
    ]
    cfg = EnvConfig(window=8, image_size=8, episode_length=16, random_start=True)
    envs = [TradingEnv(frame, cfg=cfg) for frame in frames]
    probabilities = _env_sampling_probabilities(envs, horizon=16)
    expected = np.asarray([100 - 8 - 16, 200 - 8 - 16], dtype=np.float64)
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, expected)


def test_balanced_env_schedule_matches_equal_market_quotas():
    rng = np.random.default_rng(42)
    schedule = _balanced_env_schedule(np.ones(12) / 12.0, n_episodes=48, rng=rng)

    counts = np.bincount(schedule, minlength=12)

    assert schedule.shape == (48,)
    assert counts.tolist() == [4] * 12


def test_balanced_env_schedule_respects_weighted_quotas():
    rng = np.random.default_rng(7)
    schedule = _balanced_env_schedule(np.array([0.25, 0.75]), n_episodes=8, rng=rng)

    counts = np.bincount(schedule, minlength=2)

    assert counts.tolist() == [2, 6]


def test_ppo_disables_dropout_for_likelihood_ratio():
    model = _tiny_model()
    model.train()
    trainer = PPOTrainer(model, PPOConfig(device="cpu"))
    assert trainer.model.training is False
    chart = torch.rand(2, 3, 8, 8)
    numeric = torch.rand(2, 8, 32)
    context = torch.rand(2, 10)
    first = trainer.model(chart=chart, numeric=numeric, context=context)["policy_logits"]
    second = trainer.model(chart=chart, numeric=numeric, context=context)["policy_logits"]
    torch.testing.assert_close(first, second)
