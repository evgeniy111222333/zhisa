"""Tests for the S4 PPO trainer.

The tests cover, in order of complexity:
    - :func:`compute_gae` math (no bootstrap, with bootstrap, terminal
      states, gamma=0, lambda=0)
    - :func:`ppo_loss` math (clipped vs unclipped ratio, entropy bonus
      sign, value loss direction)
    - :class:`RolloutBuffer` (add, stack, minibatch indices, clear)
    - :class:`PPOTrainer` end-to-end on a tiny synthetic market
    - NaN-guard: non-finite loss must not poison the optimiser
    - determinism under a fixed seed
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.training.s4_rl import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    Transition,
    compute_gae,
    ppo_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df():
    return generate_market(MarketConfig(n_bars=300, seed=42))


@pytest.fixture
def small_spec():
    return SampleSpec(chart_window=8, image_size=8, n_regime_states=3)


@pytest.fixture
def ppo_model(small_df, small_spec):
    ds = MarketDataset(small_df, spec=small_spec)
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]
    n_ctx = ds._time_features.shape[1]
    return build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=8, image_size=8, n_actions=9, n_regime_classes=3,
    )


@pytest.fixture
def small_env_cfg():
    return EnvConfig(window=8, image_size=8)


def _make_obs(n_feat=42, window=8, image_size=8):
    """Build a synthetic obs dict for transition tests."""
    return {
        "chart": np.zeros((3, image_size, image_size), dtype=np.float32),
        "numeric": np.zeros((window, n_feat), dtype=np.float32),
        "context": np.zeros((10,), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def test_gae_no_bootstrap_no_done_equals_discounted_returns():
    """With no terminal flags and last_value=0, GAE reduces to discounted returns."""
    rewards = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0,
                            gamma=0.9, lam=1.0)
    # With lambda=1 and zero values, GAE = discounted returns.
    expected = np.array([
        1.0 + 0.9 * 0.5,
        0.5,
        0.0,
    ], dtype=np.float32)
    np.testing.assert_allclose(ret, expected, atol=1e-5)


def test_gae_lambda_zero_equals_td_residual():
    """With lambda=0, A_t = r_t + gamma * V(s_{t+1}) - V(s_t)."""
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.5,
                            gamma=1.0, lam=0.0)
    # TD residual with V(s_3) = 0.5 bootstrap:
    # A_0 = 1 + 1*0.5 - 0.5 = 1
    # A_1 = 1 + 1*0.5 - 0.5 = 1
    # A_2 = 1 + 1*0.5 - 0.5 = 1
    np.testing.assert_allclose(adv, [1.0, 1.0, 1.0], atol=1e-5)


def test_gae_terminal_flag_zero_outs_next_bootstrap():
    """A done flag at t prevents the (t+1) value from being included.

    With lambda=1, GAE is equivalent to the discounted return minus
    the value at t, so the (1-done_t) multiplier on the future
    advantage also affects the recursion.
    """
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5], dtype=np.float32)
    dones = np.array([0.0, 1.0], dtype=np.float32)
    # Episode ends after step 1 — the bootstrap value for step 1 is 0.
    adv, ret = compute_gae(rewards, values, dones, last_value=1.0,
                            gamma=0.9, lam=1.0)
    # A_1 (terminal): delta = 1 + 0.9*0 - 0.5 = 0.5
    # A_0: delta = 1 + 0.9*0.5 - 0.5 = 0.95, plus gamma*lam*(1-done)*A_1
    #      = 0.95 + 0.9*1.0*1*0.5 = 1.40
    np.testing.assert_allclose(adv, [1.40, 0.5], atol=1e-5)


def test_gae_returns_equal_advantages_plus_values():
    """returns == advantages + values for any (rewards, values, dones)."""
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=20).astype(np.float32)
    values = rng.normal(size=20).astype(np.float32)
    dones = (rng.uniform(size=20) > 0.8).astype(np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0,
                            gamma=0.99, lam=0.95)
    np.testing.assert_allclose(ret, adv + values, atol=1e-5)


def test_gae_gamma_zero_means_only_immediate_reward():
    rewards = np.array([2.0, 5.0, 7.0], dtype=np.float32)
    values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0,
                            gamma=0.0, lam=1.0)
    np.testing.assert_allclose(adv, [1.0, 4.0, 6.0], atol=1e-5)


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def test_ppo_loss_is_zero_when_policy_unchanged_and_advantage_zero():
    """If adv=0 and ratio=1, policy_loss=0 and total=value_loss - 0."""
    new_lp = torch.tensor([-0.5, -0.7])
    old_lp = torch.tensor([-0.5, -0.7])  # identical → ratio=1
    adv = torch.tensor([0.0, 0.0])
    val = torch.tensor([0.5, 0.5])
    ret = torch.tensor([0.5, 0.5])  # no value error
    ent = torch.tensor([0.5, 0.5])
    loss = ppo_loss(new_lp, old_lp, adv, val, ret, ent, value_coef=1.0, entropy_coef=0.0)
    assert abs(float(loss["policy"])) < 1e-6
    assert abs(float(loss["value"])) < 1e-6
    assert abs(float(loss["total"])) < 1e-6


def test_ppo_loss_returns_scalars():
    """All four entries must be 0-dim scalar tensors."""
    new_lp = torch.randn(8)
    old_lp = torch.randn(8)
    adv = torch.randn(8)
    val = torch.randn(8)
    ret = torch.randn(8)
    ent = torch.randn(8)
    loss = ppo_loss(new_lp, old_lp, adv, val, ret, ent)
    for k, v in loss.items():
        assert v.dim() == 0, f"{k} has shape {v.shape}"


def test_ppo_loss_clipping_limits_ratio_effect():
    """When the new policy diverges from old, clipping kicks in."""
    new_lp = torch.tensor([0.0])  # ratio = exp(0 - (-2)) ≈ 7.4
    old_lp = torch.tensor([-2.0])
    adv = torch.tensor([1.0])  # positive advantage → encourage the action
    val = torch.tensor([0.0])
    ret = torch.tensor([0.0])
    ent = torch.tensor([0.0])
    loss = ppo_loss(new_lp, old_lp, adv, val, ret, ent,
                     clip_ratio=0.2, value_coef=0.0, entropy_coef=0.0)
    # Clipped ratio is 1.2, so policy_loss = -min(7.4, 1.2) ≈ -1.2
    assert -1.3 < float(loss["policy"]) < -1.1


def test_ppo_entropy_bonus_subtracts_from_total():
    """entropy_coef is subtracted (we *maximise* entropy = regularise)."""
    new_lp = torch.zeros(4)
    old_lp = torch.zeros(4)
    adv = torch.zeros(4)
    val = torch.zeros(4)
    ret = torch.zeros(4)
    ent = torch.ones(4)  # high entropy
    loss = ppo_loss(new_lp, old_lp, adv, val, ret, ent,
                     value_coef=0.0, entropy_coef=0.1)
    # total = policy(0) + 0 - 0.1 * 1.0 = -0.1
    assert abs(float(loss["total"]) - (-0.1)) < 1e-5


def test_ppo_value_loss_is_mse():
    new_lp = torch.zeros(2)
    old_lp = torch.zeros(2)
    adv = torch.zeros(2)
    val = torch.tensor([0.0, 1.0])
    ret = torch.tensor([1.0, 0.0])  # errors: -1, +1 → squared 1, 1 → mean 1.0
    ent = torch.zeros(2)
    loss = ppo_loss(new_lp, old_lp, adv, val, ret, ent,
                     value_coef=1.0, entropy_coef=0.0)
    assert abs(float(loss["value"]) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


def test_rollout_buffer_add_and_len():
    buf = RolloutBuffer()
    assert len(buf) == 0
    for i in range(5):
        buf.add(Transition(
            chart=np.zeros((3, 8, 8), dtype=np.float32),
            numeric=np.zeros((8, 4), dtype=np.float32),
            context=np.zeros((3,), dtype=np.float32),
            action=i % 5, reward=0.1, value=0.2, log_prob=-0.3, done=False,
        ))
    assert len(buf) == 5


def test_rollout_buffer_stack_tensors_shapes():
    buf = RolloutBuffer()
    for _ in range(7):
        buf.add(Transition(
            chart=np.random.rand(3, 8, 8).astype(np.float32),
            numeric=np.random.rand(8, 4).astype(np.float32),
            context=np.random.rand(3).astype(np.float32),
            action=1, reward=0.5, value=0.5, log_prob=-0.5, done=False,
        ))
    stacked = buf.stack_tensors()
    assert stacked["chart"].shape == (7, 3, 8, 8)
    assert stacked["numeric"].shape == (7, 8, 4)
    assert stacked["context"].shape == (7, 3)
    assert stacked["action"].shape == (7,)
    assert stacked["reward"].shape == (7,)
    assert stacked["value"].shape == (7,)
    assert stacked["log_prob"].shape == (7,)
    assert stacked["done"].shape == (7,)


def test_rollout_buffer_stack_tensors_empty():
    buf = RolloutBuffer()
    assert buf.stack_tensors() == {}


def test_rollout_buffer_minibatch_indices_cover_all_samples():
    buf = RolloutBuffer()
    for i in range(20):
        buf.add(Transition(
            chart=np.zeros((3, 4, 4), dtype=np.float32),
            numeric=np.zeros((4, 2), dtype=np.float32),
            context=np.zeros((2,), dtype=np.float32),
            action=i, reward=0.0, value=0.0, log_prob=0.0, done=False,
        ))
    rng = np.random.default_rng(0)
    seen: set[int] = set()
    n_batches = 0
    for idx in buf.minibatch_indices(batch_size=5, rng=rng):
        assert idx.shape == (5,)
        seen.update(idx.tolist())
        n_batches += 1
    assert len(seen) == 20
    assert n_batches == 4  # 20 / 5


def test_rollout_buffer_minibatch_drops_partial_final_batch():
    """If N is not a multiple of batch_size, the last partial batch is dropped."""
    buf = RolloutBuffer()
    for i in range(7):
        buf.add(Transition(
            chart=np.zeros((3, 4, 4), dtype=np.float32),
            numeric=np.zeros((4, 2), dtype=np.float32),
            context=np.zeros((2,), dtype=np.float32),
            action=i, reward=0.0, value=0.0, log_prob=0.0, done=False,
        ))
    rng = np.random.default_rng(0)
    total = 0
    for idx in buf.minibatch_indices(batch_size=4, rng=rng):
        total += len(idx)
    # Only one full batch of 4; the remaining 3 are dropped.
    assert total == 4


def test_rollout_buffer_clear_empties_data():
    buf = RolloutBuffer()
    for i in range(3):
        buf.add(Transition(
            chart=np.zeros((3, 4, 4), dtype=np.float32),
            numeric=np.zeros((4, 2), dtype=np.float32),
            context=np.zeros((2,), dtype=np.float32),
            action=i, reward=0.0, value=0.0, log_prob=0.0, done=False,
        ))
    buf.clear()
    assert len(buf) == 0
    assert buf.stack_tensors() == {}


# ---------------------------------------------------------------------------
# PPOTrainer end-to-end
# ---------------------------------------------------------------------------


def test_ppo_trainer_runs_one_iteration(ppo_model, small_df, small_env_cfg):
    cfg = PPOConfig(
        n_episodes=2, max_steps_per_episode=10,
        n_epochs=2, minibatch_size=4,
        env_cfg=small_env_cfg, log_every=1, seed=0,
    )
    trainer = PPOTrainer(ppo_model, cfg)
    result = trainer.fit(small_df)
    assert "history" in result
    assert len(result["history"]) == cfg.n_episodes
    for h in result["history"]:
        assert "mean_return" in h
        assert "total_loss" in h
        assert h["rollout_steps"] > 0


def test_ppo_trainer_writes_checkpoint(ppo_model, small_df, small_env_cfg, tmp_path):
    ckpt = tmp_path / "ppo.pt"
    cfg = PPOConfig(
        n_episodes=1, max_steps_per_episode=5,
        n_epochs=1, minibatch_size=2,
        env_cfg=small_env_cfg, log_every=1, seed=0,
        checkpoint=str(ckpt),
    )
    trainer = PPOTrainer(ppo_model, cfg)
    trainer.fit(small_df)
    assert ckpt.exists()
    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    assert "model" in payload
    assert "config" in payload
    assert "ppo_config" in payload


def test_ppo_trainer_does_not_crash_on_short_episodes(ppo_model, small_df, small_env_cfg):
    """The trainer must handle episodes that terminate before max_steps."""
    cfg = PPOConfig(
        n_episodes=3, max_steps_per_episode=500,  # very long → most end on done
        n_epochs=1, minibatch_size=4,
        env_cfg=small_env_cfg, log_every=1, seed=0,
    )
    trainer = PPOTrainer(ppo_model, cfg)
    result = trainer.fit(small_df)
    assert len(result["history"]) == 3


# ---------------------------------------------------------------------------
# NaN-guard
# ---------------------------------------------------------------------------


def test_ppo_trainer_survives_nan_loss(monkeypatch, ppo_model, small_df, small_env_cfg):
    """If a value-head produces NaN, the trainer must skip the update and continue.

    We inject NaN into the ``value`` head only (not ``policy_logits``)
    so the rollout can still sample valid actions; the NaN propagates
    through GAE → returns → value_loss and triggers the guard.
    """
    nan_injected = {"count": 0}
    original_forward = ppo_model.forward

    def faulty_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        if nan_injected["count"] < 1:
            # NaN in the value head only — keeps the rollout going.
            out["value"] = torch.full_like(out["value"], float("nan"))
            nan_injected["count"] += 1
        return out

    monkeypatch.setattr(ppo_model, "forward", faulty_forward)
    cfg = PPOConfig(
        n_episodes=2, max_steps_per_episode=8,
        n_epochs=2, minibatch_size=4,
        env_cfg=small_env_cfg, log_every=1, seed=0,
    )
    trainer = PPOTrainer(ppo_model, cfg)
    result = trainer.fit(small_df)
    assert len(result["history"]) == cfg.n_episodes


def test_ppo_select_action_falls_back_on_nan_logits(ppo_model):
    """Non-finite logits must not crash action selection."""
    cfg = PPOConfig(device="cpu", seed=0)
    trainer = PPOTrainer(ppo_model, cfg)
    obs = _make_obs(n_feat=ppo_model.cfg.in_numeric_features,
                    window=ppo_model.cfg.window,
                    image_size=ppo_model.cfg.image_size)
    # Patch the model so it returns NaN logits.
    original_forward = ppo_model.forward
    def nan_forward(*a, **kw):
        out = original_forward(*a, **kw)
        out["policy_logits"] = torch.full_like(out["policy_logits"], float("nan"))
        return out
    ppo_model.forward = nan_forward
    try:
        action, logp, value, entropy = trainer._select_action(obs)
    finally:
        ppo_model.forward = original_forward
    assert 0 <= action < ppo_model.cfg.n_actions
    assert torch.isfinite(logp)
    assert torch.isfinite(entropy)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_ppo_is_deterministic_under_seed(small_df, small_spec, small_env_cfg):
    """Same seed + same env + same model → same per-iteration metrics.

    The model must be re-built inside each call so the fixture
    (which is shared across both calls) doesn't accumulate the
    weight updates from the first run into the second.
    """
    from zhisa.models.policy import build_default_policy

    def run_once():
        torch.manual_seed(123)
        ds = MarketDataset(small_df, spec=small_spec)
        n_feat = ds._features.shape[1] + ds._time_features.shape[1]
        n_ctx = ds._time_features.shape[1]
        model = build_default_policy(
            in_numeric_features=n_feat, in_context_features=n_ctx,
            window=8, image_size=8, n_actions=9, n_regime_classes=3,
        )
        cfg = PPOConfig(
            n_episodes=2, max_steps_per_episode=8,
            n_epochs=2, minibatch_size=4,
            env_cfg=small_env_cfg, log_every=1, seed=7,
        )
        trainer = PPOTrainer(model, cfg)
        return [h["mean_return"] for h in trainer.fit(small_df)["history"]]

    a = run_once()
    b = run_once()
    assert a == b
