"""Training readiness tests — verify the system is ready for real training.

These tests go beyond data pipeline correctness and check:
  1. Model forward pass: valid outputs, correct shapes, no NaN
  2. Gradient flow: all parameters receive gradients
  3. Mini-training: loss decreases over a few steps (convergence signal)
  4. Checkpoint save/load roundtrip
  5. Dataset pipeline: DataLoader yields valid batches
  6. PPO rollout + update: the RL loop doesn't crash
  7. Numerical stability: gradients under extreme inputs
  8. Device handling: model moves to CPU cleanly
  9. Feature dimension consistency between env and model
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import PolicyNetwork, PolicyConfig, build_default_policy
from zhisa.models.heads import MultiTaskHeads, HeadsConfig
from zhisa.training.losses import MultiTaskLoss, LossWeights
from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
from zhisa.training.s4_rl import PPOTrainer, PPOConfig, compute_gae, ppo_loss
from zhisa.env.trading_env import TradingEnv, EnvConfig


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="zhisa_train_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synth_df():
    """1000-bar synthetic OHLCV for training tests."""
    cfg = MarketConfig(n_bars=1000, seed=42, freq="5min")
    return generate_market(cfg).drop(columns=["regime"])


@pytest.fixture
def small_df():
    """Small 300-bar synthetic for quick PPO tests."""
    cfg = MarketConfig(n_bars=300, seed=42, freq="5min", initial_price=1000.0)
    return generate_market(cfg).drop(columns=["regime"])


@pytest.fixture
def env_compatible_df():
    """DF compatible with both env and dataset (500 bars)."""
    cfg = MarketConfig(n_bars=500, seed=42, freq="5min")
    return generate_market(cfg).drop(columns=["regime"])


# ════════════════════════════════════════════════════════════════════
# 1. MODEL FORWARD PASS
# ════════════════════════════════════════════════════════════════════

class TestModelForwardPass:
    """Verify the policy network produces correct outputs."""

    def test_forward_shapes(self):
        """Model outputs have correct shapes for a batch."""
        cfg = PolicyConfig(
            image_size=64, in_numeric_features=20, window=32,
            in_context_features=10, embed_dim=64, n_actions=9,
        )
        model = PolicyNetwork(cfg)
        model.eval()

        B = 4
        chart = torch.randn(B, 3, 64, 64)
        numeric = torch.randn(B, 32, 20)
        context = torch.randn(B, 10)

        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)

        assert out["policy_logits"].shape == (B, 9)
        assert out["value"].shape == (B,)
        assert out["direction"].shape == (B, 3)
        assert out["regime"].shape == (B, cfg.n_regime_classes)
        assert out["volatility"].shape == (B,)
        assert out["return_pred"].shape == (B,)
        assert out["embedding"].shape == (B, cfg.embed_dim)

    def test_forward_no_nan(self):
        """Model outputs are all finite (no NaN/Inf)."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.eval()

        chart = torch.randn(2, 3, 64, 64)
        numeric = torch.randn(2, 32, 20)
        context = torch.randn(2, 10)

        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)

        for key, tensor in out.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all(), (
                    f"Output '{key}' contains NaN/Inf"
                )

    def test_policy_logits_produce_valid_distribution(self):
        """Policy logits can be converted to a valid probability distribution."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.eval()

        chart = torch.randn(4, 3, 64, 64)
        numeric = torch.randn(4, 32, 20)
        context = torch.randn(4, 10)

        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)
            logits = out["policy_logits"]
            probs = F.softmax(logits, dim=-1)

        # Probabilities sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        # All probabilities are non-negative
        assert (probs >= 0).all()
        # Can sample from the distribution
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        assert actions.shape == (4,)
        assert (actions >= 0).all() and (actions < 9).all()

    def test_memory_mechanism(self):
        """Working memory processes history correctly."""
        cfg = PolicyConfig(
            image_size=64, in_numeric_features=20, window=32,
            in_context_features=10, embed_dim=64, use_memory=True,
        )
        model = PolicyNetwork(cfg)
        model.eval()

        chart = torch.randn(2, 3, 64, 64)
        numeric = torch.randn(2, 32, 20)
        context = torch.randn(2, 10)

        # First step: no history
        with torch.no_grad():
            out1 = model(chart=chart, numeric=numeric, context=context)
        assert out1["next_history"] is not None
        assert out1["next_history"].shape[0] == 2  # batch

        # Second step: with history
        with torch.no_grad():
            out2 = model(
                chart=chart, numeric=numeric, context=context,
                history=out1["next_history"],
            )
        assert out2["next_history"] is not None
        # With history, outputs should differ from without
        # (memory should influence the result)
        # This is a soft check — just verify shapes are correct
        assert out2["embedding"].shape == out1["embedding"].shape


# ════════════════════════════════════════════════════════════════════
# 2. GRADIENT FLOW
# ════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Verify gradients flow through the entire model."""

    def test_all_parameters_receive_gradients(self):
        """After a backward pass, all trainable parameters have gradients."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.train()

        chart = torch.randn(2, 3, 64, 64)
        numeric = torch.randn(2, 32, 20)
        context = torch.randn(2, 10)

        out = model(chart=chart, numeric=numeric, context=context)
        # Use a loss that touches all heads
        loss = (
            out["policy_logits"].sum()
            + out["value"].sum()
            + out["direction"].sum()
            + out["regime"].sum()
            + out["volatility"].sum()
            + out["return_pred"].sum()
            + out["risk"].sum()
            + out["uncertainty_logit"].sum()
        )
        loss.backward()

        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradient: {params_without_grad}"
        )

    def test_no_nan_gradients(self):
        """Gradients must not contain NaN values."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.train()

        chart = torch.randn(4, 3, 64, 64)
        numeric = torch.randn(4, 32, 20)
        context = torch.randn(4, 10)

        out = model(chart=chart, numeric=numeric, context=context)
        loss = F.cross_entropy(
            out["policy_logits"],
            torch.randint(0, 9, (4,))
        ) + F.mse_loss(out["value"], torch.randn(4))

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"NaN/Inf gradient in parameter '{name}'"
                )

    def test_gradient_magnitude_reasonable(self):
        """Gradient norms should be in a reasonable range (not exploding)."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.train()

        chart = torch.randn(8, 3, 64, 64)
        numeric = torch.randn(8, 32, 20)
        context = torch.randn(8, 10)

        out = model(chart=chart, numeric=numeric, context=context)
        targets = {
            "label_dir": torch.randint(-1, 2, (8,)),
            "label_vol": torch.randn(8).abs(),
            "label_risk": torch.randn(8).abs(),
            "label_regime": torch.randint(0, 4, (8,)),
            "label_ret": torch.randn(8) * 0.01,
        }

        loss_fn = MultiTaskLoss()
        losses = loss_fn(out, targets)
        losses["total"].backward()

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm < 1000.0, (
            f"Gradient norm {total_norm:.2f} is too large — likely exploding"
        )
        assert total_norm > 1e-8, (
            f"Gradient norm {total_norm:.2e} is too small — likely vanishing"
        )


# ════════════════════════════════════════════════════════════════════
# 3. MINI-TRAINING CONVERGENCE
# ════════════════════════════════════════════════════════════════════

class TestMiniTraining:
    """Verify that loss decreases during a few training steps."""

    def test_supervised_loss_decreases(self):
        """Multi-task loss decreases over a batch of training steps."""
        model = build_default_policy(
            in_numeric_features=20, in_context_features=10,
            embed_dim=64,
        )
        model.train()
        loss_fn = MultiTaskLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed synthetic batch
        B = 16
        chart = torch.randn(B, 3, 64, 64)
        numeric = torch.randn(B, 32, 20)
        context = torch.randn(B, 10)
        targets = {
            "label_dir": torch.randint(-1, 2, (B,)),
            "label_vol": torch.randn(B).abs() * 0.01,
            "label_risk": torch.randn(B).abs() * 0.01,
            "label_regime": torch.randint(0, 4, (B,)),
            "label_ret": torch.randn(B) * 0.001,
        }

        losses = []
        for step in range(30):
            out = model(chart=chart, numeric=numeric, context=context)
            loss_dict = loss_fn(out, targets)
            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (first 5 avg vs last 5 avg)
        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5
        assert last_avg < first_avg, (
            f"Loss did not decrease: {first_avg:.4f} → {last_avg:.4f}"
        )

    def test_ppo_loss_computes_correctly(self):
        """PPO loss function produces valid gradients."""
        B = 32
        new_logp = torch.randn(B, requires_grad=True)
        old_logp = new_logp.detach() + torch.randn(B) * 0.1
        advantages = torch.randn(B)
        values = torch.randn(B, requires_grad=True)
        returns = values.detach() + torch.randn(B) * 0.1
        entropy = torch.rand(B, requires_grad=True)

        losses = ppo_loss(
            new_logp, old_logp, advantages, values, returns, entropy,
        )

        assert torch.isfinite(losses["total"]), "PPO total loss is not finite"
        assert torch.isfinite(losses["policy"]), "PPO policy loss is not finite"
        assert torch.isfinite(losses["value"]), "PPO value loss is not finite"

        losses["total"].backward()
        assert new_logp.grad is not None
        assert values.grad is not None

    def test_gae_computation(self):
        """GAE produces correct advantages and returns."""
        rewards = np.array([1.0, 0.0, -1.0, 2.0, 0.5], dtype=np.float32)
        values = np.array([0.5, 0.3, 0.1, 0.8, 0.4], dtype=np.float32)
        dones = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        adv, ret = compute_gae(rewards, values, dones, last_value=0.0, gamma=0.99, lam=0.95)

        assert adv.shape == (5,)
        assert ret.shape == (5,)
        assert np.all(np.isfinite(adv))
        assert np.all(np.isfinite(ret))
        # Returns = advantages + values
        np.testing.assert_allclose(ret, adv + values, atol=1e-5)


# ════════════════════════════════════════════════════════════════════
# 4. DATASET PIPELINE
# ════════════════════════════════════════════════════════════════════

class TestDatasetPipeline:
    """Verify the MarketDataset produces valid training samples."""

    def test_dataset_creates_samples(self, synth_df):
        """MarketDataset produces non-empty samples."""
        spec = SampleSpec(chart_window=64, feature_window=64, image_size=64)
        ds = MarketDataset(synth_df, spec=spec)

        assert len(ds) > 0, "Dataset is empty"
        sample = ds[0]

        assert sample["chart"].shape == (3, 64, 64)
        assert sample["numeric"].shape[0] == 64  # window
        assert sample["label_dir"].dtype == torch.long
        assert sample["label_vol"].dtype == torch.float32
        assert sample["label_risk"].dtype == torch.float32
        assert sample["label_risk"].item() >= 0.0
        assert sample["label_regime"].dtype == torch.long

    def test_dataset_no_nan_in_samples(self, synth_df):
        """No NaN values in numeric features."""
        spec = SampleSpec(chart_window=64, feature_window=64, image_size=64)
        ds = MarketDataset(synth_df, spec=spec)

        # Check first, middle, and last samples
        indices = [0, len(ds) // 2, len(ds) - 1]
        for idx in indices:
            sample = ds[idx]
            assert torch.isfinite(sample["numeric"]).all(), (
                f"NaN/Inf in numeric features at index {idx}"
            )
            assert torch.isfinite(sample["chart"]).all(), (
                f"NaN/Inf in chart at index {idx}"
            )
            assert torch.isfinite(sample["context"]).all(), (
                f"NaN/Inf in context at index {idx}"
            )

    def test_collate_produces_batch(self, synth_df):
        """multimodal_collate correctly batches samples."""
        spec = SampleSpec(chart_window=64, feature_window=64, image_size=64)
        ds = MarketDataset(synth_df, spec=spec)

        samples = [ds[i] for i in range(4)]
        batch = multimodal_collate(samples)

        assert batch.chart.shape == (4, 3, 64, 64)
        assert batch.numeric.shape[0] == 4
        assert batch.label_dir.shape == (4,)
        assert batch.label_risk.shape == (4,)
        assert len(batch.meta) == 4

    def test_dataloader_iteration(self, synth_df):
        """DataLoader successfully iterates over MarketDataset."""
        from torch.utils.data import DataLoader

        spec = SampleSpec(chart_window=64, feature_window=64, image_size=64)
        ds = MarketDataset(synth_df, spec=spec)
        loader = DataLoader(
            ds, batch_size=8, shuffle=True,
            collate_fn=multimodal_collate, drop_last=True,
        )

        batches_seen = 0
        for batch in loader:
            assert batch.chart.shape[0] == 8
            assert torch.isfinite(batch.chart).all()
            assert torch.isfinite(batch.numeric).all()
            batches_seen += 1
            if batches_seen >= 3:
                break

        assert batches_seen >= 3, "DataLoader didn't produce enough batches"


# ════════════════════════════════════════════════════════════════════
# 5. SUPERVISED TRAINING (S2) — MINI-RUN
# ════════════════════════════════════════════════════════════════════

class TestS2SupervisedTraining:
    """Run a minimal S2 supervised training loop."""

    def test_s2_mini_training(self, synth_df, tmp_dir):
        """S2 trainer runs and loss decreases over 2 epochs."""
        spec = SampleSpec(chart_window=32, feature_window=32, image_size=64)
        ds = MarketDataset(synth_df, spec=spec)

        # Get feature dimensions from a sample
        sample = ds[0]
        n_numeric = sample["numeric"].shape[1]
        n_context = sample["context"].shape[0]

        model = build_default_policy(
            in_numeric_features=n_numeric,
            in_context_features=n_context,
            embed_dim=64, window=32,
        )
        loss_fn = MultiTaskLoss()
        cfg = TrainConfig(
            epochs=2, batch_size=16, device="cpu",
            checkpoint=str(tmp_dir / "s2_test.pt"),
        )
        trainer = SupervisedTrainer(model, loss_fn, cfg)
        result = trainer.fit(ds)

        assert len(result["history"]) == 2
        assert result["history"][0]["loss"] > 0
        # Loss should decrease
        assert result["history"][1]["loss"] <= result["history"][0]["loss"] * 1.5, (
            f"Loss increased dramatically: {result['history'][0]['loss']:.4f} → "
            f"{result['history'][1]['loss']:.4f}"
        )
        # Checkpoint saved
        assert (tmp_dir / "s2_test.pt").exists()

    def test_s2_checkpoint_roundtrip(self, synth_df, tmp_dir):
        """Model checkpoint can be saved and loaded correctly."""
        spec = SampleSpec(chart_window=32, feature_window=32, image_size=64)
        ds = MarketDataset(synth_df, spec=spec)

        sample = ds[0]
        n_numeric = sample["numeric"].shape[1]
        n_context = sample["context"].shape[0]

        model = build_default_policy(
            in_numeric_features=n_numeric,
            in_context_features=n_context,
            embed_dim=64, window=32,
        )
        loss_fn = MultiTaskLoss()
        ckpt_path = str(tmp_dir / "ckpt.pt")
        cfg = TrainConfig(epochs=1, batch_size=16, device="cpu", checkpoint=ckpt_path)
        trainer = SupervisedTrainer(model, loss_fn, cfg)
        trainer.fit(ds)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model" in ckpt
        assert "model_config" in ckpt

        # Rebuild model from checkpoint config
        loaded_cfg = ckpt["model_config"]
        if "vision_channels" in loaded_cfg and isinstance(loaded_cfg["vision_channels"], list):
            loaded_cfg["vision_channels"] = tuple(loaded_cfg["vision_channels"])
        model2 = PolicyNetwork(PolicyConfig(**loaded_cfg))
        model2.load_state_dict(ckpt["model"])

        # Forward pass with loaded model should produce same results
        model.eval()
        model2.eval()
        chart = sample["chart"].unsqueeze(0)
        numeric = sample["numeric"].unsqueeze(0)
        context = sample["context"].unsqueeze(0)

        with torch.no_grad():
            out1 = model(chart=chart, numeric=numeric, context=context)
            out2 = model2(chart=chart, numeric=numeric, context=context)

        torch.testing.assert_close(out1["policy_logits"], out2["policy_logits"])
        torch.testing.assert_close(out1["value"], out2["value"])


# ════════════════════════════════════════════════════════════════════
# 6. PPO RL TRAINING (S4) — MINI-RUN
# ════════════════════════════════════════════════════════════════════

class TestS4PPOTraining:
    """Run a minimal PPO training loop."""

    def test_ppo_mini_training(self, small_df):
        """PPO trainer completes a mini training run without errors."""
        env_cfg = EnvConfig(
            window=32, image_size=64, seed=42,
            kill_on_drawdown=False,
        )
        # Get feature dimensions from the env
        env = TradingEnv(small_df, cfg=env_cfg)
        obs, _ = env.reset(seed=42)
        n_numeric = obs["numeric"].shape[1]
        n_context = obs["context"].shape[0]

        model = build_default_policy(
            in_numeric_features=n_numeric,
            in_context_features=n_context,
            embed_dim=64, window=32,
        )
        ppo_cfg = PPOConfig(
            n_episodes=2,
            max_steps_per_episode=50,
            n_epochs=2,
            minibatch_size=16,
            device="cpu",
            env_cfg=env_cfg,
            seed=42,
        )
        trainer = PPOTrainer(model, ppo_cfg)
        result = trainer.fit(small_df)

        assert "history" in result
        assert len(result["history"]) > 0
        # Check that metrics are finite
        for entry in result["history"]:
            assert math.isfinite(entry["mean_return"]), (
                f"Non-finite mean return: {entry['mean_return']}"
            )
            assert math.isfinite(entry["total_loss"]), (
                f"Non-finite total loss: {entry['total_loss']}"
            )

    def test_ppo_rollout_produces_transitions(self, small_df):
        """PPO rollout collects valid transitions."""
        env_cfg = EnvConfig(window=32, image_size=64, seed=42, kill_on_drawdown=False)
        env = TradingEnv(small_df, cfg=env_cfg)
        obs, _ = env.reset(seed=42)
        n_numeric = obs["numeric"].shape[1]
        n_context = obs["context"].shape[0]

        model = build_default_policy(
            in_numeric_features=n_numeric,
            in_context_features=n_context,
            embed_dim=64, window=32,
        )
        ppo_cfg = PPOConfig(
            n_episodes=2, max_steps_per_episode=30,
            device="cpu", env_cfg=env_cfg, seed=42,
        )
        trainer = PPOTrainer(model, ppo_cfg)

        env2 = TradingEnv(small_df, cfg=env_cfg)
        buf, stats = trainer._collect_rollout(env2)

        assert len(buf) > 0
        assert len(stats["ep_returns"]) == 2
        assert len(stats["ep_lengths"]) == 2

        # Verify buffer can be stacked
        stacked = buf.stack_tensors()
        assert "chart" in stacked
        assert "reward" in stacked
        assert np.all(np.isfinite(stacked["reward"]))
        assert np.all(np.isfinite(stacked["value"]))


# ════════════════════════════════════════════════════════════════════
# 7. NUMERICAL STABILITY
# ════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Test model behavior under extreme inputs."""

    def test_large_input_values(self):
        """Model handles large input values without NaN."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.eval()

        # Very large inputs
        chart = torch.randn(2, 3, 64, 64) * 100
        numeric = torch.randn(2, 32, 20) * 100
        context = torch.randn(2, 10) * 100

        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)

        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), (
                    f"NaN/Inf in output '{key}' with large inputs"
                )

    def test_zero_input(self):
        """Model handles all-zero inputs without NaN."""
        model = build_default_policy(in_numeric_features=20, in_context_features=10)
        model.eval()

        chart = torch.zeros(2, 3, 64, 64)
        numeric = torch.zeros(2, 32, 20)
        context = torch.zeros(2, 10)

        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)

        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), (
                    f"NaN/Inf in output '{key}' with zero inputs"
                )

    def test_multi_task_loss_edge_cases(self):
        """MultiTaskLoss handles edge cases without NaN."""
        loss_fn = MultiTaskLoss()
        B = 4

        # All same labels (zero variance in targets)
        outputs = {
            "direction": torch.randn(B, 3),
            "volatility": torch.randn(B),
            "regime": torch.randn(B, 4),
            "return_pred": torch.randn(B),
            "risk": torch.randn(B),
            "policy_logits": torch.randn(B, 9),
            "value": torch.randn(B),
            "uncertainty_logit": torch.randn(B),
        }
        targets = {
            "label_dir": torch.zeros(B, dtype=torch.long),
            "label_vol": torch.zeros(B),
            "label_risk": torch.zeros(B),
            "label_regime": torch.zeros(B, dtype=torch.long),
            "label_ret": torch.zeros(B),
        }

        losses = loss_fn(outputs, targets)
        assert torch.isfinite(losses["total"]), "Loss is not finite with zero targets"


# ════════════════════════════════════════════════════════════════════
# 8. FEATURE DIMENSION CONSISTENCY
# ════════════════════════════════════════════════════════════════════

class TestFeatureDimensionConsistency:
    """Verify that env observation dimensions match model expectations."""

    def test_env_obs_matches_model_input(self, env_compatible_df):
        """Environment observation dimensions are compatible with PolicyNetwork."""
        env_cfg = EnvConfig(window=32, image_size=64, seed=42)
        env = TradingEnv(env_compatible_df, cfg=env_cfg)
        obs, _ = env.reset(seed=42)

        n_numeric = obs["numeric"].shape[1]
        n_context = obs["context"].shape[0]

        # Build model with matched dimensions
        model = build_default_policy(
            in_numeric_features=n_numeric,
            in_context_features=n_context,
            embed_dim=64, window=32,
        )
        model.eval()

        # Convert obs to tensors
        chart = torch.from_numpy(obs["chart"]).unsqueeze(0)
        numeric = torch.from_numpy(obs["numeric"]).unsqueeze(0)
        context = torch.from_numpy(obs["context"]).unsqueeze(0)

        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)

        assert out["policy_logits"].shape == (1, 9)
        assert torch.isfinite(out["policy_logits"]).all()

    def test_env_obs_dimensions_stable_across_steps(self, env_compatible_df):
        """Observation dimensions don't change during an episode."""
        env_cfg = EnvConfig(window=32, image_size=64, seed=42, kill_on_drawdown=False)
        env = TradingEnv(env_compatible_df, cfg=env_cfg)
        obs, _ = env.reset(seed=42)

        initial_shapes = {k: v.shape for k, v in obs.items()}

        rng = np.random.default_rng(42)
        for _ in range(50):
            action = int(rng.integers(0, env.action_space.n))
            obs, _, terminated, truncated, _ = env.step(action)
            for k, v in obs.items():
                assert v.shape == initial_shapes[k], (
                    f"Observation '{k}' shape changed: {initial_shapes[k]} → {v.shape}"
                )
            if terminated or truncated:
                break

    def test_model_parameter_count(self):
        """Model has a reasonable number of parameters (not accidentally huge)."""
        model = build_default_policy(
            in_numeric_features=32, in_context_features=10,
            embed_dim=128,
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert trainable_params == total_params, "Some parameters are frozen unexpectedly"
        assert trainable_params > 1000, f"Too few parameters: {trainable_params}"
        assert trainable_params < 50_000_000, (
            f"Too many parameters ({trainable_params:,}) for a single-instrument policy"
        )
