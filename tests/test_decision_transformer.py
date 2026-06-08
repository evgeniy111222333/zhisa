"""Tests for the Decision Transformer body and trainer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.trajectory import Trajectory, TrajectoryWindowDataset
from zhisa.training.decision_transformer import (
    DTConfig,
    DecisionTransformer,
    DecisionTransformerConfig,
    DecisionTransformerTrainer,
    embed_trajectories,
)


def _make_trajs(n: int, T: int = 12, state_dim: int = 16, n_actions: int = 9, seed: int = 0) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n):
        states = rng.standard_normal((T, state_dim)).astype(np.float32)
        actions = rng.integers(0, n_actions, size=T).astype(np.int64)
        rewards = rng.standard_normal(T).astype(np.float32) * 0.1
        obs = []
        for t in range(T):
            o = {
                "chart": rng.standard_normal((3, 8, 8)).astype(np.float32),
                "numeric": rng.standard_normal((4, 4)).astype(np.float32),
                "context": rng.standard_normal(4).astype(np.float32),
                "state_emb": states[t],
            }
            obs.append(o)
        trajs.append(Trajectory(
            obs=obs, actions=actions, rewards=rewards,
            dones=np.zeros(T, dtype=bool),
        ))
    return trajs


def test_dt_forward_shapes():
    cfg = DecisionTransformerConfig(state_dim=16, n_actions=9, context_length=4, d_model=32, n_heads=4, n_layers=2)
    model = DecisionTransformer(cfg)
    B, T, S = 3, 4, 16
    state = torch.randn(B, T, S)
    rtg = torch.randn(B, T)
    action = torch.randint(0, 9, (B, T))
    timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, T, dtype=torch.bool)
    out = model(state, rtg, action, timesteps, mask=mask)
    assert out["action_logits"].shape == (B, T, 9)
    assert out["rtg_pred"].shape == (B, T)
    assert out["hidden"].shape == (B, T, 32)


def test_dt_predict_action_returns_in_range():
    cfg = DecisionTransformerConfig(state_dim=8, n_actions=5, context_length=3, d_model=16, n_heads=4, n_layers=1)
    model = DecisionTransformer(cfg)
    state = torch.randn(2, 3, 8)
    rtg = torch.zeros(2, 3)
    action = torch.zeros(2, 3, dtype=torch.long)
    timesteps = torch.zeros(2, 3, dtype=torch.long)
    pred = model.predict_action(state, rtg, action, timesteps, mask=torch.ones(2, 3, dtype=torch.bool))
    assert pred.shape == (2,)
    assert int(pred.min()) >= 0 and int(pred.max()) < 5


def test_dt_config_validates_divisibility():
    with pytest.raises(ValueError):
        DecisionTransformerConfig(d_model=15, n_heads=4)


def test_dt_config_validates_context_length():
    with pytest.raises(ValueError):
        DecisionTransformerConfig(context_length=0)


def test_embed_trajectories_uses_state_emb_key():
    class _StubPolicy(torch.nn.Module):
        def encode(self, chart, numeric, context):
            B = chart.size(0)
            return torch.zeros(B, 8)

    pol = _StubPolicy()
    trajs = _make_trajs(2, T=3, state_dim=8)
    out = embed_trajectories(trajs, pol, device="cpu", batch_size=4)
    assert len(out) == 2
    for t in out:
        for o in t.obs:
            assert "state_emb" in o
            assert o["state_emb"].shape == (8,)


def test_trainer_fit_reduces_loss():
    cfg = DTConfig(
        state_dim=16, n_actions=9, context_length=6, d_model=32, n_heads=4, n_layers=2,
        learning_rate=1e-3, batch_size=16, epochs=4, seed=0, device="cpu", verbose=False,
    )
    body_cfg = DecisionTransformerConfig(
        state_dim=cfg.state_dim, n_actions=cfg.n_actions, context_length=cfg.context_length,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=0.0,
    )
    model = DecisionTransformer(body_cfg)
    trajs = _make_trajs(8, T=12, state_dim=cfg.state_dim, n_actions=cfg.n_actions)
    ds = TrajectoryWindowDataset(trajs, context_length=cfg.context_length, n_actions=cfg.n_actions)
    trainer = DecisionTransformerTrainer(model, cfg)
    res = trainer.fit(ds)
    assert len(res.history) == 4
    losses = [h["loss"] for h in res.history]
    assert losses[-1] < losses[0]


def test_trainer_eval_reports_finite_loss():
    cfg = DTConfig(
        state_dim=8, n_actions=5, context_length=3, d_model=16, n_heads=4, n_layers=1,
        learning_rate=1e-3, batch_size=8, epochs=1, seed=0, device="cpu", verbose=False,
    )
    body_cfg = DecisionTransformerConfig(
        state_dim=cfg.state_dim, n_actions=cfg.n_actions, context_length=cfg.context_length,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=0.0,
    )
    model = DecisionTransformer(body_cfg)
    trajs = _make_trajs(4, T=6, state_dim=8, n_actions=5)
    full = TrajectoryWindowDataset(trajs, context_length=3, n_actions=5)
    from torch.utils.data import Subset
    train = Subset(full, list(range(0, len(full) - 4)))
    val = Subset(full, list(range(len(full) - 4, len(full))))
    trainer = DecisionTransformerTrainer(model, cfg)
    res = trainer.fit(train, val_dataset=val)
    assert np.isfinite(res.history[-1]["val_loss"])


def test_trainer_save_load_roundtrip(tmp_path):
    cfg = DTConfig(state_dim=8, n_actions=5, context_length=2, d_model=16, n_heads=4, n_layers=1,
                   epochs=1, seed=0, device="cpu", verbose=False)
    body_cfg = DecisionTransformerConfig(
        state_dim=cfg.state_dim, n_actions=cfg.n_actions, context_length=cfg.context_length,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
    )
    model = DecisionTransformer(body_cfg)
    trainer = DecisionTransformerTrainer(model, cfg)
    path = tmp_path / "dt.pt"
    trainer.save(str(path))
    payload = torch.load(str(path), weights_only=False)
    assert "model" in payload and "config" in payload
    loaded_model, loaded_cfg = DecisionTransformerTrainer.load(str(path))
    assert isinstance(loaded_model, DecisionTransformer)
    assert loaded_cfg.n_actions == cfg.n_actions
    sd = model.state_dict()
    sd_loaded = loaded_model.state_dict()
    for k in sd:
        assert k in sd_loaded
        assert torch.allclose(sd[k], sd_loaded[k])


def test_trainer_handles_short_trajectories():
    """Trajectories shorter than context_length should still be valid (padded)."""
    cfg = DTConfig(state_dim=4, n_actions=3, context_length=8, d_model=8, n_heads=2, n_layers=1,
                   epochs=1, batch_size=2, seed=0, device="cpu", verbose=False)
    body_cfg = DecisionTransformerConfig(
        state_dim=cfg.state_dim, n_actions=cfg.n_actions, context_length=cfg.context_length,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
    )
    model = DecisionTransformer(body_cfg)
    trajs = _make_trajs(3, T=3, state_dim=4, n_actions=3)
    ds = TrajectoryWindowDataset(trajs, context_length=cfg.context_length, n_actions=3)
    trainer = DecisionTransformerTrainer(model, cfg)
    res = trainer.fit(ds)
    assert np.isfinite(res.final_loss)
