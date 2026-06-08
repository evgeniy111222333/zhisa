"""Tests for the world model + world model trainer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.trajectory import Trajectory
from zhisa.models.world_model import WorldModel, WorldModelConfig
from zhisa.training.world_model_trainer import (
    WorldModelDataset,
    WorldModelTrainer,
    WorldModelTrainerConfig,
)


def _make_traj(seed: int = 0, T: int = 12, state_dim: int = 16, n_actions: int = 9):
    rng = np.random.default_rng(seed)
    obs = []
    for _ in range(T):
        obs.append({
            "chart": rng.standard_normal((3, 8, 8)).astype(np.float32),
            "numeric": rng.standard_normal((4, 4)).astype(np.float32),
            "context": rng.standard_normal(4).astype(np.float32),
            "state_emb": rng.standard_normal(state_dim).astype(np.float32),
        })
    return Trajectory(
        obs=obs,
        actions=rng.integers(0, n_actions, size=T).astype(np.int64),
        rewards=rng.standard_normal(T).astype(np.float32) * 0.1,
        dones=np.zeros(T, dtype=bool),
    )


def test_world_model_step_shapes():
    cfg = WorldModelConfig(state_dim=16, n_actions=9, dynamics_hidden=16)
    wm = WorldModel(cfg)
    z = torch.randn(4, 16)
    a = torch.tensor([0, 1, 2, 3])
    out = wm.step(z, a)
    assert out["z_next"].shape == (4, 16)
    assert out["h_next"].shape == (1, 4, 16)
    assert out["r_pred"].shape == (4,)
    assert out["d_logit"].shape == (4,)
    assert out["d_prob"].shape == (4,)
    assert (out["d_prob"] >= 0).all() and (out["d_prob"] <= 1).all()


def test_world_model_rollout_shapes():
    cfg = WorldModelConfig(state_dim=8, n_actions=5, dynamics_hidden=8)
    wm = WorldModel(cfg)
    z0 = torch.randn(3, 8)
    a = torch.randint(0, 5, (3, 7))
    out = wm.rollout(z0, a)
    assert out["z_seq"].shape == (3, 7, 8)
    assert out["r_seq"].shape == (3, 7)
    assert out["d_seq"].shape == (3, 7)


def test_world_model_save_load_roundtrip(tmp_path):
    cfg = WorldModelConfig(state_dim=8, n_actions=5, dynamics_hidden=8)
    wm = WorldModel(cfg)
    p = tmp_path / "wm.pt"
    wm.save(str(p))
    wm2 = WorldModel.load(str(p))
    sd = wm.state_dict()
    sd2 = wm2.state_dict()
    for k in sd:
        assert torch.allclose(sd[k], sd2[k])


def test_world_model_dataset_basic():
    trajs = [_make_traj(seed=i, T=8) for i in range(3)]
    ds = WorldModelDataset(trajs)
    assert len(ds) == 24
    sample = ds[0]
    assert sample["z"].shape == (16,)
    assert sample["z_next"].shape == (16,)
    assert sample["a"].dtype == torch.long
    assert sample["d"].dtype == torch.float32


def test_world_model_dataset_rejects_missing_state_emb():
    traj = _make_traj(seed=0, T=3)
    for o in traj.obs:
        o.pop("state_emb")
    with pytest.raises(ValueError):
        WorldModelDataset([traj])


def test_world_model_trainer_reduces_loss():
    cfg = WorldModelConfig(state_dim=16, n_actions=9, dynamics_hidden=32)
    wm = WorldModel(cfg)
    trajs = [_make_traj(seed=i, T=20, state_dim=16, n_actions=9) for i in range(8)]
    ds = WorldModelDataset(trajs)
    trainer = WorldModelTrainer(wm, WorldModelTrainerConfig(
        learning_rate=1e-3, batch_size=16, epochs=5, seed=0, device="cpu", verbose=False,
    ))
    res = trainer.fit(ds)
    losses = [h["train_loss"] for h in res.history]
    assert losses[-1] < losses[0]


def test_world_model_trainer_eval_finite():
    cfg = WorldModelConfig(state_dim=8, n_actions=5, dynamics_hidden=8)
    wm = WorldModel(cfg)
    trajs = [_make_traj(seed=i, T=10, state_dim=8, n_actions=5) for i in range(4)]
    ds = WorldModelDataset(trajs)
    from torch.utils.data import Subset
    train = Subset(ds, list(range(0, 20)))
    val = Subset(ds, list(range(20, len(ds))))
    trainer = WorldModelTrainer(wm, WorldModelTrainerConfig(
        learning_rate=1e-3, batch_size=8, epochs=2, seed=0, device="cpu", verbose=False,
    ))
    res = trainer.fit(train, val_dataset=val)
    assert np.isfinite(res.history[-1]["val_state_mse"])


def test_world_model_trainer_save_load_roundtrip(tmp_path):
    cfg = WorldModelConfig(state_dim=8, n_actions=5, dynamics_hidden=8)
    wm = WorldModel(cfg)
    trainer = WorldModelTrainer(wm, WorldModelTrainerConfig(epochs=1, device="cpu", verbose=False))
    p = tmp_path / "wm_trainer.pt"
    trainer.save(str(p))
    wm2, tcfg2 = WorldModelTrainer.load(str(p))
    assert wm2.cfg.state_dim == 8
    assert tcfg2.epochs == 1


def test_world_model_config_from_policy():
    from zhisa.models.policy import PolicyConfig
    pcfg = PolicyConfig(embed_dim=42, n_actions=7)
    wm_cfg = WorldModelConfig.from_policy_config(pcfg)
    assert wm_cfg.state_dim == 42
    assert wm_cfg.n_actions == 7
