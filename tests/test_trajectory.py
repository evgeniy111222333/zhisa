"""Tests for the trajectory data structures (S6 Decision Transformer)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.trajectory import (
    Trajectory,
    TrajectoryBuffer,
    TrajectoryWindowDataset,
    collect_trajectories,
    compute_returns_to_go,
)


def _dummy_obs(i: int, F: int = 4, T: int = 8) -> dict:
    rng = np.random.default_rng(i)
    return {
        "chart": rng.standard_normal((3, 16, 16)).astype(np.float32),
        "numeric": rng.standard_normal((T, F)).astype(np.float32),
        "context": rng.standard_normal(F).astype(np.float32),
        "state_emb": rng.standard_normal(F).astype(np.float32),
    }


def test_compute_returns_to_go_basic():
    rewards = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.zeros(4, dtype=bool)
    rtg = compute_returns_to_go(rewards, dones, gamma=1.0)
    assert rtg.shape == (4,)
    np.testing.assert_allclose(rtg, [4.0, 3.0, 2.0, 1.0], atol=1e-5)


def test_compute_returns_to_go_with_discount():
    rewards = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.zeros(4, dtype=bool)
    rtg = compute_returns_to_go(rewards, dones, gamma=0.5)
    np.testing.assert_allclose(rtg[3], 1.0, atol=1e-5)
    np.testing.assert_allclose(rtg[0], 1.0 + 0.5 * (1.0 + 0.5 * (1.0 + 0.5 * 1.0)), atol=1e-5)


def test_compute_returns_to_go_resets_at_dones():
    rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.array([False, False, True, False, False], dtype=bool)
    rtg = compute_returns_to_go(rewards, dones, gamma=1.0)
    assert rtg[2] == 1.0
    assert rtg[3] == 2.0
    assert rtg[4] == 1.0


def test_trajectory_defaults():
    t = Trajectory()
    assert len(t) == 0
    assert t.is_empty()


def test_trajectory_buffer_add_and_overflow():
    buf = TrajectoryBuffer(max_trajectories=2)
    for i in range(5):
        traj = Trajectory(
            obs=[_dummy_obs(i)],
            actions=np.array([0], dtype=np.int64),
            rewards=np.array([0.1], dtype=np.float32),
            dones=np.array([False], dtype=bool),
        )
        buf.add(traj)
    assert len(buf) == 2
    assert buf.total_steps() == 2
    buf.clear()
    assert len(buf) == 0


def test_window_dataset_yields_correct_shapes():
    trajs = []
    for i in range(2):
        T = 6
        trajs.append(Trajectory(
            obs=[_dummy_obs(i + 100 * t) for t in range(T)],
            actions=np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
            rewards=np.array([0.1] * T, dtype=np.float32),
            dones=np.array([False] * T, dtype=bool),
        ))
    ds = TrajectoryWindowDataset(trajs, context_length=3, n_actions=9)
    assert len(ds) == 12
    sample = ds[0]
    assert sample["rtg"].shape == (3,)
    assert sample["state"].shape == (3, 4)
    assert sample["action"].shape == (3,)
    assert sample["mask"].shape == (3,)
    assert int(sample["target_action"]) in (0, 1, 2)


def test_window_dataset_padding_uses_front_zeros():
    traj = Trajectory(
        obs=[_dummy_obs(t) for t in range(2)],
        actions=np.array([1, 2], dtype=np.int64),
        rewards=np.array([0.1, 0.1], dtype=np.float32),
        dones=np.array([False, False], dtype=bool),
    )
    ds = TrajectoryWindowDataset([traj], context_length=5)
    assert len(ds) == 2
    sample = ds[0]
    np.testing.assert_array_equal(sample["mask"], [False, False, False, False, True])
    np.testing.assert_array_equal(sample["action"], [0, 0, 0, 0, 1])
    np.testing.assert_allclose(sample["rtg"], [0, 0, 0, 0, 0.2], atol=1e-5)


def test_window_dataset_prefers_state_emb_over_numeric():
    traj = Trajectory(
        obs=[_dummy_obs(0)],
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([0.0], dtype=np.float32),
        dones=np.array([False], dtype=bool),
    )
    ds = TrajectoryWindowDataset([traj], context_length=1)
    sample = ds[0]
    assert sample["state"].shape == (1, 4)


def test_window_dataset_rejects_zero_context():
    with pytest.raises(ValueError):
        TrajectoryWindowDataset([], context_length=0)


def test_collect_trajectories_with_random_policy():
    """Smoke: roll a uniform policy in a tiny env-like stub."""
    class _StubEnv:
        def __init__(self):
            self._t = 0
            self._max = 5

        def reset(self, seed=None):
            self._t = 0
            return _dummy_obs(self._t), {}

        def step(self, action):
            self._t += 1
            obs = _dummy_obs(self._t)
            terminated = self._t >= self._max
            return obs, 0.1, terminated, False, {}

    env = _StubEnv()
    rng = np.random.default_rng(0)
    def pi(_o):
        return int(rng.integers(0, 9))
    trajs = collect_trajectories(env, pi, n_episodes=3, max_steps=10, seed=0)
    assert len(trajs) == 3
    for t in trajs:
        assert len(t) == 5
        assert t.dones[-1] is True or t.dones[-1] == True
