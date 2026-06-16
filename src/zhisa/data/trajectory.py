"""Trajectory data structures for offline RL (Decision Transformer).

A :class:`Trajectory` is a single episode of trading, captured as
aligned arrays of observations, actions and rewards. The
:class:`TrajectoryBuffer` aggregates many trajectories and exposes
fixed-length windows of ``(return-to-go, state, action)`` triples
suitable for sequence-modeling RL.

These utilities are intentionally narrow: no I/O, no env coupling.
The trainer / script is responsible for producing trajectories by
rolling a policy through :class:`TradingEnv`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Single trajectory
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One episode worth of (obs, action, reward) tuples.

    ``obs`` is a list of dicts, each containing at least
    ``"chart"`` (CHW tensor or ndarray), ``"numeric"`` (T x F) and
    ``"context"`` (C,) entries. ``actions`` is a 1-D ``int64`` array
    of :class:`DiscreteAction` indices. ``rewards`` is a 1-D
    ``float32`` array aligned with ``actions``. ``dones`` is a 1-D
    ``bool`` array aligned with ``actions`` (True at the terminal
    step of the episode).
    """

    obs: list[dict] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dones: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def is_empty(self) -> bool:
        return len(self) == 0


def compute_returns_to_go(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute the per-step return-to-go ``G_t = sum_{k>=t} gamma^k * r_{t+k}``.

    ``dones[t]`` is treated as a hard reset: the recursion is broken
    at terminal steps, so the post-terminal return is zero. This
    mirrors the convention used in :class:`zhisa.training.s4_rl.PPOTrainer`.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    T = rewards.shape[0]
    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            running = 0.0
        running = rewards[t] + float(gamma) * running
        rtg[t] = running
    return rtg


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryBuffer:
    """An append-only collection of :class:`Trajectory` episodes.

    The buffer does not enforce a capacity: trajectories are usually
    small and the user controls total size by limiting episode count.
    If a ``max_trajectories`` is set, the oldest entries are dropped
    FIFO once the limit is reached.
    """

    max_trajectories: Optional[int] = None
    _items: list[Trajectory] = field(default_factory=list)

    def add(self, traj: Trajectory) -> None:
        if traj.is_empty():
            return
        if self.max_trajectories is not None and len(self._items) >= self.max_trajectories:
            self._items.pop(0)
        self._items.append(traj)

    def extend(self, trajs: Iterable[Trajectory]) -> None:
        for t in trajs:
            self.add(t)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx: int) -> Trajectory:
        return self._items[idx]

    def total_steps(self) -> int:
        return sum(len(t) for t in self._items)

    def all_returns_to_go(self, gamma: float = 1.0) -> list[np.ndarray]:
        return [compute_returns_to_go(t.rewards, t.dones, gamma=gamma) for t in self._items]

    def clear(self) -> None:
        self._items.clear()


# ---------------------------------------------------------------------------
# Dataset of fixed-length windows
# ---------------------------------------------------------------------------


class TrajectoryWindowDataset(Dataset):
    """Yield fixed-length windows of ``(rtg, state, action)`` triples.

    Each window is built from a contiguous slice of one trajectory.
    A trajectory shorter than ``context_length`` is padded at the
    *front* with neutral tokens (rtg=0, action=SKIP) so the model
    sees a uniform ``(context_length,)`` input regardless of
    episode length.

    The state field is taken verbatim from ``obs[i]["numeric"]`` —
    a 2-D ``(T_window, F)`` array. Callers that want a richer state
    (e.g. embeddings from :class:`PolicyNetwork`) should pre-compute
    the embeddings and inject them as ``obs[i]["state_emb"]``. The
    dataset prefers ``"state_emb"`` if present, otherwise falls back
    to ``"numeric"``.

    Performance
    -----------
    All windows are precomputed in ``__init__`` and stored as numpy
    arrays. ``__getitem__`` is a pure dict access + ``torch.from_numpy``,
    which is already very fast. The class advertises
    ``__fast_getitem__ = True`` so the DataLoader factory picks
    ``num_workers=0`` (avoids IPC overhead for the trivial work per item).
    """

    # All windows are precomputed in __init__; the hot path is dict lookup.
    __fast_getitem__: bool = True

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        context_length: int = 20,
        gamma: float = 1.0,
        n_actions: int = 9,
    ) -> None:
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        self.context_length = int(context_length)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        # Pre-compute (rtg, states, actions) per trajectory.
        self._windows: list[dict] = []
        for traj in trajectories:
            if traj.is_empty():
                continue
            T = len(traj)
            rtg = compute_returns_to_go(traj.rewards, traj.dones, gamma=self.gamma)
            states: list[np.ndarray] = []
            for o in traj.obs:
                if "state_emb" in o:
                    states.append(np.asarray(o["state_emb"], dtype=np.float32))
                else:
                    # Fall back to flattened numeric features.
                    num = np.asarray(o["numeric"], dtype=np.float32)
                    states.append(num.reshape(-1))
            if not states:
                continue
            state_dim = states[0].shape[0]
            states_arr = np.stack(states, axis=0)
            actions_arr = np.asarray(traj.actions, dtype=np.int64)
            # Build per-step windows.
            for t in range(T):
                start = max(0, t - self.context_length + 1)
                win_len = t - start + 1
                pad = self.context_length - win_len
                if pad > 0:
                    rtg_win = np.concatenate([np.zeros(pad, dtype=np.float32), rtg[start : t + 1]])
                    state_win = np.concatenate(
                        [np.zeros((pad, state_dim), dtype=np.float32), states_arr[start : t + 1]],
                        axis=0,
                    )
                    act_win = np.concatenate([np.zeros(pad, dtype=np.int64), actions_arr[start : t + 1]])
                    mask = np.concatenate([np.zeros(pad, dtype=bool), np.ones(win_len, dtype=bool)])
                else:
                    rtg_win = rtg[start : t + 1]
                    state_win = states_arr[start : t + 1]
                    act_win = actions_arr[start : t + 1]
                    mask = np.ones(self.context_length, dtype=bool)
                self._windows.append({
                    "rtg": rtg_win,
                    "state": state_win,
                    "action": act_win,
                    "mask": mask,
                    "target_action": int(actions_arr[t]),
                    "target_rtg": float(rtg[t]),
                    "t": t,
                })

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        w = self._windows[idx]
        return {
            "rtg": torch.from_numpy(w["rtg"]).float(),
            "state": torch.from_numpy(w["state"]).float(),
            "action": torch.from_numpy(w["action"]).long(),
            "mask": torch.from_numpy(w["mask"]).bool(),
            "target_action": torch.tensor(w["target_action"], dtype=torch.long),
            "target_rtg": torch.tensor(w["target_rtg"], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Convenience: roll a callable policy in an env to build a buffer
# ---------------------------------------------------------------------------


def collect_trajectories(
    env,
    policy_fn,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> list[Trajectory]:
    """Roll ``policy_fn`` in ``env`` and return a list of trajectories.

    ``policy_fn`` is a callable ``policy_fn(obs) -> int`` that
    returns the next action (typically ``argmax`` of a model's
    ``policy_logits``). The collector is deterministic when the
    policy and the env are deterministic.
    """
    rng = np.random.default_rng(seed)
    out: list[Trajectory] = []
    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        traj = Trajectory()
        ep_ret = 0.0
        for _ in range(int(max_steps)):
            action = int(policy_fn(obs))
            traj.obs.append(obs)
            traj.actions = np.concatenate([traj.actions, np.array([action], dtype=np.int64)])
            obs, reward, terminated, truncated, _info = env.step(action)
            traj.rewards = np.concatenate([traj.rewards, np.array([float(reward)], dtype=np.float32)])
            traj.dones = np.concatenate([traj.dones, np.array([bool(terminated)], dtype=bool)])
            ep_ret += float(reward)
            if terminated or truncated:
                break
        out.append(traj)
    return out


__all__ = [
    "Trajectory",
    "TrajectoryBuffer",
    "TrajectoryWindowDataset",
    "collect_trajectories",
    "compute_returns_to_go",
]
