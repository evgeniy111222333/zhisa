"""Dream environment — a gym wrapper around :class:`WorldModel`.

The dream env lets a downstream RL agent be trained **purely in
imagination** without ever touching the real market. It mimics
the :class:`TradingEnv` interface (gymnasium: ``reset`` and
``step`` returning a dict observation) but the observation is a
latent vector produced by the world model, not a multimodal
market snapshot.

Observation schema::

    obs = {
        "z": (state_dim,) float32,         # current latent
        "h": (n_layers, hidden_dim) float32,  # recurrent state
    }

The agent that operates in :class:`DreamEnv` must consume ``z``
directly (e.g. :class:`zhisa.models.latent_actor_critic.LatentActorCritic`).
The :class:`PolicyNetwork` multimodal policy is **not** used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from zhisa.models.world_model import WorldModel
from zhisa.utils.seeding import set_seed


@dataclass
class DreamEnvConfig:
    n_actions: int = 9
    state_dim: int = 128
    dynamics_hidden: int = 128
    dynamics_layers: int = 1
    horizon: int = 50           # cap on episode length inside the dream
    initial_states: int = 16    # number of (z, h) pairs to seed from
    device: str = "cpu"
    seed: int = 0


class DreamEnv(gym.Env):
    """A minimal gym wrapper around :class:`WorldModel`.

    On ``reset()`` a (z, h) is sampled uniformly from the supplied
    pool of *initial* states (e.g. states collected from a real
    trajectory). On ``step(action)`` the world model advances one
    step and returns a new (z, h), the predicted reward, and a
    done flag. The done flag is the ``OR`` of the WM's predicted
    ``d_prob > 0.5`` and the horizon cap being reached.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_model: WorldModel,
        initial_pool: np.ndarray | torch.Tensor,  # (N, state_dim)
        initial_h: Optional[np.ndarray | torch.Tensor] = None,  # (N, n_layers, hidden_dim)
        cfg: Optional[DreamEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.cfg = cfg or DreamEnvConfig(
            n_actions=int(world_model.cfg.n_actions),
            state_dim=int(world_model.cfg.state_dim),
            dynamics_hidden=int(world_model.cfg.dynamics_hidden),
            dynamics_layers=int(world_model.cfg.dynamics_layers),
        )
        self.world_model = self.world_model.to(self.cfg.device).eval()
        if isinstance(initial_pool, torch.Tensor):
            self._pool_z = initial_pool.detach().cpu().float()
        else:
            self._pool_z = torch.from_numpy(np.asarray(initial_pool, dtype=np.float32))
        if initial_h is None:
            self._pool_h = torch.zeros(
                self._pool_z.size(0), self.cfg.dynamics_layers, self.cfg.dynamics_hidden,
            )
        else:
            if isinstance(initial_h, torch.Tensor):
                self._pool_h = initial_h.detach().cpu().float()
            else:
                self._pool_h = torch.from_numpy(np.asarray(initial_h, dtype=np.float32))
        assert self._pool_h.size(0) == self._pool_z.size(0), "z and h pools must have the same length"
        self.action_space = spaces.Discrete(self.cfg.n_actions)
        self.observation_space = spaces.Dict({
            "z": spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.state_dim,), dtype=np.float32),
            "h": spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.dynamics_layers, self.cfg.dynamics_hidden), dtype=np.float32),
        })
        self._rng = np.random.default_rng(self.cfg.seed)
        self._z: Optional[torch.Tensor] = None
        self._h: Optional[torch.Tensor] = None
        self._t: int = 0

    def _obs(self) -> dict:
        z_np = self._z.detach().cpu().numpy().astype(np.float32)
        if z_np.ndim == 2 and z_np.shape[0] == 1:
            z_np = z_np.squeeze(0)
        h_np = self._h.detach().cpu().numpy().astype(np.float32)
        if h_np.ndim == 3 and h_np.shape[1] == 1:
            h_np = h_np.squeeze(1)
        return {"z": z_np, "h": h_np}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            set_seed(int(seed))
        idx = int(self._rng.integers(0, self._pool_z.size(0)))
        # Add a batch dim of 1 so h is always 3D (n_layers, B, H).
        self._z = self._pool_z[idx].clone().to(self.cfg.device).unsqueeze(0)
        self._h = self._pool_h[idx].clone().to(self.cfg.device).unsqueeze(1)
        self._t = 0
        return self._obs(), {}

    def step(self, action: int):
        if self._z is None or self._h is None:
            raise RuntimeError("DreamEnv.step() called before reset()")
        a = torch.tensor(int(action), dtype=torch.long, device=self.cfg.device).view(1)
        with torch.no_grad():
            out = self.world_model.step(self._z, a, h=self._h)
        self._z = out["z_next"]
        self._h = out["h_next"]
        reward = float(out["r_pred"].item())
        d_pred = bool(out["d_prob"].item() > 0.5)
        self._t += 1
        truncated = self._t >= self.cfg.horizon
        terminated = d_pred and not truncated
        return self._obs(), reward, terminated, truncated, {"d_prob": float(out["d_prob"].item())}


__all__ = ["DreamEnv", "DreamEnvConfig"]
