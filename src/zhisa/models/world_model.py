"""World Model: encoder + latent dynamics + reward + done heads.

The world model composes three pieces:

1. A **frozen** :class:`PolicyNetwork` that maps a multimodal
   observation (chart, numeric, context) to a latent ``z``.
2. A :class:`LatentDynamics` module that predicts ``z_{t+1}`` from
   ``(z_t, a_t, h_t)``.
3. A reward and done head (also part of :class:`LatentDynamics`).

The encoder is kept frozen to avoid the *moving target* problem
where an evolving encoder destabilises the dynamics module.
Pre-computed embeddings are stored in :class:`Trajectory` under
``obs["state_emb"]`` by :func:`embed_trajectories` and consumed
here directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from zhisa.models.latent_dynamics import LatentDynamics, LatentDynamicsConfig
from zhisa.models.policy import PolicyConfig, PolicyNetwork


@dataclass
class WorldModelConfig:
    state_dim: int = 128
    n_actions: int = 9
    dynamics_hidden: int = 128
    dynamics_layers: int = 1
    dynamics_dropout: float = 0.0

    @classmethod
    def from_policy_config(cls, pcfg: PolicyConfig) -> "WorldModelConfig":
        return cls(
            state_dim=int(pcfg.embed_dim),
            n_actions=int(pcfg.n_actions),
        )


class WorldModel(nn.Module):
    """The world model. Owns a :class:`LatentDynamics`; the encoder is *external*."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dynamics = LatentDynamics(LatentDynamicsConfig(
            state_dim=cfg.state_dim,
            n_actions=cfg.n_actions,
            hidden_dim=cfg.dynamics_hidden,
            n_layers=cfg.dynamics_layers,
            dropout=cfg.dynamics_dropout,
        ))

    # ----------------------------------------------------------------- single step
    def step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> dict:
        """Single-step ``(z, a, h) -> (z', h', r, d)``."""
        z_next, h_next, r_pred, d_logit = self.dynamics(z, action, h=h)
        return {
            "z_next": z_next,
            "h_next": h_next,
            "r_pred": r_pred,
            "d_logit": d_logit,
            "d_prob": torch.sigmoid(d_logit),
        }

    # ----------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(
        self,
        z0: torch.Tensor,        # (B, state_dim) or (B, T, state_dim)
        a_seq: torch.Tensor,     # (B, T) long
        h0: Optional[torch.Tensor] = None,
    ) -> dict:
        """Closed-loop rollout in latent space.

        ``a_seq`` is **given** (sampled from a behaviour policy or
        from the LatentActorCritic). The rollout predicts the
        resulting ``z`` trajectory, recurrent state trajectory, and
        the per-step reward and done probabilities.
        """
        if z0.dim() == 2:
            B, D = z0.shape
            T = a_seq.size(1)
            z0 = z0.unsqueeze(1).expand(B, T, D).contiguous()
        z_next_seq, h_final, r_seq, d_logit_seq = self.dynamics.forward_sequence(z0, a_seq, h0=h0)
        return {
            "z_seq": z_next_seq,
            "h_final": h_final,
            "r_seq": r_seq,
            "d_seq": torch.sigmoid(d_logit_seq),
            "d_logit_seq": d_logit_seq,
        }

    def predict_sequence(
        self,
        z_seq: torch.Tensor,       # (B, T, state_dim)
        action_seq: torch.Tensor,  # (B, T)
        h0: Optional[torch.Tensor] = None,
    ) -> dict:
        """Open-loop sequence prediction — used by :class:`WorldModelTrainer`."""
        z_next_seq, h_final, r_seq, d_logit_seq = self.dynamics.forward_sequence(z_seq, action_seq, h0=h0)
        return {
            "z_next_seq": z_next_seq,
            "h_final": h_final,
            "r_seq": r_seq,
            "d_logit_seq": d_logit_seq,
        }

    # ----------------------------------------------------------------- IO
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "model": self.state_dict(),
            "config": self.cfg.__dict__,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> "WorldModel":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = WorldModelConfig(**payload["config"])
        model = WorldModel(cfg)
        model.load_state_dict(payload["model"])
        return model


__all__ = ["WorldModel", "WorldModelConfig"]
