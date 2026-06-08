"""A simple actor-critic that operates directly on latent states.

The :class:`LatentActorCritic` is the imagination-only agent: it
takes the latent ``z`` produced by :class:`zhisa.models.world_model.WorldModel`
and outputs ``(action_logits, value)``. It does not consume the
multimodal chart/numeric/context observations — those are
handled by the :class:`PolicyNetwork` in the real env.

The architecture is intentionally small (2 hidden layers, GELU)
to keep the imagination loop fast and to make PPO updates
cheap on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentActorCriticConfig:
    state_dim: int = 128
    n_actions: int = 9
    hidden_dim: int = 64
    n_hidden_layers: int = 1

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.n_hidden_layers < 0:
            raise ValueError("hidden_dim must be positive and n_hidden_layers >= 0")


class LatentActorCritic(nn.Module):
    """``z -> (action_logits, value)`` actor-critic."""

    def __init__(self, cfg: LatentActorCriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        actor_layers: list[nn.Module] = [nn.Linear(cfg.state_dim, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.n_hidden_layers):
            actor_layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        actor_layers += [nn.Linear(cfg.hidden_dim, cfg.n_actions)]
        self.actor = nn.Sequential(*actor_layers)
        critic_layers: list[nn.Module] = [nn.Linear(cfg.state_dim, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.n_hidden_layers):
            critic_layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        critic_layers += [nn.Linear(cfg.hidden_dim, 1)]
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    def act(self, z: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob)``."""
        logits, _ = self.forward(z)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            a = logits.argmax(dim=-1)
            lp = dist.log_prob(a)
            return a, lp
        a = dist.sample()
        lp = dist.log_prob(a)
        return a, lp

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(z).squeeze(-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(z)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()


__all__ = ["LatentActorCritic", "LatentActorCriticConfig"]
