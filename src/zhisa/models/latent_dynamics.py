"""Recurrent latent dynamics for the World Model.

The dynamics module owns the recurrent state ``h`` and consumes a
concatenation of the current latent state ``z`` and a one-hot action
``a``. It predicts the next latent state ``z'`` and feeds the same
``z'`` to the reward and done heads in
:class:`zhisa.models.world_model.WorldModel`.

The architecture is intentionally minimal — a single GRU with a
small MLP head. This matches the temporal coherence of OHLCV
windows (volatility clustering, regime persistence) and avoids the
overhead of a Transformer for the short horizons (H ≤ 50) used in
Dyna-style imagination.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentDynamicsConfig:
    state_dim: int = 128
    n_actions: int = 9
    hidden_dim: int = 128
    n_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")


class LatentDynamics(nn.Module):
    """GRU-based latent dynamics: ``(z, a, h) -> (z', h')``."""

    def __init__(self, cfg: LatentDynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_emb = nn.Embedding(cfg.n_actions, cfg.n_actions)
        self.input_proj = nn.Linear(cfg.state_dim + cfg.n_actions, cfg.hidden_dim)
        self.gru = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.next_state = nn.Linear(cfg.hidden_dim, cfg.state_dim)
        self.reward = nn.Linear(cfg.hidden_dim, 1)
        self.done = nn.Linear(cfg.hidden_dim, 1)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.cfg.n_layers, int(batch_size), self.cfg.hidden_dim, device=device)

    def forward(
        self,
        z: torch.Tensor,            # (B, state_dim)
        action: torch.Tensor,       # (B,) long
        h: torch.Tensor | None = None,  # (n_layers, B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step dynamics.

        Returns ``(z_next, h_next, r_pred, d_logit)``.
        """
        B = z.size(0)
        if h is None:
            h = self.initial_state(B, device=z.device)
        a_oh = self.action_emb(action)              # (B, n_actions)
        x = torch.cat([z, a_oh], dim=-1)            # (B, state_dim + n_actions)
        x = F.gelu(self.input_proj(x)).unsqueeze(1)  # (B, 1, hidden_dim)
        out, h_next = self.gru(x, h)                 # (B, 1, hidden_dim), (n_layers, B, hidden_dim)
        out = self.norm(out.squeeze(1))              # (B, hidden_dim)
        z_next = self.next_state(out)                # (B, state_dim)
        r_pred = self.reward(out).squeeze(-1)        # (B,)
        d_logit = self.done(out).squeeze(-1)         # (B,)
        return z_next, h_next, r_pred, d_logit

    def forward_sequence(
        self,
        z_seq: torch.Tensor,       # (B, T, state_dim)
        action_seq: torch.Tensor,  # (B, T) long
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence forward — used during world-model training.

        Returns ``(z_next_seq, h_final, r_seq, d_logit_seq)`` with the
        time dimension preserved. ``z_next_seq[:, t]`` is the
        prediction for time ``t+1`` given inputs at time ``t``.
        """
        B, T, _ = z_seq.shape
        if h0 is None:
            h0 = self.initial_state(B, device=z_seq.device)
        a_oh = self.action_emb(action_seq)            # (B, T, n_actions)
        x = torch.cat([z_seq, a_oh], dim=-1)          # (B, T, state_dim + n_actions)
        x = F.gelu(self.input_proj(x))                # (B, T, hidden_dim)
        out, h_final = self.gru(x, h0)                # (B, T, hidden_dim)
        out = self.norm(out)
        z_next_seq = self.next_state(out)             # (B, T, state_dim)
        r_seq = self.reward(out).squeeze(-1)          # (B, T)
        d_logit_seq = self.done(out).squeeze(-1)      # (B, T)
        return z_next_seq, h_final, r_seq, d_logit_seq


__all__ = ["LatentDynamics", "LatentDynamicsConfig"]
