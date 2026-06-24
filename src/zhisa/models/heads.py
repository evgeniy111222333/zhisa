"""Multi-task heads: direction, volatility, regime, risk, policy, value."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadsConfig:
    embed_dim: int = 128
    n_direction_classes: int = 3   # -1, 0, +1
    n_regime_classes: int = 4
    n_actions: int = 9             # discrete actions
    n_market_horizons: int = 0
    hidden_dim: int = 128
    dropout: float = 0.1


class MultiTaskHeads(nn.Module):
    """A set of heads that share the trunk embedding."""

    def __init__(self, cfg: Optional[HeadsConfig] = None) -> None:
        super().__init__()
        cfg = cfg or HeadsConfig()
        self.cfg = cfg
        self.shared = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.direction = nn.Linear(cfg.hidden_dim, cfg.n_direction_classes)
        self.regime = nn.Linear(cfg.hidden_dim, cfg.n_regime_classes)
        self.volatility = nn.Linear(cfg.hidden_dim, 1)
        self.risk = nn.Linear(cfg.hidden_dim, 1)
        self.return_pred = nn.Linear(cfg.hidden_dim, 1)
        self.direction_multi = (
            nn.Linear(cfg.hidden_dim, cfg.n_market_horizons * cfg.n_direction_classes)
            if cfg.n_market_horizons > 0 else None
        )
        self.return_multi = (
            nn.Linear(cfg.hidden_dim, cfg.n_market_horizons)
            if cfg.n_market_horizons > 0 else None
        )
        self.policy_logits = nn.Linear(cfg.hidden_dim, cfg.n_actions)
        self.value = nn.Linear(cfg.hidden_dim, 1)
        self.uncertainty_logit = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> dict:
        h = self.shared(z)
        out = {
            "direction": self.direction(h),
            "regime": self.regime(h),
            "volatility": self.volatility(h).squeeze(-1),
            "risk": self.risk(h).squeeze(-1),
            "return_pred": self.return_pred(h).squeeze(-1),
            "policy_logits": self.policy_logits(h),
            "value": self.value(h).squeeze(-1),
            "uncertainty_logit": self.uncertainty_logit(h).squeeze(-1),
        }
        if self.direction_multi is not None:
            out["direction_multi"] = self.direction_multi(h).view(
                z.size(0),
                self.cfg.n_market_horizons,
                self.cfg.n_direction_classes,
            )
        if self.return_multi is not None:
            out["return_multi"] = self.return_multi(h)
        return out
