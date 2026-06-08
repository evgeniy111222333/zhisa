"""Context encoder: combines time embeddings, instrument id, and any scalar features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ContextEncoderConfig:
    in_dim: int = 10
    out_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    n_instruments: int = 1
    instrument_emb_dim: int = 16


class ContextEncoder(nn.Module):
    """MLP over the context vector, with optional instrument embedding."""

    def __init__(self, cfg: Optional[ContextEncoderConfig] = None) -> None:
        super().__init__()
        cfg = cfg or ContextEncoderConfig()
        self.cfg = cfg
        self.instrument_emb = nn.Embedding(cfg.n_instruments, cfg.instrument_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim + cfg.instrument_emb_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, instrument_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if instrument_id is None:
            instrument_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        emb = self.instrument_emb(instrument_id)
        h = torch.cat([x, emb], dim=-1)
        return self.net(h)
