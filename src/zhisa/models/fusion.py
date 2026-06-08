"""Cross-modal fusion: combine vision, numeric, and context embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FusionConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1


class CrossModalFusion(nn.Module):
    """A small cross-attention Transformer that fuses three modalities."""

    def __init__(self, cfg: Optional[FusionConfig] = None) -> None:
        super().__init__()
        cfg = cfg or FusionConfig()
        self.cfg = cfg
        self.proj_vision = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.proj_numeric = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.proj_context = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def forward(
        self,
        v: torch.Tensor,
        n: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        v = self.proj_vision(v).unsqueeze(1)
        n = self.proj_numeric(n).unsqueeze(1)
        c = self.proj_context(c).unsqueeze(1)
        tokens = torch.cat([self.cls.expand(v.size(0), -1, -1), v, n, c], dim=1)
        out = self.encoder(tokens)
        out = self.norm(out)
        return self.out_proj(out[:, 0])
