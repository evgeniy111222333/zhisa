"""Working memory: a sequence model that maintains temporal context over bars."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MemoryConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    max_len: int = 512


class WorkingMemory(nn.Module):
    """A causal Transformer sequence layer."""

    def __init__(self, cfg: Optional[MemoryConfig] = None) -> None:
        super().__init__()
        cfg = cfg or MemoryConfig()
        self.cfg = cfg
        self.pos = nn.Embedding(cfg.max_len, cfg.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        if T > self.cfg.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.cfg.max_len}")
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos(positions)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        out = self.encoder(x, mask=mask, is_causal=True)
        return self.norm(out)
