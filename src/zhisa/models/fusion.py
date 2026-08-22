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


# ---------------------------------------------------------------------------
# Token-level cross-modal fusion (vision v2, concept C)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenFusionConfig:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 3
    dim_ff: int = 1536
    dropout: float = 0.1
    out_dim: int = 384
    n_types: int = 4


class TokenCrossFusion(nn.Module):
    """Joint transformer over token streams (vision columns + numeric patches
    + context (+ freq)), keeping bar-level cross-modal interplay.

    Each stream is ``(tensor (B, L, d), type_id, positions (B, L) long)`` —
    ``positions`` are *shared time indices* aligned across modalities (e.g.
    vision bar k -> k, numeric patch j covering bars [4j,4j+3] -> 4j). The
    cross-attention therefore sees the temporal correspondence directly
    (concept C, aligned temporal PE) on top of the type marker.
    """

    def __init__(self, cfg: Optional[TokenFusionConfig] = None):
        super().__init__()
        cfg = cfg or TokenFusionConfig()
        self.cfg = cfg
        self.type_emb = nn.Embedding(cfg.n_types, cfg.d_model)
        self.pos_pe = nn.Parameter(torch.zeros(4096, cfg.d_model))
        nn.init.normal_(self.pos_pe, mean=0.0, std=0.02)
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.out_dim)

    def forward(self, streams) -> torch.Tensor:
        """``streams``: sequence of ``(tensor (B, L, d), type_id, positions (B, L))``."""
        pieces, ids, poss = [], [], []
        for tensor, type_id, positions in streams:
            pieces.append(tensor)
            ids.append(torch.full((tensor.size(0), tensor.size(1)), int(type_id),
                                  dtype=torch.long, device=tensor.device))
            poss.append(positions.clamp(0, self.pos_pe.size(0) - 1))
        x = torch.cat(pieces, dim=1)
        t = torch.cat(ids, dim=1)
        p = torch.cat(poss, dim=1)
        x = x + self.type_emb(t) + self.pos_pe[p]
        out = self.encoder(x)
        out = self.norm(out)
        import math
        attn = torch.softmax(out @ self.query.transpose(1, 2) / math.sqrt(float(self.cfg.d_model)), dim=1)
        pooled = (out * attn).sum(dim=1)
        return self.out_proj(pooled)


# Shared time-index constants for aligned temporal PE.
POS_FREQ = 1024       # the low-frequency branch token
POS_CONTEXT = 2048    # the context token
