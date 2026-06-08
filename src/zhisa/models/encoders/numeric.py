"""Numeric encoder for multivariate time-series features.

A Patch-TST-lite: the input window is split into non-overlapping
patches, projected, augmented with positional embeddings, and passed
through a small Transformer encoder. The output is a single embedding
plus per-patch token sequence (for interpretability and downstream
heads).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import torch
import torch.nn as nn


@dataclass
class NumericEncoderConfig:
    in_features: int = 32
    window: int = 32
    patch_size: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_ff: int = 256
    dropout: float = 0.1
    out_dim: int = 128


class _SinPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NumericEncoder(nn.Module):
    """PatchTST-style numeric encoder."""

    def __init__(self, cfg: Optional[NumericEncoderConfig] = None) -> None:
        super().__init__()
        cfg = cfg or NumericEncoderConfig()
        self.cfg = cfg
        if cfg.window % cfg.patch_size != 0:
            raise ValueError("window must be divisible by patch_size")
        self.n_patches = cfg.window // cfg.patch_size
        self.patch_proj = nn.Linear(cfg.in_features * cfg.patch_size, cfg.d_model)
        self.pos = _SinPositionalEmbedding(cfg.d_model, max_len=self.n_patches + 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        B, T, F_ = x.shape
        if T != self.cfg.window or F_ != self.cfg.in_features:
            raise ValueError(
                f"Expected (B, {self.cfg.window}, {self.cfg.in_features}); got ({B}, {T}, {F_})"
            )
        # Patchify: (B, n_patches, patch_size*F)
        patches = x.view(B, self.n_patches, self.cfg.patch_size, F_).reshape(B, self.n_patches, -1)
        tokens = self.patch_proj(patches)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos(tokens)
        out = self.encoder(tokens)
        out = self.norm(out)
        cls_out = out[:, 0]
        return self.out_proj(cls_out), out
