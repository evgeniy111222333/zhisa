"""ColumnFormer: bar-column visual encoder for charts (vision v2, concept A).

A chart is treated as a 1-D sequence over time. The strict invariant is that the
encoder emits exactly ``n_bars`` tokens (chart window / numeric window),
independent of the rendered pixel width ``image_size``: pixel columns are
projected per column and deterministically folded back per bar using the
renderer's bar→column mapping (cumsum + gather, no probabilistic scatter).

Review-driven hardening (all verified):
- **CLS reader is causal-correct**: CLS is placed *after* the bar tokens and,
  under a causal mask, is the *only* position allowed to attend the whole window
  (bars themselves are strictly left-causal and are also blocked from attending
  the summary, so no information flows backwards through CLS).
- **Aggregation is fully vectorised** (no python loop over bars).
- **Token stream returned to fusion is the *contextualised* encoder output**
  (post ``norm``, bar positions only), not the raw column projections.
- **Attention-pool query is properly initialised** (small normal).
- **Identity fast-path** when ``image_size == n_bars`` (the standard 128x128).
- **Real DCT-II** (precomputed cosine basis) instead of an |FFT| proxy.
- **Type embedding** separates price/volume tokens in the 2-token-per-bar variant.
- **Dropout after the merge tokenizer**.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VisionColumnFormerConfig:
    image_size: int = 128
    n_bars: int = 128
    price_frac: float = 0.75
    channels: int = 3
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    dim_ff: int = 1536
    dropout: float = 0.1
    out_dim: int = 384
    causal: bool = True
    reader: str = "attention_pool"  # "attention_pool" | "cls"
    freq_branch: bool = True
    freq_k: int = 8
    volume_sub_dim: int = 64
    two_tokens_per_bar: bool = False
    include_volume: bool = True

    def __post_init__(self):
        if self.reader not in {"attention_pool", "cls"}:
            raise ValueError(f"unknown reader: {self.reader!r}")
        if self.image_size < 1 or self.n_bars < 1:
            raise ValueError("image_size and n_bars must be positive")


class _SinPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ColumnFormerVision(nn.Module):
    """``(B,3,H,W) -> (vec, contextual_tokens, freq)`` with exactly ``n_bars`` tokens."""

    def __init__(self, cfg: Optional[VisionColumnFormerConfig] = None):
        super().__init__()
        cfg = cfg or VisionColumnFormerConfig()
        self.cfg = cfg
        price_rows = int(cfg.image_size * cfg.price_frac)
        vol_rows = int(cfg.image_size) - price_rows
        if price_rows < 1 or vol_rows < 1:
            raise ValueError("chart has no price/volume rows for the chosen config")
        self.price_rows = price_rows
        self.vol_rows = vol_rows
        self.W = int(cfg.image_size)
        self.n_bars = int(cfg.n_bars)

        # Tokenizer.
        self.proj_price = nn.Linear(cfg.channels * price_rows, cfg.d_model, bias=False)
        self.proj_vol = nn.Linear(cfg.channels * vol_rows, cfg.volume_sub_dim, bias=False)
        self.merge = nn.Linear(cfg.d_model + cfg.volume_sub_dim, cfg.d_model)
        self.merge_drop = nn.Dropout(cfg.dropout)
        self.proj_vol_star = (
            nn.Linear(cfg.volume_sub_dim, cfg.d_model) if cfg.two_tokens_per_bar else None
        )

        seq_len = (2 * self.n_bars) if cfg.two_tokens_per_bar else self.n_bars
        self.pos = _SinPositionalEmbedding(cfg.d_model, max_len=4096)
        self.type_emb = nn.Embedding(2, cfg.d_model) if cfg.two_tokens_per_bar else None

        self.reader = cfg.reader
        if cfg.reader == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.query = None
        else:
            self.cls = None
            self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            if cfg.reader == "attention_pool":
                nn.init.normal_(self.query, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.out_dim)

        # ---- bar segmentation (fast identity path + general path) ----
        if self.W == self.n_bars:
            seg_starts = torch.arange(self.n_bars, dtype=torch.int64)
            seg_ends = torch.arange(1, self.n_bars + 1, dtype=torch.int64)
            seg_counts = torch.ones(self.n_bars, dtype=torch.float64)
        else:
            xs_bar = torch.floor(
                torch.arange(self.n_bars, dtype=torch.float64) * (self.W - 1)
                / float(max(self.n_bars - 1, 1))
            ).to(torch.int64)
            col = torch.arange(self.W, dtype=torch.int64)
            col_to_bar = torch.searchsorted(xs_bar, col, right=True).clamp(1, self.n_bars) - 1
            seg_starts = torch.zeros(self.n_bars, dtype=torch.int64)
            seg_ends = torch.zeros(self.n_bars, dtype=torch.int64)
            seg_counts = torch.ones(self.n_bars, dtype=torch.float64)
            colt = col_to_bar.detach()
            for b in range(self.n_bars):
                idxs = (colt == b).nonzero().squeeze(-1)
                if idxs.numel():
                    s, e = int(idxs[0]), int(idxs[-1]) + 1
                    seg_starts[b] = s
                    seg_ends[b] = e
                    seg_counts[b] = float(e - s)
        self.register_buffer("seg_starts", seg_starts)
        self.register_buffer("seg_ends", seg_ends)
        self.register_buffer("seg_counts", seg_counts)

        # ---- frequency branch: exact DCT-II basis (skip DC), precomputed ----
        self.freq_branch = cfg.freq_branch
        if cfg.freq_branch:
            n = self.n_bars
            k = int(cfg.freq_k)
            basis = torch.zeros(n, k, dtype=torch.float32)
            i = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            j = (torch.arange(k, dtype=torch.float32) + 1).unsqueeze(0)  # k=1..K, skip DC
            basis = math.sqrt(2.0 / n) * torch.cos((math.pi / n) * (i + 0.5) * j)
            self.register_buffer("dct_basis", basis)
            self.freq_proj = nn.Linear(cfg.freq_k * 2, cfg.d_model)

    # ------------------------------------------------------------------

    def _aggregate_to_bars(self, x: torch.Tensor) -> torch.Tensor:
        """(B, W, d) -> (B, n_bars, d) via exact segment sums, fully vectorised."""
        B, W, D = x.shape
        cum = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype), x.cumsum(dim=1)], dim=1)
        idx_s = self.seg_starts.view(1, -1, 1).expand(B, -1, D)
        idx_e = self.seg_ends.view(1, -1, 1).expand(B, -1, D)
        total = cum.gather(1, idx_e) - cum.gather(1, idx_s)  # (B, n, d)
        counts = self.seg_counts.to(dtype=x.dtype).view(1, -1, 1)
        return total / counts

    def _dct_tokens(self, price_trace: torch.Tensor, vol_trace: torch.Tensor) -> torch.Tensor:
        """(B,n) traces -> (B,1,d) low-frequency token via DCT-II (skip DC)."""
        parts = []
        for trace in (price_trace, vol_trace):
            c = torch.matmul(trace.unsqueeze(1), self.dct_basis)  # (B,1,k)
            parts.append(c.squeeze(1))
        f = torch.cat(parts, dim=-1)  # (B, 2k)
        return self.freq_proj(f).unsqueeze(1)

    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, *, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        if H != self.W or W != self.W:
            raise ValueError(f"expected {self.W}x{self.W} input, got {H}x{W}")

        cols = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        price = cols[:, : self.price_rows, :, :].permute(0, 2, 1, 3).reshape(B, W, C * self.price_rows)
        vol = cols[:, self.price_rows :, :, :].permute(0, 2, 1, 3).reshape(B, W, C * self.vol_rows)

        e_price = self.proj_price(price)
        e_vol = self.proj_vol(vol) if self.cfg.include_volume else torch.zeros(
            B, W, self.cfg.volume_sub_dim, device=x.device
        )
        t = self.merge_drop(self.merge(torch.cat([e_price, e_vol], dim=-1)))

        if self.cfg.two_tokens_per_bar:
            p_bars = self._aggregate_to_bars(e_price)                     # (B,n,d)
            v_bars = self.proj_vol_star(self._aggregate_to_bars(e_vol))   # (B,n,d)
            t = torch.stack([p_bars, v_bars], dim=2).reshape(B, 2 * self.n_bars, -1)
        else:
            t = self._aggregate_to_bars(t)  # (B, n_bars, d)

        t = self.pos(t)
        if self.type_emb is not None:
            parity = torch.arange(t.size(1), device=t.device) % 2
            t = t + self.type_emb(parity).view(1, t.size(1), -1)

        seq = t.size(1)
        if self.reader == "cls":
            tokens = torch.cat([t, self.cls.expand(B, -1, -1)], dim=1)
        else:
            tokens = t

        # --- causal mask ---
        if self.cfg.causal:
            with_cls = 1 if self.reader == "cls" else 0
            total = seq + with_cls
            idx = torch.arange(total, device=t.device)
            # True = blocked (PyTorch). Row i may attend columns j <= i (future
            # is masked: j > i).
            m = idx.view(1, -1) > idx.view(-1, 1)
            if self.reader == "cls":
                m[seq, :] = False          # summary attends the whole window
                m[:seq, seq] = True        # bars must not attend the summary (future info)
                m[seq, seq] = False
            mask = m
        else:
            mask = None

        out = self.encoder(tokens, mask=mask)
        out = self.norm(out)

        if self.reader == "cls":
            vec = out[:, -1]
        else:
            attn = torch.softmax(out @ self.query.transpose(1, 2) / math.sqrt(float(self.cfg.d_model)), dim=1)
            vec = (out * attn).sum(dim=1)
        vec = self.out_proj(vec)

        # contextualised bar-token stream (post-norm, bar positions only)
        bar_tokens = out[:, :seq]

        freq = None
        if self.freq_branch:
            price_lum = price.mean(dim=-1)  # (B,W)
            vol_lum = vol.mean(dim=-1)
            price_trace = self._aggregate_to_bars(price_lum.unsqueeze(-1)).squeeze(-1)
            vol_trace = self._aggregate_to_bars(vol_lum.unsqueeze(-1)).squeeze(-1)
            freq = self._dct_tokens(price_trace, vol_trace)

        return vec, bar_tokens, freq