"""Cross-instrument attention for multi-instrument portfolio policies.

Stage 2 of the portfolio policy: instrument tokens attend to each
other so the per-instrument policy/value heads receive context that
includes information about *all* instruments in the portfolio, not
just the focal one. The shared encoder in Stage 1 already aggregates
information through the portfolio summary vector; cross-attention
adds a direct, content-based pathway.

The module is intentionally minimal:

* :class:`CrossInstrumentAttention` is a stack of
  ``nn.TransformerEncoderLayer`` blocks operating on the
  ``(B, N, embed_dim)`` instrument-token tensor. It is
  permutation-equivariant (an instrument's output is invariant to
  the order in which the others are presented).
* An optional *instrument-id embedding* (instrument index -> learned
  vector) is added to the tokens before the first attention block,
  so the model can break the permutation symmetry when needed.
* An optional *portfolio summary bias* (B, embed_dim) is added to
  every token after the attention stack, so the portfolio state can
  bias the policy heads even without going through the MLP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CrossInstrumentConfig:
    """Configuration for :class:`CrossInstrumentAttention`."""

    embed_dim: int = 64
    depth: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    use_instrument_id: bool = True
    n_instruments_max: int = 8
    feedforward_mult: int = 4
    norm_first: bool = True
    field_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.n_heads <= 0 or self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must divide embed_dim={self.embed_dim}"
            )


class CrossInstrumentAttention(nn.Module):
    """Stacked bidirectional self-attention over instrument tokens.

    Input shape: ``(B, N, embed_dim)`` — instrument embeddings.
    Output shape: ``(B, N, embed_dim)`` — cross-instrument context.

    With ``depth=0`` the module is the identity (apart from the
    optional instrument-id embedding and portfolio bias). This
    makes it cheap to default-enable in the policy config and let
    Stage 1 callers explicitly set ``depth=0`` to disable.
    """

    def __init__(self, cfg: Optional[CrossInstrumentConfig] = None) -> None:
        super().__init__()
        cfg = cfg or CrossInstrumentConfig()
        self.cfg = cfg
        if cfg.use_instrument_id:
            self.instrument_id = nn.Embedding(cfg.n_instruments_max, cfg.embed_dim)
        else:
            self.instrument_id = None
        if cfg.depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.embed_dim * cfg.feedforward_mult,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=cfg.norm_first,
            )
            self.layers = nn.TransformerEncoder(
                layer, num_layers=cfg.depth, enable_nested_tensor=False,
            )
        else:
            self.layers = None
        self.norm: Optional[nn.LayerNorm] = None
        self.portfolio_proj: Optional[nn.Linear] = None

    def set_portfolio_dim(self, portfolio_dim: int) -> None:
        """Enable the optional portfolio bias projection.

        Call this once after construction if you want the model to
        use the portfolio summary vector as an additive bias on
        every instrument token.
        """
        if self.portfolio_proj is None and portfolio_dim > 0:
            self.portfolio_proj = nn.Linear(portfolio_dim, self.cfg.embed_dim)

    def _add_instrument_id(self, x: torch.Tensor) -> torch.Tensor:
        if self.instrument_id is None:
            return x
        B, N, D = x.shape
        if N > self.cfg.n_instruments_max:
            raise ValueError(
                f"N={N} > n_instruments_max={self.cfg.n_instruments_max}; "
                "raise n_instruments_max in the config or pre-pad instruments."
            )
        ids = torch.arange(N, device=x.device)
        return x + self.instrument_id(ids).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,                       # (B, N, D)
        portfolio: Optional[torch.Tensor] = None,  # (B, portfolio_dim)
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D) input, got {tuple(x.shape)}")
        if x.size(-1) != self.cfg.embed_dim:
            raise ValueError(
                f"input embed_dim={x.size(-1)} != config embed_dim={self.cfg.embed_dim}"
            )
        y = self._add_instrument_id(x)
        if self.layers is not None:
            y = self.layers(y)
        if portfolio is not None and self.portfolio_proj is not None:
            y = y + self.portfolio_proj(portfolio).unsqueeze(1)
        return y


__all__ = ["CrossInstrumentAttention", "CrossInstrumentConfig"]
