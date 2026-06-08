"""Multi-instrument policy network with shared encoders and factored heads.

The :class:`PortfolioPolicyNetwork` accepts a list of per-instrument
observations plus a portfolio summary vector. It encodes each
instrument with a **shared** stack of encoders (vision + numeric +
context + cross-modal fusion), concatenates the per-instrument
embeddings with the portfolio summary, and emits a factored
action distribution: ``N`` independent heads of size
``n_actions_per`` (default 9) plus a single value head.

This design replaces the exponentially-large ``9**N`` flat action
head with a linear-in-N factored head, at the cost of losing the
ability to model cross-instrument joint action probabilities
explicitly. Cross-instrument dependence is still present in the
shared embedding: the portfolio summary encodes the current
positions, gross leverage, covariance, etc.

**Stage 2 — cross-instrument attention**

Set ``cross_attn_depth > 0`` to enable a stack of
:class:`CrossInstrumentAttention` blocks that run over the ``N``
per-instrument embeddings before the portfolio MLP. This lets each
instrument's policy head attend to all other instruments directly,
breaking the Stage 1 implicit-only coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from zhisa.models.cross_instrument_attention import (
    CrossInstrumentAttention,
    CrossInstrumentConfig,
)
from zhisa.models.encoders.context import ContextEncoder, ContextEncoderConfig
from zhisa.models.encoders.numeric import NumericEncoder, NumericEncoderConfig
from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig
from zhisa.models.fusion import CrossModalFusion, FusionConfig


@dataclass
class PortfolioPolicyConfig:
    """Configuration for :class:`PortfolioPolicyNetwork`."""

    n_instruments: int = 2
    in_numeric_features: int = 32
    in_context_features: int = 10
    window: int = 16
    image_size: int = 32
    embed_dim: int = 64
    n_actions_per: int = 9
    portfolio_dim: int = 32
    fusion_hidden: int = 128
    fusion_layers: int = 2
    n_regime_classes: int = 4
    dropout: float = 0.1
    cross_attn_depth: int = 0
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.0
    cross_attn_feedforward_mult: int = 4
    field_overrides: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_instruments < 1:
            raise ValueError(f"n_instruments must be >= 1, got {self.n_instruments}")
        if self.cross_attn_depth < 0:
            raise ValueError(
                f"cross_attn_depth must be >= 0, got {self.cross_attn_depth}"
            )
        if self.cross_attn_depth > 0:
            if self.embed_dim % self.cross_attn_heads != 0:
                raise ValueError(
                    f"embed_dim={self.embed_dim} must be divisible by "
                    f"cross_attn_heads={self.cross_attn_heads}"
                )


class PortfolioPolicyNetwork(nn.Module):
    """Shared-encoder portfolio policy with N factored action heads.

    Forward signature::

        forward(instruments: dict, portfolio: Tensor) -> dict

    ``instruments`` is a dict with keys ``chart``, ``numeric``,
    ``context``, each shaped ``(B, N, ...)``. ``portfolio`` is
    ``(B, portfolio_dim)``.

    Returns a dict with:

    * ``action_logits`` — ``(B, N, n_actions_per)`` factored logits
    * ``value``         — ``(B,)`` value estimate
    * ``per_instrument_embedding`` — ``(B, N, embed_dim)`` per-instrument z
    * ``portfolio_embedding`` — ``(B, fusion_hidden)`` fused representation
    """

    def __init__(self, cfg: Optional[PortfolioPolicyConfig] = None) -> None:
        super().__init__()
        cfg = cfg or PortfolioPolicyConfig()
        self.cfg = cfg
        # Shared per-instrument encoders.
        self.vision = VisionEncoder(VisionEncoderConfig(
            image_size=cfg.image_size, out_dim=cfg.embed_dim,
            channels=(32, 64, 128, 192), dropout=cfg.dropout,
        ))
        self.numeric = NumericEncoder(NumericEncoderConfig(
            in_features=cfg.in_numeric_features, window=cfg.window,
            d_model=cfg.embed_dim, out_dim=cfg.embed_dim,
            n_layers=2, dropout=cfg.dropout,
        ))
        self.context = ContextEncoder(ContextEncoderConfig(
            in_dim=cfg.in_context_features, out_dim=cfg.embed_dim,
            n_instruments=1, dropout=cfg.dropout,
        ))
        self.fusion = CrossModalFusion(FusionConfig(
            embed_dim=cfg.embed_dim, n_layers=cfg.fusion_layers,
            dropout=cfg.dropout,
        ))
        # Stage 2: cross-instrument attention (default off).
        if cfg.cross_attn_depth > 0:
            self.cross_attn = CrossInstrumentAttention(CrossInstrumentConfig(
                embed_dim=cfg.embed_dim, depth=cfg.cross_attn_depth,
                n_heads=cfg.cross_attn_heads, dropout=cfg.cross_attn_dropout,
                feedforward_mult=cfg.cross_attn_feedforward_mult,
                n_instruments_max=max(cfg.n_instruments, 8),
            ))
            self.cross_attn.set_portfolio_dim(cfg.portfolio_dim)
        else:
            self.cross_attn = None
        # Portfolio fusion: N instrument embeds + portfolio summary.
        in_dim = cfg.n_instruments * cfg.embed_dim + cfg.portfolio_dim
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden),
            nn.GELU(),
        )
        # N independent action heads + 1 value head.
        self.action_heads = nn.ModuleList([
            nn.Linear(cfg.fusion_hidden, cfg.n_actions_per)
            for _ in range(cfg.n_instruments)
        ])
        self.value_head = nn.Linear(cfg.fusion_hidden, 1)
        # Optional regime head per instrument for compatibility.
        self.regime_head = nn.Linear(cfg.fusion_hidden, cfg.n_regime_classes)

    def encode_instruments(
        self,
        chart: torch.Tensor,    # (B, N, 3, H, W)
        numeric: torch.Tensor,  # (B, N, T, F)
        context: torch.Tensor,  # (B, N, C)
    ) -> torch.Tensor:
        B, N = chart.shape[:2]
        v = self.vision(chart.reshape(B * N, *chart.shape[2:]))
        n, _ = self.numeric(numeric.reshape(B * N, *numeric.shape[2:]))
        c = self.context(context.reshape(B * N, *context.shape[2:]))
        z = self.fusion(v, n, c)
        return z.reshape(B, N, -1)

    def forward(
        self,
        instruments: dict,
        portfolio: torch.Tensor,
    ) -> dict:
        embeds = self.encode_instruments(
            instruments["chart"], instruments["numeric"], instruments["context"]
        )
        if self.cross_attn is not None:
            embeds = self.cross_attn(embeds, portfolio=portfolio)
        B = embeds.size(0)
        flat = torch.cat([embeds.reshape(B, -1), portfolio], dim=-1)
        fused = self.portfolio_mlp(flat)
        logits = torch.stack([h(fused) for h in self.action_heads], dim=1)  # (B, N, A)
        value = self.value_head(fused).squeeze(-1)
        regime = self.regime_head(fused)
        return {
            "action_logits": logits,
            "value": value,
            "per_instrument_embedding": embeds,
            "portfolio_embedding": fused,
            "regime_logits": regime,
        }

    # ------------------------------------------------------------------ IO
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        import torch as _torch
        payload = {"model": self.state_dict(), "config": self.cfg.__dict__}
        if extra:
            payload.update(extra)
        _torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> "PortfolioPolicyNetwork":
        import torch as _torch
        payload = _torch.load(path, map_location=map_location, weights_only=False)
        cfg = PortfolioPolicyConfig(**payload["config"])
        model = PortfolioPolicyNetwork(cfg)
        model.load_state_dict(payload["model"])
        return model


def build_default_portfolio_policy(
    n_instruments: int,
    in_numeric_features: int,
    in_context_features: int,
    window: int,
    image_size: int,
    portfolio_dim: int = 32,
    embed_dim: int = 64,
    **kwargs,
) -> PortfolioPolicyNetwork:
    cfg = PortfolioPolicyConfig(
        n_instruments=int(n_instruments),
        in_numeric_features=int(in_numeric_features),
        in_context_features=int(in_context_features),
        window=int(window),
        image_size=int(image_size),
        portfolio_dim=int(portfolio_dim),
        embed_dim=int(embed_dim),
        **kwargs,
    )
    return PortfolioPolicyNetwork(cfg)


__all__ = [
    "PortfolioPolicyConfig",
    "PortfolioPolicyNetwork",
    "build_default_portfolio_policy",
]
