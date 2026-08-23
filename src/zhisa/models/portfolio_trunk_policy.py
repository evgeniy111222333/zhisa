"""Portfolio policy over the SHARED S1 trunk (S4-portfolio integration).

The legacy :class:`PortfolioPolicyNetwork` trains its own small encoders from
scratch; this module reuses the S1/S2 trunk — the exact PolicyNetwork trained
by the SSL pipeline (ColumnFormer + token fusion + cross-asset numeric) — as a
weight-shared per-instrument encoder, then adds:

* optional :class:`CrossInstrumentAttention` over instrument tokens,
* optional additive correlation bias (cross_asset Corr as attention bias),
* per-instrument factored action heads + portfolio value head,
* ``warm_start_from_s1``: shape-filtered transfer of the S1 weights.

Inputs follow the ``(B, N, ...)`` convention, aligned at one timestamp t, so
no lookahead is possible (all instruments see the same snapshot).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from zhisa.models.cross_instrument_attention import (
    CrossInstrumentAttention,
    CrossInstrumentConfig,
)
from zhisa.models.policy import PolicyConfig, PolicyNetwork


@dataclass
class PortfolioTrunkConfig:
    """Configuration for :class:`PortfolioTrunkPolicy`."""

    # Backbone PolicyConfig (its window/image_size/in_numeric_features must
    # MATCH the prepared data / rendered charts used at training time).
    backbone: Optional[PolicyConfig] = None
    use_memory: bool = False
    n_instruments: int = 2
    portfolio_dim: int = 32
    fusion_hidden: int = 128
    n_actions_per: int = 9
    n_regime_classes: int = 4
    # Cross-instrument stage
    cross_attn_depth: int = 2
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.0
    use_attention_bias: bool = False
    bias_gate: float = 1.0
    # Pooling for the value head / portfolio embedding
    pooling: str = "mean"  # "mean" | "cls" (cls = first instrument token)

    def __post_init__(self) -> None:
        if self.n_instruments < 1:
            raise ValueError("n_instruments must be >= 1")
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling {self.pooling!r}")


class PortfolioTrunkPolicy(nn.Module):
    """Shared-S1-trunk portfolio policy."""

    def __init__(self, cfg: Optional[PortfolioTrunkConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PortfolioTrunkConfig()
        bc = self.cfg.backbone or PolicyConfig()
        bc = PolicyConfig(**{**bc.__dict__, "use_memory": self.cfg.use_memory})
        bc.n_instruments = max(bc.n_instruments, self.cfg.n_instruments)
        self.trunk = PolicyNetwork(bc)
        if self.cfg.cross_attn_depth > 0:
            self.cross_attn = CrossInstrumentAttention(CrossInstrumentConfig(
                embed_dim=bc.embed_dim,
                depth=self.cfg.cross_attn_depth,
                n_heads=self.cfg.cross_attn_heads,
                dropout=self.cfg.cross_attn_dropout,
                use_instrument_id=True,
                n_instruments_max=max(self.cfg.n_instruments, 8),
                use_attention_bias=self.cfg.use_attention_bias,
                bias_gate=self.cfg.bias_gate,
            ))
            self.cross_attn.set_portfolio_dim(self.cfg.portfolio_dim)
        else:
            self.cross_attn = None
        D = bc.embed_dim
        self.head_shared = nn.Sequential(
            nn.Linear(D, self.cfg.fusion_hidden),
            nn.GELU(),
        )
        self.action_heads = nn.ModuleList([
            nn.Linear(self.cfg.fusion_hidden, self.cfg.n_actions_per)
            for _ in range(self.cfg.n_instruments)
        ])
        self.pool_proj = nn.Linear(D, self.cfg.fusion_hidden)
        self.value_head = nn.Linear(self.cfg.fusion_hidden, 1)
        self.regime_head = nn.Linear(self.cfg.fusion_hidden, self.cfg.n_regime_classes)

    def embed_instruments(
        self,
        chart: torch.Tensor,    # (B, N, 3, H, W)
        numeric: torch.Tensor,  # (B, N, T, F)
        context: torch.Tensor,  # (B, N, C)
        instrument_ids: Optional[torch.Tensor] = None,  # (B, N) symbol ids
    ) -> torch.Tensor:
        B, N = chart.shape[:2]
        flat = dict(
            chart=chart.reshape(B * N, *chart.shape[2:]),
            numeric=numeric.reshape(B * N, *numeric.shape[2:]),
            context=context.reshape(B * N, *context.shape[2:]),
        )
        if instrument_ids is not None:
            ids = instrument_ids.reshape(B * N).to(flat["chart"].device)
        else:
            ids = torch.arange(N, device=flat["chart"].device).repeat(B)
        z = self.trunk.encode(
            flat["chart"], flat["numeric"], flat["context"], instrument_id=ids
        )
        return z.view(B, N, -1)

    def forward(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        portfolio: Optional[torch.Tensor] = None,
        instrument_ids: Optional[torch.Tensor] = None,
        corr_bias: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward on aligned instrument snapshots.

        ``corr_bias`` (B, N, N): optional cross-asset correlation matrix used
        as additive attention bias when enabled.
        """
        z = self.embed_instruments(chart, numeric, context, instrument_ids)
        B, N, D = z.shape
        if N != self.cfg.n_instruments:
            raise ValueError(
                f"bundle size N={N} != cfg.n_instruments={self.cfg.n_instruments}"
            )
        if self.cross_attn is not None:
            z_ctx = self.cross_attn(z, portfolio=portfolio, bias=corr_bias)
        else:
            if corr_bias is not None:
                raise ValueError("corr_bias requires cross_attn_depth > 0")
            z_ctx = z
        if self.cfg.pooling == "mean":
            pooled = z_ctx.mean(dim=1)
        else:
            pooled = z_ctx[:, 0]
        h = self.head_shared(z_ctx)
        action_logits = torch.stack(
            [head(h[:, i]) for i, head in enumerate(self.action_heads)], dim=1
        )  # (B, N, n_actions_per)
        pool_h = self.pool_proj(pooled)
        return {
            "action_logits": action_logits,
            "value": self.value_head(pool_h).squeeze(-1),
            "regime_logits": self.regime_head(pool_h),
            "embedding": z_ctx,
            "pooled": pooled,
        }

    def _trunk_state_map(self, s1_state: dict) -> dict:
        """Map an S1 checkpoint's state dict onto the trunk (shape-filtered).

        The trunk is a DIRECT child module, so keys keep their bare form
        (``vision.conv.0.weight``), matching the trunk's own state_dict.
        """
        ref = {k: v.shape for k, v in self.trunk.named_parameters()}
        ref.update({k: v.shape for k, v in self.trunk.named_buffers()})
        out = {}
        for k, v in s1_state.items():
            base = k.split(".")[0]
            if base not in ("vision", "numeric", "context", "fusion"):
                continue
            if k in ref and tuple(v.shape) == tuple(ref[k]):
                out[k] = v
        return out

    def warm_start_from_s1(self, s1_checkpoint: str | Path) -> dict:
        """Copy the S1 encoder weights into the trunk.

        Returns stats: ``{"copied": int, "total_trunk": int}``.
        """
        payload = torch.load(s1_checkpoint, map_location="cpu", weights_only=False)
        s1_state = payload["model"] if "model" in payload else payload
        mapped = self._trunk_state_map(s1_state)
        missing, unexpected = self.trunk.load_state_dict(mapped, strict=False)
        total = sum(1 for _ in ref_keys(self.trunk)
                    if _.split(".")[0] in ("vision", "numeric", "context", "fusion"))
        return {"copied": len(mapped), "total_trunk": total,
                "missing": list(missing)[:5], "unexpected": list(unexpected)[:5]}


def ref_keys(model: nn.Module):
    for k, _ in model.named_parameters():
        yield k
    for k, _ in model.named_buffers():
        yield k


__all__ = ["PortfolioTrunkConfig", "PortfolioTrunkPolicy"]