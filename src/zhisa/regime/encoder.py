"""Trainable encoder for structured regime reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from zhisa.regime.schema import MacroRegime, MesoRegime, RegimeReport, RiskMode
from zhisa.regime.vectorizer import RegimeFeatureVectorizer, RegimeVectorizerConfig


@dataclass(frozen=True)
class RegimeEncoderConfig:
    input_dim: int | None = None
    embed_dim: int = 32
    hidden_dim: int = 96
    n_playbooks: int = 24
    dropout: float = 0.1
    vectorizer: RegimeVectorizerConfig = field(default_factory=RegimeVectorizerConfig)


class RegimeEncoder(nn.Module):
    """MLP encoder over fixed regime feature vectors."""

    def __init__(
        self,
        cfg: RegimeEncoderConfig | None = None,
        *,
        vectorizer: RegimeFeatureVectorizer | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or RegimeEncoderConfig()
        self.vectorizer = vectorizer or RegimeFeatureVectorizer(self.cfg.vectorizer)
        input_dim = self.cfg.input_dim or self.vectorizer.dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.embed_dim),
            nn.LayerNorm(self.cfg.embed_dim),
            nn.GELU(),
        )
        self.macro_head = nn.Linear(self.cfg.embed_dim, len(tuple(MacroRegime)))
        self.meso_head = nn.Linear(self.cfg.embed_dim, len(tuple(MesoRegime)))
        self.risk_head = nn.Linear(self.cfg.embed_dim, len(tuple(RiskMode)))
        self.playbook_head = nn.Linear(self.cfg.embed_dim, self.cfg.n_playbooks)
        self.tradeability_head = nn.Linear(self.cfg.embed_dim, 1)
        self.transition_head = nn.Linear(self.cfg.embed_dim, 1)

    @property
    def input_dim(self) -> int:
        return int(self.cfg.input_dim or self.vectorizer.dim)

    def vectorize_reports(
        self,
        reports: Sequence[RegimeReport],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        arr = self.vectorizer.transform_many(reports)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor | np.ndarray | Sequence[RegimeReport]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        elif isinstance(x, torch.Tensor):
            x_t = x.to(device=device, dtype=torch.float32)
        else:
            x_t = self.vectorize_reports(x, device=device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        z = self.net(x_t)
        return {
            "embedding": z,
            "macro_logits": self.macro_head(z),
            "meso_logits": self.meso_head(z),
            "risk_logits": self.risk_head(z),
            "playbook_logits": self.playbook_head(z),
            "tradeability": torch.sigmoid(self.tradeability_head(z)).squeeze(-1),
            "transition_risk": torch.sigmoid(self.transition_head(z)).squeeze(-1),
        }


def append_regime_context(
    context: torch.Tensor | np.ndarray,
    regime_embedding: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Append regime embedding to an existing policy context vector."""
    if isinstance(context, torch.Tensor) or isinstance(regime_embedding, torch.Tensor):
        ctx = context if isinstance(context, torch.Tensor) else torch.as_tensor(context, dtype=torch.float32)
        reg = regime_embedding if isinstance(regime_embedding, torch.Tensor) else torch.as_tensor(regime_embedding, dtype=ctx.dtype, device=ctx.device)
        reg = reg.to(device=ctx.device, dtype=ctx.dtype)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        if reg.dim() == 1:
            reg = reg.unsqueeze(0)
        return torch.cat([ctx, reg], dim=-1)
    ctx_np = np.asarray(context, dtype=np.float32)
    reg_np = np.asarray(regime_embedding, dtype=np.float32)
    if ctx_np.ndim == 1:
        ctx_np = ctx_np[None, :]
    if reg_np.ndim == 1:
        reg_np = reg_np[None, :]
    return np.concatenate([ctx_np, reg_np], axis=-1)


__all__ = [
    "RegimeEncoder",
    "RegimeEncoderConfig",
    "append_regime_context",
]
