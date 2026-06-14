"""Visual regime encoder and chart-supervision dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig
from zhisa.regime.dataset import (
    PLAYBOOK_NAMES,
    RegimeSupervisionBatch,
    RegimeSupervisionConfig,
    RegimeSupervisionDataset,
    RegimeSupervisionItem,
    regime_supervision_collate,
)
from zhisa.regime.schema import MacroRegime, MesoRegime, RiskMode
from zhisa.rendering.chart_renderer import render_chart


@dataclass(frozen=True)
class VisualRegimeConfig:
    image_size: int = 64
    chart_window: int = 96
    vision_dim: int = 96
    embed_dim: int = 32
    hidden_dim: int = 96
    n_playbooks: int = len(PLAYBOOK_NAMES)
    vision_channels: tuple[int, ...] = (24, 48, 96)
    dropout: float = 0.1
    tabular_embed_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.chart_window <= 1:
            raise ValueError(f"chart_window must be > 1, got {self.chart_window}")
        if self.embed_dim <= 0 or self.vision_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("vision_dim, embed_dim, and hidden_dim must be positive")
        if self.n_playbooks <= 0:
            raise ValueError(f"n_playbooks must be positive, got {self.n_playbooks}")


@dataclass(frozen=True)
class VisualRegimeSupervisionConfig:
    chart_window: int = 96
    image_size: int = 64
    cache_charts: bool = True
    base: RegimeSupervisionConfig = field(default_factory=RegimeSupervisionConfig)


@dataclass(frozen=True)
class VisualRegimeSupervisionItem:
    chart: torch.Tensor
    supervision: RegimeSupervisionItem


@dataclass(frozen=True)
class VisualRegimeSupervisionBatch:
    chart: torch.Tensor
    supervision: RegimeSupervisionBatch


@dataclass(frozen=True)
class VisualRegimeLossWeights:
    macro: float = 1.0
    meso: float = 0.75
    risk_mode: float = 0.5
    playbook: float = 0.4
    tradeability: float = 0.5
    transition_risk: float = 0.5


class VisualRegimeSupervisionDataset(Dataset):
    """Chart windows aligned to structured regime labels and forward outcomes."""

    def __init__(self, df, cfg: Optional[VisualRegimeSupervisionConfig] = None) -> None:
        self.df = df
        self.cfg = cfg or VisualRegimeSupervisionConfig()
        self.supervision = RegimeSupervisionDataset(df, self.cfg.base)
        self._chart_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.supervision)

    def _chart(self, t: int) -> torch.Tensor:
        if self.cfg.cache_charts and t in self._chart_cache:
            return self._chart_cache[t]
        start = max(0, int(t) - int(self.cfg.chart_window) + 1)
        window = self.df.iloc[start : int(t) + 1]
        chart = render_chart(window, size=int(self.cfg.image_size))
        if self.cfg.cache_charts:
            self._chart_cache[t] = chart
        return chart

    def __getitem__(self, idx: int) -> VisualRegimeSupervisionItem:
        item = self.supervision[idx]
        chart = self._chart(int(item.meta["t"]))
        return VisualRegimeSupervisionItem(chart=chart, supervision=item)


def visual_regime_collate(items: list[VisualRegimeSupervisionItem]) -> VisualRegimeSupervisionBatch:
    return VisualRegimeSupervisionBatch(
        chart=torch.stack([it.chart for it in items]),
        supervision=regime_supervision_collate([it.supervision for it in items]),
    )


def fuse_regime_embeddings(
    structured: torch.Tensor,
    visual: torch.Tensor,
    gate: torch.Tensor | float,
) -> torch.Tensor:
    """Blend structured and visual regime embeddings with a [0, 1] gate."""
    if structured.shape != visual.shape:
        raise ValueError(f"structured and visual embeddings must match, got {structured.shape} and {visual.shape}")
    gate_t = gate if isinstance(gate, torch.Tensor) else torch.tensor(float(gate), device=structured.device, dtype=structured.dtype)
    gate_t = gate_t.to(device=structured.device, dtype=structured.dtype)
    if gate_t.dim() == 0:
        gate_t = gate_t.view(1, 1)
    if gate_t.dim() == 1:
        gate_t = gate_t.unsqueeze(-1)
    return torch.clamp(gate_t, 0.0, 1.0) * structured + (1.0 - torch.clamp(gate_t, 0.0, 1.0)) * visual


class VisualRegimeEncoder(nn.Module):
    """Chart-image regime model with optional structured-embedding fusion."""

    def __init__(self, cfg: Optional[VisualRegimeConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or VisualRegimeConfig()
        self.vision = VisionEncoder(
            VisionEncoderConfig(
                image_size=self.cfg.image_size,
                out_dim=self.cfg.vision_dim,
                channels=self.cfg.vision_channels,
                dropout=self.cfg.dropout,
            )
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(self.cfg.vision_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.embed_dim),
            nn.LayerNorm(self.cfg.embed_dim),
            nn.GELU(),
        )
        if self.cfg.tabular_embed_dim is not None:
            self.tabular_proj = nn.Sequential(
                nn.Linear(int(self.cfg.tabular_embed_dim), self.cfg.embed_dim),
                nn.LayerNorm(self.cfg.embed_dim),
                nn.GELU(),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.cfg.embed_dim * 2, self.cfg.embed_dim),
                nn.GELU(),
                nn.Linear(self.cfg.embed_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.tabular_proj = None
            self.fusion_gate = None

        self.macro_head = nn.Linear(self.cfg.embed_dim, len(tuple(MacroRegime)))
        self.meso_head = nn.Linear(self.cfg.embed_dim, len(tuple(MesoRegime)))
        self.risk_head = nn.Linear(self.cfg.embed_dim, len(tuple(RiskMode)))
        self.playbook_head = nn.Linear(self.cfg.embed_dim, self.cfg.n_playbooks)
        self.tradeability_head = nn.Linear(self.cfg.embed_dim, 1)
        self.transition_head = nn.Linear(self.cfg.embed_dim, 1)

    def forward(
        self,
        chart: torch.Tensor,
        *,
        tabular_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_embedding = self.visual_proj(self.vision(chart))
        embedding = visual_embedding
        out: dict[str, torch.Tensor] = {"visual_embedding": visual_embedding}
        if tabular_embedding is not None:
            if self.tabular_proj is None or self.fusion_gate is None:
                raise ValueError("VisualRegimeConfig.tabular_embed_dim must be set to fuse tabular embeddings")
            tab = self.tabular_proj(tabular_embedding.to(device=visual_embedding.device, dtype=visual_embedding.dtype))
            gate = self.fusion_gate(torch.cat([tab, visual_embedding], dim=-1))
            embedding = fuse_regime_embeddings(tab, visual_embedding, gate)
            out["tabular_embedding"] = tab
            out["fusion_gate"] = gate.squeeze(-1)
        out.update({
            "embedding": embedding,
            "macro_logits": self.macro_head(embedding),
            "meso_logits": self.meso_head(embedding),
            "risk_logits": self.risk_head(embedding),
            "playbook_logits": self.playbook_head(embedding),
            "tradeability": torch.sigmoid(self.tradeability_head(embedding)).squeeze(-1),
            "transition_risk": torch.sigmoid(self.transition_head(embedding)).squeeze(-1),
        })
        probs = torch.softmax(out["macro_logits"], dim=-1)
        out["visual_uncertainty"] = 1.0 - probs.max(dim=-1).values
        return out


class VisualRegimeLoss(nn.Module):
    """Multi-task loss for chart-derived regime predictions."""

    def __init__(self, weights: Optional[VisualRegimeLossWeights] = None) -> None:
        super().__init__()
        self.weights = weights or VisualRegimeLossWeights()

    def forward(self, outputs: dict[str, torch.Tensor], batch: RegimeSupervisionBatch) -> dict[str, torch.Tensor]:
        losses = {
            "macro": F.cross_entropy(outputs["macro_logits"], batch.macro),
            "meso": F.cross_entropy(outputs["meso_logits"], batch.meso),
            "risk_mode": F.cross_entropy(outputs["risk_logits"], batch.risk_mode),
            "playbook": F.cross_entropy(outputs["playbook_logits"], batch.playbook_label),
            "tradeability": F.smooth_l1_loss(outputs["tradeability"], batch.tradeability),
            "transition_risk": F.smooth_l1_loss(outputs["transition_risk"], batch.transition_risk),
        }
        total = torch.zeros((), device=outputs["embedding"].device)
        for name, loss in losses.items():
            total = total + float(getattr(self.weights, name)) * loss
        losses["total"] = total
        return losses


__all__ = [
    "VisualRegimeConfig",
    "VisualRegimeEncoder",
    "VisualRegimeLoss",
    "VisualRegimeLossWeights",
    "VisualRegimeSupervisionBatch",
    "VisualRegimeSupervisionConfig",
    "VisualRegimeSupervisionDataset",
    "VisualRegimeSupervisionItem",
    "fuse_regime_embeddings",
    "visual_regime_collate",
]
