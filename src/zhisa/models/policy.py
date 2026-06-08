"""End-to-end policy network: encoders + fusion + memory + heads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from zhisa.models.encoders.context import ContextEncoder, ContextEncoderConfig
from zhisa.models.encoders.numeric import NumericEncoder, NumericEncoderConfig
from zhisa.models.encoders.vision import VisionEncoder, VisionEncoderConfig
from zhisa.models.fusion import CrossModalFusion, FusionConfig
from zhisa.models.heads import HeadsConfig, MultiTaskHeads
from zhisa.models.memory import MemoryConfig, WorkingMemory


@dataclass
class PolicyConfig:
    image_size: int = 64
    in_numeric_features: int = 32
    window: int = 32
    in_context_features: int = 10
    embed_dim: int = 128
    n_actions: int = 9
    n_regime_classes: int = 4
    n_instruments: int = 1
    use_memory: bool = True
    memory_max_len: int = 64
    fusion_layers: int = 2
    memory_layers: int = 2
    vision_channels: tuple = (32, 64, 128, 192)
    dropout: float = 0.1
    field_overrides: dict = field(default_factory=dict)


class PolicyNetwork(nn.Module):
    """The end-to-end multimodal policy / feature extractor."""

    def __init__(self, cfg: Optional[PolicyConfig] = None) -> None:
        super().__init__()
        cfg = cfg or PolicyConfig()
        self.cfg = cfg
        self.vision = VisionEncoder(VisionEncoderConfig(
            image_size=cfg.image_size, out_dim=cfg.embed_dim,
            channels=cfg.vision_channels, dropout=cfg.dropout,
        ))
        self.numeric = NumericEncoder(NumericEncoderConfig(
            in_features=cfg.in_numeric_features, window=cfg.window,
            d_model=cfg.embed_dim, out_dim=cfg.embed_dim,
            n_layers=2, dropout=cfg.dropout,
        ))
        self.context = ContextEncoder(ContextEncoderConfig(
            in_dim=cfg.in_context_features, out_dim=cfg.embed_dim,
            n_instruments=cfg.n_instruments, dropout=cfg.dropout,
        ))
        self.fusion = CrossModalFusion(FusionConfig(
            embed_dim=cfg.embed_dim, n_layers=cfg.fusion_layers,
            dropout=cfg.dropout,
        ))
        self.memory = (
            WorkingMemory(MemoryConfig(
                embed_dim=cfg.embed_dim, n_layers=cfg.memory_layers,
                max_len=cfg.memory_max_len, dropout=cfg.dropout,
            )) if cfg.use_memory else None
        )
        self.heads = MultiTaskHeads(HeadsConfig(
            embed_dim=cfg.embed_dim, n_actions=cfg.n_actions,
            n_regime_classes=cfg.n_regime_classes, dropout=cfg.dropout,
        ))

    def encode(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        instrument_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v = self.vision(chart)
        n, _ = self.numeric(numeric)
        c = self.context(context, instrument_id=instrument_id)
        z = self.fusion(v, n, c)
        return z

    def forward(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            chart:    (B, 3, H, W)
            numeric:  (B, T, F)
            context:  (B, C)
            history:  optional (B, S, D) rolling memory; updated internally.
            instrument_id: optional (B,) long.
        """
        z = self.encode(chart, numeric, context, instrument_id=instrument_id)
        if self.memory is not None and history is not None:
            z_with_hist = torch.cat([history, z.unsqueeze(1)], dim=1)
            out_seq = self.memory(z_with_hist)
            z = out_seq[:, -1]
            next_history = out_seq[:, -self.memory.cfg.max_len:]
        else:
            next_history = z.unsqueeze(1) if self.memory is not None else None
        out = self.heads(z)
        out["embedding"] = z
        out["next_history"] = next_history
        return out


def build_default_policy(
    in_numeric_features: int = 32,
    in_context_features: int = 10,
    n_actions: int = 9,
    n_regime_classes: int = 4,
    **kwargs,
) -> PolicyNetwork:
    cfg = PolicyConfig(
        in_numeric_features=in_numeric_features,
        in_context_features=in_context_features,
        n_actions=n_actions,
        n_regime_classes=n_regime_classes,
        **kwargs,
    )
    return PolicyNetwork(cfg)
