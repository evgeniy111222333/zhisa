"""Policy wrapper that injects Regime Intelligence into context."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.regime.encoder import RegimeEncoder, RegimeEncoderConfig, append_regime_context
from zhisa.regime.schema import RegimeReport


@dataclass
class RegimeAwarePolicyConfig:
    base_policy: PolicyConfig = field(default_factory=PolicyConfig)
    regime_encoder: RegimeEncoderConfig = field(default_factory=RegimeEncoderConfig)
    append_regime_embedding: bool = True
    freeze_regime_encoder: bool = False


class RegimeAwarePolicyNetwork(nn.Module):
    """PolicyNetwork with a trainable regime embedding appended to context."""

    def __init__(self, cfg: Optional[RegimeAwarePolicyConfig] = None) -> None:
        super().__init__()
        cfg = cfg or RegimeAwarePolicyConfig()
        self.cfg = cfg
        self.regime_encoder = RegimeEncoder(cfg.regime_encoder)
        extra_context = cfg.regime_encoder.embed_dim if cfg.append_regime_embedding else 0
        policy_cfg = replace(
            cfg.base_policy,
            in_context_features=cfg.base_policy.in_context_features + extra_context,
        )
        self.policy = PolicyNetwork(policy_cfg)
        if cfg.freeze_regime_encoder:
            for param in self.regime_encoder.parameters():
                param.requires_grad_(False)

    def _encode_regime(
        self,
        context: torch.Tensor,
        regime: torch.Tensor | np.ndarray | Sequence[RegimeReport] | None,
        regime_embedding: torch.Tensor | np.ndarray | None,
    ) -> dict[str, torch.Tensor]:
        if regime_embedding is not None:
            emb = regime_embedding if isinstance(regime_embedding, torch.Tensor) else torch.as_tensor(regime_embedding, dtype=context.dtype, device=context.device)
            emb = emb.to(device=context.device, dtype=context.dtype)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return {"embedding": emb}
        if regime is None:
            zeros = torch.zeros(
                context.size(0),
                self.regime_encoder.input_dim,
                device=context.device,
                dtype=context.dtype,
            )
            return self.regime_encoder(zeros)
        return self.regime_encoder(regime)

    def forward(
        self,
        chart: torch.Tensor,
        numeric: torch.Tensor,
        context: torch.Tensor,
        *,
        regime: torch.Tensor | np.ndarray | Sequence[RegimeReport] | None = None,
        regime_embedding: torch.Tensor | np.ndarray | None = None,
        history: Optional[torch.Tensor] = None,
        instrument_id: Optional[torch.Tensor] = None,
    ) -> dict:
        regime_out = self._encode_regime(context, regime, regime_embedding)
        policy_context = context
        if self.cfg.append_regime_embedding:
            policy_context = append_regime_context(context, regime_out["embedding"])
        out = self.policy(
            chart=chart,
            numeric=numeric,
            context=policy_context,
            history=history,
            instrument_id=instrument_id,
        )
        out["regime_embedding"] = regime_out["embedding"]
        for key, value in regime_out.items():
            if key != "embedding":
                out[f"regime_{key}"] = value
        return out


def build_regime_aware_policy(
    in_numeric_features: int = 32,
    in_context_features: int = 10,
    n_actions: int = 9,
    n_regime_classes: int = 4,
    regime_embed_dim: int = 32,
    **kwargs,
) -> RegimeAwarePolicyNetwork:
    base_cfg = PolicyConfig(
        in_numeric_features=in_numeric_features,
        in_context_features=in_context_features,
        n_actions=n_actions,
        n_regime_classes=n_regime_classes,
        **kwargs,
    )
    cfg = RegimeAwarePolicyConfig(
        base_policy=base_cfg,
        regime_encoder=RegimeEncoderConfig(embed_dim=regime_embed_dim),
    )
    return RegimeAwarePolicyNetwork(cfg)


__all__ = [
    "RegimeAwarePolicyConfig",
    "RegimeAwarePolicyNetwork",
    "build_regime_aware_policy",
]
