"""Contrastive representation learning utilities for regime intelligence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from zhisa.regime.dataset import RegimeSupervisionBatch


@dataclass(frozen=True)
class RegimeAugmentationConfig:
    """Feature-space augmentations for robust regime embeddings."""

    feature_dropout: float = 0.08
    continuous_noise: float = 0.025
    probability_noise: float = 0.01
    categorical_dropout: float = 0.0
    keep_scalar_prefixes: tuple[str, ...] = ("scalar.", "aggregate.", "context.", "probability.")
    dropout_protected_prefixes: tuple[str, ...] = ("scalar.", "aggregate.")

    def __post_init__(self) -> None:
        for name in ("feature_dropout", "continuous_noise", "probability_noise", "categorical_dropout"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.feature_dropout >= 1.0:
            raise ValueError(f"feature_dropout must be < 1, got {self.feature_dropout}")
        if self.categorical_dropout >= 1.0:
            raise ValueError(f"categorical_dropout must be < 1, got {self.categorical_dropout}")


@dataclass(frozen=True)
class RegimePositiveMaskConfig:
    """Controls what counts as a positive pair beyond two augmented views."""

    same_macro: bool = True
    same_meso: bool = False
    same_risk_mode: bool = False
    same_playbook: bool = True
    return_tolerance: float = 0.012
    drawdown_tolerance: float = 0.025
    vol_tolerance: float = 0.025

    def __post_init__(self) -> None:
        for name in ("return_tolerance", "drawdown_tolerance", "vol_tolerance"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


def _prefix_mask(
    feature_names: Sequence[str] | None,
    dim: int,
    device: torch.device,
    prefixes: Sequence[str],
    *,
    default: bool,
) -> torch.Tensor:
    if not feature_names:
        return torch.full((dim,), bool(default), dtype=torch.bool, device=device)
    values = [any(str(name).startswith(prefix) for prefix in prefixes) for name in feature_names]
    if len(values) != dim:
        return torch.full((dim,), bool(default), dtype=torch.bool, device=device)
    return torch.tensor(values, dtype=torch.bool, device=device)


def _continuous_mask(
    feature_names: Sequence[str] | None,
    dim: int,
    device: torch.device,
    prefixes: Sequence[str],
) -> torch.Tensor:
    return _prefix_mask(feature_names, dim, device, prefixes, default=True)


def augment_regime_features(
    x: torch.Tensor,
    cfg: RegimeAugmentationConfig | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a stochastic feature-space view while preserving tensor shape."""
    cfg = cfg or RegimeAugmentationConfig()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    out = x.clone()
    dim = out.size(-1)
    continuous = _continuous_mask(feature_names, dim, out.device, cfg.keep_scalar_prefixes)
    categorical = ~continuous
    protected_from_dropout = _prefix_mask(
        feature_names,
        dim,
        out.device,
        cfg.dropout_protected_prefixes,
        default=False,
    )
    dropout_target = continuous & ~protected_from_dropout

    if cfg.continuous_noise > 0:
        noise = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=generator)
        scale = cfg.continuous_noise
        if feature_names:
            prob = torch.tensor(
                [cfg.probability_noise if str(name).startswith("probability.") else scale for name in feature_names],
                dtype=out.dtype,
                device=out.device,
            )
            if prob.numel() == dim:
                out = out + noise * prob.unsqueeze(0) * continuous.to(out.dtype).unsqueeze(0)
            else:
                out = out + noise * scale * continuous.to(out.dtype).unsqueeze(0)
        else:
            out = out + noise * scale * continuous.to(out.dtype).unsqueeze(0)

    if cfg.feature_dropout > 0:
        keep = torch.rand(out.shape, device=out.device, generator=generator) >= cfg.feature_dropout
        keep = torch.where(dropout_target.unsqueeze(0), keep, torch.ones_like(keep, dtype=torch.bool))
        scale = torch.where(
            dropout_target.unsqueeze(0),
            torch.full_like(out, 1.0 / max(1.0 - cfg.feature_dropout, 1e-6)),
            torch.ones_like(out),
        )
        out = out * keep.to(out.dtype) * scale

    if cfg.categorical_dropout > 0 and bool(categorical.any()):
        keep_cat = torch.rand(out.shape, device=out.device, generator=generator) >= cfg.categorical_dropout
        keep_cat = torch.where(categorical.unsqueeze(0), keep_cat, torch.ones_like(keep_cat, dtype=torch.bool))
        out = out * keep_cat.to(out.dtype)

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, *, temperature: float = 0.12) -> torch.Tensor:
    """Symmetric InfoNCE/NT-Xent loss for two augmented views of a batch."""
    if z_a.shape != z_b.shape:
        raise ValueError(f"z_a and z_b must have identical shape, got {tuple(z_a.shape)} and {tuple(z_b.shape)}")
    if z_a.dim() != 2:
        raise ValueError(f"embeddings must be rank-2, got shape {tuple(z_a.shape)}")
    if z_a.size(0) < 2:
        return z_a.new_zeros(())
    temp = max(float(temperature), 1e-6)
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits_ab = a @ b.T / temp
    logits_ba = b @ a.T / temp
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))


def regime_positive_mask(
    batch: RegimeSupervisionBatch,
    cfg: RegimePositiveMaskConfig | None = None,
) -> torch.Tensor:
    """Build outcome-aware positive pairs for supervised contrastive learning."""
    cfg = cfg or RegimePositiveMaskConfig()
    device = batch.x.device
    n = int(batch.x.size(0))
    mask = torch.ones((n, n), dtype=torch.bool, device=device)
    if cfg.same_macro:
        mask &= batch.macro[:, None] == batch.macro[None, :]
    if cfg.same_meso:
        mask &= batch.meso[:, None] == batch.meso[None, :]
    if cfg.same_risk_mode:
        mask &= batch.risk_mode[:, None] == batch.risk_mode[None, :]
    if cfg.same_playbook:
        mask &= batch.playbook_label[:, None] == batch.playbook_label[None, :]
    if cfg.return_tolerance >= 0:
        mask &= (batch.forward_return[:, None] - batch.forward_return[None, :]).abs() <= cfg.return_tolerance
    if cfg.drawdown_tolerance >= 0:
        mask &= (batch.max_drawdown[:, None] - batch.max_drawdown[None, :]).abs() <= cfg.drawdown_tolerance
    if cfg.vol_tolerance >= 0:
        mask &= (batch.realized_vol[:, None] - batch.realized_vol[None, :]).abs() <= cfg.vol_tolerance
    mask.fill_diagonal_(False)
    return mask


def supervised_contrastive_loss(
    z: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.12,
) -> torch.Tensor:
    """Supervised contrastive loss using an explicit positive-pair mask."""
    if z.dim() != 2:
        raise ValueError(f"embeddings must be rank-2, got shape {tuple(z.shape)}")
    if positive_mask.shape != (z.size(0), z.size(0)):
        raise ValueError(
            "positive_mask must match embedding batch size: "
            f"got {tuple(positive_mask.shape)} for {z.size(0)} embeddings"
        )
    if z.size(0) < 2 or not bool(positive_mask.any()):
        return z.new_zeros(())

    temp = max(float(temperature), 1e-6)
    z_n = F.normalize(z, dim=-1)
    logits = z_n @ z_n.T / temp
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -torch.finfo(logits.dtype).max)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = positive_mask.to(dtype=z.dtype, device=z.device)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    mean_log_prob = (pos * log_prob).sum(dim=1)[valid] / pos_count[valid].to(z.dtype)
    return -mean_log_prob.mean()


__all__ = [
    "RegimeAugmentationConfig",
    "RegimePositiveMaskConfig",
    "augment_regime_features",
    "nt_xent_loss",
    "regime_positive_mask",
    "supervised_contrastive_loss",
]
