"""Probes: simple tests that the representations are not random / not trivial."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from zhisa.models.policy import PolicyNetwork


@torch.no_grad()
def embedding_stats(emb: torch.Tensor) -> dict:
    if emb.ndim != 2:
        emb = emb.flatten(0, -2)
    norms = emb.norm(dim=-1)
    return {
        "mean_norm": float(norms.mean().item()),
        "std_norm": float(norms.std().item()),
        "mean_abs": float(emb.abs().mean().item()),
        "active_dims_frac": float((emb.std(dim=0) > 1e-3).float().mean().item()),
    }


@torch.no_grad()
def calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    if probs.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


@torch.no_grad()
def model_smoke_test(model: PolicyNetwork, batch: dict) -> dict:
    """Run a forward pass and return sanity checks."""
    out = model(chart=batch["chart"], numeric=batch["numeric"], context=batch["context"])
    return {
        "embedding_norm": float(out["embedding"].norm(dim=-1).mean().item()),
        "policy_logits_shape": tuple(out["policy_logits"].shape),
        "value_shape": tuple(out["value"].shape),
        "has_nan": bool(torch.isnan(out["embedding"]).any().item()),
    }
