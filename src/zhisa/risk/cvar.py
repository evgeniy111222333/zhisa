"""Conditional Value at Risk (CVaR) — also known as Expected Shortfall.

CVaR at level ``alpha`` is the expected return in the worst
``alpha``-fraction of cases. For a returns distribution with
negative tail, CVaR is a *negative* number whose magnitude is the
average loss in the worst tail.

This module exposes both a NumPy (non-differentiable) and a PyTorch
(differentiable via piecewise-linear ``sort``) variant, plus a
helper that quantifies how badly a constraint of the form
``-CVaR_alpha <= threshold`` is being violated.
"""
from __future__ import annotations

import numpy as np
import torch


def _resolve_alpha(alpha: float, n: int) -> float:
    return float(np.clip(alpha, 1.0 / max(n, 1), 1.0))


def cvar_numpy(returns: np.ndarray, alpha: float = 0.1) -> float:
    """Mean of the worst ``alpha``-fraction of ``returns``.

    Args:
        returns: 1-D array of per-episode returns.
        alpha: tail level in (0, 1]. E.g. ``alpha=0.05`` averages the
            worst 5 % of returns.

    Returns:
        A Python float. Negative when the tail is dominated by losses.
    """
    r = np.asarray(returns, dtype=np.float32).reshape(-1)
    if r.size == 0:
        return 0.0
    a = _resolve_alpha(alpha, r.size)
    k = max(1, int(np.floor(a * r.size)))
    sorted_r = np.sort(r)
    return float(sorted_r[:k].mean())


def cvar_torch(returns: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Differentiable CVaR — useful as a regulariser in a PPO-style loss.

    ``torch.sort`` is piecewise-linear so gradients flow through.
    """
    if returns.numel() == 0:
        return torch.zeros((), dtype=returns.dtype, device=returns.device)
    a = _resolve_alpha(alpha, int(returns.numel()))
    k = max(1, int(np.floor(a * returns.numel())))
    sorted_r, _ = torch.sort(returns.reshape(-1))
    return sorted_r[:k].mean()


def cvar_constraint_violation(
    returns: np.ndarray,
    alpha: float = 0.1,
    threshold: float = 0.1,
) -> float:
    """Positive value if the CVaR constraint is violated; zero otherwise.

    The constraint is ``-CVaR_alpha <= threshold`` i.e. the average
    loss in the worst ``alpha``-fraction should not exceed
    ``threshold``. Returns ``max(0, -CVaR - threshold)``.
    """
    return max(0.0, -cvar_numpy(returns, alpha) - float(threshold))


__all__ = ["cvar_numpy", "cvar_torch", "cvar_constraint_violation"]
