"""Optimizer + scheduler factory used across training phases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.95)
    scheduler: str = "cosine"     # 'cosine' | 'step' | 'none'
    warmup_steps: int = 200
    step_size: int = 1000
    step_gamma: float = 0.5
    t_max: int = 10_000


def build_optimizer(model, cfg: OptimConfig) -> Optimizer:
    """Build an AdamW optimizer with weight-decay grouping.

    Accepts either a ``nn.Module`` (uses ``named_parameters`` for
    proper grouping) or an iterable of parameters (flat grouping).
    """
    if isinstance(model, nn.Module):
        params_iter = model.named_parameters()
        is_named = True
    else:
        params_iter = ((str(i), p) for i, p in enumerate(model))
        is_named = False
    decay, no_decay = [], []
    for name, p in params_iter:
        if not p.requires_grad:
            continue
        lname = name.lower() if is_named else ""
        if p.ndim < 2 or lname.endswith(".bias") or "norm" in lname or "bn" in lname:
            no_decay.append(p)
        else:
            decay.append(p)
    return AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def build_scheduler(opt: Optimizer, cfg: OptimConfig) -> Optional[object]:
    if cfg.scheduler == "cosine":
        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step) / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.t_max - cfg.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.141592653589793))))
        return LambdaLR(opt, lr_lambda=lr_lambda)
    if cfg.scheduler == "step":
        return StepLR(opt, step_size=cfg.step_size, gamma=cfg.step_gamma)
    return None
