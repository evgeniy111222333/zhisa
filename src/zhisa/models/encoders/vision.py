"""Vision encoder for chart images.

A small, deliberately lightweight CNN that maps (3, H, W) chart images
to a fixed-size embedding. We intentionally avoid heavy pretrained
backbones in the default module to keep the install footprint small;
a swap-in for ConvNeXt/EVA is straightforward via the same interface.

Note: we use :class:`nn.GroupNorm` (1 group per channel) instead of
:class:`nn.BatchNorm2d` to avoid running-stat contamination when
forward passes produce non-finite values — a real failure mode of
the S1 SSL pretrainer in v0.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class VisionEncoderConfig:
    image_size: int = 64
    in_channels: int = 3
    out_dim: int = 128
    channels: Tuple[int, ...] = (32, 64, 128, 192)
    dropout: float = 0.1


class _GN(nn.Module):
    """GroupNorm with 1 group per channel — equivalent to InstanceNorm
    but with the standard ``norm`` interface. Avoids running stats
    contamination when SSL losses occasionally explode, *and* works
    for batch sizes of 1 (which a PPO rollout produces on every
    environment step).
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        # 1 group = full channel-wise normalisation; safe for batch=1.
        self.norm = nn.GroupNorm(num_groups=1, num_channels=c, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class VisionEncoder(nn.Module):
    """A small ConvNet chart encoder."""

    def __init__(self, cfg: VisionEncoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or VisionEncoderConfig()
        self.cfg = cfg
        layers: list[nn.Module] = []
        c_in = cfg.in_channels
        for c_out in cfg.channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                _GN(c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(c_in, cfg.out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # x: (B, 3, H, W) in [0, 1]
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)
