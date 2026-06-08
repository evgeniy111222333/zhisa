"""Chart augmentation utilities: jitter, mirror, crop, noise.

Used as a regulariser during training of the vision encoder.
"""
from __future__ import annotations

import numpy as np
import torch


def color_jitter(img: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    if strength <= 0:
        return img
    noise = (torch.rand(3, 1, 1, device=img.device) * 2 - 1) * strength
    return (img + noise).clamp(0.0, 1.0)


def horizontal_mirror(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if torch.rand(1).item() < p:
        return torch.flip(img, dims=(2,))
    return img


def crop_and_resize(
    img: torch.Tensor,
    crop_frac: float = 0.85,
    size: int | None = None,
) -> torch.Tensor:
    """Random-crop a fraction of the image and resize back."""
    _, H, W = img.shape
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    y0 = int(torch.randint(0, H - ch + 1, (1,)).item()) if H - ch > 0 else 0
    x0 = int(torch.randint(0, W - cw + 1, (1,)).item()) if W - cw > 0 else 0
    cropped = img[:, y0:y0 + ch, x0:x0 + cw]
    if size is None or (cropped.shape[1] == H and cropped.shape[2] == W):
        return cropped
    # Bilinear via torch
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def additive_gaussian_noise(img: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if std <= 0:
        return img
    return (img + torch.randn_like(img) * std).clamp(0.0, 1.0)
