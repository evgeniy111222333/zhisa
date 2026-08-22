"""Chart augmentation utilities: jitter, mirror, crop, noise (deterministic).

Used as a regulariser during training of the vision encoder.

The ideal requires augmentations to be **deterministic and keyed**: applying
the same key to the same image must reproduce the identical augmented image,
and the chosen transform(s) must be recorded alongside the compiled artefact
so any run can be reproduced or audited. The stochastic pure functions below
remain for ad-hoc use; :class:`KeyedAugmentor` is the pipeline-grade entry
point that derives all randomness from a single per-sample key.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Deterministic primitives (parameters are passed explicitly, not sampled)
# ---------------------------------------------------------------------------


def _safe_rand_int(rng: np.random.Generator, low: int, high: int) -> int:
    """Integer in [low, high) via numpy RNG (used by deterministic keying)."""
    if high <= low:
        return low
    return int(rng.integers(low, high))


def color_jitter_det(
    img: torch.Tensor,
    delta: tuple[float, float, float],
    strength: float = 0.1,
) -> torch.Tensor:
    """Apply a fixed per-channel colour offset (delta in [-1, 1]^3)."""
    if strength <= 0:
        return img
    noise = (
        torch.tensor(delta, dtype=img.dtype, device=img.device).view(3, 1, 1) * strength
    )
    return (img + noise).clamp(0.0, 1.0)


def horizontal_mirror_det(img: torch.Tensor, do_flip: bool) -> torch.Tensor:
    return torch.flip(img, dims=(2,)) if do_flip else img


def crop_and_resize_det(
    img: torch.Tensor,
    y0: int,
    x0: int,
    crop_frac: float = 0.85,
    size: int | None = None,
) -> torch.Tensor:
    """Fixed-origin crop window, resized back to ``size`` (bilinear)."""
    _, H, W = img.shape
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    y0_clip = min(max(y0, 0), max(H - ch, 0)) if H - ch > 0 else 0
    x0_clip = min(max(x0, 0), max(W - cw, 0)) if W - cw > 0 else 0
    cropped = img[:, y0_clip:y0_clip + ch, x0_clip:x0_clip + cw]
    if size is None or (cropped.shape[1] == H and cropped.shape[2] == W):
        return cropped
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def additive_gaussian_noise_det(
    img: torch.Tensor,
    noise: torch.Tensor,
    std: float = 0.02,
) -> torch.Tensor:
    if std <= 0:
        return img
    return (img + noise * std).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Legacy stochastic wrappers (kept for backward compatibility)
# ---------------------------------------------------------------------------


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
    _, H, W = img.shape
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    y0 = int(torch.randint(0, H - ch + 1, (1,)).item()) if H - ch > 0 else 0
    x0 = int(torch.randint(0, W - cw + 1, (1,)).item()) if W - cw > 0 else 0
    cropped = img[:, y0:y0 + ch, x0:x0 + cw]
    if size is None or (cropped.shape[1] == H and cropped.shape[2] == W):
        return cropped
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def additive_gaussian_noise(img: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if std <= 0:
        return img
    return (img + torch.randn_like(img) * std).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Deterministic, keyed augmentation pipeline
# ---------------------------------------------------------------------------

# Transforms that can be applied; names map to stable parameter sampling.
_AVAILABLE_TRANSFORMS = (
    "mirror",
    "color_jitter",
    "crop",
    "gaussian_noise",
)


def key_from_string(key: str) -> int:
    """Stable 64-bit integer seed derived from any string key."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def transform_seed(key: str, variant: str = "") -> np.random.Generator:
    """A per-sample RNG derived from ``key`` only.

    ``variant`` lets different stages of the pipeline (or different runs under
    the same sample key) use independent draws without extra state.
    """
    base = key_from_string(key)
    if variant:
        base ^= key_from_string(variant)
    return np.random.default_rng(base)


class KeyedAugmentor:
    """Apply a fixed, deterministic augmentation pipeline from a sample key.

    All stochasticity is derived from ``key`` via a single numpy RNG, so the
    same ``(key, transforms, strength)`` reproduces byte-identical output.
    The elected params can be recorded (``last_params``) for provenance.

    Example
    -------
    >>> aug = KeyedAugmentor(transforms=("mirror", "color_jitter"), strength=0.05)
    >>> out = aug(img, key="sample-42")
    >>> out2 = aug(img, key="sample-42")   # identical pixels
    >>> torch.all(out == out2)
    True
    """

    def __init__(
        self,
        transforms: tuple[str, ...] = ("mirror", "color_jitter", "crop", "gaussian_noise"),
        strength: float = 0.05,
        crop_frac: float = 0.85,
        noise_std: float = 0.01,
    ) -> None:
        unknown = set(transforms) - set(_AVAILABLE_TRANSFORMS)
        if unknown:
            raise ValueError(f"unknown transforms: {sorted(unknown)}")
        if not (0.0 <= strength <= 1.0):
            raise ValueError("strength must be in [0, 1]")
        self.transforms = tuple(transforms)
        self.strength = float(strength)
        self.crop_frac = float(crop_frac)
        self.noise_std = float(noise_std)
        self.last_params: dict | None = None

    # ------------------------------------------------------------------

    def _sample_params(self, key: str, H: int, W: int) -> dict:
        rng = transform_seed(key)
        params: dict = {}
        if "mirror" in self.transforms:
            params["mirror"] = bool(rng.integers(0, 2))
        if "color_jitter" in self.transforms:
            params["jitter_delta"] = tuple(float(x) for x in (rng.uniform(-1, 1, 3)))
        if "crop" in self.transforms:
            ch, cw = int(H * self.crop_frac), int(W * self.crop_frac)
            params["crop_y0"] = _safe_rand_int(rng, 0, H - ch + 1) if H - ch > 0 else 0
            params["crop_x0"] = _safe_rand_int(rng, 0, W - cw + 1) if W - cw > 0 else 0
        if "gaussian_noise" in self.transforms:
            # Sample the actual noise field from the keyed RNG so the applied
            # augmentation is fully deterministic (torch.randn_like would not be).
            params["noise_std"] = self.noise_std
            params["noise"] = rng.standard_normal((3, H, W)).astype(np.float32)
        self.last_params = params
        return params

    def apply(self, img: torch.Tensor, key: str) -> torch.Tensor:
        """Augment ``img`` (3,H,W) deterministically from ``key``."""
        if img.ndim != 3:
            raise ValueError("expected a (3, H, W) image")
        H, W = img.shape[1], img.shape[2]
        p = self._sample_params(key, H, W)
        out = img
        if p.get("mirror"):
            out = horizontal_mirror_det(out, True)
        if "jitter_delta" in p:
            out = color_jitter_det(out, p["jitter_delta"], self.strength)
        if "crop_y0" in p:
            out = crop_and_resize_det(
                out, p["crop_y0"], p["crop_x0"], self.crop_frac, size=W
            )
        if "noise_std" in p and p["noise_std"] > 0:
            noise = torch.from_numpy(p["noise"]).to(device=out.device, dtype=out.dtype)
            out = additive_gaussian_noise_det(out, noise, p["noise_std"])
        return out

    def apply_fixed_defaults(self, key: str) -> dict:
        """Return the parameter record that ``apply`` used for ``key``."""
        return self.last_params or {}

    # ---- JSON-friendly config for provenance / artifact metadata ----------

    def to_meta(self) -> dict:
        return {
            "kind": "keyed_augmentor",
            "transforms": list(self.transforms),
            "strength": self.strength,
            "crop_frac": self.crop_frac,
            "noise_std": self.noise_std,
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "KeyedAugmentor":
        return cls(
            transforms=tuple(meta.get("transforms", ())),
            strength=float(meta.get("strength", 0.05)),
            crop_frac=float(meta.get("crop_frac", 0.85)),
            noise_std=float(meta.get("noise_std", 0.01)),
        )