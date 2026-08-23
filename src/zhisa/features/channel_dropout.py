"""Channel Dropout: keyed, family-aware feature masking (numeric stream).

Ideal contract: mask a WHOLE feature channel across the window AFTER
normalization (z-space, fill 0.0 = neutral mean); at most one channel per
semantic family (logret*/rv*/vol*/sma*/ema*/ctx*/beta*/corr*...); mask is
derived purely from ``key = cd:<salt>:<sample_bucket>`` (same key -> same mask;
``pair_bucket`` groups (t, t+horizon) so temporal pairs usually share a mask);
deterministic and content-hashed for reproducibility.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

CHANNEL_DROPOUT_SPEC_VERSION = "1.0.0"

_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("logret", r"^logret_\d+$"),
    ("candle_shape", r"^(body_over_range|upper_wick_over_range|lower_wick_over_range|close_over_open|hl_over_close)$"),
    ("rv", r"^rv_\d+$"),
    ("atr", r"^atr"),
    ("volume", r"^vol_"),
    ("sma", r"^sma_"),
    ("ema", r"^ema_"),
    ("bollinger", r"^bb_"),
    ("rsi", r"^rsi_\d+$"),
    ("donchian", r"^don_"),
    ("vwap", r"^vwap"),
    ("cross_asset", r"^(rel_logret_1|beta_|corr_|volume_ratio_|volvol_ratio_|resid_alpha_|breadth_|dispersion_|market_vol_|leadlag_|market_index_|vw_weight_|beta_up_|beta_down_|corr_stress_)"),
    ("context", r"^ctx_"),
)


def channel_family(column: str) -> str:
    for family, pattern in _FAMILY_PATTERNS:
        if re.search(pattern, column):
            return family
    return "other"


def key_from_string(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class ChannelDropoutSpec:
    p: float = 0.15
    max_channels: int = 3
    pair_bucket: int = 16
    seed: int = 0
    fill_mode: str = "zero"   # "zero" (hard 0.0 = mean) | "noise" (keyed N(0, noise_std))
    noise_std: float = 0.1
    version: str = CHANNEL_DROPOUT_SPEC_VERSION

    def __post_init__(self):
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0,1]")
        if self.max_channels < 1:
            raise ValueError("max_channels must be >= 1")
        if self.pair_bucket < 1:
            raise ValueError("pair_bucket must be >= 1")
        if self.fill_mode not in {"zero", "noise"}:
            raise ValueError("fill_mode must be 'zero' or 'noise'")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")

    def content_hash(self) -> str:
        payload = json.dumps(
            {k: v for k, v in asdict(self).items() if k != "version"},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(self.version.encode() + b":" + payload).hexdigest()


def _rng_for(key: str, spec: ChannelDropoutSpec) -> np.random.Generator:
    base = (key_from_string(key) ^ ((spec.seed << 16) & 0xFFFFFFFF)) % (2**63)
    return np.random.default_rng(base)


@functools.lru_cache(maxsize=2048)
def _keyed_drop_mask_cached(
    n_channels: int, columns: tuple[str, ...], key: str, spec: ChannelDropoutSpec
) -> np.ndarray:
    """Cached core: masks repeat per (salt, bucket) — one hash/regex per bucket."""
    if n_channels == 0:
        return np.zeros(0, dtype=bool)
    rng = _rng_for(key, spec)
    order = rng.permutation(n_channels)
    chosen_families: set[str] = set()
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= int(spec.max_channels):
            break
        fam = channel_family(columns[int(idx)]) if columns else "other"
        if fam in chosen_families:
            continue
        if rng.random() < float(spec.p):
            chosen.append(int(idx))
            chosen_families.add(fam)
    mask = np.zeros(int(n_channels), dtype=bool)
    if chosen:
        mask[chosen] = True
    return mask


def keyed_drop_mask(
    n_channels: int,
    columns: list[str],
    key: str,
    spec: ChannelDropoutSpec,
) -> np.ndarray:
    """Deterministic mask for one sample (True = drop the channel)."""
    return _keyed_drop_mask_cached(int(n_channels), tuple(columns), key, spec)


def apply_channel_dropout(
    normed: np.ndarray,
    columns,
    key: str,
    spec: ChannelDropoutSpec,
) -> np.ndarray:
    """Drop masked channels of a normalized (T, F) window.

    ``fill_mode="zero"`` fills with 0.0 (neutral mean in z-space);
    ``fill_mode="noise"`` replaces the channels with keyed white noise
    ``N(0, noise_std)`` — deterministic per key, so runs stay reproducible.
    Returns a copy only when something is dropped.
    """
    mask = keyed_drop_mask(normed.shape[1], list(columns), key, spec)
    if not mask.any():
        return normed
    out = normed.copy()
    if spec.fill_mode == "noise":
        rng = _rng_for(key + ":noise", spec)
        noise = rng.normal(0.0, float(spec.noise_std), size=out[:, mask].shape)
        out[:, mask] = noise.astype(out.dtype)
    else:
        out[:, mask] = 0.0
    return out