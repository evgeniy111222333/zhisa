"""Feature normalization: versioned, fast rolling z / robust (median-MAD).

Why this exists (see ``dataset.py`` hot path):

1. The previous per-sample implementation recomputed mean/std over a trailing
   history window for *every* sample — O(samples × lookback × features) of
   repeated work in the training loop.
2. Classic mean/std z-score is sensitive to fat tails (spikes, flash-crash
   bars) — one extreme bar inflates the std of an entire window.

Solutions provided here:

- :class:`PrefixStats` — a one-shot ``(sum, sum-of-squares)`` prefix table that
  answers ``mean_std(lo, hi)`` in O(features) instead of O(window × features).
  It reproduces the classic windowed z-score (default ``mode="rolling_z"``).
- :func:`robust_z` — median / MAD (1.4826 × MAD ≈ σ under normality) which is
  resistant to a few large outliers (``mode="robust_z"``).
- :class:`NormalizationSpec` — (mode, lookback, eps) as a content-addressable,
  versioned configuration so the normalization identity can be recorded next to
  a dataset / checkpoint, like the render contract.

The public legacy function ``normalize_feature_window`` is kept unchanged for
extraneous callers (env, macro path, tests); the dataset hot path uses
:class:`PrefixStats`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields

import numpy as np

NORMALIZATION_SPEC_VERSION = "1.0.0"

_MAD_SCALE = 1.4826  # converts MAD to an unbiased σ estimate for normal data


@dataclass(frozen=True)
class NormalizationSpec:
    """Deterministic description of how numeric features are normalized.

    - ``rolling_z`` (default): (x - mean) / (std + eps) over a trailing window;
    - ``robust_z``: (x - median) / (1.4826*MAD + eps) over a trailing window.
    """

    mode: str = "rolling_z"
    lookback: int = 256
    eps: float = 1e-6
    version: str = NORMALIZATION_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.mode not in {"rolling_z", "robust_z"}:
            raise ValueError(f"unknown normalization mode: {self.mode!r}")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if not (0.0 < self.eps < 1e0):
            raise ValueError("eps must be in (0, 1)")

    def content_hash(self) -> str:
        raw = json.dumps(
            {"version": self.version, **{k: v for k, v in asdict(self).items() if k != "version"}},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_meta(self) -> dict:
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: dict) -> "NormalizationSpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in meta.items() if k in known})


class PrefixStats:
    """O(1)-per-window mean/std from a one-time prefix table.

    Built over the NaN-cleaned feature matrix once per symbol; ``mean_std``
    answers any ``[lo, hi)`` slice in O(features).
    """

    __slots__ = ("n", "f", "_sum", "_sumsq", "_dtype")

    def __init__(self, table: np.ndarray) -> None:
        arr = np.ascontiguousarray(
            np.nan_to_num(np.asarray(table, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        )
        self.n, self.f = arr.shape
        self._sum = np.concatenate([np.zeros((1, self.f)), np.cumsum(arr, axis=0)], axis=0)
        self._sumsq = np.concatenate([np.zeros((1, self.f)), np.cumsum(arr * arr, axis=0)], axis=0)
        self._dtype = np.float32

    def mean_std(self, lo: int, hi: int, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        lo = max(0, int(lo))
        hi = min(int(hi), self.n)
        count = max(hi - lo, 1)
        s = self._sum[hi] - self._sum[lo]
        q = self._sumsq[hi] - self._sumsq[lo]
        mean = s / count
        var = (q / count - mean * mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var) + eps
        return mean, std

    def zscore_window(self, window: np.ndarray, lo: int, hi: int, eps: float = 1e-6) -> np.ndarray:
        """Normalize ``window`` by stats of rows ``[lo, hi)`` (classic semantics)."""
        mean, std = self.mean_std(lo, hi, eps=eps)
        clean = np.nan_to_num(np.asarray(window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return ((clean - mean) / std).astype(self._dtype)


def robust_z(
    feature_window: np.ndarray,
    history_window: np.ndarray,
    eps: float = 1e-6,
    mad_scale: float = _MAD_SCALE,
) -> np.ndarray:
    """Median/MAD z-score, resistant to fat tails.

    ``history_window`` is the trailing rows used to estimate median/MAD; any
    non-finite values are zeroed first (matching the classic contract).
    """
    hist = np.nan_to_num(np.asarray(history_window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    feat = np.nan_to_num(np.asarray(feature_window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    median = np.median(hist, axis=0)
    mad = np.median(np.abs(hist - median), axis=0)
    scale = mad_scale * mad + eps
    return ((feat - median) / scale).astype(np.float32)


def normalize_window(
    feature_window: np.ndarray,
    history_window: np.ndarray,
    spec: NormalizationSpec | None = None,
) -> np.ndarray:
    """Dispatch by spec (legacy callers keep passing the raw arrays)."""
    spec = spec or NormalizationSpec()
    if spec.mode == "robust_z":
        return robust_z(
            feature_window, history_window, eps=spec.eps,
        )
    # rolling_z: replicate classic mean/std semantics from the raw arrays.
    hist = np.nan_to_num(np.asarray(history_window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    feat = np.nan_to_num(np.asarray(feature_window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mean = hist.mean(axis=0)
    std = hist.std(axis=0) + spec.eps
    return ((feat - mean) / std).astype(np.float32)