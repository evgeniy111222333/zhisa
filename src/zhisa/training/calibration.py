"""Probability calibration utilities for the policy/value heads.

Raw neural-network outputs are notoriously poorly calibrated: a
``policy_logits`` of +0.3 vs +0.1 does not necessarily mean a 10%
higher chance of "up". For trading, miscalibrated probabilities
lead to:
  * wrong position sizing (Kelly uses the *calibrated* win prob);
  * wrong Sharpe optimisation (mean overconfident on losers);
  * wrong meta-decision thresholds (drift / kill-switch).

This module implements three post-hoc calibrators that take a
trained model and a labelled validation set, and produce a
``Calibrator`` object that wraps the model so that
``calibrator.predict_proba(logits)`` returns well-calibrated
probabilities.

Calibrators implemented:
  * :class:`TemperatureScaler`  — single-parameter T scaling
    (Guo et al. 2017). Works well for nearly any model.
  * :class:`PlattCalibrator`     — 2-parameter logistic regression
    on the logits, fit with gradient descent.
  * :class:`IsotonicCalibrator`  — non-parametric step function.
    Highest capacity but needs more data; uses
    ``sklearn.isotonic`` if available, else falls back to a numpy
    implementation.

The module also exposes a :func:`calibration_error` helper (ECE —
expected calibration error, one of the standard metrics) and a
:finc:`reliability_diagram` helper for plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------


@dataclass
class TemperatureScaler:
    """Single-parameter temperature scaling.

    Given raw logits ``z``, returns ``softmax(z / T)``. ``T`` is fit
    on a labelled validation set by minimising the negative
    log-likelihood.
    """

    temperature: float = 1.0
    n_iter: int = 200
    lr: float = 0.05

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        z = _to_numpy(logits).astype(np.float64)
        y = _to_numpy(labels).astype(np.int64)
        T = torch.nn.Parameter(torch.tensor(self.temperature, dtype=torch.float64))
        opt = torch.optim.LBFGS([T], lr=self.lr, max_iter=self.n_iter)

        z_t = torch.from_numpy(z)
        y_t = torch.from_numpy(y)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z_t / T.clamp(min=1e-3), y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(T.detach().clamp(min=1e-3).item())
        return self

    def predict_proba(self, logits) -> np.ndarray:
        z = _to_numpy(logits).astype(np.float64) / self.temperature
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"temperature": self.temperature}

    def load_state_dict(self, state: dict) -> None:
        self.temperature = float(state.get("temperature", 1.0))


# ---------------------------------------------------------------------------
# Platt scaling (logistic regression on logits)
# ---------------------------------------------------------------------------


@dataclass
class PlattCalibrator:
    """2-parameter logistic regression on a single logit column.

    Useful for binary classification (e.g. direction "up vs down").
    For multi-class, the :class:`TemperatureScaler` is usually
    sufficient and less prone to overfitting on small validation
    sets.
    """

    slope: float = 1.0
    intercept: float = 0.0
    n_iter: int = 1000
    lr: float = 0.05
    l2: float = 1e-3
    target_class: int = 1   # which class index to calibrate

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        z = _to_numpy(logits).astype(np.float64)
        y = _to_numpy(labels).astype(np.int64)
        if z.ndim == 2:
            z = z[:, self.target_class]
        z_t = torch.from_numpy(z)
        y_t = (torch.from_numpy(y) == self.target_class).double()
        a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        opt = torch.optim.Adam([a, b], lr=self.lr)
        for _ in range(int(self.n_iter)):
            opt.zero_grad()
            p = torch.sigmoid(a * z_t + b)
            loss = F.binary_cross_entropy(p, y_t) + self.l2 * (a * a + b * b)
            loss.backward()
            opt.step()
        self.slope = float(a.item())
        self.intercept = float(b.item())
        return self

    def predict_proba(self, logits) -> np.ndarray:
        z = _to_numpy(logits).astype(np.float64)
        if z.ndim == 2:
            z = z[:, self.target_class]
        p = 1.0 / (1.0 + np.exp(-(self.slope * z + self.intercept)))
        return p.astype(np.float32)

    def state_dict(self) -> dict:
        return {"slope": self.slope, "intercept": self.intercept,
                "target_class": self.target_class}

    def load_state_dict(self, state: dict) -> None:
        self.slope = float(state.get("slope", 1.0))
        self.intercept = float(state.get("intercept", 0.0))
        self.target_class = int(state.get("target_class", 1))


# ---------------------------------------------------------------------------
# Isotonic regression (non-parametric)
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator.

    Wraps :mod:`sklearn.isotonic.IsotonicRegression` if available,
    otherwise falls back to a pure-numpy PAVA implementation.
    """

    out_of_bounds: str = "clip"

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        s = _to_numpy(scores).astype(np.float64)
        y = _to_numpy(labels).astype(np.float64)
        # Isotonic regression is 1-D in / 1-D out. If the caller hands
        # us 2-D logits, calibrate the column selected by
        # ``target_class`` (default = 1). For 1-D scores, pass through.
        if s.ndim == 2:
            s = s[:, 1]
        # We always use the numpy PAVA implementation, even if sklearn
        # is available, because sklearn's IsotonicRegression cannot
        # be cleanly round-tripped through state_dict without
        # exposing its private attributes.
        impl = _IsotonicNumpy(out_of_bounds=self.out_of_bounds)
        impl.fit(s, y)
        self._impl = impl
        return self

    def predict_proba(self, scores) -> np.ndarray:
        s = _to_numpy(scores).astype(np.float64)
        if s.ndim == 2:
            s = s[:, 1]
        return self._impl.predict(s).astype(np.float32)

    def state_dict(self) -> dict:
        return {"x": self._impl.x.tolist(), "y": self._impl.y.tolist()}

    def load_state_dict(self, state: dict) -> None:
        impl = _IsotonicNumpy(out_of_bounds=self.out_of_bounds)
        impl.x = np.array(state.get("x", []), dtype=np.float64)
        impl.y = np.array(state.get("y", []), dtype=np.float64)
        self._impl = impl


class _IsotonicNumpy:
    """Pool-Adjacent-Violators-Algorithm isotonic regression in numpy.

    Used as a sklearn-free fallback so the calibrator works in
    minimal environments. The output is a monotone step function
    that is constant on each PAVA block.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_IsotonicNumpy":
        # Sort by x.
        order = np.argsort(x)
        xs = np.asarray(x[order], dtype=np.float64)
        ys = np.asarray(y[order], dtype=np.float64)

        # PAVA: blocks stored as parallel arrays of (block_xs, block_ys).
        block_xs: list[list[float]] = [[v] for v in xs.tolist()]
        block_ys: list[list[float]] = [[v] for v in ys.tolist()]

        i = 0
        while i < len(block_xs) - 1:
            m_left = float(np.mean(block_ys[i]))
            m_right = float(np.mean(block_ys[i + 1]))
            if m_left <= m_right:
                i += 1
                continue
            # Merge i and i+1 and continue checking left.
            merged_x = block_xs[i] + block_xs[i + 1]
            merged_y = block_ys[i] + block_ys[i + 1]
            # Re-sort merged block by x so the monotone property is
            # still meaningful within the block.
            order_m = np.argsort(merged_x)
            block_xs[i] = [merged_x[k] for k in order_m]
            block_ys[i] = [merged_y[k] for k in order_m]
            del block_xs[i + 1]
            del block_ys[i + 1]
            if i > 0:
                i -= 1
            # else stay at 0 and re-check.

        # Flatten into step function: each block's mean is repeated
        # for every x in the block, then concatenated.
        flat_x: list[float] = []
        flat_y: list[float] = []
        for bx, by in zip(block_xs, block_ys):
            mean = float(np.mean(by))
            flat_x.extend(bx)
            flat_y.extend([mean] * len(bx))
        # Re-sort by x in case the merges scrambled things.
        order2 = np.argsort(flat_x)
        self.x = np.array(flat_x, dtype=np.float64)[order2]
        self.y = np.array(flat_y, dtype=np.float64)[order2]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.interp(x, self.x, self.y)
        if self.out_of_bounds == "clip":
            out = np.clip(out, 0.0, 1.0)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected calibration error (ECE).

    Bins predictions into ``n_bins`` equal-width bins between 0 and
    1 and computes the weighted average of ``|bin_acc - bin_conf|``.
    Lower is better; 0 = perfectly calibrated.
    """
    p = _to_numpy(probs).astype(np.float64)
    y = _to_numpy(labels).astype(np.float64)
    if p.ndim == 2:
        p = p[:, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        bin_conf = float(p[mask].mean())
        bin_acc = float(y[mask].mean())
        ece += (mask.sum() / max(n, 1)) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Return per-bin accuracy / confidence / count for plotting."""
    p = _to_numpy(probs).astype(np.float64)
    y = _to_numpy(labels).astype(np.float64)
    if p.ndim == 2:
        p = p[:, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        bin_acc[i] = float(y[mask].mean())
        bin_conf[i] = float(p[mask].mean())
        bin_count[i] = int(mask.sum())
    return {
        "bin_edges": bins,
        "bin_accuracy": bin_acc,
        "bin_confidence": bin_conf,
        "bin_count": bin_count,
    }


# ---------------------------------------------------------------------------
# One-stop fit / save / load
# ---------------------------------------------------------------------------


def fit_calibrator(
    kind: str,
    logits: np.ndarray,
    labels: np.ndarray,
    **kwargs,
):
    """Factory: fit a calibrator of the given ``kind`` and return it.

    ``kind`` is one of ``"temperature"``, ``"platt"``, ``"isotonic"``.
    """
    if kind == "temperature":
        cal = TemperatureScaler(**{k: v for k, v in kwargs.items()
                                    if k in {"n_iter", "lr"}})
        return cal.fit(logits, labels)
    if kind == "platt":
        cal = PlattCalibrator(**{k: v for k, v in kwargs.items()
                                  if k in {"n_iter", "lr", "l2",
                                           "target_class"}})
        return cal.fit(logits, labels)
    if kind == "isotonic":
        cal = IsotonicCalibrator(**{k: v for k, v in kwargs.items()
                                     if k in {"out_of_bounds"}})
        return cal.fit(logits, labels)
    raise ValueError(f"unknown calibrator kind: {kind!r}")
