"""Unsupervised state-space regime model with GMM-style weak states."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateSpaceConfig:
    n_states: int = 4
    min_history: int = 96
    lookback: int = 256
    short_window: int = 12
    medium_window: int = 48
    vol_window: int = 24
    em_iters: int = 20
    regularization: float = 1e-4
    change_point_threshold: float = 1.25


@dataclass(frozen=True)
class StateSpaceReport:
    current_state: int
    state_label: str
    state_probability: float
    transition_probability: float
    change_point_score: float
    entropy: float
    state_probabilities: list[float]
    transition_matrix: list[list[float]]
    state_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _safe_ret(close: pd.Series, bars: int) -> pd.Series:
    return close.astype(float).pct_change(bars).replace([np.inf, -np.inf], np.nan)


def _zscore_frame(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where((std <= 1e-8) | ~np.isfinite(std), 1.0, std)
    z = (x - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, mean, std


class StateSpaceRegimeModel:
    """Small diagonal-GMM state model with empirical transition matrix."""

    def __init__(self, cfg: Optional[StateSpaceConfig] = None) -> None:
        self.cfg = cfg or StateSpaceConfig()

    def analyze(self, df: pd.DataFrame, *, t: Optional[int] = None) -> StateSpaceReport:
        if t is not None:
            if t < 0:
                raise ValueError("t must be non-negative")
            work = df.iloc[: t + 1].copy()
        else:
            work = df.copy()
        if work.empty:
            raise ValueError("df slice is empty")
        if "close" not in work.columns:
            raise ValueError("df must contain a close column")
        if len(work) < self.cfg.min_history:
            return self._default_report()

        feats = self._feature_matrix(work).iloc[-self.cfg.lookback:].dropna()
        if len(feats) < max(self.cfg.n_states * 4, 16):
            return self._default_report()
        x_raw = feats.to_numpy(dtype=np.float64)
        x, _, _ = _zscore_frame(x_raw)
        means, vars_, weights = self._fit_gmm(x)
        probs = self._posterior(x, means, vars_, weights)
        labels = probs.argmax(axis=1)
        state_labels = self._state_labels(means, feats.columns)
        trans = self._transition_matrix(labels, self.cfg.n_states)
        current = int(labels[-1])
        prev = int(labels[-2]) if labels.size > 1 else current
        current_probs = probs[-1]
        state_probability = float(current_probs[current])
        transition_probability = float(1.0 - trans[prev, current]) if labels.size > 1 else 0.0
        cp_score = self._change_point_score(x)
        entropy = float(-(current_probs * np.log(current_probs + 1e-12)).sum() / np.log(max(self.cfg.n_states, 2)))
        return StateSpaceReport(
            current_state=current,
            state_label=state_labels[current],
            state_probability=_clip01(state_probability),
            transition_probability=_clip01(max(transition_probability, cp_score * 0.35)),
            change_point_score=cp_score,
            entropy=_clip01(entropy),
            state_probabilities=[float(v) for v in current_probs],
            transition_matrix=trans.tolist(),
            state_labels=state_labels,
        )

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].astype(float)
        high = df.get("high", close).astype(float)
        low = df.get("low", close).astype(float)
        volume = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)
        logret = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan)
        vol = logret.rolling(self.cfg.vol_window, min_periods=3).std()
        vol_long = logret.rolling(max(self.cfg.medium_window, self.cfg.vol_window * 2), min_periods=5).std()
        range_high = high.rolling(self.cfg.medium_window, min_periods=5).max()
        range_low = low.rolling(self.cfg.medium_window, min_periods=5).min()
        range_pos = (close - range_low) / (range_high - range_low + 1e-12)
        volume_z = (volume - volume.rolling(64, min_periods=5).mean()) / (volume.rolling(64, min_periods=5).std() + 1e-12)
        peak = close.cummax()
        drawdown = (peak - close) / (peak + 1e-12)
        return pd.DataFrame({
            "ret_short": _safe_ret(close, self.cfg.short_window),
            "ret_medium": _safe_ret(close, self.cfg.medium_window),
            "vol": vol,
            "vol_ratio": vol / (vol_long + 1e-12),
            "range_position": range_pos,
            "volume_z": volume_z,
            "drawdown": drawdown,
        }, index=df.index).replace([np.inf, -np.inf], np.nan)

    def _fit_gmm(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, d = x.shape
        k = min(self.cfg.n_states, max(1, n))
        order = np.argsort(x[:, 0] + 0.5 * x[:, 1] + 0.25 * x[:, 2])
        quantiles = np.linspace(0, n - 1, k).round().astype(int)
        means = x[order[quantiles]].copy()
        vars_ = np.tile(np.var(x, axis=0) + self.cfg.regularization, (k, 1))
        weights = np.full(k, 1.0 / k)
        for _ in range(max(1, self.cfg.em_iters)):
            resp = self._posterior(x, means, vars_, weights)
            nk = resp.sum(axis=0) + 1e-12
            weights = nk / n
            means = (resp.T @ x) / nk[:, None]
            for j in range(k):
                diff = x - means[j]
                vars_[j] = (resp[:, j][:, None] * diff * diff).sum(axis=0) / nk[j] + self.cfg.regularization
        if k < self.cfg.n_states:
            pad = self.cfg.n_states - k
            means = np.vstack([means, np.zeros((pad, d))])
            vars_ = np.vstack([vars_, np.ones((pad, d))])
            weights = np.r_[weights, np.full(pad, 1e-6)]
            weights = weights / weights.sum()
        return means, vars_, weights

    def _posterior(self, x: np.ndarray, means: np.ndarray, vars_: np.ndarray, weights: np.ndarray) -> np.ndarray:
        logs = []
        for j in range(means.shape[0]):
            var = np.maximum(vars_[j], self.cfg.regularization)
            diff = x - means[j]
            logp = -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(diff * diff / var, axis=1))
            logs.append(np.log(max(float(weights[j]), 1e-12)) + logp)
        log_arr = np.vstack(logs).T
        log_arr -= log_arr.max(axis=1, keepdims=True)
        exp = np.exp(log_arr)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    def _transition_matrix(self, labels: np.ndarray, n_states: int) -> np.ndarray:
        mat = np.full((n_states, n_states), 1e-3, dtype=np.float64)
        for a, b in zip(labels[:-1], labels[1:]):
            mat[int(a), int(b)] += 1.0
        mat /= mat.sum(axis=1, keepdims=True)
        return mat

    def _change_point_score(self, x: np.ndarray) -> float:
        if x.shape[0] < self.cfg.medium_window:
            return 0.0
        sw = min(self.cfg.short_window, x.shape[0] // 3)
        mw = min(self.cfg.medium_window, x.shape[0] - sw)
        short = x[-sw:].mean(axis=0)
        prev = x[-mw - sw : -sw].mean(axis=0)
        dist = float(np.linalg.norm(short - prev) / np.sqrt(x.shape[1]))
        return _clip01(dist / max(self.cfg.change_point_threshold, 1e-12))

    def _state_labels(self, means: np.ndarray, columns: pd.Index) -> list[str]:
        idx = {name: i for i, name in enumerate(columns)}
        labels = []
        for row in means:
            ret = float(row[idx.get("ret_medium", 0)])
            vol = float(row[idx.get("vol_ratio", 0)])
            dd = float(row[idx.get("drawdown", 0)])
            rng = float(row[idx.get("range_position", 0)])
            if dd > 0.8 and ret < -0.2:
                labels.append("drawdown_stress")
            elif vol > 0.8:
                labels.append("high_vol")
            elif ret > 0.35 and rng > 0.0:
                labels.append("trend_up_state")
            elif ret < -0.35 and rng < 0.0:
                labels.append("trend_down_state")
            else:
                labels.append("range_state")
        return labels

    def _default_report(self) -> StateSpaceReport:
        n = int(self.cfg.n_states)
        probs = [1.0 / n for _ in range(n)]
        mat = np.eye(n, dtype=float).tolist()
        return StateSpaceReport(
            current_state=0,
            state_label="insufficient_history",
            state_probability=0.0,
            transition_probability=0.0,
            change_point_score=0.0,
            entropy=1.0,
            state_probabilities=probs,
            transition_matrix=mat,
            state_labels=["insufficient_history"] + [f"state_{i}" for i in range(1, n)],
        )


__all__ = [
    "StateSpaceConfig",
    "StateSpaceRegimeModel",
    "StateSpaceReport",
]
