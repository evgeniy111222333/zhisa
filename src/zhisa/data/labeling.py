"""Labeling: triple-barrier outcomes, regime labels, volatility targets.

All labeling functions are **strictly lagged**: they only look at data
at or before index ``t`` (or in the forward window starting at ``t+1``
for triple-barrier outcomes — never peeking back from the label index).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    """Configuration for the triple-barrier labeling method (López de Prado)."""

    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    max_holding: int = 32
    atr_window: int = 14


def triple_barrier(
    df: pd.DataFrame,
    cfg: Optional[TripleBarrierConfig] = None,
) -> pd.DataFrame:
    """Compute triple-barrier labels for a long-side view at each bar.

    Returns a DataFrame with columns:
        - ``label`` in {-1, 0, +1} (SL hit, timeout, TP hit)
        - ``ret``  : return realised when the barrier was first touched
        - ``t_hit`` : number of bars until the first touch (NaN if timeout)
    """
    cfg = cfg or TripleBarrierConfig()
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'high', 'low', 'close' columns")

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(df)

    # ATR
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = pd.Series(tr).rolling(cfg.atr_window, min_periods=1).mean().to_numpy()

    tp_arr = close + cfg.tp_atr_mult * atr
    sl_arr = close - cfg.sl_atr_mult * atr

    label = np.zeros(n, dtype=np.int64)
    ret = np.zeros(n, dtype=np.float64)
    t_hit = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        end = min(t + cfg.max_holding + 1, n)
        if end <= t + 1:
            continue
        tp = tp_arr[t]
        sl = sl_arr[t]
        entry = close[t]
        # Walk forward
        for k in range(t + 1, end):
            hit_tp = high[k] >= tp
            hit_sl = low[k] <= sl
            if hit_tp and hit_sl:
                # Both hit on same bar: decide by close vs entry
                outcome = 1 if close[k] >= entry else -1
                label[t] = outcome
                ret[t] = (close[k] - entry) / entry
                t_hit[t] = k - t
                break
            if hit_tp:
                label[t] = 1
                ret[t] = cfg.tp_atr_mult * atr[t] / entry
                t_hit[t] = k - t
                break
            if hit_sl:
                label[t] = -1
                ret[t] = -cfg.sl_atr_mult * atr[t] / entry
                t_hit[t] = k - t
                break
        # else: timeout, label stays 0, ret = realised return to last bar
        if label[t] == 0 and end > t + 1:
            ret[t] = (close[end - 1] - entry) / entry

    return pd.DataFrame(
        {"label": label, "ret": ret, "t_hit": t_hit},
        index=df.index,
    )


def realized_volatility(
    df: pd.DataFrame,
    horizon: int = 16,
    annualise: bool = True,
    periods_per_year: int = 365 * 24 * 12,
) -> pd.Series:
    """Forward realised volatility of log-returns over ``horizon`` bars.

    Strictly forward-looking: at bar ``t`` we compute std(log-returns
    over bars ``t+1 ... t+horizon``).
    """
    log_ret = np.log(df["close"]).diff().fillna(0.0).to_numpy()
    n = len(log_ret)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(n):
        end = min(t + horizon + 1, n)
        seg = log_ret[t + 1:end]
        if seg.size >= 2:
            out[t] = float(np.std(seg, ddof=0))
    s = pd.Series(out, index=df.index)
    if annualise:
        s = s * np.sqrt(periods_per_year)
    return s


def _gmm_numpy(X: np.ndarray, n_states: int, n_iter: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal NumPy GMM: spherical Gaussians, EM with random init.

    Returns (means shape (n_states, D), labels shape (N,)).
    Used as a fallback when scikit-learn is not installed.
    """
    N, D = X.shape
    means = X[rng.choice(N, size=n_states, replace=False)].copy()
    vars_ = np.full(n_states, float(np.var(X)) + 1e-12)
    weights = np.full(n_states, 1.0 / n_states)
    labels = np.zeros(N, dtype=np.int64)
    eps = 1e-12
    for _ in range(n_iter):
        # E-step
        diff = X[:, None, :] - means[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        log_p = (np.log(weights + eps) - 0.5 * D * np.log(2 * np.pi * vars_ + eps)
                 - sq / (2 * (vars_ + eps)))
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True) + eps
        labels = p.argmax(axis=1)
        # M-step
        w = p.sum(axis=0) + eps
        weights = w / w.sum()
        new_means = (p[:, :, None] * X[:, None, :]).sum(axis=0) / w[:, None]
        diff = X[:, None, :] - new_means[None, :, :]
        new_vars = (p[:, :, None] * (diff * diff)).sum(axis=(0, 2)) / (D * w + eps)
        vars_ = np.maximum(new_vars, 1e-12)
        means = new_means
    return means, labels


def hmm_regime_labels(
    df: pd.DataFrame,
    n_states: int = 4,
    lookback: int = 256,
    rebalance_period: int = 1000,
    random_state: int = 0,
    prefer_sklearn: bool = True,
) -> pd.Series:
    """HMM-style regime labelling using a rolling GMM with sorted clusters.
    
    To avoid Label Switching and slow performance, the GMM is only retrained
    every `rebalance_period` steps. The predicted clusters are sorted by their
    variance so that 0 is always the lowest volatility regime (flat market)
    and `n_states-1` is the highest volatility regime (storm/breakout).
    """
    log_ret = np.log(df["close"]).diff().fillna(0.0).to_numpy()
    n = len(log_ret)
    labels = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    use_sklearn = prefer_sklearn
    
    if use_sklearn:
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore
        except ImportError:
            use_sklearn = False

    last_gmm = None
    last_mapping = None

    for t in range(lookback, n):
        # Retrain only periodically or on the first valid step
        if last_gmm is None or t % rebalance_period == 0:
            window = log_ret[t - lookback:t].reshape(-1, 1)
            if use_sklearn:
                gmm = GaussianMixture(
                    n_components=n_states, covariance_type="full",
                    random_state=random_state, max_iter=15,
                )
                gmm.fit(window)
                # Sort components by variance
                variances = gmm.covariances_.flatten()
                sorted_idx = np.argsort(variances)
                last_mapping = {original: new_label for new_label, original in enumerate(sorted_idx)}
                last_gmm = gmm
            else:
                means, lab = _gmm_numpy(window, n_states, n_iter=15, rng=rng)
                variances = []
                for i in range(n_states):
                    cluster_points = window[lab == i]
                    variances.append(np.var(cluster_points) if len(cluster_points) > 0 else 0.0)
                sorted_idx = np.argsort(variances)
                last_mapping = {original: new_label for new_label, original in enumerate(sorted_idx)}
                last_gmm = means  # For fallback, just store means
                
        # Predict current step
        if use_sklearn:
            raw_label = int(last_gmm.predict(log_ret[t].reshape(1, -1))[0])
            labels[t] = last_mapping[raw_label]
        else:
            dist = np.abs(last_gmm - log_ret[t])
            raw_label = int(np.argmin(dist))
            labels[t] = last_mapping[raw_label]

    return pd.Series(labels, index=df.index, name="regime")
