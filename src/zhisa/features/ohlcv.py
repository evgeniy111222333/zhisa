"""OHLCV-derived numeric features: returns, ranges, body/wick ratios, vol."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from zhisa.features.indicators import (
    atr,
    bollinger,
    donchian,
    ema,
    rsi,
    sma,
    vwap_session,
)


def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def compute_ohlcv_features(
    df: pd.DataFrame,
    *,
    include_volume: bool = True,
    include_indicators: bool = True,
    ema_periods: Optional[List[int]] = None,
    sma_periods: Optional[List[int]] = None,
    atr_period: int = 14,
    rsi_period: int = 14,
    bb_period: int = 20,
    donchian_period: int = 20,
) -> pd.DataFrame:
    """Build a feature matrix aligned to ``df.index``.

    Features include log-returns at several lags, range, body/wick ratios,
    volume z-score, and a configurable set of moving-average / oscillator
    indicators. All values are returned as a numeric DataFrame with no
    look-ahead: indicators use only past data within the same frame.
    """
    ema_periods = ema_periods or [8, 21, 55]
    sma_periods = sma_periods or [10, 20, 50]
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df.get("volume", None)

    # Returns
    log_close = _safe_log(close)
    for lag in (1, 2, 4, 8, 16):
        out[f"logret_{lag}"] = log_close.diff(lag)

    # Range, body, wick ratios
    rng = (high - low).replace(0, np.nan)
    out["body_over_range"] = (close - open_).abs() / rng
    out["upper_wick_over_range"] = (high - np.maximum(close, open_)) / rng
    out["lower_wick_over_range"] = (np.minimum(close, open_) - low) / rng
    out["close_over_open"] = (close - open_) / (open_ + 1e-12)
    out["hl_over_close"] = (high - low) / close

    # Volatility
    log_ret1 = log_close.diff()
    for w in (8, 16, 32):
        out[f"rv_{w}"] = log_ret1.rolling(w, min_periods=2).std()
    if include_indicators:
        a = atr(df, period=atr_period)
        out[f"atr_{atr_period}"] = a
        out[f"atr_pct_{atr_period}"] = a / close

    # Volume
    if include_volume and vol is not None:
        for w in (16, 64):
            mv = vol.rolling(w, min_periods=2).mean()
            sv = vol.rolling(w, min_periods=2).std()
            out[f"vol_z_{w}"] = (vol - mv) / (sv + 1e-12)
        out["vol_over_avg_16"] = vol / (vol.rolling(16, min_periods=1).mean() + 1e-12)

    # Moving averages: distance to MA, slope, cross
    if include_indicators:
        for p in sma_periods:
            s = sma(close, p)
            out[f"sma_dist_{p}"] = (close - s) / (close + 1e-12)
            out[f"sma_slope_{p}"] = (s - s.shift(p // 2 or 1)) / (close + 1e-12)
        for p in ema_periods:
            e = ema(close, p)
            out[f"ema_dist_{p}"] = (close - e) / (close + 1e-12)
        bb = bollinger(close, period=bb_period)
        out[f"bb_pct_b_{bb_period}"] = bb["pct_b"]
        out[f"bb_bw_{bb_period}"] = bb["bandwidth"]
        out["rsi_" + str(rsi_period)] = rsi(close, rsi_period) / 100.0
        don = donchian(high, low, period=donchian_period)
        out[f"don_pos_{donchian_period}"] = (close - don["low"]) / (don["high"] - don["low"] + 1e-12)
        # VWAP distance
        vwap = vwap_session(df)
        out["vwap_dist"] = (close - vwap) / (close + 1e-12)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float64)


def normalize_feature_window(
    feature_window: np.ndarray,
    history_window: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize a feature window using the mean and std of a trailing history window.

    Robustly handles NaNs/Infs and returns a clean float32 array.
    """
    hist_clean = np.nan_to_num(history_window, nan=0.0, posinf=0.0, neginf=0.0)
    feat_clean = np.nan_to_num(feature_window, nan=0.0, posinf=0.0, neginf=0.0)

    mu = hist_clean.mean(axis=0)
    sd = hist_clean.std(axis=0) + eps

    normed = (feat_clean - mu) / sd
    return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
