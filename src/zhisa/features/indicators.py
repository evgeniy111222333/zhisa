"""Classic technical indicators: SMA, EMA, RSI, ATR, Bollinger, Donchian, VWAP.

All implementations are pure NumPy / Pandas and avoid look-ahead:
indicators at index ``t`` only use data up to and including ``t``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()


def bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0) -> Dict[str, pd.Series]:
    m = sma(series, period)
    sd = series.rolling(period, min_periods=1).std()
    upper = m + n_std * sd
    lower = m - n_std * sd
    pct_b = (series - lower) / (upper - lower + 1e-12)
    bw = (upper - lower) / (m + 1e-12)
    return {"mid": m, "upper": upper, "lower": lower, "pct_b": pct_b, "bandwidth": bw}


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    return {
        "high": high.rolling(period, min_periods=1).max(),
        "low": low.rolling(period, min_periods=1).min(),
    }


def vwap_session(df: pd.DataFrame, session_col: Optional[str] = None) -> pd.Series:
    """Cumulative VWAP per session (defaults to UTC day)."""
    vol = df.get("volume", pd.Series(1.0, index=df.index))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * vol
    if session_col is None:
        session = df.index.date
    else:
        session = df[session_col]
    pv_cum = pv.groupby(session).cumsum()
    vol_cum = vol.groupby(session).cumsum()
    return pv_cum / (vol_cum + 1e-12)
