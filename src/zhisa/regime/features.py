"""Causal feature extraction for Market Regime Intelligence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from zhisa.features.indicators import atr, bollinger, ema, vwap_session
from zhisa.regime.schema import RegimeFeatures
from zhisa.storage.schema import Timeframe


@dataclass(frozen=True)
class RegimeFeatureConfig:
    short_window: int = 8
    medium_window: int = 32
    long_window: int = 96
    vol_short_window: int = 16
    vol_long_window: int = 96
    atr_window: int = 14
    bb_window: int = 20
    quantile_window: int = 128
    breakout_window: int = 32
    min_bars: int = 24


def _clip_float(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, lo, hi))


def _safe_ret(close: pd.Series, bars: int) -> float:
    if len(close) <= bars:
        return 0.0
    c0 = float(close.iloc[-bars - 1])
    c1 = float(close.iloc[-1])
    if c0 <= 0 or not np.isfinite(c0) or not np.isfinite(c1):
        return 0.0
    return (c1 / c0) - 1.0


def _rolling_quantile_rank(series: pd.Series, value: float, window: int) -> float:
    hist = series.dropna().iloc[-window:]
    if hist.empty or not np.isfinite(value):
        return 0.5
    return float((hist <= value).mean())


def _infer_timeframe(df: pd.DataFrame, fallback: str = "5m") -> str:
    if len(df.index) < 3 or not isinstance(df.index, pd.DatetimeIndex):
        return fallback
    delta = pd.Series(df.index[1:] - df.index[:-1]).median()
    minutes = max(1, int(round(delta.total_seconds() / 60.0)))
    for tf in Timeframe:
        if tf.minutes == minutes:
            return tf.value
    return f"{minutes}m"


def compute_regime_features(
    df: pd.DataFrame,
    *,
    t: Optional[int] = None,
    timeframe: Optional[str] = None,
    cfg: Optional[RegimeFeatureConfig] = None,
) -> RegimeFeatures:
    """Compute a causal feature snapshot at ``t``.

    ``t`` is inclusive.  Passing ``df.iloc[:t+1]`` and omitting ``t``
    yields the same result, which is the anti-look-ahead contract used
    by the tests.
    """
    cfg = cfg or RegimeFeatureConfig()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")
    if t is not None:
        if t < 0:
            raise ValueError("t must be non-negative")
        work = df.iloc[: t + 1].copy()
    else:
        work = df.copy()
    if work.empty:
        raise ValueError("df slice is empty")

    tf = timeframe or _infer_timeframe(work)
    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    volume = work.get("volume", pd.Series(1.0, index=work.index)).astype(float)

    logret = np.log(close.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan)
    ema_fast = ema(close, 21)
    ema_slow = ema(close, 55)
    atr_s = atr(work, cfg.atr_window).replace([np.inf, -np.inf], np.nan)
    atr_pct_s = (atr_s / (close + 1e-12)).replace([np.inf, -np.inf], np.nan)
    bb = bollinger(close, cfg.bb_window)
    bb_width_s = bb["bandwidth"].replace([np.inf, -np.inf], np.nan)

    ret_short = _safe_ret(close, cfg.short_window)
    ret_medium = _safe_ret(close, cfg.medium_window)
    ret_long = _safe_ret(close, cfg.long_window)

    trend_raw = (float(ema_fast.iloc[-1]) - float(ema_slow.iloc[-1])) / (
        float(close.iloc[-1]) + 1e-12
    )
    trend_score = _clip_float(trend_raw / max(float(atr_pct_s.iloc[-1] or 0.0), 1e-4), -3.0, 3.0)

    er_window = min(cfg.medium_window, max(2, len(close) - 1))
    if len(close) > er_window:
        directional = abs(float(close.iloc[-1] - close.iloc[-er_window - 1]))
        path = float(close.diff().abs().iloc[-er_window:].sum())
        trend_eff = directional / (path + 1e-12)
    else:
        trend_eff = 0.0

    vol_short = float(logret.rolling(cfg.vol_short_window, min_periods=2).std().iloc[-1] or 0.0)
    vol_long = float(logret.rolling(cfg.vol_long_window, min_periods=2).std().iloc[-1] or 0.0)
    vol_ratio = vol_short / max(vol_long, 1e-8)

    bb_width = float(bb_width_s.iloc[-1]) if np.isfinite(float(bb_width_s.iloc[-1])) else 0.0
    bb_rank = _rolling_quantile_rank(bb_width_s, bb_width, cfg.quantile_window)

    vol_mean = volume.rolling(64, min_periods=2).mean()
    vol_std = volume.rolling(64, min_periods=2).std()
    volume_z = float((volume.iloc[-1] - vol_mean.iloc[-1]) / (vol_std.iloc[-1] + 1e-12))
    volume_z = _clip_float(volume_z, -8.0, 8.0)

    range_window = min(cfg.long_window, len(close))
    roll_high = float(high.iloc[-range_window:].max())
    roll_low = float(low.iloc[-range_window:].min())
    range_position = (float(close.iloc[-1]) - roll_low) / max(roll_high - roll_low, 1e-12)
    range_position = _clip_float(range_position, 0.0, 1.0)

    peak = close.cummax()
    trough = close.cummin()
    drawdown = float((peak.iloc[-1] - close.iloc[-1]) / (peak.iloc[-1] + 1e-12))
    rebound = float((close.iloc[-1] - trough.iloc[-1]) / (trough.iloc[-1] + 1e-12))

    bkw = min(cfg.breakout_window, max(2, len(work) - 1))
    prior_high = float(high.iloc[-bkw - 1 : -1].max()) if len(work) > bkw else float(high.iloc[:-1].max()) if len(work) > 1 else float(high.iloc[-1])
    prior_low = float(low.iloc[-bkw - 1 : -1].min()) if len(work) > bkw else float(low.iloc[:-1].min()) if len(work) > 1 else float(low.iloc[-1])
    last_close = float(close.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    breakout_up = bool(last_close > prior_high and ret_short > 0)
    breakout_down = bool(last_close < prior_low and ret_short < 0)
    sweep_high = bool(last_high > prior_high and last_close < prior_high)
    sweep_low = bool(last_low < prior_low and last_close > prior_low)

    atr_pct = float(atr_pct_s.iloc[-1]) if np.isfinite(float(atr_pct_s.iloc[-1])) else 0.0
    shock_score = abs(ret_short) / max(atr_pct * np.sqrt(max(cfg.short_window, 1)), 1e-6)

    return RegimeFeatures(
        timeframe=tf,
        n_bars=int(len(work)),
        close=last_close,
        ret_short=_clip_float(ret_short, -1.0, 1.0),
        ret_medium=_clip_float(ret_medium, -1.0, 1.0),
        ret_long=_clip_float(ret_long, -1.0, 1.0),
        trend_score=trend_score,
        trend_efficiency=_clip_float(trend_eff, 0.0, 1.0),
        realized_vol_short=max(0.0, vol_short),
        realized_vol_long=max(0.0, vol_long),
        vol_ratio=_clip_float(vol_ratio, 0.0, 10.0),
        atr_pct=max(0.0, atr_pct),
        bb_width=max(0.0, bb_width),
        bb_width_quantile=_clip_float(bb_rank, 0.0, 1.0),
        volume_z=volume_z,
        range_position=range_position,
        drawdown=_clip_float(drawdown, 0.0, 1.0),
        rebound_from_low=_clip_float(rebound, 0.0, 10.0),
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        liquidity_sweep_high=sweep_high,
        liquidity_sweep_low=sweep_low,
        shock_score=_clip_float(shock_score, 0.0, 20.0),
    )


__all__ = ["RegimeFeatureConfig", "compute_regime_features"]
