"""Built-in ZHISA feature definitions.

Wraps the existing ``zhisa.features`` module functions as
:class:`FeatureDefinition` instances and registers them in a
:class:`FeatureRegistry`.  This ensures the Feature Store can compute,
materialise, and serve the same features that the training pipeline uses.
"""
from __future__ import annotations

from typing import List

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
from zhisa.storage.registry import FeatureDefinition, FeatureRegistry


# ────────────────────────────────────────────────────────────────────
# Helper: wrap a simple function into a FeatureDefinition-compatible fn
# ────────────────────────────────────────────────────────────────────

def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


# ────────────────────────────────────────────────────────────────────
# OHLCV-derived features
# ────────────────────────────────────────────────────────────────────

def _logret(lag: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return _safe_log(df["close"]).diff(lag)
    return fn


def _body_over_range(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["open"]).abs() / rng


def _upper_wick_over_range(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["high"] - np.maximum(df["close"], df["open"])) / rng


def _lower_wick_over_range(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (np.minimum(df["close"], df["open"]) - df["low"]) / rng


def _close_over_open(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]) / (df["open"] + 1e-12)


def _hl_over_close(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]) / df["close"]


def _realised_vol(window: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return _safe_log(df["close"]).diff().rolling(window, min_periods=2).std()
    return fn


def _vol_z(window: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        vol = df["volume"]
        mv = vol.rolling(window, min_periods=2).mean()
        sv = vol.rolling(window, min_periods=2).std()
        return (vol - mv) / (sv + 1e-12)
    return fn


def _vol_over_avg(window: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        vol = df["volume"]
        return vol / (vol.rolling(window, min_periods=1).mean() + 1e-12)
    return fn


# ────────────────────────────────────────────────────────────────────
# Indicator features
# ────────────────────────────────────────────────────────────────────

def _atr_fn(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return atr(df, period=period)
    return fn


def _atr_pct_fn(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return atr(df, period=period) / df["close"]
    return fn


def _sma_dist(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        s = sma(df["close"], period)
        return (df["close"] - s) / (df["close"] + 1e-12)
    return fn


def _sma_slope(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        s = sma(df["close"], period)
        shift = period // 2 or 1
        return (s - s.shift(shift)) / (df["close"] + 1e-12)
    return fn


def _ema_dist(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        e = ema(df["close"], period)
        return (df["close"] - e) / (df["close"] + 1e-12)
    return fn


def _rsi_fn(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return rsi(df["close"], period) / 100.0
    return fn


def _bb_pct_b(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return bollinger(df["close"], period=period)["pct_b"]
    return fn


def _bb_bandwidth(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        return bollinger(df["close"], period=period)["bandwidth"]
    return fn


def _donchian_pos(period: int):
    def fn(df: pd.DataFrame) -> pd.Series:
        don = donchian(df["high"], df["low"], period=period)
        return (df["close"] - don["low"]) / (don["high"] - don["low"] + 1e-12)
    return fn


def _vwap_dist(df: pd.DataFrame) -> pd.Series:
    vwap = vwap_session(df)
    return (df["close"] - vwap) / (df["close"] + 1e-12)


# ────────────────────────────────────────────────────────────────────
# Time features
# ────────────────────────────────────────────────────────────────────

def _time_sin_cos(component: str, period: float):
    """Create sin/cos time features."""
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for time features")

        if component == "minute":
            values = idx.minute.to_numpy() + idx.second.to_numpy() / 60.0
        elif component == "hour":
            values = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
        elif component == "dow":
            values = idx.dayofweek.to_numpy() + idx.hour.to_numpy() / 24.0
        elif component == "dom":
            values = idx.day.to_numpy().astype(float)
        elif component == "month":
            values = (idx.month.to_numpy() - 1.0).astype(float)
        else:
            raise ValueError(f"Unknown time component: {component}")

        ang = 2.0 * np.pi * values / period
        return pd.DataFrame({
            f"sin_{component}": np.sin(ang),
            f"cos_{component}": np.cos(ang),
        }, index=idx)
    return fn


# ────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────

def _build_ohlcv_features() -> List[FeatureDefinition]:
    """Build all OHLCV-derived feature definitions."""
    defs = []
    # Log returns at various lags
    for lag in (1, 2, 4, 8, 16):
        defs.append(FeatureDefinition(
            name=f"logret_{lag}",
            group="ohlcv",
            compute_fn=_logret(lag),
            lookback=lag + 1,
            dependencies=["close"],
            description=f"Log-return at lag {lag}",
        ))
    # Body/wick ratios
    defs.extend([
        FeatureDefinition(name="body_over_range", group="ohlcv", compute_fn=_body_over_range,
                          lookback=0, dependencies=["open", "high", "low", "close"],
                          description="Absolute body / bar range"),
        FeatureDefinition(name="upper_wick_over_range", group="ohlcv", compute_fn=_upper_wick_over_range,
                          lookback=0, dependencies=["open", "high", "low", "close"],
                          description="Upper wick / bar range"),
        FeatureDefinition(name="lower_wick_over_range", group="ohlcv", compute_fn=_lower_wick_over_range,
                          lookback=0, dependencies=["open", "high", "low", "close"],
                          description="Lower wick / bar range"),
        FeatureDefinition(name="close_over_open", group="ohlcv", compute_fn=_close_over_open,
                          lookback=0, dependencies=["open", "close"],
                          description="(close - open) / open"),
        FeatureDefinition(name="hl_over_close", group="ohlcv", compute_fn=_hl_over_close,
                          lookback=0, dependencies=["high", "low", "close"],
                          description="(high - low) / close"),
    ])
    # Realised volatility
    for w in (8, 16, 32):
        defs.append(FeatureDefinition(
            name=f"rv_{w}",
            group="ohlcv",
            compute_fn=_realised_vol(w),
            lookback=w + 1,
            dependencies=["close"],
            description=f"Rolling std of log-returns, window={w}",
        ))
    # Volume z-score
    for w in (16, 64):
        defs.append(FeatureDefinition(
            name=f"vol_z_{w}",
            group="ohlcv",
            compute_fn=_vol_z(w),
            lookback=w,
            dependencies=["volume"],
            description=f"Volume z-score, window={w}",
        ))
    defs.append(FeatureDefinition(
        name="vol_over_avg_16",
        group="ohlcv",
        compute_fn=_vol_over_avg(16),
        lookback=16,
        dependencies=["volume"],
        description="Volume / rolling mean volume (16 bars)",
    ))
    return defs


def _build_indicator_features() -> List[FeatureDefinition]:
    """Build indicator-based feature definitions."""
    defs = []
    # ATR
    for period in (14,):
        defs.append(FeatureDefinition(
            name=f"atr_{period}", group="indicators", compute_fn=_atr_fn(period),
            lookback=period + 1, dependencies=["high", "low", "close"],
            description=f"Average True Range, period={period}"))
        defs.append(FeatureDefinition(
            name=f"atr_pct_{period}", group="indicators", compute_fn=_atr_pct_fn(period),
            lookback=period + 1, dependencies=["high", "low", "close"],
            description=f"ATR / close, period={period}"))
    # SMA distance and slope
    for p in (10, 20, 50):
        defs.append(FeatureDefinition(
            name=f"sma_dist_{p}", group="indicators", compute_fn=_sma_dist(p),
            lookback=p, dependencies=["close"],
            description=f"(close - SMA) / close, period={p}"))
        defs.append(FeatureDefinition(
            name=f"sma_slope_{p}", group="indicators", compute_fn=_sma_slope(p),
            lookback=p + p // 2, dependencies=["close"],
            description=f"SMA slope, period={p}"))
    # EMA distance
    for p in (8, 21, 55):
        defs.append(FeatureDefinition(
            name=f"ema_dist_{p}", group="indicators", compute_fn=_ema_dist(p),
            lookback=p, dependencies=["close"],
            description=f"(close - EMA) / close, period={p}"))
    # RSI
    defs.append(FeatureDefinition(
        name="rsi_14", group="indicators", compute_fn=_rsi_fn(14),
        lookback=15, dependencies=["close"],
        description="RSI(14) normalised to [0,1]"))
    # Bollinger
    defs.append(FeatureDefinition(
        name="bb_pct_b_20", group="indicators", compute_fn=_bb_pct_b(20),
        lookback=20, dependencies=["close"],
        description="Bollinger %B, period=20"))
    defs.append(FeatureDefinition(
        name="bb_bw_20", group="indicators", compute_fn=_bb_bandwidth(20),
        lookback=20, dependencies=["close"],
        description="Bollinger bandwidth, period=20"))
    # Donchian
    defs.append(FeatureDefinition(
        name="don_pos_20", group="indicators", compute_fn=_donchian_pos(20),
        lookback=20, dependencies=["high", "low", "close"],
        description="Donchian channel position, period=20"))
    # VWAP distance
    defs.append(FeatureDefinition(
        name="vwap_dist", group="indicators", compute_fn=_vwap_dist,
        lookback=0, dependencies=["high", "low", "close", "volume"],
        description="(close - VWAP) / close"))
    return defs


def _build_time_features() -> List[FeatureDefinition]:
    """Build time-embedding feature definitions."""
    components = [
        ("minute", 60.0),
        ("hour", 24.0),
        ("dow", 7.0),
        ("dom", 31.0),
        ("month", 12.0),
    ]
    defs = []
    for comp, period in components:
        defs.append(FeatureDefinition(
            name=f"time_{comp}",
            group="time",
            compute_fn=_time_sin_cos(comp, period),
            lookback=0,
            dependencies=[],
            description=f"Cyclic sin/cos embedding of {comp}",
            output_columns=[f"sin_{comp}", f"cos_{comp}"],
        ))
    return defs


def register_builtin_features(registry: FeatureRegistry) -> None:
    """Register all built-in ZHISA features in the given registry."""
    registry.register_many(_build_ohlcv_features())
    registry.register_many(_build_indicator_features())
    registry.register_many(_build_time_features())


def create_default_registry() -> FeatureRegistry:
    """Create a new registry pre-populated with all built-in features."""
    reg = FeatureRegistry()
    register_builtin_features(reg)
    return reg
