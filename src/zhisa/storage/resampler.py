"""OHLCV resampling between timeframes.

Implements correct financial-data aggregation rules:
open = first, high = max, low = min, close = last, volume = sum.
Validates that the target timeframe is a whole multiple of the source.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from zhisa.storage.schema import OHLCV_COLUMNS, Timeframe


# Aggregation rules for standard OHLCV columns
_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(
    df: pd.DataFrame,
    source_tf: Timeframe,
    target_tf: Timeframe,
    *,
    extra_columns: Optional[dict[str, str]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Resample an OHLCV DataFrame from *source_tf* to *target_tf*.

    Args:
        df: DataFrame with DatetimeIndex and at least the OHLCV columns.
        source_tf: The current timeframe of *df*.
        target_tf: The desired output timeframe (must be ≥ source).
        extra_columns: Optional mapping ``{col: agg_func}`` for non-OHLCV
            columns (e.g. ``{"regime": "last", "funding": "sum"}``).
        dropna: If True, drop bars where all OHLCV values are NaN
            (typically caused by gaps in the source data).

    Returns:
        A new DataFrame resampled to *target_tf*.

    Raises:
        ValueError: If the target timeframe is shorter than the source,
            or not an even multiple of it.
    """
    if target_tf.minutes < source_tf.minutes:
        raise ValueError(
            f"Cannot downsample from {source_tf.value} to {target_tf.value}: "
            f"target timeframe must be ≥ source"
        )
    if target_tf.minutes == source_tf.minutes:
        return df.copy()

    if not source_tf.can_resample_to(target_tf):
        raise ValueError(
            f"Cannot resample {source_tf.value} → {target_tf.value}: "
            f"{target_tf.minutes} is not an even multiple of {source_tf.minutes}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Build aggregation dict
    agg: dict[str, str] = {}
    for col, func in _OHLCV_AGG.items():
        if col in df.columns:
            agg[col] = func

    if extra_columns:
        for col, func in extra_columns.items():
            if col in df.columns:
                agg[col] = func

    freq = target_tf.pandas_freq
    # Use 'label=left, closed=left' for bar convention:
    # the bar at 14:00 covers [14:00, 14:05).
    resampled = df.resample(freq, label="left", closed="left").agg(agg)

    if dropna:
        # Drop rows where ALL OHLCV values are NaN (empty bars)
        ohlcv_cols = [c for c in OHLCV_COLUMNS if c in resampled.columns]
        if ohlcv_cols:
            resampled = resampled.dropna(subset=ohlcv_cols, how="all")

    return resampled
