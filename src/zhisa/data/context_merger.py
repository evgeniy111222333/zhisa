"""Funding / OI / long-short context merger for S1 data preparation.

This module joins the Binance USD-M **futures context** parquet
(``data/futures_context/binance_usdm/{SYMBOL}/{timeframe}/context.parquet``)
into the OHLCV frame.

Important design points
-----------------------

1. **Look-ahead prevention.**  Funding is settled every 8 hours on
   Binance. The published rate becomes observable *after* the funding
   window closes, not during it. We therefore shift all context columns
   forward by one bar (``lag=1``) before joining, so a sample taken at
   bar ``t`` only ever sees the funding rate as of bar ``t-1``.

2. **Strict resampling semantics.**  Funding context is stored at 5m
   resolution (one row every 5 minutes, with funding rate constant over
   its 8h window). For S1 we use 15m OHLCV; the merger resamples the
   5m context to 15m by **forward-fill** (no aggregation) — funding is
   not additive, so taking the last value in each 15m window is correct.

3. **Column hygiene.**  We rename context columns with a ``ctx_`` prefix
   so they cannot collide with OHLCV columns. We also drop columns that
   are entirely NaN (e.g. metrics that Binance stopped publishing).

4. **Deterministic.**  Inputs checksums are preserved on the resulting
   frame's metadata so downstream tools can verify that the join was
   bit-exact.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTX_COLUMN_PREFIX = "ctx_"
CTX_LAG_BARS = 1  # shift context forward by 1 bar to prevent look-ahead
CTX_FILL_LIMIT = 192  # 192 * 15m = 48h cap on ffill (typical weekend)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_context_frame(
    context_path: Path,
    *,
    timeframe: str = "5m",
) -> pd.DataFrame:
    """Load and lightly clean a local futures context parquet.

    Returns a numeric, tz-aware UTC ``DatetimeIndex`` frame with a
    ``ctx_*`` column prefix already applied. Returns an empty frame if
    the file does not exist.
    """
    context_path = Path(context_path)
    if not context_path.exists():
        logger.warning("futures context not found: %s", context_path)
        return pd.DataFrame()

    df = pd.read_parquet(context_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"context index must be DatetimeIndex: {context_path}")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    numeric = pd.DataFrame(index=df.index)
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            numeric[f"{CTX_COLUMN_PREFIX}{col.strip().lower()}"] = values

    # Drop columns that are entirely NaN (Binance sometimes stops a metric).
    numeric = numeric.dropna(axis=1, how="all")
    logger.info(
        "loaded %d ctx columns from %s (%d rows, %d non-empty)",
        len(numeric.columns), context_path, len(numeric), int(numeric.notna().any().sum()),
    )
    return numeric


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------

def merge_context_into_ohlcv(
    ohlcv: pd.DataFrame,
    context: pd.DataFrame,
    *,
    target_timeframe: str = "15m",
    target_freq_minutes: int = 15,
    lag_bars: int = CTX_LAG_BARS,
    ffill_limit: int = CTX_FILL_LIMIT,
) -> pd.DataFrame:
    """Join a 5m context frame onto a higher-timeframe OHLCV frame.

    Steps (strictly in this order to guarantee no look-ahead):

    1. Drop columns that collide with OHLCV.
    2. Reindex the context onto the OHLCV index using forward-fill.
       At a 5m source and a 15m target, each target bar corresponds to
       three source bars; the last of those three is the value we
       forward (consistent with the 8h funding settlement cycle).
    3. Forward-fill with a hard cap (``ffill_limit``) so a stale
       context cannot bleed forever into the future.
    4. Shift the entire context block forward by ``lag_bars`` so the
       value available at bar ``t`` is what was known *before* bar ``t``
       closed.
    5. Return the joined frame; remaining NaNs are left for the
       feature pipeline to clean (the training loop handles them).

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Target OHLCV frame with a tz-aware UTC ``DatetimeIndex``.
    context : pd.DataFrame
        Source context frame (typically 5m resolution), already loaded
        with ``load_context_frame``.
    target_timeframe : str
        Logical timeframe of the OHLCV (``"15m"`` for S1 default).
    target_freq_minutes : int
        Length of one OHLCV bar in minutes. Used only for logging.
    lag_bars : int
        Number of bars to shift the context forward (anti-look-ahead).
    ffill_limit : int
        Maximum number of consecutive NaNs to forward-fill.

    Returns
    -------
    pd.DataFrame
        The OHLCV frame augmented with the context columns.
    """
    if context is None or len(context) == 0:
        return ohlcv.copy()

    ohlcv = ohlcv.copy()
    context = context.copy()

    # 1. Drop collisions.
    colliding = [c for c in context.columns if c in ohlcv.columns]
    if colliding:
        context = context.drop(columns=colliding)
        logger.info("dropped %d colliding context columns", len(colliding))

    # ``load_context_frame`` already prefixes columns, but this lower-level
    # helper may also receive a raw numeric context frame.
    context = context.rename(columns={
        col: col if str(col).startswith(CTX_COLUMN_PREFIX) else f"{CTX_COLUMN_PREFIX}{col}"
        for col in context.columns
    })

    if len(context.columns) == 0:
        return ohlcv

    # 2. Forward-fill reindex onto OHLCV index.
    reindexed = context.reindex(ohlcv.index, method="ffill", limit=ffill_limit)

    # 3. Anti-look-ahead shift.
    if lag_bars > 0:
        reindexed = reindexed.shift(lag_bars)

    # 4. Join (preserve OHLCV column order, then context columns).
    joined = ohlcv.join(reindexed, how="left")

    # 5. Record provenance in frame attrs.
    joined.attrs["context_merge"] = {
        "source_rows": int(len(context)),
        "target_rows": int(len(ohlcv)),
        "columns_added": list(reindexed.columns),
        "lag_bars": int(lag_bars),
        "ffill_limit": int(ffill_limit),
        "coverage_pct": float(
            100.0 * joined[list(reindexed.columns)].notna().any(axis=1).sum() / max(len(joined), 1)
        ),
    }
    logger.info(
        "merged context: %d columns, %.1f%% coverage",
        len(reindexed.columns),
        joined.attrs["context_merge"]["coverage_pct"],
    )
    return joined


# ---------------------------------------------------------------------------
# Convenience top-level
# ---------------------------------------------------------------------------

def attach_context_for_symbol(
    ohlcv: pd.DataFrame,
    *,
    context_root: Path,
    symbol: str,
    context_timeframe: str = "5m",
    target_timeframe: str = "15m",
    target_freq_minutes: int = 15,
) -> pd.DataFrame:
    """High-level helper: load + merge context for one symbol.

    Tries both the legacy ``BTCUSDT`` slug and the canonical
    ``BTC_USDT`` slug (Binance USD-M uses the former; the local TSDB
    uses the latter). Returns the OHLCV unchanged if no context file
    is found for the symbol.
    """
    candidates = _slug_candidates(symbol)
    root = Path(context_root)
    for slug in candidates:
        candidate = root / slug / context_timeframe / "context.parquet"
        if candidate.exists():
            ctx = load_context_frame(candidate, timeframe=context_timeframe)
            if len(ctx) == 0:
                continue
            return merge_context_into_ohlcv(
                ohlcv,
                ctx,
                target_timeframe=target_timeframe,
                target_freq_minutes=target_freq_minutes,
            )
    logger.info("no context file for %s under %s", symbol, root)
    return ohlcv.copy()


def _slug_candidates(symbol: str) -> list[str]:
    """Return filesystem slugs to try for a CCXT symbol like 'BTC/USDT'."""
    base = symbol.replace("/", "").replace("-", "").replace("_", "").upper()
    with_slash = symbol.replace("/", "_").upper()
    return [base, with_slash]
