"""Shared CLI helpers for real-market OHLCV data.

This module deliberately handles data only. It never creates exchange
orders and never needs API keys; live execution belongs in a future,
separately gated broker adapter.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.storage.quality import audit_ohlcv
from zhisa.storage.schema import OHLCV_COLUMNS, SeriesKey, Timeframe, validate_ohlcv
from zhisa.storage.tsdb import TimeSeriesDB


DATA_SOURCES = ("synthetic", "tsdb", "csv")
DEFAULT_FUTURES_CONTEXT_ROOT = "data/futures_context/binance_usdm"


def add_market_data_args(
    parser: argparse.ArgumentParser,
    *,
    default_source: str = "synthetic",
) -> None:
    """Add common historical-data arguments to a CLI parser."""
    group = parser.add_argument_group("market data")
    group.add_argument(
        "--data-source",
        choices=DATA_SOURCES,
        default=default_source,
        help="Historical data source: synthetic, local TSDB, or CSV.",
    )
    group.add_argument("--tsdb-root", type=str, default="data/tsdb")
    group.add_argument("--symbol", type=str, default="BTC/USDT")
    group.add_argument("--timeframe", type=str, default="5m")
    group.add_argument("--csv", type=str, default=None, help="CSV path when --data-source=csv.")
    group.add_argument("--timestamp-column", type=str, default="timestamp")
    group.add_argument("--start", type=str, default=None, help="Inclusive UTC start timestamp.")
    group.add_argument("--end", type=str, default=None, help="Inclusive UTC end timestamp.")
    group.add_argument(
        "--latest-bars",
        type=int,
        default=None,
        help="Keep only the latest N bars after loading.",
    )
    group.add_argument(
        "--with-futures-context",
        action="store_true",
        help=(
            "Join public futures context columns (funding, open interest, "
            "long/short ratios, taker flow) from a local context parquet."
        ),
    )
    group.add_argument(
        "--futures-context-root",
        type=str,
        default=DEFAULT_FUTURES_CONTEXT_ROOT,
        help="Root containing SYMBOL/timeframe/context.parquet files.",
    )


def parse_utc_timestamp(value: str | None) -> Optional[pd.Timestamp]:
    """Parse a CLI timestamp as UTC, preserving timezone-aware inputs."""
    if value is None or str(value).strip() == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def timestamp_to_ms(value: str | None) -> int | None:
    """Return a UTC millisecond epoch value for exchange APIs."""
    ts = parse_utc_timestamp(value)
    if ts is None:
        return None
    return int(ts.timestamp() * 1000)


def normalize_ohlcv_frame(df: pd.DataFrame, *, keep_extra: bool = False) -> pd.DataFrame:
    """Coerce a DataFrame into the project OHLCV schema."""
    if df is None or len(df) == 0:
        raise ValueError("OHLCV frame is empty")

    out = df.copy()
    rename = {}
    for col in out.columns:
        lowered = str(col).strip().lower()
        if lowered in set(OHLCV_COLUMNS) | {"timestamp"}:
            rename[col] = lowered
    if rename:
        out = out.rename(columns=rename)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.set_index("timestamp")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("OHLCV frame must have a DatetimeIndex or a timestamp column")
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing required columns: {missing}")

    ordered = pd.DataFrame(index=out.index)
    for col in OHLCV_COLUMNS:
        ordered[col] = pd.to_numeric(out[col], errors="coerce")
    if keep_extra:
        for col in out.columns:
            if col in OHLCV_COLUMNS:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.notna().any():
                ordered[str(col)] = numeric

    out = ordered.replace([float("inf"), float("-inf")], float("nan"))
    out = out.dropna(subset=list(OHLCV_COLUMNS), how="any")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if hasattr(df, "attrs"):
        out.attrs = dict(df.attrs)

    errors = validate_ohlcv(out[list(OHLCV_COLUMNS)], strict=True)
    if errors:
        raise ValueError("Invalid OHLCV frame: " + "; ".join(errors))
    return out


def futures_context_symbol_slug(symbol: str) -> str:
    """Return the storage slug used by the local Binance USD-M context files."""
    clean = str(symbol).strip().upper()
    if ":" in clean:
        clean = clean.split(":", 1)[0]
    return clean.replace("/", "").replace("-", "").replace("_", "").replace(" ", "")


def futures_context_path(root: str | Path, symbol: str, timeframe: str) -> Path:
    """Return the expected local parquet path for a futures context series."""
    return Path(root) / futures_context_symbol_slug(symbol) / str(timeframe) / "context.parquet"


def load_futures_context_frame(
    root: str | Path,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Load a local futures context parquet as a numeric, timestamp-indexed frame."""
    path = futures_context_path(root, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"Futures context not found: {path}")

    frame = pd.read_parquet(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Futures context must have a DatetimeIndex or timestamp column: {path}")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    numeric = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        name = str(col).strip().lower()
        if not name or name in OHLCV_COLUMNS:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            numeric[name] = values
    numeric = numeric.replace([float("inf"), float("-inf")], float("nan"))
    if numeric.empty:
        raise ValueError(f"Futures context has no numeric columns: {path}")
    return numeric


def join_futures_context(
    df: pd.DataFrame,
    context: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Left-join context by bar timestamp without backfilling future values."""
    if df.empty:
        return df
    overlap = df.index.intersection(context.index)
    if len(overlap) == 0:
        raise ValueError(
            f"No timestamp overlap between OHLCV and futures context for {symbol}@{timeframe}"
        )

    context_cols = [c for c in context.columns if c not in df.columns]
    colliding = [c for c in context.columns if c in df.columns]
    if colliding:
        renamed = {c: f"futures_context_{c}" for c in colliding}
        context = context.rename(columns=renamed)
        context_cols = list(context.columns)

    # Reindex with forward fill to spread the 8-hour funding rates to 5-minute candles.
    # A limit of 120 (10 hours of 5-min bars) ensures we don't carry stale data forever
    # if the stream dies.
    context_reindexed = context[context_cols].reindex(df.index, method="ffill", limit=120)
    joined = df.join(context_reindexed, how="left")
    joined.attrs["futures_context"] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "columns": context_cols,
        "overlap_rows": int(len(overlap)),
        "coverage_pct": float(100.0 * len(overlap) / max(len(df), 1)),
    }
    return joined


def load_market_dataframe(
    args: Any,
    *,
    seed: int = 0,
    default_bars: int | None = None,
) -> pd.DataFrame:
    """Load a market DataFrame for training, evaluation, or replay."""
    source = str(getattr(args, "data_source", "synthetic"))
    bars = int(default_bars or getattr(args, "bars", 0) or 0)

    if source == "synthetic":
        n_bars = bars if bars > 0 else 8000
        df = generate_market(MarketConfig(n_bars=n_bars, seed=seed))
    elif source == "tsdb":
        tf = Timeframe.from_str(str(getattr(args, "timeframe", "5m")))
        key = SeriesKey(str(getattr(args, "symbol", "BTC/USDT")), tf)
        db = TimeSeriesDB(getattr(args, "tsdb_root", "data/tsdb"))
        start = parse_utc_timestamp(getattr(args, "start", None))
        end = parse_utc_timestamp(getattr(args, "end", None))
        df = db.read(
            key,
            start=start.to_pydatetime() if start is not None else None,
            end=end.to_pydatetime() if end is not None else None,
        )
    elif source == "csv":
        csv_path = getattr(args, "csv", None)
        if not csv_path:
            raise ValueError("--csv is required when --data-source=csv")
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        timestamp_col = str(getattr(args, "timestamp_column", "timestamp"))
        if timestamp_col != "timestamp" and timestamp_col in df.columns:
            df = df.rename(columns={timestamp_col: "timestamp"})
        start = parse_utc_timestamp(getattr(args, "start", None))
        end = parse_utc_timestamp(getattr(args, "end", None))
        df = normalize_ohlcv_frame(df, keep_extra=True)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    df = normalize_ohlcv_frame(df, keep_extra=True)
    if bool(getattr(args, "with_futures_context", False)):
        symbol = str(getattr(args, "symbol", "BTC/USDT"))
        timeframe = str(getattr(args, "timeframe", "5m"))
        context_root = getattr(args, "futures_context_root", DEFAULT_FUTURES_CONTEXT_ROOT)
        context = load_futures_context_frame(context_root, symbol, timeframe)
        df = join_futures_context(df, context, symbol=symbol, timeframe=timeframe)
        df = normalize_ohlcv_frame(df, keep_extra=True)

    limit = getattr(args, "latest_bars", None)
    if limit is None and source != "synthetic" and bars > 0:
        limit = bars
    if limit is not None and int(limit) > 0:
        df = df.iloc[-int(limit):]
    if len(df) == 0:
        raise ValueError("No bars left after applying data filters")

    name = getattr(args, "symbol", None)
    timeframe = getattr(args, "timeframe", None)
    if name and timeframe:
        df.name = f"{name}@{timeframe}"
    return df


def frame_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact JSON-friendly summary for a loaded OHLCV frame."""
    report = audit_ohlcv(df)
    context_cols = [c for c in df.columns if c not in OHLCV_COLUMNS]
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "columns": list(df.columns),
        "context_columns": context_cols,
        "context_non_null_pct": {
            col: float(100.0 * df[col].notna().sum() / max(len(df), 1))
            for col in context_cols
        },
        "quality_clean": bool(report.clean),
        "quality_issues": [
            {
                "kind": issue.kind,
                "severity": issue.severity,
                "row_count": int(issue.row_count),
                "message": issue.message,
            }
            for issue in report.issues
        ],
    }


def series_key_from_args(args: Any) -> SeriesKey:
    """Build a storage key from common CLI args."""
    return SeriesKey(
        instrument=str(getattr(args, "symbol", "BTC/USDT")),
        timeframe=Timeframe.from_str(str(getattr(args, "timeframe", "5m"))),
    )
