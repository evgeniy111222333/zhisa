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


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
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

    out = out[list(OHLCV_COLUMNS)].astype(float)
    out = out.replace([float("inf"), float("-inf")], float("nan")).dropna(how="any")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    errors = validate_ohlcv(out, strict=True)
    if errors:
        raise ValueError("Invalid OHLCV frame: " + "; ".join(errors))
    return out


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
        df = pd.read_csv(path)
        timestamp_col = str(getattr(args, "timestamp_column", "timestamp"))
        if timestamp_col != "timestamp" and timestamp_col in df.columns:
            df = df.rename(columns={timestamp_col: "timestamp"})
        start = parse_utc_timestamp(getattr(args, "start", None))
        end = parse_utc_timestamp(getattr(args, "end", None))
        df = normalize_ohlcv_frame(df)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    df = normalize_ohlcv_frame(df)
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
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "columns": list(df.columns),
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
