"""Schemas, types, and validation for the storage subsystem.

Defines the core data types used by both TimeSeriesDB and FeatureStore:
``SeriesKey``, ``SeriesMeta``, ``Timeframe``, plus OHLCV validation helpers.
"""
from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# Canonical OHLCV columns (always lowercase, always this order)
# ────────────────────────────────────────────────────────────────────
OHLCV_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ────────────────────────────────────────────────────────────────────
# Timeframe enum + conversion
# ────────────────────────────────────────────────────────────────────
class Timeframe(str, enum.Enum):
    """Supported OHLCV bar timeframes.

    The ``value`` is the canonical short string (``"1m"``, ``"5m"`` …).
    Use :pyattr:`pandas_freq` for Pandas-compatible offset aliases.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    # ── helpers ──────────────────────────────────────────────────
    @property
    def pandas_freq(self) -> str:
        """Return the Pandas-compatible offset alias."""
        _map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W",
        }
        return _map[self.value]

    @property
    def minutes(self) -> int:
        """Number of minutes per bar."""
        _map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }
        return _map[self.value]

    @classmethod
    def from_str(cls, s: str) -> "Timeframe":
        """Parse a short string into a Timeframe enum member."""
        s = s.strip().lower()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(
            f"Unknown timeframe {s!r}. "
            f"Supported: {[m.value for m in cls]}"
        )

    def can_resample_to(self, target: "Timeframe") -> bool:
        """Return True if *self* can be resampled to *target* (target ≥ self and evenly divides)."""
        if target.minutes < self.minutes:
            return False
        if target.minutes == self.minutes:
            return True
        return target.minutes % self.minutes == 0


# ────────────────────────────────────────────────────────────────────
# SeriesKey — uniquely identifies one time series
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SeriesKey:
    """Unique identifier for a stored time series.

    Convention: ``instrument`` is the exchange symbol in CCXT format
    (e.g. ``"BTC/USDT"``), normalised to a filesystem-safe slug
    internally.  ``timeframe`` is a :class:`Timeframe` member.
    """

    instrument: str
    timeframe: Timeframe

    # Filesystem-safe slug for the instrument (e.g. "BTC_USDT")
    @property
    def instrument_slug(self) -> str:
        return self.instrument.replace("/", "_").replace(" ", "_").upper()

    def __str__(self) -> str:
        return f"{self.instrument}@{self.timeframe.value}"

    @classmethod
    def from_str(cls, s: str) -> "SeriesKey":
        """Parse ``'BTC/USDT@5m'`` into a SeriesKey."""
        parts = s.split("@")
        if len(parts) != 2:
            raise ValueError(f"Invalid SeriesKey string: {s!r}  (expected 'instrument@timeframe')")
        return cls(instrument=parts[0], timeframe=Timeframe.from_str(parts[1]))


# ────────────────────────────────────────────────────────────────────
# SeriesMeta — stored alongside each Parquet file
# ────────────────────────────────────────────────────────────────────
@dataclass
class SeriesMeta:
    """Metadata for a stored time series."""

    key: SeriesKey
    start: datetime
    end: datetime
    row_count: int
    columns: List[str]
    size_bytes: int = 0
    checksum: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "instrument": self.key.instrument,
            "timeframe": self.key.timeframe.value,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "row_count": self.row_count,
            "columns": self.columns,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SeriesMeta":
        return cls(
            key=SeriesKey(
                instrument=d["instrument"],
                timeframe=Timeframe.from_str(d["timeframe"]),
            ),
            start=datetime.fromisoformat(d["start"]),
            end=datetime.fromisoformat(d["end"]),
            row_count=d["row_count"],
            columns=d["columns"],
            size_bytes=d.get("size_bytes", 0),
            checksum=d.get("checksum", ""),
            created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(d["updated_at"]) if "updated_at" in d else datetime.now(timezone.utc),
        )


# ────────────────────────────────────────────────────────────────────
# DataFrame validation
# ────────────────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame, *, strict: bool = True) -> List[str]:
    """Validate that *df* conforms to the ZHISA OHLCV schema.

    Returns a list of error messages (empty = valid).
    When *strict* is True, also checks dtype consistency and ordering.
    """
    errors: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return ["Input is not a DataFrame"]

    if len(df) == 0:
        return ["DataFrame is empty"]

    # Index
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be a DatetimeIndex")

    # Required columns
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col!r}")

    if errors:
        return errors  # can't proceed without columns

    # Dtype
    if strict:
        for col in OHLCV_COLUMNS:
            if not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Column {col!r} must be numeric, got {df[col].dtype}")

    # Monotonic index
    if strict and isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            errors.append("DatetimeIndex must be monotonic increasing")

    return errors


def compute_checksum(df: pd.DataFrame) -> str:
    """Compute a fast MD5 checksum of the DataFrame content."""
    h = hashlib.md5()
    # Hash index
    idx_bytes = df.index.asi8.tobytes() if isinstance(df.index, pd.DatetimeIndex) else str(df.index).encode()
    h.update(idx_bytes)
    # Hash values
    for col in sorted(df.columns):
        h.update(df[col].values.tobytes())
    return h.hexdigest()
