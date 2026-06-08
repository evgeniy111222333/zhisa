"""TimeSeriesDB — local Parquet-based time-series storage engine.

Stores OHLCV (and arbitrary columnar) data as Parquet files organised by
instrument and timeframe.  Provides efficient time-range queries, append
with deduplication, resampling, and quality auditing.

Layout on disk::

    {root}/
      {INSTRUMENT_SLUG}/
        {TIMEFRAME}/
          data.parquet
          meta.json

Each ``(instrument, timeframe)`` pair maps to exactly one Parquet file.
Append operations read–merge–write the file (safe for local use; not
designed for concurrent writers).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from zhisa.storage.quality import QualityReport, audit_ohlcv
from zhisa.storage.resampler import resample_ohlcv
from zhisa.storage.schema import (
    OHLCV_COLUMNS,
    SeriesKey,
    SeriesMeta,
    Timeframe,
    compute_checksum,
    validate_ohlcv,
)
from zhisa.storage.locks import FileLock
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


class TimeSeriesDBError(Exception):
    """Base exception for TimeSeriesDB operations."""


class SeriesNotFoundError(TimeSeriesDBError):
    """Raised when a requested series does not exist."""


class ValidationError(TimeSeriesDBError):
    """Raised when data fails schema validation."""


class TimeSeriesDB:
    """Local Parquet-based time-series database.

    Example::

        db = TimeSeriesDB(Path("data/tsdb"))
        key = SeriesKey("BTC/USDT", Timeframe.M5)
        db.ingest(key, ohlcv_df)
        recent = db.read_latest(key, 200)
    """

    DATA_FILE = "data.parquet"
    META_FILE = "meta.json"

    def __init__(
        self,
        root: Union[str, Path],
        *,
        lock_timeout: Optional[float] = 30.0,
        lock_stale_timeout: float = 60.0,
    ) -> None:
        """Initialise the TSDB.

        Args:
            root: Root directory for storage.
            lock_timeout: Max seconds to wait when acquiring the
                per-series file lock (None = block forever).  Set to 0
                to disable locking (NOT recommended outside of tests).
            lock_stale_timeout: Seconds before an unreleased lock is
                considered abandoned and may be forcibly stolen.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_timeout = lock_timeout
        self._lock_stale_timeout = lock_stale_timeout
        logger.debug("TimeSeriesDB initialised at %s (lock_timeout=%s)", self.root, lock_timeout)

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _series_dir(self, key: SeriesKey) -> Path:
        return self.root / key.instrument_slug / key.timeframe.value

    def _data_path(self, key: SeriesKey) -> Path:
        return self._series_dir(key) / self.DATA_FILE

    def _meta_path(self, key: SeriesKey) -> Path:
        return self._series_dir(key) / self.META_FILE

    def _write_meta(self, meta: SeriesMeta) -> None:
        path = self._meta_path(meta.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2, default=str)

    def _read_meta(self, key: SeriesKey) -> SeriesMeta:
        path = self._meta_path(key)
        if not path.exists():
            raise SeriesNotFoundError(f"No metadata for {key}")
        with open(path, "r", encoding="utf-8") as f:
            return SeriesMeta.from_dict(json.load(f))

    def _build_meta(self, key: SeriesKey, df: pd.DataFrame, data_path: Path) -> SeriesMeta:
        size = data_path.stat().st_size if data_path.exists() else 0
        return SeriesMeta(
            key=key,
            start=df.index[0].to_pydatetime(),
            end=df.index[-1].to_pydatetime(),
            row_count=len(df),
            columns=list(df.columns),
            size_bytes=size,
            checksum=compute_checksum(df),
        )

    def _lock(self, key: SeriesKey) -> FileLock:
        """Return a :class:`FileLock` guarding *key*'s data file.

        The lock directory is colocated with the data file (sibling
        ``data.parquet.lock``), so a series is locked independently of
        others.  ``lock_timeout=None`` disables locking entirely.
        """
        if self._lock_timeout is None:
            # Disabled: return a context manager that does nothing.  Use
            # a tiny shim so callers can still use ``with self._lock(k):``.
            class _NoLock:
                def __enter__(self_self): return self_self  # noqa: N805
                def __exit__(self_self, *a): return None
            return _NoLock()  # type: ignore[return-value]
        return FileLock(
            self._data_path(key),
            timeout=self._lock_timeout,
            stale_timeout=self._lock_stale_timeout,
        )

    # ──────────────────────────────────────────────────────────
    # Write operations
    # ──────────────────────────────────────────────────────────

    def ingest(
        self,
        key: SeriesKey,
        df: pd.DataFrame,
        *,
        dedup: bool = True,
        validate: bool = True,
    ) -> SeriesMeta:
        """Ingest (append or overwrite) OHLCV data for a series.

        If the series already exists on disk, the new data is merged with
        existing data, de-duplicated by timestamp, and sorted.

        Args:
            key: The series identifier.
            df: DataFrame with DatetimeIndex and OHLCV columns.
            dedup: If True, remove duplicate timestamps (keep latest).
            validate: If True, validate the schema before writing.

        Returns:
            Updated :class:`SeriesMeta` after the write.

        Raises:
            ValidationError: If *validate* is True and the data is invalid.
        """
        if validate:
            errors = validate_ohlcv(df, strict=True)
            if errors:
                raise ValidationError(
                    f"Validation failed for {key}: " + "; ".join(errors)
                )

        data_path = self._data_path(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Critical section: read–merge–write must be atomic across
        # processes/threads, otherwise two parallel ingests can lose
        # data or corrupt the parquet file.
        with self._lock(key):
            # Merge with existing data if present
            if data_path.exists():
                existing = pd.read_parquet(data_path)
                if not isinstance(existing.index, pd.DatetimeIndex):
                    if "timestamp" in existing.columns:
                        existing = existing.set_index("timestamp")
                    else:
                        existing.index = pd.to_datetime(existing.index)
                merged = pd.concat([existing, df])
            else:
                merged = df.copy()

            # Dedup and sort
            if dedup:
                merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()

            # Write
            merged.to_parquet(data_path, engine="pyarrow", index=True)

            meta = self._build_meta(key, merged, data_path)
            self._write_meta(meta)
        logger.info("Ingested %d rows for %s (total: %d)", len(df), key, meta.row_count)
        return meta

    def ingest_from_csv(
        self,
        key: SeriesKey,
        path: Union[str, Path],
        *,
        timestamp_column: str = "timestamp",
        dedup: bool = True,
    ) -> SeriesMeta:
        """Ingest OHLCV data from a CSV file.

        The CSV must contain at least the OHLCV columns and a timestamp
        column (or a DatetimeIndex-parseable index).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
            df = df.set_index(timestamp_column)
        else:
            df.index = pd.to_datetime(df.index, utc=True)

        df.index.name = "timestamp"

        # Keep only known columns
        keep = [c for c in df.columns if c in OHLCV_COLUMNS or c not in ("timestamp",)]
        df = df[keep].astype({c: float for c in OHLCV_COLUMNS if c in df.columns})

        return self.ingest(key, df, dedup=dedup)

    # ──────────────────────────────────────────────────────────
    # Read operations
    # ──────────────────────────────────────────────────────────

    def read(
        self,
        key: SeriesKey,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read a time series with optional time-range and column filtering.

        Args:
            key: The series to read.
            start: Inclusive lower bound on timestamps.
            end: Inclusive upper bound on timestamps.
            columns: Subset of columns to return (None = all).

        Returns:
            A DataFrame sorted by time.

        Raises:
            SeriesNotFoundError: If the series does not exist.
        """
        data_path = self._data_path(key)
        if not data_path.exists():
            raise SeriesNotFoundError(f"Series not found: {key}")

        # Read with column pruning if possible
        read_cols = columns if columns else None
        df = pd.read_parquet(data_path, columns=read_cols, engine="pyarrow")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index)

        # Time-range filter
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]

        return df

    def read_latest(self, key: SeriesKey, n_bars: int) -> pd.DataFrame:
        """Read the most recent *n_bars* from a series."""
        df = self.read(key)
        return df.iloc[-n_bars:] if len(df) > n_bars else df

    # ──────────────────────────────────────────────────────────
    # Catalog operations
    # ──────────────────────────────────────────────────────────

    def list_series(self) -> List[SeriesKey]:
        """List all stored series keys."""
        keys: List[SeriesKey] = []
        if not self.root.exists():
            return keys
        for instrument_dir in sorted(self.root.iterdir()):
            if not instrument_dir.is_dir() or instrument_dir.name.startswith("."):
                continue
            for tf_dir in sorted(instrument_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                meta_path = tf_dir / self.META_FILE
                if meta_path.exists():
                    try:
                        meta = self._read_meta_from_path(meta_path)
                        keys.append(meta.key)
                    except Exception:
                        # Best-effort: reconstruct from directory names
                        try:
                            tf = Timeframe.from_str(tf_dir.name)
                            instrument = instrument_dir.name.replace("_", "/")
                            keys.append(SeriesKey(instrument=instrument, timeframe=tf))
                        except ValueError:
                            pass
        return keys

    def _read_meta_from_path(self, path: Path) -> SeriesMeta:
        with open(path, "r", encoding="utf-8") as f:
            return SeriesMeta.from_dict(json.load(f))

    def has_series(self, key: SeriesKey) -> bool:
        """Check whether a series exists."""
        return self._data_path(key).exists()

    def get_meta(self, key: SeriesKey) -> SeriesMeta:
        """Get metadata for a series.

        Raises:
            SeriesNotFoundError: If the series does not exist.
        """
        if not self.has_series(key):
            raise SeriesNotFoundError(f"Series not found: {key}")
        return self._read_meta(key)

    def delete_series(self, key: SeriesKey) -> None:
        """Delete a series (data + metadata) from disk."""
        series_dir = self._series_dir(key)
        if not series_dir.exists():
            raise SeriesNotFoundError(f"Series not found: {key}")
        # Lock against concurrent ingest into the same series.
        with self._lock(key):
            import shutil
            shutil.rmtree(series_dir)
        logger.info("Deleted series %s", key)
        # Clean up empty parent directory
        parent = series_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    # ──────────────────────────────────────────────────────────
    # Resample
    # ──────────────────────────────────────────────────────────

    def resample(
        self,
        source_key: SeriesKey,
        target_tf: Union[str, Timeframe],
    ) -> SeriesKey:
        """Resample an existing series to a coarser timeframe and store it.

        Args:
            source_key: Key of the source series.
            target_tf: Target timeframe (string or Timeframe).

        Returns:
            The :class:`SeriesKey` of the newly created series.

        Raises:
            SeriesNotFoundError: If the source does not exist.
            ValueError: If the resample is invalid.
        """
        if isinstance(target_tf, str):
            target_tf = Timeframe.from_str(target_tf)

        target_key = SeriesKey(instrument=source_key.instrument, timeframe=target_tf)

        # Lock source during read so a concurrent delete() or ingest()
        # on source_key cannot race.  The target_key write is locked
        # inside ingest() — and since FileLock is reentrant per thread,
        # we can nest the two locks without deadlocking.  Source is
        # acquired first (before target) by convention: when operating
        # on multiple series, always acquire in lexicographic key order
        # to avoid deadlocks.
        with self._lock(source_key):
            source_df = self.read(source_key)
            resampled = resample_ohlcv(source_df, source_key.timeframe, target_tf)
            self.ingest(target_key, resampled, dedup=True)
        return target_key

    # ──────────────────────────────────────────────────────────
    # Quality audit
    # ──────────────────────────────────────────────────────────

    def audit(
        self,
        key: SeriesKey,
        *,
        expected_freq: Optional[str] = None,
    ) -> QualityReport:
        """Run a quality audit on a stored series.

        Args:
            key: Series to audit.
            expected_freq: Expected bar frequency for gap detection.

        Returns:
            A :class:`QualityReport` with detected issues.
        """
        df = self.read(key)
        freq = expected_freq or key.timeframe.pandas_freq
        return audit_ohlcv(df, expected_freq=freq)
