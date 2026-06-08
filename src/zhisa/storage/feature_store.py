"""Feature Store — compute, materialise, and serve features.

The Feature Store sits on top of :class:`TimeSeriesDB` and
:class:`FeatureRegistry`.  It computes features from raw OHLCV data,
caches (materialises) them to Parquet, and provides **point-in-time
correct** serving for both training and inference.

Layout on disk::

    {root}/
      {INSTRUMENT_SLUG}/
        {TIMEFRAME}/
          features_{fingerprint}.parquet
          manifest.json

The manifest tracks which features were materialised, at what version,
and when.  When feature versions change, the old cache is invalidated
and re-computed automatically.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from zhisa.storage.registry import FeatureDefinition, FeatureRegistry
from zhisa.storage.schema import SeriesKey
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureStoreError(Exception):
    """Base exception for Feature Store operations."""


class _Manifest:
    """Persistent metadata for materialised features."""

    def __init__(
        self,
        key: SeriesKey,
        features: List[str],
        fingerprint: str,
        row_count: int = 0,
        created_at: Optional[str] = None,
    ):
        self.key = key
        self.features = features
        self.fingerprint = fingerprint
        self.row_count = row_count
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.key.instrument,
            "timeframe": self.key.timeframe.value,
            "features": self.features,
            "fingerprint": self.fingerprint,
            "row_count": self.row_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_Manifest":
        from zhisa.storage.schema import Timeframe
        return cls(
            key=SeriesKey(d["instrument"], Timeframe.from_str(d["timeframe"])),
            features=d["features"],
            fingerprint=d["fingerprint"],
            row_count=d.get("row_count", 0),
            created_at=d.get("created_at"),
        )


class FeatureStore:
    """Compute, materialise, and serve features on top of TSDB.

    Example::

        from zhisa.storage.builtin_features import create_default_registry

        db = TimeSeriesDB(Path("data/tsdb"))
        registry = create_default_registry()
        store = FeatureStore(Path("data/features"), db, registry)

        key = SeriesKey("BTC/USDT", Timeframe.M5)
        store.materialize(key)

        # Point-in-time: get features for a specific moment
        row = store.get_features_at(key, some_timestamp, lookback=32)

        # Bulk: get the full training matrix
        matrix = store.get_training_matrix(key, features=["rsi_14", "logret_1"])
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        root: Union[str, Path],
        tsdb: TimeSeriesDB,
        registry: FeatureRegistry,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tsdb = tsdb
        self.registry = registry

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _series_dir(self, key: SeriesKey) -> Path:
        return self.root / key.instrument_slug / key.timeframe.value

    def _manifest_path(self, key: SeriesKey) -> Path:
        return self._series_dir(key) / self.MANIFEST_FILE

    def _parquet_path(self, key: SeriesKey, fingerprint: str) -> Path:
        return self._series_dir(key) / f"features_{fingerprint}.parquet"

    def _read_manifest(self, key: SeriesKey) -> Optional[_Manifest]:
        path = self._manifest_path(key)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return _Manifest.from_dict(json.load(f))

    def _write_manifest(self, manifest: _Manifest) -> None:
        path = self._manifest_path(manifest.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def _resolve_features(self, features: Optional[List[str]]) -> List[str]:
        """Resolve feature list: None → all registered features."""
        if features is None:
            return self.registry.list_features()
        # Validate all exist
        for name in features:
            self.registry.get(name)  # raises if missing
        return list(features)

    # ──────────────────────────────────────────────────────────
    # Compute
    # ──────────────────────────────────────────────────────────

    def compute(
        self,
        key: SeriesKey,
        features: Optional[List[str]] = None,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Compute features from raw OHLCV data (not cached).

        Reads the raw series from TSDB, adds lookback rows for warm-up,
        computes all requested features, then trims to the requested
        time range.

        Args:
            key: Series to compute features for.
            features: Feature names to compute (None = all registered).
            start: Inclusive start of the output range.
            end: Inclusive end of the output range.

        Returns:
            A DataFrame with features as columns, indexed by timestamp.
        """
        feature_names = self._resolve_features(features)
        defs = [self.registry.get(n) for n in feature_names]

        # Determine required lookback
        max_lb = max((d.lookback for d in defs), default=0)

        # Read raw data (with extra lookback rows)
        raw = self.tsdb.read(key)
        if len(raw) == 0:
            return pd.DataFrame()

        # Compute each feature
        result_parts: List[pd.DataFrame | pd.Series] = []
        for defn in defs:
            try:
                out = defn.compute_fn(raw)
                if isinstance(out, pd.Series):
                    out = out.rename(defn.name)
                    result_parts.append(out)
                elif isinstance(out, pd.DataFrame):
                    result_parts.append(out)
                else:
                    raise TypeError(
                        f"Feature {defn.name!r} returned {type(out).__name__}, "
                        f"expected Series or DataFrame"
                    )
            except Exception as exc:
                logger.warning("Failed to compute feature %r: %s", defn.name, exc)
                # Add NaN placeholder
                result_parts.append(pd.Series(np.nan, index=raw.index, name=defn.name))

        if not result_parts:
            return pd.DataFrame(index=raw.index)

        result = pd.concat(result_parts, axis=1)
        result = result.replace([np.inf, -np.inf], np.nan)

        # Trim lookback
        if max_lb > 0 and len(result) > max_lb:
            result = result.iloc[max_lb:]

        # Time-range filter
        if start is not None:
            result = result[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result[result.index <= pd.Timestamp(end)]

        return result

    # ──────────────────────────────────────────────────────────
    # Materialise (cache to disk)
    # ──────────────────────────────────────────────────────────

    def materialize(
        self,
        key: SeriesKey,
        features: Optional[List[str]] = None,
    ) -> Path:
        """Compute and cache features to a Parquet file.

        If features have already been materialised with the same version
        fingerprint, re-uses the existing cache.

        Returns:
            Path to the materialised Parquet file.
        """
        feature_names = self._resolve_features(features)
        fingerprint = self.registry.versions_fingerprint(feature_names)

        # Check if already cached with same fingerprint
        manifest = self._read_manifest(key)
        parquet_path = self._parquet_path(key, fingerprint)
        if (
            manifest is not None
            and manifest.fingerprint == fingerprint
            and set(manifest.features) == set(feature_names)
            and parquet_path.exists()
        ):
            logger.debug("Cache hit for %s (fingerprint=%s)", key, fingerprint)
            return parquet_path

        # Compute
        logger.info("Materialising %d features for %s", len(feature_names), key)
        result = self.compute(key, feature_names)

        # Write Parquet
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean up old cache files
        for old_file in parquet_path.parent.glob("features_*.parquet"):
            if old_file != parquet_path:
                old_file.unlink(missing_ok=True)

        result.to_parquet(parquet_path, engine="pyarrow", index=True)

        # Write manifest
        new_manifest = _Manifest(
            key=key,
            features=feature_names,
            fingerprint=fingerprint,
            row_count=len(result),
        )
        self._write_manifest(new_manifest)
        logger.info("Materialised %d rows × %d features for %s", len(result), len(feature_names), key)
        return parquet_path

    def is_materialized(
        self,
        key: SeriesKey,
        features: Optional[List[str]] = None,
    ) -> bool:
        """Check whether features are cached and up-to-date."""
        feature_names = self._resolve_features(features)
        fingerprint = self.registry.versions_fingerprint(feature_names)
        manifest = self._read_manifest(key)
        if manifest is None:
            return False
        if manifest.fingerprint != fingerprint:
            return False
        if set(manifest.features) != set(feature_names):
            return False
        return self._parquet_path(key, fingerprint).exists()

    def invalidate(self, key: SeriesKey) -> None:
        """Remove all cached features for a series."""
        series_dir = self._series_dir(key)
        if series_dir.exists():
            for f in series_dir.glob("features_*.parquet"):
                f.unlink(missing_ok=True)
            manifest_path = self._manifest_path(key)
            if manifest_path.exists():
                manifest_path.unlink()
            logger.info("Invalidated cache for %s", key)

    # ──────────────────────────────────────────────────────────
    # Serve (point-in-time)
    # ──────────────────────────────────────────────────────────

    def _load_or_compute(
        self,
        key: SeriesKey,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load from cache if available, else compute (but don't persist)."""
        feature_names = self._resolve_features(features)
        fingerprint = self.registry.versions_fingerprint(feature_names)
        parquet_path = self._parquet_path(key, fingerprint)

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df

        return self.compute(key, feature_names)

    def get_features_at(
        self,
        key: SeriesKey,
        timestamp: datetime,
        features: Optional[List[str]] = None,
        lookback: int = 1,
    ) -> pd.DataFrame:
        """Point-in-time feature serving.

        Returns a DataFrame with *lookback* rows of features ending at
        (or just before) *timestamp*.  **No future data is included**
        — this is the core anti-look-ahead guarantee.

        Args:
            key: Series to query.
            timestamp: The "as of" time.
            features: Feature names (None = all).
            lookback: Number of rows to return.

        Returns:
            DataFrame with *lookback* rows (or fewer if insufficient history).
        """
        all_features = self._load_or_compute(key, features)

        if len(all_features) == 0:
            return all_features

        ts = pd.Timestamp(timestamp)
        # Strict: only include rows with index ≤ timestamp
        mask = all_features.index <= ts
        available = all_features[mask]

        if lookback > 0 and len(available) > lookback:
            return available.iloc[-lookback:]
        return available

    def get_training_matrix(
        self,
        key: SeriesKey,
        features: Optional[List[str]] = None,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Get a bulk feature matrix for training.

        This is equivalent to :meth:`compute` but prefers cached data
        when available.

        Args:
            key: Series to query.
            features: Feature names (None = all).
            start: Inclusive start.
            end: Inclusive end.
            dropna: If True, drop rows with any NaN.

        Returns:
            A feature DataFrame ready for training.
        """
        all_features = self._load_or_compute(key, features)

        if start is not None:
            all_features = all_features[all_features.index >= pd.Timestamp(start)]
        if end is not None:
            all_features = all_features[all_features.index <= pd.Timestamp(end)]

        if dropna:
            all_features = all_features.dropna()

        return all_features

    # ──────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────

    def list_materialized(self) -> List[Tuple[SeriesKey, List[str]]]:
        """List all materialised (series, features) pairs."""
        result = []
        if not self.root.exists():
            return result
        for inst_dir in sorted(self.root.iterdir()):
            if not inst_dir.is_dir() or inst_dir.name.startswith("."):
                continue
            for tf_dir in sorted(inst_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                manifest_path = tf_dir / self.MANIFEST_FILE
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r") as f:
                            data = json.load(f)
                        manifest = _Manifest.from_dict(data)
                        result.append((manifest.key, manifest.features))
                    except Exception:
                        pass
        return result

    def storage_stats(self) -> Dict[str, Any]:
        """Return storage statistics."""
        total_files = 0
        total_bytes = 0
        series_count = 0
        for inst_dir in self.root.iterdir():
            if not inst_dir.is_dir() or inst_dir.name.startswith("."):
                continue
            for tf_dir in inst_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                series_count += 1
                for f in tf_dir.iterdir():
                    if f.is_file():
                        total_files += 1
                        total_bytes += f.stat().st_size
        return {
            "series_count": series_count,
            "total_files": total_files,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "root": str(self.root),
        }
