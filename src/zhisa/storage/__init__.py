"""Storage subsystem: TimeSeriesDB and FeatureStore.

Provides local Parquet-based time-series storage, feature computation,
materialisation, and point-in-time serving.
"""

from zhisa.storage.feature_store import FeatureStore, FeatureStoreError
from zhisa.storage.locks import FileLock, FileLockError
from zhisa.storage.quality import QualityIssue, QualityReport, audit_ohlcv, repair_ohlcv
from zhisa.storage.registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureRegistryError,
)
from zhisa.storage.resampler import resample_ohlcv
from zhisa.storage.schema import (
    OHLCV_COLUMNS,
    SeriesKey,
    SeriesMeta,
    Timeframe,
    compute_checksum,
    validate_ohlcv,
)
from zhisa.storage.tsdb import (
    SeriesNotFoundError,
    TimeSeriesDB,
    TimeSeriesDBError,
    ValidationError,
)

__all__ = [
    # Schema
    "OHLCV_COLUMNS",
    "SeriesKey",
    "SeriesMeta",
    "Timeframe",
    "compute_checksum",
    "validate_ohlcv",
    # TSDB
    "TimeSeriesDB",
    "TimeSeriesDBError",
    "SeriesNotFoundError",
    "ValidationError",
    # Resampler
    "resample_ohlcv",
    # Quality
    "QualityIssue",
    "QualityReport",
    "audit_ohlcv",
    "repair_ohlcv",
    # Locks
    "FileLock",
    "FileLockError",
    # Feature Registry
    "FeatureDefinition",
    "FeatureRegistry",
    "FeatureRegistryError",
    # Feature Store
    "FeatureStore",
    "FeatureStoreError",
]
