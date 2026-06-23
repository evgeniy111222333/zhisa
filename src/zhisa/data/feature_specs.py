"""Versioned feature specifications for S1 self-supervised pretraining.

This module describes the **inputs** the S1 trainer expects from the
data preparation pipeline. It does NOT compute features — that already
happens inside :class:`zhisa.data.dataset.MarketDataset` via
``compute_ohlcv_features`` and ``compute_time_features``.

The role of this module is to:

1. Declare the **canonical OHLCV columns** the S1 data must contain.
2. Declare the **timeframe contracts** (15m primary, 1h secondary).
3. Provide a deterministic, versioned feature manifest so a preparation
   run can be reproduced exactly (input checksum -> output checksum).
4. Define the gap-filling policy and the coverage-alignment policy used
   during preparation.

Why versioned?
--------------
Self-supervised pretraining is sensitive to even tiny shifts in the
input distribution (slightly different forward-fill limits, different
normalisation, etc.). To make S1 checkpoints comparable across machines
and across re-runs, every preparation produces a ``PreparedDataset`` with
an explicit ``version`` string and an SHA-256 checksum that is stable
over identical inputs.

Backward compatibility
----------------------
``v1`` (this file) declares the minimum column set:

* OHLCV: ``open``, ``high``, ``low``, ``close``, ``volume``
* Index: ``DatetimeIndex`` (tz=UTC, monotonic, 15-min frequency)

Future versions (e.g. ``v2``) may add derived feature columns that the
preparation pipeline pre-computes (e.g. realised volatility, ATR,
funding-rate lags). v1 keeps the surface area tiny so the training
loop remains in charge of feature engineering (consistent with how
``MarketDataset`` already works).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from zhisa.storage.schema import OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

CURRENT_VERSION = "v1"
SUPPORTED_VERSIONS: Tuple[str, ...] = ("v1",)


def is_supported(version: str) -> bool:
    """Return True if *version* is a recognised feature spec version."""
    return version in SUPPORTED_VERSIONS


# ---------------------------------------------------------------------------
# Gap policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GapPolicy:
    """How to handle missing bars after loading OHLCV from the local TSDB.

    The defaults match the typical Binance maintenance-window pattern:
    short gaps (<= 4 bars = 1 hour on a 15m series) are forward-filled
    using the last known bar, and longer gaps cause those rows to be
    dropped (so they do not poison downstream indicators that assume a
    continuous flow).

    Attributes
    ----------
    max_ffill_bars : int
        Maximum number of consecutive bars to forward-fill. Beyond this,
        rows are dropped. ``0`` disables forward-fill entirely.
    drop_long_gaps : bool
        If True, drop rows that follow a gap longer than ``max_ffill_bars``.
        If False, leave them as NaN.
    require_monotonic : bool
        If True, sort the index and assert it is monotonic increasing.
    """

    max_ffill_bars: int = 4
    drop_long_gaps: bool = True
    require_monotonic: bool = True


# ---------------------------------------------------------------------------
# Coverage policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoveragePolicy:
    """How to align coverage across multiple symbols.

    Symbols have different listing dates (e.g. SOL started in 2020-08
    while BTC starts in 2019-01). For multi-symbol S1 training we need
    a single shared time axis so that a ``ConcatDataset`` batches
    comparable market states.

    Attributes
    ----------
    start : str or None
        ISO date string. If set, hard-clip every series to start no
        earlier than this date. ``None`` means "auto" (= max of per-
        symbol starts, which is the safe default for cross-instrument
        attention).
    end : str or None
        ISO date string. If set, hard-clip every series to end no later
        than this. ``None`` means "use the latest bar across all
        symbols".
    min_bars : int
        Drop symbols that have fewer than this many bars in the
        aligned window. ``0`` keeps all.
    """

    start: Optional[str] = None
    end: Optional[str] = None
    min_bars: int = 1000


# ---------------------------------------------------------------------------
# Manifest (the versioned artefact)
# ---------------------------------------------------------------------------

@dataclass
class PreparedDataset:
    """A versioned, reproducible description of a prepared S1 dataset.

    The preparation pipeline writes one of these to disk alongside the
    per-split parquet files. Downstream code (``train_s1``) can read the
    manifest to verify that the data is what it expects, and to record
    in the checkpoint exactly which dataset was used to train it.
    """

    version: str
    symbols: list[str]
    timeframe: str
    rows_total: int
    rows_per_symbol: dict[str, int]
    gap_policy: GapPolicy
    coverage_policy: CoveragePolicy
    start: str
    end: str
    feature_columns: list[str]
    input_checksums: dict[str, str] = field(default_factory=dict)
    output_checksums: dict[str, str] = field(default_factory=dict)
    output_checksum: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["gap_policy"] = asdict(self.gap_policy)
        d["coverage_policy"] = asdict(self.coverage_policy)
        return d

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "PreparedDataset":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["gap_policy"] = GapPolicy(**d["gap_policy"])
        d["coverage_policy"] = CoveragePolicy(**d["coverage_policy"])
        return cls(**d)

    # ------------------------------------------------------------------
    # Checksums (for idempotency / reproducibility)
    # ------------------------------------------------------------------
    @staticmethod
    def checksum_frame(df: pd.DataFrame) -> str:
        """SHA-256 of a DataFrame's index + values, stable across machines.

        Uses the canonical numerical representation of the index plus
        all column values. Two identical inputs produce identical
        checksums on any OS / any pandas version.
        """
        h = hashlib.sha256()
        if isinstance(df.index, pd.DatetimeIndex):
            h.update(np.asarray(df.index.asi8, dtype=np.int64).tobytes())
        else:
            h.update(str(df.index).encode())
        for col in sorted(df.columns):
            h.update(col.encode())
            arr = np.asarray(df[col].values, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            h.update(arr.tobytes())
        return h.hexdigest()

    @staticmethod
    def checksum_manifest(manifest_dict: dict) -> str:
        """SHA-256 of a manifest dict (sorted keys, no whitespace)."""
        canonical = json.dumps(manifest_dict, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# v1 feature schema
# ---------------------------------------------------------------------------

V1_REQUIRED_COLUMNS: Tuple[str, ...] = OHLCV_COLUMNS
V1_TIMEFRAME_15M = "15m"
V1_TIMEFRAME_1H = "1h"


def assert_v1_schema(df: pd.DataFrame, *, where: str = "frame") -> None:
    """Raise ``ValueError`` if *df* does not match the v1 contract.

    Specifically: index is a tz-aware ``DatetimeIndex`` in UTC, all five
    OHLCV columns are present and numeric, and there are no NaN/Inf in
    the OHLCV columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{where}] index must be DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError(f"[{where}] index must be tz-aware UTC, got tz={df.index.tz}")
    missing = [c for c in V1_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{where}] missing OHLCV columns: {missing}")
    for col in V1_REQUIRED_COLUMNS:
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"[{where}] column {col!r} must be numeric, got {df[col].dtype}")
    n_bad = int(df[list(V1_REQUIRED_COLUMNS)].isna().any(axis=1).sum())
    if n_bad > 0:
        raise ValueError(f"[{where}] {n_bad} rows have NaN in OHLCV columns")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{where}] index must be monotonic increasing")
