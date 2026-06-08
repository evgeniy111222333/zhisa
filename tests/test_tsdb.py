"""Tests for TimeSeriesDB: ingest, read, catalog, resample, audit."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.storage.schema import SeriesKey, Timeframe
from zhisa.storage.tsdb import (
    SeriesNotFoundError,
    TimeSeriesDB,
    ValidationError,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> TimeSeriesDB:
    """A fresh TimeSeriesDB in a temporary directory."""
    return TimeSeriesDB(tmp_path / "tsdb")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small synthetic OHLCV DataFrame for testing."""
    return generate_market(MarketConfig(n_bars=500, seed=42))


@pytest.fixture
def btc_5m_key() -> SeriesKey:
    return SeriesKey("BTC/USDT", Timeframe.M5)


# ────────────────────────────────────────────────────────────────────
# Ingest + Read round-trip
# ────────────────────────────────────────────────────────────────────

class TestIngestAndRead:
    def test_basic_round_trip(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        """Ingest data and read it back — should be identical."""
        meta = tmp_db.ingest(btc_5m_key, sample_df)
        assert meta.row_count == len(sample_df)
        assert meta.key == btc_5m_key

        read_back = tmp_db.read(btc_5m_key)
        assert len(read_back) == len(sample_df)
        # OHLCV columns match
        for col in ("open", "high", "low", "close", "volume"):
            np.testing.assert_array_almost_equal(
                read_back[col].values, sample_df[col].values, decimal=5
            )

    def test_append_with_dedup(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        """Appending overlapping data should deduplicate by timestamp."""
        first_half = sample_df.iloc[:300]
        second_half = sample_df.iloc[200:]  # overlaps 200..300

        tmp_db.ingest(btc_5m_key, first_half)
        meta = tmp_db.ingest(btc_5m_key, second_half, dedup=True)

        # Should have 500 unique rows, not 600
        assert meta.row_count == len(sample_df)
        read_back = tmp_db.read(btc_5m_key)
        assert len(read_back) == len(sample_df)
        # No duplicate timestamps
        assert not read_back.index.duplicated().any()

    def test_append_updates_values(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        """When appending overlapping data with dedup, the latest values win."""
        tmp_db.ingest(btc_5m_key, sample_df)

        # Create modified version of last 100 rows
        modified = sample_df.iloc[-100:].copy()
        modified["close"] = modified["close"] * 1.1  # bump prices 10%
        tmp_db.ingest(btc_5m_key, modified, dedup=True)

        read_back = tmp_db.read(btc_5m_key)
        # The last 100 rows should have the modified close
        np.testing.assert_array_almost_equal(
            read_back["close"].values[-100:],
            modified["close"].values,
            decimal=5,
        )


# ────────────────────────────────────────────────────────────────────
# Time-range queries
# ────────────────────────────────────────────────────────────────────

class TestTimeRangeQueries:
    def test_start_filter(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        midpoint = sample_df.index[len(sample_df) // 2]
        result = tmp_db.read(btc_5m_key, start=midpoint.to_pydatetime())
        assert len(result) > 0
        assert result.index[0] >= midpoint

    def test_end_filter(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        midpoint = sample_df.index[len(sample_df) // 2]
        result = tmp_db.read(btc_5m_key, end=midpoint.to_pydatetime())
        assert len(result) > 0
        assert result.index[-1] <= midpoint

    def test_start_and_end(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        q1 = sample_df.index[100]
        q3 = sample_df.index[400]
        result = tmp_db.read(btc_5m_key, start=q1.to_pydatetime(), end=q3.to_pydatetime())
        assert result.index[0] >= q1
        assert result.index[-1] <= q3

    def test_read_latest(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        n = 50
        result = tmp_db.read_latest(btc_5m_key, n)
        assert len(result) == n
        # Should be the last n rows
        np.testing.assert_array_almost_equal(
            result["close"].values,
            sample_df["close"].values[-n:],
            decimal=5,
        )

    def test_column_filter(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        result = tmp_db.read(btc_5m_key, columns=["close", "volume"])
        assert list(result.columns) == ["close", "volume"]


# ────────────────────────────────────────────────────────────────────
# Catalog operations
# ────────────────────────────────────────────────────────────────────

class TestCatalog:
    def test_list_empty(self, tmp_db: TimeSeriesDB):
        assert tmp_db.list_series() == []

    def test_list_after_ingest(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        keys = tmp_db.list_series()
        assert len(keys) == 1
        assert keys[0] == btc_5m_key

    def test_has_series(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        assert not tmp_db.has_series(btc_5m_key)
        tmp_db.ingest(btc_5m_key, sample_df)
        assert tmp_db.has_series(btc_5m_key)

    def test_get_meta(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        meta = tmp_db.get_meta(btc_5m_key)
        assert meta.row_count == len(sample_df)
        assert meta.key == btc_5m_key
        assert meta.size_bytes > 0
        assert len(meta.checksum) > 0

    def test_delete_series(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        tmp_db.ingest(btc_5m_key, sample_df)
        assert tmp_db.has_series(btc_5m_key)
        tmp_db.delete_series(btc_5m_key)
        assert not tmp_db.has_series(btc_5m_key)

    def test_multiple_series(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame):
        k1 = SeriesKey("BTC/USDT", Timeframe.M5)
        k2 = SeriesKey("ETH/USDT", Timeframe.H1)
        tmp_db.ingest(k1, sample_df)
        tmp_db.ingest(k2, sample_df)
        keys = tmp_db.list_series()
        assert len(keys) == 2


# ────────────────────────────────────────────────────────────────────
# Resample
# ────────────────────────────────────────────────────────────────────

class TestResample:
    def test_resample_5m_to_1h(self, tmp_db: TimeSeriesDB, btc_5m_key: SeriesKey):
        df = generate_market(MarketConfig(n_bars=1000, seed=42))
        tmp_db.ingest(btc_5m_key, df)
        target_key = tmp_db.resample(btc_5m_key, "1h")
        assert target_key.timeframe == Timeframe.H1
        assert tmp_db.has_series(target_key)

        resampled = tmp_db.read(target_key)
        # 1000 5m bars ≈ 83 1h bars (give or take)
        assert len(resampled) > 0
        assert len(resampled) < len(df)


# ────────────────────────────────────────────────────────────────────
# Error handling
# ────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_read_nonexistent(self, tmp_db: TimeSeriesDB, btc_5m_key: SeriesKey):
        with pytest.raises(SeriesNotFoundError):
            tmp_db.read(btc_5m_key)

    def test_get_meta_nonexistent(self, tmp_db: TimeSeriesDB, btc_5m_key: SeriesKey):
        with pytest.raises(SeriesNotFoundError):
            tmp_db.get_meta(btc_5m_key)

    def test_delete_nonexistent(self, tmp_db: TimeSeriesDB, btc_5m_key: SeriesKey):
        with pytest.raises(SeriesNotFoundError):
            tmp_db.delete_series(btc_5m_key)

    def test_ingest_invalid_df(self, tmp_db: TimeSeriesDB, btc_5m_key: SeriesKey):
        bad_df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValidationError):
            tmp_db.ingest(btc_5m_key, bad_df)

    def test_ingest_empty_df(self, tmp_db: TimeSeriesDB, btc_5m_key: SeriesKey):
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValidationError):
            tmp_db.ingest(btc_5m_key, empty_df)


# ────────────────────────────────────────────────────────────────────
# Audit
# ────────────────────────────────────────────────────────────────────

class TestAudit:
    def test_audit_clean_data(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey):
        # Drop the 'regime' column if present (it's synthetic-specific)
        df = sample_df[[c for c in sample_df.columns if c in ("open", "high", "low", "close", "volume")]]
        tmp_db.ingest(btc_5m_key, df)
        report = tmp_db.audit(btc_5m_key)
        # Synthetic data should be mostly clean
        assert report.total_rows == len(df)


# ────────────────────────────────────────────────────────────────────
# CSV ingest
# ────────────────────────────────────────────────────────────────────

class TestCSVIngest:
    def test_ingest_from_csv(self, tmp_db: TimeSeriesDB, sample_df: pd.DataFrame, btc_5m_key: SeriesKey, tmp_path: Path):
        # Write a CSV
        csv_path = tmp_path / "test.csv"
        export_df = sample_df[["open", "high", "low", "close", "volume"]].copy()
        export_df.index.name = "timestamp"
        export_df.to_csv(csv_path)

        meta = tmp_db.ingest_from_csv(btc_5m_key, csv_path)
        assert meta.row_count == len(sample_df)

        read_back = tmp_db.read(btc_5m_key)
        assert len(read_back) == len(sample_df)
