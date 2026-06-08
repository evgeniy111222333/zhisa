"""Tests for Feature Store: registry, compute, materialise, serve."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.storage.builtin_features import create_default_registry, register_builtin_features
from zhisa.storage.feature_store import FeatureStore
from zhisa.storage.registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureRegistryError,
)
from zhisa.storage.schema import SeriesKey, Timeframe
from zhisa.storage.tsdb import TimeSeriesDB


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_tsdb(tmp_path: Path) -> TimeSeriesDB:
    return TimeSeriesDB(tmp_path / "tsdb")


@pytest.fixture
def tmp_store_root(tmp_path: Path) -> Path:
    return tmp_path / "features"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return generate_market(MarketConfig(n_bars=500, seed=42))


@pytest.fixture
def btc_key() -> SeriesKey:
    return SeriesKey("BTC/USDT", Timeframe.M5)


@pytest.fixture
def populated_tsdb(tmp_tsdb: TimeSeriesDB, sample_df: pd.DataFrame, btc_key: SeriesKey) -> TimeSeriesDB:
    tmp_tsdb.ingest(btc_key, sample_df)
    return tmp_tsdb


# ────────────────────────────────────────────────────────────────────
# Feature Registry tests
# ────────────────────────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_register_and_get(self):
        reg = FeatureRegistry()
        defn = FeatureDefinition(
            name="test_feat", group="test", version=1,
            compute_fn=lambda df: df["close"].diff(),
            lookback=1, dependencies=["close"],
        )
        reg.register(defn)
        assert reg.has("test_feat")
        got = reg.get("test_feat")
        assert got.name == "test_feat"

    def test_list_features(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="a", group="g1", compute_fn=lambda df: df["close"]))
        reg.register(FeatureDefinition(name="b", group="g2", compute_fn=lambda df: df["close"]))
        reg.register(FeatureDefinition(name="c", group="g1", compute_fn=lambda df: df["close"]))
        assert reg.list_features() == ["a", "b", "c"]
        assert reg.list_features("g1") == ["a", "c"]
        assert reg.list_features("g2") == ["b"]

    def test_list_groups(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="a", group="ohlcv", compute_fn=lambda df: df["close"]))
        reg.register(FeatureDefinition(name="b", group="indicators", compute_fn=lambda df: df["close"]))
        assert sorted(reg.list_groups()) == ["indicators", "ohlcv"]

    def test_get_nonexistent_raises(self):
        reg = FeatureRegistry()
        with pytest.raises(FeatureRegistryError):
            reg.get("nonexistent")

    def test_unregister(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="x", group="g", compute_fn=lambda df: df["close"]))
        assert reg.has("x")
        reg.unregister("x")
        assert not reg.has("x")

    def test_unregister_nonexistent_raises(self):
        reg = FeatureRegistry()
        with pytest.raises(FeatureRegistryError):
            reg.unregister("nope")

    def test_max_lookback(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="a", group="g", compute_fn=lambda df: df["close"], lookback=5))
        reg.register(FeatureDefinition(name="b", group="g", compute_fn=lambda df: df["close"], lookback=20))
        reg.register(FeatureDefinition(name="c", group="g", compute_fn=lambda df: df["close"], lookback=10))
        assert reg.max_lookback() == 20
        assert reg.max_lookback(["a", "c"]) == 10

    def test_versions_fingerprint(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="a", group="g", version=1, compute_fn=lambda df: df["close"]))
        fp1 = reg.versions_fingerprint(["a"])
        # Bump version
        reg.register(FeatureDefinition(name="a", group="g", version=2, compute_fn=lambda df: df["close"]))
        fp2 = reg.versions_fingerprint(["a"])
        assert fp1 != fp2

    def test_size_and_len(self):
        reg = FeatureRegistry()
        assert len(reg) == 0
        reg.register(FeatureDefinition(name="a", group="g", compute_fn=lambda df: df["close"]))
        assert len(reg) == 1

    def test_contains(self):
        reg = FeatureRegistry()
        reg.register(FeatureDefinition(name="a", group="g", compute_fn=lambda df: df["close"]))
        assert "a" in reg
        assert "b" not in reg


# ────────────────────────────────────────────────────────────────────
# Builtin features
# ────────────────────────────────────────────────────────────────────

class TestBuiltinFeatures:
    def test_default_registry_populated(self):
        reg = create_default_registry()
        assert len(reg) > 20  # Should have lots of features
        assert "logret_1" in reg
        assert "rsi_14" in reg
        assert "atr_14" in reg

    def test_groups_present(self):
        reg = create_default_registry()
        groups = reg.list_groups()
        assert "ohlcv" in groups
        assert "indicators" in groups
        assert "time" in groups

    def test_all_builtins_compute(self, sample_df: pd.DataFrame):
        """Every built-in feature should compute without error."""
        reg = create_default_registry()
        for name in reg.list_features():
            defn = reg.get(name)
            try:
                result = defn.compute_fn(sample_df)
                assert result is not None, f"Feature {name} returned None"
            except Exception as e:
                pytest.fail(f"Feature {name!r} failed to compute: {e}")


# ────────────────────────────────────────────────────────────────────
# Feature Store: compute
# ────────────────────────────────────────────────────────────────────

class TestFeatureStoreCompute:
    def test_compute_single_feature(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        result = store.compute(btc_key, features=["logret_1"])
        assert "logret_1" in result.columns
        assert len(result) > 0

    def test_compute_multiple_features(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        features = ["logret_1", "rsi_14", "atr_14"]
        result = store.compute(btc_key, features=features)
        for f in features:
            assert f in result.columns

    def test_compute_all_features(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        result = store.compute(btc_key)
        assert len(result.columns) > 20

    def test_compute_with_time_range(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey, sample_df: pd.DataFrame):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        mid = sample_df.index[250].to_pydatetime()
        result = store.compute(btc_key, features=["logret_1"], start=mid)
        assert result.index[0] >= pd.Timestamp(mid)


# ────────────────────────────────────────────────────────────────────
# Feature Store: materialise
# ────────────────────────────────────────────────────────────────────

class TestFeatureStoreMaterialize:
    def test_materialize_creates_parquet(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        features = ["logret_1", "rsi_14"]
        path = store.materialize(btc_key, features=features)
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_is_materialized(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        features = ["logret_1"]
        assert not store.is_materialized(btc_key, features)
        store.materialize(btc_key, features)
        assert store.is_materialized(btc_key, features)

    def test_cache_hit(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        """Second materialize should use cache."""
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        features = ["logret_1"]
        path1 = store.materialize(btc_key, features)
        mtime1 = path1.stat().st_mtime
        path2 = store.materialize(btc_key, features)
        mtime2 = path2.stat().st_mtime
        assert path1 == path2
        assert mtime1 == mtime2  # file wasn't rewritten

    def test_invalidate(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        features = ["logret_1"]
        store.materialize(btc_key, features)
        assert store.is_materialized(btc_key, features)
        store.invalidate(btc_key)
        assert not store.is_materialized(btc_key, features)

    def test_version_bump_invalidates(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        """Changing a feature version should cause re-materialisation."""
        reg = FeatureRegistry()
        defn = FeatureDefinition(
            name="custom", group="test", version=1,
            compute_fn=lambda df: df["close"].diff(),
            lookback=1, dependencies=["close"],
        )
        reg.register(defn)
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        store.materialize(btc_key, ["custom"])
        assert store.is_materialized(btc_key, ["custom"])

        # Bump version
        defn_v2 = FeatureDefinition(
            name="custom", group="test", version=2,
            compute_fn=lambda df: df["close"].pct_change(),
            lookback=1, dependencies=["close"],
        )
        reg.register(defn_v2)
        assert not store.is_materialized(btc_key, ["custom"])

    def test_list_materialized(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        assert store.list_materialized() == []
        store.materialize(btc_key, ["logret_1"])
        result = store.list_materialized()
        assert len(result) == 1
        assert result[0][0] == btc_key


# ────────────────────────────────────────────────────────────────────
# Feature Store: point-in-time serving
# ────────────────────────────────────────────────────────────────────

class TestPointInTimeServing:
    def test_get_features_at_no_future_leak(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey, sample_df: pd.DataFrame):
        """Point-in-time serving must not include future data."""
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        store.materialize(btc_key, ["logret_1"])

        # Pick a timestamp in the middle
        mid_ts = sample_df.index[250].to_pydatetime()
        result = store.get_features_at(btc_key, mid_ts, features=["logret_1"], lookback=10)

        # ALL returned timestamps must be <= mid_ts
        assert (result.index <= pd.Timestamp(mid_ts)).all()
        assert len(result) <= 10

    def test_get_features_at_lookback(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey, sample_df: pd.DataFrame):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        ts = sample_df.index[-1].to_pydatetime()
        result = store.get_features_at(btc_key, ts, features=["logret_1"], lookback=20)
        assert len(result) == 20

    def test_get_features_at_insufficient_history(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey, sample_df: pd.DataFrame):
        """If there's not enough history, return what's available."""
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        # Request more lookback than available data
        ts = sample_df.index[5].to_pydatetime()
        result = store.get_features_at(btc_key, ts, features=["logret_1"], lookback=1000)
        # Should return whatever is available, not crash
        assert len(result) > 0
        assert len(result) <= 1000


# ────────────────────────────────────────────────────────────────────
# Feature Store: training matrix
# ────────────────────────────────────────────────────────────────────

class TestTrainingMatrix:
    def test_get_training_matrix(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        result = store.get_training_matrix(btc_key, features=["logret_1", "rsi_14"])
        assert "logret_1" in result.columns
        assert "rsi_14" in result.columns
        # With dropna=True, should have no NaN
        assert not result.isna().any().any()

    def test_get_training_matrix_with_range(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey, sample_df: pd.DataFrame):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        start = sample_df.index[100].to_pydatetime()
        end = sample_df.index[400].to_pydatetime()
        result = store.get_training_matrix(btc_key, features=["logret_1"], start=start, end=end)
        assert result.index[0] >= pd.Timestamp(start)
        assert result.index[-1] <= pd.Timestamp(end)

    def test_training_matrix_prefers_cache(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        """If materialised, training matrix should read from cache."""
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        features = ["logret_1"]
        store.materialize(btc_key, features)
        # Should not raise / should work fine from cache
        result = store.get_training_matrix(btc_key, features=features)
        assert len(result) > 0


# ────────────────────────────────────────────────────────────────────
# Feature Store: storage stats
# ────────────────────────────────────────────────────────────────────

class TestStorageStats:
    def test_storage_stats_empty(self, tmp_store_root: Path, tmp_tsdb: TimeSeriesDB):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, tmp_tsdb, reg)
        stats = store.storage_stats()
        assert stats["series_count"] == 0
        assert stats["total_files"] == 0

    def test_storage_stats_after_materialize(self, populated_tsdb: TimeSeriesDB, tmp_store_root: Path, btc_key: SeriesKey):
        reg = create_default_registry()
        store = FeatureStore(tmp_store_root, populated_tsdb, reg)
        store.materialize(btc_key, ["logret_1"])
        stats = store.storage_stats()
        assert stats["series_count"] == 1
        assert stats["total_files"] >= 2  # parquet + manifest
        assert stats["total_bytes"] > 0
