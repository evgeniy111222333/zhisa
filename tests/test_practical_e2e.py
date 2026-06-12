"""Practical end-to-end tests simulating real ZHISA workflows.

These tests model realistic operational scenarios:
  1. Full data pipeline: generate → ingest → features → backtest
  2. Storage integrity: append, dedup, concurrent-like patterns
  3. Feature correctness: no look-ahead, no NaN/Inf leaks
  4. Quality audit: dirty data detection and repair
  5. Resampling correctness: OHLCV aggregation rules
  6. Trading environment: realistic multi-step episodes
  7. Backtest engine: equity curve sanity, metrics validity
  8. Edge cases: empty data, single-row, extreme prices
"""
from __future__ import annotations

import math
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.storage.schema import SeriesKey, Timeframe, OHLCV_COLUMNS, validate_ohlcv, compute_checksum
from zhisa.storage.tsdb import TimeSeriesDB, SeriesNotFoundError, ValidationError
from zhisa.storage.quality import audit_ohlcv, repair_ohlcv, QualityReport
from zhisa.storage.resampler import resample_ohlcv
from zhisa.storage.registry import FeatureDefinition, FeatureRegistry
from zhisa.storage.builtin_features import create_default_registry
from zhisa.storage.feature_store import FeatureStore
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.env.rewards import RewardWeights
from zhisa.backtest.engine import run_backtest, random_policy, buy_and_hold_benchmark
from zhisa.backtest.metrics import compute_metrics


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Provide a clean temporary directory and clean up afterwards."""
    d = tempfile.mkdtemp(prefix="zhisa_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synth_df():
    """Generate a moderately-sized synthetic market (5000 bars, 5min)."""
    cfg = MarketConfig(n_bars=5000, seed=42, freq="5min", initial_price=30000.0)
    df = generate_market(cfg)
    # Drop the 'regime' column to get a pure OHLCV DataFrame
    return df.drop(columns=["regime"], errors="ignore")


@pytest.fixture
def small_synth_df():
    """Small synthetic data (500 bars) for quick env tests."""
    cfg = MarketConfig(n_bars=500, seed=123, freq="5min", initial_price=1000.0)
    df = generate_market(cfg)
    return df.drop(columns=["regime"], errors="ignore")


@pytest.fixture
def tsdb(tmp_dir):
    """A TimeSeriesDB in a fresh temp directory."""
    return TimeSeriesDB(tmp_dir / "tsdb", lock_timeout=None)


@pytest.fixture
def btc_key():
    """A standard BTC/USDT 5-minute SeriesKey."""
    return SeriesKey("BTC/USDT", Timeframe.M5)


@pytest.fixture
def eth_key():
    """A standard ETH/USDT 5-minute SeriesKey."""
    return SeriesKey("ETH/USDT", Timeframe.M5)


# ════════════════════════════════════════════════════════════════════
# SCENARIO 1: Full Data Pipeline — Generate → Ingest → Read → Verify
# ════════════════════════════════════════════════════════════════════

class TestFullDataPipeline:
    """Simulate the most basic real workflow: generate data, store it,
    read it back, and verify integrity."""

    def test_generate_ingest_read_roundtrip(self, tsdb, btc_key, synth_df):
        """Data survives a full generate → ingest → read roundtrip."""
        meta = tsdb.ingest(btc_key, synth_df)

        assert meta.row_count == len(synth_df)
        assert meta.key == btc_key
        assert meta.checksum != ""

        readback = tsdb.read(btc_key)
        assert len(readback) == len(synth_df)
        assert list(readback.columns) == list(synth_df.columns)
        # Values must match exactly.
        # NOTE: Parquet roundtrip strips DatetimeIndex.freq metadata
        # (original has freq='5min', readback has freq=None). This is
        # expected — no production code relies on df.index.freq.
        pd.testing.assert_frame_equal(readback, synth_df, check_names=False, check_freq=False)

    def test_checksum_stability(self, tsdb, btc_key, synth_df):
        """Same data produces the same checksum (deterministic)."""
        c1 = compute_checksum(synth_df)
        c2 = compute_checksum(synth_df.copy())
        assert c1 == c2, "Checksum is not deterministic"

    def test_ingest_twice_dedup(self, tsdb, btc_key, synth_df):
        """Ingesting the same data twice doesn't create duplicates."""
        tsdb.ingest(btc_key, synth_df)
        tsdb.ingest(btc_key, synth_df)  # Duplicate ingest

        readback = tsdb.read(btc_key)
        assert len(readback) == len(synth_df), (
            f"Dedup failed: expected {len(synth_df)} rows, got {len(readback)}"
        )

    def test_incremental_append(self, tsdb, btc_key, synth_df):
        """Ingesting in two halves produces the same result as one full ingest."""
        mid = len(synth_df) // 2
        first_half = synth_df.iloc[:mid]
        second_half = synth_df.iloc[mid:]

        tsdb.ingest(btc_key, first_half)
        tsdb.ingest(btc_key, second_half)

        readback = tsdb.read(btc_key)
        assert len(readback) == len(synth_df)
        pd.testing.assert_frame_equal(readback, synth_df, check_names=False, check_freq=False)

    def test_overlapping_append(self, tsdb, btc_key, synth_df):
        """Overlapping appends correctly deduplicate and prefer latest."""
        mid = len(synth_df) // 2
        overlap = 100  # 100 bars of overlap

        part1 = synth_df.iloc[: mid + overlap]
        part2 = synth_df.iloc[mid:]  # Overlaps by 'overlap' bars

        tsdb.ingest(btc_key, part1)
        tsdb.ingest(btc_key, part2)

        readback = tsdb.read(btc_key)
        assert len(readback) == len(synth_df), (
            f"Overlap dedup failed: expected {len(synth_df)}, got {len(readback)}"
        )

    def test_time_range_query(self, tsdb, btc_key, synth_df):
        """Time-range queries return the correct subset."""
        tsdb.ingest(btc_key, synth_df)

        # Query middle 1/3
        start_ts = synth_df.index[len(synth_df) // 3]
        end_ts = synth_df.index[2 * len(synth_df) // 3]
        subset = tsdb.read(btc_key, start=start_ts, end=end_ts)

        assert subset.index[0] >= start_ts
        assert subset.index[-1] <= end_ts
        assert len(subset) > 0

    def test_read_latest(self, tsdb, btc_key, synth_df):
        """read_latest returns exactly the last N bars."""
        tsdb.ingest(btc_key, synth_df)
        n = 100
        latest = tsdb.read_latest(btc_key, n)
        assert len(latest) == n
        pd.testing.assert_frame_equal(latest, synth_df.iloc[-n:], check_names=False, check_freq=False)

    def test_list_and_delete(self, tsdb, btc_key, eth_key, synth_df):
        """Catalog operations work: list, has, delete."""
        tsdb.ingest(btc_key, synth_df)
        tsdb.ingest(eth_key, synth_df)

        keys = tsdb.list_series()
        assert len(keys) == 2
        assert tsdb.has_series(btc_key)
        assert tsdb.has_series(eth_key)

        tsdb.delete_series(btc_key)
        assert not tsdb.has_series(btc_key)
        assert tsdb.has_series(eth_key)

    def test_read_nonexistent_raises(self, tsdb, btc_key):
        """Reading a non-existent series raises SeriesNotFoundError."""
        with pytest.raises(SeriesNotFoundError):
            tsdb.read(btc_key)


# ════════════════════════════════════════════════════════════════════
# SCENARIO 2: Data Quality — Dirty Data Detection & Repair
# ════════════════════════════════════════════════════════════════════

class TestDataQualityPipeline:
    """Simulate receiving dirty real-world data and running the audit
    and repair pipeline."""

    def _make_clean_df(self, n=200):
        """Helper: make a clean OHLCV DataFrame."""
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        close = np.maximum(close, 1.0)  # keep positive
        return pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": np.abs(rng.normal(100, 20, n)),
        }, index=idx)

    def test_clean_data_passes_audit(self):
        """Clean generated data should pass the quality audit."""
        cfg = MarketConfig(n_bars=1000, seed=99)
        df = generate_market(cfg).drop(columns=["regime"])
        report = audit_ohlcv(df)
        # Synthetic data is well-formed; only tz_naive warning is acceptable
        errors = report.errors
        assert len(errors) == 0, f"Clean data has errors: {[str(e) for e in errors]}"

    def test_detect_nan_injection(self):
        """Audit detects NaN values injected into OHLCV data."""
        df = self._make_clean_df()
        # Inject NaNs at random positions
        df.iloc[50, df.columns.get_loc("close")] = np.nan
        df.iloc[100, df.columns.get_loc("high")] = np.nan

        report = audit_ohlcv(df)
        nan_issues = [i for i in report.issues if i.kind == "nan"]
        assert len(nan_issues) == 1
        assert nan_issues[0].row_count == 2

    def test_detect_inf_injection(self):
        """Audit detects Inf values in OHLCV data."""
        df = self._make_clean_df()
        df.iloc[30, df.columns.get_loc("volume")] = np.inf
        df.iloc[60, df.columns.get_loc("low")] = -np.inf

        report = audit_ohlcv(df)
        inf_issues = [i for i in report.issues if i.kind == "inf"]
        assert len(inf_issues) == 1
        assert inf_issues[0].row_count == 2

    def test_detect_ohlc_violations(self):
        """Audit detects OHLC constraint violations (high < close, etc.)."""
        df = self._make_clean_df()
        # Make high < close (violation)
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["close"] * 0.5
        # Make low > open (violation)
        df.iloc[20, df.columns.get_loc("low")] = df.iloc[20]["open"] * 1.5

        report = audit_ohlcv(df)
        violations = [i for i in report.issues if i.kind == "ohlc_violation"]
        assert len(violations) == 1
        assert violations[0].row_count >= 2

    def test_detect_duplicate_timestamps(self):
        """Audit detects duplicate timestamps."""
        df = self._make_clean_df(100)
        # Duplicate 5 rows
        dup = df.iloc[10:15].copy()
        df_dup = pd.concat([df, dup])

        report = audit_ohlcv(df_dup)
        dup_issues = [i for i in report.issues if i.kind == "duplicate"]
        assert len(dup_issues) == 1
        assert dup_issues[0].row_count == 5

    def test_detect_time_gaps(self):
        """Audit detects gaps in the time series."""
        df = self._make_clean_df(200)
        # Remove a chunk to create a gap
        df = pd.concat([df.iloc[:50], df.iloc[60:]])

        report = audit_ohlcv(df, expected_freq="5min")
        gap_issues = [i for i in report.issues if i.kind == "gap"]
        assert len(gap_issues) == 1

    def test_detect_price_anomalies(self):
        """Audit detects extreme price changes."""
        df = self._make_clean_df()
        # Create a 200% price spike (> 50% threshold)
        df.iloc[80, df.columns.get_loc("close")] = df.iloc[79]["close"] * 3.0

        report = audit_ohlcv(df, price_change_threshold=0.5)
        anomalies = [i for i in report.issues if i.kind == "anomaly"]
        assert len(anomalies) == 1
        assert anomalies[0].row_count >= 1

    def test_repair_fixes_nan_and_violations(self):
        """repair_ohlcv fixes NaN values and OHLC violations."""
        df = self._make_clean_df()
        # Inject problems
        df.iloc[10, df.columns.get_loc("close")] = np.nan
        df.iloc[20, df.columns.get_loc("high")] = df.iloc[20]["close"] * 0.5
        df.iloc[30, df.columns.get_loc("volume")] = -10.0

        report_before = audit_ohlcv(df)
        assert len(report_before.errors) > 0

        repaired, report_after = repair_ohlcv(df, report_before)

        # NaN should be filled
        assert not repaired["close"].isna().any()
        # OHLC constraints should be fixed
        assert (repaired["high"] >= np.maximum(repaired["open"], repaired["close"]) - 1e-12).all()
        assert (repaired["low"] <= np.minimum(repaired["open"], repaired["close"]) + 1e-12).all()
        # Negative volume should be fixed
        assert (repaired["volume"] > 0).all()

    def test_dirty_data_ingest_then_audit(self, tsdb, btc_key):
        """Simulate ingesting data, then running a quality audit through TSDB."""
        df = self._make_clean_df(500)
        # Inject zero-volume bars
        df.iloc[100:105, df.columns.get_loc("volume")] = 0.0

        tsdb.ingest(btc_key, df)
        report = tsdb.audit(btc_key, expected_freq="5min")

        zero_vol = [i for i in report.issues if i.kind == "zero_volume"]
        assert len(zero_vol) == 1
        assert zero_vol[0].row_count == 5


# ════════════════════════════════════════════════════════════════════
# SCENARIO 3: Resampling — OHLCV Aggregation Rules
# ════════════════════════════════════════════════════════════════════

class TestResamplingCorrectness:
    """Verify that OHLCV resampling follows correct financial rules."""

    def test_5m_to_1h_basic(self, synth_df):
        """5-minute bars resample correctly to 1-hour bars."""
        resampled = resample_ohlcv(synth_df, Timeframe.M5, Timeframe.H1)

        assert len(resampled) < len(synth_df)
        # Each 1h bar should aggregate ~12 5min bars
        expected_ratio = 12
        actual_ratio = len(synth_df) / len(resampled)
        assert abs(actual_ratio - expected_ratio) < 2  # Allow small variance

    def test_ohlcv_aggregation_rules(self, synth_df):
        """open=first, high=max, low=min, close=last, volume=sum."""
        resampled = resample_ohlcv(synth_df, Timeframe.M5, Timeframe.H1)

        # Check the first 1h bar manually
        first_hour_end = synth_df.index[0] + pd.Timedelta(hours=1)
        source_bars = synth_df[synth_df.index < first_hour_end]
        target_bar = resampled.iloc[0]

        assert abs(target_bar["open"] - source_bars["open"].iloc[0]) < 1e-10
        assert abs(target_bar["high"] - source_bars["high"].max()) < 1e-10
        assert abs(target_bar["low"] - source_bars["low"].min()) < 1e-10
        assert abs(target_bar["close"] - source_bars["close"].iloc[-1]) < 1e-10
        assert abs(target_bar["volume"] - source_bars["volume"].sum()) < 1e-6

    def test_resampled_ohlc_constraints(self, synth_df):
        """Resampled data must still satisfy OHLC constraints."""
        resampled = resample_ohlcv(synth_df, Timeframe.M5, Timeframe.H1)

        assert (resampled["high"] >= resampled["open"] - 1e-12).all()
        assert (resampled["high"] >= resampled["close"] - 1e-12).all()
        assert (resampled["low"] <= resampled["open"] + 1e-12).all()
        assert (resampled["low"] <= resampled["close"] + 1e-12).all()
        assert (resampled["high"] >= resampled["low"] - 1e-12).all()

    def test_volume_conservation(self, synth_df):
        """Total volume is preserved through resampling."""
        resampled = resample_ohlcv(synth_df, Timeframe.M5, Timeframe.H1)
        orig_total = synth_df["volume"].sum()
        resampled_total = resampled["volume"].sum()
        assert abs(orig_total - resampled_total) < 1e-4, (
            f"Volume not conserved: {orig_total:.2f} → {resampled_total:.2f}"
        )

    def test_resample_invalid_direction_raises(self, synth_df):
        """Cannot resample from 1h to 5m (upsampling not allowed)."""
        with pytest.raises(ValueError):
            resample_ohlcv(synth_df, Timeframe.H1, Timeframe.M5)

    def test_resample_through_tsdb(self, tsdb, btc_key, synth_df):
        """Resampling through TSDB creates a new series and stores it."""
        tsdb.ingest(btc_key, synth_df)
        new_key = tsdb.resample(btc_key, Timeframe.H1)

        assert new_key.timeframe == Timeframe.H1
        assert tsdb.has_series(new_key)

        resampled = tsdb.read(new_key)
        assert len(resampled) > 0
        assert len(resampled) < len(synth_df)


# ════════════════════════════════════════════════════════════════════
# SCENARIO 4: Feature Store — Compute, Materialize, Serve
# ════════════════════════════════════════════════════════════════════

class TestFeatureStorePipeline:
    """Simulate the real feature engineering pipeline:
    ingest data → register features → compute → materialize → serve."""

    def test_compute_builtin_features(self, tsdb, btc_key, synth_df, tmp_dir):
        """All built-in features compute without errors."""
        tsdb.ingest(btc_key, synth_df)
        registry = create_default_registry()
        store = FeatureStore(tmp_dir / "features", tsdb, registry)

        result = store.compute(btc_key)
        assert len(result) > 0
        assert result.shape[1] > 10  # Should have many features
        # No Inf values after computation
        assert not np.isinf(result.values).any(), "Feature computation produced Inf values"

    def test_materialize_and_cache_hit(self, tsdb, btc_key, synth_df, tmp_dir):
        """Materialized features are cached and served from disk."""
        tsdb.ingest(btc_key, synth_df)
        registry = create_default_registry()
        store = FeatureStore(tmp_dir / "features", tsdb, registry)

        # First materialize
        path1 = store.materialize(btc_key)
        assert path1.exists()
        assert store.is_materialized(btc_key)

        # Second call should be a cache hit (same path)
        path2 = store.materialize(btc_key)
        assert path1 == path2

    def test_point_in_time_no_future_leak(self, tsdb, btc_key, synth_df, tmp_dir):
        """Point-in-time serving must NOT include any future data.

        This is the single most critical correctness requirement in
        a trading system — a look-ahead leak invalidates all results.
        """
        tsdb.ingest(btc_key, synth_df)
        registry = create_default_registry()
        store = FeatureStore(tmp_dir / "features", tsdb, registry)
        store.materialize(btc_key)

        # Pick a timestamp in the middle
        mid_idx = len(synth_df) // 2
        as_of = synth_df.index[mid_idx].to_pydatetime()
        lookback = 32

        features = store.get_features_at(btc_key, as_of, lookback=lookback)

        # ALL returned timestamps must be ≤ as_of
        assert (features.index <= pd.Timestamp(as_of)).all(), (
            f"LOOK-AHEAD LEAK: some feature rows have timestamp > {as_of}"
        )
        assert len(features) <= lookback

    def test_training_matrix_shape_and_values(self, tsdb, btc_key, synth_df, tmp_dir):
        """Training matrix has correct shape and no NaN (after dropna)."""
        tsdb.ingest(btc_key, synth_df)
        registry = create_default_registry()
        store = FeatureStore(tmp_dir / "features", tsdb, registry)

        matrix = store.get_training_matrix(btc_key, dropna=True)
        assert len(matrix) > 0
        # After dropna, no NaN values
        assert not matrix.isna().any().any(), "Training matrix contains NaN after dropna"

    def test_invalidate_forces_recompute(self, tsdb, btc_key, synth_df, tmp_dir):
        """Invalidating cache forces recomputation on next access."""
        tsdb.ingest(btc_key, synth_df)
        registry = create_default_registry()
        store = FeatureStore(tmp_dir / "features", tsdb, registry)

        store.materialize(btc_key)
        assert store.is_materialized(btc_key)

        store.invalidate(btc_key)
        assert not store.is_materialized(btc_key)

    def test_custom_feature_registration(self, tsdb, btc_key, synth_df, tmp_dir):
        """Custom user features can be registered and computed."""
        tsdb.ingest(btc_key, synth_df)
        registry = FeatureRegistry()

        # Register a custom feature: simple return
        def my_return(df):
            return df["close"].pct_change()

        registry.register(FeatureDefinition(
            name="my_custom_return",
            group="custom",
            compute_fn=my_return,
            lookback=1,
            dependencies=["close"],
            description="Custom test return",
        ))

        store = FeatureStore(tmp_dir / "features", tsdb, registry)
        result = store.compute(btc_key, features=["my_custom_return"])
        assert "my_custom_return" in result.columns
        assert len(result) > 0


# ════════════════════════════════════════════════════════════════════
# SCENARIO 5: Trading Environment — Realistic Episode Rollouts
# ════════════════════════════════════════════════════════════════════

class TestTradingEnvironment:
    """Simulate realistic trading episodes in the Gymnasium environment."""

    def test_basic_episode_rollout(self, small_synth_df):
        """A full episode runs without errors and produces valid observations."""
        cfg = EnvConfig(
            initial_equity=1.0,
            fee_bps=4.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        env = TradingEnv(small_synth_df, cfg=cfg)
        obs, info = env.reset(seed=42)

        assert "chart" in obs
        assert "numeric" in obs
        assert "context" in obs
        assert obs["chart"].shape == (3, cfg.image_size, cfg.image_size)
        assert obs["numeric"].shape[0] == cfg.window

        # Run a few steps
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert math.isfinite(reward), f"Non-finite reward: {reward}"
            assert math.isfinite(info["equity"]), f"Non-finite equity: {info['equity']}"
            if terminated or truncated:
                break

    def test_hold_preserves_equity(self, small_synth_df):
        """Holding (action=0) with zero fees should not change equity much."""
        cfg = EnvConfig(
            initial_equity=1.0,
            fee_bps=0.0,
            slippage_bps_per_unit=0.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        env = TradingEnv(small_synth_df, cfg=cfg)
        env.reset(seed=42)

        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(0)  # HOLD
            if terminated or truncated:
                break

        # With no position ever opened and zero fees, equity = initial
        assert abs(info["equity"] - 1.0) < 1e-6, (
            f"Holding without position should preserve equity, got {info['equity']}"
        )

    def test_fees_reduce_equity(self, small_synth_df):
        """Opening and closing a position should cost fees."""
        cfg = EnvConfig(
            initial_equity=1.0,
            fee_bps=10.0,  # High fee to make it obvious
            slippage_bps_per_unit=0.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        env = TradingEnv(small_synth_df, cfg=cfg)
        env.reset(seed=42)

        # Open LONG_100
        obs, r, term, trunc, info_open = env.step(6)  # LONG_100
        # Close immediately
        obs, r, term, trunc, info_close = env.step(7)  # CLOSE

        # Equity should have decreased due to fees
        assert info_close["equity"] < 1.0, (
            f"Fees should reduce equity, got {info_close['equity']}"
        )

    def test_episode_length_cap(self, small_synth_df):
        """Episode truncates after the configured number of steps."""
        cap = 50
        cfg = EnvConfig(
            initial_equity=1.0,
            window=32,
            seed=42,
            episode_length=cap,
            kill_on_drawdown=False,
        )
        env = TradingEnv(small_synth_df, cfg=cfg)
        env.reset(seed=42)

        steps = 0
        for _ in range(cap + 100):
            _, _, terminated, truncated, _ = env.step(0)
            steps += 1
            if terminated or truncated:
                break

        assert steps == cap, f"Expected truncation at {cap} steps, ran for {steps}"

    def test_stop_loss_triggers(self, small_synth_df):
        """A tight stop-loss triggers within a reasonable number of steps."""
        cfg = EnvConfig(
            initial_equity=1.0,
            window=32,
            seed=42,
            stop_loss_pct=0.001,  # Very tight 0.1% SL
            kill_on_drawdown=False,
        )
        env = TradingEnv(small_synth_df, cfg=cfg)
        env.reset(seed=42)

        # Open a long position
        env.step(6)  # LONG_100

        # Keep holding — SL should trigger
        sl_triggered = False
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(0)  # HOLD
            if info.get("exit_reason", "").startswith("stop_loss"):
                sl_triggered = True
                break
            if terminated or truncated:
                break

        assert sl_triggered, "Stop-loss did not trigger with a very tight SL"

    def test_rewards_are_finite_throughout_episode(self, small_synth_df):
        """All rewards during a full episode must be finite."""
        cfg = EnvConfig(initial_equity=1.0, window=32, seed=42, kill_on_drawdown=False)
        env = TradingEnv(small_synth_df, cfg=cfg)
        env.reset(seed=42)

        rng = np.random.default_rng(42)
        rewards = []
        for _ in range(300):
            action = int(rng.integers(0, env.action_space.n))
            _, r, terminated, truncated, info = env.step(action)
            rewards.append(r)
            assert math.isfinite(r), f"Non-finite reward at step {len(rewards)}: {r}"
            assert math.isfinite(info["equity"]), f"Non-finite equity at step {len(rewards)}"
            if terminated or truncated:
                break

        assert len(rewards) > 0

    def test_position_bounds(self, small_synth_df):
        """Position size should never exceed the configured maximum."""
        cfg = EnvConfig(
            initial_equity=1.0,
            max_position=1.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        env = TradingEnv(small_synth_df, cfg=cfg)
        env.reset(seed=42)

        rng = np.random.default_rng(42)
        for _ in range(200):
            action = int(rng.integers(0, env.action_space.n))
            _, _, terminated, truncated, info = env.step(action)
            assert abs(info["position"]) <= cfg.max_position + 1e-6, (
                f"Position {info['position']} exceeds max {cfg.max_position}"
            )
            if terminated or truncated:
                break


# ════════════════════════════════════════════════════════════════════
# SCENARIO 6: Backtest Engine — Realistic Trading Simulation
# ════════════════════════════════════════════════════════════════════

class TestBacktestEngine:
    """Simulate running a full backtest with the engine and verifying
    that all metrics are valid."""

    def test_random_policy_backtest(self, small_synth_df):
        """A random policy backtest completes and produces valid metrics."""
        policy = random_policy(seed=42)
        cfg = EnvConfig(
            initial_equity=1.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        result = run_backtest(small_synth_df, policy, cfg=cfg, seed=42)

        assert len(result.equity) > 1
        assert len(result.positions) == len(result.equity)
        assert len(result.prices) == len(result.equity)
        assert result.metrics.n_periods > 0

    def test_backtest_metrics_are_finite(self, small_synth_df):
        """All backtest metrics must be finite numbers."""
        policy = random_policy(seed=42)
        cfg = EnvConfig(
            initial_equity=1.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        result = run_backtest(small_synth_df, policy, cfg=cfg, seed=42)
        m = result.metrics

        for field_name in [
            "total_return", "annualised_return", "annualised_vol",
            "sharpe", "sortino", "calmar", "max_drawdown",
            "win_rate", "profit_factor", "avg_trade", "stability",
        ]:
            value = getattr(m, field_name)
            assert math.isfinite(value), f"Metric {field_name} is not finite: {value}"

    def test_max_drawdown_range(self, small_synth_df):
        """Max drawdown should be in [0, 1] range."""
        policy = random_policy(seed=42)
        cfg = EnvConfig(
            initial_equity=1.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        result = run_backtest(small_synth_df, policy, cfg=cfg, seed=42)
        assert 0.0 <= result.metrics.max_drawdown <= 1.0, (
            f"Max drawdown {result.metrics.max_drawdown} outside [0,1]"
        )

    def test_buy_and_hold_benchmark(self, small_synth_df):
        """Buy-and-hold benchmark produces a valid equity curve."""
        bh = buy_and_hold_benchmark(small_synth_df)
        assert len(bh) == len(small_synth_df)
        assert abs(bh[0] - 1.0) < 1e-10  # Starts at 1.0
        assert np.all(np.isfinite(bh))
        assert np.all(bh > 0)

    def test_always_hold_policy_equity_flat(self, small_synth_df):
        """A policy that always holds (no trading) should keep equity ~1.0."""
        def hold_policy(obs):
            return 0  # HOLD

        cfg = EnvConfig(
            initial_equity=1.0,
            fee_bps=4.0,
            window=32,
            seed=42,
            kill_on_drawdown=False,
        )
        result = run_backtest(small_synth_df, hold_policy, cfg=cfg, seed=42)

        # No trades should happen
        assert result.metrics.n_trades == 0 or result.metrics.avg_trade == 0.0
        # Equity should stay at initial
        assert abs(result.equity[-1] - 1.0) < 0.01, (
            f"Hold-only policy changed equity to {result.equity[-1]}"
        )

    def test_backtest_equity_curve_monotonic_start(self, small_synth_df):
        """Equity curve should start at initial_equity."""
        policy = random_policy(seed=42)
        cfg = EnvConfig(initial_equity=1.0, window=32, seed=42, kill_on_drawdown=False)
        result = run_backtest(small_synth_df, policy, cfg=cfg, seed=42)
        assert abs(result.equity[0] - 1.0) < 1e-10

    def test_deterministic_backtest(self, small_synth_df):
        """Same seed produces identical backtest results."""
        policy1 = random_policy(seed=42)
        policy2 = random_policy(seed=42)
        cfg = EnvConfig(initial_equity=1.0, window=32, seed=42, kill_on_drawdown=False)

        result1 = run_backtest(small_synth_df, policy1, cfg=cfg, seed=42)
        result2 = run_backtest(small_synth_df, policy2, cfg=cfg, seed=42)

        np.testing.assert_array_almost_equal(result1.equity, result2.equity)
        np.testing.assert_array_almost_equal(result1.positions, result2.positions)


# ════════════════════════════════════════════════════════════════════
# SCENARIO 7: Edge Cases — Stress Tests
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases that can cause subtle bugs in production."""

    def test_very_small_prices(self):
        """System handles very small prices (penny stocks / shitcoins)."""
        cfg = MarketConfig(n_bars=500, seed=77, initial_price=0.0001)
        df = generate_market(cfg).drop(columns=["regime"])

        errors = validate_ohlcv(df, strict=True)
        assert len(errors) == 0, f"Small price validation errors: {errors}"

        report = audit_ohlcv(df)
        assert len(report.errors) == 0

    def test_very_large_prices(self):
        """System handles very large prices (BTC at $1M)."""
        cfg = MarketConfig(n_bars=500, seed=88, initial_price=1_000_000.0)
        df = generate_market(cfg).drop(columns=["regime"])

        errors = validate_ohlcv(df, strict=True)
        assert len(errors) == 0, f"Large price validation errors: {errors}"

    def test_minimum_viable_dataframe(self, tsdb, btc_key):
        """System handles a DataFrame with minimum rows (window + 1)."""
        cfg = MarketConfig(n_bars=35, seed=42)  # Just above window=32
        df = generate_market(cfg).drop(columns=["regime"])

        meta = tsdb.ingest(btc_key, df)
        assert meta.row_count == 35

        readback = tsdb.read(btc_key)
        assert len(readback) == 35

    def test_empty_dataframe_validation(self):
        """Validation rejects an empty DataFrame."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        errors = validate_ohlcv(df, strict=True)
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_synthetic_data_reproducibility(self):
        """Same seed produces identical synthetic data."""
        cfg = MarketConfig(n_bars=100, seed=42)
        df1 = generate_market(cfg)
        df2 = generate_market(cfg)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        """Different seeds produce different synthetic data."""
        df1 = generate_market(MarketConfig(n_bars=100, seed=1))
        df2 = generate_market(MarketConfig(n_bars=100, seed=2))

        assert not df1["close"].equals(df2["close"])

    def test_regime_coverage_in_synthetic(self):
        """Synthetic data covers multiple market regimes."""
        cfg = MarketConfig(n_bars=10000, seed=42)
        df = generate_market(cfg)
        unique_regimes = df["regime"].nunique()
        assert unique_regimes >= 3, (
            f"Only {unique_regimes} regimes in 10k bars — regime switching may be broken"
        )

    def test_high_volatility_environment(self):
        """Environment survives high-volatility synthetic data."""
        cfg = MarketConfig(
            n_bars=500, seed=42, base_vol=2.0,  # 4x normal vol
            shock_prob=0.01, shock_size=-0.1,
        )
        df = generate_market(cfg).drop(columns=["regime"])

        env_cfg = EnvConfig(
            initial_equity=1.0, window=32, seed=42,
            kill_on_drawdown=False,
        )
        env = TradingEnv(df, cfg=env_cfg)
        obs, _ = env.reset(seed=42)

        # Run through the entire episode
        rng = np.random.default_rng(42)
        for _ in range(300):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            assert math.isfinite(reward), f"Non-finite reward in high-vol scenario: {reward}"
            assert math.isfinite(info["equity"]), f"Non-finite equity in high-vol scenario"
            if terminated or truncated:
                break

    def test_metrics_on_flat_equity(self):
        """Metrics handle a perfectly flat equity curve without NaN/Inf."""
        flat = np.ones(1000)
        m = compute_metrics(flat)

        assert m.total_return == 0.0
        assert m.max_drawdown == 0.0
        assert math.isfinite(m.sharpe)
        assert math.isfinite(m.sortino)


# ════════════════════════════════════════════════════════════════════
# SCENARIO 8: Full Integration — End-to-End Pipeline
# ════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """The ultimate integration test: generate data, store it, compute
    features, run a backtest, and verify everything works together."""

    def test_full_pipeline_generate_to_backtest(self, tmp_dir):
        """Full pipeline: generate → TSDB → features → backtest → metrics."""
        # 1. Generate synthetic market data
        cfg = MarketConfig(n_bars=1000, seed=42, freq="5min", initial_price=30000.0)
        raw_df = generate_market(cfg)
        ohlcv = raw_df.drop(columns=["regime"])

        # 2. Ingest into TSDB
        db = TimeSeriesDB(tmp_dir / "tsdb", lock_timeout=None)
        key = SeriesKey("BTC/USDT", Timeframe.M5)
        meta = db.ingest(key, ohlcv)
        assert meta.row_count == 1000

        # 3. Audit quality
        report = db.audit(key)
        assert len(report.errors) == 0, f"Quality errors: {[str(e) for e in report.errors]}"

        # 4. Resample to 1h for comparison
        h1_key = db.resample(key, Timeframe.H1)
        h1_data = db.read(h1_key)
        assert len(h1_data) > 0

        # 5. Compute features
        registry = create_default_registry()
        store = FeatureStore(tmp_dir / "features", db, registry)
        store.materialize(key)
        assert store.is_materialized(key)

        # 6. Get training matrix
        matrix = store.get_training_matrix(key, dropna=True)
        assert len(matrix) > 0
        assert not matrix.isna().any().any()

        # 7. Point-in-time serving
        mid_time = ohlcv.index[len(ohlcv) // 2].to_pydatetime()
        pit = store.get_features_at(key, mid_time, lookback=32)
        assert len(pit) <= 32
        assert (pit.index <= pd.Timestamp(mid_time)).all()

        # 8. Run backtest
        policy = random_policy(seed=42)
        env_cfg = EnvConfig(
            initial_equity=1.0, window=32, seed=42,
            kill_on_drawdown=False,
        )
        result = run_backtest(ohlcv, policy, cfg=env_cfg, seed=42)

        # 9. Verify metrics
        assert result.metrics.n_periods > 0
        assert math.isfinite(result.metrics.sharpe)
        assert math.isfinite(result.metrics.max_drawdown)
        assert len(result.equity) > 1
        assert abs(result.equity[0] - 1.0) < 1e-10

    def test_multi_instrument_storage(self, tmp_dir):
        """Multiple instruments co-exist in the same TSDB without interference."""
        db = TimeSeriesDB(tmp_dir / "tsdb", lock_timeout=None)
        instruments = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        dfs = {}
        for inst in instruments:
            cfg = MarketConfig(n_bars=500, seed=hash(inst) % 10000, initial_price=100.0)
            df = generate_market(cfg).drop(columns=["regime"])
            key = SeriesKey(inst, Timeframe.M5)
            db.ingest(key, df)
            dfs[inst] = df

        # Verify all are stored independently
        keys = db.list_series()
        assert len(keys) == 3

        # Verify data isolation
        for inst in instruments:
            key = SeriesKey(inst, Timeframe.M5)
            readback = db.read(key)
            assert len(readback) == len(dfs[inst])
            pd.testing.assert_frame_equal(readback, dfs[inst], check_names=False, check_freq=False)

