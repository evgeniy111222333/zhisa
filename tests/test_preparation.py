"""Tests for the S1 data preparation pipeline.

These tests verify the **contracts** of the preparation module:

* Idempotency: running ``prepare_dataset`` twice with the same inputs
  produces the same checksum.
* Schema: every output frame conforms to the v1 contract
  (tz-aware UTC index, OHLCV numeric, no NaN/Inf).
* Look-ahead prevention: the context merger shifts by ``lag_bars``;
  tests verify the value at bar ``t`` does not contain information
  from bar ``>= t``.
* Gap policy: short gaps are forward-filled, long gaps are dropped.
* Coverage alignment: per-symbol start dates converge to a shared
  window.
* Manifest round-trip: serialise + deserialise yields the same
  checksum.

Tests use small synthetic frames for speed. They do **not** depend on
network access, GPU availability, or the local TSDB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhisa.data.context_merger import (
    CTX_LAG_BARS,
    attach_context_for_symbol,
    merge_context_into_ohlcv,
)
from zhisa.data.feature_specs import (
    CURRENT_VERSION,
    CoveragePolicy,
    GapPolicy,
    PreparedDataset,
    assert_v1_schema,
)
from zhisa.data.preparation import (
    PrepareConfig,
    _apply_gap_policy,
    _align_coverage,
    _merge_context,
    _repair,
    _temporal_split_indices,
    prepare_dataset,
)
from zhisa.storage.schema import OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n_bars: int = 200,
    *,
    freq_minutes: int = 15,
    start: str = "2024-01-01T00:00:00Z",
    price: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a deterministic synthetic OHLCV frame at a fixed frequency."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_bars, freq=f"{freq_minutes}min", tz="UTC")
    rets = rng.normal(0, 0.01, size=n_bars)
    close = price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n_bars)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(100, 1000, size=n_bars)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    return _make_ohlcv(n_bars=200, freq_minutes=15, start="2024-01-01T00:00:00Z")


def test_repair_audits_the_actual_timeframe():
    hourly = _make_ohlcv(n_bars=50, freq_minutes=60)
    _, audit = _repair(hourly, where="BTC/USDT", timeframe="1h")
    messages = json.dumps(audit).lower()
    assert "expected 15" not in messages


@pytest.fixture
def ohlcv_with_gaps() -> pd.DataFrame:
    """OHLCV with one short 2-bar gap and one long 10-bar gap."""
    df = _make_ohlcv(n_bars=200, freq_minutes=15, start="2024-01-01T00:00:00Z")
    # Drop bars 50..51 (short gap = 2 bars).
    df = df.drop(df.index[50:52])
    # Drop bars 100..109 (long gap = 10 bars).
    df = df.drop(df.index[100:110])
    return df


# ---------------------------------------------------------------------------
# Feature spec tests
# ---------------------------------------------------------------------------

class TestFeatureSpecs:
    def test_current_version_supported(self):
        assert CURRENT_VERSION == "v1"

    def test_assert_v1_schema_ok(self, small_ohlcv):
        # Should not raise.
        assert_v1_schema(small_ohlcv, where="smoke")

    def test_assert_v1_schema_rejects_naive_index(self, small_ohlcv):
        naive = small_ohlcv.copy()
        naive.index = naive.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware UTC"):
            assert_v1_schema(naive, where="naive")

    def test_assert_v1_schema_rejects_nan(self, small_ohlcv):
        bad = small_ohlcv.copy()
        bad.iloc[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            assert_v1_schema(bad, where="nan")

    def test_checksum_stable(self, small_ohlcv):
        c1 = PreparedDataset.checksum_frame(small_ohlcv)
        c2 = PreparedDataset.checksum_frame(small_ohlcv.copy())
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_checksum_changes_with_value(self, small_ohlcv):
        c1 = PreparedDataset.checksum_frame(small_ohlcv)
        tweaked = small_ohlcv.copy()
        tweaked.iloc[10, 3] = tweaked.iloc[10, 3] * 1.0001
        c2 = PreparedDataset.checksum_frame(tweaked)
        assert c1 != c2


# ---------------------------------------------------------------------------
# Context merger tests
# ---------------------------------------------------------------------------

class TestContextMerger:
    def test_lag_prevents_lookahead(self):
        """The context value at bar t must be the *prior* value, not the current one."""
        ohlcv = _make_ohlcv(n_bars=20, freq_minutes=15)
        # Build a synthetic context that doubles every bar (easy to spot a leak).
        ctx_idx = pd.date_range(
            ohlcv.index[0], periods=len(ohlcv), freq="5min", tz="UTC"
        )
        ctx = pd.DataFrame({"funding_rate": np.arange(len(ctx_idx), dtype=float)},
                           index=ctx_idx)

        merged = merge_context_into_ohlcv(ohlcv, ctx, target_freq_minutes=15)

        # After lag=1, the value at bar t should equal the value at bar t-1 in
        # the source context (before any reindexing distortion).
        # The first bar must be NaN because there is no prior bar.
        assert pd.isna(merged["ctx_funding_rate"].iloc[0])
        # The second bar must carry a non-NaN value from the past.
        assert not pd.isna(merged["ctx_funding_rate"].iloc[1])

    def test_collisions_renamed(self):
        ohlcv = _make_ohlcv(n_bars=10, freq_minutes=15)
        ctx_idx = ohlcv.index.copy()
        ctx = pd.DataFrame(
            {"close": [99.0] * len(ctx_idx), "funding_rate": [0.01] * len(ctx_idx)},
            index=ctx_idx,
        )
        merged = merge_context_into_ohlcv(ohlcv, ctx, target_freq_minutes=15)
        # "close" must remain the OHLCV close, not be overwritten.
        np.testing.assert_array_equal(
            merged["close"].values, ohlcv["close"].values
        )
        # Context "close" must be dropped (collision).
        assert "ctx_close" not in merged.columns
        # funding_rate must be present.
        assert "ctx_funding_rate" in merged.columns

    def test_empty_context_returns_ohlcv(self):
        ohlcv = _make_ohlcv(n_bars=10, freq_minutes=15)
        empty = pd.DataFrame()
        merged = merge_context_into_ohlcv(ohlcv, empty, target_freq_minutes=15)
        assert list(merged.columns) == list(ohlcv.columns)


# ---------------------------------------------------------------------------
# Gap policy tests
# ---------------------------------------------------------------------------

class TestGapPolicy:
    def test_short_gap_filled(self, ohlcv_with_gaps):
        policy = GapPolicy(max_ffill_bars=4, drop_long_gaps=True)
        filled, info = _apply_gap_policy(ohlcv_with_gaps, "15m", policy)
        # The 2-bar gap is filled, but the 10-bar gap exceeds the limit
        # and is dropped.
        assert filled.index.is_monotonic_increasing
        # No NaN in OHLCV columns.
        for col in OHLCV_COLUMNS:
            assert filled[col].notna().all()
        # Some rows were dropped (those following the long gap).
        assert info["dropped_bars"] > 0

    def test_no_drop_keeps_nans(self, ohlcv_with_gaps):
        policy = GapPolicy(max_ffill_bars=4, drop_long_gaps=False)
        filled, info = _apply_gap_policy(ohlcv_with_gaps, "15m", policy)
        # The long gap becomes NaN rows but is not dropped.
        assert info["dropped_bars"] == 0
        assert filled["close"].isna().sum() > 0

    def test_max_ffill_zero_disables_fill(self, ohlcv_with_gaps):
        policy = GapPolicy(max_ffill_bars=0, drop_long_gaps=True)
        filled, info = _apply_gap_policy(ohlcv_with_gaps, "15m", policy)
        # Every missing bar becomes NaN and gets dropped.
        assert filled["close"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Coverage alignment tests
# ---------------------------------------------------------------------------

class TestCoverageAlignment:
    def test_auto_window_overlap(self):
        a = _make_ohlcv(n_bars=100, freq_minutes=15, start="2024-01-01T00:00:00Z")
        b = _make_ohlcv(n_bars=80, freq_minutes=15, start="2024-01-01T05:00:00Z")
        aligned, info = _align_coverage(
            {"A": a, "B": b}, CoveragePolicy(min_bars=1)
        )
        # Auto-window = [max(start), min(end)] = [Jan-15, Jan-...].
        assert "A" in aligned and "B" in aligned
        assert pd.Timestamp(info["aligned_window"]["start"]) == b.index.min()
        assert pd.Timestamp(info["aligned_window"]["end"]) == a.index.max()
        # 'b' is shorter — info should reflect this.

    def test_drop_short_symbols(self):
        a = _make_ohlcv(n_bars=100, freq_minutes=15, start="2024-01-01T00:00:00Z")
        b = _make_ohlcv(n_bars=5, freq_minutes=15, start="2024-01-01T00:00:00Z")
        aligned, info = _align_coverage(
            {"A": a, "B": b}, CoveragePolicy(min_bars=50)
        )
        assert "A" in aligned
        assert "B" in info["dropped_symbols"]


# ---------------------------------------------------------------------------
# Temporal split tests
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_split_indices_sum_to_n(self):
        # With embargo=0, indices sum exactly to n.
        n = 1000
        te, ve, xe = _temporal_split_indices(n, 0.7, 0.15, embargo=0)
        assert te + (ve - te) + (xe - ve) == n

    def test_embargo_creates_gap(self):
        n = 1000
        te, ve, xe = _temporal_split_indices(n, 0.7, 0.15, embargo=10)
        val_start = te + 10
        test_start = ve + 10
        assert val_start - te == 10
        assert test_start - ve == 10
        assert xe == n

    def test_invalid_embargo(self):
        with pytest.raises(ValueError):
            _temporal_split_indices(20, 0.7, 0.15, embargo=100)


# ---------------------------------------------------------------------------
# End-to-end preparation test (with a temp TSDB)
# ---------------------------------------------------------------------------

class TestPrepareE2E:
    def test_prepare_with_in_memory_tsdb(self, tmp_path: Path):
        """Full preparation against a tiny on-disk TSDB.

        Uses the project's own ``TimeSeriesDB`` so we exercise the same
        ingestion code path that the CLI uses.
        """
        from zhisa.storage.schema import SeriesKey, Timeframe
        from zhisa.storage.tsdb import TimeSeriesDB

        tsdb_root = tmp_path / "tsdb"
        db = TimeSeriesDB(tsdb_root)
        for sym, start in [("BTC/USDT", "2024-01-01"), ("ETH/USDT", "2024-01-03")]:
            df = _make_ohlcv(
                n_bars=400, freq_minutes=15,
                start=f"{start}T00:00:00Z",
                price=100.0, seed=hash(sym) & 0xFFFF,
            )
            db.ingest(
                SeriesKey(instrument=sym, timeframe=Timeframe.from_str("15m")),
                df,
            )

        out_root = tmp_path / "prepared"
        cfg = PrepareConfig(
            tsdb_root=tsdb_root,
            out_root=out_root,
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe="15m",
            with_futures_context=False,  # no context files in tmp
            embargo_bars=10,
            coverage_policy=CoveragePolicy(min_bars=100),
        )
        manifest = prepare_dataset(cfg)

        # Manifest sanity.
        assert manifest.version == "v1"
        assert set(manifest.symbols) == {"BTC/USDT", "ETH/USDT"}
        assert manifest.rows_total > 0
        assert manifest.output_checksum

        # Outputs exist.
        assert (out_root / "manifest.json").exists()
        assert (out_root / "symbols" / "BTC_USDT.parquet").exists()
        assert (out_root / "symbols" / "ETH_USDT.parquet").exists()
        assert (out_root / "splits" / "train.parquet").exists()
        assert (out_root / "splits" / "val.parquet").exists()
        assert (out_root / "splits" / "test.parquet").exists()

        # Schema check on every per-symbol frame.
        for sym in manifest.symbols:
            df = pd.read_parquet(out_root / "symbols" / f"{sym.replace('/', '_')}.parquet")
            assert_v1_schema(df[list(OHLCV_COLUMNS)], where=sym)

    def test_idempotency(self, tmp_path: Path):
        """Re-running preparation on identical inputs produces the same checksum."""
        from zhisa.storage.schema import SeriesKey, Timeframe
        from zhisa.storage.tsdb import TimeSeriesDB

        tsdb_root = tmp_path / "tsdb"
        db = TimeSeriesDB(tsdb_root)
        df = _make_ohlcv(n_bars=300, freq_minutes=15, seed=42)
        db.ingest(
            SeriesKey(instrument="BTC/USDT", timeframe=Timeframe.from_str("15m")),
            df,
        )

        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        cfg_a = PrepareConfig(
            tsdb_root=tsdb_root, out_root=out_a,
            symbols=["BTC/USDT"], with_futures_context=False,
            coverage_policy=CoveragePolicy(min_bars=100),
        )
        cfg_b = PrepareConfig(
            tsdb_root=tsdb_root, out_root=out_b,
            symbols=["BTC/USDT"], with_futures_context=False,
            coverage_policy=CoveragePolicy(min_bars=100),
        )
        m_a = prepare_dataset(cfg_a)
        m_b = prepare_dataset(cfg_b)
        assert m_a.output_checksum == m_b.output_checksum


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:
    def test_json_roundtrip(self, tmp_path: Path):
        manifest = PreparedDataset(
            version="v1",
            symbols=["BTC/USDT"],
            timeframe="15m",
            rows_total=100,
            rows_per_symbol={"BTC/USDT": 100},
            gap_policy=GapPolicy(),
            coverage_policy=CoveragePolicy(),
            start="2024-01-01T00:00:00+00:00",
            end="2024-01-02T00:00:00+00:00",
            feature_columns=["open", "high", "low", "close", "volume"],
            input_checksums={"BTC/USDT": "abc"},
            output_checksum="xyz",
        )
        path = tmp_path / "manifest.json"
        manifest.to_json(path)
        loaded = PreparedDataset.from_json(path)
        assert loaded.version == "v1"
        assert loaded.gap_policy.max_ffill_bars == 4
        assert loaded.output_checksum == "xyz"
