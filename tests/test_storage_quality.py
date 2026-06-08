"""Tests for data quality audit and repair."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.storage.quality import QualityReport, audit_ohlcv, repair_ohlcv


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A clean synthetic OHLCV DataFrame."""
    df = generate_market(MarketConfig(n_bars=500, seed=42))
    # Keep only standard OHLCV columns for clean testing
    return df[["open", "high", "low", "close", "volume"]]


# ────────────────────────────────────────────────────────────────────
# Clean data
# ────────────────────────────────────────────────────────────────────

class TestCleanData:
    def test_clean_df_passes_audit(self, clean_df: pd.DataFrame):
        report = audit_ohlcv(clean_df)
        assert report.total_rows == 500
        # Synthetic data should have no errors
        # (may have warnings for gaps if freq detection fails)
        assert len(report.errors) == 0

    def test_empty_report_is_clean(self):
        report = QualityReport()
        assert report.clean
        assert report.n_issues == 0

    def test_empty_df(self):
        report = audit_ohlcv(pd.DataFrame())
        assert report.total_rows == 0


# ────────────────────────────────────────────────────────────────────
# Duplicate detection
# ────────────────────────────────────────────────────────────────────

class TestDuplicates:
    def test_detect_duplicates(self, clean_df: pd.DataFrame):
        # Create duplicates by repeating some rows
        dup_df = pd.concat([clean_df, clean_df.iloc[:10]])
        report = audit_ohlcv(dup_df)
        dup_issues = [i for i in report.issues if i.kind == "duplicate"]
        assert len(dup_issues) == 1
        assert dup_issues[0].row_count == 10

    def test_repair_removes_duplicates(self, clean_df: pd.DataFrame):
        dup_df = pd.concat([clean_df, clean_df.iloc[:10]])
        repaired, report = repair_ohlcv(dup_df)
        assert len(repaired) == len(clean_df)
        assert not repaired.index.duplicated().any()


# ────────────────────────────────────────────────────────────────────
# NaN / Inf detection
# ────────────────────────────────────────────────────────────────────

class TestNaNInf:
    def test_detect_nan(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.iloc[10, df.columns.get_loc("close")] = np.nan
        df.iloc[20, df.columns.get_loc("high")] = np.nan
        report = audit_ohlcv(df)
        nan_issues = [i for i in report.issues if i.kind == "nan"]
        assert len(nan_issues) == 1
        assert nan_issues[0].row_count == 2
        assert not report.clean

    def test_detect_inf(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.iloc[5, df.columns.get_loc("close")] = np.inf
        report = audit_ohlcv(df)
        inf_issues = [i for i in report.issues if i.kind == "inf"]
        assert len(inf_issues) == 1

    def test_repair_fills_nan(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.iloc[10, df.columns.get_loc("close")] = np.nan
        df.iloc[20, df.columns.get_loc("close")] = np.inf
        repaired, report = repair_ohlcv(df)
        assert not repaired["close"].isna().any()
        assert not repaired["close"].isin([np.inf, -np.inf]).any()


# ────────────────────────────────────────────────────────────────────
# Zero-volume detection
# ────────────────────────────────────────────────────────────────────

class TestZeroVolume:
    def test_detect_zero_volume(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.iloc[5, df.columns.get_loc("volume")] = 0
        df.iloc[15, df.columns.get_loc("volume")] = -1
        report = audit_ohlcv(df)
        vol_issues = [i for i in report.issues if i.kind == "zero_volume"]
        assert len(vol_issues) == 1
        assert vol_issues[0].row_count == 2
        # Zero-volume is a warning, not error
        assert vol_issues[0].severity == "warning"

    def test_repair_fills_zero_volume(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.iloc[5, df.columns.get_loc("volume")] = 0
        repaired, report = repair_ohlcv(df)
        assert (repaired["volume"] > 0).all()


# ────────────────────────────────────────────────────────────────────
# OHLC constraint violations
# ────────────────────────────────────────────────────────────────────

class TestOHLCViolations:
    def test_detect_high_too_low(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        # Set high below close (violation)
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["close"] * 0.5
        report = audit_ohlcv(df)
        viol_issues = [i for i in report.issues if i.kind == "ohlc_violation"]
        assert len(viol_issues) == 1
        assert not report.clean

    def test_detect_low_too_high(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        # Set low above close (violation)
        df.iloc[10, df.columns.get_loc("low")] = df.iloc[10]["close"] * 1.5
        report = audit_ohlcv(df)
        viol_issues = [i for i in report.issues if i.kind == "ohlc_violation"]
        assert len(viol_issues) >= 1

    def test_detect_high_below_low(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        h_idx = df.columns.get_loc("high")
        l_idx = df.columns.get_loc("low")
        # Swap high and low
        df.iloc[10, h_idx], df.iloc[10, l_idx] = df.iloc[10]["low"] * 0.5, df.iloc[10]["high"] * 1.5
        report = audit_ohlcv(df)
        viol_issues = [i for i in report.issues if i.kind == "ohlc_violation"]
        assert len(viol_issues) >= 1

    def test_repair_clamps_ohlc(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["close"] * 0.5
        repaired, report = repair_ohlcv(df)
        # After repair, high should be >= max(open, close)
        max_oc = np.maximum(repaired["open"].values, repaired["close"].values)
        assert (repaired["high"].values >= max_oc - 1e-12).all()


# ────────────────────────────────────────────────────────────────────
# Price anomalies
# ────────────────────────────────────────────────────────────────────

class TestPriceAnomalies:
    def test_detect_price_spike(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        # Create a 200% price jump
        df.iloc[50, df.columns.get_loc("close")] = df.iloc[49]["close"] * 3.0
        report = audit_ohlcv(df, price_change_threshold=0.5)
        anomaly_issues = [i for i in report.issues if i.kind == "anomaly"]
        assert len(anomaly_issues) >= 1
        assert anomaly_issues[0].severity == "warning"


# ────────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────────

class TestReport:
    def test_summary_string(self, clean_df: pd.DataFrame):
        report = audit_ohlcv(clean_df)
        summary = report.summary()
        assert "Quality report" in summary
        assert "500 rows" in summary

    def test_errors_vs_warnings(self):
        report = QualityReport()
        from zhisa.storage.quality import QualityIssue
        report.issues.append(QualityIssue(kind="nan", severity="error", message="test", row_count=1))
        report.issues.append(QualityIssue(kind="gap", severity="warning", message="test", row_count=1))
        assert len(report.errors) == 1
        assert len(report.warnings) == 1


# ────────────────────────────────────────────────────────────────────
# Timezone consistency (Issue #3)
# ────────────────────────────────────────────────────────────────────

class TestTimezoneConsistency:
    def test_tz_naive_emits_warning(self, clean_df: pd.DataFrame):
        # Strip tz info.
        df = clean_df.copy()
        df.index = df.index.tz_localize(None)
        report = audit_ohlcv(df)
        tz_issues = [i for i in report.issues if i.kind == "tz_naive"]
        assert len(tz_issues) == 1
        assert tz_issues[0].severity == "warning"
        assert "tz_localize" in tz_issues[0].message

    def test_tz_utc_passes(self, clean_df: pd.DataFrame):
        # clean_df is already UTC by default — must not emit tz warning.
        report = audit_ohlcv(clean_df)
        tz_issues = [i for i in report.warnings if i.kind in ("tz_naive", "tz_non_utc")]
        assert tz_issues == []

    def test_tz_non_utc_emits_warning(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.index = df.index.tz_convert("America/New_York")
        report = audit_ohlcv(df)
        tz_issues = [i for i in report.issues if i.kind == "tz_non_utc"]
        assert len(tz_issues) == 1
        assert tz_issues[0].severity == "warning"
        assert "America/New_York" in tz_issues[0].message

    def test_tz_check_disabled_when_expected_tz_none(self, clean_df: pd.DataFrame):
        df = clean_df.copy()
        df.index = df.index.tz_localize(None)
        report = audit_ohlcv(df, expected_tz=None)
        tz_issues = [i for i in report.issues if i.kind in ("tz_naive", "tz_non_utc")]
        assert tz_issues == []

    def test_custom_expected_tz_acknowledged(self, clean_df: pd.DataFrame):
        # If we say the data should be NY time, UTC data should warn.
        df = clean_df.copy()  # tz=UTC
        report = audit_ohlcv(df, expected_tz="America/New_York")
        tz_issues = [i for i in report.issues if i.kind == "tz_non_utc"]
        assert len(tz_issues) == 1
        assert "America/New_York" in tz_issues[0].message


# ────────────────────────────────────────────────────────────────────
# Variable-length offset handling (Issue #2)
# ────────────────────────────────────────────────────────────────────

class TestVariableLengthOffsets:
    def test_monthly_freq_skips_gap_check(self, clean_df: pd.DataFrame):
        from zhisa.storage.quality import _offset_to_timedelta

        # The helper returns None for variable-length offsets.
        monthly = pd.tseries.frequencies.to_offset("1ME")
        assert _offset_to_timedelta(monthly) is None
        # And for Year, Quarter, BusinessDay too.
        assert _offset_to_timedelta(pd.tseries.frequencies.to_offset("1YE")) is None
        assert _offset_to_timedelta(pd.tseries.frequencies.to_offset("1QE")) is None
        assert _offset_to_timedelta(pd.tseries.frequencies.to_offset("1B")) is None

    def test_fixed_length_offsets_return_timedelta(self):
        from zhisa.storage.quality import _offset_to_timedelta

        for s in ("1min", "5min", "1h", "1D", "1W"):
            td = _offset_to_timedelta(pd.tseries.frequencies.to_offset(s))
            assert isinstance(td, pd.Timedelta), f"failed for {s}"
            assert td > pd.Timedelta(0)

    def test_audit_with_monthly_freq_emits_gap_check_skipped(self, clean_df: pd.DataFrame):
        report = audit_ohlcv(clean_df, expected_freq="1ME")
        skipped = [i for i in report.issues if i.kind == "gap_check_skipped"]
        assert len(skipped) == 1
        assert skipped[0].severity == "warning"
        assert "month" in skipped[0].message or "ME" in skipped[0].message
