"""Data quality audit and repair for OHLCV time series.

Detects gaps, NaN/Inf values, zero-volume bars, OHLC constraint
violations, duplicates, and price anomalies.  Provides an automatic
repair function that fixes the most common issues.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from zhisa.storage.schema import OHLCV_COLUMNS


@dataclass
class QualityIssue:
    """A single quality issue detected in the data."""

    kind: str           # e.g. "gap", "nan", "zero_volume", "ohlc_violation", "duplicate", "anomaly"
    severity: str       # "warning" or "error"
    message: str
    row_count: int = 0  # number of affected rows
    indices: List = field(default_factory=list)  # affected index values (capped for memory)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.kind}: {self.message} ({self.row_count} rows)"


@dataclass
class QualityReport:
    """Result of a quality audit on an OHLCV DataFrame."""

    issues: List[QualityIssue] = field(default_factory=list)
    total_rows: int = 0
    clean: bool = True  # True if no errors (warnings are OK)

    @property
    def errors(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    def summary(self) -> str:
        lines = [f"Quality report: {self.total_rows} rows, {self.n_issues} issues"]
        for iss in self.issues:
            lines.append(f"  {iss}")
        if not self.issues:
            lines.append("  ✓ No issues found")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# Maximum number of index values stored per issue (to avoid memory bloat)
_MAX_INDICES = 100


def _offset_to_timedelta(freq) -> "Optional[pd.Timedelta]":
    """Convert a pandas frequency offset to a fixed :class:`pd.Timedelta`.

    Returns ``None`` for variable-length offsets (month, year, quarter,
    business day) where a single Timedelta cannot represent the period.
    Gap detection is impossible for those — callers should skip it.

    Uses ``pd.Timedelta(offset)`` which is the canonical, supported
    conversion path for fixed-length offsets in modern pandas.  This
    replaces an earlier ``hasattr``/``try/except`` chain that was hard
    to read and silently fell back to a guessed interval.

    ``Week`` is handled explicitly because ``pd.Timedelta(Week(...))``
    raises on some pandas versions even though the duration is fixed.
    """
    if freq is None:
        return None
    if isinstance(freq, pd.offsets.Week):
        return pd.Timedelta(days=7 * freq.n)
    try:
        return pd.Timedelta(freq)
    except (ValueError, TypeError):
        # Variable-length offsets (MonthEnd, YearBegin, BusinessDay, ...)
        # raise ValueError when converted to a fixed Timedelta.
        return None


def audit_ohlcv(
    df: pd.DataFrame,
    *,
    expected_freq: Optional[str] = None,
    expected_tz: Optional[str] = "UTC",
    price_change_threshold: float = 0.5,
) -> QualityReport:
    """Run a comprehensive quality audit on an OHLCV DataFrame.

    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns.
        expected_freq: Expected bar frequency (e.g. ``"5min"``).
            If provided, gaps are detected based on this frequency.
            If None, the frequency is inferred from the data.
        expected_tz: Expected timezone of the index.  If ``"UTC"`` (default),
            a warning is emitted when the index is tz-naive or in a
            different timezone.  Set to ``None`` to disable the check.
        price_change_threshold: Flag bars where ``|close/prev_close - 1|``
            exceeds this fraction as price anomalies (default: 50%).

    Returns:
        A :class:`QualityReport` with all detected issues.
    """
    report = QualityReport(total_rows=len(df))

    if len(df) == 0:
        return report

    # ── 1. Duplicates ──────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        dup_mask = df.index.duplicated(keep="first")
        n_dup = int(dup_mask.sum())
        if n_dup > 0:
            report.issues.append(QualityIssue(
                kind="duplicate",
                severity="error",
                message=f"{n_dup} duplicate timestamps found",
                row_count=n_dup,
                indices=list(df.index[dup_mask][:_MAX_INDICES]),
            ))

    # ── 2. NaN / Inf ──────────────────────────────────────────
    ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    if ohlcv_cols:
        nan_mask = df[ohlcv_cols].isna().any(axis=1)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            report.issues.append(QualityIssue(
                kind="nan",
                severity="error",
                message=f"{n_nan} rows contain NaN values in OHLCV columns",
                row_count=n_nan,
                indices=list(df.index[nan_mask][:_MAX_INDICES]),
            ))

        inf_mask = df[ohlcv_cols].isin([np.inf, -np.inf]).any(axis=1)
        n_inf = int(inf_mask.sum())
        if n_inf > 0:
            report.issues.append(QualityIssue(
                kind="inf",
                severity="error",
                message=f"{n_inf} rows contain Inf values in OHLCV columns",
                row_count=n_inf,
                indices=list(df.index[inf_mask][:_MAX_INDICES]),
            ))

    # ── 3. Zero-volume bars ───────────────────────────────────
    if "volume" in df.columns:
        zero_vol = df["volume"] <= 0
        n_zero = int(zero_vol.sum())
        if n_zero > 0:
            report.issues.append(QualityIssue(
                kind="zero_volume",
                severity="warning",
                message=f"{n_zero} bars have zero or negative volume",
                row_count=n_zero,
                indices=list(df.index[zero_vol][:_MAX_INDICES]),
            ))

    # ── 4. OHLC constraint violations ─────────────────────────
    #    high ≥ max(open, close) and low ≤ min(open, close)
    if all(c in df.columns for c in ("open", "high", "low", "close")):
        max_oc = np.maximum(df["open"].values, df["close"].values)
        min_oc = np.minimum(df["open"].values, df["close"].values)
        high_violation = df["high"].values < max_oc - 1e-12
        low_violation = df["low"].values > min_oc + 1e-12
        hl_violation = df["high"].values < df["low"].values - 1e-12
        combined = high_violation | low_violation | hl_violation
        n_viol = int(combined.sum())
        if n_viol > 0:
            report.issues.append(QualityIssue(
                kind="ohlc_violation",
                severity="error",
                message=f"{n_viol} bars violate OHLC constraints (high<max(O,C) or low>min(O,C) or high<low)",
                row_count=n_viol,
                indices=list(df.index[combined][:_MAX_INDICES]),
            ))

    # ── 5. Gaps (missing bars) ────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        if expected_freq is not None:
            freq = pd.tseries.frequencies.to_offset(expected_freq)
        else:
            freq = pd.infer_freq(df.index)
            if freq is not None:
                freq = pd.tseries.frequencies.to_offset(freq)

        expected_delta = _offset_to_timedelta(freq)
        if expected_delta is not None and expected_delta > pd.Timedelta(0):
            diffs = df.index[1:] - df.index[:-1]
            gap_mask = diffs > expected_delta * 1.5
            n_gaps = int(gap_mask.sum())
            if n_gaps > 0:
                gap_starts = df.index[:-1][gap_mask]
                report.issues.append(QualityIssue(
                    kind="gap",
                    severity="warning",
                    message=f"{n_gaps} time gaps detected (expected interval: {expected_delta})",
                    row_count=n_gaps,
                    indices=list(gap_starts[:_MAX_INDICES]),
                ))
        elif freq is not None and expected_delta is None:
            # Variable-length offset (Month/Year/Quarter/BD): report
            # that gap detection is not applicable, but do not flag
            # an error.
            report.issues.append(QualityIssue(
                kind="gap_check_skipped",
                severity="warning",
                message=(
                    f"Gap detection skipped: frequency {freq.name!r} has "
                    f"variable length (month/year/quarter/business-day)."
                ),
                row_count=0,
            ))

    # ── 6. Price anomalies ────────────────────────────────────
    if "close" in df.columns and len(df) > 1:
        close = df["close"].values
        prev = np.roll(close, 1)
        prev[0] = close[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_change = np.abs(close / (prev + 1e-15) - 1.0)
        anomaly_mask = pct_change > price_change_threshold
        anomaly_mask[0] = False  # first bar has no reference
        n_anomaly = int(anomaly_mask.sum())
        if n_anomaly > 0:
            report.issues.append(QualityIssue(
                kind="anomaly",
                severity="warning",
                message=f"{n_anomaly} bars have price changes exceeding {price_change_threshold:.0%}",
                row_count=n_anomaly,
                indices=list(df.index[anomaly_mask][:_MAX_INDICES]),
            ))

    # ── 7. Timezone consistency ───────────────────────────────
    # Mixing tz-naive and tz-aware data (or non-UTC timestamps) is a
    # classic source of subtle, hard-to-debug trading bugs (DST jumps,
    # off-by-N-hours in fills).  Emitting a warning is cheap insurance.
    if isinstance(df.index, pd.DatetimeIndex) and expected_tz is not None:
        actual_tz = df.index.tz
        if actual_tz is None:
            report.issues.append(QualityIssue(
                kind="tz_naive",
                severity="warning",
                message=(
                    f"DatetimeIndex is timezone-naive. Consider "
                    f"tz_localize to {expected_tz!r} to avoid DST and "
                    f"offset bugs."
                ),
                row_count=0,
            ))
        else:
            # Detect "not in expected_tz" by attempting to convert the
            # actual index to ``expected_tz`` and comparing.  This
            # catches every UTC representation (UTC, Etc/UTC, +00:00,
            # datetime.timezone.utc) uniformly and also flags
            # non-UTC data when the user asked for a non-UTC zone.
            try:
                converted = df.index.tz_convert(expected_tz)
            except (TypeError, ValueError, AttributeError):
                converted = None
            if converted is not None and not converted.equals(df.index):
                report.issues.append(QualityIssue(
                    kind="tz_non_utc",
                    severity="warning",
                    message=(
                        f"DatetimeIndex timezone is {actual_tz!s}, not "
                        f"{expected_tz!r}. Consider tz_convert to "
                        f"{expected_tz!r} for consistency."
                    ),
                    row_count=0,
                ))

    # ── Compute overall status ────────────────────────────────
    report.clean = len(report.errors) == 0
    return report


def repair_ohlcv(
    df: pd.DataFrame,
    report: Optional[QualityReport] = None,
    *,
    drop_duplicates: bool = True,
    fill_nan: bool = True,
    clamp_ohlc: bool = True,
    fill_zero_volume: bool = True,
) -> Tuple[pd.DataFrame, QualityReport]:
    """Attempt to automatically repair common OHLCV quality issues.

    Args:
        df: The input DataFrame.
        report: An existing QualityReport (recomputed if None).
        drop_duplicates: Remove duplicate timestamps (keep last).
        fill_nan: Forward-fill NaN values in OHLCV columns.
        clamp_ohlc: Fix OHLC constraint violations.
        fill_zero_volume: Replace zero/negative volume with 1.0.

    Returns:
        ``(repaired_df, new_report)`` — the repaired DataFrame and a fresh
        quality report on the repaired data.
    """
    if report is None:
        report = audit_ohlcv(df)

    df = df.copy()

    issue_kinds = {iss.kind for iss in report.issues}

    # 1. Drop duplicates
    if drop_duplicates and "duplicate" in issue_kinds:
        df = df[~df.index.duplicated(keep="last")]

    # 2. Sort by index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    # 3. Fill NaN (forward fill, then backward fill for leading NaNs)
    if fill_nan and ("nan" in issue_kinds or "inf" in issue_kinds):
        ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
        # Replace inf with NaN first
        df[ohlcv_cols] = df[ohlcv_cols].replace([np.inf, -np.inf], np.nan)
        df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()

    # 4. Clamp OHLC constraints
    if clamp_ohlc and "ohlc_violation" in issue_kinds:
        if all(c in df.columns for c in ("open", "high", "low", "close")):
            max_oc = np.maximum(df["open"].values, df["close"].values)
            min_oc = np.minimum(df["open"].values, df["close"].values)
            df["high"] = np.maximum(df["high"].values, max_oc)
            df["low"] = np.minimum(df["low"].values, min_oc)
            df["low"] = np.minimum(df["low"].values, df["high"].values)

    # 5. Fill zero volume
    if fill_zero_volume and "zero_volume" in issue_kinds:
        if "volume" in df.columns:
            df.loc[df["volume"] <= 0, "volume"] = 1.0

    # Re-audit
    new_report = audit_ohlcv(df)
    return df, new_report
