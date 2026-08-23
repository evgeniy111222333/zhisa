"""Tests for futures-context feature extraction (funding/OI/ratios).

Verifies that the same context columns produce identical ``ctx_*`` features
whether they arrive unprefixed (ad-hoc live join) or ``ctx_``-prefixed
(prepared-data merger), that no ``ctx_ctx_`` double prefix leaks, and that the
"7d" z-score window is derived from the bar frequency (5m stays 2016).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.features.ohlcv import _seven_day_window, compute_ohlcv_features


def _frame(cols: dict[str, pd.Series], n: int = 600, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0 + np.linspace(0, 10, n) + np.random.default_rng(0).normal(0, 0.2, n)
    df["high"] = df["open"] * 1.01
    df["low"] = df["open"] * 0.99
    df["close"] = df["open"] * 1.005
    df["volume"] = np.random.default_rng(1).uniform(1000, 5000, n)
    for k, v in cols.items():
        df[k] = v
    return df


def _context_cols(n: int):
    rng = np.random.default_rng(2)
    return {
        "funding_rate": np.sin(np.arange(n) / 12.0) * 1e-4,
        "open_interest": 50000 + 3000 * np.sin(np.arange(n) / 60.0),
        "top_trader_long_short_ratio": 1.0 + 0.2 * np.sin(np.arange(n) / 8.0),
        "kline_taker_buy_volume": np.abs(rng.normal(500, 100, n)),
        "kline_taker_sell_volume": np.abs(rng.normal(450, 100, n)),
    }


def test_unprefixed_and_prefixed_context_are_equivalent():
    n = 600
    raw = _context_cols(n)
    df_plain = _frame({k: v for k, v in raw.items()}, n=n)
    df_ctx = _frame({f"ctx_{k}": v for k, v in raw.items()}, n=n)

    feats_plain = compute_ohlcv_features(df_plain)
    feats_ctx = compute_ohlcv_features(df_ctx)

    # the critical computed indicators must exist in BOTH paths
    for col in ("ctx_funding_zscore_7d", "ctx_oi_zscore_7d", "ctx_ls_zscore_7d",
                "ctx_kline_taker_imbalance", "ctx_open_interest_log1p",
                "ctx_funding_rate", "ctx_available_frac"):
        assert col in feats_plain.columns, f"missing plain {col}"
        assert col in feats_ctx.columns, f"missing prefixed {col}"
        # drop NaNs for comparison (identical inputs -> identical outputs)
        a = feats_plain[col]
        b = feats_ctx[col]
        m = a.notna() & b.notna()
        assert np.allclose(a[m], b[m], equal_nan=True), f"mismatch {col}"

    # no double ctx_ctx_ prefix anywhere
    assert not any(c.startswith("ctx_ctx_") for c in feats_ctx.columns)


def test_zscore_is_finite_after_warmup():
    n = 600
    raw = _context_cols(n)
    feats = compute_ohlcv_features(_frame({"ctx_" + k: v for k, v in raw.items()}, n=n))
    z = feats["ctx_funding_zscore_7d"].dropna()
    assert len(z) > 0
    assert np.isfinite(z).all()


def test_no_context_means_no_ctx_columns():
    feats = compute_ohlcv_features(_frame({}, n=300))
    assert not any(c.startswith("ctx_") for c in feats.columns)


def test_seven_day_window_depends_on_frequency():
    base = pd.date_range("2024-01-01", periods=200, freq="1h")
    assert _seven_day_window(base) == 168          # 7 * 24
    m15 = pd.date_range("2024-01-01", periods=200, freq="15min")
    assert _seven_day_window(m15) == 672           # 7 * 24 * 4
    m5 = pd.date_range("2024-01-01", periods=200, freq="5min")
    assert _seven_day_window(m5) == 2016           # unchanged 5m behaviour
    # degenerate: <2 samples / irregular -> fallback 2016
    assert _seven_day_window(base[:1]) == 2016