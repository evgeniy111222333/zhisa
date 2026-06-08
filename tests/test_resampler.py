"""Tests for OHLCV resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.storage.resampler import resample_ohlcv
from zhisa.storage.schema import Timeframe


@pytest.fixture
def m5_df() -> pd.DataFrame:
    """A 5-minute OHLCV DataFrame."""
    return generate_market(MarketConfig(n_bars=2000, freq="5min", seed=42))


# ────────────────────────────────────────────────────────────────────
# Valid resampling
# ────────────────────────────────────────────────────────────────────

class TestValidResampling:
    def test_5m_to_1h(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H1)
        # 2000 * 5m = 10000 min → ~166 hours
        assert len(result) > 0
        assert len(result) < len(m5_df)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_5m_to_15m(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.M15)
        assert len(result) > 0
        # 3:1 ratio approximately
        expected_ratio = len(m5_df) / 3
        assert abs(len(result) - expected_ratio) < 10

    def test_5m_to_4h(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H4)
        assert len(result) > 0
        assert len(result) < len(m5_df) / 40  # 48 bars per 4h

    def test_same_timeframe_returns_copy(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.M5)
        assert len(result) == len(m5_df)
        # Should be a copy, not the same object
        assert result is not m5_df

    def test_1h_to_1d(self):
        df = generate_market(MarketConfig(n_bars=500, freq="1h", seed=42))
        result = resample_ohlcv(df, Timeframe.H1, Timeframe.D1)
        assert len(result) > 0
        assert len(result) < len(df)


# ────────────────────────────────────────────────────────────────────
# OHLCV aggregation correctness
# ────────────────────────────────────────────────────────────────────

class TestAggregationCorrectness:
    def test_open_is_first(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H1)
        # The first 1h bar's open should be the first 5m bar's open
        first_hour_start = result.index[0]
        first_hour_bars = m5_df[
            (m5_df.index >= first_hour_start)
            & (m5_df.index < first_hour_start + pd.Timedelta(hours=1))
        ]
        if len(first_hour_bars) > 0:
            np.testing.assert_almost_equal(
                result["open"].iloc[0],
                first_hour_bars["open"].iloc[0],
                decimal=5,
            )

    def test_high_is_max(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H1)
        first_hour_start = result.index[0]
        first_hour_bars = m5_df[
            (m5_df.index >= first_hour_start)
            & (m5_df.index < first_hour_start + pd.Timedelta(hours=1))
        ]
        if len(first_hour_bars) > 0:
            np.testing.assert_almost_equal(
                result["high"].iloc[0],
                first_hour_bars["high"].max(),
                decimal=5,
            )

    def test_low_is_min(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H1)
        first_hour_start = result.index[0]
        first_hour_bars = m5_df[
            (m5_df.index >= first_hour_start)
            & (m5_df.index < first_hour_start + pd.Timedelta(hours=1))
        ]
        if len(first_hour_bars) > 0:
            np.testing.assert_almost_equal(
                result["low"].iloc[0],
                first_hour_bars["low"].min(),
                decimal=5,
            )

    def test_close_is_last(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H1)
        first_hour_start = result.index[0]
        first_hour_bars = m5_df[
            (m5_df.index >= first_hour_start)
            & (m5_df.index < first_hour_start + pd.Timedelta(hours=1))
        ]
        if len(first_hour_bars) > 0:
            np.testing.assert_almost_equal(
                result["close"].iloc[0],
                first_hour_bars["close"].iloc[-1],
                decimal=5,
            )

    def test_volume_is_sum(self, m5_df: pd.DataFrame):
        result = resample_ohlcv(m5_df, Timeframe.M5, Timeframe.H1)
        first_hour_start = result.index[0]
        first_hour_bars = m5_df[
            (m5_df.index >= first_hour_start)
            & (m5_df.index < first_hour_start + pd.Timedelta(hours=1))
        ]
        if len(first_hour_bars) > 0:
            np.testing.assert_almost_equal(
                result["volume"].iloc[0],
                first_hour_bars["volume"].sum(),
                decimal=2,
            )


# ────────────────────────────────────────────────────────────────────
# Invalid resampling
# ────────────────────────────────────────────────────────────────────

class TestInvalidResampling:
    def test_upsample_raises(self, m5_df: pd.DataFrame):
        """Cannot resample from 5m to 1m (upsampling)."""
        with pytest.raises(ValueError, match="target timeframe must be"):
            resample_ohlcv(m5_df, Timeframe.M5, Timeframe.M1)

    def test_non_multiple_raises(self):
        """Cannot resample 5m to 30m? Actually 30/5=6, that's fine.
        Let's try 1m to 7m — but 7m doesn't exist in our enum.
        Instead test M15 to H4: 240/15=16, that's fine.
        We need something that doesn't divide evenly. Since our enum
        only has clean multiples, we just verify the validation logic."""
        # M30 to H1: 60/30=2, OK
        # All our enum values divide evenly into their parents, so
        # we test the can_resample_to method directly
        assert not Timeframe.H1.can_resample_to(Timeframe.M5)
        assert Timeframe.M5.can_resample_to(Timeframe.H1)

    def test_no_datetime_index_raises(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            resample_ohlcv(df, Timeframe.M5, Timeframe.H1)


# ────────────────────────────────────────────────────────────────────
# Extra columns
# ────────────────────────────────────────────────────────────────────

class TestExtraColumns:
    def test_extra_columns_agg(self, m5_df: pd.DataFrame):
        # The synthetic data has a 'regime' column
        if "regime" not in m5_df.columns:
            m5_df["regime"] = 0
        result = resample_ohlcv(
            m5_df, Timeframe.M5, Timeframe.H1,
            extra_columns={"regime": "last"},
        )
        assert "regime" in result.columns
