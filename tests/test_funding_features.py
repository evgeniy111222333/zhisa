import pandas as pd
import numpy as np
import pytest

from zhisa.scripts._real_data import join_futures_context
from zhisa.features.ohlcv import compute_ohlcv_features

def test_join_futures_context_ffill():
    # 1. Create a dummy OHLCV 5-minute dataframe
    index = pd.date_range("2026-01-01 00:00:00", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.ones(10),
            "high": np.ones(10),
            "low": np.ones(10),
            "close": np.ones(10),
            "volume": np.ones(10),
        },
        index=index,
    )
    
    # 2. Create a dummy funding rate context at a single timestamp
    context_index = pd.to_datetime(["2026-01-01 00:00:00"], utc=True)
    context = pd.DataFrame({"funding_rate": [0.015]}, index=context_index)
    
    # 3. Join
    joined = join_futures_context(df, context, symbol="BTC_USDT", timeframe="5m")
    
    # 4. Verify forward fill
    assert "funding_rate" in joined.columns
    # It should forward fill across all 10 periods (which is 50 minutes, well within the limit of 120)
    assert joined["funding_rate"].isna().sum() == 0
    assert (joined["funding_rate"] == 0.015).all()


def test_compute_ohlcv_features_funding_zscore():
    # Create 2020 periods of dummy data (more than 2016 for full rolling window)
    index = pd.date_range("2026-01-01 00:00:00", periods=2020, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.ones(2020),
            "high": np.ones(2020) * 1.1,
            "low": np.ones(2020) * 0.9,
            "close": np.ones(2020),
            "volume": np.ones(2020),
        },
        index=index,
    )
    
    # Add funding_rate directly to df (as it would be after join_futures_context)
    # Let's make the funding rate slowly drift, and then spike at the very end
    funding = np.linspace(0.001, 0.002, 2020)
    funding[-1] = 0.050 # Massive spike
    df["funding_rate"] = funding
    
    # Compute features
    features = compute_ohlcv_features(df)
    
    # Check that basic funding_rate is exposed
    assert "ctx_funding_rate" in features.columns
    assert features["ctx_funding_rate"].iloc[-1] == 0.050
    
    # Check that zscore is calculated
    assert "ctx_funding_zscore_7d" in features.columns
    
    # The last value should have a huge z-score due to the spike
    zscore_last = features["ctx_funding_zscore_7d"].iloc[-1]
    assert zscore_last > 10.0 # It should be significantly positive
