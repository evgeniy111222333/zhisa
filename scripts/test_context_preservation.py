import pandas as pd
from zhisa.scripts._real_data import normalize_ohlcv_frame
from zhisa.features.ohlcv import compute_ohlcv_features
from zhisa.env.trading_env import TradingEnv, EnvConfig

def main():
    # 1. Create a dummy dataframe
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=10, freq='5min', tz='UTC'),
        'open': [100] * 10,
        'high': [105] * 10,
        'low': [95] * 10,
        'close': [102] * 10,
        'volume': [1000] * 10,
        'funding_rate': [0.0001] * 10,
        'open_interest': [5000000] * 10,
        'top_trader_long_short_ratio': [1.5] * 10,
        'taker_buy_sell_ratio': [1.1] * 10,
    })
    
    # 2. Assign attrs like _real_data.py would
    df.attrs["futures_context"] = {
        "columns": ["funding_rate", "open_interest", "top_trader_long_short_ratio", "taker_buy_sell_ratio"]
    }
    
    # 3. Normalize the dataframe
    normalized_df = normalize_ohlcv_frame(df, keep_extra=True)
    
    # 4. Ensure attrs are preserved
    assert "futures_context" in normalized_df.attrs, "normalize_ohlcv_frame dropped attrs!"
    
    # 5. Ensure compute_ohlcv_features sees it
    features = compute_ohlcv_features(normalized_df)
    assert any("ctx_" in c for c in features.columns), "compute_ohlcv_features did not generate context features!"
    
    # 6. Ensure TradingEnv picks it up
    env = TradingEnv(normalized_df, cfg=EnvConfig(window=4))
    print(f"TradingEnv obs_numeric_dim: {env.obs_numeric_dim}")
    assert env.obs_numeric_dim > 32, f"TradingEnv only picked up {env.obs_numeric_dim} features, expected >32!"
    
    print("ALL TESTS PASSED: Context metadata is correctly preserved!")

if __name__ == "__main__":
    main()
