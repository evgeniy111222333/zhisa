import pandas as pd
from pathlib import Path

def main():
    print("Loading funding rates CSV...")
    df = pd.read_csv("data/funding_rates_BTCUSDT.csv")
    
    # Ensure timestamp index
    df["timestamp"] = pd.to_datetime(df["fundingTime"], utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()
    
    # Keep only the relevant columns and rename them to match the context expected schema
    # The context loader looks for numeric columns and skips OHLCV columns.
    # We will rename fundingRate to funding_rate for clarity.
    context_df = pd.DataFrame(index=df.index)
    context_df["funding_rate"] = df["fundingRate"]
    
    # Optional: we could include markPrice but it's not strictly a feature right now.
    
    out_dir = Path("data/futures_context/binance_usdm/BTC_USDT/5m")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "context.parquet"
    context_df.to_parquet(out_path)
    print(f"Successfully saved {len(context_df)} context records to {out_path}")

if __name__ == "__main__":
    main()
