import os
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_and_extract_metric(date_str: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
                
        if df.empty or "create_time" not in df.columns:
            return pd.DataFrame()
            
        df["timestamp"] = pd.to_datetime(df["create_time"], utc=True)
        df = df.set_index("timestamp")
        
        # Keep only the columns we need
        cols_to_keep = [
            "sum_open_interest", 
            "count_toptrader_long_short_ratio", 
            "sum_taker_long_short_vol_ratio"
        ]
        
        # Rename them to be consistent
        rename_map = {
            "sum_open_interest": "open_interest",
            "count_toptrader_long_short_ratio": "ls_ratio",
            "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio"
        }
        
        df = df[cols_to_keep].rename(columns=rename_map)
        return df
    except Exception as e:
        print(f"Error on {date_str}: {e}")
        return pd.DataFrame()

def main():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 6, 15)
    
    dates = []
    curr = start_date
    while curr <= end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
        
    print(f"Downloading metrics for {len(dates)} days...")
    
    frames = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_and_extract_metric, d): d for d in dates}
        for future in tqdm(as_completed(futures), total=len(dates)):
            df = future.result()
            if not df.empty:
                frames.append(df)
                
    if not frames:
        print("No data downloaded!")
        return
        
    all_metrics = pd.concat(frames).sort_index()
    all_metrics = all_metrics[~all_metrics.index.duplicated(keep="last")]
    print(f"Downloaded {len(all_metrics)} rows of metrics.")
    
    # Now merge with existing context.parquet
    context_path = "data/futures_context/binance_usdm/BTCUSDT/5m/context.parquet"
    if os.path.exists(context_path):
        print(f"Merging with existing {context_path}...")
        existing_context = pd.read_parquet(context_path)
        
        # If columns exist, drop them before merge to overwrite
        for col in all_metrics.columns:
            if col in existing_context.columns:
                existing_context = existing_context.drop(columns=[col])
                
        merged = existing_context.join(all_metrics, how="outer")
    else:
        print(f"{context_path} not found, creating new.")
        merged = all_metrics
        
    merged = merged.sort_index()
    
    # Save back
    merged.to_parquet(context_path)
    print(f"Saved {len(merged)} total rows to {context_path}!")
    print(merged.tail())

if __name__ == "__main__":
    main()
