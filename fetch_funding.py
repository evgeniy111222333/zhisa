import requests
import pandas as pd
import time
from datetime import datetime

def fetch_binance_funding(symbol="BTCUSDT"):
    print(f"Fetching funding rates for {symbol}...")
    
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    
    all_data = []
    # BTCUSDT futures started roughly in Sep 2019
    start_time = 1567300000000
    
    while True:
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "limit": 1000
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            break
            
        data = response.json()
        
        if not data:
            print("No more data returned.")
            break
            
        all_data.extend(data)
        
        # Get the timestamp of the last item to paginate
        last_time = data[-1]['fundingTime']
        
        # If the API returned less than limit, we've reached the end
        if len(data) < 1000:
            break
            
        # Update start_time to the last_time + 1ms to avoid duplicates
        start_time = last_time + 1
        
        print(f"Fetched {len(data)} records, up to {datetime.fromtimestamp(last_time/1000)}")
        time.sleep(0.5) # Be polite to API
        
    if not all_data:
        print("Dataset is empty!")
        return
        
    df = pd.DataFrame(all_data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['markPrice'] = pd.to_numeric(df['markPrice'], errors='coerce')
    
    # Save to CSV
    save_path = "data/funding_rates_BTCUSDT.csv"
    df.to_csv(save_path, index=False)
    
    print("\n--- DATASET VERIFICATION ---")
    print(f"Total records downloaded: {len(df)}")
    print(f"Date range: from {df['fundingTime'].min()} to {df['fundingTime'].max()}")
    print("\nBasic Statistics of Funding Rate:")
    print(df['fundingRate'].describe())
    
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))
    print(f"\nDataset saved successfully to: {save_path}")

if __name__ == "__main__":
    fetch_binance_funding("BTCUSDT")
