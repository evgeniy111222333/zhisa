"""Hybrid downloader for Binance Futures Context (Metrics from ZIP + API).

Downloads full historical context by combining Binance Vision S3 Zip Archives
for restricted metrics (like Open Interest) and Binance API for funding/mark price/index price/futures klines.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import logging
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
API_BASE_URL = "https://fapi.binance.com"


def _fetch_api_json(path: str, params: dict, retries: int = 3) -> list:
    import urllib.parse
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{API_BASE_URL}{path}?{query}" if query else f"{API_BASE_URL}{path}"
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zhisa-hybrid-context/1.0"})
            with urllib.request.urlopen(req, timeout=15) as res:
                return json.loads(res.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to fetch {url}: {e}")
                raise
            import time
            time.sleep(1)
    return []


def fetch_api_history(symbol: str, endpoint: str, parser: str, start_ts: int, end_ts: int, prefix: str = "", limit: int = 1500) -> pd.DataFrame:
    logger.info(f"Fetching API {endpoint} for {symbol}...")
    all_data = []
    cursor = start_ts
    
    while cursor <= end_ts:
        # indexPriceKlines uses 'pair' instead of 'symbol'
        sym_key = "pair" if "indexPriceKlines" in endpoint else "symbol"
        params = {sym_key: symbol, "startTime": cursor, "endTime": end_ts, "limit": limit}
        if "Klines" in endpoint or "klines" in endpoint:
            params["interval"] = "15m"
            
        data = _fetch_api_json(endpoint, params)
        if not data:
            break
            
        all_data.extend(data)
        
        # Get last timestamp
        if parser == "kline":
            last_ts = int(data[-1][0])
        else:
            last_ts = int(data[-1]["fundingTime"])
            
        if last_ts < cursor:
            break
            
        cursor = last_ts + 1
        if len(data) < limit:
            break
            
    if not all_data:
        return pd.DataFrame()

    if parser == "kline":
        df = pd.DataFrame(all_data)
        # Binance kline format:
        # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, 6: Close time, 
        # 7: Quote asset vol, 8: No. of trades, 9: Taker buy base asset vol, 10: Taker buy quote asset vol
        df["timestamp"] = pd.to_datetime(df[0].astype(int), unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        
        if prefix == "futures_":
            df = df[[1, 2, 3, 4, 5, 9]]
            df.columns = [f"{prefix}open", f"{prefix}high", f"{prefix}low", f"{prefix}close", f"{prefix}volume", "kline_taker_buy_volume"]
            for col in df.columns:
                df[col] = df[col].astype(float)
            df["kline_taker_sell_volume"] = df[f"{prefix}volume"] - df["kline_taker_buy_volume"]
        else:
            df = df[[1, 2, 3, 4]]
            df.columns = [f"{prefix}open", f"{prefix}high", f"{prefix}low", f"{prefix}close"]
            for col in df.columns:
                df[col] = df[col].astype(float)
            
        df = df[~df.index.duplicated(keep="last")]
        return df
    else:
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
        # Floor funding to 15m
        df["timestamp"] = df["timestamp"].dt.floor("15min")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df.set_index("timestamp", inplace=True)
        df = df[["funding_rate"]]
        df = df[~df.index.duplicated(keep="last")]
        return df


def download_daily_metrics(symbol: str, date: datetime.date) -> pd.DataFrame | None:
    date_str = date.strftime("%Y-%m-%d")
    url = f"{VISION_BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "zhisa/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f)
                    df["timestamp"] = pd.to_datetime(df["create_time"], utc=False).dt.tz_localize("UTC")
                    
                    # Binance metrics often come as 00:00:01 or 14:59:59. Round to nearest 15min.
                    df["timestamp"] = df["timestamp"].dt.round("15min")
                    
                    # Now we can safely drop duplicates that fell into the same 15m bucket
                    df = df.drop_duplicates(subset=["timestamp"], keep="last")
                    
                    # We only care about 15m intervals to match our requested timeframe
                    df = df[df["timestamp"].dt.minute % 15 == 0]
                    df.set_index("timestamp", inplace=True)
                    
                    rename_map = {
                        "sum_open_interest": "open_interest",
                        "sum_open_interest_value": "open_interest_value",
                        "count_toptrader_long_short_ratio": "top_trader_long_short_ratio",
                        "count_long_short_ratio": "global_long_short_ratio",
                        "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
                    }
                    cols_to_keep = [c for c in rename_map.keys() if c in df.columns]
                    df = df[cols_to_keep].rename(columns=rename_map)
                    
                    for col in df.columns:
                        df[col] = df[col].astype(float)
                    return df
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except Exception as e:
        return None


def fetch_all_metrics(symbol: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    days = (end_date - start_date).days
    dates = [start_date + datetime.timedelta(days=i) for i in range(days + 1)]
    
    logger.info(f"Downloading Vision zip metrics for {symbol} ({len(dates)} days)...")
    results = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_daily_metrics, symbol, d): d for d in dates}
        completed = 0
        for f in as_completed(futures):
            df = f.result()
            if df is not None and not df.empty:
                results.append(df)
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  {completed}/{len(dates)} days processed...")
                
    if not results:
        return pd.DataFrame()
        
    final_df = pd.concat(results).sort_index()
    final_df = final_df[~final_df.index.duplicated(keep="last")]
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2026-05-17")
    parser.add_argument("--out-dir", type=str, default="data/tsdb/binance")
    args = parser.parse_args()

    symbol_safe = args.symbol.replace("/", "_")
    symbol_binance = args.symbol.replace("/", "").replace("_", "")
    
    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date()
    
    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000) + 86400000 - 1
    
    # 1. Vision Metrics
    df_metrics = fetch_all_metrics(symbol_binance, start_date, end_date)
    
    # 2. API Klines (Tier 1/2/3)
    df_futures = fetch_api_history(symbol_binance, "/fapi/v1/klines", "kline", start_ms, end_ms, prefix="futures_")
    df_mark = fetch_api_history(symbol_binance, "/fapi/v1/markPriceKlines", "kline", start_ms, end_ms, prefix="mark_")
    df_index = fetch_api_history(symbol_binance, "/fapi/v1/indexPriceKlines", "kline", start_ms, end_ms, prefix="index_")
    df_prem = fetch_api_history(symbol_binance, "/fapi/v1/premiumIndexKlines", "kline", start_ms, end_ms, prefix="premium_")
    df_fund = fetch_api_history(symbol_binance, "/fapi/v1/fundingRate", "funding", start_ms, end_ms, limit=1000)
    
    logger.info("Merging datasets...")
    # Base index on futures_klines (which is the most robust full history)
    if not df_futures.empty:
        context = df_futures
    elif not df_metrics.empty:
        context = df_metrics
    else:
        logger.error(f"No data found at all for {symbol_binance}")
        return
        
    if not df_metrics.empty and df_metrics is not context:
        context = context.join(df_metrics, how="outer")
    if not df_mark.empty:
        context = context.join(df_mark, how="outer")
        if "mark_close" in context.columns:
            context["mark_price"] = context["mark_close"] # Explicit mark_price column
    if not df_index.empty:
        context = context.join(df_index, how="outer")
    if not df_prem.empty:
        context = context.join(df_prem, how="outer")
        if "premium_close" in context.columns:
            context["premium_index"] = context["premium_close"]
    if not df_fund.empty:
        context = context.join(df_fund, how="outer")
        
    if "funding_rate" in context.columns:
        context["funding_rate"] = context["funding_rate"].ffill().fillna(0.0)

    context = context[(context.index >= pd.Timestamp(args.start, tz="UTC")) & 
                      (context.index <= pd.Timestamp(args.end + " 23:59:59", tz="UTC"))]
                      
    out_path = Path(args.out_dir) / symbol_safe / "15m" / "futures_context.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    context.to_parquet(out_path)
    logger.info(f"Saved {len(context)} context rows to {out_path}")


if __name__ == "__main__":
    main()
