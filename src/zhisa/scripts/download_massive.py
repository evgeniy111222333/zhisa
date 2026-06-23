"""Massive data downloader for S1 pretraining."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from zhisa.data.crypto_loader import CCXTCryptoLoader
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.storage.schema import SeriesKey, Timeframe
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

def parse_date(date_str: str) -> int:
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def last_closed_bar_open_ms(now_ms: int, interval_ms: int) -> int:
    """Return the open timestamp of the most recent fully closed bar."""
    return (int(now_ms) // int(interval_ms) - 1) * int(interval_ms)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--since", type=str, default="2019-01-01")
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Inclusive bar-open cutoff; defaults to the last fully closed bar",
    )
    parser.add_argument("--db-root", type=str, default="data/tsdb")
    parser.add_argument("--chunk-bars", type=int, default=50_000)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    Path(args.db_root).mkdir(parents=True, exist_ok=True)
    db = TimeSeriesDB(args.db_root)
    loader = CCXTCryptoLoader(exchange_id=args.exchange)
    
    start_ms = parse_date(args.since)
    timeframe = Timeframe.from_str(args.timeframe)
    interval_ms = timeframe.minutes * 60_000
    now_ms = int(time.time() * 1000)
    end_ms = (
        parse_date(args.until)
        if args.until
        else last_closed_bar_open_ms(now_ms, interval_ms)
    )
    chunk_bars = max(1, int(args.chunk_bars))
    
    for symbol in symbols:
        logger.info(f"Starting massive download for {symbol}")
        key = SeriesKey(symbol, timeframe)
        
        # Check existing data to resume
        current_since = start_ms
        try:
            existing_df = db.read(key)
            if not existing_df.empty:
                last_ts = existing_df.index[-1]
                current_since = int(last_ts.timestamp() * 1000) + interval_ms
                logger.info(f"Resuming {symbol} from {last_ts}")
        except Exception:
            # File doesn't exist yet, start from the beginning
            pass
            
        retries = 0
        while current_since <= end_ms:
            logger.info(f"Fetching chunk for {symbol} starting at {datetime.fromtimestamp(current_since/1000, tz=timezone.utc)}")
            try:
                remaining = (end_ms - current_since) // interval_ms + 1
                df = loader.fetch_ohlcv(
                    symbol,
                    timeframe=args.timeframe,
                    since_ms=current_since,
                    limit=1000,
                    max_bars=min(chunk_bars, int(remaining)),
                )
                cutoff = pd.to_datetime(end_ms, unit="ms", utc=True)
                df = df[df.index <= cutoff]
                if len(df) == 0:
                    logger.info(f"No more data for {symbol} at this time.")
                    break

                db.ingest(key, df)
                current_since = int(df.index[-1].timestamp() * 1000) + interval_ms
                retries = 0
            except Exception as e:
                retries += 1
                logger.error(f"Error fetching {symbol}: {e}")
                if retries >= int(args.max_retries):
                    raise RuntimeError(
                        f"Failed {symbol} after {retries} retries at {current_since}"
                    ) from e
                time.sleep(min(60, 5 * retries))
                
    logger.info("Massive download complete!")

if __name__ == "__main__":
    main()
