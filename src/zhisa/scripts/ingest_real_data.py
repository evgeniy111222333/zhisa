"""Ingest public exchange OHLCV into the local TimeSeriesDB.

This command is data-only. It uses public candle endpoints through CCXT,
does not require API keys, and does not contain any order-placement path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from zhisa.data.crypto_loader import CCXTCryptoLoader
from zhisa.scripts._real_data import (
    frame_summary,
    normalize_ohlcv_frame,
    parse_utc_timestamp,
    series_key_from_args,
    timestamp_to_ms,
)
from zhisa.storage.tsdb import TimeSeriesDB


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest real OHLCV data into local TSDB.")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--since", type=str, default=None, help="UTC timestamp, e.g. 2024-01-01")
    parser.add_argument("--max-bars", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--db-root", type=str, default="data/tsdb")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--until", type=str, default=None, help="UTC timestamp, e.g. 2026-05-17")
    args = parser.parse_args(argv)

    loader = CCXTCryptoLoader(exchange_id=args.exchange)
    df = loader.fetch_ohlcv(
        args.symbol,
        timeframe=args.timeframe,
        since_ms=timestamp_to_ms(args.since),
        limit=int(args.limit),
        max_bars=int(args.max_bars) if args.max_bars else None,
    )
    df = normalize_ohlcv_frame(df)
    
    if args.until:
        until_ts = parse_utc_timestamp(args.until)
        if until_ts is not None:
            df = df[df.index <= until_ts]
            
    if len(df) == 0:
        raise RuntimeError("Exchange returned no OHLCV rows")

    key = series_key_from_args(args)
    db = TimeSeriesDB(args.db_root)
    meta = db.ingest(key, df)
    summary = frame_summary(db.read(key))
    payload = {
        "exchange": args.exchange,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "db_root": str(Path(args.db_root)),
        "stored_rows": int(meta.row_count),
        "fetched_rows": int(len(df)),
        "start": str(meta.start),
        "end": str(meta.end),
        "quality": summary,
        "mode": "public_ohlcv_ingest_no_orders",
    }

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        export = df.copy()
        export.index.name = "timestamp"
        export.to_csv(out_csv)
        payload["out_csv"] = str(out_csv)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

