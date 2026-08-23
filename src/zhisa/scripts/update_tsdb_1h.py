"""Extend the local TSDB 1h OHLCV series to the current date.

The S1 dataset ended 2026-06-20 while funding/OI context was downloaded to
today. This tool appends Binance USD-M ``1h`` futures klines (via a clean
``TimeSeriesDB.ingest`` merge) so the next prepared root ends at "now" and the
latest ~2 months become usable — and available for a true roll-forward OOS gate.

    python -m zhisa.scripts.update_tsdb_1h --symbols BTC/USDT,ETH/USDT --to 2026-08-23
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch  # noqa: F401  (keeps dependency parity with the rest of scripts)

from zhisa.scripts.download_full_context import _fetch_api_json
from zhisa.scripts._real_data import futures_context_symbol_slug
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.storage.schema import Timeframe, SeriesKey
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ROOT = Path("data/tsdb")
SYMBOLS_12 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
]


def fetch_1h_klines(symbol_ccxt: str, start_ts: int, end_ts: int, limit: int = 1500) -> pd.DataFrame:
    sym = futures_context_symbol_slug(symbol_ccxt)
    rows: list[dict] = []
    cursor = start_ts
    while cursor <= end_ts:
        data = _fetch_api_json("/fapi/v1/klines", {
            "symbol": sym, "interval": "1h",
            "startTime": cursor, "endTime": end_ts, "limit": limit,
        })
        if not data:
            break
        for k in data:
            rows.append({
                "timestamp": pd.Timestamp(int(k[0]), unit="ms", tz="UTC"),
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": float(k[5]),
            })
        last_ts = int(data[-1][0])
        if last_ts < cursor or len(data) < limit:
            break
        cursor = last_ts + 1
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(SYMBOLS_12))
    ap.add_argument("--to", default=None, help="inclusive end (ISO/date); default now")
    ap.add_argument("--tsdb-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    to = pd.Timestamp(args.to, tz="UTC") if args.to else pd.Timestamp.now(timezone.utc)
    end_ts = int(to.timestamp() * 1000)
    db = TimeSeriesDB(Path(args.tsdb_root))
    rc = 0
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        key = SeriesKey(sym, Timeframe.H1)
        if not db.has_series(key):
            logger.error("no existing 1h series for %s in %s", sym, args.tsdb_root)
            rc = max(rc, 1)
            continue
        meta = db.get_meta(key)
        end_t = pd.Timestamp(meta.end)
        end_t = end_t.tz_localize("UTC") if end_t.tzinfo is None else end_t.tz_convert("UTC")
        start_ts = int(end_t.timestamp() * 1000) + 3600_000
        if start_ts > end_ts:
            logger.info("%s: already fresh (end=%s)", sym, meta.end)
            continue
        df = fetch_1h_klines(sym, start_ts, end_ts)
        print(f"{sym:10s} existing={meta.row_count} end={meta.end}  fetched={len(df)} new bars "
              f"({df.index.min() if len(df) else 'n/a'} -> {df.index.max() if len(df) else 'n/a'})")
        if args.dry_run or len(df) == 0:
            continue
        merged = db.ingest(key, df)
        logger.info("%s -> rows=%d end=%s checksum=%s", sym, merged.row_count,
                    merged.end, str(merged.checksum)[:12])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())