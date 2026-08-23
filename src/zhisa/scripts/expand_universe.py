"""Expand the S1 universe: add NEW Binance USD-M perpetuals (1h OHLCV + context).

For each symbol this:
  1. ingests the FULL 1h futures-klines history into the local TSDB
     (new series dir created; existing series merged/deduped);
  2. runs the futures-context downloader (Vision OI/metrics + funding,
     mark/index/premium) into the canonical ``data/futures_context`` root so
     the next prepared dataset carries funding/OI + basis for the new symbols.

Shorter listing history is fine — the prepare pipeline aligns coverage at the
union start and the cross-asset index tolerates missing early bars.

    python -m zhisa.scripts.expand_universe \
        --symbols NEAR/USDT,ARB/USDT,OP/USDT --to 2026-08-23 --with-context --context-only-ARB
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch  # noqa: F401

from zhisa.scripts._real_data import futures_context_symbol_slug
from zhisa.scripts.download_full_context import _fetch_api_json
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.storage.schema import Timeframe, SeriesKey
from zhisa.scripts.ensure_futures_context import canonical_path
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

TSDB_ROOT = Path("data/tsdb")
CTX_ROOT = Path("data/futures_context/binance_usdm")


def fetch_1h_full(sym_ccxt: str, end_ts: int, start_ms: int = 1577836800000,
                  limit: int = 1500) -> pd.DataFrame:
    sym = futures_context_symbol_slug(sym_ccxt)
    rows: list[dict] = []
    cursor = start_ms
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
    df = pd.DataFrame(rows).set_index("timestamp")
    return df[~df.index.duplicated(keep="last")].sort_index()[["open", "high", "low", "close", "volume"]]


def ingest_symbol(sym_ccxt: str, db: TimeSeriesDB, end_ts: int) -> dict:
    df = fetch_1h_full(sym_ccxt, end_ts)
    meta = None
    if len(df):
        meta = db.ingest(SeriesKey(sym_ccxt, Timeframe.H1), df)
    return {"symbol": sym_ccxt, "rows": int(meta.row_count) if meta else 0,
            "start": str(meta.start) if meta else None,
            "end": str(meta.end) if meta else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", required=True, help="comma list, e.g. NEAR/USDT,ARB/USDT")
    ap.add_argument("--to", default=None)
    ap.add_argument("--tsdb-root", default=str(TSDB_ROOT))
    ap.add_argument("--with-context", action="store_true",
                    help="also download funding/OI context for each symbol")
    ap.add_argument("--context-only", action="store_true",
                    help="skip OHLCV; only download context for the listed symbols")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    to = pd.Timestamp(args.to, tz="UTC") if args.to else pd.Timestamp.now(tz="utc")
    end_ts = int(to.timestamp() * 1000)
    db = TimeSeriesDB(Path(args.tsdb_root))

    if not args.context_only:
        print(f"ingesting 1h OHLCV for {len(symbols)} symbols ...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ingest_symbol, s, db, end_ts): s for s in symbols}
            for f in as_completed(futs):
                print(" ", f.result())

    if args.with_context:
        ctx_need = [s for s in symbols if not canonical_path(CTX_ROOT, s).exists()]
        if ctx_need:
            print(f"downloading context for {len(ctx_need)} symbols: {ctx_need}")
            cmd = [sys.executable, "-m", "zhisa.scripts.ensure_futures_context",
                   "--symbols", ",".join(ctx_need), "--end", to.strftime("%Y-%m-%d")]
            print("  ", " ".join(cmd))
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                print("context download returned rc=%d" % res.returncode)
                return res.returncode
        else:
            print("all requested symbols already have context")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())