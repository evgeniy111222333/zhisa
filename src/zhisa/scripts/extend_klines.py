"""Extend the local data with multi-timeframe klines:

1. **Futures USD-M 15m** for ALL symbols -> ``data/tsdb/<SYM>/15m``
   (extends the 12 originals, adds the 25 new) — foundation for a future 15m
   root / intra-bar macro context.
2. **Spot 1h** for ALL symbols -> a SEPARATE spot TSDB ``data/tsdb_spot/<SYM>/1h``
   (dual-market layer; feeds the spot-basis / vol-ratio channels at prepare).

Source: Binance Vision S3 daily ``klines`` zips (free, CSV + CHECKSUM).
Resumable: days before the existing meta end are skipped; ingest merges.

    python -m zhisa.scripts.extend_klines --symbols BTC/USDT --mode 15m_futures
    python -m zhisa.scripts.extend_klines --symbols BTC/USDT --mode spot_1h
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from urllib.error import HTTPError

from zhisa.scripts._real_data import futures_context_symbol_slug
from zhisa.storage.schema import OHLCV_COLUMNS, Timeframe, SeriesKey
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

FUTURES_BASE = "https://data.binance.vision/data/futures/um"
SPOT_BASE = "https://data.binance.vision/data/spot"
FUTURES_ROOT = Path("data/tsdb")
SPOT_ROOT = Path("data/tsdb_spot")
SYMBOLS_37 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
    "NEAR/USDT", "OP/USDT", "ARB/USDT", "APT/USDT", "SUI/USDT", "ETC/USDT",
    "INJ/USDT", "FIL/USDT", "HBAR/USDT", "ICP/USDT", "TON/USDT", "SEI/USDT",
    "TIA/USDT", "ZEC/USDT", "DASH/USDT", "EGLD/USDT", "IMX/USDT", "SAND/USDT",
    "MANA/USDT", "GALA/USDT", "ENS/USDT", "BLUR/USDT", "ORDI/USDT", "AAVE/USDT",
    "AXS/USDT",
]
START = date(2017, 8, 17)  # earliest Vision archives


def fetch_kline_month(base: str, slug: str, tf: str, ym: date, retries: int = 4) -> pd.DataFrame | None:
    """Fetch a whole month of klines as a single zip (efficiency: ~74 requests
    per symbol instead of ~2100 daily ones)."""
    url = f"{base}/monthly/klines/{slug}/{tf}/{slug}-{tf}-{ym.strftime('%Y-%m')}.zip"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zhisa/1.0"})
            with urllib.request.urlopen(req, timeout=60) as res:
                data = res.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, header=None)
            if df is None or df.empty:
                return None
            # Monthly archives carry a header row; daily ones do not.
            if str(df.iloc[0, 0]).strip().lower() == "open_time":
                df = df.iloc[1:]
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(df[0].astype(int), unit="ms", utc=True),
                "open": df[1].astype(float), "high": df[2].astype(float),
                "low": df[3].astype(float), "close": df[4].astype(float),
                "volume": df[5].astype(float),
            })
            return out.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        except HTTPError as e:
            if e.code == 404:
                return None
            import time
            time.sleep(0.5 + attempt * 0.5)
        except Exception:
            import time
            time.sleep(0.5 + attempt * 0.5)
    return None


def _months(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        y = cur.year + (1 if cur.month == 12 else 0)
        m = 1 if cur.month == 12 else cur.month + 1
        cur = date(y, m, 1)


def ingest_missing(db: TimeSeriesDB, sym: str, tf: Timeframe, base: str, start: date, end: date,
                   workers: int = 4) -> dict:
    key = SeriesKey(sym, tf)
    existing_end = None
    if db.has_series(key):
        existing_end = db.get_meta(key).end.date()
        if existing_end >= end:
            return {"symbol": sym, "skipped": "fresh", "rows": db.get_meta(key).row_count}
    day_start = (existing_end + timedelta(days=1)) if existing_end else start
    months = list(_months(day_start, end))
    slug = futures_context_symbol_slug(sym)
    chunks = []
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_kline_month, base, slug, tf.value, m): m for m in months}
        for f in as_completed(futs):
            try:
                mdf = f.result()
            except Exception:
                mdf = None
            if mdf is not None and len(mdf):
                chunks.append(mdf)
                ok += 1
    if not chunks:
        return {"symbol": sym, "months": len(months), "ok": 0, "rows": db.get_meta(key).row_count if db.has_series(key) else 0}
    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    meta = db.ingest(key, full)
    return {"symbol": sym, "months": len(months), "ok": ok, "rows": int(meta.row_count)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(SYMBOLS_37))
    ap.add_argument("--mode", choices=["15m_futures", "spot_1h", "all"], default="all")
    ap.add_argument("--start", default=str(START))
    ap.add_argument("--end", default=None)
    ap.add_argument("--spot-root", default=str(SPOT_ROOT))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    end_ = (pd.Timestamp(args.end, tz="UTC").date()
            if args.end else pd.Timestamp.now(tz="utc").date())
    start_ = pd.Timestamp(args.start).date()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.mode in ("15m_futures", "all"):
        db = TimeSeriesDB(FUTURES_ROOT)
        print(f"[futures 15m] {len(syms)} symbols {start_}..{end_}")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ingest_missing, db, s, Timeframe.M15, FUTURES_BASE, start_, end_): s
                    for s in syms}
            for f in as_completed(futs):
                print("  ", f.result())

    if args.mode in ("spot_1h", "all"):
        sroot = Path(args.spot_root)
        sroot.mkdir(parents=True, exist_ok=True)
        sdb = TimeSeriesDB(sroot)
        print(f"[spot 1h] {len(syms)} symbols {start_}..{end_} -> {sroot}")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ingest_missing, sdb, s, Timeframe.H1, SPOT_BASE, start_, end_): s
                    for s in syms}
            for f in as_completed(futs):
                print("  ", f.result())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())