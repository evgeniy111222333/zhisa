"""Binance USD-M daily bookDepth -> per-1h-bar microstructure channels.

Source: ``data.binance.vision/data/futures/um/daily/bookDepth/<SYM>/``
(zip, CSV: ``timestamp,percentage,depth,notional``; ~30s snapshots, levels -5..+5).

Storage strategy (efficiency): raw zips are NEVER persisted — each day is
downloaded, parsed and immediately reduced to ~forecast statistics per hour,
then dropped. The final artifact is one small parquet per symbol aligned to the
local 1h OHLCV index (hours without snapshots -> NaN, zero-filled downstream).

Channels (per bar, all MEAN over the hour):
    bd_mean_depth_pos1/neg1      cumulative depth at +-1% (near-touch)
    bd_mean_depth_pos5/neg5      cumulative depth at +-5%
    bd_depth5_ask / bd_depth5_bid  summed depth over levels +-1..5
    bd_imb_1  / bd_imb_5         (ask-bid)/(ask+bid) at +-1 and +-5
    bd_imb_notional_1            same imbalance in notional (USD)
    bd_ratio_1_5                 first-level concentration depth_p1/depth_p5
    bd_notional_pos5             mean USD notional at +5%

Run (canary then all):
    python -m zhisa.scripts.build_bookdepth_1h --symbols BTC/USDT --to 2026-08-23
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from urllib.error import HTTPError

from zhisa.scripts._real_data import futures_context_symbol_slug
from zhisa.storage.schema import Timeframe, SeriesKey
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
DEFAULT_OUT = Path("data/bookdepth/1h")
SYMBOLS_37 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
    "NEAR/USDT", "OP/USDT", "ARB/USDT", "APT/USDT", "SUI/USDT", "ETC/USDT",
    "INJ/USDT", "FIL/USDT", "HBAR/USDT", "ICP/USDT", "TON/USDT", "SEI/USDT",
    "TIA/USDT", "ZEC/USDT", "DASH/USDT", "EGLD/USDT", "IMX/USDT", "SAND/USDT",
    "MANA/USDT", "GALA/USDT", "ENS/USDT", "BLUR/USDT", "ORDI/USDT", "AAVE/USDT",
    "AXS/USDT",
]
BD_START = date(2023, 1, 1)  # archive start; bars before this -> NaN


def fetch_day(sym: str, day: date, retries: int = 4) -> pd.DataFrame | None:
    slug = futures_context_symbol_slug(sym)
    url = f"{BASE}/{slug}/{slug}-bookDepth-{day.isoformat()}.zip"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zhisa/1.0"})
            with urllib.request.urlopen(req, timeout=30) as res:
                data = res.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f)
            if df is None or df.empty:
                return None
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except HTTPError as e:
            if e.code == 404:
                return None  # no data published for that day
            # rate-limited -> back off and retry
            import time
            time.sleep(2 + attempt * 2)
        except Exception as e:  # transient parse/network error
            import time
            time.sleep(2 + attempt * 2)
    return None


def aggregate_day(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce one day of ~30s snapshots (levels -5..+5) to 1h bar stats."""
    piv = df.pivot_table(index="timestamp", columns="percentage",
                         values=["depth", "notional"], aggfunc="mean")
    d = piv["depth"].resample("1h", closed="left", label="left").mean()
    n = piv["notional"].resample("1h", closed="left", label="left").mean()
    p1, n1 = d[1], d[-1]
    p5, npt5 = d[5], d[-5]
    ask5 = d[[1, 2, 3, 4, 5]].sum(axis=1)
    bid5 = d[[-1, -2, -3, -4, -5]].sum(axis=1)
    n1_p = n[1]; n1_n = n[-1]
    out = pd.DataFrame({
        "bd_mean_depth_pos1": p1.astype(np.float32),
        "bd_mean_depth_neg1": n1.astype(np.float32),
        "bd_mean_depth_pos5": p5.astype(np.float32),
        "bd_mean_depth_neg5": npt5.astype(np.float32),
        "bd_depth5_ask": ask5.astype(np.float32),
        "bd_depth5_bid": bid5.astype(np.float32),
        "bd_imb_1": ((p1 - n1) / (p1 + n1 + 1e-12)).astype(np.float32),
        "bd_imb_5": ((ask5 - bid5) / (ask5 + bid5 + 1e-12)).astype(np.float32),
        "bd_imb_notional_1": ((n1_p - n1_n) / (n1_p + n1_n + 1e-12)).astype(np.float32),
        "bd_ratio_1_5": (p1 / (p5 + 1e-12)).astype(np.float32),
        "bd_notional_pos5": n[5].astype(np.float32),
    })
    return out


def build_symbol(sym: str, out_root: Path, start: date, end: date) -> Path:
    p = out_root / f"{futures_context_symbol_slug(sym)}.parquet"
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    chunks = []
    ok_days = 0
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=4) as ex:
        futs = {ex.submit(fetch_day, sym, d): d for d in days}
        for f in as_completed(futs):
            day = futs[f]
            try:
                df = f.result()
            except Exception:
                df = None
            if df is not None and len(df):
                agg = aggregate_day(df)
                agg.attrs["day"] = day.isoformat()
                chunks.append(agg)
                ok_days += 1
    if not chunks:
        logger.warning("%s: no bookDepth data", sym)
        return p
    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    # align exactly onto the local 1h OHLCV index (shared hour grid)
    db = TimeSeriesDB(Path("data/tsdb"))
    t1 = db.read(SeriesKey(sym, Timeframe.H1)).index
    full = full.reindex(t1)
    p.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(p)
    logger.info("%s: %d hours, %d days -> %s", sym, int(full.notna().any(axis=1).sum()),
                ok_days, p)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(SYMBOLS_37))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--start", default=str(BD_START))
    ap.add_argument("--end", default=None, help="ISO date; default today")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    out = Path(args.out)
    end = (pd.Timestamp(args.end, tz="UTC").date()
           if args.end else pd.Timestamp.now(tz="utc").date())
    start = pd.Timestamp(args.start).date()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"bookDepth -> {out} | {len(syms)} symbols | {start}..{end}")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(build_symbol, s, out, start, end): s for s in syms}
        for f in as_completed(futs):
            s = futs[f]
            try:
                print("  done", s, "->", f.result())
            except Exception as e:
                print("  !", s, type(e).__name__, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
