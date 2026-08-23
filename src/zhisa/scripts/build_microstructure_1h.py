"""Add cheap intra-hour MICROSTRUCTURE channels to the local 1h TSDB series.

Derived entirely from the LOCAL 1m klines (OHLCV only), one aggregate per
closed hour — deterministic, causal, reproducible:

    micro_bars_1h        number of 1m bars present in the hour (data density
                         / exchange gap proxy)
    micro_range_ratio_1h intra-hour range (1m high-low path) normalised by the
                         enclosing 1h OHLC range — how *bursty* the hour is
    micro_max_1m_ret_1h  largest |1m log-return| inside the hour (jump proxy)
    micro_top_vol_share_1h  share of the hour's volume contained in a single
                         (the largest) 1m bar — volume concentration

Columns are appended to the 1h DataFrame and re-ingested via
``TimeSeriesDB.ingest`` (OHLCV first, micro columns after), so the prepared
root will surface them as ``ctx_micro_*`` channels.

    python -m zhisa.scripts.build_microstructure_1h --symbols BTC/USDT --tsdb-root data/tsdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.storage.schema import Timeframe, SeriesKey, OHLCV_COLUMNS
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ROOT = Path("data/tsdb")
SYMBOLS_12 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
]


def build_micro_1h(m1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1m OHLCV into one row per closed hour."""
    df = m1[["open", "high", "low", "close", "volume"]].copy()
    g = df.resample("1h", closed="left", label="left")
    hrs = g["close"].count()
    hr_high = g["high"].max()
    hr_low = g["low"].min()
    hr_open = g["open"].first()
    hr_vol = g["volume"].sum()
    # intra-hour "path" range = sum over 1m bars of their own (high-low)
    path = (df["high"] - df["low"]).resample("1h", closed="left", label="left").sum()
    # largest |1m log-return| within the hour
    lr = np.log(df["close"]).diff()
    max_ret = lr.abs().resample("1h", closed="left", label="left").max()
    # top-1m volume share of the hour
    top_vol = df["volume"].resample("1h", closed="left", label="left").max()
    # enclosing 1h OHLC range
    ohlc_range = (hr_high - hr_low).clip(lower=1e-12)
    out = pd.DataFrame({
        "micro_bars_1h": hrs.astype(np.float32),
        "micro_range_ratio_1h": (path / ohlc_range).astype(np.float32),
        "micro_max_1m_ret_1h": max_ret.fillna(0.0).astype(np.float32),
        "micro_top_vol_share_1h": (top_vol / hr_vol.replace(0.0, np.nan)).astype(np.float32),
    })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(SYMBOLS_12))
    ap.add_argument("--tsdb-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    db = TimeSeriesDB(Path(args.tsdb_root))
    rc = 0
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        key_h = SeriesKey(sym, Timeframe.H1)
        key_m1 = SeriesKey(sym, Timeframe.M1)
        if not db.has_series(key_m1):
            logger.warning("%s: no 1m series, skipping", sym)
            continue
        m1 = db.read(key_m1)
        h1 = db.read(key_h)
        micro = build_micro_1h(m1)
        joined = h1.join(micro[~micro.index.duplicated(keep="last")], how="left")
        # keep OHLCV columns first, then micro channels
        order = [c for c in OHLCV_COLUMNS if c in joined.columns] + \
                [c for c in micro.columns if c in joined.columns]
        joined = joined[order]
        n_micro = int(joined[list(micro.columns)].notna().sum().min())
        print(f"{sym:10s} micro bars covered={n_micro}/{len(h1)} "
              f"cols added={list(micro.columns)}")
        if args.dry_run:
            continue
        meta = db.ingest(key_h, joined)
        logger.info("%s -> rows=%d end=%s cols=%d", sym, meta.row_count, meta.end, len(joined.columns))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())