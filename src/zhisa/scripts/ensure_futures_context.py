"""Ensure the full Binance USD-M futures context (funding/OI/ratios) exists for
all S1 symbols in the canonical consumer layout.

Canonical contract (see ``zhisa.data.context_merger`` and
``prepare_s1_data --context-root``)::

    data/futures_context/binance_usdm/<SLUG>/15m/context.parquet

Sources, in order:
  1. already-downloaded context under ``data/tsdb/binance/<SYM>/15m/context.parquet`
     (copy into the canonical location);
  2. the full-hybrid downloader (Vision S3 daily metrics with Open Interest +
     fapi funding/klines) run for every missing symbol via
     ``zhisa.scripts.download_full_context``, then renamed to ``context.parquet``.

Idempotent: symbols whose canonical ``context.parquet`` already exists are skipped.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from zhisa.scripts._real_data import futures_context_symbol_slug  # reuse slug rule
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TABLE_ROOT = Path("data/tsdb/binance")
DEFAULT_CANONICAL_ROOT = Path("data/futures_context/binance_usdm")

SYMBOLS_12 = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT",
]


def canonical_path(root: Path, symbol: str, tf: str = "15m") -> Path:
    slug = futures_context_symbol_slug(symbol)
    return root / slug / tf / "context.parquet"


def already_downloaded(symbol: str) -> Path | None:
    safe = symbol.replace("/", "_")
    legacy = DEFAULT_TABLE_ROOT / safe / "15m" / "context.parquet"
    if legacy.exists():
        return legacy
    old = DEFAULT_TABLE_ROOT / safe / "15m" / "futures_context.parquet"
    if old.exists():
        return old
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", type=str, default=",".join(SYMBOLS_12))
    ap.add_argument("--root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    ap.add_argument("--start", type=str, default="2019-01-01")
    ap.add_argument("--end", type=str, default="2026-08-23")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    root = args.root
    need_download: list[str] = []
    for sym in symbols:
        dst = canonical_path(root, sym)
        if dst.exists():
            logger.info("skip %s: %s present", sym, dst)
            continue
        src = already_downloaded(sym)
        if src is not None:
            if args.dry_run:
                print(f"[copy] {src} -> {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                logger.info("copied %s -> %s", src, dst)
            continue
        need_download.append(sym)

    if args.dry_run:
        print("would download:", need_download)
        return 0

    if not need_download:
        print("ALL CONTEXT READY")
        return 0

    print(f"downloading {len(need_download)} symbols: {need_download}")
    rc = 0
    for i, sym in enumerate(need_download, start=1):
        dst = canonical_path(root, sym)
        stage = root / "_stage"
        cmd = [
            sys.executable, "-m", "zhisa.scripts.download_full_context",
            "--symbol", sym, "--start", args.start, "--end", args.end,
            "--out-dir", str(stage),
        ]
        logger.info("[%d/%d] %s : %s", i, len(need_download), sym, " ".join(cmd))
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            logger.error("%s failed (rc=%d)", sym, res.returncode)
            rc = max(rc, 1)
            continue
        produced = stage / sym.replace("/", "_") / "15m" / "futures_context.parquet"
        if not produced.exists():
            logger.error("%s produced no context parquet", sym)
            rc = max(rc, 1)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(dst))
        logger.info("moved %s -> %s", produced, dst)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())