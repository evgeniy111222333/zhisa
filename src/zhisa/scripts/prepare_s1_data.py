"""CLI: prepare an S1 self-supervised dataset from the local TSDB.

This is the data-prep counterpart of ``zhisa-train-s1``: it takes raw
OHLCV (and optional futures context) from the local time-series DB and
writes a versioned, deterministic, multi-symbol dataset ready for the
S1 trainer to consume.

Usage::

    zhisa-prepare-s1-data \
        --tsdb-root data/tsdb \
        --out-root  data/prepared/s1_15m_v1 \
        --symbols   BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT,XRP/USDT \
        --timeframe 15m \
        --with-futures-context \
        --context-root data/futures_context/binance_usdm \
        --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \
        --embargo-bars 96

Outputs (under ``--out-root``)::

    manifest.json              # version + checksums + row counts
    symbols/{SYMBOL}.parquet   # one frame per symbol, fully cleaned
    splits/{train,val,test}.parquet
    checksums.txt              # human-readable input/output checksums
    preparation_log.json       # full audit trail

The output is **idempotent** — re-running with identical inputs
produces identical checksums. This makes the S1 checkpoint produced
from it reproducible across machines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from zhisa.data.feature_specs import (
    CURRENT_VERSION,
    CoveragePolicy,
    GapPolicy,
)
from zhisa.data.preparation import PrepareConfig, prepare_dataset
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_TSDB_ROOT = "data/tsdb"
DEFAULT_CONTEXT_ROOT = "data/futures_context/binance_usdm"
DEFAULT_OUT_ROOT = "data/prepared/s1_15m_v1"
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT",
    "BNB/USDT", "ADA/USDT", "XRP/USDT",
]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zhisa-prepare-s1-data",
        description="Prepare an S1 self-supervised dataset from the local TSDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--tsdb-root", type=Path, default=Path(DEFAULT_TSDB_ROOT),
        help="Root of the local OHLCV time-series DB.",
    )
    p.add_argument(
        "--out-root", type=Path, default=Path(DEFAULT_OUT_ROOT),
        help="Where to write the prepared dataset.",
    )
    p.add_argument(
        "--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated CCXT-style symbols to include.",
    )
    p.add_argument(
        "--timeframe", type=str, default="15m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h"],
        help="Target OHLCV timeframe. 15m is the S1 default.",
    )
    p.add_argument(
        "--with-futures-context", action="store_true", default=True,
        help="Merge Binance USD-M futures context (funding, OI, LS ratios, taker flow).",
    )
    p.add_argument(
        "--no-futures-context", dest="with_futures_context",
        action="store_false",
        help="Disable futures-context merge even when context files exist.",
    )
    p.add_argument(
        "--context-root", type=Path, default=Path(DEFAULT_CONTEXT_ROOT),
        help="Root containing SYMBOL/timeframe/context.parquet files.",
    )
    p.add_argument(
        "--context-timeframe", type=str, default=None,
        help=(
            "Timeframe of the context parquet files (e.g. '5m', '15m'). "
            "Default: same as --timeframe. The merger reads "
            "<context-root>/<SYMBOL>/<context-timeframe>/context.parquet."
        ),
    )

    g = p.add_argument_group("gap & coverage policies")
    g.add_argument(
        "--max-ffill-bars", type=int, default=4,
        help="Maximum number of consecutive target-timeframe bars to forward-fill.",
    )
    g.add_argument(
        "--keep-long-gaps", action="store_true",
        help="Do NOT drop rows that follow a long gap; keep them as NaN.",
    )
    g.add_argument(
        "--coverage-start", type=str, default=None,
        help="ISO date string. Hard-clip all series to start no earlier than this.",
    )
    g.add_argument(
        "--coverage-end", type=str, default=None,
        help="ISO date string. Hard-clip all series to end no later than this.",
    )
    g.add_argument(
        "--min-bars-per-symbol", type=int, default=1000,
        help="Drop symbols whose aligned coverage is shorter than this many bars.",
    )

    s = p.add_argument_group("splits")
    s.add_argument("--train-frac", type=float, default=0.70)
    s.add_argument("--val-frac", type=float, default=0.15)
    s.add_argument("--test-frac", type=float, default=0.15)
    s.add_argument(
        "--embargo-bars", type=int, default=96,
        help="Bars (at the chosen timeframe) to drop between train/val and val/test.",
    )

    p.add_argument("--version", type=str, default=CURRENT_VERSION)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Plan the run, print the resolved config, and exit without writing.",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    # Resolve symbols.
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("error: --symbols must list at least one symbol", file=sys.stderr)
        return 2

    # Resolve policies.
    gap_policy = GapPolicy(
        max_ffill_bars=int(args.max_ffill_bars),
        drop_long_gaps=not bool(args.keep_long_gaps),
        require_monotonic=True,
    )
    coverage_policy = CoveragePolicy(
        start=args.coverage_start,
        end=args.coverage_end,
        min_bars=int(args.min_bars_per_symbol),
    )

    cfg = PrepareConfig(
        tsdb_root=Path(args.tsdb_root),
        out_root=Path(args.out_root),
        symbols=symbols,
        timeframe=str(args.timeframe),
        context_timeframe=str(args.context_timeframe) if args.context_timeframe else str(args.timeframe),
        with_futures_context=bool(args.with_futures_context),
        context_root=Path(args.context_root) if args.with_futures_context else None,
        gap_policy=gap_policy,
        coverage_policy=coverage_policy,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        embargo_bars=int(args.embargo_bars),
        version=str(args.version),
    )

    # Pretty-print the resolved config so a dry-run is informative.
    summary = {
        "tsdb_root": str(cfg.tsdb_root),
        "out_root": str(cfg.out_root),
        "symbols": cfg.symbols,
        "timeframe": cfg.timeframe,
        "with_futures_context": cfg.with_futures_context,
        "context_root": str(cfg.context_root) if cfg.context_root else None,
        "gap_policy": cfg.gap_policy.__dict__,
        "coverage_policy": cfg.coverage_policy.__dict__,
        "splits": {
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "test_frac": cfg.test_frac,
            "embargo_bars": cfg.embargo_bars,
        },
        "version": cfg.version,
    }
    print(json.dumps({"plan": summary}, indent=2))

    if args.dry_run:
        print("dry-run: not writing any files.")
        return 0

    try:
        manifest = prepare_dataset(cfg)
    except Exception as exc:
        logger.exception("preparation failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Final summary.
    print(json.dumps({
        "status": "ok",
        "manifest_path": str(cfg.out_root / "manifest.json"),
        "manifest_checksum": manifest.output_checksum,
        "symbols": manifest.symbols,
        "rows_per_symbol": manifest.rows_per_symbol,
        "rows_total": manifest.rows_total,
        "window": {"start": manifest.start, "end": manifest.end},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
