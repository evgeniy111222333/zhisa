"""Generate synthetic OHLCV data and save as parquet / CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zhisa.data.synthetic import MarketConfig, generate_market


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic OHLCV market.")
    parser.add_argument("--out", type=str, default="data/synth", help="output directory")
    parser.add_argument("--bars", type=int, default=20_000, help="number of bars")
    parser.add_argument("--freq", type=str, default="5min", help="bar frequency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    args = parser.parse_args(argv)

    cfg = MarketConfig(n_bars=args.bars, freq=args.freq, seed=args.seed)
    df = generate_market(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        df.to_parquet(out / "synth.parquet")
    else:
        df.to_csv(out / "synth.csv")
    print(f"Wrote {len(df)} bars to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
