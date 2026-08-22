"""CLI: refresh compiled chart artefacts after raw-data growth (data cycle)."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, help="Prepared dataset root with symbols/*.parquet")
    parser.add_argument("--charts-dir", required=True, help="Compiled chart store root (content-addressed)")
    parser.add_argument("--chart-window", type=int, default=128, help="Chart window (bars)")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--render-workers", type=int, default=0)
    parser.add_argument("--render-chunk", type=int, default=5_000)
    parser.add_argument("--render-engine", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--max-horizon", type=int, default=64, help="Horizon used for sample trimming (mirrors training)")
    parser.add_argument("--symbol", action="append", default=None, help="Restrict refresh to a symbol slug (repeatable)")
    args = parser.parse_args(argv)

    from zhisa.data.data_cycle import update_prepared_charts
    from zhisa.data.dataset import SampleSpec

    spec = SampleSpec(
        chart_window=int(args.chart_window),
        feature_window=int(args.chart_window),
        image_size=int(args.image_size),
        horizons=(4, 16, int(args.max_horizon)),
    )
    report = update_prepared_charts(
        args.prepared_root,
        args.charts_dir,
        spec,
        workers=args.render_workers,
        chunk=args.render_chunk,
        symbols=args.symbol,
        engine=args.render_engine,
    )
    print(report.summary())
    return 1 if report.stale_count else 0


if __name__ == "__main__":
    raise SystemExit(main())