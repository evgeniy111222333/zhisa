"""CLI: verify prepared-root lineage and (optionally) chart-store reuse.

Screams (exit code 1, explicit message) when invariants break:

    python -m zhisa.scripts.check_lineage --prepared-root data/prepared/X \
        [--tsdb-root data/tsdb] [--strict] \
        [--charts-dir data/charts --window 128 --image-size 128]

``--strict`` requires lineage.json to exist and match exactly; without it
a fresh scan is allowed but any *drift* still fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zhisa.data.lineage import (
    LineageError,
    assert_prepared_consistent,
    guard_reuse,
    probe_reuse,
    scan_prepared,
)
from zhisa.data.render_job import materialize_parallel  # noqa: F401  (render identity)
from zhisa.rendering.spec import RenderSpec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default=None, help="comma list; default = all")
    ap.add_argument("--tsdb-root", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--charts-dir", default=None)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--min-reuse-ratio", type=float, default=0.34)
    args = ap.parse_args(argv)

    root = Path(args.prepared_root)
    symbols = args.symbols.split(",") if args.symbols else None
    try:
        scan = assert_prepared_consistent(root, symbols=symbols,
                                          expect_full_recompute=False)
    except LineageError as exc:
        print(f"SCREAM: {exc}", file=sys.stderr)
        return 1
    print("lineage OK:")
    print(json.dumps(scan.as_dict(), indent=1))
    if args.tsdb_root:
        from zhisa.data.lineage import fingerprint_tsdb
        cur = fingerprint_tsdb(Path(args.tsdb_root), list(scan.scannable_symbols))
        print(f"tsdb fingerprint now:  {cur}")
        print(f"tsdb fingerprint from scan: {scan.tsdb_fingerprint}")
        if scan.tsdb_fingerprint and cur != scan.tsdb_fingerprint:
            if args.strict:
                print("SCREAM: tsdb drifted since prepare (strict mode)", file=sys.stderr)
                return 1
            print("WARN: tsdb drifted since prepare (recorded value kept)")
    if args.charts_dir:
        import pandas as pd
        seg_dfs = []
        for sym in scan.scannable_symbols[:2]:
            df = pd.read_parquet(root / "symbols" / f"{sym}.parquet").sort_index()
            delta = df.index.to_series().diff().median()
            segs = df.groupby(df.index.to_series().diff().ne(delta).cumsum(), sort=False)
            seg_dfs += [s for _, s in segs][:2]
        spec = RenderSpec(size=args.image_size)
        probe = probe_reuse(Path(args.charts_dir), seg_dfs, args.window, spec)
        print("reuse probe:", json.dumps(probe, indent=1))
        if probe["hit_rate"] < args.min_reuse_ratio:
            try:
                guard_reuse(Path(args.charts_dir), seg_dfs, args.window, spec,
                            min_reuse_ratio=args.min_reuse_ratio)
            except LineageError as exc:
                print(f"SCREAM: {exc}", file=sys.stderr)
                return 1
        print("chart-reuse OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())