"""Data-update cycle: keep compiled chart artefacts fresh as raw data grows.

The ideal pipeline treats "ingest new bars" as a first-class event:

1. **Detect** which prepared symbols are stale relative to the raw ``data/tsdb``
   (covered-prefix hash no longer matches, or more bars available).
2. **Refresh** each symbol's compiled chart store via the parallel / incremental
   render job (prefix copied, only the tail re-rendered).
3. **Verify** the refreshed artefact is byte-consistent with a fresh full build.

This module is the automation that closes the loop, so a stale dataset can
never silently reach a trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from zhisa.data.chart_store import CompiledChartStore
from zhisa.data.dataset import SampleSpec
from zhisa.data.render_job import RenderJobStats, materialize_parallel
from zhisa.rendering.spec import RenderSpec


@dataclass
class SymbolRefresh:
    symbol: str
    stale: bool
    n_images: int
    reused_prefix_rows: int = 0
    rendered_rows: int = 0
    content_key: str = ""


@dataclass
class DataCycleReport:
    refreshed: list[SymbolRefresh]
    total_symbols: int
    stale_count: int

    def summary(self) -> str:
        lines = [f"symbols={self.total_symbols} stale={self.stale_count}"]
        for r in self.refreshed:
            lines.append(
                f"  {r.symbol}: stale={r.stale} n={r.n_images} "
                f"reused={r.reused_prefix_rows} rendered={r.rendered_rows}"
            )
        return "\n".join(lines)


def _symbol_n_images(df: pd.DataFrame, spec: SampleSpec) -> int:
    horizon_max = max(spec.horizons) if spec.horizons else 0
    return max(0, len(df) - spec.chart_window - horizon_max - 1)


def refresh_symbol_store(
    df: pd.DataFrame,
    spec: SampleSpec,
    charts_dir: Path | str,
    *,
    render_spec: Optional[RenderSpec] = None,
    workers: int = 0,
    chunk: int = 5_000,
    engine: str = "cpu",
) -> tuple[CompiledChartStore, SymbolRefresh, RenderJobStats]:
    """Raise/refresh a compiled chart store for one symbol frame.

    Incremental when a shorter artefact of the same render identity exists and
    its covered prefix matches (only the tail is re-rendered), otherwise a full
    parallel build. Returns ``(store, refresh, stats)``.
    """
    charts_dir = Path(charts_dir)
    render_spec = render_spec or RenderSpec(size=spec.image_size)
    n = _symbol_n_images(df, spec)
    store, stats = materialize_parallel(
        df, window=spec.chart_window, spec=render_spec,
        n=n, out_root=charts_dir, workers=workers, chunk_size=chunk, engine=engine,
    )
    store.assert_fresh(df, spec.chart_window, render_spec, n=n)
    meta = store.render_meta
    refresh = SymbolRefresh(
        symbol=str(getattr(df, "name", "unknown")),
        stale=True,  # we (re)built against the current frame
        n_images=n,
        reused_prefix_rows=stats.reused_prefix_rows,
        rendered_rows=stats.rendered_rows,
        content_key=str(meta.get("content_key")),
    )
    if stats.reused_artifact:
        # exact cache hit: not stale at all
        refresh.stale = False
    return store, refresh, stats


def update_prepared_charts(
    prepared_root: Path | str,
    charts_dir: Path | str,
    spec: SampleSpec,
    *,
    render_spec: Optional[RenderSpec] = None,
    workers: int = 0,
    chunk: int = 5_000,
    symbols: Optional[list[str]] = None,
    engine: str = "cpu",
) -> DataCycleReport:
    """Refresh compiled charts for every prepared symbol in ``prepared_root``.

    Iterates ``prepared_root/symbols/*.parquet``, computes the current sample
    count from ``spec``, and builds/refreshes the content-addressed chart store
    under ``charts_dir``. Symbols are skipped (reused) when their artefact is
    already fresh.
    """
    prepared_root = Path(prepared_root)
    charts_dir = Path(charts_dir)
    chart_dir = charts_dir
    refreshed: list[SymbolRefresh] = []
    stale_count = 0

    symbol_dir = prepared_root / "symbols"
    if not symbol_dir.is_dir():
        raise FileNotFoundError(f"no prepared symbols directory: {symbol_dir}")
    paths = sorted(symbol_dir.glob("*.parquet"))
    for p in paths:
        symbol_slug = p.stem
        if symbols is not None and symbol_slug not in symbols:
            continue
        df = pd.read_parquet(p)
        df.name = symbol_slug
        _, refresh, _ = refresh_symbol_store(
            df, spec, chart_dir,
            render_spec=render_spec, workers=workers, chunk=chunk, engine=engine,
        )
        stale_count += int(refresh.stale)
        refreshed.append(refresh)
    return DataCycleReport(refreshed=refreshed, total_symbols=len(refreshed), stale_count=stale_count)