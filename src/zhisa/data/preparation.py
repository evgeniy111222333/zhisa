"""S1 self-supervised data preparation pipeline.

The pipeline takes raw OHLCV (and optional futures context) from the
local TSDB and produces a versioned, deterministic, multi-symbol
dataset ready to be consumed by ``zhisa-train-s1``.

Stages (run in strict order; each stage is pure given its inputs):

1. **Load** — read each symbol's OHLCV from the local TSDB at the
   requested timeframe (15m by default).
2. **Repair** — apply ``repair_ohlcv`` (forward-fill, drop dups, clamp
   OHLC, fill zero volume). This guarantees the schema is valid.
3. **Gap policy** — reindex onto a strict ``15min`` grid. Forward-fill
   short gaps (default: <= 4 bars) and drop longer gaps. This makes
   the index **dense** so downstream windowing never sees a hole.
4. **Coverage alignment** — clip every symbol to a shared window
   (``start`` = max of per-symbol starts, ``end`` = min of per-symbol
   ends). Symbols with too few bars are dropped.
5. **Context merge** — left-join the Binance USD-M futures context
   (funding, OI, long/short ratios, taker flow) for symbols that have
   it. Anti-look-ahead shift of ``lag_bars`` is applied.
6. **Schema assert** — verify the v1 contract (tz-aware UTC index,
   monotonic, OHLCV numeric, no NaN/Inf in OHLCV).
7. **Checksum** — compute the manifest checksum and persist it.
8. **Split** — produce temporal ``train/val/test`` splits with an
   embargo gap to prevent any leakage across the boundary.

Output layout
-------------

    {out_root}/
      manifest.json
      symbols/
        BTC_USDT.parquet       # cleaned + gap-filled + context-merged
        ETH_USDT.parquet
        ...
      splits/
        train.parquet          # combined train rows from every symbol
        val.parquet
        test.parquet
      checksums.txt            # human-readable checksum summary
      preparation_log.json     # the full audit trail

The per-symbol parquet files are what ``train_s1.py`` consumes via
``load_market_dataframe`` (``--data-source csv`` with ``--csv`` pointing
at the chosen symbol file). The splits parquet is provided for
downstream S2/S4 code paths that want a precomputed split.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from zhisa.data.context_merger import attach_context_for_symbol
from zhisa.data.feature_specs import (
    CoveragePolicy,
    CURRENT_VERSION,
    GapPolicy,
    PreparedDataset,
    V1_REQUIRED_COLUMNS,
    V1_TIMEFRAME_15M,
    assert_v1_schema,
    is_supported,
)
from zhisa.storage.quality import audit_ohlcv, repair_ohlcv
from zhisa.storage.schema import OHLCV_COLUMNS, SeriesKey, Timeframe
from zhisa.storage.tsdb import TimeSeriesDB
from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------

@dataclass
class PrepareConfig:
    """Top-level configuration for a single preparation run.

    Attributes
    ----------
    tsdb_root : Path
        Root of the local OHLCV time-series DB.
    out_root : Path
        Where to write the prepared dataset. Existing directories are
        not deleted — only files that this run produces are written.
    symbols : list[str]
        CCXT-style symbols to include (e.g. ``["BTC/USDT", "ETH/USDT"]``).
    timeframe : str
        Target timeframe for S1 (default ``"15m"``).
    with_futures_context : bool
        If True, also merge the Binance USD-M context parquet for
        symbols that have one.
    context_root : Path
        Root of the local futures-context parquets.
    gap_policy : GapPolicy
        How to handle short gaps and bad rows.
    coverage_policy : CoveragePolicy
        How to align coverage across symbols.
    train_frac, val_frac, test_frac : float
        Temporal split fractions (must sum to 1.0).
    embargo_bars : int
        Number of bars to drop between train/val and val/test splits
        to prevent label leakage through rolling windows.
    version : str
        Feature spec version. ``"v1"`` is the only supported one for now.
    seed : int
        Reserved for stochastic augmentations; not used in v1.
    """

    tsdb_root: Path
    out_root: Path
    symbols: list[str]
    timeframe: str = V1_TIMEFRAME_15M
    context_timeframe: str = V1_TIMEFRAME_15M
    with_futures_context: bool = True
    context_root: Optional[Path] = None
    gap_policy: GapPolicy = None  # type: ignore[assignment]
    coverage_policy: CoveragePolicy = None  # type: ignore[assignment]
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    embargo_bars: int = 96  # 96 * 15m = 24h
    version: str = CURRENT_VERSION
    seed: int = 0

    def __post_init__(self) -> None:
        self.tsdb_root = Path(self.tsdb_root)
        self.out_root = Path(self.out_root)
        if self.context_root is not None:
            self.context_root = Path(self.context_root)
        if self.gap_policy is None:
            self.gap_policy = GapPolicy()
        if self.coverage_policy is None:
            self.coverage_policy = CoveragePolicy()
        if not is_supported(self.version):
            raise ValueError(
                f"Unsupported feature spec version: {self.version!r} "
                f"(supported: v1)"
            )
        total = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Stage 1 — load
# ---------------------------------------------------------------------------

def _load_symbol(tsdb: TimeSeriesDB, symbol: str, timeframe: str) -> pd.DataFrame:
    key = SeriesKey(instrument=symbol, timeframe=Timeframe.from_str(timeframe))
    if not tsdb.has_series(key):
        raise FileNotFoundError(
            f"TSDB missing series {key}. Run `zhisa-ingest-real-data` first."
        )
    df = tsdb.read(key)
    # Defensive: enforce UTC + schema.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: index must be DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[list(OHLCV_COLUMNS)].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ---------------------------------------------------------------------------
# Stage 2 — repair
# ---------------------------------------------------------------------------

def _repair(
    df: pd.DataFrame,
    where: str,
    timeframe: str = V1_TIMEFRAME_15M,
) -> tuple[pd.DataFrame, dict]:
    report = audit_ohlcv(
        df, expected_freq=Timeframe.from_str(timeframe).pandas_freq
    )
    repaired, new_report = repair_ohlcv(df, report=report)
    audit = {
        "before": {
            "rows": int(report.total_rows),
            "errors": [i.to_dict() if hasattr(i, "to_dict") else str(i) for i in report.errors],
            "warnings": [i.to_dict() if hasattr(i, "to_dict") else str(i) for i in report.warnings],
        },
        "after": {
            "rows": int(new_report.total_rows),
            "errors": [i.to_dict() if hasattr(i, "to_dict") else str(i) for i in new_report.errors],
            "warnings": [i.to_dict() if hasattr(i, "to_dict") else str(i) for i in new_report.warnings],
        },
    }
    return repaired, audit


# ---------------------------------------------------------------------------
# Stage 3 — gap policy: reindex onto a dense grid
# ---------------------------------------------------------------------------

def _apply_gap_policy(
    df: pd.DataFrame,
    timeframe: str,
    policy: GapPolicy,
) -> tuple[pd.DataFrame, dict]:
    """Make the index dense at the target timeframe.

    The process:

    * Build a ``pd.date_range`` covering the existing span with the
      exact timeframe.
    * Reindex onto that range using ``method='ffill'`` with a limit of
      ``policy.max_ffill_bars``. Bars beyond the limit become NaN.
    * Drop NaN rows if ``policy.drop_long_gaps`` is True.
    """
    if len(df) == 0:
        return df, {"reindexed_bars": 0, "dropped_bars": 0}

    tf = Timeframe.from_str(timeframe)
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=tf.pandas_freq,
        tz="UTC",
    )
    if policy.max_ffill_bars > 0:
        reindexed = df.reindex(
            full_index, method="ffill", limit=policy.max_ffill_bars
        )
    else:
        reindexed = df.reindex(full_index)
    reindexed.index.name = "timestamp"

    n_before = len(reindexed)
    if policy.drop_long_gaps:
        reindexed = reindexed.dropna(subset=list(OHLCV_COLUMNS))
    n_after = len(reindexed)

    info = {
        "target_rows": int(n_before),
        "kept_rows": int(n_after),
        "dropped_bars": int(n_before - n_after),
        "ffill_limit": int(policy.max_ffill_bars),
    }
    return reindexed, info


# ---------------------------------------------------------------------------
# Stage 4 — coverage alignment
# ---------------------------------------------------------------------------

def _align_coverage(
    per_symbol: dict[str, pd.DataFrame],
    policy: CoveragePolicy,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Clip every symbol to a shared time window.

    Default: ``start = max(per_symbol starts)`` and
    ``end = min(per_symbol ends)``. Drops symbols that fall below
    ``policy.min_bars`` after alignment.
    """
    if not per_symbol:
        return {}, {"aligned_window": None, "dropped_symbols": []}

    original = per_symbol
    dropped = [sym for sym, df in original.items() if len(df) < policy.min_bars]
    per_symbol = {
        sym: df for sym, df in original.items() if len(df) >= policy.min_bars
    }
    if not per_symbol:
        return {}, {
            "aligned_window": None,
            "dropped_symbols": dropped,
            "per_symbol_starts": {sym: str(df.index.min()) for sym, df in original.items()},
            "per_symbol_ends": {sym: str(df.index.max()) for sym, df in original.items()},
        }

    starts = [df.index.min() for df in per_symbol.values()]
    ends = [df.index.max() for df in per_symbol.values()]

    auto_start = max(starts)
    auto_end = min(ends)

    user_start = pd.Timestamp(policy.start) if policy.start else None
    user_end = pd.Timestamp(policy.end) if policy.end else None

    final_start = max(auto_start, user_start) if user_start else auto_start
    final_end = min(auto_end, user_end) if user_end else auto_end
    if final_start >= final_end:
        raise ValueError(
            f"coverage window empty: start={final_start}, end={final_end}"
        )

    aligned: dict[str, pd.DataFrame] = {}
    for sym, df in per_symbol.items():
        clipped = df[(df.index >= final_start) & (df.index <= final_end)]
        if len(clipped) < policy.min_bars:
            dropped.append(sym)
            continue
        aligned[sym] = clipped

    info = {
        "aligned_window": {
            "start": str(final_start),
            "end": str(final_end),
            "bars_per_symbol_min": int(min((len(df) for df in aligned.values()), default=0)),
            "bars_per_symbol_max": int(max((len(df) for df in aligned.values()), default=0)),
        },
        "dropped_symbols": sorted(set(dropped)),
        "per_symbol_starts": {sym: str(df.index.min()) for sym, df in original.items()},
        "per_symbol_ends": {sym: str(df.index.max()) for sym, df in original.items()},
    }
    return aligned, info


# ---------------------------------------------------------------------------
# Stage 5 — context merge
# ---------------------------------------------------------------------------

def _merge_context(
    per_symbol: dict[str, pd.DataFrame],
    cfg: PrepareConfig,
) -> tuple[dict[str, pd.DataFrame], dict]:
    info: dict = {"merged": {}, "skipped": []}
    out: dict[str, pd.DataFrame] = {}
    for sym, df in per_symbol.items():
        if not cfg.with_futures_context or cfg.context_root is None:
            out[sym] = df
            info["skipped"].append({"symbol": sym, "reason": "context_disabled"})
            continue
        try:
            merged = attach_context_for_symbol(
                df,
                context_root=cfg.context_root,
                symbol=sym,
                context_timeframe=cfg.context_timeframe,
                target_timeframe=cfg.timeframe,
                target_freq_minutes=Timeframe.from_str(cfg.timeframe).minutes,
            )
            out[sym] = merged
            ctx_meta = merged.attrs.get("context_merge", {})
            info["merged"][sym] = ctx_meta
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("context merge failed for %s: %s", sym, exc)
            out[sym] = df
            info["skipped"].append({"symbol": sym, "reason": f"merge_failed: {exc}"})
    return out, info


# ---------------------------------------------------------------------------
# Stage 8 — temporal splits
# ---------------------------------------------------------------------------

def _temporal_split_indices(
    n: int,
    train_frac: float,
    val_frac: float,
    embargo: int,
) -> tuple[int, int, int]:
    """Return (train_end, val_end, test_end) indices with embargo gaps.

    The embargo drops ``embargo`` bars between train/val and val/test
    to prevent label leakage through rolling window features.
    """
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    usable = n - 2 * embargo
    if usable <= 0:
        raise ValueError(f"not enough bars ({n}) for embargo {embargo}")
    train_end = int(usable * train_frac)
    val_end = train_end + int(usable * val_frac)
    # Adjust for embargo.
    train_end += 0
    val_end_with_embargo = train_end + embargo + int(usable * val_frac)
    test_end_with_embargo = val_end_with_embargo + embargo + (usable - train_end - int(usable * val_frac))
    train_end += 0  # keep marker
    return train_end, val_end_with_embargo, test_end_with_embargo


def _split_combined(
    per_symbol: dict[str, pd.DataFrame],
    cfg: PrepareConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Combine per-symbol frames into temporal train/val/test frames.

    Each row keeps its original timestamp and gains a ``symbol`` column.
    The split is computed **per symbol** (so every symbol has its own
    70/15/15 split), then concatenated across symbols. This keeps the
    distribution balanced across splits.
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    split_meta: dict[str, dict] = {}

    for sym, df in per_symbol.items():
        n = len(df)
        train_end, val_end, test_end = _temporal_split_indices(
            n, cfg.train_frac, cfg.val_frac, cfg.embargo_bars
        )
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end + cfg.embargo_bars : val_end].copy()
        test = df.iloc[val_end + cfg.embargo_bars : test_end].copy()
        train["symbol"] = sym
        val["symbol"] = sym
        test["symbol"] = sym
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
        split_meta[sym] = {
            "total_bars": n,
            "train_bars": len(train),
            "val_bars": len(val),
            "test_bars": len(test),
            "train_end_ts": str(df.index[train_end - 1]) if train_end > 0 else None,
            "val_start_ts": str(df.index[train_end + cfg.embargo_bars]) if train_end + cfg.embargo_bars < n else None,
            "val_end_ts": str(df.index[val_end - 1]) if val_end > 0 else None,
            "test_start_ts": str(df.index[val_end + cfg.embargo_bars]) if val_end + cfg.embargo_bars < n else None,
            "test_end_ts": str(df.index[test_end - 1]) if test_end > 0 else None,
        }

    train_df = pd.concat(train_parts).sort_index() if train_parts else pd.DataFrame()
    val_df = pd.concat(val_parts).sort_index() if val_parts else pd.DataFrame()
    test_df = pd.concat(test_parts).sort_index() if test_parts else pd.DataFrame()
    return train_df, val_df, test_df, split_meta


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def prepare_dataset(cfg: PrepareConfig) -> PreparedDataset:
    """Run the full preparation pipeline and write outputs to disk.

    Returns the populated :class:`PreparedDataset` manifest. The same
    manifest is also written to ``{out_root}/manifest.json``.
    """
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    symbols_dir = cfg.out_root / "symbols"
    splits_dir = cfg.out_root / "splits"
    symbols_dir.mkdir(exist_ok=True)
    splits_dir.mkdir(exist_ok=True)

    tsdb = TimeSeriesDB(cfg.tsdb_root)
    log: dict = {"stages": {}}

    # 1+2. Load + repair
    repaired: dict[str, pd.DataFrame] = {}
    raw_source_checksums: dict[str, str] = {}
    repair_log: dict[str, dict] = {}
    for sym in cfg.symbols:
        raw = _load_symbol(tsdb, sym, cfg.timeframe)
        raw_source_checksums[sym] = PreparedDataset.checksum_frame(raw)
        clean, audit = _repair(raw, where=sym, timeframe=cfg.timeframe)
        repaired[sym] = clean
        repair_log[sym] = audit
    log["stages"]["load_repair"] = repair_log

    # 3. Gap policy
    gap_filled: dict[str, pd.DataFrame] = {}
    gap_log: dict[str, dict] = {}
    for sym, df in repaired.items():
        df, info = _apply_gap_policy(df, cfg.timeframe, cfg.gap_policy)
        gap_filled[sym] = df
        gap_log[sym] = info
    log["stages"]["gap_policy"] = gap_log

    # 4. Coverage alignment
    aligned, align_info = _align_coverage(gap_filled, cfg.coverage_policy)
    log["stages"]["coverage_alignment"] = align_info
    if not aligned:
        raise RuntimeError(
            "coverage alignment removed every symbol; check CoveragePolicy.min_bars "
            "and the symbols you passed."
        )
    # Hash the fixed, aligned input window. Appending newer TSDB rows after a
    # configured coverage cutoff must not change this dataset's identity.
    input_checksums = {
        sym: PreparedDataset.checksum_frame(df) for sym, df in aligned.items()
    }

    # 5. Context merge
    with_ctx, ctx_info = _merge_context(aligned, cfg)
    log["stages"]["context_merge"] = ctx_info

    # 6. Schema assert
    for sym, df in with_ctx.items():
        # We only assert on OHLCV columns — context columns may have
        # legitimate NaNs (e.g. before Binance started publishing).
        assert_v1_schema(df[list(OHLCV_COLUMNS)], where=sym)

    # 7+8. Write per-symbol parquets, splits, manifest
    rows_per_symbol: dict[str, int] = {}
    output_checksums: dict[str, str] = {}
    for sym, df in with_ctx.items():
        out_path = symbols_dir / f"{sym.replace('/', '_')}.parquet"
        # Preserve attrs via parquet metadata (best-effort).
        df.to_parquet(out_path, engine="pyarrow", index=True)
        rows_per_symbol[sym] = int(len(df))
        output_checksums[sym] = PreparedDataset.checksum_frame(df)

    train_df, val_df, test_df, split_meta = _split_combined(with_ctx, cfg)
    train_df.to_parquet(splits_dir / "train.parquet", engine="pyarrow", index=True)
    val_df.to_parquet(splits_dir / "val.parquet", engine="pyarrow", index=True)
    test_df.to_parquet(splits_dir / "test.parquet", engine="pyarrow", index=True)

    # Compute feature column list (union across symbols).
    feature_cols: list[str] = []
    for df in with_ctx.values():
        for col in df.columns:
            if col not in feature_cols:
                feature_cols.append(col)
    feature_cols = sorted(feature_cols)

    # Determine the final window from the splits themselves.
    final_start = min((df.index.min() for df in with_ctx.values()), default=None)
    final_end = max((df.index.max() for df in with_ctx.values()), default=None)

    manifest = PreparedDataset(
        version=cfg.version,
        symbols=list(with_ctx.keys()),
        timeframe=cfg.timeframe,
        rows_total=int(sum(rows_per_symbol.values())),
        rows_per_symbol=rows_per_symbol,
        gap_policy=cfg.gap_policy,
        coverage_policy=cfg.coverage_policy,
        start=str(final_start) if final_start is not None else "",
        end=str(final_end) if final_end is not None else "",
        feature_columns=feature_cols,
        input_checksums=input_checksums,
        output_checksums=output_checksums,
        output_checksum="",  # filled below
    )
    manifest.output_checksum = PreparedDataset.checksum_manifest(manifest.to_dict())
    manifest.to_json(cfg.out_root / "manifest.json")

    # Human-readable checksums summary.
    with open(cfg.out_root / "checksums.txt", "w", encoding="utf-8") as f:
        f.write(f"manifest  {manifest.output_checksum}\n")
        for sym, ck in output_checksums.items():
            f.write(f"symbol    {sym:<12}  {ck}\n")
        for sym, ck in input_checksums.items():
            f.write(f"input     {sym:<12}  {ck}\n")

    # Full preparation log (so you can audit any run later).
    log["manifest"] = manifest.to_dict()
    log["splits"] = split_meta
    log["rows_per_symbol"] = rows_per_symbol
    log["raw_source_checksums"] = raw_source_checksums
    log["feature_columns"] = feature_cols
    with open(cfg.out_root / "preparation_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    logger.info(
        "prepared dataset: %d symbols, %d rows total, manifest=%s",
        len(with_ctx), manifest.rows_total, manifest.output_checksum,
    )
    return manifest


# ---------------------------------------------------------------------------
# Helpers for downstream consumers
# ---------------------------------------------------------------------------

def load_prepared_symbol(out_root: Path, symbol: str) -> pd.DataFrame:
    """Load a single prepared symbol frame."""
    p = Path(out_root) / "symbols" / f"{symbol.replace('/', '_')}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def load_prepared_split(out_root: Path, split: str) -> pd.DataFrame:
    """Load one of the temporal splits (``"train" | "val" | "test"``)."""
    p = Path(out_root) / "splits" / f"{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)
