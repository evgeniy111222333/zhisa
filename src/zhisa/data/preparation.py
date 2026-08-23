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
    # When True, a prepared run FAILS if any symbol has no context parquet —
    # otherwise that symbol would silently end up with a different numeric
    # feature width (no ctx_* columns) and break the dataset's schema.
    require_all_context: bool = False
    # Global sentiment channel (alternative.me Fear & Greed), injected into
    # EVERY symbol as ``ctx_fng_index`` (same value, deterministic, 1-bar lag).
    with_fear_greed: bool = False
    fear_greed_cache: Optional[Path] = None
    # Intraday order-book depth (Binance bookDepth -> 1h aggregates). Joined
    # per symbol when EVERY symbol has a parquet (uniform schema; otherwise the
    # stage is skipped to avoid a per-symbol feature-width difference).
    with_bookdepth: bool = False
    bookdepth_root: Optional[Path] = None
    gap_policy: GapPolicy = None  # type: ignore[assignment]
    coverage_policy: CoveragePolicy = None  # type: ignore[assignment]
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    embargo_bars: int = 96  # 96 * 15m = 24h
    version: str = CURRENT_VERSION
    seed: int = 0
    with_cross_asset: bool = False
    cross_asset_windows: tuple = (64, 256)
    with_volume_ratios: bool = False
    cross_asset_breadth: bool = True
    cross_asset_regime_betas: bool = False
    cross_asset_stress_z: float = 2.0
    cross_asset_max_beta_clip: float = 6.0
    cross_asset_min_coverage: float = 0.5
    # v4 add-ons (additive, opt-in, contract-preserving)
    cross_asset_resid_alpha: bool = False
    cross_asset_vol_index: bool = False
    cross_asset_lead_lag_lags: tuple = ()

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
    # Keep OHLCV + any extra numeric columns (micro_* microstructure channels)
    # so they reach the prepared root (a clean OHLCV-only slice silently
    # dropped them before this fix).
    df = df.astype(float)
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
    if user_start is not None and user_start.tzinfo is None:
        user_start = user_start.tz_localize("UTC")
    if user_end is not None and user_end.tzinfo is None:
        user_end = user_end.tz_localize("UTC")

    # An explicit start/end is a hard bound on the OUTPUT window (each symbol
    # is still clipped to its own data below). The default ``None`` keeps the
    # shared auto-window. This lets a run keep DEEP history for old symbols
    # (e.g. ``--coverage-start 2020-10-01``) while late-listed symbols simply
    # contribute from their own listing date instead of dragging everyone to a
    # shallow union window.
    final_start = user_start if user_start is not None else auto_start
    final_end = user_end if user_end is not None else auto_end
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
    if (
        cfg.with_futures_context
        and cfg.context_root is not None
        and cfg.require_all_context
    ):
        missing = [
            sym for sym, df in out.items()
            if not ((df.attrs.get("context_merge", {}) or {}).get("columns_added"))
        ]
        if missing:
            raise ValueError(
                "require_all_context: symbol(s) have NO futures-context columns: "
                f"{sorted(missing)}. Add context parquets first (ensure_futures_context)"
            )
    return out, info


def _join_bookdepth(
    per_symbol: dict[str, pd.DataFrame],
    cfg: PrepareConfig,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Join per-1h order-book depth channels into every symbol's frame.

    Schema rule: the ``bd_*`` columns are only added if EVERY symbol has a depth
    parquet — otherwise the stage is skipped entirely so no symbol ends up with
    a different numeric width.
    """
    if not cfg.with_bookdepth:
        return per_symbol, {"enabled": False}
    from zhisa.scripts._real_data import futures_context_symbol_slug
    root = Path(cfg.bookdepth_root) if cfg.bookdepth_root else Path("data/bookdepth/1h")
    info: dict = {"symbols": [], "missing": []}
    joined: dict[str, pd.DataFrame] = {}
    for sym, df in per_symbol.items():
        p = root / f"{futures_context_symbol_slug(sym)}.parquet"
        if not p.is_file():
            info["missing"].append(sym)
            joined[sym] = df
            continue
        bd = pd.read_parquet(p)
        bd_cols = [c for c in bd.columns if c.startswith("bd_")]
        joined[sym] = df.join(bd[bd_cols], how="left")
        info["symbols"].append(sym)
    if info["symbols"] and not info["missing"]:
        per_symbol = joined
        info["dropped"] = False
    else:
        # Uniform schema: partial coverage would create asymmetric feature widths.
        if info["symbols"]:
            logger.warning(
                "bookdepth present for %d/%d symbols -> dropping bd_* for uniform schema",
                len(info["symbols"]), len(per_symbol),
            )
            for sym, df in joined.items():
                joined[sym] = df.drop(columns=[c for c in df.columns if c.startswith("bd_")])
            per_symbol = joined
            info["dropped"] = True
        info["symbols"] = []
    return per_symbol, info


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
    if cfg.with_cross_asset:
        from zhisa.data.cross_asset import enrich_market_frames_detailed
        aligned, cross_audit = enrich_market_frames_detailed(
            aligned, windows=cfg.cross_asset_windows,
            with_volume_ratios=cfg.with_volume_ratios,
            with_breadth=cfg.cross_asset_breadth,
            with_regime_betas=cfg.cross_asset_regime_betas,
            stress_z=cfg.cross_asset_stress_z,
            max_beta_clip=cfg.cross_asset_max_beta_clip,
            min_coverage=cfg.cross_asset_min_coverage,
            with_resid_alpha=cfg.cross_asset_resid_alpha,
            with_vol_index=cfg.cross_asset_vol_index,
            lead_lag_lags=cfg.cross_asset_lead_lag_lags,
        )
        audit_dir = cfg.out_root / "audit" / "cross_asset_index"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_summary = {}
        for sym, info in cross_audit.items():
            safe = sym.replace("/", "_")
            info["index"].to_frame().to_parquet(audit_dir / f"{safe}.parquet")
            audit_summary[sym] = {k: v for k, v in info.items() if k != "index"}
        log["stages"]["cross_asset_enrich"] = {
            "windows": list(cfg.cross_asset_windows),
            "with_volume_ratios": bool(cfg.with_volume_ratios),
            "with_regime_betas": bool(cfg.cross_asset_regime_betas),
            "columns_added": [f"rel_logret_1",
                              *(f"beta_{w}" for w in cfg.cross_asset_windows),
                              *(f"corr_{w}" for w in cfg.cross_asset_windows),
                              *(f"market_vol_{w}" for w in cfg.cross_asset_windows),
                              *(f"breadth_{w}" for w in cfg.cross_asset_windows if cfg.cross_asset_breadth),
                              *(f"beta_{d}_{w}" for d in ("up", "down") for w in cfg.cross_asset_windows if cfg.cross_asset_regime_betas),
                              *(f"corr_stress_{w}" for w in cfg.cross_asset_windows if cfg.cross_asset_regime_betas)],
            "per_symbol": audit_summary,
        }
    with_ctx, ctx_info = _merge_context(aligned, cfg)
    log["stages"]["context_merge"] = ctx_info

    # 5b. Global sentiment channel (Fear & Greed) injected into every symbol.
    if cfg.with_fear_greed:
        from zhisa.data.fear_greed import fear_greed_column, load_fear_greed
        fng = load_fear_greed(cfg.fear_greed_cache
                              if cfg.fear_greed_cache else "data/fear_greed/fear_greed.parquet")
        fng_info = {"n_days": int(len(fng))}
        for sym, df in with_ctx.items():
            with_ctx[sym] = df.assign(ctx_fng_index=fear_greed_column(df.index, fng))
        log["stages"]["fear_greed"] = fng_info

    # 5c. Intraday order-book depth (bookDepth per-1h aggregates), uniform across symbols.
    if cfg.with_bookdepth:
        with_ctx, bd_info = _join_bookdepth(with_ctx, cfg)
        log["stages"]["bookdepth"] = bd_info

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

    # Lineage guard: record the scan so later reads can SCREAM on drift.
    from zhisa.data.lineage import scan_prepared, write_lineage
    try:
        scan = scan_prepared(
            cfg.out_root,
            tsdb_root=cfg.tsdb_root,
            symbols=list(with_ctx.keys()),
        )
        write_lineage(cfg.out_root, scan)
    except Exception as exc:  # defensive: lineage is an audit trail, not a blocker
        log["lineage_warning"] = f"lineage scan failed: {exc!r}"

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


def enrich_prepared_root(
    base_root,
    out_root,
    *,
    symbols=None,
    windows=(64, 256),
    with_volume_ratios=False,
    with_breadth=False,
    with_regime_betas=False,
    stress_z=1.0,
    max_beta_clip=6.0,
    min_coverage=0.5,
    with_resid_alpha=False,
    with_vol_index=False,
    lead_lag_lags=(),
) -> dict:
    """Build an enriched dataset ON TOP of an existing prepared root.

    The cardinal rule: OHLCV rows are copied BYTE-IDENTICAL (no repair, no
    reindex, no gap fill) and only new columns (cross-asset / regime /
    breadth) are added. This is what makes the chart-store of the base root
    reusable for the enriched root — the chart content hash only depends on
    OHLCV.

    Raises ``ValueError`` if any symbol's OHLCV differs from the base root.
    """
    base_root = Path(base_root)
    out_root = Path(out_root)
    bm_path = base_root / "manifest.json"
    if not bm_path.is_file():
        raise ValueError(f"base root has no manifest: {bm_path}")
    base_manifest = json.loads(bm_path.read_text(encoding="utf-8"))
    bsymbols = list(base_manifest.get("rows_per_symbol", {}).keys())
    syms = symbols or bsymbols
    missing = [s for s in syms if not (base_root / "symbols" / f"{s}.parquet").is_file()]
    if missing:
        raise ValueError(f"base root missing symbols: {missing}")

    frames = {}
    for sym in syms:
        df = pd.read_parquet(base_root / "symbols" / f"{sym}.parquet").sort_index()
        frames[sym] = df

    from zhisa.data.cross_asset import enrich_market_frames_detailed

    enriched, audit = enrich_market_frames_detailed(
        frames, windows=windows, with_volume_ratios=with_volume_ratios,
        with_breadth=with_breadth, with_regime_betas=with_regime_betas,
        stress_z=stress_z, max_beta_clip=max_beta_clip, min_coverage=min_coverage,
        with_resid_alpha=with_resid_alpha,
        with_vol_index=with_vol_index,
        lead_lag_lags=lead_lag_lags,
    )
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    per_sym_rows = {}
    feature_cols = list(base_manifest.get("feature_columns", []))
    for sym in syms:
        base = frames[sym][ohlcv_cols].to_numpy(dtype=np.float64)
        new = enriched[sym][ohlcv_cols].to_numpy(dtype=np.float64)
        if base.shape != new.shape or not np.array_equal(base, new):
            raise ValueError(
                f"enrich-from violated OHLCV byte-identity for {sym}: shape {base.shape} vs {new.shape}"
            )
        per_sym_rows[sym] = int(len(enriched[sym]))
        feature_cols = list(enriched[sym].columns)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "symbols").mkdir(parents=True, exist_ok=True)
    for sym in syms:
        enriched[sym].to_parquet(out_root / "symbols" / f"{sym}.parquet")

    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    split_meta = {}
    for split in ("train", "val", "test"):
        src = base_root / "splits" / f"{split}.parquet"
        if not src.is_file():
            continue
        sdf = pd.read_parquet(src)
        if "symbol" not in sdf.columns:
            raise ValueError(f"base split {split} has no 'symbol' column")
        parts = []
        for sym, g in sdf.groupby("symbol", sort=False):
            g_idx = pd.DatetimeIndex(g.index)
            merged = enriched[sym].reindex(g_idx).copy()
            merged["symbol"] = sym
            parts.append(merged)
        rebuilt = pd.concat(parts) if parts else sdf
        rebuilt.to_parquet(out_root / "splits" / f"{split}.parquet")
        split_meta[split] = int(len(rebuilt))

    out_manifest = dict(base_manifest)
    out_manifest["rows_per_symbol"] = per_sym_rows
    out_manifest["rows_total"] = int(sum(per_sym_rows.values()))
    out_manifest["feature_columns"] = sorted(feature_cols)
    out_manifest["derived_from"] = {
        "base_root": str(base_root),
        "base_output_checksum": base_manifest.get("output_checksum"),
        "ohclv_byte_identical": True,
        "enrichment": {
            "with_volume_ratios": bool(with_volume_ratios),
            "with_breadth": bool(with_breadth),
            "with_regime_betas": bool(with_regime_betas),
            "stress_z": float(stress_z),
            "max_beta_clip": float(max_beta_clip),
            "min_coverage": float(min_coverage),
        },
    }
    out_manifest["output_checksum"] = PreparedDataset.checksum_manifest(out_manifest)
    (out_root / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")

    with open(out_root / "checksums.txt", "w", encoding="utf-8") as f:
        f.write(f"manifest  {out_manifest['output_checksum']}\n")
        f.write(f"base      {out_manifest['derived_from']['base_root']}  "
                f"{base_manifest.get('output_checksum')}\n")

    from zhisa.data.lineage import scan_prepared, write_lineage

    scan = scan_prepared(out_root, tsdb_root=None, symbols=syms)
    write_lineage(out_root, scan)

    return {
        "out_root": str(out_root),
        "rows_total": out_manifest["rows_total"],
        "output_checksum": out_manifest["output_checksum"],
        "base_root": str(base_root),
        "base_checksum": out_manifest["derived_from"]["base_output_checksum"],
        "splits": split_meta,
    }
