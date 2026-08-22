"""Cross-asset / market-breadth enrichment for prepared symbol frames (v3 data).

Rationale (measured): assets share a strong common factor (1h logret corr with
BTC: ETH 0.84, SOL 0.65, BNB 0.70, TRX 0.54), but each symbol frame today is
standalone + instrument-id. Adding *market-relative* features gives every
symbol explicit breadth context.

Ideal contract:

- **market index** = equal-weight mean of the *other* symbols' log-returns
  (self excluded), so every symbol gets breadth from the rest, deterministically;
- **all new columns are strictly causal** (trailing-rolling, closed bars only);
- missing early history -> NaN (downstream feature pipeline zero-fills);
- deterministic and reproducible (same inputs -> same enrichment), so the
  prepared-checksum contract stays intact;
- simple, few columns (no look-ahead transformations, no lead-lag — measured
  as ~0 at 1h and therefore excluded).

Columns added per symbol:

    rel_logret_1      logret(asset) - logret(index)
    beta_64 / beta_256  trailing Cov(asset,index)/Var(index)
    corr_64 / corr_256  trailing Pearson correlation asset<->index
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (64, 256)


def symbol_logret(frame: pd.DataFrame) -> pd.Series:
    """Closed-bar log-return of ``close`` (causal: only past closes)."""
    return np.log(frame["close"]).diff()


def build_market_index(logrets: dict[str, pd.Series], *, min_periods: int = 2) -> pd.Series:
    """Equal-weight mean of log-returns across the GIVEN symbols.

    The caller excludes the symbol being enriched so the index always
    represents *the rest of the market* (self-free breadth). ``min_periods``
    keeps a single missing log-return from pinning the index to NaN.
    """
    if not logrets:
        raise ValueError("market index requires at least one reference series")
    frame = pd.concat({k: v for k, v in logrets.items()}, axis=1)
    count = frame.notna().sum(axis=1)
    raw = frame.mean(axis=1, skipna=True)
    index = raw.where(count >= min_periods)
    index.name = "market_index_logret"
    return index


def build_index_volume(volumes: dict[str, pd.Series], *, min_periods: int = 2) -> pd.Series:
    """Equal-weight mean volume of the given symbols (self-free breadth volume)."""
    if not volumes:
        raise ValueError("index volume requires at least one reference series")
    frame = pd.concat({k: v for k, v in volumes.items()}, axis=1)
    count = frame.notna().sum(axis=1)
    raw = frame.mean(axis=1, skipna=True)
    index = raw.where(count >= min_periods)
    index.name = "market_index_volume"
    return index


def enrich_frame(
    frame: pd.DataFrame,
    ref_logret: pd.Series,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_periods_ratio: float = 0.25,
    with_volume_ratios: bool = False,
    ref_volume: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return ``frame`` + causal cross-asset columns vs ``ref_logret``.

    ``ref_logret`` must already be aligned to ``frame.index``. All statistics
    use *trailing* windows (current bar included), so nothing looks ahead.
    Volume-relative columns (``volume_ratio_w``, ``volvol_ratio_w``) are only
    added when ``with_volume_ratios`` AND ``ref_volume`` are provided; they
    measure the asset's volume and volume-volatility against the market.
    """
    # memory-friendly: only new columns are allocated, OHLCV is shared
    out = frame.copy(deep=False)
    ref = ref_logret.reindex(out.index)
    lr = symbol_logret(out)

    rel = (lr - ref).rename("rel_logret_1")
    out["rel_logret_1"] = rel

    if with_volume_ratios and ref_volume is not None:
        refv = ref_volume.reindex(out.index)
        vol = out["volume"].astype("float64")
        eps = float(vol.max() * 1e-9) + 1e-12
        for w in windows:
            w = int(w)
            min_p = max(int(w * min_periods_ratio), 3)
            index_vol = refv.rolling(w, min_periods=min_p).mean()
            out[f"volume_ratio_{w}"] = np.log1p(vol / (index_vol + eps)).replace([np.inf, -np.inf], np.nan)
            sv = vol.rolling(w, min_periods=min_p).std()
            siv = refv.rolling(w, min_periods=min_p).std()
            out[f"volvol_ratio_{w}"] = np.log1p(sv / (siv + eps)).replace([np.inf, -np.inf], np.nan)

    for w in windows:
        w = int(w)
        min_p = max(int(w * min_periods_ratio), 3)
        # market beta / correlation of the ASSET's own returns vs the index
        cov = lr.rolling(w, min_periods=min_p).cov(ref)
        var = ref.rolling(w, min_periods=min_p).var()
        beta = (cov / var).rename(f"beta_{w}")
        corr = lr.rolling(w, min_periods=min_p).corr(ref).rename(f"corr_{w}")
        out[f"beta_{w}"] = beta
        out[f"corr_{w}"] = corr

    return out


def enrich_market_frames_detailed(
    frames: dict[str, pd.DataFrame],
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_periods_ratio: float = 0.25,
    with_volume_ratios: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Enrich every symbol against an equal-weight index of the OTHERS.

    Deterministic; returns ``(enriched_frames, audit)`` where
    ``audit[sym] = {"refs": [...], "index": Series, "na_frac": float,
    "mean_beta_256": float}`` for reproducibility and analysis.
    """
    nanfree: dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
        nanfree[sym] = df

    logrets = {sym: symbol_logret(df) for sym, df in nanfree.items()}
    volumes = {sym: df["volume"].astype("float64") for sym, df in nanfree.items()}

    enriched: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict] = {}
    for sym, df in nanfree.items():
        others = [s for s in logrets if s != sym]
        if not others:
            # single-symbol universe: same schema, all cross-asset columns NaN.
            ref = pd.Series(np.nan, index=df.index, name="market_index_logret")
            refv = pd.Series(np.nan, index=df.index, name="market_index_volume") if with_volume_ratios else None
            refs = []
        else:
            refs = others
            index = build_market_index({s: logrets[s] for s in others})
            ref = index.reindex(df.index)
            refv = None
            if with_volume_ratios:
                ivol = build_index_volume({s: volumes[s] for s in others})
                refv = ivol.reindex(df.index)
        enriched[sym] = enrich_frame(
            df, ref, windows=windows, min_periods_ratio=min_periods_ratio,
            with_volume_ratios=with_volume_ratios, ref_volume=refv,
        )
        beta = enriched[sym].get("beta_256")
        audit[sym] = {
            "refs": refs,
            "index": ref,
            "na_frac": float(enriched[sym].get("rel_logret_1", pd.Series(dtype=float)).isna().mean()),
            "mean_beta_256": float(beta.dropna().mean()) if beta is not None and beta.notna().any() else None,
        }
    return enriched, audit


def enrich_market_frames(
    frames: dict[str, pd.DataFrame],
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_periods_ratio: float = 0.25,
    with_volume_ratios: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compatibility wrapper: just returns the enriched frames."""
    enriched, _ = enrich_market_frames_detailed(
        frames, windows=windows, min_periods_ratio=min_periods_ratio,
        with_volume_ratios=with_volume_ratios,
    )
    return enriched