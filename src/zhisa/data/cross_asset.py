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


def build_market_index(logrets: dict[str, pd.Series], *, min_periods: int = 2, min_frac: Optional[float] = None) -> pd.Series:
    """Equal-weight mean of log-returns across the GIVEN symbols.

    The caller excludes the symbol being enriched so the index always
    represents *the rest of the market* (self-free breadth). ``min_periods``
    keeps a single missing log-return from pinning the index to NaN;
    ``min_frac`` (when given) requires at least that fraction of the
    reference universe to be non-missing, so the index composition cannot
    silently drift to a tiny subset of symbols.
    """
    if not logrets:
        raise ValueError("market index requires at least one reference series")
    frame = pd.concat({k: v for k, v in logrets.items()}, axis=1)
    count = frame.notna().sum(axis=1)
    # required can never exceed the actual reference universe: with N=2 the
    # "others" list has ONE symbol, so a default min_periods=2 would NaN out
    # the whole index forever.
    required = min(int(min_periods), max(1, len(logrets)))
    if min_frac is not None:
        required = max(required, int(np.ceil(min_frac * len(logrets))))
    raw = frame.mean(axis=1, skipna=True)
    index = raw.where(count >= required)
    index.name = "market_index_logret"
    return index


def build_index_volume(volumes: dict[str, pd.Series], *, min_periods: int = 2) -> pd.Series:
    """Equal-weight mean volume of the given symbols (self-free breadth volume)."""
    if not volumes:
        raise ValueError("index volume requires at least one reference series")
    frame = pd.concat({k: v for k, v in volumes.items()}, axis=1)
    count = frame.notna().sum(axis=1)
    raw = frame.mean(axis=1, skipna=True)
    required = min(int(min_periods), max(1, len(volumes)))  # never > ref count
    index = raw.where(count >= required)
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
    ref_logrets_wide: Optional[pd.DataFrame] = None,
    with_breadth: bool = False,
    with_regime_betas: bool = False,
    stress_z: float = 1.0,
    max_beta_clip: float = 6.0,
    with_resid_alpha: bool = False,
    with_vol_index: bool = False,
    lead_lag_lags: tuple = (),
) -> pd.DataFrame:
    """Return ``frame`` + causal cross-asset columns vs ``ref_logret``.

    ``ref_logret`` must already be aligned to ``frame.index``. All statistics
    use *trailing* windows (current bar included), so nothing looks ahead.
    Volume-relative columns (``volume_ratio_w``, ``volvol_ratio_w``) are only
    added when ``with_volume_ratios`` AND ``ref_volume`` are provided; they
    measure the asset's volume and volume-volatility against the market.

    v3.1 additions (all causal, all deterministic):

    - ``breadth_w``        fraction of reference symbols moving in the
      direction of the index (sign agreement), when ``ref_logrets_wide``
      (one column per reference symbol) is provided and breadth enabled;
    - ``market_vol_w``     rolling std of the index log-returns
      (unconditional market volatility);
    - ``dispersion_256``   rolling mean of the cross-sectional std of the
      references' log-returns (how much the market is disagreeing);
    - ``corr_stress_w``    trailing correlation computed ONLY on bars whose
      |index log-return| exceeds ``stress_z`` sigma — proxies *crash
      comovement* without an explicit regime model;
    - ``beta_up_w/``       trailing beta conditioned on index-up / index-down
      bars (regime-conditional; heavier min-periods so the split stays
      numerically stable).
      ``beta_down_w``
    - ``beta_*``/``corr_*`` are winsorised to +/- ``max_beta_clip`` after a
      MAD-based centre so single-day blowups cannot dominate the input.
    """
    # memory-friendly: only new columns are allocated, OHLCV is shared
    for _lag in (lead_lag_lags or ()):
        if int(_lag) < 0:
            raise ValueError("lead_lag_lags must be >= 0 (causal only); "
                             "negative lags would leak the future")
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
        if with_regime_betas:
            # unconditional market volatility of the index
            out[f"market_vol_{w}"] = ref.rolling(w, min_periods=min_p).std()
            # regime-conditional beta: index-up vs index-down bars
            up = ref > 0
            min_p_up = max(int(w * 0.5 * min_periods_ratio), 3)
            cov_up = lr.where(up, np.nan).rolling(w, min_periods=min_p_up).cov(ref.where(up, np.nan))
            var_up = ref.where(up, np.nan).rolling(w, min_periods=min_p_up).var()
            cov_dn = lr.where(~up, np.nan).rolling(w, min_periods=min_p_up).cov(ref.where(~up, np.nan))
            var_dn = ref.where(~up, np.nan).rolling(w, min_periods=min_p_up).var()
            out[f"beta_up_{w}"] = cov_up / var_up.replace(0.0, np.nan)
            out[f"beta_down_{w}"] = cov_dn / var_dn.replace(0.0, np.nan)
            # crash-comovement: correlation on stress bars only. Stress bars
            # are |index logret| above max(z*rolling_std, rolling_q75) — the
            # quantile floor guarantees a non-empty stress set in ANY
            # distribution, so the stress correlation is always computable.
            abs_ref = ref.abs()
            thresh = np.maximum(
                ref.rolling(w, min_periods=min_p).std() * stress_z,
                abs_ref.rolling(w, min_periods=min_p).quantile(0.75),
            )
            stress = abs_ref > thresh
            min_p_st = max(int(w * 0.4 * min_periods_ratio), 3)
            corr_st = lr.where(stress, np.nan).rolling(w, min_periods=min_p_st).corr(
                ref.where(stress, np.nan)
            ).rename(f"corr_stress_{w}")
            out[f"corr_stress_{w}"] = corr_st

    if (with_breadth or with_regime_betas) and ref_logrets_wide is not None:
        wide = ref_logrets_wide.reindex(out.index)
        signs = np.sign(wide)
        ag = (signs == np.sign(ref.to_frame().to_numpy())).astype(float)
        for w in windows:
            w = int(w)
            min_p = max(int(w * min_periods_ratio), 3)
            out[f"breadth_{w}"] = ag.mean(axis=1).rolling(w, min_periods=min_p).mean()
        # cross-sectional dispersion (how much the market disagrees)
        cs = wide.std(axis=1)
        out["dispersion_256"] = cs.rolling(256, min_periods=64).mean()

    # Winsorise beta/corr tails (MAD-centred) so one blow-up bar cannot
    # dominate the numeric input: clip to +/- max_beta_clip.
    for w in windows:
        w = int(w)
        col = out[f"beta_{w}"]
        med = col.rolling(w, min_periods=3).median()
        mad = (col - med).abs().rolling(w, min_periods=3).median()
        lo = med - max_beta_clip * mad.replace(0.0, np.nan)
        hi = med + max_beta_clip * mad.replace(0.0, np.nan)
        out[f"beta_{w}"] = col.clip(lo, hi)

    # ---- v4 additions (all causal, all additive; v3/v3.1 contract intact) -
    if with_volume_ratios and ref_volume is not None and with_vol_index:
        # Volume-intensity-weighted market index log-return: the reference
        # (index) log-return is scaled by the asset-relative volume intensity
        # (volume / rolling-mean volume, mean ~ 1.0). NOTE: dividing by the
        # rolling SUM (~ mean*256) would shrink the market signal 256x and
        # collapse rel_logret_vw_1 into ~1*0 — scale must be preserved.
        refv = ref_volume.reindex(out.index)
        vol_intensity = refv / (refv.rolling(256, min_periods=4).mean() + 1e-12)
        ref_vw = (ref * vol_intensity.fillna(0.0))
        ref_vw = ref_vw.rolling(1, min_periods=1).sum()
        out["market_index_vw_logret"] = ref_vw
        out["rel_logret_vw_1"] = lr - ref_vw
        out["vw_weight_256"] = vol_intensity.rolling(256, min_periods=4).mean()
        for w in windows:
            w = int(w)
            min_p = max(int(w * min_periods_ratio), 3)
            cov_vw = lr.rolling(w, min_periods=min_p).cov(ref_vw)
            var_vw = ref_vw.rolling(w, min_periods=min_p).var()
            out[f"beta_vw_{w}"] = (cov_vw / var_vw.replace(0.0, np.nan))
            out[f"corr_vw_{w}"] = lr.rolling(w, min_periods=min_p).corr(ref_vw)

    if with_resid_alpha and ref_logrets_wide is not None:
        # Idiosyncratic residual alpha vs the (equal-weight, trailing-beta)
        # market factor: eps_t = R_asset - beta_w * R_market. beta_* columns
        # are computed above with the same windows -> reuse them.
        for w in windows:
            w = int(w)
            b = out.get(f"beta_{w}")
            if b is not None:
                out[f"resid_alpha_{w}"] = (lr - b * ref).rename(f"resid_alpha_{w}")

    if lead_lag_lags and ref_logrets_wide is not None:
        # Lead-lag cross-correlation of the ASSET vs the REFERENCE index with
        # causal lags ONLY (k >= 0): k>0 means the index moved BEFORE the
        # asset in window-time (pure past) — no lookahead. Negative k (asset
        # leads) would leak the future and is rejected at the loader.
        wide = ref_logrets_wide.reindex(out.index)
        lead_corr_by_ref: dict[str, pd.Series] = {}
        for c in wide.columns:
            s = wide[c]
            for k in lead_lag_lags:
                k = int(k)
                if k < 0:
                    raise ValueError("lead_lag_lags must be >= 0 (causal only)")
                shifted = s.shift(k) if k > 0 else s
                key = f"leadlag_{c}_{k}"
                shifted = shifted.reindex(out.index)
                lead_corr_by_ref[key] = lr.rolling(
                    64, min_periods=16
                ).corr(shifted)
        for k in lead_lag_lags:
            cols = [f"leadlag_{c}_{k}" for c in wide.columns]
            avg = pd.concat([lead_corr_by_ref[c] for c in cols], axis=1).mean(axis=1)
            out[f"leadlag_avg_{k}"] = avg
        for c in wide.columns:
            for k in lead_lag_lags:
                if f"leadlag_{c}_{k}" in lead_corr_by_ref:
                    out[f"leadlag_{c}_{k}"] = lead_corr_by_ref[f"leadlag_{c}_{k}"]

    return out


def enrich_market_frames_detailed(
    frames: dict[str, pd.DataFrame],
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_periods_ratio: float = 0.25,
    with_volume_ratios: bool = False,
    with_breadth: bool = False,
    with_regime_betas: bool = False,
    stress_z: float = 1.0,
    max_beta_clip: float = 6.0,
    min_coverage: float = 0.5,
    with_resid_alpha: bool = False,
    with_vol_index: bool = False,
    lead_lag_lags: tuple = (),
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Enrich every symbol against an equal-weight index of the OTHERS.

    Deterministic; returns ``(enriched_frames, audit)`` where
    ``audit[sym] = {"refs": [...], "index": Series, "na_frac": float,
    "mean_beta_256": float, "mean_coverage": float}`` for reproducibility
    and analysis. ``min_coverage`` keeps the index on at least that
    fraction of the reference universe at every bar.
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
            wide = None
            coverage = pd.Series(0.0, index=df.index)
        else:
            refs = others
            index = build_market_index({s: logrets[s] for s in others}, min_frac=min_coverage)
            ref = index.reindex(df.index)
            coverage = pd.concat(
                {s: logrets[s].reindex(df.index) for s in others}, axis=1
            ).notna().sum(axis=1) / len(others)
            wide = pd.concat({s: logrets[s].reindex(df.index) for s in others}, axis=1)
            refv = None
            if with_volume_ratios:
                ivol = build_index_volume({s: volumes[s] for s in others})
                refv = ivol.reindex(df.index)
        enriched[sym] = enrich_frame(
            df, ref, windows=windows, min_periods_ratio=min_periods_ratio,
            with_volume_ratios=with_volume_ratios, ref_volume=refv,
            ref_logrets_wide=wide, with_breadth=with_breadth,
            with_regime_betas=with_regime_betas, stress_z=stress_z,
            max_beta_clip=max_beta_clip,
            with_resid_alpha=with_resid_alpha,
            with_vol_index=with_vol_index,
            lead_lag_lags=lead_lag_lags,
        )
        beta = enriched[sym].get("beta_256")
        audit[sym] = {
            "refs": refs,
            "index": ref,
            "na_frac": float(enriched[sym].get("rel_logret_1", pd.Series(dtype=float)).isna().mean()),
            "mean_beta_256": float(beta.dropna().mean()) if beta is not None and beta.notna().any() else None,
            "mean_coverage": float(coverage.mean()),
        }
    return enriched, audit


def enrich_market_frames(
    frames: dict[str, pd.DataFrame],
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_periods_ratio: float = 0.25,
    with_volume_ratios: bool = False,
    with_breadth: bool = False,
    with_regime_betas: bool = False,
    stress_z: float = 1.0,
    max_beta_clip: float = 6.0,
    min_coverage: float = 0.5,
    with_resid_alpha: bool = False,
    with_vol_index: bool = False,
    lead_lag_lags: tuple = (),
) -> dict[str, pd.DataFrame]:
    """Compatibility wrapper: just returns the enriched frames."""
    enriched, _ = enrich_market_frames_detailed(
        frames, windows=windows, min_periods_ratio=min_periods_ratio,
        with_volume_ratios=with_volume_ratios, with_breadth=with_breadth,
        with_regime_betas=with_regime_betas, stress_z=stress_z,
        max_beta_clip=max_beta_clip, min_coverage=min_coverage,
        with_resid_alpha=with_resid_alpha,
        with_vol_index=with_vol_index,
        lead_lag_lags=lead_lag_lags,
    )
    return enriched