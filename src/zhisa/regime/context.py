"""Crowding, orderflow, and cross-asset context for regime intelligence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketContextConfig:
    lookback: int = 96
    short_lookback: int = 12
    z_window: int = 96
    lead_lag_bars: int = 3
    benchmark_symbol: str = "BTC/USDT"
    benchmark_aliases: tuple[str, ...] = ("BTC", "BTCUSDT", "BTC/USDT")
    high_funding_abs: float = 0.0005
    high_long_short_ratio: float = 1.35
    low_long_short_ratio: float = 0.75
    oi_change_threshold: float = 0.08
    liquidation_z_threshold: float = 2.5
    high_correlation_threshold: float = 0.65
    fragmented_correlation_threshold: float = 0.35
    lead_score_threshold: float = 0.12


@dataclass(frozen=True)
class CrowdingState:
    funding: float = 0.0
    funding_z: float = 0.0
    open_interest_change: float = 0.0
    long_short_ratio: float = 1.0
    long_short_z: float = 0.0
    liquidation_z: float = 0.0
    crowding_score: float = 0.0
    direction: str = "neutral"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrelationState:
    regime: str = "single_asset"
    avg_correlation: float = 0.0
    leader_lead_score: float = 0.0
    btc_lead_score: float = 0.0
    market_breadth: float = 0.0
    dispersion: float = 0.0
    leader_symbol: str = ""
    benchmark_symbol: str = ""
    n_assets: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketContextReport:
    crowding: CrowdingState = field(default_factory=CrowdingState)
    correlation: CorrelationState = field(default_factory=CorrelationState)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _finite(x: object, default: float = 0.0) -> float:
    try:
        out = float(x)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _first_existing(df: pd.DataFrame, names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _zscore(series: pd.Series, value: float, window: int) -> float:
    hist = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna().iloc[-window:]
    if hist.size < 3:
        return 0.0
    std = float(hist.std(ddof=0))
    if std <= 1e-12 or not np.isfinite(std):
        return 0.0
    return float((value - float(hist.mean())) / std)


def _safe_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float).replace(0, np.nan)
    return np.log(close).diff().replace([np.inf, -np.inf], np.nan).dropna()


class MarketContextAnalyzer:
    """Analyze derivatives crowding and cross-asset correlation causally."""

    def __init__(self, cfg: Optional[MarketContextConfig] = None) -> None:
        self.cfg = cfg or MarketContextConfig()

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        t: Optional[int] = None,
        symbol: str = "",
        assets: Optional[Mapping[str, pd.DataFrame]] = None,
        benchmark_symbol: str | None = None,
        btc_symbol: str | None = None,
    ) -> MarketContextReport:
        if t is not None:
            if t < 0:
                raise ValueError("t must be non-negative")
            work = df.iloc[: t + 1].copy()
        else:
            work = df.copy()
        if work.empty:
            raise ValueError("df slice is empty")

        warnings: list[str] = []
        crowding = self._crowding(work)
        warnings.extend(self._crowding_warnings(crowding))
        correlation = self._correlation(
            work,
            symbol=symbol,
            assets=assets,
            benchmark_symbol=benchmark_symbol or btc_symbol or self.cfg.benchmark_symbol,
            end_time=work.index[-1] if isinstance(work.index, pd.DatetimeIndex) else None,
        )
        warnings.extend(self._correlation_warnings(correlation))
        return MarketContextReport(crowding=crowding, correlation=correlation, warnings=warnings)

    def _crowding(self, df: pd.DataFrame) -> CrowdingState:
        cfg = self.cfg
        funding_col = _first_existing(df, ("funding", "funding_rate", "fundingRate"))
        oi_col = _first_existing(df, ("open_interest", "oi", "openInterest"))
        ls_col = _first_existing(df, ("long_short_ratio", "longShortRatio", "ls_ratio"))
        liq_cols = [
            c for c in (
                "liquidation_volume", "liquidations", "liq_volume",
                "long_liquidations", "short_liquidations",
            ) if c in df.columns
        ]

        funding = _finite(df[funding_col].iloc[-1]) if funding_col else 0.0
        funding_z = _zscore(df[funding_col], funding, cfg.z_window) if funding_col else 0.0

        oi_change = 0.0
        if oi_col:
            oi = df[oi_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if oi.size > cfg.short_lookback and float(oi.iloc[-cfg.short_lookback - 1]) > 0:
                oi_change = float(oi.iloc[-1] / oi.iloc[-cfg.short_lookback - 1] - 1.0)

        long_short = _finite(df[ls_col].iloc[-1], default=1.0) if ls_col else 1.0
        long_short_z = _zscore(df[ls_col], long_short, cfg.z_window) if ls_col else 0.0

        liquidation_z = 0.0
        if liq_cols:
            liq = df[liq_cols].astype(float).sum(axis=1)
            liquidation_z = _zscore(liq, float(liq.iloc[-1]), cfg.z_window)

        flags: list[str] = []
        if funding >= cfg.high_funding_abs or funding_z >= 2.0:
            flags.append("crowded_long_funding")
        if funding <= -cfg.high_funding_abs or funding_z <= -2.0:
            flags.append("crowded_short_funding")
        if long_short >= cfg.high_long_short_ratio or long_short_z >= 2.0:
            flags.append("long_short_ratio_long_crowded")
        if long_short <= cfg.low_long_short_ratio or long_short_z <= -2.0:
            flags.append("long_short_ratio_short_crowded")
        if abs(oi_change) >= cfg.oi_change_threshold:
            flags.append("open_interest_fast_change")
        if liquidation_z >= cfg.liquidation_z_threshold:
            flags.append("liquidation_spike")

        long_pressure = 0.0
        long_pressure += max(funding / max(cfg.high_funding_abs, 1e-12), 0.0)
        long_pressure += max((long_short - 1.0) / max(cfg.high_long_short_ratio - 1.0, 1e-12), 0.0)
        short_pressure = 0.0
        short_pressure += max(-funding / max(cfg.high_funding_abs, 1e-12), 0.0)
        short_pressure += max((1.0 - long_short) / max(1.0 - cfg.low_long_short_ratio, 1e-12), 0.0)
        if long_pressure > short_pressure and long_pressure > 0.5:
            direction = "long_crowded"
        elif short_pressure > long_pressure and short_pressure > 0.5:
            direction = "short_crowded"
        else:
            direction = "neutral"

        score = 0.0
        score += min(abs(funding) / max(cfg.high_funding_abs, 1e-12), 2.0) * 0.25
        score += min(abs(funding_z) / 3.0, 1.0) * 0.20
        score += min(abs(long_short - 1.0), 1.0) * 0.20
        score += min(abs(oi_change) / max(cfg.oi_change_threshold, 1e-12), 2.0) * 0.20
        score += min(max(liquidation_z, 0.0) / cfg.liquidation_z_threshold, 2.0) * 0.15

        return CrowdingState(
            funding=funding,
            funding_z=funding_z,
            open_interest_change=oi_change,
            long_short_ratio=long_short,
            long_short_z=long_short_z,
            liquidation_z=liquidation_z,
            crowding_score=_clip01(score),
            direction=direction,
            flags=flags,
        )

    def _correlation(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        assets: Optional[Mapping[str, pd.DataFrame]],
        benchmark_symbol: str,
        end_time: object,
    ) -> CorrelationState:
        if not assets:
            return CorrelationState()
        cfg = self.cfg
        series: dict[str, pd.Series] = {}
        for name, asset_df in assets.items():
            if "close" not in asset_df.columns:
                continue
            work = asset_df
            if end_time is not None and isinstance(asset_df.index, pd.DatetimeIndex):
                work = asset_df.loc[asset_df.index <= end_time]
            rets = _safe_returns(work["close"]).iloc[-cfg.lookback:]
            if rets.size >= max(8, cfg.lead_lag_bars + 3):
                series[str(name)] = rets
        if symbol and symbol not in series and "close" in df.columns:
            series[str(symbol)] = _safe_returns(df["close"]).iloc[-cfg.lookback:]
        if len(series) < 2:
            return CorrelationState(n_assets=len(series) or 1)

        ret_df = pd.DataFrame(series).dropna()
        if ret_df.shape[0] < max(8, cfg.lead_lag_bars + 3):
            return CorrelationState(n_assets=len(series))
        corr = ret_df.corr().replace([np.inf, -np.inf], np.nan)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        avg_corr = float(upper.mean()) if not upper.empty else 0.0
        latest = ret_df.iloc[-cfg.short_lookback:] if ret_df.shape[0] >= cfg.short_lookback else ret_df
        breadth = float((latest.sum(axis=0) > 0.0).mean())
        dispersion = float(latest.sum(axis=0).std(ddof=0))

        benchmark_key = benchmark_symbol if benchmark_symbol in ret_df.columns else ""
        if not benchmark_key:
            aliases = tuple(a.upper() for a in self.cfg.benchmark_aliases)
            for key in ret_df.columns:
                key_u = key.upper()
                if any(alias in key_u for alias in aliases):
                    benchmark_key = key
                    break
        lead_score = 0.0
        leader = ""
        if benchmark_key:
            lead_score = self._lead_score(ret_df, benchmark_key)
            leader = benchmark_key if lead_score > cfg.lead_score_threshold else ""
        else:
            totals = latest.sum(axis=0)
            leader = str(totals.abs().idxmax()) if totals.size else ""
            lead_score = self._lead_score(ret_df, leader) if leader else 0.0

        market_ret = float(latest.mean(axis=1).sum()) if not latest.empty else 0.0
        if avg_corr >= cfg.high_correlation_threshold and market_ret < 0:
            regime = "risk_off_sync"
        elif lead_score >= cfg.lead_score_threshold:
            regime = "benchmark_led" if benchmark_key and leader == benchmark_key else "leader_led"
        elif avg_corr >= cfg.high_correlation_threshold and market_ret >= 0:
            regime = "risk_on_sync"
        elif avg_corr <= cfg.fragmented_correlation_threshold:
            regime = "fragmented"
        else:
            regime = "mixed"

        return CorrelationState(
            regime=regime,
            avg_correlation=avg_corr if np.isfinite(avg_corr) else 0.0,
            leader_lead_score=lead_score if np.isfinite(lead_score) else 0.0,
            btc_lead_score=lead_score if benchmark_key and "BTC" in benchmark_key.upper() and np.isfinite(lead_score) else 0.0,
            market_breadth=_clip01(breadth),
            dispersion=max(0.0, dispersion if np.isfinite(dispersion) else 0.0),
            leader_symbol=leader,
            benchmark_symbol=benchmark_key,
            n_assets=int(ret_df.shape[1]),
        )

    def _lead_score(self, ret_df: pd.DataFrame, leader_key: str) -> float:
        lag = int(self.cfg.lead_lag_bars)
        if lag <= 0 or leader_key not in ret_df.columns:
            return 0.0
        leader = ret_df[leader_key]
        others = [c for c in ret_df.columns if c != leader_key]
        if not others:
            return 0.0
        forward_scores: list[float] = []
        reverse_scores: list[float] = []
        for other in others:
            alt = ret_df[other]
            aligned = pd.concat(
                [leader.shift(lag), alt, alt.shift(lag), leader],
                axis=1,
                keys=["leader_prev", "other_now", "other_prev", "leader_now"],
            ).dropna()
            if aligned.shape[0] < 5:
                continue
            forward_scores.append(float(aligned["leader_prev"].corr(aligned["other_now"])))
            reverse_scores.append(float(aligned["other_prev"].corr(aligned["leader_now"])))
        if not forward_scores:
            return 0.0
        fwd = float(np.nanmean(forward_scores))
        rev = float(np.nanmean(reverse_scores)) if reverse_scores else 0.0
        return float(np.clip(fwd - rev, -1.0, 1.0))

    def _crowding_warnings(self, crowding: CrowdingState) -> list[str]:
        warnings = []
        if crowding.crowding_score > 0.65:
            warnings.append(f"crowding elevated ({crowding.direction})")
        if "liquidation_spike" in crowding.flags:
            warnings.append("liquidation spike detected")
        if "open_interest_fast_change" in crowding.flags:
            warnings.append("open interest changing quickly")
        return warnings

    def _correlation_warnings(self, correlation: CorrelationState) -> list[str]:
        if correlation.regime == "risk_off_sync":
            return ["cross-asset risk-off synchronization"]
        if correlation.regime == "fragmented":
            return ["market is fragmented / low cross-asset confirmation"]
        if correlation.regime in {"benchmark_led", "leader_led"}:
            leader = correlation.leader_symbol or correlation.benchmark_symbol or "benchmark"
            return [f"{leader} appears to lead the cross-asset move"]
        return []


def coerce_market_context(value: object) -> MarketContextReport | None:
    """Accept a MarketContextReport or its dict representation."""
    if value is None:
        return None
    if isinstance(value, MarketContextReport):
        return value
    if not isinstance(value, Mapping):
        return None
    crowding_raw = value.get("crowding", {})
    corr_raw = value.get("correlation", {})
    crowding = crowding_raw if isinstance(crowding_raw, CrowdingState) else CrowdingState(**{
        k: crowding_raw[k] for k in CrowdingState.__dataclass_fields__ if isinstance(crowding_raw, Mapping) and k in crowding_raw
    })
    correlation = corr_raw if isinstance(corr_raw, CorrelationState) else CorrelationState(**{
        k: corr_raw[k] for k in CorrelationState.__dataclass_fields__ if isinstance(corr_raw, Mapping) and k in corr_raw
    })
    warnings = list(value.get("warnings", [])) if isinstance(value.get("warnings", []), list) else []
    return MarketContextReport(crowding=crowding, correlation=correlation, warnings=warnings)


__all__ = [
    "CorrelationState",
    "CrowdingState",
    "MarketContextAnalyzer",
    "MarketContextConfig",
    "MarketContextReport",
    "coerce_market_context",
]
