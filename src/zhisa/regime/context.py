"""Crowding, orderflow, and cross-asset context for regime intelligence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrowdingScoringConfig:
    funding_abs_weight: float = 0.25
    funding_z_weight: float = 0.20
    long_short_weight: float = 0.20
    open_interest_weight: float = 0.20
    liquidation_weight: float = 0.15
    funding_z_norm: float = 3.0
    funding_z_flag_threshold: float = 2.0
    long_short_z_flag_threshold: float = 2.0
    long_short_distance_norm: float = 1.0
    open_interest_norm_cap: float = 2.0
    liquidation_norm_cap: float = 2.0
    direction_pressure_threshold: float = 0.50
    warning_score_threshold: float = 0.65


@dataclass(frozen=True)
class OrderflowScoringConfig:
    bid_ask_weight: float = 0.22
    buy_sell_weight: float = 0.25
    delta_z_weight: float = 0.20
    spread_weight: float = 0.13
    thin_depth_weight: float = 0.12
    trade_intensity_weight: float = 0.08
    delta_z_norm: float = 3.0
    spread_norm_cap: float = 2.0
    trade_intensity_norm: float = 3.0
    direction_delta_z_norm: float = 3.0
    direction_pressure_threshold: float = 0.45
    warning_score_threshold: float = 0.65


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
    orderflow_imbalance_threshold: float = 0.30
    orderflow_delta_z_threshold: float = 2.0
    wide_spread_bps: float = 8.0
    thin_depth_threshold: float = 0.35
    crowding_scoring: CrowdingScoringConfig = field(default_factory=CrowdingScoringConfig)
    orderflow_scoring: OrderflowScoringConfig = field(default_factory=OrderflowScoringConfig)


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
    score_breakdown: dict[str, float] = field(default_factory=dict)

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
class OrderflowState:
    bid_ask_imbalance: float = 0.0
    buy_sell_imbalance: float = 0.0
    cumulative_delta: float = 0.0
    delta_z: float = 0.0
    trade_intensity_z: float = 0.0
    spread_bps: float = 0.0
    depth_imbalance: float = 0.0
    thin_depth_score: float = 0.0
    orderflow_score: float = 0.0
    direction: str = "neutral"
    flags: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketContextReport:
    crowding: CrowdingState = field(default_factory=CrowdingState)
    correlation: CorrelationState = field(default_factory=CorrelationState)
    orderflow: OrderflowState = field(default_factory=OrderflowState)
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
        extra_context: Optional[Mapping[str, Any]] = None,
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
        orderflow = self._orderflow(work, extra_context or {})
        warnings.extend(self._orderflow_warnings(orderflow))
        correlation = self._correlation(
            work,
            symbol=symbol,
            assets=assets,
            benchmark_symbol=benchmark_symbol or btc_symbol or self.cfg.benchmark_symbol,
            end_time=work.index[-1] if isinstance(work.index, pd.DatetimeIndex) else None,
        )
        warnings.extend(self._correlation_warnings(correlation))
        return MarketContextReport(crowding=crowding, correlation=correlation, orderflow=orderflow, warnings=warnings)

    def _crowding(self, df: pd.DataFrame) -> CrowdingState:
        cfg = self.cfg
        scoring = cfg.crowding_scoring
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
        if funding >= cfg.high_funding_abs or funding_z >= scoring.funding_z_flag_threshold:
            flags.append("crowded_long_funding")
        if funding <= -cfg.high_funding_abs or funding_z <= -scoring.funding_z_flag_threshold:
            flags.append("crowded_short_funding")
        if long_short >= cfg.high_long_short_ratio or long_short_z >= scoring.long_short_z_flag_threshold:
            flags.append("long_short_ratio_long_crowded")
        if long_short <= cfg.low_long_short_ratio or long_short_z <= -scoring.long_short_z_flag_threshold:
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
        if long_pressure > short_pressure and long_pressure > scoring.direction_pressure_threshold:
            direction = "long_crowded"
        elif short_pressure > long_pressure and short_pressure > scoring.direction_pressure_threshold:
            direction = "short_crowded"
        else:
            direction = "neutral"

        breakdown = {
            "funding_abs": min(abs(funding) / max(cfg.high_funding_abs, 1e-12), 2.0) * scoring.funding_abs_weight,
            "funding_z": min(abs(funding_z) / max(scoring.funding_z_norm, 1e-12), 1.0) * scoring.funding_z_weight,
            "long_short_ratio": min(abs(long_short - 1.0) / max(scoring.long_short_distance_norm, 1e-12), 1.0) * scoring.long_short_weight,
            "open_interest_change": min(abs(oi_change) / max(cfg.oi_change_threshold, 1e-12), scoring.open_interest_norm_cap) * scoring.open_interest_weight,
            "liquidation_z": min(max(liquidation_z, 0.0) / max(cfg.liquidation_z_threshold, 1e-12), scoring.liquidation_norm_cap) * scoring.liquidation_weight,
        }
        score = sum(breakdown.values())
        breakdown["raw"] = float(score)
        breakdown["score"] = _clip01(score)

        return CrowdingState(
            funding=funding,
            funding_z=funding_z,
            open_interest_change=oi_change,
            long_short_ratio=long_short,
            long_short_z=long_short_z,
            liquidation_z=liquidation_z,
            crowding_score=breakdown["score"],
            direction=direction,
            flags=flags,
            score_breakdown=breakdown,
        )

    def _orderflow(self, df: pd.DataFrame, extra: Mapping[str, Any]) -> OrderflowState:
        cfg = self.cfg
        scoring = cfg.orderflow_scoring
        bid_col = _first_existing(df, ("bid_volume", "bid_size", "bid_depth", "best_bid_size", "bids_volume"))
        ask_col = _first_existing(df, ("ask_volume", "ask_size", "ask_depth", "best_ask_size", "asks_volume"))
        bid_px_col = _first_existing(df, ("best_bid", "bid", "bid_price"))
        ask_px_col = _first_existing(df, ("best_ask", "ask", "ask_price"))
        spread_col = _first_existing(df, ("spread_bps", "bid_ask_spread_bps"))
        buy_col = _first_existing(df, ("taker_buy_volume", "buy_volume", "market_buy_volume"))
        sell_col = _first_existing(df, ("taker_sell_volume", "sell_volume", "market_sell_volume"))
        delta_col = _first_existing(df, ("volume_delta", "orderflow_delta", "delta", "cvd_delta"))
        cvd_col = _first_existing(df, ("cumulative_delta", "cvd", "cumulative_volume_delta"))
        trades_col = _first_existing(df, ("trades", "trade_count", "num_trades", "n_trades"))

        book = self._orderbook_metrics(extra.get("orderbook") or extra.get("book"))
        bid_depth = book.get("bid_depth", 0.0)
        ask_depth = book.get("ask_depth", 0.0)
        spread_bps = book.get("spread_bps", 0.0)

        if bid_col and ask_col:
            bid_depth = _finite(df[bid_col].iloc[-1])
            ask_depth = _finite(df[ask_col].iloc[-1])
        if spread_col:
            spread_bps = max(0.0, _finite(df[spread_col].iloc[-1]))
        elif bid_px_col and ask_px_col:
            bid = _finite(df[bid_px_col].iloc[-1])
            ask = _finite(df[ask_px_col].iloc[-1])
            mid = 0.5 * (bid + ask)
            if bid > 0 and ask > 0 and mid > 0 and ask >= bid:
                spread_bps = float((ask - bid) / mid * 10_000.0)

        bid_ask_imb = 0.0
        if bid_depth + ask_depth > 1e-12:
            bid_ask_imb = float(np.clip((bid_depth - ask_depth) / (bid_depth + ask_depth), -1.0, 1.0))

        buy = _finite(df[buy_col].iloc[-1]) if buy_col else 0.0
        sell = _finite(df[sell_col].iloc[-1]) if sell_col else 0.0
        if buy_col and sell_col and buy + sell > 1e-12:
            buy_sell_imb = float(np.clip((buy - sell) / (buy + sell), -1.0, 1.0))
            delta_series = (df[buy_col].astype(float) - df[sell_col].astype(float)).replace([np.inf, -np.inf], np.nan)
            delta = float(delta_series.iloc[-1])
        elif delta_col:
            delta_series = df[delta_col].astype(float).replace([np.inf, -np.inf], np.nan)
            delta = _finite(delta_series.iloc[-1])
            vol_col = _first_existing(df, ("volume", "quote_volume", "base_volume"))
            denom = abs(_finite(df[vol_col].iloc[-1], 1.0)) if vol_col else max(abs(delta), 1.0)
            buy_sell_imb = float(np.clip(delta / max(denom, 1e-12), -1.0, 1.0))
        else:
            delta_series = pd.Series(dtype=float)
            delta = 0.0
            buy_sell_imb = 0.0

        cumulative_delta = _finite(df[cvd_col].iloc[-1]) if cvd_col else 0.0
        if not cvd_col and not delta_series.empty:
            cumulative_delta = float(delta_series.dropna().iloc[-cfg.lookback:].sum())
        delta_z = _zscore(delta_series, delta, cfg.z_window) if not delta_series.empty else 0.0
        trade_intensity_z = 0.0
        if trades_col:
            trades = df[trades_col].astype(float).replace([np.inf, -np.inf], np.nan)
            trade_intensity_z = _zscore(trades, _finite(trades.iloc[-1]), cfg.z_window)

        depth_hist_col = bid_col if bid_col and ask_col else None
        thin_depth_score = 0.0
        if depth_hist_col and ask_col:
            total_depth = df[bid_col].astype(float) + df[ask_col].astype(float)
            hist = total_depth.replace([np.inf, -np.inf], np.nan).dropna().iloc[-cfg.z_window:]
            current_depth = bid_depth + ask_depth
            if hist.size >= 5 and float(hist.median()) > 0:
                thin_depth_score = _clip01(1.0 - current_depth / float(hist.median()))

        flags: list[str] = []
        if bid_ask_imb >= cfg.orderflow_imbalance_threshold:
            flags.append("book_bid_pressure")
        if bid_ask_imb <= -cfg.orderflow_imbalance_threshold:
            flags.append("book_ask_pressure")
        if buy_sell_imb >= cfg.orderflow_imbalance_threshold or delta_z >= cfg.orderflow_delta_z_threshold:
            flags.append("aggressive_buying")
        if buy_sell_imb <= -cfg.orderflow_imbalance_threshold or delta_z <= -cfg.orderflow_delta_z_threshold:
            flags.append("aggressive_selling")
        if spread_bps >= cfg.wide_spread_bps:
            flags.append("wide_spread")
        if thin_depth_score >= cfg.thin_depth_threshold:
            flags.append("thin_depth")
        if trade_intensity_z >= cfg.orderflow_delta_z_threshold:
            flags.append("trade_intensity_spike")

        buy_pressure = max(bid_ask_imb, 0.0) + max(buy_sell_imb, 0.0) + max(delta_z, 0.0) / max(scoring.direction_delta_z_norm, 1e-12)
        sell_pressure = max(-bid_ask_imb, 0.0) + max(-buy_sell_imb, 0.0) + max(-delta_z, 0.0) / max(scoring.direction_delta_z_norm, 1e-12)
        if buy_pressure > sell_pressure and buy_pressure > scoring.direction_pressure_threshold:
            direction = "buy_pressure"
        elif sell_pressure > buy_pressure and sell_pressure > scoring.direction_pressure_threshold:
            direction = "sell_pressure"
        else:
            direction = "neutral"

        breakdown = {
            "bid_ask_imbalance": min(abs(bid_ask_imb), 1.0) * scoring.bid_ask_weight,
            "buy_sell_imbalance": min(abs(buy_sell_imb), 1.0) * scoring.buy_sell_weight,
            "delta_z": min(abs(delta_z) / max(scoring.delta_z_norm, 1e-12), 1.0) * scoring.delta_z_weight,
            "spread": min(max(spread_bps, 0.0) / max(cfg.wide_spread_bps, 1e-12), scoring.spread_norm_cap) * scoring.spread_weight,
            "thin_depth": thin_depth_score * scoring.thin_depth_weight,
            "trade_intensity": min(max(trade_intensity_z, 0.0) / max(scoring.trade_intensity_norm, 1e-12), 1.0) * scoring.trade_intensity_weight,
        }
        score = sum(breakdown.values())
        breakdown["raw"] = float(score)
        breakdown["score"] = _clip01(score)

        return OrderflowState(
            bid_ask_imbalance=bid_ask_imb,
            buy_sell_imbalance=buy_sell_imb,
            cumulative_delta=cumulative_delta,
            delta_z=delta_z,
            trade_intensity_z=trade_intensity_z,
            spread_bps=spread_bps,
            depth_imbalance=bid_ask_imb,
            thin_depth_score=thin_depth_score,
            orderflow_score=breakdown["score"],
            direction=direction,
            flags=flags,
            score_breakdown=breakdown,
        )

    def _orderbook_metrics(self, orderbook: object) -> dict[str, float]:
        if not isinstance(orderbook, Mapping):
            return {}

        def side_depth(raw: object) -> tuple[float, float]:
            if raw is None:
                return 0.0, 0.0
            depth = 0.0
            best = 0.0
            rows = raw.values() if isinstance(raw, Mapping) else raw
            try:
                iterator = iter(rows)
            except TypeError:
                return 0.0, 0.0
            for i, row in enumerate(iterator):
                price = 0.0
                size = 0.0
                if isinstance(row, Mapping):
                    price = _finite(row.get("price") or row.get("px"))
                    size = _finite(row.get("size") or row.get("qty") or row.get("amount") or row.get("volume"))
                else:
                    try:
                        price = _finite(row[0])
                        size = _finite(row[1])
                    except (TypeError, IndexError):
                        continue
                if i == 0:
                    best = price
                depth += max(price, 0.0) * max(size, 0.0)
            return depth, best

        bid_depth, best_bid = side_depth(orderbook.get("bids"))
        ask_depth, best_ask = side_depth(orderbook.get("asks"))
        spread_bps = 0.0
        mid = 0.5 * (best_bid + best_ask)
        if best_bid > 0 and best_ask > 0 and best_ask >= best_bid and mid > 0:
            spread_bps = float((best_ask - best_bid) / mid * 10_000.0)
        return {"bid_depth": bid_depth, "ask_depth": ask_depth, "spread_bps": spread_bps}

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
        if crowding.crowding_score > self.cfg.crowding_scoring.warning_score_threshold:
            warnings.append(f"crowding elevated ({crowding.direction})")
        if "liquidation_spike" in crowding.flags:
            warnings.append("liquidation spike detected")
        if "open_interest_fast_change" in crowding.flags:
            warnings.append("open interest changing quickly")
        return warnings

    def _orderflow_warnings(self, orderflow: OrderflowState) -> list[str]:
        warnings = []
        if orderflow.orderflow_score > self.cfg.orderflow_scoring.warning_score_threshold:
            warnings.append(f"orderflow stress elevated ({orderflow.direction})")
        if "wide_spread" in orderflow.flags:
            warnings.append("spread is wide")
        if "thin_depth" in orderflow.flags:
            warnings.append("book depth is thin")
        if "aggressive_buying" in orderflow.flags or "aggressive_selling" in orderflow.flags:
            warnings.append("aggressive taker flow detected")
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
    orderflow_raw = value.get("orderflow", {})
    crowding = crowding_raw if isinstance(crowding_raw, CrowdingState) else CrowdingState(**{
        k: crowding_raw[k] for k in CrowdingState.__dataclass_fields__ if isinstance(crowding_raw, Mapping) and k in crowding_raw
    })
    correlation = corr_raw if isinstance(corr_raw, CorrelationState) else CorrelationState(**{
        k: corr_raw[k] for k in CorrelationState.__dataclass_fields__ if isinstance(corr_raw, Mapping) and k in corr_raw
    })
    orderflow = orderflow_raw if isinstance(orderflow_raw, OrderflowState) else OrderflowState(**{
        k: orderflow_raw[k] for k in OrderflowState.__dataclass_fields__ if isinstance(orderflow_raw, Mapping) and k in orderflow_raw
    })
    warnings = list(value.get("warnings", [])) if isinstance(value.get("warnings", []), list) else []
    return MarketContextReport(crowding=crowding, correlation=correlation, orderflow=orderflow, warnings=warnings)


__all__ = [
    "CorrelationState",
    "CrowdingState",
    "CrowdingScoringConfig",
    "MarketContextAnalyzer",
    "MarketContextConfig",
    "MarketContextReport",
    "OrderflowState",
    "OrderflowScoringConfig",
    "coerce_market_context",
]
