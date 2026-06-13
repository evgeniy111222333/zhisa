"""Market Regime Intelligence detector."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from zhisa.regime.context import (
    MarketContextAnalyzer,
    MarketContextConfig,
    MarketContextReport,
    coerce_market_context,
)
from zhisa.regime.features import RegimeFeatureConfig, compute_regime_features
from zhisa.regime.schema import (
    ExpectedDuration,
    MacroRegime,
    MesoRegime,
    MicroRegime,
    RegimeFeatures,
    RegimeReport,
    RiskMode,
)
from zhisa.regime.structure import (
    MarketStructureAnalyzer,
    MarketStructureReport,
    StructureConfig,
)
from zhisa.regime.state_space import (
    StateSpaceConfig,
    StateSpaceRegimeModel,
    StateSpaceReport,
)
from zhisa.storage.resampler import resample_ohlcv
from zhisa.storage.schema import Timeframe


@dataclass(frozen=True)
class RegimeIntelligenceConfig:
    source_timeframe: str = "5m"
    timeframes: tuple[str, ...] = ("5m", "15m", "1h")
    feature: RegimeFeatureConfig = field(default_factory=RegimeFeatureConfig)
    min_confidence: float = 0.35
    trend_threshold: float = 0.65
    strong_trend_threshold: float = 1.15
    crash_return_threshold: float = -0.055
    crash_drawdown_threshold: float = 0.12
    expansion_vol_ratio: float = 1.35
    compression_quantile: float = 0.25
    range_trend_threshold: float = 0.35
    context: MarketContextConfig = field(default_factory=MarketContextConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    vals = np.array(list(scores.values()), dtype=np.float64)
    vals = vals - vals.max()
    exp = np.exp(vals)
    probs = exp / max(float(exp.sum()), 1e-12)
    return {k: float(v) for k, v in zip(scores.keys(), probs)}


def _tf_weight(tf: str) -> float:
    minutes = Timeframe.from_str(tf).minutes if tf in {m.value for m in Timeframe} else 5
    return float(np.log1p(minutes))


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


class RegimeIntelligence:
    """Hierarchical, multi-timeframe market regime analyzer.

    The analyzer is deterministic and causal: ``analyze(df, t=i)`` uses
    only bars ``<= i``.  It is deliberately structured as a separate
    context layer so policy/risk code can consume it without depending on
    a neural model being trained first.
    """

    def __init__(self, cfg: Optional[RegimeIntelligenceConfig] = None) -> None:
        self.cfg = cfg or RegimeIntelligenceConfig()

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        t: Optional[int] = None,
        symbol: str = "",
        extra_context: Optional[dict] = None,
    ) -> RegimeReport:
        if t is not None:
            if t < 0:
                raise ValueError("t must be non-negative")
            work = df.iloc[: t + 1].copy()
        else:
            work = df.copy()
        if work.empty:
            raise ValueError("df slice is empty")

        extra = extra_context or {}
        market_context = self._market_context(work, symbol=symbol, extra=extra)
        market_structure = self._market_structure(work)
        state_space = self._state_space(work)
        features = self._multi_timeframe_features(work)
        primary_tf = self.cfg.timeframes[0]
        primary = features.get(primary_tf) or next(iter(features.values()))
        agg = self._aggregate(features)
        macro, probs = self._classify_macro(agg)
        meso = self._classify_meso(macro, primary, agg)
        micro = self._classify_micro(primary)
        confidence = max(probs.values()) if probs else self.cfg.min_confidence
        transition_risk = self._transition_risk(primary, agg, macro, meso)
        transition_risk = self._context_transition_risk(transition_risk, market_context, market_structure, state_space)
        uncertainty = _clip01((1.0 - confidence) * 0.7 + transition_risk * 0.3)
        risk_mode, size_mult = self._risk_posture(macro, meso, transition_risk, primary, market_context, market_structure)
        allowed, blocked = self._playbooks(macro, meso, micro, risk_mode, primary, market_context, market_structure, state_space)
        stop_style, tp_style = self._exit_styles(macro, meso, primary)
        tradeability = self._tradeability(macro, meso, micro, risk_mode, transition_risk, primary, market_context, market_structure)
        why, danger = self._explain(macro, meso, micro, primary, agg, extra, market_context, market_structure, state_space)

        return RegimeReport(
            primary_regime=macro.value,
            secondary_regime=meso.value,
            micro_regime=micro.value,
            confidence=_clip01(confidence),
            uncertainty=uncertainty,
            expected_duration=self._expected_duration(macro, meso, primary).value,
            transition_risk=transition_risk,
            tradeability_score=tradeability,
            allowed_playbooks=allowed,
            blocked_playbooks=blocked,
            risk_mode=risk_mode.value,
            position_size_multiplier=size_mult,
            stop_style=stop_style,
            take_profit_style=tp_style,
            explanation={"why": why, "danger": danger},
            trend_phase=market_structure.trend.phase,
            features={
                "symbol": symbol,
                "timestamp": work.index[-1].isoformat() if isinstance(work.index, pd.DatetimeIndex) else None,
                "aggregate": agg,
                "timeframes": {tf: f.to_dict() for tf, f in features.items()},
                "market_context": market_context.to_dict(),
                "market_structure": market_structure.to_dict(),
                "state_space": state_space.to_dict(),
            },
            probabilities=probs,
        )

    def _market_context(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        extra: dict,
    ) -> MarketContextReport:
        supplied = coerce_market_context(extra.get("market_context"))
        if supplied is not None:
            return supplied
        analyzer = MarketContextAnalyzer(self.cfg.context)
        return analyzer.analyze(
            df,
            symbol=symbol,
            assets=extra.get("assets"),
            benchmark_symbol=str(extra.get("benchmark_symbol", extra.get("btc_symbol", self.cfg.context.benchmark_symbol))),
        )

    def _market_structure(self, df: pd.DataFrame) -> MarketStructureReport:
        return MarketStructureAnalyzer(self.cfg.structure).analyze(df)

    def _state_space(self, df: pd.DataFrame) -> StateSpaceReport:
        return StateSpaceRegimeModel(self.cfg.state_space).analyze(df)

    def _multi_timeframe_features(self, df: pd.DataFrame) -> dict[str, RegimeFeatures]:
        out: dict[str, RegimeFeatures] = {}
        source_tf = Timeframe.from_str(self.cfg.source_timeframe)
        for tf_s in self.cfg.timeframes:
            tf = Timeframe.from_str(tf_s)
            if tf.minutes == source_tf.minutes:
                tf_df = df
            else:
                if not source_tf.can_resample_to(tf):
                    continue
                tf_df = resample_ohlcv(df, source_tf, tf, dropna=True)
            if len(tf_df) < 2:
                continue
            out[tf_s] = compute_regime_features(
                tf_df, timeframe=tf_s, cfg=self.cfg.feature,
            )
        if not out:
            out[self.cfg.source_timeframe] = compute_regime_features(
                df, timeframe=self.cfg.source_timeframe, cfg=self.cfg.feature,
            )
        return out

    def _aggregate(self, features: dict[str, RegimeFeatures]) -> dict[str, float]:
        weighted = []
        weights = []
        for tf, feat in features.items():
            w = _tf_weight(tf)
            weighted.append((w, feat))
            weights.append(w)
        denom = max(sum(weights), 1e-12)

        def avg(attr: str) -> float:
            return float(sum(w * float(getattr(f, attr)) for w, f in weighted) / denom)

        return {
            "trend_score": avg("trend_score"),
            "trend_efficiency": avg("trend_efficiency"),
            "ret_short": avg("ret_short"),
            "ret_medium": avg("ret_medium"),
            "ret_long": avg("ret_long"),
            "vol_ratio": avg("vol_ratio"),
            "bb_width_quantile": avg("bb_width_quantile"),
            "atr_pct": avg("atr_pct"),
            "volume_z": avg("volume_z"),
            "range_position": avg("range_position"),
            "drawdown": avg("drawdown"),
            "shock_score": avg("shock_score"),
        }

    def _classify_macro(self, agg: dict[str, float]) -> tuple[MacroRegime, dict[str, float]]:
        trend = agg["trend_score"]
        eff = agg["trend_efficiency"]
        ret_l = agg["ret_long"]
        vol = agg["vol_ratio"]
        dd = agg["drawdown"]
        shock = agg["shock_score"]
        atr_pct = agg["atr_pct"]

        scores = {
            MacroRegime.BULL_TREND.value: 0.9 * trend + 0.7 * eff + 4.0 * max(ret_l, 0.0),
            MacroRegime.BEAR_TREND.value: -0.9 * trend + 0.7 * eff + 4.0 * max(-ret_l, 0.0),
            MacroRegime.BROAD_RANGE.value: 1.1 * (1.0 - min(abs(trend), 2.0) / 2.0) + 0.6 * (1.0 - eff),
            MacroRegime.HIGH_VOL_CRASH.value: 1.2 * (vol > self.cfg.expansion_vol_ratio) + 3.0 * max(-ret_l, 0.0) + 2.0 * dd + 0.08 * shock,
            MacroRegime.POST_CRASH_RECOVERY.value: 2.0 * dd + 4.0 * max(agg["ret_short"], 0.0) + 2.0 * max(trend, 0.0),
            MacroRegime.LOW_LIQUIDITY_CHOP.value: 0.7 * (agg["volume_z"] < -0.75) + 0.7 * (eff < 0.25) + 0.5 * (abs(trend) < 0.3),
            MacroRegime.EVENT_DRIVEN.value: 0.12 * shock + 0.8 * (abs(agg["volume_z"]) > 3.0),
        }
        probs = _softmax(scores)

        if (
            ret_l <= self.cfg.crash_return_threshold
            or dd >= self.cfg.crash_drawdown_threshold
        ) and (vol >= 1.05 or atr_pct >= 0.035 or shock >= 2.0):
            return MacroRegime.HIGH_VOL_CRASH, probs
        if trend >= self.cfg.trend_threshold and ret_l >= 0:
            return MacroRegime.BULL_TREND, probs
        if trend <= -self.cfg.trend_threshold and ret_l <= 0:
            return MacroRegime.BEAR_TREND, probs
        best = max(probs, key=probs.get)
        return MacroRegime(best), probs

    def _classify_meso(
        self,
        macro: MacroRegime,
        primary: RegimeFeatures,
        agg: dict[str, float],
    ) -> MesoRegime:
        if primary.shock_score > 6.0 and primary.ret_short < 0 and primary.vol_ratio > 1.2:
            return MesoRegime.LIQUIDATION_CASCADE
        if primary.liquidity_sweep_high or primary.liquidity_sweep_low:
            return MesoRegime.FAILED_BREAKOUT
        if primary.bb_width_quantile <= self.cfg.compression_quantile and primary.vol_ratio < 0.95:
            return MesoRegime.COMPRESSION
        if primary.vol_ratio >= self.cfg.expansion_vol_ratio:
            return MesoRegime.EXPANSION
        if primary.breakout_up or primary.breakout_down:
            return MesoRegime.BREAKOUT
        if macro == MacroRegime.BULL_TREND and primary.ret_short < 0 and primary.range_position > 0.35:
            return MesoRegime.PULLBACK
        if macro == MacroRegime.BEAR_TREND and primary.ret_short > 0 and primary.range_position < 0.65:
            return MesoRegime.PULLBACK
        if abs(agg["trend_score"]) >= self.cfg.strong_trend_threshold and agg["trend_efficiency"] > 0.45:
            return MesoRegime.IMPULSE
        if abs(agg["trend_score"]) < self.cfg.range_trend_threshold:
            return MesoRegime.CHOP
        return MesoRegime.ACCUMULATION if macro != MacroRegime.BEAR_TREND else MesoRegime.DISTRIBUTION

    def _classify_micro(self, primary: RegimeFeatures) -> MicroRegime:
        if primary.liquidity_sweep_high or primary.liquidity_sweep_low:
            return MicroRegime.STOP_RUN
        if abs(primary.volume_z) >= 3.0:
            return MicroRegime.VOLUME_SPIKE
        if primary.shock_score >= 4.0 and primary.trend_efficiency < 0.35:
            return MicroRegime.WICK_REJECTION
        if primary.volume_z < -1.0 and primary.trend_efficiency < 0.25:
            return MicroRegime.THIN_BOOK
        if primary.trend_efficiency < 0.2:
            return MicroRegime.NOISY_CHOP
        return MicroRegime.QUIET

    def _transition_risk(
        self,
        primary: RegimeFeatures,
        agg: dict[str, float],
        macro: MacroRegime,
        meso: MesoRegime,
    ) -> float:
        risk = 0.0
        risk += 0.25 if primary.vol_ratio > 1.4 else 0.0
        risk += 0.20 if primary.bb_width_quantile < 0.2 or primary.bb_width_quantile > 0.85 else 0.0
        risk += 0.20 if meso in {MesoRegime.FAILED_BREAKOUT, MesoRegime.LIQUIDATION_CASCADE} else 0.0
        risk += 0.25 if macro in {MacroRegime.HIGH_VOL_CRASH, MacroRegime.EVENT_DRIVEN} else 0.0
        risk += 0.15 if abs(agg["trend_score"]) < 0.35 and agg["vol_ratio"] > 1.1 else 0.0
        risk += min(primary.shock_score / 20.0, 0.2)
        return _clip01(risk)

    def _context_transition_risk(
        self,
        base: float,
        context: MarketContextReport,
        structure: MarketStructureReport,
        state_space: StateSpaceReport,
    ) -> float:
        risk = float(base)
        crowding = context.crowding
        corr = context.correlation
        risk += 0.15 * crowding.crowding_score
        risk += 0.15 if "liquidation_spike" in crowding.flags else 0.0
        risk += 0.12 if corr.regime == "risk_off_sync" else 0.0
        risk += 0.08 if corr.regime == "fragmented" else 0.0
        risk += 0.05 if corr.regime in {"benchmark_led", "leader_led"} and corr.leader_lead_score > 0.2 else 0.0
        risk += 0.12 if structure.trend.phase in {"late", "exhausted"} else 0.0
        risk += 0.08 if structure.trend.exhaustion_score > 0.55 else 0.0
        risk += 0.20 * max(0.0, state_space.transition_probability - 0.30)
        risk += 0.18 * max(0.0, state_space.change_point_score - 0.50)
        risk += 0.08 * max(0.0, state_space.entropy - 0.50)
        return _clip01(risk)

    def _risk_posture(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        transition_risk: float,
        primary: RegimeFeatures,
        context: MarketContextReport,
        structure: MarketStructureReport,
    ) -> tuple[RiskMode, float]:
        if "liquidation_spike" in context.crowding.flags:
            return RiskMode.DEFENSIVE, 0.2
        if macro == MacroRegime.HIGH_VOL_CRASH or meso == MesoRegime.LIQUIDATION_CASCADE:
            return RiskMode.DEFENSIVE, 0.2
        if transition_risk > 0.65 or macro == MacroRegime.EVENT_DRIVEN:
            return RiskMode.REDUCED, 0.35
        if context.correlation.regime == "risk_off_sync" and transition_risk > 0.35:
            return RiskMode.REDUCED, 0.35
        if context.crowding.crowding_score > 0.65:
            return RiskMode.REDUCED, 0.5
        if structure.trend.phase == "exhausted":
            return RiskMode.REDUCED, 0.45
        if meso == MesoRegime.COMPRESSION:
            return RiskMode.REDUCED, 0.5
        if abs(primary.trend_score) > 1.2 and primary.trend_efficiency > 0.55:
            return RiskMode.AGGRESSIVE, 1.15
        if macro == MacroRegime.LOW_LIQUIDITY_CHOP:
            return RiskMode.DEFENSIVE, 0.25
        return RiskMode.NORMAL, 0.75

    def _playbooks(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        micro: MicroRegime,
        risk_mode: RiskMode,
        primary: RegimeFeatures,
        context: MarketContextReport,
        structure: MarketStructureReport,
        state_space: StateSpaceReport,
    ) -> tuple[list[str], list[str]]:
        allowed: set[str] = set()
        blocked: set[str] = set()
        if macro == MacroRegime.BULL_TREND:
            allowed.update({"trend_pullback_long", "breakout_retest_long"})
            blocked.update({"blind_mean_reversion_short", "late_breakout_chase_short"})
        elif macro == MacroRegime.BEAR_TREND:
            allowed.update({"trend_pullback_short", "breakout_retest_short"})
            blocked.update({"blind_mean_reversion_long", "late_breakout_chase_long"})
        elif macro == MacroRegime.BROAD_RANGE:
            allowed.update({"range_reversion_long", "range_reversion_short"})
            blocked.update({"late_breakout_chase_long", "late_breakout_chase_short"})
        elif macro == MacroRegime.HIGH_VOL_CRASH:
            allowed.update({"panic_retest_short", "capitulation_reversal_small"})
            blocked.update({"full_size_long", "late_breakout_chase_long", "blind_dip_buy"})
        else:
            allowed.add("no_trade_wait")

        if meso == MesoRegime.COMPRESSION:
            allowed.add("volatility_expansion_wait")
            blocked.update({"large_pre_breakout_position", "overtrade_chop"})
        if meso == MesoRegime.FAILED_BREAKOUT or micro == MicroRegime.STOP_RUN:
            allowed.add("liquidity_sweep_reversal")
            blocked.add("breakout_chase")
        if risk_mode in {RiskMode.DEFENSIVE, RiskMode.REDUCED}:
            blocked.add("full_size_position")
        if context.crowding.direction == "long_crowded":
            allowed.add("pullback_only_long")
            blocked.update({"late_breakout_chase_long", "crowded_long_chase"})
        if context.crowding.direction == "short_crowded":
            allowed.add("pullback_only_short")
            blocked.update({"late_breakout_chase_short", "crowded_short_chase"})
        if "liquidation_spike" in context.crowding.flags:
            allowed.add("liquidation_retest_only")
            blocked.update({"fresh_full_size_entry", "blind_liquidation_fade"})
        if context.correlation.regime == "fragmented":
            allowed.add("relative_strength_only")
            blocked.add("market_beta_chase")
        if context.correlation.regime == "risk_off_sync":
            blocked.update({"full_size_long", "correlation_blind_long"})
        if structure.trend.phase in {"late", "exhausted"}:
            allowed.add("pullback_to_value_only")
            blocked.update({"late_trend_chase", "full_size_breakout_chase"})
        if structure.liquidity.nearest_level is not None and abs(structure.liquidity.nearest_level.distance_pct) < 0.01:
            blocked.add("entry_directly_into_liquidity")
        if structure.liquidity.in_value_area:
            allowed.add("value_area_reversion")
        if state_space.change_point_score > 0.65 or state_space.transition_probability > 0.55:
            allowed.add("transition_wait")
            blocked.update({"fresh_full_size_entry", "regime_transition_chase"})
        if primary.range_position > 0.85 and macro == MacroRegime.BULL_TREND:
            blocked.add("chase_long_at_high")
        if primary.range_position < 0.15 and macro == MacroRegime.BEAR_TREND:
            blocked.add("chase_short_at_low")
        return sorted(allowed), sorted(blocked)

    def _exit_styles(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        primary: RegimeFeatures,
    ) -> tuple[str, str]:
        if macro == MacroRegime.HIGH_VOL_CRASH:
            return "wide_volatility_or_no_trade", "fast_partial_exits"
        if meso in {MesoRegime.COMPRESSION, MesoRegime.CHOP}:
            return "tight_invalidation", "wait_for_expansion"
        if abs(primary.trend_score) > 1.0:
            return "structure_based", "partial_trailing"
        return "atr_based", "range_target"

    def _tradeability(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        micro: MicroRegime,
        risk_mode: RiskMode,
        transition_risk: float,
        primary: RegimeFeatures,
        context: MarketContextReport,
        structure: MarketStructureReport,
    ) -> float:
        score = 0.65
        score += 0.15 if meso in {MesoRegime.PULLBACK, MesoRegime.BREAKOUT, MesoRegime.IMPULSE} else 0.0
        score -= 0.25 if meso in {MesoRegime.CHOP, MesoRegime.COMPRESSION} else 0.0
        score -= 0.35 if macro in {MacroRegime.HIGH_VOL_CRASH, MacroRegime.LOW_LIQUIDITY_CHOP} else 0.0
        score -= 0.25 * transition_risk
        score += 0.10 if primary.trend_efficiency > 0.55 else 0.0
        score -= 0.15 if micro in {MicroRegime.THIN_BOOK, MicroRegime.NOISY_CHOP} else 0.0
        score -= 0.10 if risk_mode == RiskMode.DEFENSIVE else 0.0
        score -= 0.15 * context.crowding.crowding_score
        score -= 0.15 if context.correlation.regime == "risk_off_sync" else 0.0
        score -= 0.10 if context.correlation.regime == "fragmented" else 0.0
        score -= 0.12 if structure.trend.phase == "late" else 0.0
        score -= 0.22 if structure.trend.phase == "exhausted" else 0.0
        score -= 0.08 if structure.liquidity.nearest_level is not None and abs(structure.liquidity.nearest_level.distance_pct) < 0.005 else 0.0
        return _clip01(score)

    def _expected_duration(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        primary: RegimeFeatures,
    ) -> ExpectedDuration:
        if meso in {MesoRegime.LIQUIDATION_CASCADE, MesoRegime.FAILED_BREAKOUT}:
            return ExpectedDuration.VERY_SHORT
        if meso in {MesoRegime.EXPANSION, MesoRegime.BREAKOUT}:
            return ExpectedDuration.SHORT
        if macro in {MacroRegime.BULL_TREND, MacroRegime.BEAR_TREND} and primary.trend_efficiency > 0.45:
            return ExpectedDuration.MEDIUM
        if macro == MacroRegime.BROAD_RANGE:
            return ExpectedDuration.MEDIUM
        return ExpectedDuration.UNKNOWN

    def _explain(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        micro: MicroRegime,
        primary: RegimeFeatures,
        agg: dict[str, float],
        extra: dict,
        context: MarketContextReport,
        structure: MarketStructureReport,
        state_space: StateSpaceReport,
    ) -> tuple[list[str], list[str]]:
        why: list[str] = []
        danger: list[str] = []
        why.append(f"macro={macro.value} from trend_score={agg['trend_score']:.2f}, ret_long={agg['ret_long']:.3f}")
        why.append(f"meso={meso.value}, vol_ratio={primary.vol_ratio:.2f}, bb_rank={primary.bb_width_quantile:.2f}")
        why.append(f"micro={micro.value}, volume_z={primary.volume_z:.2f}, efficiency={primary.trend_efficiency:.2f}")
        if primary.breakout_up:
            why.append("price closed above prior range high")
        if primary.breakout_down:
            why.append("price closed below prior range low")
        if primary.liquidity_sweep_high or primary.liquidity_sweep_low:
            danger.append("liquidity sweep / failed breakout detected")
        if primary.range_position > 0.85:
            danger.append("price near upper range / liquidity high")
        if primary.range_position < 0.15:
            danger.append("price near lower range / liquidity low")
        if primary.vol_ratio > 1.5:
            danger.append("volatility expansion raises transition risk")
        funding = extra.get("funding")
        if funding is not None and abs(float(funding)) > 0.0005:
            danger.append("funding is crowded")
        crowding = context.crowding
        corr = context.correlation
        if crowding.flags:
            why.append(f"crowding={crowding.direction}, score={crowding.crowding_score:.2f}")
        if crowding.direction != "neutral":
            danger.append(f"derivatives crowding is {crowding.direction}")
        if "open_interest_fast_change" in crowding.flags:
            danger.append("open interest is changing quickly")
        if "liquidation_spike" in crowding.flags:
            danger.append("liquidation spike / forced flow detected")
        if corr.regime != "single_asset":
            why.append(
                f"correlation_regime={corr.regime}, avg_corr={corr.avg_correlation:.2f}, "
                f"leader_lead={corr.leader_lead_score:.2f}"
            )
        if corr.regime == "risk_off_sync":
            danger.append("cross-asset risk-off synchronization")
        if corr.regime == "fragmented":
            danger.append("market is fragmented; broad confirmation is weak")
        if corr.regime in {"benchmark_led", "leader_led"}:
            leader = corr.leader_symbol or corr.benchmark_symbol or "benchmark"
            danger.append(f"{leader} appears to lead; other signals may lag")
        trend = structure.trend
        why.append(
            f"trend_phase={trend.phase}, maturity={trend.maturity_score:.2f}, exhaustion={trend.exhaustion_score:.2f}"
        )
        if trend.phase == "late":
            danger.append("trend is late; prefer pullback-to-value over chase")
        if trend.phase == "exhausted":
            danger.append("trend exhaustion risk is elevated")
        why.append(
            f"state_space={state_space.state_label}, change_point={state_space.change_point_score:.2f}, "
            f"state_transition={state_space.transition_probability:.2f}"
        )
        if state_space.change_point_score > 0.65:
            danger.append("unsupervised change-point score is elevated")
        if state_space.transition_probability > 0.55:
            danger.append("state-space model sees elevated transition probability")
        liq = structure.liquidity
        if liq.nearest_level is not None:
            danger.append(
                f"nearest liquidity {liq.nearest_level.name} is {liq.nearest_level.distance_pct:.3%} away"
            )
        if liq.in_value_area:
            why.append("price is inside value area")
        else:
            why.append(f"price is {liq.distance_to_value_mid_pct:.3%} from value area mid")
        return why, danger


__all__ = ["RegimeIntelligence", "RegimeIntelligenceConfig"]
