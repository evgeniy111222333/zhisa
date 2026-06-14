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
class RegimeClassificationConfig:
    bull_trend_weight: float = 0.90
    bull_efficiency_weight: float = 0.70
    bull_return_weight: float = 4.00
    bear_trend_weight: float = 0.90
    bear_efficiency_weight: float = 0.70
    bear_return_weight: float = 4.00
    broad_range_neutral_trend_weight: float = 1.10
    broad_range_trend_norm: float = 2.00
    broad_range_inefficiency_weight: float = 0.60
    high_vol_expansion_bonus: float = 1.20
    high_vol_return_weight: float = 3.00
    high_vol_drawdown_weight: float = 2.00
    high_vol_shock_weight: float = 0.08
    recovery_drawdown_weight: float = 2.00
    recovery_short_return_weight: float = 4.00
    recovery_trend_weight: float = 2.00
    low_liquidity_volume_z_threshold: float = -0.75
    low_liquidity_volume_bonus: float = 0.70
    low_liquidity_efficiency_threshold: float = 0.25
    low_liquidity_efficiency_bonus: float = 0.70
    low_liquidity_abs_trend_threshold: float = 0.30
    low_liquidity_trend_bonus: float = 0.50
    event_shock_weight: float = 0.12
    event_volume_z_threshold: float = 3.00
    event_volume_bonus: float = 0.80
    crash_vol_ratio_min: float = 1.05
    crash_atr_pct_min: float = 0.035
    crash_shock_score_min: float = 2.00
    liquidation_shock_score: float = 6.00
    liquidation_vol_ratio: float = 1.20
    compression_vol_ratio_max: float = 0.95
    bull_pullback_range_position_min: float = 0.35
    bear_pullback_range_position_max: float = 0.65
    impulse_efficiency_min: float = 0.45
    volume_spike_z: float = 3.00
    wick_rejection_shock_score: float = 4.00
    wick_rejection_efficiency_max: float = 0.35
    thin_book_volume_z: float = -1.00
    thin_book_efficiency_max: float = 0.25
    noisy_chop_efficiency_max: float = 0.20
    trend_duration_efficiency_min: float = 0.45
    trend_exit_abs_score: float = 1.00


@dataclass(frozen=True)
class TransitionRiskScoringConfig:
    vol_ratio_threshold: float = 1.40
    vol_ratio_weight: float = 0.25
    bb_width_low_threshold: float = 0.20
    bb_width_high_threshold: float = 0.85
    bb_extreme_weight: float = 0.20
    unstable_meso_weight: float = 0.20
    unstable_macro_weight: float = 0.25
    chop_trend_abs_threshold: float = 0.35
    chop_vol_ratio_threshold: float = 1.10
    chop_vol_weight: float = 0.15
    shock_score_norm: float = 20.0
    shock_score_cap: float = 0.20
    crowding_weight: float = 0.15
    liquidation_spike_weight: float = 0.15
    orderflow_weight: float = 0.12
    weak_book_weight: float = 0.10
    trade_intensity_weight: float = 0.08
    risk_off_weight: float = 0.12
    fragmented_weight: float = 0.08
    leader_led_weight: float = 0.05
    leader_led_threshold: float = 0.20
    late_trend_weight: float = 0.12
    exhaustion_weight: float = 0.08
    exhaustion_threshold: float = 0.55
    state_transition_weight: float = 0.20
    state_transition_floor: float = 0.30
    change_point_weight: float = 0.18
    change_point_floor: float = 0.50
    entropy_weight: float = 0.08
    entropy_floor: float = 0.50


@dataclass(frozen=True)
class TradeabilityScoringConfig:
    base_score: float = 0.65
    constructive_meso_bonus: float = 0.15
    chop_or_compression_penalty: float = 0.25
    dangerous_macro_penalty: float = 0.35
    transition_risk_penalty: float = 0.25
    efficient_trend_bonus: float = 0.10
    efficient_trend_threshold: float = 0.55
    bad_micro_penalty: float = 0.15
    defensive_risk_penalty: float = 0.10
    crowding_penalty: float = 0.15
    orderflow_penalty: float = 0.12
    weak_book_penalty: float = 0.12
    risk_off_penalty: float = 0.15
    fragmented_penalty: float = 0.10
    late_trend_penalty: float = 0.12
    exhausted_trend_penalty: float = 0.22
    nearby_liquidity_penalty: float = 0.08
    nearby_liquidity_distance_pct: float = 0.005


@dataclass(frozen=True)
class RiskPostureConfig:
    weak_book_size_multiplier: float = 0.25
    stressed_orderflow_threshold: float = 0.75
    stressed_orderflow_transition_threshold: float = 0.35
    stressed_orderflow_size_multiplier: float = 0.35
    liquidation_size_multiplier: float = 0.20
    crash_size_multiplier: float = 0.20
    high_transition_threshold: float = 0.65
    high_transition_size_multiplier: float = 0.35
    risk_off_transition_threshold: float = 0.35
    risk_off_size_multiplier: float = 0.35
    crowding_threshold: float = 0.65
    crowding_size_multiplier: float = 0.50
    exhausted_trend_size_multiplier: float = 0.45
    compression_size_multiplier: float = 0.50
    aggressive_trend_score_threshold: float = 1.20
    aggressive_trend_efficiency_threshold: float = 0.55
    aggressive_size_multiplier: float = 1.15
    low_liquidity_size_multiplier: float = 0.25
    normal_size_multiplier: float = 0.75


@dataclass(frozen=True)
class RegimePlaybookRuleConfig:
    liquidity_entry_distance_pct: float = 0.01
    state_change_point_threshold: float = 0.65
    state_transition_threshold: float = 0.55
    bull_chase_range_position: float = 0.85
    bear_chase_range_position: float = 0.15


@dataclass(frozen=True)
class RegimeExplanationConfig:
    upper_range_position: float = 0.85
    lower_range_position: float = 0.15
    volatility_expansion_ratio: float = 1.50
    crowded_funding_abs: float = 0.0005
    change_point_threshold: float = 0.65
    state_transition_threshold: float = 0.55


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
    classification: RegimeClassificationConfig = field(default_factory=RegimeClassificationConfig)
    transition_scoring: TransitionRiskScoringConfig = field(default_factory=TransitionRiskScoringConfig)
    tradeability_scoring: TradeabilityScoringConfig = field(default_factory=TradeabilityScoringConfig)
    risk_posture: RiskPostureConfig = field(default_factory=RiskPostureConfig)
    playbook_rules: RegimePlaybookRuleConfig = field(default_factory=RegimePlaybookRuleConfig)
    explanation: RegimeExplanationConfig = field(default_factory=RegimeExplanationConfig)
    inference_mode: str = "rule"
    model_checkpoint: str = ""
    model_version: str = ""
    calibration_version: str = ""


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

    def __init__(
        self,
        cfg: Optional[RegimeIntelligenceConfig] = None,
        *,
        learned_model: object | None = None,
        learned_calibration: object | None = None,
    ) -> None:
        self.cfg = cfg or RegimeIntelligenceConfig()
        self.learned_model = learned_model
        self.learned_calibration = learned_calibration
        if self.learned_model is None and self.cfg.model_checkpoint:
            from zhisa.regime.learned import load_learned_regime_model

            self.learned_model = load_learned_regime_model(self.cfg.model_checkpoint)

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        t: Optional[int] = None,
        symbol: str = "",
        extra_context: Optional[dict] = None,
    ) -> RegimeReport:
        rule_report = self._analyze_rule(df, t=t, symbol=symbol, extra_context=extra_context)
        if self.learned_model is None or self.cfg.inference_mode == "rule":
            features = dict(rule_report.features)
            features.setdefault("inference_source", "rule_only")
            features.setdefault("rule_prior", rule_report.to_dict())
            features.setdefault("guardrail_overrides", [])
            explanation = {
                "why": list(rule_report.explanation.get("why", [])),
                "danger": list(rule_report.explanation.get("danger", [])),
                "guardrails": list(rule_report.explanation.get("guardrails", [])),
            }
            return RegimeReport(
                **{
                    **rule_report.to_dict(),
                    "features": features,
                    "explanation": explanation,
                }
            )
        from zhisa.regime.learned import RegimeDecisionAdapter, RegimeDecisionAdapterConfig, predict_learned_regime

        prediction = predict_learned_regime(
            self.learned_model,
            rule_report,
            calibration=self.learned_calibration,
        )
        adapter = RegimeDecisionAdapter(
            RegimeDecisionAdapterConfig(
                mode=self.cfg.inference_mode,
                model_version=self.cfg.model_version or self.cfg.model_checkpoint or "in_memory",
                calibration_version=self.cfg.calibration_version,
            )
        )
        return adapter.adapt(rule_report, prediction)

    def _analyze_rule(
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
        transition_base, transition_breakdown = self._transition_risk(primary, agg, macro, meso)
        transition_risk, context_transition_breakdown = self._context_transition_risk(
            transition_base,
            market_context,
            market_structure,
            state_space,
        )
        uncertainty = _clip01((1.0 - confidence) * 0.7 + transition_risk * 0.3)
        risk_mode, size_mult = self._risk_posture(macro, meso, transition_risk, primary, market_context, market_structure)
        allowed, blocked = self._playbooks(macro, meso, micro, risk_mode, primary, market_context, market_structure, state_space)
        stop_style, tp_style = self._exit_styles(macro, meso, primary)
        tradeability, tradeability_breakdown = self._tradeability(
            macro,
            meso,
            micro,
            risk_mode,
            transition_risk,
            primary,
            market_context,
            market_structure,
        )
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
                "scoring": {
                    "transition_risk": {
                        **transition_breakdown,
                        **context_transition_breakdown,
                        "score": transition_risk,
                    },
                    "tradeability": tradeability_breakdown,
                },
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
            extra_context=extra,
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
        cfg = self.cfg.classification
        trend = agg["trend_score"]
        eff = agg["trend_efficiency"]
        ret_l = agg["ret_long"]
        vol = agg["vol_ratio"]
        dd = agg["drawdown"]
        shock = agg["shock_score"]
        atr_pct = agg["atr_pct"]

        scores = {
            MacroRegime.BULL_TREND.value: cfg.bull_trend_weight * trend
            + cfg.bull_efficiency_weight * eff
            + cfg.bull_return_weight * max(ret_l, 0.0),
            MacroRegime.BEAR_TREND.value: -cfg.bear_trend_weight * trend
            + cfg.bear_efficiency_weight * eff
            + cfg.bear_return_weight * max(-ret_l, 0.0),
            MacroRegime.BROAD_RANGE.value: cfg.broad_range_neutral_trend_weight
            * (1.0 - min(abs(trend), cfg.broad_range_trend_norm) / max(cfg.broad_range_trend_norm, 1e-12))
            + cfg.broad_range_inefficiency_weight * (1.0 - eff),
            MacroRegime.HIGH_VOL_CRASH.value: cfg.high_vol_expansion_bonus * float(vol > self.cfg.expansion_vol_ratio)
            + cfg.high_vol_return_weight * max(-ret_l, 0.0)
            + cfg.high_vol_drawdown_weight * dd
            + cfg.high_vol_shock_weight * shock,
            MacroRegime.POST_CRASH_RECOVERY.value: cfg.recovery_drawdown_weight * dd
            + cfg.recovery_short_return_weight * max(agg["ret_short"], 0.0)
            + cfg.recovery_trend_weight * max(trend, 0.0),
            MacroRegime.LOW_LIQUIDITY_CHOP.value: cfg.low_liquidity_volume_bonus
            * float(agg["volume_z"] < cfg.low_liquidity_volume_z_threshold)
            + cfg.low_liquidity_efficiency_bonus * float(eff < cfg.low_liquidity_efficiency_threshold)
            + cfg.low_liquidity_trend_bonus * float(abs(trend) < cfg.low_liquidity_abs_trend_threshold),
            MacroRegime.EVENT_DRIVEN.value: cfg.event_shock_weight * shock
            + cfg.event_volume_bonus * float(abs(agg["volume_z"]) > cfg.event_volume_z_threshold),
        }
        probs = _softmax(scores)

        if (
            ret_l <= self.cfg.crash_return_threshold
            or dd >= self.cfg.crash_drawdown_threshold
        ) and (
            vol >= cfg.crash_vol_ratio_min
            or atr_pct >= cfg.crash_atr_pct_min
            or shock >= cfg.crash_shock_score_min
        ):
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
        cfg = self.cfg.classification
        if primary.shock_score > cfg.liquidation_shock_score and primary.ret_short < 0 and primary.vol_ratio > cfg.liquidation_vol_ratio:
            return MesoRegime.LIQUIDATION_CASCADE
        if primary.liquidity_sweep_high or primary.liquidity_sweep_low:
            return MesoRegime.FAILED_BREAKOUT
        if primary.bb_width_quantile <= self.cfg.compression_quantile and primary.vol_ratio < cfg.compression_vol_ratio_max:
            return MesoRegime.COMPRESSION
        if primary.vol_ratio >= self.cfg.expansion_vol_ratio:
            return MesoRegime.EXPANSION
        if primary.breakout_up or primary.breakout_down:
            return MesoRegime.BREAKOUT
        if macro == MacroRegime.BULL_TREND and primary.ret_short < 0 and primary.range_position > cfg.bull_pullback_range_position_min:
            return MesoRegime.PULLBACK
        if macro == MacroRegime.BEAR_TREND and primary.ret_short > 0 and primary.range_position < cfg.bear_pullback_range_position_max:
            return MesoRegime.PULLBACK
        if abs(agg["trend_score"]) >= self.cfg.strong_trend_threshold and agg["trend_efficiency"] > cfg.impulse_efficiency_min:
            return MesoRegime.IMPULSE
        if abs(agg["trend_score"]) < self.cfg.range_trend_threshold:
            return MesoRegime.CHOP
        return MesoRegime.ACCUMULATION if macro != MacroRegime.BEAR_TREND else MesoRegime.DISTRIBUTION

    def _classify_micro(self, primary: RegimeFeatures) -> MicroRegime:
        cfg = self.cfg.classification
        if primary.liquidity_sweep_high or primary.liquidity_sweep_low:
            return MicroRegime.STOP_RUN
        if abs(primary.volume_z) >= cfg.volume_spike_z:
            return MicroRegime.VOLUME_SPIKE
        if primary.shock_score >= cfg.wick_rejection_shock_score and primary.trend_efficiency < cfg.wick_rejection_efficiency_max:
            return MicroRegime.WICK_REJECTION
        if primary.volume_z < cfg.thin_book_volume_z and primary.trend_efficiency < cfg.thin_book_efficiency_max:
            return MicroRegime.THIN_BOOK
        if primary.trend_efficiency < cfg.noisy_chop_efficiency_max:
            return MicroRegime.NOISY_CHOP
        return MicroRegime.QUIET

    def _transition_risk(
        self,
        primary: RegimeFeatures,
        agg: dict[str, float],
        macro: MacroRegime,
        meso: MesoRegime,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.cfg.transition_scoring
        breakdown = {
            "base.volatility_expansion": cfg.vol_ratio_weight * float(primary.vol_ratio > cfg.vol_ratio_threshold),
            "base.bb_width_extreme": cfg.bb_extreme_weight
            * float(primary.bb_width_quantile < cfg.bb_width_low_threshold or primary.bb_width_quantile > cfg.bb_width_high_threshold),
            "base.unstable_meso": cfg.unstable_meso_weight
            * float(meso in {MesoRegime.FAILED_BREAKOUT, MesoRegime.LIQUIDATION_CASCADE}),
            "base.unstable_macro": cfg.unstable_macro_weight
            * float(macro in {MacroRegime.HIGH_VOL_CRASH, MacroRegime.EVENT_DRIVEN}),
            "base.chop_with_volatility": cfg.chop_vol_weight
            * float(abs(agg["trend_score"]) < cfg.chop_trend_abs_threshold and agg["vol_ratio"] > cfg.chop_vol_ratio_threshold),
            "base.shock_score": min(primary.shock_score / max(cfg.shock_score_norm, 1e-12), cfg.shock_score_cap),
        }
        raw = sum(breakdown.values())
        breakdown["base.raw"] = float(raw)
        breakdown["base.score"] = _clip01(raw)
        return breakdown["base.score"], breakdown

    def _context_transition_risk(
        self,
        base: float,
        context: MarketContextReport,
        structure: MarketStructureReport,
        state_space: StateSpaceReport,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.cfg.transition_scoring
        crowding = context.crowding
        corr = context.correlation
        orderflow = context.orderflow
        breakdown = {
            "context.crowding": cfg.crowding_weight * crowding.crowding_score,
            "context.liquidation_spike": cfg.liquidation_spike_weight * float("liquidation_spike" in crowding.flags),
            "context.orderflow": cfg.orderflow_weight * orderflow.orderflow_score,
            "context.weak_book": cfg.weak_book_weight * float("wide_spread" in orderflow.flags or "thin_depth" in orderflow.flags),
            "context.trade_intensity": cfg.trade_intensity_weight * float("trade_intensity_spike" in orderflow.flags),
            "context.risk_off_sync": cfg.risk_off_weight * float(corr.regime == "risk_off_sync"),
            "context.fragmented": cfg.fragmented_weight * float(corr.regime == "fragmented"),
            "context.leader_led": cfg.leader_led_weight
            * float(corr.regime in {"benchmark_led", "leader_led"} and corr.leader_lead_score > cfg.leader_led_threshold),
            "structure.late_trend": cfg.late_trend_weight * float(structure.trend.phase in {"late", "exhausted"}),
            "structure.exhaustion": cfg.exhaustion_weight * float(structure.trend.exhaustion_score > cfg.exhaustion_threshold),
            "state.transition_probability": cfg.state_transition_weight
            * max(0.0, state_space.transition_probability - cfg.state_transition_floor),
            "state.change_point": cfg.change_point_weight * max(0.0, state_space.change_point_score - cfg.change_point_floor),
            "state.entropy": cfg.entropy_weight * max(0.0, state_space.entropy - cfg.entropy_floor),
        }
        raw = float(base) + sum(breakdown.values())
        breakdown["context.raw"] = float(raw)
        breakdown["context.score"] = _clip01(raw)
        return breakdown["context.score"], breakdown

    def _risk_posture(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        transition_risk: float,
        primary: RegimeFeatures,
        context: MarketContextReport,
        structure: MarketStructureReport,
    ) -> tuple[RiskMode, float]:
        cfg = self.cfg.risk_posture
        if "wide_spread" in context.orderflow.flags or "thin_depth" in context.orderflow.flags:
            return RiskMode.DEFENSIVE, cfg.weak_book_size_multiplier
        if (
            context.orderflow.orderflow_score > cfg.stressed_orderflow_threshold
            and transition_risk > cfg.stressed_orderflow_transition_threshold
        ):
            return RiskMode.REDUCED, cfg.stressed_orderflow_size_multiplier
        if "liquidation_spike" in context.crowding.flags:
            return RiskMode.DEFENSIVE, cfg.liquidation_size_multiplier
        if macro == MacroRegime.HIGH_VOL_CRASH or meso == MesoRegime.LIQUIDATION_CASCADE:
            return RiskMode.DEFENSIVE, cfg.crash_size_multiplier
        if transition_risk > cfg.high_transition_threshold or macro == MacroRegime.EVENT_DRIVEN:
            return RiskMode.REDUCED, cfg.high_transition_size_multiplier
        if context.correlation.regime == "risk_off_sync" and transition_risk > cfg.risk_off_transition_threshold:
            return RiskMode.REDUCED, cfg.risk_off_size_multiplier
        if context.crowding.crowding_score > cfg.crowding_threshold:
            return RiskMode.REDUCED, cfg.crowding_size_multiplier
        if structure.trend.phase == "exhausted":
            return RiskMode.REDUCED, cfg.exhausted_trend_size_multiplier
        if meso == MesoRegime.COMPRESSION:
            return RiskMode.REDUCED, cfg.compression_size_multiplier
        if (
            abs(primary.trend_score) > cfg.aggressive_trend_score_threshold
            and primary.trend_efficiency > cfg.aggressive_trend_efficiency_threshold
        ):
            return RiskMode.AGGRESSIVE, cfg.aggressive_size_multiplier
        if macro == MacroRegime.LOW_LIQUIDITY_CHOP:
            return RiskMode.DEFENSIVE, cfg.low_liquidity_size_multiplier
        return RiskMode.NORMAL, cfg.normal_size_multiplier

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
        cfg = self.cfg.playbook_rules
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
        if context.orderflow.direction == "buy_pressure":
            allowed.add("orderflow_confirmed_long")
            blocked.add("short_without_orderflow_confirmation")
        if context.orderflow.direction == "sell_pressure":
            allowed.add("orderflow_confirmed_short")
            blocked.add("long_without_orderflow_confirmation")
        if "wide_spread" in context.orderflow.flags or "thin_depth" in context.orderflow.flags:
            allowed.add("thin_book_wait")
            blocked.update({"market_order_entry", "fresh_full_size_entry"})
        if context.correlation.regime == "fragmented":
            allowed.add("relative_strength_only")
            blocked.add("market_beta_chase")
        if context.correlation.regime == "risk_off_sync":
            blocked.update({"full_size_long", "correlation_blind_long"})
        if structure.trend.phase in {"late", "exhausted"}:
            allowed.add("pullback_to_value_only")
            blocked.update({"late_trend_chase", "full_size_breakout_chase"})
        if (
            structure.liquidity.nearest_level is not None
            and abs(structure.liquidity.nearest_level.distance_pct) < cfg.liquidity_entry_distance_pct
        ):
            blocked.add("entry_directly_into_liquidity")
        if structure.liquidity.in_value_area:
            allowed.add("value_area_reversion")
        if (
            state_space.change_point_score > cfg.state_change_point_threshold
            or state_space.transition_probability > cfg.state_transition_threshold
        ):
            allowed.add("transition_wait")
            blocked.update({"fresh_full_size_entry", "regime_transition_chase"})
        if primary.range_position > cfg.bull_chase_range_position and macro == MacroRegime.BULL_TREND:
            blocked.add("chase_long_at_high")
        if primary.range_position < cfg.bear_chase_range_position and macro == MacroRegime.BEAR_TREND:
            blocked.add("chase_short_at_low")
        return sorted(allowed), sorted(blocked)

    def _exit_styles(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        primary: RegimeFeatures,
    ) -> tuple[str, str]:
        cfg = self.cfg.classification
        if macro == MacroRegime.HIGH_VOL_CRASH:
            return "wide_volatility_or_no_trade", "fast_partial_exits"
        if meso in {MesoRegime.COMPRESSION, MesoRegime.CHOP}:
            return "tight_invalidation", "wait_for_expansion"
        if abs(primary.trend_score) > cfg.trend_exit_abs_score:
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
    ) -> tuple[float, dict[str, float]]:
        cfg = self.cfg.tradeability_scoring
        nearby_liquidity = (
            structure.liquidity.nearest_level is not None
            and abs(structure.liquidity.nearest_level.distance_pct) < cfg.nearby_liquidity_distance_pct
        )
        breakdown = {
            "base": cfg.base_score,
            "constructive_meso": cfg.constructive_meso_bonus
            * float(meso in {MesoRegime.PULLBACK, MesoRegime.BREAKOUT, MesoRegime.IMPULSE}),
            "chop_or_compression": -cfg.chop_or_compression_penalty
            * float(meso in {MesoRegime.CHOP, MesoRegime.COMPRESSION}),
            "dangerous_macro": -cfg.dangerous_macro_penalty
            * float(macro in {MacroRegime.HIGH_VOL_CRASH, MacroRegime.LOW_LIQUIDITY_CHOP}),
            "transition_risk": -cfg.transition_risk_penalty * transition_risk,
            "efficient_trend": cfg.efficient_trend_bonus * float(primary.trend_efficiency > cfg.efficient_trend_threshold),
            "bad_micro": -cfg.bad_micro_penalty * float(micro in {MicroRegime.THIN_BOOK, MicroRegime.NOISY_CHOP}),
            "defensive_risk_mode": -cfg.defensive_risk_penalty * float(risk_mode == RiskMode.DEFENSIVE),
            "crowding": -cfg.crowding_penalty * context.crowding.crowding_score,
            "orderflow": -cfg.orderflow_penalty * context.orderflow.orderflow_score,
            "weak_book": -cfg.weak_book_penalty
            * float("wide_spread" in context.orderflow.flags or "thin_depth" in context.orderflow.flags),
            "risk_off_sync": -cfg.risk_off_penalty * float(context.correlation.regime == "risk_off_sync"),
            "fragmented": -cfg.fragmented_penalty * float(context.correlation.regime == "fragmented"),
            "late_trend": -cfg.late_trend_penalty * float(structure.trend.phase == "late"),
            "exhausted_trend": -cfg.exhausted_trend_penalty * float(structure.trend.phase == "exhausted"),
            "nearby_liquidity": -cfg.nearby_liquidity_penalty * float(nearby_liquidity),
        }
        raw = sum(breakdown.values())
        breakdown["raw"] = float(raw)
        breakdown["score"] = _clip01(raw)
        return breakdown["score"], breakdown

    def _expected_duration(
        self,
        macro: MacroRegime,
        meso: MesoRegime,
        primary: RegimeFeatures,
    ) -> ExpectedDuration:
        cfg = self.cfg.classification
        if meso in {MesoRegime.LIQUIDATION_CASCADE, MesoRegime.FAILED_BREAKOUT}:
            return ExpectedDuration.VERY_SHORT
        if meso in {MesoRegime.EXPANSION, MesoRegime.BREAKOUT}:
            return ExpectedDuration.SHORT
        if (
            macro in {MacroRegime.BULL_TREND, MacroRegime.BEAR_TREND}
            and primary.trend_efficiency > cfg.trend_duration_efficiency_min
        ):
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
        cfg = self.cfg.explanation
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
        if primary.range_position > cfg.upper_range_position:
            danger.append("price near upper range / liquidity high")
        if primary.range_position < cfg.lower_range_position:
            danger.append("price near lower range / liquidity low")
        if primary.vol_ratio > cfg.volatility_expansion_ratio:
            danger.append("volatility expansion raises transition risk")
        funding = extra.get("funding")
        if funding is not None and abs(float(funding)) > cfg.crowded_funding_abs:
            danger.append("funding is crowded")
        crowding = context.crowding
        corr = context.correlation
        orderflow = context.orderflow
        if crowding.flags:
            why.append(f"crowding={crowding.direction}, score={crowding.crowding_score:.2f}")
        if crowding.direction != "neutral":
            danger.append(f"derivatives crowding is {crowding.direction}")
        if "open_interest_fast_change" in crowding.flags:
            danger.append("open interest is changing quickly")
        if "liquidation_spike" in crowding.flags:
            danger.append("liquidation spike / forced flow detected")
        if orderflow.flags:
            why.append(
                f"orderflow={orderflow.direction}, score={orderflow.orderflow_score:.2f}, "
                f"delta_z={orderflow.delta_z:.2f}, spread_bps={orderflow.spread_bps:.2f}"
            )
        if orderflow.direction in {"buy_pressure", "sell_pressure"}:
            danger.append(f"orderflow pressure is {orderflow.direction}")
        if "wide_spread" in orderflow.flags:
            danger.append("spread is wide; execution quality is degraded")
        if "thin_depth" in orderflow.flags:
            danger.append("book depth is thin; slippage risk is elevated")
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
        if state_space.change_point_score > cfg.change_point_threshold:
            danger.append("unsupervised change-point score is elevated")
        if state_space.transition_probability > cfg.state_transition_threshold:
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


__all__ = [
    "RegimeClassificationConfig",
    "RegimeExplanationConfig",
    "RegimeIntelligence",
    "RegimeIntelligenceConfig",
    "RegimePlaybookRuleConfig",
    "RiskPostureConfig",
    "TradeabilityScoringConfig",
    "TransitionRiskScoringConfig",
]
