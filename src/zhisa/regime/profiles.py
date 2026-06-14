"""Named regime-intelligence profiles for different market microstructures."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from zhisa.regime.context import (
    CrowdingScoringConfig,
    MarketContextConfig,
    OrderflowScoringConfig,
)
from zhisa.regime.detector import (
    RegimeClassificationConfig,
    RegimeExplanationConfig,
    RegimeIntelligenceConfig,
    RegimePlaybookRuleConfig,
    RiskPostureConfig,
    TradeabilityScoringConfig,
    TransitionRiskScoringConfig,
)
from zhisa.regime.structure import StructureConfig, TrendScoringConfig


@dataclass(frozen=True)
class RegimeProfile:
    name: str
    description: str
    asset_class: str
    config: RegimeIntelligenceConfig

    def with_overrides(
        self,
        *,
        source_timeframe: str | None = None,
        timeframes: tuple[str, ...] | None = None,
        benchmark_symbol: str | None = None,
    ) -> RegimeIntelligenceConfig:
        cfg = self.config
        if source_timeframe is not None:
            cfg = replace(cfg, source_timeframe=source_timeframe)
        if timeframes is not None:
            cfg = replace(cfg, timeframes=timeframes)
        if benchmark_symbol is not None:
            cfg = replace(cfg, context=replace(cfg.context, benchmark_symbol=benchmark_symbol))
        return cfg


def _crypto_perp_config() -> RegimeIntelligenceConfig:
    context = MarketContextConfig(
        benchmark_symbol="BTC/USDT",
        benchmark_aliases=("BTC", "BTCUSDT", "BTC/USDT"),
        high_funding_abs=0.00045,
        liquidation_z_threshold=2.25,
        wide_spread_bps=7.0,
        crowding_scoring=CrowdingScoringConfig(
            funding_abs_weight=0.28,
            funding_z_weight=0.22,
            open_interest_weight=0.22,
            liquidation_weight=0.18,
            warning_score_threshold=0.60,
        ),
        orderflow_scoring=OrderflowScoringConfig(
            buy_sell_weight=0.27,
            delta_z_weight=0.22,
            spread_weight=0.12,
            thin_depth_weight=0.14,
            warning_score_threshold=0.60,
        ),
    )
    return RegimeIntelligenceConfig(
        context=context,
        classification=RegimeClassificationConfig(
            liquidation_shock_score=5.25,
            liquidation_vol_ratio=1.15,
            event_shock_weight=0.15,
            crash_shock_score_min=1.75,
        ),
        transition_scoring=TransitionRiskScoringConfig(
            crowding_weight=0.18,
            liquidation_spike_weight=0.18,
            orderflow_weight=0.14,
            weak_book_weight=0.12,
            state_transition_weight=0.22,
        ),
        tradeability_scoring=TradeabilityScoringConfig(
            crowding_penalty=0.18,
            orderflow_penalty=0.14,
            weak_book_penalty=0.14,
            transition_risk_penalty=0.28,
        ),
        risk_posture=RiskPostureConfig(
            crowding_threshold=0.60,
            high_transition_threshold=0.60,
            stressed_orderflow_threshold=0.68,
            normal_size_multiplier=0.70,
        ),
    )


def _btc_intraday_config() -> RegimeIntelligenceConfig:
    base = _crypto_perp_config()
    return replace(
        base,
        trend_threshold=0.60,
        strong_trend_threshold=1.05,
        classification=replace(
            base.classification,
            bull_trend_weight=1.00,
            bear_trend_weight=1.00,
            impulse_efficiency_min=0.42,
        ),
        transition_scoring=replace(
            base.transition_scoring,
            leader_led_weight=0.02,
            fragmented_weight=0.05,
        ),
        risk_posture=replace(
            base.risk_posture,
            aggressive_trend_score_threshold=1.10,
            aggressive_size_multiplier=1.10,
        ),
    )


def _high_beta_alt_config() -> RegimeIntelligenceConfig:
    base = _crypto_perp_config()
    return replace(
        base,
        min_confidence=0.40,
        crash_return_threshold=-0.045,
        crash_drawdown_threshold=0.10,
        context=replace(
            base.context,
            fragmented_correlation_threshold=0.45,
            lead_score_threshold=0.08,
            wide_spread_bps=9.0,
            orderflow_scoring=replace(
                base.context.orderflow_scoring,
                spread_weight=0.15,
                thin_depth_weight=0.16,
            ),
        ),
        classification=replace(
            base.classification,
            high_vol_return_weight=3.40,
            low_liquidity_volume_bonus=0.85,
            low_liquidity_efficiency_threshold=0.30,
        ),
        transition_scoring=replace(
            base.transition_scoring,
            fragmented_weight=0.11,
            leader_led_weight=0.08,
            weak_book_weight=0.14,
            state_transition_floor=0.25,
        ),
        tradeability_scoring=replace(
            base.tradeability_scoring,
            fragmented_penalty=0.14,
            weak_book_penalty=0.16,
            transition_risk_penalty=0.32,
        ),
        risk_posture=replace(
            base.risk_posture,
            normal_size_multiplier=0.60,
            crowding_size_multiplier=0.40,
            weak_book_size_multiplier=0.20,
        ),
        playbook_rules=RegimePlaybookRuleConfig(
            liquidity_entry_distance_pct=0.012,
            bull_chase_range_position=0.82,
            bear_chase_range_position=0.18,
        ),
    )


def _equity_intraday_config() -> RegimeIntelligenceConfig:
    context = MarketContextConfig(
        benchmark_symbol="SPY",
        benchmark_aliases=("SPY", "QQQ", "IWM"),
        high_funding_abs=0.0020,
        high_long_short_ratio=1.60,
        low_long_short_ratio=0.65,
        oi_change_threshold=0.12,
        liquidation_z_threshold=3.5,
        high_correlation_threshold=0.58,
        fragmented_correlation_threshold=0.30,
        lead_score_threshold=0.10,
        wide_spread_bps=10.0,
        orderflow_imbalance_threshold=0.25,
        crowding_scoring=CrowdingScoringConfig(
            funding_abs_weight=0.05,
            funding_z_weight=0.05,
            long_short_weight=0.12,
            open_interest_weight=0.18,
            liquidation_weight=0.04,
            warning_score_threshold=0.72,
        ),
        orderflow_scoring=OrderflowScoringConfig(
            bid_ask_weight=0.24,
            buy_sell_weight=0.28,
            delta_z_weight=0.22,
            spread_weight=0.16,
            thin_depth_weight=0.10,
        ),
    )
    return RegimeIntelligenceConfig(
        source_timeframe="5m",
        timeframes=("5m", "15m", "1h"),
        context=context,
        classification=RegimeClassificationConfig(
            high_vol_return_weight=2.50,
            event_volume_z_threshold=3.50,
            compression_vol_ratio_max=0.90,
            volume_spike_z=3.50,
            trend_duration_efficiency_min=0.40,
        ),
        structure=StructureConfig(
            trend_scoring=TrendScoringConfig(
                mature_threshold=0.70,
                late_threshold=0.82,
                phase_mature_threshold=0.50,
            )
        ),
        transition_scoring=TransitionRiskScoringConfig(
            crowding_weight=0.06,
            liquidation_spike_weight=0.04,
            orderflow_weight=0.14,
            weak_book_weight=0.12,
            risk_off_weight=0.16,
            fragmented_weight=0.10,
            late_trend_weight=0.10,
        ),
        tradeability_scoring=TradeabilityScoringConfig(
            crowding_penalty=0.05,
            orderflow_penalty=0.13,
            risk_off_penalty=0.18,
            fragmented_penalty=0.12,
            transition_risk_penalty=0.24,
        ),
        risk_posture=RiskPostureConfig(
            high_transition_threshold=0.62,
            risk_off_transition_threshold=0.30,
            crowding_threshold=0.78,
            normal_size_multiplier=0.70,
        ),
        explanation=RegimeExplanationConfig(crowded_funding_abs=0.0020),
    )


_PROFILES: dict[str, RegimeProfile] = {
    "default": RegimeProfile(
        name="default",
        description="Balanced cross-asset regime settings.",
        asset_class="generic",
        config=RegimeIntelligenceConfig(),
    ),
    "crypto_perp": RegimeProfile(
        name="crypto_perp",
        description="Crypto perpetuals with derivatives crowding and forced-flow sensitivity.",
        asset_class="crypto",
        config=_crypto_perp_config(),
    ),
    "btc_intraday": RegimeProfile(
        name="btc_intraday",
        description="BTC-like intraday trend/liquidity behavior without assuming every symbol is BTC.",
        asset_class="crypto",
        config=_btc_intraday_config(),
    ),
    "high_beta_alt": RegimeProfile(
        name="high_beta_alt",
        description="Higher-beta altcoins with stronger benchmark, liquidity, and transition penalties.",
        asset_class="crypto",
        config=_high_beta_alt_config(),
    ),
    "equity_intraday": RegimeProfile(
        name="equity_intraday",
        description="Equity/ETF intraday profile using broad-market correlation instead of crypto funding.",
        asset_class="equity",
        config=_equity_intraday_config(),
    ),
}


def list_regime_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_regime_profile(name: str) -> RegimeProfile:
    key = str(name).strip().lower()
    if key not in _PROFILES:
        valid = ", ".join(list_regime_profiles())
        raise KeyError(f"unknown regime profile '{name}', valid profiles: {valid}")
    return _PROFILES[key]


def build_regime_profile_config(
    profile: str | RegimeProfile,
    *,
    source_timeframe: str | None = None,
    timeframes: tuple[str, ...] | None = None,
    benchmark_symbol: str | None = None,
) -> RegimeIntelligenceConfig:
    selected = get_regime_profile(profile) if isinstance(profile, str) else profile
    return selected.with_overrides(
        source_timeframe=source_timeframe,
        timeframes=timeframes,
        benchmark_symbol=benchmark_symbol,
    )


def resolve_regime_profile(
    *,
    symbol: str = "",
    asset_class: str = "",
    venue: str = "",
    metadata: Mapping[str, object] | None = None,
) -> RegimeProfile:
    meta = {str(k).lower(): str(v).lower() for k, v in (metadata or {}).items()}
    asset = (asset_class or meta.get("asset_class", "")).lower()
    venue_l = (venue or meta.get("venue", "")).lower()
    symbol_u = symbol.upper()
    if asset in {"equity", "stock", "etf", "index"}:
        return get_regime_profile("equity_intraday")
    if asset in {"crypto", "perp", "future", "futures"} or any(x in symbol_u for x in ("USDT", "USDC", "PERP")):
        if "BTC" in symbol_u:
            return get_regime_profile("btc_intraday")
        if "perp" in venue_l or meta.get("contract_type") == "perpetual":
            return get_regime_profile("crypto_perp")
        return get_regime_profile("high_beta_alt")
    return get_regime_profile("default")


__all__ = [
    "RegimeProfile",
    "build_regime_profile_config",
    "get_regime_profile",
    "list_regime_profiles",
    "resolve_regime_profile",
]
