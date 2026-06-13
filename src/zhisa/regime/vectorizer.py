"""Stable vector representation for regime reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from zhisa.regime.schema import MacroRegime, MesoRegime, MicroRegime, RegimeReport, RiskMode


BASE_SCALAR_FIELDS: tuple[str, ...] = (
    "confidence",
    "uncertainty",
    "transition_risk",
    "tradeability_score",
    "position_size_multiplier",
)

AGGREGATE_FIELDS: tuple[str, ...] = (
    "trend_score",
    "trend_efficiency",
    "ret_short",
    "ret_medium",
    "ret_long",
    "vol_ratio",
    "bb_width_quantile",
    "atr_pct",
    "volume_z",
    "range_position",
    "drawdown",
    "shock_score",
)

CONTEXT_SCALAR_FIELDS: tuple[str, ...] = (
    "crowding.funding",
    "crowding.funding_z",
    "crowding.open_interest_change",
    "crowding.long_short_ratio",
    "crowding.long_short_z",
    "crowding.liquidation_z",
    "crowding.crowding_score",
    "correlation.avg_correlation",
    "correlation.leader_lead_score",
    "correlation.market_breadth",
    "correlation.dispersion",
    "structure.trend.maturity_score",
    "structure.trend.exhaustion_score",
    "structure.trend.extension_atr",
    "structure.trend.pullback_risk",
    "structure.liquidity.distance_to_value_mid_pct",
    "state_space.current_state",
    "state_space.state_probability",
    "state_space.transition_probability",
    "state_space.change_point_score",
    "state_space.entropy",
)

CROWDING_DIRECTIONS: tuple[str, ...] = ("neutral", "long_crowded", "short_crowded")
CORRELATION_REGIMES: tuple[str, ...] = (
    "single_asset",
    "benchmark_led",
    "leader_led",
    "alt_led",
    "risk_on_sync",
    "risk_off_sync",
    "fragmented",
    "mixed",
)
TREND_PHASES: tuple[str, ...] = ("none", "early", "mature", "late", "exhausted")


def _enum_values(values: Iterable) -> tuple[str, ...]:
    return tuple(str(v.value) for v in values)


@dataclass(frozen=True)
class RegimeVectorizerConfig:
    scalar_fields: tuple[str, ...] = BASE_SCALAR_FIELDS
    aggregate_fields: tuple[str, ...] = AGGREGATE_FIELDS
    context_scalar_fields: tuple[str, ...] = CONTEXT_SCALAR_FIELDS
    macro_classes: tuple[str, ...] = field(default_factory=lambda: _enum_values(MacroRegime))
    meso_classes: tuple[str, ...] = field(default_factory=lambda: _enum_values(MesoRegime))
    micro_classes: tuple[str, ...] = field(default_factory=lambda: _enum_values(MicroRegime))
    risk_modes: tuple[str, ...] = field(default_factory=lambda: _enum_values(RiskMode))
    crowding_directions: tuple[str, ...] = CROWDING_DIRECTIONS
    correlation_regimes: tuple[str, ...] = CORRELATION_REGIMES
    trend_phases: tuple[str, ...] = TREND_PHASES
    include_probabilities: bool = True
    clip_abs: float = 10.0


def _finite_float(value: object, *, clip_abs: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return float(np.clip(out, -clip_abs, clip_abs))


def _one_hot(value: str, classes: Sequence[str]) -> list[float]:
    return [1.0 if value == cls else 0.0 for cls in classes]


def _nested_get(data: dict, path: str, default: object = 0.0) -> object:
    cur: object = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


class RegimeFeatureVectorizer:
    """Convert a structured regime report into a fixed numeric vector."""

    def __init__(self, cfg: RegimeVectorizerConfig | None = None) -> None:
        self.cfg = cfg or RegimeVectorizerConfig()

    @property
    def feature_names(self) -> list[str]:
        names = [f"scalar.{name}" for name in self.cfg.scalar_fields]
        names += [f"aggregate.{name}" for name in self.cfg.aggregate_fields]
        names += [f"context.{name}" for name in self.cfg.context_scalar_fields]
        names += [f"macro.{name}" for name in self.cfg.macro_classes]
        names += [f"meso.{name}" for name in self.cfg.meso_classes]
        names += [f"micro.{name}" for name in self.cfg.micro_classes]
        names += [f"risk_mode.{name}" for name in self.cfg.risk_modes]
        names += [f"crowding_direction.{name}" for name in self.cfg.crowding_directions]
        names += [f"correlation_regime.{name}" for name in self.cfg.correlation_regimes]
        names += [f"trend_phase.{name}" for name in self.cfg.trend_phases]
        if self.cfg.include_probabilities:
            names += [f"probability.{name}" for name in self.cfg.macro_classes]
        return names

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def transform(self, report: RegimeReport) -> np.ndarray:
        cfg = self.cfg
        aggregate = report.features.get("aggregate", {}) if report.features else {}
        market_context = report.features.get("market_context", {}) if report.features else {}
        market_structure = report.features.get("market_structure", {}) if report.features else {}
        state_space = report.features.get("state_space", {}) if report.features else {}
        context_root = {**market_context, "structure": market_structure, "state_space": state_space}
        values: list[float] = []
        for name in cfg.scalar_fields:
            values.append(_finite_float(getattr(report, name, 0.0), clip_abs=cfg.clip_abs))
        for name in cfg.aggregate_fields:
            values.append(_finite_float(aggregate.get(name, 0.0), clip_abs=cfg.clip_abs))
        for name in cfg.context_scalar_fields:
            values.append(_finite_float(_nested_get(context_root, name), clip_abs=cfg.clip_abs))
        values.extend(_one_hot(report.primary_regime, cfg.macro_classes))
        values.extend(_one_hot(report.secondary_regime, cfg.meso_classes))
        values.extend(_one_hot(report.micro_regime, cfg.micro_classes))
        values.extend(_one_hot(report.risk_mode, cfg.risk_modes))
        values.extend(_one_hot(str(_nested_get(market_context, "crowding.direction", "neutral")), cfg.crowding_directions))
        values.extend(_one_hot(str(_nested_get(market_context, "correlation.regime", "single_asset")), cfg.correlation_regimes))
        values.extend(_one_hot(str(getattr(report, "trend_phase", _nested_get(market_structure, "trend.phase", "none"))), cfg.trend_phases))
        if cfg.include_probabilities:
            for name in cfg.macro_classes:
                values.append(_finite_float(report.probabilities.get(name, 0.0), clip_abs=cfg.clip_abs))
        return np.asarray(values, dtype=np.float32)

    def transform_many(self, reports: Sequence[RegimeReport]) -> np.ndarray:
        if not reports:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.transform(report) for report in reports], axis=0)


__all__ = [
    "AGGREGATE_FIELDS",
    "BASE_SCALAR_FIELDS",
    "CONTEXT_SCALAR_FIELDS",
    "CORRELATION_REGIMES",
    "CROWDING_DIRECTIONS",
    "TREND_PHASES",
    "RegimeFeatureVectorizer",
    "RegimeVectorizerConfig",
]
