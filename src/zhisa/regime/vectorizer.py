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


def _enum_values(values: Iterable) -> tuple[str, ...]:
    return tuple(str(v.value) for v in values)


@dataclass(frozen=True)
class RegimeVectorizerConfig:
    scalar_fields: tuple[str, ...] = BASE_SCALAR_FIELDS
    aggregate_fields: tuple[str, ...] = AGGREGATE_FIELDS
    macro_classes: tuple[str, ...] = field(default_factory=lambda: _enum_values(MacroRegime))
    meso_classes: tuple[str, ...] = field(default_factory=lambda: _enum_values(MesoRegime))
    micro_classes: tuple[str, ...] = field(default_factory=lambda: _enum_values(MicroRegime))
    risk_modes: tuple[str, ...] = field(default_factory=lambda: _enum_values(RiskMode))
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


class RegimeFeatureVectorizer:
    """Convert a structured regime report into a fixed numeric vector."""

    def __init__(self, cfg: RegimeVectorizerConfig | None = None) -> None:
        self.cfg = cfg or RegimeVectorizerConfig()

    @property
    def feature_names(self) -> list[str]:
        names = [f"scalar.{name}" for name in self.cfg.scalar_fields]
        names += [f"aggregate.{name}" for name in self.cfg.aggregate_fields]
        names += [f"macro.{name}" for name in self.cfg.macro_classes]
        names += [f"meso.{name}" for name in self.cfg.meso_classes]
        names += [f"micro.{name}" for name in self.cfg.micro_classes]
        names += [f"risk_mode.{name}" for name in self.cfg.risk_modes]
        if self.cfg.include_probabilities:
            names += [f"probability.{name}" for name in self.cfg.macro_classes]
        return names

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def transform(self, report: RegimeReport) -> np.ndarray:
        cfg = self.cfg
        aggregate = report.features.get("aggregate", {}) if report.features else {}
        values: list[float] = []
        for name in cfg.scalar_fields:
            values.append(_finite_float(getattr(report, name, 0.0), clip_abs=cfg.clip_abs))
        for name in cfg.aggregate_fields:
            values.append(_finite_float(aggregate.get(name, 0.0), clip_abs=cfg.clip_abs))
        values.extend(_one_hot(report.primary_regime, cfg.macro_classes))
        values.extend(_one_hot(report.secondary_regime, cfg.meso_classes))
        values.extend(_one_hot(report.micro_regime, cfg.micro_classes))
        values.extend(_one_hot(report.risk_mode, cfg.risk_modes))
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
    "RegimeFeatureVectorizer",
    "RegimeVectorizerConfig",
]
