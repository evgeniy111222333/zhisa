"""Causal feature-engine layer for regime intelligence without final decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from zhisa.regime.context import MarketContextAnalyzer, MarketContextConfig, MarketContextReport
from zhisa.regime.features import RegimeFeatureConfig, compute_regime_features
from zhisa.regime.schema import RegimeFeatures
from zhisa.regime.state_space import StateSpaceConfig, StateSpaceRegimeModel, StateSpaceReport
from zhisa.regime.structure import MarketStructureAnalyzer, MarketStructureReport, StructureConfig
from zhisa.storage.resampler import resample_ohlcv
from zhisa.storage.schema import Timeframe


@dataclass(frozen=True)
class RegimeFeatureEngineConfig:
    source_timeframe: str = "5m"
    timeframes: tuple[str, ...] = ("5m", "15m", "1h")
    feature: RegimeFeatureConfig = field(default_factory=RegimeFeatureConfig)
    context: MarketContextConfig = field(default_factory=MarketContextConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    state_space: StateSpaceConfig = field(default_factory=StateSpaceConfig)


@dataclass(frozen=True)
class RegimeFeatureSnapshot:
    symbol: str
    timestamp: str | None
    aggregate: dict[str, float]
    timeframes: dict[str, dict[str, Any]]
    market_context: dict[str, Any]
    market_structure: dict[str, Any]
    state_space: dict[str, Any]
    primary_timeframe: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tf_weight(tf: str) -> float:
    minutes = Timeframe.from_str(tf).minutes if tf in {m.value for m in Timeframe} else 5
    return float(np.log1p(minutes))


class RegimeFeatureEngine:
    """Build causal regime features without assigning regime/risk/playbook decisions."""

    def __init__(self, cfg: Optional[RegimeFeatureEngineConfig] = None) -> None:
        self.cfg = cfg or RegimeFeatureEngineConfig()

    def snapshot(
        self,
        df: pd.DataFrame,
        *,
        t: Optional[int] = None,
        symbol: str = "",
        extra_context: Optional[dict[str, Any]] = None,
    ) -> RegimeFeatureSnapshot:
        if t is not None:
            if t < 0:
                raise ValueError("t must be non-negative")
            work = df.iloc[: t + 1].copy()
        else:
            work = df.copy()
        if work.empty:
            raise ValueError("df slice is empty")
        extra = extra_context or {}
        features = self._multi_timeframe_features(work)
        context = MarketContextAnalyzer(self.cfg.context).analyze(
            work,
            symbol=symbol,
            assets=extra.get("assets"),
            benchmark_symbol=str(extra.get("benchmark_symbol", extra.get("btc_symbol", self.cfg.context.benchmark_symbol))),
            extra_context=extra,
        )
        structure = MarketStructureAnalyzer(self.cfg.structure).analyze(work)
        state_space = StateSpaceRegimeModel(self.cfg.state_space).analyze(work)
        primary_tf = self.cfg.timeframes[0] if self.cfg.timeframes else self.cfg.source_timeframe
        return RegimeFeatureSnapshot(
            symbol=symbol,
            timestamp=work.index[-1].isoformat() if isinstance(work.index, pd.DatetimeIndex) else None,
            aggregate=self._aggregate(features),
            timeframes={tf: f.to_dict() for tf, f in features.items()},
            market_context=context.to_dict(),
            market_structure=structure.to_dict(),
            state_space=state_space.to_dict(),
            primary_timeframe=primary_tf,
        )

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
            if len(tf_df) >= 2:
                out[tf_s] = compute_regime_features(tf_df, timeframe=tf_s, cfg=self.cfg.feature)
        if not out:
            out[self.cfg.source_timeframe] = compute_regime_features(
                df,
                timeframe=self.cfg.source_timeframe,
                cfg=self.cfg.feature,
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


__all__ = [
    "RegimeFeatureEngine",
    "RegimeFeatureEngineConfig",
    "RegimeFeatureSnapshot",
]
