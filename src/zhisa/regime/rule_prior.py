"""Rule-based regime prior used as weak supervision and safety fallback."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd

from zhisa.regime.detector import RegimeIntelligence, RegimeIntelligenceConfig
from zhisa.regime.schema import RegimeReport


@dataclass(frozen=True)
class RuleRegimePriorConfig:
    analyzer: RegimeIntelligenceConfig = field(default_factory=lambda: RegimeIntelligenceConfig(inference_mode="rule"))
    inference_source: str = "rule_prior"


class RuleRegimePrior:
    """Wrap the legacy rule detector as a weak prior, not the source of truth."""

    def __init__(self, cfg: Optional[RuleRegimePriorConfig] = None) -> None:
        self.cfg = cfg or RuleRegimePriorConfig()
        self.analyzer = RegimeIntelligence(replace(self.cfg.analyzer, inference_mode="rule"))

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        t: Optional[int] = None,
        symbol: str = "",
        extra_context: Optional[dict] = None,
    ) -> RegimeReport:
        report = self.analyzer.analyze(df, t=t, symbol=symbol, extra_context=extra_context)
        features = dict(report.features)
        features["inference_source"] = self.cfg.inference_source
        features["weak_prior"] = True
        return replace(report, features=features)


__all__ = [
    "RuleRegimePrior",
    "RuleRegimePriorConfig",
]
