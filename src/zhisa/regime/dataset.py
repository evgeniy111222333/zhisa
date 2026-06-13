"""Datasets and labels for supervised regime intelligence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zhisa.regime.detector import RegimeIntelligence, RegimeIntelligenceConfig
from zhisa.regime.memory import RegimeOutcome
from zhisa.regime.schema import MacroRegime, MesoRegime, RegimeReport, RiskMode
from zhisa.regime.vectorizer import RegimeFeatureVectorizer, RegimeVectorizerConfig


MACRO_TO_ID = {r.value: i for i, r in enumerate(MacroRegime)}
MESO_TO_ID = {r.value: i for i, r in enumerate(MesoRegime)}
RISK_MODE_TO_ID = {r.value: i for i, r in enumerate(RiskMode)}
PLAYBOOK_NAMES: tuple[str, ...] = (
    "no_trade_wait",
    "trend_pullback_long",
    "breakout_retest_long",
    "trend_pullback_short",
    "breakout_retest_short",
    "range_reversion_long",
    "range_reversion_short",
    "panic_retest_short",
    "capitulation_reversal_small",
    "volatility_expansion_wait",
    "liquidity_sweep_reversal",
    "pullback_only_long",
    "pullback_only_short",
    "liquidation_retest_only",
    "relative_strength_only",
    "value_area_reversion",
    "pullback_to_value_only",
)
PLAYBOOK_TO_ID = {name: i for i, name in enumerate(PLAYBOOK_NAMES)}
PLAYBOOK_DIRECTIONS: dict[str, float] = {
    "trend_pullback_long": 1.0,
    "breakout_retest_long": 1.0,
    "range_reversion_long": 1.0,
    "pullback_only_long": 1.0,
    "trend_pullback_short": -1.0,
    "breakout_retest_short": -1.0,
    "range_reversion_short": -1.0,
    "panic_retest_short": -1.0,
    "pullback_only_short": -1.0,
    "capitulation_reversal_small": 0.5,
    "liquidity_sweep_reversal": 0.0,
    "value_area_reversion": 0.0,
    "relative_strength_only": 0.0,
    "pullback_to_value_only": 0.0,
    "liquidation_retest_only": 0.0,
}


@dataclass(frozen=True)
class RegimeSupervisionConfig:
    horizon: int = 12
    stride: int = 1
    min_history: int = 128
    symbol: str = ""
    cache_items: bool = True
    analyzer: RegimeIntelligenceConfig = field(default_factory=RegimeIntelligenceConfig)
    vectorizer: RegimeVectorizerConfig = field(default_factory=RegimeVectorizerConfig)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.min_history < 2:
            raise ValueError(f"min_history must be >= 2, got {self.min_history}")


@dataclass(frozen=True)
class RegimeSupervisionItem:
    x: torch.Tensor
    macro: torch.Tensor
    meso: torch.Tensor
    risk_mode: torch.Tensor
    tradeability: torch.Tensor
    transition_risk: torch.Tensor
    forward_return: torch.Tensor
    realized_vol: torch.Tensor
    max_drawdown: torch.Tensor
    playbook_label: torch.Tensor
    playbook_scores: torch.Tensor
    report: RegimeReport
    outcome: RegimeOutcome
    meta: dict[str, Any]


@dataclass(frozen=True)
class RegimeSupervisionBatch:
    x: torch.Tensor
    macro: torch.Tensor
    meso: torch.Tensor
    risk_mode: torch.Tensor
    tradeability: torch.Tensor
    transition_risk: torch.Tensor
    forward_return: torch.Tensor
    realized_vol: torch.Tensor
    max_drawdown: torch.Tensor
    playbook_label: torch.Tensor
    playbook_scores: torch.Tensor
    reports: list[RegimeReport]
    outcomes: list[RegimeOutcome]
    meta: list[dict[str, Any]]


def _forward_outcome(close: pd.Series, t: int, horizon: int) -> RegimeOutcome:
    c0 = float(close.iloc[t])
    future = close.iloc[t + 1 : t + horizon + 1].astype(float)
    if future.empty or c0 <= 0 or not np.isfinite(c0):
        return RegimeOutcome(forward_return=0.0, realized_vol=0.0, max_drawdown=0.0)
    path = pd.concat([pd.Series([c0], index=[close.index[t]]), future])
    logret = np.log(path.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan)
    forward_return = float(future.iloc[-1] / c0 - 1.0)
    realized_vol = float(logret.dropna().std(ddof=0) or 0.0)
    max_drawdown = min(0.0, float((future / c0 - 1.0).min()))
    return RegimeOutcome(
        forward_return=forward_return if np.isfinite(forward_return) else 0.0,
        realized_vol=realized_vol if np.isfinite(realized_vol) else 0.0,
        max_drawdown=max_drawdown if np.isfinite(max_drawdown) else 0.0,
    )


def _playbook_scores(report: RegimeReport, outcome: RegimeOutcome) -> tuple[np.ndarray, int, str]:
    scores = np.full(len(PLAYBOOK_NAMES), -1.0, dtype=np.float32)
    ret = float(outcome.forward_return or 0.0)
    dd_penalty = abs(float(outcome.max_drawdown or 0.0))
    vol_penalty = float(outcome.realized_vol or 0.0)
    for playbook in report.allowed_playbooks:
        if playbook not in PLAYBOOK_TO_ID:
            continue
        direction = PLAYBOOK_DIRECTIONS.get(playbook, 0.0)
        if playbook in {"no_trade_wait", "volatility_expansion_wait"}:
            score = 0.02 - abs(ret) - 0.5 * vol_penalty
        elif direction > 0:
            score = direction * ret - 0.5 * dd_penalty - 0.25 * vol_penalty
        elif direction < 0:
            score = direction * ret - 0.5 * dd_penalty - 0.25 * vol_penalty
        else:
            score = abs(ret) * 0.3 - 0.4 * dd_penalty - 0.25 * vol_penalty
        scores[PLAYBOOK_TO_ID[playbook]] = float(score)
    if np.all(scores < -0.999):
        scores[PLAYBOOK_TO_ID["no_trade_wait"]] = 0.0
    best_id = int(np.argmax(scores))
    return scores, best_id, PLAYBOOK_NAMES[best_id]


class RegimeSupervisionDataset(Dataset):
    """Causal regime snapshots with forward outcomes for supervised learning."""

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Optional[RegimeSupervisionConfig] = None,
        *,
        analyzer: RegimeIntelligence | None = None,
        vectorizer: RegimeFeatureVectorizer | None = None,
    ) -> None:
        self.df = df
        self.cfg = cfg or RegimeSupervisionConfig()
        self.analyzer = analyzer or RegimeIntelligence(self.cfg.analyzer)
        self.vectorizer = vectorizer or RegimeFeatureVectorizer(self.cfg.vectorizer)
        if "close" not in df.columns:
            raise ValueError("df must contain a close column")
        if len(df) <= self.cfg.min_history + self.cfg.horizon:
            raise ValueError(
                "df is too short for regime supervision: "
                f"len={len(df)}, min_history={self.cfg.min_history}, horizon={self.cfg.horizon}"
            )
        start = self.cfg.min_history - 1
        stop = len(df) - self.cfg.horizon - 1
        self.indices = list(range(start, stop + 1, self.cfg.stride))
        self._cache: dict[int, RegimeSupervisionItem] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> RegimeSupervisionItem:
        if self.cfg.cache_items and idx in self._cache:
            return self._cache[idx]
        t = self.indices[idx]
        report = self.analyzer.analyze(self.df, t=t, symbol=self.cfg.symbol)
        outcome = _forward_outcome(self.df["close"], t, self.cfg.horizon)
        playbook_scores, playbook_label, best_playbook = _playbook_scores(report, outcome)
        outcome = RegimeOutcome(
            forward_return=outcome.forward_return,
            realized_vol=outcome.realized_vol,
            max_drawdown=outcome.max_drawdown,
            label=best_playbook,
        )
        x = self.vectorizer.transform(report)
        meta = {
            "t": t,
            "timestamp": report.features.get("timestamp") if report.features else None,
            "symbol": self.cfg.symbol,
            "horizon": self.cfg.horizon,
            "best_playbook": best_playbook,
            "playbook_scores": {name: float(playbook_scores[i]) for i, name in enumerate(PLAYBOOK_NAMES)},
        }
        item = RegimeSupervisionItem(
            x=torch.from_numpy(x).float(),
            macro=torch.tensor(MACRO_TO_ID[report.primary_regime], dtype=torch.long),
            meso=torch.tensor(MESO_TO_ID[report.secondary_regime], dtype=torch.long),
            risk_mode=torch.tensor(RISK_MODE_TO_ID[report.risk_mode], dtype=torch.long),
            tradeability=torch.tensor(float(report.tradeability_score), dtype=torch.float32),
            transition_risk=torch.tensor(float(report.transition_risk), dtype=torch.float32),
            forward_return=torch.tensor(float(outcome.forward_return or 0.0), dtype=torch.float32),
            realized_vol=torch.tensor(float(outcome.realized_vol or 0.0), dtype=torch.float32),
            max_drawdown=torch.tensor(float(outcome.max_drawdown or 0.0), dtype=torch.float32),
            playbook_label=torch.tensor(playbook_label, dtype=torch.long),
            playbook_scores=torch.from_numpy(playbook_scores).float(),
            report=report,
            outcome=outcome,
            meta=meta,
        )
        if self.cfg.cache_items:
            self._cache[idx] = item
        return item


def regime_supervision_collate(items: list[RegimeSupervisionItem]) -> RegimeSupervisionBatch:
    return RegimeSupervisionBatch(
        x=torch.stack([it.x for it in items]),
        macro=torch.stack([it.macro for it in items]),
        meso=torch.stack([it.meso for it in items]),
        risk_mode=torch.stack([it.risk_mode for it in items]),
        tradeability=torch.stack([it.tradeability for it in items]),
        transition_risk=torch.stack([it.transition_risk for it in items]),
        forward_return=torch.stack([it.forward_return for it in items]),
        realized_vol=torch.stack([it.realized_vol for it in items]),
        max_drawdown=torch.stack([it.max_drawdown for it in items]),
        playbook_label=torch.stack([it.playbook_label for it in items]),
        playbook_scores=torch.stack([it.playbook_scores for it in items]),
        reports=[it.report for it in items],
        outcomes=[it.outcome for it in items],
        meta=[it.meta for it in items],
    )


__all__ = [
    "MACRO_TO_ID",
    "MESO_TO_ID",
    "PLAYBOOK_NAMES",
    "PLAYBOOK_TO_ID",
    "RISK_MODE_TO_ID",
    "RegimeSupervisionBatch",
    "RegimeSupervisionConfig",
    "RegimeSupervisionDataset",
    "RegimeSupervisionItem",
    "regime_supervision_collate",
]
