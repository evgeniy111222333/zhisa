"""Trade planning from structured regime intelligence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from zhisa.env.actions import DiscreteAction
from zhisa.regime.gating import (
    RegimeActionGateConfig,
    regime_action_mask,
    regime_position_size_multiplier,
)
from zhisa.regime.schema import MacroRegime, RegimeReport, RiskMode


@dataclass(frozen=True)
class TradePlannerConfig:
    min_tradeability: float = 0.25
    max_setups: int = 4
    min_setup_score: float = 0.05
    near_liquidity_pct: float = 0.01
    transition_wait_threshold: float = 0.55
    gate: RegimeActionGateConfig = field(default_factory=RegimeActionGateConfig)


@dataclass(frozen=True)
class TradeSetup:
    playbook: str
    direction: str
    entry_style: str
    allowed_actions: list[int]
    target_position: float
    size_multiplier: float
    stop_style: str
    take_profit_style: str
    invalidation: list[str]
    confirmation: list[str]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradePlan:
    status: str
    risk_mode: str
    tradeability_score: float
    risk_budget: float
    action_mask: list[bool]
    recommended_action: int
    recommended_playbook: str
    setups: list[TradeSetup]
    no_trade_reasons: list[str]
    management_notes: list[str]
    explanation: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "risk_mode": self.risk_mode,
            "tradeability_score": self.tradeability_score,
            "risk_budget": self.risk_budget,
            "action_mask": self.action_mask,
            "recommended_action": self.recommended_action,
            "recommended_playbook": self.recommended_playbook,
            "setups": [s.to_dict() for s in self.setups],
            "no_trade_reasons": self.no_trade_reasons,
            "management_notes": self.management_notes,
            "explanation": self.explanation,
        }


LONG_PLAYBOOKS = {
    "trend_pullback_long",
    "breakout_retest_long",
    "range_reversion_long",
    "pullback_only_long",
}
SHORT_PLAYBOOKS = {
    "trend_pullback_short",
    "breakout_retest_short",
    "range_reversion_short",
    "panic_retest_short",
    "pullback_only_short",
}
WAIT_PLAYBOOKS = {
    "no_trade_wait",
    "volatility_expansion_wait",
    "transition_wait",
}
TACTICAL_PLAYBOOKS = {
    "capitulation_reversal_small",
    "liquidity_sweep_reversal",
    "liquidation_retest_only",
    "relative_strength_only",
    "value_area_reversion",
    "pullback_to_value_only",
}


def _nested_get(data: dict, path: str, default: object = None) -> object:
    cur: object = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _target_for_action(action: int, current_position: float) -> float:
    if action == int(DiscreteAction.SKIP):
        return float(current_position)
    if action == int(DiscreteAction.PARTIAL_CLOSE):
        return 0.5 * float(current_position)
    mapping = {
        int(DiscreteAction.LONG_25): 0.25,
        int(DiscreteAction.LONG_50): 0.50,
        int(DiscreteAction.LONG_100): 1.00,
        int(DiscreteAction.SHORT_25): -0.25,
        int(DiscreteAction.SHORT_50): -0.50,
        int(DiscreteAction.SHORT_100): -1.00,
        int(DiscreteAction.CLOSE): 0.0,
    }
    return mapping.get(int(action), 0.0)


class RegimeTradePlanner:
    """Build actionable trade plans from a RegimeReport."""

    def __init__(self, cfg: Optional[TradePlannerConfig] = None) -> None:
        self.cfg = cfg or TradePlannerConfig()

    def plan(
        self,
        report: RegimeReport,
        *,
        current_position: float = 0.0,
        n_actions: int = 9,
    ) -> TradePlan:
        mask = regime_action_mask(
            report,
            current_position=current_position,
            n_actions=n_actions,
            cfg=self.cfg.gate,
        )
        risk_budget = regime_position_size_multiplier(report, cfg=self.cfg.gate)
        no_trade = self._no_trade_reasons(report, risk_budget)
        setups = self._setups(report, mask, risk_budget, current_position)
        status = "tradeable" if setups and not no_trade else "no_trade"
        if setups and no_trade and float(report.tradeability_score) >= self.cfg.min_tradeability * 1.5:
            status = "conditional"
        recommended = setups[0] if setups else None
        recommended_action = self._recommended_action(recommended, mask, current_position)
        management = self._management_notes(report, current_position, risk_budget)
        return TradePlan(
            status=status,
            risk_mode=report.risk_mode,
            tradeability_score=float(report.tradeability_score),
            risk_budget=float(risk_budget),
            action_mask=[bool(x) for x in mask.tolist()],
            recommended_action=int(recommended_action),
            recommended_playbook=recommended.playbook if recommended else "no_trade_wait",
            setups=setups,
            no_trade_reasons=no_trade,
            management_notes=management,
            explanation=report.explanation,
        )

    def _no_trade_reasons(self, report: RegimeReport, risk_budget: float) -> list[str]:
        reasons: list[str] = []
        if report.risk_mode == RiskMode.OFF.value:
            reasons.append("risk mode is off")
        if float(report.tradeability_score) < self.cfg.min_tradeability:
            reasons.append("tradeability below threshold")
        if risk_budget <= 1e-6:
            reasons.append("risk budget is zero")
        state_cp = float(_nested_get(report.features, "state_space.change_point_score", 0.0) or 0.0)
        state_tp = float(_nested_get(report.features, "state_space.transition_probability", 0.0) or 0.0)
        if state_cp > self.cfg.transition_wait_threshold or state_tp > self.cfg.transition_wait_threshold:
            reasons.append("state-space transition risk is elevated")
        if report.primary_regime == MacroRegime.HIGH_VOL_CRASH.value and "capitulation_reversal_small" not in report.allowed_playbooks:
            reasons.append("crash regime without capitulation playbook")
        return reasons

    def _setups(
        self,
        report: RegimeReport,
        mask: np.ndarray,
        risk_budget: float,
        current_position: float,
    ) -> list[TradeSetup]:
        setups: list[TradeSetup] = []
        for playbook in report.allowed_playbooks:
            if playbook in WAIT_PLAYBOOKS:
                continue
            setup = self._setup_for_playbook(report, playbook, mask, risk_budget, current_position)
            if setup is not None and setup.score >= self.cfg.min_setup_score:
                setups.append(setup)
        setups.sort(key=lambda s: s.score, reverse=True)
        return setups[: self.cfg.max_setups]

    def _setup_for_playbook(
        self,
        report: RegimeReport,
        playbook: str,
        mask: np.ndarray,
        risk_budget: float,
        current_position: float,
    ) -> TradeSetup | None:
        direction = self._direction(playbook, report)
        actions = self._actions_for_direction(direction, mask)
        if not actions:
            return None
        target = self._target_position(direction, risk_budget)
        score = self._score(report, playbook, direction, risk_budget)
        invalidation = self._invalidation(report, direction)
        confirmation = self._confirmation(report, playbook, direction)
        return TradeSetup(
            playbook=playbook,
            direction=direction,
            entry_style=self._entry_style(report, playbook),
            allowed_actions=actions,
            target_position=target if abs(target) > abs(current_position) else current_position,
            size_multiplier=float(risk_budget),
            stop_style=report.stop_style,
            take_profit_style=report.take_profit_style,
            invalidation=invalidation,
            confirmation=confirmation,
            score=score,
        )

    def _direction(self, playbook: str, report: RegimeReport) -> str:
        if playbook in LONG_PLAYBOOKS:
            return "long"
        if playbook in SHORT_PLAYBOOKS:
            return "short"
        if report.primary_regime == MacroRegime.BEAR_TREND.value:
            return "short"
        if report.primary_regime in {MacroRegime.BULL_TREND.value, MacroRegime.POST_CRASH_RECOVERY.value}:
            return "long"
        return "neutral"

    def _actions_for_direction(self, direction: str, mask: np.ndarray) -> list[int]:
        if direction == "long":
            candidates = [DiscreteAction.LONG_25, DiscreteAction.LONG_50, DiscreteAction.LONG_100]
        elif direction == "short":
            candidates = [DiscreteAction.SHORT_25, DiscreteAction.SHORT_50, DiscreteAction.SHORT_100]
        else:
            candidates = [DiscreteAction.CLOSE, DiscreteAction.PARTIAL_CLOSE, DiscreteAction.SKIP]
        return [int(a) for a in candidates if int(a) < mask.size and bool(mask[int(a)])]

    def _target_position(self, direction: str, risk_budget: float) -> float:
        if direction == "long":
            return float(risk_budget)
        if direction == "short":
            return -float(risk_budget)
        return 0.0

    def _entry_style(self, report: RegimeReport, playbook: str) -> str:
        if "pullback" in playbook or playbook == "pullback_to_value_only":
            return "wait_for_pullback_to_value"
        if "retest" in playbook:
            return "breakout_retest_confirmation"
        if "reversion" in playbook:
            return "fade_extreme_after_reclaim"
        if "liquidation" in playbook:
            return "wait_for_liquidation_retest"
        if report.trend_phase in {"late", "exhausted"}:
            return "confirmation_only_no_chase"
        return "market_structure_confirmation"

    def _score(self, report: RegimeReport, playbook: str, direction: str, risk_budget: float) -> float:
        score = float(report.tradeability_score) * 0.55 + float(report.confidence) * 0.25
        score += min(risk_budget, 1.0) * 0.20
        if playbook in TACTICAL_PLAYBOOKS:
            score -= 0.05
        if report.trend_phase in {"late", "exhausted"} and "pullback" not in playbook and "value" not in playbook:
            score -= 0.20
        if direction == "long" and any("crowded_long" in x for x in report.blocked_playbooks):
            score -= 0.15
        if direction == "short" and any("crowded_short" in x for x in report.blocked_playbooks):
            score -= 0.15
        if "entry_directly_into_liquidity" in report.blocked_playbooks:
            score -= 0.10
        return float(np.clip(score, 0.0, 1.0))

    def _invalidation(self, report: RegimeReport, direction: str) -> list[str]:
        inv = [f"risk_mode changes from {report.risk_mode}", "state-space transition risk rises"]
        nearest = _nested_get(report.features, "market_structure.liquidity.nearest_level", None)
        if isinstance(nearest, dict):
            inv.append(f"failed reaction at {nearest.get('name', 'nearest_liquidity')}")
        if direction == "long":
            inv.append("close loses value area / structure support")
        elif direction == "short":
            inv.append("close reclaims value area / structure resistance")
        else:
            inv.append("directional confirmation appears")
        return inv

    def _confirmation(self, report: RegimeReport, playbook: str, direction: str) -> list[str]:
        conf = []
        if "pullback" in playbook or "value" in playbook:
            conf.append("price reacts at value area or prior structure")
        if "liquidity" in playbook or "reversion" in playbook:
            conf.append("liquidity sweep followed by reclaim")
        if direction == "long":
            conf.append("higher low / bullish continuation confirmation")
        elif direction == "short":
            conf.append("lower high / bearish continuation confirmation")
        if report.trend_phase in {"late", "exhausted"}:
            conf.append("no chase; require pullback confirmation")
        return conf or ["wait for structure confirmation"]

    def _recommended_action(self, setup: TradeSetup | None, mask: np.ndarray, current_position: float) -> int:
        if setup is not None and setup.allowed_actions:
            candidates = sorted(
                setup.allowed_actions,
                key=lambda a: abs(abs(_target_for_action(a, current_position)) - abs(setup.target_position)),
            )
            return int(candidates[0])
        for action in (DiscreteAction.CLOSE, DiscreteAction.PARTIAL_CLOSE, DiscreteAction.SKIP):
            if int(action) < mask.size and bool(mask[int(action)]):
                return int(action)
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else int(DiscreteAction.SKIP)

    def _management_notes(self, report: RegimeReport, current_position: float, risk_budget: float) -> list[str]:
        notes = [
            f"max risk budget {risk_budget:.2f}",
            f"stop style: {report.stop_style}",
            f"take profit style: {report.take_profit_style}",
        ]
        if abs(current_position) > risk_budget + 1e-9:
            notes.append("current position exceeds regime risk budget; reduce exposure")
        if report.trend_phase in {"late", "exhausted"}:
            notes.append("late/exhausted trend: use partials and avoid chase")
        if report.transition_risk > 0.55:
            notes.append("transition risk elevated: prefer smaller size or wait")
        return notes


def plan_trade(
    report: RegimeReport,
    *,
    current_position: float = 0.0,
    n_actions: int = 9,
    cfg: TradePlannerConfig | None = None,
) -> TradePlan:
    return RegimeTradePlanner(cfg).plan(
        report,
        current_position=current_position,
        n_actions=n_actions,
    )


__all__ = [
    "TradePlan",
    "TradePlannerConfig",
    "TradeSetup",
    "RegimeTradePlanner",
    "plan_trade",
]
