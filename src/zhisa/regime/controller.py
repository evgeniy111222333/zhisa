"""Adaptive feedback loop for regime intelligence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from zhisa.env.actions import DiscreteAction
from zhisa.env.trading_env import TradingEnv
from zhisa.regime.calibration import OnlineRegimeCalibrator, RegimeCalibrationConfig
from zhisa.regime.detector import RegimeIntelligence, RegimeIntelligenceConfig
from zhisa.regime.gating import RegimeActionGateConfig, regime_action_mask
from zhisa.regime.memory import RegimeMemory, RegimeMemoryConfig, RegimeOutcome
from zhisa.regime.planner import RegimeTradePlanner, TradePlan, TradePlannerConfig
from zhisa.regime.schema import RegimeReport


PolicyFn = Callable[[dict], int]


@dataclass(frozen=True)
class RegimeFeedbackConfig:
    analyzer: RegimeIntelligenceConfig = field(default_factory=RegimeIntelligenceConfig)
    memory: RegimeMemoryConfig = field(default_factory=RegimeMemoryConfig)
    calibration: RegimeCalibrationConfig = field(default_factory=RegimeCalibrationConfig)
    planner: TradePlannerConfig = field(default_factory=TradePlannerConfig)
    gate: RegimeActionGateConfig = field(default_factory=RegimeActionGateConfig)
    outcome_horizon: int = 12
    same_symbol_memory: bool = False
    calibrate_reports: bool = True
    learn_from_outcomes: bool = True
    fallback_action: int = int(DiscreteAction.CLOSE)

    def __post_init__(self) -> None:
        if self.outcome_horizon <= 0:
            raise ValueError(f"outcome_horizon must be positive, got {self.outcome_horizon}")


@dataclass(frozen=True)
class RegimeDecisionEvent:
    t: int
    symbol: str
    report: RegimeReport
    plan: TradePlan
    raw_action: int
    final_action: int
    position: float
    price: float
    closed: bool = False
    outcome: RegimeOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["report"] = self.report.to_dict()
        out["plan"] = self.plan.to_dict()
        out["outcome"] = self.outcome.to_dict() if self.outcome is not None else None
        return out


def _forward_outcome(close: pd.Series, t: int, horizon: int) -> RegimeOutcome:
    if t < 0 or t >= len(close) - 1:
        return RegimeOutcome(forward_return=0.0, realized_vol=0.0, max_drawdown=0.0)
    end = min(len(close) - 1, int(t) + int(horizon))
    c0 = float(close.iloc[t])
    future = close.iloc[t + 1 : end + 1].astype(float)
    if future.empty or c0 <= 0 or not np.isfinite(c0):
        return RegimeOutcome(forward_return=0.0, realized_vol=0.0, max_drawdown=0.0)
    path = pd.concat([pd.Series([c0], index=[close.index[t]]), future])
    logret = np.log(path.replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).dropna()
    forward_return = float(future.iloc[-1] / c0 - 1.0)
    realized_vol = float(logret.std(ddof=0) if not logret.empty else 0.0)
    max_drawdown = min(0.0, float((future / c0 - 1.0).min()))
    return RegimeOutcome(
        forward_return=forward_return if np.isfinite(forward_return) else 0.0,
        realized_vol=realized_vol if np.isfinite(realized_vol) else 0.0,
        max_drawdown=max_drawdown if np.isfinite(max_drawdown) else 0.0,
    )


class RegimeAdaptiveController:
    """Close the loop between detector, memory, calibration, planner, and policy."""

    def __init__(
        self,
        base_policy: PolicyFn,
        df: pd.DataFrame,
        *,
        cfg: Optional[RegimeFeedbackConfig] = None,
        analyzer: RegimeIntelligence | None = None,
        memory: RegimeMemory | None = None,
        calibrator: OnlineRegimeCalibrator | None = None,
        symbol: str = "",
    ) -> None:
        self.base_policy = base_policy
        self.df = df
        self.cfg = cfg or RegimeFeedbackConfig()
        self.analyzer = analyzer or RegimeIntelligence(self.cfg.analyzer)
        self.memory = memory or RegimeMemory(self.cfg.memory)
        self.calibrator = calibrator or OnlineRegimeCalibrator(self.cfg.calibration)
        self.symbol = symbol
        self.events: list[RegimeDecisionEvent] = []
        self._open_indices: list[int] = []
        self.masked_count = 0
        self.calibrated_count = 0
        self.memory_updates = 0

    def _valid_fallback(self, mask: np.ndarray, current_position: float) -> int:
        preferred = [self.cfg.fallback_action, int(DiscreteAction.PARTIAL_CLOSE), int(DiscreteAction.SKIP)]
        if abs(float(current_position)) <= 1e-9:
            preferred = [int(DiscreteAction.SKIP), self.cfg.fallback_action]
        for action in preferred:
            if 0 <= action < mask.size and bool(mask[action]):
                return int(action)
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else int(DiscreteAction.SKIP)

    def _raw_action(self, obs: dict) -> int:
        logits_fn = getattr(self.base_policy, "logits", None)
        if callable(logits_fn):
            logits = logits_fn(obs)
            arr = np.asarray(logits.detach().cpu() if hasattr(logits, "detach") else logits, dtype=np.float64)
            return int(arr.reshape(-1, arr.shape[-1]).argmax(axis=-1)[0])
        return int(self.base_policy(obs))

    def _close_mature_events(self, current_t: int) -> None:
        if not self.cfg.learn_from_outcomes or "close" not in self.df.columns:
            return
        keep: list[int] = []
        for idx in self._open_indices:
            event = self.events[idx]
            if event.closed or current_t - event.t < self.cfg.outcome_horizon:
                keep.append(idx)
                continue
            outcome = _forward_outcome(self.df["close"], event.t, self.cfg.outcome_horizon)
            playbook = event.plan.recommended_playbook
            outcome = RegimeOutcome(
                forward_return=outcome.forward_return,
                realized_vol=outcome.realized_vol,
                max_drawdown=outcome.max_drawdown,
                label=playbook,
            )
            self.memory.add(event.report, outcome=outcome, symbol=event.symbol)
            self.calibrator.update(event.report, outcome, playbook=playbook)
            self.events[idx] = RegimeDecisionEvent(
                t=event.t,
                symbol=event.symbol,
                report=event.report,
                plan=event.plan,
                raw_action=event.raw_action,
                final_action=event.final_action,
                position=event.position,
                price=event.price,
                closed=True,
                outcome=outcome,
            )
            self.memory_updates += 1
        self._open_indices = keep

    def select_action(self, *, obs: dict, env: TradingEnv) -> int:
        t = max(0, min(int(env._t) - 1, len(self.df) - 1))
        self._close_mature_events(t)
        symbol = self.symbol or getattr(self.df, "name", "") or ""
        raw_report = self.analyzer.analyze(self.df, t=t, symbol=symbol)
        report = raw_report
        if self.cfg.calibrate_reports:
            report = self.calibrator.calibrate(raw_report)
            self.calibrated_count += int(report is not raw_report)
        current_position = float(env._position)
        plan = RegimeTradePlanner(self.cfg.planner, memory=self.memory).plan(
            report,
            current_position=current_position,
            n_actions=env.action_space.n,
        )
        mask = regime_action_mask(
            report,
            current_position=current_position,
            n_actions=env.action_space.n,
            cfg=self.cfg.gate,
        )
        raw_action = self._raw_action(obs)
        final_action = raw_action if 0 <= raw_action < mask.size and bool(mask[raw_action]) else self._valid_fallback(mask, current_position)
        if final_action != raw_action:
            self.masked_count += 1
        event = RegimeDecisionEvent(
            t=t,
            symbol=symbol,
            report=report,
            plan=plan,
            raw_action=int(raw_action),
            final_action=int(final_action),
            position=current_position,
            price=float(self.df["close"].iloc[t]) if "close" in self.df.columns else 0.0,
        )
        self.events.append(event)
        self._open_indices.append(len(self.events) - 1)
        return int(final_action)

    def observe_step(self, *, obs: dict, action: int, reward: float, info: dict, env: TradingEnv) -> None:
        t = max(0, min(int(env._t) - 1, len(self.df) - 1))
        self._close_mature_events(t)

    def flush(self) -> None:
        self._close_mature_events(len(self.df) - 1)

    @property
    def reports(self) -> list[RegimeReport]:
        return [e.report for e in self.events]

    @property
    def plans(self) -> list[TradePlan]:
        return [e.plan for e in self.events]

    def summary(self) -> dict[str, Any]:
        self.flush()
        primary: dict[str, int] = {}
        playbooks: dict[str, int] = {}
        plan_status: dict[str, int] = {}
        execution: dict[str, int] = {}
        memory_adjustments: list[float] = []
        closed = 0
        for event in self.events:
            primary[event.report.primary_regime] = primary.get(event.report.primary_regime, 0) + 1
            plan_status[event.plan.status] = plan_status.get(event.plan.status, 0) + 1
            playbooks[event.plan.recommended_playbook] = playbooks.get(event.plan.recommended_playbook, 0) + 1
            execution[event.plan.execution.order_type] = execution.get(event.plan.execution.order_type, 0) + 1
            closed += int(event.closed)
            for setup in event.plan.setups:
                prior = setup.memory_prior or {}
                if "score_adjustment" in prior:
                    memory_adjustments.append(float(prior["score_adjustment"]))
        return {
            "n_events": len(self.events),
            "n_closed_outcomes": closed,
            "memory_size": len(self.memory),
            "memory_updates": self.memory_updates,
            "masked_action_rate": float(self.masked_count / max(1, len(self.events))),
            "calibrated_count": int(self.calibrated_count),
            "primary_regimes": primary,
            "plan_status": plan_status,
            "recommended_playbooks": playbooks,
            "execution_order_types": execution,
            "mean_memory_score_adjustment": float(np.mean(memory_adjustments)) if memory_adjustments else 0.0,
        }


__all__ = [
    "RegimeAdaptiveController",
    "RegimeDecisionEvent",
    "RegimeFeedbackConfig",
]
