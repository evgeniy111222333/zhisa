from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradeAudit:
    n_trades: int
    long_win_rate: float
    short_win_rate: float
    average_duration_bars: float
    max_consecutive_losses: int
    exposure_fraction: float
    turnover: float


def _safe_rate(values: np.ndarray) -> float:
    return float(np.mean(values > 0)) if values.size else 0.0


def trade_audit(
    positions: np.ndarray,
    equity: np.ndarray,
) -> TradeAudit:
    """Derive side-aware trade statistics from bar-aligned positions/equity."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    n = min(len(pos), len(eq))
    pos, eq = pos[:n], eq[:n]
    if n < 2:
        return TradeAudit(0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    trades: list[tuple[int, int, float]] = []
    start: int | None = None
    side = 0
    for i in range(n):
        current = int(np.sign(pos[i]))
        if side and current != side:
            base = eq[start] if start is not None and eq[start] != 0 else 1.0
            trades.append((side, i - int(start), float(eq[i] / base - 1.0)))
            start = None
            side = 0
        if current and not side:
            start, side = i, current
    if side and start is not None:
        base = eq[start] if eq[start] != 0 else 1.0
        trades.append((side, n - 1 - start, float(eq[-1] / base - 1.0)))
    long_rets = np.asarray([ret for s, _, ret in trades if s > 0])
    short_rets = np.asarray([ret for s, _, ret in trades if s < 0])
    streak = best = 0
    for _, _, ret in trades:
        streak = streak + 1 if ret <= 0 else 0
        best = max(best, streak)
    durations = [duration for _, duration, _ in trades]
    return TradeAudit(
        n_trades=len(trades),
        long_win_rate=_safe_rate(long_rets),
        short_win_rate=_safe_rate(short_rets),
        average_duration_bars=float(np.mean(durations)) if durations else 0.0,
        max_consecutive_losses=best,
        exposure_fraction=float(np.mean(np.abs(pos) > 1e-12)),
        turnover=float(np.abs(np.diff(pos)).sum()),
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float((predicted[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return ece


def multiclass_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def action_entropy(probabilities: np.ndarray) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    return float(np.mean(entropy / np.log(probs.shape[-1])))


def action_churn(actions: np.ndarray) -> float:
    actions = np.asarray(actions).reshape(-1)
    if len(actions) < 2:
        return 0.0
    long = np.isin(actions, (1, 2, 3))
    short = np.isin(actions, (4, 5, 6))
    flips = (long[:-1] & short[1:]) | (short[:-1] & long[1:])
    return float(flips.mean())


def tail_risk(equity: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    eq = np.asarray(equity, dtype=np.float64).reshape(-1)
    returns = np.diff(eq) / np.maximum(np.abs(eq[:-1]), 1e-12)
    if returns.size == 0:
        return {"var": 0.0, "cvar": 0.0, "ulcer_index": 0.0, "recovery_bars": 0.0}
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    peaks = np.maximum.accumulate(eq)
    drawdowns = eq / np.maximum(peaks, 1e-12) - 1.0
    trough = int(np.argmin(drawdowns))
    recovered = np.flatnonzero(eq[trough:] >= peaks[trough])
    recovery = int(recovered[0]) if recovered.size else len(eq) - 1 - trough
    return {
        "var": var,
        "cvar": float(tail.mean()) if tail.size else var,
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdowns)))),
        "recovery_bars": float(recovery),
    }
