"""Risk and performance metrics for backtest equity curves."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Metrics:
    """Standard risk-adjusted performance summary for an equity curve."""

    n_periods: int
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_dd_duration: int
    win_rate: float
    profit_factor: float
    avg_trade: float
    n_trades: int
    stability: float       # rolling sharpe consistency (0..1)
    deflated_sharpe: float # rough approximation
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in (
            "n_periods", "total_return", "annualised_return", "annualised_vol",
            "sharpe", "sortino", "calmar", "max_drawdown", "max_dd_duration",
            "win_rate", "profit_factor", "avg_trade", "n_trades",
            "stability", "deflated_sharpe",
        )}
        out.update(self.extra)
        return out


def _to_returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(equity) / equity[:-1]


def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    if equity.size == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(-dd.min()) if dd.size else 0.0
    # Duration: longest run below prior peak
    in_dd = dd < 0
    if not in_dd.any():
        return max_dd, 0
    durations = np.diff(np.concatenate([[0], in_dd.astype(int).cumsum()]))
    # Find longest streak of True
    longest = 0
    cur = 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return max_dd, int(longest)


def _deflated_sharpe(sharpe: float, n_returns: int, n_trials: int = 1) -> float:
    """Bailey & López de Prado deflated Sharpe ratio (skeleton).

    For a rigorous implementation, see ``deflated_sharpe_ratio`` in
    their paper; here we use the standard-error adjustment only.
    """
    if n_returns < 2:
        return 0.0
    se = 1.0 / np.sqrt(n_returns)
    return float(sharpe - 1.96 * se * np.sqrt(max(1, np.log(max(n_trials, 1)))))


def compute_metrics(
    equity: np.ndarray,
    *,
    periods_per_year: int = 365 * 24 * 12,
    trade_returns: Optional[np.ndarray] = None,
    rolling_window: int = 96,
) -> Metrics:
    """Compute the standard risk metrics from an equity curve.

    Args:
        equity: 1-D array of portfolio values over time.
        periods_per_year: how many periods (bars) per year (used for annualisation).
        trade_returns: optional 1-D array of per-trade returns (for win rate etc.).
        rolling_window: window for rolling sharpe stability.
    """
    equity = np.asarray(equity, dtype=np.float64).ravel()
    rets = _to_returns(equity)
    n = rets.size
    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0,
                       0.0, 0.0)

    total_return = float(equity[-1] / equity[0] - 1.0)
    years = max(n / periods_per_year, 1e-12)
    ratio = float(equity[-1] / max(equity[0], 1e-12))
    if ratio <= 0 or not np.isfinite(ratio):
        cagr = -1.0
    else:
        try:
            with np.errstate(over="raise"):
                cagr = float(np.exp(np.log(ratio) / years) - 1.0)
        except (FloatingPointError, OverflowError):
            cagr = -1.0 if ratio < 1 else 1.0
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
    mean = float(rets.mean() * periods_per_year)
    sharpe = mean / vol if vol > 0 else 0.0
    downside = rets[rets < 0]
    down_std = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if downside.size > 1 else vol
    sortino = mean / down_std if down_std > 0 else 0.0
    max_dd, dd_dur = _max_drawdown(equity)
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Trade-level
    if trade_returns is not None and trade_returns.size > 0:
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        win_rate = float(wins.size / trade_returns.size)
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_trade = float(trade_returns.mean())
        n_trades = int(trade_returns.size)
    else:
        win_rate = profit_factor = avg_trade = 0.0
        n_trades = 0

    # Rolling sharpe stability
    if n >= rolling_window * 2:
        roll_mean = pd_series(rets).rolling(rolling_window).mean().to_numpy()
        roll_std = pd_series(rets).rolling(rolling_window).std(ddof=1).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rs = (roll_mean / (roll_std + 1e-9)) * np.sqrt(periods_per_year)
        rs = np.nan_to_num(rs, nan=0.0, posinf=0.0, neginf=0.0)
        stability = float(np.mean(rs > 0.0))
    else:
        stability = 0.0

    deflated = _deflated_sharpe(sharpe, n)
    return Metrics(
        n_periods=n,
        total_return=total_return,
        annualised_return=float(cagr),
        annualised_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_dd_duration=dd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        n_trades=n_trades,
        stability=stability,
        deflated_sharpe=deflated,
    )


def pd_series(x):
    """Local helper: avoid pandas top-level import here to keep this module light."""
    import pandas as pd  # type: ignore
    return pd.Series(x)
