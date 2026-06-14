"""Backtest reports: print + save equity curve, metrics, and stress summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from zhisa.backtest.engine import BacktestResult
from zhisa.backtest.metrics import Metrics
from zhisa.env.actions import DiscreteAction


def _action_name(action: int) -> str:
    if int(action) < 0:
        return "INIT"
    try:
        return DiscreteAction(int(action)).name
    except ValueError:
        return str(int(action))


def _format_metrics(m: Metrics) -> str:
    rows = [
        ("total_return", f"{m.total_return*100:8.2f} %"),
        ("annualised_return", f"{m.annualised_return*100:8.2f} %"),
        ("annualised_vol", f"{m.annualised_vol*100:8.2f} %"),
        ("sharpe", f"{m.sharpe:8.3f}"),
        ("sortino", f"{m.sortino:8.3f}"),
        ("calmar", f"{m.calmar:8.3f}"),
        ("max_drawdown", f"{m.max_drawdown*100:8.2f} %"),
        ("max_dd_duration_bars", f"{m.max_dd_duration}"),
        ("win_rate", f"{m.win_rate*100:8.2f} %"),
        ("profit_factor", f"{m.profit_factor:8.3f}"),
        ("n_trades", f"{m.n_trades}"),
        ("stability", f"{m.stability*100:8.2f} %"),
        ("deflated_sharpe", f"{m.deflated_sharpe:8.3f}"),
    ]
    out = ["  " + f"{k:<22s}: {v}" for k, v in rows]
    return "\n".join(out)


def print_metrics(m: Metrics, title: Optional[str] = None) -> None:
    if title:
        print(f"== {title} ==")
    print(_format_metrics(m))


def metrics_to_frame(metrics_list: Iterable[Metrics], names: Optional[list[str]] = None) -> pd.DataFrame:
    records = [m.to_dict() for m in metrics_list]
    df = pd.DataFrame.from_records(records)
    if names is not None and len(names) == len(df):
        df.insert(0, "name", names)
    return df


def save_report(
    result: BacktestResult,
    out_dir: str | Path,
    name: str = "backtest",
) -> dict:
    """Save a full backtest report to a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_df = pd.DataFrame({
        "equity": result.equity,
        "position": result.positions,
        "price": result.prices,
        "action": result.actions,
        "action_name": [_action_name(a) for a in result.actions],
        "reward": result.rewards,
    })
    if result.timestamps is not None:
        ts = pd.to_datetime(result.timestamps)
        eq_df.insert(0, "timestamp", ts[: len(eq_df)])
    eq_df.to_csv(out_dir / f"{name}_equity.csv", index=False)
    metrics = result.metrics.to_dict()
    with (out_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"metrics": metrics, "equity_path": str(out_dir / f"{name}_equity.csv")}
