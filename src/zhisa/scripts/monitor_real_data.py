"""Monitor local real-market data for regime opportunities and risks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhisa.env.actions import DiscreteAction
from zhisa.regime.diagnostics import diagnose_regime_sequence
from zhisa.regime.detector import RegimeIntelligence
from zhisa.regime.memory import RegimeOutcome
from zhisa.regime.planner import TradePlan, plan_trade
from zhisa.regime.profiles import build_regime_profile_config, resolve_regime_profile
from zhisa.regime.schema import RegimeReport
from zhisa.storage.schema import SeriesKey, Timeframe
from zhisa.storage.tsdb import TimeSeriesDB


DIRECTIONAL_ACTIONS = {
    int(DiscreteAction.LONG_25),
    int(DiscreteAction.LONG_50),
    int(DiscreteAction.LONG_100),
    int(DiscreteAction.SHORT_25),
    int(DiscreteAction.SHORT_50),
    int(DiscreteAction.SHORT_100),
}


def _parse_symbols(value: str, db: TimeSeriesDB, timeframe: Timeframe) -> list[str]:
    if value.strip():
        return [s.strip() for s in value.split(",") if s.strip()]
    return [key.instrument for key in db.list_series() if key.timeframe == timeframe]


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(-dd.min()) if dd.size else 0.0


def _market_stats(df: pd.DataFrame) -> dict[str, Any]:
    close = df["close"].to_numpy(dtype=np.float64)
    logret = np.diff(np.log(np.maximum(close, 1e-12)))
    equity = close / max(close[0], 1e-12)
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else "",
        "end": str(df.index[-1]) if len(df) else "",
        "first_close": float(close[0]) if close.size else 0.0,
        "last_close": float(close[-1]) if close.size else 0.0,
        "period_return": float(close[-1] / max(close[0], 1e-12) - 1.0) if close.size else 0.0,
        "annualized_logret_vol": float(logret.std(ddof=1) * np.sqrt(288 * 365)) if logret.size > 1 else 0.0,
        "buy_hold_max_drawdown": _max_drawdown(equity),
    }


def _future_outcome(df: pd.DataFrame, t: int, horizon: int) -> RegimeOutcome:
    close = df["close"].to_numpy(dtype=np.float64)
    start = max(0, min(int(t), close.size - 1))
    end = max(start, min(start + int(horizon), close.size - 1))
    if end <= start:
        return RegimeOutcome()
    path = close[start : end + 1] / max(close[start], 1e-12) - 1.0
    logret = np.diff(np.log(np.maximum(close[start : end + 1], 1e-12)))
    return RegimeOutcome(
        forward_return=float(path[-1]),
        realized_vol=float(logret.std(ddof=1)) if logret.size > 1 else 0.0,
        max_drawdown=float(path.min()),
    )


def _action_direction(action: int) -> int:
    if int(action) in {
        int(DiscreteAction.LONG_25),
        int(DiscreteAction.LONG_50),
        int(DiscreteAction.LONG_100),
    }:
        return 1
    if int(action) in {
        int(DiscreteAction.SHORT_25),
        int(DiscreteAction.SHORT_50),
        int(DiscreteAction.SHORT_100),
    }:
        return -1
    return 0


def _direction_label(direction: int) -> str:
    if direction > 0:
        return "long"
    if direction < 0:
        return "short"
    return "neutral"


def _future_directional_stats(
    df: pd.DataFrame,
    t: int,
    horizon: int,
    direction: int,
) -> dict[str, float]:
    close = df["close"].to_numpy(dtype=np.float64)
    start = max(0, min(int(t), close.size - 1))
    end = max(start, min(start + int(horizon), close.size - 1))
    if direction == 0 or end <= start:
        return {"directional_forward_return": 0.0, "directional_max_adverse": 0.0}
    raw_path = close[start : end + 1] / max(close[start], 1e-12) - 1.0
    directional_path = float(direction) * raw_path
    return {
        "directional_forward_return": float(directional_path[-1]),
        "directional_max_adverse": float(directional_path.min()),
    }


def _action_name(action: int) -> str:
    try:
        return DiscreteAction(int(action)).name
    except ValueError:
        return str(int(action))


def _is_opportunity(
    report: RegimeReport,
    plan: TradePlan,
    *,
    min_tradeability: float,
    max_transition_risk: float,
) -> bool:
    return bool(
        plan.status in {"tradeable", "conditional"}
        and float(report.tradeability_score) >= float(min_tradeability)
        and float(report.transition_risk) <= float(max_transition_risk)
        and int(plan.recommended_action) in DIRECTIONAL_ACTIONS
        and plan.position_management.add_allowed
    )


def _opportunity_quality(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 10:
        return "insufficient_samples"
    returns = np.asarray([float(r["directional_forward_return"]) for r in rows], dtype=np.float64)
    hit_rate = float((returns > 0.0).mean())
    mean_return = float(returns.mean())
    if hit_rate >= 0.55 and mean_return > 0.0:
        return "positive_historical_edge"
    if hit_rate >= 0.50 and mean_return >= 0.0:
        return "mixed_but_nonnegative"
    return "weak_or_negative_edge"


def _row_from_report(
    symbol: str,
    timestamp: str,
    report: RegimeReport,
    plan: TradePlan,
    *,
    opportunity: bool,
    outcome: RegimeOutcome | None = None,
    directional: dict[str, float] | None = None,
) -> dict[str, Any]:
    direction = _action_direction(plan.recommended_action)
    row = {
        "symbol": symbol,
        "timestamp": timestamp,
        "primary_regime": report.primary_regime,
        "secondary_regime": report.secondary_regime,
        "micro_regime": report.micro_regime,
        "risk_mode": report.risk_mode,
        "confidence": float(report.confidence),
        "uncertainty": float(report.uncertainty),
        "transition_risk": float(report.transition_risk),
        "tradeability_score": float(report.tradeability_score),
        "position_size_multiplier": float(report.position_size_multiplier),
        "plan_status": plan.status,
        "recommended_action": int(plan.recommended_action),
        "recommended_action_name": _action_name(plan.recommended_action),
        "recommended_direction": _direction_label(direction),
        "recommended_playbook": plan.recommended_playbook,
        "execution_order_type": plan.execution.order_type,
        "execution_urgency": plan.execution.urgency,
        "reduce_only": bool(plan.execution.reduce_only),
        "allow_market": bool(plan.execution.allow_market),
        "position_intent": plan.position_management.intent,
        "add_allowed": bool(plan.position_management.add_allowed),
        "de_risk_required": bool(plan.position_management.de_risk_required),
        "opportunity": bool(opportunity),
    }
    if outcome is not None:
        row.update({
            "forward_return": float(outcome.forward_return),
            "realized_vol": float(outcome.realized_vol),
            "future_max_drawdown": float(outcome.max_drawdown),
        })
    if directional is not None:
        row.update({
            "directional_forward_return": float(directional["directional_forward_return"]),
            "directional_max_adverse": float(directional["directional_max_adverse"]),
        })
    return row


def _analyzer_for_symbol(
    symbol: str,
    *,
    profile: str,
    timeframe: str,
    benchmark_symbol: str,
) -> RegimeIntelligence:
    if profile == "auto":
        cfg = resolve_regime_profile(symbol=symbol, asset_class="crypto").with_overrides(
            source_timeframe=timeframe,
            benchmark_symbol=benchmark_symbol or None,
        )
    else:
        cfg = build_regime_profile_config(
            profile,
            source_timeframe=timeframe,
            benchmark_symbol=benchmark_symbol or None,
        )
    return RegimeIntelligence(cfg)


def _monitor_symbol(
    db: TimeSeriesDB,
    symbol: str,
    *,
    timeframe: Timeframe,
    bars: int,
    scan_bars: int,
    stride: int,
    horizon: int,
    profile: str,
    benchmark_symbol: str,
    min_tradeability: float,
    max_transition_risk: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    key = SeriesKey(symbol, timeframe)
    df = db.read_latest(key, int(bars)) if bars > 0 else db.read(key)
    if len(df) < max(80, horizon + 5):
        raise ValueError(f"{symbol}@{timeframe.value} has too few bars for monitoring: {len(df)}")

    analyzer = _analyzer_for_symbol(
        symbol,
        profile=profile,
        timeframe=timeframe.value,
        benchmark_symbol=benchmark_symbol,
    )
    audit = db.audit(key)
    latest_report = analyzer.analyze(df, t=len(df) - 1, symbol=symbol)
    latest_plan = plan_trade(latest_report)
    latest_opportunity = _is_opportunity(
        latest_report,
        latest_plan,
        min_tradeability=min_tradeability,
        max_transition_risk=max_transition_risk,
    )
    latest = _row_from_report(
        symbol,
        str(df.index[-1]),
        latest_report,
        latest_plan,
        opportunity=latest_opportunity,
    )

    reports: list[RegimeReport] = []
    plans: list[TradePlan] = []
    outcomes: list[RegimeOutcome] = []
    rows: list[dict[str, Any]] = []
    scan_start = max(0, len(df) - int(scan_bars))
    scan_stop = max(scan_start, len(df) - int(horizon) - 1)
    for t in range(scan_start, scan_stop, max(1, int(stride))):
        report = analyzer.analyze(df, t=t, symbol=symbol)
        plan = plan_trade(report)
        outcome = _future_outcome(df, t, horizon)
        directional = _future_directional_stats(
            df,
            t,
            horizon,
            _action_direction(plan.recommended_action),
        )
        opportunity = _is_opportunity(
            report,
            plan,
            min_tradeability=min_tradeability,
            max_transition_risk=max_transition_risk,
        )
        reports.append(report)
        plans.append(plan)
        outcomes.append(outcome)
        rows.append(_row_from_report(
            symbol,
            str(df.index[t]),
            report,
            plan,
            opportunity=opportunity,
            outcome=outcome,
            directional=directional,
        ))

    diagnostics = diagnose_regime_sequence(reports, outcomes, plans=plans).to_dict()
    opportunity_rows = [row for row in rows if row["opportunity"]]
    opportunity_quality = _opportunity_quality(opportunity_rows)
    summary = {
        "symbol": symbol,
        "timeframe": timeframe.value,
        "market": _market_stats(df),
        "quality": {
            "clean": bool(audit.clean),
            "issues": [
                {
                    "kind": issue.kind,
                    "severity": issue.severity,
                    "row_count": int(issue.row_count),
                    "message": issue.message,
                }
                for issue in audit.issues
            ],
        },
        "latest": latest,
        "scan": {
            "rows": len(rows),
            "scan_bars": int(scan_bars),
            "stride": int(stride),
            "horizon": int(horizon),
            "opportunity_count": len(opportunity_rows),
            "opportunity_rate": float(len(opportunity_rows) / max(1, len(rows))),
            "mean_forward_return": float(np.mean([r["forward_return"] for r in rows])) if rows else 0.0,
            "mean_opportunity_forward_return": float(np.mean([r["forward_return"] for r in opportunity_rows])) if opportunity_rows else 0.0,
            "mean_directional_forward_return": float(np.mean([r["directional_forward_return"] for r in rows])) if rows else 0.0,
            "mean_opportunity_directional_forward_return": float(np.mean([r["directional_forward_return"] for r in opportunity_rows])) if opportunity_rows else 0.0,
            "opportunity_hit_rate": float(np.mean([r["directional_forward_return"] > 0.0 for r in opportunity_rows])) if opportunity_rows else 0.0,
            "opportunity_quality": opportunity_quality,
        },
        "diagnostics": diagnostics,
    }
    return summary, latest, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor local real OHLCV data for regime opportunities.")
    parser.add_argument("--tsdb-root", type=str, default="data/tsdb")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols; default = all TSDB series for timeframe.")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--scan-bars", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--profile", type=str, default="auto")
    parser.add_argument("--benchmark-symbol", type=str, default="BTC/USDT")
    parser.add_argument("--min-tradeability", type=float, default=0.40)
    parser.add_argument("--max-transition-risk", type=float, default=0.55)
    parser.add_argument("--out", type=str, default="artifacts/monitor/real_data")
    args = parser.parse_args(argv)

    db = TimeSeriesDB(args.tsdb_root)
    timeframe = Timeframe.from_str(args.timeframe)
    symbols = _parse_symbols(args.symbols, db, timeframe)
    if not symbols:
        raise RuntimeError(f"No symbols found for timeframe {timeframe.value} in {args.tsdb_root}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        summary, latest, rows = _monitor_symbol(
            db,
            symbol,
            timeframe=timeframe,
            bars=args.bars,
            scan_bars=args.scan_bars,
            stride=args.stride,
            horizon=args.horizon,
            profile=args.profile,
            benchmark_symbol=args.benchmark_symbol,
            min_tradeability=args.min_tradeability,
            max_transition_risk=args.max_transition_risk,
        )
        summaries.append(summary)
        latest_rows.append(latest)
        scan_rows.extend(rows)

    payload = {
        "safety": {
            "mode": "local_tsdb_monitor_no_orders",
            "real_orders_enabled": False,
            "exchange_order_api_used": False,
            "api_keys_required": False,
        },
        "config": {
            "tsdb_root": args.tsdb_root,
            "symbols": symbols,
            "timeframe": timeframe.value,
            "bars": int(args.bars),
            "scan_bars": int(args.scan_bars),
            "stride": int(args.stride),
            "horizon": int(args.horizon),
            "profile": args.profile,
            "min_tradeability": float(args.min_tradeability),
            "max_transition_risk": float(args.max_transition_risk),
        },
        "symbols": summaries,
    }
    summary_path = out_dir / "market_monitor_summary.json"
    latest_path = out_dir / "market_monitor_latest.csv"
    scan_path = out_dir / "market_monitor_scan.csv"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(latest_rows).to_csv(latest_path, index=False)
    pd.DataFrame(scan_rows).to_csv(scan_path, index=False)

    compact = {
        "summary": str(summary_path),
        "latest": str(latest_path),
        "scan": str(scan_path),
        "symbols": [
            {
                "symbol": s["symbol"],
                "latest_regime": s["latest"]["primary_regime"],
                "latest_plan": s["latest"]["plan_status"],
                "latest_playbook": s["latest"]["recommended_playbook"],
                "latest_opportunity": s["latest"]["opportunity"],
                "scan_opportunity_rate": s["scan"]["opportunity_rate"],
                "opportunity_hit_rate": s["scan"]["opportunity_hit_rate"],
                "mean_opportunity_directional_forward_return": s["scan"]["mean_opportunity_directional_forward_return"],
                "opportunity_quality": s["scan"]["opportunity_quality"],
                "quality_clean": s["quality"]["clean"],
            }
            for s in summaries
        ],
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
