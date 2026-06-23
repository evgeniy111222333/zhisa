from __future__ import annotations

from zhisa.model_audit.schema import ModelStage, TestSpec


POLICY = (ModelStage.S2B, ModelStage.S3, ModelStage.S4, ModelStage.S5_PLUS)
REPRESENTATION = (ModelStage.S1, ModelStage.S2, *POLICY)
SUPERVISED = (ModelStage.S2, *POLICY)
RL = (ModelStage.S4, ModelStage.S5_PLUS)


def _t(
    id: int,
    key: str,
    title: str,
    level: int,
    objective: str,
    handler: str,
    *,
    caps: tuple[str, ...] = ("ohlcv", "timestamps"),
    stages: tuple[ModelStage, ...] = POLICY,
    policy: bool = True,
    empirical: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> TestSpec:
    return TestSpec(id, key, title, level, objective, handler, caps, stages, policy, empirical, tags)


def build_catalog() -> list[TestSpec]:
    tests = [
        _t(1, "absolute_pnl", "Absolute PnL", 1, "Net strategy return", "performance"),
        _t(2, "win_rate_by_side", "Win rate by side", 1, "Long/short hit rate", "performance"),
        _t(3, "profit_factor", "Profit factor", 1, "Gross profit divided by gross loss", "performance"),
        _t(4, "max_drawdown", "Maximum drawdown", 1, "Peak-to-trough equity loss", "performance"),
        _t(5, "sharpe", "Sharpe ratio", 1, "Risk-adjusted return", "performance"),
        _t(6, "sortino", "Sortino ratio", 1, "Downside-risk-adjusted return", "performance"),
        _t(7, "calmar", "Calmar ratio", 1, "Annual return per drawdown", "performance"),
        _t(8, "trade_duration", "Average trade duration", 1, "Holding-time behavior", "trade_anatomy"),
        _t(9, "loss_streak", "Maximum consecutive losses", 1, "Loss clustering", "trade_anatomy"),
        _t(10, "exposure", "Exposure time", 1, "Time in market versus cash", "trade_anatomy"),
        _t(11, "bull_2024", "Bull Run 2024", 2, "Behavior in a broad bull regime", "historical_window"),
        _t(12, "winter_2022", "Crypto Winter 2022", 2, "Behavior in a bear regime", "historical_window"),
        _t(13, "crab_2023", "Crab Market 2023", 2, "Whipsaw resistance", "historical_window"),
        _t(14, "flash_crash_2021", "May 2021 crash", 2, "Crash survival", "historical_window"),
        _t(15, "covid_recovery", "COVID V-shape recovery", 2, "Crash and reversal adaptation", "historical_window", caps=("ohlcv", "timestamps", "history_2020_03")),
        _t(16, "weekends", "Weekend trading", 2, "Low-liquidity calendar behavior", "calendar_slice"),
        _t(17, "volatility_buckets", "High vs low volatility", 2, "Risk adaptation across ATR buckets", "conditional_slice"),
        _t(18, "funding_extremes", "Funding extremes", 2, "Positioning under crowded funding", "conditional_slice", caps=("ohlcv", "timestamps", "funding")),
        _t(19, "liquidation_cascades", "Liquidation cascades", 2, "Behavior around forced-flow events", "event_slice", caps=("ohlcv", "timestamps", "liquidations")),
        _t(20, "holiday_volume", "Holiday volume", 2, "Thin-market holiday behavior", "calendar_slice"),
        _t(21, "zero_fee", "Zero-fee baseline", 3, "Upper execution bound", "execution_scenario"),
        _t(22, "taker_fee", "Taker fee", 3, "Fee sensitivity", "execution_scenario"),
        _t(23, "slippage", "Slippage impact", 3, "Market-order slippage sensitivity", "execution_scenario"),
        _t(24, "latency_500ms", "Micro-latency 500ms", 3, "Sub-second delay sensitivity", "execution_scenario", caps=("ohlcv", "timestamps"), empirical=("trades_1s",)),
        _t(25, "latency_5s", "Severe latency 5s", 3, "API delay sensitivity", "execution_scenario", caps=("ohlcv", "timestamps"), empirical=("trades_1s",)),
        _t(26, "partial_fills", "Partial fills", 3, "Incomplete order execution", "execution_scenario", empirical=("orderbook_l2",)),
        _t(27, "spread_widening", "Spread widening", 3, "News-time spread shock", "execution_scenario", empirical=("orderbook_l1",)),
        _t(28, "missed_signals", "Missed signals", 3, "Signal delivery failures", "execution_scenario"),
        _t(29, "margin_stress", "Margin-call stress", 3, "Leverage and liquidation proximity", "execution_scenario", stages=RL),
        _t(30, "market_impact", "Market impact at 10M USD", 3, "Capacity under own flow", "execution_scenario", empirical=("orderbook_l2",)),
        _t(31, "confidence_calibration", "Confidence calibration", 4, "Probability reliability via ECE/Brier", "calibration", stages=SUPERVISED, policy=False),
        _t(32, "action_entropy", "Action entropy", 4, "Policy decisiveness", "action_diagnostics"),
        _t(33, "action_churn", "Action churn", 4, "Unnecessary position reversals", "action_diagnostics"),
        _t(34, "no_chart", "Blindness: no chart", 4, "Visual modality contribution", "ablation", stages=REPRESENTATION, policy=False),
        _t(35, "no_numerics", "Blindness: no numerics", 4, "Numeric modality contribution", "ablation", stages=REPRESENTATION, policy=False),
        _t(36, "no_context", "Blindness: no context", 4, "Calendar/context contribution", "ablation", stages=REPRESENTATION, policy=False),
        _t(37, "fake_wick", "Fake wick attack", 4, "Robustness to candle corruption", "perturbation", stages=REPRESENTATION, policy=False),
        _t(38, "volume_spoof", "Volume spoofing", 4, "Robustness to false volume", "perturbation", stages=REPRESENTATION, policy=False),
        _t(39, "attention_maps", "Attention and attribution maps", 4, "Input attribution around price extrema", "interpretability", stages=REPRESENTATION, policy=False),
        _t(40, "value_accuracy", "State-value accuracy", 4, "Predicted value versus realized return", "value_accuracy", stages=RL),
        _t(41, "eth_transfer", "ETH transfer", 5, "Per-market generalization; seen during S1", "market_slice"),
        _t(42, "sol_transfer", "SOL transfer", 5, "Per-market generalization; seen during S1", "market_slice"),
        _t(43, "doge_transfer", "DOGE transfer", 5, "Per-market generalization; seen during S1", "market_slice"),
        _t(44, "tradfi", "Gold and S&P 500", 5, "Cross-asset-domain transfer", "external_dataset", caps=("ohlcv", "timestamps", "tradfi")),
        _t(45, "timeframe_5m", "Timeframe shift 5m", 5, "Transfer to shorter bars", "external_dataset", caps=("ohlcv", "timestamps", "timeframe_5m")),
        _t(46, "timeframe_1h", "Timeframe shift 1h", 5, "Transfer to longer bars", "external_dataset", caps=("ohlcv", "timestamps", "timeframe_1h")),
        _t(47, "ood_2018", "Out-of-distribution 2018", 5, "Transfer to an old market microstructure", "external_dataset", caps=("ohlcv", "timestamps", "history_2018")),
        _t(48, "price_mirror", "Price-mirror equivariance", 5, "Long/short consistency under a valid positive-price mirror", "perturbation", stages=REPRESENTATION, policy=False),
        _t(49, "pure_noise", "Pure-noise baseline", 5, "Detect spurious action edge", "perturbation", stages=REPRESENTATION, policy=False),
        _t(50, "cross_exchange", "Cross-exchange transfer", 5, "Venue generalization", "external_dataset", caps=("ohlcv", "timestamps", "cross_exchange")),
        _t(51, "baselines", "Strategy baselines", 6, "Compare against cash, buy-and-hold, random and rules", "baselines"),
        _t(52, "bootstrap_ci", "Bootstrap confidence intervals", 6, "Uncertainty and significance of all headline metrics", "statistics"),
        _t(53, "walk_forward", "Walk-forward stability", 6, "Stability across embargoed chronological folds", "walk_forward"),
        _t(54, "seed_stability", "Seed stability", 6, "Sensitivity to stochastic execution and policy seeds", "statistics"),
        _t(55, "leakage_audit", "Leakage and embargo audit", 6, "Detect overlap, future features and split contamination", "data_integrity", stages=REPRESENTATION, policy=False),
        _t(56, "drift", "Feature and prediction drift", 6, "PSI/KS drift by time and market", "drift", stages=SUPERVISED, policy=False),
        _t(57, "feed_faults", "Missing/stale/duplicate feed", 6, "Operational robustness to malformed data", "perturbation", stages=REPRESENTATION, policy=False),
        _t(58, "tail_risk", "Tail risk and recovery", 6, "VaR, CVaR, ulcer index and recovery time", "performance"),
        _t(59, "capacity_turnover", "Turnover and capacity curve", 6, "Fee break-even and capital scalability", "execution_scenario", empirical=("orderbook_l2",)),
        _t(60, "counterfactual_consistency", "Counterfactual consistency", 6, "Monotonicity of risk, confidence and position sizing", "counterfactual", stages=SUPERVISED, policy=False),
    ]
    assert [test.id for test in tests] == list(range(1, 61))
    return tests
