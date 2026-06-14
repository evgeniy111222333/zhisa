# ZHISA

**Visual-adaptive, multimodal, self-learning trading AI.**

ZHISA (Zhydovskyi Hybrid Intelligent Self-trading Agent) is a research-grade trading system that learns to trade financial instruments by combining visual chart perception, numeric feature analysis, and reinforcement learning. The full concept is described in [`CONCEPT.md`](CONCEPT.md).

## Status

This repository is a **working v0.1 MVP** of the concept. It contains:

- a 5-level learning pipeline (self-supervised → supervised multi-task → synthetic curriculum → RL → online continual);
- a multimodal model stack (vision encoder + numeric encoder + fusion + working memory + multi-task heads);
- a realistic trading environment (Gymnasium) with execution simulation, slippage, fees, and risk-shaped rewards;
- a backtest engine with walk-forward evaluation and risk metrics (Sharpe / Sortino / Calmar / drawdown);
- a synthetic market generator (used to bootstrap training without exchange connectivity);
- modular adapters for real data (CCXT-style) ready to plug in;
- a full test suite + benchmarks.

## Layout

```
zhisa/
├── CONCEPT.md                  # Architectural / concept document
├── pyproject.toml              # Package metadata
├── requirements.txt            # Pinned-ish dependencies
├── src/zhisa/                  # Source code
│   ├── config/                 # YAML config system
│   ├── data/                   # Data loaders (synthetic + real adapters)
│   ├── features/               # Numeric & categorical features
│   ├── rendering/              # Chart image renderer
│   ├── models/                 # Encoders, fusion, memory, heads, policy
│   ├── env/                    # Trading environment & execution sim
│   ├── training/               # Training loops for each learning phase
│   ├── backtest/               # Backtest engine & metrics
│   ├── risk/                   # Risk management (sizing, limits, stops)
│   ├── eval/                   # Probes & benchmarks
│   ├── utils/                  # Logging, seeding, timing
│   └── scripts/                # CLI entry points
├── tests/                      # Unit + integration tests
├── benchmarks/                 # Performance benchmarks
├── configs/                    # YAML configs (kept in repo for reference)
└── docs/                       # Additional docs
```

## Quick start

```bash
# Install (development mode, with all extras)
pip install -e ".[all]"

# Generate synthetic data
python -m zhisa.scripts.generate_data --out data/synth --bars 50000

# Run the S2 supervised multi-task training
python -m zhisa.scripts.train_s2 --config configs/s2_supervised.yaml

# Backtest a trained policy
python -m zhisa.scripts.backtest --checkpoint artifacts/s2/model.pt
```

## Real-data sandbox (no real money)

The real-data path is intentionally split into public data ingest and
simulated replay. Ingest uses public OHLCV endpoints only; paper runs
read local data and execute inside `TradingEnv`, never through an
exchange order API.

```bash
# Install exchange-data support
pip install -e ".[all]"

# 1) Pull public candles into the local parquet TSDB
zhisa-ingest-real-data \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 5m \
  --since 2024-01-01 \
  --max-bars 50000 \
  --db-root data/tsdb

# 2) Train on the latest real bars from TSDB
zhisa-train-s2 \
  --data-source tsdb \
  --tsdb-root data/tsdb \
  --symbol BTC/USDT \
  --timeframe 5m \
  --bars 20000 \
  --epochs 2 \
  --checkpoint artifacts/s2/btc_real.pt

# 3) Run a no-money replay and inspect metrics / decisions
zhisa-paper-run \
  --data-source tsdb \
  --tsdb-root data/tsdb \
  --symbol BTC/USDT \
  --timeframe 5m \
  --bars 5000 \
  --checkpoint artifacts/s2/btc_real.pt \
  --out artifacts/paper_run/btc_5m

# 4) Pull richer public USD-M futures context
zhisa-ingest-binance-futures-context \
  --symbols BTC/USDT,ETH/USDT,SOL/USDT \
  --timeframe 5m \
  --start 2026-05-16T00:00:00Z \
  --end 2026-06-14T20:00:00Z \
  --out-root data/futures_context/binance_usdm \
  --audit-out artifacts/real_data/binance_futures_context_audit.json

# 5) Monitor local real-data regimes and opportunity candidates
zhisa-monitor-real-data \
  --tsdb-root data/tsdb \
  --symbols BTC/USDT,ETH/USDT,SOL/USDT \
  --timeframe 5m \
  --bars 5000 \
  --scan-bars 1000 \
  --out artifacts/monitor/crypto_5m

# 6) Watch the live market and send decisions to a local paper broker
zhisa-live-shadow \
  --exchange binance_usdm \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --timeframe 5m \
  --duration-sec 300 \
  --broker local_paper \
  --out artifacts/live_shadow/binance_usdm_paper

# Optional: mirror filled local-paper orders to OKX demo trading only.
# Requires OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE from an OKX
# simulated account. The adapter sends x-simulated-trading: 1.
zhisa-live-shadow \
  --exchange okx_demo \
  --symbols BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP \
  --timeframe 5m \
  --duration-sec 300 \
  --broker okx_demo \
  --okx-fixed-size 1 \
  --i-understand-okx-demo-orders \
  --out artifacts/live_shadow/okx_demo_mirror
```

The Binance futures context ingest stores public, no-key derivatives
metrics such as futures candles, mark/index/premium klines, funding,
open interest, long/short ratios, and taker buy/sell flow under
`data/futures_context/binance_usdm/{SYMBOL}/{timeframe}/context.parquet`.
It also writes an audit JSON with exact columns, ranges, null counts, and
known unavailable endpoints.

`zhisa-live-shadow` listens to public live WebSocket market data, builds
closed-bar decisions, executes them in a local paper ledger, and writes
events, bars, decisions, orders, equity, and resolved experience samples.
It never needs API keys in `local_paper` mode. OKX demo mirroring is
explicit opt-in and requires simulated-account keys.

The same `--data-source tsdb|csv` arguments are available on S1, S2,
S2b, S4, S4-CVaR, `zhisa-backtest`, and `zhisa-eval`.

## Tests

```bash
pytest                                # full suite
pytest tests/test_features.py -v      # one module
pytest -m benchmark --benchmark-only  # benchmarks only
```

## Design principles

- **Modular over monolithic.** Every concern (data, features, models, env, risk, eval) lives in its own package.
- **No look-ahead, ever.** All features and labels are strict-lagged. Triple-barrier labeling follows the López de Prado methodology.
- **Risk-aware by default.** The reward function, training objectives, and runtime guards all penalize drawdowns, turnover, and over-leverage.
- **Tested, benchmarked, reproducible.** Every module has unit tests, performance-critical paths have benchmarks, and seeding is honoured throughout.

See `CONCEPT.md` for the full design rationale.
