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
