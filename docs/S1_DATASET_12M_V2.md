# S1 12-Market Multi-Timeframe Dataset

## Source and cutoff

The source is Binance public Spot OHLCV. No API keys and no order endpoints
are used. Binance documents public daily/monthly archives and all supported
kline intervals at:

- https://github.com/binance/binance-public-data
- https://data.binance.vision/

The local downloader uses the public Spot kline API and stores only fully
closed candles. This revision has a fixed 2026-06-20 cutoff, so it remains
reproducible after newer exchange data becomes available.

## Universe

BTC, ETH, SOL, BNB, ADA, XRP, DOGE, LINK, AVAX, LTC, DOT, and TRX against
USDT. The extension adds liquid markets with different behavior: meme,
oracle, proof-of-work, and additional layer-1 assets. Stablecoin pairs and
very young listings are excluded.

## Prepared roots

| Root | Timeframe | Rows | Train | Validation | Test | Checksum |
|---|---:|---:|---:|---:|---:|---|
| `data/prepared/s1_1h_12m_v2` | 1h | 601,452 | 420,612 | 90,120 | 90,144 | `b874977231a7cb6cdb6209ccf5b90934680870d17aaf273c4f6484872b49a54c` |
| `data/prepared/s1_15m_12m_v2` | 15m | 2,405,628 | 1,682,316 | 360,492 | 360,516 | `dc39d70e730a544f3f27c828d30e80a45c686d8c94a95b8dc6a3873b832bacf5` |

Both roots cover 2020-10-01 through 2026-06-20. Splits are temporal and
per-symbol, with at least 24 hours of embargo. The test files are not opened
during S1 training or checkpoint selection.

## Quality contract

- 12 symbols in every split.
- Identical 32-feature numeric and 10-feature calendar context contracts.
- No NaN, infinity, duplicate timestamp, or OHLC invariant violation.
- Long exchange outages are not synthesized. The training loader splits each
  symbol into contiguous segments, so chart windows and temporal pairs never
  cross a missing interval.
- Futures context is intentionally disabled in this core representation
  dataset. Adding it for only some markets would create inconsistent schemas
  and symbol-dependent missingness. It should be prepared as a separate,
  uniformly available market-context stage.

## Training curriculum

Phase A learns broad market structure at 1h:

```bash
python -m zhisa.scripts.train_s1 \
  --config configs/s1_ssl_1h_12m.yaml \
  --prepared-root data/prepared/s1_1h_12m_v2 \
  --checkpoint artifacts/s1/12m/phase1_last.pt \
  --best-checkpoint artifacts/s1/12m/phase1_best.pt \
  --workers 0 --fast-render
```

Phase B specializes the same encoder at 15m:

```bash
python -m zhisa.scripts.train_s1 \
  --config configs/s1_ssl_15m_12m.yaml \
  --prepared-root data/prepared/s1_15m_12m_v2 \
  --resume-from artifacts/s1/12m/phase1_best.pt \
  --reset-best-on-resume \
  --checkpoint artifacts/s1/12m/phase2_last.pt \
  --best-checkpoint artifacts/s1/12m/phase2_best.pt \
  --workers 0 --fast-render
```

On Linux/AWS, worker count can be benchmarked upward after memory profiling.
The conservative command uses zero workers and a disabled chart cache to
avoid multiplying dataset memory across worker processes.
