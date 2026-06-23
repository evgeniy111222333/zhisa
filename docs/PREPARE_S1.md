# Preparing the S1 dataset — operator guide

This document explains how to run the **S1 data preparation pipeline** on
the machine that will actually do the training (typically a workstation
or remote box with GPU). It assumes the S1 training run itself happens
on a different machine from where the raw data was collected.

The preparation step is **deterministic and idempotent**: running it
twice on the same inputs produces the same SHA-256 manifest checksum,
so the resulting S1 checkpoint can be reproduced on any other machine
that runs the same preparation against the same raw input.

---

## 1. What it does

Given the local TimeSeriesDB (`data/tsdb/...`) and the optional
futures context files (`data/futures_context/binance_usdm/...`), the
pipeline:

1. **Loads** OHLCV at the target timeframe (15m default).
2. **Repairs** the data (`audit_ohlcv` + `repair_ohlcv`): drops dups,
   forward-fills NaNs, clamps OHLC constraints, fixes zero-volume bars.
3. **Gap policy**: reindexes onto a strict 15-min grid. Short gaps
   (≤ `max_ffill_bars`) are forward-filled; longer gaps are dropped.
4. **Coverage alignment**: clips every symbol to a shared time window
   (`start = max(per-symbol starts)`, `end = min(per-symbol ends)`).
   Symbols with fewer than `min_bars_per_symbol` are dropped.
5. **Context merge**: left-joins Binance USD-M futures context
   (funding, OI, long/short ratios, taker flow) with a 1-bar
   anti-look-ahead shift.
6. **Schema assert**: every output frame must be tz-aware UTC, have
   all five OHLCV columns numeric with no NaN/Inf, and a monotonic
   index.
7. **Checksum**: SHA-256 of inputs and outputs is written to disk.
8. **Splits**: per-symbol temporal train/val/test split (70/15/15 by
   default) with an embargo gap between splits.

Outputs (under `--out-root`):

```
manifest.json              # version + checksums + row counts
symbols/{SYMBOL}.parquet   # one frame per symbol, fully cleaned
splits/{train,val,test}.parquet
checksums.txt              # human-readable input/output checksums
preparation_log.json       # full audit trail
```

---

## 2. Prerequisites

* Python ≥ 3.10
* The `zhisa` package installed in development mode:
  ```bash
  pip install -e ".[all]"
  ```
* The local TSDB populated with the symbols and timeframes you want
  to train on. The minimum dataset for the S1 default run is:

  ```
  data/tsdb/
    BTC/USDT/15m/data.parquet   # ≥ 200k bars recommended
    ETH/USDT/15m/data.parquet
    SOL/USDT/15m/data.parquet
    BNB/USDT/15m/data.parquet
    ADA/USDT/15m/data.parquet
    XRP/USDT/15m/data.parquet
  ```

* Optional: futures context files for symbols you want to enrich with
  funding / OI / long-short data:
  ```
  data/futures_context/binance_usdm/
    BTCUSDT/5m/context.parquet
    ETHUSDT/5m/context.parquet
    SOLUSDT/5m/context.parquet
  ```
  If you have `BTC_USDT` slug, the merger tries both `BTCUSDT` and
  `BTC_USDT` automatically. **Only BTC/ETH/SOL have these files in the
  current snapshot** — the other three symbols are prepared without
  futures context.

---

## 3. Running it

### 3.1. Dry-run (recommended first step)

```bash
zhisa-prepare-s1-data --dry-run
```

This prints the resolved configuration without writing any files. Use
it to confirm your CLI arguments are wired correctly before touching
disk.

### 3.2. Standard S1 run (6 symbols, 15m, with futures context)

```bash
zhisa-prepare-s1-data \
  --tsdb-root data/tsdb \
  --out-root  data/prepared/s1_15m_v1 \
  --symbols   BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT,XRP/USDT \
  --timeframe 15m \
  --with-futures-context \
  --context-root data/futures_context/binance_usdm \
  --train-frac 0.70 --val-frac 0.15 --test-frac 0.15 \
  --embargo-bars 96
```

Expected runtime: ~30–90 seconds on the current snapshot (6 symbols,
~1.5M total bars, 3 with futures context).

### 3.3. Conservative subset (BTC + ETH only, no futures context)

```bash
zhisa-prepare-s1-data \
  --tsdb-root data/tsdb \
  --out-root  data/prepared/s1_btc_eth_v1 \
  --symbols   BTC/USDT,ETH/USDT \
  --timeframe 15m \
  --no-futures-context \
  --embargo-bars 96
```

Useful for a fast first training run while you verify the rest of the
S1 pipeline.

### 3.3. Custom coverage window

If you want to exclude the earliest (low-liquidity) months or focus on
the post-2022 stressed-period window:

```bash
zhisa-prepare-s1-data \
  --coverage-start 2022-01-01 \
  --coverage-end   2025-12-31 \
  --out-root data/prepared/s1_post2022_v1
```

### 3.4. Adjusting the gap policy

The defaults are conservative:

* `--max-ffill-bars 4` (1 hour at 15m) — typical Binance maintenance
  window. Increase to 8 if you want to tolerate longer outages (with
  the risk that the forward-fill becomes stale).
* `--keep-long-gaps` — keep the rows that follow a long gap as NaN
  instead of dropping them. Useful if you have very long histories and
  do not want to lose data; downstream code handles NaN.

---

## 4. Verifying the output

After the run, check:

```bash
# 1. Manifest exists and is valid JSON
cat data/prepared/s1_15m_v1/manifest.json | python -m json.tool

# 2. Row counts look reasonable (1.5M total expected with 6 symbols)
python -c "
import json
m = json.load(open('data/prepared/s1_15m_v1/manifest.json'))
print('symbols:', m['symbols'])
print('rows_per_symbol:', m['rows_per_symbol'])
print('window:', m['start'], '..', m['end'])
print('checksum:', m['output_checksum'])
"

# 3. All per-symbol parquet files load
python -c "
import pandas as pd
from pathlib import Path
for p in Path('data/prepared/s1_15m_v1/symbols').glob('*.parquet'):
    df = pd.read_parquet(p)
    print(p.name, 'rows=', len(df), 'cols=', list(df.columns))
"

# 4. Splits are disjoint and complete
python -c "
import pandas as pd
train = pd.read_parquet('data/prepared/s1_15m_v1/splits/train.parquet')
val   = pd.read_parquet('data/prepared/s1_15m_v1/splits/val.parquet')
test  = pd.read_parquet('data/prepared/s1_15m_v1/splits/test.parquet')
print(f'train={len(train):,}  val={len(val):,}  test={len(test):,}')
assert len(set(train['symbol']).intersection(val['symbol'])) == 6
print('symbols in splits:', sorted(train['symbol'].unique()))
"
```

---

## 5. Training S1 from the prepared data

For the current 12-market, `1h -> 15m` curriculum and its audited checksums,
see [S1_DATASET_12M_V2.md](S1_DATASET_12M_V2.md).

### Current canonical path

Use the prepared split path directly. The trainer builds one dataset per
symbol, trains only on `train.parquet`, selects `best.pt` on `val.parquet`,
and never opens `test.parquet`.

```bash
python -m zhisa.scripts.train_s1 \
  --config configs/s1_ssl_clean.yaml \
  --prepared-root data/prepared/s1_15m_v1 \
  --checkpoint artifacts/s1/clean/last.pt \
  --best-checkpoint artifacts/s1/clean/best.pt \
  --workers 0 \
  --fast-render
```

This clean command intentionally does not load the legacy `model_epoch2.pt`:
that model was trained on direct full-history data, including periods now
assigned to validation/test, and cannot support a clean holdout claim. It can
still be used for an exploratory salvage run with
`configs/s1_ssl_continue.yaml` and `--resume-from`, but not as the final S1
quality benchmark.

Checkpoints produced by the corrected trainer contain optimizer and trainer
state. For a later restart, point `--resume-from` at `last.pt` and require the
log to report `resume_mode: full` and `optimizer_restored: True`. Keep
`--chart-cache-size -1` on large runs unless RAM use has been measured.

### Historical path (obsolete)

The S1 trainer reads per-symbol frames through
`load_market_dataframe` (`--data-source csv`). The cleanest way to wire
the prepared data in is a small driver script:

```python
# scripts/run_s1_on_prepared.py
from torch.utils.data import ConcatDataset
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.scripts._real_data import normalize_ohlcv_frame

spec = SampleSpec(chart_window=128, feature_window=128, image_size=128)
datasets = []
for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT"]:
    path = f"data/prepared/s1_15m_v1/symbols/{symbol.replace('/', '_')}.parquet"
    df = normalize_ohlcv_frame(pd.read_parquet(path))
    datasets.append(MarketDataset(df, spec=spec, compute_targets=False))
ds = ConcatDataset(datasets)
# … then build the SSLPretrainer and call fit(ds).
```

This is the **same path** `zhisa-train-s1` already takes — the script
loops over symbols and builds a `ConcatDataset` from
`MarketDataset` instances.

> **Note.** The shipped `train_s1.py` defaults to `--data-source
> synthetic` for tests and `--bars 8000` to bound memory. For a real
> run you want `--data-source tsdb` and `--bars` left unset (i.e. use
> the full history), or set it to the prepared symbol's row count.

---

## 6. Reproducing a run elsewhere

To make a checkpoint comparable across machines:

1. Copy `data/tsdb/{SYMBOL}/{TF}/data.parquet` (raw inputs) and
   `data/futures_context/binance_usdm/...` to the new machine.
2. `pip install -e ".[all]"` there.
3. Run the same `zhisa-prepare-s1-data` invocation with the **same
   arguments**.
4. Compare `manifest.json → output_checksum` between the two machines.
   Identical checksums ⇒ identical prepared dataset, so identical S1
   training trajectory (modulo non-deterministic GPU kernels).

---

## 7. Troubleshooting

**`Series not found: BTC/USDT@15m`** — the TSDB is missing the
timeframe. Either run
`zhisa-ingest-real-data --symbol BTC/USDT --timeframe 15m --since 2019-01-01 --max-bars 260000`
or change `--timeframe` to one you already have.

**`coverage alignment removed every symbol`** — every symbol dropped
because `min_bars_per_symbol` is too high for the chosen window.
Either lower `--min-bars-per-symbol`, or use `--coverage-start` /
`--coverage-end` to widen the window.

**`index must be tz-aware UTC`** — the loaded frame is tz-naive. This
should never happen for data ingested through `zhisa-ingest-real-data`,
but if you fed in a manually-built CSV, ensure it has a tz-aware
`DatetimeIndex`.

**Futures context file not found for some symbols** — expected. The
ingest pipeline only produced context for BTCUSDT, ETHUSDT, SOLUSDT.
The other three symbols will be prepared without context; the
manifest's `stages.context_merge.skipped` field lists them.

**NaN rows in OHLCV after preparation** — something upstream produced
a longer gap than `max_ffill_bars`. Either raise
`--max-ffill-bars` or investigate the source data with
`zhisa-monitor-real-data`.

---

## 8. Files written by this pipeline (summary)

| File | Purpose |
|---|---|
| `manifest.json` | Version, symbols, row counts, checksums, gap/coverage policies |
| `symbols/{SYMBOL}.parquet` | One cleaned + context-merged frame per symbol |
| `splits/{train,val,test}.parquet` | Combined temporal splits across all symbols |
| `checksums.txt` | Human-readable checksum summary (input + output + manifest) |
| `preparation_log.json` | Full audit trail: repair report, gap stats, alignment info, split ranges |

The S1 trainer consumes `symbols/*.parquet`; everything else is for
debugging and reproducibility.
