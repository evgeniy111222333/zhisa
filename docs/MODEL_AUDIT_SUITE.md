# Model Audit Suite

The audit is stage-aware. It must not turn an untrained policy head into a
convincing but meaningless Sharpe ratio.

## Stage contract

| Checkpoint | Valid claims |
|---|---|
| S1 | representation quality, collapse, modality contribution, invariance |
| S2 | supervised head quality, calibration, drift, attribution |
| S2b/S3 | imitation-policy behavior and PnL under backtest assumptions |
| S4+ | RL policy, value accuracy, leverage and tail-risk behavior |

Generate a plan before every run:

```powershell
zhisa-audit-model `
  --config configs/model_audit_60.yaml `
  --checkpoint artifacts/s2/model_best.pt `
  --prepared-root data/prepared/15m_12markets `
  --split test `
  --out artifacts/model_audit/s2_plan
```

The plan labels every check as:

- `ready`: available data and valid model stage.
- `modeled`: runnable through a simulator, but not empirically supported by
  required tick/order-book data.
- `blocked`: required data is absent.
- `not_applicable`: the checkpoint has not learned the behavior being tested.

## Corrections to the original 50

ETH, SOL and DOGE are seen-market transfer tests because they were part of the
S1 curriculum; true zero-shot testing requires held-out symbols. A negative
price transform is mathematically invalid for log-return features, so test 48
uses a positive-price return mirror and checks long/short equivariance.

Sub-bar latency requires trades or one-second data. Partial fills, spread and
capital-impact claims require L1/L2 order-book data. OHLCV-only results for
these tests are explicitly marked `modeled`.

## Added tests 51-60

The extra level covers baselines, bootstrap confidence intervals, embargoed
walk-forward stability, seed sensitivity, leakage, drift, malformed feeds,
tail risk, capacity and counterfactual consistency. Headline metrics must be
reported per symbol and regime with confidence intervals, not only as one
aggregate number.

## S2b readiness

S2b is the correct next bridge after selecting the S2 champion. The serious
path uses `configs/s2b_bc_15m_12markets.yaml`, loads all compatible S2 tensors,
reinitializes only `heads.policy_logits`, and trains jointly over prepared
market segments. DAgger remains a later environment stage; it should not use a
forward-looking triple-barrier expert in live or paper decision making.
