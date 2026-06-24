# S2 Multi-Timeframe 15m + 1h Plan

## Goal

S2 must keep the 15m stream as the execution/entry timeframe and use 1h candles only as closed higher-timeframe context. The model should learn:

- 15m local structure: entry pressure, short-horizon direction, wick/body/volume behavior.
- 1h macro context: trend, volatility regime, drawdown/risk background, broad market state.
- The difference between timeframes explicitly, so a 15m candle is never interpreted as a 1h candle.

This stage is still not a live/paper policy. It is a supervised representation and market-head stage for S2b/S4.

## Data Contract

- Every sample is anchored at a 15m primary bar `t`.
- The existing 15m `chart`, `numeric`, and `context` remain unchanged.
- A new `macro_numeric` tensor is added with shape `(macro_window, macro_features)`.
- `macro_numeric` is built from 1h OHLCV features derived causally from the same 15m stream unless an external prepared 1h source is explicitly wired later.
- For a 15m bar with timestamp `t`, the newest allowed 1h candle is the last fully closed 1h candle. With open-time indexed candles this is `floor(t, 1h) - 1h`.
- Current/incomplete 1h candles are forbidden.
- Missing early macro history is left-padded with zeros.
- Local and macro streams have separate normalization windows and separate model encoders.

## Model Contract

- Keep the original 15m visual/numeric/context path.
- Add a separate 1h numeric encoder.
- Add explicit timeframe embeddings:
  - local 15m embedding is added to the 15m numeric embedding,
  - macro 1h embedding is added to the 1h numeric embedding.
- Fuse local modalities first, then inject 1h context through a learned gate.
- Warm-start from the current S2 checkpoint must load all compatible 15m/backbone/head weights and initialize only the new macro branch randomly.

## Leakage Tests

Required tests before any serious run:

- Dataset returns `macro_numeric` only when macro context is enabled.
- `macro_numeric` shape is stable across collate.
- Mutating future/current-hour 15m bars must not change the macro context for an earlier 15m sample.
- `PolicyNetwork` forward works both with and without `macro_numeric`.
- Old S1/S2 checkpoints can warm-start a macro-enabled S2 model without losing compatible weights.

## Expected Metric Targets

Baseline current honest S2 readiness is about `0.633`.

Minimum useful result:

- Stage readiness score: `>= 0.68`.
- h4 balanced accuracy: from `~0.373` to `>= 0.390`.
- h16 balanced accuracy: from `~0.357` to `>= 0.375`.
- primary macro-F1: from `~0.355` to `>= 0.370`.
- primary ECE: `<= 0.045`.
- h4 return correlation: `>= 0.030`.
- h16 return correlation: `>= 0.025`.
- h64 return correlation: `>= 0.000`.
- vol/risk correlation: keep `>= 0.650`, target `>= 0.700`.
- No market should have negative persistence lift on direction after calibration.

Good result:

- Stage readiness score: `>= 0.72`.
- h4/h16 balanced accuracy `>= 0.40 / >= 0.385`.
- primary macro-F1 `>= 0.385`.
- return correlations positive on most markets, including weak symbols.
- TRX/AVAX/LTC/ADA must not become isolated failure cases.

## Ablation Requirements

To prove that 1h helps rather than adding noise:

- `15m + causal 1h` must beat `15m-only` on validation readiness.
- Shuffled/misaligned 1h context should hurt or at least not improve metrics.
- 1h-only should be useful for regime/volatility but worse for short-horizon entries.
- If 1h improves vol/risk but hurts h4/h16 direction, it is not ready for downstream S2b/S4.

## Reject Conditions

Reject the multi-timeframe checkpoint if any of these happen:

- Stage readiness is not better than current S2 champion.
- Direction improves only by predicting one class more often.
- ECE worsens above `0.055`.
- h4/h16 return correlations collapse toward zero or negative.
- Per-market diagnostics show new severe regressions on previously healthy symbols.
- Leakage tests fail or alignment cannot be proven.

