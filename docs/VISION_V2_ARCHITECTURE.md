# VISION v2 — Architectural Concept: ColumnFormer + token fusion

Status: **concept / design only (no implementation yet)** · Scope: A (bar-column
ColumnFormer), B (position-aware pooling), C (token-level cross-modal fusion),
D (causality), E (frequency branch).

---

## 0. Why we are rethinking vision

The shipped S1 vision encoder is a small stride-2 CNN with **global average
pooling** (channels 32→64→128→192, 128×128 input → 8×8 grid → avg → 384-d).
Measured weaknesses:

1. **1 px/bar + aggressive downsample (8×)** — the first conv already sees ~3 bars;
   the final receptive field ≈ the whole chart. Candle-level detail is gone after
   one stride layer.
2. **Global average pooling erases *where* a pattern sits in time.** For a chart
   the position along the window is semantic (old-left vs recent-right). An avg
   over 8×8 → a translation-invariant-ish blob. CNNs are shift-equivariant, which
   is wrong for a time axis.
3. **No multi-scale**, no macro-vs-micro context, no explicit time axis.
4. **Asymmetry with numeric**: numeric is a transformer over 32 patch-tokens;
   vision is a pooled blob. The current fusion mixes a *vector* with a *CLS
   token* — no token-level interplay.

Empirical confirmation (S1 probe): **vision↔numeric alignment cos ≈ 0** even
though the SSL alignment loss is low — consistent with "vision is too coarse to
form a shared structural space with numeric."

---

## 1. Design goals & non-goals

Goals
- Let the visual stream carry **spatial/temporal structure**: where in the
  window, how candles relate left→right, macro trend + micro wick/body.
- Share the **same time axis and token granularity** between vision and numeric
  so cross-modal fusion can be *token-level*.
- Keep the compiled-chart pipeline untouched (image size, render identity).
- **Gate every change by the S1 probe** before it is kept.

Non-goals (this round)
- No resolution increase (image stays 128×128 for disk) and no real-time render.
- No pretrained vision backbones (footprint/compat cost not justified for a
  tiny domain-specific input).
- No paper money — this is representation quality for S1→S2.

---

## 2. A — ColumnFormer: bar-column tokens (core)

### 2.1 Intuition
A chart is a **1-D sequence over time drawn in 2-D**: bars run along x, price
along y. The natural tokens are the **columns** — each column is one bar's
vertical slice (candle body/wicks + volume bar + overlays in that x-position).
"128 columns × features" gives exactly the **same sequence length (128) as the
numeric window (128 bars)** — perfect time-axis parity.

### 2.2 Token construction (two sub-panels per bar)
The rendered image splits cleanly into:

- **Price panel** rows `[0, price_h)` — candles, wicks, SMA overlays.
- **Volume panel** rows `[price_h, size)` — volume bars (when enabled).

Per bar-column `x` we build two sub-embeddings:

```
e_price(x) = Linear( price_rows * channels → d )      # candle/wick/overlay slice
e_vol(x)   = Linear( volume_rows * channels → d_sub ) # volume slice
t(x)       = Linear( [e_price; e_vol] → d )           # one bar-token
```

- If volume (or overlays) are disabled, the empty side is zero-filled
  deterministically (no schema drift across configs).
- A `2-token-per-bar` variant (256 tokens) is kept as a stretch option.

Tokens: `T ∈ B × 128 × d`.

### 2.3 Backbone
- **Pre-LN transformer** over the 128 column tokens (mirrors the numeric recipe).
- `d = 384` (aligned with heavy `embed_dim`), `L = 4`, `H = 6–8`, `dim_ff = 4·d`,
  dropout 0.1, **sinusoidal positional embeddings along x** (same family as the
  numeric `_SinPositionalEmbedding`).
- Reader: `CLS` token **or** attention pooling (see B). With C enabled, the
  **token sequence** is forwarded to fusion instead of a pooled vector.

### 2.4 Why this fixes the weaknesses
- **Time is explicit**: each token is a position; attention can relate bar `t`
  to `t-k` (macro trend) while the token itself keeps bar-local detail (micro).
- **No global-pool wash-out**: reader aggregates with learned focus.
- **Micro + macro in one stack**: per-bar token holds micro; attention gives macro.
- **Time-axis parity** (128 ↔ 128) → enables C.

### 2.5 Cost
- Projector ≈ `channels·size·d` ≈ ~0.15M.
- Transformer ≈ `4·L·d²` ≈ **+6–8M** at d=384, L=4.
- Heavy-v2 vision ≈ 7–9M (v1 CNN ≈ 1.4M).
- VRAM: activations for `128×d` tokens are tiny vs CNN conv maps — equal or lower;
  fits L4 comfortably.

## 3. B — Position-aware pooling (replaces global-avg)

Two compatible placements:

1. **ColumnFormer reader**: a **learned query** `q ∈ d` attends over the 128
   tokens (softmax attention → weighted sum). Cost ≈ `d` params + `128·d`
   attention. Positional preference stays learnable; cheaper than a CLS+extra block.
2. **Minimal-CNN variant** (fallback if ColumnFormer is not adopted): keep the
   4-layer CNN but replace `AdaptiveAvgPool2d(1)` with attention pooling over the
   8×8 grid (flatten → query attends 64 spatial cells). Nearly free (~+0.01M).

The reader is a **soft "which region matters"** decision, not a blind mean. The
pooled vector feeds heads (if C is off) or the fusion context (if C is on).

---

## 4. C — Token-level cross-modal fusion (the real jump)

### 4.1 Current fusion
Takes three **vectors** (vision-pooled, numeric-CLS, context) into a small
transformer — all token detail is lost.

### 4.2 Proposed
Fusion becomes a **joint transformer over token streams**:

```
tokens = [ vision columns   (128 tokens, d)      # from A
         , numeric patches  (CLS + 32 patch-tokens, d)   # from NumericEncoder
         , context          (1 token, d)          # from ContextEncoder
         ]
```

- Same `d = 384` everywhere (both streams already produce d-space).
- **Type markers**: small learnable embeddings
  `IS_VISION / IS_NUMERIC / IS_CONTEXT` added to each token (segment-like).
- **Symmetric cross-attention**: every vision token can attend to every numeric
  token and vice-versa; the extra context token anchors both.
- **Output**: a fresh `CLS`/attention-pool over the fused stream feeds the
  WorkingMemory and heads.
- Layers ~2–4; cost ≈ **+6–8M**.

### 4.3 Why C matters
- The SSL **alignment** objective operates on *vectors*; token fusion lets the
  model align at the level of *individual bars* (a specific candle ↔ the same
  bar's numeric features) — a far richer coupling than a single cosine push.
- It directly raises the probe's `alignment cos` and gives downstream heads
  access to joint structure, not two denatured pools.

## 5. D — Causality in time

The chart image only ever contains *past* bars (the window is strictly
historical). Still, a **causal mask** over the bar-column tokens rows
right-to-left attention, mirroring the numeric stream and the spirit of "no
look-ahead":

- Left bar may attend to earlier bars only; the newest (rightmost) bar is the
  one aligned to the current numeric state.
- Implementation: triangular mask in ColumnFormer (and optionally numeric),
  parameter count unchanged.
- Opens the door for a future *online* reuse (streaming inference) where the
  rightmost token is always "now".

Note: causal masking in vision is **optional**; the probe will tell us whether
it changes alignment/CPC materially. Cheap and coherent, so it is the default
recommendation.

---

## 6. E — Frequency branch (cheap global/cyclic context)

Charts/Vo trend have cyclic structure (sessions, intraday rhythms, vol regimes).
A tiny **frequency branch** gives the model an explicit global view:

- Compute **DCT over the past window** of a few informative series that are
  already available per window (close, high-low range, volume) — either on the
  numeric side or on the price-panel column means.
- Keep the **top-k low-frequency coefficients** (k ≈ 16–32) per series and
  project them to a small embedding.
- Feed them either as **1–3 extra tokens** in the fusion stream (type marker
  `IS_FREQ`) or as a context vector.
- **No look-ahead**: the window is entirely the past; causal by construction.
- Cost ≈ negligible (< 0.05M); effect target is regime/vol context, so the probe
  should see **regime-silhouette** rise. If it does not, drop E.

---

## 7. Integration surface (what changes in the model)

- `VisionEncoder` gains a `mode` (`cnn` | `columnformer`) and its own config
  (`d`, `L`, `H`, `dim_ff`, positional, reader, causality, frequency flag).
- `PolicyConfig` exposes the new knobs; `_policy_kwargs_from` passes them through.
- `CrossModalFusion` gains a **token** variant (C) alongside the legacy vector
  variant; `PolicyNetwork.forward/encode` picks it by config.
- **Dataset / render / charts**: untouched — image 128 identical, so compiled
  stores remain valid and the render contract is unchanged.
- `SampleSpec` untouched. Heads/memory unchanged (operate on the fused vector).
- **Contract note**: `model_config` footprint changes → S1↔S2 loading must match
  (`load_state_dict` strict=False tolerates shape diffs; S2 rebuilds from
  `model_config`). Decide v1-vs-v2 **before Phase B / S2**.

## 8. Cost & scale snapshot (heavy-v2 lines)

| Stream | v1 heavy | v2 (A+B) | v2 + C | notes |
|---|---:|---:|---:|---|
| vision | 1.40M | ~7–9M | ~7–9M | ColumnFormer |
| numeric | 7.30M | 7.30M | 7.30M | unchanged |
| fusion | 7.69M | 7.69M | ~14M (token variant) | C doubles fusion |
| memory | 7.15M | 7.15M | 7.15M | unchanged |
| context/E | 0.05M | 0.05M | +0.05M | freq branch |
| **total** | **~23.6M** | **~29–31M** | **~36–38M** | |
| VRAM | <7GB | ≈7GB | ≈9–10GB | L4 23GB ok |
| disk/charts | unchanged | unchanged | unchanged | |

Throughput: columnformer attention over 128 tokens is cheaper per step than the
deep CNN activations at bs 64; step-time likely **≤** heavy-v1.

---

## 9. Validation plan (the gate — same probe, harder targets)

Reuse `scripts/probe_s1_checkpoint.py` + S2 stage-readiness as the contract:

| Metric | v1 heavy (epoch0 baseline) | v2 success gate |
|---|---|---|
| vision↔numeric alignment cos | ≈ −0.003 | **≥ 0.10** (ideal ≳0.2) |
| instrument silhouette | 0.14 | **≥ 0.25** |
| regime silhouette | −0.007 | **≥ 0.05** (ideally positive clear) |
| temporal-CPC separation | 0.846 | **≥ 0.80** (do not regress) |
| direction probe (vs majority) | up 0.39 / flat 0.58 | lift kept or better; down ≥ 0.4 |
| S2 stage-readiness after warm-start | TBD | ≥ v1 heavy at same epochs |

Per-feature ablations to run (same epochs, same data):
- **A only** vs CNN (B off, C off) → isolates ColumnFormer gain.
- **A+B** → reader effect. **A+D** → causality effect. **A+E** → frequency effect.
- **A+C** → token fusion effect (expect the biggest alignment-cos jump).
- NaN/finite, train↔val gap, steps/epoch, VRAM — always measured.

Keep a change **only** if the gated metric improves; otherwise ship less.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Overfit 30M+ on 418k (Phase A) | keep EMA/weight-decay/mask; rely on Phase B (1.68M); gate by val, not train |
| Token-sequence blowup (256-token variant) | default 128 tokens; 256 only if probe proves win |
| Coordinated A–E changes break v1 behaviour | config flags default to current CNN; A/B/C/D/E opt-in |
| S1→S2 contract drift | freeze model_config decision before Phase B; strict-check on load |
| Claimed alignment rises due to locality not semantics | also check instrument/regime silhouette + direction lift (not only cos) |
| Causality changes temporal-CPC probe | locality inflation is already visible; compare with/without causal mask |
| Frequency branch noise | drop E if regime-silhouette does not improve |

---

## 11. Phased roadmap

1. **Phase 1 — A+B** (columnformer + attention reader): core structural fix;
   highest value; moderate work.
2. **Phase 2 — C** (token fusion): the alignment jump; depends on 1.
3. **Phase 3 — D/E** (optional): causality + frequency; cheap, only if probes leave
   a gap.
4. Integrate into `policy.py` + `_policy_kwargs_from` + configs; keep flags.
5. Re-run S1 (short smoke + full) → probe → S2 stage-readiness → keep/vet.

---

## 12. Open questions

- Is 1-token-per-bar enough, or does the 2-token-per-bar (256) variant win?
- Does causal masking hurt the CPC probe (locality)?
- Is `d=384` still the right profile at 30M+, or should embed climb to 512 for C?
- Do we keep the CNN as a *cheap auxiliary* stream alongside ColumnFormer for
  input-robustness, or replace outright?
- Should E ingest numeric-DCT or vision-column-DCT (or both)?

*Decision pending: user sign-off, then prototype behind config flags, gate by the
S1 probe.*