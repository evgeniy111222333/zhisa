# S1 Heavy Model — transition rationale, goals, risks

Date: 2026-08-22 · Status: choice A (recommended-heavy) accepted · Model target: **~24M params**

## 1. Why a heavy model (and why now)

The original S1 shipped as a deliberately tiny stack (~1.39M params) so the MVP
pipeline could run end-to-end quickly. That is a *smoke* size, not a *research*
size. For a 12-market, multimodal, self-supervised stage that will feed S2/S2b/S4,
the encoder is the foundation: every downstream stage builds on its embeddings.
Small capacity here caps the whole pipeline.

The heavy choice is justified by:

- **Data volume is sufficient.** Self-supervision on phase A has ~418k samples
  (1h, 12 symbols) and phase B ~1.68M (15m). 24M params on 1.68M samples is a
  reasonable capacity/data balance (≈14 params/sample) for representation
  learning, especially with EMA-teacher self-distillation acting as an implicit
  regularizer.
- **Multimodality + multi-instrument.** Two independent encoders (vision +
  numeric), cross-modal fusion and a working-memory bank each need their own
  headroom; a single shared 128-d space was the binding constraint.
- **Downstream headroom.** S2/S4 heads are added later; a richer base embedding
  gives them a better starting point and less pressure to re-learn features.
- **Cost is bounded.** Weights = 94.6 MB fp32; estimated VRAM for
  weights+gradients+Adam ≈ 285 MB. Even at batch 64 with temporal pairs the L4
  (23 GB) is far from saturated. Disk for charts is unchanged (image stays 128).

## 2. Decision A — the accepted spec

| Parameter | Tiny (old) | **Heavy (A)** |
|---|---|---|
| embed_dim / d_model | 128 | **384** |
| vision channels | (32,64,128,192) | **(64,128,256,384)** |
| numeric layers | 2 | **4** |
| fusion layers | 2 | **4** |
| memory layers / max_len | 2 / 64 | **4 / 128** |
| encoder feed-forward mult | 2.0 (ff≈256) | **4.0 (ff≈1536)** |
| n_instruments | 1 | **12** |
| batch size (phase A) | 128 | **64** |
| mask_ratio | 0.4 | **0.5** |
| **Total params** | 1.39M | **~23.6M** |

Breakdown (measured): fusion 7.69M · numeric 7.30M · memory 7.15M · vision 1.40M ·
context 0.05M · heads 0.05M.

## 3. What the stage must achieve (goals)

S1 Phase A is **self-supervised representation pretraining** — not a trading
policy, so "accuracy" is not the metric.

Success criteria at the end of Phase A:

1. **Monotonic val-total decrease** epoch over epoch
   (`total = temporal + masked + alignment`, weighted).
2. **temporal** (CPC/InfoNCE of next 4-bar state) clearly decreasing — the encoder
   actually predicts structure.
3. **masked** (numeric reconstruction at 50% mask) decreasing steadily.
4. **alignment** small and stable (chart embedding ≈ numeric embedding).
5. **No NaN/Inf** in losses or gradients; EMA teacher stays stable; train/val gap
   does not explode (overfit guard).
6. **Checkpoint completeness**: `phase1_heavy_best.pt` with `checkpoint_meta`
   containing dataset manifest checksum + render contract (reproducible) and
   `resume_granularity` for later continuation.

The hard evaluation of the *market* capability happens later (S2 stage-readiness,
baseline 0.633 → targets 0.68+/0.72).

## 4. Risks and their mitigations

| Risk | Mitigation |
|---|---|
| Overfitting 24M on 418k (phase A) | EMA decay 0.996, weight_decay 1e-4, mask 0.5, grad_clip 1.0, val monitoring each epoch; heavy model truly pays off at Phase B (1.68M) |
| Hyperparameter sensitivity when scaling up | Warmup 500, cosine-free constant LR 3e-4; checkpoint every 500 steps for resume |
| VRAM spikes on L4 | Batch 64 (halved vs tiny) + compiled charts (no in-loop render) |
| Instrument identity loss on 12 markets | n_instruments=12 + per-dataset instrument_id threaded into `ContextEncoder.instrument_emb` (implemented + tested) |
| Broken comparison vs previous runs | Render contract & fingerprints unchanged (image 128 kept); this is intentionally a *new* baseline, so comparisons restart from here |
| Long run interruptions | `checkpoint_every_steps` + per-epoch save + atomic writes; resume with `--resume-from phase1_heavy_last.pt` |

## 5. Other problems discovered & fixed during the transition

While wiring the heavy configuration we found and fixed several latent issues:

1. **No instrument identity at all** — S1 treated BTC and TRX as the same input.
   Fixed: `n_instruments=12`, `MarketDataset.instrument_id`, threaded through
   `multimodal_collate`, `temporal_pair_collate` and `SSLPretrainer._loss`, id
   passed to `model.encode(..., instrument_id=...)` for both the student and the
   EMA teacher.
2. **Numeric encoder depth was frozen** (`n_layers=2` hardcoded) — added
   `PolicyConfig.numeric_layers`.
3. **Transformer feed-forward was fixed at 256** regardless of embed_dim —
   added `PolicyConfig.encoder_ff_mult` (default 2.0 preserves legacy exactly).
4. **Model config was not configurable from YAML** — `train_s1` now merges an
   optional `model:` block into `build_default_policy` (window/image/n_actions
   come from the spec; `vision_channels` normalised to a tuple; guarded against
   duplicate `in_numeric_features`/`in_context_features`).
5. **Crash-safe workflow** — leftover `.job_tmp_*` partial chart artefacts from an
   interrupted run are cleaned before restart; compile writes are atomic
   (tmp-dir → `os.replace`).
6. **Config for S2 to follow** — S2 will receive the same `instruments=` list so
   the id space stays consistent with the S1 checkpoint it warms from (patch in
   flight, see §6).

## 6. Configuration & launch

Phase A (1h, heavy):

```bash
python -m zhisa.scripts.train_s1 \
  --config configs/s1_ssl_1h_12m_heavy.yaml \
  --prepared-root /data/datasets/s1_1h_12m_v2 \
  --charts-cache-dir /data/charts \
  --render-engine gpu --render-workers 0 --render-chunk 256 \
  --checkpoint /data/out/phase1_heavy_last.pt \
  --best-checkpoint /data/out/phase1_heavy_best.pt
```

Phase B (15m, heavy) resumes `phase1_heavy_best.pt`:

```bash
python -m zhisa.scripts.train_s1 \
  --config configs/s1_ssl_15m_12m_heavy.yaml \
  --prepared-root /data/datasets/s1_15m_12m_v2 \
  --charts-cache-dir /data/charts \
  --render-engine gpu --render-workers 0 --render-chunk 256 \
  --resume-from /data/out/phase1_heavy_best.pt --reset-best-on-resume \
  --checkpoint /data/out/phase2_heavy_last.pt \
  --best-checkpoint /data/out/phase2_heavy_best.pt
```

Disks: compiled charts for image 128 are unchanged by the heavy model (the render
identity only depends on RenderSpec, not architecture). No recompilation is
needed for switching model size.

## 7. Follow-ups (not blocking Phase A)

- Hook `instruments=` into S2 (and S2b/S4 when they build `MarketDataset` from
  prepared data) so the 12-market id space stays aligned.
- Measure wall-time per step on L4 for the heavy model; if GPU-bound, enable
  optional AMP (bf16) later (kept off for now for strict determinism).
- Decide a Smooth/stepped LR for Phase B if early epochs of Phase B are unstable.

## 8. Observed parity note (L4 / torch 2.13)

On the training instance (NVIDIA L4, torch 2.13.0+cu130) the GPU parity gate
reported `ok=False maxdiff=5.69e-02 n_diff=6` on the golden corpus (6 of ~25M
pixels differ by one palette-colour flip — an fp64 boundary-case that differs
across torch versions/GPUs). This is exactly what the gate is for: the engine
**fell back to CPU canonical**, which is the safest outcome — the compiled store
then matches a CPU-only re-render anywhere. The GPU engine stays available and
gated; known environment-specific, not a blocker.