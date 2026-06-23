"""Demo: measure per-iteration cost for the S4-CVaR run the user is about to launch.

This does NOT train. It runs a single PPO iteration at full scale (50 episodes
* 200 steps = 10000 env steps) and breaks down the cost into:

  1. env init / chart prefetch
  2. rollout collection (env.step + policy forward)
  3. PPO update (advantage computation + mini-batch updates)
  4. log/checkpoint IO

Then it extrapolates the per-iter mean to the requested total of 60 iterations
and prints the bottleneck so the user can decide whether to scale up.

Run with:
    python demo_cvar_timing.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.training.cvar_ppo import CVaRPPOConfig, CVaRPPOTrainer
from zhisa.training.s4_rl import Transition, compute_gae, ppo_loss
from zhisa.utils.timing import Timer

# ---------------------------------------------------------------------------
# Config — mirror the user's real command
# ---------------------------------------------------------------------------

USER_ITERATIONS = 60
USER_EPISODES = 50
USER_MAX_STEPS = 200
DEVICE = "cpu"  # set to "cuda" to time GPU
N_PROBE_ITERS = 1  # only need one full iteration to extrapolate


def _build_env_and_model() -> tuple[TradingEnv, torch.nn.Module, dict]:
    """Build a TradingEnv + PolicyNetwork that match the user's invocation.

    The user's command does:
        --data-source tsdb --tsdb-root data/tsdb --symbol BTC/USDT
        --timeframe 5m --load artifacts/s4/model_btc_rl.pt
        --ent-coef 0.05

    We replicate the same load path so the model has the same parameter count.
    """
    # Try to load the user's existing checkpoint; fall back to a fresh model
    # with the production config if the file is missing.
    from zhisa.scripts._real_data import load_market_dataframe

    args = type("A", (), {
        "data_source": "tsdb", "tsdb_root": "data/tsdb",
        "symbol": "BTC/USDT", "timeframe": "5m",
        "load": "artifacts/s4/model_btc_rl.pt",
    })()

    print(f"[init] Loading market data from {args.tsdb_root}/{args.symbol}/{args.timeframe}...")
    t0 = time.time()
    df = load_market_dataframe(args, seed=0, default_bars=20000)
    print(f"[init] DataFrame: {len(df)} bars in {time.time() - t0:.1f}s")

    # Use the same window/image_size as train_s4_cvar.py defaults (chart_window=16, image_size=32)
    chart_window, image_size = 16, 32
    env_cfg = EnvConfig(
        episode_length=USER_MAX_STEPS,
        window=chart_window,
        image_size=image_size,
        prefetch_charts=True,  # Tier 2.1 default for this script
    )

    print(f"[init] Building TradingEnv with prefetch_charts=True...")
    t0 = time.time()
    env = TradingEnv(df, cfg=env_cfg)
    n_feat, n_ctx = env.obs_numeric_dim, env.obs_context_dim
    print(f"[init]   numeric_dim={n_feat}, context_dim={n_ctx}, prefetch in {time.time() - t0:.1f}s")

    print(f"[init] Building PolicyNetwork...")
    t0 = time.time()
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=4,
    )
    model.to(DEVICE)
    print(f"[init]   params={sum(p.numel() for p in model.parameters()):,} in {time.time() - t0:.1f}s")

    # Try loading the production weights; failure is non-fatal for the demo.
    weights_loaded = False
    load_path = Path("artifacts/s4/model_btc_rl.pt")
    if load_path.exists():
        try:
            ckpt = torch.load(load_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            weights_loaded = True
            print(f"[init] Loaded weights from {load_path}")
        except Exception as e:
            print(f"[init] (warning) could not load {load_path}: {e}")
    if not weights_loaded:
        print(f"[init] (note) using freshly-initialised weights — per-step cost will be representative")
    model.eval()  # disable dropout for fair timing

    info = {"n_bars": len(df), "n_feat": n_feat, "n_ctx": n_ctx,
            "weights_loaded": weights_loaded}
    return env, model, info


def main() -> None:
    print("=" * 70)
    print("S4-CVaR TIMING DEMO (no training, single iteration at full scale)")
    print("=" * 70)
    print(f"  device: {DEVICE}")
    print(f"  target: {USER_ITERATIONS} iter * {USER_EPISODES} ep * {USER_MAX_STEPS} step")
    print()

    env, model, info = _build_env_and_model()
    cfg = CVaRPPOConfig(
        n_iterations=1,
        n_episodes=USER_EPISODES,
        max_steps_per_episode=USER_MAX_STEPS,
        cvar_alpha=0.1, cvar_threshold=0.1,
        cvar_lambda_init=0.0, cvar_lambda_lr=0.05,
        cvar_warmup_iters=5,  # Tier 1.4 default
        n_epochs=4, minibatch_size=256,  # Tier 1.3 default
        device=DEVICE, seed=0, log_every=1,
        env_cfg=env.cfg,
    )

    trainer = CVaRPPOTrainer(model, cfg)

    # ---- Phase 1: rollout collection ----
    print()
    print("[1/4] Rollout collection (50 episodes, up to 200 steps each)...")
    t0 = time.time()
    buf, rollout_stats = trainer._collect_rollout(env)
    rollout_time = time.time() - t0
    print(f"        done: {len(buf)} steps, {rollout_stats['ep_returns']!r}")
    print(f"        elapsed: {rollout_time:.2f}s")
    print(f"        per-step mean: {rollout_time / max(1, len(buf)) * 1000:.2f}ms")

    # ---- Phase 2: stack + GAE ----
    print()
    print("[2/4] stack_tensors + compute_gae (numpy)...")
    t0 = time.time()
    stacked = buf.stack_tensors()
    advantages, returns = compute_gae(
        stacked["reward"], stacked["value"], stacked["done"],
        last_value=0.0, gamma=cfg.gamma, lam=cfg.gae_lambda,
    )
    stack_time = time.time() - t0
    print(f"        elapsed: {stack_time:.3f}s")

    # ---- Phase 3: PPO update (4 epochs of mini-batches) ----
    print()
    print(f"[3/4] PPO update (4 epochs, mb=256) over {len(buf)} steps...")
    n_minibatches = max(1, len(buf) // cfg.minibatch_size) * cfg.n_epochs
    t0 = time.time()
    losses = trainer._cvar_ppo_update(buf, torch.from_numpy(np.array(rollout_stats["ep_returns"])))
    ppo_time = time.time() - t0
    print(f"        elapsed: {ppo_time:.2f}s ({ppo_time / n_minibatches * 1000:.1f}ms / mini-batch)")
    print(f"        losses: {losses}")

    # ---- Phase 4: lambda update + log ----
    print()
    print("[4/4] lambda update + checkpoint IO...")
    t0 = time.time()
    ep_returns_np = np.array(rollout_stats["ep_returns"], dtype=np.float32)
    from zhisa.risk.cvar import cvar_numpy
    cvar_value = cvar_numpy(ep_returns_np, cfg.cvar_alpha)
    lambda_time = time.time() - t0
    print(f"        elapsed: {lambda_time * 1000:.2f}ms")
    print(f"        cvar={cvar_value:.4f}")

    # ---- Summary ----
    iter_time = rollout_time + stack_time + ppo_time + lambda_time
    total_time = iter_time * USER_ITERATIONS

    print()
    print("=" * 70)
    print("PER-ITERATION BREAKDOWN")
    print("=" * 70)
    rows = [
        ("rollout (env.step + policy forward)", rollout_time, rollout_time / iter_time * 100),
        ("stack + GAE", stack_time, stack_time / iter_time * 100),
        ("PPO update (forward + backward)", ppo_time, ppo_time / iter_time * 100),
        ("lambda update + log", lambda_time, lambda_time / iter_time * 100),
    ]
    print(f"  {'phase':<40s} {'time':>8s} {'%':>5s}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 5}")
    for name, t, pct in rows:
        print(f"  {name:<40s} {t:7.2f}s {pct:4.1f}%")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 5}")
    print(f"  {'TOTAL per iter':<40s} {iter_time:7.2f}s  100%")

    print()
    print("=" * 70)
    print("EXTRAPOLATION")
    print("=" * 70)
    print(f"  per-iter mean:     {iter_time:.2f}s")
    print(f"  * 60 iterations:   {total_time:.1f}s = {total_time / 60:.1f}min")
    print(f"  per-step mean:     {rollout_time / max(1, len(buf)) * 1000:.2f}ms")

    # Bottleneck call-out
    bottleneck = max(rows, key=lambda r: r[1])
    print()
    print(f"  >>> BOTTLENECK: {bottleneck[0]} ({bottleneck[1]:.1f}s = {bottleneck[2]:.0f}% of iter) <<<")

    # GPU estimate (typical RTX 3060/4070 is ~5-10x faster than this CPU for the
    # rollout and ~10-30x for the PPO update; the chart-prefetch is similar
    # speed since it is pure numpy. We use conservative factors.)
    print()
    print("=" * 70)
    print("GPU ESTIMATE (rough)")
    print("=" * 70)
    gpu_rollout = rollout_time * 0.20
    gpu_ppo = ppo_time * 0.10
    gpu_other = stack_time + lambda_time
    gpu_iter = gpu_rollout + gpu_ppo + gpu_other
    print(f"  rollout:     {rollout_time:.2f}s CPU  ->  ~{gpu_rollout:.2f}s GPU")
    print(f"  PPO update:  {ppo_time:.2f}s CPU  ->  ~{gpu_ppo:.2f}s GPU")
    print(f"  other:       {gpu_other:.3f}s (numpy, no GPU win)")
    print(f"  per iter:    ~{gpu_iter:.2f}s")
    print(f"  * 60 iter:   ~{gpu_iter * 60:.0f}s = {gpu_iter * 60 / 60:.1f}min")
    print()
    print("(GPU factor is heuristic — real speedup depends on batch sizes,")
    print(" CPU<->GPU transfer overhead, and whether model is on GPU. Treat")
    print(" the GPU estimate as a rough order-of-magnitude.")


if __name__ == "__main__":
    main()
