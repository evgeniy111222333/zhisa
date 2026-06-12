import torch
import numpy as np
import pytest
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.training.s4_rl import PPOConfig, PPOTrainer

def test_working_memory_active_gradients():
    # 1) Generate tiny synthetic market data
    df = generate_market(MarketConfig(n_bars=100, seed=42))

    # 2) Create policy model with working memory enabled
    model = build_default_policy(
        in_numeric_features=32,
        in_context_features=10,
        use_memory=True,
        memory_max_len=16,
    )
    assert model.memory is not None

    # 3) Initialize trainer
    cfg = PPOConfig(
        n_episodes=1,
        max_steps_per_episode=10,
        n_epochs=1,
        minibatch_size=4,
        device="cpu",
        env_cfg=EnvConfig(episode_length=10),
    )
    trainer = PPOTrainer(model, cfg)

    # 4) Collect a single rollout
    env = TradingEnv(df, cfg=cfg.env_cfg)
    buf, stats = trainer._collect_rollout(env)

    # 5) Verify buffer has history stored with correct shapes
    assert len(buf) > 0
    stacked = buf.stack_tensors()
    assert "history" in stacked
    # max_len = 16 (so history size is 15), embed_dim = 128
    assert stacked["history"].shape == (len(buf), 15, 128)

    # 6) Zero gradients of the model
    trainer.opt.zero_grad()

    # 7) Run PPO update (which triggers backward passes)
    losses = trainer._ppo_update(buf)

    # 8) Verify gradients exist and are non-zero in the WorkingMemory module
    found_grad = False
    for name, param in model.memory.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            grad_sum = param.grad.abs().sum().item()
            assert grad_sum > 0.0, f"Parameter {name} has zero gradient!"
            found_grad = True
    assert found_grad, "No trainable parameters found in memory module!"
    print("Verification successful: WorkingMemory gradients are active and non-zero!")
