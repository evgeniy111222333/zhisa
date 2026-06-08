"""Smoke tests for the S4 PPO script and config loading."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from zhisa.config import load_config
from zhisa.env.trading_env import EnvConfig
from zhisa.scripts.train_s4 import _build_env_cfg, _build_ppo_cfg


def test_s4_config_loads():
    cfg = load_config(Path("configs/s4_rl.yaml"))
    assert cfg is not None
    assert cfg["n_episodes"] == 4
    assert cfg["gamma"] == 0.99
    assert cfg["gae_lambda"] == 0.95
    assert cfg["env_cfg"]["initial_cash"] == 10000.0
    assert cfg["optim"]["lr"] == 0.0003


def test_build_env_cfg_overrides_apply():
    from zhisa.utils.containers import Config

    cfg = Config({"env_cfg": {"initial_cash": 999.0, "fee_bps": 7.5}})
    env_cfg = _build_env_cfg(cfg)
    assert isinstance(env_cfg, EnvConfig)
    # initial_cash is not a real EnvConfig field — it must be filtered
    # out. Only fee_bps should be applied.
    assert env_cfg.fee_bps == 7.5


def test_build_env_cfg_ignores_unknown_keys():
    from zhisa.utils.containers import Config
    from zhisa.env.trading_env import EnvConfig

    cfg = Config({"env_cfg": {"this_does_not_exist": 42}})
    env_cfg = _build_env_cfg(cfg)
    assert isinstance(env_cfg, EnvConfig)


def test_build_env_cfg_empty_config():
    from zhisa.utils.containers import Config
    from zhisa.env.trading_env import EnvConfig

    env_cfg = _build_env_cfg(Config({}))
    assert isinstance(env_cfg, EnvConfig)


def test_build_ppo_cfg_defaults():
    from zhisa.utils.containers import Config

    cfg = Config({})
    ppo_cfg = _build_ppo_cfg(cfg, _Args(), EnvConfig())
    assert ppo_cfg.n_episodes == 4
    assert ppo_cfg.gamma == 0.99
    assert ppo_cfg.clip_ratio == 0.2


def test_build_ppo_cfg_cli_overrides():
    from zhisa.utils.containers import Config

    cfg = Config({"n_episodes": 7, "gamma": 0.5, "clip_ratio": 0.1})
    args = _Args(n_episodes=2, max_steps=99)
    ppo_cfg = _build_ppo_cfg(cfg, args, EnvConfig())
    assert ppo_cfg.n_episodes == 2          # CLI wins over YAML
    assert ppo_cfg.gamma == 0.5
    assert ppo_cfg.clip_ratio == 0.1
    assert ppo_cfg.max_steps_per_episode == 99


def test_s4_script_runs_smoke(tmp_path):
    """A tiny PPO run should complete and write a checkpoint."""
    out_dir = tmp_path / "artifacts"
    tiny_cfg = tmp_path / "tiny_s4.yaml"
    tiny_cfg.write_text(
        "seed: 0\n"
        "device: cpu\n"
        "chart_window: 16\n"
        "image_size: 16\n"
        "n_bars: 300\n"
        "n_episodes: 2\n"
        "max_steps_per_episode: 16\n"
        "n_epochs: 1\n"
        "minibatch_size: 4\n"
        "env_cfg:\n"
        "  window: 16\n"
        "  image_size: 16\n"
        "  episode_length: 16\n"
        "optim:\n"
        "  lr: 0.001\n"
    )
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_s4",
        "--config", str(tiny_cfg),
        "--checkpoint", str(out_dir / "s4_smoke.pt"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S4 PPO training complete" in result.stdout
    assert "checkpoint saved to" in result.stdout
    assert (out_dir / "s4_smoke.pt").exists()


class _Args:
    """Lightweight stand-in for argparse.Namespace used by build helpers."""
    def __init__(self, n_episodes=None, max_steps=None):
        self.n_episodes = n_episodes
        self.max_steps = max_steps
