"""Smoke tests for the S3 curriculum script and config loading."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from zhisa.config import load_config
from zhisa.scripts.train_s3 import _build_stages, _make_inner_factory


def test_s3_config_loads():
    cfg = load_config(Path("configs/s3_curriculum.yaml"))
    assert cfg is not None
    assert cfg["inner"] == "s1"
    assert len(cfg["stages"]) == 3
    assert [s["name"] for s in cfg["stages"]] == ["clean", "mixed", "stressed"]


def test_build_stages_from_config():
    from zhisa.utils.containers import Config

    cfg = Config({
        "stages": [
            {"name": "a", "n_bars": 100, "base_vol": 0.3, "shock_prob": 0.0,
             "student_t_df": 20.0, "epochs": 2, "mix_with_previous": 0.0},
            {"name": "b", "n_bars": 200, "base_vol": 0.6, "shock_prob": 0.001,
             "student_t_df": 8.0, "epochs": 1, "mix_with_previous": 0.2},
        ]
    })
    stages = _build_stages(cfg)
    assert len(stages) == 2
    assert stages[0].name == "a"
    assert stages[0].n_bars == 100
    assert stages[0].epochs == 2
    assert stages[1].mix_with_previous == 0.2


def test_build_stages_with_empty_config():
    from zhisa.utils.containers import Config

    assert _build_stages(Config({})) == []


def test_make_inner_factory_s1():
    from zhisa.utils.containers import Config
    from zhisa.training.s1_ssl import SSLPretrainer

    factory = _make_inner_factory(Config({"inner_batch_size": 16, "inner_lr": 1e-3, "inner_log_every": 100}), "s1")
    assert callable(factory)


def test_make_inner_factory_s2():
    from zhisa.utils.containers import Config
    from zhisa.training.s2_supervised import SupervisedTrainer

    factory = _make_inner_factory(Config({"inner_batch_size": 16, "inner_lr": 1e-3, "inner_log_every": 100}), "s2")
    assert callable(factory)


def test_make_inner_factory_unknown_kind_exits():
    from zhisa.utils.containers import Config

    with pytest.raises(SystemExit):
        _make_inner_factory(Config({}), "s5")


def test_s3_script_runs_smoke_s1(tmp_path):
    """Minimal S3 S1-inner run should complete and write a checkpoint."""
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_s3",
        "--config", "configs/s3_curriculum.yaml",
        "--inner", "s1",
        "--checkpoint", str(out_dir / "s3_smoke.pt"),
    ]
    # Patch the YAML stage sizes for speed by writing a tiny config in tmp.
    tiny_cfg = tmp_path / "tiny_s3.yaml"
    tiny_cfg.write_text(
        "seed: 0\n"
        "device: cpu\n"
        "chart_window: 16\n"
        "image_size: 16\n"
        "inner: s1\n"
        "stages:\n"
        "  - {name: a, n_bars: 150, base_vol: 0.4, shock_prob: 0.0, student_t_df: 20.0, epochs: 1, mix_with_previous: 0.0}\n"
        "  - {name: b, n_bars: 150, base_vol: 0.6, shock_prob: 0.0005, student_t_df: 8.0, epochs: 1, mix_with_previous: 0.0}\n"
        "inner_batch_size: 64\n"
        "inner_lr: 0.001\n"
        "inner_log_every: 1000\n"
    )
    cmd[cmd.index("configs/s3_curriculum.yaml")] = str(tiny_cfg)

    import os
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S3 training complete" in result.stdout
    assert (out_dir / "s3_smoke.pt").exists()
