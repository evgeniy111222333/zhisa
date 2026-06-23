"""Smoke tests for the S1 training script and SSL config loading."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zhisa.config import load_config
from zhisa.data.dataset import SampleSpec
from zhisa.scripts.train_s1 import _market_datasets_from_frame, _ssl_config_from


def test_s1_config_loads():
    """The shipped S1 config must load without errors and contain the expected keys."""
    cfg = load_config(Path("configs/s1_ssl.yaml"))
    assert cfg is not None
    assert "ssl" in cfg
    s = cfg["ssl"]
    assert s["projection_dim"] == 64
    assert s["temperature"] == 0.1
    assert s["use_ema_teacher"] is True


def test_ssl_config_factory_handles_missing_block():
    """If the config has no 'ssl' block, the factory should use defaults."""
    from zhisa.utils.containers import Config

    cfg = Config({"seed": 0})
    ssl = _ssl_config_from(cfg)
    assert ssl.projection_dim == 64
    assert ssl.use_ema_teacher is True
    assert ssl.epochs == 1  # SSLConfig default


def test_ssl_config_factory_handles_none():
    """The factory must work even if called with ``None`` config."""
    ssl = _ssl_config_from(None)
    assert ssl.projection_dim == 64
    assert ssl.temperature == 0.1


def test_prepared_loader_splits_timestamp_gaps():
    left = pd.date_range("2024-01-01", periods=220, freq="15min", tz="UTC")
    right = pd.date_range(left[-1] + pd.Timedelta(hours=3), periods=220, freq="15min", tz="UTC")
    index = left.append(right)
    close = np.linspace(100.0, 110.0, len(index))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(len(index)),
            "symbol": "BTC/USDT",
        },
        index=index,
    )
    datasets = _market_datasets_from_frame(
        frame,
        spec=SampleSpec(chart_window=16, feature_window=16, horizons=(4, 8)),
        cache_charts=False,
        chart_cache_size=-1,
        timeframe="15m",
    )
    assert len(datasets) == 2
    assert all(ds.df.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all() for ds in datasets)


def test_s1_script_runs_smoke(tmp_path):
    """A minimal S1 training run should complete end-to-end and write a checkpoint."""
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_s1",
        "--bars", "200",
        "--epochs", "1",
        "--batch-size", "64",
        "--checkpoint", str(out_dir / "s1_smoke.pt"),
    ]
    import os
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S1 training complete" in result.stdout
    assert (out_dir / "s1_smoke.pt").exists()


def test_aws_launcher_uses_absolute_epoch_targets():
    script = Path("scripts/aws_train_s1_12m.sh").read_text(encoding="utf-8")
    assert '--epochs "$PHASE1_TARGET"' in script
    assert '--epochs "$TOTAL_TARGET"' in script
    assert 'final_completed=$(checkpoint_epochs "$RUN_DIR/phase2_last.pt")' in script
