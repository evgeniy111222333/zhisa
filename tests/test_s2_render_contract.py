"""Tests for S2 render-contract chain helpers (resolve / enforce / record)."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from zhisa.data.chart_store import CompiledChartStore
from zhisa.data.dataset import MarketTargetConfig, SampleSpec
from zhisa.data.render_contract import RenderContract
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec
from zhisa.scripts.train_s2 import _enforce_parent_render_contract, _resolve_render_contract


def _frame(n_bars: int = 300) -> pd.DataFrame:
    fx = golden_fixture(n_bars)
    return pd.DataFrame(
        {
            "open": fx[:, 0],
            "high": fx[:, 1],
            "low": fx[:, 2],
            "close": fx[:, 3],
            "volume": fx[:, 4],
            "timestamp": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
        }
    ).set_index("timestamp")


def _store(df, spec: RenderSpec, window: int = 32, n: int = 120):
    return CompiledChartStore.build(df, window=window, spec=spec, indices=range(n))


def _spec(image_size: int = 64) -> SampleSpec:
    return SampleSpec(chart_window=32, feature_window=32, image_size=image_size, horizons=(4, 16, 64))


def _json_tiny_model(image_size: int = 64):
    from zhisa.models.policy import build_default_policy
    return build_default_policy(
        in_numeric_features=32, in_context_features=10,
        window=32, image_size=image_size,
    )


class _FakeDS:
    def __init__(self, source=None):
        self._chart_source = source


def test_resolve_without_store_uses_default_spec():
    contract = _resolve_render_contract([_FakeDS(None), _FakeDS(None)], image_size=128)
    assert contract == RenderContract.from_spec(RenderSpec(size=128))


def test_resolve_from_store_matches_store():
    df = _frame()
    spec = RenderSpec(size=64)
    store = _store(df, spec)
    contract = _resolve_render_contract([_FakeDS(store)], image_size=64)
    assert contract.render_spec_hash == spec.content_hash()
    assert contract.renderer_version  # non-empty


def test_resolve_rejects_inconsistent_stores():
    df = _frame()
    a = _store(df, RenderSpec(size=64))
    b = _store(df, RenderSpec(size=32))
    with pytest.raises(RuntimeError):
        _resolve_render_contract([_FakeDS(a), _FakeDS(b)], image_size=64)


def test_enforce_matching_parent_passes():
    pdf_contract = RenderContract.from_spec(RenderSpec(size=64)).to_dict()
    parent = {"checkpoint_meta": {"render": pdf_contract}}
    current = RenderContract.from_spec(RenderSpec(size=64))
    _enforce_parent_render_contract(current, parent, stage_label="S1")  # must not raise


def test_enforce_missing_parent_block_skips():
    parent = {"checkpoint_meta": {}}
    _enforce_parent_render_contract(
        RenderContract.from_spec(RenderSpec(size=64)), parent, stage_label="S1"
    )  # must not raise


def test_enforce_mismatched_parent_raises():
    parent = {"checkpoint_meta": {"render": RenderContract.from_spec(RenderSpec(size=128)).to_dict()}}
    current = RenderContract.from_spec(RenderSpec(size=64))
    with pytest.raises(RuntimeError, match="render identity"):
        _enforce_parent_render_contract(current, parent, stage_label="S1")


def test_enforce_detects_renderer_drift():
    base = RenderContract.from_spec(RenderSpec(size=64))
    parent = {
        "checkpoint_meta": {"render": RenderContract(
            renderer_version="0.0.old",
            render_spec_hash=base.render_spec_hash,
            render_fingerprint=base.render_fingerprint,
        ).to_dict()}
    }
    with pytest.raises(RuntimeError):
        _enforce_parent_render_contract(base, parent, stage_label="S1")


def test_s2_checkpoint_records_render_block(tmp_path):
    """SupervisedTrainer.save must embed checkpoint_meta['render']."""
    from zhisa.data.synthetic import MarketConfig, generate_market
    from zhisa.training.losses import LossWeights, MultiTaskLoss
    from zhisa.training.optim import OptimConfig
    from zhisa.training.s2_supervised import SupervisedTrainer, TrainConfig
    from zhisa.utils.seeding import set_seed

    set_seed(0)
    df = generate_market(MarketConfig(n_bars=120, freq="5min", seed=1))
    from zhisa.data.dataset import MarketDataset
    ds = MarketDataset(df, spec=_spec(32), compute_targets=True)
    model = _json_tiny_model(32)
    loss = MultiTaskLoss(LossWeights())
    contract = RenderContract.from_spec(RenderSpec(size=32)).to_dict()
    cfg = TrainConfig(epochs=1, batch_size=8, device="cpu",
                      optim=OptimConfig(scheduler="cosine", t_max=5),
                      render_contract=contract)
    trainer = SupervisedTrainer(model, loss, cfg)
    out = tmp_path / "s2.pt"
    trainer.save(str(out))
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["checkpoint_meta"]["stage"] == "s2_supervised"
    assert payload["checkpoint_meta"]["render"] == contract
    assert payload["train_config"]["render_contract"] == contract