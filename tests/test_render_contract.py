"""Tests for the render contract (train/inference byte-equivalence invariant)."""
from __future__ import annotations

import pytest

from zhisa.data.chart_store import CompiledChartStore
from zhisa.data.render_contract import (
    RenderContract,
    RenderContractError,
    assert_render_contract,
    compatible,
)
from zhisa.rendering.goldens import golden_fixture
from zhisa.rendering.spec import RenderSpec

import pandas as pd


def _frame(n_bars: int = 256) -> pd.DataFrame:
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


def _store(spec: RenderSpec = RenderSpec(size=32), window: int = 32, out=None):
    df = _frame()
    return CompiledChartStore.build(df, window=window, spec=spec, indices=range(80), out_root=out)


def test_contract_from_spec_and_store_agree():
    spec = RenderSpec(size=32)
    store = _store(spec)
    c_from_store = RenderContract.from_store(store)
    c_from_spec = RenderContract.from_spec(spec, store_checksum=store.render_checksum())
    assert c_from_store.renderer_version == c_from_spec.renderer_version
    assert c_from_store.render_spec_hash == c_from_spec.render_spec_hash
    assert c_from_store.render_fingerprint == c_from_spec.render_fingerprint


def test_assert_true_when_matching(tmp_path):
    spec = RenderSpec(size=32)
    store = _store(spec, out=tmp_path)
    c = RenderContract.from_spec(spec, store_checksum=store.render_checksum())
    assert_render_contract(c, store)          # identity check
    assert_render_contract(c, store, require_store_checksum=True)  # byte check
    assert compatible(c, store)


def test_mismatched_spec_fails():
    store = _store(RenderSpec(size=32))
    wrong = RenderContract.from_spec(RenderSpec(size=64))
    with pytest.raises(RenderContractError):
        assert_render_contract(wrong, store)
    assert compatible(wrong, store) is False


def test_mismatched_renderer_version_fails():
    store = _store(RenderSpec(size=32))
    c = RenderContract.from_store(store)
    c2 = RenderContract(
        renderer_version="0.0.old",
        render_spec_hash=c.render_spec_hash,
        render_fingerprint=c.render_fingerprint,
        store_checksum=c.store_checksum,
    )
    with pytest.raises(RenderContractError):
        assert_render_contract(c2, store)


def test_checksum_mismatch_detected(tmp_path):
    store = _store(RenderSpec(size=32), out=tmp_path)
    c = RenderContract.from_spec(RenderSpec(size=32), store_checksum="0" * 64)
    with pytest.raises(RenderContractError):
        assert_render_contract(c, store, require_store_checksum=True)


def test_different_content_store_byte_check_rejects(tmp_path):
    # A store built from different frame but same spec must fail the byte check
    # while passing the (weak) identity check.
    spec = RenderSpec(size=32)
    store_a = _store(spec, out=tmp_path)
    other_df = _frame(512)
    store_b = CompiledChartStore.build(other_df, window=32, spec=spec, indices=range(80))
    c = RenderContract.from_store(store_a)
    assert_render_contract(c, store_b)  # identity ok
    with pytest.raises(RenderContractError):
        assert_render_contract(
            RenderContract.from_spec(spec, store_checksum=store_a.render_checksum()),
            store_b,
            require_store_checksum=True,
        )


def test_checkpoint_meta_roundtrip():
    store = _store(RenderSpec(size=32))
    c = RenderContract.from_store(store)
    d = c.to_dict()
    meta = {"render": d}
    parsed = RenderContract.from_checkpoint_meta(meta)
    assert parsed is not None
    assert parsed == c


def test_checkpoint_meta_absent_returns_none():
    assert RenderContract.from_checkpoint_meta({}) is None
    assert RenderContract.from_checkpoint_meta({"render": {}}) is None


def test_resolve_and_enforce_shared_helpers():
    from zhisa.data.render_contract import (
        enforce_parent_render_contract,
        resolve_render_contract,
    )

    # no store -> default spec contract at image_size
    c = resolve_render_contract([_FakeDS(None)], image_size=64)
    assert c.render_spec_hash == RenderSpec(size=64).content_hash()
    # matching parent -> no raise
    parent = {"checkpoint_meta": {"render": c.to_dict()}}
    enforce_parent_render_contract(c, parent, stage_label="S2")
    # mismatched parent -> raise (RenderContractError inherits RuntimeError)
    wrong = {"checkpoint_meta": {"render": RenderContract.from_spec(RenderSpec(size=32)).to_dict()}}
    with pytest.raises(RenderContractError):
        enforce_parent_render_contract(c, wrong, stage_label="S2")


def test_serving_guard(tmp_path):
    from zhisa.data.render_contract import assert_serving_render

    rec = RenderContract.from_spec(RenderSpec(size=48)).to_dict()
    ckpt = {"checkpoint_meta": {"render": rec}}
    # same image_size serves fine
    assert_serving_render(ckpt, image_size=48)
    # different image_size -> refused
    with pytest.raises(RenderContractError):
        assert_serving_render(ckpt, image_size=64)
    # legacy checkpoint without render metadata -> not enforced
    assert_serving_render({"checkpoint_meta": {}}, image_size=64)


def test_run_render_audit_ok_and_fail(tmp_path):
    from zhisa.data.render_contract import run_render_audit
    from zhisa.models.policy import build_default_policy
    from zhisa.data.synthetic import MarketConfig, generate_market
    from zhisa.data.dataset import MarketDataset, SampleSpec
    from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer
    from zhisa.utils.seeding import set_seed

    set_seed(3)
    df = generate_market(MarketConfig(n_bars=250, freq="5min", seed=3))
    ds = MarketDataset(
        df,
        spec=SampleSpec(chart_window=32, feature_window=32, image_size=32, horizons=(4, 16, 64)),
        cache_charts=False, compute_targets=False,
    )
    model = build_default_policy(
        in_numeric_features=int(ds._features_df.shape[1]),
        in_context_features=int(ds._time_features_df.shape[1]),
        window=32, image_size=32,
    )
    contract = RenderContract.from_spec(RenderSpec(size=32))
    cfg = SSLConfig(
        device="cpu", batch_size=8, checkpoint=None, best_checkpoint=None,
        renderer_version=contract.renderer_version,
        render_spec_hash=contract.render_spec_hash,
        render_fingerprint=contract.render_fingerprint,
        render_store_checksum="0" * 64,
    )
    tr = SSLPretrainer(model, cfg)
    out = tmp_path / "s1.pt"
    tr.save(str(out))
    res = run_render_audit(out, image_size=32)
    assert res["ok"] is True
    assert res["serving_image_size"] == 32
    with pytest.raises(RenderContractError):
        run_render_audit(out, image_size=64)


class _FakeDS:
    def __init__(self, source=None):
        self._chart_source = source