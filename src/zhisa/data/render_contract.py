"""Render contract: the invariant that training == inference pixel-for-pixel.

Recording render provenance in a checkpoint is not enough — downstream stages
(S2, S2b, S4, inference) must **refuse to run** unless the chart identity they
are about to consume matches the identity a checkpoint was trained on. This
module centralises that check so every stage enforces the same rule via
:func:`assert_render_contract`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from zhisa.data.chart_store import CompiledChartStore
from zhisa.rendering.chart_renderer import CANONICAL_RENDERER_VERSION, render_fingerprint
from zhisa.rendering.spec import RenderSpec


@dataclass(frozen=True)
class RenderContract:
    """A versioned identity of how charts are produced for a stage/model."""

    renderer_version: str
    render_spec_hash: str
    render_fingerprint: str
    store_checksum: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "renderer_version": self.renderer_version,
            "render_spec_hash": self.render_spec_hash,
            "render_fingerprint": self.render_fingerprint,
            "store_checksum": self.store_checksum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RenderContract":
        return cls(
            renderer_version=str(d["renderer_version"]),
            render_spec_hash=str(d["render_spec_hash"]),
            render_fingerprint=str(d["render_fingerprint"]),
            store_checksum=str(d.get("store_checksum")),
        )

    @classmethod
    def from_spec(cls, spec: RenderSpec, *, store_checksum: Optional[str] = None) -> "RenderContract":
        """Contract for a fresh render from a spec (e.g. a new training run)."""
        return cls(
            renderer_version=CANONICAL_RENDERER_VERSION,
            render_spec_hash=spec.content_hash(),
            render_fingerprint=render_fingerprint(spec),
            store_checksum=store_checksum,
        )

    @classmethod
    def from_store(cls, store: CompiledChartStore) -> "RenderContract":
        meta = store.render_meta
        return cls(
            renderer_version=str(meta["renderer"]),
            render_spec_hash=str(meta["spec_hash"]),
            render_fingerprint=str(meta["fingerprint"]),
            store_checksum=str(meta["content_key"]),
        )

    @classmethod
    def from_checkpoint_meta(cls, meta: dict) -> Optional["RenderContract"]:
        """Parse the ``checkpoint_meta["render"]`` block written by S1."""
        render = (meta or {}).get("render")
        if not render or not render.get("renderer_version"):
            return None
        return cls.from_dict(render)


class RenderContractError(RuntimeError):
    """A stage tried to consume charts that do not match its training identity."""


def assert_render_contract(
    expected: RenderContract,
    actual: CompiledChartStore,
    *,
    require_store_checksum: bool = False,
) -> None:
    """Assert a compiled store satisfies the expected render identity.

    Raises :class:`RenderContractError` on any mismatch so a mis-rendered run
    fails loudly instead of silently shifting the visual input distribution.
    """
    problems: list[str] = []
    meta = actual.render_meta
    if str(meta.get("renderer")) != expected.renderer_version:
        problems.append(
            f"renderer version {expected.renderer_version} != store {meta.get('renderer')}"
        )
    if str(meta.get("spec_hash")) != expected.render_spec_hash:
        problems.append(f"spec hash mismatch: expected {expected.render_spec_hash}, store {meta.get('spec_hash')}")
    if str(meta.get("fingerprint")) != expected.render_fingerprint:
        problems.append(f"fingerprint mismatch: expected {expected.render_fingerprint}, store {meta.get('fingerprint')}")
    if require_store_checksum and expected.store_checksum is not None:
        contract = render_and_compare_checksum(actual, expected.store_checksum)
        if contract is not None:
            problems.append(contract)
    if problems:
        raise RenderContractError("render contract violation: " + "; ".join(problems))


def render_and_compare_checksum(store: CompiledChartStore, expected_checksum: str) -> Optional[str]:
    """Byte-equivalence: recompute the store's contract checksum and compare."""
    got = store.render_checksum()
    if got != expected_checksum:
        return f"store checksum mismatch: expected {expected_checksum[:12]}…, got {got[:12]}…"
    return None


def compatible(expected: RenderContract, actual: CompiledChartStore) -> bool:
    try:
        assert_render_contract(expected, actual)
    except RenderContractError:
        return False
    return True


# ---------------------------------------------------------------------------
# Spec-level identity comparison (no store needed)
# ---------------------------------------------------------------------------
#
# Charts are rendered with a default RenderSpec everywhere unless a compiled
# store is used. For stages that never compile stores (e.g. S2 lazy path) the
# *implicit* contract is ``RenderContract.from_spec(RenderSpec(size=image))``.
# Comparing identities at the spec level is enough to detect divergence, while
# store-level checks additionally guarantee byte-equivalence.


def default_spec_contract(image_size: int, store_checksum: Optional[str] = None) -> RenderContract:
    """The implicit render identity for charts drawn at ``image_size``."""
    return RenderContract.from_spec(RenderSpec(size=int(image_size)), store_checksum=store_checksum)


def _as_contract(actual) -> RenderContract:
    if isinstance(actual, CompiledChartStore):
        return RenderContract.from_store(actual)
    if isinstance(actual, RenderSpec):
        return RenderContract.from_spec(actual)
    if isinstance(actual, RenderContract):
        return actual
    raise TypeError(f"cannot interpret render identity from {type(actual).__name__}")


def assert_contract_identity(
    expected: RenderContract,
    actual,
    *,
    require_checksum: bool = False,
) -> None:
    """Compare the *identity* fields of two render contracts (or a store/spec).

    Unlike :func:`assert_render_contract` this does not touch chart bytes; it
    fails fast when a downstream stage would consume charts whose visual
    identity differs from the checkpoint it is initing from, even before any
    compiled store exists.
    """
    act = _as_contract(actual)
    problems: list[str] = []
    if act.renderer_version != expected.renderer_version:
        problems.append(
            f"renderer version {expected.renderer_version} != actual {act.renderer_version}"
        )
    if act.render_spec_hash != expected.render_spec_hash:
        problems.append(f"render spec hash mismatch: expected {expected.render_spec_hash[:12]}, actual {act.render_spec_hash[:12]}")
    if act.render_fingerprint != expected.render_fingerprint:
        problems.append(f"render fingerprint mismatch: expected {expected.render_fingerprint[:12]}, actual {act.render_fingerprint[:12]}")
    if (
        require_checksum
        and expected.store_checksum is not None
        and act.store_checksum is not None
        and act.store_checksum != expected.store_checksum
    ):
        problems.append("store checksum mismatch")
    if problems:
        raise RenderContractError("render identity mismatch: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# Stage plumbing: resolve / enforce / audit
# ---------------------------------------------------------------------------


def resolve_render_contract(datasets: Sequence, image_size: int) -> RenderContract:
    """The render identity actually served to a trainer.

    If any dataset has an attached compiled chart store, the contract comes
    from the stores (and all stores must agree); otherwise the implicit
    default-spec contract at ``image_size`` applies.
    """
    sources = [
        ds._chart_source
        for ds in datasets
        if getattr(ds, "_chart_source", None) is not None
    ]
    if sources:
        fingerprints = {s.render_meta["fingerprint"] for s in sources}
        if len(fingerprints) != 1:
            raise RenderContractError(
                "compiled chart stores use inconsistent render identities across "
                f"segments: {sorted(fingerprints)[:4]}"
            )
        return RenderContract.from_store(sources[0])
    return default_spec_contract(int(image_size))


def enforce_parent_render_contract(
    contract: RenderContract,
    parent_payload: Optional[dict],
    *,
    stage_label: str,
) -> None:
    """Fail fast if a parent checkpoint's visual identity differs from ours.

    Stages initing from a previous checkpoint (S1->S2->S2b->S4) must consume
    charts with the identical render identity the vision encoder was trained
    on; a mismatch is a train/serve skew and is raised loudly. Checkpoints that
    predate render metadata (no ``checkpoint_meta.render``) are skipped.
    """
    parent_render = RenderContract.from_checkpoint_meta(
        ((parent_payload or {}).get("checkpoint_meta") or {})
    )
    if parent_render is None:
        print(
            f"Render contract: no parent render block to verify against "
            f"({stage_label} checkpoint predates render metadata) — skipping. "
            f"fingerprint={contract.render_fingerprint[:12]}"
        )
        return
    assert_contract_identity(parent_render, contract, require_checksum=False)
    print(
        f"Render contract verified vs {stage_label}: "
        f"fingerprint={contract.render_fingerprint[:12]} "
        f"spec={contract.render_spec_hash[:12]} renderer={contract.renderer_version}"
    )


def load_checkpoint(path: Path | str) -> dict:
    """Load a checkpoint payload (cpu, raw)."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=False)


def checkpoint_render(payload: dict) -> Optional[RenderContract]:
    """Extract the render contract recorded in a checkpoint, if present."""
    return RenderContract.from_checkpoint_meta(
        (payload or {}).get("checkpoint_meta") or {}
    )


def assert_serving_render(payload: dict, image_size: int) -> None:
    """Inference-side guard: a model may only be served with charts that match
    the visual identity it was trained/fine-tuned on."""
    recorded = checkpoint_render(payload)
    if recorded is None:
        print(
            f"Serving render contract: checkpoint predates render metadata — "
            f"not enforced. image_size={int(image_size)}"
        )
        return
    assert_contract_identity(
        recorded, default_spec_contract(int(image_size)), require_checksum=False
    )


def run_render_audit(
    checkpoint_path: Path | str,
    image_size: Optional[int] = None,
    charts_dir: Optional[Path | str] = None,
) -> dict:
    """One-shot render audit of a checkpoint (CLI / CI entry point).

    Checks, when available:
      * the checkpoint's recorded render identity (``checkpoint_meta.render``),
      * a compiled store under ``charts_dir`` (optional) against that identity,
      * the serving (inference) identity at ``image_size``.
    Raises :class:`RenderContractError` on any inconsistency.
    """
    payload = load_checkpoint(checkpoint_path)
    out: dict = {}
    recorded = checkpoint_render(payload)
    out["checkpoint"] = str(checkpoint_path)
    out["recorded_render"] = recorded.to_dict() if recorded else None
    parent = RenderContract.from_checkpoint_meta(
        (payload.get("checkpoint_meta") or {})
    )

    if charts_dir is not None:
        from zhisa.data.chart_store import CompiledChartStore

        chart_dir = Path(charts_dir)
        artifact = next(chart_dir.glob("*/charts.bin"), None)
        if artifact is None:
            raise FileNotFoundError(f"no compiled chart artefact under {chart_dir}")
        store = CompiledChartStore.open(artifact.parent)
        actual = RenderContract.from_store(store)
        out["stores"] = [store.render_meta["fingerprint"]]
        if recorded is not None:
            assert_contract_identity(recorded, store, require_checksum=True)
        out["store_check"] = "ok"

    if image_size is not None:
        expected = default_spec_contract(int(image_size))
        if recorded is not None:
            assert_contract_identity(recorded, expected, require_checksum=False)
        out["serving_check"] = "ok"
        out["serving_image_size"] = int(image_size)

    out["ok"] = True
    return out