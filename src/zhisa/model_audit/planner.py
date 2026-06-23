from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from zhisa.model_audit.capabilities import inspect_prepared_data
from zhisa.model_audit.catalog import build_catalog
from zhisa.model_audit.schema import AuditPlan, AuditStatus, ModelStage, PlannedTest


_STAGE_ALIASES = {
    "s1_ssl": ModelStage.S1,
    "s2_supervised": ModelStage.S2,
    "s2b_imitation": ModelStage.S2B,
    "s3_curriculum": ModelStage.S3,
    "s4_ppo": ModelStage.S4,
    "s4_cvar_ppo": ModelStage.S4,
    "s4_rl": ModelStage.S4,
    "s5_continual": ModelStage.S5_PLUS,
    "s6_decision_transformer": ModelStage.S5_PLUS,
    "s7_world_model": ModelStage.S5_PLUS,
}


def checkpoint_identity(path: str | Path | None) -> tuple[ModelStage, bool, dict[str, Any]]:
    if path is None:
        return ModelStage.UNKNOWN, False, {}
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    meta = payload.get("checkpoint_meta") or {}
    raw_stage = str(meta.get("stage", "unknown"))
    stage = _STAGE_ALIASES.get(raw_stage, ModelStage.UNKNOWN)
    return stage, bool(meta.get("trading_policy_ready", stage.has_trained_policy)), meta


class AuditPlanner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def build(
        self,
        *,
        checkpoint: str | Path | None,
        prepared_root: str | Path | None,
        split: str = "test",
    ) -> AuditPlan:
        stage, policy_ready, checkpoint_meta = checkpoint_identity(checkpoint)
        data = inspect_prepared_data(prepared_root, split)
        configured = set(self.config.get("capabilities", {}).get("additional", []) or [])
        available = data.available | configured
        scenarios = self.config.get("scenarios", {}) or {}
        planned: list[PlannedTest] = []
        for spec in build_catalog():
            reasons: list[str] = []
            status = AuditStatus.READY
            if stage is not ModelStage.UNKNOWN and spec.stages and stage not in spec.stages:
                status = AuditStatus.NOT_APPLICABLE
                reasons.append(f"checkpoint stage {stage.value} is outside this test's valid stages")
            if spec.policy_required and not policy_ready:
                status = AuditStatus.NOT_APPLICABLE
                reasons.append("a trained trading policy is required; S1/S2 policy_logits are not valid")
            missing = sorted(set(spec.capabilities) - available)
            if missing and status is AuditStatus.READY:
                status = AuditStatus.BLOCKED
                reasons.append(f"missing data capabilities: {', '.join(missing)}")
            missing_empirical = sorted(set(spec.empirical_capabilities) - available)
            if missing_empirical and status is AuditStatus.READY:
                status = AuditStatus.MODELED
                reasons.append(
                    "execution can only be simulated without: " + ", ".join(missing_empirical)
                )
            scenario = dict(scenarios.get(spec.key, {}) or {})
            planned.append(PlannedTest(spec, status, reasons, scenario))
        capabilities = data.to_dict()
        capabilities["configured_additional"] = sorted(configured)
        capabilities["checkpoint_meta"] = checkpoint_meta
        return AuditPlan(str(checkpoint) if checkpoint else None, stage, policy_ready, capabilities, planned)


def save_plan(plan: AuditPlan, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit_plan.json"
    md_path = out_dir / "audit_plan.md"
    payload = plan.to_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Model audit plan",
        "",
        f"- Stage: `{plan.stage.value}`",
        f"- Trading policy ready: `{plan.trading_policy_ready}`",
        f"- Checkpoint: `{plan.checkpoint}`",
        "",
        "| ID | Level | Test | Status | Reason |",
        "|---:|---:|---|---|---|",
    ]
    for item in plan.tests:
        reason = "; ".join(item.reasons).replace("|", "/")
        lines.append(
            f"| {item.spec.id} | {item.spec.level} | {item.spec.title} | "
            f"`{item.status.value}` | {reason} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
