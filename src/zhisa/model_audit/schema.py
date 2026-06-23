from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ModelStage(str, Enum):
    S1 = "s1_ssl"
    S2 = "s2_supervised"
    S2B = "s2b_imitation"
    S3 = "s3_curriculum"
    S4 = "s4_rl"
    S5_PLUS = "s5_plus"
    UNKNOWN = "unknown"

    @property
    def has_trained_policy(self) -> bool:
        return self in {self.S2B, self.S3, self.S4, self.S5_PLUS}


class AuditStatus(str, Enum):
    READY = "ready"
    MODELED = "modeled"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class TestSpec:
    id: int
    key: str
    title: str
    level: int
    objective: str
    handler: str
    capabilities: tuple[str, ...] = ()
    stages: tuple[ModelStage, ...] = ()
    policy_required: bool = False
    empirical_capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass
class PlannedTest:
    spec: TestSpec
    status: AuditStatus
    reasons: list[str] = field(default_factory=list)
    scenario: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        out["spec"]["stages"] = [stage.value for stage in self.spec.stages]
        return out


@dataclass
class AuditPlan:
    checkpoint: str | None
    stage: ModelStage
    trading_policy_ready: bool
    capabilities: dict[str, Any]
    tests: list[PlannedTest]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "stage": self.stage.value,
            "trading_policy_ready": self.trading_policy_ready,
            "capabilities": self.capabilities,
            "summary": {
                status.value: sum(test.status is status for test in self.tests)
                for status in AuditStatus
            },
            "tests": [test.to_dict() for test in self.tests],
        }
