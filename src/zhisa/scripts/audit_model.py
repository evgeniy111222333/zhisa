"""Build a stage-aware, data-aware model audit plan without running it."""
from __future__ import annotations

import argparse
from pathlib import Path

from zhisa.config import load_config
from zhisa.model_audit.planner import AuditPlanner, save_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a model audit plan.")
    parser.add_argument("--config", default="configs/model_audit_60.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="artifacts/model_audit/plan")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    plan = AuditPlanner(cfg.to_dict()).build(
        checkpoint=args.checkpoint,
        prepared_root=args.prepared_root,
        split=args.split,
    )
    json_path, md_path = save_plan(plan, args.out)
    summary = plan.to_dict()["summary"]
    print(f"Audit plan prepared: {summary}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
