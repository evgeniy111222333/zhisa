"""CLI: audit a checkpoint's render contract (CI / pre-launch gate).

Reads a checkpoint, prints its recorded render identity, optionally verifies a
compiled chart store at byte level, and checks the serving (inference)
identity at a given image size. Exits non-zero on any inconsistency.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a ZHISA checkpoint (.pt)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Serving image size to validate against the recorded identity")
    parser.add_argument("--charts-dir", default=None,
                        help="Compiled chart store root to verify (byte-level)")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = parser.parse_args(argv)

    from zhisa.data.render_contract import (
        RenderContractError,
        run_render_audit,
    )

    try:
        result = run_render_audit(
            args.checkpoint,
            image_size=args.image_size,
            charts_dir=args.charts_dir,
        )
    except RenderContractError as exc:
        print(f"RENDER CONTRACT AUDIT FAILED: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"RENDER CONTRACT AUDIT FAILED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"checkpoint: {result['checkpoint']}")
        rec = result.get("recorded_render")
        if rec:
            print(
                f"  recorded: renderer={rec['renderer_version']} "
                f"spec={rec['render_spec_hash'][:12]} fp={rec['render_fingerprint'][:12]}"
            )
        else:
            print("  recorded render identity: <none> (predates render metadata)")
        if "serving_image_size" in result:
            print(f"  serving image_size={result['serving_image_size']} -> ok")
        if "store_check" in result:
            print(f"  compiled store byte-check -> ok")
        print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())