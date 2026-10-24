#!/usr/bin/env python3
"""Run one bounded stage of the real two-view Kaggle CUDA comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "render", "banks", "legacy", "amortized", "compare"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    # Keep --help model-free; import only the repository's checked source root.
    sys.path.insert(0, str(ROOT))
    from bench import quality_kaggle_activation as activation

    try:
        context = activation.load_request(args.request, args.request_sha256)
        if args.stage == "freeze":
            result = activation.freeze(context)
        else:
            with activation.stage(context, args.stage, cloud_execution=args.cloud_execution):
                if args.stage in ("legacy", "amortized"):
                    result = activation.run_branch(context, args.stage + "-v1")
                else:
                    result = getattr(activation, args.stage)(context)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "stage": args.stage,
                          "error": str(error)}, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
