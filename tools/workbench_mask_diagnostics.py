"""Describe saved observed-mask continuity without RGB or model execution."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from bench.workbench_mask_diagnostics import audit_saved_masks
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("clip-id", "mask-set"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--area-ratio-min", type=float, default=.5)
    parser.add_argument("--split", choices=("scoring", "onboarding", "all"), default="scoring")
    args = parser.parse_args(argv)
    try:
        report = audit_saved_masks(args.workspace, clip_id=args.clip_id, mask_set=args.mask_set,
            output_dir=args.output, area_ratio_min=args.area_ratio_min, split=args.split)
        print(json.dumps(dict(status=report["status"], output=str(args.output.absolute()),
                              frame_accounting=report["frame_accounting"], quality_claim=False)))
        return 0
    except Exception as exc:
        print(f"Mask diagnostics refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
