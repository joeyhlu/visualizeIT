"""Inspect selected source, observed-mask and composite pixels from a saved export."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from bench.workbench_export_review import review_saved_export
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "export", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--frame-id", action="append", required=True)
    parser.add_argument("--tile-width", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        report = review_saved_export(args.workspace, export_dir=args.export,
            expected_manifest_sha256=args.manifest_sha256, frame_ids=args.frame_id,
            output_dir=args.output, tile_width=args.tile_width)
        print(json.dumps(dict(status=report["status"], output=str(args.output.absolute()), quality_claim=False)))
        return 0
    except Exception as exc:
        import sys
        print(f"Export review refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
