"""Explicit offline saved-window PNG placement and verified FFV1 export."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from bench.workbench_cli import Workspace
    from bench.workbench_offline_export import export_saved_window
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "project", "bundle", "admission-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "expected-revision", "bundle-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--role", choices=("primary_neural", "frozen_baseline"), required=True)
    parser.add_argument("--playback-fps", type=float, required=True)
    parser.add_argument("--placement", help="JSON object; scale/rotation_deg/offset_t_m/offset_b_m/opacity")
    parser.add_argument("--surface-mode", choices=("planar", "mesh"), default="planar")
    parser.add_argument("--max-chart-stretch", type=float, help="Optional finite stretch cap >=1; mesh only")
    parser.add_argument("--experimental-tracking", action="store_true",
        help="Require complete saved causal prefix and verified planar chart; no historical loss prediction")
    args = parser.parse_args(argv)
    try:
        result = export_saved_window(Workspace(args.workspace), run_id=args.run_id, role=args.role,
            project_path=args.project, expected_revision=args.expected_revision, bundle_path=args.bundle,
            bundle_sha256=args.bundle_sha256, admission_dir=args.admission_dir, output_dir=args.output,
            playback_fps=args.playback_fps, placement=json.loads(args.placement) if args.placement else None,
            experimental_tracking=args.experimental_tracking, surface_mode=args.surface_mode,
            max_chart_stretch=args.max_chart_stretch)
        print(json.dumps(dict(status="verified", output=str(args.output.resolve()), video=result["video"], frames=len(result["rows"]))))
        return 0
    except Exception as exc:
        print(f"Offline export refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
