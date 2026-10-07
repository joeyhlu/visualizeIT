"""Separate paired fitting-option evaluation; estimated reference poses only."""
import argparse
import json
from pathlib import Path

import numpy as np

from .quality_assets import CACHE, ROOT, save_result
from .quality_surface_agreement import sample_surface, agreement
from .quality_failure_review import rotation_error
from .quality_contract import checked_pose
from .glb_model import read_glb


BRANCHES = ('control', 'no_guess')
REQUESTED_FRAME_COUNT = 10


def _validate_record(data):
    """Reject malformed or incomplete records before computing any metrics."""
    if not isinstance(data, dict):
        raise ValueError('Diagnostic record must be an object')
    if 'status' in data and data['status'] != 'complete':
        raise ValueError(f"Diagnostic record status must be 'complete', found {data['status']!r}")
    if 'complete' in data and data['complete'] is not True:
        raise ValueError('Diagnostic record complete field must be true')
    requested = data.get('requested_frame_ids')
    if not isinstance(requested, list) or len(requested) != REQUESTED_FRAME_COUNT:
        raise ValueError(f'Expected exactly {REQUESTED_FRAME_COUNT} requested frame IDs')
    if any(type(frame_id) is not int for frame_id in requested):
        raise ValueError('Requested frame IDs must be integers')
    if len(set(requested)) != len(requested):
        raise ValueError('Requested frame IDs must be unique')

    frames = data.get('frames')
    if not isinstance(frames, list):
        raise ValueError('Diagnostic frames must be a list')
    if len(frames) < len(requested):
        raise ValueError(f'Incomplete diagnostic record: expected {len(requested)} frames, found {len(frames)}')
    frame_ids = []
    for frame in frames:
        if not isinstance(frame, dict) or type(frame.get('frame_id')) is not int:
            raise ValueError('Every diagnostic frame must have an integer frame_id')
        frame_ids.append(frame['frame_id'])
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError('Diagnostic frame IDs must be unique')
    if frame_ids != requested:
        raise ValueError('Diagnostic frame IDs must exactly match requested IDs in requested order')

    expected_branches = set(BRANCHES)
    for frame in frames:
        branches = frame.get('branches')
        if not isinstance(branches, dict) or set(branches) != expected_branches:
            raise ValueError(f"Frame {frame['frame_id']} must contain exactly the branches {BRANCHES}")
        for name in BRANCHES:
            branch = branches[name]
            if not isinstance(branch, dict):
                raise ValueError(f"Frame {frame['frame_id']} branch {name} must be an object")
            validated = branch.get('current_image_validated')
            if type(validated) is not bool:
                raise ValueError(f"Frame {frame['frame_id']} branch {name} needs a boolean validation state")
            if 'pose' not in branch:
                raise ValueError(f"Frame {frame['frame_id']} branch {name} needs a pose field")
            if validated:
                if branch['pose'] is None:
                    raise ValueError(f"Accepted frame {frame['frame_id']} branch {name} has no public pose")
                try:
                    checked_pose(branch['pose'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Accepted frame {frame['frame_id']} branch {name} has an invalid pose: {exc}") from exc
    return requested


def _public_branch_frame(frame, name):
    branch = frame['branches'][name]
    accepted = branch['current_image_validated']
    return dict(frameId=frame['frame_id'], pose_state='tracking' if accepted else 'lost',
                cameraFromObject=branch['pose'] if accepted else None)


def _empty_surface_summary():
    """Match agreement() on an empty paired intersection without scoring one."""
    return dict(processed=0, accepted=0, reference_frames=0, reference_visible_samples=0,
                accepted_point_samples=0, sample_availability=None, median_720=None,
                p95_720=None, details=[])


def build_report(data, alias, original, mesh, points, normals):
    """Build an evaluation report from saved inputs without reading or writing files."""
    requested = _validate_record(data)
    if data.get('object') != alias:
        raise ValueError(f"Diagnostic object {data.get('object')!r} does not match requested object {alias!r}")
    if not isinstance(original, dict):
        raise ValueError('Reference object must be a report object')
    if original.get('name') is not None and original['name'] != alias:
        raise ValueError(f"Reference object {original['name']!r} does not match requested object {alias!r}")
    visibility_cache = {}
    common = [frame for frame in data['frames']
              if all(frame['branches'][name]['current_image_validated'] for name in BRANCHES)]
    summaries = {}
    own_summaries = {}
    large_disagreements = {}
    all_source_frame_availability = {}
    for name in BRANCHES:
        paired_frames = [_public_branch_frame(frame, name) for frame in common]
        summaries[name] = (agreement(dict(frames=paired_frames), original, mesh, points, normals, visibility_cache)
                           if paired_frames else _empty_surface_summary())

        own_frames = [_public_branch_frame(frame, name) for frame in data['frames']]
        own_summaries[name] = agreement(dict(frames=own_frames), original, mesh, points, normals, visibility_cache)

        accepted_ids = [frame['frame_id'] for frame in data['frames']
                        if frame['branches'][name]['current_image_validated']]
        all_source_frame_availability[name] = dict(
            requested=len(requested), accepted=len(accepted_ids), failed=len(requested) - len(accepted_ids),
            availability=len(accepted_ids) / len(requested), failed_frame_ids=[fid for fid in requested if fid not in accepted_ids])

        refs = {frame['frameId']: frame['cameraFromObject'] for frame in original['referenceFrames']}
        large_disagreements[name] = [frame['frameId'] for frame in paired_frames
            if refs.get(frame['frameId']) is not None
            and rotation_error(frame['cameraFromObject'], refs[frame['frameId']]) >= 90]

    equal = sum(np.allclose(frame['branches']['control']['pose'], frame['branches']['no_guess']['pose'],
                            atol=1e-6, rtol=0) for frame in common)
    return dict(scope=__doc__, object=alias, requested=len(requested), recorded=len(data['frames']),
        record_status='complete', common_validated_frames=len(common),
        branch_current_image_validated={name: all_source_frame_availability[name]['accepted'] for name in BRANCHES},
        all_source_frame_availability=all_source_frame_availability,
        identical_final_poses=int(equal), secondary_surface_agreement=summaries,
        own_accepted_surface_agreement=own_summaries,
        large_estimated_reference_orientation_disagreements=large_disagreements,
        independent_accuracy_verified=False, full_window_benchmark=False, sequential_recovery_test=False,
        settings=data['settings'], overall_gate_passed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=['ranch', 'keyboard', 'mug'], required=True)
    parser.add_argument('--input-root', type=Path,
                        help='directory containing neural.json; defaults to the saved object diagnostic directory')
    parser.add_argument('--output', type=Path,
                        help='report path; defaults to evaluation.json beside the selected neural.json')
    args = parser.parse_args()

    root = args.input_root or CACHE / 'diagnostics/pnp-guess' / args.object
    data = json.loads((root / 'neural.json').read_text())
    alias = args.object
    mesh = read_glb(CACHE / 'inputs' / alias / 'object.glb', alias)
    points, normals, _, _ = sample_surface(mesh)
    original = next(obj for obj in json.loads((ROOT / 'artifacts/video-60/report.json').read_text())['objects']
                    if obj['name'] == alias)
    report = build_report(data, alias, original, mesh, points, normals)
    output = args.output or root / 'evaluation.json'
    save_result(output, report)
    print(json.dumps({**report, 'secondary_surface_agreement': {
        name: {key: value for key, value in summary.items() if key != 'details'}
        for name, summary in report['secondary_surface_agreement'].items()},
        'own_accepted_surface_agreement': {
        name: {key: value for key, value in summary.items() if key != 'details'}
        for name, summary in report['own_accepted_surface_agreement'].items()}}, indent=2))


if __name__ == '__main__':
    main()
