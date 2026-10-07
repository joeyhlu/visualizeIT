"""Evaluator-only orientation diagnostics; estimated references are not truth.

Find sustained orientation disagreements which a consecutive-pose jump check
can miss. This module and its output must never be consumed by inference.
"""
import json
import numpy as np
from .quality_assets import ROOT, CACHE
from .storage import write_artifact


def rotation_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1))))


def review(result, original):
    references = {f['frameId']: f['reference'] for f in original['frames']}
    rows = []
    for f in result['frames']:
        pose, ref = f.get('cameraFromObject'), references[f['frameId']]
        angle = rotation_error(pose, ref) if pose is not None and ref is not None else None
        v = f.get('validation') or {}
        rows.append(dict(frame_id=f['frameId'], pose_state=f['pose_state'],
            reference_rotation_disagreement_degrees=angle,
            current_correspondence_median_720=v.get('median_reprojection_720'),
            current_correspondence_inliers=v.get('inliers'),
            failure_reason=f.get('failure_reason')))
    groups = []
    for r in rows:
        if r['reference_rotation_disagreement_degrees'] is None or r['reference_rotation_disagreement_degrees'] < 90:
            continue
        if not groups or r['frame_id'] != groups[-1][-1]['frame_id']+1: groups.append([])
        groups[-1].append(r)
    intervals = []
    for group in groups:
        peak = max(group, key=lambda r: r['reference_rotation_disagreement_degrees'])
        intervals.append(dict(start=group[0]['frame_id'], end=group[-1]['frame_id'],
            accepted_frames=len(group), peak_frame=peak['frame_id'],
            peak_rotation_degrees=peak['reference_rotation_disagreement_degrees']))
    return dict(frames_processed=len(rows), orientation_disagreement_intervals=intervals,
        frames=rows, bookmarks=[dict(frame_id=g['peak_frame'],
            label=f"Orientation review: frame {g['peak_frame']}") for g in intervals])


def main():
    objects = {}
    for alias in ('keyboard', 'mug', 'ranch'):
        result = CACHE/'results'/alias/'complete.json'
        if not result.exists(): continue
        original = json.loads((ROOT/'artifacts/video-60'/alias/'player.json').read_text())
        objects[alias] = review(json.loads(result.read_text()), original)
        print(alias, objects[alias]['orientation_disagreement_intervals'], flush=True)
    output = dict(scope='Evaluator-only review against estimated dataset poses from a related model family. '
        'Large disagreement identifies frames for image review, not independently proven errors. '
        'Low network correspondence residual does not prove orientation identity.',
        independent_accuracy_verified=False, objects=objects)
    write_artifact(ROOT/'artifacts/model-quality/failure-review.json', json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__': main()
