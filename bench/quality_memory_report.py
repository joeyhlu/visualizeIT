"""Separate automatic experiment report; no annotation gates are inferred."""
import json
import numpy as np
from .quality_assets import CACHE, ROOT, digest
from .quality_evaluate import prefix_equal, secondary_pose_agreement
from .glb_model import read_glb
from .storage import write_artifact


def summarize_run(result, original, mesh, visibility_cache=None):
    frames = result['frames']; accepted = [f for f in frames if f['pose_state'] == 'tracking']
    jumps = []
    for a, b in zip(accepted, accepted[1:]):
        r = np.array(a['cameraFromObject'])[:3, :3].T@np.array(b['cameraFromObject'])[:3, :3]
        degrees = float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1))))
        if degrees >= 90:
            jumps.append(dict(frame_id=b['frameId'], source_gap=b['frameId']-a['frameId'], rotation_degrees=degrees))
    return dict(processed=len(frames), accepted=len(accepted), availability=len(accepted)/len(frames),
        full_window=len(frames) == 240, large_rotations_requiring_image_review=jumps,
        secondary_pose_agreement=secondary_pose_agreement(result, original, mesh, visibility_cache),
        motion_evidence_frames=sum(f.get('validation', {}).get('model_motion', {}).get('state') == 'available'
                                  for f in frames if f.get('validation')),
        motion_contradictions=sum('contradicts_observed_model_motion' in f.get('rejection_reasons', []) for f in frames))


def main():
    root = CACHE/'results/keyboard/model-memory'
    prefix = json.loads((root/'prefix-30/complete.json').read_text())
    candidate = json.loads((root/'complete.json').read_text())
    control = json.loads((root/'without-memory.json').read_text())
    original = next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name'] == 'keyboard')
    expected = list(range(original['sourceStart']+10, original['sourceStart']+250))
    if [f['frameId'] for f in candidate['frames']] != expected or [f['frameId'] for f in control['frames']] != expected:
        raise ValueError('Both experiments must count all 240 original frames')
    mask_results = json.loads((root/'prefix-30/segmentation/results.json').read_text())
    matching_masks = all(digest(root/'prefix-30/segmentation'/f['path']) == digest(root/'segmentation'/f['path']) for f in mask_results['frames'])
    mesh = read_glb(CACHE/'inputs/keyboard/object.glb', 'keyboard')
    visibility_cache = {}
    report = dict(scope='Model-memory experiment on the unchanged keyboard window. Current-image motion checks supplement GoTrack; optical-flow-only poses are never displayed.',
        with_memory=summarize_run(candidate, original, mesh, visibility_cache), without_memory=summarize_run(control, original, mesh, visibility_cache),
        prefix_same_provenance=prefix['provenance'] == candidate['provenance'],
        segmentation_prefix_identical=matching_masks, pose_prefix_identical=prefix_equal(prefix, candidate),
        independent_accuracy_verified=False, overall_gate_passed=False,
        unverified=['40-frame independent annotations', 'forced occlusion recovery', 'frozen held-out windows', 'other objects'])
    write_artifact(ROOT/'artifacts/model-quality/model-memory-results.json', json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
