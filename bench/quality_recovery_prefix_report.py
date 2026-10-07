"""Evaluation-only repeatability across a real interruption, with cached-mask verification."""
import json
from .quality_assets import CACHE, ROOT, digest, save_result
from .quality_evaluate import prefix_equal
from .storage import write_artifact


def compare(short, long, short_masks, long_masks, short_root, long_root):
    ids = [f['frameId'] for f in short['frames']]
    if not ids or ids != [f['frameId'] for f in long['frames'][:len(ids)]]:
        raise ValueError('Ordered source prefix required')
    segmentation_ids = [short_masks['frames'][0]['frameId']]+ids
    a = short_masks['frames']; b = long_masks['frames'][:len(a)]
    if [f['frameId'] for f in a] != segmentation_ids or [f['frameId'] for f in b] != segmentation_ids:
        raise ValueError('Setup and every scored mask required')
    mask_differences = []
    for x, y in zip(a, b):
        fields = ('frameId', 'mask_state', 'failure_reason', 'automatic_detection_required', 'automatic_detection_reason')
        if any(x.get(k) != y.get(k) for k in fields) or digest(short_root/x['path']) != digest(long_root/y['path']):
            mask_differences.append(x['frameId'])
    # Pose records additionally hash their FoundPose bank; segmentation never loads it.
    # Compare each stage across runs, then verify all shared segmentation provenance.
    same_provenance = (short['provenance'] == long['provenance'] and
        short_masks['provenance'] == long_masks['provenance'] and
        all(short['provenance'].get(k)==v for k,v in short_masks['provenance'].items()))
    same_settings = (short['tracking_settings'] == long['tracking_settings'] and
        short['stress_test'] == long['stress_test'] and short_masks['stress_test'] == long_masks['stress_test'] and
        short_masks['mask_association'] == long_masks['mask_association'])
    pose_differences = [x['frameId'] for x, y in zip(short['frames'], long['frames'])
        if not prefix_equal(dict(frames=[x]), dict(frames=[y]))]
    return dict(scope=__doc__, scored_prefix_frames=len(ids), segmentation_prefix_frames=len(a),
        same_provenance=same_provenance, same_settings=same_settings,
        mask_prefix_identical=not mask_differences, pose_prefix_identical=not pose_differences,
        mask_differing_source_frames=mask_differences, pose_differing_source_frames=pose_differences,
        prefix_passed=same_provenance and same_settings and not mask_differences and not pose_differences,
        independent_accuracy_verified=False, overall_gate_passed=False)


def main():
    short_root=CACHE/'results/keyboard/recovery-association-prefix-120'
    long_root=CACHE/'results/keyboard/recovery-association-1239'
    read=lambda p: json.loads(p.read_text())
    short=read(short_root/'complete.json'); long=read(long_root/'complete.json')
    if len(short['frames']) != 120 or len(long['frames']) != 240:
        raise ValueError('Completed 120 versus 240 source frames required')
    report=compare(short,long,read(short_root/'segmentation/results.json'),read(long_root/'segmentation/results.json'),
        short_root/'segmentation',long_root/'segmentation')
    if not any(1239<=f['frameId']<1254 for f in short['frames']) or short['frames'][-1]['frameId']<1255:
        raise ValueError('Prefix must include actual interruption and recovery')
    save_result(short_root/'report.json',report)
    write_artifact(ROOT/'artifacts/model-quality/recovery-prefix-results.json',json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
