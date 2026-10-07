"""Evaluator-only summary of measured runs, preserving failures and scope."""
import json
import numpy as np
from .quality_assets import ROOT, CACHE
from .quality_evaluate import secondary_pose_agreement, prefix_equal
from .glb_model import read_glb
from .storage import write_artifact


def summarize():
    originals = json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects']
    rows = []
    for original in originals:
        alias = original['name']; root = CACHE/'results'/alias
        model = read_glb(CACHE/'inputs'/alias/'object.glb', alias)
        visibility_cache = {}
        for mode in ('controlled', 'complete'):
            path = root/f'{mode}.json'
            if not path.exists(): continue
            result = json.loads(path.read_text()); frames = result['frames']
            if not frames: continue
            ids = [f['frameId'] for f in frames]
            expected = list(range(original['sourceStart']+10, original['sourceStart']+250))
            if ids != expected[:len(ids)]: raise ValueError('Modified benchmark window')
            accepted = sum(f['pose_state'] == 'tracking' for f in frames)
            accepted_frames = [f for f in frames if f['pose_state'] == 'tracking']
            large_rotations = []
            for earlier, later in zip(accepted_frames, accepted_frames[1:]):
                a = np.array(earlier['cameraFromObject']); b = np.array(later['cameraFromObject'])
                angle = float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T@b[:3, :3])-1)/2, -1, 1))))
                if angle >= 90:
                    large_rotations.append(dict(frame_id=later['frameId'], rotation_degrees=angle,
                                                source_frame_gap=later['frameId']-earlier['frameId']))
            # Previous poses are compared over exactly the same prefix.
            player = json.loads((ROOT/'artifacts/video-60'/alias/'player.json').read_text())
            previous = {f['frameId']: f for f in player['frames']}
            baseline = sum(previous[fid]['poses'][0] is not None for fid in ids)
            faster = sum(previous[fid]['poses'][1] is not None for fid in ids)
            previous_results = dict(frames=[dict(frameId=fid, cameraFromObject=previous[fid]['poses'][0],
                pose_state='tracking' if previous[fid]['poses'][0] is not None else 'lost') for fid in ids])
            common = {f['frameId'] for f in frames if f['pose_state'] == 'tracking' and previous[f['frameId']]['poses'][0] is not None}
            common_new = dict(frames=[f for f in frames if f['frameId'] in common])
            common_old = dict(frames=[f for f in previous_results['frames'] if f['frameId'] in common])
            prefix_path = root/'prefix-v2'/f'{mode}.json'
            if not prefix_path.exists(): prefix_path = root/'prefix-30'/f'{mode}.json'
            invariant = None; prefix_same_provenance = None
            if len(ids) == 240 and prefix_path.exists():
                prefix = json.loads(prefix_path.read_text())
                prefix_same_provenance = bool(prefix.get('provenance')) and prefix.get('provenance') == result.get('provenance') and prefix.get('tracking_settings') == result.get('tracking_settings')
                if prefix_same_provenance: invariant = prefix_equal(prefix, result)
            rows.append(dict(object=alias, mode=mode, frames_processed=len(ids), frames_required=240,
                full_original_window=ids == expected, accepted=accepted, failures=len(ids)-accepted,
                availability=accepted/len(ids), previous_baseline_accepted=baseline, previous_faster_accepted=faster,
                large_rotation_transitions_requiring_image_review=large_rotations,
                secondary_pose_agreement=secondary_pose_agreement(result, original, model, visibility_cache),
                previous_baseline_secondary_agreement=secondary_pose_agreement(previous_results, original, model, visibility_cache),
                common_accepted_frames=len(common),
                common_frame_new_secondary_agreement=secondary_pose_agreement(common_new, original, model, visibility_cache),
                common_frame_previous_secondary_agreement=secondary_pose_agreement(common_old, original, model, visibility_cache),
                prefix_30_matches_full=invariant, independent_accuracy_verified=False,
                prefix_same_provenance=prefix_same_provenance,
                pose_median_ms=float(np.median([f['timings_ms'].get('pose_total', 0) for f in frames])),
                failure_reasons={reason: sum(f.get('failure_reason') == reason for f in frames)
                    for reason in sorted({f.get('failure_reason') for f in frames if f.get('failure_reason')})}))
    report = dict(schema_version=1, scope='Original consecutive source windows; partial runs are diagnostics. '
        'SHOW3D pose agreement is secondary and shares model lineage. Availability does not prove accuracy. '
        'Independent annotation, recovery and additional-window gates remain required.', runs=rows,
        docker_started=False, execution_environment='workspace-isolated Windows Python 3.10 / CUDA 12.4',
        overall_gate_passed=False)
    # Fixed, small evaluated summaries; never raw inference caches or annotations.
    report['experiments']={}
    for filename in ('appearance-results.json','render-stability-results.json','recovery-results.json',
                     'recovery-prefix-results.json','surface-agreement-results.json',
                     'bottle-render-stability-results.json','bottle-stable-appearance-results.json',
                     'bottle-texture-diagnostic-results.json'):
        summary_path=ROOT/'artifacts/model-quality'/filename
        if summary_path.exists():report['experiments'][filename]=json.loads(summary_path.read_text())
    write_artifact(ROOT/'artifacts/model-quality/measured-runs.json', json.dumps(report, indent=2, allow_nan=False))
    for r in rows:
        print(r['object'], r['mode'], str(r['accepted'])+'/'+str(r['frames_processed']),
              'previous baseline',r['previous_baseline_accepted'],
              'secondary median/P95',r['secondary_pose_agreement']['median_720'],r['secondary_pose_agreement']['p95_720'],flush=True)


if __name__ == '__main__': summarize()
