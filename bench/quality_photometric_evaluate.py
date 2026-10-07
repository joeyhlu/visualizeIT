"""Evaluation-only image alignment diagnostic; estimated controls are secondary."""
import argparse
import json
import numpy as np
from .quality_assets import CACHE, ROOT, save_result
from .quality_evaluate import secondary_pose_agreement
from .glb_model import read_glb


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--all', action='store_true')
    args = parser.parse_args(); name = 'photometric-all' if args.all else 'photometric-selected'
    trial = json.loads((CACHE/'diagnostics'/f'{name}.json').read_text())
    source = json.loads((CACHE/'results/ranch/appearance/complete.json').read_text())
    original = next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='ranch')
    mesh = read_glb(CACHE/'inputs/ranch/object.glb', 'ranch'); visibility = {}; rows=[]
    controls = {r['frameId']: r['cameraFromObject'] for r in original['referenceFrames']}
    seeds=[]; candidates=[]
    for f in trial['frames']:
        if f['seed_pose'] is None: continue
        fid=f['frame_id']; seeds.append(dict(frameId=fid, pose_state='tracking', cameraFromObject=f['seed_pose']))
        candidates.append(dict(frameId=fid, pose_state='tracking', cameraFromObject=f['proposed_pose']))
        entry=dict(frame_id=fid, optimizer_state=f['state'])
        control=controls[fid]
        if control is not None:
            for mode, pose in [('before',f['seed_pose']), ('after',f['proposed_pose'])]:
                r=np.array(control)[:3,:3].T@np.array(pose)[:3,:3]
                entry[mode+'_reference_rotation_degrees']=float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
        rows.append(entry)
    report=dict(scope=__doc__, frames_processed=len(trial['frames']), proposals=sum(f['state']=='proposal' for f in trial['frames']),
        before=secondary_pose_agreement(dict(frames=seeds),original,mesh,visibility),
        after=secondary_pose_agreement(dict(frames=candidates),original,mesh,visibility), rows=rows,
        selection='original benchmark; selected diagnostic frames' if not args.all else 'original full benchmark window',
        tracking_validated=False, independent_accuracy_verified=False, promoted=False)
    save_result(CACHE/'diagnostics'/f'{name}-evaluation.json', report)
    print(json.dumps(report, indent=2),flush=True)


if __name__ == '__main__': main()
