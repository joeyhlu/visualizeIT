"""Evaluate a full-window blank-input interruption without using it as accuracy truth."""
import json
import argparse
from pathlib import Path
import numpy as np
from .quality_assets import CACHE,ROOT,save_result
from .quality_contract import checked_pose
from .quality_evaluate import occlusion_recovery,secondary_pose_agreement
from .quality_failure_review import review
from .glb_model import read_glb
from .storage import write_artifact
from .vision import cv2


def strict_recovery(results,start):
    frames=results['frames']; ids=[f['frameId'] for f in frames]
    if len(ids)!=len(set(ids)) or any(b!=a+1 for a,b in zip(ids,ids[1:])):
        raise ValueError('Every consecutive source frame must be accounted for')
    summary=occlusion_recovery(results,start)
    hidden=[f for f in frames if start<=f['frameId']<start+15]
    private=all(f['cameraFromObject'] is None and f['pose_state']=='lost' and f['mask_state']=='lost' for f in hidden)
    returned=next((f for f in frames if f['frameId']>=start+15 and f['pose_state']=='tracking'),None)
    valid=False; confirmation=False
    if returned is not None and returned['cameraFromObject'] is not None:
        checked_pose(returned['cameraFromObject'])
        valid=returned['render_state']=='visible' and returned['mask_state']=='available'
        preceding=next((f for f in frames if f['frameId']==returned['frameId']-1),None)
        confirmation=bool(preceding and preceding['pose_state']=='recovering' and preceding['cameraFromObject'] is None
            and preceding['failure_reason']=='awaiting_second_validated_pose' and preceding.get('validation'))
    summary.update(hidden_poses_not_published=private, resumed_pose_mask_and_render_consistent=valid,
        consecutive_recovery_confirmation=confirmation)
    summary['passed']=summary['passed'] and private and valid and confirmation
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=CACHE/'results/keyboard/recovery-1239')
    root=parser.parse_args().root; result=json.loads((root/'complete.json').read_text())
    masks=json.loads((root/'segmentation/results.json').read_text())
    manifest=json.loads((CACHE/'inputs/keyboard/input.json').read_text())
    if [f['frameId'] for f in result['frames']]!=manifest['frame_ids']: raise ValueError('Full original window required')
    for data in (result,masks):
        stress=data.get('stress_test')
        if not stress or stress['occlusion_start']!=1239 or stress['source_frames']!=15:
            raise ValueError('Both actual inference stages must record the same 15-frame interruption')
    if not result['automatic'] or result['mode']!='complete': raise ValueError('Automatic initialization required')
    hidden_masks=[f for f in masks['frames'] if 1239<=f['frameId']<1254]
    empty=len(hidden_masks)==15 and all(not cv2.imread(str(root/'segmentation'/f['path']),cv2.IMREAD_GRAYSCALE).any() for f in hidden_masks)
    recovery=strict_recovery(result,1239)
    same=all(result['provenance'][k]==masks['provenance'][k] for k in ('input_manifest_sha256','adapter_sha256'))
    original=next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='keyboard')
    mesh=read_glb(CACHE/'inputs/keyboard/object.glb','keyboard')
    player=json.loads((ROOT/'artifacts/video-60/keyboard/player.json').read_text())
    detections=[dict(frame_id=f['frameId'],mask_state=f['mask_state'],reason=f.get('automatic_detection_reason'),
        elapsed_ms=f['timings_ms']['detection']) for f in masks['frames'] if 'detection' in f['timings_ms']]
    report=dict(scope=__doc__,object='keyboard',processed=240,accepted=sum(f['pose_state']=='tracking' for f in result['frames']),
        automatic_initialization=True,source_occlusion_start=1239,hidden_source_frames=15,
        mask_association=masks.get('mask_association',False),
        hidden_cached_masks_empty=empty,same_segmentation_pose_provenance=same,recovery=recovery,
        interruption_test_passed=empty and same and recovery['passed'],
        independent_accuracy_verified=False,physical_hand_occlusion_verified=False,overall_gate_passed=False,
        stages='SAM2 video memory then GoTrack/FoundPose; no manual corrections after setup',
        automatic_full_image_detection_attempts=detections,
        secondary_pose_agreement=secondary_pose_agreement(result,original,mesh),
        estimated_reference_orientation_review=review(result,player)['orientation_disagreement_intervals'],
        failure_reasons={reason:sum(f.get('failure_reason')==reason for f in result['frames'])
            for reason in sorted({f.get('failure_reason') for f in result['frames'] if f.get('failure_reason')})})
    save_result(root/'report.json',report)
    write_artifact(ROOT/'artifacts/model-quality/recovery-results.json',json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
