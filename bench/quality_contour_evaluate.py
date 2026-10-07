"""Evaluation-only secondary comparison of image-only contour proposals; not a gate."""
import json
from .quality_assets import ROOT,CACHE,save_result
from .quality_surface_agreement import sample_surface,agreement
from .glb_model import read_glb


def main():
    probe=json.loads((CACHE/'diagnostics/keyboard-contour-proposals.json').read_text())
    mesh=read_glb(CACHE/'inputs/keyboard/object.glb','keyboard');points,normals,_,_=sample_surface(mesh)
    original=next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='keyboard')
    before=[];after=[];available=[]
    for f in probe['frames']:
        if f['camera_from_object_proposal'] is None:continue
        available.append(f['frame_id'])
        # Tracking label is used only to calculate geometric diagnostic errors;
        # these proposals have not earned tracking/rendering availability.
        before.append(dict(frameId=f['frame_id'],pose_state='tracking',cameraFromObject=f['seed_camera_from_object']))
        after.append(dict(frameId=f['frame_id'],pose_state='tracking',cameraFromObject=f['camera_from_object_proposal']))
    visibility={};a=agreement(dict(frames=before),original,mesh,points,normals,visibility)
    b=agreement(dict(frames=after),original,mesh,points,normals,visibility)
    report=dict(scope=__doc__,requested=len(probe['frames']),available=len(available),compared_source_frames=available,
        before=a,after=b,independent_accuracy_verified=False,full_window_benchmark=False,overall_gate_passed=False,
        per_frame=[dict(frame_id=x['frame_id'],before_median=x['median_720'],after_median=y['median_720'],
            before_p95=x['p95_720'],after_p95=y['p95_720']) for x,y in zip(a['details'],b['details'])])
    save_result(CACHE/'diagnostics/keyboard-contour-evaluation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after')},indent=2))
    print('Before median/P95',a['median_720'],a['p95_720'],'after',b['median_720'],b['p95_720'])


if __name__=='__main__':main()
