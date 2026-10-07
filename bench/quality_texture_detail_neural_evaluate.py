"""Separate estimated-reference evaluation of ten exploratory paired refinements."""
import json
import argparse
from .quality_assets import ROOT,CACHE,save_result
from .quality_surface_agreement import sample_surface,agreement
from .glb_model import read_glb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-root',default='bottle-texture-detail',choices=['bottle-texture-detail','bottle-texture-shape','bottle-texture-joint'])
    args=parser.parse_args()
    root=CACHE/'diagnostics'/args.proposal_root
    data=json.loads((root/'neural.json').read_text())
    if len(data['frames'])!=10:raise ValueError('All ten preselected diagnostic frames required')
    mesh=read_glb(CACHE/'inputs/ranch/object.glb','ranch');points,normals,_,_=sample_surface(mesh)
    original=next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='ranch')
    common=[f for f in data['frames'] if all(b['current_image_validated'] for b in f['branches'].values())]
    visibility={};summaries={}
    for name in ('control',data.get('proposal_label','detail_seed')):
        frames=[dict(frameId=f['frame_id'],pose_state='tracking',cameraFromObject=f['branches'][name]['pose']) for f in common]
        summaries[name]=agreement(dict(frames=frames),original,mesh,points,normals,visibility)
    report=dict(scope=__doc__,requested=10,common_validated_frames=len(common),compared_source_frames=[f['frame_id'] for f in common],
        branch_current_image_validated={name:sum(f['branches'][name]['current_image_validated'] for f in data['frames']) for name in summaries},
        secondary_surface_agreement=summaries,independent_accuracy_verified=False,full_window_benchmark=False,overall_gate_passed=False)
    save_result(root/'neural-evaluation.json',report)
    print(json.dumps({**report,'secondary_surface_agreement':{name:{k:v for k,v in value.items() if k!='details'} for name,value in summaries.items()}},indent=2))


if __name__=='__main__':main()
