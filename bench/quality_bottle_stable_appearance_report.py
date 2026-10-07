"""Same-code full-window stable-rendering bottle appearance ablation; evaluation only."""
import json
from .quality_assets import ROOT,CACHE,save_result
from .quality_memory_report import summarize_run
from .quality_evaluate import prefix_equal
from .quality_surface_agreement import sample_surface,agreement
from .quality_failure_review import review
from .glb_model import read_glb
from .storage import write_artifact


def main():
    read=lambda p:json.loads(p.read_text())
    root=CACHE/'results/ranch/render-stability-appearance'
    short=read(root/'prefix-30.json');full=read(root/'complete.json')
    control=read(CACHE/'results/ranch/render-stability/complete.json')
    original=next(o for o in read(ROOT/'artifacts/video-60/report.json')['objects'] if o['name']=='ranch')
    expected=list(range(original['sourceStart']+10,original['sourceStart']+250))
    if any([f['frameId'] for f in r['frames']]!=expected for r in (full,control)) or len(short['frames'])!=30:
        raise ValueError('All original frames and a completed prefix required')
    if full['provenance']!=control['provenance']:raise ValueError('Same-code/checkpoint/bank ablation required')
    if any(a['mask_sha256']!=b['mask_sha256'] for a,b in zip(full['frames'],control['frames'])):
        raise ValueError('Ablation must use identical actual masks')
    a=dict(full['tracking_settings']);b=dict(control['tracking_settings'])
    for key in ('appearance_check','appearance_render_mode','appearance_settings'):a.pop(key);b.pop(key)
    if a!=b:raise ValueError('Only the appearance evidence stage may change')
    mesh=read_glb(CACHE/'inputs/ranch/object.glb','ranch');points,normals,_,_=sample_surface(mesh);visibility={}
    before=agreement(control,original,mesh,points,normals,visibility);after=agreement(full,original,mesh,points,normals,visibility)
    common=set(f['frameId'] for f in full['frames'] if f['pose_state']=='tracking') & set(
        f['frameId'] for f in control['frames'] if f['pose_state']=='tracking')
    paired=lambda run:agreement({**run,'frames':[f for f in run['frames'] if f['frameId'] in common]},
        original,mesh,points,normals,visibility)
    orientation=review(full,read(ROOT/'artifacts/video-60/ranch/player.json'))['orientation_disagreement_intervals']
    suppression=[]
    for frame in full['frames']:
        if frame['pose_state']=='tracking':continue
        fid=frame['frameId']
        if not suppression or fid!=suppression[-1]['end']+1:
            suppression.append(dict(start=fid,end=fid,frames=1))
        else:suppression[-1].update(end=fid,frames=suppression[-1]['frames']+1)
    for interval in suppression:
        interval['next_accepted_frame']=next((f['frameId'] for f in full['frames']
            if f['frameId']>interval['end'] and f['pose_state']=='tracking'),None)
    report=dict(scope=__doc__,new=summarize_run(full,original,mesh),control=summarize_run(control,original,mesh),
        same_inference_provenance=True,identical_native_masks=True,
        prefix_same_provenance=short['provenance']==full['provenance'],prefix_same_settings=short['tracking_settings']==full['tracking_settings'],
        prefix_identical=prefix_equal(short,full),
        current_full_surface_agreement=after,control_full_surface_agreement=before,
        common_accepted_frames=len(common),common_current_surface_agreement=paired(full),
        common_control_surface_agreement=paired(control),
        estimated_reference_orientation_review=orientation,
        accepted_large_reference_orientation_disagreements=sum(g['accepted_frames'] for g in orientation),
        automatic_suppression_intervals=suppression,
        actual_appearance_rejections=sum('contradicts_rendered_texture' in f.get('rejection_reasons',[]) for f in full['frames']),
        independent_accuracy_verified=False,overall_gate_passed=False)
    save_result(root/'report.json',report)
    summary={**report}
    for key in ('current_full_surface_agreement','control_full_surface_agreement',
                'common_current_surface_agreement','common_control_surface_agreement'):
        summary[key]={k:v for k,v in report[key].items() if k!='details'}
    write_artifact(ROOT/'artifacts/model-quality/bottle-stable-appearance-results.json',json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
