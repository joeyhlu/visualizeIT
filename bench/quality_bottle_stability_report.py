"""Evaluation-only full-window single-sample bottle rendering experiment."""
import json
from .quality_assets import ROOT,CACHE,save_result
from .quality_memory_report import summarize_run
from .quality_evaluate import prefix_equal
from .quality_surface_agreement import sample_surface,agreement
from .quality_failure_review import review
from .glb_model import read_glb
from .storage import write_artifact


def main():
    root=CACHE/'results/ranch/render-stability'
    read=lambda p:json.loads(p.read_text())
    short=read(root/'prefix-30.json');full=read(root/'complete.json')
    previous=read(CACHE/'results/ranch/appearance/without-appearance.json')
    original=next(o for o in read(ROOT/'artifacts/video-60/report.json')['objects'] if o['name']=='ranch')
    expected=list(range(original['sourceStart']+10,original['sourceStart']+250))
    if any([f['frameId'] for f in r['frames']]!=expected for r in (full,previous)) or len(short['frames'])!=30:
        raise ValueError('Unmodified full original windows and completed prefix required')
    mesh=read_glb(CACHE/'inputs/ranch/object.glb','ranch');points,normals,_,_=sample_surface(mesh);visibility={}
    report=dict(scope=__doc__,new=summarize_run(full,original,mesh),previous_unlit=summarize_run(previous,original,mesh),
        prefix_same_provenance=short['provenance']==full['provenance'],prefix_same_settings=short['tracking_settings']==full['tracking_settings'],
        prefix_identical=prefix_equal(short,full),
        common_input_masks=all(a['mask_sha256']==b['mask_sha256'] for a,b in zip(full['frames'],previous['frames'])),
        current_full_surface_agreement=agreement(full,original,mesh,points,normals,visibility),
        previous_full_surface_agreement=agreement(previous,original,mesh,points,normals,visibility),
        estimated_reference_orientation_review=review(full,read(ROOT/'artifacts/video-60/ranch/player.json'))['orientation_disagreement_intervals'],
        settings=full['tracking_settings'],independent_accuracy_verified=False,overall_gate_passed=False,
        previous_comparison_has_different_code_revision=previous['provenance']!=full['provenance'])
    report['candidate_rejected_for_quality']=bool(report['estimated_reference_orientation_review']) or (
        report['new']['accepted']<report['previous_unlit']['accepted'] and
        report['current_full_surface_agreement']['p95_720']>report['previous_full_surface_agreement']['p95_720'])
    save_result(root/'report.json',report)
    summary={**report,'current_full_surface_agreement':{k:v for k,v in report['current_full_surface_agreement'].items() if k!='details'},
        'previous_full_surface_agreement':{k:v for k,v in report['previous_full_surface_agreement'].items() if k!='details'}}
    write_artifact(ROOT/'artifacts/model-quality/bottle-render-stability-results.json',json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
