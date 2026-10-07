"""Publish bounded summaries of rejected selected-frame bottle pose diagnostics."""
import json
from .quality_assets import CACHE,ROOT,digest
from .storage import write_artifact


def main():
    results={}
    for key in ('detail','shape'):
        root=CACHE/'diagnostics'/f'bottle-texture-{key}'
        report=json.loads((root/'neural-evaluation.json').read_text())
        results[key]={**report,'secondary_surface_agreement':{name:{k:v for k,v in value.items() if k!='details'}
            for name,value in report['secondary_surface_agreement'].items()},
            'rejected_for_typical_error':True,'inference_result_sha256':digest(root/'neural.json')}
    joint=json.loads((CACHE/'diagnostics/bottle-texture-joint/probe.json').read_text())
    report=dict(scope=__doc__,selected_after_candidate_review=True,original_window_frames=10,
        full_window_benchmark=False,independent_accuracy_verified=False,tracking_defaults_changed=False,
        paired_neural_trials=results,joint_proposals_changed=sum(row.get('best_degrees',0)!=0 for row in joint['rows']),
        joint_refinement_run=False,
        reason='No joint-evidence seed differs from its control; a redundant neural run was not started.',
        next_investigation='Verify rendered-to-image correspondences and pose ambiguity; low network residual alone is insufficient.')
    write_artifact(ROOT/'artifacts/model-quality/bottle-texture-diagnostic-results.json',json.dumps(report,indent=2))
    print('Preserved rejected paired diagnostics; joint changed',report['joint_proposals_changed'],'of 10')


if __name__=='__main__':main()
