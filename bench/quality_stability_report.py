"""Evaluation-only full keyboard single-sample renderer experiment."""
import json
from .quality_assets import CACHE,ROOT,save_result
from .quality_evaluate import prefix_equal,secondary_pose_agreement
from .quality_failure_review import review
from .quality_memory_report import summarize_run
from .glb_model import read_glb
from .storage import write_artifact


def main():
    root=CACHE/'results/keyboard/render-stability'
    prefix=json.loads((root/'prefix-30.json').read_text()); full=json.loads((root/'complete.json').read_text())
    original=next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='keyboard')
    expected=list(range(original['sourceStart']+10,original['sourceStart']+250))
    if [f['frameId'] for f in full['frames']] != expected: raise ValueError('All original frames required')
    mesh=read_glb(CACHE/'inputs/keyboard/object.glb','keyboard')
    visibility={}; previous=json.loads((CACHE/'results/keyboard/complete.json').read_text())
    common={f['frameId'] for f in previous['frames'] if f['pose_state']=='tracking' and f['cameraFromObject'] is not None}
    player=json.loads((ROOT/'artifacts/video-60/keyboard/player.json').read_text())
    report=dict(scope=__doc__,**summarize_run(full,original,mesh,visibility),
        previous_complete=summarize_run(previous,original,mesh,visibility),
        common_accepted_frames=len(common),
        common_frame_secondary_agreement=secondary_pose_agreement(dict(frames=[f for f in full['frames'] if f['frameId'] in common]),original,mesh,visibility),
        estimated_reference_orientation_review=review(full,player)['orientation_disagreement_intervals'],
        prefix_same_provenance=prefix['provenance']==full['provenance'],
        prefix_same_settings=prefix['tracking_settings']==full['tracking_settings'],
        prefix_identical=prefix_equal(prefix,full),settings=full['tracking_settings'],
        independent_accuracy_verified=False,overall_gate_passed=False)
    save_result(root/'full-report.json',report)
    write_artifact(ROOT/'artifacts/model-quality/render-stability-results.json',json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
