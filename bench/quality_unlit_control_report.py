"""Incremental full-run report; appearance-assisted run may still be pending."""
import json
from .quality_assets import CACHE, ROOT
from .quality_memory_report import summarize_run
from .quality_failure_review import review
from .glb_model import read_glb
from .storage import write_artifact


def main():
    result = json.loads((CACHE/'results/ranch/appearance/without-appearance.json').read_text())
    if [f['frameId'] for f in result['frames']] != list(range(10,250)): raise ValueError('Full original bottle window required')
    original = next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='ranch')
    player = json.loads((ROOT/'artifacts/video-60/ranch/player.json').read_text())
    report = dict(scope=__doc__, experiment=summarize_run(result,original,read_glb(CACHE/'inputs/ranch/object.glb','ranch')),
        reference_orientation_review=review(result,player)['orientation_disagreement_intervals'],
        settings=result['tracking_settings'], independent_accuracy_verified=False, overall_gate_passed=False,
        remaining=['independent annotations','prefix invariance','forced occlusion','frozen additional windows'])
    write_artifact(ROOT/'artifacts/model-quality/unlit-control-results.json',json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
