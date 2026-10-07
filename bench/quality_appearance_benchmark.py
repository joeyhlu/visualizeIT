"""Paired automatic texture-recovery experiment; independent gates remain open."""
import json
from .quality_assets import CACHE, ROOT, save_result
from .quality_memory_report import summarize_run
from .quality_evaluate import prefix_equal
from .quality_failure_review import review
from .glb_model import read_glb
from .storage import write_artifact


def main():
    root = CACHE/'results/ranch/appearance'
    runs = {m: json.loads((root/p).read_text()) for m,p in
        [('with_appearance','complete.json'),('without_appearance','without-appearance.json')]}
    original = next(o for o in json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects'] if o['name']=='ranch')
    player = json.loads((ROOT/'artifacts/video-60/ranch/player.json').read_text())
    expected = list(range(original['sourceStart']+10, original['sourceStart']+250))
    if any([f['frameId'] for f in r['frames']] != expected for r in runs.values()): raise ValueError('Both full original windows required')
    mesh = read_glb(CACHE/'inputs/ranch/object.glb','ranch'); visibility_cache = {}; summaries = {}
    for mode, result in runs.items():
        prefix = json.loads((root/'prefix-30'/('complete.json' if mode=='with_appearance' else 'without-appearance.json')).read_text())
        summaries[mode] = {**summarize_run(result, original, mesh, visibility_cache),
            'prefix_same_provenance': prefix['provenance']==result['provenance'],
            'prefix_identical': prefix_equal(prefix,result),
            'texture_rejections': sum('contradicts_rendered_texture' in f.get('rejection_reasons',[]) for f in result['frames']),
            'reference_orientation_review': review(result, player)['orientation_disagreement_intervals']}
    a, b = (runs[m] for m in ('with_appearance','without_appearance'))
    report = dict(scope=__doc__, **summaries, same_provenance=a['provenance']==b['provenance'],
        settings_tuned_on_original_windows=True, independent_accuracy_verified=False, overall_gate_passed=False,
        remaining=['independent annotations','forced occlusion','frozen additional windows','other objects'])
    save_result(root/'report.json',report)
    write_artifact(ROOT/'artifacts/model-quality/appearance-results.json',json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
