"""Synthetic CPU fixtures for the completed mug comparison player integration."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import quality_player as player


SOURCE_IDS = list(player.MUG_PNP_FRAME_IDS)
MASK_BYTES = b'synthetic observed mask bytes'
MASK_SHA256 = hashlib.sha256(MASK_BYTES).hexdigest()
SOURCE_FILES = {'bench/quality_runner.py': 'a'*64}
FROZEN_FILES = {'inputs/mug.mp4': 'b'*64, 'masks/827.png': MASK_SHA256}
PROVENANCE = {'model_revision': 'synthetic-fixture', 'checkpoint_sha256': 'c'*64}


def _pose():
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]


def _frames(lost_ids, recovering_ids, masks):
    result = []
    for frame_id in SOURCE_IDS:
        if frame_id in recovering_ids:
            state = 'recovering'
        elif frame_id in lost_ids:
            state = 'lost'
        else:
            state = 'tracking'
        result.append({
            'frameId': frame_id,
            'mask_state': 'available',
            'pose_state': state,
            'render_state': 'visible' if state == 'tracking' else 'suppressed',
            'failure_reason': None if state == 'tracking' else 'synthetic_loss',
            'cameraFromObject': _pose() if state == 'tracking' else None,
            'mask_sha256': masks[str(frame_id)],
        })
    return result


def _write_json(path, data):
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')


def _populate_experiment(root, control_lost=None, candidate_lost=None,
                         control_area=(2.0, 5.0), candidate_area=(3.0, 7.0),
                         candidate_recovering=()):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    control_lost = set(control_lost if control_lost is not None else range(827, 837))
    candidate_lost = set(candidate_lost if candidate_lost is not None else range(837, 849))
    candidate_recovering = set(candidate_recovering)
    masks = {str(frame_id): MASK_SHA256 for frame_id in SOURCE_IDS}
    snapshot = {
        'schema_version': 1, 'object': 'mug', 'run_mode': 'complete',
        'setup_frame_id': 826,
        'expected_frame_ids': SOURCE_IDS, 'prefix_lengths': list(player.MUG_PNP_PREFIX_LENGTHS),
        'source_files': SOURCE_FILES, 'frozen_files': FROZEN_FILES,
        'inference_provenance': PROVENANCE, 'mask_sha256_by_frame': masks,
    }
    _write_json(root/'experiment.json', snapshot)

    full_runs = {}
    stages = []
    for branch in player.MUG_PNP_BRANCHES:
        lost_ids = control_lost if branch == 'control' else candidate_lost
        recovering_ids = set() if branch == 'control' else candidate_recovering
        for length in player.MUG_PNP_PREFIX_LENGTHS:
            stage_id = f'{branch}-{length}'
            ids = SOURCE_IDS[:length]
            settings = {
                **player.MUG_PNP_FIXED_SETTINGS,
                'pnp_use_extrinsic_guess': branch == 'control',
            }
            runtime = {'device': 'cuda', 'fixture': 'synthetic'}
            run = {
                'object': 'mug', 'mode': 'complete', 'complete': length == 240,
                # Production outputs omit status. The viewer must accept that schema.
                'frame_ids': ids,
                'frames': _frames(lost_ids & set(ids), recovering_ids & set(ids), masks)[:length],
                'tracking_settings': settings, 'runtime': runtime, 'provenance': PROVENANCE,
                'automatic': True, 'diagnostic_control': False,
                'independent_accuracy_scored': False, 'stress_test': None,
            }
            if length == 240:
                full_runs[branch] = run
            output_name = f'{stage_id}.json'
            output_path = root/output_name
            _write_json(output_path, run)
            output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
            trace_name = f'{stage_id}.trace.jsonl'
            trace_record = {'stage': stage_id, 'arrays': {'seed': [1.0]}}
            trace_bytes = (json.dumps(trace_record, separators=(',', ':'))+'\n').encode('utf-8')
            (root/trace_name).write_bytes(trace_bytes)
            trace_hash = hashlib.sha256(trace_bytes).hexdigest()
            stages.append({
                'stage_id': stage_id, 'status': 'succeeded', 'exit_code': 0,
                'terminal_exit_recorded': True, 'output': output_name,
                'output_sha256': output_hash, 'requested_frame_ids': ids,
                'result_frame_ids': ids, 'expected_complete': length == 240,
                'result_complete': length == 240,
                'pnp_use_extrinsic_guess': branch == 'control',
                'tracking_settings': settings, 'runtime': runtime,
                'source_files': SOURCE_FILES, 'frozen_files': FROZEN_FILES,
                'inference_provenance': PROVENANCE, 'trace': trace_name,
                'trace_sha256': trace_hash, 'trace_record_count': 1,
            })

    control_accepted = 240-len(control_lost)
    candidate_accepted = 240-len(candidate_lost)
    control_frames = {frame['frameId']: frame for frame in full_runs['control']['frames']}
    candidate_frames = {frame['frameId']: frame for frame in full_runs['candidate']['frames']}
    common_ids = [frame_id for frame_id in SOURCE_IDS
                  if control_frames[frame_id]['pose_state'] == 'tracking'
                  and candidate_frames[frame_id]['pose_state'] == 'tracking']
    prefixes = {}
    for key in player.MUG_PNP_PREFIX_KEYS:
        prefixes[key] = {
            'passed': True, 'runtime_equal': True, 'provenance_equal': True,
            'settings_equal': True, 'prefix_equal': True,
            'trace': {'supplied': True, 'passed': True, 'comparison': None},
        }
    common_error = {'control': control_area, 'candidate': candidate_area}
    own_error = {'control': control_area, 'candidate': candidate_area}
    gates = {
        'full_six_stage_lifecycle_exit_zero': True,
        'frozen_snapshot_intact': True,
        'paired_provenance_and_settings_match': True,
        'candidate_accepted_at_least_control': candidate_accepted >= control_accepted,
        'candidate_accepted_at_least_216': candidate_accepted >= 216,
        'common_area_median_not_worse': common_error['candidate'][0] <= common_error['control'][0],
        'common_area_p95_strictly_better': common_error['candidate'][1] < common_error['control'][1],
        'own_area_median_not_worse': own_error['candidate'][0] <= own_error['control'][0],
        'own_area_p95_not_worse': own_error['candidate'][1] <= own_error['control'][1],
        'no_new_90_degree_disagreement': True,
    }
    gates.update({f'prefix_{key}': value['passed'] for key, value in prefixes.items()})

    def accounting(count):
        return {
            'requested_source_frames': 240, 'source_frames': 240,
            'run_complete': True, 'accepted_frames': count, 'failures': 240-count,
            'accepted_pose_availability': count/240,
        }

    def metrics(area, processed, accepted):
        return {
            'processed': processed, 'accepted': accepted, 'source_frames': processed,
            'median_720': area[0] if accepted else None,
            'p95_720': area[1] if accepted else None,
        }

    report = {
        'schema_version': 1, 'object': 'mug',
        'requested_source_frames': 240, 'source_frame_ids': SOURCE_IDS,
        'branch_accounting': {
            'control': accounting(control_accepted),
            'candidate': accounting(candidate_accepted),
        },
        'own_accepted_fixed128_area_metrics': {
            'control': metrics(control_area, 240, control_accepted),
            'candidate': metrics(candidate_area, 240, candidate_accepted),
        },
        'common_accepted_frame_ids': common_ids,
        'common_accepted_frames': len(common_ids),
        'common_accepted_fixed128_area_metrics': {
            'control': metrics(control_area, len(common_ids), len(common_ids)),
            'candidate': metrics(candidate_area, len(common_ids), len(common_ids)),
        },
        'paired_source_settings_checks': {'passed': True, 'issues': []},
        'prefix_checks': prefixes,
        'terminal_stage_evidence': {'passed': True, 'issues': [], 'stages': stages},
        'frozen_snapshot_intact': True,
        'frozen_snapshot_expected_mask_sha256': masks,
        'full_window_results_complete': True,
        'full_comparison_complete': True,
        'continuation_gate_checks': gates,
        'continuation_gate_passed': all(gates.values()),
        'new_candidate_large_orientation_disagreements': {'new_candidate_frame_ids': []},
        'independent_accuracy_verified': False, 'overall_gate_passed': False,
        'reference_or_annotation_inputs': False, 'annotations_loaded': False,
    }
    _write_json(root/'report.json', report)
    (root/'observed-mask.bin').write_bytes(MASK_BYTES)
    return report


class MugComparisonPlayerTests(unittest.TestCase):
    def test_missing_report_omits_comparison_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(player._load_mug_pnp_comparison(temporary))

    def test_incomplete_report_is_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = _populate_experiment(temporary)
            report['full_comparison_complete'] = False
            _write_json(Path(temporary)/'report.json', report)
            self.assertIsNone(player._load_mug_pnp_comparison(temporary))

    def test_output_and_trace_byte_tampering_omits_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(temporary)
            output = Path(temporary)/'candidate-240.json'
            output.write_bytes(output.read_bytes()+b' ')
            self.assertIsNone(player._load_mug_pnp_comparison(temporary))

        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(temporary)
            trace = Path(temporary)/'control-120.trace.jsonl'
            trace.write_bytes(trace.read_bytes()+b' ')
            self.assertIsNone(player._load_mug_pnp_comparison(temporary))

    def test_noncomplete_status_is_rejected_but_absent_status_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(temporary)
            loaded = player._load_mug_pnp_comparison(temporary)
            self.assertIsNotNone(loaded)

            root = Path(temporary)
            stage_path = root/'control-240.json'
            run = json.loads(stage_path.read_text(encoding='utf-8'))
            run['status'] = 'running'
            _write_json(stage_path, run)
            report_path = root/'report.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            stage = next(row for row in report['terminal_stage_evidence']['stages']
                         if row['stage_id'] == 'control-240')
            stage['output_sha256'] = hashlib.sha256(stage_path.read_bytes()).hexdigest()
            _write_json(report_path, report)
            self.assertIsNone(player._load_mug_pnp_comparison(temporary))

    def test_rejected_completed_report_shows_neutral_paired_metrics_and_caveat(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = _populate_experiment(temporary)
            loaded = player._load_mug_pnp_comparison(temporary)
            self.assertIsNotNone(loaded)
            self.assertFalse(loaded['summary']['continuation_gate_passed'])
            caption = player._mug_pnp_summary_html(loaded)
            self.assertIn('Continuation gate rejected:', caption)
            self.assertIn('candidate accepted fewer frames than control', caption)
            self.assertIn('No-extrinsic-guess ablation', caption)
            self.assertIn('estimated reference poses from the same model family under study', caption)
            self.assertIn('Overall and independent-accuracy gates remain false', caption)
            self.assertNotIn('candidate improved', caption.lower())
            self.assertFalse(report['overall_gate_passed'])

    def test_passing_criteria_caption_keeps_independent_accuracy_false(self):
        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(
                temporary, candidate_lost=range(837, 845),
                candidate_area=(1.5, 4.0),
            )
            loaded = player._load_mug_pnp_comparison(temporary)
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded['summary']['continuation_gate_passed'])
            caption = player._mug_pnp_summary_html(loaded)
            self.assertIn('continuation criteria passed', caption)
            self.assertIn('Overall and independent-accuracy gates remain false', caption)
            self.assertIn('not independent physical attachment errors', caption)

    def test_exact_frame_mapping_uses_null_pose_for_lost_and_recovering_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(temporary, candidate_recovering={840})
            loaded = player._load_mug_pnp_comparison(temporary)
            self.assertIsNotNone(loaded)
            by_id = loaded['frames_by_id']
            self.assertEqual(list(by_id), SOURCE_IDS)
            self.assertEqual(by_id[840]['mug_pnp_no_extrinsic_guess']['pose_state'], 'recovering')
            self.assertIsNone(by_id[840]['mug_pnp_no_extrinsic_guess']['cameraFromObject'])
            self.assertEqual(by_id[840]['mug_pnp_no_extrinsic_guess']['render_state'], 'suppressed')
            self.assertEqual(by_id[837]['mug_pnp_no_extrinsic_guess']['pose_state'], 'lost')
            self.assertIsNone(by_id[837]['mug_pnp_no_extrinsic_guess']['cameraFromObject'])

            root = Path(temporary)
            masks = {frame_id: {'path': 'observed-mask.bin', 'mask_state': 'available'}
                     for frame_id in SOURCE_IDS}
            player_frames = [{'frameId': frame_id, 'new': {'complete': {'pose_state': 'tracking'}}}
                             for frame_id in SOURCE_IDS]
            aligned = player._mug_pnp_modes_for_player(
                {'frames': player_frames}, masks, root, loaded)
            self.assertIsNotNone(aligned)
            self.assertEqual(list(aligned), SOURCE_IDS)
            self.assertEqual(aligned[840]['mug_pnp_no_extrinsic_guess']['pose_state'], 'recovering')
            self.assertIsNone(aligned[840]['mug_pnp_no_extrinsic_guess']['cameraFromObject'])

            reordered = list(reversed(player_frames))
            self.assertIsNone(player._mug_pnp_modes_for_player(
                {'frames': reordered}, masks, root, loaded))

    def test_observed_mask_hash_mismatch_omits_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(temporary)
            loaded = player._load_mug_pnp_comparison(temporary)
            self.assertIsNotNone(loaded)
            root = Path(temporary)
            (root/'observed-mask.bin').write_bytes(b'different observed mask')
            masks = {frame_id: {'path': 'observed-mask.bin', 'mask_state': 'available'}
                     for frame_id in SOURCE_IDS}
            player_frames = [{'frameId': frame_id} for frame_id in SOURCE_IDS]
            self.assertIsNone(player._mug_pnp_modes_for_player(
                {'frames': player_frames}, masks, root, loaded))

    def test_generated_selection_clears_canvases_and_ignores_late_previous_case(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node.js is required for the generated-player mock-DOM regression')

        with tempfile.TemporaryDirectory() as temporary:
            _populate_experiment(temporary)
            comparison = player._load_mug_pnp_comparison(temporary)
            self.assertIsNotNone(comparison)
            artifacts = {}

            def fake_modes(original, masks, mask_root, verified_comparison):
                self.assertIs(verified_comparison, comparison)
                return {
                    frame_id: {
                        mode: {
                            'frameId': frame_id,
                            'pose_state': 'tracking',
                            'render_state': 'visible',
                            'cameraFromObject': _pose(),
                        }
                        for mode in player.MUG_PNP_MODES
                    }
                    for frame_id in player.MUG_PNP_FRAME_IDS
                }

            with patch.object(player, '_load_mug_pnp_comparison', return_value=comparison), \
                    patch.object(player, '_mug_pnp_modes_for_player', side_effect=fake_modes), \
                    patch.object(player, 'write_artifact', side_effect=lambda path, text: artifacts.__setitem__(Path(path), text)):
                player.build()

            page = artifacts[player.ROOT/'artifacts/model-quality/index.html']
            scripts = re.findall(r'<script[^>]*>(.*?)</script>', page, flags=re.DOTALL)
            self.assertEqual(len(scripts), 2)
            generated_js = scripts[-1]
            result = subprocess.run(
                [node, '--check'], input=generated_js, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            options_start = generated_js.index('function updateMugPnpOptions(){')
            options_end = generated_js.index('\nlet data=null', options_start)
            select_start = generated_js.index('async function selectCase(c){')
            select_end = generated_js.index('\nfor(const c of cases)', select_start)
            generated_functions = generated_js[options_start:options_end]+'\n'+generated_js[select_start:select_end]
            harness = r'''const assert=require('node:assert/strict');
let data={mug_pnp_comparison:true},activeCase={name:'mug',label:'White mug'};
let resultRevision='old-revision',loadNumber=0,ready=true,frameIndex=0,videoBlobUrl=null;
let paintCount=12,lastWall=100,drawCount=0,maskLoadCount=0,mappingLoadCount=0;
let pending=[];
const mode={value:'mug_pnp_no_extrinsic_guess',options:[
 {dataset:{mugPnp:'true'},hidden:false,disabled:false}
]};
const view={width:1280,height:436,painted:true};
const detailView={width:320,height:436,painted:true};
const ctx={clearRect(){view.painted=false;},fillRect(){},fillText(){},drawImage(){}};
const detailContext={clearRect(){detailView.painted=false;},drawImage(){}};
detailView.getContext=()=>detailContext;
const frameStatus={dataset:{signature:'mug'},replaceChildren(){this.cleared=true;}};
const reviewMoments={replaceChildren(){this.cleared=true;}};
const document={
 querySelector(selector){if(selector==='#reviewMoments')return reviewMoments;return null;},
 querySelectorAll(){return [];}
};
const video={paused:true,pause(){this.paused=true;},addEventListener(){},load(){}};
const play={textContent:'Play'},status={textContent:''};
const maskRenderer={load(){maskLoadCount++;}},mappingRenderer={load(){mappingLoadCount++;}};
function previewAsset(url,kind,ticket){return new Promise((resolve,reject)=>pending.push({url,kind,ticket,resolve,reject}));}
function draw(){drawCount++;}
function start(){}
function reviewButtons(){}
const URL={revokeObjectURL(){},createObjectURL(){return 'blob:fixture';}};
const performance={now(){return 200;}};
'''
            exercise = r'''
(async()=>{
 const oldSelection=selectCase({name:'mug',label:'White mug'});
 const oldRequests=pending.slice();
 assert.equal(oldRequests.length,2,'the previous mug selection must be waiting on its assets');
 // The fixture begins this transition with the prior mug candidate still visible.
 view.painted=true;detailView.painted=true;ready=true;
 const newSelection=selectCase({name:'keyboard',label:'Keyboard'});
 const newRequests=pending.slice(2);
 assert.equal(newRequests.length,2,'the new object requests must remain pending');
 assert.equal(view.painted,false,'main comparison canvas should clear synchronously');
 assert.equal(detailView.painted,false,'phone detail canvas should clear synchronously');
 assert.equal(mode.value,'complete','mug-only mode should reset on object switch');
 assert.equal(mode.options[0].hidden,true,'mug-only option should hide on object switch');
 assert.equal(mode.options[0].disabled,true,'mug-only option should disable on object switch');
 assert.equal(ready,false,'playback should stay stopped while loading');
 assert.equal(video.paused,true,'the old mug playback should remain paused');
 newRequests.forEach(request=>request.reject(new Error('synthetic asset failure')));
 await newSelection;
 assert.equal(status.textContent,'Comparison could not load. Tap the object to retry.');
 oldRequests[0].resolve({frames:[]});
 oldRequests[1].resolve(new ArrayBuffer(0));
 await oldSelection;
 assert.equal(view.painted,false,'late mug data must not restore the main canvas');
 assert.equal(detailView.painted,false,'late mug data must not restore the phone canvas');
 assert.equal(drawCount,0,'neither failed nor stale selections may draw');
 assert.equal(maskLoadCount,0,'a stale response must not reload the old mesh');
 assert.equal(mappingLoadCount,0,'a stale response must not reload the old mapping');
 assert.equal(ready,false,'a stale response must not resume the old case');
 assert.equal(activeCase.name,'keyboard','the current case must remain selected');
 assert.equal(mode.value,'complete','late response must not restore a mug-only mode');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
            result = subprocess.run(
                [node, '-e', harness+generated_functions+'\n'+exercise],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
