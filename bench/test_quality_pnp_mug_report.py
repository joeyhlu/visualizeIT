"""Pure CPU checks for the full-window mug PnP evaluator."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_pnp_mug_report as report
from .quality_contract import project


SETTINGS_BASE = {
    **report.EXPECTED_SETTINGS,
    'pnp_use_extrinsic_guess': True,
    'iterations': 5,
    'recovery_top_k': 5,
}
RUNTIME = {
    'device': 'cuda', 'torch': '2.6.test', 'cuda': 'test', 'gpu': 'synthetic',
    'precision': 'float32', 'batch_size': 1, 'iterations': 5, 'crop_size': [280, 280],
    'deterministic_algorithms': True, 'cudnn_deterministic': True,
    'cublas_workspace': ':4096:8', 'math_threads': {
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    },
    'torch_cpu_threads': 1,
}
PROVENANCE = {
    'input_manifest_sha256': 'input-manifest',
    'adapter_sha256': {'quality_runner.py': 'adapter'},
    'source_revisions': {'gotrack': 'revision'},
    'submodule_revisions': {'dinov2': 'submodule'},
    'checkpoint_sha256': {'gotrack_checkpoint.pt': 'checkpoint'},
    'foundpose_bank_sha256': 'bank',
}
EXPERIMENT = {
    'object': 'mug', 'device': 'cuda', 'setup_frame_id': 826,
    'source_files': {'bench/quality_runner.py': 'source-hash'},
    'frozen_files': {'input.mp4': 'video-hash', 'gotrack.pt': 'checkpoint-hash'},
    'inference_provenance': PROVENANCE,
    'mask_sha256_by_frame': {str(fid): f'{fid:064x}' for fid in (826, *report.MUG_FRAME_IDS)},
}


def _pose(rotation=None):
    value = np.eye(4)
    if rotation is not None:
        value[:3, :3] = rotation
    value[2, 3] = 2.0
    return value.tolist()


def _rotation_180():
    return np.diag([-1.0, -1.0, 1.0])


def _frame(fid, failed=False, recovering=False, rotated=False, missing_latency=False):
    state = 'recovering' if recovering else 'lost' if failed else 'tracking'
    pose = None if state != 'tracking' else _pose(_rotation_180() if rotated else None)
    row = {
        'frameId': fid,
        'mask_state': 'available',
        'pose_state': state,
        'render_state': 'visible' if state == 'tracking' else 'suppressed',
        'failure_reason': None if state == 'tracking' else ('awaiting_second_validated_pose' if recovering else 'no_valid_current_image_pose'),
        'cameraFromObject': pose,
        'mask_path': f'masks/{fid}.png',
        'mask_sha256': EXPERIMENT['mask_sha256_by_frame'][str(fid)],
        'timings_ms': {} if missing_latency else {'pose_total': 4.0},
    }
    return row


def _run(branch, length, failures=None, recovering=None, rotated=None):
    failures = set(failures or ())
    recovering = set(recovering or ())
    rotated = set(rotated or ())
    ids = list(report.MUG_FRAME_IDS[:length])
    settings = dict(SETTINGS_BASE, pnp_use_extrinsic_guess=branch == 'control')
    return {
        'object': 'mug', 'mode': 'complete', 'status': 'complete',
        'complete': length == 240,
        'frame_ids': ids,
        'frames': [_frame(fid, failed=fid in failures, recovering=fid in recovering,
                          rotated=fid in rotated, missing_latency=fid == 908 and branch == 'control') for fid in ids],
        'initialization': {'frameId': 826, 'cameraFromObject': None, 'mask_state': 'available'},
        'tracking_settings': settings,
        'provenance': copy.deepcopy(PROVENANCE),
        'runtime': copy.deepcopy(RUNTIME),
        'automatic': True, 'diagnostic_control': False,
        'independent_accuracy_scored': False,
        'stress_test': None,
    }


def _runs(control_failures=(900, 901), candidate_failures=(902,), candidate_recovering=(904,), candidate_rotated=(827,)):
    return {
        'control': {n: _run('control', n, failures=control_failures) for n in report.PREFIX_LENGTHS},
        'candidate': {n: _run('candidate', n, failures=candidate_failures,
                              recovering=candidate_recovering, rotated=candidate_rotated)
                      for n in report.PREFIX_LENGTHS},
    }


def _original():
    k = np.array([[100.0, 0, 640.0], [0, 100.0, 512.0], [0, 0, 1.0]])
    return {
        'name': 'mug', 'cameraCalibration': k.reshape(-1).tolist(),
        'nativeResolution': [1280, 1024],
        'referenceFrames': [
            {'frameId': fid, 'cameraFromObject': _pose()}
            for fid in report.MUG_FRAME_IDS
        ],
    }


def _mesh():
    rng = np.random.default_rng(7)
    positions = np.column_stack((rng.uniform(-0.15, 0.15, 128), rng.uniform(-0.15, 0.15, 128), np.zeros(128)))
    return SimpleNamespace(name='mug', positions=positions, triangles=np.zeros((1, 3), dtype=int))


def _fake_surface_visibility(mesh, points, normals, pose, k, resolution):
    expected, _ = project(points, pose, k)
    return np.ones(len(points), dtype=bool), expected


def _fake_legacy_truth(mesh, vertex_ids, pose, k, width, height):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = pose
    projected, _ = project(mesh.positions[vertex_ids], matrix, k)
    return np.arange(len(vertex_ids)), projected


def _trace_records(length):
    return [
        {'stage': f'refine/{fid}/0', 'arrays': {
            'image': {'shape': [2, 2, 3], 'dtype': 'uint8', 'sha256': f'{fid:064x}', 'mean': 0.0},
        }}
        for fid in report.MUG_FRAME_IDS[:length]
    ]


def _trace_payload(records):
    return ''.join(json.dumps(record, allow_nan=False) + '\n' for record in records).encode('utf-8')


def _trace_data():
    traces = {branch: {} for branch in report.BRANCHES}
    attestations = {}
    for stage_id in report.STAGE_IDS:
        branch, length_text = stage_id.rsplit('-', 1)
        length = int(length_text)
        records = _trace_records(length)
        payload = _trace_payload(records)
        traces[branch][length] = records
        attestations[stage_id] = {
            'path': f'{stage_id}.trace.jsonl',
            'sha256': hashlib.sha256(payload).hexdigest(),
            'record_count': len(records),
        }
    return traces, attestations


def _stage_evidence(runs, experiment=EXPERIMENT):
    stages = []
    for stage_id in report.STAGE_IDS:
        branch, length_text = stage_id.rsplit('-', 1)
        length = int(length_text)
        result = runs[branch][length]
        records = _trace_records(length)
        payload = _trace_payload(records)
        stages.append({
            'stage_id': stage_id, 'status': 'succeeded', 'exit_code': 0,
            'terminal_exit_recorded': True,
            'output': f'{stage_id}.json', 'output_sha256': 'e' * 64,
            'trace': f'{stage_id}.trace.jsonl',
            'trace_sha256': hashlib.sha256(payload).hexdigest(),
            'trace_record_count': len(records),
            'requested_frame_ids': list(report.MUG_FRAME_IDS[:length]),
            'expected_complete': length == 240,
            'pnp_use_extrinsic_guess': branch == 'control',
            'source_files': copy.deepcopy(experiment['source_files']),
            'frozen_files': copy.deepcopy(experiment['frozen_files']),
            'inference_provenance': copy.deepcopy(experiment['inference_provenance']),
            'runtime': copy.deepcopy(result['runtime']),
            'tracking_settings': copy.deepcopy(result['tracking_settings']),
            'result_frame_ids': copy.deepcopy(result['frame_ids']),
            'result_complete': result['complete'],
        })
    return {'status': 'complete', 'attempted_stage_ids': list(report.STAGE_IDS), 'stages': stages}


class SourceSnapshotTests(unittest.TestCase):
    def test_long_source_path_maps_to_short_verified_artifact_and_preserves_origin_hash(self):
        with tempfile.TemporaryDirectory(prefix='visualizeit-source-snapshot-') as temporary:
            base = Path(temporary)
            source_root = base / 'srcroot'
            experiment_root = base / 'run'
            source_root.mkdir()
            experiment_root.mkdir()
            filename = 'asset.bin'
            dirname_length = 250 - len(str(source_root)) - 2 - len(filename)
            self.assertGreater(dirname_length, 0)
            relative = f"{'d' * dirname_length}/{filename}"
            source = source_root / ('d' * dirname_length) / filename
            source.parent.mkdir(parents=True)
            source.write_bytes(b'cached inference source bytes\n')
            old_mirror_target = experiment_root / 'source-snapshot' / Path(*report.PurePosixPath(relative).parts)
            self.assertLess(len(str(source)), 260)
            self.assertGreater(len(str(old_mirror_target)), 260)

            original_hashes = {relative: report.digest(source)}
            artifacts = report.write_source_snapshot(
                original_hashes, experiment_root / 'source-snapshot', source_root)
            artifact = artifacts[relative]
            target = experiment_root / artifact['artifact_path']
            self.assertEqual(set(artifacts), set(original_hashes))
            self.assertEqual(artifact['sha256'], original_hashes[relative])
            self.assertEqual(report.digest(source), original_hashes[relative])
            self.assertEqual(report.digest(target), original_hashes[relative])
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertLess(len(str(target)), 260)

    def test_identical_sources_share_verified_content_address_and_keep_both_origins(self):
        with tempfile.TemporaryDirectory(prefix='visualizeit-source-dedupe-') as temporary:
            base = Path(temporary)
            source_root = base / 'sources'
            source_root.mkdir()
            (source_root / 'one.py').write_bytes(b'identical source bytes')
            (source_root / 'two.py').write_bytes(b'identical source bytes')
            expected = report.digest(source_root / 'one.py')
            artifacts = report.write_source_snapshot(
                {'one.py': expected, 'two.py': expected}, base / 'experiment/source-snapshot', source_root)
            self.assertEqual(artifacts['one.py']['artifact_path'], artifacts['two.py']['artifact_path'])
            self.assertEqual(artifacts['one.py']['sha256'], expected)
            self.assertEqual(artifacts['two.py']['sha256'], expected)
            self.assertEqual(len(list((base / 'experiment/source-snapshot/files').iterdir())), 1)

    def test_source_hash_mismatch_and_content_address_collision_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='visualizeit-source-integrity-') as temporary:
            base = Path(temporary)
            source_root = base / 'sources'
            source_root.mkdir()
            (source_root / 'one.py').write_bytes(b'first source bytes')
            (source_root / 'two.py').write_bytes(b'different source bytes')
            with self.assertRaisesRegex(ValueError, 'Source changed before snapshot copy'):
                report.write_source_snapshot(
                    {'one.py': '0' * 64}, base / 'bad-hash/source-snapshot', source_root)

            collision_hash = 'a' * 64
            with patch.object(report, 'digest', side_effect=lambda _path: collision_hash):
                with self.assertRaisesRegex(ValueError, 'Content-addressed source snapshot collision'):
                    report.write_source_snapshot(
                        {'one.py': collision_hash, 'two.py': collision_hash},
                        base / 'collision/source-snapshot', source_root)


class MugPnpReportTests(unittest.TestCase):
    def _report(self, runs=None, original=None, evidence=None, traces=None):
        runs = runs or _runs()
        default_traces, trace_attestations = _trace_data()
        return report.build_report(
            runs, original or _original(), _mesh(), np.zeros((128, 3)), np.zeros((128, 3)),
            copy.deepcopy(EXPERIMENT), evidence or _stage_evidence(runs), traces=traces or default_traces,
            trace_attestations=trace_attestations,
            expected_output_hashes={stage: 'e' * 64 for stage in report.STAGE_IDS},
        )

    def test_exact_completion_ids_and_public_pose_validation(self):
        prefix = _run('control', 30)
        self.assertIs(prefix['complete'], False)
        report.validate_result(prefix, report.MUG_FRAME_IDS[:30], False, 826)
        full = _run('candidate', 240)
        self.assertIs(full['complete'], True)
        report.validate_result(full, report.MUG_FRAME_IDS, True, 826)

        for key, value in (('complete', False), ('status', 'running'), ('status', 'failed')):
            bad = copy.deepcopy(full)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                report.validate_result(bad, report.MUG_FRAME_IDS, True, 826)

        duplicate = copy.deepcopy(prefix)
        duplicate['frames'][2]['frameId'] = duplicate['frames'][1]['frameId']
        with self.assertRaisesRegex(ValueError, 'exactly once and in order'):
            report.validate_result(duplicate, report.MUG_FRAME_IDS[:30], False, 826)

        rejected_pose = copy.deepcopy(prefix)
        rejected_pose['frames'][0]['pose_state'] = 'lost'
        rejected_pose['frames'][0]['render_state'] = 'suppressed'
        rejected_pose['frames'][0]['cameraFromObject'] = _pose()
        with self.assertRaisesRegex(ValueError, 'must suppress rendering and expose no pose'):
            report.validate_result(rejected_pose, report.MUG_FRAME_IDS[:30], False, 826)

    def test_asymmetric_failures_and_suppression_are_counted_over_all_240(self):
        runs = _runs(candidate_rotated=())
        with patch.object(report, 'visibility', side_effect=_fake_surface_visibility), \
             patch.object(report, 'evaluation_truth', side_effect=_fake_legacy_truth):
            result = self._report(runs)
        control = result['branch_accounting']['control']
        candidate = result['branch_accounting']['candidate']
        self.assertEqual((control['source_frames'], control['accepted_frames'], control['failures']), (240, 238, 2))
        self.assertEqual((candidate['source_frames'], candidate['accepted_frames'], candidate['failures']), (240, 238, 2))
        self.assertEqual(candidate['recovering_frames'], 1)
        self.assertEqual(candidate['latency_ms']['all_source_frames']['samples'], 240)
        self.assertEqual(candidate['latency_ms']['all_source_frames']['missing_source_frames'], 0)
        self.assertEqual(control['rendering_suppressed_frames'], 2)
        self.assertEqual(candidate['rendering_suppressed_frames'], 2)
        self.assertEqual(result['rendering_suppression_contribution']['candidate_only_suppressed_frame_ids'], [902, 904])
        self.assertEqual(result['common_accepted_frames'], 236)
        own = result['own_accepted_fixed128_area_metrics']['candidate']
        self.assertEqual(own['source_frames'], 240)
        self.assertEqual(own['missing_source_frames'], 2)
        self.assertEqual(own['reference_visible_samples'], 240 * 128)
        self.assertEqual(own['missing_point_samples'], 2 * 128)
        self.assertFalse(result['independent_accuracy_verified'])
        self.assertFalse(result['overall_gate_passed'])

    def test_only_declared_boolean_may_differ_between_phases(self):
        runs = _runs(candidate_rotated=())
        metadata = report._run_metadata_checks(runs, copy.deepcopy(EXPERIMENT))
        self.assertTrue(metadata['passed'], metadata['issues'])

        bad_pnp = copy.deepcopy(runs)
        bad_pnp['candidate'][30]['tracking_settings']['pnp_use_extrinsic_guess'] = True
        self.assertFalse(report._run_metadata_checks(bad_pnp, copy.deepcopy(EXPERIMENT))['passed'])

        bad_other_setting = copy.deepcopy(runs)
        bad_other_setting['candidate'][120]['tracking_settings']['recovery_top_k'] = 6
        self.assertFalse(report._run_metadata_checks(bad_other_setting, copy.deepcopy(EXPERIMENT))['passed'])

        bad_bank = copy.deepcopy(runs)
        bad_bank['candidate'][30]['provenance']['foundpose_bank_sha256'] = 'different'
        self.assertFalse(report._run_metadata_checks(bad_bank, copy.deepcopy(EXPERIMENT))['passed'])

    def test_prefix_pose_and_supplied_trace_comparisons(self):
        runs = _runs(candidate_rotated=())
        traces, _ = _trace_data()
        prefixes = report._prefix_checks(runs, traces)
        self.assertTrue(all(row['passed'] for row in prefixes.values()))

        changed = copy.deepcopy(runs)
        changed['candidate'][30]['frames'][0]['cameraFromObject'] = _pose(_rotation_180())
        self.assertFalse(report._prefix_checks(changed)['candidate_30_vs_120']['prefix_equal'])

        traces = {'control': {
            30: [{'stage': 'refine', 'arrays': {'input': 'short'}}],
            120: [{'stage': 'refine', 'arrays': {'input': 'long'}}],
        }}
        self.assertFalse(report._prefix_checks(runs, traces)['control_30_vs_120']['trace']['passed'])
        self.assertFalse(report._prefix_checks(runs, None)['control_30_vs_120']['trace']['passed'])

    def test_new_large_orientation_disagreement_is_reported_and_fails_gate(self):
        runs = _runs(candidate_rotated=(827,))
        with patch.object(report, 'visibility', side_effect=_fake_surface_visibility), \
             patch.object(report, 'evaluation_truth', side_effect=_fake_legacy_truth):
            result = self._report(runs)
        self.assertEqual(result['new_candidate_large_orientation_disagreements']['new_candidate_frame_ids'], [827])
        self.assertFalse(result['continuation_gate_checks']['no_new_90_degree_disagreement'])
        self.assertFalse(result['continuation_gate_passed'])

    def test_terminal_evidence_requires_ordered_zero_exit_and_frozen_hashes(self):
        runs = _runs(candidate_rotated=())
        evidence = _stage_evidence(runs)
        hashes = {stage: 'e' * 64 for stage in report.STAGE_IDS}
        traces, attestations = _trace_data()
        self.assertTrue(report._stage_lifecycle(evidence, hashes, EXPERIMENT, runs, attestations, traces)['passed'])

        for mutation in (
            lambda value: value['stages'][0].update(exit_code=1),
            lambda value: value['stages'][0].update(source_files={'bench/quality_runner.py': 'edited'}),
            lambda value: value.update(attempted_stage_ids=list(reversed(report.STAGE_IDS))),
        ):
            bad = copy.deepcopy(evidence)
            mutation(bad)
            self.assertFalse(report._stage_lifecycle(bad, hashes, EXPERIMENT, runs, attestations, traces)['passed'])

    def test_terminal_trace_loader_rejects_missing_empty_edited_and_truncated_files(self):
        stage_id = 'control-30'
        records = _trace_records(30)
        payload = _trace_payload(records)
        terminal = {
            'trace': f'{stage_id}.trace.jsonl',
            'trace_sha256': hashlib.sha256(payload).hexdigest(),
            'trace_record_count': len(records),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / terminal['trace']
            with self.assertRaises(FileNotFoundError):
                report._read_trace_file(path, stage_id, terminal)

            path.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'empty or truncated'):
                report._read_trace_file(path, stage_id, terminal)

            path.write_bytes(payload)
            loaded, attestation = report._read_trace_file(path, stage_id, terminal)
            self.assertEqual(len(loaded), 30)
            self.assertEqual(attestation['record_count'], 30)

            path.write_bytes(payload[:-1] + b' ' + b'\n')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                report._read_trace_file(path, stage_id, terminal)

            path.write_bytes(_trace_payload(records[:-1]))
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                report._read_trace_file(path, stage_id, terminal)

            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, 'empty or truncated'):
                report._read_trace_file(path, stage_id, terminal)

    def test_build_report_rejects_missing_trace_bundle(self):
        runs = _runs(candidate_rotated=())
        with self.assertRaisesRegex(ValueError, 'All six nonempty stage traces'):
            report.build_report(runs, _original(), _mesh(), np.zeros((128, 3)), np.zeros((128, 3)),
                                copy.deepcopy(EXPERIMENT), _stage_evidence(runs), traces=None)

    def test_secondary_continuation_criteria_keep_strict_p95_and_availability_rules(self):
        control = {
            'accounting': {'accepted_frames': 230},
            'area_metrics': {'median_720': 2.0, 'p95_720': 4.0},
        }
        candidate = {
            'accounting': {'accepted_frames': 230},
            'area_metrics': {'median_720': 1.9, 'p95_720': 3.9},
        }
        common_control = {'median_720': 2.0, 'p95_720': 4.0}
        common_candidate = {'median_720': 2.0, 'p95_720': 3.9}
        prefixes = {'control_30_vs_120': {'passed': True}, 'candidate_30_vs_120': {'passed': True}}
        metadata = {'passed': True}
        lifecycle = {'passed': True}
        orientation = {'new_candidate_frame_ids': []}
        checks = report._gate_checks(control, candidate, common_control, common_candidate,
                                     prefixes, metadata, lifecycle, True, orientation)
        self.assertTrue(all(checks.values()))
        lower_control = copy.deepcopy(control)
        higher_candidate = copy.deepcopy(candidate)
        lower_control['accounting']['accepted_frames'] = 210
        higher_candidate['accounting']['accepted_frames'] = 216
        eligible = report._gate_checks(lower_control, higher_candidate, common_control, common_candidate,
                                       prefixes, metadata, lifecycle, True, orientation)
        self.assertTrue(all(eligible.values()))

        cases = [
            ('candidate_accepted_at_least_control', lambda c, ca, cc, cn, p, o: ca['accounting'].update(accepted_frames=229)),
            ('candidate_accepted_at_least_216', lambda c, ca, cc, cn, p, o: ca['accounting'].update(accepted_frames=215)),
            ('common_area_p95_strictly_better', lambda c, ca, cc, cn, p, o: cn.update(p95_720=4.0)),
            ('own_area_median_not_worse', lambda c, ca, cc, cn, p, o: ca['area_metrics'].update(median_720=2.1)),
            ('own_area_p95_not_worse', lambda c, ca, cc, cn, p, o: ca['area_metrics'].update(p95_720=4.1)),
            ('no_new_90_degree_disagreement', lambda c, ca, cc, cn, p, o: o.update(new_candidate_frame_ids=[900])),
        ]
        for expected_false, mutate in cases:
            c, ca = copy.deepcopy(control), copy.deepcopy(candidate)
            cc, cn, p, o = copy.deepcopy(common_control), copy.deepcopy(common_candidate), copy.deepcopy(prefixes), copy.deepcopy(orientation)
            mutate(c, ca, cc, cn, p, o)
            result = report._gate_checks(c, ca, cc, cn, p, metadata, lifecycle, True, o)
            with self.subTest(expected_false=expected_false):
                self.assertFalse(result[expected_false])

    def test_mismatched_reference_or_mesh_object_is_rejected(self):
        runs = _runs(candidate_rotated=())
        with self.assertRaisesRegex(ValueError, 'must both identify the mug'):
            self._report(runs, original={**_original(), 'name': 'keyboard'})
        with self.assertRaisesRegex(ValueError, 'must both identify the mug'):
            report.build_report(runs, _original(), SimpleNamespace(name='keyboard', positions=np.zeros((128, 3))),
                                np.zeros((128, 3)), np.zeros((128, 3)), copy.deepcopy(EXPERIMENT),
                                _stage_evidence(runs))


if __name__ == '__main__':
    unittest.main()
