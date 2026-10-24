"""Manufactured geometry and input/label-boundary checks; no neural execution."""
from contextlib import ExitStack, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_adaptation_refinement_states as states
from .quality_adaptation_data import (CONVENTION, HOOKS, SIZE, canonical_hash,
    crop_rgb_hash, read_json, sha256)
from .quality_adaptation_prepare import crops_from_retained_capture, project_supervision
from .quality_gotrack import GoTrackRefiner


def plan_fixture():
    def pin(path):
        return dict(path=path, bytes=1, sha256='a' * 64)
    rows = []
    for split, obj, seq, pitch in states.SEQUENCES:
        for frame in range(8):
            root = '/teamspace/studios/this_studio/data/' + seq
            rows.append(dict(id=f'{seq}:frame{frame}', object_id=obj, split=split,
                sequence_id=seq, capture_group='group-' + obj, frame_id=frame,
                timestamp_s=frame / 10., yaw0_deg=0, pitch_deg=pitch,
                input=pin(root + '/input.json'), video=pin(root + '/synthetic.mp4'), mesh=pin(root + '/object.glb'),
                mask_index=pin(root + '/results.json'), mask=pin(root + f'/{frame}.png'), bank=pin(root + '/bank.pt'),
                iterations=[states.TRAIN_ITERATIONS[frame]] if split == 'train' else [0, 1, 3]))
    return dict(schema=states.SCHEMA, runtime_contract=pin('runtime.json'), source_contract=pin('source.json'),
        rows=rows, limits=states.LIMITS.copy(), sealed_test=dict(id='sealed7', object_id='7', split='test',
            sequence_id='sealed7', capture_group='sealed7', status='unavailable', query_rgb_sha256=None,
            failure_reason='held_out', unavailable_provenance='metadata only; no TEST payload'))


def arrays_fixture(value=.1234567):
    k = np.array([[300., 0., 140.], [0., 300., 140.], [0., 0., 1.]])
    pose = np.eye(4); pose[2, 3] = 1000.
    return dict(query_rgb_crop=np.full((1, 3, SIZE, SIZE), value, np.float32),
        template_rgb=np.full((1, 3, SIZE, SIZE), .2345678, np.float32),
        rendered_depth_mm=np.full((1, SIZE, SIZE), 1000., np.float32),
        rendered_mask=np.ones((1, SIZE, SIZE), bool), crop_k=k,
        crop_from_orig=np.eye(4), native_k=k,
        current_crop_camera_from_object_mm=pose,
        full_flow_crop_px=np.zeros((SIZE, SIZE, 2), np.float32), full_confidence=np.ones((SIZE, SIZE), np.float32))


class RefStateTests(unittest.TestCase):
    def setUp(self):
        temp_root = states.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = plan_fixture()

    def bypass_cloud(self, stack):
        stack.enter_context(patch.object(states, 'cloud_guard'))
        stack.enter_context(patch.object(states, '_stage_lock', side_effect=nullcontext))

    def sink(self, physical=None, rows=None, checks=None):
        physical = physical or self.plan['rows'][16]
        rows = rows or states.state_rows(self.plan)
        checks = checks if checks is not None else {i: dict(integer_centers=SIZE**2,
            max_native_pixel_difference=0.) for i in physical['iterations']}
        return states.RefStateCapture(self.root, physical, rows, checks, arrays_fixture()['native_k'])

    def packet(self, sink, iteration, kind='retained', arrays=None):
        return dict(kind=kind, context=dict(physical_id=sink.physical['id']),
            metadata=dict(frame_id=sink.physical['frame_id'], iteration=iteration, fit_state='ransac_unavailable'),
            arrays=arrays_fixture() if arrays is None else arrays)

    def test_exact_schedule_is_24_physical_frames_40_states(self):
        states.validate_plan(self.plan)
        rows = states.state_rows(self.plan)
        # The prospective schedule fixes 16 TRAIN rows (8/4/4) and 24 used VAL.
        self.assertEqual(len(rows), 40)
        self.assertEqual([sum(r['split'] == 'train' and r['iteration'] == i for r in rows) for i in (0, 1, 3)], [8, 4, 4])
        self.assertEqual(sum(r['split'] == 'val' for r in rows), 24)

    def test_schedule_annotation_test_and_extra_fields_rejected(self):
        changes = [lambda p: p['rows'][0].update(iterations=[3]),
            lambda p: p['rows'][0].update(frame_id=9), lambda p: p['rows'][0].update(generation={}),
            lambda p: p['sealed_test'].update(payload='heldout.npz'),
            lambda p: p['rows'][0]['input'].update(path='/teamspace/studios/this_studio/generation/input.json')]
        for change in changes:
            with self.subTest(change=change):
                plan = copy.deepcopy(self.plan); change(plan)
                with self.assertRaises(ValueError): states.validate_plan(plan)

    def test_each_selected_iteration_retains_exact_float_rgb_and_geometry(self):
        sink = self.sink()
        for iteration in (0, 1, 3):
            arrays = arrays_fixture(.1234567 + iteration / 100.)
            sink(self.packet(sink, iteration, arrays=arrays))
            row = next(r for r in sink.rows if r['id'] == sink.physical['id'] + f':iteration{iteration}')
            with np.load(self.root / row['crops']['path']) as archive:
                # This manufactured non-byte-valued RGB must never be quantized.
                self.assertEqual(archive['query_rgb'].dtype, np.float32)
                np.testing.assert_array_equal(archive['query_rgb'][0, 0], arrays['query_rgb_crop'][0, :, 0, 0])
            with np.load(self.root / row['capture']['path']) as archive:
                self.assertEqual(set(archive.files), states.GEOMETRY_KEYS)
                np.testing.assert_array_equal(archive['rendered_depth_mm'], arrays['rendered_depth_mm'][0])

    def test_each_selected_iteration_requires_own_sampling_check(self):
        sink = self.sink(checks={0: dict(integer_centers=SIZE**2, max_native_pixel_difference=0.)})
        sink(self.packet(sink, 0))
        with self.assertRaisesRegex(ValueError, 'own returned-camera'):
            sink(self.packet(sink, 1))

    def test_retained_crop_survives_pnp_failure_later_iteration_is_not_substituted(self):
        sink = self.sink()
        sink(self.packet(sink, 0)); sink(self.packet(sink, 0, 'result', {}))
        sink.finish()
        rows = [r for r in sink.rows if r['id'].startswith(sink.physical['id'] + ':iteration')]
        self.assertEqual([r['status'] for r in rows], ['available', 'unavailable', 'unavailable'])
        self.assertEqual(rows[0]['pnp_state'], 'ransac_unavailable')
        self.assertIn('unrun_iteration', rows[1]['failure_reason'])

    def test_foreign_frame_context_native_camera_and_duplicate_packet_rejected(self):
        for field in ('frame', 'context', 'camera', 'duplicate'):
            with self.subTest(field=field):
                folder = self.root / field; folder.mkdir()
                sink = self.sink(); sink.output = folder
                packet = self.packet(sink, 0)
                if field == 'frame': packet['metadata']['frame_id'] = 7
                elif field == 'context': packet['context']['physical_id'] = 'foreign'
                elif field == 'camera': packet['arrays']['native_k'][0, 0] += 1
                else: sink(packet)
                with self.assertRaises(ValueError): sink(packet)

    def test_manufactured_rotated_camera_mm_to_m_projection(self):
        arrays = arrays_fixture()
        # A +90-degree camera rotation maps native object +X to crop +Y.
        c = np.eye(4); c[:2, :2] = [[0., -1.], [1., 0.]]
        arrays['crop_from_orig'] = c
        native_m = np.eye(4); native_m[2, 3] = 1.
        pose_mm = native_m.copy(); pose_mm[:3, 3] *= 1000.
        arrays['current_crop_camera_from_object_mm'] = c @ pose_mm
        crops = crops_from_retained_capture(arrays, preserve_float_rgb=True)
        np.testing.assert_allclose(crops['template_xyz_m'][140, 170], [0., -.1, 0.], atol=1e-7)
        target = project_supervision(template_xyz_m=crops['template_xyz_m'], template_mask=crops['template_mask'],
            query_K=crops['query_K'], query_T_cam_from_object=c @ native_m,
            observed_depth_m=np.ones((SIZE, SIZE), np.float32), observed_object_mask=np.ones((SIZE, SIZE), bool),
            depth_tolerance_m=.002)
        # Analytic pinhole projection of the same plane is identity at integer centers.
        np.testing.assert_allclose(target['flow'], 0., atol=1e-5)
        self.assertTrue(target['visible'][140, 170])

    def test_capability_requires_authentication_exact_sink_and_profile(self):
        sink = self.sink()
        raw = states._encoded(self.plan); digest = hashlib.sha256(raw).hexdigest()
        with patch.object(states, 'verify_source_contract'):
            verified = states._VerifiedPreflight(states._ISSUER, raw)
            cap = states._issue(raw, digest, sink, verified)
            cap.validate(sink, states.inference.PROFILE)
            GoTrackRefiner(None, None, 17, 'cpu', chronological_capture_callback=sink,
                inference_geometry=states.inference.PROFILE, inference_observer_capability=cap)
            for other_sink, geometry in ((self.sink(), states.inference.PROFILE), (sink, 'legacy')):
                with self.assertRaises(ValueError): cap.validate(other_sink, geometry)
            forged = states.RefStateObserverCapability(object(), raw, digest, sink)
            with self.assertRaises(ValueError): forged.validate(sink, states.inference.PROFILE)
            with self.assertRaises(ValueError): states._issue(raw, digest, sink, object())
        with self.assertRaises(ValueError):
            GoTrackRefiner(None, None, 17, 'cpu', inference_geometry=states.inference.PROFILE,
                chronological_capture_callback=sink, inference_observer_capability=object())

    def test_source_contract_checks_exact_closure_and_source_mutation(self):
        for name in ('a.py', 'b.py'):
            (self.root / name).write_text(name)
        doc = dict(schema=states.SOURCE_SCHEMA, historical_training_manifest_sha256=states.HISTORICAL_MANIFEST_SHA256,
            sources=[dict(path=name, sha256=sha256(self.root / name)) for name in ('a.py', 'b.py')])
        (self.root / 'source.json').write_text(json.dumps(doc))
        runtime = dict(additional_runtime_files={'a.py': sha256(self.root / 'a.py')})
        (self.root / 'runtime.json').write_text(json.dumps(runtime))
        plan = copy.deepcopy(self.plan)
        for name, key in (('source.json', 'source_contract'), ('runtime.json', 'runtime_contract')):
            path = self.root / name
            plan[key] = dict(path=name, bytes=path.stat().st_size, sha256=sha256(path))
        with patch.object(states, 'ROOT', self.root), patch.object(states, 'EXTRA_SOURCES', {'b.py'}), \
             patch.object(states.inference, 'EXPECTED_RUNTIME_FILES', {'a.py'}), \
             patch.object(states.inference, '_validate_contract'), patch.object(states.inference, 'verify_source_compatibility'):
            states.verify_source_contract(plan)
            (self.root / 'b.py').write_text('mutated')
            with self.assertRaisesRegex(ValueError, 'source mismatch'): states.verify_source_contract(plan)

    def test_input_context_never_opens_generation_or_labels(self):
        row = self.plan['rows'][0]
        accepted = dict(sequence_id=row['sequence_id'], physical_object_id='hot3d:obj_000017', native_resolution=[640, 480],
            mesh_path='object.glb', mesh_sha256='a' * 64, video_path='synthetic.mp4', video_sha256='a' * 64,
            input_sha256='a' * 64, generation_receipt_path='../generation.json', generation_receipt_sha256='b' * 64)
        runtime = dict(accepted_inputs=[accepted])
        manifest = dict(schema_version=2, provenance_kind='synthetic', units='metres', asset='object.glb', video='synthetic.mp4',
            physical_object_id='hot3d:obj_000017', native_resolution=[640, 480],
            source_hashes={'object.glb': 'a' * 64, 'synthetic.mp4': 'a' * 64},
            intrinsics=[[576., 0., 319.5], [0., 576., 239.5], [0., 0., 1.]], _timing={'clock_mode': 'capture_seconds_v1'})
        calls = []
        def read(path):
            calls.append(str(path))
            if str(path) != 'runtime.json': raise AssertionError('annotation file opened before seal')
            return runtime
        with patch.object(states, 'pinned_file', return_value=Path('runtime.json')), patch.object(states, 'read_json', side_effect=read), \
             patch.object(states.inference, '_artifact_for_contract', return_value={}):
            context = states._input_context(self.plan, row, manifest)
        # Current serving calibration applies its fixed -0.5-pixel principal correction.
        self.assertEqual(context.effective_intrinsics[0, 2], 319.)
        self.assertEqual(calls, ['runtime.json'])
        self.assertNotIn('generation_receipt_path', context.accepted_input)

    def test_foreign_input_pins_rejected_before_any_data_or_manifest_read(self):
        entries = [dict(sequence_id=seq, input_sha256='a' * 64,
            video_sha256='a' * 64, mesh_sha256='a' * 64) for _, _, seq, _ in states.SEQUENCES]
        for field in ('input', 'video', 'mesh'):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                # Use the final row to require admission of ALL rows first.
                plan['rows'][-1][field]['sha256'] = 'b' * 64
                def source_only(pin, *, source=False):
                    if not source or pin != plan['runtime_contract']:
                        raise AssertionError('data payload opened before allowlist admission')
                    return Path('runtime.json')
                with patch.object(states, 'verify_source_contract'), \
                     patch.object(states, 'pinned_file', side_effect=source_only), \
                     patch.object(states, 'read_json', return_value=dict(accepted_inputs=entries)), \
                     patch('bench.quality_runner.read_input') as reader:
                    with self.assertRaisesRegex(ValueError, 'pins differ'):
                        states._preflight(plan)
                    reader.assert_not_called()

    def test_initial_all_failure_inventory_precedes_preflight_and_torch(self):
        path = self.root / 'plan-input.json'; path.write_text(json.dumps(self.plan))
        output = self.root / 'capture'
        def failed_preflight(plan):
            inventory = read_json(output / 'capture-index.json')['rows']
            self.assertEqual(len(inventory), 40)
            self.assertTrue(all(r['status'] == 'unavailable' for r in inventory))
            raise ValueError('manufactured preflight failure')
        with ExitStack() as stack:
            self.bypass_cloud(stack)
            stack.enter_context(patch.object(states, '_disk_preflight'))
            stack.enter_context(patch.object(states, '_preflight', side_effect=failed_preflight))
            with self.assertRaisesRegex(RuntimeError, 'full inventory'): states.capture(path, sha256(path), output, cloud_execution=True)
        receipt = read_json(output / 'capture-receipt.json')
        self.assertEqual(len(receipt['rows']), 40)
        self.assertEqual(receipt['test_payload_reads'], 0)
        self.assertNotIn('torch', sys.modules)

    def make_seal_fixture(self):
        capture = self.root / 'capture'; capture.mkdir()
        features = self.root / 'features'; features.mkdir()
        (capture / 'plan.json').write_text(json.dumps(self.plan))
        rows = states.state_rows(self.plan)
        for ordinal, physical in ((0, self.plan['rows'][0]), (16, self.plan['rows'][16])):
            sink = states.RefStateCapture(capture, physical, rows,
                {0: dict(integer_centers=SIZE**2, max_native_pixel_difference=0.)}, arrays_fixture()['native_k'])
            sink(self.packet(sink, 0, arrays=arrays_fixture(.1234567 + ordinal / 100.)))
            sink.finish()
        (capture / 'capture-index.json').write_text(json.dumps(dict(rows=rows, test_payload_reads=0)))
        (capture / 'capture-receipt.json').write_text(json.dumps(dict(rows=rows, plan_sha256=sha256(capture / 'plan.json'),
            test_payload_reads=0, annotation_inputs=[], started_unix_s=time.time())))
        crops = dict(schema='correspondence_input_crops_v1', conventions=CONVENTION,
            rows=[dict(id=r['id'], crops=r['crops']) for r in rows if r['status'] == 'available'])
        (capture / 'crops.json').write_text(json.dumps(crops))
        lock = read_json(states.ROOT / 'bench/runtime/adaptation-sources.json')
        recipe = dict(schema='frozen_gotrack_decoder_recipe_v1', source_lock_sha256=states.HISTORICAL_LOCK_SHA256,
            source_files_sha256=canonical_hash(lock['gotrack']['files']), upstream_revision=lock['gotrack']['revision'],
            producer_sha256=lock['gotrack']['files']['bench/quality_adaptation_prepare.py'], hooks=list(HOOKS),
            annotation_inputs=[], crop_manifest_sha256=canonical_hash(crops), rgb_encoding='uint8_div255_or_exact_float32_0_1_v1')
        index = []
        for ordinal, row in enumerate(rows):
            if row['status'] != 'available':
                index.append(dict(id=row['id'], status='unavailable', failure_reason='unrun'))
                continue
            with np.load(capture / row['crops']['path']) as crop:
                values = {k: crop[k] for k in ('template_mask', 'template_xyz_m', 'query_K')}
            values.update({f'tokens_{hook}': np.full((400, 768), ordinal / 100., np.float32) for hook in HOOKS})
            values.update(baseline_flow=np.zeros((2, SIZE, SIZE), np.float32), baseline_confidence=np.ones((SIZE, SIZE), np.float32))
            artifact = states._save_npz(features, f'features-{ordinal:02d}.npz', values)
            index.append(dict(id=row['id'], status='available', crops=row['crops'], query_rgb_sha256=row['query_rgb_sha256'], features=artifact))
        (features / 'recipe.json').write_text(json.dumps(recipe))
        (features / 'index.json').write_text(json.dumps(dict(recipe_sha256=canonical_hash(recipe), rows=index)))
        return capture, features

    def test_seal_authenticates_real_crop_depth_geometry_feature_and_parity(self):
        capture, features = self.make_seal_fixture()
        with patch.object(states, 'verify_source_contract'):
            material = states._seal_material(capture, features)
            self.assertEqual(material['available'], 2)
            self.assertEqual(len(material['identities']), 40)
            (capture / 'prelabel-seal.json').write_text(json.dumps(material))
            self.assertEqual(states.verify_prelabel_seal(capture, features), material)

    def test_seal_mutation_refuses_before_label_plan_open(self):
        capture, features = self.make_seal_fixture()
        with patch.object(states, 'verify_source_contract'):
            (capture / 'prelabel-seal.json').write_text(json.dumps(states._seal_material(capture, features)))
            with (features / 'index.json').open('a') as stream: stream.write(' ')
            with ExitStack() as stack:
                self.bypass_cloud(stack)
                label_read = stack.enter_context(patch.object(states, '_read_label_plan', side_effect=AssertionError('label opened')))
                with self.assertRaisesRegex(ValueError, 'prelabel seal changed'):
                    states.labels(capture, features, self.root / 'generation.json', 'a' * 64, self.root, self.root, cloud_execution=True)
                label_read.assert_not_called()

    def test_foreign_feature_archive_and_depth_pose_mutation_rejected(self):
        capture, features = self.make_seal_fixture()
        with patch.object(states, 'verify_source_contract'):
            index = read_json(features / 'index.json')
            index['rows'][0]['crops'] = index['rows'][16]['crops']
            (features / 'index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'crop receipt'): states._seal_material(capture, features)

    def test_available_capture_failed_extraction_remains_unavailable_without_substitution(self):
        capture, features = self.make_seal_fixture()
        index = read_json(features / 'index.json')
        index['rows'][0] = dict(id=index['rows'][0]['id'], status='unavailable', failure_reason='extraction_failure')
        (features / 'index.json').write_text(json.dumps(index))
        with patch.object(states, 'verify_source_contract'):
            seal = states._seal_material(capture, features)
        self.assertEqual(seal['available'], 1)
        self.assertEqual(len(seal['identities']), 40)

    def test_seal_reconstructs_actual_depth_geometry_after_descriptor_rebinding(self):
        capture, features = self.make_seal_fixture()
        index = read_json(capture / 'capture-index.json')
        row = index['rows'][0]
        path = capture / row['capture']['path']
        with np.load(path) as archive: values = {k: archive[k] for k in archive.files}
        values['rendered_depth_mm'] += 10.
        np.savez(path, **values)
        row['capture'] = states._artifact(capture, path)
        (capture / 'capture-index.json').write_text(json.dumps(index))
        receipt = read_json(capture / 'capture-receipt.json'); receipt['rows'] = index['rows']
        (capture / 'capture-receipt.json').write_text(json.dumps(receipt))
        with patch.object(states, 'verify_source_contract'):
            with self.assertRaisesRegex(ValueError, 'retained depth/pose'): states._seal_material(capture, features)

    def test_seal_checks_actual_prediction_parity_after_feature_descriptor_rebinding(self):
        capture, features = self.make_seal_fixture()
        index = read_json(features / 'index.json')
        path = features / index['rows'][0]['features']['path']
        with np.load(path) as archive: values = {k: archive[k] for k in archive.files}
        values['baseline_flow'] += .01
        np.savez(path, **values)
        index['rows'][0]['features'] = states._artifact(features, path)
        (features / 'index.json').write_text(json.dumps(index))
        with patch.object(states, 'verify_source_contract'):
            with self.assertRaisesRegex(ValueError, 'extractor outputs differ'): states._seal_material(capture, features)

    def test_extractor_bridge_pads_partial_inventory_to_all_40_states(self):
        capture, _ = self.make_seal_fixture()
        features = self.root / 'new-extraction'
        def historical(root, manifest, module, args, **kwargs):
            self.assertEqual(module, 'bench.quality_adaptation_prepare')
            self.assertIn('--max-rows', args)
            features.mkdir()
            (features / 'index.json').write_text(json.dumps(dict(rows=[dict(id=states.state_rows(self.plan)[0]['id'],
                status='unavailable', failure_reason='manufactured_extraction_failure')])))
            raise subprocess.TimeoutExpired('extractor', 1)
        with ExitStack() as stack:
            self.bypass_cloud(stack)
            stack.enter_context(patch.object(states, 'verify_source_contract'))
            stack.enter_context(patch.object(states, '_run_historical', side_effect=historical))
            with self.assertRaises(subprocess.TimeoutExpired):
                states.extract_stage(capture, features, self.root / 'source18', self.root, cloud_execution=True)
        index = read_json(features / 'index.json')
        self.assertEqual(len(index['rows']), 40)
        self.assertEqual(index['rows'][0]['failure_reason'], 'manufactured_extraction_failure')

    def test_historical_bridge_executes_module_in_separate_root_and_verifies_twice(self):
        with patch.object(states, 'verify_training_root') as verify, patch.object(states.subprocess, 'run',
                return_value=types.SimpleNamespace(returncode=0)) as run:
            states._run_historical(self.root, self.root / 'manifest.json', 'bench.quality_adaptation_prepare',
                ['--manifest', self.root / 'crops.json'], timeout=7)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(run.call_args.kwargs['cwd'], self.root.resolve())
        self.assertEqual(run.call_args.args[0][1:3], ['-m', 'bench.quality_adaptation_prepare'])
        self.assertNotIn('PYTHONPATH', run.call_args.kwargs['env'])

    def test_historical_bridge_rejects_lock_drift_timeout_and_accepts_incomplete_train(self):
        with self.assertRaises(ValueError): states.verify_training_root(states.ROOT, self.root / 'absent')
        with patch.object(states, 'verify_training_root'), patch.object(states.subprocess, 'run', side_effect=subprocess.TimeoutExpired('mock', 1)):
            with self.assertRaises(subprocess.TimeoutExpired): states._run_historical(self.root, self.root, 'module', [], timeout=1)
        with patch.object(states, 'verify_training_root'), patch.object(states.subprocess, 'run', return_value=types.SimpleNamespace(returncode=3, args=['module'])):
            self.assertEqual(states._run_historical(self.root, self.root, 'module', [], timeout=1, accepted_exit_codes=(0, 3)), 3)

    def test_shared_deadline_and_aggregate_disk_budget_do_not_reset(self):
        (self.root / 'capture-receipt.json').write_text(json.dumps(dict(started_unix_s=time.time() - 3601)))
        with self.assertRaises(TimeoutError): states._remaining(self.root)
        (self.root / 'capture-receipt.json').write_text(json.dumps(dict(started_unix_s=time.time())))
        with patch.object(states, 'tree_bytes', return_value=2 * 1024**3 + 1):
            with self.assertRaisesRegex(ValueError, 'private artifact'): states._remaining(self.root)

    def test_exclusive_stage_lock_refuses_overlap_and_cleans_on_failure(self):
        cache = self.root / '.cache/model-quality'; cache.mkdir(parents=True)
        with patch.object(states, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'manufactured'):
                with states._stage_lock():
                    with self.assertRaises(FileExistsError):
                        with states._stage_lock(): pass
                    raise ValueError('manufactured')
        self.assertFalse((cache / '.refinement-state-stage.lock').exists())

    def test_cloud_guard_precedes_all_worker_and_supervisor_heavy_work(self):
        with patch.object(states, 'cloud_guard', side_effect=RuntimeError('no cloud')), patch.object(states, '_stage_lock') as lock:
            with self.assertRaisesRegex(RuntimeError, 'no cloud'): states.capture(None, None, None)
            lock.assert_not_called()

    def test_selected_sampling_map_wrapper_checks_every_prescribed_iteration_and_restores(self):
        k = arrays_fixture()['crop_k']
        compute = lambda **kwargs: ({}, [object()], [np.eye(4)])
        utils = types.ModuleType('utils')
        utils.data_util = types.SimpleNamespace(compute_gotrack_inputs_from_init_poses=compute)
        utils.misc = types.SimpleNamespace(get_intrinsic_matrix=lambda camera: k)
        yy, xx = np.mgrid[:SIZE, :SIZE]; grid = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(float)
        with patch.dict(sys.modules, {'utils': utils}), \
             patch('bench.quality_gotrack._capture_actual_sampling_map', return_value=(None, None, None, grid)) as actual, \
             patch('bench.quality_gotrack._capture_native_uv', return_value=grid):
            with states.observe_selected_crops([0, 1, 3]) as checks:
                for _ in range(5): utils.data_util.compute_gotrack_inputs_from_init_poses(input_cameras=[object()])
                self.assertEqual(set(checks), {0, 1, 3})
            self.assertEqual(actual.call_count, 3)
            self.assertIs(utils.data_util.compute_gotrack_inputs_from_init_poses, compute)

    def test_full_controller_refines_each_of_24_physical_frames_once_and_closes_renderers(self):
        plan_path = self.root / 'input-plan.json'; plan_path.write_text(json.dumps(self.plan))
        output = self.root / 'captured'
        cameras, refined, closed, network_released = [], [], [], []
        k = arrays_fixture()['native_k']
        inputs = {r['id']: (self.root, dict(asset='object.glb', object_id=17,
            _timing={'time_policy': None}), types.SimpleNamespace(effective_intrinsics=k)) for r in self.plan['rows']}
        class Renderer:
            def __init__(self, *args, **kwargs): cameras.append(self)
            def close(self): closed.append(self)
        class Refiner:
            def __init__(self, *args, **kwargs): self.sink = kwargs.get('chronological_capture_callback')
            def set_chronological_capture_context(self, context): self.context = context
            def refine(self, frame, mask, seed):
                refined.append(self.context['physical_id'])
                for iteration in self.sink.physical['iterations']:
                    self.sink(dict(kind='retained', context=self.context,
                        metadata=dict(frame_id=frame.frame_id, iteration=iteration), arrays=arrays_fixture()))
                    self.sink(dict(kind='result', context=self.context,
                        metadata=dict(frame_id=frame.frame_id, iteration=iteration, fit_state='refined'), arrays={}))
        fake_torch = types.SimpleNamespace(set_num_threads=lambda n: None, use_deterministic_algorithms=lambda value: None,
            manual_seed=lambda value: None, load=lambda *args, **kwargs: {})
        network = types.SimpleNamespace(eval=lambda: None, cpu=lambda: network_released.append(True))
        with ExitStack() as stack:
            self.bypass_cloud(stack)
            stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch}))
            stack.enter_context(patch.object(states, '_disk_preflight'))
            stack.enter_context(patch.object(states, '_preflight', return_value=inputs))
            stack.enter_context(patch.object(states, 'verify_source_contract'))
            stack.enter_context(patch.object(states, 'pinned_file', side_effect=lambda pin, **kwargs: Path(pin['path'])))
            stack.enter_context(patch.object(states, 'observe_selected_crops', side_effect=lambda selected: nullcontext(
                {i: dict(integer_centers=SIZE**2, max_native_pixel_difference=0.) for i in selected})))
            stack.enter_context(patch('bench.quality_adaptation_runtime.runtime_versions', return_value={}))
            stack.enter_context(patch('bench.quality_gotrack.load_network', return_value=network))
            stack.enter_context(patch('bench.quality_gotrack.TexturedRenderer', Renderer))
            stack.enter_context(patch('bench.quality_gotrack.GoTrackRefiner', Refiner))
            stack.enter_context(patch('bench.quality_foundpose.FoundPoseRecovery', return_value=types.SimpleNamespace(recover=lambda *a, **k: [np.eye(4)])))
            stack.enter_context(patch('bench.quality_runner.read_rgb', return_value=np.zeros((SIZE, SIZE, 3), np.uint8)))
            stack.enter_context(patch('bench.vision.cv2.imread', return_value=np.ones((SIZE, SIZE), np.uint8)))
            result = states.capture(plan_path, sha256(plan_path), output, cloud_execution=True)
        self.assertEqual(result['available'], 40)
        self.assertEqual(refined, [r['id'] for r in self.plan['rows']])
        self.assertEqual(len(closed), 24)
        self.assertEqual(len(cameras), 24)
        self.assertEqual(network_released, [True])
        self.assertNotIn('torch', sys.modules)

    def labels_fixture(self):
        capture, features = self.make_seal_fixture()
        with patch.object(states, 'verify_source_contract'):
            (capture / 'prelabel-seal.json').write_text(json.dumps(states._seal_material(capture, features)))
        generation = self.root / 'generation.json'; generation.write_text('{}')
        split = self.root / 'split.json'; split.write_text('{}')
        label_plan = self.root / 'label-plan.json'; label_plan.write_text('{}')
        renderer_closes = []
        class Renderer:
            vertices_m = np.zeros((8, 3))
            def __init__(self, *args, **kwargs): pass
            def close(self): renderer_closes.append(True)
        native_m = np.eye(4); native_m[2, 3] = 1.
        from .quality_adaptation_manifest import assemble
        commands = []
        def historical(root, manifest, module, args, **kwargs):
            commands.append((module, args))
            self.assertEqual(module, 'bench.quality_adaptation_manifest')
            assemble(features, features / 'labels.json', features / 'manifest.json')
            return 0
        with ExitStack() as stack:
            self.bypass_cloud(stack)
            stack.enter_context(patch.object(states, 'verify_source_contract'))
            stack.enter_context(patch.object(states, 'verify_training_root'))
            stack.enter_context(patch.object(states, '_read_label_plan', return_value=({}, split, split,
                {seq: generation for _, _, seq, _ in states.SEQUENCES})))
            stack.enter_context(patch.object(states, 'pinned_file', side_effect=lambda pin, **kwargs: Path(pin['path'])))
            stack.enter_context(patch.object(states, '_run_historical', side_effect=historical))
            stack.enter_context(patch('bench.quality_ranker_synthetic.validate_recipe', return_value={}))
            stack.enter_context(patch('bench.quality_adaptation_smoke.validate_generation_context'))
            stack.enter_context(patch('bench.quality_adaptation_smoke.label_pose', return_value=native_m))
            stack.enter_context(patch('bench.quality_adaptation_smoke.render_query_labels', return_value=(native_m,
                np.ones((SIZE, SIZE), np.float32), np.ones((SIZE, SIZE), bool))))
            stack.enter_context(patch('bench.quality_gotrack.TexturedRenderer', Renderer))
            stack.enter_context(patch('bench.quality_runner.read_input', return_value=dict(asset='object.glb', object_id=17)))
            result = states.labels(capture, features, label_plan, sha256(label_plan), self.root / 'source18',
                self.root / 'manifest18.json', cloud_execution=True)
        self.assertEqual(result['rows'], 41)
        self.assertEqual(len(renderer_closes), 2)
        self.assertEqual(len(commands), 1)
        return capture, features

    def test_labels_use_generic_assembler_with_exact_41_row_inventory_and_real_supervision(self):
        capture, features = self.labels_fixture()
        doc = read_json(features / 'manifest.json')
        self.assertEqual(doc['rows'][-1], self.plan['sealed_test'])
        self.assertEqual(sum(r['status'] == 'available' for r in doc['rows']), 2)
        receipt = read_json(features / 'label-receipt.json')
        self.assertEqual(receipt['manifest_sha256'], sha256(features / 'manifest.json'))
        self.assertEqual(len(receipt['label_artifacts']), 2)
        self.assertFalse(receipt['labels_entered_inference'])
        from .quality_adaptation_train import validate_feature_receipt
        from .quality_adaptation_data import load_dataset
        validate_feature_receipt(load_dataset(features / 'manifest.json'))

    def test_train_partial_inventory_resume_export_delegate_real_existing_commands(self):
        capture, features = self.labels_fixture()
        from dataclasses import asdict
        from .quality_adaptation_train import TrainConfig
        cfg = asdict(TrainConfig()); cfg.update(epochs=1, max_steps=16, max_seconds=1200., max_total_seconds=3600.,
            max_output_bytes=1024**3, max_checkpoint_bytes=320 * 1024**2, checkpoint_every_steps=8)
        config = self.root / 'config.json'; config.write_text(json.dumps(cfg))
        run = self.root / 'run'
        commands = []
        def historical(root, manifest, module, args, **kwargs):
            commands.append((module, args, kwargs))
            run.mkdir(exist_ok=True)
            if args[0] == 'export':
                Path(args[args.index('--output') + 1]).write_bytes(b'fake export; no torch')
                return 0
            from .quality_adaptation_data import load_dataset
            ds = load_dataset(features / 'manifest.json')
            (run / 'config.json').write_text(json.dumps(cfg))
            (run / 'preflight.json').write_text(json.dumps(dict(config_sha256=canonical_hash(cfg), data=dict(dataset_sha256=ds.identity))))
            (run / 'status.json').write_text(json.dumps(dict(steps=1, status='incomplete')))
            (run / 'last.pt').write_bytes(b'fake checkpoint; not a model')
            return 3
        with ExitStack() as stack:
            self.bypass_cloud(stack)
            stack.enter_context(patch.object(states, 'verify_source_contract'))
            stack.enter_context(patch.object(states, '_run_historical', side_effect=historical))
            common = dict(cloud_execution=True)
            states.train_stage(capture, features, config, sha256(config), run, self.root / 'source18', self.root, stop_after_steps=1, **common)
            states.train_stage(capture, features, config, sha256(config), run, self.root / 'source18', self.root, resume=True, **common)
            states.train_stage(capture, features, config, sha256(config), run, self.root / 'source18', self.root, export_output=run / 'head.pt', **common)
        self.assertEqual([entry[1][0] for entry in commands], ['train', 'resume', 'export'])
        self.assertIn('--stop-after-steps', commands[0][1])
        receipt = read_json(run / 'refinement-state-run-receipt.json')
        self.assertEqual(receipt['actual_optimizer_steps'], 1)
        self.assertEqual(receipt['available_train_rows'], 1)
        self.assertEqual(receipt['available_used_val_rows'], 1)

    def test_train_rejects_label_mutation_before_historical_process(self):
        capture, features = self.labels_fixture()
        receipt = read_json(features / 'label-receipt.json')
        label = features / receipt['label_artifacts'][0]['path']
        with label.open('ab') as stream: stream.write(b'foreign label bytes')
        from dataclasses import asdict
        from .quality_adaptation_train import TrainConfig
        cfg = asdict(TrainConfig()); cfg.update(epochs=1, max_steps=16, max_seconds=1200., max_total_seconds=3600.,
            max_output_bytes=1024**3, max_checkpoint_bytes=320 * 1024**2, checkpoint_every_steps=8)
        config = self.root / 'config.json'; config.write_text(json.dumps(cfg))
        with ExitStack() as stack:
            self.bypass_cloud(stack)
            stack.enter_context(patch.object(states, 'verify_source_contract'))
            runner = stack.enter_context(patch.object(states, '_run_historical'))
            with self.assertRaisesRegex(ValueError, 'artifact identity'):
                states.train_stage(capture, features, config, sha256(config), self.root / 'run', self.root, self.root, cloud_execution=True)
            runner.assert_not_called()

    def supervisor_fixture(self, *, timeout=False, surviving_child=False):
        cache = self.root / '.cache/model-quality'; cache.mkdir(parents=True, exist_ok=True)
        plan_path = self.root / 'plan.json'; plan_path.write_text(json.dumps(self.plan))
        target = self.root / 'supervised-capture'
        args = types.SimpleNamespace(command='capture', output=target, plan=plan_path,
            plan_sha256=sha256(plan_path), cloud_execution=True)
        class MissingProcess(Exception): pass
        class Child:
            def memory_info(self): return types.SimpleNamespace(rss=100)
            def terminate(self): pass
            def kill(self): pass
        child = Child()
        class Process:
            pid = 12345
            returncode = None
            polls = 0
            def poll(self):
                self.polls += 1
                if not timeout and self.polls > 1: self.returncode = 0
                return self.returncode
            def wait(self, **kwargs): self.returncode = 0; return 0
            def terminate(self): self.returncode = 0
            def kill(self): self.returncode = 0
        def popen(*args, **kwargs):
            (cache / '.refinement-state-stage.lock').write_text(json.dumps(dict(pid=12345)))
            return Process()
        def ps_process(pid):
            if not timeout: raise MissingProcess('fast completed child')
            return types.SimpleNamespace(children=lambda **kwargs: [child] if surviving_child else [],
                memory_info=lambda: types.SimpleNamespace(rss=100))
        psutil = types.SimpleNamespace(Process=ps_process, NoSuchProcess=MissingProcess, AccessDenied=PermissionError,
            wait_procs=lambda children, **kwargs: ([], children if surviving_child else []))
        return args, popen, psutil, cache

    def test_supervisor_unique_monitors_allow_repeated_invocations_and_fast_exit(self):
        args, popen, psutil, cache = self.supervisor_fixture()
        with ExitStack() as stack:
            stack.enter_context(patch.object(states, 'ROOT', self.root))
            stack.enter_context(patch.object(states, 'cloud_guard'))
            stack.enter_context(patch.dict(sys.modules, {'psutil': psutil}))
            stack.enter_context(patch.object(states.subprocess, 'Popen', side_effect=popen))
            stack.enter_context(patch.object(states, 'tree_bytes', return_value=0))
            stack.enter_context(patch.object(states.time, 'sleep'))
            first = states._supervise(args, ['capture'])
            second = states._supervise(args, ['capture'])
        self.assertNotEqual(first['monitor'], second['monitor'])
        self.assertTrue(first['complete'])
        self.assertFalse((cache / '.refinement-state-stage.lock').exists())

    def test_supervisor_timeout_preserves_40_rows_and_retains_lock_on_surviving_child(self):
        args, popen, psutil, cache = self.supervisor_fixture(timeout=True, surviving_child=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(states, 'ROOT', self.root))
            stack.enter_context(patch.object(states, 'cloud_guard'))
            stack.enter_context(patch.dict(sys.modules, {'psutil': psutil}))
            stack.enter_context(patch.object(states.subprocess, 'Popen', side_effect=popen))
            stack.enter_context(patch.object(states, 'tree_bytes', return_value=0))
            stack.enter_context(patch.object(states.time, 'time', side_effect=[1000., 2001., 2002.]))
            with self.assertRaisesRegex(RuntimeError, 'process_tree_cleanup_incomplete'):
                states._supervise(args, ['capture'])
        monitor = next(self.root.glob('supervised-capture-monitor-*'))
        rows = read_json(monitor / 'state-inventory.json')['rows']
        self.assertEqual(len(rows), 40)
        self.assertTrue(all(r['status'] == 'unavailable' for r in rows))
        self.assertTrue((cache / '.refinement-state-stage.lock').exists())


if __name__ == '__main__':
    unittest.main()
