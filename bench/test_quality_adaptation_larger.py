"""Model-free admission and geometry regression checks for the larger controller."""
import copy
from dataclasses import replace
from contextlib import ExitStack, nullcontext, redirect_stdout
import hashlib
import json
import os
import io
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

_TEST_TEMP = Path(__file__).resolve().parents[1] / '.cache/cloud-prepare/test-temp'
_TEST_TEMP.mkdir(parents=True, exist_ok=True)
from . import quality_adaptation_larger as larger
from . import quality_adaptation_larger_prepare as prepare
from .quality_adaptation_data import crop_rgb_hash


def pin(path, letter='a'):
    return dict(path=path, bytes=1, sha256=letter * 64)


def spec():
    return dict(schema=prepare.SCHEMA, runtime_contract=pin('bench/runtime/gotrack-inference-v1.json'),
        source_contract=pin('bench/runtime/larger-source.json'), assets=[
            dict(object_id='1', split='train', physical_object_id='hot3d:obj_000001', mesh=pin(larger.SCRATCH + '/assets/1.glb', 'a')),
            dict(object_id='4', split='val', physical_object_id='hot3d:obj_000004', mesh=pin(larger.SCRATCH + '/assets/4.glb', 'b'))],
        views=[dict(object_id=obj, view_index=view, iterations=iterations, sequence_id=f'larger-{obj}-{view}', capture_group=f'group-{obj}')
            for obj, view, iterations in [('1', 0, [0]), ('1', 3, [1]), ('4', 2, [0, 1, 3])]],
        sealed_test=dict(id='test7-unmaterialized', object_id='7', sequence_id='sealed-test7-unmaterialized',
            capture_group='sealed-test7', split='test', status='unavailable', query_rgb_sha256=None,
            failure_reason='held_out_unmaterialized', unavailable_provenance='never opened'), limits=larger.LIMITS.copy())


def plan():
    doc = spec()
    assets = [dict(a, original_mesh_sha256=a['mesh']['sha256'], bank=pin(larger.SCRATCH + '/banks/' + a['object_id'] + '.pt', 'c')) for a in doc['assets']]
    owners = {a['object_id']: a for a in assets}
    rows = []
    for index, view in enumerate(doc['views']):
        asset = owners[view['object_id']]
        bundle = larger.SCRATCH + '/bundle-' + str(index)
        rows.append(dict(view, id=view['sequence_id'] + ':frame0', split=asset['split'], frame_id=0, timestamp_s=0.,
            input=pin(bundle + '/input.json'), video=pin(bundle + '/synthetic.mp4', 'd' if asset['split'] == 'train' else 'e'),
            mesh=dict(asset['mesh'], path=bundle + '/object.glb'), mask_index=pin(bundle + '/masks/results.json'),
            mask=pin(bundle + '/masks/0.png'), bank=asset['bank'], observation_status='available', failure_reason=None))
    return dict(schema=larger.SCHEMA, runtime_contract=doc['runtime_contract'], source_contract=doc['source_contract'],
        assets=assets, rows=rows, sealed_test=doc['sealed_test'], limits=doc['limits'])


class PlanTests(unittest.TestCase):
    def test_arbitrary_bounded_plan_and_states(self):
        doc = larger.validate_plan(plan())
        self.assertEqual(len(larger.state_rows(doc)), 5)
        self.assertEqual([r['iteration'] for r in larger.state_rows(doc)], [0, 1, 0, 1, 3])
        self.assertTrue(all(r['status'] == 'unavailable' for r in larger.state_rows(doc)))

    def test_original_mesh_path_distinct_from_copied_bundle(self):
        doc = larger.validate_plan(plan())
        self.assertNotEqual(doc['assets'][0]['mesh']['path'], doc['rows'][0]['mesh']['path'])

    def test_last_foreign_bank_rejected_before_payload_read(self):
        doc = plan(); doc['rows'][-1]['bank'] = pin(larger.SCRATCH + '/foreign.pt')
        with patch.object(larger, 'pinned_file', side_effect=AssertionError('payload opened')):
            with self.assertRaisesRegex(ValueError, 'authorization'):
                larger.preflight(doc)

    def test_last_test_payload_rejected_before_any_open(self):
        doc = plan(); doc['rows'][-1]['mask']['path'] = larger.SCRATCH + '/obj_000007/0.png'
        with patch.object(larger, 'pinned_file', side_effect=AssertionError('opened')):
            with self.assertRaisesRegex(ValueError, 'TEST'):
                larger.preflight(doc)

    def test_cross_split_mesh_hash(self):
        doc = plan(); doc['assets'][1]['mesh']['sha256'] = 'a' * 64; doc['assets'][1]['original_mesh_sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            larger.validate_plan(doc)

    def test_cross_split_capture_group(self):
        doc = plan(); doc['rows'][-1]['capture_group'] = doc['rows'][0]['capture_group']
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            larger.validate_plan(doc)

    def test_cross_split_video_hash(self):
        doc = plan(); doc['rows'][-1]['video']['sha256'] = doc['rows'][0]['video']['sha256']
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            larger.validate_plan(doc)

    def test_empty_rows(self):
        doc = plan(); doc['rows'] = []
        with self.assertRaises(ValueError): larger.validate_plan(doc)

    def test_duplicate_views(self):
        doc = plan(); doc['rows'][1]['view_index'] = 0
        with self.assertRaisesRegex(ValueError, 'duplicate'): larger.validate_plan(doc)

    def test_empty_iterations(self):
        doc = plan(); doc['rows'][0]['iterations'] = []
        with self.assertRaises(ValueError): larger.validate_plan(doc)

    def test_boolean_iteration(self):
        doc = plan(); doc['rows'][0]['iterations'] = [False]
        with self.assertRaises(ValueError): larger.validate_plan(doc)

    def test_missing_mask(self):
        doc = plan(); doc['rows'][0]['mask'] = None
        with self.assertRaisesRegex(ValueError, 'lacks'): larger.validate_plan(doc)

    def test_missing_seed_bank(self):
        doc = plan(); doc['assets'][0]['bank'] = None
        for row in doc['rows'][:2]: row['bank'] = None
        with self.assertRaisesRegex(ValueError, 'lacks'): larger.validate_plan(doc)

    def test_unavailable_preserved_with_missing_mask_bank(self):
        doc = plan(); doc['assets'][0]['bank'] = None
        for row in doc['rows'][:2]:
            row.update(bank=None, mask=None, observation_status='unavailable', failure_reason='automatic_mask_failed')
        larger.validate_plan(doc)
        self.assertEqual(len(larger.state_rows(doc)), 5)

    def test_test_metadata_payload_forbidden(self):
        doc = plan(); doc['sealed_test']['crops'] = pin(larger.SCRATCH + '/test.npz')
        with self.assertRaisesRegex(ValueError, 'TEST metadata'): larger.validate_plan(doc)

    def test_changed_limit_rejected(self):
        doc = plan(); doc['limits']['max_states'] = 313
        with self.assertRaises(ValueError): larger.validate_plan(doc)

    def test_prospective_counts_bound(self):
        doc = plan(); doc['rows'] = doc['rows'] * 100
        with self.assertRaises(ValueError): larger.validate_plan(doc)

    def test_input_plan_mutation(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            path = Path(folder) / 'plan.json'; path.write_text(json.dumps(plan()))
            digest = larger.sha256(path); path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'byte hash'): larger.read_plan(path, digest)

    def test_source_mutation(self):
        doc = plan()
        contract = dict(schema=larger.SOURCE_SCHEMA, historical_training_manifest_sha256=larger.HISTORICAL_MANIFEST_SHA256,
            sources=[dict(path='bench/quality_adaptation_larger.py', sha256='a' * 64)])
        with patch.object(larger, 'pinned_file', return_value=Path('contract')), patch.object(larger, 'read_json', return_value=contract), \
             patch.object(larger.shared, '_relative', return_value=Path('source')), patch.object(larger, 'sha256', return_value='b' * 64):
            with self.assertRaisesRegex(ValueError, 'source closure mutation'): larger.verify_source_contract(doc)

    def test_source_foreign_dependency(self):
        contract = dict(schema=larger.SOURCE_SCHEMA, historical_training_manifest_sha256=larger.HISTORICAL_MANIFEST_SHA256, sources=[])
        with patch.object(larger, 'pinned_file', return_value=Path('contract')), patch.object(larger, 'read_json', return_value=contract):
            with self.assertRaisesRegex(ValueError, 'missing/foreign'): larger.verify_source_contract(plan())

    def test_label_plan_never_opened_before_seal(self):
        with patch.object(larger, 'cloud_guard'), patch.object(larger, 'stage_lock'), patch.object(larger, 'remaining'), \
             patch.object(larger, 'verify_prelabel_seal', side_effect=ValueError('seal incomplete')), patch.object(larger, '_label_plan', side_effect=AssertionError('label opened')):
            with self.assertRaisesRegex(ValueError, 'seal incomplete'):
                larger.labels(Path('capture'), Path('features'), Path('labels'), 'a' * 64, Path('source18'), Path('manifest'), cloud_execution=True)


class PreparationTests(unittest.TestCase):
    def test_frozen_arbitrary_spec(self):
        self.assertEqual(len(prepare.validate_spec(spec())['views']), 3)

    def test_cross_split_acquisition_hash_before_open(self):
        doc = spec(); doc['assets'][1]['mesh']['sha256'] = doc['assets'][0]['mesh']['sha256']
        with patch.object(larger, 'pinned_file', side_effect=AssertionError('opened')):
            with self.assertRaisesRegex(ValueError, 'cross-split'): prepare.validate_spec(doc, verify_files=True)

    def test_unknown_view_refused(self):
        with self.assertRaises(ValueError): prepare._select(spec(), '7')
        with self.assertRaises(ValueError): prepare._select(spec(), '1', 23)

    def test_same_object_24views_balance(self):
        doc = spec()
        doc['views'] = [dict(object_id=obj, view_index=i, iterations=[(0, 1, 3)[(i + ordinal) % 3]],
            sequence_id=f'obj{obj}-v{i}', capture_group=f'obj{obj}')
            for ordinal, obj in enumerate(('1', '4')) for i in range(24)]
        prepare.validate_spec(doc)
        self.assertEqual([sum(v['iterations'] == [iteration] for v in doc['views'][:24]) for iteration in (0, 1, 3)], [8, 8, 8])

    def test_spec_source_change_rejected(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            path = Path(folder) / 'spec.json'; path.write_text(json.dumps(spec()))
            expected = larger.sha256(path); path.write_text('{}')
            with patch.object(prepare, 'cloud_guard'):
                with self.assertRaisesRegex(ValueError, 'byte hash'): prepare.freeze(path, expected, Path(folder) / 'new', cloud_execution=True)

    def test_mask_producer_requires_observed_origin(self):
        with self.assertRaisesRegex(ValueError, 'producer binding'):
            larger.validate_observed_mask(plan()['rows'][0], dict(automatic=True, mask_origin='projected_silhouette'), plan())

    def test_cnos_override_default_preserved_in_signature(self):
        import inspect
        from .quality_cnos import detect
        parameter = inspect.signature(detect).parameters['bank_path']
        self.assertIsNone(parameter.default)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)


class GeometryAndBudgetTests(unittest.TestCase):
    def test_rotated_geometry_and_metres_mm(self):
        from .quality_adaptation_prepare import crops_from_retained_capture
        size = larger.SIZE
        angle = np.deg2rad(37)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = [10., -20., 500.]
        k = np.array([[200., 0, 140.], [0, 200., 140.], [0, 0, 1.]])
        arrays = dict(query_rgb_crop=np.full((1, 3, size, size), .1234567, np.float32),
            template_rgb=np.full((1, 3, size, size), .7654321, np.float32),
            rendered_depth_mm=np.full((size, size), 500., np.float32), rendered_mask=np.ones((size, size), bool),
            crop_k=k, current_crop_camera_from_object_mm=pose)
        crop = crops_from_retained_capture(arrays, preserve_float_rgb=True)
        self.assertEqual(crop['query_rgb'].dtype, np.float32)
        self.assertEqual(float(crop['query_rgb'][0, 0, 0]), float(np.float32(.1234567)))
        expected = rotation.T @ (np.array([0., 0., .5]) - np.array([.01, -.02, .5]))
        np.testing.assert_allclose(crop['template_xyz_m'][140, 140], expected, atol=1e-7)

    def test_rotated_sampling_map_integer_centers(self):
        from .quality_gotrack import _capture_native_uv
        angle = np.deg2rad(15)
        r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        k = np.array([[200., 0, 140.], [0, 200., 140.], [0, 0, 1.]])
        points = np.array([[140., 140.], [141., 140.]])
        result = _capture_native_uv(points, k, r, k)
        np.testing.assert_allclose(result[0], [140., 140.], atol=1e-10)
        self.assertTrue(np.isfinite(result).all())

    def test_frozen_config(self):
        value = larger.expected_config()
        self.assertEqual(value['epochs'], 2); self.assertEqual(value['max_steps'], 480)
        self.assertEqual(value['seed'], 42); self.assertEqual(value['learning_rate'], 1e-5)
        self.assertGreater(value['max_output_bytes'], 3 * value['max_checkpoint_bytes'] + 8 * 1024**2)

    def test_output_escape_and_symlink_policy(self):
        with self.assertRaises(ValueError): larger._safe_output(Path('/tmp/other/capture'))
        with self.assertRaises(ValueError): larger._safe_output(Path(larger.SCRATCH + '/run'), durable=True)

    def test_aggregate_clock_expired(self):
        with patch.object(larger, 'read_json', return_value=dict(started_unix_s=time.time() - 7201)):
            with self.assertRaises(TimeoutError): larger.remaining(Path('capture'))

    def test_scratch_cap(self):
        # Lightning cwd is durable; test the byte cap on a valid scratch mount.
        scratch = Path(tempfile.gettempdir())
        with patch.object(larger, '_safe_output', return_value=scratch), \
             patch.object(larger, 'tree_bytes', return_value=larger.LIMITS['max_scratch_bytes'] + 1):
            with self.assertRaisesRegex(ValueError, 'scratch byte'): larger.resources(scratch)

    def test_observer_capability_cannot_be_forged(self):
        cap = larger.LargerObserverCapability(object(), b'{}', hashlib.sha256(b'{}').hexdigest(), object())
        with self.assertRaisesRegex(ValueError, 'invalid larger observer'): cap.validate(cap._sink, larger.inference.PROFILE)

    def test_dataset_mutation_rejected_by_full_inventory(self):
        doc = plan(); rows = larger.state_rows(doc)
        rows[-1]['object_id'] = '7'
        receipt = dict(rows=rows, plan_sha256='a' * 64, fatal_error=None, annotation_inputs=[], input_only=True, test_payload_reads=0)
        index = dict(rows=rows, test_payload_reads=0)
        with patch.object(larger, 'read_json', side_effect=[doc, index, receipt]), patch.object(larger, 'verify_source_contract'), \
             patch.object(larger, 'sha256', return_value='a' * 64):
            with self.assertRaisesRegex(ValueError, 'prospective'): larger.read_capture(Path('capture'))


class StageBoundaryTests(unittest.TestCase):
    def setUp(self):
        # The scientific runtime intentionally has no optional process monitor;
        # these tests exercise supervision state/cleanup using a bounded stand-in.
        fake_process = SimpleNamespace(children=lambda recursive=True: [], is_running=lambda: True,
            memory_info=lambda: SimpleNamespace(rss=1024))
        monitor = SimpleNamespace(Process=lambda pid: fake_process, NoSuchProcess=ProcessLookupError,
            Error=OSError, wait_procs=lambda processes, timeout: (processes, []))
        self.monitor_patch = patch.dict(sys.modules, {'psutil': monitor})
        self.monitor_patch.start()
        self.addCleanup(self.monitor_patch.stop)

    def test_config_mutation_before_historical_subprocess(self):
        with patch.object(larger, 'cloud_guard'), patch.object(larger, '_safe_output'), patch.object(larger, 'resources'), \
             patch.object(larger, 'remaining', return_value=100), patch.object(larger, 'verify_prelabel_seal'), \
             patch.object(larger, 'sha256', return_value='b' * 64), patch.object(larger.shared, '_run_historical', side_effect=AssertionError('native started')):
            with self.assertRaisesRegex(ValueError, 'configuration SHA'):
                larger.train_stage(Path('capture'), Path('features'), Path('config'), 'a' * 64,
                    Path('run'), Path('source18'), Path('manifest18'), cloud_execution=True)

    def test_old_durable_output_refused(self):
        with self.assertRaises(ValueError):
            larger._safe_output(Path('/teamspace/studios/this_studio/visualizeit-private-refinement-states-v1/run'), durable=True)

    def test_new_durable_root_admitted(self):
        with patch.object(larger.inference, '_has_symlink_component', return_value=False):
            self.assertEqual(larger._safe_output(Path(larger.DURABLE), durable=True), Path(larger.DURABLE))

    def test_durable_peak_counts_sibling_and_temporary(self):
        calls = []
        def size(path):
            calls.append(path)
            return 1_100_000_000 if path == Path(larger.DURABLE) else 0
        with patch.object(larger, '_safe_output', side_effect=lambda path, **kwargs: Path(path)), \
             patch.object(larger, 'tree_bytes', side_effect=size), \
             patch.object(larger.shutil, 'disk_usage', return_value=type('Disk', (), dict(free=100 * 1024**3))()):
            with self.assertRaisesRegex(ValueError, 'durable peak'):
                larger.resources(Path(tempfile.gettempdir()), Path(larger.DURABLE) / 'run', reserve=256 * 1024**2)
        self.assertIn(Path(larger.DURABLE), calls)

    def test_supervisor_normal_child_cleanup_receipt(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            scratch = Path(folder) / 'capture'
            with patch.object(larger, 'SCRATCH', folder), patch.object(larger, '_safe_output', side_effect=lambda path, **kw: Path(path)), \
                 patch.object(larger, 'resources'), patch.object(larger, 'workflow_started', return_value=time.time()):
                result = larger.supervise([sys.executable, '-c', 'pass'], scratch=scratch,
                    requested=[dict(id='one', split='train')])
            receipt = larger.read_json(Path(result['monitor']) / 'supervision.json')
            self.assertTrue(receipt['complete']); self.assertTrue(receipt['process_tree_cleanup_complete'])
            inventory = larger.read_json(Path(result['monitor']) / 'state-inventory.json')
            self.assertEqual([r['id'] for r in inventory['rows']], ['one'])

    def test_supervisor_failed_child_full_inventory(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            with patch.object(larger, 'SCRATCH', folder), patch.object(larger, '_safe_output', side_effect=lambda path, **kw: Path(path)), \
                 patch.object(larger, 'resources'), patch.object(larger, 'workflow_started', return_value=time.time()):
                with self.assertRaisesRegex(RuntimeError, 'worker exit'):
                    larger.supervise([sys.executable, '-c', 'raise SystemExit(9)'], scratch=Path(folder) / 'capture',
                        requested=[dict(id='one'), dict(id='two')])
            monitor = next(Path(folder).glob('monitor-*'))
            inventory = larger.read_json(monitor / 'state-inventory.json')
            self.assertEqual([r['id'] for r in inventory['rows']], ['one', 'two'])
            self.assertTrue(all('worker exit' in r['failure_reason'] for r in inventory['rows']))

    def test_upstream_lock_mutation_before_payload_hashes(self):
        with patch.object(larger, 'sha256', return_value='f' * 64):
            with self.assertRaisesRegex(ValueError, 'source lock mutation'):
                larger.verify_upstream_sources()

    def test_label_archive_accepts_feature_directory_above_old_two_gib(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            with patch.object(larger, 'resources') as reservation, \
                 patch('bench.quality_adaptation_runtime.tree_bytes', return_value=2 * 1024**3 + 1), \
                 patch('bench.quality_adaptation_runtime.shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                receipt = larger._save_npz(Path(folder), 'label.npz', dict(dummy=np.zeros(1, np.float32)))
            reservation.assert_called_once_with(Path(folder), reserve=16 * 1024**2)
            self.assertEqual(receipt['path'], 'label.npz')
            self.assertTrue((Path(folder) / 'label.npz').is_file())

    def test_no_torch_import_in_model_free_suite(self):
        self.assertNotIn('torch', sys.modules)


class CaptureIntegrationTests(unittest.TestCase):
    def run_manufactured(self, *, seed_available=True):
        from .test_quality_adaptation_refinement_states import arrays_fixture
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            root = Path(folder); doc = plan()
            path = root / 'plan.json'; path.write_text(json.dumps(doc))
            output = root / 'captured'
            refined, closed, released = [], [], []
            k = arrays_fixture()['native_k']
            inputs = {r['id']: (root, dict(asset='object.glb', object_id=1, _timing=dict(time_policy=None)),
                SimpleNamespace(effective_intrinsics=k)) for r in doc['rows']}
            class Renderer:
                def __init__(self, *a, **kw): pass
                def close(self): closed.append(True)
            class Refiner:
                def __init__(self, *a, **kw): self.sink = kw.get('chronological_capture_callback')
                def set_chronological_capture_context(self, context): self.context = context
                def refine(self, frame, mask, seed):
                    refined.append(self.context['physical_id'])
                    for iteration in self.sink.physical['iterations']:
                        self.sink(dict(kind='retained', context=self.context,
                            metadata=dict(frame_id=0, iteration=iteration), arrays=arrays_fixture()))
            network = SimpleNamespace(cpu=lambda: released.append(True))
            network.eval = lambda: network
            fake_torch = SimpleNamespace(set_num_threads=lambda n: None, use_deterministic_algorithms=lambda b: None,
                manual_seed=lambda n: None, load=lambda *a, **kw: dict(template_count=798))
            with ExitStack() as stack:
                stack.enter_context(patch.object(larger, 'cloud_guard'))
                stack.enter_context(patch.object(larger, 'stage_lock'))
                stack.enter_context(patch.object(larger, '_safe_output', side_effect=lambda p, **kw: Path(p)))
                stack.enter_context(patch.object(larger, 'resources'))
                stack.enter_context(patch.object(larger, 'preflight', return_value=inputs))
                stack.enter_context(patch.object(larger, 'verify_source_contract'))
                stack.enter_context(patch.object(larger, 'pinned_file', side_effect=lambda p, **kw: Path(p['path'])))
                stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch}))
                stack.enter_context(patch.object(larger.shared, 'observe_selected_crops', side_effect=lambda selected: nullcontext(
                    {i: dict(integer_centers=larger.SIZE**2, max_native_pixel_difference=0.) for i in selected})))
                stack.enter_context(patch('bench.quality_adaptation_runtime.runtime_versions', return_value={}))
                stack.enter_context(patch('bench.quality_gotrack.load_network', return_value=network))
                stack.enter_context(patch('bench.quality_gotrack.TexturedRenderer', Renderer))
                stack.enter_context(patch('bench.quality_gotrack.GoTrackRefiner', Refiner))
                stack.enter_context(patch('bench.quality_foundpose.FoundPoseRecovery', return_value=SimpleNamespace(
                    recover=lambda *a, **kw: [np.eye(4)] if seed_available else [])))
                stack.enter_context(patch('bench.quality_runner.read_rgb', return_value=np.zeros((larger.SIZE, larger.SIZE, 3), np.uint8)))
                stack.enter_context(patch('bench.vision.cv2.imread', return_value=np.ones((larger.SIZE, larger.SIZE), np.uint8)))
                result = larger.capture(path, larger.sha256(path), output, cloud_execution=True)
            rows = larger.read_json(output / 'capture-index.json')['rows']
            self.assertEqual(len(rows), 5); self.assertEqual(len(closed), 3); self.assertEqual(released, [True])
            self.assertEqual(len(refined), 3 if seed_available else 0)
            return result, rows

    def test_actual_variable_controller_retains_all_prescribed_states(self):
        result, rows = self.run_manufactured()
        self.assertEqual(result['available'], 5)
        self.assertTrue(all(r['pnp_state'] == 'solver_unavailable_after_retained_crop' for r in rows))

    def test_missing_estimated_seed_keeps_all_failures(self):
        result, rows = self.run_manufactured(seed_available=False)
        self.assertEqual(result['available'], 0)
        self.assertTrue(all('unavailable_seed' in r['failure_reason'] for r in rows))


class DeviceVariantTests(unittest.TestCase):
    def test_cpu_config_remains_exact_frozen_recipe(self):
        config = Path(__file__).resolve().parents[1] / '.cache/cloud-prepare/larger-gotrack-train-config-v1.json'
        self.assertEqual(larger.expected_config(), larger.read_json(config))
        cuda = larger.expected_config('cuda')
        self.assertFalse(cuda['amp'])
        self.assertEqual({**cuda, 'device': 'cpu'}, larger.expected_config())

    def test_cpu_setup_does_not_probe_cuda_or_change_old_determinism(self):
        calls = []
        fake = SimpleNamespace(set_num_threads=lambda n: calls.append(n),
            use_deterministic_algorithms=lambda b: self.fail('CPU prep default changed'),
            cuda=SimpleNamespace(is_available=lambda: self.fail('CUDA probed')))
        self.assertIs(larger.configure_torch('cpu', fake), fake)
        self.assertEqual(calls, [1])

    def test_explicit_missing_cuda_refuses_before_configuration_or_models(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
            set_num_threads=lambda n: self.fail('setup continued after missing CUDA'))
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                larger.configure_torch('cuda', fake)

    def test_cuda_flags_and_cleanup(self):
        calls = []
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
            synchronize=lambda: calls.append('synchronize'), empty_cache=lambda: calls.append('empty')),
            set_num_threads=lambda n: calls.append(('threads', n)),
            use_deterministic_algorithms=lambda b: calls.append(('deterministic', b)),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False)))
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            larger.configure_torch('cuda', fake)
            larger.cleanup_device('cuda', fake)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.backends.cudnn.allow_tf32)
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertTrue(fake.backends.cudnn.deterministic)
        self.assertEqual(calls[-2:], ['synchronize', 'empty'])

    def test_wrong_cuda_workspace_rejected_before_torch(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': 'wrong'}):
            with self.assertRaisesRegex(ValueError, 'before Torch import'):
                larger.configure_torch('cuda', SimpleNamespace())

    def test_capture_cli_default_cpu_and_explicit_cuda(self):
        for selected in (None, 'cuda'):
            args = ['capture', '--plan', 'plan', '--plan-sha256', 'a' * 64, '--output', 'capture', '--_worker']
            if selected: args += ['--device', selected]
            with patch.object(larger, 'cloud_guard'), patch.object(larger, 'capture', return_value={}) as call, redirect_stdout(io.StringIO()):
                larger.main(args)
            self.assertEqual(call.call_args.kwargs['device'], selected or 'cpu')

    def test_prepare_bank_cli_device_propagated(self):
        for selected in ('cpu', 'cuda'):
            with patch.object(prepare, 'cloud_guard'), patch.object(prepare, 'bank', return_value={}) as call, redirect_stdout(io.StringIO()):
                prepare.main(['bank', '--root', 'prepared', '--object-id', '1', '--device', selected, '--_worker'])
            self.assertEqual(call.call_args.kwargs['device'], selected)

    def test_extractor_cuda_device_and_actual_historical_hard_bounds(self):
        from .quality_adaptation_prepare import ExtractionConfig
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            root, features = Path(folder) / 'capture', Path(folder) / 'features'
            commands = []
            def run(*args, **kwargs):
                commands.append((args, kwargs))
                options = args[3]
                cfg = ExtractionConfig(int(options[options.index('--max-seconds') + 1]),
                    int(options[options.index('--max-output-bytes') + 1]),
                    int(options[options.index('--max-rows') + 1]), options[options.index('--device') + 1])
                cfg.validate()
                self.assertEqual(cfg.device, 'cuda')
                self.assertLessEqual(cfg.max_output_bytes, 8 * 1024**3)
                return 0
            with patch.object(larger, 'cloud_guard'), patch.object(larger, '_safe_output', side_effect=Path), \
                 patch.object(larger, 'read_capture', return_value=(plan(), larger.state_rows(plan()))), \
                 patch.object(larger, 'capture_device', return_value='cuda'), patch.object(larger, 'stage_lock'), \
                 patch.object(larger, 'remaining', return_value=100), patch.object(larger, 'run_historical_device', side_effect=run):
                larger.extract_stage(root, features, Path('source18'), Path('manifest18'), cloud_execution=True, device='cuda')
            self.assertEqual(commands[0][1]['device'], 'cuda')
            self.assertEqual(len(larger.read_json(features / 'index.json')['rows']), 5)

    def test_capture_extraction_wrong_device_refused_before_child(self):
        with patch.object(larger, 'cloud_guard'), patch.object(larger, '_safe_output', side_effect=Path), \
             patch.object(larger, 'read_capture', return_value=(plan(), [])), patch.object(larger, 'capture_device', return_value='cuda'), \
             patch.object(larger, 'run_historical_device') as run:
            with self.assertRaisesRegex(ValueError, 'extraction device differs'):
                larger.extract_stage(Path('capture'), Path('features'), Path('source18'), Path('manifest18'))
            run.assert_not_called()

    def test_resume_wrong_device_config_refused_before_child(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            config = Path(folder) / 'config.json'; config.write_text(json.dumps(larger.expected_config('cuda')))
            with patch.object(larger, 'cloud_guard'), patch.object(larger, '_safe_output'), patch.object(larger, 'resources'), \
                 patch.object(larger, 'remaining', return_value=100), patch.object(larger, 'verify_prelabel_seal'), \
                 patch.object(larger, 'run_historical_device') as run:
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    larger.train_stage(Path('capture'), Path('features'), config, larger.sha256(config),
                        Path('run'), Path('source18'), Path('manifest18'), resume=True, device='cpu')
                run.assert_not_called()

    def test_historical_cuda_bootstrap_and_authentication(self):
        with patch.object(larger.shared, 'verify_training_root') as verify, \
             patch.object(larger.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            larger.run_historical_device(Path('source18'), Path('manifest18'), 'bench.quality_adaptation_prepare',
                ['--device', 'cuda', '--output', 'features'], timeout=10, device='cuda')
        self.assertEqual(verify.call_count, 2)
        command = run.call_args.args[0]
        self.assertIn(larger._CUDA_BOOTSTRAP, command)
        self.assertEqual(command[-4:], ['--device', 'cuda', '--output', 'features'])
        self.assertEqual(run.call_args.kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertLess(larger._CUDA_BOOTSTRAP.index("os.environ.get('CUBLAS_WORKSPACE_CONFIG')"), larger._CUDA_BOOTSTRAP.index('import torch'))

    def test_cpu_historical_delegate_unchanged(self):
        with patch.object(larger.shared, '_run_historical', return_value=3) as run:
            value = larger.run_historical_device(Path('source18'), Path('manifest18'), 'trainer', ['train'], device='cpu', timeout=4)
        self.assertEqual(value, 3)
        self.assertEqual(run.call_args.kwargs, dict(timeout=4))

    def test_cuda_clock_explicit_variant_preserves_cpu_clock(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            cpu = Path(folder) / 'execution-clock.json'
            cpu.write_text(json.dumps(dict(started_unix_s=time.time() - 100)))
            original = cpu.read_bytes()
            start = time.time() - 5
            with patch.object(larger, 'SCRATCH', folder), patch.dict(os.environ, {'VISUALIZEIT_LARGER_CUDA_STARTED': str(start)}):
                self.assertEqual(larger.workflow_started('cuda'), start)
                self.assertEqual(larger.workflow_started('cuda'), start)
            self.assertEqual(cpu.read_bytes(), original)
            self.assertTrue((Path(folder) / 'execution-clock-cuda-v1.json').is_file())

    def test_new_cuda_clock_requires_operator_seed(self):
        with tempfile.TemporaryDirectory(dir=_TEST_TEMP) as folder:
            with patch.object(larger, 'SCRATCH', folder), patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, 'explicit operator'):
                    larger.workflow_started('cuda')
            self.assertFalse((Path(folder) / 'execution-clock-cuda-v1.json').exists())


if __name__ == '__main__':
    unittest.main()
