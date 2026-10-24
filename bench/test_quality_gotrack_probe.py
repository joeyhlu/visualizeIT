"""Analytical/mocked ordinary-solver tests. No neural inference or Torch import."""
from contextlib import nullcontext
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_gotrack_probe as probe
from . import quality_gotrack_inference as inference
from .quality_gotrack import GoTrackRefiner
from .quality_contract import Frame


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape, self.dtype = self.value.shape, self.value.dtype
    def __getitem__(self, item): return Tensor(self.value[item])
    def permute(self, *axes): return Tensor(self.value.transpose(axes))
    def float(self): return Tensor(self.value.astype(np.float32))
    def to(self, device): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def detach(self): return self
    def numel(self): return self.value.size
    def element_size(self): return self.value.itemsize
    def __truediv__(self, value): return Tensor(self.value / value)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        folder = probe.ROOT / '.cache/cloud-prepare/test-temp'
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def pin(self, relative):
        path = probe.ROOT / relative
        return dict(path=relative, bytes=path.stat().st_size, sha256=inference._sha256_file(path))

    def capability(self, sink):
        # Source and sink binding are real; only cloud/cache verification is
        # replaced by the analytical fixture. Issuance is tested separately.
        data = b'{"analytical_fixture":true}'
        pins = [self.pin('bench/runtime/gotrack-inference-probe-v1.json'), self.pin('bench/quality_gotrack_probe.py')]
        return probe.InferenceObserverCapability(probe._ISSUER, data, probe._sha(data),
                                                tuple(probe._encoded(pin) for pin in pins), sink)

    def refine(self, sink=None, *, ransac_failure=False, malformed_ransac=False, enough=True, geometry=inference.PROFILE):
        k = np.array([[300., 0., 140.], [0., 300., 140.], [0., 0., 1.]])
        rendered = np.zeros((280, 280), np.float32)
        if enough: rendered[50:70, 50:70] = 1
        else: rendered[50, 50] = 1
        depth = np.full((1, 280, 280), 1000., np.float32)
        data = dict(crop_rgbs=Tensor(np.zeros((1, 3, 280, 280), np.float32)),
                    crop_masks=Tensor(np.ones((1, 280, 280), np.float32)),
                    templates=types.SimpleNamespace(rgbs=Tensor(np.zeros((1, 3, 280, 280), np.float32)),
                        masks=Tensor(rendered[None].astype(bool)), depths=Tensor(depth)))
        utils = types.ModuleType('utils')
        utils.data_util = types.SimpleNamespace(compute_gotrack_inputs_from_init_poses=lambda **kw: (data, [None], [Tensor(np.eye(4))]))
        utils.misc = types.SimpleNamespace(get_intrinsic_matrix=lambda cam: k)
        utils.structs = types.SimpleNamespace(PinholePlaneCameraModel=lambda **kw: kw)
        from .quality_capture import integer_depth_points
        pixels, xyz = integer_depth_points(depth[0], k)
        utils.transform3d = types.SimpleNamespace(get_3d_points_from_depth=lambda *args: (Tensor(pixels[None]), Tensor(xyz[None])))
        network = lambda *args: (Tensor(np.zeros((1, 2, 280, 280), np.float32)), Tensor(np.ones((1, 280, 280), np.float32)))
        renderer = types.SimpleNamespace(render_policy='legacy_v1', coordinate_mode='integer_centers_v1',
            unlit=True, disable_multisampling=True, vertices_m=np.zeros((8, 3)))
        frame = Frame(9, np.zeros((280, 280, 3), np.uint8), k, timestamp_s=.9, clock_mode='capture_seconds_v1')
        seed = np.eye(4); seed[2, 3] = 1.
        calls = []
        def ransac(obj, target, camera, rvec, tvec):
            calls.append([value.copy() for value in (obj, target, camera, rvec, tvec)])
            if malformed_ransac: return False, rvec, tvec, np.array([[999999]], np.int32)
            if ransac_failure: return False, rvec, tvec, None
            return True, rvec, tvec, np.arange(6, dtype=np.int32).reshape(-1, 1)
        def lm(obj, target, camera, distortion, rvec, tvec):
            # Deliberate in-place mutation exercises the pre-LM snapshot.
            tvec[0] += .25
            return rvec, tvec
        fake_torch = types.SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext, state=10)
        fake_torch.get_rng_state = lambda: fake_torch.state
        fake_torch.set_rng_state = lambda value: setattr(fake_torch, 'state', value)
        fake_torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
        with patch.object(inference, 'verify_source_compatibility', return_value={}), \
             patch.dict(sys.modules, {'torch': fake_torch, 'utils': utils}), \
             patch('bench.quality_gotrack.cv2.solvePnPRefineLM', side_effect=lm):
            refiner = GoTrackRefiner(network, renderer, 17, 'cpu', inference_geometry=geometry,
                chronological_capture_callback=sink, inference_observer_capability=self.capability(sink) if sink and geometry == inference.PROFILE else None)
            refiner._solve_pnp_ransac = ransac
            refiner.set_chronological_capture_context(dict(case_id='analytical'))
            candidate = refiner.refine(frame, np.ones((280, 280), np.uint8), seed)
        self.assertEqual(fake_torch.state, 10)
        return candidate, calls

    def sink(self, suffix='capture', budget=None):
        root = self.root / suffix; root.mkdir()
        return probe.ProbeCapture(probe.OutputBudget(root, budget or probe.LIMITS), 'fixture')

    def test_ordinary_observer_matches_candidate_solver_inputs_and_rng(self):
        random.seed(128); np.random.seed(129)
        state, npstate = random.getstate(), np.random.get_state()
        baseline, basecalls = self.refine()
        sink = self.sink()
        observed, calls = self.refine(sink)
        self.assertEqual(probe.candidate_identity(baseline), probe.candidate_identity(observed))
        self.assertEqual(len(calls), 5)
        for left, right in zip(basecalls, calls):
            for a, b in zip(left, right): np.testing.assert_array_equal(a, b)
        self.assertEqual(state, random.getstate())
        np.testing.assert_array_equal(npstate[1], np.random.get_state()[1])
        report = sink.finish()
        self.assertIsNone(report['capture_failure'])
        self.assertEqual([row['status'] for row in report['iterations']], ['refined'] * 5)
        self.assertEqual(len(report['events']), 20)
        pre = np.load(sink.output.root / 'fixture-i0-pre_lm.npz')
        post = np.load(sink.output.root / 'fixture-i0-result.npz')
        self.assertEqual(pre['ransac_tvec_mm'][0], 0.)
        self.assertEqual(post['refined_tvec_mm'][0], .25)
        np.testing.assert_array_equal(pre['retained_inlier_ids'], np.arange(6))
        self.assertAlmostEqual(post['native_camera_from_object_m'][0, 3], .00025)

    def test_callback_mutation_and_exception_cannot_change_inference(self):
        baseline, baseline_calls = self.refine()
        sink = self.sink()
        original = probe.ProbeCapture.__call__
        def mutate(observer, packet):
            if packet['kind'] == 'trace':
                return original(observer, packet)
            random.random(); np.random.random()
            sys.modules['torch'].state += 100
            for value in packet['arrays'].values(): value[...] = 0
            raise RuntimeError('analytical observer failure')
        with patch.object(probe.ProbeCapture, '__call__', mutate):
            state, npstate = random.getstate(), np.random.get_state()
            candidate, calls = self.refine(sink)
            self.assertEqual(state, random.getstate())
            np.testing.assert_array_equal(npstate[1], np.random.get_state()[1])
        self.assertEqual(probe.candidate_identity(candidate), probe.candidate_identity(baseline))
        self.assertEqual(len(calls), len(baseline_calls))
        for left, right in zip(calls, baseline_calls):
            for a, b in zip(left, right): np.testing.assert_array_equal(a, b)
        self.assertIn('analytical observer failure', sink.failure)
        self.assertIsNotNone(sink.finish()['capture_failure'])

    def test_budget_refusal_preserves_inference_and_refuses_capture_completion(self):
        sink = self.sink(budget={**probe.LIMITS, 'max_raw_capture_bytes': 1})
        candidate, calls = self.refine(sink)
        self.assertIsNotNone(candidate)
        self.assertEqual(len(calls), 5)
        self.assertIsNotNone(sink.finish()['capture_failure'])
        self.assertEqual(list(sink.output.root.iterdir()), [])

    def test_failure_results_and_all_unrun_iterations_are_accounted(self):
        for enough, expected in ((True, 'ransac_unavailable'), (False, 'insufficient_retained_correspondences')):
            sink = self.sink(str(enough))
            result, calls = self.refine(sink, ransac_failure=True, enough=enough)
            self.assertIsNone(result)
            report = sink.finish()
            self.assertIsNone(report['capture_failure'])
            self.assertEqual(report['iterations'][0]['status'], expected)
            self.assertEqual([r['status'] for r in report['iterations'][1:]], ['unrun'] * 4)
            self.assertIn((0, 'result'), sink.events)

    def test_malformed_failed_solver_indices_do_not_change_return(self):
        self.assertIsNone(self.refine(malformed_ransac=True)[0])
        sink = self.sink()
        self.assertIsNone(self.refine(sink, malformed_ransac=True)[0])
        self.assertIn('solver-index snapshot', sink.failure)
        self.assertIn('solver-index snapshot', sink.finish()['capture_failure'])

    def test_missing_result_malformed_mapping_and_descriptor_refuse_completion(self):
        sink = self.sink()
        self.assertIsNotNone(sink.finish()['capture_failure'])
        sink = self.sink('malformed')
        with self.assertRaises(ValueError):
            sink.preflight({}, 'retained', {'iteration': 0}, {'bad': {'nbytes': 8}})
        sink = self.sink('mapping')
        sink._geometry[0] = dict(sample_ids=np.array([3, 8]), sample_obj_points_mm=np.zeros((2, 3)), sample_target_crop_px=np.zeros((2, 2)), crop_k=np.eye(3))
        with self.assertRaisesRegex(ValueError, 'map into retained'):
            sink(dict(kind='result', context={}, metadata=dict(iteration=0, fit_state='ransac_unavailable'),
                arrays=dict(ransac_inlier_ids=np.array([1]), retained_inlier_ids=np.array([1]))))

    def test_capability_type_sink_geometry_and_postconstruction_guards(self):
        sink = self.sink()
        with self.assertRaises(ValueError):
            GoTrackRefiner(None, None, 1, 'cpu', inference_geometry=inference.PROFILE, chronological_capture_callback=sink)
        with self.assertRaises(ValueError):
            GoTrackRefiner(None, None, 1, 'cpu', inference_geometry=inference.PROFILE, inference_observer_capability=object())
        with self.assertRaises(ValueError): self.capability(sink).validate(self.sink('other'), inference.PROFILE)
        with self.assertRaises(ValueError): self.capability(sink).validate(sink, 'legacy')
        refiner = GoTrackRefiner(None, None, 1, 'cpu', inference_geometry=inference.PROFILE)
        refiner.chronological_capture_callback = sink
        with self.assertRaisesRegex(ValueError, 'missing'): refiner.refine(None, None, None)
        refiner.inference_geometry = 'legacy'
        with self.assertRaisesRegex(ValueError, 'cannot be changed'): refiner.refine(None, None, None)
        with self.assertRaisesRegex(ValueError, 'completed exact'):
            probe._issue(b'{}', probe._sha(b'{}'), sink, None)

    def test_known_reprojection_and_conditioning_fixtures(self):
        # Unit square at 1 metre, in mm, projects exactly through this K.
        xyz = np.array([[0., 0., 1000.], [100., 0., 1000.], [0., 100., 1000.], [100., 100., 1000.]])
        k = np.diag([100., 100., 1.])
        uv = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        self.assertEqual(probe.reprojection_summary(xyz, uv, k, np.zeros(3), np.zeros(3))['rms_px'], 0.)
        self.assertEqual(probe.reprojection_summary(xyz, uv + [3., 4.], k, np.zeros(3), np.zeros(3))['rms_px'], 5.)
        self.assertEqual(probe.conditioning_summary(xyz)['rank'], 2)
        np.testing.assert_allclose(probe.conditioning_summary(xyz)['singular_values_mm'], [100., 100., 0.])

    def test_raw_budget_is_uncompressed_and_output_never_overwrites(self):
        output = probe.OutputBudget(self.root, {**probe.LIMITS, 'max_raw_capture_bytes': 100})
        output.write('small.zip', b'compressed', raw=True, raw_bytes=90)
        self.assertEqual(output.raw_remaining, 10)
        with self.assertRaises(ValueError): output.write('extra.zip', b'x', raw=True, raw_bytes=11)
        with self.assertRaises(FileExistsError): output.write('small.zip', b'x', raw=True, raw_bytes=1)
        with self.assertRaises(ValueError): output.write('../outside', b'x')

    def test_import_and_cloud_guard_do_not_import_torch_or_read_plan(self):
        self.assertNotIn('torch', sys.modules)
        with patch.object(probe.platform, 'system', return_value='Windows'), \
             self.assertRaisesRegex(ValueError, 'cloud-execution'):
            probe.run(self.root / 'missing', '0' * 64, self.root / 'output', cloud_execution=True)
        self.assertNotIn('torch', sys.modules)

    def test_historical_capture_contract_closure_changes_only_refiner(self):
        old = json.loads((probe.ROOT / inference.EXPANDED_CONTRACT_RELATIVE_PATH).read_bytes())
        # The measured ROOT23 probe remains immutable; this checkout now uses
        # a separate contract for the additional typed capture capability.
        historical_probe = probe.ROOT / 'bench/runtime/gotrack-inference-probe-v1.json'
        self.assertEqual(inference._sha256_file(historical_probe),
            'a7b5d3d2722a7f4070a0e807123cc572b60a9a7eca713082c2e4c41e3a5e99ee')
        historical = json.loads(historical_probe.read_bytes())
        self.assertEqual(historical['additional_runtime_files']['bench/quality_gotrack.py'],
            '5b451f87dfce9161210e1b58754b78747885666db3624fcb4119d6c89d04aaf3')
        refined_path = probe.ROOT / 'bench/runtime/gotrack-refinement-states-v1.json'
        self.assertEqual(inference._sha256_file(refined_path),
            '453af3dcf27e4e375bef9fc6d00b547765875a0ab71d1977e0905c264755b3d1')
        new = json.loads(refined_path.read_bytes())
        inference._validate_contract(new)
        changed = [name for name in old['additional_runtime_files'] if old['additional_runtime_files'][name] != new['additional_runtime_files'][name]]
        self.assertEqual(changed, ['bench/quality_gotrack.py'])
        self.assertNotEqual(old['additional_runtime_files']['bench/quality_gotrack.py'], inference._sha256_file(probe.ROOT / 'bench/quality_gotrack.py'))
        for name, expected in new['additional_runtime_files'].items():
            # ROOT25's archived numerical profile predates the new typed
            # observer and optional scratch CNOS bank. Current source pins are
            # checked separately by the larger-runtime integration test.
            if name in {'bench/quality_gotrack.py', 'bench/quality_cnos.py'}:
                continue
            self.assertEqual(inference._sha256_file(probe.ROOT / name), expected, name)
        self.assertEqual(new['source_closure_sha256'], inference._source_closure_hash(new['unchanged_training_files'], new['reviewed_adapter_replacements'], new['additional_runtime_files']))
        self.assertEqual(inference._sha256_file(probe.ROOT / inference.CONTRACT_RELATIVE_PATH), inference.LEGACY_CONTRACT_SHA256)

    def test_legacy_chronological_event_sequence_has_no_new_stage(self):
        class LegacySink:
            def __init__(self): self.kinds = []
            def should_capture(self, *args): return True
            def preflight(self, *args): return True
            def preflight_nbytes(self, *args): return True
            def __call__(self, packet): self.kinds.append(packet['kind'])
        sink = LegacySink()
        candidate, calls = self.refine(sink, geometry='legacy')
        self.assertIsNotNone(candidate)
        self.assertEqual(sink.kinds, ['trace', 'retained', 'result'] * 5)

    def test_source_drift_fails_capability_before_inference(self):
        sink = self.sink()
        capability = self.capability(sink)
        with patch.object(inference, 'verify_source_compatibility', side_effect=ValueError('source drift')), \
             self.assertRaisesRegex(ValueError, 'source drift'):
            capability.validate(sink, inference.PROFILE)

    def plan_fixture(self):
        cases = []
        for index in range(6):
            seed = np.eye(4); seed[2, 3] = 1
            cases.append(dict(case_id=f'case-{index}', sequence_id=f'sequence-{index}', frame_id=index,
                timestamp_s=index / 10, seed_camera_from_object_m=seed.tolist()))
        return dict(cases=cases, limits=copy.deepcopy(probe.LIMITS))

    def run_fixture(self, *, decode_failure=False, close_failure=False, parity_failure=False, load_failure=False):
        from . import quality_gotrack as gotrack, quality_runner as runner
        plan = self.plan_fixture()
        data = probe._encoded(plan)
        plan_path = self.root / 'plan.json'; plan_path.write_bytes(data)
        target = self.root / 'run'
        contexts, inputs = {}, {}
        for case in plan['cases']:
            for condition in probe.CONDITIONS:
                contexts[case['case_id'], condition] = types.SimpleNamespace(effective_intrinsics=np.eye(3))
            inputs[case['case_id']] = (self.root, dict(asset='object.glb', object_id=17), self.root / 'mask.png')
        models, renderers = [], []
        def load(device):
            if load_failure: raise RuntimeError('model load fixture failure')
            model = types.SimpleNamespace(released=False)
            model.cpu = lambda: setattr(model, 'released', True)
            models.append(model)
            return model
        def renderer(*args, **kwargs):
            value = types.SimpleNamespace(closed=False)
            def close():
                value.closed = True
                if close_failure: raise RuntimeError('close fixture failure')
            value.close = close
            renderers.append(value)
            return value
        class Refiner:
            def __init__(this, *args, **kw): this.sink = kw['chronological_capture_callback']
            def set_chronological_capture_context(this, context): this.context = context
            def refine(this, frame, mask, seed):
                if this.sink:
                    sink = this.sink
                    for i in range(5):
                        meta = dict(iteration=i, fit_state='refined')
                        geometry = dict(sample_obj_points_mm=np.array([[0., 0., 1000.], [1., 0., 1000.], [0., 1., 1000.]]),
                            sample_target_crop_px=np.array([[0., 0.], [.001, 0.], [0., .001]]), crop_k=np.eye(3), sample_ids=np.arange(3))
                        for kind, arrays in (('trace', {}), ('retained', geometry),
                            ('pre_lm', dict(ransac_inlier_ids=np.arange(3), retained_inlier_ids=np.arange(3), ransac_rvec=np.zeros(3), ransac_tvec_mm=np.zeros(3))),
                            ('result', dict(ransac_inlier_ids=np.arange(3), retained_inlier_ids=np.arange(3), refined_rvec=np.zeros(3), refined_tvec_mm=np.zeros(3)))):
                            sink(dict(kind=kind, metadata=meta, context=this.context, arrays=arrays))
                pose = seed.copy()
                if parity_failure and this.sink: pose[0, 3] += 1
                return types.SimpleNamespace(pose=pose, points_object_m=np.zeros((3, 3)), pixels_image=np.zeros((3, 2)), weights=np.ones(3))
        def read_rgb(bundle, manifest, frame_id):
            if decode_failure and frame_id == 2: raise ValueError('decode fixture failure')
            return np.zeros((4, 4, 3), np.uint8)
        torch = types.SimpleNamespace(manual_seed=lambda value: None, set_num_threads=lambda value: None,
            get_num_threads=lambda: 1,
            use_deterministic_algorithms=lambda value: None, are_deterministic_algorithms_enabled=lambda: True,
            cuda=types.SimpleNamespace(is_initialized=lambda: False))
        with patch.object(probe, '_cloud_guard'), patch.object(probe, 'validate_plan', return_value=plan), \
             patch.object(probe, '_preflight', return_value=(contexts, inputs)), patch.object(probe, '_issue', return_value=object()), \
             patch.dict(sys.modules, {'torch': torch}), patch.object(gotrack, 'load_network', side_effect=load), \
             patch.object(gotrack, 'TexturedRenderer', side_effect=renderer), patch.object(gotrack, 'GoTrackRefiner', Refiner), \
             patch.object(runner, 'read_rgb', side_effect=read_rgb), patch.object(inference, 'install_adapted_head') as install, \
             patch('bench.vision.cv2.imread', return_value=np.ones((4, 4), np.uint8)):
            result = probe.run(plan_path, probe._sha(data), target, cloud_execution=True)
            with self.assertRaises(FileExistsError): probe.run(plan_path, probe._sha(data), target, cloud_execution=True)
        return result, target, models, renderers, install.call_count

    def test_driver_runs_all_24_trajectories_with_parity_shared_inputs_and_cleanup(self):
        report, target, models, renderers, installs = self.run_fixture()
        self.assertTrue(report['complete'])
        self.assertEqual(len(report['rows']), 24)
        self.assertEqual(sum(len(row['iterations']) for row in report['rows']), 120)
        self.assertEqual(installs, 1)
        self.assertEqual(len(models), 2)
        self.assertTrue(all(model.released for model in models))
        self.assertEqual(len(renderers), 24)
        self.assertTrue(all(renderer.closed for renderer in renderers))
        self.assertTrue((target / 'completion.json').is_file())
        self.assertTrue(all(row.get('observer_parity', row.get('cross_head_input_seed_equality')) for row in report['parity']))

    def test_driver_decode_failure_preserves_requested_inventory_and_other_cases(self):
        report, target, models, renderers, installs = self.run_fixture(decode_failure=True)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['rows']), 24)
        failed = [row for row in report['rows'] if row['status'] == 'failed']
        self.assertEqual(len(failed), 4)
        self.assertTrue(all('decode fixture failure' in row['failure_reason'] for row in failed))
        self.assertEqual(sum(row['status'] == 'available' for row in report['rows']), 20)
        self.assertFalse((target / 'completion.json').exists())

    def test_driver_parity_failure_never_seals(self):
        report, target, *_ = self.run_fixture(parity_failure=True)
        self.assertFalse(report['complete'])
        self.assertFalse((target / 'completion.json').exists())

    def test_driver_close_failure_and_load_failure_have_inventory_and_no_seal(self):
        report, target, models, renderers, _ = self.run_fixture(close_failure=True)
        self.assertFalse(report['complete'])
        self.assertTrue(all(row['cleanup'] == 'failed' for row in report['rows']))
        self.assertTrue(all(model.released for model in models))
        self.assertFalse((target / 'completion.json').exists())
        self.root = self.root / 'load'; self.root.mkdir()
        report, target, *_ = self.run_fixture(load_failure=True)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['rows']), 24)
        self.assertTrue(all('model load fixture failure' in row['failure_reason'] for row in report['rows']))
        self.assertEqual(sum(len(row['iterations']) for row in report['rows']), 120)
        self.assertFalse((target / 'completion.json').exists())

    def test_plan_rejects_annotation_paths_extra_fields_and_seed_changes(self):
        inventory = []
        cases = []
        for index in range(6):
            seed = np.eye(4); seed[2, 3] = 1
            base = dict(case_id=f'case{index}', sequence_id=f'seq{index}', frame_id=index, timestamp_s=index / 10,
                        seed_camera_from_object_m=seed.tolist())
            inventory.append(copy.deepcopy(base))
            case = copy.deepcopy(base)
            for name, filename in (('input', 'input.json'), ('video', 'synthetic.mp4'), ('mesh', 'object.glb'), ('mask_index', 'results.json'), ('mask', f'{index}.png')):
                case[name] = dict(path=f'.cache/inputs/{filename}', bytes=1, sha256='a' * 64)
            cases.append(case)
        path = self.root / 'inventory.json'; path.write_bytes(probe._encoded(dict(cases=inventory)))
        plan = dict(schema=probe.SCHEMA, cases=cases, limits=copy.deepcopy(probe.LIMITS),
            runtime_contract=self.pin('bench/runtime/gotrack-inference-probe-v1.json'), probe_sources=[self.pin('bench/quality_gotrack_probe.py')],
            seed_inventory=dict(path='inventory.json', bytes=1, sha256=probe.INVENTORY_SHA256),
            design=dict(path='design.json', bytes=1, sha256=probe.DESIGN_SHA256),
            head=dict(path='head.pt', bytes=inference.EXPANDED_EXPORT_BYTES, sha256=inference.EXPANDED_EXPORT_SHA256))
        with patch.object(probe, 'pinned_file', return_value=path):
            probe.validate_plan(plan)
            for change in ('reference', 'generation', 'evaluation', 'obj_000007'):
                changed = copy.deepcopy(plan)
                changed['cases'][0]['mask']['path'] = f'.cache/{change}/0.png'
                with self.assertRaisesRegex(ValueError, 'forbidden'): probe.validate_plan(changed)
            changed = copy.deepcopy(plan); changed['cases'][0]['reference_pose'] = np.eye(4).tolist()
            with self.assertRaisesRegex(ValueError, 'unexpected'): probe.validate_plan(changed)
            changed = copy.deepcopy(plan); changed['cases'][0]['seed_camera_from_object_m'][0][3] = 1
            with self.assertRaisesRegex(ValueError, 'authenticated inventory'): probe.validate_plan(changed)
            changed = copy.deepcopy(plan); changed['design']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'inventory/design identity'): probe.validate_plan(changed)

    def test_preflight_rejects_unapproved_input_before_shared_reader_or_data_hash(self):
        from . import quality_runner as runner
        runtime = self.pin('bench/runtime/gotrack-inference-probe-v1.json')
        plan = dict(runtime_contract=runtime, probe_sources=[self.pin('bench/quality_gotrack_probe.py')],
            head=dict(path='head.pt'), limits=probe.LIMITS, cases=[dict(sequence_id='unapproved',
                input=dict(sha256='0' * 64), video=dict(sha256='0' * 64), mesh=dict(sha256='0' * 64))])
        visited = []
        def pin(record, **kw):
            visited.append(record['path'])
            return probe.ROOT / runtime['path']
        with patch.object(probe, 'pinned_file', side_effect=pin), \
             patch.object(inference, 'verify_source_compatibility'), \
             patch('bench.quality_offline.tree_bytes', return_value=0), \
             patch.object(runner, 'read_input') as reader, \
             self.assertRaisesRegex(ValueError, 'accepted-input allowlist'):
            probe._preflight(plan)
        reader.assert_not_called()
        self.assertEqual(visited, [runtime['path'], 'head.pt', 'bench/quality_gotrack_probe.py'])

    def test_supervisor_preserves_all_rows_on_timeout_and_rejects_forged_completion(self):
        from . import quality_offline as offline
        for forged in (False, True):
            directory = self.root / str(forged)
            directory.mkdir()
            plan = self.plan_fixture()
            plan.update(runtime_contract=self.pin('bench/runtime/gotrack-inference-probe-v1.json'),
                        probe_sources=[self.pin('bench/quality_gotrack_probe.py')])
            data = probe._encoded(plan); path = directory / 'plan.json'; path.write_bytes(data)
            output = directory / 'output'
            def stage(name, command, destination, **kwargs):
                self.assertEqual(command[command.index('-m') + 1], 'bench.quality_gotrack_probe')
                self.assertTrue(0 < kwargs['timeout_s'] <= 1800)
                self.assertEqual(kwargs['storage_limit_bytes'], 8 * 1024**3)
                if not forged:
                    (destination / 'probe.receipt.json').write_bytes(probe._encoded(dict(status='failed', failure_reason='runtime_limit')))
                    raise RuntimeError('runtime_limit')
                child = destination / 'capture'; child.mkdir()
                (child / 'receipt.json').write_bytes(b'{}')
                (child / 'completion.json').write_bytes(probe._encoded(dict(receipt_sha256='0' * 64)))
                return dict(status='complete')
            with patch.object(probe, '_cloud_guard'), patch.object(probe, 'validate_plan', return_value=plan), \
                 patch.object(inference, 'verify_source_compatibility'), patch.object(offline, 'run_stage', side_effect=stage):
                report = probe.supervise(path, probe._sha(data), output, cloud_execution=True)
                self.assertFalse(report['complete'])
                self.assertEqual(len(report['rows']), 24)
                self.assertEqual(sum(len(row['iterations']) for row in report['rows']), 120)
                self.assertIn('authenticate receipt' if forged else 'runtime_limit', report['failure'])
                with self.assertRaises(FileExistsError): probe.supervise(path, probe._sha(data), output, cloud_execution=True)

    def test_existing_supervisor_terminates_a_bounded_non_neural_child(self):
        from .quality_offline import run_stage
        output = self.root / 'stage'; output.mkdir()
        cache = self.root / 'cache'; cache.mkdir()
        # The science runtime deliberately has no psutil installation. Supply
        # process-inspection doubles; Popen, timeout and real child termination
        # remain the existing runner's actual implementation.
        class ProcessInspection:
            def __init__(self, pid): self.pid = pid
            def children(self, recursive=True): return []
            def memory_info(self): return types.SimpleNamespace(rss=1024)
        psutil = types.SimpleNamespace(Process=ProcessInspection, NoSuchProcess=ProcessLookupError,
            AccessDenied=PermissionError, wait_procs=lambda values, timeout: ([], []))
        with patch.dict(sys.modules, {'psutil': psutil}), self.assertRaisesRegex(RuntimeError, 'runtime_limit'):
            run_stage('probe', [sys.executable, '-B', '-c', 'import time; time.sleep(30)'], output,
                timeout_s=.1, storage_limit_bytes=1024, storage_root=cache, output_limit_bytes=1024**2)
        receipt = json.loads((output / 'probe.receipt.json').read_bytes())
        self.assertEqual(receipt['failure_reason'], 'runtime_limit')
        self.assertLess(receipt['elapsed_s'], 10)

    def test_disk_reserve_refuses_before_preflight_or_torch(self):
        plan = self.plan_fixture(); data = probe._encoded(plan)
        path = self.root / 'plan.json'; path.write_bytes(data)
        with patch.object(probe, '_cloud_guard'), patch.object(probe, 'validate_plan', return_value=plan), \
             patch.object(probe.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1)), \
             patch.object(probe, '_preflight') as preflight, self.assertRaisesRegex(ValueError, '1 GiB'):
            probe.run(path, probe._sha(data), self.root / 'output')
        preflight.assert_not_called()
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__': unittest.main()
