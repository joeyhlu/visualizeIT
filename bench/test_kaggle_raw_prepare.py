"""Model-free orchestration fixtures. These doubles establish no model accuracy.

Source authentication, input JSON parsing, video decoding, artifact writes,
capability issuance/validation, retained crop conversion and sealing are real.
Only renderer/neural computations and returned-camera measurements are fixtures.
"""
from contextlib import contextmanager, ExitStack
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_adaptation_portable as portable
from . import quality_adaptation_larger as larger
from . import quality_adaptation_larger_optimized as optimized
from . import quality_adaptation_refinement_states as states
from .quality_gotrack import GoTrackRefiner
from .quality_ranker_synthetic import make_input_manifest
from .test_quality_adaptation_refinement_states import arrays_fixture
from .vision import cv2


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def fixture_stage(context, name):
    """Only for tiny doubles: never a production or cloud admission fixture."""
    token = portable._EXECUTION.set((context.sha256, name))
    try:
        yield
    finally:
        portable._EXECUTION.reset(token)


class TinyNetwork:
    backbone = object()
    def eval(self):
        return self
    def cpu(self):
        return self


class TinyRenderer:
    def __init__(self, path, obj_id, **kwargs):
        self.asset_sha256 = portable.sha256(path)
        self.closed = False
    def close(self):
        self.closed = True


class TinySearch:
    def __init__(self, **kwargs):
        self.closed = False
    def bind(self, refiner, **kwargs):
        return self
    def recover(self, frame, mask, *, top_k):
        if top_k != 5 or not mask.any() or frame.timestamp_s != 0.:
            raise AssertionError('changed production recovery boundary')
        pose = np.eye(4); pose[2, 3] = 1.
        return [pose]
    def close(self):
        self.closed = True


class TinyRefiner(GoTrackRefiner):
    """Keeps the actual unchanged constructor and genuine capability validator."""
    def refine(self, frame, mask, seed):
        if self.chronological_capture_callback is None:
            raise AssertionError('missing retained sink')
        sink = self.chronological_capture_callback
        for iteration in sink.physical['iterations']:
            arrays = arrays_fixture(); arrays['native_k'] = frame.intrinsics
            sink(dict(kind='retained', context=self._chronological_capture_context,
                metadata=dict(iteration=iteration, frame_id=0), arrays=arrays))
            sink(dict(kind='result', context=self._chronological_capture_context,
                metadata=dict(iteration=iteration, frame_id=0, fit_state='fixture_solver_unavailable'), arrays={}))


class TinyMasks(optimized.NeuralMaskBackend):
    """Actual phase lifecycle and input/output seams; model responses are fixtures."""
    releases = []
    def load(self, phase):
        pass
    def release(self, phase):
        self.releases.append(phase)
    def proposals(self, view):
        self.input(view)
        masks = np.zeros((1, 480, 640), np.float32); masks[:, 20:60, 20:60] = 1.
        return np.array([[20, 20, 59, 59]]), masks, None
    def scores(self, view, boxes, masks):
        self.input(view)
        return [.9], self.bank_hash
    def segment(self, view, box):
        self.input(view)
        mask = np.zeros((480, 640), bool); mask[20:60, 20:60] = True
        return mask


def fixture_render(mesh, output, group, **kwargs):
    """Produce one real decodable input per object; retain 23 render failures."""
    if kwargs['view_index'] != 0:
        raise ValueError('fixture_render_unavailable')
    if kwargs['view_mode'] != 'normalized24-v1' or kwargs['frames'] != 1:
        raise AssertionError('changed existing renderer math boundary')
    output = Path(output); bundle = output / 'bundle'; bundle.mkdir(parents=True)
    (bundle / 'object.glb').write_bytes(Path(mesh).read_bytes())
    writer = cv2.VideoWriter(str(bundle / 'synthetic.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10., (640, 480))
    try:
        if not writer.isOpened():
            raise RuntimeError('fixture codec unavailable')
        rgb = np.zeros((480, 640, 3), np.uint8); rgb[20:60, 20:60] = 200
        writer.write(rgb)
    finally:
        writer.release()
    manifest = make_input_manifest(mesh_sha256=portable.sha256(mesh),
        video_sha256=portable.sha256(bundle / 'synthetic.mp4'), frames=1,
        width=640, height=480, fps=10., sequence_group=group)
    (bundle / 'input.json').write_bytes(portable.encoded(manifest))
    # This path must never be read by inference or the raw seal.
    (output / 'generation.json').write_text('{"fixture_truth":"forbidden"}')
    return dict(test_fixture=True)


class PortableTests(unittest.TestCase):
    def setUp(self):
        temp_root = portable.ROOT / '.cache/cloud-prepare/test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); self.root = self.base / 'run'
        # Model-free filesystem fixture, not a production cache-admission assertion.
        # Keep actual source/checkpoint path authentication intact; only account a
        # tiny stand-in storage tree instead of traversing historical experiments.
        accounting = self.base / 'fixture-cache-accounting'; accounting.mkdir()
        (accounting / 'existing-source-fixture').write_bytes(b'fixture-cache')
        real_usage = portable._usage
        actual_cache = (portable.ROOT / '.cache/model-quality').resolve()
        def fixture_usage(roots):
            return real_usage([accounting if Path(root).resolve() == actual_cache else root for root in roots])
        patcher = patch.object(portable, '_usage', fixture_usage)
        patcher.start(); self.addCleanup(patcher.stop)
        self.spec = dict(schema='kaggle_raw_recipe_v1', assets=[], iterations=[0], seed=42, targets=[1000, 2000, 4000])
        for obj in range(1, 4):
            mesh = self.base / f'{obj}.glb'; mesh.write_bytes(f'testfixture-only-mesh-{obj}'.encode())
            self.spec['assets'].append(dict(object_id=str(obj), physical_object_id='independent:' + str(obj),
                aliases=[], mesh=portable.pin(mesh)))
        dummy = dict(path=(self.base / 'not-opened-parity.json').as_posix(), bytes=1, sha256='a' * 64)
        doc = portable.context_document(output_root=self.root, scratch_root=self.base / 'scratch',
            parity=dict(reference=dummy, candidate=dummy), device='cpu', reserve_bytes=1)
        # Pretty/newline input exercises exact-byte preservation.
        raw = (json.dumps(doc, indent=2) + '\n').encode()
        self.context = portable.PortableContext(raw, digest(raw))
        self.recipe_path = self.base / 'recipe-source.json'
        self.recipe_path.write_text(json.dumps(self.spec, indent=2) + '\n', encoding='utf-8')

    def freeze(self):
        return portable.freeze(self.recipe_path, portable.sha256(self.recipe_path), self.context)

    def render(self):
        with fixture_stage(self.context, 'render'), patch('bench.quality_ranker_synthetic._render_bundle', fixture_render):
            portable.render(self.context)

    def bank_fixture(self):
        """Bank metadata represents controlled tiny neural responses, never production evidence."""
        def descriptor(bundle, device, *, unlit, output):
            output.parent.mkdir(exist_ok=True)
            np.savez(output, asset_sha256=portable.sha256(bundle / 'object.glb'), descriptors=np.ones((5, 4), np.float32))
            return output
        def pose_bank(renderer, backbone, device, path):
            path.write_bytes(b'fixture-pose-bank')
            return dict(template_count=798, asset_sha256=renderer.asset_sha256)
        with fixture_stage(self.context, 'banks'), \
                patch.object(larger, 'configure_torch', return_value=SimpleNamespace()), \
                patch('bench.quality_gotrack.load_network', return_value=TinyNetwork()), \
                patch('bench.quality_gotrack.TexturedRenderer', TinyRenderer), \
                patch('bench.quality_cnos.build_descriptors', descriptor), \
                patch('bench.quality_foundpose.build_bank', pose_bank):
            portable.banks(self.context)

    def masks(self):
        fake_torch = SimpleNamespace(manual_seed=lambda value: None)
        with fixture_stage(self.context, 'masks'), patch.object(larger, 'configure_torch', return_value=fake_torch), \
                patch.object(optimized, 'NeuralMaskBackend', TinyMasks):
            portable.masks(self.context)

    def prepared(self):
        self.freeze(); self.render(); self.bank_fixture(); self.masks()
        return portable.frozen(self.context)[3]

    def capture(self, *, interrupt=False):
        fake_torch = SimpleNamespace(manual_seed=lambda value: None)
        @contextmanager
        def sampling(selected):
            yield {i: dict(integer_centers=280**2, max_native_pixel_difference=0.) for i in selected}
        with ExitStack() as stack:
            stack.enter_context(fixture_stage(self.context, 'capture'))
            stack.enter_context(patch.object(larger, 'configure_torch', return_value=fake_torch))
            stack.enter_context(patch('bench.quality_gotrack.load_network', return_value=TinyNetwork()))
            stack.enter_context(patch('bench.quality_gotrack.TexturedRenderer', TinyRenderer))
            stack.enter_context(patch('bench.quality_gotrack.GoTrackRefiner', TinyRefiner))
            stack.enter_context(patch.object(optimized, 'PreparedFoundPoseSearch', TinySearch))
            stack.enter_context(patch.object(states, 'observe_selected_crops', sampling))
            if interrupt:
                stack.enter_context(patch.object(TinyRefiner, 'refine', side_effect=KeyboardInterrupt('fixture interruption')))
            return portable.capture(self.context)

    def test_pretty_context_and_recipe_exact_roundtrip(self):
        self.freeze(); portable.frozen(self.context)
        self.assertEqual((self.root / 'context.json').read_bytes(), self.context.raw)
        self.assertEqual((self.root / 'recipe.json').read_bytes(), self.recipe_path.read_bytes())

    def test_external_metadata_reads_are_bounded_before_parsing(self):
        class LimitStream:
            def __enter__(inner):
                return inner
            def __exit__(inner, *args):
                pass
            def read(inner, size):
                self.assertEqual(size, portable.MAX_METADATA + 1)
                return b'x' * size
        with patch.object(Path, 'open', return_value=LimitStream()):
            with self.assertRaisesRegex(ValueError, 'bounded read'):
                portable.read_context(self.recipe_path, 'a' * 64)

    def test_inventory_24_unique_views_no_payload_opens_and_target_honesty(self):
        with patch.object(Path, 'open', side_effect=AssertionError('payload opened')):
            plan = portable.prospective(self.spec)
        self.assertEqual(len(plan['global_manifest']['records']), 72)
        self.assertTrue(plan['feasible'])
        self.assertFalse(any(r['request_met_by_inventory'] for r in plan['target_requests']))

    def test_enough_independent_assets_can_meet_4000_without_repeated_views(self):
        spec = copy.deepcopy(self.spec); spec['iterations'] = [0, 1, 3]
        for obj in range(4, 60):
            value = digest(str(obj).encode())
            spec['assets'].append(dict(object_id=str(obj), physical_object_id=f'independent:{obj}',
                aliases=[], mesh=dict(path=f'/metadata-only/{obj}.glb', bytes=1, sha256=value)))
        plan = portable.prospective(spec)
        self.assertEqual(plan['counts']['train']['refinement_states'], 4104)
        self.assertTrue(all(r['request_met_by_inventory'] for r in plan['target_requests']))

    def test_large_accepted_inventory_freeze_roundtrip_without_payload_opens(self):
        spec = copy.deepcopy(self.spec); spec['iterations'] = [0, 1, 3]
        spec['assets'] = [dict(object_id=str(obj), physical_object_id=f'independently-audited:obj{obj}',
            aliases=[], mesh=dict(path=f'/metadata-only/obj{obj}.glb', bytes=1,
                sha256=digest(str(obj).encode()))) for obj in range(100, 238)]
        self.recipe_path.write_bytes(portable.encoded(spec))
        original_open = Path.open
        def deny_payload(path, *args, **kwargs):
            if path.suffix.lower() in ('.glb', '.mp4', '.npz', '.pt', '.png') or path.name == 'generation.json':
                raise AssertionError('prospective payload opened: ' + str(path))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', deny_payload):
            receipt = self.freeze()
            doc, frozen_spec, plan, rows = portable.frozen(self.context)
        self.assertEqual(receipt['refinement_states'], 9936)
        self.assertEqual(receipt['physical_frames'], 3312)
        self.assertGreater((self.root / 'scale-plan.json').stat().st_size, 8 * 1024**2)
        self.assertLessEqual((self.root / 'scale-plan.json').stat().st_size, portable.MAX_METADATA)
        self.assertEqual(frozen_spec, spec)
        self.assertEqual(len(plan['global_manifest']['records']), 9936)
        self.assertEqual(len(rows), 3312)
        self.assertEqual(plan['global_manifest']['split_manifest'], portable.read_json(self.root / 'split.json'))
        self.assertTrue(all(row['render'] == {'status': 'pending'} for row in rows))

    def test_portable_reader_retains_invalid_metadata_guards(self):
        path = self.base / 'invalid-metadata.json'
        for raw in (b'{"value":1,"value":2}', b'{"value":NaN}', b'{"value":1e999}',
                b'[' * 33 + b'null' + b']' * 33):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    portable.read_json(path)
        reads = []
        class LimitStream(io.BytesIO):
            def read(inner, size=-1):
                reads.append(size)
                return super().read(size)
        with patch.object(Path, 'open', return_value=LimitStream(b'x' * (portable.MAX_METADATA + 1))):
            with self.assertRaisesRegex(ValueError, 'bounded read'):
                portable.read_json(path)
        self.assertEqual(reads, [portable.MAX_METADATA + 1])

    def test_duplicate_mesh_and_new_iterations_rejected(self):
        spec = copy.deepcopy(self.spec); spec['assets'][1]['mesh'] = spec['assets'][0]['mesh']
        with self.assertRaisesRegex(ValueError, 'repeated metric'):
            portable.prospective(spec)
        spec = copy.deepcopy(self.spec); spec['iterations'] = [2]
        with self.assertRaisesRegex(ValueError, '0/1/3'):
            portable.prospective(spec)

    def test_development_aliases_are_metadata_only(self):
        spec = copy.deepcopy(self.spec); spec['assets'][0]['aliases'] = ['keyboard']
        plan = portable.prospective(spec)
        rows = portable.physical_rows(spec, plan)
        self.assertTrue(all(r['split'] == 'dev' for r in rows))  # two remaining independent groups are infeasible

    def test_altered_source_context_and_unknown_fields_fail(self):
        doc = self.context.document(); doc['source_files']['bench/quality_gotrack.py'] = 'a' * 64
        raw = portable.encoded(doc)
        with self.assertRaisesRegex(ValueError, 'executing source changed'):
            portable.verify_sources(portable.PortableContext(raw, digest(raw)))
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            replace(self.context, sha256='a' * 64).document()
        doc = self.context.document(); doc['verified'] = True
        with self.assertRaisesRegex(ValueError, 'unknown/missing'):
            portable.validate_context(doc)

    def test_declared_cache_remapping_rejected(self):
        doc = self.context.document(); doc['model_cache'] = (self.base / 'other-cache').as_posix()
        with self.assertRaisesRegex(ValueError, 'actual existing loader'):
            portable.validate_context(doc)

    def test_changed_recipe_or_split_rejected(self):
        self.freeze(); spec = copy.deepcopy(self.spec); spec['seed'] += 1
        (self.root / 'recipe.json').write_bytes(portable.encoded(spec))
        with self.assertRaisesRegex(ValueError, 'recipe differs'):
            portable.frozen(self.context)

    def test_direct_neural_stage_and_local_cloud_execution_rejected(self):
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'explicit guarded stage'):
            portable.banks(self.context)
        with self.assertRaisesRegex(RuntimeError, 'Kaggle Linux'):
            with portable.stage(self.context, 'banks', cloud_execution=False):
                self.fail('local neural admitted')

    def test_mandatory_stage_preflight_failure_records_every_requested_state(self):
        """Guard orchestration fixture; timer/RSS mechanics require real Linux validation."""
        self.freeze()
        portable.write_json(self.context, self.root / 'stage-masks.json',
            dict(context_sha256=self.context.sha256, elapsed_seconds=0., status='fixture'))
        original_dir, original_relative = Path.is_dir, Path.is_relative_to
        def is_dir(path):
            return True if path == Path('/kaggle/input') else original_dir(path)
        def is_relative(path, other):
            if (path == self.root and str(other) == '/kaggle/working') or (
                    path == self.base / 'scratch' and str(other) == '/tmp'):
                return True
            return original_relative(path, other)
        fake_signal = SimpleNamespace(SIGALRM=14, ITIMER_REAL=0, getsignal=lambda value: None,
            getitimer=lambda value: (0., 0.), signal=lambda *args: None, setitimer=lambda *args: None)
        fake_thread = SimpleNamespace(start=lambda: None, join=lambda **kwargs: None)
        from contextlib import nullcontext
        with patch.object(portable.platform, 'system', return_value='Linux'), \
                patch.object(Path, 'is_dir', is_dir), patch.object(Path, 'is_relative_to', is_relative), \
                patch.dict(portable.os.environ, VISUALIZEIT_CLOUD='1'), \
                patch.object(larger, 'stage_lock', nullcontext), patch.object(portable, 'signal', fake_signal), \
                patch.object(portable.threading, 'Thread', return_value=fake_thread), \
                patch.object(portable, 'verify_neural_environment', side_effect=RuntimeError('fixture missing checkpoint')) as verify:
            with self.assertRaisesRegex(RuntimeError, 'missing checkpoint'):
                with portable.stage(self.context, 'capture', cloud_execution=True):
                    self.fail('neural stage body entered before prerequisites')
        verify.assert_called_once_with(self.context)
        self.assertEqual(len(portable.read_json(self.root / 'raw/capture-index.json')['rows']), 72)
        self.assertIn('missing checkpoint', portable.read_json(self.root / 'stage-capture.json')['error'])
        self.assertNotIn('torch', sys.modules)

    def test_neural_prerequisites_fail_without_torch_import_or_download(self):
        self.assertNotIn('torch', sys.modules)
        with self.assertRaisesRegex(ValueError, 'bytecode|pre-existing model'):
            portable.verify_neural_environment(self.context)
        self.assertNotIn('torch', sys.modules)

    def test_missing_checkpoint_rejected_before_model_import(self):
        original_glob, original_file = Path.rglob, Path.is_file
        missing = portable.ROOT / '.cache/model-quality/checkpoints/gotrack_checkpoint.pt'
        def glob(path, pattern):
            return iter(()) if pattern == '*.pyc' else original_glob(path, pattern)
        def is_file(path):
            return False if path == missing else original_file(path)
        with patch.object(Path, 'rglob', glob), patch.object(Path, 'is_file', is_file):
            with self.assertRaisesRegex(ValueError, 'pre-existing model; no downloads'):
                portable.verify_neural_environment(self.context)
        self.assertNotIn('torch', sys.modules)

    def test_real_orchestration_writes_raw_seal_and_preserves_failures(self):
        rows = self.prepared()
        self.assertEqual(sum(r['render']['status'] == 'available' for r in rows), 2)
        self.assertEqual(sum(r['mask']['status'] == 'available' for r in rows), 2)
        result = self.capture(); self.assertEqual(result, dict(requested=72, available=2))
        # No truth sidecars may open during input preflight or raw sealing.
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'generation.json':
                raise AssertionError('ground truth opened')
            return original(path, *args, **kwargs)
        with fixture_stage(self.context, 'seal-capture'), patch.object(Path, 'open', guarded):
            seal = portable.seal_capture(self.context)
        self.assertEqual(len(seal['refinement_states']), 72)
        self.assertEqual(seal['available_crops'], 2)
        self.assertFalse(seal['training_ready']); self.assertFalse(seal['final_execution_inventory'])
        self.assertTrue(seal['prospective_inputs_are_recipe_identity_only'])
        self.assertTrue(all(r['status'] == 'unavailable' for r in seal['refinement_states'] if r['split'] == 'test'))
        self.assertNotIn('torch', sys.modules)

    def test_test_and_dev_payloads_denied_before_open(self):
        self.freeze(); row = portable.frozen(self.context)[3][0]
        for split in ('test', 'dev'):
            row = dict(row, split=split)
            with patch.object(Path, 'open', side_effect=AssertionError('payload opened')):
                with self.assertRaisesRegex(ValueError, 'denied'):
                    portable.read_observed_input(self.context, row)

    def test_last_sealed_row_payload_tamper_rejected_before_input_reads(self):
        self.freeze(); doc, spec, plan, rows = portable.frozen(self.context)
        row = next(r for r in rows if r['split'] == 'test'); row['render'] = dict(status='available', input='forbidden')
        portable._save_rows(self.context, rows)
        raw = portable.encoded(portable.capture_plan(self.context, rows))
        with patch('bench.quality_runner.read_input', side_effect=AssertionError('input opened')):
            with self.assertRaisesRegex(ValueError, 'TEST/dev'):
                portable.preflight_inputs(raw, digest(raw), self.context)

    def test_genuine_capability_unchanged_constructor_and_binding_failures(self):
        rows = self.prepared(); raw = portable.encoded(portable.capture_plan(self.context, rows))
        inputs = portable.preflight_inputs(raw, digest(raw), self.context)
        physical = next(r for r in rows if r['id'] in inputs)
        output = self.root / 'raw'; output.mkdir()
        sink = states.RefStateCapture(output, physical, states.state_rows(dict(rows=rows)), {}, inputs[physical['id']][2],
            max_output_bytes=self.context.document()['limits']['storage_bytes'], reserve_bytes=1)
        capability = larger.issue_portable_observer(raw, digest(raw), sink, context=self.context)
        GoTrackRefiner(None, None, 1, 'cpu', chronological_capture_callback=sink,
            inference_geometry=larger.inference.PROFILE, inference_observer_capability=capability)
        for changed in (replace(capability, _issuer=larger._ISSUER), replace(capability, _policy='legacy-v1'),
                replace(capability, _witness=True), replace(capability, _raw=raw + b' ')):
            with self.assertRaises(ValueError):
                changed.validate(sink, larger.inference.PROFILE)
        with self.assertRaises(ValueError):
            capability.validate(lambda value: None, larger.inference.PROFILE)
        other = states.RefStateCapture(output, physical, states.state_rows(dict(rows=rows)), {}, inputs[physical['id']][2],
            max_output_bytes=self.context.document()['limits']['storage_bytes'], reserve_bytes=1)
        with self.assertRaises(ValueError):
            capability.validate(other, larger.inference.PROFILE)
        with self.assertRaises(ValueError):
            capability.validate(sink, 'legacy')
        sink.native_k[0, 2] += 1
        with self.assertRaisesRegex(ValueError, 'geometry'):
            capability.validate(sink, larger.inference.PROFILE)

    def test_interrupted_capture_preserves_inventory_and_cleanup(self):
        self.prepared()
        with self.assertRaises(KeyboardInterrupt):
            self.capture(interrupt=True)
        receipt = portable.read_json(self.root / 'raw/capture-receipt.json')
        self.assertIn('KeyboardInterrupt', receipt['fatal_error'])
        states = portable.read_json(self.root / 'raw/capture-index.json')['rows']
        self.assertEqual(len(states), 72)
        self.assertTrue(all(row['failure_reason'] != 'not_processed' for row in states))
        self.assertEqual(receipt['cleanup_errors'], [])

    def test_recovery_failure_keeps_observed_mask_separate(self):
        self.prepared()
        with patch.object(TinySearch, 'recover', return_value=[]):
            result = self.capture()
        self.assertEqual(result['available'], 0)
        physicals = portable.frozen(self.context)[3]
        observed = [row for row in physicals if row['mask']['status'] == 'available']
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(row['seed']['status'] == 'unavailable' for row in observed))
        with fixture_stage(self.context, 'seal-capture'):
            seal = portable.seal_capture(self.context)
        self.assertEqual(len(seal['refinement_states']), 72)
        self.assertFalse(seal['training_ready'])

    def test_atomic_metadata_failure_preserves_previous_bytes(self):
        self.freeze(); path = self.root / 'safe.json'
        portable.write_json(self.context, path, dict(old=True))
        old = path.read_bytes()
        with patch('bench.quality_adaptation_portable.os.fsync', side_effect=OSError('fixture disk failure')):
            with self.assertRaises(OSError):
                portable.write_json(self.context, path, dict(old=False), replace=True)
        self.assertEqual(path.read_bytes(), old)
        self.assertFalse(path.with_name('.safe.json.part').exists())

    def test_preexisting_partial_and_target_survive_rejected_metadata_write(self):
        self.freeze()
        for target_exists, replace in ((False, False), (True, False), (True, True)):
            with self.subTest(target_exists=target_exists, replace=replace):
                path = self.root / f'attempt-{target_exists}-{replace}.json'
                old = b'{"earlier_target":true}\n'
                if target_exists:
                    path.write_bytes(old)
                partial = path.with_name('.' + path.name + '.part')
                prior = b'prior-interrupted-metadata-attempt'
                partial.write_bytes(prior)
                with self.assertRaises(FileExistsError):
                    portable.write_json(self.context, path, dict(new=True), replace=replace)
                self.assertEqual(partial.read_bytes(), prior)
                if target_exists:
                    self.assertEqual(path.read_bytes(), old)
                else:
                    self.assertFalse(path.exists())

    def test_resource_limit_precedes_large_write_and_keeps_metadata_reserve(self):
        used = portable._usage([Path(self.context.document()[key]) for key in ('output_root', 'scratch_root', 'model_cache')])
        doc = self.context.document(); doc['limits']['storage_bytes'] = used + 100
        raw = portable.encoded(doc); context = portable.PortableContext(raw, digest(raw))
        with self.assertRaisesRegex(portable.ResourceError, 'storage bound'):
            portable.resource_check(context, 101)
        result = portable.resource_check(context, 1, metadata_write=True)
        self.assertEqual(result['quota_isolation'], 'unknown')

    def test_shared_physical_files_count_once(self):
        root = self.base / 'physical'; root.mkdir(); path = root / 'a'; path.write_bytes(b'x' * 100)
        import os
        other = self.base / 'other'; other.mkdir()
        try:
            os.link(path, other / 'alias')
        except OSError as exc:
            self.skipTest('filesystem does not permit hardlinks: ' + str(exc))
        self.assertEqual(portable._usage([root, other]), 100)

    def test_help_and_import_do_not_load_torch(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import bench.quality_adaptation_portable; import sys; assert "torch" not in sys.modules'],
            cwd=portable.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-B', 'scripts/kaggle_raw_prepare.py', '--help'],
            cwd=portable.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('seal-capture', result.stdout)


class StorageTraversalTests(unittest.TestCase):
    """Filesystem controls only; no assets, renderer, encoder or training."""
    def setUp(self):
        # Keep link/rename controls inside the explicitly writable workspace;
        # Windows AppContainer temp directories may prohibit these operations.
        self.temporary = tempfile.TemporaryDirectory(prefix='.storage-test-', dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_complete_sizes_and_overlapping_roots(self):
        nested = self.root / 'nested'; nested.mkdir()
        (self.root / 'a').write_bytes(b'123')
        original = nested / 'b'; original.write_bytes(b'1234567')
        # Analytical payload sizes: three plus seven, overlapping roots once.
        self.assertEqual(portable._usage([self.root, nested, self.root]), 10)

    def test_hardlinks_count_once(self):
        import os
        original = self.root / 'a'; original.write_bytes(b'1234567')
        try:
            os.link(original, self.root / 'alias')
        except OSError as error:
            self.skipTest('hardlink creation unavailable: ' + str(error))
        self.assertEqual(portable._usage([self.root]), 7)

    def test_retry_discards_entire_partial_count(self):
        (self.root / 'a').write_bytes(b'12345')
        calls = []
        real = portable._storage_scan
        def changed(roots):
            calls.append(tuple(roots))
            if len(calls) == 1:
                (self.root / 'b').write_bytes(b'1234567')
                raise FileNotFoundError('descendant atomically published')
            return real(roots)
        with patch.object(portable, '_storage_scan', changed):
            self.assertEqual(portable._usage(iter([self.root])), 12)
        self.assertEqual(len(calls), 2)

    def test_persistent_churn_is_bounded_and_cancels(self):
        with patch.object(portable, '_storage_scan', side_effect=FileNotFoundError('churn')) as scan:
            with self.assertRaisesRegex(portable.ResourceError, 'three scans'):
                portable._usage([self.root])
        self.assertEqual(scan.call_count, 3)

    def test_permission_failure_is_not_retried_or_hidden(self):
        with patch.object(portable, '_storage_scan', side_effect=PermissionError('denied')) as scan:
            with self.assertRaises(PermissionError):
                portable._usage([self.root])
        self.assertEqual(scan.call_count, 1)

    def test_missing_prospective_root_is_zero(self):
        self.assertEqual(portable._usage([self.root / 'not-yet-created']), 0)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows directory replacement lock')
    def test_windows_directory_handle_prevents_replacement(self):
        import os
        stage = self.root / 'stage'; stage.mkdir()
        target = self.root / 'published'
        with portable._storage_windows_directory(stage):
            with self.assertRaises(PermissionError):
                os.replace(stage, target)
        self.assertTrue(stage.is_dir())
        os.replace(stage, target)
        self.assertTrue(target.is_dir())

    def test_links_and_dangling_links_are_rejected(self):
        import os
        target = self.root / 'payload'; target.write_bytes(b'1234567')
        for directory, dangling in ((False, False), (True, False), (False, True)):
            with self.subTest(directory=directory, dangling=dangling):
                link = self.root / 'link'
                destination = self.root / 'absent' if dangling else self.root if directory else target
                try:
                    os.symlink(destination, link, target_is_directory=directory)
                except OSError as error:
                    self.skipTest('link creation unavailable: ' + str(error))
                try:
                    with self.assertRaisesRegex(ValueError, 'symlink|reparse'):
                        portable._usage([self.root])
                finally:
                    link.unlink()

    @unittest.skipIf(sys.platform == 'win32', 'POSIX atomic-directory publication control')
    def test_actual_directory_publish_during_open_restarts_complete_scan(self):
        import os
        stage = self.root / '.view.stage'; stage.mkdir()
        (stage / 'payload').write_bytes(b'1234567')
        (self.root / 'other').write_bytes(b'123')
        real_open = portable.os.open
        renamed = []
        def publish(name, *args, **kwargs):
            if name == stage.name and not renamed:
                os.replace(stage, self.root / 'view'); renamed.append(True)
            return real_open(name, *args, **kwargs)
        with patch.object(portable.os, 'open', publish):
            self.assertEqual(portable._usage([self.root]), 10)
        self.assertEqual(renamed, [True])

    @unittest.skipIf(sys.platform == 'win32', 'POSIX descriptor no-follow control')
    def test_directory_substitution_never_opens_outside_target(self):
        import os
        stage = self.root / 'stage'; stage.mkdir()
        outside = self.root.parent / (self.root.name + '-outside'); outside.mkdir()
        self.addCleanup(outside.rmdir)
        payload = outside / 'secret'; payload.write_bytes(b'x' * 17)
        self.addCleanup(payload.unlink)
        real_open = portable.os.open
        substituted = []
        def substitute(name, *args, **kwargs):
            if name == stage.name and not substituted:
                stage.rmdir(); os.symlink(outside, stage); substituted.append(True)
            return real_open(name, *args, **kwargs)
        try:
            with patch.object(portable.os, 'open', substitute):
                with self.assertRaisesRegex(ValueError, 'symlink|non-directory'):
                    portable._usage([self.root])
        finally:
            if stage.is_symlink():
                stage.unlink()
        self.assertEqual(substituted, [True])


if __name__ == '__main__':
    unittest.main()
