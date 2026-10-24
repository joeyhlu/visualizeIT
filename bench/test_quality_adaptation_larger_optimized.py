"""Model-free lifecycle, source admission and input-only parity checks."""
import copy
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
import weakref
from unittest.mock import patch
import numpy as np
from . import quality_adaptation_larger as larger
from . import quality_adaptation_larger_prepare as prepare
from . import quality_adaptation_larger_optimized as optimized
from . import quality_adaptation_larger_parity as parity

TEMP = Path(__file__).resolve().parents[1] / '.cache/cloud-prepare/test-temp'
TEMP.mkdir(parents=True, exist_ok=True)


class StubMasks:
    def __init__(self, fail=None):
        self.timings = optimized.Timings()
        self.events, self.resident, self.fail = [], None, fail
        self.values = {}
    def load(self, phase):
        assert self.resident is None
        self.resident = phase
        self.events.append(('load', phase))
        if self.fail == phase + '_load':
            raise RuntimeError('load unavailable')
    def release(self, phase):
        self.events.append(('release', phase))
        self.resident = None
        if self.fail == phase + '_release':
            raise RuntimeError('release unavailable')
    def proposals(self, view):
        self.events.append(('proposals', view['sequence_id']))
        masks = np.full((1, 480, 640), .75, np.float32)
        if self.fail == view['sequence_id']:
            raise RuntimeError('view unavailable')
        return np.array([[0, 0, 639, 479]], np.float32), masks
    def scores(self, view, boxes, masks):
        self.events.append(('scores', view['sequence_id']))
        assert masks.dtype == np.float32
        assert np.all(masks == .75)  # Detect premature binary thresholding.
        return [.9], 'a' * 64
    def segment(self, view, box):
        self.events.append(('set_image', view['sequence_id']))
        if self.fail == 'sam_view':
            raise RuntimeError('prediction unavailable')
        return np.ones((480, 640), bool)
    def write(self, view, selected, row):
        self.values[view['sequence_id']] = selected.copy()
        return dict(mask='stub')


class PhaseTests(unittest.TestCase):
    def run_phases(self, backend=None, reserve=None, views=None):
        views = views or [dict(sequence_id=s) for s in ('A', 'B', 'A-repeat')]
        backend = backend or StubMasks()
        saved = []
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            scratch = Path(folder) / 'owned'
            sentinel = Path(folder) / 'unrelated'; sentinel.write_text('retain')
            result = optimized.run_mask_phases(views, scratch, backend,
                reserve=reserve or (lambda amount: None), persist=lambda rows, summary: saved.append((rows, summary)))
            self.assertFalse(scratch.exists())
            self.assertTrue(sentinel.exists())
        return result, backend, saved

    def test_lifecycles_floating_masks_and_original_order(self):
        (rows, summary), backend, _ = self.run_phases()
        self.assertEqual([r['status'] for r in rows], ['available'] * 3)
        self.assertEqual(backend.events, [('load', 'fastsam'), ('proposals', 'A'), ('proposals', 'B'),
            ('proposals', 'A-repeat'), ('release', 'fastsam'), ('load', 'dino'), ('scores', 'A'),
            ('scores', 'B'), ('scores', 'A-repeat'), ('release', 'dino'), ('load', 'sam2'),
            ('set_image', 'A'), ('set_image', 'B'), ('set_image', 'A-repeat'), ('release', 'sam2')])
        np.testing.assert_array_equal(backend.values['A'], backend.values['A-repeat'])
        self.assertTrue(summary['scratch_cleaned'])
        self.assertGreater(summary['scratch_peak_bytes'], 0)
        self.assertTrue(all('proposals_sha256' in row for row in rows))

    def test_one_view_failure_retained_and_others_continue(self):
        (rows, _), _, _ = self.run_phases(StubMasks('B'))
        self.assertEqual([r['status'] for r in rows], ['available', 'unavailable', 'available'])
        self.assertIn('view unavailable', rows[1]['failure_reason'])

    def test_each_phase_load_failure_releases_and_accounts_every_row(self):
        for phase in ('fastsam', 'dino', 'sam2'):
            with self.subTest(phase=phase):
                (rows, _), backend, _ = self.run_phases(StubMasks(phase + '_load'))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(r['status'] == 'unavailable' for r in rows))
                self.assertIn(('release', phase), backend.events)
                self.assertIsNone(backend.resident)

    def test_resource_failure_keeps_rows_without_loading_models(self):
        def denied(amount):
            raise ValueError('prospective scratch reserve unavailable')
        (rows, summary), backend, saved = self.run_phases(reserve=denied)
        self.assertEqual(backend.events, [])
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all('scratch reserve' in r['failure_reason'] for r in rows))
        self.assertTrue(summary['scratch_cleaned'])

    def test_preexisting_scratch_is_retained_even_when_empty(self):
        for occupied in (False, True):
            with self.subTest(occupied=occupied), tempfile.TemporaryDirectory(dir=TEMP) as folder:
                scratch = Path(folder) / 'prior-attempt'; scratch.mkdir()
                if occupied:
                    (scratch / 'prior.npy').write_bytes(b'preserve prior attempt')
                before = {p.name: p.read_bytes() for p in scratch.iterdir()}
                backend, saved = StubMasks(), []
                rows, summary = optimized.run_mask_phases([dict(sequence_id='A')], scratch, backend,
                    reserve=lambda amount: None, persist=lambda rows, summary: saved.append((rows, summary)))
                self.assertTrue(scratch.is_dir())
                self.assertEqual({p.name: p.read_bytes() for p in scratch.iterdir()}, before)
                self.assertFalse(summary['scratch_created'])
                self.assertFalse(summary['scratch_cleaned'])
                self.assertEqual(backend.events, [])
                self.assertEqual(rows[0]['status'], 'unavailable')
                self.assertIn('FileExistsError', rows[0]['failure_reason'])

    def test_prospective_full_object_reserve(self):
        calls = []
        self.run_phases(reserve=calls.append)
        self.assertEqual(calls[0], 3 * optimized.PROPOSAL_BYTES_PER_VIEW)
        self.assertEqual(24 * optimized.PROPOSAL_BYTES_PER_VIEW, 5_898_415_104)

    def test_release_failure_fails_and_persists_inventory(self):
        backend, saved = StubMasks('fastsam_release'), []
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            with self.assertRaisesRegex(RuntimeError, 'release failed'):
                optimized.run_mask_phases([dict(sequence_id='A')], Path(folder) / 'owned', backend,
                    reserve=lambda amount: None, persist=lambda r, s: saved.append((r, s)))
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0][1]['scratch_cleaned'])
            self.assertTrue(saved[0][1]['cleanup_errors'])

    def test_score_tie_reverses_original_argsort(self):
        masks = np.ones((2, 3, 3), np.float32)
        selected, diagnostics = optimized.select_proposal([.9, .9], masks, 'a' * 64)
        self.assertIsNotNone(selected)
        self.assertEqual(diagnostics['selected_proposal'], 1)

    def test_ambiguous_and_low_score_remain_unavailable(self):
        masks = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], np.float32)
        self.assertEqual(optimized.select_proposal([.9, .89], masks, 'a' * 64)[1]['reason'],
            'cnos_ambiguous_object_identity')
        self.assertEqual(optimized.select_proposal([.2], masks[:1], 'a' * 64)[1]['reason'],
            'cnos_low_object_identity_score')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            optimized.select_proposal([float('nan')], masks[:1], 'a' * 64)

    def test_legacy_substep_observer_preserves_outputs_and_model_lifetime(self):
        from . import quality_cnos as cnos
        fake_utils = ModuleType('utils')
        class Segmentor:
            def __init__(self, value): self.value = value
            def generate_detection_proposals(self, rgb): return self.value + rgb
        class Dino:
            def forward_features(self, value): return value * 2
        fake_utils.fastsam_util = SimpleNamespace(FastSAM=Segmentor)
        original = Dino.forward_features
        timings = optimized.Timings()
        with patch.dict(sys.modules, {'utils': fake_utils}), patch.object(cnos, 'upstream_path'), \
             patch.object(cnos, 'load_descriptor', return_value=Dino()):
            with prepare.legacy_cnos_timing(timings, 'A'):
                instance = Segmentor(3)
                self.assertEqual(instance.generate_detection_proposals(4), 7)
                ref = weakref.ref(instance); del instance
                self.assertIsNone(ref())
                model = cnos.load_descriptor('cpu')
                self.assertEqual(model.forward_features(3), 6)
                self.assertEqual(model.forward_features(5), 10)
        self.assertIs(Dino.forward_features, original)
        self.assertEqual([r['temperature'] for r in timings.events if r['step'] == 'dino_infer'], ['cold', 'warm'])


class SearchTests(unittest.TestCase):
    def test_one_object_at_a_time_and_final_disposal(self):
        events = []
        class Prepared:
            def __init__(self, **kw):
                self.obj = kw['object_id']; events.append(('load', self.obj))
            def close(self): events.append(('close', self.obj))
        cache = optimized.ObjectSearchCache(Prepared)
        def get(obj):
            return cache.get(object_id=obj, bank_pin=dict(sha256=obj * 64), asset_sha256='a' * 64, variant=optimized.VARIANT)
        a = get('1'); self.assertIs(get('1'), a)
        get('2'); get('1'); cache.close()
        self.assertEqual(events, [('load', '1'), ('close', '1'), ('load', '2'), ('close', '2'), ('load', '1'), ('close', '1')])

    def test_bind_uses_unchanged_recover_fresh_view_state_and_indices_once(self):
        from . import quality_neighbors as neighbors
        fake_utils, fake_knn = ModuleType('utils'), ModuleType('utils.knn_util')
        fake_utils.knn_util = fake_knn; fake_knn.KNN = object
        calls, disposed = [], []
        class Recovery:
            def __init__(self, refiner, bank, *, time_policy):
                calls.append(refiner)
                self.projector, self.repre, self.visual_words, self.indices = object(), object(), object(), [object()]
            def recover(self, frame, mask, top_k=5):
                return [self.refiner.identity, self.memory, self.time_policy.mode]
        pin = dict(path='bank', sha256='a' * 64)
        bank = dict(asset_sha256='b' * 64, template_count=798)
        with patch.object(larger, 'pinned_file', return_value=Path('bank')), patch.dict(sys.modules, {'utils': fake_utils}):
            prepared = optimized.PreparedFoundPoseSearch(bank_pin=pin, asset_sha256='b' * 64,
                object_id='1', variant=optimized.VARIANT, timings=optimized.Timings(), loader=lambda p: bank,
                recovery_class=Recovery, dispose=disposed.append)
            def bind(identity):
                refiner = SimpleNamespace(identity=identity, renderer=SimpleNamespace(asset_sha256='b' * 64),
                    network=SimpleNamespace(backbone=object()))
                return prepared.bind(refiner, object_id='1', bank_pin=pin, asset_sha256='b' * 64,
                    variant=optimized.VARIANT, time_policy=SimpleNamespace(mode='clock-' + identity))
            a, b, repeat = bind('A'), bind('B'), bind('A')
            self.assertIs(type(a).recover, Recovery.recover)
            self.assertIsNot(a, b); self.assertIs(a.indices, b.indices)
            self.assertEqual(a.recover(None, None), repeat.recover(None, None))
            self.assertEqual(b.recover(None, None), ['B', None, 'clock-B'])
            self.assertEqual(len(calls), 1)
            self.assertIsNone(prepared.owner.refiner)
            prepared.close(); prepared.close()
            self.assertEqual(len(disposed), 1)

    def test_object_bank_asset_variant_mismatch_and_disposal(self):
        pin = dict(path='bank', sha256='a' * 64)
        with patch.object(larger, 'pinned_file', return_value=Path('bank')):
            with self.assertRaisesRegex(ValueError, 'variant'):
                optimized.PreparedFoundPoseSearch(bank_pin=pin, asset_sha256='b' * 64, object_id='1',
                    variant='legacy-v1', timings=optimized.Timings(), loader=lambda p: {})
            with self.assertRaisesRegex(ValueError, 'authenticated'):
                optimized.PreparedFoundPoseSearch(bank_pin=pin, asset_sha256='b' * 64, object_id='1',
                    variant=optimized.VARIANT, timings=optimized.Timings(), loader=lambda p: dict(template_count=798, asset_sha256='c' * 64))

    def test_explicit_faiss_worker_delete_and_id_clear(self):
        from . import quality_neighbors as neighbors
        indexes = [SimpleNamespace(index_id=i) for i in (2, 3, 4)]
        worker = SimpleNamespace(request=lambda command: commands.append(command)); commands = []
        with patch.object(neighbors, 'client', return_value=worker):
            optimized.dispose_search(SimpleNamespace(visual_words=indexes[0], indices=indexes[1:]))
        self.assertEqual(commands, [('delete', 2), ('delete', 3), ('delete', 4)])
        self.assertTrue(all(i.index_id is None for i in indexes))

    def test_partial_search_constructor_explicitly_disposes_created_indices(self):
        from . import quality_neighbors as neighbors
        fake_utils = ModuleType('utils')
        fake_utils.knn_util = SimpleNamespace(KNN=object)
        fake_utils.cluster_util = SimpleNamespace(kmeans=object)
        class Index:
            def __init__(self): self.index_id = 1
        disposed = []
        class BrokenRecovery:
            def __init__(self, *args, **kwargs):
                neighbors.install()  # Constructor's own install must remain tracked.
                fake_utils.knn_util.KNN()
                fake_utils.knn_util.KNN()
                raise RuntimeError('index build failed')
        pin = dict(path='bank', sha256='a' * 64)
        with patch.dict(sys.modules, {'utils': fake_utils}), patch.object(neighbors, 'IsolatedKNN', Index), \
             patch.object(larger, 'pinned_file', return_value=Path('bank')):
            prepared = optimized.PreparedFoundPoseSearch(bank_pin=pin, asset_sha256='b' * 64,
                object_id='1', variant=optimized.VARIANT, timings=optimized.Timings(),
                loader=lambda path: dict(template_count=798, asset_sha256='b' * 64),
                recovery_class=BrokenRecovery, dispose=lambda owner: disposed.extend(owner.indices))
            with self.assertRaisesRegex(RuntimeError, 'index build failed'):
                prepared.bind(SimpleNamespace(renderer=SimpleNamespace(asset_sha256='b' * 64)),
                    object_id='1', bank_pin=pin, asset_sha256='b' * 64, variant=optimized.VARIANT,
                    time_policy=SimpleNamespace(mode='clock'))
            self.assertEqual(len(disposed), 2)
            self.assertIs(neighbors.IsolatedKNN, Index)
            self.assertIs(fake_utils.knn_util.KNN, object)


class DiagnosticTests(unittest.TestCase):
    def test_actual_mask_aba_api_holds_models_and_rejects_stale_repeat(self):
        from .test_quality_adaptation_larger import spec
        created = []
        class Backend(StubMasks):
            def __init__(self, *args):
                super().__init__(); created.append(self); self.calls = 0
            def input(self, view):
                pin = dict(path='input', bytes=1, sha256=str(view['view_index'] + 1) * 64)
                return {key: pin for key in ('input', 'video', 'mesh')}, None, None, None
            def segment(self, view, box):
                self.calls += 1
                selected = super().segment(view, box)
                if stale and self.calls == 3:
                    selected[0, 0] = False
                return selected
        def fixture_write(root, name, document):
            (Path(root) / name).write_text(json.dumps(document), encoding='utf-8')
        for stale in (False, True):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory(dir=TEMP) as folder, ExitStack() as stack:
                stack.enter_context(patch.object(optimized, 'cloud_guard'))
                stack.enter_context(patch.object(optimized, 'NeuralMaskBackend', Backend))
                stack.enter_context(patch.object(optimized, 'write_json', side_effect=fixture_write))
                stack.enter_context(patch.object(prepare, 'frozen', return_value=spec()))
                stack.enter_context(patch.object(prepare, '_existing_models', return_value=Path('cache')))
                stack.enter_context(patch.object(larger, '_safe_output', side_effect=Path))
                stack.enter_context(patch.object(larger, 'resources'))
                stack.enter_context(patch.object(larger, 'configure_torch', return_value=SimpleNamespace()))
                stack.enter_context(patch.object(larger, 'stage_lock', side_effect=nullcontext))
                args = (Path(folder), '1', 0, 3, Path(folder) / 'diagnostic')
                if stale:
                    with self.assertRaisesRegex(ValueError, 'ABA mask/prompt/association'):
                        optimized.masks_aba_diagnostic(*args, cloud_execution=True)
                else:
                    result = optimized.masks_aba_diagnostic(*args, cloud_execution=True)
                    self.assertTrue(result['passed'])
                inventory = json.loads((Path(folder) / 'diagnostic/inventory.json').read_text())
                self.assertEqual([r['view']['view_index'] for r in inventory['rows']], [0, 3, 0])
                self.assertEqual(len({r['view']['sequence_id'] for r in inventory['rows']}), 3)
                self.assertEqual([event for event in created[-1].events if event[0] == 'load'],
                    [('load', 'fastsam'), ('load', 'dino'), ('load', 'sam2')])

    def test_recovery_aba_api_keeps_search_and_fresh_per_view_renderers(self):
        from .test_quality_adaptation_larger import plan
        doc = plan(); doc['preparation_variant'] = optimized.VARIANT
        rendered, recovered, closed, caches = [], [], [], []
        network = SimpleNamespace(backbone=object(), cpu=lambda: None)
        network.eval = lambda: network
        class Renderer:
            def __init__(self, *args, **kwargs): rendered.append(self)
            def close(self): closed.append(self)
        class Refiner:
            def __init__(self, network, renderer, *args, **kwargs): self.renderer = renderer
        class Search:
            def bind(self, refiner, **kwargs):
                def recover(frame, mask, **kw):
                    recovered.append(refiner.renderer)
                    return [np.eye(4)]
                return SimpleNamespace(recover=recover)
        class Cache:
            def __init__(self): self.search = Search(); self.gets = 0; self.closed = False; caches.append(self)
            def get(self, **kwargs): self.gets += 1; return self.search
            def close(self): self.closed = True
        inputs = {r['id']: (Path('bundle'), dict(asset='object.glb', object_id='1',
            _timing=dict(time_policy=SimpleNamespace(mode='physical'))), SimpleNamespace(effective_intrinsics=np.eye(3)))
            for r in doc['rows']}
        def fixture_write(root, name, document):
            (Path(root) / name).write_text(json.dumps(document), encoding='utf-8')
        with tempfile.TemporaryDirectory(dir=TEMP) as folder, ExitStack() as stack:
            stack.enter_context(patch.object(optimized, 'cloud_guard'))
            stack.enter_context(patch.object(optimized, 'ObjectSearchCache', Cache))
            stack.enter_context(patch.object(optimized, 'write_json', side_effect=fixture_write))
            stack.enter_context(patch.object(larger, 'read_plan', return_value=(b'{}', doc)))
            stack.enter_context(patch.object(larger, 'preflight', return_value=inputs))
            stack.enter_context(patch.object(larger, 'pinned_file', return_value=Path('mask')))
            stack.enter_context(patch.object(larger, 'configure_torch', return_value=SimpleNamespace(manual_seed=lambda seed: None)))
            stack.enter_context(patch.object(larger, 'cleanup_device'))
            stack.enter_context(patch.object(larger, '_safe_output', side_effect=Path))
            stack.enter_context(patch.object(larger, 'resources'))
            stack.enter_context(patch.object(larger, 'stage_lock', side_effect=nullcontext))
            stack.enter_context(patch('bench.quality_gotrack.load_network', return_value=network))
            stack.enter_context(patch('bench.quality_gotrack.TexturedRenderer', Renderer))
            stack.enter_context(patch('bench.quality_gotrack.GoTrackRefiner', Refiner))
            stack.enter_context(patch('bench.quality_runner.read_rgb', return_value=np.zeros((480, 640, 3), np.uint8)))
            stack.enter_context(patch('bench.vision.cv2.imread', return_value=np.ones((480, 640), np.uint8)))
            result = optimized.recovery_aba_diagnostic('plan', 'a' * 64, '1', 0, 3,
                Path(folder) / 'diagnostic', cloud_execution=True)
        self.assertTrue(result['passed']); self.assertEqual(caches[0].gets, 3)
        self.assertTrue(caches[0].closed)
        self.assertEqual(len(set(map(id, recovered))), 3)
        self.assertEqual(rendered, closed)


def parity_fixture():
    crops = {key: np.zeros(1, np.float32) for key in larger.CROP_KEYS}
    geometry = {key: np.zeros(1, np.float32) for key in larger.GEOMETRY_KEYS}
    row = dict(id='train-A:frame0:iteration0', input_pins={key: dict(sha256='a' * 64, bytes=1)
        for key in ('input', 'video', 'mesh', 'bank')}, status='available', failure_reason=None,
        mask=dict(shape=[480, 640], dtype='uint8', sha256='b' * 64),
        detector=dict(scores=[.9], selected_proposal=0, reason=None), prompt_box_xyxy=[0, 0, 639, 479],
        candidates=[np.eye(4).tolist()], crops=crops, geometry=geometry)
    return dict(schema=parity.SCHEMA, device='cuda', rows=[row], requested_ids=[row['id']],
        fatal_error=None, test_payload_reads=0, annotation_inputs=[])


class ParityTests(unittest.TestCase):
    def test_exact_and_tolerated_predictions_and_aba(self):
        a, b = parity_fixture(), parity_fixture()
        b['rows'][0]['geometry']['actual_flow'][0] = 0.00009
        b['rows'][0]['geometry']['actual_confidence'][0] = 0.0000009
        self.assertTrue(parity.compare_aba(a, b, copy.deepcopy(a))['a_a_repeat']['passed'])

    def test_mask_proposal_order_candidates_and_input_pins_fail(self):
        for key in ('mask', 'detector', 'candidates', 'input_pins', 'prompt_box_xyxy'):
            a, b = parity_fixture(), parity_fixture()
            if key == 'mask': b['rows'][0][key]['sha256'] = 'c' * 64
            if key == 'detector': b['rows'][0][key]['selected_proposal'] = 1
            if key == 'candidates': b['rows'][0][key][0][0][0] = 2.
            if key == 'input_pins': b['rows'][0][key]['video']['sha256'] = 'c' * 64
            if key == 'prompt_box_xyxy': b['rows'][0][key][0] = 1
            with self.subTest(key=key), self.assertRaises(ValueError): parity.compare_runs(a, b)

    def test_missing_nonfinite_dtype_and_prediction_tolerance_fail(self):
        for alteration in ('missing', 'nan', 'dtype', 'flow', 'confidence'):
            a, b = parity_fixture(), parity_fixture(); row = b['rows'][0]
            if alteration == 'missing': del row['geometry']['native_k']
            if alteration == 'nan': row['geometry']['actual_flow'][0] = np.nan
            if alteration == 'dtype': row['geometry']['native_k'] = np.zeros(1, np.float64)
            if alteration == 'flow': row['geometry']['actual_flow'][0] = .0002
            if alteration == 'confidence': row['geometry']['actual_confidence'][0] = .000002
            with self.subTest(alteration=alteration), self.assertRaises(ValueError): parity.compare_runs(a, b)

    def test_failures_are_counted_cannot_drop_or_pass_only_failures(self):
        a = parity_fixture(); failed = copy.deepcopy(a['rows'][0]); failed.update(id='failed', status='unavailable',
            failure_reason='no seed', candidates=[], crops=None, geometry=None)
        a['rows'].append(failed); a['requested_ids'].append('failed')
        self.assertEqual(parity.compare_runs(a, copy.deepcopy(a))['unavailable'], 1)
        b = copy.deepcopy(a); b['rows'].pop()
        with self.assertRaises(ValueError): parity.compare_runs(a, b)
        a['rows'] = [failed]; a['requested_ids'] = ['failed']
        with self.assertRaisesRegex(ValueError, 'no available'): parity.compare_runs(a, copy.deepcopy(a))

    def test_unmaterialized_rows_keep_explicit_none_pins_and_full_inventory(self):
        a = parity_fixture(); unrun = copy.deepcopy(a['rows'][0])
        unrun.update(id='unrendered', input_pins={key: None for key in ('input', 'video', 'mesh', 'bank')},
            status='unavailable', failure_reason='render_not_attempted', mask=None, detector=None,
            prompt_box_xyxy=None, candidates=None, crops=None, geometry=None)
        a['rows'].append(unrun); a['requested_ids'].append('unrendered')
        self.assertEqual(parity.compare_runs(a, copy.deepcopy(a))['requested'], 2)
        b = copy.deepcopy(a); b['rows'][-1]['input_pins']['mesh'] = dict(sha256='a' * 64, bytes=1)
        with self.assertRaisesRegex(ValueError, 'input_pins'): parity.compare_runs(a, b)

    def test_real_npz_pin_is_authenticated_and_compared(self):
        from .quality_adaptation_data import sha256
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path = Path(folder) / 'geometry.npz'
            a, b = parity_fixture(), parity_fixture()
            np.savez(path, **a['rows'][0]['geometry'])
            descriptor = dict(pin=dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)), keys=sorted(larger.GEOMETRY_KEYS))
            b['rows'][0]['geometry'] = descriptor
            self.assertTrue(parity.compare_runs(a, b)['passed'])
            descriptor['pin']['sha256'] = 'c' * 64
            with self.assertRaisesRegex(ValueError, 'pin differs'): parity.compare_runs(a, b)


class FailureCrossPathTests(unittest.TestCase):
    def test_actual_legacy_masks_and_phased_failure_evidence_agree(self):
        """Real legacy detect/masks and phased proposal/selection; only model outputs are stubs."""
        from . import quality_cnos as cnos
        from .test_quality_adaptation_larger import spec, plan
        class Tensor:
            def __init__(self, values): self.values = np.asarray(values)
            def to(self, device): return self
            def cpu(self): return self
            def numpy(self): return self.values
            def __getitem__(self, index): return Tensor(self.values[index])
            def __matmul__(self, other): return self
            @property
            def T(self): return Tensor(self.values.T)
            def topk(self, k, dim): return SimpleNamespace(values=self)
            def mean(self): return self.values.mean()
        class Proposals:
            def __init__(self, masks):
                self.masks = Tensor(masks)
                self.boxes = Tensor(np.tile([0., 0., 639., 479.], (len(masks), 1)))
            def __len__(self): return len(self.masks.values)
            def __getitem__(self, selected): return Proposals(self.masks.values[selected])
        failed_snapshots = {}
        for case in ('low', 'ambiguous', 'no_proposals', 'no_supported'):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=TEMP) as folder, ExitStack() as stack:
                root = Path(folder); legacy = root / 'legacy'; legacy.mkdir()
                view = spec()['views'][0]
                bundle = prepare._directory(legacy, view) / 'bundle'; bundle.mkdir(parents=True)
                (bundle / 'input.json').write_text(json.dumps(dict(asset='object.glb', object='stub')))
                cache = root / 'cache'; (cache / 'checkpoints').mkdir(parents=True)
                (cache / 'lock.json').write_text(json.dumps(dict(models={'FastSAM-x.pt': dict(sha256='a' * 64)})))
                bank = root / 'bank.npz'; np.savez(bank, asset_sha256='a' * 64, descriptors=np.zeros((57, 5)))
                bank_pin = dict(path=str(bank), bytes=bank.stat().st_size, sha256='a' * 64)
                input_pin = dict(path=str(bundle / 'input.json'), bytes=(bundle / 'input.json').stat().st_size,
                    sha256=larger.sha256(bundle / 'input.json'))
                receipt = dict(input=input_pin, video=dict(input_pin, path=str(bundle / 'video.mp4')),
                    mesh=dict(input_pin, path=str(bundle / 'object.glb')))
                (legacy / 'render-000001-00.json').write_text(json.dumps(receipt))
                (legacy / 'cnos-000001.json').write_text(json.dumps(dict(bank=bank_pin)))
                masks = np.zeros((0 if case == 'no_proposals' else 2 if case == 'ambiguous' else 1, 480, 640), np.float32)
                if len(masks): masks[0, :20, :20] = .75
                if len(masks) == 2: masks[1, 40:60, 40:60] = .75
                scores = [.9, .89] if case == 'ambiguous' else [.2]
                if case in ('no_proposals', 'no_supported'): scores = []
                class Segmentor:
                    def __init__(self, *args): pass
                    def set_device(self, device): pass
                    def generate_detection_proposals(self, rgb): return Proposals(masks)
                class Dino:
                    def __init__(self): self.index = 0
                    def forward_features(self, value):
                        score = scores[self.index]; self.index += 1
                        return {'x_norm_clstoken': Tensor([[score]])}
                fake_utils = ModuleType('utils')
                fake_utils.fastsam_util = SimpleNamespace(FastSAM=Segmentor)
                fake_utils.im_util = SimpleNamespace(filter_noisy_detections=lambda *args, **kwargs:
                    np.zeros(len(masks), bool) if case == 'no_supported' else np.ones(len(masks), bool),
                    im_normalize=Tensor)
                fake_utils.crop_generation = SimpleNamespace(batch_cropping_from_bbox=lambda **kwargs:
                    SimpleNamespace(rgbs=np.ones((len(masks), 3, 2, 2)), masks=np.ones((len(masks), 2, 2))))
                fake_config = ModuleType('model.config'); fake_config.FastSAMOpts = lambda **kwargs: SimpleNamespace(**kwargs)
                torch = SimpleNamespace(from_numpy=Tensor, device=lambda device: device, inference_mode=nullcontext,
                    nn=SimpleNamespace(functional=SimpleNamespace(normalize=lambda values, dim: values)))
                stack.enter_context(patch.dict(sys.modules, {'torch': torch, 'utils': fake_utils,
                    'model': ModuleType('model'), 'model.config': fake_config}))
                stack.enter_context(patch.object(cnos, 'CACHE', cache))
                stack.enter_context(patch.object(cnos, 'digest', return_value='a' * 64))
                stack.enter_context(patch.object(cnos, 'upstream_path'))
                stack.enter_context(patch.object(cnos, 'camera_for', return_value=None))
                stack.enter_context(patch.object(cnos, 'load_descriptor', side_effect=lambda device: Dino()))
                stack.enter_context(patch.object(prepare, 'cloud_guard'))
                stack.enter_context(patch.object(prepare, 'frozen', return_value=spec()))
                stack.enter_context(patch.object(prepare, '_existing_models', return_value=cache))
                stack.enter_context(patch.object(prepare, 'legacy_cnos_timing', side_effect=lambda *args: nullcontext()))
                def fixture_write(root, name, document):
                    (Path(root) / name).write_text(json.dumps(document), encoding='utf-8')
                stack.enter_context(patch.object(prepare, 'write_json', side_effect=fixture_write))
                stack.enter_context(patch.object(larger, 'configure_torch', return_value=torch))
                stack.enter_context(patch.object(larger, 'stage_lock', side_effect=nullcontext))
                stack.enter_context(patch.object(larger, 'pinned_file', side_effect=lambda pin: Path(pin['path'])))
                stack.enter_context(patch('bench.quality_runner.read_input', return_value={}))
                rgb = np.zeros((480, 640, 3), np.uint8)
                stack.enter_context(patch('bench.quality_runner.read_rgb', return_value=rgb))
                observer_path = root / 'legacy-observer.json'
                with parity.observe_runtime(observer_path):
                    actual_legacy = prepare.masks(legacy, '1', 0, cloud_execution=True)[0]
                observations = json.loads(observer_path.read_text())
                class Backend(StubMasks):
                    def __init__(self): super().__init__(); self.segmentor = Segmentor()
                    def input(self, view): return None, None, None, rgb
                    proposals = optimized.NeuralMaskBackend.proposals
                    def scores(self, view, boxes, proposal_masks): return scores, 'a' * 64
                optimized_rows, _ = optimized.run_mask_phases([view], root / 'optimized-proposals', Backend(),
                    reserve=lambda amount: None, persist=lambda rows, summary: None)
                phased = optimized_rows[0]
                self.assertEqual(actual_legacy['failure_reason'], 'ValueError: CNOS observed initial association unavailable')
                self.assertEqual(actual_legacy['failure_reason'], phased['failure_reason'])
                # Snapshot production code must retain failed masks' observer augmentation.
                physical = plan()['rows'][0]
                physical.update(input=input_pin, video=receipt['video'], mesh=receipt['mesh'], bank=bank_pin,
                    mask=None, mask_index=None, observation_status='unavailable', failure_reason=actual_legacy['failure_reason'])
                state = dict(id=physical['id'] + ':iteration0', status='unavailable',
                    failure_reason=actual_legacy['failure_reason'])
                stack.enter_context(patch.object(larger, 'read_capture', return_value=(dict(rows=[physical]), [state])))
                (root / 'capture-receipt.json').write_text(json.dumps(dict(device='cpu')))
                snapshot = parity.snapshot_capture(root, observations=observations, prepared_root=legacy)
                failed = snapshot['rows'][0]
                self.assertEqual(failed['detector'], phased['detector'])
                expected_reason = {'low': 'cnos_low_object_identity_score', 'ambiguous': 'cnos_ambiguous_object_identity',
                    'no_proposals': 'cnos_no_proposals', 'no_supported': 'cnos_no_supported_proposals'}[case]
                self.assertEqual(failed['detector']['reason'], expected_reason)
                if not scores:
                    self.assertEqual(failed['detector']['scores'], [])
                    self.assertIsNone(failed['detector']['selected_proposal'])
                reference = parity_fixture(); reference['device'] = 'cpu'
                failed.update(id='failed-' + case)
                reference['rows'].append(failed); reference['requested_ids'].append(failed['id'])
                candidate = copy.deepcopy(reference)
                candidate['rows'][-1].update(failure_reason=phased['failure_reason'], detector=phased['detector'])
                self.assertTrue(parity.compare_runs(reference, candidate)['passed'])
                for alteration in ('failure', 'availability', 'detector'):
                    changed = copy.deepcopy(candidate)
                    if alteration == 'failure': changed['rows'][-1]['failure_reason'] = 'different failure'
                    if alteration == 'availability': changed['rows'][-1]['status'] = 'available'
                    if alteration == 'detector': changed['rows'][-1]['detector']['reason'] = 'different detector failure'
                    with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                        parity.compare_runs(reference, changed)
                failed_snapshots[case] = failed
        self.assertNotEqual(failed_snapshots['no_proposals']['detector'], failed_snapshots['no_supported']['detector'])


class AdmissionTests(unittest.TestCase):
    def test_exact_source_closure_and_variant_mismatch(self):
        self.assertEqual(larger.expected_source_files(), larger.SOURCE_FILES)
        self.assertEqual(larger.expected_source_files(optimized.VARIANT), larger.SOURCE_FILES | {
            'bench/quality_adaptation_larger_optimized.py', 'bench/quality_adaptation_larger_parity.py'})
        with self.assertRaisesRegex(ValueError, 'variant'): larger.preparation_variant({}, optimized.VARIANT)
        with self.assertRaisesRegex(ValueError, 'variant'): larger.expected_source_files('foreign')

    def test_device_and_frozen_variant_mismatch(self):
        from .test_quality_adaptation_larger import spec
        doc = spec(); doc['preparation_variant'] = optimized.VARIANT
        receipt = dict(spec_sha256='a', split_sha256='a', views_sha256=larger.canonical_hash(doc['views']),
            device='cuda', preparation_variant=optimized.VARIANT)
        with patch.object(prepare, 'read_json', side_effect=lambda path: doc if Path(path).name == 'spec.json' else receipt), \
             patch.object(prepare, 'sha256', return_value='a'), patch.object(larger, 'verify_source_contract'):
            with self.assertRaisesRegex(ValueError, 'device'): prepare.frozen('root', 'cpu', optimized.VARIANT)
            with self.assertRaisesRegex(ValueError, 'variant'): prepare.frozen('root', 'cuda', 'legacy-v1')

    def test_last_row_test_payload_refused_before_read(self):
        from .test_quality_adaptation_larger import plan
        doc = plan(); doc['preparation_variant'] = optimized.VARIANT
        doc['rows'][-1]['mask']['path'] = larger.SCRATCH + '/obj_000007/0.png'
        with patch.object(larger, 'pinned_file', side_effect=AssertionError('opened payload')):
            with self.assertRaisesRegex(ValueError, 'TEST'): larger.preflight(doc)

    def test_amortized_closure_refuses_missing_foreign_or_legacy_contract(self):
        contract = dict(schema=larger.SOURCE_SCHEMA, preparation_variant=optimized.VARIANT,
            historical_training_manifest_sha256=larger.HISTORICAL_MANIFEST_SHA256, sources=[])
        plan = dict(preparation_variant=optimized.VARIANT, source_contract={}, runtime_contract={})
        for paths in ([], sorted(larger.OPTIMIZED_SOURCE_FILES | {'foreign.py'})):
            contract['sources'] = [dict(path=p, sha256='a' * 64) for p in paths]
            with patch.object(larger, 'pinned_file', return_value=Path('contract')), \
                 patch.object(larger, 'read_json', return_value=contract), patch.object(larger, 'sha256', return_value='a' * 64), \
                 patch.object(larger.shared, '_relative', side_effect=lambda root, path: root / path):
                with self.assertRaisesRegex(ValueError, 'closure missing/foreign'): larger.verify_source_contract(plan, upstream=False)
        contract.pop('preparation_variant')
        with patch.object(larger, 'pinned_file', return_value=Path('contract')), patch.object(larger, 'read_json', return_value=contract):
            with self.assertRaisesRegex(ValueError, 'variant'): larger.verify_source_contract(plan, upstream=False)

    def test_no_torch_import(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
