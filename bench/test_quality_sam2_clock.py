"""Independent capture-plan tests; synthetic inputs, no model or media I/O."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from .quality_sam2 import prepare_cache_plan
from .quality_time import PHYSICAL, LEGACY
from . import quality_sam2 as producer


def capture_document(*, setup=True):
    """Declared captures have a source gap and no implicit nominal-FPS clock."""
    source_ids = (260, 262, 263)
    ticks = (1_000_000, 1_033_333, 1_066_667)
    return {
        'schema_version': 2,
        'object': 'synthetic-selected-object',
        'video': 'video.mp4', 'asset': 'object.glb',
        'source_hashes': {'video.mp4': 'a' * 64, 'object.glb': 'b' * 64},
        'native_resolution': [640, 480],
        'clock': {'mode': PHYSICAL, 'units': 'seconds',
                  'source_units': 'microseconds', 'timestamp_source': 'capture metadata',
                  'nominal_fps': 30},
        'timeline': [dict(frame_id=i, source_frame_id=source,
                          source_timestamp=tick, timestamp_s=tick / 1_000_000,
                          role='setup' if setup and i == 0 else 'scored')
                     for i, (source, tick) in enumerate(zip(source_ids, ticks))],
        'setup_frame_id': 0 if setup else None,
        'frame_ids': [1, 2] if setup else [0, 1, 2],
        'source_gaps': [dict(previous_source_frame_id=260,
                             current_source_frame_id=262, missing_capture_count=1)],
        'selection': dict(reviewed=True, frame_id=0, source_frame_id=260,
                          timestamp_s=1.0, points=[[20, 30], [100, 110]], labels=[1, 0]),
    }


def encode(document):
    return json.dumps(document, separators=(',', ':'), allow_nan=False).encode('utf-8')


class SAMCapturePlanTests(unittest.TestCase):
    def test_setup_source_gap_and_capture_seconds_are_preserved(self):
        raw = encode(capture_document())
        plan = prepare_cache_plan(raw)
        self.assertEqual(plan['clock_mode'], PHYSICAL)
        self.assertEqual([k.frame_id for k in plan['planned_keys']], [0, 1, 2])
        self.assertEqual([k.timestamp_s for k in plan['planned_keys']], [1., 1.033333, 1.066667])
        self.assertEqual(plan['source_frame_ids'], {0: 260, 1: 262, 2: 263})
        self.assertEqual(plan['expected_setup_frames'], 1)
        self.assertEqual(plan['expected_scored_frames'], 2)
        self.assertEqual(plan['input_manifest_sha256'].lower(), hashlib.sha256(raw).hexdigest())

    def test_prefix_keeps_full_plan_but_requests_only_setup_and_first_scored(self):
        plan = prepare_cache_plan(encode(capture_document()), limit=1)
        self.assertEqual(plan['requested_frame_ids'], (0, 1))
        self.assertEqual(len(plan['planned_keys']), 3)
        self.assertEqual(plan['requested_scored_frames'], 1)
        self.assertTrue(plan['diagnostic_prefix'])

    def test_no_setup_initial_selection_is_scored_once(self):
        plan = prepare_cache_plan(encode(capture_document(setup=False)), limit=1)
        self.assertEqual(plan['requested_frame_ids'], (0,))
        self.assertEqual(plan['expected_setup_frames'], 0)
        self.assertEqual(plan['expected_scored_frames'], 3)
        self.assertEqual(plan['selection_key'].frame_id, 0)

    def test_prefix_cannot_hide_corrupt_later_capture_time(self):
        document = capture_document()
        document['timeline'][2]['timestamp_s'] = 20.
        with self.assertRaises(ValueError):
            prepare_cache_plan(encode(document), limit=1)

    def test_selection_must_bind_exact_initial_observation(self):
        for field, value in (('frame_id', 1), ('source_frame_id', 261),
                             ('timestamp_s', 1.033333), ('reviewed', False)):
            with self.subTest(field=field):
                document = capture_document()
                document['selection'][field] = value
                with self.assertRaises(ValueError):
                    prepare_cache_plan(encode(document))

    def test_invalid_prompt_coordinates_or_labels_fail_before_runtime(self):
        for points, labels in (([[640, 0]], [1]), ([[-1, 0]], [1]),
                               ([[20, 480]], [1]), ([[True, 2]], [1]),
                               ([[1, 2]], [True]), ([[1, 2]], [0]),
                               ([[1, 2]], [1, 0])):
            with self.subTest(points=points, labels=labels):
                document = capture_document()
                document['selection'].update(points=points, labels=labels)
                with self.assertRaises(ValueError):
                    prepare_cache_plan(encode(document))

    def test_frame_age_association_is_not_enabled_on_capture_seconds(self):
        with self.assertRaises(ValueError):
            prepare_cache_plan(encode(capture_document()), mask_association=True)

    def test_invalid_limits_do_not_silently_select_full_or_empty_runs(self):
        for value in (True, 0, -1, 1.5, 2, 3):
            with self.subTest(limit=value):
                with self.assertRaises(ValueError):
                    prepare_cache_plan(encode(capture_document()), limit=value)

    def test_arbitrary_untimed_manifest_is_not_authenticated_legacy_input(self):
        raw = encode(dict(schema_version=1, object='keyboard', setup_frame_id=10,
                          frame_ids=[11], source_hashes={'video.mp4': 'a' * 64}))
        with self.assertRaises(ValueError):
            prepare_cache_plan(raw)

    def test_manifest_and_options_are_not_mutated(self):
        document = capture_document()
        frozen = copy.deepcopy(document)
        raw = encode(document)
        prepare_cache_plan(raw, limit=1)
        self.assertEqual(document, frozen)
        self.assertEqual(raw, encode(frozen))

    def test_nominal_fps_metadata_does_not_fabricate_capture_times(self):
        document = capture_document()
        first = prepare_cache_plan(encode(document))
        document['clock']['nominal_fps'] = 60
        second = prepare_cache_plan(encode(document))
        self.assertEqual(first['planned_keys'], second['planned_keys'])
        self.assertEqual(first['timestamp_table_sha256'], second['timestamp_table_sha256'])


class FakeSession:
    """Synthetic camera/predictor boundary: never decode or construct a model."""
    def __init__(self, plan, manifest, *, fault=None):
        self.plan, self.manifest, self.fault = plan, manifest, fault
        self.calls = []
        self.provenance = {'synthetic_only': True}
        self.checkpoint_sha256 = 'c' * 64

    def initialize(self):
        self.calls.append(('initialize',))
        if self.fault == 'initial_decode':
            raise producer.SourceDecodeError(0, 260)

    def add_initial_selection(self, selection):
        self.calls.append(('selection', copy.deepcopy(selection)))

    def mask(self, index, *, empty=False):
        width, height = self.manifest['native_resolution']
        value = np.zeros((height, width), dtype=np.uint8)
        if not empty:
            value[10:30, 10 + index:30 + index] = 255
        return value

    def step(self, index):
        self.calls.append(('step', index, self.plan['source_frame_ids'][index]))
        if self.fault == 'middle_decode' and index == 1:
            raise producer.SourceDecodeError(index, self.plan['source_frame_ids'][index])
        frame_idx = True if self.fault == 'boolean_ordinal' and index == 1 else index
        return dict(frame_idx=frame_idx, obj_ids=[1],
                    mask=self.mask(index, empty=(self.fault in {
                        'no_detection', 'bad_identity', 'bad_detector_shape', 'bad_offload',
                        'condition_failure', 'recovery'} or
                        str(self.fault).startswith('identity:')) and index == 1),
                    blank=False, timings_ms={'sam2': 0.1}, source=self.snapshot_source(index))

    def mask_info(self, mask):
        return dict(shape=tuple(mask.shape), dtype=str(mask.dtype),
                    binary_0_255=bool(np.all((mask == 0) | (mask == 255))),
                    foreground_pixels=int(np.count_nonzero(mask)))

    def empty_mask(self):
        return self.mask(0, empty=True)

    def write_mask(self, path, mask):
        self.calls.append(('write', Path(path).name))
        # The fake writer deliberately stores raw synthetic bytes. PNG decoding
        # is outside these lifecycle tests and never occurs in the fake seam.
        data = mask.tobytes()
        Path(path).write_bytes(data)
        return dict(sha256='bad' if self.fault == 'bad_writer_digest'
                    else hashlib.sha256(data).hexdigest(), bytes=len(data))

    def offload_for_detector(self):
        self.calls.append(('offload',))
        return dict(state_after={'non_cpu_tensor_count': 0},
                    model_after={'model_tensors_on_wrong_device': 0},
                    live_gpu_bytes_after_empty_cache=1 if self.fault == 'bad_offload' else 0)

    def run_detector(self, index, key, *, prior_frame_id=None):
        self.calls.append(('detector', index))
        identity = dict(frameId=key.frame_id, sourceFrameId=self.plan['source_frame_ids'][index],
                        timestamp_s=key.timestamp_s, clock_mode=PHYSICAL,
                        input_manifest_sha256=self.plan['input_manifest_sha256'],
                        timestamp_table_sha256=self.plan['timestamp_table_sha256'])
        if self.fault == 'bad_identity':
            identity['sourceFrameId'] += 1
        if str(self.fault).startswith('identity:'):
            field = self.fault.split(':', 1)[1]
            identity[field] = identity[field] + 1 if field in ('frameId', 'sourceFrameId', 'timestamp_s') else 'tampered'
        mask = (np.zeros((1, 1), dtype=np.uint8) if self.fault == 'bad_detector_shape'
                else self.mask(index, empty=self.fault == 'no_detection'))
        return dict(identity=identity, mask=mask, process_returncode=0, process_reaped=True,
                    artifact_sha256='d' * 64, timings_ms={'cnos': 0.2}, reason=None)

    def restore_after_detector(self):
        self.calls.append(('restore',))
        return dict(model={'model_tensors_on_wrong_device': 0}, state={'device_mismatches': 0})

    def resume_after_detector(self, index):
        self.calls.append(('resume', index))

    def condition(self, index, mask):
        self.calls.append(('condition', index))
        if self.fault == 'condition_failure':
            raise RuntimeError('synthetic conditioning failure')

    def snapshot_source(self, index):
        failed = self.fault == 'initial_decode' and index == 0 or self.fault == 'middle_decode' and index == 1
        return dict(decode_attempted=1, decode_returned=0 if failed else 1,
                    decoder_requests=1, cache_hits=0, past_reads=0, decode_ms=0.1,
                    decode_failure='rgb_decode_failed' if failed else None)

    def close(self):
        self.calls.append(('close',))
        if self.fault == 'incomplete_cleanup':
            return {}
        return dict(iterator_released=True, decoder_released=True,
                    predictor_released=True, loader_hook_restored=True)


class FakeRuntime:
    def __init__(self, *, fault=None):
        self.fault, self.session, self.opens = fault, None, 0

    def open_session(self, *, plan, manifest, bundle, output, device):
        self.opens += 1
        self.session = FakeSession(plan, manifest, fault=self.fault)
        return self.session


class SAMPhysicalLifecycleTests(unittest.TestCase):
    def execute(self, root, *, setup=True, limit=None, fault=None):
        bundle = root / 'input'
        bundle.mkdir()
        document = capture_document(setup=setup)
        document.update(units='metres', intrinsics=[[500, 0, 319.5], [0, 500, 239.5], [0, 0, 1]])
        for name in ('video.mp4', 'object.glb'):
            body = b'fake resource bytes; never decoded: ' + name.encode()
            (bundle / name).write_bytes(body)
            document['source_hashes'][name] = hashlib.sha256(body).hexdigest()
        (bundle / 'input.json').write_bytes(encode(document))
        output = root / 'masks'
        runtime = FakeRuntime(fault=fault)
        error = None
        try:
            producer.run(bundle, output, 'cpu', limit=limit, runtime=runtime)
        except RuntimeError as exc:
            error = exc
        result = json.loads((output / 'results.json').read_text(encoding='utf-8'))
        return result, runtime, error

    def test_full_table_single_selection_and_actual_source_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder))
            self.assertIsNone(error)
            self.assertTrue(result['stage_completed'])
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([r['mask_state'] for r in result['frames']], ['available'] * 3)
            self.assertEqual([r['sourceFrameId'] for r in result['frames']], [260, 262, 263])
            self.assertEqual(sum(c[0] == 'selection' for c in runtime.session.calls), 1)
            self.assertEqual([c[1:] for c in runtime.session.calls if c[0] == 'step'],
                             [(0, 260), (1, 262), (2, 263)])
            self.assertEqual(result['prompt_ledger']['initial_selection']['attempted'], 1)
            self.assertEqual(result['prompt_ledger']['initial_selection']['returned'], 1)

    def test_prefix_has_explicit_unmeasured_suffix_without_future_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), limit=1)
            self.assertIsNone(error)
            self.assertEqual(result['status'], 'diagnostic_prefix')
            self.assertEqual(len(result['frames']), 3)
            self.assertEqual([c[1] for c in runtime.session.calls if c[0] == 'step'], [0, 1])
            self.assertEqual(result['frames'][2]['measurement_state'], 'unmeasured')
            self.assertIsNone(result['frames'][2]['path'])

    def test_no_setup_initial_frame_is_measured_and_scored_once(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), setup=False)
            self.assertIsNone(error)
            self.assertEqual([r['role'] for r in result['frames']], ['scored'] * 3)
            self.assertEqual(result['frames'][0]['measurement_state'], 'measured')
            self.assertEqual(sum(c[0] == 'selection' for c in runtime.session.calls), 1)

    def test_middle_decode_fault_counts_remaining_rows_without_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='middle_decode')
            self.assertIsNotNone(error)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual([c[1] for c in runtime.session.calls if c[0] == 'step'], [0, 1])
            self.assertEqual(result['frames'][0]['measurement_state'], 'measured')
            self.assertEqual(result['frames'][1]['source_failure_reason'], 'rgb_decode_failed')
            self.assertEqual(result['frames'][2]['measurement_state'], 'unmeasured')
            self.assertIsNone(result['frames'][2]['path'])

    def test_initial_decode_fault_preserves_reason_without_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='initial_decode')
            self.assertIsNotNone(error)
            self.assertEqual(result['frames'][0]['source_failure_reason'], 'rgb_decode_failed')
            self.assertFalse(any(c[0] in ('step', 'selection', 'write') for c in runtime.session.calls))

    def test_no_detection_resumes_next_frame_without_new_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='no_detection')
            self.assertIsNone(error)
            self.assertEqual(result['frames'][1]['mask_state'], 'lost')
            self.assertEqual(result['frames'][2]['mask_state'], 'available')
            self.assertIn(('resume', 1), runtime.session.calls)
            self.assertEqual(sum(c[0] == 'selection' for c in runtime.session.calls), 1)
            self.assertFalse(any(c[0] == 'condition' for c in runtime.session.calls))

    def test_bad_detector_identity_or_shape_never_conditions_mask(self):
        for fault in ('bad_identity', 'bad_detector_shape'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                result, runtime, error = self.execute(Path(folder), fault=fault)
                self.assertIsNotNone(error)
                self.assertFalse(result['stage_completed'])
                self.assertFalse(any(c[0] == 'condition' for c in runtime.session.calls))
                self.assertIsNone(result['frames'][1]['path'])

    def test_nonzero_offloaded_gpu_bytes_prevents_detector_invocation(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='bad_offload')
            self.assertIsNotNone(error)
            self.assertFalse(any(c[0] == 'detector' for c in runtime.session.calls))
            self.assertFalse(result['stage_completed'])

    def test_malformed_writer_digest_and_missing_cleanup_flags_fail_stage(self):
        for fault in ('bad_writer_digest', 'incomplete_cleanup'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                result, runtime, error = self.execute(Path(folder), fault=fault)
                self.assertIsNotNone(error)
                self.assertFalse(result['stage_completed'])

    def test_boolean_step_ordinal_is_not_accepted_as_integer_one(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='boolean_ordinal')
            self.assertIsNotNone(error)
            self.assertIsNone(result['frames'][1]['path'])

    def test_completed_fake_cache_joins_public_pose_cache_validator(self):
        from . import quality_runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, runtime, error = self.execute(root)
            self.assertIsNone(error)
            manifest = quality_runner.read_input(root / 'input')
            cache, rows = quality_runner._physical_mask_cache(manifest, root / 'masks')
            self.assertEqual(set(rows), {0, 1, 2})
            self.assertEqual(cache['timestamp_table_sha256'], result['timestamp_table_sha256'])

    def test_each_detector_identity_field_is_authenticated(self):
        for field in ('frameId', 'sourceFrameId', 'timestamp_s', 'clock_mode',
                      'input_manifest_sha256', 'timestamp_table_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                result, runtime, error = self.execute(Path(folder), fault='identity:' + field)
                self.assertIsNotNone(error)
                self.assertFalse(any(c[0] in ('condition', 'restore') for c in runtime.session.calls))
                self.assertIsNone(result['frames'][1]['path'])

    def test_conditioning_failure_keeps_attempt_and_detector_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='condition_failure')
            self.assertIsNotNone(error)
            ledger = result['prompt_ledger']['automatic_mask_conditioning']
            self.assertEqual(len(ledger), 1)
            self.assertTrue(ledger[0]['attempted'])
            self.assertFalse(ledger[0]['returned'])
            row = result['frames'][1]
            self.assertEqual(row['operation_counts']['attempted']['automatic_conditioning'], 1)
            self.assertNotIn('automatic_conditioning', row['operation_counts']['returned'])
            self.assertEqual(row['timings_ms']['cnos'], 0.2)

    def test_automatic_recovery_conditions_once_and_keeps_initial_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            result, runtime, error = self.execute(Path(folder), fault='recovery')
            self.assertIsNone(error)
            self.assertEqual([c for c in runtime.session.calls if c[0] == 'condition'], [('condition', 1)])
            self.assertEqual(sum(c[0] == 'selection' for c in runtime.session.calls), 1)
            self.assertTrue(result['frames'][1]['automatic_conditioning'])

    def test_prefix_masks_equal_full_run_bytes(self):
        with tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as prefix:
            complete, _, error = self.execute(Path(full))
            partial, _, prefix_error = self.execute(Path(prefix), limit=1)
            self.assertIsNone(error)
            self.assertIsNone(prefix_error)
            self.assertEqual([r['mask_sha256'] for r in complete['frames'][:2]],
                             [r['mask_sha256'] for r in partial['frames'][:2]])


class SyntheticTensor:
    def __init__(self, device):
        self.device = SyntheticDevice(device)

    def detach(self):
        return self

    def to(self, device):
        return SyntheticTensor(str(device))

    def numel(self):
        return 4

    def element_size(self):
        return 4


class SyntheticDevice:
    def __init__(self, name):
        self.name, self.type = name, name.split(':')[0]

    def __str__(self):
        return self.name


class SAMTensorTransportTests(unittest.TestCase):
    def tensor(self, device):
        value = SyntheticTensor(device)
        value.device = SyntheticDevice(device)
        return value

    def test_roundtrip_preserves_cuda_pointer_cpu_storage_and_aliases(self):
        # obj_ptr must return to GPU, while intentionally offloaded output stays CPU.
        pointer, storage = self.tensor('cuda:0'), self.tensor('cpu')
        state = {'obj_ptr': pointer, 'again': [pointer], 'storage': storage}
        torch = SimpleNamespace(Tensor=SyntheticTensor)
        transport = {}
        producer._move_tensors(state, torch, 'cpu', transport=transport)
        self.assertIs(state['obj_ptr'], state['again'][0])
        self.assertEqual(producer._tensor_audit(state, torch)['tensor_count'], 2)
        restored = {}
        producer._restore_tensors(state, transport, torch, restored_devices=restored)
        self.assertIs(state['obj_ptr'], state['again'][0])
        self.assertEqual(str(state['obj_ptr'].device), 'cuda:0')
        self.assertEqual(str(state['storage'].device), 'cpu')
        self.assertEqual(producer._restored_device_audit(state, restored, torch)['device_mismatches'], 0)

    def test_unrecorded_tensor_cannot_enter_restored_state(self):
        torch = SimpleNamespace(Tensor=SyntheticTensor)
        with self.assertRaises(RuntimeError):
            producer._restore_tensors({'new': self.tensor('cuda:0')}, {}, torch)


class SAMSourceCausalityTests(unittest.TestCase):
    def source(self, *, seek=True, legacy=False):
        cap = SimpleNamespace(set=lambda prop, value: seek, read=lambda: self.fail('decoder must not run'),
                              release=lambda: None)
        cv = SimpleNamespace(VideoCapture=lambda path: cap, CAP_PROP_POS_FRAMES=1)
        plan = prepare_cache_plan(encode(capture_document()), limit=1)
        if legacy:
            plan = dict(plan, clock_mode=LEGACY)
        session = SimpleNamespace(np=np, cv2=cv, torch=object(), plan=plan,
                                  manifest={'native_resolution': [640, 480], 'video': 'fake'}, bundle=Path('.'))
        return producer._SourceFrames(session, 16)

    def test_future_reads_and_skipped_allowance_reject_before_decode(self):
        frames = self.source()
        with self.assertRaises(producer.CausalFrameAccessError):
            frames[1]
        with self.assertRaises(producer.CausalFrameAccessError):
            frames.allow(2)
        self.assertEqual(frames.metrics, {})

    def test_failed_seek_preserves_source_id_without_read(self):
        frames = self.source(seek=False)
        frames.allow(1)
        with self.assertRaises(producer.SourceDecodeError) as failure:
            frames[1]
        self.assertEqual(failure.exception.source_frame_id, 262)
        self.assertEqual(frames.snapshot(1)['decode_returned'], 0)
        self.assertEqual(frames.snapshot(1)['decode_failure'], 'rgb_decode_failed')

    def test_legacy_length_is_requested_prefix_physical_length_full_plan(self):
        self.assertEqual(len(self.source()), 3)
        self.assertEqual(len(self.source(legacy=True)), 2)


if __name__ == '__main__':
    unittest.main()
