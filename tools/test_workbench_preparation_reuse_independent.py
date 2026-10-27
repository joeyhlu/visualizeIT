"""Independent ownership/state controls; injected components are not a model."""
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType
import unittest

import numpy as np

from bench import workbench_neural as neural
from tools.test_workbench_preparation_reuse import PreparationRuntime


def compare_nested(test, a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            compare_nested(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            compare_nested(test, left, right)
    else:
        test.assertEqual(a, b)


class IndependentPreparationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the actual accepted parent implementation, not a second mode of
        # the candidate. Failure to obtain these exact bytes fails the control.
        root = Path(__file__).resolve().parents[1]
        raw = subprocess.check_output(['git', 'show',
            '1c4248c964ee0d10116d3628ee6059f76caa00f5:bench/workbench_neural.py'], cwd=root)
        if hashlib.sha256(raw).hexdigest() != 'ecac8a3e10590b425d691fdfb13fbcbbd4e8ea027350dbf5311cf045394ad045':
            raise AssertionError('Accepted parent source bytes differ')
        name = 'bench._preparation_test_accepted_parent'
        module = ModuleType(name)
        module.__file__, module.__package__ = neural.__file__, 'bench'
        sys.modules[name] = module
        exec(compile(raw, '<accepted-parent-1c4248c>', 'exec'), module.__dict__)
        cls.parent_module = module
        raw = subprocess.check_output(['git', 'show',
            'ce5c31f22bb3796d83c0ae57d381148cf55a4d84:bench/workbench_neural.py'], cwd=root)
        if hashlib.sha256(raw).hexdigest() != '96bcd0e4d4cd22a44b2522db4ab034a33051b638704505e5e26f9b5e998c33a2':
            raise AssertionError('Published P1-A source bytes differ')
        name = 'bench._preparation_test_published_p1a'
        module = ModuleType(name)
        module.__file__, module.__package__ = neural.__file__, 'bench'
        sys.modules[name] = module
        exec(compile(raw, '<published-p1a-ce5c31f>', 'exec'), module.__dict__)
        cls.published_module = module

    def make_parent_backend(self, runtime, *, published=False, enabled=False):
        parent = self.published_module if published else self.parent_module
        class CapturingParent(parent.WorkbenchIntegerRefiner):
            def trace(self, case):
                result = super().trace(case)
                runtime.traces.append(result)
                return result
        def renderer(case):
            runtime.calls.append('renderer')
            return SimpleNamespace(vertices_m=np.zeros((3, 3)))
        runtime.backend = parent.GoTrackMetricBackend('primary_neural', runtime.bundle,
            head_path=Path('unused.pt'), network_loader=lambda device: (runtime.network, runtime.provenance),
            refiner_factory=lambda network, device: CapturingParent(network, device, renderer_factory=renderer),
            **({'preparation_reuse': enabled} if published else {}))
        runtime.backend.seed_target_frame_id = 's000001'

    def test_exact_accepted_parent_and_both_candidate_arms_success_loss_error(self):
        for scenario in ('two-frame-success', 'loss', 'network-error'):
            records = []
            for arm in ('accepted-parent', 'published-disabled', 'published-enabled',
                        'candidate-disabled', 'candidate-enabled'):
                kwargs = {'valid_points': 23} if scenario == 'loss' else {'fail_iteration': 2} if scenario == 'network-error' else {}
                runtime = PreparationRuntime(**kwargs)
                if arm == 'accepted-parent':
                    self.make_parent_backend(runtime)
                elif arm.startswith('published-'):
                    self.make_parent_backend(runtime, published=True, enabled=arm == 'published-enabled')
                else:
                    runtime.make_backend(arm == 'candidate-enabled')
                state = {'_frame_id': 's000001'}
                outputs = []
                error = None
                with runtime.installed():
                    try:
                        outputs.append(runtime.estimate(state))
                        if scenario == 'two-frame-success':
                            state['_frame_id'] = 's000002'
                            outputs.append(runtime.estimate(state))
                    except RuntimeError as exc:
                        error = (type(exc).__name__, str(exc))
                records.append((runtime, state, outputs, error))
            reference = records[0]
            for actual in records[1:]:
                compare_nested(self, reference[1:], actual[1:])
                self.assertEqual(reference[0].calls, actual[0].calls)
                for expected_trace, actual_trace in zip(reference[0].traces, actual[0].traces):
                    compare_nested(self, expected_trace, actual_trace)
                self.assertEqual(len(reference[0].traces), len(actual[0].traces))
                for expected_inputs, actual_inputs in zip(reference[0].inputs, actual[0].inputs):
                    compare_nested(self, expected_inputs, actual_inputs)
            expected_calls = 10 if scenario == 'two-frame-success' else 1 if scenario == 'loss' else 3
            for runtime, _, _, _ in records:
                self.assertEqual(runtime.calls.count('network'), expected_calls)
                self.assertEqual(runtime.calls.count('crop'), expected_calls)
            self.assertEqual([x[0].rgb_preparations for x in records],
                             [10, 10, 2, 10, 2] if scenario == 'two-frame-success' else
                             [expected_calls, expected_calls, 1, expected_calls, 1])
            self.assertEqual([x[0].mask_preparations for x in records],
                             [10, 10, 2, 10, 2] if scenario == 'two-frame-success' else
                             [expected_calls, expected_calls, 1, expected_calls, 1])

    def test_tensor_creation_observer_changes_match_published_parent(self):
        # A tensor observer may mutate the live record/table while the original
        # restore is returning. Capturing both records or trusting post-callback
        # metadata early would hide these parent-visible changes.
        for mutation in ('digest', 'storage-from-mask-callback', 'valid-replacement', 'table-none',
                         'table-none-from-mask-callback', 'table-short-from-mask-callback',
                         'delete-digest-from-mask-callback'):
            records = []
            for published in (True, False):
                runtime = PreparationRuntime()
                if published:
                    self.make_parent_backend(runtime, published=True, enabled=True)
                else:
                    runtime.make_backend(True)
                original_tensor = runtime.tensor
                fired = []
                def bridge(value):
                    result = original_tensor(value)
                    frame = runtime.backend.refiner._prepared_frame
                    target = (1, 5, 6) if mutation.endswith('from-mask-callback') else (1, 3, 5, 6)
                    if frame is not None and np.asarray(value).shape == target and not fired:
                        fired.append(True)
                        if mutation == 'digest':
                            object.__setattr__(frame[1], 'digest', 'corrupt')
                        elif mutation == 'storage-from-mask-callback':
                            data = bytearray(frame[1].storage); data[0] ^= 1
                            object.__setattr__(frame[1], 'storage', bytes(data))
                        elif mutation == 'valid-replacement':
                            fresh = replace(frame[1], storage=memoryview(frame[1].storage).tobytes())
                            runtime.backend.refiner._prepared_frame = (frame[0], fresh, frame[2])
                        elif mutation == 'table-short-from-mask-callback':
                            runtime.backend.refiner._prepared_frame = (frame[0],)
                        elif mutation == 'delete-digest-from-mask-callback':
                            object.__delattr__(frame[1], 'digest')
                        else:
                            runtime.backend.refiner._prepared_frame = None
                    return result
                runtime.tensor = bridge
                state = {'_frame_id': 's000001'}
                error = None
                output = None
                with runtime.installed():
                    try:
                        output = runtime.estimate(state)
                    except (ValueError, TypeError, IndexError, AttributeError) as exc:
                        error = (type(exc).__name__, str(exc))
                self.assertEqual(fired, [True])
                self.assertIsNone(runtime.backend.refiner._prepared_frame)
                records.append((runtime, state, output, error))
            with self.subTest(mutation=mutation):
                compare_nested(self, records[0][1:], records[1][1:])
                self.assertEqual(records[0][0].calls, records[1][0].calls)
                self.assertEqual(records[1][0].calls.count('network'),
                                 5 if mutation in ('valid-replacement', 'table-none-from-mask-callback')
                                 else 0 if mutation == 'table-none' else 1)

    def test_live_validation_policy_changes_keep_parent_refusal_and_cleanup(self):
        for mutation in ('cap', 'class-digest', 'class-length', 'instance-digest', 'instance-length'):
            records = []
            for published in (True, False):
                module = self.published_module if published else neural
                original_cap = module.FRAME_PREPARATION_MAX_TENSOR_BYTES
                original_digest = module._FrameTensorBytes.__dict__['_digest']
                original_length = module._FrameTensorBytes.__dict__['_length']
                runtime = PreparationRuntime()
                if published:
                    self.make_parent_backend(runtime, published=True, enabled=True)
                else:
                    runtime.make_backend(True)
                original_tensor = runtime.tensor
                fired = []
                def fail(*args):
                    raise RuntimeError('injected validation policy changed')
                def bridge(value):
                    result = original_tensor(value)
                    frame = runtime.backend.refiner._prepared_frame
                    if frame is not None and np.asarray(value).shape == (1, 5, 6) and not fired:
                        fired.append(True)
                        if mutation == 'cap':
                            module.FRAME_PREPARATION_MAX_TENSOR_BYTES = 1
                        elif mutation.startswith('class-'):
                            setattr(module._FrameTensorBytes, '_'+mutation.split('-')[1], staticmethod(fail))
                        else:
                            object.__setattr__(frame[1], '_'+mutation.split('-')[1], fail)
                    return result
                runtime.tensor = bridge
                state = {'_frame_id': 's000001'}
                error = None
                try:
                    with runtime.installed():
                        try:
                            runtime.estimate(state)
                        except (ValueError, RuntimeError) as exc:
                            error = (type(exc).__name__, str(exc))
                finally:
                    module.FRAME_PREPARATION_MAX_TENSOR_BYTES = original_cap
                    module._FrameTensorBytes._digest = original_digest
                    module._FrameTensorBytes._length = original_length
                self.assertEqual(fired, [True])
                self.assertIsNotNone(error)
                self.assertEqual(runtime.calls.count('network'), 1)
                self.assertIsNone(runtime.backend.refiner._prepared_frame)
                self.assertIsNone(getattr(runtime.backend.refiner, '_prepared_seals', None))
                records.append((runtime, state, error))
            with self.subTest(mutation=mutation):
                compare_nested(self, records[0][1:], records[1][1:])
                self.assertEqual(records[0][0].calls, records[1][0].calls)

    def test_retained_bytes_own_values_and_restore_positive_strided_layout(self):
        # A gapped, permuted float32 layout exercises real storage bounds rather
        # than assuming every tensor is contiguous channel-first memory.
        source = np.arange(240, dtype=np.float32).reshape(10, 8, 3)[::2, ::2].transpose(2, 0, 1)[None]
        expected = source.copy()
        record = neural._FrameTensorBytes.freeze(SimpleNamespace(numpy=lambda: source))
        self.assertIs(type(record.storage), bytes)
        self.assertTrue(memoryview(record.storage).readonly)
        with self.assertRaises(ValueError):
            np.frombuffer(record.storage, dtype=np.uint8).setflags(write=True)
        source[...] = -1
        runtime = SimpleNamespace(from_numpy=lambda value: value)
        first = record.restore(runtime)
        self.assertEqual(first.strides, source.strides)
        self.assertEqual(first.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(first, expected)
        first[...] = -999
        second = record.restore(runtime)
        np.testing.assert_array_equal(second, expected)
        self.assertFalse(np.shares_memory(first, second))

    def test_corrupted_storage_refuses_before_tensor_creation(self):
        value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        record = neural._FrameTensorBytes.freeze(SimpleNamespace(numpy=lambda: value))
        calls = []
        runtime = SimpleNamespace(from_numpy=lambda array: calls.append(array))
        damaged = bytearray(record.storage); damaged[0] ^= 1
        with self.assertRaises(ValueError):
            replace(record, storage=bytes(damaged)).restore(runtime)
        self.assertEqual(calls, [])

    def test_actual_backend_reaches_tracking_on_next_frame_with_exact_parity(self):
        outcomes, states, runtimes = [], [], []
        for enabled in (False, True):
            runtime = PreparationRuntime()
            runtime.make_backend(enabled)
            state = {'_frame_id': 's000001', '_state_epoch': 'operator-test-epoch'}
            with runtime.installed():
                first = runtime.estimate(state)
                state['_frame_id'] = 's000002'
                second = runtime.estimate(state)
            self.assertEqual(first['status'], 'recovering')
            self.assertEqual(second['status'], 'tracking')
            self.assertEqual(runtime.calls.count('network'), 10)
            self.assertEqual(runtime.calls.count('crop'), 10)
            self.assertEqual(runtime.calls.count('correspondences'), 10)
            self.assertEqual([x for x in runtime.calls if isinstance(x, tuple) and x[0] == 'seed'],
                             [('seed', i) for i in range(5)]*2)
            self.assertEqual(runtime.rgb_preparations, 2 if enabled else 10)
            self.assertEqual(runtime.mask_preparations, 2 if enabled else 10)
            self.assertIsNone(runtime.backend.refiner._prepared_frame)
            outcomes.append((first, second)); states.append(state); runtimes.append(runtime)
        compare_nested(self, outcomes[0], outcomes[1])
        compare_nested(self, states[0], states[1])
        self.assertEqual(runtimes[0].calls, runtimes[1].calls)

    def test_separate_backend_instances_keep_mutable_preparation_isolated(self):
        left, right = PreparationRuntime(), PreparationRuntime()
        left.make_backend(True); right.make_backend(True)
        right.rgb[:] = 211
        left_original = left.rgb.copy(); right_original = right.rgb.copy()
        for runtime in (left, right, left):
            state = {'_frame_id': 's000001', 'prior_pose': runtime.initial_pose.copy(), 'consecutive': 1}
            with runtime.installed():
                runtime.estimate(state)
            self.assertIsNone(runtime.backend.refiner._prepared_frame)
        self.assertEqual(left.rgb_preparations, 2)
        self.assertEqual(right.rgb_preparations, 1)
        np.testing.assert_array_equal(left.rgb, left_original)
        np.testing.assert_array_equal(right.rgb, right_original)
        self.assertNotEqual(left.inputs[0][0][0].tobytes(), right.inputs[0][0][0].tobytes())


if __name__ == '__main__':
    unittest.main()
