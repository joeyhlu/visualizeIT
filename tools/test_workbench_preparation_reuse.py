"""Model-free tests of estimate-local preparation reuse in the actual backend."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import hashlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import bench
from bench import workbench_neural as neural


class PreparationTensor:
    def __init__(self, value, runtime, label=None):
        self.value = np.asarray(value)
        self.runtime, self.label = runtime, label

    def __getitem__(self, index):
        return PreparationTensor(self.value[index], self.runtime, self.label)

    def permute(self, *axes):
        return PreparationTensor(self.value.transpose(axes), self.runtime, self.label)

    def float(self):
        if self.label == "mask" and self.value.dtype == bool:
            self.runtime.mask_preparations += 1
        return PreparationTensor(self.value.astype(np.float32, order="K"), self.runtime, self.label)

    def __truediv__(self, divisor):
        if self.label == "rgb":
            self.runtime.rgb_preparations += 1
        return PreparationTensor(self.value / divisor, self.runtime, self.label)

    def to(self, device):
        self.runtime.calls.append(("to", device))
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class PreparationRuntime:
    """Injected CPU-only runtime; no Torch, cv2, rendering or model execution."""
    def __init__(self, *, valid_points=30, fail_iteration=None, mutate_inputs=False, after_trace=None):
        self.valid_points = valid_points
        self.fail_iteration, self.mutate_inputs = fail_iteration, mutate_inputs
        self.after_trace = after_trace
        self.rgb_preparations = self.mask_preparations = 0
        self.calls, self.inputs, self.poses, self.traces = [], [], [], []
        self.rgb = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
        self.mask = np.arange(30).reshape(5, 6) % 3 != 0
        self.K = np.array([[100., 0., 2.], [0., 120., 2.], [0., 0., 1.]])
        self.initial_pose = np.eye(4)
        self.initial_pose[2, 3] = 1
        self.bundle = neural.MetricBundle("stub", Path("stub.glb"), "stub-sha", 1, "integer_centers",
                                           self.initial_pose.copy(), "onboarding:stub", "s000000",
                                           "measured_metric_cad")
        self.provenance = {"head": "primary27e", "base": "stub-base"}
        self.backend = None

    def tensor(self, value):
        value = np.asarray(value)
        label = "rgb" if value.dtype == np.uint8 else "mask" if value.dtype == bool else None
        return PreparationTensor(value, self, label)

    @contextmanager
    def inference_mode(self):
        self.calls.append("inference_enter")
        try:
            yield
        finally:
            self.calls.append("inference_exit")

    def crop(self, **kwargs):
        self.calls.append("crop")
        images, masks = kwargs["input_rgbs"].value, kwargs["input_masks"].value
        self.inputs.append(tuple((value.copy(), value.dtype.str, value.shape, value.strides)
                                 for value in (images, masks)))
        self.poses.append(kwargs["init_poses_cam_from_model"].value.copy())
        template = SimpleNamespace(rgbs=self.tensor(images.copy()), masks=self.tensor(masks.copy()),
                                   depths=self.tensor(np.ones((1, 5, 6), np.float32)))
        data = {"templates": template, "crop_rgbs": self.tensor(images.copy()),
                "crop_masks": self.tensor(masks.copy())}
        if self.mutate_inputs:
            images[...] = -999
            masks[...] = -999
        return data, [None], [np.eye(4)]

    def network(self, *args):
        self.calls.append("network")
        if self.fail_iteration == len(self.poses) - 1:
            raise RuntimeError("injected network error")
        return self.tensor(np.zeros((1, 2, 5, 6), np.float32)), self.tensor(np.ones((1, 5, 6), np.float32))

    def correspondences(self, depths, crop_k, transform, pose_mm, *args):
        self.calls.append("correspondences")
        valid = np.arange(30).reshape(5, 6) < self.valid_points
        return (pose_mm.copy(), np.ones((5, 6, 3)), np.ones((5, 6, 2)), np.ones((5, 6, 2)),
                None, valid, None, None)

    def rodrigues(self, value):
        self.calls.append("rodrigues")
        return (np.zeros((3, 1)) if value.shape == (3, 3) else np.eye(3)), None

    def ransac(self, *args, **kwargs):
        self.calls.append(("ransac", kwargs))
        return True, np.zeros((3, 1)), args[5].reshape(3, 1) + [[0], [0], [1]], np.arange(24).reshape(-1, 1)

    def refine(self, *args):
        self.calls.append("refine")
        return args[-2:]

    def make_backend(self, enabled=False, *, implementation=None, **kwargs):
        module = implementation or neural
        runtime = self
        class CapturingRefiner(module.WorkbenchIntegerRefiner):
            def trace(self, case):
                result = super().trace(case)
                runtime.traces.append(result)
                if runtime.after_trace is not None:
                    runtime.after_trace(runtime, self)
                return result
        def renderer(case):
            self.calls.append("renderer")
            return SimpleNamespace(vertices_m=np.zeros((3, 3)))
        policy = {"preparation_reuse": enabled} if hasattr(module, "_FrameTensorBytes") else {}
        self.backend = module.GoTrackMetricBackend(
            "primary_neural", self.bundle, head_path=Path("unused.pt"),
            network_loader=lambda device: (self.network, self.provenance),
            refiner_factory=lambda net, dev: CapturingRefiner(net, dev, renderer_factory=renderer),
            **policy, **kwargs)
        self.backend.seed_target_frame_id = "s000001"
        return self.backend

    @contextmanager
    def installed(self):
        qg = SimpleNamespace(upstream_path=lambda: self.calls.append("upstream"),
                             _integer_profile_correspondences=self.correspondences)
        cv2 = SimpleNamespace(setRNGSeed=lambda seed: self.calls.append(("seed", seed)),
                              Rodrigues=self.rodrigues, solvePnPRansac=self.ransac,
                              solvePnPRefineLM=self.refine, SOLVEPNP_ITERATIVE=0)
        utils = SimpleNamespace(data_util=SimpleNamespace(compute_gotrack_inputs_from_init_poses=self.crop),
                                misc=SimpleNamespace(get_intrinsic_matrix=lambda camera: np.eye(3)),
                                structs=SimpleNamespace(PinholePlaneCameraModel=lambda **kwargs: kwargs))
        modules = {"torch": SimpleNamespace(from_numpy=self.tensor, inference_mode=self.inference_mode),
                   "utils": utils, "bench.vision": SimpleNamespace(cv2=cv2), "bench.quality_gotrack": qg}
        with patch.dict(sys.modules, modules), patch.object(bench, "quality_gotrack", qg, create=True):
            yield self

    def estimate(self, state=None):
        if state is None:
            state = {"_frame_id": "s000001"}
        return self.backend.estimate(self.rgb, self.mask, self.K, state)


@contextmanager
def accepted_neural(revision):
    """Execute the actual pinned Git source as a separate, temporary module."""
    expected = {
        "1c4248c964ee0d10116d3628ee6059f76caa00f5":
            "ecac8a3e10590b425d691fdfb13fbcbbd4e8ea027350dbf5311cf045394ad045",
        "ce5c31f22bb3796d83c0ae57d381148cf55a4d84":
            "96bcd0e4d4cd22a44b2522db4ab034a33051b638704505e5e26f9b5e998c33a2",
    }
    source = subprocess.check_output(["git", "show", revision + ":bench/workbench_neural.py"],
                                     cwd=Path(__file__).resolve().parents[1])
    if hashlib.sha256(source).hexdigest() != expected[revision]:
        raise AssertionError("accepted neural source differs from its pinned blob")
    name = "bench._preparation_control_" + revision[:8]
    module = ModuleType(name)
    module.__file__ = str(Path(neural.__file__).resolve())
    module.__package__ = "bench"
    with patch.dict(sys.modules, {name: module}):
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        yield module


class PreparationOperations:
    """Observe actual source operations, never wall time or hardware traffic.

    SHA metadata headers and canonical context bytes are counted separately and
    excluded from payload bytes. Copy counts cover source hash .tobytes(), frozen
    buffer writes/bytes(), and restored bytearray copies. They exclude fake
    runtime copies, Torch work, allocator overhead and all crop/model activity.
    """
    def __init__(self, module, runtime):
        self.module, self.runtime = module, runtime
        self.kind = None
        self.counts = {key: 0 for key in (
            "sha_constructions", "sha_payload_bytes", "sha_metadata_bytes",
            "prepared_digest_calls", "prepared_digest_bytes", "array_digest_calls", "array_digest_bytes",
            "canonical_sha_calls", "canonical_json_calls", "freeze_calls", "working_restores",
            "torch_bridges", "explicit_copy_bytes", "zero_fill_bytes")}
        self.stack = ExitStack()

    def __enter__(self):
        module, runtime, counts = self.module, self.runtime, self.counts
        original_sha = module.hashlib.sha256
        original_array = module.array_digest
        original_freeze = module._FrameTensorBytes.freeze
        original_canonical_sha = module.wc.canonical_sha256
        original_canonical_json = module.wc.canonical_json
        original_ndarray = np.ndarray
        original_contiguous = np.ascontiguousarray
        observer = self

        class ObservedHash:
            def __init__(self, value=b"", *args, **kwargs):
                counts["sha_constructions"] += 1
                counts["sha_metadata_bytes"] += len(value)
                # Accepted source uses repr((shape, strides, dtype)) as its
                # prepared header. Observe the actual hash without replacing
                # either purity-critical validation helper.
                self.kind = ("prepared" if observer.kind is None and value.startswith((b"((", b"(["))
                             else observer.kind)
                if self.kind == "prepared":
                    counts["prepared_digest_calls"] += 1
                self.inner = original_sha(value, *args, **kwargs)

            def update(self, value):
                if self.kind in ("array", "prepared"):
                    counts["sha_payload_bytes"] += len(value)
                    if self.kind == "prepared":
                        counts["prepared_digest_bytes"] += len(value)
                else:
                    counts["sha_metadata_bytes"] += len(value)
                self.inner.update(value)

            def hexdigest(self):
                return self.inner.hexdigest()

        @contextmanager
        def operation(kind):
            previous, observer.kind = observer.kind, kind
            try:
                yield
            finally:
                observer.kind = previous

        def array_digest(value):
            counts["array_digest_calls"] += 1
            counts["array_digest_bytes"] += np.asarray(value).nbytes
            with operation("array"):
                return original_array(value)

        def canonical_sha(value):
            counts["canonical_sha_calls"] += 1
            with operation("canonical"):
                return original_canonical_sha(value)

        def canonical_json(value):
            counts["canonical_json_calls"] += 1
            return original_canonical_json(value)

        def freeze(cls, tensor):
            counts["freeze_calls"] += 1
            return original_freeze(tensor)

        class ObservedBytearray(bytearray):
            def __init__(self, value):
                super().__init__(value)
                self.from_length = type(value) is int
                if self.from_length:
                    counts["zero_fill_bytes"] += len(self)
                else:
                    counts["working_restores"] += 1
                    counts["explicit_copy_bytes"] += len(self)

            def __bytes__(self):
                result = memoryview(self).tobytes()
                counts["explicit_copy_bytes"] += len(result)
                return result

        class ObservedArray(original_ndarray):
            def __array_finalize__(self, source):
                self.frozen_write = getattr(source, "frozen_write", False)

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if self.frozen_write:
                    counts["explicit_copy_bytes"] += self.nbytes

            def tobytes(self, *args, **kwargs):
                result = super().tobytes(*args, **kwargs)
                if observer.kind == "array":
                    counts["explicit_copy_bytes"] += len(result)
                return result

        def ndarray(*args, **kwargs):
            result = original_ndarray(*args, **kwargs).view(ObservedArray)
            buffer = kwargs.get("buffer")
            result.frozen_write = isinstance(buffer, ObservedBytearray) and buffer.from_length
            return result

        def contiguous(*args, **kwargs):
            result = original_contiguous(*args, **kwargs)
            return result.view(ObservedArray) if observer.kind == "array" else result

        original_bridge = sys.modules["torch"].from_numpy
        def bridge(value):
            counts["torch_bridges"] += 1
            return original_bridge(value)

        for target, name, replacement in (
                (module.hashlib, "sha256", ObservedHash), (module, "array_digest", array_digest),
                (module._FrameTensorBytes, "freeze", classmethod(freeze)),
                (module.wc, "canonical_sha256", canonical_sha), (module.wc, "canonical_json", canonical_json),
                (module, "bytearray", ObservedBytearray), (np, "ndarray", ndarray),
                (np, "ascontiguousarray", contiguous), (sys.modules["torch"], "from_numpy", bridge)):
            self.stack.enter_context(patch.object(target, name, replacement, create=True))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


class PreparationReuseTests(unittest.TestCase):
    def assert_runtime_equal(self, left, right):
        self.assertEqual(left.calls, right.calls)
        self.assertEqual(len(left.traces), len(right.traces))
        for a, b in zip(left.traces, right.traces):
            self.assertTrue(neural.compare_traces(a, b)["passed"])
        for a, b in zip(left.poses, right.poses):
            np.testing.assert_array_equal(a, b)
        for inputs_a, inputs_b in zip(left.inputs, right.inputs):
            for a, b in zip(inputs_a, inputs_b):
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:], b[1:])

    def test_actual_backend_five_iterations_identical_values_layout_calls_rng_and_outcome(self):
        results, runtimes = [], []
        for enabled in (False, True):
            runtime = PreparationRuntime()
            runtime.make_backend(enabled)
            with runtime.installed():
                results.append(runtime.estimate())
            runtimes.append(runtime)
        self.assertEqual(results[0], results[1])
        self.assert_runtime_equal(*runtimes)
        for runtime in runtimes:
            self.assertEqual(runtime.calls.count("network"), 5)
            self.assertEqual(runtime.calls.count("crop"), 5)
            self.assertEqual(runtime.calls.count("correspondences"), 5)
            self.assertEqual([entry for entry in runtime.calls if isinstance(entry, tuple) and entry[0] == "seed"],
                             [("seed", i) for i in range(5)])
            self.assertEqual(len({pose.tobytes() for pose in runtime.poses}), 5)
            self.assertIsNone(runtime.backend.refiner._prepared_frame)
            self.assertIsNone(runtime.backend.refiner._preparation_scope)
        self.assertEqual([runtime.rgb_preparations for runtime in runtimes], [5, 1])
        self.assertEqual([runtime.mask_preparations for runtime in runtimes], [5, 1])

    def test_next_frame_even_identical_bytes_and_reset_never_reuses_record(self):
        runtime = PreparationRuntime()
        runtime.make_backend(True)
        with runtime.installed():
            state = {"_frame_id": "s000001"}
            runtime.estimate(state)
            runtime.estimate(state)
            runtime.backend.reset()
            runtime.estimate(state)
        self.assertEqual(runtime.rgb_preparations, 3)
        self.assertEqual(runtime.mask_preparations, 3)
        self.assertEqual(runtime.calls.count("network"), 15)

    def test_mutable_upstream_inputs_cannot_poison_reuse(self):
        runtimes = []
        for enabled in (False, True):
            runtime = PreparationRuntime(mutate_inputs=True)
            runtime.make_backend(enabled)
            with runtime.installed():
                runtime.estimate()
            runtimes.append(runtime)
        self.assert_runtime_equal(*runtimes)
        for images, masks in runtimes[1].inputs:
            self.assertGreaterEqual(float(images[0].min()), 0)
            self.assertGreaterEqual(float(masks[0].min()), 0)

    def test_loss_and_runtime_errors_propagate_with_cleanup(self):
        for kwargs in ({"valid_points": 23}, {"fail_iteration": 2}):
            runtimes = []
            results = []
            for enabled in (False, True):
                runtime = PreparationRuntime(**kwargs)
                runtime.make_backend(enabled)
                state = {"_frame_id": "s000001"}
                with runtime.installed():
                    if "fail_iteration" in kwargs:
                        with self.assertRaisesRegex(RuntimeError, "injected network error"):
                            runtime.estimate(state)
                    else:
                        results.append(runtime.estimate(state))
                        self.assertEqual(state, {})
                self.assertIsNone(runtime.backend.refiner._prepared_frame)
                self.assertIsNone(runtime.backend.refiner._preparation_scope)
                runtimes.append(runtime)
            self.assert_runtime_equal(*runtimes)
            if results:
                self.assertEqual(results[0], results[1])

    def test_live_invariant_mutations_recompute_instead_of_serving_stale_inputs(self):
        mutations = {
            "rgb": lambda r: r.rgb.__setitem__((0, 0, 0), 255),
            "mask": lambda r: r.mask.__setitem__((0, 0), not r.mask[0, 0]),
            "rgb_dtype": lambda r: setattr(r.rgb, "dtype", np.int8),
            "mask_dtype": lambda r: setattr(r.mask, "dtype", np.uint8),
            "dimensions": lambda r: setattr(r.rgb, "shape", (6, 5, 3)),
            "K": lambda r: r.K.__setitem__((0, 0), 200),
            "origin": lambda r: setattr(r.backend, "bundle", replace(r.backend.bundle, principal_point_origin="corner")),
            "surface": lambda r: setattr(r.backend, "bundle", replace(r.backend.bundle, glb_sha256="changed")),
            "head": lambda r: r.backend._provenance.__setitem__("head", "changed"),
            "network": lambda r: setattr(r.backend.refiner, "network", lambda *args: r.network(*args)),
            "device": lambda r: setattr(r.backend.refiner, "device", "declared-device"),
            "backend_device": lambda r: setattr(r.backend, "device", "declared-device"),
            "role": lambda r: setattr(r.backend, "role", "changed-role"),
            "policy": lambda r: setattr(r.backend, "min_inliers", 23),
            "reset": lambda r: r.backend.reset(),
        }
        for name, mutation in mutations.items():
            def after(runtime, refiner):
                if len(runtime.traces) == 1:
                    mutation(runtime)
            with self.subTest(name=name):
                runtime = PreparationRuntime(after_trace=after)
                runtime.make_backend(True)
                with runtime.installed():
                    runtime.estimate()
                # K is converted once by the backend, so changing the source K
                # cannot affect this estimate's already fixed native_K.
                self.assertEqual(runtime.mask_preparations, 1 if name == "K" else 2)

    def test_frame_id_epoch_and_native_K_changes_within_scope_recompute(self):
        runtime = PreparationRuntime()
        backend = runtime.make_backend(True)
        state = {"_frame_id": "s000001"}
        case = {"rgb": runtime.rgb, "mask": runtime.mask, "native_K": runtime.K.copy(),
                "initial_pose_m": runtime.initial_pose, "glb_path": "stub.glb", "glb_sha256": "stub-sha", "obj_id": 1}
        with runtime.installed(), backend.refiner.preparation_scope(lambda: backend._preparation_context(state)):
            for mutate in (lambda: None, lambda: state.update(_frame_id="s000002"),
                           lambda: state.update(_state_epoch=2), lambda: case["native_K"].__setitem__((0, 0), 201)):
                mutate()
                backend.refiner.trace(case)
        self.assertEqual(runtime.rgb_preparations, 4)

    def test_corrupt_retained_bytes_are_refused_and_scope_cleans_up(self):
        def corrupt(runtime, refiner):
            if len(runtime.traces) == 1:
                record = refiner._prepared_frame
                refiner._prepared_frame = (record[0], replace(record[1], storage=b"bad"), record[2])
        runtime = PreparationRuntime(after_trace=corrupt)
        runtime.make_backend(True)
        with runtime.installed(), self.assertRaisesRegex(ValueError, "integrity check failed"):
            runtime.estimate()
        self.assertEqual(runtime.calls.count("network"), 1)
        self.assertIsNone(runtime.backend.refiner._prepared_frame)
        self.assertIsNone(runtime.backend.refiner._preparation_scope)

    def test_default_custom_refiner_and_policy_metadata(self):
        runtime = PreparationRuntime()
        default = runtime.make_backend()
        self.assertNotIn("frame_preparation", default.limits())
        self.assertNotIn("frame_preparation", default.identity_extras(None))
        enabled = runtime.make_backend(True)
        self.assertEqual(enabled.limits()["frame_preparation"], "experimental_estimate_local_immutable_bytes_v1")
        self.assertIn("frame_preparation", enabled.identity_extras(None))
        kwargs = {"network_loader": lambda device: (runtime.network, runtime.provenance),
                  "refiner_factory": lambda n, d: SimpleNamespace(trace=lambda case: {})}
        neural.GoTrackMetricBackend("frozen_baseline", runtime.bundle, **kwargs)
        with self.assertRaisesRegex(ValueError, "requires WorkbenchIntegerRefiner"):
            neural.GoTrackMetricBackend("frozen_baseline", runtime.bundle, preparation_reuse=True, **kwargs)
        with self.assertRaises(TypeError):
            neural.GoTrackMetricBackend("frozen_baseline", runtime.bundle, preparation_reuse=1, **kwargs)

    def test_owned_storage_is_immutable_and_layout_limits_are_checked(self):
        runtime = PreparationRuntime()
        original = np.arange(90, dtype=np.float32).reshape(5, 6, 3).transpose(2, 0, 1)[None]
        record = neural._FrameTensorBytes.freeze(runtime.tensor(original))
        self.assertIs(type(record.storage), bytes)
        with self.assertRaises(TypeError):
            record.storage[0] = 0
        restored = record.restore(SimpleNamespace(from_numpy=runtime.tensor)).value
        self.assertEqual(restored.strides, original.strides)
        np.testing.assert_array_equal(restored, original)
        restored[...] = -9
        np.testing.assert_array_equal(record.restore(SimpleNamespace(from_numpy=runtime.tensor)).value, original)
        with self.assertRaises(ValueError):
            neural._FrameTensorBytes.freeze(runtime.tensor(original.astype(np.float64)))
        with self.assertRaises(ValueError):
            neural._FrameTensorBytes._length((100000000,), (4,), "<f4")
        for broken in (replace(record, dtype="<f8"), replace(record, shape=(3, 5, 6)),
                       replace(record, strides=(0, 1, 2, 3)), replace(record, digest="bad")):
            with self.assertRaises(ValueError):
                broken.restore(SimpleNamespace(from_numpy=runtime.tensor))

    def test_actual_operation_counts_against_loaded_ce5_on_640_by_480_inputs(self):
        def observe(module):
            runtime = PreparationRuntime()
            runtime.rgb = np.arange(640 * 480 * 3, dtype=np.uint8).reshape(480, 640, 3)
            runtime.mask = np.ones((480, 640), bool)
            runtime.make_backend(True, implementation=module)
            with runtime.installed(), PreparationOperations(module, runtime) as operations:
                result = runtime.estimate()
            self.assertEqual(result["status"], "recovering")
            self.assertEqual(runtime.calls.count("network"), 5)
            self.assertEqual(runtime.calls.count("crop"), 5)
            self.assertEqual(runtime.calls.count("correspondences"), 5)
            self.assertEqual(sum(isinstance(call, tuple) and call[0] == "ransac" for call in runtime.calls), 5)
            return operations.counts
        with accepted_neural("ce5c31f22bb3796d83c0ae57d381148cf55a4d84") as control:
            old = observe(control)
        new = observe(neural)
        self.assertEqual((old["sha_constructions"], new["sha_constructions"]), (32, 24))
        self.assertEqual((old["sha_payload_bytes"], new["sha_payload_bytes"]), (35635560, 15974760))
        self.assertEqual((old["prepared_digest_calls"], new["prepared_digest_calls"]), (12, 4))
        self.assertEqual(old["prepared_digest_bytes"] - new["prepared_digest_bytes"], 19660800)
        unchanged = {"array_digest_calls": 15, "array_digest_bytes": 6144360,
                     "canonical_sha_calls": 5, "canonical_json_calls": 5, "freeze_calls": 2,
                     "working_restores": 10, "torch_bridges": 17,
                     "explicit_copy_bytes": 40550760, "zero_fill_bytes": 4915200}
        for name, expected in unchanged.items():
            with self.subTest(operation=name):
                self.assertEqual(old[name], expected)
                self.assertEqual(new[name], expected)

    def test_first_use_full_validation_refuses_each_corrupt_field(self):
        mutations = {
            "storage": lambda record: replace(record, storage=bytes(len(record.storage))),
            "shape": lambda record: replace(record, shape=(99,)),
            "strides": lambda record: replace(record, strides=tuple(value + 4 for value in record.strides)),
            "dtype": lambda record: replace(record, dtype="<f8"),
            "digest": lambda record: replace(record, digest="bad"),
        }
        for name, corrupt in mutations.items():
            with self.subTest(field=name):
                runtime = PreparationRuntime()
                runtime.make_backend(True)
                original = neural._FrameTensorBytes.freeze
                calls = []
                def freeze(cls, tensor):
                    result = original(tensor)
                    calls.append(result)
                    return corrupt(result) if len(calls) == 1 else result
                with runtime.installed(), patch.object(neural._FrameTensorBytes, "freeze", classmethod(freeze)), \
                        self.assertRaises(ValueError):
                    runtime.estimate()
                self.assertEqual(runtime.calls.count("network"), 0)
                self.assertIsNone(runtime.backend.refiner._prepared_seals)

    def test_after_first_use_corruption_fails_before_next_crop(self):
        for field, value in (("storage", b"bad"), ("shape", (99,)), ("strides", (1,)),
                             ("dtype", "<f8"), ("digest", "bad")):
            def after(runtime, refiner):
                if len(runtime.traces) == 1:
                    object.__setattr__(refiner._prepared_frame[1], field, value)
            with self.subTest(field=field):
                runtime = PreparationRuntime(after_trace=after)
                runtime.make_backend(True)
                with runtime.installed(), self.assertRaises(ValueError):
                    runtime.estimate()
                self.assertEqual(runtime.calls.count("network"), 1)
                self.assertIsNone(runtime.backend.refiner._prepared_seals)

    def test_valid_record_and_equal_field_replacements_require_full_revalidation(self):
        changes = {
            "record": lambda record: replace(record),
            "storage": lambda record: replace(record, storage=bytes(bytearray(record.storage))),
            "shape": lambda record: replace(record, shape=tuple(list(record.shape))),
            "strides": lambda record: replace(record, strides=tuple(list(record.strides))),
            "dtype": lambda record: replace(record, dtype=(" " + record.dtype)[1:]),
            "digest": lambda record: replace(record, digest=(" " + record.digest)[1:]),
        }
        for name, change in changes.items():
            def after(runtime, refiner):
                if len(runtime.traces) == 1:
                    previous = refiner._prepared_frame
                    record = change(previous[1])
                    if name != "record":
                        # Replace the field on the same exact record, exercising
                        # field identity rather than only record identity.
                        object.__setattr__(previous[1], name, getattr(record, name))
                    else:
                        refiner._prepared_frame = previous[0], record, previous[2]
            with self.subTest(field=name):
                runtime = PreparationRuntime(after_trace=after)
                runtime.make_backend(True)
                with runtime.installed(), PreparationOperations(neural, runtime) as operations:
                    runtime.estimate()
                self.assertEqual(operations.counts["prepared_digest_calls"], 5)
                self.assertEqual(runtime.calls.count("network"), 5)
                self.assertIsNone(runtime.backend.refiner._prepared_seals)

    def test_subclasses_mutable_metadata_and_restore_overrides_never_fast_trust(self):
        class RecordSubclass(neural._FrameTensorBytes):
            pass
        class StringSubclass(str):
            pass
        cases = ("subclass", "list_shape", "str_subclass", "instance_override", "class_override")
        for case in cases:
            runtime = PreparationRuntime()
            runtime.make_backend(True)
            original_freeze = neural._FrameTensorBytes.freeze
            original_restore = neural._FrameTensorBytes.restore
            original_digest = neural._FrameTensorBytes._digest
            overrides, records = [], []
            def freeze(cls, tensor):
                record = original_freeze(tensor)
                records.append(record)
                if len(records) == 1:
                    if case == "subclass":
                        record = RecordSubclass(**record.__dict__)
                    elif case == "list_shape":
                        record = replace(record, shape=list(record.shape))
                        record = replace(record, digest=original_digest(
                            record.storage, record.shape, record.strides, record.dtype))
                    elif case == "str_subclass":
                        record = replace(record, dtype=StringSubclass(record.dtype))
                    elif case == "instance_override":
                        def restore(torch):
                            overrides.append(1)
                            return original_restore(record, torch)
                        object.__setattr__(record, "restore", restore)
                return record
            def restore_override(record, torch):
                overrides.append(1)
                return original_restore(record, torch)
            with self.subTest(case=case), ExitStack() as stack:
                stack.enter_context(runtime.installed())
                stack.enter_context(patch.object(neural._FrameTensorBytes, "freeze", classmethod(freeze)))
                operations = stack.enter_context(PreparationOperations(neural, runtime))
                if case == "class_override":
                    stack.enter_context(patch.object(neural._FrameTensorBytes, "restore", restore_override))
                runtime.estimate()
                expected = 12 if case == "class_override" else 9 if case == "list_shape" else 8
                self.assertEqual(operations.counts["prepared_digest_calls"], expected)
                if "override" in case:
                    self.assertEqual(len(overrides), 10 if case == "class_override" else 5)
                self.assertEqual(runtime.calls.count("network"), 5)

    def test_direct_restore_always_validates_and_keeps_its_callback(self):
        runtime = PreparationRuntime()
        record = neural._FrameTensorBytes.freeze(runtime.tensor(np.ones((1, 3, 5, 6), np.float32)))
        original_digest = neural._FrameTensorBytes._digest
        with patch.object(neural._FrameTensorBytes, "_digest", wraps=original_digest) as digest:
            for _ in range(5):
                record.restore(SimpleNamespace(from_numpy=runtime.tensor))
        self.assertEqual(digest.call_count, 5)
        object.__setattr__(record, "digest", "bad")
        with self.assertRaisesRegex(ValueError, "integrity check failed"):
            record.restore(SimpleNamespace(from_numpy=lambda value: self.fail("corrupt tensor created")))

    def test_operation_counts_keep_live_validation_on_invalidation_and_failure(self):
        scenarios = {
            "invalidated": (lambda r, refiner: r.mask.__setitem__((0, 0), not r.mask[0, 0]), None),
            "failed_third_forward": (None, 2),
        }
        for name, (after, failure) in scenarios.items():
            observed = []
            with accepted_neural("ce5c31f22bb3796d83c0ae57d381148cf55a4d84") as control:
                for module in (control, neural):
                    runtime = PreparationRuntime(after_trace=after, fail_iteration=failure)
                    runtime.make_backend(True, implementation=module)
                    with runtime.installed(), PreparationOperations(module, runtime) as operations:
                        if failure is None:
                            runtime.estimate()
                        else:
                            with self.assertRaisesRegex(RuntimeError, "injected network error"):
                                runtime.estimate()
                    observed.append(operations.counts)
            old, new = observed
            with self.subTest(scenario=name):
                for key in ("array_digest_calls", "array_digest_bytes", "canonical_sha_calls", "canonical_json_calls",
                            "freeze_calls", "working_restores", "torch_bridges", "explicit_copy_bytes", "zero_fill_bytes"):
                    self.assertEqual(old[key], new[key])
                if failure is None:  # Every iteration changed the certificate; no seal can be reused.
                    self.assertEqual(old["prepared_digest_calls"], 20)
                    self.assertEqual(new["prepared_digest_calls"], 20)
                    self.assertEqual(new["array_digest_calls"], 15)
                else:
                    self.assertEqual((old["prepared_digest_calls"], new["prepared_digest_calls"]), (8, 4))
                    self.assertEqual(new["array_digest_calls"], 9)

    def test_validation_cap_changes_preserve_original_failure_boundary(self):
        original_cap = neural.FRAME_PREPARATION_MAX_TENSOR_BYTES
        for when in ("before_preparation", "before_restore", "after_first", "mask_callback"):
            def after(runtime, refiner):
                if when == "after_first" and len(runtime.traces) == 1:
                    neural.FRAME_PREPARATION_MAX_TENSOR_BYTES = 1
            runtime = PreparationRuntime(after_trace=after)
            runtime.make_backend(True)
            original_tensor = runtime.tensor
            original_freeze = neural._FrameTensorBytes.freeze
            freezes = []
            def tensor(value):
                result = original_tensor(value)
                if (when == "mask_callback" and runtime.backend.refiner._prepared_frame is not None
                        and np.asarray(value).shape == (1, 5, 6)):
                    neural.FRAME_PREPARATION_MAX_TENSOR_BYTES = 1
                return result
            def freeze(cls, value):
                result = original_freeze(value)
                freezes.append(result)
                if when == "before_restore" and len(freezes) == 2:
                    neural.FRAME_PREPARATION_MAX_TENSOR_BYTES = 1
                return result
            runtime.tensor = tensor
            with self.subTest(when=when), patch.object(neural, "FRAME_PREPARATION_MAX_TENSOR_BYTES", original_cap), \
                    runtime.installed(), patch.object(neural._FrameTensorBytes, "freeze", classmethod(freeze)):
                if when == "before_preparation":
                    neural.FRAME_PREPARATION_MAX_TENSOR_BYTES = 1
                with self.assertRaisesRegex(ValueError, "frame preparation exceeds the 128 MiB per-tensor limit"):
                    runtime.estimate()
                self.assertEqual(runtime.calls.count("network"), int(when in ("after_first", "mask_callback")))
                self.assertIsNone(runtime.backend.refiner._prepared_seals)

    def test_equal_cap_identity_or_type_changes_require_full_validation(self):
        original_cap = neural.FRAME_PREPARATION_MAX_TENSOR_BYTES
        for replacement, expected in ((int(str(original_cap)), 6), (float(original_cap), 12)):
            self.assertIsNot(replacement, original_cap)
            def after(runtime, refiner):
                if len(runtime.traces) == 1:
                    neural.FRAME_PREPARATION_MAX_TENSOR_BYTES = replacement
            runtime = PreparationRuntime(after_trace=after)
            runtime.make_backend(True)
            with self.subTest(cap_type=type(replacement).__name__), \
                    patch.object(neural, "FRAME_PREPARATION_MAX_TENSOR_BYTES", original_cap), \
                    runtime.installed(), PreparationOperations(neural, runtime) as operations:
                runtime.estimate()
            self.assertEqual(operations.counts["prepared_digest_calls"], expected)
            self.assertEqual(runtime.calls.count("network"), 5)

    def test_custom_validation_helpers_keep_full_call_semantics(self):
        for name in ("_length", "_digest"):
            original = getattr(neural._FrameTensorBytes, name)
            for mode in ("class", "instance_before", "instance_after"):
                calls, frozen = [], []
                def helper(*args):
                    calls.append(1)
                    return original(*args)
                def after(runtime, refiner):
                    if mode == "instance_after" and len(runtime.traces) == 1:
                        object.__setattr__(refiner._prepared_frame[1], name, helper)
                runtime = PreparationRuntime(after_trace=after)
                runtime.make_backend(True)
                original_freeze = neural._FrameTensorBytes.freeze
                def freeze(cls, value):
                    result = original_freeze(value)
                    frozen.append(result)
                    if mode == "instance_before" and len(frozen) == 1:
                        object.__setattr__(result, name, helper)
                    return result
                with self.subTest(helper=name, mode=mode), ExitStack() as stack:
                    stack.enter_context(runtime.installed())
                    stack.enter_context(patch.object(neural._FrameTensorBytes, "freeze", classmethod(freeze)))
                    if mode == "class":
                        stack.enter_context(patch.object(neural._FrameTensorBytes, name, staticmethod(helper)))
                    runtime.estimate()
                self.assertEqual(len(calls), 12 if mode == "class" else 5 if mode == "instance_before" else 4)
                self.assertEqual(runtime.calls.count("network"), 5)


if __name__ == "__main__":
    unittest.main()
