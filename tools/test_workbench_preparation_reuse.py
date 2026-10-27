"""Model-free tests of estimate-local preparation reuse in the actual backend."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
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

    def make_backend(self, enabled=False, **kwargs):
        runtime = self
        class CapturingRefiner(neural.WorkbenchIntegerRefiner):
            def trace(self, case):
                result = super().trace(case)
                runtime.traces.append(result)
                if runtime.after_trace is not None:
                    runtime.after_trace(runtime, self)
                return result
        def renderer(case):
            self.calls.append("renderer")
            return SimpleNamespace(vertices_m=np.zeros((3, 3)))
        self.backend = neural.GoTrackMetricBackend(
            "primary_neural", self.bundle, head_path=Path("unused.pt"),
            network_loader=lambda device: (self.network, self.provenance),
            refiner_factory=lambda net, dev: CapturingRefiner(net, dev, renderer_factory=renderer),
            preparation_reuse=enabled, **kwargs)
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


if __name__ == "__main__":
    unittest.main()
