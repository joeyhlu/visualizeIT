"""Model-free timing and real refiner control-flow regression tests."""
from contextlib import contextmanager
import itertools
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bench.workbench_refiner_timing import PHASES, RefinerStageProfiler


class ProfilerTests(unittest.TestCase):
    def test_deterministic_phases_synchronization_context_and_detachment(self):
        events = []
        ticks = iter(range(20))
        def clock():
            events.append("clock")
            return next(ticks)
        context = {"operator": ["recorded"]}
        profiler = RefinerStageProfiler(clock=clock, synchronize=lambda: events.append("sync"),
                                       device="declared-device", context=context, include_total=True)
        with profiler.trace({"rng_seed": 7}) as trace:
            for name in PHASES:
                with trace.stage(name):
                    events.append(name)
        result = profiler.snapshot()
        self.assertEqual(result["mode"], "host_synchronized_by_callback")
        self.assertTrue(result["perturbation"])
        self.assertEqual(events[1:6], ["sync", "clock", "preparation", "sync", "clock"])
        self.assertEqual([p["elapsed_seconds"] for p in result["traces"][0]["phases"]], [1] * 7)
        self.assertEqual(result["traces"][0]["elapsed_seconds"], 15)
        context["operator"].append("changed")
        result["traces"].clear()
        self.assertEqual(profiler.snapshot()["context"], {"operator": ["recorded"]})
        self.assertEqual(len(profiler.snapshot()["traces"]), 1)
        json.dumps(profiler.snapshot(), allow_nan=False)

    def test_capacity_never_calls_clock_or_sync_again(self):
        calls = []
        profiler = RefinerStageProfiler(max_traces=1, clock=lambda: calls.append(1) or len(calls),
                                       synchronize=lambda: calls.append(2), device="declared")
        with profiler.trace() as trace:
            with trace.stage(PHASES[0]):
                pass
        count = len(calls)
        with profiler.trace({"ignored": float("nan")}) as trace:
            with trace.stage(PHASES[0]):
                pass
        self.assertEqual(len(calls), count)
        self.assertEqual(profiler.snapshot()["dropped_traces"], 1)

    def test_invalid_inputs_and_clocks(self):
        for value in (True, 0, 10001, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RefinerStageProfiler(max_traces=value)
        with self.assertRaises(ValueError):
            RefinerStageProfiler(synchronize=lambda: None)
        with self.assertRaises(ValueError):
            RefinerStageProfiler(context={"bad": float("nan")})
        for clocks in ((float("nan"),), (float("inf"),), (True,), (2, 1)):
            profiler = RefinerStageProfiler(clock=iter(clocks).__next__)
            with self.subTest(clocks=clocks), self.assertRaises(ValueError):
                with profiler.trace() as trace:
                    with trace.stage(PHASES[0]):
                        pass
            row = profiler.snapshot()["traces"][0]
            self.assertEqual(row["outcome"], "failed")
            self.assertIsNone(row["phases"][0]["elapsed_seconds"])

    def test_failed_body_and_callback_propagate_and_cleanup(self):
        for source in ("body", "callback"):
            def sync():
                if source == "callback":
                    raise RuntimeError("callback")
            profiler = RefinerStageProfiler(clock=itertools.count().__next__, synchronize=sync, device="cpu")
            with self.assertRaisesRegex(RuntimeError, source):
                with profiler.trace() as trace:
                    with trace.stage(PHASES[0]):
                        raise RuntimeError("body")
            row = profiler.snapshot()["traces"][0]
            self.assertEqual(row["outcome"], "failed")
            self.assertEqual(row["phases"][0]["outcome"], "failed")
            self.assertFalse(trace.active)
            self.assertTrue(trace.closed)
            with self.assertRaises(ValueError):
                with trace.stage(PHASES[1]):
                    pass

    def test_metadata_retention_is_bounded_by_utf8_bytes(self):
        for value in ({"text": "a" * 16384}, {"text": "é" * 8192}):
            with self.assertRaises(ValueError):
                RefinerStageProfiler(context=value)
            profiler = RefinerStageProfiler()
            with self.assertRaises(ValueError):
                with profiler.trace(value):
                    self.fail("invalid metadata entered a trace")
            self.assertEqual(profiler.snapshot()["traces"], [])
            self.assertEqual(profiler.snapshot()["dropped_traces"], 0)
        for device in ("", " ", 1, "é" * 129):
            with self.assertRaises(ValueError):
                RefinerStageProfiler(device=device)

    def test_only_fixed_nonoverlapping_phases(self):
        profiler = RefinerStageProfiler(clock=itertools.count().__next__)
        with profiler.trace() as trace:
            with self.assertRaises(ValueError):
                with trace.stage("arbitrary"):
                    pass
            with trace.stage(PHASES[0]):
                with self.assertRaises(ValueError):
                    with trace.stage(PHASES[1]):
                        pass
            with self.assertRaises(ValueError):
                with trace.stage(PHASES[0]):
                    pass


class Tensor:
    def __init__(self, value, events):
        self.value = np.asarray(value)
        self.events = events

    def __getitem__(self, key):
        return Tensor(self.value[key], self.events)

    def __truediv__(self, value):
        return Tensor(self.value / value, self.events)

    def permute(self, *axes):
        return Tensor(self.value.transpose(axes), self.events)

    def float(self):
        return Tensor(self.value.astype(np.float32), self.events)

    def cpu(self):
        self.events.append("cpu")
        return self

    def numpy(self):
        return self.value

    def to(self, device):
        self.events.append(("to", device))
        return self


class RefinerStubTests(unittest.TestCase):
    def run_trace(self, count, profiler, *, network_error=False):
        import bench
        from bench import workbench_neural as neural
        events = []
        tensor = lambda value: Tensor(value, events)
        @contextmanager
        def inference_mode():
            events.append("inference_enter")
            yield
            events.append("inference_exit")
        def inputs(**kwargs):
            events.append("crop")
            np.testing.assert_array_equal(kwargs["init_poses_cam_from_model"].value[0, :3, 3], [0, 0, 1000])
            template = SimpleNamespace(rgbs=tensor(np.zeros((1, 3, 5, 5))),
                                       masks=tensor(np.ones((1, 5, 5))), depths=tensor(np.ones((1, 5, 5))))
            return {"templates": template, "crop_rgbs": tensor(np.ones((1, 3, 5, 5))),
                    "crop_masks": tensor(np.ones((1, 5, 5)))}, [None], [np.eye(4)]
        def network(*args):
            events.append("network")
            if network_error:
                raise RuntimeError("stub network failed")
            return tensor(np.zeros((1, 2, 5, 5))), tensor(np.ones((1, 5, 5)))
        def correspondences(*args):
            events.append("correspondences")
            valid = np.arange(25).reshape(5, 5) < count
            pose = np.eye(4)
            pose[2, 3] = 1000
            return pose, np.ones((5, 5, 3)), np.ones((5, 5, 2)), np.ones((5, 5, 2)), None, valid, None, None
        def rodrigues(value):
            events.append("rodrigues")
            return (np.zeros((3, 1)) if value.shape == (3, 3) else np.eye(3)), None
        def ransac(*args, **kwargs):
            events.append(("ransac", kwargs))
            return True, np.zeros((3, 1)), np.array([[0.], [0.], [1000.]]), np.arange(6).reshape(-1, 1)
        def refine(*args):
            events.append("refine")
            return args[-2:]
        cv2 = SimpleNamespace(setRNGSeed=lambda seed: events.append(("seed", seed)), Rodrigues=rodrigues,
                              solvePnPRansac=ransac, solvePnPRefineLM=refine, SOLVEPNP_ITERATIVE=0)
        qg = SimpleNamespace(upstream_path=lambda: events.append("upstream"),
                             _integer_profile_correspondences=correspondences)
        utils = SimpleNamespace(data_util=SimpleNamespace(compute_gotrack_inputs_from_init_poses=inputs),
                                misc=SimpleNamespace(get_intrinsic_matrix=lambda camera: np.eye(3)),
                                structs=SimpleNamespace(PinholePlaneCameraModel=lambda **kwargs: kwargs))
        modules = {"torch": SimpleNamespace(from_numpy=tensor, inference_mode=inference_mode),
                   "utils": utils, "bench.vision": SimpleNamespace(cv2=cv2), "bench.quality_gotrack": qg}
        case = {"rgb": np.zeros((5, 5, 3), np.uint8), "mask": np.ones((5, 5), bool),
                "native_K": np.eye(3), "initial_pose_m": np.eye(4), "glb_sha256": "fake", "obj_id": 1,
                "rng_seed": 123}
        case["initial_pose_m"][2, 3] = 1
        def renderer(case):
            events.append("renderer")
            return SimpleNamespace(vertices_m=np.zeros((3, 3)))
        with patch.dict(sys.modules, modules), patch.object(bench, "quality_gotrack", qg, create=True):
            runner = neural.WorkbenchIntegerRefiner(network, renderer_factory=renderer, profiler=profiler)
            output = runner.trace(case)
        return output, events

    def test_real_trace_success_and_lost_paths_have_array_call_and_rng_parity(self):
        from bench.workbench_neural import compare_traces
        for count in (25, 23):
            with self.subTest(count=count):
                baseline, original_events = self.run_trace(count, None)
                profiler = RefinerStageProfiler(clock=itertools.count().__next__)
                profiled, profiled_events = self.run_trace(count, profiler)
                self.assertTrue(compare_traces(baseline, profiled)["passed"])
                self.assertEqual(original_events, profiled_events)
                self.assertEqual(original_events.count("network"), 1)
                self.assertEqual(original_events.count(("seed", 123)), int(count >= 24))
                self.assertEqual(int(profiled["solver_state"]), int(count >= 24))
                self.assertEqual(original_events[original_events.index("network") - 3:original_events.index("network")],
                                 [("to", "cpu")] * 3)
                row = profiler.snapshot()["traces"][0]
                self.assertEqual(row["metadata"], {"rng_seed": 123})
                self.assertEqual([phase["name"] for phase in row["phases"]], list(PHASES))
                self.assertTrue(all(phase["outcome"] == "completed" for phase in row["phases"]))

    def test_disabled_path_never_reads_profiling_clock_or_synchronizes(self):
        with patch.object(RefinerStageProfiler, "_now", side_effect=AssertionError("clock")), \
                patch.object(RefinerStageProfiler, "_sync", side_effect=AssertionError("sync")):
            self.run_trace(25, None)

    def test_runtime_failure_records_partial_trace_and_propagates(self):
        profiler = RefinerStageProfiler(clock=itertools.count().__next__)
        with self.assertRaisesRegex(RuntimeError, "stub network failed"):
            self.run_trace(25, profiler, network_error=True)
        row = profiler.snapshot()["traces"][0]
        self.assertEqual(row["outcome"], "failed")
        self.assertEqual([phase["name"] for phase in row["phases"]], list(PHASES[:3]))
        self.assertEqual([phase["outcome"] for phase in row["phases"]],
                         ["completed", "completed", "failed"])
        self.assertEqual(row["phases"][-1]["elapsed_seconds"], 1)

    def test_helper_has_no_runtime_imports_or_file_writes(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path(__file__).with_name("workbench_refiner_timing.py").read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertEqual(set(imports), {"contextlib", "json", "math", "time"})


if __name__ == "__main__":
    unittest.main()
