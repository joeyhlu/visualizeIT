"""Independent serving guards: no GPU or model allocations are needed."""
import copy
import gc
from pathlib import Path
import tempfile
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from bench import workbench_neural as neural
from bench import workbench_telemetry as telemetry
from bench.vision import cv2


class ServingLifecycleTests(unittest.TestCase):
    def diagnostic_backend(self, proposed):
        backend = neural.GoTrackMetricBackend.__new__(neural.GoTrackMetricBackend)
        seed = np.eye(4)
        seed[2, 3] = 1.
        backend.bundle = SimpleNamespace(principal_point_origin="integer_centers", glb_path=Path("fixture.glb"),
            glb_sha256="a" * 64, obj_id=9, seed_pose_m=seed, seed_label="controlled_init:fixture",
            seed_frame_id="s000000")
        backend.seed_target_frame_id, backend._seed_used = "s000001", False
        backend.iterations, backend.min_inliers, backend.consistency = 1, 6, (30., .2)
        backend.refiner = SimpleNamespace(trace=lambda _: {"solver_state": np.array(1),
            "ransac_inliers": np.arange(24), "solved_native_pose_m": proposed.copy()})
        return backend, seed

    def test_recovery_reason_distinguishes_initialization_from_motion_ambiguity(self):
        proposed = np.eye(4)
        proposed[2, 3] = 1.
        backend, seed = self.diagnostic_backend(proposed)
        first = backend.estimate(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), np.eye(3),
                                 {"_frame_id": "s000001"})
        self.assertEqual(first["status"], "recovering")
        self.assertIsNone(first["pose_m"])
        self.assertIn("initial temporal check pending", first["reason"])
        self.assertFalse(first["evidence"]["temporal_consistency"]["checked"])
        theta = np.deg2rad(31.)
        proposed[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        backend, _ = self.diagnostic_backend(proposed)
        state = {"prior_pose": seed, "consecutive": 1}
        ambiguous = backend.estimate(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), np.eye(3), state)
        self.assertEqual(ambiguous["status"], "recovering")
        self.assertIsNone(ambiguous["pose_m"])
        self.assertIn("failed temporal consistency", ambiguous["reason"])
        self.assertAlmostEqual(ambiguous["metrics"]["temporal_rotation_delta_deg"], 31.)
        checked = backend.estimate(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), np.eye(3), state)
        self.assertEqual(checked["status"], "tracking")
        np.testing.assert_array_equal(checked["pose_m"], proposed)

    def test_existing_relative_translation_threshold_stays_inclusive(self):
        proposed = np.eye(4)
        proposed[:3, 3] = [.2, 0, 1]
        backend, seed = self.diagnostic_backend(proposed)
        result = backend.estimate(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), np.eye(3),
                                  {"prior_pose": seed, "consecutive": 1})
        self.assertEqual(result["status"], "tracking")
        self.assertAlmostEqual(result["metrics"]["temporal_relative_translation"], .2)

    def test_previous_variant_is_destroyed_before_loading_next_network(self):
        refs, events = [], []

        class Network:
            pass

        class Counter:
            MODULES = ("network",)

            def __init__(self, network):
                self.counts = {"network": 0}
                network.counter = self

            def take(self):
                return {}

            def close(self):
                events.append("counter_closed")

        class Runner:
            def __init__(self, network, _device):
                self.network = network
                refs.append(weakref.ref(self))

            def trace(self, _case):
                self.network.counter.counts["network"] += 1
                return {"flow": np.zeros((2, 2), np.float32)}

            def render_proof(self, _case):
                return {"fixture": True}

            def close(self):
                # Deliberately keep self.network, as the real refiner does.
                events.append("runner_closed")

        def load(_device):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs),
                            "previous network/refiner survived into the next load")
            network = Network()
            refs.append(weakref.ref(network))
            return network

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path, mask_path, glb_path = [root / x for x in ("rgb.png", "mask.png", "object.glb")]
            cv2.imwrite(str(rgb_path), np.zeros((4, 5, 3), np.uint8))
            cv2.imwrite(str(mask_path), np.full((4, 5), 255, np.uint8))
            glb_path.write_bytes(b"fixture")
            row = {"case_id": "fixture", "family": "fixture", "rgb_path": str(rgb_path),
                   "mask_path": str(mask_path), "rgb_sha256": neural.sha256_file(rgb_path),
                   "mask_sha256": neural.sha256_file(mask_path),
                   "glb": {"path": str(glb_path.resolve()), "sha256": neural.sha256_file(glb_path)}}
            contract = neural.WorkbenchContract({"cases": [row], "families": ["fixture"],
                "runtime_packages": neural.runtime_packages(),
                "source_closure": {"closure_sha256": "a" * 64}}, "b" * 64)
            installer = lambda network, *_args, **_kwargs: (network, {"fixture": True})
            with patch.object(neural, "glb_inventory", return_value=row["glb"]), \
                 patch.object(neural, "deterministic_torch", return_value={}), \
                 patch.object(neural, "load_reference_head", new=installer), \
                 patch.object(neural, "install_primary_head", new=installer), \
                 patch.object(neural, "ForwardCounter", Counter), \
                 patch.object(neural, "_render_factory", return_value={}):
                result = neural.run_fullimage_head_parity([{**row, "glb_path": str(glb_path)}],
                    contract=contract, head_path=root / "head.pt", reference_path=root / "best.pt",
                    output_dir=root / "parity", base_loader=load, runner_factory=Runner,
                    torch_module=object())
            self.assertTrue(result["all_passed"])
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(events.count("runner_closed"), 2)
        self.assertEqual(events.count("counter_closed"), 2)

    def test_render_proofs_cover_exact_cases_and_variants(self):
        from bench import workbench_render as render
        proof = {"policy": render.POLICY, "framebuffer_proof": {"fixture": True}}
        valid = {"variants": {v: {"render_proofs": {c: copy.deepcopy(proof) for c in ("a", "b")}}
                              for v in ("reference", "export")}}
        self.assertEqual(neural.parity_proof_reasons(valid, {"a", "b"}), [])
        for change in ("missing_case", "extra_case", "missing_variant", "extra_variant", "wrong_policy"):
            with self.subTest(change=change):
                bad = copy.deepcopy(valid)
                if change == "missing_case":
                    del bad["variants"]["export"]["render_proofs"]["b"]
                elif change == "extra_case":
                    bad["variants"]["reference"]["render_proofs"]["extra"] = proof
                elif change == "missing_variant":
                    del bad["variants"]["export"]
                elif change == "extra_variant":
                    bad["variants"]["extra"] = {}
                else:
                    bad["variants"]["export"]["render_proofs"]["b"]["policy"] = "legacy"
                self.assertTrue(neural.parity_proof_reasons(bad, {"a", "b"}))

    def test_gpu_ownership_is_specific_to_holder_thread(self):
        observations = []
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(telemetry.gpu_stage_active())
            with telemetry.gpu_stage("fixture", Path(directory)):
                self.assertTrue(telemetry.gpu_stage_active())
                thread = threading.Thread(target=lambda: observations.append(telemetry.gpu_stage_active()))
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            self.assertFalse(telemetry.gpu_stage_active())
        self.assertEqual(observations, [False])

    def test_cuda_synchronization_failure_cannot_be_silenced(self):
        from types import SimpleNamespace
        calls = []

        def failed_sync():
            raise RuntimeError("injected CUDA asynchronous failure")

        cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                               synchronize=failed_sync, empty_cache=lambda: calls.append("cache"))
        with self.assertRaisesRegex(RuntimeError, "injected CUDA"):
            neural._release_device_memory(SimpleNamespace(cuda=cuda))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
