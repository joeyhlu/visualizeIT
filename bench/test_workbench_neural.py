"""Codex-owned neural admission tests: manufactured arrays, never model inference."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_geometry as wg
from bench import workbench_neural as neural
from bench.test_workbench_contract import clip_document


class NeuralIdentityAndNumericsTests(unittest.TestCase):
    def test_primary_is_exact_new_export_and_baseline_separate(self):
        self.assertEqual(neural.primary_identity()["export_sha256"],
                         "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311")
        self.assertEqual(neural.primary_identity()["export_bytes"], 80_165_526)
        self.assertNotIn("export_sha256", neural.baseline_identity())
        self.assertEqual(neural.baseline_identity()["role"], "frozen_baseline")

    def test_authenticated_bytes_reject_wrong_size_or_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            path.write_bytes(b"not model weights")
            for size, digest in ((path.stat().st_size + 1, "a" * 64), (path.stat().st_size, "b" * 64)):
                with self.subTest(size=size), self.assertRaises((ValueError, RuntimeError)):
                    neural._authenticated_bytes(path, size, digest, "test head")

    def test_native_intrinsics_half_pixel_applied_exactly_once_by_origin(self):
        # Integer pixel centres are (i,j); COLMAP corner-origin pixel centres are (i+.5,j+.5).
        K = np.array([[100., 0., 20.], [0., 120., 25.], [0., 0., 1.]])
        np.testing.assert_array_equal(neural.native_K(K, "integer_centers"), K)
        expected = K.copy()
        expected[:2, 2] -= .5
        np.testing.assert_array_equal(neural.native_K(K, "corner"), expected)
        with self.assertRaises(ValueError):
            neural.native_K(K, "unknown")

    def test_finite_flow_and_confidence_required_even_when_nan_patterns_match(self):
        for field in ("flow", "confidence"):
            trace = {field: np.array([np.nan], np.float32)}
            with self.subTest(field=field):
                self.assertFalse(neural.compare_traces(trace, trace)["passed"])

    def test_discrete_threshold_decisions_must_match_exactly(self):
        a = {"confidence_mask": np.array([True, False])}
        b = {"confidence_mask": np.array([True, True])}
        self.assertFalse(neural.compare_traces(a, b)["passed"])

    def test_float_tolerance_does_not_allow_discrete_correspondence_change(self):
        a = {"retained_indices": np.array([1, 2, 3], np.int64)}
        b = {"retained_indices": np.array([1, 2, 4], np.int64)}
        self.assertFalse(neural.compare_traces(a, b)["passed"])

    def test_head_token_flag_is_not_a_serving_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = neural.WorkbenchContract({"cases": []}, "a" * 64)
            admission = neural.verify_serving_admission(Path(directory), contract=contract,
                geometry={"accepted": True, "cached_token_parity_passed": True})
            self.assertFalse(admission.admitted)


class FakeBackend:
    """An intentionally overconfident backend exercises the wrapper's guards."""
    def __init__(self, installed=None):
        self.installed = installed or neural.PRIMARY["export_sha256"]
        self.calls = 0

    def install_provenance(self):
        return {"head": {"installed_sha256": self.installed}}

    def estimate(self, rgb, mask, K, state):
        self.calls += 1
        return {"status": "tracking", "pose_m": np.eye(4), "metrics": {}}

    def reset(self):
        pass

    def close(self):
        pass


class NeuralSessionGuardTests(unittest.TestCase):
    def setUp(self):
        self.clip = wc.ClipSpec.from_dict(clip_document())
        self.admission = neural.ServingAdmission(True, (), {"families": ["fixture"], "contract_sha256": "c" * 64})
        self.rgb = np.zeros((480, 640, 3), np.uint8)
        self.K = np.array([[576., 0., 319.5], [0., 576., 239.5], [0., 0., 1.]])

    def session(self, backend):
        return neural.NeuralSession("primary_neural", backend, self.admission,
                                    run_id="test-run", clip=self.clip, family="fixture")

    def identity(self, frame, mask=None):
        result = {"frame_id": frame.frame_id, "image_sha256": wg.array_sha256(self.rgb)}
        if mask is not None:
            result.update(mask_frame_id=frame.frame_id, mask_sha256=wg.mask_sha256(mask))
        return result

    def test_wrong_installed_weights_never_fall_back(self):
        with self.assertRaises(RuntimeError):
            self.session(FakeBackend("0" * 64))

    def test_no_mask_never_calls_backend_or_publishes_pose(self):
        backend = FakeBackend()
        session = self.session(backend)
        frame = self.clip.frames[2]
        row = session.process(frame, self.rgb, None, self.identity(frame), self.K)
        self.assertIsNone(row.pose)
        self.assertNotEqual(row.pose_state, "tracking")
        self.assertEqual(backend.calls, 0)
        session.close()

    def test_empty_mask_never_calls_backend_or_publishes_pose(self):
        backend = FakeBackend()
        session = self.session(backend)
        frame = self.clip.frames[2]
        mask = np.zeros((480, 640), bool)
        row = session.process(frame, self.rgb, mask, self.identity(frame, mask), self.K)
        self.assertIsNone(row.pose)
        self.assertNotEqual(row.pose_state, "tracking")
        self.assertEqual(backend.calls, 0)
        session.close()

    def test_mutated_installed_identity_rejected_mid_run(self):
        backend = FakeBackend()
        session = self.session(backend)
        backend.installed = "0" * 64
        frame = self.clip.frames[2]
        with self.assertRaises(ValueError):
            session.process(frame, self.rgb, None, self.identity(frame), self.K)
        session.close()

    def test_artwork_is_current_frame_and_disappears_on_loss(self):
        backend = FakeBackend()
        calls = []
        rgba = np.zeros((480, 640, 4), np.uint8)
        rgba[20, 20] = [255, 255, 255, 255]
        def renderer(pose, observed, K):
            calls.append(observed.copy())
            return {"rgba": rgba, "stats": {"rendered_pixels": 1}}
        session = neural.NeuralSession("primary_neural", backend, self.admission,
            run_id="test-run", clip=self.clip, family="fixture", renderer=renderer)
        frame = self.clip.frames[2]
        mask = np.ones((480, 640), bool)
        row = session.process(frame, self.rgb, mask, self.identity(frame, mask), self.K)
        self.assertEqual(row.render_state, "rendered")
        self.assertIs(session.last_overlay, rgba)
        self.assertEqual(len(calls), 1)
        frame = self.clip.frames[3]
        row = session.process(frame, self.rgb, None, self.identity(frame), self.K)
        self.assertIsNone(row.pose)
        self.assertIsNone(session.last_overlay)
        self.assertNotEqual(row.render_state, "rendered")
        self.assertEqual(len(calls), 1)
        session.close()


if __name__ == "__main__":
    unittest.main()
