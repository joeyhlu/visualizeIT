"""Timing evidence and GPU ownership are checked without importing Torch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import workbench_telemetry as telemetry


class TelemetryTests(unittest.TestCase):
    def test_measured_stage_and_frame_durations_remain_separate(self):
        with patch.object(telemetry.time, "perf_counter", side_effect=[1., 2., 2.025, 3.]):
            trace = telemetry.Telemetry()
            with trace.measure("rendering") as interval:
                pass
            trace.frame("primary_neural", "f1", rendering_ms=interval["ms"], end_to_end_latency_ms=40.)
            snapshot = trace.snapshot()
        self.assertAlmostEqual(snapshot["stages"]["rendering"]["total_ms"], 25.)
        self.assertAlmostEqual(snapshot["run_wall_ms"], 2000.)
        self.assertEqual(snapshot["per_role_frame_timings"]["primary_neural"]["end_to_end_latency_ms"]["p95_ms"], 40.)

    def test_cpu_snapshot_does_not_invent_gpu_measurements(self):
        with patch.object(telemetry, "_cuda_torch", return_value=None):
            result = telemetry.gpu_peak_bytes()
        self.assertFalse(result["gpu_used"])
        self.assertIsNone(result["gpu_peak_allocated_bytes"])

    def test_cache_lookup_is_not_new_inference_latency(self):
        trace = telemetry.Telemetry()
        trace.cache_event("primary_neural", "cache_hit", 2.)
        trace.reuse_cached("primary_neural", {"tracking_ms": 99.})
        result = trace.snapshot()
        self.assertEqual(result["run_cache_mode"], "cached")
        self.assertEqual(result["per_role_frame_timings"], {})
        self.assertEqual(result["cached_original_timings"]["primary_neural"]["source"], "cached_original_run")

    def test_neural_stages_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            with telemetry.gpu_stage("segmentation", Path(d)):
                before = (Path(d) / "gpu-stage.lock").read_bytes()
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    with telemetry.gpu_stage("pose", Path(d)):
                        self.fail("overlap")
                self.assertEqual((Path(d) / "gpu-stage.lock").read_bytes(), before)
            self.assertFalse((Path(d) / "gpu-stage.lock").exists())

    def test_incomplete_foreign_lock_cannot_be_stolen(self):
        # Another process may have just created the file and not written its PID.
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "gpu-stage.lock"
            lock.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "busy"):
                with telemetry.gpu_stage("pose", Path(d)):
                    self.fail("stole an unowned lock")
            self.assertTrue(lock.exists())
            self.assertEqual(lock.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
