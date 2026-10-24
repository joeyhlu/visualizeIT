"""Codex tests for observed support, real COLMAP formats and bounded execution."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_reconstruction as reconstruction
from bench import workbench_telemetry as telemetry


class DependencyAndParserTests(unittest.TestCase):
    def test_matcher_policy_requires_explicit_thread_and_gpu_controls(self):
        for prefix in ("SiftMatching", "FeatureMatching"):
            for omitted in (None, "num_threads", "use_gpu"):
                with self.subTest(prefix=prefix, omitted=omitted):
                    def help_output(argv):
                        if argv[-1] == "help":
                            return 0, "COLMAP 3.11.1"
                        command = argv[1]
                        flags = list(reconstruction.REQUIRED_FLAGS[command])
                        if command == "feature_extractor":
                            flags.append("--SiftExtraction.use_gpu")
                        if command == "exhaustive_matcher":
                            flags.extend(f"--{prefix}.{name}" for name in ("use_gpu", "num_threads")
                                         if name != omitted)
                        return 0, " ".join(flags)
                    with patch.object(reconstruction, "find_colmap", return_value=Path("fixture.exe")):
                        result = reconstruction.preflight(runner=help_output)
                    self.assertEqual(result["available"], omitted is None)
                    if omitted:
                        self.assertIn(f"exhaustive_matcher:{omitted}", result["reason"])
                    else:
                        self.assertEqual(result["matcher_threads_flag"], f"--{prefix}.num_threads")
                        self.assertEqual(result["matcher_resource_policy"]["num_threads"], 1)
                        self.assertEqual(result["matcher_resource_policy"]["use_gpu"], 0)

    def test_colmap_cannot_spawn_during_another_gpu_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_root = root / "locks"
            with telemetry.gpu_stage("neural_fixture", lock_root):
                before = (lock_root / "gpu-stage.lock").read_bytes()
                with patch.object(reconstruction.subprocess, "Popen") as spawn:
                    result = reconstruction.run_stage("patch_match_stereo", ["colmap", "patch_match_stereo"],
                        workspace=root, log_dir=root / "logs", limits=reconstruction.Limits(free_reserve_bytes=0),
                        cancel=None, gpu_lock_dir=lock_root)
                spawn.assert_not_called()
                self.assertEqual(result["status"], "failed")
                self.assertIn("gpu_stage_busy", result["detail"])
                self.assertEqual((lock_root / "gpu-stage.lock").read_bytes(), before)

    def test_colmap_releases_lock_when_spawn_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(reconstruction.subprocess, "Popen", side_effect=OSError("injected spawn failure")):
                result = reconstruction.run_stage("features", ["colmap", "feature_extractor"],
                    workspace=root, log_dir=root / "logs", limits=reconstruction.Limits(free_reserve_bytes=0),
                    cancel=None, gpu_lock_dir=root / "locks")
            self.assertEqual(result["status"], "failed")
            self.assertIn("injected spawn", result["detail"])
            self.assertTrue((root / "logs" / result["receipt_file"]).is_file())
            self.assertFalse(telemetry.gpu_stage_active())
            self.assertFalse((root / "locks" / "gpu-stage.lock").exists())

    def test_long_raw_logs_remain_preserved_but_package_metadata_is_bounded(self):
        from bench import workbench_contract as contracts
        raw = (b"native output\n\twith-control\x01\r\n" * 2000)

        class FinishedProcess:
            def poll(self):
                return 1

            def wait(self):
                return 1

        def spawn(*_args, **kwargs):
            kwargs["stderr"].write(raw)
            return FinishedProcess()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(reconstruction.subprocess, "Popen", side_effect=spawn):
                result = reconstruction.run_stage("features", ["colmap", "feature_extractor"],
                    workspace=root, log_dir=root / "logs", limits=reconstruction.Limits(free_reserve_bytes=0),
                    cancel=None, gpu_lock_dir=root / "locks")
            self.assertEqual((root / "logs/features.stderr.log").read_bytes(), raw)
            self.assertEqual(result["returncode"], 1)
            self.assertLessEqual(len(result["stderr_tail"]), 4096)
            self.assertTrue(all(ord(ch) >= 32 for ch in result["stderr_tail"]))
            record = reconstruction.unavailable_package(package_id="fixture", clip_id="fixture",
                reason="native command failed", prefix_frames=[], masks={}, reconstruction={"stages": [result]})
            contracts.SurfacePackage.from_dict(record)
            saved = json.loads((root / "logs" / result["receipt_file"]).read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["returncode"], 1)
            self.assertTrue(all(ord(ch) >= 32 for ch in saved["stderr_tail"]))

    def test_missing_colmap_does_not_install_or_execute(self):
        with patch.object(reconstruction, "find_colmap", return_value=None), \
             patch.object(reconstruction, "_run_capture") as run:
            result = reconstruction.preflight()
            self.assertFalse(result["available"])
            self.assertIn("not_installed", result["reason"])
            run.assert_not_called()

    def test_colmap_pinhole_corner_origin_becomes_integer_center(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.txt"
            # COLMAP centres at .5, whereas the workbench centres at integer pixels.
            path.write_text("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n1 PINHOLE 32 24 16 20 16.5 12.5\n")
            camera = reconstruction.read_cameras_txt(path)[1]
            expected = np.array([[16., 0., 16.], [0., 20., 12.], [0., 0., 1.]])
            np.testing.assert_array_equal(camera["K"], expected)

    def test_dense_scalar_depth_uses_colmap_binary_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.bin"
            # COLMAP width&height&channels& header, little-endian float32 image payload.
            path.write_bytes(b"3&2&1&" + struct.pack("<6f", 1., 2., 3., 4., 5., 6.))
            np.testing.assert_array_equal(reconstruction.read_colmap_array(path),
                                          np.array([[1., 2., 3.], [4., 5., 6.]], np.float32))

    def test_actual_binary_ply_triangle_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\n"
                      "property float y\nproperty float z\nelement face 1\n"
                      "property list uchar int vertex_indices\nend_header\n").encode()
            vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]], dtype="<f4")
            path.write_bytes(header + vertices.tobytes() + struct.pack("<Biii", 3, 0, 1, 2))
            got_v, got_f = reconstruction.read_ply(path)
            np.testing.assert_array_equal(got_v, vertices)
            np.testing.assert_array_equal(got_f, [[0, 1, 2]])

    def test_no_prefix_yields_unavailable_without_fake_mesh(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(reconstruction, "preflight") as preflight:
            result = reconstruction.reconstruct([], {}, np.eye(3), output_dir=Path(directory) / "out")
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["package"].vertex_count, 0)
            self.assertEqual(result["package"].face_count, 0)
            preflight.assert_not_called()


class SupportAndResourceTests(unittest.TestCase):
    def test_vertex_occlusion_rejects_face_even_when_centroid_depth_matches(self):
        vertices = np.array([[-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]])
        K = np.array([[16., 0., 16.], [0., 16., 16.], [0., 0., 1.]])
        depth = np.full((32, 32), 2.)
        # Pinhole projection of the first vertex is exactly (8,8); foreground at z=1 occludes it.
        depth[8, 8] = 1.
        views = [{"K": K, "pose": np.eye(4), "mask": np.ones((32, 32), bool),
                  "depth": depth, "frame_id": f"v{i}"} for i in range(3)]
        support = reconstruction.face_support(vertices, np.array([[0, 1, 2]]), views, tolerance=.02)
        np.testing.assert_array_equal(support, [0])

    def test_fast_successful_process_cannot_bypass_final_storage_check(self):
        class FinishedProcess:
            pid = 123456789
            def poll(self):
                return 0
            def wait(self):
                return 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(reconstruction.subprocess, "Popen", return_value=FinishedProcess()), \
                 patch.object(reconstruction, "_tree_size", return_value=100):
                try:
                    result = reconstruction.run_stage("features", ["colmap", "feature_extractor"],
                        workspace=root, log_dir=root / "logs",
                        gpu_lock_dir=root / "locks",
                        limits=reconstruction.Limits(workspace_bytes=32, free_reserve_bytes=0),
                        cancel=threading.Event())
                except reconstruction.ReconstructionUnavailable:
                    return  # pre-spawn rejection is also correct.
            self.assertNotEqual(result["status"], "completed")

    def test_true_timestamp_prefix_over_ten_seconds_never_reaches_tool_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "source.bin"
            mask = root / "mask.bin"
            image.write_bytes(b"source bytes: no image decode should occur before prefix rejection")
            mask.write_bytes(b"mask bytes")
            frames, masks = [], {}
            for i, timestamp in enumerate((0., 20.)):
                fid = f"f{i}"
                frames.append({"frame_id": fid, "source_frame_index": i, "timestamp_s": timestamp,
                               "image_path": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                               "width": 32, "height": 32})
                masks[fid] = {"path": str(mask), "sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                              "width": 32, "height": 32}
            with patch.object(reconstruction, "preflight", return_value={"available": False,
                 "reason": "sentinel_preflight_called"}) as preflight:
                result = reconstruction.reconstruct(frames, masks,
                    np.array([[16., 0., 16.], [0., 16., 16.], [0., 0., 1.]]),
                    output_dir=root / "out", prefix_attestation={"prefix_bound_attested": True,
                                                               "prefix_duration_s": 20.})
                self.assertEqual(result["status"], "unavailable")
                preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
