"""Exact, model-free acceptance tests for opt-in sparse mesh blending."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_geometry as wg
from bench import workbench_offline_export as ex
from tools import test_workbench_offline_export as analytical_fx
from tools import test_workbench_offline_export_integration as fx


def video_frames(output):
    capture = fx.cv2.VideoCapture(str(output / "playback.avi"))
    frames = []
    try:
        if not capture.isOpened():
            raise AssertionError("Cannot open manufactured FFV1 output")
        while True:
            ok, bgr = capture.read()
            if not ok:
                return frames
            frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    finally:
        capture.release()


class SparseBlendTests(unittest.TestCase):
    @staticmethod
    def args(rgb=None, mask=None, marker=None):
        source, artwork, observed, pose, K, vertices, faces, base_marker = analytical_fx.plane_fixture()
        support = np.array([3], np.int64)
        return [source if rgb is None else rgb, artwork,
                observed if mask is None else mask, pose, K, vertices, faces, support,
                base_marker if marker is None else marker]

    @staticmethod
    def render_overlay(shape, alpha_values=None):
        h, w = shape
        overlay = np.zeros((h, w, 4), np.uint8)
        yy, xx = np.indices((h, w))
        overlay[..., :3] = np.stack(((xx * 17 + 3) % 256, (yy * 29 + 7) % 256,
                                     (xx * 11 + yy * 13 + 1) % 256), axis=-1)
        overlay[..., 3] = 0
        if alpha_values is not None:
            overlay.reshape(-1, 4)[:len(alpha_values), 3] = alpha_values
        return overlay

    def compose(self, args, overlay, *, sparse, opacity=255):
        stats = {"reason": "manufactured", "rendered_pixels": int(np.count_nonzero(overlay[..., 3]))}
        with patch.object(ex.wg, "render_marker_rgba", return_value={"rgba": overlay, "stats": stats}):
            return ex.compose_mesh(*args, dict(ex.DEFAULT_PLACEMENT, opacity=opacity), sparse_blend=sparse)

    def test_sparse_matches_dense_across_alpha_opacity_masks_views_and_byte_edges(self):
        alpha_values = [0, 1, 127, 128, 254, 255]
        base = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        base[0, 0] = [0, 255, 1]
        base[0, 1] = [255, 0, 254]
        # Reversed and strided views remain read-only and must not be modified.
        rgb = base[:, ::-1, :]
        rgb.setflags(write=False)
        mask_base = np.ones((24, 64), dtype=bool)
        mask = mask_base[:, ::2]
        mask[::3, ::4] = False
        mask.setflags(write=False)
        args = self.args(rgb=rgb, mask=mask)
        original_rgb, original_mask = rgb.copy(), mask.copy()
        overlay = self.render_overlay(rgb.shape[:2], alpha_values * 32)
        smallest = float(np.nextafter(0.0, 1.0))
        opacities = (0, -0.0, smallest, float(np.nextafter(smallest, 1.0)),
                     float(np.nextafter(0.0, 1.0)), 0.25, 0.5, float(np.nextafter(1.0, 0.0)), 1)
        for opacity in opacities:
            with self.subTest(opacity=opacity):
                expected = self.compose(args, overlay, sparse=False, opacity=opacity)
                actual = self.compose(args, overlay, sparse=True, opacity=opacity)
                self.assertEqual(actual[1:], expected[1:])
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertFalse(np.shares_memory(actual[0], rgb))
                self.assertFalse(np.shares_memory(expected[0], rgb))
        half_args = self.args()
        half_args[0][0, 0] = [0, 100, 254]
        half_overlay = self.render_overlay((24, 32), [255])
        half_overlay[0, 0, :3] = [1, 101, 253]
        half = self.compose(half_args, half_overlay, sparse=True, opacity=0.5)
        np.testing.assert_array_equal(half[0][0, 0], [1, 101, 254])
        np.testing.assert_array_equal(rgb, original_rgb)
        np.testing.assert_array_equal(mask, original_mask)

    def test_default_equals_explicit_dense_and_empty_alpha_matches(self):
        args = self.args()
        overlay = self.render_overlay((24, 32), [0] * (24 * 32))
        stats = {"reason": "empty", "rendered_pixels": 0}
        with patch.object(ex.wg, "render_marker_rgba", return_value={"rgba": overlay, "stats": stats}):
            default = ex.compose_mesh(*args)
            dense = ex.compose_mesh(*args, sparse_blend=False)
            sparse = ex.compose_mesh(*args, sparse_blend=True)
        self.assertEqual(default[1:], dense[1:])
        self.assertEqual(dense[1:], sparse[1:])
        np.testing.assert_array_equal(default[0], dense[0])
        np.testing.assert_array_equal(dense[0], sparse[0])

    def test_rgb_ndarray_subclass_keeps_dense_expression(self):
        class RGBSubclass(np.ndarray):
            pass

        args = self.args()
        args[0] = args[0].view(RGBSubclass)
        overlay = self.render_overlay((24, 32), [0, 255, 1, 127])
        with patch.object(ex.wg, "render_marker_rgba", return_value={
                "rgba": overlay, "stats": {"rendered_pixels": 3}}), \
             patch.object(ex.np, "floor", wraps=np.floor) as floor:
            actual = ex.compose_mesh(*args, sparse_blend=True)
        self.assertEqual(floor.call_count, 1)
        self.assertEqual(floor.call_args.args[0].shape, (24, 32, 3))
        self.assertEqual(actual[1], "rendered_mesh_artwork")

    def test_strict_flag_rgb_and_metadata_errors_remain(self):
        args = self.args()
        overlay = self.render_overlay((24, 32), [255])
        for invalid_rgb in ("bad", np.zeros((24, 32, 3), np.float32), np.zeros((24, 32), np.uint8)):
            invalid_args = list(args)
            invalid_args[0] = invalid_rgb
            for sparse in (False, True):
                with self.subTest(rgb_type=type(invalid_rgb), sparse=sparse), \
                     self.assertRaises(ex.ar.Refused):
                    self.compose(invalid_args, overlay, sparse=sparse)
        for bad in (0, 1, None, "true"):
            with self.subTest(bad=bad), self.assertRaises(ex.ar.Refused):
                self.compose(args, overlay, sparse=bad)
        failures = []
        for sparse in (False, True):
            with self.subTest(metadata=sparse), self.assertRaises(Exception) as caught:
                ex.compose_mesh(*self.args(marker=dict(args[-1], face_id=99)), sparse_blend=sparse)
            failures.append((type(caught.exception), str(caught.exception)))
        self.assertEqual(failures[0], failures[1])

    def test_planar_api_and_cli_refuse_before_creating_output(self):
        (fx.ROOT / ".cache").mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="sparse-blend-planar-", dir=fx.ROOT / ".cache")
        self.addCleanup(temp.cleanup)
        fixture = fx.make_fixture(Path(temp.name) / "fixture")
        args = fx.export_args(fixture)
        api_output = fixture["workspace"].derived_dir / "planar-api-sparse"
        with self.assertRaises(ex.ar.Refused):
            ex.export_saved_window(fixture["workspace"], **{**args, "output_dir": api_output},
                                   surface_mode="planar", sparse_blend=True)
        self.assertFalse(api_output.exists())

        cli_output = fixture["workspace"].derived_dir / "planar-cli-sparse"
        cli_args = ["--workspace", str(fixture["workspace"].path),
                    "--project", str(args["project_path"]), "--bundle", str(args["bundle_path"]),
                    "--admission-dir", str(args["admission_dir"]), "--output", str(cli_output),
                    "--run-id", str(args["run_id"]), "--expected-revision", str(args["expected_revision"]),
                    "--bundle-sha256", str(args["bundle_sha256"]), "--role", str(args["role"]),
                    "--playback-fps", str(args["playback_fps"]), "--surface-mode", "planar", "--sparse-blend"]
        cli_script = Path(__file__).resolve().parents[0] / "workbench_offline_export.py"
        result = subprocess.run([sys.executable, str(cli_script), *cli_args], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("sparse_blend requires mesh", result.stderr)
        self.assertFalse(cli_output.exists())

    def test_mesh_ffv1_export_parity_accounting_manifest_and_provenance(self):
        (fx.ROOT / ".cache").mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="sparse-blend-export-", dir=fx.ROOT / ".cache")
        self.addCleanup(temp.cleanup)
        fixture = fx.make_fixture(Path(temp.name) / "fixture")
        default_args = fx.export_args(fixture)
        default_args["output_dir"] = fixture["workspace"].derived_dir / "dense"
        default = ex.export_saved_window(fixture["workspace"], **default_args, surface_mode="mesh")
        sparse_args = fx.export_args(fixture)
        sparse_args["output_dir"] = fixture["workspace"].derived_dir / "sparse"
        sparse = ex.export_saved_window(fixture["workspace"], **sparse_args,
                                        surface_mode="mesh", sparse_blend=True)
        self.assertEqual(sparse["rows"], default["rows"])
        self.assertEqual(sparse["frame_accounting"], default["frame_accounting"])
        self.assertTrue(sparse["mesh"]["sparse_blend"])
        self.assertNotIn("sparse_blend", default["mesh"])
        self.assertEqual(sparse["provenance"]["exporter_sources"], default["provenance"]["exporter_sources"])
        dense_frames, sparse_frames = video_frames(default_args["output_dir"]), video_frames(sparse_args["output_dir"])
        self.assertEqual(len(sparse_frames), len(dense_frames))
        for dense, actual in zip(dense_frames, sparse_frames):
            np.testing.assert_array_equal(actual, dense)

    def test_cli_lists_sparse_switch(self):
        script = Path(__file__).resolve().parents[0] / "workbench_offline_export.py"
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True,
                                text=True, check=True)
        self.assertIn("--sparse-blend", result.stdout)


if __name__ == "__main__":
    unittest.main()
