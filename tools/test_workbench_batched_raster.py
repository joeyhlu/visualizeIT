"""Model-free parity and opt-in export checks for the experimental raster."""
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from bench import workbench_geometry as wg
from bench import workbench_offline_export as ex
from bench import workbench_raster_batch as batch
from tools import test_workbench_mesh_export as mesh_fx
from tools import test_workbench_offline_export as analytical_fx
from tools import test_workbench_offline_export_integration as fx


def assert_raster_equal(test, expected, actual):
    np.testing.assert_array_equal(actual.depth, expected.depth)
    np.testing.assert_array_equal(actual.face_id, expected.face_id)
    np.testing.assert_array_equal(actual.barycentric, expected.barycentric)
    test.assertEqual(actual.stats, expected.stats)


class BatchedRasterTests(unittest.TestCase):
    def test_invalid_pose_precedes_vertex_conversion_in_batched_mode(self):
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        K = np.eye(3)
        pose = np.zeros((4, 4))
        with self.assertRaises(wg.GeometryError) as reference_error:
            wg.rasterize("bad", faces, K, pose, 8, 8)
        with self.assertRaises(wg.GeometryError) as batched_error:
            wg.rasterize("bad", faces, K, pose, 8, 8, batched=True)
        self.assertEqual(str(batched_error.exception), str(reference_error.exception))

    def test_extreme_unreferenced_vertex_preserves_reference_warning_count(self):
        vertices = np.array([[-.2, -.2, 1.], [.2, -.2, 1.], [0., .2, 1.],
                             [1e100, 0., 1e-300]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        K = np.array([[1e100, 0., 16.], [0., 10., 12.], [0., 0., 1.]])
        pose = np.eye(4)
        with warnings.catch_warnings(record=True) as reference_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            expected = wg.rasterize(vertices, faces, K, pose, 32, 24)
        with warnings.catch_warnings(record=True) as batched_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            actual = wg.rasterize(vertices, faces, K, pose, 32, 24, batched=True)
        assert_raster_equal(self, expected, actual)
        self.assertEqual(len(batched_warnings), len(reference_warnings))

    def test_exact_raster_parity_fallback_edges_roi_and_fragment_chunks(self):
        _, rgba, _, pose, K, vertices, faces, marker = analytical_fx.plane_fixture()
        # Many small, front-facing exact ties force multiple bounded fragment chunks.
        tied_vertices = vertices.copy()
        tied_vertices[:, :2] *= .2
        tied_faces = np.repeat(faces, 24, axis=0)
        cases = [(vertices, faces, K, pose, 32, 24, None),
                 (tied_vertices, tied_faces, K, pose, 32, 24, None),
                 (vertices, faces, K, pose, 32, 24, (9, 6, 23, 18))]
        near_vertices = vertices.copy()
        near_vertices[0, 2] = -.9995  # crosses the z=0.001 near plane after pose
        behind = vertices.copy()
        behind[:, 2] = -3
        nonfinite = vertices.copy()
        nonfinite[0, 0] = np.nan
        cases.extend([(near_vertices, faces, K, pose, 32, 24, None),
                      (behind, faces, K, pose, 32, 24, None),
                      (nonfinite, faces, K, pose, 32, 24, None)])
        degenerate = vertices.copy()
        degenerate[2] = (degenerate[0] + degenerate[1]) / 2
        cases.append((degenerate, faces, K, pose, 32, 24, None))
        guard = vertices.copy()
        guard[:, 0] += 1e9
        cases.append((guard, faces, K, pose, 32, 24, None))
        with patch.object(batch, "FRAGMENT_CAP", 64):
            for args in cases:
                with self.subTest(roi=args[-1], faces=len(args[1])):
                    expected = wg.rasterize(*args[:6], roi=args[6])
                    if args[1] is tied_faces:
                        with patch.object(batch.np, "repeat", wraps=np.repeat) as repeated:
                            actual = wg.rasterize(*args[:6], roi=args[6], batched=True)
                        self.assertGreater(repeated.call_count, 2, "fixture did not cross fragment chunks")
                        self.assertEqual(int(actual.face_id[12, 16]), 0)
                    else:
                        actual = wg.rasterize(*args[:6], roi=args[6], batched=True)
                    assert_raster_equal(self, expected, actual)

    def test_rgba_parity_default_strict_bool_and_supplied_raster_precedence(self):
        rgb, art, mask, pose, K, vertices, faces, marker = analytical_fx.plane_fixture()
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[:, :] = [40, 180, 90, 200]
        support = np.array([3], dtype=np.int64)
        args = (vertices, faces, support, marker, rgba, K, pose, 32, 24, mask)
        default = wg.render_marker_rgba(*args)
        reference = wg.render_marker_rgba(*args, batched_raster=False)
        fast = wg.render_marker_rgba(*args, batched_raster=True)
        np.testing.assert_array_equal(default["rgba"], reference["rgba"])
        self.assertEqual(default["stats"], reference["stats"])
        np.testing.assert_array_equal(fast["rgba"], reference["rgba"])
        self.assertEqual(fast["stats"], reference["stats"])
        for bad in (0, 1, None, "yes"):
            with self.subTest(bad=bad), self.assertRaises(wg.GeometryError):
                wg.rasterize(vertices, faces, K, pose, 32, 24, batched=bad)
            with self.subTest(bad=bad), self.assertRaises(wg.GeometryError):
                wg.render_marker_rgba(*args, batched_raster=bad)
        supplied = wg.rasterize(vertices, faces, K, pose, 32, 24)
        with patch.object(wg, "rasterize", side_effect=AssertionError("supplied raster replaced")):
            supplied_result = wg.render_marker_rgba(*args, raster=supplied, batched_raster=True)
        np.testing.assert_array_equal(supplied_result["rgba"], reference["rgba"])

    def test_mesh_compose_and_ffv1_export_flag_and_helper_identity(self):
        rgb, rgba, mask, pose, K, vertices, faces, marker = analytical_fx.plane_fixture()
        mesh_args = (rgb, rgba, mask, pose, K, vertices, faces, np.array([3], dtype=np.int64), marker)
        expected = ex.compose_mesh(*mesh_args)
        actual = ex.compose_mesh(*mesh_args, batched_raster=True)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1:], expected[1:])

        (fx.ROOT / ".cache").mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(prefix="batched-raster-", dir=fx.ROOT / ".cache")
        self.addCleanup(tmp.cleanup)
        fixture = fx.make_fixture(Path(tmp.name) / "fixture")
        default_args = fx.export_args(fixture)
        default_args["output_dir"] = fixture["workspace"].derived_dir / "default"
        default = ex.export_saved_window(fixture["workspace"], **default_args, surface_mode="mesh")
        batched_args = fx.export_args(fixture)
        batched_args["output_dir"] = fixture["workspace"].derived_dir / "batched"
        with patch.object(ex.wg, "rasterize", wraps=ex.wg.rasterize) as raster_call:
            batched = ex.export_saved_window(fixture["workspace"], **batched_args,
                                             surface_mode="mesh", batched_raster=True)
        self.assertGreater(raster_call.call_count, 0)
        self.assertEqual(batched["rows"], default["rows"])
        self.assertEqual(batched["frame_accounting"], default["frame_accounting"])
        self.assertTrue(batched["mesh"]["batched_raster"])
        self.assertNotIn("batched_raster", default["mesh"])
        source_hashes = batched["provenance"]["exporter_sources"]
        helper_hash = __import__("hashlib").sha256(Path(batch.__file__).read_bytes()).hexdigest()
        self.assertEqual(source_hashes["bench/workbench_raster_batch.py"], helper_hash)
        self.assertNotIn("bench/workbench_raster_batch.py", default["provenance"]["exporter_sources"])

    def test_cli_exposes_experimental_switch(self):
        script = Path(__file__).resolve().parents[0] / "workbench_offline_export.py"
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True,
                                text=True, check=True)
        self.assertIn("--batched-raster", result.stdout)


if __name__ == "__main__":
    unittest.main()
