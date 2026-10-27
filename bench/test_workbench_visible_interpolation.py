"""Exact Git-parent raster controls and logical interpolation work observations."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_geometry as geometry


@contextmanager
def parent_geometry():
    revision = "c02fcb89074a283f63f4967fd39d9580e55ff230"
    source = subprocess.check_output(["git", "show", revision + ":bench/workbench_geometry.py"],
                                     cwd=Path(__file__).resolve().parents[1])
    if hashlib.sha256(source).hexdigest() != "d21f665ac2237729d342e96a32e9745136f58bd741e2f369049889d894f6763c":
        raise AssertionError("parent geometry source differs from its pinned Git blob")
    name = "bench._visible_interpolation_parent"
    module = ModuleType(name)
    module.__file__ = str(Path(geometry.__file__).resolve())
    module.__package__ = "bench"
    with patch.dict(sys.modules, {name: module}):
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        yield module


@contextmanager
def interpolation_trace(module):
    """Observe executed orig assignments and their actual array shapes.

    The source expression multiplies three weights by the 3x3 original basis,
    reduces three contributions, and divides three components per sample. This
    records actual evaluated result shapes rather than estimating mesh work.
    It does not measure elapsed time or actual memory traffic.
    """
    observations = []
    seen = set()
    code = module._raster_triangle.__code__
    def observe(frame, event, arg):
        if frame.f_code is code and event == "return":
            seen.discard(id(frame))
        if frame.f_code is code and event == "line" and "orig" in frame.f_locals:
            # External tracing state counts each invocation once without
            # changing the product function's locals or numerical operations.
            values = frame.f_locals
            if id(frame) not in seen:
                seen.add(id(frame))
                result = values["orig"]
                observations.append({"shape": tuple(result.shape),
                                     "samples": int(result.size // 3),
                                     "bbox_samples": int(values["inside"].size)})
        return observe
    previous = sys.gettrace()
    sys.settrace(observe)
    try:
        yield observations
    finally:
        sys.settrace(previous)


class VisibleInterpolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_scope = parent_geometry()
        cls.parent = cls.parent_scope.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.parent_scope.__exit__(None, None, None)

    def assert_exact_array(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        # This also checks signed zeros and the bits of any NaNs.
        self.assertEqual(left.tobytes(), right.tobytes())

    def assert_raster_equal(self, vertices, faces, K=None, pose=None, width=32, height=24, **kwargs):
        if K is None:
            K = np.array([[12., 0., 15.5], [0., 11., 11.5], [0., 0., 1.]])
        if pose is None:
            pose = np.eye(4)
        results = [module.rasterize(vertices, faces, K, pose, width, height, **kwargs)
                   for module in (self.parent, geometry)]
        for name in ("depth", "face_id", "barycentric", "coverage"):
            self.assert_exact_array(getattr(results[0], name), getattr(results[1], name))
        self.assertEqual(results[0].stats, results[1].stats)
        return results

    def test_empty_outside_behind_degenerate_near_clip_and_winding(self):
        triangles = [
            [[-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]],
            [[-1., -1., .01], [1., -1., 2.], [0., 1., 2.]],
            [[-1., -1., 2.], [1., -1., .01], [0., 1., .01]],
            [[-1., -1., -1.], [1., -1., -1.], [0., 1., -1.]],
            [[100., 100., 1.], [101., 100., 1.], [100., 101., 1.]],
            [[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]],
        ]
        self.assert_raster_equal(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
        for index, triangle in enumerate(triangles):
            for face in ([0, 1, 2], [0, 2, 1]):
                for roi in (None, (5, 3, 19, 17), (50, 50, 60, 60)):
                    with self.subTest(triangle=index, face=face, roi=roi):
                        self.assert_raster_equal(np.array(triangle), np.array([face]), near=.1, roi=roi)

    def test_shared_edges_exact_depth_ties_and_occlusion_orders(self):
        square = np.array([[-1., -1., 2.], [-1., 1., 2.], [1., -1., 2.], [1., 1., 2.]])
        vertices = np.concatenate([square, square, square * .75, square * 1.25])
        faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6], [6, 5, 7],
                          [8, 9, 10], [10, 9, 11], [12, 13, 14], [14, 13, 15]])
        for ordering in (faces, faces[::-1], faces[:, ::-1], faces[[2, 4, 0, 6, 3, 5, 1, 7]]):
            with self.subTest(order=ordering.tolist()):
                self.assert_raster_equal(vertices, ordering)

    def test_seeded_randomized_clipped_and_overlapping_scenes_are_bit_exact(self):
        random = np.random.default_rng(827310)
        for scene in range(160):
            vertices = random.uniform(-1.5, 1.5, (18, 3))
            vertices[:, 2] = random.uniform(-.3, 3., 18)
            faces = np.array([random.choice(18, 3, replace=False) for _ in range(14)])
            width, height = random.integers(8, 36, 2).tolist()
            K = np.array([[random.uniform(1, 28), 0., random.uniform(-4, width + 4)],
                          [0., random.uniform(1, 28), random.uniform(-4, height + 4)], [0., 0., 1.]])
            roi = None if scene % 3 else tuple(random.integers(-5, max(width, height) + 5, 4).tolist())
            with self.subTest(scene=scene):
                self.assert_raster_equal(vertices, faces, K=K, width=width, height=height, near=.075, roi=roi)

    def test_rgb_and_rgba_artwork_sampling_and_stats_are_exact(self):
        vertices = np.array([[-1., -1., 2.], [-1., 1., 2.], [1., -1., 2.], [1., 1., 2.]])
        faces = np.array([[0, 1, 2], [2, 1, 3]])
        K = np.array([[20., 0., 15.5], [0., 20., 15.5], [0., 0., 1.]])
        mask = np.ones((32, 32), bool)
        mask[:, :14] = False
        art = geometry.builtin_artwork("directional", edge=8)
        rgba = np.concatenate((art, np.arange(64, dtype=np.uint8).reshape(8, 8, 1) * 4), axis=-1)
        for kind, artwork in (("render_marker", art), ("render_marker_rgba", rgba)):
            results = []
            for module in (self.parent, geometry):
                pose = np.eye(4)
                hit = module.anchor_from_pixel(vertices, faces, K, pose, 16, 16, width=32, height=32)
                marker = module.make_marker(vertices, faces, hit, pose, .8, "a" * 64)
                results.append(getattr(module, kind)(vertices, faces, np.full(2, 8), marker, artwork,
                                                     K, pose, 32, 32, mask))
            with self.subTest(renderer=kind):
                self.assert_exact_array(results[0]["rgba"], results[1]["rgba"])
                self.assertEqual(results[0]["stats"], results[1]["stats"])
                self.assertGreater(results[1]["stats"]["rendered_pixels"], 0)

    def direct_triangle(self, module, *, fill=np.inf, window=None, basis=None):
        depth = np.full((9, 9), fill, dtype=np.float64)
        face_id = np.full((9, 9), -1, np.int32)
        bary = np.zeros((9, 9, 3), np.float64)
        points = np.array([[0., 0., 1.], [8., 0., 1.], [0., 8., 1.]])
        if basis is None:
            basis = np.eye(3)
        stats = {"guard_band_skipped": 0, "degenerate_screen": 0}
        with interpolation_trace(module) as work:
            module._raster_triangle(7, list(zip(points, basis)), 1., 1., 0., 0., 9, 9,
                                    depth, face_id, bary, stats, window)
        return depth, face_id, bary, stats, work

    def test_actual_interpolation_shapes_show_only_visible_work_and_none_when_occluded(self):
        for fill in (np.inf, 1., .5):
            old = self.direct_triangle(self.parent, fill=fill)
            new = self.direct_triangle(geometry, fill=fill)
            for left, right in zip(old[:3], new[:3]):
                self.assert_exact_array(left, right)
            self.assertEqual(old[3], new[3])
            self.assertEqual(old[4], [{"shape": (9, 9, 3), "samples": 81, "bbox_samples": 81}])
            if fill == np.inf:
                retained = int(np.count_nonzero(new[1] >= 0))
                self.assertEqual(retained, 28)
                self.assertEqual(new[4], [{"shape": (28, 3), "samples": 28, "bbox_samples": 81}])
            else:
                self.assertEqual(new[4], [])

    def test_selected_nan_basis_and_signed_zero_bits_match_parent(self):
        for basis in (np.array([[np.nan, 0., -0.], [0., 1., 0.], [0., 0., 1.]]),
                      np.array([[1., -0., 0.], [-0., 1., 0.], [0., 0., 1.]])):
            old = self.direct_triangle(self.parent, basis=basis)
            new = self.direct_triangle(geometry, basis=basis)
            for left, right in zip(old[:3], new[:3]):
                self.assert_exact_array(left, right)

    def test_strict_external_error_policy_and_nonfinite_depth_comparisons_match(self):
        signaling_nan = np.array([0x7ff0000000000001], dtype=np.uint64).view(np.float64)[0]
        original_policy = np.geterr()
        for fill in (np.inf, np.nan, signaling_nan, -0., .5):
            with self.subTest(fill=repr(fill)), np.errstate(all="raise"):
                old = self.direct_triangle(self.parent, fill=fill)
                new = self.direct_triangle(geometry, fill=fill)
                for left, right in zip(old[:3], new[:3]):
                    self.assert_exact_array(left, right)
        self.assertEqual(np.geterr(), original_policy)

    def test_invalid_public_inputs_keep_error_type_and_message(self):
        vertices = np.array([[-1., -1., 2.], [1., -1., 2.], [0., 1., 2.]])
        for kwargs in ({"near": 0}, {"near": float("nan")}, {"roi": (1, 2)},
                       {"K": np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]])}):
            errors = []
            for module in (self.parent, geometry):
                options = dict(kwargs)
                K = options.pop("K", np.eye(3))
                try:
                    module.rasterize(vertices, np.array([[0, 1, 2]]), K, np.eye(4), 9, 9, **options)
                except Exception as error:
                    errors.append((type(error).__name__, str(error)))
            with self.subTest(options=kwargs):
                self.assertEqual(len(errors), 2)
                self.assertEqual(errors[0], errors[1])


if __name__ == "__main__":
    unittest.main()
