"""Independent sampling-footprint and integer-camera tests; no model/media I/O."""
import ast
from fractions import Fraction
from pathlib import Path
import unittest

import numpy as np

from . import quality_capture as capture
from .vision import cv2


def point_oracle(valid, uv):
    x, y = (int(np.floor(v)) for v in uv)
    height, width = valid.shape
    if min(x, y) < 0 or x + 1 >= width or y + 1 >= height:
        return False
    return all(bool(valid[yy, xx]) for yy in (y, y + 1) for xx in (x, x + 1))


def exact_area_oracle(valid, shape):
    """Enumerate strictly positive rational cell overlaps, independent of integral images."""
    height, width = valid.shape
    oh, ow = shape
    out = np.ones(shape, dtype=bool)
    for oy in range(oh):
        top, bottom = Fraction(oy * height, oh), Fraction((oy + 1) * height, oh)
        ys = [iy for iy in range(height) if min(bottom, iy + 1) > max(top, iy)]
        for ox in range(ow):
            left, right = Fraction(ox * width, ow), Fraction((ox + 1) * width, ow)
            xs = [ix for ix in range(width) if min(right, ix + 1) > max(left, ix)]
            out[oy, ox] = all(bool(valid[y, x]) for y in ys for x in xs)
    return out


class CaptureMathTests(unittest.TestCase):
    def test_top_level_imports_are_only_standard_library(self):
        source = Path(capture.__file__).read_text(encoding='utf-8')
        allowed = {'__future__', 'argparse', 'hashlib', 'json', 'math', 'os', 're',
                   'tempfile', 'collections', 'dataclasses', 'fractions', 'pathlib',
                   'types', 'typing', 'datetime', 'sys', 'time', 'uuid', 'shutil'}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.assertIn(name.name.split('.')[0], allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or '').split('.')[0], allowed)

    def test_integer_points_require_even_zero_weight_neighbors(self):
        valid = np.ones((5, 7), dtype=bool)
        valid[2, 3] = False
        points = np.array([[2, 1], [3, 2], [6, 1], [1, 4],
                           [1.25, .75], [-.001, 1], [3.001, 3.001]])
        expected = np.array([point_oracle(valid, point) for point in points])
        np.testing.assert_array_equal(capture.supported_points(valid, points), expected)
        self.assertFalse(expected[0], 'integer coordinates still require the fourth neighbor')
        self.assertTrue(expected[4])

    def test_fractional_support_and_input_arrays_are_unchanged(self):
        rng = np.random.default_rng(392)
        valid = rng.random((13, 17)) > .1
        points = rng.uniform([-2, -2], [19, 15], (150, 2))
        before_valid, before_points = valid.copy(), points.copy()
        expected = [point_oracle(valid, point) for point in points]
        np.testing.assert_array_equal(capture.supported_points(valid, points), expected)
        np.testing.assert_array_equal(valid, before_valid)
        np.testing.assert_array_equal(points, before_points)

    def test_point_support_rejects_malformed_evidence(self):
        valid = np.ones((4, 5), dtype=bool)
        for points in (np.array([[np.nan, 1]]), np.array([[np.inf, 1]]),
                       np.zeros((2, 3)), np.ones((2, 2), dtype=bool)):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    capture.supported_points(valid, points)
        with self.assertRaises(ValueError):
            capture.supported_points(valid.astype(np.uint8), np.zeros((1, 2)))

    def test_foreground_intersection_is_fresh_and_excludes_invalid_pixels(self):
        valid = np.ones((4, 5), dtype=bool)
        valid[1, 2] = False
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 1:4] = 255
        before = mask.copy()
        actual = capture.restrict_foreground(mask, valid)
        np.testing.assert_array_equal(actual, (mask > 0) & valid)
        self.assertEqual(actual.dtype, np.bool_)
        actual[:] = False
        np.testing.assert_array_equal(mask, before)
        self.assertTrue(valid[2, 2])
        for bad in (np.full((4, 5), 17, dtype=np.uint8), np.ones((4, 5), dtype=float),
                    np.ones((3, 5), dtype=bool)):
            with self.assertRaises(ValueError):
                capture.restrict_foreground(bad, valid)

    def test_area_support_matches_rational_noninteger_downsampling(self):
        rng = np.random.default_rng(571)
        for source, shape in (((17, 22), (11, 14)), ((7, 13), (3, 5)),
                              ((6, 9), (3, 3)), ((5, 7), (5, 7))):
            valid = rng.random(source) > .08
            with self.subTest(source=source, shape=shape):
                actual = capture.resize_validity(valid, shape, 'INTER_AREA')
                np.testing.assert_array_equal(actual, exact_area_oracle(valid, shape))
                invalid_weight = cv2.resize((~valid).astype(np.float64),
                                            (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(actual, invalid_weight == 0)

    def test_area_hole_spreads_to_every_actual_contributor(self):
        valid = np.ones((5, 7), dtype=bool)
        valid[2, 3] = False
        actual = capture.resize_validity(valid, (3, 4), 3)
        np.testing.assert_array_equal(actual, exact_area_oracle(valid, (3, 4)))
        self.assertFalse(actual[1, 1])
        self.assertFalse(actual[1, 2])
        np.testing.assert_array_equal(capture.resize_validity(valid, (3, 4), 'area'), actual)

    def test_area_exact_rational_boundary_does_not_include_adjacent_cell(self):
        # 11*(30/22) rounds just below 15 in float64. The exact footprint
        # starts at cell 15, so an invalid cell 14 must not exclude row 11.
        valid = np.ones((30, 5), dtype=bool)
        valid[14, :] = False
        expected = exact_area_oracle(valid, (22, 3))
        self.assertTrue(expected[11].all())
        actual = capture.resize_validity(valid, (22, 3), 'area')
        np.testing.assert_array_equal(actual, expected)

    def test_area_resizing_rejects_interpolation_and_boolean_dimensions(self):
        valid = np.ones((5, 7), dtype=bool)
        for shape, kernel in (((True, 4), 'INTER_AREA'), ((0, 4), 3),
                              ((3, 4), 'nearest'), ((3, 4), 'bilinear')):
            with self.assertRaises(ValueError):
                capture.resize_validity(valid, shape, kernel)

    def test_integer_depth_lifts_full_asymmetric_skew_intrinsics(self):
        depth = np.array([[700, 850, 910], [1000, 1100, 1200]], dtype=float)
        k = np.array([[420., 8., 100.], [0., 280., 55.], [0., 0., 1.]])
        pixels, xyz = capture.integer_depth_points(depth, k)
        expected_pixels = np.array([[[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]]])
        np.testing.assert_array_equal(pixels, expected_pixels)
        expected = np.empty((2, 3, 3))
        for y in range(2):
            for x in range(3):
                yn = (y - 55) / 280
                expected[y, x] = [((x - 100) - 8 * yn) / 420 * depth[y, x],
                                  yn * depth[y, x], depth[y, x]]
        np.testing.assert_allclose(xyz, expected, atol=1e-12, rtol=0)
        projected = xyz @ k.T
        np.testing.assert_allclose(projected[..., :2] / projected[..., 2:3],
                                   expected_pixels, atol=3e-14, rtol=0)

    def test_invalid_depth_is_not_geometry(self):
        depth = np.array([[0., -1., np.nan], [np.inf, 1000., 500.]])
        before = depth.copy()
        pixels, xyz = capture.integer_depth_points(depth, np.eye(3))
        self.assertEqual(pixels.shape, (2, 3, 2))
        self.assertTrue(np.isnan(xyz[0]).all())
        self.assertTrue(np.isnan(xyz[1, 0]).all())
        np.testing.assert_array_equal(xyz[1, 1], [1000, 1000, 1000])
        np.testing.assert_array_equal(depth, before)
        with self.assertRaises(ValueError):
            capture.integer_depth_points(np.ones((2, 3), dtype=bool), np.eye(3))
        with self.assertRaises(ValueError):
            capture.integer_depth_points(depth, np.zeros((3, 3)))

    def test_crop_identity_retains_four_neighbor_border_policy(self):
        valid = np.ones((8, 9), dtype=bool)
        valid[3, 4] = False
        expected = np.zeros_like(valid)
        for y in range(8):
            for x in range(9):
                expected[y, x] = point_oracle(valid, [x, y])
        actual = capture.crop_validity(valid, np.eye(3), np.eye(3), np.eye(3), valid.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_crop_rotation_uses_native_from_crop_and_real_intrinsics(self):
        valid = np.ones((70, 95), dtype=bool)
        valid[28:33, 49:53] = False
        native_k = np.array([[70., 0., 47.], [0., 60., 35.], [0., 0., 1.]])
        crop_k = np.array([[20., 0., 4.], [0., 22., 3.], [0., 0., 1.]])
        angle = .19
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        expected = np.zeros((7, 9), dtype=bool)
        for y in range(7):
            for x in range(9):
                ray = np.linalg.solve(rotation, np.linalg.solve(crop_k, [x, y, 1]))
                q = native_k @ ray
                expected[y, x] = q[2] > 0 and point_oracle(valid, q[:2] / q[2])
        actual = capture.crop_validity(valid, native_k, crop_k, rotation, (7, 9))
        np.testing.assert_array_equal(actual, expected)

    def test_crop_rejects_improper_rotation_and_masks_backwards_rays(self):
        valid = np.ones((7, 9), dtype=bool)
        for bad in (np.diag([1., 1., -1.]), 2 * np.eye(3)):
            with self.assertRaises(ValueError):
                capture.crop_validity(valid, np.eye(3), np.eye(3), bad, (4, 4))
        actual = capture.crop_validity(valid, np.eye(3), np.eye(3),
                                       np.diag([-1., 1., -1.]), (4, 4))
        self.assertFalse(actual.any())

    def test_lk_support_contains_complete_pyramid_ancestor_footprints(self):
        valid = np.ones((63, 65), dtype=bool)
        valid[31, 32] = False
        output = capture.lk_support_pyramid(valid, levels=3, window=3)
        self.assertEqual([array.shape for array in output], [(63, 65), (32, 33), (16, 17)])
        for level, actual in enumerate(output):
            factor = 2**level
            radius = factor * 2 + 2 * (factor - 1)
            expected = np.zeros_like(actual)
            for y in range(actual.shape[0]):
                for x in range(actual.shape[1]):
                    cy, cx = y * factor, x * factor
                    if cy - radius < 0 or cx - radius < 0 or cy + radius >= 63 or cx + radius >= 65:
                        continue
                    expected[y, x] = valid[cy-radius:cy+radius+1, cx-radius:cx+radius+1].all()
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(actual.flags.writeable)

    def test_lk_rectangular_stencils_and_unsupported_coarsest_level(self):
        valid = np.ones((31, 35), dtype=bool)
        output = capture.lk_support_pyramid(valid, levels=2, window=(3, 5))
        self.assertTrue(output[0][2, 3])
        self.assertFalse(output[0][2, 2])
        self.assertTrue(output[1][3, 4])
        self.assertFalse(output[1][2, 4])
        coarse = capture.lk_support_pyramid(np.ones((63, 65), dtype=bool), levels=3, window=21)
        self.assertFalse(coarse[2].any())
        for levels, window in ((True, 3), (0, 3), (2, 4), (2, (3, 0))):
            with self.assertRaises(ValueError):
                capture.lk_support_pyramid(valid, levels=levels, window=window)


if __name__ == '__main__':
    unittest.main()
