"""Model-free straight-alpha and actual-surface sampling checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_geometry as g
from bench.test_workbench_geometry import cylinder_mesh
from bench.test_workbench_chart_distortion import strip


class RgbaMarkerTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[-.5, -.5, 0], [-.5, .5, 0], [.5, -.5, 0], [.5, .5, 0.]])
        self.f = np.array([[0, 1, 2], [2, 1, 3]])
        self.s = np.full(2, 3)
        self.K = np.array([[40., 0, 31.5], [0, 40., 31.5], [0, 0, 1.]])
        self.pose = np.eye(4)
        self.pose[2, 3] = 2.
        self.mask = np.ones((64, 64), bool)
        self.m = dict(face_id=0, barycentric=[0., .5, .5], tangent=[1., 0, 0],
                      bitangent=[0., -1., 0], normal=[0., 0., -1.], size_object_units=1.)
        self.art = np.zeros((8, 8, 4), np.uint8)
        self.art[..., :3] = [17, 83, 201]
        self.art[:, :3, 3] = 0
        self.art[:, 3:6, 3] = 128
        self.art[:, 6:, 3] = 255

    def render(self, **kwargs):
        return g.render_marker_rgba(self.v, self.f, self.s, self.m, self.art,
                                    self.K, self.pose, 64, 64, self.mask, **kwargs)

    def test_plane_independent_pinhole_uv_and_exact_straight_channels(self):
        with patch.object(g, 'rasterize', wraps=g.rasterize) as rasterize:
            result = self.render()
        self.assertEqual(rasterize.call_count, 1)
        legacy = g.render_marker(self.v, self.f, self.s, self.m, self.art[..., :3],
                                 self.K, self.pose, 64, 64, self.mask)
        geometric = legacy['rgba'][..., 3] > 0
        yy, xx = np.nonzero(geometric)
        # Independent plane ray intersection: x=(pixel-cx)*depth/fx.
        columns = np.clip((((xx - 31.5)*2/40) + .5)*8, 0, 7).astype(int)
        expected_alpha = self.art[0, columns, 3]
        np.testing.assert_array_equal(result['rgba'][yy, xx, 3], expected_alpha)
        np.testing.assert_array_equal(result['rgba'][..., :3], legacy['rgba'][..., :3])
        self.assertEqual(set(expected_alpha), {0, 128, 255})
        self.assertEqual(result['stats']['rendered_pixels'], int(np.count_nonzero(expected_alpha)))
        self.assertEqual(result['stats']['geometry_rendered_pixels'], len(xx))
        self.assertEqual(result['stats']['alpha_zero_rejected_pixels'], int(np.sum(expected_alpha == 0)))
        self.assertEqual(legacy['stats']['rendered_pixels'], len(xx))
        with self.assertRaises(g.GeometryError):
            g.render_marker(self.v, self.f, self.s, self.m, self.art,
                            self.K, self.pose, 64, 64, self.mask)

    def test_transparent_empty_mask_and_support_refresh(self):
        stale = g.marker_patch(self.v, self.f, self.s, self.m)
        self.art[..., 3] = 0
        result = self.render()
        self.assertEqual(result['stats']['rendered_pixels'], 0)
        self.assertGreater(result['stats']['geometry_rendered_pixels'], 0)
        self.s[:] = 0
        result = self.render(patch=stale)
        self.assertEqual(result['stats']['reason'], 'anchor_face_unsupported')
        self.assertEqual(result['stats']['geometry_rendered_pixels'], 0)
        self.s[:] = 3
        self.mask[:] = False
        self.assertFalse(self.render()['rgba'].any())

    def test_large_triangles_crossing_small_chart_keep_full_analytic_footprint(self):
        self.v *= .4
        self.m['size_object_units'] = .16
        self.art[..., 3] = 255
        result = self.render()
        yy, xx = np.indices((64, 64))
        expected = (np.abs((xx-31.5)*2/40) <= .08) & (np.abs((yy-31.5)*2/40) <= .08)
        np.testing.assert_array_equal(result['rgba'][..., 3] > 0, expected)
        self.assertEqual(result['stats']['patch_faces'], 2)
        default = g.marker_patch(self.v, self.f, self.s, self.m)
        explicit = g.marker_patch(self.v, self.f, self.s, self.m, conservative_chart_reach=False)
        np.testing.assert_array_equal(default['faces'], [0])
        np.testing.assert_array_equal(default['faces'], explicit['faces'])
        legacy = g.render_marker(self.v, self.f, self.s, self.m, self.art[..., :3],
                                 self.K, self.pose, 64, 64, self.mask)
        self.assertLess(legacy['stats']['rendered_pixels'], result['stats']['rendered_pixels'])
        self.s[1] = 0
        self.assertEqual(self.render()['stats']['patch_faces'], 1)
        with self.assertRaises(g.GeometryError):
            g.marker_patch(self.v, self.f, self.s, self.m, conservative_chart_reach=1)

    def test_invalid_inputs_and_json_finite_guard(self):
        for art in (np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 4), float),
                    np.zeros((7, 8, 4), np.uint8), np.zeros((8, 1025, 4), np.uint8)):
            self.art = art
            with self.assertRaises(g.GeometryError):
                self.render()
        self.art = np.zeros((8, 8, 4), np.uint8)
        for mask in (None, np.ones((63, 64), bool)):
            self.mask = mask
            with self.assertRaises(g.GeometryError):
                self.render()
        self.mask = np.ones((64, 64), bool)
        for cap in (True, .9, float('inf'), float('nan')):
            with self.assertRaises(g.GeometryError):
                self.render(max_chart_stretch=cap)
        json.dumps(self.render(max_chart_stretch=1.)['stats'], allow_nan=False)

    def test_cap_stale_patch_and_real_depth_occluder(self):
        self.v, self.f, self.s, self.m = strip()
        self.art[..., 3] = 255
        stale = g.marker_patch(self.v, self.f, self.s, self.m)
        result = self.render(patch=stale, max_chart_stretch=1.5)
        raster = g.rasterize(self.v, self.f, self.K, self.pose, 64, 64)
        visible = result['rgba'][..., 3] > 0
        self.assertGreater(visible.sum(), 0)
        self.assertFalse(np.any(visible & np.isin(raster.face_id, [2, 3])))
        self.assertLessEqual(result['stats']['max_accepted_chart_stretch'], 1.5)
        occluder = self.v[:3].copy()
        occluder[:, 2] -= .2
        self.v = np.concatenate((self.v, occluder))
        self.f = np.concatenate((self.f, [[6, 7, 8]]))
        self.s = np.full(5, 3)
        raster = g.rasterize(self.v, self.f, self.K, self.pose, 64, 64)
        self.assertTrue(np.any(raster.face_id == 4))
        result = self.render(max_chart_stretch=1.5)
        self.assertFalse(np.any((result['rgba'][..., 3] > 0) & (raster.face_id == 4)))

    def test_actual_cylinder_surface_and_mask(self):
        self.v, self.f = cylinder_mesh(segments=48, rows=6)
        self.s = np.full(len(self.f), 3)
        hit = g.anchor_from_pixel(self.v, self.f, self.K, self.pose, 31.5, 31.5,
                                  width=64, height=64, observed_mask=self.mask)
        self.m = g.make_marker(self.v, self.f, hit, self.pose, .6, 'a'*64)
        self.art[..., 3] = 128
        result = self.render()
        ys, xs = np.nonzero(result['rgba'][..., 3])
        self.assertGreater(len(xs), 0)
        raster = g.rasterize(self.v, self.f, self.K, self.pose, 64, 64)
        points = g.surface_points(self.v, self.f, raster.face_id[ys, xs], raster.barycentric[ys, xs])
        radial = np.sqrt(points[:, 0]**2 + points[:, 2]**2)
        self.assertLess(float(np.max(np.abs(radial - .5))), .002)
        self.assertGreater(np.ptp(points[:, 2]), .01)
        self.assertTrue(np.all(result['rgba'][ys, xs, 3] == 128))
        self.mask[:, :32] = False
        result = self.render()
        self.assertFalse(result['rgba'][:, :32].any())
        self.assertGreater(result['stats']['mask_rejected_pixels'], 0)


if __name__ == '__main__':
    unittest.main()
