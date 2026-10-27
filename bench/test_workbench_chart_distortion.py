"""Model-free local chart-stretch and actual renderer regression tests."""
import copy
import json
import math
import unittest

import numpy as np

from bench import workbench_geometry as wg
from bench import workbench_neural as neural
from bench import workbench_contract as wc


def strip():
    vertices = np.array([[-.5, -.5, 0], [-0., -.5, 0], [-.5, .5, 0],
                         [0., .5, 0], [.25, -.5, math.sqrt(3)/4],
                         [.25, .5, math.sqrt(3)/4]], dtype=np.float64)
    faces = np.array([[0, 2, 1], [1, 2, 3], [1, 3, 4], [4, 3, 5]], dtype=np.int64)
    marker = dict(face_id=0, barycentric=[1/3]*3, tangent=[1., 0., 0.],
                  bitangent=[0., -1., 0.], normal=[0., 0., -1.], size_object_units=4.)
    return vertices, faces, np.full(4, 3), marker


class ChartStretchTests(unittest.TestCase):
    def test_plane_one_and_sixty_degree_two(self):
        v, f, support, marker = strip()
        np.testing.assert_allclose(wg.face_chart_stretches(v, f, marker), [1, 1, 2, 2], atol=1e-14)
        patch = wg.marker_patch(v, f, support, marker, max_chart_stretch=1.)
        np.testing.assert_array_equal(patch['faces'], [0, 1])
        self.assertEqual(patch['max_accepted_chart_stretch'], 1.)
        self.assertEqual(patch['chart_stretch_rejected_faces'], 1)
        self.assertEqual(len(wg.marker_patch(v, f, support, marker, max_chart_stretch=2.01)['faces']), 4)

    def test_cylinder_facets_against_chord_projection_ratio(self):
        angles = np.radians([-70., -40., -10., 20., 50., 80.])
        row = np.column_stack((np.sin(angles), np.zeros(len(angles)), np.cos(angles)))
        v = np.concatenate((row, row + [0, 1, 0]))
        f = np.array([[i, i+1, i+len(row)] for i in range(len(row)-1)])
        marker = strip()[3]
        measured = wg.face_chart_stretches(v, f, marker)
        chord = np.diff(row, axis=0)
        oracle = np.linalg.norm(chord, axis=1) / np.abs(chord[:, 0])
        np.testing.assert_allclose(measured, oracle, rtol=1e-14)

    def test_invalid_caps_and_marker_frames_refused(self):
        v, f, s, m = strip()
        for cap in (True, False, np.bool_(True), float('nan'), float('inf'), -1, .99, '2'):
            with self.subTest(cap=cap), self.assertRaises(wg.GeometryError):
                wg.marker_patch(v, f, s, m, max_chart_stretch=cap)
        for replacement in ([2., 0., 0.], [0., -1., 0.], [float('nan'), 0., 0.]):
            bad = dict(m, tangent=replacement)
            with self.assertRaises(wg.GeometryError):
                wg.face_chart_stretches(v, f, bad)
        with self.assertRaises(wg.GeometryError):
            wg.face_chart_stretches(v, f, dict(m, normal=[0., 0., 1.]))
        with self.assertRaises(wg.GeometryError):
            wg.marker_patch(v, f.astype(float) + .25, s, m, max_chart_stretch=2.)

    def test_degenerate_and_grazing_anchor_suppress_with_json_safe_stats(self):
        _, _, _, m = strip()
        for v in (np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
                  np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                  np.array([[0., 0., 0.], [0., 1., 0.], [1e-14, 0., 1.]])):
            patch = wg.marker_patch(v, np.array([[0, 1, 2]]), [3], m, max_chart_stretch=1e100)
            self.assertEqual(len(patch['faces']), 0)
            self.assertIsNone(patch['max_accepted_chart_stretch'])
            self.assertEqual(patch['chart_stretch_unavailable_faces'], 1)
            json.dumps({k: val for k, val in patch.items() if k.startswith('chart_')
                        or k == 'max_accepted_chart_stretch'}, allow_nan=False)

    def test_units_rigid_transform_and_inputs_immutable(self):
        v, f, s, m = strip()
        original = (v.copy(), f.copy(), copy.deepcopy(m))
        theta = .63
        rotation = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0],
                             [-math.sin(theta), 0, math.cos(theta)]])
        for scale in (1e-12, 1., 1e12):
            moved = v @ rotation.T * scale + np.array([.3, -.2, .7])*scale
            marker = dict(m, size_object_units=m['size_object_units']*scale)
            for key in ('tangent', 'bitangent', 'normal'):
                marker[key] = rotation @ m[key]
            np.testing.assert_allclose(wg.face_chart_stretches(moved, f, marker), [1, 1, 2, 2], rtol=1e-14)
            np.testing.assert_array_equal(wg.marker_patch(moved, f, s, marker, max_chart_stretch=1.5)['faces'], [0, 1])
        np.testing.assert_array_equal(v, original[0])
        np.testing.assert_array_equal(f, original[1])
        self.assertEqual(m, original[2])

    def test_rejected_bridge_cannot_reach_later_good_face(self):
        v, f, s, m = strip()
        v = np.concatenate((v, [[.75, .5, math.sqrt(3)/4]]))
        # A plane-parallel face beyond a rejected 60-degree bridge.
        f = np.concatenate((f, [[4, 5, 6]]))
        patch = wg.marker_patch(v, f, np.full(5, 3), m, max_chart_stretch=1.5)
        np.testing.assert_array_equal(patch['faces'], [0, 1])
        s[1] = 0
        np.testing.assert_array_equal(wg.marker_patch(v[:6], f[:4], s, m, max_chart_stretch=3.)['faces'], [0])

    def test_actual_render_stale_patch_mask_depth_and_default_compatibility(self):
        v, f, s, m = strip()
        K = np.array([[100., 0, 47.5], [0, 100., 47.5], [0, 0, 1.]])
        pose = np.eye(4)
        pose[2, 3] = 2.
        mask = np.ones((96, 96), bool)
        art = wg.builtin_artwork('directional', 32)
        stale = wg.marker_patch(v, f, s, m)
        def render(**kwargs):
            return wg.render_marker(v, f, s, m, art, K, pose, 96, 96, mask, **kwargs)
        legacy = render(patch=stale)
        explicit_none = render(patch=stale, max_chart_stretch=None)
        np.testing.assert_array_equal(legacy['rgba'], explicit_none['rgba'])
        self.assertEqual(legacy['stats'], explicit_none['stats'])
        self.assertNotIn('chart_stretch_limit', legacy['stats'])
        guarded = render(patch=stale, max_chart_stretch=1.5)
        raster = wg.rasterize(v, f, K, pose, 96, 96)
        alpha = guarded['rgba'][..., 3] > 0
        self.assertGreater(int(alpha.sum()), 0)
        self.assertGreater(legacy['stats']['rendered_pixels'], guarded['stats']['rendered_pixels'])
        self.assertFalse(np.any(alpha & np.isin(raster.face_id, [2, 3])))
        mask[:] = False
        self.assertFalse(render(patch=stale, max_chart_stretch=1.5)['rgba'].any())
        mask[:] = True
        # A nearer disconnected face is depth-tested but never joins the patch.
        occluder = v[:3].copy()
        occluder[:, 2] -= .2
        v2 = np.concatenate((v, occluder))
        f2 = np.concatenate((f, [[6, 7, 8]]))
        ras2 = wg.rasterize(v2, f2, K, pose, 96, 96)
        result = wg.render_marker(v2, f2, np.full(5, 3), m, art, K, pose, 96, 96, mask,
                                  max_chart_stretch=1.5)
        self.assertTrue(np.any(ras2.face_id == 4))
        self.assertFalse(np.any((result['rgba'][..., 3] > 0) & (ras2.face_id == 4)))

    def test_neural_policy_is_read_only_and_private_mutation_refused(self):
        # Manufacture only the already-bound state; parent tests the real GLB
        # constructor separately. No inference or source assets are needed.
        obj = neural.NeuralArtwork.__new__(neural.NeuralArtwork)
        obj._max_chart_stretch = 1.5
        obj.record = {'chart_stretch_policy': {'max_chart_stretch': 1.5}}
        obj.sha256 = wc.canonical_sha256(obj.record)
        with self.assertRaises(AttributeError):
            obj.max_chart_stretch = 2.
        obj._max_chart_stretch = 2.
        with self.assertRaises(wg.GeometryError):
            obj.render(np.eye(4), np.ones((8, 8), bool))
        obj._max_chart_stretch = 1.5
        obj.record['chart_stretch_policy']['max_chart_stretch'] = 2.
        with self.assertRaises(wg.GeometryError):
            obj.render(np.eye(4), np.ones((8, 8), bool))


if __name__ == '__main__':
    unittest.main()
