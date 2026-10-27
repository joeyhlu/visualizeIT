"""Independent manufactured geometry oracles; no model or physical accuracy claim."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_geometry as geom
from bench import workbench_neural as neural
from bench.test_workbench_geometry import cylinder_mesh
from tools.test_workbench_offline_export_integration import make_fixture


def curved_fixture():
    vertices, faces = cylinder_mesh(segments=48, rows=4)
    K = np.array([[100., 0., 63.5], [0., 100., 63.5], [0., 0., 1.]])
    pose = np.eye(4); pose[2, 3] = 2.
    mask = np.ones((128, 128), dtype=bool)
    hit = geom.anchor_from_pixel(vertices, faces, K, pose, 63.5, 63.5,
                                 width=128, height=128, observed_mask=mask)
    marker = geom.make_marker(vertices, faces, hit, pose, 1.2, 'a'*64)
    support = np.full(len(faces), 3, dtype=np.int32)
    return vertices, faces, K, pose, mask, marker, support


class IndependentChartDistortionChecks(unittest.TestCase):
    def test_exact_plane_and_sixty_degree_projection(self):
        marker = dict(face_id=0, barycentric=[1., 0., 0.], size_object_units=1.,
                      tangent=[1., 0., 0.], bitangent=[0., -1., 0.], normal=[0., 0., -1.])
        for degrees, expected in ((0, 1.), (60, 2.)):
            angle = math.radians(degrees)
            vertices = np.array([[0., 0., 0.], [0., 1., 0.],
                                 [math.cos(angle), 0., -math.sin(angle)]])
            # Analytic construction: unit horizontal edge projects to length
            # cos(theta), while the vertical unit edge is unchanged.
            actual = geom.face_chart_stretches(vertices, np.array([[0, 1, 2]]), marker)
            np.testing.assert_allclose(actual, [expected], rtol=1e-12, atol=1e-12)

    def test_cylinder_stretch_matches_independent_edge_length_ratio(self):
        vertices, faces, K, pose, mask, marker, support = curved_fixture()
        basis = np.array([marker['tangent'], marker['bitangent']])
        expected = []
        for tri in vertices[faces].astype(np.float64):
            # Each manufactured cylinder triangle has one circumferential edge
            # with equal endpoint y. Its 3D/chart length ratio is the exact
            # inverse stretch; no face-normal formula is used by this oracle.
            edge = next(tri[b]-tri[a] for a, b in ((0, 1), (1, 2), (2, 0))
                        if tri[b, 1] == tri[a, 1])
            ratio = np.linalg.norm(basis @ edge) / np.linalg.norm(edge)
            # The declared numerical grazing domain is projected length ratio
            # <=1e-12; those inverse maps are unavailable rather than finite.
            expected.append(1. / ratio if ratio > 1e-12 else np.inf)
        actual = geom.face_chart_stretches(vertices, faces, marker)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        # Rigid coordinate changes and metres-to-millimetres scaling preserve
        # this dimensionless ratio, independently of absolute metric size.
        angle = .37
        R = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                      [-math.sin(angle), 0., math.cos(angle)]])
        moved_marker = copy.deepcopy(marker)
        for key in ('tangent', 'bitangent', 'normal'):
            moved_marker[key] = (R @ np.array(marker[key])).tolist()
        moved = vertices.astype(np.float64) @ R.T * 1000. + np.array([5., 10., -3.])
        np.testing.assert_allclose(geom.face_chart_stretches(moved, faces, moved_marker),
                                   expected, rtol=2e-11, atol=2e-11)

    def test_actual_curved_renderer_enforces_cap_despite_legacy_patch(self):
        vertices, faces, K, pose, mask, marker, support = curved_fixture()
        old_marker = copy.deepcopy(marker)
        old_vertices, old_faces = vertices.copy(), faces.copy()
        art = geom.builtin_artwork('directional', edge=32)
        raster = geom.rasterize(vertices, faces, K, pose, 128, 128)
        legacy = geom.marker_patch(vertices, faces, support, marker)
        def render(cap, observed=mask, supplied=legacy):
            return geom.render_marker(vertices, faces, support, marker, art, K, pose,
                128, 128, observed, raster=raster, patch=supplied, max_chart_stretch=cap)
        plain = render(None)
        limited = render(1.2)
        plain_alpha = plain['rgba'][..., 3] > 0
        alpha = limited['rgba'][..., 3] > 0
        self.assertGreater(int(alpha.sum()), 0)
        self.assertLess(int(alpha.sum()), int(plain_alpha.sum()))
        # Every painted face satisfies the independently tested local ratio.
        lengths = geom.face_chart_stretches(vertices, faces, marker)
        self.assertTrue(np.all(lengths[raster.face_id[alpha]] <= 1.2))
        self.assertGreater(limited['stats']['chart_stretch_rejected_faces'], 0)
        self.assertEqual(int(render(1.2, np.zeros_like(mask))['rgba'][..., 3].sum()), 0)
        np.testing.assert_array_equal(vertices, old_vertices)
        np.testing.assert_array_equal(faces, old_faces)
        self.assertEqual(marker, old_marker)

    def test_partial_supported_mesh_does_not_acquire_disconnected_faces(self):
        vertices, faces, K, pose, mask, marker, support = curved_fixture()
        # A nearby parallel triangle with separate vertices is not connected to
        # the prepared patch. A cap must not invent that connection.
        extended_v = np.vstack((vertices, vertices[faces[marker['face_id']]] + [0., 0., .001]))
        extended_f = np.vstack((faces, np.arange(len(vertices), len(vertices)+3)))
        extended_support = np.r_[support, 3]
        patch = geom.marker_patch(extended_v, extended_f, extended_support, marker,
                                  max_chart_stretch=1.2)
        self.assertNotIn(len(faces), patch['faces'])
        extended_support[marker['face_id']] = 0
        unsupported = geom.marker_patch(extended_v, extended_f, extended_support, marker,
                                        max_chart_stretch=1.2)
        self.assertEqual(len(unsupported['faces']), 0)
        self.assertEqual(unsupported['reason'], 'anchor_face_unsupported')

    def test_actual_metric_artwork_binds_guard_and_default_plane_pixels(self):
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix='chart-metric-plane-') as folder:
            fixture = make_fixture(Path(folder))
            workspace = fixture['workspace']
            run = wc.RunSpec.from_dict(wc.loads_strict(
                (workspace.runs_dir / fixture['run_id'] / 'run.json').read_bytes())['run'])
            clip = workspace.load_clip(run.clip_id)
            bundle = neural.MetricBundle.from_file(fixture['bundle_path'])
            args = dict(clip_K=clip.intrinsics.K, width=clip.width, height=clip.height,
                        selection_mask=fixture['masks']['f0'])
            plain = neural.NeuralArtwork(bundle, run.onboarding.selection, run.artwork_kind, **args)
            bounded = neural.NeuralArtwork(bundle, run.onboarding.selection, run.artwork_kind,
                                            max_chart_stretch=1.25, **args)
            self.assertNotIn('chart_stretch_policy', plain.record)
            self.assertNotEqual(plain.sha256, bounded.sha256)
            self.assertEqual(plain.marker, bounded.marker)
            # The manufactured GLB is exactly planar: the cap removes no faces
            # and cannot change a pixel, regardless of camera perspective.
            expected = plain.render(bundle.seed_pose_m, fixture['masks']['f0'])['rgba']
            actual = bounded.render(bundle.seed_pose_m, fixture['masks']['f0'])['rgba']
            np.testing.assert_array_equal(actual, expected)
            bounded.record['chart_stretch_policy']['max_chart_stretch'] = 2.
            with self.assertRaises(geom.GeometryError):
                bounded.render(bundle.seed_pose_m, fixture['masks']['f0'])


if __name__ == '__main__':
    unittest.main()
