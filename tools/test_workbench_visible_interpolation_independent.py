"""Exact Git-parent controls for visibility-filtered surface interpolation.

Manufactured scenes test software parity and work counts, not tracking quality
or representative performance. The existing cylinder oracle remains separate.
"""
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest

import numpy as np

from bench import workbench_geometry as current
from tools import test_workbench_cylinder_oracle as cylinder


BASE = 'c02fcb89074a283f63f4967fd39d9580e55ff230'
BASE_GEOMETRY_SHA256 = 'd21f665ac2237729d342e96a32e9745136f58bd741e2f369049889d894f6763c'


class IndependentVisibleInterpolationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = subprocess.check_output(
            ['git', 'show', BASE + ':bench/workbench_geometry.py'],
            cwd=Path(__file__).resolve().parents[1])
        if hashlib.sha256(raw).hexdigest() != BASE_GEOMETRY_SHA256:
            raise AssertionError('Exact published raster parent bytes differ')
        name = 'bench._visible_interpolation_independent_parent'
        cls.parent = ModuleType(name)
        cls.parent.__file__, cls.parent.__package__ = current.__file__, 'bench'
        sys.modules[name] = cls.parent
        exec(compile(raw, '<raster-parent-c02fcb8>', 'exec'), cls.parent.__dict__)

    def assert_raster_exact(self, left, right):
        self.assertEqual(left.stats, right.stats)
        for name in ('depth', 'face_id', 'barycentric'):
            a, b = getattr(left, name), getattr(right, name)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            # Include signed zeros and exact NaN payloads in the parity contract.
            self.assertEqual(a.tobytes(), b.tobytes(), name)

    def test_seeded_clipped_overlapping_scenes_match_parent_bits(self):
        K = np.array([[39.3, 0., 29.1], [0., 42.7, 21.8], [0., 0., 1.]])
        for seed in range(48):
            rng = np.random.default_rng(seed)
            vertices = rng.uniform(-.8, .8, (36, 3))
            vertices[:, 2] = rng.uniform(-.2, 2.0, 36)
            # Mix in a coplanar duplicate to exercise strict depth ties.
            vertices[3:6] = vertices[:3]
            faces = np.arange(36).reshape(-1, 3)
            faces[::2] = faces[::2, ::-1]
            pose = np.eye(4)
            pose[:3, 3] = [rng.uniform(-.05, .05), 0., .1]
            for roi in (None, (7, 5, 41, 29), (35, 28, 12, 10)):
                with self.subTest(seed=seed, roi=roi):
                    args = (vertices, faces, K, pose, 60, 44)
                    before = (vertices.copy(), faces.copy(), K.copy(), pose.copy())
                    expected = self.parent.rasterize(*args, near=.13, roi=roi)
                    actual = current.rasterize(*args, near=.13, roi=roi)
                    self.assert_raster_exact(expected, actual)
                    for source, snapshot in zip(args[:4], before):
                        self.assertEqual(source.tobytes(), snapshot.tobytes())

    def test_curved_rgba_and_rgb_sampler_match_parent_including_known_case(self):
        # This checks exact parent preservation, not a new accuracy verdict for
        # the known cylinder texel discrepancy at the second pose.
        for missing, occluder in ((None, False), (1, False), (None, True)):
            vertices, faces, support, marker, art = cylinder.fixture(missing, occluder)
            for pose in cylinder.POSES:
                mask = np.ones((cylinder.HEIGHT, cylinder.WIDTH), dtype=bool)
                mask[20:32, 30:50] = False
                for mode in ('render_marker', 'render_marker_rgba'):
                    args = (vertices, faces, support, marker,
                            art[..., :3] if mode == 'render_marker' else art,
                            cylinder.K, pose, cylinder.WIDTH, cylinder.HEIGHT, mask)
                    with self.subTest(missing=missing, occluder=occluder, mode=mode):
                        expected = getattr(self.parent, mode)(*args)
                        actual = getattr(current, mode)(*args)
                        self.assertEqual(expected['stats'], actual['stats'])
                        self.assertEqual(expected['rgba'].tobytes(), actual['rgba'].tobytes())

    def test_invalid_input_errors_match_parent(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        faces = np.array([[0, 1, 2]])
        for near, skew in ((0., 0.), (-.1, 0.), (float('nan'), 0.), (.01, .1)):
            outcomes = []
            for module in (self.parent, current):
                K = np.eye(3)
                K[0, 1] = skew
                try:
                    module.rasterize(vertices, faces, K, np.eye(4), 8, 8, near=near)
                except Exception as error:
                    outcomes.append((type(error).__name__, str(error)))
                else:
                    self.fail('Invalid input unexpectedly accepted')
            self.assertEqual(outcomes[0], outcomes[1])

    def test_actual_interpolation_arrays_exclude_discarded_pixels(self):
        vertices = np.array([[0., 0., 1.], [30., 0., 1.], [0., 30., 1.],
                             [0., 0., 2.], [60., 0., 2.], [0., 60., 2.]])
        # Front, exactly tied, then fully occluded triangle.
        faces = np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5]])
        totals = []
        results = []
        for module in (self.parent, current):
            sizes = []
            def trace(frame, event, arg):
                if frame.f_code is module._raster_triangle.__code__ and event == 'return':
                    original = frame.f_locals.get('orig')
                    sizes.append(0 if original is None else original.size // 3)
                return trace
            previous = sys.gettrace()
            try:
                sys.settrace(trace)
                results.append(module.rasterize(vertices, faces, np.eye(3), np.eye(4), 32, 32))
            finally:
                sys.settrace(previous)
            totals.append(sizes)
        self.assert_raster_exact(*results)
        self.assertEqual(totals[0], [31 * 31] * 3)
        visible = int(results[1].coverage.sum())
        self.assertGreater(visible, 0)
        self.assertEqual(totals[1], [visible, 0, 0])
        self.assertLess(sum(totals[1]), sum(totals[0]))


if __name__ == '__main__':
    unittest.main()
