import unittest
import numpy as np
from .model import Camera, Mesh, texture_uv
from .renderer import render, ray_triangle


def triangle(z=1):
    return Mesh('triangle', [[-.3, -.3, z], [.3, -.3, z], [-.3, .3, z]],
                [[0, 0, -1]]*3, [[0, 0], [1, 0], [0, 1]], [[0, 2, 1]])


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera.look_at([0, 0, 0], [0, 0, 1], width=96, height=96, focal=80)

    def test_camera_calibration_axes(self):
        pixels, depth = self.camera.project([[0, 0, 1], [.1, 0, 1], [0, .1, 1]])
        np.testing.assert_allclose(pixels, [[48, 48], [56, 48], [48, 40]])
        np.testing.assert_allclose(depth, 1)

    def test_project_unproject_in_nontrivial_pose(self):
        camera = Camera.look_at([.28, .26, -.45])
        points = np.array([[-.03, .07, .02], [.04, .15, 0], [0, .02, -.03]])
        pixels, depth = camera.project(points)
        np.testing.assert_allclose(camera.eye+camera.rays(pixels)*depth[:, None], points, atol=1e-12)

    def test_depth_is_order_independent(self):
        near, far = triangle(1), triangle(2)
        a = render(near, self.camera, object_id=1)
        render(far, self.camera, frame=a, object_id=2)
        b = render(far, self.camera, object_id=2)
        render(near, self.camera, frame=b, object_id=1)
        np.testing.assert_array_equal(a.color, b.color)
        np.testing.assert_array_equal(a.depth, b.depth)
        self.assertEqual(a.object_id[50, 40], 1)

    def test_perspective_uv_matches_independent_ray_oracle(self):
        mesh = triangle()
        mesh.positions[1, 2] = 2
        good = render(mesh, self.camera)
        wrong = render(mesh, self.camera, perspective=False)
        errors = []
        for py, px in np.argwhere(good.triangle >= 0)[::13]:
            ray = self.camera.rays([px+.5, py+.5])
            hit = ray_triangle(self.camera.eye, ray, mesh.positions, mesh.uv)
            self.assertIsNotNone(hit)
            np.testing.assert_allclose(good.uv[py, px], hit[1], atol=1e-12)
            self.assertAlmostEqual(good.depth[py, px], hit[0], places=12)
            errors.append(np.linalg.norm(wrong.uv[py, px]-hit[1]))
        self.assertGreater(max(errors), .05)

    def test_near_clipping_preserves_uvs(self):
        mesh = triangle()
        mesh.positions = np.array([[-.015, -.015, .01], [.3, -.3, 1], [-.3, .3, 1]])
        result = render(mesh, self.camera, cull=False)
        coordinates = np.argwhere(result.triangle >= 0)
        self.assertGreater(len(coordinates), 0)
        for py, px in coordinates[::19]:
            hit = ray_triangle(self.camera.eye, self.camera.rays([px+.5, py+.5]), mesh.positions, mesh.uv)
            self.assertIsNotNone(hit)
            np.testing.assert_allclose(result.uv[py, px], hit[1], atol=1e-10)
            self.assertGreaterEqual(result.depth[py, px], self.camera.near-1e-10)

    def test_behind_camera_has_no_pixels(self):
        self.assertFalse((render(triangle(-1), self.camera, cull=False).triangle >= 0).any())

    def test_occluder_masks_rear_surface(self):
        rear = render(triangle(2), self.camera, object_id=0)
        mask_before = rear.object_id == 0
        render(triangle(1), self.camera, frame=rear, object_id=1)
        self.assertGreater((mask_before & (rear.object_id == 1)).sum(), 0)

    def test_shared_edge_has_no_crack(self):
        mesh = Mesh('quad', [[-.3, -.3, 1], [.3, -.3, 1], [.3, .3, 1], [-.3, .3, 1]],
                    [[0, 0, -1]]*4, [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 2, 1], [0, 3, 2]])
        result = render(mesh, self.camera)
        self.assertTrue((result.triangle[25:71, 25:71] >= 0).all())

    def test_mapping_aspect_and_rotation(self):
        np.testing.assert_allclose(texture_uv([[.04, .08]], .04, 2), [[1, 1]])
        np.testing.assert_allclose(texture_uv([[.04, 0]], .04, 1, 90), [[0, 1]], atol=1e-12)

    def test_invalid_calibration_and_geometry_rejected(self):
        with self.assertRaises(ValueError):
            Camera.look_at([0, 0, 0], [0, 0, 1], focal=0)
        with self.assertRaises(ValueError):
            Mesh('bad', [[0, 0, 0]], [[0, 1, 0]], [[0, 0]], [[0, 1, 2]])
        with self.assertRaises(ValueError):
            texture_uv([[0, 0]], .05, float('nan'))


if __name__ == '__main__':
    unittest.main()
