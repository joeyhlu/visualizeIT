"""Analytic geometry tests independent of the workbench implementation."""
from __future__ import annotations

import math
import unittest

import numpy as np

from bench import workbench_geometry as geom


class CameraGeometryTests(unittest.TestCase):
    def test_pinhole_projection_matches_independent_camera_equation(self):
        K = np.array([[120.0, 0.0, 31.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]])
        angle = math.radians(30.0)
        pose = np.eye(4)
        pose[:3, :3] = [[math.cos(angle), 0.0, math.sin(angle)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(angle), 0.0, math.cos(angle)]]
        pose[:3, 3] = [0.1, -0.2, 2.0]
        points = np.array([[0.2, 0.1, 0.0], [-0.4, 0.3, 0.2], [0.0, -0.1, 0.4]])
        expected_cam = points @ pose[:3, :3].T + pose[:3, 3]
        expected_uv = np.column_stack((
            K[0, 0] * expected_cam[:, 0] / expected_cam[:, 2] + K[0, 2],
            K[1, 1] * expected_cam[:, 1] / expected_cam[:, 2] + K[1, 2],
        ))
        uv, depth = geom.project(points, K, pose)
        np.testing.assert_allclose(uv, expected_uv, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(depth, expected_cam[:, 2], rtol=0.0, atol=1e-12)

    def test_normalizing_surface_and_pose_preserves_all_image_projections(self):
        vertices = np.array([
            [-0.30, -0.20, -0.12], [0.25, -0.18, -0.10],
            [-0.24, 0.22, 0.11], [0.29, 0.19, 0.15],
            [0.03, 0.01, 0.23], [-0.08, -0.04, -0.28],
        ], dtype=np.float64)
        angle = math.radians(17.0)
        pose = np.eye(4)
        pose[:3, :3] = [[math.cos(angle), 0.0, math.sin(angle)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(angle), 0.0, math.cos(angle)]]
        pose[:3, 3] = [0.03, -0.02, 2.0]
        K = np.array([[700.0, 0.0, 320.0], [0.0, 680.0, 240.0], [0.0, 0.0, 1.0]])

        normalized, record = geom.normalize_geometry(vertices)
        normalized_pose = geom.normalize_pose(pose, record)
        original_uv, original_depth = geom.project(vertices, K, pose)
        normalized_uv, normalized_depth = geom.project(normalized, K, normalized_pose)

        np.testing.assert_allclose(normalized_uv, original_uv, rtol=0.0, atol=2e-5)
        np.testing.assert_allclose(normalized_depth * record.scale, original_depth, rtol=0.0, atol=2e-6)
        self.assertGreater(record.scale, 0.0)

    def test_integer_pixel_center_and_half_pixel_colmap_round_trip(self):
        K_center = np.array([[400.0, 0.0, 319.5], [0.0, 400.0, 239.5], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[2, 3] = 2.0
        point = np.array([[0.25, -0.10, 0.0]])

        center_pixels, _ = geom.project(point, K_center, pose)
        expected_center = np.array([[369.5, 219.5]])
        np.testing.assert_allclose(center_pixels, expected_center, rtol=0.0, atol=1e-12)

        # COLMAP's corner-origin principal point is half a pixel larger. A
        # round trip through that convention must recover the source centers.
        K_colmap = K_center.copy()
        K_colmap[0, 2] += 0.5
        K_colmap[1, 2] += 0.5
        corner_pixels, _ = geom.project(point, K_colmap, pose)
        restored = corner_pixels - np.array([0.5, 0.5])
        np.testing.assert_allclose(corner_pixels, center_pixels + 0.5, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(restored, center_pixels, rtol=0.0, atol=1e-12)

    def test_points_behind_the_camera_are_not_reported_as_visible_pixels(self):
        K = np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        uv, depth = geom.project(points, K, pose)
        self.assertTrue(np.isnan(uv[0]).all())
        self.assertEqual(depth[0], -1.0)
        np.testing.assert_allclose(uv[1], [8.0, 8.0])


class RasterizationTests(unittest.TestCase):
    def test_depth_and_original_triangle_barycentrics_are_perspective_correct(self):
        K = np.eye(3, dtype=np.float64)
        pose = np.eye(4)
        vertices = np.array([
            [0.0, 0.0, 1.0],   # projects to (0, 0)
            [12.0, 0.0, 2.0],  # projects to (6, 0)
            [0.0, 24.0, 4.0],  # projects to (0, 6)
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        raster = geom.rasterize(vertices, faces, K, pose, 8, 8)
        # At pixel (2,2), screen barycentrics are exactly 1/3 each. The
        # object-space barycentrics are proportional to screen_lambda / z.
        expected_bary = np.array([4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0])
        self.assertEqual(int(raster.face_id[2, 2]), 0)
        np.testing.assert_allclose(raster.barycentric[2, 2], expected_bary, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(raster.depth[2, 2]), 12.0 / 7.0, places=12)
        point = geom.surface_points(vertices, faces, np.array([0]), raster.barycentric[2, 2][None])[0]
        np.testing.assert_allclose(point, expected_bary @ vertices, rtol=0.0, atol=1e-12)

    def test_z_buffer_keeps_nearest_surface_independent_of_face_order(self):
        K = np.eye(3, dtype=np.float64)
        pose = np.eye(4)
        vertices = np.array([
            [0.0, 0.0, 4.0], [24.0, 0.0, 4.0], [0.0, 24.0, 4.0],
            [0.0, 0.0, 2.0], [12.0, 0.0, 2.0], [0.0, 12.0, 2.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        raster = geom.rasterize(vertices, faces, K, pose, 8, 8)
        self.assertEqual(int(raster.face_id[2, 2]), 1)
        self.assertAlmostEqual(float(raster.depth[2, 2]), 2.0, places=12)

    def test_near_plane_crossing_triangle_is_clipped_and_behind_triangle_is_culled(self):
        K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        near = 0.01
        crossing = np.array([[-0.08, -0.08, 1.0], [0.08, -0.08, 1.0], [0.0, 0.0, -1.0]])
        behind = np.array([[-0.08, -0.08, -1.0], [0.08, -0.08, -1.0], [0.0, 0.08, -1.0]])
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        clipped = geom.rasterize(crossing, faces, K, pose, 64, 64, near=near)
        self.assertEqual(clipped.stats["clipped"], 1)
        self.assertTrue(clipped.coverage.any())
        self.assertTrue(np.all(clipped.depth[clipped.coverage] >= near - 1e-12))

        culled = geom.rasterize(behind, faces, K, pose, 64, 64, near=near)
        self.assertEqual(culled.stats["culled_behind"], 1)
        self.assertFalse(culled.coverage.any())


def cylinder_mesh(segments=96, rows=8, radius=0.5, height=1.0):
    vertices = []
    for j in range(rows + 1):
        y = -height / 2 + height * j / rows
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            vertices.append([radius * math.sin(theta), y, -radius * math.cos(theta)])

    def index(j, i):
        return j * segments + (i % segments)

    faces = []
    for j in range(rows):
        for i in range(segments):
            a, b = index(j, i), index(j + 1, i)
            c, d = index(j, i + 1), index(j + 1, i + 1)
            faces.extend(([a, b, c], [c, b, d]))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


class CurvedSurfaceArtworkTests(unittest.TestCase):
    def test_directional_artwork_is_rasterized_on_cylinder_faces_and_observed_mask(self):
        vertices, faces = cylinder_mesh()
        mesh = geom.validate_mesh(vertices, faces)
        self.assertTrue(mesh["consistently_oriented"])
        self.assertTrue(mesh["edge_manifold"])

        width = height = 96
        K = np.array([[80.0, 0.0, 47.5], [0.0, 80.0, 47.5], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[2, 3] = 2.0
        observed = np.ones((height, width), dtype=bool)
        hit = geom.anchor_from_pixel(vertices, faces, K, pose, 47.5, 47.5,
                                     width=width, height=height, observed_mask=observed)
        marker = geom.make_marker(vertices, faces, hit, pose, 0.4, "a" * 64)
        support = np.full(len(faces), 8, dtype=np.int32)
        patch = geom.marker_patch(vertices, faces, support, marker, overlap_grid=64)
        raster = geom.rasterize(vertices, faces, K, pose, width, height)
        art = geom.builtin_artwork("directional", edge=64)

        rendered = geom.render_marker(vertices, faces, support, marker, art, K, pose,
                                      width, height, observed, raster=raster, patch=patch)
        alpha = rendered["rgba"][..., 3] > 0
        self.assertGreater(int(alpha.sum()), 0)
        self.assertEqual(int(alpha.sum()), rendered["stats"]["rendered_pixels"])
        self.assertEqual(rendered["rgba"].dtype, np.uint8)

        # The rendered samples lift back to the actual cylinder triangles;
        # they are not a projected flat quad/silhouette substitute.
        ys, xs = np.nonzero(alpha)
        face_ids = raster.face_id[ys, xs]
        self.assertTrue(np.all(face_ids >= 0))
        points = geom.surface_points(vertices, faces, face_ids, raster.barycentric[ys, xs])
        radial = np.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2)
        self.assertLess(float(np.max(np.abs(radial - 0.5))), 0.002)

        empty_mask = np.zeros((height, width), dtype=bool)
        hidden = geom.render_marker(vertices, faces, support, marker, art, K, pose,
                                    width, height, empty_mask, raster=raster, patch=patch)
        self.assertEqual(int(hidden["rgba"][..., 3].sum()), 0)
        self.assertEqual(hidden["stats"]["rendered_pixels"], 0)

    def test_marker_render_rejects_missing_or_misaligned_observed_mask(self):
        vertices, faces = cylinder_mesh(segments=24, rows=3)
        K = np.array([[50.0, 0.0, 31.5], [0.0, 50.0, 31.5], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[2, 3] = 2.0
        hit = geom.anchor_from_pixel(vertices, faces, K, pose, 31.5, 31.5,
                                     width=64, height=64)
        marker = geom.make_marker(vertices, faces, hit, pose, 0.2, "b" * 64)
        support = np.full(len(faces), 4, dtype=np.int32)
        artwork = geom.builtin_artwork("checkerboard", edge=16)
        with self.assertRaises(geom.GeometryError):
            geom.render_marker(vertices, faces, support, marker, artwork, K, pose,
                               64, 64, None)
        with self.assertRaises(geom.GeometryError):
            geom.render_marker(vertices, faces, support, marker, artwork, K, pose,
                               64, 64, np.ones((63, 64), dtype=bool))


if __name__ == "__main__":
    unittest.main()
