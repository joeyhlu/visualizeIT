import tempfile
import unittest
from pathlib import Path
import numpy as np
from .obj_model import read_obj, obj_index
from .scan_mapping import mapping_quality, seam_quality
from .model import Camera, Mesh, unit
from .renderer import render


class ScanMappingTests(unittest.TestCase):
    def test_obj_reflection_preserves_outward_normal_and_negative_indices(self):
        data = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nf -3/1/1 -2/2/1 -1/3/1\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'triangle.obj'
            path.write_text(data)
            scan = read_obj(path)
        p = scan.positions[scan.triangles[0]]
        np.testing.assert_allclose(np.cross(p[1]-p[0], p[2]-p[0]), [0, 1, 0])
        np.testing.assert_allclose(scan.original.normals, [[0, 1, 0]]*3)
        self.assertEqual(scan.repairedNormalVertices, 0)

    def test_obj_invalid_normal_repaired_without_changing_vertices(self):
        data = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn nan nan nan\nf 1/1/1 2/2/1 3/3/1\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'triangle.obj'
            path.write_text(data)
            scan = read_obj(path)
        self.assertEqual(scan.repairedNormalVertices, 3)
        np.testing.assert_allclose(scan.positions, [[-.5, 0, -.5], [.5, 0, -.5], [-.5, 0, .5]])
        np.testing.assert_allclose(scan.original.normals, [[0, 1, 0]]*3)

    def test_obj_index_zero_and_outside_range_rejected(self):
        for value in ('0', '4', '-4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                obj_index(value, 3)

    def test_metric_mapping_detects_known_anisotropic_stretch(self):
        positions = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        perfect = np.array([[0, 0], [2, 0], [0, 1]])
        quality, _, trusted = mapping_quality(positions, faces, perfect)
        self.assertAlmostEqual(quality['areaWeightedP95ScaleError'], 0)
        self.assertTrue(trusted[0])
        quality, _, trusted = mapping_quality(positions, faces, perfect*np.array([.5, 1]))
        self.assertAlmostEqual(quality['areaWeightedP95ScaleError'], 1)
        self.assertAlmostEqual(quality['areaWeightedP95Anisotropy'], 2)
        self.assertFalse(trusted[0])

    def test_invalid_geometry_and_collapsed_uv_are_reported_separately(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        uv = np.array([[0, 0], [1, 0], [2, 0], [2, 1]])
        quality, _, trusted = mapping_quality(positions, faces, uv)
        self.assertEqual(quality['degenerateGeometryTriangles'], 1)
        self.assertEqual(quality['collapsedUVTriangles'], 1)
        self.assertAlmostEqual(quality['invalidSurfaceAreaFraction'], 1)
        self.assertIsNone(quality['areaWeightedP95ScaleError'])
        self.assertFalse(trusted.any())

    def test_seam_measurement_matches_edge_coordinates_when_winding_differs(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        corners = p[faces, :2].astype(float)
        charts = np.array([[0]*3, [1]*3])
        quality = seam_quality(p, faces, corners, charts, 1)
        self.assertEqual(quality['chartSeamEdges'], 1)
        self.assertEqual(quality['boundaryEdges'], 4)
        self.assertAlmostEqual(quality['lengthWeightedSeamPhaseMismatch'], 0)
        corners[1, :, 0] += .25
        quality = seam_quality(p, faces, corners, charts, 1)
        self.assertAlmostEqual(quality['lengthWeightedSeamPhaseMismatch'], .25)

    def test_photographic_texture_uv_origin_and_interpolation(self):
        # Constant UV faces sample known image corners/centre, independently of raster interpolation.
        camera = Camera.look_at([0, 0, 0], [0, 0, 1], width=48, height=48, focal=40)
        texture = np.array([[[200, 0, 0], [0, 200, 0]], [[0, 0, 200], [200, 200, 200]]], dtype=np.uint8)
        light = unit([-.5, 1, -.7])
        for uv, color in (([0, 1], [200, 0, 0]), ([0, 0], [0, 0, 200]), ([.5, .5], [100, 100, 100]), ([-1, 2], [200, 0, 0])):
            with self.subTest(uv=uv):
                mesh = Mesh('texture test', [[-.3, -.3, 1], [.3, -.3, 1], [-.3, .3, 1]],
                            [light]*3, [uv]*3, [[0, 2, 1]])
                frame = render(mesh, camera, texture_image=texture)
                np.testing.assert_allclose(frame.color[26, 18], color, atol=1)


if __name__ == '__main__':
    unittest.main()
