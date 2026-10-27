"""Independent pinhole/texel/blend controls for saved metric mesh exports."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_geometry as geom
from bench import workbench_offline_export as exporter
from tools.test_workbench_offline_export_integration import make_fixture, export_args, cv2


def selected_png():
    rgba = np.zeros((16, 32, 4), np.uint8)
    rgba[:8, :16] = [240, 20, 30, 255]
    rgba[:8, 16:] = [10, 220, 40, 128]
    rgba[8:, :16] = [20, 40, 230, 255]
    rgba[8:, 16:] = [220, 220, 20, 0]
    return rgba


def plane_reference(rgb, rgba, mask, marker, K, pose, placement):
    """Intersect camera rays with z=0; evaluate declared floor texels directly.

    This oracle does not call rasterize, surface_points, rendering, texture
    preparation or mesh composition. Manufactured artwork lies inside the plane.
    """
    height, width = rgb.shape[:2]
    ys, xs = np.mgrid[:height, :width]
    rays = np.stack((xs, ys, np.ones_like(xs)), axis=-1) @ np.linalg.inv(K).T
    R, translation = pose[:3, :3], pose[:3, 3]
    origin = -R.T @ translation
    directions = rays @ R
    distance = -origin[2] / directions[..., 2]
    points = origin + distance[..., None] * directions
    # Fixtures use the origin as their independently specified surface anchor.
    theta = math.radians(placement['rotation_deg'])
    s0 = points @ np.array(marker['tangent'])
    r0 = points @ np.array(marker['bitangent'])
    s = math.cos(theta)*s0 + math.sin(theta)*r0
    r = -math.sin(theta)*s0 + math.cos(theta)*r0
    h, w = rgba.shape[:2]
    edge = max(8, h, w)
    extent = marker['size_object_units'] * placement['scale'] * edge/max(h, w)
    column = np.clip(np.floor((s/extent+.5)*edge).astype(int), 0, edge-1) - (edge-w)//2
    row = np.clip(np.floor((.5-r/extent)*edge).astype(int), 0, edge-1) - (edge-h)//2
    valid = (distance > 0) & mask & (np.abs(s) <= extent/2) & (np.abs(r) <= extent/2)
    valid &= (column >= 0) & (column < w) & (row >= 0) & (row < h)
    sampled = rgba[np.clip(row, 0, h-1), np.clip(column, 0, w-1)]
    alpha = sampled[..., 3].astype(float)/255 * placement['opacity'] * valid
    # Straight alpha and round-half-up are the specified RGB output convention.
    return np.floor(sampled[..., :3]*alpha[..., None] + rgb*(1-alpha[..., None])+.5).astype(np.uint8)


class IndependentMeshExportChecks(unittest.TestCase):
    def test_plane_pixels_match_ray_texel_oracle_with_rotation_scale_opacity(self):
        vertices = np.array([[-.2, -.2, 0.], [-.2, .2, 0.], [.2, -.2, 0.], [.2, .2, 0.]])
        faces = np.array([[0, 1, 2], [2, 1, 3]])
        support = np.array([3, 3])
        marker = dict(face_id=0, barycentric=[0., .5, .5], tangent=[1., 0., 0.],
                      bitangent=[0., -1., 0.], normal=[0., 0., -1.],
                      size_object_units=.16, artwork_sha256='a'*64)
        original_marker = copy.deepcopy(marker)
        K = np.array([[200., 0., 47.5], [0., 200., 47.5], [0., 0., 1.]])
        pose = np.eye(4); pose[2, 3] = 1.
        rgb = np.full((96, 96, 3), [7, 13, 19], dtype=np.uint8)
        mask = np.ones((96, 96), bool); mask[:, :40] = False
        art = selected_png()
        for rotation, scale, opacity in ((0, 1., 1.), (90, 1., .4), (0, .7, .4)):
            with self.subTest(rotation=rotation, scale=scale, opacity=opacity):
                placement = dict(exporter.DEFAULT_PLACEMENT, rotation_deg=rotation,
                                 scale=scale, opacity=opacity)
                actual, reason, stats = exporter.compose_mesh(rgb, art, mask, pose, K,
                    vertices, faces, support, marker, placement, max_chart_stretch=1.25)
                expected = plane_reference(rgb, art, mask, marker, K, pose, placement)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual[~mask], rgb[~mask])
                self.assertEqual(reason, 'rendered_mesh_artwork')
                self.assertGreater(stats['composited_pixels'], 0)
        self.assertEqual(marker, original_marker)

    def test_decoded_mesh_video_preserves_failed_frames_and_matches_first_plane(self):
        cache = Path(__file__).resolve().parents[1]/'.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix='mesh-independent-') as folder:
            f = make_fixture(Path(folder))
            original = (f['workspace'].runs_dir/f['run_id']/'neural-artwork.json').read_bytes()
            manifest = exporter.export_saved_window(f['workspace'], **export_args(f),
                surface_mode='mesh', max_chart_stretch=1.25)
            self.assertEqual(manifest['frame_accounting']['scheduled'], 6)
            self.assertEqual(manifest['frame_accounting']['rendered'], 2)
            self.assertFalse(manifest['new_neural_inference'])
            capture = cv2.VideoCapture(str(f['output_dir']/'playback.avi'))
            try:
                for i, record in enumerate(manifest['rows']):
                    ok, bgr = capture.read(); self.assertTrue(ok)
                    rgb = np.ascontiguousarray(bgr[..., ::-1])
                    fid = record['original']['frame_id']
                    if i == 0:
                        saved = wc.loads_strict(original)
                        K = np.array(saved['effective_K']).reshape(3, 3)
                        expected = plane_reference(f['images'][fid], selected_png(), f['masks'][fid],
                            saved['marker'], K, np.array(record['original']['pose']),
                            exporter.DEFAULT_PLACEMENT)
                        np.testing.assert_array_equal(rgb, expected)
                    if record['original']['pose_state'] in ('lost', 'recovering', 'unavailable'):
                        np.testing.assert_array_equal(rgb, f['images'][fid])
                    np.testing.assert_array_equal(rgb[~f['masks'][fid]], f['images'][fid][~f['masks'][fid]])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            self.assertEqual(original, (f['workspace'].runs_dir/f['run_id']/'neural-artwork.json').read_bytes())

    def test_tiny_rectangle_keeps_declared_physical_size(self):
        marker = dict(face_id=0, barycentric=[0., .5, .5], tangent=[1., 0., 0.],
                      bitangent=[0., -1., 0.], normal=[0., 0., -1.],
                      size_object_units=.2, artwork_sha256='a'*64)
        for shape in ((2, 4), (4, 2), (16, 32)):
            rgba = np.full((*shape, 4), 255, dtype=np.uint8)
            texture, derived = exporter.prepare_mesh_artwork(rgba, marker)
            # Exact cell geometry: selected longest edge / padded edge times
            # physical canvas extent must equal the requested .2 metre extent.
            self.assertAlmostEqual(max(shape)/texture.shape[0]*derived['size_object_units'], .2)
            self.assertEqual(derived['barycentric'], marker['barycentric'])


if __name__ == '__main__':
    unittest.main()
