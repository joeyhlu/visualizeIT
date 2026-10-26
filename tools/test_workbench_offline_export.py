"""Manufactured CPU geometry tests; no footage or learned runtime."""
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from bench import workbench_offline_export as export


def plane_fixture():
    """Pure analytical 32x24 plane, +x tangent/-y bitangent/-z normal.

    Identity R, T=(0,0,1)m and focal length100px projects the .1m square
    corners to (11,7),(21,7),(21,17),(11,17). PNG top is camera top.
    """
    rgb = np.full((24, 32, 3), 40, np.uint8)
    rgba = np.zeros((3, 3, 4), np.uint8)
    rgba[0, 0] = [255, 0, 0, 255]
    rgba[0, 2] = [0, 255, 0, 128]
    rgba[2, 0] = [0, 0, 255, 255]
    rgba[2, 2] = [255, 255, 255, 0]
    vertices = np.array([[-.15, -.15, 0], [.15, -.15, 0], [0, .15, 0]], np.float64)
    faces = np.array([[0, 1, 2]])
    marker = dict(face_id=0, barycentric=[.25, .25, .5], tangent=[1, 0, 0],
                  bitangent=[0, -1, 0], normal=[0, 0, -1], size_object_units=.1)
    pose = np.eye(4)
    pose[2, 3] = 1
    K = np.array([[100., 0, 16], [0, 100, 12], [0, 0, 1]])
    return rgb, rgba, np.ones((24, 32), bool), pose, K, vertices, faces, marker


class PlanarExportTests(unittest.TestCase):
    def test_analytical_projection_alpha_channels_and_orientation(self):
        args = plane_fixture()
        rgb, reason, corners = export.compose_planar(*args)
        self.assertEqual(reason, "rendered_planar_approximation")
        # Pinhole projection: (x,y)*100/1 + (16,12), x,y=+/-.05m.
        np.testing.assert_allclose(corners, [[11, 7], [21, 7], [21, 17], [11, 17]])
        np.testing.assert_array_equal(rgb[7, 11], [255, 0, 0])
        # Straight alpha: round((0,255,0)*128/255 + (40,40,40)*127/255).
        np.testing.assert_array_equal(rgb[7, 21], [20, 148, 20])
        np.testing.assert_array_equal(rgb[17, 11], [0, 0, 255])
        np.testing.assert_array_equal(rgb[17, 21], [40, 40, 40])
        np.testing.assert_array_equal(rgb[0, 0], [40, 40, 40])

    def test_suppression_retains_original_pixels(self):
        for cause in ("behind", "backface", "mask", "invalid", "transparent"):
            args = list(plane_fixture())
            if cause == "behind":
                args[3][2, 3] = -1
            elif cause == "backface":
                args[3][:3, :3] = np.diag([1, -1, -1])
            elif cause == "mask":
                args[2][:] = False
            elif cause == "transparent":
                args[1][:, :, 3] = 0
            else:
                args[-1]["size_object_units"] = 0
            with self.subTest(cause=cause):
                result, reason, _ = export.compose_planar(*args)
                self.assertNotEqual(reason, "rendered_planar_approximation")
                np.testing.assert_array_equal(result, args[0])

    def test_aspect_ratio_and_rotation(self):
        args = list(plane_fixture())
        args[1] = np.full((2, 4, 4), 255, np.uint8)
        _, _, corners = export.compose_planar(*args)
        # A 4:2 PNG uses physical dimensions .1m:.05m at depth1m, focal100px.
        np.testing.assert_allclose(corners, [[11, 9.5], [21, 9.5], [21, 14.5], [11, 14.5]])
        placement = dict(export.DEFAULT_PLACEMENT, rotation_deg=90)
        _, _, turned = export.compose_planar(*args, placement)
        # +90deg rotates t toward b=-camera_y; width points upward in image space.
        np.testing.assert_allclose(turned, [[13.5, 17], [13.5, 7], [18.5, 7], [18.5, 17]])

    def test_pose_integrity_and_bounded_placement(self):
        args = list(plane_fixture())
        args[3][0, 0] = 2
        with self.assertRaises(ValueError):
            export.compose_planar(*args)
        for changes in ({"scale": True}, {"scale": float("nan")}, {"opacity": 2}, {"offset_t_m": .51}, {"recipe": "execute"}):
            with self.subTest(changes=changes), self.assertRaises(export.ar.Refused):
                export.placement_values(dict(export.DEFAULT_PLACEMENT, **changes))

    def test_import_guard_no_torch_models_or_admission_runtime(self):
        script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torchvision', 'gotrack'):
        raise AssertionError('forbidden model runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from bench import workbench_offline_export
from bench import workbench_cli, workbench_neural
from tools import workbench_offline_export
workbench_neural.load_admission = lambda *a, **k: (_ for _ in ()).throw(AssertionError('runtime admission'))
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
