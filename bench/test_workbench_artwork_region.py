"""Exact synthetic parity gates for the opt-in artwork-region raster."""
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from bench import workbench_geometry as geom


class ArtworkRegionTests(unittest.TestCase):
    def scene(self, shift=(0., 0.), scale=1., z=(1., 1., 1.)):
        xy = np.array([[8.25, 8.25], [8.25, 23.75], [23.75, 8.25]])
        xy = (xy - 16) * scale + 16 + np.asarray(shift)
        vertices = np.column_stack((xy * np.asarray(z)[:, None], z))
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        marker = dict(face_id=0, barycentric=[1/3]*3, tangent=[1., 0., 0.],
                      bitangent=[0., 1., 0.], normal=[0., 0., -1.],
                      size_object_units=100.)
        return vertices, faces, np.array([3]), marker

    def compare(self, scene, *, mask=None, supplied=None, **kwargs):
        v, f, support, marker = scene
        art = np.arange(8*8*4, dtype=np.uint8).reshape(8, 8, 4)
        art[::2, :, 3] = 0
        art[1::2, :, 3] = 117
        if mask is None:
            mask = np.ones((32, 32), bool)
            mask[:, 12:15] = False
        args = (v, f, support, marker, art, np.eye(3), np.eye(4), 32, 32, mask)
        original = geom.rasterize
        captures = []
        def capture(*a, **kw):
            result = original(*a, **kw)
            captures.append((a, kw, result))
            return result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with patch.object(geom, "rasterize", side_effect=capture):
                off = geom.render_marker_rgba(*args, raster=supplied, **kwargs)
                on = geom.render_marker_rgba(*args, raster=supplied,
                                             artwork_region_raster=True, **kwargs)
        np.testing.assert_array_equal(on["rgba"], off["rgba"])
        self.assertEqual(on["stats"], off["stats"])
        if len(captures) == 2:
            self.assertEqual(captures[0][1], {})  # exact old default call
            self.assertIs(captures[1][0][1], f)  # all full-mesh occluders
            roi = captures[1][1]["roi"]
            x0, y0, x1, y1 = roi if roi is not None else (0, 0, 31, 31)
            region = np.s_[max(0,y0):max(0,min(32,y1+1)),
                           max(0,x0):max(0,min(32,x1+1))]
            for name in ("depth", "face_id", "barycentric"):
                np.testing.assert_array_equal(getattr(captures[0][2], name)[region],
                                              getattr(captures[1][2], name)[region])
            return roi, on, captures
        return None, on, captures

    def test_default_and_opt_in_exact_with_mask_and_partial_alpha(self):
        roi, result, _ = self.compare(self.scene())
        self.assertEqual(roi, (7, 7, 25, 25))
        self.assertGreater(result["stats"]["rendered_pixels"], 0)
        self.assertGreater(result["stats"]["mask_rejected_pixels"], 0)
        self.assertGreater(result["stats"]["alpha_zero_rejected_pixels"], 0)

    def test_occluders_outside_patch_and_exact_depth_ties(self):
        for depth in (0.5, 1.):
            with self.subTest(depth=depth):
                v, f, support, marker = self.scene()
                # A disconnected unsupported face still occludes the artwork.
                v = np.concatenate((v, v * depth))
                f = np.array([[0,1,2], [3,4,5]], np.int32)
                _, result, captures = self.compare((v, f, np.array([3,0]), marker))
                self.assertEqual(result["stats"]["rendered_pixels"] == 0, depth < 1)
                self.assertIn(1 if depth < 1 else 0, captures[1][2].face_id)

    def test_near_crossing_equal_near_and_behind_fall_back(self):
        for z in ((.0005, 1., 1.), (.001, 1., 1.), (-1., 1., 1.)):
            with self.subTest(z=z):
                roi, _, _ = self.compare(self.scene(z=z))
                self.assertIsNone(roi)

    def test_image_bounds_empty_full_and_subpixel_windows(self):
        for shift, scale in (((-15.,0.),1.), ((15.,15.),1.),
                             ((100.,100.),1.), ((0.,0.),8.),
                             ((-.25,.25),1.)):
            with self.subTest(shift=shift, scale=scale):
                self.compare(self.scene(shift=shift, scale=scale))

    def test_nonfinite_projection_falls_back(self):
        # Keep patch validation finite, but simulate a nonfinite projection
        # result: no unsafe float-to-integer conversion may occur.
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                with patch.object(geom, "project", return_value=(
                        np.array([[bad,0.], [1.,1.], [2.,2.]]), np.ones(3))):
                    roi, _, _ = self.compare(self.scene())
                self.assertIsNone(roi)

    def test_guard_band_and_nonfinite_occluders_preserve_parity(self):
        for bad in (np.nan, np.inf, geom.GUARD_BAND_PX * 2.):
            with self.subTest(value=bad):
                v, f, support, marker = self.scene()
                extra = np.array([[bad, 0., 1.], [0.,1.,1.], [1.,0.,1.]])
                self.compare((np.concatenate((v, extra)),
                              np.array([[0,1,2],[3,4,5]]), np.array([3,0]), marker))

    def test_empty_patch_and_stale_supplied_patch(self):
        v, f, support, marker = self.scene()
        _, result, captures = self.compare((v,f,np.array([0]),marker),
                                           patch={"faces": np.array([0])})
        self.assertEqual(captures, [])
        self.assertEqual(result["stats"]["reason"], "anchor_face_unsupported")
        roi, _, _ = self.compare(self.scene(), patch={"faces": np.array([999])})
        self.assertEqual(roi, (7,7,25,25))

    def test_supplied_raster_used_unchanged(self):
        v, f, _, _ = self.scene()
        supplied = geom.rasterize(v, f, np.eye(3), np.eye(4), 32, 32)
        _, _, captures = self.compare(self.scene(), supplied=supplied)
        self.assertEqual(captures, [])

    def test_flag_requires_boolean(self):
        v, f, support, marker = self.scene()
        for value in (1, None, "yes", np.bool_(True)):
            with self.subTest(value=value), self.assertRaisesRegex(geom.GeometryError, "must be bool"):
                geom.render_marker_rgba(v,f,support,marker,np.zeros((8,8,4),np.uint8),
                    np.eye(3),np.eye(4),32,32,np.ones((32,32)), artwork_region_raster=value)


if __name__ == "__main__":
    unittest.main()
