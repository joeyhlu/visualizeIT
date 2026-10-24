"""Independent camera/lifecycle rejection tests; no neural inference."""
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_geometry as geometry


class GeometryBoundaryTests(unittest.TestCase):
    def test_non_homogeneous_pose_is_not_se3(self):
        # SE(3) in homogeneous coordinates has exact bottom row [0,0,0,1].
        pose = np.eye(4)
        pose[3, 0] = .2
        with self.assertRaises(ValueError):
            geometry.check_pose(pose)

    def test_projection_rejects_intrinsics_terms_it_does_not_model(self):
        # The implemented pinhole projection omits K[1,0] and assumes bottom row [0,0,1].
        for where, value in (((1, 0), .2), ((2, 0), .1), ((2, 2), 2.)):
            K = np.array([[100., 0., 20.], [0., 100., 20.], [0., 0., 1.]])
            K[where] = value
            with self.subTest(where=where), self.assertRaises(ValueError):
                geometry.project(np.array([[0., 0., 2.]]), K, np.eye(4))

    def test_actual_current_static_frames_not_rejected_as_stale_by_pixels_alone(self):
        tracker = geometry.RelativeTracker(min_inliers=4)
        points = np.array([[-.1, -.1, 1.], [.1, -.1, 1.], [-.1, .1, 1.], [.1, .1, 1.]])
        tracker.initialize({"package_id": "static", "status": "available", "frozen": True,
                            "points": points, "descriptors": np.zeros((4, 32), np.uint8)})
        image = np.zeros((48, 48, 3), np.uint8)
        mask = np.ones((48, 48), bool)
        K = np.array([[100., 0., 24.], [0., 100., 24.], [0., 0., 1.]])
        def identity(index):
            return {"frame_id": f"f{index}", "source_frame_index": index,
                    "image_sha256": geometry.array_sha256(image),
                    "mask_frame_id": f"f{index}", "mask_sha256": geometry.mask_sha256(mask)}
        # A physically static scene may encode identical RGB in two different current frames.
        # This fixture has no texture, so insufficient correspondences is expected, not stale input.
        first = tracker.process(image, mask, identity(0), K)
        second = tracker.process(image, mask, identity(1), K)
        self.assertNotIn("stale_image", first.failure_reason or "")
        self.assertNotIn("stale_image", second.failure_reason or "")
        tracker.close()


if __name__ == "__main__":
    unittest.main()
