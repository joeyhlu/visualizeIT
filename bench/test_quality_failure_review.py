import unittest
import numpy as np
from .quality_failure_review import review, rotation_error


class FailureReviewTests(unittest.TestCase):
    def test_slow_orientation_drift_is_found_without_a_large_step(self):
        base = np.eye(4); frames = []; reference = []
        for i in range(20):
            a = np.deg2rad(i*10); p = base.copy()
            p[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            frames.append(dict(frameId=i, cameraFromObject=p.tolist(), pose_state='tracking',
                validation={'median_reprojection_720': .2, 'inliers': 1000}))
            reference.append(dict(frameId=i, reference=base.tolist()))
        r = review({'frames': frames}, {'frames': reference})
        self.assertEqual(r['orientation_disagreement_intervals'][0]['peak_frame'], 18)
        self.assertGreaterEqual(r['orientation_disagreement_intervals'][0]['accepted_frames'], 10)
        self.assertAlmostEqual(rotation_error(frames[17]['cameraFromObject'], frames[18]['cameraFromObject']), 10)

    def test_failed_frames_remain_present_and_split_intervals(self):
        p = np.eye(4); p[:2, :2] *= -1
        frames = [dict(frameId=i, cameraFromObject=None if i == 1 else p.tolist(),
            pose_state='lost' if i == 1 else 'tracking') for i in range(3)]
        r = review({'frames': frames}, {'frames': [dict(frameId=i, reference=np.eye(4).tolist()) for i in range(3)]})
        self.assertEqual(len(r['frames']), 3)
        self.assertEqual(len(r['orientation_disagreement_intervals']), 2)
        self.assertIsNone(r['frames'][1]['reference_rotation_disagreement_degrees'])


if __name__ == '__main__': unittest.main()
