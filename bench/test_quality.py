"""Numerical and failure-path checks; these are not pretrained-model results."""
import unittest
import numpy as np
from .quality_contract import Frame, MaskPrediction, PoseCandidate, SequentialTracker, checked_pose, pose_units, project, validate, canonical_pose
from .quality_evaluate import mask_metrics, prefix_equal, occlusion_recovery, secondary_pose_agreement


K = np.array([[500., 0, 160], [0, 500., 120], [0, 0, 1.]])
POSE = np.eye(4); POSE[2, 3] = 1.
xy = np.array([(x, y) for x in np.linspace(-.12, .12, 6) for y in np.linspace(-.12, .12, 6)])
XYZ = np.c_[xy, np.zeros(len(xy))]
MASK = np.zeros((240, 320), np.uint8); MASK[30:211, 50:271] = 255


def frame(i): return Frame(i, np.zeros((240, 320, 3), np.uint8), K)
def candidate(pose=POSE): return PoseCandidate(pose.copy(), XYZ.copy(), project(XYZ, pose, K)[0], np.ones(len(XYZ)))
def segmentation(i): return MaskPrediction(i, MASK.copy())


class Backend:
    def __init__(self): self.bad = False; self.seeds = []
    def refine(self, f, mask, seed):
        self.seeds.append(seed.copy())
        return None if self.bad else candidate(seed)
    def recover(self, f, mask, top_k=5): return [] if self.bad else [POSE.copy()]


class QualityTests(unittest.TestCase):
    def test_canonical_pose_removes_roundoff_without_repairing_bad_rotations(self):
        noisy = POSE.copy(); noisy[0, 1] += 1e-7; noisy[1, 1] += 1e-7
        corrected = canonical_pose(noisy)
        np.testing.assert_allclose(corrected[:3, :3].T@corrected[:3, :3], np.eye(3), atol=1e-14)
        np.testing.assert_array_equal(corrected[:3, 3], noisy[:3, 3])
        self.assertLess(np.max(np.abs(project(XYZ, corrected, K)[0]-project(XYZ, noisy, K)[0])), .001)
        bad = POSE.copy(); bad[0, 0] = .9
        with self.assertRaises(ValueError): canonical_pose(bad)

    def test_secondary_visibility_cache_preserves_failure_accounting(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        original = dict(cameraCalibration=K.ravel().tolist(), nativeResolution=[320, 240],
                        referenceFrames=[dict(frameId=i, cameraFromObject=POSE.tolist()) for i in range(3)])
        results = dict(frames=[dict(frameId=i, pose_state='lost' if i == 1 else 'tracking',
                                   cameraFromObject=None if i == 1 else POSE.tolist()) for i in range(3)])
        mesh = SimpleNamespace(positions=XYZ)
        truth = (np.arange(len(XYZ)), project(XYZ, POSE, K)[0])
        cache = {}
        with patch('bench.show3d_experiment.evaluation_truth', return_value=truth) as oracle:
            a = secondary_pose_agreement(results, original, mesh, cache)
            b = secondary_pose_agreement(results, original, mesh, cache)
            self.assertEqual(oracle.call_count, 2)
        self.assertEqual(a, b)
        self.assertEqual(a['reference_frames'], 3)
        self.assertEqual(a['accepted_point_samples'], 2*len(XYZ))
        self.assertEqual(a['median_720'], 0.)

    def test_unit_boundary_round_trip_and_projection(self):
        mm = pose_units(POSE, 1000)
        self.assertEqual(mm[2, 3], 1000)
        np.testing.assert_array_equal(pose_units(mm, .001), POSE)
        np.testing.assert_allclose(project(XYZ, POSE, K)[0], project(XYZ*1000, mm, K)[0])

    def test_invalid_pose_rotation_rejected(self):
        bad = POSE.copy(); bad[0, 0] = -1
        with self.assertRaises(ValueError): checked_pose(bad)

    def test_current_correspondences_validate(self):
        ok, reason, stats = validate(candidate(), frame(0), MASK)
        self.assertTrue(ok); self.assertIsNone(reason); self.assertEqual(stats['inliers'], len(XYZ))

    def test_hand_pixels_excluded_before_validation(self):
        self.assertFalse(validate(candidate(), frame(0), np.zeros_like(MASK))[0])

    def test_mask_alone_does_not_establish_pose(self):
        c = candidate(); c.pixels_image += 25
        self.assertFalse(validate(c, frame(0), MASK)[0])

    def test_localized_support_rejected(self):
        c = candidate(); c.points_object_m *= .02; c.pixels_image = project(c.points_object_m, c.pose, K)[0]
        self.assertEqual(validate(c, frame(0), MASK)[1], 'localized_or_ambiguous_support')

    def test_no_stale_pose_after_loss(self):
        backend = Backend(); t = SequentialTracker(backend, POSE)
        self.assertEqual(t.update(frame(0), segmentation(0))['render_state'], 'visible')
        backend.bad = True; result = t.update(frame(1), segmentation(1))
        self.assertIsNone(result['cameraFromObject']); self.assertEqual(result['render_state'], 'suppressed')

    def test_texture_contradiction_suppresses_geometrically_valid_pose(self):
        class BadAppearance(Backend):
            def validate_motion(self, c, frame_id):
                return False, 'contradicts_rendered_texture', {'appearance': {'state': 'measured'}}
        result = SequentialTracker(BadAppearance(), POSE).update(frame(0), segmentation(0))
        self.assertEqual(result['failure_reason'], 'contradicts_rendered_texture')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(result['render_state'], 'suppressed')

    def test_recovery_requires_two_consecutive_validated_poses(self):
        backend = Backend(); t = SequentialTracker(backend)
        one = t.update(frame(0), segmentation(0)); two = t.update(frame(1), segmentation(1))
        self.assertEqual(one['pose_state'], 'recovering'); self.assertIsNone(one['cameraFromObject'])
        self.assertEqual(two['pose_state'], 'tracking')

    def test_missing_mask_resets_recovery_confirmation(self):
        t = SequentialTracker(Backend())
        t.update(frame(0), segmentation(0))
        self.assertIsNone(t.update(frame(1), MaskPrediction(1, None, 'lost'))['cameraFromObject'])
        self.assertEqual(t.update(frame(2), segmentation(2))['pose_state'], 'recovering')

    def test_ambiguous_symmetric_recovery_rejected(self):
        class Ambiguous(Backend):
            def recover(self, f, mask, top_k=5):
                other = POSE.copy(); other[:3, :3] = np.diag([-1., -1., 1.])
                return [POSE.copy(), other]
        result = SequentialTracker(Ambiguous()).update(frame(0), segmentation(0))
        self.assertEqual(result['failure_reason'], 'ambiguous_recovery_hypotheses')

    def test_chronological_frame_and_mask_checks(self):
        t = SequentialTracker(Backend(), POSE); t.update(frame(0), segmentation(0))
        with self.assertRaises(ValueError): t.update(frame(2), segmentation(2))
        t = SequentialTracker(Backend())
        with self.assertRaises(ValueError): t.update(frame(0), segmentation(1))

    def test_symmetric_flip_cannot_pass_on_reprojection_alone(self):
        class Flip(Backend):
            def refine(self, f, mask, seed):
                pose = POSE.copy()
                if f.frame_id: pose[:3, :3] = np.diag([-1., -1., 1.])
                return candidate(pose)
        t = SequentialTracker(Flip(), POSE)
        t.update(frame(0), segmentation(0))
        result = t.update(frame(1), segmentation(1))
        self.assertEqual(result['failure_reason'], 'implausible_pose_jump')
        self.assertIsNone(result['cameraFromObject'])

    def test_short_failure_keeps_private_seed_but_hides_stale_design(self):
        class TemporarilyWeak(Backend):
            def refine(self, f, mask, seed):
                self.seeds.append(seed.copy())
                return None if f.frame_id == 1 else candidate(seed)
            def recover(self, f, mask, top_k=5): return []
        b = TemporarilyWeak(); t = SequentialTracker(b, POSE)
        t.update(frame(0), segmentation(0))
        self.assertEqual(t.update(frame(1), segmentation(1))['render_state'], 'suppressed')
        self.assertEqual(t.update(frame(2), segmentation(2))['pose_state'], 'recovering')
        self.assertEqual(t.update(frame(3), segmentation(3))['pose_state'], 'tracking')
        np.testing.assert_array_equal(b.seeds[-1], POSE)

    def test_mask_accuracy_separate_from_availability(self):
        truth = MASK > 0; pred = np.ones_like(truth); hands = np.zeros_like(truth); hands[:, :50] = True
        m = mask_metrics(pred, truth, hands)
        self.assertLess(m['iou'], .9); self.assertGreater(m['hand_leakage'], .01)

    def test_lost_mask_counts_as_zero_iou_and_boundary_failure(self):
        m = mask_metrics(np.zeros_like(MASK), MASK, np.zeros_like(MASK))
        self.assertEqual(m['iou'], 0); self.assertIsNone(m['boundary_p95_720']); self.assertTrue(m['boundary_failure'])

    def test_exact_mask_zero_boundary_error(self):
        m = mask_metrics(MASK, MASK, np.zeros_like(MASK))
        self.assertEqual(m['iou'], 1); self.assertEqual(m['boundary_p95_720'], 0)

    def test_sequential_wrapper_prefix_invariance(self):
        def run(n):
            t = SequentialTracker(Backend(), POSE)
            results = [t.update(frame(i), segmentation(i)) for i in range(n)]
            for r in results: r['mask_sha256'] = 'synthetic-mask-fixture'
            return {'frames': results}
        self.assertTrue(prefix_equal(run(4), run(8)))
        modified = run(8); modified['frames'][0]['mask_state'] = 'lost'
        self.assertFalse(prefix_equal(run(4), modified))

    def test_forced_occlusion_hides_and_recovers_without_clicks(self):
        t = SequentialTracker(Backend(), POSE); results = []
        for i in range(30):
            s = MaskPrediction(i, None, 'lost', 'blank_current_image') if 5 <= i < 20 else segmentation(i)
            results.append(t.update(frame(i), s))
        result = occlusion_recovery({'frames': results}, 5)
        self.assertTrue(result['passed']); self.assertEqual(result['recovery_source_frames'], 1)


if __name__ == '__main__': unittest.main()
