"""Independent denominator/isolation and camera-boundary regression tests."""
import unittest
import numpy as np
from .prototype_benchmark import (cv_render_camera, score_trace, validate_test_records,
    validate_pose_trace, validate_evaluator_binding, geometry_binding, recovery_metrics)


class BenchmarkTests(unittest.TestCase):
    def test_camera_reflection_and_pixel_center(self):
        angle = .63
        r = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        t = np.array([.06, -.07, .8])
        k = np.array([[503., 0, 211.3], [0, 417., 167.1], [0, 0, 1]])
        transform = np.eye(4)
        transform[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
        transform[:3, 3] = [.02, .15, -.04]
        points = np.array([[.03, .07, -.02], [-.1, .05, .12], [.1, -.06, .07]])
        camera = cv_render_camera(r, t, {'cam_K': k.ravel()}, transform, 640, 480)
        world = points @ transform[:3, :3].T + transform[:3, 3]
        xyz = points @ r.T + t
        expected = xyz[:, :2] / xyz[:, 2, None] @ k[:2, :2].T + k[:2, 2]
        projected = camera.project_camera(camera.camera_points(world)) - .5
        np.testing.assert_allclose(projected, expected, atol=1e-10)
        for p, pixel in zip(world, expected):
            ray = camera.rays(pixel + .5)
            physical = p - camera.eye
            np.testing.assert_allclose(ray / np.linalg.norm(ray),
                                       physical / np.linalg.norm(physical), atol=1e-10)

    def test_failures_without_visible_landmarks_stay_in_denominator(self):
        records = [dict(frameId=i, evaluationInfo={'visib_fract': 1.}) for i in range(110)]
        frames = [dict(frameId=i, state='tracking' if i < 10 else 'lost',
                       failureReason=None if i < 10 else 'no_evidence') for i in range(110)]
        score = score_trace(records, frames, {i: [0.] for i in range(10)})
        self.assertEqual(score['scheduledFrames'], 110)
        self.assertAlmostEqual(score['allFrameAvailability'], 10 / 110)
        self.assertFalse(score['attachmentGatePassed'])

    def test_no_poses_or_masks_in_test_input(self):
        records = [dict(frameId=i, rgb={}, calibration={}) for i in range(110)]
        validate_test_records(records)
        for name in ('evaluationPose', 'evaluationMask', 'depth', 'initialPose'):
            contaminated = [dict(r) for r in records]
            contaminated[23][name] = {}
            with self.assertRaises(ValueError):
                validate_test_records(contaminated)

    def test_trace_join_requires_every_frame(self):
        with self.assertRaises(ValueError):
            score_trace([{'frameId': 1}], [], {})

    def test_improper_pose_and_stale_pose_rejected(self):
        reflected = np.eye(4); reflected[0, 0] = -1
        for state, pose in [('tracking', reflected.tolist()), ('lost', np.eye(4).tolist())]:
            with self.assertRaises(ValueError):
                validate_pose_trace([dict(state=state, cvCameraFromSourceObject=pose)])
        validate_pose_trace([dict(state='lost', cvCameraFromSourceObject=None)])

    def test_evaluator_geometry_cannot_be_swapped(self):
        obj = dict(objectId=1, name='one', model={'sha256': 'x'},
                   frames=[dict(frameId=12, calibration={'cam_K': [1]}, rgb={'sha256': 'y'})])
        seal = dict(benchmarkId='b', selectionDigest='s', geometryBindings=[geometry_binding(obj)])
        evaluator = dict(benchmarkId='b', selectionDigest='s', objects=[obj])
        validate_evaluator_binding(evaluator, seal)
        evaluator['objects'][0]['frames'][0]['calibration'] = {'cam_K': [2]}
        with self.assertRaises(ValueError):
            validate_evaluator_binding(evaluator, seal)

    def test_wrong_first_reacquisition_cannot_hide_in_whole_trial_p95(self):
        records = [dict(frameId=i, evaluationInfo={'visib_fract': 1.}) for i in range(110)]
        frames = [dict(frameId=i, state='tracking') for i in range(110)]
        per_frame = [dict(frameId=i, pairedPoints=3, medianErrorPixels=1., p95ErrorPixels=2.)
                     for i in range(110)]
        per_frame[55]['p95ErrorPixels'] = 100.
        metrics = recovery_metrics(records, frames, per_frame)
        self.assertTrue(metrics['wrongOrUnscorableFirstReacquisition'])
        self.assertFalse(metrics['recoveryGatePassed'])
        self.assertEqual(metrics['recoveryObservableFrames'], 3)

    def test_no_loss_during_interruption_cannot_count_as_recovery(self):
        records = [dict(frameId=i, evaluationInfo={'visib_fract': 1.}) for i in range(110)]
        frames = [dict(frameId=i, state='tracking') for i in range(110)]
        per_frame = [dict(frameId=i, pairedPoints=3, medianErrorPixels=1., p95ErrorPixels=2.)
                     for i in range(110)]
        metrics = recovery_metrics(records, frames, per_frame)
        self.assertFalse(metrics['interruptionObserved'])
        self.assertFalse(metrics['recoveryGatePassed'])


if __name__ == '__main__':
    unittest.main()
