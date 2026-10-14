"""Regression tests for final-support PnP and witnessed flow/fresh fusion.

All geometric truth in the numerical tests is scalar pinhole projection of
manufactured source-metre anchors. The image tests create their own deterministic
textured planes and use actual ORB, BF matching, LK, RANSAC and LM calls.
"""
import unittest
from unittest.mock import patch

import numpy as np

from .prototype_tracking import (
    FRAME_INDEX_CLOCK,
    FrameInputV1,
    PrototypeTrackerV1,
    crop_resize_native_from_input,
    transform_pixels,
)
from .tracker import RigidTracker, _canonicalize_correspondences, reconcile_flow_fresh
from .vision import cv2


K = np.array([[700., 0., 320.], [0., 710., 240.], [0., 0., 1.]])
R = np.eye(3)
T = np.array([.01, -.01, .72])


def _project_scalar(points, rotation=R, translation=T, k=K):
    """Independent oracle: integer-centered CV pinhole projection in metres."""
    rows = []
    for point in np.asarray(points, dtype=np.float64):
        x, y, z = rotation@point+translation
        rows.append([k[0, 0]*x/z+k[0, 2], k[1, 1]*y/z+k[1, 2]])
    return np.asarray(rows, dtype=np.float64)


def _bank():
    x, y, z = np.meshgrid(np.linspace(-.08, .08, 7),
                          np.linspace(-.06, .06, 6),
                          [-.025, .025], indexing='ij')
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _tracker(points, model=None):
    descriptors = np.random.default_rng(23).integers(0, 256, (len(points), 32), dtype=np.uint8)
    return RigidTracker(points, descriptors, points if model is None else model)


def _landmark_error(points, rotation, translation, truth_rotation=R, truth_translation=T):
    actual = _project_scalar(points, rotation, translation)
    expected = _project_scalar(points, truth_rotation, truth_translation)
    return np.linalg.norm(actual-expected, axis=1)


class FinalSupportSolverTests(unittest.TestCase):
    def test_actual_planar_inliers_from_nonplanar_proposal_are_limited(self):
        grid_x, grid_y = np.meshgrid(np.linspace(-.08, .08, 8), np.linspace(-.06, .06, 7))
        planar = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])
        outliers = _bank()[::7].copy()
        outliers[:, 2] += .06
        points = np.concatenate([planar, outliers])
        pixels = np.concatenate([_project_scalar(planar),
                                 _project_scalar(outliers)+np.array([90., -70.])])

        result = _tracker(points, _bank()).solve(pixels, points, K, 1)

        self.assertFalse(isinstance(result, tuple))
        self.assertEqual(result.state, 'limited')
        self.assertIn('inliers', result.reason)
        self.assertIsNone(result.rotation)
        self.assertIsNone(result.translation)
        self.assertEqual(result.statistics['inliers'], 56)
        self.assertLess(result.statistics['acceptedNonplanarityRatio'], .001)

    def test_perfect_nonplanar_control_matches_independent_scalar_oracle(self):
        points = _bank()
        solved = _tracker(points).solve(_project_scalar(points), points, K, 2)

        self.assertIsInstance(solved, tuple)
        result, retained_pixels, retained_points = solved
        self.assertEqual(result.state, 'tracking')
        errors = np.linalg.norm(_project_scalar(retained_points, result.rotation,
                                                result.translation)-retained_pixels, axis=1)
        np.testing.assert_allclose(_project_scalar(points, result.rotation, result.translation),
                                   _project_scalar(points), atol=.001)
        self.assertLess(float(errors.max()), .001)
        self.assertEqual(result.statistics['inliers'], len(retained_points))
        self.assertEqual(result.statistics['correspondences'], len(points))
        self.assertAlmostEqual(result.statistics['inlierRatio'], 1.)
        self.assertLessEqual(result.statistics['maximumReprojectionErrorPixels'], 3.)
        self.assertLessEqual(result.statistics['p95ReprojectionErrorPixels'], 3.)
        self.assertGreaterEqual(result.statistics['acceptedNonplanarityRatio'], .001)

    def test_fixed_six_lm_stress_seeds_have_only_final_three_pixel_support(self):
        points = _bank()
        for seed in [41, 42, 43, 44, 45, 46]:
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                noise = rng.normal(0., 1.05, (len(points), 2))
                noise[:16] += np.array([2.45, .4])
                pixels = _project_scalar(points)+noise
                solved = _tracker(points).solve(pixels, points, K, seed)
                if not isinstance(solved, tuple):
                    self.assertIsNone(solved.rotation)
                    self.assertIsNone(solved.translation)
                    continue
                result, retained_pixels, retained_points = solved
                errors = np.linalg.norm(_project_scalar(retained_points, result.rotation,
                                                        result.translation)-retained_pixels, axis=1)
                self.assertTrue(np.isfinite(errors).all())
                self.assertTrue((errors <= 3.).all())
                self.assertEqual(result.statistics['inliers'], len(retained_points))
                self.assertAlmostEqual(result.statistics['inlierRatio'],
                                       len(retained_points)/len(points))
                self.assertLessEqual(result.statistics['maximumReprojectionErrorPixels'], 3.)
                self.assertGreaterEqual(result.statistics['acceptedNonplanarityRatio'], .001)

    def test_exact_observation_collision_cannot_turn_six_pixels_into_twelve_inliers(self):
        base = _bank()[::14]
        pixels = _project_scalar(base)
        nearby_anchors = base+np.array([0., 0., .001])
        proposals = np.concatenate([base, nearby_anchors])
        observations = np.concatenate([pixels, pixels])
        repeated_pair_pixels = np.concatenate([observations, observations[:1]])
        repeated_pair_points = np.concatenate([proposals, proposals[:1]])

        result = _tracker(proposals).solve(repeated_pair_pixels, repeated_pair_points, K, 3)

        self.assertFalse(isinstance(result, tuple))
        self.assertEqual(result.reason, 'insufficient_correspondences')
        self.assertEqual(result.statistics['correspondences'], 12)
        self.assertEqual(result.statistics['inliers'], 0)
        self.assertEqual(result.statistics['excludedCollidingAnchors'], 12)
        self.assertEqual(result.statistics['duplicatePairsCollapsed'], 1)

    def test_exact_canonicalization_preserves_nearby_distinct_anchors_and_pixels(self):
        anchors = np.array([[0., 0., 0.], [.0002, 0., 0.], [.0002, 0., .0001]])
        pixels = np.array([[10., 10.], [10.0001, 10.], [10.0001, 10.0001]])
        kept_pixels, kept_points, proposed, duplicate_count, ambiguous_count, collision_count = (
            _canonicalize_correspondences(pixels, anchors))

        self.assertEqual(len(kept_points), 3)
        self.assertEqual(proposed, 3)
        self.assertEqual(duplicate_count, 0)
        self.assertEqual(ambiguous_count, 0)
        self.assertEqual(collision_count, 0)
        np.testing.assert_array_equal(kept_points, anchors)
        np.testing.assert_array_equal(kept_pixels, pixels)

    def test_multiple_pixels_for_one_exact_anchor_cannot_inflate_independent_support(self):
        points = _bank()
        pixels = _project_scalar(points)
        ambiguous_indices = np.arange(0, len(points), 7)
        repeated_points = np.concatenate([points, points[ambiguous_indices]])
        repeated_pixels = np.concatenate([pixels, pixels[ambiguous_indices]+np.array([.2, 0.])])

        solved = _tracker(points).solve(repeated_pixels, repeated_points, K, 31)

        self.assertIsInstance(solved, tuple)
        result, retained_pixels, retained_points = solved
        self.assertLessEqual(result.statistics['inliers'], len(points))
        self.assertLessEqual(result.statistics['inlierRatio'], 1.)
        self.assertEqual(result.statistics['proposedAnchors'], len(points))
        self.assertEqual(result.statistics['excludedAmbiguousAnchors'], len(ambiguous_indices))
        self.assertEqual(len(np.unique(retained_points, axis=0)), len(retained_points))
        errors = np.linalg.norm(_project_scalar(retained_points, result.rotation,
                                                result.translation)-retained_pixels, axis=1)
        self.assertTrue((errors <= 3.).all())

    def test_proposal_count_is_validated_and_cannot_reduce_ratio_denominator(self):
        points = _bank()
        pixels = _project_scalar(points)
        tracker = _tracker(points)
        for bad_count in (True, -1, 83, 1.5):
            with self.subTest(proposal_count=bad_count):
                result = tracker.solve(pixels, points, K, 4, proposal_count=bad_count)
                self.assertEqual(result.reason, 'invalid_proposal_count')
                self.assertIsNone(result.rotation)

    def test_malformed_ransac_indices_fail_closed_without_refinement(self):
        points = _bank()
        pixels = _project_scalar(points)
        rvec = np.zeros((3, 1), dtype=np.float64)
        translation = T.reshape(3, 1)
        malformed = np.array([[0], [1], [1], [3]], dtype=np.int32)
        with patch('bench.tracker.cv2.solvePnPRansac',
                   return_value=(True, rvec, translation, malformed)):
            result = _tracker(points).solve(pixels, points, K, 5)
        self.assertEqual(result.reason, 'invalid_pnp_inliers')
        self.assertIsNone(result.rotation)
        self.assertIsNone(result.translation)

    def test_three_refinement_calls_that_keep_changing_support_fail_unstable(self):
        points = _bank()
        translations = [T, T+np.array([-.0015, 0., 0.]),
                        T+np.array([-.003, 0., 0.])]
        projected = [_project_scalar(points, translation=value) for value in translations]
        pixels = projected[0].copy()
        # Three disjoint nine-anchor groups cross the fixed 3 px residual gate
        # on successive mocked LM responses. Every pass retains >=57 of 84
        # distributed anchors, so the test reaches the third-call resource bound.
        pixels[:9] = projected[0][:9]+np.array([4.2, 0.])
        pixels[9:18] = projected[1][9:18]+np.array([4.2, 0.])
        pixels[18:27] = projected[2][18:27]+np.array([4.2, 0.])
        pixels[27:] = .5*(projected[0][27:]+projected[2][27:])
        inliers = np.arange(len(points), dtype=np.int32).reshape(-1, 1)
        responses = [(np.zeros((3, 1), dtype=np.float64), value.reshape(3, 1))
                     for value in translations]
        with patch('bench.tracker.cv2.solvePnPRansac',
                   return_value=(True, np.zeros((3, 1)), T.reshape(3, 1), inliers)), \
             patch('bench.tracker.cv2.solvePnPRefineLM', side_effect=responses) as refine:
            result = _tracker(points).solve(pixels, points, K, 6)
        self.assertEqual(refine.call_count, 3)
        self.assertEqual([len(call.args[0]) for call in refine.call_args_list], [84, 75, 66])
        self.assertEqual(result.reason, 'refinement_support_unstable')
        self.assertEqual(result.statistics['inliers'], 57)
        self.assertGreaterEqual(result.statistics['inlierRatio'], .5)
        self.assertIsNone(result.rotation)
        self.assertIsNone(result.translation)

    def test_actual_ransac_post_lm_support_collapse_clears_legacy_cache(self):
        points = _bank()
        pixels = _project_scalar(points)
        tracker = _tracker(points)
        tracker.previous_gray = np.zeros((480, 640), dtype=np.uint8)
        tracker.previous_pixels = pixels.copy()
        tracker.previous_points = points.copy()
        tracker.lost = False
        failed_translation = (T+np.array([.004, 0., 0.])).reshape(3, 1)

        # RANSAC and projectPoints are real; the mocked LM pose deliberately
        # moves all exact observations beyond 3 px to exercise post-pruning loss.
        with patch.object(tracker, 'flow', return_value=(pixels.copy(), points.copy())), \
             patch.object(tracker, 'match',
                          return_value=(np.empty((0, 2)), np.empty((0, 3)))), \
             patch('bench.tracker.cv2.solvePnPRefineLM',
                   return_value=(np.zeros((3, 1)), failed_translation)) as refine:
            result = tracker.update(np.zeros((480, 640), dtype=np.uint8), K, 7)

        self.assertEqual(refine.call_count, 1)
        self.assertEqual(result.state, 'lost')
        self.assertEqual(result.reason, 'insufficient_inliers')
        self.assertEqual(result.statistics['inliers'], 0)
        self.assertIsNone(result.rotation)
        self.assertIsNone(result.translation)
        self.assertTrue(tracker.lost)
        self.assertIsNone(tracker.previous_gray)
        self.assertIsNone(tracker.previous_pixels)
        self.assertIsNone(tracker.previous_points)


class WitnessedFusionTests(unittest.TestCase):
    def setUp(self):
        self.points = _bank()
        self.truth_pixels = _project_scalar(self.points)

    def _fuse_and_solve(self, flow_pixels, fresh_pixels, *, pixel_to_native=None):
        tracker = _tracker(self.points)
        pixels, points, proposal_count, diagnostics = reconcile_flow_fresh(
            flow_pixels, self.points, fresh_pixels, self.points, K, 7,
            tracker.solve, pixel_to_native=pixel_to_native)
        self.assertIsNone(diagnostics['fusionFailure'])
        solved = tracker.solve(pixels, points, K, 7, proposal_count=proposal_count)
        return solved, pixels, points, proposal_count, diagnostics

    def test_below_threshold_drift_preserves_flow_first_correspondences(self):
        flow = self.truth_pixels.copy()
        offsets = np.tile([[2.2, 0.], [-2.1, 0.], [1.8, 0.]], (10, 1))[:28]
        flow[::3] += offsets
        solved, pixels, points, count, diagnostics = self._fuse_and_solve(flow, self.truth_pixels)

        np.testing.assert_array_equal(points, self.points)
        np.testing.assert_array_equal(pixels, flow)
        self.assertEqual(count, len(self.points))
        self.assertEqual(diagnostics['flowFreshConflicts'], 0)
        self.assertEqual(diagnostics['selectedFresh'], 0)
        self.assertEqual(diagnostics['selectedFlow'], len(self.points))
        self.assertIsInstance(solved, tuple)

    def test_fresh_only_rows_follow_flow_rows_without_reordering_existing_measurements(self):
        flow_count = 56
        tracker = _tracker(self.points)
        pixels, points, proposal_count, diagnostics = reconcile_flow_fresh(
            self.truth_pixels[:flow_count], self.points[:flow_count],
            self.truth_pixels[flow_count:], self.points[flow_count:], K, 9,
            tracker.solve)

        np.testing.assert_array_equal(points, self.points)
        np.testing.assert_array_equal(pixels, self.truth_pixels)
        self.assertEqual(proposal_count, 84)
        self.assertEqual(diagnostics['flowFreshConflicts'], 0)
        self.assertEqual(diagnostics['selectedFlow'], flow_count)
        self.assertEqual(diagnostics['selectedFresh'], 84-flow_count)

    def test_independent_intact_majority_selects_correct_fresh_flow_measurements(self):
        flow = self.truth_pixels.copy()
        displaced = np.arange(0, len(flow), 3)
        offsets = np.random.default_rng(31).choice([-1., 1.], size=(len(displaced), 2))*np.array([9., 7.])
        flow[displaced] += offsets

        solved, _, _, count, diagnostics = self._fuse_and_solve(flow, self.truth_pixels)

        self.assertIsInstance(solved, tuple)
        result, retained_pixels, retained_points = solved
        self.assertEqual(count, 84)
        self.assertEqual(diagnostics['flowFreshConflicts'], 28)
        self.assertEqual(diagnostics['witnessState'], 'tracking')
        self.assertGreaterEqual(diagnostics['witnessSupport'], 56)
        self.assertEqual(diagnostics['selectedFresh'], 28)
        self.assertEqual(diagnostics['excludedUnsupportedAnchors'], 0)
        self.assertLess(float(_landmark_error(self.points, result.rotation,
                                              result.translation).max()), .001)
        fit_errors = np.linalg.norm(_project_scalar(retained_points, result.rotation,
                                                    result.translation)-retained_pixels, axis=1)
        self.assertTrue((fit_errors <= 3.).all())

    def test_independent_intact_majority_rejects_wrong_fresh_measurements(self):
        fresh = self.truth_pixels.copy()
        displaced = np.arange(0, len(fresh), 3)
        fresh[displaced] += np.random.default_rng(32).choice([-1., 1.], size=(len(displaced), 2))*np.array([8., 7.])

        solved, _, _, _, diagnostics = self._fuse_and_solve(self.truth_pixels, fresh)

        self.assertIsInstance(solved, tuple)
        result = solved[0]
        self.assertEqual(diagnostics['flowFreshConflicts'], 28)
        self.assertEqual(diagnostics['witnessState'], 'tracking')
        self.assertEqual(diagnostics['selectedFresh'], 0)
        self.assertEqual(diagnostics['selectedFlow'], 84)
        self.assertLess(float(_landmark_error(self.points, result.rotation,
                                              result.translation).max()), .001)

    def test_all_conflict_global_hypotheses_remain_unresolved_in_both_directions(self):
        wrong_pose = T+np.array([.025, 0., 0.])
        wrong_pixels = _project_scalar(self.points, translation=wrong_pose)
        for flow, fresh in ((wrong_pixels, self.truth_pixels),
                            (self.truth_pixels, wrong_pixels)):
            with self.subTest(flow_truth=np.array_equal(flow, self.truth_pixels)):
                solved, _, _, count, diagnostics = self._fuse_and_solve(flow, fresh)
                self.assertFalse(isinstance(solved, tuple))
                self.assertEqual(count, 84)
                self.assertEqual(diagnostics['flowFreshConflicts'], 84)
                self.assertEqual(diagnostics['witnessSupport'], 0)
                self.assertEqual(diagnostics['selectedFlow'], 0)
                self.assertEqual(diagnostics['selectedFresh'], 0)
                self.assertEqual(diagnostics['excludedUnsupportedAnchors'], 84)
                self.assertIsNone(solved.rotation)
                self.assertIsNone(solved.translation)

    def test_one_displaced_anchor_is_adjudicated_by_the_other_eighty_three(self):
        flow = self.truth_pixels.copy()
        flow[17] += np.array([10., -8.])

        solved, _, _, _, diagnostics = self._fuse_and_solve(flow, self.truth_pixels)

        self.assertIsInstance(solved, tuple)
        self.assertEqual(diagnostics['flowFreshConflicts'], 1)
        self.assertEqual(diagnostics['witnessSupport'], 83)
        self.assertEqual(diagnostics['selectedFresh'], 1)
        self.assertLess(float(_landmark_error(self.points, solved[0].rotation,
                                              solved[0].translation).max()), .001)

    def test_original_collision_is_excluded_before_conflict_selection(self):
        transforms = [np.eye(3),
                      np.array([[2., 0., 0.], [0., 1.5, 0.], [0., 0., 1.]])]
        for native_from_input in transforms:
            with self.subTest(native_from_input=native_from_input.tolist()):
                input_from_native = np.linalg.inv(native_from_input)
                flow_input = transform_pixels(self.truth_pixels, input_from_native)
                fresh_input = flow_input.copy()
                # Anchor 0's unused fresh alternative exactly reuses anchor 1's
                # flow pixel. A separate conflict forces an independent witness.
                fresh_input[0] = flow_input[1]
                fresh_input[2, 0] += 10./native_from_input[0, 0]
                tracker = _tracker(self.points)

                native_pixels, points, count, diagnostics = reconcile_flow_fresh(
                    flow_input, self.points, fresh_input, self.points, K, 8,
                    tracker.solve, pixel_to_native=native_from_input)
                solved = tracker.solve(native_pixels, points, K, 8, proposal_count=count)

                self.assertEqual(count, 84)
                self.assertEqual(diagnostics['flowFreshConflicts'], 1)
                self.assertEqual(diagnostics['excludedCollidingAnchors'], 2)
                self.assertEqual(diagnostics['witnessState'], 'tracking')
                self.assertEqual(diagnostics['witnessSupport'], 81)
                # Scalar truth keeps all flow pixels correct. After excluding
                # the two collided identities, the witness restores anchor 2's
                # truthful flow rather than its deliberately shifted fresh row:
                # 84 - 2 = 82 selected flow observations, and no fresh selection.
                self.assertEqual(diagnostics['selectedFlow'], 82)
                self.assertEqual(diagnostics['selectedFresh'], 0)
                self.assertIsInstance(solved, tuple)
                result, retained_pixels, retained_points = solved
                self.assertEqual(len(retained_points), 82)
                self.assertFalse(any(np.array_equal(point, self.points[0]) or
                                     np.array_equal(point, self.points[1])
                                     for point in retained_points))
                self.assertLess(float(_landmark_error(self.points, result.rotation,
                                                      result.translation).max()), .001)

    def test_below_threshold_alternative_collision_is_not_laundered(self):
        flow = self.truth_pixels.copy()
        fresh = self.truth_pixels.copy()
        fresh[0] += np.array([2., 0.])
        # Both source rows for anchor 1 share the unused alternative of anchor 0.
        flow[1] = fresh[0]
        fresh[1] = fresh[0]
        tracker = _tracker(self.points)

        pixels, points, count, diagnostics = reconcile_flow_fresh(
            flow, self.points, fresh, self.points, K, 10, tracker.solve)
        solved = tracker.solve(pixels, points, K, 10, proposal_count=count)

        self.assertLess(float(np.linalg.norm(flow[0]-fresh[0])), 3.)
        self.assertEqual(count, 84)
        self.assertEqual(diagnostics['flowFreshConflicts'], 0)
        self.assertEqual(diagnostics['excludedCollidingAnchors'], 2)
        self.assertEqual(diagnostics['selectedFlow'], 82)
        self.assertEqual(diagnostics['selectedFresh'], 0)
        self.assertIsInstance(solved, tuple)
        self.assertEqual(len(solved[2]), 82)
        self.assertEqual(solved[0].statistics['correspondences'], 84)
        self.assertEqual(solved[0].statistics['inliers'], 82)
        self.assertFalse(any(np.array_equal(point, self.points[0]) or
                             np.array_equal(point, self.points[1])
                             for point in solved[2]))
        self.assertLess(float(_landmark_error(self.points, solved[0].rotation,
                                              solved[0].translation).max()), .001)

    def test_anisotropic_resize_conflict_is_measured_in_native_pixels_both_directions(self):
        native_from_input = np.array([[2., 0., 0.], [0., 1.5, 0.], [0., 0., 1.]])
        input_from_native = np.linalg.inv(native_from_input)
        true_input = transform_pixels(self.truth_pixels, input_from_native)
        for corrupt_flow in (False, True):
            with self.subTest(corrupt_flow=corrupt_flow):
                flow_input = true_input.copy()
                fresh_input = true_input.copy()
                if corrupt_flow:
                    flow_input[23, 0] += 2.
                else:
                    fresh_input[23, 0] += 2.
                tracker = _tracker(self.points)
                native_pixels, points, count, diagnostics = reconcile_flow_fresh(
                    flow_input, self.points, fresh_input, self.points, K, 8,
                    tracker.solve, pixel_to_native=native_from_input)
                solved = tracker.solve(native_pixels, points, K, 8, proposal_count=count)
                native_flow = transform_pixels(flow_input, native_from_input)
                native_fresh = transform_pixels(fresh_input, native_from_input)

                self.assertEqual(count, 84)
                self.assertAlmostEqual(float(np.linalg.norm(native_flow[23]-native_fresh[23])), 4.)
                self.assertEqual(diagnostics['flowFreshConflicts'], 1)
                self.assertEqual(diagnostics['witnessSupport'], 83)
                self.assertEqual(diagnostics['selectedFresh'], 1 if corrupt_flow else 0)
                self.assertEqual(diagnostics['selectedFlow'], 83 if corrupt_flow else 84)
                self.assertIsInstance(solved, tuple)
                result, retained_pixels, retained_points = solved
                pixel_by_anchor = {tuple(point): pixel
                                   for point, pixel in zip(retained_points, retained_pixels)}
                self.assertEqual(len(pixel_by_anchor), 84)
                for index, point in enumerate(self.points):
                    np.testing.assert_allclose(pixel_by_anchor[tuple(point)],
                                               self.truth_pixels[index], atol=1e-10)
                self.assertLess(float(_landmark_error(self.points, result.rotation,
                                                      result.translation).max()), .001)


class _TexturedFixture:
    """Three disjoint random-texture planes with scalar-derived source anchors."""

    def __init__(self, texture_seed=71):
        self.k = K.copy()
        self.native_size = (640, 480)
        self.rotation = np.eye(3)
        self.translation = np.array([0., 0., .72])
        rng = np.random.default_rng(texture_seed)
        self.patches = []
        for center_x, depth in [(-.22, -.025), (0., .025), (.22, -.025)]:
            texture = rng.integers(0, 256, (192, 192), dtype=np.uint8)
            self.patches.append((center_x, depth, texture))
        self.seed_rgb = self.render(self.rotation, self.translation)
        orb = cv2.ORB_create(nfeatures=2000)
        gray = cv2.cvtColor(self.seed_rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        selected = []
        descriptor_rows = []
        for center_x, depth, _ in self.patches:
            candidates = []
            for index, keypoint in enumerate(keypoints):
                x, y = keypoint.pt
                point = self._intersect_plane(x, y, depth)
                if (abs(point[0]-center_x) <= .07 and abs(point[1]) <= .065):
                    candidates.append((keypoint.response, index, point))
            candidates.sort(reverse=True, key=lambda item: item[0])
            for _, index, point in candidates[:28]:
                selected.append((keypoints[index].pt, point))
                descriptor_rows.append(descriptors[index])
        self.seed_pixels = np.asarray([row[0] for row in selected], dtype=np.float64)
        self.points = np.asarray([row[1] for row in selected], dtype=np.float64)
        self.descriptors = np.asarray(descriptor_rows, dtype=np.uint8)
        self.model = self.points.copy()

    def _intersect_plane(self, u, v, plane_z):
        camera_origin_source = -self.rotation.T@self.translation
        ray_camera = np.linalg.inv(self.k)@np.array([u, v, 1.])
        ray_source = self.rotation.T@ray_camera
        distance = (plane_z-camera_origin_source[2])/ray_source[2]
        return camera_origin_source+distance*ray_source

    @staticmethod
    def _project_quad(center_x, depth, rotation, translation, k):
        source = np.array([[center_x-.07, -.065, depth],
                           [center_x+.07, -.065, depth],
                           [center_x+.07, .065, depth],
                           [center_x-.07, .065, depth]])
        return _project_scalar(source, rotation, translation, k)

    def render(self, rotation=None, translation=None, textures=None):
        rotation = self.rotation if rotation is None else rotation
        translation = self.translation if translation is None else translation
        textures = [patch[2] for patch in self.patches] if textures is None else textures
        image = np.zeros((480, 640), dtype=np.uint8)
        for index, (center_x, depth, _) in enumerate(self.patches):
            source = np.array([[0., 0.], [191., 0.], [191., 191.], [0., 191.]], dtype=np.float32)
            destination = self._project_quad(center_x, depth, rotation, translation, self.k).astype(np.float32)
            homography = cv2.getPerspectiveTransform(source, destination)
            warped = cv2.warpPerspective(textures[index], homography, (640, 480),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpPerspective(np.full((192, 192), 255, np.uint8), homography,
                                       (640, 480), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT)
            image[mask > 0] = warped[mask > 0]
        return np.repeat(image[:, :, None], 3, axis=2)

    def frame(self, frame_id, rgb=None, *, rotation=None, translation=None,
              session='repair-session', object_id=15, reference='sha256:repair-bank',
              native_from_input=None):
        return FrameInputV1(session, object_id, reference, frame_id, frame_id, None,
                            FRAME_INDEX_CLOCK,
                            self.render(rotation, translation) if rgb is None else rgb,
                            self.k, self.native_size, native_from_input)


class ActualImagePathTests(unittest.TestCase):
    def _seeded(self, fixture):
        seed = fixture.frame(0)
        return PrototypeTrackerV1(
            fixture.points, fixture.descriptors, fixture.model,
            session_id='repair-session', object_id=15,
            reference_id='sha256:repair-bank', onboarding_seed=seed,
            seed_pixels=fixture.seed_pixels, seed_points=fixture.points)

    def _unseeded(self, fixture):
        return PrototypeTrackerV1(
            fixture.points, fixture.descriptors, fixture.model,
            session_id='repair-session', object_id=15,
            reference_id='sha256:repair-bank')

    def _assert_truth_limits(self, fixture, output, truth_rotation=None, truth_translation=None):
        self.assertEqual(output.state, 'tracking')
        self.assertIsNotNone(output.camera_from_object)
        rotation = output.camera_from_object[:3, :3]
        translation = output.camera_from_object[:3, 3]
        errors = _landmark_error(fixture.points, rotation, translation,
                                 fixture.rotation if truth_rotation is None else truth_rotation,
                                 fixture.translation if truth_translation is None else truth_translation)
        self.assertLessEqual(float(np.median(errors)), 1.)
        self.assertLessEqual(float(np.percentile(errors, 95)), 3.)

    def test_seeded_and_descriptor_start_paths_survive_blanks_and_real_lk_return(self):
        fixture = _TexturedFixture()
        self.assertGreaterEqual(len(fixture.points), 60)
        self.assertEqual(len(np.unique(fixture.points, axis=0)), len(fixture.points))
        self.assertGreaterEqual(np.unique(fixture.points[:, 2]).size, 2)

        seeded = self._seeded(fixture)
        initial = seeded.update(fixture.frame(1))
        self._assert_truth_limits(fixture, initial)
        seeded_outputs = [initial]
        small_rotation = cv2.Rodrigues(np.array([.002, -.003, .001]))[0]
        small_translation = np.array([.001, -.001, .72])
        for frame_id in range(2, 6):
            output = seeded.update(fixture.frame(frame_id, rotation=small_rotation,
                                                  translation=small_translation))
            self._assert_truth_limits(fixture, output, small_rotation, small_translation)
            seeded_outputs.append(output)
        blank_rgb = np.zeros_like(fixture.seed_rgb)
        blank_outputs = [seeded.update(fixture.frame(frame_id, blank_rgb))
                         for frame_id in range(6, 11)]
        self.assertTrue(all(output.state != 'tracking' and output.camera_from_object is None
                            for output in blank_outputs))
        returned = [seeded.update(fixture.frame(frame_id)) for frame_id in range(11, 15)]
        self.assertEqual(returned[0].state, 'recovering')
        self.assertIsNone(returned[0].camera_from_object)
        self.assertGreaterEqual(returned[1].statistics.get('temporalLinks', 0), 12)
        for output in returned[1:]:
            self._assert_truth_limits(fixture, output)
        self.assertEqual(len(seeded_outputs)+len(blank_outputs)+len(returned), 14)
        self.assertGreaterEqual(sum(output.state == 'tracking' for output in returned[1:]) /
                                len(returned[1:]), .9)

        descriptor_start = self._unseeded(fixture)
        start_blanks = [descriptor_start.update(fixture.frame(frame_id, blank_rgb))
                        for frame_id in range(20, 25)]
        self.assertTrue(all(output.state != 'tracking' and output.camera_from_object is None
                            for output in start_blanks))
        fresh_returns = [descriptor_start.update(fixture.frame(frame_id))
                         for frame_id in range(25, 29)]
        self.assertEqual(fresh_returns[0].state, 'recovering')
        self.assertIsNone(fresh_returns[0].camera_from_object)
        self.assertGreaterEqual(fresh_returns[1].statistics.get('temporalLinks', 0), 12)
        for output in fresh_returns[1:]:
            self._assert_truth_limits(fixture, output)
        self.assertEqual(len(start_blanks)+len(fresh_returns), 9)
        self.assertGreaterEqual(sum(output.state == 'tracking' for output in fresh_returns[1:]) /
                                len(fresh_returns[1:]), .9)

    def test_actual_orb_lk_mixed_conflict_uses_intact_image_witness(self):
        fixture = _TexturedFixture(texture_seed=73)
        tracker = self._seeded(fixture)
        for frame_id in range(1, 6):
            output = tracker.update(fixture.frame(frame_id))
            self._assert_truth_limits(fixture, output)

        # Predeclared LK initialization fault on one quarter of cached locations.
        # Pixels and source anchors stay paired as they enter actual OpenCV LK.
        previous_pixels = tracker.rigid.previous_pixels.copy()
        minority = np.arange(0, len(previous_pixels), 4)
        previous_pixels[minority] += np.array([12., -9.])
        tracker.rigid.previous_pixels = previous_pixels
        repaired = tracker.update(fixture.frame(6))

        if repaired.statistics.get('flowFreshConflicts', 0) == 0:
            self.skipTest('Actual LK/ORB stimulus produced no >3-native-pixel conflict; B remains unproven.')
        self._assert_truth_limits(fixture, repaired)
        self.assertGreater(repaired.statistics['flowFreshConflicts'], 0)
        self.assertGreaterEqual(repaired.statistics['witnessSupport'], 12)
        self.assertGreater(repaired.statistics['selectedFresh'], 0)

    def test_wrong_texture_and_identity_mismatch_never_publish_a_pose(self):
        fixture = _TexturedFixture(texture_seed=75)
        unrelated = _TexturedFixture(texture_seed=76)
        tracker = self._unseeded(fixture)
        unrelated_outputs = [tracker.update(unrelated.frame(frame_id)) for frame_id in (1, 2, 3)]
        self.assertTrue(all(output.state != 'tracking' and output.camera_from_object is None
                            for output in unrelated_outputs))

        seeded = self._seeded(fixture)
        rejected = seeded.update(fixture.frame(1, session='foreign-session'))
        self.assertEqual(rejected.state, 'lost')
        self.assertEqual(rejected.reason, 'identity_mismatch')
        self.assertIsNone(rejected.camera_from_object)

        crop_tracker = self._seeded(fixture)
        first = crop_tracker.update(fixture.frame(1))
        self._assert_truth_limits(fixture, first)
        resized = cv2.resize(fixture.seed_rgb, (320, 240), interpolation=cv2.INTER_LINEAR)
        native_from_input = crop_resize_native_from_input(
            (640, 480), (0., 0., 640., 480.), (320, 240))
        changed = crop_tracker.update(fixture.frame(
            2, resized, native_from_input=native_from_input))
        self.assertNotEqual(changed.state, 'tracking')
        self.assertIsNone(changed.camera_from_object)

    def test_legacy_update_uses_the_same_actual_flow_fresh_reconciler(self):
        fixture = _TexturedFixture(texture_seed=77)
        gray = cv2.cvtColor(fixture.seed_rgb, cv2.COLOR_RGB2GRAY)
        legacy = RigidTracker(fixture.points, fixture.descriptors, fixture.model,
                              previous_gray=gray, previous_pixels=fixture.seed_pixels.copy(),
                              previous_points=fixture.points.copy())
        legacy.lost = False

        result = legacy.update(fixture.seed_rgb.copy(), fixture.k, 1)

        self.assertEqual(result.state, 'tracking')
        self.assertIsNotNone(result.rotation)
        self.assertIn('flowFreshConflicts', result.statistics)
        errors = _landmark_error(fixture.points, result.rotation, result.translation,
                                 fixture.rotation, fixture.translation)
        self.assertLessEqual(float(np.median(errors)), 1.)
        self.assertLessEqual(float(np.percentile(errors, 95)), 3.)


if __name__ == '__main__':
    unittest.main()
