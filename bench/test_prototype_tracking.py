"""Synthetic geometry and recovery contract tests for PrototypeTrackerV1."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .model import Camera, Mesh
from .prototype_tracking import (
    FRAME_INDEX_CLOCK,
    FrameInputV1,
    PHYSICAL_CLOCK,
    PrototypeTrackerV1,
    camera_from_source_to_bench,
    crop_resize_input_from_native,
    crop_resize_native_from_input,
    cv_intrinsics_to_renderer,
    is_proper_rotation,
    project_points_cv,
    raster_centers_to_cv,
    triangles_for_basis,
    transform_pixels,
)
from .renderer import ray_triangle, render


def _manual_project(points, rotation, translation, k):
    """Independent scalar pinhole projection in CV X-right/Y-down/Z-forward axes."""
    result = []
    for point in points:
        x, y, z = rotation@point+translation
        result.append([k[0, 0]*x/z+k[0, 2], k[1, 1]*y/z+k[1, 2]])
    return np.asarray(result)


def _point_bank():
    x, y, z = np.meshgrid(np.linspace(-.05, .05, 5),
                          np.linspace(-.04, .04, 4),
                          np.linspace(-.02, .02, 3), indexing='ij')
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _frame(frame_id, *, rgb=None, k=None, logical_index=None,
           session='session-A', object_id=15, reference='sha256:refs-A',
           timestamp=None, timestamp_kind=FRAME_INDEX_CLOCK,
           native_size=(640, 480), native_from_input=None):
    if rgb is None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    if k is None:
        k = np.array([[700., 0., 320.], [0., 710., 240.], [0., 0., 1.]])
    return FrameInputV1(session, object_id, reference, frame_id,
                        frame_id if logical_index is None else logical_index,
                        timestamp, timestamp_kind, rgb, k, native_size,
                        native_from_input)


def _tracker(reference_points=None, *, seed=None, seed_pixels=None, seed_points=None):
    points = _point_bank() if reference_points is None else reference_points
    descriptors = np.random.default_rng(23).integers(0, 256, (len(points), 32), dtype=np.uint8)
    model = points.copy()
    tracker = PrototypeTrackerV1(points, descriptors, model, session_id='session-A',
                                 object_id=15, reference_id='sha256:refs-A',
                                 onboarding_seed=seed, seed_pixels=seed_pixels,
                                 seed_points=seed_points)
    return tracker, points, descriptors, model


def _rotation_y(degrees):
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), 0., np.sin(angle)],
                     [0., 1., 0.],
                     [-np.sin(angle), 0., np.cos(angle)]])


class CameraAndPixelConventionTests(unittest.TestCase):
    def test_projection_uses_source_metres_and_cv_camera_axes(self):
        k = np.array([[800., 0., 320.], [0., 900., 240.], [0., 0., 1.]])
        points = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.],
                           [0., 0., .10], [.01, -.02, .03]])
        translation = np.array([.02, -.01, 1.])
        pixels, depth = project_points_cv(points, np.eye(3), translation, k)
        expected = _manual_project(points, np.eye(3), translation, k)
        np.testing.assert_allclose(pixels, expected, atol=1e-10)
        np.testing.assert_allclose(depth, points[:, 2]+1., atol=1e-12)
        self.assertAlmostEqual(pixels[1, 0]-pixels[0, 0], 8., places=8)
        self.assertAlmostEqual(pixels[2, 1]-pixels[0, 1], 9., places=8)
        self.assertLess(pixels[3, 0], pixels[0, 0])
        self.assertGreater(pixels[3, 1], pixels[0, 1])

    def test_ninety_and_one_eighty_degree_orientations_match_analytic_projection(self):
        k = np.array([[640., 0., 160.], [0., 660., 120.], [0., 0., 1.]])
        points = np.array([[.012, .021, .034], [-.025, .014, .008]])
        poses = [np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]),
                 np.diag([-1., -1., 1.])]
        for rotation in poses:
            with self.subTest(rotation=rotation.tolist()):
                self.assertTrue(is_proper_rotation(rotation))
                translation = np.array([.03, -.02, .6])
                actual, _ = project_points_cv(points, rotation, translation, k)
                np.testing.assert_allclose(actual, _manual_project(points, rotation, translation, k), atol=1e-10)

    def test_crop_resize_propagates_nonuniform_scale_and_half_center_offset(self):
        native_size = (1280, 720)
        crop = (100., 40., 800., 600.)
        output_size = (320, 300)
        native_from_input = crop_resize_native_from_input(native_size, crop, output_size)
        input_from_native = crop_resize_input_from_native(native_size, crop, output_size)
        self.assertAlmostEqual(input_from_native[0, 0], .4)
        self.assertAlmostEqual(input_from_native[1, 1], .5)
        self.assertNotAlmostEqual(input_from_native[0, 0], input_from_native[1, 1])
        self.assertAlmostEqual(input_from_native[0, 2], .4*(.5-100.)-.5)
        self.assertAlmostEqual(input_from_native[1, 2], .5*(.5-40.)-.5)
        np.testing.assert_allclose(input_from_native@native_from_input, np.eye(3), atol=1e-12)

        k_native = np.array([[900., 0., 640.], [0., 880., 360.], [0., 0., 1.]])
        rotation = _rotation_y(18.)
        translation = np.array([.02, -.015, .8])
        points = _point_bank()[::7]
        native_pixels = _manual_project(points, rotation, translation, k_native)
        input_pixels = transform_pixels(native_pixels, input_from_native)
        k_input = input_from_native@k_native
        projected_input, _ = project_points_cv(points, rotation, translation, k_input)
        np.testing.assert_allclose(input_pixels, projected_input, atol=1e-9)
        np.testing.assert_allclose(transform_pixels(input_pixels, native_from_input), native_pixels, atol=1e-9)

    def test_renderer_camera_uses_corner_centers_and_explicit_half_pixel_conversion(self):
        k_cv = np.array([[500., 0., 42.25], [0., 520., 31.75], [0., 0., 1.]])
        k_corner = cv_intrinsics_to_renderer(k_cv)
        point = np.array([[.01, -.02, .7]])
        cv_pixel = _manual_project(point, np.eye(3), np.zeros(3), k_cv)
        raster_center = _manual_project(point, np.eye(3), np.zeros(3), k_corner)
        np.testing.assert_allclose(raster_centers_to_cv(raster_center), cv_pixel, atol=1e-10)

    def test_reflected_source_basis_reverses_winding_and_preserves_camera_mapping(self):
        bench_from_source = np.eye(4)
        bench_from_source[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]
        bench_from_source[:3, 3] = [.02, -.03, .01]
        self.assertLess(np.linalg.det(bench_from_source[:3, :3]), 0.)
        faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int64)
        np.testing.assert_array_equal(triangles_for_basis(faces, bench_from_source),
                                      np.array([[0, 2, 1], [2, 0, 3]]))
        source = np.array([[.01, .02, .03], [-.04, .01, .08], [.02, -.03, .12]])
        rotation = _rotation_y(90.)
        translation = np.array([.01, .02, .8])
        source_camera = source@rotation.T+translation
        bench = source@bench_from_source[:3, :3].T+bench_from_source[:3, 3]
        camera_from_bench = camera_from_source_to_bench(rotation, translation, bench_from_source)
        bench_camera = np.column_stack([bench, np.ones(len(bench))])@camera_from_bench.T
        np.testing.assert_allclose(bench_camera[:, :3], source_camera, atol=1e-12)

    def test_behind_camera_and_improper_rotations_are_rejected(self):
        k = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        with self.assertRaises(ValueError):
            project_points_cv(np.array([[0., 0., 0.]]), np.eye(3), [0., 0., .02], k)
        with self.assertRaises(ValueError):
            project_points_cv(np.array([[0., 0., 0.]]), np.diag([-1., 1., 1.]), [0., 0., .5], k)

    def test_near_clipped_slanted_mesh_matches_independent_ray_uv_depth_and_texture_oracle(self):
        positions = np.array([[-.035, -.025, .025], [.04, -.025, .12], [.005, .05, .11]])
        uv = np.array([[0., 0.], [1., .08], [.12, 1.]])
        mesh = Mesh('slanted', positions, np.tile([0., 0., -1.], (3, 1)), uv,
                    np.array([[0, 1, 2]], dtype=np.int64))
        texture = np.empty((8, 8, 3), dtype=np.uint8)
        for y in range(8):
            for x in range(8):
                texture[y, x] = [x*31, y*29, (x*17+y*13) % 256]
        texture[0, 0], texture[0, -1] = [255, 0, 0], [0, 255, 0]
        texture[-1, 0], texture[-1, -1] = [0, 0, 255], [255, 255, 0]
        k_cv = np.array([[170., 0., 80.], [0., 175., 60.], [0., 0., 1.]])
        k_render = cv_intrinsics_to_renderer(k_cv)
        camera = Camera(160, 120, k_render[0, 0], k_render[1, 1],
                        k_render[0, 2], k_render[1, 2], np.zeros(3), np.eye(3), near=.05)
        frame = render(mesh, camera, cull=False, texture_image=texture, shade=False)
        visible = np.argwhere(frame.triangle == 0)
        self.assertGreater(len(visible), 100)
        self.assertLess(positions[:, 2].min(), camera.near)
        self.assertGreaterEqual(frame.depth[frame.triangle == 0].min(), camera.near-1e-8)
        for y, x in visible:
            # Raster sample (x+.5,y+.5) is the OpenCV ray at integer center (x,y).
            ray = np.array([(x-k_cv[0, 2])/k_cv[0, 0],
                            (y-k_cv[1, 2])/k_cv[1, 1], 1.])
            hit = ray_triangle(np.zeros(3), ray, positions, uv)
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(frame.depth[y, x], hit[0], delta=2e-6)
            np.testing.assert_allclose(frame.uv[y, x], hit[1], atol=2e-6)
            u, v = frame.uv[y, x]
            tx, ty = np.clip(u, 0, 1)*7, (1-np.clip(v, 0, 1))*7
            x0, y0 = int(np.floor(tx)), int(np.floor(ty))
            x1, y1 = min(x0+1, 7), min(y0+1, 7)
            dx, dy = tx-x0, ty-y0
            expected = ((1-dx)*(1-dy)*texture[y0, x0]+dx*(1-dy)*texture[y0, x1]+
                        (1-dx)*dy*texture[y1, x0]+dx*dy*texture[y1, x1]).astype(np.uint8)
            np.testing.assert_array_equal(frame.color[y, x], expected)


class PrototypeRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.points = _point_bank()
        self.k = np.array([[700., 0., 320.], [0., 710., 240.], [0., 0., 1.]])
        self.rotation = np.eye(3)
        self.translation = np.array([.01, -.01, .72])
        self.pixels, _ = project_points_cv(self.points, self.rotation, self.translation, self.k)

    @staticmethod
    def _identity_lk(_previous, _current, pixels, *_args, **_kwargs):
        value = np.asarray(pixels).copy()
        return value, np.ones((len(value), 1), dtype=np.uint8), None

    def test_two_consecutive_pnp_candidates_are_required_before_pose_is_visible(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        with patch('bench.prototype_tracking.cv2.calcOpticalFlowPyrLK', side_effect=self._identity_lk):
            first = tracker.update(_frame(10, k=self.k, logical_index=10))
            self.assertEqual(first.state, 'recovering')
            self.assertIsNone(first.camera_from_object)
            self.assertIsNone(first.manifest()['cameraFromObject'])
            second = tracker.update(_frame(11, k=self.k, logical_index=11))
        self.assertEqual(second.state, 'tracking')
        self.assertIsNone(second.reason)
        self.assertIsNotNone(second.camera_from_object)
        self.assertGreaterEqual(second.statistics['temporalLinks'], 12)
        self.assertGreaterEqual(second.statistics['consistencyFraction'], .5)
        self.assertTrue(is_proper_rotation(second.camera_from_object[:3, :3]))
        np.testing.assert_allclose(second.camera_from_object[:3, 3], self.translation, atol=2e-3)

    def test_tracking_failure_hides_old_pose_and_recovery_starts_from_references(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        with patch('bench.prototype_tracking.cv2.calcOpticalFlowPyrLK', side_effect=self._identity_lk):
            tracker.update(_frame(12, k=self.k, logical_index=12))
            visible = tracker.update(_frame(13, k=self.k, logical_index=13))
        self.assertEqual(visible.state, 'tracking')
        self.assertIsNotNone(visible.camera_from_object)
        tracker.rigid.flow = Mock(return_value=(np.empty((0, 2)), np.empty((0, 3))))
        tracker.rigid.match = Mock(return_value=(np.empty((0, 2)), np.empty((0, 3))))
        failed = tracker.update(_frame(14, k=self.k, logical_index=14))
        self.assertEqual(failed.state, 'lost')
        self.assertIsNone(failed.camera_from_object)
        self.assertIsNone(tracker.rigid.previous_gray)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        first_recovery = tracker.update(_frame(15, k=self.k, logical_index=15))
        self.assertEqual(first_recovery.state, 'recovering')
        self.assertIsNone(first_recovery.camera_from_object)

    def test_inconsistent_second_candidate_restarts_pending_then_recovers(self):
        tracker, points, _, _ = _tracker(self.points)
        bad_rotation = self.rotation
        bad_translation = self.translation+np.array([.04, 0., 0.])
        poses = [(self.rotation, self.translation), (bad_rotation, bad_translation),
                 (bad_rotation, bad_translation)]
        projected = [project_points_cv(self.points, r, t, self.k)[0] for r, t in poses]
        tracker.rigid.match = Mock(side_effect=[(pixels.copy(), points.copy()) for pixels in projected])
        with patch('bench.prototype_tracking.cv2.calcOpticalFlowPyrLK', side_effect=self._identity_lk):
            first = tracker.update(_frame(20, k=self.k, logical_index=20))
            second = tracker.update(_frame(21, k=self.k, logical_index=21))
            third = tracker.update(_frame(22, k=self.k, logical_index=22))
        self.assertEqual(first.state, 'recovering')
        self.assertEqual(second.state, 'recovering')
        self.assertEqual(second.reason, 'recovery_consensus_failed')
        self.assertIsNone(second.camera_from_object)
        self.assertEqual(third.state, 'tracking')
        np.testing.assert_allclose(third.camera_from_object[:3, :3], bad_rotation, atol=2e-3)
        np.testing.assert_allclose(third.camera_from_object[:3, 3], bad_translation, atol=2e-3)

    def test_failed_lk_clears_public_pose_and_keeps_only_a_new_private_candidate(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        with patch('bench.prototype_tracking.cv2.calcOpticalFlowPyrLK',
                   side_effect=[(None, None, None), self._identity_lk(None, None, self.pixels),
                                self._identity_lk(None, None, self.pixels)]):
            first = tracker.update(_frame(30, k=self.k, logical_index=30))
            failed = tracker.update(_frame(31, k=self.k, logical_index=31))
        self.assertEqual(first.state, 'recovering')
        self.assertEqual(failed.state, 'recovering')
        self.assertEqual(failed.reason, 'recovery_consensus_failed')
        self.assertIsNone(failed.camera_from_object)
        self.assertEqual(failed.statistics['temporalLinks'], 0)

    def test_failed_pnp_clears_pending_and_legacy_tracker_resets_temporal_arrays(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels[:8].copy(), points[:8].copy()))
        failed = tracker.update(_frame(40, k=self.k, logical_index=40))
        self.assertEqual(failed.state, 'lost')
        self.assertIsNone(failed.camera_from_object)
        self.assertIsNone(tracker._pending)
        self.assertTrue(tracker.rigid.lost)
        self.assertIsNone(tracker.rigid.previous_gray)
        self.assertIsNone(tracker.rigid.previous_pixels)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        recovered1 = tracker.update(_frame(41, k=self.k, logical_index=41))
        self.assertEqual(recovered1.state, 'recovering')
        self.assertIsNone(recovered1.camera_from_object)

    def test_lk_none_arrays_and_shape_mismatch_are_safe(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.previous_gray = np.zeros((20, 20), np.uint8)
        tracker.rigid.previous_pixels = np.array([[5., 5.], [6., 6.]])
        tracker.rigid.previous_points = points[:2].copy()
        with patch('bench.tracker.cv2.calcOpticalFlowPyrLK', return_value=(None, None, None)):
            pixels, anchors = tracker.rigid.flow(np.zeros((20, 20), np.uint8))
        self.assertEqual(pixels.shape, (0, 2))
        self.assertEqual(anchors.shape, (0, 3))
        with patch('bench.tracker.cv2.calcOpticalFlowPyrLK',
                   side_effect=[(np.zeros((1, 1, 2), np.float32), np.ones((1, 1), np.uint8), None)]):
            pixels, anchors = tracker.rigid.flow(np.zeros((20, 20), np.uint8))
        self.assertEqual(pixels.shape, (0, 2))
        self.assertEqual(anchors.shape, (0, 3))

    def test_onboarding_inputs_are_private_immutable_and_exact_anchor_identity_is_preserved(self):
        references = self.points.copy()
        descriptors = np.random.default_rng(4).integers(0, 256, (len(references), 32), dtype=np.uint8)
        model = references.copy()
        reference_snapshot, descriptor_snapshot, model_snapshot = references.copy(), descriptors.copy(), model.copy()
        tracker = PrototypeTrackerV1(references, descriptors, model, session_id='session-A',
                                     object_id=15, reference_id='sha256:refs-A')
        references[:] = 0; descriptors[:] = 0; model[:] = 0
        np.testing.assert_array_equal(tracker.rigid.reference, reference_snapshot)
        np.testing.assert_array_equal(tracker.rigid.descriptors, descriptor_snapshot)
        np.testing.assert_array_equal(tracker.rigid.model, model_snapshot)
        self.assertFalse(tracker.rigid.reference.flags.writeable)
        self.assertFalse(tracker.rigid.descriptors.flags.writeable)
        self.assertFalse(tracker.rigid.model.flags.writeable)
        nearby = np.array([[0., 0., 0.], [.0002, 0., 0.]])
        pixels = np.array([[10., 10.], [11., 11.]])
        kept_pixels, kept_points = tracker._deduplicate(pixels, nearby)
        self.assertEqual(len(kept_points), 2)
        np.testing.assert_array_equal(kept_pixels, pixels)

    def test_whitelisted_frame_mapping_rejects_evaluation_fields(self):
        mapping = {
            'sessionId': 'session-A', 'objectId': 15, 'referenceId': 'sha256:refs-A',
            'frameId': 1, 'logicalIndex': 1, 'timestampSeconds': None,
            'timestampKind': FRAME_INDEX_CLOCK, 'rgb': np.zeros((8, 8, 3), np.uint8),
            'KNative': np.array([[10., 0., 4.], [0., 10., 4.], [0., 0., 1.]]),
            'nativeSize': (8, 8),
        }
        self.assertEqual(FrameInputV1.from_mapping(mapping).frame_id, 1)
        with self.assertRaises(ValueError):
            FrameInputV1.from_mapping({**mapping, 'evaluationPose': np.eye(4)})

    def test_bad_identity_and_replayed_frame_reset_without_lowering_highwatermarks(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        pending = tracker.update(_frame(50, k=self.k, logical_index=50))
        self.assertEqual(pending.state, 'recovering')
        foreign = _frame(51, k=self.k, logical_index=51, session='other-session')
        rejected = tracker.update(foreign)
        self.assertEqual(rejected.state, 'lost')
        self.assertEqual(rejected.reason, 'identity_mismatch')
        self.assertIsNone(tracker._pending)
        valid = tracker.update(_frame(51, k=self.k, logical_index=51))
        self.assertEqual(valid.state, 'recovering')
        duplicate = tracker.update(_frame(51, k=self.k, logical_index=51))
        self.assertEqual(duplicate.state, 'lost')
        self.assertEqual(duplicate.reason, 'out_of_order_frame')
        self.assertEqual(tracker._last_seen_frame_id, 51)
        self.assertIsNone(duplicate.camera_from_object)
        self.assertEqual(tracker.update(_frame(50, k=self.k, logical_index=52)).reason,
                         'out_of_order_frame')

    def test_invalid_calibration_and_calibration_change_never_reuse_old_pose(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        invalid_k = self.k.copy(); invalid_k[0, 0] = 0
        invalid = tracker.update(_frame(60, k=invalid_k, logical_index=60))
        self.assertEqual(invalid.state, 'lost')
        self.assertIsNone(invalid.camera_from_object)
        first = tracker.update(_frame(61, k=self.k, logical_index=61))
        self.assertEqual(first.state, 'recovering')
        changed_k = self.k.copy(); changed_k[0, 0] += 15
        changed = tracker.update(_frame(62, k=changed_k, logical_index=62))
        self.assertEqual(changed.state, 'recovering')
        self.assertEqual(changed.reason, 'recovery_restarted_after_calibration_change')
        self.assertIsNone(changed.camera_from_object)

    def test_crop_transform_outside_native_resolution_is_rejected_before_matching(self):
        tracker, _, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(side_effect=AssertionError('invalid calibration must not reach vision'))
        outside = np.array([[1., 0., 500.], [0., 1., 0.], [0., 0., 1.]])
        frame = _frame(65, k=self.k, logical_index=65, native_from_input=outside)
        result = tracker.update(frame)
        self.assertEqual(result.state, 'lost')
        self.assertEqual(result.reason, 'calibration_resolution_mismatch')
        self.assertIsNone(result.camera_from_object)

    def test_physical_timestamps_must_increase_and_frame_index_clock_never_fakes_seconds(self):
        tracker, points, _, _ = _tracker(self.points)
        tracker.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        first = tracker.update(_frame(70, k=self.k, logical_index=70, timestamp=1.25,
                                      timestamp_kind=PHYSICAL_CLOCK))
        self.assertEqual(first.timestamp_seconds, 1.25)
        backwards = tracker.update(_frame(71, k=self.k, logical_index=71, timestamp=1.20,
                                          timestamp_kind=PHYSICAL_CLOCK))
        self.assertEqual(backwards.state, 'lost')
        self.assertEqual(backwards.reason, 'out_of_order_frame')
        self.assertIsNone(backwards.camera_from_object)

        logical, _, _, _ = _tracker(self.points)
        logical.rigid.match = Mock(return_value=(self.pixels.copy(), points.copy()))
        record = logical.update(_frame(70, k=self.k, logical_index=70))
        self.assertIsNone(record.timestamp_seconds)
        self.assertEqual(record.timestamp_kind, FRAME_INDEX_CLOCK)

    def test_onboarding_seed_can_initialize_flow_but_never_uses_pose_fallback(self):
        seed_tracker, points, _, _ = _tracker(self.points)
        seed_frame = _frame(9, k=self.k, logical_index=9)
        seeded, _, _, _ = _tracker(self.points, seed=seed_frame,
                                   seed_pixels=self.pixels, seed_points=points)
        seeded.rigid.flow = Mock(return_value=(self.pixels.copy(), points.copy()))
        seeded.rigid.match = Mock(return_value=(np.empty((0, 2)), np.empty((0, 3))))
        result = seeded.update(_frame(10, k=self.k, logical_index=10))
        self.assertEqual(result.state, 'tracking')
        self.assertIsNotNone(result.camera_from_object)
        # The seeded frame contains no pose; only fresh image correspondences can expose one.
        failed, _, _, _ = _tracker(self.points, seed=seed_frame,
                                   seed_pixels=self.pixels, seed_points=points)
        failed.rigid.flow = Mock(return_value=(np.empty((0, 2)), np.empty((0, 3))))
        failed.rigid.match = Mock(return_value=(np.empty((0, 2)), np.empty((0, 3))))
        hidden = failed.update(_frame(10, k=self.k, logical_index=10))
        self.assertEqual(hidden.state, 'lost')
        self.assertIsNone(hidden.camera_from_object)


if __name__ == '__main__':
    unittest.main()
