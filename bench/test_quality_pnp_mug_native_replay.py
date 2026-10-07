"""CPU tests for full-retained/native PnP replay and opt-in adapter capture."""
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from .quality_contract import Frame, validate
from .quality_gotrack import GoTrackRefiner
from .quality_assets import save_result
from .quality_pnp_mug_native_probe import (
    FRAME_IDS, ITERATIONS, PACKET_BUDGET_BYTES, VALIDATION_STATS_ATOL,
    VALIDATION_STATS_RTOL, NativePacketCapture, _array_info, _copy_observed_mask,
)
from .quality_pnp_mug_native_replay import (
    _compact_replay_pair, _load_capture, _load_packet,
    compare_control_capture, replay_native_packet,
)
from .vision import cv2


def _pose_from_rvec_tvec(rvec, tvec):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    result[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return result


def _native_camera_mask(candidate_pixels, height=720, width=1280, padding=20):
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels = np.asarray(candidate_pixels, dtype=np.float64)
    finite = np.isfinite(pixels).all(axis=1)
    xy = pixels[finite].astype(int)
    if not len(xy):
        return mask
    x0 = max(0, int(xy[:, 0].min()) - padding)
    x1 = min(width, int(xy[:, 0].max()) + padding + 1)
    y0 = max(0, int(xy[:, 1].min()) - padding)
    y1 = min(height, int(xy[:, 1].max()) + padding + 1)
    mask[y0:y1, x0:x1] = 255
    return mask


def _synthetic_native_packet(count=420, corrupt_unsampled=False):
    rng = np.random.default_rng(4102026)
    points = rng.uniform([-38., -29., -16.], [41., 31., 18.], size=(count, 3)).astype(np.float64)
    native_rvec = np.array([.19, -.12, .085], dtype=np.float64)
    native_tvec_mm = np.array([17., -12., 655.], dtype=np.float64)
    native_pose_mm = _pose_from_rvec_tvec(native_rvec, native_tvec_mm)
    c_rvec = np.array([-.075, .11, .035], dtype=np.float64)
    crop_from_orig = _pose_from_rvec_tvec(c_rvec, np.array([.2, -.15, .1]))
    crop_pose = crop_from_orig @ native_pose_mm
    crop_rvec = cv2.Rodrigues(crop_pose[:3, :3])[0]
    crop_tvec = crop_pose[:3, 3].copy()
    crop_k = np.array([[735., 0., 140.], [0., 748., 139.], [0., 0., 1.]], dtype=np.float64)
    native_k = np.array([[825., 0., 640.], [0., 818., 360.], [0., 0., 1.]], dtype=np.float64)
    target, _ = cv2.projectPoints(points, crop_rvec, crop_tvec, crop_k, None)
    target = target.reshape(-1, 2)
    target += rng.normal(0., .025, size=target.shape)
    ids = np.linspace(0, count-1, min(count, 10000), dtype=int)
    if corrupt_unsampled:
        sampled = np.zeros(count, dtype=bool)
        sampled[ids] = True
        target[~sampled] += np.array([22., 14.])
    initial_rvec = crop_rvec + np.array([[.009], [-.007], [.005]])
    initial_tvec = crop_tvec + np.array([1.5, -1.2, 3.5])
    return dict(
        full_obj_points_mm=points,
        full_target_crop_px=target.astype(np.float64),
        full_weights=np.full(count, .92, dtype=np.float32),
        sample_ids=ids,
        crop_k=crop_k,
        crop_from_orig=crop_from_orig,
        native_k=native_k,
        current_crop_pose_mm=_pose_from_rvec_tvec(initial_rvec, initial_tvec),
        initial_rvec=initial_rvec,
        initial_tvec_mm=initial_tvec,
        known_native_pose_m=pose_units_for_test(native_pose_mm),
        known_crop_pose=crop_pose,
    )


def pose_units_for_test(pose_mm):
    result = np.array(pose_mm, copy=True)
    result[:3, 3] *= .001
    return result


class QualityPnpMugNativeReplayTests(unittest.TestCase):
    def test_report_compaction_keeps_evidence_without_candidate_correspondence_payload(self):
        stats = dict(correspondences=640, inliers=512, score=.81)
        candidate_arrays = {
            'pose_m': dict(shape=[4, 4], dtype='float64', sha256='pose-hash'),
            'points_object_m': dict(shape=[640, 3], dtype='float64', sha256='points-hash'),
            'pixels_native': dict(shape=[640, 2], dtype='float64', sha256='pixels-hash'),
            'weights': dict(shape=[640], dtype='float32', sha256='weights-hash'),
        }
        branch = dict(
            fit_state='refined', solver_success=True, validation_state='accepted',
            validation_reason=None, validation_stats=stats,
            ransac_inlier_ids=[1, 7, 12], retained_ransac_inlier_ids=[8, 56, 96],
            candidate=dict(
                pose_m=np.eye(4).tolist(), points_object_m=[[.1, .2, .3]] * 640,
                pixels_native=[[100., 110.]] * 640, weights=[.9] * 640,
            ),
            candidate_arrays=candidate_arrays,
            full_retained_native_residuals_720=dict(
                retained_count=1000, finite_residual_points=998,
                nonfinite_residual_points=2, residual_all_finite=dict(p95=1.7),
            ),
            packet_array_hashes=dict(full_obj_points_mm=dict(sha256='input-hash')),
        )
        pair = dict(
            frame_id=827, iteration=2, attempted=True,
            packet_path='packets/frame-0827-iteration-2.npz', packet_sha256='packet-hash',
            packet_array_hashes=dict(full_obj_points_mm=dict(sha256='input-hash')),
            control_equivalence=dict(matched=True, mismatches=[]),
            branches=dict(control_use_extrinsic_guess=branch),
        )

        detail, index = _compact_replay_pair(pair)
        compact = detail['branches']['control_use_extrinsic_guess']
        self.assertEqual(compact['candidate'], {'pose_m': np.eye(4).tolist()})
        self.assertEqual(compact['candidate_arrays'], candidate_arrays)
        self.assertEqual(compact['validation_stats'], stats)
        self.assertEqual(compact['ransac_inlier_ids'], [1, 7, 12])
        self.assertEqual(compact['retained_ransac_inlier_ids'], [8, 56, 96])
        self.assertEqual(compact['full_retained_native_residuals_720'],
                         branch['full_retained_native_residuals_720'])
        self.assertEqual(detail['packet_array_hashes'], pair['packet_array_hashes'])
        self.assertNotIn('points_object_m', compact['candidate'])
        self.assertNotIn('pixels_native', compact['candidate'])
        self.assertNotIn('weights', compact['candidate'])
        self.assertEqual(index['packet_path'], pair['packet_path'])
        self.assertEqual(index['control_equivalence'], pair['control_equivalence'])
        self.assertEqual(index['branches']['control_use_extrinsic_guess']['validation_state'], 'accepted')
        self.assertNotIn('packet_array_hashes', index)
        self.assertNotIn('control_expected_capture', index)

    def test_mask_budget_preflight_leaves_no_file_or_counter_change(self):
        mask = np.full((8, 11), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.png'
            self.assertTrue(cv2.imwrite(str(source), mask))
            source_bytes = source.stat().st_size
            aggregate = PACKET_BUDGET_BYTES - source_bytes + 1
            record = dict(
                aggregate_capture_bytes=aggregate, observed_mask_bytes=13,
                packet_failures=[],
            )

            with self.assertRaisesRegex(ValueError, 'aggregate budget'):
                _copy_observed_mask(root, record, 827, source, mask)

            self.assertFalse((root / 'observed_masks' / 'frame-0827.png').exists())
            self.assertEqual(record['aggregate_capture_bytes'], aggregate)
            self.assertEqual(record['observed_mask_bytes'], 13)
            self.assertEqual(record['packet_failures'][-1]['attempted_bytes'], source_bytes)

    def test_replay_handles_missing_ransac_inliers_for_both_guess_flags(self):
        arrays = _synthetic_native_packet()
        arrays.pop('known_native_pose_m')
        arrays.pop('known_crop_pose')

        class NoInliersCV:
            error = cv2.error
            SOLVEPNP_ITERATIVE = cv2.SOLVEPNP_ITERATIVE

            @staticmethod
            def setRNGSeed(_):
                pass

            @staticmethod
            def solvePnPRansac(*_, **__):
                return False, None, None, None

        frame = Frame(827, np.zeros((720, 1280, 3), dtype=np.uint8), arrays['native_k'])
        mask = np.full((720, 1280), 255, dtype=np.uint8)
        for guess in (True, False):
            with self.subTest(use_extrinsic_guess=guess):
                result = replay_native_packet(arrays, 827 * 17, guess, frame, mask, NoInliersCV)
                self.assertEqual(result['fit_state'], 'ransac_unavailable')
                self.assertFalse(result['solver_success'])
                self.assertEqual(result['ransac_inlier_count'], 0)
                self.assertEqual(result['ransac_inlier_ids'], [])
                self.assertEqual(result['retained_ransac_inlier_ids'], [])
                self.assertIsNone(result['candidate'])
                self.assertEqual(result['validation_state'], 'not_run')
                self.assertEqual(result['validation_reason'], 'candidate_unavailable_ransac_unavailable')

    def test_native_loader_rejects_tolerance_drift_or_unfrozen_flag(self):
        good_tolerance = dict(
            atol=VALIDATION_STATS_ATOL, rtol=VALIDATION_STATS_RTOL,
            frozen_before_capture=True,
        )
        capture = dict(
            object='mug', requested_frame_ids=FRAME_IDS, requested_iterations=ITERATIONS,
            status='complete', complete=True,
            frames=[dict(
                frame_id=frame_id,
                iteration_accounting=[dict(iteration=i, attempted=False) for i in range(ITERATIONS)],
            ) for frame_id in FRAME_IDS],
            aggregate_capture_bytes=0, captured_packet_bytes=0, observed_mask_bytes=0,
            packets=[], validation_stats_tolerance=good_tolerance,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / 'capture.json'
            save_result(manifest, capture)
            _load_capture(manifest)

            drifted = dict(capture, validation_stats_tolerance=dict(good_tolerance, atol=1e-9))
            missing = dict(capture)
            missing.pop('validation_stats_tolerance')
            integer_flag = dict(
                capture,
                validation_stats_tolerance=dict(good_tolerance, frozen_before_capture=1),
            )
            for invalid in (drifted, missing, integer_flag):
                with self.subTest(tolerance=invalid.get('validation_stats_tolerance')):
                    save_result(manifest, invalid)
                    with self.assertRaisesRegex(ValueError, 'tolerance'):
                        _load_capture(manifest)

    def test_nonidentity_transform_and_mm_to_metre_conversion_replay_on_full_set(self):
        arrays = _synthetic_native_packet()
        native_pose = arrays.pop('known_native_pose_m')
        arrays.pop('known_crop_pose')
        frame = Frame(827, np.zeros((720, 1280, 3), dtype=np.uint8), arrays['native_k'])
        # Derive an observed mask from the known native pose's object projections.
        camera_points = arrays['full_obj_points_mm'] * .001 @ native_pose[:3, :3].T + native_pose[:3, 3]
        projected_h = camera_points @ arrays['native_k'].T
        known_pixels = projected_h[:, :2] / projected_h[:, 2:3]
        mask = _native_camera_mask(known_pixels)
        source_copies = {name: value.copy() for name, value in arrays.items()}

        results = {
            guess: replay_native_packet(arrays, 827 * 17 + 2, guess, frame, mask)
            for guess in (True, False)
        }
        for result in results.values():
            self.assertEqual(result['fit_state'], 'refined')
            pose = np.asarray(result['candidate']['pose_m'])
            self.assertTrue(np.isfinite(pose).all())
            np.testing.assert_allclose(pose, native_pose, atol=.0015, rtol=0)
            self.assertEqual(result['full_retained_crop_residuals']['retained_count'], len(arrays['full_obj_points_mm']))
            self.assertEqual(result['full_retained_native_residuals_720']['retained_count'], len(arrays['full_obj_points_mm']))
            self.assertEqual(result['full_retained_crop_residuals']['depth_units'], 'millimetres')
            self.assertEqual(result['full_retained_crop_residuals']['positive_depth_threshold'], 10.)
            self.assertEqual(result['full_retained_native_residuals_720']['depth_units'], 'metres')
            self.assertEqual(result['full_retained_native_residuals_720']['positive_depth_threshold'], .01)
            self.assertEqual(result['validation_state'], 'accepted')
            self.assertIsNone(result['validation_reason'])
        for name, expected in source_copies.items():
            np.testing.assert_array_equal(arrays[name], expected)
        self.assertNotEqual(results[True]['use_extrinsic_guess'], results[False]['use_extrinsic_guess'])

        actual = results[True]
        expected_capture = dict(
            fit_state=actual['fit_state'], solver_success=actual['solver_success'],
            rng_seed=actual['rng_seed'], ransac_inlier_count=actual['ransac_inlier_count'],
            retained_count=actual['input_retained_count'], sampled_count=actual['sampled_input_count'],
            ransac_inlier_ids=actual['ransac_inlier_ids'],
            retained_inlier_ids=actual['retained_ransac_inlier_ids'],
            pose_m=np.array(actual['candidate']['pose_m'], copy=True),
            candidate_arrays=actual['candidate_arrays'],
            validation_state=actual['validation_state'],
            validation_reason=actual['validation_reason'],
            validation_stats=actual['validation_stats'],
        )
        expected_capture['pose_m'][0, 3] += 5e-7
        equivalence = compare_control_capture(actual, expected_capture)
        self.assertTrue(equivalence['matched'], repr(equivalence))
        expected_capture['candidate_arrays'] = dict(actual['candidate_arrays'])
        expected_capture['candidate_arrays']['pixels_native'] = dict(actual['candidate_arrays']['pixels_native'])
        expected_capture['candidate_arrays']['pixels_native']['sha256'] = 'changed'
        self.assertFalse(compare_control_capture(actual, expected_capture)['matched'])

    def test_unsampled_bad_points_change_full_retained_validation(self):
        arrays = _synthetic_native_packet(count=20001, corrupt_unsampled=True)
        sampled_target = arrays['full_target_crop_px'][arrays['sample_ids']]
        base = _synthetic_native_packet(count=20001, corrupt_unsampled=False)
        np.testing.assert_array_equal(sampled_target, base['full_target_crop_px'][arrays['sample_ids']])
        frame = Frame(853, np.zeros((720, 1280, 3), dtype=np.uint8), arrays['native_k'])
        mask = np.full((720, 1280), 255, dtype=np.uint8)

        result = replay_native_packet(arrays, 853 * 17, True, frame, mask)

        self.assertEqual(result['fit_state'], 'refined')
        self.assertEqual(result['sampled_input_count'], 10000)
        self.assertEqual(result['input_retained_count'], 20001)
        self.assertEqual(result['full_retained_native_residuals_720']['retained_count'], 20001)
        self.assertEqual(result['full_retained_native_residuals_720']['finite_residual_points'], 20001)
        self.assertGreater(result['full_retained_native_residuals_720']['residual_all_finite']['p95'], 8.)
        self.assertEqual(result['validation_state'], 'rejected')
        self.assertEqual(result['validation_reason'], 'inconsistent_current_image_correspondences')
        self.assertLess(result['validation_stats']['inliers'] / result['validation_stats']['correspondences'], .6)

    def test_observed_mask_failure_is_kept_as_validation_failure(self):
        arrays = _synthetic_native_packet()
        arrays.pop('known_native_pose_m')
        arrays.pop('known_crop_pose')
        frame = Frame(880, np.zeros((720, 1280, 3), dtype=np.uint8), arrays['native_k'])
        result = replay_native_packet(
            arrays, 880 * 17, True, frame, np.zeros((720, 1280), dtype=np.uint8))
        self.assertEqual(result['fit_state'], 'refined')
        self.assertEqual(result['validation_state'], 'rejected')
        self.assertEqual(result['validation_reason'], 'insufficient_visible_correspondences')
        self.assertEqual(result['validation_stats']['correspondences'], 0)

    def test_packet_roundtrip_keeps_full_retained_arrays_and_native_mask_link(self):
        arrays = _synthetic_native_packet()
        arrays.pop('known_native_pose_m')
        arrays.pop('known_crop_pose')
        frame = Frame(906, np.zeros((720, 1280, 3), dtype=np.uint8), arrays['native_k'])
        camera_points = arrays['full_obj_points_mm'] * .001 @ _synthetic_native_packet()['known_native_pose_m'][:3, :3].T
        camera_points += _synthetic_native_packet()['known_native_pose_m'][:3, 3]
        projected = camera_points @ arrays['native_k'].T
        mask = _native_camera_mask(projected[:, :2] / projected[:, 2:3])
        result = replay_native_packet(arrays, 906 * 17 + 1, True, frame, mask)
        self.assertEqual(result['fit_state'], 'refined')
        expected_sample = arrays['sample_ids']

        retained = dict(arrays)
        retained['sample_obj_points_mm'] = arrays['full_obj_points_mm'][expected_sample].copy()
        retained['sample_target_crop_px'] = arrays['full_target_crop_px'][expected_sample].copy()
        result_arrays = dict(
            ransac_inlier_ids=np.asarray(result['ransac_inlier_ids'], dtype=np.int32),
            retained_inlier_ids=np.asarray(result['retained_ransac_inlier_ids'], dtype=np.int64),
            refined_rvec=np.asarray(result['refined_rvec'], dtype=np.float64),
            refined_tvec_mm=np.asarray(result['refined_tvec_mm'], dtype=np.float64),
            crop_camera_from_object_mm=np.asarray(result['crop_camera_from_object_mm'], dtype=np.float64),
            native_camera_from_object_m=np.asarray(result['candidate']['pose_m'], dtype=np.float64),
            candidate_points_object_m=np.asarray(result['candidate']['points_object_m'], dtype=np.float64),
            candidate_pixels_native=np.asarray(result['candidate']['pixels_native'], dtype=np.float64),
            candidate_weights=np.asarray(result['candidate']['weights']),
        )
        record = dict(captured_packet_bytes=0, observed_mask_bytes=0,
                      aggregate_capture_bytes=0, packet_failures=[], packets=[],
                      packet_budget_bytes=64 * 1024**2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_result(root / 'capture.json', record)
            capturer = NativePacketCapture(root, record, {})
            mask_info = dict(path='observed_masks/frame-0906.png', bytes=10,
                             file_sha256='mask-file-hash',
                             array=_array_info(mask))
            context = dict(
                frame_id=906, frame=frame, mask=mask,
                input_hashes=dict(rgb_sha256='rgb', mask_sha256=_array_info(mask)['sha256'],
                                  mask_artifact_sha256='mask-file-hash', seed_pose_sha256='seed'),
                observed_mask=mask_info,
            )
            pending = dict(
                context=context, arrays=retained,
                sampled_entry=dict(path='packets/old.npz', sha256='old-packet-hash'),
            )
            event = dict(stage='result', frame_id=906, iteration=1,
                         rng_seed=906 * 17 + 1, fit_state='refined', solver_success=True,
                         ransac_inlier_count=result['ransac_inlier_count'])
            capturer._write_packet((906, 1), pending, event, result_arrays)
            entry = record['packets'][0]
            _, loaded_iteration, loaded_seed, loaded = _load_packet(root / 'capture.json', entry)
            self.assertEqual(loaded_iteration, 1)
            self.assertEqual(loaded_seed, 906 * 17 + 1)
            np.testing.assert_array_equal(loaded['full_obj_points_mm'], arrays['full_obj_points_mm'])
            np.testing.assert_array_equal(loaded['full_target_crop_px'], arrays['full_target_crop_px'])
            np.testing.assert_array_equal(loaded['full_weights'], arrays['full_weights'])
            np.testing.assert_array_equal(loaded['sample_ids'], arrays['sample_ids'])
            self.assertEqual(entry['observed_mask'], mask_info)
            self.assertEqual(entry['sample_boundary']['arrays']['obj_points']['sha256'],
                             _array_info(retained['sample_obj_points_mm'])['sha256'])

    def test_default_capture_hook_does_no_snapshot_work_and_callback_is_isolated(self):
        source = np.arange(12, dtype=np.float64).reshape(4, 3)
        default = GoTrackRefiner(None, None, 'mug', 'cpu')
        with mock.patch('bench.quality_gotrack.np.array', side_effect=AssertionError('unexpected snapshot')):
            default._emit_diagnostic_capture({'stage': 'retained'}, {'source': source})
        self.assertIsNone(default.diagnostic_capture_callback)

        source_copy = source.copy()
        np.random.seed(1234)
        random.seed(5678)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        captured = []

        def mutating_callback(packet):
            captured.append(packet['arrays']['source'].copy())
            packet['arrays']['source'][:] = -1
            np.random.seed(9)
            random.seed(10)

        enabled = GoTrackRefiner(None, None, 'mug', 'cpu', diagnostic_capture_callback=mutating_callback)
        enabled._emit_diagnostic_capture({'stage': 'retained'}, {'source': source})
        np.testing.assert_array_equal(source, source_copy)
        np.testing.assert_array_equal(captured[0], source_copy)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertEqual(random.getstate(), python_state)

    def test_normal_and_callback_enabled_refiners_return_identical_candidates(self):
        import torch

        intrinsics = np.array([[500., 0., 140.], [0., 500., 140.], [0., 0., 1.]], dtype=np.float64)
        object_points = []
        pixels = np.zeros((280, 280, 2), dtype=np.float32)
        xyz_camera = np.zeros((280, 280, 3), dtype=np.float32)
        retained_mask = np.zeros((280, 280), dtype=bool)
        pose_mm = np.eye(4, dtype=np.float64)
        pose_mm[2, 3] = 500.
        for yi, y in enumerate((-30., -20., -10., 0., 10., 20., 30.)):
            for xi, x in enumerate((-36., -24., -12., 0., 12., 24., 36.)):
                z = float(((xi * 3 + yi * 5) % 7) - 3)
                point = np.array([x, y, z], dtype=np.float64)
                uv, _ = cv2.projectPoints(point[None], np.zeros((3, 1)), np.array([[0.], [0.], [500.]]), intrinsics, None)
                u, v = uv.reshape(2)
                px, py = int(np.floor(u)), int(np.floor(v))
                retained_mask[py, px] = True
                pixels[py, px] = (u, v)
                xyz_camera[py, px] = (point + np.array([0., 0., 500.])).astype(np.float32)
                object_points.append(point)
        self.assertGreater(int(retained_mask.sum()), 24)
        pixel_tensor = torch.from_numpy(pixels[None])
        xyz_tensor = torch.from_numpy(xyz_camera[None])
        model_mask = torch.from_numpy(retained_mask[None])
        template = types.SimpleNamespace(
            rgbs=torch.zeros((1, 3, 280, 280)),
            masks=model_mask,
            depths=torch.zeros((1, 1, 280, 280)),
        )
        data = dict(
            crop_rgbs=torch.zeros((1, 3, 280, 280)),
            crop_masks=torch.ones((1, 280, 280)),
            templates=template,
        )
        transform3d = types.SimpleNamespace(
            get_3d_points_from_depth=lambda *_: (pixel_tensor, xyz_tensor))
        data_util = types.SimpleNamespace(
            compute_gotrack_inputs_from_init_poses=lambda **_: (data, [object()], torch.eye(4)[None]))
        misc = types.SimpleNamespace(get_intrinsic_matrix=lambda _: intrinsics)
        structs = types.SimpleNamespace(PinholePlaneCameraModel=lambda **kwargs: types.SimpleNamespace(**kwargs))
        utils_module = types.ModuleType('utils')
        utils_module.data_util = data_util
        utils_module.misc = misc
        utils_module.structs = structs
        utils_module.transform3d = transform3d

        class Renderer:
            vertices_m = np.zeros((1, 3), dtype=np.float64)

        def network(*_):
            return torch.zeros((1, 2, 280, 280)), torch.ones((1, 280, 280))

        rgb = np.zeros((280, 280, 3), dtype=np.uint8)
        observed_mask = np.full((280, 280), 255, dtype=np.uint8)
        frame = Frame(827, rgb, intrinsics)
        seed = np.eye(4, dtype=np.float64)
        seed[2, 3] = .5
        events = []

        with mock.patch.dict(sys.modules, {'utils': utils_module}):
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            normal_refiner = GoTrackRefiner(network, Renderer(), 'mug', 'cpu')
            normal_candidate = normal_refiner.refine(frame, observed_mask, seed)
            self.assertIsNone(normal_refiner.diagnostic_capture_callback)

            def callback(packet):
                events.append(packet)
                # A callback may use the global RNG; the adapter restores it.
                np.random.seed(301 + int(packet['event']['iteration']))
                random.seed(700 + int(packet['event']['iteration']))

            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            captured_refiner = GoTrackRefiner(
                network, Renderer(), 'mug', 'cpu', diagnostic_capture_callback=callback)
            captured_candidate = captured_refiner.refine(frame, observed_mask, seed)

        self.assertEqual(len(events), 10)
        np.testing.assert_array_equal(normal_candidate.pose, captured_candidate.pose)
        np.testing.assert_array_equal(normal_candidate.points_object_m, captured_candidate.points_object_m)
        np.testing.assert_array_equal(normal_candidate.pixels_image, captured_candidate.pixels_image)
        np.testing.assert_array_equal(normal_candidate.weights, captured_candidate.weights)

        retained = events[-2]['arrays']
        result_arrays = events[-1]['arrays']
        replay_arrays = dict(retained)
        replay_frame = Frame(827, rgb, intrinsics)
        replay = replay_native_packet(
            replay_arrays, events[-1]['event']['rng_seed'], True, replay_frame, observed_mask)
        valid, reason, stats = validate(captured_candidate, frame, observed_mask)
        expected = dict(
            fit_state=events[-1]['event']['fit_state'],
            solver_success=events[-1]['event']['solver_success'],
            rng_seed=events[-1]['event']['rng_seed'],
            retained_count=len(retained['full_obj_points_mm']),
            sampled_count=len(retained['sample_ids']),
            ransac_inlier_count=events[-1]['event']['ransac_inlier_count'],
            ransac_inlier_ids=result_arrays['ransac_inlier_ids'].astype(int).tolist(),
            retained_inlier_ids=result_arrays['retained_inlier_ids'].astype(int).tolist(),
            pose_m=captured_candidate.pose,
            candidate_arrays={
                'pose_m': _array_info(captured_candidate.pose),
                'points_object_m': _array_info(captured_candidate.points_object_m),
                'pixels_native': _array_info(captured_candidate.pixels_image),
                'weights': _array_info(captured_candidate.weights),
            },
            validation_state='accepted' if valid else 'rejected',
            validation_reason=reason,
            validation_stats=stats,
        )
        equivalence = compare_control_capture(replay, expected)
        self.assertTrue(equivalence['matched'], repr(equivalence))


if __name__ == '__main__':
    unittest.main()
