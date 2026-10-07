"""CPU-only tests for the captured-input PnP replay contract."""
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from .quality_pnp_mug_replay import replay_solver_inputs
from .quality_pnp_mug_replay import _create_output_root as _create_replay_output_root
from .quality_gotrack import GoTrackRefiner
from .quality_pnp_mug_probe import (
    PacketCapturingRefiner,
    _create_output_root as _create_probe_output_root,
    _packet_writer,
)
from .vision import cv2


def _synthetic_packet():
    rng = np.random.default_rng(20261002)
    points = rng.uniform([-35., -24., -15.], [38., 27., 19.], size=(160, 3)).astype(np.float64)
    known_rvec = np.array([0.31, -0.19, 0.23], dtype=np.float64).reshape(3, 1)
    known_tvec = np.array([11., -7., 430.], dtype=np.float64).reshape(3, 1)
    camera = np.array([[780., 0., 320.], [0., 755., 240.], [0., 0., 1.]], dtype=np.float64)
    target, _ = cv2.projectPoints(points, known_rvec, known_tvec, camera, None)
    target = target.reshape(-1, 2)
    target += rng.normal(0., .035, size=target.shape)
    outliers = rng.choice(len(points), size=25, replace=False)
    target[outliers] += rng.uniform(-60., 60., size=(len(outliers), 2))
    initial_rvec = known_rvec + np.array([[.025], [-.018], [.012]])
    initial_tvec = known_tvec + np.array([[3.], [-2.], [6.]])
    return points, target, camera, initial_rvec, initial_tvec, known_rvec, known_tvec, outliers


class QualityPnpMugReplayTests(unittest.TestCase):
    def test_both_options_recover_a_known_nonidentity_se3_pose(self):
        points, target, camera, initial_rvec, initial_tvec, known_rvec, known_tvec, outliers = _synthetic_packet()
        results = {}
        for use_guess in (True, False):
            results[use_guess] = replay_solver_inputs(
                points, target, camera, initial_rvec, initial_tvec,
                rng_seed=827 * 17 + 2, use_extrinsic_guess=use_guess,
            )
        known_rotation, _ = cv2.Rodrigues(known_rvec)
        for result in results.values():
            self.assertTrue(result['solver_success'])
            self.assertEqual(result['fit_state'], 'refined')
            self.assertGreaterEqual(result['ransac_inlier_count'], len(points) - len(outliers) - 2)
            self.assertGreaterEqual(result['full_solver_input_projected_inlier_count_at_2px'],
                                    len(points) - len(outliers) - 2)
            pose = np.asarray(result['crop_camera_from_object_mm'])
            rotation_error = np.linalg.norm(cv2.Rodrigues(pose[:3, :3] @ known_rotation.T)[0])
            self.assertLess(rotation_error, .004)
            self.assertLess(np.linalg.norm(pose[:3, 3] - known_tvec[:, 0]), .8)
            self.assertLess(result['full_solver_input_residual_px']['median'], .15)
            self.assertEqual(result['input_point_count'], len(points))
        self.assertEqual(results[True]['use_extrinsic_guess'], True)
        self.assertEqual(results[False]['use_extrinsic_guess'], False)

    def test_options_change_only_the_declared_flag_and_reset_rng(self):
        points, target, camera, initial_rvec, initial_tvec, *_ = _synthetic_packet()
        original_solve = cv2.solvePnPRansac
        calls = []
        seeds = []

        def observe_solve(*args, **kwargs):
            calls.append((
                tuple(np.array(value, copy=True) if isinstance(value, np.ndarray) else value for value in args),
                {key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
                 for key, value in kwargs.items()},
            ))
            return original_solve(*args, **kwargs)

        original_set_seed = cv2.setRNGSeed

        def observe_seed(value):
            seeds.append(int(value))
            return original_set_seed(value)

        source_copies = [value.copy() for value in (points, target, camera, initial_rvec, initial_tvec)]
        with mock.patch.object(cv2, 'solvePnPRansac', side_effect=observe_solve), \
                mock.patch.object(cv2, 'setRNGSeed', side_effect=observe_seed):
            replay_solver_inputs(points, target, camera, initial_rvec, initial_tvec,
                                 123456, True, cv2_module=cv2)
            replay_solver_inputs(points, target, camera, initial_rvec, initial_tvec,
                                 123456, False, cv2_module=cv2)

        self.assertEqual(seeds, [123456, 123456])
        self.assertEqual(len(calls), 2)
        expected_names = {'rvec', 'tvec', 'useExtrinsicGuess', 'iterationsCount',
                          'reprojectionError', 'confidence', 'flags'}
        for _, kwargs in calls:
            self.assertEqual(set(kwargs), expected_names)
            self.assertEqual(kwargs['iterationsCount'], 3000)
            self.assertEqual(kwargs['reprojectionError'], 2.)
            self.assertEqual(kwargs['confidence'], .999)
            self.assertEqual(kwargs['flags'], cv2.SOLVEPNP_ITERATIVE)
        self.assertEqual(calls[0][1]['useExtrinsicGuess'], True)
        self.assertEqual(calls[1][1]['useExtrinsicGuess'], False)
        for key in expected_names - {'useExtrinsicGuess'}:
            left, right = calls[0][1][key], calls[1][1][key]
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                self.assertEqual(left, right)
        for original, unchanged in zip((points, target, camera, initial_rvec, initial_tvec), source_copies):
            np.testing.assert_array_equal(original, unchanged)

    def test_report_keeps_captured_initial_vectors_when_solver_mutates_inputs(self):
        points, target, camera, initial_rvec, initial_tvec, *_ = _synthetic_packet()
        expected_rvec = initial_rvec.reshape(3).copy()
        expected_tvec = initial_tvec.reshape(3).copy()

        class MutatingCv2:
            SOLVEPNP_ITERATIVE = cv2.SOLVEPNP_ITERATIVE
            error = cv2.error

            @staticmethod
            def setRNGSeed(_):
                pass

            @staticmethod
            def solvePnPRansac(obj, image, k, distortion, **kwargs):
                kwargs['rvec'][...] = np.array([[.11], [.22], [.33]])
                kwargs['tvec'][...] = np.array([[2.], [-3.], [430.]])
                return True, kwargs['rvec'], kwargs['tvec'], np.arange(12, dtype=np.int32).reshape(-1, 1)

            @staticmethod
            def solvePnPRefineLM(_obj, _image, _k, _distortion, rvec, tvec):
                return rvec, tvec

            @staticmethod
            def Rodrigues(*args, **kwargs):
                return cv2.Rodrigues(*args, **kwargs)

            @staticmethod
            def projectPoints(*args, **kwargs):
                return cv2.projectPoints(*args, **kwargs)

        result = replay_solver_inputs(
            points, target, camera, initial_rvec, initial_tvec,
            rng_seed=99, use_extrinsic_guess=True, cv2_module=MutatingCv2,
        )
        np.testing.assert_array_equal(result['initial_rvec'], expected_rvec)
        np.testing.assert_array_equal(result['initial_tvec_mm'], expected_tvec)
        np.testing.assert_array_equal(initial_rvec.reshape(3), expected_rvec)
        np.testing.assert_array_equal(initial_tvec.reshape(3), expected_tvec)

    def test_capture_hook_snapshots_exact_solver_arrays_and_iteration_seed(self):
        writer = mock.Mock()
        refiner = PacketCapturingRefiner(
            None, None, 'mug', 'cpu', pnp_use_extrinsic_guess=True,
            packet_writer=writer,
        )
        refiner._capture_context = dict(
            frame_id=933, rgb_sha256='rgb-hash', mask_sha256='mask-hash',
            seed_pose_sha256='seed-hash',
        )
        refiner._capture_iteration = 2
        arrays = (
            np.arange(24, dtype=np.float64).reshape(8, 3),
            np.arange(16, dtype=np.float64).reshape(8, 2),
            np.array([[700., 0., 140.], [0., 710., 140.], [0., 0., 1.]]),
            np.array([[.1], [.2], [-.3]]),
            np.array([[1.], [2.], [430.]]),
        )
        with mock.patch.object(GoTrackRefiner, '_solve_pnp_ransac', return_value='solver-result') as base:
            result = refiner._solve_pnp_ransac(*arrays)

        self.assertEqual(result, 'solver-result')
        call = writer.call_args.kwargs
        self.assertEqual(call['context']['frame_id'], 933)
        self.assertEqual(call['iteration'], 2)
        self.assertEqual(call['rng_seed'], (933 * 17 + 2) & 0x7fffffff)
        self.assertEqual(len(base.call_args.args), len(arrays))
        for forwarded, original in zip(base.call_args.args, arrays):
            self.assertIs(forwarded, original)
        for captured, original in zip(call['arrays'].values(), arrays):
            np.testing.assert_array_equal(captured, original)
            self.assertFalse(np.shares_memory(captured, original))

    def test_packet_writer_records_hashes_and_refuses_packet_replacement(self):
        points, target, camera, rvec, tvec, *_ = _synthetic_packet()
        arrays = dict(obj_points=points[:32], target_points=target[:32],
                      crop_k=camera, rvec=rvec, tvec=tvec)
        context = dict(frame_id=827, rgb_sha256='rgb', mask_sha256='mask',
                       seed_pose_sha256='seed')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = dict(captured_packet_bytes=0, packets=[])
            writer = _packet_writer(root, record)
            writer(context=context, iteration=0, rng_seed=14059, arrays=arrays)
            entry = record['packets'][0]
            packet = root / entry['path']
            self.assertEqual(packet.stat().st_size, entry['bytes'])
            self.assertEqual(entry['input_hashes'], dict(rgb_sha256='rgb', mask_sha256='mask',
                                                          seed_pose_sha256='seed'))
            with np.load(packet, allow_pickle=False) as archive:
                for name, expected in arrays.items():
                    np.testing.assert_array_equal(archive[name], expected)
                np.testing.assert_array_equal(archive['point_indices'], np.arange(32, dtype=np.int32))
                self.assertEqual(int(archive['frame_id']), 827)
                self.assertEqual(int(archive['iteration']), 0)
                self.assertEqual(int(archive['rng_seed']), 14059)
            with self.assertRaises(FileExistsError):
                writer(context=context, iteration=0, rng_seed=14059, arrays=arrays)
            self.assertEqual(len(record['packets']), 1)

    def test_both_entrypoints_refuse_existing_output_without_changing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'existing-result'
            output.mkdir()
            sentinel = output / 'preserve.txt'
            sentinel.write_text('original bytes', encoding='utf-8')
            for create_root in (_create_probe_output_root, _create_replay_output_root):
                with self.assertRaises(FileExistsError):
                    create_root(output)
                self.assertEqual(sentinel.read_text(encoding='utf-8'), 'original bytes')


if __name__ == '__main__':
    unittest.main()
