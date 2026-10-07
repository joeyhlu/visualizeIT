import copy
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_pnp_guess_evaluate as evaluator


class PnPGuessEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.requested = list(range(10, 20))
        identity = np.eye(4).tolist()
        self.data = dict(
            object='keyboard',
            requested_frame_ids=self.requested.copy(),
            frames=[dict(frame_id=frame_id, branches={
                'control': dict(current_image_validated=True, pose=copy.deepcopy(identity)),
                'no_guess': dict(current_image_validated=True, pose=copy.deepcopy(identity)),
            }) for frame_id in self.requested],
            settings={'iterations': 5},
        )
        self.original = dict(
            name='keyboard',
            referenceFrames=[dict(frameId=frame_id, cameraFromObject=copy.deepcopy(identity))
                             for frame_id in self.requested],
            cameraCalibration=np.eye(3).tolist(), nativeResolution=[640, 480],
        )
        self.calls = []

    def fake_agreement(self, results, *_args):
        frames = results['frames']
        accepted = [frame for frame in frames if frame['pose_state'] == 'tracking'
                    and frame['cameraFromObject'] is not None]
        self.calls.append(copy.deepcopy(frames))
        visible_samples = len(frames) * 10
        accepted_samples = len(accepted) * 10
        return dict(processed=len(frames), accepted=len(accepted), reference_frames=len(frames),
                    reference_visible_samples=visible_samples, accepted_point_samples=accepted_samples,
                    sample_availability=accepted_samples / visible_samples if visible_samples else None,
                    median_720=float(len(accepted)) if accepted else None,
                    p95_720=float(len(accepted) + 1) if accepted else None, details=[])

    def build(self, data=None):
        with patch.object(evaluator, 'agreement', side_effect=self.fake_agreement):
            return evaluator.build_report(self.data if data is None else data, 'keyboard', self.original, object(),
                                          np.zeros((1, 3)), np.ones((1, 3)))

    def test_asymmetric_failures_keep_common_comparison_and_add_full_source_accounting(self):
        self.data['frames'][1]['branches']['control']['current_image_validated'] = False
        self.data['frames'][1]['branches']['control']['pose'] = None
        self.data['frames'][3]['branches']['control']['current_image_validated'] = False
        self.data['frames'][5]['branches']['no_guess']['current_image_validated'] = False
        report = self.build()

        self.assertEqual(report['record_status'], 'complete')
        self.assertEqual(report['recorded'], 10)
        self.assertEqual(report['common_validated_frames'], 7)
        self.assertEqual(report['branch_current_image_validated'], {'control': 8, 'no_guess': 9})
        self.assertEqual(report['all_source_frame_availability']['control'], dict(
            requested=10, accepted=8, failed=2, availability=.8, failed_frame_ids=[11, 13]))
        self.assertEqual(report['all_source_frame_availability']['no_guess'], dict(
            requested=10, accepted=9, failed=1, availability=.9, failed_frame_ids=[15]))
        self.assertEqual(report['secondary_surface_agreement']['control']['processed'], 7)
        self.assertEqual(report['secondary_surface_agreement']['no_guess']['processed'], 7)
        self.assertEqual(report['own_accepted_surface_agreement']['control']['processed'], 10)
        self.assertEqual(report['own_accepted_surface_agreement']['control']['accepted'], 8)
        self.assertEqual(report['own_accepted_surface_agreement']['no_guess']['processed'], 10)
        self.assertEqual(report['own_accepted_surface_agreement']['no_guess']['accepted'], 9)

        own_calls = [frames for frames in self.calls if len(frames) == 10]
        self.assertEqual(len(own_calls), 2)
        for frames in own_calls:
            self.assertEqual([frame['frameId'] for frame in frames], self.requested)
            for frame in frames:
                if frame['pose_state'] == 'lost':
                    self.assertIsNone(frame['cameraFromObject'])
        self.assertFalse(report['independent_accuracy_verified'])
        self.assertFalse(report['full_window_benchmark'])

    def test_both_branches_can_fail_without_empty_surface_scoring_or_false_complete_state(self):
        for index, frame in enumerate(self.data['frames']):
            frame['branches']['control']['current_image_validated'] = False
            frame['branches']['no_guess']['current_image_validated'] = False
            if index % 2:
                frame['branches']['control']['pose'] = None
                frame['branches']['no_guess']['pose'] = None
        report = self.build()

        self.assertEqual(report['record_status'], 'complete')
        self.assertEqual(report['common_validated_frames'], 0)
        self.assertEqual(report['branch_current_image_validated'], {'control': 0, 'no_guess': 0})
        self.assertEqual(report['all_source_frame_availability']['control']['availability'], 0)
        self.assertEqual(report['all_source_frame_availability']['no_guess']['availability'], 0)
        self.assertEqual(report['secondary_surface_agreement']['control'], evaluator._empty_surface_summary())
        for name in evaluator.BRANCHES:
            summary = report['own_accepted_surface_agreement'][name]
            self.assertEqual(summary['processed'], 10)
            self.assertEqual(summary['accepted'], 0)
            self.assertIsNone(summary['median_720'])
            self.assertIsNone(summary['p95_720'])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(len(frames) == 10 for frames in self.calls))
        self.assertTrue(all(frame['cameraFromObject'] is None for frames in self.calls for frame in frames))

    def test_partial_record_is_rejected_as_incomplete(self):
        partial = copy.deepcopy(self.data)
        partial['frames'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete diagnostic record'):
            evaluator._validate_record(partial)

    def test_finalization_fields_reject_running_failed_and_false_completion(self):
        for status in ('running', 'failed'):
            with self.subTest(status=status):
                record = copy.deepcopy(self.data)
                record['status'] = status
                with self.assertRaisesRegex(ValueError, "status must be 'complete'"):
                    evaluator._validate_record(record)

        false_completion = copy.deepcopy(self.data)
        false_completion['complete'] = False
        with self.assertRaisesRegex(ValueError, 'complete field must be true'):
            evaluator._validate_record(false_completion)

        integer_completion = copy.deepcopy(self.data)
        integer_completion['complete'] = 1
        with self.assertRaisesRegex(ValueError, 'complete field must be true'):
            evaluator._validate_record(integer_completion)

        completed = copy.deepcopy(self.data)
        completed['status'] = 'complete'
        completed['complete'] = True
        self.assertEqual(len(evaluator._validate_record(completed)), 10)
        # Pre-status legacy diagnostics remain accepted.
        self.assertEqual(len(evaluator._validate_record(self.data)), 10)

    def test_duplicate_frame_id_is_rejected(self):
        duplicate = copy.deepcopy(self.data)
        duplicate['frames'][1]['frame_id'] = duplicate['frames'][0]['frame_id']
        with self.assertRaisesRegex(ValueError, 'frame IDs must be unique'):
            evaluator._validate_record(duplicate)

    def test_requested_frame_ids_must_be_unique_and_match_record_order(self):
        duplicate_requested = copy.deepcopy(self.data)
        duplicate_requested['requested_frame_ids'][1] = duplicate_requested['requested_frame_ids'][0]
        with self.assertRaisesRegex(ValueError, 'Requested frame IDs must be unique'):
            evaluator._validate_record(duplicate_requested)

        reordered = copy.deepcopy(self.data)
        reordered['frames'][0], reordered['frames'][1] = reordered['frames'][1], reordered['frames'][0]
        with self.assertRaisesRegex(ValueError, 'requested IDs in requested order'):
            evaluator._validate_record(reordered)

    def test_branch_keys_and_accepted_pose_are_validated(self):
        bad_branches = copy.deepcopy(self.data)
        bad_branches['frames'][0]['branches'].pop('no_guess')
        with self.assertRaisesRegex(ValueError, 'exactly the branches'):
            evaluator._validate_record(bad_branches)

        missing_pose = copy.deepcopy(self.data)
        missing_pose['frames'][0]['branches']['control']['pose'] = None
        with self.assertRaisesRegex(ValueError, 'has no public pose'):
            evaluator._validate_record(missing_pose)

        invalid_pose = copy.deepcopy(self.data)
        invalid_pose['frames'][0]['branches']['control']['pose'][0][0] = 2
        with self.assertRaisesRegex(ValueError, 'has an invalid pose'):
            evaluator._validate_record(invalid_pose)

    def test_record_and_reference_objects_must_match_the_report_object(self):
        wrong_input = copy.deepcopy(self.data)
        wrong_input['object'] = 'mug'
        with self.assertRaisesRegex(ValueError, 'does not match requested object'):
            self.build(wrong_input)

        with self.assertRaisesRegex(ValueError, 'Reference object .* does not match'):
            evaluator.build_report(self.data, 'keyboard', dict(self.original, name='mug'), object(),
                                   np.zeros((1, 3)), np.ones((1, 3)))


if __name__ == '__main__':
    unittest.main()
