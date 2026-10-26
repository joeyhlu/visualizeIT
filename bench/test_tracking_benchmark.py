"""Accuracy, isolation and failure-accounting checks for portable tracking evals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import tracking_benchmark as benchmark
from .prototype_benchmark import digest, write_json


POINTS = np.array([[-.02, -.02, -.02], [.02, -.02, -.02],
                   [.02, .02, -.02], [-.02, .02, .02]], dtype=float)
K = np.array([[500., 0., 160.], [0., 500., 120.], [0., 0., 1.]])


def frame(index, visibility=None):
    record = dict(frameId=index, logicalIndex=index,
                  calibration=dict(cam_K=K.ravel().tolist(), depth_scale=1.),
                  rgb=dict(path='rgb.png', sha256='0' * 64))
    if visibility is not None:
        record.update(evaluationPose=dict(cam_R_m2c=np.eye(3).ravel().tolist(),
                                         cam_t_m2c=[0., 0., 500.]),
                      evaluationInfo=dict(visib_fract=visibility))
    return record


def prediction(index, public=True, offset=0.):
    pose = np.eye(4)
    pose[:3, 3] = [offset, 0., .5]
    return dict(frameId=index, state='tracking' if public else 'lost',
                failureReason=None if public else 'test_loss',
                cvCameraFromSourceObject=pose.tolist() if public else None)


def manifest(scored=3, count=1):
    return dict(schemaVersion=1, benchmarkId='unit-test-v1', dataset='unit-test',
                evidenceKind='synthetic_test_fixture', evaluatorManifestSHA256='0' * 64,
                objects=[dict(objectId=i + 1, name=f'sequence-{i + 1}',
                              model=dict(path='object.ply', sha256='0' * 64),
                              onboarding=[frame(j, 1.) for j in range(10)],
                              frames=[frame(j + 10) for j in range(scored)])
                         for i in range(count)])


class ManifestAndResourceTests(unittest.TestCase):
    def test_duplicate_systems_and_nonpositive_sequence_limit_rejected(self):
        for kwargs in (dict(systems=['current_rigid', 'current_rigid']),
                       dict(limit=0), dict(limit=-1)):
            with tempfile.TemporaryDirectory() as temp, self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, 'Invalid benchmark configuration'):
                    benchmark.run_inference(Path(temp) / 'not-needed', Path(temp) / 'output', **kwargs)

    def test_variable_length_not_110_hardcoded(self):
        for count in (1, 7, 111, 301):
            with self.subTest(count=count):
                benchmark.validate_manifest(manifest(count))

    def test_each_forbidden_label_rejected_in_scored_rgb_record(self):
        for key in benchmark.FORBIDDEN:
            value = manifest()
            value['objects'][0]['frames'][0][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.validate_manifest(value)

    def test_unknown_input_field_rejected(self):
        value = manifest()
        value['objects'][0]['frames'][0]['futurePoseHint'] = []
        with self.assertRaises(ValueError):
            benchmark.validate_manifest(value)

    def test_first_ten_authorized_pose_records_only(self):
        benchmark.validate_manifest(manifest())
        for count in (0, 9, 11):
            value = manifest()
            value['objects'][0]['onboarding'] = [frame(j, 1.) for j in range(count)]
            with self.subTest(count=count), self.assertRaises(ValueError):
                benchmark.validate_manifest(value)

    def test_duplicate_id_and_reordering_rejected(self):
        value = manifest(count=2)
        value['objects'][1]['objectId'] = value['objects'][0]['objectId']
        with self.assertRaises(ValueError):
            benchmark.validate_manifest(value)
        value = manifest()
        value['objects'][0]['frames'].reverse()
        with self.assertRaises(ValueError):
            benchmark.validate_manifest(value)

    def test_invalid_intrinsics_rejected(self):
        for bad in (float('nan'), 0., -1.):
            value = manifest()
            value['objects'][0]['frames'][0]['calibration']['cam_K'][0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                benchmark.validate_manifest(value)

    def test_resource_escape_symlink_and_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / 'data'; root.mkdir()
            outside = parent / 'outside'; outside.write_text('outside')
            inside = root / 'inside'; inside.write_text('inside')
            spec = dict(path='inside', sha256=digest(inside))
            self.assertEqual(benchmark.checked_path(root, spec), inside)
            with self.assertRaises(ValueError):
                benchmark.checked_path(root, dict(path='../outside', sha256=digest(outside)))
            (root / 'link').symlink_to(outside)
            with self.assertRaises(ValueError):
                benchmark.checked_path(root, dict(path='link', sha256=digest(outside)))
            with self.assertRaises(ValueError):
                benchmark.checked_path(root, dict(path='inside', sha256='0' * 64))
            with self.assertRaises(ValueError):
                benchmark.checked_path(root, dict(**spec, extra='no'))


class ScoringTests(unittest.TestCase):
    def test_identity_pose_has_zero_reference_error(self):
        summary, details = benchmark.score_rows([frame(10, 1.)], [prediction(10)], POINTS)
        self.assertEqual(summary['accurateVisibleFrames'], 1)
        self.assertAlmostEqual(summary['reprojectionPixelsMedian'], 0.)
        self.assertAlmostEqual(summary['addMetresMedian'], 0.)
        self.assertAlmostEqual(summary['rotationDegreesMedian'], 0.)
        self.assertAlmostEqual(summary['translationMetresMedian'], 0.)
        self.assertTrue(details[0]['accurate'])

    def test_known_translation_has_analytic_add_and_projection_error(self):
        summary, _ = benchmark.score_rows([frame(10, 1.)], [prediction(10, offset=.02)], POINTS)
        expected_pixels = float(np.mean(500. * .02 / (POINTS[:, 2] + .5)))
        self.assertAlmostEqual(summary['addMetresMedian'], .02)
        self.assertAlmostEqual(summary['translationMetresMedian'], .02)
        self.assertAlmostEqual(summary['reprojectionPixelsMedian'], expected_pixels)
        self.assertEqual(summary['accurateVisibleFrames'], 0)

    def test_missing_reordered_extra_rows_rejected(self):
        records = [frame(10, 1.), frame(11, 1.)]
        for rows in ([prediction(10)], [prediction(11), prediction(10)],
                     [prediction(10), prediction(11), prediction(12)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                benchmark.score_rows(records, rows, POINTS)

    def test_lost_frames_retained_and_hidden_public_pose_counted(self):
        records = [frame(10, 1.), frame(11, 1.), frame(12, 0.), frame(13, 1.)]
        rows = [prediction(10), prediction(11, public=False), prediction(12), prediction(13)]
        summary, _ = benchmark.score_rows(records, rows, POINTS)
        self.assertEqual(summary['frames'], 4)
        self.assertEqual(summary['visibleFrames'], 3)
        self.assertEqual(summary['publicTrackingFrames'], 3)
        self.assertEqual(summary['trackingAvailability'], .75)
        self.assertAlmostEqual(summary['visibleTrackingAvailability'], 2 / 3)
        self.assertAlmostEqual(summary['accurateVisibleAvailability'], 2 / 3)
        self.assertEqual(summary['hiddenPublicPoseFrames'], 1)
        self.assertEqual(summary['failures']['test_loss'], 1)
        self.assertEqual(summary['longestAccurateVisibleRun'], 1)

    def test_all_lost_uses_null_not_fake_zero_conditional_error(self):
        summary, _ = benchmark.score_rows([frame(10, 1.)], [prediction(10, public=False)], POINTS)
        self.assertEqual(summary['accurateVisibleAvailability'], 0.)
        self.assertEqual(summary['trackingAvailability'], 0.)
        self.assertIsNone(summary['reprojectionPixelsMedian'])

    def test_stale_nontracking_pose_and_improper_public_pose_rejected(self):
        row = prediction(10)
        row['state'] = 'lost'
        with self.assertRaises(ValueError):
            benchmark.score_rows([frame(10, 1.)], [row], POINTS)
        row = prediction(10)
        row['cvCameraFromSourceObject'][0][0] = -1.
        with self.assertRaises(ValueError):
            benchmark.score_rows([frame(10, 1.)], [row], POINTS)

    def test_two_consecutive_accurate_visible_outputs_required_for_recovery(self):
        records = [frame(i, v) for i, v in enumerate([1., 0., 1., 1., 1.], 10)]
        rows = [prediction(10), prediction(11, public=False), prediction(12),
                prediction(13, public=False), prediction(14)]
        summary, _ = benchmark.score_rows(records, rows, POINTS)
        self.assertEqual(len(summary['recovery']), 1)
        self.assertIsNone(summary['recovery'][0]['observableFramesToTwoAccurate'])
        rows[3] = prediction(13)
        summary, _ = benchmark.score_rows(records, rows, POINTS)
        self.assertEqual(summary['recovery'][0]['observableFramesToTwoAccurate'], 2)

    def test_gradual_reappearance_still_creates_recovery_episode(self):
        records = [frame(i, v) for i, v in enumerate([1., 0., .2, .7, 1.], 10)]
        rows = [prediction(10), prediction(11, public=False), prediction(12, public=False),
                prediction(13), prediction(14)]
        summary, _ = benchmark.score_rows(records, rows, POINTS)
        self.assertEqual(len(summary['recovery']), 1)
        self.assertEqual(summary['recovery'][0]['observableFramesToTwoAccurate'], 2)

    def test_recovery_cannot_be_borrowed_from_later_occlusion_episode(self):
        records = [frame(i, v) for i, v in enumerate([0., 1., 0., 1., 1.], 10)]
        rows = [prediction(10, public=False), prediction(11), prediction(12, public=False),
                prediction(13), prediction(14)]
        summary, _ = benchmark.score_rows(records, rows, POINTS)
        self.assertEqual(len(summary['recovery']), 2)
        self.assertIsNone(summary['recovery'][0]['observableFramesToTwoAccurate'])
        self.assertEqual(summary['recovery'][1]['observableFramesToTwoAccurate'], 2)


class InferenceIsolationTests(unittest.TestCase):
    def fixture(self, root, objects=1):
        value = manifest(scored=2, count=objects)
        model = root / 'object.ply'; model.write_text('patched model fixture')
        for obj in value['objects']:
            obj['model']['sha256'] = digest(model)
        evaluator = dict(schemaVersion=1, benchmarkId=value['benchmarkId'], dataset=value['dataset'],
                         objects=[dict(objectId=obj['objectId'], name=obj['name'], model=obj['model'],
                                       frames=[frame(r['frameId'], 1.) for r in obj['frames']])
                                  for obj in value['objects']])
        write_json(root / 'evaluator.json', evaluator)
        value['evaluatorManifestSHA256'] = digest(root / 'evaluator.json')
        write_json(root / 'inference.json', value)
        bank = (np.tile(POINTS, (4, 1)), np.ones((16, 32), dtype=np.uint8),
                np.zeros((8, 8), dtype=np.uint8), np.ones((16, 2)), np.tile(POINTS, (4, 1)))
        return value, bank

    def test_inference_never_reads_evaluator_and_seals_every_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / 'dataset'; dataset.mkdir()
            _, bank = self.fixture(dataset)
            real_json_read = benchmark.json_read
            opened = []
            def guarded(path):
                opened.append(Path(path).name)
                if Path(path).name == 'evaluator.json':
                    raise AssertionError('Inference read evaluator labels')
                return real_json_read(path)
            with patch.object(benchmark, 'json_read', side_effect=guarded), \
                 patch.object(benchmark, 'read_ply', return_value=(None, POINTS, None)), \
                 patch.object(benchmark, 'build_depth_bank', return_value=bank), \
                 patch.object(benchmark, 'load_rgb', return_value=np.zeros((8, 8, 3), dtype=np.uint8)), \
                 patch.object(benchmark, 'make_tracker', return_value=object()), \
                 patch.object(benchmark, 'update_tracker', side_effect=lambda s,t,r,*a: prediction(r['frameId'])):
                benchmark.run_inference(dataset, root / 'output', systems=['current_rigid'], repeats=1, warmup=0)
            self.assertNotIn('evaluator.json', opened)
            seal = real_json_read(root / 'output' / 'inference-seal.json')
            self.assertFalse(seal['evaluatorOpenedDuringInference'])
            self.assertEqual(len(seal['sequences']), 1)
            self.assertEqual(len(seal['trialHashes']), 1)
            relative = next(iter(seal['trialHashes']))
            self.assertEqual(len(real_json_read(root / 'output' / relative)['frames']), 2)
            # Sealed predictions cannot be edited and then accepted by evaluation.
            with (root / 'output' / relative).open('a') as output:
                output.write(' ')
            with self.assertRaisesRegex(ValueError, 'Prediction tampered'):
                benchmark.run_evaluation(dataset, root / 'output')

    def test_decode_failure_preserves_denominator_and_does_not_count_as_timed_update(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / 'dataset'; dataset.mkdir()
            _, bank = self.fixture(dataset)
            class Tracker:
                def reset_temporal(self):
                    pass
            def decode(_, record):
                if record['frameId'] == 11:
                    raise ValueError('hash mismatch test')
                return np.zeros((8, 8, 3), dtype=np.uint8)
            with patch.object(benchmark, 'read_ply', return_value=(None, POINTS, None)), \
                 patch.object(benchmark, 'build_depth_bank', return_value=bank), \
                 patch.object(benchmark, 'load_rgb', side_effect=decode), \
                 patch.object(benchmark, 'make_tracker', return_value=Tracker()), \
                 patch.object(benchmark, 'update_tracker', side_effect=lambda s,t,r,*a: prediction(r['frameId'])):
                benchmark.run_inference(dataset, root / 'output', systems=['current_rigid'], repeats=1, warmup=0)
                benchmark.run_evaluation(dataset, root / 'output')
            results = benchmark.json_read(root / 'output' / 'results.json')
            total = results['aggregate']['current_rigid']
            self.assertEqual(total['framesWithRepeats'], 2)
            self.assertEqual(total['timedUpdateFrames'], 1)
            self.assertEqual(total['skippedUpdateFrames'], 1)
            self.assertEqual(total['trackingAvailability'], .5)
            self.assertEqual(total['accurateVisibleAvailability'], .5)

    def test_setup_failure_is_retained_and_has_no_misleading_speed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / 'dataset'; dataset.mkdir()
            _, bank = self.fixture(dataset)
            with patch.object(benchmark, 'read_ply', return_value=(None, POINTS, None)), \
                 patch.object(benchmark, 'build_depth_bank', return_value=bank), \
                 patch.object(benchmark, 'load_rgb', return_value=np.zeros((8, 8, 3), dtype=np.uint8)), \
                 patch.object(benchmark, 'make_tracker', side_effect=RuntimeError('setup failed test')):
                benchmark.run_inference(dataset, root / 'output', systems=['current_rigid'], repeats=1, warmup=0)
                benchmark.run_evaluation(dataset, root / 'output')
            results = benchmark.json_read(root / 'output' / 'results.json')
            total = results['aggregate']['current_rigid']
            self.assertEqual(total['framesWithRepeats'], 2)
            self.assertEqual(total['timedUpdateFrames'], 0)
            self.assertEqual(total['skippedUpdateFrames'], 2)
            self.assertEqual(total['trackingAvailability'], 0.)
            self.assertIsNone(total['updateFramesPerSecond'])
            self.assertIsNone(total['replayFramesPerSecond'])
            self.assertIsNone(total['perSequenceMedianUpdateMs'])
            self.assertIn('setup failed test', results['sequenceTrials'][0]['setupFailure'])

    def test_mixed_initialization_failure_does_not_inflate_fps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / 'dataset'; dataset.mkdir()
            _, bank = self.fixture(dataset, objects=2)
            def initialize(_, obj):
                if obj['objectId'] == 2:
                    raise ValueError('insufficient initialization test')
                return bank
            with patch.object(benchmark, 'read_ply', return_value=(None, POINTS, None)), \
                 patch.object(benchmark, 'build_depth_bank', side_effect=initialize), \
                 patch.object(benchmark, 'load_rgb', return_value=np.zeros((8, 8, 3), dtype=np.uint8)), \
                 patch.object(benchmark, 'make_tracker', return_value=object()), \
                 patch.object(benchmark, 'update_tracker', side_effect=lambda s,t,r,*a: prediction(r['frameId'])):
                benchmark.run_inference(dataset, root / 'output', systems=['current_rigid'], repeats=1, warmup=0)
                benchmark.run_evaluation(dataset, root / 'output')
            results = benchmark.json_read(root / 'output' / 'results.json')
            total = results['aggregate']['current_rigid']
            attempted = results['sequenceTrials'][0]
            self.assertEqual(total['framesWithRepeats'], 4)
            self.assertEqual(total['timedUpdateFrames'], 2)
            self.assertEqual(total['initializationFailureSequences'], 1)
            self.assertEqual(total['trackingAvailability'], .5)
            self.assertAlmostEqual(total['updateFramesPerSecond'], 2 / attempted['updateTotalSeconds'])

    def test_changed_source_invalidates_inference_and_leaves_no_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / 'dataset'; dataset.mkdir()
            _, bank = self.fixture(dataset)
            with patch.object(benchmark, 'read_ply', return_value=(None, POINTS, None)), \
                 patch.object(benchmark, 'build_depth_bank', return_value=bank), \
                 patch.object(benchmark, 'load_rgb', return_value=np.zeros((8, 8, 3), dtype=np.uint8)), \
                 patch.object(benchmark, 'make_tracker', return_value=object()), \
                 patch.object(benchmark, 'source_hashes', side_effect=[{'code': 'before'}, {'code': 'after'}]), \
                 patch.object(benchmark, 'update_tracker', side_effect=lambda s,t,r,*a: prediction(r['frameId'])):
                with self.assertRaisesRegex(ValueError, 'Source/input changed'):
                    benchmark.run_inference(dataset, root / 'output', systems=['current_rigid'], repeats=1, warmup=0)
            self.assertFalse((root / 'output' / 'inference-seal.json').exists())

    def test_evaluation_rejects_different_harness_source(self):
        with patch.object(benchmark, 'json_read', return_value=dict(sourceHashes={'different': 'hash'})):
            with self.assertRaisesRegex(ValueError, 'Benchmark source differs'):
                benchmark.run_evaluation('not-opened', 'not-opened')

    def test_duplicate_evaluator_identity_rejected_even_when_hash_is_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = root / 'dataset'; dataset.mkdir()
            value, _ = self.fixture(dataset)
            evaluator = benchmark.json_read(dataset / 'evaluator.json')
            evaluator['objects'].append(copy.deepcopy(evaluator['objects'][0]))
            write_json(dataset / 'evaluator.json', evaluator)
            output = root / 'output'; output.mkdir()
            write_json(output / 'freeze.json', dict(test='frozen'))
            write_json(output / 'inference-seal.json', dict(
                sourceHashes=benchmark.source_hashes(), freezeSHA256=digest(output / 'freeze.json'),
                trialHashes={}, evaluatorSHA256=digest(dataset / 'evaluator.json'),
                benchmarkId=value['benchmarkId']))
            with self.assertRaisesRegex(ValueError, 'Duplicate evaluator sequence identity'):
                benchmark.run_evaluation(dataset, output)


if __name__ == '__main__':
    unittest.main()
