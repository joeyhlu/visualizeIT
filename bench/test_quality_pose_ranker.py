"""CPU tests for the optional quality pose-candidate rankers."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .quality_contract import (
    Frame, MaskPrediction, PoseCandidate, SequentialTracker, ValidationSettings,
    distinct_pose, project,
)
from .quality_pose_ranker import (
    GeometricPoseRanker, LearnedPoseRanker, LegacyPoseRanker, RankCandidate,
    candidate_content_ids, make_ranker,
)
from .quality_ranker_features import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FEATURE_VECTOR_SHAPE,
    extract_features,
)


K = np.array([[500., 0., 160.], [0., 500., 120.], [0., 0., 1.]])
POSE = np.eye(4)
POSE[2, 3] = 1.
OTHER = POSE.copy()
OTHER[:3, :3] = np.diag([-1., -1., 1.])
xy = np.array([(x, y) for x in np.linspace(-.12, .12, 6)
               for y in np.linspace(-.12, .12, 6)])
XYZ = np.c_[xy, np.zeros(len(xy))]
MASK = np.zeros((240, 320), np.uint8)
MASK[30:211, 50:271] = 255


def frame(index):
    return Frame(index, np.zeros((240, 320, 3), np.uint8), K)


def segmentation(index):
    return MaskPrediction(index, MASK.copy())


def candidate(pose=POSE, *, source='refinement', stats_extra=None):
    return PoseCandidate(
        np.array(pose, dtype=np.float64, copy=True), XYZ.copy(),
        project(XYZ, pose, K)[0], np.ones(len(XYZ)), source=source,
        ranker_features=stats_extra)


def rank_candidate(value, candidate_id, score, index=0, features=None):
    stats = {
        'score': score, 'correspondences': 36, 'inliers': 36,
        'inlier_ratio': 1., 'mean_inlier_confidence': .9,
        'median_reprojection_720': .2, 'p95_reprojection_720': .5,
        'spatial_support': .7,
    }
    if features is None:
        features = extract_features(stats)
    return RankCandidate(candidate_id, index, value, stats, features,
                         value.ranker_features)


def linear_checkpoint(*, bias=0., acceptance=.5, margin=.05):
    width = FEATURE_VECTOR_SHAPE[0]
    return {
        'schema_version': 1,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'feature_names': list(FEATURE_NAMES),
        'mean': [0.] * width,
        'scale': [1.] * width,
        'layers': [{'weight': [[0.] * width], 'bias': [bias],
                    'activation': 'identity'}],
        'architecture': 'linear',
        'acceptance_probability': acceptance,
        'distinct_pose_margin_probability': margin,
        'provenance': {'kind': 'synthetic_test'},
    }


def trainer_checkpoint():
    """Small structurally valid payload matching the trainer's export schema."""
    digest = hashlib.sha256(b'test-identity').hexdigest()
    data = {name: digest for name in (
        'feature_cache_sha256', 'label_cache_sha256', 'split_sha256',
        'source_trace_sha256', 'reference_labels_sha256',
        'cache_manifest_sha256')}
    data.update(feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_names=list(FEATURE_NAMES))
    identity = {
        'manifest_sha256': digest,
        'data': data,
        'source': {name: digest for name in (
            'quality_ranker_train.py', 'quality_ranker_features.py',
            'quality_pose_ranker.py', 'quality_ranker_data.py',
            'quality_contract.py', 'quality_gotrack.py')},
        'config_sha256': digest,
        'feature_names': list(FEATURE_NAMES),
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'runtime': {'torch': '2.5.1', 'numpy': '2.0', 'python': '3.12',
                    'cuda': None, 'cudnn': None, 'device': 'cpu'},
    }
    payload = linear_checkpoint()
    payload['provenance'] = {
        'identity': identity,
        'checkpoint_sha256': digest,
        'calibration': {
            'temperature': 1., 'bias': 0.,
            'acceptance_probability': .5, 'validation_bce': .69,
            'validation_balanced_accuracy': .5,
        },
        'calibration_split': 'validation',
        'test_read': False,
    }
    return payload


def integration_checkpoint(*, bias=0.):
    digest = hashlib.sha256(b'integration-identity').hexdigest()
    data = {name: digest for name in (
        'feature_cache_sha256', 'label_cache_sha256', 'split_sha256',
        'source_trace_sha256', 'reference_labels_sha256',
        'cache_manifest_sha256')}
    data.update(feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_names=list(FEATURE_NAMES))
    identity = {
        'manifest_sha256': digest,
        'data': data,
        'source': {name: digest for name in (
            'quality_ranker_train.py', 'quality_ranker_features.py',
            'quality_pose_ranker.py', 'quality_ranker_data.py',
            'quality_contract.py', 'quality_gotrack.py')},
        'config_sha256': digest,
        'purpose': 'integration_smoke_v1',
        'feature_names': list(FEATURE_NAMES),
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'runtime': {'torch': '2.5.1', 'numpy': '2.0', 'python': '3.12',
                    'cuda': None, 'cudnn': None, 'device': 'cpu'},
    }
    return {
        'schema_version': 2,
        'artifact_purpose': 'integration_smoke_v1',
        'selection_enabled': False,
        'calibration_unavailable_reason': 'integration_smoke_does_not_fit_calibration',
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'feature_names': list(FEATURE_NAMES),
        'mean': [0.] * FEATURE_VECTOR_SHAPE[0],
        'scale': [1.] * FEATURE_VECTOR_SHAPE[0],
        'layers': [{'weight': [[0.] * FEATURE_VECTOR_SHAPE[0]], 'bias': [bias],
                    'activation': 'identity'}],
        'architecture': 'linear',
        'provenance': {
            'artifact_purpose': 'integration_smoke_v1',
            'identity': identity,
            'checkpoint_sha256': digest,
            'test_read': False,
        },
    }


class Backend:
    def __init__(self, *, output=None, hypotheses=()):
        self.output = output
        self.hypotheses = list(hypotheses)

    def refine(self, _frame, _mask, seed):
        if callable(self.output):
            return self.output(seed)
        return self.output

    def recover(self, _frame, _mask, top_k=5):
        return self.hypotheses[:top_k]


class PoseRankerTests(unittest.TestCase):
    def test_feature_vector_has_frozen_shape_and_missing_indicators(self):
        features = extract_features({
            'correspondences': 36, 'inliers': 27,
            'median_reprojection_720': 2., 'p95_reprojection_720': 5.,
            'spatial_support': .25, 'score': .6,
            'mean_inlier_confidence': .8,
            'frames_since_last_valid': 3,
        }, {'appearance_patch_cosine_00': .75,
            'appearance_valid_fraction': .5})
        self.assertEqual(len(FEATURE_NAMES), FEATURE_VECTOR_SHAPE[0])
        self.assertEqual(len(features), FEATURE_VECTOR_SHAPE[0])
        self.assertAlmostEqual(features[0], math.log1p(36))
        self.assertEqual(features[1], 1.)
        self.assertAlmostEqual(features[2], .75)
        self.assertEqual(features[3], 1.)
        self.assertEqual(features[4], .8)
        self.assertEqual(features[5], 1.)
        self.assertEqual(features[18], 0.)  # physical seconds are missing
        self.assertEqual(features[19], 0.)
        self.assertEqual(features[20], 3.)
        self.assertEqual(features[21], 1.)
        self.assertEqual(features[-2:], [.5, 1.])
        missing = extract_features({})
        self.assertTrue(all(value == 0. for value in missing))

    def test_legacy_stable_score_ties_and_recovery_ambiguity_match_old_rule(self):
        first = rank_candidate(candidate(POSE, source='first'), 'first', .8, 0)
        second = rank_candidate(candidate(OTHER, source='second'), 'second', .8, 1)
        decision = LegacyPoseRanker().rank({'recovery': False}, [first, second])
        self.assertEqual(decision.ordered_ids, ('first', 'second'))
        self.assertEqual(decision.selected_id, 'first')

        recovery = LegacyPoseRanker().rank({
            'recovery': True, 'recovery_ambiguity_ratio': .9,
            'recovery_rotation_degrees': 20., 'recovery_translation_m': .03,
        }, [first, second])
        old_ambiguous = (second.stats['score'] >= first.stats['score'] * .9 and
                         distinct_pose(first.pose, second.pose, ValidationSettings()))
        self.assertTrue(old_ambiguous)
        self.assertIsNone(recovery.selected_id)
        self.assertEqual(recovery.rejection_reason, 'ambiguous_recovery_hypotheses')

    def test_explicit_legacy_and_default_tracker_have_same_public_selection(self):
        class AmbiguousBackend(Backend):
            def __init__(self):
                super().__init__(hypotheses=[POSE.copy(), OTHER.copy()])

            def refine(self, _frame, _mask, seed):
                return candidate(seed)

        default = SequentialTracker(AmbiguousBackend())
        explicit = SequentialTracker(AmbiguousBackend(),
                                     pose_ranker=LegacyPoseRanker())
        old_result = default.update(frame(0), segmentation(0))
        rank_result = explicit.update(frame(0), segmentation(0))
        for key in old_result:
            if key == 'timings_ms':
                continue
            self.assertEqual(old_result[key], rank_result[key], key)

    def test_candidate_ids_and_geometric_selection_are_permutation_invariant(self):
        a = candidate(POSE, source='a')
        b = candidate(OTHER, source='b')
        ids = candidate_content_ids([a, b], stage='recovery')
        permuted_ids = candidate_content_ids([b, a], stage='recovery')
        self.assertEqual(ids[0], permuted_ids[1])
        self.assertEqual(ids[1], permuted_ids[0])
        rows = [rank_candidate(a, ids[0], .7, 0),
                rank_candidate(b, ids[1], .7, 1)]
        rows_permuted = [rank_candidate(b, permuted_ids[0], .7, 0),
                         rank_candidate(a, permuted_ids[1], .7, 1)]
        ranker = GeometricPoseRanker()
        result = ranker.rank({'recovery': True}, rows)
        result_permuted = ranker.rank({'recovery': True}, rows_permuted)
        self.assertEqual(result.ordered_ids, result_permuted.ordered_ids)
        self.assertEqual(result.selected_id, result_permuted.selected_id)

        # Summary timing is measurement metadata, not candidate content.
        timed = candidate(stats_extra={
            'appearance_patch_cosine_00': .8, 'appearance_summary_ms': 1.25})
        differently_timed = candidate(stats_extra={
            'appearance_patch_cosine_00': .8, 'appearance_summary_ms': 99.})
        self.assertEqual(candidate_content_ids([timed]),
                         candidate_content_ids([differently_timed]))

    def test_learned_forward_uses_logit_and_immutable_checkpoint_snapshot(self):
        document = linear_checkpoint()
        ranker = LearnedPoseRanker(document)
        features = [0.] * FEATURE_VECTOR_SHAPE[0]
        self.assertAlmostEqual(ranker.predict_logits(features), 0.)
        self.assertAlmostEqual(ranker.probability(features), .5)
        document['layers'][0]['bias'][0] = 2.
        self.assertAlmostEqual(ranker.predict_logits(features), 0.)
        shifted = LearnedPoseRanker(linear_checkpoint(bias=2.))
        self.assertAlmostEqual(shifted.probability(features), 1. / (1. + math.exp(-2.)))
        with self.assertRaises(ValueError):
            ranker.mean.setflags(write=True)

    def test_learned_absolute_and_distinct_pose_probability_margins_reject(self):
        features = [0.] * FEATURE_VECTOR_SHAPE[0]
        low = LearnedPoseRanker(linear_checkpoint(acceptance=.8))
        only = rank_candidate(candidate(), 'one', .9, features=features)
        rejected = low.rank({'recovery': False}, [only])
        self.assertIsNone(rejected.selected_id)
        self.assertEqual(rejected.rejection_reason,
                         'learned_acceptance_probability_below_threshold')

        confident = LearnedPoseRanker(linear_checkpoint(
            bias=3., acceptance=.8, margin=.1))
        first = rank_candidate(candidate(POSE), 'first', .9, 0, features)
        second = rank_candidate(candidate(OTHER), 'second', .8, 1, features)
        ambiguous = confident.rank({
            'recovery': True, 'recovery_rotation_degrees': 20.,
            'recovery_translation_m': .03,
        }, [first, second])
        self.assertIsNone(ambiguous.selected_id)
        self.assertEqual(
            ambiguous.rejection_reason,
            'learned_distinct_pose_probability_margin_below_threshold')

    def test_checkpoint_rejects_bad_shapes_scales_and_feature_order(self):
        bad = linear_checkpoint()
        bad['scale'][0] = 0.
        with self.assertRaises(ValueError):
            LearnedPoseRanker(bad)
        bad = linear_checkpoint()
        bad['feature_names'][0], bad['feature_names'][1] = (
            bad['feature_names'][1], bad['feature_names'][0])
        with self.assertRaises(ValueError):
            LearnedPoseRanker(bad)

    def test_runtime_factory_requires_validation_calibrated_trainer_provenance(self):
        # Trainer export schema includes validation calibration, fixed identity
        # hashes, and an explicit statement that test data was not read.
        ranker = make_ranker('learned', trainer_checkpoint())
        # The all-zero synthetic linear parameters produce a zero logit, whose
        # sigmoid is exactly one half by definition.
        self.assertEqual(ranker.probability([0.] * FEATURE_VECTOR_SHAPE[0]), .5)
        with self.assertRaisesRegex(ValueError, 'trainer provenance'):
            make_ranker('learned', linear_checkpoint())
        with self.assertRaisesRegex(ValueError, 'trainer provenance'):
            LearnedPoseRanker.from_checkpoint(linear_checkpoint())

        bad = trainer_checkpoint()
        bad['provenance']['test_read'] = True
        with self.assertRaisesRegex(ValueError, 'test data was not read'):
            LearnedPoseRanker.from_checkpoint(bad)
        bad = trainer_checkpoint()
        bad['provenance']['calibration_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation data'):
            make_ranker('learned', bad)
        bad = trainer_checkpoint()
        bad['provenance']['checkpoint_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            make_ranker('learned', bad)
        bad = linear_checkpoint()
        bad['layers'][0]['weight'] = [[0.]]
        with self.assertRaises(ValueError):
            LearnedPoseRanker(bad)

    def test_integration_v2_roundtrip_keeps_scores_but_never_selects(self):
        payload = integration_checkpoint()
        ranker = make_ranker('learned', payload)
        self.assertFalse(ranker.selection_enabled)
        self.assertEqual(ranker.artifact_purpose, 'integration_smoke_v1')
        self.assertEqual(ranker.calibration_unavailable_reason,
                         'integration_smoke_does_not_fit_calibration')
        self.assertNotIn('acceptance_probability', ranker.to_checkpoint())
        self.assertNotIn('distinct_pose_margin_probability', ranker.to_checkpoint())

        restored_payload = json.loads(json.dumps(ranker.to_checkpoint()))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_path = Path(temp) / 'integration-ranker.json'
            checkpoint_path.write_text(json.dumps(restored_payload), encoding='utf-8')
            restored = make_ranker('learned', checkpoint_path)
        self.assertFalse(restored.selection_enabled)
        self.assertEqual(restored.to_checkpoint(), restored_payload)
        with self.assertRaises(AttributeError):
            restored.selection_enabled = True
        first = rank_candidate(candidate(POSE), 'z', .5)
        second = rank_candidate(candidate(OTHER), 'a', .5)
        tied = restored.rank({'recovery': True}, [first, second])
        self.assertEqual(tied.ordered_ids, ('a', 'z'))
        self.assertIsNone(tied.selected_id)
        self.assertEqual(tied.rejection_reason, 'integration_only_selection_disabled')
        empty = restored.rank({}, [])
        self.assertEqual(empty.ordered_ids, ())
        self.assertIsNone(empty.selected_id)
        self.assertEqual(empty.rejection_reason, 'integration_only_selection_disabled')

        supplied = restored.rank_with_scores({}, [first, second], [1., 0.])
        self.assertEqual(supplied.ordered_ids, ('z', 'a'))
        self.assertEqual(dict(supplied.scores), {'z': 1., 'a': 0.})
        self.assertIsNone(supplied.selected_id)
        self.assertEqual(supplied.rejection_reason, 'integration_only_selection_disabled')
        for bias, expected_score in ((-1000., 0.), (1000., 1.)):
            extreme = make_ranker('learned', integration_checkpoint(bias=bias))
            self.assertEqual(extreme.probability([0.] * FEATURE_VECTOR_SHAPE[0]), expected_score)
            result = extreme.rank({}, [first])
            self.assertEqual(dict(result.scores), {'z': expected_score})
            self.assertIsNone(result.selected_id)
            self.assertEqual(result.rejection_reason,
                             'integration_only_selection_disabled')

    def test_integration_v2_rejects_missing_malformed_and_contradictory_contract_fields(self):
        mutations = []
        bad = integration_checkpoint()
        bad['artifact_purpose'] = 'production'
        mutations.append(bad)
        bad = integration_checkpoint()
        bad['selection_enabled'] = True
        mutations.append(bad)
        bad = integration_checkpoint()
        del bad['selection_enabled']
        mutations.append(bad)
        bad = integration_checkpoint()
        bad['calibration_unavailable_reason'] = ''
        mutations.append(bad)
        bad = integration_checkpoint()
        bad['provenance']['artifact_purpose'] = 'production'
        mutations.append(bad)
        bad = integration_checkpoint()
        bad['provenance']['identity']['purpose'] = 'substantive_v1'
        mutations.append(bad)
        bad = integration_checkpoint()
        bad['provenance']['kind'] = 'synthetic_test'
        mutations.append(bad)
        bad = integration_checkpoint()
        bad['acceptance_probability'] = 1.
        mutations.append(bad)
        for document in mutations:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    make_ranker('learned', document)

        # Production schema v1 retains the established exact fields and calibrated path.
        production = make_ranker('learned', trainer_checkpoint())
        roundtrip = production.to_checkpoint()
        self.assertEqual(set(roundtrip), {
            'schema_version', 'feature_schema_version', 'feature_names', 'mean',
            'scale', 'layers', 'architecture', 'acceptance_probability',
            'distinct_pose_margin_probability', 'provenance'})
        self.assertTrue(make_ranker('learned', json.loads(json.dumps(roundtrip))).selection_enabled)

    def test_gotrack_patch_summary_uses_masked_corresponding_cells(self):
        from .quality_gotrack import GoTrackRefiner

        features = np.ones((2, 3, 4, 4), dtype=np.float32)
        observed = np.zeros((8, 8), dtype=np.uint8)
        rendered = np.zeros((8, 8), dtype=np.uint8)
        observed[:4, :4] = 255
        rendered[:4, :4] = 255
        summary = GoTrackRefiner._summarize_ranker_features(
            features, observed, rendered)
        self.assertEqual(summary['appearance_patch_cosine_00'], 1.)
        self.assertIsNone(summary['appearance_patch_cosine_01'])
        self.assertIsNone(summary['appearance_patch_cosine_10'])
        self.assertIsNone(summary['appearance_patch_cosine_11'])
        self.assertEqual(summary['appearance_valid_fraction'], .25)

    def test_tracker_sink_records_none_rejection_and_final_state_once_per_frame(self):
        records = []
        tracker = SequentialTracker(
            Backend(output=None), initial_pose=POSE,
            candidate_sink=records.append,
            trace_context={
                'physical_object_id': 'object-a', 'sequence_id': 'sequence-a',
                'scene_id': 'scene-a', 'background_id': 'background-a',
                'provenance_kind': 'fixture', 'background_kind': 'fixture',
            })
        result = tracker.update(frame(0), segmentation(0))
        self.assertEqual(len(records), 1)
        trace = records[0]
        self.assertEqual(trace['frame_id'], 0)
        self.assertEqual(trace['context']['sequence_id'], 'sequence-a')
        self.assertEqual(trace['pose_state'], result['pose_state'])
        self.assertEqual(trace['render_state'], result['render_state'])
        self.assertEqual(len(trace['candidates']), 1)
        self.assertIsNone(trace['candidates'][0]['pose'])
        self.assertFalse(trace['candidates'][0]['eligible'])
        self.assertEqual(trace['candidates'][0]['rejection'], 'no_candidate_returned')
        self.assertEqual(len(trace['candidates'][0]['features']), FEATURE_VECTOR_SHAPE[0])
        self.assertIn('pose_total', trace['timings_ms'])
        self.assertIn('feature_extraction', trace['timings_ms'])
        self.assertIn('rank_decision', trace['timings_ms'])
        self.assertGreaterEqual(trace['timings_ms']['feature_extraction'], 0.)

    def test_invalid_pose_attempt_is_retained_with_null_pose(self):
        bad = candidate()
        bad.pose[0, 0] = 2.
        records = []
        tracker = SequentialTracker(Backend(output=bad), POSE,
                                   candidate_sink=records.append)
        with self.assertRaisesRegex(ValueError, 'Improper object rotation'):
            tracker.update(frame(0), segmentation(0))
        attempt = records[0]['candidates'][0]
        self.assertIsNone(attempt['pose'])
        self.assertFalse(attempt['eligible'])
        self.assertEqual(attempt['rejection'], 'invalid_pose')

    def test_trace_includes_refiner_appearance_summary_clock(self):
        class TimedRefiner:
            def __init__(self):
                self._ranker_appearance_summary_ms = 0.

            @property
            def ranker_appearance_summary_ms(self):
                return self._ranker_appearance_summary_ms

        class FoundPoseRecovery:
            def __init__(self, refiner):
                self.refiner = refiner

            def refine(self, _frame, _mask, seed):
                # Model five crop/refinement iterations. The wrapper's counter
                # is on the refiner, not on the backend passed to the tracker.
                for _ in range(5):
                    self.refiner._ranker_appearance_summary_ms += 1.25
                return None

            def recover(self, _frame, _mask, top_k=5):
                return []

        class AppearanceCheckedBackend:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def refine(self, *args):
                return self.wrapped.refine(*args)

            def recover(self, *args, **kwargs):
                return self.wrapped.recover(*args, **kwargs)

        records = []
        refiner = TimedRefiner()
        backend = AppearanceCheckedBackend(FoundPoseRecovery(refiner))
        tracker = SequentialTracker(
            backend, POSE, candidate_sink=records.append,
            appearance_timing_source=refiner)
        tracker.update(frame(0), segmentation(0))
        # Five iterations each add 1.25 ms to the wrapped refiner's counter.
        self.assertEqual(records[0]['timings_ms']['appearance_summary'], 6.25)

    def test_ranker_rejection_never_bypasses_current_image_gates_or_two_frame_recovery(self):
        records = []
        good = candidate()
        bad = candidate()
        bad.pixels_image = bad.pixels_image + 40.
        tracker = SequentialTracker(
            Backend(output=bad), initial_pose=POSE,
            pose_ranker=GeometricPoseRanker(),
            candidate_sink=records.append)
        result = tracker.update(frame(0), segmentation(0))
        self.assertEqual(result['pose_state'], 'lost')
        self.assertIsNone(result['cameraFromObject'])
        self.assertTrue(any(row['rejection'] == 'inconsistent_current_image_correspondences'
                            for row in records[0]['candidates']))

        tracker = SequentialTracker(Backend(output=good, hypotheses=[POSE]),
                                    pose_ranker=GeometricPoseRanker())
        first = tracker.update(frame(0), segmentation(0))
        second = tracker.update(frame(1), segmentation(1))
        self.assertEqual(first['pose_state'], 'recovering')
        self.assertIsNone(first['cameraFromObject'])
        self.assertEqual(second['pose_state'], 'tracking')

    def test_sink_errors_propagate_after_tracker_update(self):
        def fail(_record):
            raise RuntimeError('sink failed')
        tracker = SequentialTracker(Backend(output=candidate()), POSE,
                                   candidate_sink=fail)
        with self.assertRaisesRegex(RuntimeError, 'sink failed'):
            tracker.update(frame(0), segmentation(0))


if __name__ == '__main__':
    unittest.main()
