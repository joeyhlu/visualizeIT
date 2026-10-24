"""Fixed-set evaluator accounting checks using analytical candidate traces."""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from .quality_ranker_evaluate import evaluate_frames
from .quality_ranker_features import FEATURE_NAMES


def _pose(*, z_degrees=0., translation=(0., 0., 0.)):
    radians = np.deg2rad(z_degrees)
    value = np.eye(4)
    value[:3, :3] = [[np.cos(radians), -np.sin(radians), 0.],
                     [np.sin(radians), np.cos(radians), 0.], [0., 0., 1.]]
    value[:3, 3] = translation
    return value.tolist()


def _candidate(candidate_id, *, eligible=True, pose=None):
    return {"candidate_id": candidate_id, "original_index": 0, "eligible": eligible,
            "pose": pose, "features": [0.] * len(FEATURE_NAMES),
            "validation_stats": {}, "rejection": None if eligible else "test_gate"}


class EvaluationAccountingTests(unittest.TestCase):
    def test_research_kind_is_explicit_and_never_opens_test_cache(self):
        from .quality_ranker_evaluate import evaluate_cache, _method_choices
        with patch('bench.quality_ranker_evaluate.audit_cache') as audit:
            with self.assertRaisesRegex(ValueError, 'limited to train/validation'):
                evaluate_cache('unopened', split='test', learned_kind='ordering-research')
            audit.assert_not_called()
        frame = {'object_id': 'a', 'sequence_id': 's', 'frame_id': 0,
                 'candidates': [], 'context': {}, 'legacy_choice_candidate_id': None}
        with patch('bench.quality_pose_ranker.make_ranker') as factory:
            factory.return_value.rank.return_value = MagicMock(selected_id=None)
            choices = _method_choices([frame], learned_checkpoint='explicit.json',
                                      learned_kind='ordering-research')
            factory.assert_any_call('ordering-research', 'explicit.json')
            self.assertIsNone(choices['learned'][('a', 's', 0)])

    def test_research_cache_must_match_original_source_identity(self):
        from .quality_ranker_evaluate import evaluate_cache
        with patch('bench.quality_ranker_evaluate.audit_cache',
                   return_value={'approved': True, 'identity': {'cache': 'foreign'}}) as audit, \
             patch('bench.quality_pose_ranker.make_ranker') as factory, \
             patch('bench.quality_ranker_evaluate._cache_manifest') as shards:
            factory.return_value.to_checkpoint.return_value = {
                'provenance': {'identity': {'ordering_evidence': {
                    'sources': [{'identity': {'cache': 'bound'}}]}}}}
            with self.assertRaisesRegex(ValueError, 'not a bound development source'):
                evaluate_cache('foreign', split='validation', learned_kind='ordering-research',
                               learned_checkpoint='explicit.json')
            audit.assert_called_once_with('foreign', training_only=True)
            shards.assert_not_called()

    def test_failures_oracle_counterfactual_ambiguity_and_trace_samples(self):
        frames = [
            {"object_id": "object-a", "sequence_id": "sequence-a", "frame_id": 0,
             "source_frame_id": 10, "background_id": "procedural:gray",
             "provenance_kind": "synthetic", "context": {},
             "candidates": [
                 _candidate("ineligible-perfect", eligible=False, pose=_pose()),
                 _candidate("eligible-bad", pose=_pose(z_degrees=90, translation=(.1, 0., 0.))),
             ], "legacy_choice_candidate_id": "ineligible-perfect",
             "timings_ms": {"rank_decision": 1.5},
             "resource_memory": {"host_peak_rss_bytes": 4096}},
            {"object_id": "object-a", "sequence_id": "sequence-a", "frame_id": 1,
             "source_frame_id": 11, "background_id": "procedural:gray",
             "provenance_kind": "synthetic", "context": {}, "candidates": [],
             "legacy_choice_candidate_id": None, "timings_ms": {}, "resource_memory": {}},
            {"object_id": "object-a", "sequence_id": "sequence-a", "frame_id": 2,
             "source_frame_id": 12, "background_id": "procedural:gray",
             "provenance_kind": "synthetic", "context": {},
             "candidates": [_candidate("no-pose", eligible=False, pose=None)],
             "legacy_choice_candidate_id": "ghost", "timings_ms": {}, "resource_memory": {}},
        ]
        labels = [
            {"object_id": "object-a", "sequence_id": "sequence-a", "frame_id": 0,
             "candidate_id": "ineligible-perfect", "quality_target": 1,
             "rotation_error_deg": None, "translation_error_m": 0.,
             "observationally_ambiguous": True, "label_kind": "synthetic_renderer_pose"},
            {"object_id": "object-a", "sequence_id": "sequence-a", "frame_id": 0,
             "candidate_id": "eligible-bad", "quality_target": 0,
             "rotation_error_deg": 90., "translation_error_m": .1,
             "observationally_ambiguous": False, "label_kind": "synthetic_renderer_pose"},
        ]
        report = evaluate_frames(
            frames, labels, {"max_rotation_error_deg": 5., "max_translation_error_m": .05},
            bootstrap_draws=200)

        self.assertEqual(report["candidate_audit"]["empty_candidate_frames"], 1)
        self.assertEqual(report["candidate_audit"]["all_gated_frames"], 1)
        self.assertEqual(report["candidate_audit"]["candidates_before_gates"], 3)
        self.assertEqual(report["candidate_audit"]["eligible_after_gates"], 1)
        self.assertEqual(report["methods"]["legacy"]["failures"], 3)
        self.assertEqual(report["methods"]["legacy"]["failure_counts"]["selected_candidate_unknown"], 1)
        self.assertEqual(report["methods"]["oracle_before_gates"]["available_pose_count"], 0)
        self.assertEqual(report["methods"]["oracle_before_gates"]["rotation_error_deg"]["count"], 0)
        self.assertEqual(report["methods"]["oracle_before_gates"]["rotation_error_deg"]["excluded_ambiguous_selected"], 1)
        self.assertEqual(report["methods"]["oracle_before_gates"]["metric_selection_scope"],
                         "counterfactual_candidate_pose_including_ineligible")
        self.assertEqual(report["methods"]["oracle_after_gates"]["available_pose_count"], 1)
        self.assertEqual(report["methods"]["oracle_after_gates"]["rotation_error_deg"]["count"], 1)
        self.assertEqual(report["trace_measurements"]["frame_timings_ms"]["rank_decision"]["count"], 1)
        self.assertEqual(report["trace_measurements"]["resource_memory"]["host_peak_rss_bytes"]["max"], 4096.)
        self.assertEqual(report["provenance_counts_by_frame"]["synthetic"], 3)
        # One object/sequence is not an independent cluster sample.
        self.assertIsNone(report["methods"]["legacy"]["uncertainty"]["by_object"]["availability_value"]["ci95"])


if __name__ == "__main__":
    unittest.main()
