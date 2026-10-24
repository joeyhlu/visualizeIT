"""CPU analytical tests for the research-only legacy confidence veto."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .quality_confidence_evaluate import (
    bad_class_average_precision, evaluate_confidence_cache, probability_metrics,
    selective_metrics,
)
from .quality_confidence_rejection import _prepare_split, load_research_dataset, research_eda
from .quality_pipeline import RankingOptions, configure_ranking
from .quality_pose_ranker import (
    ConfidenceRejectionRanker, RankCandidate, make_ranker,
)
from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .quality_ranker_train import (
    TrainConfig, _validate_research_completion, _validate_research_config,
    _write_research_completion, atomic_json, calibration, canonical_hash,
    _confidence_decision_signatures, _confidence_parity_rows,
    _confidence_partial_progress,
)


HASH = "a" * 64
RESEARCH_SOURCES = {
    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
        "quality_ranker_train.py", "quality_ranker_features.py", "quality_pose_ranker.py",
        "quality_ranker_data.py", "quality_contract.py", "quality_gotrack.py",
        "quality_confidence_rejection.py", "quality_confidence_evaluate.py",
        "quality_pipeline.py", "quality_runner.py", "quality_offline.py",
    )
}


def artifact(*, bias=0., threshold=.5):
    width = len(FEATURE_NAMES)
    data = {name: HASH for name in (
        "feature_cache_sha256", "label_cache_sha256", "split_sha256",
        "source_trace_sha256", "reference_labels_sha256", "cache_manifest_sha256")}
    data.update(feature_schema_version=FEATURE_SCHEMA_VERSION,
                feature_names=list(FEATURE_NAMES))
    config = {
        "architecture": "linear", "seed": 42, "epochs": 40,
        "batch_size": 8, "accumulation": 4, "learning_rate": .001,
        "weight_decay": .0001, "pairwise_weight": .5, "patience": 6,
        "min_delta": .00001, "max_seconds": 300.,
        "max_output_bytes": 64 * 1024 * 1024,
        "max_checkpoint_bytes": 16 * 1024 * 1024,
        "max_candidates_per_group": 256, "max_feature_rows": 100000,
        "amp": False, "device": "cpu"}
    config_hash = hashlib.sha256(json.dumps(
        {"purpose": "confidence_rejection_research_v1", "config": config},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    identity = {
        "manifest_sha256": HASH, "data": data, "source": RESEARCH_SOURCES,
        "config": config, "config_sha256": config_hash,
        "purpose": "confidence_rejection_research_v1",
        "policy": "legacy_veto_v1", "label_policy_sha256": HASH,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "runtime": {"torch": "2.5.1", "numpy": "1.26.4", "python": "3.11",
                    "cuda": None, "cudnn": None, "device": "cpu"},
    }
    counts = {"eligible_labeled": 2, "eligible_good": 1, "eligible_bad": 1,
              "legacy_selected_labeled": 2, "legacy_selected_good": 1,
              "legacy_selected_bad": 1}
    calibrated = {"temperature": 1., "bias": 0.,
                  "acceptance_probability": threshold,
                  "validation_bce": .5, "validation_balanced_accuracy": 1.}
    frozen_policy = {"max_rotation_error_deg": 5.,
                     "max_translation_error_m": .05,
                     "max_surface_error_m": None,
                     "max_projection_error_px": None}
    policy_hash = hashlib.sha256(json.dumps(
        frozen_policy, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    identity["label_policy_sha256"] = policy_hash
    return {
        "schema_version": 3, "artifact_purpose": "confidence_rejection_research_v1",
        "policy": "legacy_veto_v1", "research_only": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES), "architecture": "linear",
        "mean": [0.] * width, "scale": [1.] * width,
        "layers": [{"weight": [[0.] * width], "bias": [bias],
                    "activation": "identity"}],
        "acceptance_probability": threshold,
        "provenance": {
            "identity": identity, "checkpoint_sha256": HASH,
            "label_policy": {"sha256": policy_hash,
                             "frozen": frozen_policy,
                             "quality_target_definition": "frozen_available_dimension_quality",
                             "angular_observability_interpretation": "ambiguous_frozen_before_inference_rotation_excluded_unchanged"},
            "supervision_counts": {"train": counts, "validation": counts},
            "calibration": calibrated, "calibration_split": "validation",
            "calibration_scope": "legacy_selected_labeled_eligible",
            "calibration_rule": "fixed_grid_bce_then_balanced_accuracy_conservative_v1",
            "checkpoint_selection": "legacy_selected_validation_uncalibrated_bce",
            "development_fitting": True, "test_read": False,
            "unsupported_claims": ["candidate_ordering", "independent_test_accuracy",
                                   "object_generalization", "physical_attachment_accuracy"],
        },
    }


def candidate(name, score, index, feature=0., x=0.):
    pose = np.eye(4); pose[0, 3] = x
    values = [0.] * len(FEATURE_NAMES); values[0] = feature
    return RankCandidate(name, index, SimpleNamespace(pose=pose),
                         {"score": score}, values)


class ConfidencePolicyTests(unittest.TestCase):
    def test_export_probability_rounding_is_separate_from_exact_veto_parity(self):
        ranker = ConfidenceRejectionRanker(artifact(threshold=.5))
        candidates = [candidate("legacy", 1., 0)]
        legacy = ranker.rank_with_selected_probability({}, candidates, .9)
        selected_all = [{"legacy_decision": legacy, "candidates": candidates,
                         "frame": {"object_id": "software-fixture", "sequence_id": "s",
                                   "frame_id": 0, "context": {}}, "label": None}]
        expected = _confidence_parity_rows(selected_all, np.array([.5]), ranker)
        rounded = _confidence_parity_rows(selected_all, np.array([.50000003]), ranker)
        # FP32 rounding can change diagnostics while leaving p>=threshold true.
        np.testing.assert_allclose([.5], [.50000003], rtol=1e-5, atol=1e-6)
        self.assertNotEqual(expected, rounded)
        self.assertEqual(_confidence_decision_signatures(expected),
                         _confidence_decision_signatures(rounded))
        crossed = _confidence_parity_rows(selected_all, np.array([.49999999]), ranker)
        # Numerical tolerance alone must never excuse crossing the veto boundary.
        np.testing.assert_allclose([.5], [.49999999], rtol=1e-5, atol=1e-6)
        self.assertNotEqual(_confidence_decision_signatures(expected),
                            _confidence_decision_signatures(crossed))

    def test_diagnostic_is_current_for_ranked_empty_and_unranked_trace_frames(self):
        payload = artifact(bias=2.)
        ranker = ConfidenceRejectionRanker(payload)
        ranker.rank({}, [candidate("selected", 1., 0)])
        self.assertEqual(ranker.research_diagnostic["legacy_selected_id"], "selected")
        ranker.rank({}, [])
        self.assertIsNone(ranker.research_diagnostic["legacy_selected_id"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checkpoint = root / "model.json"
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            options = RankingOptions("confidence-research", checkpoint,
                                     root / "trace.jsonl", False)
            refiner = SimpleNamespace(collect_ranker_features=False)
            configured, sink = configure_ranking(
                options, refiner=refiner,
                manifest={"object": "o", "sequence_id": "s", "asset": "mesh.glb",
                          "source_hashes": {"mesh.glb": HASH}},
                provenance={"input_manifest_sha256": HASH})
            decision = configured.rank({}, [candidate("selected", 1., 0)])
            sink({"frame_id": 0, "rank_decision": {
                "ordered_ids": list(decision.ordered_ids),
                "scores": dict(decision.scores),
                "selected_id": decision.selected_id,
                "rejection_reason": decision.rejection_reason},
                "candidates": []})
            # quality_contract seeds this non-None decision on a frame where
            # candidate ranking never ran.
            sink({"frame_id": 1, "rank_decision": {
                "ordered_ids": [], "scores": {}, "selected_id": None,
                "rejection_reason": None}, "candidates": []})
            rows = [json.loads(line) for line in
                    (root / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["research_diagnostic"]["legacy_selected_id"],
                             "selected")
            self.assertIsNone(rows[1]["research_diagnostic"])

    def test_threshold_exact_retains_below_veto_and_other_score_cannot_swap(self):
        ranker = ConfidenceRejectionRanker(artifact(bias=0., threshold=.5))
        first = candidate("legacy", .8, 0, feature=-100.)
        other = candidate("other", .7, 1, feature=100.)
        kept = ranker.rank_with_selected_probability({}, [first, other], .5)
        self.assertEqual(kept.selected_id, "legacy")
        self.assertEqual(kept.ordered_ids, ("legacy", "other"))
        self.assertEqual(dict(kept.scores), {"legacy": .8, "other": .7})
        rejected = ranker.rank_with_selected_probability({}, [first, other], .499)
        self.assertIsNone(rejected.selected_id)
        self.assertEqual(rejected.rejection_reason,
                         "confidence_research_probability_below_threshold")
        self.assertEqual(dict(rejected.scores), dict(kept.scores))

    def test_empty_and_ambiguous_legacy_rejections_are_exact_and_ignore_probability(self):
        ranker = ConfidenceRejectionRanker(artifact())
        empty = ranker.rank_with_selected_probability({}, [], float("nan"))
        self.assertEqual(empty.ordered_ids, ())
        self.assertIsNone(empty.selected_id)
        first = candidate("a", 1., 0, x=0.)
        second = candidate("b", .95, 1, x=.1)
        ambiguous = ranker.rank_with_selected_probability(
            {"recovery": True, "recovery_ambiguity_ratio": .9},
            [first, second], float("nan"))
        self.assertIsNone(ambiguous.selected_id)
        self.assertEqual(ambiguous.rejection_reason, "ambiguous_recovery_hypotheses")

    def test_bad_selected_feature_fails_closed_and_schema3_is_explicit_only(self):
        ranker = ConfidenceRejectionRanker(artifact())
        malformed = candidate("bad", 1., 0)
        malformed = RankCandidate(malformed.candidate_id, malformed.original_index,
                                  malformed.candidate, malformed.stats, [0.])
        with self.assertRaisesRegex(RuntimeError, "confidence_research_error"):
            ranker.rank({}, [malformed])
        with self.assertRaisesRegex(ValueError, "rejects confidence research"):
            make_ranker("learned", artifact())
        self.assertIsInstance(make_ranker("confidence-research", artifact()),
                              ConfidenceRejectionRanker)

    def test_strict_schema_types_fields_hashes_and_threshold(self):
        for mutate in (
            lambda row: row.update(schema_version=True),
            lambda row: row.update(research_only=1),
            lambda row: row.update(acceptance_probability=0.),
            lambda row: row["provenance"].update(test_read=True),
            lambda row: row["provenance"]["identity"]["source"].update(extra=HASH),
            lambda row: row["provenance"].update(checkpoint_sha256="A" * 64),
        ):
            payload = copy.deepcopy(artifact()); mutate(payload)
            with self.assertRaises(ValueError):
                ConfidenceRejectionRanker(payload)

    def test_bool_arrays_incoherent_counts_and_forged_runtime_config_fail(self):
        mutations = (
            lambda row: row["mean"].__setitem__(0, True),
            lambda row: row["layers"][0]["weight"][0].__setitem__(0, False),
            lambda row: row["provenance"]["supervision_counts"]["train"].update(eligible_good=2),
            lambda row: row["provenance"]["supervision_counts"]["train"].update(
                legacy_selected_good=2, legacy_selected_bad=0),
            lambda row: row["provenance"]["supervision_counts"]["validation"].update(
                legacy_selected_labeled=1, legacy_selected_bad=0),
            lambda row: row["provenance"]["identity"]["runtime"].update(torch="2.5.2"),
            lambda row: row["provenance"]["identity"]["runtime"].update(numpy="2.0.0"),
            lambda row: row["provenance"]["identity"]["runtime"].update(device="cuda"),
            lambda row: row["provenance"]["identity"]["config"].update(batch_size=4),
        )
        for mutate in mutations:
            payload = copy.deepcopy(artifact()); mutate(payload)
            with self.assertRaises(ValueError):
                ConfidenceRejectionRanker(payload)

    def test_exact_cuda_build_runtime_and_two_epoch_smoke_config_are_allowed(self):
        payload = artifact()
        identity = payload["provenance"]["identity"]
        identity["runtime"]["torch"] = "2.5.1+cu124"
        identity["config"]["epochs"] = 2
        identity["config_sha256"] = canonical_hash({
            "purpose": "confidence_rejection_research_v1",
            "config": identity["config"],
        })
        self.assertIsInstance(ConfidenceRejectionRanker(payload),
                              ConfidenceRejectionRanker)

    def test_label_changes_cannot_change_prelabel_legacy_choice(self):
        pose = np.eye(4).tolist()
        frame = {"object_id": "o", "sequence_id": "s", "frame_id": 1,
                 "source_frame_id": 10, "provenance_kind": "synthetic",
                 "context": {"provenance_kind": "synthetic"}, "candidates": [
                     {"candidate_id": "first", "original_index": 0, "pose": pose,
                      "eligible": True, "validation_stats": {"score": 2.},
                      "features": [0.] * len(FEATURE_NAMES)},
                     {"candidate_id": "second", "original_index": 1, "pose": pose,
                      "eligible": True, "validation_stats": {"score": 1.},
                      "features": [0.] * len(FEATURE_NAMES)}]}
        labels = [{"object_id": "o", "sequence_id": "s", "frame_id": 1,
                   "candidate_id": "first", "quality_target": 0},
                  {"object_id": "o", "sequence_id": "s", "frame_id": 1,
                   "candidate_id": "second", "quality_target": 1}]
        one = _prepare_split([frame], labels, "validation")
        labels[0]["quality_target"], labels[1]["quality_target"] = 1, 0
        two = _prepare_split([frame], labels, "validation")
        self.assertEqual(one["selected_all"][0]["legacy_decision"].selected_id, "first")
        self.assertEqual(two["selected_all"][0]["legacy_decision"].selected_id, "first")


class ConfidenceAnalyticsTests(unittest.TestCase):
    def test_partial_smoke_receipt_excludes_unobserved_infinite_loss(self):
        # Stopping after step one precedes validation: best_loss is an initialized
        # sentinel in the resumable checkpoint, not an observed finite loss.
        progress = {"epoch": 0, "cursor": 32, "step": 1,
                    "best_loss": float("inf"), "bad_epochs": 0,
                    "elapsed_seconds": .1, "order": [1, 0]}
        summary = _confidence_partial_progress(progress)
        self.assertEqual(summary, {"epoch": 0, "cursor": 32, "step": 1})
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)
        self.assertEqual(progress["best_loss"], float("inf"))
        with self.assertRaises(AssertionError):
            _confidence_partial_progress({**progress, "step": 2})

    def test_evaluator_distinguishes_inventory_actual_empty_and_candidate_traces(self):
        for marker in ("not_processed", "not_processed_within_approved_compute_budget"):
            with self.subTest(marker=marker):
                self._assert_evaluator_processing_counts(marker)

    def _assert_evaluator_processing_counts(self, marker):
        frames = [
            {"object_id": "fixture", "sequence_id": "software-test", "frame_id": 0,
             "context": {}, "candidates": [], "failure_reason": marker},
            {"object_id": "fixture", "sequence_id": "software-test", "frame_id": 1,
             "context": {}, "candidates": [], "failure_reason": "no_candidates"},
            {"object_id": "fixture", "sequence_id": "software-test", "frame_id": 2,
             "context": {}, "candidates": [{"candidate_id": "ineligible",
                                                "eligible": False, "pose": None}],
             "failure_reason": "hard_gate"},
        ]
        payload = artifact()
        audit_identity = payload["provenance"]["identity"]["data"]
        frozen_policy = payload["provenance"]["label_policy"]["frozen"]
        policy_sha256 = payload["provenance"]["label_policy"]["sha256"]
        fake_ranker = SimpleNamespace(to_checkpoint=lambda: payload)
        requested = []

        def load(_cache, split):
            requested.append(split)
            if split == "test":
                raise AssertionError("test shard opened")
            return (frames, []) if split == "validation" else ([], [])

        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.json"
            checkpoint.write_text("{}", encoding="utf-8")
            with patch("bench.quality_confidence_evaluate.audit_cache", return_value={
                     "approved": True, "audit_scope": "train_validation_only",
                     "identity": audit_identity}), \
                 patch("bench.quality_confidence_evaluate.make_ranker",
                       return_value=fake_ranker), \
                 patch("bench.quality_confidence_evaluate._read_label_policy",
                       return_value=(frozen_policy, policy_sha256)), \
                 patch("bench.quality_confidence_evaluate._cache_manifest",
                       return_value=({"frame_count": 3}, None)), \
                 patch("bench.quality_confidence_evaluate.load_cache_split",
                       side_effect=load):
                report = evaluate_confidence_cache(
                    Path("software-test-cache"), split="validation",
                    checkpoint=checkpoint)
        self.assertEqual(requested, ["validation", "train", "validation"])
        self.assertEqual(report["inventory"]["full_frozen_frames"], 3)
        self.assertEqual(report["inventory"]["actual_traced_frames"], 2)
        self.assertEqual(report["inventory"]["candidate_containing_trace_frames"], 1)
        self.assertEqual(report["inventory"]["unprocessed_inventory_frames"], 1)
        self.assertEqual(report["inventory"]["selected_split_inventory_frames"], 3)
        self.assertEqual(report["inventory"]["selected_split_traced_frames"], 2)
        self.assertEqual(
            report["inventory"]["selected_split_candidate_containing_trace_frames"], 1)
        self.assertEqual(report["inventory"]["selected_split_unprocessed_frames"], 1)
        self.assertEqual(report["failure_categories"]["unprocessed"], 1)
        self.assertEqual(report["failure_categories"]["empty"], 1)
        self.assertEqual(report["failure_categories"]["hard_gate"], 1)

    def test_completion_writer_loads_full_final_checkpoint_not_progress_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); best = root / "best.pt"; last = root / "last.pt"
            best.write_bytes(b"best"); last.write_bytes(b"last")
            config = TrainConfig(**artifact()["provenance"]["identity"]["config"])
            identity = {"config_sha256": canonical_hash({
                "purpose": "confidence_rejection_research_v1",
                "config": config.__dict__,
            })}
            selected = {"purpose": "confidence_rejection_research_v1",
                        "identity": identity, "config": config.__dict__, "model": {},
                        "progress": {"best_loss": .2}}
            final = {"purpose": "confidence_rejection_research_v1",
                     "identity": identity, "config": config.__dict__,
                     "progress": {"epoch": 7, "step": 14, "best_loss": .2}}
            with patch("bench.quality_ranker_train._read_checkpoint",
                       side_effect=[selected, final]), \
                 patch("bench.quality_ranker_train._torch_state_sha256",
                       return_value=HASH):
                receipt = _write_research_completion(
                    SimpleNamespace(), root, best, last, config, "early_stopping")
            self.assertEqual(receipt["epochs_completed"], 7)
            self.assertEqual(receipt["optimizer_steps"], 14)
            self.assertEqual(receipt["identity_sha256"], canonical_hash(identity))

    def test_evaluator_rejects_cross_cache_before_any_split_load(self):
        payload = artifact()
        payload["provenance"]["identity"]["data"] = {"cache": "artifact"}
        fake_ranker = SimpleNamespace(to_checkpoint=lambda: payload)
        with patch("bench.quality_confidence_evaluate.audit_cache",
                   return_value={"approved": True, "audit_scope": "train_validation_only",
                                 "identity": {"cache": "different"}}), \
             patch("bench.quality_confidence_evaluate.make_ranker", return_value=fake_ranker), \
             patch("bench.quality_confidence_evaluate._read_label_policy",
                   return_value=({}, HASH)), \
             patch("bench.quality_confidence_evaluate.load_cache_split") as load:
            with self.assertRaisesRegex(ValueError, "data identity"):
                evaluate_confidence_cache(Path("cache"), split="validation",
                                          checkpoint=Path("model.json"))
            load.assert_not_called()

    def test_completion_requires_best_and_binds_final_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); best = root / "best.pt"; last = root / "last.pt"
            best.write_bytes(b"best"); last.write_bytes(b"last")
            config = artifact()["provenance"]["identity"]["config"]
            identity = {"config_sha256": artifact()["provenance"]["identity"]["config_sha256"]}
            selected = {"identity": identity, "config": config, "model": {},
                        "progress": {"best_loss": .2}}
            final = {"purpose": "confidence_rejection_research_v1",
                     "identity": identity, "config": config,
                     "progress": {"epoch": 7, "step": 14, "best_loss": .2,
                                  "bad_epochs": 6}}
            receipt = {
                "schema_version": 1, "artifact_purpose": "confidence_rejection_research_v1",
                "policy": "legacy_veto_v1", "complete": True,
                "termination_reason": "early_stopping",
                "checkpoint_selection": "legacy_selected_validation_uncalibrated_bce",
                "identity_sha256": canonical_hash(identity),
                "config_sha256": identity["config_sha256"],
                "selected_checkpoint": "best.pt",
                "selected_checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "selected_model_sha256": HASH, "selected_validation_bce": .2,
                "final_checkpoint": "last.pt",
                "final_checkpoint_sha256": hashlib.sha256(b"last").hexdigest(),
                "epochs_completed": 7, "optimizer_steps": 14, "test_read": False}
            atomic_json(root / "completion.json", receipt)
            with patch("bench.quality_ranker_train._torch_state_sha256", return_value=HASH), \
                 patch("bench.quality_ranker_train._read_checkpoint", return_value=final):
                result = _validate_research_completion(SimpleNamespace(), best, selected)
            self.assertTrue(result["complete"])
            malformed = dict(receipt); malformed["optimizer_steps"] = True
            atomic_json(root / "completion.json", malformed)
            with patch("bench.quality_ranker_train._torch_state_sha256", return_value=HASH):
                with self.assertRaisesRegex(ValueError, "progress is malformed"):
                    _validate_research_completion(SimpleNamespace(), best, selected)
            (root / "completion.json").unlink()
            with self.assertRaisesRegex(ValueError, "completed research run"):
                _validate_research_completion(SimpleNamespace(), best, selected)
            with self.assertRaisesRegex(ValueError, "validation-selected best.pt"):
                _validate_research_completion(SimpleNamespace(), last, selected)

    def test_research_loader_never_requests_test_shard(self):
        identity = {"feature_names": list(FEATURE_NAMES),
                    "feature_schema_version": FEATURE_SCHEMA_VERSION}
        counts = {"eligible_labeled": 2, "eligible_good": 1, "eligible_bad": 1,
                  "legacy_selected_labeled": 2, "legacy_selected_good": 1,
                  "legacy_selected_bad": 1}
        prepared = {"groups": [], "selected_all": [], "selected_labeled": [],
                    "counts": counts}
        requested = []
        def load(cache, split):
            requested.append(split)
            if split == "test":
                raise AssertionError("test shard opened")
            return [], []
        with patch("bench.quality_confidence_rejection.audit_cache",
                   return_value={"approved": True, "audit_scope": "train_validation_only",
                                 "identity": identity, "ranking_signal_available": False}), \
             patch("bench.quality_confidence_rejection._read_label_policy",
                   return_value=({"max_rotation_error_deg": 5.}, HASH)), \
             patch("bench.quality_confidence_rejection.load_cache_split", side_effect=load), \
             patch("bench.quality_confidence_rejection._prepare_split", return_value=prepared):
            load_research_dataset(Path("cache_manifest.json"))
        self.assertEqual(requested, ["train", "validation"])

    def test_known_perfect_veto_and_tied_constant_average_precision(self):
        targets = np.array([1] * 15 + [0] * 4)
        perfect = np.array([True] * 15 + [False] * 4)
        metrics = selective_metrics(targets, perfect)
        self.assertEqual(metrics["coverage"], 15 / 19)
        self.assertEqual(metrics["balanced_accuracy"], 1.)
        self.assertEqual(metrics["selective_error"], 0.)
        probability = probability_metrics(np.array([.9] * 15 + [.1] * 4), targets)
        self.assertAlmostEqual(probability["brier"], .01)
        self.assertAlmostEqual(probability["bce"], -np.log(.9))
        tied = bad_class_average_precision(np.full(19, 49 / 51), targets)
        self.assertAlmostEqual(tied["value"], 4 / 19)

    def test_zero_retained_denominator_has_explicit_reason(self):
        metrics = selective_metrics(np.array([1, 0]), np.array([False, False]))
        self.assertIsNone(metrics["selective_error"])
        self.assertEqual(metrics["null_reasons"]["selective_error"],
                         "no_retained_labeled_legacy_choices")

    def test_calibration_grid_is_deterministic_and_strict(self):
        logits = np.array([2., 1., -1., -2.])
        targets = np.array([1., 1., 0., 0.])
        result = calibration(logits, targets, strict_threshold=True)
        self.assertEqual(result, calibration(logits, targets, strict_threshold=True))
        self.assertGreater(result["acceptance_probability"], 0.)
        self.assertLess(result["acceptance_probability"], 1.)

    def test_eda_counts_all_bad_candidates_and_uses_train_iqr(self):
        groups = [{"object_id": "o", "sequence_id": "s", "frame_id": 1,
                   "candidates": [{"features": [0.] * len(FEATURE_NAMES),
                                   "quality_target": 0, "eligible": True},
                                  {"features": [1.] * len(FEATURE_NAMES),
                                   "quality_target": 0, "eligible": True}]}]
        report = research_eda(groups, list(FEATURE_NAMES))
        self.assertEqual(report["class_counts"], {"good": 0, "bad": 2})
        self.assertEqual(report["frame_count"], 1)

    def test_frozen_cpu_configs_and_pipeline_name(self):
        root = Path(__file__).with_name("runtime")
        for name, architecture in (("ranker-confidence-linear-cpu.json", "linear"),
                                   ("ranker-confidence-mlp-cpu.json", "mlp64_32")):
            config = TrainConfig(**json.loads((root / name).read_text()))
            self.assertEqual(config.architecture, architecture)
            _validate_research_config(config)
        with self.assertRaises(ValueError):
            RankingOptions("confidence-research", None).validate()


if __name__ == "__main__":
    unittest.main()

