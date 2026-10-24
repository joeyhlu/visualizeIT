"""Analytic safeguards only: these tests never import torch or optimize."""
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from .quality_ranker_train import (
    TrainConfig, _TrainingPurpose, _flatten, _integration_supervision_summary,
    _load_dataset, _require_checkpoint_purpose, _validate_resume_identity,
    _resume_with_partial_snapshot, _train_core, atomic_json, bce, calibrate_margin, calibration, canonical_hash,
    cloud_guard, fit_normalization, ordering_pairs, sigmoid, state_equal,
    storage_guard, train,
)


class TrainingBoundaryTests(unittest.TestCase):
    def test_local_training_rejected_before_torch_or_dataset_read(self):
        with patch("platform.system", return_value="Windows"), patch(
                "bench.quality_ranker_train._load_dataset") as loader:
            with self.assertRaisesRegex(RuntimeError, "approved Linux cloud"):
                train(Path("absent"), Path("unused"), TrainConfig(), cloud_execution=True)
            loader.assert_not_called()
        with patch("platform.system", return_value="Linux"):
            with self.assertRaises(RuntimeError):
                cloud_guard(False)

    def test_normalization_train_only_and_constant_columns(self):
        # Analytical population mean/std of [1,3] are 2 and 1; a constant
        # column uses scale1. A future validation value100 cannot enter the fit.
        mean, scale = fit_normalization(np.array([[1., 5.], [3., 5.]]))
        np.testing.assert_array_equal(mean, [2., 5.])
        np.testing.assert_array_equal(scale, [1., 1.])
        np.testing.assert_array_equal((np.array([[100., 5.]]) - mean) / scale, [[98., 0.]])

    def test_pairs_never_invent_all_bad_or_equal_target_order(self):
        high, low = ordering_pairs(np.array([0., 0., 0.]))
        self.assertEqual(len(high), 0)
        # Definition: pair i>j exists iff target_i > target_j.
        high, low = ordering_pairs(np.array([0., 1., 1.]))
        np.testing.assert_array_equal(high, [1, 2])
        np.testing.assert_array_equal(low, [0, 0])

    def test_bce_stable_matches_analytical_log2_and_extremes(self):
        # A zero logit gives p=1/2, so either binary label costs log(2).
        self.assertAlmostEqual(bce(np.array([0., 0.]), np.array([0., 1.])), np.log(2.))
        np.testing.assert_array_equal(sigmoid(np.array([-1000., 0., 1000.])), [0., .5, 1.])
        self.assertEqual(bce(np.array([-1000., 1000.]), np.array([0., 1.])), 0.)

    def test_frame_flatten_preserves_all_bad_and_limits(self):
        groups = [{"candidates": [
            {"features": [1., 0.], "quality_target": 0, "eligible": True},
            {"features": [2., 0.], "quality_target": 0, "eligible": True},
            {"features": [999., 0.], "quality_target": 1, "eligible": False}]}]
        x, y, indices = _flatten(groups, ["a", "b"], TrainConfig())
        np.testing.assert_array_equal(x, [[1., 0.], [2., 0.]])
        np.testing.assert_array_equal(y, [0., 0.])
        np.testing.assert_array_equal(indices[0], [0, 1])
        with self.assertRaisesRegex(ValueError, "Candidate count"):
            _flatten(groups, ["a", "b"], TrainConfig(max_candidates_per_group=1))
        with self.assertRaisesRegex(ValueError, "memory limit"):
            _flatten(groups, ["a", "b"], TrainConfig(max_feature_rows=1))

    def test_calibration_is_deterministic_and_rejects_one_class(self):
        z, y = np.array([-2., -1., 1., 2.]), np.array([0., 0., 1., 1.])
        result = calibration(z, y)
        self.assertEqual(result, calibration(z, y))
        # Separable symmetric logits remain separable for the grid calibration.
        self.assertEqual(result["validation_balanced_accuracy"], 1.)
        self.assertLess(result["validation_bce"], bce(z, y))
        with self.assertRaisesRegex(ValueError, "both good and bad"):
            calibration(z, np.zeros(4))

    def test_loader_audits_only_training_shards_and_never_requests_test(self):
        module = importlib.import_module("bench.quality_ranker_data")
        from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
        calls = []
        def load(path, split):
            calls.append(split)
            return [{"split": split}]
        identity = {"frozen": "hash", "feature_names": list(FEATURE_NAMES), "feature_schema_version": FEATURE_SCHEMA_VERSION}
        with patch.object(module, "audit_cache", return_value={"approved": True, "identity": identity,
                                                              "ranking_signal_available": True}) as audit, patch.object(
                module, "load_training_split", side_effect=load):
            result = _load_dataset(Path("cache/cache_manifest.json"))
        audit.assert_called_once_with(Path("cache"), training_only=True)
        self.assertEqual(calls, ["train", "validation"])
        self.assertNotIn("test", result)
        with patch.object(module, "audit_cache", return_value={"approved": False}):
            with self.assertRaisesRegex(ValueError, "audit"):
                _load_dataset(Path("cache/cache_manifest.json"))
        with patch.object(module, "audit_cache", return_value={"approved": True, "identity": {}}):
            with self.assertRaisesRegex(ValueError, "feature schema differs"):
                _load_dataset(Path("cache/cache_manifest.json"))
        for signal in (False, None):
            with patch.object(module, "audit_cache", return_value={"approved": True, "identity": identity,
                                                                  "ranking_signal_available": signal}), patch.object(
                    module, "load_training_split") as load_split:
                with self.assertRaisesRegex(ValueError, "no usable ranking alternatives"):
                    _load_dataset(Path("cache/cache_manifest.json"))
                load_split.assert_not_called()

        calls.clear()
        with patch.object(module, "audit_cache", return_value={"approved": True, "identity": identity,
                                                              "ranking_signal_available": False}) as audit, patch.object(
                module, "load_training_split", side_effect=load):
            result = _load_dataset(Path("cache/cache_manifest.json"),
                                   purpose=_TrainingPurpose.INTEGRATION_SMOKE)
        audit.assert_called_once_with(Path("cache"), training_only=True)
        self.assertEqual(calls, ["train", "validation"])
        self.assertFalse(result["ranking_signal_available"])
        self.assertNotIn("test", result)

    def test_integration_supervision_reports_zero_signal_without_inventing_pairs(self):
        groups = [{
            "context": {"provenance_kind": "real"},
            "candidates": [
                {"features": [1., 0.], "quality_target": 0, "eligible": True},
                {"features": [2., 0.], "quality_target": 0, "eligible": True},
                {"features": [3., 0.], "quality_target": 0, "eligible": False},
            ],
        }]
        summary = _integration_supervision_summary(groups, ["a", "b"], TrainConfig())
        self.assertEqual(summary, {
            "eligible_groups": 1,
            "eligible_candidates": 2,
            "good_label_count": 0,
            "bad_label_count": 2,
            "mixed_group_count": 0,
            "ordering_pair_count": 0,
            "pair_supervision_available": False,
        })
        positive = [{"context": {"provenance_kind": "real"},
                     "candidates": [{"features": [4., 0.], "quality_target": 1, "eligible": True}]}]
        val = _integration_supervision_summary(positive, ["a", "b"], TrainConfig())
        self.assertEqual((val["good_label_count"], val["bad_label_count"],
                          val["mixed_group_count"], val["ordering_pair_count"]), (1, 0, 0, 0))

    def test_integration_supervision_rejects_fixture_malformed_and_empty_splits(self):
        fixture = [{"context": {"provenance_kind": "fixture"},
                    "candidates": [{"features": [1., 0.], "quality_target": 1, "eligible": True}]}]
        with self.assertRaisesRegex(ValueError, "fixture-provenance"):
            _integration_supervision_summary(fixture, ["a", "b"], TrainConfig())
        malformed = [{"context": {"provenance_kind": "real"},
                      "candidates": [{"features": [float("nan"), 0.],
                                      "quality_target": 1, "eligible": True}]}]
        with self.assertRaisesRegex(ValueError, "finite"):
            _integration_supervision_summary(malformed, ["a", "b"], TrainConfig())
        nonbinary = [{"context": {"provenance_kind": "real"},
                      "candidates": [{"features": [1., 0.], "quality_target": .5, "eligible": True}]}]
        with self.assertRaisesRegex(ValueError, "binary labels"):
            _integration_supervision_summary(nonbinary, ["a", "b"], TrainConfig())
        empty = [{"context": {"provenance_kind": "real"},
                  "candidates": [{"features": [1., 0.], "quality_target": 1, "eligible": False}]}]
        with self.assertRaisesRegex(ValueError, "eligible finite feature rows"):
            _integration_supervision_summary(empty, ["a", "b"], TrainConfig())

    def test_resume_and_export_purpose_binding_reject_cross_purpose_use(self):
        config = TrainConfig()
        config_values = {
            "architecture": config.architecture, "seed": config.seed,
            "epochs": config.epochs, "batch_size": config.batch_size,
            "accumulation": config.accumulation, "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay, "pairwise_weight": config.pairwise_weight,
            "patience": config.patience, "min_delta": config.min_delta,
            "max_seconds": config.max_seconds, "max_output_bytes": config.max_output_bytes,
            "max_checkpoint_bytes": config.max_checkpoint_bytes,
            "max_candidates_per_group": config.max_candidates_per_group,
            "max_feature_rows": config.max_feature_rows, "amp": config.amp,
            "device": config.device,
        }
        integration_purpose = _TrainingPurpose.INTEGRATION_SMOKE.value
        identity = {"purpose": integration_purpose,
                    "config_sha256": canonical_hash({"purpose": integration_purpose,
                                                       "config": config_values})}
        loaded = {"purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
                  "identity": identity, "config": config_values}
        with self.assertRaisesRegex(ValueError, "Resume purpose"):
            _validate_resume_identity(loaded, identity, config, _TrainingPurpose.SUBSTANTIVE)
        integration_identity = identity
        integration_state = {"purpose": _TrainingPurpose.INTEGRATION_SMOKE.value,
                             "identity": integration_identity, "config": config_values}
        _require_checkpoint_purpose(integration_state, _TrainingPurpose.INTEGRATION_SMOKE,
                                    operation="export")
        tampered = {"purpose": _TrainingPurpose.SUBSTANTIVE.value,
                    "identity": dict(integration_identity), "config": config_values}
        tampered["identity"]["purpose"] = _TrainingPurpose.SUBSTANTIVE.value
        with self.assertRaisesRegex(ValueError, "purpose/config identity"):
            _require_checkpoint_purpose(tampered, _TrainingPurpose.SUBSTANTIVE,
                                        operation="export")
        with self.assertRaisesRegex(ValueError, "Normal export explicitly rejects"):
            _require_checkpoint_purpose(integration_state, _TrainingPurpose.SUBSTANTIVE,
                                        operation="export")
        module = importlib.import_module("bench.quality_ranker_train")
        with patch("platform.system", return_value="Linux"), patch.dict(
                "sys.modules", {"torch": ModuleType("torch")}), patch.object(
                module, "_read_checkpoint", return_value=integration_state), patch.object(
                module, "_load_dataset") as loader:
            with self.assertRaisesRegex(ValueError, "Normal export explicitly rejects"):
                module.export(Path("integration.pt"), Path("different/cache_manifest.json"),
                              Path("production.json"), cloud_execution=True)
            loader.assert_not_called()
        with patch("bench.quality_ranker_train.cloud_guard"):
            with self.assertRaisesRegex(ValueError, "config is fixed"):
                _train_core(Path("cache/cache_manifest.json"), Path("unused"),
                            TrainConfig(epochs=1), cloud_execution=True,
                            purpose=_TrainingPurpose.INTEGRATION_SMOKE)

    def test_integration_cli_does_not_expose_training_overrides(self):
        module = importlib.import_module("bench.quality_ranker_train")
        with patch.object(module, "integration_smoke", return_value={"purpose": "ok"}) as run, patch(
                "builtins.print"):
            module.main(["integration-smoke", "--manifest", "cache/cache_manifest.json",
                         "--output", "new-output", "--cloud-execution"])
            run.assert_called_once_with(Path("cache/cache_manifest.json"), Path("new-output"),
                                        cloud_execution=True)
        with patch("builtins.print"), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                module.main(["integration-smoke", "--manifest", "cache/cache_manifest.json",
                             "--output", "new-output", "--architecture", "mlp64_32"])

    def test_integration_resume_snapshots_mutable_partial_before_export_and_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            last_checkpoint = Path(temp) / "last.pt"
            last_checkpoint.write_text("partial checkpoint", encoding="utf-8")
            reached = []

            def snapshot(path):
                self.assertEqual(path.read_text(encoding="utf-8"), "partial checkpoint")
                return {"step": 1, "sha256": "partial-hash"}

            def resume(path):
                self.assertEqual(path.read_text(encoding="utf-8"), "partial checkpoint")
                path.write_text("resumed final checkpoint", encoding="utf-8")
                return path

            def finish(path, partial_snapshot):
                self.assertEqual(path.read_text(encoding="utf-8"), "resumed final checkpoint")
                self.assertEqual(partial_snapshot, {"step": 1, "sha256": "partial-hash"})
                reached.append("export")
                reached.append("receipt")
                return {"exported": True, "receipt_written": True,
                        "partial_checkpoint": partial_snapshot["sha256"]}

            receipt = _resume_with_partial_snapshot(last_checkpoint,
                                                    snapshot_partial=snapshot,
                                                    resume_run=resume,
                                                    finish=finish)

        self.assertEqual(reached, ["export", "receipt"])
        self.assertEqual(receipt, {"exported": True, "receipt_written": True,
                                   "partial_checkpoint": "partial-hash"})

    def test_atomic_json_and_storage_limits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            directory = Path(temp)
            path = directory / "state.json"
            atomic_json(path, {"step": 1})
            atomic_json(path, {"step": 2})
            self.assertEqual(json.loads(path.read_text()), {"step": 2})
            self.assertFalse(path.with_name("state.json.tmp").exists())
            with self.assertRaisesRegex(RuntimeError, "storage budget"):
                storage_guard(directory, TrainConfig(max_output_bytes=1))

    def test_config_rejects_invalid_resource_and_amp_modes(self):
        for config in (TrainConfig(accumulation=0), TrainConfig(amp=True),
                       TrainConfig(max_seconds=float("nan")), TrainConfig(pairwise_weight=-1)):
            with self.assertRaises(ValueError):
                config.validate()
        TrainConfig().validate()

    def test_resume_identity_and_recursive_state_comparison(self):
        # Canonical JSON hash is invariant to dictionary insertion order.
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))
        self.assertTrue(state_equal({"state": (np.array([1, 2]), [3])}, {"state": (np.array([1, 2]), [3])}))
        self.assertFalse(state_equal({"state": np.array([1, 2])}, {"state": np.array([1, 3])}))

    def test_validation_margin_matches_distinct_pose_geometry(self):
        same, distinct = np.eye(4), np.eye(4)
        distinct[0, 3] = .04  # Analytic distance .04m exceeds fixed .03m gate.
        groups = [
            {"context": {"recovery": True}, "candidates": [{"candidate_id": "a", "pose": same, "quality_target": 1},
                            {"candidate_id": "b", "pose": distinct, "quality_target": 0}]},
            {"context": {"recovery": True}, "candidates": [{"candidate_id": "a", "pose": same, "quality_target": 0},
                            {"candidate_id": "b", "pose": distinct, "quality_target": 1}]},
        ]
        # Good gap=.7, wrong gap=.05: midpoint .375 separates them without
        # placing a cutoff exactly on either observed floating-point gap.
        self.assertAlmostEqual(calibrate_margin(groups, np.array([.9, .2, .8, .75]), .5), .375)

    def test_export_decision_check_detects_rounding_at_threshold(self):
        from .quality_ranker_train import decision_signatures, interior_thresholds
        groups = [{"context": {}, "candidates": [{"candidate_id": "a", "pose": np.eye(4)}]}]
        left = decision_signatures(groups, np.array([.5]), .5, 0.)
        right = decision_signatures(groups, np.array([.5 - 1e-8]), .5, 0.)
        # These probabilities pass allclose but give different accept decisions.
        np.testing.assert_allclose([.5], [.5 - 1e-8], rtol=1e-5, atol=1e-6)
        self.assertNotEqual(left, right)
        cutoffs = interior_thresholds([.2, .8])
        self.assertIn(.5, cutoffs)
        self.assertNotIn(.2, cutoffs)
        self.assertNotIn(.8, cutoffs)


if __name__ == "__main__":
    unittest.main()
