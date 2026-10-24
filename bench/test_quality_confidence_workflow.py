"""Analytical/mock tests for the portable confidence workflow.

These fixtures are software-only contract examples.  They are not model
outputs, accuracy evidence, or cloud verification.  The tests mock every stage
launch and do not import Torch or perform model inference.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import quality_confidence_workflow as workflow


HASH = "a" * 64
IDENTITY = {"fixture_identity": "analytical-mock-only"}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(**extra) -> dict:
    return {
        "artifact_purpose": workflow.ARTIFACT_PURPOSE,
        "policy": workflow.POLICY,
        "research_only": True,
        "test_read": False,
        **extra,
    }


def _mock_configs() -> dict:
    result = {}
    for alias, architecture in (("linear", "linear"), ("mlp", "mlp64_32")):
        effective = {"architecture": architecture, "analytical_fixture": True}
        result[alias] = {
            "path": f"/analytical fixture/{alias}.json",
            "sha256": HASH,
            "document": effective,
            "effective": effective,
            "identity_config_sha256": HASH,
        }
    return result


def _mock_source_hashes() -> dict[str, str]:
    return {name: HASH for name in workflow.SOURCE_PATHS}


def _create_analytical_artifacts(output: Path, configs: dict,
                                 source_hashes: dict[str, str]) -> None:
    """Create explicit non-neural documents for receipt-validation tests."""
    parity = {name: True for name in workflow._PARITY_FIELDS}
    for alias, architecture in (("linear", "linear"), ("mlp", "mlp64_32")):
        _write_json(output / f"smoke-{alias}" / "confidence-smoke-receipt.json", _contract(
            resume_exact_selected_model_parity=True,
            resume_exact_selected_model_optimizer_scheduler_scaler_rng_progress_parity=True,
            resume_exact_final_model_optimizer_scheduler_scaler_rng_progress_parity=True,
            export=_contract(**parity),
        ))
        run = output / alias
        run.mkdir(parents=True, exist_ok=True)
        best, final = run / "best.pt", run / "last.pt"
        best.write_bytes(b"analytical best fixture")
        final.write_bytes(b"analytical final fixture")
        _write_json(run / "completion.json", {
            "artifact_purpose": workflow.ARTIFACT_PURPOSE,
            "policy": workflow.POLICY,
            "complete": True,
            "test_read": False,
            "selected_checkpoint": "best.pt",
            "final_checkpoint": "last.pt",
            "selected_checkpoint_sha256": _hash(best),
            "final_checkpoint_sha256": _hash(final),
            "config_sha256": configs[alias]["identity_config_sha256"],
        })
        model = _contract(
            schema_version=3,
            architecture=architecture,
            provenance={
                "checkpoint_sha256": _hash(best),
                "identity": {
                    "data": IDENTITY,
                    "config": configs[alias]["effective"],
                    "config_sha256": configs[alias]["identity_config_sha256"],
                },
            },
        )
        _write_json(run / "model.json", model)
        (run / "model.onnx").write_bytes(b"analytical ONNX fixture")
        _write_json(run / "model.export.json", _contract(
            checkpoint_sha256=_hash(best),
            model_sha256=_hash(run / "model.json"),
            onnx_sha256=_hash(run / "model.onnx"),
            **parity,
        ))
        for split in ("train", "validation"):
            _write_json(run / f"{split}.json", _contract(
                kind="confidence-rejection-research-evaluation-v1",
                split=split,
                cache_identity=IDENTITY,
                checkpoint_sha256=_hash(run / "model.json"),
                source_hashes=workflow._evaluation_source_hashes(source_hashes),
            ))


class ConfidenceWorkflowTests(unittest.TestCase):
    def test_plan_is_exact_ten_stage_sequence_and_preserves_paths_with_spaces(self):
        root = Path("/repository with spaces")
        manifest = Path("/cache with spaces/cache_manifest.json")
        output = Path("/output with spaces/run")
        plans = workflow.build_stage_plan(manifest, output, root=root)
        self.assertEqual([plan["name"] for plan in plans], [
            "smoke-linear", "train-linear", "export-linear",
            "evaluate-linear-train", "evaluate-linear-validation",
            "smoke-mlp", "train-mlp", "export-mlp",
            "evaluate-mlp-train", "evaluate-mlp-validation",
        ])
        self.assertEqual(len(plans), 10)
        self.assertIn(str(manifest.resolve()), plans[0]["command"])
        self.assertIn(str(output.resolve() / "linear" / "model.json"), plans[3]["command"])
        self.assertIn(
            str(root.resolve() / "bench/runtime/ranker-confidence-linear-cpu.json"),
            plans[1]["command"],
        )
        self.assertIn(
            str(root.resolve() / "bench/runtime/ranker-confidence-mlp-cpu.json"),
            plans[6]["command"],
        )
        self.assertNotIn("confidence-resume", json.dumps(plans))

    def test_local_dry_run_reads_no_dataset_launches_nothing_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "new output"
            loader = Mock(side_effect=AssertionError("dataset loader called"))
            runner = Mock(side_effect=AssertionError("stage launched"))
            with patch.object(workflow, "load_research_dataset", loader), \
                 patch.object(workflow, "run_stage", runner), \
                 patch.object(workflow, "_load_frozen_configs",
                              side_effect=AssertionError("trainer imported")), \
                 redirect_stdout(io.StringIO()):
                result = workflow.main([
                    "--manifest", str(root / "missing cache_manifest.json"),
                    "--output", str(output), "--dry-run",
                ])
            self.assertEqual(len(result["stages"]), 10)
            self.assertFalse(output.exists())
            loader.assert_not_called()
            runner.assert_not_called()

    def test_cloud_guard_precedes_loader_and_output_access(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "must stay absent"
            loader = Mock()
            with patch.object(workflow.platform, "system", return_value="Windows"), \
                 patch.object(workflow, "load_research_dataset", loader), \
                 patch.object(workflow, "_validate_output_target",
                              side_effect=AssertionError("output accessed")):
                with self.assertRaisesRegex(RuntimeError, "approved Linux cloud host"):
                    workflow.main([
                        "--manifest", str(Path(folder) / "cache_manifest.json"),
                        "--output", str(output), "--cloud-execution",
                    ])
            loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_occupied_and_symlink_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            occupied = Path(folder) / "occupied"
            occupied.mkdir()
            with self.assertRaisesRegex(FileExistsError, "fresh output"):
                workflow._validate_output_target(occupied)
            absent = Path(folder) / "analytical-link"
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(FileExistsError, "symlink"):
                    workflow._validate_output_target(absent)

    def test_main_rejects_unresolved_output_symlink_before_resolve_loader_or_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            requested = (root / "direct-output-link").absolute()
            absent_target = root / "must-stay-absent"
            loader, runner = Mock(), Mock()
            original_is_symlink = Path.is_symlink

            def is_symlink(path):
                return True if path == requested else original_is_symlink(path)

            for extra in (("--dry-run",), ("--cloud-execution",)):
                with self.subTest(mode=extra[0]), \
                     patch.object(workflow.platform, "system", return_value="Linux"), \
                     patch.object(workflow, "load_research_dataset", loader), \
                     patch.object(workflow, "run_stage", runner), \
                     patch.object(Path, "is_symlink", autospec=True,
                                  side_effect=is_symlink), \
                     patch.object(Path, "resolve", autospec=True,
                                  side_effect=AssertionError("symlink was resolved")):
                    with self.assertRaisesRegex(FileExistsError, "symlink"):
                        workflow.main([
                            "--manifest", str(root / "cache_manifest.json"),
                            "--output", str(requested), *extra,
                        ])
            loader.assert_not_called()
            runner.assert_not_called()
            self.assertFalse(absent_target.exists())

    def test_execute_rejects_existing_requested_output_before_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "occupied"
            output.mkdir()
            loader = Mock()
            with patch.object(workflow.platform, "system", return_value="Linux"), \
                 patch.object(workflow, "load_research_dataset", loader):
                with self.assertRaisesRegex(FileExistsError, "fresh output"):
                    workflow.main([
                        "--manifest", str(root / "cache_manifest.json"),
                        "--output", str(output), "--cloud-execution",
                    ])
            loader.assert_not_called()

    def test_wrong_architecture_in_fixed_alias_fails_before_cache_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "bench/runtime"
            runtime.mkdir(parents=True)
            linear = json.loads((workflow.ROOT /
                "bench/runtime/ranker-confidence-linear-cpu.json").read_text())
            mlp = json.loads((workflow.ROOT /
                "bench/runtime/ranker-confidence-mlp-cpu.json").read_text())
            linear["architecture"] = "mlp64_32"
            _write_json(runtime / "ranker-confidence-linear-cpu.json", linear)
            _write_json(runtime / "ranker-confidence-mlp-cpu.json", mlp)
            manifest = root / "cache_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            output = root / "must-stay-absent"
            loader = Mock()
            load_configs = workflow._load_frozen_configs
            with patch.object(workflow.platform, "system", return_value="Linux"), \
                 patch.object(workflow, "load_research_dataset", loader), \
                 patch.object(workflow, "_load_frozen_configs",
                              side_effect=lambda: load_configs(root)):
                with self.assertRaisesRegex(ValueError, "linear.*must use linear"):
                    workflow.main([
                        "--manifest", str(manifest), "--output", str(output),
                        "--cloud-execution",
                    ])
            loader.assert_not_called()
            self.assertFalse(output.exists())

    def _run_mocked(self, root: Path, run_stage_mock: Mock, *, timeout=880.0):
        manifest = root / "cache_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        output = root / "workflow output"
        configs = _mock_configs()
        with patch.object(workflow.platform, "system", return_value="Linux"), \
             patch.object(workflow, "_load_frozen_configs", return_value=configs), \
             patch.object(workflow, "load_research_dataset",
                          return_value={"identity": IDENTITY}), \
             patch.object(workflow, "_source_hashes", return_value=_mock_source_hashes()), \
             patch.object(workflow, "run_stage", run_stage_mock), \
             patch.object(workflow, "_validate_required_outputs", return_value=["fixture"]), \
             patch.object(workflow, "OVERALL_TIMEOUT_SECONDS", timeout), \
             redirect_stdout(io.StringIO()):
            result = workflow.main([
                "--manifest", str(manifest), "--output", str(output),
                "--cloud-execution",
            ])
        return output, result

    def test_all_stages_finish_inside_one_decreasing_deadline_and_output_ceiling(self):
        calls = []

        def run(stage, command, output, **kwargs):
            calls.append((stage, kwargs))
            return {"stage": stage, "command": command, "status": "complete"}

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(workflow.time, "monotonic",
                          side_effect=[float(index) for index in range(23)]):
            output, result = self._run_mocked(Path(folder), Mock(side_effect=run), timeout=100.0)
        self.assertTrue(result["complete"])
        self.assertEqual([call[0] for call in calls], [
            "smoke-linear", "train-linear", "export-linear",
            "evaluate-linear-train", "evaluate-linear-validation",
            "smoke-mlp", "train-mlp", "export-mlp",
            "evaluate-mlp-train", "evaluate-mlp-validation",
        ])
        timeouts = [call[1]["timeout_s"] for call in calls]
        self.assertEqual(timeouts, [99.0, 97.0, 90.0, 60.0, 60.0,
                                    89.0, 87.0, 85.0, 60.0, 60.0])
        self.assertTrue(all(call[1]["storage_root"] == output for call in calls))
        self.assertTrue(all(call[1]["output_limit_bytes"] == 256 * 1024**2 for call in calls))

    def test_stage_failure_stops_launches_and_writes_atomic_incomplete_receipt(self):
        launched = []

        def run(stage, command, output, **kwargs):
            launched.append(stage)
            if len(launched) == 2:
                raise RuntimeError("analytical stage failure")
            return {"stage": stage, "command": command, "status": "complete"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "analytical stage failure"):
                output, _ = self._run_mocked(root, Mock(side_effect=run))
            receipt = json.loads((root / "workflow output" / workflow.AGGREGATE_NAME).read_text())
            self.assertEqual(launched, ["smoke-linear", "train-linear"])
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["failure"]["current_stage"], "train-linear")
            self.assertEqual(receipt["failure"]["exception_type"], "RuntimeError")
            self.assertFalse((root / "workflow output" /
                              f"{workflow.AGGREGATE_NAME}.tmp").exists())

    def test_final_stage_overrun_is_incomplete_before_artifact_validation(self):
        launched = []

        def run(stage, command, output, **kwargs):
            launched.append(stage)
            return {"stage": stage, "command": command, "status": "complete"}

        # Start + nine before/after checks remain in budget.  The final stage
        # starts at t=90 and returns at t=101, after the shared t=100 deadline.
        clock = [0.0] + [float(index) for index in range(1, 19)] + [90.0, 101.0, 102.0]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(workflow.time, "monotonic", side_effect=clock):
            root = Path(folder)
            with self.assertRaisesRegex(TimeoutError, "after evaluate-mlp-validation"):
                self._run_mocked(root, Mock(side_effect=run), timeout=100.0)
            receipt_path = root / "workflow output" / workflow.AGGREGATE_NAME
            receipt = json.loads(receipt_path.read_text())
            temporary_exists = receipt_path.with_suffix(receipt_path.suffix + ".tmp").exists()
        self.assertEqual(len(launched), 10)
        self.assertEqual(len(receipt["stages"]), 10)
        self.assertFalse(receipt["complete"])
        self.assertEqual(receipt["current_stage"], "evaluate-mlp-validation")
        self.assertEqual(receipt["failure"]["exception_type"], "TimeoutError")
        self.assertEqual(receipt["required_artifacts"], [])
        self.assertFalse(temporary_exists)

    def test_artifact_validation_overrun_cannot_mark_complete(self):
        def run(stage, command, output, **kwargs):
            return {"stage": stage, "command": command, "status": "complete"}

        # All ten stages return by t=20.  Artifact validation then consumes the
        # remaining budget, so its postcondition check sees t=101.
        clock = [0.0] + [float(index) for index in range(1, 21)] + [101.0, 102.0]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(workflow.time, "monotonic", side_effect=clock):
            root = Path(folder)
            with self.assertRaisesRegex(TimeoutError, "after artifact validation"):
                self._run_mocked(root, Mock(side_effect=run), timeout=100.0)
            receipt_path = root / "workflow output" / workflow.AGGREGATE_NAME
            receipt = json.loads(receipt_path.read_text())
            temporary_exists = receipt_path.with_suffix(receipt_path.suffix + ".tmp").exists()
        self.assertEqual(len(receipt["stages"]), 10)
        self.assertFalse(receipt["complete"])
        self.assertEqual(receipt["current_stage"], "artifact-validation")
        self.assertEqual(receipt["required_artifacts"], ["fixture"])
        self.assertEqual(receipt["failure"]["exception_type"], "TimeoutError")
        self.assertFalse(temporary_exists)

    def test_missing_required_artifact_prevents_complete(self):
        def run(stage, command, output, **kwargs):
            return {"stage": stage, "command": command, "status": "complete"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "cache_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            output = root / "output"
            with patch.object(workflow.platform, "system", return_value="Linux"), \
                 patch.object(workflow, "_load_frozen_configs", return_value=_mock_configs()), \
                 patch.object(workflow, "load_research_dataset",
                              return_value={"identity": IDENTITY}), \
                 patch.object(workflow, "_source_hashes", return_value=_mock_source_hashes()), \
                 patch.object(workflow, "run_stage", side_effect=run), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "smoke receipt"):
                    workflow.main([
                        "--manifest", str(manifest), "--output", str(output),
                        "--cloud-execution",
                    ])
            receipt = json.loads((output / workflow.AGGREGATE_NAME).read_text())
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["failure"]["current_stage"], "artifact-validation")
            self.assertEqual(len(receipt["stages"]), 10)

    def test_analytical_receipts_require_hash_bound_completion_export_and_evaluations(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            configs = _mock_configs()
            source_hashes = _mock_source_hashes()
            _create_analytical_artifacts(output, configs, source_hashes)
            with patch("bench.quality_pose_ranker.ConfidenceRejectionRanker.from_checkpoint") as validator:
                artifacts = workflow._validate_required_outputs(
                    output, configs, IDENTITY, source_hashes)
            self.assertEqual(validator.call_count, 2)
            self.assertEqual(len(artifacts), 18)
            (output / "mlp" / "validation.json").unlink()
            with patch("bench.quality_pose_ranker.ConfidenceRejectionRanker.from_checkpoint"):
                with self.assertRaisesRegex(ValueError, "validation evaluation"):
                    workflow._validate_required_outputs(
                        output, configs, IDENTITY, source_hashes)

    def test_evaluation_source_closure_requires_exact_captured_keys_and_hashes(self):
        expected_names = {
            "quality_confidence_evaluate.py", "quality_confidence_rejection.py",
            "quality_contract.py", "quality_gotrack.py", "quality_offline.py",
            "quality_pipeline.py", "quality_pose_ranker.py", "quality_ranker_data.py",
            "quality_ranker_features.py", "quality_ranker_train.py", "quality_runner.py",
        }
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            configs = _mock_configs()
            source_hashes = _mock_source_hashes()
            expected = workflow._evaluation_source_hashes(source_hashes)
            self.assertEqual(set(expected), expected_names)
            _create_analytical_artifacts(output, configs, source_hashes)
            evaluation_path = output / "linear" / "train.json"
            original = json.loads(evaluation_path.read_text())
            stale = dict(expected)
            stale["quality_runner.py"] = "b" * 64
            missing = dict(expected)
            missing.pop("quality_runner.py")
            for name, closure in (("stale-valid-sha256", stale),
                                  ("missing-source-key", missing)):
                with self.subTest(name=name):
                    changed = dict(original)
                    changed["source_hashes"] = closure
                    _write_json(evaluation_path, changed)
                    with patch("bench.quality_pose_ranker.ConfidenceRejectionRanker.from_checkpoint"):
                        with self.assertRaisesRegex(ValueError, "train evaluation"):
                            workflow._validate_required_outputs(
                                output, configs, IDENTITY, source_hashes)
            _write_json(evaluation_path, original)

    def test_source_contains_no_torch_import_or_test_split_command(self):
        source = Path(workflow.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        commands = json.dumps(workflow.build_stage_plan(
            Path("cache_manifest.json"), Path("fresh-output")))
        self.assertNotIn('"test"', commands)
        self.assertNotIn("--split', 'test", commands)


if __name__ == "__main__":
    unittest.main()
