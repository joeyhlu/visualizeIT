"""Mock-only checks for the cached-head visibility threshold evaluator."""
from contextlib import nullcontext
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import quality_adaptation_data as data
from . import quality_adaptation_runtime as runtime
from . import quality_adaptation_visibility_threshold as visibility


class TensorDouble:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class HeadDouble:
    def __init__(self, *, zero_positive=False):
        self.zero_positive = zero_positive
        self.forward_calls = 0

    def __call__(self, tokens, shape):
        if shape != (data.SIZE, data.SIZE):
            raise AssertionError("frozen DPT output shape was not requested")
        if any(tokens[hook] is None for hook in data.HOOKS):
            raise AssertionError("cached decoder token hooks were not supplied")
        if any(tokens[hook] is not None for hook in set(range(12)) - set(data.HOOKS)):
            raise AssertionError("unexpected cached decoder token hook was supplied")
        self.forward_calls += 1
        flow = np.zeros((1, 2, data.SIZE, data.SIZE), dtype=np.float32)
        confidence = np.zeros((1, data.SIZE, data.SIZE), dtype=np.float32)
        if not self.zero_positive:
            confidence[0, :2, 0] = np.array([0.6, 0.25], dtype=np.float32)
            confidence[0, 2:4, 0] = np.float32(0.2)
        return TensorDouble(flow), TensorDouble(confidence)


class TorchDouble:
    def __init__(self):
        self.cuda = SimpleNamespace(is_available=lambda: False,
                                    synchronize=lambda: None,
                                    empty_cache=lambda: None)

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def from_numpy(value):
        return TensorDouble(value)


class VisibilityThresholdTests(unittest.TestCase):
    def test_strict_float32_boundary_ties_and_largest_threshold(self):
        confidence = np.zeros((data.SIZE, data.SIZE), dtype=np.float32)
        known = np.zeros((data.SIZE, data.SIZE), dtype=bool)
        visible = np.zeros((data.SIZE, data.SIZE), dtype=bool)
        confidence[:3, 0] = np.array([0.3, np.nextafter(np.float32(0.3), np.float32(1)), 0.5], dtype=np.float32)
        known[:3, 0] = True
        visible[:2, 0] = True
        counts = visibility._counts(confidence, known, visible, np.float32(0.3))
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 1, 1))

        fitted = visibility.fit_global_threshold(
            [np.array([0.5, 0.5, 0.25], dtype=np.float32)],
            [np.array([0.2], dtype=np.float32)],
            frozen_train_tp=2, frozen_train_fp=0)
        expected = np.nextafter(np.float32(0.5), np.float32(-np.inf))
        self.assertEqual(fitted["status"], "passed")
        self.assertEqual(np.float32(fitted["threshold"]).view(np.uint32), expected.view(np.uint32))
        self.assertEqual(fitted["adapted_train_tp"], 2)
        self.assertEqual(fitted["threshold_float32_bits"], f"0x{int(expected.view(np.uint32)):08x}")

    def test_fitted_threshold_minimizes_fp_at_required_recall(self):
        positive = np.array([0.8, 0.2], dtype=np.float32)
        negative = np.array([0.9, 0.1], dtype=np.float32)
        result = visibility.fit_global_threshold(positive, negative,
                                                 frozen_train_tp=1, frozen_train_fp=1)
        expected = np.nextafter(np.float32(0.8), np.float32(-np.inf))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(np.float32(result["threshold"]).view(np.uint32), expected.view(np.uint32))
        self.assertEqual(result["adapted_train_tp"], 1)
        self.assertEqual(result["adapted_train_fp"], 1)
        self.assertIn("minimizes FP", result["false_positive_optimality"])

    def test_impossible_zero_confidence_target_and_zero_baseline_case(self):
        impossible = visibility.fit_global_threshold(
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32), frozen_train_tp=1, frozen_train_fp=0)
        self.assertEqual(impossible["status"], "rejected")
        self.assertIsNone(impossible["threshold"])
        self.assertIn("zero-confidence", impossible["reason"])

        no_target = visibility.fit_global_threshold(
            np.array([], dtype=np.float32), np.array([0.7, 1.0], dtype=np.float32),
            frozen_train_tp=0, frozen_train_fp=0)
        self.assertEqual(no_target["status"], "passed")
        self.assertEqual(no_target["threshold"], 1.0)
        self.assertEqual((no_target["adapted_train_tp"], no_target["adapted_train_fp"]), (0, 0))

    def test_only_known_support_is_counted_and_visible_must_be_known(self):
        confidence = np.ones((data.SIZE, data.SIZE), dtype=np.float32)
        known = np.zeros((data.SIZE, data.SIZE), dtype=bool)
        visible = np.zeros((data.SIZE, data.SIZE), dtype=bool)
        known[:2, 0] = True
        visible[0, 0] = True
        counts = visibility._counts(confidence, known, visible, np.float32(0.3))
        self.assertEqual(counts, {"known_pixels": 2, "visible_pixels": 1,
                                  "tp": 1, "fp": 1, "fn": 0, "tn": 0})
        visible[2, 0] = True
        with self.assertRaisesRegex(ValueError, "subset"):
            visibility._counts(confidence, known, visible, np.float32(0.3))

    def test_budget_is_bounded_and_times_out_per_process(self):
        for value in (True, 0, 901, float("nan")):
            with self.assertRaises(ValueError):
                visibility.Budget(value)
        with patch.object(visibility.time, "monotonic", side_effect=[0.0, 0.6]):
            with self.assertRaisesRegex(TimeoutError, "per-process"):
                visibility.Budget(0.5)

    def test_visibility_loader_materializes_no_flow_or_unused_geometry_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature_path = root / "train-features.npz"
            labels_path = root / "train-labels.npz"
            known = np.zeros((data.SIZE, data.SIZE), dtype=bool)
            visible = np.zeros((data.SIZE, data.SIZE), dtype=bool)
            known[:2, 0] = True
            visible[0, 0] = True
            features = {f"tokens_{hook}": np.zeros((400, 768), dtype=np.float32)
                        for hook in data.HOOKS}
            features.update(template_mask=np.ones((data.SIZE, data.SIZE), dtype=bool),
                template_xyz_m=np.zeros((data.SIZE, data.SIZE, 3), dtype=np.float32),
                query_K=np.array([[280., 0., 140.], [0., 280., 140.], [0., 0., 1.]], dtype=np.float64),
                baseline_flow=np.zeros((2, data.SIZE, data.SIZE), dtype=np.float32),
                baseline_confidence=np.full((data.SIZE, data.SIZE), 0.4, dtype=np.float32))
            np.savez(feature_path, **features)
            np.savez(labels_path, flow=np.zeros((2, data.SIZE, data.SIZE), dtype=np.float32),
                     known=known, visible=visible)
            row = {"id": "train-000", "split": "train", "status": "available",
                   "features": {"path": feature_path.name, "bytes": feature_path.stat().st_size,
                                "sha256": data.sha256(feature_path)},
                   "labels": {"path": labels_path.name, "bytes": labels_path.stat().st_size,
                              "sha256": data.sha256(labels_path)}}
            ds = SimpleNamespace(root=root, rows=lambda split: [row] if split == "train" else [])
            original_load = np.load
            loaded = []

            class ArchiveDouble:
                def __init__(self, archive):
                    self.archive = archive

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    self.archive.close()

                def __getitem__(self, key):
                    loaded.append(key)
                    return self.archive[key]

            def load_spy(path, *, allow_pickle):
                self.assertFalse(allow_pickle)
                return ArchiveDouble(original_load(path, allow_pickle=allow_pickle))

            with patch.object(visibility.np, "load", side_effect=load_spy):
                tokens, baseline, loaded_known, loaded_visible = visibility.load_visibility_inputs(ds, row)
            self.assertEqual(set(loaded), visibility.SELECTED_FEATURE_KEYS | visibility.SELECTED_LABEL_KEYS)
            self.assertNotIn("flow", loaded)
            self.assertNotIn("baseline_flow", loaded)
            self.assertEqual(tokens[2].shape, (400, 768))
            self.assertEqual(baseline.dtype, np.float32)
            self.assertTrue(np.array_equal(loaded_known, known))
            self.assertTrue(np.array_equal(loaded_visible, visible))

            test_row = dict(row, split="test")
            with patch.object(data, "descriptor", side_effect=AssertionError("TEST payload descriptor touched")):
                with self.assertRaisesRegex(ValueError, "only selected available"):
                    visibility.load_visibility_inputs(ds, test_row)

    def make_fixture(self, root: Path):
        record = data.read_json(visibility.MANIFEST)
        export_bytes = b"mock-only-existing-head-export"
        export_path = root / "head.pt"
        export_path.write_bytes(export_bytes)
        record["export_sha256"] = hashlib.sha256(export_bytes).hexdigest()
        record["export_bytes"] = len(export_bytes)
        rows = []
        train_available = [f"train-{index:03d}" for index in range(95)]
        val_available = [f"view{index:02d}" for index in range(1, 21)]
        test_available = [f"test-{index:03d}" for index in range(24)]
        for split, available_ids, unavailable_ids in (
                ("train", train_available, ["train-unavailable"]),
                ("val", val_available, [f"view{index:02d}-unavailable" for index in range(21, 25)]),
                ("test", test_available, [])):
            for row_id in available_ids + unavailable_ids:
                available = row_id not in unavailable_ids
                row = {"id": row_id, "object_id": f"object-{split}",
                       "sequence_id": f"sequence-{split}", "capture_group": f"capture-{split}",
                       "query_rgb_sha256": hashlib.sha256((split + row_id).encode()).hexdigest(),
                       "split": split, "status": "available" if available else "unavailable"}
                if available:
                    row["features"] = {"path": f"payloads/{split}/{row_id}-features.npz",
                                       "bytes": 128, "sha256": hashlib.sha256((row_id + "-f").encode()).hexdigest()}
                    row["labels"] = {"path": f"payloads/{split}/{row_id}-labels.npz",
                                     "bytes": 128, "sha256": hashlib.sha256((row_id + "-l").encode()).hexdigest()}
                else:
                    row["failure_reason"] = "fixture_unavailable"
                rows.append(row)
        recipe = {"schema": "frozen_gotrack_decoder_recipe_v1", "annotation_inputs": [],
                  "source_lock_sha256": data.sha256(runtime.LOCK), "hooks": list(data.HOOKS),
                  "producer_sha256": data.sha256(visibility.ROOT / "bench/quality_adaptation_prepare.py")}
        recipe_hash = data.canonical_hash(recipe)
        (root / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
        index_rows = [{"id": row["id"], "split": row["split"], "status": row["status"],
                       **({"features": row["features"], "query_rgb_sha256": row["query_rgb_sha256"]}
                          if row["status"] == "available" else {})} for row in rows]
        index = {"recipe_sha256": recipe_hash, "rows": index_rows}
        (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
        doc = {"schema": data.SCHEMA, "conventions": data.CONVENTION,
               "provenance": {"feature_recipe_sha256": recipe_hash,
                              "source_lock_sha256": data.sha256(runtime.LOCK),
                              "annotation_inputs": [],
                              "label_origin": "synthetic_renderer_training_only"},
               "rows": rows}
        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps(doc), encoding="utf-8")
        dataset_hash = data.canonical_hash(doc)
        record["dataset_sha256"] = dataset_hash
        manifest_path = root / "artifact.json"
        manifest_path.write_text(json.dumps(record), encoding="utf-8")
        pins = {"PINNED_MANIFEST_SHA256": data.sha256(manifest_path),
                "PINNED_EXPORT_SHA256": record["export_sha256"],
                "PINNED_EXPORT_BYTES": record["export_bytes"],
                "PINNED_DATASET_SHA256": dataset_hash,
                "PINNED_ARTIFACT_ID": record["artifact_id"]}
        return SimpleNamespace(record=record, export_path=export_path,
                               manifest_path=manifest_path,
                               manifest_hash=data.sha256(manifest_path),
                               dataset_path=dataset_path, pins=pins)

    def invoke(self, root: Path, *, zero_positive=False, timeout=False,
               fail_receipt_write=False):
        fixture = self.make_fixture(root)
        output = root / "run-output"
        torch = TorchDouble()
        head = HeadDouble(zero_positive=zero_positive)
        loaded_splits = []

        def load_inputs(ds, row):
            loaded_splits.append(row["split"])
            if row["split"] == "val":
                self.assertTrue((output / visibility.SIDECAR_NAME).is_file(),
                                "VAL payload opened before frozen TRAIN evidence was persisted")
            tokens = [None] * 12
            shared = np.zeros((400, 768), dtype=np.float32)
            for hook in data.HOOKS:
                tokens[hook] = shared
            baseline = np.zeros((data.SIZE, data.SIZE), dtype=np.float32)
            known = np.zeros((data.SIZE, data.SIZE), dtype=bool)
            visible = np.zeros((data.SIZE, data.SIZE), dtype=bool)
            known[:4, 0] = True
            visible[:2, 0] = True
            baseline[:3, 0] = np.float32(0.4)
            baseline[3, 0] = np.float32(0.2)
            return tokens, baseline, known, visible

        if timeout:
            class ImmediateTimeout:
                def __init__(self, _seconds):
                    pass

                @staticmethod
                def check():
                    raise TimeoutError("per-process visibility evaluation budget expired")

            budget_patch = patch.object(visibility, "Budget", ImmediateTimeout)
        else:
            budget_patch = patch.object(visibility, "Budget", visibility.Budget)
        original_writer = visibility._write_exclusive_json

        def write_artifact(root, name, value):
            if fail_receipt_write and name == visibility.RECEIPT_NAME:
                raise OSError("injected final receipt write failure")
            return original_writer(root, name, value)

        with patch.multiple(visibility, **fixture.pins), \
                patch.object(runtime, "cloud_guard") as cloud_guard, \
                patch.object(runtime, "runtime_versions", return_value={"mock": "no neural runtime"}), \
                patch.object(runtime, "verify_sources", return_value={
                    "source_lock_sha256": fixture.record["training_lock_sha256"],
                    "upstream_revision": fixture.record["upstream_revision"]}), \
                patch.object(runtime, "load_head", return_value=head) as load_head, \
                patch.object(visibility.installer, "install_adapted_head",
                    side_effect=lambda network, context, *, device, torch_module: (
                        network, {"head_installation": {"requested": True, "installed": True}})) as install, \
                patch.object(visibility, "load_visibility_inputs", side_effect=load_inputs), \
                patch.object(visibility, "_write_exclusive_json", side_effect=write_artifact), \
                patch.dict(sys.modules, {"torch": torch}), budget_patch:
            result = visibility.run(artifact_manifest=fixture.manifest_path,
                artifact_manifest_sha256=fixture.manifest_hash, export=fixture.export_path,
                dataset=fixture.dataset_path, output=output, max_seconds=30,
                device="cpu", cloud_execution=True)
        cloud_guard.assert_called_once_with(True)
        if timeout:
            load_head.assert_not_called()
            install.assert_not_called()
        else:
            load_head.assert_called_once_with(initialize=False, device="cpu")
            install.assert_called_once()
        return result, fixture, head, loaded_splits

    def test_train_freeze_then_exact_val_pass_hashes_and_unavailable_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            result, fixture, head, loaded_splits = self.invoke(Path(temp))
            self.assertEqual(result["status"], "passed", result)
            self.assertTrue(result["candidate_accepted"])
            self.assertEqual(result["selected_train_rows"], 95)
            self.assertEqual(result["selected_validation_rows"], 20)
            self.assertEqual(result["inventory"]["train"], {"total": 96, "available": 95, "unavailable": 1})
            self.assertEqual(result["inventory"]["val"], {"total": 24, "available": 20, "unavailable": 4})
            self.assertEqual(result["inventory"]["test"]["total"], 24)
            self.assertEqual(result["test_payload_reads"], 0)
            self.assertEqual(result["validation_payload_reads"], 20)
            self.assertEqual(result["training_payload_reads"], 95)
            self.assertEqual(head.forward_calls, 115)
            self.assertEqual(loaded_splits.count("train"), 95)
            self.assertEqual(loaded_splits.count("val"), 20)
            self.assertNotIn("test", loaded_splits)
            self.assertEqual(result["frozen_train_pooled"]["tp"], 190)
            self.assertEqual(result["frozen_train_pooled"]["fp"], 95)
            expected_threshold = np.nextafter(np.float32(0.25), np.float32(-np.inf))
            self.assertEqual(np.float32(result["threshold_float32"]).view(np.uint32),
                             expected_threshold.view(np.uint32))
            self.assertEqual(result["validation_pooled"]["frozen_baseline_gt_0_3"]["tp"], 40)
            self.assertEqual(result["validation_pooled"]["frozen_baseline_gt_0_3"]["fp"], 20)
            self.assertEqual(result["validation_pooled"]["adapted_train_fitted_threshold"]["tp"], 40)
            self.assertEqual(result["validation_pooled"]["adapted_train_fitted_threshold"]["fp"], 0)
            view12 = next(row for row in result["validation_per_frame"] if row["id"] == "view12")
            self.assertEqual(view12["frozen_baseline_gt_0_3"]["tp"], 2)
            self.assertEqual(view12["adapted_train_fitted_threshold"]["tp"], 2)
            self.assertFalse(result["flow_metrics_computed"])
            self.assertFalse(result["serving_threshold_promoted"])
            sidecar_path = Path(temp) / "run-output" / visibility.SIDECAR_NAME
            sidecar = data.read_json(sidecar_path)
            self.assertEqual(result["train_threshold_sidecar_sha256"], data.sha256(sidecar_path))
            self.assertEqual(sidecar["artifact"]["manifest_sha256"], fixture.manifest_hash)
            self.assertEqual(sidecar["artifact"]["export_sha256"], fixture.record["export_sha256"])
            self.assertEqual(sidecar["dataset_sha256"], fixture.record["dataset_sha256"])
            self.assertEqual(sidecar["evaluator_code_sha256"], data.sha256(Path(visibility.__file__)))
            self.assertEqual(sidecar["train_selection_sha256"], data.canonical_hash(sidecar["train_selection"]))
            self.assertEqual(sidecar["threshold_fit"]["adapted_train_tp"], 190)
            self.assertEqual(sidecar["threshold_fit"]["adapted_train_fp"], 0)
            retained_receipt = data.read_json(Path(temp) / "run-output" / visibility.RECEIPT_NAME)
            self.assertEqual(retained_receipt["train_threshold_sidecar_sha256"], data.sha256(sidecar_path))
            self.assertEqual(retained_receipt, result)

    def test_train_rejection_persists_evidence_and_never_opens_val(self):
        with tempfile.TemporaryDirectory() as temp:
            result, fixture, _head, loaded_splits = self.invoke(Path(temp), zero_positive=True)
            self.assertEqual(result["status"], "rejected_training")
            self.assertFalse(result["candidate_accepted"])
            self.assertEqual(loaded_splits, ["train"] * 95)
            self.assertEqual(result["validation_payload_reads"], 0)
            self.assertNotIn("validation_per_frame", result)
            sidecar = data.read_json(Path(temp) / "run-output" / visibility.SIDECAR_NAME)
            self.assertEqual(sidecar["status"], "rejected")
            self.assertEqual(sidecar["threshold_fit"]["threshold"], None)
            self.assertEqual(sidecar["validation_payloads_read"], False)

    def test_timeout_returns_failure_without_loading_head(self):
        with tempfile.TemporaryDirectory() as temp:
            result, _fixture, _head, loaded_splits = self.invoke(Path(temp), timeout=True)
            self.assertEqual(result["status"], "failed")
            self.assertIn("per-process", result["error"]["message"])
            self.assertEqual(loaded_splits, [])

    def test_export_digest_mismatch_fails_before_model_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.make_fixture(Path(temp))
            fixture.export_path.write_bytes(b"corrupt")
            torch = TorchDouble()
            output = Path(temp) / "mismatch-output"
            with patch.multiple(visibility, **fixture.pins), \
                    patch.object(runtime, "cloud_guard"), \
                    patch.object(runtime, "runtime_versions", return_value={"mock": "none"}), \
                    patch.object(runtime, "verify_sources", return_value={
                        "source_lock_sha256": fixture.record["training_lock_sha256"],
                        "upstream_revision": fixture.record["upstream_revision"]}), \
                    patch.object(runtime, "load_head") as load_head, \
                    patch.dict(sys.modules, {"torch": torch}):
                result = visibility.run(artifact_manifest=fixture.manifest_path,
                    artifact_manifest_sha256=fixture.manifest_hash, export=fixture.export_path,
                    dataset=fixture.dataset_path, output=output, max_seconds=30, cloud_execution=True)
            self.assertEqual(result["status"], "failed")
            self.assertIn("digest or size mismatch", result["error"]["message"])
            load_head.assert_not_called()

    def test_final_receipt_write_failure_fails_closed_and_preserves_train_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            result, _fixture, _head, loaded_splits = self.invoke(
                Path(temp), fail_receipt_write=True)
            output = Path(temp) / "run-output"
            sidecar = output / visibility.SIDECAR_NAME
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["candidate_accepted"])
            self.assertEqual(result["receipt_write_error"]["type"], "OSError")
            self.assertIn("injected final receipt write failure",
                          result["receipt_write_error"]["message"])
            self.assertEqual(loaded_splits.count("train"), 95)
            self.assertEqual(loaded_splits.count("val"), 20)
            self.assertTrue(sidecar.is_file())
            self.assertFalse((output / visibility.RECEIPT_NAME).exists())
            self.assertEqual(data.read_json(sidecar)["status"], "passed")

    def test_cli_returns_nonzero_if_receipt_write_error_is_reported(self):
        simulated = {"status": "passed", "candidate_accepted": True,
                     "receipt_write_error": {"type": "OSError", "message": "injected"}}
        args = ["--export", "head.pt", "--dataset", "dataset.json",
                "--output", "unused-output", "--max-seconds", "30"]
        with patch.object(visibility, "run", return_value=simulated), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(visibility.main(args), 1)


if __name__ == "__main__":
    unittest.main()
