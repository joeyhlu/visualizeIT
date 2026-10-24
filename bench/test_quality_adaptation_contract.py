"""CPU numerical/contract tests only: no model construction or optimization."""
from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from bench import quality_adaptation_data as data
from bench import quality_adaptation_runtime as runtime
from bench import quality_adaptation_prepare as prepare
from bench.quality_adaptation_prepare import project_supervision, extract_one, ExtractionConfig, crops_from_retained_capture
from bench.quality_adaptation_train import TrainConfig, epoch_batches, train
from bench.quality_adaptation_manifest import assemble
from bench.quality_adaptation_train import preflight, validate_feature_receipt


def geometry():
    mask = np.zeros((280, 280), bool)
    mask[10, 10:12] = True
    xyz = np.zeros((280, 280, 3), np.float32)
    xyz[10, 10] = [.1, .1, 1.]
    xyz[10, 11] = [.11, .1, 1.]
    return {"template_mask": mask, "template_xyz_m": xyz,
            "query_K": np.diag([100., 100., 1.])}


def arrays(marker=0):
    features = geometry()
    features.update({f"tokens_{hook}": np.full((400, 768), marker, np.float32) for hook in data.HOOKS})
    features.update(baseline_flow=np.zeros((2, 280, 280), np.float32),
                    baseline_confidence=np.zeros((280, 280), np.float32))
    labels = {"flow": np.zeros((2, 280, 280), np.float32),
              "visible": features["template_mask"].copy(), "known": features["template_mask"].copy()}
    # Different valid labels give distinct content identities across holdouts.
    labels["flow"][0, 10, 10:12] = marker * .01
    return features, labels


def save_npz(root, name, value):
    path = root / name
    np.savez(path, **value)
    return {"path": name, "bytes": path.stat().st_size, "sha256": data.sha256(path)}


def dataset_fixture(root):
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        row = {"id": f"row-{i}", "object_id": f"object-{i}", "sequence_id": f"seq-{i}",
               "capture_group": f"capture-{i}", "query_rgb_sha256": f"{i + 1:064x}", "split": split,
               "status": "available"}
        if split != "test":
            f, y = arrays(i)
            row["features"] = save_npz(root, f"features-{i}.npz", f)
            row["labels"] = save_npz(root, f"labels-{i}.npz", y)
        else:
            # Sealed test payloads intentionally do not exist on this host.
            row["features"] = {"path": "sealed/test-features.npz", "bytes": 100, "sha256": "e" * 64}
            row["labels"] = {"path": "sealed/test-labels.npz", "bytes": 100, "sha256": "f" * 64}
        rows.append(row)
    doc = {"schema": data.SCHEMA, "conventions": data.CONVENTION,
           "provenance": {"feature_recipe_sha256": "1" * 64, "source_lock_sha256": "2" * 64,
                          "annotation_inputs": [], "label_origin": "independent_annotations"}, "rows": rows}
    path = root / "manifest.json"
    path.write_text(json.dumps(doc))
    return path, doc


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, self.doc = dataset_fixture(self.root)

    def write(self):
        self.manifest.write_text(json.dumps(self.doc))

    def test_sealed_test_payloads_not_opened_and_explicitly_rejected(self):
        ds = data.load_dataset(self.manifest)
        self.assertEqual(data.data_report(ds)["splits"]["test"]["total"], 1)
        with self.assertRaisesRegex(ValueError, "sealed test"):
            ds.arrays(ds.rows("test")[0])

    def test_each_indirect_group_leakage_rejected(self):
        for key in ("object_id", "sequence_id", "capture_group", "query_rgb_sha256"):
            with self.subTest(key=key):
                old = self.doc["rows"][1][key]
                self.doc["rows"][1][key] = self.doc["rows"][0][key]
                self.write()
                with self.assertRaisesRegex(ValueError, "split leakage"):
                    data.load_dataset(self.manifest, inspect_arrays=False)
                self.doc["rows"][1][key] = old

    def test_duplicate_archive_across_splits_rejected(self):
        self.doc["rows"][1]["features"]["sha256"] = self.doc["rows"][0]["features"]["sha256"]
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate archive"):
            data.load_dataset(self.manifest, inspect_arrays=False)

    def test_annotations_in_feature_provenance_rejected(self):
        self.doc["provenance"]["annotation_inputs"] = ["reference_pose.json"]
        self.write()
        with self.assertRaisesRegex(ValueError, "input-only"):
            data.load_dataset(self.manifest)

    def test_units_and_camera_mismatch_rejected(self):
        self.doc["conventions"] = dict(data.CONVENTION, object_xyz="millimetres")
        self.write()
        with self.assertRaisesRegex(ValueError, "conventions"):
            data.load_dataset(self.manifest)

    def test_failed_frames_retained_in_availability(self):
        row = dict(self.doc["rows"][0], id="failed-frame", status="unavailable", failure_reason="mask_missing")
        row.pop("features")
        row.pop("labels")
        self.doc["rows"].append(row)
        self.write()
        report = data.data_report(data.load_dataset(self.manifest))
        self.assertEqual(report["splits"]["train"]["total"], 2)
        self.assertEqual(report["splits"]["train"]["available"], 1)

    def test_changed_artifact_rejected(self):
        with (self.root / "features-0.npz").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            data.load_dataset(self.manifest)

    def test_path_traversal_rejected(self):
        for relative in ("../outside.npz", "/tmp/out.npz", "C:/secret", "a\\b"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                data.local_path(self.root, relative)

    def test_unexpected_archive_member_rejected_before_loading(self):
        p = self.root / "injected.npz"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("pose.npy", b"not an array")
        with self.assertRaisesRegex(ValueError, "members"):
            data.load_npz(p, data.FEATURE_KEYS)

    def test_duplicate_json_rejected(self):
        self.manifest.write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            data.read_json(self.manifest)

    def receipt(self, extra_rows=()):
        recipe = {"schema": "frozen_gotrack_decoder_recipe_v1", "annotation_inputs": [],
                  "source_lock_sha256": data.sha256(runtime.LOCK), "hooks": list(data.HOOKS),
                  "producer_sha256": data.sha256(runtime.ROOT / "bench/quality_adaptation_prepare.py")}
        index = {"recipe_sha256": data.canonical_hash(recipe), "rows": [
            {"id": row["id"], "status": "available", "features": row["features"],
             "query_rgb_sha256": row["query_rgb_sha256"]}
            for row in self.doc["rows"] if row["split"] != "test" and row["status"] == "available"] + list(extra_rows)}
        (self.root / "recipe.json").write_text(json.dumps(recipe))
        (self.root / "index.json").write_text(json.dumps(index))
        self.doc["provenance"].update(feature_recipe_sha256=data.canonical_hash(recipe),
                                      source_lock_sha256=data.sha256(runtime.LOCK))
        self.write()
        return index

    def test_preflight_rejects_forged_query_hash_even_when_feature_receipt_matches(self):
        self.receipt()
        self.doc["rows"][0]["query_rgb_sha256"] = "c" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "query RGB hash differs"):
            preflight(self.manifest, TrainConfig())

    def test_preflight_rejects_dropped_failed_receipt_and_duplicate_ids(self):
        failed = dict(self.doc["rows"][0], id="failed-extraction", status="unavailable",
                      failure_reason="decoder_failure")
        failed.pop("features")
        failed.pop("labels")
        self.doc["rows"].append(failed)
        index = self.receipt([{"id": failed["id"], "status": "unavailable", "failure_reason": "decoder_failure"}])
        validate_feature_receipt(data.load_dataset(self.manifest))
        self.doc["rows"].pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "omits extracted rows/failures"):
            preflight(self.manifest, TrainConfig())
        index["rows"].pop()
        index["rows"].append(dict(index["rows"][0]))
        (self.root / "index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "duplicate extraction ids"):
            preflight(self.manifest, TrainConfig())

    def test_actual_query_pixels_bind_extraction_and_reject_forged_split_hashes(self):
        crops_rows = []
        query = np.full((280, 280, 3), 17, np.uint8)
        for i in range(2):
            crops = dict(geometry(), query_rgb=query, template_rgb=np.full_like(query, i))
            crops_rows.append({"id": f"row-{i}", "crops": save_npz(self.root, f"crop-{i}.npz", crops)})
        crop_manifest = self.root / "crops.json"
        crop_manifest.write_text(json.dumps({"schema": "correspondence_input_crops_v1",
                                            "conventions": data.CONVENTION, "rows": crops_rows}))
        output = self.root / "extracted"
        def fake_extract(network, crops, *, device):
            return arrays(int(crops["template_rgb"][0, 0, 0]))[0]
        with patch.object(prepare, "cloud_guard"), patch.object(prepare, "runtime_versions"), \
                patch.object(prepare, "verify_sources", return_value={"source_lock_sha256": data.sha256(runtime.LOCK)}), \
                patch.object(prepare, "load_frozen_network", return_value=object()), \
                patch.object(prepare, "extract_one", side_effect=fake_extract):
            self.assertEqual(prepare.extract(crop_manifest, output, ExtractionConfig()), "completed")
        index_bytes = (output / "index.json").read_bytes()
        index = data.read_json(output / "index.json")
        expected = data.query_rgb_hash(query)
        self.assertEqual([r["query_rgb_sha256"] for r in index["rows"]], [expected, expected])
        self.assertNotEqual(index["rows"][0]["features"]["sha256"], index["rows"][1]["features"]["sha256"])
        inventory_rows = []
        for i, original in enumerate(self.doc["rows"]):
            row = dict(original)
            if row["split"] != "test":
                row.pop("features")
                row["labels"] = save_npz(output, f"labels-{i}.npz", arrays(i)[1])
            inventory_rows.append(row)
        inventory = output / "inventory.json"
        def write_inventory():
            inventory.write_text(json.dumps({"schema": "correspondence_label_inventory_v1",
                                            "label_origin": "independent_annotations", "rows": inventory_rows}))
        write_inventory()
        with self.assertRaisesRegex(ValueError, "query RGB hash differs"):
            assemble(output, inventory, output / "assembled.json")
        self.assertFalse((output / "assembled.json").exists())
        self.assertEqual((output / "index.json").read_bytes(), index_bytes)
        for row in inventory_rows[:2]:
            row["query_rgb_sha256"] = expected
        write_inventory()
        with self.assertRaisesRegex(ValueError, "split leakage: shared query_rgb_sha256"):
            assemble(output, inventory, output / "assembled.json")
        self.assertEqual(list(output.glob(".adaptation-manifest-*")), [])

    def test_extractor_failure_preserves_failed_and_unprocessed_index_rows(self):
        crops = dict(geometry(), query_rgb=np.zeros((280, 280, 3), np.uint8),
                     template_rgb=np.zeros((280, 280, 3), np.uint8))
        descriptor = save_npz(self.root, "crop.npz", crops)
        manifest = self.root / "crops.json"
        manifest.write_text(json.dumps({"schema": "correspondence_input_crops_v1", "conventions": data.CONVENTION,
                                        "rows": [{"id": str(i), "crops": descriptor} for i in range(2)]}))
        output = self.root / "failed-extraction"
        with patch.object(prepare, "cloud_guard"), patch.object(prepare, "runtime_versions"), \
                patch.object(prepare, "verify_sources", return_value={}), \
                patch.object(prepare, "load_frozen_network", return_value=object()), \
                patch.object(prepare, "extract_one", side_effect=RuntimeError("mock decoder failure")):
            with self.assertRaisesRegex(RuntimeError, "mock decoder failure"):
                prepare.extract(manifest, output, ExtractionConfig())
        index = data.read_json(output / "index.json")
        self.assertEqual(index["status"], "failed")
        self.assertEqual([r["failure_reason"] for r in index["rows"]], ["mock decoder failure", "not_processed"])
        self.assertTrue(all(r["status"] == "unavailable" for r in index["rows"]))
        self.assertEqual(list(output.glob(".adaptation-*.tmp")), [])

    def test_assembly_and_preflight_bind_extractor_receipt_without_test_reads(self):
        recipe = {"schema": "frozen_gotrack_decoder_recipe_v1", "annotation_inputs": [],
                  "source_lock_sha256": data.sha256(runtime.LOCK), "hooks": list(data.HOOKS),
                  "producer_sha256": data.sha256(runtime.ROOT / "bench/quality_adaptation_prepare.py")}
        (self.root / "recipe.json").write_text(json.dumps(recipe))
        index_rows = [{"id": row["id"], "status": "available", "features": row["features"],
                       "query_rgb_sha256": row["query_rgb_sha256"]}
                      for row in self.doc["rows"] if row["split"] != "test"]
        (self.root / "index.json").write_text(json.dumps({"recipe_sha256": data.canonical_hash(recipe), "rows": index_rows}))
        inventory_rows = []
        for row in self.doc["rows"]:
            row = dict(row)
            if row["split"] != "test": row.pop("features")
            inventory_rows.append(row)
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps({"schema": "correspondence_label_inventory_v1",
                                        "label_origin": "independent_annotations", "rows": inventory_rows}))
        output = self.root / "assembled.json"
        assemble(self.root, inventory, output)
        # Manufactured feature/label joins do not execute the historical
        # trainer checkout. Its production source authentication stays intact;
        # only that environment-dependent boundary is doubled in this test.
        lock = data.read_json(runtime.LOCK)
        source = dict(source_lock_sha256=data.sha256(runtime.LOCK),
                      upstream_revision=lock['gotrack']['revision'],
                      source_files_sha256=data.canonical_hash(lock['gotrack']['files']))
        with patch('bench.quality_adaptation_train.verify_sources', return_value=source) as verified:
            result = preflight(output, TrainConfig())
            verified.assert_called_once_with()
        self.assertEqual(result["data"]["splits"]["test"]["total"], 1)
        self.assertFalse(result["test_arrays_opened"])
        self.assertEqual(list(self.root.glob(".adaptation-manifest-*")), [])
        index_rows[0]["features"] = dict(index_rows[0]["features"], sha256="b" * 64)
        (self.root / "index.json").write_text(json.dumps({"recipe_sha256": data.canonical_hash(recipe), "rows": index_rows}))
        with self.assertRaisesRegex(ValueError, "differs from extractor"):
            validate_feature_receipt(data.load_dataset(output))


class NumericalContractTests(unittest.TestCase):
    def test_query_hash_uses_c_order_decoded_uint8_crop_pixels(self):
        import hashlib
        rgb = np.arange(280 * 280 * 3, dtype=np.uint8).reshape(280, 280, 3)
        expected = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
        self.assertEqual(data.query_rgb_hash(rgb), expected)
        self.assertEqual(data.query_rgb_hash(np.asfortranarray(rgb)), expected)
        for wrong in (rgb.astype(np.float32), rgb[:, :, 0], rgb[:279]):
            with self.assertRaisesRegex(ValueError, "uint8"):
                data.query_rgb_hash(wrong)

    def test_existing_capture_conversion_preserves_integer_centers_and_metres(self):
        g = geometry()
        depth = np.ones((1, 280, 280), np.float32) * 1000
        capture = {"query_rgb_crop": np.ones((1, 3, 280, 280), np.float32),
                   "template_rgb": np.zeros((1, 3, 280, 280), np.float32),
                   "rendered_depth_mm": depth, "rendered_mask": g["template_mask"][None],
                   "crop_k": g["query_K"], "current_crop_camera_from_object_mm": np.eye(4)}
        crops = crops_from_retained_capture(capture)
        # Backprojection: (u,v)=(10,10),f=100,Z=1000mm ->(.1,.1,1)m.
        np.testing.assert_allclose(crops["template_xyz_m"][10, 10], [.1, .1, 1.], atol=1e-7)
        self.assertEqual(crops["query_rgb"].dtype, np.uint8)
        self.assertTrue(np.all(crops["query_rgb"] == 255))

    def labels(self, *, tx=0., depth=1.):
        g = geometry()
        pose = np.eye(4)
        pose[0, 3] = tx
        z = np.full((280, 280), depth, np.float32)
        observed = np.ones((280, 280), bool)
        return project_supervision(**g, query_T_cam_from_object=pose, observed_depth_m=z,
                                   observed_object_mask=observed, depth_tolerance_m=.001)

    def test_identity_projection_and_metric_translation(self):
        identity = self.labels()
        translated = self.labels(tx=.01)
        # Analytic pinhole projection u=fx*X/Z: X=.1m,Z=1m,f=100px ->10px.
        np.testing.assert_allclose(identity["flow"][:, 10, 10], [0., 0.], atol=2e-6)
        # A .01m X translation produces exactly 1px at f=100px and Z=1m.
        np.testing.assert_allclose(translated["flow"][:, 10, 10], [1., 0.], atol=2e-6)
        self.assertTrue(translated["visible"][10, 10])

    def test_occlusion_and_unknown_depth_are_distinct(self):
        g = geometry()
        depth = np.ones((280, 280), np.float32)
        depth[10, 10] = .5
        label = project_supervision(**g, query_T_cam_from_object=np.eye(4), observed_depth_m=depth,
                                    observed_object_mask=np.ones((280, 280), bool), depth_tolerance_m=.001)
        self.assertTrue(label["known"][10, 10])
        self.assertFalse(label["visible"][10, 10])
        depth[10, 10] = 0
        label = project_supervision(**g, query_T_cam_from_object=np.eye(4), observed_depth_m=depth,
                                    observed_object_mask=np.ones((280, 280), bool), depth_tolerance_m=.001)
        self.assertFalse(label["known"][10, 10])

    def test_projected_silhouette_alone_cannot_label_foreground(self):
        g = geometry()
        observed = np.ones((280, 280), bool)
        observed[10, 10] = False
        label = project_supervision(**g, query_T_cam_from_object=np.eye(4),
                                    observed_depth_m=np.ones((280, 280), np.float32),
                                    observed_object_mask=observed, depth_tolerance_m=.001)
        self.assertFalse(label["visible"][10, 10])
        self.assertTrue(label["known"][10, 10])

    def test_improper_pose_and_nonfinite_geometry_rejected(self):
        g = geometry()
        pose = np.eye(4)
        pose[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "proper rigid"):
            project_supervision(**g, query_T_cam_from_object=pose,
                                observed_depth_m=np.ones((280, 280), np.float32),
                                observed_object_mask=np.ones((280, 280), bool), depth_tolerance_m=.001)

    def test_epe_analytical_345_triangle_and_object_macro(self):
        pred = np.zeros((2, 280, 280), np.float32)
        pred[:, 0, 0] = [3, 4]
        visible = np.zeros((280, 280), bool)
        visible[0, 0] = True
        # Euclidean displacement sqrt(3^2+4^2)=5px, independent of image size.
        self.assertEqual(data.flow_metrics(pred, np.zeros_like(pred), visible)["epe_px"], 5.)
        summary = data.object_summary([{"object_id": "a", "epe_px": 1.}] * 10 +
                                      [{"object_id": "b", "epe_px": 3.}], seed=42)
        # Macro object mean is (1+3)/2=2, not frame mean 13/11.
        self.assertEqual(summary["object_macro_epe_px"], 2.)
        self.assertEqual(summary["bootstrap_95ci_px"], [1., 3.])

    def test_single_object_cannot_supply_independent_ci(self):
        summary = data.object_summary([{"object_id": "a", "epe_px": 1.}], seed=42)
        self.assertIsNone(summary["bootstrap_95ci_px"])


class RuntimeTests(unittest.TestCase):
    def test_no_cloud_assertion_fails_before_torch_or_model_load(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(runtime.platform, "system", return_value="Linux"):
            with patch("bench.quality_adaptation_train.load_head") as load:
                with self.assertRaisesRegex(RuntimeError, "approved Linux"):
                    train(Path("none"), Path("none"), TrainConfig(), cloud_execution=True)
                load.assert_not_called()

    def test_windows_rejects_even_with_cloud_flag(self):
        with patch.dict("os.environ", {"VISUALIZEIT_CLOUD": "1"}), patch.object(runtime.platform, "system", return_value="Windows"):
            with self.assertRaises(RuntimeError):
                runtime.cloud_guard(True)

    def test_config_validation_limits_and_nan(self):
        TrainConfig().validate()
        for cfg in (replace(TrainConfig(), amp=True), replace(TrainConfig(), batch_size=True),
                    replace(TrainConfig(), learning_rate=float("nan")), replace(TrainConfig(), max_seconds=90000),
                    replace(TrainConfig(), max_output_bytes=100), replace(TrainConfig(), epochs=1001)):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                cfg.validate()

    def test_deterministic_epoch_cursor_partition(self):
        rows = [{"id": str(i)} for i in range(11)]
        cfg = replace(TrainConfig(), batch_size=3)
        a = epoch_batches(rows, 2, cfg)
        b = epoch_batches(list(reversed(rows)), 2, cfg)
        self.assertEqual(a, b)
        self.assertEqual(sum(a[:2] + b[2:], []), sum(a, []))

    def test_resume_rejects_changed_dataset_config_sources(self):
        state = {"schema": "correspondence_head_checkpoint_v1", "dataset_sha256": "d", "config_sha256": "c",
                 "source_lock_sha256": "s", "epoch": 1, "batch_cursor": 2, "steps": 3,
                 "bad_epochs": 0, "elapsed_seconds": 5.}
        runtime.check_resume(state, identity="d", config_hash="c", source_hash="s")
        for args in (("changed", "c", "s"), ("d", "changed", "s"), ("d", "c", "changed")):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                runtime.check_resume(state, identity=args[0], config_hash=args[1], source_hash=args[2])

    def test_atomic_failure_preserves_previous_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "last.pt").write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "serializer failed"):
                with runtime.atomic_stream(root, root / "last.pt", max_file_bytes=10,
                                           max_output_bytes=100, reserve_bytes=0) as f:
                    f.write(b"new")
                    raise RuntimeError("serializer failed")
            self.assertEqual((root / "last.pt").read_bytes(), b"old")
            self.assertEqual([p.name for p in root.iterdir()], ["last.pt"])

    def test_atomic_size_budget_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "reserved file"):
                with runtime.atomic_stream(root, root / "last.pt", max_file_bytes=4,
                                           max_output_bytes=100, reserve_bytes=0) as f:
                    f.write(b"too many bytes")
            self.assertEqual(list(root.iterdir()), [])

    def test_checkpoint_retention_names_and_serializer_wiring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = SimpleNamespace(max_checkpoint_bytes=16, max_output_bytes=100)
            calls = []
            fake = SimpleNamespace(save=lambda value, f: (calls.append(value), f.write(b"serialized")))
            runtime.save_checkpoint(root, "last.pt", {"epoch": 3}, cfg, torch_module=fake)
            self.assertEqual(calls, [{"epoch": 3}])
            self.assertEqual((root / "last.pt").read_bytes(), b"serialized")
            with self.assertRaisesRegex(ValueError, "retention"):
                runtime.save_checkpoint(root, "epoch-3.pt", {}, cfg, torch_module=fake)


class Tensor:
    """Numpy mock for extraction call wiring, not a model or Torch substitute."""
    def __init__(self, value): self.value = value
    def to(self, **kw): return self
    def __truediv__(self, other): return Tensor(self.value / other)
    def chunk(self, count, dim): return tuple(Tensor(v) for v in np.array_split(self.value, count, axis=dim))
    def __getitem__(self, key): return Tensor(self.value[key])
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value.astype(np.float32)


class WiringTests(unittest.TestCase):
    def test_export_loader_strictly_replaces_head_only(self):
        import sys
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temp:
            export_path = Path(temp) / "adapted.pt"
            export_path.write_bytes(b"mock-checkpoint")
            head = Mock()
            network = SimpleNamespace(pose_head=head, eval=Mock())
            state = {"schema": "correspondence_head_export_v1", "source_lock_sha256": data.sha256(runtime.LOCK),
                     "base_checkpoint_sha256": runtime.GOTRACK_CHECKPOINT, "head": {"head.0.weight": "fake"}}
            fake_torch = SimpleNamespace(load=Mock(return_value=state))
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(runtime, "cloud_guard"), patch.object(runtime, "verify_sources"):
                runtime.load_export_into_network(network, export_path, expected_sha256=data.sha256(export_path), cloud_execution=True)
            head.load_state_dict.assert_called_once_with(state["head"], strict=True)
            fake_torch.load.assert_called_once_with(export_path, map_location="cpu", weights_only=True)
            network.eval.assert_called_once()

    def test_extraction_calls_actual_interfaces_and_never_passes_labels(self):
        from contextlib import nullcontext
        calls = []
        reference = [Tensor(np.ones((1, 400, 768), np.float32) * i) for i in range(12)]
        def backbone(rgb):
            calls.append(("backbone", rgb.value.shape, float(rgb.value.max())))
            return {"feature_maps": Tensor(np.zeros((2, 384, 20, 20), np.float32))}
        def decoder(**kwargs):
            calls.append(("decoder", set(kwargs), kwargs["crop_size"]))
            return None, reference
        def head(tokens, size):
            self.assertIs(tokens, reference)
            calls.append(("head", size))
            return Tensor(np.zeros((1, 2, 280, 280), np.float32)), Tensor(np.ones((1, 280, 280), np.float32) * .5)
        crops = geometry()
        crops.update(query_rgb=np.full((280, 280, 3), 255, np.uint8), template_rgb=np.zeros((280, 280, 3), np.uint8))
        fake = SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext, float32="float32")
        result = extract_one(SimpleNamespace(backbone=backbone, decoder=decoder, pose_head=head),
                             crops, device="cpu", torch_module=fake)
        self.assertEqual(calls[0], ("backbone", (2, 3, 280, 280), 1.))
        self.assertEqual(calls[1][1], {"features_query", "features_reference", "crop_size"})
        self.assertEqual(set(result), data.FEATURE_KEYS)
        # Hook identity is an interface oracle: hook 11 must retain mock value11.
        self.assertTrue(np.all(result["tokens_11"] == 11))
        crops["reference_pose"] = np.eye(4)
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            extract_one(None, crops, device="cpu", torch_module=fake)


class LossTensor(Tensor):
    def float(self): return LossTensor(self.value.astype(np.float32))
    def __add__(self, other): return LossTensor(self.value + (other.value if isinstance(other, Tensor) else other))
    def __mul__(self, other): return LossTensor(self.value * (other.value if isinstance(other, Tensor) else other))
    __rmul__ = __mul__
    def __truediv__(self, other): return LossTensor(self.value / (other.value if isinstance(other, Tensor) else other))
    def mean(self, dim=None): return LossTensor(self.value.mean(axis=dim))
    def sum(self, dim=None): return LossTensor(self.value.sum(axis=dim))
    def flatten(self, dim): return LossTensor(self.value.reshape(self.value.shape[:dim] + (-1,)))
    def clamp(self, low, high): return LossTensor(self.value.clip(low, high))


class LossTests(unittest.TestCase):
    def test_actual_loss_wiring_uses_per_frame_known_and_visible_masks(self):
        from bench.quality_adaptation_train import adaptation_loss
        def smooth(a, b, *, reduction, beta):
            self.assertEqual((reduction, beta), ("none", 1.))
            d = np.abs(a.value - b.value)
            return LossTensor(np.where(d < beta, .5 * d**2 / beta, d - .5 * beta))
        def bce(a, b, *, reduction):
            return LossTensor(-b.value * np.log(a.value) - (1-b.value) * np.log(1-a.value))
        fake = SimpleNamespace(isfinite=lambda a: np.isfinite(a.value),
                               nn=SimpleNamespace(functional=SimpleNamespace(smooth_l1_loss=smooth, binary_cross_entropy=bce)))
        pred = np.zeros((2, 2, 2, 2), np.float32)
        pred[0, 0, 0, 0] = 2
        visible = np.zeros((2, 2, 2), bool)
        visible[:, 0, 0] = True
        labels = {"flow": LossTensor(np.zeros_like(pred)), "visible": LossTensor(visible), "known": LossTensor(visible)}
        loss = adaptation_loss(LossTensor(pred), LossTensor(np.full((2, 2, 2), .5)), labels, TrainConfig(), fake)
        # Analytic SmoothL1(2)=1.5; two XY components and two frames give1.5/4.
        # A .5 visibility probability contributes -log(.5) on each known pixel.
        self.assertAlmostEqual(float(loss.value), .375 + np.log(2), places=7)


if __name__ == "__main__":
    unittest.main()
