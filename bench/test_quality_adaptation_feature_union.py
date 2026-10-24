"""Synthetic provenance and storage tests for frozen-feature cache union."""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import quality_adaptation_data as data_module
from . import quality_adaptation_feature_union as union
from . import quality_adaptation_flow_terminal as flow_terminal
from . import quality_adaptation_train as train_module
from .quality_adaptation_data import (CONVENTION, SCHEMA, canonical_hash,
                                      load_dataset, sha256)
from .quality_adaptation_runtime import LOCK


def _write_json(path: Path, value) -> bytes:
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _pin(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _payload(root: Path, name: str, content: bytes) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    raw = {"path": name, "sha256": hashlib.sha256(content).hexdigest(),
           "bytes": len(content)}
    return raw


def _available_row(root: Path, child: str, split: str, ordinal: int,
                   object_id: str, *, create_payloads=True) -> dict:
    row_id = f"{child}-{split}-{ordinal:03d}"
    rgb_hash = hashlib.sha256(("rgb:" + row_id).encode()).hexdigest()
    prefix = f"source_payloads/{row_id}"
    features = _payload(root, f"{prefix}/features.npz", f"features:{row_id}".encode()) if create_payloads else {
        "path": f"{prefix}/features.npz", "sha256": hashlib.sha256(("features:" + row_id).encode()).hexdigest(),
        "bytes": len(f"features:{row_id}".encode())}
    labels = _payload(root, f"{prefix}/labels.npz", f"labels:{row_id}".encode()) if create_payloads else {
        "path": f"{prefix}/labels.npz", "sha256": hashlib.sha256(("labels:" + row_id).encode()).hexdigest(),
        "bytes": len(f"labels:{row_id}".encode())}
    return {"id": row_id, "object_id": object_id,
            "sequence_id": f"sequence:{row_id}", "capture_group": f"capture:{row_id}",
            "query_rgb_sha256": rgb_hash, "split": split, "status": "available",
            "features": features, "labels": labels}


def _unavailable_row(child: str, split: str, ordinal: int, object_id: str) -> dict:
    row_id = f"{child}-{split}-failed-{ordinal:03d}"
    return {"id": row_id, "object_id": object_id,
            "sequence_id": f"sequence:{row_id}", "capture_group": f"capture:{row_id}",
            "query_rgb_sha256": hashlib.sha256(("rgb:" + row_id).encode()).hexdigest(),
            "split": split, "status": "unavailable", "failure_reason": "synthetic fixture failure"}


def _make_child(root: Path, name: str) -> tuple[Path, Path, dict]:
    metadata_root = root / name / "metadata"
    payload_root = root / name / "payloads"
    rows = []
    if name == "current":
        train_objects = ["13", "17", "18", "20"]
        for i in range(95):
            rows.append(_available_row(payload_root, name, "train", i,
                                       train_objects[i % len(train_objects)]))
        rows.append(_unavailable_row(name, "train", 95, "13"))
        for i in range(20):
            rows.append(_available_row(payload_root, name, "val", i, "4"))
        rows.extend(_unavailable_row(name, "val", i, "4") for i in range(20, 24))
        rows.extend(_unavailable_row(name, "test", i, "7") for i in range(24))
        crop = "a" * 64
    else:
        for i in range(72):
            rows.append(_available_row(payload_root, name, "train", i,
                                       ["1", "6", "10"][i % 3]))
        # These rows stay in original metadata only. Their payload paths are
        # intentionally absent so an accidental validation/test read fails.
        rows.append(_available_row(payload_root, name, "val", 0, "old-val",
                                   create_payloads=False))
        rows.append(_available_row(payload_root, name, "test", 0, "old-test",
                                   create_payloads=False))
        crop = "b" * 64
    recipe = {
        "schema": "frozen_gotrack_decoder_recipe_v1",
        "crop_manifest_sha256": crop,
        "annotation_inputs": [],
        "hooks": [2, 5, 8, 11],
        "producer_sha256": "1" * 64,
        "source_files_sha256": "2" * 64,
        "source_lock_sha256": "3" * 64,
        "upstream_revision": "fixture-pinned-revision",
        "normalization": "upstream_DinoFeatureExtractor_RGB_0_1",
    }
    recipe_raw = _write_json(metadata_root / "recipe.json", recipe)
    index = {"recipe_sha256": canonical_hash(recipe), "rows": []}
    index_raw = _write_json(metadata_root / "index.json", index)
    document = {"schema": SCHEMA, "conventions": CONVENTION,
                "provenance": {"feature_recipe_sha256": canonical_hash(recipe),
                               "source_lock_sha256": sha256(LOCK),
                               "annotation_inputs": [],
                               "label_origin": "synthetic_renderer_training_only"},
                "rows": rows}
    manifest_raw = _write_json(metadata_root / "manifest.json", document)
    files = {"manifest": _pin(metadata_root / "manifest.json"),
             "recipe": _pin(metadata_root / "recipe.json"),
             "index": _pin(metadata_root / "index.json")}
    if name == "current":
        (metadata_root / "final-request.json").write_bytes(b'{"audit_id":"current-final"}\n')
        files["final_request"] = {"path": "final-request.json",
                                  **_pin(metadata_root / "final-request.json")}
    spec_child = {
        "name": name,
        "metadata_root": str(metadata_root),
        "payload_root": str(payload_root),
        "files": files,
        "archive": {"sha256": ("4" if name == "current" else "5") * 64,
                    "bytes": 1000, "max_bytes": 2000},
        "transport_identity": {"schema": "existing_private_backup_transport_v1",
                               "identity_sha256": ("6" if name == "current" else "7") * 64,
                               "artifact_id": f"fixture-backup-{name}"},
    }
    return metadata_root, payload_root, spec_child


class UnionFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.children = [_make_child(self.root, name)[2] for name in ("current", "old72")]
        self.spec_path = self.root / "operator.json"
        self._write_spec()
        self.destination = self.root / "union"
        self.validator_patch = patch.object(train_module, "validate_feature_receipt")
        self.validator = self.validator_patch.start()
        self.addCleanup(self.validator_patch.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_spec(self):
        _write_json(self.spec_path, {"schema": union.OPERATOR_SCHEMA,
                                     "children": self.children})

    def _assemble(self):
        result = union.assemble_union(self.spec_path, self.destination)
        manifest = self.destination / "manifest.json"
        return result, load_dataset(manifest, inspect_arrays=False)


class FeatureUnionTests(UnionFixture):
    def test_crop_manifest_is_the_only_permitted_child_recipe_difference(self):
        current = {"normalization": "rgb01", "crop_manifest_sha256": "a" * 64}
        old = {"normalization": "rgb01", "crop_manifest_sha256": "b" * 64}
        self.assertEqual(union._same_except_crop_manifest([current, old]), {"normalization": "rgb01"})
        old["normalization"] = "other"
        with self.assertRaisesRegex(ValueError, "only in crop_manifest_sha256"):
            union._same_except_crop_manifest([current, old])

    def test_assembly_preserves_original_metadata_and_generic_loader_counts(self):
        source_manifest = Path(self.children[0]["metadata_root"]) / "manifest.json"
        source_recipe = Path(self.children[1]["metadata_root"]) / "recipe.json"
        result, ds = self._assemble()
        self.assertEqual((len(ds.rows("train", available=False)), len(ds.rows("train"))), (168, 167))
        self.assertEqual((len(ds.rows("val", available=False)), len(ds.rows("val"))), (24, 20))
        self.assertEqual((len(ds.rows("test", available=False)), len(ds.rows("test"))), (24, 0))
        self.assertEqual(result["children"], ["current", "old72"])
        self.assertEqual((self.destination / "lineage/current/manifest.json").read_bytes(),
                         source_manifest.read_bytes())
        self.assertEqual((self.destination / "lineage/old72/recipe.json").read_bytes(),
                         source_recipe.read_bytes())
        self.assertEqual((self.destination / "lineage/current/final-request.json").read_bytes(),
                         b'{"audit_id":"current-final"}\n')
        self.assertFalse((self.destination / "payloads/old72" /
                          hashlib.sha256(b"old72-val-000").hexdigest() / "features.npz").exists())
        self.assertEqual(union.validate_union_receipt(ds)["test_payload_reads"], 0)
        self.assertEqual(self.validator.call_count, 8)

    def test_logical_payload_union_above_one_gib_is_within_two_gib_contract(self):
        loaded = [union._load_child(child, child["name"], self.spec_path.parent)
                  for child in self.children]
        for child in loaded:
            for row in child["dataset"].document["rows"]:
                if union._selected(child["name"], row) and row["status"] == "available":
                    row["features"]["bytes"] = 3 * 1024**2
                    row["labels"]["bytes"] = 3 * 1024**2
        _, _, _, payload_bytes, _ = union._assemble_documents(loaded, None)
        self.assertGreater(payload_bytes, 1024**3)
        self.assertLess(payload_bytes, 2 * 1024**3)
        self.assertEqual(union.MAX_UNION_PAYLOAD_BYTES, 2 * 1024**3)

    def test_omitted_index_rows_changed_rgb_and_altered_lineage_are_rejected(self):
        _, ds = self._assemble()
        index_path = self.destination / "index.json"
        index = json.loads(index_path.read_text())
        index["children"][0]["rows"].pop()
        _write_json(index_path, index)
        with self.assertRaisesRegex(ValueError, "original child rows"):
            union.validate_union_receipt(ds)

        # Restore and alter a valid-shaped RGB digest in the original-row bind.
        _, ds = self._assemble_again()
        index = json.loads(index_path.read_text())
        index["children"][0]["rows"][0]["original_row"]["query_rgb_sha256"] = "f" * 64
        _write_json(index_path, index)
        with self.assertRaisesRegex(ValueError, "changed original row"):
            union.validate_union_receipt(ds)

        _, ds = self._assemble_again()
        lineage = self.destination / "lineage/current/manifest.json"
        changed = json.loads(lineage.read_text())
        changed["provenance"]["label_origin"] = "independent_annotations"
        _write_json(lineage, changed)
        with self.assertRaisesRegex(ValueError, "metadata bytes changed"):
            union.validate_union_receipt(ds)

    def _assemble_again(self):
        import shutil
        shutil.rmtree(self.destination)
        return self._assemble()

    def test_generic_loader_rejects_cross_split_object_leakage(self):
        _, ds = self._assemble()
        rows = json.loads(json.dumps(ds.document["rows"]))
        rows[0]["object_id"] = "4"
        document = dict(ds.document, rows=rows)
        bad = self.destination / "leak.json"
        _write_json(bad, document)
        with self.assertRaisesRegex(ValueError, "split leakage"):
            load_dataset(bad, inspect_arrays=False)

    def test_destination_collision_and_one_gib_reserve_fail_before_publication(self):
        existing = self.root / "already-there"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            union.assemble_union(self.spec_path, existing)
        with patch.object(union.shutil, "disk_usage", return_value=SimpleNamespace(free=union.MIN_FREE_BYTES)):
            with self.assertRaisesRegex(ValueError, "1 GiB reserve"):
                union.assemble_union(self.spec_path, self.destination)
        self.assertFalse(self.destination.exists())

    def test_selected_payload_hash_failure_retains_unpublished_partial(self):
        source = Path(self.children[1]["payload_root"]) / "source_payloads/old72-train-000/features.npz"
        source.write_bytes(b"changed after metadata pin")
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            union.assemble_union(self.spec_path, self.destination)
        self.assertTrue(self.destination.is_dir())
        self.assertFalse((self.destination / "manifest.json").exists())
        self.assertTrue((self.destination / "lineage/old72/manifest.json").exists())

    def test_cross_child_row_id_collision_is_rejected_before_destination_creation(self):
        path = Path(self.children[0]["metadata_root"]) / "manifest.json"
        document = json.loads(path.read_text())
        document["rows"][0]["id"] = "old72-train-000"
        _write_json(path, document)
        self.children[0]["files"]["manifest"] = {**_pin(path)}
        self._write_spec()
        with self.assertRaisesRegex(ValueError, "row id collision"):
            union.assemble_union(self.spec_path, self.destination)
        self.assertFalse(self.destination.exists())

    def test_cross_filesystem_copy_path_is_bounded_and_published_last(self):
        real_link = os.link

        def exdev(*_args, **_kwargs):
            raise OSError(errno.EXDEV, "fixture cross-device")

        with patch.object(union.os, "link", side_effect=exdev):
            result, ds = self._assemble()
        self.assertEqual(result["selected_rows"], 216)
        self.assertEqual(len(ds.rows("train")), 167)
        source = Path(self.children[0]["payload_root"]) / "source_payloads/current-train-000/features.npz"
        translated = self.destination / "payloads/current" / hashlib.sha256(b"current-train-000").hexdigest() / "features.npz"
        self.assertEqual(translated.read_bytes(), source.read_bytes())
        self.assertNotEqual(translated.stat().st_ino, source.stat().st_ino)

        self._assemble_again()
        with patch.object(union.os, "replace", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                union.assemble_union(self.spec_path, self.destination)
        self.assertFalse((self.destination / "manifest.json").exists())
        self.assertTrue((self.destination / ".manifest.partial.json").exists())

    def test_restricted_receipt_validation_hashes_old72_train_only(self):
        _, ds = self._assemble()
        receipt = union.validate_union_receipt(ds, payload_scope={("old72", "train")})
        self.assertEqual(receipt["payload_scope"], "old72_train_only")
        self.assertEqual(receipt["payload_files_verified"], 144)
        # There are deliberately missing old72 VAL/TEST files; all validation
        # and test metadata stayed present while their payloads were untouched.
        self.assertTrue((self.destination / "lineage/old72/manifest.json").is_file())

    def test_invalid_symlink_metadata_root_is_rejected_when_available(self):
        link = self.root / "current-link"
        try:
            link.symlink_to(Path(self.children[0]["metadata_root"]), target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks are unavailable in this Windows sandbox")
        self.children[0]["metadata_root"] = str(link)
        self._write_spec()
        with self.assertRaisesRegex(ValueError, "symlinks and junctions"):
            union.assemble_union(self.spec_path, self.destination)


class TransferGuardTests(unittest.TestCase):
    def test_transfer_cloud_guard_precedes_filesystem_or_model_access(self):
        with patch("bench.quality_adaptation_runtime.cloud_guard", side_effect=RuntimeError("cloud required")), \
             patch.object(union, "load_dataset") as load:
            with self.assertRaisesRegex(RuntimeError, "cloud required"):
                union.transfer_diagnostic(Path("missing"), Path("export"), "0" * 64,
                                          Path("result.json"), cloud_execution=False)
        load.assert_not_called()

    def test_transfer_pairs_old72_train_rows_and_strict_loads_current_export(self):
        class Output:
            def __getitem__(self, _index):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.zeros((2, 2, 2), dtype=np.float32)

        class Head:
            def __init__(self):
                self.strict = None
                self.evaluated = False

            def load_state_dict(self, _state, *, strict):
                self.strict = strict

            def eval(self):
                self.evaluated = True

        rows = [{"id": f"old-{index:03d}", "object_id": ["1", "6", "10"][index % 3],
                 "split": "train", "status": "available"} for index in range(72)]

        class DatasetDouble:
            root = Path("unused-union-root")
            identity = "d" * 64

            def __init__(self):
                self.opened = []

            def arrays(self, row):
                self.opened.append(row["id"])
                features = {f"tokens_{hook}": np.zeros((1,), dtype=np.float32)
                            for hook in flow_terminal.HOOKS}
                features.update(baseline_flow=np.zeros((2, 2, 2), dtype=np.float32),
                                baseline_confidence=np.zeros((2, 2), dtype=np.float32))
                labels = {"flow": np.zeros((2, 2, 2), dtype=np.float32),
                          "visible": np.ones((2, 2), dtype=np.bool_),
                          "known": np.ones((2, 2), dtype=np.bool_)}
                return features, labels

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_path = root / "head.pt"
            export_path.write_bytes(b"synthetic strict-head export")
            expected_sha = sha256(export_path)
            output_path = root / "diagnostic.json"
            dataset = DatasetDouble()
            current_ds = SimpleNamespace(identity="c" * 64)
            artifact = {
                "schema": "correspondence_head_export_v1", "head": {"fixture": "strict"},
                "source_lock_sha256": "1" * 64, "base_checkpoint_sha256": "2" * 64,
                "dataset_sha256": current_ds.identity, "config_sha256": "3" * 64,
                "validation": {}, "quality_claim": False,
                "deployment": "explicit_cloud_research_only_pose_head_replacement",
            }
            torch = SimpleNamespace(load=lambda *_args, **_kwargs: artifact,
                                    inference_mode=nullcontext,
                                    from_numpy=lambda value: value)
            head = Head()
            with patch("bench.quality_adaptation_runtime.cloud_guard"), \
                 patch.object(union, "load_dataset", side_effect=[dataset, current_ds]), \
                 patch.object(union, "validate_union_receipt", return_value={
                     "recipe_sha256": "4" * 64}), \
                 patch.object(union, "read_json", return_value={"children": [{}, {}]}), \
                 patch.object(union, "_selected_old72_rows", return_value=rows), \
                 patch.dict("sys.modules", {"torch": torch}), \
                 patch.object(flow_terminal, "load_head", return_value=head) as load_head, \
                 patch.object(flow_terminal, "_model_outputs", return_value=(Output(), Output())), \
                 patch.object(union, "flow_metrics", return_value={"epe_px": 1.0}) as flow_metric, \
                 patch.object(union, "visibility_metrics", return_value={"accuracy": 1.0}) as vis_metric:
                report = union.transfer_diagnostic(
                    root / "union/manifest.json", export_path, expected_sha,
                    output_path, cloud_execution=True)

            self.assertTrue(output_path.is_file())
            self.assertEqual(report["inventory"], {
                "selected": 72, "available": 72, "evaluated": 72,
                "failed": 0, "unavailable": 0,
            }, report["failures"][:2])
            self.assertEqual(len(dataset.opened), 72)
            self.assertEqual(len(report["baseline"]), 72)
            self.assertEqual(len(report["adapted"]), 72)
            self.assertEqual(head.strict, True)
            self.assertTrue(head.evaluated)
            load_head.assert_called_once_with(initialize=False, device="cpu")
            self.assertEqual(flow_metric.call_count, 144)
            self.assertEqual(vis_metric.call_count, 144)
            self.assertFalse(report["validation_arrays_opened"])
            self.assertEqual(report["test_payload_reads"], 0)
            self.assertFalse(report["training_performed"])


if __name__ == "__main__":
    unittest.main()
