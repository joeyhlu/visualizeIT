"""Manufactured boundary tests for the additive expanded-capture controller.

These tests use only metadata, NumPy arrays and short local child processes; no
neural model, Torch import, network access or external data is required.
"""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from . import quality_adaptation_expanded as expanded
from .quality_adaptation_data import SIZE, sha256

# The managed Windows TEMP mount is not writable in this sandbox; the repository
# cache parent is writable and every TemporaryDirectory below cleans its child.
temp_root = Path(__file__).resolve().parents[1] / ".cache" / "cloud-prepare"
temp_root.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temp_root)


def _pin(path: Path, digest: str = "a" * 64) -> dict:
    return {"path": str(path), "sha256": digest, "bytes": 1}


def _plan(root: Path) -> dict:
    rows = []
    for split, object_id, sequence, pitch in expanded.EXPECTED_ROWS:
        for frame_id in range(8):
            prefix = f"{sequence}-frame{frame_id}"
            rows.append({
                "id": f"{sequence}:frame{frame_id}:iteration0", "object_id": object_id,
                "split": split, "sequence_id": sequence, "capture_group": f"group-{sequence}",
                "frame_id": frame_id, "yaw0_deg": 0, "pitch_deg": pitch,
                "timestamp_s": frame_id / 10.0,
                "input": _pin(root / f"{prefix}-input.json"),
                "mask_index": _pin(root / f"{prefix}-mask-index.json"),
                "mask": _pin(root / f"{prefix}-mask.png", hashlib.sha256(prefix.encode()).hexdigest()),
                "bank": _pin(root / f"{prefix}-bank.pt"),
                "generation": _pin(root / f"{prefix}-generation.json"),
            })
    return {
        "schema": expanded.PLAN_SCHEMA, "rows": rows,
        "sealed_test": {"id": "test7-unmaterialized", "object_id": "7",
            "sequence_id": "sealed-test7-unmaterialized", "capture_group": "test-only",
            "split": "test", "status": "unavailable", "query_rgb_sha256": None,
            "failure_reason": "held_out_unmaterialized", "unavailable_provenance": "no payload"},
        "max_seconds": 1800, "depth_tolerance_m": .002,
        "native_raster_convention": expanded.RASTER_CONVENTION,
    }


class ExpandedPlanTests(unittest.TestCase):
    def test_capture_plan_bytes_survive_downstream_frozen_plan_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input-plan.json"
            capture = root / "capture"
            capture.mkdir()
            plan = _plan(root)
            # Deliberately retain a different indentation/key order from
            # write_json: semantic equality alone cannot satisfy a byte pin.
            raw = (json.dumps(plan, indent=4) + "\n").encode()
            source.write_bytes(raw)
            with mock.patch.object(expanded, "FROZEN_PLAN_SHA256", hashlib.sha256(raw).hexdigest()):
                expanded._persist_frozen_plan(source, capture)
                self.assertEqual((capture / "plan.json").read_bytes(), raw)
                loaded = expanded.read_frozen_plan(capture / "plan.json", verify_files=False)
                self.assertEqual(loaded, plan)
                source.write_bytes(raw + b" ")
                with self.assertRaisesRegex(ValueError, "changed before persistence"):
                    expanded._persist_frozen_plan(source, capture)
                self.assertEqual((capture / "plan.json").read_bytes(), raw)

    def test_exact_24_frame_inventory_and_test7_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = _plan(Path(directory))
            self.assertIs(expanded.validate_plan(plan, verify_files=False), plan)
            self.assertEqual(len(plan["rows"]), 24)
            self.assertEqual([r["frame_id"] for r in plan["rows"][:8]], list(range(8)))
            ids = [r["id"] for r in plan["rows"]]
            self.assertEqual(len(set(ids)), 24, "the frozen plan defines 24 unique frame IDs")
            joined = expanded._index_by_id([{"id": value} for value in ids], ids, "manufactured capture")
            self.assertEqual(set(joined), set(ids))
            self.assertEqual(plan["sealed_test"]["status"], "unavailable")
            self.assertIsNone(plan["sealed_test"]["query_rgb_sha256"])
            self.assertNotIn("features", plan["sealed_test"])

    def test_plan_rejects_duplicate_extra_split_and_test_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = _plan(Path(directory))
            cases = []
            duplicate = json.loads(json.dumps(baseline))
            duplicate["rows"][7]["id"] = duplicate["rows"][6]["id"]
            cases.append(duplicate)
            extra = json.loads(json.dumps(baseline))
            extra["rows"].append(dict(extra["rows"][-1]))
            cases.append(extra)
            wrong_split = json.loads(json.dumps(baseline))
            wrong_split["rows"][16]["split"] = "test"
            cases.append(wrong_split)
            exposed_test = json.loads(json.dumps(baseline))
            exposed_test["sealed_test"]["query_rgb_sha256"] = "b" * 64
            cases.append(exposed_test)
            ids = [row["id"] for row in baseline["rows"]]
            for candidate in cases:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ValueError):
                        expanded.validate_plan(candidate, verify_files=False)
            with self.assertRaises(ValueError):
                expanded._index_by_id([{"id": value} for value in ids[:-1]], ids, "missing capture")

    def test_frame7_uses_actual_physical_timestamp_and_bad_clock_fails(self):
        from .quality_time import FrameKey, PHYSICAL
        row = {"frame_id": 7}
        manifest = {"_timing": {"clock_mode": PHYSICAL,
            "scored_keys": [FrameKey(7, .7, PHYSICAL)]}}
        # The expected value comes from the frozen f/10 physical timeline and FrameKey policy.
        self.assertEqual(expanded.manifest_frame_timestamp(manifest, row), .7)
        bad_clock = {"_timing": {"clock_mode": PHYSICAL,
            "scored_keys": [FrameKey(7, .71, PHYSICAL)]}}
        with self.assertRaises(ValueError):
            expanded.manifest_frame_timestamp(bad_clock, row)
        with tempfile.TemporaryDirectory() as directory:
            plan = _plan(Path(directory))
            plan["rows"][7]["timestamp_s"] = .71
            with self.assertRaises(ValueError):
                expanded.validate_plan(plan, verify_files=False)

    def test_absolute_posix_cloud_pins_validate_as_metadata_only_on_windows(self):
        plan = _plan(Path("C:/not-opened"))
        for row in plan["rows"]:
            for pin_name in ("input", "mask_index", "mask", "bank", "generation"):
                row[pin_name]["path"] = f"/teamspace/cache/{row['id'].replace(':', '-')}-{pin_name}"
        self.assertTrue(all(row["input"]["path"].startswith("/teamspace/") for row in plan["rows"]))
        with mock.patch.object(Path, "is_file", side_effect=AssertionError(
                "metadata-only validation must not probe cloud plan payload paths")):
            self.assertIs(expanded.validate_plan(plan, verify_files=False), plan)

    def test_mask_binding_requires_selected_frame_path_and_explicit_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _plan(root)
            row = plan["rows"][7]
            mask_path = Path(row["mask"]["path"])
            expected = {"frames": [{"frameId": 7, "path": mask_path.name,
                "mask_state": "available", "sha256": row["mask"]["sha256"]}]}
            expanded.validate_mask_binding(row, expected, root, mask_path)
            for bad in (
                {"frames": [{**expected["frames"][0], "frameId": 6}]},
                {"frames": [{**expected["frames"][0], "path": "other.png"}]},
                {"frames": [{**expected["frames"][0], "sha256": "c" * 64}]},
                {"frames": [{k: v for k, v in expected["frames"][0].items() if k != "sha256"}]},
            ):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        expanded.validate_mask_binding(row, bad, root, mask_path)


class ExpandedCliAndGeometryTests(unittest.TestCase):
    def test_supervisor_commands_parse_with_the_real_cli_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = expanded._workflow_commands(root / "plan.json", root / "labels.json",
                root / "config.json", root / "output")
            self.assertEqual(len(commands), 9)
            parser = expanded._build_parser()
            for name, command in commands:
                with self.subTest(stage=name):
                    parsed = parser.parse_args(command[3:])
                    self.assertEqual(parsed.command, {
                        "train_step8": "train", "inspect_step8": "interruption"}.get(name, name))
            self.assertNotIn("torch", sys.modules)
        self.assertTrue(callable(expanded.extract_stage),
            "module CLI entry point must be reached only after extract_stage is defined")

    def test_retained_capture_accepts_proper_4x4_camera_transform(self):
        transform = np.eye(4, dtype=np.float64)
        np.testing.assert_array_equal(expanded.validate_retained_geometry(transform), transform)
        for invalid in (
            np.eye(3),
            np.diag([-1., 1., 1., 1.]),
            np.array([[1., 0., 0., 0.1], [0., 1., 0., 0.],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]]),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    expanded.validate_retained_geometry(invalid)


class ExpandedIntegrityTests(unittest.TestCase):
    def test_frozen_capture_contract_pins_controller_config_and_all_historical_sources(self):
        document = expanded.read_json(expanded.CONTRACT_PATH)
        source_map = {pin["path"]: pin["sha256"] for pin in document["historical_sources"]}
        self.assertEqual(len(source_map), expanded.HISTORICAL_SOURCE_COUNT)
        self.assertEqual(expanded.canonical_hash(source_map), expanded.HISTORICAL_SOURCE_MAP_SHA256)
        self.assertEqual(document["historical_source_manifest_sha256"],
                         expanded.HISTORICAL_SOURCE_MANIFEST_SHA256)
        self.assertEqual(source_map["bench/runtime/adaptation-sources.json"],
                         expanded.HISTORICAL_LOCK_SHA256)
        self.assertEqual(document["controller"]["sha256"], sha256(
            Path(__file__).resolve().parents[1] / document["controller"]["path"]))
        self.assertEqual(document["config"]["sha256"], sha256(
            Path(__file__).resolve().parents[1] / document["config"]["path"]))

    def test_generic_extraction_exception_persists_all_capture_and_feature_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            plan = _plan(root)
            capture_rows = [{"id": row["id"], "status": "available"} for row in plan["rows"]]
            capture_rows[-1] = {"id": capture_rows[-1]["id"], "status": "unavailable",
                                "failure_reason": "capture failed on this frame"}
            (root / "capture-index.json").write_text(json.dumps({"rows": capture_rows,
                "test_payload_reads": 0}), encoding="utf-8")
            (root / "contract.json").write_text("{}", encoding="utf-8")
            features = root / "features"
            fake_prepare = types.SimpleNamespace()
            class FakeConfig:
                def __init__(self, **kwargs):
                    self.max_output_bytes = kwargs["max_output_bytes"]
            def failing_extract(_manifest, output, _config, **_kwargs):
                output.mkdir()
                partial = [{"id": plan["rows"][0]["id"], "status": "available",
                    "crops": {"path": "crop.npz", "sha256": "a" * 64, "bytes": 1},
                    "query_rgb_sha256": "b" * 64,
                    "features": {"path": "feature.npz", "sha256": "c" * 64, "bytes": 1}}]
                (output / "index.json").write_text(json.dumps({"status": "failed", "rows": partial}),
                    encoding="utf-8")
                raise RuntimeError("manufactured extractor interruption")
            fake_prepare.ExtractionConfig = FakeConfig
            fake_prepare.extract = failing_extract
            with mock.patch.object(expanded, "cloud_guard"), \
                 mock.patch.object(expanded, "verify_capture_contract"), \
                 mock.patch.object(expanded, "verify_sources", return_value={"source": "pinned"}), \
                 mock.patch.object(expanded, "read_frozen_plan", return_value=plan), \
                 mock.patch.object(expanded, "ROOT", source_root), \
                 mock.patch.object(expanded, "CONTRACT_PATH", root / "contract.json"), \
                 mock.patch.dict(sys.modules, {"bench.quality_adaptation_prepare": fake_prepare}):
                with self.assertRaisesRegex(RuntimeError, "manufactured extractor interruption"):
                    expanded.extract_stage(root, features, cloud_execution=True)
            index = json.loads((features / "index.json").read_text(encoding="utf-8"))
            receipt = json.loads((features / "failure-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index["rows"]), 24)
            self.assertEqual(len({row["id"] for row in index["rows"]}), 24)
            self.assertEqual(sum(row["status"] == "available" for row in index["rows"]), 1)
            self.assertEqual(index["rows"][-1]["failure_reason"], "capture failed on this frame")
            self.assertEqual(receipt["requested"], 24)
            self.assertEqual(receipt["available_features"], 1)
            self.assertNotIn("torch", sys.modules)

    def test_label_plan_is_not_read_until_outer_seal_passes(self):
        events = []
        with mock.patch.object(expanded, "verify_prelabel_seal",
                side_effect=lambda *_: events.append("seal") or {"ok": True}), \
             mock.patch.object(expanded, "sha256", side_effect=lambda _: events.append("hash-label-plan") or expanded.FROZEN_LABEL_PLAN_SHA256), \
             mock.patch.object(expanded, "read_json", side_effect=lambda _: events.append("read-label-plan") or {"label": "plan"}):
            plan, seal = expanded.read_label_plan_after_seal(Path("capture"), Path("features"), {}, Path("labels.json"))
        self.assertEqual(events, ["seal", "hash-label-plan", "read-label-plan"])
        self.assertEqual(plan, {"label": "plan"})
        self.assertEqual(seal, {"ok": True})

    def test_missing_actual_flow_or_confidence_parity_is_rejected(self):
        zero_flow = np.zeros((SIZE, SIZE, 2), dtype=np.float32)
        zero_confidence = np.zeros((SIZE, SIZE), dtype=np.float32)
        feature = {"baseline_flow": zero_flow.transpose(2, 0, 1),
                   "baseline_confidence": zero_confidence}
        with self.assertRaisesRegex(ValueError, "missing actual retained"):
            expanded._check_parity({"actual_flow": zero_flow}, feature,
                                   np.ones((SIZE, SIZE), dtype=bool))

    def test_crop_manifest_and_features_bind_exact_retained_crop_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _plan(root)
            capture_rows, crop_rows, descriptors = {}, [], []
            for ordinal, row in enumerate(plan["rows"]):
                name = f"crop-{ordinal:02d}.npz"
                payload = f"manufactured-crop-{ordinal}".encode()
                (root / name).write_bytes(payload)
                crop = {"path": name, "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload)}
                descriptors.append(crop)
                capture_rows[row["id"]] = {"id": row["id"], "status": "available", "crops": crop}
                crop_rows.append({"id": row["id"], "crops": crop})
            crop_manifest = {"schema": "correspondence_input_crops_v1",
                "conventions": expanded.CONVENTION, "rows": crop_rows}
            recipe = {"crop_manifest_sha256": expanded.canonical_hash(crop_manifest)}
            (root / "crops.json").write_text(json.dumps(crop_manifest), encoding="utf-8")
            self.assertEqual(expanded._validate_capture_crop_manifest(root, plan, capture_rows, recipe),
                             crop_manifest)
            expanded._validate_extracted_crop_binding(
                capture_rows[plan["rows"][0]["id"]],
                {"crops": descriptors[0]})

            swapped_manifest = json.loads(json.dumps(crop_manifest))
            swapped_manifest["rows"][0]["crops"] = descriptors[1]
            (root / "crops.json").write_text(json.dumps(swapped_manifest), encoding="utf-8")
            swapped_recipe = {"crop_manifest_sha256": expanded.canonical_hash(swapped_manifest)}
            with self.assertRaisesRegex(ValueError, "exactly join"):
                expanded._validate_capture_crop_manifest(root, plan, capture_rows, swapped_recipe)
            with self.assertRaisesRegex(ValueError, "exact retained capture crop"):
                expanded._validate_extracted_crop_binding(
                    capture_rows[plan["rows"][0]["id"]], {"crops": descriptors[1]})

            (root / "crops.json").write_text(json.dumps(crop_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                expanded._validate_capture_crop_manifest(root, plan, capture_rows,
                    {"crop_manifest_sha256": "0" * 64})

    def test_contract_authenticates_controller_config_and_historical_lock_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "lock": root / "bench/runtime/adaptation-sources.json",
                "controller": root / "bench/quality_adaptation_expanded.py",
                "config": root / "bench/runtime/adaptation-expanded-cpu-v1.json",
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())
            lock_hash = sha256(paths["lock"])
            source_map = {"bench/runtime/adaptation-sources.json": lock_hash}
            source_map_hash = expanded.canonical_hash(source_map)
            doc = {"schema": expanded.CONTRACT_SCHEMA,
                "historical_source_lock_sha256": lock_hash,
                "historical_source_manifest_sha256": "f" * 64,
                "controller": {"path": "bench/quality_adaptation_expanded.py", "sha256": sha256(paths["controller"])},
                "config": {"path": "bench/runtime/adaptation-expanded-cpu-v1.json", "sha256": sha256(paths["config"])},
                "historical_sources": [{"path": "bench/runtime/adaptation-sources.json", "sha256": lock_hash}]}
            kwargs = {"root": root, "expected_lock_sha256": lock_hash,
                "expected_controller_path": doc["controller"]["path"],
                "expected_config_path": doc["config"]["path"],
                "expected_source_manifest_sha256": doc["historical_source_manifest_sha256"],
                "expected_source_map_sha256": source_map_hash, "expected_source_count": 1}
            expanded.verify_contract_document(doc, **kwargs)
            for key in ("controller", "config", "lock"):
                original = paths[key].read_bytes()
                paths[key].write_bytes(original + b"tampered")
                with self.subTest(source=key):
                    with self.assertRaises(ValueError):
                        expanded.verify_contract_document(doc, **kwargs)
                paths[key].write_bytes(original)

    def test_expanded_config_hash_must_match_capture_contract(self):
        config_digest = sha256(expanded.CONFIG_PATH)
        with mock.patch.object(expanded, "verify_capture_contract",
                return_value={"config_sha256": config_digest}):
            config = expanded._expanded_config(expanded.CONFIG_PATH)
        self.assertEqual(config.max_steps, 16)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "config.json"
            changed.write_bytes(expanded.CONFIG_PATH.read_bytes() + b" ")
            with mock.patch.object(expanded, "verify_capture_contract",
                    return_value={"config_sha256": config_digest}):
                with self.assertRaisesRegex(ValueError, "hash-bound"):
                    expanded._expanded_config(changed)

    def test_fatal_source_preflight_persists_full_failure_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _plan(root)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "capture"
            with mock.patch.object(expanded, "cloud_guard"), \
                 mock.patch.object(expanded, "verify_capture_contract"), \
                 mock.patch.object(expanded, "FROZEN_PLAN_SHA256", sha256(plan_path)), \
                 mock.patch.object(expanded, "validate_plan", side_effect=lambda doc, **_: doc), \
                 mock.patch.object(expanded, "verify_sources", side_effect=RuntimeError("manufactured source mismatch")):
                with self.assertRaisesRegex(RuntimeError, "full failure inventory persisted"):
                    expanded.capture(plan_path, output)
            receipt = json.loads((output / "capture-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["rows"]), 24)
            self.assertEqual(receipt["requested"], 24)
            self.assertTrue(all(row["status"] == "unavailable" for row in receipt["rows"]))
            self.assertTrue(all("manufactured source mismatch" in row["failure_reason"]
                                for row in receipt["rows"]))

    def test_retained_float_rgb_is_preserved_without_mutating_callback_arrays(self):
        from .quality_adaptation_prepare import crops_from_retained_capture
        query = np.full((1, 3, SIZE, SIZE), np.float32(.37123457), dtype=np.float32)
        template = np.full((1, 3, SIZE, SIZE), np.float32(.61876541), dtype=np.float32)
        query_before, template_before = query.copy(), template.copy()
        intrinsics = np.array([[200., 0., SIZE / 2], [0., 200., SIZE / 2], [0., 0., 1.]], dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = 1000.
        arrays = {"query_rgb_crop": query, "template_rgb": template,
            "rendered_depth_mm": np.full((SIZE, SIZE), 1000., dtype=np.float32),
            "rendered_mask": np.ones((SIZE, SIZE), dtype=np.uint8),
            "crop_k": intrinsics, "current_crop_camera_from_object_mm": pose}
        result = crops_from_retained_capture(arrays, preserve_float_rgb=True)
        # The existing retained-capture API promises float32 RGB copied without quantization.
        np.testing.assert_array_equal(result["query_rgb"], query_before[0].transpose(1, 2, 0),
            err_msg="query RGB must preserve exact captured float32 values")
        np.testing.assert_array_equal(result["template_rgb"], template_before[0].transpose(1, 2, 0),
            err_msg="template RGB must preserve exact captured float32 values")
        np.testing.assert_array_equal(query, query_before, err_msg="callback query bytes must remain unchanged")
        np.testing.assert_array_equal(template, template_before, err_msg="callback template bytes must remain unchanged")
        self.assertEqual(result["query_rgb"].dtype, np.float32)
        self.assertNotIn("torch", sys.modules, "manufactured capture tests must not import the neural runtime")


class ExpandedBudgetTests(unittest.TestCase):
    def test_aggregate_budget_counts_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temp_artifact = root / ".features.partial"
            temp_artifact.write_bytes(b"12345678901")
            from .quality_adaptation_runtime import tree_bytes
            self.assertEqual(tree_bytes(root), 11, "storage accounting includes in-flight temporary files")
            with mock.patch.object(expanded, "PRIVATE_BYTES", 10):
                with self.assertRaises(ValueError):
                    expanded._check_aggregate(1., tree_bytes(root))
            with self.assertRaises(TimeoutError):
                expanded._check_aggregate(expanded.AGGREGATE_SECONDS, 0)

    def test_deadline_stops_and_reaps_child_and_persists_failure_receipt(self):
        from .quality_offline import run_stage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            pid_file = root / "child.pid"
            marker = root / "after-timeout.marker"
            code = ("import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); "
                    "time.sleep(1.2); open(sys.argv[2],'w').write('survived')")
            command = [sys.executable, "-c", code, str(pid_file), str(marker)]
            class FakeProcess:
                def __init__(self, pid):
                    self.pid = pid
                def children(self, recursive=False):
                    return []
                def memory_info(self):
                    return types.SimpleNamespace(rss=0)
            class FakePsutil:
                class NoSuchProcess(Exception): pass
                class AccessDenied(Exception): pass
                Process = FakeProcess
                @staticmethod
                def wait_procs(processes, timeout):
                    return processes, []
            with mock.patch.dict(sys.modules, {"psutil": FakePsutil}):
                with self.assertRaisesRegex(RuntimeError, "runtime_limit"):
                    run_stage("manufactured_timeout", command, logs, timeout_s=.3,
                        storage_limit_bytes=1024 * 1024, storage_root=root,
                        output_limit_bytes=1024 * 1024)
            receipt = json.loads((logs / "manufactured_timeout.receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["failure_reason"], "runtime_limit")
            self.assertTrue(pid_file.is_file(), "child must start and publish its PID before deadline")
            pid = int(pid_file.read_text(encoding="utf-8"))
            # The supervisor's Popen.wait() must have reaped the process before
            # returning; a later marker write would prove it survived cleanup.
            self.assertGreater(pid, 0)
            time.sleep(1.0)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
