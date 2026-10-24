"""Model-free guard tests for the explicit expanded consumer lifecycle."""
from __future__ import annotations

import json
import copy
import hashlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
import numpy as np
from contextlib import redirect_stdout
import io
from unittest.mock import patch
from contextlib import nullcontext

from bench import quality_adaptation_portable_prepare as prepared
from scripts import kaggle_sharded_prepared_v1 as sharded
from bench.test_quality_adaptation_portable_prepare import known_role_fixture
from bench.test_quality_adaptation_portable_prepare import Fixture, ROOT, write
from bench import quality_adaptation_portable as raw
from bench.quality_adaptation_data import CONVENTION, HOOKS
from bench.quality_adaptation_prepare import project_supervision
from bench.quality_adaptation_manifest import assemble
from scripts import kaggle_prepared_v2 as original_prepared


def _prepared_child_fixture(fixture):
    """Build an original-format prepared child with analytical label doubles."""
    fixture.prepare_features()
    inventory = []
    for ordinal, source in enumerate(fixture.captures):
        if source["split"] == "dev":
            continue
        row = {"id": source["id"], "object_id": source["object_id"],
            "sequence_id": source["sequence_id"], "capture_group": source["capture_group"],
            "split": "val" if source["split"] == "validation" else source["split"],
            "status": "unavailable", "query_rgb_sha256": source.get("query_rgb_sha256"),
            "failure_reason": source.get("failure_reason", "fixture_label_failure")}
        if source["id"] in fixture.arrays:
            crops, _ = fixture.arrays[source["id"]]
            target = project_supervision(template_xyz_m=crops["template_xyz_m"],
                template_mask=crops["template_mask"], query_K=crops["query_K"],
                query_T_cam_from_object=np.eye(4),
                observed_depth_m=np.ones((280, 280), np.float32),
                observed_object_mask=np.ones((280, 280), bool), depth_tolerance_m=.002)
            label_path = fixture.features / f"labels-{ordinal}.npz"
            np.savez(label_path, **target)
            row.pop("failure_reason")
            row.update(status="available", labels={**prepared.pin(label_path),
                "path": label_path.name})
        if row["query_rgb_sha256"] is None:
            row["unavailable_provenance"] = "MODEL_FREE_TEST_FIXTURE; no crop materialized"
        inventory.append(row)
    write(fixture.features / "labels.json", {"schema": "correspondence_label_inventory_v1",
        "label_origin": "synthetic_renderer_training_only", "rows": inventory})
    assemble(fixture.features, fixture.features / "labels.json", fixture.features / "manifest.json")
    write(fixture.features / "label-receipt.json", {"schema": "kaggle_label_receipt_v2",
        "prelabel_seal": prepared.pin(fixture.execution / "prelabel-seal.json"),
        "inventory": prepared.pin(fixture.features / "labels.json"),
        "dataset": prepared.pin(fixture.features / "manifest.json"),
        "labels_entered_inference": False, "test_payload_reads": 0,
        "independent_quality_claim": False,
        "label_artifacts": [{"id": row["id"], **row["labels"]}
            for row in inventory if row["status"] == "available"]})
    runtime = write(fixture.base / "runtime.json", {"MODEL_FREE_TEST_FIXTURE": True})
    config = write(fixture.base / "config.json", {"device": "cuda", "epochs": 1,
        "max_steps": 24, "max_seconds": 300., "max_total_seconds": 600.,
        "max_checkpoint_bytes": 16 * prepared.MIB, "max_output_bytes": 96 * prepared.MIB})
    evidence = write(fixture.base / "execution-evidence.json", {"MODEL_FREE_TEST_FIXTURE": True})
    return original_prepared.request_document(
        preparation={"source_root": str(ROOT), "execution_root": str(fixture.execution),
            "features_root": str(fixture.features), "raw_root": str(fixture.root),
            "exposure_audit": fixture.audit_pin}, runtime=runtime["path"], config=config["path"],
        source_install=fixture.base / "source-install", python=Path(sys.executable),
        input_root=fixture.base / "input", working_root=fixture.base,
        output_root=fixture.base / "run", profile={"kind": "bounded_smoke",
            "train_states_requested": 1}, limits={"max_seconds": 600,
            "max_rss_bytes": 8 * prepared.GIB, "max_output_bytes": 2 * prepared.GIB,
            "max_log_bytes": 8 * prepared.MIB, "reserve_bytes": prepared.GIB},
        quota_roots=[fixture.base], storage_limit_bytes=20 * prepared.GIB,
        execution_evidence=evidence["path"])


def model_free_ui_and_evidence(root, *, stage, context, raw_seal, source_files,
                               identity=None, storage_scope, storage_limit,
                               observed_at=None, remaining=4000):
    """Declare a test-only provider observation; this is never cloud evidence."""
    now = observed_at or datetime.now(timezone.utc)
    stamp = now.isoformat().replace("+00:00", "Z")
    ui_path = root / (stage + "-ui.json")
    evidence_path = root / (stage + "-execution.json")
    ui = {"MODEL_FREE_TEST_FIXTURE": True, "provider": "kaggle",
        "evidence_basis": "actual Kaggle UI", "session_label": "MODEL_FREE_TEST_SESSION",
        "paid": False, "gpu_enabled": True, "gpu_remaining_seconds": remaining + 100,
        "session_remaining_seconds": remaining + 100, "observed_at": stamp}
    write(ui_path, ui)
    if identity is None:
        evidence = {"MODEL_FREE_TEST_FIXTURE": True, "schema": "kaggle_preparation_evidence_v2",
            "provider": "kaggle", "raw_seal": raw_seal, "context": context,
            "source_files": source_files, "provider_ui_observation": prepared.pin(ui_path),
            "session_id": ui["session_label"], "observed_at": stamp,
            "expires_at": (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
            "quota": {"paid": False, "gpu_enabled": True, "storage_scope": storage_scope,
                "storage_limit_bytes": storage_limit, "gpu_remaining_seconds": remaining,
                "session_remaining_seconds": remaining},
            "gpu": {"uuid": "GPU-MODEL-FREE-FIXTURE", "name": "Fixture GPU"}}
    else:
        identity_sha256 = (identity.get("preparation_identity_sha256")
            if isinstance(identity, dict) else identity)
        evidence = {"MODEL_FREE_TEST_FIXTURE": True,
            "schema": "visualizeit_sharded_execution_evidence_v1", "provider": "kaggle",
            "paid": False, "gpu_enabled": True, "session_id": ui["session_label"],
            "preparation_identity_sha256": identity_sha256,
            "partition_sha256": identity["partition_sha256"],
            "aggregate_sha256": identity["aggregate_sha256"],
            "source_files": source_files, "ui_context": prepared.pin(ui_path), "stage": stage,
            "observed_unix_s": now.timestamp(), "expires_unix_s": (now + timedelta(minutes=30)).timestamp(),
            "quota": {"paid": False, "gpu_enabled": True, "storage_scope": storage_scope,
                "storage_limit_bytes": storage_limit, "gpu_remaining_seconds": remaining,
                "session_remaining_seconds": remaining},
            "gpu": {"uuid": "GPU-MODEL-FREE-FIXTURE", "name": "Fixture GPU"}}
    write(evidence_path, evidence)
    return prepared.pin(ui_path), prepared.pin(evidence_path)


def fixture_working_root():
    """Use the operator's bounded working fixture root on Linux."""
    if os.name == "nt":
        return Path(__file__).resolve().parents[1]
    value = os.environ.get("TMPDIR")
    if not value:
        raise RuntimeError("Linux focused tests require the bounded TMPDIR fixture root")
    root = Path(value).absolute()
    if not root.is_dir():
        raise RuntimeError("configured Linux TMPDIR fixture root must already exist")
    return root


def has_kaggle_cross_filesystem_layout():
    if os.name == "nt":
        return False
    working = Path(os.environ.get("TMPDIR", ""))
    features = Path(os.environ.get("VISUALIZEIT_CPU_FEATURE_FIXTURE_ROOT", ""))
    return (working.is_absolute() and features.is_absolute() and working.is_dir()
        and features.is_dir() and working.stat().st_dev != features.stat().st_dev)


def temporary_working_directory(prefix):
    return tempfile.TemporaryDirectory(dir=fixture_working_root(), prefix=prefix)


class FakeFacade:
    """Small software fixture for Facade's SIGTERM-handler ownership contract."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.handler_called = threading.Event()

    def supervise(self, *_args, **_kwargs):
        self.calls += 1
        return self.behavior(self)

    def install_managed_handler(self):
        outer = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: self.handler_called.set())
        return outer

    @staticmethod
    def restore_handler(outer):
        signal.signal(signal.SIGTERM, outer)


def watch_call(facade, sample, *, initial=0):
    return sharded._supervise_with_quota_watch(facade, argv=["fixture"], cwd=Path.cwd(),
        output_root=Path.cwd(), working=Path.cwd(), log_path=Path.cwd() / "fixture.log",
        limits={}, env={}, sample=sample, initial_resident=initial, thread_name="test-watch")


class ShardedWrapperGuardTests(unittest.TestCase):
    def test_training_identity_matches_original_dataset_contract_with_full_raw_checks(self):
        for role in ('train', 'validation', 'test'):
            state = dict(id='fixture:frame0:iteration1', physical_object_id='fixture-object',
                split=role, sequence_id='fixture-sequence', scene_id='fixture-scene',
                frame_id=0, refinement_state=1)
            numeric = {'fixture-object': '13'}
            capture = sharded.sharded._portable_state_identity(state, numeric)
            expected = dict(id=state['id'], object_id='13', sequence_id='fixture-sequence',
                capture_group='fixture-scene', split='val' if role == 'validation' else role)
            with self.subTest(role=role), patch.object(prepared, 'verify_pin',
                    side_effect=AssertionError('identity verification must not read payloads')):
                self.assertEqual(sharded._training_row_identity(state, numeric, capture), expected)
                sharded._verify_training_row_identity(state, numeric, capture, expected, expected)
                for key in expected:
                    for which in ('label', 'dataset'):
                        corrupted = {**expected, key: 'FOREIGN'}
                        label, final = (corrupted, expected) if which == 'label' else (expected, corrupted)
                        with self.subTest(key=key, which=which), self.assertRaisesRegex(ValueError, 'row identity'):
                            sharded._verify_training_row_identity(state, numeric, capture, label, final)
                for key in capture:
                    corrupted = {**capture, key: 'FOREIGN'}
                    with self.subTest(raw_key=key), self.assertRaisesRegex(ValueError, 'frame/time/state'):
                        sharded._training_row_identity(state, numeric, corrupted)

    def test_file_identity_includes_device_when_inode_numbers_repeat(self):
        left = SimpleNamespace(st_dev=17, st_ino=420)
        copied = SimpleNamespace(st_dev=29, st_ino=420)
        hardlink = SimpleNamespace(st_dev=17, st_ino=420)
        self.assertFalse(sharded._same_file_identity(left, copied))
        self.assertTrue(sharded._same_file_identity(left, hardlink))

    def test_child_artifact_receipt_uses_original_execution_order(self):
        with temporary_working_directory("sharded-test-") as directory:
            root = Path(directory)
            recipe, _audit = known_role_fixture(root)
            local_recipe = sharded.sharded._projected_recipe(recipe,
                sharded.sharded.TRAIN_GROUPS["group1"])
            plan = sharded.raw.prospective(local_recipe)
            plan_path = root / "execution-plan.json"
            prepared.atomic_json(plan_path, plan)
            active = [row for row in plan["global_manifest"]["records"]
                if row["split"] in ("train", "validation")]
            train = next(row for row in active if row["split"] == "train")
            validation = next(row for row in active if row["split"] == "validation")
            rows = []
            for row in plan["global_manifest"]["records"]:
                if row["split"] == "dev":
                    continue
                if row["id"] in (train["id"], validation["id"]):
                    rows.append({"id": row["id"], "status": "available",
                        "labels": {"path": row["id"] + ".npz", "bytes": 1,
                            "sha256": "a" * 64}})
                else:
                    rows.append({"id": row["id"], "status": "unavailable",
                        "failure_reason": "MODEL_FREE_TEST_FIXTURE"})
            inventory = {"schema": "correspondence_label_inventory_v1", "rows": list(reversed(rows))}
            artifacts = [dict(id=row["id"], **row["labels"]) for row in inventory["rows"]
                if row["status"] == "available"]
            source_root = Path(__file__).resolve().parents[1]
            request = {"sharded_child": {"supervision_sources":
                sharded._child_supervision_sources(source_root)}}
            actual = sharded._ordered_child_label_artifacts(request, root, inventory, artifacts)
            expected_ids = [row["id"] for row in plan["global_manifest"]["records"]
                if row["id"] in (train["id"], validation["id"])]
            self.assertEqual([row["id"] for row in actual], expected_ids)
            self.assertNotEqual([row["id"] for row in artifacts], expected_ids)

    def test_cuda_bootstrap_preserves_module_and_all_arguments(self):
        request = {"python": "/usr/bin/python"}
        command = [request["python"], "-E", "-s", "-B", "-m", "bench.quality_adaptation_larger",
            "--mode", "extract", "--flag", "two words"]
        self.assertEqual(sharded._cuda_bootstrap_argv(request, command, "BOOTSTRAP"),
            [request["python"], "-E", "-s", "-B", "-c", "BOOTSTRAP",
             "bench.quality_adaptation_larger", "--mode", "extract", "--flag", "two words"])
        with self.assertRaisesRegex(ValueError, "pinned Python"):
            sharded._cuda_bootstrap_argv(request, [request["python"], "-s", "-m", "module"], "BOOTSTRAP")

    def test_cli_refresh_writes_a_hash_pinned_request(self):
        request = {"schema": sharded.REQUEST_SCHEMA, "limits": {"max_seconds": 900}}
        refreshed = {**request, "ui_context": {"path": "ui", "sha256": "a" * 64}}
        with temporary_working_directory("sharded-test-") as directory:
            root = Path(directory)
            source, destination = root / "input.json", root / "refreshed.json"
            source.write_text("{}", encoding="utf-8")
            with patch.object(sharded.p, "load_request", return_value=request), \
                    patch.object(sharded, "child_stage_request", return_value=refreshed) as refresh, \
                    redirect_stdout(io.StringIO()):
                status = sharded.main(["refresh-child", "--request", str(source),
                    "--request-sha256", "b" * 64, "--output", str(destination),
                    "--evidence-stage", "extract", "--ui-context", str(root / "ui.json"),
                    "--execution-evidence", str(root / "execution.json")])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), refreshed)
            refresh.assert_called_once_with(request, stage="extract",
                ui_context=root / "ui.json", execution_evidence=root / "execution.json")
            self.assertEqual(prepared.pin(destination)["sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())

    def test_cli_global_verify_serializes_compact_valid_result(self):
        request = {"schema": sharded.REQUEST_SCHEMA}
        joined = {"identity_sha256": "identity", "dataset_sha256": "dataset",
            "partition": {"partition_sha256": "partition"}, "raw": {"root": Path("/tmp/raw")},
            "docs": {"dataset": {"rows": []}}, "counts": {"train": {"requested": 0}},
            "failures": []}
        with patch.object(sharded.p, "load_request", return_value=request), \
                patch.object(sharded, "verify_prepared", return_value=joined), \
                redirect_stdout(io.StringIO()) as output:
            status = sharded.main(["global-verify", "--request", "request.json",
                "--request-sha256", "a" * 64])
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["partition_sha256"], "partition")
        self.assertNotIn("raw", result)

    def test_expanded_config_is_frozen_fp32_and_log_cap_is_sixteen_mib(self):
        config = {"device": "cuda", "epochs": 3, "max_steps": 936, "seed": 521,
            "batch_size": 1, "learning_rate": 1e-5, "weight_decay": 1e-4,
            "visibility_weight": 1.0, "amp": False, "patience": 3,
            "min_delta_px": 0.001, "gradient_clip": 1.0,
            "checkpoint_every_steps": 8, "max_seconds": 900.0,
            "max_total_seconds": 7200.0, "max_checkpoint_bytes": 335544320,
            "max_output_bytes": 1073741824}
        # Keep the fixture inside the writable checkout on locked-down Windows;
        # production and clean CI use the same ordinary temporary-file API.
        with temporary_working_directory("sharded-test-") as directory:
            path = Path(directory) / "fixture-config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            request = {"config": prepared.pin(path)}
            limits = {"max_output_bytes": 8 * sharded.GIB, "max_seconds": 900}
            self.assertEqual(sharded._validate_full_config(request, limits), config)
            config["amp"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            request["config"] = prepared.pin(path)
            with self.assertRaisesRegex(ValueError, "frozen FP32"):
                sharded._validate_full_config(request, limits)
        limits16 = {"max_rss_bytes": 8 * sharded.GIB, "max_output_bytes": 8 * sharded.GIB,
            "max_log_bytes": sharded.MAX_STAGE_LOG, "reserve_bytes": sharded.GIB}
        sharded._validate_sharded_limits(limits16)
        limits16["max_log_bytes"] = 32 * sharded.MIB
        with self.assertRaisesRegex(ValueError, "16-MiB log"):
            sharded._validate_sharded_limits(limits16)

    def test_full_pilot_admission_requires_pinned_four_step_invocation_stop(self):
        with temporary_working_directory("sharded-pilot-proof-") as directory:
            root = Path(directory)
            def measurement(case, status, *, requested=4, steps=4, claimed_reason=None):
                output = root / case
                request = {"pilot_output_root": str(output)}
                status_path = output / "train" / "status.json"
                status_path.parent.mkdir(parents=True)
                prepared.atomic_json(status_path, status)
                value = {"requested_stop_after_steps": requested,
                    "verified_optimizer_steps": steps,
                    "trainer_status_reason": claimed_reason if claimed_reason is not None else status.get("reason"),
                    "trainer_status": prepared.pin(status_path),
                    "trainer_observation": {"status": "incomplete", "returncode": 3}}
                return request, value

            request, valid = measurement("valid", {"status": "incomplete", "steps": 4,
                "reason": "invocation_step_limit"})
            self.assertTrue(sharded._full_pilot_admission_proof(request, valid))

            request, deadline_at_one = measurement("deadline", {"status": "incomplete", "steps": 1,
                "reason": "invocation_time_limit"}, steps=1)
            self.assertFalse(sharded._full_pilot_admission_proof(request, deadline_at_one))

            request, forged_reason = measurement("forged", {"status": "incomplete", "steps": 4,
                "reason": "total_time_limit"}, claimed_reason="invocation_step_limit")
            self.assertFalse(sharded._full_pilot_admission_proof(request, forged_reason))

            request, short_diagnostic = measurement("short", {"status": "incomplete", "steps": 1,
                "reason": "invocation_step_limit"}, requested=1, steps=1)
            self.assertFalse(sharded._full_pilot_admission_proof(request, short_diagnostic))

    def test_two_filesystem_feature_envelope_reserves_only_remaining_headroom(self):
        # Device 1 is /kaggle/working; device 2 is the dedicated /tmp cache.
        cap, existing, reserve = sharded.GLOBAL_FEATURE_ENVELOPE, 3 * sharded.GIB, sharded.GIB
        writes = sharded._stage_write_allocations("extract", working_device=1,
            feature_device=2, cap=cap, existing=existing)
        self.assertEqual(writes[1], sharded.MAX_CONTROL_WRITE)
        self.assertEqual(writes[2], cap - existing)
        resident = 22 * sharded.GIB
        free = {1: reserve + writes[1], 2: reserve + writes[2]}
        sharded._enforce_storage_reserve(resident, writes, free, reserve, 32 * sharded.GIB)
        with self.assertRaisesRegex(ValueError, "filesystem write allocation"):
            sharded._enforce_storage_reserve(resident, writes,
                {1: free[1], 2: free[2] - 1}, reserve, 32 * sharded.GIB)
        with self.assertRaisesRegex(ValueError, "combined resident"):
            sharded._enforce_storage_reserve(31 * sharded.GIB, writes, free,
                reserve, 32 * sharded.GIB)
        with self.assertRaisesRegex(ValueError, "dedicated filesystem"):
            sharded._stage_write_allocations("extract", working_device=1,
                feature_device=1, cap=cap, existing=existing)
        full = sharded._stage_write_allocations("extract", working_device=1,
            feature_device=2, cap=cap, existing=cap)
        self.assertEqual(full[2], 0)

    def test_two_filesystem_child_label_stage_reserves_remaining_tree_and_both_gates(self):
        cap, existing, reserve = 10 * sharded.GIB, 4 * sharded.GIB, sharded.GIB
        writes = sharded._stage_write_allocations("child-labels", working_device=11,
            feature_device=22, cap=cap, existing=existing)
        self.assertEqual(writes[11], sharded.MAX_CONTROL_WRITE)
        self.assertEqual(writes[22], cap - existing)
        enough_free = {11: reserve + writes[11], 22: reserve + writes[22]}
        sharded._enforce_storage_reserve(24 * sharded.GIB, writes, enough_free,
            reserve, 32 * sharded.GIB)
        with self.assertRaisesRegex(ValueError, "combined resident"):
            sharded._enforce_storage_reserve(26 * sharded.GIB, writes, enough_free,
                reserve, 32 * sharded.GIB)
        with self.assertRaisesRegex(ValueError, "filesystem write allocation"):
            sharded._enforce_storage_reserve(24 * sharded.GIB, writes,
                {11: enough_free[11], 22: enough_free[22] - 1}, reserve,
                32 * sharded.GIB)

    def test_child_labels_records_assembler_result_and_runtime(self):
        with temporary_working_directory("sharded-test-") as directory:
            root = Path(directory)
            features, execution = root / "features", root / "execution"
            features.mkdir()
            execution.mkdir()
            raw_root = root / "raw"
            raw_root.mkdir()
            split_path = raw_root / "split.json"
            split_path.write_bytes(b"frozen split bytes")
            (features / "extraction-binding.json").write_text(
                json.dumps({"supervision": {"fixture": True}}), encoding="utf-8")
            (features / "index.json").write_text(json.dumps({"rows": [
                {"id": "test-000", "status": "unavailable", "failure_reason": "sealed"}]}),
                encoding="utf-8")
            (execution / "prelabel-seal.json").write_text("{}", encoding="utf-8")
            raw_state = {"root": raw_root, "physical": [], "captures": [{
                "id": "test-000", "object_id": "object-0", "sequence_id": "sequence-0",
                "capture_group": "group-0", "split": "test", "status": "unavailable",
                "failure_reason": "sealed", "query_rgb_sha256": None,
                "unavailable_provenance": "sealed TEST metadata"}]}
            plan = {"schema": "kaggle_label_plan_v2", "split_manifest": prepared.pin(split_path),
                "generations": []}
            request = {"features_root": str(features), "execution_root": str(execution),
                "label_plan": plan, "sharded_child": {"shard_id": "group1"},
                "adapter": {}, "python": "fixture-python", "limits": {"max_seconds": 60},
                "ui_context": {"path": "ui", "sha256": "a" * 64},
                "execution_evidence": {"path": "evidence", "sha256": "b" * 64},
                "stage_evidence_history": [{"stage": "labels"}],
                "stage_request_identity_sha256": "identity"}
            result = {"status": "completed", "returncode": 0, "outputs_preserved": True}
            runtime = {"runtime_root": str(root / "runtime"), "sha256": "runtime-pin"}
            assembler = SimpleNamespace(command=lambda *_args: ["fixture-assembler"])
            def run_child(_request, *, output_root, **_kwargs):
                (Path(output_root) / "manifest.json").write_text("{}", encoding="utf-8")
                return result, runtime
            with patch.object(sharded.p, "cloud_guard"), \
                    patch.object(sharded.p, "verify_prelabel", return_value={"parity": []}), \
                    patch.object(sharded, "_child_supervision_provenance", return_value={"fixture": True}), \
                    patch.object(sharded, "_verify_child_supervision"), \
                    patch.object(sharded, "metadata", return_value=plan), \
                    patch.object(sharded.p, "source_runtime", return_value=(object(), None, runtime)), \
                    patch.object(sharded.p, "preparation_lock", return_value=nullcontext()), \
                    patch.object(sharded.p, "load_source", return_value=assembler), \
                    patch.object(sharded, "_ordered_child_label_artifacts", return_value=[]), \
                    patch.object(sharded, "_write_child_stage_observation", return_value={"path": "observation"}), \
                    patch.object(sharded, "_run_child_original", side_effect=run_child) as run, \
                    patch("bench.quality_ranker_data.read_split_manifest", return_value={}), \
                    patch("bench.quality_adaptation_larger.configure_torch", return_value=object()), \
                    patch("bench.quality_adaptation_larger.cleanup_device"):
                receipt = sharded._child_labels(request, raw_state, execution)
            self.assertEqual(receipt["assemble_result"], result)
            self.assertEqual(receipt["assembler_runtime"], runtime)
            run.assert_called_once()

    def test_label_import_reserves_cross_filesystem_copy_bytes_on_feature_device(self):
        copies = 7 * sharded.MIB
        writes = sharded._stage_write_allocations("labels", working_device=1,
            feature_device=2, label_copy_bytes=copies)
        self.assertEqual(writes[1], sharded.MAX_CONTROL_WRITE)
        self.assertEqual(writes[2], sharded.MAX_CONTROL_WRITE + copies)
        reserve = sharded.GIB
        resident = 20 * sharded.GIB
        free = {1: reserve + writes[1], 2: reserve + writes[2]}
        sharded._enforce_storage_reserve(resident, writes, free, reserve, 32 * sharded.GIB)
        with self.assertRaisesRegex(ValueError, "filesystem write allocation"):
            sharded._enforce_storage_reserve(resident, writes,
                {1: free[1], 2: free[2] - 1}, reserve, 32 * sharded.GIB)

    def test_preseal_cross_filesystem_copy_allowance_uses_only_owned_active_row_counts(self):
        with temporary_working_directory("sharded-test-") as directory:
            root = Path(directory)
            features = root / "global"
            shard_ids = sharded.sharded.SHARD_IDS
            paths = {sid: root / sid for sid in shard_ids}
            for path in [features, *paths.values()]:
                path.mkdir()
            request = {"features_root": str(features)}
            partition = {"full_plan": {"global_manifest": {"records": [
                {"id": "train", "split": "train"},
                {"id": "validation", "split": "validation"},
                {"id": "test", "split": "test"}]}},
                "children": {sid: {"owned_state_ids": []} for sid in shard_ids}}
            partition["children"]["smoke0"]["owned_state_ids"] = ["train", "validation", "test"]
            child_requests = {sid: {"request": {"features_root": str(paths[sid])}}
                for sid in shard_ids}
            devices = {features: 1, paths["smoke0"]: 2,
                **{paths[sid]: 1 for sid in shard_ids if sid != "smoke0"}}
            with patch.object(sharded, "_root_device", side_effect=lambda path: devices[Path(path)]):
                allowance = sharded._prospective_label_copy_allowance(request, partition, child_requests)
            self.assertEqual(allowance, 2 * prepared.PER_STATE_ENVELOPE + prepared.TEMP_ENVELOPE)

    @unittest.skipUnless(has_kaggle_cross_filesystem_layout(),
        "requires distinct /kaggle/working and /tmp filesystems; run on the pinned Linux CPU host")
    def test_cross_filesystem_label_copy_preserves_source_and_exact_bytes(self):
        payload = (b"analytic software fixture; no model data" * 40000)
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="sharded-label-source-") as source_dir, \
                tempfile.TemporaryDirectory(dir=os.environ["VISUALIZEIT_CPU_FEATURE_FIXTURE_ROOT"], prefix="sharded-label-target-") as target_dir:
            source_root, target_root = Path(source_dir), Path(target_dir)
            (source_root / "labels").mkdir(parents=True)
            (target_root / "labels").mkdir(parents=True)
            source = source_root / "labels" / "source.npz"
            source.write_bytes(payload)
            target = target_root / "labels" / "joined.npz"
            descriptor = {"path": "labels/source.npz", "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}
            source_before = source.read_bytes()
            progress = []
            self.assertNotEqual(source.stat().st_dev, target.parent.stat().st_dev)
            imported = sharded._import_label(descriptor, source_root, target,
                on_copy_progress=progress.append)
            self.assertEqual(imported["transport"], "atomic_copy")
            self.assertEqual(imported["source"], descriptor)
            self.assertEqual(imported["target"], prepared.pin(target))
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(target.stat().st_size, descriptor["bytes"])
            self.assertEqual(prepared.sha256(target), descriptor["sha256"])
            self.assertEqual(sum(progress), descriptor["bytes"])
            self.assertNotEqual((source.stat().st_dev, source.stat().st_ino),
                (target.stat().st_dev, target.stat().st_ino))

    def test_same_filesystem_label_import_is_original_hardlink(self):
        payload = b"same filesystem analytic label fixture"
        with temporary_working_directory("sharded-test-") as directory:
            root = Path(directory)
            source_root, target_root = root / "child", root / "global"
            source_root.mkdir()
            target_root.mkdir()
            source = source_root / "source.npz"
            source.write_bytes(payload)
            descriptor = {"path": "source.npz", "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}
            target = target_root / "joined.npz"
            try:
                imported = sharded._import_label(descriptor, source_root, target)
            except PermissionError as exc:
                self.skipTest("host policy denies hardlinks in the workspace: " + str(exc))
            self.assertEqual(imported["transport"], "hardlink")
            self.assertEqual((source.stat().st_dev, source.stat().st_ino),
                (target.stat().st_dev, target.stat().st_ino))

    def test_preparation_identity_allows_evidence_to_bind_identity_without_a_hash_cycle(self):
        request = {"partition": {"sha256": "partition"}, "source_files": {"wrapper": "pin"},
            "adapter": {"sha256": "adapter"}, "source_install": "/install", "python": "/python",
            "runtime": {"sha256": "runtime"}, "config": {"sha256": "config"},
            "initial_ui_context": {"path": "/evidence/ui.json", "sha256": "ui"},
            "neural_inputs": [], "limits": {"max_seconds": 900}, "storage_limit_bytes": 32 * sharded.GIB,
            "quota_roots": ["/work", "/tmp/features"], "input_root": "/input",
            "working_root": "/work", "aggregate_root": "/work/raw", "execution_root": "/work/exec",
            "features_root": "/tmp/features/global", "feature_storage_root": "/tmp/features",
            "output_root": "/work/full", "pilot_output_root": "/work/pilot",
            "child_requests": [], "shard_refs": [], "initial_execution_evidence": {"sha256": "placeholder"}}
        raw_state = {"seal": {"aggregate_sha256": "aggregate"}}
        identity = sharded._preparation_identity(request, raw_state)
        with temporary_working_directory("sharded-test-") as directory:
            path = Path(directory) / "execution-evidence.json"
            evidence = {"schema": "visualizeit_sharded_execution_evidence_v1",
                "preparation_identity_sha256": identity,
                "ui_context": request["initial_ui_context"], "stage": "identity"}
            path.write_text(json.dumps(evidence), encoding="utf-8")
            request["initial_execution_evidence"] = prepared.pin(path)
            self.assertEqual(sharded._preparation_identity(request, raw_state), identity)
            self.assertEqual(prepared.metadata(request["initial_execution_evidence"]), evidence)

    def test_prelaunch_quota_failure_never_enters_facade(self):
        facade = FakeFacade(lambda _facade: {"status": "completed"})
        original = signal.getsignal(signal.SIGTERM)

        def fail_prelaunch():
            raise OSError("quota unavailable")

        with self.assertRaisesRegex(OSError, "quota unavailable"):
            watch_call(facade, fail_prelaunch)
        self.assertEqual(facade.calls, 0)
        self.assertIs(signal.getsignal(signal.SIGTERM), original)

    def test_sigterm_before_managed_handler_cancels_without_launch(self):
        launched = threading.Event()

        def slow_before_managed_handler(facade):
            # A real Facade does source/setup checks before it installs its
            # managed process-group handler; this exercises that boundary.
            outer = signal.getsignal(signal.SIGTERM)
            outer(signal.SIGTERM, None)
            launched.set()  # unreachable: the outer handler aborts before launch.
            return {"status": "completed"}

        facade = FakeFacade(slow_before_managed_handler)
        result, watch = watch_call(facade, lambda: {"resident_bytes": 0})
        self.assertFalse(launched.is_set())
        self.assertEqual(result["reason"], "interrupted_before_managed_launch")
        self.assertTrue(result["outputs_preserved"])
        self.assertTrue(watch["outer_cancelled"])
        self.assertTrue(watch["sampler_stopped"])

    def test_live_quota_violation_reaches_only_the_managed_handler(self):
        calls = 0

        def sample():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"resident_bytes": 10}
            raise OSError("live filesystem reserve exhausted")

        def active_child(facade):
            outer = facade.install_managed_handler()
            try:
                self.assertTrue(facade.handler_called.wait(timeout=2))
                return {"status": "incomplete", "reason": "managed cancellation",
                    "outputs_preserved": True}
            finally:
                facade.restore_handler(outer)

        facade = FakeFacade(active_child)
        # Windows os.kill(SIGTERM) terminates the process instead of dispatching
        # a Python signal; this fixture invokes only the installed handler.
        def dispatch_to_installed_handler(_pid, signum):
            signal.getsignal(signum)(signum, None)

        with patch.object(sharded.os, "kill", side_effect=dispatch_to_installed_handler):
            result, watch = watch_call(facade, sample, initial=10)
        self.assertEqual(facade.calls, 1)
        self.assertEqual(result["reason"], "managed cancellation")
        self.assertTrue(result["outputs_preserved"])
        self.assertTrue(any("live filesystem reserve exhausted" in error
            for error in watch["monitor_errors"]))
        self.assertTrue(watch["sampler_stopped"])

    def test_late_sampler_failure_cannot_signal_after_fast_facade_exit(self):
        sample_started, release_sample = threading.Event(), threading.Event()
        calls = 0

        def delayed_sample():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"resident_bytes": 0}
            sample_started.set()
            self.assertTrue(release_sample.wait(timeout=2))
            raise OSError("late quota read failure")

        def fast_child(facade):
            outer = facade.install_managed_handler()
            try:
                self.assertTrue(sample_started.wait(timeout=2))
                threading.Timer(0.05, release_sample.set).start()
                return {"status": "completed", "reason": "primary result", "outputs_preserved": True}
            finally:
                facade.restore_handler(outer)

        original = signal.getsignal(signal.SIGTERM)
        result, watch = watch_call(FakeFacade(fast_child), delayed_sample)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["reason"], "primary result")
        self.assertTrue(result["outputs_preserved"])
        self.assertTrue(any("late quota read failure" in error for error in watch["monitor_errors"]))
        self.assertTrue(watch["sampler_stopped"])
        self.assertIs(signal.getsignal(signal.SIGTERM), original)


class ShardedGlobalPreparedIntegrationTests(unittest.TestCase):
    """Actual RAW/feature/label/dataset validators around declared software doubles."""

    def _fresh_raw_seals(self, fixtures, frozen):
        with patch.object(raw, "frozen", side_effect=frozen), \
                patch.object(raw, "_require_stage"), patch.object(raw, "resource_check"):
            for fixture in fixtures.values():
                (fixture.root / "raw-seal.json").unlink()
                closing = fixture.root / "stage-seal-capture.json"
                closing_bytes = closing.read_bytes()
                closing.unlink()
                raw.seal_capture(fixture.context)
                closing.write_bytes(closing_bytes)
                state = prepared.raw_metadata(fixture.context,
                    prepared.pin(fixture.root / "raw-seal.json"), verify_payloads=False)
                fixture.state = state
                fixture.captures, fixture.physical = state["captures"], state["physical"]
                fixture.seal, fixture.seal_pin = state["seal"], prepared.pin(fixture.root / "raw-seal.json")

    def _child_roots(self, request, fixture, ui_pin, evidence_pin, label_plan_pin,
                     retained_raw_roots):
        context = raw.read_context(Path(request["context"]["path"]), request["context"]["sha256"])
        raw_doc = context.document()
        runtime = prepared.metadata(request["runtime"])
        roots = {Path(request[key]).absolute() for key in (
            "working_root", "source_root", "source_install", "execution_root", "features_root",
            "feature_storage_root", "control_root")}
        roots.update((Path(raw_doc["output_root"]).absolute(), Path(raw_doc["scratch_root"]).absolute(),
            Path(raw_doc["model_cache"]).absolute(), Path(runtime["runtime_root"]).absolute(),
            (Path(request["source_root"]) / ".cache/model-quality").absolute(),
            (Path(request["source_install"]) / original_prepared.p.load_source(
                request["adapter"], "_test_child_adapter").ROOT18 / ".cache/model-quality").absolute(),
            Path(ui_pin["path"]).parent.absolute(), Path(evidence_pin["path"]).parent.absolute(),
            Path(request["exposure_audit"]["path"]).parent.absolute(),
            Path(label_plan_pin["path"]).parent.absolute()))
        roots.update(Path(value).absolute() for value in retained_raw_roots)
        return sorted(map(str, roots))

    def _child_request(self, fixture, sid, partition_pin, projection, cache_root,
                       working, support, retained_raw_roots, base_request):
        source_root = Path(base_request["source_root"])
        source_install = Path(base_request["source_install"])
        source_install.mkdir(parents=True, exist_ok=True)
        input_root = fixture.base / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        runtime_root = working / "runtime-root"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_path = fixture.base / "runtime.json"
        runtime_pin = write(runtime_path, {"MODEL_FREE_TEST_FIXTURE": True,
            "runtime_root": str(runtime_root)})
        label_plan_path = fixture.base / "label-plan.json"
        label_plan_pin = write(label_plan_path, {"schema": "kaggle_label_plan_v2",
            "split_manifest": prepared.pin(fixture.root / "split.json"), "generations": []})
        limits = dict(base_request["limits"])
        limits["max_log_bytes"] = 16 * prepared.MIB
        limits["max_output_bytes"] = 2 * prepared.GIB
        limits["reserve_bytes"] = prepared.GIB
        ui_pin, evidence_pin = model_free_ui_and_evidence(support / sid,
            stage="identity", context=prepared.pin(fixture.root / "context.json"),
            raw_seal=prepared.pin(fixture.root / "raw-seal.json"),
            source_files=base_request["source_files"],
            storage_scope=retained_raw_roots, storage_limit=sharded.AGGREGATE_STORAGE_LIMIT,
            remaining=3600)
        roots = self._child_roots({"context": prepared.pin(fixture.root / "context.json"),
            "adapter": base_request["adapter"],
            "runtime": runtime_pin, "working_root": str(working), "source_root": str(source_root),
            "source_install": str(source_install), "execution_root": str(fixture.execution),
            "features_root": str(fixture.features), "feature_storage_root": str(cache_root),
            "control_root": str(fixture.base / "control"),
            "exposure_audit": fixture.audit_pin}, fixture, ui_pin, evidence_pin,
            label_plan_pin, retained_raw_roots)
        # Evidence scopes use the request's complete normalized root roster.
        ui_doc = prepared.metadata(ui_pin)
        evidence_doc = prepared.metadata(evidence_pin)
        evidence_doc["quota"]["storage_scope"] = roots
        write(Path(evidence_pin["path"]), evidence_doc)
        evidence_pin = prepared.pin(evidence_pin["path"])
        request = sharded.child_request_document(partition=partition_pin, shard_id=sid,
            context=prepared.pin(fixture.root / "context.json"),
            raw_seal=prepared.pin(fixture.root / "raw-seal.json"), projection=projection,
            exposure_audit=fixture.audit_pin, label_plan=label_plan_pin,
            source_root=source_root, source_install=source_install,
            python=base_request["python"], runtime=runtime_pin, config=base_request["config"],
            adapter=base_request["adapter"], input_root=input_root, working_root=working,
            execution_root=fixture.execution, features_root=fixture.features,
            feature_storage_root=cache_root, retained_raw_roots=retained_raw_roots,
            control_root=fixture.base / "control", ui_context=ui_pin,
            execution_evidence=evidence_pin, limits=limits, quota_roots=roots)
        history_inputs = [
            ("extract", 3500), ("seal-features", 3400), ("labels", 3300)]
        for stage, remaining in history_inputs:
            ui_pin, evidence_pin = model_free_ui_and_evidence(support / sid,
                stage=stage, context=request["context"], raw_seal=request["raw_seal"],
                source_files=request["source_files"], storage_scope=request["quota_roots"],
                storage_limit=sharded.AGGREGATE_STORAGE_LIMIT, remaining=remaining)
            request = sharded.child_stage_request(request, stage=stage,
                ui_context=ui_pin, execution_evidence=evidence_pin)
        return request

    def _bind_child_outputs(self, fixture, request):
        features = Path(request["features_root"])
        execution = Path(request["execution_root"])
        history = request["stage_evidence_history"]
        binding_path = features / "extraction-binding.json"
        binding = prepared.read_json(binding_path)
        provenance = sharded._child_supervision_provenance(request)
        binding.update(sharded_projection=request["sharded_child"]["projection"],
            sharded_partition_sha256=sharded.sharded.verify_partition(request["sharded_child"]["partition"])["partition_sha256"],
            source_files=request["source_files"], runtime=request["runtime"], stage="extract",
            stage_request_identity_sha256=request["stage_request_identity_sha256"],
            ui_context=history[1]["ui_context"], execution_evidence=history[1]["execution_evidence"],
            stage_evidence_history=history[:2], supervision=provenance)
        write(binding_path, binding)
        (execution / "prelabel-seal.json").unlink()
        prepared.seal_features(fixture.state, execution, features)
        receipt_path = features / "label-receipt.json"
        receipt = prepared.read_json(receipt_path)
        receipt.update(prelabel_seal=prepared.pin(execution / "prelabel-seal.json"),
            stage="labels", ui_context=history[-1]["ui_context"],
            execution_evidence=history[-1]["execution_evidence"], stage_evidence_history=history,
            supervision=provenance, sharded_child=request["sharded_child"])
        inventory = prepared.read_json(features / "labels.json")
        artifacts = [dict(id=row["id"], **row["labels"])
            for row in inventory["rows"] if row.get("status") == "available"]
        receipt["label_artifacts"] = sharded._ordered_child_label_artifacts(
            request, execution, inventory, artifacts)
        write(receipt_path, receipt)
        return sharded.finalize_child_request(request)

    def test_authentic_global_identity_prelabel_label_join_and_prepared_validators(self):
        if not has_kaggle_cross_filesystem_layout():
            self.skipTest("requires the bounded Linux Kaggle WORK/tmp fixture devices")
        feature_fixture_root = Path(os.environ["VISUALIZEIT_CPU_FEATURE_FIXTURE_ROOT"]).absolute()
        self._exercise_global(fixture_working_root(), feature_fixture_root)

    def test_analytical_global_metadata_workflow_with_declared_platform_doubles(self):
        if os.environ.get('VISUALIZEIT_LOCAL_ANALYTICAL_FIXTURE') != '1':
            self.skipTest('explicit local analytical fixture opt-in required; no Linux admission claim')
        # This exercises the complete arrays/receipt/assembler workflow locally.
        # Only cloud root/device admission and provider observations are doubled;
        # actual Linux cross-device/guard evidence is recorded separately.
        with temporary_working_directory('a-') as temporary:
            base = Path(temporary).absolute()
            work_parent, feature_parent = base / 'work', base / 'features'
            work_parent.mkdir(); feature_parent.mkdir()
            native_path = type(Path())
            class FixturePath(native_path):
                def __new__(cls, *parts):
                    if parts == ('/kaggle/working',):
                        parts = (str(work_parent),)
                    return super().__new__(cls, *parts)
                def __init__(self, *parts):
                    if parts == ('/kaggle/working',):
                        parts = (str(work_parent),)
                    # Path initialization moved from __new__ to __init__ in
                    # newer Python; retain both supported fixture runtimes.
                    if native_path.__init__ is not object.__init__:
                        super().__init__(*parts)
            def logical_device(path):
                return 2 if Path(path).absolute().is_relative_to(feature_parent) else 1
            actual_device = sharded._root_device
            def actual_transport(source_root, destination):
                return ('hardlink' if actual_device(source_root) == actual_device(destination)
                        else 'atomic_copy')
            actual_run = subprocess.run
            def provider_observation(argv, *args, **kwargs):
                if argv[0] == 'nvidia-smi':
                    return SimpleNamespace(stdout='GPU-MODEL-FREE-FIXTURE, Fixture GPU\n', returncode=0, stderr='')
                return actual_run(argv, *args, **kwargs)
            with patch.object(sharded, 'Path', FixturePath), \
                    patch.object(sharded, '_root_device', side_effect=logical_device), \
                    patch.object(sharded, '_label_import_transport', side_effect=actual_transport), \
                    patch.object(sharded.subprocess, 'run', side_effect=provider_observation):
                self._exercise_global(work_parent, feature_parent)

    def _exercise_global(self, working_parent, feature_fixture_root):
        with tempfile.TemporaryDirectory(dir=working_parent, prefix="w-") as work_tmp, \
                tempfile.TemporaryDirectory(dir=feature_fixture_root, prefix="f-") as feature_tmp:
            base, cache_root = Path(work_tmp), Path(feature_tmp)
            working, support = base / "working", base / "support"
            working.mkdir()
            support.mkdir()
            recipe, full_audit = known_role_fixture(base)
            fixtures = {}
            original_full = np.full
            for ordinal, sid in enumerate(sharded.sharded.SHARD_IDS):
                shard_base = working / ("fixture-" + sid)
                shard_base.mkdir()
                child_recipe = sharded.sharded._projected_recipe(recipe,
                    sharded.sharded.TRAIN_GROUPS[sid])
                with patch.object(np, "full", side_effect=lambda shape, fill_value, *args,
                                  _ordinal=ordinal, **kwargs: original_full(shape,
                                      (0.25 + 0.05 * _ordinal if tuple(shape) == (1, 3, 280, 280)
                                       and float(fill_value) == 0.25 else fill_value), *args, **kwargs)):
                    fixture = Fixture(shard_base, recipe=child_recipe)
                if sid != "smoke0":
                    fixture.features = cache_root / sid
                    fixture.features.mkdir()
                fixtures[sid] = fixture
            smoke = fixtures["smoke0"]
            seed_frame = next(row for row in smoke.physical
                if row["split"] == "train" and row["render"]["status"] == "available")
            seed_frame["seed"] = {"status": "available", "poses": [[
                [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
                "selection": "first_FoundPose_quality_sorted_hypothesis"}
            write(smoke.root / "physical-index.json", {"rows": smoke.physical})

            def original_frozen(context):
                # Offline fixture adapter only: keep the product's source,
                # projection, seal and aggregate validators real while supplying
                # the fixture's original analytical rows in place of a cloud RAW producer.
                raw.verify_sources(context)
                for fixture in fixtures.values():
                    if fixture.context.sha256 == context.sha256:
                        return fixture.context.document(), fixture.spec, fixture.plan, fixture.physical
                raise AssertionError("unexpected raw context")

            self._fresh_raw_seals(fixtures, original_frozen)
            with patch.object(raw, "frozen", side_effect=original_frozen):
                partition_result = sharded.sharded.freeze_partition(recipe, full_audit,
                    {"context": prepared.pin(smoke.root / "context.json"),
                     "raw_seal": prepared.pin(smoke.root / "raw-seal.json")},
                    working / "partition")
                partition_pin, partition = partition_result["partition"], sharded.sharded.verify_partition(
                    partition_result["partition"])
                refs = [{"shard_id": "smoke0", "context": prepared.pin(smoke.root / "context.json"),
                    "raw_seal": prepared.pin(smoke.root / "raw-seal.json"),
                    "projection": partition_result["imported_shard0_projection"]}]
                for sid in sharded.sharded.SHARD_IDS[1:]:
                    fixture = fixtures[sid]
                    projection = sharded.sharded.seal_shard(partition_pin, sid,
                        prepared.pin(fixture.root / "context.json"), prepared.pin(fixture.root / "raw-seal.json"),
                        working / (sid + "-projection.json"))
                    refs.append({"shard_id": sid, "context": prepared.pin(fixture.root / "context.json"),
                        "raw_seal": prepared.pin(fixture.root / "raw-seal.json"), "projection": projection})
                aggregate_result = sharded.sharded.aggregate_raw(partition_pin, refs, working / "aggregate")
                aggregate_root = Path(aggregate_result["root"])
            with patch.object(raw, "frozen", side_effect=original_frozen):
                aggregate_state = sharded.sharded.verify_aggregate_raw(partition_pin,
                    aggregate_root, payloads=True)

                # Build authentic child RAW/execution/features/labels using the
                # existing analytical fixture and unchanged validators.
                base_requests = {sid: _prepared_child_fixture(fixtures[sid])
                    for sid in sharded.sharded.SHARD_IDS}
                smoke_request = base_requests["smoke0"]
                smoke_request["storage_limit_bytes"] = 18 * prepared.GIB
                smoke_binding_path = Path(smoke_request["features_root"]) / "extraction-binding.json"
                smoke_binding = prepared.read_json(smoke_binding_path)
                smoke_binding["supervision"] = {"version": "kaggle_feature_supervision_v1",
                    "sources": {"bridge": prepared.pin(Path(smoke_request["source_root"]) / "bench/quality_adaptation_supervised_v1.py"),
                        "supervisor": prepared.pin(Path(smoke_request["source_root"]) / "scripts/kaggle_supervision_v1.py"),
                        "scanner": prepared.pin(Path(smoke_request["source_root"]) / "bench/quality_adaptation_portable.py")},
                    "validation_adapter": smoke_request["adapter"],
                    "supervisor": "scripts.kaggle_supervision_v1.Facade", "output_scanner": "raw._usage"}
                write(smoke_binding_path, smoke_binding)
                (Path(smoke_request["execution_root"]) / "prelabel-seal.json").unlink()
                prepared.seal_features(fixtures["smoke0"].state,
                    Path(smoke_request["execution_root"]), Path(smoke_request["features_root"]))
                smoke_receipt_path = Path(smoke_request["features_root"]) / "label-receipt.json"
                smoke_receipt = prepared.read_json(smoke_receipt_path)
                smoke_receipt["prelabel_seal"] = prepared.pin(Path(smoke_request["execution_root"]) / "prelabel-seal.json")
                write(smoke_receipt_path, smoke_receipt)
                smoke_request["prelabel_seal"] = prepared.pin(
                    Path(smoke_request["execution_root"]) / "prelabel-seal.json")
                for key, filename in (("extraction_binding", "extraction-binding.json"),
                        ("label_receipt", "label-receipt.json")):
                    smoke_request[key] = prepared.pin(Path(smoke_request["features_root"]) / filename)

                retained_raw_roots = sorted({str(working), str(cache_root), *[
                    str(Path(fixtures[sid].state[key]).absolute())
                    for sid in fixtures for key in ("root",)]} | {
                    str(Path(fixtures[sid].state["context"][key]).absolute())
                    for sid in fixtures for key in ("scratch_root", "model_cache")})
                # Include all immutable original RAW output/scratch/cache roots,
                # plus every sibling cache and current working fixture root.
                for fixture in fixtures.values():
                    doc = fixture.context.document()
                    retained_raw_roots.extend(str(Path(doc[key]).absolute())
                        for key in ("output_root", "scratch_root", "model_cache"))
                retained_raw_roots = sorted(set(retained_raw_roots))

                child_requests = [{"shard_id": "smoke0", "request": smoke_request,
                    "projection": refs[0]["projection"]}]
                for sid in sharded.sharded.SHARD_IDS[1:]:
                    feature_root = Path(fixtures[sid].features)
                    (fixtures[sid].base / "control").mkdir(exist_ok=True)
                    request = self._child_request(fixtures[sid], sid, partition_pin,
                        next(ref["projection"] for ref in refs if ref["shard_id"] == sid),
                        cache_root, working, support, retained_raw_roots, base_requests[sid])
                    request = self._bind_child_outputs(fixtures[sid], request)
                    child_requests.append({"shard_id": sid, "request": request,
                        "projection": request["sharded_child"]["projection"]})
                    self.assertEqual(Path(request["features_root"]), feature_root)
                    sharded.verify_child_request(request, payloads=True)
                original_prepared.verify_prepared(smoke_request, payloads=True)

                # Freeze a complete global stage request before artifacts exist;
                # each global stage below writes its own original-format receipt.
                runtime_path = working / "global-runtime.json"
                runtime_root = working / "runtime-root"
                runtime_root.mkdir(exist_ok=True)
                runtime_pin = write(runtime_path, {"MODEL_FREE_TEST_FIXTURE": True,
                    "runtime_root": str(runtime_root)})
                config_path = working / "expanded-config.json"
                config_pin = write(config_path, {"device": "cuda", "epochs": 3,
                    "max_steps": 936, "max_seconds": 900., "max_total_seconds": 7200.,
                    "max_checkpoint_bytes": 335544320, "max_output_bytes": 1073741824,
                    "seed": 521, "batch_size": 1, "learning_rate": 1e-5,
                    "weight_decay": 1e-4, "visibility_weight": 1., "amp": False,
                    "patience": 3, "min_delta_px": .001, "gradient_clip": 1.,
                    "checkpoint_every_steps": 8})
                global_limits = {"max_seconds": 900, "max_rss_bytes": 8 * prepared.GIB,
                    "max_output_bytes": 8 * prepared.GIB, "max_log_bytes": 16 * prepared.MIB,
                    "reserve_bytes": prepared.GIB}
                global_source_files = sharded._root_sources(Path(smoke_request["source_root"]))
                global_ui, _placeholder_evidence = model_free_ui_and_evidence(support,
                    stage="identity", context=refs[0]["context"], raw_seal=refs[0]["raw_seal"],
                    source_files=global_source_files,
                    storage_scope=[], storage_limit=sharded.AGGREGATE_STORAGE_LIMIT,
                    remaining=3600)
                # Build the prospective identity before writing its bound evidence.
                global_features = cache_root / "global"
                global_exec = working / "global-execution"
                global_work = working
                global_source_install = working / "global-source-install"
                global_source_install.mkdir(exist_ok=True)
                input_root = working / "readonly-input-fixture"
                input_root.mkdir(exist_ok=True)
                global_output, pilot_output = working / "full-run", working / "pilot-run"
                # Exact root union used by global admission; read-only input is
                # deliberately outside recursive quota scans.
                global_roots = {str(Path(value).absolute()) for value in (
                    global_work, Path(smoke_request["source_root"]), global_source_install,
                    aggregate_root, global_exec, global_features, cache_root, global_output,
                    pilot_output, runtime_root, Path(smoke_request["source_root"]) / ".cache/model-quality",
                    global_source_install / original_prepared.p.load_source(
                        smoke_request["adapter"], "_test_global_adapter").ROOT18 / ".cache/model-quality",
                    support, working)}
                for item in child_requests:
                    child = item["request"]
                    global_roots.update(child.get("retained_raw_roots", []))
                    global_roots.update(str(Path(child[key]).absolute()) for key in
                        ("source_root", "source_install", "features_root", "execution_root", "working_root"))
                    if child.get("control_root"):
                        global_roots.add(str(Path(child["control_root"]).absolute()))
                    raw_context = raw.read_context(Path(child["context"]["path"]), child["context"]["sha256"])
                    raw_doc = raw_context.document()
                    global_roots.update(str(Path(raw_doc[key]).absolute()) for key in
                        ("output_root", "scratch_root", "model_cache"))
                    for key in ("ui_context", "execution_evidence", "exposure_audit", "label_plan", "runtime", "config"):
                        desc = child.get(key)
                        if isinstance(desc, dict) and isinstance(desc.get("path"), str):
                            global_roots.add(str(Path(desc["path"]).parent.absolute()))
                global_roots.add(str(Path(global_ui["path"]).parent.absolute()))
                global_roots.update((str(config_path.parent.absolute()),
                    str(Path(partition_pin["path"]).parent.absolute())))
                global_roots = sorted(global_roots)
                identity_arguments = dict(partition=partition_pin,
                    aggregate_root=aggregate_root, execution_root=global_exec,
                    features_root=global_features, feature_storage_root=cache_root,
                    runtime=runtime_pin, config=config_pin, source_install=global_source_install,
                    python=Path(sys.executable), input_root=input_root, working_root=global_work,
                    output_root=global_output, pilot_output_root=pilot_output,
                    child_requests=child_requests, shard_refs=refs, ui_context=global_ui,
                    limits=global_limits, quota_roots=global_roots)
                identity = sharded.prospective_preparation_identity(**identity_arguments)
                binding_path = global_work / (".visualizeit-sharded-global-stage-" + identity + ".json")
                self.assertFalse(binding_path.exists(), "identity preview must not publish a request binding")
                identity_fields = {"preparation_identity_sha256": identity,
                    "partition_sha256": partition["partition_sha256"],
                    "aggregate_sha256": aggregate_state["seal"]["aggregate_sha256"]}
                # Keep the exact UI pin already included in the semantic identity;
                # only the evidence document can be rebuilt after that identity
                # is known.
                initial_ui_doc = prepared.metadata(global_ui)
                observed_dt = datetime.fromisoformat(initial_ui_doc["observed_at"].replace("Z", "+00:00"))
                evidence_path = support / "identity-bound-execution.json"
                evidence_doc = {"MODEL_FREE_TEST_FIXTURE": True,
                    "schema": "visualizeit_sharded_execution_evidence_v1", "provider": "kaggle",
                    "paid": False, "gpu_enabled": True, "session_id": initial_ui_doc["session_label"],
                    "preparation_identity_sha256": identity,
                    "partition_sha256": partition["partition_sha256"],
                    "aggregate_sha256": aggregate_state["seal"]["aggregate_sha256"],
                    "source_files": global_source_files, "ui_context": global_ui, "stage": "identity",
                    "observed_unix_s": observed_dt.timestamp(),
                    "expires_unix_s": (observed_dt + timedelta(minutes=30)).timestamp(),
                    "quota": {"paid": False, "gpu_enabled": True, "storage_scope": global_roots,
                        "storage_limit_bytes": sharded.AGGREGATE_STORAGE_LIMIT,
                        "gpu_remaining_seconds": 3600, "session_remaining_seconds": 3600},
                    "gpu": {"uuid": "GPU-MODEL-FREE-FIXTURE", "name": "Fixture GPU"}}
                write(evidence_path, evidence_doc)
                evidence_pin = prepared.pin(evidence_path)
                fake_runtime = {"MODEL_FREE_TEST_FIXTURE": True}
                gpu_result = SimpleNamespace(stdout="GPU-MODEL-FREE-FIXTURE, Fixture GPU\n")
                with patch.object(sharded.subprocess, "run", return_value=gpu_result):
                    request = sharded.preparation_request_document(
                        **identity_arguments, execution_evidence=evidence_pin)
                self.assertEqual(request["stage_request_identity_sha256"], identity)
                self.assertEqual(request["initial_execution_evidence"], evidence_pin)
                self.assertEqual(request["stage_evidence_history"], [{"stage": "identity",
                    "ui_context": global_ui, "execution_evidence": evidence_pin}])
                adapter = original_prepared.p.load_source(request["adapter"], "_test_global_adapter")
                with patch.object(sharded.subprocess, "run", return_value=gpu_result):
                    sharded._verify_global_stage_history(request, aggregate_state)
                with patch.object(prepared, "cloud_guard"), patch.object(sharded, "quota_check", return_value={}), \
                        patch.object(prepared, "source_runtime", return_value=(adapter, working, fake_runtime)), \
                        patch.object(sharded.subprocess, "run", return_value=gpu_result):
                    sharded.prepare_global_identity(request)

                def refresh_global(current, stage, remaining):
                    ui, evidence = model_free_ui_and_evidence(support,
                        stage=stage, context=refs[0]["context"], raw_seal=refs[0]["raw_seal"],
                        source_files=current["source_files"], identity=identity_fields,
                        storage_scope=global_roots,
                        storage_limit=sharded.AGGREGATE_STORAGE_LIMIT, remaining=remaining)
                    with patch.object(sharded.subprocess, "run", return_value=gpu_result):
                        return sharded.global_stage_request(current, stage=stage,
                            ui_context=ui, execution_evidence=evidence)

                request = refresh_global(request, "extract", 3500)
                def extraction_double(current, *, stage, argv, output_root, log_path,
                                      raw_state, **_kwargs):
                    self.assertEqual(stage, "extract")
                    features = Path(output_root)
                    features.mkdir()
                    crop_pin = prepared.pin(Path(raw_state["root"]) / "raw/v2-crops.json")
                    recipe_doc = prepared.read_json(Path(smoke_request["features_root"]) / "recipe.json")
                    recipe_doc["crop_manifest_sha256"] = prepared.canonical(prepared.metadata(crop_pin))
                    recipe_doc["device"] = raw_state["context"]["device"]
                    write(features / "recipe.json", recipe_doc)
                    captures = prepared.indexed(raw_state["captures"], "global fixture captures")
                    ownership = {ident: sid for sid, child in partition["children"].items()
                        for ident in child["owned_state_ids"]}
                    child_feature_rows = {sid: prepared.indexed(prepared.read_json(
                        Path(next(row["request"] for row in child_requests if row["shard_id"] == sid)
                            ["features_root"]) / "index.json")["rows"], "child fixture features")
                        for sid in sharded.sharded.SHARD_IDS}
                    original_rows = []
                    for capture in raw_state["captures"]:
                        if capture["split"] not in ("train", "validation") or capture["status"] != "available":
                            continue
                        ident = capture["id"]
                        sid = ownership[ident]
                        child_request = next(row["request"] for row in child_requests
                            if row["shard_id"] == sid)
                        child_feature = child_feature_rows[sid][ident]
                        source = prepared.verify_pin(child_feature["features"],
                            Path(child_request["features_root"]))
                        target = features / "features" / (hashlib.sha256(ident.encode()).hexdigest() + ".npz")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if source.stat().st_dev == target.parent.stat().st_dev:
                            os.link(source, target)
                        else:
                            shutil.copyfile(source, target)
                        original_rows.append({"id": ident, "status": "available",
                            "crops": capture["crops"],
                            "query_rgb_sha256": capture["query_rgb_sha256"],
                            "features": {**prepared.pin(target),
                                "path": target.relative_to(features).as_posix()}})
                    write(features / "index.json", {"recipe_sha256": prepared.canonical(recipe_doc),
                        "status": "completed", "elapsed_seconds": .01, "rows": original_rows})
                    return ({"status": "completed", "returncode": 0,
                        "outputs_preserved": True, "MODEL_FREE_TEST_FIXTURE": True}, fake_runtime)

                with patch.object(prepared, "cloud_guard"), patch.object(sharded, "quota_check", return_value={}), \
                        patch.object(prepared, "source_runtime", return_value=(adapter, working, fake_runtime)), \
                        patch.object(sharded.subprocess, "run", return_value=gpu_result), \
                        patch.object(sharded, "_run_original", side_effect=extraction_double):
                    extraction_observation = sharded.extract_global(request)
                self.assertEqual(extraction_observation["result"]["status"], "completed")
                features = Path(request["features_root"])
                request = refresh_global(request, "seal-features", 3400)
                with patch.object(prepared, "cloud_guard"), patch.object(sharded, "quota_check", return_value={}), \
                        patch.object(prepared, "source_runtime", return_value=(adapter, working, fake_runtime)), \
                        patch.object(sharded.subprocess, "run", return_value=gpu_result):
                    sharded.seal_global_features(request)
                request = refresh_global(request, "labels", 3300)

                def assembler_double(current, *, stage, argv, output_root, log_path,
                                     raw_state, label_copy_bytes=None, **_kwargs):
                    self.assertEqual(stage, "labels")
                    result = assemble(Path(current["features_root"]),
                        Path(current["features_root"]) / "labels.json",
                        Path(current["features_root"]) / "manifest.json")
                    return ({"status": "completed", "returncode": 0,
                        "outputs_preserved": True, "MODEL_FREE_TEST_FIXTURE": True,
                        "assembler": result}, {"MODEL_FREE_TEST_FIXTURE": True})

                with patch.object(prepared, "cloud_guard"), patch.object(sharded, "quota_check", return_value={}), \
                        patch.object(prepared, "source_runtime", return_value=(adapter, working, fake_runtime)), \
                        patch.object(sharded.subprocess, "run", return_value=gpu_result), \
                        patch.object(sharded, "_run_original", side_effect=assembler_double):
                    sharded.join_labels_after_prelabel(request)
                finalized = sharded.finalize_prepared_request(request)
                joined = sharded.metadata_join(finalized)
                self.assertEqual(joined["counts"]["train"]["jointly_usable"], 4)
                self.assertEqual(joined["counts"]["validation"]["jointly_usable"], 1)
                admission = sharded.assess_admission(joined)
                self.assertFalse(admission["admitted"])
                self.assertIn("fewer_than_240_globally_usable_train_rows", admission["blockers"])
                payloads = sharded.verify_prepared(finalized, payloads=True)
                self.assertEqual(payloads["identity_sha256"], joined["identity_sha256"])
                required = sharded._required_roots(finalized, partition,
                    joined["child_requests"], adapter)
                self.assertNotIn(Path(finalized["input_root"]).absolute(), required)

                # A late sealed TEST descriptor in the final child must fail
                # during its authentic metadata validator, before either child
                # payload verifier is entered.
                last_child = copy.deepcopy(next(item["request"]
                    for item in finalized["child_requests"] if item["shard_id"] == "group3"))
                dataset_path = Path(last_child["dataset"]["path"])
                receipt_path = Path(last_child["label_receipt"]["path"])
                original_dataset_bytes = dataset_path.read_bytes()
                original_receipt_bytes = receipt_path.read_bytes()
                dataset_doc = json.loads(original_dataset_bytes)
                last_test_id = next(row["id"] for row in reversed(
                    prepared.read_json(last_child["execution_plan"]["path"])["global_manifest"]["records"])
                    if row["split"] == "test")
                test_row = next(row for row in dataset_doc["rows"] if row["id"] == last_test_id)
                test_row["features"] = {"path": "forbidden-test-feature.npz",
                    "bytes": 1, "sha256": "0" * 64}
                new_dataset_pin = write(dataset_path, dataset_doc)
                receipt = json.loads(original_receipt_bytes)
                receipt["dataset"] = new_dataset_pin
                write(receipt_path, receipt)
                last_child["dataset"] = new_dataset_pin
                last_child["label_receipt"] = prepared.pin(receipt_path)
                tampered_global = copy.deepcopy(finalized)
                next(item for item in tampered_global["child_requests"]
                    if item["shard_id"] == "group3")["request"] = last_child
                try:
                    with patch.object(original_prepared, "verify_prepared",
                                      side_effect=AssertionError("payload verifier ran before TEST metadata rejection")), \
                            patch.object(sharded, "verify_child_request",
                                      side_effect=AssertionError("child payload verifier ran before TEST metadata rejection")), \
                            patch.object(sharded, "_verify_global_stage_history",
                                      return_value=finalized["stage_evidence_history"]), \
                            patch.object(sharded, "_check_fresh_stage_evidence", return_value={}), \
                            patch.object(sharded, "verify_wrapper_sources"), \
                            patch.object(prepared, "cloud_guard"), \
                            patch.object(sharded, "quota_check", return_value={}), \
                            patch.object(prepared, "source_runtime",
                                         return_value=(adapter, working, fake_runtime)):
                        with self.assertRaises(ValueError):
                            sharded.join_labels_after_prelabel(tampered_global)
                finally:
                    dataset_path.write_bytes(original_dataset_bytes)
                    receipt_path.write_bytes(original_receipt_bytes)


if __name__ == "__main__":
    unittest.main()
