"""Model-free boundary tests; manufactured control metadata is not training data."""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import kaggle_adaptation as k
from scripts import kaggle_pilot as p

REPO = Path(__file__).resolve().parents[1]


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=REPO / ".cache")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.working = self.root / "working"
        self.working.mkdir()
        self.input = self.root / "input"
        self.input.mkdir()
        self.source = self.working / "source"
        self.source.mkdir()
        self.root18 = self.source / k.ROOT18
        (self.root18 / "bench").mkdir(parents=True)
        (self.root18 / "bench/quality_adaptation_train.py").write_text("# non-executable control\n")
        (self.root18 / "bench/runtime").mkdir()
        (self.root18 / "bench/runtime/adaptation-sources.json").write_text('{"control":"source lock"}')
        self.python = self.working / "runtime/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text("benign interpreter identity control; never executed")
        self.output = self.working / "pilot"
        self.full = self.working / "full"
        extraction = self.input / "prepared"
        extraction.mkdir()
        self.cfg = {"device": "cuda", "max_steps": 20, "max_seconds": 900, "max_checkpoint_bytes": 1000,
                    "max_output_bytes": 1536 * 1024**2}
        runtime = self.doc("runtime.json", {"runtime_root": str(self.working / "runtime")})
        config = self.doc("config.json", self.cfg)
        plan = self.doc("plan.json", {"plan_sha256": "plan", "target_requests": [{"train_states_requested": 1000}]})
        reports = []
        for role in ("input_review", "parity_report"):
            report = {"passed": True, "test_arrays_opened": False, "dataset_sha256": "dataset-bytes",
                      "config_sha256": config["sha256"], "plan_sha256": "plan",
                      "checks": [{"name": "manufactured contract check", "passed": True,
                                  "observed": 1, "expected": 1}]}
            reports.append({"role": role, "file": self.doc(role + ".json", report)})
        self.pilot = {"schema": "kaggle_train_pilot_request_v1",
            "pilot_id": "0e5d25cf-8fa7-48bb-84fa-ab7c30dd43b0", "full_run_id": "c81049b0-f2ac-4dad-beb1-57a17b5761cb",
            "adapter": self.descriptor(REPO / "scripts/kaggle_adaptation.py"),
            "pilot_output_root": str(self.output), "stop_after_steps": 2,
            "storage_limit_bytes": 20 * p.GIB, "train_states_requested": 1000,
            "limits": {"max_seconds": 30, "max_rss_bytes": p.GIB, "max_output_bytes": 2 * p.GIB,
                       "max_log_bytes": 1024 * 1024, "reserve_bytes": p.GIB},
            "training": {"schema": "kaggle_adaptation_request_v1", "mode": "prepared",
                "python": str(self.python), "working_root": str(self.working), "input_root": str(self.input),
                "source_install": str(self.source), "output_root": str(self.full), "inputs": reports,
                "config": config, "runtime": runtime, "global_inventory": plan,
                "identity_bridge": self.doc("bridge.json", {}), "crop_manifest": self.doc("crops.json", {}),
                "labels": self.doc("labels.json", {}),
                "dataset": {"path": str(extraction / "manifest.json"), "sha256": "dataset-bytes", "bytes": 1},
                "paths": {"train_dir": str(self.full / "train"), "extraction_dir": str(extraction),
                          "manifest_output": str(extraction / "manifest.json"),
                          "export_output": str(self.full / "train/head.pt")},
                "limits": {"max_seconds": 900, "max_rss_bytes": 8 * p.GIB, "max_output_bytes": 2 * p.GIB,
                           "max_log_bytes": 1024 * 1024, "reserve_bytes": p.GIB}}}
        self.runtime = {"python": k.PYTHON_VERSION, "packages": {"torch": "2.5.1+cu124"}}
        now = datetime.now(timezone.utc)
        self.evidence = {"schema": "kaggle_train_pilot_evidence_v1", "provider": "kaggle", "stage": "pilot",
            "identity_sha256": p.seal(p.pilot_identity(k, self.pilot, self.runtime)),
            "session_id": "manufactured-session", "evidence_source": "manufactured operator contract",
            "notebook_url": "https://www.kaggle.com/code/control/pilot",
            "observed_at": (now - timedelta(seconds=1)).isoformat(), "expires_at": (now + timedelta(seconds=120)).isoformat(),
            "quota": {"paid": False, "gpu_enabled": True, "gpu_remaining_seconds": 900, "session_remaining_seconds": 900},
            "gpu": {"uuid": "GPU-control", "name": "Control GPU"}, "plan_sha256": "plan",
            "parity_passed": True, "authentic_inputs_reviewed": True, "test_arrays_opened": False}
        self.pilot["evidence"] = self.doc("evidence.json", self.evidence)

    def descriptor(self, path):
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def doc(self, name, value):
        path = self.working / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return self.descriptor(path)

    def context(self):
        return {"root18": self.root18, "runtime": self.runtime,
                "identity": p.seal(p.pilot_identity(k, self.pilot, self.runtime)),
                "identity_document": p.pilot_identity(k, self.pilot, self.runtime),
                "evidence": self.evidence, "resident": 1234, "free": 10 * p.GIB, "links": {},
                "metadata": {"plan_sha256": "plan"}}

    def reconciliation(self):
        before = {"config": self.cfg, "config_sha256": k.canonical(self.cfg), "dataset_sha256": "canonical-dataset",
                  "source_lock_sha256": k.file_sha256(self.root18 / "bench/runtime/adaptation-sources.json"),
                  "versions": {"torch": "2.5.1+cu124"}}
        after = deepcopy(before)
        after["checkpoint"] = {key: before[key] for key in
                               ("config_sha256", "dataset_sha256", "source_lock_sha256", "versions")}
        after["checkpoint"].update(schema="correspondence_head_checkpoint_v1", completed=False, steps=2,
                                    epoch=0, batch_cursor=2, elapsed_seconds=.2)
        after.update(optimizer_steps=[2, 2], optimizer_state_entries=2, resume_state_present=True)
        result = {"status": "incomplete", "reason": "child_reported_incomplete", "returncode": 3,
                  "elapsed_seconds": .5, "peak_group_rss_bytes": 12345}
        status = {"status": "incomplete", "reason": "invocation_step_limit", "steps": 2, "epoch": 0,
                  "elapsed_seconds": .3, "resumed_from": None, "quality_claim": False, "test_arrays_opened": False}
        progress = {"freshness": "fresh", "last_confirmed_steps": 2}
        return result, before, after, status, progress

    def evidence_patches(self):
        self.enterContext(patch.object(p.platform, "system", return_value="Linux"))
        self.enterContext(patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}))
        self.enterContext(patch.object(Path, "is_dir", return_value=True))
        self.enterContext(patch.object(p.subprocess, "run", return_value=subprocess.CompletedProcess([], 0,
                                               "GPU-control, Control GPU, 12000, 16000\n", "")))

    def validation_patches(self):
        """Mocks model prerequisites only; never evidence of actual cloud admission."""
        real_contained = k.contained
        def mapped_contained(path, root):
            if root == Path("/kaggle/working"):
                root = self.working
            if root == Path("/kaggle/input"):
                root = self.input
            return real_contained(path, root)
        self.enterContext(patch.object(k, "contained", side_effect=mapped_contained))
        self.source_mock = self.enterContext(patch.object(k, "verify_source", return_value=self.root18))
        self.enterContext(patch.object(k, "metadata_document", side_effect=lambda d, _roots: k.read_json(Path(d["path"])) if Path(d["path"]).is_file() else {}))
        self.runtime_mock = self.enterContext(patch.object(k, "verify_runtime", return_value=self.runtime))
        self.metadata_mock = self.enterContext(patch.object(k, "run_metadata_check", return_value={"plan_sha256": "plan"}))
        self.enterContext(patch.object(k, "verify_descriptor", return_value=self.root))
        self.enterContext(patch.object(k, "verify_neural_assets"))
        self.enterContext(patch.object(k, "runtime_link_policy", return_value={}))
        self.enterContext(patch.object(k, "unique_tree_bytes", return_value=1234))
        self.enterContext(patch.object(k, "free_by_filesystem", return_value={1: 10 * p.GIB}))
        self.enterContext(patch.object(p, "validate_evidence", return_value=self.evidence))

    def test_exact_frozen_adapter_and_archive_pins(self):
        self.assertEqual(k.file_sha256(REPO / "scripts/kaggle_adaptation.py"), p.ADAPTER_SHA256)
        pins = {"larger-amortized-three-root-source-v1.zip": k.ARCHIVE_SHA256,
                "kaggle-execution-adapter-v1.zip": "3314861b6bcd8df1283f2f6332461eb655f02cc0206e0bbfb34c70d82eeb1ff8"}
        for name, expected in pins.items():
            self.assertEqual(k.file_sha256(REPO / ".cache/cloud-prepare" / name), expected)

    def test_import_and_help_without_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys; from scripts import kaggle_pilot; assert 'torch' not in sys.modules"], cwd=REPO, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", str(REPO / "scripts/kaggle_pilot.py"), "--help"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_authenticated_adapter_ignores_bytecode(self):
        module = p.load_adapter(REPO / "scripts/kaggle_adaptation.py")
        self.assertEqual(module.ROOT18, k.ROOT18)
        tampered = self.working / "adapter.py"
        tampered.write_bytes((REPO / "scripts/kaggle_adaptation.py").read_bytes() + b"\nraise RuntimeError('untrusted')")
        with self.assertRaisesRegex(p.PilotError, "SHA-256"):
            p.load_adapter(tampered)

    def test_request_byte_hash_and_duplicate_rejection(self):
        path = self.working / "request.json"
        path.write_text(json.dumps(self.pilot))
        self.assertEqual(p.read_request(path, k.file_sha256(path)), self.pilot)
        with self.assertRaisesRegex(p.PilotError, "SHA-256"):
            p.read_request(path, "0" * 64)
        path.write_text('{"schema":"kaggle_train_pilot_request_v1","schema":"kaggle_train_pilot_request_v1"}')
        with self.assertRaisesRegex(p.PilotError, "duplicate"):
            p.read_request(path, k.file_sha256(path))

    def test_invocation_argv_preserves_dataset_and_config(self):
        argv = p.trainer_argv(k, self.pilot)
        original = k.command("train", self.pilot["training"])
        expected = original[:]
        expected[expected.index("--output") + 1] = str(self.output / "train")
        self.assertEqual(argv, expected + ["--stop-after-steps", "2"])
        self.assertNotIn("--resume", argv)
        self.assertEqual(self.pilot["training"]["paths"]["train_dir"], str(self.full / "train"))

    def test_hard_caps_and_global_deadline(self):
        for key, value in (("stop_after_steps", 0), ("stop_after_steps", 5), ("stop_after_steps", True)):
            bad = deepcopy(self.pilot)
            bad[key] = value
            with self.assertRaises(ValueError):
                p.validate_caps(k, bad)
        for key, value in (("max_seconds", 901), ("max_rss_bytes", 8 * p.GIB + 1),
                           ("max_output_bytes", 2 * p.GIB + 1), ("reserve_bytes", p.GIB - 1)):
            bad = deepcopy(self.pilot)
            bad["limits"][key] = value
            with self.assertRaises(ValueError):
                p.validate_caps(k, bad)
        with patch.object(p.time, "monotonic", return_value=7):
            self.assertEqual(p.remaining_limits(self.pilot, 9)["max_seconds"], 2)
            with self.assertRaisesRegex(p.PilotError, "deadline"):
                p.remaining_limits(self.pilot, 7)

    def test_validation_reuses_original_preflight_not_full_admission(self):
        self.validation_patches()
        with patch.object(k, "validate_request", side_effect=AssertionError("full admission cannot be bypassed")):
            context = p.validate_pilot(k, self.pilot)
        self.assertEqual(context["metadata"]["plan_sha256"], "plan")
        self.metadata_mock.assert_called_once_with(self.pilot["training"], "preflight", self.root18)

    def test_source_and_runtime_reject_before_metadata_or_launch(self):
        self.validation_patches()
        for field in (self.source_mock, self.runtime_mock):
            field.side_effect = k.ContractError("identity changed")
            self.metadata_mock.reset_mock()
            with self.assertRaisesRegex(ValueError, "identity changed"):
                p.validate_pilot(k, self.pilot)
            self.metadata_mock.assert_not_called()
            field.side_effect = None

    def test_preflight_failure_stops_before_launch(self):
        self.validation_patches()
        self.metadata_mock.side_effect = k.ContractError("sealed overlap")
        with patch.object(k, "supervise") as launch:
            with self.assertRaisesRegex(ValueError, "sealed overlap"):
                p.validate_pilot(k, self.pilot)
            launch.assert_not_called()

    def test_fresh_path_and_alias_rejection(self):
        self.validation_patches()
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            p.validate_pilot(k, self.pilot)
        self.output.rmdir()
        for path in (self.full / "pilot", self.source / "pilot", self.working):
            self.pilot["pilot_output_root"] = str(path)
            with self.assertRaises(ValueError):
                p.validate_pilot(k, self.pilot)

    def test_insufficient_actual_disk_rejects(self):
        self.validation_patches()
        with patch.object(k, "free_by_filesystem", return_value={1: p.GIB}):
            with self.assertRaisesRegex(ValueError, "storage"):
                p.validate_pilot(k, self.pilot)

    def test_actual_evidence_gpu_and_detailed_reports(self):
        self.evidence_patches()
        value = p.validate_evidence(k, self.pilot, self.evidence["identity_sha256"], self.pilot["limits"])
        # Source: manufactured nvidia-smi output declares 12000 MiB free.
        self.assertEqual(value["actual_gpu_free_bytes_at_admission"], 12000 * 1024**2)
        report = k.read_json(Path(self.pilot["training"]["inputs"][0]["file"]["path"]))
        report["checks"] = []
        self.pilot["training"]["inputs"][0]["file"] = self.doc("input_review.json", report)
        with self.assertRaisesRegex(ValueError, "detailed"):
            p.validate_evidence(k, self.pilot, self.evidence["identity_sha256"], self.pilot["limits"])

    def test_stale_paid_wrong_gpu_and_missing_evidence_reject(self):
        self.evidence_patches()
        mutations = [lambda e: e.update(observed_at="2020-01-01T00:00:00+00:00"),
                     lambda e: e["quota"].update(paid=True), lambda e: e["gpu"].update(uuid="GPU-other"),
                     lambda e: e.update(parity_passed=False), lambda e: e.update(identity_sha256="wrong")]
        for mutation in mutations:
            value = deepcopy(self.evidence)
            mutation(value)
            self.pilot["evidence"] = self.doc("evidence.json", value)
            with self.assertRaises(ValueError):
                p.validate_evidence(k, self.pilot, self.evidence["identity_sha256"], self.pilot["limits"])

    def test_clean_step_limit_is_measurement_success_but_not_train_completion(self):
        args = self.reconciliation()
        self.assertEqual(p.reconcile(k, self.pilot, *args), 2)
        self.assertFalse(args[2]["checkpoint"]["completed"])

    def test_zero_negative_excess_and_optimizer_mismatch_reject(self):
        for steps in (0, -1, 3, True):
            args = self.reconciliation()
            args[2]["checkpoint"]["steps"] = steps
            with self.assertRaisesRegex(ValueError, "step delta"):
                p.reconcile(k, self.pilot, *args)
        args = self.reconciliation()
        args[2]["optimizer_steps"] = [1, 2]
        with self.assertRaises(ValueError):
            p.reconcile(k, self.pilot, *args)

    def test_checkpoint_config_dataset_source_runtime_and_status_mismatch_reject(self):
        for key in ("config_sha256", "dataset_sha256", "source_lock_sha256", "versions"):
            args = self.reconciliation()
            args[2]["checkpoint"][key] = "wrong"
            with self.assertRaisesRegex(ValueError, "identity"):
                p.reconcile(k, self.pilot, *args)
        args = self.reconciliation()
        args[1]["config_sha256"] = args[2]["config_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "canonical"):
            p.reconcile(k, self.pilot, *args)
        args = self.reconciliation()
        args[3]["steps"] = 1
        with self.assertRaises(ValueError):
            p.reconcile(k, self.pilot, *args)

    def test_timeout_crash_completed_stale_status_never_produce_measurement_success(self):
        for mutation in (lambda a: a[0].update(reason="deadline_exceeded"),
                         lambda a: a[0].update(returncode=1, status="failed"),
                         lambda a: a[0].update(returncode=0, status="completed"),
                         lambda a: a[4].update(freshness="stale"),
                         lambda a: a[3].update(reason="max_steps")):
            args = self.reconciliation()
            mutation(args)
            with self.assertRaises(ValueError):
                p.reconcile(k, self.pilot, *args)

    def test_measurements_distinguish_observation_bound_forecast(self):
        result, before, *_ = self.reconciliation()
        value = p.measurement_records(k, self.pilot, self.context(), result, before, 2, 2234, 9 * p.GIB)
        # Source: the manufactured two-step .5 s observation implies 5 s /20 steps.
        self.assertEqual(value["forecast"]["training_seconds_linear"], 5)
        self.assertEqual(value["observed"]["observed_additional_peak_bytes_lower_bound"], 1000)
        self.assertEqual(value["observed"]["enforced_additional_output_cap_bytes"], 2 * p.GIB)
        self.assertFalse(value["observed"]["peak_is_exact"])
        self.assertFalse(value["observed"]["full_training_admitted"])
        self.assertFalse(value["admission_records_emitted"])
        self.assertNotIn("scale_measurements", value)
        self.assertNotIn("stage_measurements", value)

    def test_full_gate_still_rejects_missing_measurements(self):
        # No calls/patches to the full validator are made by pilot admission.
        self.assertEqual(k.file_sha256(REPO / "scripts/kaggle_adaptation.py"), p.ADAPTER_SHA256)
        request = deepcopy(self.pilot["training"])
        request["schema"] = "kaggle_train_pilot_request_v1"
        with self.assertRaisesRegex(ValueError, "request schema"):
            k.validate_request(request, "train")

    def test_real_non_ml_process_measurement_and_timeout_cleanup(self):
        # A benign stdlib child exercises actual process supervision, not a cloud
        # success simulation. Windows reports RSS unavailable (zero), as declared.
        self.output.mkdir()
        limits = {**self.pilot["limits"], "max_seconds": 2}
        with patch.object(k, "free_by_filesystem", return_value={1: 10 * p.GIB}):
            value = k.supervise([sys.executable, "-B", "-c", "import time; print('control'); time.sleep(.15)"],
                cwd=self.root, output_root=self.output, working=self.working, log_path=self.output / "real.log",
                limits=limits, env=k.clean_environment())
            self.assertEqual(value["status"], "completed", value)
            self.assertGreater(value["elapsed_seconds"], .1)
            self.assertIn(b"control", (self.output / "real.log").read_bytes())
            value = k.supervise([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                cwd=self.root, output_root=self.output, working=self.working, log_path=self.output / "timeout.log",
                limits={**limits, "max_seconds": .1}, env=k.clean_environment())
            self.assertEqual(value["reason"], "deadline_exceeded", value)
            self.assertIsNotNone(value["returncode"])
            self.assertTrue(value["outputs_preserved"])

    def test_execute_failure_retains_result_without_measurement_fields(self):
        desc = self.doc("request.json", self.pilot)
        with patch.object(p, "load_adapter", return_value=k), patch.object(k, "verify_descriptor"), \
             patch.object(p, "validate_pilot", return_value=self.context()), \
             patch.object(p, "inspect_training", side_effect=p.PilotError("manufactured inspection failure")):
            value = p.execute(Path(desc["path"]), desc["sha256"])
        self.assertEqual(value["status"], "failed")
        self.assertFalse(value["measurement_complete"])
        self.assertNotIn("scale_measurements", value)
        self.assertEqual(k.read_json(self.output / "pilot-result.json"), value)
        self.assertFalse((self.output / ".adapter-lock").exists())

    def test_execute_mocked_child_integrates_status_checkpoint_and_measurements(self):
        # Manufactured child interface only. This does not claim a cloud/GPU run.
        if sys.platform != "linux":
            self.enterContext(patch.object(p, "execution_deadline", side_effect=lambda _deadline: nullcontext()))
        result, before, after, status, _ = self.reconciliation()
        def child(argv, **kwargs):
            self.assertEqual(argv[-2:], ["--stop-after-steps", "2"])
            train = self.output / "train"
            train.mkdir()
            (train / "last.pt").write_bytes(b"opaque checkpoint interface control; no tensors")
            (train / "status.json").write_text(json.dumps(status))
            after["checkpoint_sha256"] = k.file_sha256(train / "last.pt")
            return result
        desc = self.doc("request.json", self.pilot)
        with patch.object(p, "load_adapter", return_value=k), patch.object(k, "verify_descriptor"), \
             patch.object(p, "validate_pilot", return_value=self.context()), \
             patch.object(p, "validate_pilot_for_launch", side_effect=lambda _k, _p, c: c), \
             patch.object(p, "inspect_training", side_effect=[before, after]), \
             patch.object(k, "supervise", side_effect=child):
            value = p.execute(Path(desc["path"]), desc["sha256"])
        self.assertEqual(value["status"], "measurement_complete", value)
        self.assertTrue(value["measurement_complete"])
        self.assertEqual(value["trainer_status"], "incomplete")
        self.assertEqual(value["optimizer_step_delta"], 2)
        self.assertFalse(value["full_training_admitted"])
        self.assertEqual(value["checkpoint"]["sha256"], after["checkpoint_sha256"])
        self.assertIn("observed", value)
        self.assertNotIn("scale_measurements", value)
        self.assertFalse((self.output / ".adapter-lock").exists())

    def test_original_scale_consumer_accepts_record_and_still_blocks_bad_projection(self):
        from bench import quality_adaptation_scale as scale
        from bench.test_quality_adaptation_scale import hash_text
        plan, rows, context, before, extraction = self.export_fixture()
        result, *_ = self.reconciliation()
        value = p.measurement_records(k, self.pilot, context, result, before, 2, 2234, 9 * p.GIB)
        self.assertTrue(value["admission_records_emitted"], value)
        record = value["scale_measurements"]
        self.assertEqual(record["measured_additional_peak_bytes"], self.pilot["training"]["limits"]["max_output_bytes"])
        self.assertEqual(value["stage_measurements"]["additional_peak_bytes"], record["measured_additional_peak_bytes"])
        self.assertFalse(value["full_run_allowance"]["is_measured_peak"])
        report = scale.assess_admission(plan, rows, measurements=record, output_root=extraction,
                    producer_sha256=hash_text("producer"), config_sha256=hash_text("config"))
        self.assertTrue(report["admitted"], report)
        record["training_seconds"] = 10000
        report = scale.assess_admission(plan, rows, measurements=record, output_root=extraction,
                    producer_sha256=hash_text("producer"), config_sha256=hash_text("config"))
        self.assertFalse(report["admitted"])
        self.assertIn("measured_time_projection_exceeds_limit", report["blockers"])

    def export_fixture(self):
        """Tiny genuine filesystem/legacy-consumer controls; no model or cloud."""
        from bench import quality_adaptation_scale as scale
        from bench.test_quality_adaptation_scale import inventory, receipts
        plan = scale.plan_scale(inventory(), shard_size=2, targets=[1, 2])
        extraction = self.full / "prepared"
        extraction.mkdir(parents=True)
        rows = receipts(plan, extraction)
        r = self.pilot["training"]
        r["mode"] = "extract"
        r["paths"]["extraction_dir"] = str(extraction)
        r["paths"]["manifest_output"] = str(extraction / "manifest.json")
        dataset = {"schema": "manufactured identity-only dataset control", "rows": []}
        (extraction / "manifest.json").write_text(json.dumps(dataset))
        r["dataset"] = self.descriptor(extraction / "manifest.json")
        r["global_inventory"] = self.doc("plan.json", plan)
        self.pilot["train_states_requested"] = 1
        self.evidence["quota"].update(storage_limit_bytes=self.pilot["storage_limit_bytes"], storage_scope=str(self.working))
        context = {**self.context(), "metadata": {"plan_sha256": plan["plan_sha256"]}}
        _, before, *_ = self.reconciliation()
        before["dataset_sha256"] = k.canonical(dataset)
        self.enterContext(patch.object(k, "verify_source", return_value=self.root18))
        self.enterContext(patch.object(k, "run_metadata_check", return_value=context["metadata"]))
        self.enterContext(patch.object(p, "validate_evidence", return_value=self.evidence))
        self.enterContext(patch.object(k, "runtime_link_policy", return_value={}))
        self.enterContext(patch.object(k, "free_by_filesystem", return_value={1: 9 * p.GIB}))
        if sys.platform != "linux":
            # Windows control fixtures test storage/identity only. The actual
            # interrupting deadline is exercised by the real Linux test below.
            self.enterContext(patch.object(p, "execution_deadline", side_effect=lambda _deadline: nullcontext()))
        return plan, rows, context, before, extraction

    def test_low_headroom_withholds_records_and_original_consumer_cannot_admit(self):
        from bench import quality_adaptation_scale as scale
        from bench.test_quality_adaptation_scale import hash_text
        plan, rows, context, before, extraction = self.export_fixture()
        result, *_ = self.reconciliation()
        low_free = p.GIB + 64 * 1024**2
        with patch.object(k, "free_by_filesystem", return_value={1: low_free}):
            value = p.measurement_records(k, self.pilot, context, result, before, 2, 64 * 1024**2, low_free)
        self.assertFalse(value["admission_records_emitted"])
        self.assertIn("allowance", value["admission_records_withheld_reason"])
        self.assertIn("observed", value)
        self.assertNotIn("scale_measurements", value)
        self.assertNotIn("stage_measurements", value)
        report = scale.assess_admission(plan, rows, measurements=value.get("scale_measurements"), output_root=extraction,
                   producer_sha256=hash_text("producer"), config_sha256=hash_text("config"))
        self.assertFalse(report["admitted"])
        self.assertIn("measured_time_storage_and_limits_required", report["blockers"])
        # Even copying the valid conservative record into the original consumer
        # with low current headroom must block; provenance is irrelevant to fit.
        record = p.measurement_records(k, self.pilot, context, result, before, 2, 2234, 9 * p.GIB)["scale_measurements"]
        record["actual_free_bytes"] = low_free
        report = scale.assess_admission(plan, rows, measurements=record, output_root=extraction,
                   producer_sha256=hash_text("producer"), config_sha256=hash_text("config"))
        self.assertFalse(report["admitted"])
        self.assertIn("additional_storage_exceeds_free_headroom_or_reserve", report["blockers"])

    def test_exporter_withholds_missing_storage_scope_stale_quota_and_changed_identity(self):
        _, _, context, before, extraction = self.export_fixture()
        result, *_ = self.reconciliation()
        def records():
            return p.measurement_records(k, self.pilot, context, result, before, 2, 2234, 9 * p.GIB)
        del self.evidence["quota"]["storage_scope"]
        self.assertFalse(records()["admission_records_emitted"])
        self.evidence["quota"]["storage_scope"] = str(self.working)
        self.evidence["quota"]["gpu_remaining_seconds"] = 31
        self.assertFalse(records()["admission_records_emitted"])
        self.evidence["quota"]["gpu_remaining_seconds"] = 900
        with patch.object(p, "validate_evidence", side_effect=p.PilotError("expired")):
            self.assertFalse(records()["admission_records_emitted"])
        before["dataset_sha256"] = "different measured dataset"
        self.assertFalse(records()["admission_records_emitted"])
        before["dataset_sha256"] = k.canonical(k.read_json(extraction / "manifest.json"))
        before["config"]["max_steps"] += 1
        self.assertFalse(records()["admission_records_emitted"])

    def test_exporter_withholds_readonly_input_scope_and_insufficient_full_output_envelope(self):
        _, _, context, before, _ = self.export_fixture()
        result, *_ = self.reconciliation()
        self.pilot["training"]["paths"]["extraction_dir"] = str(self.input / "prepared")
        context["identity"] = p.seal(p.pilot_identity(k, self.pilot, self.runtime))
        value = p.measurement_records(k, self.pilot, context, result, before, 2, 2234, 9 * p.GIB)
        self.assertFalse(value["admission_records_emitted"])
        self.assertIn("quota scope", value["admission_records_withheld_reason"])
        self.pilot["training"]["paths"]["extraction_dir"] = str(self.full / "prepared")
        self.pilot["training"]["limits"]["max_output_bytes"] = before["config"]["max_output_bytes"]
        context["identity"] = p.seal(p.pilot_identity(k, self.pilot, self.runtime))
        value = p.measurement_records(k, self.pilot, context, result, before, 2, 2234, 9 * p.GIB)
        self.assertFalse(value["admission_records_emitted"])
        self.assertIn("envelope", value["admission_records_withheld_reason"])

    def test_optimizer_fractional_step_cannot_be_truncated_to_success(self):
        args = self.reconciliation()
        args[2]["optimizer_steps"] = [2.5, 2]
        with self.assertRaises(ValueError):
            p.reconcile(k, self.pilot, *args)
        args[2]["optimizer_steps"] = [2.0, 2.0]  # Actual AdamW scalar tensor representation.
        self.assertEqual(p.reconcile(k, self.pilot, *args), 2)

    def test_expired_export_keeps_observations_and_emits_no_records(self):
        result, before, *_ = self.reconciliation()
        with patch.object(p, "_full_run_allowance", side_effect=AssertionError("expired export did work")):
            value = p.measurement_records(k, self.pilot, self.context(), result, before,
                                         2, 2234, 9 * p.GIB, deadline=time.monotonic() - 1)
        self.assertFalse(value["admission_records_emitted"])
        self.assertIn("deadline", value["admission_records_withheld_reason"])
        self.assertIn("observed", value)
        self.assertNotIn("scale_measurements", value)
        self.assertNotIn("stage_measurements", value)

    @unittest.skipUnless(sys.platform == "linux", "actual Linux SIGALRM deadline; run this control under WSL")
    def test_real_linux_late_export_deadline_interrupts_and_reaps_non_ml_child(self):
        result, before, *_ = self.reconciliation()
        marker = self.working / "late-child-finished.txt"
        children = []
        real_popen = subprocess.Popen
        def child_spy(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        def delayed_export(*_args):
            # This actual benign child represents the frozen metadata child.
            # subprocess.run must kill/reap it when SIGALRM interrupts the call.
            subprocess.run([sys.executable, "-B", "-c",
                            "import pathlib,sys,time; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('finished')",
                            str(marker)], check=True, capture_output=True, timeout=120)
            raise AssertionError("late metadata child survived the shared deadline")
        previous_handler = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        with patch.object(p, "_full_run_allowance", side_effect=delayed_export), \
             patch.object(subprocess, "Popen", side_effect=child_spy):
            value = p.measurement_records(k, self.pilot, self.context(), result, before,
                                         2, 2234, 9 * p.GIB, deadline=started + .05)
        self.assertLess(time.monotonic() - started, 1, "unbounded metadata child consumed its 2s body")
        self.assertFalse(value["admission_records_emitted"])
        self.assertIn("deadline", value["admission_records_withheld_reason"])
        self.assertIn("observed", value)
        self.assertNotIn("scale_measurements", value)
        self.assertNotIn("stage_measurements", value)
        self.assertFalse(marker.exists())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode, "interrupted child must be reaped")
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)
        # Also interrupt in-process hashing/filesystem-style work, rather than
        # merely checking expiry after a blocking call has finished.
        with patch.object(p, "_full_run_allowance", side_effect=lambda *_args: time.sleep(2)):
            started = time.monotonic()
            value = p.measurement_records(k, self.pilot, self.context(), result, before,
                                         2, 2234, 9 * p.GIB, deadline=started + .05)
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(value["admission_records_emitted"])
        # A normal hash and a quick real stdlib child fit the SAME mechanism.
        with p.execution_deadline(time.monotonic() + 2):
            checksum = k.file_sha256(self.root18 / "bench/quality_adaptation_train.py")
            quick = subprocess.run([sys.executable, "-B", "-c", "print('quick control')"],
                                   check=True, capture_output=True, text=True, timeout=120)
        self.assertEqual(len(checksum), 64)
        self.assertEqual(quick.stdout.strip(), "quick control")
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)


if __name__ == "__main__":
    unittest.main()
