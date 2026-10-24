from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import json

from bench import quality_kaggle_activation as activation


ROOT = Path(__file__).resolve().parents[1]


class KaggleActivationHelperTests(unittest.TestCase):
    def test_actual_torch_uuid_object_and_nvidia_prefix_bind_same_physical_device(self):
        actual_uuid = "da4c3627-d207-b41e-a17e-8ef9d23076c3"
        class CUuuid:
            def __str__(self):
                return actual_uuid
        properties = SimpleNamespace(uuid=CUuuid())
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            current_device=lambda: 0, get_device_properties=lambda device: properties,
            get_device_name=lambda device: "Tesla T4")
        torch = SimpleNamespace(cuda=cuda, are_deterministic_algorithms_enabled=lambda: True,
            get_num_threads=lambda: 1, backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=False)))
        provenance = {"cuda_device_identity": {"name": "Tesla T4", "uuid": "GPU-" + actual_uuid}}
        self.assertEqual(activation._cuda_identity(torch, provenance)["uuid"], actual_uuid)
        properties.uuid = None
        with patch.object(activation.subprocess, "run", return_value=SimpleNamespace(stdout="GPU-" + actual_uuid + "\n")) as smi:
            self.assertEqual(activation._cuda_identity(torch, provenance)["uuid"], actual_uuid)
            smi.assert_called_once()
        properties.uuid = "malformed"
        with self.assertRaisesRegex(activation.ActivationError, "stable physical UUID"):
            activation._cuda_identity(torch, provenance)
        properties.uuid = CUuuid()
        provenance["cuda_device_identity"]["uuid"] = "00000000-0000-0000-0000-000000000000"
        with self.assertRaisesRegex(activation.ActivationError, "physical identity"):
            activation._cuda_identity(torch, provenance)

    def test_last_plan_and_sink_identity_fail_before_any_input_payload_reader(self):
        from bench.quality_adaptation_refinement_states import RefStateCapture
        for poison in ("plan_last", "sink_last", "none"):
            with self.subTest(poison=poison), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                root = Path(temporary)
                physical = [{"id": f"fixture-view{v}", "object_id": "1", "view_index": v,
                             "split": "train", "sequence_id": f"sequence-{v}",
                             "capture_group": f"group-{v}", "frame_id": 0,
                             "timestamp_s": 0., "iterations": [0]} for v in range(2)]
                rows = [{"physical": dict(p), "input": None, "video": None, "mesh": None,
                         "bank": None, "mask": None, "mask_index": None,
                         "mask_status": "unavailable", "mask_failure": "fixture"} for p in physical]
                context = activation.ActivationContext(root / "request.json", b"{}", "a" * 64)
                branch = root / "branches" / "legacy-v1"
                doc = {"output_root": str(root), "source_files": {},
                       "limits": {"storage_bytes": 18 * 1024**3, "reserve_bytes": 1024**3}}
                (root / "activation-plan.json").write_bytes(b"{}")
                plan = {"schema": activation.OBSERVER_PLAN_SCHEMA, "request_sha256": context.sha256,
                        "activation_plan_sha256": hashlib.sha256(b"{}").hexdigest(),
                        "source_files_sha256": hashlib.sha256(activation.canonical_bytes({})).hexdigest(),
                        "variant": "legacy-v1", "branch_root": branch.as_posix(),
                        "capture_root": (branch / "capture").as_posix(), "rows": rows,
                        "test_payload_reads": 0, "annotation_inputs": []}
                sink = RefStateCapture.__new__(RefStateCapture)
                sink.rows = activation._physical_state_rows(rows)
                sink.physical = activation._activation_physical(physical[0], {"status": "unavailable"},
                    {"status": "unavailable"}, {"status": "unavailable", "failure_reason": "fixture"})
                sink.output = branch / "capture"
                sink.max_output_bytes = doc["limits"]["storage_bytes"]
                sink.reserve_bytes = doc["limits"]["reserve_bytes"]
                sink.resource_check = lambda: None
                if poison == "plan_last":
                    rows[-1]["physical"]["timestamp_s"] = .1
                elif poison == "sink_last":
                    sink.rows[-1]["timestamp_s"] = .1
                raw = activation.canonical_bytes(plan)
                with patch.object(activation, "_frozen", return_value=(doc, {}, physical)), \
                        patch.object(activation, "verify_sources"), \
                        patch.object(activation, "read_metadata", return_value=(raw, plan)), \
                        patch.object(activation, "_read_render_index", side_effect=AssertionError("payload reader reached")) as read, \
                        patch.object(activation, "_read_banks", side_effect=AssertionError("bank payload reached")), \
                        patch.object(activation, "_manifest_for", side_effect=AssertionError("manifest payload reached")):
                    if poison == "none":
                        with self.assertRaisesRegex(AssertionError, "payload reader reached"):
                            activation.observer_preflight(raw, hashlib.sha256(raw).hexdigest(), sink, context=context)
                        read.assert_called_once_with(context)
                    else:
                        with self.assertRaisesRegex(ValueError, "identity/order"):
                            activation.observer_preflight(raw, hashlib.sha256(raw).hexdigest(), sink, context=context)
                        read.assert_not_called()

    def test_callback_replacement_is_rejected_against_original_witness(self):
        from bench import quality_adaptation_larger as larger
        from bench.quality_adaptation_refinement_states import RefStateCapture
        raw = b"{}"; digest = hashlib.sha256(raw).hexdigest()
        context = activation.ActivationContext(ROOT / "fixture-request.json", raw, digest)
        sink = RefStateCapture.__new__(RefStateCapture)
        sink.resource_check = lambda: None
        witness = activation._ActivationWitness(activation._ACTIVATION_ISSUER,
            digest, digest, "legacy-v1", "branch", "capture", sink,
            sink.resource_check, "physical", (), (), "bindings")
        cap = larger.LargerObserverCapability(activation._ACTIVATION_ISSUER,
            raw, digest, sink, activation.POLICY, context, witness)
        # Exact capability/sink/context types; only disk preflight is a declared double.
        with patch.object(activation, "observer_preflight", side_effect=lambda *args, **kwargs:
                          replace(witness, resource_callback=sink.resource_check)):
            cap.validate(sink, larger.inference.PROFILE)
            sink.resource_check = lambda: None
            with self.assertRaisesRegex(ValueError, "witness changed"):
                cap.validate(sink, larger.inference.PROFILE)

    def test_real_stage_receipts_use_existing_neighbor_cleanup(self):
        from bench import quality_adaptation_larger as larger
        from bench import quality_neighbors as neighbors
        for fails in (False, True):
            with self.subTest(cleanup_failure=fails), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                root = Path(temporary)
                context = activation.ActivationContext(root / "request.json", b"{}", "f" * 64)
                doc = {"output_root": str(root), "scratch_root": str(root / "scratch"),
                       "limits": {"wall_seconds": 100, "rss_bytes": 1024**3}}
                client = Mock()
                if fails:
                    client.close.side_effect = RuntimeError("fixture cleanup failed")
                original_relative, original_is_dir = Path.is_relative_to, Path.is_dir
                def admitted(path, other):
                    return True if str(other) in ("/kaggle/working", "/tmp") else original_relative(path, other)
                def is_dir(path):
                    return True if path == Path("/kaggle/input") else original_is_dir(path)
                # Declared Linux/provider/signal/watchdog doubles; no model or
                # device work. Actual stage entry, marker/receipt IO and existing
                # neighbor cleanup execute, including the failure path.
                with ExitStack() as stack:
                    patches = [
                        patch.object(activation, "_frozen", return_value=(doc, {}, [])),
                        patch.object(activation.ActivationContext, "document", return_value=doc),
                        patch.object(activation.platform, "system", return_value="Linux"),
                        patch.object(Path, "is_relative_to", admitted), patch.object(Path, "is_dir", is_dir),
                        patch.dict(activation.os.environ, {"VISUALIZEIT_CLOUD": "1"}),
                        patch.object(larger, "stage_lock", side_effect=lambda: nullcontext()),
                        patch.object(activation, "resource_check"), patch.object(activation, "verify_sources"),
                        patch.object(activation, "_verify_neural_environment"),
                        patch.object(activation, "_stage_elapsed", return_value=0),
                        patch.object(activation.signal, "SIGALRM", 14, create=True),
                        patch.object(activation.signal, "ITIMER_REAL", 0, create=True),
                        patch.object(activation.signal, "getitimer", return_value=(0., 0.), create=True),
                        patch.object(activation.signal, "setitimer", create=True),
                        patch.object(activation.signal, "getsignal"), patch.object(activation.signal, "signal"),
                        patch.object(activation.threading, "Thread", return_value=Mock()),
                        patch.object(neighbors, "_client", client),
                    ]
                    for item in patches:
                        stack.enter_context(item)
                    def execute():
                        with activation.stage(context, "render", cloud_execution=True):
                            activation._require_stage(context, "render")
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                            execute()
                    else:
                        execute()
                        self.assertIsNone(neighbors._client)
                    client.close.assert_called_once_with()
                    marker, terminal = activation._stage_paths(root, "render")
                    record = json.loads(terminal.read_text())
                    self.assertTrue(marker.is_file())
                    self.assertEqual(record["status"], "failed" if fails else "completed")
                    self.assertEqual(record["cleanup_success"], not fails)
                    self.assertEqual(bool(record["cleanup_errors"]), fails)

    def test_mask_receipt_names_the_executed_helper_as_producer(self):
        import numpy as np
        from bench import vision
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            context = SimpleNamespace(sha256="a" * 64)
            physical = {"id": "fixture-view0", "view_index": 0}
            rendered = {key: {"path": "fixture", "bytes": 0, "sha256": "b" * 64}
                        for key in ("input", "video", "mesh")}
            bank = {"cnos_bank": {}, "pose_bank": {}}
            def write_fixture(path, pixels):
                Path(path).write_bytes(b"declared software fixture, not an observed image")
                return True
            with patch.object(activation, "resource_check"), \
                    patch.object(vision, "cv2", SimpleNamespace(imwrite=write_fixture)):
                value = activation._write_observed_mask(context, root, physical,
                    rendered, bank, "legacy-v1", np.ones((480, 640), np.bool_), {}, [0, 0, 1, 1])
            record = json.loads(Path(value["mask_index"]["path"]).read_text())
            self.assertEqual(record["producer_source"], "bench/quality_kaggle_activation.py")
            self.assertEqual(record["producer_source_sha256"],
                hashlib.sha256((ROOT / record["producer_source"]).read_bytes()).hexdigest())

    def test_known_numeric_string_object_inventory_preserves_exact_roles(self):
        roles = {obj: "train" for obj in activation.EXPECTED_TRAIN_OBJECTS}
        roles.update({"4": "validation", "7": "test"})
        # Same 15-object/360-view metadata inventory as the full seed-521 recipe;
        # lexical order puts object 6 after 33. No fixture payload is opened.
        rows = [{"object_id": obj, "split": roles[obj], "view_index": view,
                 "iterations": [0], "id": f"{obj}:view{view:02d}"}
                for obj in sorted(roles) for view in range(24)]
        spec = {"assets": list(roles), "seed": 521, "iterations": [0]}
        plan = {"global_manifest": {"split_manifest": {"groups": []}}}
        inventory = activation._check_split(spec, plan, rows)
        self.assertEqual(inventory["physical_rows"], 360)
        self.assertEqual(inventory["train_objects"], list(activation.EXPECTED_TRAIN_OBJECTS))
        changed = [dict(row, object_id="9") if row["object_id"] == "6" else row for row in rows]
        with self.assertRaisesRegex(ValueError, "prospective split"):
            activation._check_split(spec, plan, changed)

    def test_stored_hwc_query_and_template_reconstruct_through_actual_converter(self):
        import numpy as np
        size = 280
        crops = {"query_rgb": np.full((size, size, 3), .25, np.float32),
                 "template_rgb": np.full((size, size, 3), .75, np.float32)}
        pose = np.eye(4); pose[2, 3] = 1000.
        geometry = {"rendered_depth_mm": np.full((size, size), 1000., np.float32),
                    "rendered_mask": np.ones((size, size), np.bool_),
                    "crop_k": np.array([[200., 0., 139.5], [0., 200., 139.5], [0., 0., 1.]]),
                    "current_crop_camera_from_object_mm": pose}
        result = activation._reconstruct_captured_crops(crops, geometry)
        np.testing.assert_array_equal(result["query_rgb"], crops["query_rgb"])
        np.testing.assert_array_equal(result["template_rgb"], crops["template_rgb"])
        # Analytic pinhole ray at (140,140): (0.5/200,0.5/200,1),
        # depth 1000 mm, then subtract the estimated camera translation.
        np.testing.assert_allclose(result["template_xyz_m"][140, 140], [.0025, .0025, 0.], atol=1e-8)

    def test_frozen_sam_inference_checks_only_its_actual_runtime_dependencies(self):
        from bench import quality_adaptation_runtime as runtime
        with patch.object(runtime, "runtime_versions", return_value={"torch": "2.5.1+cu124"}) as base, \
                patch("importlib.metadata.version", side_effect=lambda name: {"hydra-core": "1.3.2", "omegaconf": "2.3.0"}[name]):
            versions = activation._inference_runtime_versions()
            base.assert_called_once_with(sam=False)
        self.assertEqual(versions["hydra-core"], "1.3.2")
        with patch.object(runtime, "runtime_versions", return_value={}), \
                patch("importlib.metadata.version", return_value="0.0.0"):
            with self.assertRaisesRegex(activation.ActivationError, "inference dependency"):
                activation._inference_runtime_versions()

    def test_import_is_model_free_in_fresh_interpreter(self):
        code = (
            "import sys; import bench.quality_kaggle_activation; "
            "assert 'torch' not in sys.modules; "
            "assert 'sam2' not in sys.modules"
        )
        completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
            check=False, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_strict_metadata_rejects_duplicate_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            activation.strict_json_bytes(b'{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            activation.strict_json_bytes(b'{"a":NaN}')

    def test_larger_dispatch_rejects_spoofed_activation_capability(self):
        from bench import quality_adaptation_larger as larger

        cap = larger.LargerObserverCapability(object(), b"{}",
            hashlib.sha256(b"{}").hexdigest(), object(), activation.POLICY,
            object(), object())
        with self.assertRaisesRegex(ValueError, "invalid Kaggle activation"):
            cap.validate(object(), larger.inference.PROFILE)

    def test_row_identity_binds_final_view_and_timestamp(self):
        first = {"id": "one", "object_id": "1", "split": "train",
            "sequence_id": "seq-a", "capture_group": "group-a", "frame_id": 0,
            "timestamp_s": 0.0, "iteration": 0}
        changed_timestamp = dict(first, timestamp_s=0.1)
        changed_view = dict(first, id="two", sequence_id="seq-b", capture_group="group-b")
        self.assertNotEqual(activation._row_identity(first), activation._row_identity(changed_timestamp))
        self.assertNotEqual(activation._row_identity(first), activation._row_identity(changed_view))

    def test_storage_accounting_counts_regular_files(self):
        with tempfile.TemporaryDirectory(prefix=".activation-test-", dir=ROOT) as temporary:
            root = Path(temporary)
            payload = root / "payload.bin"
            payload.write_bytes(b"actual-fixture")
            size, files = activation._lstat_tree_bytes([root])
            self.assertEqual((size, files), (len(b"actual-fixture"), 1))

    def test_detector_evidence_requires_finite_scores_and_shared_bank(self):
        actual = {"scores": [0.5, 0.4], "selected_proposal": 0,
            "reason": None, "bank_sha256": "a" * 64}
        self.assertEqual(activation._detector_record(actual, None, "a" * 64), actual)
        with self.assertRaisesRegex(activation.ActivationError, "bank"):
            activation._detector_record(actual, None, "b" * 64)
        malformed = dict(actual, scores=[0.5, float("nan")])
        with self.assertRaisesRegex(activation.ActivationError, "nonfinite"):
            activation._detector_record(malformed, None, "a" * 64)


if __name__ == "__main__":
    unittest.main()
