"""Object-ID ingress tests with bounded JSON and no actual model/asset reads."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_neural as neural


SHA = "a" * 64
INVALID_IDS = (True, False, 1., 1.9, .9, "3", "0", -1, 2**31, None, [], {})


def bundle_document(obj_id):
    return {"schema": "workbench_metric_bundle_v1", "family": "bounded-test",
            "glb_path": "never-read.glb", "glb_sha256": SHA, "obj_id": obj_id,
            "principal_point_origin": "integer_centers", "seed_pose_m": None,
            "seed_label": None, "seed_frame_id": None, "metric_provenance": "measured_metric_cad"}


def case_document(obj_id):
    return {"case_id": "bounded-test", "family": "bounded-test", "obj_id": obj_id,
            "rgb_path": "never-read-rgb.png", "rgb_sha256": SHA,
            "mask_path": "never-read-mask.png", "mask_sha256": SHA,
            "segmentation": {"producer": "manufactured-test-only", "manifest_sha256": SHA},
            "K": np.eye(3).ravel().tolist(), "principal_point_origin": "integer_centers",
            "initial_pose_m": np.eye(4).tolist(), "initialization_source": "onboarding:manufactured",
            "glb_path": "never-read.glb", "glb_sha256": SHA, "metric_provenance": "measured_metric_cad"}


@contextmanager
def contract_prerequisites(module):
    """Stub unrelated prerequisites, never authenticate actual weights or mint admission."""
    with ExitStack() as stack:
        for name, replacement in (
                ("verify_primary_record", lambda root: {"record_sha256": SHA}),
                ("_authenticated_bytes", lambda *args, **kwargs: None),
                ("_regular", lambda path, *args: path), ("sha256_file", lambda path: SHA),
                ("glb_inventory", lambda path: {"sha256": SHA}),
                ("source_closure", lambda *args, **kwargs: {"fixture_only": True}),
                ("runtime_packages", lambda: {"fixture_only": True}),
                ("_render_factory", lambda **kwargs: {"fixture_only": True})):
            stack.enter_context(patch.object(module, name, replacement))
        stack.enter_context(patch.object(Path, "is_file", return_value=True))
        stack.enter_context(patch.object(Path, "stat", return_value=SimpleNamespace(
            st_size=module.PRIMARY["base_checkpoint_bytes"])))
        yield


def prepare(module, cases):
    return module.prepare_workbench_contract(cases, head_path=Path("never-read-head.pt"),
        reference_path=Path("never-read-reference.pt"), verify_base=False, include_upstream=False)


@contextmanager
def accepted_parent():
    revision = "6b9c4571d9f9f518848a28443d4be4c22b4315b6"
    raw = subprocess.check_output(["git", "show", revision + ":bench/workbench_neural.py"],
                                  cwd=Path(__file__).resolve().parents[1])
    if hashlib.sha256(raw).hexdigest() != "d0df83d2e9045ffc6b6076e7c9910bc2da353512c7b5a820090aabeea34e541f":
        raise AssertionError("parent neural source differs from pinned Git blob")
    name = "bench._object_identity_parent"
    module = ModuleType(name)
    module.__package__, module.__file__ = "bench", neural.__file__
    with patch.dict(sys.modules, {name: module}):
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
        yield module


class ObjectIdentityTests(unittest.TestCase):
    def load_bundle(self, module, document):
        raw = json.dumps(document).encode()
        self.assertLess(len(raw), 2048)
        with patch.object(Path, "read_bytes", return_value=raw), \
                patch.object(module, "glb_inventory", return_value={"sha256": SHA}):
            return module.MetricBundle.from_file(Path("never-read-bundle.json"))

    def test_bundle_rejects_noninteger_and_out_of_range_json_before_glb_read(self):
        for value in INVALID_IDS:
            with self.subTest(obj_id=value), patch.object(Path, "read_bytes", return_value=json.dumps(
                    bundle_document(value)).encode()), patch.object(neural, "glb_inventory") as glb:
                with self.assertRaises(neural.wc.ContractError):
                    neural.MetricBundle.from_file(Path("never-read-bundle.json"))
                glb.assert_not_called()

    def test_contract_rejects_noninteger_and_out_of_range_before_model_authentication(self):
        for value in (*INVALID_IDS, np.int64(1), np.bool_(True)):
            with self.subTest(obj_id=repr(value)), patch.object(neural, "verify_primary_record") as verify, \
                    patch.object(neural, "_authenticated_bytes") as weights, \
                    patch.object(neural, "glb_inventory") as glb:
                with self.assertRaises(neural.wc.ContractError):
                    prepare(neural, [case_document(value)])
                verify.assert_not_called()
                weights.assert_not_called()
                glb.assert_not_called()

    def test_later_malformed_case_also_blocks_all_downstream_authentication(self):
        with patch.object(neural, "verify_primary_record") as verify, \
                patch.object(neural, "_authenticated_bytes") as weights:
            with self.assertRaises(neural.wc.ContractError):
                prepare(neural, [case_document(1), case_document(True)])
            verify.assert_not_called()
            weights.assert_not_called()

    def test_valid_json_integer_bundle_outputs_match_actual_parent(self):
        with accepted_parent() as parent:
            for value in (0, 1, 2, 4, 18, 2**31 - 1):
                with self.subTest(obj_id=value):
                    before = self.load_bundle(parent, bundle_document(value))
                    after = self.load_bundle(neural, bundle_document(value))
                    self.assertEqual(before.__dict__, after.__dict__)
                    self.assertIs(type(after.obj_id), int)
                    self.assertEqual(after.obj_id, value)

    def test_valid_contract_bodies_and_canonical_hash_match_actual_parent(self):
        with accepted_parent() as parent:
            for value in (0, 1, 2, 4, 18, 2**31 - 1):
                cases = [case_document(value)]
                snapshot = json.dumps(cases, sort_keys=True)
                with self.subTest(obj_id=value):
                    with contract_prerequisites(parent):
                        before = prepare(parent, cases)
                    with contract_prerequisites(neural):
                        after = prepare(neural, cases)
                    self.assertEqual(before.body, after.body)
                    self.assertEqual(before.sha256, after.sha256)
                    self.assertEqual(before.to_dict(), after.to_dict())
                    self.assertEqual(json.dumps(cases, sort_keys=True), snapshot)

    def test_actual_parent_coercion_bug_is_reproduced_at_both_ingresses(self):
        with accepted_parent() as parent:
            for value, coerced in ((True, 1), (False, 0), (1.9, 1), (.9, 0), ("3", 3), (-1, -1), (2**31, 2**31)):
                with self.subTest(original=value):
                    self.assertEqual(self.load_bundle(parent, bundle_document(value)).obj_id, coerced)
                    with contract_prerequisites(parent):
                        self.assertEqual(prepare(parent, [case_document(value)]).body["cases"][0]["obj_id"], coerced)

    def test_malformed_bundle_json_and_missing_id_fail_without_glb_access(self):
        missing = bundle_document(1)
        missing.pop("obj_id")
        for raw in (b"{broken", b'{"obj_id":1,"obj_id":2}', json.dumps(missing).encode()):
            with self.subTest(raw=raw), patch.object(Path, "read_bytes", return_value=raw), \
                    patch.object(neural, "glb_inventory") as glb:
                with self.assertRaises(ValueError):
                    neural.MetricBundle.from_file(Path("never-read-bundle.json"))
                glb.assert_not_called()


if __name__ == "__main__":
    unittest.main()
