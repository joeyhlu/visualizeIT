"""Fresh-machine portability tests: tiny authenticated files, no Torch/model runs.

Fixture pins are patched only inside tests. Production constants and installer
are never changed by the opt-in path.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from bench import workbench_contract as wc
from bench import workbench_neural as wn
from bench import workbench_cli as cli


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reseal(proof):
    value = copy.deepcopy(proof)
    value.pop("proof_sha256", None)
    return {**value, "proof_sha256": wc.canonical_sha256(value)}


class FreshRecordedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.pins = dict(wn.PRIMARY)
        self.paths = {}
        for name, relative, size_key, hash_key in (
            ("head", "head.pt", "export_bytes", "export_sha256"),
            ("reference", "best.pt", "reference_checkpoint_bytes", "reference_checkpoint_sha256"),
            ("base", wn.BASE_CHECKPOINT_RELATIVE, "base_checkpoint_bytes", "base_checkpoint_sha256"),
            ("installer", "bench/quality_gotrack_inference.py", None, "installer_sha256"),
            ("training_lock", "bench/runtime/adaptation-sources.json", None, "training_lock_sha256"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (name + "-tiny-authenticated-test-fixture").encode()
            path.write_bytes(raw)
            self.paths[name] = path
            self.pins[hash_key] = digest(raw)
            if size_key:
                self.pins[size_key] = len(raw)
        self.config = self.root / "training-config.json"
        self.config.write_text('{"fixture":true,"epochs":3}')
        self.pins["training_config_sha256"] = digest(b'{"epochs":3,"fixture":true}')
        record = json.loads((wn.ROOT / wn.RECORD_RELATIVE).read_text())
        record.update({k: v for k, v in self.pins.items() if k in record})
        raw_record = wc.canonical_json(record)
        (self.root / wn.RECORD_RELATIVE).write_bytes(raw_record)
        self.closure = {"historical_missing": [], "unreviewed_replacements": {}, "reviewed_replacements": {},
                        "review_evidence_present": True, "upstream_checked": True,
                        "current_files": {"bench/workbench_neural.py": "a" * 64}, "closure_sha256": "b" * 64}
        self.stack.enter_context(patch.object(wn, "PRIMARY", self.pins))
        self.stack.enter_context(patch.object(wn, "RECORD_SHA256", digest(raw_record)))
        self.source_mock = self.stack.enter_context(patch.object(wn, "source_closure", return_value=self.closure))
        self.package_mock = self.stack.enter_context(patch.object(wn, "runtime_packages", return_value={"fixture": "1"}))
        common = {"config_sha256": self.pins["training_config_sha256"],
                  "source_lock_sha256": self.pins["training_lock_sha256"], "dataset_sha256": record["dataset_sha256"]}
        self.headers = {
            "head": {**common, "schema": "correspondence_head_export_v1",
                     "base_checkpoint_sha256": self.pins["base_checkpoint_sha256"],
                     "deployment": record["export_deployment"], "quality_claim": False},
            "reference": {**common, "schema": "correspondence_head_checkpoint_v1"},
        }
        self.metadata_reader = wn._checkpoint_training_metadata
        self.metadata_mock = self.stack.enter_context(patch.object(wn, "_checkpoint_training_metadata",
            side_effect=lambda path, *_args, **_kwargs: copy.deepcopy(
                self.headers["head" if path == self.paths["head"] else "reference"])))

    def earn(self):
        return wn.create_recorded_artifact_proof(root=self.root, head_path=self.paths["head"],
            reference_path=self.paths["reference"], training_config_path=self.config)

    def contract(self, proof):
        body = {"recorded_artifact_proof": proof, "execution_scope": wn.RECORDED_EXECUTION_SCOPE,
                "live_eligible": False, "artifact_lineage": wn._recorded_lineage(proof)}
        return wn.WorkbenchContract(body, wc.canonical_sha256(body))

    def test_explicit_metadata_only_mode_crossbinds_identity_without_raw_config_claim(self):
        self.config.unlink()  # no training dictionary is synthesized or required
        proof = wn.create_recorded_artifact_proof(root=self.root, head_path=self.paths["head"],
            reference_path=self.paths["reference"], training_config_metadata_only=True)
        config = proof["training_configuration"]
        self.assertFalse(config["raw_config_available"])
        self.assertFalse(config["historical_training_reproduction_claim"])
        self.assertEqual(config["verification"], wn.CONFIG_METADATA_VERIFICATION)
        self.assertEqual(config["identity_sha256"], self.pins["training_config_sha256"])
        self.assertNotIn("training_config", proof["files"])
        self.assertEqual(wn.verify_recorded_artifact_proof(proof, root=self.root), proof)
        for raw_claim in (True, "true"):
            bad = copy.deepcopy(proof)
            bad["training_configuration"]["raw_config_available"] = raw_claim
            with self.assertRaises(wn.AdmissionError):
                wn.verify_recorded_artifact_proof(reseal(bad), root=self.root)
        for key, value in (("historical_training_reproduction_claim", True), ("verification", "raw_hash_verified"),
                           ("identity_sha256", "0" * 64)):
            bad = copy.deepcopy(proof)
            bad["training_configuration"][key] = value
            with self.subTest(key=key), self.assertRaises(wn.AdmissionError):
                wn.verify_recorded_artifact_proof(reseal(bad), root=self.root)

    def test_config_metadata_mode_is_never_an_implicit_fallback(self):
        for kwargs in ({}, {"training_config_path": self.config, "training_config_metadata_only": True},
                       {"training_config_metadata_only": "true"}):
            with self.assertRaisesRegex(wn.AdmissionError, "choose exactly one"):
                wn.create_recorded_artifact_proof(root=self.root, head_path=self.paths["head"],
                    reference_path=self.paths["reference"], **kwargs)

    def test_mismatched_authenticated_export_or_reference_metadata_is_rejected(self):
        for role in self.headers:
            for key in self.headers[role]:
                original = self.headers[role][key]
                self.headers[role][key] = "wrong" if key != "quality_claim" else True
                try:
                    with self.subTest(role=role, key=key), self.assertRaises(wn.AdmissionError):
                        wn.create_recorded_artifact_proof(root=self.root, head_path=self.paths["head"],
                            reference_path=self.paths["reference"], training_config_metadata_only=True)
                finally:
                    self.headers[role][key] = original
        proof = self.earn()
        bad = copy.deepcopy(proof)
        bad["authenticated_checkpoint_metadata"]["reference"]["config_sha256"] = "0" * 64
        with self.assertRaises(wn.AdmissionError):
            wn.verify_recorded_artifact_proof(reseal(bad), root=self.root)

    def test_checkpoint_metadata_reader_authenticates_before_strict_weights_only_load(self):
        from types import SimpleNamespace
        loaded = {"head": {}, **self.headers["head"], "optimizer": "not returned"}
        loader = Mock(return_value=loaded)
        module = SimpleNamespace(load=loader)
        result = self.metadata_reader(self.paths["head"], self.pins["export_bytes"],
            self.pins["export_sha256"], torch_module=module)
        self.assertEqual(result, self.headers["head"])
        self.assertEqual(loader.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        loader.reset_mock()
        with self.assertRaises(wn.AdmissionError):
            self.metadata_reader(self.paths["head"], self.pins["export_bytes"], "0" * 64, torch_module=module)
        loader.assert_not_called()
        for output in (None, {"head": 1}):
            loader.return_value = output
            with self.assertRaises(wn.AdmissionError):
                self.metadata_reader(self.paths["head"], self.pins["export_bytes"],
                    self.pins["export_sha256"], torch_module=module)

    def test_default_still_requires_exact_historical_small_records(self):
        with self.assertRaisesRegex(wn.NeuralUnavailable, "small-records file is missing"):
            wn.verify_primary_record(self.root)
        proof = self.earn()
        actual = wn.verify_primary_record(self.root, recorded_artifact=proof)
        self.assertFalse(actual["artifact_lineage"]["historical_small_records_verified"])
        self.assertFalse(actual["artifact_lineage"]["historical_receipts_verified"])
        self.assertFalse(proof["execution_admission_granted"])
        self.assertFalse((self.root / wn.SMALL_RECORDS_RELATIVE).exists())

    def test_earned_proof_reloads_exact_files_without_importing_torch(self):
        import sys
        torch_before = "torch" in sys.modules
        proof = self.earn()
        path = self.root / "proof.json"
        path.write_bytes(wc.canonical_json(proof))
        self.assertEqual(wn.load_recorded_artifact_proof(path, root=self.root), proof)
        self.assertEqual("torch" in sys.modules, torch_before)
        self.assertTrue(self.source_mock.call_args.kwargs["include_upstream"])

    def test_unsealed_or_resealed_scope_identity_history_claim_tamper_rejected(self):
        proof = self.earn()
        for key, value in (("execution_scope", "live"), ("live_eligible", True),
                           ("historical_small_records_verified", True), ("historical_receipts_verified", True),
                           ("execution_admission_granted", True), ("quality_claim", True),
                           ("lineage", "historical"), ("record_sha256", "0" * 64),
                           ("primary", {**proof["primary"], "artifact_id": "wrong"}), ("extra", "ignored?")):
            tampered = copy.deepcopy(proof)
            tampered[key] = value
            for candidate in (tampered, reseal(tampered)):
                with self.subTest(key=key, sealed=candidate is not tampered), self.assertRaises(wn.AdmissionError):
                    wn.verify_recorded_artifact_proof(candidate, root=self.root)

    def test_each_model_installer_lock_record_and_config_byte_drift_rejected(self):
        proof = self.earn()
        paths = [*self.paths.values(), self.root / wn.RECORD_RELATIVE, self.config]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            try:
                with self.subTest(path=path.name), self.assertRaises((wn.AdmissionError, ValueError)):
                    wn.verify_recorded_artifact_proof(proof, root=self.root)
            finally:
                path.write_bytes(original)

    def test_missing_size_symlink_and_wrong_config_fail_before_admission(self):
        original = self.paths["head"].read_bytes()
        for raw in (b"", original + b"extra"):
            self.paths["head"].write_bytes(raw)
            with self.assertRaises(wn.AdmissionError):
                self.earn()
        self.paths["head"].unlink()
        self.paths["head"].symlink_to(self.paths["reference"])
        with self.assertRaises(wn.NeuralUnavailable):
            self.earn()
        self.paths["head"].unlink()
        self.paths["head"].write_bytes(original)
        self.config.write_text('{"fixture":false,"epochs":3}')
        with self.assertRaisesRegex(wn.AdmissionError, "canonical identity"):
            self.earn()

    def test_source_and_runtime_drift_unreviewed_missing_or_unbound_rejected(self):
        proof = self.earn()
        for key, value in (("historical_missing", ["missing.py"]), ("unreviewed_replacements", {"bad": {}}),
                           ("review_evidence_present", False), ("current_files", {"missing": None}),
                           ("closure_sha256", "0" * 64)):
            self.source_mock.return_value = {**self.closure, key: value}
            with self.subTest(key=key), self.assertRaises(wn.AdmissionError):
                wn.verify_recorded_artifact_proof(proof, root=self.root)
        self.source_mock.return_value = self.closure
        self.package_mock.return_value = {"fixture": "2"}
        with self.assertRaisesRegex(wn.AdmissionError, "drift"):
            wn.verify_recorded_artifact_proof(proof, root=self.root)

    def test_location_config_presentation_and_proof_file_bounds(self):
        proof = self.earn()
        with tempfile.TemporaryDirectory() as another, self.assertRaisesRegex(wn.AdmissionError, "another repository"):
            wn.verify_recorded_artifact_proof(proof, root=Path(another))
        self.config.write_text('{ "epochs": 3, "fixture": true }')
        with self.assertRaisesRegex(wn.AdmissionError, "drift"):
            wn.verify_recorded_artifact_proof(proof, root=self.root)
        path = self.root / "huge.json"
        path.write_bytes(b" " * ((1 << 20) + 1))
        with self.assertRaisesRegex(wn.AdmissionError, "1 MiB"):
            wn.load_recorded_artifact_proof(path, root=self.root)

    def test_contract_requires_explicit_same_proof_and_forbids_scope_drift(self):
        proof = self.earn()
        contract = self.contract(proof)
        wn._contract_recorded_proof(contract, proof, root=self.root)
        for c, p in ((contract, None), (wn.WorkbenchContract({}, "x"), proof),
                     (self.contract({**proof, "proof_sha256": "0" * 64}), proof)):
            with self.assertRaises(wn.AdmissionError):
                wn._contract_recorded_proof(c, p, root=self.root)
        contract.body["live_eligible"] = True
        with self.assertRaises(wn.AdmissionError):
            wn._contract_recorded_proof(contract, proof, root=self.root)

    def test_fresh_contract_cannot_skip_base_or_upstream_verification(self):
        proof = self.earn()
        for kwargs in ({"verify_base": False}, {"include_upstream": False}):
            with self.assertRaisesRegex(wn.AdmissionError, "complete base/source"):
                wn.prepare_workbench_contract([], root=self.root, head_path=self.paths["head"],
                    reference_path=self.paths["reference"], recorded_artifact=proof, **kwargs)

    def test_artifact_proof_without_measured_parity_never_admits(self):
        proof = self.earn()
        adm = wn.verify_serving_admission(self.root / "no-parity", contract=self.contract(proof),
            geometry={"admission_eligible": True}, root=self.root, recorded_artifact=proof)
        self.assertFalse(adm.admitted)
        self.assertIn("parity receipt unreadable", adm.reasons[0])

    def test_live_context_rejected_before_file_reverification_with_or_without_opt_in(self):
        proof = self.earn()
        with patch.object(wn, "verify_recorded_artifact_proof", side_effect=AssertionError("must reject before I/O")):
            for opted in (None, proof):
                adm = wn.verify_serving_admission(self.root, contract=self.contract(proof), geometry={}, root=self.root,
                    recorded_artifact=opted, execution_context="live")
                self.assertFalse(adm.admitted)
                self.assertIn("LIVE use is forbidden", adm.reasons[0])

    def test_parity_missing_explicit_proof_rejects_before_torch_or_output(self):
        proof = self.earn()
        out = self.root / "parity"
        with self.assertRaisesRegex(wn.AdmissionError, "explicit recorded artifact"):
            wn.run_fullimage_head_parity([], contract=self.contract(proof), head_path=self.paths["head"],
                reference_path=self.paths["reference"], output_dir=out)
        self.assertFalse(out.exists())

    def test_native_live_factory_rejects_even_a_positive_recorded_only_admission(self):
        from bench import workbench_live_pose as live
        # The independent live receipt is a test fixture, not a published admission.
        factory = object.__new__(live.ConfiguredGoTrackFactory)
        factory.root = self.root
        binding = {"head_sha256": self.pins["export_sha256"],
                   "base_sha256": self.pins["base_checkpoint_sha256"], "assets": {}}
        receipt = {"schema": "native_live_admission_v1", "contract_id": "test-live", "bindings": binding,
                   "serving_contract_sha256": "c" * 64, "calibrations": [], "evidence": {}}
        path = self.root / "live-test-receipt.json"
        path.write_bytes(wc.canonical_json(receipt))
        factory.config = {"live_admission_path": path.name, "live_admission_sha256": digest(path.read_bytes()),
                          "serving_admission_dir": "admission"}
        for extra in ({"execution_scope": wn.RECORDED_EXECUTION_SCOPE}, {"live_eligible": False}):
            admission = wn.ServingAdmission(True, (), {"contract_sha256": "c" * 64, **extra})
            with patch.object(factory, "_pins", return_value=binding), patch.object(live, "HEAD", self.pins["export_sha256"]), \
                 patch.object(wn, "load_admission", return_value=admission) as loader:
                with self.assertRaisesRegex(live.NativeError, "cannot authorize LIVE"):
                    factory.admission()
                self.assertEqual(loader.call_args.kwargs["execution_context"], "live")

    def test_primary_session_cannot_mix_historical_and_recorded_install_lineage(self):
        from bench.test_workbench_contract import clip_document
        clip = wc.ClipSpec.from_dict(clip_document())
        proof = self.earn()
        lineage = wn._recorded_lineage(proof)
        historical = {"families": ["fixture"], "contract_sha256": "c" * 64}
        fresh = {**historical, "artifact_lineage": lineage, "execution_scope": wn.RECORDED_EXECUTION_SCOPE,
                 "live_eligible": False}
        class Backend:
            def __init__(self, extras): self.extras = extras
            def install_provenance(self):
                return {"head": {"installed_sha256": wn.PRIMARY["export_sha256"]}, **self.extras}
        for body, extras in ((fresh, {}), (historical, {"artifact_lineage": lineage,
                "execution_scope": wn.RECORDED_EXECUTION_SCOPE}),
                (fresh, {"artifact_lineage": {**lineage, "proof_sha256": "0" * 64},
                         "execution_scope": wn.RECORDED_EXECUTION_SCOPE})):
            with self.assertRaisesRegex(wn.NeuralUnavailable, "differs from recorded-only admission"):
                wn.NeuralSession("primary_neural", Backend(extras), wn.ServingAdmission(True, (), body),
                                 run_id="test", clip=clip, family="fixture")
        session = wn.NeuralSession("primary_neural", Backend({"artifact_lineage": lineage,
            "execution_scope": wn.RECORDED_EXECUTION_SCOPE}), wn.ServingAdmission(True, (), fresh),
            run_id="test", clip=clip, family="fixture")
        self.assertFalse(session.identity()["live_eligible"])
        self.assertEqual(session.identity()["artifact_lineage"], lineage)

    def test_cli_default_and_explicit_flags_are_separate(self):
        p = cli.build_parser()
        self.assertIsNone(p.parse_args('gate verify --admission d'.split()).recorded_artifact)
        for command in ('gate contract --cases c --head h --reference b --out d',
                        'gate parity --contract c --cases c --head h --reference b --out d',
                        'gate verify --admission d', 'run c'):
            args = p.parse_args((command + ' --recorded-artifact proof.json').split())
            self.assertEqual(args.recorded_artifact, 'proof.json')
        args = p.parse_args('gate artifact --head h --reference b --training-config c --out p'.split())
        self.assertEqual(args.gate, 'artifact')
        args = p.parse_args('gate artifact --head h --reference b --training-config-metadata-only --out p'.split())
        self.assertTrue(args.training_config_metadata_only)
        self.assertIsNone(args.training_config)


if __name__ == '__main__':
    unittest.main()
