"""Model-free sharding contract tests; fixtures make no quality claims."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as prepared
from bench import quality_adaptation_sharded_v1 as sharded
from bench.test_quality_adaptation_portable_prepare import Fixture, known_role_fixture, write


class ShardedRawIntegrationTests(unittest.TestCase):
    """Use original RAW plan/state/seal producers around explicit software fixtures."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.recipe, self.audit = known_role_fixture(base)
        self.assertEqual(self.recipe["seed"], 521)
        self.assertEqual(self.recipe["iterations"], [0])
        self.assertEqual(prepared.canonical(self.recipe), self.audit["recipe_sha256"])
        self.fixtures = {}
        for shard_id in sharded.SHARD_IDS:
            recipe = sharded._projected_recipe(self.recipe, sharded.TRAIN_GROUPS[shard_id])
            fixture_root = base / shard_id
            fixture_root.mkdir()
            fixture = Fixture(fixture_root, recipe=recipe)
            # The child fixture delegates the complete prospective plan and all
            # capture identities to the unchanged original RAW implementation.
            self.fixtures[shard_id] = fixture

        # A real-format seed record is retained on one available smoke frame;
        # this is an analytical fixture, not model evidence.
        smoke = self.fixtures["smoke0"]
        frame = next(row for row in smoke.physical
            if row["split"] == "train" and row["render"]["status"] == "available")
        frame["seed"] = {"status": "available", "poses": [
            [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]],
            "selection": "first_FoundPose_quality_sorted_hypothesis"}
        write(smoke.root / "physical-index.json", {"rows": smoke.physical})

        def original_frozen(context):
            raw.verify_sources(context)
            for fixture in self.fixtures.values():
                if fixture.context.sha256 == context.sha256:
                    return fixture.context.document(), fixture.spec, fixture.plan, fixture.physical
            raise AssertionError("test context is not one of the frozen original RAW fixtures")

        self.original_frozen = original_frozen
        self._seal_original_children()

    def _seal_original_children(self):
        # This invokes the real seal_capture producer. Only cloud-stage/resource
        # admission is disabled, as in the existing raw-preparation tests.
        with patch.object(raw, "frozen", side_effect=self.original_frozen), \
                patch.object(raw, "_require_stage"), patch.object(raw, "resource_check"):
            for fixture in self.fixtures.values():
                (fixture.root / "raw-seal.json").unlink()
                closing = fixture.root / "stage-seal-capture.json"
                closing_bytes = closing.read_bytes()
                closing.unlink()
                raw.seal_capture(fixture.context)
                closing.write_bytes(closing_bytes)

    def _pin(self, fixture, name):
        return prepared.pin(fixture.root / name)

    def test_original_raw_projection_aggregate_and_global_identity_roundtrip(self):
        smoke = self.fixtures["smoke0"]
        with patch.object(raw, "frozen", side_effect=self.original_frozen):
            frozen = sharded.freeze_partition(
                self.recipe, self.audit,
                {"context": self._pin(smoke, "context.json"),
                 "raw_seal": self._pin(smoke, "raw-seal.json")},
                Path(self.temp.name) / "partition")
            partition_pin = frozen["partition"]
            partition_doc = sharded.verify_partition(partition_pin)
            self.assertEqual(set(partition_doc["source_files"]), {
                "bench/quality_adaptation_sharded_v1.py"})
            owner = {}
            for sid, child in partition_doc["children"].items():
                owner.update({ident: sid for ident in child["owned_state_ids"]})
            for row in partition_doc["full_plan"]["global_manifest"]["records"]:
                if row["split"] in ("validation", "test"):
                    self.assertEqual(owner[row["id"]], "smoke0")
            self.assertEqual(partition_doc["fixed_disclosure"]["duplicate_validation_is_ignored"], True)
            refs = [{"shard_id": "smoke0",
                "context": self._pin(smoke, "context.json"),
                "raw_seal": self._pin(smoke, "raw-seal.json"),
                "projection": frozen["imported_shard0_projection"]}]
            for shard_id in sharded.SHARD_IDS[1:]:
                fixture = self.fixtures[shard_id]
                projection = sharded.seal_shard(partition_pin, shard_id,
                    self._pin(fixture, "context.json"), self._pin(fixture, "raw-seal.json"),
                    Path(self.temp.name) / (shard_id + "-projection.json"))
                refs.append({"shard_id": shard_id,
                    "context": self._pin(fixture, "context.json"),
                    "raw_seal": self._pin(fixture, "raw-seal.json"),
                    "projection": projection})

            # Exercise the valid metadata-only path called out in independent
            # review, then import all four original RAW children.
            for ref in refs:
                state = sharded.verify_shard(partition_pin, ref["shard_id"],
                    ref["context"], ref["raw_seal"], ref["projection"], payloads=False)
                self.assertFalse(state["raw_state"]["seal"].get("training_ready"))
            aggregate = sharded.aggregate_raw(partition_pin, refs,
                Path(self.temp.name) / "aggregate")
            verified = sharded.verify_aggregate_raw(partition_pin, aggregate["root"], payloads=False)
            payload_verified = sharded.verify_aggregate_raw(partition_pin, aggregate["root"], payloads=True)
            self.assertEqual(payload_verified["seal"]["test_payload_reads"], 0)

            # The unchanged numerical identity bridge consumes the distinct
            # aggregate state without disguising its new RAW seal schema.
            execution = Path(self.temp.name) / "aggregate-execution"
            with patch.object(raw, "frozen", side_effect=self.original_frozen):
                identity = prepared.execution_identity(verified, execution)
                self.assertEqual(prepared.verify_execution(verified, execution), identity)
            self.assertEqual(verified["seal"]["schema"], sharded.AGGREGATE_SCHEMA)

            # A last TEST descriptor is rejected in the metadata pass before
            # any optional child-payload verifier can run.
            aggregate_root = Path(aggregate["root"])
            capture_index_path = aggregate_root / "raw/capture-index.json"
            capture_index = prepared.read_json(capture_index_path)
            test_ids = {row["id"] for row in partition_doc["full_plan"]["global_manifest"]["records"]
                if row["split"] == "test"}
            tampered = next(row for row in reversed(capture_index["rows"]) if row["id"] in test_ids)
            tampered["crops"] = {"path": "forbidden-test-payload", "bytes": 1, "sha256": "0" * 64}
            seal_path = aggregate_root / "raw-seal.json"
            seal = prepared.read_json(seal_path)
            lineage = prepared.indexed(seal["lineage"], "aggregate lineage")
            lineage[tampered["id"]]["aggregate_capture_sha256"] = sharded._canonical(tampered)
            seal["lineage"] = list(lineage.values())
            seal["refinement_states"] = capture_index["rows"]
            seal = sharded._sealed({key: value for key, value in seal.items()
                if key not in ("aggregate_sha256", "seal_sha256")}, "aggregate_sha256")
            seal["seal_sha256"] = seal["aggregate_sha256"]
            capture_index_path.unlink()
            seal_path.unlink()
            prepared.atomic_json(capture_index_path, capture_index)
            prepared.atomic_json(seal_path, seal)
            payload_calls = []
            original_verify_shard = sharded.verify_shard
            def track_payload_calls(*args, **kwargs):
                payload_calls.append(kwargs.get("payloads", False))
                return original_verify_shard(*args, **kwargs)
            with patch.object(sharded, "verify_shard", side_effect=track_payload_calls):
                with self.assertRaises(ValueError):
                    sharded.verify_aggregate_raw(partition_pin, aggregate_root, payloads=True)
            self.assertNotIn(True, payload_calls)

        partition_doc = prepared.metadata(partition_pin)
        self.assertEqual(verified["plan"], partition_doc["full_plan"])
        self.assertEqual(prepared.read_json(aggregate["root"] / "scale-plan.json"), verified["plan"])
        self.assertEqual(len(verified["captures"]), len(verified["plan"]["global_manifest"]["records"]))


class ShardedSourceBoundaryTests(unittest.TestCase):
    def test_partition_bridge_does_not_pin_preparation_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bench").mkdir()
            (root / "scripts").mkdir()
            bridge = root / "bench/quality_adaptation_sharded_v1.py"
            bridge.write_bytes(Path(sharded.__file__).read_bytes())
            wrapper = root / "scripts/kaggle_sharded_prepared_v1.py"
            wrapper.write_bytes(b"first dummy wrapper bytes")
            pins = sharded._pin_code(root)
            self.assertEqual(set(pins), {"bench/quality_adaptation_sharded_v1.py"})
            wrapper.write_bytes(b"different dummy wrapper bytes")
            sharded._verify_code_pins(pins, root)
            bridge.write_bytes(bridge.read_bytes() + b"\n# drift")
            with self.assertRaises(ValueError):
                sharded._verify_code_pins(pins, root)

    def test_original_raw_source_closure_still_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = set(raw.source_paths()) | {"bench/runtime/adaptation-sources.json"}
            for name in names:
                source = Path(raw.__file__).resolve().parents[1] / name
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for name in ("parity-reference.txt", "parity-candidate.txt"):
                (root / name).write_text("software fixture\n", encoding="utf-8")
            with patch.object(raw, "ROOT", root):
                context_doc = raw.context_document(output_root=root / "work/raw",
                    scratch_root=root / "scratch/raw",
                    parity={"reference": raw.pin(root / "parity-reference.txt"),
                            "candidate": raw.pin(root / "parity-candidate.txt")}, device="cpu")
                context = raw.PortableContext(raw.encoded(context_doc),
                    hashlib.sha256(raw.encoded(context_doc)).hexdigest())
                raw.verify_sources(context)
                target = root / "bench/quality_adaptation_portable.py"
                target.write_bytes(target.read_bytes() + b"\n# original-source drift")
                with self.assertRaisesRegex(ValueError, "portable executing source changed"):
                    raw.verify_sources(context)


if __name__ == "__main__":
    unittest.main()
