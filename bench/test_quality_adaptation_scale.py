"""Tiny manufactured metadata/bytes only; these fixtures establish no accuracy."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import quality_adaptation_scale as scale


def hash_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def inventory(count=6):
    # Synthetic provenance describes candidate metadata only. No images, labels,
    # neural features, or independent accuracy evidence are manufactured here.
    return {"label_policy": {"max_rotation_error_deg": 5, "max_translation_error_m": .02},
        "records": [{"id": f"state-{i}", "physical_object_id": f"new:{i}", "scene_id": f"scene:{i}",
            "sequence_id": f"sequence:{i}", "background_id": "procedural:neutral", "background_kind": "procedural",
            "provenance_kind": "synthetic", "frame_id": 0, "source_frame_id": i,
            "refinement_state": 0, "mesh_sha256": hash_text(f"mesh:{i}"),
            "inputs_sha256": hash_text(f"input:{i}"), "payload": "/denied/nonexistent/never-open"}
            for i in range(count)]}


def receipts(plan, root):
    result = []
    for shard in plan["shards"]:
        rows = []
        for row in shard["rows"]:
            path = root / (row["id"] + ".bin")
            path.write_bytes(row["id"].encode())
            rows.append({**row, "status": "available", "output": {"path": path.name,
                "bytes": path.stat().st_size, "sha256": scale.digest(path)}})
        result.append({"manifest_sha256": plan["global_manifest"]["manifest_sha256"],
            "shard_sha256": shard["shard_sha256"], "producer_sha256": hash_text("producer"),
            "config_sha256": hash_text("config"), "rows": rows, "preparation_seconds": .1})
    return result


def reseal(plan):
    """Simulate a corrupt producer recomputing every affected digest."""
    manifest = plan["global_manifest"]
    manifest["split_manifest"]["manifest_sha256"] = scale.ranker._split_digest(manifest["split_manifest"])
    plan["global_manifest"] = scale._seal({k: v for k, v in manifest.items() if k != "manifest_sha256"}, "manifest_sha256")
    for index, shard in enumerate(plan["shards"]):
        shard["manifest_sha256"] = plan["global_manifest"]["manifest_sha256"]
        plan["shards"][index] = scale._seal({k: v for k, v in shard.items() if k != "shard_sha256"}, "shard_sha256")
    return scale._seal({k: v for k, v in plan.items() if k != "plan_sha256"}, "plan_sha256")


class ScaleTests(unittest.TestCase):
    def plan(self, data=None):
        return scale.plan_scale(data or inventory(), shard_size=2, targets=[1, 2])

    def verify(self, plan, receipt, root, **kwargs):
        return scale.validate_preparation_receipt(plan, receipt, output_root=root,
            producer_sha256=kwargs.get("producer", hash_text("producer")), config_sha256=hash_text("config"))

    def test_reordering_preserves_split_manifest_and_shards(self):
        data = inventory()
        a = self.plan(data)
        data["records"].reverse()
        b = self.plan(data)
        self.assertEqual(a, b)
        self.assertEqual(a["counts"]["train"]["refinement_states"], 4)
        self.assertEqual(a["raw_token_bytes_per_state"], 4_915_200)

    def test_each_shared_dimension_and_coobjects_stay_together(self):
        for dimension in ("mesh_sha256", "scene_id", "physical_object_id", "alias", "observed"):
            with self.subTest(dimension=dimension):
                data = inventory(7)
                a, b = data["records"][:2]
                if dimension in ("mesh_sha256", "scene_id", "physical_object_id"):
                    b[dimension] = a[dimension]
                elif dimension == "alias":
                    b["aliases"] = [a["physical_object_id"]]
                else:
                    a.update(background_kind="observed", background_id="photo-a")
                    b.update(background_kind="observed", background_id="photo-b")
                    data["background_catalog"] = [{"background_id": row["background_id"], "scene_id": row["scene_id"],
                        "inventory_complete": True, "physical_object_ids": [row["physical_object_id"], "co-occurring-shared"]}
                        for row in (a, b)]
                plan = self.plan(data)
                mapping = {r["id"]: r["split"] for r in plan["global_manifest"]["records"]}
                self.assertEqual(mapping[a["id"]], mapping[b["id"]])

    def test_dev_identity_propagates_and_infeasible_is_explicit(self):
        data = inventory()
        data["records"][0]["physical_object_id"] = "bop-ycbv:000002"
        data["records"][1]["mesh_sha256"] = data["records"][0]["mesh_sha256"]
        plan = self.plan(data)
        self.assertEqual([r["split"] for r in plan["global_manifest"]["records"][:2]], ["dev", "dev"])
        small = self.plan(inventory(2))
        self.assertFalse(small["feasible"])
        self.assertIn("fewer_than_three_independent_nondev_components", small["blockers"])
        self.assertFalse(small["shards"])

    def test_multiple_refinement_states_are_not_new_frames(self):
        data = inventory()
        second = {**data["records"][0], "id": "state-0-refined", "refinement_state": 1}
        data["records"].append(second)
        plan = self.plan(data)
        self.assertEqual(sum(x["refinement_states"] for x in plan["counts"].values()), 7)
        self.assertEqual(sum(x["source_frames"] for x in plan["counts"].values()), 6)

    def test_changed_plan_resume_and_bytes_reject(self):
        plan = self.plan()
        changed = deepcopy(plan)
        changed["counts"]["train"]["objects"] += 1
        with self.assertRaisesRegex(ValueError, "changed plan_sha256"):
            scale.validate_plan(changed)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            receipt = receipts(plan, root)[0]
            self.assertGreater(self.verify(plan, receipt, root)["available"], 0)
            for key in ("manifest_sha256", "config_sha256", "producer_sha256", "shard_sha256"):
                bad = deepcopy(receipt)
                bad[key] = hash_text("changed")
                with self.assertRaises(ValueError):
                    self.verify(plan, bad, root)
            (root / receipt["rows"][0]["output"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.verify(plan, receipt, root)

    def test_failed_rows_retained_test_identity_rejects_before_open(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            receipt = receipts(plan, root)[0]
            receipt["rows"][0].update(status="failed", failure_reason="fixture preparation failed")
            receipt["rows"][0]["output"]["path"] = "does-not-exist"
            summary = self.verify(plan, receipt, root)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(len(summary["failed_rows"]), 1)
            self.assertEqual(summary["total"], len(receipt["rows"]))
            test_id = next(r["id"] for r in plan["global_manifest"]["records"] if r["split"] == "test")
            receipt["rows"][0]["id"] = test_id
            with patch.object(scale, "digest", side_effect=AssertionError("payload opened")):
                with self.assertRaisesRegex(ValueError, "retain every shard"):
                    self.verify(plan, receipt, root)

    def test_metadata_planning_never_opens_any_payload(self):
        with patch.object(Path, "open", side_effect=AssertionError("payload opened")):
            self.assertTrue(self.plan()["feasible"])

    def test_resealed_inconsistent_plans_reject_before_any_payload_access(self):
        original = self.plan()
        test = next(r for r in original["global_manifest"]["records"] if r["split"] == "test")
        variants = {}
        for name in ("test_substitution", "unknown_id", "changed_input", "duplicate_shard_row", "omitted_shard_row",
                     "wrong_shard_split", "inflated_counts", "duplicate_global_id", "omitted_global_record",
                     "relabeled_global_split", "changed_global_input", "false_target_coverage"):
            plan = deepcopy(original)
            shard = plan["shards"][0]
            if name == "test_substitution":
                shard["rows"][0] = {"id": test["id"], "inputs_sha256": test["inputs_sha256"]}
            elif name == "unknown_id":
                shard["rows"][0]["id"] = "fabricated-id"
            elif name == "changed_input":
                shard["rows"][0]["inputs_sha256"] = hash_text("changed input")
            elif name == "duplicate_shard_row":
                shard["rows"].append(deepcopy(shard["rows"][0]))
            elif name == "omitted_shard_row":
                shard["rows"].pop()
            elif name == "wrong_shard_split":
                shard["split"] = "validation"
            elif name == "inflated_counts":
                plan["counts"]["train"]["refinement_states"] = 1000
            elif name == "duplicate_global_id":
                plan["global_manifest"]["records"].append(deepcopy(plan["global_manifest"]["records"][0]))
            elif name == "omitted_global_record":
                plan["global_manifest"]["records"].pop()
            elif name == "relabeled_global_split":
                next(r for r in plan["global_manifest"]["records"] if r["id"] == test["id"])["split"] = "train"
            elif name == "changed_global_input":
                plan["global_manifest"]["records"][0]["inputs_sha256"] = hash_text("changed input")
            else:
                plan["target_requests"][0]["request_met_by_inventory"] = False
            variants[name] = reseal(plan)
        for name, plan in variants.items():
            with self.subTest(name=name), patch.object(Path, "open", side_effect=AssertionError("payload opened")), \
                 patch.object(scale, "digest", side_effect=AssertionError("payload hashed")):
                with self.assertRaises(ValueError):
                    self.verify(plan, {}, Path("/denied/nonexistent"))
                with self.assertRaises(ValueError):
                    scale.assess_admission(plan, [{}], output_root=Path("/denied/nonexistent"))

    def test_resource_gate_needs_measured_evidence_and_rechecks_outputs(self):
        plan = self.plan()
        self.assertFalse(scale.assess_admission(plan)["admitted"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            items = receipts(plan, root)
            kwargs = {"producer_sha256": hash_text("producer"), "config_sha256": hash_text("config"), "output_root": root}
            self.assertFalse(scale.assess_admission(plan, items, **kwargs)["admitted"])
            measured = {"plan_sha256": plan["plan_sha256"], "train_states_requested": 1,
                "measured_training_steps": 2, "training_seconds": 1., "planned_optimizer_steps": 4,
                "actual_free_bytes": 100000, "storage_limit_bytes": 100000,
                "time_limit_seconds": 10, "reserve_bytes": 100, "storage_baseline": "after_preparation",
                "measured_resident_bytes": 1000, "measured_additional_peak_bytes": 100}
            report = scale.assess_admission(plan, items, measurements=measured, **kwargs)
            self.assertTrue(report["admitted"])
            self.assertIn("not_training_runtime", report["admission_scope"])
            measured["time_limit_seconds"] = .01
            self.assertFalse(scale.assess_admission(plan, items, measurements=measured, **kwargs)["admitted"])
            measured["time_limit_seconds"] = float("nan")
            with self.assertRaises(ValueError):
                scale.assess_admission(plan, items, measurements=measured, **kwargs)

    def test_post_preparation_free_space_and_total_quota_are_independent(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            items = receipts(plan, root)
            prepared = sum(row["output"]["bytes"] for receipt in items for row in receipt["rows"])
            self.assertEqual(prepared, 35)
            kwargs = {"producer_sha256": hash_text("producer"), "config_sha256": hash_text("config"), "output_root": root}
            measured = {"plan_sha256": plan["plan_sha256"], "train_states_requested": 1,
                "measured_training_steps": 2, "training_seconds": 1., "planned_optimizer_steps": 4,
                "actual_free_bytes": 140, "storage_limit_bytes": 100000, "time_limit_seconds": 10,
                "reserve_bytes": 10, "storage_baseline": "after_preparation",
                "measured_resident_bytes": prepared, "measured_additional_peak_bytes": 100}
            report = scale.assess_admission(plan, items, measurements=measured, **kwargs)
            self.assertTrue(report["admitted"])
            self.assertEqual(report["resources"]["projected_peak_bytes"], 135)
            measured.update(actual_free_bytes=110, storage_limit_bytes=145)
            self.assertTrue(scale.assess_admission(plan, items, measurements=measured, **kwargs)["admitted"])
            measured["actual_free_bytes"] = 109
            report = scale.assess_admission(plan, items, measurements=measured, **kwargs)
            self.assertEqual(report["blockers"], ["additional_storage_exceeds_free_headroom_or_reserve"])
            measured.update(actual_free_bytes=140, storage_limit_bytes=144)
            report = scale.assess_admission(plan, items, measurements=measured, **kwargs)
            self.assertEqual(report["blockers"], ["total_storage_exceeds_quota_or_reserve"])
            # Other already-resident assets consume quota, not newly measured free space.
            measured.update(storage_limit_bytes=145, measured_resident_bytes=36)
            self.assertIn("total_storage_exceeds_quota_or_reserve",
                          scale.assess_admission(plan, items, measurements=measured, **kwargs)["blockers"])
            for changes in ({"measured_resident_bytes": prepared - 1}, {"storage_baseline": "before_preparation"},
                            {"measured_additional_peak_bytes": -1}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    scale.assess_admission(plan, items, measurements={**measured, **changes}, **kwargs)


if __name__ == "__main__":
    unittest.main()
