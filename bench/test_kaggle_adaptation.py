"""Model-free contracts for the standalone Kaggle boundary; no toy training data.

Subprocess mocks test CLI wiring, not model correctness or authentic supervision.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import ast
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import signal
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings
import zipfile

from scripts import kaggle_adaptation as k


REPO = Path(__file__).resolve().parents[1]


class AdapterTest(unittest.TestCase):
    def setUp(self):
        parent = REPO / ".cache/kaggle-adapter-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def descriptor(self, path):
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": k.file_sha256(path)}

    def archive(self, mutation=None):
        data = b"authentic installation control bytes, not a training fixture"
        manifest = json.dumps({"files": {"file.txt": k.digest(data)}}).encode()
        members = [("root/file.txt", data), ("root/cloud-source-manifest.json", manifest)]
        if mutation:
            members = mutation(members)
        path = self.root / ("source-" + str(len(list(self.root.glob("*.zip")))) + ".zip")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                for name, value in members:
                    z.writestr(name, value)
        spec = k.ArchiveSpec(k.file_sha256(path), path.stat().st_size,
                             {"root": (1, k.digest(manifest))})
        return path, spec

    def request(self):
        output = self.root / "output"
        extraction = output / "extraction"
        request = {"schema": "kaggle_adaptation_request_v1", "python": sys.executable,
            "source_install": str(self.root / "source"), "input_root": str(self.root / "input"),
            "working_root": str(self.root), "output_root": str(output),
            "runtime": {"path": "runtime.json", "sha256": "a" * 64, "bytes": 1},
            "config": self.descriptor(self.write("config.json", {"device": "cuda", "max_steps": 2})),
            "evidence": self.descriptor(self.write("evidence.json", {"gpu": {"uuid": "GPU-mocked-control"}})),
            "global_inventory": {"path": "global-plan.json", "sha256": "b" * 64, "bytes": 1},
            "inputs": [{"role": "parity_report", "file": {"path": "parity.json", "sha256": "c" * 64, "bytes": 1}}],
            "crop_manifest": {"path": "crops.json", "sha256": "d" * 64, "bytes": 1},
            "labels": {"path": "labels.json", "sha256": "e" * 64, "bytes": 1},
            "extraction": {"max_rows": 2, "max_seconds": 1, "max_output_bytes": 1024},
            "paths": {"extraction_dir": str(extraction), "manifest_output": str(extraction / "manifest.json"),
                      "train_dir": str(output / "train"), "export_output": str(output / "train/head.pt")},
            "limits": {"max_seconds": 5, "max_rss_bytes": k.GIB, "max_output_bytes": k.GIB,
                       "max_log_bytes": 1024, "reserve_bytes": k.GIB}}
        return request

    def validation(self, request, stage):
        runtime = {"python": k.PYTHON_VERSION, "executable_sha256": "f" * 64}
        return REPO, runtime, k.canonical(k.identity_document(request, runtime))

    def metadata_fixture(self):
        """Manufactured metadata/opaque bytes test joins, never model validity."""
        from bench import quality_adaptation_scale as scale
        r = self.request()
        r["extraction"]["max_rows"] = 64
        r["working_root"] = str(REPO)
        input_root = self.root / "input"
        input_root.mkdir()
        r["input_root"] = str(input_root)
        extraction = Path(r["paths"]["extraction_dir"])
        extraction.mkdir(parents=True)
        inventory = {"label_policy": {"max_rotation_error_deg": 5, "max_translation_error_m": .02}, "records": []}
        bundles, references = {}, []
        blobs = {}
        for i in range(5):
            ident = f"state-{i}"
            def blob(name, value):
                path = input_root / name
                raw = value.encode()
                blobs[str(path)] = raw
                return {"path": str(path), "bytes": len(raw), "sha256": k.digest(raw)}
            mesh = blob(f"mesh-{i}.bin", f"metric mesh metadata control {i}")
            identity = {"id": ident, "physical_object_id": f"new:{i}", "scene_id": f"scene:{i}",
                "sequence_id": f"sequence:{i}", "frame_id": 0, "source_frame_id": i, "refinement_state": 0,
                "mesh_sha256": mesh["sha256"], "background_id": "procedural:neutral", "background_kind": "procedural",
                "provenance_kind": "synthetic", "aliases": []}
            bundle = {"schema": "kaggle_input_identity_v1", "identity": identity, "capture_group": identity["scene_id"],
                "status": "available", "query_rgb_sha256": k.digest(f"query {i}".encode()), "mesh": mesh,
                "source_frame": blob(f"source-{i}.bin", f"source metadata control {i}"),
                "crop": blob(f"crop-{i}.npz", f"opaque non-neural crop control {i}"),
                "labels": blob(f"original-label-{i}.npz", f"opaque non-neural label control {i}"),
                "label_output": f"label-{i}.npz"}
            bundle_path = self.write(f"input/provenance-{i}.json", bundle)
            ref = self.descriptor(bundle_path)
            bundles[ident] = bundle
            references.append({"id": ident, "input_provenance": ref})
            inventory["records"].append({**identity, "inputs_sha256": ref["sha256"]})
        plan = scale.plan_scale(inventory, shard_size=2, targets=[1])
        r["global_inventory"] = self.descriptor(self.write("plan.json", plan))
        bridge = {"schema": "kaggle_correspondence_identity_bridge_v1", "plan_sha256": plan["plan_sha256"],
            "manifest_sha256": plan["global_manifest"]["manifest_sha256"], "rows": references}
        bridge["bridge_sha256"] = k.canonical(bridge)
        r["identity_bridge"] = self.descriptor(self.write("bridge.json", bridge))
        r["scale_sources"] = {name: self.descriptor(REPO / "bench" / f"{name}.py")
            for name in ("quality_adaptation_scale", "quality_kaggle_preflight")}
        label_rows, crop_rows, index_rows = [], [], []
        for state in plan["global_manifest"]["records"]:
            ident = state["id"]
            bundle = bundles[ident]
            split = {"validation": "val"}.get(state["split"], state["split"])
            row = {"id": ident, "object_id": state["physical_object_id"], "sequence_id": state["sequence_id"],
                "capture_group": state["scene_id"], "query_rgb_sha256": bundle["query_rgb_sha256"], "split": split}
            if split == "test":
                row.update(status="unavailable", failure_reason="sealed metadata only")
            else:
                for key in ("mesh", "source_frame", "crop", "labels"):
                    path = bundle[key]["path"]
                    Path(path).write_bytes(blobs[path])
                label = {**bundle["labels"], "path": bundle["label_output"]}
                (extraction / label["path"]).write_bytes(blobs[bundle["labels"]["path"]])
                row.update(status="available", labels=label)
                crop = {**bundle["crop"], "path": Path(bundle["crop"]["path"]).name}
                crop_rows.append({"id": ident, "crops": crop})
                feature_path = extraction / f"features-{ident}.npz"
                feature_path.write_bytes(f"opaque feature metadata control {ident}".encode())
                feature = {**self.descriptor(feature_path), "path": feature_path.name}
                index_rows.append({"id": ident, "status": "available", "crops": crop,
                    "query_rgb_sha256": bundle["query_rgb_sha256"], "features": feature})
            label_rows.append(row)
        crops = {"schema": "correspondence_input_crops_v1", "conventions": {}, "rows": crop_rows}
        r["crop_manifest"] = self.descriptor(self.write("input/crops.json", crops))
        r["labels"] = self.descriptor(self.write("labels.json", {"schema": "correspondence_label_inventory_v1",
            "label_origin": "synthetic_renderer_training_only", "rows": label_rows}))
        recipe = {"schema": "frozen_gotrack_decoder_recipe_v1", "crop_manifest_sha256": k.canonical(crops),
            "producer_sha256": k.file_sha256(REPO / "bench/quality_adaptation_prepare.py"),
            "annotation_inputs": [], "device": "cuda"}
        self.write(str(extraction.relative_to(self.root) / "recipe.json"), recipe)
        self.write(str(extraction.relative_to(self.root) / "index.json"), {"recipe_sha256": k.canonical(recipe), "rows": index_rows})
        feature_by_id = {row["id"]: row["features"] for row in index_rows}
        dataset = {"schema": "correspondence_head_dataset_v1", "conventions": {},
            "provenance": {"feature_recipe_sha256": k.canonical(recipe)},
            "rows": [{**row, **({"features": feature_by_id[row["id"]]} if row["status"] == "available" else {})} for row in label_rows]}
        dataset_path = self.write(str(Path(r["paths"]["manifest_output"]).relative_to(self.root)), dataset)
        r["dataset"] = self.descriptor(dataset_path)
        config = self.descriptor(self.write("extraction-config.json", {**r["extraction"], "device": "cuda"}))
        producer = self.descriptor(REPO / "bench/quality_adaptation_prepare.py")
        receipts = []
        for i, shard in enumerate(plan["shards"]):
            receipt = {"manifest_sha256": plan["global_manifest"]["manifest_sha256"], "shard_sha256": shard["shard_sha256"],
                "producer_sha256": producer["sha256"], "config_sha256": config["sha256"], "preparation_seconds": .1,
                "rows": [{**row, "status": "available", "output": feature_by_id[row["id"]]} for row in shard["rows"]]}
            receipts.append(receipt)
        measured_bytes = sum(row["features"]["bytes"] for row in index_rows)
        measurements = {"plan_sha256": plan["plan_sha256"], "train_states_requested": 1, "measured_training_steps": 2,
            "training_seconds": 1, "planned_optimizer_steps": 2, "actual_free_bytes": 2*k.GIB,
            "storage_limit_bytes": 8*k.GIB, "time_limit_seconds": 900, "reserve_bytes": k.GIB,
            "storage_baseline": "after_preparation", "measured_resident_bytes": measured_bytes, "measured_additional_peak_bytes": 10000}
        r["preparation"] = {"producer": producer, "config": config, "output_root": str(extraction),
            "receipts": [self.descriptor(self.write(f"receipt-{i}.json", receipt)) for i, receipt in enumerate(receipts)],
            "measurements": self.descriptor(self.write("measurements.json", measurements))}
        admission = scale.assess_admission(plan, receipts, measurements=measurements,
            producer_sha256=producer["sha256"], config_sha256=config["sha256"], output_root=extraction)
        r["scale_admission"] = self.descriptor(self.write("admission.json", admission))
        r["inputs"] = [{"role": role, "file": self.descriptor(self.write(role+".json", {"control": True}))}
            for role in ("parity_report", "input_review")]
        return r, plan

    def invoke(self, stage, request):
        path = self.write("request.json", request)
        return k.execute(stage, path, k.file_sha256(path))

    def test_tiny_archive_roundtrip_and_no_overwrite(self):
        archive, spec = self.archive()
        target = self.root / "fresh"
        self.assertTrue(k.install_source(archive, target, spec=spec)["completed"])
        self.assertEqual((target / "root/file.txt").read_bytes(), b"authentic installation control bytes, not a training fixture")
        with self.assertRaisesRegex(k.ContractError, "fresh destination"):
            k.install_source(archive, target, spec=spec)

    def test_production_archive_exact_three_manifests_and_counts(self):
        archive = REPO / ".cache/cloud-prepare/larger-amortized-three-root-source-v1.zip"
        target = self.root / "production-install"
        receipt = k.install_source(archive, target)
        self.assertEqual(receipt["files"], 971)
        self.assertEqual(k.verify_source(target), target / k.ROOT18)
        for root, (count, expected) in k.ROOTS.items():
            manifest = target / root / "cloud-source-manifest.json"
            self.assertEqual(k.file_sha256(manifest), expected)
            self.assertEqual(len(k.read_json(manifest)["files"]), count)
        # Kaggle actually expands uploaded ZIP inputs; authenticate that shape
        # without claiming the original ZIP byte hash was observed there.
        (target / "source-install.json").unlink()
        expanded = self.root / "expanded-install"
        receipt = k.install_source(None, expanded, source_directory=target)
        self.assertEqual(receipt["kind"], "expanded_directory")
        self.assertNotIn("archive_sha256", receipt)
        self.assertEqual(k.verify_source(expanded), expanded / k.ROOT18)
        k.atomic_json(target / "source-install.json", k.source_receipt("archive"))
        touched = target / k.ROOT18 / "bench/__init__.py"
        touched.write_bytes(b"changed")
        with self.assertRaisesRegex(k.ContractError, "installed source changed"):
            k.verify_source(target)

    def test_expanded_source_corruption_extras_and_links_reject_before_writes(self):
        archive, spec = self.archive()
        source = self.root / "expanded"
        k.install_source(archive, source, spec=spec)
        (source / "source-install.json").unlink()
        with patch.object(k, "ROOTS", spec.roots):
            (source / "root/file.txt").write_bytes(b"changed")
            with self.assertRaises(k.ContractError):
                k.install_source(None, self.root / "never-corrupt", source_directory=source)
            self.assertFalse((self.root / "never-corrupt").exists())
            (source / "root/file.txt").write_bytes(b"authentic installation control bytes, not a training fixture")
            (source / "extra-empty").mkdir()
            with self.assertRaisesRegex(k.ContractError, "directories"):
                k.install_source(None, self.root / "never-extra", source_directory=source)
            (source / "extra-empty").rmdir()
            marker = source / "root/file.txt"
            original = Path.is_symlink
            with patch.object(Path, "is_symlink", lambda path: path == marker or original(path)):
                with self.assertRaisesRegex(k.ContractError, "symlink"):
                    k.install_source(None, self.root / "never-link", source_directory=source)
            self.assertFalse((self.root / "never-link").exists())

    def test_export_layout_satisfies_actual_archived_destination_guard(self):
        r = self.request()
        train = Path(r["paths"]["train_dir"])
        train.mkdir(parents=True)
        with zipfile.ZipFile(REPO / ".cache/cloud-prepare/larger-amortized-three-root-source-v1.zip") as z:
            tree = ast.parse(z.read(k.ROOT18 + "/bench/quality_adaptation_train.py"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "export")
        guard = next(node.test for node in ast.walk(function) if isinstance(node, ast.If)
            and "destination.exists()" in ast.unparse(node.test))
        code = compile(ast.Expression(guard), "authenticated ROOT18 export guard", "eval")
        destination = Path(r["paths"]["export_output"])
        k.validate_output_layout(r["paths"])
        self.assertFalse(eval(code, {"destination": destination, "run": train}))
        destination.write_bytes(b"existing artifact")
        self.assertTrue(eval(code, {"destination": destination, "run": train}))
        wrong = {**r["paths"], "export_output": str(train.parent / "sibling.pt")}
        with self.assertRaisesRegex(k.ContractError, "direct-child"):
            k.validate_output_layout(wrong)

    def test_metadata_join_reuses_actual_scale_validator_and_receipts(self):
        r, plan = self.metadata_fixture()
        result = k.metadata_check(r, "train", REPO)
        self.assertEqual(result["plan_sha256"], plan["plan_sha256"])
        self.assertTrue(result["prepared_admission"]["admitted"])
        self.assertFalse(result["sealed_payloads_opened"])
        self.assertNotIn("torch", sys.modules)

    def test_helper_loader_executes_authenticated_source_once_despite_stale_bytecode_constant(self):
        r = self.request()
        helpers = self.root / "helpers"
        helpers.mkdir()
        r["scale_sources"] = {}
        for name in ("quality_adaptation_scale", "quality_kaggle_preflight"):
            path = helpers / f"{name}.py"
            path.write_bytes((REPO / "bench" / path.name).read_bytes())
            r["scale_sources"][name] = self.descriptor(path)
        source = helpers / "quality_adaptation_scale.py"
        cached = Path(importlib.util.cache_from_source(str(source)))
        cached.parent.mkdir()
        # Ordinary stale-cache control: the alternate code contains only a
        # harmless different constant. No model/data behavior is executed.
        alternate = compile("RAW_TOKEN_BYTES_PER_STATE = 7\n", str(source), "exec")
        header = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0,
            int(source.stat().st_mtime) & 0xffffffff, source.stat().st_size)
        cached.write_bytes(header + marshal.dumps(alternate))
        spec = importlib.util.spec_from_file_location("_stale_constant_control", source)
        control = {}
        exec(spec.loader.get_code("_stale_constant_control"), control)
        self.assertEqual(control["RAW_TOKEN_BYTES_PER_STATE"], 7)
        watched = {helpers / "quality_adaptation_scale.py", helpers / "quality_kaggle_preflight.py",
                   REPO / "bench/quality_ranker_data.py"}
        reads = {path: 0 for path in watched}
        original = Path.open
        def record(path, *args, **kwargs):
            if path in reads:
                reads[path] += 1
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", record):
            module = k.load_scale_validator(r, REPO)
        self.assertEqual(module.RAW_TOKEN_BYTES_PER_STATE, 4_915_200)
        self.assertTrue(callable(module.validate_plan))
        self.assertEqual(set(reads.values()), {1})

    def update_frozen_provenance(self, r, plan, ident, bundle):
        """Freeze a manufactured metadata topology before testing its join."""
        from bench import quality_adaptation_scale as scale
        bridge = k.read_json(Path(r["identity_bridge"]["path"]))
        entry = next(row for row in bridge["rows"] if row["id"] == ident)
        path = Path(entry["input_provenance"]["path"])
        k.atomic_json(path, bundle)
        entry["input_provenance"] = self.descriptor(path)
        inventory = {"records": [{key: value for key, value in row.items() if key != "split"}
            for row in plan["global_manifest"]["records"]],
            "label_policy": plan["global_manifest"]["split_manifest"]["label_policy"],
            "background_catalog": plan["global_manifest"]["split_manifest"]["background_catalog"]}
        next(row for row in inventory["records"] if row["id"] == ident)["inputs_sha256"] = entry["input_provenance"]["sha256"]
        updated = scale.plan_scale(inventory, seed=plan["global_manifest"]["seed"], shard_size=2,
            targets=[row["train_states_requested"] for row in plan["target_requests"]])
        plan_path = Path(r["global_inventory"]["path"])
        k.atomic_json(plan_path, updated)
        r["global_inventory"] = self.descriptor(plan_path)
        bridge.update(plan_sha256=updated["plan_sha256"], manifest_sha256=updated["global_manifest"]["manifest_sha256"])
        bridge["bridge_sha256"] = k.canonical({key: value for key, value in bridge.items() if key != "bridge_sha256"})
        bridge_path = Path(r["identity_bridge"]["path"])
        k.atomic_json(bridge_path, bridge)
        r["identity_bridge"] = self.descriptor(bridge_path)

    def label_overlap_control(self, *, hardlink):
        r, plan = self.metadata_fixture()
        bridge = k.read_json(Path(r["identity_bridge"]["path"]))
        entries = {row["id"]: row["input_provenance"] for row in bridge["rows"]}
        active = next(row["id"] for row in plan["global_manifest"]["records"] if row["split"] == "train")
        sealed = next(row["id"] for row in plan["global_manifest"]["records"] if row["split"] == "test")
        active_bundle = k.read_json(Path(entries[active]["path"]))
        sealed_bundle = k.read_json(Path(entries[sealed]["path"]))
        descriptor = deepcopy(active_bundle["labels"])
        alias_context = nullcontext()
        if hardlink:
            alias = self.root / "input/held-out-label-alias-control.npz"
            alias_context = self.file_alias_control(Path(descriptor["path"]), alias)
            descriptor["path"] = str(alias)
        sealed_bundle["labels"] = descriptor
        self.update_frozen_provenance(r, plan, sealed, sealed_bundle)
        original = k.verify_descriptor
        def metadata_only(descriptor, roots):
            if Path(descriptor["path"]).suffix not in (".json", ".py"):
                raise AssertionError("payload hash attempted before held-out overlap rejection")
            return original(descriptor, roots)
        expected = "physically aliases a sealed payload" if hardlink else "overlaps a sealed payload path"
        with alias_context, patch.object(k, "verify_descriptor", side_effect=metadata_only):
            for stage in ("extract", "assemble", "preflight", "train", "resume", "export"):
                with self.subTest(stage=stage), self.assertRaisesRegex(k.ContractError, expected):
                    k.metadata_check(r, stage, REPO)

    def test_shared_active_and_sealed_label_path_rejects_before_payload_hash(self):
        self.label_overlap_control(hardlink=False)

    def test_shared_active_and_sealed_label_hardlink_rejects_before_payload_hash(self):
        self.label_overlap_control(hardlink=True)

    def file_alias_control(self, source, alias):
        """Real links on Linux; explicit inode control when Windows denies links."""
        try:
            os.link(source, alias)
            return nullcontext()
        except PermissionError:
            # The Windows workspace denies even owned hardlinks (WinError5).
            # This tests the stat-identity branch, not native link creation.
            source_info = source.stat()
            alias.write_bytes(b"manufactured inode-metadata control; no annotation array")
            original = Path.stat
            def same_inode(path, *args, **kwargs):
                return source_info if path == alias else original(path, *args, **kwargs)
            return patch.object(Path, "stat", same_inode)

    def test_request_input_hardlink_to_sealed_metadata_control_rejects_without_reads(self):
        sealed = self.root / "held-out-metadata-control.bin"
        sealed.write_bytes(b"plain manufactured control bytes, not an annotation array")
        requested = self.root / "request-alias.bin"
        alias_context = self.file_alias_control(sealed, requested)
        with alias_context, patch.object(k, "file_sha256", side_effect=AssertionError("payload content read")):
            with self.assertRaisesRegex(k.ContractError, "physically aliases"):
                k.reject_sealed_payload_aliases(set(), {str(sealed)}, {str(requested)})

    def test_extract_defers_original_label_bytes_but_assembly_requires_them(self):
        r, plan = self.metadata_fixture()
        bridge = k.read_json(Path(r["identity_bridge"]["path"]))
        active = next(row["id"] for row in plan["global_manifest"]["records"] if row["split"] == "train")
        entry = next(row for row in bridge["rows"] if row["id"] == active)
        bundle = k.read_json(Path(entry["input_provenance"]["path"]))
        labels = Path(bundle["labels"]["path"])
        labels.unlink()  # Only this owned manufactured control; no real arrays.
        result = k.metadata_check(r, "extract", REPO)
        self.assertFalse(result["sealed_payloads_opened"])
        with self.assertRaisesRegex(k.ContractError, "missing or changed file identity"):
            k.metadata_check(r, "assemble", REPO)

    def test_prepared_mode_reads_immutable_input_dataset_without_raw_source_copy(self):
        r, plan = self.metadata_fixture()
        source = Path(r["paths"]["extraction_dir"])
        target = Path(r["input_root"]) / "prepared"
        shutil.copytree(source, target)
        r["mode"] = "prepared"
        r["paths"]["extraction_dir"] = str(target)
        r["paths"]["manifest_output"] = str(target / "manifest.json")
        r["dataset"] = self.descriptor(target / "manifest.json")
        r["preparation"]["output_root"] = str(target)
        # Original source/crop descriptors remain frozen provenance metadata.
        # Head training uses prepared features/labels; no raw source reacquisition.
        for path in Path(r["input_root"]).iterdir():
            if path.is_file() and path.suffix != ".json":
                path.unlink()
        roots = {"input_root": Path(r["input_root"]), "output_root": Path(r["output_root"])}
        k.validate_stage_paths(r, roots, "prepared")
        result = k.metadata_check(r, "train", REPO)
        self.assertTrue(result["prepared_admission"]["admitted"])
        with self.assertRaisesRegex(k.ContractError, "does not extract"):
            k.metadata_check(r, "extract", REPO)
        with self.assertRaisesRegex(k.ContractError, "escapes explicit root"):
            k.validate_stage_paths(r, roots, "extract")

    def test_metadata_failures_preserved_and_block_training(self):
        r, plan = self.metadata_fixture()
        extraction = Path(r["paths"]["extraction_dir"])
        index_path = extraction / "index.json"
        index = k.read_json(index_path)
        failed_id = index["rows"][0]["id"]
        index["rows"][0] = {"id": failed_id, "status": "unavailable", "failure_reason": "actual extraction failure control"}
        k.atomic_json(index_path, index)
        for key in ("labels", "dataset"):
            path = Path(r[key]["path"])
            value = k.read_json(path)
            for row in value["rows"]:
                if row["id"] == failed_id:
                    row.pop("features", None)
                    row.pop("labels", None)
                    row.update(status="unavailable", failure_reason="actual extraction failure control")
            k.atomic_json(path, value)
            r[key] = self.descriptor(path)
        for i, ref in enumerate(r["preparation"]["receipts"]):
            path = Path(ref["path"])
            receipt = k.read_json(path)
            for row in receipt["rows"]:
                if row["id"] == failed_id:
                    row.pop("output", None)
                    row.update(status="failed", failure_reason="actual extraction failure control")
            k.atomic_json(path, receipt)
            r["preparation"]["receipts"][i] = self.descriptor(path)
        self.assertFalse(k.metadata_check(r, "preflight", REPO)["sealed_payloads_opened"])
        with self.assertRaisesRegex(k.ContractError, "recomputed original receipts"):
            k.metadata_check(r, "train", REPO)
        self.assertEqual(k.read_json(index_path)["rows"][0]["status"], "unavailable")

    def test_test_id_in_original_receipt_blocks_before_any_payload_hash(self):
        r, plan = self.metadata_fixture()
        test_id = next(row["id"] for row in plan["global_manifest"]["records"] if row["split"] == "test")
        ref = r["preparation"]["receipts"][0]
        path = Path(ref["path"])
        receipt = k.read_json(path)
        receipt["rows"][0]["id"] = test_id
        k.atomic_json(path, receipt)
        r["preparation"]["receipts"][0] = self.descriptor(path)
        original = k.verify_descriptor
        def metadata_only(descriptor, roots):
            if Path(descriptor["path"]).suffix not in (".json", ".py"):
                raise AssertionError("array opened before receipt membership rejection")
            return original(descriptor, roots)
        with patch.object(k, "verify_descriptor", side_effect=metadata_only):
            with self.assertRaisesRegex(k.ContractError, "shard membership"):
                k.metadata_check(r, "train", REPO)

    def test_metadata_join_blocks_test_relabel_before_any_payload_hash(self):
        r, plan = self.metadata_fixture()
        test_id = next(row["id"] for row in plan["global_manifest"]["records"] if row["split"] == "test")
        dataset_path = Path(r["dataset"]["path"])
        dataset = k.read_json(dataset_path)
        next(row for row in dataset["rows"] if row["id"] == test_id)["split"] = "train"
        k.atomic_json(dataset_path, dataset)
        r["dataset"] = self.descriptor(dataset_path)
        original = k.verify_descriptor
        def metadata_only(descriptor, roots):
            if Path(descriptor["path"]).suffix not in (".json", ".py"):
                raise AssertionError("payload opened before metadata rejection")
            return original(descriptor, roots)
        with patch.object(k, "verify_descriptor", side_effect=metadata_only):
            with self.assertRaisesRegex(k.ContractError, "dataset row/split"):
                k.metadata_check(r, "train", REPO)

    def test_metadata_join_rejects_changed_crops_source_bridge_and_supplement(self):
        r, plan = self.metadata_fixture()
        crops_path = Path(r["crop_manifest"]["path"])
        crops = k.read_json(crops_path)
        crops["rows"][0]["crops"]["sha256"] = "0" * 64
        k.atomic_json(crops_path, crops)
        r["crop_manifest"] = self.descriptor(crops_path)
        with self.assertRaisesRegex(k.ContractError, "actual crop descriptor"):
            k.metadata_check(r, "extract", REPO)
        r, plan = self.metadata_fixture_reset()
        bridge_path = Path(r["identity_bridge"]["path"])
        bridge = k.read_json(bridge_path)
        bridge["rows"][0]["input_provenance"]["sha256"] = "0" * 64
        bridge["bridge_sha256"] = k.canonical({key: value for key, value in bridge.items() if key != "bridge_sha256"})
        k.atomic_json(bridge_path, bridge)
        r["identity_bridge"] = self.descriptor(bridge_path)
        with self.assertRaisesRegex(k.ContractError, "original frozen inputs"):
            k.metadata_check(r, "extract", REPO)
        changed = deepcopy(r)
        changed["scale_sources"]["quality_adaptation_scale"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(k.ContractError, "validator descriptor changed"):
            k.metadata_check(changed, "extract", REPO)

    def metadata_fixture_reset(self):
        # Give a second isolated metadata control its own fresh workspace tree.
        self.temporary.cleanup()
        self.setUp()
        return self.metadata_fixture()

    def test_resealed_wrong_global_split_is_rejected_by_original_constructor(self):
        r, plan = self.metadata_fixture()
        plan["global_manifest"]["records"][0]["split"] = "train" if plan["global_manifest"]["records"][0]["split"] != "train" else "test"
        manifest = plan["global_manifest"]
        manifest["manifest_sha256"] = k.canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        plan["plan_sha256"] = k.canonical({key: value for key, value in plan.items() if key != "plan_sha256"})
        path = self.write("plan.json", plan)
        r["global_inventory"] = self.descriptor(path)
        with self.assertRaisesRegex(ValueError, "metadata disagrees"):
            k.metadata_check(r, "extract", REPO)

    def test_metadata_worker_uses_explicit_isolated_interpreter_and_surfaces_blocker(self):
        r = self.request()
        with patch.object(k.subprocess, "run", return_value=Mock(returncode=0, stdout='{"rows":3}')) as child:
            self.assertEqual(k.run_metadata_check(r, "train", REPO), {"rows": 3})
            argv = child.call_args.args[0]
            self.assertEqual(argv[:3], [sys.executable, "-I", "-B"])
            self.assertEqual(json.loads(child.call_args.kwargs["input"]), r)
        with patch.object(k.subprocess, "run", return_value=Mock(returncode=2, stderr="frozen split changed")):
            with self.assertRaisesRegex(k.ContractError, "frozen split changed"):
                k.run_metadata_check(r, "train", REPO)

    def test_authenticated_runtime_links_account_once_and_reject_external_or_new_links(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        target = runtime / "managed-python"
        target.write_bytes(b"1234567")
        alias = runtime / "python"
        alias.write_bytes(b"simulated link, no Windows symlink privilege")
        original_link, original_resolve = Path.is_symlink, Path.resolve
        def is_link(path): return path == alias or original_link(path)
        def resolve(path, *args, **kwargs): return target if path == alias else original_resolve(path, *args, **kwargs)
        receipt = {"runtime_root": str(runtime), "runtime_links": [{"path": "python", "target": "managed-python"}]}
        with patch.object(Path, "is_symlink", is_link), patch.object(Path, "resolve", resolve), patch.object(k.os, "readlink", return_value="managed-python"):
            policy = k.runtime_link_policy(receipt, self.root)
            self.assertEqual(k.unique_tree_bytes((runtime, target), approved_links=policy), 7)
            with self.assertRaisesRegex(k.ContractError, "unapproved"):
                k.unique_tree_bytes((runtime,))
            with self.assertRaisesRegex(k.ContractError, "unapproved runtime links"):
                k.runtime_link_policy({"runtime_root": str(runtime), "runtime_links": []}, self.root)
        with patch.object(Path, "is_symlink", is_link), patch.object(k.os, "readlink", return_value="../external"):
            escaping = {"runtime_root": str(runtime), "runtime_links": [{"path": "python", "target": "../external"}]}
            with self.assertRaisesRegex(k.ContractError, "escapes"):
                k.runtime_link_policy(escaping, self.root)
        with patch.object(Path, "is_symlink", is_link), patch.object(k.os, "readlink", return_value="managed-python"), patch.object(Path, "resolve", side_effect=RuntimeError("cycle")):
            with self.assertRaisesRegex(k.ContractError, "broken/cyclic"):
                k.runtime_link_policy(receipt, self.root)

    def status_control(self, status="completed", steps=3):
        r = self.request()
        output = Path(r["output_root"])
        train = Path(r["paths"]["train_dir"])
        train.mkdir(parents=True)
        value = {"status": status, "steps": steps, "epoch": 0, "elapsed_seconds": .5,
                 "quality_claim": False, "test_arrays_opened": False}
        path = train / "status.json"
        path.write_text(json.dumps(value))
        state = {"status": status, "identity": k.identity_document(r, {}),
            "stages": [{"stage": "train", "status": status,
                "started_at": (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),
                "trainer_source_sha256": "a"*64}]}
        k.atomic_json(output / "adapter-state.json", state)
        return output, path, state

    def test_status_reports_actual_steps_source_and_staleness_without_checkpoint_reads(self):
        output, path, state = self.status_control()
        result = k.read_status(output)
        progress = result["optimizer_progress"]
        self.assertEqual(progress["last_confirmed_steps"], 3)
        self.assertTrue(progress["completion_confirmed"])
        self.assertFalse(progress["current_progress_known"])
        self.assertEqual(progress["source"]["sha256"], k.file_sha256(path))
        old = datetime.now(timezone.utc).timestamp()-500
        os.utime(path, (old, old))
        result = k.read_status(output)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["optimizer_progress"]["completion_confirmed"])

    def test_status_incomplete_zero_missing_invalid_and_crash(self):
        output, path, state = self.status_control("incomplete", 0)
        self.assertEqual(k.read_status(output)["optimizer_progress"]["last_confirmed_steps"], 0)
        path.unlink()
        self.assertIsNone(k.read_status(output)["optimizer_progress"]["last_confirmed_steps"])
        path.write_text('{"steps":true}')
        self.assertEqual(k.read_status(output)["optimizer_progress"]["freshness"], "invalid")
        path.write_text('invalid JSON')
        self.assertEqual(k.read_status(output)["optimizer_progress"]["freshness"], "invalid")
        path.write_text(json.dumps({"status": "running", "steps": 2, "epoch": 0, "elapsed_seconds": 1}))
        state["status"] = "running"
        state["stages"][-1]["status"] = "running"
        k.atomic_json(output / "adapter-state.json", state)
        self.assertEqual(k.read_status(output)["optimizer_progress"]["freshness"], "adapter_liveness_unconfirmed")

    def test_status_atomic_replacement_keeps_steps_hash_and_asof_one_observation(self):
        output, path, state = self.status_control()
        original_bytes = path.read_bytes()
        original = k.parse_json_bytes
        replaced = []
        def replace_after_read(data, filename):
            if filename == path and not replaced:
                replacement = json.loads(data)
                replacement["steps"] = 4
                k.atomic_json(path, replacement)
                replaced.append(True)
            return original(data, filename)
        with patch.object(k, "parse_json_bytes", side_effect=replace_after_read):
            result = k.read_status(output)
        progress = result["optimizer_progress"]
        self.assertEqual(progress["last_confirmed_steps"], 3)
        self.assertEqual(progress["source"]["sha256"], k.digest(original_bytes))
        self.assertNotEqual(progress["source"]["sha256"], k.file_sha256(path))

    def test_failed_or_changed_trainer_status_does_not_claim_completion(self):
        output, path, state = self.status_control()
        state["stages"][-1]["status"] = "failed"
        state["status"] = "failed"
        k.atomic_json(output / "adapter-state.json", state)
        result = k.read_status(output)
        self.assertFalse(result["optimizer_progress"]["completion_confirmed"])
        self.assertEqual(result["optimizer_progress"]["freshness"], "failed_invocation")
        state["optimizer_progress"] = result["optimizer_progress"]
        state["stages"][-1]["status"] = "completed"
        k.atomic_json(output / "adapter-state.json", state)
        value = k.read_json(path)
        value["steps"] = 999
        k.atomic_json(path, value)
        result = k.read_status(output)
        self.assertEqual(result["optimizer_progress"]["last_confirmed_steps"], 3)
        self.assertEqual(result["optimizer_progress"]["freshness"], "changed_after_invocation")

    def test_status_invalid_external_path_rejected_before_read(self):
        output, path, state = self.status_control()
        state["identity"]["paths"]["train_dir"] = str(self.root / "outside")
        k.atomic_json(output / "adapter-state.json", state)
        original = k.read_json
        opened = []
        def record(path): opened.append(str(path));return original(path)
        with patch.object(k, "read_json", side_effect=record):
            with self.assertRaisesRegex(k.ContractError, "escapes"):
                k.read_status(output)
        self.assertEqual(opened, [str(output / "adapter-state.json")])

    def test_archive_sha_corruption_blocks_before_any_destination_write(self):
        archive, spec = self.archive()
        raw = bytearray(archive.read_bytes())
        raw[len(raw) // 2] ^= 1
        archive.write_bytes(raw)
        target = self.root / "never-created"
        with self.assertRaisesRegex(k.ContractError, "SHA-256"):
            k.install_source(archive, target, spec=spec)
        self.assertFalse(target.exists())

    def test_unsafe_duplicate_extra_and_missing_members_block_before_writes(self):
        mutations = [lambda m: m + [("../escape", b"x")],
                     lambda m: m + [("/absolute", b"x")],
                     lambda m: m + [("root\\escape", b"x")],
                     lambda m: m + [("root/./escape", b"x")],
                     lambda m: m + [("C:/escape", b"x")],
                     lambda m: m + [m[0]], lambda m: m + [("root/extra", b"x")],
                     lambda m: m[1:], lambda m: [(m[0][0], b"changed"), m[1]]]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                archive, spec = self.archive(mutation)
                target = self.root / f"never-{index}"
                with self.assertRaises(k.ContractError):
                    k.install_source(archive, target, spec=spec)
                self.assertFalse(target.exists())

    def test_symlink_archive_member_is_rejected_before_writes(self):
        archive, spec = self.archive()
        with zipfile.ZipFile(archive, "a") as z:
            link = zipfile.ZipInfo("root/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(link, "file.txt")
        spec = k.ArchiveSpec(k.file_sha256(archive), archive.stat().st_size, spec.roots)
        target = self.root / "never-link"
        with self.assertRaisesRegex(k.ContractError, "nonregular"):
            k.install_source(archive, target, spec=spec)
        self.assertFalse(target.exists())

    def test_containment_and_symlink_parent_fail(self):
        with self.assertRaisesRegex(k.ContractError, "escapes"):
            k.contained(self.root / ".." / "outside", self.root)
        marker = self.root / "linked"
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda p: p == marker or original(p)):
            with self.assertRaisesRegex(k.ContractError, "symlink path component"):
                k.no_symlinks(marker / "child" / "future-file")

    def test_shared_filesystem_and_nested_roots_count_once(self):
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "file").write_bytes(b"1234567")
        self.assertEqual(k.unique_tree_bytes((self.root, nested)), 7)
        disk = Mock(free=12345)
        with patch.object(k.shutil, "disk_usage", return_value=disk):
            self.assertEqual(len(k.free_by_filesystem((self.root, nested))), 1)
            self.assertEqual(list(k.free_by_filesystem((self.root, nested)).values()), [12345])

    def test_exact_existing_cli_contracts(self):
        r = self.request()
        base = [sys.executable, "-E", "-s", "-B", "-m"]
        self.assertEqual(k.command("extract", r), base + ["bench.quality_adaptation_prepare", "--manifest", "crops.json",
            "--output", r["paths"]["extraction_dir"], "--device", "cuda", "--max-seconds", "1",
            "--max-output-bytes", "1024", "--max-rows", "2", "--cloud-execution"])
        self.assertEqual(k.command("assemble", r), base + ["bench.quality_adaptation_manifest", "--extraction",
            r["paths"]["extraction_dir"], "--labels", "labels.json", "--output", r["paths"]["manifest_output"]])
        r["dataset"] = {"path": r["paths"]["manifest_output"]}
        for stage in ("preflight", "train", "resume"):
            expected = base + ["bench.quality_adaptation_train", stage, "--manifest", r["dataset"]["path"],
                               "--config", r["config"]["path"]]
            if stage != "preflight":
                expected += ["--output", r["paths"]["train_dir"], "--cloud-execution"]
            self.assertEqual(k.command(stage, r), expected)
        self.assertEqual(k.command("export", r), base + ["bench.quality_adaptation_train", "export", "--run",
            r["paths"]["train_dir"], "--output", r["paths"]["export_output"], "--cloud-execution"])

    def test_mocked_stage_order_checkpoint_identity_and_incomplete_resume(self):
        r = self.request()
        calls = []
        def child(argv, **kwargs):
            stage = calls_stage[0]
            calls.append(stage)
            p = r["paths"]
            if stage == "extract":
                Path(p["extraction_dir"]).mkdir()
            elif stage == "assemble":
                Path(p["manifest_output"]).write_text('{"mock_subprocess_output":true}')
            elif stage in ("train", "resume"):
                train = Path(p["train_dir"])
                train.mkdir(exist_ok=True)
                (train / "last.pt").write_bytes(b"mock checkpoint bytes; not a Torch archive")
                (train / "best.pt").write_bytes(b"mock best checkpoint bytes")
                (train / "status.json").write_text(json.dumps({"status": "incomplete" if stage == "train" else "completed",
                    "steps": 0 if stage == "train" else 2, "epoch": 0, "elapsed_seconds": 0.1,
                    "quality_claim": False, "test_arrays_opened": False}))
            elif stage == "export":
                Path(p["export_output"]).write_bytes(b"mock export bytes")
            return {"status": "incomplete" if stage == "train" else "completed", "returncode": 3 if stage == "train" else 0}
        calls_stage = [None]
        with patch.object(k, "validate_request", side_effect=self.validation), patch.object(k, "supervise", side_effect=child):
            for stage in k.STAGES:
                calls_stage[0] = stage
                if stage == "preflight":
                    r["dataset"] = self.descriptor(Path(r["paths"]["manifest_output"]))
                result = self.invoke(stage, r)
                self.assertEqual(result["status"], "incomplete" if stage == "train" else "completed")
        self.assertEqual(calls, list(k.STAGES))
        state = k.read_json(Path(r["output_root"]) / "adapter-state.json")
        self.assertFalse(state["quality_claim"])
        self.assertFalse(state["test_arrays_opened"])
        self.assertEqual(state["dataset"], r["dataset"])
        self.assertIn("last.pt", state["checkpoint_files"])

    def test_missing_or_changed_request_identity_blocks_before_child(self):
        path = self.write("request.json", self.request())
        with patch.object(k, "supervise") as child:
            with self.assertRaisesRegex(k.ContractError, "request SHA"):
                k.execute("train", path, "0" * 64)
            child.assert_not_called()
        with patch.object(k, "supervise") as child:
            with self.assertRaisesRegex(k.ContractError, "schema"):
                k.validate_request({}, "train")
            child.assert_not_called()

    def test_config_global_inventory_runtime_interpreter_inputs_and_source_change_identity(self):
        r = self.request()
        runtime = {"python": k.PYTHON_VERSION, "executable_sha256": "a" * 64}
        identity = k.canonical(k.identity_document(r, runtime))
        mutations = []
        for key in ("config", "global_inventory", "runtime", "crop_manifest", "labels"):
            changed = deepcopy(r)
            changed[key]["sha256"] = "9" * 64
            mutations.append((changed, runtime))
        changed = deepcopy(r)
        changed["python"] = "/different/python"
        mutations.append((changed, runtime))
        mutations.append((r, {**runtime, "executable_sha256": "b" * 64}))
        for changed, changed_runtime in mutations:
            with self.subTest(change=changed):
                self.assertNotEqual(identity, k.canonical(k.identity_document(changed, changed_runtime)))
        with patch.object(k, "ARCHIVE_SHA256", "1" * 64):
            self.assertNotEqual(identity, k.canonical(k.identity_document(r, runtime)))

    def test_resume_rejects_changed_dataset_and_checkpoint_before_child(self):
        r = self.request()
        output = Path(r["output_root"])
        output.mkdir()
        dataset_path = Path(r["paths"]["manifest_output"])
        dataset_path.parent.mkdir()
        dataset_path.write_bytes(b"{}")
        r["dataset"] = self.descriptor(dataset_path)
        identity = self.validation(r, "resume")[2]
        train = Path(r["paths"]["train_dir"])
        train.mkdir()
        checkpoint = train / "last.pt"
        checkpoint.write_bytes(b"original checkpoint")
        state = {"identity_sha256": identity, "dataset": r["dataset"], "training_started": True,
            "stages": [{"stage": "preflight", "status": "completed"}],
            "checkpoint_files": {"last.pt": self.descriptor(checkpoint)}}
        k.atomic_json(output / "adapter-state.json", state)
        with patch.object(k, "validate_request", side_effect=self.validation), patch.object(k, "supervise") as child:
            changed = deepcopy(r)
            changed["dataset"]["sha256"] = "7" * 64
            with self.assertRaisesRegex(k.ContractError, "global dataset changed"):
                self.invoke("resume", changed)
            checkpoint.write_bytes(b"altered checkpoint")
            with self.assertRaisesRegex(k.ContractError, "changed file identity"):
                self.invoke("resume", r)
            child.assert_not_called()

    def test_preflight_required_before_train_and_export(self):
        r = self.request()
        r["dataset"] = {"path": r["paths"]["manifest_output"], "bytes": 2, "sha256": "f" * 64}
        with patch.object(k, "validate_request", side_effect=self.validation), patch.object(k, "supervise") as child:
            for stage in ("train", "export"):
                with self.assertRaisesRegex(k.ContractError, "preflight required"):
                    self.invoke(stage, r)
            child.assert_not_called()

    def test_changed_config_inventory_interpreter_or_runtime_reject_resume_before_child(self):
        r = self.request()
        output = Path(r["output_root"])
        output.mkdir()
        state = {"identity_sha256": self.validation(r, "resume")[2], "stages": []}
        k.atomic_json(output / "adapter-state.json", state)
        with patch.object(k, "validate_request", side_effect=self.validation), patch.object(k, "supervise") as child:
            for key in ("config", "global_inventory", "runtime"):
                changed = deepcopy(r)
                changed[key]["sha256"] = "0" * 64
                with self.assertRaisesRegex(k.ContractError, "run identity changed"):
                    self.invoke("resume", changed)
            changed = deepcopy(r)
            changed["python"] = "/different/python"
            with self.assertRaisesRegex(k.ContractError, "run identity changed"):
                self.invoke("resume", changed)
            child.assert_not_called()

    def test_missing_authentic_checkpoint_blocks_before_neural_child(self):
        self.write("bench/runtime/adaptation-sources.json", {"gotrack": {"files": {}}})
        with patch.object(k.subprocess, "Popen") as child:
            with self.assertRaisesRegex(k.ContractError, "authentic pinned GoTrack checkpoint"):
                k.verify_neural_assets(self.root, [])
            child.assert_not_called()

    def test_runtime_wrong_patch_and_cpu_wheel_block(self):
        observed = {"python": k.PYTHON_VERSION, "prefix": "venv", "base_prefix": "managed",
                    "executable_sha256": k.file_sha256(Path(sys.executable)),
                    "packages": {"torch": "2.5.1+cpu"}}
        receipt = {"schema": "kaggle_isolated_runtime_v1", "uv_version": k.UV_VERSION, "runtime": observed}
        with patch.object(k, "runtime_probe", return_value=observed):
            with self.assertRaisesRegex(k.ContractError, "torch==2.5.1"):
                k.verify_runtime(Path(sys.executable), receipt, self.root, (self.root,))
        changed = {**observed, "python": "3.13.15"}
        with patch.object(k, "runtime_probe", return_value=changed):
            with self.assertRaisesRegex(k.ContractError, "runtime identity"):
                k.verify_runtime(Path(sys.executable), receipt, self.root, (self.root,))

    def test_changed_interpreter_binary_rejected_before_probe(self):
        with patch.object(k, "runtime_probe") as probe:
            with self.assertRaisesRegex(k.ContractError, "interpreter executable identity changed"):
                k.verify_runtime(Path(sys.executable), {"runtime": {"executable_sha256": "0" * 64}}, self.root, (self.root,))
            probe.assert_not_called()

    def test_non_kaggle_and_missing_parity_block_before_neural(self):
        r = self.request()
        with patch.object(k.platform, "system", return_value="Windows"), patch.object(k.subprocess, "Popen") as child:
            with self.assertRaisesRegex(k.ContractError, "actual Kaggle"):
                k.verify_admission({}, "train", r, "identity", r["limits"], self.root)
            child.assert_not_called()
        now = datetime.now(timezone.utc)
        evidence = {"schema": "kaggle_execution_evidence_v1", "provider": "kaggle", "stage": "train",
            "identity_sha256": "identity", "session_id": "actual-session-id", "evidence_source": "operator capture",
            "notebook_url": "https://www.kaggle.com/code/achitaan/example",
            "observed_at": (now - timedelta(seconds=5)).isoformat(), "expires_at": (now + timedelta(seconds=60)).isoformat(),
            "quota": {"paid": False, "gpu_enabled": True, "gpu_remaining_seconds": 900, "session_remaining_seconds": 900},
            "gpu": {"uuid": "GPU-example", "name": "Tesla T4"}, "parity_passed": False}
        with patch.object(k.platform, "system", return_value="Linux"), patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}), patch.object(Path, "is_dir", return_value=True), patch.object(k.subprocess, "run", return_value=Mock(stdout="GPU-example, Tesla T4\n")):
            with self.assertRaisesRegex(k.ContractError, "parity"):
                k.verify_admission(evidence, "train", r, "identity", r["limits"], self.root)

    def supervise_child(self, code, **overrides):
        output = self.root / "process-output"
        output.mkdir(exist_ok=True)
        limits = {"max_seconds": 5, "max_rss_bytes": 8 * k.GIB, "max_output_bytes": k.GIB,
                  "max_log_bytes": 1024, "reserve_bytes": k.GIB}
        limits.update(overrides)
        # Isolate process semantics from unrelated concurrent disk occupancy.
        # Production admission/reserve checks are not weakened. Test their failure
        # separately below; no large sparse/real files are used to force pressure.
        with patch.object(k, "free_by_filesystem", return_value={1: 2 * k.GIB}):
            return k.supervise([sys.executable, "-I", "-B", "-c", code], cwd=self.root,
                output_root=output, working=self.root, log_path=output / "child.log", limits=limits, env=k.clean_environment())

    def test_reserve_failure_terminates_child_and_preserves_incomplete_receipt(self):
        output = self.root / "reserve-output"
        output.mkdir()
        limits = self.request()["limits"]
        with patch.object(k, "free_by_filesystem", return_value={1: 0}):
            result = k.supervise([sys.executable, "-I", "-c", "import time;time.sleep(30)"],
                cwd=self.root, output_root=output, working=self.root, log_path=output / "log",
                limits=limits, env=k.clean_environment())
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "filesystem_reserve_exhausted")
        self.assertIsNotNone(result["returncode"])

    def test_real_success_subprocess_and_log(self):
        result = self.supervise_child("print('model-free subprocess control')")
        self.assertEqual(result["status"], "completed", result)
        self.assertIn("model-free subprocess control", Path(result["log"]).read_text())

    def test_real_failure_exit_seven_is_failed(self):
        result = self.supervise_child("import sys;print('failure retained');sys.exit(7)")
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["returncode"], 7)
        self.assertTrue(result["outputs_preserved"])

    def test_real_exit_three_is_incomplete_never_success(self):
        result = self.supervise_child("import sys;sys.exit(3)")
        self.assertEqual(result["status"], "incomplete", result)
        self.assertEqual(result["reason"], "child_reported_incomplete")

    def test_real_timeout_cleanup_and_log_cap(self):
        result = self.supervise_child("import time;time.sleep(30)", max_seconds=0.2)
        self.assertEqual(result["status"], "incomplete", result)
        self.assertEqual(result["reason"], "deadline_exceeded")
        self.assertIsNotNone(result["returncode"])
        Path(result["log"]).unlink()
        result = self.supervise_child("import sys;sys.stdout.write('x'*1000000)", max_log_bytes=16)
        self.assertEqual(result["status"], "incomplete", result)
        self.assertEqual(Path(result["log"]).stat().st_size, 16)

    def test_interruption_and_rss_failure_cleanup(self):
        with patch.object(k, "group_rss", side_effect=KeyboardInterrupt):
            result = self.supervise_child("import time;time.sleep(30)")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "interrupted")
        self.assertIsNotNone(result["returncode"])
        Path(result["log"]).unlink()
        with patch.object(k, "group_rss", return_value=16 * k.GIB):
            result = self.supervise_child("import time;time.sleep(30)")
        self.assertEqual(result["reason"], "process_group_rss_exceeded")
        self.assertIsNotNone(result["returncode"])

    def test_group_cleanup_kills_descendants_after_leader_exit(self):
        child = Mock(pid=12345, returncode=0)
        child.wait.return_value = 0
        with patch.object(k.os, "name", "posix"), patch.object(k.os, "killpg", create=True) as kill:
            k.terminate_group(child, grace_seconds=0)
            self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
            self.assertEqual(kill.call_count, 2)

    def test_sanitized_environment_and_metadata_probe_no_model_import(self):
        with patch.dict(os.environ, {"PYTHONPATH": "untrusted", "PYTHONHOME": "untrusted", "PYTHONUSERBASE": "untrusted", "LD_PRELOAD": "untrusted", "VIRTUAL_ENV": "other"}):
            env = k.clean_environment()
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTHONHOME", env)
            self.assertNotIn("PYTHONUSERBASE", env)
            self.assertNotIn("LD_PRELOAD", env)
            self.assertNotIn("VIRTUAL_ENV", env)
        observed = k.runtime_probe(Path(sys.executable))
        self.assertEqual(observed["executable_sha256"], k.file_sha256(Path(sys.executable)))
        self.assertIn("packages", observed)
        self.assertNotIn("torch", sys.modules)

    def test_help_and_status_are_stdlib_only_and_do_not_create_output(self):
        launcher = REPO / "scripts/kaggle_adaptation.py"
        help_result = subprocess.run([sys.executable, "-I", "-B", str(launcher), "--help"], capture_output=True, text=True, check=True)
        self.assertIn("install-source", help_result.stdout)
        missing = self.root / "never-started"
        result = subprocess.run([sys.executable, "-I", "-B", str(launcher), "status", "--output-root", str(missing)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(json.loads(result.stdout)["status"], "not_started")
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
