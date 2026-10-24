"""CPU-only integrity, split, label-join, and analytic metric checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .quality_ranker_data import (
    _audit_split, _cache_manifest, _split_digest, adapt_bop_ycbv_manifests, audit_cache,
    freeze_split_manifest, load_cache_split, load_training_split,
    prepare_candidate_cache, read_split_manifest,
    rotation_error_degrees, translation_error_m, validate_split_manifest,
)
from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


POLICY = {"max_rotation_error_deg": 5., "max_translation_error_m": .05,
          "max_surface_error_m": None, "max_projection_error_px": None}


def _record(object_id, sequence_id, split, frame_id=0, source_frame_id=None, **extra):
    return {
        "physical_object_id": object_id,
        "object_id": object_id,
        "scene_id": f"scene-{sequence_id}",
        "sequence_id": sequence_id,
        "background_id": "procedural:gray",
        "background_kind": "procedural",
        "provenance_kind": "real",
        "frame_id": frame_id,
        "source_frame_id": frame_id if source_frame_id is None else source_frame_id,
        "split": split,
        **extra,
    }


def _inventory(*, train_frames=2):
    rows = []
    for split, object_id, sequence, count, base in (
        ("train", "object-train", "sequence-train", train_frames, 100),
        ("validation", "object-validation", "sequence-validation", 1, 200),
        ("test", "object-test", "sequence-test", 1, 300),
        ("dev", "object-dev", "sequence-dev", 1, 400),
    ):
        for ordinal in range(count):
            rows.append(_record(object_id, sequence, split, ordinal, base + ordinal))
    return {"records": rows, "label_policy": POLICY}


def _rz(degrees):
    radians = np.deg2rad(degrees)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(radians), -np.sin(radians), 0.],
                    [np.sin(radians), np.cos(radians), 0.], [0., 0., 1.]]
    return pose


def _candidate(candidate_id, pose, eligible=True, rejection=None, offset=0.):
    value = np.eye(4) if pose is None else np.asarray(pose, dtype=float).copy()
    if pose is not None:
        value[0, 3] += offset
    return {"candidate_id": candidate_id, "original_index": 0, "stage": "test",
            "source": "analytic", "pose": None if pose is None else value.tolist(),
            "eligible": eligible, "rejection": rejection,
            "validation_stats": {"inliers": 8}, "features": [i / 32 for i in range(len(FEATURE_NAMES))],
            "legacy_score": .1}


def _trace_and_labels(split_manifest):
    trace, labels = [], []
    for group in split_manifest["groups"]:
        for mapping in group["frames"]:
            frame_id = mapping["frame_id"]
            source_id = mapping["source_frame_id"]
            candidates = []
            if frame_id == 0:
                candidates = [
                    _candidate("good", np.eye(4)),
                    _candidate("bad", np.eye(4), offset=.1),
                    _candidate("rejected", np.eye(4), eligible=False,
                               rejection="validation_gate", offset=.2),
                    _candidate("no_pose", None, eligible=False, rejection="pose_missing"),
                ]
            trace_row = {
                "trace_schema_version": 1,
                "object_id": group["physical_object_id"],
                "physical_object_id": group["physical_object_id"],
                "scene_id": group["scene_id"], "sequence_id": group["sequence_id"],
                "background_id": group["background_id"],
                "background_kind": group["background_kind"],
                "provenance_kind": group["provenance_kind"],
                "frame_id": frame_id, "source_frame_id": source_id,
                "timestamp_s": float(frame_id), "clock_mode": "capture_seconds_v1",
                "context": {"object_id": group["physical_object_id"],
                            "physical_object_id": group["physical_object_id"],
                            "sequence_id": group["sequence_id"],
                            "scene_id": group["scene_id"],
                            "background_id": group["background_id"],
                            "background_kind": group["background_kind"],
                            "provenance_kind": group["provenance_kind"],
                            "recovery": False},
                "candidates": candidates,
                "legacy_choice_candidate_id": "bad" if candidates else None,
                "timings_ms": {"rank_decision": 1.25, "feature_extraction": .5},
                "resource_memory": {"host_peak_rss_bytes": 4096},
                "pose_state": "valid" if candidates else "lost",
                "render_state": "rendered" if candidates else "hidden",
                "failure_reason": None if candidates else "no_candidates",
                "cameraFromObject": None,
            }
            trace.append(trace_row)
            if candidates:
                labels.append({"object_id": group["physical_object_id"],
                               "sequence_id": group["sequence_id"],
                               "frame_id": frame_id, "source_frame_id": source_id,
                               "pose": np.eye(4).tolist(),
                               "observationally_ambiguous": False,
                               "label_kind": "analytic_unit_test_reference"})
    return trace, labels


class FrozenSplitTests(unittest.TestCase):
    def test_historical_and_inspected_object_aliases_are_dev_only(self):
        for identity, aliases in (
                ("keyboard", []), ("mug", []), ("ranch", []),
                ("bop-ycbv:000002", []), ("ycbv:000018", []),
                ("hb:000001", []), ("bop-hb:000010", []),
                ("new-mesh-id", ["bop-ycbv:000006"])):
            with self.subTest(identity=identity, aliases=aliases):
                with self.assertRaisesRegex(ValueError, "dev-only"):
                    freeze_split_manifest({"records": [_record(identity, "s", "train", aliases=aliases)],
                                           "label_policy": POLICY})

    def test_unknown_observed_background_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown observed background"):
            freeze_split_manifest({"records": [_record("obj", "seq", "train",
                                                          background_id="scene-observed",
                                                          background_kind="observed")],
                                   "label_policy": POLICY})

    def test_observed_scene_inventory_is_complete_and_atomic_across_splits(self):
        catalog = [{"background_id": "bg-scene", "scene_id": "scene-shared",
                    "inventory_complete": True, "physical_object_ids": ["object-a", "object-b"]}]
        records = [
            {**_record("object-a", "seq-a", "train", background_id="bg-scene",
                       background_kind="observed"), "scene_id": "scene-shared"},
            {**_record("object-b", "seq-b", "test", background_id="bg-scene",
                       background_kind="observed"), "scene_id": "scene-shared"},
        ]
        with self.assertRaisesRegex(ValueError, "scene_id|co-occurs|background_id|physical_object_id"):
            freeze_split_manifest({"records": records, "background_catalog": catalog,
                                   "label_policy": POLICY})
        with self.assertRaisesRegex(ValueError, "physical_object_ids"):
            freeze_split_manifest({"records": [records[0]],
                                   "background_catalog": [{**catalog[0],
                                       "inventory_complete": False}],
                                   "label_policy": POLICY})

    def test_observed_inventory_cannot_hide_dev_only_cooccurring_identity(self):
        record = {**_record("object-a", "sequence-a", "train",
                            background_id="bg-scene", background_kind="observed"),
                  "scene_id": "scene-shared"}
        catalog = [{"background_id": "bg-scene", "scene_id": "scene-shared",
                    "inventory_complete": True,
                    "physical_object_ids": ["object-a", "ycbv:000002"]}]
        with self.assertRaisesRegex(ValueError, "co-occurring physical identity.*dev-only"):
            freeze_split_manifest({"records": [record], "background_catalog": catalog,
                                   "label_policy": POLICY})

        manifest = freeze_split_manifest({"records": [record], "background_catalog": [{
            **catalog[0], "physical_object_ids": ["object-a", "object-c"]}],
            "label_policy": POLICY})
        manifest["background_catalog"][0]["physical_object_ids"].append("ycbv:000002")
        manifest["background_catalog"][0]["physical_object_ids"].sort()
        manifest["manifest_sha256"] = _split_digest(manifest)
        with self.assertRaisesRegex(ValueError, "co-occurring identity.*dev-only"):
            validate_split_manifest(manifest)

    def test_duplicate_source_identity_and_conflicting_assignments_rejected(self):
        duplicate = [_record("obj", "seq", "train", 0, 7),
                     _record("obj", "seq", "train", 1, 7)]
        with self.assertRaisesRegex(ValueError, "Duplicate frame/source"):
            freeze_split_manifest({"records": duplicate, "label_policy": POLICY})
        row = _record("obj", "seq", "train")
        with self.assertRaisesRegex(ValueError, "Conflicting split assignments"):
            freeze_split_manifest({"records": [row], "label_policy": POLICY}, {"seq": "test"})

    def test_mesh_alias_scene_and_sequence_leakage_rejected(self):
        digest = "a" * 64
        for pair in (
                [_record("same", "s1", "train"), _record("same", "s2", "test")],
                [_record("obj1", "same-sequence", "train"), _record("obj2", "same-sequence", "test")],
                [_record("obj1", "s1", "train", mesh_sha256=digest),
                 _record("obj2", "s2", "test", mesh_sha256=digest)],
                [_record("obj1", "s1", "train", aliases=["shared-physical-alias"]),
                 _record("obj2", "s2", "test", aliases=["shared-physical-alias"])],
        ):
            with self.subTest(records=pair):
                with self.assertRaises(ValueError):
                    freeze_split_manifest({"records": pair, "label_policy": POLICY})

    def test_synthetic_identity_requires_mesh_hash_and_procedural_can_be_reused(self):
        with self.assertRaisesRegex(ValueError, "mesh_sha256"):
            freeze_split_manifest({"records": [_record("synthetic-x", "s", "train",
                                                          provenance_kind="synthetic")],
                                   "label_policy": POLICY})
        records = [
            _record("obj-a", "s-a", "train", background_id="procedural:gray"),
            _record("obj-b", "s-b", "test", background_id="procedural:gray"),
        ]
        manifest = freeze_split_manifest({"records": records, "label_policy": POLICY})
        self.assertEqual({row["split"] for row in manifest["groups"]}, {"train", "test"})


class AnalyticLabelTests(unittest.TestCase):
    def test_rotation_and_translation_errors_match_known_transform(self):
        reference = np.eye(4)
        predicted = _rz(90)
        predicted[:3, 3] = [.1, -.2, .2]
        self.assertAlmostEqual(rotation_error_degrees(predicted, reference), 90.)
        self.assertAlmostEqual(translation_error_m(predicted, reference), np.sqrt(.09))

    def test_ambiguous_orientation_drops_forced_rotation_supervision(self):
        from .quality_ranker_data import _make_label
        frame = {"object_id": "obj", "sequence_id": "seq", "frame_id": 0,
                 "source_frame_id": 10, "split": "test", "provenance_kind": "real"}
        candidate = _candidate("turned", _rz(90))
        reference = {"pose": np.eye(4).tolist(), "observationally_ambiguous": True,
                     "label_kind": "symmetric_object_reference"}
        label = _make_label(candidate, frame, reference, POLICY)
        self.assertIsNone(label["rotation_error_deg"])
        self.assertFalse(label["orientation_supervision"])
        self.assertEqual(label["translation_error_m"], 0.)
        self.assertEqual(label["quality_target"], 1)

    def test_surface_error_is_optional_and_metric(self):
        from .quality_ranker_data import _surface_chamfer_p95
        reference = np.eye(4)
        predicted = np.eye(4)
        predicted[0, 3] = .1
        error = _surface_chamfer_p95(predicted, reference, [[0., 0., 0.]])
        self.assertAlmostEqual(error["p95_m"], .1)
        self.assertEqual(error["metric"], "symmetric_nearest_neighbor")


class CandidateCacheTests(unittest.TestCase):
    def test_features_labels_stay_separate_and_empty_frames_count(self):
        split = freeze_split_manifest(_inventory())
        trace, references = _trace_and_labels(split)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            cache = Path(temp) / "cache"
            manifest = prepare_candidate_cache(cache, split, trace, references,
                                               feature_names=list(FEATURE_NAMES),
                                               feature_schema_version=FEATURE_SCHEMA_VERSION)
            features_train = (cache / "features.train.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("quality_target", features_train)
            self.assertNotIn("reference_pose", features_train)
            labels_train = (cache / "labels.train.jsonl").read_text(encoding="utf-8")
            self.assertIn("quality_target", labels_train)
            self.assertEqual(manifest["frame_count"], len(trace))

            audit = audit_cache(cache, training_only=True)
            self.assertTrue(audit["approved"], audit["issues"])
            self.assertTrue(audit["ranking_signal_available"])
            self.assertEqual(audit["ranking_signal"]["train_rankable_groups"], 1)
            self.assertEqual(audit["ranking_signal"]["validation_rankable_groups"], 1)
            train_audit = audit["candidate_error_audit"]["train"]
            self.assertEqual(train_audit["frames"], 2)
            self.assertEqual(train_audit["empty_candidate_frames"], 1)
            self.assertEqual(train_audit["legacy_choice_correct"], 0)
            self.assertEqual(train_audit["legacy_choice_incorrect"], 1)
            self.assertEqual(train_audit["legacy_choice_incorrect_before_gates"], 1)
            self.assertEqual(train_audit["legacy_choice_incorrect_after_gates"], 1)
            self.assertEqual(train_audit["rankable_groups_after_gates"], 1)
            self.assertEqual(train_audit["candidate_targets_unavailable"], 1)

            groups = load_training_split(cache, "train")
            self.assertEqual(len(groups), 2)
            self.assertEqual({row["candidate_id"] for row in groups[0]["candidates"]},
                             {"good", "bad", "rejected"})
            self.assertNotIn("no_pose", {row["candidate_id"] for row in groups[0]["candidates"]})
            with self.assertRaisesRegex(ValueError, "only train or validation"):
                load_training_split(cache, "test")

            # Pre-optimizer audit verifies train/validation only and never opens
            # the test feature file; full evaluation audit detects this change.
            (cache / "features.test.jsonl").write_text("tampered\n", encoding="utf-8")
            self.assertTrue(audit_cache(cache, training_only=True)["approved"])
            self.assertFalse(audit_cache(cache)["approved"])

    def test_trace_label_injection_identity_mismatch_and_mesh_mismatch_fail(self):
        mesh = "a" * 64
        records = [
            _record("obj", "seq", "train", mesh_sha256=mesh),
        ]
        split = freeze_split_manifest({"records": records, "label_policy": POLICY})
        trace = [{"trace_schema_version": 1, **{key: records[0][key] for key in (
                    "physical_object_id", "object_id", "scene_id", "sequence_id", "background_id",
                    "background_kind", "provenance_kind", "frame_id", "source_frame_id")},
                 "mesh_sha256": mesh, "context": {"mesh_sha256": mesh}, "candidates": [],
                 "legacy_choice_candidate_id": None}]
        with self.assertRaisesRegex(ValueError, "label/evaluation field"):
            prepare_candidate_cache(Path(tempfile.gettempdir()) / "unused-ranker-cache",
                                    split, [{**trace[0], "quality_target": 1}], [])
        mismatch = {**trace[0], "mesh_sha256": "b" * 64,
                    "context": {"mesh_sha256": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "mesh_sha256"):
            prepare_candidate_cache(Path(tempfile.gettempdir()) / "unused-ranker-cache",
                                    split, [mismatch], [])

    def test_trace_and_cached_frame_ordinals_match_frozen_source_mapping(self):
        split = freeze_split_manifest(_inventory(train_frames=1))
        trace, references = _trace_and_labels(split)
        wrong = [dict(row) for row in trace]
        wrong[0]["frame_id"] = 77
        with self.assertRaisesRegex(ValueError, "frame_id differs from the frozen ordinal/source mapping"):
            prepare_candidate_cache(Path(tempfile.gettempdir()) / "unused-ranker-cache",
                                    split, wrong, references)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            cache = Path(temp) / "cache"
            prepare_candidate_cache(cache, split, trace, references,
                                    feature_names=list(FEATURE_NAMES),
                                    feature_schema_version=FEATURE_SCHEMA_VERSION)
            cache_manifest, _ = _cache_manifest(cache)
            frames, labels = load_cache_split(cache, "train")
            frames[0]["frame_id"] = 77
            issues = _audit_split(frames, labels, "train", cache_manifest["feature_count"],
                                  read_split_manifest(cache / "split_manifest.json"))
            self.assertTrue(any("cached frame_id differs from frozen ordinal/source mapping" in issue
                                for issue in issues), issues)

    def test_bop_adapter_requires_full_inventory_maps_ordinals_and_converts_mm(self):
        pose = {"cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [100., 200., 300.]}
        inference = {"schemaVersion": 1, "dataset": "BOP YCB-Video", "benchmarkId": "b",
                     "selectionDigest": "s", "objects": [{
                         "objectId": 1, "canonicalYcbId": "001_chips_can", "sceneId": "000001",
                         "referenceId": "ref", "model": {"sha256": "a" * 64},
                         "frames": [{"frameId": 42}, {"frameId": 44}]}]}
        evaluator = {"schemaVersion": 1, "dataset": "BOP YCB-Video", "benchmarkId": "b",
                     "selectionDigest": "s", "labelKind": "dataset_reference",
                     "objects": [{"objectId": 1, "sceneId": "000001", "referenceId": "ref",
                                  "frames": [{"frameId": 42, "evaluationPose": pose},
                                             {"frameId": 44, "evaluationPose": pose}]}]}
        evaluator_bytes = json.dumps(evaluator, sort_keys=True, separators=(",", ":")).encode()
        inference["evaluatorManifestSHA256"] = hashlib.sha256(evaluator_bytes).hexdigest()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            (root / "inference.json").write_text(json.dumps(inference, sort_keys=True, separators=(",", ":")))
            (root / "evaluator.json").write_bytes(evaluator_bytes)
            inventory, references = adapt_bop_ycbv_manifests(
                root / "inference.json", root / "evaluator.json",
                scene_inventory={"scenes": [{"scene_id": "000001", "inventory_complete": True,
                                               "object_ids": [1, 2]}]})
            with self.assertRaisesRegex(ValueError, "scene inventory"):
                adapt_bop_ycbv_manifests(root / "inference.json", root / "evaluator.json",
                                         scene_inventory={"scenes": []})
        self.assertEqual([row["frame_id"] for row in inventory["records"]], [0, 1])
        self.assertEqual([row["source_frame_id"] for row in inventory["records"]], [42, 44])
        self.assertEqual(references[0]["frame_id"], 0)
        self.assertEqual(references[0]["source_frame_id"], 42)
        np.testing.assert_allclose(np.asarray(references[0]["pose"])[:3, 3], [.1, .2, .3])
        self.assertTrue(references[0]["observationally_ambiguous"])


if __name__ == "__main__":
    unittest.main()
