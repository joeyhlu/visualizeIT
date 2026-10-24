"""CPU-only analytical checks for the frozen-cache candidate-signal report."""
from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .quality_ranker_data import freeze_split_manifest, prepare_candidate_cache
from .quality_ranker_signal import (
    MAX_REPORT_BYTES,
    _output_bytes,
    _write_output,
    diagnose_cache,
    diagnose_groups,
    main,
)


POLICY = {
    "max_rotation_error_deg": 5.0,
    "max_translation_error_m": 0.05,
    "max_surface_error_m": None,
    "max_projection_error_px": None,
}


def _rz(degrees: float, x: float = 0.0) -> list[list[float]]:
    radians = np.deg2rad(degrees)
    pose = np.eye(4)
    pose[:3, :3] = [
        [np.cos(radians), -np.sin(radians), 0.0],
        [np.sin(radians), np.cos(radians), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[0, 3] = x
    return pose.tolist()


def _candidate(candidate_id, pose, *, eligible=True, rejection=None,
               stage="initial", source="foundpose"):
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "source": source,
        "pose": pose,
        "eligible": eligible,
        "rejection": rejection,
        "validation_stats": {"inliers": 8},
        "features": [0.25],
    }


def _frame(candidates, *, split="train", frame_id=0, failure_reason=None,
           pose_state="valid", render_state="rendered", context=None):
    object_id = f"object-{split}"
    sequence_id = f"sequence-{split}"
    return {
        "split": split,
        "object_id": object_id,
        "physical_object_id": object_id,
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "source_frame_id": 1000 + frame_id,
        "timestamp_s": float(frame_id),
        "clock_mode": "capture_seconds_v1",
        "context": {} if context is None else context,
        "candidates": candidates,
        "pose_state": pose_state,
        "render_state": render_state,
        "failure_reason": failure_reason,
    }


def _label(candidate_id, target, *, split="train", frame_id=0,
           ambiguous=False, rotation=0.0, translation=0.0, reason=None):
    return {
        "object_id": f"object-{split}",
        "sequence_id": f"sequence-{split}",
        "frame_id": frame_id,
        "source_frame_id": 1000 + frame_id,
        "candidate_id": candidate_id,
        "split": split,
        "quality_target": target,
        "observationally_ambiguous": ambiguous,
        "orientation_supervision": not ambiguous,
        "rotation_error_deg": None if ambiguous else rotation,
        "translation_error_m": translation,
        "label_missing_reason": reason,
        "label_kind": "analytic_reference",
    }


def _inventory_record(split):
    return {
        "physical_object_id": f"object-{split}",
        "object_id": f"object-{split}",
        "scene_id": f"scene-{split}",
        "sequence_id": f"sequence-{split}",
        "background_id": "procedural:gray",
        "background_kind": "procedural",
        "provenance_kind": "real",
        "frame_id": 0,
        "source_frame_id": 1000,
        "split": split,
    }


def _make_cache(root: Path) -> Path:
    inventory = {"records": [_inventory_record(split) for split in
                              ("train", "validation", "test", "dev")],
                 "label_policy": POLICY}
    split_manifest = freeze_split_manifest(inventory)
    trace = []
    references = []
    for split in ("train", "validation", "test", "dev"):
        record = _inventory_record(split)
        candidates = [
            {
                "candidate_id": "candidate-a",
                "original_index": 0,
                "stage": "initial",
                "source": "analytic",
                "pose": _rz(0),
                "eligible": True,
                "rejection": None,
                "validation_stats": {"inliers": 8},
                "features": [0.25],
                "legacy_score": 0.1,
            }
        ]
        trace.append({
            "trace_schema_version": 1,
            **{key: record[key] for key in (
                "physical_object_id", "object_id", "scene_id", "sequence_id",
                "background_id", "background_kind", "provenance_kind",
                "frame_id", "source_frame_id",
            )},
            "timestamp_s": 0.0,
            "clock_mode": "capture_seconds_v1",
            "context": {"mask_state": "available"},
            "candidates": candidates,
            "legacy_choice_candidate_id": "candidate-a",
            "pose_state": "valid",
            "render_state": "rendered",
            "failure_reason": None,
            "cameraFromObject": None,
        })
        references.append({
            "object_id": record["object_id"],
            "sequence_id": record["sequence_id"],
            "frame_id": 0,
            "source_frame_id": 1000,
            "pose": _rz(0),
            "observationally_ambiguous": False,
            "label_kind": "analytic_reference",
        })
    cache = root / "cache"
    prepare_candidate_cache(
        cache,
        split_manifest,
        trace,
        references,
        feature_names=["analytic_feature"],
        feature_schema_version=1,
    )
    return cache


class GroupDiagnosticTests(unittest.TestCase):
    def test_manufactured_pair_distances_cover_zero_ninety_and_one_eighty(self):
        candidates = [
            _candidate("zero", _rz(0, 0.0)),
            _candidate("ninety", _rz(90, 1.0)),
            _candidate("one-eighty", _rz(180, 2.0)),
        ]
        labels = [_label(row["candidate_id"], 1) for row in candidates]
        result = diagnose_groups([_frame(candidates)], labels, split="train",
                                 label_policy=POLICY)
        pairs = result["pose_pair_distances"]["within_stage"]["pairs"]
        self.assertEqual([round(row["rotation_deg"]) for row in pairs], [90, 180, 90])
        self.assertEqual([row["translation_m"] for row in pairs], [1.0, 2.0, 1.0])
        self.assertEqual(result["candidate_counts"]["valid_pose"], 3)

    def test_identical_poses_are_observed_zero_and_no_pair_is_no_evidence(self):
        candidates = [_candidate("a", _rz(0)), _candidate("b", _rz(0))]
        labels = [_label("a", 1), _label("b", 1)]
        observed = diagnose_groups([_frame(candidates)], labels, split="train",
                                   label_policy=POLICY)
        summary = observed["pose_pair_distances"]["overall_summary"]
        self.assertEqual(summary["evidence_status"], "observed_pairs")
        self.assertEqual(summary["rotation_deg"]["max"], 0.0)
        empty = diagnose_groups([_frame([candidates[0]])], [labels[0]], split="train",
                                label_policy=POLICY)
        summary = empty["pose_pair_distances"]["overall_summary"]
        self.assertEqual(summary["evidence_status"], "no_pairs_no_dispersion_evidence")
        self.assertIsNone(summary["rotation_deg"])

    def test_ambiguous_positive_angular_distance_is_recomputed_without_relabeling(self):
        candidates = [_candidate("a", _rz(0)), _candidate("b", _rz(90))]
        labels = [_label("a", 1, ambiguous=True), _label("b", 1, ambiguous=True)]
        result = diagnose_groups([_frame(candidates)], labels, split="train",
                                 label_policy=POLICY)
        pairs = result["pose_pair_distances"]
        self.assertEqual(pairs["ambiguous_positive_pair_count"], 1)
        self.assertAlmostEqual(pairs["ambiguous_positive_summary"]["rotation_deg"]["mean"], 90.0)
        for row in result["frames"][0]["candidate_labels"]:
            self.assertIsNone(row["rotation_error_deg"])
            self.assertFalse(row["orientation_supervision"])

    def test_gates_report_removed_good_bad_and_surviving_classes(self):
        candidates = [
            _candidate("good-removed", _rz(0), eligible=False, rejection="gate-a"),
            _candidate("bad-removed", _rz(10), eligible=False, rejection="gate-b"),
            _candidate("good-kept", _rz(20)),
            _candidate("bad-kept", _rz(30)),
        ]
        labels = [
            _label("good-removed", 1), _label("bad-removed", 0),
            _label("good-kept", 1), _label("bad-kept", 0),
        ]
        result = diagnose_groups([_frame(candidates)], labels, split="train",
                                 label_policy=POLICY)
        self.assertEqual(result["removed_quality_classes_by_rejection"]["gate-a"], {"good": 1})
        self.assertEqual(result["removed_quality_classes_by_rejection"]["gate-b"], {"bad": 1})
        frame = result["frames"][0]
        self.assertEqual(frame["surviving_quality_classes"], {"bad": 1, "good": 1})
        self.assertEqual(result["mixed_good_bad_groups_before_gates"], 1)
        self.assertEqual(result["mixed_good_bad_groups_after_gates"], 1)

    def test_missing_pose_target_and_label_remain_distinct(self):
        candidates = [
            _candidate("missing-pose", None, eligible=False, rejection="pose_missing"),
            _candidate("target-unavailable", _rz(0)),
            _candidate("missing-label", _rz(5)),
        ]
        labels = [
            _label("missing-pose", None, reason="candidate_pose_unavailable", translation=None),
            _label("target-unavailable", None, reason="surface_metric_required", translation=0.0),
        ]
        result = diagnose_groups([_frame(candidates)], labels, split="train",
                                 label_policy=POLICY)
        self.assertEqual(result["composition_before_gates"], {
            "good": 0, "bad": 0, "target_unavailable": 2, "missing_label": 1,
        })
        self.assertEqual(result["candidate_counts"]["attempts"], 3)
        self.assertEqual(result["candidate_counts"]["valid_pose"], 2)
        reasons = {row["candidate_id"]: row["label_missing_reason"]
                   for row in result["frames"][0]["candidate_labels"]}
        self.assertEqual(reasons["missing-pose"], "candidate_pose_unavailable")
        self.assertIsNone(reasons["missing-label"])

    def test_placeholders_processed_empty_unknown_and_cross_stage_stay_separate(self):
        frames = [
            _frame([], frame_id=0, failure_reason="not_processed", pose_state=None,
                   render_state=None),
            _frame([], frame_id=1, failure_reason="no_candidates", pose_state="lost",
                   render_state="hidden"),
            _frame([], frame_id=2, failure_reason=None, pose_state=None,
                   render_state=None, context={}),
            _frame([
                _candidate("initial", _rz(0), stage="initial", source="foundpose"),
                _candidate("recovery", _rz(90), stage="recovery", source="gotrack"),
            ], frame_id=3),
        ]
        labels = [_label("initial", 1, frame_id=3), _label("recovery", 0, frame_id=3)]
        result = diagnose_groups(frames, labels, split="train", label_policy=POLICY)
        self.assertEqual(result["processing_status_counts"], {
            "processed_empty": 1,
            "processed_with_candidates": 1,
            "unknown_status_empty": 1,
            "unprocessed": 1,
        })
        self.assertEqual(result["pose_pair_distances"]["within_stage"]["summary"]["pair_count"], 0)
        self.assertEqual(result["pose_pair_distances"]["cross_stage"]["summary"]["pair_count"], 1)

    def test_pair_bound_raises_instead_of_truncating(self):
        candidates = [_candidate(str(index), _rz(index * 10)) for index in range(3)]
        labels = [_label(str(index), index % 2) for index in range(3)]
        with self.assertRaisesRegex(ValueError, "exceeds configured bound"):
            diagnose_groups([_frame(candidates)], labels, split="train",
                            label_policy=POLICY, max_pairs=2)

    def test_no_torch_import_and_no_pose_or_feature_vectors_in_report(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("torch must not be imported")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            report = diagnose_groups(
                [_frame([_candidate("a", _rz(0))])],
                [_label("a", 1)], split="train", label_policy=POLICY,
            )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn('"features"', serialized)
        self.assertNotIn('"pose":', serialized)


class CacheAndCliTests(unittest.TestCase):
    def test_cache_diagnosis_reads_train_validation_only_even_if_test_is_tampered(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            cache = _make_cache(Path(temp))
            (cache / "features.test.jsonl").write_text("tampered test shard\n", encoding="utf-8")
            report = diagnose_cache(cache / "cache_manifest.json")
            self.assertFalse(report["test_read"])
            self.assertFalse(report["dev_read"])
            self.assertEqual(set(report["splits"]), {"train", "validation"})
            self.assertEqual(report["frozen_label_policy"], POLICY)
            self.assertFalse(report["raw_gate_vectors"]["available"])
            self.assertEqual(
                set(report["cache_provenance"]["diagnostic_source_sha256"]),
                {"quality_ranker_signal.py", "quality_ranker_data.py"},
            )
            self.assertTrue(all(
                len(value) == 64
                for value in report["cache_provenance"]["diagnostic_source_sha256"].values()
            ))

    def test_cache_identity_change_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            cache = _make_cache(Path(temp))
            from . import quality_ranker_signal as module
            real_audit = module.audit_cache
            first = real_audit(cache, training_only=True)
            second = copy.deepcopy(first)
            second["identity"]["source_trace_sha256"] = "0" * 64
            with mock.patch.object(module, "audit_cache", side_effect=[first, second]):
                with self.assertRaisesRegex(ValueError, "changed during diagnosis"):
                    diagnose_cache(cache / "cache_manifest.json")

    def test_cli_refuses_existing_output(self):
        tiny = {"splits": {"train": {"frame_count": 1},
                           "validation": {"frame_count": 1}}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            output = root / "report.json"
            output.write_text("keep", encoding="utf-8")
            with mock.patch("bench.quality_ranker_signal.diagnose_cache", return_value=tiny):
                with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                    main(["--manifest", str(root / "ignored.json"), "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_direct_output_symlink_is_refused_before_open(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            link = Path(temp) / "linked-report.json"
            # Windows CI may not grant symlink creation privilege.  Exercise the
            # same pathlib state deterministically and assert no open is tried.
            real_exists = Path.exists
            real_symlink = Path.is_symlink

            def exists(path):
                return False if path == link else real_exists(path)

            def is_symlink(path):
                return True if path == link else real_symlink(path)

            with mock.patch.object(Path, "exists", exists), \
                    mock.patch.object(Path, "is_symlink", is_symlink), \
                    mock.patch.object(Path, "open", side_effect=AssertionError("must not open symlink")):
                with self.assertRaisesRegex(FileExistsError, "Refusing to replace"):
                    _write_output(link, b"replacement")

    def test_cli_refuses_output_inside_cache_before_diagnosis(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            cache = _make_cache(Path(temp))
            output = cache / "forbidden-report.json"
            with mock.patch("bench.quality_ranker_signal.diagnose_cache") as diagnose:
                with self.assertRaisesRegex(ValueError, "inside the immutable cache"):
                    main(["--manifest", str(cache / "cache_manifest.json"),
                          "--output", str(output)])
            diagnose.assert_not_called()
            self.assertFalse(output.exists())

    def test_report_byte_bound_raises_explicitly(self):
        with mock.patch("bench.quality_ranker_signal.MAX_REPORT_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                _output_bytes({"value": "larger than eight bytes"})
        self.assertEqual(MAX_REPORT_BYTES, 16 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
