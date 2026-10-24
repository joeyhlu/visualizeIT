from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import quality_postseal_pose_evaluate as evaluator
from .quality_ranker_data import rotation_error_degrees, translation_error_m


SEQUENCE = "ranker-multiview-hot3d17-v1-yaw0-pitchm25"
OBJECT = "hot3d:obj_000017"


def _temporary_directory():
    # The sandbox's OS temp folder is read-only; keep fixtures in the writable
    # project cache and let TemporaryDirectory remove its exact created path.
    return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".cache")


def _pose(x=0.0, y=0.0, z=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (x, y, z)
    return pose


def _reference_frames(count=24):
    return {
        frame_id: {
            "frame_id": frame_id,
            "source_frame_id": frame_id,
            "timestamp_s": frame_id / 10.0,
            "pose": _pose(frame_id / 10.0),
            "label_kind": "synthetic_renderer_pose",
            "observationally_ambiguous": True,
        }
        for frame_id in range(count)
    }


def _complete_row(frame_id, *, pose=None, state="tracking", render="visible"):
    row = {
        "frameId": frame_id,
        "sourceFrameId": frame_id,
        "timestamp_s": frame_id / 10.0,
        "pose_state": state,
        "render_state": render,
        "failure_reason": None,
        "mask_state": "available",
        "mask_measurement_state": "measured",
        "cameraFromObject": pose,
    }
    if pose is not None:
        row["_pose"] = np.asarray(pose, dtype=np.float64)
    return row


def _candidate_row(frame_id, *, selected=True, candidate_pose=None, eligible=True):
    candidate = {
        "candidate_id": "candidate-0",
        "pose": _pose(0.0 if candidate_pose is None else candidate_pose, 0.0, 0.0).tolist(),
        "eligible": eligible,
        "rejection": None if eligible else "gate_rejected",
        "gates": {"current_image_validation": eligible},
    }
    return {
        "frame_id": frame_id,
        "source_frame_id": frame_id,
        "timestamp_s": frame_id / 10.0,
        "candidates": [candidate],
        "rank_decision": {
            "selected_id": "candidate-0" if selected else None,
            "ordered_ids": ["candidate-0"],
        },
        "pose_state": "tracking" if selected else "recovering",
        "render_state": "visible" if selected else "suppressed",
        "cameraFromObject": _pose().tolist() if selected else None,
    }


def _reference_generation(**overrides):
    result = {
        "kind": evaluator._GENERATION_KIND,
        "schema_version": 1,
        "sequence_id": SEQUENCE,
        "physical_object_id": OBJECT,
        "split": "train",
        "units": "metres",
        "labels_relative_path": "poses.jsonl",
        "labels_sha256": "a" * 64,
    }
    result.update(overrides)
    return result


def _reference_records(count=24):
    return [
        {
            "frame_id": frame_id,
            "label_kind": "synthetic_renderer_pose",
            "object_id": OBJECT,
            "observationally_ambiguous": True,
            "pose": _pose(frame_id / 10.0).tolist(),
            "sequence_id": SEQUENCE,
            "source_frame_id": frame_id,
            "timestamp_s": frame_id / 10.0,
        }
        for frame_id in range(count)
    ]


class PairedReportSizeTests(unittest.TestCase):
    def test_expanded_report_accepts_between_two_and_three_mib(self):
        raw = b'{}' + b' ' * (evaluator.MAX_JSON_BYTES - 1)
        self.assertEqual(evaluator._parse_paired_report_bytes(
            raw, policy_id='expanded24'), {})
        with self.assertRaisesRegex(ValueError, '2097152-byte bound'):
            evaluator._parse_paired_report_bytes(raw, policy_id='legacy8')
        with self.assertRaisesRegex(ValueError, '2097152-byte bound'):
            evaluator._parse_json_bytes(raw, 'other.json')

    def test_expanded_report_keeps_three_mib_limit(self):
        raw = b'{}' + b' ' * (evaluator.MAX_EXPANDED_PAIRED_REPORT_BYTES - 2)
        self.assertEqual(evaluator._parse_paired_report_bytes(
            raw, policy_id='expanded24'), {})
        with self.assertRaisesRegex(ValueError, '3145728-byte bound'):
            evaluator._parse_paired_report_bytes(raw + b' ', policy_id='expanded24')

    def test_legacy_exact_two_mib_boundary_is_unchanged(self):
        raw = b'{}' + b' ' * (evaluator.MAX_JSON_BYTES - 2)
        self.assertEqual(evaluator._parse_paired_report_bytes(
            raw, policy_id='legacy8'), {})
        with self.assertRaisesRegex(ValueError, '2097152-byte bound'):
            evaluator._parse_paired_report_bytes(raw + b' ', policy_id='legacy8')


class PoseMetricTests(unittest.TestCase):
    def test_translation_error_uses_metres(self):
        # Analytical Euclidean distance: sqrt(0.03^2 + 0.04^2) = 0.05 m.
        self.assertAlmostEqual(translation_error_m(_pose(0.03, 0.04, 0), _pose()), 0.05, places=12)

    def test_rotation_helpers_cover_90_and_180_degrees(self):
        ninety = np.eye(4)
        ninety[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        one_eighty = np.eye(4)
        one_eighty[:3, :3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        # The constructed proper rotations are exactly 90 and 180 degrees by axis-angle.
        self.assertAlmostEqual(rotation_error_degrees(ninety, np.eye(4)), 90.0, places=10)
        self.assertAlmostEqual(rotation_error_degrees(one_eighty, np.eye(4)), 180.0, places=10)

    def test_ambiguous_reference_keeps_primary_orientation_unmeasured(self):
        refs = _reference_frames()
        row = evaluator.evaluate_sequence(
            sequence_id=SEQUENCE, split="train", object_id=OBJECT,
            reference_frames=refs, complete_frames={}, candidate_frames={}, condition="frozen",
        )
        self.assertEqual(row["orientation_primary"]["status"],
                         "unmeasured_observationally_ambiguous_reference")
        self.assertEqual(row["orientation_primary"]["scored_error_values_count"], 0)
        self.assertEqual(row["orientation_primary"]["eligible_denominator"], 0)
        self.assertIsNone(row["orientation_primary"]["mean_degrees"])
        self.assertEqual(row["severe_orientation"]["eligible_denominator"], 0)
        self.assertIsNone(row["severe_orientation"]["failure_count"])

    def test_constant_translation_bias_is_absolute_error_without_increment_residual(self):
        refs = {frame_id: _pose(0.1 * frame_id) for frame_id in range(4)}
        predicted = {frame_id: _pose(0.1 * frame_id + 0.02) for frame_id in range(4)}
        absolute = [translation_error_m(predicted[i], refs[i]) for i in range(4)]
        transitions = evaluator.translation_increment_residuals(predicted, refs, [0, 1, 2, 3])
        # Manufactured constant-bias example: each absolute error is |0.02| m.
        self.assertTrue(all(abs(error - 0.02) < 1e-12 for error in absolute))
        self.assertEqual(len(transitions), 3)
        # Subtracting the same 0.02 m offset at both endpoints leaves zero increment residual.
        self.assertTrue(all(abs(row["residual_m"]) < 1e-12 for row in transitions))

        shifted = {0: _pose(0.0), 1: _pose(0.11)}
        baseline = {0: _pose(0.0), 1: _pose(0.1)}
        changed = evaluator.translation_increment_residuals(shifted, baseline, [0, 1])
        # The constructed increments differ by 0.11 - 0.10 = 0.01 m.
        self.assertAlmostEqual(changed[0]["residual_m"], 0.01, places=12)

    def test_selected_candidate_with_null_suppressed_output_is_coverage_only(self):
        refs = _reference_frames()
        result = evaluator.evaluate_sequence(
            sequence_id=SEQUENCE, split="train", object_id=OBJECT,
            reference_frames=refs,
            complete_frames={0: _complete_row(0, pose=None, state="recovering", render="suppressed")},
            candidate_frames={0: _candidate_row(0, selected=True, candidate_pose=0.05)},
            condition="frozen",
        )
        self.assertEqual(result["requested_count"], 8)
        self.assertEqual(result["produced_count"], 1)
        self.assertEqual(result["emitted_pose_count"], 0)
        self.assertEqual(result["emitted_translation_errors"]["count"], 0)
        self.assertEqual(result["candidate_coverage"]["selected_candidate_pose_available_count"], 1)
        self.assertEqual(result["candidate_coverage"]["valid_candidate_count_before_gates"], 1)
        self.assertEqual(result["recovering_count"], 1)
        self.assertEqual(result["suppressed_count"], 1)
        availability = result["frame_availability"]
        self.assertEqual(availability["requested_count"], 8)
        self.assertEqual(availability["emitted_pose_unavailable_frame_ids"], list(range(8)))
        self.assertIsNone(availability["rows"][0]["translation_error_m"])

    def test_missing_output_keeps_fixed_denominator_and_breaks_both_neighbor_transitions(self):
        refs = _reference_frames()
        result = evaluator.evaluate_sequence(
            sequence_id=SEQUENCE, split="train", object_id=OBJECT,
            reference_frames=refs,
            complete_frames={0: _complete_row(0, pose=_pose(0.01)),
                             2: _complete_row(2, pose=_pose(0.21))},
            candidate_frames={}, condition="frozen", requested_frame_ids=[0, 1, 2],
        )
        self.assertEqual(result["requested_count"], 3)
        self.assertEqual(result["produced_count"], 2)
        self.assertEqual(result["missing_output_row_count"], 1)
        self.assertEqual(result["temporal_translation_increment_residuals"][
            "possible_consecutive_requested_transitions"], 2)
        self.assertEqual(result["temporal_translation_increment_residuals"]["count"], 0)
        availability = result["frame_availability"]
        self.assertEqual(availability["missing_output_frame_ids"], [1])
        self.assertEqual(len(availability["rows"]), 3)
        self.assertIsNone(availability["rows"][1]["translation_error_m"])
        self.assertEqual(availability["rows"][1]["emitted_pose_unavailable_reason"],
                         "missing_output_row")

    def test_inconsistent_pose_and_state_combinations_fail_closed(self):
        refs = _reference_frames()
        with self.assertRaisesRegex(ValueError, "requires tracking and visible"):
            evaluator.evaluate_sequence(
                sequence_id=SEQUENCE, split="train", object_id=OBJECT,
                reference_frames=refs,
                complete_frames={0: _complete_row(0, pose=_pose(0.01),
                                                  state="recovering", render="suppressed")},
                candidate_frames={}, condition="frozen", requested_frame_ids=[0],
            )
        with self.assertRaisesRegex(ValueError, "must emit cameraFromObject"):
            evaluator.evaluate_sequence(
                sequence_id=SEQUENCE, split="train", object_id=OBJECT,
                reference_frames=refs,
                complete_frames={0: _complete_row(0, pose=None,
                                                  state="tracking", render="visible")},
                candidate_frames={}, condition="frozen", requested_frame_ids=[0],
            )

    def test_evaluator_rejects_unsorted_ids_and_nonambiguous_orientation_claim(self):
        refs = _reference_frames()
        with self.assertRaisesRegex(ValueError, "ascending"):
            evaluator.evaluate_sequence(
                sequence_id=SEQUENCE, split="train", object_id=OBJECT,
                reference_frames=refs, complete_frames={}, candidate_frames={},
                condition="frozen", requested_frame_ids=[1, 0],
            )
        refs[0]["observationally_ambiguous"] = False
        with self.assertRaisesRegex(ValueError, "ambiguous references"):
            evaluator.evaluate_sequence(
                sequence_id=SEQUENCE, split="train", object_id=OBJECT,
                reference_frames=refs, complete_frames={}, candidate_frames={},
                condition="frozen", requested_frame_ids=[0],
            )

    def test_candidate_coverage_does_not_add_prefix_or_unrequested_frames(self):
        self.assertEqual(len(evaluator.EXPECTED_RUNS), 6)
        self.assertEqual(evaluator.REQUESTED_FRAME_IDS, tuple(range(8)))
        self.assertFalse(any(condition.startswith("four-") or sequence.startswith("four-")
                             for condition, sequence in evaluator.EXPECTED_RUNS))
        self.assertTrue(any(path.startswith("four-") for path in evaluator.EXPECTED_SEALED_MEMBERS))
        self.assertEqual(evaluator.REFERENCE_FRAME_IDS, tuple(range(24)))


class InputValidationTests(unittest.TestCase):
    def test_reference_duplicate_identity_timestamp_object_and_units_fail(self):
        base = _reference_records()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            evaluator.validate_reference_records(
                _reference_generation(), [base[0], *base[:-1]], expected_sequence_id=SEQUENCE,
                expected_split="train", expected_object_id=OBJECT, reference_units="metres",
            )

        duplicate_time = _reference_records()
        duplicate_time[1]["timestamp_s"] = duplicate_time[0]["timestamp_s"]
        with self.assertRaisesRegex(ValueError, "timestamps"):
            evaluator.validate_reference_records(
                _reference_generation(), duplicate_time, expected_sequence_id=SEQUENCE,
                expected_split="train", expected_object_id=OBJECT, reference_units="metres",
            )

        wrong_object = _reference_records()
        wrong_object[0]["object_id"] = "hot3d:obj_000027"
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluator.validate_reference_records(
                _reference_generation(), wrong_object, expected_sequence_id=SEQUENCE,
                expected_split="train", expected_object_id=OBJECT, reference_units="metres",
            )

        with self.assertRaisesRegex(ValueError, "units"):
            evaluator.validate_reference_records(
                _reference_generation(units="millimetres"), _reference_records(),
                expected_sequence_id=SEQUENCE, expected_split="train", expected_object_id=OBJECT,
                reference_units="millimetres",
            )

    def test_reflected_pose_is_rejected(self):
        reflected = np.eye(4)
        reflected[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "proper finite"):
            evaluator._pose_or_none(reflected, "test pose")

    def test_duplicate_and_timestamp_output_rows_fail(self):
        refs = _reference_frames()
        row = {"frameId": 0, "sourceFrameId": 0, "timestamp_s": 0.0}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluator._validated_frame_rows(
                [row, row], stream_name="complete", frame_id_key="frameId",
                source_frame_id_key="sourceFrameId", reference_frames=refs,
                sequence_id=SEQUENCE, object_id=OBJECT, require_identity_fields=False,
            )
        wrong_timestamp = dict(row, timestamp_s=0.25)
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            evaluator._validated_frame_rows(
                [wrong_timestamp], stream_name="complete", frame_id_key="frameId",
                source_frame_id_key="sourceFrameId", reference_frames=refs,
                sequence_id=SEQUENCE, object_id=OBJECT, require_identity_fields=False,
            )

    def test_duplicate_json_keys_and_nonfinite_numbers_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluator._parse_json_bytes(b'{"a":1,"a":2}', "fixture")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluator._parse_json_bytes(b'{"a":NaN}', "fixture")

    def test_member_manifest_authentication_detects_tampering(self):
        with _temporary_directory() as temporary:
            root = Path(temporary).resolve()
            member = b"sealed member\n"
            (root / "member.json").write_bytes(member)
            manifest_raw = json.dumps(
                {"files": {"member.json": hashlib.sha256(member).hexdigest()}},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            (root / "artifact-manifest.json").write_bytes(manifest_raw)
            pin = hashlib.sha256(manifest_raw).hexdigest()
            _, snapshot = evaluator.authenticate_member_manifest(
                root, pin, expected_members={"member.json"})
            self.assertEqual(snapshot["member.json"], member)
            (root / "member.json").write_bytes(b"altered\n")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                evaluator.authenticate_member_manifest(
                    root, pin, expected_members={"member.json"})

    def test_member_path_and_output_root_escapes_fail(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            evaluator._safe_relative_name("../outside.json")
        with self.assertRaisesRegex(ValueError, "drive prefix"):
            evaluator._safe_relative_name("C:/outside.json")
        with _temporary_directory() as temporary:
            root = Path(temporary).resolve()
            nested = root / "input"
            nested.mkdir()
            with self.assertRaisesRegex(ValueError, "inside"):
                evaluator._check_output_destination(nested / "report.json", [nested])

    def test_input_bytes_remain_unchanged_and_module_has_no_inference_imports(self):
        with _temporary_directory() as temporary:
            root = Path(temporary).resolve()
            member = b"unchanged\n"
            (root / "member.json").write_bytes(member)
            manifest_raw = json.dumps(
                {"files": {"member.json": hashlib.sha256(member).hexdigest()}},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            (root / "artifact-manifest.json").write_bytes(manifest_raw)
            original = {p.name: p.read_bytes() for p in root.iterdir()}
            evaluator.authenticate_member_manifest(
                root, hashlib.sha256(manifest_raw).hexdigest(), expected_members={"member.json"})
            self.assertEqual(original, {p.name: p.read_bytes() for p in root.iterdir()})
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("quality_runner", source)
        self.assertNotIn("quality_gotrack", source)


class ExpandedPolicyTests(unittest.TestCase):
    @staticmethod
    def _policy_document(**overrides):
        value = {
            "schema": evaluator.POSTSEAL_POLICY_SCHEMA,
            "policy_id": "expanded24",
            "design_sha256": "1" * 64,
            "execution_bindings_sha256": "2" * 64,
            "sealed_manifest_sha256": "3" * 64,
            "sealed_archive_sha256": "4" * 64,
            "diagnostic_sha256": evaluator.PINNED_DIAGNOSTIC_SHA256,
            "checkpoint_sha256": evaluator.EXPANDED_HEAD_SHA256,
            "requested_frame_ids": list(range(24)),
            "archive_bytes_max": evaluator.MAX_EXPANDED_ARCHIVE_BYTES,
        }
        value.update(overrides)
        return value

    def test_expanded_policy_is_sha_pinned_and_requires_exact_24_frame_request(self):
        with _temporary_directory() as temporary:
            path = Path(temporary) / "policy.json"
            raw = (json.dumps(self._policy_document(), sort_keys=True,
                              separators=(",", ":")) + "\n").encode()
            path.write_bytes(raw)
            policy = evaluator._load_expanded_policy(path, hashlib.sha256(raw).hexdigest())
            self.assertEqual(policy.requested_frame_ids, tuple(range(24)))
            self.assertEqual(policy.output_prefix, "full")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                evaluator._load_expanded_policy(path, "0" * 64)
            malformed = self._policy_document(requested_frame_ids=list(range(23)))
            path.write_text(json.dumps(malformed), encoding="utf-8")
            raw = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "exactly frames 0..23"):
                evaluator._load_expanded_policy(path, hashlib.sha256(raw).hexdigest())

    def test_archive_hash_and_unpacked_inventory_must_match_the_sealed_root(self):
        members = {"artifact-manifest.json": b'{"files":{}}', "design.json": b"{}"}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        raw_archive = stream.getvalue()
        with _temporary_directory() as temporary:
            path = Path(temporary) / "sealed.zip"
            path.write_bytes(raw_archive)
            policy = replace(evaluator._LEGACY_POLICY,
                archive_sha256=hashlib.sha256(raw_archive).hexdigest(),
                archive_bytes_max=evaluator.MAX_EXPANDED_ARCHIVE_BYTES,
                sealed_total_bytes_max=evaluator.MAX_EXPANDED_SEALED_TOTAL_BYTES)
            evaluator._verify_archive_matches_snapshot(path, policy, members)
            with self.assertRaisesRegex(ValueError, "inventory or bytes differ"):
                evaluator._verify_archive_matches_snapshot(path, policy,
                    {**members, "extra.json": b"{}"})

    def test_expanded_policy_and_archive_authenticate_before_reference_reads(self):
        with _temporary_directory() as temporary:
            root = Path(temporary).resolve()
            sealed_root = root / "sealed"
            sealed_root.mkdir()
            member_bytes = {name: b"{}" for name in evaluator._expanded_required_members()}
            files = {name: hashlib.sha256(raw).hexdigest()
                     for name, raw in member_bytes.items()}
            manifest_raw = (json.dumps({"files": files}, sort_keys=True,
                                       separators=(",", ":")) + "\n").encode()
            (sealed_root / "artifact-manifest.json").write_bytes(manifest_raw)
            for name, raw in member_bytes.items():
                target = sealed_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)

            archive_path = root / "sealed.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("artifact-manifest.json", manifest_raw)
                for name, raw in sorted(member_bytes.items()):
                    archive.writestr(name, raw)
            archive_raw = archive_path.read_bytes()
            document = self._policy_document(
                design_sha256=hashlib.sha256(member_bytes["design.json"]).hexdigest(),
                execution_bindings_sha256=hashlib.sha256(
                    member_bytes["execution-bindings.json"]).hexdigest(),
                sealed_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                sealed_archive_sha256=hashlib.sha256(archive_raw).hexdigest(),
            )
            policy_path = root / "policy.json"
            policy_raw = (json.dumps(document, sort_keys=True,
                                     separators=(",", ":")) + "\n").encode()
            policy_path.write_bytes(policy_raw)

            with patch.object(evaluator, "_load_references",
                              side_effect=AssertionError("reference labels read before metadata validation")):
                with self.assertRaisesRegex(ValueError, "Unsupported paired design schema"):
                    evaluator.evaluate_postseal(
                        sealed_root=sealed_root,
                        sealed_manifest_sha256=document["sealed_manifest_sha256"],
                        reference_root=root / "unread-reference-root",
                        diagnostic_path=root / "unread-diagnostic.json",
                        experiment_policy="expanded24",
                        experiment_policy_json=policy_path,
                        experiment_policy_sha256=hashlib.sha256(policy_raw).hexdigest(),
                        sealed_archive=archive_path,
                    )

    def test_expanded_inventory_requires_all_six_full_and_two_prefix_run_packets(self):
        required = set(evaluator._expanded_required_members())
        manifest = {"files": {name: "a" * 64 for name in required}}
        snapshot = {name: b"" for name in required}
        evaluator._validate_expanded_inventory(manifest, snapshot)
        missing = "full-adapted-ranker-multiview-hot3d17-v1-yaw0-pitchm25/pose.receipt.json"
        del manifest["files"][missing]
        snapshot.pop(missing)
        with self.assertRaisesRegex(ValueError, "missing required run or master"):
            evaluator._validate_expanded_inventory(manifest, snapshot)

    def test_expanded_inventory_rejects_labels_and_unapproved_binary_members(self):
        required = set(evaluator._expanded_required_members())
        manifest = {"files": {name: "a" * 64 for name in required}}
        snapshot = {name: b"" for name in required}
        for name in ("diagnostics/poses.jsonl", "diagnostics/checkpoint.pt",
                     "unknown/result.json"):
            with self.subTest(name=name):
                candidate_manifest = {"files": {**manifest["files"], name: "b" * 64}}
                candidate_snapshot = {**snapshot, name: b"{}"}
                with self.assertRaisesRegex(ValueError, "unapproved non-run member"):
                    evaluator._validate_expanded_inventory(candidate_manifest, candidate_snapshot)

    def test_full_video_math_keeps_missing_rows_in_the_24_frame_denominator(self):
        row = evaluator.evaluate_sequence(
            sequence_id=SEQUENCE, split="train", object_id=OBJECT,
            reference_frames=_reference_frames(), complete_frames={}, candidate_frames={},
            condition="frozen", requested_frame_ids=range(24))
        self.assertEqual(row["requested_count"], 24)
        self.assertEqual(row["produced_count"], 0)
        self.assertEqual(row["missing_output_row_count"], 24)
        self.assertEqual(row["frame_availability"]["missing_output_frame_ids"], list(range(24)))
        self.assertEqual(row["orientation_primary"]["eligible_denominator"], 0)

    def test_expanded_metadata_requires_exact_six_runs_and_two_prefixes(self):
        policy = replace(evaluator._LEGACY_POLICY, policy_id="expanded24",
            checkpoint_sha256=evaluator.EXPANDED_HEAD_SHA256,
            requested_frame_ids=tuple(range(24)),
            design_schema=evaluator.EXPANDED_DESIGN_SCHEMA,
            bindings_schema=evaluator.EXPANDED_BINDINGS_SCHEMA,
            report_schema=evaluator.EXPANDED_REPORT_SCHEMA,
            output_prefix="full")
        design = {
            "schema": policy.design_schema,
            "profile": evaluator._INFERENCE_PROFILE,
            "test_object_7_read": False,
            "annotations_during_inference": False,
            "normal_geometry_changed": False,
            "optimizer_steps": 0,
            "checkpoint_sha256": evaluator.EXPANDED_HEAD_SHA256,
            "requested_frames_per_sequence": 24,
            "frame_timestamps_seconds": [i / 10 for i in range(24)],
            "runs": [{
                "condition": condition,
                "sequence_id": sequence_id,
                "split": spec["design_split"],
                "setup_frame_ids": [],
                "scored_frame_ids": list(range(24)),
            } for condition, sequence_id in sorted(evaluator.EXPECTED_RUNS)
              for spec in [evaluator.SEQUENCE_SPECS[sequence_id]]],
        }
        bindings = {
            "schema": policy.bindings_schema,
            "config": {"sha256": "c" * 64},
            "source_closure_sha256": "d" * 64,
            "cases": [{
                "sequence_id": sequence_id,
                "object_id": spec["binding_object_id"],
                "split": spec["design_split"],
                "generation": {"sha256": "a" * 64},
                "input": {"sha256": "b" * 64},
            } for sequence_id, spec in evaluator.SEQUENCE_SPECS.items()],
        }
        report = {
            "schema": policy.report_schema,
            "reference_pose_files_read": False,
            "annotations_during_inference": False,
            "runs": [{
                "condition": condition, "sequence_id": sequence_id,
                "status": "complete", "failure_reason": None,
                "requested_frame_ids": list(range(24)),
                "output_path": f"/sealed/full-{condition}-{sequence_id}/complete.json",
                "trace_path": f"/sealed/full-{condition}-{sequence_id}/candidates.jsonl",
            } for condition, sequence_id in sorted(evaluator.EXPECTED_RUNS)],
            "causality_prefix_runs": [{
                "condition": condition,
                "sequence_id": SEQUENCE,
                "status": "complete", "failure_reason": None,
                "requested_frame_ids": list(range(4)),
                "output_path": f"/sealed/four-{condition}-TRAIN17minus25/complete.json",
                "trace_path": f"/sealed/four-{condition}-TRAIN17minus25/candidates.jsonl",
            } for condition in ("frozen", "adapted")],
        }
        by_sequence, by_identity = evaluator._validate_sealed_metadata(
            design, bindings, report, policy=policy)
        self.assertEqual(set(by_sequence), set(evaluator.SEQUENCE_SPECS))
        self.assertEqual(len(by_identity), 6)
        self.assertEqual(by_sequence[SEQUENCE]["runtime_contract_sha256"], "c" * 64)
        bad_report = dict(report, causality_prefix_runs=report["causality_prefix_runs"][:1])
        with self.assertRaisesRegex(ValueError, "exactly two causality-prefix"):
            evaluator._validate_sealed_metadata(design, bindings, bad_report, policy=policy)

    def test_expanded_output_identity_pins_head_and_runtime_closure_separately(self):
        spec = evaluator.SEQUENCE_SPECS[SEQUENCE]
        binding = {
            "generation_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "runtime_contract_sha256": "c" * 64,
            "source_closure_sha256": "d" * 64,
        }
        head = evaluator._head_expectation("adapted", evaluator.EXPANDED_HEAD_SHA256)
        inference = {
            "schema": "gotrack_inference_contract_v1",
            "profile": evaluator._INFERENCE_PROFILE,
            "geometry": {"pose_units": evaluator._EXPECTED_OUTPUT_POSE_UNITS},
            "head": head,
            "input": {
                "sequence_id": SEQUENCE,
                "physical_object_id": spec["physical_object_id"],
                "split": spec["split"],
                "generation_receipt_sha256": binding["generation_sha256"],
                "input_sha256": binding["input_sha256"],
                "clock_mode": "capture_seconds_v1",
                "video_sha256": "e" * 64,
                "mesh_sha256": "f" * 64,
            },
            "runtime_contract_sha256": binding["runtime_contract_sha256"],
            "runtime": {"source_closure_sha256": binding["source_closure_sha256"]},
        }
        result = evaluator._check_inference_identity(
            inference, condition="adapted", sequence_id=SEQUENCE,
            binding=binding, checkpoint_sha256=evaluator.EXPANDED_HEAD_SHA256)
        self.assertEqual(result["head"]["requested_sha256"], evaluator.EXPANDED_HEAD_SHA256)
        wrong_runtime = dict(inference, runtime_contract_sha256="9" * 64)
        with self.assertRaisesRegex(ValueError, "runtime contract differs"):
            evaluator._check_inference_identity(
                wrong_runtime, condition="adapted", sequence_id=SEQUENCE,
                binding=binding, checkpoint_sha256=evaluator.EXPANDED_HEAD_SHA256)

    def test_four_frame_prefix_outputs_are_joined_and_compared_with_full_video(self):
        spec = evaluator.SEQUENCE_SPECS[SEQUENCE]
        binding = {
            "generation_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "runtime_contract_sha256": "c" * 64,
            "source_closure_sha256": "d" * 64,
        }
        references = _reference_frames()
        snapshot = {}
        full = {"frozen": {}, "adapted": {}}

        def output_docs(condition):
            head = evaluator._head_expectation(
                condition, evaluator.EXPANDED_HEAD_SHA256)
            identity = {
                "schema": "gotrack_inference_contract_v1",
                "profile": evaluator._INFERENCE_PROFILE,
                "geometry": {"pose_units": evaluator._EXPECTED_OUTPUT_POSE_UNITS},
                "head": head,
                "input": {
                    "sequence_id": SEQUENCE,
                    "physical_object_id": spec["physical_object_id"],
                    "split": spec["split"],
                    "generation_receipt_sha256": binding["generation_sha256"],
                    "input_sha256": binding["input_sha256"],
                    "clock_mode": "capture_seconds_v1",
                    "video_sha256": "e" * 64,
                    "mesh_sha256": "f" * 64,
                },
                "runtime_contract_sha256": binding["runtime_contract_sha256"],
                "runtime": {"source_closure_sha256": binding["source_closure_sha256"]},
            }
            frames = []
            candidates = []
            for frame_id in range(4):
                pose = _pose(frame_id / 10.0).tolist()
                frame = {
                    "frameId": frame_id, "sourceFrameId": frame_id,
                    "timestamp_s": frame_id / 10.0,
                    "pose_state": "tracking", "render_state": "visible",
                    "failure_reason": None, "mask_state": "available",
                    "mask_measurement_state": "measured",
                    "cameraFromObject": pose, "gotrack_inference": identity,
                }
                frames.append(frame)
                candidate = {
                    "candidate_id": "candidate-0", "pose": pose,
                    "eligible": True, "gates": {"current_image_validation": True},
                    "rejection": None, "legacy_score": 0.5,
                }
                candidates.append({
                    "frame_id": frame_id, "source_frame_id": frame_id,
                    "timestamp_s": frame_id / 10.0,
                    "sequence_id": SEQUENCE, "object_id": spec["physical_object_id"],
                    "candidates": [candidate],
                    "rank_decision": {"selected_id": "candidate-0",
                                      "ordered_ids": ["candidate-0"],
                                      "scores": {"candidate-0": 0.5}},
                    "memory": {"last_valid_frame_id": frame_id,
                               "last_valid_pose": pose},
                    "pose_state": "tracking", "render_state": "visible",
                    "cameraFromObject": pose,
                    "provenance": {"gotrack_inference": identity},
                })
            return identity, frames, candidates

        for condition in ("frozen", "adapted"):
            identity, frames, candidates = output_docs(condition)
            full[condition][SEQUENCE] = {
                "complete": {i: {**row, "_pose": evaluator._pose_or_none(
                    row["cameraFromObject"], "fixture complete pose")}
                    for i, row in enumerate(frames)},
                "candidates": {row["frame_id"]: {**row, "_pose": evaluator._pose_or_none(
                    row["cameraFromObject"], "fixture trace pose")}
                    for row in candidates},
            }
            for directory, requested in ((f"full-{condition}-{SEQUENCE}", list(range(24))),
                                         (f"four-{condition}-TRAIN17minus25", list(range(4)))):
                complete = {
                    "requested_frame_ids": requested,
                    "clock_mode": "capture_seconds_v1",
                    "provenance": {"gotrack_inference": identity},
                    "frames": frames,
                }
                snapshot[f"{directory}/complete.json"] = json.dumps(complete).encode()
                snapshot[f"{directory}/candidates.jsonl"] = (
                    "\n".join(json.dumps(row) for row in candidates) + "\n").encode()

        prefixes, hashes = evaluator._load_causality_prefix_outputs(
            snapshot,
            {SEQUENCE: binding},
            {SEQUENCE: references},
            full,
            checkpoint_sha256=evaluator.EXPANDED_HEAD_SHA256,
        )
        self.assertEqual(len(prefixes), 2)
        self.assertEqual(hashes.keys(), {
            "four-frozen-TRAIN17minus25/complete.json",
            "four-frozen-TRAIN17minus25/candidates.jsonl",
            "four-adapted-TRAIN17minus25/complete.json",
            "four-adapted-TRAIN17minus25/candidates.jsonl",
        })
        self.assertTrue(all(row["causal_prefix_equal_to_full_video"] for row in prefixes))
        self.assertTrue(all(row["requested_count"] == 4 and row["produced_count"] == 4
                            for row in prefixes))

        prefix_candidates = "four-frozen-TRAIN17minus25/candidates.jsonl"

        def changed_prefix_trace(change):
            changed_snapshot = dict(snapshot)
            rows = [json.loads(line) for line in
                    changed_snapshot[prefix_candidates].decode().splitlines()]
            change(rows[0])
            changed_snapshot[prefix_candidates] = (
                "\n".join(json.dumps(row) for row in rows) + "\n").encode()
            return changed_snapshot

        semantic_changes = (
            lambda row: row["candidates"][0].update(legacy_score=0.75),
            lambda row: row["rank_decision"]["scores"].update({"candidate-0": 0.75}),
            lambda row: row["memory"].update(last_valid_frame_id=99),
        )
        for change in semantic_changes:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "differ in candidates"):
                    evaluator._load_causality_prefix_outputs(
                        changed_prefix_trace(change), {SEQUENCE: binding},
                        {SEQUENCE: references}, full,
                        checkpoint_sha256=evaluator.EXPANDED_HEAD_SHA256)

        profiling_only = changed_prefix_trace(lambda row: (
            row.update(timings_ms={"pose_total": 999.0},
                       resource_memory={"peak_rss_bytes": 999}),
            row["candidates"][0].update(appearance_summary_ms=999.0),
        ))
        evaluator._load_causality_prefix_outputs(
            profiling_only, {SEQUENCE: binding}, {SEQUENCE: references}, full,
            checkpoint_sha256=evaluator.EXPANDED_HEAD_SHA256)


class AggregationTests(unittest.TestCase):
    def test_condition_summaries_keep_train_and_used_val_denominators_separate(self):
        condition_results = {"frozen": {}, "adapted": {}}
        for condition in condition_results:
            for sequence_id, spec in evaluator.SEQUENCE_SPECS.items():
                condition_results[condition][sequence_id] = evaluator.evaluate_sequence(
                    sequence_id=sequence_id,
                    split=spec["split"],
                    object_id=spec["physical_object_id"],
                    reference_frames=_reference_frames(),
                    complete_frames={},
                    candidate_frames={},
                    condition=condition,
                )
        summaries = evaluator._summarize_conditions(condition_results)
        for condition_summary in summaries.values():
            train = condition_summary["split_aggregates"]["train"]
            used_val = condition_summary["split_aggregates"]["used_val"]
            self.assertEqual(train["split"], "train")
            self.assertEqual(train["partition"], "training_resubstitution")
            self.assertEqual(train["requested_count"], 16)
            self.assertEqual(used_val["split"], "validation")
            self.assertEqual(used_val["partition"],
                             "previously_used_development_validation")
            self.assertEqual(used_val["requested_count"], 8)


if __name__ == "__main__":
    unittest.main()
