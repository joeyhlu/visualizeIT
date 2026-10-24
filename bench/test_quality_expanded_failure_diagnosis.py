"""Synthetic contract tests for saved-output diagnosis; no prediction claim."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
import unittest

from . import quality_expanded_failure_diagnosis as diagnosis
from . import quality_postseal_pose_evaluate as evaluator


_CHECKPOINT = "a" * 64
_GENERATION = "b" * 64
_INPUT = "c" * 64
_VIDEO = "d" * 64
_MESH = "e" * 64
_RUNTIME = "f" * 64
_CLOSURE = "0" * 64
_MANIFEST = "1" * 64
_ARCHIVE = "2" * 64
_POLICY = {
    "policy_id": "expanded24",
    "manifest_sha256": _MANIFEST,
    "archive_sha256": _ARCHIVE,
    "diagnostic_sha256": "3" * 64,
    "checkpoint_sha256": _CHECKPOINT,
    "output_prefix": "full",
    "requested_frame_ids": tuple(range(24)),
}


def _pose(x: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _inference(condition: str, sequence_id: str) -> dict[str, Any]:
    spec = evaluator.SEQUENCE_SPECS[sequence_id]
    binding = {
        "generation_sha256": _GENERATION,
        "input_sha256": _INPUT,
        "runtime_contract_sha256": _RUNTIME,
        "source_closure_sha256": _CLOSURE,
    }
    head = ({
        "selection": "adapted_pose_head",
        "requested_sha256": _CHECKPOINT,
        "installed_sha256": _CHECKPOINT,
    } if condition == "adapted" else {
        "selection": "frozen_base",
        "requested_sha256": None,
        "installed_sha256": None,
    })
    return {
        "schema": "gotrack_inference_contract_v1",
        "profile": "gotrack_integer_legacy_synthetic_v1",
        "geometry": {"pose_units": "native_camera_from_object_metres;PnP_crop_camera_from_object_millimetres"},
        "head": head,
        "input": {
            "sequence_id": sequence_id,
            "physical_object_id": spec["physical_object_id"],
            "split": spec["split"],
            "generation_receipt_sha256": binding["generation_sha256"],
            "input_sha256": binding["input_sha256"],
            "clock_mode": "capture_seconds_v1",
            "video_sha256": _VIDEO,
            "mesh_sha256": _MESH,
        },
        "runtime_contract_sha256": _RUNTIME,
        "runtime": {"source_closure_sha256": _CLOSURE},
    }


def _trace_candidate(candidate_id: str, pose: list[list[float]], *, stats: Any = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "original_index": 0,
        "stage": "refinement",
        "source": "refinement",
        "pose": pose,
        "eligible": True,
        "rejection": None,
        "gates": {
            "current_image_validation": True,
            "backend_motion": True,
            "causal_motion": True,
        },
        "validation_stats": stats,
        "features": [],
        "ranker_features": None,
        "appearance_summary_ms": None,
        "legacy_score": 0.5,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    # These are intentionally synthetic evaluator fixtures, not observed metrics.
    return {
        "count": len(values), "mean": 0.2 if values else None,
        "median": 0.1 if values else None, "p95": 0.3 if values else None,
        "unit": "metres",
    }


def _fixture(*, multi: bool = False, suppression: bool = False,
              missing_stats: bool = False) -> tuple[diagnosis.AuthenticatedSnapshot, dict[str, Any]]:
    members: dict[str, bytes] = {
        "artifact-manifest.json": b"synthetic manifest bytes",
        "design.json": json.dumps({
            "frame_timestamps_seconds": [frame_id / 10 for frame_id in range(24)],
        }, sort_keys=True, separators=(",", ":")).encode(),
        "execution-bindings.json": b"synthetic bindings receipt",
        "paired-report.json": b"synthetic paired report receipt",
    }
    binding_by_sequence = {
        sequence_id: {
            "generation_sha256": _GENERATION,
            "input_sha256": _INPUT,
            "runtime_contract_sha256": _RUNTIME,
            "source_closure_sha256": _CLOSURE,
        }
        for sequence_id in diagnosis._SEQUENCES
    }
    report_by_identity = {}
    conditions_report: dict[str, Any] = {}

    for condition in diagnosis._CONDITIONS:
        sequence_summaries = []
        total_emitted = 0
        total_error_values: list[float] = []
        total_candidate_values: list[float] = []
        for sequence_id in diagnosis._SEQUENCES:
            spec = evaluator.SEQUENCE_SPECS[sequence_id]
            inference = _inference(condition, sequence_id)
            complete_frames = []
            trace_rows = []
            availability_rows = []
            error_rows = []
            before_rows = []
            after_rows = []
            sequence_error_values = []
            sequence_candidate_values = []
            for frame_id in range(24):
                timestamp = frame_id / 10
                is_suppressed = suppression and condition == "adapted" and sequence_id == diagnosis._SEQUENCES[0] and frame_id == 10
                emitted_pose = None if is_suppressed else _pose(0.01 * frame_id)
                pose_state = "lost" if is_suppressed else "tracking"
                render_state = "suppressed" if is_suppressed else "visible"
                complete_frame = {
                    "frameId": frame_id,
                    "sourceFrameId": frame_id,
                    "sequence_id": sequence_id,
                    "object_id": spec["physical_object_id"],
                    "timestamp_s": timestamp,
                    "cameraFromObject": emitted_pose,
                    "pose_state": pose_state,
                    "render_state": render_state,
                    "gotrack_inference": inference,
                }
                complete_frames.append(complete_frame)

                stats = None if missing_stats and condition == "adapted" and sequence_id == diagnosis._SEQUENCES[0] and frame_id == 9 else {
                    "correspondences": 50,
                    "inliers": 40,
                    "model_motion": {"state": "disabled"},
                    "seconds_since_last_valid": 0.25,
                    "translation_from_last_valid_m": 0.10,
                    "rotation_from_last_valid_degrees": 20.0,
                }
                candidate_pose = _pose(0.01 * frame_id)
                candidate_id = "candidate-main"
                candidates = [_trace_candidate(candidate_id, candidate_pose, stats=stats)]
                if multi and condition == "adapted" and sequence_id == diagnosis._SEQUENCES[0] and frame_id == 9:
                    candidates.append({
                        **_trace_candidate("candidate-second", _pose(0.03), stats=stats),
                        "original_index": 1,
                    })
                trace_row = {
                    "trace_schema_version": 1,
                    "frame_id": frame_id,
                    "source_frame_id": frame_id,
                    "sequence_id": sequence_id,
                    "object_id": spec["physical_object_id"],
                    "timestamp_s": timestamp,
                    "clock_mode": "capture_seconds_v1",
                    "context": {"sequence_id": sequence_id, "mask_state": "available", "recovery": False},
                    "memory": {"last_valid_frame_id": max(0, frame_id - 1),
                               "last_valid_timestamp_s": timestamp - 0.25},
                    "candidates": candidates,
                    "rank_decision": {"selected_id": candidate_id, "ordered_ids": [c["candidate_id"] for c in candidates], "scores": {}},
                    "legacy_choice_candidate_id": candidate_id,
                    "pose_state": pose_state,
                    "render_state": render_state,
                    "failure_reason": None,
                    "cameraFromObject": emitted_pose,
                    "provenance": {"gotrack_inference": inference},
                }
                trace_rows.append(trace_row)

                error = None if is_suppressed else (0.2 if condition == "adapted" and sequence_id == diagnosis._SEQUENCES[0] and frame_id == 9 else 0.001)
                minimum_before = 0.15 if condition == "adapted" and sequence_id == diagnosis._SEQUENCES[0] and frame_id == 9 else 0.001
                minimum_after = 0.10 if condition == "adapted" and sequence_id == diagnosis._SEQUENCES[0] and frame_id == 9 else 0.001
                availability_rows.append({
                    "sequence_id": sequence_id,
                    "physical_object_id": spec["physical_object_id"],
                    "frame_id": frame_id,
                    "output_row_available": True,
                    "pose_state": pose_state,
                    "render_state": render_state,
                    "mask_state": "available",
                    "mask_measurement_state": "measured",
                    "failure_reason": None,
                    "emitted_pose_available": error is not None,
                    "emitted_pose_unavailable_reason": None if error is not None else "null_cameraFromObject",
                    "translation_error_m": error,
                })
                if error is not None:
                    error_rows.append({"sequence_id": sequence_id, "physical_object_id": spec["physical_object_id"],
                                       "frame_id": frame_id, "translation_error_m": error})
                    sequence_error_values.append(error)
                before_rows.append({"sequence_id": sequence_id, "physical_object_id": spec["physical_object_id"],
                                    "frame_id": frame_id, "minimum_translation_error_m": minimum_before})
                after_rows.append({"sequence_id": sequence_id, "physical_object_id": spec["physical_object_id"],
                                   "frame_id": frame_id, "minimum_translation_error_m": minimum_after})
                sequence_candidate_values.append(minimum_after)
            prefix = evaluator._expected_output_member(condition, sequence_id, "complete.json", "full").rsplit("/", 1)[0]
            complete_doc = {
                "requested_frame_ids": list(range(24)),
                "clock_mode": "capture_seconds_v1",
                "provenance": {"gotrack_inference": inference},
                "tracking_settings": {
                    "rotation_margin_degrees": 2.0,
                    "translation_margin_m": 0.01,
                    "time_policy": {
                        "mode": "capture_seconds_v1",
                        "max_angular_rate_deg_s": 120.0,
                        "max_translation_rate_m_s": 0.5,
                        "private_pose_memory_s": 0.5,
                    },
                },
                "frames": complete_frames,
            }
            complete_name = f"{prefix}/complete.json"
            candidates_name = f"{prefix}/candidates.jsonl"
            members[complete_name] = json.dumps(complete_doc, sort_keys=True, separators=(",", ":")).encode()
            members[candidates_name] = ("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"))
                                                    for row in trace_rows) + "\n").encode()
            report_by_identity[(condition, sequence_id)] = {"status": "complete"}
            sequence_error_values_summary = sequence_error_values
            sequence_summaries.append({
                "condition": condition,
                "sequence_id": sequence_id,
                "split": spec["split"],
                "physical_object_id": spec["physical_object_id"],
                "requested_count": 24,
                "produced_count": 24,
                "missing_output_row_count": 0,
                "candidate_trace_row_count": 24,
                "emitted_pose_count": len(error_rows),
                "requested_frame_ids": list(range(24)),
                "frame_availability": {
                    "requested_count": 24,
                    "output_row_available_count": 24,
                    "emitted_pose_available_count": len(error_rows),
                    "rows": availability_rows,
                },
                "emitted_translation_errors": {
                    **_distribution(sequence_error_values_summary), "rows": error_rows,
                },
                "candidate_coverage": {
                    "minimum_translation_error_before_gates": {
                        **_distribution([row["minimum_translation_error_m"] for row in before_rows]),
                        "counterfactual_within_head_candidate_set": True,
                        "rows": before_rows,
                    },
                    "minimum_translation_error_after_gates": {
                        **_distribution([row["minimum_translation_error_m"] for row in after_rows]),
                        "counterfactual_within_head_candidate_set": True,
                        "rows": after_rows,
                    },
                },
            })
            total_emitted += len(error_rows)
            total_error_values.extend(sequence_error_values)
            total_candidate_values.extend(sequence_candidate_values)

        conditions_report[condition] = {
            "condition": condition,
            "requested_count": 72,
            "emitted_pose_count": total_emitted,
            "missing_output_row_count": 0,
            "candidate_trace_row_count": 72,
            "emitted_translation_errors": {**_distribution(total_error_values), "rows": []},
            "candidate_coverage": {
                "minimum_translation_error_before_gates": {
                    **_distribution(total_candidate_values),
                    "counterfactual_within_head_candidate_set": True,
                },
                "minimum_translation_error_after_gates": {
                    **_distribution(total_candidate_values),
                    "counterfactual_within_head_candidate_set": True,
                },
            },
            "split_aggregates": {"train_resubstitution": {"sequence_count": 2},
                                 "used_validation": {"sequence_count": 1}},
            "sequences": sequence_summaries,
        }

    # Causality-only prefix members are included in the synthetic input receipt,
    # but never joined into the full-video diagnosis.
    for condition in diagnosis._CONDITIONS:
        prefix = f"four-{condition}-TRAIN17minus25"
        members[f"{prefix}/complete.json"] = b"prefix complete bytes excluded from quality rows"
        members[f"{prefix}/candidates.jsonl"] = b"prefix trace bytes excluded from quality rows"

    receipt = {
        "sealed_archive_sha256": _ARCHIVE,
        "sealed_manifest_sha256": _MANIFEST,
        "authenticated_sealed_member_count_including_manifest": len(members),
        "consumed_sealed_members_sha256": {
            name: hashlib.sha256(members[name]).hexdigest()
            for name in diagnosis._expected_consumed_members("full")
        },
        "reference_inputs": {"synthetic_sequence": {"poses_jsonl_sha256": "4" * 64}},
        "diagnostic_sha256": _POLICY["diagnostic_sha256"],
    }
    report = {
        "schema_version": 1,
        "kind": evaluator._EVALUATOR_KIND,
        "status": "development_synthetic_pose_agreement_only",
        "claim_scope": {"quality_promotion": False},
        "input_hash_receipt": receipt,
        "conditions": conditions_report,
    }
    snapshot = diagnosis.AuthenticatedSnapshot(
        policy=_POLICY,
        sealed_manifest_sha256=_MANIFEST,
        sealed_archive_sha256=_ARCHIVE,
        evaluation_policy_sha256="6" * 64,
        evaluation_report_sha256="5" * 64,
        members=members,
        binding_by_sequence=binding_by_sequence,
        report_by_identity=report_by_identity,
    )
    return snapshot, report


class ExpandedFailureDiagnosisTests(unittest.TestCase):
    def test_manufactured_join_keeps_144_rows_and_fixed_windows(self) -> None:
        snapshot, report = _fixture()
        # Model a caller-pinned replay whose adapted error maximum differs from
        # the predeclared reviewed-study focus frame. The diagnostic must keep
        # its fixed context and must not relabel it as this report's maximum.
        adapted_seq1 = report["conditions"]["adapted"]["sequences"][0]
        availability_row = next(row for row in adapted_seq1["frame_availability"]["rows"]
                                if row["frame_id"] == 0)
        availability_row["translation_error_m"] = 0.5
        error_row = next(row for row in adapted_seq1["emitted_translation_errors"]["rows"]
                         if row["frame_id"] == 0)
        error_row["translation_error_m"] = 0.5
        result = diagnosis.build_diagnosis(authenticated_snapshot=snapshot, evaluation_report=report)
        self.assertEqual(len(result["full_frames"]), 144)
        self.assertEqual(len(result["fixed_focus_context_windows"]), 3)
        self.assertEqual(result["fixed_focus_context_windows"][0]["frame_ids"], [7, 8, 9, 10, 11])
        self.assertEqual(result["fixed_focus_context_windows"][2]["frame_ids"], [9, 10, 11, 12, 13])
        self.assertEqual(result["fixed_focus_context_windows"][0]["fixed_focus_frame_id"], 9)
        self.assertEqual(len(result["fixed_focus_context_windows"][0]["condition_rows"]["frozen"]), 5)
        self.assertFalse(result["fixed_focus_selection"]["current_report_maximum_recomputed"])
        self.assertEqual(
            result["fixed_focus_selection"]["reference_evaluation_report_sha256"],
            "7e8a48f829bbe971cde8147242f0d5ed666c613329c26bdaf9a4e9b0446ac70f",
        )
        reported_error_at_zero = next(row["emitted_translation_error_m_from_evaluator"]
                                      for row in result["full_frames"]
                                      if row["condition"] == "adapted" and
                                      row["sequence_id"] == diagnosis._SEQUENCES[0] and
                                      row["frame_id"] == 0)
        self.assertEqual(reported_error_at_zero, 0.5)
        # Synthetic evaluator fixture records emitted error .20 m and after-gate
        # minimum .10 m; the diagnosis subtracts only those supplied values.
        row = next(row for row in result["full_frames"]
                   if row["condition"] == "adapted" and row["sequence_id"] == diagnosis._SEQUENCES[0]
                   and row["frame_id"] == 9)
        self.assertAlmostEqual(row["selected_minus_candidate_minimum_m"]["after_gates"], 0.10)

    def test_single_multi_candidate_and_suppressed_selected_pose(self) -> None:
        snapshot, report = _fixture(multi=True, suppression=True)
        result = diagnosis.build_diagnosis(authenticated_snapshot=snapshot, evaluation_report=report)
        target = next(row for row in result["full_frames"]
                      if row["condition"] == "adapted" and row["sequence_id"] == diagnosis._SEQUENCES[0]
                      and row["frame_id"] == 9)
        self.assertEqual(target["candidate_counts"]["cardinality"], "multiple")
        self.assertEqual(target["candidate_counts"]["valid_pose_count_before_gates"], 2)
        single = next(row for row in result["full_frames"]
                      if row["condition"] == "frozen" and row["sequence_id"] == diagnosis._SEQUENCES[0]
                      and row["frame_id"] == 9)
        self.assertEqual(single["candidate_counts"]["cardinality"], "single")
        suppressed = next(row for row in result["full_frames"]
                          if row["condition"] == "adapted" and row["sequence_id"] == diagnosis._SEQUENCES[0]
                          and row["frame_id"] == 10)
        self.assertFalse(suppressed["selected_candidate_matches_emitted_pose_exactly"])
        self.assertIsNone(suppressed["selected_candidate_error_m_from_evaluator"])
        self.assertIsNone(suppressed["selected_minus_candidate_minimum_m"]["after_gates"])
        self.assertEqual(suppressed["evaluator_frame_availability"]["render_state"], "suppressed")

    def test_missing_validation_stats_are_explicit_not_fatal(self) -> None:
        snapshot, report = _fixture(missing_stats=True)
        result = diagnosis.build_diagnosis(authenticated_snapshot=snapshot, evaluation_report=report)
        row = next(row for row in result["full_frames"]
                   if row["condition"] == "adapted" and row["sequence_id"] == diagnosis._SEQUENCES[0]
                   and row["frame_id"] == 9)
        self.assertIsNone(row["candidate_evidence"][0]["validation_stats"])
        self.assertEqual(row["candidate_evidence"][0]["backend_motion_status_from_recorded_metadata"], "unknown")
        self.assertEqual(row["candidate_evidence"][0]["appearance_evidence_status"], "missing")

    def test_duplicate_and_mismatched_trace_ids_are_refused(self) -> None:
        for mutate in ("duplicate", "mismatch"):
            with self.subTest(mutate=mutate):
                snapshot, report = _fixture()
                name = evaluator._expected_output_member("adapted", diagnosis._SEQUENCES[0], "candidates.jsonl", "full")
                rows = snapshot.members[name].decode().splitlines()
                row = json.loads(rows[0])
                if mutate == "duplicate":
                    rows.append(rows[0])
                else:
                    row["sequence_id"] = diagnosis._SEQUENCES[1]
                    rows[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
                snapshot.members[name] = ("\n".join(rows) + "\n").encode()
                report["input_hash_receipt"]["consumed_sealed_members_sha256"][name] = hashlib.sha256(snapshot.members[name]).hexdigest()
                with self.assertRaises(ValueError):
                    diagnosis.build_diagnosis(authenticated_snapshot=snapshot, evaluation_report=report)

    def test_receipt_hash_tampering_is_refused(self) -> None:
        snapshot, report = _fixture()
        report["input_hash_receipt"]["consumed_sealed_members_sha256"]["design.json"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "does not match authenticated sealed member"):
            diagnosis.build_diagnosis(authenticated_snapshot=snapshot, evaluation_report=report)

    def test_physical_bounds_match_tracker_formula(self) -> None:
        # SequentialTracker uses margin + rate * elapsed physical seconds;
        # these hand-computed values are .01 + .5*.25 = .135 m and 2 + 120*.25 = 32 degrees.
        candidate = {"validation_stats": {
            "seconds_since_last_valid": 0.25,
            "translation_from_last_valid_m": 0.10,
            "rotation_from_last_valid_degrees": 20.0,
        }}
        policy = {
            "translation_margin_m": 0.01,
            "max_translation_rate_m_s": 0.5,
            "rotation_margin_degrees": 2.0,
            "max_angular_rate_deg_s": 120.0,
            "private_pose_memory_s": 0.5,
            "mode": "capture_seconds_v1",
        }
        evidence = diagnosis._motion_bound_evidence({"memory": {"last_valid_frame_id": 4}}, candidate, policy)
        self.assertAlmostEqual(evidence["translation_bound_m"], 0.135)
        self.assertEqual(evidence["rotation_bound_degrees"], 32.0)
        self.assertTrue(evidence["within_recorded_physical_bounds"])

    def test_create_only_writer_refuses_overwrite(self) -> None:
        with self.subTest("existing destination"):
            # The CLI delegates to this existing create-only writer.
            from tempfile import TemporaryDirectory
            temp_root = Path.cwd() / ".cache" / "workspace.cache"
            temp_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=temp_root) as temp_dir:
                target = Path(temp_dir) / "report.json"
                target.write_text("keep", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    evaluator.write_new_report(target, {"ok": True}, input_roots=())
                self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_import_does_not_load_torch(self) -> None:
        command = [sys.executable, "-c",
                   "import sys; import bench.quality_expanded_failure_diagnosis; "
                   "assert 'torch' not in sys.modules"]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
