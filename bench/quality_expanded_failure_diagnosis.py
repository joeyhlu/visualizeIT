"""Diagnose saved expanded GoTrack outputs without rerunning inference.

This report joins the authenticated candidate traces to an already-produced
post-seal evaluation. It does not load reference sidecars or recompute pose
errors. Candidate minima remain explicit counterfactuals within the recorded
head's candidate set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import quality_postseal_pose_evaluate as evaluator


_SEQUENCES = tuple(evaluator.SEQUENCE_SPECS)
_CONDITIONS = ("frozen", "adapted")
_REQUESTED = tuple(range(24))
_FOCUS_WINDOWS = (
    ("train_resubstitution_object_17_pitch_minus_25", _SEQUENCES[0], 9),
    ("train_resubstitution_object_17_pitch_plus_25", _SEQUENCES[1], 9),
    ("used_validation_object_27_pitch_minus_25", _SEQUENCES[2], 11),
)
_FOCUS_REFERENCE_EVALUATION_SHA256 = "7e8a48f829bbe971cde8147242f0d5ed666c613329c26bdaf9a4e9b0446ac70f"
_MISSING_INTERNAL_EVIDENCE = {
    "iteration_indexed_flow_and_confidence": (
        "not present in the sealed candidate traces; the traces contain final candidate summaries, "
        "not per-refinement-iteration flow or confidence payloads"
    ),
    "three_dimensional_point_conditioning": (
        "not present in the sealed candidate traces; no point-cloud conditioning diagnostic was recorded"
    ),
    "ransac_sample_indices_and_inliers": (
        "not present in the sealed candidate traces; validation_stats.inliers are current-image "
        "residual-validation inliers, not OpenCV RANSAC inliers"
    ),
    "solver_trajectory": (
        "not present in the sealed candidate traces; no per-iteration PnP or LM solver trajectory was recorded"
    ),
}


@dataclass(frozen=True, slots=True)
class AuthenticatedSnapshot:
    """Bytes and metadata already authenticated by the bounded CLI loader."""

    policy: Mapping[str, Any]
    sealed_manifest_sha256: str
    sealed_archive_sha256: str
    evaluation_policy_sha256: str
    evaluation_report_sha256: str
    members: Mapping[str, bytes]
    binding_by_sequence: Mapping[str, Mapping[str, str]]
    report_by_identity: Mapping[tuple[str, str], Mapping[str, Any]]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative" if nonnegative
                         else f"{name} must be finite")
    return result


def _expected_consumed_members(output_prefix: str) -> frozenset[str]:
    names = {"design.json", "execution-bindings.json", "paired-report.json"}
    for condition in _CONDITIONS:
        for sequence_id in _SEQUENCES:
            for filename in ("complete.json", "candidates.jsonl"):
                names.add(evaluator._expected_output_member(
                    condition, sequence_id, filename, output_prefix))
    for condition in _CONDITIONS:
        prefix = f"four-{condition}-TRAIN17minus25"
        names.update((f"{prefix}/complete.json", f"{prefix}/candidates.jsonl"))
    return frozenset(names)


def _validate_evaluation_receipt(
    snapshot: AuthenticatedSnapshot,
    evaluation_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    if evaluation_report.get("kind") != evaluator._EVALUATOR_KIND:
        raise ValueError("Evaluation report is not the post-seal pose evaluator output")
    if evaluation_report.get("schema_version") != 1:
        raise ValueError("Evaluation report schema version is unsupported")
    if evaluation_report.get("status") != "development_synthetic_pose_agreement_only":
        raise ValueError("Evaluation report has an unexpected claim status")
    receipt = evaluation_report.get("input_hash_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Evaluation report lacks its input hash receipt")
    expected_receipt_fields = {
        "sealed_archive_sha256", "sealed_manifest_sha256",
        "authenticated_sealed_member_count_including_manifest",
        "consumed_sealed_members_sha256", "reference_inputs", "diagnostic_sha256",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError("Evaluation input receipt has unexpected or missing fields")
    if receipt.get("sealed_archive_sha256") != snapshot.policy.get("archive_sha256"):
        raise ValueError("Evaluation receipt archive SHA-256 differs from the authenticated policy")
    if receipt.get("sealed_archive_sha256") != snapshot.sealed_archive_sha256:
        raise ValueError("Evaluation receipt archive SHA-256 differs from the authenticated archive")
    if receipt.get("sealed_manifest_sha256") != snapshot.policy.get("manifest_sha256"):
        raise ValueError("Evaluation receipt manifest SHA-256 differs from the authenticated policy")
    if receipt.get("sealed_manifest_sha256") != snapshot.sealed_manifest_sha256:
        raise ValueError("Evaluation receipt manifest SHA-256 differs from the authenticated seal")
    if receipt.get("authenticated_sealed_member_count_including_manifest") != len(snapshot.members):
        raise ValueError("Evaluation receipt sealed-member count differs from the authenticated snapshot")
    if receipt.get("diagnostic_sha256") != snapshot.policy.get("diagnostic_sha256"):
        raise ValueError("Evaluation receipt diagnostic SHA-256 differs from the authenticated policy")

    consumed = receipt.get("consumed_sealed_members_sha256")
    if not isinstance(consumed, Mapping):
        raise ValueError("Evaluation receipt consumed-member hashes must be an object")
    expected_names = _expected_consumed_members(str(snapshot.policy.get("output_prefix")))
    if set(consumed) != expected_names:
        raise ValueError("Evaluation receipt consumed-member set differs from the expanded evaluation contract")
    for name, digest in consumed.items():
        raw = snapshot.members.get(name)
        if not isinstance(raw, bytes) or evaluator._sha256(raw) != digest:
            raise ValueError(f"Evaluation receipt does not match authenticated sealed member {name}")
    reference_inputs = receipt.get("reference_inputs")
    if not isinstance(reference_inputs, Mapping) or not reference_inputs:
        raise ValueError("Evaluation receipt lacks authenticated reference-input hashes")
    return receipt


def _index_report_rows(
    rows: Any,
    *,
    collection_name: str,
    frame_key: str,
) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation report {collection_name} must be a list")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Evaluation report {collection_name} contains a non-object row")
        frame_id = row.get(frame_key)
        if type(frame_id) is not int or frame_id not in _REQUESTED:
            raise ValueError(f"Evaluation report {collection_name} has an unexpected frame ID")
        if frame_id in indexed:
            raise ValueError(f"Evaluation report {collection_name} repeats frame ID {frame_id}")
        indexed[frame_id] = dict(row)
    return indexed


def _distribution_summary(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation report {name} distribution is malformed")
    count = value.get("count")
    if type(count) is not int or count < 0:
        raise ValueError(f"Evaluation report {name} distribution count is invalid")
    result: dict[str, Any] = {"count": count}
    for field in ("mean", "median", "p95"):
        value_for_field = value.get(field)
        result[field] = (None if value_for_field is None else
                         _number(value_for_field, f"{name}.{field}", nonnegative=True))
    if value.get("unit") != "metres":
        raise ValueError(f"Evaluation report {name} distribution unit must be metres")
    result["unit"] = "metres"
    return result


def _index_sequences(condition: Mapping[str, Any], *, condition_name: str) -> dict[str, dict[str, Any]]:
    rows = condition.get("sequences")
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation report {condition_name} sequences must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Evaluation report {condition_name} sequence row is malformed")
        sequence_id = row.get("sequence_id")
        if sequence_id not in _SEQUENCES or sequence_id in result:
            raise ValueError(f"Evaluation report {condition_name} has an unexpected or duplicate sequence")
        result[sequence_id] = dict(row)
    if set(result) != set(_SEQUENCES):
        raise ValueError(f"Evaluation report {condition_name} sequence set is incomplete")
    return result


def _index_errors(sequence_report: Mapping[str, Any], sequence_id: str) -> dict[int, float]:
    section = sequence_report.get("emitted_translation_errors")
    if not isinstance(section, Mapping) or section.get("unit") != "metres":
        raise ValueError("Evaluation report emitted-error section is malformed")
    rows = section.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Evaluation report emitted-error rows must be a list")
    result: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("sequence_id") != sequence_id:
            raise ValueError("Evaluation report emitted-error sequence identity mismatch")
        frame_id = row.get("frame_id")
        if type(frame_id) is not int or frame_id not in _REQUESTED or frame_id in result:
            raise ValueError("Evaluation report emitted-error frame identity is invalid or duplicated")
        result[frame_id] = _number(row.get("translation_error_m"),
                                   "emitted translation error", nonnegative=True)
    return result


def _index_candidate_minima(
    sequence_report: Mapping[str, Any],
    *,
    sequence_id: str,
    gate_stage: str,
) -> dict[int, float]:
    coverage = sequence_report.get("candidate_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("Evaluation report candidate coverage is malformed")
    field = f"minimum_translation_error_{gate_stage}"
    section = coverage.get(field)
    if not isinstance(section, Mapping) or section.get("unit") != "metres":
        raise ValueError(f"Evaluation report {field} section is malformed")
    if section.get("counterfactual_within_head_candidate_set") is not True:
        raise ValueError(f"Evaluation report {field} is not identified as a within-head counterfactual")
    rows = section.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation report {field} rows must be a list")
    result: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("sequence_id") != sequence_id:
            raise ValueError(f"Evaluation report {field} sequence identity mismatch")
        frame_id = row.get("frame_id")
        if type(frame_id) is not int or frame_id not in _REQUESTED or frame_id in result:
            raise ValueError(f"Evaluation report {field} has an invalid or duplicate frame ID")
        result[frame_id] = _number(row.get("minimum_translation_error_m"),
                                   f"{field} minimum error", nonnegative=True)
    return result


def _pose_equal(first: Any, second: Any) -> bool:
    """Exact JSON-value comparison; this is identity checking, not pose scoring."""
    return first is not None and second is not None and first == second


def _motion_policy(complete_doc: Mapping[str, Any]) -> dict[str, Any] | None:
    settings = complete_doc.get("tracking_settings")
    if not isinstance(settings, Mapping):
        return None
    time_policy = settings.get("time_policy")
    if not isinstance(time_policy, Mapping) or time_policy.get("mode") != "capture_seconds_v1":
        return None
    positive = (
        ("max_angular_rate_deg_s", time_policy.get("max_angular_rate_deg_s")),
        ("max_translation_rate_m_s", time_policy.get("max_translation_rate_m_s")),
        ("private_pose_memory_s", time_policy.get("private_pose_memory_s")),
    )
    values = {key: _number(value, f"tracking_settings.{key}", nonnegative=True)
              for key, value in positive}
    if any(values[key] <= 0.0 for key, _ in positive):
        raise ValueError("Physical TimePolicy rate and memory settings must be positive")
    for key in ("rotation_margin_degrees", "translation_margin_m"):
        values[key] = _number(settings.get(key), f"tracking_settings.{key}", nonnegative=True)
    values["mode"] = "capture_seconds_v1"
    return values


def _np_isclose(value: float, bound: float, *, rtol: float, atol: float) -> bool:
    # Match numpy.isclose(value, bound, ...) used by SequentialTracker: its
    # relative tolerance is scaled against the recorded bound argument.
    return abs(value - bound) <= atol + rtol * abs(bound)


def _motion_bound_evidence(
    trace_row: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    motion_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stats = candidate.get("validation_stats")
    if not isinstance(stats, Mapping):
        stats = {}
    elapsed = stats.get("seconds_since_last_valid")
    distance = stats.get("translation_from_last_valid_m")
    angle = stats.get("rotation_from_last_valid_degrees")
    if motion_policy is None or elapsed is None:
        return {
            "status": "unavailable",
            "seconds_since_last_valid": elapsed,
            "translation_from_last_valid_m": distance,
            "rotation_from_last_valid_degrees": angle,
            "translation_bound_m": None,
            "rotation_bound_degrees": None,
            "within_recorded_physical_bounds": None,
            "formula": "translation_margin_m + max_translation_rate_m_s * elapsed_seconds; "
                       "rotation_margin_degrees + max_angular_rate_deg_s * elapsed_seconds",
        }
    elapsed_s = _number(elapsed, "seconds_since_last_valid", nonnegative=True)
    # These are the same physical-clock bounds applied by SequentialTracker.
    translation_bound = (motion_policy["translation_margin_m"] +
                         motion_policy["max_translation_rate_m_s"] * elapsed_s)
    rotation_bound = (motion_policy["rotation_margin_degrees"] +
                      motion_policy["max_angular_rate_deg_s"] * elapsed_s)
    distance_value = None if distance is None else _number(
        distance, "translation_from_last_valid_m", nonnegative=True)
    angle_value = None if angle is None else _number(
        angle, "rotation_from_last_valid_degrees", nonnegative=True)
    if distance_value is None or angle_value is None:
        within = None
    else:
        translation_ok = (distance_value <= translation_bound or _np_isclose(
            distance_value, translation_bound, rtol=1e-12, atol=1e-12))
        rotation_ok = (angle_value <= rotation_bound or _np_isclose(
            angle_value, rotation_bound, rtol=1e-12, atol=1e-9))
        within = bool(translation_ok and rotation_ok)
    memory = trace_row.get("memory") if isinstance(trace_row, Mapping) else None
    last_valid_frame_id = memory.get("last_valid_frame_id") if isinstance(memory, Mapping) else None
    return {
        "status": "derived_from_recorded_stats_and_authenticated_settings",
        "last_valid_frame_id": last_valid_frame_id,
        "seconds_since_last_valid": elapsed_s,
        "translation_from_last_valid_m": distance_value,
        "rotation_from_last_valid_degrees": angle_value,
        "translation_bound_m": translation_bound,
        "rotation_bound_degrees": rotation_bound,
        "within_recorded_physical_bounds": within,
        "formula": "translation_margin_m + max_translation_rate_m_s * elapsed_seconds; "
                   "rotation_margin_degrees + max_angular_rate_deg_s * elapsed_seconds",
        "inclusive_boundary_comparison": True,
    }


def _backend_motion_status(stats: Any) -> str:
    if not isinstance(stats, Mapping) or "model_motion" not in stats:
        return "unknown"
    model_motion = stats.get("model_motion")
    if not isinstance(model_motion, Mapping):
        return "unknown"
    state = model_motion.get("state")
    if state == "disabled":
        return "disabled"
    if state == "applied" or model_motion.get("validation_applied") is True:
        return "applied"
    return "unknown"


def _candidate_evidence(
    candidate: Mapping[str, Any],
    *,
    trace_row: Mapping[str, Any] | None,
    motion_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stats = candidate.get("validation_stats")
    gates = candidate.get("gates")
    backend_status = _backend_motion_status(stats)
    ranker_features = candidate.get("ranker_features")
    if ranker_features is None:
        appearance_status = "missing"
    else:
        appearance_status = "recorded_feature_payload_present_not_accuracy_evidence"
    return {
        "candidate_id": candidate.get("candidate_id"),
        "original_index": candidate.get("original_index"),
        "stage": candidate.get("stage"),
        "source": candidate.get("source"),
        "eligible": candidate.get("eligible"),
        "rejection": candidate.get("rejection"),
        "gates": dict(gates) if isinstance(gates, Mapping) else gates,
        "validation_stats": stats,
        "validation_stats_inlier_semantics": (
            "quality_contract.validate current-image residual inliers; these are distinct from RANSAC inliers"
            if isinstance(stats, Mapping) and "inliers" in stats else None
        ),
        "backend_motion_status_from_recorded_metadata": backend_status,
        "backend_motion_stats": stats.get("model_motion") if isinstance(stats, Mapping) else None,
        "backend_motion_gate_interpretation": (
            "The recorded backend motion metadata is disabled; a true gate value does not establish an independent check"
            if backend_status == "disabled" and isinstance(gates, Mapping) and gates.get("backend_motion") is True
            else "Interpret only from the retained gate and metadata fields"
        ),
        "appearance_evidence_status": appearance_status,
        "ranker_features": ranker_features,
        "appearance_summary_ms_duration_only": candidate.get("appearance_summary_ms"),
        "appearance_duration_semantics": "timing only; not appearance error or quality",
        "motion_bound_evidence": _motion_bound_evidence(trace_row, candidate, motion_policy),
    }


def _trace_frame_rows(
    snapshot: AuthenticatedSnapshot,
    evaluation_report: Mapping[str, Any],
    *,
    condition_name: str,
    sequence_id: str,
    sequence_report: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    prefix = str(snapshot.policy["output_prefix"])
    complete_name = evaluator._expected_output_member(
        condition_name, sequence_id, "complete.json", prefix)
    candidates_name = evaluator._expected_output_member(
        condition_name, sequence_id, "candidates.jsonl", prefix)
    complete_doc = evaluator._parse_json_bytes(snapshot.members[complete_name], complete_name)
    raw_candidate_rows = evaluator._parse_jsonl_bytes(snapshot.members[candidates_name], candidates_name)
    if sequence_report.get("candidate_trace_row_count") != len(raw_candidate_rows):
        raise ValueError("Evaluation candidate-trace row count differs from authenticated trace bytes")
    if (sequence_report.get("requested_count") != len(_REQUESTED) or
            sequence_report.get("requested_frame_ids") != list(_REQUESTED)):
        raise ValueError("Evaluation sequence denominator differs from the 24-frame policy")

    # The timing map comes only from the authenticated design. These are row
    # identities, not labels: no reference pose file or reference-root path is read.
    design = evaluator._parse_json_bytes(snapshot.members["design.json"], "design.json")
    timestamps = design.get("frame_timestamps_seconds") if isinstance(design, Mapping) else None
    if timestamps != [frame_id / 10 for frame_id in _REQUESTED]:
        raise ValueError("Authenticated design lacks the expected physical 10 Hz row identity table")
    reference_rows = {
        frame_id: {"source_frame_id": frame_id, "timestamp_s": timestamps[frame_id]}
        for frame_id in _REQUESTED
    }
    complete_rows, candidate_by_frame, _identity = evaluator._validate_output_and_candidate_rows(
        complete_doc,
        raw_candidate_rows,
        sequence_id=sequence_id,
        object_id=evaluator.SEQUENCE_SPECS[sequence_id]["physical_object_id"],
        split=evaluator.SEQUENCE_SPECS[sequence_id]["split"],
        condition=condition_name,
        binding=snapshot.binding_by_sequence[sequence_id],
        reference_frames=reference_rows,
        requested_frame_ids=_REQUESTED,
        checkpoint_sha256=str(snapshot.policy["checkpoint_sha256"]),
    )

    # Candidate and complete row identities have now passed the same parser
    # checks as post-seal evaluation. Index original decoded trace rows so every
    # field remains available in the diagnosis unchanged.
    trace_by_frame: dict[int, dict[str, Any]] = {}
    for row in raw_candidate_rows:
        frame_id = row.get("frame_id")
        if type(frame_id) is not int or frame_id not in _REQUESTED:
            raise ValueError(f"Candidate trace has an unexpected frame ID in {candidates_name}")
        if frame_id in trace_by_frame:
            raise ValueError(f"Candidate trace repeats frame ID {frame_id} in {candidates_name}")
        trace_by_frame[frame_id] = dict(row)

    availability = sequence_report.get("frame_availability")
    if not isinstance(availability, Mapping):
        raise ValueError("Evaluation sequence lacks frame availability rows")
    availability_rows = _index_report_rows(
        availability.get("rows"), collection_name="frame availability", frame_key="frame_id")
    if set(availability_rows) != set(_REQUESTED):
        raise ValueError("Evaluation frame availability must retain all 24 requested frames")
    emitted_errors = _index_errors(sequence_report, sequence_id)
    before_minima = _index_candidate_minima(
        sequence_report, sequence_id=sequence_id, gate_stage="before_gates")
    after_minima = _index_candidate_minima(
        sequence_report, sequence_id=sequence_id, gate_stage="after_gates")

    motion_policy = _motion_policy(complete_doc)
    all_rows: dict[int, dict[str, Any]] = {}
    for frame_id in _REQUESTED:
        availability_row = availability_rows[frame_id]
        emitted_error = emitted_errors.get(frame_id)
        emitted_available = availability_row.get("emitted_pose_available")
        if type(emitted_available) is not bool:
            raise ValueError("Evaluation frame availability emitted-pose flag must be boolean")
        if emitted_available != (emitted_error is not None):
            raise ValueError(f"Evaluation error and availability disagree at frame {frame_id}/{condition_name}")
        trace_row = trace_by_frame.get(frame_id)
        validated_trace_row = candidate_by_frame.get(frame_id)
        complete_row = complete_rows.get(frame_id)
        output_available = availability_row.get("output_row_available")
        if type(output_available) is not bool or output_available != (complete_row is not None):
            raise ValueError(f"Evaluation output-row availability disagrees with sealed rows at frame {frame_id}")
        if availability_row.get("translation_error_m") != emitted_error:
            raise ValueError(f"Evaluation frame and emitted-error values disagree at frame {frame_id}")
        if trace_row is None and validated_trace_row is not None:
            raise ValueError("Validated candidate-trace identity is missing from the original trace map")
        raw_candidates = [] if trace_row is None else trace_row.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError(f"Candidate trace candidates must be a list at frame {frame_id}")
        valid_pose = [candidate for candidate in raw_candidates
                      if isinstance(candidate, Mapping) and candidate.get("pose") is not None]
        declared_eligible = [candidate for candidate in raw_candidates
                             if isinstance(candidate, Mapping) and candidate.get("eligible") is True]
        after_gates = [candidate for candidate in valid_pose
                       if candidate.get("eligible") is True and
                       isinstance(candidate.get("gates"), Mapping) and
                       all(value is True for value in candidate["gates"].values())]
        if bool(valid_pose) != (frame_id in before_minima):
            raise ValueError(f"Recorded before-gate minimum and valid candidates disagree at frame {frame_id}")
        if bool(after_gates) != (frame_id in after_minima):
            raise ValueError(f"Recorded after-gate minimum and eligible candidates disagree at frame {frame_id}")

        rank_decision = (trace_row.get("rank_decision")
                         if isinstance(trace_row, Mapping) else {})
        selected_id = rank_decision.get("selected_id") if isinstance(rank_decision, Mapping) else None
        legacy_id = trace_row.get("legacy_choice_candidate_id") if isinstance(trace_row, Mapping) else None
        selected = next((candidate for candidate in raw_candidates
                         if isinstance(candidate, Mapping) and candidate.get("candidate_id") == selected_id), None)
        emitted_pose = trace_row.get("cameraFromObject") if isinstance(trace_row, Mapping) else None
        selected_matches_emitted = bool(
            selected is not None and _pose_equal(selected.get("pose"), emitted_pose) and
            emitted_error is not None and availability_row.get("emitted_pose_available") is True)
        selected_error = emitted_error if selected_matches_emitted else None
        before_minimum = before_minima.get(frame_id)
        after_minimum = after_minima.get(frame_id)
        motion_statuses = [
            _backend_motion_status(candidate.get("validation_stats"))
            for candidate in raw_candidates if isinstance(candidate, Mapping)
        ]
        if "disabled" in motion_statuses:
            row_backend_motion_status = "disabled"
        elif "applied" in motion_statuses:
            row_backend_motion_status = "applied"
        else:
            row_backend_motion_status = "unknown"

        # The complete output and evaluator availability preserve unavailable,
        # suppressed, failure, and mask states alongside the unmodified trace.
        all_rows[frame_id] = {
            "condition": condition_name,
            "sequence_id": sequence_id,
            "split": sequence_report.get("split"),
            "frame_id": frame_id,
            "output_row_present": complete_row is not None,
            "evaluator_frame_availability": availability_row,
            "emitted_translation_error_m_from_evaluator": emitted_error,
            "counterfactual_candidate_minimum_error_m": {
                "before_gates": before_minimum,
                "after_gates": after_minimum,
                "counterfactual_within_this_head_candidate_set": True,
            },
            "selected_candidate_id": selected_id,
            "legacy_choice_candidate_id": legacy_id,
            "selected_candidate_matches_emitted_pose_exactly": selected_matches_emitted,
            "selected_candidate_error_m_from_evaluator": selected_error,
            "selected_minus_candidate_minimum_m": {
                "before_gates": (None if selected_error is None or before_minimum is None
                                  else selected_error - before_minimum),
                "after_gates": (None if selected_error is None or after_minimum is None
                                 else selected_error - after_minimum),
                "defined_only_when_emitted_pose_equals_selected_candidate": True,
            },
            "candidate_counts": {
                "total_attempts": len(raw_candidates),
                "valid_pose_count_before_gates": len(valid_pose),
                "declared_eligible_count": len(declared_eligible),
                "valid_eligible_count_after_all_recorded_gates": len(after_gates),
                "cardinality": ("none" if not raw_candidates else
                                "single" if len(raw_candidates) == 1 else "multiple"),
            },
            "backend_motion_status_from_recorded_metadata": row_backend_motion_status,
            "candidate_evidence": [
                _candidate_evidence(candidate, trace_row=trace_row, motion_policy=motion_policy)
                for candidate in raw_candidates if isinstance(candidate, Mapping)
            ],
            "tracking_settings_time_policy": motion_policy,
            "appearance_evidence_note": (
                "Observed-mask availability is not independent mask accuracy; appearance_summary_ms is timing only"
            ),
            "recorded_trace": trace_row,
        }
    return all_rows


def _condition_summary(
    condition_name: str,
    condition_report: Mapping[str, Any],
    sequence_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    def candidate_dist(summary: Mapping[str, Any], gate_stage: str) -> dict[str, Any]:
        coverage = summary.get("candidate_coverage")
        if not isinstance(coverage, Mapping):
            raise ValueError("Evaluation condition candidate coverage is malformed")
        return _distribution_summary(coverage.get(f"minimum_translation_error_{gate_stage}"),
                                     name=f"{condition_name}.{gate_stage} candidate minimum")

    emitted = _distribution_summary(condition_report.get("emitted_translation_errors"),
                                     name=f"{condition_name}.emitted")
    return {
        "condition": condition_name,
        "denominator": {
            "requested_frames": condition_report.get("requested_count"),
            "emitted_pose_frames": condition_report.get("emitted_pose_count"),
                    "emitted_pose_unavailable_frames": (
                        condition_report.get("requested_count", 0) -
                        condition_report.get("emitted_pose_count", 0)),
            "missing_output_rows": condition_report.get("missing_output_row_count"),
            "candidate_trace_rows": condition_report.get("candidate_trace_row_count"),
                    "missing_candidate_trace_rows": (
                        condition_report.get("requested_count", 0) -
                        condition_report.get("candidate_trace_row_count", 0)),
        },
        "emitted_translation_error_m": emitted,
        "candidate_minimum_translation_error_m": {
            "before_gates": candidate_dist(condition_report, "before_gates"),
            "after_gates": candidate_dist(condition_report, "after_gates"),
        },
        "split_aggregates_from_evaluator": condition_report.get("split_aggregates"),
        "sequence_distributions": [
            {
                "sequence_id": sequence_id,
                "split": sequence_reports[sequence_id].get("split"),
                "denominator": {
                    "requested_frames": sequence_reports[sequence_id].get("requested_count"),
                    "emitted_pose_frames": sequence_reports[sequence_id].get("emitted_pose_count"),
                    "emitted_pose_unavailable_frames": (
                        sequence_reports[sequence_id].get("requested_count", 0) -
                        sequence_reports[sequence_id].get("emitted_pose_count", 0)),
                    "missing_output_rows": sequence_reports[sequence_id].get("missing_output_row_count"),
                    "candidate_trace_rows": sequence_reports[sequence_id].get("candidate_trace_row_count"),
                    "missing_candidate_trace_rows": (
                        sequence_reports[sequence_id].get("requested_count", 0) -
                        sequence_reports[sequence_id].get("candidate_trace_row_count", 0)),
                },
                "emitted_translation_error_m": _distribution_summary(
                    sequence_reports[sequence_id].get("emitted_translation_errors"),
                    name=f"{condition_name}.{sequence_id}.emitted"),
                "candidate_minimum_translation_error_m": {
                    "before_gates": candidate_dist(sequence_reports[sequence_id], "before_gates"),
                    "after_gates": candidate_dist(sequence_reports[sequence_id], "after_gates"),
                },
            }
            for sequence_id in _SEQUENCES
        ],
        "statistics_source": "copied from authenticated post-seal evaluator report; no prediction errors recomputed",
    }


def build_diagnosis(
    *,
    authenticated_snapshot: AuthenticatedSnapshot,
    evaluation_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a pure, saved-output diagnosis from an authenticated seal and report."""
    if not isinstance(authenticated_snapshot, AuthenticatedSnapshot):
        raise ValueError("authenticated_snapshot must be an AuthenticatedSnapshot")
    if not isinstance(evaluation_report, Mapping):
        raise ValueError("evaluation_report must be an object")
    for name, digest in (
        ("sealed manifest", authenticated_snapshot.sealed_manifest_sha256),
        ("sealed archive", authenticated_snapshot.sealed_archive_sha256),
        ("evaluation policy", authenticated_snapshot.evaluation_policy_sha256),
        ("evaluation report", authenticated_snapshot.evaluation_report_sha256),
        ("policy manifest", authenticated_snapshot.policy.get("manifest_sha256")),
        ("policy archive", authenticated_snapshot.policy.get("archive_sha256")),
    ):
        if not evaluator._is_sha256(digest):
            raise ValueError(f"Authenticated {name} SHA-256 must be lowercase hex")
    receipt = _validate_evaluation_receipt(authenticated_snapshot, evaluation_report)
    if authenticated_snapshot.policy.get("policy_id") != evaluator.EXPANDED_POLICY_ID:
        raise ValueError("Saved-output diagnosis requires the expanded24 policy")
    if tuple(authenticated_snapshot.policy.get("requested_frame_ids", ())) != _REQUESTED:
        raise ValueError("Expanded evaluation policy must request exactly frames 0..23")
    if authenticated_snapshot.sealed_manifest_sha256 != authenticated_snapshot.policy.get("manifest_sha256"):
        raise ValueError("Authenticated snapshot manifest digest differs from its policy")
    if authenticated_snapshot.sealed_archive_sha256 != authenticated_snapshot.policy.get("archive_sha256"):
        raise ValueError("Authenticated snapshot archive digest differs from its policy")
    claim_scope = evaluation_report.get("claim_scope")
    if not isinstance(claim_scope, Mapping) or claim_scope.get("quality_promotion") is not False:
        raise ValueError("Evaluation report does not retain its non-promotion claim boundary")

    conditions = evaluation_report.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(_CONDITIONS):
        raise ValueError("Evaluation report must contain frozen and adapted conditions")
    sequence_reports = {
        condition: _index_sequences(conditions[condition], condition_name=condition)
        for condition in _CONDITIONS
    }

    frame_map: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    shared_motion_policy: dict[str, Any] | None = None
    for condition_name in _CONDITIONS:
        for sequence_id in _SEQUENCES:
            run_identity = (condition_name, sequence_id)
            if run_identity not in authenticated_snapshot.report_by_identity:
                raise ValueError(f"Authenticated seal lacks run identity {run_identity}")
            rows = _trace_frame_rows(
                authenticated_snapshot, evaluation_report,
                condition_name=condition_name,
                sequence_id=sequence_id,
                sequence_report=sequence_reports[condition_name][sequence_id],
            )
            frame_map[run_identity] = rows
            for row in rows.values():
                settings = row.get("tracking_settings_time_policy")
                if settings is None:
                    continue
                if shared_motion_policy is None:
                    shared_motion_policy = dict(settings)
                elif dict(settings) != shared_motion_policy:
                    raise ValueError("Full-video runs have inconsistent authenticated physical motion settings")

    frames = [
        frame_map[(condition_name, sequence_id)][frame_id]
        for condition_name in _CONDITIONS
        for sequence_id in _SEQUENCES
        for frame_id in _REQUESTED
    ]
    if len(frames) != 144:
        raise ValueError("Diagnosis must retain exactly 144 full condition/sequence/frame rows")

    windows = []
    for label, sequence_id, center in _FOCUS_WINDOWS:
        frame_ids = tuple(range(max(0, center - 2), min(23, center + 2) + 1))
        windows.append({
            "focus": label,
            "sequence_id": sequence_id,
            "split": evaluator.SEQUENCE_SPECS[sequence_id]["split"],
            "fixed_focus_frame_id": center,
            "fixed_window_radius_frames": 2,
            "selection_basis": "predeclared from the reviewed study; not recomputed as a maximum of this input report",
            "frame_ids": list(frame_ids),
            "condition_rows": {
                condition_name: [frame_map[(condition_name, sequence_id)][frame_id]
                                 for frame_id in frame_ids]
                for condition_name in _CONDITIONS
            },
            "comparison_note": "Frozen and adapted rows are retained together for the same fixed frame window.",
        })

    return {
        "schema": "expanded_gotrack_saved_output_failure_diagnosis_v1",
        "status": "model_free_saved_output_diagnosis",
        "source_hashes": {
            "sealed_manifest_sha256": authenticated_snapshot.sealed_manifest_sha256,
            "sealed_archive_sha256": authenticated_snapshot.sealed_archive_sha256,
            "evaluation_policy_sha256": authenticated_snapshot.evaluation_policy_sha256,
            "evaluation_report_sha256": authenticated_snapshot.evaluation_report_sha256,
            "evaluation_policy_id": authenticated_snapshot.policy.get("policy_id"),
        },
        "evaluation_input_receipt": {
            "authenticated_sealed_member_count_including_manifest": receipt[
                "authenticated_sealed_member_count_including_manifest"],
            "consumed_sealed_members_sha256": dict(receipt["consumed_sealed_members_sha256"]),
            "reference_input_hashes_from_evaluator_receipt": receipt["reference_inputs"],
            "reference_sidecars_reopened": False,
        },
        "scope": {
            "full_video_frame_rows": 144,
            "full_video_frames_per_condition_sequence": 24,
            "four_frame_prefix_runs_included": False,
            "prefix_run_role": "causality-only; excluded from full-video quality rows",
            "training_rows": "TRAIN object 17 views are resubstitution and are kept separate by sequence",
            "validation_rows": "object 27 is previously inspected used development validation",
            "independent_frame_confidence_intervals": False,
            "hypothesis_tests": False,
            "threshold_tuning_or_search": False,
            "prediction_errors_recomputed": False,
            "candidate_minima_are_counterfactual": True,
            "candidate_minima_scope": "recorded candidates from the same head; never an emitted replacement pose",
        },
        "motion_policy": shared_motion_policy,
        "condition_distributions": {
            condition_name: _condition_summary(
                condition_name, conditions[condition_name], sequence_reports[condition_name])
            for condition_name in _CONDITIONS
        },
        "full_frames": frames,
        "fixed_focus_selection": {
            "basis": "predeclared reviewed-study frame IDs; no search over the supplied report",
            "reference_evaluation_report_sha256": _FOCUS_REFERENCE_EVALUATION_SHA256,
            "reference_role": "reference-only provenance; not an assertion about the supplied report maxima",
            "current_report_maximum_recomputed": False,
        },
        "fixed_focus_context_windows": windows,
        "missing_internal_evidence": dict(_MISSING_INTERNAL_EVIDENCE),
        "interpretation_limits": [
            "A recorded gate admission describes the saved contract decision; it does not identify a network-level root cause.",
            "validation_stats.inliers are current-image residual-validation inliers, not RANSAC inliers.",
            "A true backend_motion gate with disabled metadata is not evidence of an independent motion check.",
            "appearance_summary_ms is timing only; absent appearance features remain missing evidence.",
            "Observed mask availability does not measure independent mask accuracy.",
            "Frames are dependent within synthetic sequences; no frame-independent uncertainty claim is made.",
        ],
    }


def _authenticated_snapshot_from_paths(
    *,
    sealed_root: Path,
    sealed_manifest_sha256: str,
    sealed_archive: Path,
    evaluation_policy: Path,
    evaluation_policy_sha256: str,
    evaluation_report: Path,
    evaluation_report_sha256: str,
) -> tuple[AuthenticatedSnapshot, dict[str, Any]]:
    policy = evaluator._load_expanded_policy(evaluation_policy, evaluation_policy_sha256)
    if sealed_manifest_sha256 != policy.manifest_sha256:
        raise ValueError("Supplied sealed-manifest SHA-256 differs from the authenticated policy")
    safe_sealed_root = evaluator._safe_root(sealed_root, "sealed artifact")
    manifest, members = evaluator.authenticate_member_manifest(
        safe_sealed_root,
        sealed_manifest_sha256,
        total_bytes_max=policy.sealed_total_bytes_max,
    )
    evaluator._verify_archive_matches_snapshot(sealed_archive, policy, members)
    evaluator._validate_expanded_inventory(manifest, members)

    design = evaluator._parse_json_bytes(members["design.json"], "design.json")
    bindings = evaluator._parse_json_bytes(members["execution-bindings.json"], "execution-bindings.json")
    paired_report = evaluator._parse_paired_report_bytes(
        members["paired-report.json"], policy_id=policy.policy_id)
    if evaluator._sha256(members["design.json"]) != policy.design_sha256:
        raise ValueError("Expanded design bytes do not match their authenticated policy pin")
    if evaluator._sha256(members["execution-bindings.json"]) != policy.bindings_sha256:
        raise ValueError("Expanded execution-binding bytes do not match their authenticated policy pin")
    binding_by_sequence, report_by_identity = evaluator._validate_sealed_metadata(
        design, bindings, paired_report, policy=policy)

    report_path = Path(evaluation_report).expanduser().absolute()
    raw_report = evaluator._read_bounded_file(
        report_path, max_bytes=evaluator.MAX_JSON_BYTES, name="post-seal evaluation report")
    if not evaluator._is_sha256(evaluation_report_sha256):
        raise ValueError("Expected evaluation-report SHA-256 must be lowercase hex")
    if evaluator._sha256(raw_report) != evaluation_report_sha256:
        raise ValueError("Evaluation-report SHA-256 mismatch")
    report_doc = evaluator._parse_json_bytes(raw_report, "post-seal evaluation report")
    if not isinstance(report_doc, Mapping):
        raise ValueError("Post-seal evaluation report must be a JSON object")

    snapshot = AuthenticatedSnapshot(
        policy={
            "policy_id": policy.policy_id,
            "manifest_sha256": policy.manifest_sha256,
            "archive_sha256": policy.archive_sha256,
            "diagnostic_sha256": policy.diagnostic_sha256,
            "checkpoint_sha256": policy.checkpoint_sha256,
            "output_prefix": policy.output_prefix,
            "requested_frame_ids": tuple(policy.requested_frame_ids),
        },
        sealed_manifest_sha256=sealed_manifest_sha256,
        sealed_archive_sha256=policy.archive_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
        evaluation_policy_sha256=evaluation_policy_sha256,
        members=members,
        binding_by_sequence=binding_by_sequence,
        report_by_identity=report_by_identity,
    )
    return snapshot, report_doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--sealed-manifest-sha256", required=True)
    parser.add_argument("--sealed-archive", type=Path, required=True)
    parser.add_argument("--evaluation-policy", type=Path, required=True)
    parser.add_argument("--evaluation-policy-sha256", required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--evaluation-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        snapshot, evaluation_report = _authenticated_snapshot_from_paths(
            sealed_root=args.sealed_root,
            sealed_manifest_sha256=args.sealed_manifest_sha256,
            sealed_archive=args.sealed_archive,
            evaluation_policy=args.evaluation_policy,
            evaluation_policy_sha256=args.evaluation_policy_sha256,
            evaluation_report=args.evaluation_report,
            evaluation_report_sha256=args.evaluation_report_sha256,
        )
        report = build_diagnosis(
            authenticated_snapshot=snapshot,
            evaluation_report=evaluation_report,
        )
        output_sha256 = evaluator.write_new_report(
            args.output,
            report,
            input_roots=(args.sealed_root.absolute(), args.sealed_archive.absolute(),
                         args.evaluation_policy.absolute(), args.evaluation_report.absolute()),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"expanded saved-output diagnosis refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "written", "output": str(args.output),
                      "report_sha256": output_sha256}, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
