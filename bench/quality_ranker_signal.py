"""Post-hoc candidate-signal diagnostics for a frozen pose-ranker cache.

This module is deliberately analytic.  It reads only the audited train and
validation shards, never loads a model, and never changes the cache or its
frozen label policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .quality_ranker_data import (
    _cache_manifest,
    _label_key,
    _safe_cache_file,
    audit_cache,
    load_cache_split,
    read_split_manifest,
    rotation_error_degrees,
    translation_error_m,
)


REPORT_SCHEMA_VERSION = 1
MAX_POSE_PAIRS = 100_000
MAX_REPORT_BYTES = 16 * 1024 * 1024
UNPROCESSED_MARKERS = frozenset({
    "not_processed",
    "not_processed_within_approved_compute_budget",
})
_SPLITS = ("train", "validation")
_SOURCE_FILES = ("quality_ranker_signal.py", "quality_ranker_data.py")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _counter_dict(values: Counter) -> dict[str, int]:
    return {str(key): int(values[key]) for key in sorted(values, key=str)}


def _state_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return "non_scalar"


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize observed pair distances without inventing zero evidence."""
    if not rows:
        return {
            "pair_count": 0,
            "evidence_status": "no_pairs_no_dispersion_evidence",
            "rotation_deg": None,
            "translation_m": None,
        }

    def values(name: str) -> dict[str, float]:
        data = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        return {
            "min": float(np.min(data)),
            "median": float(np.median(data)),
            "mean": float(np.mean(data)),
            "p95": float(np.percentile(data, 95)),
            "max": float(np.max(data)),
        }

    return {
        "pair_count": len(rows),
        "evidence_status": "observed_pairs",
        "rotation_deg": values("rotation_deg"),
        "translation_m": values("translation_m"),
    }


def _label_class(label: Mapping[str, Any] | None) -> str:
    if label is None:
        return "missing_label"
    target = label.get("quality_target")
    if target == 1:
        return "good"
    if target == 0:
        return "bad"
    return "target_unavailable"


def _identity(frame: Mapping[str, Any], split: str) -> dict[str, Any]:
    return {
        "split": split,
        "object_id": frame.get("object_id"),
        "physical_object_id": frame.get("physical_object_id"),
        "sequence_id": frame.get("sequence_id"),
        "frame_id": frame.get("frame_id"),
        "source_frame_id": frame.get("source_frame_id"),
        "timestamp_s": frame.get("timestamp_s"),
        "clock_mode": frame.get("clock_mode"),
    }


def _processing_status(frame: Mapping[str, Any]) -> str:
    reason = frame.get("failure_reason")
    candidates = frame.get("candidates") or []
    if reason in UNPROCESSED_MARKERS:
        return "unprocessed"
    if candidates:
        return "processed_with_candidates"
    context = frame.get("context") if isinstance(frame.get("context"), dict) else {}
    explicit = (
        reason is not None
        or frame.get("pose_state") is not None
        or frame.get("render_state") is not None
        or frame.get("mask_state") is not None
        or context.get("mask_state") is not None
    )
    return "processed_empty" if explicit else "unknown_status_empty"


def _candidate_descriptor(candidate: Mapping[str, Any], label: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "stage": candidate.get("stage"),
        "source": candidate.get("source"),
        "eligible": candidate.get("eligible") is True,
        "label_class": _label_class(label),
        "observationally_ambiguous": (
            None if label is None else label.get("observationally_ambiguous")
        ),
    }


def _candidate_label_row(candidate: Mapping[str, Any], label: Mapping[str, Any] | None) -> dict[str, Any]:
    row = _candidate_descriptor(candidate, label)
    row.update({
        "orientation_supervision": None if label is None else label.get("orientation_supervision"),
        "rotation_error_deg": None if label is None else label.get("rotation_error_deg"),
        "translation_error_m": None if label is None else label.get("translation_error_m"),
        "label_missing_reason": None if label is None else label.get("label_missing_reason"),
        "label_kind": None if label is None else label.get("label_kind"),
    })
    return row


def _frame_label_index(labels: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for label in labels:
        key = _label_key(label)
        if key in result:
            raise ValueError(f"Duplicate label key in diagnostic input: {key}")
        result[key] = label
    return result


def _frame_candidate_key(frame: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        frame.get("object_id"),
        frame.get("sequence_id"),
        frame.get("frame_id"),
        candidate.get("candidate_id"),
    )


def _composition(candidates: Iterable[Mapping[str, Any]], labels: Mapping[tuple[Any, ...], Mapping[str, Any]],
                 frame: Mapping[str, Any]) -> dict[str, int]:
    counts = Counter({name: 0 for name in ("good", "bad", "target_unavailable", "missing_label")})
    for candidate in candidates:
        counts[_label_class(labels.get(_frame_candidate_key(frame, candidate)))] += 1
    return {name: int(counts[name]) for name in ("good", "bad", "target_unavailable", "missing_label")}


def _has_mixed(composition: Mapping[str, int]) -> bool:
    return composition["good"] > 0 and composition["bad"] > 0


def diagnose_groups(
    frames: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    split: str,
    label_policy: Mapping[str, Any],
    max_pairs: int = MAX_POSE_PAIRS,
) -> dict[str, Any]:
    """Diagnose recorded candidate groups without changing labels or gate policy.

    Pairwise rotation is intentionally recomputed from recorded candidate poses,
    including observationally ambiguous positives whose reference-relative
    rotation label is unavailable.  No angular collapse threshold is applied.
    """
    if split not in _SPLITS:
        raise ValueError("Candidate-signal diagnosis may use only train or validation")
    if type(max_pairs) is not int or max_pairs < 0:
        raise ValueError("max_pairs must be a nonnegative integer")
    if not isinstance(label_policy, Mapping):
        raise ValueError("A frozen label policy is required")

    label_index = _frame_label_index(labels)
    composition_raw = Counter({name: 0 for name in ("good", "bad", "target_unavailable", "missing_label")})
    composition_eligible = Counter(composition_raw)
    processing = Counter()
    state_counts = {name: Counter() for name in ("mask_state", "pose_state", "render_state", "failure_reason")}
    stage_source = defaultdict(lambda: {"raw": 0, "eligible": 0, "valid_pose": 0})
    removed_by_reason: dict[str, Counter] = defaultdict(Counter)
    within_pairs: list[dict[str, Any]] = []
    cross_pairs: list[dict[str, Any]] = []
    ambiguous_positive_pairs: list[dict[str, Any]] = []
    frame_reports = []
    valid_pose_count = 0
    eligible_valid_pose_count = 0
    candidate_attempts = 0
    eligible_attempts = 0
    mixed_before = 0
    mixed_after = 0

    for frame in frames:
        if frame.get("split", split) != split:
            raise ValueError("Frame split differs from requested diagnostic split")
        candidates = frame.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Every diagnostic frame must contain a candidate list")
        identity = _identity(frame, split)
        status = _processing_status(frame)
        processing[status] += 1
        context = frame.get("context") if isinstance(frame.get("context"), dict) else {}
        states = {
            "mask_state": frame.get("mask_state", context.get("mask_state")),
            "pose_state": frame.get("pose_state", context.get("pose_state")),
            "render_state": frame.get("render_state", context.get("render_state")),
            "failure_reason": frame.get("failure_reason"),
        }
        for name, value in states.items():
            state_counts[name][_state_value(value)] += 1

        eligible = [row for row in candidates if row.get("eligible") is True]
        raw_composition = _composition(candidates, label_index, frame)
        eligible_composition = _composition(eligible, label_index, frame)
        composition_raw.update(raw_composition)
        composition_eligible.update(eligible_composition)
        mixed_before += _has_mixed(raw_composition)
        mixed_after += _has_mixed(eligible_composition)
        candidate_attempts += len(candidates)
        eligible_attempts += len(eligible)
        pose_candidates = [row for row in candidates if row.get("pose") is not None]
        valid_pose_count += len(pose_candidates)
        eligible_valid_pose_count += sum(row.get("eligible") is True for row in pose_candidates)

        removed = []
        surviving_classes = Counter()
        label_rows = []
        for candidate in candidates:
            label = label_index.get(_frame_candidate_key(frame, candidate))
            label_class = _label_class(label)
            label_rows.append(_candidate_label_row(candidate, label))
            stage_key = json.dumps(
                {"stage": candidate.get("stage"), "source": candidate.get("source")},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            stage_source[stage_key]["raw"] += 1
            stage_source[stage_key]["eligible"] += candidate.get("eligible") is True
            stage_source[stage_key]["valid_pose"] += candidate.get("pose") is not None
            if candidate.get("eligible") is True:
                surviving_classes[label_class] += 1
            else:
                reason = candidate.get("rejection")
                reason_key = "missing_rejection_reason" if reason is None else str(reason)
                removed_by_reason[reason_key][label_class] += 1
                removed.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "label_class": label_class,
                    "rejection_reason": reason,
                })

        for left_index, left in enumerate(pose_candidates):
            left_label = label_index.get(_frame_candidate_key(frame, left))
            for right in pose_candidates[left_index + 1:]:
                if len(within_pairs) + len(cross_pairs) >= max_pairs:
                    raise ValueError(f"Pose-pair count exceeds configured bound {max_pairs}")
                right_label = label_index.get(_frame_candidate_key(frame, right))
                pair = {
                    "frame_identity": identity,
                    "candidate_a": _candidate_descriptor(left, left_label),
                    "candidate_b": _candidate_descriptor(right, right_label),
                    "rotation_deg": rotation_error_degrees(left["pose"], right["pose"]),
                    "translation_m": translation_error_m(left["pose"], right["pose"]),
                }
                if left.get("stage") == right.get("stage"):
                    pair["relation"] = "within_stage"
                    within_pairs.append(pair)
                else:
                    pair["relation"] = "cross_stage_non_simultaneous"
                    cross_pairs.append(pair)
                if (
                    _label_class(left_label) == "good"
                    and _label_class(right_label) == "good"
                    and left_label is not None
                    and right_label is not None
                    and left_label.get("observationally_ambiguous") is True
                    and right_label.get("observationally_ambiguous") is True
                ):
                    ambiguous_positive_pairs.append(pair)

        frame_reports.append({
            "identity": identity,
            "processing_status": status,
            "states": states,
            "candidate_attempts": len(candidates),
            "valid_pose_candidates": len(pose_candidates),
            "eligible_candidate_attempts": len(eligible),
            "eligible_valid_pose_candidates": sum(
                row.get("eligible") is True for row in pose_candidates
            ),
            "composition_before_gates": raw_composition,
            "composition_after_gates": eligible_composition,
            "mixed_good_bad_before_gates": _has_mixed(raw_composition),
            "mixed_good_bad_after_gates": _has_mixed(eligible_composition),
            "removed_candidates": removed,
            "surviving_quality_classes": _counter_dict(surviving_classes),
            "candidate_labels": label_rows,
        })

    all_pairs = within_pairs + cross_pairs
    return {
        "split": split,
        "frozen_label_policy": dict(label_policy),
        "frame_count": len(frames),
        "processing_status_counts": _counter_dict(processing),
        "candidate_counts": {
            "attempts": candidate_attempts,
            "valid_pose": valid_pose_count,
            "eligible_attempts": eligible_attempts,
            "eligible_valid_pose": eligible_valid_pose_count,
        },
        "composition_before_gates": _counter_dict(composition_raw),
        "composition_after_gates": _counter_dict(composition_eligible),
        "mixed_good_bad_groups_before_gates": int(mixed_before),
        "mixed_good_bad_groups_after_gates": int(mixed_after),
        "removed_quality_classes_by_rejection": {
            reason: _counter_dict(classes)
            for reason, classes in sorted(removed_by_reason.items())
        },
        "stage_source_candidate_counts": [
            {**json.loads(key), **value}
            for key, value in sorted(stage_source.items())
        ],
        "state_counts": {name: _counter_dict(values) for name, values in state_counts.items()},
        "frames": frame_reports,
        "pose_pair_distances": {
            "overall_summary": _summary(all_pairs),
            "within_stage": {"summary": _summary(within_pairs), "pairs": within_pairs},
            "cross_stage": {
                "summary": _summary(cross_pairs),
                "pairs": cross_pairs,
                "interpretation": (
                    "Cross-stage candidates can represent failed tracking and recovery; "
                    "they are reported separately and are not assumed simultaneous."
                ),
            },
            "ambiguous_positive_summary": _summary(ambiguous_positive_pairs),
            "ambiguous_positive_pair_count": len(ambiguous_positive_pairs),
            "interpretation": (
                "Distances are recomputed between recorded post-refinement poses. "
                "No angular collapse threshold or label retuning is applied; zero pairs means no evidence."
            ),
        },
    }


def _manifest_location(manifest: Path) -> tuple[Path, Path]:
    supplied = Path(manifest)
    if supplied.is_dir():
        supplied = supplied / "cache_manifest.json"
    if not supplied.is_file():
        raise FileNotFoundError(f"Candidate cache manifest is missing: {supplied}")
    cache_dir = supplied.parent
    expected = cache_dir / "cache_manifest.json"
    if not expected.is_file() or not supplied.samefile(expected):
        raise ValueError("--manifest must identify CACHE/cache_manifest.json")
    return supplied, cache_dir


def diagnose_cache(manifest: Path, *, max_pairs: int = MAX_POSE_PAIRS) -> dict[str, Any]:
    """Audit and diagnose only train/validation rows from a frozen cache."""
    manifest_path, cache_dir = _manifest_location(Path(manifest))
    source_hashes_before = _source_hashes()
    cache_before, raw_before = _cache_manifest(cache_dir)
    if manifest_path.read_bytes() != raw_before:
        raise ValueError("Supplied manifest identity differs from the cache manifest")
    audit_before = audit_cache(cache_dir, training_only=True)
    if audit_before.get("approved") is not True or audit_before.get("audit_scope") != "train_validation_only":
        raise ValueError("Candidate cache failed the required training-only audit: " + "; ".join(audit_before.get("issues", [])))

    split_path = _safe_cache_file(cache_dir, cache_before["files"].get("split_manifest"))
    split_manifest = read_split_manifest(split_path)
    if split_manifest.get("manifest_sha256") != audit_before["identity"]["split_sha256"]:
        raise ValueError("Authenticated split metadata differs from the audited cache identity")
    label_policy = split_manifest["label_policy"]
    reports = {}
    pair_budget_used = 0
    for split in _SPLITS:
        frames, labels = load_cache_split(cache_dir, split)
        report = diagnose_groups(
            frames,
            labels,
            split=split,
            label_policy=label_policy,
            max_pairs=max_pairs - pair_budget_used,
        )
        pair_budget_used += report["pose_pair_distances"]["overall_summary"]["pair_count"]
        reports[split] = report

    audit_after = audit_cache(cache_dir, training_only=True)
    cache_after, raw_after = _cache_manifest(cache_dir)
    source_hashes_after = _source_hashes()
    if (
        audit_after.get("approved") is not True
        or audit_before["identity"] != audit_after.get("identity")
        or cache_before != cache_after
        or raw_before != raw_after
        or source_hashes_before != source_hashes_after
    ):
        raise ValueError("Candidate cache or diagnostic source identity changed during diagnosis")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "quality-ranker-candidate-signal-diagnostic-v1",
        "audit_scope": "train_validation_only",
        "test_read": False,
        "dev_read": False,
        "cache_identity": audit_before["identity"],
        "cache_provenance": {
            "cache_schema_version": cache_before["schema_version"],
            "feature_schema_version": cache_before["feature_schema_version"],
            "source_trace_sha256": cache_before["source_trace_sha256"],
            "reference_labels_sha256": cache_before["reference_labels_sha256"],
            "split_manifest_sha256": cache_before["split_manifest_sha256"],
            "cache_manifest_sha256": cache_before["cache_manifest_sha256"],
            "diagnostic_source_sha256": source_hashes_before,
        },
        "frozen_label_policy": dict(label_policy),
        "audit_counters": {
            "frames_checked": audit_before["frames_checked"],
            "candidates_checked": audit_before["candidates_checked"],
            "labels_checked": audit_before["labels_checked"],
        },
        "report_limits": {
            "maximum_pose_pairs": max_pairs,
            "maximum_output_bytes": MAX_REPORT_BYTES,
            "overflow_policy": "raise_without_truncation_or_output",
            "reported_candidate_payload": "identities_labels_errors_and_derived_distances_only",
            "pose_matrices_feature_vectors_and_model_weights_included": False,
        },
        "splits": reports,
        "raw_gate_vectors": {
            "available": False,
            "reason": "raw gate vectors are not stored in the frozen candidate cache and were not reconstructed",
        },
        "causal_limits": [
            "Candidate seeds are not recorded, so the report cannot distinguish FoundPose seed collapse from GoTrack refinement collapse.",
            "Before-gate candidates in this cache are already post-refinement recorded poses.",
            "Pair distances diagnose recorded pose diversity only; they do not establish attachment accuracy or generalization.",
            "No ground-truth-generated negative poses, gate relaxation, label retuning, or model execution was used.",
        ],
    }


def _output_bytes(report: Mapping[str, Any]) -> bytes:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError(
            f"Diagnostic report is {len(payload)} bytes and exceeds the {MAX_REPORT_BYTES}-byte bound"
        )
    return payload


def _validate_output_target(path: Path, *, forbidden_root: Path | None = None) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace diagnostic report: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Diagnostic output parent does not exist: {path.parent}")
    if any(parent.is_symlink() for parent in (path.parent, *path.parent.parents)):
        raise ValueError("Refusing to write through a symlinked output directory")
    if forbidden_root is not None:
        resolved_output = path.resolve(strict=False)
        resolved_root = Path(forbidden_root).resolve(strict=True)
        if resolved_output == resolved_root or resolved_output.is_relative_to(resolved_root):
            raise ValueError("Refusing to write a diagnostic report inside the immutable cache")
    return path


def _write_output(path: Path, payload: bytes, *, forbidden_root: Path | None = None) -> None:
    path = _validate_output_target(path, forbidden_root=forbidden_root)
    with path.open("xb") as stream:
        stream.write(payload)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Frozen CACHE/cache_manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh JSON report path; existing paths are refused")
    args = parser.parse_args(argv)
    # Refuse an occupied or symlinked destination before reading the cache.
    _validate_output_target(args.output)
    manifest_path, cache_dir = _manifest_location(args.manifest)
    _validate_output_target(args.output, forbidden_root=cache_dir)
    report = diagnose_cache(manifest_path)
    payload = _output_bytes(report)
    _write_output(args.output, payload, forbidden_root=cache_dir)
    print(json.dumps({
        "status": "diagnosed",
        "output": str(args.output),
        "bytes": len(payload),
        "train_frames": report["splits"]["train"]["frame_count"],
        "validation_frames": report["splits"]["validation"]["frame_count"],
        "test_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
