"""Inference-neutral attribution of FoundPose seeds to final GoTrack returns.

The adapter in this module observes the existing backend boundary.  It never
calls a model, changes candidate order, or supplies a pose to inference.  Raw
pose snapshots stay in memory and are discarded when the small derived report
is finalized.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np

from .quality_ranker_data import (
    rotation_error_degrees,
    translation_error_m,
)
from .quality_time import FrameKey


REPORT_SCHEMA_VERSION = 1
_SOURCE_FILES = (
    "quality_seed_attribution.py",
    "quality_runner.py",
    "quality_ranker_data.py",
)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _finite_pose(value, name: str) -> np.ndarray:
    """Return a private, validated float64 snapshot of one metric pose."""
    try:
        shape = getattr(value, "shape", None)
        if shape is not None:
            if tuple(shape) != (4, 4):
                raise ValueError(f"{name} must be a finite homogeneous 4x4 pose")
        elif isinstance(value, (list, tuple)):
            if (len(value) != 4 or any(not isinstance(row, (list, tuple)) or
                                       len(row) != 4 for row in value)):
                raise ValueError(f"{name} must be a finite homogeneous 4x4 pose")
        else:
            raise ValueError(f"{name} must be a finite homogeneous 4x4 pose")
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite homogeneous 4x4 pose") from None
    try:
        pose = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite homogeneous 4x4 pose") from exc
    # Reuse the frozen diagnostic helpers for the complete pose validation.
    rotation_error_degrees(pose, pose)
    translation_error_m(pose, pose)
    return pose


def _json_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("distance summary contains a non-finite value")
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _distance_summary(rows: list[tuple[np.ndarray, np.ndarray]]) -> dict | None:
    if not rows:
        return None
    rotations = [rotation_error_degrees(left, right) for left, right in rows]
    translations = [translation_error_m(left, right) for left, right in rows]
    return {
        "sample_count": len(rows),
        "rotation_deg": _stats(rotations),
        "translation_m": _stats(translations),
    }


def _publish_no_clobber(temporary: Path, output: Path):
    """Atomically publish a complete file without replacing an existing one."""
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise
    except OSError:
        if os.name != "nt":
            raise
        # Windows rename is atomic and fails when the destination exists. This
        # also supports managed Windows sandboxes that disallow hard links.
        os.rename(temporary, output)


def _safe_scalar(value, name: str):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{name} must be a string or nonnegative integer")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if isinstance(value, str) and (not value or len(value) > 512):
        raise ValueError(f"{name} must be nonempty and at most 512 characters")
    return value


class SeedAttributionObserver:
    """Bounded in-memory observer with an exclusive derived-report finalizer."""

    def __init__(self, *, object_id: str, sequence_id: str,
                 source_frame_ids: Mapping[int, int | str],
                 max_events: int = 256, max_failures: int = 32,
                 max_trace_bytes: int = 64 * 1024**2,
                 max_report_bytes: int = 1024**2,
                 max_pairs: int = 10_000,
                 max_trace_rows: int = 4096):
        self.object_id = _safe_scalar(object_id, "object_id")
        self.sequence_id = _safe_scalar(sequence_id, "sequence_id")
        if not isinstance(source_frame_ids, Mapping):
            raise TypeError("source_frame_ids must be a mapping")
        detached = {}
        for frame_id, source_id in source_frame_ids.items():
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                raise ValueError("source_frame_ids keys must be nonnegative integers")
            detached[frame_id] = _safe_scalar(source_id, "source_frame_id")
        if not detached:
            raise ValueError("source_frame_ids must not be empty")
        for name, value in (
                ("max_events", max_events), ("max_failures", max_failures),
                ("max_trace_bytes", max_trace_bytes),
                ("max_report_bytes", max_report_bytes),
                ("max_pairs", max_pairs), ("max_trace_rows", max_trace_rows)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.source_frame_ids = detached
        self.max_events = max_events
        self.max_failures = max_failures
        self.max_trace_bytes = max_trace_bytes
        self.max_report_bytes = max_report_bytes
        self.max_pairs = max_pairs
        self.max_trace_rows = max_trace_rows

        self._events: list[dict] = []
        self._failures: list[dict] = []
        self._failure_count = 0
        self._dropped_events = 0
        self._counts = Counter()
        self._frame_occurrences = Counter()
        self._active_frame_base = None
        self._active_frame = None
        self._recover_ordinals = Counter()
        self._refine_ordinals = Counter()
        self._stage_indices = Counter()
        self._active_recovery = None
        self._finalized = False

    def _failure(self, kind: str, *, detail: str | None = None,
                 identity: Mapping | None = None):
        self._failure_count += 1
        if len(self._failures) < self.max_failures:
            row = {"kind": str(kind)[:128]}
            if detail is not None:
                row["detail"] = str(detail)[:512]
            if identity is not None:
                row["identity"] = dict(identity)
            self._failures.append(row)

    def _adapter_failure(self, method_name: str, exc: BaseException):
        self._failure("observer_callback_failure",
                      detail=f"{method_name}:{type(exc).__name__}")

    def _frame_identity(self, descriptor) -> dict:
        if not isinstance(descriptor, Mapping):
            raise ValueError("seed attribution requires detached frame metadata")
        try:
            key = FrameKey(descriptor["frame_id"], descriptor.get("timestamp_s"),
                           descriptor["clock_mode"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("seed attribution frame metadata is invalid") from exc
        if key.frame_id not in self.source_frame_ids:
            raise ValueError("source frame identity is missing")
        base = (
            self.object_id, self.sequence_id,
            self.source_frame_ids[key.frame_id], key.frame_id,
            key.timestamp_s, key.clock_mode,
        )
        if base != self._active_frame_base:
            occurrence = self._frame_occurrences[base]
            self._frame_occurrences[base] += 1
            self._active_frame_base = base
            self._active_frame = (base, occurrence)
            self._active_recovery = None
        base, occurrence = self._active_frame
        return {
            "object_id": base[0],
            "sequence_id": base[1],
            "source_frame_id": base[2],
            "frame_key": {
                "frame_id": base[3],
                "timestamp_s": base[4],
                "clock_mode": base[5],
            },
            "frame_occurrence": occurrence,
        }

    @staticmethod
    def _occurrence_key(identity: Mapping) -> tuple:
        key = identity["frame_key"]
        return (
            identity["object_id"], identity["sequence_id"],
            identity["source_frame_id"], key["frame_id"],
            key["timestamp_s"], key["clock_mode"],
            identity["frame_occurrence"],
        )

    def _reserve(self, event: dict) -> dict | None:
        if len(self._events) >= self.max_events:
            self._dropped_events += 1
            if self._dropped_events == 1:
                self._failure("event_cap_exhausted",
                              detail=f"max_events={self.max_events}")
            return None
        self._events.append(event)
        return event

    # The following four methods are called only through SeedAttributionBackend's
    # RNG-isolated boundary. They are private to keep inference code from using
    # observer state as input.
    def _recover_started(self, frame_descriptor, top_k, snapshot_error=None):
        self._counts["recover_calls"] += 1
        if snapshot_error is not None:
            self._failure("recover_frame_snapshot_failed", detail=snapshot_error)
        try:
            identity = self._frame_identity(frame_descriptor)
        except Exception as exc:
            self._failure("recover_identity_snapshot_failed",
                          detail=type(exc).__name__)
            return None
        occurrence = self._occurrence_key(identity)
        ordinal = self._recover_ordinals[occurrence]
        self._recover_ordinals[occurrence] += 1
        event = self._reserve({
            "kind": "recover",
            "identity": identity,
            "recover_ordinal": ordinal,
            "top_k": top_k,
            "seeds": [],
            "outcome": "pending",
        })
        self._active_recovery = {
            "occurrence": occurrence,
            "recover_ordinal": ordinal,
            "next_index": 0,
            "generated": 0,
        }
        return event

    def _recover_finished(self, token, seeds, outcome, exception_type=None,
                          snapshot_errors=(), actual_count=None):
        if outcome == "exception":
            self._counts["recover_exceptions"] += 1
            if token is not None:
                token["outcome"] = "exception"
                token["exception_type"] = exception_type
            self._failure("recover_exception", detail=exception_type,
                          identity=None if token is None else token["identity"])
            self._active_recovery = None
            return
        if outcome != "returned" or seeds is None:
            self._failure("recover_result_not_sequence",
                          detail="detached recovery result unavailable",
                          identity=None if token is None else token["identity"])
            if token is not None:
                token["outcome"] = "invalid_result"
            self._active_recovery = None
            return
        generated_count = len(seeds) if actual_count is None else actual_count
        self._counts["generated_seeds"] += generated_count
        if self._active_recovery is not None:
            self._active_recovery["generated"] = generated_count
        if token is None:
            return
        token["outcome"] = "returned"
        error_by_index = {index: detail for index, detail in snapshot_errors
                          if index >= 0}
        for index, detail in snapshot_errors:
            if index < 0:
                self._failure("recovery_seed_snapshot_overflow", detail=detail,
                              identity=token["identity"])
        for original_index, seed in enumerate(seeds):
            snapshot = None
            if original_index in error_by_index:
                self._failure("recovery_seed_snapshot_failed",
                              detail=f"index={original_index}:{error_by_index[original_index]}",
                              identity=token["identity"])
            else:
                try:
                    snapshot = _finite_pose(seed, "recovery seed")
                except Exception as exc:
                    self._failure("recovery_seed_snapshot_failed",
                                  detail=f"index={original_index}:{type(exc).__name__}",
                                  identity=token["identity"])
            token["seeds"].append({
                "original_index": original_index,
                "pose": snapshot,
                "consumed": False,
                "refine_ordinal": None,
            })

    def _refine_started(self, frame_descriptor, seed, snapshot_error=None):
        self._counts["consumed_refine_attempts"] += 1
        if snapshot_error is not None:
            self._failure("refine_entry_snapshot_failed", detail=snapshot_error)
        try:
            identity = self._frame_identity(frame_descriptor)
        except Exception as exc:
            self._failure("refine_identity_snapshot_failed",
                          detail=type(exc).__name__)
            return None
        occurrence = self._occurrence_key(identity)
        refine_ordinal = self._refine_ordinals[occurrence]
        self._refine_ordinals[occurrence] += 1
        recovery = self._active_recovery
        if recovery is not None and recovery["occurrence"] == occurrence:
            stage = "recovery"
            original_index = recovery["next_index"]
            recovery["next_index"] += 1
            recover_ordinal = recovery["recover_ordinal"]
            self._counts["consumed_recovery_seeds"] += 1
            generated = recovery["generated"]
            if original_index >= generated:
                self._failure("refine_without_generated_seed",
                              detail=f"index={original_index},generated={generated}",
                              identity=identity)
        else:
            stage = "refinement"
            index_key = (occurrence, stage)
            original_index = self._stage_indices[index_key]
            self._stage_indices[index_key] += 1
            recover_ordinal = None
            self._counts["consumed_refinement_seeds"] += 1
        try:
            seed_pose = _finite_pose(seed, "refine-entry seed")
        except Exception as exc:
            seed_pose = None
            self._failure("refine_seed_snapshot_failed",
                          detail=type(exc).__name__, identity=identity)
        event = self._reserve({
            "kind": "refine",
            "identity": identity,
            "recover_ordinal": recover_ordinal,
            "refine_ordinal": refine_ordinal,
            "stage": stage,
            "original_index": original_index,
            "seed_pose": seed_pose,
            "returned_pose": None,
            "outcome": "pending",
        })
        if event is not None and stage == "recovery":
            for recover_event in reversed(self._events[:-1]):
                if (recover_event.get("kind") == "recover" and
                        recover_event.get("recover_ordinal") == recover_ordinal and
                        self._occurrence_key(recover_event["identity"]) == occurrence):
                    seeds = recover_event["seeds"]
                    if original_index < len(seeds):
                        seeds[original_index]["consumed"] = True
                        seeds[original_index]["refine_ordinal"] = refine_ordinal
                        generated_pose = seeds[original_index]["pose"]
                        if generated_pose is not None and seed_pose is not None and not np.allclose(
                                generated_pose, seed_pose, rtol=0., atol=1e-12):
                            self._failure("generated_consumed_seed_mismatch",
                                          detail=f"index={original_index}",
                                          identity=identity)
                    break
        return event

    def _refine_finished(self, token, returned_pose, outcome,
                         exception_type=None, snapshot_error=None):
        if outcome == "exception":
            self._counts["refine_exceptions"] += 1
            if token is not None:
                token["outcome"] = "exception"
                token["exception_type"] = exception_type
            self._failure("refine_exception", detail=exception_type,
                          identity=None if token is None else token["identity"])
            return
        if outcome == "none":
            self._counts["none_returns"] += 1
            if token is not None:
                token["outcome"] = "none"
            return
        self._counts["returned_candidates"] += 1
        if token is None:
            return
        token["outcome"] = "returned"
        if snapshot_error is not None:
            self._counts["invalid_returned_pose_snapshots"] += 1
            self._failure("returned_pose_snapshot_failed",
                          detail=snapshot_error, identity=token["identity"])
            return
        try:
            token["returned_pose"] = _finite_pose(
                returned_pose, "returned candidate pose")
        except Exception as exc:
            self._counts["invalid_returned_pose_snapshots"] += 1
            self._failure("returned_pose_snapshot_failed",
                          detail=type(exc).__name__, identity=token["identity"])

    @staticmethod
    def _trace_base(row: Mapping) -> tuple:
        return (
            row.get("object_id"), row.get("sequence_id"),
            row.get("source_frame_id"), row.get("frame_id"),
            row.get("timestamp_s"), row.get("clock_mode"),
        )

    def _read_trace(self, path: Path):
        path = Path(path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValueError("candidate trace is missing") from exc
        if size < 0 or size > self.max_trace_bytes:
            raise ValueError("candidate trace exceeds its byte cap")
        occurrence_counts = Counter()
        candidates = {}
        trace_frames = 0
        zero_call_frames = 0
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if line_number > self.max_trace_rows:
                    raise ValueError("candidate trace exceeds its row cap")
                if len(raw) > self.max_report_bytes:
                    raise ValueError("candidate trace row exceeds its line cap")
                if not raw.strip():
                    raise ValueError("candidate trace contains a blank row")
                try:
                    row = json.loads(
                        raw.decode("utf-8"), object_pairs_hook=_json_unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"nonfinite JSON number: {value}")))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("candidate trace row is invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError("candidate trace row is not an object")
                base = self._trace_base(row)
                if (base[0] != self.object_id or base[1] != self.sequence_id or
                        base[3] is None or base[5] is None):
                    raise ValueError("candidate trace identity differs from the observer")
                try:
                    trace_key = FrameKey(base[3], base[4], base[5])
                except (TypeError, ValueError) as exc:
                    raise ValueError("candidate trace FrameKey is invalid") from exc
                if (trace_key.frame_id != base[3] or
                        trace_key.timestamp_s != base[4] or
                        trace_key.clock_mode != base[5]):
                    raise ValueError("candidate trace FrameKey changed during validation")
                expected_source = self.source_frame_ids.get(base[3])
                if expected_source is None or base[2] != expected_source:
                    raise ValueError("candidate trace source-frame identity differs")
                occurrence = occurrence_counts[base]
                occurrence_counts[base] += 1
                trace_frames += 1
                rows = row.get("candidates")
                if not isinstance(rows, list):
                    raise ValueError("candidate trace candidates must be a list")
                if not rows:
                    zero_call_frames += 1
                for candidate in rows:
                    if not isinstance(candidate, dict):
                        raise ValueError("candidate trace entry is not an object")
                    stage = candidate.get("stage")
                    index = candidate.get("original_index")
                    if stage not in ("refinement", "recovery") or isinstance(index, bool) or not isinstance(index, int) or index < 0:
                        raise ValueError("candidate trace stage/index is invalid")
                    key = (*base, occurrence, stage, index)
                    if key in candidates:
                        raise ValueError("candidate trace contains a duplicate full candidate key")
                    pose_value = candidate.get("pose")
                    pose = None if pose_value is None else _finite_pose(
                        pose_value, "candidate trace pose")
                    candidates[key] = pose
        if trace_frames == 0:
            raise ValueError("candidate trace has no explicit frame rows")
        return candidates, trace_frames, zero_call_frames

    @staticmethod
    def _attempt_key(event: Mapping) -> tuple:
        identity = event["identity"]
        key = identity["frame_key"]
        return (
            identity["object_id"], identity["sequence_id"],
            identity["source_frame_id"], key["frame_id"],
            key["timestamp_s"], key["clock_mode"],
            identity["frame_occurrence"], event["stage"],
            event["original_index"],
        )

    def _build_report(self, candidate_trace: Path) -> dict:
        trace_candidates = {}
        trace_frames = 0
        zero_call_frames = 0
        try:
            trace_candidates, trace_frames, zero_call_frames = self._read_trace(candidate_trace)
        except Exception as exc:
            self._failure("candidate_trace_validation_failed",
                          detail=type(exc).__name__ + ":" + str(exc))

        attempts = [event for event in self._events if event.get("kind") == "refine"]
        matched = []
        unmatched_attempts = 0
        used_trace_keys = set()
        for event in attempts:
            key = self._attempt_key(event)
            if key not in trace_candidates:
                unmatched_attempts += 1
                self._failure("missing_candidate_trace_match",
                              detail=f"stage={event['stage']},index={event['original_index']}",
                              identity=event["identity"])
                continue
            used_trace_keys.add(key)
            trace_pose = trace_candidates[key]
            returned_pose = event.get("returned_pose")
            outcome = event.get("outcome")
            if outcome == "none":
                consistent = trace_pose is None
            elif outcome == "returned" and returned_pose is not None:
                consistent = trace_pose is not None and np.allclose(
                    returned_pose, trace_pose, rtol=0., atol=1e-12)
            else:
                consistent = False
            if not consistent:
                unmatched_attempts += 1
                self._failure("candidate_trace_return_mismatch",
                              detail=f"stage={event['stage']},index={event['original_index']}",
                              identity=event["identity"])
                continue
            event["trace_matched"] = True
            matched.append(event)
        unmatched_trace = len(set(trace_candidates) - used_trace_keys)
        if unmatched_trace:
            self._failure("unmatched_candidate_trace_entries",
                          detail=f"count={unmatched_trace}")

        seed_return_by_stage = {}
        for stage in ("refinement", "recovery"):
            pairs = [
                (event["seed_pose"], event["returned_pose"])
                for event in matched
                if event["stage"] == stage and event.get("seed_pose") is not None
                and event.get("returned_pose") is not None
            ]
            seed_return_by_stage[stage] = _distance_summary(pairs)
        all_seed_return = [
            (event["seed_pose"], event["returned_pose"])
            for event in matched
            if event.get("seed_pose") is not None and event.get("returned_pose") is not None
        ]

        surviving = defaultdict(list)
        for event in matched:
            if (event["stage"] == "recovery" and event.get("seed_pose") is not None
                    and event.get("returned_pose") is not None):
                surviving[self._occurrence_key(event["identity"])].append(event)
        seed_dispersion = []
        returned_dispersion = []
        pair_count = 0
        pair_cap_hit = False
        for group in surviving.values():
            group.sort(key=lambda event: (event["recover_ordinal"],
                                          event["original_index"],
                                          event["refine_ordinal"]))
            for left_index in range(len(group)):
                for right_index in range(left_index + 1, len(group)):
                    if pair_count >= self.max_pairs:
                        pair_cap_hit = True
                        break
                    left, right = group[left_index], group[right_index]
                    seed_dispersion.append((left["seed_pose"], right["seed_pose"]))
                    returned_dispersion.append((left["returned_pose"], right["returned_pose"]))
                    pair_count += 1
                if pair_cap_hit:
                    break
            if pair_cap_hit:
                break
        if pair_cap_hit:
            self._failure("pair_cap_exhausted", detail=f"max_pairs={self.max_pairs}")
        paired = None
        if pair_count:
            paired = {
                "pair_count": pair_count,
                "seed_dispersion": _distance_summary(seed_dispersion),
                "returned_dispersion": _distance_summary(returned_dispersion),
                "pairing_rule": "same_full_frame_occurrence_and_same_surviving_recovery_attempt_pair",
            }

        generated = [seed for event in self._events if event.get("kind") == "recover"
                     for seed in event.get("seeds", [])]
        unconsumed_generated = max(
            0, int(self._counts["generated_seeds"]) -
            int(self._counts["consumed_recovery_seeds"]))
        if (unconsumed_generated and not any(
                row.get("kind") == "unconsumed_generated_seeds"
                for row in self._failures)):
            self._failure("unconsumed_generated_seeds",
                          detail=f"count={unconsumed_generated}")
        failed_attempts = []
        for event in attempts:
            invalid_return = (event.get("outcome") == "returned" and
                              event.get("returned_pose") is None)
            if event.get("outcome") not in ("none", "exception") and not invalid_return:
                continue
            if len(failed_attempts) >= self.max_failures:
                break
            failed_attempts.append({
                "identity": event["identity"],
                "recover_ordinal": event.get("recover_ordinal"),
                "refine_ordinal": event.get("refine_ordinal"),
                "stage": event.get("stage"),
                "original_index": event.get("original_index"),
                "outcome": "invalid_return" if invalid_return else event.get("outcome"),
                "exception_type": event.get("exception_type"),
            })

        counts = {
            "recover_calls": int(self._counts["recover_calls"]),
            "generated_seeds": int(self._counts["generated_seeds"]),
            "generated_seed_snapshots": len(generated),
            "consumed_refine_attempts": int(self._counts["consumed_refine_attempts"]),
            "consumed_refinement_seeds": int(self._counts["consumed_refinement_seeds"]),
            "consumed_recovery_seeds": int(self._counts["consumed_recovery_seeds"]),
            "returned_candidates": int(self._counts["returned_candidates"]),
            "none_returns": int(self._counts["none_returns"]),
            "refine_exceptions": int(self._counts["refine_exceptions"]),
            "recover_exceptions": int(self._counts["recover_exceptions"]),
            "invalid_returned_pose_snapshots": int(
                self._counts["invalid_returned_pose_snapshots"]),
            "matched_attempts": len(matched),
            "unmatched_attempts": unmatched_attempts,
            "unmatched_trace_candidates": unmatched_trace,
            "unconsumed_generated_seeds": unconsumed_generated,
            "trace_frames": trace_frames,
            "zero_call_trace_frames": zero_call_frames,
            "recorded_events": len(self._events),
            "dropped_events": self._dropped_events,
        }
        complete = (
            self._failure_count == 0 and self._dropped_events == 0
            and unmatched_attempts == 0 and unmatched_trace == 0
            and len(attempts) == self._counts["consumed_refine_attempts"]
            and len(matched) == len(attempts)
        )
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "diagnostic": "foundpose_seed_to_gotrack_return_attribution_v1",
            "status": "complete" if complete else "incomplete",
            "complete": bool(complete),
            "identity": {
                "object_id": self.object_id,
                "sequence_id": self.sequence_id,
            },
            "scope": (
                "Derived inference diagnostics only; no labels, reference poses, "
                "model-quality claim, or tracking decision changes."
            ),
            "caps": {
                "max_events": self.max_events,
                "max_failures": self.max_failures,
                "max_trace_bytes": self.max_trace_bytes,
                "max_trace_rows": self.max_trace_rows,
                "max_report_bytes": self.max_report_bytes,
                "max_pairs": self.max_pairs,
            },
            "counts": counts,
            "seed_to_return": {
                "all": _distance_summary(all_seed_return),
                **seed_return_by_stage,
            },
            "paired_recovery_dispersion": paired,
            "failed_attempts": failed_attempts,
            "failures": {
                "total": self._failure_count,
                "reported": list(self._failures),
                "truncated": self._failure_count > len(self._failures),
            },
            "source_hashes": _source_hashes(),
        }

    def report(self, candidate_trace: Path) -> dict:
        """Build a derived report without writing or clearing raw snapshots."""
        if self._finalized:
            raise RuntimeError("seed attribution observer is already finalized")
        return self._build_report(Path(candidate_trace))

    def finalize(self, candidate_trace: Path, output: Path) -> dict:
        """Write one exclusive report and always discard raw pose snapshots."""
        if self._finalized:
            raise RuntimeError("seed attribution observer is already finalized")
        output = Path(output)
        temporary = None
        report = None
        try:
            report = self._build_report(Path(candidate_trace))
            if output.exists():
                self._failure("report_output_already_exists")
                return self._build_report(Path(candidate_trace))
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            if len(payload) > self.max_report_bytes:
                self._failure("report_byte_cap_exhausted",
                              detail=f"bytes={len(payload)}")
                report = self._build_report(Path(candidate_trace))
                payload = json.dumps(
                    report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
                if len(payload) > self.max_report_bytes:
                    return report
            temporary = output.with_name(
                f".{output.name}.{uuid.uuid4().hex}.part")
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
            try:
                # A hard link publishes the fully written same-filesystem temp
                # atomically and fails if a concurrent finalizer won the name.
                _publish_no_clobber(temporary, output)
            except FileExistsError:
                self._failure("report_output_collision")
                return self._build_report(Path(candidate_trace))
            return report
        except Exception as exc:
            self._failure("report_finalize_failed",
                          detail=type(exc).__name__ + ":" + str(exc))
            try:
                return self._build_report(Path(candidate_trace))
            except Exception:
                return {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "diagnostic": "foundpose_seed_to_gotrack_return_attribution_v1",
                    "status": "incomplete",
                    "complete": False,
                    "identity": {"object_id": self.object_id,
                                 "sequence_id": self.sequence_id},
                    "failures": {"total": self._failure_count,
                                 "reported": list(self._failures),
                                 "truncated": self._failure_count > len(self._failures)},
                }
        finally:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    self._failure("temporary_report_cleanup_failed")
            for event in self._events:
                if event.get("kind") == "refine":
                    event["seed_pose"] = None
                    event["returned_pose"] = None
                elif event.get("kind") == "recover":
                    for seed in event.get("seeds", []):
                        seed["pose"] = None
            self._events.clear()
            self._active_recovery = None
            self._finalized = True


class SeedAttributionBackend:
    """Delegate each backend call exactly once while observing its boundary."""

    def __init__(self, backend, observer: SeedAttributionObserver):
        if backend is None:
            raise TypeError("seed attribution backend requires a delegate")
        if observer is None:
            raise TypeError("seed attribution backend requires an observer")
        self.backend = backend
        self.observer = observer

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @staticmethod
    def _rng_snapshot():
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch = sys.modules.get("torch")
        torch_state = None
        cuda_states = None
        errors = []
        if torch is not None:
            try:
                torch_state = torch.get_rng_state()
                if torch.cuda.is_initialized():
                    cuda_states = torch.cuda.get_rng_state_all()
            except Exception as exc:
                torch_state = None
                cuda_states = None
                errors.append("torch_rng_snapshot:" + type(exc).__name__)
        return (python_state, numpy_state, torch, torch_state, cuda_states), errors

    @staticmethod
    def _rng_restore(snapshot):
        python_state, numpy_state, torch, torch_state, cuda_states = snapshot
        errors = []
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            try:
                torch.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception as exc:
                errors.append("torch_rng_restore:" + type(exc).__name__)
        return errors

    def _mark_adapter_failure(self, method_name: str, detail):
        try:
            if isinstance(self.observer, SeedAttributionObserver):
                SeedAttributionObserver._adapter_failure(
                    self.observer, method_name,
                    detail if isinstance(detail, BaseException)
                    else RuntimeError(str(detail)))
            else:
                self.observer._adapter_failure(
                    method_name,
                    detail if isinstance(detail, BaseException)
                    else RuntimeError(str(detail)))
        except BaseException:
            pass

    def _observe(self, method_name: str, *args):
        snapshot, snapshot_errors = self._rng_snapshot()
        for detail in snapshot_errors:
            self._mark_adapter_failure(method_name, detail)
        try:
            return getattr(self.observer, method_name)(*args)
        except BaseException as exc:
            self._mark_adapter_failure(method_name, exc)
            return None
        finally:
            for detail in self._rng_restore(snapshot):
                self._mark_adapter_failure(method_name, detail)

    @staticmethod
    def _frame_descriptor(frame):
        try:
            key = frame.key
            if not isinstance(key, FrameKey):
                raise TypeError("frame.key is not a FrameKey")
            return {
                "frame_id": key.frame_id,
                "timestamp_s": key.timestamp_s,
                "clock_mode": key.clock_mode,
            }, None
        except Exception as exc:
            return None, "frame:" + type(exc).__name__

    @staticmethod
    def _pose_descriptor(value, name):
        try:
            shape = getattr(value, "shape", None)
            if shape is not None:
                if tuple(shape) != (4, 4):
                    return None, name + ":unexpected_shape"
            elif isinstance(value, (list, tuple)):
                if (len(value) != 4 or any(not isinstance(row, (list, tuple)) or
                                           len(row) != 4 for row in value)):
                    return None, name + ":unexpected_shape"
            else:
                return None, name + ":shape_unavailable"
        except Exception:
            return None, name + ":invalid_shape"
        try:
            return np.array(value, dtype=np.float64, copy=True, order="C"), None
        except Exception as exc:
            return None, name + ":" + type(exc).__name__

    def recover(self, frame, mask, top_k=5):
        frame_descriptor, frame_error = self._frame_descriptor(frame)
        token = self._observe("_recover_started", frame_descriptor, top_k,
                              frame_error)
        try:
            result = self.backend.recover(frame, mask, top_k=top_k)
        except BaseException as exc:
            self._observe("_recover_finished", token, None, "exception",
                          type(exc).__name__, (), None)
            raise
        detached = None
        snapshot_errors = []
        actual_count = None
        try:
            if isinstance(result, Sequence) and not isinstance(
                    result, (str, bytes, bytearray)):
                actual_count = len(result)
                detached_rows = []
                observer_cap = getattr(self.observer, "max_events", 256)
                if isinstance(observer_cap, bool) or not isinstance(observer_cap, int) or observer_cap <= 0:
                    observer_cap = 256
                top_k_cap = top_k if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0 else observer_cap
                snapshot_cap = min(observer_cap, top_k_cap)
                for index in range(min(actual_count, snapshot_cap)):
                    copied, error = self._pose_descriptor(result[index], "recovery_seed")
                    detached_rows.append(copied)
                    if error is not None:
                        snapshot_errors.append((index, error))
                if actual_count > snapshot_cap:
                    snapshot_errors.append((-1,
                        f"returned={actual_count},snapshot_cap={snapshot_cap}"))
                detached = tuple(detached_rows)
                outcome = "returned"
            else:
                outcome = "invalid_result"
        except BaseException as exc:
            detached = None
            outcome = "invalid_result"
            snapshot_errors = [(-1, "recovery_detach:" + type(exc).__name__)]
        self._observe("_recover_finished", token, detached, outcome, None,
                      tuple(snapshot_errors), actual_count)
        return result

    def refine(self, frame, mask, seed):
        frame_descriptor, frame_error = self._frame_descriptor(frame)
        seed_descriptor, seed_error = self._pose_descriptor(seed, "refine_seed")
        snapshot_error = ";".join(
            value for value in (frame_error, seed_error) if value is not None) or None
        token = self._observe("_refine_started", frame_descriptor,
                              seed_descriptor, snapshot_error)
        try:
            result = self.backend.refine(frame, mask, seed)
        except BaseException as exc:
            self._observe("_refine_finished", token, None, "exception",
                          type(exc).__name__, None)
            raise
        if result is None:
            self._observe("_refine_finished", token, None, "none", None, None)
        else:
            returned, error = self._pose_descriptor(
                getattr(result, "pose", None), "returned_pose")
            self._observe("_refine_finished", token, returned, "returned",
                          None, error)
        return result
