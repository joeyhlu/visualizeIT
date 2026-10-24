"""Evaluate sealed frozen/adapted poses against authenticated synthetic references.

This is a bounded, post-seal evaluator. It does not run inference or apply new
quality thresholds. Its primary measure is translation agreement in metres;
the supplied geometries are explicitly observationally ambiguous, so primary
orientation correctness is reported as unmeasured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import stat
from io import BytesIO
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .quality_ranker_data import (
    _validate_pose,
    translation_error_m,
)


PINNED_SEALED_MANIFEST_SHA256 = "c0564566222712bc8e322d0eef9fa051b31ac1d2ea8e8505fc6e847a68dcfe69"
PINNED_SEALED_ARCHIVE_SHA256 = "f44c19f4dbf9847b687922ac2f13078935103aa72ca349dacab7e17ee39a0e26"
PINNED_DIAGNOSTIC_SHA256 = "504bb0a2cf638ed5274ed65c6a652616e34ad51f94c8da11abae69044b5de5b1"
ADAPTED_HEAD_SHA256 = "942b35b52536fe303a9701296b27e90009b88a89216b0883cf045cf221ed7b41"
EXPANDED_HEAD_SHA256 = "e23794bcf91f2b3de48d30c0a1fed9c088c6e0119d8d64b5c1b2718adc4e3b96"
POSTSEAL_POLICY_SCHEMA = "gotrack_postseal_evaluation_policy_v1"
EXPANDED_POLICY_ID = "expanded24"
EXPANDED_DESIGN_SCHEMA = "expanded_gotrack_full_video_development_design_v1"
EXPANDED_BINDINGS_SCHEMA = "expanded_full_video_execution_bindings_v1"
EXPANDED_REPORT_SCHEMA = "expanded_full_video_development_report_v1"
MAX_EXPANDED_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_SEALED_TOTAL_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_ARCHIVE_FILES = 256
MAX_EXPANDED_PAIRED_REPORT_BYTES = 3 * 1024 * 1024

REQUESTED_FRAME_IDS = tuple(range(8))
REFERENCE_FRAME_IDS = tuple(range(24))
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_SEALED_TOTAL_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSONL_ROWS = 24
MAX_REFERENCE_TOTAL_BYTES = 1024 * 1024
MAX_ROOT_ENTRIES = 128
_HEX = frozenset("0123456789abcdef")
_GENERATION_KIND = "quality-ranker-synthetic-generation-v1"
_EVALUATOR_KIND = "quality-postseal-pose-evaluation-v1"
_INFERENCE_PROFILE = "gotrack_integer_legacy_synthetic_v1"
_EXPECTED_LABEL_KIND = "synthetic_renderer_pose"
_EXPECTED_OUTPUT_POSE_UNITS = "native_camera_from_object_metres;PnP_crop_camera_from_object_millimetres"
_PROFILE_ONLY_FIELDS = frozenset({
    "timings_ms", "timing_ms", "resource_memory", "elapsed_s",
    "appearance_summary_ms", "trace_serialization_ms",
})
_EXPECTED_REFERENCE_FIELDS = frozenset({
    "frame_id", "label_kind", "object_id", "observationally_ambiguous",
    "pose", "sequence_id", "source_frame_id", "timestamp_s",
})

SEQUENCE_SPECS: dict[str, dict[str, Any]] = {
    "ranker-multiview-hot3d17-v1-yaw0-pitchm25": {
        "physical_object_id": "hot3d:obj_000017", "design_split": "train",
        "split": "train", "binding_object_id": "17",
    },
    "ranker-multiview-hot3d17-v1-yaw0-pitchp25": {
        "physical_object_id": "hot3d:obj_000017", "design_split": "train",
        "split": "train", "binding_object_id": "17",
    },
    "ranker-multiview-hot3d27-v1-yaw0-pitchm25": {
        "physical_object_id": "hot3d:obj_000027", "design_split": "val",
        "split": "validation", "binding_object_id": "27",
    },
}
EXPECTED_RUNS = frozenset(
    (condition, sequence_id)
    for condition in ("frozen", "adapted")
    for sequence_id in SEQUENCE_SPECS
)


@dataclass(frozen=True, slots=True)
class _EvaluationPolicy:
    policy_id: str
    design_sha256: str
    bindings_sha256: str
    manifest_sha256: str
    archive_sha256: str
    diagnostic_sha256: str
    checkpoint_sha256: str
    requested_frame_ids: tuple[int, ...]
    archive_bytes_max: int
    design_schema: str
    bindings_schema: str
    report_schema: str
    output_prefix: str
    sealed_total_bytes_max: int


_LEGACY_POLICY = _EvaluationPolicy(
    policy_id="legacy8", design_sha256="", bindings_sha256="",
    manifest_sha256=PINNED_SEALED_MANIFEST_SHA256,
    archive_sha256=PINNED_SEALED_ARCHIVE_SHA256,
    diagnostic_sha256=PINNED_DIAGNOSTIC_SHA256,
    checkpoint_sha256=ADAPTED_HEAD_SHA256,
    requested_frame_ids=REQUESTED_FRAME_IDS,
    archive_bytes_max=0,
    design_schema="adapted_gotrack_paired_development_design_v1",
    bindings_schema="adapted_paired_execution_bindings_v1",
    report_schema="",
    output_prefix="eight",
    sealed_total_bytes_max=MAX_SEALED_TOTAL_BYTES,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(char in _HEX for char in value))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number is not permitted: {value}")


def _check_finite_tree(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ValueError("JSON nesting exceeds the bounded depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON number is not permitted")
    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            _check_finite_tree(child, depth=depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_finite_tree(child, depth=depth + 1)


def _parse_json_bytes(raw: bytes, name: str, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    if len(raw) > max_bytes:
        raise ValueError(f"JSON input exceeds the {max_bytes}-byte bound: {name}")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON input: {name}") from exc
    _check_finite_tree(value)
    return value


def _load_expanded_policy(path: str | Path, expected_sha256: str) -> _EvaluationPolicy:
    if not _is_sha256(expected_sha256):
        raise ValueError("Expected evaluation-policy SHA-256 must be lowercase hex")
    policy_path = Path(path).expanduser().absolute()
    for component in reversed((policy_path, *policy_path.parents)):
        if component.is_symlink():
            raise ValueError(f"Evaluation-policy path cannot contain symlinks: {component}")
    if not policy_path.is_file():
        raise ValueError("Evaluation policy must be an existing regular file")
    raw = _read_bounded_file(policy_path, max_bytes=MAX_JSON_BYTES,
                             name="expanded24 evaluation policy")
    if _sha256(raw) != expected_sha256:
        raise ValueError("Evaluation-policy SHA-256 mismatch")
    value = _parse_json_bytes(raw, "expanded24 evaluation policy")
    required = {
        "schema", "policy_id", "design_sha256", "execution_bindings_sha256",
        "sealed_manifest_sha256", "sealed_archive_sha256", "diagnostic_sha256",
        "checkpoint_sha256", "requested_frame_ids", "archive_bytes_max",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Expanded evaluation policy has unexpected or missing fields")
    if value["schema"] != POSTSEAL_POLICY_SCHEMA or value["policy_id"] != EXPANDED_POLICY_ID:
        raise ValueError("Unsupported post-seal evaluation policy")
    for key in ("design_sha256", "execution_bindings_sha256", "sealed_manifest_sha256",
                "sealed_archive_sha256", "diagnostic_sha256", "checkpoint_sha256"):
        if not _is_sha256(value[key]):
            raise ValueError(f"Expanded evaluation policy {key} must be lowercase SHA-256")
    if value["checkpoint_sha256"] != EXPANDED_HEAD_SHA256:
        raise ValueError("Expanded evaluation policy selects an unknown checkpoint")
    if value["diagnostic_sha256"] != PINNED_DIAGNOSTIC_SHA256:
        raise ValueError("Expanded evaluation policy selects an unknown label diagnostic")
    if value["requested_frame_ids"] != list(REFERENCE_FRAME_IDS):
        raise ValueError("Expanded evaluation policy must request exactly frames 0..23")
    if value["archive_bytes_max"] != MAX_EXPANDED_ARCHIVE_BYTES:
        raise ValueError("Expanded evaluation policy archive cap differs from the approved 32 MiB bound")
    return _EvaluationPolicy(
        policy_id=EXPANDED_POLICY_ID,
        design_sha256=value["design_sha256"],
        bindings_sha256=value["execution_bindings_sha256"],
        manifest_sha256=value["sealed_manifest_sha256"],
        archive_sha256=value["sealed_archive_sha256"],
        diagnostic_sha256=value["diagnostic_sha256"],
        checkpoint_sha256=value["checkpoint_sha256"],
        requested_frame_ids=tuple(value["requested_frame_ids"]),
        archive_bytes_max=value["archive_bytes_max"],
        design_schema=EXPANDED_DESIGN_SCHEMA,
        bindings_schema=EXPANDED_BINDINGS_SCHEMA,
        report_schema=EXPANDED_REPORT_SCHEMA,
        output_prefix="full",
        sealed_total_bytes_max=MAX_EXPANDED_SEALED_TOTAL_BYTES,
    )


def _verify_archive_matches_snapshot(path: str | Path, policy: _EvaluationPolicy,
                                     snapshot: Mapping[str, bytes]) -> None:
    archive_path = Path(path).expanduser().absolute()
    for component in reversed((archive_path, *archive_path.parents)):
        if component.is_symlink():
            raise ValueError(f"Sealed archive path cannot contain symlinks: {component}")
    try:
        mode = archive_path.stat().st_mode
        size = archive_path.stat().st_size
    except OSError as exc:
        raise ValueError("Sealed archive must be an existing regular file") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("Sealed archive must be an existing regular file")
    if size > policy.archive_bytes_max:
        raise ValueError("Sealed archive exceeds the policy byte bound")
    raw = archive_path.read_bytes()
    if len(raw) != size or _sha256(raw) != policy.archive_sha256:
        raise ValueError("Sealed archive changed or does not match its authenticated SHA-256")
    try:
        with zipfile.ZipFile(BytesIO(raw), "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_EXPANDED_ARCHIVE_FILES:
                raise ValueError("Sealed archive has an invalid member count")
            if sum(member.file_size for member in members) > policy.sealed_total_bytes_max:
                raise ValueError("Sealed archive exceeds the uncompressed byte bound")
            inventory: dict[str, bytes] = {}
            for member in members:
                safe = _safe_relative_name(member.filename)
                name = safe.as_posix()
                mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if (name != member.filename or member.is_dir() or
                        (file_type and file_type != stat.S_IFREG) or member.flag_bits & 0x1):
                    raise ValueError(f"Unsupported path or file type in sealed archive: {member.filename!r}")
                if name in inventory:
                    raise ValueError(f"Duplicate sealed archive member: {name}")
                inventory[name] = archive.read(member)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("Sealed archive is not a readable plain ZIP") from exc
    if inventory != dict(snapshot):
        raise ValueError("Sealed archive member inventory or bytes differ from the authenticated sealed root")


def _parse_jsonl_bytes(raw: bytes, name: str, *, max_rows: int = MAX_JSONL_ROWS) -> list[dict[str, Any]]:
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"JSONL input exceeds the bounded file size: {name}")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL input: {name}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL row in {name} at line {line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_unique_pairs,
                             parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid JSONL in {name} at line {line_number}") from exc
        _check_finite_tree(row)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} in {name} must be an object")
        rows.append(row)
        if len(rows) > max_rows:
            raise ValueError(f"JSONL row count exceeds the configured bound: {name}")
    return rows


def _safe_relative_name(value: Any) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ValueError("Input member path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Input member path escapes its root: {value!r}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"Input member path contains a drive prefix: {value!r}")
    return path


def _safe_root(path: str | Path, label: str) -> Path:
    root = Path(path).expanduser()
    absolute = root.absolute()
    for component in reversed((absolute, *absolute.parents)):
        if component.is_symlink():
            raise ValueError(f"{label} path cannot contain symlinks: {component}")
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} root does not resolve to an existing path") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} root must be a directory")
    for index, entry in enumerate(resolved.rglob("*"), start=1):
        if index > MAX_ROOT_ENTRIES:
            raise ValueError(f"{label} tree exceeds the {MAX_ROOT_ENTRIES}-entry bound")
        if entry.is_symlink():
            raise ValueError(f"Symlinks are not permitted in {label}: {entry}")
        try:
            entry.resolve(strict=True).relative_to(resolved)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{label} contains a path that escapes its root: {entry}") from exc
    return resolved


def _read_bounded_file(path: Path, *, max_bytes: int, name: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot stat bounded input file: {name}") from exc
    if size > max_bytes:
        raise ValueError(f"Input file exceeds the {max_bytes}-byte bound: {name}")
    raw = path.read_bytes()
    if len(raw) != size or len(raw) > max_bytes:
        raise ValueError(f"Input file changed or exceeded its byte bound while reading: {name}")
    return raw


def _safe_member(root: Path, relative: str) -> Path:
    relative_path = _safe_relative_name(relative)
    candidate = root.joinpath(*relative_path.parts)
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink is not permitted in input member: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Input member is missing or escapes its root: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"Input member must be a regular file: {relative}")
    return resolved


def _expected_sealed_members() -> frozenset[str]:
    members = {"design.json", "execution-bindings.json", "paired-report.json"}
    for condition, sequence_id in EXPECTED_RUNS:
        prefix = f"eight-{condition}-{sequence_id}"
        members.update({
            f"{prefix}/complete.json", f"{prefix}/candidates.jsonl",
            f"{prefix}/pose.log", f"{prefix}/pose.receipt.json",
        })
    prefix_sequence = "ranker-multiview-hot3d17-v1-yaw0-pitchm25"
    for condition in ("frozen", "adapted"):
        prefix = f"four-{condition}-{prefix_sequence}"
        members.update({
            f"{prefix}/complete.json", f"{prefix}/candidates.jsonl",
            f"{prefix}/pose.log", f"{prefix}/pose.receipt.json",
        })
    for sequence_id in (
        "ranker-multiview-hot3d17-v1-yaw0-pitchm25",
        "ranker-multiview-hot3d27-v1-yaw0-pitchm25",
    ):
        prefix = f"prerequisite-{sequence_id}"
        members.update({
            f"{prefix}/smoke.json", f"{prefix}/smoke.log",
            f"{prefix}/smoke.receipt.json",
        })
    return frozenset(members)


EXPECTED_SEALED_MEMBERS = _expected_sealed_members()


def _expanded_required_members() -> frozenset[str]:
    members = {"design.json", "execution-bindings.json", "paired-report.json"}
    prefix_sequence = "ranker-multiview-hot3d17-v1-yaw0-pitchm25"
    for condition, sequence_id in EXPECTED_RUNS:
        prefix = f"full-{condition}-{sequence_id}"
        members.update({
            f"{prefix}/complete.json", f"{prefix}/candidates.jsonl",
            f"{prefix}/pose.log", f"{prefix}/pose.receipt.json",
        })
    for condition in ("frozen", "adapted"):
        prefix = f"four-{condition}-TRAIN17minus25"
        members.update({
            f"{prefix}/complete.json", f"{prefix}/candidates.jsonl",
            f"{prefix}/pose.log", f"{prefix}/pose.receipt.json",
        })
    return frozenset(members)


def _validate_expanded_inventory(manifest: Mapping[str, Any], snapshot: Mapping[str, bytes]) -> None:
    files = manifest.get("files")
    required = _expanded_required_members()
    if not isinstance(files, dict) or not required.issubset(files):
        missing = sorted(required - set(files if isinstance(files, dict) else ()))
        raise ValueError(f"Expanded seal is missing required run or master members: {missing}")
    allowed_extras = {
        "gate-adapted.json", "gate-adapted.log", "gate-adapted.receipt.json",
        "reused-baseline-gate.json", "reused-baseline-gate-provenance.json",
        "mask-preparation/protocol.json",
        "mask-preparation/prefix-inventory.json",
        "mask-preparation/operator-completion.json",
        "mask-preparation/reuse-audit.json",
    }
    for sequence_id in SEQUENCE_SPECS:
        prefix = f"mask-preparation/{sequence_id}"
        allowed_extras.update({
            f"{prefix}/results.json", f"{prefix}/segment.log",
            f"{prefix}/segment.receipt.json", f"{prefix}/reuse.receipt.json",
        })
    extras = set(files) - set(required)
    for name in extras:
        if name not in allowed_extras:
            raise ValueError(f"Expanded seal contains an unapproved non-run member: {name}")
        raw = snapshot[name]
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Expanded diagnostic member must be UTF-8 text: {name}") from exc
        if name.endswith(".json"):
            _parse_json_bytes(raw, name, max_bytes=MAX_JSON_BYTES)


def authenticate_member_manifest(
    root: Path,
    expected_manifest_sha256: str,
    *,
    expected_members: frozenset[str] | set[str] | None = None,
    total_bytes_max: int = MAX_SEALED_TOTAL_BYTES,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Authenticate a manifest and snapshot every declared member's bytes.

    `expected_manifest_sha256` is supplied by the caller. Production passes the
    compiled seal pin; the optional member set makes the verifier independently
    testable with small temporary fixtures.
    """
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("Expected artifact-manifest SHA-256 must be lowercase hex")
    manifest_path = _safe_member(root, "artifact-manifest.json")
    manifest_raw = _read_bounded_file(manifest_path, max_bytes=MAX_JSON_BYTES,
                                      name="artifact-manifest.json")
    if _sha256(manifest_raw) != expected_manifest_sha256:
        raise ValueError("Sealed artifact-manifest.json hash does not match its pin")
    manifest = _parse_json_bytes(manifest_raw, "artifact-manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"files"}:
        raise ValueError("Artifact manifest must contain exactly a files mapping")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Artifact manifest files must be a non-empty object")
    if expected_members is not None and set(files) != set(expected_members):
        raise ValueError("Artifact manifest has unexpected or missing member identities")

    declared = set()
    total_bytes = len(manifest_raw)
    snapshot: dict[str, bytes] = {"artifact-manifest.json": manifest_raw}
    for relative, expected_digest in files.items():
        safe_relative = _safe_relative_name(relative)
        normalized = safe_relative.as_posix()
        if normalized != relative:
            raise ValueError(f"Non-canonical artifact member name: {relative!r}")
        if normalized in declared or not _is_sha256(expected_digest):
            raise ValueError(f"Invalid artifact member digest or duplicate path: {relative!r}")
        declared.add(normalized)
        member_path = _safe_member(root, normalized)
        member_size = member_path.stat().st_size
        if total_bytes + member_size > total_bytes_max:
            raise ValueError("Sealed artifact tree exceeds its total byte bound")
        raw = _read_bounded_file(member_path,
                                 max_bytes=total_bytes_max - total_bytes,
                                 name=normalized)
        if _sha256(raw) != expected_digest:
            raise ValueError(f"Sealed member digest mismatch: {normalized}")
        total_bytes += len(raw)
        if total_bytes > total_bytes_max:
            raise ValueError("Sealed artifact tree exceeds its total byte bound")
        snapshot[normalized] = raw

    actual_files = set()
    for entry in root.rglob("*"):
        if entry.is_file():
            actual_files.add(entry.relative_to(root).as_posix())
    expected_actual = declared | {"artifact-manifest.json"}
    if actual_files != expected_actual:
        raise ValueError("Sealed artifact tree has unexpected or missing files")
    return manifest, snapshot


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_sealed_metadata(
    design: Any,
    bindings: Any,
    paired_report: Any,
    *,
    policy: _EvaluationPolicy = _LEGACY_POLICY,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if not isinstance(design, dict) or design.get("schema") != policy.design_schema:
        raise ValueError("Unsupported paired design schema")
    if design.get("profile") != _INFERENCE_PROFILE or design.get("test_object_7_read") is not False:
        raise ValueError("Design profile or untouched test-object declaration is invalid")
    if policy.policy_id == EXPANDED_POLICY_ID:
        _expect(design.get("annotations_during_inference") is False,
                "Expanded design must prove inference ran without annotations")
        _expect(design.get("normal_geometry_changed") is False,
                "Expanded design must preserve the authenticated geometry")
        _expect(design.get("optimizer_steps") == 0,
                "Expanded full-video replay must not train or update the checkpoint")
    if design.get("checkpoint_sha256") != policy.checkpoint_sha256:
        raise ValueError("Design adapted-head identity differs from the pinned head")
    design_runs = design.get("runs")
    if not isinstance(design_runs, list) or len(design_runs) != 6:
        raise ValueError("Design must declare exactly six paired full-video quality runs")
    if policy.policy_id == EXPANDED_POLICY_ID:
        timestamps = design.get("frame_timestamps_seconds")
        expected_times = [frame_id / 10 for frame_id in policy.requested_frame_ids]
        _expect(design.get("requested_frames_per_sequence") == 24,
                "Expanded design must request all 24 frames")
        _expect(timestamps == expected_times,
                "Expanded design timestamps must match physical 10 Hz frame timing")
    design_identities = set()
    for run in design_runs:
        if not isinstance(run, dict):
            raise ValueError("Design run entry must be an object")
        sequence_id, condition = run.get("sequence_id"), run.get("condition")
        spec = SEQUENCE_SPECS.get(sequence_id)
        if spec is None or condition not in ("frozen", "adapted"):
            raise ValueError("Design contains an unexpected sequence or condition")
        identity = (condition, sequence_id)
        if identity in design_identities:
            raise ValueError("Design contains duplicate run identities")
        design_identities.add(identity)
        _expect(run.get("split") == spec["design_split"], "Design split identity mismatch")
        _expect(run.get("setup_frame_ids") == [], "Quality runs must not include setup/prefix frames")
        _expect(run.get("scored_frame_ids") == list(policy.requested_frame_ids),
                "Design quality frame IDs differ from the authenticated policy")
    if design_identities != EXPECTED_RUNS:
        raise ValueError("Design must declare the exact six frozen/adapted run identities")

    if not isinstance(bindings, dict) or bindings.get("schema") != policy.bindings_schema:
        raise ValueError("Unsupported execution-bindings schema")
    cases = bindings.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("Execution bindings must contain the exact three sequence cases")
    binding_by_sequence: dict[str, dict[str, Any]] = {}
    runtime_contract_sha256 = None
    source_closure_sha256 = None
    if policy.policy_id == EXPANDED_POLICY_ID:
        config_binding = bindings.get("config")
        if not isinstance(config_binding, dict) or not _is_sha256(config_binding.get("sha256")):
            raise ValueError("Expanded bindings lack the authenticated runtime config SHA-256")
        runtime_contract_sha256 = config_binding["sha256"]
        source_closure_sha256 = bindings.get("source_closure_sha256")
        if not _is_sha256(source_closure_sha256):
            raise ValueError("Expanded bindings lack the authenticated runtime source-closure SHA-256")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Execution binding case must be an object")
        sequence_id = case.get("sequence_id")
        spec = SEQUENCE_SPECS.get(sequence_id)
        if spec is None or sequence_id in binding_by_sequence:
            raise ValueError("Execution bindings contain an unexpected or duplicate sequence")
        _expect(case.get("object_id") == spec["binding_object_id"],
                "Execution binding object identity mismatch")
        _expect(case.get("split") == spec["design_split"], "Execution binding split mismatch")
        generation = case.get("generation")
        input_binding = case.get("input")
        if not isinstance(generation, dict) or not _is_sha256(generation.get("sha256")):
            raise ValueError("Execution binding generation SHA-256 is malformed")
        if not isinstance(input_binding, dict) or not _is_sha256(input_binding.get("sha256")):
            raise ValueError("Execution binding input SHA-256 is malformed")
        binding_by_sequence[sequence_id] = {
            "generation_sha256": generation["sha256"],
            "input_sha256": input_binding["sha256"],
            **({"runtime_contract_sha256": runtime_contract_sha256,
                "source_closure_sha256": source_closure_sha256}
               if policy.policy_id == EXPANDED_POLICY_ID else {}),
        }
    if set(binding_by_sequence) != set(SEQUENCE_SPECS):
        raise ValueError("Execution binding sequence set is incomplete")

    if not isinstance(paired_report, dict) or not isinstance(paired_report.get("runs"), list):
        raise ValueError("Paired execution report is malformed")
    if policy.report_schema and paired_report.get("schema") != policy.report_schema:
        raise ValueError("Expanded execution report schema mismatch")
    if paired_report.get("reference_pose_files_read") is not False:
        raise ValueError("Sealed execution report does not prove reference isolation")
    if (policy.policy_id == EXPANDED_POLICY_ID and
            paired_report.get("annotations_during_inference") is not False):
        raise ValueError("Expanded execution report must prove inference/labels were isolated")
    runs = paired_report["runs"]
    if len(runs) != 6:
        raise ValueError("Paired execution report must contain exactly six full-video runs")
    report_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Paired execution run must be an object")
        condition = run.get("condition")
        sequence_id = run.get("sequence_id")
        identity = (condition, sequence_id)
        if identity not in EXPECTED_RUNS or identity in report_by_identity:
            raise ValueError("Paired execution report contains an unexpected or duplicate run")
        _expect(run.get("status") == "complete" and run.get("failure_reason") is None,
                "Every sealed full-video run must be complete")
        _expect(run.get("requested_frame_ids") == list(policy.requested_frame_ids),
                "Sealed execution requested a different frame set")
        expected_prefix = f"{policy.output_prefix}-{condition}-{sequence_id}"
        output_path = run.get("output_path")
        trace_path = run.get("trace_path")
        if not isinstance(output_path, str) or not output_path.replace("\\", "/").endswith(
                f"/{expected_prefix}/complete.json"):
            raise ValueError("Sealed execution output path does not match its run identity")
        if not isinstance(trace_path, str) or not trace_path.replace("\\", "/").endswith(
                f"/{expected_prefix}/candidates.jsonl"):
            raise ValueError("Sealed execution trace path does not match its run identity")
        report_by_identity[identity] = run
    if set(report_by_identity) != EXPECTED_RUNS:
        raise ValueError("Paired execution report is missing a frozen/adapted run")
    if policy.policy_id == EXPANDED_POLICY_ID:
        prefixes = paired_report.get("causality_prefix_runs")
        if not isinstance(prefixes, list) or len(prefixes) != 2:
            raise ValueError("Expanded report must contain exactly two causality-prefix runs")
        prefix_sequence = "ranker-multiview-hot3d17-v1-yaw0-pitchm25"
        prefix_seen = set()
        prefix_ids = list(range(4))
        for run in prefixes:
            if not isinstance(run, dict):
                raise ValueError("Causality-prefix run entry must be an object")
            condition = run.get("condition")
            identity = (condition, run.get("sequence_id"))
            if (condition not in ("frozen", "adapted") or
                    identity[1] != prefix_sequence or identity in prefix_seen):
                raise ValueError("Expanded report has an unknown or duplicate causality-prefix run")
            prefix_seen.add(identity)
            _expect(run.get("status") == "complete" and run.get("failure_reason") is None,
                    "Every causality-prefix run must complete")
            _expect(run.get("requested_frame_ids") == prefix_ids,
                    "Causality-prefix runs must request exactly frames 0..3")
            expected_prefix = f"four-{condition}-TRAIN17minus25"
            for path_key, filename in (("output_path", "complete.json"),
                                       ("trace_path", "candidates.jsonl")):
                value = run.get(path_key)
                if (not isinstance(value, str) or not value.replace("\\", "/").endswith(
                        f"/{expected_prefix}/{filename}")):
                    raise ValueError("Causality-prefix output path does not match its run identity")
        if prefix_seen != {(condition, prefix_sequence) for condition in ("frozen", "adapted")}:
            raise ValueError("Expanded report is missing a frozen/adapted causality-prefix run")
    return binding_by_sequence, report_by_identity


def _head_expectation(condition: str, checkpoint_sha256: str = ADAPTED_HEAD_SHA256) -> dict[str, Any]:
    if condition == "adapted":
        return {
            "selection": "adapted_pose_head",
            "requested_sha256": checkpoint_sha256,
            "installed_sha256": checkpoint_sha256,
        }
    return {
        "selection": "frozen_base",
        "requested_sha256": None,
        "installed_sha256": None,
    }


def _check_inference_identity(
    inference: Any,
    *,
    condition: str,
    sequence_id: str,
    binding: Mapping[str, str],
    checkpoint_sha256: str = ADAPTED_HEAD_SHA256,
) -> dict[str, Any]:
    if not isinstance(inference, dict):
        raise ValueError("Output lacks GoTrack inference provenance")
    spec = SEQUENCE_SPECS[sequence_id]
    _expect(inference.get("schema") == "gotrack_inference_contract_v1",
            "Output inference provenance schema mismatch")
    _expect(inference.get("profile") == _INFERENCE_PROFILE,
            "Output inference profile mismatch")
    geometry = inference.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("pose_units") != _EXPECTED_OUTPUT_POSE_UNITS:
        raise ValueError("Output pose convention must be native cameraFromObject in metres")
    head = inference.get("head")
    if not isinstance(head, dict):
        raise ValueError("Output inference provenance lacks head identity")
    expected_head = _head_expectation(condition, checkpoint_sha256)
    for key, value in expected_head.items():
        _expect(head.get(key) == value, f"Output {condition} head field {key} mismatch")
    if condition == "adapted":
        _expect(head.get("requested_sha256") == head.get("installed_sha256"),
                "Adapted head requested and installed hashes must match")
    input_meta = inference.get("input")
    if not isinstance(input_meta, dict):
        raise ValueError("Output inference provenance lacks input identity")
    expected_identity = {
        "sequence_id": sequence_id,
        "physical_object_id": spec["physical_object_id"],
        "split": spec["split"],
        "generation_receipt_sha256": binding["generation_sha256"],
        "input_sha256": binding["input_sha256"],
    }
    for key, value in expected_identity.items():
        _expect(input_meta.get(key) == value, f"Output input provenance {key} mismatch")
    _expect(input_meta.get("clock_mode") == "capture_seconds_v1",
            "Output input clock metadata mismatch")
    for digest_name in ("video_sha256", "mesh_sha256"):
        if not _is_sha256(input_meta.get(digest_name)):
            raise ValueError(f"Output input provenance {digest_name} is malformed")
    if "runtime_contract_sha256" in binding:
        _expect(inference.get("runtime_contract_sha256") == binding["runtime_contract_sha256"],
                "Output runtime contract differs from the frozen execution binding")
        runtime = inference.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("Output inference provenance lacks runtime closure")
        _expect(runtime.get("source_closure_sha256") == binding["source_closure_sha256"],
                "Output runtime source closure differs from the frozen execution binding")
    return {
        "input": dict(input_meta),
        "head": dict(head),
        "profile": inference["profile"],
        "generation_sha256": input_meta["generation_receipt_sha256"],
        "input_sha256": input_meta["input_sha256"],
        "video_sha256": input_meta["video_sha256"],
        "mesh_sha256": input_meta["mesh_sha256"],
    }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _frame_id(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _pose_or_none(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return _validate_pose(value, name)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a proper finite cameraFromObject SE(3) pose") from exc


def _same_time(first: Any, second: Any, name: str) -> bool:
    return math.isclose(_finite_number(first, name), _finite_number(second, name),
                        rel_tol=0.0, abs_tol=1e-9)


def validate_reference_records(
    generation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_sequence_id: str,
    expected_split: str,
    expected_object_id: str,
    reference_units: str,
) -> dict[int, dict[str, Any]]:
    """Validate all frozen reference identities and normalize frame poses.

    Poses use cameraFromObject in metres. Frame IDs 8..23 are authenticated
    here but deliberately remain outside the requested inference sample set.
    """
    if not isinstance(generation, Mapping):
        raise ValueError("Reference generation metadata must be an object")
    expected_generation = {
        "kind": _GENERATION_KIND,
        "sequence_id": expected_sequence_id,
        "physical_object_id": expected_object_id,
        "split": expected_split,
    }
    for key, value in expected_generation.items():
        if generation.get(key) != value:
            raise ValueError(f"Reference generation {key} identity mismatch")
    if reference_units != "metres":
        raise ValueError("Reference pose units must be metres")
    if generation.get("labels_relative_path") != "poses.jsonl":
        raise ValueError("Reference labels path must be exactly poses.jsonl")
    if "units" in generation and generation.get("units") != "metres":
        raise ValueError("Reference generation pose units must be metres when declared")
    if not _is_sha256(generation.get("labels_sha256")):
        raise ValueError("Reference labels SHA-256 is malformed")

    if len(records) != len(REFERENCE_FRAME_IDS):
        raise ValueError("Reference sidecar must contain exactly 24 rows")
    by_frame: dict[int, dict[str, Any]] = {}
    seen_sources: set[int] = set()
    seen_times: set[float] = set()
    label_kinds: set[str] = set()
    for row_number, row in enumerate(records, 1):
        if not isinstance(row, Mapping) or set(row) != _EXPECTED_REFERENCE_FIELDS:
            raise ValueError(f"Reference row {row_number} has unexpected or missing fields")
        frame_id = _frame_id(row.get("frame_id"), "reference frame_id")
        if frame_id not in REFERENCE_FRAME_IDS or frame_id in by_frame:
            raise ValueError("Reference frame identity is unexpected or duplicated")
        source_frame_id = _frame_id(row.get("source_frame_id"), "reference source_frame_id")
        if source_frame_id in seen_sources:
            raise ValueError("Reference source-frame identity is duplicated")
        if row.get("object_id") != expected_object_id or row.get("sequence_id") != expected_sequence_id:
            raise ValueError("Reference object or sequence identity mismatch")
        if row.get("observationally_ambiguous") is not True:
            raise ValueError("Reference ambiguity declaration must be true for every row")
        label_kind = row.get("label_kind")
        if label_kind != _EXPECTED_LABEL_KIND:
            raise ValueError("Reference label_kind must identify the pinned synthetic renderer pose")
        label_kinds.add(label_kind)
        timestamp = _finite_number(row.get("timestamp_s"), "reference timestamp_s")
        if timestamp in seen_times:
            raise ValueError("Reference timestamps must be unique")
        pose = _pose_or_none(row.get("pose"), "reference pose")
        if pose is None:
            raise ValueError("Reference cameraFromObject pose cannot be null")
        by_frame[frame_id] = {
            "frame_id": frame_id,
            "source_frame_id": source_frame_id,
            "timestamp_s": timestamp,
            "pose": pose,
            "label_kind": label_kind,
            "observationally_ambiguous": True,
        }
        seen_sources.add(source_frame_id)
        seen_times.add(timestamp)
    if set(by_frame) != set(REFERENCE_FRAME_IDS):
        raise ValueError("Reference frame rows must cover exactly frames 0..23")
    if len(label_kinds) != 1:
        raise ValueError("Reference label_kind must not change within a sequence")
    ordered = [by_frame[index] for index in REFERENCE_FRAME_IDS]
    if any(ordered[index]["source_frame_id"] != index for index in REFERENCE_FRAME_IDS):
        raise ValueError("Reference source-frame mapping must match frame IDs 0..23")
    if any(ordered[index]["timestamp_s"] >= ordered[index + 1]["timestamp_s"]
           for index in range(len(ordered) - 1)):
        raise ValueError("Reference timestamps must strictly increase by frame ID")
    return by_frame


def _validated_frame_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stream_name: str,
    frame_id_key: str,
    source_frame_id_key: str,
    reference_frames: Mapping[int, Mapping[str, Any]],
    sequence_id: str,
    object_id: str,
    allowed_frame_ids: Sequence[int] = REQUESTED_FRAME_IDS,
    require_identity_fields: bool = True,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{stream_name} row {row_number} must be an object")
        frame_id = _frame_id(row.get(frame_id_key), f"{stream_name} {frame_id_key}")
        if frame_id not in allowed_frame_ids:
            raise ValueError(f"{stream_name} contains unexpected quality frame {frame_id}")
        if frame_id in result:
            raise ValueError(f"Duplicate {stream_name} frame identity {frame_id}")
        if require_identity_fields or "sequence_id" in row or "object_id" in row:
            if row.get("sequence_id") != sequence_id or row.get("object_id") != object_id:
                raise ValueError(f"{stream_name} object or sequence identity mismatch")
        ref = reference_frames[frame_id]
        source_id = _frame_id(row.get(source_frame_id_key), f"{stream_name} {source_frame_id_key}")
        if source_id != ref["source_frame_id"]:
            raise ValueError(f"{stream_name} source-frame identity mismatch for frame {frame_id}")
        if not _same_time(row.get("timestamp_s"), ref["timestamp_s"],
                          f"{stream_name} timestamp_s"):
            raise ValueError(f"{stream_name} timestamp mismatch for frame {frame_id}")
        result[frame_id] = dict(row)
    return result


def _validate_output_and_candidate_rows(
    complete_doc: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    sequence_id: str,
    object_id: str,
    split: str,
    condition: str,
    binding: Mapping[str, str],
    reference_frames: Mapping[int, Mapping[str, Any]],
    requested_frame_ids: Sequence[int] = REQUESTED_FRAME_IDS,
    checkpoint_sha256: str = ADAPTED_HEAD_SHA256,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    if not isinstance(complete_doc, Mapping):
        raise ValueError("complete.json must be an object")
    _expect(complete_doc.get("requested_frame_ids") == list(requested_frame_ids),
            "Complete output requested-frame identities differ from the policy")
    _expect(complete_doc.get("clock_mode") == "capture_seconds_v1",
            "Complete output clock mode mismatch")
    complete_provenance = complete_doc.get("provenance")
    if not isinstance(complete_provenance, Mapping):
        raise ValueError("Complete output lacks provenance")
    output_identity = _check_inference_identity(
        complete_provenance.get("gotrack_inference"), condition=condition,
        sequence_id=sequence_id, binding=binding,
        checkpoint_sha256=checkpoint_sha256)
    complete_frames = complete_doc.get("frames")
    if not isinstance(complete_frames, list):
        raise ValueError("Complete output frames must be a list")
    all_complete_rows = _validated_frame_rows(
        complete_frames, stream_name="complete.json", frame_id_key="frameId",
        source_frame_id_key="sourceFrameId", reference_frames=reference_frames,
        sequence_id=sequence_id, object_id=object_id,
        allowed_frame_ids=REFERENCE_FRAME_IDS, require_identity_fields=False,
    )
    complete_rows = {frame_id: row for frame_id, row in all_complete_rows.items()
                     if frame_id in requested_frame_ids}
    for frame_id, row in complete_rows.items():
        if "cameraFromObject" not in row:
            raise ValueError(f"complete.json frame {frame_id} lacks cameraFromObject")
        pose = _pose_or_none(row.get("cameraFromObject"), "complete cameraFromObject")
        row["_pose"] = pose
        frame_inference = row.get("gotrack_inference")
        if frame_inference is None:
            raise ValueError(f"complete.json frame {frame_id} lacks inference provenance")
        frame_identity = _check_inference_identity(
            frame_inference, condition=condition, sequence_id=sequence_id, binding=binding,
            checkpoint_sha256=checkpoint_sha256)
        for key in ("generation_sha256", "input_sha256", "video_sha256", "mesh_sha256"):
            _expect(frame_identity[key] == output_identity[key],
                    f"Complete frame {frame_id} changes input provenance {key}")

    candidate_by_frame = _validated_frame_rows(
        candidate_rows, stream_name="candidates.jsonl", frame_id_key="frame_id",
        source_frame_id_key="source_frame_id", reference_frames=reference_frames,
        sequence_id=sequence_id, object_id=object_id,
        allowed_frame_ids=requested_frame_ids,
    )
    for frame_id, row in candidate_by_frame.items():
        if frame_id not in complete_rows:
            # Candidate traces can still establish candidate coverage when a
            # final complete row is absent, but all trace identity remains checked.
            pass
        row_pose = _pose_or_none(row.get("cameraFromObject"), "candidate trace cameraFromObject")
        row["_pose"] = row_pose
        if frame_id in complete_rows:
            complete = complete_rows[frame_id]
            if row.get("pose_state") != complete.get("pose_state"):
                raise ValueError(f"Candidate and complete pose_state disagree at frame {frame_id}")
            if row.get("render_state") != complete.get("render_state"):
                raise ValueError(f"Candidate and complete render_state disagree at frame {frame_id}")
            if (row_pose is None) != (complete["_pose"] is None):
                raise ValueError(f"Candidate and complete emitted-pose availability disagree at frame {frame_id}")
            if row_pose is not None and not np.allclose(row_pose, complete["_pose"], rtol=0.0, atol=1e-9):
                raise ValueError(f"Candidate and complete cameraFromObject disagree at frame {frame_id}")
        candidate_inference = row.get("provenance", {}).get("gotrack_inference")
        candidate_identity = _check_inference_identity(
            candidate_inference, condition=condition, sequence_id=sequence_id, binding=binding,
            checkpoint_sha256=checkpoint_sha256)
        for key in ("generation_sha256", "input_sha256", "video_sha256", "mesh_sha256"):
            _expect(candidate_identity[key] == output_identity[key],
                    f"Candidate frame {frame_id} changes input provenance {key}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Candidate trace frame {frame_id} must contain a candidates list")
        candidate_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError(f"Candidate trace frame {frame_id} has a non-object candidate")
            candidate_id = candidate.get("candidate_id")
            if type(candidate_id) is not str or not candidate_id or candidate_id in candidate_ids:
                raise ValueError(f"Candidate trace frame {frame_id} has unexpected candidate identity")
            candidate_ids.add(candidate_id)
            _pose_or_none(candidate.get("pose"), "candidate pose")
            if type(candidate.get("eligible")) is not bool:
                raise ValueError("Candidate eligible field must be boolean")
            gates = candidate.get("gates")
            if not isinstance(gates, Mapping) or not gates:
                raise ValueError("Candidate gates must be a non-empty object")
            if any(type(value) is not bool for value in gates.values()):
                raise ValueError("Candidate gate states must be boolean")
            rejection = candidate.get("rejection")
            if rejection is not None and type(rejection) is not str:
                raise ValueError("Candidate rejection must be a string or null")
        decision = row.get("rank_decision")
        if not isinstance(decision, Mapping):
            raise ValueError(f"Candidate trace frame {frame_id} lacks rank_decision")
        selected_id = decision.get("selected_id")
        if selected_id is not None and (type(selected_id) is not str or selected_id not in candidate_ids):
            raise ValueError(f"Selected candidate identity is absent from frame {frame_id}")
        ordered = decision.get("ordered_ids")
        if not isinstance(ordered, list) or any(item not in candidate_ids for item in ordered):
            raise ValueError(f"Candidate ordering contains an unknown identity at frame {frame_id}")
        if len(ordered) != len(set(ordered)):
            raise ValueError(f"Candidate ordering repeats an identity at frame {frame_id}")
    return complete_rows, candidate_by_frame, output_identity


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Metric values must remain finite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def translation_increment_residuals(
    predicted_poses: Mapping[int, Any],
    reference_poses: Mapping[int, Any],
    requested_frame_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return translation-increment residuals only across consecutive valid IDs."""
    result = []
    ids = tuple(requested_frame_ids)
    for previous_id, current_id in zip(ids, ids[1:]):
        if current_id != previous_id + 1:
            continue
        if previous_id not in predicted_poses or current_id not in predicted_poses:
            continue
        if previous_id not in reference_poses or current_id not in reference_poses:
            continue
        predicted_previous = _validate_pose(predicted_poses[previous_id], "predicted previous pose")
        predicted_current = _validate_pose(predicted_poses[current_id], "predicted current pose")
        reference_previous = _validate_pose(reference_poses[previous_id], "reference previous pose")
        reference_current = _validate_pose(reference_poses[current_id], "reference current pose")
        predicted_increment = predicted_current[:3, 3] - predicted_previous[:3, 3]
        reference_increment = reference_current[:3, 3] - reference_previous[:3, 3]
        residual = float(np.linalg.norm(predicted_increment - reference_increment))
        if not math.isfinite(residual):
            raise ValueError("Translation increment residual is non-finite")
        result.append({
            "previous_frame_id": previous_id,
            "frame_id": current_id,
            "residual_m": residual,
        })
    return result


def evaluate_sequence(
    *,
    sequence_id: str,
    split: str,
    object_id: str,
    reference_frames: Mapping[int, Mapping[str, Any]],
    complete_frames: Mapping[int, Mapping[str, Any]],
    candidate_frames: Mapping[int, Mapping[str, Any]],
    condition: str,
    requested_frame_ids: Sequence[int] = REQUESTED_FRAME_IDS,
) -> dict[str, Any]:
    """Pure numerical summary for one condition and requested video window."""
    if sequence_id not in SEQUENCE_SPECS or condition not in ("frozen", "adapted"):
        raise ValueError("Unexpected sequence or condition identity")
    if split != SEQUENCE_SPECS[sequence_id]["split"] or object_id != SEQUENCE_SPECS[sequence_id]["physical_object_id"]:
        raise ValueError("Sequence split or physical-object identity mismatch")
    requested = tuple(requested_frame_ids)
    if not requested or len(set(requested)) != len(requested) or any(
            type(frame_id) is not int or frame_id not in REFERENCE_FRAME_IDS for frame_id in requested):
        raise ValueError("Requested frame denominator must use unique IDs from the 24-frame reference window")
    if tuple(sorted(requested)) != requested:
        raise ValueError("Requested frame IDs must be in ascending order")
    if any(frame_id not in reference_frames for frame_id in requested):
        raise ValueError("A requested frame is missing its required reference row")
    for frame_id in requested:
        reference = reference_frames[frame_id]
        if reference.get("observationally_ambiguous") is not True:
            raise ValueError("Primary orientation is unmeasured only for ambiguous references")
        _validate_pose(reference.get("pose"), "reference cameraFromObject")
    emitted_poses: dict[int, np.ndarray] = {}
    emitted_errors: list[dict[str, Any]] = []
    pose_state_counts: Counter[str] = Counter()
    render_state_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    frame_availability_rows: list[dict[str, Any]] = []
    observed_mask_available = 0
    observed_mask_measured = 0
    emitted_pose_rows = 0
    candidate_count = 0
    valid_candidates_before_count = 0
    valid_candidates_after_count = 0
    valid_before_frames: set[int] = set()
    valid_after_frames: set[int] = set()
    selected_candidate_rows = 0
    selected_candidate_pose_available = 0
    candidate_rejections: Counter[str] = Counter()
    minimum_before: list[tuple[int, float]] = []
    minimum_after: list[tuple[int, float]] = []

    for frame_id in requested:
        frame = complete_frames.get(frame_id)
        pose_state = None
        render_state = None
        failure_reason = None
        mask_state = None
        mask_measurement_state = None
        pose = None
        error = None
        if frame is not None:
            pose_state = frame.get("pose_state")
            render_state = frame.get("render_state")
            if type(pose_state) is not str or type(render_state) is not str:
                raise ValueError("Produced output rows must declare string pose_state and render_state")
            pose_state_counts[pose_state] += 1
            render_state_counts[render_state] += 1
            failure_reason = frame.get("failure_reason")
            if failure_reason is not None:
                if type(failure_reason) is not str:
                    raise ValueError("Output failure_reason must be a string or null")
                failure_reasons[failure_reason] += 1
            if frame.get("mask_state") == "available":
                observed_mask_available += 1
            if frame.get("mask_measurement_state") == "measured":
                observed_mask_measured += 1
            mask_state = frame.get("mask_state")
            mask_measurement_state = frame.get("mask_measurement_state")
            pose = frame.get("_pose")
            if pose is None and "_pose" not in frame:
                pose = _pose_or_none(frame.get("cameraFromObject"), "emitted cameraFromObject")
            if pose is not None:
                if pose_state != "tracking" or render_state != "visible":
                    raise ValueError(
                        f"Emitted pose at frame {frame_id} requires tracking and visible states")
                emitted_pose_rows += 1
                emitted_poses[frame_id] = pose
                reference_pose = reference_frames[frame_id]["pose"]
                error = translation_error_m(pose, reference_pose)
                emitted_errors.append({
                    "sequence_id": sequence_id,
                    "physical_object_id": object_id,
                    "frame_id": frame_id,
                    "translation_error_m": error,
                })
            elif pose_state == "tracking" and render_state == "visible":
                raise ValueError(
                    f"Tracking/visible output at frame {frame_id} must emit cameraFromObject")

        frame_availability_rows.append({
            "sequence_id": sequence_id,
            "physical_object_id": object_id,
            "frame_id": frame_id,
            "output_row_available": frame is not None,
            "pose_state": pose_state,
            "render_state": render_state,
            "mask_state": mask_state,
            "mask_measurement_state": mask_measurement_state,
            "failure_reason": failure_reason,
            "emitted_pose_available": pose is not None,
            "emitted_pose_unavailable_reason": (
                None if pose is not None else
                "missing_output_row" if frame is None else "null_cameraFromObject"
            ),
            "translation_error_m": error,
        })

        trace = candidate_frames.get(frame_id)
        if trace is None:
            continue
        candidates = trace.get("candidates", [])
        valid_before: list[float] = []
        valid_after: list[float] = []
        candidate_count += len(candidates)
        for candidate in candidates:
            pose = _pose_or_none(candidate.get("pose"), "candidate pose")
            if pose is None:
                reason = candidate.get("rejection") or "missing_pose"
                candidate_rejections[str(reason)] += 1
                continue
            valid_candidates_before_count += 1
            valid_before_frames.add(frame_id)
            ref_pose = reference_frames[frame_id]["pose"]
            candidate_error = translation_error_m(pose, ref_pose)
            valid_before.append(candidate_error)
            gates = candidate.get("gates", {})
            passed = candidate.get("eligible") is True and all(value is True for value in gates.values())
            if passed:
                valid_candidates_after_count += 1
                valid_after_frames.add(frame_id)
                valid_after.append(candidate_error)
            else:
                rejection = candidate.get("rejection")
                if rejection:
                    candidate_rejections[rejection] += 1
                failed_gates = sorted(name for name, value in gates.items() if value is False)
                for gate in failed_gates:
                    candidate_rejections[f"gate:{gate}"] += 1
                if not failed_gates and not rejection:
                    candidate_rejections["ineligible_without_reason"] += 1
        if valid_before:
            minimum_before.append((frame_id, min(valid_before)))
        if valid_after:
            minimum_after.append((frame_id, min(valid_after)))
        decision = trace.get("rank_decision", {})
        selected_id = decision.get("selected_id")
        if selected_id is not None:
            selected_candidate_rows += 1
            selected = next((candidate for candidate in candidates
                             if candidate.get("candidate_id") == selected_id), None)
            if selected is not None and _pose_or_none(selected.get("pose"), "selected candidate pose") is not None:
                selected_candidate_pose_available += 1

    transitions = translation_increment_residuals(
        emitted_poses,
        {frame_id: reference_frames[frame_id]["pose"] for frame_id in requested},
        requested,
    )
    for row in transitions:
        row["sequence_id"] = sequence_id
        row["physical_object_id"] = object_id
    emitted_errors.sort(key=lambda row: row["frame_id"])
    errors = [row["translation_error_m"] for row in emitted_errors]
    requested_count = len(requested)
    produced_count = sum(frame_id in complete_frames for frame_id in requested)
    trace_count = sum(frame_id in candidate_frames for frame_id in requested)
    missing_output_frames = [row["frame_id"] for row in frame_availability_rows
                             if not row["output_row_available"]]
    unavailable_emitted_frames = [row["frame_id"] for row in frame_availability_rows
                                  if not row["emitted_pose_available"]]
    return {
        "condition": condition,
        "sequence_id": sequence_id,
        "split": split,
        "physical_object_id": object_id,
        "requested_frame_ids": list(requested),
        "requested_count": requested_count,
        "produced_count": produced_count,
        "missing_output_row_count": requested_count - produced_count,
        "candidate_trace_row_count": trace_count,
        "tracking_count": pose_state_counts["tracking"],
        "recovering_count": pose_state_counts["recovering"],
        "lost_count": pose_state_counts["lost"],
        "suppressed_count": render_state_counts["suppressed"],
        "emitted_pose_count": emitted_pose_rows,
        "emitted_reference_join_count": len(emitted_errors),
        "observed_mask_available_count": observed_mask_available,
        "observed_mask_measured_count": observed_mask_measured,
        "pose_state_counts": dict(sorted(pose_state_counts.items())),
        "render_state_counts": dict(sorted(render_state_counts.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "frame_availability": {
            "requested_count": requested_count,
            "output_row_available_count": produced_count,
            "emitted_pose_available_count": emitted_pose_rows,
            "missing_output_frame_ids": missing_output_frames,
            "emitted_pose_unavailable_frame_ids": unavailable_emitted_frames,
            "rows": frame_availability_rows,
        },
        "emitted_translation_errors": {
            **_distribution(errors), "unit": "metres", "rows": emitted_errors,
        },
        "candidate_coverage": {
            "candidate_count": candidate_count,
            "valid_candidate_count_before_gates": valid_candidates_before_count,
            "valid_candidate_count_after_gates": valid_candidates_after_count,
            "frames_with_valid_candidates_before_gates": len(valid_before_frames),
            "frames_with_valid_candidates_after_gates": len(valid_after_frames),
            "selected_candidate_rows": selected_candidate_rows,
            "selected_candidate_pose_available_count": selected_candidate_pose_available,
            "minimum_translation_error_before_gates": {
                **_distribution([value for _, value in minimum_before]), "unit": "metres",
                "counterfactual_within_head_candidate_set": True,
                "rows": [{"sequence_id": sequence_id, "physical_object_id": object_id,
                          "frame_id": frame_id, "minimum_translation_error_m": value}
                         for frame_id, value in minimum_before],
            },
            "minimum_translation_error_after_gates": {
                **_distribution([value for _, value in minimum_after]), "unit": "metres",
                "counterfactual_within_head_candidate_set": True,
                "rows": [{"sequence_id": sequence_id, "physical_object_id": object_id,
                          "frame_id": frame_id, "minimum_translation_error_m": value}
                         for frame_id, value in minimum_after],
            },
            "rejection_reasons": dict(sorted(candidate_rejections.items())),
        },
        "temporal_translation_increment_residuals": {
            **_distribution([row["residual_m"] for row in transitions]),
            "unit": "metres",
            "possible_consecutive_requested_transitions": sum(
                1 for first, second in zip(requested, requested[1:]) if second == first + 1),
            "valid_emitted_endpoint_count": len(transitions),
            "rows": transitions,
        },
        "orientation_primary": {
            "status": "unmeasured_observationally_ambiguous_reference",
            "scored_error_values_count": 0,
            "eligible_denominator": 0,
            "mean_degrees": None,
            "median_degrees": None,
            "p95_degrees": None,
        },
        "severe_orientation": {
            "status": "unmeasured_observationally_ambiguous_reference",
            "eligible_denominator": 0,
            "failure_count": None,
            "failure_rate": None,
        },
        "surface_attachment": "unmeasured",
        "projection_iou": "unmeasured",
        "occlusion_return": "unmeasured",
    }


def _summarize_condition(sequences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not sequences:
        raise ValueError("Condition summary requires sequence results")
    result: dict[str, Any] = {"condition": sequences[0]["condition"]}
    count_fields = (
        "requested_count", "produced_count", "missing_output_row_count",
        "candidate_trace_row_count", "tracking_count", "recovering_count",
        "lost_count", "suppressed_count", "emitted_pose_count",
        "emitted_reference_join_count", "observed_mask_available_count",
        "observed_mask_measured_count",
    )
    for field in count_fields:
        result[field] = sum(int(row[field]) for row in sequences)
    emitted_rows = [item for row in sequences for item in row["emitted_translation_errors"]["rows"]]
    candidate_rows = [row["candidate_coverage"] for row in sequences]
    result["emitted_translation_errors"] = {
        **_distribution([item["translation_error_m"] for item in emitted_rows]),
        "unit": "metres", "rows": emitted_rows,
    }
    result["candidate_coverage"] = {
        "candidate_count": sum(row["candidate_count"] for row in candidate_rows),
        "valid_candidate_count_before_gates": sum(
            row["valid_candidate_count_before_gates"] for row in candidate_rows),
        "valid_candidate_count_after_gates": sum(
            row["valid_candidate_count_after_gates"] for row in candidate_rows),
        "frames_with_valid_candidates_before_gates": sum(
            row["frames_with_valid_candidates_before_gates"] for row in candidate_rows),
        "frames_with_valid_candidates_after_gates": sum(
            row["frames_with_valid_candidates_after_gates"] for row in candidate_rows),
        "selected_candidate_rows": sum(row["selected_candidate_rows"] for row in candidate_rows),
        "selected_candidate_pose_available_count": sum(
            row["selected_candidate_pose_available_count"] for row in candidate_rows),
        "minimum_translation_error_before_gates": {
            **_distribution([item["minimum_translation_error_m"]
                             for row in candidate_rows
                             for item in row["minimum_translation_error_before_gates"]["rows"]]),
            "unit": "metres", "pooled_across_sequence_frames": True,
            "counterfactual_within_head_candidate_set": True,
        },
        "minimum_translation_error_after_gates": {
            **_distribution([item["minimum_translation_error_m"]
                             for row in candidate_rows
                             for item in row["minimum_translation_error_after_gates"]["rows"]]),
            "unit": "metres", "pooled_across_sequence_frames": True,
            "counterfactual_within_head_candidate_set": True,
        },
        "rejection_reasons": dict(sorted(sum(
            (Counter(row["rejection_reasons"]) for row in candidate_rows), Counter()).items())),
    }
    temporal_rows = [item for row in sequences
                     for item in row["temporal_translation_increment_residuals"]["rows"]]
    result["temporal_translation_increment_residuals"] = {
        **_distribution([item["residual_m"] for item in temporal_rows]),
        "unit": "metres",
        "possible_consecutive_requested_transitions": sum(
            row["temporal_translation_increment_residuals"]["possible_consecutive_requested_transitions"]
            for row in sequences),
        "valid_emitted_endpoint_count": len(temporal_rows),
        "rows": temporal_rows,
    }
    result["failure_reasons"] = dict(sorted(sum(
        (Counter(row["failure_reasons"]) for row in sequences), Counter()).items()))
    result["orientation_primary"] = {
        "status": "unmeasured_observationally_ambiguous_reference",
        "scored_error_values_count": 0,
        "eligible_denominator": 0,
        "mean_degrees": None,
        "median_degrees": None,
        "p95_degrees": None,
    }
    result["severe_orientation"] = {
        "status": "unmeasured_observationally_ambiguous_reference",
        "eligible_denominator": 0,
        "failure_count": None,
        "failure_rate": None,
    }
    result["surface_attachment"] = "unmeasured"
    result["projection_iou"] = "unmeasured"
    result["occlusion_return"] = "unmeasured"
    frame_rows = [item for row in sequences for item in row["frame_availability"]["rows"]]
    missing_output_frames = [item for item in frame_rows if not item["output_row_available"]]
    unavailable_pose_frames = [item for item in frame_rows if not item["emitted_pose_available"]]
    result["frame_availability"] = {
        "requested_count": len(frame_rows),
        "output_row_available_count": sum(item["output_row_available"] for item in frame_rows),
        "emitted_pose_available_count": sum(item["emitted_pose_available"] for item in frame_rows),
        "missing_output_frame_ids": [
            {"sequence_id": item["sequence_id"],
             "physical_object_id": item["physical_object_id"],
             "frame_id": item["frame_id"]}
            for item in missing_output_frames
        ],
        "emitted_pose_unavailable_frame_ids": [
            {"sequence_id": item["sequence_id"],
             "physical_object_id": item["physical_object_id"],
             "frame_id": item["frame_id"],
             "reason": item["emitted_pose_unavailable_reason"]}
            for item in unavailable_pose_frames
        ],
        "rows": frame_rows,
    }
    result["sequences"] = list(sequences)
    return result


def _summarize_conditions(
    condition_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    requested_frame_ids: Sequence[int] = REQUESTED_FRAME_IDS,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    split_specs = (
        ("train", "train", "training_resubstitution"),
        ("used_val", "validation", "previously_used_development_validation"),
    )
    for condition in ("frozen", "adapted"):
        by_sequence = condition_results.get(condition)
        if not isinstance(by_sequence, Mapping) or set(by_sequence) != set(SEQUENCE_SPECS):
            raise ValueError(f"{condition} results must cover the exact three quality sequences")
        sequences = [by_sequence[sequence_id] for sequence_id in SEQUENCE_SPECS]
        summary = _summarize_condition(sequences)
        split_aggregates = {}
        for key, split, partition in split_specs:
            split_sequences = [row for row in sequences if row["split"] == split]
            if not split_sequences:
                raise ValueError(f"Condition {condition} has no {split} sequences")
            split_summary = _summarize_condition(split_sequences)
            split_summary["split"] = split
            split_summary["partition"] = partition
            split_aggregates[key] = split_summary
        summary["split_aggregates"] = split_aggregates
        summaries[condition] = summary
    return summaries


def _compare_conditions(
    condition_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    requested_frame_ids: Sequence[int] = REQUESTED_FRAME_IDS,
) -> list[dict[str, Any]]:
    comparisons = []
    for sequence_id, spec in SEQUENCE_SPECS.items():
        frozen = condition_results["frozen"][sequence_id]
        adapted = condition_results["adapted"][sequence_id]
        frozen_errors = {
            row["frame_id"]: row["translation_error_m"]
            for row in frozen["emitted_translation_errors"]["rows"]
        }
        adapted_errors = {
            row["frame_id"]: row["translation_error_m"]
            for row in adapted["emitted_translation_errors"]["rows"]
        }
        common = sorted(set(frozen_errors) & set(adapted_errors))
        differences = [adapted_errors[frame_id] - frozen_errors[frame_id] for frame_id in common]
        comparisons.append({
            "sequence_id": sequence_id,
            "split": spec["split"],
            "physical_object_id": spec["physical_object_id"],
            "requested_count_per_condition": len(requested_frame_ids),
            "frozen_emitted_reference_count": len(frozen_errors),
            "adapted_emitted_reference_count": len(adapted_errors),
            "common_emitted_reference_count": len(common),
            "common_frame_ids": common,
            "adapted_minus_frozen_translation_error_m": {
                **_distribution(differences), "unit": "metres",
                "paired_on_common_emitted_reference_intersection": True,
                "rows": [
                    {"sequence_id": sequence_id,
                     "physical_object_id": spec["physical_object_id"],
                     "frame_id": frame_id,
                     "adapted_minus_frozen_m": adapted_errors[frame_id] - frozen_errors[frame_id]}
                    for frame_id in common
                ],
            },
        })
    all_common_differences = [
        row["adapted_minus_frozen_translation_error_m"]["rows"]
        for row in comparisons
    ]
    flat = [item["adapted_minus_frozen_m"] for rows in all_common_differences for item in rows]
    comparisons.append({
        "aggregate": True,
        "sequence_count": len(SEQUENCE_SPECS),
        "requested_count_per_condition": len(SEQUENCE_SPECS) * len(requested_frame_ids),
        "frozen_emitted_reference_count": sum(row["frozen_emitted_reference_count"] for row in comparisons),
        "adapted_emitted_reference_count": sum(row["adapted_emitted_reference_count"] for row in comparisons),
        "common_emitted_reference_count": len(flat),
        "adapted_minus_frozen_translation_error_m": {
            **_distribution(flat), "unit": "metres",
            "paired_on_common_emitted_reference_intersection": True,
        },
    })
    return comparisons


def _expected_output_member(condition: str, sequence_id: str, name: str,
                            output_prefix: str = "eight") -> str:
    return f"{output_prefix}-{condition}-{sequence_id}/{name}"


def _load_sealed_outputs(
    snapshot: Mapping[str, bytes],
    binding_by_sequence: Mapping[str, Mapping[str, str]],
    reference_frames: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    policy: _EvaluationPolicy = _LEGACY_POLICY,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]], dict[str, str]]:
    all_sequences: dict[str, dict[str, dict[str, Any]]] = {"frozen": {}, "adapted": {}}
    identities: dict[str, dict[str, Any]] = {}
    consumed_hashes: dict[str, str] = {}
    for condition, sequence_id in sorted(EXPECTED_RUNS):
        complete_name = _expected_output_member(condition, sequence_id, "complete.json",
                                                policy.output_prefix)
        candidate_name = _expected_output_member(condition, sequence_id, "candidates.jsonl",
                                                  policy.output_prefix)
        complete_raw = snapshot[complete_name]
        candidates_raw = snapshot[candidate_name]
        consumed_hashes[complete_name] = _sha256(complete_raw)
        consumed_hashes[candidate_name] = _sha256(candidates_raw)
        complete_doc = _parse_json_bytes(complete_raw, complete_name)
        candidate_rows = _parse_jsonl_bytes(candidates_raw, candidate_name)
        spec = SEQUENCE_SPECS[sequence_id]
        if sequence_id not in reference_frames:
            raise ValueError("Authenticated references are incomplete for a sealed output sequence")
        complete_rows, candidate_by_frame, identity = _validate_output_and_candidate_rows(
            complete_doc, candidate_rows, sequence_id=sequence_id,
            object_id=spec["physical_object_id"], split=spec["split"],
            condition=condition, binding=binding_by_sequence[sequence_id],
            reference_frames=reference_frames[sequence_id],
            requested_frame_ids=policy.requested_frame_ids,
            checkpoint_sha256=policy.checkpoint_sha256,
        )
        existing = identities.get(sequence_id)
        if existing is not None:
            for key in ("generation_sha256", "input_sha256", "video_sha256", "mesh_sha256"):
                _expect(existing[key] == identity[key],
                        f"Frozen/adapted outputs disagree on sequence input pin {key}")
        else:
            identities[sequence_id] = identity
        all_sequences[condition][sequence_id] = {
            "complete": complete_rows,
            "candidates": candidate_by_frame,
        }
    return all_sequences, identities, consumed_hashes


def _causal_signature(value: Any) -> Any:
    """Remove only profiling fields, preserving every tracking and rank decision."""
    if isinstance(value, Mapping):
        return {key: _causal_signature(item) for key, item in value.items()
                if key not in _PROFILE_ONLY_FIELDS}
    if isinstance(value, list):
        return [_causal_signature(item) for item in value]
    if isinstance(value, np.ndarray):
        return _causal_signature(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_causality_prefix_outputs(
    snapshot: Mapping[str, bytes],
    binding_by_sequence: Mapping[str, Mapping[str, str]],
    reference_frames: Mapping[str, Mapping[int, Mapping[str, Any]]],
    full_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    sequence_id = "ranker-multiview-hot3d17-v1-yaw0-pitchm25"
    spec = SEQUENCE_SPECS[sequence_id]
    requested_ids = tuple(range(4))
    results = []
    consumed = {}
    for condition in ("frozen", "adapted"):
        prefix = f"four-{condition}-TRAIN17minus25"
        complete_name = f"{prefix}/complete.json"
        candidate_name = f"{prefix}/candidates.jsonl"
        complete_raw, candidates_raw = snapshot[complete_name], snapshot[candidate_name]
        consumed[complete_name] = _sha256(complete_raw)
        consumed[candidate_name] = _sha256(candidates_raw)
        complete_doc = _parse_json_bytes(complete_raw, complete_name)
        candidate_rows = _parse_jsonl_bytes(candidates_raw, candidate_name)
        prefix_complete, prefix_candidates, _identity = _validate_output_and_candidate_rows(
            complete_doc, candidate_rows, sequence_id=sequence_id,
            object_id=spec["physical_object_id"], split=spec["split"],
            condition=condition, binding=binding_by_sequence[sequence_id],
            reference_frames=reference_frames[sequence_id],
            requested_frame_ids=requested_ids,
            checkpoint_sha256=checkpoint_sha256,
        )
        full = full_outputs[condition][sequence_id]
        for frame_id in requested_ids:
            full_row = full["complete"].get(frame_id)
            prefix_row = prefix_complete.get(frame_id)
            if ((full_row is None) != (prefix_row is None) or
                    (full_row is not None and
                     _causal_signature(full_row) !=
                     _causal_signature(prefix_row))):
                raise ValueError(
                    f"Four-frame prefix and full replay differ in emitted tracking at frame {frame_id}/{condition}")
            full_trace = full["candidates"].get(frame_id)
            prefix_trace = prefix_candidates.get(frame_id)
            if ((full_trace is None) != (prefix_trace is None) or
                    (full_trace is not None and
                     _causal_signature(full_trace) !=
                     _causal_signature(prefix_trace))):
                raise ValueError(
                    f"Four-frame prefix and full replay differ in candidates at frame {frame_id}/{condition}")
        prefix_summary = evaluate_sequence(
            sequence_id=sequence_id, split=spec["split"],
            object_id=spec["physical_object_id"],
            reference_frames=reference_frames[sequence_id],
            complete_frames=prefix_complete,
            candidate_frames=prefix_candidates,
            condition=condition,
            requested_frame_ids=requested_ids,
        )
        results.append({
            "condition": condition,
            "sequence_id": sequence_id,
            "requested_frame_ids": list(requested_ids),
            "requested_count": len(requested_ids),
            "produced_count": prefix_summary["produced_count"],
            "missing_output_row_count": prefix_summary["missing_output_row_count"],
            "candidate_trace_row_count": prefix_summary["candidate_trace_row_count"],
            "emitted_pose_count": prefix_summary["emitted_pose_count"],
            "causal_prefix_equal_to_full_video": True,
        })
    return results, consumed


def _parse_diagnostic(raw: bytes) -> dict[str, dict[str, Any]]:
    if _sha256(raw) != PINNED_DIAGNOSTIC_SHA256:
        raise ValueError("Adaptation-label diagnostic SHA-256 mismatch")
    document = _parse_json_bytes(raw, "adaptation-label-diagnostic-v1.json")
    if not isinstance(document, list) or len(document) != 3:
        raise ValueError("Pinned label diagnostic must contain exactly three sequence records")
    result = {}
    for entry in document:
        if not isinstance(entry, dict):
            raise ValueError("Label diagnostic sequence entry must be an object")
        sequence = entry.get("sequence")
        generation = entry.get("generation")
        group = entry.get("group")
        manifest = entry.get("manifest")
        spec = SEQUENCE_SPECS.get(sequence.get("sequence_id") if isinstance(sequence, dict) else None)
        if spec is None or sequence["sequence_id"] in result:
            raise ValueError("Label diagnostic contains an unexpected or duplicate sequence")
        for metadata in (generation, group, manifest):
            if not isinstance(metadata, dict):
                raise ValueError("Label diagnostic metadata is incomplete")
        sequence_id = sequence["sequence_id"]
        expected = {
            "physical_object_id": spec["physical_object_id"],
            "sequence_id": sequence_id,
            "split": spec["split"],
        }
        for source in (generation, group, manifest, sequence):
            for key, value in expected.items():
                actual = source.get(key)
                if key == "split" and actual is None and source is manifest:
                    continue
                if actual != value:
                    # Diagnostic split metadata uses the design alias `val`.
                    if key == "split" and actual == "val" and spec["design_split"] == "val":
                        continue
                    raise ValueError(f"Pinned label diagnostic {key} identity mismatch")
        if sequence.get("observationally_ambiguous") is not True:
            raise ValueError("Pinned diagnostic does not declare ambiguous orientation")
        if manifest.get("units") != "metres":
            raise ValueError("Pinned original recipe metadata must declare metre pose units")
        clock = manifest.get("clock")
        if (not isinstance(clock, dict) or clock.get("mode") != "capture_seconds_v1" or
                clock.get("units") != "seconds"):
            raise ValueError("Pinned original recipe clock metadata is invalid")
        if not _is_sha256(generation.get("labels_sha256")):
            raise ValueError("Pinned diagnostic label SHA-256 is malformed")
        result[sequence_id] = entry
    if set(result) != set(SEQUENCE_SPECS):
        raise ValueError("Pinned label diagnostic is missing a sequence")
    return result


def _validate_generation_against_diagnostic(
    generation: Any,
    diagnostic_entry: Mapping[str, Any],
    *,
    sequence_id: str,
) -> None:
    if not isinstance(generation, dict):
        raise ValueError("Reference generation.json must be an object")
    diag_generation = diagnostic_entry["generation"]
    diag_manifest = diagnostic_entry["manifest"]
    diag_group = diagnostic_entry["group"]
    spec = SEQUENCE_SPECS[sequence_id]
    for key, value in {
        "kind": _GENERATION_KIND,
        "schema_version": 1,
        "sequence_id": sequence_id,
        "physical_object_id": spec["physical_object_id"],
        "split": spec["split"],
    }.items():
        if generation.get(key) != value:
            raise ValueError(f"Reference generation {key} identity mismatch")
    for key in ("mesh_sha256", "recipe_sha256", "video_sha256", "labels_sha256"):
        digest = generation.get(key)
        if not _is_sha256(digest) or digest != diag_generation.get(key):
            raise ValueError(f"Reference generation {key} differs from the pinned diagnostic")
    if generation.get("mesh_sha256") != diag_manifest.get("source_hashes", {}).get("object.glb"):
        raise ValueError("Reference mesh SHA-256 differs from the original recipe manifest")
    if generation.get("video_sha256") != diag_manifest.get("source_hashes", {}).get("synthetic.mp4"):
        raise ValueError("Reference video SHA-256 differs from the original recipe manifest")
    if generation.get("mesh_sha256") != diag_group.get("mesh_sha256"):
        raise ValueError("Reference mesh identity differs from the frozen group")
    if generation.get("labels_relative_path") != "poses.jsonl":
        raise ValueError("Reference generation label path must be exactly poses.jsonl")
    if generation.get("inference_bundle_relative_path") != "bundle":
        raise ValueError("Reference generation bundle path must be exactly bundle")
    files = generation.get("inference_bundle_files")
    if files != ["input.json", "object.glb", "synthetic.mp4"]:
        raise ValueError("Reference generation bundle identity is unexpected")


def _reference_inventory(reference_root: Path) -> set[str]:
    expected_files = {
        f"{sequence_id}/{name}"
        for sequence_id in SEQUENCE_SPECS
        for name in ("generation.json", "poses.jsonl")
    }
    actual_files = {
        entry.relative_to(reference_root).as_posix()
        for entry in reference_root.rglob("*") if entry.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Reference root contains unexpected or missing input identities")
    total_bytes = sum((reference_root / relative).stat().st_size for relative in actual_files)
    if total_bytes > MAX_REFERENCE_TOTAL_BYTES:
        raise ValueError("Reference inputs exceed the 1 MiB total byte bound")
    return expected_files


def _load_references(
    reference_root: Path,
    diagnostic_raw: bytes,
    binding_by_sequence: Mapping[str, Mapping[str, str]],
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    diagnostic_by_sequence = _parse_diagnostic(diagnostic_raw)
    _reference_inventory(reference_root)
    references: dict[str, dict[int, dict[str, Any]]] = {}
    reference_hashes: dict[str, Any] = {}
    for sequence_id, spec in SEQUENCE_SPECS.items():
        generation_rel = f"{sequence_id}/generation.json"
        poses_rel = f"{sequence_id}/poses.jsonl"
        generation_raw = _read_bounded_file(
            _safe_member(reference_root, generation_rel),
            max_bytes=MAX_REFERENCE_TOTAL_BYTES, name=generation_rel)
        poses_raw = _read_bounded_file(
            _safe_member(reference_root, poses_rel),
            max_bytes=MAX_REFERENCE_TOTAL_BYTES, name=poses_rel)
        expected_generation_sha = binding_by_sequence[sequence_id]["generation_sha256"]
        if _sha256(generation_raw) != expected_generation_sha:
            raise ValueError(f"Reference generation bytes do not match sealed binding: {sequence_id}")
        generation = _parse_json_bytes(generation_raw, generation_rel)
        _validate_generation_against_diagnostic(
            generation, diagnostic_by_sequence[sequence_id], sequence_id=sequence_id,
        )
        output_identity = identities[sequence_id]
        for key in ("generation_sha256", "video_sha256", "mesh_sha256"):
            ref_value = expected_generation_sha if key == "generation_sha256" else generation.get(key)
            if output_identity[key] != ref_value:
                raise ValueError(f"Sealed output and reference generation disagree on {key}: {sequence_id}")
        if _sha256(poses_raw) != generation.get("labels_sha256"):
            raise ValueError(f"Reference pose sidecar SHA-256 mismatch: {sequence_id}")
        records = _parse_jsonl_bytes(poses_raw, poses_rel)
        references[sequence_id] = validate_reference_records(
            generation, records, expected_sequence_id=sequence_id,
            expected_split=spec["split"], expected_object_id=spec["physical_object_id"],
            reference_units=diagnostic_by_sequence[sequence_id]["manifest"]["units"],
        )
        reference_hashes[sequence_id] = {
            "generation_json_sha256": _sha256(generation_raw),
            "poses_jsonl_sha256": _sha256(poses_raw),
            "labels_sha256": generation["labels_sha256"],
        }
    return references, reference_hashes


def _parse_paired_report_bytes(raw: bytes, *, policy_id: str) -> Any:
    # Expanded receipts repeat authenticated provenance for eight runs. Keep
    # this larger bound confined to that report; other JSON and legacy limits
    # remain unchanged, underneath the authenticated 32 MiB expanded seal cap.
    limit = (MAX_EXPANDED_PAIRED_REPORT_BYTES
             if policy_id == EXPANDED_POLICY_ID else MAX_JSON_BYTES)
    return _parse_json_bytes(raw, "paired-report.json", max_bytes=limit)


def evaluate_postseal(
    *,
    sealed_root: str | Path,
    sealed_manifest_sha256: str,
    reference_root: str | Path,
    diagnostic_path: str | Path,
    experiment_policy: str = "legacy8",
    experiment_policy_json: str | Path | None = None,
    experiment_policy_sha256: str | None = None,
    sealed_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate, join, and summarize the sealed comparison and references."""
    if experiment_policy == "legacy8":
        if any(value is not None for value in (
                experiment_policy_json, experiment_policy_sha256, sealed_archive)):
            raise ValueError("Expanded policy inputs require --experiment-policy expanded24")
        policy = _LEGACY_POLICY
    elif experiment_policy == EXPANDED_POLICY_ID:
        if experiment_policy_json is None or experiment_policy_sha256 is None or sealed_archive is None:
            raise ValueError(
                "expanded24 requires a policy JSON, its SHA-256, and a sealed archive")
        policy = _load_expanded_policy(experiment_policy_json, experiment_policy_sha256)
    else:
        raise ValueError("Unknown experiment policy; expected legacy8 or expanded24")
    if sealed_manifest_sha256 != policy.manifest_sha256:
        raise ValueError("Supplied sealed-manifest SHA-256 differs from the selected policy")
    sealed = _safe_root(sealed_root, "sealed artifact")
    _manifest, snapshot = authenticate_member_manifest(
        sealed, sealed_manifest_sha256,
        expected_members=EXPECTED_SEALED_MEMBERS if experiment_policy == "legacy8" else None,
        total_bytes_max=policy.sealed_total_bytes_max)
    if experiment_policy == "legacy8" and len(snapshot) != 42:
        raise ValueError("Pinned paired result must contain exactly 42 authenticated members")
    if experiment_policy == EXPANDED_POLICY_ID:
        _verify_archive_matches_snapshot(sealed_archive, policy, snapshot)
        _validate_expanded_inventory(_manifest, snapshot)

    design = _parse_json_bytes(snapshot["design.json"], "design.json")
    bindings = _parse_json_bytes(snapshot["execution-bindings.json"], "execution-bindings.json")
    paired_report = _parse_paired_report_bytes(
        snapshot["paired-report.json"], policy_id=policy.policy_id)
    if experiment_policy == EXPANDED_POLICY_ID:
        if _sha256(snapshot["design.json"]) != policy.design_sha256:
            raise ValueError("Expanded design bytes do not match their policy pin")
        if _sha256(snapshot["execution-bindings.json"]) != policy.bindings_sha256:
            raise ValueError("Expanded execution-binding bytes do not match their policy pin")
    binding_by_sequence, _report_by_identity = _validate_sealed_metadata(
        design, bindings, paired_report, policy=policy)

    # The parent supplies this small metadata diagnostic as an independently
    # pinned source recipe record; authenticate its bytes before parsing it.
    diagnostic_file = Path(diagnostic_path).expanduser().absolute()
    for component in reversed((diagnostic_file, *diagnostic_file.parents)):
        if component.is_symlink():
            raise ValueError(f"Diagnostic path cannot contain symlinks: {component}")
    if not diagnostic_file.is_file():
        raise ValueError("Diagnostic must be an existing regular file")
    diagnostic_raw = _read_bounded_file(
        diagnostic_file, max_bytes=MAX_JSON_BYTES, name="adaptation-label-diagnostic-v1.json")
    if _sha256(diagnostic_raw) != policy.diagnostic_sha256:
        raise ValueError("Adaptation-label diagnostic bytes do not match the selected policy SHA-256")

    refs_root = _safe_root(reference_root, "reference")
    _reference_inventory(refs_root)
    # Validate reference-generation bytes against the sealed pins before parsing
    # any pose sidecar. Only then can pose labels be opened and scored.
    generation_bytes: dict[str, bytes] = {}
    generations: dict[str, dict[str, Any]] = {}
    for sequence_id in SEQUENCE_SPECS:
        generation_rel = f"{sequence_id}/generation.json"
        raw = _read_bounded_file(
            _safe_member(refs_root, generation_rel),
            max_bytes=MAX_REFERENCE_TOTAL_BYTES, name=generation_rel)
        if _sha256(raw) != binding_by_sequence[sequence_id]["generation_sha256"]:
            raise ValueError(f"Reference generation hash is not sealed for {sequence_id}")
        generation_bytes[sequence_id] = raw
        generations[sequence_id] = _parse_json_bytes(raw, generation_rel)

    # Check output identities against generation metadata before parsing labels.
    preliminary_identities: dict[str, dict[str, Any]] = {}
    for condition, sequence_id in sorted(EXPECTED_RUNS):
        spec = SEQUENCE_SPECS[sequence_id]
        complete_name = _expected_output_member(condition, sequence_id, "complete.json",
                                                policy.output_prefix)
        complete_doc = _parse_json_bytes(snapshot[complete_name], complete_name)
        provenance_doc = complete_doc.get("provenance")
        if not isinstance(provenance_doc, dict):
            raise ValueError(f"Complete output lacks provenance: {complete_name}")
        provenance = provenance_doc.get("gotrack_inference")
        identity = _check_inference_identity(
            provenance, condition=condition, sequence_id=sequence_id,
            binding=binding_by_sequence[sequence_id],
            checkpoint_sha256=policy.checkpoint_sha256,
        )
        generation = generations[sequence_id]
        for key in ("video_sha256", "mesh_sha256"):
            _expect(identity[key] == generation.get(key),
                    f"Sealed output and generation disagree on {key}: {sequence_id}")
        if identity["generation_sha256"] != _sha256(generation_bytes[sequence_id]):
            raise ValueError(f"Sealed output generation identity mismatch: {sequence_id}")
        preliminary_identities.setdefault(sequence_id, identity)

    # All sealed members and generation pins are authenticated at this point.
    # Only now read pose-reference labels.
    references, reference_hashes = _load_references(
        refs_root, diagnostic_raw, binding_by_sequence, preliminary_identities)
    outputs, output_identities, consumed_hashes = _load_sealed_outputs(
        snapshot, binding_by_sequence, references, policy=policy)
    prefix_results = None
    if experiment_policy == EXPANDED_POLICY_ID:
        prefix_results, prefix_hashes = _load_causality_prefix_outputs(
            snapshot, binding_by_sequence, references, outputs,
            checkpoint_sha256=policy.checkpoint_sha256)
        consumed_hashes.update(prefix_hashes)
    for sequence_id, identity in output_identities.items():
        for key in ("generation_sha256", "input_sha256", "video_sha256", "mesh_sha256"):
            _expect(identity[key] == preliminary_identities[sequence_id][key],
                    f"Authenticated output changed while parsing: {sequence_id}/{key}")

    condition_results: dict[str, dict[str, dict[str, Any]]] = {"frozen": {}, "adapted": {}}
    for condition in ("frozen", "adapted"):
        for sequence_id, spec in SEQUENCE_SPECS.items():
            output = outputs[condition][sequence_id]
            condition_results[condition][sequence_id] = evaluate_sequence(
                sequence_id=sequence_id, split=spec["split"],
                object_id=spec["physical_object_id"],
                reference_frames=references[sequence_id],
                complete_frames=output["complete"], candidate_frames=output["candidates"],
                condition=condition,
                requested_frame_ids=policy.requested_frame_ids,
            )
    condition_summaries = _summarize_conditions(
        condition_results, requested_frame_ids=policy.requested_frame_ids)
    report = {
        "schema_version": 1,
        "kind": _EVALUATOR_KIND,
        "status": "development_synthetic_pose_agreement_only",
        "claim_scope": {
            "primary_metric": "translation_error_m on emitted cameraFromObject poses",
            "reference_kind": "synthetic_reference_geometry",
            "orientation_primary": "unmeasured because all three references are observationally ambiguous",
            "training_sequence_is_resubstitution": True,
            "validation_sequence_is_already_used_development_validation": True,
            "independent_object_count": 2,
            "dependent_sequence_count": 3,
            "frame_independent_confidence_intervals": False,
            "quality_promotion": False,
        },
        "input_hash_receipt": {
            "sealed_archive_sha256": policy.archive_sha256,
            "sealed_manifest_sha256": _sha256(snapshot["artifact-manifest.json"]),
            "authenticated_sealed_member_count_including_manifest": len(snapshot),
            "consumed_sealed_members_sha256": {
                "design.json": _sha256(snapshot["design.json"]),
                "execution-bindings.json": _sha256(snapshot["execution-bindings.json"]),
                "paired-report.json": _sha256(snapshot["paired-report.json"]),
                **dict(sorted(consumed_hashes.items())),
            },
            "reference_inputs": reference_hashes,
            "diagnostic_sha256": policy.diagnostic_sha256,
        },
        "coverage": ({
            "quality_sequences": list(SEQUENCE_SPECS),
            "conditions": ["frozen", "adapted"],
            "requested_frames_per_sequence_condition": len(REQUESTED_FRAME_IDS),
            "requested_frames_per_condition": len(SEQUENCE_SPECS) * len(REQUESTED_FRAME_IDS),
            "frames_8_through_23": "not_quality_samples; not failures",
            "four_frame_prefix_runs": "causality-only; excluded from quality samples",
        } if experiment_policy == "legacy8" else {
            "quality_sequences": list(SEQUENCE_SPECS),
            "conditions": ["frozen", "adapted"],
            "requested_frame_ids": list(policy.requested_frame_ids),
            "requested_frames_per_sequence_condition": len(policy.requested_frame_ids),
            "requested_frames_per_condition": len(SEQUENCE_SPECS) * len(policy.requested_frame_ids),
            "full_video_frames": "all requested quality samples; failures and missing rows remain in the denominator",
            "four_frame_prefix_runs": "causality-only; excluded from full-video quality samples",
            "training_split": "resubstitution; frames 17-minus-25 and 17-plus-25 are dependent views of object 17",
            "used_validation_split": "previously inspected development validation; object 27",
        }),
        "conditions": condition_summaries,
        "paired_comparison": {
            "metric": "adapted translation error minus frozen translation error in metres",
            "paired_on": "common emitted-pose and authenticated-reference intersection within each sequence",
            "availability_reported_for_both_conditions": True,
            "sequences": _compare_conditions(
                condition_results, requested_frame_ids=policy.requested_frame_ids),
        },
        "unsupported_metrics": {
            "primary_rotation_error": "unmeasured_observationally_ambiguous_reference",
            "supplementary_synthetic_reference_angles": "not computed",
            "surface_attachment": "unmeasured",
            "projection_iou": "unmeasured",
            "occlusion_return": "unmeasured",
            "quality_thresholds": "none added",
        },
    }
    if prefix_results is not None:
        report["causality_prefix_runs"] = prefix_results
    return report


def _check_output_destination(path: str | Path, roots: Sequence[Path]) -> Path:
    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Output path must be new and must not overwrite an existing file")
    parent = target.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("Output parent must be an existing non-symlink directory")
    resolved_parent = parent.resolve(strict=True)
    if target.name in ("", ".", ".."):
        raise ValueError("Output filename is invalid")
    resolved_target = resolved_parent / target.name
    for root in roots:
        if resolved_target == root or resolved_target.is_relative_to(root):
            raise ValueError("Output path cannot be inside a sealed or reference input root")
    return resolved_target


def write_new_report(path: str | Path, report: Mapping[str, Any], *, input_roots: Sequence[Path]) -> str:
    target = _check_output_destination(path, input_roots)
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return _sha256(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--sealed-manifest-sha256", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--experiment-policy", choices=("legacy8", "expanded24"),
                        default="legacy8")
    parser.add_argument("--experiment-policy-json", type=Path)
    parser.add_argument("--experiment-policy-sha256")
    parser.add_argument("--sealed-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        sealed_root = _safe_root(args.sealed_root, "sealed artifact")
        report = evaluate_postseal(
            sealed_root=sealed_root,
            sealed_manifest_sha256=args.sealed_manifest_sha256,
            reference_root=args.reference_root,
            diagnostic_path=args.diagnostic,
            experiment_policy=args.experiment_policy,
            experiment_policy_json=args.experiment_policy_json,
            experiment_policy_sha256=args.experiment_policy_sha256,
            sealed_archive=args.sealed_archive,
        )
        reference_root = _safe_root(args.reference_root, "reference")
        output_sha256 = write_new_report(
            args.output, report, input_roots=(sealed_root, reference_root))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"post-seal pose evaluation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "written", "output": str(args.output),
                      "report_sha256": output_sha256}, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
