"""Authenticated rectified HOT3D capture resources and causal frame access.

The module-level imports are deliberately from the Python standard library.
NumPy, the reviewed camera adapter, and OpenCV are loaded only by the explicit
numerical or conversion/read functions that need them.

This module describes and verifies a source contract. It does not load tracking
models, choose prompts, inspect evaluation labels, launch workers, or publish
results. Conversion is an explicit private preprocessing operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any


SOURCE_KIND = "rectified_image_sequence_v1"
COORDINATE_MODE = "integer_centers_v1"
INPUT_SCHEMA_VERSION = 2
TABLE_SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_TABLE_BYTES = 2 * 1024 * 1024
MAX_INVENTORY_BYTES = 2 * 1024 * 1024
MAX_CALIBRATION_BYTES = 1024 * 1024
MAX_ENCODED_IMAGE_BYTES = 16 * 1024 * 1024
MAX_SEQUENCE_FRAMES = 150
MAX_RESOURCE_CACHE_FRAMES = 1
MAX_CONVERSION_TRANSIENT_BYTES = 128 * 1024 * 1024
MAX_SINGLE_WARP_BYTES = 16 * 1024 * 1024
MAX_RAW_RGB_BYTES = 5_947_392
SOURCE_INVENTORY_RELATIVE = "source-captures.json"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_STREAM_SUFFIX_RE = re.compile(r"(?:^|/)\d+\.image_([A-Za-z0-9_.-]+)\.[A-Za-z0-9]+$")
_FAILURE_REASONS = frozenset({
    "raw_missing", "raw_decode_failed", "raw_dimension_mismatch",
    "calibration_invalid", "warp_failed", "artifact_write_failed",
})

OUTPUT_CAMERA: dict[str, Any] = {
    "width": 720,
    "height": 720,
    "fx": 360.0,
    "fy": 360.0,
    "cx": 359.5,
    "cy": 359.5,
    "rotation_output_from_source": [
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ],
    "pixel_center_convention": "integer-centers; edges[-0.5,size-0.5]",
}
OUTPUT_INTRINSICS: tuple[tuple[float, float, float], ...] = (
    (360.0, 0.0, 359.5),
    (0.0, 360.0, 359.5),
    (0.0, 0.0, 1.0),
)
_INPUT_FIELDS = frozenset({
    "schema_version", "object", "object_id", "units", "asset", "asset_receipt",
    "native_resolution", "intrinsics", "rgb_source", "source_hashes", "clock",
    "setup_frame_id", "frame_ids", "timeline", "source_gaps", "selection",
})
_INPUT_REQUIRED_FIELDS = _INPUT_FIELDS - {"source_gaps"}
_TABLE_FIELDS = frozenset({
    "schema_version", "source_kind", "source_revision", "clip_id", "stream_id",
    "archive_sha256", "source_inventory_sha256", "output_camera",
    "output_camera_sha256", "coordinate_mode", "converter", "coverage_complete",
    "planned_count", "rows",
})
_CONVERTER_FIELDS = frozenset({"source_closure", "runtime_decoder"})
_ROW_FIELDS = frozenset({
    "frame_id", "source_frame_id", "source_timestamp_ns", "timestamp_s", "role",
    "raw_rgb", "decoded_raw_rgb_pixel_sha256", "raw_calibration",
    "raw_camera_metadata_sha256", "calibration_values_sha256", "warp_sha256",
    "map_sha256", "geometric_valid_sha256", "domain_valid_sha256",
    "sampling_valid_sha256", "output_rgb", "sampling_valid", "conversion_state",
    "failure_reason", "conversion_metadata", "conversion_metadata_sha256",
    "capture_row_sha256",
})
_RAW_RESOURCE_FIELDS = frozenset({"path", "sha256", "byte_count"})
_IMAGE_RESOURCE_FIELDS = frozenset({"path", "file_sha256", "byte_count", "pixel_sha256"})
_METADATA_RESOURCE_FIELDS = frozenset({"path", "file_sha256", "byte_count"})
_VERIFIED_PLAN_TOKEN = object()
_CAPTURE_PLAN_TOKEN = object()
_SINGLE_FRAME_TOKEN = object()


class CaptureIntegrityError(ValueError):
    """A capture plan, row, or authenticated resource changed or disagrees."""

    def __init__(self, message: str, *, ordinal: int | None = None,
                 source_frame_id: int | None = None, reason: str = "capture_integrity_failed"):
        super().__init__(message)
        self.ordinal = ordinal
        self.source_frame_id = source_frame_id
        self.reason = reason


class CaptureDecodeError(ValueError):
    """A pinned encoded capture resource could not be decoded as declared."""

    def __init__(self, message: str, *, ordinal: int, source_frame_id: int,
                 reason: str = "capture_decode_failed"):
        super().__init__(message)
        self.ordinal = ordinal
        self.source_frame_id = source_frame_id
        self.reason = reason


class CaptureUnavailable(ValueError):
    """The converter recorded a typed unavailable source row."""

    def __init__(self, message: str, *, ordinal: int, source_frame_id: int,
                 reason: str):
        if reason not in _FAILURE_REASONS:
            raise ValueError("unknown capture unavailability reason")
        super().__init__(message)
        self.ordinal = ordinal
        self.source_frame_id = source_frame_id
        self.reason = reason


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """Pure, fully joined sequence plan, detached from the source byte buffers."""

    manifest: Mapping[str, Any]
    table: Mapping[str, Any]
    timing: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    input_manifest_sha256: str
    capture_table_sha256: str
    selection: Mapping[str, Any]
    _plan_token: Any = field(default=None, repr=False, compare=False)
    _semantic_sha256: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def frame_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class VerifiedCapturePlan:
    """A plan whose declared bundle resources were streamed and authenticated."""

    plan: CapturePlan
    bundle: Path
    verified_resources: Mapping[str, str]
    _verification_token: Any = field(default=None, repr=False, compare=False)
    _verification_sha256: str | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.plan.manifest

    @property
    def table(self) -> Mapping[str, Any]:
        return self.plan.table

    @property
    def timing(self) -> Mapping[str, Any]:
        return self.plan.timing

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return self.plan.rows

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.plan.selection

    @property
    def input_manifest_sha256(self) -> str:
        return self.plan.input_manifest_sha256

    @property
    def capture_table_sha256(self) -> str:
        return self.plan.capture_table_sha256


@dataclass(frozen=True, slots=True)
class DecodedCapture:
    """One shared immutable decoded RGB/validity record from a pinned row."""

    key: Any
    source_frame_id: int
    rgb: Any
    sampling_valid: Any
    intrinsics: tuple[tuple[float, float, float], ...]
    capture_row_sha256: str
    capture_table_sha256: str
    rgb_pixel_sha256: str
    sampling_valid_sha256: str
    warp_sha256: str
    calibration_sha256: str

    @property
    def capture_binding(self) -> Mapping[str, str]:
        return MappingProxyType({
            "capture_table_sha256": self.capture_table_sha256,
            "capture_row_sha256": self.capture_row_sha256,
            "rgb_pixel_sha256": self.rgb_pixel_sha256,
            "sampling_valid_sha256": self.sampling_valid_sha256,
            "warp_sha256": self.warp_sha256,
            "calibration_sha256": self.calibration_sha256,
        })


@dataclass(frozen=True, slots=True)
class SingleFrameCapability:
    """A plan-bound capability for one canonical row only."""

    input_manifest_sha256: str
    capture_table_sha256: str
    ordinal: int
    source_frame_id: int
    capture_row_sha256: str
    _capability_token: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ConversionReceipt:
    """Immutable summary of a private conversion table and its copied inventory."""

    output_root: Path
    capture_table_path: Path
    capture_table_sha256: str
    source_inventory_path: Path
    source_inventory_sha256: str
    planned_count: int
    available_count: int
    unavailable_count: int
    rows: tuple[Mapping[str, Any], ...]
    table: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StageResourceDescriptor:
    """Authenticated sidecar metadata; artifact bytes are hashed, never loaded."""

    sidecar_path: Path
    sidecar_sha256: str
    resource_key: str
    resource_kind: str
    artifact_path: Path
    artifact_sha256: str
    artifact_byte_count: int
    recipe: Mapping[str, Any]
    source_closure: Mapping[str, str]
    descriptor: Mapping[str, Any]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError("value is not canonical finite JSON") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _valid_digest(value: Any, name: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.upper()


def _contract_digest(value: Any, name: str) -> str:
    """Validate an uppercase digest used by a newly-authored contract."""

    digest = _valid_digest(value, name)
    if value != digest:
        raise ValueError(f"{name} must use uppercase SHA-256 hex")
    return digest


def _json_no_constants(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_tree(value: Any, path: str = "JSON", depth: int = 0) -> None:
    if depth > 64:
        raise ValueError(f"{path} exceeds the maximum JSON nesting depth")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} has a non-string key")
            _finite_tree(child, f"{path}.{key}", depth + 1)
        return
    raise ValueError(f"{path} contains a non-JSON value")


def _parse_json_document(raw: bytes, *, limit: int, name: str) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > limit:
        raise ValueError(f"{name} must be immutable bytes no larger than {limit} bytes")
    try:
        document = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_json_no_constants,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    _finite_tree(document, name)
    return document


def _strict_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        raise ValueError(f"{name} fields mismatch (missing={missing}, unknown={unknown})")


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _relative_path(value: Any, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty normalized relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (posix.is_absolute() or windows.is_absolute() or windows.drive or
            any(part in ("", ".", "..") for part in value.split("/")) or
            ":" in value):
        raise ValueError(f"{name} must not be absolute, aliased, or traversing")
    forbidden = {"evaluator", "evaluation", "eval", "annotations", "ground-truth",
                 "ground_truth", "reference-poses", "reference_poses", "labels"}
    if any(part.casefold() in forbidden for part in value.split("/")):
        raise ValueError(f"{name} cannot reference evaluator or annotation resources")
    return value


def _parse_input(raw_input_bytes: bytes) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if type(raw_input_bytes) is not bytes or len(raw_input_bytes) > MAX_INPUT_BYTES:
        raise ValueError("input manifest must be immutable bytes no larger than 2 MiB")
    document = _parse_json_document(raw_input_bytes, limit=MAX_INPUT_BYTES,
                                    name="input manifest")
    unknown_fields = set(document) - _INPUT_FIELDS
    missing_fields = _INPUT_REQUIRED_FIELDS - set(document)
    if unknown_fields or missing_fields:
        raise ValueError(
            "input manifest fields mismatch "
            f"(missing={sorted(missing_fields)}, unknown={sorted(unknown_fields)})"
        )
    if "video" in document:
        raise ValueError("sequence source cannot be combined with video")
    if type(document["schema_version"]) is not int or document["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ValueError("capture input must use schema_version 2")
    if not isinstance(document.get("object"), str) or not document["object"].strip():
        raise ValueError("object must be a nonempty string")
    _nonnegative_int(document.get("object_id"), "object_id")
    if document.get("units") != "metres":
        raise ValueError("capture object units must be metres")
    if document.get("native_resolution") != [720, 720] or any(
            type(item) is not int for item in document.get("native_resolution", [])):
        raise ValueError("capture native_resolution must be exactly [720, 720]")
    k = document.get("intrinsics")
    if not isinstance(k, list) or len(k) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in k):
        raise ValueError("intrinsics must be a 3x3 matrix")
    for ri, row in enumerate(k):
        for ci, scalar in enumerate(row):
            _finite_real(scalar, f"intrinsics[{ri}][{ci}]")
    if tuple(tuple(float(x) for x in row) for row in k) != OUTPUT_INTRINSICS:
        raise ValueError("capture input intrinsics do not match the fixed output camera")
    rgb_source = document.get("rgb_source")
    _strict_keys(rgb_source, frozenset({"kind", "table"}), "rgb_source")
    if rgb_source["kind"] != SOURCE_KIND:
        raise ValueError("unsupported rgb_source kind")
    table_path = _relative_path(rgb_source["table"], "rgb_source.table")
    if "controlled_initial_pose" in document:
        raise ValueError("capture input cannot contain a controlled pose")

    source_hashes = document.get("source_hashes")
    if not isinstance(source_hashes, dict) or len(source_hashes) != 3:
        raise ValueError("sequence source_hashes must contain exactly asset, receipt, and table")
    asset = _relative_path(document.get("asset"), "asset")
    asset_receipt = _relative_path(document.get("asset_receipt"), "asset_receipt")
    if set(source_hashes) != {asset, asset_receipt, table_path}:
        raise ValueError("source_hashes must bind exactly asset, asset_receipt, and capture table")
    for path, digest in source_hashes.items():
        _relative_path(path, "source_hashes path")
        _contract_digest(digest, f"source_hashes[{path!r}]")
    if len({path.casefold() for path in source_hashes}) != 3:
        raise ValueError("source_hashes contains case-insensitive path aliases")

    clock = document.get("clock")
    if not isinstance(clock, dict):
        raise ValueError("capture input requires a clock object")
    clock_required = frozenset({
        "mode", "units", "source_units", "time_base", "timestamp_source", "nominal_fps",
    })
    if not clock_required.issubset(clock) or set(clock) - clock_required - {"timeline_span_seconds"}:
        raise ValueError("capture clock fields do not match the physical timing allowlist")
    if clock.get("mode") != "capture_seconds_v1" or clock.get("units") != "seconds" or clock.get("source_units") != "nanoseconds":
        raise ValueError("capture clock must use the authenticated nanosecond-to-second policy")
    if clock.get("time_base") != {"num": 1, "den": 1_000_000_000}:
        raise ValueError("capture clock time_base must be exactly 1/1000000000")
    if not isinstance(clock.get("timestamp_source"), str) or not clock["timestamp_source"].startswith(
            "HOT3D source-captures.json:source_timestamp_ns;stream"):
        raise ValueError("timestamp_source must identify the HOT3D source timestamp and stream")
    if _finite_real(clock.get("nominal_fps"), "nominal_fps") != 30.0:
        raise ValueError("nominal_fps metadata must be the frozen 30 FPS declaration")
    timeline = document.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("capture input requires a nonempty timeline")
    timeline_fields = frozenset({"frame_id", "source_frame_id", "source_timestamp", "timestamp_s", "role"})
    for ordinal, row in enumerate(timeline):
        _strict_keys(row, timeline_fields, f"input timeline row {ordinal}")

    # This call is deliberately before bundle path resolution or table access.
    from .quality_time import PHYSICAL, parse_timing_manifest
    timing = parse_timing_manifest(raw_input_bytes)
    if timing["clock_mode"] != PHYSICAL:
        raise ValueError("rectified capture sequences require the physical capture clock")
    return document, timing


def _expected_camera() -> dict[str, Any]:
    # Copy the module constant so callers cannot mutate the canonical binding.
    return json.loads(json.dumps(OUTPUT_CAMERA))


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _deeply_frozen_json(value: Any, depth: int = 0) -> bool:
    """Check the exact immutable representation emitted by _freeze."""

    if depth > 64:
        return False
    if type(value) is MappingProxyType:
        return all(
            type(key) is str and _deeply_frozen_json(child, depth + 1)
            for key, child in value.items()
        )
    if type(value) is tuple:
        return all(_deeply_frozen_json(child, depth + 1) for child in value)
    if value is None or type(value) in (str, bool, int):
        return True
    return type(value) is float and math.isfinite(value)


def _timing_semantic_document(value: Any) -> Any:
    """Encode quality_time's frozen records and integer-key maps canonically."""

    from .quality_time import FrameKey, LEGACY, PHYSICAL, TimePolicy

    def encode(node: Any, depth: int) -> Any:
        if depth > 64:
            raise ValueError("timing content exceeds the maximum semantic depth")
        if type(node) is MappingProxyType:
            entries: list[tuple[tuple[str, Any], list[Any]]] = []
            for key, child in node.items():
                if type(key) is str:
                    key_identity = ("str", key)
                    key_value = ["str", key]
                elif type(key) is int:
                    key_identity = ("int", key)
                    key_value = ["int", key]
                else:
                    raise ValueError("timing mappings require immutable string or integer keys")
                entries.append((key_identity, [key_value, encode(child, depth + 1)]))
            entries.sort(key=lambda item: item[0])
            return {"mapping": [entry for _, entry in entries]}
        if type(node) is tuple:
            return {"tuple": [encode(child, depth + 1) for child in node]}
        if type(node) is TimePolicy:
            values = (
                node.mode, node.max_angular_rate_deg_s,
                node.max_translation_rate_m_s, node.private_pose_memory_s,
            )
            if (type(node.mode) is not str or node.mode not in (LEGACY, PHYSICAL) or
                    any(type(value) is not float or not math.isfinite(value) or value <= 0.0
                        for value in values[1:])):
                raise ValueError("timing policy record has invalid immutable fields")
            return {"TimePolicy": list(values)}
        if type(node) is FrameKey:
            if (type(node.frame_id) is not int or node.frame_id < 0 or
                    type(node.clock_mode) is not str or node.clock_mode not in (LEGACY, PHYSICAL) or
                    (node.timestamp_s is not None and
                     (type(node.timestamp_s) is not float or not math.isfinite(node.timestamp_s)))):
                raise ValueError("timing frame key has invalid immutable fields")
            if (node.clock_mode == LEGACY and node.timestamp_s is not None) or (
                    node.clock_mode == PHYSICAL and node.timestamp_s is None):
                raise ValueError("timing frame key fields disagree")
            return {"FrameKey": [node.frame_id, node.timestamp_s, node.clock_mode]}
        if node is None or type(node) in (str, bool, int):
            return node
        if type(node) is float and math.isfinite(node):
            return node
        raise ValueError("timing content is not deeply immutable")

    return encode(value, 0)


def _capture_plan_semantic_sha256(plan: CapturePlan) -> str:
    return _canonical_digest({
        "manifest": plan.manifest,
        "table": plan.table,
        "timing": _timing_semantic_document(plan.timing),
        "rows": plan.rows,
        "selection": plan.selection,
        "input_manifest_sha256": plan.input_manifest_sha256,
        "capture_table_sha256": plan.capture_table_sha256,
    })


def _validate_capture_plan_record(plan: Any) -> CapturePlan:
    """Reject forged, replaced, or mutable plan records without I/O."""

    if type(plan) is not CapturePlan or plan._plan_token is not _CAPTURE_PLAN_TOKEN:
        raise CaptureIntegrityError("capture plan is not an unchanged issued CapturePlan")
    for name in ("manifest", "table", "selection"):
        value = getattr(plan, name)
        if type(value) is not MappingProxyType or not _deeply_frozen_json(value):
            raise CaptureIntegrityError(f"capture plan {name} is not deeply immutable")
    if type(plan.timing) is not MappingProxyType:
        raise CaptureIntegrityError("capture plan timing is not deeply immutable")
    if (type(plan.rows) is not tuple or
            not _deeply_frozen_json(plan.rows)):
        raise CaptureIntegrityError("capture plan rows are not deeply immutable")
    try:
        _contract_digest(plan.input_manifest_sha256, "input_manifest_sha256")
        _contract_digest(plan.capture_table_sha256, "capture_table_sha256")
        semantic_sha256 = _capture_plan_semantic_sha256(plan)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CaptureIntegrityError("capture plan semantic fields are invalid") from exc
    if (type(plan._semantic_sha256) is not str or
            plan._semantic_sha256 != semantic_sha256):
        raise CaptureIntegrityError("capture plan semantic content seal mismatch")
    return plan


def _normalized_bundle_identity(bundle: Any) -> str:
    if type(bundle) is not type(Path()) or not bundle.is_absolute():
        raise CaptureIntegrityError("verified bundle path is not normalized and absolute")
    return os.path.normcase(os.path.abspath(os.fspath(bundle)))


def _verified_capture_plan_semantic_sha256(plan: VerifiedCapturePlan) -> str:
    return _canonical_digest({
        "plan_semantic_sha256": plan.plan._semantic_sha256,
        "bundle_identity": _normalized_bundle_identity(plan.bundle),
        "verified_resources": plan.verified_resources,
    })


def _validate_verified_capture_plan_record(plan: Any) -> VerifiedCapturePlan:
    """Validate the sealed plan, bundle identity, and immutable resource map."""

    if (type(plan) is not VerifiedCapturePlan or
            plan._verification_token is not _VERIFIED_PLAN_TOKEN):
        raise CaptureIntegrityError("capture preflight result is not an unchanged issued VerifiedCapturePlan")
    _validate_capture_plan_record(plan.plan)
    if (type(plan.verified_resources) is not MappingProxyType or
            not _deeply_frozen_json(plan.verified_resources)):
        raise CaptureIntegrityError("verified resource map is not deeply immutable")
    try:
        verified_sha256 = _verified_capture_plan_semantic_sha256(plan)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CaptureIntegrityError("verified capture plan semantic fields are invalid") from exc
    if (type(plan._verification_sha256) is not str or
            plan._verification_sha256 != verified_sha256):
        raise CaptureIntegrityError("verified capture plan semantic content seal mismatch")
    return plan


def _check_resource_object(resource: Any, *, kind: str, name: str,
                           unavailable_allowed: bool) -> dict[str, Any] | None:
    if resource is None:
        if unavailable_allowed and kind != "raw":
            return None
        raise ValueError(f"{name} must preserve its declared source resource identity")
    fields = {
        "raw": _RAW_RESOURCE_FIELDS,
        "image": _IMAGE_RESOURCE_FIELDS,
        "metadata": _METADATA_RESOURCE_FIELDS,
    }[kind]
    _strict_keys(resource, fields, name)
    path = resource.get("path")
    digest_key = "sha256" if kind == "raw" else "file_sha256"
    digest = _contract_digest(resource.get(digest_key), f"{name}.{digest_key}")
    byte_count = resource.get("byte_count")
    if path is None:
        if not unavailable_allowed or byte_count is not None:
            raise ValueError(f"{name} null path is allowed only for unavailable raw source bytes")
    else:
        _relative_path(path, f"{name}.path")
        _positive_int(byte_count, f"{name}.byte_count")
    if kind == "image" and (path is None or resource.get("pixel_sha256") is None):
        raise ValueError(f"{name} must bind an artifact path and pixel digest")
    if kind == "image":
        _contract_digest(resource.get("pixel_sha256"), f"{name}.pixel_sha256")
    copy = dict(resource)
    copy[digest_key] = digest
    if kind == "image":
        copy["pixel_sha256"] = _contract_digest(resource["pixel_sha256"], f"{name}.pixel_sha256")
    return copy


def _validate_inventory(raw_inventory: bytes, expected_sha256: str) -> tuple[dict[str, Any], str, tuple[dict[str, Any], ...]]:
    _contract_digest(expected_sha256, "expected_inventory_sha256")
    if _sha256_bytes(raw_inventory) != expected_sha256.upper():
        raise CaptureIntegrityError("source inventory bytes do not match the expected SHA-256")
    inventory = _parse_json_document(raw_inventory, limit=MAX_INVENTORY_BYTES,
                                     name="source inventory")
    _strict_keys(inventory, frozenset({
        "schema_version", "source_revision", "clip_id", "archive_sha256", "source_units", "frames",
        "inference_ready", "limitation",
    }), "source inventory")
    # These two fields are present in the pinned acquisition inventory. They
    # describe producer readiness only; neither is copied into inference rows.
    if inventory.get("inference_ready") is not False:
        raise ValueError("source inventory inference_ready must be false")
    limitation = inventory.get("limitation")
    if not isinstance(limitation, str) or not limitation.strip() or len(limitation) > 512:
        raise ValueError("source inventory limitation must be a nonempty string of at most 512 characters")
    if type(inventory.get("schema_version")) is not int or inventory["schema_version"] != 1:
        raise ValueError("source inventory must use schema_version 1")
    if not isinstance(inventory.get("source_revision"), str) or not inventory["source_revision"].strip():
        raise ValueError("source_revision must be a nonempty string")
    _nonnegative_int(inventory.get("clip_id"), "clip_id")
    _valid_digest(inventory.get("archive_sha256"), "archive_sha256")
    if inventory.get("source_units") != "nanoseconds":
        raise ValueError("source inventory must use nanosecond timestamps")
    frames = inventory.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_SEQUENCE_FRAMES:
        raise ValueError("source inventory must contain 1..150 frame rows")
    frame_fields = frozenset({
        "source_frame_id", "calibration_path", "calibration_sha256",
        "raw_camera_metadata_sha256", "rgb_path", "rgb_sha256", "source_timestamp_ns",
    })
    validated: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    last_id = -1
    last_timestamp = -1
    paths: dict[str, tuple[str, str]] = {}
    for ordinal, entry in enumerate(frames):
        _strict_keys(entry, frame_fields, f"source inventory frame {ordinal}")
        source_id = _nonnegative_int(entry["source_frame_id"], f"frames[{ordinal}].source_frame_id")
        timestamp_ns = _nonnegative_int(entry["source_timestamp_ns"], f"frames[{ordinal}].source_timestamp_ns")
        if source_id in seen_ids or source_id <= last_id or timestamp_ns <= last_timestamp:
            raise ValueError("source inventory IDs and timestamps must be unique and chronological")
        seen_ids.add(source_id)
        last_id, last_timestamp = source_id, timestamp_ns
        rgb_path = _relative_path(entry["rgb_path"], f"frames[{ordinal}].rgb_path")
        calibration_path = _relative_path(entry["calibration_path"], f"frames[{ordinal}].calibration_path")
        if rgb_path == calibration_path:
            raise ValueError("one source inventory row cannot use one path for RGB and calibration")
        for candidate, resource_kind in ((rgb_path, "rgb"), (calibration_path, "calibration")):
            folded = candidate.casefold()
            old = paths.get(folded)
            if old is not None and not (
                    old == (candidate, "calibration") and resource_kind == "calibration"):
                raise ValueError("source inventory contains case-insensitive path aliases")
            paths[folded] = (candidate, resource_kind)
        validated.append({
            "frame_id": ordinal,
            "source_frame_id": source_id,
            "source_timestamp_ns": timestamp_ns,
            "timestamp_s": float(Fraction(timestamp_ns, 1_000_000_000)),
            "role": "scored",
            "rgb_path": rgb_path,
            "rgb_sha256": _valid_digest(entry["rgb_sha256"], f"frames[{ordinal}].rgb_sha256"),
            "calibration_path": calibration_path,
            "calibration_sha256": _valid_digest(entry["calibration_sha256"], f"frames[{ordinal}].calibration_sha256"),
            "raw_camera_metadata_sha256": _valid_digest(
                entry["raw_camera_metadata_sha256"], f"frames[{ordinal}].raw_camera_metadata_sha256"),
        })
    return inventory, expected_sha256.upper(), tuple(validated)


def _validate_capture_table(raw_table_bytes: bytes, manifest: Mapping[str, Any],
                            timing: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    table = _parse_json_document(raw_table_bytes, limit=MAX_TABLE_BYTES, name="capture table")
    _strict_keys(table, _TABLE_FIELDS, "capture table")
    if type(table.get("schema_version")) is not int or table["schema_version"] != TABLE_SCHEMA_VERSION:
        raise ValueError("capture table must use schema_version 1")
    if table.get("source_kind") != SOURCE_KIND:
        raise ValueError("capture table source_kind mismatch")
    if not isinstance(table.get("source_revision"), str) or not table["source_revision"].strip():
        raise ValueError("capture table source_revision must be nonempty")
    _nonnegative_int(table.get("clip_id"), "capture table clip_id")
    stream_id = table.get("stream_id")
    if not isinstance(stream_id, str) or not stream_id.strip():
        raise ValueError("capture table stream_id must be a nonempty string")
    _contract_digest(table.get("archive_sha256"), "capture table archive_sha256")
    _contract_digest(table.get("source_inventory_sha256"), "capture table source_inventory_sha256")
    output = table.get("output_camera")
    _strict_keys(output, frozenset(OUTPUT_CAMERA), "output_camera")
    if output != OUTPUT_CAMERA:
        raise ValueError("capture table output_camera differs from the fixed camera")
    output_sha = _contract_digest(table.get("output_camera_sha256"), "output_camera_sha256")
    if output_sha != _canonical_digest(OUTPUT_CAMERA):
        raise ValueError("output_camera_sha256 does not bind the fixed output camera")
    if table.get("coordinate_mode") != COORDINATE_MODE:
        raise ValueError("capture table coordinate_mode mismatch")
    converter = table.get("converter")
    _strict_keys(converter, _CONVERTER_FIELDS, "converter")
    source_closure = converter.get("source_closure")
    if not isinstance(source_closure, dict) or set(source_closure) != {
            "bench/quality_capture.py", "bench/quality_camera.py", "bench/quality_time.py"}:
        raise ValueError("converter source_closure must pin the exact capture/camera/time source set")
    for source_path, digest in source_closure.items():
        _relative_path(source_path, "converter source closure path")
        _contract_digest(digest, f"converter.source_closure[{source_path!r}]")
    if len({path.casefold() for path in source_closure}) != len(source_closure):
        raise ValueError("converter source closure contains path aliases")
    if not isinstance(converter.get("runtime_decoder"), str) or not converter["runtime_decoder"].strip():
        raise ValueError("converter runtime_decoder must identify the pinned decoder")
    if table.get("coverage_complete") is not True:
        raise ValueError("capture conversion table must declare complete planned coverage")
    count = _positive_int(table.get("planned_count"), "planned_count", maximum=MAX_SEQUENCE_FRAMES)
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != count or count != len(manifest["timeline"]):
        raise ValueError("capture table rows, planned_count, and input timeline must match")
    expected_stream = manifest["clock"]["timestamp_source"].split(";stream", 1)[-1]
    if expected_stream != stream_id:
        raise ValueError("capture table stream_id disagrees with the clock timestamp source")

    table_rows: list[dict[str, Any]] = []
    seen_paths: dict[str, tuple[str, str, str]] = {}
    for ordinal, (row, timeline_row) in enumerate(zip(rows, manifest["timeline"])):
        _strict_keys(row, _ROW_FIELDS, f"capture table row {ordinal}")
        frame_id = _nonnegative_int(row.get("frame_id"), f"rows[{ordinal}].frame_id")
        source_id = _nonnegative_int(row.get("source_frame_id"), f"rows[{ordinal}].source_frame_id")
        source_ns = _nonnegative_int(row.get("source_timestamp_ns"), f"rows[{ordinal}].source_timestamp_ns")
        stamp = _finite_real(row.get("timestamp_s"), f"rows[{ordinal}].timestamp_s")
        role = row.get("role")
        if role not in ("setup", "scored"):
            raise ValueError(f"rows[{ordinal}].role must be setup or scored")
        if (frame_id != ordinal or frame_id != timeline_row["frame_id"] or
                source_id != timeline_row["source_frame_id"] or
                source_ns != timeline_row["source_timestamp"] or
                stamp != timeline_row["timestamp_s"] or role != timeline_row["role"]):
            raise ValueError(f"capture row {ordinal} does not exactly join its input timeline")
        normalized: dict[str, Any] = dict(row)
        normalized["frame_id"] = frame_id
        normalized["source_frame_id"] = source_id
        normalized["source_timestamp_ns"] = source_ns
        normalized["timestamp_s"] = stamp
        normalized["raw_rgb"] = _check_resource_object(
            row.get("raw_rgb"), kind="raw", name=f"rows[{ordinal}].raw_rgb", unavailable_allowed=True)
        normalized["raw_calibration"] = _check_resource_object(
            row.get("raw_calibration"), kind="raw", name=f"rows[{ordinal}].raw_calibration", unavailable_allowed=True)
        for digest_field in ("raw_camera_metadata_sha256",):
            normalized[digest_field] = _contract_digest(row.get(digest_field), f"rows[{ordinal}].{digest_field}")
        for digest_field in ("decoded_raw_rgb_pixel_sha256", "calibration_values_sha256", "warp_sha256",
                             "map_sha256", "geometric_valid_sha256", "domain_valid_sha256",
                             "sampling_valid_sha256", "conversion_metadata_sha256"):
            value = row.get(digest_field)
            normalized[digest_field] = None if value is None else _contract_digest(value, f"rows[{ordinal}].{digest_field}")
        for resource_field in ("output_rgb", "sampling_valid", "conversion_metadata"):
            resource_kind = "metadata" if resource_field == "conversion_metadata" else "image"
            normalized[resource_field] = _check_resource_object(
                row.get(resource_field), kind=resource_kind, name=f"rows[{ordinal}].{resource_field}",
                unavailable_allowed=True)
        state = row.get("conversion_state")
        failure = row.get("failure_reason")
        if state == "available":
            if failure is not None:
                raise ValueError(f"available row {ordinal} cannot have a failure_reason")
            required_digests = (
                "decoded_raw_rgb_pixel_sha256", "calibration_values_sha256", "warp_sha256",
                "map_sha256", "geometric_valid_sha256", "domain_valid_sha256",
                "sampling_valid_sha256", "conversion_metadata_sha256",
            )
            if any(normalized[field] is None for field in required_digests):
                raise ValueError(f"available row {ordinal} lacks a required digest")
            if any(normalized[field] is None for field in (
                    "raw_rgb", "raw_calibration", "output_rgb", "sampling_valid", "conversion_metadata")):
                raise ValueError(f"available row {ordinal} lacks an authenticated resource")
            if normalized["raw_rgb"]["path"] is None or normalized["raw_calibration"]["path"] is None:
                raise ValueError(f"available row {ordinal} cannot have missing raw inputs")
            if normalized["output_rgb"]["pixel_sha256"] != normalized["output_rgb"]["pixel_sha256"].upper():
                raise ValueError("pixel digest normalization failed")
            if normalized["conversion_metadata_sha256"] != normalized["conversion_metadata"]["file_sha256"]:
                raise ValueError(f"available row {ordinal} metadata digests disagree")
        elif state == "unavailable":
            if failure not in _FAILURE_REASONS:
                raise ValueError(f"unavailable row {ordinal} requires a typed failure_reason")
            if normalized["output_rgb"] is not None or normalized["sampling_valid"] is not None:
                raise ValueError(f"unavailable row {ordinal} cannot provide output image or validity")
        else:
            raise ValueError(f"row {ordinal} conversion_state must be available or unavailable")

        raw_hash = _contract_digest(row.get("capture_row_sha256"), f"rows[{ordinal}].capture_row_sha256")
        hash_document = dict(row)
        del hash_document["capture_row_sha256"]
        if _canonical_digest(hash_document) != raw_hash:
            raise ValueError(f"capture_row_sha256 mismatch for row {ordinal}")
        normalized["capture_row_sha256"] = raw_hash
        for resource_field in ("raw_rgb", "raw_calibration", "output_rgb", "sampling_valid", "conversion_metadata"):
            resource = normalized[resource_field]
            if resource is None or resource.get("path") is None:
                continue
            path = resource["path"]
            folded = path.casefold()
            identity = (
                path,
                resource_field,
                resource.get("sha256", resource.get("file_sha256", "")),
            )
            old = seen_paths.get(folded)
            repeat_calibration = (
                old is not None and resource_field == "raw_calibration" and
                old[0] == path and old[1] == "raw_calibration" and old[2] == identity[2]
            )
            if old is not None and not repeat_calibration:
                raise ValueError("capture table contains duplicate resource identities or path aliases")
            if old is None:
                seen_paths[folded] = identity
        table_rows.append(normalized)
    return table, tuple(table_rows)


def prepare_capture_plan(raw_input_bytes: bytes, raw_table_bytes: bytes) -> CapturePlan:
    """Validate and join sequence configuration, physical timing, and table bytes.

    This function is pure: it uses no filesystem, decoder, NumPy, camera, model,
    or output services. Structural timing validation happens before table
    parsing, which is the ordering used by the filesystem preflight as well.
    """

    manifest, timing = _parse_input(raw_input_bytes)
    if type(raw_table_bytes) is not bytes or len(raw_table_bytes) > MAX_TABLE_BYTES:
        raise ValueError("capture table must be immutable bytes no larger than 2 MiB")
    table, rows = _validate_capture_table(raw_table_bytes, manifest, timing)
    table_path = manifest["rgb_source"]["table"]
    top_paths = {manifest["asset"], manifest["asset_receipt"], table_path}
    if len({path.casefold() for path in top_paths}) != len(top_paths):
        raise ValueError("asset, asset receipt, and capture table paths must be distinct")
    top_folded = {path.casefold() for path in top_paths}
    if SOURCE_INVENTORY_RELATIVE.casefold() in top_folded:
        raise ValueError("capture source inventory sidecar path collides with a top-level resource")
    for ordinal, row in enumerate(rows):
        for resource_field in ("raw_rgb", "raw_calibration", "output_rgb", "sampling_valid", "conversion_metadata"):
            resource = row[resource_field]
            if resource is not None and resource["path"] is not None and resource["path"].casefold() in top_folded:
                raise ValueError(f"capture row {ordinal} resource aliases a top-level inference resource")
            if resource is not None and resource["path"] is not None and resource["path"].casefold() == SOURCE_INVENTORY_RELATIVE.casefold():
                raise ValueError(f"capture row {ordinal} resource aliases the source inventory sidecar")
    expected_table_sha = _valid_digest(manifest["source_hashes"][table_path], "capture table source hash")
    actual_table_sha = _sha256_bytes(raw_table_bytes)
    if actual_table_sha != expected_table_sha:
        raise CaptureIntegrityError("capture table bytes do not match input source_hashes")
    selection = manifest["selection"]
    _strict_keys(selection, frozenset({
        "reviewed", "frame_id", "source_frame_id", "timestamp_s", "points", "labels",
        "capture_row_sha256", "output_rgb_sha256", "sampling_valid_sha256", "origin",
    }), "selection")
    if selection.get("reviewed") is not True or selection.get("origin") != "image_only_once":
        raise ValueError("sequence selection must be reviewed exactly once from the image")
    if not rows:
        raise ValueError("capture table cannot be empty")
    first = rows[0]
    if (type(selection.get("frame_id")) is not int or selection["frame_id"] != 0 or
            type(selection.get("source_frame_id")) is not int or
            selection["source_frame_id"] != first["source_frame_id"] or
            _finite_real(selection.get("timestamp_s"), "selection.timestamp_s") != first["timestamp_s"]):
        raise ValueError("selection must bind to the first canonical row")
    _contract_digest(selection.get("capture_row_sha256"), "selection.capture_row_sha256")
    _contract_digest(selection.get("output_rgb_sha256"), "selection.output_rgb_sha256")
    _contract_digest(selection.get("sampling_valid_sha256"), "selection.sampling_valid_sha256")
    if (selection["capture_row_sha256"].upper() != first["capture_row_sha256"] or
            selection["output_rgb_sha256"].upper() != (
                "" if first["output_rgb"] is None else first["output_rgb"]["pixel_sha256"].upper()) or
            selection["sampling_valid_sha256"].upper() != (first["sampling_valid_sha256"] or "").upper()):
        raise ValueError("selection digests do not bind the first capture row")
    points = selection.get("points")
    labels = selection.get("labels")
    if not isinstance(points, list) or not points or not isinstance(labels, list) or len(labels) != len(points):
        raise ValueError("selection points and labels must be nonempty parallel lists")
    positive = False
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"selection point {index} must be [x,y]")
        x = _finite_real(point[0], f"selection.points[{index}].x")
        y = _finite_real(point[1], f"selection.points[{index}].y")
        if not (0.0 <= x < 719.0 and 0.0 <= y < 719.0):
            raise ValueError(f"selection point {index} lacks a complete in-frame sampling footprint")
        if type(labels[index]) is not int or labels[index] not in (0, 1):
            raise ValueError(f"selection label {index} must be integer 0 or 1")
        positive |= labels[index] == 1
    if not positive:
        raise ValueError("selection requires at least one positive object point")
    if table["source_revision"] == "" or table["archive_sha256"].upper() == "0" * 64:
        raise ValueError("capture table acquisition identity is invalid")
    table_stream = table["stream_id"]
    if not manifest["clock"]["timestamp_source"].endswith(f";stream{table_stream}"):
        raise ValueError("capture clock timestamp source does not identify the table stream")

    capture_plan = CapturePlan(
        manifest=_freeze(manifest),
        table=_freeze(table),
        timing=_freeze(dict(timing)),
        rows=tuple(_freeze(row) for row in rows),
        input_manifest_sha256=_sha256_bytes(raw_input_bytes),
        capture_table_sha256=actual_table_sha,
        selection=_freeze(selection),
        _plan_token=_CAPTURE_PLAN_TOKEN,
    )
    object.__setattr__(
        capture_plan, "_semantic_sha256", _capture_plan_semantic_sha256(capture_plan))
    return _validate_capture_plan_record(capture_plan)


class _PathRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._spellings: dict[str, str] = {}
        self._resolved: dict[str, str] = {}

    def resolve(self, relative: str, *, must_exist: bool = True) -> Path:
        normalized = _relative_path(relative, "bundle resource path")
        folded = normalized.casefold()
        prior = self._spellings.get(folded)
        if prior is not None and prior != normalized:
            raise ValueError("bundle contains case-insensitive path aliases")
        self._spellings[folded] = normalized
        candidate = self.root.joinpath(*normalized.split("/"))
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise CaptureIntegrityError(f"cannot resolve declared resource {normalized!r}") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise CaptureIntegrityError(f"resource {normalized!r} escapes the inference bundle") from exc
        resolved_key = os.path.normcase(str(resolved)).casefold()
        old = self._resolved.get(resolved_key)
        if old is not None and old != normalized:
            raise ValueError("bundle contains two paths resolving to the same resource identity")
        self._resolved[resolved_key] = normalized
        if must_exist and not resolved.is_file():
            raise CaptureIntegrityError(f"declared resource {normalized!r} is not a regular file")
        return resolved


def _stream_hash(path: Path, *, expected_bytes: int | None, maximum: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if maximum is not None and count > maximum:
                    raise CaptureIntegrityError(f"resource exceeds its {maximum}-byte limit")
                digest.update(block)
    except CaptureIntegrityError:
        raise
    except OSError as exc:
        raise CaptureIntegrityError(f"cannot read authenticated resource {path}") from exc
    if expected_bytes is not None and count != expected_bytes:
        raise CaptureIntegrityError(f"resource byte count mismatch for {path}")
    return digest.hexdigest().upper(), count


def _plan_from_verified_or_plan(plan: CapturePlan | VerifiedCapturePlan) -> CapturePlan:
    if type(plan) is VerifiedCapturePlan:
        return _validate_verified_capture_plan_record(plan).plan
    if type(plan) is CapturePlan:
        return _validate_capture_plan_record(plan)
    raise TypeError("plan must be CapturePlan or VerifiedCapturePlan")


def verify_capture_resources(bundle: str | os.PathLike[str],
                             plan: CapturePlan | VerifiedCapturePlan) -> VerifiedCapturePlan:
    """Stream and authenticate all declared resources without decoding images."""

    capture_plan = _plan_from_verified_or_plan(plan)
    if capture_plan._plan_token is not _CAPTURE_PLAN_TOKEN:
        raise CaptureIntegrityError("capture plan was not produced by prepare_capture_plan")
    root = Path(bundle).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("capture bundle must be a directory")
    registry = _PathRegistry(root)
    verified: dict[str, str] = {}
    manifest = capture_plan.manifest
    inventory_path = registry.resolve(SOURCE_INVENTORY_RELATIVE)
    inventory_bytes, _ = _read_bounded(inventory_path, MAX_INVENTORY_BYTES)
    inventory, inventory_sha, inventory_rows = _validate_inventory(
        inventory_bytes, capture_plan.table["source_inventory_sha256"])
    if (inventory["source_revision"] != capture_plan.table["source_revision"] or
            inventory["clip_id"] != capture_plan.table["clip_id"] or
            _valid_digest(inventory["archive_sha256"], "inventory.archive_sha256") !=
            capture_plan.table["archive_sha256"] or
            _inventory_stream_id(inventory_rows) != capture_plan.table["stream_id"] or
            len(inventory_rows) != len(capture_plan.rows)):
        raise CaptureIntegrityError("source inventory acquisition identity disagrees with the capture table")
    for ordinal, (source_row, table_row) in enumerate(zip(inventory_rows, capture_plan.rows)):
        raw_rgb = table_row["raw_rgb"]
        raw_calibration = table_row["raw_calibration"]
        if (table_row["source_frame_id"] != source_row["source_frame_id"] or
                table_row["source_timestamp_ns"] != source_row["source_timestamp_ns"] or
                raw_rgb["sha256"] != source_row["rgb_sha256"] or
                raw_calibration["sha256"] != source_row["calibration_sha256"] or
                table_row["raw_camera_metadata_sha256"] != source_row["raw_camera_metadata_sha256"]):
            raise CaptureIntegrityError("source inventory identity disagrees with capture row",
                                        ordinal=ordinal, source_frame_id=table_row["source_frame_id"])
        for resource, expected_path in ((raw_rgb, source_row["rgb_path"]),
                                        (raw_calibration, source_row["calibration_path"])):
            source_path = expected_path.casefold()
            top_paths = {value.casefold() for value in (
                capture_plan.manifest["asset"], capture_plan.manifest["asset_receipt"],
                capture_plan.manifest["rgb_source"]["table"], SOURCE_INVENTORY_RELATIVE,
            )}
            if source_path in top_paths:
                raise CaptureIntegrityError("source inventory path aliases a top-level bundle resource",
                                            ordinal=ordinal, source_frame_id=table_row["source_frame_id"])
            for output_field in ("output_rgb", "sampling_valid", "conversion_metadata"):
                output_resource = table_row[output_field]
                if (output_resource is not None and output_resource["path"] is not None and
                        output_resource["path"].casefold() == source_path):
                    raise CaptureIntegrityError("source inventory path aliases a converted artifact",
                                                ordinal=ordinal, source_frame_id=table_row["source_frame_id"])
            if resource["path"] is None:
                if table_row["conversion_state"] != "unavailable" or table_row["failure_reason"] != "raw_missing":
                    raise CaptureIntegrityError("only a typed raw_missing row may omit a source path",
                                                ordinal=ordinal, source_frame_id=table_row["source_frame_id"])
            elif resource["path"] != expected_path:
                raise CaptureIntegrityError("capture row source path disagrees with its inventory",
                                            ordinal=ordinal, source_frame_id=table_row["source_frame_id"])
    verified[SOURCE_INVENTORY_RELATIVE] = inventory_sha
    runtime_closure = _source_runtime_closure()
    if runtime_closure != _thaw(capture_plan.table["converter"]["source_closure"]):
        raise CaptureIntegrityError("capture conversion source closure differs from current reviewed adapters")
    resource_limits = {
        manifest["asset"]: 8 * 1024 * 1024 * 1024,
        manifest["asset_receipt"]: MAX_TABLE_BYTES,
        manifest["rgb_source"]["table"]: MAX_TABLE_BYTES,
    }
    for relative, expected in manifest["source_hashes"].items():
        resolved = registry.resolve(relative)
        actual, _ = _stream_hash(resolved, expected_bytes=None, maximum=resource_limits[relative])
        if actual != expected.upper():
            raise CaptureIntegrityError(f"top-level source resource hash mismatch: {relative}")
        verified[relative] = actual
    table_path = manifest["rgb_source"]["table"]
    if verified[table_path] != capture_plan.capture_table_sha256:
        raise CaptureIntegrityError("capture table changed after pure preparation")

    for ordinal, row in enumerate(capture_plan.rows):
        for resource_name in ("raw_rgb", "raw_calibration", "output_rgb", "sampling_valid", "conversion_metadata"):
            resource = row[resource_name]
            if resource is None or resource["path"] is None:
                continue
            relative = resource["path"]
            resolved = registry.resolve(relative)
            digest_key = "sha256" if resource_name in ("raw_rgb", "raw_calibration") else "file_sha256"
            expected_digest = resource[digest_key]
            byte_limit = MAX_CALIBRATION_BYTES if resource_name == "raw_calibration" else (
                MAX_ENCODED_IMAGE_BYTES if resource_name in ("raw_rgb", "output_rgb", "sampling_valid") else MAX_TABLE_BYTES
            )
            actual, count = _stream_hash(
                resolved, expected_bytes=resource["byte_count"], maximum=byte_limit)
            if actual != expected_digest.upper():
                raise CaptureIntegrityError(
                    f"capture resource hash mismatch for row {ordinal} {resource_name}",
                    ordinal=ordinal, source_frame_id=row["source_frame_id"],
                )
            previous = verified.get(relative)
            if previous is not None and previous != actual:
                raise CaptureIntegrityError(f"resource binding conflict for {relative}")
            verified[relative] = actual
        if row["conversion_state"] == "available":
            _verify_conversion_metadata(root, registry, row, ordinal)
    verified_plan = VerifiedCapturePlan(
        plan=capture_plan,
        bundle=root,
        verified_resources=_freeze(verified),
        _verification_token=_VERIFIED_PLAN_TOKEN,
    )
    object.__setattr__(
        verified_plan, "_verification_sha256",
        _verified_capture_plan_semantic_sha256(verified_plan))
    return _validate_verified_capture_plan_record(verified_plan)


def _verify_conversion_metadata(root: Path, registry: _PathRegistry,
                                row: Mapping[str, Any], ordinal: int) -> None:
    resource = row["conversion_metadata"]
    path = registry.resolve(resource["path"])
    if resource["byte_count"] > MAX_TABLE_BYTES:
        raise CaptureIntegrityError("conversion metadata exceeds the 2 MiB JSON limit", ordinal=ordinal)
    raw, _ = _read_bounded(path, MAX_TABLE_BYTES)
    if len(raw) != resource["byte_count"] or _sha256_bytes(raw) != resource["file_sha256"].upper():
        raise CaptureIntegrityError("conversion metadata changed after resource verification", ordinal=ordinal)
    document = _parse_json_document(raw, limit=MAX_TABLE_BYTES, name="conversion metadata")
    if _canonical_json_bytes(document) != raw:
        raise CaptureIntegrityError("conversion metadata is not canonical immutable JSON", ordinal=ordinal)
    if _sha256_bytes(raw) != row["conversion_metadata_sha256"].upper():
        raise CaptureIntegrityError("conversion metadata digest disagrees with its row", ordinal=ordinal)
    expected_fields = frozenset({
        "schema", "source_rgb_sha256", "calibration_sha256", "canonical_calibration_sha256",
        "warp_sha256", "sampling_valid_sha256", "output_rgb_sha256", "output", "sampling",
    })
    if set(document) != set(expected_fields):
        raise CaptureIntegrityError("conversion metadata field allowlist mismatch", ordinal=ordinal)
    if document["schema"] != "quality-camera-warped-rgb-v1":
        raise CaptureIntegrityError("conversion metadata schema mismatch", ordinal=ordinal)
    expected_hashes = {
        "source_rgb_sha256": row["decoded_raw_rgb_pixel_sha256"],
        "calibration_sha256": row["raw_calibration"]["sha256"],
        "canonical_calibration_sha256": row["calibration_values_sha256"],
        "warp_sha256": row["warp_sha256"],
        "sampling_valid_sha256": row["sampling_valid_sha256"],
        "output_rgb_sha256": row["output_rgb"]["pixel_sha256"],
    }
    for key, expected_digest in expected_hashes.items():
        if _valid_digest(document.get(key), f"conversion_metadata.{key}") != expected_digest.upper():
            raise CaptureIntegrityError(f"conversion metadata {key} differs from row {ordinal}", ordinal=ordinal)
    expected_sampling = {
        "channel_order": "RGB",
        "interpolation": "bilinear-four-neighbor",
        "neighbor_footprint": "all four integer pixel centers must be in bounds and declared valid disk",
        "rounding": "floor(value+0.5)",
        "clip_range": [0, 255],
        "invalid_fill_rgb": [128, 128, 128],
    }
    if document["output"] != OUTPUT_CAMERA or document["sampling"] != expected_sampling:
        raise CaptureIntegrityError("conversion metadata camera/sampling configuration mismatch", ordinal=ordinal)
    # Map, geometric-valid and domain-valid digests are authenticated directly
    # by the capture row; the reviewed WarpedRGB metadata schema does not emit
    # those three values, so no synthetic metadata fields are invented here.


def load_capture_plan(bundle: str | os.PathLike[str], raw_input_bytes: bytes) -> VerifiedCapturePlan:
    """Authenticate the table, then the entire declared resource closure.

    Timing and input structure are validated before resolving or reading the
    table path, so malformed clocks cannot cause any table-resource access.
    """

    manifest, _ = _parse_input(raw_input_bytes)
    table_relative = manifest["rgb_source"]["table"]
    root = Path(bundle).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("capture bundle must be a directory")
    registry = _PathRegistry(root)
    table_path = registry.resolve(table_relative)
    raw_table, table_size = _read_bounded(table_path, MAX_TABLE_BYTES)
    if _sha256_bytes(raw_table) != manifest["source_hashes"][table_relative].upper():
        raise CaptureIntegrityError("capture table bytes do not match input source_hashes")
    plan = prepare_capture_plan(raw_input_bytes, raw_table)
    return verify_capture_resources(root, plan)


def _read_bounded(path: Path, maximum: int) -> tuple[bytes, int]:
    data = bytearray()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(min(1024 * 1024, maximum + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data) > maximum:
                    raise CaptureIntegrityError(f"resource exceeds its {maximum}-byte limit")
    except CaptureIntegrityError:
        raise
    except OSError as exc:
        raise CaptureIntegrityError(f"cannot read resource {path}") from exc
    return bytes(data), len(data)


def _strict_ordinal(ordinal: Any, frame_count: int) -> int:
    if type(ordinal) is not int or ordinal < 0 or ordinal >= frame_count:
        raise ValueError("ordinal must be a non-bool integer inside the capture plan")
    return ordinal


def _encoded_dimensions(encoded: bytes) -> tuple[int, int] | None:
    """Read PNG/JPEG dimensions without allocating a decoded image buffer."""

    if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(encoded) < 24 or encoded[12:16] != b"IHDR":
            return None
        width = int.from_bytes(encoded[16:20], "big")
        height = int.from_bytes(encoded[20:24], "big")
        return width, height
    if not encoded.startswith(b"\xff\xd8"):
        return None
    index = 2
    frame_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while index < len(encoded):
        if encoded[index] != 0xFF:
            index += 1
            continue
        while index < len(encoded) and encoded[index] == 0xFF:
            index += 1
        if index >= len(encoded):
            break
        marker = encoded[index]
        index += 1
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            if marker == 0xD9:
                break
            continue
        if index + 2 > len(encoded):
            break
        segment_length = int.from_bytes(encoded[index:index + 2], "big")
        if segment_length < 2 or index + segment_length > len(encoded):
            return None
        if marker in frame_markers:
            if segment_length < 8:
                return None
            height = int.from_bytes(encoded[index + 3:index + 5], "big")
            width = int.from_bytes(encoded[index + 5:index + 7], "big")
            return width, height
        index += segment_length
    return None


def _camera_pixel_digests(valid: Any) -> tuple[str, str]:
    import numpy as np

    canonical = np.ascontiguousarray(valid, dtype=np.uint8)
    raw_digest = hashlib.sha256(memoryview(canonical).cast("B")).hexdigest().upper()
    hasher = hashlib.sha256()
    hasher.update(b"sampling_valid\0")
    hasher.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii") + b"\0")
    hasher.update(b"|u1\0")
    hasher.update(memoryview(canonical).cast("B"))
    return raw_digest, hasher.hexdigest().upper()


def _decode_result(decoder: Any, encoded: bytes, *, expected: str) -> tuple[Any, str]:
    """Run an injected decoder and return (array, channel_order).

    A decoder is a callable ``decoder(encoded_bytes)`` or an object exposing
    ``decode(encoded_bytes)``. It returns an array or ``(array, order)`` where
    order is RGB, BGR, or GRAY. The built-in lazy OpenCV decoder returns BGR or
    GRAY according to the encoded PNG/JPEG channels.
    """

    dimensions = _encoded_dimensions(encoded)
    if dimensions is not None:
        width, height = dimensions
        if width <= 0 or height <= 0 or width * height * 3 > MAX_RAW_RGB_BYTES:
            raise ValueError("encoded image dimensions exceed the reviewed raw RGB bound")
    elif decoder is None:
        raise ValueError("default decoder accepts only bounded PNG or JPEG resources")
    if decoder is None:
        import cv2
        import numpy as np

        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("OpenCV could not decode the encoded image")
        order = "GRAY" if decoded.ndim == 2 else ("BGRA" if decoded.ndim == 3 and decoded.shape[2] == 4 else "BGR")
        return decoded, order
    method = getattr(decoder, "decode", None)
    value = method(encoded) if callable(method) else decoder(encoded)
    order = getattr(decoder, "channel_order", None)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        value, order = value
    if order is None:
        order = "GRAY" if getattr(value, "ndim", None) == 2 else "RGB"
    if order not in ("RGB", "BGR", "GRAY"):
        raise ValueError("decoder channel_order must be RGB, BGR, or GRAY")
    if expected == "GRAY" and order != "GRAY":
        raise ValueError("validity decoder must return one-channel GRAY pixels")
    if expected == "RGB" and order == "GRAY":
        raise ValueError("RGB decoder returned a grayscale image")
    return value, order


def _as_rgb(array: Any, order: str, *, width: int, height: int) -> Any:
    import numpy as np

    value = np.asarray(array)
    if value.dtype != np.uint8 or value.shape != (height, width, 3):
        raise ValueError("decoded RGB must be HxWx3 uint8 at the declared dimensions")
    if order == "BGR":
        value = value[..., ::-1]
    elif order != "RGB":
        raise ValueError("decoded RGB channel order is invalid")
    return np.ascontiguousarray(value)


def _as_validity(array: Any, order: str, *, width: int = 720, height: int = 720) -> Any:
    import numpy as np

    value = np.asarray(array)
    if order != "GRAY" or value.dtype != np.uint8 or value.shape != (height, width):
        raise ValueError("decoded sampling validity must be a 720x720 uint8 grayscale PNG")
    if not np.isin(value, (0, 255)).all():
        raise ValueError("sampling-valid PNG values must be exactly 0 or 255")
    return np.ascontiguousarray(value == 255)


def _read_verified_resource(bundle: Path, resource: Mapping[str, Any], *,
                            digest_key: str, maximum: int) -> bytes:
    path = _PathRegistry(bundle).resolve(resource["path"])
    data, count = _read_bounded(path, maximum)
    if count != resource["byte_count"] or _sha256_bytes(data) != resource[digest_key].upper():
        raise CaptureIntegrityError(f"resource changed after preflight: {resource['path']}")
    return data


class CaptureReader:
    """One-record, explicit-allowance reader over a VerifiedCapturePlan."""

    def __init__(self, bundle: str | os.PathLike[str], verified_plan: VerifiedCapturePlan,
                 *, decoder: Any = None, maximum_cache_frames: int = 1,
                 _single_frame: SingleFrameCapability | None = None):
        _validate_verified_capture_plan_record(verified_plan)
        if type(maximum_cache_frames) is not int or maximum_cache_frames not in (0, 1):
            raise ValueError("maximum_cache_frames must be 0 or 1")
        capability_ordinal: int | None = None
        if _single_frame is not None:
            if (type(_single_frame) is not SingleFrameCapability or
                    _single_frame._capability_token is not _SINGLE_FRAME_TOKEN):
                raise CaptureIntegrityError("single-frame capability was not issued by this module")
            capability_ordinal = _strict_ordinal(_single_frame.ordinal, len(verified_plan.rows))
            row = verified_plan.rows[capability_ordinal]
            if (type(_single_frame.input_manifest_sha256) is not str or
                    type(_single_frame.capture_table_sha256) is not str or
                    type(_single_frame.source_frame_id) is not int or
                    type(_single_frame.capture_row_sha256) is not str or
                    _single_frame.input_manifest_sha256 != verified_plan.input_manifest_sha256 or
                    _single_frame.capture_table_sha256 != verified_plan.capture_table_sha256 or
                    _single_frame.source_frame_id != row["source_frame_id"] or
                    _single_frame.capture_row_sha256 != row["capture_row_sha256"]):
                raise CaptureIntegrityError("single-frame capability does not bind this plan row")
        self.bundle = Path(bundle).resolve(strict=True)
        if _normalized_bundle_identity(self.bundle) != _normalized_bundle_identity(verified_plan.bundle):
            raise CaptureIntegrityError("CaptureReader bundle differs from preflight bundle")
        self.plan = verified_plan
        self.decoder = decoder
        self.maximum_cache_frames = maximum_cache_frames
        self._capability = _single_frame
        self._allowance = 0 if capability_ordinal is None else capability_ordinal
        self._cache: tuple[int, DecodedCapture] | None = None
        self._highest_returned = -1
        self._closed = False
        self._terminal_error: CaptureIntegrityError | CaptureDecodeError | None = None
        self._stats: dict[int, dict[str, Any]] = {}
        self._totals = {
            "attempts": 0, "returns": 0, "cache_hits": 0, "past_reads": 0,
            "decode_calls": 0, "bytes_read": 0, "failures": 0,
        }

    @classmethod
    def from_capability(cls, bundle: str | os.PathLike[str], verified_plan: VerifiedCapturePlan,
                        capability: SingleFrameCapability, *, decoder: Any = None,
                        maximum_cache_frames: int = 1) -> "CaptureReader":
        if type(capability) is not SingleFrameCapability or capability._capability_token is not _SINGLE_FRAME_TOKEN:
            raise CaptureIntegrityError("single-frame capability was not issued by this module")
        return cls(bundle, verified_plan, decoder=decoder,
                   maximum_cache_frames=maximum_cache_frames, _single_frame=capability)

    def allow(self, ordinal: int) -> None:
        if self._closed or self._terminal_error is not None:
            raise CaptureIntegrityError("capture reader is terminal or closed")
        target = _strict_ordinal(ordinal, len(self.plan.rows))
        if self._capability is not None:
            if target != self._capability.ordinal:
                raise CaptureIntegrityError("single-frame capability cannot allow another row",
                                            ordinal=target, source_frame_id=self.plan.rows[target]["source_frame_id"])
            return
        if target < self._allowance:
            raise CaptureIntegrityError("capture allowance cannot move backward", ordinal=target,
                                        source_frame_id=self.plan.rows[target]["source_frame_id"])
        if target > self._allowance + 1:
            raise CaptureIntegrityError("capture allowance can advance by at most one row", ordinal=target,
                                        source_frame_id=self.plan.rows[target]["source_frame_id"])
        self._allowance = target

    def _row_stats(self, ordinal: int) -> dict[str, Any]:
        return self._stats.setdefault(ordinal, {
            "attempts": 0, "returns": 0, "cache_hits": 0, "past_reads": 0,
            "decode_calls": 0, "bytes_read": 0, "failures": 0, "failure_reason": None,
        })

    def read(self, ordinal: int) -> DecodedCapture:
        index = _strict_ordinal(ordinal, len(self.plan.rows))
        row = self.plan.rows[index]
        if self._capability is not None and index != self._capability.ordinal:
            raise CaptureIntegrityError("single-frame capability cannot read another row",
                                        ordinal=index, source_frame_id=row["source_frame_id"])
        if index > self._allowance:
            raise CaptureIntegrityError("capture row is beyond the current causal allowance",
                                        ordinal=index, source_frame_id=row["source_frame_id"])
        if self._closed:
            raise CaptureIntegrityError("capture reader is closed", ordinal=index,
                                        source_frame_id=row["source_frame_id"])
        if self._terminal_error is not None:
            raise CaptureIntegrityError("capture reader is terminal after an earlier failure",
                                        ordinal=index, source_frame_id=row["source_frame_id"])
        stats = self._row_stats(index)
        stats["attempts"] += 1
        self._totals["attempts"] += 1
        if row["conversion_state"] == "unavailable":
            error = CaptureUnavailable(
                f"capture row {index} is unavailable: {row['failure_reason']}",
                ordinal=index, source_frame_id=row["source_frame_id"], reason=row["failure_reason"],
            )
            stats["failure_reason"] = error.reason
            stats["failures"] += 1
            self._totals["failures"] += 1
            raise error
        if self._cache is not None and self._cache[0] == index:
            stats["cache_hits"] += 1
            stats["returns"] += 1
            self._totals["cache_hits"] += 1
            self._totals["returns"] += 1
            return self._cache[1]
        if index < self._highest_returned:
            stats["past_reads"] += 1
            self._totals["past_reads"] += 1
        try:
            import numpy as np
            rgb_resource = row["output_rgb"]
            valid_resource = row["sampling_valid"]
            rgb_bytes = _read_verified_resource(
                self.bundle, rgb_resource, digest_key="file_sha256", maximum=MAX_ENCODED_IMAGE_BYTES)
            stats["bytes_read"] += len(rgb_bytes)
            self._totals["bytes_read"] += len(rgb_bytes)
            self._totals["decode_calls"] += 1
            stats["decode_calls"] += 1
            decoded_rgb, rgb_order = _decode_result(self.decoder, rgb_bytes, expected="RGB")
            rgb = np.array(_as_rgb(decoded_rgb, rgb_order, width=720, height=720), copy=True)
            del rgb_bytes, decoded_rgb
            valid_bytes = _read_verified_resource(
                self.bundle, valid_resource, digest_key="file_sha256", maximum=MAX_ENCODED_IMAGE_BYTES)
            stats["bytes_read"] += len(valid_bytes)
            self._totals["bytes_read"] += len(valid_bytes)
            self._totals["decode_calls"] += 1
            stats["decode_calls"] += 1
            decoded_valid, valid_order = _decode_result(self.decoder, valid_bytes, expected="GRAY")
            valid = np.array(_as_validity(decoded_valid, valid_order), copy=True)
            del valid_bytes, decoded_valid
            rgb_digest = hashlib.sha256(memoryview(np_contiguous(rgb)).cast("B")).hexdigest().upper()
            valid_pixel_digest, camera_valid_digest = _camera_pixel_digests(valid)
            if rgb_digest != rgb_resource["pixel_sha256"].upper():
                raise CaptureIntegrityError("decoded RGB pixel digest mismatch", ordinal=index,
                                            source_frame_id=row["source_frame_id"])
            if valid_pixel_digest != valid_resource["pixel_sha256"].upper():
                raise CaptureIntegrityError("decoded sampling-valid pixel digest mismatch", ordinal=index,
                                            source_frame_id=row["source_frame_id"])
            if camera_valid_digest != row["sampling_valid_sha256"].upper():
                raise CaptureIntegrityError("decoded sampling-valid camera digest mismatch", ordinal=index,
                                            source_frame_id=row["source_frame_id"])
            if _thaw(self.plan.table["output_camera"]) != OUTPUT_CAMERA:
                raise CaptureIntegrityError("verified output camera changed", ordinal=index,
                                            source_frame_id=row["source_frame_id"])
            if index == 0:
                points = np_array(self.plan.selection["points"], dtype="float64")
                if not supported_points(valid, points).all():
                    raise CaptureIntegrityError("initial selection point lies outside supported sampling validity",
                                                ordinal=index, source_frame_id=row["source_frame_id"])
            # Rebuild arrays from immutable bytes so neither the decoder nor a
            # downstream diagnostic can mutate the shared record.
            rgb_backing = np_contiguous(rgb).tobytes()
            valid_backing = np_contiguous(valid, dtype="bool").tobytes()
            readonly_rgb = np_frombuffer(rgb_backing, dtype="uint8", shape=(720, 720, 3))
            readonly_valid = np_frombuffer(valid_backing, dtype="bool", shape=(720, 720))
            from .quality_time import FrameKey, PHYSICAL
            record = DecodedCapture(
                key=FrameKey(index, row["timestamp_s"], PHYSICAL),
                source_frame_id=row["source_frame_id"],
                rgb=readonly_rgb,
                sampling_valid=readonly_valid,
                intrinsics=OUTPUT_INTRINSICS,
                capture_row_sha256=row["capture_row_sha256"],
                capture_table_sha256=self.plan.capture_table_sha256,
                rgb_pixel_sha256=rgb_digest,
                sampling_valid_sha256=camera_valid_digest,
                warp_sha256=row["warp_sha256"],
                calibration_sha256=row["calibration_values_sha256"],
            )
        except CaptureIntegrityError as exc:
            stats["failure_reason"] = exc.reason
            stats["failures"] += 1
            self._totals["failures"] += 1
            self._terminal_error = exc
            self._cache = None
            raise
        except Exception as exc:
            error = CaptureDecodeError(
                f"cannot decode authenticated capture row {index}: {exc}", ordinal=index,
                source_frame_id=row["source_frame_id"],
            )
            stats["failure_reason"] = error.reason
            stats["failures"] += 1
            self._totals["failures"] += 1
            self._terminal_error = error
            self._cache = None
            raise error from exc
        stats["returns"] += 1
        self._totals["returns"] += 1
        self._highest_returned = max(self._highest_returned, index)
        if self.maximum_cache_frames:
            self._cache = (index, record)
        else:
            self._cache = None
        return record

    def snapshot(self, ordinal: int) -> Mapping[str, Any]:
        index = _strict_ordinal(ordinal, len(self.plan.rows))
        row = self.plan.rows[index]
        stats = self._stats.get(index, {
            "attempts": 0, "returns": 0, "cache_hits": 0, "past_reads": 0,
            "decode_calls": 0, "bytes_read": 0, "failures": 0, "failure_reason": None,
        })
        return _freeze({
            "ordinal": index,
            "source_frame_id": row["source_frame_id"],
            "allowed_through_ordinal": self._allowance,
            "closed": self._closed,
            "terminal": self._terminal_error is not None,
            "capture_row_sha256": row["capture_row_sha256"],
            "capture_table_sha256": self.plan.capture_table_sha256,
            "maximum_cache_frames": self.maximum_cache_frames,
            "cache_ordinal": None if self._cache is None else self._cache[0],
            "attempts": stats["attempts"], "returns": stats["returns"],
            "cache_hits": stats["cache_hits"], "past_reads": stats["past_reads"],
            "decode_calls": stats["decode_calls"], "bytes_read": stats["bytes_read"],
            "failures": stats["failures"], "failure_reason": stats["failure_reason"],
            "totals": dict(self._totals),
        })

    def close(self) -> None:
        self._cache = None
        self._capability = None
        self.decoder = None
        self._closed = True


def np_array(value: Any, *, dtype: str) -> Any:
    import numpy as np
    return np.asarray(value, dtype=dtype)


def np_contiguous(value: Any, *, dtype: str | None = None) -> Any:
    import numpy as np
    return np.ascontiguousarray(value, dtype=dtype)


def np_frombuffer(data: bytes, *, dtype: str, shape: tuple[int, ...]) -> Any:
    import numpy as np
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)


def single_frame_capability(verified_plan: VerifiedCapturePlan, ordinal: int) -> SingleFrameCapability:
    """Create a one-row capability bound to the full authenticated plan."""

    _validate_verified_capture_plan_record(verified_plan)
    index = _strict_ordinal(ordinal, len(verified_plan.rows))
    row = verified_plan.rows[index]
    if row["conversion_state"] != "available":
        raise CaptureUnavailable("single-frame capability cannot target an unavailable row",
                                 ordinal=index, source_frame_id=row["source_frame_id"],
                                 reason=row["failure_reason"])
    return SingleFrameCapability(
        input_manifest_sha256=verified_plan.input_manifest_sha256,
        capture_table_sha256=verified_plan.capture_table_sha256,
        ordinal=index,
        source_frame_id=row["source_frame_id"],
        capture_row_sha256=row["capture_row_sha256"],
        _capability_token=_SINGLE_FRAME_TOKEN,
    )


def supported_points(valid: Any, uv: Any) -> Any:
    """Return whether each finite integer-center point has all four valid neighbors."""

    import numpy as np

    validity = np.asarray(valid)
    points = np.asarray(uv)
    if validity.ndim != 2 or validity.dtype != np.bool_ or not validity.size:
        raise ValueError("valid must be a nonempty two-dimensional bool map")
    if points.ndim == 1 and points.shape == (2,):
        points = points.reshape((1, 2))
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "iuf" or points.dtype.kind == "b":
        raise ValueError("uv must be a finite real Nx2 array")
    points = points.astype(np.float64, copy=False)
    if not np.isfinite(points).all():
        raise ValueError("uv must contain only finite values")
    h, w = validity.shape
    x0 = np.floor(points[:, 0]).astype(np.int64)
    y0 = np.floor(points[:, 1]).astype(np.int64)
    in_bounds = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    result = np.zeros((len(points),), dtype=np.bool_)
    eligible = np.flatnonzero(in_bounds)
    if eligible.size:
        xx, yy = x0[eligible], y0[eligible]
        result[eligible] = (
            validity[yy, xx] & validity[yy, xx + 1] &
            validity[yy + 1, xx] & validity[yy + 1, xx + 1]
        )
    return result


def restrict_foreground(mask: Any, valid: Any) -> Any:
    """Return a fresh bool mask containing only foreground on supported pixels."""

    import numpy as np

    foreground = np.asarray(mask)
    validity = np.asarray(valid)
    if validity.ndim != 2 or validity.dtype != np.bool_ or not validity.size:
        raise ValueError("valid must be a nonempty two-dimensional bool map")
    if foreground.shape != validity.shape or foreground.ndim != 2:
        raise ValueError("foreground and validity maps must have the same 2D shape")
    if foreground.dtype == np.bool_:
        pass
    elif foreground.dtype == np.uint8 and np.isin(foreground, (0, 1, 255)).all():
        foreground = foreground != 0
    else:
        raise ValueError("foreground must be bool or binary uint8")
    return np.logical_and(foreground, validity)


def _matrix3(value: Any, name: str) -> Any:
    import numpy as np

    matrix = np.asarray(value)
    if matrix.shape != (3, 3) or matrix.dtype.kind not in "iuf" or matrix.dtype.kind == "b":
        raise ValueError(f"{name} must be a real 3x3 matrix")
    matrix = matrix.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError(f"{name} must be finite and invertible")
    return matrix


def _shape2(shape: Any, name: str) -> tuple[int, int]:
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(f"{name} must be (height,width)")
    h, w = shape
    if type(h) is not int or type(w) is not int or h <= 0 or w <= 0 or h * w > 16_777_216:
        raise ValueError(f"{name} dimensions must be positive bounded integers")
    return h, w


def crop_validity(valid: Any, K_native: Any, K_crop: Any,
                  R_crop_from_native: Any, shape: Sequence[int]) -> Any:
    """Map integer crop centers into native coordinates and test four-neighbor support."""

    import numpy as np

    validity = np.asarray(valid)
    if validity.ndim != 2 or validity.dtype != np.bool_ or not validity.size:
        raise ValueError("valid must be a nonempty two-dimensional bool map")
    native_k = _matrix3(K_native, "K_native")
    crop_k = _matrix3(K_crop, "K_crop")
    rotation = _matrix3(R_crop_from_native, "R_crop_from_native")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6) or abs(np.linalg.det(rotation) - 1.0) > 1e-6:
        raise ValueError("R_crop_from_native must be a proper rotation matrix")
    out_h, out_w = _shape2(shape, "shape")
    yy, xx = np.indices((out_h, out_w), dtype=np.float64)
    crop_pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape((-1, 3)).T
    rays_crop = np.linalg.inv(crop_k) @ crop_pixels
    rays_native = rotation.T @ rays_crop
    native_h = native_k @ rays_native
    z = native_h[2]
    front = np.isfinite(native_h).all(axis=0) & (z > 0.0)
    uv = np.full((native_h.shape[1], 2), np.nan, dtype=np.float64)
    uv[front, 0] = native_h[0, front] / z[front]
    uv[front, 1] = native_h[1, front] / z[front]
    support = np.zeros((len(front),), dtype=np.bool_)
    if front.any():
        support[front] = supported_points(validity, uv[front])
    return support.reshape((out_h, out_w))


def _area_bounds(source: int, destination: int) -> tuple[Any, Any]:
    import numpy as np

    coordinates = np.arange(destination, dtype=np.int64)
    first = (coordinates * source) // destination
    last = ((coordinates + 1) * source + destination - 1) // destination - 1
    first = np.clip(first, 0, source - 1)
    last = np.clip(last, 0, source - 1)
    return first, last


def _rect_any(invalid: Any, y0: Any, y1: Any, x0: Any, x1: Any) -> Any:
    import numpy as np

    integral = np.pad(invalid.astype(np.int64), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    yy0, yy1 = y0, y1 + 1
    xx0, xx1 = x0, x1 + 1
    return integral[yy1[:, None], xx1[None, :]] - integral[yy0[:, None], xx1[None, :]] \
        - integral[yy1[:, None], xx0[None, :]] + integral[yy0[:, None], xx0[None, :]]


def resize_validity(valid: Any, output_shape: Sequence[int], kernel: Any) -> Any:
    """Conservatively resize support using every input pixel touched by INTER_AREA."""

    import numpy as np

    validity = np.asarray(valid)
    if validity.ndim != 2 or validity.dtype != np.bool_ or not validity.size:
        raise ValueError("valid must be a nonempty two-dimensional bool map")
    out_h, out_w = _shape2(output_shape, "output_shape")
    if not (kernel == "area" or kernel == "INTER_AREA" or (type(kernel) is int and kernel == 3)):
        raise ValueError("resize_validity supports only the INTER_AREA kernel")
    in_h, in_w = validity.shape
    y0, y1 = _area_bounds(in_h, out_h)
    x0, x1 = _area_bounds(in_w, out_w)
    unsupported_count = _rect_any(~validity, y0, y1, x0, x1)
    return unsupported_count == 0


def integer_depth_points(depth: Any, K: Any) -> tuple[Any, Any]:
    """Return integer pixel centers and millimetre XYZ lifted with full inverse K."""

    import numpy as np

    depth_map = np.asarray(depth)
    if depth_map.ndim != 2 or not depth_map.size or depth_map.dtype.kind not in "iuf" or depth_map.dtype.kind == "b":
        raise ValueError("depth must be a nonempty real 2D millimetre map")
    camera = _matrix3(K, "K")
    height, width = depth_map.shape
    if height * width > 16_777_216:
        raise ValueError("depth dimensions exceed the bounded numerical-helper limit")
    yy, xx = np.indices((height, width), dtype=np.int64)
    pixels = np.stack((xx, yy), axis=-1)
    homogeneous = np.concatenate((pixels.astype(np.float64), np.ones((height, width, 1), dtype=np.float64)), axis=-1)
    rays = np.einsum("ij,hwj->hwi", np.linalg.inv(camera), homogeneous)
    ray_z = rays[..., 2]
    d = depth_map.astype(np.float64, copy=False)
    valid = np.isfinite(d) & (d > 0.0) & np.isfinite(rays).all(axis=-1) & (ray_z > 0.0)
    result = np.full((height, width, 3), np.nan, dtype=np.float64)
    scale = np.zeros_like(d)
    scale[valid] = d[valid] / ray_z[valid]
    result[valid] = rays[valid] * scale[valid, None]
    return pixels, result


def _erode_square(valid: Any, radius: int) -> Any:
    import numpy as np

    height, width = valid.shape
    out = np.zeros((height, width), dtype=np.bool_)
    if height < 2 * radius + 1 or width < 2 * radius + 1:
        return out
    ys = np.arange(radius, height - radius, dtype=np.int64)
    xs = np.arange(radius, width - radius, dtype=np.int64)
    y0 = ys - radius
    y1 = ys + radius
    x0 = xs - radius
    x1 = xs + radius
    unsupported = _rect_any(~valid, y0, y1, x0, x1) == 0
    out[np.ix_(ys, xs)] = unsupported
    return out


def _pyrdown_support(valid: Any) -> Any:
    import numpy as np

    height, width = valid.shape
    out_h, out_w = (height + 1) // 2, (width + 1) // 2
    centers_y = 2 * np.arange(out_h, dtype=np.int64)
    centers_x = 2 * np.arange(out_w, dtype=np.int64)
    eligible_y = (centers_y >= 2) & (centers_y + 2 < height)
    eligible_x = (centers_x >= 2) & (centers_x + 2 < width)
    result = np.zeros((out_h, out_w), dtype=np.bool_)
    if eligible_y.any() and eligible_x.any():
        ys = centers_y[eligible_y]
        xs = centers_x[eligible_x]
        unsupported = _rect_any(
            ~valid,
            ys - 2,
            ys + 2,
            xs - 2,
            xs + 2,
        ) != 0
        result[np.ix_(np.flatnonzero(eligible_y), np.flatnonzero(eligible_x))] = ~unsupported
    return result


def lk_support_pyramid(valid: Any, levels: int, window: int | Sequence[int]) -> tuple[Any, ...]:
    """Return per-level supported LK centers, including windows and 5x5 ancestors.

    A returned true pixel has a complete square LK window plus a one-pixel
    derivative stencil at that level, and every contributing five-tap pyramid
    ancestor lies inside supported image evidence. Pixels outside the image are
    always unsupported, including at the coarsest level.
    """

    import numpy as np

    validity = np.asarray(valid)
    if validity.ndim != 2 or validity.dtype != np.bool_ or not validity.size:
        raise ValueError("valid must be a nonempty two-dimensional bool map")
    _positive_int(levels, "levels", maximum=16)
    if type(window) is int:
        wh = ww = window
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        wh, ww = window
    else:
        raise ValueError("window must be an odd integer or (height,width)")
    if type(wh) is not int or type(ww) is not int or min(wh, ww) <= 0 or wh % 2 == 0 or ww % 2 == 0:
        raise ValueError("window dimensions must be positive odd integers")
    base = np.array(validity, dtype=np.bool_, copy=True)
    output: list[Any] = []
    current = base
    radius_y, radius_x = wh // 2 + 1, ww // 2 + 1
    for level in range(levels):
        # Rectangular stencils are kept exact for asymmetric windows.
        supported = _erode_rect(current, radius_y, radius_x)
        supported.setflags(write=False)
        output.append(supported)
        if level + 1 < levels:
            current = _pyrdown_support(current)
    return tuple(output)


def _erode_rect(valid: Any, radius_y: int, radius_x: int) -> Any:
    import numpy as np

    height, width = valid.shape
    out = np.zeros((height, width), dtype=np.bool_)
    if height < 2 * radius_y + 1 or width < 2 * radius_x + 1:
        return out
    ys = np.arange(radius_y, height - radius_y, dtype=np.int64)
    xs = np.arange(radius_x, width - radius_x, dtype=np.int64)
    is_supported = _rect_any(
        ~valid,
        ys - radius_y,
        ys + radius_y,
        xs - radius_x,
        xs + radius_x,
    ) == 0
    out[np.ix_(ys, xs)] = is_supported
    return out


def _output_camera_from_quality_camera(camera_module: Any) -> dict[str, Any]:
    output = camera_module.PinholeOutput()
    return {
        "width": output.width,
        "height": output.height,
        "fx": output.fx,
        "fy": output.fy,
        "cx": output.cx,
        "cy": output.cy,
        "rotation_output_from_source": list(output.rotation_output_from_source),
        "pixel_center_convention": output.pixel_center_convention,
    }


def _opencv_decoder_identifier() -> str:
    import cv2
    return f"opencv-python:{getattr(cv2, '__version__', 'unknown')}:imdecode-unchanged"


def _service_identifier(service: Any, default: str) -> str:
    identifier = getattr(service, "identifier", None)
    if isinstance(identifier, str) and identifier.strip():
        return identifier
    if service is None:
        return default
    target = getattr(service, "encode", None) or getattr(service, "decode", None) or service
    module = getattr(target, "__module__", type(target).__module__)
    name = getattr(target, "__qualname__", type(target).__qualname__)
    return f"{module}.{name}"


def _opencv_writer(array: Any, image_format: str, channel_order: str) -> bytes:
    import cv2
    import numpy as np

    if image_format != "png":
        raise ValueError("capture resources are written only as PNG")
    data = np.asarray(array)
    if channel_order == "RGB":
        data = data[..., ::-1]
    elif channel_order != "GRAY":
        raise ValueError("writer channel_order must be RGB or GRAY")
    success, encoded = cv2.imencode(".png", data, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise ValueError("OpenCV PNG encoding failed")
    return encoded.tobytes()


def _write_result(writer: Any, array: Any, channel_order: str) -> bytes:
    if writer is None:
        return _opencv_writer(array, "png", channel_order)
    method = getattr(writer, "encode", None)
    encoded = method(array, "png", channel_order) if callable(method) else writer(array, "png", channel_order)
    if type(encoded) is not bytes or not encoded:
        raise ValueError("writer must return nonempty immutable encoded bytes")
    return encoded


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _safe_source_file(source_root: Path, relative: str, *, maximum: int | None) -> tuple[Path, bytes]:
    candidate = source_root.joinpath(*_relative_path(relative, "source path").split("/"))
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(relative) from exc
    except (OSError, RuntimeError) as exc:
        raise CaptureIntegrityError(f"cannot safely resolve source resource {relative!r}") from exc
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise CaptureIntegrityError(f"source resource {relative!r} escapes source_root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(relative)
    data, _ = _read_bounded(resolved, maximum if maximum is not None else (2**63 - 1))
    return resolved, data


def _source_runtime_closure() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    closure: dict[str, str] = {}
    for relative in ("bench/quality_capture.py", "bench/quality_camera.py", "bench/quality_time.py"):
        path = root / relative
        if not path.is_file():
            raise CaptureIntegrityError(f"converter source closure file is missing: {relative}")
        digest, _ = _stream_hash(path, expected_bytes=None)
        closure[relative] = digest
    return closure


def _inventory_stream_id(rows: Sequence[Mapping[str, Any]]) -> str:
    stream_ids = set()
    for row in rows:
        match = _STREAM_SUFFIX_RE.search(row["rgb_path"])
        if match is None:
            raise ValueError("inventory RGB paths must identify their HOT3D camera stream")
        stream_ids.add(match.group(1))
    if len(stream_ids) != 1:
        raise ValueError("source inventory contains multiple RGB camera streams")
    return next(iter(stream_ids))


def _row_digest(row: Mapping[str, Any]) -> str:
    copy = dict(row)
    copy.pop("capture_row_sha256", None)
    return _canonical_digest(copy)


def convert_capture_sequence(
    source_root: str | os.PathLike[str],
    inventory_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    expected_inventory_sha256: str,
    decoder: Any = None,
    writer: Any = None,
) -> ConversionReceipt:
    """Convert a chronological inventory into lossless RGB/valid PNG resources.

    Decoder contract: ``decoder(encoded_bytes)`` or ``decoder.decode(encoded)``
    returns an ndarray, optionally paired with ``"RGB"``, ``"BGR"`` or
    ``"GRAY"``. Writer contract: ``writer(array, "png", "RGB"|"GRAY")`` returns
    encoded bytes. The default OpenCV services are imported only when called.
    """

    source = Path(source_root).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source_root must be a directory")
    inventory_file = Path(inventory_path).resolve(strict=True)
    try:
        inventory_file.relative_to(source)
    except ValueError as exc:
        raise ValueError("inventory_path must be inside source_root") from exc
    raw_inventory, _ = _read_bounded(inventory_file, MAX_INVENTORY_BYTES)
    inventory, inventory_sha, entries = _validate_inventory(raw_inventory, expected_inventory_sha256)
    stream_id = _inventory_stream_id(entries)
    output = Path(output_root).resolve(strict=False)
    if output.exists():
        raise FileExistsError("conversion output_root must be fresh and absent")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("conversion output_root and source_root must not overlap")
    peak_bound = (
        64 * 1024 * 1024 + MAX_RAW_RGB_BYTES + MAX_SINGLE_WARP_BYTES
        + 10 * 720 * 720 + 2 * MAX_ENCODED_IMAGE_BYTES
    )
    if peak_bound > MAX_CONVERSION_TRANSIENT_BYTES:
        raise MemoryError("reviewed conversion transient-memory bound exceeded")

    from . import quality_camera as camera
    output_metadata = _output_camera_from_quality_camera(camera)
    if output_metadata != OUTPUT_CAMERA:
        raise CaptureIntegrityError("quality_camera output camera differs from the frozen sequence contract")
    output_camera_sha = _canonical_digest(output_metadata)
    runtime_decoder = _service_identifier(
        decoder, _opencv_decoder_identifier() if decoder is None else "injected-decoder")
    closure = _source_runtime_closure()
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    (output / "sampling-valid").mkdir()
    (output / "conversion-metadata").mkdir()
    rows: list[dict[str, Any]] = []
    last_calibration_sha: str | None = None
    last_warp: Any = None

    for entry in entries:
        ordinal = entry["frame_id"]
        source_id = entry["source_frame_id"]
        raw_rgb: dict[str, Any] = {
            "path": entry["rgb_path"], "sha256": entry["rgb_sha256"], "byte_count": None,
        }
        raw_calibration: dict[str, Any] = {
            "path": entry["calibration_path"], "sha256": entry["calibration_sha256"], "byte_count": None,
        }
        row: dict[str, Any] = {
            "frame_id": ordinal,
            "source_frame_id": source_id,
            "source_timestamp_ns": entry["source_timestamp_ns"],
            "timestamp_s": entry["timestamp_s"],
            "role": "scored",
            "raw_rgb": raw_rgb,
            "decoded_raw_rgb_pixel_sha256": None,
            "raw_calibration": raw_calibration,
            "raw_camera_metadata_sha256": entry["raw_camera_metadata_sha256"],
            "calibration_values_sha256": None,
            "warp_sha256": None,
            "map_sha256": None,
            "geometric_valid_sha256": None,
            "domain_valid_sha256": None,
            "sampling_valid_sha256": None,
            "output_rgb": None,
            "sampling_valid": None,
            "conversion_state": "unavailable",
            "failure_reason": "raw_missing",
            "conversion_metadata": None,
            "conversion_metadata_sha256": None,
        }
        raw_rgb_bytes: bytes | None = None
        calibration_bytes: bytes | None = None
        decoded_raw: Any = None
        warped: Any = None
        verify_rgb: Any = None
        verify_valid: Any = None
        valid_png: Any = None
        rgb_encoded: bytes | None = None
        valid_encoded: bytes | None = None
        metadata: bytes | None = None
        try:
            try:
                _, raw_rgb_bytes = _safe_source_file(source, entry["rgb_path"], maximum=MAX_ENCODED_IMAGE_BYTES)
                actual_rgb_sha = _sha256_bytes(raw_rgb_bytes)
                if actual_rgb_sha != entry["rgb_sha256"]:
                    raise CaptureIntegrityError(
                        f"raw RGB hash mismatch for source frame {source_id}", ordinal=ordinal,
                        source_frame_id=source_id,
                    )
                raw_rgb["byte_count"] = len(raw_rgb_bytes)
            except FileNotFoundError:
                raw_rgb["path"] = None
                row["failure_reason"] = "raw_missing"
            except CaptureIntegrityError:
                raise
            except Exception as exc:
                row["failure_reason"] = "raw_decode_failed"
                raise _ConversionRowFailure("raw_decode_failed") from exc

            try:
                _, calibration_bytes = _safe_source_file(
                    source, entry["calibration_path"], maximum=MAX_CALIBRATION_BYTES)
                actual_cal_sha = _sha256_bytes(calibration_bytes)
                if actual_cal_sha != entry["calibration_sha256"]:
                    raise CaptureIntegrityError(
                        f"raw calibration hash mismatch for source frame {source_id}", ordinal=ordinal,
                        source_frame_id=source_id,
                    )
                raw_calibration["byte_count"] = len(calibration_bytes)
            except FileNotFoundError:
                raw_calibration["path"] = None
                row["failure_reason"] = "raw_missing"
            except CaptureIntegrityError:
                raise

            # Missing inputs take precedence only after every declared-present
            # source buffer above has been read and authenticated. In particular,
            # keep present-resource digest mismatches fatal, but do not normalize
            # or build a warp for an otherwise-known missing-input row.
            if raw_rgb_bytes is None or calibration_bytes is None:
                row["failure_reason"] = "raw_missing"
                raise _ConversionRowFailure("raw_missing")

            normalized_calibration = None
            if calibration_bytes is not None:
                try:
                    normalized_calibration = camera.normalize_calibration(calibration_bytes)
                    if normalized_calibration.stream_id != stream_id:
                        raise CaptureIntegrityError(
                            f"calibration stream mismatch for source frame {source_id}", ordinal=ordinal,
                            source_frame_id=source_id,
                        )
                except CaptureIntegrityError:
                    raise
                except Exception as exc:
                    row["failure_reason"] = "calibration_invalid"
                    raise _ConversionRowFailure("calibration_invalid") from exc
                row["calibration_values_sha256"] = normalized_calibration.canonical_values_sha256.upper()
                cal_key = normalized_calibration.canonical_values_sha256.upper()
                try:
                    if last_warp is None or last_calibration_sha != cal_key:
                        last_warp = camera.build_warp(normalized_calibration, tile_rows=32)
                        last_calibration_sha = cal_key
                    else:
                        # A reused map has only the same canonical values. Attach
                        # this row's authenticated raw calibration for metadata.
                        last_warp = replace(last_warp, calibration=normalized_calibration)
                except Exception as exc:
                    row["failure_reason"] = "warp_failed"
                    raise _ConversionRowFailure("warp_failed") from exc
                row.update({
                    "warp_sha256": last_warp.warp_sha256.upper(),
                    "map_sha256": last_warp.map_sha256.upper(),
                    "geometric_valid_sha256": last_warp.geometric_valid_sha256.upper(),
                    "domain_valid_sha256": last_warp.domain_valid_sha256.upper(),
                    "sampling_valid_sha256": last_warp.sampling_valid_sha256.upper(),
                })

            try:
                decoded_raw, raw_order = _decode_result(decoder, raw_rgb_bytes, expected="RGB")
                import numpy as np
                decoded_raw = np.asarray(decoded_raw)
                if decoded_raw.dtype != np.uint8 or decoded_raw.ndim != 3 or decoded_raw.shape[2] != 3:
                    raise _ConversionRowFailure("raw_decode_failed")
                raw_h, raw_w = decoded_raw.shape[:2]
                if raw_h * raw_w * 3 > MAX_RAW_RGB_BYTES:
                    raise _ConversionRowFailure("raw_dimension_mismatch")
                if raw_order == "BGR":
                    decoded_raw = decoded_raw[..., ::-1]
                elif raw_order != "RGB":
                    raise _ConversionRowFailure("raw_decode_failed")
                decoded_raw = np.ascontiguousarray(decoded_raw)
                decoded_raw_digest = hashlib.sha256(memoryview(decoded_raw).cast("B")).hexdigest().upper()
                row["decoded_raw_rgb_pixel_sha256"] = decoded_raw_digest
            except _ConversionRowFailure:
                raise
            except Exception as exc:
                row["failure_reason"] = "raw_decode_failed"
                raise _ConversionRowFailure("raw_decode_failed") from exc

            assert normalized_calibration is not None and last_warp is not None
            if (raw_h, raw_w) != (normalized_calibration.image_height, normalized_calibration.image_width):
                row["failure_reason"] = "raw_dimension_mismatch"
                raise _ConversionRowFailure("raw_dimension_mismatch")
            try:
                warped = camera.warp_rgb(decoded_raw, last_warp)
            except Exception as exc:
                row["failure_reason"] = "warp_failed"
                raise _ConversionRowFailure("warp_failed") from exc
            metadata = warped.metadata_json_bytes
            warped.export_metadata()
            if _sha256_bytes(metadata) != warped.metadata_sha256.upper():
                row["failure_reason"] = "warp_failed"
                raise _ConversionRowFailure("warp_failed")

            rgb_relative = f"rgb/{ordinal:06d}.png"
            valid_relative = f"sampling-valid/{ordinal:06d}.png"
            metadata_relative = f"conversion-metadata/{ordinal:06d}.json"
            try:
                rgb_encoded = _write_result(writer, warped.rgb, "RGB")
                valid_png = np.where(warped.sampling_valid, 255, 0).astype(np.uint8)
                valid_encoded = _write_result(writer, valid_png, "GRAY")
                if max(len(rgb_encoded), len(valid_encoded)) > MAX_ENCODED_IMAGE_BYTES:
                    raise ValueError("encoded PNG exceeds the 16 MiB resource limit")
                # Decode the exact encoded buffers with the same pinned decoder.
                verify_rgb, verify_rgb_order = _decode_result(decoder, rgb_encoded, expected="RGB")
                verify_rgb = _as_rgb(verify_rgb, verify_rgb_order, width=720, height=720)
                verify_valid, verify_valid_order = _decode_result(decoder, valid_encoded, expected="GRAY")
                verify_valid = _as_validity(verify_valid, verify_valid_order)
                if not np.array_equal(verify_rgb, warped.rgb) or not np.array_equal(verify_valid, warped.sampling_valid):
                    raise ValueError("PNG round trip did not preserve exact RGB/validity pixels")
                rgb_pixel_sha = hashlib.sha256(memoryview(np.ascontiguousarray(verify_rgb)).cast("B")).hexdigest().upper()
                valid_pixel_sha = hashlib.sha256(
                    memoryview(np.ascontiguousarray(verify_valid, dtype=np.uint8)).cast("B")).hexdigest().upper()
                rgb_file_sha = _sha256_bytes(rgb_encoded)
                valid_file_sha = _sha256_bytes(valid_encoded)
                _atomic_write(output / rgb_relative, rgb_encoded)
                _atomic_write(output / valid_relative, valid_encoded)
                _atomic_write(output / metadata_relative, metadata)
                row["output_rgb"] = {
                    "path": rgb_relative, "file_sha256": rgb_file_sha,
                    "byte_count": len(rgb_encoded), "pixel_sha256": rgb_pixel_sha,
                }
                row["sampling_valid"] = {
                    "path": valid_relative, "file_sha256": valid_file_sha,
                    "byte_count": len(valid_encoded), "pixel_sha256": valid_pixel_sha,
                }
                row["conversion_metadata"] = {
                    "path": metadata_relative, "file_sha256": _sha256_bytes(metadata),
                    "byte_count": len(metadata),
                }
                row["conversion_metadata_sha256"] = _sha256_bytes(metadata)
                row["conversion_state"] = "available"
                row["failure_reason"] = None
            except Exception as exc:
                for relative in (rgb_relative, valid_relative, metadata_relative):
                    try:
                        (output / relative).unlink()
                    except FileNotFoundError:
                        pass
                row["failure_reason"] = "artifact_write_failed"
                row["conversion_state"] = "unavailable"
                row["output_rgb"] = None
                row["sampling_valid"] = None
                row["conversion_metadata"] = None
                row["conversion_metadata_sha256"] = None
            finally:
                decoded_raw = None
                warped = None
                verify_rgb = None
                verify_valid = None
                valid_png = None
                rgb_encoded = None
                valid_encoded = None
                metadata = None
                raw_rgb_bytes = None
                calibration_bytes = None
        except _ConversionRowFailure as failure:
            row["conversion_state"] = "unavailable"
            row["failure_reason"] = failure.reason
            row["output_rgb"] = None
            row["sampling_valid"] = None
            row["conversion_metadata"] = None
            row["conversion_metadata_sha256"] = None
        row["capture_row_sha256"] = _row_digest(row)
        rows.append(row)

    table: dict[str, Any] = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "source_revision": inventory["source_revision"],
        "clip_id": inventory["clip_id"],
        "stream_id": stream_id,
        "archive_sha256": _valid_digest(inventory["archive_sha256"], "archive_sha256"),
        "source_inventory_sha256": inventory_sha,
        "output_camera": output_metadata,
        "output_camera_sha256": output_camera_sha,
        "coordinate_mode": COORDINATE_MODE,
        "converter": {
            "source_closure": closure,
            "runtime_decoder": runtime_decoder,
        },
        "coverage_complete": True,
        "planned_count": len(rows),
        "rows": rows,
    }
    raw_table = _canonical_json_bytes(table)
    if len(raw_table) > MAX_TABLE_BYTES:
        raise ValueError("capture table exceeds the 2 MiB resource limit")
    # The source-only inventory allowlist was checked above, so exact bytes are
    # safe to preserve as a private acquisition sidecar. It is not an inference
    # source_hashes entry and contains no evaluation annotations.
    inventory_relative = SOURCE_INVENTORY_RELATIVE
    _atomic_write(output / inventory_relative, raw_inventory)
    _atomic_write(output / "captures.json", raw_table)
    available = sum(row["conversion_state"] == "available" for row in rows)
    return ConversionReceipt(
        output_root=output,
        capture_table_path=output / "captures.json",
        capture_table_sha256=_sha256_bytes(raw_table),
        source_inventory_path=output / inventory_relative,
        source_inventory_sha256=inventory_sha,
        planned_count=len(rows),
        available_count=available,
        unavailable_count=len(rows) - available,
        rows=tuple(_freeze(row) for row in rows),
        table=_freeze(table),
    )


class _ConversionRowFailure(Exception):
    def __init__(self, reason: str):
        if reason not in _FAILURE_REASONS:
            raise ValueError("invalid conversion failure reason")
        self.reason = reason


def _canonical_receipt(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, bytes):
        document = _parse_json_document(value, limit=MAX_TABLE_BYTES, name=name)
        if _canonical_json_bytes(document) != value:
            raise ValueError(f"{name} bytes must be canonical UTF-8 JSON")
        return document
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON mapping or bounded bytes")
    document = _thaw(value)
    _finite_tree(document, name)
    return document


def capture_resource_key(asset_receipt: Any, output_camera: Mapping[str, Any],
                         recipe: Mapping[str, Any], source_closure: Mapping[str, str]) -> str:
    """Return a canonical content key for one capture smoke/bank resource."""

    receipt = _canonical_receipt(asset_receipt, "asset receipt")
    if not isinstance(output_camera, Mapping) or not isinstance(recipe, Mapping) or not isinstance(source_closure, Mapping):
        raise ValueError("output_camera, recipe, and source_closure must be JSON mappings")
    asset_sha = _contract_digest(receipt.get("asset_sha256"), "asset receipt asset_sha256")
    unit_receipt_sha = _contract_digest(
        receipt.get("unit_receipt_sha256"), "asset receipt unit_receipt_sha256")
    object_id = _nonnegative_int(receipt.get("object_id"), "asset receipt object_id")
    if "original_asset_sha256" in receipt:
        _contract_digest(receipt["original_asset_sha256"], "asset receipt original_asset_sha256")
    normalized_closure: dict[str, str] = {}
    for path, digest in source_closure.items():
        safe = _relative_path(path, "source closure path")
        normalized_closure[safe] = _contract_digest(digest, f"source_closure[{safe!r}]")
    if not normalized_closure or len({path.casefold() for path in normalized_closure}) != len(normalized_closure):
        raise ValueError("source closure must be nonempty and contain no path aliases")
    camera = _thaw(output_camera)
    if camera != OUTPUT_CAMERA:
        raise ValueError("capture resource key requires the fixed output camera")
    recipe_doc = _thaw(recipe)
    _finite_tree(recipe_doc, "recipe")
    key_material = {
        "schema": "quality-capture-resource-key-v1",
        "asset_sha256": asset_sha,
        "unit_receipt_sha256": unit_receipt_sha,
        "asset_receipt": receipt,
        "object_id": object_id,
        "coordinate_mode": COORDINATE_MODE,
        "output_camera": camera,
        "render_flags": recipe_doc.get("render_flags", {}),
        "recipe": recipe_doc,
        "source_closure": normalized_closure,
    }
    return _canonical_digest(key_material)


def verify_stage_resource_sidecar(
    path: str | os.PathLike[str],
    *,
    expected_key: str,
    expected_recipe: Mapping[str, Any],
    expected_source_closure: Mapping[str, str],
) -> StageResourceDescriptor:
    """Authenticate a smoke/bank sidecar and its artifact before deserialization."""

    key = _contract_digest(expected_key, "expected_key")
    sidecar = Path(path).resolve(strict=True)
    if not sidecar.is_file() or sidecar.stat().st_size > MAX_TABLE_BYTES:
        raise CaptureIntegrityError("stage sidecar must be a bounded regular file")
    parts = [part.casefold() for part in sidecar.parts]
    try:
        resource_index = parts.index("capture-resources")
    except ValueError as exc:
        raise CaptureIntegrityError("stage sidecar must live under CACHE/capture-resources") from exc
    if resource_index == 0 or parts[resource_index - 1] not in ("cache", ".cache"):
        raise CaptureIntegrityError("stage sidecar must be under the cache capture-resources directory")
    if resource_index + 1 >= len(parts) or sidecar.parts[resource_index + 1].casefold() != key.casefold():
        raise CaptureIntegrityError("stage sidecar path does not use the expected content key")
    root = sidecar.parent.resolve(strict=True)
    if root.name.casefold() != key.casefold():
        raise CaptureIntegrityError("stage sidecar must be directly inside its keyed resource directory")
    raw, _ = _read_bounded(sidecar, MAX_TABLE_BYTES)
    document = _parse_json_document(raw, limit=MAX_TABLE_BYTES, name="stage sidecar")
    allowed = frozenset({
        "schema_version", "resource_kind", "resource_key", "recipe", "dimensions",
        "coordinate_mode", "artifact", "runtime", "source_closure",
    })
    _strict_keys(document, allowed, "stage sidecar")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise CaptureIntegrityError("stage sidecar schema_version must be 1")
    if _contract_digest(document.get("resource_key"), "stage resource_key") != key:
        raise CaptureIntegrityError("stage sidecar resource key mismatch")
    if document.get("coordinate_mode") != COORDINATE_MODE:
        raise CaptureIntegrityError("stage sidecar coordinate mode mismatch")
    expected_recipe_doc = _thaw(expected_recipe)
    _finite_tree(expected_recipe_doc, "expected_recipe")
    if document.get("recipe") != expected_recipe_doc:
        raise CaptureIntegrityError("stage sidecar recipe mismatch")
    dimensions = document.get("dimensions")
    if (not isinstance(dimensions, list) or not 1 <= len(dimensions) <= 8 or
            any(type(size) is not int or size <= 0 or size > 1_000_000 for size in dimensions)):
        raise CaptureIntegrityError("stage sidecar dimensions must be a bounded positive integer shape")
    dimension_product = math.prod(dimensions)
    if dimension_product > 8 * 1024 * 1024 * 1024:
        raise CaptureIntegrityError("stage sidecar dimensions exceed the 8 GiB artifact bound")
    closure = document.get("source_closure")
    if not isinstance(closure, dict) or closure != _thaw(expected_source_closure):
        raise CaptureIntegrityError("stage sidecar source closure mismatch")
    for source, digest in closure.items():
        _relative_path(source, "stage source closure path")
        _contract_digest(digest, f"stage source_closure[{source!r}]")
    _strict_keys(document.get("artifact"), frozenset({"path", "sha256", "byte_count"}), "stage artifact")
    _strict_keys(document.get("runtime"), frozenset({"device", "build"}), "stage runtime")
    if not isinstance(document["resource_kind"], str) or not document["resource_kind"].strip():
        raise CaptureIntegrityError("stage resource_kind must be nonempty")
    if not isinstance(document["runtime"]["device"], str) or not document["runtime"]["device"].strip() or not isinstance(document["runtime"]["build"], str) or not document["runtime"]["build"].strip():
        raise CaptureIntegrityError("stage runtime must declare device and build")
    artifact_relative = _relative_path(document["artifact"]["path"], "stage artifact.path")
    expected_artifact_sha = _contract_digest(document["artifact"]["sha256"], "stage artifact.sha256")
    artifact_bytes = _positive_int(document["artifact"]["byte_count"], "stage artifact.byte_count")
    artifact_path = root.joinpath(*artifact_relative.split("/"))
    try:
        artifact_path = artifact_path.resolve(strict=True)
        artifact_path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CaptureIntegrityError("stage artifact escapes or is missing from keyed resource directory") from exc
    if not artifact_path.is_file():
        raise CaptureIntegrityError("stage artifact is not a regular file")
    actual_artifact_sha, actual_count = _stream_hash(
        artifact_path, expected_bytes=artifact_bytes, maximum=8 * 1024 * 1024 * 1024)
    if actual_artifact_sha != expected_artifact_sha:
        raise CaptureIntegrityError("stage artifact hash mismatch")
    return StageResourceDescriptor(
        sidecar_path=sidecar,
        sidecar_sha256=_sha256_bytes(raw),
        resource_key=key,
        resource_kind=document["resource_kind"],
        artifact_path=artifact_path,
        artifact_sha256=actual_artifact_sha,
        artifact_byte_count=actual_count,
        recipe=_freeze(document["recipe"]),
        source_closure=_freeze(closure),
        descriptor=_freeze(document),
    )


def _main_convert(args: argparse.Namespace) -> int:
    receipt = convert_capture_sequence(
        args.source_root,
        args.inventory,
        args.output_root,
        expected_inventory_sha256=args.expected_inventory_sha256,
    )
    print(json.dumps({
        "capture_table_path": str(receipt.capture_table_path),
        "capture_table_sha256": receipt.capture_table_sha256,
        "source_inventory_path": str(receipt.source_inventory_path),
        "source_inventory_sha256": receipt.source_inventory_sha256,
        "planned_count": receipt.planned_count,
        "available_count": receipt.available_count,
        "unavailable_count": receipt.unavailable_count,
        "result": "conversion_preprocessing_only",
    }, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="convert an authenticated source inventory to PNG sequence resources")
    convert.add_argument("--source-root", required=True)
    convert.add_argument("--inventory", required=True)
    convert.add_argument("--output-root", required=True)
    convert.add_argument("--expected-inventory-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        return _main_convert(args)
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":  # pragma: no cover - explicit private conversion CLI
    raise SystemExit(main())
