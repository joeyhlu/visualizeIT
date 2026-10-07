"""Bind and screen the twelve frozen R8 actual-source rows.

This is a pure-return adapter. It reads pinned R1/R3 metadata, six strictly
preflighted SOURCE packets, and the pinned CPU GLB geometry. It never reads
counterpart packets, query pixels, evaluation truth, or closed R5/R6 outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ADAPTER_SCHEMA_VERSION = "r8-actual-source-adapter-v1"
MAX_SHARED_WORKING_BYTES = 128 * 1024**2
MAX_WALL_SECONDS = 900.0
MAX_RECEIPT_BYTES = 8 * 1024**2
HASH_CHUNK_BYTES = 1024**2
REPORT_RESERVE_BYTES = 2 * 1024**2
GLB_JSON_MAX_BYTES = 8 * 1024**2

# Paths are relative to .cache/model-quality/. Values are deliberately mutable
# for isolated tests; production defaults are the hashes in the frozen design.
PINNED_FILE_SPECS: dict[str, dict[str, object]] = {
    "diagnostics/bottle-identity-v2/capture.json": {
        "bytes": 615_937,
        "sha256": "33efa32778c4094a5df0690d49658cb3bfb0be522204b3dc4e1279d9f0d03063",
    },
    "diagnostics/bottle-zero-view-calibration-v3/zero_view.json": {
        "bytes": 473_094,
        "sha256": "dccbff28e6ea84693fe2a64d40006304f318532daefcff955646827b6129819d",
    },
    "inputs/ranch/input.json": {
        "bytes": 11_742,
        "sha256": "f012f28918534f2a7ac3efd59841a347a8bad384d9585130db88eaac56377d5d",
    },
    "inputs/ranch/object.glb": {
        "bytes": 2_913_424,
        "sha256": "ed3ac3767202c8da56d30051d766bf7d444d3d356c30671f4bbb2a73e3a1b321",
    },
    "diagnostics/bottle-source-observability-pose-ablation-v1/pose_ablation.json": {
        "bytes": 872_845,
        "sha256": "a394bf54f51ddf527549ee5b3b6c7e8f623546b2c277a49051df2ce1c4d95259",
    },
    "diagnostics/bottle-r5-parent-v1/terminal.json": {
        "bytes": 33_103,
        "sha256": "7d193b09d6d9dda241e039fce757b0a19d5c82e0fc6bc84b287d73e8915392ae",
    },
    "diagnostics/bottle-patch-pose-calibration-v2/calibration.json": {
        "bytes": 1_531_789,
        "sha256": "857a52df0982adee17388e3de5cb82836ea2bbd0249ef878c4741cb909c341ad",
    },
    "diagnostics/bottle-r6-parent-v2/terminal.json": {
        "bytes": 503_612,
        "sha256": "214ed8c051d7865d46bb254821a662dcb85b1a0e776d88ecc727e5da46c97ccf",
    },
}

# The adapter depends on exactly reviewed R8 cores plus their source-only R5/R7
# implementations and CPU GLB reader. Rechecking these pins brackets every
# source read and capacity call.
PINNED_CORE_SPECS: dict[str, str] = {
    "bench/quality_bottle_source_bindings.py":
        "739e8d0cf770cc35cfe7fbf134972cbff6163cc7dfb743d6157c4833d227aa4a",
    "bench/quality_bottle_source_packet.py":
        "8dcaac8c79c1cc560bf0634a17d0c3cd46bbc44a52843a5d7917854286e6201f",
    "bench/quality_bottle_source_capacity.py":
        "8196e0bc188971affa15aa2361034d456cf20aa5756f15ab3571678cd456eef1",
    "bench/quality_bottle_source_reservation.py":
        "679876b7bca979bca6ab61e96e3d9ff90de3d0fba2f261da3cadeba39ff79b39",
    "bench/quality_bottle_reservation_selection.py":
        "753d51cc4e9ed1ef1b40e5bda64d9f83bcbf46d24432f42ec8edc9d4e7d5b944",
    "bench/quality_bottle_pose_ablation.py":
        "54bdd2f4ca4fb82551492e5397ab42cea2e464c5dad1f83360f7b23d33926542",
    "bench/quality_bottle_identity_audit.py":
        "954d4b4e3a460187963894988cce53a85a077abaf07e085a0833131d14418f6a",
    "bench/quality_bottle_joint_prerequisite.py":
        "cf74fdf73a2e6f36a1ae4bf783091ac23e4b2fd5364ce5b5e721304cb7e1684a",
    "bench/glb_model.py":
        "71f2d0461d97cbdebea2f2f1acfd4719ef14a1f94c17db4ec3db14659a1ece5e",
    "docs/r8-source-adapter-spec.md":
        "e6c7df9b9b3d61fcfb1b845f94006573a4dba46575bc58e6e23ae894625426e8",
    ".cache/bottle-source-reservation-r8-spec-v2.md":
        "32758dda6ad550a5edd94f5354b7e8401ff206a63b884e8c88ab37f71ee6d9cb",
}

CANONICAL_PLAN_IDS = tuple(
    f"{prefix}-{frame}-{suffix}"
    for frame in (10, 50, 100)
    for prefix, suffix in (
        ("syn", "q8-rgb-t0-rgb"),
        ("zero", "full"),
        ("zero", "clipped"),
        ("syn", "q8-rgb-t180-rgb"),
    )
)
EXPECTED_VIEWS = tuple(
    (frame, offset, f"template-{frame:04d}-{offset:03d}")
    for frame in (10, 50, 100)
    for offset in (0, 180)
)
COUNTERPART_ARRAYS = (
    "observed_crop_mask", "crop_k", "crop_from_native", "native_k",
)
SOURCE_ARRAY_NAMES = (
    "template_rgb", "template_gray_rgb", "template_depth_mm", "template_mask",
    "source_indices", "source_pixels_xy", "source_points_object_m",
    "observed_crop_mask", "crop_k", "crop_from_native", "native_k",
    "seed_pose_m", "template_pose_m",
)
CAPACITY_SOURCE_KEYS = frozenset(set(SOURCE_ARRAY_NAMES) - {"template_gray_rgb", "seed_pose_m"})
PRIMARY_GATE_NAMES = (
    "witnesses_ge_8", "fit_ge_24", "bank_ge_1",
    "qh_possible_cells_ge_3", "qh_maximum_bbox_hull_ge_0_12",
)


class SourceAdapterError(ValueError):
    """A terminal adapter failure with all twelve frozen rows accounted for."""

    def __init__(self, message: str, receipt: Mapping[str, object]):
        super().__init__(message)
        self.receipt = dict(receipt)


def _check_deadline(deadline: float, phase: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"R8 source adapter exceeded its shared 900-second deadline at {phase}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_copy(value: object, label: str) -> object:
    try:
        return json.loads(_canonical_json(value).decode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not compact JSON-safe evidence") from exc


def _deep_size(value: object, deadline: float, label: str) -> int:
    """Estimate retained Python-object bytes without following cycles twice."""
    total = 0
    seen: set[int] = set()
    stack = [value]
    visited = 0
    while stack:
        item = stack.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        visited += 1
        if visited % 2048 == 0:
            _check_deadline(deadline, f"{label} resident-size accounting")
        if isinstance(item, Mapping):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
    return total


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _lower_sha(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _verify_file_pin(
    path: object,
    expected_bytes: int,
    expected_sha256: str,
    deadline_monotonic: float | None = None,
    label: str = "pinned file",
) -> dict[str, object]:
    """Stream-check a file pin; never parse a report passed through this helper."""
    deadline = deadline_monotonic
    if deadline is not None:
        _check_deadline(deadline, f"{label} pin check")
    target = Path(path)
    expected_bytes = _nonnegative_int(expected_bytes, f"{label} byte pin")
    expected_sha256 = _lower_sha(expected_sha256, f"{label} SHA pin")
    try:
        before = target.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError(f"{label} type or byte count differs from its frozen pin")
        digest = hashlib.sha256()
        byte_count = 0
        with target.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                    before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValueError(f"{label} changed before its pinned read")
            while True:
                if deadline is not None:
                    _check_deadline(deadline, f"{label} pin hashing")
                chunk = stream.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
                del chunk
            after = os.fstat(stream.fileno())
        if (byte_count != expected_bytes or digest.hexdigest() != expected_sha256 or
                (opened.st_ino, opened.st_size, opened.st_mtime_ns) !=
                (after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError(f"{label} changed or differs from its frozen SHA256 pin")
    except OSError as exc:
        raise ValueError(f"{label} could not be read for pin verification") from exc
    return {"bytes": expected_bytes, "sha256": expected_sha256}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _verify_core_pins(repo_root: Path, deadline: float) -> dict[str, dict[str, object]]:
    checked: dict[str, dict[str, object]] = {}
    for relative, expected_hash in PINNED_CORE_SPECS.items():
        _check_deadline(deadline, "core source pin verification")
        path = repo_root.joinpath(*Path(relative).parts)
        size = path.stat().st_size
        checked[relative] = _verify_file_pin(
            path, size, expected_hash, deadline, f"core source {relative}",
        )
    return checked


def _model_quality_root(input_root: object) -> tuple[Path, Path]:
    input_path = Path(input_root).resolve(strict=True)
    if not input_path.is_dir() or input_path.name != "ranch" or input_path.parent.name != "inputs":
        raise ValueError("input_root must be the frozen inputs/ranch directory")
    model_root = input_path.parent.parent
    return model_root, input_path


def _resolve_input_roots(capture_root: object, zero_root: object, input_root: object) -> tuple[Path, Path, Path, Path]:
    model_root, input_path = _model_quality_root(input_root)
    capture_path = Path(capture_root).resolve(strict=True)
    zero_path = Path(zero_root).resolve(strict=True)
    expected_capture = model_root / "diagnostics" / "bottle-identity-v2"
    expected_zero = model_root / "diagnostics" / "bottle-zero-view-calibration-v3"
    if capture_path != expected_capture or zero_path != expected_zero:
        raise ValueError("capture_root and zero_root must be the frozen R1/R3 directories for input_root")
    if not capture_path.is_dir() or not zero_path.is_dir():
        raise ValueError("capture_root and zero_root must be existing directories")
    return model_root, capture_path, zero_path, input_path


def _verify_input_pins(model_root: Path, deadline: float) -> dict[str, dict[str, object]]:
    checked: dict[str, dict[str, object]] = {}
    for relative, specification in PINNED_FILE_SPECS.items():
        _check_deadline(deadline, "immutable input pin verification")
        if not isinstance(specification, Mapping):
            raise ValueError(f"Pinned-file specification for {relative} is malformed")
        expected_bytes = _nonnegative_int(specification.get("bytes"), f"{relative} byte pin")
        expected_sha = _lower_sha(specification.get("sha256"), f"{relative} SHA pin")
        path = model_root.joinpath(*Path(relative).parts)
        checked[relative] = _verify_file_pin(
            path, expected_bytes, expected_sha, deadline, f"input {relative}",
        )
    return checked


def _no_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Pinned JSON metadata has a duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Pinned JSON metadata contains nonstandard numeric constant {value}")


def _read_json_metadata(path: Path, deadline: float, label: str) -> dict[str, object]:
    _check_deadline(deadline, f"{label} metadata parse")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    _check_deadline(deadline, f"{label} metadata parse")
    return value


def _array_metadata(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"shape", "dtype", "sha256"}:
        raise ValueError(f"{label} must contain exactly shape, dtype, and sha256")
    shape = value.get("shape")
    if (not isinstance(shape, list) or not shape or
            any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)):
        raise ValueError(f"{label}.shape is invalid")
    dtype = value.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{label}.dtype is invalid")
    return {"shape": list(shape), "dtype": dtype, "sha256": _lower_sha(value.get("sha256"), f"{label}.sha256")}


def _validate_bindings(binding: object, plans: list[dict[str, object]], binder: object) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    if not isinstance(binding, Mapping):
        raise ValueError("Frozen metadata binder did not return a mapping")
    if (binding.get("schema_version") != "r8-source-metadata-bindings-v1" or
            binding.get("metadata_bindings_verified") is not True or
            binding.get("counterpart_context_bindings_verified") is not True or
            binding.get("source_packet_file_pins_verified") is not False or
            binding.get("source_packet_arrays_decoded") is not False or
            binding.get("matching_started") is not False or
            binding.get("fitting_started") is not False):
        raise ValueError("Frozen metadata binder receipt has an unexpected verification state")
    template_rows = binding.get("template_bindings")
    plan_rows = binding.get("plan_bindings")
    if not isinstance(template_rows, list) or not isinstance(plan_rows, list):
        raise ValueError("Frozen metadata binder omitted its six templates or twelve plans")
    if len(template_rows) != 6 or len(plan_rows) != 12:
        raise ValueError("Frozen metadata binder did not account for exactly six templates and twelve plans")
    expected_plans = tuple(plan["condition_id"] for plan in plans)
    if expected_plans != CANONICAL_PLAN_IDS:
        raise ValueError("R5 canonical plan IDs differ from the frozen twelve-row order")
    expected_by_template = {context: (frame, offset) for frame, offset, context in EXPECTED_VIEWS}
    source_views: list[dict[str, object]] = []
    namespaces: set[tuple[str, str]] = set()
    template_by_context: dict[str, dict[str, object]] = {}
    for index, (entry, expected) in enumerate(zip(template_rows, EXPECTED_VIEWS)):
        frame_id, offset, context_id = expected
        if not isinstance(entry, Mapping):
            raise ValueError(f"Template binding {index} is not a mapping")
        if (entry.get("context_id") != context_id or entry.get("frame_id") != frame_id or
                entry.get("template_offset_deg") != offset):
            raise ValueError("Frozen metadata binder changed the fixed six-view order")
        source_entry = entry.get("source_entry")
        namespace = entry.get("source_namespace")
        if not isinstance(source_entry, Mapping) or not isinstance(namespace, Mapping):
            raise ValueError(f"Template binding {context_id} omitted its sanitized source entry/namespace")
        if (set(source_entry) != {"context_id", "role", "frame_id", "template_offset_deg",
                                  "path", "bytes", "sha256", "arrays", "geometry_hashes"} or
                not isinstance(source_entry.get("arrays"), Mapping) or
                set(source_entry.get("arrays", {})) != set(SOURCE_ARRAY_NAMES) or
                source_entry.get("context_id") != context_id or
                source_entry.get("role") != "template" or source_entry.get("frame_id") != frame_id or
                source_entry.get("template_offset_deg") != offset or
                source_entry.get("path") != f"packets/contexts/{context_id}.npz"):
            raise ValueError(f"Template binding {context_id} is not the exact sanitized source entry")
        packet_bytes = _nonnegative_int(source_entry.get("bytes"), f"{context_id} packet bytes")
        packet_sha = _lower_sha(source_entry.get("sha256"), f"{context_id} packet SHA")
        source_arrays = source_entry.get("arrays")
        checked_arrays = {
            key: _array_metadata(source_arrays[key], f"{context_id}.{key}")
            for key in SOURCE_ARRAY_NAMES
        }
        entry_meta = entry.get("array_metadata")
        if not isinstance(entry_meta, Mapping) or set(entry_meta) != set(COUNTERPART_ARRAYS):
            raise ValueError(f"Template binding {context_id} omitted exact camera/mask metadata")
        checked_counterparts = {
            key: _array_metadata(entry_meta[key], f"{context_id}.{key}")
            for key in COUNTERPART_ARRAYS
        }
        for key in COUNTERPART_ARRAYS:
            if checked_counterparts[key] != checked_arrays[key]:
                raise ValueError(f"Template binding {context_id} camera/mask metadata changed")
        source_context = namespace.get("context_id")
        source_rgb_sha = _lower_sha(namespace.get("source_rgb_sha256"), f"{context_id} source RGB SHA")
        if source_context != context_id or source_rgb_sha != checked_arrays["template_rgb"]["sha256"]:
            raise ValueError(f"Template binding {context_id} has a malformed image-qualified namespace")
        namespace_key = (source_context, source_rgb_sha)
        if namespace_key in namespaces:
            raise ValueError("Distinct 000/180 source views aliased one image namespace")
        namespaces.add(namespace_key)
        geometry_group = entry.get("geometry_group")
        if geometry_group != f"frame-{frame_id}":
            raise ValueError(f"Template binding {context_id} geometry group changed")
        geometry_hashes = source_entry.get("geometry_hashes")
        if (not isinstance(geometry_hashes, Mapping) or
                geometry_hashes.get("depth_mm") != checked_arrays["template_depth_mm"]["sha256"] or
                geometry_hashes.get("mask") != checked_arrays["template_mask"]["sha256"]):
            raise ValueError(f"Template binding {context_id} geometry hashes do not match source metadata")
        source_view = {
            "context_id": context_id,
            "frame_id": frame_id,
            "template_offset_deg": offset,
            "geometry_group": geometry_group,
            "source_namespace": {"context_id": source_context, "source_rgb_sha256": source_rgb_sha},
            "source_entry": dict(source_entry),
            "source_array_metadata": checked_arrays,
            "counterpart_array_metadata": checked_counterparts,
            "counterpart_array_metadata_sha256": _canonical_sha256(checked_counterparts),
            "packet_bytes": packet_bytes,
            "packet_sha256": packet_sha,
        }
        source_views.append(source_view)
        template_by_context[context_id] = source_view

    row_by_id: dict[str, dict[str, object]] = {}
    for index, (row, plan) in enumerate(zip(plan_rows, plans)):
        condition_id = plan["condition_id"]
        if not isinstance(row, Mapping) or row.get("condition_id") != condition_id:
            raise ValueError(f"Frozen plan binding {index} changed the canonical row order")
        frame_id = plan["frame_id"]
        offset = plan["template_offset_deg"]
        template_context = f"template-{frame_id:04d}-{offset:03d}"
        observed_context = f"real-frame-{frame_id:04d}"
        synthetic_context = f"synthetic-query-{frame_id:04d}-008"
        expected_refs = (
            {"query": synthetic_context, "template": template_context, "observed_frame": observed_context}
            if plan["kind"] in ("synthetic_positive", "synthetic_negative")
            else {"source_template": template_context, "competitor_template": f"template-{frame_id:04d}-180",
                  "observed_frame": observed_context}
        )
        refs = row.get("context_refs")
        if not isinstance(refs, Mapping) or dict(refs) != expected_refs:
            raise ValueError(f"Plan binding {condition_id} context refs differ from the frozen plan")
        if (row.get("frame_id") != frame_id or row.get("template_offset_deg") != offset or
                row.get("template_context_id") != template_context or
                row.get("observed_frame_context_id") != observed_context or
                row.get("source_namespace") != template_by_context[template_context]["source_namespace"] or
                row.get("plan_binding_verified") is not True or
                row.get("counterpart_binding_verified") is not True):
            raise ValueError(f"Plan binding {condition_id} failed its exact source/counterpart identity check")
        if row.get("kind") != plan["kind"]:
            raise ValueError(f"Plan binding {condition_id} kind differs from the frozen R5 plan")
        if plan["kind"] in ("synthetic_positive", "synthetic_negative"):
            expected_counterpart_contexts = {"observed_frame", "synthetic_query"}
            if row.get("synthetic_query_context_id") != synthetic_context:
                raise ValueError(f"Plan binding {condition_id} lost its synthetic counterpart")
        else:
            expected_counterpart_contexts = {"observed_frame"}
            if row.get("synthetic_query_context_id") is not None:
                raise ValueError(f"Self-control binding {condition_id} has a synthetic counterpart")
        mask_records = row.get("counterpart_mask_metadata")
        digest_records = row.get("counterpart_array_metadata_sha256")
        if (not isinstance(mask_records, Mapping) or set(mask_records) != expected_counterpart_contexts or
                not isinstance(digest_records, Mapping) or set(digest_records) != expected_counterpart_contexts):
            raise ValueError(f"Plan binding {condition_id} omitted its exact counterpart metadata")
        safe_masks = {
            key: _array_metadata(value, f"{condition_id}.{key}.observed_crop_mask")
            for key, value in mask_records.items()
        }
        for counterpart in expected_counterpart_contexts:
            if safe_masks[counterpart] != template_by_context[template_context]["source_array_metadata"]["observed_crop_mask"]:
                raise ValueError(f"Plan binding {condition_id} observed mask metadata differs from its source")
            _lower_sha(digest_records[counterpart], f"{condition_id}.{counterpart} metadata digest")
            expected_digest = _canonical_sha256(
                template_by_context[template_context]["counterpart_array_metadata"]
            )
            if digest_records[counterpart] != expected_digest:
                raise ValueError(f"Plan binding {condition_id} camera/mask metadata digest differs from its template")
        checked_row = {
            "plan": dict(plan),
            "binding": dict(row),
            "counterpart_mask_metadata": safe_masks,
            "expected_counterpart_contexts": expected_counterpart_contexts,
        }
        row_by_id[condition_id] = checked_row
    return source_views, [row_by_id[condition_id] for condition_id in CANONICAL_PLAN_IDS], 0


def _parse_glb_and_estimate_reader_peak(path: Path, glb_bytes: int, deadline: float) -> dict[str, int]:
    """Read only the pinned GLB JSON chunk and bound the later geometry reader."""
    _check_deadline(deadline, "GLB metadata preflight")
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("Pinned GLB header is truncated")
        magic, version, length = struct.unpack("<III", header)
        if magic != 0x46546C67 or version != 2 or length != glb_bytes:
            raise ValueError("Pinned mesh is not the complete GLB 2.0 file")
        offset = 12
        document: dict[str, object] | None = None
        binary_count = 0
        chunk_count = 0
        while offset < length:
            _check_deadline(deadline, "GLB metadata chunk walk")
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError("Pinned GLB chunk header is truncated")
            chunk_size, chunk_type = struct.unpack("<II", chunk_header)
            offset += 8
            if chunk_size > length - offset:
                raise ValueError("Pinned GLB chunk exceeds its file boundary")
            chunk_count += 1
            if chunk_type == 0x4E4F534A:
                if document is not None or chunk_size > GLB_JSON_MAX_BYTES:
                    raise ValueError("Pinned GLB has duplicate or oversized JSON metadata")
                payload = stream.read(chunk_size)
                if len(payload) != chunk_size:
                    raise ValueError("Pinned GLB JSON chunk is truncated")
                document_value = json.loads(
                    payload.decode("utf-8"), object_pairs_hook=_no_duplicate_pairs,
                    parse_constant=_reject_json_constant,
                )
                del payload
                if not isinstance(document_value, dict):
                    raise ValueError("Pinned GLB JSON root must be an object")
                document = document_value
            else:
                if chunk_type == 0x004E4942:
                    binary_count += 1
                stream.seek(chunk_size, os.SEEK_CUR)
            offset += chunk_size
        if offset != length or chunk_count < 2 or document is None or binary_count != 1:
            raise ValueError("Pinned GLB lacks exactly one JSON and one embedded BIN chunk")

    if any("uri" in buffer for buffer in document.get("buffers", [])):
        raise ValueError("Pinned GLB has an external buffer")
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    meshes = document.get("meshes")
    nodes = document.get("nodes")
    scenes = document.get("scenes")
    if not all(isinstance(value, list) for value in (accessors, views, meshes, nodes, scenes)):
        raise ValueError("Pinned GLB geometry metadata is incomplete")

    def accessor_info(index: object, expected_type: str, label: str) -> tuple[int, int]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(accessors):
            raise ValueError(f"Pinned GLB {label} accessor is invalid")
        accessor = accessors[index]
        if not isinstance(accessor, Mapping) or "sparse" in accessor or accessor.get("normalized"):
            raise ValueError(f"Pinned GLB {label} accessor is unsupported")
        if accessor.get("type") != expected_type:
            raise ValueError(f"Pinned GLB {label} accessor has the wrong component shape")
        component_type = accessor.get("componentType")
        if component_type not in (5121, 5123, 5125, 5126):
            raise ValueError(f"Pinned GLB {label} accessor has an unsupported component type")
        count = accessor.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 300_000:
            raise ValueError(f"Pinned GLB {label} accessor count is outside the CPU reader bound")
        view_index = accessor.get("bufferView")
        if (isinstance(view_index, bool) or not isinstance(view_index, int) or
                not 0 <= view_index < len(views) or not isinstance(views[view_index], Mapping) or
                views[view_index].get("buffer", 0) != 0):
            raise ValueError(f"Pinned GLB {label} accessor buffer view is invalid")
        return count, component_type

    scene_index = document.get("scene", 0)
    if isinstance(scene_index, bool) or not isinstance(scene_index, int) or not 0 <= scene_index < len(scenes):
        raise ValueError("Pinned GLB default scene is invalid")
    scene_roots = scenes[scene_index].get("nodes") if isinstance(scenes[scene_index], Mapping) else None
    if not isinstance(scene_roots, list):
        raise ValueError("Pinned GLB scene has no root nodes")
    total_vertices = 0
    total_index_values = 0
    visiting_nodes = 0

    def visit_node(node_index: object, ancestors: frozenset[int], depth: int) -> None:
        nonlocal total_vertices, total_index_values, visiting_nodes
        if (isinstance(node_index, bool) or not isinstance(node_index, int) or
                not 0 <= node_index < len(nodes)):
            raise ValueError("Pinned GLB scene references an invalid node")
        if node_index in ancestors:
            raise ValueError("Pinned GLB node graph contains a cycle")
        if depth > 1024:
            raise ValueError("Pinned GLB node graph exceeds the reader recursion bound")
        node = nodes[node_index]
        if not isinstance(node, Mapping):
            raise ValueError("Pinned GLB node record is malformed")
        visiting_nodes += 1
        if visiting_nodes > 100_000:
            raise ValueError("Pinned GLB node visitation exceeds the reader bound")
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            if isinstance(mesh_index, bool) or not isinstance(mesh_index, int) or not 0 <= mesh_index < len(meshes):
                raise ValueError("Pinned GLB node references an invalid mesh")
            mesh_record = meshes[mesh_index]
            primitives = mesh_record.get("primitives") if isinstance(mesh_record, Mapping) else None
            if not isinstance(primitives, list):
                raise ValueError("Pinned GLB mesh has no primitives")
            for primitive in primitives:
                if not isinstance(primitive, Mapping) or primitive.get("mode", 4) != 4 or "extensions" in primitive:
                    raise ValueError("Pinned GLB contains unsupported non-triangle geometry")
                attributes = primitive.get("attributes")
                if not isinstance(attributes, Mapping):
                    raise ValueError("Pinned GLB primitive has no attribute map")
                p_count, _ = accessor_info(attributes.get("POSITION"), "VEC3", "POSITION")
                n_count, _ = accessor_info(attributes.get("NORMAL"), "VEC3", "NORMAL")
                uv_count, _ = accessor_info(attributes.get("TEXCOORD_0"), "VEC2", "TEXCOORD_0")
                index_count, _ = accessor_info(primitive.get("indices"), "SCALAR", "indices")
                if p_count != n_count or p_count != uv_count or index_count % 3:
                    raise ValueError("Pinned GLB primitive attribute or triangle counts disagree")
                total_vertices += p_count
                total_index_values += index_count
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ValueError("Pinned GLB child-node list is malformed")
        next_ancestors = ancestors | {node_index}
        for child in children:
            visit_node(child, next_ancestors, depth + 1)

    for root_node in scene_roots:
        visit_node(root_node, frozenset(), 0)
    if total_vertices < 1 or total_index_values < 3:
        raise ValueError("Pinned GLB contains no triangle geometry")
    return {
        "vertex_count": total_vertices,
        "index_value_count": total_index_values,
        "triangle_count": total_index_values // 3,
        # read_glb copies the file/BIN, per-primitive accessors, transformed
        # arrays, concatenations, and final Mesh arrays; this bound is checked
        # before invoking it.
        "reader_peak_estimate_bytes": (
            8 * glb_bytes + 256 * total_vertices + 64 * total_index_values + 8 * 1024**2
        ),
    }


def _typed_array_sha256(array: Any, label: str) -> str:
    import numpy as np
    if (not isinstance(array, np.ndarray) or array.dtype.hasobject or
            not array.flags.c_contiguous):
        raise ValueError(f"Native mesh {label} must be a contiguous numeric array")
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}) + b"\0"
    digest = hashlib.sha256(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_capacity_record(
    result: object,
    *,
    context_id: str,
    source_namespace: Mapping[str, object],
    packet_sha256: str,
    mesh_sha256: str,
    deadline: float,
) -> dict[str, object]:
    if not isinstance(result, Mapping):
        raise ValueError(f"Capacity helper for {context_id} did not return a mapping")
    record = _json_copy(result, f"{context_id} capacity receipt")
    if not isinstance(record, dict):
        raise ValueError(f"Capacity helper for {context_id} returned a non-object receipt")
    if (record.get("capacity_only") is not True or
            record.get("full_28_case_screen_complete") is not False or
            record.get("counterpart_context_bindings_verified") is not False or
            record.get("source_plan_bindings_verified") is not False or
            record.get("matching_started") is not False or record.get("fitting_started") is not False or
            record.get("query_rgb_or_evaluator_truth_read") is not False or
            record.get("query_observed_mask_origin") != "caller_supplied_observed" or
            record.get("source_context_id") != context_id or
            record.get("packet_sha256") != packet_sha256 or record.get("mesh_sha256") != mesh_sha256):
        raise ValueError(f"Capacity helper for {context_id} changed a source-only boundary or pin")
    claims = record.get("claims")
    if (not isinstance(claims, Mapping) or claims.get("fullscreen") is not False or
            claims.get("accuracy") is not False or claims.get("pose") is not False):
        raise ValueError(f"Capacity helper for {context_id} emitted an unsupported claim")
    gates = record.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(PRIMARY_GATE_NAMES):
        raise ValueError(f"Capacity helper for {context_id} omitted the exact five capacity gates")
    expected_gates = {
        "witnesses_ge_8": _nonnegative_int(record.get("witness_count"), "witness count") >= 8,
        "fit_ge_24": _nonnegative_int(record.get("fit_count"), "fit count") >= 24,
        "bank_ge_1": _nonnegative_int(record.get("bank_anchor_count"), "bank anchor count") >= 1,
        "qh_possible_cells_ge_3": _nonnegative_int(record.get("qh_possible_cell_count"), "QH possible-cell count") >= 3,
    }
    hull = record.get("qh_maximum_bbox_hull_fraction")
    if (isinstance(hull, bool) or not isinstance(hull, (int, float)) or
            not math.isfinite(float(hull)) or not 0 <= float(hull) <= 1):
        raise ValueError(f"Capacity helper for {context_id} emitted an invalid QH hull bound")
    expected_gates["qh_maximum_bbox_hull_ge_0_12"] = float(hull) >= 0.12
    if any(type(gates.get(name)) is not bool or gates.get(name) != expected
           for name, expected in expected_gates.items()):
        raise ValueError(f"Capacity helper for {context_id} gate receipt disagrees with its counts")
    expected_failed = [name for name in PRIMARY_GATE_NAMES if not expected_gates[name]]
    if (record.get("failed_conjuncts") != expected_failed or
            record.get("capacity_available") is not (not expected_failed) or
            record.get("state") != ("available" if not expected_failed else "unavailable")):
        raise ValueError(f"Capacity helper for {context_id} availability disagrees with its frozen gates")
    bank = record.get("source_fit_bank")
    witness = record.get("witness_reservation")
    if not isinstance(bank, Mapping) or not isinstance(witness, Mapping):
        raise ValueError(f"Capacity helper for {context_id} omitted compact new-bank provenance")
    if (witness.get("image_namespace") != dict(source_namespace) or
            bank.get("image_namespace") != dict(source_namespace) or
            bank.get("witness_selected_ids_sha256") != record.get("witness_ids_sha256") or
            record.get("fit_union_intersection_with_witness_count") != 0 or
            record.get("bank_intersection_with_witness_count") != 0 or
            bank.get("fit_union_intersection_with_witness_count") != 0 or
            bank.get("bank_intersection_with_witness_count") != 0 or
            bank.get("bank_anchor_count") != record.get("bank_anchor_count") or
            bank.get("bank_intersection_with_witness_count") != 0 or
            bank.get("bank_state") != record.get("bank_state") or
            (_nonnegative_int(record.get("bank_anchor_count"), "bank anchor count") > 0 and
             bank.get("bank_anchor_dependency_sha256") is None)):
        raise ValueError(f"Capacity helper for {context_id} failed its source-supported bank checks")
    resource = record.get("resource")
    if not isinstance(resource, Mapping) or resource.get("limit_bytes") != MAX_SHARED_WORKING_BYTES:
        raise ValueError(f"Capacity helper for {context_id} omitted the frozen 128 MiB resource ledger")
    phase_peaks = resource.get("phase_peak_estimates_bytes")
    if not isinstance(phase_peaks, Mapping) or not phase_peaks:
        raise ValueError(f"Capacity helper for {context_id} omitted phase peak estimates")
    for phase, peak in phase_peaks.items():
        if _nonnegative_int(peak, f"{context_id} {phase} peak") > MAX_SHARED_WORKING_BYTES:
            raise ValueError(f"Capacity helper for {context_id} exceeded the shared 128 MiB cap")
    elapsed = resource.get("elapsed_seconds")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
            not math.isfinite(float(elapsed)) or elapsed < 0 or elapsed > MAX_WALL_SECONDS):
        raise ValueError(f"Capacity helper for {context_id} omitted a valid elapsed-time record")
    _check_deadline(deadline, f"{context_id} capacity receipt validation")
    return record


def _structural_bank(record: Mapping[str, object], context_id: str) -> dict[str, object]:
    bank = record["source_fit_bank"]
    assert isinstance(bank, Mapping)
    anchors = _nonnegative_int(record.get("bank_anchor_count"), f"{context_id} bank anchors")
    intersection = _nonnegative_int(
        record.get("bank_intersection_with_witness_count"), f"{context_id} bank/W intersection",
    )
    source_supported = _nonnegative_int(
        record.get("source_supported_count"), f"{context_id} source-supported count",
    )
    bank_state = record.get("bank_state")
    supported_disjoint = (
        anchors >= 1 and anchors <= source_supported and bank_state == "eligible" and
        bank.get("bank_state") == "eligible" and intersection == 0 and
        bank.get("bank_intersection_with_witness_count") == 0
    )
    return {
        "required_gate": "source_supported_disjoint_bank_ge_1",
        "passed": supported_disjoint,
        "anchor_count": anchors,
        "source_supported_count": source_supported,
        "bank_state": bank_state,
        "bank_intersection_with_witness_count": intersection,
        "bank_anchor_ids_sha256": record.get("bank_ids_sha256"),
        "bank_anchor_dependency_sha256": bank.get("bank_anchor_dependency_sha256"),
        "bank_dependency_union_sha256": bank.get("bank_dependency_union_sha256"),
        "bank_manifest_sha256": record.get("new_bank_manifest_sha256"),
    }


def _verify_source_packet_pin(
    capture_root: Path,
    source_view: Mapping[str, object],
    deadline: float,
) -> dict[str, object]:
    context_id = source_view.get("context_id")
    relative_path = source_view.get("packet_path")
    expected_relative = f"packets/contexts/{context_id}.npz"
    if relative_path != expected_relative or not isinstance(context_id, str):
        raise ValueError("Bound source packet path does not match its frozen template context")
    current = capture_root
    for part in ("packets", "contexts", f"{context_id}.npz"):
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ValueError("Bound source packet path contains a symbolic link")
        except OSError as exc:
            raise ValueError("Bound source packet path disappeared before final recheck") from exc
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(capture_root):
        raise ValueError("Bound source packet escaped the frozen capture root")
    pin = _verify_file_pin(
        resolved,
        _nonnegative_int(source_view.get("packet_bytes"), "source packet byte pin"),
        _lower_sha(source_view.get("packet_sha256"), "source packet SHA pin"),
        deadline,
        f"source packet {context_id}",
    )
    return {"context_id": context_id, "relative_path": relative_path, **pin}


def _row_decisions(
    row_bindings: list[dict[str, object]],
    source_view_receipts: list[dict[str, object]],
    deadline: float,
) -> list[dict[str, object]]:
    views = {
        (view["frame_id"], view["template_offset_deg"]): view
        for view in source_view_receipts
    }
    output: list[dict[str, object]] = []
    for bound in row_bindings:
        _check_deadline(deadline, "twelve ordered row decisions")
        plan = bound["plan"]
        binding = bound["binding"]
        frame_id = plan["frame_id"]
        offset = plan["template_offset_deg"]
        primary = views[(frame_id, offset)]
        competitor_offset = 180 - offset
        competitor = views[(frame_id, competitor_offset)]
        primary_capacity = primary["capacity"]
        competitor_capacity = competitor["capacity"]
        if primary["context_id"] == competitor["context_id"] or (
                primary["source_namespace"] == competitor["source_namespace"]):
            raise ValueError(f"{plan['condition_id']} aliases its primary and competitor source namespaces")
        primary_gates = dict(primary_capacity["gates"])
        competitor_gates = dict(competitor_capacity["gates"])
        competitor_structural = _structural_bank(competitor_capacity, competitor["context_id"])
        failed = [name for name in PRIMARY_GATE_NAMES if not primary_gates[name]]
        if not competitor_structural["passed"]:
            failed.append("competitor_source_supported_disjoint_bank_ge_1")
        available = not failed
        counterpart_digests = dict(binding["counterpart_array_metadata_sha256"])
        counterpart_masks = dict(bound["counterpart_mask_metadata"])
        template = primary["source_array_metadata"]
        row = {
            "condition_id": plan["condition_id"],
            "kind": plan["kind"],
            "frame_id": frame_id,
            "template_offset_deg": offset,
            "self_condition": plan.get("self_condition"),
            "state": "available" if available else "unavailable",
            "capacity_possible": available,
            "plan_binding_verified": True,
            "counterpart_binding_verified": True,
            "counterpart_context_refs": dict(binding["context_refs"]),
            "counterpart_array_metadata_sha256": counterpart_digests,
            "counterpart_observed_crop_mask_metadata": counterpart_masks,
            "template_camera_mask_metadata_sha256": _canonical_sha256({
                key: primary["counterpart_array_metadata"][key] for key in COUNTERPART_ARRAYS
            }),
            "template_observed_crop_mask_metadata": template["observed_crop_mask"],
            "primary_source_view": {
                "context_id": primary["context_id"],
                "source_rgb_sha256": primary["source_namespace"]["source_rgb_sha256"],
                "packet_sha256": primary["packet_sha256"],
                "source_array_metadata_sha256": primary["source_array_metadata_sha256"],
            },
            "competitor_source_view": {
                "context_id": competitor["context_id"],
                "source_rgb_sha256": competitor["source_namespace"]["source_rgb_sha256"],
                "packet_sha256": competitor["packet_sha256"],
                "source_array_metadata_sha256": competitor["source_array_metadata_sha256"],
            },
            "primary_gates": primary_gates,
            "primary_metrics": {
                "witness_count": primary_capacity["witness_count"],
                "fit_count": primary_capacity["fit_count"],
                "bank_anchor_count": primary_capacity["bank_anchor_count"],
                "qh_possible_center_count": primary_capacity["qh_possible_center_count"],
                "qh_possible_cell_count": primary_capacity["qh_possible_cell_count"],
                "qh_maximum_bbox_hull_fraction": primary_capacity["qh_maximum_bbox_hull_fraction"],
                "source_supported_count": primary_capacity["source_supported_count"],
            },
            "primary_failed_conjuncts": [name for name in PRIMARY_GATE_NAMES if not primary_gates[name]],
            "competitor_structural_bank": competitor_structural,
            "competitor_metrics": {
                "witness_count": competitor_capacity["witness_count"],
                "fit_count": competitor_capacity["fit_count"],
                "bank_anchor_count": competitor_capacity["bank_anchor_count"],
                "qh_possible_center_count": competitor_capacity["qh_possible_center_count"],
                "qh_possible_cell_count": competitor_capacity["qh_possible_cell_count"],
                "qh_maximum_bbox_hull_fraction": competitor_capacity["qh_maximum_bbox_hull_fraction"],
            },
            "competitor_other_gates": {
                name: passed for name, passed in competitor_gates.items() if name != "bank_ge_1"
            },
            "competitor_other_failed_conjuncts": [
                name for name, passed in competitor_gates.items()
                if name != "bank_ge_1" and not passed
            ],
            "failed_conjuncts": failed,
            "old_r6_bank_comparison": {
                "state": "unavailable",
                "reason": "closed R6 report was pinned by hash only; no safe sealed comparison metadata was available",
            },
            "capacity_only": True,
            "matching_started": False,
            "fitting_started": False,
            "accuracy": False,
        }
        output.append(row)
    if tuple(row["condition_id"] for row in output) != CANONICAL_PLAN_IDS:
        raise ValueError("R8 source adapter changed the twelve canonical row order")
    return output


def _make_failure_receipt(
    stage: str,
    reason: str,
    source_views: list[dict[str, object]],
    *,
    context_id: str | None = None,
    failed_view: Mapping[str, object] | None = None,
) -> dict[str, object]:
    failure: dict[str, object] = {"stage": stage, "reason": reason[:500]}
    if context_id is not None:
        failure["context_id"] = context_id
    receipt: dict[str, object] = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "state": "failed",
        "complete": False,
        "capacity_screen_complete": False,
        "capacity_available": False,
        "full_28_case_screen_complete": False,
        "source_views": source_views,
        "rows": [
            {"condition_id": condition_id, "state": "not_evaluated_due_terminal_failure"}
            for condition_id in CANONICAL_PLAN_IDS
        ],
        "failure": failure,
        "counts": {
            "source_views_completed": len(source_views),
            "source_views_expected": 6,
            "rows_terminal_failure": len(CANONICAL_PLAN_IDS),
            "rows_expected": len(CANONICAL_PLAN_IDS),
        },
        "claims": {
            "capacity_only": True,
            "full_28_case_screen_complete": False,
            "accuracy": False,
            "pose": False,
            "model_calls": 0,
            "render_calls": 0,
            "ncc_calls": 0,
            "optimizer_calls": 0,
            "evaluator_truth_read": False,
        },
        "matching_started": False,
        "fitting_started": False,
        "accuracy": False,
        "query_rgb_read": False,
        "evaluator_truth_read": False,
    }
    if failed_view is not None:
        receipt["failed_view"] = dict(failed_view)
    return receipt


def bind_and_screen_actual_rows(
    capture_root: object,
    zero_root: object,
    input_root: object,
    *,
    resident_bytes: int = 0,
    deadline_monotonic: float | None = None,
) -> dict[str, object]:
    """Load six distinct source views once and emit twelve plan-bound decisions.

    Any integrity, metadata, resource, deadline, or helper failure raises
    :class:`SourceAdapterError`; its ``receipt`` preserves completed compact
    source-view receipts and marks all twelve canonical rows unmeasured.
    """
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    source_view_receipts: list[dict[str, object]] = []
    stage = "entry"
    active_context: str | None = None
    failed_view: dict[str, object] | None = None
    try:
        if isinstance(resident_bytes, bool) or not isinstance(resident_bytes, int) or resident_bytes < 0:
            raise ValueError("resident_bytes must be a nonnegative integer")
        if deadline_monotonic is not None:
            if (isinstance(deadline_monotonic, bool) or
                    not isinstance(deadline_monotonic, (int, float)) or
                    not math.isfinite(float(deadline_monotonic))):
                raise ValueError("deadline_monotonic must be a finite monotonic timestamp")
            deadline = min(deadline, float(deadline_monotonic))
        _check_deadline(deadline, "entry")
        if resident_bytes + HASH_CHUNK_BYTES + REPORT_RESERVE_BYTES > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("Caller residency plus bounded pin-reader/report workspace exceeds 128 MiB")

        stage = "root_resolution"
        model_root, capture_path, zero_path, input_path = _resolve_input_roots(
            capture_root, zero_root, input_root,
        )
        repo_root = _repo_root()

        stage = "core_pin_preflight"
        core_pins = _verify_core_pins(repo_root, deadline)
        stage = "input_pin_preflight"
        input_pins = _verify_input_pins(model_root, deadline)
        pin_record_bytes = (
            _deep_size(input_pins, deadline, "input pin record") +
            _deep_size(core_pins, deadline, "core pin record")
        )

        capture_rel = "diagnostics/bottle-identity-v2/capture.json"
        zero_rel = "diagnostics/bottle-zero-view-calibration-v3/zero_view.json"
        capture_bytes = int(PINNED_FILE_SPECS[capture_rel]["bytes"])
        zero_bytes = int(PINNED_FILE_SPECS[zero_rel]["bytes"])
        metadata_parse_peak = resident_bytes + 32 * (capture_bytes + zero_bytes) + 16 * 1024**2
        if metadata_parse_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 pinned metadata parse/bind prospective peak exceeds 128 MiB")

        # Load only R1/R3 manifests, then let the frozen binder inspect its
        # strict safe projection. Closed R5/R6 reports remain hash-only.
        stage = "metadata_parse"
        capture_metadata = _read_json_metadata(capture_path / "capture.json", deadline, "R1 capture")
        zero_metadata = _read_json_metadata(zero_path / "zero_view.json", deadline, "R3 zero-view")

        stage = "core_import"
        from . import quality_bottle_pose_ablation as r5
        from . import quality_bottle_source_bindings as binder
        from . import quality_bottle_source_capacity as capacity
        from . import quality_bottle_source_packet as packet
        from . import glb_model

        plans = r5._expected_condition_plan()
        if (not isinstance(plans, list) or
                tuple(plan.get("condition_id") for plan in plans if isinstance(plan, Mapping)) != CANONICAL_PLAN_IDS):
            raise ValueError("R5 canonical plan builder differs from the twelve frozen source rows")
        stage = "metadata_binding"
        if metadata_parse_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 parsed metadata plus binding receipt prospective peak exceeds 128 MiB")
        metadata_bindings = binder.bind_source_plans(capture_metadata, zero_metadata, plans)
        template_bindings, row_bindings, _ = _validate_bindings(metadata_bindings, plans, binder)
        binding_bytes = _deep_size(metadata_bindings, deadline, "R8 binding receipt")
        binding_bytes += _deep_size(plans, deadline, "R5 plans")
        if resident_bytes + binding_bytes + REPORT_RESERVE_BYTES > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 retained metadata bindings exceed the shared 128 MiB cap")
        del capture_metadata, zero_metadata, metadata_bindings, plans

        # Pin and inspect the CPU mesh only after a prospective bound includes
        # the retained binding metadata and GLB reader's temporary allocations.
        stage = "mesh_preflight"
        mesh_rel = "inputs/ranch/object.glb"
        mesh_pin = input_pins[mesh_rel]
        mesh_path = input_path / "object.glb"
        mesh_bytes = int(mesh_pin["bytes"])
        mesh_json_preflight_peak = (
            resident_bytes + binding_bytes + pin_record_bytes + mesh_bytes * 8 + REPORT_RESERVE_BYTES
        )
        if mesh_json_preflight_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 GLB JSON preflight prospective peak exceeds 128 MiB")
        mesh_estimate = _parse_glb_and_estimate_reader_peak(mesh_path, mesh_bytes, deadline)
        mesh_reader_peak = (
            resident_bytes + binding_bytes +
            pin_record_bytes + int(mesh_estimate["reader_peak_estimate_bytes"]) + REPORT_RESERVE_BYTES
        )
        if mesh_reader_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 pinned GLB CPU reader prospective peak exceeds 128 MiB")

        stage = "mesh_read"
        mesh = glb_model.read_glb(mesh_path, "ranch")
        import numpy as np
        positions = mesh.positions
        triangles = mesh.triangles
        normals = mesh.normals
        uv = mesh.uv
        if (not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1] != 3 or
                not isinstance(triangles, np.ndarray) or triangles.ndim != 2 or triangles.shape[1] != 3 or
                len(positions) != mesh_estimate["vertex_count"] or
                len(triangles) != mesh_estimate["triangle_count"] or
                not np.isfinite(positions).all()):
            raise ValueError("Pinned CPU GLB reader returned geometry unlike its preflighted native mesh")
        mesh_position_sha = _typed_array_sha256(positions, "positions")
        mesh_triangle_sha = _typed_array_sha256(triangles, "triangles")
        mesh_full_attributes_bytes = int(positions.nbytes + triangles.nbytes + normals.nbytes + uv.nbytes)
        released_renderer_attributes_bytes = int(normals.nbytes + uv.nbytes)
        mesh_geometry_bytes = int(positions.nbytes + triangles.nbytes)
        if resident_bytes + binding_bytes + pin_record_bytes + mesh_full_attributes_bytes + REPORT_RESERVE_BYTES > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("Pinned CPU mesh retained attributes exceed the shared 128 MiB cap")
        # Capacity needs only original metre positions and original triangle
        # order/winding. Release renderer-only normals and UVs immediately.
        mesh.normals = np.empty((0, 3), dtype=normals.dtype)
        mesh.uv = np.empty((0, 2), dtype=uv.dtype)
        del normals, uv
        if resident_bytes + binding_bytes + pin_record_bytes + mesh_geometry_bytes + REPORT_RESERVE_BYTES > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("Pinned native positions/triangles exceed the shared 128 MiB cap")
        mesh_receipt = {
            "relative_path": mesh_rel,
            "bytes": mesh_bytes,
            "sha256": mesh_pin["sha256"],
            "positions": {"shape": list(positions.shape), "dtype": positions.dtype.str,
                          "typed_sha256": mesh_position_sha},
            "triangles": {"shape": list(triangles.shape), "dtype": triangles.dtype.str,
                          "typed_sha256": mesh_triangle_sha},
            "vertex_count": int(len(positions)),
            "triangle_count": int(len(triangles)),
            "triangle_order_and_winding": "original_CPU_GLTF_reader_output",
            "released_renderer_attributes_bytes": released_renderer_attributes_bytes,
            "reader_peak_estimate_bytes": int(mesh_estimate["reader_peak_estimate_bytes"]),
            "retained_geometry_bytes": mesh_geometry_bytes,
        }

        # One adapter-created JSON allowance covers compact prior receipts,
        # row assembly, and return serialization. Packet/capacity helpers add
        # their own arrays, geometry, ZIP and phase-specific estimates.
        phase_peaks: dict[str, int] = {
            "metadata_parse_and_bind": metadata_parse_peak,
            "mesh_json_preflight": mesh_json_preflight_peak,
            "mesh_reader": mesh_reader_peak,
            "mesh_retained": resident_bytes + binding_bytes + pin_record_bytes + mesh_geometry_bytes + REPORT_RESERVE_BYTES,
        }
        required_capacity_fields = CAPACITY_SOURCE_KEYS

        for index, template in enumerate(template_bindings):
            frame_id, offset, context_id = EXPECTED_VIEWS[index]
            active_context = context_id
            failed_view = {
                "context_id": context_id,
                "frame_id": frame_id,
                "template_offset_deg": offset,
                "state": "failed_before_view_complete",
            }
            stage = "source_packet_load"
            binding_rows_for_view = [
                row for row in row_bindings
                if row["plan"]["frame_id"] == frame_id and
                row["plan"]["template_offset_deg"] == offset and
                row["plan"]["kind"] in ("synthetic_positive", "synthetic_negative")
            ]
            if len(binding_rows_for_view) != 1:
                raise ValueError(f"{context_id} lacks its unique synthetic counterpart plan binding")
            synthetic_binding = binding_rows_for_view[0]
            counterpart_masks = synthetic_binding["counterpart_mask_metadata"]
            if set(counterpart_masks) != {"observed_frame", "synthetic_query"}:
                raise ValueError(f"{context_id} did not independently bind both observed counterparts")
            source_entry = template["source_entry"]
            source_namespace = template["source_namespace"]
            source_metadata = template["source_array_metadata"]
            current_report_bytes = _deep_size(source_view_receipts, deadline, "completed source receipts")
            external_resident = (
                resident_bytes + binding_bytes + mesh_geometry_bytes +
                pin_record_bytes + current_report_bytes + REPORT_RESERVE_BYTES
            )
            if external_resident > MAX_SHARED_WORKING_BYTES:
                raise MemoryError("R8 source packet reader prospective external residency exceeds 128 MiB")
            loaded = packet.load_source_template(
                capture_path,
                source_entry,
                expected_context_id=context_id,
                expected_frame_id=frame_id,
                expected_offset_deg=offset,
                real_frame_mask_metadata=counterpart_masks["observed_frame"],
                synthetic_mask_metadata=counterpart_masks["synthetic_query"],
                require_synthetic_mask=True,
                resident_bytes=external_resident,
                deadline_monotonic=deadline,
            )
            if not isinstance(loaded, Mapping) or set(loaded) != {"arrays", "receipt", "image_namespace"}:
                raise ValueError(f"{context_id} source packet loader changed its projection API")
            source_arrays = loaded.get("arrays")
            packet_receipt = loaded.get("receipt")
            loaded_namespace = loaded.get("image_namespace")
            if (not isinstance(source_arrays, Mapping) or set(source_arrays) != required_capacity_fields or
                    not isinstance(packet_receipt, Mapping) or not isinstance(loaded_namespace, Mapping)):
                raise ValueError(f"{context_id} source packet loader returned unexpected fields")
            if (loaded_namespace != source_namespace or
                    packet_receipt.get("state") != "loaded_source_template" or
                    packet_receipt.get("context_id") != context_id or
                    packet_receipt.get("frame_id") != frame_id or
                    packet_receipt.get("template_offset_deg") != offset or
                    packet_receipt.get("packet_sha256_rechecked") is not True or
                    packet_receipt.get("r5_source_only_loader_calls") != 1 or
                    packet_receipt.get("counterpart_context_bindings_verified") is not False or
                    packet_receipt.get("query_or_evaluator_data_read") is not False or
                    packet_receipt.get("fitting_or_matching_started") is not False or
                    packet_receipt.get("capacity_screen_complete") is not False):
                raise ValueError(f"{context_id} source packet receipt failed its namespace, pin, or boundary checks")
            packet_field = packet_receipt.get("packet")
            if (not isinstance(packet_field, Mapping) or
                    packet_field.get("bytes") != template["packet_bytes"] or
                    packet_field.get("sha256") != template["packet_sha256"] or
                    packet_field.get("relative_path") != f"packets/contexts/{context_id}.npz"):
                raise ValueError(f"{context_id} packet receipt differs from its bound source entry")
            projected_metadata = packet_receipt.get("source_array_metadata")
            expected_projected_metadata = {
                key: source_metadata[key] for key in SOURCE_ARRAY_NAMES
                if key not in {"template_gray_rgb", "seed_pose_m"}
            }
            if projected_metadata != expected_projected_metadata:
                raise ValueError(f"{context_id} packet loader changed the allowed eleven-field metadata projection")
            counterpart_checks = packet_receipt.get("counterpart_mask_checks")
            if counterpart_checks != [
                {"label": "real_frame", "matches_source_observed_crop_mask": True},
                {"label": "synthetic_query", "matches_source_observed_crop_mask": True},
            ]:
                raise ValueError(f"{context_id} source packet did not pass both observed-mask metadata bindings")
            if packet_receipt.get("observed_crop_mask_metadata") != source_metadata["observed_crop_mask"]:
                raise ValueError(f"{context_id} packet observed-mask metadata differs from frozen R1 metadata")

            stage = "source_capacity"
            if set(source_arrays) != required_capacity_fields:
                raise ValueError(f"{context_id} source array projection changed before capacity")
            observed_mask = source_arrays["observed_crop_mask"]
            # The observed mask is explicit even though it aliases a source
            # projection field; no fallback mask or counterpart packet is read.
            capacity_resident = (
                resident_bytes + binding_bytes + current_report_bytes +
                pin_record_bytes +
                _deep_size(packet_receipt, deadline, f"{context_id} packet receipt") +
                REPORT_RESERVE_BYTES
            )
            if capacity_resident > MAX_SHARED_WORKING_BYTES:
                raise MemoryError(f"{context_id} capacity call prospective residency exceeds 128 MiB")
            capacity_record = capacity.evaluate_source_capacity(
                source_arrays,
                source_context_id=context_id,
                packet_sha256=str(template["packet_sha256"]),
                mesh_positions_m=positions,
                mesh_triangles=triangles,
                mesh_sha256=str(mesh_pin["sha256"]),
                query_observed_mask=observed_mask,
                resident_bytes=capacity_resident,
                deadline_monotonic=deadline,
            )
            capacity_record = _validate_capacity_record(
                capacity_record,
                context_id=context_id,
                source_namespace=source_namespace,
                packet_sha256=str(template["packet_sha256"]),
                mesh_sha256=str(mesh_pin["sha256"]),
                deadline=deadline,
            )
            capacity_resource = capacity_record["resource"]
            assert isinstance(capacity_resource, Mapping)
            phase_capacity = capacity_resource["phase_peak_estimates_bytes"]
            assert isinstance(phase_capacity, Mapping)
            for phase, peak in phase_capacity.items():
                phase_peaks[f"{context_id}:{phase}"] = int(peak)

            compact_packet_receipt = _json_copy(packet_receipt, f"{context_id} packet-loader receipt")
            compact_capacity = _json_copy(capacity_record, f"{context_id} capacity receipt")
            source_metadata_digest = _canonical_sha256(source_metadata)
            source_view_receipt = {
                "context_id": context_id,
                "frame_id": frame_id,
                "template_offset_deg": offset,
                "geometry_group": template["geometry_group"],
                "source_namespace": dict(source_namespace),
                "packet_bytes": template["packet_bytes"],
                "packet_path": source_entry["path"],
                "packet_sha256": template["packet_sha256"],
                "source_array_metadata_sha256": source_metadata_digest,
                "source_array_metadata": source_metadata,
                "source_array_sha256": {
                    key: source_metadata[key]["sha256"] for key in SOURCE_ARRAY_NAMES
                },
                "counterpart_array_metadata": template["counterpart_array_metadata"],
                "camera_mask_metadata_sha256": template["counterpart_array_metadata_sha256"],
                "observed_crop_mask_metadata": source_metadata["observed_crop_mask"],
                "counterpart_context_bindings_verified": True,
                "plan_bindings_verified": [row["plan"]["condition_id"] for row in row_bindings
                                           if row["binding"]["template_context_id"] == context_id],
                "packet_loader_receipt": compact_packet_receipt,
                "capacity": compact_capacity,
            }
            safe_view_receipt = _json_copy(source_view_receipt, f"{context_id} compact source view")
            if not isinstance(safe_view_receipt, dict):
                raise ValueError(f"{context_id} compact source view is not a JSON object")
            source_view_receipts.append(safe_view_receipt)
            failed_view = None
            active_context = None
            del observed_mask, source_arrays, loaded, packet_receipt, capacity_record

        stage = "row_binding"
        if len(source_view_receipts) != 6:
            raise ValueError("R8 source adapter did not evaluate all six source views")
        rows = _row_decisions(row_bindings, source_view_receipts, deadline)
        if len(rows) != 12 or tuple(row["condition_id"] for row in rows) != CANONICAL_PLAN_IDS:
            raise ValueError("R8 source adapter did not emit all twelve canonical ordered decisions")

        # Release parsed bindings and native attributes that are no longer
        # needed before final pin checks and compact receipt serialization.
        del template_bindings, row_bindings, binding_bytes
        del mesh, positions, triangles
        del template, synthetic_binding, counterpart_masks, source_entry, source_namespace, source_metadata
        del binding_rows_for_view, current_report_bytes, external_resident, source_view_receipt, safe_view_receipt
        del compact_packet_receipt, compact_capacity, source_metadata_digest, projected_metadata
        del expected_projected_metadata, packet_field, counterpart_checks, capacity_resident
        del capacity_resource, phase_capacity, loaded_namespace

        stage = "final_pin_recheck"
        final_pin_peak = resident_bytes + _deep_size(source_view_receipts, deadline, "final pin receipt set")
        final_pin_peak += _deep_size(rows, deadline, "final pin decision rows")
        final_pin_peak += pin_record_bytes
        final_pin_peak += HASH_CHUNK_BYTES + REPORT_RESERVE_BYTES
        if final_pin_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 final source/input/core pin recheck exceeds the shared 128 MiB cap")
        phase_peaks["final_pin_recheck"] = final_pin_peak
        source_packet_pins = [
            _verify_source_packet_pin(capture_path, view, deadline)
            for view in source_view_receipts
        ]
        final_input_pins = _verify_input_pins(model_root, deadline)
        final_core_pins = _verify_core_pins(repo_root, deadline)
        if input_pins != final_input_pins or core_pins != final_core_pins:
            raise ValueError("An immutable R8 input or core pin changed during the adapter run")

        stage = "receipt_seal"
        elapsed = time.monotonic() - started
        if elapsed > MAX_WALL_SECONDS:
            raise TimeoutError("R8 source adapter exceeded its shared 900-second wall cap")
        successful = sum(row["capacity_possible"] is True for row in rows)
        _check_deadline(deadline, "prebudget receipt allocation")
        receipt_graph_estimate = (
            _deep_size(source_view_receipts, deadline, "source view receipts") +
            _deep_size(rows, deadline, "twelve decision rows") +
            _deep_size(input_pins, deadline, "input pin receipt") +
            _deep_size(core_pins, deadline, "core pin receipt") +
            _deep_size(source_packet_pins, deadline, "source packet pin receipt") +
            _deep_size(mesh_receipt, deadline, "native mesh receipt") + 8192
        )
        receipt_preallocation_peak = resident_bytes + 2 * receipt_graph_estimate + REPORT_RESERVE_BYTES
        if receipt_preallocation_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 compact result and serialization prospective peak exceeds 128 MiB")
        if receipt_graph_estimate > MAX_RECEIPT_BYTES:
            raise MemoryError("Compact R8 actual-source adapter receipt prospectively exceeds 8 MiB")
        receipt: dict[str, object] = {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "state": "complete",
            "complete": True,
            "capacity_screen_complete": True,
            "capacity_only": True,
            "capacity_available": successful == len(rows),
            "full_28_case_screen_complete": False,
            "input_pins": input_pins,
            "core_pins": core_pins,
            "source_packet_pins": source_packet_pins,
            "mesh": mesh_receipt,
            "old_r6_bank_comparison": {
                "state": "unavailable",
                "reason": "closed R6 report was pinned by hash only; no safe sealed comparison metadata was available",
            },
            "source_views": source_view_receipts,
            "rows": rows,
            "counts": {
                "source_views_expected": 6,
                "source_views_evaluated_once": len(source_view_receipts),
                "rows_expected": 12,
                "rows_decided": len(rows),
                "rows_capacity_possible": successful,
                "rows_capacity_unavailable": len(rows) - successful,
            },
            "resource": {
                "limit_bytes": MAX_SHARED_WORKING_BYTES,
                "wall_limit_seconds": MAX_WALL_SECONDS,
                "elapsed_seconds": elapsed,
                "prospective_peak_estimate_bytes": max(phase_peaks.values(), default=0),
                "phase_peak_estimates_bytes": phase_peaks,
                "adapter_external_resident_bytes": resident_bytes,
                "receipt_size_limit_bytes": MAX_RECEIPT_BYTES,
            },
            "claims": {
                "capacity_only": True,
                "full_28_case_screen_complete": False,
                "accuracy": False,
                "pose": False,
                "model_calls": 0,
                "render_calls": 0,
                "ncc_calls": 0,
                "optimizer_calls": 0,
                "evaluator_truth_read": False,
            },
            "matching_started": False,
            "fitting_started": False,
            "accuracy": False,
            "query_rgb_read": False,
            "evaluator_truth_read": False,
        }
        # Bound the Python receipt graph plus two serialized copies before
        # creating the final JSON byte string. Dense source arrays and parsed
        # manifests have already been released at this point.
        _check_deadline(deadline, "prebudget receipt serialization")
        receipt_graph_bytes = _deep_size(receipt, deadline, "final compact receipt")
        receipt_serialization_peak = resident_bytes + 2 * receipt_graph_bytes + REPORT_RESERVE_BYTES
        if receipt_serialization_peak > MAX_SHARED_WORKING_BYTES:
            raise MemoryError("R8 adapter receipt serialization exceeds the shared 128 MiB cap")
        if receipt_graph_bytes > MAX_RECEIPT_BYTES:
            raise MemoryError("Compact R8 actual-source adapter receipt exceeds 8 MiB")
        phase_peaks["return_receipt_serialization"] = receipt_serialization_peak
        receipt["resource"]["phase_peak_estimates_bytes"] = phase_peaks
        receipt["resource"]["prospective_peak_estimate_bytes"] = max(phase_peaks.values())
        serialized = _canonical_json(receipt)
        _check_deadline(deadline, "receipt serialization")
        if len(serialized) > MAX_RECEIPT_BYTES:
            raise MemoryError("Compact R8 actual-source adapter receipt exceeds 8 MiB")
        _check_deadline(deadline, "adapter return")
        return receipt
    except Exception as exc:
        reason = str(exc) or exc.__class__.__name__
        receipt = _make_failure_receipt(
            stage, reason, source_view_receipts,
            context_id=active_context,
            failed_view=failed_view,
        )
        raise SourceAdapterError(reason, receipt) from exc


__all__ = [
    "ADAPTER_SCHEMA_VERSION", "CANONICAL_PLAN_IDS", "MAX_RECEIPT_BYTES",
    "MAX_SHARED_WORKING_BYTES", "MAX_WALL_SECONDS", "PINNED_CORE_SPECS",
    "PINNED_FILE_SPECS", "SourceAdapterError", "bind_and_screen_actual_rows",
]
