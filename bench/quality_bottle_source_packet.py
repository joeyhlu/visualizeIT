"""Strict, source-only preflight for frozen R8 template packets.

The preflight validates packet identity, metadata, file pins, and ZIP member
names without reading or decoding any NPZ member payload. Source array loading
is intentionally a separate, later operation.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import struct
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping


SPEC_V2_SHA256 = "32758DDA6AD550A5EDD94F5354B7E8401FF206A63B884E8C88AB37F71EE6D9CB"
MAX_WORKING_BYTES = 128 * 1024**2
MAX_WALL_SECONDS = 900.0
HASH_CHUNK_BYTES = 1024**2
NPY_HEADER_ALLOWANCE_BYTES = 64 * 1024
DECODE_FIXED_ALLOWANCE_BYTES = 1024**2
ZIP_DIRECTORY_METADATA_ALLOWANCE_BYTES = 64 * 1024
ZIP_DIRECTORY_MAX_BYTES = 4 * 1024
ZIP_TRAILER_READ_BYTES = 22 + 65535
ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
MAX_PACKET_OVERHEAD_BYTES = 2 * 1024**2
TEMPLATE_SIZE = 280
FROZEN_TEMPLATE_FRAME_IDS = (10, 50, 100)

SOURCE_ARRAY_KEYS = (
    "template_rgb",
    "template_gray_rgb",
    "template_depth_mm",
    "template_mask",
    "source_indices",
    "source_pixels_xy",
    "source_points_object_m",
    "observed_crop_mask",
    "crop_k",
    "crop_from_native",
    "native_k",
    "seed_pose_m",
    "template_pose_m",
)

_FIXED_ARRAY_SCHEMA = {
    "template_rgb": ((TEMPLATE_SIZE, TEMPLATE_SIZE, 3), "float32", 4),
    "template_gray_rgb": ((TEMPLATE_SIZE, TEMPLATE_SIZE, 3), "float32", 4),
    "template_depth_mm": ((TEMPLATE_SIZE, TEMPLATE_SIZE), "float32", 4),
    "template_mask": ((TEMPLATE_SIZE, TEMPLATE_SIZE), "uint8", 1),
    "observed_crop_mask": ((TEMPLATE_SIZE, TEMPLATE_SIZE), "uint8", 1),
    "crop_k": ((3, 3), "float64", 8),
    "crop_from_native": ((4, 4), "float64", 8),
    "native_k": ((3, 3), "float64", 8),
    "seed_pose_m": ((4, 4), "float64", 8),
    "template_pose_m": ((4, 4), "float64", 8),
}
_DYNAMIC_ARRAY_SCHEMA = {
    "source_indices": (1, "int64", 8),
    "source_pixels_xy": (2, "float64", 8),
    "source_points_object_m": (2, "float64", 8),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PacketPreflightError(ValueError):
    """A frozen template packet does not match the source-only contract."""


class PacketResourceLimitError(PacketPreflightError):
    """Packet preflight or prospective decoding exceeds a frozen resource cap."""


def _check_deadline(deadline: float, label: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PacketResourceLimitError(f"R8 template packet deadline exceeded during {label}")
    return remaining


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PacketPreflightError(f"{label} must be a nonnegative integer")
    return value


def _metadata_record(value: object, label: str) -> tuple[dict, int]:
    if not isinstance(value, Mapping) or set(value) != {"shape", "dtype", "sha256"}:
        raise PacketPreflightError(f"{label} metadata must contain exactly shape, dtype, and sha256")
    shape_value = value["shape"]
    if not isinstance(shape_value, list) or len(shape_value) > 4:
        raise PacketPreflightError(f"{label} shape must be a short list of integer dimensions")
    shape = []
    for dimension in shape_value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise PacketPreflightError(f"{label} shape contains a non-integer or negative dimension")
        shape.append(dimension)
    dtype = value["dtype"]
    if not isinstance(dtype, str) or dtype not in {"float32", "float64", "uint8", "int64"}:
        raise PacketPreflightError(f"{label} has an unsupported or object dtype")
    digest = value["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise PacketPreflightError(f"{label} SHA256 metadata is malformed")
    count = math.prod(shape) if shape else 1
    itemsize = {"float32": 4, "float64": 8, "uint8": 1, "int64": 8}[dtype]
    nbytes = count * itemsize
    return {"shape": shape, "dtype": dtype, "sha256": digest}, nbytes


def _validate_array_metadata(value: object) -> tuple[dict[str, dict], int, int]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_ARRAY_KEYS):
        if isinstance(value, Mapping):
            keys = set(value)
            missing = sorted(set(SOURCE_ARRAY_KEYS) - keys)
            extra = sorted((repr(key) for key in keys - set(SOURCE_ARRAY_KEYS)))
        else:
            missing = list(SOURCE_ARRAY_KEYS)
            extra = []
        raise PacketPreflightError(
            f"Template packet must declare exactly the 13 frozen source arrays; missing={missing}, extra={extra}")
    result: dict[str, dict] = {}
    decoded_bytes = 0
    max_array_bytes = 0
    dynamic_count = None
    for key in SOURCE_ARRAY_KEYS:
        info, nbytes = _metadata_record(value[key], key)
        if key in _FIXED_ARRAY_SCHEMA:
            expected_shape, expected_dtype, _ = _FIXED_ARRAY_SCHEMA[key]
            if tuple(info["shape"]) != expected_shape or info["dtype"] != expected_dtype:
                raise PacketPreflightError(
                    f"{key} metadata differs from the frozen 280x280 packet schema")
        else:
            dimensions, expected_dtype, _ = _DYNAMIC_ARRAY_SCHEMA[key]
            expected_tail = {
                "source_indices": (),
                "source_pixels_xy": (2,),
                "source_points_object_m": (3,),
            }[key]
            if (len(info["shape"]) != dimensions or info["dtype"] != expected_dtype or
                    tuple(info["shape"][1:]) != expected_tail):
                raise PacketPreflightError(f"{key} metadata has an invalid shape or dtype")
            count = info["shape"][0]
            if count > TEMPLATE_SIZE * TEMPLATE_SIZE:
                raise PacketPreflightError(f"{key} has more rows than the 280x280 source image")
            if dynamic_count is None:
                dynamic_count = count
            elif dynamic_count != count:
                raise PacketPreflightError("Source ID, pixel, and object-point counts differ")
        result[key] = info
        decoded_bytes += nbytes
        max_array_bytes = max(max_array_bytes, nbytes)
    return result, decoded_bytes, max_array_bytes


def _safe_packet_path(capture_root: object, relative_path: object) -> tuple[Path, str]:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise PacketPreflightError("Template packet path must be a nonempty POSIX relative path")
    relative = PurePosixPath(relative_path)
    if (relative.is_absolute() or not relative.parts or
            relative.as_posix() != relative_path or
            any(part in {"", ".", ".."} or ":" in part for part in relative.parts)):
        raise PacketPreflightError("Template packet path is absolute, path-like, or escapes its root")
    root = Path(capture_root).resolve(strict=True)
    if not root.is_dir():
        raise PacketPreflightError("Capture root is not a directory")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise PacketPreflightError("Template packet path does not exist") from exc
        if stat.S_ISLNK(mode):
            raise PacketPreflightError("Template packet path contains a symbolic link")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PacketPreflightError("Template packet escaped the capture root") from exc
    if not resolved.is_file():
        raise PacketPreflightError("Template packet path is not a regular file")
    return resolved, relative.as_posix()


def _hash_and_list_members(path: Path, expected_bytes: int, expected_sha256: str,
                           deadline: float) -> tuple[list[zipfile.ZipInfo], int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise PacketPreflightError("Template packet file size/type differs from its capture pin")
        while True:
            _check_deadline(deadline, "packet hashing")
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            del chunk
        after_hash = os.fstat(stream.fileno())
        if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
                (after_hash.st_size, after_hash.st_mtime_ns, after_hash.st_ino)):
            raise PacketPreflightError("Template packet changed while its pin was being checked")
        if digest.hexdigest() != expected_sha256:
            raise PacketPreflightError("Template packet SHA256 differs from its capture pin")
        # Check the bounded ZIP trailer count before ZipFile allocates one
        # Python ZipInfo instance per central-directory record.
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        tail_bytes = min(size, ZIP_TRAILER_READ_BYTES)
        stream.seek(size - tail_bytes)
        tail = stream.read(tail_bytes)
        _check_deadline(deadline, "ZIP trailer validation")
        index = tail.rfind(ZIP_EOCD_SIGNATURE)
        if index < 0 or len(tail) - index < 22:
            raise PacketPreflightError("Template NPZ has no complete ZIP end-of-central-directory record")
        try:
            (disk, central_disk, entries_on_disk, total_entries, central_size,
             central_offset, comment_length) = struct.unpack_from("<HHHHIIH", tail, index + 4)
        except struct.error as exc:
            raise PacketPreflightError("Template NPZ ZIP trailer is malformed") from exc
        record_end = index + 22 + comment_length
        eocd_absolute_offset = size - tail_bytes + index
        if (record_end != len(tail) or disk != 0 or central_disk != 0 or
                entries_on_disk != total_entries or total_entries != len(SOURCE_ARRAY_KEYS) or
                central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF or
                entries_on_disk == 0xFFFF or central_size > ZIP_DIRECTORY_MAX_BYTES or
                central_offset + central_size != eocd_absolute_offset):
            raise PacketPreflightError("Template NPZ ZIP directory count/layout is invalid")
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream, mode="r") as archive:
                members = archive.infolist()
                uncompressed_bytes = 0
                for member in members:
                    _check_deadline(deadline, "ZIP directory validation")
                    name = member.filename
                    if (member.is_dir() or "/" in name or "\\" in name or
                            name in {"", ".", ".."} or ":" in name or
                            stat.S_ISLNK(member.external_attr >> 16)):
                        raise PacketPreflightError("Template NPZ contains a path-like or non-file ZIP member")
                    uncompressed_bytes += int(member.file_size)
                after_zip = os.fstat(stream.fileno())
                if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
                        (after_zip.st_size, after_zip.st_mtime_ns, after_zip.st_ino)):
                    raise PacketPreflightError("Template packet changed during ZIP member validation")
                return members, uncompressed_bytes
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            raise PacketPreflightError("Template packet is not a valid readable NPZ archive") from exc


def preflight_template_packet(
    capture_root: object,
    entry: Mapping[str, object],
    *,
    expected_context_id: str,
    expected_frame_id: int,
    expected_offset_deg: int,
    counterpart_mask_metadata: Mapping[str, Mapping[str, object]] | None = None,
    resident_bytes: int = 0,
    deadline_monotonic: float | None = None,
) -> dict:
    """Validate one frozen template packet without decoding an NPZ member.

    ``counterpart_mask_metadata`` maps labels to already-read manifest array
    metadata for the corresponding real-frame and optional synthetic-query
    ``observed_crop_mask``. No counterpart packet path is opened here.
    """
    started = time.monotonic()
    local_deadline = started + MAX_WALL_SECONDS
    if deadline_monotonic is not None:
        if (isinstance(deadline_monotonic, bool) or
                not isinstance(deadline_monotonic, (int, float)) or
                not math.isfinite(float(deadline_monotonic))):
            raise PacketPreflightError("Caller deadline must be a finite monotonic timestamp")
        local_deadline = min(local_deadline, float(deadline_monotonic))
    _check_deadline(local_deadline, "entry validation")

    if isinstance(resident_bytes, bool) or not isinstance(resident_bytes, int) or resident_bytes < 0:
        raise PacketPreflightError("resident_bytes must be a nonnegative integer")
    if resident_bytes > MAX_WORKING_BYTES:
        raise PacketResourceLimitError("Existing resident work already exceeds the 128 MiB cap")
    if not isinstance(entry, Mapping):
        raise PacketPreflightError("Template context entry must be a mapping")
    if (not isinstance(expected_context_id, str) or not expected_context_id or
            isinstance(expected_frame_id, bool) or not isinstance(expected_frame_id, int) or
            isinstance(expected_offset_deg, bool) or not isinstance(expected_offset_deg, int) or
            expected_offset_deg not in (0, 180)):
        raise PacketPreflightError("Expected template identity is malformed")
    exact_context_id = f"template-{expected_frame_id:04d}-{expected_offset_deg:03d}"
    if expected_context_id != exact_context_id:
        raise PacketPreflightError("Expected context ID is not the frozen template frame/offset name")
    if expected_frame_id not in FROZEN_TEMPLATE_FRAME_IDS:
        raise PacketPreflightError("Expected frame is outside the six frozen R8 source templates")
    context_id = entry.get("context_id")
    frame_id = entry.get("frame_id")
    offset_deg = entry.get("template_offset_deg")
    if (context_id != expected_context_id or entry.get("role") != "template" or
            isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id != expected_frame_id or
            isinstance(offset_deg, bool) or not isinstance(offset_deg, int) or offset_deg != expected_offset_deg):
        raise PacketPreflightError("Template context role, frame, offset, or context ID differs from the frozen plan")
    if expected_frame_id <= 0:
        raise PacketPreflightError("Expected source frame ID must be positive")

    array_metadata, decoded_bytes, max_array_bytes = _validate_array_metadata(entry.get("arrays"))
    mask_info = array_metadata["observed_crop_mask"]
    checks = []
    if counterpart_mask_metadata is not None:
        if not isinstance(counterpart_mask_metadata, Mapping):
            raise PacketPreflightError("Counterpart mask metadata must be a label-to-array-info mapping")
        for label, info in counterpart_mask_metadata.items():
            if not isinstance(label, str) or not label:
                raise PacketPreflightError("Counterpart mask metadata labels must be nonempty strings")
            checked, _ = _metadata_record(info, f"counterpart {label} observed mask")
            if checked != mask_info:
                raise PacketPreflightError(f"Observed crop mask metadata differs from counterpart {label}")
            checks.append({"label": label, "matches_source_observed_crop_mask": True})

    bytes_value = entry.get("bytes")
    packet_bytes = _strict_nonnegative_int(bytes_value, "Template packet byte count")
    packet_sha = entry.get("sha256")
    if not isinstance(packet_sha, str) or not _SHA256_RE.fullmatch(packet_sha):
        raise PacketPreflightError("Template packet SHA256 pin is malformed")
    packet_path, relative_path = _safe_packet_path(capture_root, entry.get("path"))
    expected_packet_path = f"packets/contexts/{expected_context_id}.npz"
    if relative_path != expected_packet_path:
        raise PacketPreflightError("Template packet path does not match its frozen context ID")

    # Account for retained decoded arrays, the largest temporary decode/copy,
    # ZIP member expansion, and fixed reader overhead before opening the ZIP.
    prospective_peak = (resident_bytes + HASH_CHUNK_BYTES +
                        ZIP_DIRECTORY_METADATA_ALLOWANCE_BYTES + ZIP_TRAILER_READ_BYTES +
                        2 * decoded_bytes + max_array_bytes +
                        DECODE_FIXED_ALLOWANCE_BYTES)
    if packet_bytes > decoded_bytes + MAX_PACKET_OVERHEAD_BYTES:
        raise PacketResourceLimitError("Template packet bytes exceed the frozen source-array expansion bound")
    _check_deadline(local_deadline, "metadata and decode-budget validation")
    if prospective_peak > MAX_WORKING_BYTES:
        raise PacketResourceLimitError(
            f"Template packet decode requires an estimated {prospective_peak} bytes, "
            f"over the {MAX_WORKING_BYTES}-byte working cap")

    members, uncompressed_bytes = _hash_and_list_members(
        packet_path, packet_bytes, packet_sha, local_deadline)
    expected_names = {f"{key}.npy" for key in SOURCE_ARRAY_KEYS}
    member_names = [member.filename for member in members]
    if len(member_names) != len(set(member_names)):
        raise PacketPreflightError("Template NPZ contains duplicate ZIP member names")
    if len(member_names) != len(expected_names) or set(member_names) != expected_names:
        raise PacketPreflightError("Template NPZ members do not exactly match its 13 source-array keys")
    for member in members:
        info_key = member.filename[:-4]
        _, expected_array_bytes = _metadata_record(array_metadata[info_key], info_key)
        if (member.file_size < expected_array_bytes or
                member.file_size - expected_array_bytes > NPY_HEADER_ALLOWANCE_BYTES):
            raise PacketPreflightError(f"Template NPZ member size disagrees with {info_key} metadata")
    # ZIP headers count toward the actual expanded-size record but are already
    # covered by the decoded-array estimate's largest-array and fixed reserves.
    if uncompressed_bytes > decoded_bytes + len(SOURCE_ARRAY_KEYS) * NPY_HEADER_ALLOWANCE_BYTES:
        raise PacketResourceLimitError("Template NPZ expanded size exceeds its prospective decode allowance")
    receipt = {
        "schema_version": 1,
        "state": "validated_source_packet",
        "context_id": expected_context_id,
        "frame_id": expected_frame_id,
        "template_offset_deg": expected_offset_deg,
        "packet": {
            "relative_path": relative_path,
            "bytes": packet_bytes,
            "sha256": packet_sha,
        },
        "array_metadata": array_metadata,
        "observed_crop_mask_metadata": mask_info,
        "counterpart_mask_checks": checks,
        "archive": {
            "member_count": len(members),
            "uncompressed_bytes": uncompressed_bytes,
            "payloads_decoded": False,
        },
        "budget": {
            "resident_bytes": resident_bytes,
            "decoded_array_bytes": decoded_bytes,
            "peak_estimate_bytes": prospective_peak,
            "limit_bytes": MAX_WORKING_BYTES,
            "deadline_seconds_remaining": None,
        },
        "decode_authorized": False,
    }
    receipt["budget"]["deadline_seconds_remaining"] = _check_deadline(
        local_deadline, "final packet receipt sealing")
    return receipt


def _loader_deadline(deadline_monotonic: float | None) -> float:
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    if deadline_monotonic is not None:
        if (isinstance(deadline_monotonic, bool) or
                not isinstance(deadline_monotonic, (int, float)) or
                not math.isfinite(float(deadline_monotonic))):
            raise PacketPreflightError("Caller deadline must be a finite monotonic timestamp")
        deadline = min(deadline, float(deadline_monotonic))
    _check_deadline(deadline, "source-loader entry")
    return deadline


def _verify_packet_file_pin(capture_root: object, packet: Mapping[str, object],
                            deadline: float) -> None:
    path, relative_path = _safe_packet_path(capture_root, packet.get("relative_path"))
    if relative_path != packet.get("relative_path"):
        raise PacketPreflightError("Source packet relative path changed after preflight")
    expected_bytes = _strict_nonnegative_int(packet.get("bytes"), "Template packet byte count")
    expected_sha = packet.get("sha256")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        raise PacketPreflightError("Source packet SHA256 pin is malformed")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise PacketPreflightError("Source packet size/type changed after preflight")
        while True:
            _check_deadline(deadline, "post-load source packet rehash")
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            del chunk
        after = os.fstat(stream.fileno())
    if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
            (after.st_size, after.st_mtime_ns, after.st_ino) or
            digest.hexdigest() != expected_sha):
        raise PacketPreflightError("Source packet changed after decoded array validation")


def _counterpart_array_info(value: object, label: str) -> dict:
    info, _ = _metadata_record(value, label)
    return info


def _canonical_entry_sha256(value: Mapping[str, object]) -> str:
    import json

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_source_template(
    capture_root: object,
    entry: Mapping[str, object],
    *,
    expected_context_id: str,
    expected_frame_id: int,
    expected_offset_deg: int,
    real_frame_mask_metadata: Mapping[str, object],
    synthetic_mask_metadata: Mapping[str, object] | None = None,
    require_synthetic_mask: bool = False,
    resident_bytes: int = 0,
    deadline_monotonic: float | None = None,
) -> dict:
    """Load one already-preflighted template via R5's source-only loader.

    Counterpart arguments are metadata records only; their paths are never
    accessed. The returned projection excludes ``template_gray_rgb`` and
    ``seed_pose_m`` and contains no query, evaluator, or result fields.
    """
    deadline = _loader_deadline(deadline_monotonic)
    if not isinstance(real_frame_mask_metadata, Mapping):
        raise PacketPreflightError("A real-frame observed-mask metadata record is required")
    counterpart_metadata = {
        "real_frame": _counterpart_array_info(
            real_frame_mask_metadata, "real-frame observed crop mask"),
    }
    if require_synthetic_mask and synthetic_mask_metadata is None:
        raise PacketPreflightError("This frozen plan requires synthetic-query mask metadata")
    if synthetic_mask_metadata is not None:
        if not isinstance(synthetic_mask_metadata, Mapping):
            raise PacketPreflightError("Synthetic-query observed-mask metadata must be an array-info record")
        counterpart_metadata["synthetic_query"] = _counterpart_array_info(
            synthetic_mask_metadata, "synthetic-query observed crop mask")
    preflight = preflight_template_packet(
        capture_root, entry,
        expected_context_id=expected_context_id,
        expected_frame_id=expected_frame_id,
        expected_offset_deg=expected_offset_deg,
        counterpart_mask_metadata=counterpart_metadata,
        resident_bytes=resident_bytes,
        deadline_monotonic=deadline,
    )
    _check_deadline(deadline, "source entry projection")

    geometry_hashes = entry.get("geometry_hashes") if isinstance(entry, Mapping) else None
    if not isinstance(geometry_hashes, Mapping) or set(geometry_hashes) != {"depth_mm", "mask"}:
        raise PacketPreflightError("Template context lacks the exact source depth/mask geometry hashes")
    if (geometry_hashes.get("depth_mm") != preflight["array_metadata"]["template_depth_mm"]["sha256"] or
            geometry_hashes.get("mask") != preflight["array_metadata"]["template_mask"]["sha256"]):
        raise PacketPreflightError("Template geometry hashes differ from the preflighted source arrays")

    import json

    sanitized_entry = json.loads(json.dumps({
        "context_id": expected_context_id,
        "role": "template",
        "frame_id": expected_frame_id,
        "template_offset_deg": expected_offset_deg,
        "path": preflight["packet"]["relative_path"],
        "bytes": preflight["packet"]["bytes"],
        "sha256": preflight["packet"]["sha256"],
        "arrays": preflight["array_metadata"],
        "geometry_hashes": {
            "depth_mm": geometry_hashes["depth_mm"],
            "mask": geometry_hashes["mask"],
        },
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    sanitized_entry_sha256 = _canonical_entry_sha256(sanitized_entry)
    minimal_bundle = {
        "capture_root": Path(capture_root).resolve(strict=True),
        "r1_contexts": {expected_context_id: sanitized_entry},
    }
    minimal_plan = {
        "frame_id": expected_frame_id,
        "template_offset_deg": expected_offset_deg,
        "condition_id": f"source-packet-preflight:{expected_context_id}",
    }

    # Import only after the strict metadata, file-pin, path, and ZIP checks.
    from . import quality_bottle_pose_ablation as r5

    _check_deadline(deadline, "R5 source-only loader entry")
    returned_entry, loaded_arrays = r5._condition_source_only_inputs(minimal_bundle, minimal_plan)
    if (returned_entry is not sanitized_entry or
            _canonical_entry_sha256(sanitized_entry) != sanitized_entry_sha256 or
            _canonical_entry_sha256(returned_entry) != sanitized_entry_sha256):
        raise PacketPreflightError("R5 source-only loader returned a different template entry")
    if not isinstance(loaded_arrays, Mapping) or set(loaded_arrays) != set(SOURCE_ARRAY_KEYS):
        raise PacketPreflightError("R5 source-only loader returned an unexpected array set")

    audit = r5.audit
    decoded_bytes = int(preflight["budget"]["decoded_array_bytes"])
    max_array_bytes = max(
        math.prod(info["shape"]) * {"float32": 4, "float64": 8, "uint8": 1, "int64": 8}[info["dtype"]]
        for info in preflight["array_metadata"].values())
    live_estimate = resident_bytes + decoded_bytes + max_array_bytes + HASH_CHUNK_BYTES + DECODE_FIXED_ALLOWANCE_BYTES
    if live_estimate > MAX_WORKING_BYTES:
        raise PacketResourceLimitError("Loaded source arrays exceed the 128 MiB resident working cap")
    for name in SOURCE_ARRAY_KEYS:
        _check_deadline(deadline, f"decoded metadata validation for {name}")
        if audit.array_info(loaded_arrays[name]) != preflight["array_metadata"][name]:
            raise PacketPreflightError(f"Decoded source array differs from preflight metadata: {name}")
    _verify_packet_file_pin(capture_root, preflight["packet"], deadline)

    projection_keys = tuple(key for key in SOURCE_ARRAY_KEYS
                            if key not in {"template_gray_rgb", "seed_pose_m"})
    projected_arrays = {key: loaded_arrays[key] for key in projection_keys}
    if set(projected_arrays) != set(projection_keys):
        raise AssertionError("Source-only projection did not retain the exact allowed fields")
    projected_metadata = {key: preflight["array_metadata"][key] for key in projection_keys}
    image_namespace = {
        "context_id": expected_context_id,
        "source_rgb_sha256": projected_metadata["template_rgb"]["sha256"],
    }
    _check_deadline(deadline, "source projection receipt sealing")
    receipt = {
        "schema_version": 1,
        "state": "loaded_source_template",
        "context_id": expected_context_id,
        "frame_id": expected_frame_id,
        "template_offset_deg": expected_offset_deg,
        "packet": dict(preflight["packet"]),
        "source_array_metadata": projected_metadata,
        "observed_crop_mask_metadata": preflight["observed_crop_mask_metadata"],
        "counterpart_mask_checks": list(preflight["counterpart_mask_checks"]),
        "archive_member_count": preflight["archive"]["member_count"],
        "packet_sha256_rechecked": True,
        "r5_source_only_loader_calls": 1,
        "dropped_source_arrays": ["template_gray_rgb", "seed_pose_m"],
        "resident_bytes": resident_bytes,
        "loaded_array_bytes": decoded_bytes,
        "prospective_peak_bytes": preflight["budget"]["peak_estimate_bytes"],
        "working_limit_bytes": MAX_WORKING_BYTES,
        "query_or_evaluator_data_read": False,
        "fitting_or_matching_started": False,
        "capacity_screen_complete": False,
        "counterpart_context_bindings_verified": False,
        "observed_mask_provenance": "caller_supplied_matching_array_metadata_only",
    }
    receipt["deadline_seconds_remaining"] = _check_deadline(
        deadline, "final source-loader receipt sealing")
    del loaded_arrays
    return {
        "arrays": projected_arrays,
        "receipt": receipt,
        "image_namespace": image_namespace,
    }
