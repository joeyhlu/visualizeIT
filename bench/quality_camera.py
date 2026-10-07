# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 VisualizeIT contributors.
# Modified: independent NumPy implementation with bounded calibration and warp APIs.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A full copy of the License is included at the end of this file.

"""Pure, content-bound HOT3D FISHEYE624 image-warp primitives.

This module has no media, model, filesystem, network, or native-library side
effects.  It implements the camera equations documented by the Apache-2.0
licensed Meta hand_tracking_toolkit at revision
``bc628e9286c18444b47a0971a89d075ea6f00747`` (camera.py and
camera_distortion.py).  The implementation is independent Python/NumPy code;
the pinned source closure and its license are retained under the excluded
``.cache/model-quality/generalization-metadata/camera-oracle`` directory.

The input is calibration-only metadata. Camera extrinsics are retained only in
the raw-input digest and never affect image projection. Output samples use
integer pixel centers, a fixed 720-square pinhole view, clockwise source-to-
output orientation, four-neighbor bilinear interpolation, RGB128 invalid fill,
and explicit round-half-up conversion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


UPSTREAM_REVISION = "bc628e9286c18444b47a0971a89d075ea6f00747"
UPSTREAM_CAMERA_SHA256 = "20565a0294208f6472e72216db258bef96f33f5e0c78c70ea0e72e7b1431a400"
UPSTREAM_DISTORTION_SHA256 = "83815230ee4401c1e45208149263f1606164452e710e0090179b18d2ef5cd047"
MAX_CALIBRATION_BYTES = 1_048_576
MAX_METADATA_BYTES = 16 * 1024
MAX_SOURCE_PIXELS = 1408 * 1408
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 720
OUTPUT_FOCAL = 360.0
OUTPUT_CENTER = 359.5
INVALID_RGB = (128, 128, 128)
MAX_TILE_ROWS = 32
WORKING_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024


class CalibrationError(ValueError):
    """Calibration data is unsupported, malformed, or outside safe bounds."""


class WarpError(ValueError):
    """A warp or RGB input does not match the fixed sampling contract."""


@dataclass(frozen=True, slots=True)
class NormalizedCalibration:
    """Immutable numeric FISHEYE624 calibration used by the pure projector."""

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    radial: tuple[float, float, float, float, float, float]
    p1: float
    p2: float
    prism: tuple[float, float, float, float]
    max_solid_angle: float
    valid_radius: float | None
    raw_calibration_sha256: str
    canonical_values_sha256: str
    projection_model_type: str = "CameraModelType.FISHEYE624"
    label: str | None = None
    serial_number: str | None = None
    stream_id: str | None = None

    @property
    def source_width(self) -> int:
        return self.image_width

    @property
    def source_height(self) -> int:
        return self.image_height


@dataclass(frozen=True, slots=True)
class PinholeOutput:
    """The single supported rectified output camera and orientation."""

    width: int = OUTPUT_WIDTH
    height: int = OUTPUT_HEIGHT
    fx: float = OUTPUT_FOCAL
    fy: float = OUTPUT_FOCAL
    cx: float = OUTPUT_CENTER
    cy: float = OUTPUT_CENTER
    # Column-vector R_output_from_source, row-major.
    rotation_output_from_source: tuple[float, ...] = (
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    )
    pixel_center_convention: str = "integer-centers; edges[-0.5,size-0.5]"


@dataclass(frozen=True, slots=True)
class WarpMaps:
    """Output-to-source maps and separated geometry/domain/sample masks."""

    calibration: NormalizedCalibration
    output: PinholeOutput
    map_x: np.ndarray
    map_y: np.ndarray
    geometric_valid: np.ndarray
    domain_valid: np.ndarray
    sampling_valid: np.ndarray
    tile_rows: int
    map_sha256: str
    geometric_valid_sha256: str
    domain_valid_sha256: str
    sampling_valid_sha256: str
    warp_sha256: str
    peak_working_bytes_limit: int = WORKING_MEMORY_LIMIT_BYTES


@dataclass(frozen=True, slots=True, init=False)
class WarpedRGB:
    """Rectified RGB pixels and an immutable, content-bound metadata snapshot.

    ``metadata`` remains a JSON-compatible mapping API, but each access decodes
    a detached copy from canonical immutable JSON bytes. Mutating that copy
    cannot change the authoritative snapshot. ``metadata_json_bytes`` exposes
    the exact canonical snapshot and ``metadata_sha256`` binds those bytes.
    """

    rgb: np.ndarray
    sampling_valid: np.ndarray
    calibration_sha256: str
    canonical_calibration_sha256: str
    warp_sha256: str
    rgb_sha256: str
    sampling_valid_sha256: str
    output_sha256: str
    metadata_sha256: str
    _metadata_json: bytes

    def __init__(
        self,
        rgb: np.ndarray,
        sampling_valid: np.ndarray,
        calibration_sha256: str,
        warp_sha256: str,
        rgb_sha256: str,
        sampling_valid_sha256: str,
        output_sha256: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        canonical_calibration_sha256: str | None = None,
        metadata_sha256: str | None = None,
        _metadata_json: bytes | None = None,
    ) -> None:
        # Keep the former metadata= mapping constructor for callers, while
        # dataclasses.replace can round-trip the authoritative private bytes.
        if metadata is not None and _metadata_json is not None:
            raise WarpError("provide metadata mapping or canonical metadata bytes, not both")
        if _metadata_json is None:
            if not isinstance(metadata, Mapping):
                raise WarpError("warped RGB metadata must be a JSON mapping")
            try:
                metadata_document = dict(metadata)
                estimated_bytes = _estimated_json_bytes(metadata_document)
                if estimated_bytes > MAX_METADATA_BYTES:
                    raise WarpError("warped RGB metadata exceeds the 16 KiB limit")
                metadata_bytes = _canonical_json(metadata_document)
                if len(metadata_bytes) > MAX_METADATA_BYTES:
                    raise WarpError("warped RGB metadata exceeds the 16 KiB limit")
            except WarpError:
                raise
            except (CalibrationError, TypeError, ValueError) as exc:
                raise WarpError("warped RGB metadata is not canonical JSON data") from exc
            computed_metadata_sha256 = _digest_bytes(metadata_bytes)
            if metadata_sha256 is not None and metadata_sha256 != computed_metadata_sha256:
                raise WarpError("warped RGB metadata digest does not match its JSON bytes")
            metadata_sha256 = computed_metadata_sha256
        else:
            if type(_metadata_json) is not bytes:
                raise WarpError("authoritative metadata snapshot must be immutable bytes")
            metadata_bytes = _metadata_json
            if len(metadata_bytes) > MAX_METADATA_BYTES:
                raise WarpError("warped RGB metadata exceeds the 16 KiB limit")
            if metadata is not None:
                raise WarpError("provide metadata mapping or canonical metadata bytes, not both")
        if canonical_calibration_sha256 is None and metadata is not None:
            canonical_calibration_sha256 = metadata.get("canonical_calibration_sha256")

        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "sampling_valid", sampling_valid)
        object.__setattr__(self, "calibration_sha256", calibration_sha256)
        object.__setattr__(self, "canonical_calibration_sha256", canonical_calibration_sha256)
        object.__setattr__(self, "warp_sha256", warp_sha256)
        object.__setattr__(self, "rgb_sha256", rgb_sha256)
        object.__setattr__(self, "sampling_valid_sha256", sampling_valid_sha256)
        object.__setattr__(self, "output_sha256", output_sha256)
        object.__setattr__(self, "metadata_sha256", metadata_sha256)
        object.__setattr__(self, "_metadata_json", metadata_bytes)
        self.__post_init__()

    def __post_init__(self) -> None:
        self._validate_metadata_snapshot()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a detached JSON mapping for backward-compatible callers."""

        return self.export_metadata()

    @property
    def metadata_json_bytes(self) -> bytes:
        """Return the immutable canonical JSON bytes bound by metadata_sha256."""

        self._validate_metadata_snapshot()
        return self._metadata_json

    def export_metadata(self) -> dict[str, Any]:
        """Decode and return a detached JSON-safe metadata mapping."""

        self._validate_metadata_snapshot()
        return _decode_metadata_json(self._metadata_json)

    def _validate_metadata_snapshot(self) -> None:
        if not _is_sha256_hex(self.metadata_sha256):
            raise WarpError("warped RGB metadata SHA-256 has an invalid format")
        if not _is_sha256_hex(self.calibration_sha256):
            raise WarpError("warped RGB calibration SHA-256 has an invalid format")
        if not _is_sha256_hex(self.canonical_calibration_sha256):
            raise WarpError("warped RGB canonical calibration SHA-256 has an invalid format")
        if not _is_sha256_hex(self.warp_sha256):
            raise WarpError("warped RGB warp SHA-256 has an invalid format")
        if not _is_sha256_hex(self.rgb_sha256):
            raise WarpError("warped RGB source SHA-256 has an invalid format")
        if not _is_sha256_hex(self.sampling_valid_sha256):
            raise WarpError("warped RGB sampling SHA-256 has an invalid format")
        if not _is_sha256_hex(self.output_sha256):
            raise WarpError("warped RGB output SHA-256 has an invalid format")
        if type(self._metadata_json) is not bytes:
            raise WarpError("authoritative metadata snapshot must be immutable bytes")
        if len(self._metadata_json) > MAX_METADATA_BYTES:
            raise WarpError("warped RGB metadata exceeds the 16 KiB limit")
        if _digest_bytes(self._metadata_json) != self.metadata_sha256:
            raise WarpError("warped RGB metadata digest mismatch")

        document = _decode_metadata_json(self._metadata_json)
        expected = {
            "schema": "quality-camera-warped-rgb-v1",
            "source_rgb_sha256": self.rgb_sha256,
            "calibration_sha256": self.calibration_sha256,
            "canonical_calibration_sha256": self.canonical_calibration_sha256,
            "warp_sha256": self.warp_sha256,
            "sampling_valid_sha256": self.sampling_valid_sha256,
            "output_rgb_sha256": self.output_sha256,
            "output": _output_metadata(PinholeOutput()),
            "sampling": _sampling_metadata(),
        }
        if document != expected:
            raise WarpError("warped RGB metadata is inconsistent with its digests or configuration")

        if (
            not isinstance(self.rgb, np.ndarray)
            or self.rgb.dtype != np.uint8
            or self.rgb.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)
        ):
            raise WarpError("warped RGB record pixels must be a 720x720 uint8 RGB array")
        if (
            not isinstance(self.sampling_valid, np.ndarray)
            or self.sampling_valid.dtype != np.bool_
            or self.sampling_valid.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH)
        ):
            raise WarpError("warped RGB sampling-valid record must be a 720x720 boolean map")
        if _digest_bytes(memoryview(np.ascontiguousarray(self.rgb)).cast("B")) != self.output_sha256:
            raise WarpError("warped RGB pixels do not match the bound output digest")
        if _hash_single_array("sampling_valid", self.sampling_valid, "|u1") != self.sampling_valid_sha256:
            raise WarpError("warped RGB validity map does not match the bound sampling digest")
        self.rgb.setflags(write=False)
        self.sampling_valid.setflags(write=False)


def _reject_constant(value: str) -> None:
    raise CalibrationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_json_tree(value: Any, path: str = "calibration") -> None:
    """Reject non-JSON values, booleans where numeric values are later read,
    and exponent overflow (JSON ``1e999`` parses as an infinity in Python).
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CalibrationError(f"{path} has a non-string key")
            _finite_json_tree(child, f"{path}.{key}")
        return
    raise CalibrationError(f"{path} is not JSON data")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise CalibrationError("calibration cannot be canonically encoded") from exc


def _is_sha256_hex(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _decode_metadata_json(data: bytes) -> dict[str, Any]:
    """Decode one canonical metadata object without accepting mutable aliases."""

    if type(data) is not bytes:
        raise WarpError("authoritative metadata snapshot must be immutable bytes")
    if len(data) > MAX_METADATA_BYTES:
        raise WarpError("warped RGB metadata exceeds the 16 KiB limit")
    try:
        document = json.loads(
            data.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(document) is not dict:
            raise WarpError("warped RGB metadata JSON root must be an object")
        if _canonical_json(document) != data:
            raise WarpError("warped RGB metadata bytes are not canonical JSON")
        return document
    except WarpError:
        raise
    except (CalibrationError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise WarpError("warped RGB metadata must be bounded canonical UTF-8 JSON") from exc


def _estimated_json_bytes(value: Any, *, depth: int = 0) -> int:
    """Conservative bounded-size estimate before canonical JSON allocation."""

    if depth > 64:
        raise CalibrationError("calibration JSON nesting exceeds 64 levels")
    if value is None:
        return 4
    if value is True:
        return 4
    if value is False:
        return 5
    if isinstance(value, str):
        # ensure_ascii=True emits at most six ASCII bytes per code point.
        return 2 + 6 * len(value)
    if isinstance(value, int):
        bits = abs(value).bit_length()
        return 2 + (bits * 30103 // 100000) + (1 if value < 0 else 0)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalibrationError("calibration contains a non-finite number")
        return 32
    if type(value) is list:
        total = 2
        for child in value:
            total += _estimated_json_bytes(child, depth=depth + 1) + 1
            if total > MAX_CALIBRATION_BYTES:
                raise CalibrationError("calibration JSON exceeds the 1 MiB limit")
        return total
    if type(value) is dict:
        total = 2
        for key, child in value.items():
            if not isinstance(key, str):
                raise CalibrationError("calibration has a non-string key")
            total += 2 + 6 * len(key) + 1
            total += _estimated_json_bytes(child, depth=depth + 1) + 1
            if total > MAX_CALIBRATION_BYTES:
                raise CalibrationError("calibration JSON exceeds the 1 MiB limit")
        return total
    raise CalibrationError("calibration contains a non-JSON value")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{field} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise CalibrationError(f"{field} must be a finite real number") from exc
    if not math.isfinite(result):
        raise CalibrationError(f"{field} must be a finite real number")
    return result


def _positive_dimension(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{field} must be a positive integer")
    return value


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unpack_calibration(
    calibration: dict[str, Any] | bytes | bytearray | memoryview,
) -> tuple[dict[str, Any], str]:
    if isinstance(calibration, (bytes, bytearray, memoryview)):
        input_size = calibration.nbytes if isinstance(calibration, memoryview) else len(calibration)
        if input_size > MAX_CALIBRATION_BYTES:
            raise CalibrationError("calibration JSON exceeds the 1 MiB limit")
        raw = bytes(calibration)
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except CalibrationError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise CalibrationError("calibration must be bounded UTF-8 JSON") from exc
        raw_sha256 = _digest_bytes(raw)
    elif type(calibration) is dict:
        estimate = _estimated_json_bytes(calibration)
        if estimate > MAX_CALIBRATION_BYTES:
            raise CalibrationError("calibration JSON exceeds the 1 MiB limit")
        canonical_input = _canonical_json(calibration)
        if len(canonical_input) > MAX_CALIBRATION_BYTES:
            raise CalibrationError("calibration JSON exceeds the 1 MiB limit")
        try:
            value = json.loads(canonical_input.decode("ascii"), parse_constant=_reject_constant)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise CalibrationError("calibration mapping cannot be normalized") from exc
        raw_sha256 = _digest_bytes(canonical_input)
    else:
        raise CalibrationError("calibration must be a mapping or JSON bytes")

    if not isinstance(value, dict):
        raise CalibrationError("calibration JSON root must be an object")
    if _estimated_json_bytes(value) > MAX_CALIBRATION_BYTES:
        raise CalibrationError("calibration JSON exceeds the 1 MiB limit")
    _finite_json_tree(value)
    if set(value) == {"calibration"}:
        value = value["calibration"]
    if not isinstance(value, dict):
        raise CalibrationError("calibration must be a calibration-only object")
    return value, raw_sha256


_CALIBRATION_FIELDS = frozenset(
    {
        "image_width",
        "image_height",
        "projection_model_type",
        "projection_params",
        "max_solid_angle",
        "valid_radius",
        "label",
        "serial_number",
        "stream_id",
        "T_device_from_camera",
    }
)


def _canonical_calibration_values(
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    radial: tuple[float, float, float, float, float, float],
    p1: float,
    p2: float,
    prism: tuple[float, float, float, float],
    max_solid_angle: float,
    valid_radius: float | None,
    label: str | None,
    serial_number: str | None,
    stream_id: str | None,
) -> dict[str, Any]:
    """Return the one canonical numeric record used by parser and validator."""

    return {
        "projection_model_type": "CameraModelType.FISHEYE624",
        "image_width": image_width,
        "image_height": image_height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "radial": list(radial),
        "p1": p1,
        "p2": p2,
        "prism": list(prism),
        "max_solid_angle": max_solid_angle,
        "valid_radius": valid_radius,
        "label": label,
        "serial_number": serial_number,
        "stream_id": stream_id,
    }


def _validate_normalized_calibration(calibration: NormalizedCalibration) -> NormalizedCalibration:
    """Strictly revalidate public records and their verifiable provenance."""

    if type(calibration) is not NormalizedCalibration:
        raise CalibrationError("normalized calibration must use the exact supported record type")

    width, height = calibration.image_width, calibration.image_height
    if type(width) is not int or width <= 0:
        raise CalibrationError("image_width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise CalibrationError("image_height must be a positive integer")
    if width > 1408 or height > 1408 or width * height > MAX_SOURCE_PIXELS:
        raise CalibrationError("source image exceeds the 1408x1408 pixel budget")

    if type(calibration.projection_model_type) is not str or calibration.projection_model_type != "CameraModelType.FISHEYE624":
        raise CalibrationError("only CameraModelType.FISHEYE624 is supported")

    fx = _number(calibration.fx, "fx")
    fy = _number(calibration.fy, "fy")
    cx = _number(calibration.cx, "cx")
    cy = _number(calibration.cy, "cy")
    if fx <= 0.0 or fy <= 0.0:
        raise CalibrationError("FISHEYE624 focal lengths must be positive")

    if type(calibration.radial) is not tuple or len(calibration.radial) != 6:
        raise CalibrationError("radial coefficients must be an immutable six-value tuple")
    radial = tuple(_number(value, f"radial[{index}]") for index, value in enumerate(calibration.radial))
    p1 = _number(calibration.p1, "p1")
    p2 = _number(calibration.p2, "p2")
    if type(calibration.prism) is not tuple or len(calibration.prism) != 4:
        raise CalibrationError("prism coefficients must be an immutable four-value tuple")
    prism = tuple(_number(value, f"prism[{index}]") for index, value in enumerate(calibration.prism))

    theta = _number(calibration.max_solid_angle, "max_solid_angle")
    if not 0.0 < theta < (math.pi / 2.0):
        raise CalibrationError("max_solid_angle must be strictly between zero and pi/2 radians")
    radius = None if calibration.valid_radius is None else _number(calibration.valid_radius, "valid_radius")
    if radius is not None and radius <= 0.0:
        raise CalibrationError("valid_radius must be positive when declared")

    strings: dict[str, str | None] = {}
    for field_name in ("label", "serial_number", "stream_id"):
        value = getattr(calibration, field_name)
        if value is not None and type(value) is not str:
            raise CalibrationError(f"{field_name} must be a string when present")
        strings[field_name] = value

    if not _is_sha256_hex(calibration.raw_calibration_sha256):
        raise CalibrationError("raw_calibration_sha256 must be 64 lowercase hexadecimal characters")
    if not _is_sha256_hex(calibration.canonical_values_sha256):
        raise CalibrationError("canonical_values_sha256 must be 64 lowercase hexadecimal characters")

    canonical_values = _canonical_calibration_values(
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        radial,  # type: ignore[arg-type]
        p1,
        p2,
        prism,  # type: ignore[arg-type]
        theta,
        radius,
        strings["label"],
        strings["serial_number"],
        strings["stream_id"],
    )
    if _estimated_json_bytes(canonical_values) > MAX_CALIBRATION_BYTES:
        raise CalibrationError("normalized calibration exceeds the 1 MiB limit")
    expected_sha256 = _digest_bytes(_canonical_json(canonical_values))
    if calibration.canonical_values_sha256 != expected_sha256:
        raise CalibrationError("normalized calibration values do not match their canonical SHA-256")
    numeric_fields = (calibration.fx, calibration.fy, calibration.cx, calibration.cy,
                      calibration.p1, calibration.p2, calibration.max_solid_angle)
    canonical_numeric_types = all(type(value) is float for value in numeric_fields)
    canonical_numeric_types = canonical_numeric_types and all(
        type(value) is float for value in (*calibration.radial, *calibration.prism)
    )
    canonical_numeric_types = canonical_numeric_types and (
        calibration.valid_radius is None or type(calibration.valid_radius) is float
    )
    if canonical_numeric_types:
        return calibration
    # Re-materialize the validated values so an instance built by callers with
    # integer-valued floats cannot pass validation and later serialize a
    # different JSON representation from the one bound by the canonical digest.
    return NormalizedCalibration(
        image_width=width,
        image_height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        radial=radial,  # type: ignore[arg-type]
        p1=p1,
        p2=p2,
        prism=prism,  # type: ignore[arg-type]
        max_solid_angle=theta,
        valid_radius=radius,
        raw_calibration_sha256=calibration.raw_calibration_sha256,
        canonical_values_sha256=calibration.canonical_values_sha256,
        projection_model_type=calibration.projection_model_type,
        label=strings["label"],
        serial_number=strings["serial_number"],
        stream_id=strings["stream_id"],
    )


def _validate_unused_transform(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) not in (3, 4):
        raise CalibrationError("T_device_from_camera must be a finite 3x4 or 4x4 list")
    if any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise CalibrationError("T_device_from_camera must be a finite 3x4 or 4x4 list")
    for ri, row in enumerate(value):
        for ci, scalar in enumerate(row):
            _number(scalar, f"T_device_from_camera[{ri}][{ci}]")


def normalize_calibration(
    calibration: NormalizedCalibration | dict[str, Any] | bytes | bytearray | memoryview,
) -> NormalizedCalibration:
    """Parse and freeze one calibration-only FISHEYE624 record.

    Byte inputs are bounded to 1 MiB and are hashed exactly as received. A
    mapping input has a stable compact-JSON digest because it has no original
    byte representation. The optional sole-key ``calibration`` wrapper is
    accepted; adjacent pose or annotation data is rejected.
    """

    if isinstance(calibration, NormalizedCalibration):
        return _validate_normalized_calibration(calibration)
    record, raw_sha256 = _unpack_calibration(calibration)
    unknown = set(record) - _CALIBRATION_FIELDS
    if unknown:
        raise CalibrationError(f"unsupported calibration fields: {sorted(unknown)!r}")

    width = _positive_dimension(record.get("image_width"), "image_width")
    height = _positive_dimension(record.get("image_height"), "image_height")
    if width > 1408 or height > 1408 or width * height > MAX_SOURCE_PIXELS:
        raise CalibrationError("source image exceeds the 1408x1408 pixel budget")
    model = record.get("projection_model_type")
    if model != "CameraModelType.FISHEYE624":
        raise CalibrationError("only CameraModelType.FISHEYE624 is supported")
    params = record.get("projection_params")
    if not isinstance(params, list) or len(params) not in (15, 16):
        raise CalibrationError("FISHEYE624 parameters must contain exactly 15 or 16 values")
    values = tuple(_number(value, f"projection_params[{i}]") for i, value in enumerate(params))
    if len(values) == 15:
        focal, cx, cy = values[:3]
        fx = fy = focal
        distortion = values[3:]
    else:
        fx, fy, cx, cy = values[:4]
        distortion = values[4:]
    if fx <= 0 or fy <= 0:
        raise CalibrationError("FISHEYE624 focal lengths must be positive")
    theta = _number(record.get("max_solid_angle"), "max_solid_angle")
    if not 0 < theta < (math.pi / 2):
        raise CalibrationError("max_solid_angle must be strictly between zero and pi/2 radians")
    radius_value = record.get("valid_radius")
    radius = None if radius_value is None else _number(radius_value, "valid_radius")
    if radius is not None and radius <= 0:
        raise CalibrationError("valid_radius must be positive when declared")

    for field in ("label", "serial_number", "stream_id"):
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            raise CalibrationError(f"{field} must be a string when present")
    _validate_unused_transform(record.get("T_device_from_camera"))

    canonical = _canonical_calibration_values(
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        tuple(distortion[:6]),  # type: ignore[arg-type]
        distortion[6],
        distortion[7],
        tuple(distortion[8:12]),  # type: ignore[arg-type]
        theta,
        radius,
        record.get("label"),
        record.get("serial_number"),
        record.get("stream_id"),
    )
    canonical_sha256 = _digest_bytes(_canonical_json(canonical))
    return NormalizedCalibration(
        image_width=width,
        image_height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        radial=tuple(distortion[:6]),  # type: ignore[arg-type]
        p1=distortion[6],
        p2=distortion[7],
        prism=tuple(distortion[8:12]),  # type: ignore[arg-type]
        max_solid_angle=theta,
        valid_radius=radius,
        raw_calibration_sha256=raw_sha256,
        canonical_values_sha256=canonical_sha256,
        projection_model_type=model,
        label=record.get("label"),
        serial_number=record.get("serial_number"),
        stream_id=record.get("stream_id"),
    )


def calibration_metadata(
    calibration: NormalizedCalibration | dict[str, Any] | bytes | bytearray | memoryview,
) -> dict[str, Any]:
    """Return JSON-safe normalized calibration values and their two digests."""

    cal = normalize_calibration(calibration)
    return {
        "schema": "quality-camera-calibration-v1",
        "projection_model_type": cal.projection_model_type,
        "source_width": cal.image_width,
        "source_height": cal.image_height,
        "fx": cal.fx,
        "fy": cal.fy,
        "cx": cal.cx,
        "cy": cal.cy,
        "radial": list(cal.radial),
        "tangential_p1_p2": [cal.p1, cal.p2],
        "thin_prism_s1_s2_s3_s4": list(cal.prism),
        "max_solid_angle_radians": cal.max_solid_angle,
        "valid_radius_pixels": cal.valid_radius,
        "label": cal.label,
        "serial_number": cal.serial_number,
        "stream_id": cal.stream_id,
        "raw_calibration_sha256": cal.raw_calibration_sha256,
        "canonical_values_sha256": cal.canonical_values_sha256,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_camera_sha256": UPSTREAM_CAMERA_SHA256,
        "upstream_distortion_sha256": UPSTREAM_DISTORTION_SHA256,
    }


def _project_chunk(rays: np.ndarray, cal: NormalizedCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one bounded N-by-3 chunk, preserving a separate usable domain."""

    n = rays.shape[0]
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    geometric = np.zeros((n,), dtype=np.bool_)
    domain = np.zeros((n,), dtype=np.bool_)
    finite = np.isfinite(rays).all(axis=1)
    if not finite.any():
        return uv, geometric, domain

    # Scale each ray before any norm/square operation. Projection depends only
    # on direction; this keeps finite values near float64 limits safe.
    safe = np.where(finite[:, None], rays, 0.0)
    scale = np.max(np.abs(safe), axis=1)
    nonzero = scale > 0.0
    normalized = safe / np.where(nonzero, scale, 1.0)[:, None]
    front = normalized[:, 2] > 0.0
    candidate = finite & nonzero & front
    if not candidate.any():
        return uv, geometric, domain

    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]
    rho = np.hypot(x, y)
    angle = np.arctan2(rho, z)
    factor = np.divide(angle, rho, out=np.zeros_like(angle), where=rho > 0.0)
    x0 = x * factor
    y0 = y * factor

    # OVRFisheye624: angular plane, six radial terms, p2/p1 tangent order,
    # then thin-prism terms. Tangent and prism use radial-distorted r^2.
    r2 = np.minimum(x0 * x0 + y0 * y0, math.pi * math.pi)
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r4 * r4
    r10 = r8 * r2
    r12 = r6 * r6
    k1, k2, k3, k4, k5, k6 = cal.radial
    with np.errstate(over="ignore", invalid="ignore"):
        radial_scale = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        xr = x0 * radial_scale
        yr = y0 * radial_scale
        r2r = xr * xr + yr * yr
        r4r = r2r * r2r
        xt = xr + 2.0 * cal.p2 * xr * yr + cal.p1 * (r2r + 2.0 * xr * xr)
        yt = yr + 2.0 * cal.p1 * xr * yr + cal.p2 * (r2r + 2.0 * yr * yr)
        xd = xt + cal.prism[0] * r2r + cal.prism[1] * r4r
        yd = yt + cal.prism[2] * r2r + cal.prism[3] * r4r
        u = xd * cal.fx + cal.cx
        v = yd * cal.fy + cal.cy
    finite_projection = np.isfinite(u) & np.isfinite(v) & candidate
    uv[:, 0] = np.where(finite_projection, u, np.nan)
    uv[:, 1] = np.where(finite_projection, v, np.nan)
    geometric[:] = finite_projection

    # max_solid_angle is the polar half-angle (radians). Sensor coordinates
    # span edge coordinates [-.5, size-.5]; radius, if present, is a second
    # declared disk restriction centered on the principal point.
    with np.errstate(over="ignore", invalid="ignore"):
        inside_rect = (
            (u >= -0.5)
            & (u <= cal.image_width - 0.5)
            & (v >= -0.5)
            & (v <= cal.image_height - 0.5)
        )
        if cal.valid_radius is None:
            inside_disk = np.ones_like(inside_rect)
        else:
            inside_disk = np.hypot(u - cal.cx, v - cal.cy) <= cal.valid_radius
    domain[:] = finite_projection & (angle <= cal.max_solid_angle) & inside_rect & inside_disk
    return uv, geometric, domain


def project_source_rays(
    rays: np.ndarray,
    calibration: NormalizedCalibration | dict[str, Any] | bytes | bytearray | memoryview,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ``[...,3]`` source-camera rays to ``[...,2]`` pixels.

    ``geometric_valid`` means finite, nonzero, front-facing rays with a finite
    projected coordinate. ``domain_valid`` further applies the angular limit,
    sensor edge rectangle, and optional principal-point-centered valid disk.
    Out-of-domain but geometrically projectable rays retain their coordinates.
    """

    cal = normalize_calibration(calibration)
    if not isinstance(rays, np.ndarray):
        raise ValueError("rays must be a NumPy array with final dimension 3")
    if rays.ndim < 1 or rays.shape[-1] != 3 or rays.size // 3 > MAX_SOURCE_PIXELS:
        raise ValueError("rays must have shape [...,3] within the source pixel budget")
    if rays.dtype.kind not in "iuf":
        raise ValueError("rays must use a real integer or floating dtype")
    shape = rays.shape[:-1]
    flat = rays.reshape((-1, 3))
    count = flat.shape[0]
    uv = np.full((count, 2), np.nan, dtype=np.float64)
    geometric = np.zeros((count,), dtype=np.bool_)
    domain = np.zeros((count,), dtype=np.bool_)
    # Conversion is per-chunk so an integer/float32 input does not require a
    # second full-resolution float64 copy alongside the output arrays.
    chunk_size = 16_384
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk = flat[start:stop].astype(np.float64, copy=False)
        out_uv, out_geometric, out_domain = _project_chunk(chunk, cal)
        uv[start:stop] = out_uv
        geometric[start:stop] = out_geometric
        domain[start:stop] = out_domain
    return (
        uv.reshape(shape + (2,)),
        geometric.reshape(shape),
        domain.reshape(shape),
    )


def _hash_array(hasher: Any, name: str, value: np.ndarray, dtype: str) -> None:
    canonical = np.asarray(value, dtype=np.dtype(dtype), order="C")
    hasher.update(name.encode("ascii") + b"\0")
    hasher.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii") + b"\0")
    hasher.update(dtype.encode("ascii") + b"\0")
    hasher.update(memoryview(canonical).cast("B"))


def _hash_single_array(name: str, value: np.ndarray, dtype: str) -> str:
    hasher = hashlib.sha256()
    _hash_array(hasher, name, value, dtype)
    return hasher.hexdigest()


def _sampling_footprint(
    map_x: np.ndarray,
    map_y: np.ndarray,
    domain: np.ndarray,
    calibration: NormalizedCalibration,
) -> np.ndarray:
    valid = np.zeros(domain.shape, dtype=np.bool_)
    for row_start in range(0, OUTPUT_HEIGHT, 16):
        row_stop = min(row_start + 16, OUTPUT_HEIGHT)
        x = map_x[row_start:row_stop]
        y = map_y[row_start:row_stop]
        in_domain = domain[row_start:row_stop]
        safe_x = np.where(in_domain, x, 0.0)
        safe_y = np.where(in_domain, y, 0.0)
        x0 = np.floor(safe_x).astype(np.int32)
        y0 = np.floor(safe_y).astype(np.int32)
        inside = (
            in_domain
            & (x0 >= 0)
            & (y0 >= 0)
            & (x0 + 1 < calibration.image_width)
            & (y0 + 1 < calibration.image_height)
        )
        if calibration.valid_radius is not None and inside.any():
            ax = x0.astype(np.float64)
            ay = y0.astype(np.float64)
            rad = calibration.valid_radius
            c_x, c_y = calibration.cx, calibration.cy
            inside &= (
                (np.hypot(ax - c_x, ay - c_y) <= rad)
                & (np.hypot(ax + 1.0 - c_x, ay - c_y) <= rad)
                & (np.hypot(ax - c_x, ay + 1.0 - c_y) <= rad)
                & (np.hypot(ax + 1.0 - c_x, ay + 1.0 - c_y) <= rad)
            )
        valid[row_start:row_stop] = inside
    return valid


def build_warp(
    calibration: NormalizedCalibration | dict[str, Any] | bytes | bytearray | memoryview,
    *,
    tile_rows: int = 16,
) -> WarpMaps:
    """Build deterministic float64 destination-to-source maps in bounded tiles."""

    cal = normalize_calibration(calibration)
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or not 1 <= tile_rows <= MAX_TILE_ROWS:
        raise ValueError(f"tile_rows must be an integer from 1 to {MAX_TILE_ROWS}")
    output = PinholeOutput()
    map_x = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH), -1.0, dtype=np.float64)
    map_y = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH), -1.0, dtype=np.float64)
    geometric = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.bool_)
    domain = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.bool_)
    u = np.arange(OUTPUT_WIDTH, dtype=np.float64)[None, :]
    for row_start in range(0, OUTPUT_HEIGHT, tile_rows):
        row_stop = min(row_start + tile_rows, OUTPUT_HEIGHT)
        v = np.arange(row_start, row_stop, dtype=np.float64)[:, None]
        # R_output_from_source.T @ K^-1 [u,v,1] = [y,-x,1].
        source_x = np.broadcast_to((v - output.cy) / output.fy, (row_stop - row_start, OUTPUT_WIDTH))
        source_y = np.broadcast_to(-(u - output.cx) / output.fx, (row_stop - row_start, OUTPUT_WIDTH))
        ones = np.ones_like(source_x)
        ray = np.stack((source_x, source_y, ones), axis=-1)
        uv, geom, usable = project_source_rays(ray, cal)
        map_x[row_start:row_stop] = np.where(geom, uv[..., 0], -1.0)
        map_y[row_start:row_stop] = np.where(geom, uv[..., 1], -1.0)
        geometric[row_start:row_stop] = geom
        domain[row_start:row_stop] = usable
    sampling = _sampling_footprint(map_x, map_y, domain, cal)

    # Freeze arrays before binding their digest into a returned record.
    for array in (map_x, map_y, geometric, domain, sampling):
        array.setflags(write=False)
    h_map = hashlib.sha256()
    _hash_array(h_map, "map_x", map_x, "<f8")
    _hash_array(h_map, "map_y", map_y, "<f8")
    map_sha = h_map.hexdigest()
    geom_sha = _hash_single_array("geometric_valid", geometric, "|u1")
    domain_sha = _hash_single_array("domain_valid", domain, "|u1")
    sampling_sha = _hash_single_array("sampling_valid", sampling, "|u1")
    warp_config = _warp_configuration(
        cal, output, tile_rows, map_sha, geom_sha, domain_sha, sampling_sha
    )
    warp_sha = _digest_bytes(_canonical_json(warp_config))
    return WarpMaps(
        calibration=cal,
        output=output,
        map_x=map_x,
        map_y=map_y,
        geometric_valid=geometric,
        domain_valid=domain,
        sampling_valid=sampling,
        tile_rows=tile_rows,
        map_sha256=map_sha,
        geometric_valid_sha256=geom_sha,
        domain_valid_sha256=domain_sha,
        sampling_valid_sha256=sampling_sha,
        warp_sha256=warp_sha,
    )


def _output_metadata(output: PinholeOutput) -> dict[str, Any]:
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


def _sampling_metadata() -> dict[str, Any]:
    return {
        "channel_order": "RGB",
        "interpolation": "bilinear-four-neighbor",
        "neighbor_footprint": "all four integer pixel centers must be in bounds and declared valid disk",
        "rounding": "floor(value+0.5)",
        "clip_range": [0, 255],
        "invalid_fill_rgb": list(INVALID_RGB),
    }


def _warp_configuration(
    calibration: NormalizedCalibration,
    output: PinholeOutput,
    tile_rows: int,
    map_sha256: str,
    geometric_valid_sha256: str,
    domain_valid_sha256: str,
    sampling_valid_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "quality-camera-warp-v1",
        "calibration_canonical_sha256": calibration.canonical_values_sha256,
        "map_sha256": map_sha256,
        "geometric_valid_sha256": geometric_valid_sha256,
        "domain_valid_sha256": domain_valid_sha256,
        "sampling_valid_sha256": sampling_valid_sha256,
        "output": _output_metadata(output),
        "sampling": _sampling_metadata(),
        "tile_rows": tile_rows,
        "working_memory_limit_bytes": WORKING_MEMORY_LIMIT_BYTES,
    }


def warp_metadata(warp: WarpMaps) -> dict[str, Any]:
    """Return all source/output configuration and map-content digests."""

    _validate_warp_structure(warp, verify_digests=True)
    return {
        "schema": "quality-camera-warp-v1",
        "calibration": calibration_metadata(warp.calibration),
        "output": _output_metadata(warp.output),
        "sampling": _sampling_metadata(),
        "tile_rows": warp.tile_rows,
        "working_memory_limit_bytes": warp.peak_working_bytes_limit,
        "map_sha256": warp.map_sha256,
        "geometric_valid_sha256": warp.geometric_valid_sha256,
        "domain_valid_sha256": warp.domain_valid_sha256,
        "sampling_valid_sha256": warp.sampling_valid_sha256,
        "warp_sha256": warp.warp_sha256,
    }


def _validate_warp_structure(warp: WarpMaps, *, verify_digests: bool) -> None:
    if not isinstance(warp, WarpMaps):
        raise WarpError("warp must be a WarpMaps record")
    try:
        cal = normalize_calibration(warp.calibration)
    except (CalibrationError, TypeError, AttributeError) as exc:
        raise WarpError("warp calibration is invalid") from exc
    if cal is not warp.calibration and cal != warp.calibration:
        raise WarpError("warp calibration is not normalized")
    if warp.output != PinholeOutput():
        raise WarpError("warp output camera does not match the fixed configuration")
    expected_shape = (OUTPUT_HEIGHT, OUTPUT_WIDTH)
    arrays = (warp.map_x, warp.map_y)
    masks = (warp.geometric_valid, warp.domain_valid, warp.sampling_valid)
    if any(not isinstance(array, np.ndarray) or array.shape != expected_shape or array.dtype != np.float64 for array in arrays):
        raise WarpError("warp coordinate maps must be 720x720 float64 arrays")
    if any(not isinstance(mask, np.ndarray) or mask.shape != expected_shape or mask.dtype != np.bool_ for mask in masks):
        raise WarpError("warp validity maps must be 720x720 boolean arrays")
    if np.any(warp.domain_valid & ~warp.geometric_valid) or np.any(warp.sampling_valid & ~warp.domain_valid):
        raise WarpError("warp validity masks violate their subset ordering")
    if not np.isfinite(warp.map_x).all() or not np.isfinite(warp.map_y).all():
        raise WarpError("warp coordinates must be finite")
    if isinstance(warp.tile_rows, bool) or not isinstance(warp.tile_rows, int) or not 1 <= warp.tile_rows <= MAX_TILE_ROWS:
        raise WarpError("warp tile size is invalid")
    if warp.peak_working_bytes_limit != WORKING_MEMORY_LIMIT_BYTES:
        raise WarpError("warp memory bound does not match the reviewed contract")
    if verify_digests:
        hm = hashlib.sha256()
        _hash_array(hm, "map_x", warp.map_x, "<f8")
        _hash_array(hm, "map_y", warp.map_y, "<f8")
        if hm.hexdigest() != warp.map_sha256:
            raise WarpError("warp coordinate map digest mismatch")
        if _hash_single_array("geometric_valid", warp.geometric_valid, "|u1") != warp.geometric_valid_sha256:
            raise WarpError("geometric validity digest mismatch")
        if _hash_single_array("domain_valid", warp.domain_valid, "|u1") != warp.domain_valid_sha256:
            raise WarpError("domain validity digest mismatch")
        if _hash_single_array("sampling_valid", warp.sampling_valid, "|u1") != warp.sampling_valid_sha256:
            raise WarpError("sampling validity digest mismatch")
        expected = _sampling_footprint(warp.map_x, warp.map_y, warp.domain_valid, cal)
        if not np.array_equal(expected, warp.sampling_valid):
            raise WarpError("sampling validity does not match the complete four-pixel footprint")
        configuration = _warp_configuration(
            cal,
            warp.output,
            warp.tile_rows,
            warp.map_sha256,
            warp.geometric_valid_sha256,
            warp.domain_valid_sha256,
            warp.sampling_valid_sha256,
        )
        if _digest_bytes(_canonical_json(configuration)) != warp.warp_sha256:
            raise WarpError("warp configuration digest mismatch")


def warp_rgb(rgb: np.ndarray, warp: WarpMaps) -> WarpedRGB:
    """Apply the fixed bilinear warp to an already decoded uint8 RGB image."""

    if not isinstance(rgb, np.ndarray):
        raise WarpError("rgb must be an already-decoded NumPy RGB array")
    _validate_warp_structure(warp, verify_digests=True)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape != (
        getattr(getattr(warp, "calibration", None), "image_height", -1),
        getattr(getattr(warp, "calibration", None), "image_width", -1),
        3,
    ):
        raise WarpError("rgb must have shape (source_height,source_width,3) and uint8 dtype")

    output = np.empty((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    output[...] = INVALID_RGB
    for row_start in range(0, OUTPUT_HEIGHT, 16):
        row_stop = min(row_start + 16, OUTPUT_HEIGHT)
        valid_local = warp.sampling_valid[row_start:row_stop]
        rr, cc = np.nonzero(valid_local)
        if rr.size == 0:
            continue
        rows = rr + row_start
        x = warp.map_x[rows, cc]
        y = warp.map_y[rows, cc]
        x0 = np.floor(x).astype(np.intp)
        y0 = np.floor(y).astype(np.intp)
        # Validate before indexing, even though build_warp already enforced it.
        if (
            np.any(x0 < 0)
            or np.any(y0 < 0)
            or np.any(x0 + 1 >= warp.calibration.image_width)
            or np.any(y0 + 1 >= warp.calibration.image_height)
        ):
            raise WarpError("sampling-valid map references an incomplete source footprint")
        dx = (x - x0).astype(np.float64)[:, None]
        dy = (y - y0).astype(np.float64)[:, None]
        top_left = rgb[y0, x0].astype(np.float64)
        top_right = rgb[y0, x0 + 1].astype(np.float64)
        bottom_left = rgb[y0 + 1, x0].astype(np.float64)
        bottom_right = rgb[y0 + 1, x0 + 1].astype(np.float64)
        values = (
            top_left * ((1.0 - dx) * (1.0 - dy))
            + top_right * (dx * (1.0 - dy))
            + bottom_left * ((1.0 - dx) * dy)
            + bottom_right * (dx * dy)
        )
        rounded = np.floor(np.clip(values, 0.0, 255.0) + 0.5).astype(np.uint8)
        output[rows, cc] = rounded

    valid = np.array(warp.sampling_valid, copy=True)
    output.setflags(write=False)
    valid.setflags(write=False)
    rgb_sha = _digest_bytes(memoryview(np.ascontiguousarray(rgb)).cast("B"))
    output_sha = _digest_bytes(memoryview(output).cast("B"))
    meta = {
        "schema": "quality-camera-warped-rgb-v1",
        "source_rgb_sha256": rgb_sha,
        "calibration_sha256": warp.calibration.raw_calibration_sha256,
        "canonical_calibration_sha256": warp.calibration.canonical_values_sha256,
        "warp_sha256": warp.warp_sha256,
        "sampling_valid_sha256": warp.sampling_valid_sha256,
        "output_rgb_sha256": output_sha,
        "output": _output_metadata(warp.output),
        "sampling": _sampling_metadata(),
    }
    return WarpedRGB(
        rgb=output,
        sampling_valid=valid,
        calibration_sha256=warp.calibration.raw_calibration_sha256,
        warp_sha256=warp.warp_sha256,
        rgb_sha256=rgb_sha,
        sampling_valid_sha256=warp.sampling_valid_sha256,
        output_sha256=output_sha,
        canonical_calibration_sha256=warp.calibration.canonical_values_sha256,
        metadata=meta,
    )

# The pinned upstream work is Apache-2.0 licensed; its complete license text follows.
APACHE_LICENSE_2_0 = r'''
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

'''
