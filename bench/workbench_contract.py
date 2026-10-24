"""Versioned, strictly validated JSON contracts for the video validation workbench.

Conventions (shared by every workbench module):

* Camera axes are OpenCV: x right, y down, z forward.
* Pixel coordinates use integer pixel centres: pixel ``(i, j)`` covers
  ``[i - 0.5, i + 0.5) x [j - 0.5, j + 0.5)``; ``K`` principal points use the
  same convention. Converting to/from COLMAP (corner origin) adds/subtracts 0.5.
* Poses are camera-from-object 4x4 matrices acting on column vectors.
* Geometry units are explicit: ``object_unit`` (scale-free, never aliased to
  metres) or ``metric``.

Every ``from_dict`` rejects unknown keys, non-finite numbers and booleans used
as numbers. ``to_dict`` output round-trips exactly. Nothing in this module reads
media, evaluator files or model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

CLIP_SCHEMA = "workbench_clip_v1"
SURFACE_SCHEMA = "workbench_surface_package_v1"
FRAME_RESULT_SCHEMA = "workbench_frame_result_v1"
RUN_SCHEMA = "workbench_run_v1"
REPORT_SCHEMA = "workbench_report_v1"

COORDINATE_CONVENTION = {
    "camera_axes": "opencv_x_right_y_down_z_forward",
    "pixel_origin": "integer_pixel_centers",
    "pose": "camera_from_object",
    "vectors": "column",
}

SOURCE_KINDS = ("video_file", "image_sequence")
CLOCK_KINDS = ("timestamp_seconds", "frame_index")
INTRINSICS_STATUS = ("measured", "dataset_reference", "unavailable")
UNITS = ("object_unit", "metric")
MASK_STATES = ("observed", "unavailable", "rejected", "blocked")
POSE_STATES = ("tracking", "recovering", "lost", "unavailable", "blocked")
RENDER_STATES = ("rendered", "suppressed", "unavailable")
BACKEND_ROLES = ("primary_neural", "frozen_baseline", "relative_geometry")
RUN_MODES = ("smoke", "focused", "regression")
ARTWORK_KINDS = ("checkerboard", "directional", "imported")

MAX_KEYFRAMES = 24
MAX_PREFIX_SECONDS = 10.0
MAX_FRAMES = 100_000
MAX_POINTS = 64
MAX_MESH_TRIANGLES = 20_000
MAX_PACKAGE_BYTES = 20_000_000
MAX_ARTWORK_EDGE = 1024
SMOKE_FRAME_COUNT = 10
NEW_INPUT_BUDGET_BYTES = 250_000_000
DERIVED_BUDGET_BYTES = 1 << 30
FREE_RESERVE_BYTES = 1 << 30

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ROOT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")


class ContractError(ValueError):
    """A workbench object violates its versioned contract."""


# --------------------------------------------------------------------------
# primitive validators

def _keys(value: Any, required: Iterable[str], optional: Iterable[str] = (), name: str = "object") -> dict:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    required, optional = set(required), set(optional)
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ContractError(f"{name} is missing {sorted(missing)}")
    if extra:
        raise ContractError(f"{name} has unknown keys {sorted(extra)}")
    return dict(value)


def _str(value: Any, name: str, *, max_len: int = 512, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty) or len(value) > max_len:
        raise ContractError(f"{name} must be a bounded string")
    if any(ord(ch) < 32 for ch in value):
        raise ContractError(f"{name} contains control characters")
    return value


def _opt_str(value: Any, name: str, **kw) -> str | None:
    return None if value is None else _str(value, name, **kw)


def _ident(value: Any, name: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ContractError(f"{name} must match {ID_RE.pattern}")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _opt_sha(value: Any, name: str) -> str | None:
    return None if value is None else _sha(value, name)


def _enum(value: Any, allowed: Sequence[str], name: str) -> str:
    if value not in allowed:
        raise ContractError(f"{name} must be one of {list(allowed)}")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int | None = 0, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{name} must be <= {maximum}")
    return value


def _float(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None,
           positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite")
    if positive and value <= 0:
        raise ContractError(f"{name} must be positive")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{name} must be <= {maximum}")
    return value


def _opt_float(value: Any, name: str, **kw) -> float | None:
    return None if value is None else _float(value, name, **kw)


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ContractError(f"{name} must contain {n} finite numbers")
    return tuple(_float(v, f"{name}[{i}]") for i, v in enumerate(value))


def _list(value: Any, name: str, *, max_len: int) -> list:
    if not isinstance(value, (list, tuple)) or len(value) > max_len:
        raise ContractError(f"{name} must be a list of at most {max_len} items")
    return list(value)


def _json_value(value: Any, name: str, depth: int = 0) -> Any:
    """Bounded plain JSON value with finite numbers only."""
    if depth > 8:
        raise ContractError(f"{name} is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str):
            _str(value, name, max_len=4096, allow_empty=True)
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _float(value, name)
    if isinstance(value, (list, tuple)):
        return [_json_value(v, name, depth + 1) for v in _list(value, name, max_len=4096)]
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ContractError(f"{name} has too many keys")
        return {_str(k, f"{name} key", max_len=128): _json_value(v, f"{name}.{k}", depth + 1)
                for k, v in value.items()}
    raise ContractError(f"{name} must be plain JSON")


def _pose(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContractError(f"{name} must be a 4x4 matrix")
    rows = tuple(_floats(row, 4, f"{name}[{i}]") for i, row in enumerate(value))
    if rows[3] != (0.0, 0.0, 0.0, 1.0):
        raise ContractError(f"{name} last row must be [0,0,0,1]")
    r = [row[:3] for row in rows[:3]]
    # Proper rotation: orthonormal and determinant +1 (tolerant to float noise).
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-4:
                raise ContractError(f"{name} rotation is not orthonormal")
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if abs(det - 1.0) > 1e-4:
        raise ContractError(f"{name} rotation determinant must be +1")
    return rows


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def loads_strict(raw: bytes | str) -> Any:
    def reject(token):
        raise ContractError(f"non-finite JSON value {token}")

    def pairs(items):
        out = {}
        for key, val in items:
            if key in out:
                raise ContractError(f"duplicate JSON key {key}")
            out[key] = val
        return out
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw, parse_constant=reject, object_pairs_hook=pairs)


# --------------------------------------------------------------------------
# clip

@dataclass(frozen=True)
class MediaRef:
    root_id: str
    path: str
    sha256: str | None
    bytes: int | None

    @classmethod
    def from_dict(cls, value: Any) -> "MediaRef":
        d = _keys(value, ("root_id", "path"), ("sha256", "bytes"), "media")
        root = d["root_id"]
        if not isinstance(root, str) or not ROOT_ID_RE.fullmatch(root):
            raise ContractError("media.root_id is invalid")
        path = _relative_path(d["path"], "media.path")
        size = d.get("bytes")
        return cls(root, path, _opt_sha(d.get("sha256"), "media.sha256"),
                   None if size is None else _int(size, "media.bytes", minimum=1))

    def to_dict(self) -> dict:
        return {"root_id": self.root_id, "path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def _relative_path(value: Any, name: str) -> str:
    _str(value, name, max_len=1024)
    if "\\" in value or "\x00" in value or ":" in value or value.startswith("/"):
        raise ContractError(f"{name} must be a POSIX relative path")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ContractError(f"{name} must not traverse")
    return value


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    source_frame_index: int
    timestamp_s: float | None = None
    media_time_s: float | None = None
    image_path: str | None = None
    image_sha256: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "FrameSpec":
        d = _keys(value, ("frame_id", "source_frame_index"),
                  ("timestamp_s", "media_time_s", "image_path", "image_sha256"), "frame")
        path = d.get("image_path")
        return cls(_ident(d["frame_id"], "frame_id"),
                   _int(d["source_frame_index"], "source_frame_index"),
                   _opt_float(d.get("timestamp_s"), "timestamp_s", minimum=0.0),
                   _opt_float(d.get("media_time_s"), "media_time_s", minimum=0.0),
                   None if path is None else _relative_path(path, "image_path"),
                   _opt_sha(d.get("image_sha256"), "image_sha256"))

    def to_dict(self) -> dict:
        out = {"frame_id": self.frame_id, "source_frame_index": self.source_frame_index,
               "timestamp_s": self.timestamp_s, "media_time_s": self.media_time_s}
        if self.image_path is not None or self.image_sha256 is not None:
            out["image_path"] = self.image_path
            out["image_sha256"] = self.image_sha256
        return out


@dataclass(frozen=True)
class Intrinsics:
    status: str
    K: tuple[float, ...] | None
    source: str

    @classmethod
    def from_dict(cls, value: Any) -> "Intrinsics":
        d = _keys(value, ("status", "K", "source"), (), "intrinsics")
        status = _enum(d["status"], INTRINSICS_STATUS, "intrinsics.status")
        K = d["K"]
        if status == "unavailable":
            if K is not None:
                raise ContractError("unavailable intrinsics must have K null")
        else:
            K = _pinhole_K(K, "intrinsics.K")
        return cls(status, K, _str(d["source"], "intrinsics.source"))

    def to_dict(self) -> dict:
        return {"status": self.status, "K": None if self.K is None else list(self.K), "source": self.source}

    def matrix(self):
        if self.K is None:
            raise ContractError("camera intrinsics are unavailable")
        import numpy as np
        return np.array(self.K, dtype=np.float64).reshape(3, 3)


@dataclass(frozen=True)
class Selection:
    """User/reviewer prompt points in source pixels (integer pixel centres)."""
    frame_id: str
    positive: tuple[tuple[float, float], ...]
    negative: tuple[tuple[float, float], ...]
    reviewed: bool
    reviewed_by: str | None

    @classmethod
    def from_dict(cls, value: Any) -> "Selection":
        d = _keys(value, ("frame_id", "positive", "negative", "reviewed"), ("reviewed_by",), "selection")
        pos = tuple(_point(p, "positive") for p in _list(d["positive"], "positive", max_len=MAX_POINTS))
        neg = tuple(_point(p, "negative") for p in _list(d["negative"], "negative", max_len=MAX_POINTS))
        if not pos:
            raise ContractError("selection requires at least one positive point")
        return cls(_ident(d["frame_id"], "selection.frame_id"), pos, neg,
                   _bool(d["reviewed"], "selection.reviewed"),
                   _opt_str(d.get("reviewed_by"), "selection.reviewed_by"))

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "positive": [list(p) for p in self.positive],
                "negative": [list(p) for p in self.negative], "reviewed": self.reviewed,
                "reviewed_by": self.reviewed_by}

    def check_bounds(self, width: int, height: int) -> None:
        for x, y in self.positive + self.negative:
            if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
                raise ContractError("selection point lies outside the source frame")


def _pinhole_K(value: Any, name: str) -> tuple[float, ...]:
    K = _floats(value, 9, name)
    if K[0] <= 0 or K[4] <= 0 or K[1] != 0 or K[3] != 0 or K[6:] != (0.0, 0.0, 1.0):
        raise ContractError(f"{name} must be a zero-skew OpenCV pinhole matrix with positive focal lengths")
    return K


def _point(value: Any, name: str) -> tuple[float, float]:
    x, y = _floats(value, 2, name)
    return (x, y)


@dataclass(frozen=True)
class Onboarding:
    keyframe_ids: tuple[str, ...]
    selection: Selection | None

    @classmethod
    def from_dict(cls, value: Any) -> "Onboarding":
        d = _keys(value, ("keyframe_ids", "selection"), (), "onboarding")
        ids = tuple(_ident(v, "keyframe_id") for v in _list(d["keyframe_ids"], "keyframe_ids", max_len=MAX_KEYFRAMES))
        if not ids:
            raise ContractError("onboarding needs at least one keyframe")
        if len(set(ids)) != len(ids):
            raise ContractError("onboarding keyframes must be unique")
        sel = None if d["selection"] is None else Selection.from_dict(d["selection"])
        if sel is not None and sel.frame_id not in ids:
            raise ContractError("selection must be reviewed on an onboarding keyframe")
        return cls(ids, sel)

    def to_dict(self) -> dict:
        return {"keyframe_ids": list(self.keyframe_ids),
                "selection": None if self.selection is None else self.selection.to_dict()}


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    title: str
    source_kind: str
    clock_kind: str
    width: int
    height: int
    media: MediaRef
    frames: tuple[FrameSpec, ...]
    intrinsics: Intrinsics
    onboarding: Onboarding | None
    scoring_frame_ids: tuple[str, ...]
    evidence_role: str
    evaluator_inputs_read: bool = False
    excluded_source_fields: tuple[str, ...] = ()
    clock_source: str | None = None
    provenance: dict = field(default_factory=dict)
    schema: str = CLIP_SCHEMA
    coordinates: dict = field(default_factory=lambda: dict(COORDINATE_CONVENTION))

    OPTIONAL = ("excluded_source_fields", "clock_source", "provenance", "coordinates", "onboarding")

    @classmethod
    def from_dict(cls, value: Any) -> "ClipSpec":
        d = _keys(value, ("schema", "clip_id", "title", "source_kind", "clock_kind", "width", "height",
                          "media", "frames", "intrinsics", "scoring_frame_ids", "evidence_role",
                          "evaluator_inputs_read"), cls.OPTIONAL, "clip")
        if d["schema"] != CLIP_SCHEMA:
            raise ContractError(f"clip schema must be {CLIP_SCHEMA}")
        coords = d.get("coordinates", dict(COORDINATE_CONVENTION))
        if coords != COORDINATE_CONVENTION:
            raise ContractError("clip coordinate convention differs from the workbench convention")
        if d["evaluator_inputs_read"] is not False:
            raise ContractError("evaluator inputs may never be read for a workbench clip")
        frames = tuple(FrameSpec.from_dict(f) for f in _list(d["frames"], "frames", max_len=MAX_FRAMES))
        onboarding = d.get("onboarding")
        clip = cls(
            clip_id=_ident(d["clip_id"], "clip_id"),
            title=_str(d["title"], "title", max_len=240),
            source_kind=_enum(d["source_kind"], SOURCE_KINDS, "source_kind"),
            clock_kind=_enum(d["clock_kind"], CLOCK_KINDS, "clock_kind"),
            width=_int(d["width"], "width", minimum=1, maximum=16384),
            height=_int(d["height"], "height", minimum=1, maximum=16384),
            media=MediaRef.from_dict(d["media"]),
            frames=frames,
            intrinsics=Intrinsics.from_dict(d["intrinsics"]),
            onboarding=None if onboarding is None else Onboarding.from_dict(onboarding),
            scoring_frame_ids=tuple(_ident(v, "scoring_frame_id") for v in
                                    _list(d["scoring_frame_ids"], "scoring_frame_ids", max_len=MAX_FRAMES)),
            evidence_role=_str(d["evidence_role"], "evidence_role", max_len=128),
            evaluator_inputs_read=False,
            excluded_source_fields=tuple(_str(v, "excluded_source_field", max_len=128) for v in
                                         _list(d.get("excluded_source_fields", []), "excluded_source_fields", max_len=64)),
            clock_source=_opt_str(d.get("clock_source"), "clock_source", max_len=128),
            provenance=_json_value(d.get("provenance", {}), "provenance"),
        )
        clip.validate()
        return clip

    def validate(self) -> None:
        if not self.frames:
            raise ContractError("clip needs at least one frame")
        ids = [f.frame_id for f in self.frames]
        if len(set(ids)) != len(ids):
            raise ContractError("frame IDs must be unique")
        indices = [f.source_frame_index for f in self.frames]
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ContractError("source_frame_index must be strictly increasing in frame order")
        if self.clock_kind == "timestamp_seconds":
            stamps = [f.timestamp_s for f in self.frames]
            if any(s is None for s in stamps):
                raise ContractError("timestamp_seconds clips require timestamp_s on every frame")
            if any(b <= a for a, b in zip(stamps, stamps[1:])):
                raise ContractError("timestamp_s must be strictly increasing")
            if not self.clock_source:
                raise ContractError("timestamp_seconds clips must name their clock_source")
        else:
            if any(f.timestamp_s is not None for f in self.frames):
                raise ContractError("frame_index clips must keep timestamp_s null")
        media_times = [f.media_time_s for f in self.frames if f.media_time_s is not None]
        if any(b <= a for a, b in zip(media_times, media_times[1:])):
            raise ContractError("media_time_s must be strictly increasing when present")
        if self.source_kind == "image_sequence":
            if any(f.image_path is None or f.image_sha256 is None for f in self.frames):
                raise ContractError("image_sequence frames need image_path and image_sha256")
            paths = [f.image_path for f in self.frames]
            if len(set(paths)) != len(paths):
                raise ContractError("image_sequence frame sources must be unique")
        elif any(f.image_path is not None for f in self.frames):
            raise ContractError("video_file frames are addressed by source_frame_index, not image_path")
        position = {fid: i for i, fid in enumerate(ids)}
        if len(set(self.scoring_frame_ids)) != len(self.scoring_frame_ids):
            raise ContractError("scoring frames must be unique")
        if any(fid not in position for fid in self.scoring_frame_ids):
            raise ContractError("scoring frame is not a clip frame")
        scoring_pos = [position[f] for f in self.scoring_frame_ids]
        if scoring_pos != sorted(scoring_pos):
            raise ContractError("scoring frames must follow clip frame order")
        if self.onboarding is not None:
            check_onboarding(self, self.onboarding, self.scoring_frame_ids)

    def frame_position(self) -> dict[str, int]:
        return {f.frame_id: i for i, f in enumerate(self.frames)}

    def frame(self, frame_id: str) -> FrameSpec:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f
        raise KeyError(frame_id)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "clip_id": self.clip_id, "title": self.title,
            "source_kind": self.source_kind, "clock_kind": self.clock_kind,
            "clock_source": self.clock_source, "width": self.width, "height": self.height,
            "media": self.media.to_dict(), "frames": [f.to_dict() for f in self.frames],
            "intrinsics": self.intrinsics.to_dict(),
            "onboarding": None if self.onboarding is None else self.onboarding.to_dict(),
            "scoring_frame_ids": list(self.scoring_frame_ids), "evidence_role": self.evidence_role,
            "evaluator_inputs_read": False, "excluded_source_fields": list(self.excluded_source_fields),
            "provenance": self.provenance, "coordinates": dict(self.coordinates),
        }


def check_onboarding(clip: ClipSpec, onboarding: Onboarding, scheduled: Sequence[str]) -> dict:
    """Validate causal onboarding; return the prefix attestation."""
    position = clip.frame_position()
    for fid in onboarding.keyframe_ids:
        if fid not in position:
            raise ContractError(f"onboarding keyframe {fid} is not a clip frame")
    key_pos = [position[f] for f in onboarding.keyframe_ids]
    if key_pos != sorted(key_pos):
        raise ContractError("onboarding keyframes must follow clip frame order")
    last = key_pos[-1]
    for fid in scheduled:
        if fid not in position:
            raise ContractError(f"scheduled frame {fid} is not a clip frame")
        if position[fid] <= last:
            raise ContractError("scored frames must strictly follow the last onboarding frame")
    if onboarding.selection is not None:
        onboarding.selection.check_bounds(clip.width, clip.height)
    first = clip.frames[key_pos[0]]
    final = clip.frames[last]
    if clip.clock_kind == "timestamp_seconds":
        duration = float(final.timestamp_s) - float(first.timestamp_s)
        if duration > MAX_PREFIX_SECONDS + 1e-9:
            raise ContractError("onboarding prefix exceeds ten seconds")
        return {"prefix_bound_attested": True, "prefix_duration_s": duration,
                "keyframe_count": len(key_pos)}
    return {"prefix_bound_attested": False, "prefix_duration_s": None,
            "keyframe_count": len(key_pos),
            "note": "frame_index clock cannot attest the ten-second onboarding bound"}


# --------------------------------------------------------------------------
# surface package

@dataclass(frozen=True)
class FileRef:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(cls, value: Any, name: str = "file") -> "FileRef":
        d = _keys(value, ("path", "sha256", "bytes"), (), name)
        return cls(_relative_path(d["path"], f"{name}.path"), _sha(d["sha256"], f"{name}.sha256"),
                   _int(d["bytes"], f"{name}.bytes", minimum=1, maximum=MAX_PACKAGE_BYTES))

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class RegisteredCamera:
    frame_id: str
    image_sha256: str
    mask_sha256: str
    width: int
    height: int
    K: tuple[float, ...]
    pose: tuple[tuple[float, ...], ...]
    depth: FileRef | None

    @classmethod
    def from_dict(cls, value: Any) -> "RegisteredCamera":
        d = _keys(value, ("frame_id", "image_sha256", "mask_sha256", "width", "height", "K", "pose"),
                  ("depth",), "camera")
        depth = d.get("depth")
        return cls(_ident(d["frame_id"], "camera.frame_id"), _sha(d["image_sha256"], "camera.image_sha256"),
                   _sha(d["mask_sha256"], "camera.mask_sha256"), _int(d["width"], "camera.width", minimum=1),
                   _int(d["height"], "camera.height", minimum=1), _pinhole_K(d["K"], "camera.K"),
                   _pose(d["pose"], "camera.pose"),
                   None if depth is None else FileRef.from_dict(depth, "camera.depth"))

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "image_sha256": self.image_sha256, "mask_sha256": self.mask_sha256,
                "width": self.width, "height": self.height, "K": list(self.K),
                "pose": [list(r) for r in self.pose], "depth": None if self.depth is None else self.depth.to_dict()}


@dataclass(frozen=True)
class Marker:
    face_id: int
    barycentric: tuple[float, float, float]
    tangent: tuple[float, float, float]
    bitangent: tuple[float, float, float]
    normal: tuple[float, float, float]
    size_object_units: float
    artwork_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "Marker":
        d = _keys(value, ("face_id", "barycentric", "tangent", "bitangent", "normal",
                          "size_object_units", "artwork_sha256"), (), "marker")
        bary = _floats(d["barycentric"], 3, "marker.barycentric")
        if min(bary) < 0 or abs(sum(bary) - 1.0) > 1e-6:
            raise ContractError("marker barycentric coordinates must be >= 0 and sum to 1")
        t, b, n = (_floats(d[k], 3, f"marker.{k}") for k in ("tangent", "bitangent", "normal"))
        vecs = (t, b, n)
        for i in range(3):
            for j in range(3):
                dot = sum(vecs[i][k] * vecs[j][k] for k in range(3))
                if abs(dot - (1.0 if i == j else 0.0)) > 1e-5:
                    raise ContractError("marker tangent chart must be orthonormal")
        cross = (t[1] * b[2] - t[2] * b[1], t[2] * b[0] - t[0] * b[2], t[0] * b[1] - t[1] * b[0])
        if sum(c * m for c, m in zip(cross, n)) < 1 - 1e-5:
            raise ContractError("marker tangent chart must be right-handed (t x b = n)")
        return cls(_int(d["face_id"], "marker.face_id"), bary, t, b, n,
                   _float(d["size_object_units"], "marker.size_object_units", positive=True),
                   _sha(d["artwork_sha256"], "marker.artwork_sha256"))

    def to_dict(self) -> dict:
        return {"face_id": self.face_id, "barycentric": list(self.barycentric), "tangent": list(self.tangent),
                "bitangent": list(self.bitangent), "normal": list(self.normal),
                "size_object_units": self.size_object_units, "artwork_sha256": self.artwork_sha256}


ADMISSION_GATES = {
    "min_registered_views": 8,
    "min_sparse_points_3plus_views": 100,
    "min_median_triangulation_angle_deg": 2.0,
    "max_median_reprojection_px": 2.0,
    "min_face_support_views": 3,
}


@dataclass(frozen=True)
class SurfacePackage:
    package_id: str
    clip_id: str
    status: str  # available | unavailable
    unavailable_reason: str | None
    units: str
    metric_scale_available: bool
    normalization: dict | None
    vertices: FileRef | None
    faces: FileRef | None
    face_support_views: FileRef | None
    vertex_count: int
    face_count: int
    cameras: tuple[RegisteredCamera, ...]
    descriptor_points: FileRef | None
    descriptor_values: FileRef | None
    admission: dict
    marker: Marker | None
    artwork: dict | None
    frozen: bool
    frozen_after_frame_id: str | None
    prefix_frame_ids: tuple[str, ...]
    reconstruction: dict
    preview: dict
    evaluator_inputs_read: bool = False
    schema: str = SURFACE_SCHEMA
    coordinates: dict = field(default_factory=lambda: dict(COORDINATE_CONVENTION))

    REQUIRED = ("schema", "package_id", "clip_id", "status", "unavailable_reason", "units",
                "metric_scale_available", "normalization", "vertices", "faces", "face_support_views",
                "vertex_count", "face_count", "cameras", "descriptor_points", "descriptor_values",
                "admission", "marker", "artwork", "frozen", "frozen_after_frame_id", "prefix_frame_ids",
                "reconstruction", "preview", "evaluator_inputs_read")

    @classmethod
    def from_dict(cls, value: Any) -> "SurfacePackage":
        d = _keys(value, cls.REQUIRED, ("coordinates",), "surface_package")
        if d["schema"] != SURFACE_SCHEMA:
            raise ContractError(f"surface schema must be {SURFACE_SCHEMA}")
        if d.get("coordinates", COORDINATE_CONVENTION) != COORDINATE_CONVENTION:
            raise ContractError("surface coordinate convention differs")
        if d["evaluator_inputs_read"] is not False:
            raise ContractError("surface packages may never use evaluator inputs")
        status = _enum(d["status"], ("available", "unavailable"), "surface.status")
        units = _enum(d["units"], UNITS, "surface.units")
        metric = _bool(d["metric_scale_available"], "metric_scale_available")
        if units == "object_unit" and metric:
            raise ContractError("object_unit geometry cannot claim metric scale")
        if units == "metric" and not metric:
            raise ContractError("metric units require metric_scale_available")
        opt_ref = lambda k: None if d[k] is None else FileRef.from_dict(d[k], k)  # noqa: E731
        norm = d["normalization"]
        if norm is not None:
            n = _keys(norm, ("center", "scale", "method"), (), "normalization")
            norm = {"center": list(_floats(n["center"], 3, "normalization.center")),
                    "scale": _float(n["scale"], "normalization.scale", positive=True),
                    "method": _str(n["method"], "normalization.method")}
        pkg = cls(
            package_id=_ident(d["package_id"], "package_id"), clip_id=_ident(d["clip_id"], "clip_id"),
            status=status, unavailable_reason=_opt_str(d["unavailable_reason"], "unavailable_reason", max_len=2000),
            units=units, metric_scale_available=metric, normalization=norm,
            vertices=opt_ref("vertices"), faces=opt_ref("faces"), face_support_views=opt_ref("face_support_views"),
            vertex_count=_int(d["vertex_count"], "vertex_count"),
            face_count=_int(d["face_count"], "face_count", maximum=MAX_MESH_TRIANGLES),
            cameras=tuple(RegisteredCamera.from_dict(c) for c in _list(d["cameras"], "cameras", max_len=MAX_KEYFRAMES)),
            descriptor_points=opt_ref("descriptor_points"), descriptor_values=opt_ref("descriptor_values"),
            admission=_json_value(d["admission"], "admission"),
            marker=None if d["marker"] is None else Marker.from_dict(d["marker"]),
            artwork=None if d["artwork"] is None else _artwork(d["artwork"]),
            frozen=_bool(d["frozen"], "frozen"),
            frozen_after_frame_id=None if d["frozen_after_frame_id"] is None else _ident(d["frozen_after_frame_id"], "frozen_after_frame_id"),
            prefix_frame_ids=tuple(_ident(v, "prefix_frame_id") for v in _list(d["prefix_frame_ids"], "prefix_frame_ids", max_len=MAX_KEYFRAMES)),
            reconstruction=_json_value(d["reconstruction"], "reconstruction"),
            preview=_json_value(d["preview"], "preview"),
        )
        pkg.validate()
        return pkg

    def validate(self) -> None:
        if self.status == "available":
            if self.unavailable_reason is not None:
                raise ContractError("available package cannot carry an unavailable reason")
            for name in ("vertices", "faces", "face_support_views", "normalization"):
                if getattr(self, name) is None:
                    raise ContractError(f"available package requires {name}")
            if self.face_count < 1 or self.vertex_count < 3:
                raise ContractError("available package needs a nonempty mesh")
            if not self.frozen or self.frozen_after_frame_id is None:
                raise ContractError("available package must be frozen before scoring")
            if self.admission.get("passed") is not True:
                raise ContractError("available package must pass its frozen admission gates")
            if self.units == "object_unit" and self.normalization is None:
                raise ContractError("object_unit package requires its normalization record")
            total = sum(r.bytes for r in (self.vertices, self.faces, self.face_support_views,
                                          self.descriptor_points, self.descriptor_values) if r is not None)
            total += sum(c.depth.bytes for c in self.cameras if c.depth is not None)
            if total > MAX_PACKAGE_BYTES:
                raise ContractError("surface package exceeds 20 MB")
            if self.frozen_after_frame_id not in self.prefix_frame_ids:
                raise ContractError("package must freeze at an onboarding prefix frame")
        else:
            if not self.unavailable_reason:
                raise ContractError("unavailable package requires a reason")
            if self.vertices is not None or self.faces is not None or self.marker is not None:
                raise ContractError("unavailable package must not carry geometry or a marker")
        if self.marker is not None and self.marker.face_id >= self.face_count:
            raise ContractError("marker face_id is outside the mesh")
        if self.marker is not None and self.artwork is not None and self.marker.artwork_sha256 != self.artwork["sha256"]:
            raise ContractError("marker artwork hash differs from the package artwork")

    def to_dict(self) -> dict:
        ref = lambda r: None if r is None else r.to_dict()  # noqa: E731
        return {
            "schema": self.schema, "package_id": self.package_id, "clip_id": self.clip_id,
            "status": self.status, "unavailable_reason": self.unavailable_reason, "units": self.units,
            "metric_scale_available": self.metric_scale_available, "normalization": self.normalization,
            "vertices": ref(self.vertices), "faces": ref(self.faces), "face_support_views": ref(self.face_support_views),
            "vertex_count": self.vertex_count, "face_count": self.face_count,
            "cameras": [c.to_dict() for c in self.cameras],
            "descriptor_points": ref(self.descriptor_points), "descriptor_values": ref(self.descriptor_values),
            "admission": self.admission, "marker": None if self.marker is None else self.marker.to_dict(),
            "artwork": self.artwork, "frozen": self.frozen, "frozen_after_frame_id": self.frozen_after_frame_id,
            "prefix_frame_ids": list(self.prefix_frame_ids), "reconstruction": self.reconstruction,
            "preview": self.preview, "evaluator_inputs_read": False, "coordinates": dict(self.coordinates),
        }


def _artwork(value: Any) -> dict:
    d = _keys(value, ("kind", "edge_px", "sha256", "generation"), (), "artwork")
    return {"kind": _enum(d["kind"], ARTWORK_KINDS, "artwork.kind"),
            "edge_px": _int(d["edge_px"], "artwork.edge_px", minimum=8, maximum=MAX_ARTWORK_EDGE),
            "sha256": _sha(d["sha256"], "artwork.sha256"),
            "generation": _str(d["generation"], "artwork.generation", max_len=128)}


# --------------------------------------------------------------------------
# frame result

@dataclass(frozen=True)
class FrameResult:
    run_id: str
    clip_id: str
    frame_id: str
    source_frame_index: int
    timestamp_s: float | None
    media_time_s: float | None
    backend_role: str
    backend_identity: dict
    mask_state: str
    pose_state: str
    render_state: str
    failure_reason: str | None
    pose: tuple[tuple[float, ...], ...] | None
    pose_units: str | None
    image_sha256: str | None
    mask_frame_id: str | None
    mask_sha256: str | None
    metrics: dict
    evidence: dict = field(default_factory=dict)
    schema: str = FRAME_RESULT_SCHEMA

    REQUIRED = ("schema", "run_id", "clip_id", "frame_id", "source_frame_index", "timestamp_s",
                "media_time_s", "backend_role", "backend_identity", "mask_state", "pose_state",
                "render_state", "failure_reason", "pose", "pose_units", "image_sha256",
                "mask_frame_id", "mask_sha256", "metrics")

    @classmethod
    def from_dict(cls, value: Any) -> "FrameResult":
        d = _keys(value, cls.REQUIRED, ("evidence",), "frame_result")
        if d["schema"] != FRAME_RESULT_SCHEMA:
            raise ContractError(f"frame result schema must be {FRAME_RESULT_SCHEMA}")
        metrics = d["metrics"]
        if not isinstance(metrics, Mapping) or len(metrics) > 64:
            raise ContractError("metrics must be a bounded object")
        clean_metrics = {_str(k, "metric name", max_len=64): _opt_float(v, f"metric {k}") for k, v in metrics.items()}
        row = cls(
            run_id=_ident(d["run_id"], "run_id"), clip_id=_ident(d["clip_id"], "clip_id"),
            frame_id=_ident(d["frame_id"], "frame_id"),
            source_frame_index=_int(d["source_frame_index"], "source_frame_index"),
            timestamp_s=_opt_float(d["timestamp_s"], "timestamp_s", minimum=0),
            media_time_s=_opt_float(d["media_time_s"], "media_time_s", minimum=0),
            backend_role=_enum(d["backend_role"], BACKEND_ROLES, "backend_role"),
            backend_identity=_json_value(d["backend_identity"], "backend_identity"),
            mask_state=_enum(d["mask_state"], MASK_STATES, "mask_state"),
            pose_state=_enum(d["pose_state"], POSE_STATES, "pose_state"),
            render_state=_enum(d["render_state"], RENDER_STATES, "render_state"),
            failure_reason=_opt_str(d["failure_reason"], "failure_reason", max_len=2000),
            pose=None if d["pose"] is None else _pose(d["pose"], "pose"),
            pose_units=None if d["pose_units"] is None else _enum(d["pose_units"], UNITS, "pose_units"),
            image_sha256=_opt_sha(d["image_sha256"], "image_sha256"),
            mask_frame_id=None if d["mask_frame_id"] is None else _ident(d["mask_frame_id"], "mask_frame_id"),
            mask_sha256=_opt_sha(d["mask_sha256"], "mask_sha256"),
            metrics=clean_metrics,
            evidence=_json_value(d.get("evidence", {}), "evidence"),
        )
        row.validate()
        return row

    def validate(self) -> None:
        public = self.pose is not None
        if public:
            if self.pose_state != "tracking":
                raise ContractError("a public pose requires pose_state tracking")
            if self.mask_state != "observed" or self.mask_frame_id != self.frame_id or self.mask_sha256 is None:
                raise ContractError("a public pose requires a fresh observed mask for the same frame")
            if self.image_sha256 is None:
                raise ContractError("a public pose requires the processed image identity")
            if self.pose_units is None:
                raise ContractError("a public pose requires explicit units")
        else:
            if self.pose_state == "tracking":
                raise ContractError("tracking state requires its validated pose")
            if self.pose_units is not None:
                raise ContractError("pose_units must be null without a pose")
            if not self.failure_reason:
                raise ContractError("non-tracking rows must carry a failure reason")
        if self.render_state == "rendered" and not public:
            raise ContractError("rendering requires a validated tracking pose; stale overlays are forbidden")
        if self.mask_state == "observed" and (self.mask_frame_id != self.frame_id or self.mask_sha256 is None):
            raise ContractError("observed mask must belong to this exact frame")

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "run_id": self.run_id, "clip_id": self.clip_id, "frame_id": self.frame_id,
            "source_frame_index": self.source_frame_index, "timestamp_s": self.timestamp_s,
            "media_time_s": self.media_time_s, "backend_role": self.backend_role,
            "backend_identity": self.backend_identity, "mask_state": self.mask_state,
            "pose_state": self.pose_state, "render_state": self.render_state,
            "failure_reason": self.failure_reason,
            "pose": None if self.pose is None else [list(r) for r in self.pose],
            "pose_units": self.pose_units, "image_sha256": self.image_sha256,
            "mask_frame_id": self.mask_frame_id, "mask_sha256": self.mask_sha256,
            "metrics": dict(self.metrics), "evidence": self.evidence,
        }


def blocked_result(run_id: str, clip: ClipSpec, frame: FrameSpec, role: str, identity: dict,
                   reason: str, *, mask_state: str = "blocked", pose_state: str = "blocked") -> FrameResult:
    """A scheduled frame that could not be processed; never carries a pose."""
    return FrameResult(run_id, clip.clip_id, frame.frame_id, frame.source_frame_index, frame.timestamp_s,
                       frame.media_time_s, role, identity, mask_state, pose_state,
                       "unavailable" if pose_state == "blocked" else "suppressed", reason, None, None,
                       None, None, None, {})


# --------------------------------------------------------------------------
# run

@dataclass(frozen=True)
class RunSpec:
    run_id: str
    clip_id: str
    mode: str
    onboarding: Onboarding
    scheduled_frame_ids: tuple[str, ...]
    roles: tuple[str, ...]
    primary: dict
    surface_package_id: str | None
    artwork_kind: str
    config: dict
    schema: str = RUN_SCHEMA

    REQUIRED = ("schema", "run_id", "clip_id", "mode", "onboarding", "scheduled_frame_ids", "roles",
                "primary", "surface_package_id", "artwork_kind", "config")

    @classmethod
    def from_dict(cls, value: Any) -> "RunSpec":
        d = _keys(value, cls.REQUIRED, (), "run")
        if d["schema"] != RUN_SCHEMA:
            raise ContractError(f"run schema must be {RUN_SCHEMA}")
        roles = tuple(_enum(r, BACKEND_ROLES, "role") for r in _list(d["roles"], "roles", max_len=3))
        if len(set(roles)) != len(roles) or not roles:
            raise ContractError("roles must be unique and nonempty")
        if "frozen_baseline" in roles and "primary_neural" not in roles:
            raise ContractError("the frozen baseline is a comparator and cannot replace the primary model")
        primary = _keys(d["primary"], ("artifact_id", "export_sha256", "export_bytes"), (), "primary")
        primary = {"artifact_id": _ident(primary["artifact_id"], "primary.artifact_id"),
                   "export_sha256": _sha(primary["export_sha256"], "primary.export_sha256"),
                   "export_bytes": _int(primary["export_bytes"], "primary.export_bytes", minimum=1)}
        scheduled = tuple(_ident(v, "scheduled_frame_id") for v in
                          _list(d["scheduled_frame_ids"], "scheduled_frame_ids", max_len=MAX_FRAMES))
        if not scheduled or len(set(scheduled)) != len(scheduled):
            raise ContractError("scheduled frames must be unique and nonempty")
        sp = d["surface_package_id"]
        return cls(_ident(d["run_id"], "run_id"), _ident(d["clip_id"], "clip_id"),
                   _enum(d["mode"], RUN_MODES, "mode"), Onboarding.from_dict(d["onboarding"]), scheduled,
                   roles, primary, None if sp is None else _ident(sp, "surface_package_id"),
                   _enum(d["artwork_kind"], ARTWORK_KINDS, "artwork_kind"),
                   _json_value(d["config"], "config"))

    def validate_against(self, clip: ClipSpec) -> dict:
        if self.clip_id != clip.clip_id:
            raise ContractError("run clip identity differs")
        if self.onboarding.selection is None:
            raise ContractError("a run needs a reviewed positive selection on an onboarding frame")
        scoring = set(clip.scoring_frame_ids)
        if any(fid not in scoring for fid in self.scheduled_frame_ids):
            raise ContractError("scheduled frames must be declared scoring frames")
        position = clip.frame_position()
        pos = [position[f] for f in self.scheduled_frame_ids]
        if pos != sorted(pos):
            raise ContractError("scheduled frames must follow clip order")
        return check_onboarding(clip, self.onboarding, self.scheduled_frame_ids)

    def to_dict(self) -> dict:
        return {"schema": self.schema, "run_id": self.run_id, "clip_id": self.clip_id, "mode": self.mode,
                "onboarding": self.onboarding.to_dict(), "scheduled_frame_ids": list(self.scheduled_frame_ids),
                "roles": list(self.roles), "primary": dict(self.primary),
                "surface_package_id": self.surface_package_id, "artwork_kind": self.artwork_kind,
                "config": self.config}


def schedule_frames(clip: ClipSpec, mode: str, onboarding: Onboarding, *,
                    first_frame_id: str | None = None, count: int | None = None) -> tuple[str, ...]:
    """Choose scheduled scoring frames strictly after the last onboarding keyframe."""
    position = clip.frame_position()
    last = max(position[f] for f in onboarding.keyframe_ids)
    eligible = [f for f in clip.scoring_frame_ids if position[f] > last]
    if mode == "regression":
        chosen = eligible
    elif mode == "smoke":
        chosen = eligible[:SMOKE_FRAME_COUNT]
    elif mode == "focused":
        if first_frame_id is None or count is None or count < 1:
            raise ContractError("focused runs need --first-frame and --count")
        if first_frame_id not in eligible:
            raise ContractError("focused first frame must be an eligible scoring frame")
        start = eligible.index(first_frame_id)
        chosen = eligible[start:start + count]
    else:
        raise ContractError("unknown run mode")
    if not chosen:
        raise ContractError("no scoring frames remain after onboarding")
    return tuple(chosen)


# --------------------------------------------------------------------------
# report

def build_report(run: RunSpec, clip: ClipSpec, results: Sequence[FrameResult], *,
                 annotations_available: bool = False, stage_status: Mapping[str, Any] | None = None) -> dict:
    """Count every scheduled frame per role; never fabricate accuracy."""
    by_role: dict[str, dict[str, FrameResult]] = {r: {} for r in run.roles}
    for row in results:
        if row.run_id != run.run_id or row.clip_id != clip.clip_id:
            raise ContractError("result belongs to another run or clip")
        if row.backend_role not in by_role:
            raise ContractError("result role was not scheduled")
        if row.frame_id in by_role[row.backend_role]:
            raise ContractError("duplicate frame result")
        by_role[row.backend_role][row.frame_id] = row
    roles = {}
    for role, rows in by_role.items():
        missing = [f for f in run.scheduled_frame_ids if f not in rows]
        extra = [f for f in rows if f not in set(run.scheduled_frame_ids)]
        if extra:
            raise ContractError("result for a frame that was not scheduled")
        counts = {k: {s: 0 for s in states} for k, states in
                  (("mask_state", MASK_STATES), ("pose_state", POSE_STATES), ("render_state", RENDER_STATES))}
        reasons: dict[str, int] = {}
        for fid in run.scheduled_frame_ids:
            row = rows.get(fid)
            if row is None:
                continue
            counts["mask_state"][row.mask_state] += 1
            counts["pose_state"][row.pose_state] += 1
            counts["render_state"][row.render_state] += 1
            if row.failure_reason:
                key = row.failure_reason.split(":", 1)[0][:96]
                reasons[key] = reasons.get(key, 0) + 1
        n = len(run.scheduled_frame_ids)
        tracking = counts["pose_state"]["tracking"]
        roles[role] = {
            "scheduled_frames": n, "result_rows": len(rows), "missing_rows": missing,
            "counts": counts, "failure_reasons": dict(sorted(reasons.items())),
            "availability": tracking / n if n else None,
            "availability_target": 0.90,
        }
    return {
        "schema": REPORT_SCHEMA, "run_id": run.run_id, "clip_id": clip.clip_id, "mode": run.mode,
        "scheduled_frames": len(run.scheduled_frame_ids), "roles": roles,
        "primary": dict(run.primary),
        "accuracy": {
            "status": "available" if annotations_available else "unavailable_no_independent_annotations",
            "median_px_720": None, "p95_px_720": None,
            "targets": {"median_px_720": 5.0, "p95_px_720": 10.0, "availability": 0.90},
            "gates_verified": False,
        },
        "stages": dict(stage_status or {}),
        "claims": "No independent accuracy claim without human annotations; availability counts only.",
    }
