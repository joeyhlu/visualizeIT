"""Pure physical-time policy and bounded input-manifest timing parser.

This module deliberately imports only the Python standard library. It performs
no file, image, model, clock, renderer, or network work; runner code may use it
to reject malformed timing metadata before loading inference dependencies.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from fractions import Fraction


LEGACY = "legacy_nominal_60_v1"
PHYSICAL = "capture_seconds_v1"
_MODES = frozenset((LEGACY, PHYSICAL))
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_TIMELINE_ROWS = 10_000
_SOURCE_UNITS = {
    "seconds": 1,
    "milliseconds": 1_000,
    "microseconds": 1_000_000,
    "nanoseconds": 1_000_000_000,
}


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _nonnegative_id(value, name="frame_id"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_timestamp(value, name="timestamp_s"):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


@dataclass(frozen=True)
class TimePolicy:
    """The selected clock mode and physical bounds for one tracker stream."""

    mode: str = LEGACY
    max_angular_rate_deg_s: float = 720.0
    max_translation_rate_m_s: float = 1.2
    private_pose_memory_s: float = 0.5

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in _MODES:
            raise ValueError(f"Unknown clock mode: {self.mode!r}")
        object.__setattr__(self, "max_angular_rate_deg_s",
                           _finite_positive(self.max_angular_rate_deg_s, "max_angular_rate_deg_s"))
        object.__setattr__(self, "max_translation_rate_m_s",
                           _finite_positive(self.max_translation_rate_m_s, "max_translation_rate_m_s"))
        object.__setattr__(self, "private_pose_memory_s",
                           _finite_positive(self.private_pose_memory_s, "private_pose_memory_s"))


LEGACY_POLICY = TimePolicy()


@dataclass(frozen=True)
class FrameKey:
    """Chronology identity for a frame or a known unavailable source row."""

    frame_id: int
    timestamp_s: float | None = None
    clock_mode: str = LEGACY

    def __post_init__(self):
        _nonnegative_id(self.frame_id)
        if not isinstance(self.clock_mode, str) or self.clock_mode not in _MODES:
            raise ValueError(f"Unknown clock mode: {self.clock_mode!r}")
        if self.clock_mode == LEGACY:
            if self.timestamp_s is not None:
                raise ValueError("Legacy frame keys cannot carry timestamps")
        else:
            if self.timestamp_s is None:
                raise ValueError("Physical frame keys require timestamp_s")
            object.__setattr__(self, "timestamp_s", _finite_timestamp(self.timestamp_s))


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number is not permitted: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite JSON number")
    # JSON integers have arbitrary precision.  Do not coerce source ticks before
    # applying their declared time base: a huge nanosecond count can still map to
    # a finite number of seconds.
    if isinstance(value, int):
        return value
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite JSON number") from exc
    if not finite:
        raise ValueError(f"{name} must be a finite JSON number")
    return value


def _parse_legacy(manifest, manifest_sha256):
    if "clock" in manifest or "timeline" in manifest:
        raise ValueError("Legacy manifests cannot mix in clock or timeline fields")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        raise ValueError("Untimed legacy input must use schema_version 1")
    if "setup_frame_id" not in manifest or "frame_ids" not in manifest:
        raise ValueError("Legacy timing requires setup_frame_id and frame_ids")
    setup_id = manifest["setup_frame_id"]
    if setup_id is not None:
        _nonnegative_id(setup_id, "setup_frame_id")
    frame_ids = manifest["frame_ids"]
    if not isinstance(frame_ids, list) or not frame_ids:
        raise ValueError("Legacy frame_ids must be a nonempty list")
    if len(frame_ids) > _MAX_TIMELINE_ROWS:
        raise ValueError("Timing table exceeds the row limit")
    for frame_id in frame_ids:
        _nonnegative_id(frame_id)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Legacy frame_ids must be unique")
    if setup_id is not None and setup_id in frame_ids:
        raise ValueError("Legacy setup frame cannot also be scored")
    # Existing authenticated v1 inputs use source IDs directly as tracker IDs.
    keys = tuple(FrameKey(frame_id, None, LEGACY) for frame_id in frame_ids)
    source_ids = {key.frame_id: key.frame_id for key in keys}
    setup_key = None if setup_id is None else FrameKey(setup_id, None, LEGACY)
    if setup_key is not None:
        source_ids[setup_key.frame_id] = setup_key.frame_id
    return {
        "clock_mode": LEGACY,
        "time_policy": LEGACY_POLICY,
        "manifest_sha256": manifest_sha256,
        "timestamp_table_sha256": None,
        "setup_key": setup_key,
        "scored_keys": keys,
        "source_frame_ids": source_ids,
    }


def _source_seconds(source_timestamp, units):
    if units not in _SOURCE_UNITS:
        raise ValueError("Unsupported source_units")
    value = _json_number(source_timestamp, "source_timestamp")
    scale = _SOURCE_UNITS[units]
    # Preserve integer ticks through unit conversion; avoid float(value)/scale,
    # which discards low-order clock ticks for large nanosecond timestamps.
    try:
        if isinstance(value, int):
            seconds = float(Fraction(value, scale))
        else:
            seconds = float(value) / scale
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Converted timestamp must be finite") from exc
    if not math.isfinite(seconds):
        raise ValueError("Converted timestamp must be finite")
    return seconds


def _canonical_table_sha256(rows):
    canonical_rows = [
        {
            "frame_id": row["frame_id"],
            "source_frame_id": row["source_frame_id"],
            "timestamp_s": row["timestamp_s"],
            "role": row["role"],
        }
        for row in rows
    ]
    encoded = json.dumps(canonical_rows, sort_keys=True, allow_nan=False,
                         separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _parse_physical(manifest, manifest_sha256):
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 2:
        raise ValueError("Physical time requires schema_version 2")
    clock = manifest.get("clock")
    if not isinstance(clock, dict):
        raise ValueError("Physical input requires a clock object")
    required_clock = {"mode", "units", "source_units", "timestamp_source", "nominal_fps"}
    if not required_clock.issubset(clock):
        raise ValueError("Physical clock is missing required timing fields")
    if clock["mode"] != PHYSICAL or clock["units"] != "seconds":
        raise ValueError("Physical clock mode/units mismatch")
    if not isinstance(clock["source_units"], str) or clock["source_units"] not in _SOURCE_UNITS:
        raise ValueError("Unsupported source_units")
    if not isinstance(clock["timestamp_source"], str) or not clock["timestamp_source"].strip():
        raise ValueError("timestamp_source must identify declared capture metadata/PTS")
    _finite_positive(clock["nominal_fps"], "nominal_fps")
    if "time_base" in clock:
        time_base = clock["time_base"]
        if not isinstance(time_base, dict) or set(time_base) != {"num", "den"}:
            raise ValueError("time_base must contain exactly positive integer num and den")
        numerator = time_base["num"]
        denominator = time_base["den"]
        if (isinstance(numerator, bool) or not isinstance(numerator, int) or numerator <= 0 or
                isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0):
            raise ValueError("time_base must contain exactly positive integer num and den")
        if Fraction(numerator, denominator) != Fraction(1, _SOURCE_UNITS[clock["source_units"]]):
            raise ValueError("time_base conflicts with declared source_units")

    timeline = manifest.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("Physical input requires a nonempty timeline")
    if len(timeline) > _MAX_TIMELINE_ROWS:
        raise ValueError("Timing table exceeds the row limit")
    rows = []
    last_time = None
    seen_source_ids = set()
    previous_source_id = None
    required_gaps = []
    for ordinal, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            raise ValueError("Each timeline entry must be an object")
        required = {"frame_id", "source_frame_id", "source_timestamp", "timestamp_s", "role"}
        if not required.issubset(entry):
            raise ValueError("Timeline entry is missing required timing fields")
        frame_id = _nonnegative_id(entry["frame_id"])
        source_frame_id = _nonnegative_id(entry["source_frame_id"], "source_frame_id")
        if frame_id != ordinal:
            raise ValueError("Physical frame_id values must be exactly 0..N-1")
        if source_frame_id in seen_source_ids:
            raise ValueError("Original source frame IDs must be unique")
        if previous_source_id is not None:
            if source_frame_id <= previous_source_id:
                raise ValueError("Original source frame IDs must be strictly increasing")
            missing = source_frame_id - previous_source_id - 1
            if missing:
                required_gaps.append({
                    "previous_source_frame_id": previous_source_id,
                    "current_source_frame_id": source_frame_id,
                    "missing_capture_count": missing,
                })
        seen_source_ids.add(source_frame_id)
        previous_source_id = source_frame_id
        role = entry["role"]
        if role not in ("setup", "scored"):
            raise ValueError("Timeline role must be setup or scored")
        seconds = _source_seconds(entry["source_timestamp"], clock["source_units"])
        canonical_seconds = _finite_timestamp(entry["timestamp_s"], "timestamp_s")
        if canonical_seconds != seconds:
            raise ValueError("timestamp_s does not exactly match source timestamp conversion")
        if last_time is not None and seconds <= last_time:
            raise ValueError("Physical timestamps must be strictly increasing")
        last_time = seconds
        rows.append({
            "frame_id": frame_id,
            "source_frame_id": source_frame_id,
            "source_timestamp": entry["source_timestamp"],
            "timestamp_s": canonical_seconds,
            "role": role,
        })

    if "setup_frame_id" not in manifest:
        raise ValueError("Physical input requires an explicit setup_frame_id (ordinal or null)")
    setup_id = manifest["setup_frame_id"]
    if setup_id is not None:
        _nonnegative_id(setup_id, "setup_frame_id")
    if setup_id is None:
        if rows[0]["role"] != "scored" or any(row["role"] != "scored" for row in rows):
            raise ValueError("Without separate setup, all timeline rows must be scored")
    else:
        if setup_id != 0 or rows[0]["role"] != "setup" or sum(row["role"] == "setup" for row in rows) != 1:
            raise ValueError("Physical setup must be the unique ordinal-0 setup row")
        if any(row["role"] != "scored" for row in rows[1:]):
            raise ValueError("Only ordinal 0 may be setup")

    frame_ids = manifest.get("frame_ids")
    if not isinstance(frame_ids, list) or not frame_ids:
        raise ValueError("Physical frame_ids must be a nonempty list")
    for frame_id in frame_ids:
        _nonnegative_id(frame_id)
    scored_ordinals = [row["frame_id"] for row in rows if row["role"] == "scored"]
    if frame_ids != scored_ordinals:
        raise ValueError("frame_ids must list all scored timeline ordinals in order")

    source_gaps = manifest.get("source_gaps", [])
    if not isinstance(source_gaps, list):
        raise ValueError("source_gaps must be a list when present")
    for gap in source_gaps:
        if not isinstance(gap, dict) or set(gap) != {
                "previous_source_frame_id", "current_source_frame_id", "missing_capture_count"}:
            raise ValueError("Each source gap must declare its previous/current IDs and missing count")
        _nonnegative_id(gap["previous_source_frame_id"], "previous_source_frame_id")
        _nonnegative_id(gap["current_source_frame_id"], "current_source_frame_id")
        _nonnegative_id(gap["missing_capture_count"], "missing_capture_count")
    if source_gaps != required_gaps:
        if required_gaps:
            raise ValueError("source_gaps must exactly describe every nonconsecutive source-ID pair")
        raise ValueError("source_gaps must be omitted or empty for a contiguous source-ID table")

    if "timeline_span_seconds" in clock:
        span = clock["timeline_span_seconds"]
        if isinstance(span, bool) or not isinstance(span, Real):
            raise ValueError("timeline_span_seconds must be finite and nonnegative")
        try:
            span = float(span)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("timeline_span_seconds must be finite and nonnegative") from exc
        if not math.isfinite(span) or span < 0.0:
            raise ValueError("timeline_span_seconds must be finite and nonnegative")
        actual_span = rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]
        if abs(span - actual_span) > 1e-6:
            raise ValueError("timeline_span_seconds disagrees with the normalized timeline")

    table_sha256 = _canonical_table_sha256(rows)
    scored_keys = tuple(FrameKey(row["frame_id"], row["timestamp_s"], PHYSICAL)
                        for row in rows if row["role"] == "scored")
    setup_row = rows[0] if setup_id is not None else None
    source_ids = {row["frame_id"]: row["source_frame_id"] for row in rows}
    return {
        "clock_mode": PHYSICAL,
        "time_policy": TimePolicy(mode=PHYSICAL),
        "manifest_sha256": manifest_sha256,
        "timestamp_table_sha256": table_sha256,
        "setup_key": None if setup_row is None else FrameKey(
            setup_row["frame_id"], setup_row["timestamp_s"], PHYSICAL),
        "scored_keys": scored_keys,
        "source_frame_ids": source_ids,
    }


def parse_timing_manifest(raw_manifest_bytes):
    """Validate timing fields from bounded raw input-manifest JSON bytes.

    This deliberately does not authenticate an input against historical pins or
    inspect resource paths/hashes; the production runner owns those checks.
    """
    if not isinstance(raw_manifest_bytes, bytes):
        raise ValueError("raw_manifest_bytes must be bytes")
    if len(raw_manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("Input manifest exceeds the 2 MiB timing limit")
    try:
        text = raw_manifest_bytes.decode("utf-8", errors="strict")
        manifest = json.loads(text, object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Input manifest must be valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Input manifest must be a JSON object")

    def reject_nonfinite_tree(value):
        # json.loads parses a syntactically valid exponent such as 1e400 as inf,
        # even though parse_constant only catches the nonstandard Infinity token.
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite JSON number is not permitted")
        if isinstance(value, dict):
            for child in value.values():
                reject_nonfinite_tree(child)
        elif isinstance(value, list):
            for child in value:
                reject_nonfinite_tree(child)

    reject_nonfinite_tree(manifest)
    manifest_sha256 = hashlib.sha256(raw_manifest_bytes).hexdigest().upper()
    if "clock" not in manifest and "timeline" not in manifest:
        return _parse_legacy(manifest, manifest_sha256)
    if "clock" not in manifest or "timeline" not in manifest:
        raise ValueError("Clock and timeline must be declared together")
    return _parse_physical(manifest, manifest_sha256)
