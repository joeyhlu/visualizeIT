"""Pure metadata bindings for the frozen R8 actual-source rows.

This module consumes already-parsed R1/R3 metadata mappings. It never opens a
packet or reads a forward/result field, and its receipt is not a capacity
screen or a verification of the source packet files.
"""

from collections.abc import Mapping
import hashlib
import json
import re


FROZEN_FRAMES = (10, 50, 100)
TEMPLATE_OFFSETS = (0, 180)
COUNTERPART_ARRAYS = (
    "observed_crop_mask",
    "crop_k",
    "crop_from_native",
    "native_k",
)
SOURCE_ARRAYS = (
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
_ARRAY_INFO_FIELDS = ("shape", "dtype", "sha256")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceBindingError(ValueError):
    """Frozen source or plan metadata does not satisfy the R8 binding contract."""


def _fail(message):
    raise SourceBindingError(message)


def _mapping(value, label):
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _sequence(value, label):
    if not isinstance(value, list):
        _fail(f"{label} must be a parsed JSON list")
    return value


def _field(record, name, label):
    _mapping(record, label)
    try:
        return record[name]
    except (KeyError, TypeError):
        _fail(f"{label} is missing required field {name!r}")


def _text(value, label, *, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value):
        _fail(f"{label} must be a string")
    return value


def _integer(value, label, *, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        _fail(f"{label} must be an integer")
    return value


def _sha256(value, label):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA256 digest")
    return value


def _array_info(value, label):
    """Return only the three producer-defined, JSON-safe array metadata fields."""
    _mapping(value, label)
    try:
        keys = set(value.keys())
    except (AttributeError, TypeError):
        _fail(f"{label} has invalid metadata fields")
    if keys != set(_ARRAY_INFO_FIELDS):
        _fail(f"{label} must contain exactly shape, dtype, and sha256")

    shape = _field(value, "shape", label)
    if not isinstance(shape, list) or not shape:
        _fail(f"{label}.shape must be a nonempty list")
    safe_shape = []
    for dimension in shape:
        safe_shape.append(_integer(dimension, f"{label}.shape item", minimum=1))
    dtype = _text(_field(value, "dtype", label), f"{label}.dtype")
    digest = _sha256(_field(value, "sha256", label), f"{label}.sha256")
    return {"shape": safe_shape, "dtype": dtype, "sha256": digest}


def _array_descriptor(arrays, name, label):
    _mapping(arrays, f"{label}.arrays")
    try:
        raw = arrays[name]
    except (KeyError, TypeError):
        _fail(f"{label} is missing permitted array metadata {name!r}")
    return _array_info(raw, f"{label}.{name}")


def _canonical_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_digest(metadata):
    return _canonical_digest({name: metadata[name] for name in COUNTERPART_ARRAYS})


def _canonical_plans():
    rows = []
    for frame_id in FROZEN_FRAMES:
        rows.append({
            "condition_id": f"syn-{frame_id}-q8-rgb-t0-rgb",
            "frame_id": frame_id,
            "kind": "synthetic_positive",
            "template_offset_deg": 0,
            "appearance": "rgb",
        })
        for self_condition in ("full", "clipped"):
            rows.append({
                "condition_id": f"zero-{frame_id}-{self_condition}",
                "frame_id": frame_id,
                "kind": "self_control",
                "template_offset_deg": 0,
                "appearance": "rgb",
                "self_condition": self_condition,
            })
        rows.append({
            "condition_id": f"syn-{frame_id}-q8-rgb-t180-rgb",
            "frame_id": frame_id,
            "kind": "synthetic_negative",
            "template_offset_deg": 180,
            "appearance": "rgb",
        })
    return rows


def _validate_expected_plans(expected_plans):
    plans = _sequence(expected_plans, "expected_plans")
    frozen = _canonical_plans()
    if len(plans) != len(frozen):
        _fail("expected_plans must contain the exact twelve frozen rows")
    for index, (actual, expected) in enumerate(zip(plans, frozen)):
        label = f"expected_plans[{index}]"
        _mapping(actual, label)
        if set(actual.keys()) != set(expected.keys()):
            _fail(f"{label} has unexpected or missing fields")
        for name, expected_value in expected.items():
            value = _field(actual, name, label)
            if type(value) is not type(expected_value) or value != expected_value:
                _fail(f"{label}.{name} differs from the frozen plan")
    return frozen


def _context_index(capture_metadata):
    raw_contexts = _sequence(
        _field(capture_metadata, "contexts", "capture_metadata"),
        "capture_metadata.contexts",
    )
    by_id = {}
    identities = set()
    for index, entry in enumerate(raw_contexts):
        label = f"capture_metadata.contexts[{index}]"
        _mapping(entry, label)
        context_id = _text(_field(entry, "context_id", label), f"{label}.context_id")
        role = _text(_field(entry, "role", label), f"{label}.role")
        frame_id = _integer(_field(entry, "frame_id", label), f"{label}.frame_id", minimum=1)
        geometry_group = _text(
            _field(entry, "geometry_group", label), f"{label}.geometry_group"
        )
        if geometry_group != f"frame-{frame_id}":
            _fail(f"{label} has an unexpected geometry_group")
        if context_id in by_id:
            _fail("capture_metadata contains duplicate context IDs")

        if role == "real_frame":
            expected_id = f"real-frame-{frame_id:04d}"
            identity = (role, frame_id)
        elif role == "template":
            offset = _integer(
                _field(entry, "template_offset_deg", label),
                f"{label}.template_offset_deg",
                minimum=0,
            )
            if offset not in TEMPLATE_OFFSETS:
                _fail(f"{label} has an unexpected template offset")
            expected_id = f"template-{frame_id:04d}-{offset:03d}"
            identity = (role, frame_id, offset)
        elif role == "synthetic_query":
            offset = _integer(
                _field(entry, "query_offset_deg", label),
                f"{label}.query_offset_deg",
                minimum=0,
            )
            if offset not in (8, 188):
                _fail(f"{label} has an unexpected query offset")
            expected_id = f"synthetic-query-{frame_id:04d}-{offset:03d}"
            identity = (role, frame_id, offset)
        else:
            _fail(f"{label} has an unexpected context role")

        if context_id != expected_id:
            _fail(f"{label} context ID does not match its role/frame/offset")
        if identity in identities:
            _fail("capture_metadata contains duplicate role/frame/offset identities")
        identities.add(identity)
        by_id[context_id] = entry
    return by_id


def _row_index(rows, label):
    values = _sequence(rows, label)
    indexed = {}
    for index, entry in enumerate(values):
        row_label = f"{label}[{index}]"
        _mapping(entry, row_label)
        condition_id = _text(
            _field(entry, "condition_id", row_label),
            f"{row_label}.condition_id",
        )
        if condition_id in indexed:
            _fail(f"{label} contains duplicate condition IDs")
        indexed[condition_id] = entry
    return indexed


def _source_template_entry(entry, context_id, frame_id, offset):
    label = f"template context {context_id}"
    arrays = _field(entry, "arrays", label)
    _mapping(arrays, f"{label}.arrays")
    try:
        if set(arrays.keys()) != set(SOURCE_ARRAYS):
            _fail(f"{label} does not have the exact frozen source-array metadata set")
    except (AttributeError, TypeError):
        _fail(f"{label}.arrays has invalid keys")

    source_arrays = {
        name: _array_descriptor(arrays, name, label) for name in SOURCE_ARRAYS
    }
    geometry_hashes = _field(entry, "geometry_hashes", label)
    _mapping(geometry_hashes, f"{label}.geometry_hashes")
    try:
        if set(geometry_hashes.keys()) != {"depth_mm", "mask"}:
            _fail(f"{label} must have only depth_mm and mask geometry hashes")
    except (AttributeError, TypeError):
        _fail(f"{label}.geometry_hashes has invalid keys")
    depth_hash = _sha256(
        _field(geometry_hashes, "depth_mm", label), f"{label}.geometry_hashes.depth_mm"
    )
    mask_hash = _sha256(
        _field(geometry_hashes, "mask", label), f"{label}.geometry_hashes.mask"
    )
    if (depth_hash != source_arrays["template_depth_mm"]["sha256"] or
            mask_hash != source_arrays["template_mask"]["sha256"]):
        _fail(f"{label} geometry hashes do not bind its depth and mask metadata")

    path = _text(_field(entry, "path", label), f"{label}.path")
    expected_path = f"packets/contexts/{context_id}.npz"
    if path != expected_path:
        _fail(f"{label} has an unexpected source packet path")
    packet_bytes = _integer(_field(entry, "bytes", label), f"{label}.bytes", minimum=1)
    packet_sha256 = _sha256(_field(entry, "sha256", label), f"{label}.sha256")

    source_entry = {
        "context_id": context_id,
        "role": "template",
        "frame_id": frame_id,
        "template_offset_deg": offset,
        "path": path,
        "bytes": packet_bytes,
        "sha256": packet_sha256,
        "arrays": source_arrays,
        "geometry_hashes": {"depth_mm": depth_hash, "mask": mask_hash},
    }
    return source_entry


def _context(entry, expected_id, role, frame_id, offset=None):
    label = f"context {expected_id}"
    if _field(entry, "context_id", label) != expected_id:
        _fail(f"{label} identity mismatch")
    if _field(entry, "role", label) != role:
        _fail(f"{label} role mismatch")
    if _integer(_field(entry, "frame_id", label), f"{label}.frame_id", minimum=1) != frame_id:
        _fail(f"{label} frame mismatch")
    if _field(entry, "geometry_group", label) != f"frame-{frame_id}":
        _fail(f"{label} geometry_group mismatch")
    if role == "template":
        if _integer(
            _field(entry, "template_offset_deg", label),
            f"{label}.template_offset_deg",
            minimum=0,
        ) != offset:
            _fail(f"{label} template offset mismatch")
    elif role == "synthetic_query":
        if _integer(
            _field(entry, "query_offset_deg", label),
            f"{label}.query_offset_deg",
            minimum=0,
        ) != offset:
            _fail(f"{label} query offset mismatch")
    arrays = _field(entry, "arrays", label)
    metadata = {
        name: _array_descriptor(arrays, name, label) for name in COUNTERPART_ARRAYS
    }
    return metadata


def _exact_refs(value, expected, label):
    _mapping(value, label)
    try:
        if set(value.keys()) != set(expected):
            _fail(f"{label} does not have the exact frozen context-ref keys")
    except (AttributeError, TypeError):
        _fail(f"{label} has invalid context refs")
    result = {}
    for key, expected_value in expected.items():
        actual = _field(value, key, label)
        if actual != expected_value:
            _fail(f"{label}.{key} differs from the frozen reference")
        result[key] = expected_value
    return result


def _forward_rows(capture_metadata, zero_metadata):
    r1_conditions = _row_index(
        _field(capture_metadata, "conditions", "capture_metadata"),
        "capture_metadata.conditions",
    )
    r1_forwards = _row_index(
        _field(capture_metadata, "forwards", "capture_metadata"),
        "capture_metadata.forwards",
    )
    r3_conditions = _row_index(
        _field(zero_metadata, "conditions", "zero_metadata"),
        "zero_metadata.conditions",
    )
    r3_forwards = _row_index(
        _field(zero_metadata, "forwards", "zero_metadata"),
        "zero_metadata.forwards",
    )
    return r1_conditions, r1_forwards, r3_conditions, r3_forwards


def bind_source_plans(capture_metadata, zero_metadata, expected_plans):
    """Bind the exact twelve R5 plans to safe R1/R3 source metadata.

    Only parsed metadata mappings are accepted. This function does not read
    packet files, import or call model code, inspect result fields, or evaluate
    any capacity gate.
    """
    _mapping(capture_metadata, "capture_metadata")
    _mapping(zero_metadata, "zero_metadata")
    plans = _validate_expected_plans(expected_plans)
    contexts = _context_index(capture_metadata)
    r1_conditions, r1_forwards, r3_conditions, r3_forwards = _forward_rows(
        capture_metadata, zero_metadata
    )

    template_entries = {}
    for frame_id in FROZEN_FRAMES:
        for offset in TEMPLATE_OFFSETS:
            context_id = f"template-{frame_id:04d}-{offset:03d}"
            entry = contexts.get(context_id)
            if entry is None:
                _fail(f"required source template context {context_id} is missing")
            template_metadata = _context(entry, context_id, "template", frame_id, offset)
            source_entry = _source_template_entry(entry, context_id, frame_id, offset)
            source_rgb_sha256 = source_entry["arrays"]["template_rgb"]["sha256"]
            template_entries[(frame_id, offset)] = {
                "context_id": context_id,
                "entry": entry,
                "metadata": template_metadata,
                "source_entry": source_entry,
                "source_namespace": {
                    "context_id": context_id,
                    "source_rgb_sha256": source_rgb_sha256,
                },
            }

    counterpart_metadata = {}
    template_bindings = []
    for frame_id in FROZEN_FRAMES:
        frame_context_id = f"real-frame-{frame_id:04d}"
        frame_entry = contexts.get(frame_context_id)
        if frame_entry is None:
            _fail(f"required observed-frame context {frame_context_id} is missing")
        frame_metadata = _context(frame_entry, frame_context_id, "real_frame", frame_id)
        frame_group = _field(frame_entry, "geometry_group", f"context {frame_context_id}")

        query_context_id = f"synthetic-query-{frame_id:04d}-008"
        query_entry = contexts.get(query_context_id)
        if query_entry is None:
            _fail(f"required synthetic-query context {query_context_id} is missing")
        query_metadata = _context(query_entry, query_context_id, "synthetic_query", frame_id, 8)
        query_group = _field(query_entry, "geometry_group", f"context {query_context_id}")
        if frame_group != query_group:
            _fail(f"frame {frame_id} counterparts have different geometry groups")

        counterpart_metadata[(frame_id, "observed_frame")] = frame_metadata
        counterpart_metadata[(frame_id, "synthetic_query")] = query_metadata
        templates = [template_entries[(frame_id, offset)] for offset in TEMPLATE_OFFSETS]
        first_metadata = templates[0]["metadata"]
        first_group = _field(templates[0]["entry"], "geometry_group", "template")
        for template in templates:
            if _field(template["entry"], "geometry_group", "template") != first_group:
                _fail(f"frame {frame_id} template views have different geometry groups")
            for name in COUNTERPART_ARRAYS:
                reference = template["metadata"][name]
                if reference != frame_metadata[name]:
                    _fail(f"frame {frame_id} {name} differs from its observed-frame metadata")
                if reference != query_metadata[name]:
                    _fail(f"frame {frame_id} {name} differs from its synthetic-query metadata")
                if reference != first_metadata[name]:
                    _fail(f"frame {frame_id} {name} differs between template views")

        for template in templates:
            metadata = template["metadata"]
            template_bindings.append({
                "context_id": template["context_id"],
                "frame_id": frame_id,
                "template_offset_deg": template["source_entry"]["template_offset_deg"],
                "geometry_group": first_group,
                "source_namespace": dict(template["source_namespace"]),
                "array_metadata": {
                    name: metadata[name] for name in COUNTERPART_ARRAYS
                },
                "source_entry": template["source_entry"],
            })

    # Bind every row only after all six immutable source contexts and their
    # allowed counterpart metadata have passed the shared-camera checks.
    plan_bindings = []
    for plan in plans:
        condition_id = plan["condition_id"]
        frame_id = plan["frame_id"]
        offset = plan["template_offset_deg"]
        template = template_entries[(frame_id, offset)]
        template_context_id = template["context_id"]
        frame_context_id = f"real-frame-{frame_id:04d}"
        query_context_id = f"synthetic-query-{frame_id:04d}-008"
        template_metadata = template["metadata"]
        frame_metadata = counterpart_metadata[(frame_id, "observed_frame")]

        if plan["kind"] in ("synthetic_positive", "synthetic_negative"):
            condition = r1_conditions.get(condition_id)
            forward = r1_forwards.get(condition_id)
            if condition is None or forward is None:
                _fail(f"R1 condition or forward {condition_id} is missing")
            condition_label = f"R1 condition {condition_id}"
            if (_field(condition, "condition_id", condition_label) != condition_id or
                    _integer(_field(condition, "frame_id", condition_label), f"{condition_label}.frame_id") != frame_id or
                    _field(condition, "kind", condition_label) != "synthetic" or
                    _integer(_field(condition, "query_offset_deg", condition_label), f"{condition_label}.query_offset_deg") != 8 or
                    _field(condition, "query_appearance", condition_label) != "rgb" or
                    _integer(_field(condition, "template_offset_deg", condition_label), f"{condition_label}.template_offset_deg") != offset or
                    _field(condition, "template_appearance", condition_label) != "rgb" or
                    _field(condition, "state", condition_label) != "captured"):
                _fail(f"R1 condition metadata differs from the frozen synthetic plan {condition_id}")

            frame_position = FROZEN_FRAMES.index(frame_id)
            expected_index = frame_position * 16 + (offset // 180) * 2
            expected_seed = (frame_id * 17 + (offset // 180) * 2) & 0x7FFFFFFF
            condition_index = _integer(
                _field(condition, "forward_index", condition_label),
                f"{condition_label}.forward_index",
                minimum=0,
            )
            condition_seed = _integer(
                _field(condition, "rng_seed", condition_label),
                f"{condition_label}.rng_seed",
                minimum=0,
            )
            if condition_index != expected_index or condition_seed != expected_seed:
                _fail(f"R1 condition forward index or RNG seed differs for {condition_id}")
            refs = {
                "query": query_context_id,
                "template": template_context_id,
                "observed_frame": frame_context_id,
            }
            copied_refs = _exact_refs(
                _field(condition, "context_refs", condition_label),
                refs,
                f"{condition_label}.context_refs",
            )
            forward_label = f"R1 forward {condition_id}"
            if (_field(forward, "condition_id", forward_label) != condition_id or
                    _integer(_field(forward, "forward_index", forward_label), f"{forward_label}.forward_index", minimum=0) != condition_index or
                    _integer(_field(forward, "rng_seed", forward_label), f"{forward_label}.rng_seed", minimum=0) != condition_seed):
                _fail(f"R1 forward identity, index, or RNG seed differs for {condition_id}")
            _exact_refs(
                _field(forward, "context_refs", forward_label),
                refs,
                f"{forward_label}.context_refs",
            )
            query_metadata = counterpart_metadata[(frame_id, "synthetic_query")]
            counterpart_hashes = {
                "observed_frame": _metadata_digest(frame_metadata),
                "synthetic_query": _metadata_digest(query_metadata),
            }
            counterpart_masks = {
                "observed_frame": frame_metadata["observed_crop_mask"],
                "synthetic_query": query_metadata["observed_crop_mask"],
            }
            plan_bindings.append({
                "condition_id": condition_id,
                "kind": plan["kind"],
                "frame_id": frame_id,
                "template_offset_deg": offset,
                "template_context_id": template_context_id,
                "observed_frame_context_id": frame_context_id,
                "synthetic_query_context_id": query_context_id,
                "geometry_group": template_bindings[
                    FROZEN_FRAMES.index(frame_id) * 2 + (offset // 180)
                ]["geometry_group"],
                "source_namespace": dict(template["source_namespace"]),
                "forward_index": expected_index,
                "rng_seed": expected_seed,
                "self_condition": None,
                "context_refs": copied_refs,
                "counterpart_mask_metadata": counterpart_masks,
                "counterpart_array_metadata_sha256": counterpart_hashes,
                "plan_binding_verified": True,
                "counterpart_binding_verified": True,
            })
        else:
            self_condition = plan["self_condition"]
            condition = r3_conditions.get(condition_id)
            forward = r3_forwards.get(condition_id)
            if condition is None or forward is None:
                _fail(f"R3 condition or forward {condition_id} is missing")
            condition_label = f"R3 condition {condition_id}"
            if (_field(condition, "condition_id", condition_label) != condition_id or
                    _integer(_field(condition, "frame_id", condition_label), f"{condition_label}.frame_id") != frame_id or
                    _integer(_field(condition, "template_offset_deg", condition_label), f"{condition_label}.template_offset_deg") != 0 or
                    _field(condition, "condition", condition_label) != self_condition or
                    _field(condition, "appearance", condition_label) != "rgb" or
                    _field(condition, "state", condition_label) != "captured"):
                _fail(f"R3 condition metadata differs from the frozen self-control plan {condition_id}")

            expected_index = FROZEN_FRAMES.index(frame_id) * 2 + (1 if self_condition == "clipped" else 0)
            condition_index = _integer(
                _field(condition, "forward_index", condition_label),
                f"{condition_label}.forward_index",
                minimum=0,
            )
            if condition_index != expected_index:
                _fail(f"R3 condition forward index differs for {condition_id}")
            competitor_context_id = f"template-{frame_id:04d}-180"
            refs = {
                "source_template": template_context_id,
                "competitor_template": competitor_context_id,
                "observed_frame": frame_context_id,
            }
            copied_refs = _exact_refs(
                _field(condition, "context_refs", condition_label),
                refs,
                f"{condition_label}.context_refs",
            )
            forward_label = f"R3 forward {condition_id}"
            if (_field(forward, "condition_id", forward_label) != condition_id or
                    _integer(_field(forward, "forward_index", forward_label), f"{forward_label}.forward_index", minimum=0) != condition_index):
                _fail(f"R3 forward identity or index differs for {condition_id}")
            _exact_refs(
                _field(forward, "context_refs", forward_label),
                refs,
                f"{forward_label}.context_refs",
            )
            counterpart_hashes = {
                "observed_frame": _metadata_digest(frame_metadata),
            }
            counterpart_masks = {
                "observed_frame": frame_metadata["observed_crop_mask"],
            }
            plan_bindings.append({
                "condition_id": condition_id,
                "kind": "self_control",
                "frame_id": frame_id,
                "template_offset_deg": offset,
                "template_context_id": template_context_id,
                "observed_frame_context_id": frame_context_id,
                "synthetic_query_context_id": None,
                "geometry_group": template_bindings[
                    FROZEN_FRAMES.index(frame_id) * 2
                ]["geometry_group"],
                "source_namespace": dict(template["source_namespace"]),
                "forward_index": expected_index,
                "rng_seed": None,
                "self_condition": self_condition,
                "context_refs": copied_refs,
                "counterpart_mask_metadata": counterpart_masks,
                "counterpart_array_metadata_sha256": counterpart_hashes,
                "plan_binding_verified": True,
                "counterpart_binding_verified": True,
            })

    if len(template_bindings) != 6 or len(plan_bindings) != 12:
        raise AssertionError("R8 source binding accounting changed")
    namespaces = [
        (row["source_namespace"]["context_id"], row["source_namespace"]["source_rgb_sha256"])
        for row in template_bindings
    ]
    if len(set(namespaces)) != 6:
        _fail("0/180 source image namespaces are not distinct")

    return {
        "schema_version": "r8-source-metadata-bindings-v1",
        "metadata_bindings_verified": True,
        "counterpart_context_bindings_verified": True,
        "source_packet_file_pins_verified": False,
        "source_packet_members_verified": False,
        "source_packet_arrays_decoded": False,
        "mesh_pin_verified": False,
        "capacity_screen_complete": False,
        "matching_started": False,
        "fitting_started": False,
        "template_bindings": template_bindings,
        "plan_bindings": plan_bindings,
        "counts": {"template_bindings": 6, "plan_bindings": 12},
    }

