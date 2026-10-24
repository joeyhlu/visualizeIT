"""Frozen, label-isolated candidate caches for pose-ranker experiments.

This module only prepares and audits data.  It never runs a tracker, downloads a
dataset, or makes inference decisions.  Candidate features and reference labels
are written to separate JSONL files so the tracker/trainer boundary stays
reviewable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


SPLIT_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test", "dev")
MAX_INVENTORY_ROWS = 250_000
MAX_FRAMES = 100_000
MAX_CANDIDATES_PER_FRAME = 256
MAX_CANDIDATES_TOTAL = 2_000_000
MAX_FEATURE_COUNT = 512
MAX_CACHE_BYTES = 512 * 1024 * 1024
MAX_SURFACE_POINTS = 4096
MAX_JSONL_BYTES = 512 * 1024 * 1024

# Exact identities already inspected in this project.  Names are namespaced
# where the source supplies a dataset ID; short aliases are retained to fail
# closed for the historical HOT3D bundles.
DEV_ONLY_PHYSICAL_OBJECT_IDS = frozenset({
    "keyboard", "mug", "ranch", "mug_white", "bottle_ranch",
    "hot3d:keyboard:28", "hot3d:28:keyboard",
    "hot3d:mug_white:9", "hot3d:9:mug_white",
    "hot3d:bottle_ranch:15", "hot3d:15:bottle_ranch",
    "bop-ycbv:000002", "bop-ycbv:000004", "bop-ycbv:000006",
    "bop-ycbv:000009", "bop-ycbv:000018",
    "ycbv:000002", "ycbv:000004", "ycbv:000006", "ycbv:000009", "ycbv:000018",
    *(f"hb:{value:06d}" for value in range(1, 11)),
    *(f"bop-hb:{value:06d}" for value in range(1, 11)),
    "003_cracker_box", "005_tomato_soup_can", "007_tuna_fish_can",
    "010_potted_meat_can", "040_large_marker",
})

_LABEL_KEY_PARTS = ("label", "ground_truth", "groundtruth", "annotation",
                    "evaluation_pose", "reference_pose", "quality_target")
def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_load(path):
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_JSONL_BYTES:
            raise ValueError("JSON input exceeds the bounded file size")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON input: {path}") from exc
    _check_finite_tree(value)
    return value, raw


def _check_finite_tree(value, depth=0):
    if depth > 32:
        raise ValueError("JSON nesting exceeds the bounded depth")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON number is not permitted")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _check_finite_tree(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_finite_tree(child, depth + 1)


def _read_jsonl(path, *, max_rows=MAX_FRAMES):
    path = Path(path)
    if path.stat().st_size > MAX_JSONL_BYTES:
        raise ValueError("JSONL input exceeds the bounded file size")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at line {line_number}")
            try:
                row = json.loads(line, object_pairs_hook=_unique_pairs,
                                 parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            _check_finite_tree(row)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
            if len(rows) > max_rows:
                raise ValueError("JSONL frame count exceeds the configured bound")
    return rows


def _read_records(source, *, list_key="records"):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".jsonl":
            return _read_jsonl(path), _file_sha256(path)
        doc, raw = _json_load(path)
        if isinstance(doc, list):
            rows = doc
        elif isinstance(doc, dict) and isinstance(doc.get(list_key), list):
            rows = doc[list_key]
        elif isinstance(doc, dict) and isinstance(doc.get("frames"), list):
            rows = doc["frames"]
        else:
            raise ValueError(f"Expected a JSON list or object containing {list_key!r}")
        if len(rows) > MAX_INVENTORY_ROWS:
            raise ValueError("JSON record count exceeds the configured row bound")
        return rows, _sha256(raw)
    if isinstance(source, dict):
        rows = source.get(list_key, source.get("frames"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected object field {list_key!r} or 'frames'")
        if len(rows) > MAX_INVENTORY_ROWS or len(_json_bytes(rows)) > MAX_JSONL_BYTES:
            raise ValueError("JSON records exceed the configured row/byte bound")
        return rows, _sha256(_json_bytes(source))
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        rows = list(source)
        if len(rows) > MAX_INVENTORY_ROWS or len(_json_bytes(rows)) > MAX_JSONL_BYTES:
            raise ValueError("JSON records exceed the configured row/byte bound")
        return rows, _sha256(_json_bytes(rows))
    raise TypeError("Expected a file path, JSON object, or sequence of records")


def _positive_finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validate_label_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("A frozen label_policy is required")
    result = {
        "max_rotation_error_deg": _positive_finite(
            policy.get("max_rotation_error_deg"), "max_rotation_error_deg"),
        "max_translation_error_m": _positive_finite(
            policy.get("max_translation_error_m"), "max_translation_error_m"),
        "max_surface_error_m": None,
        "max_projection_error_px": None,
    }
    for name in ("max_surface_error_m", "max_projection_error_px"):
        if policy.get(name) is not None:
            result[name] = _positive_finite(policy[name], name)
    return result


def _string(value, name, *, maximum=256):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > maximum or value.lower() in {"unknown", "none", "null", "n/a"}:
        raise ValueError(f"{name} must be a known, non-empty identity")
    return value


def _optional_mesh_digest(row):
    value = row.get("mesh_sha256", row.get("asset_sha256"))
    if value is None:
        return None
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value.lower()) or
            set(value.lower()) == {"0"}):
        raise ValueError("mesh_sha256 must be a non-placeholder SHA-256 digest")
    return value.lower()


def _frame_id(value, name="frame_id"):
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.strip() and len(value) <= 128:
        if value.lower() in {"unknown", "none", "null"}:
            raise ValueError(f"{name} must be a known frame identity")
        return value
    raise ValueError(f"{name} must be a nonnegative integer or stable string")


def _normalized_background(row, known_background_ids):
    kind = row.get("background_kind")
    background = _string(row.get("background_id"), "background_id")
    if kind == "observed":
        if background not in known_background_ids:
            raise ValueError(f"Unknown observed background {background!r}; declare a complete scene inventory in background_catalog")
    elif kind in ("clean", "procedural"):
        prefix = kind + ":"
        if not background.lower().startswith(prefix):
            background = prefix + background
    else:
        raise ValueError("background_kind must explicitly be observed, clean, or procedural")
    return background, kind


def _identity(row, *, known_background_ids=None):
    context = row.get("context", {})
    if not isinstance(context, dict):
        context = {}
    def pick(name, *alternates):
        for key in (name,) + alternates:
            value = row.get(key)
            if value is None:
                value = context.get(key)
            if value is not None:
                return value
        return None
    physical = pick("physical_object_id", "object_id")
    if physical is None:
        physical = pick("object_id")
    physical = _string(physical, "physical_object_id")
    scene = _string(pick("scene_id"), "scene_id")
    sequence = _string(pick("sequence_id"), "sequence_id")
    background, background_kind = _normalized_background({
        "background_id": pick("background_id"),
        "background_kind": pick("background_kind"),
    }, known_background_ids or set())
    provenance = pick("provenance_kind")
    if provenance not in ("real", "synthetic", "fixture"):
        raise ValueError("provenance_kind must explicitly be real, synthetic, or fixture")
    frame = _frame_id(row.get("frame_id"))
    source_frame = _frame_id(row.get("source_frame_id", frame), "source_frame_id")
    return {
        "physical_object_id": physical,
        "object_id": physical,
        "scene_id": scene,
        "sequence_id": sequence,
        "background_id": background,
        "background_kind": background_kind,
        "provenance_kind": provenance,
        "frame_id": frame,
        "source_frame_id": source_frame,
    }


def _known_backgrounds(inventory):
    return set(_background_inventory(inventory))


def _background_inventory(inventory):
    """Validate full observed-scene identity inventories; target-only lists fail closed."""
    catalog = inventory.get("background_catalog", []) if isinstance(inventory, dict) else []
    if not isinstance(catalog, list):
        raise ValueError("background_catalog must list complete observed-scene identities")
    result = {}
    for row in catalog:
        if not isinstance(row, dict) or row.get("inventory_complete") is not True:
            raise ValueError("Observed backgrounds require inventory_complete=true and all physical_object_ids")
        identity = _string(row.get("background_id"), "background_catalog identity")
        scene = _string(row.get("scene_id"), "background scene_id")
        objects = row.get("physical_object_ids")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"Observed background {identity!r} lacks its full physical_object_ids list")
        object_ids = sorted({_string(value, "background physical_object_id") for value in objects})
        if len(object_ids) != len(objects):
            raise ValueError(f"Observed background {identity!r} has duplicate physical object identities")
        canonical = {"background_id": identity, "scene_id": scene,
                     "inventory_complete": True, "physical_object_ids": object_ids}
        if identity in result and result[identity] != canonical:
            raise ValueError(f"Conflicting full inventories for observed background {identity!r}")
        result[identity] = canonical
    return result


def _aliases(row):
    values = row.get("aliases", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("aliases must be a list of strings")
    return {_string(value, "physical-object alias").lower() for value in values}


def _is_dev_only(identity, aliases=()):
    values = {identity.lower(), *(alias.lower() for alias in aliases)}
    for value in values:
        if value in DEV_ONLY_PHYSICAL_OBJECT_IDS:
            return True
        # Accept common namespace spellings for the five inspected BOP pilot IDs.
        normalized = value.lower().replace("bop:ycbv:", "bop-ycbv:")
        if normalized.startswith(("bop-ycbv:", "ycbv:", "hb:", "bop-hb:")):
            suffix = normalized.split(":", 1)[1]
            suffix = suffix.removeprefix("object-").removeprefix("obj_").lstrip("0") or "0"
            namespace = normalized.split(":", 1)[0]
            if namespace in {"bop-ycbv", "ycbv"} and suffix in {"2", "4", "6", "9", "18"}:
                return True
            if namespace in {"hb", "bop-hb"} and suffix.isdecimal() and 1 <= int(suffix) <= 10:
                return True
    return False


def _split_digest(manifest):
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _sha256(_json_bytes(body))


def freeze_split_manifest(inventory, assignments=None, *, label_policy=None):
    """Freeze explicit split assignments and identity groups before evaluation.

    `inventory` is an object with `records` and, for observed scenes,
    `background_catalog`.  Each record is one frame and must name all identities.
    A sequence assignment may be supplied separately or as each row's `split`.
    """
    if isinstance(inventory, dict):
        rows = inventory.get("records", inventory.get("frames"))
        policy = label_policy or inventory.get("label_policy")
    else:
        rows = inventory
        policy = label_policy
    if not isinstance(rows, list) or not rows or len(rows) > MAX_INVENTORY_ROWS:
        raise ValueError("Inventory must contain 1..250000 frame records")
    policy = _validate_label_policy(policy)
    assignments = {} if assignments is None else dict(assignments)
    background_catalog = _background_inventory(inventory) if isinstance(inventory, dict) else {}
    known_background_ids = set(background_catalog)
    groups = {}
    frame_keys = set()
    ownership = {}
    identity_splits = {}
    sequence_physical_owner = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Every inventory record must be an object")
        ident = _identity(raw, known_background_ids=known_background_ids)
        sequence = ident["sequence_id"]
        explicit = raw.get("split")
        assigned = assignments.get(sequence)
        if explicit is not None and assigned is not None and explicit != assigned:
            raise ValueError(f"Conflicting split assignments for sequence {sequence!r}")
        split = explicit if explicit is not None else assigned
        if split not in SPLITS:
            raise ValueError(f"Sequence {sequence!r} must have an explicit split in {SPLITS}")
        aliases = _aliases(raw)
        physical_owner = sequence_physical_owner.setdefault(sequence, ident["physical_object_id"])
        if physical_owner != ident["physical_object_id"]:
            raise ValueError(f"Sequence {sequence!r} names multiple physical objects")
        mesh_digest = _optional_mesh_digest(raw)
        if ident["provenance_kind"] == "synthetic" and mesh_digest is None:
            raise ValueError("Synthetic split groups must freeze the exact metric mesh_sha256 before rendering")
        if _is_dev_only(ident["physical_object_id"], aliases) and split != "dev":
            raise ValueError(f"Previously inspected physical identity {ident['physical_object_id']!r} is dev-only")
        if ident["provenance_kind"] == "fixture" and split != "dev":
            raise ValueError("Fixture-provenance sequences are dev-only and cannot enter training or evaluation splits")
        if ident["background_kind"] == "observed":
            observed = background_catalog.get(ident["background_id"])
            if (observed is None or observed["scene_id"] != ident["scene_id"] or
                    ident["physical_object_id"] not in observed["physical_object_ids"]):
                raise ValueError("Observed background must bind the complete physical-object inventory for its scene")
        group_key = (ident["physical_object_id"], sequence)
        group = groups.get(group_key)
        invariant = (ident["scene_id"], ident["background_id"], ident["background_kind"],
                     ident["provenance_kind"], split, mesh_digest)
        if group is None:
            group = groups[group_key] = {
                **{key: ident[key] for key in ("physical_object_id", "scene_id", "sequence_id",
                                                "background_id", "background_kind", "provenance_kind")},
                "split": split, "_invariant": invariant, "_frames": [], "aliases": sorted(aliases),
            }
            if mesh_digest is not None:
                group["mesh_sha256"] = mesh_digest
        elif group["_invariant"] != invariant:
            raise ValueError(f"Sequence {sequence!r} changes identity or split within its frames")
        elif aliases:
            group["aliases"] = sorted(set(group["aliases"]) | aliases)
        frame_key = (ident["physical_object_id"], sequence, ident["frame_id"])
        if frame_key in frame_keys or any(row["source_frame_id"] == ident["source_frame_id"]
                                          for row in group["_frames"]):
            raise ValueError(f"Duplicate frame/source identity in sequence {sequence!r}")
        frame_keys.add(frame_key)
        group["_frames"].append({"frame_id": ident["frame_id"],
                                 "source_frame_id": ident["source_frame_id"]})
        for dimension, value in (("physical_object_id", ident["physical_object_id"]),
                                 ("scene_id", ident["scene_id"]),
                                 ("sequence_id", sequence)):
            old = ownership.setdefault((dimension, value), split)
            if old != split:
                raise ValueError(f"{dimension} {value!r} occurs in conflicting splits")
        if mesh_digest is not None:
            old = ownership.setdefault(("mesh_sha256", mesh_digest), split)
            if old != split:
                raise ValueError(f"mesh_sha256 {mesh_digest!r} occurs in conflicting splits")
        if ident["background_kind"] == "observed":
            old = ownership.setdefault(("background_id", ident["background_id"]), split)
            if old != split:
                raise ValueError(f"observed background_id {ident['background_id']!r} occurs in conflicting splits")
            # Does the training set contain indirect information about a test
            # frame? Full scene inventories make co-occurring physical objects
            # atomic too, even when only one target is tracked per sequence.
            for scene_object in background_catalog[ident["background_id"]]["physical_object_ids"]:
                if _is_dev_only(scene_object) and split != "dev":
                    raise ValueError(f"Previously inspected co-occurring physical identity {scene_object!r} is dev-only")
                old = ownership.setdefault(("physical_object_id", scene_object), split)
                if old != split:
                    raise ValueError(f"physical_object_id {scene_object!r} co-occurs across conflicting scene splits")
        for identity_alias in {ident["physical_object_id"].lower(), *aliases}:
            old = identity_splits.setdefault(identity_alias, split)
            if old != split:
                raise ValueError(f"Physical-object identity/alias {identity_alias!r} occurs in conflicting splits")

    manifest_groups = []
    for group in groups.values():
        frames = sorted(group.pop("_frames"), key=lambda row: _sort_frame(row["frame_id"]))
        group.pop("_invariant", None)
        group["frames"] = frames
        group["frame_ids"] = [row["source_frame_id"] for row in frames]
        manifest_groups.append(group)
    manifest_groups.sort(key=lambda row: (row["split"], row["physical_object_id"], row["sequence_id"]))
    if sum(len(group["frames"]) for group in manifest_groups) > MAX_FRAMES:
        raise ValueError("Frozen inventory exceeds the frame bound")
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "kind": "quality-ranker-split-v1",
        "split_names": list(SPLITS),
        "label_policy": policy,
        "atomic_identity_dimensions": ["physical_object_id", "mesh_sha256", "scene_id", "background_id", "sequence_id"],
        "unknown_background_policy": "reject unless explicitly clean or procedural",
        "dev_only_identity_policy": "historical keyboard/mug/ranch and inspected YCBV pilot identities",
        "background_catalog": sorted(background_catalog.values(), key=lambda row: row["background_id"]),
        "groups": manifest_groups,
    }
    manifest["manifest_sha256"] = _split_digest(manifest)
    return manifest


def _sort_frame(value):
    return (0, value) if type(value) is int else (1, str(value))


def validate_split_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SPLIT_SCHEMA_VERSION or manifest.get("kind") != "quality-ranker-split-v1":
        raise ValueError("Unsupported frozen split manifest")
    if manifest.get("manifest_sha256") != _split_digest(manifest):
        raise ValueError("Frozen split manifest digest mismatch")
    if not isinstance(manifest.get("groups"), list) or not manifest["groups"]:
        raise ValueError("Frozen split manifest has no groups")
    _validate_label_policy(manifest.get("label_policy"))
    seen = set()
    seen_groups = set()
    owner = {}
    identity_splits = {}
    sequence_physical_owner = {}
    for group in manifest["groups"]:
        if not isinstance(group, dict) or group.get("split") not in SPLITS:
            raise ValueError("Invalid split group")
        for name in ("physical_object_id", "scene_id", "sequence_id", "background_id", "background_kind", "provenance_kind"):
            _string(group.get(name), name)
        if group["provenance_kind"] not in ("real", "synthetic", "fixture"):
            raise ValueError("Invalid provenance kind in frozen split")
        if group["provenance_kind"] == "fixture" and group["split"] != "dev":
            raise ValueError("Fixture-provenance sequence escaped the dev-only split")
        mesh_digest = _optional_mesh_digest(group)
        if group["provenance_kind"] == "synthetic" and mesh_digest is None:
            raise ValueError("Synthetic split groups must bind the exact metric mesh_sha256")
        group_key = (group["physical_object_id"], group["sequence_id"])
        if group_key in seen_groups:
            raise ValueError("Duplicate frozen object/sequence group")
        seen_groups.add(group_key)
        physical_owner = sequence_physical_owner.setdefault(group["sequence_id"], group["physical_object_id"])
        if physical_owner != group["physical_object_id"]:
            raise ValueError(f"Sequence {group['sequence_id']!r} names multiple physical objects")
        aliases = _aliases(group)
        if _is_dev_only(group["physical_object_id"], aliases) and group["split"] != "dev":
            raise ValueError("Previously inspected identity escaped the dev-only split")
        for identity_alias in {group["physical_object_id"].lower(), *aliases}:
            prior = identity_splits.setdefault(identity_alias, group["split"])
            if prior != group["split"]:
                raise ValueError(f"Physical-object identity/alias {identity_alias!r} crosses splits")
        if group["background_kind"] not in ("observed", "clean", "procedural"):
            raise ValueError("Invalid background kind in frozen split")
        if _is_dev_only(group["physical_object_id"], group.get("aliases", [])) and group["split"] != "dev":
            raise ValueError("Previously inspected identity escaped the dev-only split")
        for dimension in ("physical_object_id", "scene_id", "sequence_id"):
            key = (dimension, group[dimension])
            prior = owner.setdefault(key, group["split"])
            if prior != group["split"]:
                raise ValueError(f"Split leakage across {dimension}")
        if mesh_digest is not None:
            key = ("mesh_sha256", mesh_digest)
            prior = owner.setdefault(key, group["split"])
            if prior != group["split"]:
                raise ValueError(f"Split leakage across metric mesh digest {mesh_digest}")
    background_catalog = _background_inventory(manifest)
    for group in manifest["groups"]:
        if group["background_kind"] == "observed":
            key = ("background_id", group["background_id"])
            prior = owner.setdefault(key, group["split"])
            if prior != group["split"]:
                raise ValueError(f"Split leakage across observed background {group['background_id']!r}")
            observed = background_catalog.get(group["background_id"])
            if (observed is None or observed["scene_id"] != group["scene_id"] or
                    group["physical_object_id"] not in observed["physical_object_ids"]):
                raise ValueError("Frozen observed background lacks the matching complete scene inventory")
            for scene_object in observed["physical_object_ids"]:
                if _is_dev_only(scene_object) and group["split"] != "dev":
                    raise ValueError(f"Previously inspected co-occurring identity {scene_object!r} escaped the dev-only split")
                key = ("physical_object_id", scene_object)
                prior = owner.setdefault(key, group["split"])
                if prior != group["split"]:
                    raise ValueError(f"Split leakage across co-occurring scene object {scene_object!r}")
        frames = group.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Split sequence has no frozen frames")
        frame_ids = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Frozen frame mapping must be an object")
            tracker_id = _frame_id(frame.get("frame_id"))
            source_id = _frame_id(frame.get("source_frame_id"), "source_frame_id")
            key = (group["physical_object_id"], group["sequence_id"], tracker_id)
            if key in seen:
                raise ValueError("Duplicate frozen frame identity")
            seen.add(key)
            frame_ids.append(source_id)
        if group.get("frame_ids") != frame_ids:
            raise ValueError("Frozen source frame IDs disagree with frame mappings")
    return manifest


def read_split_manifest(path):
    manifest, _ = _json_load(path)
    return validate_split_manifest(manifest)


def _label_key(row):
    return (_string(row.get("object_id"), "label object_id"),
            _string(row.get("sequence_id"), "label sequence_id"),
            _frame_id(row.get("frame_id"), "label frame_id"),
            _string(row.get("candidate_id"), "label candidate_id", maximum=128))


def _validate_pose(value, name="pose"):
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 4x4 pose") from exc
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} must be a finite homogeneous 4x4 pose")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or abs(np.linalg.det(rotation)-1) > 1e-3:
        raise ValueError(f"{name} rotation must be a proper SO(3) matrix")
    return pose


def rotation_error_degrees(predicted_pose, reference_pose):
    predicted = _validate_pose(predicted_pose, "predicted_pose")
    reference = _validate_pose(reference_pose, "reference_pose")
    relative = predicted[:3, :3] @ reference[:3, :3].T
    cosine = float(np.clip((np.trace(relative)-1.0)*0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def translation_error_m(predicted_pose, reference_pose):
    predicted = _validate_pose(predicted_pose, "predicted_pose")
    reference = _validate_pose(reference_pose, "reference_pose")
    return float(np.linalg.norm(predicted[:3, 3] - reference[:3, 3]))


def _project(points, pose, intrinsics):
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Projection object points must be a non-empty finite Nx3 array")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Projection intrinsics must be a finite 3x3 matrix")
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera_points @ matrix.T
    valid = np.isfinite(homogeneous).all(axis=1) & (camera_points[:, 2] > 0) & (homogeneous[:, 2] > 0)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return pixels, valid


def projected_landmark_error_px(predicted_pose, intrinsics, landmarks):
    """Return median/P95 reprojection error against independently reviewed pixels."""
    if not isinstance(landmarks, list) or not landmarks:
        return None
    points, observed = [], []
    for row in landmarks:
        if not isinstance(row, dict) or row.get("visible", True) is not True:
            continue
        point = np.asarray(row.get("object_point_m"), dtype=np.float64)
        pixel = np.asarray(row.get("pixel"), dtype=np.float64)
        if point.shape != (3,) or pixel.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(pixel).all():
            raise ValueError("Reviewed landmark rows require finite object_point_m and pixel")
        points.append(point)
        observed.append(pixel)
    if not points:
        return None
    pose = _validate_pose(predicted_pose)
    projected, valid = _project(points, pose, intrinsics)
    if not valid.any():
        return None
    errors = np.linalg.norm(projected[valid] - np.asarray(observed)[valid], axis=1)
    return {"median_px": float(np.median(errors)), "p95_px": float(np.percentile(errors, 95)),
            "visible_landmarks": int(len(errors)), "missing_behind_camera": int(len(points)-len(errors))}


def _surface_chamfer_p95(predicted_pose, reference_pose, object_points_m):
    """Symmetric nearest-neighbour mesh-point error (ADD-S style), in metres."""
    points = np.asarray(object_points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 or len(points) > MAX_SURFACE_POINTS or not np.isfinite(points).all():
        raise ValueError(f"Surface points must be a finite Nx3 array with N <= {MAX_SURFACE_POINTS}")
    predicted = _validate_pose(predicted_pose)
    reference = _validate_pose(reference_pose)
    a = points @ predicted[:3, :3].T + predicted[:3, 3]
    b = points @ reference[:3, :3].T + reference[:3, 3]

    def directed(source, target):
        values = []
        for start in range(0, len(source), 64):
            block = source[start:start+64]
            distances = np.sum((block[:, None, :] - target[None, :, :])**2, axis=2)
            values.extend(np.sqrt(np.min(distances, axis=1)).tolist())
        return np.asarray(values, dtype=np.float64)

    errors = np.concatenate((directed(a, b), directed(b, a)))
    return {"mean_m": float(np.mean(errors)), "p95_m": float(np.percentile(errors, 95)),
            "surface_points": int(len(points)), "metric": "symmetric_nearest_neighbor"}


def _reference_index(reference_rows):
    index = {}
    for row in reference_rows:
        if not isinstance(row, dict):
            raise ValueError("Reference labels must be objects")
        obj = _string(row.get("object_id", row.get("physical_object_id")), "label object_id")
        seq = _string(row.get("sequence_id"), "label sequence_id")
        frame_id = _frame_id(row.get("source_frame_id", row.get("frame_id")), "label frame_id")
        key = (obj, seq, frame_id)
        if key in index:
            raise ValueError(f"Duplicate reference label row for {key}")
        if type(row.get("observationally_ambiguous")) is not bool:
            raise ValueError("Every reference row needs an explicit observationally_ambiguous boolean")
        index[key] = row
    return index


def _make_label(candidate, frame, reference, policy, *, surface_points_by_object=None):
    ambiguous = reference["observationally_ambiguous"]
    row = {
        "object_id": frame["object_id"], "sequence_id": frame["sequence_id"],
        "frame_id": frame["frame_id"], "source_frame_id": frame["source_frame_id"],
        "candidate_id": candidate["candidate_id"], "split": frame["split"],
        "provenance_kind": frame["provenance_kind"],
        "label_kind": reference.get("label_kind", "unspecified_reference"),
        "observationally_ambiguous": ambiguous,
        "orientation_supervision": not ambiguous,
        "rotation_error_deg": None, "translation_error_m": None,
        "surface_error": None, "projection_error": None,
        "quality_target": None, "label_missing_reason": None,
    }
    pose = candidate.get("pose")
    if pose is None:
        row["label_missing_reason"] = "candidate_pose_unavailable"
        return row
    predicted = _validate_pose(pose, "candidate pose")
    reference_pose = _validate_pose(reference.get("pose"), "reference pose")
    if not ambiguous:
        row["rotation_error_deg"] = rotation_error_degrees(predicted, reference_pose)
    row["translation_error_m"] = translation_error_m(predicted, reference_pose)
    points = (surface_points_by_object or {}).get(frame["object_id"])
    if points is not None:
        row["surface_error"] = _surface_chamfer_p95(predicted, reference_pose, points)
    if reference.get("landmarks") is not None:
        provenance = reference.get("landmark_provenance")
        if provenance != "independent_image_annotation":
            raise ValueError("Projection labels require independent_image_annotation provenance")
        result = projected_landmark_error_px(predicted, reference.get("intrinsics"), reference["landmarks"])
        row["projection_error"] = result
    checks = [row["translation_error_m"] <= policy["max_translation_error_m"]]
    if not ambiguous:
        checks.append(row["rotation_error_deg"] <= policy["max_rotation_error_deg"])
    missing = []
    if policy["max_surface_error_m"] is not None:
        if row["surface_error"] is None:
            missing.append("surface_metric_required")
        else:
            checks.append(row["surface_error"]["p95_m"] <= policy["max_surface_error_m"])
    if policy["max_projection_error_px"] is not None and not ambiguous:
        if row["projection_error"] is None:
            missing.append("projection_metric_required")
        else:
            checks.append(row["projection_error"]["p95_px"] <= policy["max_projection_error_px"])
    if missing:
        row["label_missing_reason"] = ",".join(missing)
    else:
        row["quality_target"] = int(all(checks))
    return row


def _feature_schema(feature_dim, feature_names=None, feature_schema_version=None):
    try:
        from . import quality_ranker_features as schema
    except ImportError:
        schema = None
    if feature_names is None and schema is not None:
        feature_names = list(schema.FEATURE_NAMES)
    if feature_schema_version is None and schema is not None:
        feature_schema_version = schema.FEATURE_SCHEMA_VERSION
    if feature_names is None:
        feature_names = [f"feature_{index:02d}" for index in range(feature_dim)]
    if feature_schema_version is None:
        feature_schema_version = 1
    if (type(feature_schema_version) is not int or feature_schema_version < 1 or
            not isinstance(feature_names, list) or len(feature_names) != feature_dim or
            any(not isinstance(name, str) or not name for name in feature_names) or
            len(set(feature_names)) != len(feature_names)):
        raise ValueError("Feature schema identity does not match candidate feature vectors")
    if feature_dim < 1 or feature_dim > MAX_FEATURE_COUNT:
        raise ValueError("Candidate feature width exceeds the configured bound")
    return feature_schema_version, feature_names


def _has_label_key(value, depth=0):
    if depth > 32:
        raise ValueError("Trace context nesting exceeds the bound")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower().replace("-", "_")
            if any(token in lowered for token in _LABEL_KEY_PARTS):
                return True
            if _has_label_key(child, depth + 1):
                return True
    elif isinstance(value, list):
        return any(_has_label_key(child, depth + 1) for child in value)
    return False


def _frame_cache_row(record, ident, split, feature_dim):
    if _has_label_key(record):
        raise ValueError("Inference candidate trace contains a label/evaluation field")
    if record.get("trace_schema_version") != 1:
        raise ValueError("Unsupported candidate trace schema")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES_PER_FRAME:
        raise ValueError("Every frame requires a bounded candidates list, including [] for failures")
    context = record.get("context", {})
    if not isinstance(context, dict):
        raise ValueError("Candidate context must be a JSON object")
    for field in ("timings_ms", "resource_memory"):
        value = record.get(field)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{field} must be a numeric object or null")
    result_candidates = []
    seen = set()
    for raw in candidates:
        if not isinstance(raw, dict):
            raise ValueError("Candidate records must be objects")
        candidate_id = _string(raw.get("candidate_id"), "candidate_id", maximum=128)
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id {candidate_id!r} within a frame")
        seen.add(candidate_id)
        if type(raw.get("eligible")) is not bool:
            raise ValueError("Candidate eligibility must be an explicit boolean")
        rejection = raw.get("rejection")
        if rejection is not None and (not isinstance(rejection, str) or len(rejection) > 256):
            raise ValueError("Candidate rejection must be null or a bounded string")
        if raw["eligible"] and rejection is not None:
            raise ValueError("An eligible candidate cannot carry a rejection reason")
        appearance_summary_ms = raw.get("appearance_summary_ms")
        if appearance_summary_ms is not None and (
                isinstance(appearance_summary_ms, bool) or
                not isinstance(appearance_summary_ms, (int, float)) or
                not math.isfinite(float(appearance_summary_ms)) or appearance_summary_ms < 0):
            raise ValueError("appearance_summary_ms must be a finite nonnegative number or null")
        pose = raw.get("pose")
        if pose is not None:
            pose = _validate_pose(pose).tolist()
        elif raw["eligible"]:
            raise ValueError("Eligible candidates must include a valid pose")
        features = raw.get("features")
        if not isinstance(features, list) or len(features) != feature_dim:
            raise ValueError("Candidate feature width differs from the fixed feature schema")
        feature_values = []
        for value in features:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("Candidate feature vector must contain finite numbers only")
            feature_values.append(float(value))
        validation_stats = raw.get("validation_stats")
        if validation_stats is not None and not isinstance(validation_stats, dict):
            raise ValueError("validation_stats must be an object or null")
        candidate_row = {
            "candidate_id": candidate_id,
            "original_index": raw.get("original_index"),
            "stage": raw.get("stage"), "source": raw.get("source"),
            "pose": pose, "eligible": raw["eligible"], "rejection": rejection,
            "validation_stats": validation_stats, "features": feature_values,
            "legacy_score": raw.get("legacy_score"),
            # Timings are useful for evaluation-only latency summaries. The
            # training loader intentionally never exports these fields.
            "appearance_summary_ms": appearance_summary_ms,
        }
        _check_finite_tree(candidate_row)
        result_candidates.append(candidate_row)
    legacy_choice = record.get("legacy_choice_candidate_id")
    if legacy_choice is not None and legacy_choice not in seen:
        raise ValueError("legacy_choice_candidate_id must refer to a candidate in the same frame")
    # Preserve only prediction-safe fields from the trace. The reference label
    # document is joined after this projection and never enters this structure.
    safe_context = dict(context)
    for key, value in ident.items():
        if key in safe_context and safe_context[key] != value:
            raise ValueError(f"Candidate context {key} conflicts with the frozen frame identity")
        safe_context[key] = value
    frame = {
        "schema_version": 1,
        **ident,
        "split": split,
        "timestamp_s": record.get("timestamp_s"),
        "clock_mode": record.get("clock_mode"),
        "timings_ms": record.get("timings_ms"),
        "resource_memory": record.get("resource_memory"),
        "context": safe_context,
        "candidates": result_candidates,
        "rank_decision": record.get("rank_decision"),
        "legacy_choice_candidate_id": legacy_choice,
        "pose_state": record.get("pose_state"),
        "render_state": record.get("render_state"),
        "failure_reason": record.get("failure_reason"),
        "cameraFromObject": record.get("cameraFromObject"),
    }
    if frame["cameraFromObject"] is not None:
        frame["cameraFromObject"] = _validate_pose(frame["cameraFromObject"], "final pose").tolist()
    _check_finite_tree(frame)
    return frame


def _validate_trace_grouping(records, split_manifest):
    known_backgrounds = {row["background_id"] for row in split_manifest.get("background_catalog", [])
                         if isinstance(row, dict) and isinstance(row.get("background_id"), str)}
    expected = {(g["physical_object_id"], g["sequence_id"], frame["source_frame_id"]): (g, frame)
                for g in split_manifest["groups"] for frame in g["frames"]}
    seen = set()
    normalized = []
    for row in records:
        ident = _identity(row, known_background_ids=known_backgrounds)
        key = (ident["physical_object_id"], ident["sequence_id"], ident["source_frame_id"])
        if key not in expected:
            raise ValueError(f"Candidate trace contains a frame outside the frozen inventory: {key}")
        if key in seen:
            raise ValueError(f"Duplicate candidate trace frame {key}")
        seen.add(key)
        frozen, frozen_frame = expected[key]
        if ident["frame_id"] != frozen_frame["frame_id"]:
            raise ValueError(f"Candidate trace frame_id differs from the frozen ordinal/source mapping: {key}")
        for name in ("scene_id", "background_id", "background_kind", "provenance_kind"):
            if ident[name] != frozen[name]:
                raise ValueError(f"Candidate trace {name} differs from the frozen split manifest")
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        trace_mesh = row.get("mesh_sha256", context.get("mesh_sha256"))
        if row.get("mesh_sha256") is not None and context.get("mesh_sha256") is not None and row["mesh_sha256"] != context["mesh_sha256"]:
            raise ValueError("Candidate trace top-level/context mesh_sha256 values disagree")
        frozen_mesh = frozen.get("mesh_sha256")
        if frozen_mesh is None:
            if trace_mesh is not None:
                raise ValueError("Candidate trace asset identity was not bound into the frozen split manifest")
        else:
            try:
                trace_mesh = _optional_mesh_digest({"mesh_sha256": trace_mesh})
            except ValueError as exc:
                raise ValueError("Candidate trace lacks the frozen metric mesh_sha256") from exc
            if trace_mesh != frozen_mesh:
                raise ValueError("Candidate trace mesh_sha256 differs from the frozen split manifest")
            ident["mesh_sha256"] = frozen_mesh
        normalized.append((row, ident, frozen["split"]))
    if seen != set(expected):
        missing = len(set(expected) - seen)
        raise ValueError(f"Candidate trace omits {missing} frozen frame(s), including possible empty/failure frames")
    return normalized


def _canonical_frame_sort(row):
    return (row["split"], row["physical_object_id"], row["sequence_id"], _sort_frame(row["frame_id"]))


def _line_bytes(rows):
    if not rows:
        return b""
    return b"".join(_json_bytes(row) + b"\n" for row in rows)


def _ref_for_frame(index, frame):
    return index.get((frame["object_id"], frame["sequence_id"], frame["source_frame_id"]))


def _write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing cache artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def prepare_candidate_cache(output_dir, split_manifest, candidate_trace, reference_labels, *,
                            feature_names=None, feature_schema_version=None,
                            surface_points_by_object=None, maximum_bytes=MAX_CACHE_BYTES):
    """Validate/freeze trace rows and write split-sharded features and labels."""
    split_manifest = validate_split_manifest(split_manifest)
    trace_rows, trace_sha = _read_records(candidate_trace, list_key="frames")
    reference_rows, reference_sha = _read_records(reference_labels, list_key="frames")
    normalized = _validate_trace_grouping(trace_rows, split_manifest)
    feature_dim = None
    for row, _, _ in normalized:
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Candidate trace rows need a candidates list")
        for candidate in candidates:
            values = candidate.get("features") if isinstance(candidate, dict) else None
            if not isinstance(values, list):
                raise ValueError("Every candidate needs a fixed numeric feature vector")
            if feature_dim is None:
                feature_dim = len(values)
            elif len(values) != feature_dim:
                raise ValueError("Candidate feature width changed within a cache")
    if feature_dim is None:
        # Empty candidate caches remain useful audit artifacts when the core
        # feature schema is installed and known.
        try:
            from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
            feature_dim = len(FEATURE_NAMES)
            feature_names = feature_names or list(FEATURE_NAMES)
            feature_schema_version = feature_schema_version or FEATURE_SCHEMA_VERSION
        except ImportError as exc:
            raise ValueError("Cannot freeze an all-empty trace without an installed feature schema") from exc
    schema_version, names = _feature_schema(feature_dim, feature_names, feature_schema_version)
    identity_to_split = {(g["physical_object_id"], g["sequence_id"]): g["split"]
                         for g in split_manifest["groups"]}
    feature_frames = []
    for row, ident, split in normalized:
        feature_frames.append(_frame_cache_row(row, ident, split, feature_dim))
    feature_frames.sort(key=_canonical_frame_sort)
    if len(feature_frames) > MAX_FRAMES:
        raise ValueError("Prepared feature cache exceeds the frame bound")
    candidate_count = sum(len(row["candidates"]) for row in feature_frames)
    if candidate_count > MAX_CANDIDATES_TOTAL:
        raise ValueError("Prepared candidate count exceeds the configured bound")
    ref_index = _reference_index(reference_rows)
    frozen_reference_frames = {
        (group["physical_object_id"], group["sequence_id"], frame["source_frame_id"])
        for group in split_manifest["groups"] for frame in group["frames"]
    }
    unexpected_reference_frames = set(ref_index) - frozen_reference_frames
    if unexpected_reference_frames:
        raise ValueError(f"Reference labels contain frames outside the frozen inventory: {sorted(unexpected_reference_frames, key=str)[:3]}")
    policy = split_manifest["label_policy"]
    labels = []
    for frame in feature_frames:
        reference = _ref_for_frame(ref_index, frame)
        if reference is None:
            continue
        for candidate in frame["candidates"]:
            labels.append(_make_label(candidate, frame, reference, policy,
                                      surface_points_by_object=surface_points_by_object))
    labels.sort(key=lambda row: (row["split"], row["object_id"], row["sequence_id"],
                                 _sort_frame(row["frame_id"]), row["candidate_id"]))
    split_doc = split_manifest
    files = {}
    payloads = {}
    for split in SPLITS:
        frames_for_split = [row for row in feature_frames if row["split"] == split]
        labels_for_split = [row for row in labels if row["split"] == split]
        for kind, rows in (("features", frames_for_split), ("labels", labels_for_split)):
            filename = f"{kind}.{split}.jsonl"
            data = _line_bytes(rows)
            payloads[filename] = data
            files[f"{kind}_{split}"] = {"path": filename, "sha256": _sha256(data),
                                         "bytes": len(data), "rows": len(rows)}
    split_bytes = _json_bytes(split_doc) + b"\n"
    payloads["split_manifest.json"] = split_bytes
    files["split_manifest"] = {"path": "split_manifest.json", "sha256": _sha256(split_bytes),
                               "bytes": len(split_bytes), "rows": len(split_doc["groups"])}
    total_bytes = sum(len(data) for data in payloads.values())
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 1:
        raise ValueError("maximum_bytes must be a positive integer")
    if total_bytes > maximum_bytes:
        raise ValueError(f"Prepared cache requires {total_bytes} bytes, over the {maximum_bytes} byte bound")
    by_split = {split: {"frames": files[f"features_{split}"]["rows"],
                        "candidates": sum(len(row["candidates"]) for row in feature_frames if row["split"] == split),
                        "labeled_candidates": files[f"labels_{split}"]["rows"],
                        "features_file": files[f"features_{split}"]["path"],
                        "features_sha256": files[f"features_{split}"]["sha256"],
                        "labels_file": files[f"labels_{split}"]["path"],
                        "labels_sha256": files[f"labels_{split}"]["sha256"]}
                for split in SPLITS}
    provenance_counts = {kind: sum(row["provenance_kind"] == kind for row in feature_frames)
                         for kind in ("real", "synthetic", "fixture")}
    feature_hash = _sha256(_json_bytes({split: files[f"features_{split}"]["sha256"] for split in SPLITS}))
    label_hash = _sha256(_json_bytes({split: files[f"labels_{split}"]["sha256"] for split in SPLITS}))
    cache_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "quality-ranker-candidate-cache-v1",
        "feature_schema_version": schema_version,
        "feature_names": names,
        "feature_count": feature_dim,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "source_trace_sha256": trace_sha,
        "reference_labels_sha256": reference_sha,
        "feature_cache_sha256": feature_hash,
        "label_cache_sha256": label_hash,
        "frame_count": len(feature_frames),
        "candidate_count": candidate_count,
        "label_row_count": len(labels),
        "provenance_counts_by_frame": provenance_counts,
        "cache_bytes": total_bytes,
        "files": files,
        "splits": by_split,
        "inference_label_boundary": "features contain predictions/features only; references and quality targets are separate",
    }
    cache_manifest["cache_manifest_sha256"] = _sha256(_json_bytes(cache_manifest))
    manifest_bytes = _json_bytes(cache_manifest) + b"\n"
    if total_bytes + len(manifest_bytes) > maximum_bytes:
        raise ValueError("Cache manifest would exceed the configured byte bound")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Refusing to write into a non-empty cache directory")
    for name, data in payloads.items():
        _write_new(output_dir / name, data)
    _write_new(output_dir / "cache_manifest.json", manifest_bytes)
    return cache_manifest


def _cache_manifest(cache_dir):
    manifest, raw = _json_load(Path(cache_dir) / "cache_manifest.json")
    expected = manifest.get("cache_manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "cache_manifest_sha256"}
    if expected != _sha256(_json_bytes(body)):
        raise ValueError("Candidate cache manifest digest mismatch")
    if manifest.get("kind") != "quality-ranker-candidate-cache-v1" or manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate cache manifest")
    if not isinstance(manifest.get("files"), dict) or not isinstance(manifest.get("splits"), dict):
        raise ValueError("Candidate cache manifest has no frozen file inventory")
    feature_count = manifest.get("feature_count")
    names = manifest.get("feature_names")
    if (type(feature_count) is not int or not 1 <= feature_count <= MAX_FEATURE_COUNT or
            not isinstance(names, list) or len(names) != feature_count or
            any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names) or
            type(manifest.get("feature_schema_version")) is not int or manifest["feature_schema_version"] < 1):
        raise ValueError("Candidate cache feature schema is malformed")
    for key in ("source_trace_sha256", "reference_labels_sha256", "split_manifest_sha256",
                "feature_cache_sha256", "label_cache_sha256"):
        value = manifest.get(key)
        if (not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value.lower())):
            raise ValueError(f"Candidate cache manifest has invalid {key}")
    for split in SPLITS:
        for kind in ("features", "labels"):
            entry = manifest["files"].get(f"{kind}_{split}")
            if not isinstance(entry, dict) or not {"path", "sha256", "bytes", "rows"}.issubset(entry):
                raise ValueError(f"Candidate cache manifest lacks {kind} {split} shard metadata")
            if (type(entry["bytes"]) is not int or entry["bytes"] < 0 or
                    type(entry["rows"]) is not int or entry["rows"] < 0 or
                    not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64 or
                    any(char not in "0123456789abcdef" for char in entry["sha256"].lower())):
                raise ValueError(f"Candidate cache manifest has malformed {kind} {split} shard metadata")
        if not isinstance(manifest["splits"].get(split), dict):
            raise ValueError(f"Candidate cache manifest lacks {split} summary")
    return manifest, raw


def _safe_cache_file(cache_dir, entry):
    if not isinstance(entry, dict) or not {"path", "sha256", "bytes", "rows"}.issubset(entry):
        raise ValueError("Cache file entry is incomplete")
    relative = entry["path"]
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Cache file path must remain inside the cache directory")
    path = (Path(cache_dir) / relative).resolve()
    if not path.is_relative_to(Path(cache_dir).resolve()) or not path.is_file():
        raise ValueError("Cache file is missing or escapes the cache directory")
    return path


def _load_split_rows(cache_dir, manifest, split, *, verify_hash=True):
    if split not in SPLITS:
        raise ValueError(f"Unknown cache split {split!r}")
    data = manifest["splits"].get(split)
    if not isinstance(data, dict):
        raise ValueError(f"Cache has no {split!r} split record")
    features_entry = manifest["files"].get(f"features_{split}")
    labels_entry = manifest["files"].get(f"labels_{split}")
    if not features_entry or not labels_entry:
        raise ValueError(f"Cache lacks {split!r} feature/label shards")
    feature_path = _safe_cache_file(cache_dir, features_entry)
    label_path = _safe_cache_file(cache_dir, labels_entry)
    if verify_hash:
        if _file_sha256(feature_path) != features_entry["sha256"] or feature_path.stat().st_size != features_entry["bytes"]:
            raise ValueError(f"{split} feature shard digest/size mismatch")
        if _file_sha256(label_path) != labels_entry["sha256"] or label_path.stat().st_size != labels_entry["bytes"]:
            raise ValueError(f"{split} label shard digest/size mismatch")
    frames = _read_jsonl(feature_path)
    labels = _read_jsonl(label_path, max_rows=MAX_CANDIDATES_TOTAL)
    if len(frames) != features_entry["rows"] or len(labels) != labels_entry["rows"]:
        raise ValueError(f"{split} JSONL row count differs from its manifest")
    return frames, labels


def _audit_split(frames, labels, split, feature_count, split_manifest):
    issues = []
    groups = {(group["physical_object_id"], group["sequence_id"]): group
              for group in split_manifest["groups"]}
    expected_frames = {(group["physical_object_id"], group["sequence_id"], item["source_frame_id"]): (group, item)
                       for group in split_manifest["groups"] if group["split"] == split
                       for item in group["frames"]}
    candidates = {}
    frame_candidates = {}
    frames_by_key = {}
    frame_label_counts = {}
    seen_frames = set()
    for frame in frames:
        try:
            object_id = _string(frame.get("object_id"), "cached object_id")
            sequence_id = _string(frame.get("sequence_id"), "cached sequence_id")
            source_frame_id = _frame_id(frame.get("source_frame_id"), "cached source_frame_id")
            tracker_frame_id = _frame_id(frame.get("frame_id"), "cached frame_id")
        except ValueError:
            issues.append("feature frame has malformed identity")
            continue
        key = (object_id, sequence_id, source_frame_id)
        if frame.get("split") != split or key not in expected_frames:
            issues.append(f"feature frame outside frozen {split} inventory: {key}")
            continue
        if key in seen_frames:
            issues.append(f"duplicate feature frame: {key}")
        seen_frames.add(key)
        if _has_label_key(frame):
            issues.append(f"label/evaluation field leaked into features: {key}")
        group, frozen_frame = expected_frames[key]
        if tracker_frame_id != frozen_frame["frame_id"]:
            issues.append(f"cached frame_id differs from frozen ordinal/source mapping: {key}")
        for identity_field in ("physical_object_id", "scene_id", "background_id", "background_kind", "provenance_kind"):
            actual = frame.get(identity_field)
            if actual != group[identity_field]:
                issues.append(f"{identity_field} mismatch for {key}")
        if frame.get("mesh_sha256") != group.get("mesh_sha256"):
            issues.append(f"mesh_sha256 mismatch for {key}")
        context = frame.get("context")
        if not isinstance(context, dict):
            issues.append(f"context is not an object: {key}")
        elif context.get("reference_pose") is not None:
            issues.append(f"reference pose leaked into context: {key}")
        if frame.get("source_frame_id") not in [row["source_frame_id"] for row in group["frames"]]:
            issues.append(f"source frame id outside frozen sequence: {key}")
        if split in ("train", "validation") and frame.get("provenance_kind") == "fixture":
            issues.append(f"fixture-provenance data is not eligible for training: {key}")
        if _has_label_key(frame):
            issues.append(f"label/evaluation field leaked into cached frame: {key}")
        candidate_rows = frame.get("candidates")
        if not isinstance(candidate_rows, list):
            issues.append(f"candidate list is malformed for frame {key}")
            continue
        frame_candidate_keys = set()
        tracker_key = (object_id, sequence_id, tracker_frame_id)
        for candidate in candidate_rows:
            if not isinstance(candidate, dict):
                issues.append(f"candidate row is malformed for frame {key}")
                continue
            try:
                cid = _string(candidate.get("candidate_id"), "candidate_id", maximum=128)
            except ValueError:
                issues.append(f"candidate ID is malformed for frame {key}")
                continue
            candidate_key = (*tracker_key, cid)
            if candidate_key in candidates:
                issues.append(f"duplicate candidate key {candidate_key}")
            candidates[candidate_key] = candidate
            frame_candidate_keys.add(candidate_key)
            if type(candidate.get("eligible")) is not bool:
                issues.append(f"candidate eligibility is not boolean for {candidate_key}")
            if candidate.get("eligible") is True and candidate.get("rejection") is not None:
                issues.append(f"eligible candidate has a rejection reason for {candidate_key}")
            features = candidate.get("features")
            if not isinstance(features, list) or len(features) != feature_count:
                issues.append(f"feature width mismatch for {candidate_key}")
            else:
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in features):
                    issues.append(f"non-finite feature value for {candidate_key}")
            if candidate.get("pose") is not None:
                try:
                    _validate_pose(candidate.get("pose"), "candidate pose")
                except ValueError:
                    issues.append(f"candidate pose invalid for {candidate_key}")
            elif candidate.get("eligible") is True:
                issues.append(f"eligible candidate pose missing for {candidate_key}")
        frame_key = tracker_key
        frame_candidates[frame_key] = frame_candidate_keys
        frames_by_key[frame_key] = frame
    if seen_frames != set(expected_frames):
        issues.append(f"{len(set(expected_frames) - seen_frames)} frozen {split} frame(s) missing")
    seen_labels = set()
    for label in labels:
        try:
            key = _label_key(label)
        except (KeyError, TypeError, ValueError):
            issues.append("malformed candidate label key")
            continue
        if key in seen_labels:
            issues.append(f"duplicate candidate label {key}")
        seen_labels.add(key)
        if label.get("split") != split:
            issues.append(f"label in wrong split: {key}")
        if key not in candidates:
            issues.append(f"label has no same-frame candidate: {key}")
        label_frame_key = key[:3]
        frame = frames_by_key.get(label_frame_key)
        if frame is not None:
            if label.get("source_frame_id") != frame.get("source_frame_id"):
                issues.append(f"label source frame identity mismatch: {key}")
            if label.get("provenance_kind") != frame.get("provenance_kind"):
                issues.append(f"label provenance mismatch: {key}")
        frame_label_counts[label_frame_key] = frame_label_counts.get(label_frame_key, 0) + 1
        target = label.get("quality_target")
        if ((target is not None and type(target) is not int) or target not in (0, 1, None)):
            issues.append(f"invalid quality target: {key}")
        if type(label.get("observationally_ambiguous")) is not bool:
            issues.append(f"label lacks explicit ambiguity state: {key}")
        if label.get("observationally_ambiguous") and (label.get("orientation_supervision") is not False or label.get("rotation_error_deg") is not None):
            issues.append(f"ambiguous label forces orientation supervision: {key}")
        if label.get("rotation_error_deg") is not None and (not isinstance(label["rotation_error_deg"], (int, float)) or not math.isfinite(float(label["rotation_error_deg"]))):
            issues.append(f"invalid rotation error: {key}")
        if label.get("translation_error_m") is not None and (not isinstance(label["translation_error_m"], (int, float)) or not math.isfinite(float(label["translation_error_m"]))):
            issues.append(f"invalid translation error: {key}")
    for frame_key, count in frame_label_counts.items():
        expected_count = len(frame_candidates.get(frame_key, ()))
        if count != expected_count:
            issues.append(f"candidate label coverage is incomplete for frame {frame_key}")
    return issues


def _candidate_error_audit(frames, labels):
    """Summarize gate loss, label coverage, legacy choice, and fixed-set signal."""
    label_index = {_label_key(row): row for row in labels}
    totals = {
        "frames": len(frames), "candidates_before_gates": 0,
        "eligible_candidates_after_gates": 0, "rejected_candidates": 0,
        "empty_candidate_frames": 0, "all_gated_frames": 0,
        "missing_reference_frames": 0, "frames_with_any_ambiguous_label": 0,
        "frames_with_missing_candidate_labels": 0,
        "candidate_labels_missing": 0, "candidate_targets_unavailable": 0,
        "legacy_choices": 0, "legacy_choice_unknown": 0,
        "legacy_choice_ineligible": 0, "legacy_choice_unlabeled": 0,
        "legacy_choice_correct": 0, "legacy_choice_incorrect": 0,
        "legacy_choice_correct_before_gates": 0, "legacy_choice_incorrect_before_gates": 0,
        "legacy_choice_correct_after_gates": 0, "legacy_choice_incorrect_after_gates": 0,
        "rankable_groups_before_gates": 0, "rankable_groups_after_gates": 0,
        "rankable_pairs_after_gates": 0,
        "all_bad_groups_after_gates": 0,
        "oracle_before_gates_frames": 0, "oracle_after_gates_frames": 0,
        "oracle_before_gates_best_translation_error_m": [],
        "oracle_after_gates_best_translation_error_m": [],
        "oracle_before_gates_best_rotation_error_deg": [],
        "oracle_after_gates_best_rotation_error_deg": [],
        "provenance_counts_by_frame": {kind: 0 for kind in ("real", "synthetic", "fixture")},
        "signal_groups_by_provenance": {kind: 0 for kind in ("real", "synthetic", "fixture")},
    }
    for frame in frames:
        candidates = frame.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        eligible = [candidate for candidate in candidates
                    if candidate.get("eligible") and candidate.get("pose") is not None]
        totals["provenance_counts_by_frame"][frame.get("provenance_kind", "fixture")] += 1
        totals["candidates_before_gates"] += len(candidates)
        totals["eligible_candidates_after_gates"] += len(eligible)
        totals["rejected_candidates"] += sum(not candidate.get("eligible") for candidate in candidates)
        totals["empty_candidate_frames"] += not candidates
        totals["all_gated_frames"] += bool(candidates) and not eligible
        frame_labels = []
        missing_count = 0
        for candidate in candidates:
            key = (frame["object_id"], frame["sequence_id"], frame["frame_id"], candidate["candidate_id"])
            label = label_index.get(key)
            if label is None:
                missing_count += 1
                continue
            frame_labels.append(label)
            if label.get("quality_target") is None:
                totals["candidate_targets_unavailable"] += 1
        totals["candidate_labels_missing"] += missing_count
        totals["frames_with_missing_candidate_labels"] += missing_count > 0
        totals["missing_reference_frames"] += not frame_labels
        totals["frames_with_any_ambiguous_label"] += any(
            label.get("observationally_ambiguous") is True for label in frame_labels)

        def labeled_candidates(rows):
            result = []
            for candidate in rows:
                key = (frame["object_id"], frame["sequence_id"], frame["frame_id"], candidate["candidate_id"])
                label = label_index.get(key)
                if label is not None and label.get("quality_target") in (0, 1):
                    result.append((candidate, label))
            return result

        before_rows = labeled_candidates([candidate for candidate in candidates if candidate.get("pose") is not None])
        after_rows = labeled_candidates(eligible)
        for tag, rows in (("before", before_rows), ("after", after_rows)):
            values = {label["quality_target"] for _, label in rows}
            if values == {0, 1}:
                totals[f"rankable_groups_{tag}_gates"] += 1
                if tag == "after":
                    good = sum(label["quality_target"] == 1 for _, label in rows)
                    bad = sum(label["quality_target"] == 0 for _, label in rows)
                    totals["rankable_pairs_after_gates"] += good * bad
                    kind = frame.get("provenance_kind")
                    if kind in totals["signal_groups_by_provenance"]:
                        totals["signal_groups_by_provenance"][kind] += 1
            elif (tag == "after" and eligible and len(rows) == len(eligible) and
                  rows and values == {0}):
                totals["all_bad_groups_after_gates"] += 1

        for tag, rows in (("before", before_rows), ("after", after_rows)):
            if not rows:
                continue
            totals[f"oracle_{tag}_gates_frames"] += 1
            translations = [label.get("translation_error_m") for _, label in rows
                            if label.get("translation_error_m") is not None]
            rotations = [label.get("rotation_error_deg") for _, label in rows
                         if label.get("rotation_error_deg") is not None]
            if translations:
                totals[f"oracle_{tag}_gates_best_translation_error_m"].append(min(translations))
            if rotations:
                totals[f"oracle_{tag}_gates_best_rotation_error_deg"].append(min(rotations))

        legacy = frame.get("legacy_choice_candidate_id")
        if legacy is not None:
            totals["legacy_choices"] += 1
            selected = next((candidate for candidate in candidates
                             if candidate.get("candidate_id") == legacy), None)
            if selected is None:
                totals["legacy_choice_unknown"] += 1
            else:
                totals["legacy_choice_ineligible"] += not selected.get("eligible")
                label = label_index.get((frame["object_id"], frame["sequence_id"],
                                         frame["frame_id"], legacy))
                if label is None or label.get("quality_target") is None:
                    totals["legacy_choice_unlabeled"] += 1
                elif label["quality_target"] == 1:
                    totals["legacy_choice_correct"] += 1
                    totals["legacy_choice_correct_before_gates"] += 1
                    if selected.get("eligible") and selected.get("pose") is not None:
                        totals["legacy_choice_correct_after_gates"] += 1
                else:
                    totals["legacy_choice_incorrect"] += 1
                    totals["legacy_choice_incorrect_before_gates"] += 1
                    if selected.get("eligible") and selected.get("pose") is not None:
                        totals["legacy_choice_incorrect_after_gates"] += 1

    for key in ("oracle_before_gates_best_translation_error_m",
                "oracle_after_gates_best_translation_error_m",
                "oracle_before_gates_best_rotation_error_deg",
                "oracle_after_gates_best_rotation_error_deg"):
        values = totals[key]
        totals[key] = {"count": len(values), "mean": None if not values else float(np.mean(values)),
                       "median": None if not values else float(np.median(values)),
                       "p95": None if not values else float(np.percentile(values, 95))}
    totals["ranking_signal_available"] = bool(
        totals["rankable_groups_after_gates"] > 0 and
        frames and all(row.get("provenance_kind") != "fixture" for row in frames))
    totals["notes"] = [
        "Oracle errors are counterfactual within the recorded candidate set; before-gate values may use ineligible candidates.",
        "Availability and gate counts always use eligible candidates with poses.",
        "Rankable means at least one eligible labeled good and one eligible labeled bad candidate in the same frame.",
    ]
    return totals


def audit_cache(cache_dir, *, training_only=False):
    """Check cache identity, JSONL integrity, split isolation, and label joins.

    `training_only=True` reads train/validation shards only, so a trainer can
    perform its required pre-optimizer audit without opening test rows.
    """
    cache_dir = Path(cache_dir)
    manifest, _ = _cache_manifest(cache_dir)
    split_path = _safe_cache_file(cache_dir, manifest["files"].get("split_manifest"))
    if _file_sha256(split_path) != manifest["files"]["split_manifest"]["sha256"]:
        raise ValueError("Frozen split-manifest file digest mismatch")
    split_manifest, _ = _json_load(split_path)
    validate_split_manifest(split_manifest)
    if split_manifest["manifest_sha256"] != manifest.get("split_manifest_sha256"):
        raise ValueError("Cache and frozen split manifests disagree")
    expected_feature_hash = _sha256(_json_bytes({
        split: manifest["files"][f"features_{split}"]["sha256"] for split in SPLITS}))
    expected_label_hash = _sha256(_json_bytes({
        split: manifest["files"][f"labels_{split}"]["sha256"] for split in SPLITS}))
    manifest_issues = []
    if expected_feature_hash != manifest.get("feature_cache_sha256"):
        manifest_issues.append("aggregate feature-cache digest does not match shard inventory")
    if expected_label_hash != manifest.get("label_cache_sha256"):
        manifest_issues.append("aggregate label-cache digest does not match shard inventory")
    expected_frame_count = sum(manifest["files"][f"features_{split}"]["rows"] for split in SPLITS)
    expected_label_count = sum(manifest["files"][f"labels_{split}"]["rows"] for split in SPLITS)
    if expected_frame_count != manifest.get("frame_count") or expected_label_count != manifest.get("label_row_count"):
        manifest_issues.append("aggregate cache row counts do not match shard inventory")
    expected_candidates = 0
    for split in SPLITS:
        split_record = manifest["splits"][split]
        feature_entry = manifest["files"][f"features_{split}"]
        label_entry = manifest["files"][f"labels_{split}"]
        if (split_record.get("frames") != feature_entry["rows"] or
                split_record.get("labeled_candidates") != label_entry["rows"] or
                split_record.get("features_sha256") != feature_entry["sha256"] or
                split_record.get("labels_sha256") != label_entry["sha256"]):
            manifest_issues.append(f"{split} summary disagrees with its shard inventory")
        count = split_record.get("candidates")
        if type(count) is not int or count < 0:
            manifest_issues.append(f"{split} candidate count is malformed")
        else:
            expected_candidates += count
    if expected_candidates != manifest.get("candidate_count"):
        manifest_issues.append("aggregate candidate count does not match split summaries")
    audit_splits = ("train", "validation") if training_only else SPLITS
    issues = list(manifest_issues)
    frame_rows_by_split = {}
    label_rows_by_split = {}
    for split in audit_splits:
        feature_entry = manifest["files"].get(f"features_{split}")
        label_entry = manifest["files"].get(f"labels_{split}")
        if feature_entry is None or label_entry is None:
            issues.append(f"missing {split} feature/label shard")
            continue
        try:
            frames, labels = _load_split_rows(cache_dir, manifest, split)
            issues.extend(_audit_split(frames, labels, split, manifest["feature_count"], split_manifest))
            frame_rows_by_split[split] = frames
            label_rows_by_split[split] = labels
        except (OSError, ValueError, KeyError) as exc:
            issues.append(f"{split} shard invalid: {exc}")
    candidate_audits = {
        split: _candidate_error_audit(frame_rows_by_split[split], label_rows_by_split[split])
        for split in audit_splits if split in frame_rows_by_split and split in label_rows_by_split
    }
    train_signal = candidate_audits.get("train", {}).get("rankable_groups_after_gates", 0)
    validation_signal = candidate_audits.get("validation", {}).get("rankable_groups_after_gates", 0)
    ranking_signal_available = bool(train_signal and validation_signal and not issues)
    # Training audit proves train/validation shard hashes and frozen split
    # identity. The manifest still records the untouched test hashes, which are
    # verified by the parent full audit before reporting benchmark results.
    identity = {
        "feature_cache_sha256": manifest["feature_cache_sha256"],
        "label_cache_sha256": manifest["label_cache_sha256"],
        "split_sha256": manifest["split_manifest_sha256"],
        "feature_schema_version": manifest["feature_schema_version"],
        "feature_names": list(manifest["feature_names"]),
        "source_trace_sha256": manifest["source_trace_sha256"],
        "reference_labels_sha256": manifest["reference_labels_sha256"],
        "cache_manifest_sha256": manifest["cache_manifest_sha256"],
    }
    return {"approved": not issues, "audit_scope": "train_validation_only" if training_only else "all_splits",
            "identity": identity, "issues": issues,
            "frames_checked": sum(manifest["splits"][split]["frames"] for split in audit_splits),
            "candidates_checked": sum(manifest["splits"][split]["candidates"] for split in audit_splits),
            "labels_checked": sum(manifest["splits"][split]["labeled_candidates"] for split in audit_splits),
            "candidate_error_audit": candidate_audits,
            "ranking_signal": {
                "available": ranking_signal_available,
                "train_rankable_groups": train_signal,
                "validation_rankable_groups": validation_signal,
                "required_for_substantive_training": True,
                "reason": (None if ranking_signal_available else
                           "requires at least one eligible labeled good/bad candidate group in both train and validation"),
            },
            "ranking_signal_available": ranking_signal_available}


def load_cache_split(cache_dir, split, *, verify_hash=True):
    manifest, _ = _cache_manifest(cache_dir)
    split_manifest_path = _safe_cache_file(cache_dir, manifest["files"]["split_manifest"])
    split_manifest, _ = _json_load(split_manifest_path)
    validate_split_manifest(split_manifest)
    return _load_split_rows(cache_dir, manifest, split, verify_hash=verify_hash)


def load_training_split(cache_dir, split):
    """Return frame groups with candidate labels; never opens test shards."""
    if split not in ("train", "validation"):
        raise ValueError("The trainer may load only train or validation data")
    audit = audit_cache(cache_dir, training_only=True)
    if not audit["approved"]:
        raise ValueError("Training cache failed pre-optimizer audit: " + "; ".join(audit["issues"]))
    frames, labels = load_cache_split(cache_dir, split)
    label_index = {_label_key(row): row for row in labels}
    groups = []
    for frame in frames:
        candidates = []
        for candidate in frame["candidates"]:
            key = (frame["object_id"], frame["sequence_id"], frame["frame_id"], candidate["candidate_id"])
            label = label_index.get(key)
            if label is None or label.get("quality_target") is None:
                continue
            candidates.append({
                "candidate_id": candidate["candidate_id"],
                "features": list(candidate["features"]),
                "quality_target": label.get("quality_target"),
                "eligible": candidate["eligible"],
                "pose": candidate["pose"],
            })
        groups.append({"object_id": frame["object_id"], "sequence_id": frame["sequence_id"],
                       "frame_id": frame["frame_id"], "source_frame_id": frame["source_frame_id"],
                       "context": frame["context"], "candidates": candidates})
    return groups


def _read_bop_scene_inventory(source):
    if isinstance(source, (str, Path)):
        document, _ = _json_load(source)
    else:
        document = source
    if not isinstance(document, dict) or not isinstance(document.get("scenes"), list):
        raise ValueError("BOP needs a complete scene inventory sidecar with a 'scenes' list")
    result = {}
    for row in document["scenes"]:
        if not isinstance(row, dict) or row.get("inventory_complete") is not True:
            raise ValueError("BOP scene inventory rows must declare inventory_complete=true")
        scene_id = _string(str(row.get("scene_id")), "BOP scene inventory scene_id")
        ids = row.get("object_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("Each BOP scene inventory row must list every observed object_id")
        physical_ids = []
        for value in ids:
            if isinstance(value, bool):
                raise ValueError("BOP object ids must be positive integer IDs")
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("BOP object ids must be positive integer IDs") from exc
            if number <= 0 or str(value).strip() != str(number):
                raise ValueError("BOP object ids must be canonical positive integer IDs")
            physical_ids.append(f"bop-ycbv:{number:06d}")
        if len(set(physical_ids)) != len(physical_ids):
            raise ValueError("BOP scene inventory contains duplicate object IDs")
        scene_key = f"bop-ycbv:scene:{scene_id}"
        catalog_row = {"background_id": scene_key, "scene_id": scene_key,
                       "inventory_complete": True,
                       "physical_object_ids": sorted(physical_ids)}
        if scene_key in result and result[scene_key] != catalog_row:
            raise ValueError(f"Conflicting BOP scene inventories for {scene_key!r}")
        result[scene_key] = catalog_row
    return result


def adapt_bop_ycbv_manifests(inference_path, evaluator_path, *, scene_inventory):
    """Read an already acquired bounded BOP YCBV subset without network access.

    BOP translations are millimetres; emitted reference poses are metres. A
    complete scene inventory sidecar is mandatory: selected target-only
    evaluator rows cannot establish which other physical objects were visible.
    The adapter only accepts manifests written by prototype_dataset and never
    downloads source data.
    """
    inference, inference_raw = _json_load(inference_path)
    evaluator, evaluator_raw = _json_load(evaluator_path)
    if not isinstance(inference, dict) or not isinstance(evaluator, dict):
        raise ValueError("BOP inference and evaluator manifests must be JSON objects")
    if inference.get("schemaVersion") != 1 or evaluator.get("schemaVersion") != 1:
        raise ValueError("Unsupported BOP prototype manifest version")
    if inference.get("benchmarkId") != evaluator.get("benchmarkId") or inference.get("selectionDigest") != evaluator.get("selectionDigest"):
        raise ValueError("BOP inference/evaluator benchmark identities disagree")
    evaluator_digest = inference.get("evaluatorManifestSHA256")
    if not isinstance(evaluator_digest, str) or evaluator_digest != _sha256(evaluator_raw):
        raise ValueError("BOP evaluator manifest does not match its frozen inference hash")
    if inference.get("dataset") != "BOP YCB-Video" or evaluator.get("dataset") != "BOP YCB-Video":
        raise ValueError("BOP adapter accepts only YCB-Video prototype manifests")
    inf_objects, eval_objects = inference.get("objects"), evaluator.get("objects")
    if not isinstance(inf_objects, list) or not isinstance(eval_objects, list) or len(inf_objects) != len(eval_objects):
        raise ValueError("BOP inference/evaluator object inventories disagree")
    eval_by_reference = {row.get("referenceId"): row for row in eval_objects}
    records, references = [], []
    catalog = _read_bop_scene_inventory(scene_inventory)
    for inf_obj in inf_objects:
        reference_id = inf_obj.get("referenceId")
        eval_obj = eval_by_reference.get(reference_id)
        if eval_obj is None or inf_obj.get("objectId") != eval_obj.get("objectId") or inf_obj.get("sceneId") != eval_obj.get("sceneId"):
            raise ValueError("BOP reference object/scene identity mismatch")
        try:
            object_num = int(inf_obj["objectId"])
            scene_id = str(inf_obj["sceneId"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("BOP object/scene IDs are required") from exc
        canonical = inf_obj.get("canonicalYcbId")
        if not isinstance(canonical, str) or not canonical:
            raise ValueError("BOP canonical physical identity is required")
        physical = f"bop-ycbv:{object_num:06d}"
        mesh_ref = inf_obj.get("model")
        mesh_digest = None
        if isinstance(mesh_ref, dict):
            try:
                mesh_digest = _optional_mesh_digest({"mesh_sha256": mesh_ref.get("sha256")})
            except ValueError:
                # prototype_dataset uses zero placeholders before acquisition;
                # never treat that placeholder as a real mesh identity.
                if mesh_ref.get("sha256") != "0" * 64:
                    raise
        sequence = f"bop-ycbv:scene-{scene_id}:object-{object_num:06d}"
        background = f"bop-ycbv:scene:{scene_id}"
        if background not in catalog:
            raise ValueError(f"BOP scene {scene_id!r} is missing from the complete scene inventory sidecar")
        frames = inf_obj.get("frames")
        eval_frames = eval_obj.get("frames")
        if not isinstance(frames, list) or not isinstance(eval_frames, list) or len(frames) != len(eval_frames):
            raise ValueError("BOP evaluation frame rows differ between manifests")
        eval_by_frame = {row.get("frameId"): row for row in eval_frames}
        if len(eval_by_frame) != len(eval_frames):
            raise ValueError("Duplicate BOP evaluator frame IDs")
        if [row.get("frameId") for row in frames] != [row.get("frameId") for row in eval_frames]:
            raise ValueError("BOP evaluator frames must match inference frames in order")
        source_ids = [frame.get("frameId") for frame in frames]
        if (any(type(value) is not int or value < 0 for value in source_ids) or
                len(set(source_ids)) != len(source_ids) or
                any(right <= left for left, right in zip(source_ids, source_ids[1:]))):
            raise ValueError("BOP source frame IDs must be unique, increasing nonnegative integers")
        for ordinal, frame in enumerate(frames):
            frame_id = frame.get("frameId")
            evaluator_frame = eval_by_frame[frame_id]
            if evaluator_frame.get("frameId") != frame_id:
                raise ValueError("BOP frame ID mismatch")
            records.append({"physical_object_id": physical, "object_id": physical,
                            "scene_id": f"bop-ycbv:scene:{scene_id}",
                            "sequence_id": sequence, "background_id": background,
                            "background_kind": "observed",
                            "provenance_kind": "real", "frame_id": ordinal,
                            "source_frame_id": int(frame_id), "split": "dev" if _is_dev_only(physical) else None,
                            "aliases": [canonical], **({"mesh_sha256": mesh_digest} if mesh_digest else {})})
            pose = evaluator_frame.get("evaluationPose")
            if not isinstance(pose, dict):
                raise ValueError("BOP evaluator frame lacks evaluationPose")
            rotation = np.asarray(pose.get("cam_R_m2c"), dtype=np.float64)
            translation = np.asarray(pose.get("cam_t_m2c"), dtype=np.float64)
            if rotation.shape != (9,) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                raise ValueError("BOP pose fields are malformed")
            metric = np.eye(4, dtype=np.float64)
            metric[:3, :3] = rotation.reshape(3, 3)
            metric[:3, 3] = translation * 0.001
            _validate_pose(metric, "BOP converted pose")
            references.append({"object_id": physical, "sequence_id": sequence,
                               "frame_id": ordinal, "source_frame_id": int(frame_id),
                               "pose": metric.tolist(),
                               # Dataset pose rows do not prove that orientation
                               # is visually recoverable for this specific view.
                               "observationally_ambiguous": evaluator_frame.get("observationallyAmbiguous", True),
                               "label_kind": evaluator.get("labelKind", "dataset_reference")})
    return {"schema_version": 1, "source": "already_acquired_bop_ycbv",
            "background_catalog": sorted(catalog.values(), key=lambda row: row["background_id"]), "records": records,
            "inference_sha256": _sha256(inference_raw), "evaluator_sha256": _sha256(evaluator_raw)}, references


def _write_json(path, value):
    _write_new(Path(path), json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="freeze explicit sequence split assignments")
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--assignments", type=Path, help="JSON object mapping sequence_id to split")
    freeze.add_argument("--max-rotation-error-deg", type=float, required=True)
    freeze.add_argument("--max-translation-error-m", type=float, required=True)
    freeze.add_argument("--max-surface-error-m", type=float)
    freeze.add_argument("--max-projection-error-px", type=float)
    bop = sub.add_parser("bop-adapter", help="adapt an already acquired BOP YCBV subset; no downloads")
    bop.add_argument("--inference", type=Path, required=True)
    bop.add_argument("--evaluator", type=Path, required=True)
    bop.add_argument("--scene-inventory", type=Path, required=True,
                     help="complete already acquired BOP scene object inventory JSON; target-only rows are rejected")
    bop.add_argument("--output-dir", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="write a bounded, label-isolated candidate cache")
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--trace", type=Path, required=True)
    prepare.add_argument("--reference-labels", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    audit = sub.add_parser("audit", help="verify candidate cache integrity and identity boundaries")
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--training-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        inventory, _ = _json_load(args.inventory)
        assignments = None
        if args.assignments:
            assignments, _ = _json_load(args.assignments)
        policy = {"max_rotation_error_deg": args.max_rotation_error_deg,
                  "max_translation_error_m": args.max_translation_error_m,
                  "max_surface_error_m": args.max_surface_error_m,
                  "max_projection_error_px": args.max_projection_error_px}
        manifest = freeze_split_manifest(inventory, assignments, label_policy=policy)
        _write_json(args.output, manifest)
        print(json.dumps({"status": "frozen", "manifest_sha256": manifest["manifest_sha256"],
                          "groups": len(manifest["groups"]),
                          "frames": sum(len(row["frames"]) for row in manifest["groups"])}))
    elif args.command == "bop-adapter":
        inventory, references = adapt_bop_ycbv_manifests(
            args.inference, args.evaluator, scene_inventory=args.scene_inventory)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if any(args.output_dir.iterdir()):
            raise FileExistsError("Refusing to write BOP adapter output into a non-empty directory")
        _write_json(args.output_dir / "inventory.json", inventory)
        _write_new(args.output_dir / "references.jsonl", _line_bytes(references))
        print(json.dumps({"status": "adapted", "frames": len(inventory["records"]),
                          "references": len(references), "network_access": False}))
    elif args.command == "prepare":
        split_manifest = read_split_manifest(args.split_manifest)
        result = prepare_candidate_cache(args.output_dir, split_manifest, args.trace, args.reference_labels)
        print(json.dumps({"status": "prepared", "frames": result["frame_count"],
                          "candidates": result["candidate_count"],
                          "cache_manifest_sha256": result["cache_manifest_sha256"]}))
    else:
        result = audit_cache(args.cache_dir, training_only=args.training_only)
        print(json.dumps(result, indent=2))
        if not result["approved"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
