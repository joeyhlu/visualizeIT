"""Render a bounded synthetic RGB bundle with labels kept in a sidecar.

This is an executable cloud-side data tool. It consumes a caller-supplied
metric GLB, a split manifest frozen before rendering, and emits only the RGB
video plus strict schema-2 physical-time input metadata. Pose labels are stored
separately and are never passed to inference candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .quality_ranker_data import _file_sha256, _write_json, _write_new, read_split_manifest
from .quality_time import PHYSICAL


MAX_FRAMES = 300
MAX_DIMENSION = 1280
MAX_MESH_BYTES = 64 * 1024 * 1024
MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_FPS = 60

BROADER_RECIPE_SCHEMA = 1
BROADER_RECIPE_KIND = "quality-ranker-synthetic-recipe-v1"
ORIENTATION_RECIPE_KIND = "quality-ranker-synthetic-orientation-recipe-v1"
SINGLE_VIEW_RECIPE_KIND = "quality-ranker-synthetic-single-view-recipe-v1"
_RECIPE_SHA_FIELDS = {"schema_version", "kind", "recipe_id", "split_binding",
                      "options", "sequences", "recipe_sha256"}
_RECIPE_OPTION_FIELDS = {"frames", "width", "height", "fps", "seed", "mesh_units",
                         "background_rgb", "intrinsics", "trajectory", "occluder",
                         "label_policy", "limits"}
_RECIPE_SEQUENCE_FIELDS = {"sequence_id", "physical_object_id", "scene_id", "split",
                           "mesh_sha256", "yaw0_deg", "pitch_deg",
                           "observationally_ambiguous"}
_ORIENTATION_RECIPE_FIELDS = _RECIPE_SHA_FIELDS | {"source_binding"}
_ORIENTATION_SOURCE_FIELDS = {"generator_path", "generator_sha256",
                              "preparation_path", "preparation_sha256"}
_ORIENTATION_VIEW_PAIRS = ((45, -15), (225, 15))
_SINGLE_VIEW_PAIR = (75, -15)
_ORIENTATION_SCHEDULE = (
    (17, "train", 45, -15, "yaw45-pitchm15"),
    (27, "validation", 45, -15, "yaw45-pitchm15"),
    (17, "train", 225, 15, "yaw225-pitchp15"),
    (27, "validation", 225, 15, "yaw225-pitchp15"),
)
ORIENTATION_ASSET_PINS = {
    "hot3d:obj_000017": "e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc",
    "hot3d:obj_000027": "b5354f0785c6a4b97dcadb3fe2622e9dc7e2c794bd69d648b72482dc7bdb8e0a",
}
SINGLE_VIEW_ASSET_PIN = ORIENTATION_ASSET_PINS["hot3d:obj_000017"]


def _canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def seal_recipe(recipe):
    """Return a recipe with its canonical self-digest populated."""
    if not isinstance(recipe, dict):
        raise ValueError("Synthetic recipe must be a JSON object")
    body = dict(recipe)
    body.pop("recipe_sha256", None)
    try:
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("Synthetic recipe must contain finite JSON values") from exc
    return {**body, "recipe_sha256": digest}


def _read_unique_json_bytes(raw, *, name):
    def unique_pairs(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key in {name}: {key!r}")
            value[key] = child
        return value
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {name}") from exc


def _checked_sha256(value, name):
    if (type(value) is not str or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value) or
            len(set(value)) == 1):
        raise ValueError(f"{name} must be a non-placeholder lowercase SHA-256 digest")
    return value


def _validate_nested_json_numbers(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot contain boolean numeric values")
    if type(value) in (int, float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} cannot contain non-finite numbers")
    elif isinstance(value, list):
        for child in value:
            _validate_nested_json_numbers(child, name)
    elif isinstance(value, dict):
        for child in value.values():
            _validate_nested_json_numbers(child, name)


def _validate_recipe_shape(recipe):
    if not isinstance(recipe, dict) or set(recipe) != _RECIPE_SHA_FIELDS:
        raise ValueError("Synthetic recipe has unknown or missing top-level fields")
    if recipe.get("schema_version") != BROADER_RECIPE_SCHEMA or type(recipe.get("schema_version")) is not int:
        raise ValueError("Unsupported synthetic recipe schema_version")
    if recipe.get("kind") != BROADER_RECIPE_KIND:
        raise ValueError("Unsupported synthetic recipe kind")
    if type(recipe.get("recipe_id")) is not str or not recipe["recipe_id"].strip():
        raise ValueError("Synthetic recipe_id must be a nonempty string")
    binding = recipe.get("split_binding")
    if not isinstance(binding, dict) or set(binding) != {"manifest_sha256", "file_sha256"}:
        raise ValueError("Synthetic recipe split_binding has unknown or missing fields")
    _checked_sha256(binding["manifest_sha256"], "split manifest_sha256")
    _checked_sha256(binding["file_sha256"], "split file_sha256")

    options = recipe.get("options")
    if not isinstance(options, dict) or set(options) != _RECIPE_OPTION_FIELDS:
        raise ValueError("Synthetic recipe options have unknown or missing fields")
    expected_scalars = {"frames": 24, "width": 640, "height": 480,
                        "fps": 10.0, "seed": 0}
    for field, expected in expected_scalars.items():
        if type(options.get(field)) not in (int, float) or isinstance(options[field], bool):
            raise ValueError(f"Synthetic recipe {field} has the wrong type")
        if options[field] != expected:
            raise ValueError(f"Synthetic recipe {field} differs from its frozen value")
    if type(options["frames"]) is not int or type(options["width"]) is not int or type(options["height"]) is not int or type(options["seed"]) is not int:
        raise ValueError("Synthetic recipe frame, size and seed options must be integers")
    if type(options["fps"]) not in (int, float) or not math.isfinite(float(options["fps"])):
        raise ValueError("Synthetic recipe fps must be finite")
    if options.get("mesh_units") != "metres" or type(options.get("mesh_units")) is not str:
        raise ValueError("Synthetic recipe mesh_units must be metres")
    if options["background_rgb"] != [127, 127, 127] or type(options["background_rgb"]) is not list:
        raise ValueError("Synthetic recipe background_rgb must be [127,127,127]")
    _validate_nested_json_numbers(options["background_rgb"], "background_rgb")
    expected_k = [[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]]
    if options["intrinsics"] != expected_k:
        raise ValueError("Synthetic recipe intrinsics differ from the frozen 640x480 camera")
    _validate_nested_json_numbers(options["intrinsics"], "intrinsics")
    expected_trajectory = {"rotation_order": "RyRx", "yaw_sweep_deg": 40.0,
                           "roll_deg": 0.0, "yaw_origins_deg": [0, 90, 180, 270],
                           "pitch_deg": [-25, 25]}
    if options["trajectory"] != expected_trajectory:
        raise ValueError("Synthetic recipe trajectory differs from the frozen schedule")
    _validate_nested_json_numbers(options["trajectory"], "trajectory")
    expected_occluder = {
        "frames_inclusive": [6, 13], "width_fraction": 0.22,
        "height_fraction": 0.70, "rgb": [32, 32, 32],
        "rounding": "half_up", "placement_rounding": "floor_plus_half",
        "center_x_fractions": [0.2, 0.8], "center_y_fraction": 0.5,
    }
    if options["occluder"] != expected_occluder:
        raise ValueError("Synthetic recipe occluder differs from the frozen rectangle policy")
    _validate_nested_json_numbers(options["occluder"], "occluder")
    limits = options["limits"]
    if (not isinstance(limits, dict) or set(limits) != {"max_output_bytes", "max_video_bytes"} or
            limits.get("max_output_bytes") != 268435456 or
            limits.get("max_video_bytes") != 67108864 or
            any(type(limits.get(name)) is not int for name in limits)):
        raise ValueError("Synthetic recipe limits differ from the frozen output bounds")
    if not isinstance(options["label_policy"], dict):
        raise ValueError("Synthetic recipe label_policy must be an object")

    sequences = recipe.get("sequences")
    if type(sequences) is not list or len(sequences) != 16:
        raise ValueError("Synthetic recipe must freeze exactly 16 sequence rows")
    seen = set()
    by_object = {}
    for row in sequences:
        if not isinstance(row, dict) or set(row) != _RECIPE_SEQUENCE_FIELDS:
            raise ValueError("Synthetic recipe sequence row has unknown or missing fields")
        if any(type(row.get(key)) is not str or not row[key].strip()
               for key in ("sequence_id", "physical_object_id", "scene_id")):
            raise ValueError("Synthetic recipe sequence identity fields must be nonempty strings")
        if row["sequence_id"] in seen:
            raise ValueError("Synthetic recipe contains a duplicate sequence_id")
        seen.add(row["sequence_id"])
        if row.get("split") not in ("train", "validation"):
            raise ValueError("Synthetic recipe sequences must belong to train or validation")
        _checked_sha256(row.get("mesh_sha256"), "sequence mesh_sha256")
        if type(row.get("yaw0_deg")) is not int or row["yaw0_deg"] not in (0, 90, 180, 270):
            raise ValueError("Synthetic recipe yaw0_deg must be one of 0,90,180,270")
        if type(row.get("pitch_deg")) is not int or row["pitch_deg"] not in (-25, 25):
            raise ValueError("Synthetic recipe pitch_deg must be -25 or 25")
        if row.get("observationally_ambiguous") is not True:
            raise ValueError("Broader synthetic labels must conservatively mark orientation ambiguous")
        by_object.setdefault(row["physical_object_id"], []).append(row)
    if len(by_object) != 2:
        raise ValueError("Synthetic recipe must contain exactly two physical objects")
    splits = {}
    for physical_id, rows in by_object.items():
        object_splits = {row["split"] for row in rows}
        if len(rows) != 8 or len(object_splits) != 1:
            raise ValueError("Each recipe object must own exactly eight sequences in one split")
        split_name = next(iter(object_splits))
        if split_name in splits:
            raise ValueError("Each synthetic object must have a distinct frozen split")
        splits[split_name] = physical_id
        if {(row["yaw0_deg"], row["pitch_deg"]) for row in rows} != {
                (yaw, pitch) for yaw in (0, 90, 180, 270) for pitch in (-25, 25)}:
            raise ValueError("Each object must cross all four yaws with both pitches exactly once")
    if set(splits) != {"train", "validation"}:
        raise ValueError("Recipe objects must be assigned to train and validation separately")
    _checked_sha256(recipe.get("recipe_sha256"), "recipe_sha256")
    unsigned = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    expected_digest = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if recipe["recipe_sha256"] != expected_digest:
        raise ValueError("Synthetic recipe self-digest mismatch")


def _orientation_identity(number, split, yaw0, pitch, suffix):
    return {
        "sequence_id": f"ranker-orientation-hot3d{number}-v1-{suffix}",
        "physical_object_id": f"hot3d:obj_{number:06d}",
        "scene_id": f"synthetic-scene:hot3d{number}:orientation-v1:{suffix}",
        "split": split,
        "yaw0_deg": yaw0,
        "pitch_deg": pitch,
        "observationally_ambiguous": True,
    }


def _validate_orientation_recipe_shape(recipe):
    if not isinstance(recipe, dict) or set(recipe) != _ORIENTATION_RECIPE_FIELDS:
        raise ValueError("Orientation recipe has unknown or missing top-level fields")
    if type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("Unsupported orientation recipe schema_version")
    if recipe.get("kind") != ORIENTATION_RECIPE_KIND:
        raise ValueError("Unsupported orientation recipe kind")
    if recipe.get("recipe_id") != "ranker-orientation-synthetic-v1":
        raise ValueError("Orientation recipe_id differs from its frozen value")
    binding = recipe.get("split_binding")
    if not isinstance(binding, dict) or set(binding) != {"manifest_sha256", "file_sha256"}:
        raise ValueError("Orientation recipe split_binding has unknown or missing fields")
    _checked_sha256(binding.get("manifest_sha256"), "split manifest_sha256")
    _checked_sha256(binding.get("file_sha256"), "split file_sha256")
    source = recipe.get("source_binding")
    if not isinstance(source, dict) or set(source) != _ORIENTATION_SOURCE_FIELDS:
        raise ValueError("Orientation recipe source_binding has unknown or missing fields")
    if (source.get("generator_path") != "bench/quality_ranker_synthetic.py" or
            source.get("preparation_path") != "scripts/cloud_ranker_prepare_orientation.py"):
        raise ValueError("Orientation recipe source paths differ from their frozen values")
    _checked_sha256(source.get("generator_sha256"), "generator_sha256")
    _checked_sha256(source.get("preparation_sha256"), "preparation_sha256")

    options = recipe.get("options")
    if not isinstance(options, dict) or set(options) != _RECIPE_OPTION_FIELDS:
        raise ValueError("Orientation recipe options have unknown or missing fields")
    expected_scalars = {"frames": 12, "width": 640, "height": 480,
                        "fps": 10.0, "seed": 0}
    for field, expected in expected_scalars.items():
        value = options.get(field)
        if isinstance(value, bool) or type(value) not in (int, float) or value != expected:
            raise ValueError(f"Orientation recipe {field} differs from its frozen value")
    if any(type(options[name]) is not int for name in ("frames", "width", "height", "seed")):
        raise ValueError("Orientation recipe frame, size and seed options must be integers")
    if type(options["fps"]) not in (int, float) or not math.isfinite(float(options["fps"])):
        raise ValueError("Orientation recipe fps must be finite")
    if options.get("mesh_units") != "metres" or type(options.get("mesh_units")) is not str:
        raise ValueError("Orientation recipe mesh_units must be metres")
    if type(options.get("background_rgb")) is not list or options["background_rgb"] != [127, 127, 127]:
        raise ValueError("Orientation recipe background_rgb must be [127,127,127]")
    expected_k = [[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]]
    if options.get("intrinsics") != expected_k:
        raise ValueError("Orientation recipe intrinsics differ from the frozen 640x480 camera")
    expected_trajectory = {
        "rotation_order": "RyRx", "yaw_sweep_deg": 20.0, "roll_deg": 0.0,
        "paired_views": [
            {"yaw0_deg": 45, "pitch_deg": -15},
            {"yaw0_deg": 225, "pitch_deg": 15},
        ],
    }
    if options.get("trajectory") != expected_trajectory:
        raise ValueError("Orientation recipe trajectory differs from the frozen paired schedule")
    if options.get("occluder") is not None:
        raise ValueError("Orientation recipe must not add an occluder")
    expected_policy = {"max_rotation_error_deg": 10.0,
                       "max_translation_error_m": 0.02,
                       "max_surface_error_m": None,
                       "max_projection_error_px": None}
    if options.get("label_policy") != expected_policy:
        raise ValueError("Orientation recipe label policy differs from the frozen thresholds")
    limits = options.get("limits")
    if (not isinstance(limits, dict) or set(limits) != {"max_output_bytes", "max_video_bytes"} or
            any(type(limits.get(name)) is not int for name in limits) or
            limits.get("max_output_bytes") != 32 * 1024**2 or
            limits.get("max_video_bytes") != 8 * 1024**2):
        raise ValueError("Orientation recipe limits differ from the frozen output bounds")
    _validate_nested_json_numbers(options, "orientation options")

    sequences = recipe.get("sequences")
    if type(sequences) is not list or len(sequences) != 4:
        raise ValueError("Orientation recipe must freeze exactly four sequence rows")
    expected_rows = [_orientation_identity(*schedule) for schedule in _ORIENTATION_SCHEDULE]
    seen = set()
    for index, (row, expected) in enumerate(zip(sequences, expected_rows)):
        if not isinstance(row, dict) or set(row) != _RECIPE_SEQUENCE_FIELDS:
            raise ValueError("Orientation recipe sequence row has unknown or missing fields")
        _checked_sha256(row.get("mesh_sha256"), "sequence mesh_sha256")
        for field, value in expected.items():
            if row.get(field) != value or (field == "observationally_ambiguous" and row[field] is not True):
                raise ValueError(f"Orientation recipe sequence {index} differs from its fixed paired view")
        if row["mesh_sha256"] != ORIENTATION_ASSET_PINS[row["physical_object_id"]]:
            raise ValueError("Orientation recipe mesh SHA-256 differs from its pinned HOT3D asset")
        if row["sequence_id"] in seen:
            raise ValueError("Orientation recipe contains a duplicate sequence_id")
        seen.add(row["sequence_id"])
    _checked_sha256(recipe.get("recipe_sha256"), "recipe_sha256")
    unsigned = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    expected_digest = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if recipe["recipe_sha256"] != expected_digest:
        raise ValueError("Orientation recipe self-digest mismatch")


def validate_orientation_recipe(recipe_path, split_manifest_path, mesh_path, sequence_id, *,
                                frames, width, height, fps, mesh_units,
                                observationally_ambiguous):
    """Authenticate the strict four-sequence orientation recipe and selected mesh."""
    recipe_path = Path(recipe_path).resolve(strict=True)
    split_path = Path(split_manifest_path).resolve(strict=True)
    mesh_path = Path(mesh_path).resolve(strict=True)
    recipe_raw = recipe_path.read_bytes()
    if len(recipe_raw) > 16 * 1024 * 1024:
        raise ValueError("Orientation recipe exceeds the 16 MiB inventory bound")
    recipe = _read_unique_json_bytes(recipe_raw, name="orientation recipe")
    _validate_orientation_recipe_shape(recipe)

    source_root = Path(__file__).resolve().parents[1]
    generator_path = Path(__file__).resolve()
    preparation_path = source_root / "scripts" / "cloud_ranker_prepare_orientation.py"
    if (not preparation_path.is_file() or
            _file_sha256(generator_path) != recipe["source_binding"]["generator_sha256"] or
            _file_sha256(preparation_path) != recipe["source_binding"]["preparation_sha256"]):
        raise ValueError("Orientation recipe executing source bytes differ from source_binding")

    split_raw = split_path.read_bytes()
    split = read_split_manifest(split_path)
    if recipe["split_binding"] != {
            "manifest_sha256": split["manifest_sha256"],
            "file_sha256": hashlib.sha256(split_raw).hexdigest()}:
        raise ValueError("Orientation recipe is bound to different frozen split bytes")
    if recipe["options"]["label_policy"] != split["label_policy"]:
        raise ValueError("Orientation recipe label policy differs from the frozen split")
    groups_by_id = {group["sequence_id"]: group for group in split["groups"]}
    rows_by_id = {row["sequence_id"]: row for row in recipe["sequences"]}
    if (len(groups_by_id) != 4 or len(rows_by_id) != 4 or
            set(rows_by_id) != set(groups_by_id)):
        raise ValueError("Orientation recipe must exactly cover the four frozen split groups")
    expected_frames = [{"frame_id": ordinal, "source_frame_id": ordinal}
                       for ordinal in range(12)]
    for frozen_id, group in groups_by_id.items():
        row = rows_by_id[frozen_id]
        for key in ("sequence_id", "physical_object_id", "scene_id", "split", "mesh_sha256"):
            if row[key] != group.get(key):
                raise ValueError(f"Orientation recipe sequence {key} differs from frozen split membership")
        if (group.get("provenance_kind") != "synthetic" or
                group.get("background_kind") != "procedural" or
                group.get("background_id") != "procedural:gray"):
            raise ValueError("Orientation recipe requires procedural-gray synthetic groups")
        if group.get("frames") != expected_frames or group.get("frame_ids") != list(range(12)):
            raise ValueError("Orientation recipe split groups must freeze frame IDs 0..11")
    if type(sequence_id) is not str or sequence_id not in groups_by_id:
        raise ValueError("Requested sequence_id must resolve to one orientation split group")
    row, group = rows_by_id[sequence_id], groups_by_id[sequence_id]
    if (type(frames) is not int or frames != 12 or type(width) is not int or width != 640 or
            type(height) is not int or height != 480 or isinstance(fps, bool) or
            type(fps) not in (int, float) or not math.isfinite(float(fps)) or float(fps) != 10.0 or
            mesh_units != "metres" or type(observationally_ambiguous) is not bool or
            observationally_ambiguous is not True):
        raise ValueError("Orientation generation options differ from the frozen recipe")
    if mesh_path.suffix.lower() != ".glb" or mesh_path.stat().st_size > MAX_MESH_BYTES:
        raise ValueError(f"A caller-supplied GLB of at most {MAX_MESH_BYTES} bytes is required")
    mesh_digest = _file_sha256(mesh_path)
    if mesh_digest != row["mesh_sha256"] or mesh_digest != group.get("mesh_sha256"):
        raise ValueError("Selected metric mesh bytes differ from the orientation recipe/split SHA-256")
    return {"recipe": recipe, "recipe_sha256": recipe["recipe_sha256"],
            "sequence": row, "group": group, "split": split,
            "mesh_sha256": mesh_digest}


def _single_view_identity():
    return {
        "sequence_id": "ranker-single-view-hot3d17-v1-yaw75-pitchm15",
        "physical_object_id": "hot3d:obj_000017",
        "scene_id": "synthetic-scene:hot3d17:single-view-v1:yaw75-pitchm15",
        "split": "train",
        "yaw0_deg": 75,
        "pitch_deg": -15,
        "observationally_ambiguous": True,
    }


def _validate_single_view_recipe_shape(recipe):
    if not isinstance(recipe, dict) or set(recipe) != _ORIENTATION_RECIPE_FIELDS:
        raise ValueError("Single-view recipe has unknown or missing top-level fields")
    if type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("Unsupported single-view recipe schema_version")
    if recipe.get("kind") != SINGLE_VIEW_RECIPE_KIND:
        raise ValueError("Unsupported single-view recipe kind")
    if recipe.get("recipe_id") != "ranker-single-view-synthetic-v1":
        raise ValueError("Single-view recipe_id differs from its frozen value")
    binding = recipe.get("split_binding")
    if not isinstance(binding, dict) or set(binding) != {"manifest_sha256", "file_sha256"}:
        raise ValueError("Single-view recipe split_binding has unknown or missing fields")
    _checked_sha256(binding.get("manifest_sha256"), "split manifest_sha256")
    _checked_sha256(binding.get("file_sha256"), "split file_sha256")
    source = recipe.get("source_binding")
    if not isinstance(source, dict) or set(source) != _ORIENTATION_SOURCE_FIELDS:
        raise ValueError("Single-view recipe source_binding has unknown or missing fields")
    if (source.get("generator_path") != "bench/quality_ranker_synthetic.py" or
            source.get("preparation_path") != "scripts/cloud_ranker_prepare_single_view.py"):
        raise ValueError("Single-view recipe source paths differ from their frozen values")
    _checked_sha256(source.get("generator_sha256"), "generator_sha256")
    _checked_sha256(source.get("preparation_sha256"), "preparation_sha256")

    options = recipe.get("options")
    if not isinstance(options, dict) or set(options) != _RECIPE_OPTION_FIELDS:
        raise ValueError("Single-view recipe options have unknown or missing fields")
    expected_scalars = {"frames": 1, "width": 640, "height": 480,
                        "fps": 10.0, "seed": 0}
    for field, expected in expected_scalars.items():
        value = options.get(field)
        if isinstance(value, bool) or type(value) not in (int, float) or value != expected:
            raise ValueError(f"Single-view recipe {field} differs from its frozen value")
    if any(type(options[name]) is not int for name in ("frames", "width", "height", "seed")):
        raise ValueError("Single-view recipe frame, size and seed options must be integers")
    if type(options["fps"]) not in (int, float) or not math.isfinite(float(options["fps"])):
        raise ValueError("Single-view recipe fps must be finite")
    if options.get("mesh_units") != "metres" or type(options.get("mesh_units")) is not str:
        raise ValueError("Single-view recipe mesh_units must be metres")
    if type(options.get("background_rgb")) is not list or options["background_rgb"] != [127, 127, 127]:
        raise ValueError("Single-view recipe background_rgb must be [127,127,127]")
    expected_k = [[576.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]]
    if options.get("intrinsics") != expected_k:
        raise ValueError("Single-view recipe intrinsics differ from the frozen 640x480 camera")
    expected_trajectory = {
        "rotation_order": "RyRx", "frame_index": 0, "yaw0_deg": 75,
        "pitch_deg": -15, "yaw_sweep_deg": 20.0, "roll_deg": 0.0,
    }
    if options.get("trajectory") != expected_trajectory:
        raise ValueError("Single-view recipe trajectory differs from its frozen pose")
    if options.get("occluder") is not None:
        raise ValueError("Single-view recipe must not add an occluder")
    expected_policy = {"max_rotation_error_deg": 10.0,
                       "max_translation_error_m": 0.02,
                       "max_surface_error_m": None,
                       "max_projection_error_px": None}
    if options.get("label_policy") != expected_policy:
        raise ValueError("Single-view recipe label policy differs from the frozen thresholds")
    limits = options.get("limits")
    if (not isinstance(limits, dict) or set(limits) != {"max_output_bytes", "max_video_bytes"} or
            any(type(limits.get(name)) is not int for name in limits) or
            limits.get("max_output_bytes") != 16 * 1024**2 or
            limits.get("max_video_bytes") != 4 * 1024**2):
        raise ValueError("Single-view recipe limits differ from the frozen output bounds")
    _validate_nested_json_numbers(options, "single-view options")

    sequences = recipe.get("sequences")
    if type(sequences) is not list or len(sequences) != 1:
        raise ValueError("Single-view recipe must freeze exactly one sequence row")
    row = sequences[0]
    if not isinstance(row, dict) or set(row) != _RECIPE_SEQUENCE_FIELDS:
        raise ValueError("Single-view recipe sequence row has unknown or missing fields")
    _checked_sha256(row.get("mesh_sha256"), "sequence mesh_sha256")
    expected = _single_view_identity()
    for field, value in expected.items():
        if row.get(field) != value or (field == "observationally_ambiguous" and row[field] is not True):
            raise ValueError("Single-view recipe sequence identity differs from its frozen value")
    if row["mesh_sha256"] != SINGLE_VIEW_ASSET_PIN:
        raise ValueError("Single-view recipe mesh SHA-256 differs from its pinned HOT3D asset")
    _checked_sha256(recipe.get("recipe_sha256"), "recipe_sha256")
    unsigned = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    expected_digest = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if recipe["recipe_sha256"] != expected_digest:
        raise ValueError("Single-view recipe self-digest mismatch")


def validate_single_view_recipe(recipe_path, split_manifest_path, mesh_path, sequence_id, *,
                                frames, width, height, fps, mesh_units,
                                observationally_ambiguous):
    """Authenticate the strict one-frame HOT3D17 initialization recipe."""
    recipe_path = Path(recipe_path).resolve(strict=True)
    split_path = Path(split_manifest_path).resolve(strict=True)
    mesh_path = Path(mesh_path).resolve(strict=True)
    recipe_raw = recipe_path.read_bytes()
    if len(recipe_raw) > 16 * 1024 * 1024:
        raise ValueError("Single-view recipe exceeds the 16 MiB inventory bound")
    recipe = _read_unique_json_bytes(recipe_raw, name="single-view recipe")
    _validate_single_view_recipe_shape(recipe)

    source_root = Path(__file__).resolve().parents[1]
    preparation_path = source_root / "scripts" / "cloud_ranker_prepare_single_view.py"
    if (not preparation_path.is_file() or
            _file_sha256(Path(__file__).resolve()) != recipe["source_binding"]["generator_sha256"] or
            _file_sha256(preparation_path) != recipe["source_binding"]["preparation_sha256"]):
        raise ValueError("Single-view recipe executing source bytes differ from source_binding")

    split_raw = split_path.read_bytes()
    split = read_split_manifest(split_path)
    if recipe["split_binding"] != {
            "manifest_sha256": split["manifest_sha256"],
            "file_sha256": hashlib.sha256(split_raw).hexdigest()}:
        raise ValueError("Single-view recipe is bound to different frozen split bytes")
    if recipe["options"]["label_policy"] != split["label_policy"]:
        raise ValueError("Single-view recipe label policy differs from the frozen split")
    groups = split["groups"]
    if type(groups) is not list or len(groups) != 1:
        raise ValueError("Single-view recipe must exactly cover one frozen split group")
    row, group = recipe["sequences"][0], groups[0]
    if type(sequence_id) is not str or sequence_id != row["sequence_id"]:
        raise ValueError("Requested sequence_id must resolve to the single-view split group")
    for key in ("sequence_id", "physical_object_id", "scene_id", "split", "mesh_sha256"):
        if row[key] != group.get(key):
            raise ValueError(f"Single-view recipe sequence {key} differs from frozen split membership")
    if (group.get("provenance_kind") != "synthetic" or
            group.get("background_kind") != "procedural" or
            group.get("background_id") != "procedural:gray"):
        raise ValueError("Single-view recipe requires a procedural-gray synthetic group")
    if (group.get("frames") != [{"frame_id": 0, "source_frame_id": 0}] or
            group.get("frame_ids") != [0]):
        raise ValueError("Single-view split group must freeze only frame ID 0")
    if (type(frames) is not int or frames != 1 or type(width) is not int or width != 640 or
            type(height) is not int or height != 480 or isinstance(fps, bool) or
            type(fps) not in (int, float) or not math.isfinite(float(fps)) or float(fps) != 10.0 or
            mesh_units != "metres" or type(observationally_ambiguous) is not bool or
            observationally_ambiguous is not True):
        raise ValueError("Single-view generation options differ from the frozen recipe")
    if mesh_path.suffix.lower() != ".glb" or mesh_path.stat().st_size > MAX_MESH_BYTES:
        raise ValueError(f"A caller-supplied GLB of at most {MAX_MESH_BYTES} bytes is required")
    mesh_digest = _file_sha256(mesh_path)
    if mesh_digest != row["mesh_sha256"] or mesh_digest != group.get("mesh_sha256"):
        raise ValueError("Selected metric mesh bytes differ from the single-view recipe/split SHA-256")
    return {"recipe": recipe, "recipe_sha256": recipe["recipe_sha256"],
            "sequence": row, "group": group, "split": split,
            "mesh_sha256": mesh_digest}


def validate_recipe(recipe_path, split_manifest_path, mesh_path, sequence_id, *,
                    frames, width, height, fps, mesh_units,
                    observationally_ambiguous):
    """Authenticate a frozen recipe, its selected split group and metric mesh bytes."""
    from .quality_ranker_data import _file_sha256, read_split_manifest

    recipe_path = Path(recipe_path).resolve(strict=True)
    split_path = Path(split_manifest_path).resolve(strict=True)
    mesh_path = Path(mesh_path).resolve(strict=True)
    recipe_raw = recipe_path.read_bytes()
    if len(recipe_raw) > 16 * 1024 * 1024:
        raise ValueError("Synthetic recipe exceeds the 16 MiB inventory bound")
    recipe = _read_unique_json_bytes(recipe_raw, name="synthetic recipe")
    if isinstance(recipe, dict) and recipe.get("kind") == ORIENTATION_RECIPE_KIND:
        return validate_orientation_recipe(
            recipe_path, split_path, mesh_path, sequence_id, frames=frames,
            width=width, height=height, fps=fps, mesh_units=mesh_units,
            observationally_ambiguous=observationally_ambiguous)
    if isinstance(recipe, dict) and recipe.get("kind") == SINGLE_VIEW_RECIPE_KIND:
        return validate_single_view_recipe(
            recipe_path, split_path, mesh_path, sequence_id, frames=frames,
            width=width, height=height, fps=fps, mesh_units=mesh_units,
            observationally_ambiguous=observationally_ambiguous)
    _validate_recipe_shape(recipe)
    split_raw = split_path.read_bytes()
    split = read_split_manifest(split_path)
    split_file_sha = hashlib.sha256(split_raw).hexdigest()
    if recipe["split_binding"] != {"manifest_sha256": split["manifest_sha256"],
                                   "file_sha256": split_file_sha}:
        raise ValueError("Synthetic recipe is bound to different frozen split bytes")
    if recipe["options"]["label_policy"] != split["label_policy"]:
        raise ValueError("Synthetic recipe label policy differs from the frozen split")
    groups_by_id = {group["sequence_id"]: group for group in split["groups"]}
    rows_by_id = {row["sequence_id"]: row for row in recipe["sequences"]}
    if len(groups_by_id) != 16 or set(rows_by_id) != set(groups_by_id):
        raise ValueError("Synthetic recipe must exactly cover the 16 frozen split groups")
    for frozen_id, frozen_group in groups_by_id.items():
        frozen_row = rows_by_id[frozen_id]
        for key in ("sequence_id", "physical_object_id", "scene_id", "split", "mesh_sha256"):
            if frozen_row[key] != frozen_group.get(key):
                raise ValueError(f"Recipe sequence {key} differs from frozen split membership")
        if (frozen_group.get("provenance_kind") != "synthetic" or
                frozen_group.get("background_kind") != "procedural" or
                frozen_group.get("background_id") != "procedural:gray"):
            raise ValueError("Broader recipe inventory requires procedural-gray synthetic groups")
        expected_frames = [{"frame_id": ordinal, "source_frame_id": ordinal}
                           for ordinal in range(24)]
        if (frozen_group.get("frames") != expected_frames or
                frozen_group.get("frame_ids") != list(range(24))):
            raise ValueError("Broader recipe split groups must freeze frame IDs 0..23")
    groups = [group for group in split["groups"] if group["sequence_id"] == sequence_id]
    if len(groups) != 1:
        raise ValueError("Requested sequence_id must resolve to one validated frozen split group")
    group = groups[0]
    rows = [row for row in recipe["sequences"] if row["sequence_id"] == sequence_id]
    if len(rows) != 1:
        raise ValueError("Requested sequence is not uniquely frozen in the recipe")
    row = rows[0]
    for key in ("sequence_id", "physical_object_id", "scene_id", "split", "mesh_sha256"):
        if row[key] != group.get(key):
            raise ValueError(f"Recipe sequence {key} differs from validated split membership")
    if group.get("provenance_kind") != "synthetic" or group.get("background_kind") != "procedural" or group.get("background_id") != "procedural:gray":
        raise ValueError("Broader synthetic recipe requires its matching procedural-gray split group")
    expected_frames = [{"frame_id": ordinal, "source_frame_id": ordinal}
                       for ordinal in range(24)]
    if (group.get("frames") != expected_frames or
            group.get("frame_ids") != list(range(24))):
        raise ValueError("Broader synthetic split group must freeze frame IDs 0..23")
    if (type(frames) is not int or frames != recipe["options"]["frames"] or
            type(width) is not int or width != recipe["options"]["width"] or
            type(height) is not int or height != recipe["options"]["height"] or
            isinstance(fps, bool) or not isinstance(fps, (int, float)) or
            float(fps) != float(recipe["options"]["fps"]) or mesh_units != recipe["options"]["mesh_units"] or
            type(observationally_ambiguous) is not bool or
            observationally_ambiguous != row["observationally_ambiguous"]):
        raise ValueError("Synthetic generation options differ from the frozen recipe")
    if mesh_path.suffix.lower() != ".glb" or mesh_path.stat().st_size > MAX_MESH_BYTES:
        raise ValueError(f"A caller-supplied GLB of at most {MAX_MESH_BYTES} bytes is required")
    mesh_digest = _file_sha256(mesh_path)
    if mesh_digest != row["mesh_sha256"] or mesh_digest != group.get("mesh_sha256"):
        raise ValueError("Selected metric mesh bytes differ from the frozen recipe/split SHA-256")
    return {"recipe": recipe, "recipe_sha256": recipe["recipe_sha256"],
            "sequence": row, "group": group, "split": split,
            "mesh_sha256": mesh_digest}


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _broader_geometry(vertices_m, intrinsics=None, *, width=640, height=480):
    """Validate metric vertices and the frozen full-trajectory projection bounds."""
    vertices = np.asarray(vertices_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError("Broader synthetic geometry requires a nonempty finite Nx3 vertices_m array")
    matrix = np.asarray(intrinsics if intrinsics is not None else
                        [[576., 0., 319.5], [0., 576., 239.5], [0., 0., 1.]],
                        dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Broader synthetic intrinsics must be a finite 3x3 matrix")
    if (matrix[0, 0] <= 0. or matrix[1, 1] <= 0. or
            not np.array_equal(matrix[2], [0., 0., 1.]) or
            matrix[0, 1] != 0. or matrix[1, 0] != 0.):
        raise ValueError("Broader synthetic intrinsics must be zero-skew pinhole K")
    for row in (0, 1):
        if any(matrix[row, column] != 0. for column in range(3) if column not in (row, 2)):
            raise ValueError("Broader synthetic intrinsics must be zero-skew pinhole K")
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or
            not 0. <= matrix[0, 2] <= width - 1 or not 0. <= matrix[1, 2] <= height - 1):
        raise ValueError("Broader synthetic principal point must lie within native pixel bounds")
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        extent = upper - lower
        center = lower + extent * .5
        offsets = vertices - center
        squared = np.sum(offsets * offsets, axis=1)
        radius = float(np.sqrt(np.max(squared)))
    if (not np.isfinite(extent).all() or not np.isfinite(center).all() or
            not np.isfinite(squared).all() or not math.isfinite(radius) or radius <= 0.):
        raise ValueError("Broader synthetic metric geometry has overflow or degenerate radius")
    radius = float(radius)
    distance = max(4. * radius, radius + 576. * radius / (0.32 * 480.))
    if not math.isfinite(distance) or distance - radius <= 0.01 or distance + radius >= 30.:
        raise ValueError("Broader synthetic geometry exceeds the renderer clip bounds")
    bound_x = float(matrix[0, 0]) * radius / (distance - radius)
    bound_y = float(matrix[1, 1]) * radius / (distance - radius)
    if (not math.isfinite(bound_x) or not math.isfinite(bound_y) or
            bound_x > min(float(matrix[0, 2]), width - 1. - float(matrix[0, 2])) or
            bound_y > min(float(matrix[1, 2]), height - 1. - float(matrix[1, 2]))):
        raise ValueError("Broader synthetic enclosing-sphere projection leaves native pixel bounds")
    return center, radius, distance, matrix


def broader_trajectory_pose(frame_index, *, yaw0_deg, pitch_deg, center_m, radius_m):
    """Return one recipe-bound object-to-camera pose in metres."""
    if type(frame_index) is not int or not 0 <= frame_index < 24:
        raise ValueError("Broader trajectory frame_index must be in [0,23]")
    if type(yaw0_deg) is not int or yaw0_deg not in (0, 90, 180, 270):
        raise ValueError("Broader trajectory yaw origin must be 0,90,180 or 270 degrees")
    if type(pitch_deg) is not int or pitch_deg not in (-25, 25):
        raise ValueError("Broader trajectory pitch must be -25 or +25 degrees")
    center = np.asarray(center_m, dtype=np.float64)
    radius = float(radius_m)
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0.:
        raise ValueError("Broader trajectory requires a finite center and positive radius")
    yaw = math.radians(yaw0_deg + 40. * frame_index / 23.)
    pitch = math.radians(pitch_deg)
    rotation = _rotation_y(yaw) @ _rotation_x(pitch)
    distance = max(4. * radius, radius + 576. * radius / (0.32 * 480.))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.array([0., 0., distance]) - rotation @ center
    return pose


def validate_broader_trajectory(vertices_m, intrinsics=None, *, width=640, height=480):
    """Check SO(3), inverse and every actual vertex projection before rendering."""
    vertices = np.asarray(vertices_m, dtype=np.float64)
    center, radius, distance, matrix = _broader_geometry(
        vertices, intrinsics, width=width, height=height)
    poses = {}
    inverse_tolerance = 1e-10
    for yaw0 in (0, 90, 180, 270):
        for pitch in (-25, 25):
            for frame in range(24):
                pose = broader_trajectory_pose(frame, yaw0_deg=yaw0,
                                               pitch_deg=pitch,
                                               center_m=center, radius_m=radius)
                rotation = pose[:3, :3]
                if not np.isfinite(pose).all():
                    raise ValueError("Broader synthetic trajectory produced a nonfinite transform")
                try:
                    inverse = np.linalg.inv(pose)
                except np.linalg.LinAlgError as exc:
                    raise ValueError("Broader synthetic trajectory transform is unstable") from exc
                if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=inverse_tolerance, rtol=0.) or
                        abs(float(np.linalg.det(rotation)) - 1.) > inverse_tolerance or
                        not np.array_equal(pose[3], [0., 0., 0., 1.]) or
                        not np.isfinite(inverse).all() or
                        not np.allclose(pose @ inverse, np.eye(4), atol=inverse_tolerance, rtol=0.)):
                    raise ValueError("Broader synthetic trajectory produced an invalid rigid transform")
                camera_points = vertices @ rotation.T + pose[:3, 3]
                depth = camera_points[:, 2]
                if (not np.isfinite(camera_points).all() or
                        np.any(depth <= 0.01) or np.any(depth >= 30.)):
                    raise ValueError("Broader synthetic transformed vertices exceed clip bounds")
                pixels = camera_points @ matrix.T
                pixels = pixels[:, :2] / pixels[:, 2:3]
                if (not np.isfinite(pixels).all() or np.any(pixels[:, 0] < 0.) or
                        np.any(pixels[:, 0] > width - 1.) or np.any(pixels[:, 1] < 0.) or
                        np.any(pixels[:, 1] > height - 1.)):
                    raise ValueError("Broader synthetic transformed vertices project outside native pixels")
                poses[(yaw0, pitch, frame)] = pose
    return {"center_m": center, "radius_m": radius, "distance_m": distance,
            "poses": poses, "intrinsics": matrix}


def orientation_trajectory_pose(frame_index, *, yaw0_deg, pitch_deg, center_m, radius_m):
    """Return one fixed orientation-study object-to-camera pose in metres."""
    if type(frame_index) is not int or not 0 <= frame_index < 12:
        raise ValueError("Orientation trajectory frame_index must be in [0,11]")
    if (type(yaw0_deg) is not int or type(pitch_deg) is not int or
            (yaw0_deg, pitch_deg) not in (*_ORIENTATION_VIEW_PAIRS, _SINGLE_VIEW_PAIR)):
        raise ValueError("Orientation trajectory requires a frozen paired yaw/pitch view")
    try:
        center = np.asarray(center_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Orientation trajectory requires a finite center and positive radius") from exc
    if isinstance(radius_m, bool) or type(radius_m) not in (int, float, np.integer, np.floating):
        raise ValueError("Orientation trajectory requires a finite center and positive radius")
    radius = float(radius_m)
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0.:
        raise ValueError("Orientation trajectory requires a finite center and positive radius")
    yaw = math.radians(yaw0_deg + 20. * frame_index / 11.)
    pitch = math.radians(pitch_deg)
    rotation = _rotation_y(yaw) @ _rotation_x(pitch)
    distance = max(4. * radius, radius + 576. * radius / (0.32 * 480.))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.array([0., 0., distance]) - rotation @ center
    return pose


def _validate_orientation_pose(pose, vertices, matrix, *, width, height, name):
    """Validate one object-to-camera rigid transform and all vertex projections."""
    rotation = pose[:3, :3]
    tolerance = 1e-10
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} produced a nonfinite transform")
    try:
        inverse = np.linalg.inv(pose)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} transform is unstable") from exc
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=0.) or
            abs(float(np.linalg.det(rotation)) - 1.) > tolerance or
            not np.array_equal(pose[3], [0., 0., 0., 1.]) or
            not np.isfinite(inverse).all() or
            not np.allclose(pose @ inverse, np.eye(4), atol=tolerance, rtol=0.) or
            not np.allclose(inverse @ pose, np.eye(4), atol=tolerance, rtol=0.)):
        raise ValueError(f"{name} produced an invalid rigid transform")
    camera_points = vertices @ rotation.T + pose[:3, 3]
    depth = camera_points[:, 2]
    if (not np.isfinite(camera_points).all() or np.any(depth <= 0.01) or
            np.any(depth >= 30.)):
        raise ValueError(f"{name} transformed vertices exceed clip bounds")
    pixels = camera_points @ matrix.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    if (not np.isfinite(pixels).all() or np.any(pixels[:, 0] < 0.) or
            np.any(pixels[:, 0] > width - 1.) or np.any(pixels[:, 1] < 0.) or
            np.any(pixels[:, 1] > height - 1.)):
        raise ValueError(f"{name} transformed vertices project outside native pixels")
    return inverse


def validate_orientation_trajectory(vertices_m, intrinsics=None, *, width=640, height=480):
    """Validate all 24 fixed orientation poses against every supplied mesh vertex."""
    vertices = np.asarray(vertices_m, dtype=np.float64)
    center, radius, distance, matrix = _broader_geometry(
        vertices, intrinsics, width=width, height=height)
    poses = {}
    for yaw0, pitch in _ORIENTATION_VIEW_PAIRS:
        for frame in range(12):
            pose = orientation_trajectory_pose(
                frame, yaw0_deg=yaw0, pitch_deg=pitch,
                center_m=center, radius_m=radius)
            _validate_orientation_pose(
                pose, vertices, matrix, width=width, height=height,
                name="Orientation trajectory")
            poses[(yaw0, pitch, frame)] = pose
    return {"center_m": center, "radius_m": radius, "distance_m": distance,
            "poses": poses, "intrinsics": matrix}


def validate_single_view_trajectory(vertices_m, intrinsics=None, *, width=640, height=480):
    """Validate the one frozen single-view pose against every mesh vertex."""
    vertices = np.asarray(vertices_m, dtype=np.float64)
    center, radius, distance, matrix = _broader_geometry(
        vertices, intrinsics, width=width, height=height)
    yaw0, pitch = _SINGLE_VIEW_PAIR
    pose = orientation_trajectory_pose(
        0, yaw0_deg=yaw0, pitch_deg=pitch, center_m=center, radius_m=radius)
    _validate_orientation_pose(
        pose, vertices, matrix, width=width, height=height,
        name="Single-view trajectory")
    return {"center_m": center, "radius_m": radius, "distance_m": distance,
            "poses": {(yaw0, pitch, 0): pose}, "intrinsics": matrix}


def composite_broader_occluder(rgb, frame_index):
    """Paint the frozen opaque rectangle onto an RGB frame for ordinals 6..13."""
    image = np.asarray(rgb)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or
            type(frame_index) is not int or not 0 <= frame_index < 24):
        raise ValueError("Broader occluder requires a native uint8 RGB frame and ordinal 0..23")
    if not 6 <= frame_index <= 13:
        return image
    height, width = image.shape[:2]
    rect_width = int(math.floor(0.22 * width + 0.5))
    rect_height = int(math.floor(0.70 * height + 0.5))
    phase = (frame_index - 6) / 7.
    center_x = (0.20 + 0.60 * phase) * width
    center_y = 0.5 * height
    x0 = int(math.floor(center_x - rect_width / 2. + 0.5))
    y0 = int(math.floor(center_y - rect_height / 2. + 0.5))
    left, right = max(0, x0), min(width, x0 + rect_width)
    top, bottom = max(0, y0), min(height, y0 + rect_height)
    image[top:bottom, left:right] = (32, 32, 32)
    return image


NORMALIZED_VIEW_MODE = "normalized24-v1"
NORMALIZED_VIEW_SCHEDULE = tuple((elevation, azimuth)
                                 for elevation in (-60, 0, 60)
                                 for azimuth in range(0, 360, 45))


def _validate_view_options(view_mode, view_index, *, frames, width, height, fps,
                           observationally_ambiguous, recipe_path):
    if view_mode == "legacy":
        if view_index is not None:
            raise ValueError("--view-index requires normalized24-v1")
        return False
    if view_mode != NORMALIZED_VIEW_MODE:
        raise ValueError("Unknown synthetic view mode")
    if type(view_index) is not int or not 0 <= view_index < 24:
        raise ValueError("Normalized view index must be an integer in [0,23]")
    if (frames, width, height, fps) != (1, 640, 480, 10):
        raise ValueError("normalized24-v1 requires one 640x480 frame at 10 fps")
    if observationally_ambiguous is not True or recipe_path is not None:
        raise ValueError("normalized24-v1 requires ambiguous labels and no recipe")
    return True


def normalized_view_pose(view_index, *, center_m, radius_m):
    """Fixed elevation-major viewing schedule; mesh coordinates remain unchanged."""
    if type(view_index) is not int or not 0 <= view_index < 24:
        raise ValueError("Normalized view index must be an integer in [0,23]")
    center = np.asarray(center_m, dtype=np.float64)
    radius = float(radius_m)
    if (center.shape != (3,) or not np.isfinite(center).all() or
            not math.isfinite(radius) or radius <= 0.):
        raise ValueError("Normalized views require finite center and positive radius")
    elevation, azimuth = NORMALIZED_VIEW_SCHEDULE[view_index]
    rotation = _rotation_x(math.radians(elevation)) @ _rotation_y(math.radians(azimuth))
    distance = max(4. * radius, radius + 576. * radius / (0.32 * 480.))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.array([0., 0., distance]) - rotation @ center
    return pose


def validate_normalized_views(vertices_m, intrinsics=None, *, width=640, height=480):
    """Validate every fixed view before any selected view is rendered."""
    if (width, height) != (640, 480):
        raise ValueError("normalized24-v1 requires 640x480 native pixels")
    vertices = np.asarray(vertices_m, dtype=np.float64)
    center, radius, distance, matrix = _broader_geometry(
        vertices, intrinsics, width=width, height=height)
    poses = {}
    for index in range(24):
        pose = normalized_view_pose(index, center_m=center, radius_m=radius)
        _validate_orientation_pose(pose, vertices, matrix, width=width, height=height,
                                   name=f"normalized view {index}")
        poses[index] = pose
    return {"center_m": center, "radius_m": radius, "distance_m": distance,
            "intrinsics": matrix, "poses": poses}


def trajectory_pose(frame_index: int, frame_count: int) -> np.ndarray:
    """Analytic object-to-camera pose in metres for one deterministic frame."""
    if type(frame_count) is not int or not 1 <= frame_count <= MAX_FRAMES:
        raise ValueError(f"frame_count must be an integer in [1, {MAX_FRAMES}]")
    if type(frame_index) is not int or not 0 <= frame_index < frame_count:
        raise ValueError("frame_index must be an ordinal inside the declared trajectory")
    phase = 0. if frame_count == 1 else 2. * math.pi * frame_index / (frame_count - 1)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _rotation_z(.10 * math.sin(phase)) @ _rotation_y(.30 * math.sin(phase)) @ _rotation_x(.12 * math.cos(phase))
    pose[:3, 3] = (.055 * math.sin(phase), .025 * math.cos(phase), .62 + .025 * math.sin(2. * phase))
    return pose


def _validate_generation_options(frame_count, width, height, fps):
    if type(frame_count) is not int or not 1 <= frame_count <= MAX_FRAMES:
        raise ValueError(f"frames must be an integer in [1, {MAX_FRAMES}]")
    if (type(width) is not int or type(height) is not int or
            not 64 <= width <= MAX_DIMENSION or not 64 <= height <= MAX_DIMENSION):
        raise ValueError(f"width and height must be integers in [64, {MAX_DIMENSION}]")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or not 1 <= fps <= MAX_FPS:
        raise ValueError(f"fps must be finite and in [1, {MAX_FPS}]")


def make_input_manifest(*, mesh_sha256, video_sha256, frames, width, height,
                        fps, sequence_group, intrinsics=None):
    """Build and strictly self-validate the schema-2 inference-only bundle index."""
    _validate_generation_options(frames, width, height, fps)
    for digest in (mesh_sha256, video_sha256):
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest.lower())):
            raise ValueError("asset and video source digests must be SHA-256 hex strings")
    if not isinstance(sequence_group, dict):
        raise ValueError("A group from the pre-frozen split manifest is required")
    physical_id = sequence_group.get("physical_object_id")
    scene_id = sequence_group.get("scene_id")
    sequence_id = sequence_group.get("sequence_id")
    if sequence_group.get("mesh_sha256") != mesh_sha256.lower():
        raise ValueError("Frozen sequence must bind the exact metric mesh SHA-256 before rendering")
    if sequence_group.get("provenance_kind") != "synthetic":
        raise ValueError("Synthetic rendering requires a sequence frozen with synthetic provenance")
    if (sequence_group.get("background_kind") != "procedural" or
            sequence_group.get("background_id") != "procedural:gray"):
        raise ValueError("Synthetic split group must explicitly name the procedural gray background")
    frozen_frames = sequence_group.get("frames")
    if not isinstance(frozen_frames, list) or len(frozen_frames) != frames:
        raise ValueError("Requested render frame count differs from the pre-frozen sequence")
    source_ids = [row.get("source_frame_id") for row in frozen_frames]
    if source_ids != list(range(frames)):
        raise ValueError("Synthetic source frame IDs must be the contiguous video ordinals 0..N-1")
    if [row.get("frame_id") for row in frozen_frames] != source_ids:
        raise ValueError("Synthetic tracker frame IDs must match the source video ordinals")
    timeline = []
    for ordinal in range(frames):
        timestamp = ordinal / float(fps)
        timeline.append({"frame_id": ordinal, "source_frame_id": ordinal,
                         "source_timestamp": timestamp, "timestamp_s": timestamp,
                         "role": "scored"})
    if intrinsics is None:
        focal = float(max(width, height)) * .9
        intrinsics = [[focal, 0., (width - 1.) * .5],
                      [0., focal, (height - 1.) * .5],
                      [0., 0., 1.]]
    else:
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
                matrix[0, 0] <= 0. or matrix[1, 1] <= 0. or
                not np.array_equal(matrix[2], [0., 0., 1.]) or
                matrix[0, 1] != 0. or matrix[1, 0] != 0. or
                not 0. <= matrix[0, 2] <= width - 1 or
                not 0. <= matrix[1, 2] <= height - 1):
            raise ValueError("Explicit synthetic intrinsics must be a valid zero-skew pinhole matrix")
        intrinsics = matrix.tolist()
    return {
        "schema_version": 2,
        "object": "synthetic_" + mesh_sha256[:16].lower(),
        "object_id": "synthetic_" + mesh_sha256[:16].lower(),
        "physical_object_id": physical_id,
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "background_id": "procedural:gray",
        "background_kind": "procedural",
        "provenance_kind": "synthetic",
        "units": "metres",
        "video": "synthetic.mp4",
        "asset": "object.glb",
        "source_hashes": {"synthetic.mp4": video_sha256.lower(),
                          "object.glb": mesh_sha256.lower()},
        "intrinsics": intrinsics,
        "native_resolution": [width, height],
        "setup_frame_id": None,
        "frame_ids": list(range(frames)),
        "clock": {"mode": PHYSICAL, "units": "seconds", "source_units": "seconds",
                  "timestamp_source": "synthetic renderer trajectory ordinal/fps",
                  "nominal_fps": float(fps)},
        "timeline": timeline,
    }


def _cloud_guard(cloud_execution):
    if not cloud_execution or platform.system() != "Linux":
        raise RuntimeError("Synthetic video rendering requires an approved Linux cloud host and --cloud-execution")


def synthetic_pose_label(physical_object_id, sequence_id, frame_index, frame_count,
                         *, observationally_ambiguous):
    if type(observationally_ambiguous) is not bool:
        raise ValueError("Synthetic orientation observability must be explicitly true or false")
    return {"object_id": physical_object_id, "sequence_id": sequence_id,
            "frame_id": frame_index, "source_frame_id": frame_index,
            "pose": trajectory_pose(frame_index, frame_count).tolist(),
            "observationally_ambiguous": observationally_ambiguous,
            "label_kind": "synthetic_renderer_pose"}


def _render_bundle(mesh_path: Path, output_dir: Path, group: dict, *, frames: int,
                   width: int, height: int, fps: float, mesh_units: str,
                   observationally_ambiguous: bool, cloud_execution: bool,
                   recipe_path: Path | None = None,
                   split_manifest_path: Path | None = None,
                   view_mode: str = "legacy", view_index: int | None = None):
    _cloud_guard(cloud_execution)
    _validate_generation_options(frames, width, height, fps)
    normalized_run = _validate_view_options(
        view_mode, view_index, frames=frames, width=width, height=height, fps=fps,
        observationally_ambiguous=observationally_ambiguous, recipe_path=recipe_path)
    normalized_split = None
    normalized_split_sha = None
    normalized_source_sha = _file_sha256(Path(__file__)) if normalized_run else None
    if normalized_run:
        if split_manifest_path is None:
            raise ValueError("normalized24-v1 requires the pre-frozen split manifest")
        normalized_split_sha = _file_sha256(split_manifest_path)
        normalized_split = read_split_manifest(split_manifest_path)
        if [row for row in normalized_split["groups"]
                if row["sequence_id"] == group.get("sequence_id")] != [group]:
            raise ValueError("Normalized view group differs from frozen split membership")
    if mesh_units != "metres":
        raise ValueError("Synthetic rendering accepts only an explicitly declared metric GLB")
    if type(observationally_ambiguous) is not bool:
        raise ValueError("Synthetic orientation observability must be explicitly true or false")
    mesh_path = Path(mesh_path).resolve(strict=True)
    if mesh_path.suffix.lower() != ".glb" or mesh_path.stat().st_size > MAX_MESH_BYTES:
        raise ValueError(f"A caller-supplied GLB of at most {MAX_MESH_BYTES} bytes is required")
    recipe_context = None
    if recipe_path is not None:
        if split_manifest_path is None:
            raise ValueError("--recipe requires --split-manifest for split-byte authentication")
        recipe_context = validate_recipe(
            recipe_path, split_manifest_path, mesh_path, group["sequence_id"],
            frames=frames, width=width, height=height, fps=fps,
            mesh_units=mesh_units,
            observationally_ambiguous=observationally_ambiguous)
        if recipe_context["group"] != group:
            raise ValueError("Passed sequence group differs from validated split membership")
    mesh_sha = _file_sha256(mesh_path)
    if recipe_context is not None and mesh_sha != recipe_context["mesh_sha256"]:
        raise ValueError("Metric mesh changed after recipe validation")
    mesh_bytes = mesh_path.read_bytes()
    if hashlib.sha256(mesh_bytes).hexdigest() != mesh_sha:
        raise ValueError("Metric mesh changed while staging synthetic output")
    intrinsics = None if recipe_context is None else recipe_context["recipe"]["options"]["intrinsics"]
    manifest = make_input_manifest(mesh_sha256=mesh_sha, video_sha256="0" * 64,
                                   frames=frames, width=width, height=height,
                                   fps=fps, sequence_group=group, intrinsics=intrinsics)
    output_dir = Path(output_dir)
    recipe_run = recipe_context is not None
    staged_run = recipe_run or normalized_run
    claim_path = None
    stage_dir = None
    if staged_run:
        output_dir = output_dir.absolute()
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("Broader recipe output requires a fresh, non-symlink destination")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        claim_path = output_dir.with_name(output_dir.name + ".claim")
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            stage_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-",
                                              dir=output_dir.parent))
            work_dir = stage_dir / "bundle"
            work_dir.mkdir()
        except Exception:
            if stage_dir is not None and stage_dir.exists():
                try:
                    _write_json(stage_dir / "incomplete.json", {
                        "schema_version": 1, "status": "incomplete",
                        "sequence_id": group.get("sequence_id"),
                        "recipe_sha256": recipe_context["recipe_sha256"] if recipe_run else None,
                        "error_type": "StagingSetupError",
                        "error": "Could not create recipe bundle staging directory",
                        "partial_file_bytes": {}, "promoted": False,
                    })
                except Exception:
                    pass
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass
            raise
        max_output_bytes = (recipe_context["recipe"]["options"]["limits"]["max_output_bytes"]
                            if recipe_run else MAX_VIDEO_BYTES + MAX_MESH_BYTES)
        max_video_bytes = (recipe_context["recipe"]["options"]["limits"]["max_video_bytes"]
                           if recipe_run else MAX_VIDEO_BYTES)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise FileExistsError("Refusing to render into a non-empty output directory")
        work_dir = output_dir
        max_output_bytes = MAX_VIDEO_BYTES + MAX_MESH_BYTES
        max_video_bytes = MAX_VIDEO_BYTES
    asset_path = work_dir / "object.glb"
    video_path = work_dir / "synthetic.mp4"
    writer = None
    renderer = None
    try:
        _write_new(asset_path, mesh_bytes)
        if _file_sha256(asset_path) != mesh_sha:
            raise ValueError("Staged metric mesh digest changed")
        # Heavy renderer imports are lazy so split/pose/schema tests stay CPU-only.
        from .quality_gotrack import TexturedRenderer, upstream_path
        from .vision import cv2
        from .quality_time import parse_timing_manifest
        upstream_path()
        from utils import renderer_base, structs

        k = np.asarray(manifest["intrinsics"], dtype=np.float64)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps), (width, height), True)
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the bounded MP4 writer")
        labels = []
        renderer = TexturedRenderer(asset_path, manifest["object_id"], unlit=True)
        validated_trajectory = None
        selected_recipe_row = None
        recipe_kind = None
        if normalized_run:
            validated_trajectory = validate_normalized_views(
                renderer.vertices_m, k, width=width, height=height)
        if recipe_run:
            recipe_kind = recipe_context["recipe"]["kind"]
            if recipe_kind == ORIENTATION_RECIPE_KIND:
                validated_trajectory = validate_orientation_trajectory(
                    renderer.vertices_m, k, width=width, height=height)
            elif recipe_kind == SINGLE_VIEW_RECIPE_KIND:
                validated_trajectory = validate_single_view_trajectory(
                    renderer.vertices_m, k, width=width, height=height)
            else:
                validated_trajectory = validate_broader_trajectory(
                    renderer.vertices_m, k, width=width, height=height)
            selected_recipe_row = recipe_context["sequence"]
        for ordinal in range(frames):
            if normalized_run:
                camera_from_object = validated_trajectory["poses"][view_index]
            elif recipe_run:
                yaw0, pitch = selected_recipe_row["yaw0_deg"], selected_recipe_row["pitch_deg"]
                camera_from_object = validated_trajectory["poses"][(yaw0, pitch, ordinal)]
            else:
                camera_from_object = trajectory_pose(ordinal, frames)
            camera_to_object = np.linalg.inv(camera_from_object)
            camera_to_object[:3, 3] *= 1000.
            camera = structs.PinholePlaneCameraModel(
                width=width, height=height,
                f=(k[0, 0], k[1, 1]), c=(k[0, 2], k[1, 2]),
                T_world_from_eye=camera_to_object)
            rendered = renderer.render_object_model(manifest["object_id"], camera)
            color = np.asarray(rendered[renderer_base.RenderType.COLOR])
            mask = np.asarray(rendered[renderer_base.RenderType.MASK], dtype=bool)
            if (color.shape != (height, width, 3) or not np.isfinite(color).all() or
                    mask.shape != (height, width)):
                raise RuntimeError("TexturedRenderer returned malformed synthetic RGB/mask arrays")
            rgb = np.clip(np.rint(color * 255.), 0, 255).astype(np.uint8)
            # The geometric mask is used only to composite the declared gray
            # renderer background; it is never serialized into the input bundle.
            rgb[~mask] = (127, 127, 127)
            if staged_run:
                if recipe_kind == BROADER_RECIPE_KIND:
                    composite_broader_occluder(rgb, ordinal)
                labels.append({"object_id": manifest["physical_object_id"],
                    "sequence_id": manifest["sequence_id"], "frame_id": ordinal,
                    "source_frame_id": ordinal,
                    "timestamp_s": ordinal / 10.0,
                    "pose": camera_from_object.tolist(),
                    "observationally_ambiguous": True,
                    "label_kind": "synthetic_renderer_pose"})
            else:
                labels.append(synthetic_pose_label(
                    manifest["physical_object_id"], manifest["sequence_id"], ordinal, frames,
                    observationally_ambiguous=observationally_ambiguous))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            video_size = video_path.stat().st_size
            if video_size > max_video_bytes or asset_path.stat().st_size + video_size > max_output_bytes:
                raise ValueError("Synthetic output exceeded its frozen byte bound")
        writer.release()
        writer = None
        renderer.close()
        renderer = None
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError("Synthetic renderer produced an empty video")
        video_sha = _file_sha256(video_path)
        manifest = make_input_manifest(mesh_sha256=mesh_sha, video_sha256=video_sha,
                                       frames=frames, width=width, height=height,
                                       fps=fps, sequence_group=group, intrinsics=intrinsics)
        input_path = work_dir / "input.json"
        _write_json(input_path, manifest)
        parse_timing_manifest(input_path.read_bytes())
        if staged_run:
            poses_path = stage_dir / "poses.jsonl"
            labels_bytes = b"".join(
                _canonical_json_bytes(row) + b"\n" for row in labels)
            _write_new(poses_path, labels_bytes)
            provenance = {
                "schema_version": 1,
                "kind": "quality-ranker-synthetic-generation-v1",
                "recipe_kind": recipe_kind,
                "recipe_sha256": recipe_context["recipe_sha256"] if recipe_run else None,
                "split_manifest_sha256": (recipe_context["split"]["manifest_sha256"]
                                          if recipe_run else normalized_split["manifest_sha256"]),
                "split_file_sha256": (recipe_context["recipe"]["split_binding"]["file_sha256"]
                                      if recipe_run else normalized_split_sha),
                "sequence_id": group["sequence_id"],
                "physical_object_id": group["physical_object_id"],
                "split": group["split"],
                "mesh_sha256": mesh_sha,
                "video_sha256": video_sha,
                "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
                "inference_bundle_files": ["input.json", "object.glb", "synthetic.mp4"],
                "inference_bundle_relative_path": "bundle",
                "labels_relative_path": "poses.jsonl",
                "orientation_review_required_before_candidate_generation":
                    normalized_run or recipe_kind in (ORIENTATION_RECIPE_KIND, SINGLE_VIEW_RECIPE_KIND),
            }
            if normalized_run:
                if _file_sha256(split_manifest_path) != normalized_split_sha:
                    raise ValueError("Frozen normalized split changed during rendering")
                if _file_sha256(Path(__file__)) != normalized_source_sha:
                    raise ValueError("Normalized renderer source changed during rendering")
                provenance.update(view_mode=view_mode, view_index=view_index,
                    view_schedule=[list(row) for row in NORMALIZED_VIEW_SCHEDULE],
                    center_m=validated_trajectory["center_m"].tolist(),
                    radius_m=validated_trajectory["radius_m"],
                    distance_m=validated_trajectory["distance_m"],
                    source_sha256=normalized_source_sha,
                    input_sha256=_file_sha256(input_path))
            else:
                _write_json(stage_dir / "generation.json", provenance)
            bundle_files = sorted(path.name for path in work_dir.iterdir())
            if bundle_files != ["input.json", "object.glb", "synthetic.mp4"]:
                raise RuntimeError("Recipe inference bundle must contain exactly its three bound files")
            total_bytes = sum(path.stat().st_size for path in stage_dir.rglob("*") if path.is_file())
            if video_path.stat().st_size > max_video_bytes or total_bytes > max_output_bytes:
                raise ValueError("Synthetic output exceeded its frozen byte bound after writer release")
            if normalized_run:
                receipt_bytes = len(json.dumps(provenance, indent=2, sort_keys=True).encode()) + 1024
                if total_bytes + receipt_bytes > max_output_bytes:
                    raise ValueError("Normalized generation receipt exceeds output byte bound")
                _write_json(stage_dir / "generation.json", provenance)
            if output_dir.exists() or output_dir.is_symlink():
                raise FileExistsError("Broader recipe output was created during generation")
            os.replace(stage_dir, output_dir)
            final_input = output_dir / "bundle" / "input.json"
            final_labels = output_dir / "poses.jsonl"
            result = {"input": str(final_input), "labels": str(final_labels),
                      "generation": str(output_dir / "generation.json"),
                      "recipe_sha256": recipe_context["recipe_sha256"] if recipe_run else None,
                      "video_sha256": video_sha, "asset_sha256": mesh_sha,
                      "frames": frames, "labels_in_inference_manifest": False}
        else:
            poses_path = output_dir / "poses.jsonl"
            _write_new(poses_path, b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                for row in labels))
            result = {"input": str(input_path), "labels": str(poses_path),
                      "video_sha256": video_sha, "asset_sha256": mesh_sha,
                      "frames": frames, "labels_in_inference_manifest": False}
        return result
    except Exception as exc:
        # Release the writer before measuring retained partial bytes. A failed
        # codec flush must not mask the original exception or skip renderer close.
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
            writer = None
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
            renderer = None
        if staged_run and stage_dir is not None and stage_dir.exists():
            partial = sorted({str(path.relative_to(stage_dir)).replace("\\", "/"):
                              path.stat().st_size for path in stage_dir.rglob("*")
                              if path.is_file()}.items())
            try:
                _write_json(stage_dir / "incomplete.json", {
                    "schema_version": 1, "status": "incomplete",
                    "sequence_id": group.get("sequence_id"),
                    "recipe_sha256": recipe_context["recipe_sha256"] if recipe_run else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:512], "partial_file_bytes": dict(partial),
                    "promoted": False,
                })
            except Exception:
                pass
        raise
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
        if claim_path is not None:
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True,
                        help="caller-supplied metric GLB; no asset download or conversion")
    parser.add_argument("--split-manifest", type=Path, required=True,
                        help="frozen with the synthetic sequence identities before rendering")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path,
                        help="opt-in frozen v1 recipe; without this flag legacy trajectory behavior is retained")
    parser.add_argument("--view-mode", choices=("legacy", NORMALIZED_VIEW_MODE), default="legacy",
                        help="normalized24-v1 centers the camera on any supported metric mesh")
    parser.add_argument("--view-index", type=int,
                        help="fixed schedule index 0..23; use separate pre-frozen one-frame sequences")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--mesh-units", choices=("metres",), required=True,
                        help="explicitly declare that the caller-supplied GLB uses metres")
    parser.add_argument("--observationally-ambiguous", choices=("yes", "no"), required=True,
                        help="explicit orientation policy for labels on possibly symmetric geometry")
    parser.add_argument("--cloud-execution", action="store_true",
                        help="required explicit assertion and Linux-host guard before renderer imports")
    args = parser.parse_args(argv)
    split_manifest = read_split_manifest(args.split_manifest)
    groups = [row for row in split_manifest["groups"] if row["sequence_id"] == args.sequence_id]
    if len(groups) != 1:
        raise ValueError("--sequence-id must resolve to exactly one group in the frozen manifest")
    result = _render_bundle(args.mesh, args.output_dir, groups[0], frames=args.frames,
                            width=args.width, height=args.height, fps=args.fps,
                            mesh_units=args.mesh_units,
                            observationally_ambiguous=args.observationally_ambiguous == "yes",
                            cloud_execution=args.cloud_execution,
                            recipe_path=args.recipe,
                            split_manifest_path=args.split_manifest,
                            view_mode=args.view_mode, view_index=args.view_index)
    print(json.dumps({"status": "rendered", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
