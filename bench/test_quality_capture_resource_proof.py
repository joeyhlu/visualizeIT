"""Independent, synthetic-only tests for capture renderer-proof binding.

Every bundle, metric asset, renderer proof, sidecar and execution receipt in this
module is created below a temporary directory.  The tests never issue the real
canonical asset receipt, deserialize a checkpoint, open a renderer, or execute a
model.  Passing these tests verifies receipt semantics and key plumbing only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from . import quality_assets as assets
from . import quality_capture as capture
from . import test_quality_capture_protocol as protocol
from .test_quality_capture_foundation import _fake_resource_tree


_ROOT = Path(__file__).resolve().parents[1]
_OBJECT_MIN = (-0.05, 0.01, -0.06)
_OBJECT_EXTENTS = (0.10, 0.20, 0.30)
_OBJECT_MAX = tuple(a + b for a, b in zip(_OBJECT_MIN, _OBJECT_EXTENTS))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _label_sha(label: str) -> str:
    return _sha(("synthetic-render-proof-fixture|" + label).encode("ascii"))


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _sha_record(label: str, shape, dtype: str):
    return {"dtype": dtype, "order": "C", "sha256": _label_sha(label),
            "shape": list(shape)}


def _source_path(relative: str) -> Path:
    """Resolve the validator's explicit repo-relative geometry source slice."""
    normalized = relative.replace("\\", "/")
    return Path(assets.ROOT).joinpath(*normalized.split("/"))


def _proof_source_hashes():
    paths = tuple(assets._CAPTURE_RENDERER_SOURCE_PATHS)
    return {relative: _sha(_source_path(relative).read_bytes()) for relative in paths}


def _source_records(source_hashes):
    return {
        relative: {"bytes": _source_path(relative).stat().st_size,
                   "sha256": source_hashes[relative]}
        for relative in source_hashes
    }


def _source_key(source_hashes, suffix: str) -> str:
    matches = [key for key in source_hashes if key.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"Expected one proof source ending in {suffix!r}; got {matches!r}")
    return matches[0]


def _source_identities(source_hashes):
    identities = {}
    for name, suffix, module in (
        ("pyrender_offscreen", "pyrender/offscreen.py", "pyrender.offscreen"),
        ("pyrender_renderer", "pyrender/renderer.py", "pyrender.renderer"),
        ("quality_gotrack", "bench/quality_gotrack.py", "bench.quality_gotrack"),
        ("quality_render_stability", "bench/quality_render_stability.py",
         "bench.quality_render_stability"),
    ):
        key = _source_key(source_hashes, suffix)
        identities[name] = {"file": str(_source_path(key).resolve()),
                            "module": module, "sha256": source_hashes[key]}
    return identities


def _allocation_attachment(dimensions, role: str, generation: int = 1):
    width, height = dimensions
    if role == "color":
        format_name = "GL_RGBA"
    else:
        format_name = "GL_DEPTH_COMPONENT24"
    return {"format_name": format_name, "height": height,
            "ordinary_storage_delegated": True,
            "renderbuffer_id": 2 * generation + (1 if role == "color" else 2),
            "samples": 4, "success": True, "target_name": "GL_RENDERBUFFER",
            "width": width}


def _allocation_pair(dimensions, generation: int = 1):
    return {"color": _allocation_attachment(dimensions, "color", generation),
            "depth": _allocation_attachment(dimensions, "depth", generation),
            "dimensions": list(dimensions), "passed": True}


def _framebuffer_fields(dimensions):
    return {"multisample_dimensions": list(dimensions),
            "multisample_draw_fbo": 2, "single_sample_read_fbo": 1}


def _gpu_metadata(dimensions, generation: int, source_hashes,
                  *, offscreen_identity: int | None = None, allocation_calls=None):
    width, height = dimensions
    pair = _allocation_pair(dimensions, generation)
    identities = _source_identities(source_hashes)
    return {
        "current_context_released": True,
        "dimension_match": True,
        "expected_dimensions": list(dimensions),
        "external_allocation_hook_used": False,
        "framebuffer_bindings_before_query": {"draw": 1, "read": 1},
        "framebuffer_bindings_restored": True,
        "framebuffer_complete": True,
        "framebuffer_fields": _framebuffer_fields(dimensions),
        "framebuffer_status": 36053,
        "gl_sample_buffers": 0,
        "gl_samples": 0,
        "multisample_enabled_state": True,
        "production_policy_metadata": {
            "allocation_calls": list(allocation_calls or ()),
            "allocation_generation": generation,
            "allocation_pair": pair,
            "color_storage_policy": "ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage",
            "coordinate_mode": "integer_centers_v1",
            "current_context_released": True,
            "depth_storage_policy": "ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage",
            "depth_units": "millimetres",
            "dimension_match": True,
            "dimensions": list(dimensions),
            "framebuffer_bindings_restored": True,
            "framebuffer_complete": True,
            "framebuffer_fields": _framebuffer_fields(dimensions),
            "gl_sample_buffers": 0,
            "gl_samples": 0,
            "offscreen_identity": generation if offscreen_identity is None else offscreen_identity,
            "render_policy": "capture_zero_sample_v1",
            "schema_version": 1,
            "source_identities": identities,
        },
        "query_scope": "fresh just-rendered offscreen draw framebuffer; observational only",
        "sample_positions": [],
        "sample_positions_status": "single_sample_no_multisample_positions",
        "status": "queried",
        "zero_sample_verified": True,
    }


def _sample_pixel_rows(camera, dimensions, *, depth_error_m=0.000001):
    width, height = dimensions
    fx, fy, cx, cy = (camera[name] for name in ("fx", "fy", "cx", "cy"))
    pixels = ((width // 3, height // 3), (2 * width // 3, height // 3),
              (width // 3, 2 * height // 3), (2 * width // 3, 2 * height // 3),
              (width // 2, height // 2), (width // 3, height // 2),
              (2 * width // 3, height // 2), (width // 2, height // 3))
    rows = []
    for index, (u, v) in enumerate(pixels):
        ray_x, ray_y = (u - cx) / fx, (v - cy) / fy
        expected = 1.0 / (1.0 - 0.25 * ray_x + 0.15 * ray_y)
        actual = expected + depth_error_m * (0.75 + 0.05 * index)
        rows.append({"absolute_error_m": abs(actual - expected), "actual_m": actual,
                     "expected_m": expected, "pixel": [u, v]})
    return rows


def _viewport_rows(source_hashes):
    dimensions_sequence = ((280, 280), (280, 280), (720, 720),
                           (1120, 1120), (280, 280), (280, 280))
    generations = (2, 2, 3, 4, 5, 6)
    reuse = (True, True, False, False, False, False)
    records = []
    for index, (dimensions, generation, reused) in enumerate(
            zip(dimensions_sequence, generations, reuse)):
        width, height = dimensions
        camera_scale = width / 720.0
        camera = {
            "cx": (338.75 + 0.5) * camera_scale - 0.5,
            "cy": (352.125 + 0.5) * camera_scale - 0.5,
            "fx": 367.5 * camera_scale,
            "fy": 391.25 * camera_scale,
        }
        rows = _sample_pixel_rows(camera, dimensions)
        allocation_calls = ([] if reused else [
            _allocation_attachment(dimensions, "color", generation),
            _allocation_attachment(dimensions, "depth", generation),
        ])
        records.append({
            "allocation_generation": generation,
            "camera": camera,
            "close_before_render": index == 5,
            "color_hash": _sha_record(f"viewport-{index}-rgb", (height, width, 3), "float32"),
            "depth_gate_passed": True,
            "depth_hash": _sha_record(f"viewport-{index}-depth", (height, width), "float32"),
            "dimensions": list(dimensions),
            "gpu_sampling_metadata": _gpu_metadata(
                dimensions, generation, source_hashes,
                offscreen_identity=(202 if index < 2 else generation + 200),
                allocation_calls=allocation_calls),
            "mask_hash": _sha_record(f"viewport-{index}-mask", (height, width), "bool"),
            "max_absolute_depth_error_m": max(row["absolute_error_m"] for row in rows),
            "rows": rows,
            "same_frame_mask_depth_agreement": True,
            "same_size_reuse": reused,
        })
    return records


def _identity_matrix3():
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _multiply3(a, b):
    return tuple(tuple(sum(a[r][k] * b[k][c] for k in range(3))
                       for c in range(3)) for r in range(3))


def _rotation_matrix(rx: float, ry: float, rz: float):
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rxm = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    rym = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rzm = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _multiply3(_multiply3(rzm, rym), rxm)


def _homogeneous(rotation, translation):
    return [list(rotation[0]) + [translation[0]],
            list(rotation[1]) + [translation[1]],
            list(rotation[2]) + [translation[2]],
            [0.0, 0.0, 0.0, 1.0]]


def _inverse_rigid(rotation, translation):
    inverse_rotation = tuple(tuple(rotation[c][r] for c in range(3)) for r in range(3))
    inverse_translation = tuple(-sum(inverse_rotation[r][c] * translation[c]
                                     for c in range(3)) for r in range(3))
    return inverse_rotation, inverse_translation


def _object_control_rows(asset_sha256: str):
    angles = ((0.11, -0.17, 0.23), (-0.21, 0.29, -0.13), (0.31, 0.19, 0.37))
    result = []
    for index, (rx, ry, rz) in enumerate(angles):
        rotation = _rotation_matrix(rx, ry, rz)
        translation = (0.01 * (index - 1), -0.015 * index, 1.0 + 0.1 * index)
        inverse_rotation, inverse_translation = _inverse_rigid(rotation, translation)
        inverse_mm = tuple(value * 1000.0 for value in inverse_translation)
        inverse_pose = _homogeneous(inverse_rotation, inverse_translation)
        renderer_pose = _homogeneous(inverse_rotation, inverse_mm)
        result.append({
            "T_camera_from_object_m": _homogeneous(rotation, translation),
            "T_object_from_cv_camera_m": inverse_pose,
            "T_object_from_cv_camera_renderer_input_mm": renderer_pose,
            "arrays": {
                "cpu_depth_m": _sha_record(f"control-{index}-cpu-depth", (280, 280), "float64"),
                "cpu_mask": _sha_record(f"control-{index}-cpu-mask", (280, 280), "bool"),
                "cpu_rgb": _sha_record(f"control-{index}-cpu-rgb", (280, 280, 3), "uint8"),
                "depth_abs_error_m": _sha_record(f"control-{index}-error", (280, 280), "float64"),
                "textured_depth_m": _sha_record(f"control-{index}-textured-depth-m", (280, 280), "float64"),
                "textured_depth_mm": _sha_record(f"control-{index}-textured-depth-mm", (280, 280), "float32"),
                "textured_mask": _sha_record(f"control-{index}-textured-mask", (280, 280), "bool"),
                "textured_rgb": _sha_record(f"control-{index}-textured-rgb", (280, 280, 3), "uint8"),
            },
        "camera_center_object_m": list(inverse_translation),
            "camera_space_bounds_max_m": [0.1, 0.2, 1.5],
            "camera_space_bounds_min_m": [-0.1, -0.2, 0.8],
            "camera_space_min_z_m": 0.8,
            "camera_target_center_m": [0.0, 0.0, 1.0],
            "camera_transform_roundtrip_passed": True,
            "common_depth_pixel_count": 500,
            "cpu_camera_principal_point": [140.0, 140.0],
            "cv_to_gl_axis_conversion_passed": True,
            "depth_gate_passed": True,
            "mask_gate_passed": True,
            "median_common_absolute_depth_error_m": 0.000001,
            "nonsymmetric_orientation": True,
            "output_camera": {"cx": 139.5, "cy": 139.5, "fx": 140.0, "fy": 140.0,
                              "height": 280, "width": 280},
            "proper_rotation": True,
            "pyrender_intrinsics_center_after_boundary_shift": [140.0, 140.0],
            "recipe": f"front-oblique-{chr(ord('a') + index)}",
            "recipe_parameters": {"center_camera_extent": [0.0, 0.0, 3.0 + index],
                                  "name": f"front-oblique-{chr(ord('a') + index)}",
                                  "rx_deg": math.degrees(rx), "ry_deg": math.degrees(ry),
                                  "rz_deg": math.degrees(rz)},
            "renderer_mm_to_m_full_pose_passed": True,
            "renderer_translation_input_mm": list(inverse_mm),
            "textured_mask_iou": 0.98,
            "textured_renderer_boundary_coordinate_mode": "integer_centers_v1",
            "textured_renderer_input_intrinsics": {"cx": 139.5, "cy": 139.5,
                                                   "fx": 140.0, "fy": 140.0},
        })
    return result


def _synthetic_metric_asset_document(bundle, *, asset_bytes=None):
    if asset_bytes is not None:
        bundle.write(bundle.asset_path, asset_bytes)
    bundle.manifest["object"] = "hot3d_obj_000008"
    bundle.manifest["object_id"] = 8
    asset_sha = _sha(bundle.path(bundle.asset_path).read_bytes())
    source_hashes = _proof_source_hashes()
    catalog_sha = _label_sha("object-8-catalog")
    geometry_sha = _label_sha("independent-geometry-receipt")
    receipt = {
        "schema_version": 1,
        "kind": "quality-capture-metric-asset-v1",
        "object_id": 8,
        "object_name": "hot3d_obj_000008",
        "asset_sha256": asset_sha,
        "original_asset_sha256": asset_sha,
        "source_units": "metres",
        "conversion_to_metres": 1.0,
        "unit_receipt_sha256": _label_sha("synthetic-unit-receipt"),
        "official_unit_evidence": {
            "identifier": "synthetic-test-fixture-only",
            "sha256": _label_sha("synthetic-official-unit-reference"),
        },
        "models_info": {
            "sha256": catalog_sha,
            "object_bounds_m": {
                "min": list(_OBJECT_MIN), "extents": list(_OBJECT_EXTENTS),
                "diameter": math.sqrt(sum(value * value for value in _OBJECT_EXTENTS)),
            },
        },
        "post_node_bounds_m": {
            "min": list(_OBJECT_MIN), "max": list(_OBJECT_MAX),
            "extents": list(_OBJECT_EXTENTS),
        },
        "object_frame": {
            "origin": "original glTF model origin; no recentering",
            "axes": "glTF right-handed +Y up; camera-from-object explicitly converted at consumers",
        },
        "loaders": [
            {"name": "custom_glb",
             "source_sha256": source_hashes[_source_key(source_hashes, "bench/glb_model.py")]},
            {"name": "trimesh_gltf",
             "source_sha256": source_hashes[_source_key(source_hashes, "trimesh/exchange/gltf.py")]},
        ],
        "geometry_receipt_sha256": geometry_sha,
        # The proof digest is filled after proof serialization.  The proof
        # itself binds only the asset/catalog/geometry identities, so there is
        # no circular digest dependency.
        "renderer_receipt_sha256": _label_sha("pending-synthetic-proof-digest"),
    }
    return receipt


def _synthetic_metric_receipt(bundle, proof_bytes: bytes, *, asset_bytes=None,
                              asset_document=None):
    receipt = (copy.deepcopy(asset_document) if asset_document is not None else
               _synthetic_metric_asset_document(bundle, asset_bytes=asset_bytes))
    receipt["renderer_receipt_sha256"] = _sha(proof_bytes)
    bundle.write(bundle.receipt_path, _canonical(receipt))
    bundle.write("assets/renderer-proof.json", proof_bytes)
    bundle.seal()
    bundle.write("input.json", bundle.input_bytes)
    return receipt


def _plane_samples(camera, *, depth_error_m=0.000001):
    width, height = camera["width"], camera["height"]
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    pixels = ((8, 11), (55, 11), (8, 41), (55, 41),
              (17, 29), (43, 13), (31, 36), (15, 17))
    rows = []
    errors = (depth_error_m, depth_error_m * 1.2, depth_error_m * 0.8,
              depth_error_m * 1.1, depth_error_m * 0.9, depth_error_m * 1.3,
              depth_error_m * 0.7, depth_error_m * 1.05)
    for index, ((u, v), error_m) in enumerate(zip(pixels, errors)):
        ray_x, ray_y = (u - cx) / fx, (v - cy) / fy
        denominator = 1.0 - 0.25 * ray_x + 0.15 * ray_y
        expected_z = 1.0 / denominator
        expected_x, expected_y = ray_x * expected_z, ray_y * expected_z
        observed_m = expected_z + error_m
        raw_mm = observed_m * 1000.0
        measured_m = raw_mm / 1000.0
        legacy_x, legacy_y = (u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy
        legacy_z = 1.0 / (1.0 - 0.25 * legacy_x + 0.15 * legacy_y)
        legacy_error = abs(measured_m - legacy_z)
        reprojected = (expected_x / expected_z * fx + cx,
                       expected_y / expected_z * fy + cy)
        reprojection_error = max(abs(reprojected[0] - u), abs(reprojected[1] - v))
        rows.append({
            "absolute_depth_error_m": abs(measured_m - expected_z),
            "expected_camera_point_m": [expected_x, expected_y, expected_z],
            "expected_raw_depth_m": expected_z,
            "integer_center_ray_xy": [ray_x, ray_y],
            "inverse_center_depth_A": 1.0 / expected_z,
            "legacy_unshifted_counterfactual_depth_m": legacy_z,
            "legacy_unshifted_counterfactual_error_m": legacy_error,
            "observed_gl_sample_positions_and_predictions": "",
            "pixel_integer_center": [u, v],
            "renderer_depth_m_after_single_mm_conversion": measured_m,
            "renderer_depth_raw_mm": raw_mm,
            "renderer_depth_raw_repr": format(raw_mm, ".17g"),
            "renderer_depth_sample_valid": True,
            "reprojected_pixel": list(reprojected),
            "reprojection_error_pixels": reprojection_error,
            "sample_label": f"integer-center-{index:02d}-u{u}-v{v}",
            "signed_depth_error_m_actual_minus_center": measured_m - expected_z,
            "t_observed_A_minus_inverse_actual_depth": denominator - 1.0 / measured_m,
        })
    return rows


def _legacy_baseline_gpu_metadata():
    sample_coordinates = ((0.13, 0.81), (0.42, 0.68),
                          (0.73, 0.29), (0.91, 0.54))
    sample_positions = []
    for index, (sample_x, sample_y) in enumerate(sample_coordinates):
        sample_positions.append({
            "sample_index": index,
            "label": f"gl-sample-{index}",
            "sample_x_from_left": sample_x,
            "sample_y_from_bottom": sample_y,
            "du_from_pixel_center_x": sample_x - 0.5,
            "dv_top_down_from_pixel_center_y": 0.5 - sample_y,
        })
    return {
        "status": "queried",
        "sample_positions_status": "queried",
        "gl_samples": 4,
        "gl_sample_buffers": 1,
        "framebuffers_supported": True,
        "framebuffer_bindings_restored": True,
        "framebuffer_fields": {
            "multisample_draw_fbo": 2,
            "single_sample_read_fbo": 1,
            "multisample_dimensions": [64, 48],
        },
        "draw_framebuffer_bound_for_sample_query": 2,
        "framebuffer_bindings_before_query": {"draw": 1, "read": 1},
        "multisample_enabled_state": True,
        "sample_positions": sample_positions,
    }


def _synthetic_plane_report(source_hashes, *, mode: str, passed: bool):
    camera = {"cx": 23.25, "cy": 18.75, "fx": 110.0, "fy": 130.0,
              "height": 48, "width": 64}
    metadata = (_gpu_metadata((64, 48), 1, source_hashes, offscreen_identity=101)
                if passed else _legacy_baseline_gpu_metadata())
    rows = _plane_samples(camera, depth_error_m=0.000009 if passed else 0.0012)
    maximum_error = max(row["absolute_depth_error_m"] for row in rows)
    maximum_reprojection = max(row["reprojection_error_pixels"] for row in rows)
    minimum_legacy_error = min(row["legacy_unshifted_counterfactual_error_m"] for row in rows)
    status = ("public_renderer_verified_stage_callers_not_integrated" if passed
              else "proposed_only_not_integrated")
    render_policy = "capture_zero_sample_v1" if passed else "legacy_v1"
    if passed:
        policy = metadata["production_policy_metadata"]
        policy["render_policy"] = render_policy
    runtime_sources = {
        relative: source_hashes[relative]
        for relative in assets._CAPTURE_RENDERER_IDENTITY_PATHS.values()
    }
    candidate_calls = [
        {**_allocation_attachment((64, 48), "color"), "height": 48, "width": 64},
        {**_allocation_attachment((64, 48), "depth"), "height": 48, "width": 64},
    ]
    if passed:
        policy["allocation_calls"] = candidate_calls
    report = {
        "all_sample_depths_valid": True,
        "attachment_storage_policy": "ordinary_color_depth_storage" if passed else "pinned_default_renderer_storage",
        "camera": camera,
        "camera_pose_cv_to_gl": "T_world_from_eye_cv @ diag(1,-1,-1,1); identity CV fixture pose",
        "capture_policy_status": status,
        "coordinate_mode": "integer_centers_v1",
        "depth_array_m_after_conversion": _sha_record(f"plane-{mode}-metres", (48, 64), "float64"),
        "depth_gate_passed": passed,
        "depth_units": {"conversion_count": 1, "expected": "metres", "renderer_output": "millimetres"},
        "disable_multisampling_option": mode == "proposed_msaa_rasterization_disabled",
        "gpu_sampling_metadata": metadata,
        "kind": "capture_mode_textured_renderer_paired_slanted_plane_depth_control",
        "legacy_unshifted_negative_control_passed": True,
        "max_absolute_depth_error_m": maximum_error,
        "maximum_accepted_depth_error_m": 0.00003,
        "maximum_reprojection_error_pixels": maximum_reprojection,
        "minimum_legacy_distinguishing_error_m": 0.0002,
        "minimum_legacy_unshifted_counterfactual_error_m": minimum_legacy_error,
        "mode": mode,
        "model_only_control": True,
        "observed_foreground": False,
        "plane": {"equation": "z = 1 + 0.25*x - 0.15*y", "slope_x": 0.25, "slope_y": -0.15},
        "production_render_policy": render_policy,
        "rasterization_description": "synthetic ordinary single-sample attachments" if passed else "synthetic four-sample baseline",
        "raw_depth_array_mm": _sha_record(f"plane-{mode}-millimetres", (48, 64), "float32"),
        "reprojection_gate_passed": True,
        "rows": rows,
        "sample_count": 8,
        "source_sha256": runtime_sources,
        "top_down_readback": {"source_pin": _source_key(source_hashes, "pyrender/renderer.py"),
                              "source_sha256": source_hashes[_source_key(source_hashes, "pyrender/renderer.py")],
                              "used_array_index": "raw_depth[q_y, q_x]"},
        "tracking_accuracy_claim": False,
    }
    if passed:
        report["external_allocation_hook_used"] = False
        report["zero_sample_storage_allocation_calls"] = candidate_calls
    return report


def _zero_sample_control(source_hashes):
    generations = []
    allocation_calls = []
    observed = ((64, 48), (280, 280), (720, 720), (1120, 1120), (280, 280), (280, 280))
    for generation, dimensions in enumerate(observed, start=1):
        pair = _allocation_pair(dimensions, generation)
        generations.append({"generation": generation, "pair": pair})
        allocation_calls.extend([pair["color"], pair["depth"]])
    plane_pairs = [_allocation_pair((64, 48), 1)]
    initial_pairs = [_allocation_pair((64, 48), 1), _allocation_pair((280, 280), 2)]
    all_dimensions = {
        "allocation_generations": generations,
        "initial_plane_and_object_pairs": {
            "expected_dimensions": [[64, 48], [280, 280]],
            "pairs": initial_pairs,
            "passed": True,
        },
        "passed": True,
        "scope": "every observed allocation generation independently verified",
    }
    return {
        "all_dimensions_allocation_pair_validation": all_dimensions,
        "allocation_calls": allocation_calls,
        "object_controls_result_summary": {"metadata_count": 3, "recipe_count": 3},
        "plane_allocation_pair_validation": {
            "expected_dimensions": [[64, 48]], "pairs": plane_pairs, "passed": True,
        },
    }


def _synthetic_renderer_proof(asset_receipt, source_hashes=None):
    source_hashes = _proof_source_hashes() if source_hashes is None else dict(source_hashes)
    asset_document = asset_receipt
    asset_sha = asset_document["asset_sha256"]
    catalog_sha = asset_document["models_info"]["sha256"]
    geometry_sha = asset_document["geometry_receipt_sha256"]
    bounds = asset_document["models_info"]["object_bounds_m"]
    minimum, extents = bounds["min"], bounds["extents"]
    maximum = [a + b for a, b in zip(minimum, extents)]
    size = list(extents)
    signature = _label_sha("independent-surface-position-uv-signature")
    uv_hash = _label_sha("synthetic-uv-source")
    plane_runs = {
        "default_msaa_enabled": _synthetic_plane_report(
            source_hashes, mode="default_msaa_enabled", passed=False),
        "proposed_msaa_rasterization_disabled": _synthetic_plane_report(
            source_hashes, mode="proposed_msaa_rasterization_disabled", passed=False),
        "verified_zero_sample_attachments": _synthetic_plane_report(
            source_hashes, mode="verified_zero_sample_attachments", passed=True),
    }
    viewport = _viewport_rows(source_hashes)
    zero = _zero_sample_control(source_hashes)
    camera_checks = {
        "axis_projection_pixels": [[139.5, 139.5], [279.5, 139.5], [139.5, 279.5]],
        "center_sample_ray_xy": [-139.5 / 140.0, -139.5 / 140.0],
        "cpu_principal_point": [140.0, 140.0],
        "cpu_sample_offset_pixels": [0.5, 0.5],
        "positive_x_right": True,
        "positive_y_down": True,
        "positive_z_forward": True,
        "textured_integer_principal_point": [139.5, 139.5],
    }
    custom_glb = {
        "bounds_max_m": maximum,
        "bounds_min_m": list(minimum),
        "independent_trimesh_uv_conversion": {
            "conversion": "detached comparison copy only: raw_v = 1 - trimesh_bottom_left_v",
            "custom_cpu_sampler_conversion": "renderer.py samples y = (1 - v) * (height - 1)",
            "source_pin": _source_key(source_hashes, "trimesh/exchange/gltf.py"),
            "source_sha256": source_hashes[_source_key(source_hashes, "trimesh/exchange/gltf.py")],
        },
        "raw_gltf_uv_convention": "top-left glTF TEXCOORD_0; original reader values retained",
        "size_m": size,
        "surface_position_uv_signature_sha256": signature,
        "surface_signature_match": True,
        "triangle_count": 12,
        "trimesh_surface_position_uv_signature_sha256": signature,
        "uv_max": [0.9, 0.9],
        "uv_min": [0.1, 0.1],
        "vertex_count": 8,
    }
    base_texture_sha = _label_sha("synthetic-base-texture-rgba")
    image_sha = _label_sha("synthetic-embedded-base-image")
    gltf = {
        "bounds_max_m": maximum, "bounds_min_m": list(minimum),
        "node_count": 1,
        "nodes": [{
            "base_color_texture": {"mode": "RGB", "rgba_pixel_sha256": base_texture_sha,
                    "shape": [2048, 2048, 4]},
            "expected_flat_corner_count": 36, "geometry": "GLTF",
            "gpu_position_float32_max_abs_rounding_m": 0.00000001,
            "gpu_uv_float32_max_abs_rounding": 0.000001,
            "material_class": "PBRMaterial", "material_name": "synthetic-material-8",
            "node": "synthetic-object8-mesh",
            "transform_world_from_node": [[1.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]],
            "transformed_bounds_max_m": maximum, "transformed_bounds_min_m": list(minimum),
            "triangle_count": 12, "uv_max": [0.9, 0.9], "uv_min": [0.1, 0.1],
            "vertex_count": 8,
        }],
        "raw_gltf_uv_max_after_inverse_v": [0.9, 0.9],
        "raw_gltf_uv_min_after_inverse_v": [0.1, 0.1],
        "size_m": size, "triangle_count": 12,
        "uv_convention": "trimesh bottom-left; raw glTF restored only on comparison copy with v := 1-v",
        "uv_max": [0.9, 0.9], "uv_min": [0.1, 0.1], "vertex_count": 8,
    }
    textured = {
        "asset_sha256": asset_sha,
        "bounds_max_m": maximum, "bounds_min_m": list(minimum),
        "flat_primitive_validation": {
            "actual_corners": 36, "actual_materials": ["synthetic-material-8"],
            "actual_signature_shape": [12, 15],
            "connectivity_multiplicity_and_winding_preserved": True,
            "equality": True,
            "expected_conversion": "independent transformed indexed triangles -> flat corners -> float32 attributes",
            "expected_corners": 36, "expected_nodes": gltf["nodes"],
            "expected_signature_shape": [12, 15], "first_mismatched_triangle": None,
            "indexed_vertices": 8, "position_quantization_m": 0.00000001,
            "uv_quantization": 0.000001,
        },
        "material_count": 1,
        "materials": [{"mesh_index": 0, "primitive_count": 1,
                       "primitives": [{"primitive_index": 0, "material_index": 0}]}],
        "primitive_position_uv_matches_trimesh": True,
        "primitive_position_uv_signature_sha256": signature,
        "transformed_position_signature_sha256": signature,
        "transformed_positions_match_trimesh": True,
        "transformed_vertex_count": 8,
    }
    embedded = {
        "material": {"alpha_mode": "OPAQUE", "base_color_factor": [1, 1, 1, 1],
                     "base_color_image_index": 0, "base_color_texture_index": 0,
                     "base_color_texture_info": {"index": 0}, "extensions": None,
                     "extras": None, "metallic_factor": 0.0, "name": "synthetic-material-8",
                     "other_texture_refs": {}, "roughness_factor": 1.0},
        "samplers": [{"magFilter": 9729, "minFilter": 9987,
                      "wrapS": 10497, "wrapT": 10497}],
        "textures": [{"extensions": None, "extras": None, "index": 0,
                      "name": "synthetic-base", "sampler": 0, "source": 0}],
        "images": [{"buffer_view": 0, "encoded_bytes": 16,
                    "encoded_sha256": image_sha, "extensions": None, "extras": None,
                    "height": 2048, "index": 0, "mime_type": "image/png",
                    "name": "synthetic-base", "rgba_dtype": "uint8",
                    "rgba_pixel_sha256": base_texture_sha, "width": 2048}],
        "cpu_texture_sampling": {"custom_renderer_uv_origin": "bottom-left; renderer flips v when sampling",
                                 "texture_repeat": True, "wrap_s": "repeat", "wrap_t": "repeat"},
    }
    renderer = {
        "camera_checks": camera_checks,
        "external_allocation_hook_used": False,
        "capture_depth_policy": {
            "current_default_depth_gate_passed": False,
            "current_default_disable_multisampling": False,
            "object_control_framebuffer_metadata": [
                _gpu_metadata(
                    (280, 280), 2, source_hashes, offscreen_identity=202,
                    allocation_calls=(
                        [_allocation_attachment((280, 280), "color", 2),
                         _allocation_attachment((280, 280), "depth", 2)]
                        if index == 0 else []))
                for index in range(4)],
            "proposed_configuration_status": "private_diagnostic_candidate_only_not_integrated",
            "proposed_rasterization_disable_depth_gate_passed": False,
            "tracking_accuracy_claim": False,
            "verified_zero_sample_attachment_depth_gate_passed": True,
            "verified_zero_sample_attachments": copy.deepcopy(zero),
        },
        "catalog": {
            "asset_receipt_bounds_match": True, "metric_extent_atol_m": 0.00001,
            "metric_extent_rtol": 0.0001, "object_id": 8,
            "object_name": "mug_patterned", "original_id": "synthetic-original-id-8",
            "size_m": size,
        },
        "custom_glb_reader": custom_glb,
        "dependency_versions": {},
        "embedded_textures": embedded,
        "production_viewport_controls": viewport,
        "render_settings": {
            "cpu_near_plane_m": 0.03, "cpu_shade": True, "cpu_texture_repeat": True,
            "disable_multisampling": False,
            "multisample_draw_attachment_storage": "candidate-local ordinary color/depth storage substitution",
            "renderer_mm_to_m_scale": 0.001,
            "textured_renderer_coordinate_mode": "integer_centers_v1",
            "textured_renderer_coordinate_mode_parameter": True,
            "textured_renderer_unlit": False,
        },
        "runtime_platform": {"system": "synthetic-test-fixture", "machine": "synthetic"},
        "runtime_python": "synthetic-test-fixture-python",
        "slanted_plane_depth_runs": copy.deepcopy(plane_runs),
        "slanted_plane_ray_fixture": copy.deepcopy(plane_runs["verified_zero_sample_attachments"]),
        "source_gltf": {"asset_sha256": asset_sha, "schema": "synthetic glTF object 8"},
        "textured_renderer": textured,
        "textured_renderer_api": {
            "constructor_signature": "(self, path, obj_id, unlit=False, disable_multisampling=False, coordinate_mode='legacy', *, render_policy='legacy_v1')",
            "explicit_coordinate_mode_supported": True,
            "legacy_coordinate_mode_default": "legacy",
            "selected_coordinate_mode": "integer_centers_v1",
        },
        "textured_renderer_coordinate_mode": "integer_centers_v1",
        "trimesh_base_color_texture_pixel_match": True,
        "trimesh_loader": gltf,
        "zero_sample_control": zero,
    }
    proof_rows = _object_control_rows(asset_sha)
    proof = {
        "schema_version": 1,
        "id": "synthetic-public-renderer-proof-fixture",
        "category": "diagnostic",
        "scope": "synthetic unit-test fixture only; no renderer execution",
        "status": "completed",
        "diagnostic_passed": True,
        "completed_at": "2026-10-04T00:00:01Z",
        "started_at": "2026-10-04T00:00:00Z",
        "asset_sha256": asset_sha,
        "catalog_sha256": catalog_sha,
        "geometry_receipt_sha256": geometry_sha,
        "object_id": 8,
        "object_name": "hot3d_obj_000008",
        "external_allocation_hook_used": False,
        "captured_rgb_read": False,
        "evaluator_data_read": False,
        "inference_executed": False,
        "neural_execution": False,
        "network_used": False,
        "model_stage_child_process_used": False,
        "tracking_accuracy_claim": False,
        "capture_depth_policy_integrated": False,
        "capture_stage_callers_integrated": False,
        "public_production_render_policy_implemented": True,
        "synthetic_mask_is_not_observed_foreground": True,
        "torch_checkpoint_load_blocked": True,
        "default_capture_depth_gate_passed": False,
        "default_capture_policy_gate_failed": True,
        "proposed_capture_policy_is_diagnostic_only": True,
        "proposed_capture_msaa_disabled_depth_gate_passed": False,
        "verified_zero_sample_attachment_depth_gate_passed": True,
        "source_sha256": _source_records(source_hashes),
        "gates": {
            "all_mask_iou_at_least_0_94": True,
            "all_median_common_depth_error_at_most_0_002m": True,
            "all_three_recipes_front_facing": True,
            "all_three_rotations_proper_and_nonsymmetric": True,
            "camera_axis_pixel_center_and_mm_checks": True,
            "independent_trimesh_custom_glb_position_uv_match": True,
            "paired_plane_reports_retain_eight_labeled_samples_each": True,
            "public_viewport_resize_reuse_and_recreation": True,
            "renderer_base_texture_matches_embedded_glb": True,
            "renderer_extents_match_catalog_and_receipt": True,
            "textured_renderer_integer_centers_v1_api": True,
            "verified_zero_sample_integer_center_plane_gate": True,
        },
        "renderer": renderer,
        "rows": proof_rows,
        "production_viewport_controls": copy.deepcopy(viewport),
        "paired_plane_run_artifacts": [
            {"bytes": 0, "depth_gate_passed": False, "mode": "default_msaa_enabled",
             "negative_control_passed": True, "path": "synthetic-default-msaa.json",
             "row_count": 8, "sha256": _label_sha("synthetic-default-plane-artifact"),
             "source_sha256": _source_records(source_hashes)},
            {"bytes": 0, "depth_gate_passed": False,
             "mode": "proposed_msaa_rasterization_disabled",
             "negative_control_passed": True, "path": "synthetic-disabled-msaa.json",
             "row_count": 8, "sha256": _label_sha("synthetic-disabled-plane-artifact"),
             "source_sha256": _source_records(source_hashes)},
            {"bytes": 0, "depth_gate_passed": True,
             "mode": "verified_zero_sample_attachments",
             "negative_control_passed": True, "path": "synthetic-zero-sample-plane.json",
             "row_count": 8, "sha256": _label_sha("synthetic-zero-sample-plane-artifact"),
             "source_sha256": _source_records(source_hashes)},
        ],
    }
    return proof


def _new_bundle(root: Path, *, clip_id=1944, proof_mutator=None, raw_proof=None):
    bundle = protocol._SyntheticBundle(root)
    bundle.table["clip_id"] = clip_id
    bundle.inventory["clip_id"] = clip_id
    bundle.manifest["object"] = "hot3d_obj_000008"
    bundle.manifest["object_id"] = 8
    if raw_proof is None:
        asset_document = _synthetic_metric_asset_document(bundle)
        proof = _synthetic_renderer_proof(asset_document)
        if proof_mutator is not None:
            proof_mutator(proof)
        raw_proof = _canonical(proof)
        receipt = _synthetic_metric_receipt(
            bundle, raw_proof, asset_document=asset_document)
    else:
        receipt = _synthetic_metric_receipt(bundle, raw_proof)
    bundle.manifest["object"] = "hot3d_obj_000008"
    bundle.manifest["object_id"] = 8
    bundle.seal()
    bundle.write("input.json", bundle.input_bytes)
    plan = capture.load_capture_plan(bundle.root, bundle.input_bytes)
    return bundle, receipt, raw_proof, plan


@contextmanager
def _fake_resource_environment(assets_module, root: Path):
    """Set up only inert source/runtime/checkpoint bytes in a temporary tree."""
    with _fake_resource_tree(assets_module, None, root) as tree:
        # The proof validator reads this exact repository-relative slice, while
        # runtime-closure validation separately uses the fake distributions.
        # Seed only inert synthetic bytes at the fixed validator paths.
        for relative in assets_module._CAPTURE_RENDERER_SOURCE_PATHS:
            normalized = relative.replace("\\", "/")
            if normalized == "bench/quality_camera.py":
                # _fake_resource_tree already copies the exact converter source
                # bytes recorded in the synthetic preflight plan. Preserve that
                # pin; the proof-only renderer slice may still be synthetic.
                continue
            target = root.joinpath(*relative.replace("\\", "/").split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(("synthetic renderer source fixture|" + relative).encode("utf-8"))
        yield tree


def _mutate_path(document, path, value):
    current = document
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _write_hash_only_smoke_sidecar(use):
    """Write well-hashed structural bytes with no valid smoke semantics."""
    directory = use.sidecar_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    smoke_bytes = _canonical({
        "resource_kind": use.resource_kind,
        "resource_key": use.resource_key,
        "status": "passed",
        "synthetic_fixture_only": True,
    })
    execution_bytes = _canonical({"kind": "synthetic renderer execution fixture"})
    (directory / "smoke.json").write_bytes(smoke_bytes)
    (directory / "renderer-execution.json").write_bytes(execution_bytes)
    sidecar = {
        "schema_version": 1,
        "resource_kind": use.resource_kind,
        "resource_key": use.resource_key,
        "recipe": _plain(use.recipe),
        "dimensions": [720, 720],
        "coordinate_mode": capture.COORDINATE_MODE,
        "artifact": {"path": "smoke.json", "sha256": _sha(smoke_bytes),
                     "byte_count": len(smoke_bytes)},
        "runtime": {"device": use.recipe["producer_device"],
                    "build": "synthetic-hash-only-fixture"},
        "source_closure": _plain(use.source_closure),
    }
    use.sidecar_path.write_bytes(_canonical(_plain(sidecar)))


class CaptureResourceProofTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hot3d-capture-proof-")
        self.temporary_root = Path(self.temporary.name)
        self.assertEqual(Path(assets.__file__).resolve(), (_ROOT / "bench" / "quality_assets.py").resolve())

    def tearDown(self):
        self.temporary.cleanup()

    def _bundle(self, name="bundle", **kwargs):
        return _new_bundle(self.temporary_root / name, **kwargs)

    def test_00_public_asset_helper_binds_fixed_renderer_proof_bytes(self):
        bundle, _, raw_proof, plan = self._bundle()
        receipt = assets.read_capture_asset_receipt(plan)
        self.assertEqual(receipt.renderer_proof_sha256, _sha(raw_proof))
        self.assertEqual(receipt.renderer_proof_sha256,
                         receipt.document["renderer_receipt_sha256"])

        with self.assertRaises(TypeError):
            receipt.document["object_name"] = "mutated"
        with self.assertRaises(TypeError):
            receipt.document["models_info"]["object_bounds_m"]["min"][0] = 99.0
        with self.assertRaises(TypeError):
            receipt.renderer_proof["proof_sha256"] = "0" * 64
        source_key = next(iter(receipt.renderer_proof["source_closure"]))
        with self.assertRaises(TypeError):
            receipt.renderer_proof["source_closure"][source_key] = "0" * 64

        bundle.write("assets/renderer-proof.json", b"{}")
        with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
            assets.read_capture_asset_receipt(plan)

    def test_bounded_renderer_proof_json_rejects_duplicate_nonfinite_and_large_documents(self):
        invalid_documents = (
            b'{"schema_version":1,"schema_version":1}',
            b'{"value":NaN}',
            b"{}" + b" " * (2 * 1024 * 1024),
        )
        for index, raw in enumerate(invalid_documents):
            with self.subTest(invalid_proof=index):
                _, _, _, plan = self._bundle(f"invalid-json-{index}", raw_proof=raw)
                with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
                    assets.read_capture_asset_receipt(plan)

    def test_proof_semantics_recompute_nested_evidence_instead_of_trusting_header_gates(self):
        def set_first_iou(proof):
            proof["rows"][0]["textured_mask_iou"] = 0.50

        def set_failed_plane_sample(proof):
            candidate = proof["renderer"]["slanted_plane_ray_fixture"]
            candidate["rows"][0]["renderer_depth_m_after_single_mm_conversion"] += 0.01

        def select_private_mode(proof):
            proof["renderer"]["textured_renderer_api"]["selected_coordinate_mode"] = "legacy"

        def enable_hook(proof):
            proof["external_allocation_hook_used"] = True

        def stale_mm_boundary(proof):
            proof["renderer"]["render_settings"]["renderer_mm_to_m_scale"] = 1.0

        def stale_camera_k(proof):
            proof["rows"][0]["textured_renderer_input_intrinsics"]["cx"] += 0.5

        def stale_dimensions(proof):
            proof["renderer"]["production_viewport_controls"][2]["dimensions"] = [719, 720]

        def stale_generation(proof):
            proof["renderer"]["production_viewport_controls"][0]["gpu_sampling_metadata"][
                "production_policy_metadata"]["allocation_generation"] = 99

        def stale_cleanup(proof):
            proof["renderer"]["production_viewport_controls"][0]["gpu_sampling_metadata"][
                "current_context_released"] = False

        def missing_gl_query(proof):
            proof["renderer"]["production_viewport_controls"][0]["gpu_sampling_metadata"].pop("gl_samples")

        def stale_baseline(proof):
            proof["renderer"]["slanted_plane_depth_runs"]["default_msaa_enabled"][
                "depth_gate_passed"] = True

        def missing_viewport(proof):
            proof["renderer"]["production_viewport_controls"].pop()

        def contradictory_viewport_sample(proof):
            sample = proof["renderer"]["production_viewport_controls"][0]["rows"][0]
            sample["actual_m"] += 0.01

        def stale_nested_source(proof):
            proof["renderer"]["production_viewport_controls"][0]["gpu_sampling_metadata"][
                "production_policy_metadata"]["source_identities"]["pyrender_offscreen"][
                    "sha256"] = _label_sha("stale-per-render-source")

        def changed_top_source(proof):
            key = _source_key(proof["source_sha256"], "bench/quality_gotrack.py")
            proof["source_sha256"][key]["sha256"] = _label_sha("stale-gotrack-source")

        mutations = (
            ("header failure", lambda proof: proof.update({"diagnostic_passed": False})),
            ("private allocation hook", enable_hook),
            ("private coordinate mode", select_private_mode),
            ("nested IoU contradicts PASS header", set_first_iou),
            ("nested plane numeric contradiction", set_failed_plane_sample),
            ("millimetre conversion", stale_mm_boundary),
            ("camera K", stale_camera_k),
            ("viewport dimensions", stale_dimensions),
            ("stale allocation generation", stale_generation),
            ("missing cleanup", stale_cleanup),
            ("missing queried GL field", missing_gl_query),
            ("rewritten failed baseline", stale_baseline),
            ("missing viewport control", missing_viewport),
            ("viewport row depth contradicts its error", contradictory_viewport_sample),
            ("per-render source identity", stale_nested_source),
            ("relevant top-level source identity", changed_top_source),
            ("observed RGB read", lambda proof: proof.update({"captured_rgb_read": True})),
            ("evaluator data read", lambda proof: proof.update({"evaluator_data_read": True})),
            ("inference ran", lambda proof: proof.update({"inference_executed": True})),
            ("capture callers falsely integrated", lambda proof: proof.update({"capture_stage_callers_integrated": True})),
        )
        for label, mutate in mutations:
            with self.subTest(proof_mutation=label):
                _, _, _, plan = self._bundle("invalid-proof-" + label.replace(" ", "-"),
                                             proof_mutator=mutate)
                with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
                    assets.read_capture_asset_receipt(plan)

    def test_legacy_baseline_query_evidence_is_measured_and_bound(self):
        mutations = (
            ("query status", ("gpu_sampling_metadata", "status"), "unavailable"),
            ("sample count", ("gpu_sampling_metadata", "gl_samples"), 0),
            ("sample buffer count", ("gpu_sampling_metadata", "gl_sample_buffers"), 0),
            ("framebuffer support", ("gpu_sampling_metadata", "framebuffers_supported"), False),
            ("binding restoration", ("gpu_sampling_metadata", "framebuffer_bindings_restored"), False),
            ("draw target binding", ("gpu_sampling_metadata", "draw_framebuffer_bound_for_sample_query"), 9),
            ("invalid draw target", ("gpu_sampling_metadata", "framebuffer_fields",
                                      "multisample_draw_fbo"), 0),
            ("aliased draw/read targets", ("gpu_sampling_metadata", "framebuffer_fields",
                                            "single_sample_read_fbo"), 2),
            ("queried dimensions", ("gpu_sampling_metadata", "framebuffer_fields",
                                    "multisample_dimensions"), [63, 48]),
            ("boolean saved binding", ("gpu_sampling_metadata",
                                        "framebuffer_bindings_before_query", "draw"), True),
            ("sample coordinate outside pixel", ("gpu_sampling_metadata", "sample_positions",
                                                  0, "sample_x_from_left"), 1.25),
            ("duplicate sample index", ("gpu_sampling_metadata", "sample_positions",
                                         1, "sample_index"), 0),
            ("inconsistent sample offset", ("gpu_sampling_metadata", "sample_positions",
                                             0, "du_from_pixel_center_x"), 0.25),
            ("baseline hook claim", ("external_allocation_hook_used",), True),
        )
        for baseline_name in ("default_msaa_enabled",
                              "proposed_msaa_rasterization_disabled"):
            for label, suffix, value in mutations:
                path = ("renderer", "slanted_plane_depth_runs", baseline_name) + suffix

                def mutate(proof, path=path, value=value):
                    _mutate_path(proof, path, value)

                with self.subTest(baseline=baseline_name, mutation=label):
                    _, _, _, plan = self._bundle(
                        f"invalid-baseline-{baseline_name}-{label.replace(' ', '-')}"
                        f"-{path[-1]}",
                        proof_mutator=mutate)
                    with self.assertRaises((ValueError, capture.CaptureIntegrityError,
                                            TypeError)):
                        assets.read_capture_asset_receipt(plan)

    def test_proof_failures_precede_resource_and_stage_output_hooks(self):
        def contradictory_evidence(proof):
            proof["rows"][1]["median_common_absolute_depth_error_m"] = 0.1

        _, _, _, plan = self._bundle("poisoned-proof", proof_mutator=contradictory_evidence)
        forbidden = AssertionError("resource or output planning ran before proof rejection")
        with (
            patch.object(assets, "_capture_source_closure", side_effect=forbidden) as closure,
            patch.object(assets, "_capture_resource_plan", side_effect=forbidden) as output,
            patch.object(capture, "verify_stage_resource_sidecar", side_effect=forbidden) as sidecar,
        ):
            with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
                assets.capture_stage_resources(plan, stage="smoke", device="cpu")
        closure.assert_not_called()
        output.assert_not_called()
        sidecar.assert_not_called()

    def test_non_object_renderer_proof_json_types_fail_before_output_planning(self):
        invalid_roots = (b"[]", b"null", b"true")
        for index, raw in enumerate(invalid_roots):
            with self.subTest(invalid_proof_root=index):
                _, _, _, plan = self._bundle(
                    f"invalid-proof-root-{index}", raw_proof=raw)
                forbidden = AssertionError("malformed renderer proof reached output planning")
                with (
                    patch.object(assets, "_capture_source_closure",
                                 side_effect=forbidden) as closure,
                    patch.object(assets, "_capture_resource_plan",
                                 side_effect=forbidden) as output,
                    patch.object(capture, "verify_stage_resource_sidecar",
                                 side_effect=forbidden) as sidecar,
                ):
                    with self.assertRaises((ValueError, capture.CaptureIntegrityError,
                                            TypeError)):
                        assets.capture_stage_resources(
                            plan, stage="smoke", device="cpu")
                closure.assert_not_called()
                output.assert_not_called()
                sidecar.assert_not_called()

    def test_fixed_proof_source_slice_excludes_bookkeeping_but_detects_renderer_and_runtime_changes(self):
        fake_root = self.temporary_root / "source-slice-root"
        with _fake_resource_environment(assets, fake_root):
            bundle, _, _, plan = self._bundle("source-slice-bundle")
            baseline_receipt = assets.read_capture_asset_receipt(plan)
            baseline = assets.capture_stage_resources(plan, stage="smoke", device="cpu").output

            bookkeeping = fake_root / "bench" / "quality_assets.py"
            bookkeeping.write_bytes(bookkeeping.read_bytes() + b"\nsynthetic bookkeeping-only change")
            self.assertEqual(assets.read_capture_asset_receipt(plan).renderer_proof_sha256,
                             baseline_receipt.renderer_proof_sha256)
            bookkeeping_changed = assets.capture_stage_resources(
                plan, stage="smoke", device="cpu").output
            self.assertNotEqual(bookkeeping_changed.resource_key, baseline.resource_key)

            runner = fake_root / "bench" / "quality_runner.py"
            runner.write_bytes(runner.read_bytes() + b"\nsynthetic consumer adapter change")
            runner_changed = assets.capture_stage_resources(
                plan, stage="smoke", device="cpu").output
            self.assertNotEqual(runner_changed.resource_key, bookkeeping_changed.resource_key)

            sam_consumer = fake_root / "bench" / "quality_sam2.py"
            sam_consumer.write_bytes(sam_consumer.read_bytes() + b"\nsynthetic consumer-only change")
            sam_changed = assets.capture_stage_resources(
                plan, stage="smoke", device="cpu").output
            self.assertEqual(sam_changed.resource_key, runner_changed.resource_key)

            gotrack = fake_root / "bench" / "quality_gotrack.py"
            gotrack.write_bytes(gotrack.read_bytes() + b"\nsynthetic geometry renderer change")
            with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
                assets.read_capture_asset_receipt(plan)

    def test_renderer_proof_and_effective_flags_change_current_recipe_keys(self):
        fake_root = self.temporary_root / "recipe-root"
        with _fake_resource_environment(assets, fake_root):
            _, _, _, plan = self._bundle("recipe-bundle")
            receipt = assets.read_capture_asset_receipt(plan)
            base = assets.capture_stage_resources(plan, stage="smoke", device="cpu").output
            unlit = assets.capture_stage_resources(
                plan, stage="smoke", render_flags={"unlit_templates": True}, device="cpu").output
            self.assertNotEqual(base.resource_key, unlit.resource_key)
            self.assertEqual(base.recipe["asset_receipt_sha256"], receipt.receipt_sha256)
            self.assertEqual(base.recipe["renderer_proof_sha256"], receipt.renderer_proof_sha256)
            self.assertEqual(base.recipe["rendering_contract"]["render_policy"],
                             "capture_zero_sample_v1")
            self.assertEqual(base.recipe["rendering_contract"]["coordinate_mode"],
                             "integer_centers_v1")
            self.assertFalse(base.recipe["rendering_contract"]["disable_multisampling"])

            forbidden = AssertionError("disabled capture mode reached resource planning")
            with (
                patch.object(assets, "_capture_source_closure", side_effect=forbidden) as closure,
                patch.object(assets, "_capture_resource_plan", side_effect=forbidden) as output,
            ):
                with self.assertRaises(ValueError):
                    assets.capture_stage_resources(
                        plan, stage="smoke", render_flags={"disable_multisampling": True},
                        device="cpu")
            closure.assert_not_called()
            output.assert_not_called()

            def change_proof_id(proof):
                proof["id"] = _label_sha("second semantically identical synthetic proof")[:32]

            _, _, _, second_plan = self._bundle(
                "recipe-bundle-proof-variant", proof_mutator=change_proof_id)
            changed_proof = assets.capture_stage_resources(
                second_plan, stage="smoke", device="cpu").output
            self.assertNotEqual(base.resource_key, changed_proof.resource_key)

    def test_identical_asset_producer_recipe_reuses_smoke_key_across_clips(self):
        fake_root = self.temporary_root / "cross-clip-root"
        with _fake_resource_environment(assets, fake_root):
            first, _, _, first_plan = self._bundle("clip-one", clip_id=1944)
            second, _, _, second_plan = self._bundle("clip-two", clip_id=1945)
            self.assertNotEqual(first_plan.capture_table_sha256,
                                second_plan.capture_table_sha256)
            self.assertNotEqual(first_plan.input_manifest_sha256,
                                second_plan.input_manifest_sha256)
            first_use = assets.capture_stage_resources(
                first_plan, stage="smoke", device="cpu").output
            second_use = assets.capture_stage_resources(
                second_plan, stage="smoke", device="cpu").output
            self.assertEqual(first_use.resource_key, second_use.resource_key)

    def test_hash_and_kind_only_smoke_prerequisites_fail_closed_before_bank_output(self):
        fake_root = self.temporary_root / "smoke-prerequisite-root"
        with _fake_resource_environment(assets, fake_root):
            _, _, _, plan = self._bundle("hash-only-smoke-bundle")
            smoke = assets.capture_stage_resources(
                plan, stage="smoke", device="cpu").output
            _write_hash_only_smoke_sidecar(smoke)

            original = assets._capture_resource_plan
            planned_resources = []

            def record_resource_plan(resource_name, *args, **kwargs):
                planned_resources.append(resource_name)
                return original(resource_name, *args, **kwargs)

            with patch.object(assets, "_capture_resource_plan",
                              side_effect=record_resource_plan):
                with self.assertRaises((ValueError, capture.CaptureIntegrityError,
                                        OSError, TypeError)):
                    assets.capture_stage_resources(plan, stage="banks", device="cpu")
            # Implementations may reject the smoke prerequisite either before or
            # immediately after deriving its expected key.  The invariant here is
            # that no downstream bank resource/output is planned from a
            # hash-only receipt; do not couple the test to that private ordering.
            self.assertNotIn("foundpose-bank", planned_resources)
            self.assertNotIn("cnos-bank", planned_resources)

