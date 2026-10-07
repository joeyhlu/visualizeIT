"""Source-only R8 reservation primitives.

This module contains no matching, fitting, inference, or rendering entry point.
The first implementation slice is limited to the frozen crop/native source-ray
bridge and a full-population native-mesh association pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np


SPEC_V1_SHA256 = "823255763EAF95DE0C2DF59209E41DDA52103CFCEAD49DA3802CB30ABC1E4675"
SPEC_V2_SHA256 = "32758DDA6AD550A5EDD94F5354B7E8401FF206A63B884E8C88AB37F71EE6D9CB"
MAX_WORKING_BYTES = 128 * 1024**2
MAX_WALL_SECONDS = 900
DEFAULT_RAY_CHUNK = 256
GEOMETRY_EPSILON_ABS_M = 1e-5
GEOMETRY_EPSILON_DIAGONAL_FRACTION = 1e-4


class SourceReservationError(ValueError):
    """A frozen source-capacity geometry prerequisite is invalid."""


class SourceCapacityLimitError(SourceReservationError):
    """The source-only association exceeds a frozen memory or time cap."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _input_array(value: object, label: str, *, ndim: int,
                 kinds: str, item_sizes: tuple[int, ...]) -> np.ndarray:
    """Inspect a caller-owned array without coercing or copying it."""
    if not isinstance(value, np.ndarray):
        raise SourceReservationError(f"{label} must be supplied as a preallocated NumPy array")
    array = value
    if (array.ndim != ndim or not array.flags.c_contiguous or
            array.dtype.kind not in kinds or array.dtype.itemsize not in item_sizes):
        raise SourceReservationError(
            f"{label} must be C-contiguous {ndim}D data with dtype kind {kinds!r}")
    return array


def _validate_intrinsics(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
            matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or
            abs(float(matrix[2, 0])) > 1e-12 or
            abs(float(matrix[2, 1])) > 1e-12 or
            abs(float(matrix[2, 2]) - 1.0) > 1e-12 or
            abs(float(np.linalg.det(matrix))) <= 1e-12):
        raise SourceReservationError(f"{label} is not a finite pinhole intrinsic matrix")
    return matrix


def _validate_rigid_transform(value: object, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SourceReservationError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-10):
        raise SourceReservationError(f"{label} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6) or
            determinant <= 0.0 or abs(determinant - 1.0) > 1e-6):
        raise SourceReservationError(f"{label} rotation is not rigid with positive determinant")
    return transform


def crop_source_rays_object(source_pixels_xy: Sequence[Sequence[float]],
                            crop_k: np.ndarray,
                            template_pose_m: np.ndarray,
                            crop_from_native: np.ndarray) -> dict:
    """Convert captured crop pixels to normalized object-frame rays.

    ``template_pose_m`` is native-camera-from-object in metres.  The captured
    ``crop_from_native`` transform acts on native-camera millimetres, matching
    the original packet producer.  Source depth is camera-z, so the returned
    unnormalized crop rays are also retained for the producer-point equation.
    """
    pixels = np.asarray(source_pixels_xy, dtype=np.float64)
    if pixels.size == 0:
        pixels = np.empty((0, 2), dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise SourceReservationError("Source pixel centers must be finite Nx2 coordinates")
    k = _validate_intrinsics(crop_k, "crop_k")
    pose_native_m = _validate_rigid_transform(template_pose_m, "template_pose_m")
    c = _validate_rigid_transform(crop_from_native, "crop_from_native")

    pose_native_mm = pose_native_m.copy()
    pose_native_mm[:3, 3] *= 1000.0
    pose_crop_mm = c @ pose_native_mm
    if not np.isfinite(pose_crop_mm).all():
        raise SourceReservationError("Derived crop pose is nonfinite")
    rotation_crop = pose_crop_mm[:3, :3]
    if (not np.allclose(rotation_crop.T @ rotation_crop, np.eye(3), rtol=0.0, atol=1e-6) or
            float(np.linalg.det(rotation_crop)) <= 0.0 or
            abs(float(np.linalg.det(rotation_crop)) - 1.0) > 1e-6 or
            not np.allclose(pose_crop_mm[3], (0.0, 0.0, 0.0, 1.0),
                            rtol=0.0, atol=1e-10)):
        raise SourceReservationError("Derived crop-camera pose is not a valid rigid transform")

    crop_rays = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float64))) @ np.linalg.inv(k).T
    directions = crop_rays @ rotation_crop
    norms = np.linalg.norm(directions, axis=1)
    if len(norms) and (not np.isfinite(norms).all() or np.any(norms <= 0.0)):
        raise SourceReservationError("A crop source ray is invalid")
    if len(directions):
        directions = directions / norms[:, None]
    origin = -(rotation_crop.T @ (pose_crop_mm[:3, 3] * 0.001))
    if not np.isfinite(origin).all():
        raise SourceReservationError("Derived object-frame ray origin is nonfinite")

    return dict(
        origin_object_m=origin,
        directions_object_unit=directions,
        crop_rays_unnormalized=crop_rays,
        crop_pose_mm=pose_crop_mm,
        crop_pose_sha256=_array_sha256(pose_crop_mm),
        bridge=dict(
            units="native pose metres; C translation millimetres; crop pose millimetres; source depth camera-z millimetres",
            formula="T_crop_mm=C@T_native_mm; origin=-R.T@(t_mm*0.001); direction=normalize(([x,y,1]@inv(K_crop).T)@R)",
        ),
    )


def _check_budget(required_bytes: int, label: str) -> None:
    if int(required_bytes) < 0 or int(required_bytes) > MAX_WORKING_BYTES:
        raise SourceCapacityLimitError(
            f"{label} requires {int(required_bytes)} bytes over the {MAX_WORKING_BYTES}-byte working cap")


def _require_sha256(value: str, label: str) -> str:
    digest = str(value).casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SourceReservationError(f"{label} must be a 64-character SHA256")
    return digest


def _r7_geometry_helpers():
    # Kept local so this pure bridge remains importable before the frozen R7
    # geometry module is needed by a caller.  The R7 implementation is reused
    # without editing or copying its ray and topology kernels.
    from . import quality_bottle_joint_prerequisite as r7
    return r7


def native_source_association_all(
        source_ids: Sequence[int],
        source_pixels_xy: Sequence[Sequence[float]],
        source_depth_mm: np.ndarray,
        source_points_object_m: Sequence[Sequence[float]],
        *, crop_k: np.ndarray,
        crop_from_native: np.ndarray,
        native_k: np.ndarray,
        template_pose_m: np.ndarray,
        mesh_positions_m: np.ndarray,
        mesh_triangles: np.ndarray,
        model_diagonal_m: float,
        source_context_id: str,
        packet_sha256: str,
        mesh_sha256: str,
        raycast_fn: Callable[..., Mapping[str, object]] | None = None,
        chunk_size: int = DEFAULT_RAY_CHUNK,
        resident_bytes: int = 0,
        deadline_monotonic: float | None = None,
) -> dict:
    """Validate the native first-hit association for every original source ID.

    Rays are generated in the captured crop camera, but transformed through
    the frozen crop/native pose bridge before calling the unchanged R7 first-
    hit kernel.  Full-population source IDs are retained in order; unsupported
    samples remain present in ``support_bits`` and the denominator.
    """
    started = time.monotonic()
    hard_deadline = started + MAX_WALL_SECONDS
    if deadline_monotonic is None:
        deadline = hard_deadline
    else:
        supplied_deadline = float(deadline_monotonic)
        if not math.isfinite(supplied_deadline):
            raise SourceReservationError("Caller deadline must be finite")
        deadline = min(hard_deadline, supplied_deadline)
    if deadline <= started:
        raise SourceCapacityLimitError("Native source association deadline has already expired")
    if int(resident_bytes) < 0:
        raise SourceReservationError("Resident-byte reservation cannot be negative")

    raw_ids = _input_array(source_ids, "source_ids", ndim=1, kinds="iu", item_sizes=(4, 8))
    raw_pixels = _input_array(source_pixels_xy, "source_pixels_xy", ndim=2,
                              kinds="f", item_sizes=(4, 8))
    depth = _input_array(source_depth_mm, "source_depth_mm", ndim=2,
                         kinds="f", item_sizes=(4, 8))
    raw_points = _input_array(source_points_object_m, "source_points_object_m", ndim=2,
                              kinds="f", item_sizes=(4, 8))
    raw_vertices = _input_array(mesh_positions_m, "mesh_positions_m", ndim=2,
                                 kinds="f", item_sizes=(4, 8))
    raw_triangles = _input_array(mesh_triangles, "mesh_triangles", ndim=2,
                                 kinds="iu", item_sizes=(4, 8))
    if (raw_ids.ndim != 1 or raw_pixels.ndim != 2 or raw_pixels.shape[1] != 2 or
            raw_points.ndim != 2 or raw_points.shape[1] != 3 or
            raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or
            raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3):
        raise SourceReservationError("Original source or native mesh arrays have invalid dimensions")
    # Reserve conversion copies before making float64/int64 buffers.  The
    # packet arrays remain caller-owned and live until this function returns.
    cast_bytes = 0
    source_cast_bytes = 0
    for array, dtype, is_source in ((raw_ids, np.dtype(np.int64), True),
                                    (raw_pixels, np.dtype(np.float64), True),
                                    (raw_points, np.dtype(np.float64), True),
                                    (raw_vertices, np.dtype(np.float64), False),
                                    (raw_triangles, np.dtype(np.int64), False)):
        if array.dtype != dtype:
            required = int(array.size * dtype.itemsize)
            cast_bytes += required
            if is_source:
                source_cast_bytes += required
    raw_input_bytes = int(raw_ids.nbytes + raw_pixels.nbytes + depth.nbytes +
                          raw_points.nbytes + raw_vertices.nbytes + raw_triangles.nbytes +
                          int(resident_bytes))
    validation_seal_reserve_bytes = (
        max(8 * int(raw_ids.shape[0]), 256 * 1024) + 64 * 1024 +
        int(raw_vertices.size + raw_triangles.size))
    mesh_tri_count = int(raw_triangles.shape[0])
    topology_workspace_bytes = int(mesh_tri_count * 3 * 256)
    topology_preflight_bytes = (raw_input_bytes + cast_bytes + topology_workspace_bytes +
                                validation_seal_reserve_bytes)
    _check_budget(topology_preflight_bytes, "all-original-ID topology preflight")

    ids = raw_ids.astype(np.int64, copy=False)
    pixels = raw_pixels.astype(np.float64, copy=False)
    points = raw_points.astype(np.float64, copy=False)
    vertices = raw_vertices.astype(np.float64, copy=False)
    triangles = raw_triangles.astype(np.int64, copy=False)
    native_k_value = _validate_intrinsics(native_k, "native_k")
    k = _validate_intrinsics(crop_k, "crop_k")
    pose_native = _validate_rigid_transform(template_pose_m, "template_pose_m")
    c = _validate_rigid_transform(crop_from_native, "crop_from_native")
    if not str(source_context_id):
        raise SourceReservationError("Source context identity is required")
    packet_hash = _require_sha256(packet_sha256, "packet_sha256")
    mesh_hash = _require_sha256(mesh_sha256, "mesh_sha256")
    if (depth.ndim != 2 or not len(depth.shape) or pixels.shape != (len(ids), 2) or
            points.shape != (len(ids), 3)):
        raise SourceReservationError("Original source IDs, pixels, depth, and object points do not align")
    if len(ids) and np.any(ids[:-1] >= ids[1:]):
        raise SourceReservationError("Original source IDs must remain in increasing packet order")
    height, width = depth.shape
    if not height or not width:
        raise SourceReservationError("Source depth dimensions must be positive")
    if len(ids) and (int(ids[0]) < 0 or int(ids[-1]) >= height * width):
        raise SourceReservationError("A source ID is outside the original depth image")
    for start in range(0, len(ids), 4096):
        stop = min(len(ids), start + 4096)
        expected_x = ids[start:stop] % width + 0.5
        expected_y = ids[start:stop] // width + 0.5
        if (not np.array_equal(pixels[start:stop, 0], expected_x) or
                not np.array_equal(pixels[start:stop, 1], expected_y)):
            raise SourceReservationError("Source ID/pixel-center binding differs from row-major packet IDs")
    if (not len(vertices) or not len(triangles) or
            int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices) or
            not np.isfinite(vertices).all()):
        raise SourceReservationError("Native mesh geometry is empty, nonfinite, or out of range")
    if not math.isfinite(float(model_diagonal_m)) or float(model_diagonal_m) <= 0.0:
        raise SourceReservationError("Native model diagonal must be finite and positive")
    r7 = _r7_geometry_helpers()
    frozen_raycast_fn = r7.raycast_first_hits
    if raycast_fn is None:
        raycast_fn = frozen_raycast_fn
    if not callable(raycast_fn):
        raise SourceReservationError("The unchanged first-hit ray kernel is required")
    raycast_kernel_identity = f"{getattr(raycast_fn, '__module__', '')}.{getattr(raycast_fn, '__qualname__', type(raycast_fn).__name__)}"
    raycast_kernel_is_frozen = raycast_fn is frozen_raycast_fn
    chunk = int(chunk_size)
    if chunk <= 0 or chunk > DEFAULT_RAY_CHUNK:
        raise SourceReservationError(f"Ray chunk must be between 1 and {DEFAULT_RAY_CHUNK}")
    requested_chunk = chunk

    # Topology discovery has its own conservative prospective bound, including
    # packet arrays resident beside the R7 geometric-edge table.
    small_geometry_bytes = int(k.nbytes + native_k_value.nbytes + pose_native.nbytes + c.nbytes)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap")
    topology = r7.native_mesh_topology(vertices, triangles)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native mesh topology exceeded the 900-second wall cap")
    topology_state = str(topology.get("state", "unknown"))
    unsupported_triangle_ids = np.asarray(
        topology.get("unsupported_triangle_ids", ()), dtype=np.int64).reshape(-1)
    if len(unsupported_triangle_ids) and np.any(
            unsupported_triangle_ids[:-1] >= unsupported_triangle_ids[1:]):
        unsupported_triangle_ids = np.sort(unsupported_triangle_ids)
    unsupported_triangle_count = int(len(unsupported_triangle_ids))
    unsupported_triangle_mask = np.zeros(len(triangles), dtype=bool)
    if unsupported_triangle_count:
        if (int(unsupported_triangle_ids[0]) < 0 or
                int(unsupported_triangle_ids[-1]) >= len(triangles)):
            raise SourceReservationError("Native topology returned an out-of-range triangle ID")
        unsupported_triangle_mask[unsupported_triangle_ids] = True
    del topology
    epsilon_m = max(GEOMETRY_EPSILON_ABS_M,
                    GEOMETRY_EPSILON_DIAGONAL_FRACTION * float(model_diagonal_m))

    # The outputs are support (1N), hit (1N), triangle ID (8N), three metric
    # error vectors (24N), reason code (1N), and an ordered source-ID copy
    # (8N): 43N total.  Hashing uses memoryview below, so it adds no full-array
    # serialization copy.  Include caller arrays, cast copies and topology
    # metadata before allocating these vectors.
    output_bytes = int(len(ids) * 43)
    topology_metadata_bytes = int(unsupported_triangle_ids.nbytes + len(triangles) + 1024)
    non_mesh_input_bytes = int(raw_ids.nbytes + raw_pixels.nbytes + depth.nbytes +
                               raw_points.nbytes + int(resident_bytes) + source_cast_bytes +
                               small_geometry_bytes)
    raw_mesh_copy_bytes = 0
    if vertices is not raw_vertices:
        raw_mesh_copy_bytes += int(raw_vertices.nbytes)
    if triangles is not raw_triangles:
        raw_mesh_copy_bytes += int(raw_triangles.nbytes)
    external_live = (non_mesh_input_bytes + raw_mesh_copy_bytes + output_bytes +
                     int(unsupported_triangle_mask.nbytes) + topology_metadata_bytes +
                     validation_seal_reserve_bytes)
    kernel_fixed_bytes = int(2 * vertices.nbytes + 2 * triangles.nbytes +
                             len(triangles) * 3 * 8 * 3 + 4096)
    wrapper_per_ray_bytes = 1024
    kernel_pair_per_ray_bytes = int(len(triangles) * 256)
    affordable_chunk = min(
        chunk,
        max(0, (MAX_WORKING_BYTES - external_live - kernel_fixed_bytes) //
            max(1, wrapper_per_ray_bytes + kernel_pair_per_ray_bytes + 65)),
    )
    if affordable_chunk < 1:
        raise SourceCapacityLimitError("No complete original-source ray chunk fits the 128 MiB working cap")
    chunk = int(affordable_chunk)
    estimated_peak_bytes = int(external_live + kernel_fixed_bytes + chunk *
                               (wrapper_per_ray_bytes + kernel_pair_per_ray_bytes + 65))
    _check_budget(estimated_peak_bytes, "all-original-ID ray/output peak")
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap")

    support_bits = np.zeros(len(ids), dtype=bool)
    hit_bits = np.zeros(len(ids), dtype=bool)
    triangle_ids = np.full(len(ids), -1, dtype=np.int64)
    point_error_m = np.full(len(ids), np.inf, dtype=np.float64)
    producer_error_m = np.full(len(ids), np.inf, dtype=np.float64)
    distance_error_m = np.full(len(ids), np.inf, dtype=np.float64)
    reason_codes = np.full(len(ids), 1, dtype=np.uint8)
    max_ray_scratch_bytes = 0
    used_chunk = 0

    pose_native_mm = pose_native.copy()
    pose_native_mm[:3, 3] *= 1000.0
    pose_crop_mm = c @ pose_native_mm
    rotation_crop = pose_crop_mm[:3, :3]
    translation_crop_m = pose_crop_mm[:3, 3] * 0.001
    origin_object_m = -(rotation_crop.T @ translation_crop_m)
    flat_depth = depth.reshape(-1)

    for start in range(0, len(ids), chunk):
        if time.monotonic() > deadline:
            raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap")
        stop = min(len(ids), start + chunk)
        bridge = crop_source_rays_object(pixels[start:stop], k, pose_native, c)
        origins = np.broadcast_to(origin_object_m, bridge["directions_object_unit"].shape).copy()
        external_chunk_bytes = int(external_live + (stop - start) * wrapper_per_ray_bytes)
        ray = raycast_fn(origins, bridge["directions_object_unit"], vertices, triangles,
                         chunk_size=chunk,
                         resident_bytes=external_chunk_bytes)
        n = stop - start
        hit = np.asarray(ray.get("hit"), dtype=bool).reshape(-1)
        tri = np.asarray(ray.get("triangle_id"), dtype=np.int64).reshape(-1)
        first_hit = np.asarray(ray.get("point_m"), dtype=np.float64).reshape(-1, 3)
        distance = np.asarray(ray.get("distance_m"), dtype=np.float64).reshape(-1)
        if not (len(hit) == len(tri) == len(first_hit) == len(distance) == n):
            raise SourceReservationError("First-hit kernel returned a malformed chunk")
        hit_bits[start:stop] = hit
        triangle_ids[start:stop] = tri
        max_ray_scratch_bytes = max(max_ray_scratch_bytes,
                                    int(ray.get("estimated_scratch_bytes", 0)))
        used_chunk = max(used_chunk, int(ray.get("chunk_size", n)))

        z_mm = np.asarray(flat_depth[ids[start:stop]], dtype=np.float64)
        raw_camera_rays = bridge["crop_rays_unnormalized"]
        expected_object = ((z_mm[:, None] * raw_camera_rays -
                            pose_crop_mm[:3, 3][None, :]) @ rotation_crop) * 0.001
        producer_error = np.linalg.norm(expected_object - points[start:stop], axis=1)
        first_hit_error = np.linalg.norm(first_hit - expected_object, axis=1)
        travel_expected = z_mm * 0.001 * np.linalg.norm(raw_camera_rays, axis=1)
        travel_error = np.abs(distance - travel_expected)
        producer_error_m[start:stop] = producer_error
        point_error_m[start:stop] = np.maximum(first_hit_error, producer_error)
        distance_error_m[start:stop] = travel_error

        invalid_depth = ~np.isfinite(z_mm) | (z_mm <= 0.0)
        invalid_point = ~np.isfinite(points[start:stop]).all(axis=1)
        valid_hit = hit & np.isfinite(first_hit).all(axis=1) & (tri >= 0) & (tri < len(triangles))
        topo_bad = np.zeros(n, dtype=bool)
        if valid_hit.any():
            topo_bad[valid_hit] = unsupported_triangle_mask[tri[valid_hit]]
        good = (valid_hit & ~invalid_depth & ~invalid_point & ~topo_bad &
                np.isfinite(point_error_m[start:stop]) &
                (point_error_m[start:stop] <= epsilon_m) &
                np.isfinite(travel_error) & (travel_error <= epsilon_m))
        support_bits[start:stop] = good

        codes = np.zeros(n, dtype=np.uint8)
        codes[~good] = 2
        codes[~valid_hit] = 3
        codes[invalid_depth] = 1
        codes[~invalid_depth & invalid_point] = 4
        valid_original = valid_hit & ~invalid_depth & ~invalid_point
        codes[valid_original & (point_error_m[start:stop] > epsilon_m)] = 5
        codes[valid_original & (travel_error > epsilon_m)] = 6
        codes[valid_original & topo_bad] = 7
        reason_codes[start:stop] = codes
        del bridge, origins, ray, hit, tri, first_hit, distance, z_mm
        del raw_camera_rays, expected_object, producer_error, first_hit_error
        del travel_expected, travel_error, invalid_depth, invalid_point
        del valid_hit, topo_bad, good, codes, valid_original

    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap")

    reason_names = {
        0: "supported",
        1: "invalid_original_depth",
        2: "unsupported_first_hit",
        3: "no_valid_native_first_hit",
        4: "invalid_original_object_point",
        5: "native_first_hit_point_error_over_epsilon",
        6: "native_first_hit_camera_z_distance_error_over_epsilon",
        7: "native_hit_triangle_topology_unsupported",
    }
    reason_counts = {name: int(np.count_nonzero(reason_codes == code))
                     for code, name in reason_names.items()}
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap while sealing results")
    support_bits_sha = hashlib.sha256(memoryview(support_bits).cast("B")).hexdigest()
    triangle_ids_sha = _array_sha256(triangle_ids.astype("<i8", copy=False))
    point_error_sha = _array_sha256(point_error_m)
    producer_error_sha = _array_sha256(producer_error_m)
    distance_error_sha = _array_sha256(distance_error_m)
    crop_pose_sha = _array_sha256(pose_crop_mm)
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap while hashing outputs")
    association_payload = {
        "schema_version": 2,
        "source_context_id": str(source_context_id),
        "packet_sha256": packet_hash,
        "ordered_source_ids_sha256": _array_sha256(ids.astype("<i8", copy=False)),
        "source_pixel_xy_sha256": _array_sha256(pixels),
        "source_depth_sha256": _array_sha256(depth),
        "source_points_object_m_sha256": _array_sha256(points),
        "crop_from_native_sha256": _array_sha256(c),
        "crop_k_sha256": _array_sha256(k),
        "native_k_sha256": _array_sha256(native_k_value),
        "native_template_pose_m_sha256": _array_sha256(pose_native),
        "derived_crop_pose_mm_sha256": crop_pose_sha,
        "mesh_sha256": mesh_hash,
        "mesh_positions_m_sha256": _array_sha256(vertices),
        "mesh_triangles_sha256": _array_sha256(triangles),
        "triangle_ids_sha256": triangle_ids_sha,
        "point_error_m_sha256": point_error_sha,
        "producer_point_error_m_sha256": producer_error_sha,
        "camera_z_distance_error_m_sha256": distance_error_sha,
        "support_bits_sha256": support_bits_sha,
        "epsilon_m": float(epsilon_m),
        "source_count": int(len(ids)),
        "associated_count": int(support_bits.sum()),
        "unsupported_count": int((~support_bits).sum()),
        "reason_counts": reason_counts,
    }
    association_sha = hashlib.sha256(json.dumps(
        association_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap while sealing provenance")
    record = dict(
        schema_version=2,
        state="computed",
        source_context_id=str(source_context_id),
        packet_sha256=packet_hash,
        mesh_sha256=mesh_hash,
        source_count=int(len(ids)),
        associated_count=int(support_bits.sum()),
        unsupported_count=int((~support_bits).sum()),
        reason_counts=reason_counts,
        epsilon_m=float(epsilon_m),
        geometry=dict(
            crop_from_native_sha256=_array_sha256(c),
            crop_k_sha256=_array_sha256(k),
            native_k_sha256=_array_sha256(native_k_value),
            native_template_pose_m_sha256=_array_sha256(pose_native),
            derived_crop_pose_mm_sha256=crop_pose_sha,
            source_pixels_xy_sha256=_array_sha256(pixels),
            source_depth_mm_sha256=_array_sha256(depth),
            source_points_object_m_sha256=_array_sha256(points),
            mesh_positions_m_sha256=_array_sha256(vertices),
            mesh_triangles_sha256=_array_sha256(triangles),
            native_topology_state=topology_state,
            native_topology_unsupported_triangle_count=unsupported_triangle_count,
            ordered_triangle_ids_sha256=triangle_ids_sha,
            point_error_m_sha256=point_error_sha,
            producer_point_error_m_sha256=producer_error_sha,
            camera_z_distance_error_m_sha256=distance_error_sha,
            support_bits_sha256=support_bits_sha,
            association_sha256=association_sha,
        ),
        raycast_kernel_identity=raycast_kernel_identity,
        raycast_kernel_is_frozen_r7=bool(raycast_kernel_is_frozen),
        ray=dict(chunk_size=int(used_chunk), effective_chunk_size=int(chunk),
                 max_scratch_bytes=int(max_ray_scratch_bytes),
                 requested_chunk_size=int(requested_chunk)),
        resource=dict(
            state="within_budget", estimated_peak_bytes=int(estimated_peak_bytes),
            working_budget_bytes=MAX_WORKING_BYTES,
            elapsed_seconds=float(time.monotonic() - started),
            wall_budget_seconds=MAX_WALL_SECONDS,
        ),
        topology=dict(
            unsupported_triangle_ids_sha256=hashlib.sha256(
                memoryview(unsupported_triangle_ids.astype("<i8", copy=False)).cast("B")
            ).hexdigest(),
            unsupported_triangle_count=unsupported_triangle_count,
        ),
    )
    if time.monotonic() > deadline:
        raise SourceCapacityLimitError("Native source association exceeded the 900-second wall cap before return")
    return dict(
        source_ids=ids.copy(),
        support_bits=support_bits,
        triangle_ids=triangle_ids,
        point_error_m=point_error_m,
        producer_point_error_m=producer_error_m,
        camera_z_distance_error_m=distance_error_m,
        reason_codes=reason_codes,
        reason_names=reason_names,
        record=record,
        **record,
    )
