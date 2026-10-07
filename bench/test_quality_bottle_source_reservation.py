"""Independent CPU tests for R8 source-first capacity reservation only."""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import tempfile
import time
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_joint_prerequisite as r7
from . import quality_bottle_source_reservation as r8
from .vision import cv2


SPEC_V2_SHA256 = "32758dda6ad550a5edd94f5354b7e8401ff206a63b884e8c88ab37f71ee6d9cb"
PERMITTED_TEMPLATE_KEYS = frozenset((
    "template_rgb", "template_gray_rgb", "template_depth_mm", "template_mask",
    "source_indices", "source_pixels_xy", "source_points_object_m",
    "observed_crop_mask", "crop_k", "crop_from_native", "native_k",
    "seed_pose_m", "template_pose_m",
))
FORBIDDEN_SOURCE_KEYS = frozenset((
    "query_rgb", "query_depth_mm", "synthetic_query_mask", "known_query_pose_m",
    "reference_pose", "evaluator_truth", "annotations", "ncc_scores", "matches",
))


def _reference_reflect101(index: int, length: int) -> int:
    if length == 1:
        return 0
    period = 2 * (length - 1)
    value = index % period
    return value if value < length else period - value


def _reference_axis_taps(coordinate: float, length: int) -> tuple[int, ...]:
    lower = math.floor(coordinate)
    fraction = coordinate - lower
    taps = []
    if 1.0 - fraction != 0.0:
        taps.append(lower)
    if fraction != 0.0:
        taps.append(lower + 1)
    if any(index < 0 or index >= length for index in taps):
        raise IndexError("nonzero bilinear tap outside image")
    return tuple(taps)


def _reference_D(center_xy, width: int, height: int) -> tuple[int, ...]:
    """Literal set expansion independent of R7/R8 dependency helpers."""
    q = np.asarray(center_xy, dtype=np.float64).reshape(2)
    raw_x: set[int] = set()
    raw_y: set[int] = set()
    for offset in range(-5, 6):
        x_hi = _reference_axis_taps(float(q[0] - .5 + offset), width)
        y_hi = _reference_axis_taps(float(q[1] - .5 + offset), height)
        for highpass_x in x_hi:
            raw_x.update(_reference_reflect101(highpass_x + delta, width)
                         for delta in range(-8, 9))
        for highpass_y in y_hi:
            raw_y.update(_reference_reflect101(highpass_y + delta, height)
                         for delta in range(-8, 9))
    return tuple(sorted(y * width + x for y in raw_y for x in raw_x))


def _rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.asarray(((1., 0., 0.), (0., cx, -sx), (0., sx, cx)))
    rym = np.asarray(((cy, 0., sy), (0., 1., 0.), (-sy, 0., cy)))
    rzm = np.asarray(((cz, -sz, 0.), (sz, cz, 0.), (0., 0., 1.)))
    return rzm @ rym @ rxm


def _reference_crop_pose_mm(native_pose_m: np.ndarray,
                            crop_from_native: np.ndarray) -> np.ndarray:
    pose_mm = np.asarray(native_pose_m, dtype=np.float64).copy()
    pose_mm[:3, 3] *= 1000.0
    return np.asarray(crop_from_native, dtype=np.float64) @ pose_mm


def _reference_producer_point(source_xy, source_depth_mm: float, crop_k: np.ndarray,
                              native_pose_m: np.ndarray,
                              crop_from_native: np.ndarray) -> np.ndarray:
    crop_pose_mm = _reference_crop_pose_mm(native_pose_m, crop_from_native)
    r, t_mm = crop_pose_mm[:3, :3], crop_pose_mm[:3, 3]
    d_crop = np.linalg.inv(np.asarray(crop_k, dtype=np.float64)) @ np.asarray(
        (float(source_xy[0]), float(source_xy[1]), 1.0), dtype=np.float64)
    return ((float(source_depth_mm) * d_crop - t_mm) @ r) * .001


def _reference_crop_object_ray(source_xy, crop_k: np.ndarray,
                               native_pose_m: np.ndarray,
                               crop_from_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crop_pose_mm = _reference_crop_pose_mm(native_pose_m, crop_from_native)
    r, t_mm = crop_pose_mm[:3, :3], crop_pose_mm[:3, 3]
    d_crop = np.linalg.inv(np.asarray(crop_k, dtype=np.float64)) @ np.asarray(
        (float(source_xy[0]), float(source_xy[1]), 1.0), dtype=np.float64)
    origin = -r.T @ (t_mm * .001)
    direction = r.T @ d_crop
    direction /= np.linalg.norm(direction)
    return origin, direction


def _reference_ray_triangle(origin, direction, vertices, triangles,
                            tie_tolerance_m: float = 1e-12):
    """Small independent two-sided first-hit oracle for the geometry fixture."""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    d = d / np.linalg.norm(d)
    vtx = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    hits = []
    for triangle_id, face in enumerate(faces):
        v0, v1, v2 = vtx[face]
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(d, e2)
        det = float(np.dot(e1, pvec))
        if abs(det) <= 1e-12:
            continue
        inv_det = 1.0 / det
        tvec = o - v0
        u = float(np.dot(tvec, pvec) * inv_det)
        qvec = np.cross(tvec, e1)
        v = float(np.dot(d, qvec) * inv_det)
        distance = float(np.dot(e2, qvec) * inv_det)
        if (u >= -1e-12 and v >= -1e-12 and u + v <= 1.0 + 1e-12 and
                distance > 0.0 and math.isfinite(distance)):
            hits.append((distance, triangle_id))
    if not hits:
        return None
    nearest = min(distance for distance, _ in hits)
    triangle_id = min(tid for distance, tid in hits
                      if distance <= nearest + tie_tolerance_m)
    distance = next(distance for distance, tid in hits if tid == triangle_id)
    return dict(distance_m=distance, triangle_id=triangle_id,
                point_m=o + distance * d)


def _project_native_object_point(point_m, native_pose_m, crop_from_native, crop_k):
    crop_pose_mm = _reference_crop_pose_mm(native_pose_m, crop_from_native)
    camera_mm = crop_pose_mm[:3, :3] @ (np.asarray(point_m, dtype=np.float64) * 1000.0) + crop_pose_mm[:3, 3]
    homogeneous = np.asarray(crop_k, dtype=np.float64) @ camera_mm
    return homogeneous[:2] / homogeneous[2], float(camera_mm[2])


def _source_geometry_fixture():
    width, height = 320, 200
    native_pose = np.eye(4, dtype=np.float64)
    native_pose[:3, :3] = _rotation_xyz(math.radians(5.0), math.radians(-11.0),
                                        math.radians(4.0))
    native_pose[:3, 3] = (.04, -.02, .8)
    crop_from_native = np.eye(4, dtype=np.float64)
    crop_from_native[:3, :3] = _rotation_xyz(0.0, 0.0, math.radians(17.0))
    crop_from_native[:3, 3] = (31.0, -18.0, 9.0)  # millimetres
    crop_k = np.asarray(((713.0, 0.0, 132.25),
                         (0.0, 589.0, 87.75),
                         (0.0, 0.0, 1.0)), dtype=np.float64)
    native_k = np.asarray(((521.0, 0.0, 161.3),
                           (0.0, 947.0, 96.2),
                           (0.0, 0.0, 1.0)), dtype=np.float64)
    pixels = np.asarray(((173.5, 101.5), (177.5, 103.5)), dtype=np.float64)
    plane_z = .21
    origins, directions, exact_hits, depths_mm, stored_points = [], [], [], [], []
    for pixel in pixels:
        origin, direction = _reference_crop_object_ray(
            pixel, crop_k, native_pose, crop_from_native)
        distance = (plane_z - float(origin[2])) / float(direction[2])
        point = origin + distance * direction
        _, depth_mm = _project_native_object_point(
            point, native_pose, crop_from_native, crop_k)
        stored_depth_mm = float(np.float32(depth_mm))
        producer_point = _reference_producer_point(
            pixel, stored_depth_mm, crop_k, native_pose, crop_from_native)
        origins.append(origin)
        directions.append(direction)
        exact_hits.append(point)
        depths_mm.append(stored_depth_mm)
        stored_points.append(producer_point)
    origins = np.asarray(origins)
    directions = np.asarray(directions)
    exact_hits = np.asarray(exact_hits)
    depths_mm = np.asarray(depths_mm, dtype=np.float64)
    stored_points = np.asarray(stored_points)
    # Second candidate has internally consistent pixel/depth provenance but a
    # deliberately wrong native object point; it must still be raycast and
    # counted before later source selection can exclude it.
    stored_points[1] += np.asarray((.02, 0.0, 0.0))
    point = exact_hits[0]
    pixel = pixels[0]
    front = np.asarray((point + (-.3, -.3, 0.0),
                        point + (.3, -.3, 0.0),
                        point + (0.0, .3, 0.0)), dtype=np.float64)
    back = front + np.asarray((0.0, 0.0, .15), dtype=np.float64)
    vertices = np.vstack((front, back))
    triangles = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    source_ids = np.asarray([int(math.floor(q[1]) * width + math.floor(q[0]))
                             for q in pixels], dtype=np.int64)
    # The projected point is chosen from a pixel-center ray, so this equality
    # also checks the frozen flat source-ID/pixel binding.
    for row, q in enumerate(pixels):
        pixel_xy, projected_depth = _project_native_object_point(
            exact_hits[row], native_pose, crop_from_native, crop_k)
        np.testing.assert_allclose(pixel_xy, q, atol=1e-12, rtol=0.0)
        reconstructed = _reference_producer_point(
            q, depths_mm[row], crop_k, native_pose, crop_from_native)
        np.testing.assert_allclose(reconstructed, stored_points[row] -
                                   ((np.asarray((.02, 0., 0.)) if row == 1 else 0.)),
                                   atol=1e-12, rtol=1e-12)
        assert abs(projected_depth - depths_mm[row]) < 1e-3
    depth = np.zeros((height, width), dtype=np.float32)
    depth.reshape(-1)[source_ids] = depths_mm.astype(np.float32)
    return dict(width=width, height=height, native_pose=native_pose,
                crop_from_native=crop_from_native, crop_k=crop_k,
                native_k=native_k, pixels=pixels, pixel=pixel, point=point,
                points=stored_points, depth_mm=depths_mm[0], depth=depth,
                source_ids=source_ids, source_id=int(source_ids[0]),
                exact_hits=exact_hits, depths_mm=depths_mm,
                vertices=vertices, triangles=triangles)


class PoisonMapping(dict):
    """A dict that records and rejects reads of query/evaluator-only inputs."""
    def __init__(self, *args, forbidden=FORBIDDEN_SOURCE_KEYS, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden = set(forbidden)
        self.forbidden_reads = []

    def __getitem__(self, key):
        if key in self.forbidden:
            self.forbidden_reads.append(key)
            raise AssertionError(f"forbidden source-capacity read: {key}")
        return super().__getitem__(key)


class FrozenSpecTests(unittest.TestCase):
    def test_v2_spec_hash_is_pinned(self):
        spec = Path(__file__).resolve().parents[1] / ".cache" / \
            "bottle-source-reservation-r8-spec-v2.md"
        digest = hashlib.sha256(spec.read_bytes()).hexdigest()
        self.assertEqual(digest, SPEC_V2_SHA256)


class CropNativeAssociationTests(unittest.TestCase):
    def test_actual_all_id_association_uses_producer_crop_bridge(self):
        fixture = _source_geometry_fixture()
        source_ids = fixture["source_ids"]
        pixels = fixture["pixels"]
        points = fixture["points"]
        calls = []

        def recorded_raycast(origins, directions, vertices, triangles, **kwargs):
            calls.append((np.asarray(origins).copy(), np.asarray(directions).copy()))
            return r7.raycast_first_hits(origins, directions, vertices, triangles, **kwargs)

        result = r8.native_source_association_all(
            source_ids, pixels, fixture["depth"], points,
            crop_k=fixture["crop_k"], crop_from_native=fixture["crop_from_native"],
            native_k=fixture["native_k"], template_pose_m=fixture["native_pose"],
            mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
            model_diagonal_m=.6, source_context_id="template-geometry-fixture",
            packet_sha256="a" * 64, mesh_sha256="b" * 64,
            raycast_fn=recorded_raycast, chunk_size=1)

        self.assertEqual(len(calls), 2, "chunk_size=1 must process every original source ID")
        for row, pixel in enumerate(fixture["pixels"]):
            expected_origin, expected_direction = _reference_crop_object_ray(
                pixel, fixture["crop_k"], fixture["native_pose"],
                fixture["crop_from_native"])
            np.testing.assert_allclose(calls[row][0], expected_origin[None, :],
                                       atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(calls[row][1], expected_direction[None, :],
                                       atol=1e-12, rtol=1e-12)
        expected_origin, expected_direction = _reference_crop_object_ray(
            fixture["pixel"], fixture["crop_k"], fixture["native_pose"],
            fixture["crop_from_native"])
        oracle = _reference_ray_triangle(expected_origin, expected_direction,
                                         fixture["vertices"], fixture["triangles"])
        self.assertIsNotNone(oracle)
        self.assertEqual(oracle["triangle_id"], 0)
        np.testing.assert_allclose(oracle["point_m"], fixture["exact_hits"][0],
                                   atol=1e-12, rtol=1e-12)
        self.assertEqual(result["state"], "computed")
        self.assertEqual(result["source_count"], 2)
        self.assertEqual(result["associated_count"], 1)
        self.assertEqual(result["unsupported_count"], 1)
        np.testing.assert_array_equal(result["source_ids"], source_ids)
        np.testing.assert_array_equal(result["support_bits"], (True, False))
        np.testing.assert_array_equal(result["triangle_ids"], (0, 0))
        self.assertLessEqual(float(result["point_error_m"][0]), result["epsilon_m"])
        self.assertLessEqual(float(result["producer_point_error_m"][0]), result["epsilon_m"])
        self.assertGreater(float(result["point_error_m"][1]), result["epsilon_m"])
        self.assertLessEqual(float(result["camera_z_distance_error_m"][0]),
                             result["epsilon_m"])
        self.assertEqual(result["ray"]["chunk_size"], 1)
        self.assertEqual(result["ray"]["requested_chunk_size"], 1)

        # Camera-z depth is not Euclidean ray travel at this off-axis pixel.
        crop_direction = np.linalg.inv(fixture["crop_k"]) @ np.asarray(
            (*fixture["pixel"], 1.0), dtype=np.float64)
        expected_distance = fixture["depth_mm"] * .001 * np.linalg.norm(crop_direction)
        self.assertGreater(abs(expected_distance - fixture["depth_mm"] * .001), 1e-3)
        self.assertAlmostEqual(float(oracle["distance_m"]), float(expected_distance), delta=1e-7)

        # The prior shortcut (crop K ray applied directly to the native pose)
        # must miss the same native surface by far more than the frozen epsilon.
        native_r = fixture["native_pose"][:3, :3]
        native_t = fixture["native_pose"][:3, 3]
        wrong_origin = -native_r.T @ native_t
        wrong_direction = native_r.T @ crop_direction
        wrong_direction /= np.linalg.norm(wrong_direction)
        wrong_hit = _reference_ray_triangle(wrong_origin, wrong_direction,
                                            fixture["vertices"], fixture["triangles"])
        self.assertIsNotNone(wrong_hit)
        self.assertGreater(float(np.linalg.norm(wrong_hit["point_m"] - fixture["exact_hits"][0])),
                           result["epsilon_m"] * 10)

    def test_mismatched_crop_transform_and_flat_id_pixel_binding_fail_closed(self):
        fixture = _source_geometry_fixture()
        source_ids = fixture["source_ids"]
        pixels = fixture["pixels"]
        points = fixture["points"]
        calls = []

        def recorded_raycast(origins, directions, vertices, triangles, **kwargs):
            calls.append(1)
            return r7.raycast_first_hits(origins, directions, vertices, triangles, **kwargs)

        common = dict(
            native_k=fixture["native_k"], template_pose_m=fixture["native_pose"],
            mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
            model_diagonal_m=.6, source_context_id="template-geometry-fixture",
            packet_sha256="a" * 64, mesh_sha256="b" * 64,
            raycast_fn=recorded_raycast, chunk_size=1)
        wrong_c = fixture["crop_from_native"].copy()
        wrong_c[:3, 3] += (20.0, -10.0, 7.0)
        mismatched = r8.native_source_association_all(
            source_ids, pixels, fixture["depth"], points,
            crop_k=fixture["crop_k"], crop_from_native=wrong_c, **common)
        self.assertEqual(mismatched["source_count"], 2)
        self.assertEqual(mismatched["associated_count"], 0)
        self.assertEqual(mismatched["unsupported_count"], 2)
        self.assertFalse(bool(mismatched["support_bits"][0]))

        call_count_before_bad_binding = len(calls)
        with self.assertRaises(r8.SourceReservationError):
            r8.native_source_association_all(
                source_ids + 1, pixels, fixture["depth"], points,
                crop_k=fixture["crop_k"], crop_from_native=fixture["crop_from_native"],
                **common)
        self.assertEqual(len(calls), call_count_before_bad_binding,
                         "flat source ID/pixel mismatch must fail before raycasting")

    def test_association_bits_and_seal_are_chunk_invariant_for_all_original_ids(self):
        fixture = _source_geometry_fixture()

        def run(chunk_size):
            ray_batches = []

            def recorded_raycast(origins, directions, vertices, triangles, **kwargs):
                ray_batches.append(len(origins))
                return r7.raycast_first_hits(origins, directions, vertices, triangles,
                                             **kwargs)

            result = r8.native_source_association_all(
                fixture["source_ids"], fixture["pixels"], fixture["depth"],
                fixture["points"], crop_k=fixture["crop_k"],
                crop_from_native=fixture["crop_from_native"], native_k=fixture["native_k"],
                template_pose_m=fixture["native_pose"],
                mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
                model_diagonal_m=.6, source_context_id="template-geometry-fixture",
                packet_sha256="a" * 64, mesh_sha256="b" * 64,
                raycast_fn=recorded_raycast, chunk_size=chunk_size)
            return result, ray_batches

        single, single_batches = run(1)
        pair, pair_batches = run(2)
        self.assertEqual(single_batches, [1, 1])
        self.assertEqual(pair_batches, [2])
        np.testing.assert_array_equal(single["source_ids"], pair["source_ids"])
        np.testing.assert_array_equal(single["support_bits"], pair["support_bits"])
        np.testing.assert_array_equal(single["triangle_ids"], pair["triangle_ids"])
        np.testing.assert_allclose(single["point_error_m"], pair["point_error_m"],
                                   atol=0.0, rtol=0.0, equal_nan=True)
        np.testing.assert_allclose(single["camera_z_distance_error_m"],
                                   pair["camera_z_distance_error_m"],
                                   atol=0.0, rtol=0.0, equal_nan=True)
        self.assertEqual(single["geometry"]["association_sha256"],
                         pair["geometry"]["association_sha256"])
        self.assertEqual(single["source_count"], len(fixture["source_ids"]))
        self.assertEqual(pair["source_count"], len(fixture["source_ids"]))

    def test_zero_and_nan_original_depths_remain_counted_as_unsupported(self):
        fixture = _source_geometry_fixture()
        depth = fixture["depth"].copy()
        flat = depth.reshape(-1)
        flat[int(fixture["source_ids"][0])] = 0.0
        flat[int(fixture["source_ids"][1])] = np.nan
        calls = []

        def recorded_raycast(origins, directions, vertices, triangles, **kwargs):
            calls.append(len(origins))
            return r7.raycast_first_hits(origins, directions, vertices, triangles,
                                         **kwargs)

        result = r8.native_source_association_all(
            fixture["source_ids"], fixture["pixels"], depth, fixture["points"],
            crop_k=fixture["crop_k"], crop_from_native=fixture["crop_from_native"],
            native_k=fixture["native_k"], template_pose_m=fixture["native_pose"],
            mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
            model_diagonal_m=.6, source_context_id="template-invalid-depth-fixture",
            packet_sha256="c" * 64, mesh_sha256="d" * 64,
            raycast_fn=recorded_raycast, chunk_size=1)
        self.assertEqual(calls, [1, 1])
        self.assertEqual(result["source_count"], 2)
        self.assertEqual(result["associated_count"], 0)
        self.assertEqual(result["unsupported_count"], 2)
        np.testing.assert_array_equal(result["support_bits"], (False, False))
        self.assertEqual(result["reason_names"][int(result["reason_codes"][0])],
                         "invalid_original_depth")
        self.assertEqual(result["reason_names"][int(result["reason_codes"][1])],
                         "invalid_original_depth")

    def test_topology_unsupported_first_hit_is_not_accepted(self):
        fixture = _source_geometry_fixture()
        # Triangles 0 and 1 share the geometric edge 0→1 in the same
        # direction. Their native topology is ambiguous even though ray 0
        # still intersects the interior of triangle 0.
        inconsistent = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int64)
        result = r8.native_source_association_all(
            fixture["source_ids"][:1], fixture["pixels"][:1], fixture["depth"],
            fixture["points"][:1], crop_k=fixture["crop_k"],
            crop_from_native=fixture["crop_from_native"], native_k=fixture["native_k"],
            template_pose_m=fixture["native_pose"],
            mesh_positions_m=fixture["vertices"], mesh_triangles=inconsistent,
            model_diagonal_m=.6, source_context_id="template-topology-fixture",
            packet_sha256="e" * 64, mesh_sha256="f" * 64,
            raycast_fn=r7.raycast_first_hits, chunk_size=1)
        self.assertEqual(result["triangle_ids"].tolist(), [0])
        self.assertEqual(result["support_bits"].tolist(), [False])
        self.assertEqual(result["reason_names"][int(result["reason_codes"][0])],
                         "native_hit_triangle_topology_unsupported")
        self.assertGreater(result["topology"]["unsupported_triangle_count"], 0)

    def test_over_cap_source_association_fails_before_topology_or_ray_output_allocation(self):
        fixture = _source_geometry_fixture()
        calls = []

        def must_not_raycast(*args, **kwargs):
            calls.append("raycast")
            raise AssertionError("over-cap source pass reached the ray kernel")

        # Independent lower bound includes the already-resident packet and the
        # immutable 128 MiB cap, so all input/output/topology/scratch bytes put
        # the request strictly over cap without allocating large arrays.
        self.assertEqual(r8.MAX_WORKING_BYTES, 128 * 1024**2)
        with mock.patch.object(r7, "native_mesh_topology") as topology_spy, \
             mock.patch.object(r8.np, "zeros", side_effect=AssertionError(
                 "over-cap pass allocated association outputs")) as zeros_spy:
            with self.assertRaises(r8.SourceCapacityLimitError):
                r8.native_source_association_all(
                    fixture["source_ids"], fixture["pixels"], fixture["depth"],
                    fixture["points"], crop_k=fixture["crop_k"],
                    crop_from_native=fixture["crop_from_native"], native_k=fixture["native_k"],
                    template_pose_m=fixture["native_pose"],
                    mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
                    model_diagonal_m=.6, source_context_id="template-budget-fixture",
                    packet_sha256="1" * 64, mesh_sha256="2" * 64,
                    raycast_fn=must_not_raycast, chunk_size=1,
                    resident_bytes=r8.MAX_WORKING_BYTES)
        self.assertEqual(calls, [])
        topology_spy.assert_not_called()
        zeros_spy.assert_not_called()

    def test_expired_nonfinite_and_negative_resource_contracts_fail_before_raycast(self):
        fixture = _source_geometry_fixture()
        calls = []

        def must_not_raycast(*args, **kwargs):
            calls.append("raycast")
            raise AssertionError("invalid deadline/resource contract reached raycasting")

        common = dict(
            source_ids=fixture["source_ids"], source_pixels_xy=fixture["pixels"],
            source_depth_mm=fixture["depth"], source_points_object_m=fixture["points"],
            crop_k=fixture["crop_k"], crop_from_native=fixture["crop_from_native"],
            native_k=fixture["native_k"], template_pose_m=fixture["native_pose"],
            mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
            model_diagonal_m=.6, source_context_id="template-invalid-budget-fixture",
            packet_sha256="3" * 64, mesh_sha256="4" * 64,
            raycast_fn=must_not_raycast, chunk_size=2)
        for options, error in (
                ({"deadline_monotonic": time.monotonic() - 1.0},
                 r8.SourceCapacityLimitError),
                ({"deadline_monotonic": float("nan")}, r8.SourceReservationError),
                ({"resident_bytes": -1}, r8.SourceReservationError)):
            with self.subTest(options=options):
                with self.assertRaises(error):
                    r8.native_source_association_all(**common, **options)
        self.assertEqual(calls, [])

    def test_far_future_deadline_cannot_extend_the_frozen_900_second_cap(self):
        fixture = _source_geometry_fixture()
        calls = []

        def must_not_raycast(*args, **kwargs):
            calls.append("raycast")
            raise AssertionError("hard deadline overrun reached raycasting")

        # The helper receives a far-future caller deadline. Inject a monotonic
        # clock at t=1000, then t=1901: the hard deadline is 1900, regardless
        # of the caller's 1e6-second request.
        with mock.patch.object(r8.time, "monotonic", side_effect=(1000.0, 1901.0)), \
             mock.patch.object(r7, "native_mesh_topology") as topology_spy:
            with self.assertRaises(r8.SourceCapacityLimitError):
                r8.native_source_association_all(
                    fixture["source_ids"], fixture["pixels"], fixture["depth"],
                    fixture["points"], crop_k=fixture["crop_k"],
                    crop_from_native=fixture["crop_from_native"], native_k=fixture["native_k"],
                    template_pose_m=fixture["native_pose"],
                    mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
                    model_diagonal_m=.6, source_context_id="template-hard-deadline-fixture",
                    packet_sha256="5" * 64, mesh_sha256="6" * 64,
                    raycast_fn=must_not_raycast, chunk_size=1,
                    deadline_monotonic=1_000_000.0)
        topology_spy.assert_not_called()
        self.assertEqual(calls, [])

    def test_adaptive_chunk_reduces_from_256_without_losing_original_ids(self):
        fixture = _source_geometry_fixture()
        default = r8.native_source_association_all(
            fixture["source_ids"], fixture["pixels"], fixture["depth"],
            fixture["points"], crop_k=fixture["crop_k"],
            crop_from_native=fixture["crop_from_native"], native_k=fixture["native_k"],
            template_pose_m=fixture["native_pose"],
            mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
            model_diagonal_m=.6, source_context_id="template-adaptive-chunk-fixture",
            packet_sha256="7" * 64, mesh_sha256="8" * 64,
            raycast_fn=r7.raycast_first_hits, chunk_size=2)
        calls = []

        def recorded_raycast(origins, directions, vertices, triangles, **kwargs):
            calls.append((len(origins), int(kwargs["chunk_size"])))
            return r7.raycast_first_hits(origins, directions, vertices, triangles,
                                         **kwargs)

        # The R8 frozen preflight/seal reserve is max(8*N, 256 KiB)+64 KiB;
        # together with this fixture it leaves room for one ray but not two.
        # This temporary cap does not change the production 128 MiB constant.
        test_cap = 592_000
        seal_reserve = max(8 * len(fixture["source_ids"]), 256 * 1024) + 64 * 1024
        self.assertEqual(seal_reserve, 327_680)
        self.assertLess(test_cap, r8.MAX_WORKING_BYTES)
        with mock.patch.object(r8, "MAX_WORKING_BYTES", test_cap):
            adapted = r8.native_source_association_all(
                fixture["source_ids"], fixture["pixels"], fixture["depth"],
                fixture["points"], crop_k=fixture["crop_k"],
                crop_from_native=fixture["crop_from_native"], native_k=fixture["native_k"],
                template_pose_m=fixture["native_pose"],
                mesh_positions_m=fixture["vertices"], mesh_triangles=fixture["triangles"],
                model_diagonal_m=.6, source_context_id="template-adaptive-chunk-fixture",
                packet_sha256="7" * 64, mesh_sha256="8" * 64,
                raycast_fn=recorded_raycast, chunk_size=256)
        self.assertEqual(adapted["ray"]["chunk_size"], 1)
        self.assertLess(adapted["ray"]["chunk_size"], 256)
        self.assertEqual(calls, [(1, 1), (1, 1)])
        self.assertEqual(adapted["source_count"], len(fixture["source_ids"]))
        np.testing.assert_array_equal(adapted["source_ids"], fixture["source_ids"])
        np.testing.assert_array_equal(adapted["support_bits"], default["support_bits"])
        np.testing.assert_array_equal(adapted["triangle_ids"], default["triangle_ids"])
        self.assertLessEqual(adapted["resource"]["estimated_peak_bytes"], test_cap)


if __name__ == "__main__":
    unittest.main()
