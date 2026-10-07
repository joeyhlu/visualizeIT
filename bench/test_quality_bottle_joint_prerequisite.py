"""CPU regressions for the frozen R7 Stage A1 prerequisite.

These tests exercise geometry, indexing and fail-closed behavior only.  They do
not load the actual R1/R5/R6 cache, write an A1 manifest, or run a fit.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from . import quality_bottle_identity_audit as r5_audit
from . import quality_bottle_joint_prerequisite as joint
from .vision import cv2


# Populated from the sealed Stage A1 procedural declaration run.  A source,
# texture, mask, mesh, pose, seed or truth change requires a reviewed new hash.
FROZEN_HELPER_SHA256 = "cf74fdf73a2e6f36a1ae4bf783091ac23e4b2fd5364ce5b5e721304cb7e1684a"
FROZEN_R5_AUDIT_SHA256 = "954d4b4e3a460187963894988cce53a85a077abaf07e085a0833131d14418f6a"
FROZEN_CONTROL_RECORD_SHA256: dict[str, str] = {
    "distinct_markers-seed-000": "3f7b921b5f98256c287d43f85e89658398fb59027b2c23a538a6457b430b216b",
    "distinct_markers-seed-180": "91cf3631070fe532c4d1478398490a640b9269f9d5512314ef27d717281773d8",
    "blank_constant-seed-000": "7214450ccbf493477504b6f23a4657e42279e1d375fe2d2692e7c0f06acce128",
    "blank_constant-seed-180": "79c063d034de79ef5c90a1a364fbccbfad5f0d5d71469b0f06829a1838e0c726",
    "repeated_2d_periodic-seed-000": "4696d1d80ba949e672880878982ab7335bff7c3d24bbe20d99bc2bcfaaec96e1",
    "repeated_2d_periodic-seed-180": "03b1ae8a55a416a61ea088d94aac20ae3ea06e6fca4e27c76e2cfb36e301b64a",
    "coherent_wrong_180_packet-seed-000": "0b519d33a1a82ee7798a6b5f8e400d694bda093f80f2a95eea04089e35deb849",
    "coherent_wrong_180_packet-seed-180": "659e0ee27272a85a9329a7685f89b2fe154ed28b1374c902563229aaeafb1c9d",
    "query_occlusion-seed-000": "451bdc3120b309ced15804b87da13668aaa3b08c1a02eb28e3a373ae1862e739",
    "query_occlusion-seed-180": "adee558b850782a9e6a8156e817d1224ca05371259c93f481249698dafe00a15",
    "mask_leakage_surrogate-seed-000": "77de2384ed5eaa43e33d8b564ef54f0c3d5e1754d31479cb2e7babebf8d98ea6",
    "mask_leakage_surrogate-seed-180": "785b743253dfd563289214503c225e3ff176f292d191b3275b78022487c6fe64",
    "global_affine-seed-000": "26492ce4437135e991dad127aa746dedb837cefcf304662e9e8e61b63449dd36",
    "global_affine-seed-180": "72e4847fa7e41b159415fca417b2395851fabe7a9f67f588c8af3752fa2cf101",
    "appearance_mismatch-seed-000": "3be9f71094a40529782f99ffaef4f86a69bb9aa83445e0b9ea1f06baf493897b",
    "appearance_mismatch-seed-180": "304e3b8e801da4a4de1213905e712253ce511e94150d712c90047b2ba6108b5c",
}


def _reference_reflect101(index: int, length: int) -> int:
    if length == 1:
        return 0
    period = 2 * (length - 1)
    value = index % period
    return value if value < length else period - value


def _reference_taps(coordinate: float, length: int) -> tuple[int, ...]:
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


def _reference_raw_dependencies(center_xy, width: int, height: int) -> tuple[int, ...]:
    """Literal set expansion of all nonzero descriptor and Gaussian taps."""
    q = np.asarray(center_xy, dtype=np.float64)
    raw_x: set[int] = set()
    raw_y: set[int] = set()
    for offset in range(-5, 6):
        x_hi = _reference_taps(float(q[0] - .5 + offset), width)
        y_hi = _reference_taps(float(q[1] - .5 + offset), height)
        for highpass_x in x_hi:
            raw_x.update(_reference_reflect101(highpass_x + delta, width)
                         for delta in range(-8, 9))
        for highpass_y in y_hi:
            raw_y.update(_reference_reflect101(highpass_y + delta, height)
                         for delta in range(-8, 9))
    return tuple(sorted(y * width + x for y in raw_y for x in raw_x))


def _ordered_values(record_or_values):
    if isinstance(record_or_values, dict):
        return list(record_or_values.get("ordered_ids", []))
    return list(record_or_values)


def _reference_pose_y(degrees: float, translation=(0., 0., 1.)) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = ((c, 0., s), (0., 1., 0.), (-s, 0., c))
    pose[:3, 3] = translation
    return pose


class FootprintAndPartitionTests(unittest.TestCase):
    def test_implementation_source_matches_frozen_helper_hash(self):
        self.assertEqual(joint.file_sha256(Path(joint.__file__)), FROZEN_HELPER_SHA256)
        self.assertEqual(joint.file_sha256(Path(r5_audit.__file__)), FROZEN_R5_AUDIT_SHA256)

    def test_reflect101_and_full_nonzero_descriptor_dependencies(self):
        self.assertEqual([joint.reflect101(index, 1) for index in range(-9, 10)],
                         [0] * 19)
        self.assertEqual([joint.reflect101(index, 4) for index in range(-5, 10)],
                         [_reference_reflect101(index, 4) for index in range(-5, 10)])
        self.assertEqual(joint.gaussian_raw_axis_dependencies(0, 12),
                         tuple(sorted({_reference_reflect101(i, 12)
                                       for i in range(-8, 9)})))

        width, height = 41, 43
        for center in ((20.75, 21.25), (12.5, 13.5)):
            expected = _reference_raw_dependencies(center, width, height)
            actual = joint.highpass_patch_raw_dependencies(center, width, height)
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), len(set(actual)))

    def test_bilinear_zero_weight_taps_are_omitted_and_nonzero_oob_taps_fail(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5)
        allowed = np.zeros((5, 5), dtype=bool)
        allowed[2, 3] = True

        # Edge-coordinate center (3.5, 2.5) maps exactly to array [2, 3].
        # Its zero-weight right/bottom neighbors do not become reads.
        sampled = joint.guarded_bilinear_sample(image, (3.5, 2.5), allowed)
        self.assertEqual(float(sampled), float(image[2, 3]))
        self.assertEqual(joint.bilinear_raw_dependencies((3.5, 2.5), 5, 5),
                         (2 * 5 + 3,))

        poison = image.copy()
        poison[:, :2] = np.nan
        with self.assertRaises(joint.DependencyViolation):
            joint.guarded_bilinear_sample(poison, (1.25, 2.5), allowed)
        with self.assertRaises(joint.DependencyViolation):
            joint.guarded_bilinear_sample(image, (.25, 2.5), np.ones((5, 5), bool))
        with self.assertRaises(joint.DependencyViolation):
            joint.guarded_bilinear_sample(image, (float("nan"), 2.5), np.ones((5, 5), bool))

    def test_alternatives_are_rejected_before_reader_and_oob_is_terminally_counted(self):
        image = np.zeros((80, 80, 3), dtype=np.float32)
        allowed = np.ones((80, 80), dtype=bool)
        allowed[40, 40] = False
        good = (20.5, 20.5)
        crossing = (40.5, 40.5)
        outside = (5.25, 20.5)
        calls = []

        def reader(rgb, dependencies):
            calls.append(tuple(dependencies))
            return "read"

        outcomes = joint.guarded_alternative_reads(
            (good, crossing, outside), image, allowed, reader)
        self.assertEqual([row["state"] for row in outcomes],
                         ["read_after_dependency_check", "rejected_before_read",
                          "rejected_before_read"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(allowed.reshape(-1)[list(calls[0])]))

    def test_sanitization_preserves_legal_filtered_values_and_ignores_forbidden_mutations(self):
        rng = np.random.default_rng(7741)
        rgb = rng.random((80, 84, 3), dtype=np.float32)
        allowed = np.ones(rgb.shape[:2], dtype=bool)
        allowed[:8, :12] = False
        center = (42.5, 39.5)
        dependencies = joint.highpass_patch_raw_dependencies(center, rgb.shape[1], rgb.shape[0])
        self.assertTrue(allowed.reshape(-1)[list(dependencies)].all())

        changed = rgb.copy()
        changed[~allowed] = np.float32(9.0)
        sanitized = joint.partition_sanitized_image(rgb, allowed)
        sanitized_changed = joint.partition_sanitized_image(changed, allowed)
        np.testing.assert_array_equal(sanitized[allowed], rgb[allowed])
        self.assertFalse(np.any(sanitized[~allowed]))
        np.testing.assert_array_equal(sanitized, sanitized_changed)

        def highpass(value):
            gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY).astype(np.float32)
            blur = cv2.GaussianBlur(gray, (17, 17), 2.0,
                                    borderType=cv2.BORDER_REFLECT_101)
            return gray - blur

        full = highpass(rgb)
        full_changed = highpass(changed)
        filtered_sanitized = highpass(sanitized)
        filtered_sanitized_changed = highpass(sanitized_changed)
        patch_full = joint.sample_highpass_patch(full, center, allowed)
        patch_full_changed = joint.sample_highpass_patch(full_changed, center, allowed)
        patch_sanitized = joint.sample_highpass_patch(filtered_sanitized, center, allowed)
        patch_sanitized_changed = joint.sample_highpass_patch(filtered_sanitized_changed,
                                                               center, allowed)
        np.testing.assert_array_equal(patch_full, patch_full_changed)
        np.testing.assert_array_equal(patch_full, patch_sanitized)
        np.testing.assert_array_equal(patch_sanitized, patch_sanitized_changed)

    def test_opencv_sigma_two_path_has_exact_explicit_17_by_17_parity(self):
        kernel = cv2.getGaussianKernel(17, 2.0, cv2.CV_32F).reshape(-1)
        np.testing.assert_array_equal(joint.gaussian_kernel_1d(), kernel)
        record = joint.gaussian_blur_kernel_record()
        self.assertEqual(record["size"], [17, 17])
        self.assertEqual(record["border"], "BORDER_REFLECT_101")
        self.assertEqual(record["coefficients_2d_nonzero"], 17 * 17)

        rng = np.random.default_rng(449)
        image = rng.random((47, 51), dtype=np.float32)
        edge_impulse = np.zeros_like(image)
        edge_impulse[0, 0] = 1.0
        center_impulse = np.zeros_like(image)
        center_impulse[23, 25] = 1.0
        for sample in (image, edge_impulse, center_impulse):
            automatic = cv2.GaussianBlur(sample, (0, 0), 2.0)
            explicit = cv2.GaussianBlur(sample, (17, 17), 2.0,
                                        borderType=cv2.BORDER_REFLECT_101)
            np.testing.assert_array_equal(automatic, explicit)
        self.assertEqual(joint._opencv_gaussian_path_audit(cv2)["state"], "passed")

    def test_legacy_mask_uses_all_four_bilinear_neighbors_including_zero_weight(self):
        observed = np.ones((48, 48), dtype=bool)
        partition = np.ones_like(observed)
        eroded = np.ones_like(observed)
        x, y = 20, 20
        center = (x + .5, y + .5)
        # The final +5 patch column lands exactly on array column x+5.  Its
        # legacy four-tap mask check also inspects x+6, whose data weight is 0.
        last_patch_center = (center[0] + 5, center[1])
        self.assertEqual(joint.bilinear_raw_dependencies(last_patch_center, 48, 48),
                         (y * 48 + x + 5,))
        eroded[y, x + 6] = False
        allowed = joint.allowed_target_center_bitmap(observed, partition, eroded)
        self.assertFalse(allowed[y, x])

        eroded[y, x + 6] = True
        allowed = joint.allowed_target_center_bitmap(observed, partition, eroded)
        self.assertTrue(allowed[y, x])
        self.assertTrue(joint.dependencies_inside_partition(center, partition))

    def test_checkerboard_partitions_and_dependency_safe_center_lists(self):
        verify, fitting = joint.raw_pixel_partitions(280, 280)
        y, x = np.indices((280, 280), dtype=np.int64)
        expected_verify = (((4 * x) // 280 + (4 * y) // 280) % 2) == 0
        np.testing.assert_array_equal(verify, expected_verify)
        np.testing.assert_array_equal(fitting, ~expected_verify)
        self.assertFalse(np.any(verify & fitting))
        self.assertTrue(np.all(verify | fitting))

        observed = np.ones((280, 280), dtype=bool)
        eroded = np.ones_like(observed)
        qh = joint.allowed_target_center_bitmap(observed, verify, eroded)
        qf = joint.allowed_target_center_bitmap(observed, fitting, eroded)
        self.assertTrue(qh[35, 35])
        self.assertTrue(qf[35, 105])
        for bitmap, partition in ((qh, verify), (qf, fitting)):
            centers_yx = np.argwhere(bitmap)
            self.assertGreater(len(centers_yx), 0)
            sample = centers_yx[np.linspace(0, len(centers_yx) - 1,
                                             min(25, len(centers_yx)), dtype=int)]
            for cy, cx in sample:
                self.assertTrue(joint.dependencies_inside_partition(
                    (float(cx) + .5, float(cy) + .5), partition))
        decoded = joint.decode_bitmap(joint.encode_bitmap(qh))
        np.testing.assert_array_equal(decoded, qh)


class SelectionAndGeometryTests(unittest.TestCase):
    def test_empty_witness_fit_and_nonfinite_geometry_fail_closed(self):
        empty_ids = np.asarray([], dtype=np.int64)
        empty_points = np.empty((0, 2), dtype=np.float64)
        empty_bool = np.asarray([], dtype=bool)
        empty_float = np.asarray([], dtype=np.float64)
        mask = np.ones((40, 40), dtype=bool)
        witness = joint.select_witness_ids(
            empty_ids, empty_points, empty_float, empty_bool, mask, lambda _q: (), ())
        self.assertEqual(witness["state"], "unavailable")
        self.assertEqual(witness["denominator"], 0)
        self.assertEqual(witness["usable_count"], 0)
        fit = joint.select_fit_ids(empty_ids, empty_points, empty_bool, mask,
                                   lambda _q: (), ())
        self.assertEqual(fit["state"], "unavailable")
        self.assertEqual(fit["count"], 0)

        small_ids = np.asarray((1, 2), dtype=np.int64)
        small_points = np.asarray(((10., 10.), (20., 20.)), dtype=np.float64)
        small_fit = joint.select_fit_ids(
            small_ids, small_points, np.ones(2, bool), mask,
            lambda q: (int(np.floor(q[1])) * 40 + int(np.floor(q[0])),), ())
        self.assertEqual(small_fit["state"], "unavailable")
        self.assertEqual(small_fit["count"], 2)

        with self.assertRaises(ValueError):
            joint.raycast_first_hits(np.asarray(((float("nan"), 0., 0.),)),
                                     np.asarray(((0., 0., 1.),)),
                                     np.asarray(((0., 0., 1.), (1., 0., 1.),
                                                 (0., 1., 1.))),
                                     np.asarray(((0, 1, 2),)))

    def test_witness_bank_keeps_original_denominator_after_bank_overlap_without_refill(self):
        mask = np.ones((80, 80), dtype=bool)
        points = []
        std = []
        for cell_col in range(4):
            x0 = cell_col * 20 + 5
            for delta, score in ((0, .9), (1, .8), (2, .7)):
                points.append((x0 + delta, 10.0))
                std.append(score)
        points = np.asarray(points, dtype=np.float64)
        ids = (np.floor(points[:, 1]).astype(np.int64) * 80 +
               np.floor(points[:, 0]).astype(np.int64))
        order = np.argsort(ids)
        ids, points = ids[order], points[order]
        std = np.asarray(std, dtype=np.float64)[order]
        bank_overlap_ids = {int(10 * 80 + 5), int(10 * 80 + 25)}

        # Dependency identity is deliberately injective here so this fixture
        # isolates selection, retention and fixed-denominator behavior.
        def dependencies(q):
            x, y = np.floor(q).astype(np.int64)
            return (int(y * 80 + x),)

        result = joint.select_witness_ids(
            ids, points, std, np.ones(len(ids), dtype=bool), mask,
            dependencies, bank_overlap_ids, max_witnesses=8, per_cell=2)
        self.assertEqual(result["denominator"], 8)
        self.assertEqual(result["usable_count"], 6)
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["reason"],
                         "fewer_than_8_usable_frozen_source_witness_ids_after_fixed_bank_exclusion")
        selected_ids = result["ordered_ids"]["ordered_ids"]
        self.assertEqual(len(selected_ids), 8)
        self.assertTrue(bank_overlap_ids.issubset(set(selected_ids)))
        self.assertEqual(selected_ids, [
            int(10 * 80 + 5), int(10 * 80 + 6), int(10 * 80 + 25), int(10 * 80 + 26),
            int(10 * 80 + 45), int(10 * 80 + 46), int(10 * 80 + 65), int(10 * 80 + 66)])
        self.assertEqual(result["unusable_fixed_bank_overlap_count"], 2)
        source_rgb = np.arange(80 * 80 * 3, dtype=np.uint8).reshape(80, 80, 3)
        source_sha = joint.array_record(source_rgb)["sha256"]
        serialized = joint._serializable_witness_bank(result, "fixture-source", source_sha)
        self.assertEqual(serialized["source_image_id"], "fixture-source")
        self.assertEqual(serialized["source_rgb_sha256"], source_sha)
        self.assertTrue(serialized["dependency_union"]["image_qualified"])
        self.assertEqual(serialized["dependency_union"]["source_image_id"], "fixture-source")
        self.assertEqual(serialized["dependency_union"]["source_rgb_sha256"], source_sha)

    def test_fit_selection_excludes_reserved_dependencies_and_round_robins_cells(self):
        mask = np.ones((80, 80), dtype=bool)
        points = np.asarray([(x + 2.0 + offset, y + 2.0)
                             for y in (5, 25, 45, 65)
                             for x in (5, 25, 45, 65)
                             for offset in (0., 3.)], dtype=np.float64)
        ids = np.arange(32, dtype=np.int64)
        eligible = np.ones(32, dtype=bool)

        def dependencies(q):
            row = int(np.argmin(np.sum((points - np.asarray(q)) ** 2, axis=1)))
            return (int(ids[row]),)

        reserved = {0, 16}
        result = joint.select_fit_ids(ids, points, eligible, mask,
                                     dependencies, reserved)
        selected = result["ordered_ids"]["ordered_ids"]
        self.assertEqual(len(selected), 30)
        self.assertFalse(reserved.intersection(selected))
        self.assertEqual(result["state"], "frozen")

    def test_cylinder_triangle_ids_winding_normals_uv_seam_and_ray_tie(self):
        mesh = joint.fixture_cylinder_mesh()
        vertices = mesh["positions"]
        faces = mesh["triangles"]
        uv = mesh["uv"]
        normals = mesh["triangle_normals"]
        self.assertEqual(vertices.shape, (132, 3))
        self.assertEqual(faces.shape, (256, 3))
        self.assertTrue(np.array_equal(mesh["triangle_ids"], np.arange(256)))
        np.testing.assert_array_equal(faces[:4], np.asarray(((0, 2, 1), (2, 3, 1),
                                                            (2, 4, 3), (4, 5, 3))))
        np.testing.assert_array_equal(faces[128], (130, 2, 0))
        np.testing.assert_array_equal(faces[191], (130, 0, 126))
        np.testing.assert_array_equal(faces[192], (131, 1, 3))
        np.testing.assert_array_equal(faces[255], (131, 127, 1))
        np.testing.assert_allclose(vertices[0], vertices[128], atol=1e-15, rtol=0.)
        self.assertEqual(float(uv[0, 0]), 0.0)
        self.assertEqual(float(uv[128, 0]), 1.0)
        side_centers = vertices[faces[:128]].mean(axis=1)
        radial_dot = normals[:128, 0] * side_centers[:, 0] + normals[:128, 2] * side_centers[:, 2]
        self.assertTrue(np.all(radial_dot > 0.0))
        self.assertTrue(np.all(normals[128:192, 1] < 0.0))
        self.assertTrue(np.all(normals[192:, 1] > 0.0))

        plane = np.asarray(((-1., -1., 0.), (1., -1., 0.), (0., 1., 0.)))
        duplicate_faces = np.asarray(((0, 1, 2), (0, 1, 2)), dtype=np.int64)
        tie = joint.raycast_first_hits(np.asarray(((0., 0., 1.),)),
                                      np.asarray(((0., 0., -1.),)),
                                      plane, duplicate_faces, chunk_size=1)
        self.assertTrue(bool(tie["hit"][0]))
        self.assertEqual(int(tie["triangle_id"][0]), 0)
        self.assertAlmostEqual(float(tie["distance_m"][0]), 1.0, places=12)

    def test_fixture_renderer_uses_nearest_texel_floor_without_antialiasing(self):
        mesh = joint.fixture_cylinder_mesh()
        texture = joint.fixture_texture("markers")
        k = np.asarray(((100., 0., 32.), (0., 100., 32.), (0., 0., 1.)), dtype=np.float64)
        pose = _reference_pose_y(0.)
        rendered = joint._render_fixture_view(mesh, texture, pose, k, width=64, height=64)
        hit = rendered["observed_mask"]
        tri = rendered["hit_triangle_ids"][hit]
        bary = rendered["hit_barycentric"][hit]
        uv = np.einsum("ni,nij->nj", bary, mesh["uv"][mesh["triangles"][tri]], optimize=False)
        tex_x = np.floor(256. * uv[:, 0]).astype(np.int64) % 256
        tex_y = np.clip(np.floor(256. * uv[:, 1]).astype(np.int64), 0, 255)
        expected = np.full((len(tri), 3), .6, dtype=np.float32)
        side = tri < 128
        expected[side] = texture[tex_y[side], tex_x[side]]
        np.testing.assert_array_equal(rendered["rgb"][hit], expected)
        np.testing.assert_array_equal(rendered["rgb"][~hit],
                                      np.full((int((~hit).sum()), 3), .5, dtype=np.float32))
        depth = rendered["depth_mm"]
        self.assertTrue(np.all(np.isfinite(depth[hit])))
        self.assertTrue(np.all(depth[hit] > 0.0))
        self.assertTrue(np.all(~np.isfinite(depth[~hit]) | (depth[~hit] <= 0.0)),
                        "background pixels must never carry valid positive source depth")
        self.assertTrue(np.all(rendered["hit_triangle_ids"][hit] >= 0))
        self.assertTrue(np.all(rendered["hit_triangle_ids"][~hit] == -1))

    def test_texture_formula_periodicity_and_case_variant_boundaries(self):
        periodic = joint.fixture_texture("periodic_2d")
        self.assertEqual(periodic.dtype, np.float32)
        np.testing.assert_array_equal(periodic[:, :128], periodic[:, 128:])
        self.assertGreater(float(np.ptp(periodic[0, :, 0])), 0.)
        self.assertGreater(float(np.ptp(periodic[:, 0, 0])), 0.)
        for tx, ty in ((0, 0), (17, 33), (127, 255), (128, 1)):
            u, v = tx % 128, ty
            p = ((37 * u + 17 * v + 13 * u * v + 11) % 251) / 250.0
            expected = np.float32(.15 + .70 * p)
            np.testing.assert_array_equal(periodic[ty, tx],
                                          np.asarray((expected,) * 3, dtype=np.float32))

        markers = joint.fixture_texture("markers")
        base = joint.fixture_texture("base")
        self.assertEqual(markers.dtype, np.float32)
        self.assertEqual(base.dtype, np.float32)
        marker_rules = ((92, 164, 12, 72, 11, (0, 1, 2)),
                        (92, 164, 88, 168, 53, (1, 2, 0)),
                        (92, 164, 184, 244, 97, (2, 0, 1)))
        for x0, _x1, y0, _y1, seed, order in marker_rules:
            tx, ty = x0 + 5, y0 + 7
            local_x, local_y = tx - x0, ty - y0
            p = ((37 * local_x + 17 * local_y + 13 * local_x * local_y + seed) % 251) / 250.0
            colors = ((.15 + .75 * p, .15 + .30 * p, .20 + .15 * p)
                      if order == (0, 1, 2) else
                      (.20 + .15 * p, .15 + .75 * p, .15 + .30 * p)
                      if order == (1, 2, 0) else
                      (.15 + .30 * p, .20 + .15 * p, .15 + .75 * p))
            np.testing.assert_array_equal(markers[ty, tx], np.asarray(colors, dtype=np.float32))
            self.assertFalse(np.array_equal(markers[ty, tx], base[ty, tx]))

        declarations = joint.control_case_declarations()
        expected = [f"{variant}-seed-{seed:03d}"
                    for variant in ("distinct_markers", "blank_constant", "repeated_2d_periodic",
                                    "coherent_wrong_180_packet", "query_occlusion",
                                    "mask_leakage_surrogate", "global_affine", "appearance_mismatch")
                    for seed in (0, 180)]
        self.assertEqual([row["case_id"] for row in declarations], expected)
        self.assertEqual([row["case_id"] for row in declarations], joint._planned_synthetic_case_ids())
        self.assertTrue(all(row["control_state"] == "declared_only" and
                            row["stage_a1_action"] == "no_fit_or_score" for row in declarations))

    def test_sixteen_procedural_cases_freeze_hashes_splits_ids_and_truth_separation(self):
        declarations = joint.control_case_declarations()
        self.assertEqual(len(declarations), 16)
        expected_ids = [f"{variant}-seed-{seed:03d}"
                        for variant in ("distinct_markers", "blank_constant", "repeated_2d_periodic",
                                        "coherent_wrong_180_packet", "query_occlusion",
                                        "mask_leakage_surrogate", "global_affine", "appearance_mismatch")
                        for seed in (0, 180)]
        self.assertEqual([row["case_id"] for row in declarations], expected_ids)
        baseline = {}
        observed_record_digests = {}
        for declaration in declarations:
            case = joint.generate_control_case(declaration["variant"],
                                               declaration["source_seed_deg"])
            self.assertEqual(case["case_id"], declaration["case_id"])
            self.assertLessEqual(case["resource_budget"]["peak_bytes"], joint.MAX_WORKING_BYTES)
            inputs = case["estimator_inputs"]
            truth = case["evaluator_truth"]
            digest_before = joint.fixture_case_digest(case)
            self.assertTrue(digest_before["truth_fields_separate_from_estimator"])
            self.assertEqual(digest_before["stage_a1_state"],
                             "declared_arrays_only_no_fit")
            self.assertEqual(inputs["source_rgb"].dtype, np.float32)
            self.assertEqual(inputs["query_rgb"].dtype, np.float32)
            self.assertEqual(inputs["source_ids"].dtype, np.int64)
            self.assertTrue(np.all(np.diff(inputs["source_ids"]) > 0))
            self.assertEqual(inputs["source_ids"].shape[0], int(inputs["source_observed_mask"].sum()))

            seed = int(declaration["source_seed_deg"])
            variant = declaration["variant"]
            if variant == "distinct_markers":
                baseline[seed] = dict(query_rgb=inputs["query_rgb"].copy(),
                                      query_mask=inputs["query_observed_mask"].copy(),
                                      truth_depth=truth["query_depth_mm"].copy(),
                                      endpoint_stub_q0=inputs["endpoint_stub_q0"].copy())
            elif variant == "coherent_wrong_180_packet":
                np.testing.assert_array_equal(inputs["query_rgb"], baseline[seed]["query_rgb"])
                np.testing.assert_array_equal(inputs["query_observed_mask"],
                                              baseline[seed]["query_mask"])
                self.assertFalse(np.array_equal(inputs["endpoint_stub_q0"],
                                                baseline[seed]["endpoint_stub_q0"]))
            elif variant == "query_occlusion":
                band = np.zeros((280, 280), dtype=bool)
                band[:, 105:175] = True
                np.testing.assert_array_equal(inputs["query_observed_mask"],
                                              baseline[seed]["query_mask"] & ~band)
                self.assertTrue(np.all(inputs["query_rgb"][band] == np.float32(.5)))
            elif variant == "mask_leakage_surrogate":
                np.testing.assert_array_equal(inputs["query_observed_mask"],
                                              baseline[seed]["query_mask"])
                xs = np.arange(280, dtype=np.int64)
                stripe = np.float32(.25) + np.float32(.5) * ((xs // 8) % 2).astype(np.float32)
                np.testing.assert_array_equal(inputs["query_rgb"][:, 105:175, :],
                                              np.broadcast_to(stripe[None, 105:175, None],
                                                              (280, 70, 3)))
            elif variant == "global_affine":
                expected_rgb = (.8 * baseline[seed]["query_rgb"].astype(np.float64) + .05).astype(np.float32)
                np.testing.assert_array_equal(inputs["query_rgb"], expected_rgb)
            elif variant == "appearance_mismatch":
                np.testing.assert_array_equal(inputs["query_observed_mask"],
                                              baseline[seed]["query_mask"])
                np.testing.assert_array_equal(truth["query_depth_mm"],
                                              baseline[seed]["truth_depth"])
                np.testing.assert_array_equal(inputs["fixed_mesh"]["texture_rgb"],
                                              joint.fixture_texture("markers"))
                self.assertFalse(np.array_equal(inputs["query_rgb"], baseline[seed]["query_rgb"]))

            if variant == "blank_constant":
                np.testing.assert_array_equal(inputs["fixed_mesh"]["texture_rgb"],
                                              joint.fixture_texture("constant"))
            elif variant == "repeated_2d_periodic":
                np.testing.assert_array_equal(inputs["fixed_mesh"]["texture_rgb"],
                                              joint.fixture_texture("periodic_2d"))
            elif variant not in ("blank_constant", "repeated_2d_periodic"):
                np.testing.assert_array_equal(inputs["fixed_mesh"]["texture_rgb"],
                                              joint.fixture_texture("markers"))

            inputs_before_truth_edit = digest_before["estimator_inputs"]
            query_asset_before = digest_before["generator_arrays"]["query_texture"]
            case["generator_arrays"]["query_texture"][...] = np.float32(.321)
            if isinstance(truth.get("query_depth_mm"), np.ndarray):
                truth["query_depth_mm"][...] = -123.0
            if isinstance(truth.get("query_pose_camera_from_object_m"), np.ndarray):
                truth["query_pose_camera_from_object_m"][...] = -7.0
            self.assertEqual(joint.fixture_case_digest(case)["estimator_inputs"],
                             inputs_before_truth_edit)
            self.assertNotEqual(joint.fixture_case_digest(case)["generator_arrays"]["query_texture"],
                                query_asset_before)
            verify_partition, fitting_partition = joint.raw_pixel_partitions(280, 280)
            fixture_eroded = joint._square_erode(inputs["query_observed_mask"], 6)
            qh = joint.allowed_target_center_bitmap(inputs["query_observed_mask"],
                                                    verify_partition, fixture_eroded)
            qf = joint.allowed_target_center_bitmap(inputs["query_observed_mask"],
                                                    fitting_partition, fixture_eroded)
            source_ids_le = np.asarray(inputs["source_ids"], dtype="<i8")
            content_record = dict(
                case_id=digest_before["case_id"], variant=digest_before["variant"],
                source_seed_deg=digest_before["source_seed_deg"],
                estimator_inputs=digest_before["estimator_inputs"],
                fixed_mesh=digest_before["fixed_mesh"],
                generator_arrays=digest_before["generator_arrays"],
                geometry=digest_before["geometry"],
                generator_rules=digest_before["generator_rules"],
                evaluator_truth=digest_before["evaluator_truth"],
                ordered_source_ids=dict(count=int(len(source_ids_le)),
                                        sha256=joint.sha256_bytes(source_ids_le.tobytes())),
                target_splits=dict(verification_raw=joint.array_record(verify_partition),
                    fitting_raw=joint.array_record(fitting_partition),
                    verification_centers=joint.encode_bitmap(qh),
                    fitting_centers=joint.encode_bitmap(qf)))
            observed_record_digests[case["case_id"]] = joint.sha256_bytes(
                joint.canonical_json_bytes(content_record))

        self.assertEqual(set(observed_record_digests), set(expected_ids))
        self.assertTrue(all(len(value) == 64 for value in observed_record_digests.values()))
        if FROZEN_CONTROL_RECORD_SHA256:
            self.assertEqual(observed_record_digests, FROZEN_CONTROL_RECORD_SHA256)


class SupportTopologyAndUpperBoundTests(unittest.TestCase):
    def test_source_appearance_requires_full_raw_support_and_valid_depth(self):
        center = (22.5, 22.5)
        observed = np.zeros((64, 64), dtype=bool)
        observed[10:50, 10:50] = True

        # Establish that a varied source image remains eligible under the
        # unchanged R5 erosion and all-four bilinear mask-tap rule.
        rgb = np.random.default_rng(917).random((64, 64, 3), dtype=np.float32)
        high = r5_audit._highpass(r5_audit._gray_image(rgb), 2.)
        legacy_eroded = r5_audit._eroded_mask(observed, 6)
        legacy_patch, _ = r5_audit._bilinear_patch(high, center, legacy_eroded)
        self.assertIsNotNone(legacy_patch)
        _descriptor, legacy_std = r5_audit._normalize_patch(legacy_patch)
        self.assertGreaterEqual(legacy_std, r5_audit.PATCH_STD_MIN)
        self.assertTrue(np.all(legacy_eroded[17:29, 17:29]))
        reference_dependencies = _reference_raw_dependencies(center, 64, 64)
        self.assertIn(9, {pixel % 64 for pixel in reference_dependencies})
        self.assertIn(13 * 64 + 22, reference_dependencies)
        depth = np.full((64, 64), 1500.0, dtype=np.float64)
        result = joint.source_appearance_support(center, observed, depth)
        self.assertEqual(result["state"], "unsupported")
        self.assertIn("source_mask_false_raw_dependency", result["reasons"])
        self.assertEqual(result["dependency_count"], len(reference_dependencies))
        self.assertGreater(result["unsupported_mask_count"], 0)
        self.assertEqual(result["invalid_depth_count"], 0)
        self.assertFalse(result["out_of_bounds"])

        full_mask = np.ones((64, 64), dtype=bool)
        supported = joint.source_appearance_support(center, full_mask, depth)
        self.assertEqual(supported["state"], "supported")
        self.assertEqual(supported["reasons"], [])
        for invalid_depth in (0.0, float("nan")):
            with self.subTest(invalid_depth=invalid_depth):
                broken_depth = depth.copy()
                broken_depth[13, 22] = invalid_depth
                invalid = joint.source_appearance_support(center, full_mask, broken_depth)
                self.assertEqual(invalid["state"], "unsupported")
                self.assertIn("source_depth_invalid_raw_dependency", invalid["reasons"])
                self.assertGreaterEqual(invalid["invalid_depth_count"], 1)
                self.assertEqual(invalid["unsupported_mask_count"], 0)

    def test_unsupported_selected_source_does_not_refill_witness_denominator(self):
        observed = np.zeros((64, 64), dtype=bool)
        observed[10:50, 10:50] = True
        points = np.asarray(((22.5, 22.5), (25.5, 25.5),
                             (30.5, 25.5), (31.5, 25.5),
                             (25.5, 30.5), (25.5, 31.5),
                             (30.5, 30.5), (31.5, 31.5)), dtype=np.float64)
        ids = np.arange(1000, 1008, dtype=np.int64)
        point_to_row = {tuple(point): row for row, point in enumerate(points.tolist())}
        # Isolate immutable selection from the later appearance gate. The
        # support helper below independently expands the true raw D(q).
        witness = joint.select_witness_ids(
            ids, points, np.arange(8, 0, -1, dtype=np.float64), np.ones(8, bool),
            observed, lambda q: (100_000 + point_to_row[tuple(q)],), (),
            max_witnesses=8, per_cell=2)
        self.assertEqual(witness["denominator"], 8)
        self.assertEqual(witness["usable_count"], 8)
        self.assertEqual(witness["ordered_ids"]["ordered_ids"], ids.tolist())

        depth = np.full((64, 64), 1500.0, dtype=np.float64)
        support, supported_mask = joint._source_appearance_support_for_ids(
            ids, points, observed, depth)
        self.assertEqual(support["source_count"], witness["denominator"])
        self.assertEqual(support["supported_count"], 7)
        self.assertEqual(support["unsupported_count"], 1)
        self.assertEqual(_ordered_values(support["unsupported_ids"]), [1000])
        self.assertEqual(_ordered_values(support["supported_ids"]), ids[1:].tolist())
        self.assertEqual(supported_mask.tolist(), [False] + [True] * 7)
        self.assertEqual(witness["denominator"], 8,
                         "appearance filtering must preserve the frozen denominator")
        self.assertEqual(witness["ordered_ids"]["ordered_ids"], ids.tolist(),
                         "appearance filtering must not refill or rewrite reserved IDs")

    def test_native_topology_accepts_boundaries_and_fixture_seam_but_rejects_ambiguity(self):
        open_triangle = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
        boundary = joint.native_mesh_topology(open_triangle,
                                              np.asarray(((0, 1, 2),), dtype=np.int64))
        self.assertEqual(boundary["boundary_edge_count"], 3)
        self.assertEqual(boundary["nonmanifold_edge_count"], 0)
        self.assertEqual(boundary["inconsistent_winding_edge_count"], 0)
        self.assertEqual(_ordered_values(boundary["unsupported_triangle_ids"]), [])

        square = np.asarray(((0., 0., 0.), (1., 0., 0.),
                             (1., 1., 0.), (0., 1., 0.)))
        same_direction = joint.native_mesh_topology(
            square, np.asarray(((0, 1, 2), (0, 3, 2)), dtype=np.int64))
        self.assertEqual(same_direction["inconsistent_winding_edge_count"], 1)
        self.assertEqual(set(_ordered_values(same_direction["unsupported_triangle_ids"])), {0, 1})

        three_way = np.vstack((square, np.asarray(((.5, -1., 0.),))))
        nonmanifold = joint.native_mesh_topology(
            three_way, np.asarray(((0, 1, 2), (0, 2, 3), (0, 2, 4)), dtype=np.int64))
        self.assertEqual(nonmanifold["nonmanifold_edge_count"], 1)
        self.assertEqual(set(_ordered_values(nonmanifold["unsupported_triangle_ids"])), {0, 1, 2})

        cylinder = joint.fixture_cylinder_mesh()
        seam_topology = joint.native_mesh_topology(cylinder["positions"], cylinder["triangles"])
        self.assertEqual(_ordered_values(seam_topology["unsupported_triangle_ids"]), [])
        self.assertEqual(seam_topology["boundary_edge_count"], 0)
        self.assertEqual(seam_topology["nonmanifold_edge_count"], 0)
        self.assertEqual(seam_topology["inconsistent_winding_edge_count"], 0)

    def test_verification_support_upper_bound_separates_impossible_from_possible(self):
        observed = np.zeros((64, 64), dtype=bool)
        observed[10:50, 10:50] = True

        two_cells = np.zeros_like(observed)
        two_cells[10, 10] = True  # bbox cell 0
        two_cells[49, 49] = True  # bbox cell 15, far from the first
        two = joint.qh_verification_support_upper_bound(two_cells, observed)
        self.assertEqual(two["state"], "known_impossible")
        self.assertEqual(len(two["represented_cells"]), 2)

        compact_hull = np.zeros_like(observed)
        compact_hull[25, 25] = True  # cells 5, 6, and 9 form a noncollinear set
        compact_hull[25, 35] = True
        compact_hull[35, 25] = True
        compact = joint.qh_verification_support_upper_bound(compact_hull, observed)
        self.assertEqual(len(compact["represented_cells"]), 3)
        self.assertLess(compact["maximum_hull_fraction"], .12)
        self.assertEqual(compact["state"], "known_impossible")

        broad_hull = np.zeros_like(observed)
        broad_hull[10, 10] = True
        broad_hull[10, 49] = True
        broad_hull[49, 10] = True
        possible = joint.qh_verification_support_upper_bound(broad_hull, observed)
        self.assertGreaterEqual(possible["maximum_hull_fraction"], .12)
        self.assertEqual(possible["state"], "possible_upper_bound",
                         "an upper-bound pass is possibility, never witness acceptance")
        self.assertTrue(possible["upper_bound_only"])
        self.assertFalse(possible["sufficient_support"])

    def test_control_coverage_blocks_mandatory_unavailable_but_allows_blank_abstention(self):
        declarations = joint.control_case_declarations()
        rows = []
        for declaration in declarations:
            prerequisite_state = (
                "terminal_unavailable" if declaration["variant"] == "blank_constant"
                else "preflight_possible")
            rows.append(dict(
                variant=declaration["variant"], case_id=declaration["case_id"],
                a1_source_split_preflight=dict(
                    state="frozen_source_only_preflight",
                    prerequisite_state=prerequisite_state)))

        summary = joint.summarize_control_preflight_coverage(rows)
        self.assertEqual(summary["state"], "passed")
        self.assertTrue(summary["all_rows_accounted"])
        self.assertEqual(summary["accounted_count"], 16)
        self.assertEqual(set(summary["mandatory_variants"]), {
            "distinct_markers", "repeated_2d_periodic",
            "coherent_wrong_180_packet", "global_affine"})
        self.assertEqual(set(summary["blank_negative_abstention_case_ids"]), {
            row["case_id"] for row in declarations if row["variant"] == "blank_constant"})
        self.assertEqual(summary["mandatory_unavailable_case_ids"], [])

        # The preflight can be source-frozen yet unusable. Every mandatory
        # positive/alias control must then fail the aggregate, including the
        # repeated-2D control whose two finite modes are required later.
        for variant in summary["mandatory_variants"]:
            with self.subTest(variant=variant):
                altered = [dict(row, a1_source_split_preflight=dict(
                    row["a1_source_split_preflight"])) for row in rows]
                target = next(row for row in altered if row["variant"] == variant)
                target["a1_source_split_preflight"]["prerequisite_state"] = "terminal_unavailable"
                failed = joint.summarize_control_preflight_coverage(altered)
                self.assertEqual(failed["state"], "failed")
                self.assertTrue(failed["all_rows_accounted"])
                self.assertIn(target["case_id"], failed["mandatory_unavailable_case_ids"])

    def test_association_support_filters_frozen_ids_and_uses_count_thresholds(self):
        witness_ids = np.arange(100, 109, dtype=np.int64)
        fit_ids = np.arange(1000, 1025, dtype=np.int64)
        association = {int(source_id): True
                       for source_id in np.concatenate((witness_ids, fit_ids))}
        association[int(witness_ids[0])] = False
        association[int(fit_ids[0])] = False
        filtered = joint.association_filtered_source_support(
            witness_ids, witness_ids, fit_ids, fit_ids, association)

        self.assertEqual(filtered["witness_state"], "frozen")
        self.assertEqual(filtered["fit_state"], "frozen")
        self.assertEqual(filtered["witness_selected_ids"], witness_ids.tolist())
        self.assertEqual(filtered["witness_usable_ids"], witness_ids[1:].tolist())
        self.assertEqual(filtered["witness_unsupported_selected_ids"], [int(witness_ids[0])])
        self.assertEqual(filtered["fit_selected_ids"], fit_ids.tolist())
        self.assertEqual(filtered["fit_usable_ids"], fit_ids[1:].tolist())
        self.assertEqual(filtered["fit_unsupported_selected_ids"], [int(fit_ids[0])])

        possible_row = joint.source_row_prerequisite_state(
            witness_state=filtered["witness_state"], fit_state=filtered["fit_state"],
            fixed_bank_supported=True, qh_upper_bound_state="possible_upper_bound",
            unsupported_association_count=2)
        self.assertEqual(possible_row["state"], "prepared")
        self.assertEqual(possible_row["unavailable_reasons"], [])
        self.assertEqual(possible_row["unsupported_association_count"], 2)

        unsupported_bank = joint.source_row_prerequisite_state(
            witness_state=filtered["witness_state"], fit_state=filtered["fit_state"],
            fixed_bank_supported=False, qh_upper_bound_state="possible_upper_bound",
            unsupported_association_count=2)
        self.assertEqual(unsupported_bank["state"], "terminal_unavailable")
        self.assertIn("fixed_bank_source_appearance_unsupported",
                      unsupported_bank["unavailable_reasons"])

        impossible_qh = joint.source_row_prerequisite_state(
            witness_state=filtered["witness_state"], fit_state=filtered["fit_state"],
            fixed_bank_supported=True, qh_upper_bound_state="known_impossible",
            unsupported_association_count=0)
        self.assertEqual(impossible_qh["state"], "terminal_unavailable")
        self.assertIn("verification_support_upper_bound_known_impossible",
                      impossible_qh["unavailable_reasons"])

        short_witness = np.arange(200, 208, dtype=np.int64)
        short_fit = np.arange(2000, 2024, dtype=np.int64)
        short_association = {int(source_id): True
                             for source_id in np.concatenate((short_witness, short_fit))}
        short_association[int(short_witness[0])] = False
        short_association[int(short_fit[0])] = False
        below = joint.association_filtered_source_support(
            short_witness, short_witness, short_fit, short_fit, short_association)
        self.assertEqual(below["witness_usable_ids"], short_witness[1:].tolist())
        self.assertEqual(below["fit_usable_ids"], short_fit[1:].tolist())
        self.assertEqual(below["witness_state"], "unavailable")
        self.assertEqual(below["fit_state"], "unavailable")
        below_row = joint.source_row_prerequisite_state(
            witness_state=below["witness_state"], fit_state=below["fit_state"],
            fixed_bank_supported=True, qh_upper_bound_state="possible_upper_bound",
            unsupported_association_count=2)
        self.assertEqual(below_row["state"], "terminal_unavailable")


class FixturePreflightSchemaTests(unittest.TestCase):
    def test_real_procedural_preflight_uses_upstream_anchor_schema_and_freezes_w0(self):
        original_builder = r5_audit.build_template_patch_bank
        observed_public_anchors = []
        observed_anchor_geometry = []

        def record_public_schema(*args, **kwargs):
            bank = original_builder(*args, **kwargs)
            rows = bank["serialized_anchors"]
            public_rows = []
            for row in rows:
                self.assertIn("source_index", row)
                self.assertIn("highpass_std", row)
                self.assertNotIn("source_id", row)
                source_index = row["source_index"]
                self.assertIsInstance(source_index, (int, np.integer))
                highpass_std = float(row["highpass_std"])
                self.assertTrue(math.isfinite(highpass_std))
                public_rows.append(dict(source_index=int(source_index),
                                        highpass_std=highpass_std))
            observed_public_anchors.append(public_rows)
            observed_anchor_geometry.append([
                dict(source_index=int(anchor["source_index"]),
                     source_xy_crop=np.asarray(anchor["source_xy_crop"], dtype=np.float64).copy())
                for anchor in bank["anchors"]])
            return bank

        with mock.patch.object(r5_audit, "build_template_patch_bank",
                               side_effect=record_public_schema):
            case = joint.generate_control_case("distinct_markers", 0)
            preflight = joint._fixture_case_a1_preflight(case, cv2, r5_audit)

        self.assertEqual(case["case_id"], "distinct_markers-seed-000")
        self.assertEqual(len(observed_public_anchors), 1)
        public_rows = observed_public_anchors[0]
        bank_anchors = observed_anchor_geometry[0]
        self.assertEqual(preflight["state"], "frozen_source_only_preflight")
        fixed_bank = preflight["fixed_bank"]
        self.assertEqual(fixed_bank["anchor_count"], len(public_rows))
        self.assertEqual(fixed_bank["anchor_ids"]["ordered_ids"],
                         [row["source_index"] for row in public_rows])
        expected_digest = joint.sha256_bytes(joint.canonical_json_bytes(public_rows))
        self.assertEqual(fixed_bank["manifest_sha256"], expected_digest)
        self.assertGreater(fixed_bank["anchor_count"], 0)

        # Independently expand the literal nonzero high-pass/bilinear footprint
        # for each real audit anchor and selected witness. This catches the
        # source-index schema regression without trusting the helper's overlap bit.
        inputs = case["estimator_inputs"]
        width = int(inputs["source_rgb"].shape[1])
        height = int(inputs["source_rgb"].shape[0])
        bank_dependency_union = set()
        for anchor in bank_anchors:
            bank_dependency_union.update(_reference_raw_dependencies(
                anchor["source_xy_crop"], width, height))
        witness_records = preflight["witness_bank"]["dependencies"]
        source_ids = np.asarray(inputs["source_ids"], dtype=np.int64)
        source_xy = np.asarray(inputs["source_pixels_xy"], dtype=np.float64)
        source_mask = np.asarray(inputs["source_observed_mask"], dtype=bool).reshape(-1)
        source_depth = np.asarray(inputs["source_depth_mm"]).reshape(-1)
        source_row_by_id = {int(source_id): row
                            for row, source_id in enumerate(source_ids.tolist())}
        independent_overlap = []
        independent_appearance_support = []
        for witness in witness_records:
            source_id = int(witness["source_id"])
            deps = set(_reference_raw_dependencies(
                source_xy[source_row_by_id[source_id]], width, height))
            independent_overlap.append(bool(deps & bank_dependency_union))
            dep_ids = np.fromiter(sorted(deps), dtype=np.int64)
            depth = source_depth[dep_ids]
            independent_appearance_support.append(bool(
                source_mask[dep_ids].all() and np.isfinite(depth).all() and (depth > 0).all()))
        self.assertEqual(len(witness_records), 31)
        self.assertEqual(preflight["witness_bank"]["reserved_denominator"], 31)
        self.assertTrue(all(independent_overlap))
        self.assertEqual(sum(independent_overlap),
                         preflight["witness_bank"]["unusable_fixed_bank_overlap_count"])
        self.assertEqual([row["fixed_bank_overlap"] for row in witness_records],
                         independent_overlap)
        self.assertEqual(sum(independent_appearance_support), 20)
        self.assertEqual([row["appearance_supported"] for row in witness_records],
                         independent_appearance_support)
        self.assertEqual(preflight["witness_bank"]["usable_count"], 0)
        self.assertEqual(len(preflight["witness_bank"]["appearance_unsupported_ids"]["ordered_ids"]),
                         11)

        # Independently check each bank anchor's D(q) against the original
        # observed mask and depth before asserting the fixed-bank failure count.
        unsupported_anchor_ids = []
        for anchor in bank_anchors:
            source_id = int(anchor["source_index"])
            dep_ids = np.fromiter(sorted(_reference_raw_dependencies(
                anchor["source_xy_crop"], width, height)), dtype=np.int64)
            depth = source_depth[dep_ids]
            supported = bool(source_mask[dep_ids].all() and
                             np.isfinite(depth).all() and (depth > 0).all())
            if not supported:
                unsupported_anchor_ids.append(source_id)
        self.assertEqual(unsupported_anchor_ids,
                         fixed_bank["unsupported_appearance_ids"]["ordered_ids"])
        self.assertEqual(len(unsupported_anchor_ids), 7)

        fit_sources = preflight["fit_sources"]
        self.assertEqual(fit_sources["selected_count"], 1193)
        self.assertEqual(fit_sources["count"], 1193)
        self.assertEqual(preflight["prerequisite_state"], "terminal_unavailable")
        self.assertIn("source_witness_appearance_or_geometry_support_unavailable",
                      preflight["prerequisite_reasons"])
        self.assertIn("fixed_bank_source_appearance_unsupported",
                      preflight["prerequisite_reasons"])
        self.assertFalse(preflight["matching_started"])
        self.assertFalse(preflight["fitting_started"])
        self.assertEqual(preflight["ncc_calls"], 0)
        self.assertEqual(preflight["forward_calls"], 0)
        self.assertEqual(preflight["optimization_calls"], 0)
        self.assertEqual(preflight["production_render_calls"], 0)
        encoded = joint.canonical_json_bytes(preflight)
        self.assertLess(len(encoded), joint.MAX_OUTPUT_BYTES)
        decoded = json.loads(encoded.decode("utf-8"))
        self.assertEqual(decoded["fixed_bank"]["manifest_sha256"], expected_digest)


class CoordinatesAndGateTests(unittest.TestCase):
    def test_nonidentity_crop_native_conversion_preserves_mm_boundary(self):
        crop_pose = _reference_pose_y(23.0, (0.12, -0.08, .91))
        c = np.eye(4, dtype=np.float64)
        c[:3, :3] = np.asarray(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
        c[:3, 3] = (35., -40., 22.)  # C translation follows the frozen millimetre boundary.
        crop_pose_mm = crop_pose.copy()
        crop_pose_mm[:3, 3] *= 1000.0
        expected_native_mm = np.linalg.inv(c) @ crop_pose_mm
        expected_native = expected_native_mm.copy()
        expected_native[:3, 3] *= .001
        native = joint.crop_to_native_pose_m(crop_pose, c)
        np.testing.assert_allclose(native, expected_native, atol=1e-12, rtol=1e-12)
        crop_mm = joint.native_to_crop_pose_mm(native, c)
        expected_crop_mm = c @ expected_native_mm
        np.testing.assert_allclose(crop_mm, expected_crop_mm, atol=1e-10, rtol=1e-12)

    def test_offcenter_unequal_intrinsics_ray_and_pose_coordinate_algebra(self):
        k = np.asarray(((813., 0., 137.25), (0., 607., 148.75), (0., 0., 1.)),
                       dtype=np.float64)
        center = np.asarray((.07, -.03, .11), dtype=np.float64)
        pose = _reference_pose_y(17.0, (.04, -.02, .87))
        points = np.asarray(((.1, -.1, .25), (-.2, .05, -.02)), dtype=np.float64)
        camera = (points - center) @ pose[:3, :3].T + pose[:3, :3] @ center + pose[:3, 3]
        homogeneous = camera @ k.T
        expected_xy = homogeneous[:, :2] / homogeneous[:, 2:3]
        xy, valid = joint.project_crop_points(points, pose, k, center)
        np.testing.assert_array_equal(valid, (True, True))
        np.testing.assert_allclose(xy, expected_xy, atol=1e-12, rtol=1e-12)

        c = np.eye(4, dtype=np.float64)
        angle = math.radians(9.0)
        c[:3, :3] = ((math.cos(angle), -math.sin(angle), 0.),
                     (math.sin(angle), math.cos(angle), 0.), (0., 0., 1.))
        crop_q = np.asarray((121.25, 166.75), dtype=np.float64)
        native_k = np.asarray(((503., 0., 319.5), (0., 721., 241.25),
                               (0., 0., 1.)), dtype=np.float64)
        crop_ray = np.linalg.inv(k) @ np.asarray((crop_q[0], crop_q[1], 1.))
        native_ray = c[:3, :3].T @ crop_ray
        expected_native_xy_h = native_k @ native_ray
        expected_native_xy = expected_native_xy_h[:2] / expected_native_xy_h[2]
        actual_native_xy = joint.crop_ray_to_native_xy(crop_q, k, c, native_k)
        np.testing.assert_allclose(actual_native_xy, expected_native_xy, atol=1e-12, rtol=1e-12)

        xi = np.asarray((.01, -.02, .03, .02, -.01, .005), dtype=np.float64)
        diagonal = .43
        moved = joint.local_pose_from_normalized(pose, center, diagonal, xi)
        np.testing.assert_allclose(moved[:3, :3].T @ moved[:3, :3], np.eye(3),
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(moved[:3, :3] @ center + moved[:3, 3],
                                   pose[:3, :3] @ center + pose[:3, 3] + diagonal * xi[3:],
                                   atol=1e-12, rtol=1e-12)

        def projected_at(value):
            candidate = joint.local_pose_from_normalized(pose, center, diagonal, value)
            return joint.project_crop_points(points, candidate, k, center)[0]

        zero = np.zeros(6, dtype=np.float64)
        numeric = joint.central_difference(projected_at, zero, step=1e-6).reshape(2, 2, 6)
        relative = points - center
        camera0 = relative @ pose[:3, :3].T + pose[:3, :3] @ center + pose[:3, 3]
        homogeneous0 = camera0 @ k.T
        projection_jac = np.empty((2, 2, 3), dtype=np.float64)
        for row in range(2):
            projection_jac[:, row, :] = (
                k[row][None, :] * camera0[:, 2:3] -
                homogeneous0[:, row:row + 1] * k[2][None, :]) / camera0[:, 2:3] ** 2
        expected_jac = np.empty_like(numeric)
        basis = np.eye(3, dtype=np.float64)
        for point_index, local in enumerate(relative):
            d_camera = np.column_stack([pose[:3, :3] @ np.cross(basis[column], local)
                                        for column in range(3)] +
                                       [diagonal * basis[:, column] for column in range(3)])
            expected_jac[point_index] = projection_jac[point_index] @ d_camera
        np.testing.assert_allclose(numeric, expected_jac, atol=1e-5, rtol=1e-4)

    def test_nuisance_projection_matches_small_qr_reference_without_n_by_n_output(self):
        rng = np.random.default_rng(9102)
        n = 257
        j_pose = rng.normal(size=(n, 6))
        j_ab = rng.normal(size=(n, 2))
        q, _ = np.linalg.qr(j_ab, mode="reduced")
        expected = j_pose - q @ (q.T @ j_pose)
        actual = joint.project_nuisance_jacobian(j_pose, j_ab)
        self.assertEqual(actual.shape, (n, 6))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

        large_pose = np.zeros((4096, 6), dtype=np.float64)
        large_nuisance = np.ones((4096, 2), dtype=np.float64)
        large = joint.project_nuisance_jacobian(large_pose, large_nuisance)
        self.assertEqual(large.shape, (4096, 6))
        self.assertFalse(np.shares_memory(large, large_pose))

    def test_budget_edges_and_hard_closed_stages(self):
        self.assertEqual(joint.file_sha256(joint.SPEC_PATH),
                         "82b7add0ddf5dcde101139979186511487b4e3f6e6b7f10fc2778f2b33410d57")
        self.assertEqual(joint.SPEC_SHA256,
                         "82b7add0ddf5dcde101139979186511487b4e3f6e6b7f10fc2778f2b33410d57")
        exact = joint.prospective_resource_budget(
            live_bytes=0, requested_bytes=joint.MAX_WORKING_BYTES,
            output_bytes=joint.MAX_OUTPUT_BYTES,
            cache_bytes=joint.MAX_CACHE_BYTES - joint.MAX_OUTPUT_BYTES)
        self.assertEqual(exact["peak_bytes"], 128 * 1024**2)
        with self.assertRaises(joint.ResourceLimitError):
            joint.prospective_resource_budget(live_bytes=0,
                requested_bytes=joint.MAX_WORKING_BYTES + 1, cache_bytes=0)
        with self.assertRaises(joint.ResourceLimitError):
            joint.prospective_resource_budget(live_bytes=0, requested_bytes=0,
                output_bytes=joint.MAX_OUTPUT_BYTES + 1, cache_bytes=0)
        with self.assertRaises(joint.ResourceLimitError):
            joint.prospective_resource_budget(live_bytes=0, requested_bytes=0,
                output_bytes=1, cache_bytes=joint.MAX_CACHE_BYTES)
        with self.assertRaises(joint.ResourceLimitError):
            joint.prospective_resource_budget(live_bytes=-1, requested_bytes=0, cache_bytes=0)

        self.assertEqual(joint.expected_actual_row_ids(), [
            "syn-10-q8-rgb-t0-rgb", "zero-10-full", "zero-10-clipped",
            "syn-10-q8-rgb-t180-rgb", "syn-50-q8-rgb-t0-rgb", "zero-50-full",
            "zero-50-clipped", "syn-50-q8-rgb-t180-rgb", "syn-100-q8-rgb-t0-rgb",
            "zero-100-full", "zero-100-clipped", "syn-100-q8-rgb-t180-rgb"])
        self.assertEqual(len(joint.expected_actual_row_ids()), 12)
        terminal_rows = joint._account_terminal_rows(
            ("ready", "skipped", "missing"),
            {"ready": dict(state="prepared", evidence="bound"),
             "skipped": dict(state="skipped", reason="gate")},
            "not_attempted")
        self.assertEqual([row["condition_id"] for row in terminal_rows],
                         ["ready", "skipped", "missing"])
        self.assertEqual([row["state"] for row in terminal_rows],
                         ["prepared", "terminal_unavailable", "terminal_unavailable"])
        self.assertEqual(terminal_rows[2]["reason"], "not_attempted")
        with self.assertRaises(joint.StageGateError):
            joint.prepare_actual_a1(root_scheduled=False)
        with self.assertRaises(joint.StageGateError):
            joint.run_stage_a2()
        with self.assertRaises(joint.StageGateError):
            joint.run_stage_b()
        with self.assertRaises(joint.StageGateError):
            joint.validate_prebinding_contract(None, verify_files=False)

        source_relative = "bench/test_quality_bottle_joint_prerequisite.py"
        source_sha = "a" * 64
        pinned_executable = (joint.ROOT / ".cache" / "quality-windows" /
                             "Scripts" / "python.exe").resolve()
        self.assertEqual(Path(sys.executable).resolve(), pinned_executable)
        runtime_root = (pinned_executable.parents[1] / "Lib" / "site-packages").resolve()
        base_prefix = Path(sys.base_prefix).resolve()
        wrong_base_runtime_root = (base_prefix / "Lib" / "site-packages").resolve()
        self.assertNotEqual(runtime_root, wrong_base_runtime_root)
        contract = dict(
            schema_version=1, status="sealed_preimport",
            workspace_root=str(joint.ROOT),
            runtime_root=str(runtime_root),
            runtime_identity=None,
            interpreter=dict(path=str(Path(sys.executable).resolve()),
                             python_version=sys.version,
                             sha256=joint.file_sha256(Path(sys.executable))),
            import_closure=[dict(module="bench.test_quality_bottle_joint_prerequisite",
                                 relative_path=source_relative, path=source_relative,
                                 sha256=source_sha)],
            source_pins={source_relative: source_sha},
            runtime_pins={"python.exe": source_sha},
            output_root=str(joint.CACHE / "diagnostics" / "prebinding-contract-fixture"),
            limits=dict(working_bytes=joint.MAX_WORKING_BYTES, wall_seconds=900,
                        artifact_bytes=joint.MAX_OUTPUT_BYTES,
                        model_cache_bytes=joint.MAX_CACHE_BYTES))
        # `-S` may set sys.prefix to the base Python installation. The runtime
        # root must still follow the pinned venv executable, not that prefix.
        with mock.patch.object(joint.sys, "prefix", str(base_prefix)):
            self.assertEqual(Path(joint.sys.prefix).resolve(), base_prefix)
            self.assertNotEqual(Path(joint.sys.prefix).resolve(), runtime_root)
            contract["runtime_identity"] = joint._runtime_identity_record(
                runtime_root, contract["runtime_pins"])
            contract["contract_sha256"] = joint._contract_digest(contract)
            checked = joint.validate_prebinding_contract(contract, verify_files=False)
            self.assertEqual(checked["state"], "structurally_valid")
            self.assertEqual(checked["runtime_root"], str(runtime_root))
            self.assertEqual(checked["limits"]["wall_seconds"], 900)

            wrong_root = dict(contract)
            wrong_root["runtime_root"] = str(wrong_base_runtime_root)
            wrong_root["runtime_identity"] = joint._runtime_identity_record(
                wrong_base_runtime_root, wrong_root["runtime_pins"])
            wrong_root["contract_sha256"] = joint._contract_digest(wrong_root)
            with self.assertRaises(joint.StageGateError):
                joint.validate_prebinding_contract(wrong_root, verify_files=False)

            wrong_executable = dict(contract)
            wrong_interpreter = dict(contract["interpreter"])
            wrong_interpreter["path"] = str((base_prefix / "python.exe").resolve())
            self.assertNotEqual(Path(wrong_interpreter["path"]), pinned_executable)
            wrong_executable["interpreter"] = wrong_interpreter
            wrong_executable["contract_sha256"] = joint._contract_digest(wrong_executable)
            with self.assertRaises(joint.StageGateError):
                joint.validate_prebinding_contract(wrong_executable, verify_files=False)

            altered = dict(contract)
            altered["limits"] = dict(contract["limits"], wall_seconds=901)
            altered["contract_sha256"] = joint._contract_digest(altered)
            with self.assertRaises(joint.StageGateError):
                joint.validate_prebinding_contract(altered, verify_files=False)

        with mock.patch.object(joint.time, "monotonic", return_value=1000.0):
            self.assertEqual(joint._check_wall_deadline(100.0), 900.0)
        with mock.patch.object(joint.time, "monotonic", return_value=1001.0):
            with self.assertRaises(joint.ResourceLimitError):
                joint._check_wall_deadline(100.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "must-not-exist.json"
            with self.assertRaises(joint.StageGateError):
                joint.write_frozen_a1_manifest({}, target, root_scheduled=False)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
