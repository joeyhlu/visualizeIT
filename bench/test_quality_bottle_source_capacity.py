"""Independent synthetic source-only R8 capacity regressions.

These tests use a procedural native plane and the frozen R5/R7 CPU helpers.
They never read an R1/R3 cache packet, query RGB, evaluator truth, or run a
matcher, fitter, renderer, optimizer, or model.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_joint_prerequisite as joint
from . import quality_bottle_pose_ablation as r5
from . import quality_bottle_source_reservation as geometry
from . import quality_bottle_source_capacity as capacity


SOURCE_KEYS = (
    "template_rgb", "template_mask", "template_depth_mm", "source_indices",
    "source_pixels_xy", "source_points_object_m", "observed_crop_mask",
    "crop_k", "crop_from_native", "native_k", "template_pose_m",
)
FORBIDDEN_SOURCE_KEYS = {
    "query_rgb", "query_depth_mm", "known_query_pose_m", "evaluator_truth",
    "annotations", "reference_pose", "synthetic_query_mask", "current_rgb",
}
GRID_COORDS = np.asarray((9, 44, 79, 114, 149, 184, 219, 254), dtype=np.int64)
IMAGE_SIZE = 280


class PoisonSourceArrays(dict):
    """Mapping whose whitelisted source fields work and all truth/query reads fail."""

    def _reject(self, key):
        if key in FORBIDDEN_SOURCE_KEYS:
            raise AssertionError(f"source capacity accessed forbidden field {key}")

    def __getitem__(self, key):
        self._reject(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._reject(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._reject(key)
        return super().__contains__(key)


def _mesh_hash(vertices: np.ndarray, triangles: np.ndarray) -> str:
    """Independent typed/shape-bound SHA for the two native mesh arrays."""
    digest = hashlib.sha256()
    for value in (vertices, triangles):
        contiguous = np.ascontiguousarray(value)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(repr(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _fixture(seed: int = 20261003, *, poison: bool = False):
    """Build a 64-ID source packet projection and an independently known plane."""
    height = width = IMAGE_SIZE
    yy, xx = np.meshgrid(GRID_COORDS, GRID_COORDS, indexing="ij")
    ids = (yy.reshape(-1) * width + xx.reshape(-1)).astype(np.int64)
    pixels = np.column_stack((xx.reshape(-1) + 0.5,
                              yy.reshape(-1) + 0.5)).astype(np.float64)

    crop_k = np.asarray(((200.0, 0.0, 140.0),
                         (0.0, 200.0, 140.0),
                         (0.0, 0.0, 1.0)), dtype=np.float64)
    crop_from_native = np.eye(4, dtype=np.float64)
    template_pose_m = np.eye(4, dtype=np.float64)
    native_k = crop_k.copy()
    rays = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float64))) @ np.linalg.inv(crop_k).T
    points_object_m = (rays * 1.0).astype(np.float64)  # producer equation: z=1000 mm, T=I

    rng = np.random.default_rng(seed)
    template_rgb = rng.random((height, width, 3), dtype=np.float32)
    template_mask = np.ones((height, width), dtype=np.uint8)
    observed_crop_mask = np.ones((height, width), dtype=np.uint8)
    template_depth_mm = np.full((height, width), 1000.0, dtype=np.float32)
    source_arrays = {
        "template_rgb": template_rgb,
        "template_mask": template_mask,
        "template_depth_mm": template_depth_mm,
        "source_indices": ids,
        "source_pixels_xy": pixels,
        "source_points_object_m": points_object_m,
        "observed_crop_mask": observed_crop_mask,
        "crop_k": crop_k,
        "crop_from_native": crop_from_native,
        "native_k": native_k,
        "template_pose_m": template_pose_m,
    }
    if poison:
        source_arrays = PoisonSourceArrays(source_arrays)

    # A consistently wound two-triangle square at z=1 m. Each source ray's
    # independently unprojected producer point lies on this known surface.
    vertices = np.asarray(((-2.0, -2.0, 1.0),
                           (2.0, -2.0, 1.0),
                           (2.0, 2.0, 1.0),
                           (-2.0, 2.0, 1.0)), dtype=np.float64)
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    return {
        "source_arrays": source_arrays,
        "ids": ids,
        "pixels": pixels,
        "points": points_object_m,
        "crop_k": crop_k,
        "vertices": vertices,
        "triangles": triangles,
        "mesh_sha256": _mesh_hash(vertices, triangles),
        "packet_sha256": hashlib.sha256(b"procedural-r8-template-packet-v2").hexdigest(),
        "source_context_id": "template-fixture-0010-000",
    }


def _reflect101(index: int, length: int) -> int:
    if length == 1:
        return 0
    period = 2 * length - 2
    folded = int(index) % period
    return folded if folded < length else period - folded


def _literal_raw_dependencies(center_xy, width=IMAGE_SIZE, height=IMAGE_SIZE):
    """Literal nested-loop oracle for all raw RGB reads behind the 11×11 high-pass patch."""
    qx, qy = map(float, center_xy)
    result = set()
    for patch_dy in range(-5, 6):
        y = qy - 0.5 + patch_dy
        y0 = math.floor(y)
        fy = y - y0
        ys = []
        if 1.0 - fy != 0.0:
            ys.append(y0)
        if fy != 0.0:
            ys.append(y0 + 1)
        for patch_dx in range(-5, 6):
            x = qx - 0.5 + patch_dx
            x0 = math.floor(x)
            fx = x - x0
            xs = []
            if 1.0 - fx != 0.0:
                xs.append(x0)
            if fx != 0.0:
                xs.append(x0 + 1)
            for high_y in ys:
                for high_x in xs:
                    for blur_dy in range(-8, 9):
                        raw_y = _reflect101(high_y + blur_dy, height)
                        for blur_dx in range(-8, 9):
                            raw_x = _reflect101(high_x + blur_dx, width)
                            result.add(raw_y * width + raw_x)
    return result


def _evaluate(fixture, *, source_arrays=None, query_mask=None, **kwargs):
    return capacity.evaluate_source_capacity(
        fixture["source_arrays"] if source_arrays is None else source_arrays,
        source_context_id=fixture["source_context_id"],
        packet_sha256=fixture["packet_sha256"],
        mesh_positions_m=fixture["vertices"],
        mesh_triangles=fixture["triangles"],
        mesh_sha256=fixture["mesh_sha256"],
        query_observed_mask=query_mask,
        **kwargs,
    )


def _direct_qh_summary(query_mask):
    qh_partition, _ = joint.raw_pixel_partitions(IMAGE_SIZE, IMAGE_SIZE)
    eroded = audit._eroded_mask(query_mask, 6)
    centers = joint.allowed_target_center_bitmap(query_mask, qh_partition, eroded)
    return joint.qh_verification_support_upper_bound(centers, query_mask)


class SourceCapacityFixtureTests(unittest.TestCase):
    def test_fixture_has_real_r5_eligibility_default_bank_and_disjoint_literal_footprints(self):
        fixture = _fixture()
        arrays = fixture["source_arrays"]
        self.assertEqual(len(fixture["ids"]), 64)
        np.testing.assert_array_equal(
            fixture["ids"],
            np.floor(fixture["pixels"][:, 1]).astype(np.int64) * IMAGE_SIZE +
            np.floor(fixture["pixels"][:, 0]).astype(np.int64),
        )

        # Check the point formula independently from the later capacity helper.
        ray = np.column_stack((fixture["pixels"], np.ones(64))) @ np.linalg.inv(fixture["crop_k"]).T
        expected_points = ((1000.0 * ray - np.zeros(3)) @ np.eye(3)) * 0.001
        np.testing.assert_allclose(fixture["points"], expected_points, rtol=0.0, atol=1e-14)

        selector = r5.freeze_source_selection(arrays)
        self.assertEqual(selector["total_sources"], 64)
        self.assertEqual(selector["eligible_count"], 64)
        self.assertTrue(bool(np.asarray(selector["eligible"]).all()))
        bank = audit.build_template_patch_bank(
            arrays["template_rgb"], arrays["template_mask"], fixture["ids"],
            fixture["points"], grid_size=16, max_anchors=64,
        )
        self.assertEqual(bank["selected_anchor_count"], 64)
        self.assertEqual(
            [int(anchor["source_index"]) for anchor in bank["serialized_anchors"]],
            fixture["ids"].tolist(),
        )

        dependency_sets = [_literal_raw_dependencies(pixel) for pixel in fixture["pixels"]]
        for pixel, expected in zip(fixture["pixels"], dependency_sets):
            self.assertEqual(
                expected,
                set(joint.highpass_patch_raw_dependencies(pixel, IMAGE_SIZE, IMAGE_SIZE)),
            )
        self.assertEqual(len(set(fixture["ids"].tolist())), 64)
        for left in range(len(dependency_sets)):
            for right in range(left + 1, len(dependency_sets)):
                self.assertFalse(dependency_sets[left].intersection(dependency_sets[right]))

    def test_capacity_consumes_the_loader_11_field_projection_and_reports_independent_gates(self):
        fixture = _fixture(poison=True)
        arrays = fixture["source_arrays"]
        self.assertEqual(set(arrays), set(SOURCE_KEYS))
        with (mock.patch.object(r5, "_condition_source_only_inputs",
                                side_effect=AssertionError("capacity cannot reload cache packets")) as packet_spy,
              mock.patch.object(audit, "fit_learned_packet",
                                side_effect=AssertionError("capacity cannot fit poses")) as fit_spy,
              mock.patch.object(audit, "audit_patch_identity",
                                side_effect=AssertionError("capacity cannot match query patches")) as match_spy):
            result = _evaluate(fixture, source_arrays=arrays)
        self.assertEqual(packet_spy.call_count, 0)
        self.assertEqual(fit_spy.call_count, 0)
        self.assertEqual(match_spy.call_count, 0)

        self.assertEqual(result["source_population_count"], 64)
        self.assertEqual(result["r5_eligible_count"], 64)
        self.assertEqual(result["r7_eligible_count"], 64)
        self.assertTrue(result["r5_r7_bit_exact"])
        self.assertEqual(result["native_supported_count"], 64)
        self.assertEqual(result["appearance_supported_count"], 64)
        self.assertEqual(result["source_supported_count"], 64)
        self.assertEqual(result["witness_count"], 32)
        self.assertEqual(result["fit_pool_count"], 32)
        self.assertEqual(result["fit_count"], 32)
        self.assertEqual(result["bank_anchor_count"], 32)
        self.assertGreaterEqual(result["qh_possible_cell_count"], 3)
        self.assertGreaterEqual(result["qh_maximum_bbox_hull_fraction"], 0.12)
        self.assertEqual(result["failed_conjuncts"], [])
        self.assertEqual(result["state"], "available")
        self.assertTrue(result["capacity_only"])
        self.assertTrue(result["native_raycast_kernel_frozen_r7"])
        self.assertEqual(result["bank_state"], "eligible")
        self.assertFalse(result["query_rgb_or_evaluator_truth_read"])
        self.assertFalse(result["matching_started"])
        self.assertFalse(result["fitting_started"])
        self.assertFalse(result["claims"]["fullscreen"])
        self.assertFalse(result["claims"]["accuracy"])
        self.assertFalse(result["claims"]["pose"])

        # The capacity row must retain a bounded JSON-only receipt of the
        # independent native geometry, witness reservation, and fit/bank
        # provenance. It must not leak dense arrays or per-ID payloads.
        receipt = json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
        self.assertLessEqual(len(json.dumps(receipt, separators=(",", ":"))), 24_000)
        self.assertEqual(receipt["native_association"]["schema_version"], 2)
        native = receipt["native_association"]
        self.assertEqual(native["source_count"], 64)
        self.assertEqual(native["associated_count"], 64)
        self.assertEqual(native["unsupported_count"], 0)
        self.assertEqual(native["raycast_kernel_is_frozen_r7"], True)
        self.assertEqual(native["geometry"]["native_topology_state"], "supported")
        self.assertEqual(native["geometry"]["association_sha256"], result["native_association_sha256"])
        self.assertEqual(native["ray"]["requested_chunk_size"], 256)
        self.assertGreaterEqual(native["ray"]["effective_chunk_size"], 1)
        self.assertLessEqual(native["ray"]["effective_chunk_size"], 256)
        self.assertEqual(native["resource"]["state"], "within_budget")
        self.assertLessEqual(native["resource"]["estimated_peak_bytes"],
                             native["resource"]["working_budget_bytes"])

        witness = receipt["witness_reservation"]
        self.assertEqual(witness["spec_v2_sha256"].casefold(), "32758dda6ad550a5edd94f5354b7e8401ff206a63b884e8c88ab37f71ee6d9cb")
        self.assertEqual(witness["image_namespace"]["context_id"], fixture["source_context_id"])
        self.assertEqual(witness["denominator"], 32)
        self.assertEqual(witness["selected_ids_sha256"], result["witness_ids_sha256"])
        self.assertGreater(witness["dependency_union_count"], 0)
        self.assertEqual(len(witness["dependency_union_sha256"]), 64)
        self.assertIsInstance(witness["cell_counts"], dict)

        fit_bank = receipt["source_fit_bank"]
        self.assertEqual(fit_bank["witness_denominator"], 32)
        self.assertEqual(fit_bank["fit_pool_count"], 32)
        self.assertEqual(fit_bank["fit_count"], 32)
        self.assertEqual(fit_bank["bank_anchor_count"], 32)
        self.assertEqual(fit_bank["fit_union_intersection_with_witness_count"], 0)
        self.assertEqual(fit_bank["bank_intersection_with_witness_count"], 0)
        self.assertGreater(fit_bank["fit_dependency_union_count"], 0)
        self.assertGreater(fit_bank["bank_dependency_union_count"], 0)
        for field in ("fit_pool_dependency_sha256", "fit_dependency_sha256",
                      "fit_dependency_union_sha256", "bank_anchor_dependency_sha256",
                      "bank_dependency_union_sha256", "bank_manifest_sha256"):
            self.assertEqual(len(fit_bank[field]), 64, field)

        def assert_compact_json_only(value):
            self.assertNotIsInstance(value, np.ndarray)
            if isinstance(value, dict):
                for child in value.values():
                    assert_compact_json_only(child)
            elif isinstance(value, list):
                self.assertLessEqual(len(value), 64)
                for child in value:
                    assert_compact_json_only(child)
        assert_compact_json_only(receipt)

    def test_nonpositive_depth_and_raw_dependency_mask_holes_reduce_source_support(self):
        fixture = _fixture()
        independent_dependencies = [
            _literal_raw_dependencies(pixel) for pixel in fixture["pixels"]
        ]
        poison_pixel = 1 * IMAGE_SIZE + 1
        self.assertIn(poison_pixel, independent_dependencies[0])
        self.assertTrue(all(poison_pixel not in dependencies
                            for dependencies in independent_dependencies[1:]))

        for label, field, invalid_value in (
                ("nonpositive raw depth", "template_depth_mm", 0.0),
                ("false original support mask", "observed_crop_mask", 0)):
            with self.subTest(case=label):
                changed = dict(fixture["source_arrays"])
                changed_array = np.array(changed[field], copy=True)
                changed_array[1, 1] = invalid_value
                changed[field] = changed_array
                result = _evaluate(fixture, source_arrays=changed)
                self.assertEqual(result["source_population_count"], 64)
                self.assertEqual(result["appearance_supported_count"], 63)
                self.assertEqual(result["source_supported_count"], 63)
                self.assertGreaterEqual(result["witness_count"], 8)
                self.assertGreaterEqual(result["fit_pool_count"], 24)
                self.assertGreaterEqual(result["bank_anchor_count"], 1)

    def test_eleven_field_contract_rejects_full_thirteen_array_packet_projection(self):
        fixture = _fixture()
        arrays = dict(fixture["source_arrays"])
        arrays["template_gray_rgb"] = arrays["template_rgb"]
        arrays["seed_pose_m"] = arrays["template_pose_m"]
        with mock.patch.object(r5, "freeze_source_selection",
                               side_effect=AssertionError("wrong projection must fail before R5")) as spy:
            with self.assertRaises(capacity.SourceCapacityError):
                capacity.evaluate_source_capacity(
                    arrays,
                    source_context_id=fixture["source_context_id"],
                    packet_sha256=fixture["packet_sha256"],
                    mesh_positions_m=fixture["vertices"],
                    mesh_triangles=fixture["triangles"],
                    mesh_sha256=fixture["mesh_sha256"],
                )
        self.assertEqual(spy.call_count, 0)

    def test_wrong_source_point_is_counted_unsupported_before_source_selection(self):
        fixture = _fixture()
        original = fixture["source_arrays"]
        changed = dict(original)
        changed_points = np.array(fixture["points"], copy=True)
        changed_points[0, 0] += 0.25
        changed["source_points_object_m"] = changed_points

        result = _evaluate(fixture, source_arrays=changed)
        self.assertEqual(result["source_population_count"], 64)
        self.assertEqual(result["native_supported_count"], 63)
        self.assertEqual(
            result["native_reason_counts"]["native_first_hit_point_error_over_epsilon"], 1)
        self.assertEqual(result["source_supported_count"], 63)
        self.assertGreaterEqual(result["witness_count"], 8)
        self.assertGreaterEqual(result["fit_pool_count"], 24)
        self.assertGreaterEqual(result["bank_anchor_count"], 1)
        self.assertEqual(result["state"], "available")

    def test_witness_capacity_survives_zero_disjoint_fit_pool_and_single_empty_bank(self):
        fixture = _fixture()
        arrays = dict(fixture["source_arrays"])
        # Retain original rows at checkerboard grid positions 0,2,4,6 in both
        # axes. Each of the 16 observed-bbox cells has one source; their true
        # D(q) footprints are disjoint, so every selected point stays in W and
        # the complete W union leaves no fit or structural-bank candidate.
        keep = np.asarray([grid_y * 8 + grid_x
                           for grid_y in (0, 2, 4, 6)
                           for grid_x in (0, 2, 4, 6)], dtype=np.int64)
        for field in ("source_indices", "source_pixels_xy", "source_points_object_m"):
            arrays[field] = np.ascontiguousarray(arrays[field][keep])
        self.assertEqual(len(arrays["source_indices"]), 16)

        with (mock.patch.object(capacity.selection, "select_fit_and_bank",
                                wraps=capacity.selection.select_fit_and_bank) as selection_spy,
              mock.patch.object(audit, "build_template_patch_bank",
                                wraps=audit.build_template_patch_bank) as bank_spy):
            result = _evaluate(fixture, source_arrays=arrays)

        self.assertEqual(selection_spy.call_count, 1)
        self.assertEqual(bank_spy.call_count, 1)
        self.assertEqual(bank_spy.call_args.kwargs["grid_size"], 16)
        self.assertEqual(bank_spy.call_args.kwargs["max_anchors"], 64)
        self.assertEqual(result["source_population_count"], 16)
        self.assertEqual(result["native_supported_count"], 16)
        self.assertEqual(result["appearance_supported_count"], 16)
        self.assertEqual(result["source_supported_count"], 16)
        self.assertEqual(result["witness_count"], 16)
        self.assertEqual(result["fit_pool_count"], 0)
        self.assertEqual(result["fit_count"], 0)
        self.assertEqual(result["bank_anchor_count"], 0)
        self.assertTrue(result["gates"]["witnesses_ge_8"])
        self.assertTrue(result["gates"]["qh_possible_cells_ge_3"])
        self.assertTrue(result["gates"]["qh_maximum_bbox_hull_ge_0_12"])
        self.assertFalse(result["gates"]["fit_ge_24"])
        self.assertFalse(result["gates"]["bank_ge_1"])
        self.assertEqual(result["failed_conjuncts"], ["fit_ge_24", "bank_ge_1"])
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["witness_reservation"]["denominator"], 16)
        self.assertEqual(result["source_fit_bank"]["witness_denominator"], 16)
        self.assertEqual(result["source_fit_bank"]["fit_pool_count"], 0)
        self.assertEqual(result["source_fit_bank"]["fit_count"], 0)
        self.assertEqual(result["source_fit_bank"]["bank_anchor_count"], 0)
        self.assertEqual(result["source_fit_bank"]["bank_builder_calls"], 1)

    def test_qh_bounds_use_the_observed_bbox_and_fail_independently_of_source_counts(self):
        fixture = _fixture()
        baseline = _evaluate(fixture)
        source_counts = tuple(baseline[key] for key in (
            "source_population_count", "r5_eligible_count", "native_supported_count",
            "appearance_supported_count", "source_supported_count", "witness_count",
            "fit_pool_count", "fit_count", "bank_anchor_count",
        ))

        # A 60×60 off-centre observed crop sits wholly in one full-image 4×4
        # checkerboard cell, yet its own observed-bbox grid represents 16 cells.
        offcentre = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        offcentre[80:140, 80:140] = 1
        direct_possible = _direct_qh_summary(offcentre)
        self.assertEqual(direct_possible["state"], "possible_upper_bound")
        self.assertEqual(direct_possible["represented_cell_count"], 16)
        self.assertAlmostEqual(direct_possible["maximum_hull_fraction"], 0.36, places=12)
        possible = _evaluate(fixture, query_mask=offcentre)
        self.assertEqual(possible["qh_possible_center_count"], direct_possible["allowed_center_count"])
        self.assertEqual(possible["qh_possible_cell_count"], direct_possible["represented_cell_count"])
        self.assertAlmostEqual(possible["qh_maximum_bbox_hull_fraction"],
                               direct_possible["maximum_hull_fraction"], places=12)
        self.assertEqual(tuple(possible[key] for key in (
            "source_population_count", "r5_eligible_count", "native_supported_count",
            "appearance_supported_count", "source_supported_count", "witness_count",
            "fit_pool_count", "fit_count", "bank_anchor_count",
        )), source_counts)
        self.assertEqual(possible["state"], "available")

        # Two small observed components expose only two cells in their own
        # bbox and also fail the independent 0.12 hull upper bound.
        two_cells = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        two_cells[30:62, 30:62] = 1
        two_cells[230:262, 230:262] = 1
        direct_impossible = _direct_qh_summary(two_cells)
        self.assertEqual(direct_impossible["state"], "known_impossible")
        self.assertEqual(direct_impossible["represented_cell_count"], 2)
        self.assertLess(direct_impossible["maximum_hull_fraction"], 0.12)
        impossible = _evaluate(fixture, query_mask=two_cells)
        self.assertEqual(impossible["qh_possible_center_count"], direct_impossible["allowed_center_count"])
        self.assertEqual(impossible["qh_possible_cell_count"], direct_impossible["represented_cell_count"])
        self.assertAlmostEqual(impossible["qh_maximum_bbox_hull_fraction"],
                               direct_impossible["maximum_hull_fraction"], places=12)
        self.assertEqual(tuple(impossible[key] for key in (
            "source_population_count", "r5_eligible_count", "native_supported_count",
            "appearance_supported_count", "source_supported_count", "witness_count",
            "fit_pool_count", "fit_count", "bank_anchor_count",
        )), source_counts)
        self.assertEqual(impossible["state"], "unavailable")
        self.assertEqual(impossible["failed_conjuncts"], [
            "qh_possible_cells_ge_3", "qh_maximum_bbox_hull_ge_0_12",
        ])

    def test_memory_and_expired_deadline_reject_before_source_helpers(self):
        fixture = _fixture()
        with (mock.patch.object(r5, "freeze_source_selection",
                                side_effect=AssertionError("cap check must precede R5")) as r5_spy,
              mock.patch.object(geometry, "native_source_association_all",
                                side_effect=AssertionError("cap check must precede geometry")) as geometry_spy):
            with self.assertRaises(geometry.SourceCapacityLimitError):
                _evaluate(fixture, resident_bytes=capacity.MAX_CAPACITY_BYTES)
            with self.assertRaises(geometry.SourceCapacityLimitError):
                _evaluate(fixture, deadline_monotonic=__import__("time").monotonic() - 1.0)
        self.assertEqual(r5_spy.call_count, 0)
        self.assertEqual(geometry_spy.call_count, 0)


if __name__ == "__main__":
    unittest.main()
