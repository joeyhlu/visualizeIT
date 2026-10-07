"""Independent CPU regressions for R8's source-first selection core."""
from __future__ import annotations

import hashlib
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_reservation_selection as selection


SPEC_V2_SHA256 = "32758dda6ad550a5edd94f5354b7e8401ff206a63b884e8c88ab37f71ee6d9cb"
SPEC_V2_PATH = Path(__file__).resolve().parents[1] / ".cache" / "bottle-source-reservation-r8-spec-v2.md"


def _literal_reflect101(index: int, length: int) -> int:
    """OpenCV BORDER_REFLECT_101 index, expanded without project helpers."""
    if length <= 0:
        raise ValueError("image axis must be nonempty")
    if length == 1:
        return 0
    period = 2 * (length - 1)
    folded = index % period
    return folded if folded < length else period - folded


def _literal_bilinear_axis_support(array_coordinate: float, length: int) -> tuple[int, ...]:
    lower = math.floor(array_coordinate)
    fraction = array_coordinate - lower
    taps = []
    if 1.0 - fraction != 0.0:
        taps.append(lower)
    if fraction != 0.0:
        taps.append(lower + 1)
    if any(index < 0 or index >= length for index in taps):
        raise IndexError("nonzero bilinear tap falls outside the image")
    return tuple(taps)


def _literal_raw_dependencies(center_xy, width: int, height: int) -> tuple[int, ...]:
    """Literal D(q): 11x11 nonzero bilinear taps followed by 17x17 blur support."""
    qx, qy = (float(value) for value in np.asarray(center_xy, dtype=np.float64).reshape(2))
    raw_x: set[int] = set()
    raw_y: set[int] = set()
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            # Source coordinates use edge coordinates; array sample centers are
            # at integer coordinates after subtracting one half pixel.
            x_taps = _literal_bilinear_axis_support(qx - 0.5 + dx, width)
            y_taps = _literal_bilinear_axis_support(qy - 0.5 + dy, height)
            for x in x_taps:
                raw_x.update(_literal_reflect101(x + blur_dx, width)
                             for blur_dx in range(-8, 9))
            for y in y_taps:
                raw_y.update(_literal_reflect101(y + blur_dy, height)
                             for blur_dy in range(-8, 9))
    return tuple(sorted(y * width + x for y in raw_y for x in raw_x))


def _reservation_fixture():
    width = height = 512
    observed = np.ones((height, width), dtype=bool)
    ids, pixels, std, eligible, supported = [], [], [], [], []
    expected = []
    overlap_rejects = []
    for cell_y in range(4):
        for cell_x in range(4):
            x0, y0 = cell_x * (width // 4), cell_y * (height // 4)
            y = y0 + 64
            # These two high-STD decoys must be removed before ranking: one is
            # unsupported and the other failed R5 eligibility.
            rows = (
                (x0 + 20, 3.0, True, False),
                (x0 + 24, 2.0, False, True),
                (x0 + 30, 0.90, True, True),
                (x0 + 33, 0.85, True, True),  # D(q) overlaps the first winner.
                (x0 + 75, 0.80, True, True),
                (x0 + 115, 0.80, True, True),  # ID tie-break loses quota position.
            )
            winners_for_cell = []
            for x, score, r5_ok, source_ok in rows:
                source_id = y * width + x
                ids.append(source_id)
                pixels.append((x + 0.5, y + 0.5))
                std.append(score)
                eligible.append(r5_ok)
                supported.append(source_ok)
                if x == x0 + 30 or x == x0 + 75:
                    winners_for_cell.append(source_id)
                if x == x0 + 33:
                    overlap_rejects.append(source_id)
            expected.extend(winners_for_cell)

    return dict(
        width=width,
        height=height,
        observed=observed,
        ids=np.asarray(ids, dtype=np.int64),
        pixels=np.asarray(pixels, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
        eligible=np.asarray(eligible, dtype=bool),
        supported=np.asarray(supported, dtype=bool),
        expected_ids=expected,
        expected_overlap_rejects=overlap_rejects,
    )


def _literal_pixel_center_dependencies(source_id: int, width: int, height: int) -> tuple[int, ...]:
    """Expand the pixel-center support as one contiguous blur-expanded span."""
    x, y = int(source_id) % width, int(source_id) // width
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("source ID is outside its image")
    # For q=[x+.5,y+.5], all 11 patch samples have integer array centers.
    # Their nonzero source taps cover x/y +/-5; sigma-2's 17-tap blur adds 8
    # on every side. Expand every raw index through the literal REFLECT_101
    # map, then form the Cartesian product of the separable 2D support.
    raw_x = sorted({_literal_reflect101(index, width)
                    for index in range(x - 13, x + 14)})
    raw_y = sorted({_literal_reflect101(index, height)
                    for index in range(y - 13, y + 14)})
    return tuple(yy * width + xx for yy in raw_y for xx in raw_x)


class _PoisonSourceArrays(dict):
    """Mapping that fails if a capacity consumer reads query or truth fields."""

    FORBIDDEN = frozenset(("query_rgb", "query_depth_mm", "evaluator_truth",
                           "annotations", "reference_pose", "ncc_scores", "matches"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden_reads = []

    def __getitem__(self, key):
        if key in self.FORBIDDEN:
            self.forbidden_reads.append(key)
            raise AssertionError(f"source capacity read forbidden field {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.FORBIDDEN:
            self.forbidden_reads.append(key)
            raise AssertionError(f"source capacity read forbidden field {key!r}")
        return super().get(key, default)


def _bank_fixture():
    width = height = 512
    rng = np.random.default_rng(832771)
    by_id = {}
    grid = np.rint((np.arange(16, dtype=np.float64) + 0.5) * width / 16).astype(int)
    observed = np.zeros((height, width), dtype=bool)
    observed[8:504, 8:504] = True
    for cell_y in range(4):
        for cell_x in range(4):
            x0, y0 = 8 + cell_x * 124, 8 + cell_y * 124
            # A stratified random population gives each source-grid cell more
            # than the frozen 3000 fit cap after independent S_H exclusion.
            offsets = rng.choice(84 * 61, size=500, replace=False)
            for offset in offsets.tolist():
                x = x0 + 20 + offset % 84
                y = y0 + 20 + offset // 84
                by_id[y * width + x] = (0.5, True, True)

            # Supply every possible grid anchor to the real, frozen R5 bank
            # selector. Two selected witnesses per 4x4 cell are declared at
            # the top row; bank candidates beyond those remain available.
            for grid_y in range(4 * cell_y, 4 * cell_y + 4):
                for grid_x in range(4 * cell_x, 4 * cell_x + 4):
                    x, y = int(grid[grid_x]), int(grid[grid_y])
                    supported = 21 <= x <= 490 and 21 <= y <= 490
                    by_id[y * width + x] = (0.25, True, supported)
            witness_a = (y0 + 30) * width + (x0 + 30)
            witness_b = (y0 + 30) * width + (x0 + 80)
            by_id[witness_a] = (10.0, True, True)
            by_id[witness_b] = (9.0, True, True)

            # Higher-scoring but disqualified rows prove source support and R5
            # eligibility are intersected before W ranking.
            unsupported = (y0 + 30) * width + (x0 + 32)
            r5_rejected = (y0 + 30) * width + (x0 + 35)
            by_id[unsupported] = (100.0, True, False)
            by_id[r5_rejected] = (99.0, False, True)

    ids = np.asarray(sorted(by_id), dtype=np.int64)
    x = ids % width
    y = ids // width
    pixels = np.column_stack((x + 0.5, y + 0.5)).astype(np.float64, copy=False)
    source_std = np.asarray([by_id[int(source_id)][0] for source_id in ids], dtype=np.float64)
    r5_eligible = np.asarray([by_id[int(source_id)][1] for source_id in ids], dtype=bool)
    supported = np.asarray([by_id[int(source_id)][2] for source_id in ids], dtype=bool)
    points = np.column_stack((x * 0.001, y * 0.001,
                              0.15 + ((3 * x + y) % 97) * 0.001)).astype(np.float64)
    rgb = rng.random((height, width, 3), dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)
    namespace = {"context_id": "synthetic-template-source-000",
                 "source_rgb_sha256": audit.array_sha256(rgb)}
    arrays = _PoisonSourceArrays(
        template_rgb=rgb,
        template_mask=mask,
        observed_crop_mask=observed,
        query_rgb=object(),
        query_depth_mm=object(),
        evaluator_truth=object(),
        annotations=object(),
        reference_pose=object(),
        ncc_scores=object(),
        matches=object(),
    )
    return dict(width=width, height=height, ids=ids, pixels=pixels, std=source_std,
                eligible=r5_eligible, supported=supported, points=points, rgb=rgb,
                mask=mask, observed=observed, namespace=namespace, arrays=arrays)


def _namespace(context_id, rgb_sha256="0" * 64):
    return {"context_id": context_id, "source_rgb_sha256": rgb_sha256}


def _reserve(fixture, *, namespace=None):
    width, height = fixture["width"], fixture["height"]
    if namespace is None:
        namespace = _namespace("template-frame-000")
    return selection.reserve_witnesses(
        fixture["ids"], fixture["pixels"], fixture["std"],
        fixture["eligible"], fixture["supported"], fixture["observed"],
        lambda q: _literal_raw_dependencies(q, width, height),
        image_namespace=namespace,
    )


def _bank_dependency_fn(width, height):
    def dependencies(center_xy):
        x, y = np.asarray(center_xy, dtype=np.float64).reshape(2)
        pixel_x, pixel_y = int(round(x - 0.5)), int(round(y - 0.5))
        if x != pixel_x + 0.5 or y != pixel_y + 0.5:
            raise AssertionError("fixture source IDs must remain bound to pixel centers")
        return _literal_pixel_center_dependencies(pixel_y * width + pixel_x, width, height)
    return dependencies


def _independent_fit_pool(fixture, selected_witness_ids):
    width, height = fixture["width"], fixture["height"]
    witness_dependencies = set()
    for source_id in selected_witness_ids:
        witness_dependencies.update(
            _literal_pixel_center_dependencies(int(source_id), width, height))
    pool = []
    for source_id, is_eligible, is_supported in zip(
            fixture["ids"], fixture["eligible"], fixture["supported"]):
        source_id = int(source_id)
        if not (bool(is_eligible) and bool(is_supported)):
            continue
        dependencies = _literal_pixel_center_dependencies(source_id, width, height)
        if all(dependency not in witness_dependencies for dependency in dependencies):
            pool.append(source_id)
    return pool, witness_dependencies


def _independent_fit_round_robin(fixture, pool_ids, max_samples=3000):
    observed = fixture.get("observed", fixture.get("mask"))
    ys, xs = np.nonzero(observed)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_width, bbox_height = max_x - min_x + 1, max_y - min_y + 1
    buckets = [[] for _ in range(16)]
    for source_id in pool_ids:
        source_id = int(source_id)
        x, y = source_id % fixture["width"], source_id // fixture["width"]
        cell_x = min(3, max(0, (4 * (x - min_x)) // bbox_width))
        cell_y = min(3, max(0, (4 * (y - min_y)) // bbox_height))
        buckets[cell_y * 4 + cell_x].append(source_id)
    for bucket in buckets:
        bucket.sort()
    ordered = []
    depth = 0
    while len(ordered) < max_samples:
        advanced = False
        for bucket in buckets:
            if depth < len(bucket) and len(ordered) < max_samples:
                ordered.append(bucket[depth])
                advanced = True
        if not advanced:
            break
        depth += 1
    return ordered


class FrozenSpecTests(unittest.TestCase):
    def test_v2_spec_hash_is_pinned(self):
        self.assertEqual(hashlib.sha256(SPEC_V2_PATH.read_bytes()).hexdigest(), SPEC_V2_SHA256)


class WitnessReservationTests(unittest.TestCase):
    def test_literal_footprints_drive_ranked_two_per_cell_reservation(self):
        fixture = _reservation_fixture()
        result = _reserve(fixture)

        self.assertEqual(result["state"], "reserved")
        self.assertEqual(result["image_namespace"], _namespace("template-frame-000"))
        self.assertEqual(result["source_count"], len(fixture["ids"]))
        self.assertEqual(result["r5_eligible_count"], int(fixture["eligible"].sum()))
        self.assertEqual(result["source_supported_count"], int(fixture["supported"].sum()))
        self.assertEqual(result["denominator"], 32)
        self.assertEqual(result["reserved_count"], 32)
        np.testing.assert_array_equal(result["selected_ids"], fixture["expected_ids"])
        self.assertEqual(result["cell_counts"], {str(cell): 2 for cell in range(16)})
        self.assertEqual(set(result["rejected_witness_overlap_ids"].tolist()),
                         set(fixture["expected_overlap_rejects"]))

        # Reconstruct every reserved dependency and S_H from the literal
        # bilinear/Gaussian support oracle; do not trust serialized counts.
        id_to_pixel = {int(source_id): xy
                       for source_id, xy in zip(fixture["ids"], fixture["pixels"])}
        per_witness = [_literal_raw_dependencies(id_to_pixel[int(source_id)],
                                                 fixture["width"], fixture["height"])
                       for source_id in result["selected_ids"]]
        for left in range(len(per_witness)):
            for right in range(left + 1, len(per_witness)):
                self.assertTrue(set(per_witness[left]).isdisjoint(per_witness[right]))
        expected_union = sorted(set().union(*(set(values) for values in per_witness)))
        np.testing.assert_array_equal(result["dependency_union_ids"], expected_union)
        self.assertEqual(result["dependency_union_count"], len(expected_union))
        self.assertEqual(result["dependency_union_mask"].shape,
                         fixture["observed"].shape)
        self.assertEqual(int(result["dependency_union_mask"].sum()), len(expected_union))
        self.assertEqual(set(np.flatnonzero(result["dependency_union_mask"]).tolist()),
                         set(expected_union))

    def test_shuffled_aligned_candidates_preserve_ordered_reservation_and_seal(self):
        fixture = _reservation_fixture()
        first = _reserve(fixture)
        order = np.random.default_rng(221).permutation(len(fixture["ids"]))
        shuffled = dict(fixture)
        for name in ("ids", "pixels", "std", "eligible", "supported"):
            shuffled[name] = fixture[name][order]
        second = _reserve(shuffled)

        for key in ("selected_ids", "dependency_union_ids", "rejected_witness_overlap_ids"):
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        for key in ("state", "selected_ids_sha256", "denominator", "dependency_union_count",
                    "dependency_union_sha256"):
            self.assertEqual(first[key], second[key], key)

    def test_image_namespace_qualifies_reused_integer_ids(self):
        fixture = _reservation_fixture()
        shared_hash = "5a" * 32
        zero_ns = _namespace("template-frame-000", shared_hash)
        half_turn_ns = _namespace("template-frame-180", shared_hash)
        revised_source_ns = _namespace("template-frame-000", "c3" * 32)
        zero = _reserve(fixture, namespace=zero_ns)
        half_turn = _reserve(fixture, namespace=half_turn_ns)
        revised_source = _reserve(fixture, namespace=revised_source_ns)

        np.testing.assert_array_equal(zero["selected_ids"], half_turn["selected_ids"])
        np.testing.assert_array_equal(zero["qualified_ids"], half_turn["qualified_ids"])
        self.assertEqual(zero["selected_source_identity"]["image_namespace"], zero_ns)
        np.testing.assert_array_equal(
            zero["selected_source_identity"]["source_ids"], zero["selected_ids"])
        self.assertEqual(half_turn["selected_source_identity"]["image_namespace"], half_turn_ns)
        self.assertNotEqual(zero["selected_source_identity"]["source_ids_sha256"],
                            half_turn["selected_source_identity"]["source_ids_sha256"])
        self.assertEqual(revised_source["selected_ids"].tolist(), zero["selected_ids"].tolist())
        self.assertNotEqual(zero["selected_source_identity"]["source_ids_sha256"],
                            revised_source["selected_source_identity"]["source_ids_sha256"])
        self.assertNotEqual(zero["image_namespace"], half_turn["image_namespace"])
        self.assertNotEqual(zero["image_namespace_sha256"], half_turn["image_namespace_sha256"])
        self.assertNotEqual(zero["selected_ids_sha256"], half_turn["selected_ids_sha256"])

    def test_dependency_bound_is_checked_before_python_set_construction(self):
        fixture = _reservation_fixture()
        armed = False
        builtin_set = set

        def guarded_set(*args, **kwargs):
            if armed:
                raise AssertionError("oversized dependency IDs were materialized as a Python set")
            return builtin_set(*args, **kwargs)

        def oversized_dependencies(_center_xy):
            nonlocal armed
            armed = True
            return np.arange(730, dtype=np.int64)

        with mock.patch.object(selection, "set", guarded_set, create=True):
            with self.assertRaises(selection.SourceReservationError):
                selection.reserve_witnesses(
                    fixture["ids"], fixture["pixels"], fixture["std"],
                    fixture["eligible"], fixture["supported"], fixture["observed"],
                    oversized_dependencies,
                    image_namespace=_namespace("template-frame-000"),
                )

    def test_unavailable_selection_keeps_original_population_and_no_refill(self):
        fixture = _reservation_fixture()
        keep = np.zeros(len(fixture["ids"]), dtype=bool)
        # Seven supported, eligible and spatially separate IDs cannot reach the
        # frozen eight-witness minimum. Unsupported candidates remain in N's
        # original source population and cannot be used to refill this result.
        keep_rows = [index for index, source_id in enumerate(fixture["ids"])
                     if int(source_id) in fixture["expected_ids"][:7]]
        keep[keep_rows] = True
        fixture["eligible"] = fixture["eligible"] & keep
        fixture["supported"] = fixture["supported"] & keep

        result = _reserve(fixture)
        self.assertEqual(result["source_count"], len(fixture["ids"]))
        self.assertEqual(result["denominator"], 7)
        self.assertEqual(result["reserved_count"], 7)
        np.testing.assert_array_equal(result["selected_ids"], fixture["expected_ids"][:7])
        self.assertEqual(result["state"], "unavailable")


class FitPoolAndBankTests(unittest.TestCase):
    def _reserve_bank_fixture(self, fixture):
        dependency_fn = _bank_dependency_fn(fixture["width"], fixture["height"])
        witness = selection.reserve_witnesses(
            fixture["ids"], fixture["pixels"], fixture["std"],
            fixture["eligible"], fixture["supported"], fixture["observed"],
            dependency_fn, image_namespace=fixture["namespace"])
        return dependency_fn, witness

    def test_full_disjoint_pool_builds_one_real_audit_bank_with_original_bindings(self):
        fixture = _bank_fixture()
        dependency_fn, witness = self._reserve_bank_fixture(fixture)
        expected_witness_ids = []
        for cell_y in range(4):
            for cell_x in range(4):
                x0, y0 = 8 + cell_x * 124, 8 + cell_y * 124
                expected_witness_ids.extend(((y0 + 30) * fixture["width"] + x0 + 30,
                                             (y0 + 30) * fixture["width"] + x0 + 80))
        np.testing.assert_array_equal(witness["selected_ids"], expected_witness_ids)
        self.assertEqual(witness["denominator"], 32)

        # The fast pixel-center span and the literal 11x11-by-17x17 oracle
        # must agree before it is used to validate thousands of source rows.
        sample_ids = [int(fixture["ids"][index])
                      for index in np.linspace(0, len(fixture["ids"]) - 1, 17, dtype=int)]
        for source_id in sample_ids:
            pixel = fixture["pixels"][np.searchsorted(fixture["ids"], source_id)]
            np.testing.assert_array_equal(
                _literal_pixel_center_dependencies(source_id, fixture["width"], fixture["height"]),
                _literal_raw_dependencies(pixel, fixture["width"], fixture["height"]))

        expected_pool, witness_dependency_set = _independent_fit_pool(
            fixture, witness["selected_ids"])
        self.assertGreater(len(expected_pool), 3000)
        source_point_by_id = {int(source_id): fixture["points"][row]
                              for row, source_id in enumerate(fixture["ids"])}
        call_events = []
        builder_calls = []
        original_reserve = selection.reserve_witnesses

        def reserve_spy(*args, **kwargs):
            call_events.append("reserve")
            return original_reserve(*args, **kwargs)

        def real_audit_builder(rgb, mask, filtered_ids, filtered_points,
                               grid_size=16, max_anchors=64):
            call_events.append("bank")
            call = {
                "rgb": rgb,
                "mask": mask,
                "ids": np.asarray(filtered_ids).copy(),
                "points": np.asarray(filtered_points).copy(),
                "grid_size": grid_size,
                "max_anchors": max_anchors,
            }
            call["bank"] = audit.build_template_patch_bank(
                rgb, mask, filtered_ids, filtered_points,
                grid_size=grid_size, max_anchors=max_anchors)
            builder_calls.append(call)
            return call["bank"]

        with mock.patch.object(selection, "reserve_witnesses", side_effect=reserve_spy):
            result = selection.select_fit_and_bank(
                fixture["arrays"], fixture["ids"], fixture["pixels"], fixture["points"],
                fixture["std"], fixture["eligible"], fixture["supported"], witness,
                dependency_fn, bank_builder=real_audit_builder,
                image_namespace=fixture["namespace"])

        self.assertEqual(call_events, ["reserve", "bank"])
        self.assertEqual(len(builder_calls), 1)
        self.assertIs(builder_calls[0]["rgb"], fixture["rgb"])
        self.assertIs(builder_calls[0]["mask"], fixture["mask"])
        self.assertEqual(builder_calls[0]["grid_size"], 16)
        self.assertEqual(builder_calls[0]["max_anchors"], 64)
        np.testing.assert_array_equal(result["fit_pool_ids"], expected_pool)
        self.assertEqual(result["fit_pool_count"], len(expected_pool))
        np.testing.assert_array_equal(builder_calls[0]["ids"], expected_pool)
        np.testing.assert_array_equal(
            builder_calls[0]["points"],
            np.asarray([source_point_by_id[source_id] for source_id in expected_pool]))

        expected_fit_ids = _independent_fit_round_robin(fixture, expected_pool)
        self.assertEqual(len(expected_fit_ids), 3000)
        np.testing.assert_array_equal(result["fit_ids"], expected_fit_ids)
        self.assertEqual(result["fit_count"], 3000)
        self.assertGreater(result["fit_pool_count"], result["fit_count"])
        self.assertEqual(result["fit_union_intersection_with_witness_count"], 0)
        self.assertEqual(result["bank_intersection_with_witness_count"], 0)

        bank = builder_calls[0]["bank"]
        public_anchors = bank["serialized_anchors"]
        private_anchors = bank["anchors"]
        self.assertEqual(len(public_anchors), bank["selected_anchor_count"])
        self.assertGreaterEqual(len(public_anchors), 1)
        self.assertLessEqual(len(public_anchors), 64)
        anchor_ids = []
        expected_bank_union = set()
        fit_id_set = set(expected_fit_ids)
        pool_set = set(expected_pool)
        for private, public in zip(private_anchors, public_anchors):
            self.assertIn("source_index", public)
            source_id = int(public["source_index"])
            anchor_ids.append(source_id)
            self.assertIn(source_id, pool_set)
            x, y = source_id % fixture["width"], source_id // fixture["width"]
            expected_xy = np.asarray((x + 0.5, y + 0.5), dtype=np.float64)
            expected_point = np.asarray(source_point_by_id[source_id], dtype=np.float64)
            np.testing.assert_array_equal(private["source_xy_crop"], expected_xy)
            np.testing.assert_array_equal(private["object_xyz_m"], expected_point)
            self.assertEqual(public["source_xy_crop"], expected_xy.tolist())
            self.assertEqual(public["object_xyz_m"], expected_point.tolist())
            self.assertEqual(private["source_index"], source_id)
            dependencies = set(_literal_pixel_center_dependencies(
                source_id, fixture["width"], fixture["height"]))
            self.assertTrue(dependencies.isdisjoint(witness_dependency_set))
            expected_bank_union.update(dependencies)

        np.testing.assert_array_equal(result["bank_anchor_ids"], anchor_ids)
        self.assertEqual(result["bank_anchor_count"], len(anchor_ids))
        self.assertTrue(set(anchor_ids).issubset(pool_set))
        self.assertTrue(set(anchor_ids).difference(fit_id_set),
                        "the full source pool, including IDs beyond the 3000 fit cap, must feed the bank")
        np.testing.assert_array_equal(result["bank_dependency_union_ids"],
                                      sorted(expected_bank_union))
        self.assertEqual(result["bank_dependency_union_count"], len(expected_bank_union))
        self.assertEqual(result["bank_builder_calls"], 1)
        self.assertEqual(result["state"], "source_pool_and_bank_ready")
        self.assertTrue(result["capacity_only"])
        self.assertFalse(result["pose_quality_claim"])
        np.testing.assert_array_equal(result["witness_selected_ids"], witness["selected_ids"])
        self.assertEqual(result["witness_denominator"], witness["denominator"])
        self.assertEqual(fixture["arrays"].forbidden_reads, [])

    def test_empty_bank_is_terminal_without_witness_refill_or_second_build(self):
        fixture = _reservation_fixture()
        rng = np.random.default_rng(552)
        rgb = rng.random((fixture["height"], fixture["width"], 3), dtype=np.float32)
        mask = np.ones((fixture["height"], fixture["width"]), dtype=bool)
        namespace = {"context_id": "synthetic-empty-bank", "source_rgb_sha256": audit.array_sha256(rgb)}
        arrays = _PoisonSourceArrays(
            template_rgb=rgb, template_mask=mask, observed_crop_mask=mask,
            query_rgb=object(), evaluator_truth=object())
        points = np.column_stack((fixture["ids"] % fixture["width"] * 0.001,
                                  fixture["ids"] // fixture["width"] * 0.001,
                                  np.full(len(fixture["ids"]), 0.4))).astype(np.float64)
        witness = selection.reserve_witnesses(
            fixture["ids"], fixture["pixels"], fixture["std"], fixture["eligible"],
            fixture["supported"], mask,
            lambda q: _literal_raw_dependencies(q, fixture["width"], fixture["height"]),
            image_namespace=namespace)
        calls = []

        def empty_bank_builder(rgb_arg, mask_arg, ids_arg, points_arg,
                               grid_size=16, max_anchors=64):
            calls.append((grid_size, max_anchors, len(ids_arg)))
            return {"state": "no_distinctive_source_patches", "candidate_grid_count": 256,
                    "eligible_source_patch_count": 0, "selected_anchor_count": 0,
                    "anchors": [], "serialized_anchors": []}

        result = selection.select_fit_and_bank(
            arrays, fixture["ids"], fixture["pixels"], points, fixture["std"],
            fixture["eligible"], fixture["supported"], witness,
            lambda q: _literal_raw_dependencies(q, fixture["width"], fixture["height"]),
            bank_builder=empty_bank_builder, image_namespace=namespace)

        self.assertEqual(calls, [(16, 64, result["fit_pool_count"])])
        self.assertEqual(result["bank_builder_calls"], 1)
        self.assertEqual(result["bank_anchor_count"], 0)
        self.assertEqual(result["state"], "unavailable")
        np.testing.assert_array_equal(result["witness_selected_ids"], witness["selected_ids"])
        self.assertEqual(result["witness_denominator"], witness["denominator"])
        self.assertEqual(result["witness_denominator"], 32)
        self.assertEqual(arrays.forbidden_reads, [])

    def test_empty_disjoint_fit_pool_hashes_zero_rows_and_keeps_witness_denominator(self):
        width = height = 512
        observed = np.ones((height, width), dtype=bool)
        ids = []
        pixels = []
        # One source in each frozen 4x4 observed-bbox cell, with enough space
        # between centers that their independently expanded D(q) sets do not
        # overlap. Every selected source is itself in W, so S_H excludes the
        # full qualified population and leaves a correctly empty fit pool.
        for cell_y in range(4):
            for cell_x in range(4):
                x = cell_x * 128 + 30
                y = cell_y * 128 + 30
                ids.append(y * width + x)
                pixels.append((x + 0.5, y + 0.5))
        ids = np.asarray(ids, dtype=np.int64)
        pixels = np.asarray(pixels, dtype=np.float64)
        std = np.linspace(1.0, 2.0, len(ids), dtype=np.float64)
        eligible = np.ones(len(ids), dtype=bool)
        supported = np.ones(len(ids), dtype=bool)
        points = np.column_stack((pixels[:, 0] * 0.001,
                                  pixels[:, 1] * 0.001,
                                  np.full(len(ids), 0.5))).astype(np.float64)
        rng = np.random.default_rng(20261003)
        rgb = rng.random((height, width, 3), dtype=np.float32)
        mask = np.ones((height, width), dtype=bool)
        namespace = {"context_id": "synthetic-empty-fit-pool",
                     "source_rgb_sha256": audit.array_sha256(rgb)}
        arrays = _PoisonSourceArrays(
            template_rgb=rgb, template_mask=mask, observed_crop_mask=observed,
            query_rgb=object(), evaluator_truth=object())
        dependency_fn = _bank_dependency_fn(width, height)
        witness = selection.reserve_witnesses(
            ids, pixels, std, eligible, supported, observed, dependency_fn,
            image_namespace=namespace)
        self.assertEqual(witness["state"], "reserved")
        self.assertEqual(witness["denominator"], 16)
        np.testing.assert_array_equal(witness["selected_ids"], ids)

        with mock.patch.object(selection, "reserve_witnesses", wraps=selection.reserve_witnesses) as reserve_spy:
            result = selection.select_fit_and_bank(
                arrays, ids, pixels, points, std, eligible, supported, witness,
                dependency_fn, image_namespace=namespace)

        self.assertEqual(reserve_spy.call_count, 1)
        self.assertEqual(result["witness_denominator"], 16)
        np.testing.assert_array_equal(result["witness_selected_ids"], ids)
        self.assertEqual(result["fit_pool_count"], 0)
        self.assertEqual(result["fit_count"], 0)
        self.assertEqual(result["fit_state"], "unavailable")
        self.assertEqual(result["bank_builder_calls"], 1)
        self.assertEqual(result["bank_anchor_count"], 0)
        self.assertEqual(result["state"], "unavailable")
        self.assertTrue(any("fewer_than_24" in reason for reason in result["reasons"]))
        self.assertTrue(any("no_s_h_disjoint_structural_bank_anchor" == reason
                            for reason in result["reasons"]))
        self.assertEqual(arrays.forbidden_reads, [])

    def test_short_witness_keeps_diagnostic_fit_pool_and_real_bank_counts(self):
        width = height = 512
        observed = np.ones((height, width), dtype=bool)
        witness_ids = [30 * width + 30, 30 * width + 80]
        fit_ids = [90 * width + x for x in range(20, 110, 3)]
        bank_id = 80 * width + 80  # one real 16x16-grid anchor
        ids = np.asarray(witness_ids + fit_ids + [bank_id], dtype=np.int64)
        pixels = np.column_stack((ids % width + 0.5, ids // width + 0.5)).astype(np.float64)
        std = np.asarray([10.0, 9.0] + [0.5] * len(fit_ids) + [0.1], dtype=np.float64)
        eligible = np.ones(len(ids), dtype=bool)
        supported = np.ones(len(ids), dtype=bool)
        points = np.column_stack((ids % width * 0.001, ids // width * 0.001,
                                  0.2 + (ids % 37) * 0.001)).astype(np.float64)
        rng = np.random.default_rng(992)
        rgb = rng.random((height, width, 3), dtype=np.float32)
        template_mask = np.ones((height, width), dtype=bool)
        namespace = {"context_id": "synthetic-short-witness", "source_rgb_sha256": audit.array_sha256(rgb)}
        arrays = _PoisonSourceArrays(
            template_rgb=rgb,
            template_mask=template_mask,
            observed_crop_mask=observed,
            query_rgb=object(),
            evaluator_truth=object(),
        )
        dependency_fn = _bank_dependency_fn(width, height)
        witness = selection.reserve_witnesses(
            ids, pixels, std, eligible, supported, observed, dependency_fn,
            image_namespace=namespace)
        np.testing.assert_array_equal(witness["selected_ids"], witness_ids)
        self.assertEqual(witness["denominator"], 2)
        calls = []

        def real_audit_builder(rgb_arg, mask_arg, filtered_ids, filtered_points,
                               grid_size=16, max_anchors=64):
            calls.append((grid_size, max_anchors))
            return audit.build_template_patch_bank(
                rgb_arg, mask_arg, filtered_ids, filtered_points,
                grid_size=grid_size, max_anchors=max_anchors)

        result = selection.select_fit_and_bank(
            arrays, ids, pixels, points, std, eligible, supported, witness,
            dependency_fn, bank_builder=real_audit_builder, image_namespace=namespace)

        self.assertEqual(result["state"], "unavailable")
        self.assertTrue(any("fewer_than_8" in reason for reason in result["reasons"]))
        self.assertEqual(result["witness_denominator"], 2)
        np.testing.assert_array_equal(result["witness_selected_ids"], witness_ids)
        self.assertEqual(result["fit_pool_count"], 31)
        self.assertEqual(result["fit_count"], 31)
        self.assertEqual(result["fit_state"], "frozen")
        self.assertEqual(calls, [(16, 64)])
        self.assertEqual(result["bank_builder_calls"], 1)
        self.assertEqual(result["bank_anchor_count"], 1)
        np.testing.assert_array_equal(result["bank_anchor_ids"], [bank_id])
        self.assertEqual(result["bank_intersection_with_witness_count"], 0)
        self.assertEqual(result["fit_union_intersection_with_witness_count"], 0)

        detached_calls = []

        def detached_point_bank_builder(rgb_arg, mask_arg, filtered_ids, filtered_points,
                                        grid_size=16, max_anchors=64):
            detached_calls.append(True)
            bank = audit.build_template_patch_bank(
                rgb_arg, mask_arg, filtered_ids, filtered_points,
                grid_size=grid_size, max_anchors=max_anchors)
            self.assertEqual(len(bank["anchors"]), 1)
            # Corrupt both the private and public copies consistently. The
            # consumer must still bind the anchor's 3D point to its original
            # flat source ID, rather than trusting matching serialized rows.
            for anchor in (bank["anchors"][0], bank["serialized_anchors"][0]):
                detached = list(anchor["object_xyz_m"])
                detached[2] += 0.001
                anchor["object_xyz_m"] = detached
            return bank

        with self.assertRaises(selection.SourceReservationError):
            selection.select_fit_and_bank(
                arrays, ids, pixels, points, std, eligible, supported, witness,
                dependency_fn, bank_builder=detached_point_bank_builder,
                image_namespace=namespace)
        self.assertEqual(detached_calls, [True])
        self.assertEqual(arrays.forbidden_reads, [])

    def test_tampered_frozen_witness_is_rejected_before_pool_or_bank(self):
        fixture = _reservation_fixture()
        rng = np.random.default_rng(1007)
        rgb = rng.random((fixture["height"], fixture["width"], 3), dtype=np.float32)
        mask = np.ones((fixture["height"], fixture["width"]), dtype=bool)
        namespace = {"context_id": "synthetic-witness-integrity",
                     "source_rgb_sha256": audit.array_sha256(rgb)}
        arrays = _PoisonSourceArrays(template_rgb=rgb, template_mask=mask,
                                     observed_crop_mask=mask, query_rgb=object(),
                                     evaluator_truth=object())
        points = np.column_stack((fixture["ids"] % fixture["width"] * 0.001,
                                  fixture["ids"] // fixture["width"] * 0.001,
                                  np.full(len(fixture["ids"]), 0.45))).astype(np.float64)
        dependency_fn = _bank_dependency_fn(fixture["width"], fixture["height"])
        witness = selection.reserve_witnesses(
            fixture["ids"], fixture["pixels"], fixture["std"], fixture["eligible"],
            fixture["supported"], mask, dependency_fn, image_namespace=namespace)
        original_ids = witness["selected_ids"].copy()
        original_denominator = witness["denominator"]
        bank_calls = []

        def forbidden_bank_call(*_args, **_kwargs):
            bank_calls.append(True)
            raise AssertionError("a tampered witness reached bank construction")

        tampered_ids = dict(witness)
        tampered_ids["selected_ids"] = witness["selected_ids"].copy()
        tampered_ids["selected_ids"][0] += 1
        tampered_mask = dict(witness)
        tampered_mask["dependency_union_mask"] = witness["dependency_union_mask"].copy()
        tampered_mask["dependency_union_mask"][0, 0] = True
        tampered_namespace = dict(witness)
        tampered_namespace["selected_source_identity"] = dict(
            witness["selected_source_identity"],
            image_namespace=dict(namespace, context_id="different-template-source"))

        for altered in (tampered_ids, tampered_mask, tampered_namespace):
            with self.subTest(altered_field=("selected_ids" if altered is tampered_ids
                                             else "dependency_union_mask" if altered is tampered_mask
                                             else "selected_source_identity.image_namespace")):
                with self.assertRaises(selection.SourceReservationError):
                    selection.select_fit_and_bank(
                        arrays, fixture["ids"], fixture["pixels"], points, fixture["std"],
                        fixture["eligible"], fixture["supported"], altered, dependency_fn,
                        bank_builder=forbidden_bank_call, image_namespace=namespace)
        np.testing.assert_array_equal(witness["selected_ids"], original_ids)
        self.assertEqual(witness["denominator"], original_denominator)
        self.assertEqual(bank_calls, [])
        self.assertEqual(arrays.forbidden_reads, [])


if __name__ == "__main__":
    unittest.main()
