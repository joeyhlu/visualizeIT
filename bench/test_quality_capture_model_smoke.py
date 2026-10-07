"""Synthetic public-API checks for the model-only capture smoke core.

These tests use the shared numerical validator and refiner with a narrowly
faked GoTrack/renderer boundary. Their results establish contract and geometry
behavior only; they are not model-quality or runtime evidence.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from . import quality_gotrack as gotrack
from .quality_contract import (
    Frame,
    ModelSmokeFrame,
    PoseCandidate,
    ValidationSettings,
    check_capture_frame,
    check_model_smoke_frame,
    make_model_smoke_frame,
    validate,
    validate_model_smoke,
)
from .test_quality_capture_refiner import (
    _ChronologicalSpy,
    _Harness,
    _NATIVE_K,
    _capture_frame,
)


def _model_frame(*, case_id="b", ordinal=11, sampling_valid=None):
    valid = (np.ones((720, 720), dtype=np.bool_) if sampling_valid is None
             else sampling_valid)
    return make_model_smoke_frame(
        case_id=case_id,
        ordinal=ordinal,
        rgb=np.zeros((720, 720, 3), dtype=np.uint8),
        intrinsics=_NATIVE_K.copy(),
        sampling_valid=valid,
    )


def _four_neighbor_support(valid, uv):
    """Independent integer-center support calculation for synthetic points."""
    validity = np.asarray(valid, dtype=np.bool_)
    points = np.asarray(uv, dtype=np.float64).reshape((-1, 2))
    result = np.zeros(len(points), dtype=np.bool_)
    finite_ids = np.flatnonzero(np.isfinite(points).all(axis=1))
    if not finite_ids.size:
        return result
    x = np.floor(points[finite_ids, 0]).astype(np.int64)
    y = np.floor(points[finite_ids, 1]).astype(np.int64)
    inside = ((x >= 0) & (y >= 0) & (x + 1 < validity.shape[1]) &
              (y + 1 < validity.shape[0]))
    ids = finite_ids[inside]
    x, y = x[inside], y[inside]
    result[ids] = (validity[y, x] & validity[y, x + 1] &
                   validity[y + 1, x] & validity[y + 1, x + 1])
    return result


def _analytical_validation_fixture():
    """Build exact projections plus isolated support, boundary and weight cases."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (0.0, 0.0, 1.2)
    grid = np.asarray(
        [[70.25 + 80.0 * x, 70.25 + 80.0 * y]
         for y in range(8) for x in range(8)],
        dtype=np.float64,
    )
    special = np.asarray([
        [250.25, 250.25],   # one invalid sampling-valid neighbor
        [480.25, 450.25],   # one foreground hole among four neighbors
        [719.25, 600.25],   # finite, but outside the four-neighbor raster
        [350.25, 350.25],   # below the public visibility threshold
        [np.nan, 420.25],   # nonfinite and excluded before support counting
    ], dtype=np.float64)
    uv = np.vstack((grid, special))
    z = pose[2, 3]
    xyz = np.column_stack((
        (uv[:, 0] - _NATIVE_K[0, 2]) * z / _NATIVE_K[0, 0],
        (uv[:, 1] - _NATIVE_K[1, 2]) * z / _NATIVE_K[1, 1],
        np.zeros(len(uv), dtype=np.float64),
    ))
    weights = np.ones(len(uv), dtype=np.float64)
    weights[0] = 0.3  # public validation is inclusive at the default cutoff
    weights[len(grid) + 3] = 0.299
    candidate = PoseCandidate(pose, xyz, uv.copy(), weights)
    valid = np.ones((720, 720), dtype=np.bool_)
    valid[251, 251] = False
    mask = np.ones((720, 720), dtype=np.bool_)
    mask[451, 481] = False
    return candidate, uv, weights, valid, mask


def _install_generation_sequence(harness):
    """Give each fake template render a separately measured generation label."""
    original = harness.fake_utils.data_util.compute_gotrack_inputs_from_init_poses
    generations = []

    def compute(**kwargs):
        result = original(**kwargs)
        generation = len(generations) + 1
        generations.append(generation)
        harness.renderer.render_policy_metadata = {
            "dimensions": (280, 280),
            "allocation_generation": generation,
        }
        return result

    harness.fake_utils.data_util.compute_gotrack_inputs_from_init_poses = compute
    return generations


class ModelSmokeFrameContractTests(unittest.TestCase):
    def test_factory_detaches_exact_immutable_frame_and_is_keyword_only(self):
        rgb = np.full((720, 720, 3), 7, dtype=np.uint8)
        valid = np.ones((720, 720), dtype=np.bool_)
        intrinsics = _NATIVE_K.copy()
        frame = make_model_smoke_frame(
            case_id="a", ordinal=0, rgb=rgb, intrinsics=intrinsics,
            sampling_valid=valid)

        self.assertIs(type(frame), ModelSmokeFrame)
        self.assertEqual(frame.case_id, "a")
        self.assertEqual(frame.ordinal, 0)
        self.assertFalse(hasattr(frame, "frame_id"))
        self.assertFalse(hasattr(frame, "timestamp_s"))
        self.assertFalse(hasattr(frame, "capture_binding"))
        self.assertFalse(hasattr(frame, "clock_mode"))
        for snapshot in (frame.rgb, frame.intrinsics, frame.sampling_valid):
            self.assertFalse(snapshot.flags.writeable)
            base = snapshot
            while isinstance(base, np.ndarray):
                base = base.base
            self.assertIs(type(base), bytes)

        rgb.fill(0)
        valid.fill(False)
        intrinsics[0, 0] = 1.0
        self.assertTrue(np.all(frame.rgb == 7))
        self.assertTrue(np.all(frame.sampling_valid))
        np.testing.assert_array_equal(frame.intrinsics, _NATIVE_K)
        check_model_smoke_frame(frame)
        with self.assertRaises((AttributeError, TypeError)):
            frame.ordinal = 1
        with self.assertRaises(TypeError):
            make_model_smoke_frame("a", 0, rgb, _NATIVE_K, valid)

    def test_factory_rejects_wrong_case_ordinal_raster_dtype_and_camera(self):
        rgb = np.zeros((720, 720, 3), dtype=np.uint8)
        valid = np.ones((720, 720), dtype=np.bool_)
        intrinsics = _NATIVE_K.copy()
        invalid_inputs = [
            {"case_id": "d"},
            {"case_id": 1},
            {"ordinal": True},
            {"ordinal": -1},
            {"rgb": np.zeros((719, 720, 3), dtype=np.uint8)},
            {"rgb": np.zeros((720, 720, 3), dtype=np.uint16)},
            {"sampling_valid": np.ones((719, 720), dtype=np.bool_)},
            {"sampling_valid": np.ones((720, 720), dtype=np.uint8)},
            {"intrinsics": np.eye(3, dtype=np.float64)},
            {"intrinsics": np.ones((3, 3), dtype=np.bool_)},
        ]
        defaults = dict(case_id="a", ordinal=3, rgb=rgb,
                        intrinsics=intrinsics, sampling_valid=valid)
        for override in invalid_inputs:
            with self.subTest(override=tuple(override)):
                args = dict(defaults)
                args.update(override)
                with self.assertRaises((TypeError, ValueError)):
                    make_model_smoke_frame(**args)

    def test_checker_rechecks_corrupted_snapshots_and_exact_type(self):
        frame = _model_frame()
        with self.assertRaises(ValueError):
            check_model_smoke_frame(Frame(1, np.zeros((1, 1, 3), dtype=np.uint8),
                                          np.eye(3)))

        class DerivedModelSmokeFrame(ModelSmokeFrame):
            __slots__ = ()

        derived = DerivedModelSmokeFrame(
            case_id=frame.case_id, ordinal=frame.ordinal, rgb=frame.rgb,
            intrinsics=frame.intrinsics, sampling_valid=frame.sampling_valid)
        with self.assertRaises(ValueError):
            check_model_smoke_frame(derived)

        class RGBLikeDerivedFrame(ModelSmokeFrame):
            __slots__ = ()

            @property
            def ndim(self):
                return self.rgb.ndim

            @property
            def shape(self):
                return self.rgb.shape

            @property
            def dtype(self):
                return self.rgb.dtype

        rgb_like = RGBLikeDerivedFrame(
            case_id=frame.case_id, ordinal=frame.ordinal, rgb=frame.rgb,
            intrinsics=frame.intrinsics, sampling_valid=frame.sampling_valid)
        with self.assertRaises(ValueError):
            check_model_smoke_frame(rgb_like)

        class CaptureAttributeTrap(ModelSmokeFrame):
            __slots__ = ()

            def __getattribute__(self, name):
                if name in ("sampling_valid", "capture_binding"):
                    raise AssertionError(f"unexpected pre-gate read of {name}")
                return super().__getattribute__(name)

        trap = object.__new__(CaptureAttributeTrap)
        for name in ("case_id", "ordinal", "rgb", "intrinsics", "sampling_valid"):
            object.__setattr__(trap, name, getattr(frame, name))

        # Frame protects all five fields that could carry a model-only object.
        # The RGB-like subclass also demonstrates why the check must precede
        # ordinary RGB shape/dtype access.
        ordinary_frame_args = {
            "frame_id": 0,
            "rgb": np.zeros((720, 720, 3), dtype=np.uint8),
            "intrinsics": _NATIVE_K.copy(),
            "sampling_valid": None,
            "capture_binding": None,
        }
        model_values = (frame, derived, rgb_like, trap)
        for model_value in model_values:
            for field in ("frame_id", "rgb", "intrinsics", "sampling_valid",
                          "capture_binding"):
                with self.subTest(model_value=type(model_value).__name__, field=field):
                    values = dict(ordinary_frame_args)
                    values[field] = model_value
                    with self.assertRaisesRegex(ValueError, "ModelSmokeFrame"):
                        Frame(**values)

        corrupted = _model_frame()
        object.__setattr__(corrupted, "rgb", corrupted.rgb.copy())
        with self.assertRaisesRegex(ValueError, "immutable byte-backed"):
            check_model_smoke_frame(corrupted)

        corrupted = _model_frame()
        bad_k = corrupted.intrinsics.copy()
        bad_k[0, 0] += 1.0
        object.__setattr__(corrupted, "intrinsics", bad_k)
        with self.assertRaises(ValueError):
            check_model_smoke_frame(corrupted)

        corrupted = _model_frame()
        object.__setattr__(corrupted, "ordinal", True)
        with self.assertRaises(ValueError):
            check_model_smoke_frame(corrupted)

    def test_model_frame_cannot_enter_capture_frame_validation_or_refinement(self):
        frame = _model_frame()
        class DerivedModelSmokeFrame(ModelSmokeFrame):
            __slots__ = ()

        derived = DerivedModelSmokeFrame(
            case_id=frame.case_id, ordinal=frame.ordinal, rgb=frame.rgb,
            intrinsics=frame.intrinsics, sampling_valid=frame.sampling_valid)

        class RGBLikeDerivedFrame(DerivedModelSmokeFrame):
            __slots__ = ()

            @property
            def ndim(self):
                return self.rgb.ndim

            @property
            def shape(self):
                return self.rgb.shape

            @property
            def dtype(self):
                return self.rgb.dtype

        rgb_like = RGBLikeDerivedFrame(
            case_id=frame.case_id, ordinal=frame.ordinal, rgb=frame.rgb,
            intrinsics=frame.intrinsics, sampling_valid=frame.sampling_valid)

        class CaptureAttributeTrap(ModelSmokeFrame):
            __slots__ = ()

            def __getattribute__(self, name):
                if name in ("sampling_valid", "capture_binding"):
                    raise AssertionError(f"unexpected pre-gate read of {name}")
                return super().__getattribute__(name)

        trap = object.__new__(CaptureAttributeTrap)
        for name in ("case_id", "ordinal", "rgb", "intrinsics", "sampling_valid"):
            object.__setattr__(trap, name, getattr(frame, name))

        model_values = (frame, derived, rgb_like, trap)
        mask = np.ones((720, 720), dtype=np.bool_)
        for model_value in model_values:
            with self.subTest(model_value=type(model_value).__name__):
                with self.assertRaisesRegex(ValueError, "model-only capture smoke"):
                    check_capture_frame(model_value)
                # The frame gate must run before candidate math in ordinary
                # validation and before any mask/coercion work on a trap.
                with self.assertRaisesRegex(ValueError, "model-only capture smoke"):
                    validate(None, model_value, mask)

        self.assertIsNone(check_model_smoke_frame(frame))
        for model_value in (derived, rgb_like, trap):
            with self.subTest(model_only_subclass=type(model_value).__name__):
                with self.assertRaisesRegex(ValueError, "exact ModelSmokeFrame"):
                    check_model_smoke_frame(model_value)
                with self.assertRaisesRegex(ValueError, "exact ModelSmokeFrame"):
                    validate_model_smoke(None, model_value, None)

        harness = _Harness()
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            for model_value in model_values:
                with self.subTest(ordinary_refine=type(model_value).__name__):
                    with self.assertRaisesRegex(ValueError, "model-only capture smoke"):
                        refiner.refine(model_value, mask, np.eye(4))
            for model_value in (derived, rgb_like, trap):
                with self.subTest(model_smoke_refine=type(model_value).__name__):
                    with self.assertRaisesRegex(ValueError, "exact ModelSmokeFrame"):
                        refiner.refine_model_smoke(model_value, None, np.eye(4))
        self.assertEqual(harness.torch_imports, 0)
        self.assertEqual(harness.utils_imports, 0)
        self.assertEqual(harness.network_calls, 0)
        self.assertEqual(harness.template_requests, [])
        self.assertEqual(harness.warp_calls, 0)


class ModelSmokeValidationTests(unittest.TestCase):
    def test_support_aware_validation_matches_independent_projection_and_capture(self):
        candidate, uv, weights, valid, mask = _analytical_validation_fixture()
        model_frame = _model_frame(sampling_valid=valid)
        capture_frame = _capture_frame(valid.copy())
        settings = ValidationSettings()

        self.assertEqual(settings.visibility_weight, 0.3)
        self.assertEqual(settings.min_correspondences, 24)
        self.assertEqual(settings.min_inliers, 20)
        self.assertEqual(settings.min_inlier_ratio, 0.6)
        self.assertEqual(settings.max_median_error_720, 3.0)
        self.assertEqual(settings.max_p95_error_720, 8.0)
        self.assertEqual(settings.min_spatial_support, 0.12)

        model_result = validate_model_smoke(candidate, model_frame, mask)
        capture_result = validate(candidate, capture_frame, mask)
        self.assertEqual(model_result, capture_result)
        passed, reason, stats = model_result
        self.assertTrue(passed)
        self.assertIsNone(reason)
        self.assertEqual(set(stats), {
            "correspondences", "inliers", "median_reprojection_720",
            "p95_reprojection_720", "spatial_support", "score",
            "raw_correspondences", "retained_correspondences",
            "unsupported_correspondences",
        })

        foreground = mask & valid
        finite = (np.isfinite(candidate.points_object_m).all(axis=1) &
                  np.isfinite(uv).all(axis=1) & np.isfinite(weights))
        confidence_eligible = finite & (weights >= settings.visibility_weight)
        four_neighbor_ok = (_four_neighbor_support(valid, uv) &
                            _four_neighbor_support(foreground, uv))
        expected_retained = int(np.count_nonzero(confidence_eligible & four_neighbor_ok))
        expected_unsupported = int(np.count_nonzero(
            confidence_eligible & ~four_neighbor_ok))
        self.assertEqual(expected_retained, 64)
        self.assertEqual(expected_unsupported, 3)
        self.assertEqual(stats["raw_correspondences"], len(uv))
        self.assertEqual(stats["retained_correspondences"], expected_retained)
        self.assertEqual(stats["correspondences"], expected_retained)
        self.assertEqual(stats["unsupported_correspondences"], expected_unsupported)
        self.assertEqual(stats["inliers"], expected_retained)
        self.assertLessEqual(stats["median_reprojection_720"], 1e-10)
        self.assertGreaterEqual(stats["spatial_support"], settings.min_spatial_support)

    def test_empty_or_missing_model_foreground_cannot_validate(self):
        candidate, _uv, weights, valid, _mask = _analytical_validation_fixture()
        frame = _model_frame(sampling_valid=valid)
        with self.assertRaisesRegex(ValueError, "same-render"):
            validate_model_smoke(candidate, frame, None)

        passed, reason, stats = validate_model_smoke(
            candidate, frame, np.zeros((720, 720), dtype=np.bool_))
        eligible = (np.isfinite(candidate.points_object_m).all(axis=1) &
                    np.isfinite(candidate.pixels_image).all(axis=1) &
                    np.isfinite(weights) & (weights >= 0.3))
        self.assertFalse(passed)
        self.assertEqual(reason, "insufficient_visible_correspondences")
        self.assertEqual(stats["correspondences"], 0)
        self.assertEqual(stats["retained_correspondences"], 0)
        self.assertEqual(stats["unsupported_correspondences"],
                         int(np.count_nonzero(eligible)))

        with self.assertRaises(ValueError):
            validate_model_smoke(candidate, frame,
                                  np.ones((719, 720), dtype=np.bool_))
        with self.assertRaises(ValueError):
            validate_model_smoke(candidate, Frame(1, np.zeros((1, 1, 3), dtype=np.uint8),
                                                   np.eye(3)),
                                  np.ones((720, 720), dtype=np.bool_))

    def test_default_correspondence_floor_is_still_24(self):
        candidate, _uv, _weights, valid, mask = _analytical_validation_fixture()
        frame = _model_frame(sampling_valid=valid)
        short = PoseCandidate(candidate.pose.copy(), candidate.points_object_m[:23].copy(),
                              candidate.pixels_image[:23].copy(),
                              candidate.weights[:23].copy())
        passed, reason, stats = validate_model_smoke(short, frame, mask)
        self.assertFalse(passed)
        self.assertEqual(reason, "insufficient_visible_correspondences")
        self.assertEqual(stats["correspondences"], 23)
        self.assertEqual(stats["retained_correspondences"], 23)


class ModelSmokeRefinerTests(unittest.TestCase):
    def test_each_fixed_model_case_completes_five_public_refinement_iterations(self):
        for ordinal, case_id in enumerate(("a", "b", "c")):
            with self.subTest(case_id=case_id):
                harness = _Harness()
                _install_generation_sequence(harness)
                refiner = harness.make_refiner()
                with harness.guarded_imports(), patch.object(
                        gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                        gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
                    result = refiner.refine_model_smoke(
                        _model_frame(case_id=case_id, ordinal=ordinal),
                        np.ones((720, 720), dtype=np.bool_), np.eye(4))

                self.assertIsNotNone(result.candidate)
                self.assertEqual(result.case_id, case_id)
                self.assertEqual(len(result.iterations), 5)
                self.assertEqual(harness.network_calls, 5)
                self.assertEqual(harness.pnp_calls, 5)
                self.assertTrue(all(row.solver_success for row in result.iterations))

    def test_public_refiner_runs_five_measured_calls_and_returns_native_metre_result(self):
        harness = _Harness()
        generations = _install_generation_sequence(harness)
        original_warp = harness.fake_utils.im_util.warp_image
        warp_calls = []

        def warp(**kwargs):
            warp_calls.append(dict(kwargs))
            return original_warp(**kwargs)

        harness.fake_utils.im_util.warp_image = warp
        frame = _model_frame(case_id="c", ordinal=23)
        mask = np.ones((720, 720), dtype=np.bool_)
        seed = np.eye(4, dtype=np.float64)
        seed[:3, 3] = (0.12, -0.04, 1.2)
        refiner = harness.make_refiner()
        rng_seeds = []
        trace_names = []

        with harness.guarded_imports(), patch.object(
                gotrack, "trace",
                lambda name, **_kwargs: trace_names.append(name)), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm), patch.object(
                gotrack.cv2, "setRNGSeed", rng_seeds.append):
            result = refiner.refine_model_smoke(frame, mask, seed)

        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.case_id, "c")
        self.assertEqual(result.ordinal, 23)
        self.assertEqual(dict(result.metadata), {
            "model_only": True, "case_id": "c", "ordinal": 23,
            "timestamp_s": None,
        })
        with self.assertRaises(TypeError):
            result.metadata["frame_id"] = 23
        self.assertEqual(len(result.iterations), 5)
        self.assertEqual(harness.network_calls, 5)
        self.assertEqual(harness.pnp_calls, 5)
        self.assertEqual(harness.lm_calls, 5)
        self.assertEqual(len(warp_calls), 5)
        self.assertEqual(generations, [1, 2, 3, 4, 5])
        self.assertEqual(rng_seeds, [(23 * 17 + index) & 0x7fffffff
                                     for index in range(5)])
        self.assertEqual(trace_names, [
            f"refine/model-smoke/c/23/{index}" for index in range(5)])

        seed_mm = seed.copy()
        seed_mm[:3, 3] *= 1000.0
        np.testing.assert_allclose(harness.initial_poses[0], seed_mm,
                                   rtol=0.0, atol=2e-4)
        for index in range(1, 5):
            np.testing.assert_allclose(
                harness.initial_poses[index],
                harness.desired_native_pose_mm(index - 1),
                rtol=0.0, atol=2e-4)
        expected_pose_m = harness.desired_native_pose_mm(4)
        expected_pose_m[:3, 3] *= 0.001
        np.testing.assert_allclose(result.candidate.pose, expected_pose_m,
                                   rtol=0.0, atol=2e-6)
        final_ids = np.linspace(
            0, len(result.candidate.points_object_m) - 1,
            min(len(result.candidate.points_object_m), 10_000), dtype=int)
        np.testing.assert_allclose(
            result.candidate.points_object_m[final_ids] * 1000.0,
            harness.pnp_records[-1]["obj_points"], rtol=0.0, atol=2e-4)

        for index, (row, warp_call) in enumerate(zip(result.iterations, warp_calls)):
            record = row.as_dict()
            self.assertEqual(set(record), {
                "index", "solver_success", "crop_dimensions",
                "query_rewarp_factor", "max_sampling_map_difference_px",
                "sampling_map_eligible_count", "render_generation",
            })
            self.assertEqual(record["index"], index)
            self.assertTrue(record["solver_success"])
            self.assertEqual(record["crop_dimensions"], [280, 280])
            self.assertEqual(record["query_rewarp_factor"], 1)
            self.assertEqual(warp_call["factor_to_downsample"], 1)
            self.assertTrue(warp_call["depth_check"])
            self.assertEqual(warp_call["src_camera"].width, 720)
            self.assertEqual(warp_call["dst_camera"].width, 280)
            self.assertIsNotNone(record["max_sampling_map_difference_px"])
            self.assertTrue(np.isfinite(record["max_sampling_map_difference_px"]))
            self.assertLess(record["max_sampling_map_difference_px"], 1.0)
            self.assertGreater(record["sampling_map_eligible_count"], 0)
            self.assertEqual(record["render_generation"], index + 1)

    def test_renderer_modes_reject_before_optional_import_or_render_effects(self):
        for field, wrong in (("coordinate_mode", "legacy"),
                             ("render_policy", "legacy_v1")):
            harness = _Harness()
            setattr(harness.renderer, field, wrong)
            refiner = harness.make_refiner()
            with harness.guarded_imports(), patch.object(
                    gotrack, "trace", lambda *_args, **_kwargs: None):
                with self.assertRaises(ValueError):
                    refiner.refine_model_smoke(
                        _model_frame(), np.ones((720, 720), dtype=np.bool_),
                        np.eye(4))
            self.assertEqual(harness.torch_imports, 0)
            self.assertEqual(harness.utils_imports, 0)
            self.assertEqual(harness.template_requests, [])
            self.assertEqual(harness.network_calls, 0)
            self.assertEqual(harness.warp_calls, 0)

    def test_invalid_seed_missing_generation_and_map_fault_never_return_success(self):
        harness = _Harness()
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            with self.assertRaises(ValueError):
                refiner.refine_model_smoke(
                    _model_frame(), np.ones((720, 720), dtype=np.bool_),
                    np.full((4, 4), np.nan))
        self.assertEqual(harness.network_calls, 0)
        self.assertEqual(harness.template_requests, [])

        harness = _Harness()
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            with self.assertRaisesRegex(ValueError, "renderer metadata"):
                refiner.refine_model_smoke(
                    _model_frame(), np.ones((720, 720), dtype=np.bool_), np.eye(4))
        self.assertEqual(harness.network_calls, 0)
        self.assertEqual(len(harness.template_requests), 1)

        harness = _Harness(map_fault="nonfinite_map")
        _install_generation_sequence(harness)
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            with self.assertRaisesRegex(ValueError, "sampling map is nonfinite"):
                refiner.refine_model_smoke(
                    _model_frame(), np.ones((720, 720), dtype=np.bool_), np.eye(4))
        self.assertEqual(harness.network_calls, 0)

    def test_partial_solver_failure_returns_only_measured_rows_and_no_candidate(self):
        harness = _Harness()
        _install_generation_sequence(harness)
        refiner = harness.make_refiner()
        original_solver = refiner._solve_pnp_ransac
        solver_calls = []

        def fail_third(obj_points, target_points, crop_k, rvec, tvec):
            solver_calls.append(len(solver_calls))
            if len(solver_calls) == 3:
                return False, rvec, tvec, None
            return original_solver(obj_points, target_points, crop_k, rvec, tvec)

        refiner._solve_pnp_ransac = fail_third
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            result = refiner.refine_model_smoke(
                _model_frame(case_id="a", ordinal=9),
                np.ones((720, 720), dtype=np.bool_), np.eye(4))

        self.assertIsNone(result.candidate)
        self.assertEqual(len(solver_calls), 3)
        self.assertEqual(harness.network_calls, 3)
        self.assertEqual(len(result.iterations), 3)
        self.assertEqual([row.solver_success for row in result.iterations],
                         [True, True, False])
        self.assertEqual([row.render_generation for row in result.iterations],
                         [1, 2, 3])

    def test_empty_support_has_no_candidate_or_claimed_sampling_map(self):
        harness = _Harness()
        _install_generation_sequence(harness)
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            result = refiner.refine_model_smoke(
                _model_frame(), np.zeros((720, 720), dtype=np.bool_), np.eye(4))

        self.assertIsNone(result.candidate)
        self.assertEqual(len(result.iterations), 1)
        row = result.iterations[0]
        self.assertFalse(row.solver_success)
        self.assertEqual(row.sampling_map_eligible_count, 0)
        self.assertIsNone(row.max_sampling_map_difference_px)
        self.assertEqual(harness.pnp_calls, 0)


class CaptureAndLegacyRegressionTests(unittest.TestCase):
    def test_genuine_capture_keeps_rng_identity_and_observer_metadata(self):
        harness = _Harness()
        _install_generation_sequence(harness)
        frame = _capture_frame()
        observer = _ChronologicalSpy()
        refiner = harness.make_refiner(observer)
        refiner.set_chronological_capture_context({"test": "capture-regression"})
        rng_seeds = []

        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm), patch.object(
                gotrack.cv2, "setRNGSeed", rng_seeds.append):
            candidate = refiner.refine(
                frame, np.ones((720, 720), dtype=np.bool_), np.eye(4))

        self.assertIsNotNone(candidate)
        self.assertEqual(rng_seeds, [(frame.frame_id * 17 + index) & 0x7fffffff
                                     for index in range(5)])
        self.assertEqual(len(observer.retained_preflights), 5)
        for index, (metadata, _descriptors) in enumerate(observer.retained_preflights):
            self.assertEqual(metadata["frame_id"], frame.frame_id)
            self.assertEqual(metadata["timestamp_s"], frame.timestamp_s)
            self.assertEqual(metadata["capture_binding"], dict(frame.capture_binding))
            self.assertEqual(metadata["rng_seed"],
                             (frame.frame_id * 17 + index) & 0x7fffffff)

    def test_legacy_refiner_still_uses_legacy_solver_without_capture_helpers(self):
        harness = _Harness()
        harness.renderer.coordinate_mode = "legacy"
        refiner = gotrack.GoTrackRefiner(
            network=harness.network, renderer=harness.renderer,
            obj_id="synthetic", device="cpu")
        legacy_k = np.array([[110.0, 0.0, 60.0],
                             [0.0, 105.0, 40.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
        legacy_frame = Frame(3, np.zeros((80, 120, 3), dtype=np.uint8), legacy_k)

        with harness.guarded_imports(forbid_capture_helper=True), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack, "_capture_erode3",
                side_effect=AssertionError("legacy path used capture erosion")), patch.object(
                gotrack, "_capture_actual_sampling_map",
                side_effect=AssertionError("legacy path used capture sampling maps")), patch.object(
                gotrack.cv2, "solvePnPRansac", harness.legacy_solve_pnp), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            candidate = refiner.refine(
                legacy_frame, np.ones((80, 120), dtype=np.bool_), np.eye(4))

        self.assertIsNotNone(candidate)
        self.assertEqual(harness.capture_helper_imports, 0)
        self.assertEqual(harness.warp_calls, 0)
        self.assertEqual(harness.network_calls, 5)
        self.assertEqual(len(harness.legacy_solver_calls), 5)
        self.assertEqual(harness.lm_calls, 5)
        self.assertEqual(len(harness.template_requests), 5)
        for request in harness.template_requests:
            self.assertEqual(request["crop_size"], (280, 280))
            self.assertEqual(request["cropping_type"], "perspective_2d_box")
            self.assertEqual(request["ssaa_factor"], 1.0)
        for args, kwargs in harness.legacy_solver_calls:
            self.assertEqual(args[0].shape[1], 3)
            self.assertEqual(args[1].shape[1], 2)
            self.assertEqual(kwargs["iterationsCount"], 3000)
            self.assertEqual(kwargs["reprojectionError"], 2.0)
            self.assertEqual(kwargs["confidence"], 0.999)
            self.assertEqual(kwargs["flags"], gotrack.cv2.SOLVEPNP_ITERATIVE)


if __name__ == "__main__":
    unittest.main()
