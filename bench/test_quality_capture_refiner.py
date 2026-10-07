"""Independent synthetic tests for capture refiner pose feedback and camera maps.

The upstream boundary is faked narrowly: only ``torch`` and ``utils`` imports
inside the tested call are intercepted. All geometry and solver decisions still
pass through the public ``GoTrackRefiner.refine`` implementation.
"""

from __future__ import annotations

import builtins
import hashlib
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_gotrack as gotrack
from .quality_contract import Frame
from .quality_time import LEGACY, PHYSICAL


_NATIVE_K = np.array([[360.0, 0.0, 359.5],
                      [0.0, 360.0, 359.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _capture_frame(valid: np.ndarray | None = None) -> Frame:
    rgb = np.zeros((720, 720, 3), dtype=np.uint8)
    valid = np.ones((720, 720), dtype=np.bool_) if valid is None else valid
    rgb_hash = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest().upper()
    framed_valid = hashlib.sha256(
        b"sampling_valid\0[720,720]\0|u1\0" +
        np.ascontiguousarray(valid, dtype=np.uint8).tobytes()
    ).hexdigest().upper()
    binding = {
        "capture_table_sha256": "A" * 64,
        "capture_row_sha256": "B" * 64,
        "rgb_pixel_sha256": rgb_hash,
        "sampling_valid_sha256": framed_valid,
        "warp_sha256": "C" * 64,
        "calibration_sha256": "D" * 64,
    }
    return Frame(frame_id=17, rgb=rgb, intrinsics=_NATIVE_K.copy(),
                 timestamp_s=1.25, clock_mode=PHYSICAL,
                 sampling_valid=valid, capture_binding=binding)


def _crop_to_native_uv(harness: "_Harness", pixels: np.ndarray) -> np.ndarray:
    """Independent projection used to place/check synthetic support holes."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape((-1, 2))
    crop_k = np.array([[harness.crop_f[0], 0.0, harness.crop_c[0]],
                       [0.0, harness.crop_f[1], harness.crop_c[1]],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    crop_rays = homogeneous @ np.linalg.inv(crop_k).T
    native_rays = crop_rays @ harness.crop_transform[:3, :3]
    native_h = native_rays @ _NATIVE_K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return native_h[:, :2] / native_h[:, 2:3]


def _paint_native_patch(array: np.ndarray, uv: np.ndarray, radius: int = 5) -> None:
    point = np.asarray(uv, dtype=np.float64)
    if not np.isfinite(point).all():
        raise AssertionError("synthetic patch center must be projectable")
    x, y = np.floor(point).astype(int)
    h, w = array.shape
    array[max(0, y - radius):min(h, y + radius + 1),
          max(0, x - radius):min(w, x + radius + 1)] = False


def _four_neighbor_support(valid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Test a record's four native integer-center neighbors without clipping."""
    valid = np.asarray(valid)
    uv = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    result = np.zeros(len(uv), dtype=np.bool_)
    finite = np.isfinite(uv).all(axis=1)
    ids = np.flatnonzero(finite)
    if ids.size:
        x = np.floor(uv[ids, 0]).astype(np.int64)
        y = np.floor(uv[ids, 1]).astype(np.int64)
        inside = (x >= 0) & (y >= 0) & (x + 1 < valid.shape[1]) & (y + 1 < valid.shape[0])
        selected = ids[inside]
        x, y = x[inside], y[inside]
        result[selected] = (valid[y, x] & valid[y, x + 1] &
                            valid[y + 1, x] & valid[y + 1, x + 1])
    return result


def _expected_retained_base_bytes(metadata, sample_count: int) -> int:
    """Fixture-derived bytes before the three deferred support arrays."""
    retained = int(metadata["retained_count"])
    pixels = 280 * 280
    fixed_float32 = (2 * 3 * pixels * np.dtype(np.float32).itemsize +
                     2 * pixels * np.dtype(np.float32).itemsize)
    fixed_bool = pixels * np.dtype(np.bool_).itemsize
    fixed_float64_maps = (2 * pixels * 8 + pixels * 8)
    retained_records = retained * (3 * 8 + 2 * 8)
    samples = sample_count * (np.dtype(int).itemsize + 3 * 8 + 2 * 8)
    camera_arrays = (9 * 8 + 16 * 8 + 9 * 8 + 16 * 8 + 16 * 8 + 3 * 8 + 3 * 8)
    return fixed_float32 + fixed_bool + fixed_float64_maps + retained_records + samples + camera_arrays


class _Tensor:
    """Small ndarray-backed Tensor surface used by the real capture method."""

    def __init__(self, value: object, device: str = "cpu"):
        self.array = value if isinstance(value, np.ndarray) else np.asarray(value)
        self.device = device

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __array__(self, dtype=None, copy=None):
        value = self.array if dtype is None else self.array.astype(dtype, copy=False)
        return value.copy() if copy else value

    def __getitem__(self, key):
        return _Tensor(self.array[key], self.device)

    def __truediv__(self, other):
        return _Tensor(self.array / other, self.device)

    def permute(self, *axes):
        return _Tensor(np.transpose(self.array, axes), self.device)

    def float(self):
        return _Tensor(self.array.astype(np.float32, copy=False), self.device)

    def to(self, device):
        return _Tensor(self.array, str(device))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def numel(self):
        return int(self.array.size)

    def element_size(self):
        return int(self.array.dtype.itemsize)


class _Camera:
    """Independent camera API implementing the pinned upstream method chain."""

    def __init__(self, harness, width, height, f, c, T_world_from_eye):
        self.harness = harness
        self.width, self.height = int(width), int(height)
        self.f, self.c = tuple(f), tuple(c)
        self.K = np.array([[self.f[0], 0.0, self.c[0]],
                           [0.0, self.f[1], self.c[1]],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.T_world_from_eye = np.array(T_world_from_eye, dtype=np.float64, copy=True)
        if self.width == 720:
            harness.native_camera = self

    def window_to_eye(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64)
        homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
        return homogeneous @ np.linalg.inv(self.K).T

    def eye_to_world(self, points):
        points = np.asarray(points, dtype=np.float64)
        rotation = self.T_world_from_eye[:3, :3]
        return points @ rotation.T + self.T_world_from_eye[:3, 3]

    def world_to_eye(self, points):
        points = np.asarray(points, dtype=np.float64)
        rotation = self.T_world_from_eye[:3, :3]
        eye = (points - self.T_world_from_eye[:3, 3]) @ rotation
        if self.width == 720 and self.harness.map_fault in ("zero_depth", "negative_depth"):
            eye = np.array(eye, copy=True)
            eye[self.harness.fault_index, 2] = (
                0.0 if self.harness.map_fault == "zero_depth" else -1.0)
        return eye

    def eye_to_window(self, points):
        points = np.asarray(points, dtype=np.float64)
        homogeneous = points @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        if self.width == 720 and self.harness.map_fault == "nonfinite_map":
            pixels = np.array(pixels, copy=True)
            pixels[self.harness.fault_index, 0] = np.nan
        return pixels


class _Harness:
    def __init__(self, *, map_fault: str | None = None,
                 large_rotation: bool = False, solve_success: bool = True):
        self.map_fault = map_fault
        self.large_rotation = large_rotation
        self.solve_success = solve_success
        self.fault_index = 140 * 280 + 140
        self.events: list[str] = []
        self.initial_poses: list[np.ndarray] = []
        self.network_calls = 0
        self.warp_calls = 0
        self.pnp_calls = 0
        self.lm_calls = 0
        self.torch_imports = 0
        self.utils_imports = 0
        self.capture_helper_imports = 0
        self.template_requests = []
        self.pnp_records = []
        self.lm_records = []
        self.native_camera = None
        self.template_mask = np.ones((280, 280), dtype=np.bool_)
        self.flow = np.zeros((1, 2, 280, 280), dtype=np.float32)
        self.confidence = np.ones((1, 280, 280), dtype=np.float32)
        self.legacy_solver_calls = []
        self.crop_transform = np.eye(4, dtype=np.float64)
        if large_rotation:
            self.crop_transform[:3, :3] = _rotation_y(0.60)
            self.crop_f = (80.0, 200.0)
            self.crop_c = (139.5, 141.25)
        else:
            self.crop_transform[:3, :3] = _rotation_y(0.075) @ _rotation_x(-0.04)
            self.crop_f = (178.0, 186.0)
            self.crop_c = (132.25, 146.75)
        self.renderer = SimpleNamespace(
            coordinate_mode="integer_centers_v1",
            render_policy="capture_zero_sample_v1",
            vertices_m=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                                 [0.0, 0.1, 0.0]], dtype=np.float64),
        )
        self.fake_torch = SimpleNamespace(
            from_numpy=lambda value: _Tensor(value),
            inference_mode=nullcontext,
            get_rng_state=lambda: b"fake-rng",
            set_rng_state=lambda _state: None,
            cuda=SimpleNamespace(is_initialized=lambda: False),
        )
        self.fake_utils = self._make_utils_module()

    def desired_native_pose_mm(self, iteration: int) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _rotation_z(0.006 * (iteration + 1))
        pose[:3, 3] = [43.0 + 4.0 * iteration,
                       -27.0 + 3.0 * iteration,
                       875.0 + 9.0 * iteration]
        return pose

    def _make_utils_module(self):
        harness = self
        utils = ModuleType("utils")

        class _CameraFactory:
            @staticmethod
            def PinholePlaneCameraModel(width, height, f, c, T_world_from_eye):
                return _Camera(harness, width, height, f, c, T_world_from_eye)

        def _compute(**kwargs):
            harness.events.append("template")
            harness.template_requests.append(kwargs)
            pose = kwargs["init_poses_cam_from_model"]
            harness.initial_poses.append(np.array(pose.array[0], copy=True))
            crop_camera_pose = np.linalg.inv(harness.crop_transform)
            if harness.map_fault == "offset":
                crop_camera_pose[0, 3] += 0.01
            crop_camera = _Camera(harness, 280, 280, harness.crop_f,
                                  harness.crop_c, crop_camera_pose)
            template = SimpleNamespace(
                rgbs=_Tensor(np.zeros((1, 3, 280, 280), dtype=np.float32)),
                depths=_Tensor(np.full((1, 280, 280), 1000.0, dtype=np.float32)),
                masks=_Tensor(harness.template_mask[None].copy()),
            )
            data = {"templates": template,
                    "crop_rgbs": _Tensor(np.zeros((1, 3, 280, 280), dtype=np.float32)),
                    "crop_masks": _Tensor(np.ones((1, 280, 280), dtype=np.float32))}
            return data, [crop_camera], [_Tensor(harness.crop_transform.copy())]

        def _warp_image(**kwargs):
            harness.events.append("warp")
            harness.warp_calls += 1
            return np.zeros((280, 280, 3), dtype=np.uint8)

        class _Network:
            def __call__(self, *_args):
                harness.events.append("network")
                harness.network_calls += 1
                return _Tensor(harness.flow.copy()), _Tensor(harness.confidence.copy())

        def _get_3d_points_from_depth(depth, intrinsic):
            depth_array = np.asarray(depth)
            depth_map = depth_array[0]
            camera_k = np.asarray(intrinsic)[0].astype(np.float64)
            height, width = depth_map.shape
            yy, xx = np.indices((height, width), dtype=np.float64)
            homogeneous = np.stack((xx + 0.5, yy + 0.5, np.ones_like(xx)), axis=-1)
            rays = homogeneous @ np.linalg.inv(camera_k).T
            xyz = rays * depth_map[..., None]
            return (_Tensor(np.stack((xx + 0.5, yy + 0.5), axis=-1)[None].astype(np.float32)),
                    _Tensor(xyz[None].astype(np.float32)))

        def _legacy_solve_pnp(*args, **kwargs):
            harness.legacy_solver_calls.append((args, dict(kwargs)))
            inliers = np.arange(min(6, len(args[0])), dtype=np.int32).reshape((-1, 1))
            return True, np.array(kwargs["rvec"], copy=True), np.array(kwargs["tvec"], copy=True), inliers

        self.network = _Network()
        utils.data_util = SimpleNamespace(
            compute_gotrack_inputs_from_init_poses=_compute)
        utils.im_util = SimpleNamespace(warp_image=_warp_image)
        utils.misc = SimpleNamespace(get_intrinsic_matrix=lambda camera: camera.K)
        utils.structs = _CameraFactory
        utils.transform3d = SimpleNamespace(get_3d_points_from_depth=_get_3d_points_from_depth)
        self.legacy_solve_pnp = _legacy_solve_pnp
        return utils

    def guarded_imports(self, *, forbid_capture_helper=False):
        harness = self
        original_import = builtins.__import__

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name == "torch":
                harness.torch_imports += 1
                return harness.fake_torch
            if level == 0 and name == "utils":
                harness.utils_imports += 1
                return harness.fake_utils
            if level == 1 and name == "quality_capture":
                harness.capture_helper_imports += 1
                if forbid_capture_helper:
                    raise AssertionError("legacy path imported the capture-only helper")
            return original_import(name, globals, locals, fromlist, level)

        return patch("builtins.__import__", guarded)

    def make_refiner(self, chronological_capture_callback=None):
        refiner = gotrack.GoTrackRefiner(
            network=self.network, renderer=self.renderer,
            obj_id="synthetic", device="cpu",
            chronological_capture_callback=chronological_capture_callback)
        refiner._solve_pnp_ransac = self.solve_pnp_ransac
        return refiner

    def solve_pnp_ransac(self, obj_points, target_points, crop_k, rvec, tvec):
        self.events.append("pnp")
        self.pnp_calls += 1
        self.pnp_records.append({
            "obj_points": np.array(obj_points, copy=True),
            "target_points": np.array(target_points, copy=True),
            "crop_k": np.array(crop_k, copy=True),
        })
        if not self.solve_success:
            return False, rvec, tvec, None
        inliers = np.arange(min(6, len(obj_points)), dtype=np.int32).reshape((-1, 1))
        return True, rvec, tvec, inliers

    def solve_lm(self, obj_points, target_points, _crop_k, _dist, _rvec, _tvec):
        self.events.append("lm")
        self.lm_records.append({
            "obj_points": np.array(obj_points, copy=True),
            "target_points": np.array(target_points, copy=True),
        })
        iteration = self.lm_calls
        self.lm_calls += 1
        native_pose = self.desired_native_pose_mm(iteration)
        crop_pose = self.crop_transform @ native_pose
        crop_rvec = gotrack.cv2.Rodrigues(crop_pose[:3, :3])[0]
        return crop_rvec, crop_pose[:3, 3].reshape((3, 1))


class _ChronologicalSpy:
    def __init__(self, *, accept_retained=True, allocation_tracker=None):
        self.retained_packets = []
        self.retained_estimates = []
        self.retained_byte_preflights = []
        self.retained_preflights = []
        self.accept_retained = accept_retained
        self.allocation_tracker = allocation_tracker
        self.rejected_budget = False

    def should_capture(self, _context, _kind, _metadata):
        return True

    def preflight(self, _context, kind, metadata, descriptors):
        if kind == "retained":
            self.retained_preflights.append((dict(metadata), dict(descriptors)))
        return True

    def preflight_nbytes(self, _context, kind, metadata, estimated_bytes):
        if kind == "retained":
            record = (dict(metadata), int(estimated_bytes))
            self.retained_byte_preflights.append(record)
            self.retained_estimates.append(int(estimated_bytes))
            if not self.accept_retained:
                self.rejected_budget = True
                if self.allocation_tracker is not None:
                    self.allocation_tracker.rejected_budget = True
                return False
        return True

    def __call__(self, packet):
        if packet["kind"] == "retained":
            self.retained_packets.append(packet)

    def _disable_capture(self, _reason):
        raise AssertionError("synthetic capture observer unexpectedly disabled")


class _AllocationTracker:
    def __init__(self):
        self.rejected_budget = False
        self.pre_rejection_index_events = []
        self.post_rejection_allocations = []


class _TrackedIndexArray(np.ndarray):
    """Observe advanced-index support arrays created after a failed preflight."""

    def __new__(cls, value, tracker, role="unclassified"):
        result = np.asarray(value).view(cls)
        result.tracker = tracker
        result.role = role
        return result

    def __array_finalize__(self, parent):
        self.tracker = getattr(parent, "tracker", None)
        self.role = getattr(parent, "role", "unclassified")

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        # Keep confidence instrumentation through production's float64 weight
        # promotion. Drop the integer-grid marker on float conversion so it
        # cannot follow ordinary projected target points into solver indexing.
        keep_marker = self.role == "confidence"
        return np.ndarray.astype(
            self, dtype, order=order, casting=casting,
            subok=keep_marker, copy=copy)

    def __getitem__(self, key):
        tracker = self.tracker
        if (tracker is not None and isinstance(key, np.ndarray) and
                key.dtype.kind in "biu" and self.role in ("confidence", "integer_grid")):
            record = (self.role, self.shape, key.shape, str(key.dtype))
            if tracker.rejected_budget:
                tracker.post_rejection_allocations.append(record)
            else:
                tracker.pre_rejection_index_events.append(record)
        return super().__getitem__(key)


def _tracked_stack(original_stack, tracker):
    def stack(values, axis=0, out=None, **kwargs):
        result = original_stack(values, axis=axis, out=out, **kwargs)
        if result.shape == (280, 280, 2) and result.dtype == np.dtype(np.int64):
            return _TrackedIndexArray(result, tracker, role="integer_grid")
        return result
    return stack


class CaptureRefinerRepairTests(unittest.TestCase):
    def test_five_successful_lm_updates_feed_native_millimetre_pose_forward(self):
        harness = _Harness()
        frame = _capture_frame()
        observed = np.ones((720, 720), dtype=np.bool_)
        seed = np.eye(4, dtype=np.float64)
        seed[:3, 3] = [0.12, -0.04, 1.2]
        chronology = _ChronologicalSpy()
        refiner = harness.make_refiner(chronology)
        refiner.set_chronological_capture_context({"synthetic": True})

        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            candidate = refiner.refine(frame, observed, seed)

        self.assertEqual(len(harness.initial_poses), 5)
        self.assertEqual(harness.network_calls, 5)
        self.assertEqual(harness.pnp_calls, 5)
        self.assertEqual(harness.lm_calls, 5)
        initial_mm = seed.copy()
        initial_mm[:3, 3] *= 1000.0
        np.testing.assert_allclose(harness.initial_poses[0], initial_mm,
                                   rtol=0.0, atol=2e-4)
        for iteration in range(1, 5):
            np.testing.assert_allclose(
                harness.initial_poses[iteration],
                harness.desired_native_pose_mm(iteration - 1),
                rtol=0.0, atol=2e-4,
                err_msg="next GoTrack render must receive inv(C) @ prior LM pose in mm")
        self.assertIsNotNone(candidate)
        expected_final = harness.desired_native_pose_mm(4)
        expected_final[:3, 3] *= 0.001
        np.testing.assert_allclose(candidate.pose, expected_final, rtol=0.0, atol=2e-6)
        self.assertEqual(candidate.source, "refinement")
        self.assertEqual(candidate.points_object_m.shape[1], 3)
        self.assertTrue(np.isfinite(candidate.points_object_m).all())
        self.assertEqual(len(chronology.retained_packets), 5)
        self.assertEqual(len(chronology.retained_preflights), 5)
        final_retained_mm = chronology.retained_packets[-1]["arrays"]["full_obj_points_mm"]
        np.testing.assert_allclose(candidate.points_object_m, final_retained_mm * 0.001,
                                   rtol=0.0, atol=2e-6,
                                   err_msg="public object points must convert mm to m once")
        for iteration, packet in enumerate(chronology.retained_packets):
            arrays = packet["arrays"]
            source_ids = arrays["source_flat_indices"]
            self.assertEqual(source_ids.dtype, np.dtype(np.int64))
            self.assertEqual(arrays["source_crop_integer_pixels"].dtype,
                             np.dtype(np.int64))
            self.assertEqual(arrays["sample_weights"].dtype, np.dtype(np.float64))
            expected_pixels = np.column_stack((source_ids % 280, source_ids // 280))
            np.testing.assert_array_equal(arrays["source_crop_integer_pixels"], expected_pixels)
            sample_ids = arrays["sample_ids"]
            expected_weights = arrays["full_confidence"].reshape(-1)[source_ids[sample_ids]]
            np.testing.assert_array_equal(arrays["sample_weights"], expected_weights)
            np.testing.assert_array_equal(
                harness.pnp_records[iteration]["obj_points"],
                arrays["sample_obj_points_mm"])
            np.testing.assert_array_equal(
                harness.pnp_records[iteration]["target_points"],
                arrays["sample_target_crop_px"])
            np.testing.assert_array_equal(
                harness.lm_records[iteration]["obj_points"],
                harness.pnp_records[iteration]["obj_points"][:6])
            np.testing.assert_array_equal(
                harness.lm_records[iteration]["target_points"],
                harness.pnp_records[iteration]["target_points"][:6])
            metadata, descriptors = chronology.retained_preflights[iteration]
            byte_metadata, estimated_bytes = chronology.retained_byte_preflights[iteration]
            self.assertEqual(byte_metadata["retained_count"], metadata["retained_count"])
            support_names = {"source_flat_indices", "source_crop_integer_pixels", "sample_weights"}
            self.assertTrue(support_names.issubset(descriptors))
            final_descriptor_bytes = sum(item["nbytes"] for item in descriptors.values())
            support_descriptor_bytes = sum(descriptors[name]["nbytes"] for name in support_names)
            expected_support_bytes = (metadata["retained_count"] * (8 + 2 * 8) +
                                      len(sample_ids) * 8)
            self.assertEqual(support_descriptor_bytes, expected_support_bytes)
            self.assertEqual(estimated_bytes, final_descriptor_bytes,
                             "early estimate must equal the complete emitted array packet")
            self.assertEqual(
                final_descriptor_bytes - support_descriptor_bytes,
                _expected_retained_base_bytes(metadata, len(sample_ids)),
                "base payload estimate is checked independently of the deferred arrays")

    def test_actual_camera_map_faults_reject_before_query_warp_network_or_solver(self):
        for fault in ("offset", "nonfinite_map", "zero_depth", "negative_depth"):
            with self.subTest(fault=fault):
                harness = _Harness(map_fault=fault)
                frame = _capture_frame()
                refiner = harness.make_refiner()
                with harness.guarded_imports(), patch.object(
                        gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                        gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
                    with self.assertRaisesRegex(ValueError, "Capture returned-camera"):
                        refiner.refine(frame, np.ones((720, 720), dtype=np.bool_), np.eye(4))
                self.assertEqual(harness.events, ["template"])
                self.assertEqual(harness.warp_calls, 0)
                self.assertEqual(harness.network_calls, 0)
                self.assertEqual(harness.pnp_calls, 0)
                self.assertEqual(harness.lm_calls, 0)

    def test_unsupported_behind_camera_crop_pixels_do_not_poison_eligible_map(self):
        harness = _Harness(large_rotation=True, solve_success=False)
        frame = _capture_frame()
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            result = refiner.refine(
                frame, np.ones((720, 720), dtype=np.bool_), np.eye(4))

        # The intentionally wide-angle crop sends some integer centers behind
        # the native camera. They are unsupported by crop_validity and must not
        # fail the map-consistency gate; valid interior samples still reach the
        # ordinary solver-unavailable path.
        self.assertIsNone(result)
        self.assertEqual(harness.warp_calls, 1)
        self.assertEqual(harness.network_calls, 1)
        self.assertEqual(harness.pnp_calls, 1)
        self.assertEqual(harness.lm_calls, 0)

    def test_capture_wrong_renderer_mode_fails_before_torch_import(self):
        harness = _Harness()
        harness.renderer.coordinate_mode = "legacy"
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            with self.assertRaisesRegex(ValueError, "integer_centers_v1"):
                refiner.refine(
                    _capture_frame(), np.ones((720, 720), dtype=np.bool_), np.eye(4))
        self.assertEqual(harness.initial_poses, [])
        self.assertEqual(harness.torch_imports, 0)
        self.assertEqual(harness.utils_imports, 0)
        self.assertEqual(harness.network_calls, 0)
        self.assertEqual(harness.warp_calls, 0)

    def test_rejected_chronology_budget_does_not_build_support_arrays_or_emit_retained(self):
        tracker = _AllocationTracker()
        harness = _Harness(solve_success=False)
        harness.confidence = _TrackedIndexArray(
            harness.confidence, tracker, role="confidence")
        chronology = _ChronologicalSpy(
            accept_retained=False, allocation_tracker=tracker)
        refiner = harness.make_refiner(chronology)
        refiner.set_chronological_capture_context({"synthetic": True})
        original_flatnonzero = np.flatnonzero
        original_stack = np.stack
        original_asarray = np.asarray

        def flatnonzero(values):
            if tracker.rejected_budget:
                tracker.post_rejection_allocations.append(("flatnonzero", np.shape(values)))
            return original_flatnonzero(values)

        def tracked_asarray(value, *args, **kwargs):
            result = original_asarray(value, *args, **kwargs)
            if (isinstance(value, _TrackedIndexArray) and
                    value.role == "confidence"):
                marked = result.view(_TrackedIndexArray)
                marked.tracker = value.tracker
                marked.role = value.role
                return marked
            return result

        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm), patch.object(
                gotrack.np, "flatnonzero", flatnonzero), patch.object(
                gotrack.np, "stack", _tracked_stack(original_stack, tracker)), patch.object(
                gotrack.np, "asarray", tracked_asarray):
            result = refiner.refine(
                _capture_frame(), np.ones((720, 720), dtype=np.bool_), np.eye(4))

        self.assertIsNone(result)
        self.assertTrue(chronology.rejected_budget)
        self.assertEqual(chronology.retained_preflights, [],
                         "a rejected byte preflight must not enter descriptor preflight")
        self.assertEqual(chronology.retained_packets, [])
        self.assertEqual(len(chronology.retained_byte_preflights), 1)
        metadata, estimated_bytes = chronology.retained_byte_preflights[0]
        self.assertGreater(metadata["retained_count"], 24)
        sample_count = min(metadata["retained_count"], 10_000)
        expected_additional_bytes = metadata["retained_count"] * (8 + 2 * 8) + sample_count * 8
        expected_base_bytes = _expected_retained_base_bytes(metadata, sample_count)
        self.assertEqual(estimated_bytes, expected_base_bytes + expected_additional_bytes)
        self.assertTrue(
            any(role == "confidence" and source_shape == (280 * 280,) and
                key_shape == (280 * 280,) and key_dtype == "bool"
                for role, source_shape, key_shape, key_dtype
                in tracker.pre_rejection_index_events),
            "confidence marker must reach retained-weight indexing before preflight")
        self.assertEqual(tracker.post_rejection_allocations, [],
                         "flat indices, integer pixel rows and sample weights must not be allocated after rejection")

    def test_native_validity_and_current_foreground_filter_solver_and_candidate_records(self):
        harness = _Harness()
        valid = np.ones((720, 720), dtype=np.bool_)
        observed = np.ones((720, 720), dtype=np.bool_)

        source_valid_hole = (65, 150)
        destination_valid_hole = (225, 150)
        source_hand_hole = (180, 70)
        destination_hand_hole = (120, 215)
        for pixel in (source_valid_hole, destination_valid_hole):
            _paint_native_patch(valid, _crop_to_native_uv(harness, np.array([pixel]))[0])
        for pixel in (source_hand_hole, destination_hand_hole):
            _paint_native_patch(observed, _crop_to_native_uv(harness, np.array([pixel]))[0])

        # One source point moves into each destination hole. Other marked
        # source locations exercise source validity, hand foreground, render
        # visibility, confidence thresholding, NaN flow, and an unclipped
        # outside-crop target.
        harness.flow[0, 0, 150, 85] = destination_valid_hole[0] - 85
        harness.flow[0, 0, 215, 190] = destination_hand_hole[0] - 190
        harness.confidence[0, 20, 20] = 0.0
        harness.confidence[0, 20, 30] = 0.3
        harness.confidence[0, 20, 60] = np.nan
        harness.confidence[0, 20, 70] = np.inf
        harness.flow[0, 0, 20, 40] = np.nan
        harness.flow[0, 0, 20, 50] = 1000.0
        harness.template_mask[160, 160] = False
        frame = _capture_frame(valid)
        chronology = _ChronologicalSpy()
        refiner = harness.make_refiner(chronology)
        refiner.set_chronological_capture_context({"synthetic": True})

        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            candidate = refiner.refine(frame, observed, np.eye(4))

        self.assertIsNotNone(candidate)
        self.assertEqual(harness.pnp_calls, 5)
        self.assertEqual(harness.lm_calls, 5)
        self.assertEqual(len(chronology.retained_packets), 5)
        rejected_source_ids = {
            150 * 280 + 65,   # native sampling-validity hole at source
            150 * 280 + 85,   # valid source, invalid native destination
            70 * 280 + 180,   # current observed hand-like source region
            215 * 280 + 190,  # source aimed at hand-like destination region
            160 * 280 + 160,  # rendered source visibility hole
            20 * 280 + 20,    # zero confidence
            20 * 280 + 30,    # strict > 0.3 confidence boundary
            20 * 280 + 60,    # nonfinite confidence
            20 * 280 + 70,    # infinite confidence
            20 * 280 + 40,    # nonfinite flow
            20 * 280 + 50,    # target outside crop; must not be clipped
        }
        final_arrays = chronology.retained_packets[-1]["arrays"]
        retained_source_ids = final_arrays["source_flat_indices"]
        self.assertTrue(rejected_source_ids.isdisjoint(set(retained_source_ids.tolist())))
        np.testing.assert_array_equal(
            candidate.weights,
            final_arrays["full_confidence"].reshape(-1)[retained_source_ids])
        source_uv = _crop_to_native_uv(
            harness, final_arrays["source_crop_integer_pixels"])
        self.assertTrue(_four_neighbor_support(frame.sampling_valid, source_uv).all())
        self.assertTrue(_four_neighbor_support(observed, source_uv).all())

        candidate_uv = candidate.pixels_image
        self.assertTrue(np.isfinite(candidate_uv).all())
        self.assertTrue(_four_neighbor_support(frame.sampling_valid, candidate_uv).all())
        self.assertTrue(_four_neighbor_support(observed, candidate_uv).all())
        self.assertTrue(np.all(candidate.weights > 0.3))
        self.assertEqual(len(candidate_uv), len(final_arrays["source_flat_indices"]))
        np.testing.assert_allclose(
            candidate_uv, _crop_to_native_uv(harness, final_arrays["full_target_crop_px"]),
            rtol=0.0, atol=1e-9)

        # The captured retained samples are the exact arrays handed to PnP;
        # the first six accepted RANSAC records are the exact LM inputs.
        for iteration, packet in enumerate(chronology.retained_packets):
            arrays = packet["arrays"]
            np.testing.assert_array_equal(
                harness.pnp_records[iteration]["obj_points"], arrays["sample_obj_points_mm"])
            np.testing.assert_array_equal(
                harness.pnp_records[iteration]["target_points"], arrays["sample_target_crop_px"])
            np.testing.assert_array_equal(
                harness.lm_records[iteration]["obj_points"],
                harness.pnp_records[iteration]["obj_points"][:6])
            np.testing.assert_array_equal(
                harness.lm_records[iteration]["target_points"],
                harness.pnp_records[iteration]["target_points"][:6])
            sampled_native_uv = _crop_to_native_uv(
                harness, harness.pnp_records[iteration]["target_points"])
            self.assertTrue(_four_neighbor_support(frame.sampling_valid, sampled_native_uv).all())
            self.assertTrue(_four_neighbor_support(observed, sampled_native_uv).all())

    def test_confidence_cutoff_uses_native_float32_and_float64_ordering(self):
        for native_dtype in (np.float32, np.float64):
            with self.subTest(native_dtype=np.dtype(native_dtype).name):
                harness = _Harness()
                harness.template_mask[:] = False
                boundary_pixels = [(100, 100), (101, 100), (102, 100)]
                other_pixels = [(110 + index, 100) for index in range(24)]
                for x, y in boundary_pixels + other_pixels:
                    harness.template_mask[y, x] = True

                dtype = np.dtype(native_dtype)
                threshold = dtype.type(0.3)
                below_value = np.nextafter(
                    np.array(threshold, dtype=dtype),
                    np.array(-np.inf, dtype=dtype))[()]
                above_value = np.nextafter(
                    np.array(threshold, dtype=dtype),
                    np.array(np.inf, dtype=dtype))[()]
                harness.confidence = np.ones((1, 280, 280), dtype=dtype)
                harness.confidence[0, 100, 100] = below_value
                harness.confidence[0, 100, 101] = threshold
                harness.confidence[0, 100, 102] = above_value
                below_value = harness.confidence[0, 100, 100]
                equal_value = harness.confidence[0, 100, 101]
                above_value = harness.confidence[0, 100, 102]

                chronology = _ChronologicalSpy()
                refiner = harness.make_refiner(chronology)
                refiner.set_chronological_capture_context({"synthetic": True})
                with harness.guarded_imports(), patch.object(
                        gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                        gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
                    candidate = refiner.refine(
                        _capture_frame(), np.ones((720, 720), dtype=np.bool_), np.eye(4))

                self.assertIsNotNone(candidate)
                below_id, equal_id, above_id = (
                    100 * 280 + 100, 100 * 280 + 101, 100 * 280 + 102)
                self.assertEqual(len(chronology.retained_packets), 5)
                final_arrays = chronology.retained_packets[-1]["arrays"]
                final_source_ids = final_arrays["source_flat_indices"]
                self.assertNotIn(below_id, final_source_ids)
                self.assertNotIn(equal_id, final_source_ids)
                self.assertIn(above_id, final_source_ids)
                self.assertEqual(len(final_source_ids), 25)
                sample_source_ids = final_source_ids[final_arrays["sample_ids"]]
                lm_source_ids = sample_source_ids[:6]
                self.assertNotIn(below_id, sample_source_ids)
                self.assertNotIn(equal_id, sample_source_ids)
                self.assertIn(above_id, sample_source_ids)
                self.assertNotIn(below_id, lm_source_ids)
                self.assertNotIn(equal_id, lm_source_ids)
                self.assertIn(above_id, lm_source_ids)

                self.assertEqual(candidate.weights.dtype, np.dtype(np.float64))
                self.assertEqual(final_arrays["sample_weights"].dtype, np.dtype(np.float64))
                expected_candidate_weights = (
                    harness.confidence.reshape(-1)[final_source_ids].astype(np.float64))
                expected_sample_weights = (
                    harness.confidence.reshape(-1)[sample_source_ids].astype(np.float64))
                np.testing.assert_array_equal(candidate.weights, expected_candidate_weights)
                np.testing.assert_array_equal(final_arrays["sample_weights"], expected_sample_weights)
                self.assertEqual(candidate.weights[0], np.float64(above_value))
                self.assertEqual(final_arrays["sample_weights"][0], np.float64(above_value))

    def test_capture_keeps_the_existing_twenty_four_correspondence_floor(self):
        harness = _Harness()
        harness.template_mask[:] = False
        selected = [(80 + (index % 6), 80 + (index // 6)) for index in range(23)]
        for x, y in selected:
            harness.template_mask[y, x] = True
        refiner = harness.make_refiner()

        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            result = refiner.refine(
                _capture_frame(), np.ones((720, 720), dtype=np.bool_), np.eye(4))

        self.assertIsNone(result)
        self.assertEqual(harness.network_calls, 1)
        self.assertEqual(harness.pnp_calls, 0)
        self.assertEqual(harness.lm_calls, 0)

    def test_legacy_refine_keeps_legacy_crop_and_solver_path_without_capture_helpers(self):
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
                gotrack, "_capture_erode3", side_effect=AssertionError("capture erosion used")), patch.object(
                gotrack, "_capture_actual_sampling_map",
                side_effect=AssertionError("capture map helper used")), patch.object(
                gotrack.cv2, "solvePnPRansac", harness.legacy_solve_pnp), patch.object(
                gotrack.cv2, "solvePnPRefineLM", harness.solve_lm):
            candidate = refiner.refine(
                legacy_frame, np.ones((80, 120), dtype=np.bool_), np.eye(4))

        self.assertIsNotNone(candidate)
        self.assertEqual(harness.capture_helper_imports, 0)
        self.assertEqual(harness.warp_calls, 0)
        self.assertEqual(harness.network_calls, 5)
        self.assertEqual(len(harness.template_requests), 5)
        self.assertEqual(len(harness.legacy_solver_calls), 5)
        self.assertEqual(harness.lm_calls, 5)
        for kwargs in harness.template_requests:
            self.assertEqual(kwargs["crop_size"], (280, 280))
            self.assertEqual(kwargs["cropping_type"], "perspective_2d_box")
            self.assertEqual(kwargs["ssaa_factor"], 1.0)
        for args, kwargs in harness.legacy_solver_calls:
            self.assertEqual(args[0].shape[1], 3)
            self.assertEqual(args[1].shape[1], 2)
            self.assertEqual(kwargs["iterationsCount"], 3000)
            self.assertEqual(kwargs["reprojectionError"], 2.0)
            self.assertEqual(kwargs["confidence"], 0.999)
            self.assertEqual(kwargs["flags"], gotrack.cv2.SOLVEPNP_ITERATIVE)


if __name__ == "__main__":
    unittest.main()
