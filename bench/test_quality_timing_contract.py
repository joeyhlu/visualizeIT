"""Synthetic consumer-state tests for explicit capture-time tracking."""

import builtins
import json
import math
import socket
import subprocess
import sys
import tempfile
import unittest
import types
from contextlib import ExitStack
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from .quality_contract import (
    Frame,
    MaskPrediction,
    PoseCandidate,
    SequentialTracker,
    ValidationSettings,
    project,
)
from .quality_appearance import AppearanceCheckedBackend
from .quality_memory import ModelPointMemory
from .quality_time import FrameKey, LEGACY, LEGACY_POLICY, PHYSICAL, TimePolicy


K = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]])
RGB = np.zeros((240, 320, 3), dtype=np.uint8)
MASK = np.zeros((240, 320), dtype=np.uint8)
MASK[10:230, 10:310] = 255
XY = np.array([(x, y) for x in np.linspace(-0.10, 0.10, 8)
               for y in np.linspace(-0.10, 0.10, 8)], dtype=np.float64)
XYZ = np.column_stack((XY, np.zeros(len(XY), dtype=np.float64)))
PHYSICAL_POLICY = TimePolicy(mode=PHYSICAL)


def _pose(angle_deg=0.0, x_m=0.0, y_m=0.0, z_m=1.0):
    theta = math.radians(angle_deg)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((math.cos(theta), -math.sin(theta), 0.0),
                      (math.sin(theta), math.cos(theta), 0.0),
                      (0.0, 0.0, 1.0))
    result[:3, 3] = (x_m, y_m, z_m)
    return result


def _candidate(pose):
    pixels, _ = project(XYZ, pose, K)
    return PoseCandidate(pose.copy(), XYZ.copy(), pixels.copy(), np.ones(len(XYZ)))


def _frame(frame_id, timestamp_s=None, clock_mode=LEGACY):
    return Frame(frame_id, RGB.copy(), K.copy(), timestamp_s=timestamp_s, clock_mode=clock_mode)


def _mask(frame_id, timestamp_s=None, clock_mode=LEGACY, *, state="available", mask=MASK):
    return MaskPrediction(frame_id, mask.copy(), state=state,
                          timestamp_s=timestamp_s, clock_mode=clock_mode)


class PhysicalBackend:
    """A deterministic source-free backend whose only evidence is the fixture pose."""

    def __init__(self, poses=None, *, fail_refine_ids=(), clear_fails=False):
        self.time_policy = PHYSICAL_POLICY
        self.clock_modes_supported = frozenset((PHYSICAL,))
        self.poses = dict(poses or {})
        self.fail_refine_ids = set(fail_refine_ids)
        self.clear_fails = clear_fails
        self.calls = []
        self.refine_seeds = []

    def _target(self, frame):
        return self.poses.get(frame.frame_id, _pose())

    def observe(self, frame, mask):
        self.calls.append(("observe", frame.frame_id, frame.timestamp_s))

    def refine(self, frame, mask, seed):
        self.calls.append(("refine", frame.frame_id, frame.timestamp_s))
        self.refine_seeds.append(None if seed is None else np.array(seed, copy=True))
        if frame.frame_id in self.fail_refine_ids:
            return None
        return _candidate(self._target(frame))

    def recover(self, frame, mask, top_k=5):
        self.calls.append(("recover", frame.frame_id, frame.timestamp_s))
        return [self._target(frame)]

    def motion_seed(self, frame):
        self.calls.append(("motion_seed", frame.frame_id, frame.timestamp_s))
        return None

    def validate_motion(self, candidate, frame):
        self.calls.append(("validate_motion", frame.frame_id, frame.timestamp_s))
        return True, None, {"state": "synthetic_current_frame"}

    def commit(self, frame, mask, pose):
        self.calls.append(("commit", frame.frame_id, frame.timestamp_s))

    def clear_motion(self, key, *, reason):
        self.calls.append(("clear_motion", key, reason))
        if self.clear_fails:
            raise RuntimeError("synthetic cleanup failure")


class LegacyBackend:
    def __init__(self, recovery_pose=None, candidate_pose=None):
        self.calls = []
        self.refine_seeds = []
        self.recovery_pose = _pose() if recovery_pose is None else recovery_pose
        self.candidate_pose = candidate_pose

    def refine(self, frame, mask, seed):
        self.calls.append(("refine", frame.frame_id))
        self.refine_seeds.append(None if seed is None else np.array(seed, copy=True))
        pose = self.candidate_pose
        if pose is None:
            pose = seed if seed is not None else self.recovery_pose
        return _candidate(pose)

    def recover(self, frame, mask, top_k=5):
        self.calls.append(("recover", frame.frame_id))
        return [self.recovery_pose]


def _snapshot(tracker):
    snapshot = {}
    for name, value in vars(tracker).items():
        if name == "backend":
            continue
        if isinstance(value, np.ndarray):
            snapshot[name] = (value.dtype.str, value.shape, value.tobytes())
        elif isinstance(value, list):
            snapshot[name] = tuple(repr(item) for item in value)
        else:
            snapshot[name] = repr(value)
    return snapshot


class _NoSubprocessTestCase(unittest.TestCase):
    def setUp(self):
        guard = mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("subprocess execution is forbidden in tests"),
        )
        guard.start()
        self.addCleanup(guard.stop)


class PhysicalTrackerTests(_NoSubprocessTestCase):
    def tracker(self, backend=None, *, pose=None, seed_time=0.0, settings=None):
        backend = PhysicalBackend() if backend is None else backend
        tracker = SequentialTracker(
            backend,
            initial_pose=_pose() if pose is None else pose,
            settings=ValidationSettings() if settings is None else settings,
            time_policy=PHYSICAL_POLICY,
            initial_frame_id=0,
            initial_timestamp_s=seed_time,
        )
        return tracker, backend

    def test_physical_constructors_require_explicit_finite_timestamps(self):
        frame = _frame(0, 0.0, PHYSICAL)
        mask = _mask(0, 0.0, PHYSICAL)
        self.assertEqual(frame.timestamp_s, mask.timestamp_s)
        with self.assertRaises((TypeError, ValueError)):
            _frame(0, None, PHYSICAL)
        for bad in (float("nan"), float("inf"), True, "1"):
            with self.subTest(timestamp=bad), self.assertRaises((TypeError, ValueError)):
                _frame(0, bad, PHYSICAL)
        with self.assertRaises((TypeError, ValueError)):
            _frame(0, 1.0, LEGACY)
        with self.assertRaises((TypeError, ValueError)):
            _mask(0, None, PHYSICAL)
        with self.assertRaises(ValueError):
            SequentialTracker(
                PhysicalBackend(), initial_pose=_pose(), time_policy=PHYSICAL_POLICY,
                initial_frame_id=0,
            )

    def test_exact_physical_rate_boundaries_at_30_and_60_hz(self):
        cases = ((1 / 60, 17.0, 0.030), (1 / 30, 29.0, 0.050))
        for dt, angle, distance in cases:
            with self.subTest(dt=dt, boundary="inclusive"):
                backend = PhysicalBackend(poses={1: _pose(angle, distance)})
                tracker, _ = self.tracker(backend)
                result = tracker.update(_frame(1, dt, PHYSICAL), _mask(1, dt, PHYSICAL))
                self.assertEqual(result["pose_state"], "tracking")
                self.assertIsNotNone(result["cameraFromObject"])
            with self.subTest(dt=dt, boundary="angular_plus_epsilon"):
                backend = PhysicalBackend(poses={1: _pose(angle + 0.001, 0.0)})
                tracker, _ = self.tracker(backend)
                result = tracker.update(_frame(1, dt, PHYSICAL), _mask(1, dt, PHYSICAL))
                self.assertEqual(result["failure_reason"], "implausible_pose_jump")
                self.assertIsNone(result["cameraFromObject"])
            with self.subTest(dt=dt, boundary="translation_plus_epsilon"):
                backend = PhysicalBackend(poses={1: _pose(0.0, distance + 1e-5)})
                tracker, _ = self.tracker(backend)
                result = tracker.update(_frame(1, dt, PHYSICAL), _mask(1, dt, PHYSICAL))
                self.assertEqual(result["failure_reason"], "implausible_pose_jump")

    def test_same_physical_trajectory_has_same_common_time_state_at_30_and_60_hz(self):
        def run(fps):
            poses = {
                i: _pose(angle_deg=60.0 * (i / fps), x_m=0.2 * (i / fps))
                for i in range(int(0.1 * fps) + 1)
            }
            backend = PhysicalBackend(poses=poses)
            tracker, _ = self.tracker(backend)
            results = []
            for frame_id in range(1, int(0.1 * fps) + 1):
                timestamp = frame_id / fps
                results.append(tracker.update(
                    _frame(frame_id, timestamp, PHYSICAL),
                    _mask(frame_id, timestamp, PHYSICAL),
                ))
            return results[-1]

        at_30 = run(30)
        at_60 = run(60)
        self.assertEqual(at_30["pose_state"], "tracking")
        self.assertEqual(at_60["pose_state"], "tracking")
        np.testing.assert_allclose(at_30["cameraFromObject"], at_60["cameraFromObject"], atol=1e-12)

    def test_invalid_time_or_same_frame_mask_mismatch_is_preflight_only(self):
        cases = (
            (_frame(1, 0.0, PHYSICAL), _mask(1, 0.0, PHYSICAL)),
            (_frame(1, -0.1, PHYSICAL), _mask(1, -0.1, PHYSICAL)),
            (_frame(1, 0.1, PHYSICAL), _mask(1, 0.2, PHYSICAL)),
            (_frame(1, None, LEGACY), _mask(1, None, LEGACY)),
        )
        for frame, mask in cases:
            with self.subTest(frame=(frame.frame_id, frame.timestamp_s, frame.clock_mode),
                              mask=(mask.frame_id, mask.timestamp_s, mask.clock_mode)):
                backend = PhysicalBackend()
                tracker, _ = self.tracker(backend)
                before = _snapshot(tracker)
                with self.assertRaises(ValueError):
                    tracker.update(frame, mask)
                self.assertEqual(_snapshot(tracker), before)
                self.assertEqual(backend.calls, [])

        backend = PhysicalBackend(poses={1: _pose()})
        tracker, _ = self.tracker(backend)
        tracker.update(_frame(1, 0.1, PHYSICAL), _mask(1, 0.1, PHYSICAL))
        before = _snapshot(tracker)
        calls_before = list(backend.calls)
        with self.assertRaises(ValueError):
            tracker.update(_frame(2, 0.05, PHYSICAL), _mask(2, 0.05, PHYSICAL))
        self.assertEqual(_snapshot(tracker), before)
        self.assertEqual(backend.calls, calls_before)

    def test_bad_backend_policy_and_missing_cleanup_reject_at_construction(self):
        class WrongPolicy(PhysicalBackend):
            def __init__(self):
                super().__init__()
                self.time_policy = LEGACY_POLICY

        class NoCleanup(PhysicalBackend):
            clear_motion = None

        for backend in (WrongPolicy(), NoCleanup()):
            with self.subTest(backend=type(backend).__name__), self.assertRaises(ValueError):
                self.tracker(backend)
            self.assertEqual(backend.calls, [])

    def test_unavailable_row_suppresses_pose_clears_flow_and_requires_two_observations(self):
        backend = PhysicalBackend(poses={2: _pose(), 3: _pose()})
        tracker, _ = self.tracker(backend)
        unavailable = tracker.update_unavailable(
            FrameKey(1, 0.1, PHYSICAL), reason="rgb_decode_failed",
        )
        self.assertIsNone(unavailable["cameraFromObject"])
        self.assertEqual(unavailable["render_state"], "suppressed")
        self.assertEqual(unavailable["failure_reason"], "rgb_decode_failed")
        self.assertEqual([call[0] for call in backend.calls], ["clear_motion"])
        first = tracker.update(_frame(2, 0.2, PHYSICAL), _mask(2, 0.2, PHYSICAL))
        self.assertEqual(first["pose_state"], "recovering")
        self.assertIsNone(first["cameraFromObject"])
        second = tracker.update(_frame(3, 0.3, PHYSICAL), _mask(3, 0.3, PHYSICAL))
        self.assertEqual(second["pose_state"], "tracking")

    def test_unavailable_retains_private_pose_through_inclusive_ttl_only(self):
        at_ttl, _ = self.tracker()
        at_ttl.update_unavailable(FrameKey(1, 0.5, PHYSICAL), reason="mask_unavailable")
        self.assertIsNotNone(at_ttl.last_valid_pose)
        self.assertEqual(at_ttl.last_valid_time_s, 0.0)
        past_ttl, _ = self.tracker()
        past_ttl.update_unavailable(FrameKey(1, 0.500001, PHYSICAL), reason="mask_unavailable")
        self.assertIsNone(past_ttl.last_valid_pose)
        self.assertIsNone(past_ttl.last_valid_time_s)

    def test_cleanup_failure_is_terminal_and_never_resumes_backend_calls(self):
        backend = PhysicalBackend(clear_fails=True)
        tracker, _ = self.tracker(backend)
        result = tracker.update_unavailable(FrameKey(1, 0.1, PHYSICAL), reason="rgb_decode_failed")
        self.assertEqual(result["failure_reason"], "motion_cleanup_unconfirmed")
        self.assertTrue(result["terminal"])
        self.assertTrue(result["unmeasured_failure"])
        calls_after_cleanup_failure = list(backend.calls)
        self.assertIsNone(tracker.accepted)
        self.assertIsNone(tracker.pending)
        later = tracker.update(_frame(2, 0.2, PHYSICAL), _mask(2, 0.2, PHYSICAL))
        self.assertIsNone(later["cameraFromObject"])
        self.assertTrue(later["terminal"])
        self.assertEqual(backend.calls, calls_after_cleanup_failure)

    def test_backend_stage_exceptions_are_terminal_and_suppress_later_backend_hooks(self):
        class FailingBackend(PhysicalBackend):
            def __init__(self, failed_stage):
                super().__init__()
                self.failed_stage = failed_stage

            def observe(self, frame, mask):
                super().observe(frame, mask)
                if self.failed_stage == "observe" and frame.frame_id == 1:
                    raise RuntimeError("synthetic observe failure")

            def refine(self, frame, mask, seed):
                if self.failed_stage == "refine" and frame.frame_id == 1:
                    self.calls.append(("refine", frame.frame_id, frame.timestamp_s))
                    raise RuntimeError("synthetic refine failure")
                return super().refine(frame, mask, seed)

            def commit(self, frame, mask, pose):
                super().commit(frame, mask, pose)
                if self.failed_stage == "commit" and frame.frame_id == 1:
                    raise RuntimeError("synthetic commit failure")

        for failed_stage in ("observe", "refine", "commit"):
            with self.subTest(stage=failed_stage):
                backend = FailingBackend(failed_stage)
                tracker, _ = self.tracker(backend)
                result = tracker.update(_frame(1, 0.1, PHYSICAL), _mask(1, 0.1, PHYSICAL))
                self.assertTrue(result["terminal"])
                self.assertTrue(result["unmeasured_failure"])
                self.assertIsNone(result["cameraFromObject"])
                self.assertEqual(result["render_state"], "suppressed")
                self.assertEqual(result["frameId"], 1)
                self.assertEqual(result["timestamp_s"], 0.1)
                calls_at_failure = list(backend.calls)
                for frame_id in (2, 3):
                    later = tracker.update(
                        _frame(frame_id, frame_id / 10, PHYSICAL),
                        _mask(frame_id, frame_id / 10, PHYSICAL),
                    )
                    self.assertTrue(later["terminal"])
                    self.assertIsNone(later["cameraFromObject"])
                    self.assertEqual(later["render_state"], "suppressed")
                    self.assertEqual(later["frameId"], frame_id)
                self.assertEqual(backend.calls, calls_at_failure)

    def test_large_physical_gap_expires_old_pose_before_current_image_recovery(self):
        backend = PhysicalBackend(poses={1: _pose()})
        tracker, _ = self.tracker(backend)
        result = tracker.update(_frame(1, 0.500001, PHYSICAL), _mask(1, 0.500001, PHYSICAL))
        self.assertTrue(any(call[0] == "recover" for call in backend.calls))
        self.assertEqual(result["pose_state"], "recovering")
        self.assertIsNone(result["cameraFromObject"])

    def test_large_gap_before_first_acceptance_clears_pending_confirmation(self):
        backend = PhysicalBackend(poses={0: _pose(), 1: _pose()})
        tracker = SequentialTracker(backend, time_policy=PHYSICAL_POLICY)
        first = tracker.update(_frame(0, 0.0, PHYSICAL), _mask(0, 0.0, PHYSICAL))
        self.assertEqual(first["pose_state"], "recovering")
        later = tracker.update(_frame(1, 10.0, PHYSICAL), _mask(1, 10.0, PHYSICAL))
        self.assertEqual(later["pose_state"], "recovering")
        self.assertIsNone(later["cameraFromObject"])
        self.assertGreaterEqual(sum(call[0] == "recover" for call in backend.calls), 2)

    def test_physical_prefix_is_invariant_to_unseen_suffix(self):
        poses = {i: _pose(2.0 * i, 0.001 * i) for i in range(9)}

        def run(count):
            backend = PhysicalBackend(poses=poses)
            tracker, _ = self.tracker(backend)
            output = []
            for i in range(1, count + 1):
                timestamp = i / 60
                output.append(tracker.update(
                    _frame(i, timestamp, PHYSICAL), _mask(i, timestamp, PHYSICAL),
                ))
            return output

        short = run(3)
        long = run(8)
        self.assertEqual(len(short), 3)
        for earlier, extended in zip(short, long):
            self.assertEqual(earlier["pose_state"], extended["pose_state"])
            self.assertEqual(earlier["failure_reason"], extended["failure_reason"])
            self.assertIsNotNone(earlier["cameraFromObject"])
            np.testing.assert_allclose(earlier["cameraFromObject"], extended["cameraFromObject"],
                                       atol=0.0, rtol=0.0)


class MemoryTimeTests(_NoSubprocessTestCase):
    @staticmethod
    def texture():
        rng = np.random.default_rng(71)
        return rng.integers(0, 256, RGB.shape, dtype=np.uint8)

    def setUp(self):
        from .vision import cv2
        cv2.setNumThreads(1)
        self.rgb = self.texture()
        self.mask = np.full(RGB.shape[:2], 255, dtype=np.uint8)
        self.render_calls = []

        def render_depth(pose, intrinsics, width, height):
            self.render_calls.append((width, height))
            return np.ones((height, width), dtype=np.float32)

        self.memory = ModelPointMemory(render_depth, time_policy=PHYSICAL_POLICY)

    def frame(self, frame_id, timestamp):
        return Frame(frame_id, self.rgb.copy(), K.copy(),
                     timestamp_s=timestamp, clock_mode=PHYSICAL)

    def test_commit_requires_the_exact_current_frame_and_timestamp(self):
        frame = self.frame(0, 0.0)
        self.memory.observe(frame, self.mask)
        self.memory.commit(frame, self.mask, _pose())
        self.assertEqual(len(self.render_calls), 1)
        mismatch = self.frame(0, 0.001)
        with self.assertRaises(ValueError):
            self.memory.commit(mismatch, self.mask, _pose())
        self.assertEqual(len(self.render_calls), 1)

    def test_motion_seed_and_validation_are_bound_to_exact_physical_frame_key(self):
        frame = self.frame(4, 0.25)
        mismatch = self.frame(4, 0.250001)
        self.memory.motion_pose = _pose()
        pixels, _ = project(XYZ, _pose(), K)
        self.memory.evidence = {
            "key": frame.key,
            "xyz": XYZ.copy(),
            "pixels": pixels.copy(),
            "intrinsics": K.copy(),
            "scale_720": 1.0,
            "support": 0.5,
            "frame_id": frame.frame_id,
            "timestamp_s": frame.timestamp_s,
            "clock_mode": frame.clock_mode,
        }
        np.testing.assert_array_equal(self.memory.seed(frame), _pose())
        self.assertIsNone(self.memory.seed(mismatch))
        candidate = _candidate(_pose())
        self.assertEqual(self.memory.validate(candidate, mismatch)[2]["state"], "unavailable")
        self.assertEqual(self.memory.validate(candidate, frame)[2]["state"], "available")

    def test_clear_motion_requires_observation_before_commit(self):
        first = self.frame(0, 0.0)
        self.memory.observe(first, self.mask)
        self.memory.commit(first, self.mask, _pose())
        key = FrameKey(1, 0.1, PHYSICAL)
        self.memory.clear_motion(key, reason="known_gap")
        renders_before = len(self.render_calls)
        state_before = (self.memory.last_id, self.memory.last_time_s,
                        self.memory.anchor_id, self.memory.anchor_time_s,
                        self.memory.motion_pose, self.memory.evidence)
        current = self.frame(1, 0.1)
        with self.assertRaises(ValueError):
            self.memory.commit(current, self.mask, _pose())
        self.assertEqual(len(self.render_calls), renders_before)
        self.assertEqual((self.memory.last_id, self.memory.last_time_s,
                          self.memory.anchor_id, self.memory.anchor_time_s,
                          self.memory.motion_pose, self.memory.evidence), state_before)
        self.memory.observe(current, self.mask)
        self.memory.commit(current, self.mask, _pose())
        self.assertEqual(len(self.render_calls), renders_before + 1)

    def test_clear_motion_removes_all_evidence_and_allows_one_same_key_observation(self):
        frame0 = self.frame(0, 0.0)
        self.memory.observe(frame0, self.mask)
        self.memory.commit(frame0, self.mask, _pose())
        self.memory.points = np.ones((24, 2), dtype=np.float32)
        self.memory.xyz = np.ones((24, 3), dtype=np.float64)
        self.memory.evidence = {"key": frame0.key}
        key = FrameKey(1, 0.1, PHYSICAL)
        self.memory.clear_motion(key, reason="known_unavailable_row")
        for field in ("gray", "points", "xyz", "anchor_id", "anchor_time_s", "motion_pose", "evidence"):
            with self.subTest(field=field):
                self.assertIsNone(getattr(self.memory, field))
        # A TTL-gap cleanup may be followed by the current frame's observation
        # exactly once, without inventing another ID or timestamp.
        frame1 = self.frame(1, 0.1)
        self.memory.observe(frame1, self.mask)
        with self.assertRaises(ValueError):
            self.memory.observe(frame1, self.mask)

    def test_seconds_ttl_is_inclusive_at_half_second_then_expires(self):
        at_limit = ModelPointMemory(lambda *args: np.ones((120, 160), np.float32),
                                    time_policy=PHYSICAL_POLICY)
        f0 = self.frame(0, 0.0)
        at_limit.observe(f0, self.mask)
        at_limit.commit(f0, self.mask, _pose())
        at_limit.points = None
        at_limit.observe(self.frame(1, 0.5), self.mask)
        self.assertEqual(at_limit.anchor_id, 0)
        self.assertEqual(at_limit.anchor_time_s, 0.0)

        expired = ModelPointMemory(lambda *args: np.ones((120, 160), np.float32),
                                  time_policy=PHYSICAL_POLICY)
        expired.observe(f0, self.mask)
        expired.commit(f0, self.mask, _pose())
        expired.points = None
        expired.observe(self.frame(1, 0.500001), self.mask)
        self.assertIsNone(expired.anchor_id)
        self.assertIsNone(expired.anchor_time_s)
        self.assertIsNone(expired.motion_pose)

    def test_physical_mode_refuses_legacy_age_override(self):
        with self.assertRaises(ValueError):
            ModelPointMemory(lambda *args: None, max_age=29, time_policy=PHYSICAL_POLICY)
        self.assertEqual(self.memory.max_age, 30)
        with self.assertRaises(ValueError):
            self.memory.max_age = 29


class TimedWrapperTests(_NoSubprocessTestCase):
    def test_appearance_key_mismatch_is_rejected_before_backend_validation(self):
        backend = PhysicalBackend()
        backend.refiner = SimpleNamespace(renderer=object())
        wrapper = AppearanceCheckedBackend(
            backend, object(), time_policy=PHYSICAL_POLICY,
        )
        frame = _frame(0, 0.0, PHYSICAL)
        wrapper.observe(frame, MASK.copy())
        before = list(backend.calls)
        wrong_same_id = _frame(0, 0.01, PHYSICAL)
        with self.assertRaises(ValueError):
            wrapper.validate_motion(_candidate(_pose()), wrong_same_id)
        self.assertEqual(backend.calls, before)

    def test_appearance_observe_rejects_bad_clock_or_duplicate_key_without_state_change(self):
        backend = PhysicalBackend()
        backend.refiner = SimpleNamespace(renderer=object())
        wrapper = AppearanceCheckedBackend(backend, object(), time_policy=PHYSICAL_POLICY)
        frame = _frame(0, 0.0, PHYSICAL)
        original_mask = MASK.copy()
        wrapper.observe(frame, original_mask)
        before = list(backend.calls)
        for bad_frame in (_frame(1, None, LEGACY), _frame(0, 0.01, PHYSICAL)):
            with self.subTest(key=bad_frame.key), self.assertRaises(ValueError):
                wrapper.observe(bad_frame, np.zeros_like(MASK))
            self.assertIs(wrapper.frame, frame)
            np.testing.assert_array_equal(wrapper.mask, original_mask)
            self.assertEqual(backend.calls, before)

    def test_appearance_invalid_cleanup_preserves_state_and_valid_partial_cleanup_clears(self):
        backend = PhysicalBackend()
        backend.refiner = SimpleNamespace(renderer=object())
        wrapper = AppearanceCheckedBackend(backend, object(), time_policy=PHYSICAL_POLICY)
        frame = _frame(0, 0.0, PHYSICAL)
        mask = MASK.copy()
        wrapper.observe(frame, mask)
        before = list(backend.calls)
        invalid_requests = (
            (FrameKey(1, None, LEGACY), "gap"),
            (FrameKey(1, 0.1, PHYSICAL), ""),
        )
        for key, reason in invalid_requests:
            with self.subTest(key=key, reason=reason), self.assertRaises(ValueError):
                wrapper.clear_motion(key, reason=reason)
            self.assertIs(wrapper.frame, frame)
            np.testing.assert_array_equal(wrapper.mask, mask)
            self.assertEqual(backend.calls, before)

        backend.clear_fails = True
        with self.assertRaises(RuntimeError):
            wrapper.clear_motion(FrameKey(1, 0.1, PHYSICAL), reason="gap")
        self.assertIsNone(wrapper.frame)
        self.assertIsNone(wrapper.mask)
        self.assertEqual(backend.calls[-1][0], "clear_motion")

    def test_foundpose_disabled_memory_rejects_wrong_clock_frames_before_backend_access(self):
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            root_name = name.split(".", 1)[0]
            if root_name in {"torch", "OpenGL", "pyrender", "trimesh"} or name.startswith("utils"):
                raise AssertionError(f"forbidden inference import in disabled-memory test: {name}")
            return real_import(name, *args, **kwargs)
        with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
            from .quality_foundpose import FoundPoseRecovery

        recovery = FoundPoseRecovery.__new__(FoundPoseRecovery)
        recovery.time_policy = PHYSICAL_POLICY
        recovery.memory = None
        wrong_mode = _frame(7, None, LEGACY)
        with self.assertRaises(ValueError):
            recovery.motion_seed(wrong_mode)
        with self.assertRaises(ValueError):
            recovery.validate_motion(_candidate(_pose()), wrong_mode)

        legacy = FoundPoseRecovery.__new__(FoundPoseRecovery)
        legacy.time_policy = LEGACY_POLICY
        legacy.memory = None
        wrong_mode_physical = _frame(7, 0.2, PHYSICAL)
        with self.assertRaises(ValueError):
            legacy.motion_seed(wrong_mode_physical)
        with self.assertRaises(ValueError):
            legacy.validate_motion(_candidate(_pose()), wrong_mode_physical)

    def test_appearance_cleanup_clears_retained_frame_and_mask(self):
        backend = PhysicalBackend()
        backend.refiner = SimpleNamespace(renderer=object())
        wrapper = AppearanceCheckedBackend(backend, object(), time_policy=PHYSICAL_POLICY)
        frame = _frame(0, 0.0, PHYSICAL)
        wrapper.observe(frame, MASK.copy())
        wrapper.clear_motion(FrameKey(1, 0.1, PHYSICAL), reason="unavailable")
        self.assertIsNone(wrapper.frame)
        self.assertIsNone(wrapper.mask)
        self.assertEqual(backend.calls[-1][0], "clear_motion")

    def test_foundpose_wrapper_forwards_frame_keys_without_importing_models(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            root_name = name.split(".", 1)[0]
            if root_name in {"torch", "OpenGL", "pyrender", "trimesh"} or name.startswith("utils"):
                raise AssertionError(f"forbidden model/runtime import in wrapper test: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
            from .quality_foundpose import FoundPoseRecovery

        class MemorySpy:
            def __init__(self):
                self.calls = []

            def observe(self, *args): self.calls.append(("observe", args))
            def commit(self, *args): self.calls.append(("commit", args))
            def seed(self, *args): self.calls.append(("seed", args)); return _pose()
            def validate(self, *args): self.calls.append(("validate", args)); return True, None, {}
            def clear_motion(self, *args, **kwargs): self.calls.append(("clear", args, kwargs))

        recovery = FoundPoseRecovery.__new__(FoundPoseRecovery)
        recovery.time_policy = PHYSICAL_POLICY
        recovery.memory = MemorySpy()
        frame = _frame(2, 0.2, PHYSICAL)
        recovery.observe(frame, MASK.copy())
        recovery.commit(frame, MASK.copy(), _pose())
        self.assertIsNotNone(recovery.motion_seed(frame))
        self.assertTrue(recovery.validate_motion(_candidate(_pose()), frame)[0])
        with self.assertRaises(ValueError):
            recovery.motion_seed(2)
        before_wrong_clock = list(recovery.memory.calls)
        wrong_clock_frame = _frame(3, None, LEGACY)
        with self.assertRaises(ValueError):
            recovery.motion_seed(wrong_clock_frame)
        with self.assertRaises(ValueError):
            recovery.validate_motion(_candidate(_pose()), wrong_clock_frame)
        self.assertEqual(recovery.memory.calls, before_wrong_clock)
        self.assertEqual([call[0] for call in recovery.memory.calls],
                         ["observe", "commit", "seed", "validate"])
        recovery.clear_motion(FrameKey(3, 0.3, PHYSICAL), reason="gap")
        self.assertEqual(recovery.memory.calls[-1][0], "clear")


class LegacyCompatibilityTests(_NoSubprocessTestCase):
    def test_legacy_60fps_bounds_and_private_pose_ttl_endpoint(self):
        backend = LegacyBackend()
        tracker = SequentialTracker(backend, initial_pose=_pose())
        exact = tracker.update(Frame(100, RGB.copy(), K.copy()), _mask(100))
        self.assertEqual(exact["pose_state"], "tracking")
        # Existing settings are the preserved nominal-60 contract: one ID step
        # allows 5 + 12 degrees and 0.01 + 0.02 metres.
        exact_angle = LegacyBackend(candidate_pose=_pose(17.0))
        result = SequentialTracker(exact_angle, initial_pose=_pose()).update(
            Frame(100, RGB.copy(), K.copy()), _mask(100),
        )
        self.assertEqual(result["pose_state"], "tracking")
        too_far = LegacyBackend(candidate_pose=_pose(17.001))
        result = SequentialTracker(too_far, initial_pose=_pose()).update(
            Frame(100, RGB.copy(), K.copy()), _mask(100),
        )
        self.assertEqual(result["failure_reason"], "implausible_pose_jump")
        too_translated = LegacyBackend(candidate_pose=_pose(0.0, 0.03001))
        result = SequentialTracker(too_translated, initial_pose=_pose()).update(
            Frame(100, RGB.copy(), K.copy()), _mask(100),
        )
        self.assertEqual(result["failure_reason"], "implausible_pose_jump")

        def first_valid_after_gap(gap):
            fresh = _pose(35.0, 0.12)
            legacy = LegacyBackend(recovery_pose=fresh)
            tracker = SequentialTracker(legacy, initial_pose=_pose())
            tracker.update(Frame(100, RGB.copy(), K.copy()), _mask(100))
            for frame_id in range(101, 100 + gap):
                tracker.update(Frame(frame_id, RGB.copy(), K.copy()),
                               _mask(frame_id, state="lost", mask=np.zeros_like(MASK)))
            result = tracker.update(Frame(100 + gap, RGB.copy(), K.copy()), _mask(100 + gap))
            return legacy, result

        at_ttl, at_result = first_valid_after_gap(30)
        self.assertEqual(at_result["pose_state"], "recovering")
        np.testing.assert_array_equal(at_ttl.refine_seeds[-1], _pose())
        expired, expired_result = first_valid_after_gap(31)
        self.assertTrue(any(call[0] == "recover" for call in expired.calls))
        self.assertEqual(expired_result["pose_state"], "recovering")
        np.testing.assert_array_equal(expired.refine_seeds[-1], _pose(35.0, 0.12))


class RunnerPreflightTests(_NoSubprocessTestCase):
    def test_missing_physical_timestamps_fail_before_source_hash_or_model_import(self):
        from . import quality_runner

        malformed = {
            "schema_version": 2,
            "clock": {"mode": PHYSICAL, "units": "seconds", "source_units": "milliseconds",
                      "timestamp_source": "synthetic", "nominal_fps": 30},
            "timeline": [{"frame_id": 0, "source_frame_id": 400,
                          "source_timestamp": 1000, "role": "setup"}],
            "setup_frame_id": 0,
            "frame_ids": [],
            "source_hashes": {"missing.mp4": "0" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.json").write_text(json.dumps(malformed), encoding="utf-8")
            with mock.patch.object(quality_runner, "digest") as digest:
                real_import = builtins.__import__

                def guarded_import(name, *args, **kwargs):
                    root_name = name.split(".", 1)[0]
                    if root_name in {"torch", "OpenGL", "pyrender", "trimesh", "PIL"} or name.endswith(
                        ("quality_gotrack", "glb_model", "show3d_experiment", "renderer")
                    ):
                        raise AssertionError(f"forbidden inference import before timing validation: {name}")
                    return real_import(name, *args, **kwargs)

                with mock.patch("subprocess.Popen", side_effect=AssertionError("Popen is forbidden")), \
                        mock.patch("socket.create_connection", side_effect=AssertionError("network is forbidden")), \
                        mock.patch.object(builtins, "__import__", side_effect=guarded_import):
                    with self.assertRaises(ValueError):
                        quality_runner.smoke(root, root / "smoke.json", "cpu")
                digest.assert_not_called()

    @staticmethod
    def _write_physical_bundle(root):
        from . import quality_runner

        root.mkdir(parents=True, exist_ok=True)
        source = root / "source.bin"
        asset = root / "object.bin"
        source.write_bytes(b"synthetic source bytes")
        asset.write_bytes(b"synthetic metric asset bytes")
        source_ids = (90, 92, 95)
        source_ticks = (1_000_000, 1_033_333, 1_066_666)
        seconds = tuple(value / 1_000_000 for value in source_ticks)
        manifest = {
            "schema_version": 2,
            "object": "keyboard",
            "object_id": "synthetic-object-id",
            "video": source.name,
            "asset": asset.name,
            "units": "metres",
            "intrinsics": K.tolist(),
            "controlled_initial_pose": _pose().tolist(),
            "source_hashes": {
                source.name: quality_runner.digest(source),
                asset.name: quality_runner.digest(asset),
            },
            "clock": {
                "mode": PHYSICAL,
                "units": "seconds",
                "source_units": "microseconds",
                "timestamp_source": "synthetic declared capture metadata",
                "nominal_fps": 30,
                "time_base": {"num": 1, "den": 1_000_000},
                "timeline_span_seconds": seconds[-1] - seconds[0],
            },
            "timeline": [
                {"frame_id": ordinal, "source_frame_id": source_id,
                 "source_timestamp": tick, "timestamp_s": time_s,
                 "role": "setup" if ordinal == 0 else "scored"}
                for ordinal, (source_id, tick, time_s) in enumerate(
                    zip(source_ids, source_ticks, seconds)
                )
            ],
            "source_gaps": [
                {"previous_source_frame_id": 90, "current_source_frame_id": 92,
                 "missing_capture_count": 1},
                {"previous_source_frame_id": 92, "current_source_frame_id": 95,
                 "missing_capture_count": 2},
            ],
            "setup_frame_id": 0,
            "frame_ids": [1, 2],
        }
        raw = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode("utf-8")
        (root / "input.json").write_bytes(raw)
        return manifest, raw

    def test_pose_main_rejects_missing_physical_timestamps_before_runtime_or_video_hooks(self):
        from . import quality_runner

        malformed = {
            "schema_version": 2,
            "object": "keyboard",
            "units": "metres",
            "source_hashes": {"missing.bin": "0" * 64},
            "clock": {"mode": PHYSICAL, "units": "seconds", "source_units": "milliseconds",
                      "timestamp_source": "synthetic", "nominal_fps": 30},
            "timeline": [{"frame_id": 0, "source_frame_id": 400,
                          "source_timestamp": 1000, "role": "setup"}],
            "setup_frame_id": 0,
            "frame_ids": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.json").write_text(json.dumps(malformed), encoding="utf-8")
            fake_args = ["quality_runner", "pose", "--bundle", str(root), "--output",
                         str(root / "pose.json"), "--masks", str(root / "masks"), "--device", "cpu"]
            real_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name.split(".", 1)[0] == "torch" or name.endswith(("quality_gotrack", "quality_foundpose")):
                    raise AssertionError(f"runtime imported before timing preflight: {name}")
                return real_import(name, *args, **kwargs)
            with mock.patch("sys.argv", fake_args), \
                    mock.patch.object(quality_runner.cv2, "VideoCapture",
                                      side_effect=AssertionError("video opened before preflight")), \
                    mock.patch.object(quality_runner.cv2, "imread",
                                      side_effect=AssertionError("mask decoded before preflight")), \
                    mock.patch.object(builtins, "__import__", side_effect=guarded_import):
                with self.assertRaises((ValueError, SystemExit)):
                    quality_runner.main()

    def _write_physical_cache_results(self, bundle, masks_root, *, requested, diagnostic=False):
        from . import quality_runner

        manifest = quality_runner.read_input(bundle)
        masks_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for source_row in manifest["timeline"]:
            frame_id = source_row["frame_id"]
            if frame_id in requested:
                relative = f"{frame_id}.mask"
                artifact = masks_root / relative
                artifact.write_bytes(f"synthetic mask row {frame_id}".encode("ascii"))
                row = {
                    "mask_state": "available",
                    "measurement_state": "measured",
                    "path": relative,
                    "mask_sha256": quality_runner.digest(artifact),
                    "failure_reason": None,
                }
            else:
                row = {
                    "mask_state": "lost",
                    "measurement_state": "unmeasured",
                    "path": None,
                    "mask_sha256": None,
                    "failure_reason": "outside_requested_prefix",
                }
            row.update(
                frameId=frame_id,
                sourceFrameId=source_row["source_frame_id"],
                timestamp_s=source_row["timestamp_s"],
                clock_mode=PHYSICAL,
                role=source_row["role"],
            )
            rows.append(row)
        timing = manifest["_timing"]
        record = {
            "schema_version": 2,
            "clock_mode": PHYSICAL,
            "input_manifest_sha256": timing["manifest_sha256"],
            "timestamp_table_sha256": timing["timestamp_table_sha256"],
            "coverage_complete": True,
            "status": "complete" if len(requested) == 3 else "diagnostic_prefix",
            "stage_completed": True,
            "expected_setup_frames": 1,
            "expected_scored_frames": 2,
            "planned_frame_ids": [0, 1, 2],
            "requested_frame_ids": list(requested),
            "frames": rows,
            "diagnostic_control": diagnostic,
        }
        (masks_root / "results.json").write_text(json.dumps(record), encoding="utf-8")

    def test_pose_main_rejects_short_cache_prefix_and_diagnostic_overwrite_before_runtime(self):
        from . import quality_runner

        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            root_name = name.split(".", 1)[0]
            if root_name == "torch" or name.endswith(("quality_gotrack", "quality_foundpose")):
                raise AssertionError(f"runtime imported before physical preflight: {name}")
            return real_import(name, *args, **kwargs)

        for label, requested, diagnostic, output_name in (
                ("short_cache_prefix", [0, 1], False, "pose.json"),
                ("diagnostic_output_protection", [0, 1, 2], True, "complete.json")):
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = root / "bundle"
                self._write_physical_bundle(bundle)
                masks_root = root / "masks"
                self._write_physical_cache_results(
                    bundle, masks_root, requested=requested, diagnostic=diagnostic,
                )
                argv = ["quality_runner", "pose", "--bundle", str(bundle), "--output",
                        str(root / output_name), "--masks", str(masks_root), "--device", "cpu",
                        "--mode", "complete", "--limit", "2"]
                with mock.patch("sys.argv", argv), \
                        mock.patch.object(quality_runner.cv2, "VideoCapture",
                                          side_effect=AssertionError("video opened before preflight")), \
                        mock.patch.object(quality_runner.cv2, "imread",
                                          side_effect=AssertionError("mask decoded before preflight")), \
                        mock.patch.object(builtins, "__import__", side_effect=guarded_import):
                    with self.assertRaises(SystemExit):
                        quality_runner.main()

    def test_legacy_runner_requires_exact_raw_manifest_bytes_and_three_pinned_inputs(self):
        from . import quality_runner

        self.assertEqual(set(quality_runner._LEGACY_INPUTS), {
            "3CFC13197E096739698B5867383CD34808EB571FFA5A2E387C17E84819D00B80",
            "4422C7373A09564112B0069153AAB40C803148F109EACEAA737BFB0398A268A3",
            "F012F28918534F2A7AC3EFD59841A347A8BAD384D9585130DB88EAAC56377D5D",
        })
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            source = bundle / "source.bin"
            source.write_bytes(b"fixture video bytes")
            manifest = {
                "schema_version": 1,
                "object": "keyboard",
                "units": "metres",
                "source_hashes": {source.name: quality_runner.digest(source)},
                "setup_frame_id": 10,
                "frame_ids": [11],
            }
            raw = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            pin = quality_runner.parse_timing_manifest(raw)["manifest_sha256"]
            with mock.patch.dict(quality_runner._LEGACY_INPUTS, {
                    pin: {"object": "keyboard", "source_hashes": manifest["source_hashes"]}}, clear=True):
                (bundle / "input.json").write_bytes(raw)
                self.assertEqual(quality_runner.read_input(bundle)["_timing"]["clock_mode"], LEGACY)
                (bundle / "input.json").write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    quality_runner.read_input(bundle)

    def test_physical_prefix_cache_joins_setup_scored_ordinals_source_ids_and_times(self):
        from . import quality_runner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            manifest, raw = self._write_physical_bundle(bundle)
            parsed = quality_runner.read_input(bundle)
            self.assertEqual([key.frame_id for key in parsed["_timing"]["scored_keys"]], [1, 2])
            self.assertEqual(parsed["_timing"]["source_frame_ids"], {0: 90, 1: 92, 2: 95})
            masks_root = root / "masks"
            masks_root.mkdir()
            requested_rows = []
            rows = []
            for row in manifest["timeline"]:
                frame_id = row["frame_id"]
                if frame_id < 2:
                    relative = f"{frame_id}.bin"
                    artifact = masks_root / relative
                    artifact.write_bytes(f"synthetic mask {frame_id}".encode("ascii"))
                    entry = {
                        "frameId": frame_id,
                        "sourceFrameId": row["source_frame_id"],
                        "timestamp_s": row["timestamp_s"],
                        "clock_mode": PHYSICAL,
                        "role": row["role"],
                        "mask_state": "available",
                        "measurement_state": "measured",
                        "path": relative,
                        "mask_sha256": quality_runner.digest(artifact),
                        "failure_reason": None,
                    }
                    requested_rows.append(frame_id)
                else:
                    entry = {
                        "frameId": frame_id,
                        "sourceFrameId": row["source_frame_id"],
                        "timestamp_s": row["timestamp_s"],
                        "clock_mode": PHYSICAL,
                        "role": row["role"],
                        "mask_state": "lost",
                        "measurement_state": "unmeasured",
                        "path": None,
                        "mask_sha256": None,
                        "failure_reason": "outside_requested_prefix",
                    }
                rows.append(entry)
            timing = parsed["_timing"]
            record = {
                "schema_version": 2,
                "clock_mode": PHYSICAL,
                "input_manifest_sha256": timing["manifest_sha256"],
                "timestamp_table_sha256": timing["timestamp_table_sha256"],
                "coverage_complete": True,
                "status": "diagnostic_prefix",
                "stage_completed": True,
                "expected_setup_frames": 1,
                "expected_scored_frames": 2,
                "planned_frame_ids": [0, 1, 2],
                "requested_frame_ids": requested_rows,
                "frames": rows,
            }
            (masks_root / "results.json").write_text(json.dumps(record), encoding="utf-8")
            loaded, by_id = quality_runner._physical_mask_cache(parsed, masks_root)
            self.assertEqual(loaded["requested_frame_ids"], [0, 1])
            self.assertEqual([by_id[index]["sourceFrameId"] for index in range(3)], [90, 92, 95])
            self.assertEqual([by_id[index]["timestamp_s"] for index in range(3)],
                             [row["timestamp_s"] for row in manifest["timeline"]])
            self.assertEqual(by_id[2]["failure_reason"], "outside_requested_prefix")
            self.assertIsNone(by_id[2]["path"])
            self.assertEqual(
                [quality_runner._frame_key(parsed, frame_id) for frame_id in (0, 1, 2)],
                [FrameKey(0, 1.0, PHYSICAL), FrameKey(1, 1.033333, PHYSICAL),
                 FrameKey(2, 1.066666, PHYSICAL)],
            )

            corruptions = {}
            missing_suffix = json.loads(json.dumps(record))
            missing_suffix["frames"].pop()
            corruptions["missing_full_table_suffix"] = missing_suffix
            duplicate = json.loads(json.dumps(record))
            duplicate["frames"].append(dict(duplicate["frames"][0]))
            corruptions["duplicate_frame_key"] = duplicate
            wrong_source = json.loads(json.dumps(record))
            wrong_source["frames"][1]["sourceFrameId"] = 999
            corruptions["wrong_source_id"] = wrong_source
            wrong_time = json.loads(json.dumps(record))
            wrong_time["frames"][1]["timestamp_s"] += 0.001
            corruptions["wrong_timestamp"] = wrong_time
            wrong_mode = json.loads(json.dumps(record))
            wrong_mode["frames"][1]["clock_mode"] = LEGACY
            corruptions["wrong_clock_mode"] = wrong_mode

            real_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                root_name = name.split(".", 1)[0]
                if root_name == "torch" or name.endswith(("quality_gotrack", "quality_foundpose")):
                    raise AssertionError(f"runtime imported before full mask-cache validation: {name}")
                return real_import(name, *args, **kwargs)
            for label, corrupted in corruptions.items():
                with self.subTest(corruption=label):
                    (masks_root / "results.json").write_text(
                        json.dumps(corrupted), encoding="utf-8",
                    )
                    with mock.patch.object(quality_runner.cv2, "VideoCapture",
                                           side_effect=AssertionError("video opened before mask validation")), \
                            mock.patch.object(quality_runner.cv2, "imread",
                                              side_effect=AssertionError("mask decoded before full-table validation")), \
                            mock.patch.object(builtins, "__import__", side_effect=guarded_import):
                        with self.assertRaises(ValueError):
                            quality_runner.pose_stage(
                                bundle, masks_root, root / "pose.json", "cpu", "complete", limit=1,
                            )

    def _run_stubbed_physical_pose(self, root, *, limit, failed_stage=None):
        from . import quality_runner

        bundle = root / "bundle"
        self._write_physical_bundle(bundle)
        masks_root = root / "masks"
        self._write_physical_cache_results(bundle, masks_root, requested=[0, 1, 2])
        cache = root / "cache"
        bank_path = cache / "banks" / "keyboard-foundpose.pt"
        bank_path.parent.mkdir(parents=True)
        bank_path.write_bytes(b"synthetic fake checkpoint; never deserialized")
        smoke_path = cache / "smoke" / "keyboard-cpu.json"
        smoke_path.parent.mkdir(parents=True)
        smoke_path.write_text(json.dumps({"smoke_passed": True}), encoding="utf-8")

        class FakeRenderer:
            def __init__(self, *args, **kwargs): self.closed = False
            def close(self): self.closed = True

        class FailingBackend(PhysicalBackend):
            def __init__(self): super().__init__()
            def refine(self, frame, mask, seed):
                if failed_stage == "refine" and frame.frame_id == 1:
                    self.calls.append(("refine", frame.frame_id, frame.timestamp_s))
                    raise RuntimeError("synthetic terminal refine failure")
                return super().refine(frame, mask, seed)

        fake_torch = types.ModuleType("torch")
        fake_torch.load = lambda *args, **kwargs: {"synthetic": True}
        fake_gotrack = types.ModuleType("bench.quality_gotrack")
        fake_gotrack.load_network = lambda device: object()
        fake_gotrack.TexturedRenderer = FakeRenderer
        fake_gotrack.GoTrackRefiner = object
        fake_foundpose = types.ModuleType("bench.quality_foundpose")
        fake_foundpose.FoundPoseRecovery = lambda *args, **kwargs: FailingBackend()
        fake_appearance = types.ModuleType("bench.quality_appearance")
        fake_appearance.AppearanceCheckedBackend = object
        fake_appearance.AppearanceSettings = object
        fake_modules = {
            "torch": fake_torch,
            "bench.quality_gotrack": fake_gotrack,
            "bench.quality_foundpose": fake_foundpose,
            "bench.quality_appearance": fake_appearance,
        }
        read_calls = []
        mask_calls = []

        def read_rgb(_bundle, _manifest, frame_id):
            read_calls.append(frame_id)
            return RGB.copy()

        def read_mask(path, _mode):
            mask_calls.append(Path(path).name)
            return MASK.copy()

        # The process harness blocks importing torch and model modules before
        # consulting sys.modules.  Intercept only these explicit synthetic
        # modules; every unrelated import still reaches the harness guard.
        real_import = builtins.__import__

        def import_synthetic_only(name, globals=None, locals=None, fromlist=(), level=0):
            fake_name = name
            if level == 1 and name in (
                    "quality_gotrack", "quality_foundpose", "quality_appearance"):
                fake_name = f"bench.{name}"
            if fake_name in fake_modules:
                return fake_modules[fake_name]
            return real_import(name, globals, locals, fromlist, level)

        output = root / "pose.json"
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, fake_modules))
            stack.enter_context(mock.patch.object(
                builtins, "__import__", side_effect=import_synthetic_only,
            ))
            stack.enter_context(mock.patch.object(quality_runner, "CACHE", cache))
            stack.enter_context(mock.patch.object(quality_runner, "load_network", create=True,
                                                   side_effect=AssertionError("unexpected loader")))
            stack.enter_context(mock.patch.object(quality_runner, "TexturedRenderer", create=True,
                                                   side_effect=AssertionError("unexpected renderer")))
            stack.enter_context(mock.patch.object(quality_runner, "_make_gotrack_refiner", return_value=object()))
            stack.enter_context(mock.patch.object(quality_runner, "inference_provenance", return_value={"synthetic": True}))
            stack.enter_context(mock.patch.object(quality_runner, "device_info", return_value={"device": "cpu"}))
            stack.enter_context(mock.patch.object(quality_runner, "read_rgb", side_effect=read_rgb))
            stack.enter_context(mock.patch.object(quality_runner.cv2, "imread", side_effect=read_mask))
            quality_runner.pose_stage(
                bundle, masks_root, output, "cpu", "controlled", limit=limit,
            )
        return json.loads(output.read_text(encoding="utf-8")), read_calls, mask_calls

    def test_physical_pose_prefix_saves_all_scored_rows_without_decoding_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, read_calls, mask_calls = self._run_stubbed_physical_pose(
                Path(temporary), limit=1,
            )
        self.assertEqual(read_calls, [1])
        self.assertEqual(mask_calls, ["1.mask"])
        self.assertEqual(result["status"], "diagnostic_prefix")
        self.assertFalse(result["complete"])
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(len(result["frames"]), 2)
        requested, suffix = result["frames"]
        self.assertEqual((requested["frameId"], requested["sourceFrameId"], requested["timestamp_s"]),
                         (1, 92, 1.033333))
        self.assertEqual((suffix["frameId"], suffix["sourceFrameId"], suffix["timestamp_s"]),
                         (2, 95, 1.066666))
        self.assertEqual(suffix["failure_reason"], "outside_requested_prefix")
        self.assertEqual(suffix["rgb_state"], "unmeasured")

    def test_physical_backend_exception_keeps_remaining_rows_terminal_without_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, read_calls, mask_calls = self._run_stubbed_physical_pose(
                Path(temporary), limit=2, failed_stage="refine",
            )
        self.assertEqual(read_calls, [1])
        self.assertEqual(mask_calls, ["1.mask"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["stage_completed"])
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(len(result["frames"]), 2)
        failed, remaining = result["frames"]
        self.assertTrue(failed["terminal"])
        self.assertTrue(failed["unmeasured_failure"])
        self.assertIsNone(failed["cameraFromObject"])
        self.assertEqual(remaining["failure_reason"], "motion_cleanup_unconfirmed")
        self.assertEqual(remaining["rgb_state"], "unmeasured")
        self.assertEqual(remaining["mask_measurement_state"], "measured")
        self.assertEqual(remaining["mask_path"], "2.mask")

    def test_quality_time_hash_is_emitted_and_mutation_rejects_prepared_replay(self):
        from . import quality_assets, quality_mug_chronological_capture as capture_module

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            (bundle / "input.json").write_text("{}", encoding="utf-8")
            emitted = quality_assets.inference_provenance(bundle)
            quality_time_path = Path(__file__).with_name("quality_time.py")
            expected_digest = quality_assets.digest(quality_time_path)
            self.assertEqual(emitted["adapter_sha256"]["quality_time"], expected_digest)

            bound = json.loads(json.dumps(emitted))
            bound["foundpose_bank_sha256"] = "B" * 64
            bound["adapter_sha256"]["quality_time"] = "0" * 64
            capture = object.__new__(capture_module.MugChronologicalCapture)
            capture.source_root = Path(temporary)
            capture.manifest = {"bindings": {
                "bundle_root": str(bundle.resolve()),
                "masks_root": str((Path(temporary) / "masks").resolve()),
                "current_source_sha256": {},
                "frozen_files": {"banks/mug-foundpose.pt": "B" * 64},
                "current_inference_provenance": bound,
            }}
            with mock.patch.object(capture_module, "_verify_manifest_source_freeze"), \
                    mock.patch.object(capture_module, "_safe_relative_path", return_value=Path("synthetic-bank")), \
                    mock.patch.object(capture_module, "digest", return_value="B" * 64):
                with self.assertRaisesRegex(ValueError, "inference provenance differs"):
                    capture._validate_runtime_bindings(
                        bundle, Path(temporary) / "masks",
                    )


if __name__ == "__main__":
    unittest.main()
