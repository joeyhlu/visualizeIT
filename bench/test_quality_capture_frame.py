"""Independent capture Frame/public-pose gates; synthetic arrays only."""
import hashlib
import unittest

import numpy as np

from .quality_contract import Frame, PoseCandidate, validate
from .quality_time import LEGACY, PHYSICAL


K = np.array([[360., 0., 359.5], [0., 360., 359.5], [0., 0., 1.]])


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def binding(rgb, valid):
    valid_hash = hashlib.sha256(b'sampling_valid\0[720,720]\0|u1\0' +
                                valid.astype(np.uint8).tobytes()).hexdigest().upper()
    return dict(capture_table_sha256='A' * 64, capture_row_sha256='B' * 64,
                rgb_pixel_sha256=digest(rgb), sampling_valid_sha256=valid_hash,
                warp_sha256='C' * 64, calibration_sha256='D' * 64)


def capture_frame(rgb=None, valid=None, **overrides):
    rgb = np.zeros((720, 720, 3), dtype=np.uint8) if rgb is None else rgb
    valid = np.ones((720, 720), dtype=bool) if valid is None else valid
    fields = dict(frame_id=0, rgb=rgb, intrinsics=K.copy(), timestamp_s=1.,
                  clock_mode=PHYSICAL, sampling_valid=valid, capture_binding=binding(rgb, valid))
    fields.update(overrides)
    return Frame(**fields)


def candidate(uv=None):
    if uv is None:
        xx, yy = np.meshgrid(np.arange(120., 601., 80.), np.arange(120., 601., 80.))
        uv = np.column_stack((xx.ravel(), yy.ravel()))
    rays = np.column_stack(((uv[:, 0] - 359.5) / 360,
                            (uv[:, 1] - 359.5) / 360, np.ones(len(uv))))
    return PoseCandidate(np.eye(4), rays, uv.copy(), np.ones(len(uv)))


class PoisonCandidate:
    def __getattr__(self, name):
        raise AssertionError('Candidate touched before capture-frame validation: ' + name)


class CaptureFrameTests(unittest.TestCase):
    def test_legacy_positional_and_physical_without_capture_fields_still_work(self):
        rgb = np.zeros((80, 120, 3), dtype=np.uint8)
        k = np.array([[110., 0., 44.], [0., 90., 37.], [0., 0., 1.]])
        old = Frame(2, rgb, k)
        self.assertEqual(old.clock_mode, LEGACY)
        self.assertIs(old.rgb, rgb)
        physical = Frame(2, rgb, k, 12.25, PHYSICAL)
        self.assertEqual(physical.timestamp_s, 12.25)
        self.assertIs(physical.rgb, rgb)

    def test_capture_fields_are_both_or_neither(self):
        with self.assertRaises(ValueError):
            capture_frame(capture_binding=None)
        with self.assertRaises(ValueError):
            capture_frame(sampling_valid=None)

    def test_capture_requires_physical_clock_fixed_camera_and_bool_validity(self):
        for fields in ({'clock_mode': LEGACY, 'timestamp_s': None},
                       {'intrinsics': np.eye(3)},
                       {'sampling_valid': np.ones((720, 720), dtype=np.uint8)},
                       {'sampling_valid': np.ones((719, 720), dtype=bool)},
                       {'rgb': np.zeros((720, 719, 3), dtype=np.uint8)}):
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                capture_frame(**fields)

    def test_binding_requires_exact_keys_and_uppercase_digest_format(self):
        good = dict(capture_frame().capture_binding)
        bads = [dict(good, extra_hash='E' * 64), {k: v for k, v in good.items() if k != 'warp_sha256'},
                dict(good, warp_sha256='c' * 64), dict(good, capture_row_sha256='F' * 63),
                dict(good, capture_table_sha256='G' * 64)]
        for item in bads:
            with self.subTest(keys=list(item)), self.assertRaises(ValueError):
                capture_frame(capture_binding=item)

    def test_rgb_and_framed_validity_content_must_match_binding(self):
        good = dict(capture_frame().capture_binding)
        for bad in (dict(good, rgb_pixel_sha256='A' * 64),
                    dict(good, sampling_valid_sha256=digest(np.ones((720, 720), dtype=np.uint8)))):
            with self.assertRaises(ValueError):
                capture_frame(capture_binding=bad)

    def test_constructor_detaches_arrays_and_binding_from_caller(self):
        rgb = np.zeros((720, 720, 3), dtype=np.uint8)
        valid = np.ones((720, 720), dtype=bool)
        k, hashes = K.copy(), binding(rgb, valid)
        frame = capture_frame(rgb, valid, intrinsics=k, capture_binding=hashes)
        rgb[:] = 255
        valid[:] = False
        k[0, 0] = 99
        hashes['warp_sha256'] = 'F' * 64
        self.assertFalse(frame.rgb.any())
        self.assertTrue(frame.sampling_valid.all())
        np.testing.assert_array_equal(frame.intrinsics, K)
        self.assertEqual(frame.capture_binding['warp_sha256'], 'C' * 64)
        with self.assertRaises(TypeError):
            frame.capture_binding['warp_sha256'] = 'F' * 64

    def test_array_readonly_cannot_be_reenabled_on_mutable_owner(self):
        frame = capture_frame()
        for array in (frame.rgb, frame.sampling_valid, frame.intrinsics):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_all_false_support_is_valid_but_not_pose_evidence(self):
        frame = capture_frame(valid=np.zeros((720, 720), dtype=bool))
        ok, reason, stats = validate(candidate(), frame, np.ones((720, 720), dtype=bool))
        self.assertFalse(ok)
        self.assertEqual(reason, 'insufficient_visible_correspondences')
        self.assertEqual(stats['correspondences'], 0)

    def test_public_validation_checks_frame_before_candidate(self):
        frame = capture_frame()
        object.__setattr__(frame, 'sampling_valid', None)
        with self.assertRaises(ValueError):
            validate(PoisonCandidate(), frame, np.ones((720, 720), dtype=bool))

    def test_public_validation_recomputes_current_array_digests(self):
        for key, replacement in (('rgb', np.ones((720, 720, 3), dtype=np.uint8)),
                                 ('sampling_valid', np.zeros((720, 720), dtype=bool))):
            frame = capture_frame()
            object.__setattr__(frame, key, replacement)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(PoisonCandidate(), frame, np.ones((720, 720), dtype=bool))

    def test_public_validation_checks_immutable_record_even_with_equal_values(self):
        for field in ('rgb', 'sampling_valid', 'intrinsics', 'capture_binding'):
            frame = capture_frame()
            old = getattr(frame, field)
            replacement = dict(old) if field == 'capture_binding' else old.copy()
            object.__setattr__(frame, field, replacement)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(PoisonCandidate(), frame, np.ones((720, 720), dtype=bool))
        frame = capture_frame()
        replacement = frame.rgb.copy()
        replacement.setflags(write=False)  # A mutable owner can re-enable this flag.
        object.__setattr__(frame, 'rgb', replacement)
        with self.assertRaises(ValueError):
            validate(PoisonCandidate(), frame, np.ones((720, 720), dtype=bool))

    def test_public_validation_rechecks_current_physical_frame_key(self):
        for field, value in (('timestamp_s', np.nan), ('timestamp_s', True),
                             ('frame_id', -1), ('frame_id', True)):
            frame = capture_frame()
            object.__setattr__(frame, field, value)
            with self.subTest(field=field, value=str(value)), self.assertRaises(ValueError):
                validate(PoisonCandidate(), frame, np.ones((720, 720), dtype=bool))

    def test_public_validation_rejects_nonbinary_mask(self):
        frame = capture_frame()
        for mask in (np.full((720, 720), 7, dtype=np.uint8),
                     np.ones((720, 720), dtype=float), np.ones((719, 720), dtype=bool)):
            with self.subTest(dtype=str(mask.dtype)), self.assertRaises(ValueError):
                validate(candidate(), frame, mask)

    def test_complete_four_neighbor_sampling_precedes_pose_statistics(self):
        valid = np.ones((720, 720), dtype=bool)
        valid[121, 121] = False  # candidate at120,120 still has true own pixel.
        frame = capture_frame(valid=valid)
        ok, reason, stats = validate(candidate(), frame, np.ones_like(valid))
        self.assertTrue(ok, reason)
        self.assertEqual(stats['correspondences'], 48)
        self.assertEqual(stats['inliers'], 48)

    def test_complete_four_neighbor_foreground_precedes_pose_statistics(self):
        frame = capture_frame()
        mask = np.ones((720, 720), dtype=bool)
        mask[121, 121] = False
        before = mask.copy()
        ok, reason, stats = validate(candidate(), frame, mask)
        self.assertTrue(ok, reason)
        self.assertEqual(stats['correspondences'], 48)
        np.testing.assert_array_equal(mask, before)

    def test_nonfinite_and_outside_correspondences_cannot_be_clipped_into_support(self):
        frame = capture_frame()
        c = candidate()
        c.pixels_image[:4] = [[np.nan, 120], [120, np.inf], [-.01, 120], [719, 120]]
        ok, reason, stats = validate(c, frame, np.ones((720, 720), dtype=bool))
        self.assertTrue(ok, reason)
        self.assertEqual(stats['correspondences'], 45)

    def test_capture_does_not_weaken_minimum_correspondences(self):
        frame = capture_frame()
        uv = candidate().pixels_image[:23]
        ok, reason, stats = validate(candidate(uv), frame, np.ones((720, 720), dtype=bool))
        self.assertFalse(ok)
        self.assertEqual(reason, 'insufficient_visible_correspondences')
        self.assertEqual(stats['correspondences'], 23)


if __name__ == '__main__':
    unittest.main()
