"""Manufactured opening-seed isolation fixtures; no real pose labels are read."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.cache/video-workbench/scratch'
IDENTITY_R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
EXPECTED_SEED_M = np.array([[1, 0, 0, .1], [0, 1, 0, .2], [0, 0, 1, 1.0], [0, 0, 0, 1]], dtype=float)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def manufactured_scene(*, selected=None, duplicate_frame=False, future_number='1e400'):
    """Future numbers are lexically valid JSON but overflow if converted to float.

    A conforming extractor may validate their syntax without materializing their
    numeric values. Only the selected frame's translation is a millimetre seed.
    """
    selected = selected if selected is not None else [
        {'obj_id': 3, 'cam_R_m2c': IDENTITY_R, 'cam_t_m2c': [100, 200, 1000]}]
    frame = json.dumps(selected, separators=(',', ':'), allow_nan=False)
    prefix = '"0":[{"obj_id":1,"cam_R_m2c":[1,0,0,0,1,0,0,0,1],"cam_t_m2c":[9000,0,100]}],'
    chosen = '"9":' + frame
    future = ',"10":[{"obj_id":99,"NEVER_MATERIALIZE_FUTURE":' + future_number + '}]'
    duplicate = ',"9":' + frame if duplicate_frame else ''
    return ('{' + prefix + chosen + future + duplicate + '}').encode('ascii')


class HBOpeningSeedTests(unittest.TestCase):
    def setUp(self):
        from tools import workbench_hb_opening_seed as seed
        self.seed = seed

    def extract(self, raw, chunk=7):
        return self.seed.extract_frame_value(raw[i:i + chunk] for i in range(0, len(raw), chunk))

    def test_only_frame_nine_materializes_after_streaming_at_arbitrary_boundaries(self):
        for chunk in (1, 2, 7, 64 << 10):
            with self.subTest(chunk=chunk):
                raw = self.extract(manufactured_scene(), chunk)
                self.assertNotIn(b'NEVER_MATERIALIZE_FUTURE', raw)
                self.assertNotIn(b'9000', raw)
                pose = self.seed.select_pose(json.loads(raw), 3, 0)
                np.testing.assert_array_equal(pose, EXPECTED_SEED_M)

    def test_skipped_future_number_is_never_converted_to_a_float_or_full_document(self):
        raw = manufactured_scene()
        with mock.patch('builtins.float', side_effect=AssertionError('numeric materialization in lexical scanner')), \
             mock.patch.object(json, 'loads', side_effect=AssertionError('full JSON parsing in lexical scanner')), \
             mock.patch.object(json, 'load', side_effect=AssertionError('full JSON loading in lexical scanner')):
            selected = self.extract(raw)
        self.assertEqual(json.loads(selected)[0]['cam_t_m2c'], [100, 200, 1000])

    def test_duplicate_top_level_selected_key_rejected_even_after_first_capture(self):
        with self.assertRaisesRegex(self.seed.mi.InputsError, 'duplicate'):
            self.extract(manufactured_scene(duplicate_frame=True))

    def test_duplicate_selected_pose_keys_rejected_by_strict_selected_parser(self):
        raw = manufactured_scene().replace(b'"obj_id":3', b'"obj_id":3,"obj_id":3', 1)
        selected = self.extract(raw)
        with self.assertRaisesRegex(self.seed.mi.InputsError, 'duplicate'):
            self.seed.mi.strict_json(selected, 'selected synthetic frame')

    def test_missing_selected_frame_and_truncation_fail(self):
        for raw in (b'{"0":[]}', manufactured_scene()[:-1]):
            with self.subTest(raw=raw[-20:]), self.assertRaises(self.seed.mi.InputsError):
                self.extract(raw)

    def test_invalid_literal_escape_or_bracket_does_not_yield_a_seed(self):
        for raw in (b'{"9":[],"10":[NaN]}', b'{"\\u0039":[]}', b'{"9":[}]}'):
            with self.subTest(raw=raw), self.assertRaises(self.seed.mi.InputsError):
                self.extract(raw)

    def test_capture_depth_and_selected_size_limits_fail_closed(self):
        with mock.patch.object(self.seed, 'MAX_DEPTH', 3):
            with self.assertRaises(self.seed.mi.InputsError):
                self.extract(b'{"9":[[[[1]]]]}')
        scanner = self.seed.TopLevelFrameExtractor('9', max_capture=2)
        with self.assertRaises(self.seed.mi.InputsError):
            scanner.feed(b'{"9":[1,2]}')

    def test_selected_instance_is_explicit_and_never_first_matching_by_default(self):
        first = {'obj_id': 1, 'cam_R_m2c': IDENTITY_R, 'cam_t_m2c': [9000, 0, 100]}
        chosen = {'obj_id': 3, 'cam_R_m2c': IDENTITY_R, 'cam_t_m2c': [100, 200, 1000]}
        np.testing.assert_array_equal(self.seed.select_pose([first, chosen], 3, 1), EXPECTED_SEED_M)
        for index in (0, 2):
            with self.subTest(index=index), self.assertRaises(self.seed.mi.InputsError):
                self.seed.select_pose([first, chosen], 3, index)

    def test_mm_seed_conversion_does_not_mutate_source_or_scale_rotation(self):
        frame = json.loads(self.extract(manufactured_scene()))
        before = json.dumps(frame, sort_keys=True)
        first = self.seed.select_pose(frame, 3, 0)
        second = self.seed.select_pose(frame, 3, 0)
        np.testing.assert_array_equal(first, EXPECTED_SEED_M)
        np.testing.assert_array_equal(second, EXPECTED_SEED_M)
        self.assertEqual(json.dumps(frame, sort_keys=True), before)

    def test_improper_rotation_nonfinite_and_boolean_pose_values_are_refused(self):
        variants = [('cam_R_m2c', [-1, 0, 0, 0, 1, 0, 0, 0, 1]),
                    ('cam_R_m2c', [2, 0, 0, 0, 1, 0, 0, 0, 1]),
                    ('cam_t_m2c', [True, 200, 1000]),
                    ('cam_t_m2c', [float('inf'), 200, 1000])]
        for key, value in variants:
            entry = {'obj_id': 3, 'cam_R_m2c': IDENTITY_R, 'cam_t_m2c': [100, 200, 1000], key: value}
            with self.subTest(key=key, value=value), self.assertRaises(self.seed.mi.InputsError):
                self.seed.select_pose([entry], 3, 0)

    def test_scored_frame_argument_is_refused_before_opening_any_source(self):
        ctx = self.seed.mi.Context()
        with mock.patch.object(self.seed.mi, 'contained_input', side_effect=AssertionError('path touched')):
            with self.assertRaises(self.seed.mi.UsageError):
                self.seed.extract_seed(ctx, scene_gt='never-opened/scene_gt.json', expected_scene_gt_sha256='a' * 64,
                    frame_index=10, object_id=3, instance_index=0, selection_review='never-opened/review.json',
                    expected_selection_review_sha256='b' * 64, allowlist='never-opened/allowlist.json',
                    expected_allowlist_sha256='c' * 64, out='never-created')
