"""Codex independent manufactured JSON/K controls; no dataset or pose labels."""
import binascii
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import workbench_hb_camera_k as ck

K = [500.0, 0, 319.5, 0, 501.0, 239.5, 0, 0, 1]


def document(entries=None):
    return json.dumps(entries if entries is not None else {
        '0': {'cam_K': K, 'cam_R_w2c': [9, 8, 7], 'cam_t_w2c': [6, 5, 4]},
        '1': {'cam_K': K}, '120': {'cam_K': [1, 2]}}).encode()


class CameraScannerTests(unittest.TestCase):
    def scan(self, raw, requested=(0, 1), chunk=7):
        return ck.scan_camera_json((raw[i:i + chunk] for i in range(0, len(raw), chunk)), requested)

    def test_only_requested_intrinsics_are_captured_as_lexemes(self):
        result = self.scan(document())
        self.assertEqual(set(result['captured']), {0, 1})
        self.assertEqual(result['stats']['numbers_captured'], 18)
        self.assertEqual(result['stats']['numbers_skipped_unconverted'], 8)
        # Manufactured zero-skew pinhole camera above; no origin/scale conversion.
        checked = ck.validate_k_window(result['captured'], (0, 1), (640, 480))
        self.assertEqual(checked['values'], K)
        self.assertEqual(checked['cx'], 319.5)
        self.assertEqual(checked['cy'], 239.5)

    def test_every_byte_boundary_matches_one_chunk(self):
        raw = document({'0': {'cam_K': K, 'comment': 'snowman ☃ and \\"'}, '1': {'cam_K': K}})
        expected = self.scan(raw, chunk=len(raw))
        for boundary in range(1, len(raw)):
            with self.subTest(boundary=boundary):
                self.assertEqual(ck.scan_camera_json([raw[:boundary], raw[boundary:]], (0, 1)), expected)

    def test_skipped_overflow_and_long_integer_are_never_converted(self):
        raw = b'{"0":{"cam_K":' + json.dumps(K).encode() + b',"pose":[1e400,-1e-99999,' + b'9' * 10000 + b']}}'
        with mock.patch('builtins.float', side_effect=AssertionError('no numeric conversion during scan')):
            result = self.scan(raw, requested=(0,), chunk=1)
        self.assertEqual(result['stats']['numbers_skipped_unconverted'], 3)

    def test_out_of_window_values_are_skipped_without_k_validation(self):
        raw = document({'0': {'cam_K': K}, '120': {'cam_K': 'not a camera', 'pose': [1, 2]}})
        self.assertEqual(set(self.scan(raw, (0,))['captured']), {0})

    def test_duplicate_decoded_keys_at_all_levels_refused(self):
        camera = json.dumps(K).encode()
        cases = [b'{"0":{"cam_K":' + camera + b'},"0":{}}',
            b'{"0":{"cam_K":' + camera + b',"cam_\\u004b":' + camera + b'}}',
            b'{"0":{"cam_K":' + camera + b',"pose":{"x":1,"x":2}}}']
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ck.CameraError):
                self.scan(raw, (0,))

    def test_missing_k_or_requested_frame_refused(self):
        for obj in ({'0': {}}, {'1': {'cam_K': K}}):
            with self.subTest(obj=obj), self.assertRaises(ck.CameraError):
                self.scan(document(obj), (0,))

    def test_non_bmp_duplicate_key_spelling_cannot_bypass_validation(self):
        raw = ('{"0":{"cam_K":' + json.dumps(K) + ',"pose":{"😀":1,"\\ud83d\\ude00":2}}}').encode()
        for chunk in (1, 7, len(raw)):
            with self.subTest(chunk=chunk), self.assertRaises(ck.CameraError):
                self.scan(raw, (0,), chunk=chunk)

    def test_k_must_be_flat_nine_numeric_entries(self):
        for camera in (K[:-1], K + [1], [K], [True] + K[1:], [None] + K[1:], ['500'] + K[1:]):
            with self.subTest(camera=camera), self.assertRaises(ck.CameraError):
                self.scan(document({'0': {'cam_K': camera}}), (0,))

    def test_json_grammar_errors_are_refused_even_in_skipped_poses(self):
        prefix = b'{"0":{"cam_K":' + json.dumps(K).encode() + b',"pose":'
        for value in (b'[1,]', b'{"x":1,}', b'01', b'1.', b'1e', b'+1', b'NaN', b'Infinity', b'truefalse', b'"bad\\q"'):
            with self.subTest(value=value), self.assertRaises(ck.CameraError):
                self.scan(prefix + value + b'}}', (0,), chunk=1)

    def test_trailing_data_truncation_utf8_and_bom_are_refused(self):
        raw = document({'0': {'cam_K': K}})
        for invalid in (raw + b'{}', raw[:-1], b'\xef\xbb\xbf' + raw, raw[:-1] + b'\xff}'):
            with self.subTest(raw=invalid), self.assertRaises(ck.CameraError):
                self.scan(invalid, (0,), chunk=1)

    def test_noncanonical_frame_keys_refused(self):
        for key in ('00', '-1', '1.0', 'frame0'):
            with self.subTest(key=key), self.assertRaises(ck.CameraError):
                self.scan(document({key: {'cam_K': K}}), (0,))

    def test_depth_limit_is_enforced_on_skipped_data(self):
        raw = b'{"0":{"cam_K":' + json.dumps(K).encode() + b',"pose":' + b'[' * 100 + b'0' + b']' * 100 + b'}}'
        with self.assertRaises(ck.CameraError):
            self.scan(raw, (0,))

    def test_requested_boolean_cannot_be_hidden_by_integer_deduplication(self):
        for requested in ((False,), (0, False), (1, True)):
            with self.subTest(requested=requested), self.assertRaises(ck.mi.UsageError):
                ck.StructuralScanner(requested)

    def test_requested_indices_must_be_nonnegative_integers(self):
        for requested in ((), (-1,), ('0',), (0.0,)):
            with self.subTest(requested=requested), self.assertRaises(ck.mi.UsageError):
                ck.StructuralScanner(requested)

    def test_k_window_checks_original_requested_types_before_deduplication(self):
        captured = self.scan(document())['captured']
        for requested in ((0, False), (1, True), (0, 'x'), (-1,), (), (0.0,)):
            with self.subTest(requested=requested), self.assertRaises(ck.mi.UsageError):
                ck.validate_k_window(captured, requested, (640, 480))

    def test_k_window_rejects_nonfinite_changed_and_incompatible_cameras(self):
        captured = self.scan(document())['captured']
        for position, text in ((0, '0'), (4, '-1'), (1, '0.1'), (8, '2'), (2, '640'), (5, '480'), (0, '1e400')):
            bad = {i: list(values) for i, values in captured.items()}
            for values in bad.values():
                values[position] = text
            with self.subTest(position=position, text=text), self.assertRaises(ck.CameraError):
                ck.validate_k_window(bad, (0, 1), (640, 480))
        changed = {i: list(values) for i, values in captured.items()}
        changed[1][0] = '501'
        with self.assertRaises(ck.CameraError):
            ck.validate_k_window(changed, (0, 1), (640, 480))

    def test_equivalent_number_spelling_preserves_first_raw_lexemes(self):
        raw = b'{"0":{"cam_K":[500,0,319.5,0,501,239.5,0,0,1]},"1":{"cam_K":[5e2,0,319.5,0,501,239.5,0,0,1]}}'
        result = ck.validate_k_window(self.scan(raw)['captured'], (0, 1), (640, 480))
        self.assertEqual(result['values'], K)
        self.assertEqual(result['lexemes'][0], '500')
        self.assertFalse(result['raw_lexemes_identical_across_window'])

    def test_two_pass_file_hash_size_and_crc_then_scan(self):
        raw = document()
        with tempfile.TemporaryDirectory(prefix='hb-k-unit-', dir='.cache') as directory:
            path = Path(directory) / 'manufactured-camera.json'
            path.write_bytes(raw)
            sealed = ck.hash_bounded(path, expected_bytes=len(raw))
            self.assertEqual(sealed['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(sealed['crc32'], binascii.crc32(raw) & 0xffffffff)
            result = ck.scan_camera_file(path, expected_bytes=len(raw), expected_sha256=sealed['sha256'], requested=(0, 1), chunk=1)
            self.assertEqual(result['captured'], self.scan(raw)['captured'])
            path.write_bytes(raw.replace(b'500', b'501'))
            with self.assertRaises(ck.CameraError):
                ck.scan_camera_file(path, expected_bytes=len(raw), expected_sha256=sealed['sha256'], requested=(0, 1))

    def test_hash_pass_refuses_size_or_cap_mismatch(self):
        with tempfile.TemporaryDirectory(prefix='hb-k-unit-', dir='.cache') as directory:
            path = Path(directory) / 'camera.json'
            path.write_bytes(b'{}')
            for expected, cap in ((3, 100), (2, 1), (True, 100), (0, 100)):
                with self.subTest(expected=expected, cap=cap), self.assertRaises(ck.CameraError):
                    ck.hash_bounded(path, expected_bytes=expected, cap=cap)


if __name__ == '__main__':
    unittest.main()
