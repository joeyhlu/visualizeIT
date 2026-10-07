"""CPU-only tests for the reviewed bottle patch-calibration viewer summary."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bench import quality_player as player


BASE_SOURCE_SHA256 = 'D57A4055AE75A27F2BF8BAE8B364D182FAB231E0B3A92CB0DA636B96B13E8047'
PRE_HELP_SOURCE_SHA256 = '1F16BBEF5FCC0B83B6D38E93CC16079995386E4522A4E30646B1FC58216C8209'
OBJECT_SELECTION_HELP_BLOCK_SHA256 = '387285F6202A6E34CED34FD1EA2A0719D54C0236586CA3AB6C84ED90AC6DD6C5'
R8_CAPACITY_CONSTANTS_SHA256 = 'CDBB863300EDA393E4541576D8842265092BE0AFFF7532F81071D5A40290B084'
R8_CAPACITY_HELPERS_SHA256 = '7DE92124195512D630E2C916B2DFF698B22D0C240F56C58D42C1B201E7FFEC92'
SOURCE = Path(player.__file__).resolve()


def _real_records():
    return (
        json.loads(player.BOTTLE_PATCH_CALIBRATION_REPORT.read_text(encoding='utf-8')),
        json.loads(player.BOTTLE_PATCH_CALIBRATION_TERMINAL.read_text(encoding='utf-8')),
        json.loads(player.BOTTLE_PATCH_CALIBRATION_REPEAT.read_text(encoding='utf-8')),
    )


def _remove_object_selection_help(source_bytes):
    """Remove only the approved help disclosure and restore the prior replay line."""
    start_marker = b'_OFFLINE_REPLAY_NOTICE = ('
    build_marker = b'def build():'
    if source_bytes.count(start_marker) != 1 or source_bytes.count(build_marker) != 1:
        raise AssertionError('object-selection help block anchors are missing or duplicated')
    start = source_bytes.rfind(b'\n', 0, source_bytes.index(start_marker)) + 1
    end = source_bytes.index(build_marker, start)
    help_block = source_bytes[start:end]
    if hashlib.sha256(help_block).hexdigest().upper() != OBJECT_SELECTION_HELP_BLOCK_SHA256:
        raise AssertionError('object-selection help block differs from its exact reviewed bytes')
    if (help_block.count(b'def _object_selection_help_html():') != 1 or
            help_block.count(b'def _insert_object_selection_help(html):') != 1):
        raise AssertionError('object-selection help functions changed or are duplicated')
    restored = source_bytes[:start] + source_bytes[end:]

    injection = b'    html = _insert_object_selection_help(html)\n'
    if restored.count(injection) != 1:
        raise AssertionError('object-selection help build insertion is missing or duplicated')
    restored = restored.replace(injection, b'', 1)

    current_replay_line = (
        b"    html = replace_once(html, '<h1>Real moving objects \xc2\xb7 60 FPS recordings</h1>', "
        b"'<h1>Model tracking \xc2\xb7 same footage, four views</h1>'+_OFFLINE_REPLAY_NOTICE)\n")
    previous_replay_line = (
        b"    html = replace_once(html, '<h1>Real moving objects \xc2\xb7 60 FPS recordings</h1>', "
        b"'<h1>Model tracking \xc2\xb7 same footage, four views</h1><p id=\"replayNotice\" "
        b"role=\"note\"><strong>Offline replay.</strong> Masks and poses are precomputed. FPS below "
        b"measures browser playback, not live tracking.</p>')\n")
    if restored.count(current_replay_line) != 1:
        raise AssertionError('object-selection help replay-note insertion is missing or duplicated')
    return restored.replace(current_replay_line, previous_replay_line, 1)


def _remove_r8_capacity_card(source_bytes):
    """Remove only the exact R8 capacity-card additions for older source proofs."""
    constants_start = b'# Hash-bound R8 actual-source and terminal full-screen receipts.'
    constants_end = b'# Hash-bound closed calibration evidence pins.'
    helpers_start = b'def _bottle_r8_capacity_summary_from_records('
    helpers_end = b'# Fail-closed presentation helpers for closed calibration evidence.'
    for marker in (constants_start, constants_end, helpers_start, helpers_end):
        if source_bytes.count(marker) != 1:
            raise AssertionError('R8 capacity-card block anchors are missing or duplicated')

    constants_begin = source_bytes.rfind(b'\n', 0, source_bytes.index(constants_start)) + 1
    constants_finish = source_bytes.rfind(b'\n', 0, source_bytes.index(constants_end)) + 1
    constants_block = source_bytes[constants_begin:constants_finish]
    if hashlib.sha256(constants_block).hexdigest().upper() != R8_CAPACITY_CONSTANTS_SHA256:
        raise AssertionError('R8 capacity constants differ from their exact reviewed bytes')
    source_bytes = source_bytes[:constants_begin] + source_bytes[constants_finish:]

    helpers_begin = source_bytes.rfind(b'\n', 0, source_bytes.index(helpers_start)) + 1
    helpers_finish = source_bytes.rfind(b'\n', 0, source_bytes.index(helpers_end)) + 1
    helpers_block = source_bytes[helpers_begin:helpers_finish]
    if hashlib.sha256(helpers_block).hexdigest().upper() != R8_CAPACITY_HELPERS_SHA256:
        raise AssertionError('R8 capacity helpers differ from their exact reviewed bytes')
    if source_bytes[helpers_begin - 1:helpers_begin] != b'\n':
        raise AssertionError('R8 capacity helper insertion separator is missing')
    source_bytes = source_bytes[:helpers_begin - 1] + source_bytes[helpers_finish:]

    exact_removals = (
        b'    bottle_r8_capacity_summary = _load_bottle_r8_capacity_summary()\n',
        b'    bottle_r8_capacity_note = _bottle_r8_capacity_summary_html(bottle_r8_capacity_summary)\n',
    )
    for line in exact_removals:
        if source_bytes.count(line) != 1:
            raise AssertionError('R8 capacity-card build binding is missing or duplicated')
        source_bytes = source_bytes.replace(line, b'', 1)

    table_insertion = (
        b'+bottle_pose_note+bottle_r8_capacity_note+'
        b'_bottle_patch_calibration_summary_html('
    )
    table_without_r8 = b'+bottle_pose_note+_bottle_patch_calibration_summary_html('
    if source_bytes.count(table_insertion) != 1:
        raise AssertionError('R8 capacity-card table insertion is missing or duplicated')
    return source_bytes.replace(table_insertion, table_without_r8, 1)


def _reconstruct_pinned_base(source_bytes):
    """Inverse approved viewer additions and return the exact D57 base bytes."""
    source_bytes = _remove_r8_capacity_card(source_bytes)
    source_bytes = _remove_object_selection_help(source_bytes)
    build_addition = (
        b'+bottle_pose_note+_bottle_patch_calibration_summary_html('
        b'_load_bottle_patch_calibration_summary())+mug_pnp_note+')
    build_base = b'+bottle_pose_note+mug_pnp_note+'
    if source_bytes.count(build_addition) != 1:
        raise AssertionError('integrated summary call is missing or duplicated')
    restored = source_bytes.replace(build_addition, build_base, 1)

    constants_line = (
        b"BOTTLE_POSE_ABLATION_TEST_SHA256 = "
        b"'4FAF00469E154FE56E9352CCA375674A848D6E21104340D52CD51D7AC685CC91'\n")
    if restored.count(constants_line) != 1:
        raise AssertionError('pinned base constants anchor is missing or duplicated')
    constants_end = restored.index(constants_line) + len(constants_line)
    require_def = b'def _require(condition, message):'
    require_start = restored.index(require_def, constants_end)
    restored = (restored[:constants_end] + b'\n\ndef _require(condition, message):'
                + restored[require_start + len(require_def):])

    r5_close = b"+ figure + '</details>')"
    if restored.count(r5_close) != 1:
        raise AssertionError('pinned R5 insertion anchor is missing or duplicated')
    r5_line_end = restored.index(b'\n', restored.index(r5_close)) + 1
    contours_def = b'def contours(mask, simplify=0):'
    contours_start = restored.index(contours_def, r5_line_end)
    restored = (restored[:r5_line_end] + b'\n\ndef contours(mask, simplify=0):'
                + restored[contours_start + len(contours_def):])
    return restored


class BottlePatchCalibrationSummaryTests(unittest.TestCase):
    def test_actual_bench_module_loads_exact_closed_record_and_renders_caption(self):
        self.assertEqual(SOURCE.name, 'quality_player.py')
        summary = player._load_bottle_patch_calibration_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(summary['condition_count'], 12)
        self.assertEqual(summary['measured_count'], 12)
        self.assertEqual(summary['candidate_not_run_count'], 12)
        self.assertEqual(summary['candidate_fit_calls'], 0)
        self.assertEqual(summary['self_control_pass_count'], 6)
        self.assertEqual(summary['positive_pass_count'], 2)
        self.assertEqual(summary['negative_unobservable_count'], 3)
        self.assertEqual(summary['negative_original_visible_counts'], [8, 0, 0])
        self.assertTrue(summary['prior_source_binding_failure_preserved'])
        self.assertEqual(
            [(row['frame_id'], row['precision_percent'], row['coverage_percent'], row['passed'])
             for row in summary['positive_results']],
            [(10, 99.22, 42.43, False), (50, 98.41, 65.44, True),
             (100, 99.19, 69.73, True)])

        markup = player._bottle_patch_calibration_summary_html(summary)
        self.assertIn('<details class="bottlePatchCalibrationSummary"', markup)
        self.assertIn('Rejected calibration — no new video poses', markup)
        self.assertIn('all 12/12 frozen conditions were measured', markup.lower())
        self.assertIn('12/12 remained not run; 0 fit calls', markup)
        self.assertIn('99.22%', markup)
        self.assertIn('42.43%', markup)
        self.assertIn('98.41%', markup)
        self.assertIn('65.44%', markup)
        self.assertIn('99.19%', markup)
        self.assertIn('69.73%', markup)
        self.assertIn('correct among verified confident true-visible points', markup)
        self.assertIn('verified coverage among source-observable true-visible samples', markup)
        self.assertIn('9.60%', markup)
        self.assertIn('12% minimum', markup)
        self.assertIn('V=8/0/0', markup)
        self.assertIn('zero verified true-visible matches', markup)
        self.assertIn('do not show false-pose rejection', markup)
        self.assertIn('does not establish real-video mapping', markup)
        self.assertIn('At calibration time, independent labels were 0/120 reviewed', markup)
        self.assertIn('earlier dense-correspondence and pose prerequisites remain failed', markup)
        self.assertIn('phone predictions were not changed', markup)
        self.assertNotIn('original dense R4/R5 gates remain failed', markup)
        for private_detail in (
                '.cache', 'calibration.json', 'terminal.json', '857A52DF',
                '214ED8C0', 'AB9DCA72', '81614CB6', '761C9A44'):
            self.assertNotIn(private_detail, markup)

    def test_missing_report_and_wrong_report_hash_hide_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / 'missing.json'
            self.assertIsNone(player._load_bottle_patch_calibration_summary(
                report_path=missing,
                terminal_path=player.BOTTLE_PATCH_CALIBRATION_TERMINAL,
                repeat_path=player.BOTTLE_PATCH_CALIBRATION_REPEAT))
            changed = Path(temporary) / 'changed.json'
            changed.write_bytes(player.BOTTLE_PATCH_CALIBRATION_REPORT.read_bytes() + b' ')
            self.assertIsNone(player._load_bottle_patch_calibration_summary(
                report_path=changed,
                terminal_path=player.BOTTLE_PATCH_CALIBRATION_TERMINAL,
                repeat_path=player.BOTTLE_PATCH_CALIBRATION_REPEAT))

    def test_partial_records_hide_summary(self):
        report, terminal, repeat = _real_records()
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            {'schema_version': 1}, terminal, repeat))
        partial = copy.deepcopy(report)
        partial['conditions'].pop()
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            partial, terminal, repeat))

    def test_wrong_parent_receipt_hides_summary(self):
        report, terminal, repeat = _real_records()
        terminal['exit_code'] = 1
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            report, terminal, repeat))

    def test_wrong_calibration_gate_hides_summary(self):
        report, terminal, repeat = _real_records()
        report['calibration_gate']['passed'] = True
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            report, terminal, repeat))

    def test_fitted_candidate_hides_summary(self):
        report, terminal, repeat = _real_records()
        report['conditions'][0]['candidate']['state'] = 'fit_complete'
        report['candidate_fit_calls'] = 1
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            report, terminal, repeat))

    def test_wrong_source_pin_or_repeat_receipt_hides_summary(self):
        report, terminal, repeat = _real_records()
        report['source_pins']['bench/quality_bottle_patch_pose_calibration.py'] = '0' * 64
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            report, terminal, repeat))
        report, terminal, repeat = _real_records()
        repeat['prior_source_binding_failure_preserved'] = False
        self.assertIsNone(player._bottle_patch_calibration_summary_from_records(
            report, terminal, repeat))

    def test_inverse_additions_reconstruct_exact_d57_base(self):
        reconstructed = _reconstruct_pinned_base(SOURCE.read_bytes())
        self.assertEqual(hashlib.sha256(reconstructed).hexdigest().upper(),
                         BASE_SOURCE_SHA256)

    def test_object_help_is_the_only_change_from_pinned_pre_help_source(self):
        pre_help = _remove_object_selection_help(
            _remove_r8_capacity_card(SOURCE.read_bytes()))
        self.assertEqual(hashlib.sha256(pre_help).hexdigest().upper(),
                         PRE_HELP_SOURCE_SHA256)

    def test_inverse_rejects_unrecognized_code_inside_help_block(self):
        source = SOURCE.read_bytes()
        build_marker = b'def build():'
        self.assertEqual(source.count(build_marker), 1)
        tampered = source.replace(
            build_marker,
            b'def unexpected_helper():\n    return None\n\n' + build_marker,
            1,
        )
        with self.assertRaisesRegex(AssertionError, 'exact reviewed bytes'):
            _remove_object_selection_help(tampered)

    def test_product_source_contains_no_private_task_markers(self):
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn('_bottle_patch_calibration_summary_html(', source)
        self.assertNotIn('BEGIN F8 PRIVATE', source)
        self.assertNotIn('END F8 PRIVATE', source)


if __name__ == '__main__':
    unittest.main()
