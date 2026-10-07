"""Fail-closed presentation tests for the reviewed R5 bottle pose ablation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import quality_player as player


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class BottlePoseSelectionSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report_bytes = player.BOTTLE_POSE_ABLATION_REPORT.read_bytes()
        cls.terminal_bytes = player.BOTTLE_POSE_ABLATION_TERMINAL.read_bytes()
        cls.report = json.loads(cls.report_bytes.decode('utf-8'))
        cls.terminal = json.loads(cls.terminal_bytes.decode('utf-8'))

    def _summary_from_records(self, report=None, terminal=None):
        return player._bottle_pose_summary_from_records(
            copy.deepcopy(self.report if report is None else report),
            copy.deepcopy(self.terminal if terminal is None else terminal),
        )

    def _assert_rejected(self, mutate_report=None, mutate_terminal=None):
        report = copy.deepcopy(self.report)
        terminal = copy.deepcopy(self.terminal)
        if mutate_report:
            mutate_report(report)
        if mutate_terminal:
            mutate_terminal(terminal)
        self.assertIsNone(player._bottle_pose_summary_from_records(report, terminal))

    def test_exact_closed_record_renders_truthful_rejected_summary(self):
        self.assertEqual(hashlib.sha256(self.report_bytes).hexdigest().upper(),
                         player.BOTTLE_POSE_ABLATION_REPORT_SHA256)
        self.assertEqual(hashlib.sha256(self.terminal_bytes).hexdigest().upper(),
                         player.BOTTLE_POSE_ABLATION_TERMINAL_SHA256)
        summary = player._load_bottle_pose_summary()
        self.assertIsNotNone(summary)
        figure_data_uri = summary['figure_data_uri']
        markup_with_figure = player._bottle_pose_summary_html(summary)
        summary['figure_data_uri'] = None
        markup = player._bottle_pose_summary_html(summary)

        self.assertIn('<details class="bottlePoseSummary"', markup)
        self.assertNotIn('<details open', markup)
        self.assertIn('12/12 conditions and 24/24 fit arms', markup)
        self.assertIn('27/27 immutable inputs and deterministic repeats matched', markup)
        self.assertIn('zero model, forward or rendering calls', markup)
        self.assertIn('cached synthetic query views and self-view controls', markup)
        self.assertIn('all-match arms accepted all three opposite-view poses', markup.lower())
        self.assertIn('one proposal unavailable below 24 correspondences', markup)
        self.assertIn('rejected two for localized/ambiguous spatial support', markup)
        self.assertIn('do not demonstrate corrected identity', markup)
        self.assertIn('Positive query 10:', markup)
        self.assertIn('2.394 → 3.104°', markup)
        self.assertIn('1.246 → 0.703% of model diagonal', markup)
        self.assertIn('1.797 → 1.672 px at 720p, report-only', markup)
        self.assertIn('Both arms were accepted in 6/6 self controls', markup)
        self.assertIn('failed the no-worse comparison in 6/6', markup)
        self.assertIn('3/6 controls failed the frozen 1° / 0.01D self bound in one or both arms', markup)
        self.assertIn('exploratory pose prerequisite failed', markup)
        self.assertIn('Runtime/default integration remains off', markup)
        self.assertIn('original dense identity and R4 eligibility gates remain failed', markup)
        self.assertIsNotNone(figure_data_uri)
        self.assertIn('data:image/png;base64,', markup_with_figure)
        self.assertIn('max-width:100%;height:auto', markup_with_figure)
        self.assertNotIn('<video', markup)

    def test_missing_optional_figure_keeps_summary_and_readable_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing_figure = Path(temporary)/'missing.png'
            summary = player._load_bottle_pose_summary(figure_path=missing_figure)
        self.assertIsNotNone(summary)
        self.assertIsNone(summary['figure_data_uri'])
        markup = player._bottle_pose_summary_html(summary)
        self.assertIn('Positive query 50:', markup)
        self.assertIn('Positive query 100:', markup)
        self.assertNotIn('<img ', markup)

    def test_optional_figure_is_skipped_when_wrong_or_over_two_mib(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'figure.png'
            path.write_bytes(b'not the pinned figure')
            wrong = player._optional_bottle_pose_figure_data_uri(path)
            self.assertIsNone(wrong)
            path.write_bytes(b'\x89PNG\r\n\x1a\n'+b'x'*player.BOTTLE_POSE_ABLATION_FIGURE_MAX_BYTES)
            too_large = player._optional_bottle_pose_figure_data_uri(path)
            self.assertIsNone(too_large)

    def test_missing_and_corrupt_closed_files_omit_the_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report_path = directory/'pose_ablation.json'
            terminal_path = directory/'terminal.json'
            report_path.write_bytes(self.report_bytes)
            self.assertIsNone(player._load_bottle_pose_summary(
                report_path=report_path, terminal_path=terminal_path, figure_path=None))

            terminal_path.write_bytes(self.terminal_bytes)
            bad_report = bytearray(self.report_bytes)
            bad_report[0] ^= 1
            report_path.write_bytes(bad_report)
            self.assertIsNone(player._load_bottle_pose_summary(
                report_path=report_path, terminal_path=terminal_path, figure_path=None))

            report_path.write_bytes(self.report_bytes)
            bad_terminal = bytearray(self.terminal_bytes)
            bad_terminal[0] ^= 1
            terminal_path.write_bytes(bad_terminal)
            self.assertIsNone(player._load_bottle_pose_summary(
                report_path=report_path, terminal_path=terminal_path, figure_path=None))

    def test_nonterminal_or_unsuccessful_terminal_receipt_is_rejected(self):
        self._assert_rejected(mutate_terminal=lambda item: item.update(state='running'))
        self._assert_rejected(mutate_terminal=lambda item: item.update(exit_code=1))

    def test_source_freeze_and_immutable_input_mismatches_are_rejected(self):
        self._assert_rejected(mutate_terminal=lambda item: item.update(source_freeze_matched=False))

        def changed_pin(item):
            item['source_pins_after']['bench/quality_bottle_pose_ablation.py'] = '0'*64

        self._assert_rejected(mutate_terminal=changed_pin)
        self._assert_rejected(mutate_report=lambda item: item['immutable_input_revalidation'].__setitem__(
            'changed_or_missing', 1))

        def changed_input(item):
            item['immutable_input_revalidation']['files'][0]['actual_sha256'] = '0'*64

        self._assert_rejected(mutate_report=changed_input)

    def test_partial_condition_or_arm_accounting_is_rejected(self):
        self._assert_rejected(mutate_report=lambda item: item.update(planned_conditions=11))
        self._assert_rejected(mutate_report=lambda item: item['conditions'].pop())
        self._assert_rejected(mutate_report=lambda item: item.update(fit_arm_outcome_count=23))
        self._assert_rejected(mutate_report=lambda item: item['conditions'][0]['arms'].pop())

    def test_repeat_mismatch_is_rejected(self):
        self._assert_rejected(mutate_report=lambda item: item['exploratory_pose_prerequisite'].__setitem__(
            'deterministic_repeats_match', False))

        def repeat_changed(item):
            item['conditions'][0]['arms'][0]['deterministic_repeat']['matched'] = False

        self._assert_rejected(mutate_report=repeat_changed)

    def test_changed_gate_or_any_promotion_omits_summary(self):
        self._assert_rejected(mutate_report=lambda item: item['exploratory_pose_prerequisite'].__setitem__(
            'passed', True))
        self._assert_rejected(mutate_report=lambda item: item['exploratory_pose_prerequisite'].__setitem__(
            'runtime_integration_authorized', True))
        self._assert_rejected(mutate_report=lambda item: item.update(default_or_runtime_promotion=True))
        self._assert_rejected(mutate_report=lambda item: item.update(real_pose_accuracy_claimed=True))

    def test_negative_result_requires_wrong_surface_accounting_and_support_reasons(self):
        def erase_wrong_surface_evidence(item):
            item['exploratory_pose_prerequisite']['negative_rows'][0][
                'full_original_wrong_surface_evidence'] = False

        self._assert_rejected(mutate_report=erase_wrong_surface_evidence)

        def change_support_reason(item):
            item['conditions'][7]['arms'][1]['validation_reason'] = 'other_reason'

        self._assert_rejected(mutate_report=change_support_reason)


if __name__ == '__main__':
    unittest.main()
