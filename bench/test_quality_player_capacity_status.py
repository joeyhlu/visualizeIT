"""Fail-closed presentation checks for the closed R8 bottle capacity receipts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import quality_player as player


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class BottleR8CapacityStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = (
            player.BOTTLE_R8_ACTUAL_WORKER,
            player.BOTTLE_R8_ACTUAL_TERMINAL,
            player.BOTTLE_R8_FULL28_CONTRACT,
            player.BOTTLE_R8_FULL28_TERMINAL,
        )
        cls.hashes = (
            player.BOTTLE_R8_ACTUAL_WORKER_SHA256,
            player.BOTTLE_R8_ACTUAL_TERMINAL_SHA256,
            player.BOTTLE_R8_FULL28_CONTRACT_SHA256,
            player.BOTTLE_R8_FULL28_TERMINAL_SHA256,
        )
        cls.records = []
        for path, expected_hash in zip(cls.paths, cls.hashes):
            raw = path.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest().upper()
            if actual_hash != expected_hash:
                raise AssertionError(f'closed R8 receipt hash changed: {path.name}')
            cls.records.append(json.loads(raw.decode('utf-8')))

    def _summary_from_records(self, records=None):
        return player._bottle_r8_capacity_summary_from_records(
            *(copy.deepcopy(self.records if records is None else records)))

    def _assert_rejected(self, mutate):
        records = copy.deepcopy(self.records)
        mutate(*records)
        self.assertIsNone(self._summary_from_records(records))

    def test_exact_closed_receipts_render_compact_capacity_only_status(self):
        summary = player._load_bottle_r8_capacity_summary()
        self.assertIsNotNone(summary)
        self.assertEqual(
            [row['context_id'] for row in summary['source_views']],
            list(player.BOTTLE_R8_SOURCE_CONTEXTS),
        )
        self.assertEqual(
            [row['witness_count'] for row in summary['source_views']],
            list(player.BOTTLE_R8_WITNESS_COUNTS),
        )
        self.assertTrue(all(row['fit_pool_count'] == row['fit_count'] == row['bank_anchor_count'] == 0
                            for row in summary['source_views']))
        self.assertEqual(summary['actual_rows'], 12)
        self.assertEqual(summary['actual_capacity_possible'], 0)
        self.assertEqual(summary['full28_rows'], 28)
        self.assertEqual(summary['full28_unmeasured'], 28)
        self.assertEqual(summary['wall_limit_seconds'], 900)

        markup = player._bottle_r8_capacity_summary_html(summary)
        self.assertIn('<details class="bottleR8CapacitySummary"', markup)
        self.assertNotIn('<details open', markup)
        self.assertIn('0/12 rows possible', markup)
        self.assertIn('Witness counts (frames 10, 50, 100; 0° then 180°) were 1, 0, 5, 1, 7, 1', markup)
        self.assertIn('zero disjoint fit-pool IDs, zero selected fit IDs and zero bank anchors', markup)
        self.assertIn('does not show that source texture is absent', markup)
        self.assertIn('shared 900-second limit', markup)
        self.assertIn('All 28/28 cases remain unmeasured', markup)
        self.assertIn('No matching, fitting or new poses were run', markup)
        self.assertIn('epsilon remains at its frozen value', markup)
        self.assertIn('renderer-to-camera depth parity is pending', markup)
        self.assertNotIn(str(player.CACHE), markup)
        self.assertNotIn('41431FEA', markup)

    def test_actual_view_order_and_exact_denominators_are_required(self):
        self._assert_rejected(lambda worker, *_: worker['actual']['source_views'].pop())

        def reorder(worker, *_):
            worker['actual']['source_views'][0], worker['actual']['source_views'][1] = (
                worker['actual']['source_views'][1], worker['actual']['source_views'][0])

        self._assert_rejected(reorder)

        def change_witness_count(worker, *_):
            worker['actual']['source_views'][0]['capacity']['witness_count'] += 1

        self._assert_rejected(change_witness_count)

        def omit_row(worker, *_):
            worker['actual']['rows'].pop()

        self._assert_rejected(omit_row)

    def test_fit_or_capacity_success_claims_cannot_be_smuggled_into_card(self):
        def add_fit(worker, *_):
            worker['actual']['source_views'][0]['capacity']['fit_pool_count'] = 1

        self._assert_rejected(add_fit)

        def add_capacity_success(worker, *_):
            worker['actual']['rows'][0]['capacity_possible'] = True

        self._assert_rejected(add_capacity_success)

        def add_accuracy(worker, *_):
            worker['actual']['claims']['accuracy'] = True

        self._assert_rejected(add_accuracy)

    def test_worker_terminal_binding_and_success_state_are_required(self):
        self._assert_rejected(lambda worker, terminal, contract, full_terminal:
                              terminal.update(return_code=1))
        self._assert_rejected(lambda worker, terminal, contract, full_terminal:
                              terminal.update(worker_sha256='0' * 64))
        self._assert_rejected(lambda worker, *_: worker.update(state='partial'))

    def test_full28_timeout_contract_and_all_unmeasured_cases_are_required(self):
        self._assert_rejected(lambda worker, terminal, contract, full_terminal:
                              contract['limits'].__setitem__('wall_seconds', 901))
        self._assert_rejected(lambda *records: records[3].update(timed_out=False))
        self._assert_rejected(lambda *records: records[3]['all_cases'].pop())
        self._assert_rejected(lambda *records: records[3]['all_cases'][0].update(
            state='capacity_measured'))
        self._assert_rejected(lambda *records: records[3].update(full_28_case_screen_complete=True))

    def test_missing_or_changed_fixed_receipt_falls_back_to_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad_receipt = Path(temporary) / 'terminal.json'
            bad_receipt.write_bytes(self.paths[3].read_bytes() + b' ')
            with mock.patch.object(player, 'BOTTLE_R8_FULL28_TERMINAL', bad_receipt):
                self.assertIsNone(player._load_bottle_r8_capacity_summary())

            missing = Path(temporary) / 'missing-worker.json'
            with mock.patch.object(player, 'BOTTLE_R8_ACTUAL_WORKER', missing):
                self.assertIsNone(player._load_bottle_r8_capacity_summary())

    def test_malformed_nonfinite_or_non_json_receipt_records_are_rejected(self):
        self._assert_rejected(lambda worker, *_: worker.update(actual=None))

        def invalid_count(worker, *_):
            worker['actual']['counts']['rows_decided'] = float('nan')

        self._assert_rejected(invalid_count)


if __name__ == '__main__':
    unittest.main()
