"""Independent recovery invariants; manufactured data, no model execution."""
import copy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from tools import workbench_mug_artifact_recovery as recovery
from bench import workbench_image_io as imageio


class RecoveryTests(unittest.TestCase):
    def test_terminal_seal_rejects_boolean_integer_aliases(self):
        seal = {'run_id': recovery.SOURCE_RUN_ID, 'exit_code': 1, 'worker_exited': True,
                'original_report_status': 'failed', 'rows': 480,
                'role_stages': {r: 'completed' for r in recovery.ROLES},
                'missing_overlays': sorted(recovery.MISSING), 'original_run_immutable': True,
                'no_new_neural_execution': True,
                'launcher_plan': {'path': recovery.LAUNCHER_PLAN_PATH, 'sha256': recovery.LAUNCHER_PLAN_SHA256},
                'run_files': {str(i): {} for i in range(recovery.SEAL_RUN_FILES)},
                'source_snapshot': {str(i): {} for i in range(recovery.SEAL_SNAPSHOTS)},
                'failure_stage': {'stage_cache_store_06': {}},
                'models': {r: {'base_checkpoint_sha256': recovery.BASE_SHA256,
                               'installed_head_sha256': recovery.HEAD_SHA256 if r == 'primary_neural' else None}
                           for r in recovery.ROLES}}
        recovery.check_seal(seal)
        for key, value in [('exit_code', True), ('worker_exited', 1),
                           ('original_run_immutable', 1), ('no_new_neural_execution', 1), ('rows', 480.0)]:
            bad = copy.deepcopy(seal); bad[key] = value
            with self.subTest(key=key), self.assertRaises(recovery.RecoveryError): recovery.check_seal(bad)

    def rows(self):
        rows = []
        for role in recovery.ROLES:
            for index, fid in enumerate(recovery.FRAME_IDS):
                count = recovery.EXPECTED_COUNTS[role]
                rendered = index < count['rendered']
                pixels = recovery.MISSING.get(f'evidence/{role}/{fid}.png', (None, None, 3))[2]
                identity = {'role': role, 'base_checkpoint_sha256': recovery.BASE_SHA256,
                            'admission_contract_sha256': 'contract', 'installed_head_sha256': None}
                if role == 'primary_neural':
                    identity.update(export_sha256=recovery.HEAD_SHA256, installed_head_sha256=recovery.HEAD_SHA256)
                evidence = {'overlay': f'evidence/{role}/{fid}.png', 'render_stats': {'rendered_pixels': pixels}} if rendered else {}
                rows.append({'run_id': recovery.SOURCE_RUN_ID, 'clip_id': recovery.CLIP_ID,
                             'backend_role': role, 'frame_id': fid, 'mask_frame_id': fid,
                             'mask_state': 'observed', 'backend_identity': identity,
                             'pose_state': 'tracking' if index < count['tracking'] else 'recovering',
                             'render_state': 'rendered' if rendered else 'suppressed',
                             'failure_reason': None, 'evidence': evidence})
        return rows

    def test_complete_accounting_and_each_identity_mutation_rejected(self):
        rows = self.rows()
        self.assertEqual(len(recovery.check_source_rows(rows, 'contract')), 371)
        mutations = [('base_checkpoint_sha256', 'wrong'), ('installed_head_sha256', 'older'),
                     ('export_sha256', 'older'), ('admission_contract_sha256', 'other')]
        for key, value in mutations:
            changed = copy.deepcopy(rows)
            changed[0]['backend_identity'][key] = value
            with self.subTest(key=key), self.assertRaises(recovery.RecoveryError):
                recovery.check_source_rows(changed, 'contract')

    def test_missing_duplicate_mask_mismatch_and_unrecorded_artifact_error_rejected(self):
        rows = self.rows()
        cases = [rows[:-1], [rows[0]] + rows[0:-1]]
        for key, value in [('mask_frame_id', 'other'), ('render_state', 'suppressed')]:
            changed = copy.deepcopy(rows); changed[0][key] = value; cases.append(changed)
        changed = copy.deepcopy(rows); changed[0]['evidence']['artifact_write_error'] = 'disk'; cases.append(changed)
        for changed in cases:
            with self.subTest(), self.assertRaises(recovery.RecoveryError):
                recovery.check_source_rows(changed, 'contract')

    def test_rows_preserved_except_identifier_including_failed_pose(self):
        rows = self.rows(); rows[-1]['pose'] = None
        derived = recovery.derivative_rows(rows, 'new')
        self.assertTrue(recovery.rows_equal_except_run_id(rows, derived))
        self.assertEqual(rows[0]['run_id'], recovery.SOURCE_RUN_ID)
        for key, value in [('pose', []), ('failure_reason', 'changed'), ('mask_state', 'missing')]:
            changed = copy.deepcopy(derived); changed[-1][key] = value
            self.assertFalse(recovery.rows_equal_except_run_id(rows, changed))
        self.assertFalse(recovery.rows_equal_except_run_id(rows, derived[:-1]))

    def drawing(self):
        rgba = np.zeros((3, 4, 4), np.uint8); rgba[0, :2] = [21, 47, 88, 255]
        stats = {'rendered_pixels': 2, 'patch_faces': 5, 'reason': None, 'rejected': 0}
        return stats, {'stats': dict(stats), 'rgba': rgba}

    def test_every_render_stat_and_reason_must_match(self):
        stats, drawn = self.drawing()
        np.testing.assert_array_equal(recovery.check_regenerated(stats, drawn, (3, 4), 2), drawn['rgba'])
        for key, value in [('reason', 'unexpected'), ('patch_faces', 6), ('rejected', 1)]:
            changed = copy.deepcopy(drawn); changed['stats'][key] = value
            with self.subTest(key=key), self.assertRaises(recovery.RecoveryError):
                recovery.check_regenerated(stats, changed, (3, 4), 2)

    def test_shape_dtype_binary_alpha_and_count_are_independent_requirements(self):
        stats, drawn = self.drawing()
        images = [drawn['rgba'].astype(float), drawn['rgba'][:, :, :3], drawn['rgba'][:2]]
        alpha = drawn['rgba'].copy(); alpha[0, 0, 3] = 128; images.append(alpha)
        count = drawn['rgba'].copy(); count[1, 0, 3] = 255; images.append(count)
        for image in images:
            with self.subTest(), self.assertRaises(recovery.RecoveryError):
                recovery.check_regenerated(stats, dict(drawn, rgba=image), (3, 4), 2)
        with self.assertRaises(recovery.RecoveryError):
            recovery.check_regenerated(stats, drawn, (3, 4), 3)

    def test_copied_png_is_strict_and_alpha_count_matches_record(self):
        _, drawn = self.drawing(); raw = imageio.encode_png_rgba(drawn['rgba'])
        self.assertEqual(recovery.check_copied_png(raw, (3, 4), 2)['alpha_on_pixels'], 2)
        for data, shape, pixels in [(b'broken', (3, 4), 2), (raw, (4, 3), 2), (raw, (3, 4), 3)]:
            with self.subTest(), self.assertRaises(recovery.RecoveryError):
                recovery.check_copied_png(data, shape, pixels)

    def test_guards_block_all_execution_and_restore_after_exception(self):
        from bench import workbench_cli as cli, workbench_neural as neural, workbench_cache as cache
        original = neural.NeuralSession
        with recovery.recovery_guards() as attempts:
            for call in [lambda: neural.NeuralSession(), lambda: neural.GoTrackMetricBackend(),
                         lambda: neural.segment_prompted(), lambda: cli.execute_run(),
                         lambda: cli.Pipeline.execute(None), lambda: cache.StageCache.store(None),
                         lambda: importlib.import_module('torch')]:
                with self.assertRaises(recovery.RecoveryViolation): call()
            self.assertEqual(len(attempts), 7)
        self.assertIs(neural.NeuralSession, original)

    def test_bad_plan_stops_before_guards_and_any_recovery(self):
        with mock.patch.object(recovery, 'load_plan', side_effect=recovery.RecoveryError('pin mismatch')), \
             mock.patch.object(recovery, '_recover') as execute, mock.patch.object(recovery, '_failure_receipt') as receipt:
            with self.assertRaises(recovery.RecoveryError): recovery.run(Path('missing.json'), 'bad')
            execute.assert_not_called(); receipt.assert_not_called()

    def test_closure_failure_records_failure_and_never_returns_success(self):
        plan = {'derivative': {'dir': 'unused'}}
        with mock.patch.object(recovery, 'load_plan', return_value=plan), \
             mock.patch.object(recovery, '_recover', side_effect=recovery.RecoveryError('closure changed')), \
             mock.patch.object(recovery, '_failure_receipt') as receipt:
            with self.assertRaises(recovery.RecoveryFailed): recovery.run(Path('manufactured'), 'pin')
            receipt.assert_called_once()
            self.assertIn('closure changed', str(receipt.call_args.args[-1]))

    def test_reserve_contains_outputs_and_largest_temp_and_enforces_eight_mib(self):
        reserve = recovery.budget_reserve(100, 200, 50, 60)
        self.assertGreater(reserve, 410 + recovery.MAX_REPORT_BYTES + 2 * recovery.MAX_REGENERATED_PNG_BYTES)
        self.assertLess(reserve, recovery.MAX_DERIVATIVE_BYTES)
        workspace = mock.Mock()
        with mock.patch.object(recovery, '_tree_size', return_value=recovery.MAX_DERIVATIVE_BYTES):
            with self.assertRaises(recovery.RecoveryError):
                recovery._require_room(workspace, recovery.ROOT, 1, reserve)
        workspace.require_budget.assert_not_called()


if __name__ == '__main__': unittest.main()
