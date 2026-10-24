"""Analytical recovery fixtures only; no models, training, or quality claims."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import quality_adaptation_mask_recovery_v1 as recovery
from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench.test_quality_adaptation_portable_prepare import known_role_fixture

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


class ResourceTests(unittest.TestCase):
    def test_prior_stage_safety_checks(self):
        class Context:
            sha256 = 'a' * 64
        valid = dict(stage='banks', context_sha256=Context.sha256, status='completed',
                     error=None, test_payload_reads=0, training_ready=False, elapsed_seconds=12.)
        recovery._clean_receipt(Context(), 'banks', valid)
        for change in ({'error': 'cleanup_failed'}, {'test_payload_reads': 1},
                       {'training_ready': True}, {'elapsed_seconds': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery._clean_receipt(Context(), 'banks', {**valid, **change})

    def test_context_limits_device_and_parity_cannot_change_on_retry(self):
        old = dict(output_root='old', scratch_root='old-scratch', device='cpu',
                   limits=dict(storage_bytes=32 * GIB, wall_seconds=7200, reserve_bytes=GIB),
                   parity={'reference': 'same'})
        new = {**old, 'output_root': 'new', 'scratch_root': 'new-scratch'}
        recovery._same_context(old, new)
        for change in ({'device': 'cuda'}, {'limits': {**old['limits'], 'wall_seconds': 7201}},
                       {'parity': {'reference': 'different'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery._same_context(old, {**new, **change})

    def test_actual_failed_layout_reserves_proposals_on_scratch_only(self):
        result = recovery.allocation_check(resident=15_155_000_000, limit=32 * GIB,
            reserve=GIB, future_bytes=5_898_415_104, scratch_free=10**12,
            output_free=5_598_273_536, output_future=112 * 1024**2,
            other_free=[5_598_273_536])
        self.assertEqual(result['future_bytes'], 5_898_415_104)

    def test_scratch_full_union_and_output_limits_remain_enforced(self):
        base = dict(resident=15 * GIB, limit=32 * GIB, reserve=GIB,
            future_bytes=6 * GIB, scratch_free=100 * GIB,
            output_free=5 * GIB, output_future=GIB, other_free=[5 * GIB])
        for change in ({'scratch_free': 6 * GIB}, {'resident': 25 * GIB},
                       {'output_free': GIB}, {'other_free': [GIB - 1]},
                       {'same_device': True, 'scratch_free': 7 * GIB}):
            with self.subTest(change=change), self.assertRaises(raw.ResourceError):
                recovery.allocation_check(**{**base, **change})
        with self.assertRaises(ValueError):
            recovery.allocation_check(**{**base, 'limit': 100 * GIB})

    def test_nonfinite_or_boolean_resource_measurements_rejected(self):
        base = dict(resident=1, limit=32 * GIB, reserve=GIB,
            future_bytes=0, scratch_free=2 * GIB, other_free=[])
        for value in (True, -1, float('nan'), 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recovery.allocation_check(**{**base, 'future_bytes': value})


class ImportTests(unittest.TestCase):
    def setUp(self):
        # This suite tests metadata/import behavior and the pure allocation
        # checks above; it must not recursively scan the laptop's model cache.
        guard = patch.object(raw, 'resource_check')
        guard.start()
        self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR') or ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.recipe, _ = known_role_fixture(self.base)
        self.contexts = []
        for name in ('old', 'retry'):
            parity_path = self.base / 'fixture-parity.json'
            if not parity_path.exists():
                parity_path.write_text('{"MODEL_FREE_TEST_FIXTURE":true}')
            document = raw.context_document(output_root=self.base / name,
                scratch_root=self.base / (name + '-scratch'),
                parity={k: p.pin(parity_path) for k in ('reference', 'candidate')},
                device='cpu', storage_bytes=32 * GIB)
            data = raw.encoded(document)
            self.contexts.append(raw.PortableContext(data, hashlib.sha256(data).hexdigest()))
        self.old, self.retry = self.contexts
        self.old_root = Path(self.old.document()['output_root'])
        self.retry_root = Path(self.retry.document()['output_root'])
        recipe_path = self.base / 'recipe.json'
        p.atomic_json(recipe_path, self.recipe)
        raw.freeze(recipe_path, p.sha256(recipe_path), self.old)
        _, spec, plan, rows = raw.frozen(self.old)
        self.fresh = raw.physical_rows(spec, plan)
        for row in rows:
            if row['split'] in ('train', 'validation'):
                for kind in ('render', 'bank', 'mask', 'seed'):
                    row[kind] = dict(status='unavailable', failure_reason='analytical_fixture_failure')
        for split in ('train', 'validation'):
            row = next(r for r in rows if r['split'] == split)
            directory = raw._view_directory(self.old_root, row)
            bundle = directory / 'bundle'
            bundle.mkdir(parents=True)
            files = {}
            for key, name in (('input', 'input.json'), ('video', 'synthetic.mp4'), ('mesh', 'object.glb')):
                path = bundle / name
                path.write_bytes(b'MODEL_FREE_TEST_FIXTURE_' + key.encode())
                files[key] = p.pin(path)
            row['render'] = dict(status='available', **files)
            # Invalid JSON is intentional: generation must never be opened here.
            (directory / 'generation.json').write_bytes(b'POISON_GENERATION_DO_NOT_OPEN')
            banks = self.old_root / 'banks'
            banks.mkdir(exist_ok=True)
            pins = {}
            for key in ('bank', 'cnos_bank'):
                path = banks / (row['object_id'] + '-' + key + '.npz')
                path.write_bytes(b'MODEL_FREE_TEST_FIXTURE_BANK_' + key.encode())
                pins[key] = p.pin(path)
            row['bank'] = dict(status='available', **pins)
            p.atomic_json(self.old_root / f"cnos-{int(row['object_id']):06d}.json", {'bank': pins['cnos_bank']})
        raw._save_rows(self.old, rows)
        self.rows = rows
        for name, elapsed in (('render', 10.), ('banks', 20.), ('masks', 3.)):
            p.atomic_json(self.old_root / ('stage-' + name + '.json'),
                dict(stage=name, context_sha256=self.old.sha256,
                     status='failed' if name == 'masks' else 'completed',
                     error='fixture reserve failure' if name == 'masks' else None,
                     elapsed_seconds=elapsed, test_payload_reads=0, training_ready=False))
        review_path = self.base / 'MODEL_FREE_TEST_FIXTURE-review.json'
        p.atomic_json(review_path, dict(MODEL_FREE_TEST_FIXTURE=True,
            schema=recovery.REVIEW_SCHEMA, decision='APPROVE',
            original_raw_source_files=self.old.document()['source_files'],
            files={recovery.SOURCE_NAME: p.sha256(Path(recovery.__file__))}))
        self.review = p.pin(review_path)

    def test_last_test_metadata_rejected_before_any_path_or_hash(self):
        tampered = copy.deepcopy(self.rows)
        next(row for row in reversed(tampered) if row['split'] == 'test')['mask'] = {'status': 'available'}
        with patch.object(p, 'no_links', side_effect=AssertionError('path traversal')), \
                patch.object(p, 'verify_pin', side_effect=AssertionError('payload hash')):
            with self.assertRaisesRegex(ValueError, 'sealed TEST'):
                recovery.import_rows(self.fresh, tampered)

    def test_fresh_pending_barrier_precedes_payload_open(self):
        fresh = copy.deepcopy(self.fresh)
        next(row for row in reversed(fresh) if row['split'] == 'train')['mask'] = {'status': 'available'}
        with patch.object(p, 'no_links', side_effect=AssertionError('path traversal')):
            with self.assertRaisesRegex(ValueError, 'fresh recovery states'):
                recovery.link_inputs(self.rows, fresh, old_root=self.old_root,
                    new_root=self.retry_root, check_storage=lambda: None)

    def test_outside_source_rejected_before_payload_hash_or_path_traversal(self):
        tampered = copy.deepcopy(self.rows)
        row = next(r for r in tampered if r['render']['status'] == 'available')
        row['render']['input']['path'] = (self.base / 'SEALED_TEST_DO_NOT_OPEN').as_posix()
        with patch.object(p, 'verify_pin', side_effect=AssertionError('payload hash')), \
                patch.object(p, 'no_links', side_effect=AssertionError('path traversal')):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                recovery.link_inputs(tampered, self.fresh, old_root=self.old_root,
                    new_root=self.retry_root, check_storage=lambda: None)

    def test_fresh_mode_binds_original_stages_without_import_or_clock_reset(self):
        (self.old_root / 'stage-masks.json').unlink()
        for row in self.rows:
            if row['split'] in ('train', 'validation'):
                row['mask'] = {'status': 'pending'}
                row['seed'] = {'status': 'pending'}
        raw._save_rows(self.old, self.rows)
        recovery.bind_fresh(self.old, self.review)
        source = recovery.validate_extension(self.old, self.review)
        self.assertEqual(source['mode'], 'fresh')
        self.assertEqual(source['elapsed_seconds'], 0.)
        self.assertFalse((self.old_root / recovery.IMPORT_MARKER).exists())
        path = self.old_root / 'stage-banks.json'
        receipt = p.read_json(path)
        receipt['error'] = 'cleanup_failed'
        path.write_bytes(raw.encoded(receipt))
        with self.assertRaisesRegex(ValueError, 'safety differs'):
            recovery.validate_extension(self.old, self.review)

    def test_mask_callback_preserves_unwritten_proposal_cap_and_cleanup(self):
        from bench import quality_adaptation_larger as larger
        from bench import quality_adaptation_larger_optimized as optimized
        cap = optimized.PROPOSAL_BYTES_PER_VIEW
        observed = []
        def phases(views, scratch, backend, *, reserve, persist):
            reserve(cap)
            reserve(0)
            raise raw.ResourceError('fixture cancellation')
        # There are two available objects in this fixture, one view each. The
        # first cancellation must account for every remaining active view.
        with patch.object(raw, '_require_stage'), patch.object(larger, 'configure_torch', return_value=object()), \
                patch.object(larger, 'cleanup_device') as cleanup, patch.object(optimized, 'NeuralMaskBackend'), \
                patch.object(optimized, 'run_mask_phases', side_effect=phases), \
                patch.object(raw, '_usage', side_effect=[0, 4096]), \
                patch.object(recovery, 'proposal_reserve', side_effect=lambda ctx, amount, **kw: observed.append((amount, kw))):
            with self.assertRaises(raw.ResourceError):
                recovery.masks(self.old, sample_combined=lambda: {'resident_bytes': 1024})
            cleanup.assert_called_once()
        self.assertEqual([value[0] for value in observed], [cap, cap - 4096])
        self.assertTrue(all(value[1]['output_future'] == 2 * 1024**2 + p.TEMP_ENVELOPE for value in observed))
        self.assertTrue(all(row['mask']['status'] == 'unavailable' for row in raw.frozen(self.old)[3]
                            if row['split'] in ('train', 'validation')))

    @unittest.skipIf(os.name == 'nt', 'same-inode recovery import requires Linux cloud verification')
    def test_actual_freeze_import_validation_preserves_failures_and_opaque_generation(self):
        before = {p.name: p.read_bytes() for p in self.old_root.glob('stage-*.json')}
        old_index = (self.old_root / 'physical-index.json').read_bytes()
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.name == 'generation.json':
                raise AssertionError('generation payload opened before labels')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open):
            result = recovery.prepare_retry(self.old, self.retry, self.review,
                sample_combined=lambda: {'resident_bytes': 1024})
            marker = recovery.validate_extension(self.retry, self.review)
        self.assertEqual(result['context'], p.pin(self.retry_root / 'context.json'))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.old_root.glob('stage-*.json')})
        self.assertEqual(old_index, (self.old_root / 'physical-index.json').read_bytes())
        expected_prior = p.read_json(self.old_root / 'freeze.json')['elapsed_seconds'] + 3.
        self.assertEqual(marker['charged_prior_seconds'], expected_prior)
        self.assertEqual(marker['elapsed_seconds'], expected_prior + marker['transport_elapsed_seconds'])
        total = p.read_json(self.retry_root / 'freeze.json')['elapsed_seconds']
        total += sum(p.read_json(path)['elapsed_seconds'] for path in self.retry_root.glob('stage-*.json'))
        self.assertAlmostEqual(total, expected_prior + marker['transport_elapsed_seconds'] + 30.
            + p.read_json(self.retry_root / 'freeze.json')['elapsed_seconds'])
        for row in raw.frozen(self.retry)[3]:
            if row['split'] in ('train', 'validation'):
                self.assertEqual(row['mask'], {'status': 'pending'})
                self.assertEqual(row['seed'], {'status': 'pending'})
            if row['bank']['status'] == 'available':
                pointer = self.retry_root / f"cnos-{int(row['object_id']):06d}.json"
                self.assertEqual(p.read_json(pointer), {'bank': row['bank']['cnos_bank']})
        self.assertEqual(marker['test_payload_reads'], 0)
        self.assertEqual(marker['generation_payload_reads'], 0)
        with self.assertRaises(FileExistsError):
            recovery.prepare_retry(self.old, self.retry, self.review,
                sample_combined=lambda: {'resident_bytes': 1024})
        # Elapsed tampering cannot release previously spent RAW wall time.
        path = self.retry_root / recovery.IMPORT_MARKER
        value = p.read_json(path)
        value['elapsed_seconds'] = 0.
        path.write_bytes(raw.encoded(value))
        with self.assertRaisesRegex(ValueError, 'charged exactly once'):
            recovery.validate_extension(self.retry, self.review)


if __name__ == '__main__':
    unittest.main()
