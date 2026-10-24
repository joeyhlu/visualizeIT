"""Codex-owned RAM replay contract checks using manufactured rows only."""
from dataclasses import replace
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from bench import workbench_artwork_replay as ar, workbench_contract as wc


def manufactured_row(**overrides):
    pose = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 1.), (0., 0., 0., 1.))
    row = wc.FrameResult(run_id='manufactured-run', clip_id='manufactured-clip', frame_id='f1',
        source_frame_index=1, timestamp_s=None, media_time_s=None, backend_role='primary_neural',
        backend_identity={}, mask_state='observed', pose_state='tracking', render_state='rendered',
        failure_reason=None, pose=pose, pose_units='metric', image_sha256='a' * 64,
        mask_frame_id='f1', mask_sha256='b' * 64, metrics={},
        evidence={'artifact_write_error': 'manufactured publication refusal', 'render_stats': {'rendered_pixels': 1}})
    return replace(row, **overrides)


class ArtworkRamReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ram-replay-unit-', dir='.cache')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_eligible_rendered_saved_pose_with_publication_error(self):
        state, reason = ar.eligibility(manufactured_row(), self.root)
        self.assertEqual(state, 'eligible')
        self.assertIn('publication', reason)

    def test_loss_recovery_block_and_suppression_refused_before_filesystem(self):
        cases = [dict(pose_state=state) for state in ('lost', 'recovering', 'blocked', 'unavailable')]
        cases += [dict(render_state='suppressed'), dict(render_state='unavailable'), dict(pose=None),
            dict(pose_units='object_unit'), dict(mask_state='unavailable'), dict(mask_frame_id='f0'),
            dict(mask_sha256=None), dict(backend_role='relative_geometry')]
        with mock.patch.object(Path, 'is_file', side_effect=AssertionError('must refuse before artifacts')):
            for fields in cases:
                with self.subTest(fields=fields):
                    self.assertEqual(ar.eligibility(manufactured_row(**fields), self.root)[0], 'ineligible')

    def test_positive_render_count_requires_integer_not_boolean(self):
        for count in (False, True, -1, 0, 1.5, '1', None):
            row = manufactured_row(evidence={'artifact_write_error': 'fixture', 'render_stats': {'rendered_pixels': count}})
            with self.subTest(count=count):
                self.assertEqual(ar.eligibility(row, self.root)[0], 'ineligible')

    def test_all_recorded_counter_types_are_exact(self):
        for value in (False, True, 0.0, -1, '0'):
            row = manufactured_row(evidence={'artifact_write_error': 'fixture',
                'render_stats': {'rendered_pixels': 1, 'mask_rejected_pixels': value}})
            with self.subTest(value=value):
                self.assertEqual(ar.eligibility(row, self.root)[0], 'ineligible')

    def test_wrong_saved_frame_metadata_refused_before_artwork_or_source_reads(self):
        from bench.test_workbench_contract import clip_document, run_document
        from bench import workbench_neural as wn
        clip = wc.ClipSpec.from_dict(clip_document())
        run = wc.RunSpec.from_dict(run_document())
        row = manufactured_row(run_id=run.run_id, clip_id=clip.clip_id, frame_id='f2',
            source_frame_index=1150, timestamp_s=0.2, media_time_s=0.2, mask_frame_id='f2')
        service = ar.ArtworkRamReplay(SimpleNamespace(load_clip=lambda _: clip),
            bundle_resolver=lambda _: None, admission_dir=None, repo_root=self.root)
        for changes in ({'clip_id': 'wrong-clip'}, {'source_frame_index': 1151},
                {'timestamp_s': 0.3}, {'media_time_s': None}, {'source_frame_index': 1150.0}):
            state = {'run': run, 'report': {'stages': {'primary_neural': {'status': 'completed'}}},
                'dir': self.root, 'table': {('primary_neural', 'f2'): replace(row, **changes)}}
            with self.subTest(changes=changes), mock.patch.object(service, '_load_run', return_value=state), \
                    mock.patch.object(wn, 'NeuralArtwork', side_effect=AssertionError('no render on wrong identity')) as art:
                status, response = service.replay(run.run_id, 'f2', 'primary_neural')
            self.assertEqual(status, 422)
            self.assertIn('differs', response['reason'])
            self.assertEqual(art.call_count, 0)

    def test_recorded_existing_png_takes_precedence_even_if_invalid_bytes(self):
        expected = self.root / 'evidence/primary_neural/f1.png'
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b'not-a-png; replay must not silently substitute')
        row = manufactured_row(evidence={'overlay': 'evidence/primary_neural/f1.png', 'render_stats': {'rendered_pixels': 1}})
        self.assertEqual(ar.eligibility(row, self.root)[0], 'original_recorded')
        self.assertEqual(expected.read_bytes(), b'not-a-png; replay must not silently substitute')

    def test_ambiguous_unrecorded_existing_artifact_refused(self):
        path = self.root / 'evidence/primary_neural/f1.png'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'foreign-file')
        self.assertEqual(ar.eligibility(manufactured_row(), self.root)[0], 'ineligible')
        self.assertEqual(path.read_bytes(), b'foreign-file')

    def test_cross_role_or_traversal_overlay_path_refused(self):
        for path in ('../other.png', 'evidence/frozen_baseline/f1.png', 'evidence/primary_neural/f2.png'):
            row = manufactured_row(evidence={'overlay': path, 'render_stats': {'rendered_pixels': 1}})
            with self.subTest(path=path):
                self.assertEqual(ar.eligibility(row, self.root)[0], 'ineligible')

    def test_absent_artifact_needs_recorded_explanation(self):
        row = manufactured_row(evidence={'render_stats': {'rendered_pixels': 1}})
        self.assertEqual(ar.eligibility(row, self.root)[0], 'ineligible')

    def test_descriptor_is_adjacent_and_does_not_mutate_row(self):
        row = manufactured_row()
        before = row.to_dict()
        doc = ar.artwork_sources('manufactured-run', [row], self.root)
        self.assertEqual(row.to_dict(), before)
        self.assertEqual(doc['rows'][0]['ram_replay'], 'eligible')
        self.assertIn('Historical bytes cannot be verified', doc['label'])
        self.assertNotIn('png', row.evidence)

    def service(self):
        return ar.ArtworkRamReplay(SimpleNamespace(), bundle_resolver=lambda _: None,
            admission_dir=None, repo_root=self.root)

    def test_busy_request_never_loads_or_queues_work(self):
        service = self.service()
        service._lock.acquire()
        self.addCleanup(service._lock.release)
        with mock.patch.object(service, '_replay', side_effect=AssertionError('no queued work')):
            status, result = service.replay('run', 'f1', 'primary_neural')
        self.assertEqual(status, 429)
        self.assertEqual(result['status'], 'busy')
        self.assertFalse(result['new_neural_inference'])
        self.assertFalse(result['disk_files_written'])

    def test_refusal_or_exception_releases_lock(self):
        service = self.service()
        for error in (ar.Refused('not_ready', 'unfinished'), RuntimeError('manufactured error')):
            with self.subTest(error=error), mock.patch.object(service, '_replay', side_effect=error):
                status, result = service.replay('run', 'f1', 'primary_neural')
            self.assertIn(status, (409, 422))
            self.assertTrue(service._lock.acquire(blocking=False))
            service._lock.release()
            self.assertFalse(result['historical_png_bytes_verified'])

    def test_bounded_read_refuses_before_reading_oversize_file(self):
        path = self.root / 'oversized.json'
        path.write_bytes(b'12345')
        with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('oversized input read')), \
                self.assertRaises(ar.Refused):
            ar._read_bounded(path, 4)

    def test_file_growth_cannot_bypass_read_or_hash_byte_cap(self):
        for operation in (ar._read_bounded, ar._sha_file):
            reads = []
            class GrowingStream(io.BytesIO):
                def read(self, count=-1):
                    data = super().read(count)
                    reads.append(len(data))
                    return data
            with self.subTest(operation=operation.__name__), \
                    mock.patch.object(Path, 'is_symlink', return_value=False), \
                    mock.patch.object(Path, 'is_file', return_value=True), \
                    mock.patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=2)), \
                    mock.patch.object(Path, 'open', return_value=GrowingStream(b'x' * 100)):
                with self.assertRaises(ar.Refused):
                    operation(Path('manufactured-growing.bin'), 4)
                self.assertLessEqual(sum(reads), 5)

    def test_unfinished_run_refuses_without_model_or_renderer_import(self):
        service = ar.ArtworkRamReplay(SimpleNamespace(runs_dir=self.root), bundle_resolver=lambda _: None,
            admission_dir=None, repo_root=self.root)
        folder = self.root / 'unfinished'
        folder.mkdir()
        status, result = service.replay('unfinished', 'f1', 'primary_neural')
        self.assertEqual(status, 409)
        self.assertEqual(result['status'], 'not_ready')
        self.assertFalse(result['new_pose_estimation'])


if __name__ == '__main__':
    unittest.main()
