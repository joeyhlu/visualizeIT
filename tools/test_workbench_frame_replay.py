"""Codex replay integration: manufactured memory and tiny temporary image/mask files."""
from pathlib import Path
import dataclasses
from contextlib import contextmanager
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_frame_boundary as fb
from bench import workbench_geometry as wg
from bench import workbench_neural as wn
from bench.test_workbench_contract import clip_document, run_document
from bench.test_workbench_frame_boundary import neural_fixture, relative_fixture, NumericalTestDouble, rendered_test_double
from tools import workbench_frame_replay as replay

ROOT = Path(__file__).resolve().parents[1]


class MemoryReader:
    def __init__(self, context, missing=()):
        self.context = context; self.missing = set(missing); self.closed = 0
        self.reads = []

    def verify_media(self):
        return {'kind': 'manufactured-memory-source', 'not_real_video_evidence': True}

    def read(self, frame):
        self.reads.append(frame.frame_id)
        if frame.frame_id in self.missing:
            raise OSError('manufactured unreadable frame')
        rgb = np.zeros((self.context.height, self.context.width, 3), np.uint8)
        return rgb, {'frame_id': frame.frame_id, 'source_frame_index': frame.source_frame_index,
            'image_sha256': wg.array_sha256(rgb), 'decoder_media_time_s': frame.media_time_s}

    def close(self):
        self.closed += 1


class MemoryMasks:
    def __init__(self, missing=()):
        self.missing = set(missing)

    def load(self, frame_id, shape):
        if frame_id in self.missing:
            return None, None
        mask = np.ones(shape, bool)
        return mask, {'mask_frame_id': frame_id, 'mask_sha256': wg.mask_sha256(mask)}


class RecordedReplayTests(unittest.TestCase):
    def test_malformed_context_refusal_closes_transferred_reader_and_session(self):
        for field, value in (('binding_json', b'not-json'), ('prefix_attestation_json', b'not-json')):
            with self.subTest(field=field):
                context, adapter, backend = neural_fixture(bundle=True)
                reader = MemoryReader(context)
                malformed = dataclasses.replace(context, **{field: value})
                try:
                    result = replay.replay_recorded(malformed, adapter, reader, MemoryMasks())
                except (fb.BoundaryRejected, ValueError):
                    pass  # Refusal may raise; resource ownership is unconditional.
                self.assertEqual((reader.closed, backend.closes), (1, 1))

    def test_neural_stage_release_failure_preserves_completed_rows_and_cleanup_error(self):
        for reader_fails in (False, True):
            with self.subTest(reader_fails=reader_fails):
                context, backend, events, factories = self.neural_entry_fixture(reader_fails=reader_fails)
                @contextmanager
                def failing_stage(role):
                    events.append('stage_enter')
                    try:
                        yield
                    finally:
                        events.append('stage_exit')
                        raise OSError('manufactured GPU stage release failure')
                result = self.invoke_neural_entry(context, dataclasses.replace(factories, gpu_stage=failing_stage))
                self.assertEqual(result['status'], 'blocked' if reader_fails else 'completed_with_cleanup_errors')
                self.assertEqual((len(result['warmup']), len(result['scored'])), (1, 2))
                self.assertIn('stage release failure', str(result['cleanup_errors']))
                self.assertEqual(result['provenance']['evidence_class'], 'test_double_execution')
                self.assertEqual(backend.closes, 1)

    def test_malformed_context_reports_reader_cleanup_failure_without_leaking_backend(self):
        context, adapter, backend = neural_fixture(bundle=True)
        reader = MemoryReader(context)
        def fail_close():
            reader.closed += 1
            raise OSError('manufactured invalid-context close failure')
        reader.close = fail_close
        with self.assertRaises(fb.BoundaryRejected) as caught:
            replay.replay_recorded(dataclasses.replace(context, binding_json=b'not-json'), adapter, reader, MemoryMasks())
        self.assertEqual((reader.closed, backend.closes), (1, 1))
        self.assertIn('invalid-context close failure', str(caught.exception.cleanup_errors))

    def test_package_directory_proof_matches_memory_and_refuses_tampered_bytes(self):
        context, adapter, tracker = relative_fixture()
        proof = adapter.proof
        package, arrays, digests = proof.verify()
        with tempfile.TemporaryDirectory(prefix='portable-package-', dir=ROOT / '.cache/video-workbench/scratch') as directory:
            base = Path(directory)
            (base / 'surface.json').write_bytes(proof._surface_json)
            for key, raw in proof._components:
                (base / getattr(package, key).path).write_bytes(raw)
            disk_proof = fb.RelativePackageProof.from_package_dir(base)
            disk_package, disk_arrays, disk_digests = disk_proof.verify()
            self.assertEqual(disk_proof.source, 'package_dir')
            self.assertEqual(disk_package.to_dict(), package.to_dict())
            self.assertEqual(disk_digests, digests)
            for key in arrays:
                np.testing.assert_array_equal(disk_arrays[key], arrays[key])
            changed = base / package.vertices.path
            raw = bytearray(changed.read_bytes()); raw[-1] ^= 1; changed.write_bytes(raw)
            with self.assertRaises(fb.BoundaryRejected):
                fb.RelativePackageProof.from_package_dir(base).verify()

    def test_real_gpu_lock_prevents_nested_execution_and_releases_after_exception(self):
        from bench import workbench_telemetry as wt
        with tempfile.TemporaryDirectory(prefix='portable-lock-', dir=ROOT / '.cache/video-workbench/scratch') as directory:
            base = Path(directory)
            with self.assertRaisesRegex(ValueError, 'manufactured failure'):
                with wt.gpu_stage('portable CPU contract test', lock_dir=base):
                    self.assertTrue(wt.gpu_stage_active())
                    with self.assertRaisesRegex(RuntimeError, 'gpu_stage_busy'):
                        with wt.gpu_stage('nested refused', lock_dir=base):
                            self.fail('nested stage entered')
                    raise ValueError('manufactured failure')
            self.assertFalse(wt.gpu_stage_active())
            self.assertFalse((base / 'gpu-stage.lock').exists())
            with wt.gpu_stage('subsequent CPU contract test', lock_dir=base):
                self.assertTrue(wt.gpu_stage_active())

    def test_real_video_decoder_feeds_matching_timestamped_frames_through_boundary(self):
        from bench.vision import cv2
        from bench.workbench_cli import Workspace, FrameReader
        with tempfile.TemporaryDirectory(prefix='portable-video-', dir=ROOT / '.cache/video-workbench/scratch') as directory:
            root = Path(directory); media = root / 'source'; media.mkdir()
            path = media / 'tiny.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
            self.assertTrue(writer.isOpened(), 'CPU MJPG encoder is required for this integration control')
            try:
                for index in range(4):
                    writer.write(np.full((24, 32, 3), index * 40, np.uint8))
            finally:
                writer.release()
            ws = Workspace(root / 'workspace', repo_root=ROOT); ws.init(roots={'fixture': media})
            doc = clip_document(start_index=0); doc.update(width=32, height=24)
            doc['media'] = {'root_id': 'fixture', 'path': path.name,
                'sha256': wn.sha256_file(path), 'bytes': path.stat().st_size}
            doc['intrinsics']['K'] = [100, 0, 15.5, 0, 100, 11.5, 0, 0, 1]
            doc['onboarding']['selection'].update(positive=[[8, 8]], negative=[[0, 0]])
            clip = wc.ClipSpec.from_dict(doc)
            run_doc = run_document(); run_doc['primary'] = {k: wn.PRIMARY[k] for k in ('artifact_id', 'export_sha256', 'export_bytes')}
            run = wc.RunSpec.from_dict(run_doc)
            backend = NumericalTestDouble()
            admission = wn.ServingAdmission(True, (), {'families': ['synthetic-family'], 'contract_sha256': 'c' * 64})
            session = wn.NeuralSession('primary_neural', backend, admission, run_id=run.run_id, clip=clip,
                family='synthetic-family', renderer=rendered_test_double)
            context = fb.FrameContextV1.build(clip, run, 'primary_neural', fb.neural_binding(clip, run, family='synthetic-family'))
            result = replay.replay_recorded(context, fb.NeuralSessionAdapter(session), FrameReader(ws, clip), MemoryMasks())
            self.assertEqual(result['status'], 'completed', result.get('failure'))
            self.assertEqual([fid for fid, _ in backend.calls], ['f2', 'f3'])
            self.assertEqual([row['source_frame_index'] for row in result['scored']], [2, 3])
            self.assertEqual(backend.closes, 1)

    def neural_entry_fixture(self, *, admitted=True, busy=False, reader_fails=False):
        context, adapter, backend = neural_fixture(bundle=True)
        events = []
        @contextmanager
        def stage(role):
            events.append('stage_enter')
            if busy:
                raise RuntimeError('manufactured GPU busy')
            try:
                yield
            finally:
                events.append('stage_exit')
        def make_backend(role, bundle, **kwargs):
            self.assertEqual(events, ['stage_enter'])
            events.append('backend')
            return backend
        def make_reader(ws, clip):
            events.append('reader')
            if reader_fails:
                raise OSError('manufactured reader construction failure')
            return MemoryReader(context)
        admission = wn.ServingAdmission(admitted, () if admitted else ('manufactured refusal',),
            {'families': ['synthetic-family'], 'contract_sha256': 'c' * 64})
        factories = replay.NeuralFactories(bundle_loader=lambda path: backend.bundle,
            admission_loader=lambda path: admission, backend_factory=make_backend,
            artwork_factory=lambda *a, **kw: SimpleNamespace(render=rendered_test_double),
            gpu_stage=stage, reader_factory=make_reader, masks_factory=lambda *a: MemoryMasks())
        return context, backend, events, factories

    def invoke_neural_entry(self, context, factories):
        return replay.run_neural_replay(None, context.clip, context.run, role=context.role,
            metric_bundle_path=Path('NEVER-READ-bundle.json'), admission_dir=Path('NEVER-READ-admission'),
            head_path=Path('NEVER-READ-head.pt'), mask_set='manufactured', factories=factories)

    def test_neural_entry_owns_stage_backend_and_all_frames_without_promoting_test_evidence(self):
        context, backend, events, factories = self.neural_entry_fixture()
        result = self.invoke_neural_entry(context, factories)
        self.assertEqual(result['status'], 'completed', result.get('failure'))
        self.assertEqual(events, ['stage_enter', 'backend', 'reader', 'stage_exit'])
        self.assertEqual([fid for fid, _ in backend.calls], ['f1', 'f2', 'f3'])
        self.assertEqual(backend.closes, 1)
        self.assertEqual(result['provenance']['evidence_class'], 'test_double_execution')
        self.assertEqual(result['provenance']['injected_factories'], factories.injected())

    def test_neural_entry_admission_or_busy_stage_cannot_construct_backend(self):
        for options in ({'admitted': False}, {'busy': True}):
            with self.subTest(options=options):
                context, backend, events, factories = self.neural_entry_fixture(**options)
                result = self.invoke_neural_entry(context, factories)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual((len(result['warmup']), len(result['scored'])), (1, 2))
                self.assertEqual(backend.calls, [])
                self.assertNotIn('backend', events)
                self.assertNotIn('reader', events)

    def test_neural_entry_reader_failure_closes_backend_and_releases_stage(self):
        context, backend, events, factories = self.neural_entry_fixture(reader_fails=True)
        result = self.invoke_neural_entry(context, factories)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual((len(result['warmup']), len(result['scored'])), (1, 2))
        self.assertEqual(backend.closes, 1)
        self.assertEqual(events[-1], 'stage_exit')

    def test_warmup_precedes_scoring_and_all_rows_are_valid(self):
        context, adapter, backend = neural_fixture(bundle=True)
        reader = MemoryReader(context)
        result = replay.replay_recorded(context, adapter, reader, MemoryMasks())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([fid for fid, K in backend.calls], ['f1', 'f2', 'f3'])
        self.assertEqual([r['frame_id'] for r in result['warmup']], ['f1'])
        self.assertEqual([r['frame_id'] for r in result['scored']], ['f2', 'f3'])
        for row in result['warmup'] + result['scored']:
            wc.FrameResult.from_dict(row)
        self.assertEqual((reader.closed, backend.closes), (1, 1))
        self.assertFalse(result['flags']['cached'])
        self.assertFalse(result['flags']['phone_or_continuous_camera_claim'])

    def test_missing_rgb_counts_all_frames_without_rearming_or_running_backend(self):
        context, adapter, backend = neural_fixture()
        reader = MemoryReader(context, missing=('f2',))
        result = replay.replay_recorded(context, adapter, reader, MemoryMasks())
        self.assertEqual(len(result['scored']), 2)
        self.assertEqual([r['pose_state'] for r in result['scored']], ['blocked', 'blocked'])
        self.assertEqual((backend.calls, backend.resets, reader.closed, backend.closes), ([], 0, 1, 1))

    def test_failed_media_verification_counts_every_frame_and_closes_unopened_adapter(self):
        context, adapter, backend = neural_fixture(bundle=True)
        reader = MemoryReader(context)
        def refuse():
            raise ValueError('manufactured source hash changed')
        reader.verify_media = refuse
        result = replay.replay_recorded(context, adapter, reader, MemoryMasks())
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual((len(result['warmup']), len(result['scored'])), (1, 2))
        self.assertEqual((backend.calls, reader.closed, backend.closes), ([], 1, 1))

    def test_mask_failure_is_counted_and_never_reuses_previous_overlay(self):
        context, adapter, backend = neural_fixture()
        outputs = []
        result = replay.replay_recorded(context, adapter, MemoryReader(context), MemoryMasks(missing=('f3',)),
            on_output=outputs.append)
        self.assertEqual(len(result['scored']), 2)
        self.assertEqual(len(backend.calls), 1)
        self.assertIsNotNone(outputs[0].overlay_rgba)
        self.assertIsNone(outputs[1].overlay_rgba)
        self.assertIsNone(outputs[1].result.pose)

    def test_callback_failure_keeps_completed_row_and_accounts_for_remaining_frame(self):
        context, adapter, backend = neural_fixture()
        reader = MemoryReader(context)
        def fail(output):
            raise RuntimeError('manufactured callback failure')
        result = replay.replay_recorded(context, adapter, reader, MemoryMasks(), on_output=fail)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual([r['pose_state'] for r in result['scored']], ['tracking', 'blocked'])
        self.assertEqual((reader.closed, backend.closes), (1, 1))

    def test_cleanup_errors_are_visible_with_complete_frame_accounting(self):
        context, adapter, backend = neural_fixture()
        reader = MemoryReader(context)
        def fail_close():
            reader.closed += 1
            raise OSError('manufactured decoder close error')
        reader.close = fail_close
        result = replay.replay_recorded(context, adapter, reader, MemoryMasks())
        self.assertEqual(result['status'], 'completed_with_cleanup_errors')
        self.assertIn('decoder close error', str(result['cleanup_errors']))
        self.assertEqual(len(result['scored']), 2)
        self.assertEqual(backend.closes, 1)

    def test_real_frame_reader_and_mask_store_reach_real_neural_session(self):
        from bench.vision import cv2
        from bench.workbench_cli import Workspace, FrameReader, MaskStore, MASK_SCHEMA
        scratch = ROOT / '.cache/video-workbench/scratch'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='portable-replay-unit-', dir=scratch) as directory:
            root = Path(directory); media_root = root / 'media'; sequence = media_root / 'sequence'
            sequence.mkdir(parents=True)
            ws = Workspace(root / 'workspace', repo_root=ROOT); ws.init(roots={'fixture': media_root})
            doc = clip_document(); doc.update(source_kind='image_sequence', width=32, height=24)
            doc['media'] = {'root_id': 'fixture', 'path': 'sequence', 'sha256': None, 'bytes': None}
            doc['intrinsics']['K'] = [100, 0, 15.5, 0, 100, 11.5, 0, 0, 1]
            doc['onboarding']['selection'].update(positive=[[8, 8]], negative=[[0, 0]])
            for index, row in enumerate(doc['frames']):
                rgb = np.full((24, 32, 3), index * 40, np.uint8)
                ok, encoded = cv2.imencode('.png', rgb)
                self.assertTrue(ok)
                path = sequence / (row['frame_id'] + '.png'); path.write_bytes(encoded.tobytes())
                row.update(image_path=path.name, image_sha256=wn.sha256_file(path))
            clip = wc.ClipSpec.from_dict(doc)
            run_doc = run_document(); run_doc['primary'] = {k: wn.PRIMARY[k] for k in ('artifact_id', 'export_sha256', 'export_bytes')}
            run = wc.RunSpec.from_dict(run_doc)
            mask_dir = ws.masks_dir / clip.clip_id / 'synthetic-observed'; mask_dir.mkdir(parents=True)
            rows = {}
            for fid in run.scheduled_frame_ids:
                path = mask_dir / (fid + '.png')
                ok, raw = cv2.imencode('.png', np.full((24, 32), 255, np.uint8)); self.assertTrue(ok)
                path.write_bytes(raw.tobytes())
                rows[fid] = {'path': path.name, 'sha256': wn.sha256_file(path), 'width': 32, 'height': 24}
            (mask_dir / 'manifest.json').write_bytes(wc.canonical_json({'schema': MASK_SCHEMA, 'clip_id': clip.clip_id,
                'mask_set': 'synthetic-observed', 'producer': 'manufactured observed mask fixture',
                'evaluator_inputs_read': False, 'masks': rows}))
            backend = NumericalTestDouble()
            admission = wn.ServingAdmission(True, (), {'families': ['synthetic-family'], 'contract_sha256': 'c' * 64})
            session = wn.NeuralSession('primary_neural', backend, admission, run_id=run.run_id, clip=clip,
                family='synthetic-family', renderer=rendered_test_double)
            parts = replay.build_neural_replay(ws, clip, run, session, role='primary_neural',
                family='synthetic-family', mask_set='synthetic-observed')
            self.assertIs(type(parts['reader']), FrameReader)
            self.assertIs(type(parts['masks']), MaskStore)
            result = replay.replay_recorded(**parts)
            self.assertEqual(result['status'], 'completed', result.get('failure'))
            self.assertEqual([fid for fid, K in backend.calls], ['f2', 'f3'])
            self.assertEqual([r['mask_state'] for r in result['scored']], ['observed', 'observed'])
            self.assertEqual(backend.closes, 1)
