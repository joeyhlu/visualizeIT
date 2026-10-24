"""Codex main-runner integration controls with manufactured media and estimators.

Actual Workspace, FrameReader, MaskStore, StageCache, NeuralSession and boundary
execute. The model constructor and artwork are explicit test doubles; no weights
or evaluator data are read and no GPU is used.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import copy
import json
import unittest
from unittest import mock

import numpy as np

from bench import workbench_cli as cli, workbench_contract as wc
from bench import workbench_frame_boundary as fb, workbench_neural as wn
from bench import workbench_telemetry as wt
from bench.test_workbench_contract import clip_document
from bench.test_workbench_frame_boundary import NumericalTestDouble, rendered_test_double
from bench.test_workbench_frame_boundary import relative_proof_components

ROOT = Path(__file__).resolve().parents[1]


class BoundaryAdoptionTests(unittest.TestCase):
    def setUp(self):
        from bench.vision import cv2
        scratch = ROOT / '.cache/video-workbench/scratch'; scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='boundary-adoption-unit-', dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); images = self.root / 'images'; images.mkdir()
        self.ws = cli.Workspace(self.root / 'workspace', repo_root=ROOT)
        self.ws.init({'fixture': str(self.root)})
        doc = clip_document(start_index=0)
        doc.update(source_kind='image_sequence', width=32, height=24)
        doc['media'] = {'root_id': 'fixture', 'path': 'images', 'sha256': None, 'bytes': None}
        doc['intrinsics']['K'] = [100, 0, 15.5, 0, 100, 11.5, 0, 0, 1]
        doc['onboarding']['selection'].update(positive=[[8, 8]], negative=[[0, 0]])
        for index, row in enumerate(doc['frames']):
            ok, raw = cv2.imencode('.png', np.full((24, 32, 3), index * 40, np.uint8)); self.assertTrue(ok)
            path = images / (row['frame_id'] + '.png'); path.write_bytes(raw.tobytes())
            row.update(image_path=path.name, image_sha256=wn.sha256_file(path))
        self.clip = wc.ClipSpec.from_dict(doc); self.ws.save_clip(self.clip)
        mask_dir = self.ws.masks_dir / self.clip.clip_id / 'manufactured'; mask_dir.mkdir(parents=True)
        masks = {}
        for row in doc['frames']:
            path = mask_dir / (row['frame_id'] + '.png')
            ok, raw = cv2.imencode('.png', np.full((24, 32), 255, np.uint8)); self.assertTrue(ok)
            path.write_bytes(raw.tobytes())
            masks[row['frame_id']] = {'path': path.name, 'sha256': wn.sha256_file(path), 'width': 32, 'height': 24}
        (mask_dir / 'manifest.json').write_bytes(wc.canonical_json({'schema': cli.MASK_SCHEMA,
            'clip_id': self.clip.clip_id, 'mask_set': 'manufactured', 'producer': 'Codex manufactured observations',
            'evaluator_inputs_read': False, 'masks': masks}))
        self.bundle_path = self.root / 'manufactured-bundle.json'; self.bundle_path.write_bytes(b'{}')
        self.head_path = self.root / 'manufactured-head.pt'; self.head_path.write_bytes(b'test-double identity only')
        pose = np.eye(4); pose[2, 3] = 1
        self.bundle = wn.MetricBundle('synthetic-family', Path('NEVER-READ-synthetic.glb'), 'a' * 64,
            1, 'integer_centers', pose, 'controlled_init:manufactured', 'f0', 'dataset_metric_cad')
        self.admission = wn.ServingAdmission(True, (), {'families': ['synthetic-family'],
            'contract_sha256': 'c' * 64, 'test_double_only': True})
        self.backends, self.stages, self.boundary_frames = [], [], []
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(wn, 'neural_preflight', return_value={'ready': True}))
        self.stack.enter_context(mock.patch.object(wn.MetricBundle, 'from_file', return_value=self.bundle))
        self.stack.enter_context(mock.patch.object(wn, 'NeuralArtwork', return_value=SimpleNamespace(
            record={'manufactured': True}, sha256='b' * 64, render=rendered_test_double)))
        self.stack.enter_context(mock.patch.object(wn, 'GoTrackMetricBackend', side_effect=self.make_backend))
        actual_sha = cli.sha256_file
        def identities(path):
            path = Path(path)
            if path == self.head_path:
                return wn.PRIMARY['export_sha256']
            if path == ROOT / wn.BASE_CHECKPOINT_RELATIVE:
                return wn.PRIMARY['base_checkpoint_sha256']
            return actual_sha(path)
        self.stack.enter_context(mock.patch.object(cli, 'sha256_file', side_effect=identities))
        @contextmanager
        def gpu_stage(role):
            self.stages.append(('enter', role))
            try:
                yield
            finally:
                self.stages.append(('exit', role))
        self.stack.enter_context(mock.patch.object(wt, 'gpu_stage', side_effect=gpu_stage))
        actual_process = fb.FrameBoundary.process
        def observed(boundary, packet):
            self.boundary_frames.append(packet.frame.frame_id)
            return actual_process(boundary, packet)
        self.stack.enter_context(mock.patch.object(fb.FrameBoundary, 'process', observed))

    def make_backend(self, role, bundle, **kwargs):
        backend = NumericalTestDouble(); backend.bundle = bundle
        backend.seed_target_frame_id = None; backend._seed_used = False
        backend.identity_extras = lambda K: {'principal_point_origin': bundle.principal_point_origin,
            'initialization': bundle.seed_label, 'seed_frame_id': bundle.seed_frame_id,
            'seed_target_frame_id': backend.seed_target_frame_id, 'glb_sha256': bundle.glb_sha256,
            'metric_units': 'metres', 'effective_K': wn.native_K(np.asarray(K, float).ravel(),
                bundle.principal_point_origin).ravel().tolist()}
        self.backends.append(backend)
        return backend

    def run_pipeline(self, *, overlays=True, roles=('primary_neural', 'frozen_baseline')):
        run = cli.build_run_spec(self.clip, mode='regression', roles=roles)
        options = cli.RunOptions(mask_set='manufactured', head_path=str(self.head_path),
            admission=self.admission, metric_bundle=str(self.bundle_path), write_overlays=overlays)
        result = cli.execute_run(self.ws, self.clip, run, options, progress=lambda _: None)
        return run, result, cli.load_results(self.ws, run.run_id)

    def test_main_neural_roles_use_boundary_and_unchanged_repeat_has_zero_estimator_or_stage_work(self):
        run, result, (_, rows, report) = self.run_pipeline()
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.backends), 2)
        self.assertEqual(self.boundary_frames, ['f1', 'f2', 'f3'] * 2)
        self.assertEqual([len(b.calls) for b in self.backends], [3, 3])
        self.assertTrue(all(b.closes == 1 for b in self.backends))
        self.assertEqual(len(rows), 4)
        counts = (len(self.backends), len(self.stages), len(self.boundary_frames))
        second, result, (_, replayed, report) = self.run_pipeline()
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual((len(self.backends), len(self.stages), len(self.boundary_frames)), counts)
        self.assertEqual(len(replayed), 4)
        self.assertTrue(all(r.evidence['execution_provenance'] == 'cache_replay_of_boundary_computed_rows'
                            for r in replayed))

    def test_no_overlay_option_is_honored_and_does_not_share_overlay_enabled_cache(self):
        run, result, (_, rows, _) = self.run_pipeline(overlays=False)
        self.assertEqual(result['status'], 'completed', result)
        self.assertFalse(list((self.ws.runs_dir / run.run_id / 'evidence').rglob('*.png')))
        self.assertTrue(all('overlay' not in r.evidence for r in rows))
        count = len(self.backends)
        run, result, (_, rows, _) = self.run_pipeline(overlays=True)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(self.backends), count + 2)
        self.assertEqual(len(list((self.ws.runs_dir / run.run_id / 'evidence').rglob('*.png'))), 4)

    def test_gpu_release_failure_retains_computed_rows_and_prevents_cache_commit(self):
        @contextmanager
        def failing_stage(role):
            try:
                yield
            finally:
                raise OSError('manufactured stage release failure')
        with mock.patch.object(wt, 'gpu_stage', side_effect=failing_stage), \
             mock.patch.object(cli.Pipeline, '_cache_store') as commit:
            run, result, (_, rows, _) = self.run_pipeline(roles=('primary_neural',))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.pose_state == 'tracking' for row in rows))
        commit.assert_not_called()

    def test_decoder_close_failure_is_visible_and_no_tracking_entry_is_committed(self):
        actual_close = cli.FrameReader.close
        def close_then_fail(reader):
            actual_close(reader)
            raise OSError('manufactured decoder release failure')
        with mock.patch.object(cli.FrameReader, 'close', close_then_fail), \
             mock.patch.object(cli.Pipeline, '_cache_store') as commit:
            run, result, (_, rows, report) = self.run_pipeline()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.pose_state == 'tracking' for row in rows))
        self.assertIn('decoder release failure', str(report))
        commit.assert_not_called()

    def test_cached_rows_must_match_manifest_units_model_and_role_overlay_path(self):
        run, result, (_, rows, _) = self.run_pipeline(roles=('primary_neural',))
        options = cli.RunOptions(mask_set='manufactured', admission=self.admission,
            metric_bundle=str(self.bundle_path), head_path=str(self.head_path))
        pipe = cli.Pipeline(self.ws, self.clip, run, options, progress=lambda _: None)
        context = fb.FrameContextV1.build(self.clip, run, 'primary_neural',
            fb.neural_binding(self.clip, run, family='synthetic-family', bundle=self.bundle))
        provenance = cli.boundary_provenance(context)
        for field, value in [('source_frame_index', 999999), ('timestamp_s', 999.),
                             ('pose_units', 'object_unit'), ('installed_head_sha256', '0' * 64),
                             ('overlay', 'evidence/frozen_baseline/f2.png')]:
            with self.subTest(field=field):
                cached = [row.to_dict() for row in rows]
                if field == 'installed_head_sha256':
                    cached[0]['backend_identity'][field] = value
                elif field == 'overlay':
                    cached[0]['evidence'][field] = value
                else:
                    cached[0][field] = value
                record = {'payload': {'execution': {**provenance, 'computed_by': 'fresh_boundary_execution'},
                    'results': cached}, 'files': {}}
                store = SimpleNamespace(lookup=lambda key: record)
                hit, invalid = pipe._adopted_lookup(store, 'manufactured', 'primary_neural', provenance)
                self.assertIsNotNone(invalid, 'An inconsistent cache row must fail before public replay')

    def relative_case(self):
        from bench.vision import cv2
        from bench import workbench_geometry as wg, workbench_cache as cache
        clip = wc.ClipSpec.from_dict(clip_document())
        run = cli.build_run_spec(clip, mode='regression', roles=('relative_geometry',))
        rng = np.random.default_rng(521)
        rgb = np.repeat(rng.integers(0, 256, (480, 640), np.uint8)[:, :, None], 3, axis=2)
        mask = np.zeros((480, 640), bool); mask[35:445, 110:530] = True
        keypoints, descriptors = cv2.ORB_create(nfeatures=1500).detectAndCompute(rgb[:, :, 0], mask.astype(np.uint8) * 255)
        self.assertGreater(len(keypoints), 100)
        K = clip.intrinsics.matrix(); pixels = np.array([k.pt for k in keypoints])
        points = np.column_stack(((pixels[:, 0] - K[0, 2]) / K[0, 0],
            (pixels[:, 1] - K[1, 2]) / K[1, 1], np.zeros(len(pixels)))).astype(np.float32)
        vertices = np.array([[-.4, -.4, 0], [.4, -.4, 0], [.4, .4, 0], [-.4, .4, 0]], np.float32)
        faces = np.array([[0, 2, 1], [0, 3, 2]], np.int32)
        pose = np.eye(4); pose[2, 3] = 1
        art_sha = wg.array_sha256(wg.builtin_artwork('checkerboard', 64))
        marker = wc.Marker.from_dict(wg.make_marker(vertices, faces,
            {'face_id': 0, 'barycentric': [.5, .25, .25]}, pose, .12, art_sha))
        package, raw, components = relative_proof_components(clip,
            {'descriptor_points': points, 'descriptor_values': descriptors, 'vertices': vertices,
             'faces': faces, 'face_support_views': np.array([3, 3], np.int32)}, marker=marker,
            artwork={'kind': 'checkerboard', 'edge_px': 64, 'sha256': art_sha, 'generation': 'fixed_builtin_v1'})
        folder = self.root / 'relative-package'; folder.mkdir()
        (folder / 'surface.json').write_bytes(raw)
        for key, data in components.items():
            (folder / getattr(package, key).path).write_bytes(data)
        reader = SimpleNamespace(read=lambda frame: (rgb, {'frame_id': frame.frame_id,
            'source_frame_index': frame.source_frame_index, 'decoder_media_time_s': frame.media_time_s,
            'image_sha256': wg.array_sha256(rgb)}))
        masks = SimpleNamespace(load=lambda fid, shape: (mask,
            {'mask_frame_id': fid, 'mask_sha256': wg.mask_sha256(mask)}))
        return clip, run, folder, reader, masks

    def test_relative_artifact_exception_keeps_computed_tracking_row_and_skips_cache(self):
        from bench.vision import cv2
        from bench import workbench_cache as cache
        clip, run, folder, reader, masks = self.relative_case()
        pipe = cli.Pipeline(self.ws, clip, run, cli.RunOptions(), progress=lambda _: None)
        pipe.run_dir.mkdir(parents=True)
        store = mock.Mock(); store.lookup.return_value = None
        with mock.patch.object(pipe, '_input_identity', return_value={}), \
             mock.patch.object(cv2, 'imwrite', side_effect=OSError('manufactured relative PNG failure')), \
             mock.patch.object(pipe, '_cache_store') as commit:
            try:
                pipe._track(reader, masks, folder, {'role': 'relative_geometry'}, (cache, store))
            except OSError:
                pass  # The assertion below distinguishes a lost computed row from accounting.
        self.assertEqual(len(pipe.results), 2)
        self.assertEqual(pipe.results[-1].pose_state, 'tracking')
        self.assertEqual(pipe.results[-1].render_state, 'rendered')
        self.assertIn('artifact_write_error', pipe.results[-1].evidence)
        self.assertNotIn('overlay', pipe.results[-1].evidence)
        commit.assert_not_called()

    def test_ordinary_cache_hits_reject_all_row_bundle_identity_changes_and_missing_fields(self):
        from bench import workbench_cache as cache
        run, result, (_, rows, _) = self.run_pipeline(overlays=False, roles=('primary_neural',))
        self.assertEqual(result['status'], 'completed')
        records = list((self.ws.derived_dir / 'stage-cache/entries').glob('*/record.json'))
        self.assertEqual(len(records), 1)
        original = json.loads(records[0].read_bytes())
        count = len(self.backends)
        for field, value in [('glb_sha256', '0' * 64), ('seed_target_frame_id', 'f3'),
                             ('effective_K', [1.] * 9), ('glb_sha256', None), ('effective_K', None)]:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                for row in changed['payload']['results']:
                    if value is None:
                        row['backend_identity'].pop(field)
                    else:
                        row['backend_identity'][field] = value
                with mock.patch.object(cache.StageCache, 'lookup', return_value=changed):
                    _, result, (_, rejected, _) = self.run_pipeline(overlays=False, roles=('primary_neural',))
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(all(r.pose is None for r in rejected))
                self.assertEqual(len(self.backends), count, 'Invalid entries must fail closed without recomputation')

    def test_source_verification_and_decoder_close_failures_are_both_in_final_saved_report(self):
        actual_close = cli.FrameReader.close
        def fail_close(reader):
            actual_close(reader)
            raise OSError('manufactured close after source failure')
        with mock.patch.object(cli.FrameReader, 'verify_media', side_effect=OSError('manufactured source refusal')), \
             mock.patch.object(cli.FrameReader, 'close', fail_close), \
             mock.patch.object(cli.Pipeline, '_cache_store') as commit:
            run, result, (_, rows, report) = self.run_pipeline()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r.pose is None for r in rows))
        self.assertIn('manufactured source refusal', str(report))
        self.assertIn('manufactured close after source failure', str(report))
        self.assertEqual(result['report'], report)
        commit.assert_not_called()

    def test_relative_tracking_cache_rejects_a_constant_wrong_package_identity(self):
        from bench import workbench_cache as cache
        clip, run, folder, reader, masks = self.relative_case()
        pipe = cli.Pipeline(self.ws, clip, run, cli.RunOptions(write_overlays=False), progress=lambda _: None)
        pipe.run_dir.mkdir(parents=True)
        store = mock.Mock(); store.lookup.return_value = None
        with mock.patch.object(pipe, '_input_identity', return_value={}):
            pipe._track(reader, masks, folder, {'role': 'relative_geometry'}, (cache, store))
        payload = copy.deepcopy(pipe._pending_commits[0][3])
        for row in payload['results']:
            row['backend_identity']['package_sha256'] = '0' * 64
        store.lookup.return_value = {'payload': payload, 'files': {}}
        second = cli.build_run_spec(clip, mode='regression', roles=('relative_geometry',))
        repeated = cli.Pipeline(self.ws, clip, second, cli.RunOptions(write_overlays=False), progress=lambda _: None)
        repeated.run_dir.mkdir(parents=True)
        with mock.patch.object(repeated, '_input_identity', return_value={}), \
             mock.patch.object(fb.FrameBoundary, 'process', side_effect=AssertionError('no recompute')):
            repeated._track(reader, masks, folder, {'role': 'relative_geometry'}, (cache, store))
        self.assertEqual(repeated.stages['relative_tracking']['status'], 'failed')
        self.assertTrue(all(r.pose is None for r in repeated.results))

