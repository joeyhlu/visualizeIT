"""Codex-owned portable-boundary controls, entirely manufactured and in memory.

The real NeuralSession and RelativeTracker are exercised. The neural estimator
is a named test double, never trained model evidence or a serving admission.
"""
import dataclasses
import hashlib
import io
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_frame_boundary as fb
from bench import workbench_geometry as wg
from bench import workbench_neural as wn
from bench.test_workbench_contract import clip_document, run_document


class NumericalTestDouble:
    def __init__(self):
        self.calls = []; self.resets = 0; self.closes = 0
        self.installed = wn.PRIMARY['export_sha256']

    def install_provenance(self):
        return {'head': {'installed_sha256': self.installed}}

    def estimate(self, rgb, mask, K, state):
        self.calls.append((state.get('_frame_id'), np.array(K, copy=True)))
        pose = np.eye(4); pose[2, 3] = 1
        return {'status': 'tracking', 'pose_m': pose, 'metrics': {}}

    def reset(self):
        self.resets += 1

    def close(self):
        self.closes += 1


def rendered_test_double(pose, mask, K):
    rgba = np.full((*mask.shape, 4), 173, np.uint8)
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    return {'rgba': rgba, 'stats': {'rendered_pixels': int(mask.sum())}}


def neural_fixture(*, clock_kind='timestamp_seconds', bundle=False):
    clip = wc.ClipSpec.from_dict(clip_document(clock_kind=clock_kind))
    run_doc = run_document()
    run_doc['primary'] = {key: wn.PRIMARY[key] for key in ('artifact_id', 'export_sha256', 'export_bytes')}
    run = wc.RunSpec.from_dict(run_doc)
    backend = NumericalTestDouble()
    metric = None
    if bundle:
        pose = np.eye(4); pose[2, 3] = 1
        metric = wn.MetricBundle('synthetic-family', Path('NEVER-READ-synthetic.glb'), 'a' * 64, 1,
            'integer_centers', pose, 'controlled_init:manufactured', 'f0', 'dataset_metric_cad')
        backend.bundle = metric
        backend.seed_target_frame_id = 'f1'
    admission = wn.ServingAdmission(True, (), {'families': ['synthetic-family'],
        'contract_sha256': 'c' * 64, 'test_double_only': True})
    session = wn.NeuralSession('primary_neural', backend, admission, run_id=run.run_id,
        clip=clip, family='synthetic-family', renderer=rendered_test_double)
    binding = fb.neural_binding(clip, run, family='synthetic-family', bundle=metric)
    context = fb.FrameContextV1.build(clip, run, 'primary_neural', binding)
    return context, fb.NeuralSessionAdapter(session), backend


def relative_proof_components(clip, arrays, *, marker=None, artwork=None, updates=None):
    components = {}
    for key, values in arrays.items():
        stream = io.BytesIO(); np.save(stream, values, allow_pickle=False)
        components[key] = stream.getvalue()
    ref = lambda key: wc.FileRef(key + '.npy', hashlib.sha256(components[key]).hexdigest(), len(components[key]))
    package = wc.SurfacePackage(package_id='synthetic-package', clip_id=clip.clip_id, status='available',
        unavailable_reason=None, units='object_unit', metric_scale_available=False,
        normalization={'center': [0, 0, 0], 'scale': 1, 'method': 'manufactured'},
        vertices=ref('vertices'), faces=ref('faces'), face_support_views=ref('face_support_views'),
        vertex_count=len(arrays['vertices']), face_count=len(arrays['faces']), cameras=(), descriptor_points=ref('descriptor_points'),
        descriptor_values=ref('descriptor_values'), admission={'passed': True, 'synthetic_only': True},
        marker=marker, artwork=artwork, frozen=True, frozen_after_frame_id='f1', prefix_frame_ids=('f0', 'f1'),
        reconstruction={'synthetic_only': True}, preview={})
    package = wc.SurfacePackage.from_dict(package.to_dict())
    if updates:
        package = wc.SurfacePackage.from_dict(dict(package.to_dict(), **updates))
    return package, wc.canonical_json(package.to_dict()), components


def relative_fixture(*, source_mode='recorded_finite'):
    clip = wc.ClipSpec.from_dict(clip_document())
    run_doc = run_document(); run_doc['roles'] = ['relative_geometry']
    run = wc.RunSpec.from_dict(run_doc)
    arrays = {'descriptor_points': np.column_stack((np.linspace(0, .1, 16), np.zeros(16), np.ones(16))).astype(np.float32),
        'descriptor_values': np.arange(16 * 32, dtype=np.uint8).reshape(16, 32),
        'vertices': np.array([[0, 0, 1], [.1, 0, 1], [0, .1, 1]], np.float32),
        'faces': np.array([[0, 1, 2]], np.int32), 'face_support_views': np.array([3], np.int32)}
    package, raw, components = relative_proof_components(clip, arrays)
    tracker = wg.RelativeTracker()
    proof = fb.RelativePackageProof.from_memory(raw, components)
    binding = fb.relative_binding(proof, artwork_kind=run.artwork_kind)
    context = fb.FrameContextV1.build(clip, run, 'relative_geometry', binding, source_mode=source_mode)
    return context, fb.RelativeTrackerAdapter(tracker, proof), tracker


def packet(context, frame_id=None, *, rgb=None, mask=True, **overrides):
    frame_id = frame_id or context.permitted_order[0]
    rgb = np.zeros((context.height, context.width, 3), np.uint8) if rgb is None else rgb
    observed = np.ones((context.height, context.width), bool) if mask is True else mask
    args = dict(image_sha256=wg.array_sha256(rgb), decoder_media_time_s=context.frame(frame_id).media_time_s,
        mask=observed, mask_frame_id=frame_id if observed is not None else None,
        mask_sha256=wg.mask_sha256(observed) if observed is not None else None)
    args.update(overrides)
    return fb.FrameInputV1.create(context.frame(frame_id), rgb, **args)


class PortableBoundaryTests(unittest.TestCase):
    def test_evaluator_seed_label_is_refused_for_direct_metric_bundle(self):
        context, adapter, backend = neural_fixture(bundle=True)
        backend.bundle = dataclasses.replace(backend.bundle, seed_label='evaluator:ground_truth_pose')
        self.addCleanup(adapter.close)
        with self.assertRaises(fb.BoundaryRejected):
            binding = fb.neural_binding(context.clip, context.run, family='synthetic-family', bundle=backend.bundle)
            forged = fb.FrameContextV1.build(context.clip, context.run, context.role, binding)
            fb.FrameBoundary().initialize(forged, adapter)
        self.assertEqual(backend.calls, [])

    def test_preloaded_neural_session_state_is_refused_without_reset_or_estimation(self):
        for field, value in (('_state', {'prior_pose': 'future-pose-sentinel', 'consecutive': 10}),
                             ('_last_index', 9999), ('last_overlay', np.zeros((1, 1, 4), np.uint8)),
                             ('backend._seed_used', True)):
            with self.subTest(field=field):
                context, adapter, backend = neural_fixture(bundle=True)
                if field == 'backend._seed_used':
                    backend._seed_used = value
                else:
                    setattr(adapter.session, field, value)
                boundary = fb.FrameBoundary()
                self.addCleanup(boundary.close)
                with self.assertRaises(fb.BoundaryRejected):
                    boundary.initialize(context, adapter)
                self.assertEqual((backend.calls, backend.resets), ([], 0))

    def open_neural(self, **kwargs):
        context, adapter, backend = neural_fixture(**kwargs)
        boundary = fb.FrameBoundary(); boundary.initialize(context, adapter)
        self.addCleanup(boundary.close)
        return context, boundary, backend

    def test_both_actual_sessions_return_valid_results_through_one_boundary(self):
        for make_fixture in (neural_fixture, relative_fixture):
            with self.subTest(adapter=make_fixture.__name__):
                context, adapter, backend = make_fixture()
                boundary = fb.FrameBoundary()
                try:
                    boundary.initialize(context, adapter)
                    output = boundary.process(packet(context))
                    parsed = wc.FrameResult.from_dict(output.result.to_dict())
                    self.assertEqual(parsed.frame_id, context.permitted_order[0])
                    self.assertEqual(parsed.backend_role, context.role)
                    self.assertEqual(output.phase, 'scored')
                    if parsed.pose is not None:
                        self.assertEqual(parsed.pose_units, 'metric' if context.role == 'primary_neural' else 'object_unit')
                finally:
                    boundary.close()

    def test_forged_frame_metadata_is_rejected_before_backend_or_cursor_changes(self):
        context, boundary, backend = self.open_neural()
        original = packet(context)
        for field, value in (('timestamp_s', .25), ('source_frame_index', 9999), ('media_time_s', .29)):
            forged = dataclasses.replace(original, frame=dataclasses.replace(original.frame, **{field: value}))
            with self.subTest(field=field), self.assertRaises(fb.BoundaryRejected):
                boundary.process(forged)
            self.assertEqual(boundary.cursor, 0)
        self.assertEqual(backend.calls, [])

    def test_skip_duplicate_and_exhaustion_never_call_backend(self):
        context, boundary, backend = self.open_neural()
        with self.assertRaises(fb.BoundaryRejected):
            boundary.process(packet(context, 'f3'))
        boundary.process(packet(context, 'f2'))
        with self.assertRaises(fb.BoundaryRejected):
            boundary.process(packet(context, 'f2'))
        boundary.process(packet(context, 'f3'))
        with self.assertRaises(fb.BoundaryRejected):
            boundary.process(packet(context, 'f3'))
        self.assertEqual([fid for fid, K in backend.calls], ['f2', 'f3'])

    def test_bad_image_hash_or_dimensions_count_failure_and_invalidate_without_reset(self):
        for bad in ('hash', 'dimensions'):
            with self.subTest(bad=bad):
                context, boundary, backend = self.open_neural()
                inp = packet(context, image_sha256='f' * 64) if bad == 'hash' else packet(context, rgb=np.zeros((2, 2, 3), np.uint8))
                output = boundary.process(inp)
                self.assertIsNone(output.overlay_rgba)
                self.assertIsNone(output.result.pose)
                later = boundary.process(packet(context, 'f3'))
                self.assertEqual(later.result.pose_state, 'blocked')
                self.assertEqual((backend.calls, backend.resets), ([], 0))

    def test_missing_rgb_counts_blocked_frame_and_never_rearms_seed(self):
        context, boundary, backend = self.open_neural(bundle=True)
        inp = fb.FrameInputV1.create(context.frame('f1'), None, rgb_unavailable_reason='synthetic missing RGB')
        output = boundary.process(inp)
        self.assertEqual(output.result.pose_state, 'blocked')
        self.assertEqual(boundary.process(packet(context, 'f2')).result.pose_state, 'blocked')
        self.assertEqual((backend.calls, backend.resets), ([], 0))
        self.assertEqual(backend.seed_target_frame_id, 'f1')

    def test_stale_missing_and_empty_masks_suppress_pose_and_current_overlay(self):
        for mask in ('missing', 'stale', 'empty'):
            with self.subTest(mask=mask):
                context, boundary, backend = self.open_neural()
                first = boundary.process(packet(context, 'f2'))
                self.assertIsNotNone(first.overlay_rgba)
                if mask == 'missing':
                    inp = packet(context, 'f3', mask=None)
                elif mask == 'empty':
                    inp = packet(context, 'f3', mask=np.zeros((480, 640), bool))
                else:
                    inp = packet(context, 'f3', mask_frame_id='f2')
                output = boundary.process(inp)
                self.assertIsNone(output.overlay_rgba)
                self.assertIsNone(output.result.pose)
                self.assertEqual(len(backend.calls), 1)

    def test_reset_replays_original_warmup_and_preserves_seed_target(self):
        context, boundary, backend = self.open_neural(bundle=True)
        self.assertEqual(context.warmup_frame_ids, ('f1',))
        self.assertEqual(boundary.process(packet(context, 'f1')).phase, 'warmup')
        boundary.process(packet(context, 'f2'))
        boundary.reset(reason='replay_restart')
        self.assertEqual((boundary.epoch, boundary.cursor, boundary.expected_next), (1, 0, 'f1'))
        self.assertEqual((backend.resets, backend.seed_target_frame_id), (1, 'f1'))
        with self.assertRaises(fb.BoundaryRejected):
            boundary.process(packet(context, 'f2'))
        self.assertEqual(boundary.process(packet(context, 'f1')).epoch, 1)

    def test_output_and_input_arrays_are_owned_and_cannot_leak_stale_overlays(self):
        context, boundary, backend = self.open_neural()
        rgb = np.zeros((480, 640, 3), np.uint8)
        inp = packet(context, 'f2', rgb=rgb)
        rgb[:] = 255
        first = boundary.process(inp)
        first.overlay_rgba[:] = 0
        second = boundary.process(packet(context, 'f3'))
        self.assertTrue(np.any(second.overlay_rgba))
        self.assertIsNone(boundary._adapter.session.last_overlay)

    def test_camera_style_neural_is_explicitly_unsupported(self):
        context, adapter, backend = neural_fixture()
        with self.assertRaises(fb.LiveNeuralUnsupported):
            fb.FrameContextV1.build(context.clip, context.run, context.role, context.binding,
                source_mode='camera_style_synthetic')
        adapter.close()

    def test_camera_style_relative_uses_same_timestamped_software_protocol(self):
        context, adapter, tracker = relative_fixture(source_mode='camera_style_synthetic')
        boundary = fb.FrameBoundary()
        try:
            boundary.initialize(context, adapter)
            self.assertIsInstance(boundary.process(packet(context)).result, wc.FrameResult)
        finally:
            boundary.close()

    def test_lifecycle_close_is_idempotent_and_processing_outside_lifecycle_fails(self):
        context, adapter, backend = neural_fixture()
        boundary = fb.FrameBoundary()
        with self.assertRaises(fb.BoundaryStateError):
            boundary.process(packet(context))
        boundary.initialize(context, adapter)
        boundary.close(); boundary.close()
        self.assertEqual(backend.closes, 1)
        with self.assertRaises(fb.BoundaryStateError):
            boundary.process(packet(context))

    def test_context_calibration_cannot_be_forged_by_replacing_dataclass_fields(self):
        context, adapter, backend = neural_fixture()
        changed_K = list(context.K); changed_K[0] *= 2
        forged = dataclasses.replace(context, K=tuple(changed_K))
        boundary = fb.FrameBoundary()
        try:
            with self.assertRaises(fb.BoundaryError):
                boundary.initialize(forged, adapter)
        finally:
            boundary.close(); adapter.close()
        self.assertEqual(backend.calls, [])

    def test_bad_context_schema_is_rejected_at_initialize(self):
        context, adapter, backend = neural_fixture()
        boundary = fb.FrameBoundary()
        try:
            with self.assertRaises(fb.BoundaryError):
                boundary.initialize(dataclasses.replace(context, schema='unsupported-v999'), adapter)
        finally:
            boundary.close(); adapter.close()

    def test_index_clock_never_accepts_a_fabricated_timestamp(self):
        context, boundary, backend = self.open_neural(clock_kind='frame_index')
        inp = packet(context)
        with self.assertRaises(fb.BoundaryRejected):
            boundary.process(dataclasses.replace(inp, frame=dataclasses.replace(inp.frame, timestamp_s=.2)))
        self.assertEqual(backend.calls, [])

    def test_frame_metadata_cannot_supply_calibration_or_admission(self):
        context, adapter, backend = neural_fixture()
        for key in ('K', 'intrinsics', 'admission', 'family'):
            with self.subTest(key=key), self.assertRaises(fb.BoundaryRejected):
                packet(context, metadata={key: 'forged'})
        adapter.close()

    def test_public_packet_schema_is_checked_again_before_backend_and_cursor(self):
        context, boundary, backend = self.open_neural()
        with self.assertRaises(fb.BoundaryError):
            boundary.process(dataclasses.replace(packet(context), schema='unsupported-v999'))
        self.assertEqual((backend.calls, boundary.cursor), ([], 0))

    def test_replaced_packet_metadata_cannot_bypass_reserved_calibration_check(self):
        context, boundary, backend = self.open_neural()
        forged = dataclasses.replace(packet(context), metadata_json=wc.canonical_json({'K': [1] * 9}))
        with self.assertRaises(fb.BoundaryError):
            boundary.process(forged)
        self.assertEqual((backend.calls, boundary.cursor), ([], 0))

    def test_changed_seed_pose_before_initialization_is_rejected(self):
        context, adapter, backend = neural_fixture(bundle=True)
        backend.bundle.seed_pose_m[0, 3] = 123
        boundary = fb.FrameBoundary()
        try:
            with self.assertRaises(fb.BoundaryError):
                boundary.initialize(context, adapter)
        finally:
            boundary.close(); adapter.close()

    def test_changed_object_id_before_initialization_is_rejected(self):
        context, adapter, backend = neural_fixture(bundle=True)
        backend.bundle = dataclasses.replace(backend.bundle, obj_id=99)
        boundary = fb.FrameBoundary()
        try:
            with self.assertRaises(fb.BoundaryError):
                boundary.initialize(context, adapter)
        finally:
            boundary.close(); adapter.close()

    def test_seed_pose_drift_after_initialization_stops_before_estimate(self):
        context, boundary, backend = self.open_neural(bundle=True)
        backend.bundle.seed_pose_m[0, 3] = 123
        with self.assertRaises((fb.BoundaryError, fb.BoundaryFailure)):
            boundary.process(packet(context, 'f1'))
        self.assertEqual((backend.calls, backend.resets), ([], 0))

    def test_admission_contents_drift_stops_before_estimate(self):
        context, boundary, backend = self.open_neural()
        boundary._adapter.session.admission.body['families'] = []
        with self.assertRaises((fb.BoundaryError, fb.BoundaryFailure)):
            boundary.process(packet(context))
        self.assertEqual(backend.calls, [])

    def test_run_requested_head_must_agree_with_the_actual_primary_identity(self):
        context, adapter, backend = neural_fixture()
        primary = dict(context.run.primary, export_sha256='f' * 64)
        boundary = fb.FrameBoundary()
        try:
            with self.assertRaises(fb.BoundaryError):
                forged = fb.FrameContextV1.build(context.clip, dataclasses.replace(context.run, primary=primary),
                    context.role, context.binding)
                boundary.initialize(forged, adapter)
        finally:
            boundary.close(); adapter.close()

    def test_decoder_pts_is_required_when_declared_and_must_be_finite_nonnegative(self):
        for value in (None, -1, float('nan'), float('inf')):
            with self.subTest(value=value):
                context, boundary, backend = self.open_neural()
                forged = dataclasses.replace(packet(context), decoder_media_time_s=value)
                with self.assertRaises(fb.BoundaryError):
                    boundary.process(forged)
                self.assertEqual((backend.calls, boundary.cursor), ([], 0))

    def test_context_mutation_after_initialize_closes_before_backend(self):
        context, boundary, backend = self.open_neural()
        context.run.config['changed_after_init'] = True
        with self.assertRaises(fb.ContextMutated):
            boundary.process(packet(context))
        self.assertEqual((backend.calls, backend.closes), ([], 1))

    def test_unserializable_adapter_open_record_closes_transferred_resources(self):
        context, adapter, backend = neural_fixture()
        original = adapter.open
        def bad_open(ctx):
            original(ctx)
            return {'bad': object()}
        boundary = fb.FrameBoundary()
        with mock.patch.object(adapter, 'open', side_effect=bad_open):
            with self.assertRaises(fb.BoundaryError):
                boundary.initialize(context, adapter)
        self.assertEqual(backend.closes, 1)

    def test_failed_context_initialization_reports_cleanup_failure(self):
        context, adapter, backend = neural_fixture()
        def broken_close():
            backend.closes += 1
            raise OSError('manufactured initialization cleanup failure')
        backend.close = broken_close
        boundary = fb.FrameBoundary()
        with self.assertRaises(fb.BoundaryError):
            boundary.initialize(dataclasses.replace(context, schema='bad-schema'), adapter)
        self.assertEqual(backend.closes, 1)
        self.assertIn('initialization cleanup failure', str(boundary.cleanup_errors))

    def test_relative_proof_refuses_changed_component_bytes_metric_relabel_and_future_prefix(self):
        context, adapter, tracker = relative_fixture()
        package, arrays, digests = adapter.proof.verify()
        _, raw, components = relative_proof_components(context.clip, arrays)
        changed = dict(components); changed['descriptor_points'] = changed['descriptor_points'][:-1] + b'x'
        with self.assertRaises(fb.BoundaryError):
            fb.RelativePackageProof.from_memory(raw, changed).verify()
        for updates in ({'units': 'metric', 'metric_scale_available': True},
                        {'frozen_after_frame_id': 'f3', 'prefix_frame_ids': ['f2', 'f3']}):
            _, raw, comps = relative_proof_components(context.clip, arrays, updates=updates)
            proof = fb.RelativePackageProof.from_memory(raw, comps)
            with self.subTest(updates=updates), self.assertRaises(fb.BoundaryError):
                binding = fb.relative_binding(proof, artwork_kind=context.run.artwork_kind)
                fb.FrameContextV1.build(context.clip, context.run, context.role, binding)
        adapter.close()

    def test_real_relative_tracker_renders_supported_marker_and_suppresses_loss_and_recovery(self):
        """Known textured plane: actual ORB/LK/PnP and the actual mesh renderer.

        This is manufactured numerical coverage, not real-video attachment evidence.
        """
        from bench.vision import cv2
        cv2.setRNGSeed(521)
        clip = wc.ClipSpec.from_dict(clip_document(frame_count=7))
        run_doc = run_document(scheduled=('f2', 'f3', 'f4', 'f5', 'f6'))
        run_doc['roles'] = ['relative_geometry']
        run = wc.RunSpec.from_dict(run_doc)
        K = np.array(clip.intrinsics.K).reshape(3, 3)
        gray = np.random.default_rng(521).integers(0, 256, (480, 640), dtype=np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        observed = np.zeros(gray.shape, bool); observed[35:445, 110:530] = True
        keys, descriptors = cv2.ORB_create(nfeatures=1500).detectAndCompute(gray, observed.astype(np.uint8) * 255)
        self.assertGreater(len(keys), 100)
        pixels = np.array([key.pt for key in keys], np.float64)
        points = np.column_stack(((pixels[:, 0] - K[0, 2]) / K[0, 0],
                                  (pixels[:, 1] - K[1, 2]) / K[1, 1], np.zeros(len(keys)))).astype(np.float32)
        vertices = np.array([[-.4, -.4, 0], [.4, -.4, 0], [.4, .4, 0], [-.4, .4, 0]], np.float32)
        faces = np.array([[0, 2, 1], [0, 3, 2]], np.int32)
        art = wg.builtin_artwork('checkerboard', 64); art_sha = wg.array_sha256(art)
        pose = np.eye(4); pose[2, 3] = 1
        marker_doc = wg.make_marker(vertices, faces, {'face_id': 0, 'barycentric': [.5, .25, .25]}, pose, .12, art_sha)
        marker = wc.Marker.from_dict(marker_doc)
        artwork = {'kind': 'checkerboard', 'edge_px': 64, 'sha256': art_sha, 'generation': 'fixed_builtin_v1'}
        arrays = {'descriptor_points': points, 'descriptor_values': descriptors,
                  'vertices': vertices, 'faces': faces, 'face_support_views': np.array([3, 3], np.int32)}
        package, raw, components = relative_proof_components(clip, arrays, marker=marker, artwork=artwork)
        proof = fb.RelativePackageProof.from_memory(raw, components)
        context = fb.FrameContextV1.build(clip, run, 'relative_geometry',
            fb.relative_binding(proof, artwork_kind='checkerboard'))
        tracker = wg.RelativeTracker()
        boundary = fb.FrameBoundary(); boundary.initialize(context, fb.RelativeTrackerAdapter(tracker, proof))
        self.addCleanup(boundary.close)
        results = []
        for fid in run.scheduled_frame_ids:
            results.append(boundary.process(packet(context, fid, rgb=rgb, mask=None if fid == 'f4' else observed)))
        # No observed mask means the public pose is unavailable; recovery must
        # still be confirmed on subsequent observed frames before rendering.
        self.assertEqual([out.result.pose_state for out in results], ['recovering', 'tracking', 'unavailable', 'recovering', 'tracking'])
        self.assertIsNone(results[0].overlay_rgba)
        self.assertIsNone(results[2].overlay_rgba)
        self.assertIsNone(results[3].overlay_rgba)
        anchor = np.array(marker_doc['barycentric']) @ vertices[faces[0]]
        expected_pixel = [K[0, 0] * anchor[0] + K[0, 2], K[1, 1] * anchor[1] + K[1, 2]]
        for out in (results[1], results[4]):
            self.assertEqual(out.result.render_state, 'rendered')
            self.assertEqual(out.result.pose_units, 'object_unit')
            rgba = out.overlay_rgba
            self.assertEqual((rgba.dtype, rgba.shape), (np.dtype('uint8'), (480, 640, 4)))
            ys, xs = np.nonzero(rgba[:, :, 3])
            self.assertGreater(len(xs), 100)
            self.assertTrue(observed[ys, xs].all())
            np.testing.assert_allclose([xs.mean(), ys.mean()], expected_pixel, atol=2)
