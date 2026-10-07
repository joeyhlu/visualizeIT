"""CPU fixtures for the frozen bottle correspondence identity diagnostic."""
import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_identity_probe as probe
from .quality_bottle_identity_probe import _planned_conditions
from .quality_contract import Frame
from .vision import cv2


def _pose(rvec, tvec):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    value[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return value


def _manual_patch_bank(image, positions, object_points):
    gray = audit._gray_image(image)
    high = audit._highpass(gray, 2.)
    anchors = []
    for (x, y), xyz in zip(positions, object_points):
        index = y * image.shape[1] + x
        patch = image[y-audit.PATCH_RADIUS:y+audit.PATCH_RADIUS+1,
                      x-audit.PATCH_RADIUS:x+audit.PATCH_RADIUS+1].copy()
        descriptor, std = audit._normalize_patch(
            high[y-audit.PATCH_RADIUS:y+audit.PATCH_RADIUS+1,
                 x-audit.PATCH_RADIUS:x+audit.PATCH_RADIUS+1])
        if descriptor is None:
            raise AssertionError('Test patch unexpectedly has low texture')
        anchors.append(dict(
            source_index=int(index), source_xy_crop=[float(x + .5), float(y + .5)],
            object_xyz_m=np.asarray(xyz, dtype=np.float64).tolist(),
            highpass_std=float(std), descriptor=descriptor, patch_rgb=patch,
        ))
    return dict(anchors=anchors, selected_anchor_count=len(anchors))


def _decision_fixture(gray_template_improvement=False, false_acceptance=None,
                      uninformative_query=None, uninformative_template=None):
    synthetic = []
    contexts = []
    false_acceptances = set()
    if false_acceptance is not None:
        if len(false_acceptance) == 5 and not isinstance(false_acceptance[0], tuple):
            false_acceptances.add(tuple(false_acceptance))
        else:
            false_acceptances.update(tuple(row) for row in false_acceptance)
    for frame_id in audit.SYNTHETIC_CARRIERS:
        for query_offset in audit.QUERY_OFFSETS:
            state = ('appearance_ablation_uninformative'
                     if uninformative_query == (frame_id, query_offset) else 'informative')
            contexts.append(dict(
                context_id=f'synthetic-query-{frame_id}-{query_offset}', role='synthetic_query',
                frame_id=frame_id, query_offset_deg=query_offset,
                query_appearance_ablation=dict(state=state),
            ))
    for frame_id in audit.FRAME_IDS:
        for template_offset in audit.TEMPLATE_OFFSETS:
            state = ('appearance_ablation_uninformative'
                     if uninformative_template == (frame_id, template_offset) else 'informative')
            contexts.append(dict(
                context_id=f'template-{frame_id}-{template_offset}', role='template',
                frame_id=frame_id, template_offset_deg=template_offset,
                appearance_ablation=dict(state=state),
            ))
    for query_offset in audit.QUERY_OFFSETS:
        near_offset = 0 if query_offset == 8 else 180
        for query_appearance in audit.APPEARANCES:
            for frame_id in audit.SYNTHETIC_CARRIERS:
                for template_offset in audit.TEMPLATE_OFFSETS:
                    for template_appearance in audit.APPEARANCES:
                        correct = 95
                        if (gray_template_improvement and query_offset == 8 and
                                query_appearance == 'gray' and template_offset == near_offset):
                            correct = 50 if template_appearance == 'rgb' else 80
                        identity = dict(
                            truly_visible_sources=100, confidence_gt_03_visible=100,
                            correct_confident_visible=correct,
                            confident_invisible_source_endpoints=0,
                            confident_masked_correspondences=100,
                        )
                        negative = dict(state='below_frozen_negative_gate')
                        if (frame_id, query_offset, query_appearance,
                                template_offset, template_appearance) in false_acceptances:
                            negative = dict(state='validated_wrong_surface_false_acceptance')
                        synthetic.append(dict(
                            condition_id=(f'syn-{frame_id}-{query_offset}-{query_appearance}-'
                                          f'{template_offset}-{template_appearance}'),
                            kind='synthetic', frame_id=frame_id, state='captured',
                            query_offset_deg=query_offset, query_appearance=query_appearance,
                            template_offset_deg=template_offset,
                            template_appearance=template_appearance,
                            synthetic_identity=identity,
                            validated_wrong_surface_false_acceptance=negative,
                            patch_identity=dict(
                                state='distinctive_current_image_support', source_eligible=16,
                                supported=12, supported_cells_4x4=4, supported_hull_fraction=.5),
                        ))
    real = []
    for frame_id in audit.FRAME_IDS:
        for template_offset in audit.TEMPLATE_OFFSETS:
            for template_appearance in audit.APPEARANCES:
                supported = 8 if template_appearance == 'rgb' else 12
                real.append(dict(
                    condition_id=f'real-{frame_id}-{template_offset}-{template_appearance}',
                    kind='real', frame_id=frame_id, state='captured',
                    template_offset_deg=template_offset, template_appearance=template_appearance,
                    patch_identity=dict(
                        state='distinctive_current_image_support', source_eligible=16,
                        source_eligible_before_cap=16, target_eligible=8, rejected=0,
                        supported=supported, supported_cells_4x4=4, supported_hull_fraction=.5),
                    learned_pnp_contract=dict(validation_state='accepted',
                                              validation_stats=dict(score=1.)),
                ))
    report = dict(
        synthetic=synthetic, real=real,
        oracle_pnp_controls=[dict(state='passed') for _ in range(8)],
        invariants=dict(repeat_hashes_match=True, shared_camera_depth_masks_match=True,
                        projection_parity_passed=True, real_records_leak_free=True),
    )
    return report, dict(contexts=contexts), {frame_id: [] for frame_id in audit.SYNTHETIC_CARRIERS}


class BottleIdentityAuditTests(unittest.TestCase):
    def test_probe_import_has_no_neural_or_inference_import_at_module_scope(self):
        source = Path(__file__).with_name('quality_bottle_identity_probe.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                self.assertNotIn('torch', {alias.name.split('.')[0] for alias in statement.names})
            elif isinstance(statement, ast.ImportFrom):
                self.assertNotEqual(statement.module, 'torch')
                self.assertNotIn('quality_gotrack', statement.module or '')

    def test_frozen_forward_plan_is_exactly_64_synthetic_40_real_and_four_repeats(self):
        conditions = _planned_conditions()
        self.assertEqual(len(conditions), 108)
        self.assertEqual(sum(row['kind'] == 'synthetic' for row in conditions), 64)
        self.assertEqual(sum(row['kind'] == 'real' for row in conditions), 40)
        repeats = [row for row in conditions if row['kind'] == 'repeat']
        self.assertEqual(len(repeats), 4)
        self.assertEqual({row['repeat_of'] for row in repeats}, {
            row['condition_id'] for row in conditions if row['kind'] == 'real' and row['frame_id'] == 10
        })
        self.assertEqual([row['forward_index'] for row in conditions], list(range(108)))
        capture = dict(
            schema_version=1, object='ranch', requested_real_frame_ids=list(audit.FRAME_IDS),
            requested_synthetic_carriers=list(audit.SYNTHETIC_CARRIERS),
            planned_forward_cap=108, packet_budget_bytes=audit.PACKET_BUDGET_BYTES,
            packet_bytes=0, conditions=[dict(row, state='unavailable') for row in conditions],
            contexts=[], forwards=[], reference_or_annotations_loaded=False, complete=True,
        )
        audit._validate_capture_manifest(capture)
        capture['conditions'][0]['template_offset_deg'] = 90
        with self.assertRaisesRegex(ValueError, 'condition matrix'):
            audit._validate_capture_manifest(capture)

    def test_cpu_audit_enumerates_unavailable_rows_without_dropping_conditions(self):
        conditions = [dict(row, state='unavailable', reason='fixture_input_unavailable')
                      for row in _planned_conditions()]
        capture = dict(
            schema_version=1, object='ranch', status='complete', complete=True,
            requested_real_frame_ids=list(audit.FRAME_IDS),
            requested_synthetic_carriers=list(audit.SYNTHETIC_CARRIERS),
            planned_forward_cap=audit.FORWARD_CAP,
            packet_budget_bytes=audit.PACKET_BUDGET_BYTES, packet_bytes=0,
            reference_or_annotations_loaded=False, contexts=[], forwards=[], conditions=conditions,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'capture'
            root.mkdir()
            (root / 'capture.json').write_text(json.dumps(capture), encoding='utf-8')
            report = audit.audit_capture(root, Path(temporary) / 'audit')
        self.assertEqual(report['status'], 'invariants_failed')
        self.assertTrue(report['complete'])  # The audit completed; the capture carries 108 explicit unavailable rows.
        self.assertEqual(len(report['forwards']), 108)
        self.assertEqual(len(report['synthetic']), 64)
        self.assertEqual(len(report['real']), 40)
        self.assertEqual(len(report['decisions']['oracle_pnp_controls']['rows']), 8)
        self.assertEqual(report['decisions']['oracle_pnp_controls']['unavailable'], 8)
        self.assertTrue(all(row['state'] == 'unavailable' for row in report['forwards']))

    def test_nonidentity_crop_transform_roundtrips_metres_and_millimetres(self):
        native_pose_m = _pose([.13, -.08, .04], [.27, -.11, .82])
        crop_from_native = _pose([-.06, .09, .025], [13., -7., 4.])
        crop_pose_mm = audit.native_to_crop_pose_mm(native_pose_m, crop_from_native)
        recovered = audit.crop_to_native_pose_m(crop_pose_mm, crop_from_native)
        np.testing.assert_allclose(recovered, native_pose_m, atol=1e-12, rtol=0)
        native_pose_mm = native_pose_m.copy()
        native_pose_mm[:3, 3] *= 1000.
        self.assertAlmostEqual(crop_pose_mm[2, 3],
                               (crop_from_native @ native_pose_mm)[2, 3],
                               places=8)

    def test_depth_unprojection_uses_pixel_centres_and_object_pose_in_millimetres(self):
        depth = np.full((3, 4), 1250., dtype=np.float64)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        k = np.array([[7., 0., 2.], [0., 8., 1.5], [0., 0., 1.]])
        crop_pose_mm = _pose([.1, -.16, .06], [12., -8., 920.])
        ids, pixels, points = audit.object_points_from_depth(depth, k, crop_pose_mm, mask)
        self.assertEqual(ids.tolist(), [6])
        np.testing.assert_array_equal(pixels, [[2.5, 1.5]])
        ray = np.linalg.inv(k) @ np.array([2.5, 1.5, 1.])
        camera_mm = ray * 1250.
        expected_m = (camera_mm - crop_pose_mm[:3, 3]) @ crop_pose_mm[:3, :3] * .001
        np.testing.assert_allclose(points[0], expected_m, atol=1e-12, rtol=0)

    def test_same_2d_projection_can_be_a_wrong_3d_surface_identity(self):
        height = width = 25
        k = np.array([[20., 0., 12.5], [0., 20., 12.5], [0., 0., 1.]])
        query_pose = np.eye(4)
        points = np.array([[0., 0., 1.5]])
        source_pixels = np.array([[12.5, 12.5]])
        source_ids = np.array([12 * width + 12])
        endpoints = source_pixels.copy()  # Perfect 2D endpoint match at the visible front surface.
        confidence = np.array([.95])
        depth = np.zeros((height, width), dtype=np.float32)
        depth[12, 12] = 1000.
        mask = np.zeros((height, width), dtype=bool)
        mask[12, 12] = True
        summary, arrays = audit.synthetic_identity_metrics(
            points, source_pixels, source_ids, endpoints, confidence, query_pose,
            depth, mask, k, diagonal_m=1.)
        self.assertEqual(float(arrays['endpoint_error_px'][0]), 0.)
        self.assertFalse(bool(arrays['true_visible'][0]))  # 1.5 m source is occluded by the 1 m surface.
        self.assertGreater(float(arrays['identity_distance_fraction'][0]), .1)
        self.assertEqual(summary['confident_masked_correspondences'], 1)
        self.assertEqual(summary['confident_invisible_source_endpoints'], 1)
        self.assertEqual(summary['correct_all_eligible'], 0)

    def test_shared_visible_surface_passes_and_mask_failure_is_counted(self):
        k = np.array([[20., 0., 12.5], [0., 20., 12.5], [0., 0., 1.]])
        pose = np.eye(4)
        points = np.array([[0., 0., 1.]])
        xy = np.array([[12.5, 12.5]])
        ids = np.array([12 * 25 + 12])
        depth = np.zeros((25, 25), dtype=np.float32)
        depth[12, 12] = 1000.
        mask = np.zeros((25, 25), dtype=bool)
        mask[12, 12] = True
        summary, arrays = audit.synthetic_identity_metrics(points, xy, ids, xy, np.array([.9]),
                                                            pose, depth, mask, k, 1.)
        self.assertTrue(bool(arrays['true_visible'][0]))
        self.assertTrue(bool(arrays['correct_identity'][0]))
        self.assertEqual(summary['correct_confident_visible'], 1)
        occluded = mask.copy(); occluded[12, 12] = False
        rejected, rejected_arrays = audit.synthetic_identity_metrics(
            points, xy, ids, xy, np.array([.9]), pose, depth, occluded, k, 1.)
        self.assertFalse(bool(rejected_arrays['true_visible'][0]))
        self.assertFalse(bool(rejected_arrays['predicted_target_valid'][0]))
        self.assertEqual(rejected['confident_masked_correspondences'], 0)
        self.assertEqual(rejected['predicted_endpoint_foreground'], 0)

    def test_oracle_pnp_recovers_known_synthetic_pose_and_accounts_for_too_few_points(self):
        xy_object = np.array([(x, y) for x in np.linspace(-.2, .2, 20)
                              for y in np.linspace(-.2, .2, 20)])
        points = np.column_stack((xy_object, .025 * np.sin(xy_object[:, 0] * 12.)))
        k = np.array([[400., 0., 140.], [0., 405., 140.], [0., 0., 1.]])
        near_pose = _pose([0., 0., 0.], [0., 0., .9])
        query_pose = _pose([.018, -.012, .006], [.004, -.003, .91])
        source_pixels = audit.project_object_points(points, near_pose, k)[0]
        endpoints = audit.project_object_points(points, query_pose, k)[0]
        result = audit.oracle_pnp(points, endpoints, np.ones(len(points), dtype=bool),
                                  source_pixels, near_pose, query_pose, k, .7, 238)
        self.assertEqual(result['state'], 'passed', result)
        self.assertGreaterEqual(result['available_points'], 128)
        self.assertLessEqual(result['rotation_error_degrees'], .1)
        self.assertLessEqual(result['translation_error_m'], .0007)
        small = audit.oracle_pnp(points[:20], endpoints[:20], np.ones(20, dtype=bool),
                                 source_pixels[:20], near_pose, query_pose, k, .7, 238)
        self.assertEqual(small['state'], 'unavailable')
        self.assertEqual(small['reason'], 'fewer_than_128_spatially_distributed_visible_points')

    def test_learned_packet_fits_only_retained_pairs_and_applies_contract(self):
        object_xy = np.array([(x, y) for x in np.linspace(-.2, .2, 30)
                              for y in np.linspace(-.2, .2, 30)])
        points = np.column_stack((object_xy, .015 * np.sin(object_xy[:, 0] * 9.)))
        k = np.array([[400., 0., 140.], [0., 405., 140.], [0., 0., 1.]])
        pose = _pose([.012, -.008, .003], [.002, -.001, .9])
        endpoints = audit.project_object_points(points, pose, k)[0]
        count = len(points)
        source_ids = np.arange(2000, 2000 + count, dtype=np.int64)
        mask = np.ones((280, 280), dtype=bool)
        frame = Frame(10, np.zeros((280, 280, 3), dtype=np.uint8), k)
        result = audit.fit_learned_packet(
            points, endpoints, np.full(count, .95), source_ids, mask, mask,
            pose, k, np.eye(4), k, frame, np.ones((280, 280), dtype=np.uint8), 227)
        self.assertEqual(result['state'], 'refined')
        self.assertEqual(result['retained_correspondences'], count)
        self.assertEqual(result['validation_state'], 'accepted', result['validation_stats'])
        self.assertEqual(result['candidate_arrays']['points_object_m']['shape'], [count, 3])
        low = audit.fit_learned_packet(
            points[:5], endpoints[:5], np.full(5, .95), source_ids[:5], mask, mask,
            pose, k, np.eye(4), k, frame, np.ones((280, 280), dtype=np.uint8), 227)
        self.assertEqual(low['state'], 'unavailable')
        self.assertIsNone(low['pose_native_m'])

    def test_fixed_patch_bank_is_deterministic_and_bounded(self):
        rng = np.random.default_rng(8128)
        gray = rng.random((280, 280), dtype=np.float32)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        mask = np.ones((280, 280), dtype=bool)
        ids = np.arange(280 * 280, dtype=np.int64)
        yy, xx = np.indices((280, 280))
        points = np.column_stack((xx.reshape(-1) * .002, yy.reshape(-1) * .002,
                                  np.zeros(280 * 280)))
        first = audit.build_template_patch_bank(image, mask, ids, points)
        second = audit.build_template_patch_bank(image, mask, ids, points)
        self.assertEqual(first['selected_anchor_count'], 64)
        self.assertEqual(first['serialized_anchors'], second['serialized_anchors'])
        self.assertLessEqual(first['selected_anchor_count'], 64)
        ablated, info = audit.gray_rgb(image, mask)
        self.assertTrue(np.array_equal(ablated[:, :, 0], ablated[:, :, 1]))
        self.assertTrue(np.array_equal(ablated[:, :, 1], ablated[:, :, 2]))
        self.assertEqual(info['foreground_pixels'], mask.size)
        color = rng.random((64, 64, 3), dtype=np.float32)
        foreground = np.zeros((64, 64), dtype=bool)
        foreground[8:56, 8:56] = True
        gray_color, _ = audit.gray_rgb(color, foreground)
        self.assertTrue(np.array_equal(color[~foreground], gray_color[~foreground]))
        self.assertTrue(np.allclose(gray_color[foreground, 0], gray_color[foreground, 1], atol=1e-6))
        self.assertTrue(np.allclose(gray_color[foreground, 1], gray_color[foreground, 2], atol=1e-6))

    def test_distinct_texture_is_supported_but_repeated_and_blank_are_ambiguous(self):
        rng = np.random.default_rng(7781)
        base = rng.random((280, 280), dtype=np.float32)
        texture = np.repeat(base[:, :, None], 3, axis=2)
        mask = np.zeros((280, 280), dtype=bool)
        mask[15:265, 15:265] = True
        positions = [(40 + 60 * x, 40 + 60 * y) for y in range(4) for x in range(4)]
        object_points = [(float(x) * .25, float(y) * .25, 0.)
                         for y in range(4) for x in range(4)]
        bank = _manual_patch_bank(texture, positions, object_points)
        confidence = np.zeros(mask.shape, dtype=np.float32)
        for x, y in positions:
            confidence[y, x] = .9
        flow = np.zeros((*mask.shape, 2), dtype=np.float32)
        support = audit.audit_patch_identity(texture, mask, flow, confidence, bank, [bank], 1.)
        self.assertEqual(support['summary']['state'], 'distinctive_current_image_support')
        self.assertGreaterEqual(support['summary']['supported'], 8)
        self.assertGreaterEqual(support['summary']['supported_cells_4x4'], 3)
        self.assertTrue(all(row['source_object_xyz_m'] and row['target_xy_crop'] for row in support['rows']))

        yy, xx = np.indices((280, 280))
        checker = (((xx + yy) % 2).astype(np.float32))
        repeated = np.repeat(checker[:, :, None], 3, axis=2)
        repeated_bank = _manual_patch_bank(repeated, positions, object_points)
        ambiguous = audit.audit_patch_identity(repeated, mask, flow, confidence,
                                               repeated_bank, [repeated_bank], 1.)
        self.assertEqual(ambiguous['summary']['state'], 'unobservable_or_ambiguous')
        self.assertEqual(ambiguous['summary']['supported'], 0)

        blank = np.full_like(texture, .5)
        blank_bank = audit.build_template_patch_bank(blank, mask, np.arange(280 * 280),
                                                     np.zeros((280 * 280, 3)))
        no_bank = audit.audit_patch_identity(blank, mask, flow, confidence,
                                             blank_bank, [blank_bank], 1.)
        self.assertEqual(no_bank['summary']['state'], 'identity_bank_uninformative')
        self.assertEqual(no_bank['summary']['source_eligible'], 0)

    def test_patch_mosaic_writes_three_gray_panels_and_supported_label(self):
        entry = dict(
            source=np.full((audit.PATCH_SIZE, audit.PATCH_SIZE, 3), .2, dtype=np.float32),
            target=np.full((audit.PATCH_SIZE, audit.PATCH_SIZE, 3), .5, dtype=np.float32),
            competitor=np.full((audit.PATCH_SIZE, audit.PATCH_SIZE, 3), .8, dtype=np.float32),
            source_index=123, state='supported',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'mosaic.png'
            written = audit._write_mosaic(path, [entry])
            self.assertEqual(written, str(path))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape,
                         (audit.PATCH_SIZE * 2 + 18, audit.PATCH_SIZE * 6 * 16, 3))
        np.testing.assert_array_equal(image[3, 3], [51, 51, 51])
        np.testing.assert_array_equal(image[3, 26], [128, 128, 128])
        np.testing.assert_array_equal(image[3, 48], [204, 204, 204])
        label_region = image[audit.PATCH_SIZE * 2 + 1:, :audit.PATCH_SIZE * 2]
        self.assertTrue(np.any(np.all(label_region == [0, 255, 0], axis=2)))

    def test_empty_observed_mask_and_invalid_flow_keep_failure_counts(self):
        rng = np.random.default_rng(21)
        gray = rng.random((280, 280), dtype=np.float32)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        positions = [(40 + 60 * x, 40 + 60 * y) for y in range(4) for x in range(4)]
        object_points = [(float(x) * .25, float(y) * .25, 0.)
                         for y in range(4) for x in range(4)]
        bank = _manual_patch_bank(image, positions, object_points)
        flow = np.full((280, 280, 2), 1000., dtype=np.float32)
        confidence = np.ones((280, 280), dtype=np.float32)
        empty = audit.audit_patch_identity(image, np.zeros((280, 280), dtype=bool), flow,
                                           confidence, bank, [bank], 1.)
        self.assertEqual(empty['summary']['target_eligible'], 0)
        self.assertEqual(empty['summary']['state'], 'unobservable_or_ambiguous')
        self.assertEqual(empty['summary']['supported'], 0)
        self.assertEqual(empty['summary']['rejected'], bank['selected_anchor_count'])
        self.assertTrue(all(row['reason'] == 'target_patch_outside_eroded_observed_mask_or_image'
                            for row in empty['rows']))

    def test_prerun_patch_selection_is_bound_deterministic_ordered_and_tamper_evident(self):
        rng = np.random.default_rng(919)
        luminance = rng.random((280, 280), dtype=np.float32)
        image = np.repeat(luminance[:, :, None], 3, axis=2)
        mask = np.ones((280, 280), dtype=np.uint8)
        ids = np.arange(280 * 280, dtype=np.int64)
        yy, xx = np.indices((280, 280))
        points = np.column_stack((xx.reshape(-1) * .002, yy.reshape(-1) * .002,
                                  np.zeros(280 * 280)))
        arrays = dict(template_rgb=image, template_mask=mask, source_indices=ids,
                      source_pixels_xy=np.column_stack((xx.reshape(-1) + .5,
                                                        yy.reshape(-1) + .5)),
                      source_points_object_m=points)
        bank = audit.build_template_patch_bank(image, mask, ids, points)
        frozen = audit.fixed_patch_bank_manifest(bank, arrays)
        entry = dict(context_id='fixture-template', fixed_patch_bank=frozen)
        first = audit.verify_fixed_patch_bank(entry, arrays)
        second = audit.verify_fixed_patch_bank(entry, arrays)
        self.assertEqual(first['serialized_anchors'], second['serialized_anchors'])
        self.assertEqual([row['source_index'] for row in first['serialized_anchors']],
                         [row['source_index'] for row in bank['serialized_anchors']])
        self.assertEqual(first['selected_anchor_count'], 64)
        self.assertEqual(set(frozen['template_input_sha256']), set(audit._PATCH_BANK_INPUTS))
        tampered = dict(entry, fixed_patch_bank=json.loads(json.dumps(frozen)))
        tampered['fixed_patch_bank']['anchors'][0]['source_xy_crop'][0] += .5
        with self.assertRaisesRegex(ValueError, 'patch bank or input binding changed'):
            audit.verify_fixed_patch_bank(tampered, arrays)

        preparation_source = inspect.getsource(probe._prepare_frame)
        capture_source = inspect.getsource(probe.run_capture)
        self.assertIn('fixed_patch_bank_manifest', preparation_source)
        self.assertLess(capture_source.index('frame_context, seed_pose, reason = _prepare_frame'),
                        capture_source.index('entry = _forward_one'))

    def test_synthetic_query_preparation_uses_object_id_from_input_manifest(self):
        class CameraModel:
            def __init__(self, width, height, f, c, T_world_from_eye):
                self.width, self.height = width, height

        class Structs:
            PinholePlaneCameraModel = CameraModel

        class RenderType:
            COLOR = 'color'
            DEPTH = 'depth'
            MASK = 'mask'

        renderer_base = type('RendererBase', (), {'RenderType': RenderType})

        class Renderer:
            def __init__(self):
                self.object_ids = []

            def render_object_model(self, object_id, camera):
                self.object_ids.append(int(object_id))
                shape = (camera.height, camera.width)
                return {
                    RenderType.COLOR: np.full((*shape, 3), .35, dtype=np.float32),
                    RenderType.DEPTH: np.full(shape, 900., dtype=np.float32),
                    RenderType.MASK: np.ones(shape, dtype=np.uint8),
                }

        manifest = dict(
            object_id=731, intrinsics=[[30., 0., 2.], [0., 31., 1.5], [0., 0., 1.]],
            native_resolution=[4, 3],
        )
        capture = dict(packet_bytes=0, packet_failures=[], contexts=[],
                       **probe._capture_manifest_fields(manifest))
        seed_pose = _pose([0., 0., 0.], [0., 0., .9])
        frame_context = dict(arrays=dict(
            native_k=np.asarray(manifest['intrinsics'], dtype=np.float64),
            crop_k=np.asarray(manifest['intrinsics'], dtype=np.float64),
            crop_from_native=np.eye(4, dtype=np.float64),
            observed_crop_mask=np.ones((280, 280), dtype=np.uint8),
            seed_pose_m=seed_pose,
        ))
        renderer = Renderer()
        with tempfile.TemporaryDirectory() as temporary:
            result = probe._prepare_synthetic_query(
                10, 8, frame_context, seed_pose, np.array([0., 0., 1.]),
                np.zeros(3), renderer, renderer_base, Structs, Path(temporary), capture,
                np.ones((3, 4), dtype=bool))
            self.assertEqual(renderer.object_ids, [manifest['object_id'], manifest['object_id']])
            self.assertEqual(result['entry']['role'], 'synthetic_query')
            self.assertEqual(result['arrays']['known_query_pose_m'].shape, (4, 4))
            self.assertTrue(np.all(result['arrays']['synthetic_native_mask']))
            self.assertEqual(capture['object_id'], manifest['object_id'])
            persisted = json.loads((Path(temporary) / 'capture.json').read_text(encoding='utf-8'))
            self.assertEqual(persisted['object_id'], manifest['object_id'])

    def test_decision_uses_real_patch_state_for_texture_support_conclusion(self):
        report, capture, banks = _decision_fixture()
        decision = audit._decision_summary(report, capture, banks)
        self.assertEqual(
            decision['permitted_conclusion'],
            'real_current_image_texture_support_exists_for_tested_identity_only')

    def test_color_domain_gate_counts_both_view_hypotheses_and_all_negative_states(self):
        report, capture, banks = _decision_fixture(
            gray_template_improvement=True,
            false_acceptance=[
                (10, 8, 'rgb', 0, 'rgb'),
                (10, 8, 'gray', 180, 'gray'),
            ])
        decision = audit._decision_summary(report, capture, banks)
        comparison = decision['gray_query_native_vs_gray_template']['8']
        negatives = comparison['negative_false_acceptance_by_query_appearance']
        self.assertEqual(negatives['rgb']['validated_false_acceptances'], 1)
        self.assertEqual(negatives['gray']['validated_false_acceptances'], 1)
        self.assertEqual(negatives['rgb']['expected_conditions'], 16)
        self.assertEqual(negatives['gray']['state_counts']['captured'], 16)
        self.assertTrue(comparison['all_32_negative_conditions_accounted'])
        by_template = comparison['negative_false_acceptance_by_template_appearance_on_gray_query']
        self.assertEqual(by_template['rgb']['validated_false_acceptances'], 0)
        self.assertEqual(by_template['gray']['validated_false_acceptances'], 1)
        self.assertEqual(by_template['rgb']['expected_conditions'], 8)
        self.assertEqual(by_template['gray']['state_counts']['captured'], 8)
        opposite = [row for row in comparison['negative_false_acceptance_by_template_hypothesis']
                    if row['template_offset_deg'] == 180 and row['template_appearance'] == 'gray']
        self.assertEqual(len(opposite), 2)
        self.assertEqual(sum(row['validated_false_acceptances'] for row in opposite), 1)
        self.assertFalse(comparison['no_increase_in_negative_false_acceptance'])
        self.assertFalse(comparison['controlled_color_domain_gap'])

        decreased_report, decreased_capture, decreased_banks = _decision_fixture(
            gray_template_improvement=True,
            false_acceptance=[
                (10, 8, 'gray', 0, 'rgb'),
                (50, 8, 'gray', 180, 'rgb'),
                (10, 8, 'gray', 180, 'gray'),
            ])
        decreased = audit._decision_summary(
            decreased_report, decreased_capture, decreased_banks)['gray_query_native_vs_gray_template']['8']
        decreased_by_template = decreased['negative_false_acceptance_by_template_appearance_on_gray_query']
        self.assertEqual(decreased_by_template['rgb']['validated_false_acceptances'], 2)
        self.assertEqual(decreased_by_template['gray']['validated_false_acceptances'], 1)
        self.assertTrue(decreased['no_increase_in_negative_false_acceptance'])
        self.assertTrue(decreased['controlled_color_domain_gap'])

        for missing_state in ('unavailable', 'failed'):
            with self.subTest(missing_state=missing_state):
                missing_report, missing_capture, missing_banks = _decision_fixture(
                    gray_template_improvement=True)
                omitted = next(row for row in missing_report['synthetic']
                               if row['query_offset_deg'] == 8 and row['query_appearance'] == 'rgb' and
                               row['template_offset_deg'] == 180 and row['template_appearance'] == 'gray')
                omitted.update(state=missing_state, validated_wrong_surface_false_acceptance=None)
                missing_decision = audit._decision_summary(missing_report, missing_capture, missing_banks)
                missing = missing_decision['gray_query_native_vs_gray_template']['8']
                rgb_denominator = missing['negative_false_acceptance_by_query_appearance']['rgb']
                self.assertEqual(rgb_denominator['state_counts'][missing_state], 1)
                self.assertEqual(rgb_denominator['state_counts']['captured'], 15)
                self.assertFalse(rgb_denominator['complete'])
                self.assertFalse(missing['no_increase_in_negative_false_acceptance'])
                self.assertFalse(missing['controlled_color_domain_gap'])

    def test_uninformative_query_or_template_ablation_blocks_color_conclusions(self):
        for field, uninformative in (
            ('query', (10, 8)), ('template', (10, 0)),
        ):
            with self.subTest(field=field):
                report, capture, banks = _decision_fixture(
                    gray_template_improvement=True,
                    uninformative_query=uninformative if field == 'query' else None,
                    uninformative_template=uninformative if field == 'template' else None)
                decision = audit._decision_summary(report, capture, banks)
                comparison = decision['gray_query_native_vs_gray_template']['8']
                self.assertAlmostEqual(comparison['correct_identity_availability_improvement'], .3)
                self.assertFalse(comparison['appearance_informative'])
                self.assertTrue(comparison['appearance_ablation_uninformative'])
                self.assertFalse(comparison['controlled_color_domain_gap'])
                real_gate = decision['gray_template_real_support_gain_gate']
                self.assertTrue(any(row['supported_anchor_gain'] == 4
                                    for row in decision['real_template_appearance_comparison']))
                if field == 'template':
                    self.assertFalse(real_gate['passed'])
                    self.assertFalse(real_gate['appearance_informative'])
                else:
                    self.assertTrue(real_gate['passed'])
                    self.assertTrue(real_gate['appearance_informative'])

    def test_real_leak_guard_rejects_hashed_synthetic_pose_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'capture'
            root.mkdir()
            record = dict(packet_bytes=0, packet_failures=[], contexts=[])
            frame_id = audit.FRAME_IDS[0]
            real_arrays = dict(
                native_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                native_mask=np.ones((2, 2), dtype=np.uint8),
                seed_pose_m=np.eye(4, dtype=np.float64),
                known_query_pose_m=np.eye(4, dtype=np.float64),
            )
            real_entry = probe._write_context(
                root, record, 'real-frame', 'real_frame', frame_id, 'frame', real_arrays)
            template_entry = probe._write_context(
                root, record, 'template', 'template', frame_id, 'frame',
                dict(template_pose_m=np.eye(4, dtype=np.float64)))
            conditions = _planned_conditions()
            target = next(row for row in conditions if row['kind'] == 'real' and
                          row['frame_id'] == frame_id and row['template_offset_deg'] == 0 and
                          row['template_appearance'] == 'rgb')
            refs = dict(query='real-frame', template='template', observed_frame='real-frame')
            target.update(state='captured', context_refs=refs)
            for row in conditions:
                if row is not target:
                    row.update(state='unavailable', reason='fixture_unavailable')
            forward = probe._write_packet(
                root, record, Path('packets/forwards/fixture.npz'),
                dict(flow=np.zeros((280, 280, 2), dtype=np.float32),
                     confidence=np.ones((280, 280), dtype=np.float32)))
            forward.update(condition_id=target['condition_id'], context_refs=refs)
            capture = dict(
                schema_version=1, object='ranch', complete=False, status='incomplete',
                requested_real_frame_ids=list(audit.FRAME_IDS),
                requested_synthetic_carriers=list(audit.SYNTHETIC_CARRIERS),
                planned_forward_cap=audit.FORWARD_CAP,
                packet_budget_bytes=audit.PACKET_BUDGET_BYTES,
                packet_bytes=record['packet_bytes'], reference_or_annotations_loaded=False,
                conditions=conditions, contexts=[real_entry, template_entry], forwards=[forward],
            )
            (root / 'capture.json').write_text(json.dumps(capture), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'synthetic oracle/annotation fields'):
                audit.audit_capture(root, Path(temporary) / 'audit')
            clean_real_entry = dict(
                real_entry,
                arrays={name: info for name, info in real_entry['arrays'].items()
                        if name != 'known_query_pose_m'},
            )
            self.assertTrue(audit._real_branch_is_leak_free(
                {target['condition_id']: target},
                {'real-frame': clean_real_entry, 'template': template_entry}))

    def test_aggregate_packet_budget_failure_preserves_record_without_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = dict(packet_bytes=audit.PACKET_BUDGET_BYTES, packet_failures=[])
            relative = Path('packets/contexts/over-budget.npz')
            with self.assertRaisesRegex(ValueError, 'aggregate exceeds'):
                probe._write_packet(root, capture, relative, dict(value=np.ones(1, dtype=np.uint8)))
            self.assertEqual(capture['packet_bytes'], audit.PACKET_BUDGET_BYTES)
            self.assertEqual(len(capture['packet_failures']), 1)
            self.assertGreater(capture['packet_failures'][0]['aggregate_bytes'],
                               audit.PACKET_BUDGET_BYTES)
            self.assertFalse((root / relative).exists())
            self.assertFalse((root / relative).with_suffix('.npz.tmp').exists())
            preserved = json.loads((root / 'capture.json').read_text(encoding='utf-8'))
            self.assertEqual(preserved['packet_bytes'], audit.PACKET_BUDGET_BYTES)
            self.assertEqual(preserved['packet_failures'], capture['packet_failures'])

    def test_mocked_forward_boundary_preserves_direction_mask_and_cap(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def to(self, _device):
                return self

            def __getitem__(self, index):
                return Tensor(self.value[index])

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Torch:
            class cuda:
                @staticmethod
                def synchronize():
                    pass

            @staticmethod
            def from_numpy(value):
                return Tensor(value)

            @staticmethod
            def inference_mode():
                class Inference:
                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        return False
                return Inference()

        calls = []

        def network(query, template, mask):
            calls.append((query.value.copy(), template.value.copy(), mask.value.copy()))
            return Tensor(np.zeros((1, 2, 280, 280), dtype=np.float32)), Tensor(
                np.full((1, 280, 280), .8, dtype=np.float32))

        query_rgb = np.full((280, 280, 3), .25, dtype=np.float32)
        template_rgb = np.full((280, 280, 3), .75, dtype=np.float32)
        template_mask = np.zeros((280, 280), dtype=np.uint8)
        template_mask[10:20, 10:20] = 1
        condition = dict(
            kind='real', condition_id='fixture-forward', rng_seed=17,
            query_appearance='rgb', template_appearance='rgb', forward_index=0,
            context_refs=dict(query='query', template='template', observed_frame='query'))
        query_context = dict(arrays=dict(crop_rgb=query_rgb))
        template_context = dict(arrays=dict(
            template_rgb=template_rgb, template_mask=template_mask,
            source_indices=np.array([10 * 280 + 10], dtype=np.int64),
            source_pixels_xy=np.array([[10.5, 10.5]], dtype=np.float64),
            source_points_object_m=np.array([[0., 0., .5]], dtype=np.float64)))
        with tempfile.TemporaryDirectory() as temporary:
            capture = dict(forward_calls=0, packet_bytes=0, packet_failures=[], forwards=[])
            entry = probe._forward_one(condition, query_context, template_context,
                                       network, Torch, Path(temporary), capture, 'cpu')
            self.assertEqual(capture['forward_calls'], 1)
            self.assertEqual(entry['condition_id'], condition['condition_id'])
            self.assertEqual(len(calls), 1)
            query, template, mask = calls[0]
            np.testing.assert_allclose(query, .25)
            np.testing.assert_allclose(template, .75)
            self.assertEqual(query.shape, (1, 3, 280, 280))
            self.assertEqual(template.shape, (1, 3, 280, 280))
            self.assertEqual(mask.shape, (1, 280, 280))
            self.assertEqual(float(mask.sum()), 100.)
            at_cap = dict(forward_calls=audit.FORWARD_CAP, packet_bytes=0,
                          packet_failures=[], forwards=[])
            with self.assertRaisesRegex(ValueError, 'budget reached'):
                probe._forward_one(condition, query_context, template_context,
                                   network, Torch, Path(temporary), at_cap, 'cpu')
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
