import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_pose_ablation as ablation
from .quality_contract import Frame
from .vision import cv2


def _pose(rotation_vector=(0., 0., 0.), translation=(0., 0., .9)):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))[0]
    value[:3, 3] = np.asarray(translation, dtype=np.float64)
    return value


def _scene(grid=30, seed=8675309):
    xy = np.array([(x, y) for x in np.linspace(-.2, .2, grid)
                   for y in np.linspace(-.2, .2, grid)], dtype=np.float64)
    points = np.column_stack((xy, .015 * np.sin(xy[:, 0] * 9.) + .01 * np.cos(xy[:, 1] * 7.)))
    k = np.array([[400., 0., 140.], [0., 405., 140.], [0., 0., 1.]], dtype=np.float64)
    template_pose = _pose()
    truth_pose = _pose((.018, -.012, .006), (.004, -.003, .91))
    source_pixels = audit.project_object_points(points, template_pose, k)[0]
    endpoints = audit.project_object_points(points, truth_pose, k)[0]
    rng = np.random.default_rng(seed)
    gray = rng.random((280, 280), dtype=np.float32)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.ones((280, 280), dtype=bool)
    ids = np.arange(len(points), dtype=np.int64)
    source = dict(
        template_rgb=rgb, template_depth_mm=np.full((280, 280), 900., dtype=np.float32),
        template_mask=mask.copy(), observed_crop_mask=mask.astype(np.uint8),
        source_indices=ids, source_pixels_xy=source_pixels.astype(np.float64),
        source_points_object_m=points.astype(np.float64), template_pose_m=template_pose,
        crop_k=k, crop_from_native=np.eye(4, dtype=np.float64), native_k=k.copy())
    frame = dict(native_rgb=np.zeros((280, 280, 3), dtype=np.uint8),
                 native_mask=np.ones((280, 280), dtype=np.uint8), native_k=k.copy())
    forward = dict(flow=np.zeros((280, 280, 2), dtype=np.float32),
                   confidence=np.ones((280, 280), dtype=np.float32))
    payload = dict(plan=dict(condition_id='fixture', frame_id=10), source=source,
                   frame=frame, query=None, forward=forward,
                   endpoints=endpoints.astype(np.float64),
                   confidence=np.full(len(points), .95, dtype=np.float32), seed=170)
    visible_rows = np.arange(len(points), dtype=np.int64)
    visible_ids = ids.copy()
    visible = dict(visible_rows=visible_rows, visible_source_ids=visible_ids,
                   source_id_sha256=ablation._array_digest(visible_ids),
                   count=len(visible_rows), truth_pose_crop_m=truth_pose, truth_valid=True)
    selection = dict(eligible=np.ones(len(points), dtype=bool), total_sources=len(points),
                     eligible_count=len(points))
    return payload, selection, visible, truth_pose


class BottlePoseAblationTests(unittest.TestCase):
    def test_exact_frozen_plan_contains_only_the_twelve_required_rows(self):
        plans = ablation._expected_condition_plan()
        self.assertEqual(len(plans), 12)
        self.assertEqual(len({row['condition_id'] for row in plans}), 12)
        self.assertEqual(sum(row['kind'] == 'synthetic_positive' for row in plans), 3)
        self.assertEqual(sum(row['kind'] == 'self_control' for row in plans), 6)
        self.assertEqual(sum(row['kind'] == 'synthetic_negative' for row in plans), 3)
        self.assertEqual({row['frame_id'] for row in plans}, {10, 50, 100})
        self.assertEqual({row['condition_id'] for row in plans if row['kind'] == 'self_control'},
                         {f'zero-{frame}-{state}' for frame in (10, 50, 100)
                          for state in ('full', 'clipped')})

    def test_source_selection_counts_boundary_and_low_texture_without_flow_inputs(self):
        rng = np.random.default_rng(12)
        base = rng.random((64, 64), dtype=np.float32)
        image = np.repeat(base[:, :, None], 3, axis=2)
        image[10:54, 15:55] = .5
        template_mask = np.ones((64, 64), dtype=bool)
        observed = template_mask.copy()
        ids = np.array([15 * 64 + 15, 25 * 64 + 35, 3 * 64 + 3], dtype=np.int64)
        pixels = np.array([[15.5, 15.5], [35.5, 25.5], [3.5, 3.5]], dtype=np.float64)
        source = dict(template_rgb=image, template_mask=template_mask,
                      observed_crop_mask=observed, source_indices=ids,
                      source_pixels_xy=pixels, source_points_object_m=np.zeros((3, 3)))
        flow_and_truth = dict(flow=np.full((64, 64, 2), np.nan),
                              known_query_pose_m=np.full((4, 4), np.nan),
                              query_depth_mm=np.full((64, 64), np.nan))
        first = ablation.freeze_source_selection(source)
        flow_and_truth['flow'][:] = 0.
        flow_and_truth['known_query_pose_m'][:] = np.eye(4)
        second = ablation.freeze_source_selection(source)
        self.assertEqual(first['eligible_array']['sha256'], second['eligible_array']['sha256'])
        self.assertEqual(first['total_sources'], 3)
        self.assertEqual(first['boundary_excluded'], 1)
        self.assertGreaterEqual(first['low_std_excluded'], 1)
        self.assertEqual(first['eligible_count'] + first['boundary_excluded'] +
                         first['low_std_excluded'], 3)
        self.assertFalse(first['eligible'].flags.writeable)

    def test_each_template_offset_uses_its_own_source_texture_selection(self):
        rng = np.random.default_rng(99)
        textured = np.repeat(rng.random((96, 96), dtype=np.float32)[:, :, None], 3, axis=2)
        blank = np.full((96, 96, 3), .5, dtype=np.float32)
        mask = np.zeros((96, 96), dtype=bool)
        mask[8:88, 8:88] = True
        yy, xx = np.indices(mask.shape)
        ids = (yy[mask] * 96 + xx[mask]).astype(np.int64)
        pixels = np.column_stack((xx[mask] + .5, yy[mask] + .5)).astype(np.float64)
        common = dict(template_mask=mask, observed_crop_mask=mask,
                      source_indices=ids, source_pixels_xy=pixels,
                      source_points_object_m=np.column_stack((pixels, np.zeros(len(ids)))))
        t0 = ablation.freeze_source_selection(dict(common, template_rgb=textured))
        t180 = ablation.freeze_source_selection(dict(common, template_rgb=blank))
        self.assertGreater(t0['eligible_count'], 0)
        self.assertEqual(t180['eligible_count'], 0)
        self.assertNotEqual(t0['eligible_array']['sha256'], t180['eligible_array']['sha256'])

    def test_arm_subsetting_preserves_measured_endpoint_and_confidence_bytes(self):
        payload, selection, _visible, _truth = _scene()
        selection['eligible'][::2] = False
        endpoint_before = payload['endpoints'].tobytes()
        confidence_before = payload['confidence'].tobytes()
        endpoint_hash = ablation._array_digest(payload['endpoints'])
        confidence_hash = ablation._array_digest(payload['confidence'])
        all_inputs = ablation._arm_inputs(payload, selection, 'all')
        observed_inputs = ablation._arm_inputs(payload, selection, 'source_observable')
        self.assertEqual(all_inputs['endpoint_confidence_hashes_before']['confidence_dtype'], 'float32')
        self.assertEqual(observed_inputs['confidence'].dtype, payload['confidence'].dtype)
        self.assertEqual(ablation._array_digest(all_inputs['endpoints_crop']), endpoint_hash)
        self.assertEqual(ablation._array_digest(all_inputs['confidence']), confidence_hash)
        np.testing.assert_array_equal(observed_inputs['endpoints_crop'], payload['endpoints'][selection['eligible']])
        np.testing.assert_array_equal(observed_inputs['confidence'], payload['confidence'][selection['eligible']])
        self.assertEqual(payload['endpoints'].tobytes(), endpoint_before)
        self.assertEqual(payload['confidence'].tobytes(), confidence_before)

    def test_shared_solver_receives_same_camera_masks_seed_and_unchanged_inputs(self):
        payload, selection, _visible, _truth = _scene()
        selection['eligible'][::3] = False
        captured = []

        def fake_fit(*args):
            captured.append(args)
            return dict(state='unavailable', reason='fixture', retained_correspondences=0,
                        sampled_correspondences=0, rng_seed=args[-1], ransac_inlier_ids=[],
                        retained_inlier_ids=[], pose_crop_m=None, pose_native_m=None,
                        validation_state='not_run', validation_reason='fixture', validation_stats=None)

        with mock.patch.object(audit, 'fit_learned_packet', side_effect=fake_fit):
            for arm_name in ablation.ARMS:
                inputs = ablation._arm_inputs(payload, selection, arm_name)
                ablation._shared_fit_call(inputs, payload)
        self.assertEqual(len(captured), 2)
        expected_near_pose = audit.crop_pose_from_context(payload['source'])
        for call, arm_name in zip(captured, ablation.ARMS):
            inputs = ablation._arm_inputs(payload, selection, arm_name)
            np.testing.assert_array_equal(call[0], inputs['source_points_m'])
            np.testing.assert_array_equal(call[1], inputs['endpoints_crop'])
            np.testing.assert_array_equal(call[2], inputs['confidence'])
            np.testing.assert_array_equal(call[3], inputs['source_indices'])
            np.testing.assert_array_equal(call[4], payload['source']['observed_crop_mask'].astype(bool))
            np.testing.assert_array_equal(call[5], payload['source']['observed_crop_mask'].astype(bool))
            np.testing.assert_array_equal(call[6], expected_near_pose)
            np.testing.assert_array_equal(call[7], payload['source']['crop_k'])
            np.testing.assert_array_equal(call[8], payload['source']['crop_from_native'])
            np.testing.assert_array_equal(call[9], payload['frame']['native_k'])
            self.assertIsInstance(call[10], Frame)
            np.testing.assert_array_equal(call[11], payload['frame']['native_mask'])
            self.assertEqual(call[12], payload['seed'])
        self.assertEqual(captured[0][-1], captured[1][-1])

    def test_fit_calls_reset_the_same_rng_seed_for_deterministic_repeat(self):
        payload, selection, visible, _truth = _scene(grid=22)
        original_set_seed = audit.cv2.setRNGSeed
        seeds = []

        def record_seed(seed):
            seeds.append(int(seed))
            return original_set_seed(seed)

        with mock.patch.object(audit.cv2, 'setRNGSeed', side_effect=record_seed):
            result = ablation._arm_result(payload, selection, visible, 'all', .7)
        self.assertEqual(seeds, [payload['seed'], payload['seed']])
        self.assertTrue(result['deterministic_repeat']['matched'])
        self.assertEqual(result['shared_solver']['function'],
                         'quality_bottle_identity_audit.fit_learned_packet')

    def test_metres_crop_native_transform_parity_with_rotated_crop(self):
        native = _pose((.05, -.08, .02), (.1, -.03, .7))
        crop_from_native = _pose((-.02, .04, .12), (.015, -.008, .025))
        crop_mm = audit.native_to_crop_pose_mm(native, crop_from_native)
        reconstructed = audit.crop_to_native_pose_m(crop_mm, crop_from_native)
        np.testing.assert_allclose(reconstructed, native, atol=1e-11, rtol=0)
        crop_m = crop_mm.copy()
        crop_m[:3, 3] *= .001
        parity = ablation._unit_pose_parity(
            dict(pose_crop_m=crop_m.tolist(), pose_native_m=native.tolist()), crop_from_native)
        self.assertEqual(parity['state'], 'checked')
        self.assertLess(parity['native_translation_roundtrip_error_m'], 1e-10)
        self.assertLess(parity['crop_translation_roundtrip_error_m'], 1e-10)

    def test_textured_rigid_matches_recover_pose_and_score_the_frozen_full_surface(self):
        payload, selection, visible, truth = _scene(grid=22)
        result = ablation._arm_result(payload, selection, visible, 'all', .7)
        self.assertEqual(result['fit_state'], 'refined')
        self.assertEqual(result['accepted_pose_state'], 'accepted')
        self.assertLess(result['proposal_rotation_error']['rotation_error_degrees'], .01)
        self.assertLess(result['proposal_rotation_error']['translation_error_m'], .0001)
        surface = result['proposal_surface_projection_720']
        self.assertEqual(surface['state'], 'scored')
        self.assertEqual(surface['denominator']['count'], len(payload['source']['source_indices']))
        self.assertEqual(surface['denominator']['ordered_source_ids_sha256'], visible['source_id_sha256'])
        self.assertEqual(result['current_image_residuals_all_observed_sources']['denominator'],
                         len(payload['source']['source_indices']))
        self.assertTrue(result['deterministic_repeat']['matched'])

    def test_misleading_textured_rigid_field_can_fit_well_but_miss_all_surface(self):
        payload, selection, visible, truth = _scene(grid=22)
        misleading = _pose(translation=(.055, -.003, .91))
        payload['endpoints'] = audit.project_object_points(
            payload['source']['source_points_object_m'], misleading,
            payload['source']['crop_k'])[0]
        result = ablation._arm_result(payload, selection, visible, 'all', .7)
        self.assertEqual(result['fit_state'], 'refined')
        self.assertEqual(result['accepted_pose_state'], 'accepted')
        residual = result['current_image_residuals_all_observed_sources']
        surface = result['proposal_surface_projection_720']
        self.assertLess(residual['median_720'], .001)
        self.assertGreater(result['proposal_rotation_error']['translation_error_m'], .04)
        self.assertGreater(surface['p95_720'], 10.)
        self.assertEqual(surface['denominator']['count'], len(visible['visible_rows']))

    def test_report_only_current_image_contract_validates_all_original_observed_rows(self):
        payload, selection, visible, _truth = _scene(grid=22)
        selection['eligible'][::2] = False
        original_rows = ablation._filtered_observed_rows(
            payload['source'], payload['endpoints'], payload['confidence'],
            payload['source']['observed_crop_mask'])
        fit_rows = ablation._observed_filtered_rows_for_arm(
            ablation._arm_inputs(payload, selection, 'source_observable'))
        original_validate = audit.validate
        calls = []

        def capture_candidate(*args, **kwargs):
            calls.append(args)
            return original_validate(*args, **kwargs)

        with mock.patch.object(audit, 'validate', side_effect=capture_candidate):
            result = ablation._arm_result(payload, selection, visible, 'source_observable', .7)
        validation = result['current_image_validation_all_observed_sources']
        self.assertEqual(len(calls), 3)  # primary fit, deterministic repeat, full-original report metric
        candidate, frame, native_mask = calls[-1][:3]
        self.assertEqual(len(candidate.points_object_m), len(original_rows))
        self.assertGreater(len(candidate.points_object_m), len(fit_rows))
        np.testing.assert_array_equal(
            candidate.points_object_m,
            payload['source']['source_points_object_m'][original_rows])
        np.testing.assert_array_equal(candidate.weights, payload['confidence'][original_rows])
        np.testing.assert_allclose(
            candidate.pixels_image,
            ablation._native_xy_from_crop_endpoints(
                payload['endpoints'][original_rows], payload['source']['crop_k'],
                payload['source']['crop_from_native'], payload['frame']['native_k']))
        np.testing.assert_array_equal(frame.rgb, payload['frame']['native_rgb'])
        np.testing.assert_array_equal(native_mask, payload['frame']['native_mask'])
        self.assertEqual(validation['state'], 'accepted')
        self.assertEqual(validation['validation_stats']['correspondences'], len(original_rows))
        self.assertEqual(validation['source_ids']['count'], len(original_rows))
        self.assertTrue(validation['report_only'])
        self.assertFalse(validation['changes_fit_or_acceptance'])

    def test_empty_or_nonspatial_selection_cannot_fall_back_to_seed_pose(self):
        payload, selection, visible, _truth = _scene(grid=22)
        selection['eligible'][:] = False
        empty = ablation._arm_result(payload, selection, visible, 'source_observable', .7)
        self.assertEqual(empty['fit_state'], 'unavailable')
        self.assertEqual(empty['accepted_pose_state'], 'unavailable')
        self.assertIsNone(empty['proposal_pose'])
        self.assertIsNone(empty['accepted_pose'])
        self.assertEqual(empty['proposal_surface_projection_720']['state'], 'unavailable_candidate_pose')
        self.assertGreater(empty['proposal_surface_projection_720']['denominator']['count'], 0)

        payload, selection, visible, _truth = _scene(grid=22)
        selection['eligible'][:] = False
        selection['eligible'][230:260] = True
        local = ablation._arm_result(payload, selection, visible, 'source_observable', .7)
        self.assertEqual(local['fit_state'], 'refined')
        self.assertEqual(local['accepted_pose_state'], 'rejected')
        self.assertEqual(local['validation_reason'], 'localized_or_ambiguous_support')
        self.assertIsNone(local['accepted_pose'])

    def test_all_source_residual_diagnostic_keeps_full_observed_denominator(self):
        payload, selection, visible, _truth = _scene(grid=22)
        selection['eligible'][:] = False
        selection['eligible'][::2] = True
        full_count = len(ablation._observed_filtered_rows_for_arm(
            ablation._arm_inputs(payload, selection, 'all')))
        result = ablation._arm_result(payload, selection, visible, 'source_observable', .7)
        self.assertLess(result['arm_input']['source_count'], full_count)
        self.assertEqual(result['current_image_residuals_all_observed_sources']['denominator'], full_count)

    def test_truth_visible_denominator_is_independent_of_candidate_and_selection(self):
        points = np.array([[-.1, -.1, 0.], [.1, -.1, 0.], [.0, .1, 0.]])
        truth_crop = _pose()
        crop_from_native = _pose((.01, .02, -.03), (.02, -.01, .04))
        truth_native = ablation._truth_native_pose(truth_crop, crop_from_native)
        native_k = np.array([[300., 0., 100.], [0., 300., 100.], [0., 0., 1.]])
        frame = Frame(1, np.zeros((720, 200, 3), dtype=np.uint8), native_k)
        visible_rows = np.arange(3, dtype=np.int64)
        visible_ids = np.array([41, 75, 99], dtype=np.int64)
        a = ablation.surface_projection_score(points, visible_rows, truth_crop, truth_native,
                                              crop_from_native, native_k, frame,
                                              visible_source_ids=visible_ids)
        wrong = truth_native.copy()
        wrong[0, 3] += .02
        b = ablation.surface_projection_score(points, visible_rows, truth_crop, wrong,
                                              crop_from_native, native_k, frame,
                                              visible_source_ids=visible_ids)
        self.assertEqual(a['state'], 'scored')
        self.assertAlmostEqual(a['median_720'], 0., places=10)
        self.assertGreater(b['median_720'], 1.)
        self.assertEqual(a['denominator'], b['denominator'])
        self.assertEqual(a['denominator']['ordered_source_ids_sha256'],
                         ablation._array_digest(visible_ids))

    def test_surface_metric_rejects_truth_behind_camera_and_null_candidate_without_shrinking_V(self):
        points = np.array([[-.02, -.02, 0.], [.02, -.02, 0.], [.0, .02, 0.]])
        truth = _pose()
        k = np.array([[250., 0., 140.], [0., 250., 140.], [0., 0., 1.]])
        frame = Frame(1, np.zeros((280, 280, 3), dtype=np.uint8), k)
        rows = np.arange(3, dtype=np.int64)
        ids = np.array([10, 20, 30], dtype=np.int64)
        null_result = ablation.surface_projection_score(
            points, rows, truth, None, np.eye(4), k, frame, visible_source_ids=ids)
        self.assertEqual(null_result['state'], 'unavailable_candidate_pose')
        self.assertEqual(null_result['denominator']['count'], 3)
        behind = _pose(translation=(0., 0., -.5))
        invalid = ablation.surface_projection_score(
            points, rows, truth, behind, np.eye(4), k, frame, visible_source_ids=ids)
        self.assertEqual(invalid['state'], 'invalid_projection')
        self.assertEqual(invalid['denominator']['count'], 3)
        self.assertIsNone(invalid['median_720'])
        invalid_truth = ablation.surface_projection_score(
            points, rows, _pose(translation=(0., 0., -.5)), truth,
            np.eye(4), k, frame, visible_source_ids=ids)
        self.assertEqual(invalid_truth['state'], 'invalid_truth_projection')

    def test_negative_false_acceptance_keeps_full_field_wrong_surface_evidence(self):
        baseline = dict(original_wrong_surface_evidence=dict(
            thresholds_passed=True, confident_invisible_endpoints=100,
            confident_invisible_fraction=.2, median_invisible_identity_distance_fraction=.3))
        record = dict(conditions=[dict(
            condition_id='syn-10-q8-rgb-t180-rgb', arms=[
                dict(arm='all', accepted_pose_state='rejected'),
                dict(arm='source_observable', accepted_pose_state='accepted',
                     arm_input=dict(selected_source_count=0))])])
        outcome = ablation._exploratory_pose_prerequisite(
            record, {}, {'syn-10-q8-rgb-t180-rgb': baseline}, .7)
        row = next(row for row in outcome['negative_rows']
                   if row['condition_id'] == 'syn-10-q8-rgb-t180-rgb')
        self.assertTrue(row['full_original_wrong_surface_evidence'])
        self.assertTrue(row['source_observable_false_acceptance'])
        self.assertTrue(row['new_negative_false_acceptance'])
        self.assertFalse(outcome['no_new_negative_false_acceptance'])

    def test_negative_empty_V_is_report_only_but_does_not_disable_original_false_acceptance_gate(self):
        plans = ablation._expected_condition_plan()
        visible_sets = {}
        baselines = {}
        conditions = []
        for plan in plans:
            condition_id = plan['condition_id']
            if plan['kind'] == 'synthetic_negative':
                count = {10: 8, 50: 0, 100: 0}[plan['frame_id']]
                visible_sets[condition_id] = dict(count=count, truth_valid=count > 0)
                baselines[condition_id] = dict(original_wrong_surface_evidence={
                    'thresholds_passed': True, 'confident_invisible_endpoints': 40,
                    'confident_invisible_fraction': .1,
                    'median_invisible_identity_distance_fraction': .2})
                all_state = observable_state = 'rejected'
            else:
                visible_sets[condition_id] = dict(count=1, truth_valid=True)
                all_state = observable_state = 'accepted'
            if plan['kind'] == 'synthetic_positive':
                all_error = dict(rotation_error_degrees=2., translation_error_m=.01,
                                 translation_error_fraction_D=.014)
                observable_error = dict(rotation_error_degrees=1., translation_error_m=.005,
                                        translation_error_fraction_D=.007)
            else:
                all_error = dict(rotation_error_degrees=.2, translation_error_m=.001,
                                 translation_error_fraction_D=.0014)
                observable_error = dict(rotation_error_degrees=.1, translation_error_m=.0005,
                                        translation_error_fraction_D=.0007)
            conditions.append(dict(condition_id=condition_id, arms=[
                dict(arm='all', accepted_pose_state=all_state,
                     proposal_rotation_error=all_error,
                     deterministic_repeat=dict(matched=True)),
                dict(arm='source_observable', accepted_pose_state=observable_state,
                     proposal_rotation_error=observable_error,
                     deterministic_repeat=dict(matched=True))]))

        result = ablation._exploratory_pose_prerequisite(
            dict(conditions=conditions), visible_sets, baselines, .7)
        self.assertTrue(result['positive_and_self_truth_sets_valid'])
        self.assertTrue(result['all_three_positive_source_observable_passed'], result['positive_rows'])
        self.assertTrue(result['at_least_one_positive_improves'], result['positive_improvements'])
        self.assertTrue(result['all_six_self_controls_passed'], result['self_rows'])
        self.assertTrue(result['no_new_negative_false_acceptance'], result['negative_rows'])
        self.assertTrue(result['deterministic_repeats_match'])
        self.assertTrue(result['passed'])
        self.assertTrue(result['negative_truth_visible_sets_report_only'])
        negative = result['negative_rows']
        self.assertEqual([row['truth_visible_count'] for row in negative], [8, 0, 0])
        self.assertEqual([row['projection_state'] for row in negative], [
            'report_only_projection_available', 'unobservable_empty_truth_visible_set',
            'unobservable_empty_truth_visible_set'])
        self.assertTrue(all(row['full_original_wrong_surface_evidence'] for row in negative))

        # A V-empty negative can still fail the frozen dense-evidence false-acceptance check.
        row50 = next(row for row in conditions
                     if row['condition_id'] == 'syn-50-q8-rgb-t180-rgb')
        row50['arms'][1]['accepted_pose_state'] = 'accepted'
        failed = ablation._exploratory_pose_prerequisite(
            dict(conditions=conditions), visible_sets, baselines, .7)
        self.assertFalse(failed['no_new_negative_false_acceptance'])
        self.assertFalse(failed['passed'])

        # Invalid required positive/self V remains a prerequisite failure.
        visible_sets['zero-10-full'] = dict(count=0, truth_valid=False)
        invalid_calibration = ablation._exploratory_pose_prerequisite(
            dict(conditions=conditions), visible_sets, baselines, .7)
        self.assertFalse(invalid_calibration['positive_and_self_truth_sets_valid'])
        self.assertFalse(invalid_calibration['passed'])

    def test_hash_and_working_output_caps_fail_closed(self):
        payload, selection, _visible, _truth = _scene()
        original = ablation._check_working_budget(payload)
        self.assertLess(original, ablation.MAX_WORKING_BYTES)
        with self.assertRaises(ablation.WorkingSetLimitError):
            ablation._check_working_budget(payload, ablation.MAX_WORKING_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ablation._initial_record(root, ablation._expected_condition_plan())
            with self.assertRaises(ablation.OutputLimitError):
                ablation._write_report(root, dict(large='x' * 100), limit_bytes=1)
            file = root / 'packet.bin'
            file.write_bytes(b'original')
            bundle = dict(input_ledger={str(file): dict(bytes=8,
                                                       sha256=ablation.digest(file))})
            file.write_bytes(b'changed')
            self.assertEqual(ablation._frozen_input_revalidation(bundle)['state'], 'failed')

    def test_terminal_failure_preserves_all_twenty_four_arm_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.mkdir(exist_ok=True)
            record = ablation._initial_record(root, ablation._expected_condition_plan())
            result = ablation._report_failure(root, record, RuntimeError('fixture failure'))
            self.assertEqual(record['status'], 'failed')
            self.assertTrue(record['all_twelve_conditions_accounted'])
            self.assertTrue(record['all_24_arms_accounted'])
            self.assertEqual(sum(len(row['arms']) for row in record['conditions']), 24)
            self.assertTrue(Path(result['path']).is_file())

    def test_actual_closed_twelve_condition_loader_never_calls_fitter(self):
        with mock.patch.object(audit, 'fit_learned_packet', side_effect=AssertionError('fit called')):
            result = ablation.load_closed_conditions_only()
        self.assertEqual(len(result['condition_ids']), 12)
        self.assertEqual(len(result['source_selection']), 6)
        self.assertEqual(len(result['visible_truth_sets']), 12)
        self.assertEqual(len(result['original_dense_baselines']), 12)
        self.assertEqual(len(result['preflight']), 12)
        self.assertTrue(result['all_twelve_loaded_before_any_fit'])
        self.assertTrue(result['source_selection_frozen_before_forward_loading'])
        self.assertEqual(result['fit_calls'], 0)
        self.assertEqual(result['forward_calls'], 0)
        self.assertEqual(result['render_calls'], 0)
        self.assertEqual(result['input_revalidation']['state'], 'matched')
        self.assertTrue(all(row['all_endpoint_sha256'] and row['all_confidence_sha256']
                            for row in result['preflight']))
        self.assertTrue(all(row['observable_endpoint_sha256'] and
                            row['observable_confidence_sha256']
                            for row in result['preflight']))
        self.assertTrue(all(item['eligible_array']['sha256'] for item in
                            result['source_selection'].values()))
        self.assertTrue(all(item['ordered_source_id_sha256'] for item in
                            result['visible_truth_sets'].values()))
        self.assertEqual([result['visible_truth_sets'][
            f'syn-{frame}-q8-rgb-t180-rgb']['count'] for frame in (10, 50, 100)], [8, 0, 0])
        for frame, expected_count in ((10, 13782), (50, 13033), (100, 14874)):
            for condition in ('full', 'clipped'):
                condition_id = f'zero-{frame}-{condition}'
                visible = result['visible_truth_sets'][condition_id]
                baseline = result['original_dense_baselines'][condition_id]
                self.assertEqual(visible['count'], expected_count)
                self.assertEqual(visible['evaluator_mask'],
                                 'R3 frozen common_mask (template_mask & observed_crop_mask)')
                self.assertEqual(baseline['source'], 'unchanged closed R3 CPU score')
                self.assertEqual(baseline['visible_denominator']['count'], expected_count)
                self.assertEqual(baseline['visible_denominator']['source_indices_sha256'],
                                 visible['source_id_sha256'])
        self.assertTrue(all(
            row['source'].startswith('recomputed unchanged R1')
            for condition_id, row in result['original_dense_baselines'].items()
            if condition_id.startswith('syn-')))


if __name__ == '__main__':
    unittest.main()
