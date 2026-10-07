"""CPU fixtures for the frozen R6 current-image patch calibration."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_patch_pose_calibration as r6
from . import quality_bottle_pose_ablation as r5
from .vision import cv2


def _random_rgb(seed=721, size=280):
    rng = np.random.default_rng(seed)
    gray = rng.normal(.5, .18, (size, size)).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), .65)
    gray = np.clip(gray, 0., 1.)
    return np.repeat(gray[..., None], 3, axis=2).astype(np.float32)


def _translated_rgb(rgb, dx, dy):
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    sampled, valid = audit._bilinear_sample(rgb, xx - float(dx), yy - float(dy))
    out = np.asarray(sampled, dtype=np.float32)
    out[~valid] = .5
    return out


def _fixture(rgb=None, center=(140.5, 140.5), points=None, ids=None):
    rgb = _random_rgb() if rgb is None else np.asarray(rgb, dtype=np.float32)
    count = 1 if ids is None else len(ids)
    ids = (np.asarray([int(center[1]) * rgb.shape[1] + int(center[0])], dtype=np.int64)
           if ids is None else np.asarray(ids, dtype=np.int64))
    pixels = np.repeat(np.asarray(center, dtype=np.float64)[None, :], count, axis=0)
    points = (np.zeros((count, 3), dtype=np.float64) if points is None
              else np.asarray(points, dtype=np.float64).reshape(count, 3))
    source = dict(
        template_rgb=rgb,
        template_mask=np.ones(rgb.shape[:2], dtype=bool),
        observed_crop_mask=np.ones(rgb.shape[:2], dtype=bool),
        source_indices=ids,
        source_pixels_xy=pixels,
        source_points_object_m=points,
    )
    eligible = np.ones(count, dtype=bool)
    selection = dict(eligible=eligible, eligible_count=count)
    descriptors = r6.build_source_descriptors(source, selection)
    return source, selection, descriptors


def _fixed_banks(descriptors, points, duplicate_distant=False):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    anchors = []
    for index, (descriptor, point) in enumerate(zip(descriptors['descriptors'], points)):
        anchors.append(dict(descriptor=descriptor.copy(), object_xyz_m=point.copy(),
                            source_index=int(index), source_xy_crop=[140.5, 140.5]))
    if duplicate_distant:
        anchors.append(dict(descriptor=descriptors['descriptors'][0].copy(),
                            object_xyz_m=np.array([1., 0., 0.]), source_index=900001,
                            source_xy_crop=[30.5, 30.5]))
    else:
        anchors.append(dict(descriptor=-descriptors['descriptors'][0].copy(),
                            object_xyz_m=np.array([1., 0., 0.]), source_index=900001,
                            source_xy_crop=[30.5, 30.5]))
    return [dict(anchors=anchors)]


def _match(source, selection, descriptors, query, endpoints, confidence=None,
           observed=None, banks=None, batch_size=8):
    confidence = np.full(len(endpoints), .9, dtype=np.float32) if confidence is None else confidence
    observed = np.ones(query.shape[:2], dtype=bool) if observed is None else observed
    banks = _fixed_banks(descriptors, source['source_points_object_m']) if banks is None else banks
    return r6.match_current_image_patches(
        query, observed, np.asarray(endpoints, dtype=np.float64), confidence,
        source['source_indices'], source['source_points_object_m'], selection['eligible'],
        descriptors, banks, 1., batch_size=batch_size)


class BottlePatchCalibrationTests(unittest.TestCase):
    def test_batch_scalar_sampler_ncc_and_matcher_parity(self):
        rgb = _random_rgb(83)
        centers = np.asarray([[83.5, 88.5], [147.5, 104.5], [190.5, 191.5]], dtype=np.float64)
        ids = np.asarray([int(y) * 280 + int(x) for x, y in centers], dtype=np.int64)
        source, selection, descriptors = _fixture(rgb, tuple(centers[0]),
                                                  points=[[0., 0., 0.], [.2, 0., 0.],
                                                          [.4, 0., 0.]], ids=ids)
        source['source_pixels_xy'][:] = centers
        descriptors = r6.build_source_descriptors(source, selection)
        query = _translated_rgb(rgb, 2.5, -.5)
        q0 = centers + np.array([2.5, -.5]) - np.array([[3., -2.], [-2., 4.], [0., 0.]])
        qhigh = r6._query_highpass(query)
        target_mask = audit._eroded_mask(np.ones(query.shape[:2], dtype=bool), 6)
        batched = r6._score_candidate_batch(qhigh, target_mask, q0,
                                           descriptors['descriptors'])
        scalar = [r6._score_candidate_scalar(qhigh, target_mask, q0[row],
                                             descriptors['descriptors'][row])
                  for row in range(len(q0))]
        for row in range(len(q0)):
            np.testing.assert_array_equal(batched['footprint_valid'][row],
                                          scalar[row]['footprint_valid'])
            np.testing.assert_array_equal(batched['target_valid'][row], scalar[row]['target_valid'])
            np.testing.assert_allclose(batched['scores'][row], scalar[row]['scores'],
                                       atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(batched['descriptors'][row], scalar[row]['descriptors'],
                                       atol=1e-10, rtol=1e-9)
        banks = _fixed_banks(descriptors, source['source_points_object_m'])
        bank_data = r6._bank_descriptor_arrays(banks)
        for row in range(len(q0)):
            resolved_batch = r6._resolve_candidate(
                batched['scores'][row], batched['target_valid'][row],
                batched['footprint_valid'][row], batched['target_std'][row],
                batched['descriptors'][row], bank_data,
                source['source_points_object_m'][row], 1.)
            resolved_scalar = r6._resolve_candidate(
                scalar[row]['scores'], scalar[row]['target_valid'],
                scalar[row]['footprint_valid'], scalar[row]['target_std'],
                scalar[row]['descriptors'], bank_data,
                source['source_points_object_m'][row], 1.)
            self.assertEqual(resolved_batch, resolved_scalar)
        result1 = _match(source, selection, descriptors, query, q0,
                         banks=banks, batch_size=1)
        result8 = _match(source, selection, descriptors, query, q0,
                         banks=banks, batch_size=8)
        np.testing.assert_array_equal(result1['status_codes'], result8['status_codes'])
        np.testing.assert_array_equal(result1['verified_rows'], result8['verified_rows'])
        np.testing.assert_allclose(result1['verified_endpoints'], result8['verified_endpoints'],
                                   atol=1e-10, rtol=1e-9)
        np.testing.assert_array_equal(result1['verified_offsets_xy'], result8['verified_offsets_xy'])
        self.assertEqual(result1['source_outcome_ledger'], result8['source_outcome_ledger'])
        self.assertEqual(result1['correction_summary'], result8['correction_summary'])
        for row, expected in enumerate(([3, -2], [-2, 4], [0, 0])):
            self.assertEqual(result8['verified_offsets_xy'][row].tolist(), expected)

    def test_all_289_integer_offsets_and_subpixel_phase(self):
        rgb = _random_rgb(91)
        center = np.array([142.5, 138.5], dtype=np.float64)
        query = _translated_rgb(rgb, .5, -.25)
        target = center + np.array([.5, -.25])
        offsets = r6.SEARCH_OFFSETS.copy()
        ids = np.arange(10000, 10000 + len(offsets), dtype=np.int64)
        points = np.zeros((len(offsets), 3), dtype=np.float64)
        source, selection, descriptors = _fixture(rgb, tuple(center), points, ids)
        source['source_pixels_xy'][:] = center
        descriptors = r6.build_source_descriptors(source, selection)
        banks = _fixed_banks(descriptors, points)
        q0 = np.repeat(target[None, :], len(offsets), axis=0) - offsets
        result = _match(source, selection, descriptors, query, q0, banks=banks, batch_size=8)
        self.assertEqual(len(result['verified_rows']), len(offsets))
        np.testing.assert_array_equal(result['verified_offsets_xy'], offsets.astype(np.int16))
        true_endpoints = np.repeat(target[None, :], len(offsets), axis=0)
        np.testing.assert_allclose(result['verified_endpoints'], true_endpoints,
                                   atol=1e-10, rtol=1e-9)
        self.assertTrue(np.all(np.linalg.norm(result['verified_endpoints'] - true_endpoints,
                                              axis=1) <= 3.))
        self.assertEqual(result['correction_summary']['search_boundary_count'], 64)
        self.assertEqual(result['search']['candidate_count'], 289)
        self.assertTrue(np.all(result['verified_endpoints'][:, 0] % 1. == .0))
        self.assertTrue(np.all(result['verified_endpoints'][:, 1] % 1. == .25))

    def test_false_support_controls_and_forbidden_fallbacks(self):
        rgb = _random_rgb(121)
        center = np.array([140.5, 140.5], dtype=np.float64)
        source, selection, descriptors = _fixture(rgb, tuple(center))
        banks = _fixed_banks(descriptors, source['source_points_object_m'])

        blank = np.zeros_like(rgb)
        blank_result = _match(source, selection, descriptors, blank, [center], banks=banks)
        self.assertEqual(int(blank_result['source_outcome_ledger']['status_counts']['low_target_texture']), 1)
        self.assertEqual(len(blank_result['verified_rows']), 0)

        repeated = .5 + .2 * np.sin(2. * np.pi * np.arange(280)[None, :, None] / 4.)
        repeated = np.broadcast_to(repeated, (280, 280, 3)).astype(np.float32).copy()
        repeated_ids = np.array([12001, 12002, 12003], dtype=np.int64)
        repeated_source, repeated_selection, repeated_desc = _fixture(
            repeated, tuple(center), np.zeros((3, 3)), repeated_ids)
        repeated_banks = _fixed_banks(repeated_desc, repeated_source['source_points_object_m'])
        repeated_endpoints = np.repeat(center[None, :], 3, axis=0)
        repeated_result = _match(repeated_source, repeated_selection, repeated_desc,
                                 repeated, repeated_endpoints, banks=repeated_banks, batch_size=8)
        repeated_scalar_batching = _match(
            repeated_source, repeated_selection, repeated_desc, repeated,
            repeated_endpoints, banks=repeated_banks, batch_size=1)
        self.assertEqual(len(repeated_result['verified_rows']), 0)
        self.assertEqual(int(repeated_result['source_outcome_ledger']['status_counts']['spatial_ambiguity']), 3)
        np.testing.assert_array_equal(repeated_result['status_codes'],
                                      repeated_scalar_batching['status_codes'])
        self.assertEqual(repeated_result['source_outcome_ledger'],
                         repeated_scalar_batching['source_outcome_ledger'])
        self.assertEqual(repeated_result['matcher_input_hashes'],
                         repeated_scalar_batching['matcher_input_hashes'])
        repeated_high = r6._query_highpass(repeated)
        repeated_mask = audit._eroded_mask(np.ones(repeated.shape[:2], dtype=bool), 6)
        repeated_scores = r6._score_candidate_batch(
            repeated_high, repeated_mask, repeated_endpoints,
            repeated_desc['descriptors'])
        repeated_scalar = r6._score_candidate_scalar(
            repeated_high, repeated_mask, repeated_endpoints[0], repeated_desc['descriptors'][0])
        np.testing.assert_allclose(repeated_scores['scores'][0], repeated_scalar['scores'],
                                   atol=1e-10, rtol=1e-9)
        resolved_batch = r6._resolve_candidate(
            repeated_scores['scores'][0], repeated_scores['target_valid'][0],
            repeated_scores['footprint_valid'][0], repeated_scores['target_std'][0],
            repeated_scores['descriptors'][0], r6._bank_descriptor_arrays(repeated_banks),
            repeated_source['source_points_object_m'][0], 1.)
        resolved_scalar = r6._resolve_candidate(
            repeated_scalar['scores'], repeated_scalar['target_valid'],
            repeated_scalar['footprint_valid'], repeated_scalar['target_std'],
            repeated_scalar['descriptors'], r6._bank_descriptor_arrays(repeated_banks),
            repeated_source['source_points_object_m'][0], 1.)
        self.assertEqual(resolved_batch, resolved_scalar)
        self.assertEqual(resolved_batch['status'], 'spatial_ambiguity')

        duplicate_result = _match(source, selection, descriptors, rgb, [center],
                                  banks=_fixed_banks(descriptors, source['source_points_object_m'],
                                                     duplicate_distant=True))
        self.assertEqual(len(duplicate_result['verified_rows']), 0)
        self.assertEqual(int(duplicate_result['source_outcome_ledger']['status_counts'][
            'ncc_or_distant_margin_rejection']), 1)

        blocked = np.zeros(rgb.shape[:2], dtype=bool)
        blocked_result = _match(source, selection, descriptors, rgb, [center],
                                observed=blocked, banks=banks)
        self.assertEqual(len(blocked_result['verified_rows']), 0)
        self.assertEqual(int(blocked_result['source_outcome_ledger']['status_counts']['footprint_rejection']), 1)
        edge_result = _match(source, selection, descriptors, rgb, [[-3.5, center[1]]],
                             observed=np.ones(rgb.shape[:2], dtype=bool), banks=banks)
        self.assertEqual(len(edge_result['verified_rows']), 0)
        self.assertEqual(int(edge_result['source_outcome_ledger']['status_counts'][
            'footprint_rejection']), 1)

        invalid_result = _match(source, selection, descriptors, rgb,
                                [[np.nan, 140.5]], confidence=[.9], banks=banks)
        self.assertEqual(int(invalid_result['source_outcome_ledger']['status_counts'][
            'nonfinite_endpoint_or_confidence']), 1)
        low_confidence = _match(source, selection, descriptors, rgb, [center],
                                confidence=[.3], banks=banks)
        self.assertEqual(int(low_confidence['source_outcome_ledger']['status_counts']['low_confidence']), 1)
        out_of_range = _match(source, selection, descriptors, rgb, [[-100., 140.5]], banks=banks)
        self.assertEqual(int(out_of_range['source_outcome_ledger']['status_counts']['footprint_rejection']), 1)
        no_competitor = _match(source, selection, descriptors, rgb, [center], banks=[])
        self.assertEqual(len(no_competitor['verified_rows']), 0)
        self.assertEqual(int(no_competitor['source_outcome_ledger']['status_counts'][
            'missing_distant_competitor']), 1)

        outside_query = _translated_rgb(rgb, 20., 0.)
        outside = _match(source, selection, descriptors, outside_query, [center], banks=banks)
        self.assertEqual(len(outside['verified_rows']), 0)
        self.assertEqual(int(outside['source_outcome_ledger']['status_counts']['verified_unchanged']) +
                         int(outside['source_outcome_ledger']['status_counts']['verified_corrected']), 0)

    def test_truth_mutation_cannot_change_selector_or_matcher(self):
        rgb = _random_rgb(181)
        center = (140.5, 140.5)
        source, selection, descriptors = _fixture(rgb, center)
        query = _translated_rgb(rgb, 2.5, 1.25)
        endpoint = np.asarray([center], dtype=np.float64) + [[2., 1.]]
        first = _match(source, selection, descriptors, query, endpoint)
        evaluator = dict(pose=np.eye(4), depth=np.ones((280, 280)), mask=np.ones((280, 280), bool))
        second = _match(source, selection, descriptors, query, endpoint)
        selector_a = r5.freeze_source_selection(source)
        evaluator['pose'][:] = np.nan
        evaluator['depth'][:] = 0.
        evaluator['mask'][:] = False
        selector_b = r5.freeze_source_selection(source)
        third = _match(source, selection, descriptors, query, endpoint)
        self.assertEqual(selector_a['eligible_array']['sha256'], selector_b['eligible_array']['sha256'])
        self.assertEqual(first['matcher_input_hashes'], second['matcher_input_hashes'])
        self.assertEqual(first['matcher_input_hashes'], third['matcher_input_hashes'])
        np.testing.assert_array_equal(first['status_codes'], third['status_codes'])
        np.testing.assert_allclose(first['verified_endpoints'], third['verified_endpoints'],
                                   atol=0., rtol=0.)

    def test_native_rays_and_nonidentity_crop_mm_m_parity(self):
        angle = .19
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.],
                             [0., 0., 1.]], dtype=np.float64)
        crop_from_native = np.eye(4, dtype=np.float64)
        crop_from_native[:3, :3] = rotation
        crop_from_native[:3, 3] = [12., -7., 4.]
        crop_k = np.array([[112., 0., 140.], [0., 108., 136.], [0., 0., 1.]])
        native_k = np.array([[420., 0., 640.], [0., 415., 360.], [0., 0., 1.]])
        endpoints = np.array([[87.25, 102.75], [171.5, 199.125]], dtype=np.float64)
        actual = r5._native_xy_from_crop_endpoints(endpoints, crop_k, crop_from_native, native_k)
        crop_rays = np.column_stack((endpoints, np.ones(len(endpoints)))) @ np.linalg.inv(crop_k).T
        native_rays = crop_rays @ crop_from_native[:3, :3]
        projected = native_rays @ native_k.T
        expected = projected[:, :2] / projected[:, 2:3]
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)

        crop_pose_m = np.eye(4, dtype=np.float64)
        crop_pose_m[:3, 3] = [.12, -.06, .91]
        crop_pose_mm = crop_pose_m.copy()
        crop_pose_mm[:3, 3] *= 1000.
        native_pose_m = audit.crop_to_native_pose_m(crop_pose_mm, crop_from_native)
        parity = r5._unit_pose_parity(dict(pose_crop_m=crop_pose_m.tolist(),
                                           pose_native_m=native_pose_m.tolist()),
                                     crop_from_native)
        self.assertEqual(parity['state'], 'checked')
        self.assertLess(parity['native_translation_roundtrip_error_m'], 1e-12)
        self.assertLess(parity['crop_translation_roundtrip_error_m'], 1e-12)
        self.assertLess(parity['native_rotation_roundtrip_max_abs'], 1e-12)
        self.assertLess(parity['crop_rotation_roundtrip_max_abs'], 1e-12)
        self.assertEqual(parity['metres_to_millimetres_factor'], 1000.)
        self.assertEqual(parity['millimetres_to_metres_factor'], .001)

    def test_support_geometry_and_immutable_conditional_denominators(self):
        xx, yy = np.meshgrid(np.linspace(-.4, .4, 5), np.linspace(-.4, .4, 5))
        points = np.column_stack((xx.ravel(), yy.ravel(), .04 * np.sin(xx.ravel() * 4)))
        pixels = np.column_stack((np.linspace(24.5, 255.5, len(points)),
                                  np.tile(np.linspace(24.5, 255.5, 5), 5)))
        source = dict(source_points_object_m=points, source_pixels_xy=pixels,
                      source_indices=np.arange(len(points), dtype=np.int64),
                      template_pose_m=np.eye(4), crop_from_native=np.eye(4),
                      crop_k=np.array([[100., 0., 140.], [0., 100., 140.], [0., 0., 1.]]))
        source['template_pose_m'][2, 3] = 1.
        mask = np.ones((280, 280), dtype=bool)
        rows = np.arange(len(points), dtype=np.int64)
        metrics = r6._pose_jacobian_metrics(source, rows, mask, pixels.copy(), 1.)
        self.assertEqual(metrics['point_count'], 25)
        self.assertEqual(len(metrics['point_singular_values']), 3)
        self.assertEqual(len(metrics['pose_jacobian_singular_values']), 6)
        self.assertEqual(metrics['pose_jacobian_rank'], 6)
        self.assertEqual(metrics['source_support']['source_count'], 25)
        self.assertEqual(metrics['current_support']['source_count'], 25)
        self.assertGreaterEqual(metrics['source_support']['grid_cells_4x4'], 3)

        ids = np.arange(8, dtype=np.int64)
        eligible = np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
        visible = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        correct = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
        summary, denominator, verified_visible = r6._conditional_visible_summary(
            ids, eligible, visible, [0, 2, 7], correct)
        self.assertEqual(int(denominator.sum()), 3)
        self.assertEqual(int(verified_visible.sum()), 1)
        self.assertEqual(summary['eligible_true_visible_denominator'], 3)
        self.assertEqual(summary['verified_confident_true_visible_count'], 1)
        self.assertAlmostEqual(summary['verified_confident_coverage'], 1. / 3.)
        self.assertEqual(summary['correct_verified_confident_true_visible'], 1)
        self.assertFalse(summary['empty_eligible_V_unavailable'])
        empty, empty_denominator, _ = r6._conditional_visible_summary(
            ids, np.zeros(8, bool), visible, [], correct)
        self.assertEqual(empty_denominator.sum(), 0)
        self.assertIsNone(empty['verified_confident_coverage'])
        self.assertTrue(empty['empty_eligible_V_unavailable'])

        before = r6._visible_set_hash(ids, visible)
        endpoints_a = np.zeros((8, 2), dtype=np.float64)
        endpoints_b = np.full((8, 2), 1e6, dtype=np.float64)
        self.assertFalse(np.array_equal(endpoints_a, endpoints_b))
        self.assertEqual(before, r6._visible_set_hash(ids, visible))

    def test_unavailable_candidate_repeat_is_deterministic_and_gate_safe(self):
        stable = dict(decision='unavailable', reason='fewer_than_24', retained=3)
        stable_repeat = r6._repeat_unavailable_precondition_decision(
            stable, lambda: dict(decision='unavailable', reason='fewer_than_24', retained=3))
        changed_repeat = r6._repeat_unavailable_precondition_decision(
            stable, lambda: dict(decision='fit_preconditions_passed', reason=None, retained=24))
        self.assertTrue(stable_repeat['matched'])
        self.assertEqual(stable_repeat['state'], 'matched_no_fit_unavailable_decision')
        self.assertFalse(changed_repeat['matched'])

        plans = r5._expected_condition_plan()
        candidate_arms, r5_rows = {}, []
        for plan in plans:
            cid, kind, frame = plan['condition_id'], plan['kind'], plan['frame_id']
            if kind == 'synthetic_positive':
                baseline = dict(rotation_error_degrees=2., translation_error_m=.02,
                                translation_error_fraction_D=.02)
                candidate = dict(accepted_pose_state='accepted',
                                 proposal_rotation_error=dict(rotation_error_degrees=1.5,
                                                              translation_error_m=.015,
                                                              translation_error_fraction_D=.015),
                                 deterministic_repeat=dict(matched=True))
            elif kind == 'self_control':
                baseline = dict(rotation_error_degrees=.5, translation_error_m=.005,
                                translation_error_fraction_D=.005)
                candidate = dict(accepted_pose_state='accepted',
                                 proposal_rotation_error=dict(rotation_error_degrees=.4,
                                                              translation_error_m=.004,
                                                              translation_error_fraction_D=.004),
                                 deterministic_repeat=dict(matched=True))
            else:
                baseline = None
                candidate = dict(accepted_pose_state='unavailable', proposal_pose=None,
                                 deterministic_repeat=stable_repeat)
            arms = [dict(arm='all', proposal_rotation_error=baseline),
                    dict(arm='source_observable', proposal_rotation_error=baseline)]
            r5_rows.append(dict(condition_id=cid, arms=arms,
                                original_dense_baseline=dict(
                                    original_wrong_surface_evidence={'count': 1})))
            candidate_arms[cid] = candidate
        gates = r6._pose_candidate_gates([], dict(conditions=r5_rows), candidate_arms)
        self.assertTrue(gates['zero_accepted_opposite_template_candidates'])
        self.assertTrue(gates['every_candidate_deterministic'])
        self.assertTrue(gates['passed'])
        negative_id = 'syn-10-q8-rgb-t180-rgb'
        candidate_arms[negative_id]['deterministic_repeat'] = changed_repeat
        changed_gates = r6._pose_candidate_gates([], dict(conditions=r5_rows), candidate_arms)
        self.assertFalse(changed_gates['every_candidate_deterministic'])
        self.assertFalse(changed_gates['passed'])

    def test_measurement_boundary_preserves_full_verified_and_separates_masks(self):
        ids = np.array([0, 1, 2], dtype=np.int64)
        observed = np.ones((280, 280), dtype=bool)
        truth_mask = np.zeros((280, 280), dtype=bool)
        confidence = np.array([.9, .8, .7], dtype=np.float32)
        endpoints = np.array([[140.5, 140.5], [141.5, 140.5], [142.5, 140.5]],
                             dtype=np.float64)
        points = np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]], dtype=np.float64)
        source = dict(
            observed_crop_mask=observed, source_indices=ids,
            source_pixels_xy=endpoints.copy(), source_points_object_m=points,
            template_pose_m=np.eye(4), crop_from_native=np.eye(4),
            crop_k=np.eye(3), native_k=np.eye(3))
        source['template_pose_m'][2, 3] = 1.
        frame = dict(native_k=np.eye(3), native_mask=np.ones((280, 280), dtype=np.uint8),
                     native_rgb=np.zeros((280, 280, 3), dtype=np.uint8))
        flow = np.zeros((280, 280, 2), dtype=np.float32)
        confidence_field = np.ones((280, 280), dtype=np.float32)
        payload = dict(
            plan=dict(condition_id='syn-10-q8-rgb-t0-rgb', kind='synthetic_positive',
                      frame_id=10, template_offset_deg=0),
            source=source, frame=frame,
            query=dict(known_query_pose_m=np.eye(4), query_depth_mm=np.ones((280, 280)),
                       synthetic_query_mask=truth_mask),
            forward=dict(flow=flow, confidence=confidence_field),
            endpoints=endpoints.copy(), confidence=confidence.copy(), seed=34,
            provenance={'template': {'sha256': '1' * 64}},
        )
        eligible = np.array([True, True, True], dtype=bool)
        selection = dict(eligible=eligible, eligible_count=3,
                         eligible_array={'sha256': '2' * 64})
        visible = np.array([True, False, False])
        visible_set = dict(count=1,
                           source_id_sha256=r6._visible_set_hash(ids, visible)[
                               'ordered_source_ids_sha256'])
        descriptor = dict(descriptor_sha256='3' * 64)
        verified_rows = np.array([0, 2], dtype=np.int64)
        match = dict(
            verified_rows=verified_rows,
            verified_endpoints=endpoints[verified_rows] + [[1., 0.], [2., 0.]],
            verified_offsets_xy=np.array([[1, 0], [2, 0]], dtype=np.int16),
            verified_confidence=confidence[verified_rows].copy(),
            source_outcome_ledger=dict(status_counts={name: 0 for name in r6.STATUS_NAMES}),
            correction_summary={}, search={'candidate_count': 289},
            matcher_input_hashes={}, query_highpass_sha256='4' * 64,
            query_observed_eroded_mask_sha256='5' * 64)
        true_visible = visible.copy()
        correct = np.array([True, False, False])
        metric_arrays = dict(
            true_visible=true_visible, correct_identity=correct,
            identity_distance_fraction=np.array([.01, .2, .3]))
        metrics = dict(thresholds={}, confidence_availability_on_visible=1.,
                       correct_fraction_confident_visible=1.)
        evaluator_masks = []

        def fixed_anchor(_rgb, mask, *_args):
            evaluator_masks.append(np.asarray(mask, dtype=bool).copy())
            state = ('distinctive_current_image_support' if np.asarray(mask).all()
                     else 'unobservable_or_ambiguous')
            return dict(summary=dict(state=state, supported=8,
                                     supported_cells_4x4=3,
                                     supported_hull_fraction=.12), rows=[])

        source_id = r5._source_template_refs(10, 0)
        opposite_id = r5._source_template_refs(10, 180)
        banks = {
            source_id: dict(anchors=[dict(source_index=0)], entry={'sha256': '6' * 64},
                            manifest_sha256='7' * 64, selected_anchor_count=1),
            opposite_id: dict(anchors=[dict(source_index=1)], entry={'sha256': '8' * 64},
                             manifest_sha256='9' * 64, selected_anchor_count=1),
        }
        query_rgb = _random_rgb(291)
        with mock.patch.object(r6, '_condition_query_rgb',
                               return_value=(query_rgb, audit.array_sha256(query_rgb), 'toy')), \
                mock.patch.object(r6, 'build_source_descriptors', return_value=descriptor), \
                mock.patch.object(r6, 'match_current_image_patches', return_value=match) as matcher, \
                mock.patch.object(r6, '_evaluator_inputs', return_value=dict(
                    pose_crop_m=np.eye(4), depth_mm=np.ones((280, 280)),
                    mask=truth_mask, mask_role='toy evaluator mask')), \
                mock.patch.object(audit, 'synthetic_identity_metrics',
                                  return_value=(metrics, metric_arrays)), \
                mock.patch.object(audit, 'audit_patch_identity', side_effect=fixed_anchor), \
                mock.patch.object(r6, '_geometry_for_condition', return_value={'groups': {}}):
            row, fit_input = r6._condition_measurement(
                {}, payload['plan'], payload, selection, banks, visible_set,
                dict(original_dense_baseline=dict(dense_qualification=True)), 1.)
        matcher.assert_called_once()
        self.assertTrue(np.array_equal(matcher.call_args.args[1], observed))
        self.assertEqual(len(evaluator_masks), 2)
        self.assertTrue(evaluator_masks[0].all())
        self.assertFalse(evaluator_masks[1].any())
        self.assertEqual(row['evaluator_only_evidence']['verified_confident_invisible_sources'], 1)
        self.assertEqual(row['evaluator_only_evidence']['verified_invisible_wrong_surface_count'], 1)
        self.assertEqual(row['conditional_observable_measurement']['eligible_true_visible_denominator'], 1)
        self.assertEqual(row['conditional_observable_measurement']['verified_confident_true_visible_count'], 1)
        self.assertTrue(row['fixed_anchor_spatial_calibration']['qualified'])
        self.assertFalse(row['legacy_evaluator_mask_fixed_anchor_audit']['qualified'])
        self.assertFalse(row['hypothetical_full_field']['dense_qualification'])
        np.testing.assert_array_equal(fit_input['source_rows'], verified_rows)
        np.testing.assert_array_equal(fit_input['confidence'], confidence[verified_rows])
        self.assertEqual(row['endpoint_evidence']['original_endpoint_sha256'],
                         r6._array_sha256(endpoints))
        self.assertEqual(row['endpoint_evidence']['original_confidence_sha256'],
                         r6._array_sha256(confidence))
        self.assertTrue(np.all(flow == 0.))

    def test_working_limit_includes_all_twelve_candidate_buffers_before_matching(self):
        n = 36000
        entry = dict(arrays={
            'source_indices': dict(shape=[n], dtype='int64'),
            'source_pixels_xy': dict(shape=[n, 2], dtype='float64'),
            'source_points_object_m': dict(shape=[n, 3], dtype='float64'),
            'observed_crop_mask': dict(shape=[280, 280], dtype='bool'),
            'template_rgb': dict(shape=[280, 280, 3], dtype='float32'),
        })
        row_alone = r6._estimated_working_bytes(
            [(Path('.'), entry)], n, retained_fit_input_bytes=0)
        self.assertLess(row_alone['estimated_peak_bytes'], r6.MAX_WORKING_BYTES)
        matcher = mock.Mock(side_effect=AssertionError('matcher crossed a failed memory preflight'))
        plans = r5._expected_condition_plan()
        template_entry = dict(arrays={
            'source_indices': dict(shape=[n], dtype='int64'),
            'source_pixels_xy': dict(shape=[n, 2], dtype='float64'),
            'source_points_object_m': dict(shape=[n, 3], dtype='float64'),
            'observed_crop_mask': dict(shape=[280, 280], dtype='bool'),
            'template_rgb': dict(shape=[280, 280, 3], dtype='float32'),
        })
        forward_entry = dict(arrays={'confidence': dict(shape=[280, 280], dtype='float32')})
        selectors = {r5._source_template_refs(frame, offset): dict(eligible_count=n)
                     for frame in r6.CARRIERS for offset in (0, 180)}
        with mock.patch.object(r6, '_condition_entries', return_value=[
                (Path('.'), template_entry), (Path('.'), forward_entry)]), \
                mock.patch.object(r6, 'match_current_image_patches', matcher):
            with self.assertRaises(r6.WorkingSetLimitError):
                r6._preflight_working_sets(
                    {}, plans, selectors, resident_state_bytes=80 * 1024**2)
        matcher.assert_not_called()

    def test_calibration_failure_accounts_twelve_rows_and_skips_every_fit(self):
        plans = r5._expected_condition_plan()
        template_ids = {r5._source_template_refs(plan['frame_id'],
                                                plan['template_offset_deg']) for plan in plans}
        selections = {key: dict(eligible_count=1, eligible=np.ones(1, dtype=bool))
                      for key in template_ids}
        visible = {plan['condition_id']: dict(count=1, source_id_sha256='0' * 64)
                   for plan in plans}
        banks = {key: dict(entry=dict(sha256='a' * 64), manifest_sha256='b' * 64,
                           selected_anchor_count=1) for key in template_ids}
        r5_report = dict(conditions=[dict(condition_id=plan['condition_id']) for plan in plans])
        bundle = dict(
            capture_root=r6.CAPTURE_ROOT, zero_root=r6.ZERO_ROOT,
            capture_sha256=r6.EXPECTED_R1_CAPTURE_SHA256,
            zero_sha256=r6.EXPECTED_R3_REPORT_SHA256,
            r5_report_sha256=r6.EXPECTED_R5_REPORT_SHA256,
            r5_terminal_sha256=r6.EXPECTED_R5_TERMINAL_SHA256,
            capture=dict(object_bbox_diagonal_m=1.),
            source_pins=dict(capture_snapshot={'sha256': 'c' * 64}),
            r6_source_pins={}, source_selection_summary={}, r5_report=r5_report)

        def failed_measurement(_bundle, plan, *_args):
            result = dict(condition_id=plan['condition_id'], state='measured',
                          gates=dict(conditional_coverage_passed=False,
                                     conditional_identity_passed=False,
                                     fixed_anchor_spatial_calibration_passed=False,
                                     passed=False),
                          evaluator_only_evidence={})
            if plan['kind'] == 'synthetic_negative':
                result['evaluator_only_evidence'] = dict(
                    visible_count={10: 8, 50: 0, 100: 0}[plan['frame_id']],
                    expected_negative_V_count={10: 8, 50: 0, 100: 0}[plan['frame_id']],
                    negative_V_count_matches_frozen_cohort=True,
                    original_wrong_surface_evidence={})
            if plan['kind'] == 'self_control':
                result['gates']['self_full_V_dense_qualification_retained'] = False
            return result, dict(test_fixture=True)

        budgets = dict(rows=[dict(condition_id=plan['condition_id'],
                                  working_set=dict(estimated_peak_bytes=1024)) for plan in plans])
        matcher = mock.Mock(side_effect=AssertionError('calibration failure entered matcher'))
        fitter = mock.Mock(side_effect=AssertionError('calibration failure entered fitter'))
        with tempfile.TemporaryDirectory(dir=r6.CACHE / 'diagnostics') as temporary:
            output_root = Path(temporary) / 'fresh-output'
            with mock.patch.object(r6, '_load_r6_bundle', return_value=bundle), \
                    mock.patch.object(r6, '_run_calibration_toys', return_value=dict(passed=True)), \
                    mock.patch.object(r6, '_record_r5_controls'), \
                    mock.patch.object(r6, '_source_selector_working_preflight',
                                      return_value=dict(max_estimated_peak_bytes=1024)), \
                    mock.patch.object(r6, '_freeze_and_bind_source_selections',
                                      return_value=selections), \
                    mock.patch.object(r6, '_unique_array_owner_bytes', return_value=0), \
                    mock.patch.object(r6, '_visible_set_upper_bound', return_value=0), \
                    mock.patch.object(r6, '_preflight_working_sets', return_value=budgets), \
                    mock.patch.object(r6.r5, 'freeze_visible_evaluation_sets', return_value=visible), \
                    mock.patch.object(r6, '_load_fixed_banks', return_value=banks), \
                    mock.patch.object(r6.r5, '_frozen_input_revalidation',
                                      return_value=dict(state='matched')), \
                    mock.patch.object(r6.r5, '_load_condition_payload',
                                      side_effect=lambda _bundle, plan: dict(plan=plan)), \
                    mock.patch.object(r6, '_condition_measurement', side_effect=failed_measurement), \
                    mock.patch.object(r6, '_fit_candidate', fitter), \
                    mock.patch.object(r6, 'match_current_image_patches', matcher):
                record = r6.run_calibration(output_root=output_root)
        self.assertEqual(record['status'], 'calibration_failed')
        self.assertFalse(record['calibration_passed'])
        self.assertTrue(record['all_twelve_conditions_accounted'])
        self.assertTrue(record['all_twelve_candidates_accounted'])
        self.assertEqual(record['candidate_fit_calls'], 0)
        self.assertEqual(len(record['conditions']), 12)
        self.assertTrue(all(row['state'] == 'measured' for row in record['conditions']))
        self.assertTrue(all(row['candidate']['state'] == 'candidate_not_run'
                            for row in record['conditions']))
        matcher.assert_not_called()
        fitter.assert_not_called()

    def test_descriptor_and_atomic_output_failures_are_preserved(self):
        with self.assertRaises(r6.CalibrationInputError):
            r6._entry_array_nbytes(dict(path='bad.npz', arrays={'x': dict(shape=[-1, 3], dtype='float32')}))
        with self.assertRaises(r6.CalibrationInputError):
            r6._entry_array_nbytes(dict(path='bad.npz', arrays={'x': dict(shape=[2], dtype='no-such-dtype')}))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'report.json'
            with self.assertRaises(r6.OutputLimitError):
                r6._atomic_json(report, {'payload': 'x' * 100}, max_bytes=16)
            self.assertFalse(report.exists())
            result = r6._atomic_json(report, {'ok': True})
            self.assertTrue(report.is_file())
            self.assertEqual(result['sha256'], r6.digest(report))
            with mock.patch.object(r6, 'MAX_OUTPUT_BYTES', 20):
                with self.assertRaises(r6.OutputLimitError):
                    r6._write_terminal(root, {'state': 'terminal', 'payload': 'x' * 32})
            terminal = root / 'terminal.json'
            r6._atomic_json(terminal, {'state': 'terminal'}, exclusive=True)
            with self.assertRaises(FileExistsError):
                r6._atomic_json(terminal, {'state': 'replacement'}, exclusive=True)
            self.assertEqual(r6._serialize_record({'state': 'terminal'}), terminal.read_bytes())

    def test_actual_cache_loader_stops_before_matcher_and_fitter(self):
        matcher = mock.Mock(side_effect=AssertionError('actual preflight called matcher'))
        fitter = mock.Mock(side_effect=AssertionError('actual preflight called fitter'))
        with mock.patch.object(r6, 'match_current_image_patches', matcher), \
                mock.patch.object(r6.audit, 'fit_learned_packet', fitter):
            result = r6.preflight_actual_cache()
        self.assertEqual(len(result['condition_ids']), 12)
        self.assertEqual(result['fixed_bank_count'], 6)
        self.assertEqual(len(result['source_selection']), 6)
        self.assertEqual(len(result['preflight']), 12)
        self.assertIn('bench/quality_bottle_patch_pose_calibration.py', result['r6_source_pins'])
        self.assertIn('bench/test_quality_bottle_patch_pose_calibration.py', result['r6_source_pins'])
        self.assertEqual(result['r6_source_pins']['bench/quality_bottle_pose_ablation.py'],
                         r6.EXPECTED_R5_SOURCE_SHA256)
        self.assertEqual(result['matcher_calls'], 0)
        self.assertEqual(result['fitter_calls'], 0)
        self.assertFalse(result['matching_started'])
        self.assertFalse(result['fitting_started'])
        self.assertLess(result['max_estimated_working_bytes'], r6.MAX_WORKING_BYTES)
        self.assertEqual(result['input_revalidation']['state'], 'matched')
        matcher.assert_not_called()
        fitter.assert_not_called()


def run_toy_fixture_checks():
    """Small fail-fast controls run before any actual-row matching or fitting."""
    suite = unittest.TestSuite()
    for name in (
            'test_batch_scalar_sampler_ncc_and_matcher_parity',
            'test_all_289_integer_offsets_and_subpixel_phase',
            'test_false_support_controls_and_forbidden_fallbacks',
            'test_truth_mutation_cannot_change_selector_or_matcher',
            'test_native_rays_and_nonidentity_crop_mm_m_parity',
            'test_support_geometry_and_immutable_conditional_denominators'):
        suite.addTest(BottlePatchCalibrationTests(name))
    outcome = unittest.TestResult()
    suite.run(outcome)
    if outcome.errors or outcome.failures:
        detail = '\n'.join(item[1] for item in outcome.errors + outcome.failures)
        return dict(passed=False, tests_run=outcome.testsRun,
                    errors=len(outcome.errors), failures=len(outcome.failures),
                    detail=detail[-4000:])
    return dict(passed=True, tests_run=outcome.testsRun,
                errors=0, failures=0,
                controls=['scalar/batched parity', 'all 289 integer offsets',
                          'fractional pixel phase', 'zero-support false cases',
                          'truth mutation invariance', 'native/crop mm/m parity',
                          'support, geometry and immutable V denominators'])


if __name__ == '__main__':
    unittest.main()
