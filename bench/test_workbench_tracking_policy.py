"""Model-free policy controls; synthetic rows are not tracking quality evidence."""
import copy
import dataclasses
import hashlib
import unittest

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_tracking_policy as tp


def row(n=0, *, state='tracking', mask=True, timestamp=True, pose=None):
    p = np.eye(4) if pose is None else pose
    p[2, 3] = 1 if pose is None else p[2, 3]
    return wc.FrameResult(run_id='run', clip_id='clip', frame_id=f'f{n}', source_frame_index=n,
        timestamp_s=n*.03 if timestamp else None, media_time_s=None, backend_role='primary_neural',
        backend_identity={'head': 'synthetic'}, mask_state='observed' if mask else 'unavailable',
        pose_state=state, render_state='rendered' if state == 'tracking' else 'unavailable',
        failure_reason=None if state == 'tracking' else 'synthetic loss',
        pose=tuple(tuple(float(v) for v in a) for a in p) if state == 'tracking' else None,
        pose_units='metric' if state == 'tracking' else None, image_sha256='a'*64,
        mask_frame_id=f'f{n}' if mask else None, mask_sha256='b'*64 if mask else None, metrics={})


def surface_fixture():
    chart = dict(chart_id='chart', domain='plane', orientation_valid=True, overlap_free=True,
                 distortion=.01, max_distortion=.05)
    anchor = dict(anchor_id='anchor', chart_id='chart', geometry_sha256='a'*64)
    admission = tp.SurfaceAdmission('metric', 'dataset_metric_cad', 'b'*64, 'a'*64,
                                   wc.canonical_sha256(anchor), wc.canonical_sha256(chart))
    return admission, anchor, chart


class TrackingPolicyTests(unittest.TestCase):
    def test_every_state_and_mask_trust_combination(self):
        for state in ('tracking', 'recovering', 'lost', 'unavailable', 'blocked'):
            for trusted in (False, True):
                with self.subTest(state=state, trusted=trusted):
                    policy = tp.TrackingPolicy(allow_prediction=True)
                    policy.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
                    d = policy.consume(row(1, state=state), ordinal=1, epoch=0, current_mask_trusted=trusted)
                    expected = ('measured' if trusted else 'lost') if state == 'tracking' else (
                        'recovering' if state == 'recovering' else 'predicted' if trusted and state in ('lost', 'unavailable') else 'lost')
                    self.assertEqual(d.state, expected)
                    self.assertEqual(d.accepted_measured, expected == 'measured')
                    if expected in ('recovering', 'lost'):
                        self.assertEqual((d.pose, d.opacity), (None, 0))
                    with self.assertRaises(dataclasses.FrozenInstanceError):
                        d.opacity = 1

    def test_expiry_fade_and_null_timestamp(self):
        for timestamp in (True, False):
            policy = tp.TrackingPolicy(allow_prediction=True)
            policy.consume(row(timestamp=timestamp), ordinal=0, epoch=0, current_mask_trusted=True)
            decisions = [policy.consume(row(n, state='lost', timestamp=timestamp), ordinal=n, epoch=0,
                                        current_mask_trusted=True) for n in (1, 2, 3)]
            self.assertEqual([d.state for d in decisions], ['predicted', 'predicted', 'lost'])
            self.assertEqual([d.opacity for d in decisions], [.67, .33, 0])
            self.assertEqual(decisions[1].age_frames, 2)
            self.assertEqual(decisions[1].age_ms, 60 if timestamp else None)
            self.assertTrue(all(not d.accepted_measured for d in decisions))

    def test_clock_expiry_epoch_and_default_no_prediction(self):
        policy = tp.TrackingPolicy(allow_prediction=True)
        policy.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
        late = dataclasses.replace(row(1, state='lost'), timestamp_s=.101)
        self.assertEqual(policy.consume(late, ordinal=1, epoch=0, current_mask_trusted=True).state, 'lost')
        policy.reset(1)
        self.assertEqual(policy.consume(row(0, state='lost'), ordinal=0, epoch=1, current_mask_trusted=True).state, 'lost')
        with self.assertRaises(tp.PolicyRejected):
            policy.consume(row(1), ordinal=1, epoch=0, current_mask_trusted=True)
        plain = tp.TrackingPolicy()
        plain.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
        self.assertEqual(plain.consume(row(1, state='lost'), ordinal=1, epoch=0, current_mask_trusted=True).state, 'lost')

    def test_missing_current_mask_or_ambiguity_immediately_suppresses(self):
        for mask, ambiguous in ((False, False), (True, True)):
            p = tp.TrackingPolicy(allow_prediction=True)
            p.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
            d = p.consume(row(1, state='lost', mask=mask), ordinal=1, epoch=0,
                          current_mask_trusted=True, ambiguous_identity=ambiguous)
            self.assertEqual((d.state, d.pose), ('lost', None))

    def test_invalid_units_identity_clock_and_skipped_ordinals_refuse(self):
        variants = [dataclasses.replace(row(1), pose_units='object_unit'),
                    dataclasses.replace(row(1), clip_id='other'),
                    dataclasses.replace(row(1), timestamp_s=0), dataclasses.replace(row(1), timestamp_s=None)]
        for bad in variants:
            p = tp.TrackingPolicy(); p.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
            with self.assertRaises((tp.PolicyRejected, wc.ContractError)):
                p.consume(bad, ordinal=1, epoch=0, current_mask_trusted=True)
        p = tp.TrackingPolicy()
        p.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
        with self.assertRaises(tp.PolicyRejected):
            p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True)

    def proof(self, n, **changes):
        return dataclasses.replace(tp.RecoveryObservation('user-approved', f'obs{n}', f'f{n}', 0,
            'a'*64, 'b'*64, 'onboarding:fresh', True, True, True, True, True, 'd'*64), **changes)

    def lost_policy(self):
        p = tp.TrackingPolicy()
        p.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
        p.consume(row(1, state='lost'), ordinal=1, epoch=0, current_mask_trusted=True)
        return p

    def test_fresh_two_observation_recovery_and_bad_evidence(self):
        for changes in ({'fresh_seed':False}, {'seed_label':'controlled_init:consumed'},
                        {'seed_label':'scored_reference:pose'}, {'frame_id':'f99'},
                        {'mask_sha256':'c'*64}, {'epoch':1}, {'identity_valid':False}):
            p = self.lost_policy()
            for n in (2, 3):
                d = p.consume(row(n), ordinal=n, epoch=0, current_mask_trusted=True,
                              recovery=self.proof(n, **changes))
                self.assertEqual((d.state, d.pose), ('recovering', None))
        p = self.lost_policy()
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(self.proof(2), self.proof(3)), consumed_seed_sha256='c'*64)
        first = p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=self.proof(2))
        second = p.consume(row(3), ordinal=3, epoch=0, current_mask_trusted=True, recovery=self.proof(3))
        self.assertEqual((first.state, first.pose), ('recovering', None))
        self.assertEqual(second.state, 'measured')

    def test_recovery_consistency_and_consecutive_gate(self):
        p = self.lost_policy()
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=tuple(self.proof(n) for n in (2,3,5)), consumed_seed_sha256='c'*64)
        p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=self.proof(2))
        moved = np.eye(4); moved[0, 3] = .11; moved[2, 3] = 1
        self.assertEqual(p.consume(row(3, pose=moved), ordinal=3, epoch=0,
            current_mask_trusted=True, recovery=self.proof(3)).state, 'recovering')
        p.consume(row(4, state='recovering'), ordinal=4, epoch=0, current_mask_trusted=True)
        self.assertEqual(p.consume(row(5), ordinal=5, epoch=0,
            current_mask_trusted=True, recovery=self.proof(5)).state, 'recovering')

    def test_surface_units_calibration_anchor_chart_integrity(self):
        admission, anchor, chart = surface_fixture()
        verify = lambda a=admission, ar=anchor, ch=chart, k='b'*64: a.verify(
            calibration_sha256=k, geometry_sha256='a'*64, anchor_record=ar, chart_record=ch)
        verify()
        for units in ('object_unit', 'unknown', 'relative'):
            with self.assertRaises(tp.PolicyRejected):
                verify(dataclasses.replace(admission, units=units))
        for provenance in ('relative', 'unknown'):
            with self.assertRaises(tp.PolicyRejected):
                verify(dataclasses.replace(admission, metric_provenance=provenance))
        with self.assertRaises(tp.PolicyRejected): verify(k='c'*64)
        with self.assertRaises(tp.PolicyRejected): verify(ar=dict(anchor, anchor_id='moved'))
        for update in ({'domain':'fold'}, {'orientation_valid':False}, {'overlap_free':False}, {'distortion':.2}):
            bad = dict(chart, **update)
            with self.assertRaises(tp.PolicyRejected):
                verify(dataclasses.replace(admission, chart_sha256=wc.canonical_sha256(bad)), ch=bad)

    def test_checkpoint_requires_expected_arm_full_prefix_and_state(self):
        state = b'same-arm-state'
        identity = tp.ReplayIdentity(*(['a'*64]*5), hashlib.sha256(state).hexdigest(), 3)
        body = dict(schema='tracking_checkpoint_v1', identity=dataclasses.asdict(identity),
                    prefix_frame_ids=['f0', 'f1', 'f2'], state_hex=state.hex())
        verify = lambda b=body, i=identity, prefix=('f0','f1','f2'), raw=state: tp.verify_checkpoint(
            wc.canonical_json(b), expected=i, expected_prefix=prefix, expected_state=raw)
        verify()
        for name in ('source_sha256', 'mask_sha256', 'head_sha256', 'config_sha256', 'policy_sha256', 'state_sha256', 'epoch'):
            altered = dict(body, identity=dict(body['identity'], **{name: 4 if name == 'epoch' else 'b'*64}))
            with self.assertRaises(tp.PolicyRejected): verify(b=altered)
        with self.assertRaises(tp.PolicyRejected): verify(prefix=('f1','f2'))
        with self.assertRaises(tp.PolicyRejected): verify(raw=b'forged')
        forged = dict(body, state_hex=b'forged'.hex())
        with self.assertRaises(tp.PolicyRejected): verify(b=forged)


    def test_self_claimed_proof_and_consumed_seed_are_not_authorization(self):
        p = self.lost_policy()
        for n in (2, 3):
            self.assertEqual(p.consume(row(n), ordinal=n, epoch=0, current_mask_trusted=True,
                                       recovery=self.proof(n)).state, 'recovering')
        with self.assertRaises(tp.PolicyRejected):
            p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
                expected_observations=(self.proof(4),), consumed_seed_sha256='d'*64)

    def test_recovery_rotation_gate_and_epoch_clears_authorization(self):
        p = self.lost_policy()
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(self.proof(2), self.proof(3)), consumed_seed_sha256='c'*64)
        p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=self.proof(2))
        angle = np.deg2rad(11)
        moved = np.eye(4); moved[2, 3] = 1
        moved[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        self.assertEqual(p.consume(row(3, pose=moved), ordinal=3, epoch=0,
            current_mask_trusted=True, recovery=self.proof(3)).state, 'recovering')
        p.reset(1)
        self.assertEqual(p._authorized_observations, {})

    def test_full_prefix_rejects_real_scheduled_gap(self):
        from bench.test_workbench_frame_boundary import neural_fixture
        from bench import workbench_frame_boundary as fb
        context, adapter, backend = neural_fixture(bundle=True)
        try:
            # Legacy context permits f1 warmup, f2 scored followed by a gap to f4.
            # Use the immutable manifest and a later scored subset to test helper
            # directly without bypassing its full context contract.
            fake = dataclasses.replace(context, permitted_order=('f1','f3'))
            with self.assertRaises(tp.PolicyRejected):
                tp.verify_full_prefix(fake)
            tp.verify_full_prefix(context)
        finally:
            adapter.close()


    def test_policy_configuration_readonly_and_original_seed_binding_persists(self):
        p = self.lost_policy()
        with self.assertRaises(AttributeError):
            p.allow_prediction = True
        p.bind_initial_seed('c'*64)
        p.reset(1)
        with self.assertRaises(tp.PolicyRejected):
            p.bind_initial_seed('d'*64)
        with self.assertRaises(tp.PolicyRejected):
            p.authorize_reonboarding(authorization_id='user-approved', epoch=1,
                expected_observations=(dataclasses.replace(self.proof(4), epoch=1),),
                consumed_seed_sha256='d'*64)


    def test_completed_event_retires_all_registered_unused_proofs(self):
        p = self.lost_policy()
        proofs = tuple(self.proof(n) for n in (2, 3, 5, 6))
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=proofs, consumed_seed_sha256='c'*64)
        self.assertEqual(p.consume(row(2), ordinal=2, epoch=0,
            current_mask_trusted=True, recovery=proofs[0]).state, 'recovering')
        self.assertEqual(p.consume(row(3), ordinal=3, epoch=0,
            current_mask_trusted=True, recovery=proofs[1]).state, 'measured')
        self.assertEqual(p._authorized_observations, {})
        self.assertIsNone(p._active_authorization)
        p.consume(row(4, state='lost'), ordinal=4, epoch=0, current_mask_trusted=True)
        for n, proof in zip((5, 6), proofs[2:]):
            out = p.consume(row(n), ordinal=n, epoch=0,
                current_mask_trusted=True, recovery=proof)
            self.assertFalse(out.accepted_measured)
            self.assertIsNone(out.pose)

    def test_successful_pair_consumes_first_and_second_seed_identities(self):
        p = self.lost_policy()
        first, second = self.proof(2), self.proof(3, seed_sha256='e'*64)
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(first, second), consumed_seed_sha256='c'*64)
        p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=first)
        self.assertEqual(p.consume(row(3), ordinal=3, epoch=0,
            current_mask_trusted=True, recovery=second).state, 'measured')
        self.assertTrue({'c'*64, 'd'*64, 'e'*64}.issubset(p._consumed_seed_hashes))
        p.consume(row(4, state='lost'), ordinal=4, epoch=0, current_mask_trusted=True)
        for seed in ('d'*64, 'e'*64):
            with self.assertRaises(tp.PolicyRejected):
                p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
                    expected_observations=(self.proof(5, seed_sha256=seed),),
                    consumed_seed_sha256='c'*64)

    def test_failed_consistency_consumes_every_used_candidate_before_reauthorization(self):
        p = self.lost_policy()
        first, second = self.proof(2), self.proof(3, seed_sha256='e'*64)
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(first, second), consumed_seed_sha256='c'*64)
        p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=first)
        moved = np.eye(4); moved[0, 3] = .11; moved[2, 3] = 1
        self.assertEqual(p.consume(row(3, pose=moved), ordinal=3, epoch=0,
            current_mask_trusted=True, recovery=second).state, 'recovering')
        for seed in ('d'*64, 'e'*64):
            with self.assertRaises(tp.PolicyRejected):
                p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
                    expected_observations=(self.proof(4, seed_sha256=seed),),
                    consumed_seed_sha256='c'*64)

    def test_pending_candidate_seed_remains_consumed_across_epoch_reset(self):
        p = self.lost_policy()
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(self.proof(2), self.proof(3)), consumed_seed_sha256='c'*64)
        p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=self.proof(2))
        p.reset(1)
        self.assertIsNone(p._active_authorization)
        with self.assertRaises(tp.PolicyRejected):
            p.authorize_reonboarding(authorization_id='user-approved', epoch=1,
                expected_observations=(self.proof(4, epoch=1),), consumed_seed_sha256='c'*64)


    def test_source_order_rejections_are_atomic_and_valid_next_frame_recovers(self):
        p = tp.TrackingPolicy()
        p.consume(row(0, timestamp=False), ordinal=0, epoch=0, current_mask_trusted=True)
        p.consume(row(1, state='lost', timestamp=False), ordinal=1, epoch=0, current_mask_trusted=True)
        proofs = (self.proof(2), self.proof(3))
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=proofs, consumed_seed_sha256='c'*64)
        first = row(2, timestamp=False)
        self.assertEqual(p.consume(first, ordinal=2, epoch=0,
            current_mask_trusted=True, recovery=proofs[0]).state, 'recovering')
        duplicate_proof = dataclasses.replace(proofs[0], observation_id='different-id')
        bad_rows = (
            first,  # Same frame/null time, different observation ID/ordinal.
            dataclasses.replace(row(3, timestamp=False), source_frame_index=1),
            dataclasses.replace(first, source_frame_index=3),
            dataclasses.replace(row(3, timestamp=False), source_frame_index=2),
            dataclasses.replace(first, image_sha256='c'*64, source_frame_index=3),
        )
        for bad in bad_rows:
            before = copy.deepcopy(p.__dict__)
            with self.assertRaises(tp.PolicyRejected):
                p.consume(bad, ordinal=3, epoch=0, current_mask_trusted=True, recovery=duplicate_proof)
            self.assertEqual(p.__dict__, before)
        # Rejections did not consume ordinal 3, the active proof or candidate.
        accepted = p.consume(row(3, timestamp=False), ordinal=3, epoch=0,
            current_mask_trusted=True, recovery=proofs[1])
        self.assertTrue(accepted.accepted_measured)
        self.assertEqual(accepted.state, 'measured')

    def test_duplicate_frame_authorization_refuses_before_any_state_mutation(self):
        p = tp.TrackingPolicy()
        proof = self.proof(2)
        duplicate = dataclasses.replace(proof, observation_id='different-id')
        before = copy.deepcopy(p.__dict__)
        with self.assertRaises(tp.PolicyRejected):
            p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
                expected_observations=(proof, duplicate), consumed_seed_sha256='c'*64)
        self.assertEqual(p.__dict__, before)
        # A valid registration can still bind the correct original seed.
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(proof, self.proof(3)), consumed_seed_sha256='e'*64)
        self.assertEqual(p._consumed_seed_sha256, 'e'*64)

    def test_epoch_reset_clears_source_order_without_forgetting_consumed_seeds(self):
        p = self.lost_policy()
        p.authorize_reonboarding(authorization_id='user-approved', epoch=0,
            expected_observations=(self.proof(2), self.proof(3)), consumed_seed_sha256='c'*64)
        p.consume(row(2), ordinal=2, epoch=0, current_mask_trusted=True, recovery=self.proof(2))
        p.reset(1)
        self.assertEqual(p._frame_ids, set())
        self.assertIsNone(p._source_index)
        self.assertTrue({'c'*64, 'd'*64}.issubset(p._consumed_seed_hashes))
        fresh = p.consume(row(0), ordinal=0, epoch=1, current_mask_trusted=True)
        self.assertTrue(fresh.accepted_measured)
        self.assertEqual(p._frame_ids, {'f0'})

    def test_decoded_checkpoint_identity_requires_exact_keys_and_strict_types(self):
        state = b'strict-identity-state'
        expected = tp.ReplayIdentity(*(['a'*64]*5), hashlib.sha256(state).hexdigest(), 1)
        body = dict(schema='tracking_checkpoint_v1', identity=dataclasses.asdict(expected),
                    prefix_frame_ids=['f0','f1'], state_hex=state.hex())
        def verify(identity):
            tp.verify_checkpoint(wc.canonical_json(dict(body, identity=identity)),
                expected=expected, expected_prefix=('f0','f1'), expected_state=state)
        verify(dict(body['identity']))  # Genuine integer epoch remains valid.
        variants = [dict(body['identity'], epoch=True), dict(body['identity'], epoch=1.0),
                    dict(body['identity'], unexpected='x'),
                    {k:v for k,v in body['identity'].items() if k != 'epoch'},
                    dict(body['identity'], source_sha256=1),
                    dict(body['identity'], mask_sha256='A'*64),
                    dict(body['identity'], head_sha256='short')]
        for identity in variants:
            with self.subTest(identity=identity), self.assertRaises(tp.PolicyRejected):
                verify(identity)


    def test_nested_recovery_epoch_authorization_rejects_types_atomically(self):
        for bad_epoch in (True, False, 1.0, -1.0, -1):
            for from_json in (False, True):
                with self.subTest(epoch=bad_epoch, from_json=from_json):
                    p = tp.TrackingPolicy(); p.reset(1)
                    first = self.proof(2, epoch=1)
                    bad = self.proof(3, epoch=bad_epoch)
                    if from_json:
                        bad = tp.RecoveryObservation(**wc.loads_strict(wc.canonical_json(dataclasses.asdict(bad))))
                    before = copy.deepcopy(p.__dict__)
                    with self.assertRaises(tp.PolicyRejected):
                        p.authorize_reonboarding(authorization_id='user-approved', epoch=1,
                            expected_observations=(first, bad), consumed_seed_sha256='c'*64)
                    self.assertEqual(p.__dict__, before)
                    self.assertIsNone(p._consumed_seed_sha256)
                    # The failed event did not bind a seed or poison registration.
                    p.authorize_reonboarding(authorization_id='user-approved', epoch=1,
                        expected_observations=(first, self.proof(3, epoch=1)), consumed_seed_sha256='e'*64)
                    self.assertEqual(p._consumed_seed_sha256, 'e'*64)

    def test_nested_recovery_epoch_consume_rejects_atomically_and_fresh_pair_still_recovers(self):
        p = tp.TrackingPolicy(); p.reset(1)
        p.consume(row(0, timestamp=False), ordinal=0, epoch=1, current_mask_trusted=True)
        p.consume(row(1, state='lost', timestamp=False), ordinal=1, epoch=1, current_mask_trusted=True)
        proofs = (self.proof(2, epoch=1), self.proof(3, epoch=1))
        p.authorize_reonboarding(authorization_id='user-approved', epoch=1,
            expected_observations=proofs, consumed_seed_sha256='c'*64)
        for n, proof in zip((2, 3), proofs):
            for bad_epoch in (True, False, 1.0, -1.0, -1):
                with self.subTest(frame=n, epoch=bad_epoch):
                    bad = dataclasses.replace(proof, epoch=bad_epoch)
                    # Dataclass equality alone would accept True/1.0 as int 1.
                    if bad_epoch is True or type(bad_epoch) is float and bad_epoch == 1.0:
                        self.assertEqual(bad, proof)
                    before = copy.deepcopy(p.__dict__)
                    self.assertFalse(p._recovery_valid(bad, row(n, timestamp=False), 1, True))
                    with self.assertRaises(tp.PolicyRejected):
                        p.consume(row(n, timestamp=False), ordinal=n, epoch=1,
                            current_mask_trusted=True, recovery=bad)
                    self.assertEqual(p.__dict__, before)
            result = p.consume(row(n, timestamp=False), ordinal=n, epoch=1,
                current_mask_trusted=True, recovery=proof)
            self.assertEqual(result.state, 'recovering' if n == 2 else 'measured')
            self.assertEqual(result.accepted_measured, n == 3)
