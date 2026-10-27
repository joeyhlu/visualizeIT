"""Independent integration controls for Packet A; no model or real-data claims."""
import copy
import dataclasses
import hashlib
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from bench import workbench_contract as wc
from bench import workbench_frame_boundary as fb
from bench import workbench_offline_export as export
from bench import workbench_tracking_policy as policy
from bench.test_workbench_frame_boundary import neural_fixture, packet
from bench.test_workbench_tracking_policy import row
from tools.test_workbench_offline_export_integration import make_fixture, export_args


class IndependentPacketChecks(unittest.TestCase):
    def boundary(self):
        ctx, adapter, backend = neural_fixture(bundle=True)
        chart = dict(chart_id='independent-plane', domain='plane', orientation_valid=True,
                     overlap_free=True, distortion=1., max_distortion=1.)
        anchor = dict(anchor_id='independent-anchor', chart_id=chart['chart_id'],
                      geometry_sha256=backend.bundle.glb_sha256)
        admitted = policy.SurfaceAdmission('metric', backend.bundle.metric_provenance,
            wc.canonical_sha256(list(ctx.K)), backend.bundle.glb_sha256,
            wc.canonical_sha256(anchor), wc.canonical_sha256(chart))
        display = policy.TrackingPolicy(allow_prediction=True)
        boundary = fb.FrameBoundary(tracking_policy=display, surface_admission=admitted,
                                    anchor_record=anchor, chart_record=chart)
        boundary.initialize(ctx, adapter)
        self.addCleanup(boundary.close)
        return ctx, boundary, backend, display

    def test_policy_configuration_cannot_change_after_initialization(self):
        ctx, boundary, backend, display = self.boundary()
        before = len(backend.calls)
        try:
            display.allow_prediction = False
        except (AttributeError, dataclasses.FrozenInstanceError):
            return  # A read-only configuration also satisfies the invariant.
        with self.assertRaises((fb.BoundaryError, policy.PolicyRejected)):
            boundary.process(packet(ctx, 'f1'))
        self.assertEqual(len(backend.calls), before)

    def test_integrity_failure_closes_instead_of_becoming_a_prediction(self):
        ctx, boundary, backend, display = self.boundary()
        accepted = boundary.process(packet(ctx, 'f1'))
        self.assertTrue(accepted.display_decision.accepted_measured)
        calls = len(backend.calls)
        with self.assertRaises(fb.BoundaryFailure):
            boundary.process(packet(ctx, 'f2', mask_sha256='f' * 64))
        self.assertEqual(boundary.state, 'failed')
        self.assertEqual(len(backend.calls), calls)

    def test_reset_cannot_display_a_previous_epoch_pose_without_mask(self):
        ctx, boundary, backend, display = self.boundary()
        boundary.process(packet(ctx, 'f1'))
        boundary.reset()
        out = boundary.process(packet(ctx, 'f1', mask=None))
        self.assertEqual(out.epoch, 1)
        self.assertIsNone(out.overlay_rgba)
        self.assertFalse(out.display_decision.accepted_measured)
        self.assertIsNone(out.display_decision.pose)

    def test_genuine_time_limit_is_independent_of_frame_count(self):
        display = policy.TrackingPolicy(allow_prediction=True)
        display.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
        # XHigh proposal: the first dropout still expires after 100 ms.
        stale = dataclasses.replace(row(1, state='unavailable'), timestamp_s=.101)
        out = display.consume(stale, ordinal=1, epoch=0, current_mask_trusted=True)
        self.assertEqual((out.state, out.pose, out.opacity), ('lost', None, 0.))

    def test_checkpoint_cannot_authenticate_itself_after_head_change(self):
        state = b'controlled-state-v1'
        expected = policy.ReplayIdentity('1'*64, '2'*64, '3'*64, '4'*64,
                                        '5'*64, hashlib.sha256(state).hexdigest(), 2)
        forged = dataclasses.replace(expected, head_sha256='6'*64)
        payload = wc.canonical_json(dict(schema='tracking_checkpoint_v1',
            identity=dataclasses.asdict(forged), prefix_frame_ids=['f1', 'f2'],
            state_hex=state.hex()))
        with self.assertRaises(policy.PolicyRejected):
            policy.verify_checkpoint(payload, expected=expected,
                                     expected_prefix=('f1', 'f2'), expected_state=state)

    def proof(self, n, seed='d'):
        return policy.RecoveryObservation('independent-event', f'obs{n}', f'f{n}', 0,
            'a'*64, 'b'*64, 'onboarding:independent-current', True,
            True, True, True, True, seed*64)

    def recovery_policy(self):
        display = policy.TrackingPolicy()
        display.consume(row(), ordinal=0, epoch=0, current_mask_trusted=True)
        display.consume(row(1, state='lost'), ordinal=1, epoch=0, current_mask_trusted=True)
        return display

    def test_one_reonboarding_event_cannot_authorize_a_second_recovery(self):
        display = self.recovery_policy()
        observations = tuple(self.proof(n) for n in (2, 3, 5, 6))
        display.authorize_reonboarding(authorization_id='independent-event', epoch=0,
            expected_observations=observations, consumed_seed_sha256='c'*64)
        for n in (2, 3):
            out = display.consume(row(n), ordinal=n, epoch=0,
                                  current_mask_trusted=True, recovery=self.proof(n))
        self.assertTrue(out.accepted_measured)
        display.consume(row(4, state='lost'), ordinal=4, epoch=0, current_mask_trusted=True)
        for n in (5, 6):
            out = display.consume(row(n), ordinal=n, epoch=0,
                                  current_mask_trusted=True, recovery=self.proof(n))
            self.assertFalse(out.accepted_measured)
            self.assertIsNone(out.pose)

    def test_all_used_recovery_seed_identities_remain_consumed(self):
        display = self.recovery_policy()
        observations = (self.proof(2, 'd'), self.proof(3, 'e'))
        display.authorize_reonboarding(authorization_id='independent-event', epoch=0,
            expected_observations=observations, consumed_seed_sha256='c'*64)
        for n, proof in zip((2, 3), observations):
            out = display.consume(row(n), ordinal=n, epoch=0,
                                  current_mask_trusted=True, recovery=proof)
        self.assertTrue(out.accepted_measured)
        display.consume(row(4, state='lost'), ordinal=4, epoch=0, current_mask_trusted=True)
        with self.assertRaises(policy.PolicyRejected):
            display.authorize_reonboarding(authorization_id='independent-event', epoch=0,
                expected_observations=(self.proof(5, 'd'), self.proof(6, 'd')),
                consumed_seed_sha256='c'*64)

    def test_duplicate_authorized_source_frame_refuses_without_mutation(self):
        display = self.recovery_policy()
        before = copy.deepcopy(vars(display))
        duplicate = dataclasses.replace(self.proof(2), observation_id='different-id')
        with self.assertRaises(policy.PolicyRejected):
            display.authorize_reonboarding(authorization_id='independent-event', epoch=0,
                expected_observations=(self.proof(2), duplicate), consumed_seed_sha256='c'*64)
        self.assertEqual(vars(display), before)

    def test_null_clock_duplicate_backward_and_mutated_sources_are_atomic(self):
        # Ordinal advancement must never turn one RGB observation into two
        # recovery measurements. All variants are valid FrameResult records.
        for variant in ('same', 'backward', 'same-id-new-index', 'new-id-same-index'):
            with self.subTest(variant=variant):
                display = policy.TrackingPolicy()
                display.consume(row(timestamp=False), ordinal=0, epoch=0, current_mask_trusted=True)
                display.consume(row(1, state='lost', timestamp=False), ordinal=1,
                                epoch=0, current_mask_trusted=True)
                display.authorize_reonboarding(authorization_id='independent-event', epoch=0,
                    expected_observations=(self.proof(2), self.proof(3)), consumed_seed_sha256='c'*64)
                display.consume(row(2, timestamp=False), ordinal=2, epoch=0,
                                current_mask_trusted=True, recovery=self.proof(2))
                source = row(2, timestamp=False)
                if variant == 'backward':
                    source = row(1, timestamp=False)
                elif variant == 'same-id-new-index':
                    source = dataclasses.replace(source, source_frame_index=3, image_sha256='f'*64)
                elif variant == 'new-id-same-index':
                    source = dataclasses.replace(source, frame_id='different-frame', mask_frame_id='different-frame')
                before = copy.deepcopy(vars(display))
                with self.assertRaises(policy.PolicyRejected):
                    display.consume(source, ordinal=3, epoch=0, current_mask_trusted=True,
                                    recovery=self.proof(3))
                self.assertEqual(vars(display), before)
                # Rejecting the duplicate must not consume the ordinal, seed or
                # authorization needed by the genuine next source frame.
                fresh = display.consume(row(3, timestamp=False), ordinal=3, epoch=0,
                                        current_mask_trusted=True, recovery=self.proof(3))
                self.assertTrue(fresh.accepted_measured)

    def test_decoded_checkpoint_identity_requires_exact_integer_epoch_and_keys(self):
        state = b'independent-valid-checkpoint'
        expected = policy.ReplayIdentity(*(['a'*64]*5), hashlib.sha256(state).hexdigest(), 1)
        body = dict(schema='tracking_checkpoint_v1', identity=dataclasses.asdict(expected),
                    prefix_frame_ids=['f0', 'f1'], state_hex=state.hex())
        def verify(value):
            policy.verify_checkpoint(wc.canonical_json(value), expected=expected,
                                     expected_prefix=('f0', 'f1'), expected_state=state)
        verify(body)
        for epoch in (True, 1.0, '1', None):
            with self.subTest(epoch=repr(epoch)), self.assertRaises(policy.PolicyRejected):
                verify(dict(body, identity=dict(body['identity'], epoch=epoch)))
        for identity in (dict(body['identity'], unexpected='x'),
                         {k: v for k, v in body['identity'].items() if k != 'epoch'}):
            with self.assertRaises(policy.PolicyRejected):
                verify(dict(body, identity=identity))

    def epoch_one_recovery(self):
        display = policy.TrackingPolicy()
        display.reset(1)
        display.consume(row(timestamp=False), ordinal=0, epoch=1, current_mask_trusted=True)
        display.consume(row(1, state='lost', timestamp=False), ordinal=1,
                        epoch=1, current_mask_trusted=True)
        proofs = tuple(dataclasses.replace(self.proof(n), epoch=1) for n in (2, 3))
        return display, proofs

    def test_nested_epoch_authorization_strict_types_and_atomicity(self):
        for bad_epoch in (True, 1.0, -1, '1', None):
            with self.subTest(epoch=repr(bad_epoch)):
                display, proofs = self.epoch_one_recovery()
                invalid = tuple(dataclasses.replace(proof, epoch=bad_epoch) for proof in proofs)
                before = copy.deepcopy(vars(display))
                with self.assertRaises(policy.PolicyRejected):
                    display.authorize_reonboarding(authorization_id='independent-event', epoch=1,
                        expected_observations=invalid, consumed_seed_sha256='c'*64)
                self.assertEqual(vars(display), before)

    def test_nested_epoch_consume_rejects_equal_boolean_float_before_mutation(self):
        display, proofs = self.epoch_one_recovery()
        display.authorize_reonboarding(authorization_id='independent-event', epoch=1,
            expected_observations=proofs, consumed_seed_sha256='c'*64)
        for bad_epoch in (True, 1.0, -1, '1', None):
            with self.subTest(epoch=repr(bad_epoch)):
                before = copy.deepcopy(vars(display))
                with self.assertRaises(policy.PolicyRejected):
                    display.consume(row(2, timestamp=False), ordinal=2, epoch=1,
                        current_mask_trusted=True,
                        recovery=dataclasses.replace(proofs[0], epoch=bad_epoch))
                self.assertEqual(vars(display), before)
        # Python equality would equate true/1.0 with 1 above. Only the genuine
        # integer-typed observations may complete this two-frame recovery.
        first = display.consume(row(2, timestamp=False), ordinal=2, epoch=1,
                                current_mask_trusted=True, recovery=proofs[0])
        second = display.consume(row(3, timestamp=False), ordinal=3, epoch=1,
                                 current_mask_trusted=True, recovery=proofs[1])
        self.assertFalse(first.accepted_measured)
        self.assertTrue(second.accepted_measured)

    def test_actual_export_keeps_failed_rgb_and_separate_flat_pose(self):
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix='independent-packet-a-') as folder:
            fixture = make_fixture(Path(folder))
            manifest = export.export_saved_window(fixture['workspace'], **export_args(fixture))
            capture = cv2.VideoCapture(str(fixture['output_dir'] / 'playback.avi'))
            try:
                for record in manifest['rows']:
                    ok, bgr = capture.read()
                    self.assertTrue(ok)
                    decision = record['display_decision']
                    original = record['original']
                    self.assertNotIn('display_decision', original)
                    if original['pose'] is not None:
                        # Shared display contract is flat row-major SE(3); the
                        # historical evidence keeps its original nested matrix.
                        self.assertEqual(np.asarray(original['pose']).shape, (4, 4))
                        self.assertEqual(np.asarray(decision['pose']).shape, (16,))
                    if original['pose_state'] in ('recovering', 'lost', 'unavailable'):
                        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                        # The persisted image digest binds dtype and shape as
                        # well as pixels; this independently assembles that wire format.
                        raw = f'{rgb.dtype.str}|{rgb.shape}|'.encode('ascii') + rgb.tobytes()
                        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                         record['source_rgb_sha256'])
                        self.assertEqual(decision['opacity'], 0.)
                        self.assertIsNone(decision['pose'])
            finally:
                capture.release()


if __name__ == '__main__':
    unittest.main()
