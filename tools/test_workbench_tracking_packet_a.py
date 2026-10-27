"""Independent integration controls for Packet A; no model or real-data claims."""
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
