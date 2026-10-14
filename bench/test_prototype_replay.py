"""Focused source contracts for the bounded offline service replay client."""
import base64
import unittest

from . import prototype_replay as replay


class PrototypeReplayTests(unittest.TestCase):
    def test_packet_contains_only_rgb_identity_and_frame_index_clock(self):
        source = {
            'frameId': 371, 'logicalIndex': 10,
            'evaluationPose': {'cam_R_m2c': [1]*9},
            'evaluationInfo': {'visib_fract': 0.9},
            'evaluationMask': {'path': 'secret.png'}, 'depth': {'path': 'depth.png'},
        }
        packet = replay._make_packet('offline-session', 2, 'a'*64, 'dataset-2',
                                     source, 640, 480, b'jpeg-bytes')
        self.assertEqual(set(packet), replay.FRAME_FIELDS)
        self.assertEqual(packet['captureTimestampSeconds'], None)
        self.assertEqual(packet['timestampKind'], 'frame_index_no_fps')
        self.assertEqual(packet['calibrationId'], 'dataset-2')
        self.assertEqual(base64.b64decode(packet['rgbJPEGBase64']), b'jpeg-bytes')
        self.assertFalse({'evaluationPose', 'evaluationInfo', 'evaluationMask', 'depth'} & set(packet))

    def test_service_url_is_loopback_only_and_https_requires_ca(self):
        target = replay._service_target('http://127.0.0.1:8765')
        self.assertEqual((target['host'], target['port'], target['path']),
                         ('127.0.0.1', 8765, '/v1/frames'))
        with self.assertRaises(ValueError):
            replay._service_target('https://localhost:8765')
        with self.assertRaises(ValueError):
            replay._service_target('http://example.com:8765')
        with self.assertRaises(ValueError):
            replay._service_target('http://127.0.0.1:8765/path')
        with self.assertRaises(ValueError):
            replay._service_target('http://127.0.0.1:8765', ca_file='unused.pem')

    def test_public_pose_requires_current_identity_and_valid_se3(self):
        expected = dict(sessionId='session', objectId=2, referenceId='a'*64,
            calibrationId='dataset-2', frameId=371, logicalIndex=10,
            timestampKind='frame_index_no_fps', captureTimestampSeconds=None)
        payload = {**expected, 'poseState': 'tracking', 'failureReason': None,
            'cameraFromObject': [[1, 0, 0, .1], [0, 1, 0, .2],
                                 [0, 0, 1, .3], [0, 0, 0, 1]],
            'units': 'metres',
            'cameraConvention': replay.SE3_CONVENTION}
        valid = replay._interpret_response(payload, 200, expected)
        self.assertTrue(valid['freshValidPublicPose'])
        payload['frameId'] += 1
        mismatched = replay._interpret_response(payload, 200, expected)
        self.assertFalse(mismatched['freshValidPublicPose'])
        self.assertEqual(mismatched['failureReason'], 'response_identity_mismatch')

    def test_nontracking_pose_is_never_counted_as_valid(self):
        expected = dict(sessionId='session', objectId=2, referenceId='a'*64,
            calibrationId='dataset-2', frameId=371, logicalIndex=10,
            timestampKind='frame_index_no_fps', captureTimestampSeconds=None)
        payload = {**expected, 'poseState': 'lost', 'failureReason': 'no_match',
                   'cameraFromObject': [[1, 0, 0, 0], [0, 1, 0, 0],
                                        [0, 0, 1, 0], [0, 0, 0, 1]]}
        result = replay._interpret_response(payload, 200, expected)
        self.assertFalse(result['freshValidPublicPose'])
        self.assertEqual(result['failureReason'], 'nontracking_response_contains_pose')

    def test_latency_summary_reports_percentiles_without_inventing_values(self):
        summary = replay._latency_summary([1., 2., 3., None])
        self.assertEqual(summary['count'], 3)
        self.assertEqual(summary['p50Ms'], 2.)
        self.assertEqual(summary['p95Ms'], 2.9)
        self.assertEqual(summary['maxMs'], 3.)
        self.assertIsNone(replay._latency_summary([])['p95Ms'])

    def test_mixed_outcomes_use_all_scheduled_frames_for_rates_and_availability(self):
        frames = [
            {'requestAttempted': True, 'responseComplete': True, 'httpStatus': 200,
             'freshValidPublicPose': True, 'poseState': 'tracking',
             'failureReason': None, 'networkLatencyMs': 10., 'endToEndLatencyMs': 15.,
             'jpegEncodeLatencyMs': 3., 'rgbPreparationLatencyMs': 2.},
            {'requestAttempted': True, 'responseComplete': True, 'httpStatus': 200,
             'freshValidPublicPose': False, 'poseState': 'lost',
             'failureReason': 'no_match', 'networkLatencyMs': 12., 'endToEndLatencyMs': 18.,
             'jpegEncodeLatencyMs': 3., 'rgbPreparationLatencyMs': 2.},
            {'requestAttempted': True, 'responseComplete': True, 'httpStatus': 429,
             'freshValidPublicPose': False, 'poseState': None,
             'failureReason': 'http_429', 'networkLatencyMs': 4., 'endToEndLatencyMs': 8.,
             'jpegEncodeLatencyMs': 3., 'rgbPreparationLatencyMs': 2.},
            {'requestAttempted': True, 'responseComplete': False, 'httpStatus': None,
             'freshValidPublicPose': False, 'poseState': None,
             'failureReason': 'timeout', 'transportFailureType': 'TimeoutError',
             'networkLatencyMs': 20., 'endToEndLatencyMs': 25.,
             'jpegEncodeLatencyMs': 3., 'rgbPreparationLatencyMs': 2.},
            {'requestAttempted': False, 'responseComplete': False, 'httpStatus': None,
             'freshValidPublicPose': False, 'poseState': None,
             'failureReason': 'local_input_or_encoding:ReplayError:missing',
             'networkLatencyMs': None, 'endToEndLatencyMs': 1.,
             'jpegEncodeLatencyMs': None, 'rgbPreparationLatencyMs': None},
        ]
        summary = replay._summarize_object(frames, 2.)
        self.assertEqual(summary['scheduledFrames'], 5)
        self.assertEqual(summary['requestAttempts'], 4)
        self.assertEqual(summary['completedResponses'], 3)
        self.assertEqual(summary['freshValidPublicPoses'], 1)
        self.assertEqual(summary['completedResponsesPerSecond'], 1.5)
        self.assertEqual(summary['freshValidPublicPosesPerSecond'], 0.5)
        self.assertEqual(summary['completedResponseAvailability'],
                         {'numerator': 3, 'denominator': 5, 'fraction': 0.6})
        self.assertEqual(summary['allFrameAvailability'],
                         {'numerator': 1, 'denominator': 5, 'fraction': 0.2})
        self.assertEqual(summary['droppedFrames'], 2)
        self.assertEqual(summary['http429Count'], 1)
        self.assertEqual(summary['timeoutCount'], 1)

    def test_progress_journal_uses_exclusive_companion_name(self):
        self.assertEqual(replay._progress_journal_path('run-v2.json').name,
                         'run-v2.frames.jsonl')


if __name__ == '__main__':
    unittest.main()
