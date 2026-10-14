"""Local-only HTTP and integrity tests for the RGB prototype service."""
import base64
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .prototype_service import (
    MAX_HTTP_WORKERS, MAX_REQUEST_BYTES, PrototypeService, create_server, load_asset_registry,
    validate_bind_security,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT/'.cache'/'prototype-sprint'/'service-test-tmp'


def _is_within(path, root):
    return path == root or root in path.parents


class _FakeOutput:
    def __init__(self, frame):
        self.frame = frame

    def manifest(self):
        pose = np.eye(4, dtype=float)
        pose[2, 3] = 1.2
        return {
            'sessionId': self.frame.session_id,
            'objectId': self.frame.object_id,
            'referenceId': self.frame.reference_id,
            'frameId': self.frame.frame_id,
            'logicalIndex': self.frame.logical_index,
            'timestampSeconds': self.frame.timestamp_seconds,
            'timestampKind': self.frame.timestamp_kind,
            'state': 'tracking',
            'failureReason': None,
            'cameraFromObject': pose.tolist(),
        }


class _FakeTracker:
    def __init__(self):
        self.frames = []
        self.resets = 0

    def update(self, frame):
        self.frames.append(frame)
        return _FakeOutput(frame)

    def reset_temporal(self):
        self.resets += 1


class _BlockingTracker(_FakeTracker):
    def __init__(self, entered, release):
        super().__init__()
        self.entered = entered
        self.release = release
        self.active_updates = 0
        self.maximum_active_updates = 0
        self.reset_counts_at_update = []
        self._update_lock = threading.Lock()

    def update(self, frame):
        with self._update_lock:
            self.active_updates += 1
            self.maximum_active_updates = max(self.maximum_active_updates, self.active_updates)
            self.frames.append(frame)
            self.reset_counts_at_update.append(self.resets)
        try:
            if len(self.frames) == 1:
                self.entered.set()
                if not self.release.wait(5.):
                    raise TimeoutError('Blocking tracker was not released by its test.')
            return _FakeOutput(frame)
        finally:
            with self._update_lock:
                self.active_updates -= 1


def _write_asset(root):
    root = Path(root)
    directory = root/'object-000001'
    directory.mkdir(parents=True)
    positions = np.array([
        [-.1, -.1, 0.], [.1, -.1, 0.], [0., .1, 0.], [0., 0., .1],
    ], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
    references = np.arange(36, dtype=np.float32).reshape(12, 3)/100
    descriptors = np.arange(12*32, dtype=np.uint8).reshape(12, 32)
    normals = np.zeros_like(positions)
    uv = np.zeros((len(positions), 2), dtype=np.float32)
    archive = directory/'tracking-asset.npz'
    np.savez_compressed(
        archive, referencePoints=references, referenceDescriptors=descriptors,
        modelPositions=positions, positions=positions, normals=normals,
        triangles=triangles, uv=uv, benchFromSource=np.eye(4),
        schemaVersion=np.asarray(1, dtype=np.int32),
    )
    asset_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (directory/'asset.json').write_text(json.dumps({
        'objectId': 1, 'name': 'Synthetic cup', 'assetPath': archive.name,
        'assetSHA256': asset_hash, 'sourceBasis': 'Original right-handed BOP source model',
        'evidenceKind': 'dataset_reference', 'units': 'metres',
    }), encoding='utf-8')
    return asset_hash


def _write_calibrations(path, rows=None):
    if rows is None:
        rows = [{
            'calibrationId': 'measured-32x24', 'width': 32, 'height': 24,
            'K': [[30., 0., 16.], [0., 30., 12.], [0., 0., 1.]],
            'kind': 'physical_measured', 'objectId': 1,
        }]
    Path(path).write_text(json.dumps({'schemaVersion': 1, 'calibrations': rows}), encoding='utf-8')


def _jpeg_base64(width=32, height=24):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 1] = 100
    pixels[::2, ::2, 0] = 255
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format='JPEG', quality=90)
    return base64.b64encode(encoded.getvalue()).decode('ascii')


def _packet(frame_id=1, *, session='session-a', calibration='measured-32x24', **updates):
    packet = {
        'sessionId': session, 'frameId': frame_id, 'logicalIndex': frame_id,
        'captureTimestampSeconds': frame_id/30., 'timestampKind': 'physical',
        'objectId': 1, 'referenceId': updates.pop('referenceId', None),
        'calibrationId': calibration, 'width': 32, 'height': 24,
        'rgbJPEGBase64': _jpeg_base64(),
    }
    packet['referenceId'] = packet['referenceId'] or updates.pop('_asset_hash')
    packet.update(updates)
    return packet


class PrototypeServiceTests(unittest.TestCase):
    def setUp(self):
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        workspace = PROJECT_ROOT.resolve()
        scratch = TEST_TMP_ROOT.resolve()
        if scratch == workspace or not _is_within(scratch, workspace):
            raise RuntimeError('Service test scratch directory escaped the workspace.')
        self.temp = tempfile.TemporaryDirectory(dir=scratch, ignore_cleanup_errors=True)
        self.root = Path(self.temp.name).resolve()
        if self.root == scratch or not _is_within(self.root, scratch):
            raise RuntimeError('Service test fixture escaped its scratch directory.')
        self.assets = self.root/'assets'
        self.assets.mkdir()
        self.reference_id = _write_asset(self.assets)
        self.calibrations = self.root/'calibrations.json'
        _write_calibrations(self.calibrations)
        self.client = self.root/'client.html'
        self.client.write_text('<!doctype html><title>local test client</title>', encoding='utf-8')
        self.created_trackers = []
        self.servers = []

        def tracker_factory(*args, **kwargs):
            tracker = _FakeTracker()
            self.created_trackers.append(tracker)
            return tracker

        self.tracker_factory = tracker_factory

    def tearDown(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(2.)
        scratch = TEST_TMP_ROOT.resolve()
        root = self.root.resolve()
        if root == scratch or not _is_within(root, scratch):
            raise RuntimeError('Refusing to clean a service fixture outside the workspace scratch root.')
        self.temp.cleanup()

    def service(self, calibrations=True):
        return PrototypeService(
            self.assets, self.calibrations if calibrations else None,
            client_path=self.client, tracker_factory=self.tracker_factory,
        )

    def start_server(self, service):
        server = create_server('127.0.0.1', 0, service)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        return f'http://127.0.0.1:{server.server_port}'

    @staticmethod
    def get_json(url):
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, json.loads(response.read().decode('utf-8'))

    @staticmethod
    def post_json(url, packet):
        request = urllib.request.Request(
            url, data=json.dumps(packet).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST',
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode('utf-8'))

    @staticmethod
    def post_json_status(url, packet):
        request = urllib.request.Request(
            url, data=json.dumps(packet).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as response:
            try:
                return response.code, json.loads(response.read().decode('utf-8'))
            finally:
                response.close()

    @staticmethod
    def post_json_error(url, packet):
        request = urllib.request.Request(
            url, data=json.dumps(packet).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST',
        )
        try:
            urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as response:
            try:
                return response.code, json.loads(response.read().decode('utf-8'))
            finally:
                response.close()
        raise AssertionError('Expected the malformed packet to be rejected.')

    def test_loopback_http_status_and_same_frame_pose_are_explicit(self):
        service = self.service()
        base = self.start_server(service)
        status, health = self.get_json(base+'/healthz')
        self.assertEqual(status, 200)
        self.assertEqual(health['status'], 'ok')
        status, document = self.get_json(base+'/v1/status')
        self.assertEqual(status, 200)
        self.assertEqual(document['assets'], [{
            'objectId': 1, 'referenceId': self.reference_id, 'name': 'Synthetic cup',
            'sourceBasis': 'Original right-handed BOP source model',
            'evidenceKind': 'dataset_reference',
        }])
        self.assertEqual(document['calibrations'][0]['kind'], 'physical_measured')
        with urllib.request.urlopen(base+'/', timeout=3) as response:
            self.assertIn(b'local test client', response.read())

        packet = _packet(_asset_hash=self.reference_id, cameraFrameId=17, cameraMediaTime=.5)
        status, result = self.post_json(base+'/v1/frames', packet)
        self.assertEqual(status, 200)
        self.assertEqual((result['sessionId'], result['frameId'], result['logicalIndex']),
                         ('session-a', 1, 1))
        self.assertEqual(result['referenceId'], self.reference_id)
        self.assertEqual(result['calibrationId'], 'measured-32x24')
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(result['maskState'], 'unavailable')
        self.assertEqual(result['occlusionState'], 'unsupported')
        self.assertEqual(result['renderState'], 'diagnostic_wireframe')
        self.assertEqual(result['units'], 'metres')
        self.assertEqual(result['cameraConvention'], 'CV X right, Y down, Z forward; column vectors')
        self.assertEqual(result['objectBasis'], 'Original right-handed source-object axes')
        self.assertTrue(result['freshCameraFrameVerified'])
        self.assertEqual(result['cameraFrameId'], 17)
        self.assertEqual(result['cameraMediaTime'], .5)
        self.assertEqual(len(result['cameraFromObject']), 4)
        self.assertTrue(result['projectionSegments'])
        self.assertGreaterEqual(result['processingSeconds'], 0.)
        self.assertEqual(len(self.created_trackers), 1)
        self.assertEqual(self.created_trackers[0].frames[0].frame_id, 1)

    def test_missing_calibration_and_bad_resolution_never_return_a_pose(self):
        service = self.service(calibrations=False)
        base = self.start_server(service)
        packet = _packet(_asset_hash=self.reference_id, calibration=None)
        status, result = self.post_json(base+'/v1/frames', packet)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'calibration_required')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(result['maskState'], 'unavailable')
        self.assertEqual(result['renderState'], 'suppressed')
        self.assertEqual(self.created_trackers, [])

        service = self.service()
        base = self.start_server(service)
        packet = _packet(_asset_hash=self.reference_id, width=31)
        status, result = self.post_json(base+'/v1/frames', packet)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'calibration_resolution_mismatch')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers, [])

    def test_ninety_degree_calibration_resolution_mismatch_suppresses_pose(self):
        rows = [
            {'calibrationId': 'measured-32x24', 'width': 32, 'height': 24,
             'K': [[30., 0., 16.], [0., 30., 12.], [0., 0., 1.]],
             'kind': 'physical_measured', 'objectId': 1},
            {'calibrationId': 'rotated-24x32', 'width': 24, 'height': 32,
             'K': [[30., 0., 12.], [0., 30., 16.], [0., 0., 1.]],
             'kind': 'physical_measured', 'objectId': 1},
        ]
        _write_calibrations(self.calibrations, rows)
        base = self.start_server(self.service())
        packet = _packet(_asset_hash=self.reference_id, calibration='rotated-24x32')
        status, result = self.post_json(base+'/v1/frames', packet)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'calibration_resolution_mismatch')
        self.assertEqual(result['calibrationKind'], 'physical_measured')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(result['renderState'], 'suppressed')
        self.assertEqual(self.created_trackers, [])

    def test_bad_frame_resets_tracker_and_replay_cannot_restore_stale_pose(self):
        service = self.service()
        base = self.start_server(service)
        first = _packet(_asset_hash=self.reference_id)
        status, result = self.post_json(base+'/v1/frames', first)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(status, 200)

        second = _packet(2, _asset_hash=self.reference_id, calibration=None)
        status, result = self.post_json(base+'/v1/frames', second)
        self.assertEqual(result['failureReason'], 'calibration_required')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(result['renderState'], 'suppressed')
        self.assertEqual(self.created_trackers[0].resets, 1)

        status, result = self.post_json(base+'/v1/frames', first)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'out_of_order_frame')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[0].resets, 2)

    def test_malformed_identity_resets_known_session_and_preserves_frame_highwatermark(self):
        base = self.start_server(self.service())
        first = _packet(_asset_hash=self.reference_id)
        _, result = self.post_json(base+'/v1/frames', first)
        self.assertEqual(result['poseState'], 'tracking')

        missing_reference = _packet(2, _asset_hash=self.reference_id)
        del missing_reference['referenceId']
        status, result = self.post_json_error(base+'/v1/frames', missing_reference)
        self.assertEqual(status, 400)
        self.assertEqual(result['failureReason'], 'invalid_identity')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[0].resets, 1)

        # Frame 2 remains available because the rejected malformed packet did
        # not advance the session clock; the tracker itself was reset first.
        corrected = _packet(2, _asset_hash=self.reference_id)
        status, result = self.post_json(base+'/v1/frames', corrected)
        self.assertEqual(status, 200)
        self.assertEqual(result['poseState'], 'tracking')

        malformed_object = _packet(3, _asset_hash=self.reference_id, objectId='not-an-integer')
        status, result = self.post_json_error(base+'/v1/frames', malformed_object)
        self.assertEqual(status, 400)
        self.assertEqual(result['failureReason'], 'invalid_identity')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[0].resets, 2)

    def test_busy_frame_gets_bounded_429_and_next_accepted_frame_resets(self):
        entered = threading.Event()
        release = threading.Event()
        trackers = []

        def blocking_factory(*args, **kwargs):
            tracker = _BlockingTracker(entered, release)
            trackers.append(tracker)
            return tracker

        service = PrototypeService(
            self.assets, self.calibrations, client_path=self.client,
            tracker_factory=blocking_factory,
        )
        base = self.start_server(service)
        server = self.servers[-1][0]
        self.assertEqual(server.request_queue_size, 4)

        first_packet = _packet(_asset_hash=self.reference_id)
        first_done = threading.Event()
        first_result = {}

        def send_first():
            try:
                first_result['value'] = self.post_json_status(base+'/v1/frames', first_packet)
            except Exception as exc:
                first_result['error'] = exc
            finally:
                first_done.set()

        first_thread = threading.Thread(target=send_first, daemon=True)
        first_thread.start()
        second_thread = None
        try:
            self.assertTrue(entered.wait(2.), 'first tracker update should enter and block')
            self.assertEqual(len(trackers), 1)
            self.assertEqual(len(trackers[0].frames), 1)

            busy_packet = _packet(2, _asset_hash=self.reference_id)
            second_done = threading.Event()
            second_result = {}

            def send_second():
                try:
                    second_result['value'] = self.post_json_status(base+'/v1/frames', busy_packet)
                except Exception as exc:
                    second_result['error'] = exc
                finally:
                    second_done.set()

            second_thread = threading.Thread(target=send_second, daemon=True)
            second_thread.start()
            self.assertTrue(second_done.wait(2.), 'overloaded frame should receive an immediate response')
            self.assertNotIn('error', second_result)
            busy_status, busy_result = second_result['value']
            self.assertEqual(busy_status, 429)
            self.assertEqual(busy_result['failureReason'], 'service_busy')
            self.assertEqual(busy_result['frameId'], 2)
            self.assertEqual(busy_result['sessionId'], 'session-a')
            self.assertIsNone(busy_result['cameraFromObject'])
            self.assertEqual(busy_result['renderState'], 'suppressed')
            self.assertEqual(len(trackers[0].frames), 1, 'busy frames must not enter tracker processing')

            status, metrics = self.get_json(base+'/v1/status')
            self.assertEqual(status, 200)
            self.assertEqual(metrics['frameRequests'], 2)
            self.assertEqual(metrics['acceptedFrameRequests'], 1)
            self.assertEqual(metrics['busy429Responses'], 1)
            self.assertEqual(metrics['connectionAdmissionDrops'], 0)
            self.assertGreaterEqual(metrics['activeWorkers'], 2)
            self.assertLessEqual(metrics['activeWorkers'], MAX_HTTP_WORKERS)
            self.assertEqual(metrics['maxWorkers'], MAX_HTTP_WORKERS)
            self.assertEqual(trackers[0].maximum_active_updates, 1)
        finally:
            release.set()
            first_thread.join(3.)
            if second_thread is not None:
                second_thread.join(3.)

        self.assertTrue(first_done.is_set())
        self.assertNotIn('error', first_result)
        first_status, first_response = first_result['value']
        self.assertEqual(first_status, 200)
        self.assertEqual(first_response['poseState'], 'tracking')

        after_busy = _packet(3, _asset_hash=self.reference_id)
        status, result = self.post_json_status(base+'/v1/frames', after_busy)
        self.assertEqual(status, 200)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(trackers[0].reset_counts_at_update, [0, 1])

        replay = _packet(2, _asset_hash=self.reference_id)
        status, result = self.post_json_status(base+'/v1/frames', replay)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'out_of_order_frame')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(trackers[0].reset_counts_at_update, [0, 1])

    def test_busy_reset_is_queued_atomically_before_next_admission(self):
        entered = threading.Event()
        release_first = threading.Event()
        busy_queue_entered = threading.Event()
        allow_busy_enqueue = threading.Event()
        trackers = []

        def blocking_factory(*args, **kwargs):
            tracker = _BlockingTracker(entered, release_first)
            trackers.append(tracker)
            return tracker

        service = PrototypeService(
            self.assets, self.calibrations, client_path=self.client,
            tracker_factory=blocking_factory,
        )
        original_queue = service._queue_busy_reset_locked

        def pause_busy_enqueue(packet):
            busy_queue_entered.set()
            if not allow_busy_enqueue.wait(5.):
                raise TimeoutError('Busy enqueue was not released by its test.')
            original_queue(packet)

        # Hold the admission mutex after the failed gate acquisition. Releasing
        # the active inference here recreates the precise gap that could let a
        # later frame enter before the busy reset was recorded.
        service._queue_busy_reset_locked = pause_busy_enqueue
        results = {}
        done = {name: threading.Event() for name in ('first', 'busy', 'next')}

        def send(name, packet):
            try:
                results[name] = service.process_frame(packet)
            except Exception as exc:
                results[name+'_error'] = exc
            finally:
                done[name].set()

        first_thread = threading.Thread(
            target=send, args=('first', _packet(_asset_hash=self.reference_id)), daemon=True)
        busy_thread = threading.Thread(
            target=send, args=('busy', _packet(2, _asset_hash=self.reference_id)), daemon=True)
        next_thread = threading.Thread(
            target=send, args=('next', _packet(3, _asset_hash=self.reference_id)), daemon=True)
        try:
            first_thread.start()
            self.assertTrue(entered.wait(2.), 'first tracker update should enter and block')
            busy_thread.start()
            self.assertTrue(busy_queue_entered.wait(2.), 'busy request should hold admission while queuing reset')
            release_first.set()
            self.assertTrue(done['first'].wait(2.), 'active frame should finish and release inference admission')
            next_thread.start()
            self.assertFalse(done['next'].wait(.05), 'later frame must wait until busy reset has been recorded')
            allow_busy_enqueue.set()
            self.assertTrue(done['busy'].wait(2.), 'busy frame should return after reset queueing')
            self.assertTrue(done['next'].wait(2.), 'next frame should proceed after admission mutex is released')
        finally:
            release_first.set()
            allow_busy_enqueue.set()
            first_thread.join(2.)
            busy_thread.join(2.)
            next_thread.join(2.)

        for name in ('first', 'busy', 'next'):
            self.assertNotIn(name+'_error', results)
        self.assertEqual(results['first'][0], 200)
        self.assertEqual(results['first'][1]['poseState'], 'tracking')
        self.assertEqual(results['busy'][0], 429)
        self.assertEqual(results['busy'][1]['failureReason'], 'service_busy')
        self.assertIsNone(results['busy'][1]['cameraFromObject'])
        self.assertEqual(results['next'][0], 200)
        self.assertEqual(results['next'][1]['poseState'], 'tracking')
        self.assertEqual(trackers[0].reset_counts_at_update, [0, 1])

    def test_http_worker_cap_closes_excess_connections_and_counts_drops(self):
        self.assertEqual(MAX_HTTP_WORKERS, 8, 'browser asset loading requires six workers plus service requests')
        base = self.start_server(self.service())
        server = self.servers[-1][0]
        port = int(base.rsplit(':', 1)[1])
        held = []
        try:
            for _ in range(MAX_HTTP_WORKERS):
                connection = socket.create_connection(('127.0.0.1', port), timeout=2.)
                connection.settimeout(2.)
                connection.sendall(b'GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n')
                held.append(connection)

            deadline = time.monotonic()+2.
            while server.worker_metrics()['activeWorkers'] < MAX_HTTP_WORKERS:
                if time.monotonic() >= deadline:
                    self.fail('HTTP workers did not reach the configured cap.')
                time.sleep(.01)
            self.assertEqual(server.worker_metrics()['activeWorkers'], MAX_HTTP_WORKERS)

            overflow = socket.create_connection(('127.0.0.1', port), timeout=2.)
            overflow.settimeout(2.)
            held.append(overflow)
            deadline = time.monotonic()+2.
            while server.worker_metrics()['connectionAdmissionDrops'] < 1:
                if time.monotonic() >= deadline:
                    self.fail('Excess connection was not counted as an admission drop.')
                time.sleep(.01)
        finally:
            for connection in held:
                connection.close()

        deadline = time.monotonic()+2.
        while server.worker_metrics()['activeWorkers']:
            if time.monotonic() >= deadline:
                self.fail('HTTP workers did not release after clients disconnected.')
            time.sleep(.01)
        status, metrics = self.get_json(base+'/v1/status')
        self.assertEqual(status, 200)
        self.assertEqual(metrics['connectionAdmissionDrops'], 1)
        self.assertEqual(metrics['activeWorkers'], 1)
        self.assertEqual(metrics['maxWorkers'], MAX_HTTP_WORKERS)

    def test_logical_and_frame_order_errors_return_null_pose_without_advancing_clock(self):
        base = self.start_server(self.service())
        first = _packet(_asset_hash=self.reference_id)
        _, result = self.post_json(base+'/v1/frames', first)
        self.assertEqual(result['poseState'], 'tracking')

        bad_logical = _packet(2, _asset_hash=self.reference_id, logicalIndex=1)
        status, result = self.post_json(base+'/v1/frames', bad_logical)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'out_of_order_frame')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[0].resets, 1)

        corrected = _packet(2, _asset_hash=self.reference_id, logicalIndex=2)
        status, result = self.post_json(base+'/v1/frames', corrected)
        self.assertEqual(status, 200)
        self.assertEqual(result['poseState'], 'tracking')

        bad_frame = _packet(1, _asset_hash=self.reference_id, logicalIndex=3)
        status, result = self.post_json(base+'/v1/frames', bad_frame)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'out_of_order_frame')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[0].resets, 2)

    def test_reconnected_sessions_are_independent_and_identity_bound(self):
        base = self.start_server(self.service())
        first_session = _packet(session='session-a', _asset_hash=self.reference_id)
        _, result = self.post_json(base+'/v1/frames', first_session)
        self.assertEqual(result['poseState'], 'tracking')

        reconnected = _packet(session='session-b', _asset_hash=self.reference_id)
        _, result = self.post_json(base+'/v1/frames', reconnected)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(len(self.created_trackers), 2)
        self.assertEqual(self.created_trackers[0].frames[0].session_id, 'session-a')
        self.assertEqual(self.created_trackers[1].frames[0].session_id, 'session-b')

        foreign = _packet(2, session='session-b', _asset_hash=self.reference_id)
        foreign['referenceId'] = 'foreign-reference'
        status, result = self.post_json(base+'/v1/frames', foreign)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'identity_mismatch')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[1].resets, 1)

        own_session_b = _packet(2, session='session-b', _asset_hash=self.reference_id)
        _, result = self.post_json(base+'/v1/frames', own_session_b)
        self.assertEqual(result['poseState'], 'tracking')
        own_session_a = _packet(2, session='session-a', _asset_hash=self.reference_id)
        _, result = self.post_json(base+'/v1/frames', own_session_a)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(len(self.created_trackers), 2)
        self.assertEqual(self.created_trackers[0].resets, 0)
        self.assertEqual(self.created_trackers[1].resets, 1)

    def test_calibration_identity_cannot_change_inside_a_session(self):
        rows = [
            {'calibrationId': 'measured-a', 'width': 32, 'height': 24,
             'K': [[30., 0., 16.], [0., 30., 12.], [0., 0., 1.]],
             'kind': 'physical_measured', 'objectId': 1},
            {'calibrationId': 'measured-b', 'width': 32, 'height': 24,
             'K': [[30., 0., 16.], [0., 30., 12.], [0., 0., 1.]],
             'kind': 'physical_measured', 'objectId': 1},
        ]
        _write_calibrations(self.calibrations, rows)
        base = self.start_server(self.service())
        first = _packet(_asset_hash=self.reference_id, calibration='measured-a')
        _, result = self.post_json(base+'/v1/frames', first)
        self.assertEqual(result['poseState'], 'tracking')
        second = _packet(2, _asset_hash=self.reference_id, calibration='measured-b')
        status, result = self.post_json(base+'/v1/frames', second)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'calibration_identity_mismatch')
        self.assertIsNone(result['cameraFromObject'])
        self.assertEqual(self.created_trackers[0].resets, 1)

    def test_duplicate_camera_frame_metadata_is_rejected_and_offline_clock_is_not_forged(self):
        base = self.start_server(self.service())
        first = _packet(_asset_hash=self.reference_id, cameraFrameId=41, cameraMediaTime=1.25)
        _, result = self.post_json(base+'/v1/frames', first)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertTrue(result['freshCameraFrameVerified'])
        repeated_camera_frame = _packet(2, _asset_hash=self.reference_id,
                                        cameraFrameId=41, cameraMediaTime=1.28)
        status, result = self.post_json(base+'/v1/frames', repeated_camera_frame)
        self.assertEqual(status, 200)
        self.assertEqual(result['failureReason'], 'out_of_order_camera_frame')
        self.assertIsNone(result['cameraFromObject'])
        self.assertFalse(result['freshCameraFrameVerified'])

        rows = [{
            'calibrationId': 'dataset-reference-32x24', 'width': 32, 'height': 24,
            'cam_K': [30., 0., 16., 0., 30., 12., 0., 0., 1.],
            'kind': 'dataset_reference', 'objectId': 1,
        }]
        _write_calibrations(self.calibrations, rows)
        offline_service = self.service()
        offline_base = self.start_server(offline_service)
        physical_packet = _packet(session='offline-a', _asset_hash=self.reference_id,
                                  calibration='dataset-reference-32x24')
        _, result = self.post_json(offline_base+'/v1/frames', physical_packet)
        self.assertEqual(result['failureReason'], 'calibration_clock_mismatch')
        self.assertIsNone(result['cameraFromObject'])
        self.assertFalse(result['freshCameraFrameVerified'])

        offline_packet = _packet(session='offline-b', _asset_hash=self.reference_id,
                                 calibration='dataset-reference-32x24',
                                 timestampKind='frame_index_no_fps', captureTimestampSeconds=None)
        _, result = self.post_json(offline_base+'/v1/frames', offline_packet)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(result['calibrationKind'], 'dataset_reference')
        self.assertIsNone(result['captureTimestampSeconds'])
        self.assertFalse(result['freshCameraFrameVerified'])

    def test_synthetic_fallback_cannot_emit_a_physical_camera_pose(self):
        metadata_path = self.assets/'object-000001'/'asset.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata['evidenceKind'] = 'synthetic_control'
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        rows = [{
            'calibrationId': 'physical-calibration', 'width': 32, 'height': 24,
            'K': [[30., 0., 16.], [0., 30., 12.], [0., 0., 1.]],
            'kind': 'physical_measured', 'objectId': 1,
        }, {
            'calibrationId': 'synthetic-calibration', 'width': 32, 'height': 24,
            'K': [[30., 0., 16.], [0., 30., 12.], [0., 0., 1.]],
            'kind': 'synthetic', 'objectId': 1,
        }]
        _write_calibrations(self.calibrations, rows)
        base = self.start_server(self.service())
        physical = _packet(session='synthetic-live', _asset_hash=self.reference_id,
                           calibration='physical-calibration', cameraFrameId=1, cameraMediaTime=.1)
        _, result = self.post_json(base+'/v1/frames', physical)
        self.assertEqual(result['failureReason'], 'calibration_clock_mismatch')
        self.assertIsNone(result['cameraFromObject'])
        offline = _packet(session='synthetic-offline', _asset_hash=self.reference_id,
                          calibration='synthetic-calibration', timestampKind='frame_index_no_fps',
                          captureTimestampSeconds=None)
        _, result = self.post_json(base+'/v1/frames', offline)
        self.assertEqual(result['poseState'], 'tracking')
        self.assertEqual(result['calibrationKind'], 'synthetic')
        self.assertFalse(result['freshCameraFrameVerified'])

    def test_evaluation_fields_are_rejected_and_lan_requires_tls(self):
        service = self.service()
        base = self.start_server(service)
        packet = _packet(_asset_hash=self.reference_id, evaluationPose={'R': [1, 0, 0]})
        try:
            self.post_json(base+'/v1/frames', packet)
        except urllib.error.HTTPError as response:
            self.assertEqual(response.code, 400)
            body = json.loads(response.read().decode('utf-8'))
            self.assertEqual(body['failureReason'], 'unsupported_frame_fields')
        else:
            self.fail('Evaluation metadata must not enter inference.')

        validate_bind_security('127.0.0.1')
        with self.assertRaises(ValueError):
            validate_bind_security('10.20.30.40')
        with self.assertRaises(ValueError):
            validate_bind_security('8.8.8.8', 'cert.pem', 'key.pem')
        with self.assertRaises(ValueError):
            validate_bind_security('0.0.0.0')
        cert, key = self.root/'server-cert.pem', self.root/'server-key.pem'
        cert.write_text('test certificate placeholder', encoding='ascii')
        key.write_text('test private key placeholder', encoding='ascii')
        self.assertEqual(str(validate_bind_security('10.20.30.40', cert, key)), '10.20.30.40')

    def test_asset_hash_is_checked_before_loading(self):
        archive = self.assets/'object-000001'/'tracking-asset.npz'
        with archive.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_asset_registry(self.assets)

    def test_demo_route_cannot_serve_private_asset_files_or_escape_root(self):
        gallery = self.assets/'gallery.json'
        gallery.write_text(json.dumps({'objects': [{'objectId': 1, 'name': 'Synthetic cup'}]}),
                           encoding='utf-8')
        service = PrototypeService(
            self.assets, self.calibrations, demo_root=self.assets,
            client_path=self.client, tracker_factory=self.tracker_factory,
        )
        base = self.start_server(service)
        with urllib.request.urlopen(base+'/demo/gallery.json', timeout=3) as response:
            self.assertEqual(json.loads(response.read().decode('utf-8'))['objects'][0]['objectId'], 1)
        for path in ('/demo/object-000001/asset.json',
                     '/demo/object-000001/tracking-asset.npz',
                     '/demo/object-000001/gallery.json',
                     '/demo/%2e%2e%2fobject-000001/asset.json'):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base+path, timeout=3)
            self.assertEqual(caught.exception.code, 404)

    def test_http_body_limit_is_enforced_before_body_read(self):
        base = self.start_server(self.service())
        from http.client import HTTPConnection
        connection = HTTPConnection('127.0.0.1', int(base.rsplit(':', 1)[1]), timeout=3)
        connection.putrequest('POST', '/v1/frames', skip_host=True)
        connection.putheader('Host', f'127.0.0.1:{connection.port}')
        connection.putheader('Content-Type', 'application/json')
        connection.putheader('Content-Length', str(MAX_REQUEST_BYTES+1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read().decode('utf-8'))['error'], 'request_too_large')
        connection.close()


if __name__ == '__main__':
    unittest.main()
