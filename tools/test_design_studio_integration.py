"""Parent-owned HTTP tests for the phone studio and its narrow workbench bridge.

Fixtures are manufactured artwork and an explicitly fake upstream, never tracking evidence.
The actual disk reserve remains enabled. No model, GPU or external network is used.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]


def fixture_png(edge=256):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    pixels = (b'\x00' + bytes((20, 64, 90, 255)) * edge) * edge
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', edge, edge, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def design_payload():
    return {'schema': 'visualizeit_design_v1', 'name': 'HTTP test design',
            'prompt': 'navy and cream stripes with "HELLO"',
            'recipe': {'pattern': 'stripes', 'colors': ['#123456', '#F2E8D5'], 'text': 'HELLO',
                       'scale': 1, 'rotation_deg': 0, 'seed': 1, 'width': 256, 'height': 256},
            'png_base64': base64.b64encode(fixture_png()).decode('ascii')}


class StudioHTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        if shutil.disk_usage(ROOT).free < (1 << 30) + (32 << 20):
            self.skipTest('Actual disk reserve plus studio allowance is unavailable.')
        from tools.serve_design_studio import create_server
        self.make_server = create_server
        self.temp = tempfile.TemporaryDirectory(prefix='studio-http-', dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.upstream_requests = []
        request_log = self.upstream_requests

        class Upstream(BaseHTTPRequestHandler):
            def do_GET(inner):
                request_log.append((inner.command, inner.path))
                if inner.path == '/api/clips':
                    body = {'clips': [{'clip_id': 'fixture-mug', 'title': 'Manufactured catalog clip',
                                       'private_path': 'DO_NOT_PUBLISH_PRIVATE_PATH'}], 'imports': []}
                elif inner.path == '/api/status':
                    body = {'service': 'visualizeit-workbench', 'schema': 'workbench_service_v2',
                            'registered_clip_count': 1, 'secret': 'DO_NOT_PUBLISH_PRIVATE_PATH'}
                else:
                    inner.send_error(404)
                    return
                encoded = json.dumps(body).encode()
                inner.send_response(200)
                inner.send_header('Content-Type', 'application/json')
                inner.send_header('Content-Length', str(len(encoded)))
                inner.end_headers()
                inner.wfile.write(encoded)

            def log_message(inner, *_args):
                pass

        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.addCleanup(self.stop_upstream)
        self.server = self.make_server('127.0.0.1', 0, storage_root=Path(self.temp.name) / 'studio',
                                      upstream_port=self.upstream.server_port)
        self.start_server()
        self.addCleanup(self.stop_server)

    def start_server(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=3)
            self.server = None

    def stop_upstream(self):
        if self.upstream is not None:
            self.upstream.shutdown()
            self.upstream.server_close()
            self.upstream_thread.join(timeout=3)
            self.upstream = None

    def request(self, method, path, payload=None, *, auth=True, extra=None, raw=None):
        headers = {'Origin': f'http://127.0.0.1:{self.server.server_port}'}
        if auth:
            headers['Authorization'] = f'Bearer {self.server.token}'
        if payload is not None or raw is not None:
            headers['Content-Type'] = 'application/json'
        headers.update(extra or {})
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        try:
            client.request(method, path, body=body, headers=headers)
            response = client.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), data
        finally:
            client.close()

    def json_request(self, method, path, payload=None, **kwargs):
        status, headers, body = self.request(method, path, payload, **kwargs)
        return status, headers, json.loads(body)

    def save(self):
        status, _, body = self.json_request('POST', '/api/studio/designs', design_payload())
        self.assertEqual(status, 201, body)
        return body

    def test_actual_http_png_to_handoff_roundtrip_and_persistence(self):
        created = self.save()
        design_id = created['id']
        status, _, png = self.request('GET', created['image_url'])
        self.assertEqual(status, 200)
        self.assertEqual(png, fixture_png())
        self.assertEqual(created['png_sha256'], hashlib.sha256(png).hexdigest())
        status, _, handoff = self.json_request('POST', '/api/studio/handoffs',
                                               {'design_id': design_id, 'clip_id': 'fixture-mug'})
        self.assertEqual(status, 201, handoff)
        self.assertEqual(handoff['schema'], 'visualizeit_artwork_handoff_v1')
        self.assertEqual(handoff['state'], 'ready_for_manual_import')
        self.assertIs(handoff['tracking_applied'], False)
        self.assertIs(handoff['native_import_verified'], False)
        self.assertEqual(handoff['png_sha256'], created['png_sha256'])
        self.assertEqual(handoff['clip_id'], 'fixture-mug')
        self.assertEqual(self.json_request('GET', '/api/studio/handoffs/' + handoff['id'])[2], handoff)
        self.stop_server()
        self.server = self.make_server('127.0.0.1', 0, storage_root=Path(self.temp.name) / 'studio',
                                      upstream_port=self.upstream.server_port)
        self.start_server()
        self.assertEqual(self.json_request('GET', '/api/studio/designs/' + design_id)[2], created)
        self.assertEqual(self.request('GET', created['image_url'])[2], png)

    def test_bridge_sanitizes_actual_upstream_and_has_no_job_proxy(self):
        for route in ('status', 'clips'):
            status, headers, body = self.request('GET', '/api/studio/' + route)
            self.assertEqual(status, 200, body)
            self.assertNotIn(b'DO_NOT_PUBLISH_PRIVATE_PATH', body)
            self.assertNotIn('Access-Control-Allow-Origin', headers)
        clips = self.json_request('GET', '/api/studio/clips')[2]
        self.assertEqual(clips['state'], 'connected')
        self.assertEqual(clips['clips'][0]['clip_id'], 'fixture-mug')
        self.assertIs(self.json_request('GET', '/api/studio/status')[2]['custom_artwork_tracking_supported'], False)
        before = len(self.upstream_requests)
        for path in ('/api/runs', '/api/jobs', '/api/studio/proxy', '/api/studio/../../runs'):
            with self.subTest(path=path):
                self.assertIn(self.request('POST', path, {})[0], (400, 404, 405))
        self.assertEqual(len(self.upstream_requests), before)
        self.assertTrue(all(method == 'GET' and path in ('/api/status', '/api/clips')
                            for method, path in self.upstream_requests))

    def test_auth_origin_host_and_methods_refuse_before_publication(self):
        endpoint = '/api/studio/designs'
        for kwargs in ({'auth': False}, {'extra': {'Authorization': 'Bearer wrong'}},
                       {'extra': {'Origin': 'https://attacker.invalid'}},
                       {'extra': {'Host': 'attacker.invalid'}},
                       {'extra': {'Sec-Fetch-Site': 'cross-site'}}):
            with self.subTest(kwargs=kwargs):
                self.assertIn(self.request('POST', endpoint, design_payload(), **kwargs)[0], (400, 401, 403, 421))
        for method in ('PUT', 'DELETE', 'PATCH', 'OPTIONS'):
            self.assertIn(self.request(method, endpoint, {})[0], (403, 405))
        self.assertEqual(self.json_request('GET', endpoint)[2]['designs'], [])

    def test_malformed_payloads_and_photo_data_are_not_published(self):
        endpoint = '/api/studio/designs'
        candidates = []
        payload = design_payload()
        payload['photo_base64'] = 'PRIVATE_PHOTO_MUST_NOT_BE_ACCEPTED'
        candidates.append(payload)
        payload = design_payload()
        payload['recipe']['width'] = True
        candidates.append(payload)
        payload = design_payload()
        payload['png_base64'] = base64.b64encode(fixture_png()[:-8]).decode()
        candidates.append(payload)
        for payload in candidates:
            self.assertEqual(self.request('POST', endpoint, payload)[0], 400)
        self.assertEqual(self.request('POST', endpoint, raw=b'{"schema":"a","schema":"b"}')[0], 400)
        self.assertEqual(self.json_request('GET', endpoint)[2]['designs'], [])

    def test_disconnected_backend_does_not_prevent_unbound_artwork_handoff(self):
        saved = self.save()
        self.stop_upstream()
        status, _, state = self.json_request('GET', '/api/studio/clips')
        self.assertEqual(status, 200)
        self.assertEqual(state['state'], 'disconnected')
        self.assertEqual(state['clips'], [])
        status, _, response = self.json_request('POST', '/api/studio/handoffs',
                                                 {'design_id': saved['id'], 'clip_id': 'fixture-mug'})
        self.assertIn(status, (400, 409, 503), response)
        status, _, response = self.json_request('POST', '/api/studio/handoffs',
                                                 {'design_id': saved['id'], 'clip_id': None})
        self.assertEqual(status, 201, response)
        self.assertIsNone(response['clip_id'])


if __name__ == '__main__':
    unittest.main()
