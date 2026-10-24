"""Narrow phone serving/HTTPS tests; no model inference or device-readiness claims."""
import datetime
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path
import ssl
import socket
import time
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools import phone_runtime as runtime
from tools import serve_design_studio as web


class PhoneStaticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='phone-static-', dir=web.REPO / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def server(self, **kwargs):
        server = web.create_server(port=0, storage_root=self.root / 'store', **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.shutdown()
            server.server_close()
            thread.join(5)
        self.addCleanup(close)
        return server

    def get(self, server, path, headers=None, context=None):
        cls = http.client.HTTPSConnection if context else http.client.HTTPConnection
        options = {'context': context} if context else {}
        conn = cls('127.0.0.1', server.server_port, timeout=10, **options)
        try:
            conn.request('GET', path, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_camera_routes_permissions_and_no_broad_directory_access(self):
        server = self.server()
        status, headers, body = self.get(server, '/camera')
        self.assertEqual(status, 200)
        self.assertIn(b'id="start"', body)
        self.assertIn(b'/phone-tracker.js', body)
        self.assertEqual(self.get(server, '/phone-tracker.js')[0], 200)
        self.assertIn("worker-src 'self'", headers['Content-Security-Policy'])
        self.assertIn("'wasm-unsafe-eval'", headers['Content-Security-Policy'])
        self.assertEqual(headers['Permissions-Policy'], 'camera=(self), microphone=(), geolocation=()')
        for path in ['/phone-runtime/../head.pt', '/phone-runtime/vision.d.ts', '/.cache/show3d/models/interactive_segmentation.task']:
            self.assertIn(self.get(server, path)[0], (400, 404, 503))
        self.assertEqual(self.get(server, '/api/studio/designs')[0], 401)

    def test_missing_runtime_is_explicit_and_keeps_camera_page_available(self):
        with patch.object(runtime, 'RUNTIME', self.root / 'absent'):
            server = self.server()
        self.assertEqual(self.get(server, '/camera')[0], 200)
        status, _, body = self.get(server, '/phone-runtime/manifest.json')
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)['error']['code'], 'phone_runtime_unavailable')

    def test_phone_manifest_and_icon_are_exact_fresh_routes(self):
        server = self.server()
        status, headers, body = self.get(server, '/phone.webmanifest')
        self.assertEqual(status, 200)
        self.assertIn('manifest+json', headers['Content-Type'])
        self.assertEqual(headers['Cache-Control'], 'no-store')
        manifest = json.loads(body)
        self.assertEqual(manifest['start_url'], '/camera')
        self.assertEqual(manifest['display'], 'standalone')
        self.assertTrue(manifest['icons'])
        for icon in manifest['icons']:
            self.assertTrue(icon['src'].startswith('/phone-icon'))
            self.assertEqual(self.get(server, icon['src'])[0], 200)
        for path in ['/phone-icons/', '/phone-icon-private.svg', '/service-worker.js']:
            self.assertEqual(self.get(server, path)[0], 404)

    def test_manifest_mismatch_fails_closed(self):
        folder = self.root / 'bad-runtime'
        folder.mkdir()
        (folder / 'manifest.json').write_text('{}')
        with patch.object(runtime, 'RUNTIME', folder):
            self.assertIsNone(runtime.PhoneRuntime().manifest)

    def test_pinned_asset_hash_and_size_verified_on_read(self):
        path = self.root / 'asset.js'
        path.write_bytes(b'known module')
        item = {'bytes': 12, 'sha256': hashlib.sha256(b'known module').hexdigest()}
        self.assertEqual(runtime.PhoneRuntime._read(path, item), b'known module')
        path.write_bytes(b'changed text')
        with self.assertRaises(ValueError):
            runtime.PhoneRuntime._read(path, item)

    def test_tls_requires_both_files(self):
        with self.assertRaisesRegex(ValueError, 'both'):
            web.create_server(port=0, tls_cert='unused.pem', storage_root=self.root / 'store')

    def test_actual_tls_and_origin_scheme(self):
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            self.skipTest('Existing runtime lacks optional certificate fixture generator')
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'phone test fixture')])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1))
                .not_valid_after(now+datetime.timedelta(hours=1))
                .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
                .sign(key, hashes.SHA256()))
        cert_path, key_path = self.root / 'cert.pem', self.root / 'key.pem'
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        server = self.server(tls_cert=cert_path, tls_key=key_path)
        context = ssl.create_default_context(cafile=str(cert_path))
        host = f'127.0.0.1:{server.server_port}'
        self.assertEqual(self.get(server, '/camera', {'Origin': 'https://' + host}, context)[0], 200)
        self.assertEqual(self.get(server, '/camera', {'Origin': 'http://' + host}, context)[0], 403)
        self.assertEqual(self.get(server, '/key.pem', context=context)[0], 404)
        stalled = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
        try:
            started = time.monotonic()
            self.assertEqual(self.get(server, '/camera', context=context)[0], 200)
            self.assertLess(time.monotonic()-started, 2.5, 'Silent TLS peer must not block another connection')
        finally:
            stalled.close()


if __name__ == '__main__':
    unittest.main()
