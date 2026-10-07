"""Verify the shared preview's route scope, cache refresh and video seeking."""
import tempfile
import gzip
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from tools import serve_quality_preview as preview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'model-quality/mug').mkdir(parents=True)
        (self.root/'model-quality/index.html').write_text('<html>comparison</html>')
        (self.root/'model-quality/mug/player.json').write_text('{"frames": []}')
        (self.root/'video-60/mug').mkdir(parents=True)
        (self.root/'video-60/mug/original.webm').write_bytes(bytes(range(100)))
        self.patch = patch.object(preview, 'ROOT', self.root); self.patch.start()
        class QuietHandler(preview.PreviewHandler):
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.patch.stop(); self.temp.cleanup()

    def test_unrelated_paths_and_encoded_traversal_are_not_exposed(self):
        for path in ('/.git/config', '/model-quality/annotations/private.json', '/video-60/',
                     '/model-quality/%2e%2e/secrets.txt', '/model-quality/mug/player.json/extra'):
            with self.assertRaises(HTTPError) as error: urlopen(self.url+path)
            self.assertEqual(error.exception.code, 404)

    def test_video_byte_ranges_and_head(self):
        path = self.url+'/video-60/mug/original.webm'
        with urlopen(Request(path, headers={'Range': 'bytes=10-19'})) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), bytes(range(10, 20)))
            self.assertEqual(response.headers['Content-Range'], 'bytes 10-19/100')
        with urlopen(Request(path, method='HEAD')) as response:
            self.assertEqual(response.read(), b'')
            self.assertEqual(response.headers['Content-Length'], '100')

    def test_revision_changes_when_saved_results_change(self):
        path = self.url+'/model-quality/mug/player.json'
        with urlopen(Request(path, method='HEAD')) as response: before = response.headers['ETag']
        (self.root/'model-quality/mug/player.json').write_text('{"frames": [1]}')
        with urlopen(Request(path, method='HEAD')) as response: after = response.headers['ETag']
        self.assertNotEqual(before, after)
        with self.assertRaises(HTTPError) as error: urlopen(Request(path, headers={'If-None-Match': after}))
        self.assertEqual(error.exception.code, 304)

    def test_compressed_results_preserve_bytes_and_have_representation_etags(self):
        body=b'{"frames": ['+b'123, '*10000+b'0]}'
        (self.root/'model-quality/mug/player.json').write_bytes(body)
        path=self.url+'/model-quality/mug/player.json'
        with urlopen(Request(path,headers={'Accept-Encoding':'gzip'})) as response:
            encoded=response.read();tag=response.headers['ETag']
            self.assertEqual(response.headers['Content-Encoding'],'gzip')
            self.assertEqual(gzip.decompress(encoded),body)
            self.assertLess(len(encoded),len(body)/10)
        with urlopen(Request(path,method='HEAD',headers={'Accept-Encoding':'gzip'})) as response:
            self.assertEqual(int(response.headers['Content-Length']),len(encoded))
            self.assertEqual(response.headers['ETag'],tag)
        with urlopen(Request(path,headers={'Accept-Encoding':'gzip;q=0'})) as response:
            self.assertIsNone(response.headers.get('Content-Encoding'))
            self.assertEqual(response.read(),body);self.assertNotEqual(response.headers['ETag'],tag)
        with urlopen(Request(path,headers={'Accept-Encoding':'gzip','Range':'bytes=0-9'})) as response:
            self.assertEqual(response.status,206);self.assertIsNone(response.headers.get('Content-Encoding'))
            self.assertEqual(response.read(),body[:10])


if __name__ == '__main__': unittest.main()
