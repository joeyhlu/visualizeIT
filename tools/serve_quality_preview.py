"""Read-only, route-limited comparison preview for an outbound internet tunnel."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import mimetypes
import gzip
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]/'artifacts'


def accepts_gzip(header):
    for item in header.split(','):
        parts=item.strip().lower().split(';')
        if parts[0]!='gzip':continue
        quality=1.
        for parameter in parts[1:]:
            if parameter.strip().startswith('q='):
                try:quality=float(parameter.strip()[2:])
                except ValueError:return False
        return 0<quality<=1
    return False


def preview_path(request_path):
    route = unquote(urlsplit(request_path).path)
    if route in ('/', '/model-quality', '/model-quality/'):
        return ROOT/'model-quality/index.html'
    if route in ('/model-quality/measured-runs.json', '/model-quality/model-memory-results.json'):
        return ROOT/route.lstrip('/')
    if re.fullmatch(r'/model-quality/(keyboard|mug|ranch)/player\.json', route):
        return ROOT/route.lstrip('/')
    if re.fullmatch(r'/video-60/(keyboard|mug|ranch)/(mesh\.bin|original\.webm)', route):
        return ROOT/route.lstrip('/')
    return None


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self): self.respond(False)
    def do_HEAD(self): self.respond(True)

    def respond(self, head):
        path = preview_path(self.path)
        if path is None or not path.is_file():
            self.send_error(404, 'Only comparison preview assets are exposed'); return
        # Reject links that resolve outside this comparison's artifact roots.
        resolved = path.resolve()
        if not any(resolved.is_relative_to((ROOT/name).resolve()) for name in ('model-quality', 'video-60')):
            self.send_error(404); return
        stat = path.stat()
        partial = self.headers.get('Range')
        compressible=path.suffix in ('.html','.json')
        compressed=compressible and not partial and accepts_gzip(self.headers.get('Accept-Encoding',''))
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}'+('-gzip' if compressed else '')+'"'
        if self.headers.get('If-None-Match') == etag:
            self.send_response(304); self.send_header('ETag', etag)
            if compressible:self.send_header('Vary','Accept-Encoding')
            self.end_headers(); return
        body = path.read_bytes()
        if path.name == 'index.html':
            body = body.replace(b'<a href="annotate.html">Review image annotations</a>', b'')
            body = body.replace(b'<a href="manifest.json">All traces, source hashes and measurement scope</a>', b'')
            body = body.replace(b'<a href="../video-hd/visual.html">Earlier box experiment</a>', b'')
        if compressed:body=gzip.compress(body,compresslevel=6,mtime=0)
        start, end = 0, len(body)-1
        if partial:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', partial)
            if not match or (not match[1] and not match[2]):
                self.send_error(416); return
            if match[1]:
                start = int(match[1]); end = min(end, int(match[2])) if match[2] else end
            else:
                start = max(0, len(body)-int(match[2]))
            if start > end or start >= len(body):
                self.send_response(416); self.send_header('Content-Range', f'bytes */{len(body)}'); self.end_headers(); return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end-start+1))
        if compressed:self.send_header('Content-Encoding','gzip')
        if compressible:self.send_header('Vary','Accept-Encoding')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('ETag', etag)
        self.send_header('Last-Modified', self.date_time_string(stat.st_mtime))
        self.send_header('X-Content-Type-Options', 'nosniff')
        if partial: self.send_header('Content-Range', f'bytes {start}-{end}/{len(body)}')
        self.end_headers()
        if not head: self.wfile.write(body[start:end+1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), PreviewHandler)
    print(f'Read-only comparison preview: http://127.0.0.1:{args.port}/model-quality/', flush=True)
    server.serve_forever()
