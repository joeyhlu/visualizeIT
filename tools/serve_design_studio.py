"""Private local phone artwork UI and read-only, fixed-loopback workbench bridge."""
from __future__ import annotations

import argparse
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import logging
from pathlib import Path
import re
import secrets
import socket
import ssl
import threading
import time

from tools.design_studio import (StudioStore, StudioError, MAX_JSON_BYTES, MAX_PNG_BYTES,
                                 MAX_DESIGNS, MAX_EDGE, MAX_STORAGE_BYTES, CLIP_RE,
                                 _keys, _no_links, json_bytes, loads_strict, validate_handoff_request)
from tools.phone_runtime import PhoneRuntime
from tools.phone_pipeline_bridge import disconnected_status, sanitize_tracking_status

LOG = logging.getLogger("visualizeit.design_studio")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/studio.css": ("studio.css", "text/css; charset=utf-8"),
          "/studio.js": ("studio.js", "text/javascript; charset=utf-8"),
          "/design-engine.js": ("design-engine.js", "text/javascript; charset=utf-8")}
STATIC.update({"/camera": ("phone.html", "text/html; charset=utf-8"),
               "/phone.html": ("phone.html", "text/html; charset=utf-8"),
               "/phone.css": ("phone.css", "text/css; charset=utf-8"),
               **{f"/{name}.js": (f"{name}.js", "text/javascript; charset=utf-8")
                  for name in ("phone-camera", "phone-segment-worker", "phone-selection", "phone-tracker", "phone-pipeline")}})
REPO = Path(__file__).resolve().parents[1]
STATIC.update({
    "/phone.webmanifest": ("phone.webmanifest", "application/manifest+json"),
    "/phone-icon.svg": ("phone-icon.svg", "image/svg+xml"),
    "/phone-icon.png": ("phone-icon.png", "image/png"),
    "/phone-glass-helpers.js": ("phone-glass-helpers.js", "text/javascript; charset=utf-8"),
})


class WorkbenchBridge:
    """Two fixed GET routes only. No proxy environment, hostnames, or redirects."""
    def __init__(self, port=8765):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("upstream port must be within 1..65535")
        self.port = port

    def _get(self, path):
        if path not in ("/api/status", "/api/clips"):
            raise ValueError("unsupported bridge route")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
        deadline = None
        try:
            connection.connect()
            # Inactivity timeouts alone permit an upstream to drip bytes forever.
            upstream_socket = connection.sock
            # Connect stays 1 s; reads may take up to 2 s (cold status), still bounded by the hard 2 s deadline.
            upstream_socket.settimeout(2.0)
            def expire():
                try:
                    upstream_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            deadline = threading.Timer(2, expire)
            deadline.daemon = True
            deadline.start()
            connection.request("GET", path, headers={"Accept": "application/json", "Host": f"127.0.0.1:{self.port}"})
            response = connection.getresponse()
            if response.status != 200 or response.getheader("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("unavailable upstream")
            raw = response.read(262145)
            if len(raw) > 262144:
                raise ValueError("upstream response exceeds limit")
            return loads_strict(raw)
        finally:
            if deadline is not None:
                deadline.cancel()
            connection.close()

    def status(self):
        try:
            result = self._get("/api/status")
            if type(result) is not dict:
                raise ValueError("invalid upstream status")
            return {"state": "connected", "message": "Local workbench is reachable. Custom artwork requires manual native import."}
        except (OSError, ValueError, StudioError, http.client.HTTPException):
            return {"state": "disconnected", "message": "Local workbench is unavailable. Local design creation still works."}

    def tracking_status(self):
        """Allowlisted projection of the real WorkbenchService status. Fails closed."""
        try:
            raw = self._get("/api/status")
        except (OSError, ValueError, StudioError, http.client.HTTPException):
            return disconnected_status()
        return sanitize_tracking_status(raw)

    def clips(self):
        try:
            body = self._get("/api/clips")
            if type(body) is not dict or type(body.get("clips")) is not list or len(body["clips"]) > 1000:
                raise ValueError("invalid upstream clips")
            clips, seen = [], set()
            for row in body["clips"]:
                if type(row) is not dict:
                    raise ValueError("invalid upstream clip")
                clip_id = row.get("clip_id")
                if type(clip_id) is not str or not CLIP_RE.fullmatch(clip_id) or clip_id in seen:
                    raise ValueError("invalid upstream clip id")
                label = clip_id
                for candidate in (row.get("label"), row.get("title")):
                    if type(candidate) is str and 1 <= len(candidate) <= 160 and not any(ord(c) < 32 for c in candidate):
                        label = candidate
                        break
                clips.append({"clip_id": clip_id, "label": label})
                seen.add(clip_id)
            return {"state": "connected", "clips": clips, "message": "Existing workbench clips; handoffs do not apply artwork tracking."}
        except (OSError, ValueError, StudioError, http.client.HTTPException):
            return {"state": "disconnected", "clips": [], "message": "Local workbench clips are unavailable."}


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    # Slow clients cannot create an unbounded number of active decoder threads.
    def __init__(self, address, handler):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            if isinstance(request, ssl.SSLSocket):
                request.settimeout(3)
                try:
                    request.do_handshake()
                except (OSError, ssl.SSLError):
                    self.shutdown_request(request)
                    return
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "VisualizeItStudio/1"
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)
        connection = self.connection
        def expire():
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.deadline = threading.Timer(10, expire)
        self.deadline.daemon = True
        self.deadline.start()

    def finish(self):
        self.deadline.cancel()
        super().finish()

    def log_message(self, *_):
        # Request lines can contain secrets in malformed URLs; never print them.
        pass

    def __getattr__(self, name):
        if name.startswith("do_"):
            return self._handle
        raise AttributeError(name)

    def _send(self, body, content_type="application/json; charset=utf-8", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; media-src 'self' blob:; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status=200):
        self._send(json_bytes(value), status=status)

    def _guard(self):
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in self.server.allowed_hosts:
            raise StudioError("forbidden_host", "Use the private studio host from the pairing URL.", 403)
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] != self.server.scheme + "://" + hosts[0]):
            raise StudioError("forbidden_origin", "Cross-origin requests are not accepted.", 403)
        fetch_sites = self.headers.get_all("Sec-Fetch-Site", [])
        if len(fetch_sites) > 1 or (fetch_sites and fetch_sites[0] not in ("none", "same-origin")):
            raise StudioError("cross_site", "Cross-site requests are not accepted.", 403)
        if "?" in self.path or "#" in self.path or not self.path.startswith("/") or self.path.startswith("//"):
            raise StudioError("invalid_path", "Use a studio route without a query string.", 400)
        if self.path.startswith("/api/studio/"):
            auth = self.headers.get_all("Authorization", [])
            expected = "Bearer " + self.server.token
            if len(auth) != 1 or not hmac.compare_digest(auth[0].encode("utf-8"), expected.encode("utf-8")):
                raise StudioError("unauthorized", "Pair this browser with the startup token.", 401)

    def _body(self):
        if self.headers.get("Transfer-Encoding") is not None:
            raise StudioError("invalid_body", "Transfer encoding is not supported.")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,8}", lengths[0]):
            raise StudioError("invalid_body", "A valid Content-Length is required.")
        length = int(lengths[0])
        if length > MAX_JSON_BYTES:
            raise StudioError("body_too_large", "Artwork request exceeds the 3 MiB limit.", 413)
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            raise StudioError("invalid_content_type", "Content-Type must be application/json.", 415)
        data = self.rfile.read(length)
        if len(data) != length:
            raise StudioError("invalid_body", "Request body is incomplete.")
        return loads_strict(data)

    def _dispatch(self):
        self._guard()
        path, store = self.path, self.server.store
        if self.command == "GET":
            if path.startswith('/phone-runtime/'):
                data, kind = self.server.phone_runtime.get(path)
                return self._send(data, kind)
            if path in STATIC:
                name, kind = STATIC[path]
                target = _no_links(self.server.static_root / name)
                if not target.is_file() or target.stat().st_size > 1024 * 1024:
                    raise StudioError("not_found", "Studio file is unavailable.", 404)
                return self._send(target.read_bytes(), kind)
            if path == "/api/studio/status":
                return self._json({"schema": "visualizeit_studio_status_v1",
                                   "limits": {"max_designs": MAX_DESIGNS, "max_storage_bytes": MAX_STORAGE_BYTES,
                                              "max_png_bytes": MAX_PNG_BYTES, "max_edge": MAX_EDGE},
                                   "workbench": self.server.bridge.status(), "custom_artwork_tracking_supported": False})
            if path == "/api/studio/tracking-status":
                return self._json(self.server.bridge.tracking_status())
            if path == "/api/studio/clips":
                return self._json(self.server.bridge.clips())
            if path == "/api/studio/designs":
                return self._json({"designs": store.list_designs()})
            match = re.fullmatch(r"/api/studio/designs/([0-9a-f]{32})(/image\.png)?", path)
            if match:
                if match[2]:
                    return self._send(store.get_image(match[1]), "image/png")
                return self._json(store.get_design(match[1]))
            match = re.fullmatch(r"/api/studio/handoffs/([0-9a-f]{32})", path)
            if match:
                return self._json(store.get_handoff(match[1]))
        elif self.command == "POST":
            if path == "/api/studio/tracking-frames":
                # Rejected before any body read, decode or backend construction.
                # Nothing is forwarded; live neural tracking has no reviewed contract.
                raise StudioError("live_neural_contract_unsupported",
                                  "Live phone frames are not accepted. Trained tracking is recorded-clip only.", 501)
            if path == "/api/studio/designs":
                return self._json(store.create_design(self._body()), 201)
            if path == "/api/studio/handoffs":
                body = self._body()
                _keys(body, ("design_id", "clip_id"))
                validate_handoff_request(body["design_id"], body["clip_id"])
                clips = self.server.bridge.clips() if body["clip_id"] is not None else {"state": "connected", "clips": []}
                if clips["state"] != "connected":
                    raise StudioError("workbench_unavailable", "Reconnect the workbench or create an unbound handoff.", 503)
                return self._json(store.create_handoff(body["design_id"], body["clip_id"],
                                  available_clip_ids={c["clip_id"] for c in clips["clips"]}), 201)
        else:
            raise StudioError("method_not_allowed", "This method is not supported.", 405)
        raise StudioError("not_found", "Studio route was not found.", 404)

    def _handle(self):
        started = time.monotonic()
        try:
            self._dispatch()
        except StudioError as exc:
            LOG.info("studio request rejected code=%s status=%s", exc.code, exc.status)
            self._json({"error": {"code": exc.code, "message": exc.message}}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (socket.timeout, TimeoutError):
            LOG.info("studio client request timed out")
        except Exception:
            LOG.error("studio operation failed; inspect local storage and runtime", exc_info=True)
            self._json({"error": {"code": "internal_error", "message": "Studio operation failed; inspect the local operator log."}}, 500)
        finally:
            LOG.debug("studio request elapsed_ms=%.1f", (time.monotonic() - started) * 1000)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = do_TRACE = do_CONNECT = _handle


def _validate_host(host):
    if host == "127.0.0.1":
        return
    try:
        address = ipaddress.IPv4Address(host)
        if any(address in ipaddress.IPv4Network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
            return
    except ipaddress.AddressValueError:
        pass
    raise ValueError("Bind only 127.0.0.1 or an explicit RFC1918 private IPv4 LAN address")


def create_server(host="127.0.0.1", port=8891, *, store=None, storage_root=None, static_root=None,
                  token=None, upstream_port=8765, tls_cert=None, tls_key=None):
    _validate_host(host)
    token = secrets.token_urlsafe(32) if token is None else token
    if type(token) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", token):
        raise ValueError("pairing token must be 16..128 URL-safe characters")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be within 0..65535")
    if bool(tls_cert) != bool(tls_key):
        raise ValueError('Provide both --tls-cert and --tls-key for HTTPS')
    tls = None
    if tls_cert:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(str(tls_cert), str(tls_key))
    bridge = WorkbenchBridge(upstream_port)
    store = store if store is not None else StudioStore(storage_root or REPO / ".cache" / "design-studio")
    server = StudioServer((host, port), Handler)
    server.scheme = 'https' if tls else 'http'
    if tls:
        try:
            # Defer each accepted connection's handshake to a bounded worker.
            # A silent TCP peer must not block the main accept loop.
            server.socket = tls.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        except Exception:
            server.server_close()
            raise
    server.token, server.store, server.bridge = token, store, bridge
    server.phone_runtime = PhoneRuntime()
    server.static_root = Path(static_root or REPO / "bench" / "web" / "design-studio")
    server.allowed_hosts = {f"{host}:{server.server_port}"}
    if host == "127.0.0.1":
        server.allowed_hosts.add(f"localhost:{server.server_port}")
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--lan-host", help="Explicit RFC1918 private IPv4 address; default is loopback only")
    parser.add_argument("--storage-root", type=Path, default=REPO / ".cache" / "design-studio")
    parser.add_argument("--upstream-port", type=int, default=8765)
    parser.add_argument('--tls-cert', type=Path, help='Certificate trusted by the phone, with this LAN IP in its SAN')
    parser.add_argument('--tls-key', type=Path, help='Private key for --tls-cert; never served')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        server = create_server(args.lan_host or "127.0.0.1", args.port, storage_root=args.storage_root,
                               upstream_port=args.upstream_port, tls_cert=args.tls_cert, tls_key=args.tls_key)
    except (ValueError, OSError, StudioError) as exc:
        parser.error(str(exc))
    host, port = server.server_address
    print(f"Private pairing URL: {server.scheme}://{host}:{port}/#token={server.token}", flush=True)
    print(f"On-device camera workspace: {server.scheme}://{host}:{port}/camera", flush=True)
    if server.scheme != 'https':
        print('Phone live camera requires HTTPS with a certificate trusted by that phone. Loopback supports desktop checks.', flush=True)
    print("Local recipe generator; manual PNG import only. LAN HTTP is for a trusted private network.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
