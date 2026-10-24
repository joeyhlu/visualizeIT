"""Bounded native HTTP adapter; all GoTrack lifecycle logic lives in bench."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import hmac
import json
import logging
import re
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bench import workbench_contract as wc
from bench.workbench_live_pose import ConfiguredGoTrackFactory, HEAD, LIMITS, NativeError

LOG = logging.getLogger("visualizeit.native_pose")
SESSION_PATH = re.compile(r"/v1/native/sessions/([a-f0-9]{32})(/frames)?\Z")


class NativePoseBridge:
    """One GPU session and operation at a time, including timed-out operations.

    A Python thread cannot safely interrupt a CUDA call. A timed-out operation
    stays quarantined and retains the exclusive lock until it actually exits;
    only that worker then closes its resources. No second model can overlap it.
    """
    def __init__(self, factory: ConfiguredGoTrackFactory | None = None):
        self.factory, self.session = factory, None
        self._operation = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-gotrack")
        self._shutdown = False
        self._quarantined = threading.Event()
        self._stop_reaper = threading.Event()
        self._reaper = threading.Thread(target=self._expire_idle, daemon=True, name="native-session-expiry")
        self._reaper.start()

    def _expire_idle(self):
        while not self._stop_reaper.wait(1.0):
            if not self._operation.acquire(blocking=False):
                continue
            try:
                if (self.session is not None and self.factory is not None and
                        time.monotonic()-self.session.last_activity > self.factory.config["session_idle_timeout_s"]):
                    LOG.info("native idle session expired")
                    self._close_session()
            finally:
                self._operation.release()

    def capabilities(self):
        if self.factory is None:
            return {"schema_version": 1, "model_head": HEAD, "installed_head_sha256": None,
                    "admission_ready": False, "live_camera_supported": False,
                    "recorded_finite_supported": True,
                    "blockers": ["no reviewed live GoTrack operator configuration", "runtime/live admission pending"],
                    "limits": dict(LIMITS)}
        if self._operation.locked():
            return {"schema_version": 1, "model_head": HEAD, "installed_head_sha256": None,
                    "admission_ready": False, "live_camera_supported": False,
                    "recorded_finite_supported": True, "blockers": ["native operation in flight"],
                    "limits": dict(LIMITS)}
        caps = self.factory.capabilities()
        if self.session is not None and not self.session._closed:
            caps["installed_head_sha256"] = self.session.installed_head
        if self._shutdown or self._quarantined.is_set():
            caps["admission_ready"] = caps["live_camera_supported"] = False
            caps["blockers"].append("native server shutting down or quarantined")
        return caps

    def _call(self, action):
        if self._shutdown:
            raise NativeError("server_closing", "native server is closing", 503)
        if self.factory is None:
            raise NativeError("live_unavailable", "reviewed live GoTrack configuration is required", 503)
        if self._quarantined.is_set() or not self._operation.acquire(blocking=False):
            raise NativeError("native_busy", "one native operation is already in flight", 409)
        def run():
            try:
                return action()
            except NativeError:
                raise
            except Exception as exc:
                LOG.exception("native operation failed")
                raise NativeError("backend_unavailable", "trusted native provider/backend failed; session closed", 503) from exc
        try:
            future = self._executor.submit(run)
        except BaseException:
            self._operation.release()
            raise
        handed_off = False
        try:
            return future.result(timeout=self.factory.config["frame_timeout_s"])
        except FutureTimeout as exc:
            # The request retains the operation lock until either the result
            # has been consumed or this completion callback has closed it.
            # Even a worker completing exactly at the deadline cannot overlap
            # a new session or escape timeout cleanup.
            self._quarantined.set()
            handed_off = True
            def cleanup(_):
                try:
                    self._close_session()
                finally:
                    self._quarantined.clear()
                    self._operation.release()
            future.add_done_callback(cleanup)
            LOG.warning("native operation timed out; worker quarantined until completion")
            raise NativeError("frame_timeout", "native operation timed out; session is closing", 504) from exc
        finally:
            if not handed_off:
                self._operation.release()

    def _close_session(self):
        session, self.session = self.session, None
        if session is not None:
            try:
                session.close()
            except Exception:
                LOG.exception("native session cleanup failed")

    def create(self, body):
        def action():
            if self.session is not None:
                idle = time.monotonic()-self.session.last_activity
                if self.session._closed or idle > self.factory.config["session_idle_timeout_s"]:
                    self._close_session()
                else:
                    raise NativeError("session_conflict", "close the existing native session first", 409)
            self.session = self.factory.create(body)
            LOG.info("native session created: %s", self.session.session_id)
            return self.session.describe()
        return self._call(action)

    def frame(self, session_id, body):
        def action():
            session = self.session
            if session is None or session.session_id != session_id:
                raise NativeError("session_not_found", "native session not found", 404)
            try:
                return session.process(body)
            except BaseException:
                if session._closed:
                    self._close_session()
                raise
        return self._call(action)

    def delete(self, session_id):
        def action():
            if self.session is None or self.session.session_id != session_id:
                raise NativeError("session_not_found", "native session not found", 404)
            self._close_session()
            return {"schema_version": 1, "session_id": session_id, "state": "closed"}
        return self._call(action)

    def close(self):
        self._shutdown = True
        self._stop_reaper.set()
        self._reaper.join(timeout=2.0)
        if self._operation.acquire(blocking=False):
            try:
                self._close_session()
            finally:
                self._operation.release()
        # A running worker owns its cleanup. Do not force-free CUDA resources
        # concurrently with estimation; process shutdown waits for that thread.
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._operation:
            self._close_session()


def make_handler(bridge: NativePoseBridge, token: str):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("operator Bearer token must contain at least 32 non-whitespace characters")

    class NativeHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "VisualizeITNative/1"
        timeout = 10

        def log_message(self, fmt, *args):
            # Never log token, source images, provider output or JSON bodies.
            LOG.info("native HTTP %s", args[1] if len(args) > 1 else "request")

        def _reply(self, status, body):
            if status >= 400:
                body = {"state": "unavailable", "mask_state": ("rejected" if body.get("error") in
                        ("mask_identity", "mask_contract", "image_identity", "invalid_jpeg") else "unavailable"),
                        "pose_state": "unavailable", "render_state": "unavailable", "pose": None,
                        "render": None, "outline_segments": [], **body}
            data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(data)

        def _authorize(self):
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1 or not auth[0].isascii() or not hmac.compare_digest(auth[0], "Bearer "+token):
                raise NativeError("unauthorized", "operator Bearer token required", 401)
            # Native URLSession sends no Origin. This listener grants no browser
            # cross-origin upload capability and has no OPTIONS/CORS exception.
            if self.headers.get("Origin") is not None:
                raise NativeError("origin_forbidden", "browser-origin requests are unsupported", 403)
            if self.headers.get("Transfer-Encoding") is not None or self.headers.get("Expect") is not None:
                raise NativeError("streaming_unsupported", "streaming/expect request bodies are unsupported", 400)

        def handle_expect_100(self):
            self._reply(417, {"schema_version": 1, "error": "streaming_unsupported",
                              "reason": "Expect requests are unsupported"})
            return False

        def _body(self):
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,9}", lengths[0]):
                raise NativeError("length_required", "exact bounded Content-Length is required", 411)
            length = int(lengths[0])
            if not 0 < length <= LIMITS["max_body_bytes"]:
                raise NativeError("body_too_large", "request exceeds native body limit", 413)
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise NativeError("content_type", "application/json is required", 415)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise NativeError("truncated_body", "request body ended early", 400)
            try:
                return wc.loads_strict(raw)
            except (ValueError, UnicodeError) as exc:
                raise NativeError("invalid_json", "request must be strict UTF-8 JSON", 400) from exc

        def _dispatch(self):
            try:
                self._authorize()
                if self.command == "GET" and self.path == "/v1/native/capabilities":
                    return self._reply(200, bridge.capabilities())
                if self.command == "POST" and self.path == "/v1/native/sessions":
                    return self._reply(201, bridge.create(self._body()))
                match = SESSION_PATH.fullmatch(self.path)
                if match and self.command == "POST" and match.group(2) == "/frames":
                    return self._reply(200, bridge.frame(match.group(1), self._body()))
                if match and self.command == "DELETE" and match.group(2) is None:
                    if self.headers.get("Content-Length", "0") != "0":
                        raise NativeError("unexpected_body", "DELETE has no request body", 400)
                    return self._reply(200, bridge.delete(match.group(1)))
                raise NativeError("route_not_found", "native route not found", 404)
            except NativeError as exc:
                self._reply(exc.status, {"schema_version": 1, "error": exc.code, "reason": str(exc)})
            except (socket.timeout, TimeoutError):
                self._reply(408, {"schema_version": 1, "error": "request_timeout", "reason": "request body timed out"})
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception:
                LOG.exception("native HTTP failure")
                self._reply(500, {"schema_version": 1, "error": "internal_error", "reason": "native request failed"})

        do_GET = do_POST = do_DELETE = _dispatch

    return NativeHandler


class BoundedNativeServer(ThreadingHTTPServer):
    """Bound accepted HTTP workers as well as model operations."""
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, handler):
        self._connections = threading.BoundedSemaphore(8)
        self._socket_lock = threading.Lock()
        self._accepted_sockets = set()
        super().__init__(address, handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(10)
        with self._socket_lock:
            self._accepted_sockets.add(request)
        return request, address

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._socket_lock:
                self._accepted_sockets.discard(request)

    def server_close(self):
        try:
            super().server_close()
        finally:
            with self._socket_lock:
                accepted = list(self._accepted_sockets)
                self._accepted_sockets.clear()
            for request in accepted:
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                finally:
                    request.close()

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            # The listener accepts without negotiating TLS. Handshake only
            # after obtaining a bounded worker slot and setting the accepted
            # socket's timeout; idle unauthenticated peers cannot stall accept.
            if isinstance(request, ssl.SSLSocket):
                try:
                    request.do_handshake()
                except (ssl.SSLError, OSError):
                    LOG.info("native TLS handshake failed or timed out")
                    self.shutdown_request(request)
                    return
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()
