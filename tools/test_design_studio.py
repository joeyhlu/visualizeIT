"""Security and persistence tests for the private artwork studio; no neural work."""
import base64
import copy
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zlib

from PIL import Image
from tools import design_studio as ds
from tools import serve_design_studio as web


def png_bytes(width=256, height=256):
    output = io.BytesIO()
    Image.new("RGBA", (width, height), (31, 80, 170, 255)).save(output, format="PNG")
    return output.getvalue()


def request_payload():
    return {"schema": "visualizeit_design_v1", "name": "Blue waves", "prompt": "blue waves",
            "recipe": {"pattern": "waves", "colors": ["#ffffff", "#336699"], "text": "Hello 🌊",
                       "scale": 1, "rotation_deg": 0, "seed": 42, "width": 256, "height": 256},
            "png_base64": base64.b64encode(png_bytes()).decode("ascii")}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "studio"
        self.store = ds.StudioStore(self.root)

    def test_save_restart_image_and_manual_handoff(self):
        payload = request_payload()
        saved = self.store.create_design(payload)
        self.assertEqual(saved["png_sha256"], hashlib.sha256(png_bytes()).hexdigest())
        self.assertNotIn(str(self.root), json.dumps(saved))
        restarted = ds.StudioStore(self.root)
        self.assertEqual(restarted.get_image(saved["id"]), png_bytes())
        self.assertEqual(restarted.list_designs(), [saved])
        handoff = restarted.create_handoff(saved["id"])
        self.assertEqual(restarted.get_handoff(handoff["id"]), handoff)
        self.assertFalse(handoff["tracking_applied"])
        self.assertFalse(handoff["native_import_verified"])
        self.assertEqual(handoff["artwork"]["edge_px"], 256)

    def test_recipe_and_request_contracts(self):
        edits = [("name", ""), ("name", "n" * 81), ("prompt", "x" * 501),
                 ("schema", "bad"), ("png_base64", "!"), ("extra", 1)]
        recipe_edits = [("pattern", []), ("colors", ["#fff", "#000"]), ("colors", ["#ffffff"]),
                        ("text", "x" * 81), ("scale", True), ("scale", float("nan")),
                        ("scale", 10**400), ("rotation_deg", float("inf")), ("seed", True),
                        ("seed", -1), ("width", 255), ("height", 1025), ("seed", 1.0)]
        for key, value in edits:
            with self.subTest(key=key, value=str(value)[:20]):
                payload = request_payload()
                payload[key] = value
                with self.assertRaises(ds.StudioError):
                    self.store.create_design(payload)
        for key, value in recipe_edits:
            with self.subTest(recipe_key=key, value=str(value)[:20]):
                payload = request_payload()
                payload["recipe"][key] = value
                with self.assertRaises(ds.StudioError):
                    self.store.create_design(payload)
        self.assertEqual(self.store.list_designs(), [])

    def test_strict_json(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'\xff', b'[[['):
            with self.subTest(raw=raw), self.assertRaises(ds.StudioError):
                ds.loads_strict(raw)

    def test_png_crc_truncation_dimension_and_decode_rejected(self):
        png = png_bytes()
        bad_crc = bytearray(png)
        bad_crc[-1] ^= 1
        # Preserve valid container CRCs but corrupt the actual zlib payload.
        decode_bad = bytearray(png)
        offset = 8
        while offset < len(decode_bad):
            length, kind = struct.unpack_from(">I4s", decode_bad, offset)
            if kind == b"IDAT":
                decode_bad[offset + 8:offset + 8 + length] = b"x" * length
                struct.pack_into(">I", decode_bad, offset + 8 + length,
                                 zlib.crc32(decode_bad[offset + 4:offset + 8 + length]) & 0xffffffff)
                break
            offset += length + 12
        oversized_header = bytearray(png)
        struct.pack_into(">I", oversized_header, 16, 100000)
        struct.pack_into(">I", oversized_header, 29, zlib.crc32(oversized_header[12:29]) & 0xffffffff)
        for data in (png[:-6], bytes(bad_crc), bytes(decode_bad), bytes(oversized_header),
                     png + b"trailing", png_bytes(257), b"\x89PNG\r\n\x1a\n"):
            with self.subTest(size=len(data)):
                payload = request_payload()
                payload["png_base64"] = base64.b64encode(data).decode("ascii")
                with self.assertRaises(ds.StudioError) as caught:
                    self.store.create_design(payload)
                self.assertEqual(caught.exception.code, "invalid_png")

    def test_tamper_is_fail_closed(self):
        saved = self.store.create_design(request_payload())
        png_path = self.root / "designs" / saved["id"] / "image.png"
        png_path.write_bytes(png_bytes(257))
        with self.assertRaises(ds.StudioError) as caught:
            self.store.get_image(saved["id"])
        self.assertEqual(caught.exception.code, "corrupt_record")
        with self.assertRaises(ds.StudioError):
            self.store.list_designs()

    def test_metadata_and_handoff_tamper(self):
        saved = self.store.create_design(request_payload())
        handoff = self.store.create_handoff(saved["id"])
        manifest = self.root / "handoffs" / handoff["id"] / "manifest.json"
        changed = copy.deepcopy(handoff)
        changed["tracking_applied"] = True
        manifest.write_bytes(ds.json_bytes(changed))
        with self.assertRaises(ds.StudioError):
            self.store.get_handoff(handoff["id"])
        metadata = self.root / "designs" / saved["id"] / "metadata.json"
        changed = copy.deepcopy(saved)
        changed["image_url"] = "file:///private"
        metadata.write_bytes(ds.json_bytes(changed))
        with self.assertRaises(ds.StudioError):
            self.store.get_design(saved["id"])

    def test_quota_and_free_reserve_leave_no_visible_records(self):
        for constant, value in (("MAX_DESIGNS", 0), ("MAX_STORAGE_BYTES", 1)):
            with patch.object(ds, constant, value), self.assertRaises(ds.StudioError) as caught:
                self.store.create_design(request_payload())
            self.assertEqual(caught.exception.status, 507)
        with patch.object(ds.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(2**40, 0, 1)), self.assertRaises(ds.StudioError) as caught:
            self.store.create_design(request_payload())
        self.assertEqual(caught.exception.code, "disk_reserve")
        self.assertEqual(list((self.root / "designs").iterdir()), [])

    def test_atomic_publish_failure_rolls_back(self):
        with patch.object(ds.os, "rename", side_effect=OSError("forced publication failure")), self.assertRaises(OSError):
            self.store.create_design(request_payload())
        self.assertEqual(list((self.root / "designs").iterdir()), [])
        saved = self.store.create_design(request_payload())
        with patch.object(ds.os, "fsync", side_effect=OSError("forced disk failure")), self.assertRaises(OSError):
            self.store.create_handoff(saved["id"])
        self.assertEqual(list((self.root / "handoffs").iterdir()), [])
        self.assertEqual(self.store.get_image(saved["id"]), png_bytes())

    def test_retries_are_explicit_new_immutable_records(self):
        one = self.store.create_design(request_payload())
        two = self.store.create_design(request_payload())
        self.assertNotEqual(one["id"], two["id"])
        self.assertEqual(one["png_sha256"], two["png_sha256"])

    def test_clip_validation_and_handoff_limit(self):
        saved = self.store.create_design(request_payload())
        with self.assertRaises(ds.StudioError):
            self.store.create_handoff(saved["id"], "unknown")
        accepted = self.store.create_handoff(saved["id"], "real-clip", available_clip_ids={"real-clip"})
        self.assertEqual(accepted["clip_id"], "real-clip")
        with patch.object(ds, "MAX_HANDOFFS", 1), self.assertRaises(ds.StudioError):
            self.store.create_handoff(saved["id"])

    def test_path_traversal_rejected(self):
        for item_id in ("../private", "a" * 32 + "/../../", "C:\\Windows", "a" * 31):
            with self.assertRaises(ds.StudioError):
                self.store.get_design(item_id)

    def test_symlink_rejected(self):
        link = self.root / "designs" / ("a" * 32)
        try:
            link.symlink_to(self.root / "handoffs", target_is_directory=True)
        except OSError:
            self.skipTest("OS account cannot create symlinks; Windows reparse guard remains separately tested")
        with self.assertRaises(ds.StudioError):
            self.store.get_design("a" * 32)
        with self.assertRaises(ds.StudioError):
            ds.StudioStore(self.root)

    def test_windows_reparse_attribute_rejected(self):
        fake_stat = type("ReparseStat", (), {"st_mode": stat_mode(), "st_file_attributes": 0x400})()
        with patch.object(Path, "lstat", return_value=fake_stat), self.assertRaises(ds.StudioError):
            ds._no_links(self.root)


def stat_mode():
    import stat
    return stat.S_IFDIR


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        static = Path(self.tmp.name) / "static"
        static.mkdir()
        for name, _ in web.STATIC.values():
            (static / name).write_text("local static", encoding="utf-8")
        self.server = web.create_server(port=0, storage_root=Path(self.tmp.name) / "store", static_root=static,
                                        token="test-pairing-token-1234", upstream_port=1)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, path, method="GET", body=None, headers=None, auth=True):
        actual_headers = {"Authorization": "Bearer " + self.server.token} if auth else {}
        actual_headers.update(headers or {})
        if type(body) is dict:
            body = ds.json_bytes(body)
            actual_headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=actual_headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_real_http_roundtrip_and_authenticated_image(self):
        status, _, raw = self.request("/api/studio/designs", "POST", request_payload())
        self.assertEqual(status, 201)
        saved = json.loads(raw)
        self.assertEqual(self.request(saved["image_url"])[2], png_bytes())
        self.assertEqual(self.request(saved["image_url"], auth=False)[0], 401)
        self.assertEqual(self.request("/api/studio/designs")[0], 200)
        status, _, raw = self.request("/api/studio/handoffs", "POST", {"design_id": saved["id"], "clip_id": None})
        self.assertEqual(status, 201)
        handoff = json.loads(raw)
        self.assertEqual(json.loads(self.request("/api/studio/handoffs/" + handoff["id"])[2]), handoff)

    def test_host_origin_fetch_metadata_and_token_boundaries(self):
        cases = [({"Host": "evil.test"}, 403), ({"Origin": "https://evil.test"}, 403),
                 ({"Origin": "null"}, 403), ({"Sec-Fetch-Site": "cross-site"}, 403),
                 ({"Authorization": "Bearer wrong-token"}, 401)]
        for headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/api/studio/designs", headers=headers)[0], expected)
        host = f"127.0.0.1:{self.server.server_port}"
        self.assertEqual(self.request("/api/studio/designs", headers={"Origin": "http://" + host})[0], 200)
        self.assertEqual(self.request("/api/studio/designs?token=secret")[0], 400)
        self.assertEqual(self.request("/", auth=False)[0], 200)

    def test_unknown_methods_and_routes_json_errors(self):
        for method in ("DELETE", "OPTIONS", "CUSTOM", "TRACE", "CONNECT"):
            status, headers, raw = self.request("/api/studio/designs", method)
            self.assertEqual(status, 405)
            self.assertIn("error", json.loads(raw))
            self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request("/../private")[0], 404)
        self.assertEqual(self.request("/api/studio/unknown")[0], 404)

    def test_bad_request_bodies_do_not_persist(self):
        cases = [(b'{"name":1,"name":2}', {"Content-Type": "application/json"}, 400),
                 (b"{}", {"Content-Type": "text/plain"}, 415),
                 (b"{}", {"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
                 (b"", {"Content-Length": str(ds.MAX_JSON_BYTES + 1), "Content-Type": "application/json"}, 413)]
        for body, headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/api/studio/designs", "POST", body, headers)[0], expected)
        self.assertEqual(self.server.store.list_designs(), [])

    def test_disconnected_bridge_and_no_unvalidated_clip_handoff(self):
        status = json.loads(self.request("/api/studio/status")[2])
        self.assertEqual(status["workbench"]["state"], "disconnected")
        self.assertFalse(status["custom_artwork_tracking_supported"])
        self.assertEqual(json.loads(self.request("/api/studio/clips")[2])["clips"], [])
        saved = self.server.store.create_design(request_payload())
        status, _, _ = self.request("/api/studio/handoffs", "POST", {"design_id": saved["id"], "clip_id": "unknown"})
        self.assertEqual(status, 503)
        self.assertEqual(list((self.server.store.root / "handoffs").iterdir()), [])
        status, _, _ = self.request("/api/studio/handoffs", "POST", {"design_id": saved["id"], "clip_id": []})
        self.assertEqual(status, 400)

    def test_security_headers(self):
        _, headers, _ = self.request("/")
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertNotIn("unsafe-inline", headers["Content-Security-Policy"])
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["Cache-Control"], "no-store")


class BridgeTests(unittest.TestCase):
    def start_upstream(self, response, code=200, extra_headers=None):
        class Upstream(BaseHTTPRequestHandler):
            def do_GET(handler):
                handler.send_response(code)
                handler.send_header("Content-Type", "application/json")
                for key, value in (extra_headers or {}).items():
                    handler.send_header(key, value)
                raw = response if type(response) is bytes else ds.json_bytes(response)
                handler.send_header("Content-Length", str(len(raw)))
                handler.end_headers()
                handler.wfile.write(raw)

            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(2)))
        return web.WorkbenchBridge(server.server_port)

    def test_real_fixed_loopback_catalog_sanitizes_private_fields(self):
        bridge = self.start_upstream({"clips": [{"clip_id": "dev-mug", "label": None, "title": "Mug", "private_path": "C:/private"}]})
        self.assertEqual(bridge.clips()["clips"], [{"clip_id": "dev-mug", "label": "Mug"}])
        self.assertEqual(bridge.status()["state"], "connected")
        with self.assertRaises(ValueError):
            bridge._get("http://evil.test/")

    def test_redirects_are_disconnected_never_followed(self):
        bridge = self.start_upstream({}, 302, {"Location": "http://192.168.1.1/private"})
        self.assertEqual(bridge.clips()["state"], "disconnected")
        self.assertEqual(bridge.status()["state"], "disconnected")

    def test_slow_drip_upstream_has_total_deadline(self):
        stopped = threading.Event()
        class Slow(BaseHTTPRequestHandler):
            def do_GET(handler):
                try:
                    handler.send_response(200)
                    handler.send_header("Content-Type", "application/json")
                    handler.send_header("Content-Length", "10000")
                    handler.end_headers()
                    while not stopped.wait(.04):
                        handler.wfile.write(b" ")
                        handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.monotonic()
            self.assertEqual(web.WorkbenchBridge(server.server_port).clips()["state"], "disconnected")
            self.assertLess(time.monotonic() - started, 3.5)
        finally:
            stopped.set()
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_oversized_or_invalid_catalog_fails_closed(self):
        for body in (b" " * 262145, {"clips": [{"clip_id": "../private"}]},
                     {"clips": [{"clip_id": "x"}, {"clip_id": "x"}]}):
            bridge = self.start_upstream(body)
            self.assertEqual(bridge.clips()["state"], "disconnected")

    def test_only_numeric_private_bind_and_bounded_ports(self):
        for host in ("0.0.0.0", "localhost", "8.8.8.8", "169.254.1.1", "::", "100.64.0.1"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                web.create_server(host=host)
        for host in ("127.0.0.1", "10.1.2.3", "172.16.1.1", "192.168.1.2"):
            web._validate_host(host)
        for port in (0, -1, 65536, "8765", True):
            with self.assertRaises(ValueError):
                web.WorkbenchBridge(port)


if __name__ == "__main__":
    unittest.main()
