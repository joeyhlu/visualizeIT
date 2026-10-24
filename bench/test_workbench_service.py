"""Codex-authored adversarial boundary tests for the local video workbench."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bench import workbench_service as service


class PathAndBudgetTests(unittest.TestCase):
    def test_status_rechecks_admission_and_retains_other_weight_blockers(self):
        from bench import workbench_neural as neural, workbench_reconstruction as reconstruction
        with tempfile.TemporaryDirectory() as directory:
            app = service.WorkbenchService(directory, run_options={"admission": "fixture", "mask_set": "observed-fixture"})
            accepted = SimpleNamespace(admitted=True, reasons=(), body={"profile": "fixture"})
            rejected = SimpleNamespace(admitted=False, reasons=("source changed",), body={})
            preflight = {"blocked_reasons": ["serving_admission", "reference_best_checkpoint", "checkpoint_mismatch"],
                         "primary": {"fixture": True}}
            with patch.object(neural, "load_admission", side_effect=[accepted, rejected]) as load, \
                 patch.object(neural, "neural_preflight", return_value=preflight), \
                 patch.object(neural, "segmentation_preflight", return_value={"ready": False, "blocked_reasons": []}), \
                 patch.object(reconstruction, "preflight", return_value={"available": False}):
                first = app.status()
                second = app.status()
            self.assertEqual(load.call_count, 2)
            self.assertEqual(first["capabilities"]["neural_blocked"], ["checkpoint_mismatch"])
            self.assertTrue(first["admission"]["admitted"])
            self.assertFalse(second["admission"]["admitted"])
            self.assertIn("serving_admission", second["capabilities"]["neural_blocked"])
            self.assertEqual(second["admission"]["reasons"], ["source changed"])
            self.assertEqual(first["defaults"]["mask_set"], "observed-fixture")

    def test_unreadable_admission_stays_blocked(self):
        from bench import workbench_neural as neural
        with tempfile.TemporaryDirectory() as directory:
            app = service.WorkbenchService(directory, run_options={"admission": "fixture"})
            with patch.object(neural, "load_admission", side_effect=ValueError("altered receipt")):
                status = app._admission_status()
            self.assertFalse(status["admitted"])
            self.assertIn("altered receipt", status["reasons"][0])

    def test_paths_cannot_escape_registered_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside", "/absolute", "C:/secret", "nested\\secret", "x\x00y"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    service._safe_member(root, name, require_file=False)

    def test_only_registered_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "good.mp4").write_bytes(b"video fixture")
            self.assertEqual(service._safe_member(root, "good.mp4"), (root / "good.mp4").resolve())
            with self.assertRaises((ValueError, FileNotFoundError)):
                service._safe_member(root, "absent.mp4")

    def test_loopback_only_even_when_port_is_ephemeral(self):
        with tempfile.TemporaryDirectory() as directory:
            app = service.WorkbenchService(directory)
            for host in ("0.0.0.0", "192.168.1.2", "example.com"):
                with self.subTest(host=host), self.assertRaises(ValueError):
                    service.create_server(app, host, 0)

    def test_input_budget_is_decimal_250mb(self):
        # The human's approved cap is 250,000,000 bytes, not 250 MiB.
        self.assertLessEqual(service.MAX_UPLOAD_BYTES, 250_000_000)
        self.assertLessEqual(service.MAX_IMPORT_TOTAL_BYTES, 250_000_000)

    def test_aggregate_uploads_refuse_over_budget_and_preserve_old(self):
        with tempfile.TemporaryDirectory() as directory:
            app = service.WorkbenchService(directory)
            with patch.object(service, "MAX_UPLOAD_BYTES", 16), patch.object(service, "MAX_IMPORT_TOTAL_BYTES", 20):
                app.accept_upload("first.mp4", "video/mp4", b"123456789012")
                before = sorted(path.name for path in app.import_root.iterdir())
                with self.assertRaises(ValueError):
                    app.accept_upload("second.mp4", "video/mp4", b"123456789")
                self.assertEqual(before, sorted(path.name for path in app.import_root.iterdir()))

    def test_unknown_clip_and_run_cannot_start_work(self):
        with tempfile.TemporaryDirectory() as directory:
            app = service.WorkbenchService(directory)
            with self.assertRaises((KeyError, ValueError)):
                app.create_run({"clip_id": "not-registered"})
            with self.assertRaises((KeyError, ValueError)):
                app.create_job("not-registered")


class HTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = service.WorkbenchService(self.temp.name)
        self.server = service.create_server(self.app, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def test_status_is_real_local_service(self):
        with urlopen(self.url + "/api/status", timeout=3) as response:
            self.assertEqual(response.status, 200)
            self.assertIsInstance(json.load(response), dict)
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_cross_origin_mutation_rejected(self):
        request = Request(self.url + "/api/jobs", data=b"{}", method="POST",
                          headers={"Content-Type": "application/json", "Origin": "https://example.com"})
        with self.assertRaises(HTTPError) as result:
            urlopen(request, timeout=3)
        self.assertEqual(result.exception.code, 403)

    def test_unregistered_files_are_not_served(self):
        (Path(self.temp.name) / "private.txt").write_text("private")
        for path in ("/private.txt", "/media/no-such-clip", "/api/clips/no-such-clip"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as result:
                urlopen(self.url + path, timeout=3)
            self.assertIn(result.exception.code, (400, 404))


if __name__ == "__main__":
    unittest.main()
