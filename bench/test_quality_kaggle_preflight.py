"""Manufactured filesystem/package evidence; no model or actual Kaggle runtime."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench import quality_kaggle_preflight as probe


class KagglePreflightTests(unittest.TestCase):
    def request(self, root):
        archive = root / "source.zip"
        archive.write_bytes(b"manufactured source archive identity; not a runtime")
        return {"provenance": {"provider": "kaggle", "notebook_url": "https://www.kaggle.com/code/fixture/fixture",
                "observed_at": "2026-10-17T00:00:00Z", "evidence_source": "manufactured_test"},
            "proposed_roots": {"working": str(root)},
            "source_archive": {"path": str(archive), "sha256": probe.digest(archive), "bytes": archive.stat().st_size},
            "assets": [{"path": str(archive), "sha256": probe.digest(archive), "bytes": archive.stat().st_size}]}

    def inspect(self, request):
        with patch.object(probe.platform, "system", return_value="Linux"), \
             patch.object(probe.platform, "python_version", return_value="3.11.9"), \
             patch.object(probe.platform, "python_version_tuple", return_value=("3", "11", "9")), \
             patch.object(probe.metadata, "version", side_effect=lambda name: probe.REQUIREMENTS[name]):
            return probe.inspect_environment(request)

    def test_provenance_archive_and_actual_disk(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            request = self.request(Path(temp))
            report = self.inspect(request)
            self.assertTrue(report["source_archive"]["verified"])
            self.assertGreater(report["roots"]["working"]["actual_free_bytes"], 0)
            self.assertFalse(report["ready_for_training"])
            self.assertIn("train_serve_parity_required", report["blockers"])
            del request["provenance"]
            self.assertIn("explicit_kaggle_provenance_missing", self.inspect(request)["blockers"])
            Path(request["source_archive"]["path"]).write_bytes(b"changed")
            self.assertIn("source_archive_missing_or_changed", self.inspect(request)["blockers"])

    def test_unsupported_actual_cpu_image_reported_without_repair(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            request = self.request(Path(temp))
            with patch.object(probe.platform, "python_version_tuple", return_value=("3", "13", "15")), \
                 patch.object(probe.metadata, "version", return_value="2.11.0+cpu"):
                report = probe.inspect_environment(request)
            self.assertIn("unsupported_python_requires_3.10_or_3.11", report["blockers"])
            self.assertIn("unsupported_package:torch", report["blockers"])
            self.assertEqual(report["packages"]["torch"]["installed"], "2.11.0+cpu")

    def test_no_neural_import_process_launch_or_input_mutation(self):
        source = Path(probe.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
        self.assertFalse(imports & {"torch", "numpy", "subprocess", "requests"})
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("torch", "subprocess")) for n in ast.walk(tree)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            request = self.request(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch("subprocess.Popen", side_effect=AssertionError("process launched")):
                self.inspect(request)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
            with self.assertRaises(FileExistsError):
                probe.write_report(root / "source.zip", {})

    def test_bounded_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "request.json"
            for text in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    probe.read_json(path)


if __name__ == "__main__":
    unittest.main()
