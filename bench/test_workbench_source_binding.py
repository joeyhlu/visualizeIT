"""Changed sources and mesh resources must invalidate serving evidence."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from bench import workbench_neural as wn


class SourceBindingTests(unittest.TestCase):
    def test_failure_record_helper_is_bound_into_neural_source_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bench/workbench_reconstruction.py"
            source.parent.mkdir()
            source.write_bytes(b"original failure-record helper")
            with patch.object(wn, "historical_source_map", return_value={}):
                before = wn.source_closure(root)
                source.write_bytes(b"changed failure-record helper")
                after = wn.source_closure(root)
            self.assertNotEqual(before["closure_sha256"], after["closure_sha256"])
            self.assertIn("bench/workbench_reconstruction.py", before["current_files"])

    def test_review_is_exact_path_old_hash_and_current_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "bench/quality_gotrack.py"
            path.parent.mkdir()
            path.write_bytes(b"reviewed current source")
            old = "a" * 64
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            table = {"bench/quality_gotrack.py": {"training_sha256": old, "current_sha256": current,
                                                   "scope": "fixture", "reason": "fixture reviewed"}}
            with patch.object(wn, "historical_source_map", return_value={"bench/quality_gotrack.py": old}), \
                    patch.object(wn, "REVIEWED_REPLACEMENTS", table):
                closure = wn.source_closure(root)
                self.assertIn("bench/quality_gotrack.py", closure["reviewed_replacements"])
                first_hash = closure["closure_sha256"]
                path.write_bytes(b"changed after review")
                changed = wn.source_closure(root)
                self.assertIn("bench/quality_gotrack.py", changed["unreviewed_replacements"])
                self.assertNotEqual(first_hash, changed["closure_sha256"])

    def test_unlisted_historical_replacement_is_never_approved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "other.py").write_bytes(b"new bytes")
            with patch.object(wn, "historical_source_map", return_value={"other.py": "a" * 64}):
                result = wn.source_closure(root)
            self.assertIn("other.py", result["unreviewed_replacements"])
            self.assertEqual(result["reviewed_replacements"], {})

    def test_external_mesh_resources_are_rejected_before_loader(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = root / "plane.glb"
            wn.write_plane_glb(original)
            raw = original.read_bytes()
            length = struct.unpack_from("<I", raw, 12)[0]
            document = json.loads(raw[20:20 + length])
            remaining = raw[20 + length:]
            for key in ("images", "buffers"):
                with self.subTest(key=key):
                    changed = dict(document)
                    changed[key] = [{"uri": "unbound-external-file.png", "byteLength": 16}]
                    blob = json.dumps(changed).encode()
                    blob += b" " * (-len(blob) % 4)
                    body = struct.pack("<I4s", len(blob), b"JSON") + blob + remaining
                    target = root / f"external-{key}.glb"
                    target.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
                    with self.assertRaises(wn.AdmissionError):
                        wn.glb_inventory(target)


if __name__ == "__main__":
    unittest.main()
