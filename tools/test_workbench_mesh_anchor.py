"""Reviewed manual seed-click controls; manufactured metric plane, no inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bench import workbench_cli as cli
from bench import workbench_offline_export as ex
from tools.test_workbench_offline_export_integration import make_fixture, export_args, sha

ROOT = Path(__file__).resolve().parents[1]


class MeshAnchorTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.addCleanup(self.tmp.cleanup)
        self.f = make_fixture(Path(self.tmp.name))
        self.path = Path(self.tmp.name) / "anchor.json"
        self.request = dict(schema="workbench_mesh_anchor_v1", clip_id="synthetic-offline-clip",
            frame_id="f0", pixel=[80., 40.], reviewed=True, reviewed_by="synthetic reviewer")
        self.write(self.request)

    def write(self, request):
        self.path.write_text(json.dumps(request), encoding="utf-8")

    def export(self, **kwargs):
        return ex.export_saved_window(self.f["workspace"], **{**export_args(self.f),
            "surface_mode": "mesh", "mesh_anchor_path": self.path, **kwargs})

    def assert_clean(self):
        self.assertFalse(self.f["output_dir"].exists())
        self.assertFalse(list(self.f["workspace"].derived_dir.glob(".offline-export-*")))

    def test_plane_projection_decoded_video_and_original_inputs(self):
        root = self.f["workspace"].path
        originals = {p: sha(p) for p in root.rglob("*") if p.is_file()}
        manifest = self.export()
        revision = manifest["mesh_anchor"]
        point = np.asarray(revision["anchor_object_xyz_m"])
        # Native integer-centre calibration is unchanged (principal point 63.5,47.5).
        np.testing.assert_allclose(point, [(80-63.5)/200, (40-47.5)/200, 0], atol=1e-8)
        self.assertEqual(revision["request_sha256"], sha(self.path))
        self.assertNotEqual(revision["marker_sha256"], revision["original_anchor_sha256"])
        self.assertEqual(manifest["provenance"]["original_artwork_record_sha256"],
                         revision["original_artwork_record_sha256"])
        self.assertIn("f0", revision["seed"]["source_frames"])
        self.assertNotIn("f0", manifest["provenance"]["source_frames"])
        cap = cv2.VideoCapture(str(self.f["output_dir"] / "playback.avi"))
        try:
            decoded = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                decoded.append(np.ascontiguousarray(bgr[:, :, ::-1]))
        finally:
            cap.release()
        self.assertEqual(len(decoded), 6)
        for i in (1, 2):  # lost and recovering never show the manual anchor
            np.testing.assert_array_equal(decoded[i], self.f["images"]["f" + str(i+2)])
        changed = np.any(decoded[0] != self.f["images"]["f2"], axis=2)
        y, x = np.where(changed)
        self.assertGreater(len(x), 0)
        self.assertLess(abs(float(x.mean()) - 80), 8)
        self.assertLess(abs(float(y.mean()) - 40), 8)
        for p, digest in originals.items():
            self.assertEqual(sha(p), digest, str(p))

    def test_strict_requests(self):
        variants = [dict(extra=1), dict(schema="future"), dict(clip_id="wrong"),
            dict(frame_id="f2"), dict(reviewed=False), dict(reviewed=1), dict(reviewed_by="  "),
            dict(pixel=[True, 40]), dict(pixel=[float("nan"), 40]), dict(pixel=[128, 40]),
            dict(pixel=[80]), dict(pixel="80,40")]
        for change in variants:
            with self.subTest(change=change):
                self.write({**self.request, **change})
                with self.assertRaises(Exception):
                    self.export()
                self.assert_clean()
        self.path.write_bytes(b" " * ((64 << 10) + 1))
        with self.assertRaisesRegex(Exception, "bound"):
            self.export()
        self.assert_clean()

    def test_duplicate_keys_and_directory_refused(self):
        self.path.write_text(json.dumps(self.request)[:-1] + ',"reviewed":true}')
        with self.assertRaisesRegex(Exception, "duplicate"):
            self.export()
        with self.assertRaisesRegex(Exception, "regular"):
            self.export(mesh_anchor_path=self.path.parent)
        self.assert_clean()

    def test_mode_and_offset_guards(self):
        for kwargs in (dict(surface_mode="planar"), dict(experimental_tracking=True),
                       dict(placement={**ex.DEFAULT_PLACEMENT, "offset_t_m": .01})):
            with self.subTest(kwargs=kwargs), self.assertRaises(Exception):
                self.export(**kwargs)
        self.assert_clean()

    def test_mask_miss_and_no_mesh_hit(self):
        for pixel, reason in (([0, 0], "mask"), ([110, 47.5], "hit")):
            self.write({**self.request, "pixel": pixel})
            with self.subTest(pixel=pixel), self.assertRaisesRegex(Exception, reason):
                self.export()
            self.assert_clean()

    def test_request_mutation_before_publication(self):
        original = ex.compose_mesh
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.write({**self.request, "reviewed_by": "changed"})
            return result
        with patch.object(ex, "compose_mesh", mutate), self.assertRaisesRegex(Exception, "changed"):
            self.export()
        self.assert_clean()

    def test_seed_pixel_mutation_and_reader_cleanup(self):
        original = cli.FrameReader.read
        closes = []
        original_close = cli.FrameReader.close
        seed_reads = 0
        def read(reader, frame):
            nonlocal seed_reads
            image, identity = original(reader, frame)
            if frame.frame_id == "f0":
                seed_reads += 1
                if seed_reads > 1:
                    identity = {**identity, "image_sha256": "0" * 64}
            return image, identity
        def close(reader):
            closes.append(reader)
            original_close(reader)
        with patch.object(cli.FrameReader, "read", read), patch.object(cli.FrameReader, "close", close), \
                self.assertRaisesRegex(Exception, "changed"):
            self.export()
        self.assertEqual(seed_reads, 2)
        self.assertGreaterEqual(len(closes), 4)
        self.assert_clean()

    def test_exclusive_publication_preserves_existing_directory(self):
        original = ex._publish_new_directory
        def occupied(staging, output):
            output.mkdir()
            (output / "owner.txt").write_text("other writer")
            return original(staging, output)
        with patch.object(ex, "_publish_new_directory", occupied), self.assertRaises(OSError):
            self.export()
        self.assertEqual((self.f["output_dir"] / "owner.txt").read_text(), "other writer")
        self.assertFalse(list(self.f["workspace"].derived_dir.glob(".offline-export-*")))

    def test_cli_argument_forwarding(self):
        from tools.workbench_offline_export import main
        f = self.f
        argv = []
        for key in ("project", "bundle", "admission-dir", "output", "run-id", "role",
                    "expected-revision", "bundle-sha256", "playback-fps"):
            name = {"project": "project_path", "bundle": "bundle_path", "output": "output_dir"}.get(key, key.replace("-", "_"))
            argv += ["--" + key, str(f[name])]
        argv += ["--workspace", str(f["workspace"].path), "--surface-mode", "mesh", "--mesh-anchor", str(self.path)]
        self.assertEqual(main(argv), 0)
        manifest = json.loads((f["output_dir"] / "manifest.json").read_text())
        self.assertEqual(manifest["mesh_anchor"]["request"], self.request)


if __name__ == "__main__":
    unittest.main()
