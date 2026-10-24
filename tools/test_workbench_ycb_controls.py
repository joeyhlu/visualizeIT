"""Independent numerical and source-admission tests for prepared CAD controls."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tools import workbench_ycb_controls as controls


class YcbControlTests(unittest.TestCase):
    def test_manifest_mismatch_is_rejected_before_interpreting_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(b"{}")
            with self.assertRaisesRegex(controls.ControlError, "SHA-256|hash"):
                controls.load_manifest(path)

    def test_millimetres_become_metres_once_and_face_order_is_preserved(self):
        mesh = {"positions_mm": np.array([[0., 0., 0.], [1000., 0., 0.], [0., 1000., 0.], [0., 0., 1000.]]),
                "triangles": np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.uint32),
                "normals": None, "uv": None, "colors": None}
        glb, summary = controls.build_glb(mesh, name="manufactured", texture=None)
        _, arrays, image = controls.read_glb(glb)
        np.testing.assert_array_equal(arrays["POSITION"], mesh["positions_mm"] / 1000.)
        np.testing.assert_array_equal(arrays["indices"].reshape(-1, 3), mesh["triangles"])
        self.assertIsNone(image)
        self.assertEqual(summary["appearance"], "unsupported")
        self.assertAlmostEqual(controls.verify_roundtrip(glb, mesh, None)["signed_volume_m3"], 1 / 6)

    def test_seed_uses_final_onboarding_source_id_and_metric_translation(self):
        pose = {"obj_id": 2, "cam_R_m2c": np.eye(3).reshape(-1).tolist(), "cam_t_m2c": [100., 200., 1000.]}
        obj = {"onboarding": [{"logicalIndex": i, "frameId": 361 + i, "evaluationPose": pose} for i in range(10)],
               "frames": [{"frameId": 371}]}
        seed = controls.onboarding_seed(obj, 2)
        self.assertEqual(seed["frame_id"], 370)
        np.testing.assert_array_equal(np.array(seed["pose_m"])[:3, 3], [.1, .2, 1.])
        obj["frames"] = [{"frameId": 370}]
        with self.assertRaisesRegex(controls.ControlError, "scored"):
            controls.onboarding_seed(obj, 2)

    def test_matching_texture_filename_cannot_replace_official_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obj_000002.png"
            path.write_bytes(b"same name, wrong contents")
            with self.assertRaisesRegex(controls.ControlError, "trusted pin"):
                controls.admit_texture(2, {"uv": np.zeros((3, 2)), "texture_files": [path.name]},
                                       {"datasetRevision": controls.DATASET_REVISION}, path)

    def test_full_decode_rejects_header_only_even_when_identity_fixture_matches(self):
        stream = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
        truncated = stream.getvalue()[:33]
        trusted = {**dict(controls.TRUSTED_TEXTURES[2]), "sha256": hashlib.sha256(truncated).hexdigest(),
                   "bytes": len(truncated), "dimensions": (3, 2)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obj_000002.png"
            path.write_bytes(truncated)
            with patch.object(controls, "TRUSTED_TEXTURES", {2: trusted}):
                with self.assertRaisesRegex(controls.ControlError, "full decoding"):
                    controls.admit_texture(2, {"uv": np.zeros((3, 2)), "texture_files": [path.name]},
                                           {"datasetRevision": controls.DATASET_REVISION}, path)

    def test_trusted_texture_bindings_are_immutable(self):
        with self.assertRaises(TypeError):
            controls.TRUSTED_TEXTURES[2] = {}
        with self.assertRaises(TypeError):
            controls.TRUSTED_TEXTURES[2]["sha256"] = "0" * 64

    def test_actual_authenticated_texture_still_cannot_publish_unverified_uv_bundle(self):
        texture = controls.ROOT / ".cache/video-workbench/imports/ycb-textures-v1/obj_000002.png"
        if not texture.is_file() or not controls.DEFAULT_MANIFEST.is_file():
            self.skipTest("optional pinned local CAD/texture fixture unavailable")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "candidate"
            with patch.object(controls, "onboarding_seed", side_effect=AssertionError("no seed authorized")):
                result = controls.export_control(controls.DEFAULT_MANIFEST, 2, target, "integer_centers", False, texture)
            self.assertEqual(result["status"], "blocked_unverified_uv")
            self.assertFalse((target / "metric-bundle.json").exists())
            self.assertTrue((target / "candidate-model.glb").is_file())


if __name__ == "__main__":
    unittest.main()
