"""Analytical metadata-only checks; no media conversion or model execution."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from .quality_ranker_bop_bundle import bundle_manifest, checked_source, plan_bundle, verify_lossless_video


class BopBundleTests(unittest.TestCase):
    def fixture(self):
        group = {"physical_object_id": "bop-ycbv:000001", "sequence_id": "bop-ycbv:scene-42:object-000001",
                 "scene_id": "bop-ycbv:scene:42", "background_id": "bop-ycbv:scene:42",
                 "background_kind": "observed", "provenance_kind": "real", "mesh_sha256": "a" * 64,
                 "frames": [{"frame_id": 0, "source_frame_id": 10}, {"frame_id": 1, "source_frame_id": 20}]}
        k = [100., 0., 31., 0., 100., 23., 0., 0., 1.]
        obj = {"objectId": 1, "sceneId": 42, "referenceId": "object-one", "model": {"sha256": "a" * 64},
               "onboarding": [{"evaluationPose": "MUST NOT BE CONSUMED"}],
               "frames": [{"frameId": i, "calibration": {"cam_K": k}} for i in (10, 20)]}
        inference = {"schemaVersion": 1, "dataset": "BOP YCB-Video", "objects": [obj]}
        timestamps = {"timestamp_source": "analytic clock fixture, not dataset evidence", "nominal_fps": 10.,
                      "frames": [{"source_frame_id": 10, "timestamp_s": 1.},
                                 {"source_frame_id": 20, "timestamp_s": 2.}]}
        return inference, group, timestamps

    def test_original_ids_preserved_without_leaking_onboarding(self):
        inference, group, timestamps = self.fixture()
        plan = plan_bundle(inference, "object-one", group, timestamps)
        manifest = bundle_manifest(plan, group, resolution=(64, 48), video_sha256="b" * 64, asset_sha256="c" * 64)
        # Decoder IDs are ordinals; original source10/20 and times1/2 survive.
        self.assertEqual(manifest["original_source_frame_ids"], {"0": 10, "1": 20})
        self.assertEqual([r["source_frame_id"] for r in manifest["timeline"]], [0, 1])
        self.assertEqual([r["timestamp_s"] for r in manifest["timeline"]], [1., 2.])
        self.assertNotIn("evaluationPose", json.dumps(manifest))
        from .quality_runner import read_input
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("recording.avi", "object.glb"):
                blob = name.encode()
                (root / name).write_bytes(blob)
                manifest["source_hashes"][name] = hashlib.sha256(blob).hexdigest()
            (root / "input.json").write_text(json.dumps(manifest))
            self.assertEqual(read_input(root)["_timing"]["source_frame_ids"], {0: 0, 1: 1})

    def test_timestamp_identity_and_intrinsics_fail_closed(self):
        inference, group, timestamps = self.fixture()
        for bad in ({}, {**timestamps, "nominal_fps": 0},
                    {**timestamps, "frames": list(reversed(timestamps["frames"]))}):
            with self.assertRaises(ValueError):
                plan_bundle(inference, "object-one", group, bad)
        altered = copy.deepcopy(inference)
        altered["objects"][0]["frames"][1]["calibration"]["cam_K"] = [101., 0., 31., 0., 100., 23., 0., 0., 1.]
        with self.assertRaisesRegex(ValueError, "variable-intrinsics"):
            plan_bundle(altered, "object-one", group, timestamps)
        with self.assertRaisesRegex(ValueError, "mesh SHA"):
            plan_bundle(inference, "object-one", {**group, "mesh_sha256": "d" * 64}, timestamps)

    def test_source_hashes_and_path_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").write_bytes(b"fixture")
            ref = {"path": "image.png", "available": True, "sha256": hashlib.sha256(b"fixture").hexdigest()}
            self.assertEqual(checked_source(root, ref), (root / "image.png").resolve())
            for bad in ({**ref, "sha256": "0" * 64}, {**ref, "available": False}, {**ref, "path": str(root / "image.png")}):
                with self.assertRaises(ValueError):
                    checked_source(root, bad)

    def test_lossless_frame_count_content_and_cleanup(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        for decoded, should_pass in (([(True, source), (False, None)], True),
                                     ([(False, None)], False),
                                     ([(True, source + 1)], False),
                                     ([(True, source), (True, source)], False)):
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.side_effect = decoded
            cv2 = SimpleNamespace(VideoCapture=lambda path: capture,
                                  imread=lambda path, flag: source, IMREAD_COLOR=1)
            if should_pass:
                verify_lossless_video(cv2, "unused.avi", ["source.png"])
            else:
                with self.assertRaises(RuntimeError):
                    verify_lossless_video(cv2, "unused.avi", ["source.png"])
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
