"""Manufactured FFV1 inspection checks; these are not tracking evaluations."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from bench import workbench_export_review as review
from bench import workbench_contract as wc
from bench.workbench_offline_export import export_saved_window
from tools.test_workbench_offline_export_integration import make_fixture, export_args

ROOT = Path(__file__).resolve().parents[1]


class ExportReviewTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.addCleanup(self.temp.cleanup)
        self.fixture = make_fixture(Path(self.temp.name))
        self.manifest = export_saved_window(self.fixture["workspace"], **export_args(self.fixture))
        self.export = self.fixture["output_dir"]
        self.output = self.fixture["workspace"].derived_dir / "review"
        self.manifest_path = self.export / "manifest.json"

    def invoke(self, **changes):
        args = dict(export_dir=self.export, expected_manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                    frame_ids=["f7", "f3", "f6"], output_dir=self.output, tile_width=128)
        args.update(changes)
        return review.review_saved_export(self.fixture["workspace"], **args)

    def test_exact_source_mask_composite_and_state(self):
        report = self.invoke()
        self.assertEqual(report["selected_frame_ids"], ["f3", "f6", "f7"])
        sheet = np.array(Image.open(self.output / "review.png"))
        for row in report["rows"]:
            fid = row["frame_id"]
            x, y, w, h = row["panels"]["source"]
            np.testing.assert_array_equal(sheet[y:y+h, x:x+w], self.fixture["images"][fid])
            x, y, w, h = row["panels"]["mask"]
            expected = np.repeat((self.fixture["masks"][fid].astype(np.uint8)*255)[:, :, None], 3, axis=2)
            np.testing.assert_array_equal(sheet[y:y+h, x:x+w], expected)
        self.assertEqual(report["rows"][0]["pose_state"], "lost")
        self.assertEqual(report["rows"][1]["mask_state"], "empty")
        self.assertIs(report["quality_claim"], False)
        self.assertIsNone(report["independent_accuracy"])

    def test_invalid_requests_and_existing_output(self):
        for change in ({"frame_ids": []}, {"frame_ids": ["f3", "f3"]}, {"frame_ids": ["f0"]},
                       {"tile_width": True}, {"tile_width": 63}, {"expected_manifest_sha256": "A"*64}):
            with self.subTest(change=change), self.assertRaises(Exception):
                self.invoke(**change)
            self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(Exception):
            self.invoke()

    def test_video_tamper_and_png_budget(self):
        with patch.object(review, "MAX_PNG_BYTES", 1), self.assertRaises(Exception):
            self.invoke()
        self.assertFalse(self.output.exists())
        video = self.export / "playback.avi"
        with video.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(Exception):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_manifest_order_and_nonfinite_refused(self):
        original = self.manifest_path.read_bytes()
        for alter in (lambda m: m["rows"][0].update(output_index=True),
                      lambda m: m["rows"][0]["original"].update(timestamp_s=float("nan"))):
            m = copy.deepcopy(self.manifest)
            alter(m)
            # JSON writer intentionally permits NaN to check the strict reader.
            import json
            self.manifest_path.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaises(Exception):
                self.invoke()
            self.assertFalse(self.output.exists())
        self.manifest_path.write_bytes(original)

    def test_change_during_publication_refused(self):
        original_sources = review._sources
        calls = []
        def changed():
            result = original_sources()
            calls.append(1)
            if len(calls) > 1:
                result["bench/workbench_export_review.py"] = "0"*64
            return result
        with patch.object(review, "_sources", changed), self.assertRaises(Exception):
            self.invoke()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob(".export-review-*")))


if __name__ == "__main__":
    unittest.main()
