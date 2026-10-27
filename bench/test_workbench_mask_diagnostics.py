"""Analytical and integrity tests; only synthetic metadata and mask PNGs."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_cli as cli
from bench import workbench_contract as wc
from bench import workbench_mask_diagnostics as md
from bench.test_workbench_contract import clip_document
from bench.vision import cv2


class MaskDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        cache = md.ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.ws = cli.Workspace(Path(self.temp.name) / "ws")
        self.ws.init()
        doc = clip_document(clock_kind="frame_index", media_times=False, frame_count=6)
        doc.update(width=8, height=8, onboarding=None, scoring_frame_ids=[f"f{i}" for i in range(6)])
        self.ws.save_clip(wc.ClipSpec.from_dict(doc))
        self.doc = doc
        self.directory = self.ws.masks_dir / "clip-01" / "observed"
        self.directory.mkdir(parents=True)
        self.manifest = dict(schema=cli.MASK_SCHEMA, clip_id="clip-01", mask_set="observed",
                             evaluator_inputs_read=False, masks={})
        self.output = self.ws.derived_dir / "audit"

    def masks(self, areas):
        for i, area in enumerate(areas):
            if area is None:
                continue
            a = np.zeros((8, 8), np.uint8)
            a[:area//4, :4] = 255
            ok, buf = cv2.imencode(".png", a)
            self.assertTrue(ok)
            raw = buf.tobytes()
            name = f"{i}.png"
            (self.directory / name).write_bytes(raw)
            self.manifest["masks"][f"f{i}"] = dict(path=name, sha256=hashlib.sha256(raw).hexdigest(),
                content_sha256=md.mask_sha256(a > 127), empty=not bool(area), width=8, height=8)
        self.save_manifest()

    def save_manifest(self):
        (self.directory / "manifest.json").write_bytes(wc.canonical_json(self.manifest))

    def audit(self, **kwargs):
        return md.audit_saved_masks(self.ws, clip_id="clip-01", mask_set="observed", output_dir=self.output, **kwargs)

    def test_analytical_areas_threshold_and_csv(self):
        self.masks([16, 8, 4, 4, None, 0])
        with patch.object(cli.FrameReader, "read", side_effect=AssertionError("RGB forbidden")):
            report = self.audit()
        rows = report["rows"]
        self.assertEqual([r["area_px"] for r in rows], [16, 8, 4, 4, None, 0])
        self.assertEqual(rows[1]["temporal_iou"], .5)
        self.assertFalse(rows[1]["area_drop"])
        self.assertEqual(rows[2]["baseline_area_ratio"], .25)
        self.assertEqual(rows[0]["bbox_xyxy"], [0, 0, 4, 4])
        self.assertEqual(rows[0]["centroid_xy"], [1.5, 1.5])
        self.assertEqual([e["frame_ids"] for e in report["episodes"]], [["f2", "f3"], ["f5"]])
        self.assertTrue(all(e["timestamp_span_s"] is None for e in report["episodes"]))
        with (self.output / "report.csv").open(newline="") as stream:
            self.assertEqual([r["frame_id"] for r in csv.DictReader(stream)], report["scheduled_frame_ids"])

    def test_empty_prefix_has_no_future_baseline(self):
        self.masks([0, 0, 16, 0, 0, 4])
        r = self.audit()["rows"]
        self.assertIsNone(r[0]["baseline_frame_id"])
        self.assertIsNone(r[1]["temporal_iou"])
        self.assertEqual(r[1]["pair_status"], "both_empty")
        self.assertEqual(r[2]["pair_status"], "previous_empty")
        self.assertIsNone(r[2]["area_ratio"])
        self.assertEqual(r[3]["area_ratio"], 0)
        self.assertIsNone(r[4]["temporal_iou"])
        self.assertEqual(r[5]["baseline_frame_id"], "f2")

    def test_all_missing(self):
        self.masks([None]*6)
        report = self.audit()
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["frame_accounting"]["unavailable"], 6)
        self.assertIsNone(report["independent_accuracy"])

    def test_gap_closes_episode_and_pairs(self):
        self.masks([16, 4, 4, 4, 4, 4])
        for f in self.doc["frames"][2:]:
            f["source_frame_index"] += 2
        self.ws.clip_path("clip-01").write_bytes(wc.canonical_json(self.doc))
        report = self.audit()
        self.assertIsNone(report["rows"][2]["temporal_iou"])
        self.assertEqual([e["frame_count"] for e in report["episodes"]], [1, 4])

    def test_semantic_and_file_tampering_refused(self):
        self.masks([16]*6)
        self.manifest["masks"]["f0"]["empty"] = True
        self.save_manifest()
        with self.assertRaisesRegex(cli.WorkbenchCLIError, "empty"):
            self.audit()
        self.manifest["masks"]["f0"]["empty"] = False
        self.save_manifest()
        (self.directory / "0.png").write_bytes(b"bad PNG")
        with self.assertRaises(cli.WorkbenchCLIError):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_mutation_cleanup_and_budget(self):
        self.masks([16]*6)
        original = self.ws.require_budget
        count = 0
        def mutate(**kwargs):
            nonlocal count
            count += 1
            original(**kwargs)
            if count == 2:
                self.ws.clip_path("clip-01").write_bytes(b"changed")
        with patch.object(self.ws, "require_budget", side_effect=mutate):
            with self.assertRaisesRegex(cli.WorkbenchCLIError, "closure"):
                self.audit()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.ws.derived_dir.iterdir()), [])

    def test_no_overwrite_and_invalid_policy(self):
        self.masks([16]*6)
        with self.assertRaises(cli.WorkbenchCLIError):
            self.audit(area_ratio_min=1)
        self.audit()
        before = (self.output / "report.json").read_bytes()
        with self.assertRaises(cli.WorkbenchCLIError):
            self.audit()
        self.assertEqual((self.output / "report.json").read_bytes(), before)

    def test_translation_native_and_px720(self):
        self.masks([16]*6)
        a = np.zeros((8, 8), np.uint8)
        a[:4, 2:6] = 255
        ok, buf = cv2.imencode(".png", a)
        self.assertTrue(ok)
        raw = buf.tobytes()
        (self.directory / "1.png").write_bytes(raw)
        self.manifest["masks"]["f1"].update(sha256=hashlib.sha256(raw).hexdigest(),
                                              content_sha256=md.mask_sha256(a > 127))
        self.save_manifest()
        row = self.audit()["rows"][1]
        self.assertAlmostEqual(row["temporal_iou"], 1/3)
        self.assertEqual(row["centroid_distance_px"], 2)
        self.assertEqual(row["centroid_distance_px720"], 180)

    def test_real_clock_episode_span(self):
        self.masks([16, 4, 4, None, 4, 16])
        self.doc.update(clock_kind="timestamp_seconds", clock_source="synthetic measured clock")
        for i, frame in enumerate(self.doc["frames"]):
            frame["timestamp_s"] = i * .25
        self.ws.clip_path("clip-01").write_bytes(wc.canonical_json(self.doc))
        self.assertEqual([e["timestamp_span_s"] for e in self.audit()["episodes"]], [.25, 0])

    def test_source_closure_and_budget_cleanup(self):
        self.masks([16]*6)
        with patch.object(md, "_source_hashes", side_effect=[{"source": "before"}, {"source": "after"}]):
            with self.assertRaisesRegex(cli.WorkbenchCLIError, "closure"):
                self.audit()
        self.assertEqual(list(self.ws.derived_dir.iterdir()), [])
        with patch.object(self.ws, "require_budget", side_effect=cli.WorkbenchCLIError("budget")):
            with self.assertRaisesRegex(cli.WorkbenchCLIError, "budget"):
                self.audit()
        self.assertEqual(list(self.ws.derived_dir.iterdir()), [])

    def test_publish_race_preserves_competing_output(self):
        self.masks([16]*6)
        publish = md._publish_new_directory
        def race(staging, output):
            output.mkdir()
            (output / "competitor").write_bytes(b"keep")
            return publish(staging, output)
        with patch.object(md, "_publish_new_directory", side_effect=race):
            with self.assertRaises(OSError):
                self.audit()
        self.assertEqual((self.output / "competitor").read_bytes(), b"keep")
        self.assertEqual(list(self.ws.derived_dir.iterdir()), [self.output])

    def test_header_limits_precede_decoder_and_symlink_refusal(self):
        import struct
        self.masks([16]*6)
        path = self.directory / "0.png"
        raw = bytearray(path.read_bytes())
        raw[16:24] = struct.pack(">II", 100000, 100000)
        path.write_bytes(raw)
        self.manifest["masks"]["f0"]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.save_manifest()
        with patch.object(cv2, "imdecode", side_effect=AssertionError("header must refuse first")):
            with self.assertRaisesRegex(cli.WorkbenchCLIError, "header"):
                self.audit()
        path.unlink()
        try:
            path.symlink_to(self.directory / "1.png")
        except OSError:
            self.skipTest("symlink creation unavailable on this Windows account")
        with self.assertRaisesRegex(cli.WorkbenchCLIError, "symlink"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
