"""Model-free saved-export contract tests using decoded synthetic FFV1 RGB."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_artwork_replay as ar
from bench import workbench_contract as wc
from bench import workbench_offline_export as exporter
from bench import workbench_neural as wn
from tools import test_workbench_offline_export_integration as fixture_module
from tools.test_workbench_offline_export_integration import make_fixture, export_args, cv2, write_json


class TrackingExportTests(unittest.TestCase):
    def setUp(self):
        cache = fixture_module.ROOT / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="tracking-export-", dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rewrite_rows(self, fixture, rows):
        path = fixture["workspace"].runs_dir / fixture["run_id"] / "results.jsonl"
        path.write_bytes(b"\n".join(wc.canonical_json(row) for row in rows) + b"\n")

    def decoded(self, fixture):
        reader = cv2.VideoCapture(str(fixture["output_dir"] / "playback.avi"))
        try:
            output = []
            while True:
                ok, pixels = reader.read()
                if not ok:
                    return output
                output.append(np.ascontiguousarray(pixels[:, :, ::-1]))
        finally:
            reader.release()

    def test_pristine_cache_setup_uses_isolated_root(self):
        root = self.root / "pristine"
        self.assertFalse((root / ".cache").exists())
        case = fixture_module.OfflineVideoIntegrationTests()
        try:
            with patch.object(fixture_module, "ROOT", root), patch.object(fixture_module, "make_fixture", return_value={}):
                case.setUp()
            self.assertTrue((root / ".cache").is_dir())
            self.assertEqual(Path(case.temp.name).parent, root / ".cache")
        finally:
            case.doCleanups()

    def test_all_historical_states_and_mask_validities_export_or_refuse(self):
        # Tracking without a current observed mask is invalid at the saved-row
        # contract boundary, hence refused; every other combination exports RGB.
        for state in wc.POSE_STATES:
            for mask_state in wc.MASK_STATES:
                with self.subTest(state=state, mask_state=mask_state):
                    f = make_fixture(self.root / (state + "-" + mask_state))
                    rows = copy.deepcopy(f["rows"])
                    row = rows[0]
                    row["pose_state"], row["mask_state"] = state, mask_state
                    if state != "tracking":
                        row.update(pose=None, pose_units=None, render_state="suppressed", failure_reason="synthetic " + state)
                    if mask_state != "observed":
                        row.update(mask_frame_id=None, mask_sha256=None)
                    self.rewrite_rows(f, rows)
                    if state == "tracking" and mask_state != "observed":
                        with self.assertRaises((wc.ContractError, ar.Refused)):
                            exporter.export_saved_window(f["workspace"], **export_args(f))
                        self.assertFalse(f["output_dir"].exists())
                        continue
                    result = exporter.export_saved_window(f["workspace"], **export_args(f))
                    record = result["rows"][0]
                    self.assertEqual(record["original"], row)
                    self.assertEqual(record["display_decision"]["accepted_measured"], state == "tracking")
                    self.assertNotEqual(record["display_decision"]["state"], "predicted")
                    if state != "tracking":
                        np.testing.assert_array_equal(self.decoded(f)[0], f["images"]["f2"])
                    else:
                        self.assertFalse(np.array_equal(self.decoded(f)[0], f["images"]["f2"]))

    def test_accounting_preserves_rows_and_denominators(self):
        f = make_fixture(self.root / "accounting")
        result = exporter.export_saved_window(f["workspace"], **export_args(f))
        self.assertEqual([r["original"] for r in result["rows"]], f["rows"])
        self.assertEqual(len(result["rows"][0]["display_decision"]["pose"]), 16)
        self.assertEqual(len(result["rows"][0]["original"]["pose"]), 4)
        counts = result["frame_accounting"]
        self.assertEqual({k: counts[k] for k in ("accepted_measurements", "proposed_predictions", "recovering", "lost", "unavailable", "render_unavailable")},
            dict(accepted_measurements=4, proposed_predictions=0, recovering=1, lost=1, unavailable=0, render_unavailable=4))
        self.assertEqual(sum(counts[k] for k in ("accepted_measurements", "proposed_predictions", "recovering", "lost", "unavailable")), counts["scheduled"])
        self.assertEqual(result["tracking_policy"]["mode"], "legacy_saved_context")
        self.assertFalse(result["tracking_policy"]["prefix_verified"])
        self.assertIn("bench/workbench_tracking_policy.py", result["provenance"]["exporter_sources"])

    def test_experimental_late_window_refuses_without_fabricating_prefix(self):
        f = make_fixture(self.root / "late")
        with self.assertRaisesRegex(ar.Refused, "complete causal prefix"):
            exporter.export_saved_window(f["workspace"], **export_args(f), experimental_tracking=True)
        self.assertFalse(f["output_dir"].exists())

    def test_experimental_full_prefix_process_and_original_loss_rgb(self):
        f = make_fixture(self.root / "prefix", schedule_start=1)
        command = [sys.executable, "-B", "-m", "tools.workbench_offline_export",
            "--workspace", str(f["workspace"].path), "--run-id", f["run_id"], "--role", f["role"],
            "--project", str(f["project_path"]), "--expected-revision", f["expected_revision"],
            "--bundle", str(f["bundle_path"]), "--bundle-sha256", f["bundle_sha256"],
            "--admission-dir", str(f["admission_dir"]), "--output", str(f["output_dir"]),
            "--playback-fps", "10", "--experimental-tracking"]
        done = subprocess.run(command, cwd=fixture_module.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        result = wc.loads_strict((f["output_dir"] / "manifest.json").read_bytes())
        self.assertTrue(result["tracking_policy"]["prefix_verified"])
        self.assertFalse(result["tracking_policy"]["predictions_enabled"])
        self.assertEqual(result["tracking_policy"]["epoch"], 0)
        self.assertEqual([r["original"] for r in result["rows"]], f["rows"])
        pixels = self.decoded(f)
        for index, row in enumerate(f["rows"]):
            if row["pose_state"] != "tracking":
                np.testing.assert_array_equal(pixels[index], f["images"][row["frame_id"]])

    def test_relative_bundle_cannot_be_relabelled_metric(self):
        f = make_fixture(self.root / "relative", schedule_start=1)
        bundle = wc.loads_strict(f["bundle_path"].read_bytes())
        bundle["metric_provenance"] = "relative_reconstruction"
        write_json(f["bundle_path"], bundle)
        f["bundle_sha256"] = fixture_module.sha(f["bundle_path"])
        with self.assertRaisesRegex(wn.AdmissionError, "relative reconstructions"):
            exporter.export_saved_window(f["workspace"], **export_args(f), experimental_tracking=True)
        self.assertFalse(f["output_dir"].exists())

    def test_invalid_anchor_or_chart_refuses_actual_export(self):
        for fault in ("anchor", "chart"):
            with self.subTest(fault=fault):
                f = make_fixture(self.root / fault, schedule_start=1)
                original = exporter._anchor_integrity
                def corrupt(artwork):
                    if fault == "anchor":
                        artwork.marker["barycentric"] = [2., 0., 0.]
                    else:
                        artwork.patch["overlap_counts"][:] = 2
                    return original(artwork)
                with patch.object(exporter, "_anchor_integrity", side_effect=corrupt), self.assertRaisesRegex(ar.Refused, "anchor/chart|overlap-free"):
                    exporter.export_saved_window(f["workspace"], **export_args(f), experimental_tracking=True)
                self.assertFalse(f["output_dir"].exists())

    def test_experimental_epoch_change_refuses_without_publication(self):
        f = make_fixture(self.root / "epoch", schedule_start=1)
        rows = copy.deepcopy(f["rows"])
        rows[1]["evidence"]["epoch"] = 1
        self.rewrite_rows(f, rows)
        with self.assertRaisesRegex(ar.Refused, "epoch"):
            exporter.export_saved_window(f["workspace"], **export_args(f), experimental_tracking=True)
        self.assertFalse(f["output_dir"].exists())


if __name__ == "__main__":
    unittest.main()
