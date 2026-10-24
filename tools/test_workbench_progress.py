"""Evidence tests: saved correspondence data must never become video accuracy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import workbench_progress as progress


class ProgressEvidenceTests(unittest.TestCase):
    def video_fixture(self, **changes):
        report = {"schema": "workbench_evaluation_v1", "installed_head_sha256": progress.HEAD_SHA256,
                  "inference_executed": True, "clip_id": "fixture", "mode": "regression",
                  "scheduled_frames": 10, "clip_scoring_frames": 10, "failed_frames": 1,
                  "full_scoring_window_scheduled": True, "independent_annotations_complete": False,
                  "metrics": {"availability_fraction": .9, "attachment_median_px_720": None,
                              "attachment_p95_px_720": None, "wrong_surface_events": None,
                              "correct_recovery_fraction": None, "peak_memory_bytes": 1024,
                              "latency_p95_ms": 123.5, "artwork_render_fraction": .8}}
        report.update(changes)
        return report

    def collect_video_fixture(self, directory, report):
        path = Path(directory) / "video-report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return self.collect_fixture(directory, reports=[path])

    def fixture(self):
        # Analytic paired example: 3 - 4 = -1; it is a parser fixture, not a model result.
        row = {"id": "frame", "baseline_epe_px": 4., "adapted_epe_px": 3., "delta_epe_px": -1.}
        development = {
            "export_sha256": progress.HEAD_SHA256,
            "baseline_epe_px": 4., "adapted_epe_px": 3.,
            "baseline_pooled": {}, "adapted_pooled": {}, "confidence_parity": {},
            "optimizer_updates": 1, "validation_available": 1, "validation_total": 2,
            "validation_objects": 1, "train_rows": 2, "train_objects": 1,
            "per_frame": [row],
        }
        return {"development-comparison.json": development}

    def collect_fixture(self, directory, fixture=None, reports=None):
        path = Path(directory) / "records.json"
        raw = json.dumps(fixture or self.fixture()).encode()
        path.write_bytes(raw)
        with patch.object(progress, "RECORDS_SHA256", hashlib.sha256(raw).hexdigest()):
            return progress.collect(path, evaluation_reports=reports)

    def test_missing_tracking_metrics_are_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.collect_fixture(directory)
        self.assertEqual(snapshot["video_evaluations"], [])
        self.assertEqual(snapshot["phone_validation"], "unmeasured")
        self.assertFalse(snapshot["quality_claim"])

    def test_pinned_receipt_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verified receipt"):
                progress.collect(path)

    def test_other_weights_not_charted_as_current(self):
        fixture = self.fixture()
        fixture["development-comparison.json"]["export_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                self.collect_fixture(directory, fixture)

    def test_inconsistent_pair_rejected(self):
        fixture = self.fixture()
        fixture["development-comparison.json"]["per_frame"][0]["delta_epe_px"] = 100.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                self.collect_fixture(directory, fixture)

    def test_blocked_replay_not_measured_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"schema": "workbench_evaluation_v1",
                "installed_head_sha256": progress.HEAD_SHA256,
                "inference_executed": False, "metrics": {"availability_fraction": 0}}))
            with self.assertRaisesRegex(ValueError, "unexecuted"):
                self.collect_fixture(directory, reports=[report])

    def test_smoke_scope_survives_graph_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"schema": "workbench_evaluation_v1",
                "installed_head_sha256": progress.HEAD_SHA256, "inference_executed": True,
                "clip_id": "fixture", "mode": "smoke", "scheduled_frames": 10,
                "clip_scoring_frames": 240, "full_scoring_window_scheduled": False,
                "metrics": {"availability_fraction": .9}}))
            row = self.collect_fixture(directory, reports=[report])["video_evaluations"][0]
            self.assertEqual(row["mode"], "smoke")
            self.assertEqual(row["clip_scoring_frames"], 240)
            self.assertFalse(row["full_scoring_window_scheduled"])

    def test_duplicate_and_nonfinite_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
                report.write_text(raw)
                with self.assertRaises(ValueError):
                    progress._read(report)

    def test_changed_sources_make_old_checks_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            records = Path(directory) / "records.json"
            raw = json.dumps(self.fixture()).encode()
            records.write_bytes(raw)
            checks = Path(directory) / "checks.json"
            checks.write_text(json.dumps({"schema": "workbench_software_checks_v1",
                "passed": 1, "failed": 0, "skipped": 0,
                "source_files": {"tools/workbench_progress.py": "0" * 64}}))
            with patch.object(progress, "RECORDS_SHA256", hashlib.sha256(raw).hexdigest()):
                snapshot = progress.collect(records, checks_path=checks)
            self.assertEqual(snapshot["software_checks"]["stale_source_files"],
                             ["tools/workbench_progress.py"])

    def test_actual_saved_evidence_has_expected_pairing(self):
        snapshot = progress.collect(progress.ROOT / ".cache/cloud-prepare/flow-terminal-small-records-20261021.json")
        # Reference: actual pinned saved receipt, not generated predictions.
        rows = snapshot["paired_development_rows"]
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["delta_epe_px"] < 0 for row in rows), 15)
        self.assertEqual(sum(row["delta_epe_px"] > 0 for row in rows), 5)

    def test_all_requested_video_metrics_keep_measured_and_unknown_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.video_fixture()
            row = self.collect_video_fixture(directory, report)["video_evaluations"][0]
        self.assertEqual(row["metrics"], report["metrics"])
        self.assertEqual(row["failed_frames"], 1)
        self.assertEqual(row["scheduled_frames"], 10)
        self.assertFalse(row["independent_annotations_complete"])

    def test_event_counts_reject_fractional_and_boolean_values(self):
        for value in (True, False, .5, -1, "0"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                report = self.video_fixture(metrics={"wrong_surface_events": value})
                with self.assertRaises(ValueError):
                    self.collect_video_fixture(directory, report)

    def test_video_metric_domains_reject_nonfinite_and_invalid_fractions(self):
        cases = [("availability_fraction", 1.01), ("correct_recovery_fraction", -.1),
                 ("artwork_render_fraction", True), ("peak_memory_bytes", -1),
                 ("latency_p95_ms", float("inf")), ("attachment_p95_px_720", float("nan"))]
        for metric, value in cases:
            with self.subTest(metric=metric, value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.collect_video_fixture(directory, self.video_fixture(metrics={metric: value}))

    def test_invalid_frame_counts_cannot_supply_complete_coverage(self):
        cases = [{"scheduled_frames": True}, {"scheduled_frames": -1}, {"scheduled_frames": 10.5},
                 {"scheduled_frames": 11}, {"clip_scoring_frames": False},
                 {"failed_frames": 11}, {"failed_frames": -1}, {"failed_frames": True},
                 {"scheduled_frames": 9, "full_scoring_window_scheduled": True}]
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.collect_video_fixture(directory, self.video_fixture(**changes))

    def test_duplicate_clip_report_cannot_inflate_suite_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            for path in (first, second):
                path.write_text(json.dumps(self.video_fixture()), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.collect_fixture(directory, reports=[first, second])

    def test_unknown_frame_counts_remain_unknown(self):
        report = self.video_fixture(scheduled_frames=None, clip_scoring_frames=None,
                                    failed_frames=None, full_scoring_window_scheduled=False)
        with tempfile.TemporaryDirectory() as directory:
            row = self.collect_video_fixture(directory, report)["video_evaluations"][0]
        self.assertIsNone(row["scheduled_frames"])
        self.assertIsNone(row["clip_scoring_frames"])
        self.assertIsNone(row["failed_frames"])

    def test_measured_zero_events_are_distinct_from_unmeasured_events(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.collect_video_fixture(directory, self.video_fixture(metrics={"wrong_surface_events": 0}))["video_evaluations"][0]
        self.assertEqual(row["metrics"]["wrong_surface_events"], 0)

    def test_scheduled_full_window_with_missing_rows_is_not_completed(self):
        for result_rows in (None, 0, 9):
            with self.subTest(result_rows=result_rows):
                coverage = progress.validate_counts(self.video_fixture(result_rows=result_rows))
                self.assertNotEqual(coverage['coverage_status'], 'complete')

    def test_complete_coverage_requires_accounting_for_failed_frames_too(self):
        coverage = progress.validate_counts(self.video_fixture(result_rows=10))
        self.assertEqual(coverage['coverage_status'], 'complete')
        self.assertEqual(coverage['completed_window_fraction'], 1)
        self.assertEqual(coverage['failed_frames'], 1)

    def test_empty_window_is_not_completed_suite_coverage(self):
        coverage = progress.validate_counts(self.video_fixture(scheduled_frames=0, clip_scoring_frames=0,
                                                               result_rows=0, failed_frames=0))
        self.assertNotEqual(coverage['coverage_status'], 'complete')
        self.assertIsNone(coverage['completed_window_fraction'])

    def test_missing_schedule_does_not_disable_known_total_bounds(self):
        for changes in ({'result_rows': 11}, {'failed_frames': 11}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    progress.validate_counts(self.video_fixture(scheduled_frames=None,
                        full_scoring_window_scheduled=False, **changes))

    def test_output_cannot_escape_through_a_resolved_directory_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'project'
            external = Path(directory) / 'external'
            project.mkdir(); external.mkdir()
            alias = project / 'alias'
            try:
                alias.symlink_to(external, target_is_directory=True)
            except OSError as error:
                self.skipTest(f'Windows symlink creation unavailable: {error}')
            with patch.object(progress, 'ROOT', project.resolve()):
                with self.assertRaises(ValueError):
                    progress._output_dir(alias / 'new-results')
            self.assertEqual(list(external.iterdir()), [])

    def test_csv_separates_development_and_video_metrics_without_filling_unknowns(self):
        import csv
        import io
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.collect_video_fixture(directory, self.video_fixture(result_rows=10))
        rows = list(csv.DictReader(io.StringIO(progress.csv_text(snapshot))))
        development = [row for row in rows if row['row_kind'] == progress.ROW_KIND_DEVELOPMENT]
        video = [row for row in rows if row['row_kind'] == progress.ROW_KIND_VIDEO]
        self.assertEqual(len(development), 1)
        self.assertEqual(len(video), 1)
        for metric in progress.TRACKING_METRICS:
            self.assertEqual(development[0][metric], '')
        self.assertEqual(video[0]['attachment_median_px_720'], '')
        self.assertEqual(video[0]['wrong_surface_events'], '')
        self.assertEqual(video[0]['availability_fraction'], '0.9')
        self.assertEqual(video[0]['peak_memory_bytes'], '1024')
        self.assertEqual(video[0]['failed_frames'], '1')

    def test_recorded_comparison_status_does_not_imply_a_live_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.collect_video_fixture(directory, self.video_fixture(comparison_status='baseline pending'))
        lines = progress.status_lines(snapshot)
        self.assertTrue(any('baseline pending' in line for line in lines))
        self.assertTrue(any('no liveness claim' in line for line in lines))
        self.assertFalse(any('still running' in line for line in lines))


if __name__ == "__main__":
    unittest.main()
