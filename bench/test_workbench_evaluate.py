"""Manufactured independent landmarks establish evaluator arithmetic, not tracking quality."""
import copy
import unittest

from bench import workbench_contract as wc
from bench import workbench_evaluate as evaluator
from bench.test_workbench_contract import clip_document, run_document, frame_result_document


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.clip = wc.ClipSpec.from_dict(clip_document())
        run = run_document()
        run["primary"] = {"artifact_id": "flow-terminal-20261021-final285",
                          "export_sha256": evaluator.HEAD, "export_bytes": 80_165_526}
        self.run = wc.RunSpec.from_dict(run)
        self.rows = []
        for frame in self.clip.frames[2:]:
            row = frame_result_document(frame_id=frame.frame_id, source_frame_index=frame.source_frame_index)
            row.update(timestamp_s=frame.timestamp_s, media_time_s=frame.media_time_s, run_id=self.run.run_id,
                       pose=[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
                       pose_units="metric", backend_identity={"installed_head_sha256": evaluator.HEAD})
            self.rows.append(wc.FrameResult.from_dict(row))
        self.labels = {"schema": "workbench_landmarks_v1", "source_clip_sha256": wc.canonical_sha256(self.clip.to_dict()),
                       "coordinate_units": "metric", "operator": "manufactured test fixture",
                       "resolution": [640, 480], "frames": []}
        for fid, shift in (("f2", 0.), ("f3", 20 / 3)):
            self.labels["frames"].append({"frame_id": fid, "landmark_status": "reviewed", "landmarks": [
                {"landmark_id": "fixed-point", "object_point_m": [0., 0., 1.],
                 "pixel": [319.5, 239.5 + shift], "visible": True, "reviewed": True}]})
        self.evidence = {"stages": {"primary_neural": {"status": "completed"}}}

    def evaluate(self, **kwargs):
        return evaluator.evaluate(self.clip, self.run, kwargs.pop("rows", self.rows),
                                  annotations=kwargs.pop("labels", self.labels),
                                  run_evidence=kwargs.pop("evidence", self.evidence), **kwargs)

    def test_normalized_pixel_errors_and_percentile_are_analytic(self):
        report = self.evaluate()
        # z=1 and x=y=0 project to principal point. A 20/3 native-px shift scales by720/480 to10px.
        self.assertAlmostEqual(report["metrics"]["attachment_median_px_720"], 5.)
        # Linear 95th percentile of [0,10] is9.5.
        self.assertAlmostEqual(report["metrics"]["attachment_p95_px_720"], 9.5)
        self.assertEqual(report["metrics"]["availability_fraction"], 1.)

    def test_partial_window_keeps_its_scope_in_the_receipt(self):
        document = self.run.to_dict()
        document["scheduled_frame_ids"] = document["scheduled_frame_ids"][:1]
        document["mode"] = "focused"
        run = wc.RunSpec.from_dict(document)
        report = evaluator.evaluate(self.clip, run, self.rows[:1], run_evidence=self.evidence)
        self.assertEqual(report["mode"], "focused")
        self.assertEqual(report["scheduled_frames"], 1)
        self.assertEqual(report["clip_scoring_frames"], 2)
        self.assertFalse(report["full_scoring_window_scheduled"])

    def test_render_coverage_is_separate_from_pose_availability(self):
        documents = [row.to_dict() for row in self.rows]
        documents[0]["render_state"] = "rendered"
        documents[1]["render_state"] = "unavailable"
        report = self.evaluate(rows=[wc.FrameResult.from_dict(row) for row in documents])
        self.assertEqual(report["metrics"]["availability_fraction"], 1.)
        self.assertEqual(report["metrics"]["artwork_render_fraction"], .5)
        self.assertEqual(report["render_states"], {"rendered": 1, "unavailable": 1})

    def test_lost_frame_remains_in_availability_and_missing_landmark_counts(self):
        row = self.rows[1].to_dict()
        row.update(pose=None, pose_units=None, pose_state="lost", render_state="suppressed", failure_reason="lost")
        report = self.evaluate(rows=[self.rows[0], wc.FrameResult.from_dict(row)])
        self.assertEqual(report["metrics"]["availability_fraction"], .5)
        self.assertEqual(report["landmarks"]["without_pose"], 1)
        self.assertTrue(report["independent_annotations_complete"])
        self.assertFalse(report["accuracy_gates_passed"])

    def test_unexecuted_run_is_unmeasured_instead_of_zero(self):
        report = self.evaluate(evidence={"stages": {"primary_neural": {"status": "blocked"}}})
        self.assertFalse(report["inference_executed"])
        self.assertEqual(report["schema"], "workbench_evaluation_unavailable_v1")
        self.assertIsNone(report["metrics"]["availability_fraction"])
        self.assertIsNone(report["metrics"]["attachment_p95_px_720"])

    def test_pending_annotations_cannot_pass_accuracy_gates(self):
        labels = copy.deepcopy(self.labels)
        labels["frames"][1]["landmark_status"] = "pending"
        report = self.evaluate(labels=labels)
        self.assertFalse(report["independent_annotations_complete"])
        self.assertIsNone(report["accuracy_gates_passed"])

    def test_model_coordinate_units_cannot_be_relabeled(self):
        labels = copy.deepcopy(self.labels)
        labels["coordinate_units"] = "object_unit"
        with self.assertRaisesRegex(ValueError, "different units"):
            self.evaluate(labels=labels)

    def test_duplicate_results_and_wrong_annotation_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.evaluate(rows=self.rows + [self.rows[0]])
        labels = copy.deepcopy(self.labels)
        labels["source_clip_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "clip hash"):
            self.evaluate(labels=labels)

    def test_wrong_timestamp_cannot_be_scored_as_matching_frame(self):
        row = self.rows[0].to_dict()
        row["timestamp_s"] += 100
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.evaluate(rows=[wc.FrameResult.from_dict(row), self.rows[1]])

    def test_verified_cached_results_retain_measured_status(self):
        report = self.evaluate(evidence={"stages": {"primary_neural": {"status": "cache_hit"}}})
        self.assertTrue(report["inference_executed"])
        self.assertEqual(report["inference_source"], "verified_cache")

    def test_decoder_pts_can_fill_unspecified_media_time(self):
        clip = clip_document(media_times=False)
        self.clip = wc.ClipSpec.from_dict(clip)
        self.labels["source_clip_sha256"] = wc.canonical_sha256(self.clip.to_dict())
        report = self.evaluate()
        self.assertTrue(report["inference_executed"])

    def test_corner_camera_uses_same_effective_K_as_neural_backend(self):
        rows = []
        for row in self.rows:
            doc = row.to_dict()
            doc["backend_identity"]["principal_point_origin"] = "corner"
            doc["backend_identity"]["effective_K"] = [576., 0., 319., 0., 576., 239., 0., 0., 1.]
            rows.append(wc.FrameResult.from_dict(doc))
        labels = copy.deepcopy(self.labels)
        for label in labels["frames"]:
            label["landmarks"][0]["pixel"] = [319., 239.]
        self.assertEqual(self.evaluate(rows=rows, labels=labels)["metrics"]["attachment_p95_px_720"], 0.)


if __name__ == "__main__":
    unittest.main()
