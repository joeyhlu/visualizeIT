"""CPU fixtures for independent mask and landmark review states."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_annotations as annotations
from . import quality_evaluate as evaluator
from .vision import cv2


FIXTURE_SELECTION = tuple((100 + index, "uniform") for index in range(40))
FIXTURE_SCORED_IDS = tuple(range(100, 340))
FIXTURE_RESOLUTION = (32, 24)
FIXTURE_SOURCE_HASH = "a" * 64


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def polygon(points):
    return [{"points": points, "hole": False}]


def v2_document():
    return {
        "schema_version": 2,
        "object": "keyboard",
        "resolution": list(FIXTURE_RESOLUTION),
        "selection_status": annotations.REVIEWED_SELECTION_STATUS,
        "operator": "Independent fixture reviewer",
        "source_sha256": FIXTURE_SOURCE_HASH,
        "frames": [{
            "frame_id": frame_id,
            "image": f"{frame_id}.jpg",
            "group": group,
            "status": "pending",
            "visibility": None,
            "visible_object": [],
            "overlapping_hands": [],
            "landmarks": [],
            "landmark_status": "pending",
            "review_notes": "",
        } for frame_id, group in FIXTURE_SELECTION],
    }


def v1_document():
    document = v2_document()
    document["schema_version"] = 1
    for row in document["frames"]:
        row.pop("landmark_status")
    document["selection_status"] = annotations.REVIEWED_SELECTION_STATUS
    for row in document["frames"]:
        row.update(
            status="reviewed", visibility="visible",
            visible_object=polygon(rectangle(3, 3, 8, 9)),
            landmarks=[{
                "landmark_id": "center",
                "object_point_m": [0., 0., 0.],
                "pixel": [16., 12.],
                "visible": True,
                "reviewed": True,
            }],
        )
    return document


def _review_mask(row, visibility="visible"):
    row["status"] = "reviewed"
    row["visibility"] = visibility
    if visibility == "visible":
        row["visible_object"] = polygon(rectangle(3, 3, 8, 9))
        row["landmark_status"] = "pending"
    else:
        row["visible_object"] = []
        row["landmarks"] = []
        row["landmark_status"] = "not_applicable_fully_hidden"


def _review_landmark(row):
    row["landmarks"] = [{
        "landmark_id": "center",
        "object_point_m": [0., 0., 0.],
        "pixel": [16., 12.],
        "visible": True,
        "reviewed": True,
    }]
    row["landmark_status"] = "reviewed"
    row.pop("landmark_review_reason", None)


class AnnotationReviewStateTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            patch.object(annotations, "FROZEN_ANNOTATION_SELECTIONS", {"keyboard": FIXTURE_SELECTION}),
            patch.object(annotations, "FROZEN_SCORED_FRAME_IDS", {"keyboard": FIXTURE_SCORED_IDS}),
            patch.object(annotations, "FROZEN_RESOLUTION", FIXTURE_RESOLUTION),
            patch.object(annotations, "FROZEN_SOURCE_SHA256", {"keyboard": FIXTURE_SOURCE_HASH}),
        ]
        for item in self._patches:
            item.start()
        private_root = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=private_root)
        self.root = Path(self.temp.name)
        self.mask_root = self.root / "masks"
        self.mask_root.mkdir()
        truth = np.zeros((FIXTURE_RESOLUTION[1], FIXTURE_RESOLUTION[0]), np.uint8)
        cv2.fillPoly(truth, [np.asarray(rectangle(3, 3, 8, 9), np.int32)], 255)
        ok, encoded = cv2.imencode(".png", truth)
        if not ok:
            raise AssertionError("Could not encode annotation fixture mask")
        (self.mask_root / "truth.png").write_bytes(encoded.tobytes())
        self.intrinsics = np.array([[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]])

    def tearDown(self):
        self.temp.cleanup()
        for item in reversed(self._patches):
            item.stop()

    def results(self, *, missing_prediction_frames=(), hidden_false_frame=None,
                lost_mask_frames=(), invalid_projection_frames=()):
        missing_prediction_frames = set(missing_prediction_frames)
        lost_mask_frames = set(lost_mask_frames)
        invalid_projection_frames = set(invalid_projection_frames)
        pose = np.eye(4)
        pose[2, 3] = 1.
        invalid_pose = np.eye(4)
        invalid_pose[2, 3] = -1.
        frames = []
        for frame_id in FIXTURE_SCORED_IDS:
            failed = frame_id in missing_prediction_frames
            lost = frame_id in lost_mask_frames
            frames.append({
                "frameId": frame_id,
                "mask_state": "lost" if lost else "available",
                "mask_path": None if lost else "truth.png",
                "pose_state": "failed" if failed else "tracking",
                "cameraFromObject": None if failed else (
                    invalid_pose if frame_id in invalid_projection_frames else pose).tolist(),
                "render_state": "suppressed" if failed or lost else "visible",
            })
        if hidden_false_frame is not None:
            frames[hidden_false_frame - FIXTURE_SCORED_IDS[0]]["render_state"] = "visible"
        return {"automatic": True, "frames": frames}

    def score(self, document, **result_options):
        annotations.validate_annotation_document(document, "keyboard")
        return evaluator.evaluate(
            self.results(**result_options), document, self.mask_root,
            self.intrinsics, list(FIXTURE_SCORED_IDS))

    def test_valid_legacy_v1_is_accepted_and_normalized_without_mutation(self):
        document = v1_document()
        before = json.dumps(document, sort_keys=True)
        self.assertIs(annotations.validate_annotation_document(document, "keyboard"), document)
        normalized = annotations.normalize_annotation_document(document, "keyboard")
        self.assertEqual(normalized["schema_version"], 2)
        self.assertTrue(all(row["landmark_status"] == "reviewed" for row in normalized["frames"]))
        self.assertEqual(json.dumps(document, sort_keys=True), before)
        report = evaluator.evaluate(
            self.results(), document, self.mask_root, self.intrinsics, list(FIXTURE_SCORED_IDS))
        self.assertEqual(report["mask_mean_iou"], 1.)
        self.assertEqual(report["attachment_median_720"], 0.)
        self.assertTrue(report["gates"]["mask"])
        self.assertTrue(report["gates"]["attachment"])

    def test_legacy_v1_visible_review_still_requires_reviewed_landmarks(self):
        document = v1_document()
        document["frames"][0]["landmarks"][0]["reviewed"] = False
        with self.assertRaisesRegex(ValueError, "reviewed visible landmarks"):
            annotations.validate_annotation_document(document, "keyboard")

    def test_v1_fully_hidden_review_normalizes_only_in_memory(self):
        document = v1_document()
        row = document["frames"][0]
        row.update(visibility="fully_hidden", visible_object=[], landmarks=[])
        before = json.dumps(document, sort_keys=True)
        normalized = annotations.normalize_annotation_document(document, "keyboard")
        self.assertEqual(normalized["frames"][0]["landmark_status"], "not_applicable_fully_hidden")
        self.assertEqual(json.dumps(document, sort_keys=True), before)

    def test_mask_review_with_pending_landmarks_passes_only_mask_label_readiness(self):
        document = v2_document()
        for row in document["frames"]:
            _review_mask(row)
        report = self.score(document)
        self.assertTrue(report["mask_labels_ready"])
        self.assertTrue(report["gates"]["mask"])
        self.assertFalse(report["attachment_labels_ready"])
        self.assertIsNone(report["gates"]["attachment"])
        self.assertFalse(report["independent_accuracy_ready"])
        self.assertEqual(report["landmark_review_status_counts"]["pending"], 40)

    def test_unobservable_requires_reason_and_rejects_contradictory_landmarks(self):
        document = v2_document()
        row = document["frames"][0]
        row.update(visibility="visible", landmark_status="unobservable",
                   landmark_review_reason="Visible target has no measurable surface point")
        self.assertIs(annotations.validate_annotation_document(document, "keyboard"), document)
        row.pop("landmark_review_reason")
        with self.assertRaisesRegex(ValueError, "explicit reason"):
            annotations.validate_annotation_document(document, "keyboard")
        row["landmark_review_reason"] = "No measurable point"
        row["landmarks"] = [{
            "landmark_id": "contradiction", "object_point_m": [0., 0., 0.],
            "pixel": [16., 12.], "visible": True, "reviewed": True,
        }]
        with self.assertRaisesRegex(ValueError, "no landmarks"):
            annotations.validate_annotation_document(document, "keyboard")

    def test_reviewed_and_unobservable_landmark_states_require_reviewer_independent_of_mask(self):
        document = v2_document()
        document["selection_status"] = annotations.PENDING_SELECTION_STATUS
        document["operator"] = None
        row = document["frames"][0]
        row["visibility"] = "visible"
        _review_landmark(row)
        self.assertEqual(row["status"], "pending")
        with self.assertRaisesRegex(ValueError, "landmark review requires a reviewer name"):
            annotations.validate_annotation_document(document, "keyboard")

        document["operator"] = "Independent landmark reviewer"
        self.assertIs(annotations.validate_annotation_document(document, "keyboard"), document)

        document["operator"] = None
        row["landmarks"] = []
        row["landmark_status"] = "unobservable"
        row["landmark_review_reason"] = "No independently measurable point"
        with self.assertRaisesRegex(ValueError, "landmark review requires a reviewer name"):
            annotations.validate_annotation_document(document, "keyboard")
        document["operator"] = "Independent landmark reviewer"
        self.assertIs(annotations.validate_annotation_document(document, "keyboard"), document)

    def test_fully_hidden_visibility_cannot_preserve_visible_label_contradictions(self):
        document = v2_document()
        row = document["frames"][0]
        row["visibility"] = "fully_hidden"
        row["visible_object"] = polygon(rectangle(3, 3, 8, 9))
        row["landmarks"] = [{
            "landmark_id": "center", "object_point_m": [0., 0., 0.],
            "pixel": [16., 12.], "visible": True, "reviewed": True,
        }]
        with self.assertRaisesRegex(ValueError, "Fully hidden frame .* cannot contain target polygons or landmarks"):
            annotations.validate_annotation_document(document, "keyboard")

        row["visible_object"] = []
        row["landmarks"] = []
        self.assertIs(annotations.validate_annotation_document(document, "keyboard"), document)

    def test_mixed_landmark_states_keep_attachment_gate_unready(self):
        document = v2_document()
        for index, row in enumerate(document["frames"]):
            if index < 20:
                _review_mask(row)
                _review_landmark(row)
            elif index < 30:
                _review_mask(row)
                row["landmark_status"] = "unobservable"
                row["landmark_review_reason"] = "No independently measurable point"
            elif index < 35:
                _review_mask(row)
            else:
                _review_mask(row, "fully_hidden")
        report = self.score(document)
        self.assertTrue(report["gates"]["mask"])
        self.assertEqual(report["landmark_frame_coverage"], .5)
        self.assertEqual(report["landmark_review_status_counts"], {
            "reviewed": 20, "unobservable": 10, "pending": 5,
            "not_applicable_fully_hidden": 5,
        })
        self.assertFalse(report["attachment_labels_ready"])
        self.assertIsNone(report["gates"]["attachment"])

    def test_landmark_review_can_be_ready_while_mask_review_is_pending(self):
        document = v2_document()
        for row in document["frames"]:
            row.update(visibility="visible", landmark_status="pending")
            _review_landmark(row)
        report = self.score(document)
        self.assertFalse(report["mask_labels_ready"])
        self.assertTrue(report["attachment_labels_ready"])
        self.assertFalse(report["independent_accuracy_ready"])
        self.assertIsNone(report["gates"]["mask"])
        self.assertTrue(report["gates"]["attachment"])

    def test_all_hidden_has_no_visible_mask_or_attachment_pass_and_pending_has_no_score(self):
        hidden = v2_document()
        for row in hidden["frames"]:
            _review_mask(row, "fully_hidden")
        report = self.score(hidden)
        self.assertIsNone(report["gates"]["mask"])
        self.assertIsNone(report["visible_mask_mean_iou"])
        self.assertIsNone(report["gates"]["attachment"])
        self.assertEqual(report["fully_hidden_false_foreground_frames"], 40)
        self.assertFalse(report["gates"]["hidden_false_foreground"])
        self.assertFalse(report["gates"]["hidden_render_leakage"])

        pending = v2_document()
        pending_report = self.score(pending)
        self.assertEqual(pending_report["details"], [])
        self.assertIsNone(pending_report["gates"]["mask"])
        self.assertIsNone(pending_report["gates"]["attachment"])
        self.assertFalse(pending_report["independent_accuracy_ready"])

    def test_complete_landmark_labels_fail_attachment_when_poses_are_missing(self):
        document = v2_document()
        for row in document["frames"]:
            _review_mask(row)
            _review_landmark(row)
        missing = [frame_id for frame_id, _ in FIXTURE_SELECTION]
        report = self.score(document, missing_prediction_frames=missing)
        self.assertTrue(report["attachment_labels_ready"])
        self.assertTrue(report["independent_accuracy_ready"])
        self.assertFalse(report["gates"]["attachment"])
        self.assertEqual(report["visible_landmarks_without_accepted_pose"], 40)
        self.assertEqual(report["attachment_missing_frames"], 40)
        self.assertEqual(report["failures"], 40)
        self.assertEqual(report["frames"], 240)
        self.assertEqual(report["accepted_pose_availability"], 200 / 240)
        self.assertFalse(report["overall_gate_passed"])

    def test_all_invalid_landmark_projections_are_counted_and_fail_attachment(self):
        document = v2_document()
        for row in document["frames"]:
            _review_mask(row)
            _review_landmark(row)
            row["landmarks"].append({
                "landmark_id": "edge", "object_point_m": [0.1, 0., 0.],
                "pixel": [17., 12.], "visible": True, "reviewed": True,
            })
        frame_id = FIXTURE_SELECTION[0][0]
        report = self.score(document, invalid_projection_frames=(frame_id,))
        self.assertEqual(report["visible_landmarks_with_invalid_projection"], 2)
        self.assertEqual(report["attachment_missing_landmarks"], 2)
        self.assertEqual(report["attachment_missing_frames"], 1)
        self.assertFalse(report["gates"]["attachment"])

    def test_lost_visible_mask_and_hidden_false_foreground_leakage_are_separate(self):
        document = v2_document()
        for index, row in enumerate(document["frames"]):
            if index == 1:
                _review_mask(row, "fully_hidden")
            else:
                _review_mask(row)
        report = self.score(document, lost_mask_frames=(100,), hidden_false_frame=101)
        self.assertFalse(report["gates"]["mask"])
        self.assertEqual(report["visible_mask_lost_frame_ids"], [100])
        self.assertEqual(report["visible_mask_empty_prediction_frame_ids"], [100])
        self.assertEqual(report["fully_hidden_false_foreground_frame_ids"], [101])
        self.assertEqual(report["fully_hidden_render_leakage_frame_ids"], [101])
        self.assertFalse(report["gates"]["hidden_false_foreground"])
        self.assertFalse(report["gates"]["hidden_render_leakage"])

    def test_frozen_provenance_and_object_hand_geometry_remain_enforced(self):
        document = v2_document()
        _review_mask(document["frames"][0])
        document["source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "source hash"):
            annotations.validate_annotation_document(document, "keyboard")
        document["source_sha256"] = FIXTURE_SOURCE_HASH
        document["frames"][0]["overlapping_hands"] = polygon(rectangle(3, 3, 8, 9))
        with self.assertRaisesRegex(ValueError, "overlap"):
            annotations.validate_annotation_document(document, "keyboard")


if __name__ == "__main__":
    unittest.main()
