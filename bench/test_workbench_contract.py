"""Independent boundary tests for the workbench's public JSON contracts.

These fixtures use synthetic metadata only. They never open media, weights, or
evaluator annotations.
"""
from __future__ import annotations

import copy
import unittest

from bench import workbench_contract as wc


HASH_A = "a" * 64
HASH_B = "b" * 64


def clip_document(*, clock_kind="timestamp_seconds", start_index=1148,
                  prefix_step=0.1, media_times=True, frame_count=4):
    frames = []
    for ordinal in range(frame_count):
        frames.append({
            "frame_id": f"f{ordinal}",
            "source_frame_index": start_index + ordinal,
            "timestamp_s": ordinal * prefix_step if clock_kind == "timestamp_seconds" else None,
            "media_time_s": ordinal * prefix_step if media_times else None,
        })
    return {
        "schema": wc.CLIP_SCHEMA,
        "clip_id": "clip-01",
        "title": "Synthetic sequence",
        "source_kind": "video_file",
        "clock_kind": clock_kind,
        "clock_source": "fixture presentation timestamps" if clock_kind == "timestamp_seconds" else None,
        "width": 640,
        "height": 480,
        "media": {"root_id": "workspace", "path": "source/video.mp4", "sha256": HASH_A, "bytes": 4096},
        "frames": frames,
        "intrinsics": {
            "status": "dataset_reference",
            "K": [576.0, 0.0, 319.5, 0.0, 576.0, 239.5, 0.0, 0.0, 1.0],
            "source": "fixture calibration",
        },
        "onboarding": {
            "keyframe_ids": ["f0", "f1"],
            "selection": {
                "frame_id": "f0",
                "positive": [[0.0, 0.0]],
                "negative": [[639.0, 479.0]],
                "reviewed": True,
                "reviewed_by": "fixture-reviewer",
            },
        },
        "scoring_frame_ids": [f"f{i}" for i in range(2, frame_count)],
        "evidence_role": "synthetic_test_only",
        "evaluator_inputs_read": False,
        "provenance": {"fixture": True},
    }


def run_document(*, scheduled=("f2", "f3"), onboarding_ids=("f0", "f1")):
    return {
        "schema": wc.RUN_SCHEMA,
        "run_id": "run-01",
        "clip_id": "clip-01",
        "mode": "focused",
        "onboarding": {
            "keyframe_ids": list(onboarding_ids),
            "selection": {
                "frame_id": onboarding_ids[0],
                "positive": [[1.0, 1.0]],
                "negative": [[2.0, 2.0]],
                "reviewed": True,
                "reviewed_by": "fixture-reviewer",
            },
        },
        "scheduled_frame_ids": list(scheduled),
        "roles": ["primary_neural"],
        "primary": {"artifact_id": "test-artifact", "export_sha256": HASH_A, "export_bytes": 1024},
        "surface_package_id": None,
        "artwork_kind": "checkerboard",
        "config": {"fixture": True},
    }


def frame_result_document(*, frame_id="f2", source_frame_index=1150,
                          mask_state="observed", pose_state="tracking",
                          render_state="rendered", pose=None, failure_reason=None,
                          mask_frame_id=None, mask_sha256=None, image_sha256=HASH_A):
    if pose is None and pose_state == "tracking":
        pose = [[1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 1.5],
                [0.0, 0.0, 0.0, 1.0]]
    if pose_state != "tracking" and failure_reason is None:
        failure_reason = "synthetic unavailable frame"
    if mask_state == "observed":
        mask_frame_id = frame_id if mask_frame_id is None else mask_frame_id
        mask_sha256 = HASH_B if mask_sha256 is None else mask_sha256
    return {
        "schema": wc.FRAME_RESULT_SCHEMA,
        "run_id": "run-01",
        "clip_id": "clip-01",
        "frame_id": frame_id,
        "source_frame_index": source_frame_index,
        "timestamp_s": 0.2,
        "media_time_s": 0.2,
        "backend_role": "primary_neural",
        "backend_identity": {"model": "test-double"},
        "mask_state": mask_state,
        "pose_state": pose_state,
        "render_state": render_state,
        "failure_reason": failure_reason,
        "pose": pose,
        "pose_units": "object_unit" if pose is not None else None,
        "image_sha256": image_sha256 if pose is not None else None,
        "mask_frame_id": mask_frame_id,
        "mask_sha256": mask_sha256,
        "metrics": {"confidence": 0.75},
    }


class ClipContractTests(unittest.TestCase):
    def test_round_trip_keeps_absolute_source_index_and_exact_media_pts(self):
        document = clip_document(start_index=1148)
        parsed = wc.ClipSpec.from_dict(document)
        self.assertEqual(parsed.frames[0].source_frame_index, 1148)
        self.assertEqual(parsed.frames[0].media_time_s, 0.0)
        self.assertEqual(wc.ClipSpec.from_dict(parsed.to_dict()).to_dict(), parsed.to_dict())

    def test_frame_index_clock_keeps_source_time_null_without_inventing_fps(self):
        document = clip_document(clock_kind="frame_index", media_times=True)
        parsed = wc.ClipSpec.from_dict(document)
        self.assertTrue(all(frame.timestamp_s is None for frame in parsed.frames))
        self.assertEqual([frame.media_time_s for frame in parsed.frames], [0.0, 0.1, 0.2, 0.30000000000000004])
        attestation = wc.check_onboarding(parsed, parsed.onboarding, parsed.scoring_frame_ids)
        self.assertFalse(attestation["prefix_bound_attested"])
        self.assertIsNone(attestation["prefix_duration_s"])
        self.assertIn("cannot attest", attestation["note"])

    def test_frame_index_clock_rejects_populated_dataset_timestamps(self):
        document = clip_document(clock_kind="frame_index")
        document["frames"][1]["timestamp_s"] = 0.04
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)

    def test_timestamp_clock_requires_complete_strictly_increasing_times_and_source(self):
        document = clip_document()
        document["frames"][2]["timestamp_s"] = document["frames"][1]["timestamp_s"]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)
        document = clip_document()
        document["clock_source"] = None
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)

    def test_frame_ids_and_source_indices_are_unique_and_ordered(self):
        document = clip_document()
        document["frames"][1]["frame_id"] = document["frames"][0]["frame_id"]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)
        document = clip_document()
        document["frames"][2]["source_frame_index"] = document["frames"][1]["source_frame_index"]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)

    def test_timestamp_prefix_boundary_is_measured_from_source_clock(self):
        exact = wc.ClipSpec.from_dict(clip_document(prefix_step=10.0))
        attestation = wc.check_onboarding(exact, exact.onboarding, exact.scoring_frame_ids)
        self.assertTrue(attestation["prefix_bound_attested"])
        self.assertEqual(attestation["prefix_duration_s"], 10.0)
        # Two onboarding samples span 10.0005 seconds.
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(clip_document(prefix_step=10.0005))

    def test_onboarding_selection_checks_both_signs_at_pixel_edges(self):
        document = clip_document()
        document["onboarding"]["selection"]["positive"] = [[-0.5, -0.5]]
        document["onboarding"]["selection"]["negative"] = [[639.499, 479.499]]
        parsed = wc.ClipSpec.from_dict(document)
        parsed.onboarding.selection.check_bounds(parsed.width, parsed.height)
        document["onboarding"]["selection"]["negative"] = [[639.5, 200.0]]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)
        document["onboarding"]["selection"]["positive"] = [[-0.50001, 0.0]]
        document["onboarding"]["selection"]["negative"] = [[20.0, 20.0]]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)

    def test_image_sequence_requires_distinct_hashed_member_paths(self):
        document = clip_document()
        document["source_kind"] = "image_sequence"
        for i, row in enumerate(document["frames"]):
            row["image_path"] = f"frames/{i:04d}.png"
            row["image_sha256"] = HASH_A
        self.assertEqual(len(wc.ClipSpec.from_dict(document).frames), 4)
        document["frames"][1]["image_path"] = document["frames"][0]["image_path"]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)

    def test_media_and_image_paths_cannot_escape_their_registered_root(self):
        for path in ("../outside.mp4", "/tmp/video.mp4", "C:/private.mp4", "nested\\video.mp4"):
            with self.subTest(path=path):
                document = clip_document()
                document["media"]["path"] = path
                with self.assertRaises(wc.ContractError):
                    wc.ClipSpec.from_dict(document)

    def test_unknown_fields_and_boolean_indices_are_rejected(self):
        document = clip_document()
        document["fps"] = 30
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)
        document = clip_document()
        document["frames"][0]["source_frame_index"] = True
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)
        document = clip_document()
        document["width"] = True
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(document)

    def test_duplicate_ids_and_nonfinite_json_fail_at_parse_boundary(self):
        for raw in ('{"clip_id":"a","clip_id":"b"}', '{"time":NaN}', '{"time":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                wc.loads_strict(raw)


class RunAndReportTests(unittest.TestCase):
    def test_run_must_score_only_after_last_onboarding_keyframe(self):
        clip = wc.ClipSpec.from_dict(clip_document())
        valid = wc.RunSpec.from_dict(run_document())
        self.assertEqual(valid.validate_against(clip)["prefix_duration_s"], 0.1)
        early = wc.RunSpec.from_dict(run_document(scheduled=("f1", "f2")))
        with self.assertRaises(wc.ContractError):
            early.validate_against(clip)

    def test_schedule_cannot_include_unlisted_or_out_of_order_scoring_frames(self):
        clip_doc = clip_document()
        clip_doc["scoring_frame_ids"] = ["f3", "f2"]
        with self.assertRaises(wc.ContractError):
            wc.ClipSpec.from_dict(clip_doc)
        clip = wc.ClipSpec.from_dict(clip_document())
        extra = wc.RunSpec.from_dict(run_document(scheduled=("f2", "f0")))
        with self.assertRaises(wc.ContractError):
            extra.validate_against(clip)

    def test_keyframe_cap_and_unique_ids_are_enforced(self):
        data = {"keyframe_ids": [f"k{i}" for i in range(25)], "selection": None}
        with self.assertRaises(wc.ContractError):
            wc.Onboarding.from_dict(data)
        data["keyframe_ids"] = ["k0", "k0"]
        with self.assertRaises(wc.ContractError):
            wc.Onboarding.from_dict(data)

    def test_frozen_baseline_cannot_replace_primary_neural_role(self):
        document = run_document()
        document["roles"] = ["frozen_baseline"]
        with self.assertRaises(wc.ContractError):
            wc.RunSpec.from_dict(document)

    def test_report_retains_each_scheduled_failure_and_counts_availability(self):
        clip = wc.ClipSpec.from_dict(clip_document())
        run = wc.RunSpec.from_dict(run_document())
        success = frame_result_document()
        failed = frame_result_document(
            frame_id="f3", source_frame_index=1151, mask_state="unavailable",
            pose_state="lost", render_state="suppressed", pose=None,
            failure_reason="tracking lost after recovery")
        rows = [wc.FrameResult.from_dict(success), wc.FrameResult.from_dict(failed)]
        report = wc.build_report(run, clip, rows, annotations_available=False)
        primary = report["roles"]["primary_neural"]
        self.assertEqual(primary["scheduled_frames"], 2)
        self.assertEqual(primary["result_rows"], 2)
        self.assertEqual(primary["missing_rows"], [])
        self.assertEqual(primary["counts"]["pose_state"]["tracking"], 1)
        self.assertEqual(primary["counts"]["pose_state"]["lost"], 1)
        self.assertEqual(primary["availability"], 0.5)
        self.assertEqual(report["accuracy"]["status"], "unavailable_no_independent_annotations")
        self.assertFalse(report["accuracy"]["gates_verified"])

    def test_report_exposes_missing_rows_and_rejects_duplicate_or_unscheduled_results(self):
        clip = wc.ClipSpec.from_dict(clip_document())
        run = wc.RunSpec.from_dict(run_document())
        one = wc.FrameResult.from_dict(frame_result_document())
        report = wc.build_report(run, clip, [one])
        self.assertEqual(report["roles"]["primary_neural"]["missing_rows"], ["f3"])
        with self.assertRaises(wc.ContractError):
            wc.build_report(run, clip, [one, one])
        extra = wc.FrameResult.from_dict(frame_result_document(frame_id="f1", source_frame_index=1149))
        with self.assertRaises(wc.ContractError):
            wc.build_report(run, clip, [one, extra])


class GeometryAndFrameStateContractTests(unittest.TestCase):
    def test_metric_and_object_unit_status_are_explicit_and_consistent(self):
        package = {
            "schema": wc.SURFACE_SCHEMA, "package_id": "surface-01", "clip_id": "clip-01",
            "status": "unavailable", "unavailable_reason": "fixture has no geometry",
            "units": "object_unit", "metric_scale_available": False, "normalization": None,
            "vertices": None, "faces": None, "face_support_views": None,
            "vertex_count": 0, "face_count": 0, "cameras": [],
            "descriptor_points": None, "descriptor_values": None, "admission": {},
            "marker": None, "artwork": None, "frozen": False,
            "frozen_after_frame_id": None, "prefix_frame_ids": [],
            "reconstruction": {}, "preview": {}, "evaluator_inputs_read": False,
        }
        parsed = wc.SurfacePackage.from_dict(package)
        self.assertEqual(parsed.units, "object_unit")
        self.assertFalse(parsed.metric_scale_available)
        package["units"] = "metric"
        with self.assertRaises(wc.ContractError):
            wc.SurfacePackage.from_dict(package)

    def test_improper_pose_rotations_and_nonfinite_or_boolean_metrics_are_rejected(self):
        invalid_poses = [
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.1, 1.0]],
        ]
        for pose in invalid_poses:
            with self.subTest(pose=pose):
                data = frame_result_document(pose=pose)
                with self.assertRaises(wc.ContractError):
                    wc.FrameResult.from_dict(data)
        data = frame_result_document()
        data["metrics"]["confidence"] = True
        with self.assertRaises(wc.ContractError):
            wc.FrameResult.from_dict(data)
        data = frame_result_document()
        data["metrics"]["confidence"] = float("nan")
        with self.assertRaises(wc.ContractError):
            wc.FrameResult.from_dict(data)

    def test_pose_and_render_are_never_public_without_fresh_observed_mask_and_tracking(self):
        invalid = [
            frame_result_document(mask_state="unavailable"),
            frame_result_document(mask_state="observed", mask_frame_id="f1"),
            frame_result_document(pose_state="lost"),
            frame_result_document(pose=None, pose_state="lost", mask_state="observed",
                                  render_state="rendered", failure_reason="lost"),
            frame_result_document(pose=None, pose_state="unavailable", mask_state="unavailable",
                                  render_state="rendered", failure_reason="no mask"),
        ]
        for data in invalid:
            with self.subTest(pose_state=data["pose_state"], mask=data["mask_state"], render=data["render_state"]):
                with self.assertRaises(wc.ContractError):
                    wc.FrameResult.from_dict(data)

    def test_unavailable_and_recovering_frames_keep_reason_without_stale_pose(self):
        for pose_state in ("recovering", "lost", "unavailable", "blocked"):
            with self.subTest(state=pose_state):
                data = frame_result_document(
                    mask_state="unavailable", pose_state=pose_state, render_state="suppressed",
                    pose=None, failure_reason=f"synthetic {pose_state}")
                parsed = wc.FrameResult.from_dict(data)
                self.assertIsNone(parsed.pose)
                self.assertIsNotNone(parsed.failure_reason)
                self.assertNotEqual(parsed.render_state, "rendered")

    def test_registered_camera_intrinsics_obey_the_same_opencv_pinhole_contract(self):
        camera = {
            "frame_id": "f0", "image_sha256": HASH_A, "mask_sha256": HASH_B,
            "width": 640, "height": 480,
            "K": [500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0],
            "pose": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            "depth": None,
        }
        wc.RegisteredCamera.from_dict(camera)
        for invalid_K in (
            [0.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0],
            [500.0, 2.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0],
            [500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.1, 1.0],
        ):
            with self.subTest(K=invalid_K):
                bad = copy.deepcopy(camera)
                bad["K"] = invalid_K
                with self.assertRaises(wc.ContractError):
                    wc.RegisteredCamera.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
