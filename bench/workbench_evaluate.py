"""Post-run evaluation, isolated from inference and its caches.

Image labels use the existing annotation editor's landmark fields:
landmark_id, object_point_m, pixel, visible, reviewed. New clips may supply
workbench_landmarks_v1 with source_clip_sha256 and explicit coordinate_units.
Unreviewed labels and missing runs never become zero-error measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from . import workbench_contract as wc
from . import workbench_geometry as wg

HEAD = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"


def evaluate(clip: wc.ClipSpec, run: wc.RunSpec, rows: list[wc.FrameResult], *,
             role: str = "primary_neural", annotations: dict | None = None,
             run_evidence: dict | None = None) -> dict:
    """Describe measured outcomes; independent accuracy remains explicitly gated."""
    run.validate_against(clip)
    scheduled = list(run.scheduled_frame_ids)
    by_id = {}
    for row in rows:
        if row.backend_role != role:
            continue
        row = wc.FrameResult.from_dict(row.to_dict())
        if row.run_id != run.run_id or row.clip_id != clip.clip_id or row.frame_id not in scheduled:
            raise ValueError("Result identity differs from the sealed run")
        expected = clip.frame(row.frame_id)
        if row.source_frame_index != expected.source_frame_index:
            raise ValueError("Result source-frame identity differs from the clip")
        for name in ("timestamp_s", "media_time_s"):
            actual_time, expected_time = getattr(row, name), getattr(expected, name)
            # Frame-index clips may acquire real media PTS only when decoded.
            # An unspecified PTS is not permission to alter the source index.
            if name == "media_time_s" and expected_time is None:
                continue
            if (actual_time is None) != (expected_time is None) or (actual_time is not None and
                    not np.isclose(actual_time, expected_time, atol=1e-9, rtol=0)):
                raise ValueError("Result timestamp differs from its matching source frame")
        if row.frame_id in by_id:
            raise ValueError("Duplicate scored result")
        by_id[row.frame_id] = row
    evidence = run_evidence or {}
    stage = evidence.get("stages", {}).get(role, {})
    installed = {r.backend_identity.get("installed_head_sha256") for r in by_id.values()}
    executed = (role == "primary_neural" and stage.get("status") in ("completed", "cache_hit") and HEAD in installed
                and installed <= {HEAD} and bool(by_id))
    result = {
        "schema": "workbench_evaluation_v1" if executed else "workbench_evaluation_unavailable_v1",
        "run_id": run.run_id, "clip_id": clip.clip_id, "backend_role": role,
        "mode": run.mode, "clip_scoring_frames": len(clip.scoring_frame_ids),
        "full_scoring_window_scheduled": scheduled == list(clip.scoring_frame_ids),
        "installed_head_sha256": HEAD if executed else None,
        "inference_executed": executed,
        "inference_source": "verified_cache" if executed and stage.get("status") == "cache_hit" else "current_run" if executed else None,
        "scheduled_frames": len(scheduled), "result_rows": len(by_id),
        "missing_frame_ids": [fid for fid in scheduled if fid not in by_id],
        "failed_frames": sum(fid not in by_id or by_id[fid].pose_state != "tracking" for fid in scheduled),
        "pose_states": dict(Counter(row.pose_state for row in by_id.values())),
        "render_states": dict(Counter(row.render_state for row in by_id.values())),
        "independent_annotations_complete": False, "evidence_kind": "unverified",
        "metrics": {name: None for name in ("attachment_median_px_720", "attachment_p95_px_720",
            "availability_fraction", "wrong_surface_events", "correct_recovery_fraction",
            "latency_p95_ms", "peak_memory_bytes", "artwork_render_fraction")},
        "landmarks": {"reviewed_visible": 0, "measured": 0, "without_pose": 0,
                      "invalid_projection": 0, "reviewed_frames": 0},
        "accuracy_gates_passed": None,
        "percentile_method": "numpy linear percentile across measured matching frames",
        "note": "Missing or blocked execution is unmeasured. Landmark errors are conditional on public poses; losses remain in the denominator.",
    }
    if not executed:
        result["unavailable_reason"] = "No completed primary run with verified installed new-head identity"
        return result
    result["metrics"]["availability_fraction"] = (len(scheduled) - result["failed_frames"]) / len(scheduled)
    result["metrics"]["artwork_render_fraction"] = sum(row.render_state == "rendered" for row in by_id.values()) / len(scheduled)
    result["availability_definition"] = "tracking poses / all scheduled frames; artwork_render_fraction is separate and does not establish correctness or expected visibility"
    latencies = [r.metrics["end_to_end_latency_ms"] for r in by_id.values()
                 if r.metrics.get("end_to_end_latency_ms") is not None]
    memory = [r.metrics["process_peak_rss_bytes"] for r in by_id.values()
              if r.metrics.get("process_peak_rss_bytes") is not None]
    if latencies:
        result["metrics"]["latency_p95_ms"] = float(np.percentile(latencies, 95))
    if memory:
        result["metrics"]["peak_memory_bytes"] = max(memory)
    elif stage.get("status") == "completed":
        result["metrics"]["peak_memory_bytes"] = evidence.get("telemetry", {}).get("process_peak_rss_bytes")
    if annotations is None:
        return result
    source_ids = {clip.frame(fid).source_frame_index: fid for fid in scheduled}
    if annotations.get("schema") == "workbench_landmarks_v1":
        if annotations.get("source_clip_sha256") != wc.canonical_sha256(clip.to_dict()):
            raise ValueError("Annotation clip hash differs from the frozen clip")
        label_rows = annotations.get("frames", [])
        units = annotations.get("coordinate_units")
        operator = annotations.get("operator")
    else:
        from .quality_annotations import normalize_annotation_document
        annotations = normalize_annotation_document(annotations)
        if clip.media.sha256 is None or annotations["source_sha256"].lower() != clip.media.sha256:
            raise ValueError("Annotation source video hash differs from the run")
        label_rows, units, operator = annotations["frames"], "metric", annotations.get("operator")
    if annotations.get("resolution") != [clip.width, clip.height]:
        raise ValueError("Annotation image dimensions differ from the run")
    if units not in ("metric", "object_unit"):
        raise ValueError("Annotations require explicit metric or object_unit coordinates")
    labels = {}
    for label in label_rows:
        raw_id = label.get("frame_id")
        fid = raw_id if isinstance(raw_id, str) and raw_id in scheduled else source_ids.get(raw_id)
        if fid is None or fid in labels:
            raise ValueError("Annotation frame is outside the scoring window or duplicated")
        labels[fid] = label
    errors, frame_details, reviewed_frames, completed_label_frames = [], [], set(), set()
    K = clip.intrinsics.matrix()
    if K is None:
        raise ValueError("Attachment projection requires verified calibration")
    origins = {r.backend_identity.get("principal_point_origin", "integer_centers") for r in by_id.values()}
    if len(origins) != 1 or not origins <= {"integer_centers", "corner"}:
        raise ValueError("Inconsistent camera conventions in one run")
    K = np.array(K, dtype=float, copy=True)
    if origins == {"corner"}:
        K[:2, 2] -= .5
    for row in by_id.values():
        recorded = row.backend_identity.get("effective_K")
        if recorded is not None and not np.allclose(np.asarray(recorded).reshape(3, 3), K, rtol=0, atol=1e-9):
            raise ValueError("Recorded effective K differs from the declared camera convention")
    for fid, label in labels.items():
        if label.get("landmark_status") in ("reviewed", "unobservable", "not_applicable_fully_hidden"):
            completed_label_frames.add(fid)
        if label.get("landmark_status") != "reviewed":
            continue
        if not operator:
            raise ValueError("Reviewed labels require the independent operator's name")
        reviewed_frames.add(fid)
        seen = set()
        for landmark in label.get("landmarks", []):
            identity = landmark.get("landmark_id")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("Landmark IDs must be unique nonempty strings")
            seen.add(identity)
            if landmark.get("visible", True) is False:
                continue
            if landmark.get("reviewed") is not True:
                raise ValueError("Every scored correspondence must be reviewed independently")
            xyz = np.asarray(landmark.get("object_point_m"), dtype=float)
            pixel = np.asarray(landmark.get("pixel"), dtype=float)
            if xyz.shape != (3,) or pixel.shape != (2,) or not np.isfinite(xyz).all() or not np.isfinite(pixel).all():
                raise ValueError("Landmark coordinates must be finite 3D/image points")
            if np.any(pixel < 0) or pixel[0] >= clip.width or pixel[1] >= clip.height:
                raise ValueError("Human image point lies outside the source image")
            result["landmarks"]["reviewed_visible"] += 1
            row = by_id.get(fid)
            if row is None or row.pose_state != "tracking" or row.pose is None:
                result["landmarks"]["without_pose"] += 1
                continue
            if row.pose_units != units:
                raise ValueError("Pose and human landmark coordinates use different units")
            uv, z = wg.project(xyz[None], K, np.asarray(row.pose))
            if z[0] <= 0 or not np.isfinite(uv).all():
                result["landmarks"]["invalid_projection"] += 1
                continue
            error = float(np.linalg.norm(uv[0] - pixel) * 720 / clip.height)
            errors.append(error)
            frame_details.append({"frame_id": fid, "landmark_id": identity, "error_px_720": error})
    result["landmarks"].update(measured=len(errors), reviewed_frames=len(reviewed_frames))
    result["landmark_errors"] = frame_details
    if errors:
        result["metrics"].update(attachment_median_px_720=float(np.median(errors)),
                                  attachment_p95_px_720=float(np.percentile(errors, 95)))
        result["evidence_kind"] = "reviewed_image_correspondences"
    complete = bool(operator) and completed_label_frames == set(scheduled)
    result["independent_annotations_complete"] = complete
    if complete and errors:
        result["accuracy_gates_passed"] = (result["metrics"]["attachment_median_px_720"] < 5
            and result["metrics"]["attachment_p95_px_720"] < 10
            and result["metrics"]["availability_fraction"] >= .9
            and not result["landmarks"]["without_pose"] and not result["landmarks"]["invalid_projection"])
    return result


def main(argv=None):
    from .workbench_cli import Workspace, load_results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = Workspace(args.workspace)
    run_doc, rows, evidence = load_results(workspace, args.run_id)
    run = wc.RunSpec.from_dict(run_doc.get("run", run_doc))
    clip = workspace.load_clip(run.clip_id)
    annotations = wc.loads_strict(args.annotations.read_bytes()) if args.annotations else None
    result = evaluate(clip, run, rows, annotations=annotations, run_evidence=evidence)
    if args.annotations:
        result["annotation_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(wc.canonical_json(result))
    print(json.dumps({"path": str(args.output.resolve()), "schema": result["schema"],
                      "inference_executed": result["inference_executed"]}))


if __name__ == "__main__":
    main()
