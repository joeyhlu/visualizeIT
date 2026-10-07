"""Audit frozen automatic mask caches against independently reviewed RGB labels.

This is an evaluation-only CPU report. It never invokes inference or passes labels
to an inference stage. Supply one frozen input manifest, automatic full-window
result, matching segmentation result index, mask root, annotation JSON, and a new
output path. Existing outputs are never replaced.

For a reviewed fully hidden frame, the annotation row must explicitly contain
``visibility: "fully_hidden"`` and an empty ``visible_object`` list. An empty
polygon list without that explicit reviewed-absence label is ambiguous and is
rejected. The current annotation editor cannot create this label yet.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from .quality_evaluate import mask_metrics, polygon_mask
from .vision import cv2


# These are the original scored windows recorded in the frozen input manifests.
# Keep the accepted CLI scope fixed to these source IDs and native dimensions.
ORIGINAL_WINDOWS = {
    "keyboard": {"setup_frame_id": 1148, "frame_ids": tuple(range(1149, 1389)), "resolution": (1024, 1280)},
    "mug": {"setup_frame_id": 826, "frame_ids": tuple(range(827, 1067)), "resolution": (1024, 1280)},
    "ranch": {"setup_frame_id": 9, "frame_ids": tuple(range(10, 250)), "resolution": (1024, 1280)},
}
# Literal frozen candidate selections from the canonical pending annotation files.
# These independent constants prevent a caller from moving a label onto a more
# convenient frame after seeing a cached prediction.
FROZEN_ANNOTATION_SELECTIONS = MappingProxyType({
    "keyboard": (
        (1149, "uniform"), (1150, "hard_candidate"), (1154, "hard_candidate"), (1157, "uniform"),
        (1165, "uniform"), (1168, "hard_candidate"), (1174, "uniform"), (1182, "uniform"),
        (1187, "hard_candidate"), (1190, "uniform"), (1198, "uniform"), (1207, "uniform"),
        (1208, "hard_candidate"), (1215, "uniform"), (1223, "uniform"), (1231, "uniform"),
        (1233, "hard_candidate"), (1240, "uniform"), (1248, "uniform"), (1256, "uniform"),
        (1257, "hard_candidate"), (1264, "uniform"), (1273, "uniform"), (1281, "uniform"),
        (1286, "hard_candidate"), (1289, "uniform"), (1297, "uniform"), (1306, "uniform"),
        (1314, "uniform"), (1315, "hard_candidate"), (1322, "uniform"), (1330, "uniform"),
        (1339, "uniform"), (1347, "uniform"), (1348, "hard_candidate"), (1355, "uniform"),
        (1363, "uniform"), (1372, "uniform"), (1380, "uniform"), (1388, "uniform"),
    ),
    "mug": (
        (827, "uniform"), (828, "hard_candidate"), (832, "hard_candidate"), (835, "uniform"),
        (843, "uniform"), (846, "hard_candidate"), (852, "uniform"), (860, "uniform"),
        (865, "hard_candidate"), (868, "uniform"), (876, "uniform"), (885, "uniform"),
        (886, "hard_candidate"), (893, "uniform"), (901, "uniform"), (909, "uniform"),
        (911, "hard_candidate"), (918, "uniform"), (926, "uniform"), (934, "uniform"),
        (935, "hard_candidate"), (942, "uniform"), (951, "uniform"), (959, "uniform"),
        (964, "hard_candidate"), (967, "uniform"), (975, "uniform"), (984, "uniform"),
        (992, "uniform"), (993, "hard_candidate"), (1000, "uniform"), (1008, "uniform"),
        (1017, "uniform"), (1025, "uniform"), (1026, "hard_candidate"), (1033, "uniform"),
        (1041, "uniform"), (1050, "uniform"), (1058, "uniform"), (1066, "uniform"),
    ),
    "ranch": (
        (10, "uniform"), (11, "hard_candidate"), (15, "hard_candidate"), (18, "uniform"),
        (26, "uniform"), (29, "hard_candidate"), (35, "uniform"), (43, "uniform"),
        (48, "hard_candidate"), (51, "uniform"), (59, "uniform"), (68, "uniform"),
        (69, "hard_candidate"), (76, "uniform"), (84, "uniform"), (92, "uniform"),
        (94, "hard_candidate"), (101, "uniform"), (109, "uniform"), (117, "uniform"),
        (118, "hard_candidate"), (125, "uniform"), (134, "uniform"), (142, "uniform"),
        (147, "hard_candidate"), (150, "uniform"), (158, "uniform"), (167, "uniform"),
        (175, "uniform"), (176, "hard_candidate"), (183, "uniform"), (191, "uniform"),
        (200, "uniform"), (208, "uniform"), (209, "hard_candidate"), (216, "uniform"),
        (224, "uniform"), (233, "uniform"), (241, "uniform"), (249, "uniform"),
    ),
})
ANNOTATION_FRAME_COUNT = 40
USABLE_MASK_MIN_PIXELS = 24
DEFICIENCY_THRESHOLDS = {"iou_below": 0.90, "boundary_p95_720_above": 3.0, "hand_leakage_above": 0.01}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, label):
    path = Path(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {label}: {path}")
    return value, _sha256(raw)


def _ordered_ids(records, key, label):
    if not isinstance(records, list):
        raise ValueError(f"{label} must be a list")
    ids = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid {label} record")
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} IDs must be integers")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {label}")
    if ids != sorted(ids):
        raise ValueError(f"{label} IDs must be unique and ordered")
    return ids


def _safe_file(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Missing relative path for {label}")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError(f"Unsafe {label} path: {relative}")
    root = Path(root).resolve()
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} path escapes its root: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    return path


def _hash_verified_input_assets(input_path, manifest):
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("Input manifest must contain source asset hashes")
    bundle = Path(input_path).parent
    verified = {}
    for field in ("video", "asset"):
        relative = manifest.get(field)
        expected = source_hashes.get(relative) if isinstance(relative, str) else None
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Input manifest is missing the {field} source hash")
        path = _safe_file(bundle, relative, f"input {field}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"Input {field} hash does not match the manifest")
        verified[relative] = actual
    return verified


def _load_mask(path, width, height, frame_id):
    raw = path.read_bytes()
    digest = _sha256(raw)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Corrupt cached mask for frame {frame_id}: {path}")
    if mask.ndim != 2 or mask.shape != (height, width):
        raise ValueError(f"Cached mask for frame {frame_id} is not native resolution {width}x{height}")
    return raw, digest, mask


def _validate_annotation_rows(annotations, object_name, resolution, source_video_hash, expected_ids):
    if annotations.get("object") != object_name:
        raise ValueError("Annotation object does not match the frozen run")
    if annotations.get("resolution") != list(resolution):
        raise ValueError("Annotation resolution must exactly match the native input resolution")
    if annotations.get("source_sha256") != source_video_hash:
        raise ValueError("Annotations are not tied to the original source RGB video")
    rows = annotations.get("frames")
    if not isinstance(rows, list) or len(rows) != ANNOTATION_FRAME_COUNT:
        raise ValueError(f"Expected the frozen {ANNOTATION_FRAME_COUNT}-frame annotation selection")
    ids = _ordered_ids(rows, "frame_id", "annotation")
    frozen_selection = FROZEN_ANNOTATION_SELECTIONS[object_name]
    frozen_ids = [frame_id for frame_id, _ in frozen_selection]
    if ids != frozen_ids:
        raise ValueError("Annotation IDs do not match the immutable original candidate selection")
    if any(frame_id not in expected_ids for frame_id in ids):
        raise ValueError("Annotation frame is outside the original scored window")
    for row, (frame_id, group) in zip(rows, frozen_selection):
        if row.get("group") != group or row.get("image") != f"{frame_id}.jpg":
            raise ValueError(f"Annotation group or original RGB image changed at frame {frame_id}")
        if row.get("status") not in ("pending", "reviewed"):
            raise ValueError("Annotation status must be explicitly pending or reviewed")
        if not isinstance(row.get("visible_object", []), list) or not isinstance(row.get("overlapping_hands", []), list):
            raise ValueError("Visible-object and overlapping-hand labels must be polygon lists")
    return rows


def _metadata_differences(run_provenance, cache_provenance):
    if not isinstance(cache_provenance, dict):
        return {"segmentation_cache_provenance": {
            "automatic_result_metadata_present": True,
            "segmentation_cache_metadata_present": isinstance(cache_provenance, dict),
        }}
    fields = set(run_provenance) | set(cache_provenance)
    return {
        field: {"automatic_result": run_provenance.get(field), "segmentation_cache": cache_provenance.get(field)}
        for field in sorted(fields)
        if run_provenance.get(field) != cache_provenance.get(field)
    }


def _validate_explicit_mask_generation_claim(mask_results, input_sha, asset_hashes):
    claim = mask_results.get("mask_generation_provenance")
    if claim is None:
        return {"present": False, "verified": False, "reason": "no explicit mask-generation provenance claim"}
    if not isinstance(claim, dict):
        raise ValueError("Explicit mask-generation provenance must be an object")
    if claim.get("input_manifest_sha256") != input_sha:
        raise ValueError("Explicit mask-generation provenance does not match the input manifest")
    claimed_assets = claim.get("source_hashes")
    if claimed_assets is not None and claimed_assets != asset_hashes:
        raise ValueError("Explicit mask-generation provenance does not match the verified input assets")
    return {"present": True, "verified": True, "input_manifest_sha256": input_sha,
            "source_hashes_checked": claimed_assets is not None}


def _annotation_visibility(row, visible_truth):
    declared = row.get("visibility")
    if declared == "fully_hidden":
        if visible_truth.any():
            raise ValueError("A fully hidden reviewed frame cannot contain visible-object polygons")
        return "fully_hidden"
    if declared not in (None, "visible"):
        raise ValueError("Reviewed visibility must be 'visible' or explicit 'fully_hidden'")
    if not visible_truth.any():
        raise ValueError("An empty reviewed target needs explicit visibility='fully_hidden'")
    return "visible"


def _mean(values):
    return float(np.mean(values)) if values else None


def _score_summary(rows):
    if not rows:
        return {"frames": 0, "mean_iou": None, "worst_boundary_p95_720": None,
                "mean_hand_leakage": None, "boundary_failure_frames": 0}
    metrics = [row["metrics"] for row in rows]
    boundaries = [metric["boundary_p95_720"] for metric in metrics if metric["boundary_p95_720"] is not None]
    return {
        "frames": len(rows),
        "mean_iou": _mean([metric["iou"] for metric in metrics]),
        "worst_boundary_p95_720": max(boundaries) if boundaries else None,
        "mean_hand_leakage": _mean([metric["hand_leakage"] for metric in metrics]),
        "boundary_failure_frames": sum(bool(metric["boundary_failure"]) for metric in metrics),
    }


def audit_cached_masks(object_name, input_path, results_path, mask_results_path,
                       mask_root, annotations_path):
    """Validate and score a frozen automatic run against reviewed source-RGB labels."""
    if object_name not in ORIGINAL_WINDOWS:
        raise ValueError(f"Unknown object: {object_name}")
    window = ORIGINAL_WINDOWS[object_name]
    expected_ids = list(window["frame_ids"])
    setup_id = window["setup_frame_id"]
    width, height = window["resolution"]

    # Read and hash every provenance record before any label metrics are computed.
    manifest, input_sha = _read_json(input_path, "input manifest")
    results, results_sha = _read_json(results_path, "automatic result")
    mask_results, mask_index_sha = _read_json(mask_results_path, "segmentation result index")
    annotations, annotations_sha = _read_json(annotations_path, "annotation")

    if manifest.get("object") != object_name or manifest.get("native_resolution") != [width, height]:
        raise ValueError("Input object or native resolution does not match the original window")
    if manifest.get("setup_frame_id") != setup_id or manifest.get("frame_ids") != expected_ids:
        raise ValueError("Input manifest does not contain the exact original 240 scored IDs")
    if results.get("object") != object_name or results.get("mode") != "complete" or not results.get("automatic"):
        raise ValueError("A completed automatic full-pipeline result is required")
    if results.get("diagnostic_control") or results.get("complete") is not True:
        raise ValueError("Diagnostic controls and incomplete results cannot be audited")
    if results.get("frame_ids") != expected_ids:
        raise ValueError("Automatic result manifest IDs do not match the original 240-frame window")
    if mask_results.get("object") not in (None, object_name):
        raise ValueError("Segmentation result object does not match the frozen run")
    if mask_results.get("diagnostic_control") is True or mask_results.get("automatic") is False:
        raise ValueError("Diagnostic or explicitly non-automatic mask caches cannot be audited as automatic predictions")
    if results.get("stress_test") is not None or mask_results.get("stress_test") is not None:
        raise ValueError("Stress-test runs and mask caches cannot be audited as original RGB predictions")

    result_ids = _ordered_ids(results.get("frames"), "frameId", "automatic result")
    mask_ids = _ordered_ids(mask_results.get("frames"), "frameId", "segmentation result")
    if result_ids != expected_ids:
        raise ValueError("Automatic result must contain every original scored ID exactly once and in order")
    if mask_ids != [setup_id] + expected_ids:
        raise ValueError("Mask cache must contain setup and every original scored ID exactly once and in order")

    run_provenance = results.get("provenance")
    if not isinstance(run_provenance, dict) or run_provenance.get("input_manifest_sha256") != input_sha:
        raise ValueError("Automatic result does not match the supplied input manifest hash")
    cache_provenance = mask_results.get("provenance")
    mask_association = bool(mask_results.get("mask_association", False))
    asset_hashes = _hash_verified_input_assets(input_path, manifest)
    explicit_generation_claim = _validate_explicit_mask_generation_claim(mask_results, input_sha, asset_hashes)
    provenance_comparison = {
        "pose_run_global_metadata_compared_as_mask_generation_provenance": False,
        "recorded_metadata_differences": _metadata_differences(run_provenance, cache_provenance),
        "explicit_mask_generation_claim": explicit_generation_claim,
    }
    source_video_hash = asset_hashes[manifest["video"]]
    annotation_rows = _validate_annotation_rows(annotations, object_name, (width, height),
                                                source_video_hash, expected_ids)
    annotation_by_id = {row["frame_id"]: row for row in annotation_rows}
    reviewed_rows = {frame_id: row for frame_id, row in annotation_by_id.items() if row["status"] == "reviewed"}

    run_by_id = {frame["frameId"]: frame for frame in results["frames"]}
    cache_by_id = {frame["frameId"]: frame for frame in mask_results["frames"]}
    mask_hashes = {}
    raw_mask_areas = {}
    effective_mask_areas = {}
    reviewed_masks = {}

    # Preflight every setup/scored PNG before interpreting any label. Missing or
    # malformed cache entries are errors, even when the corresponding state says lost.
    for frame_id in mask_ids:
        cache_record = cache_by_id[frame_id]
        if cache_record.get("mask_state") not in ("available", "lost"):
            raise ValueError(f"Invalid cached mask state at frame {frame_id}")
        path = _safe_file(mask_root, cache_record.get("path"), f"cached mask at frame {frame_id}")
        _, mask_hash, mask = _load_mask(path, width, height, frame_id)
        raw_pixels = int(np.count_nonzero(mask))
        state = cache_record["mask_state"]
        # SAM propagation uses a 24-pixel cutoff, while a successful CNOS
        # detection is accepted when it contains any foreground. Preserve the
        # recorded available state for those small detector masks.
        if state == "available" and raw_pixels == 0:
            raise ValueError(f"Available cache mask is empty at frame {frame_id}")
        if state == "lost" and raw_pixels >= USABLE_MASK_MIN_PIXELS:
            raise ValueError(f"Lost cache mask meets the {USABLE_MASK_MIN_PIXELS}-pixel availability cutoff at frame {frame_id}")
        effective_mask = (mask > 0) if state == "available" else np.zeros((height, width), dtype=bool)
        mask_hashes[str(frame_id)] = mask_hash
        raw_mask_areas[frame_id] = raw_pixels
        effective_mask_areas[frame_id] = int(np.count_nonzero(effective_mask))
        if frame_id != setup_id:
            result = run_by_id[frame_id]
            if result.get("mask_path") != cache_record.get("path"):
                raise ValueError(f"Automatic result and segmentation path disagree at frame {frame_id}")
            if result.get("mask_state") != cache_record.get("mask_state"):
                raise ValueError(f"Automatic result and segmentation state disagree at frame {frame_id}")
            if result.get("mask_sha256") != mask_hash:
                raise ValueError(f"Cached mask hash does not match the automatic result at frame {frame_id}")
            if frame_id in reviewed_rows:
                reviewed_masks[frame_id] = effective_mask
        else:
            initialization = results.get("initialization")
            if isinstance(initialization, dict) and initialization.get("mask_state") not in (None, cache_record["mask_state"]):
                raise ValueError("Setup mask state disagrees with the automatic initialization record")
    # The complete frozen record is assembled before score computation.
    frozen = {
        "object": object_name,
        "input_manifest_path": str(Path(input_path)),
        "input_manifest_sha256": input_sha,
        "verified_input_asset_sha256": asset_hashes,
        "automatic_result_path": str(Path(results_path)),
        "automatic_result_sha256": results_sha,
        "segmentation_index_path": str(Path(mask_results_path)),
        "segmentation_index_sha256": mask_index_sha,
        "mask_root": str(Path(mask_root)),
        "annotation_path": str(Path(annotations_path)),
        "annotation_sha256": annotations_sha,
        "setup_frame_id": setup_id,
        "scored_source_frame_ids": expected_ids,
        "setup_and_scored_mask_sha256": mask_hashes,
        "tracking_settings": results.get("tracking_settings"),
        "automatic_run_provenance": run_provenance,
        "segmentation_cache_provenance": cache_provenance,
        "mask_association_enabled": mask_association,
        "segmentation_provenance_comparison": provenance_comparison,
        "annotation_selection_frame_ids": [frame_id for frame_id, _ in FROZEN_ANNOTATION_SELECTIONS[object_name]],
        "mask_cache_match_basis": "verified input assets plus ordered setup/scored frame IDs and exact per-scored-frame paths, states, and PNG hashes against the automatic result",
    }

    reviewed_details = []
    for frame_id, row in reviewed_rows.items():
        visible_truth = polygon_mask(row.get("visible_object", []), (height, width))
        hands = polygon_mask(row.get("overlapping_hands", []), (height, width))
        if np.any(visible_truth & hands):
            raise ValueError(f"Visible-object annotation overlaps a hand at frame {frame_id}")
        visibility = _annotation_visibility(row, visible_truth)
        predicted = reviewed_masks[frame_id]
        metrics = mask_metrics(predicted, visible_truth, hands)
        cache_record = cache_by_id[frame_id]
        run_record = run_by_id[frame_id]
        reviewed_details.append({
            "frame_id": frame_id,
            "annotation_status": "reviewed",
            "visibility": visibility,
            "hand_overlap_reviewed": bool(hands.any()),
            "visible_truth_pixels": int(visible_truth.sum()),
            "hand_pixels": int(hands.sum()),
            "false_foreground_pixels_on_fully_hidden_frame": metrics["predicted_pixels"] if visibility == "fully_hidden" else None,
            "mask_state": cache_record["mask_state"],
            "pose_state": run_record.get("pose_state"),
            "render_state": run_record.get("render_state"),
            "metrics": metrics,
        })

    details_by_id = {detail["frame_id"]: detail for detail in reviewed_details}
    all_details = []
    for frame_id in expected_ids:
        cache_record = cache_by_id[frame_id]
        run_record = run_by_id[frame_id]
        annotation = annotation_by_id.get(frame_id)
        timings = cache_record.get("timings_ms") or {}
        if not isinstance(timings, dict):
            raise ValueError(f"Invalid timing record at frame {frame_id}")
        detection_timing = timings.get("detection")
        detection_reason = cache_record.get("automatic_detection_reason")
        has_detection_record = detection_timing is not None or bool(detection_reason)
        if detection_timing is not None:
            if isinstance(detection_timing, bool) or not isinstance(detection_timing, (int, float)) or not math.isfinite(detection_timing) or detection_timing < 0:
                raise ValueError(f"Invalid detector timing at frame {frame_id}")
        if detection_reason is not None and not isinstance(detection_reason, str):
            raise ValueError(f"Invalid detector reason at frame {frame_id}")
        detail = {
            "frame_id": frame_id,
            "annotation_status": annotation["status"] if annotation else "not_selected",
            "mask_state": cache_record["mask_state"],
            "raw_mask_artifact_pixels": raw_mask_areas[frame_id],
            "predicted_pixels": effective_mask_areas[frame_id],
            "pose_state": run_record.get("pose_state"),
            "render_state": run_record.get("render_state"),
            "failure_reason": run_record.get("failure_reason"),
            "automatic_detection_required_raw": cache_record.get("automatic_detection_required"),
            "detection_evidence": "recorded" if has_detection_record else "unknown",
            "detection_timing_ms": detection_timing,
            "detection_reason": detection_reason,
            "metrics": details_by_id[frame_id]["metrics"] if frame_id in details_by_id else None,
        }
        all_details.append(detail)

    reviewed_ids = [frame_id for frame_id in expected_ids if frame_id in reviewed_rows]
    pending_ids = [frame_id for frame_id in expected_ids if frame_id in annotation_by_id and annotation_by_id[frame_id]["status"] == "pending"]
    unreviewed_ids = [frame_id for frame_id in expected_ids if frame_id not in reviewed_rows]
    available_reviewed = [row for row in reviewed_details if row["mask_state"] == "available"]

    deficient = []
    reason_counts = {"iou_below_0_90": 0, "boundary_above_3px_or_missing": 0, "hand_leakage_above_0_01": 0}
    for detail in available_reviewed:
        metric = detail["metrics"]
        reasons = []
        if metric["iou"] < DEFICIENCY_THRESHOLDS["iou_below"]:
            reasons.append("iou_below_0_90")
        if metric["boundary_p95_720"] is None or metric["boundary_p95_720"] > DEFICIENCY_THRESHOLDS["boundary_p95_720_above"]:
            reasons.append("boundary_above_3px_or_missing")
        if metric["hand_leakage"] > DEFICIENCY_THRESHOLDS["hand_leakage_above"]:
            reasons.append("hand_leakage_above_0_01")
        for reason in reasons:
            reason_counts[reason] += 1
        if reasons:
            deficient.append({"frame_id": detail["frame_id"], "reasons": reasons})

    hand_rows = [row for row in reviewed_details if row["hand_overlap_reviewed"]]
    no_hand_rows = [row for row in reviewed_details if not row["hand_overlap_reviewed"]]
    fully_hidden_rows = [row for row in reviewed_details if row["visibility"] == "fully_hidden"]
    state_counts = {state: sum(cache_by_id[frame_id]["mask_state"] == state for frame_id in expected_ids)
                    for state in ("available", "lost")}
    transitions = []
    previous = None
    for frame_id in expected_ids:
        state = cache_by_id[frame_id]["mask_state"]
        if previous is not None and state != previous["state"]:
            transitions.append({"frame_id": frame_id, "from": previous["state"], "to": state})
        previous = {"frame_id": frame_id, "state": state}

    detection_recorded_ids = [detail["frame_id"] for detail in all_details if detail["detection_evidence"] == "recorded"]
    detection_unknown_ids = [detail["frame_id"] for detail in all_details if detail["detection_evidence"] == "unknown"]
    required_flag_ids = [detail["frame_id"] for detail in all_details if detail["automatic_detection_required_raw"] is True]
    scored_summary = _score_summary(reviewed_details)
    label_set_complete = (
        len(reviewed_ids) == ANNOTATION_FRAME_COUNT
        and annotations.get("selection_status") == "reviewed_image_only"
        and isinstance(annotations.get("operator"), str)
        and bool(annotations["operator"].strip())
    )
    mask_gate = bool(label_set_complete and scored_summary["mean_iou"] is not None
        and scored_summary["mean_iou"] >= 0.90
        and scored_summary["worst_boundary_p95_720"] is not None
        and scored_summary["worst_boundary_p95_720"] <= 3.0
        and scored_summary["mean_hand_leakage"] is not None
        and scored_summary["mean_hand_leakage"] <= 0.01
        and scored_summary["boundary_failure_frames"] == 0)
    per_hand = {
        "hand_overlap": _score_summary(hand_rows),
        "no_hand_overlap": _score_summary(no_hand_rows),
    }
    reason_counts_with_denominators = {
        reason: {"count": count, "denominator_available_reviewed": len(available_reviewed)}
        for reason, count in reason_counts.items()
    }

    return {
        "schema_version": 1,
        "scope": "Frozen automatic SAM mask audit against independent human polygons reviewed from original RGB only; sparse labels do not measure all-frame identity persistence or recovery latency.",
        "object": object_name,
        "processed_original_scored_frames": len(expected_ids),
        "source_frame_ids": expected_ids,
        "native_resolution": [width, height],
        "frozen_provenance": frozen,
        "reference_or_annotation_inputs_to_inference": False,
        "annotation_basis": "human-reviewed visible-target and hand polygons from original RGB; pending labels are excluded",
        "reviewed_frame_ids": reviewed_ids,
        "pending_candidate_frame_ids": pending_ids,
        "unreviewed_frame_ids": unreviewed_ids,
        "annotation_frames_reviewed": len(reviewed_ids),
        "annotation_frames_required": ANNOTATION_FRAME_COUNT,
        "independent_accuracy_scope": "mask-only diagnostic; does not satisfy the combined independent gate requiring landmarks, accepted-pose availability, and recovery evidence",
        "independent_accuracy_ready": label_set_complete,
        "combined_independent_accuracy_ready": False,
        "mask_scores": scored_summary,
        "hand_no_hand_strata": per_hand,
        "available_but_deficient_reviewed_masks": {
            "count": len(deficient),
            "denominator_available_reviewed": len(available_reviewed),
            "frame_ids_and_reasons": deficient,
            "thresholds": DEFICIENCY_THRESHOLDS,
            "reason_counts_over_available_reviewed_denominator": reason_counts_with_denominators,
        },
        "fully_hidden_reviewed_frames": {
            "count": len(fully_hidden_rows),
            "denominator_reviewed": len(reviewed_details),
            "false_foreground_frame_count": sum(row["metrics"]["predicted_pixels"] > 0 for row in fully_hidden_rows),
            "false_foreground_pixels_by_frame": [
                {"frame_id": row["frame_id"], "pixels": row["false_foreground_pixels_on_fully_hidden_frame"]}
                for row in fully_hidden_rows
            ],
        },
        "mask_state_counts_over_original_240": state_counts,
        "lost_mask_residual_artifact_pixels_by_frame": [
            {"frame_id": frame_id, "pixels": raw_mask_areas[frame_id]}
            for frame_id in expected_ids
            if cache_by_id[frame_id]["mask_state"] == "lost" and raw_mask_areas[frame_id] > 0
        ],
        "mask_state_transition_frame_ids": transitions,
        "automatic_detection_evidence": {
            "recorded_evidence_frame_ids": detection_recorded_ids,
            "unknown_frame_ids": detection_unknown_ids,
            "automatic_detection_required_true_frame_ids_raw_only": required_flag_ids,
            "automatic_detection_required_interpreted_as_attempt": False,
        },
        "temporal_evidence": {
            "recovery_latency_measured": False,
            "sparse_labels_are_recovery_proof": False,
            "recovery_gate_passed": False,
        },
        "mask_gate_passed": mask_gate,
        "overall_gate_passed": False,
        "frames": all_details,
    }


def _write_new_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False)
    # Exclusive create is deliberate: a frozen diagnostic can never replace a
    # prior report or baseline, even if a caller reuses an output path.
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
        output.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, choices=tuple(ORIGINAL_WINDOWS))
    parser.add_argument("--input", type=Path, required=True, help="Frozen input manifest used by the run")
    parser.add_argument("--results", type=Path, required=True, help="Automatic full-window result JSON")
    parser.add_argument("--mask-results", type=Path, required=True, help="Matching segmentation results.json")
    parser.add_argument("--mask-root", type=Path, required=True, help="Directory containing mask paths in --mask-results")
    parser.add_argument("--annotations", type=Path, required=True, help="Source-RGB reviewed/pending annotation JSON")
    parser.add_argument("--output", type=Path, help="New report path; defaults inside .cache/quality-windows")
    args = parser.parse_args(argv)
    report = audit_cached_masks(args.object, args.input, args.results, args.mask_results,
                                args.mask_root, args.annotations)
    output = args.output
    if output is None:
        root = Path(args.input).resolve().parents[3] / "quality-windows" / "quality-mask-audit"
        output = root / f"{args.object}-{report['frozen_provenance']['automatic_result_sha256'][:12]}.json"
    _write_new_report(output, report)
    print(json.dumps({"output": str(output), "object": args.object,
                      "reviewed": report["annotation_frames_reviewed"],
                      "independent_accuracy_ready": report["independent_accuracy_ready"],
                      "overall_gate_passed": report["overall_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
