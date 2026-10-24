"""Additive 24-frame GoTrack adaptation workflow.

The capture path uses estimated seeds and the existing retained-crop API. Generic
feature extraction, supervision projection, training, and export remain delegated
to their existing modules. This is a bounded development experiment, not an
accuracy, generalization, or deployment claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import time

import numpy as np

from .quality_adaptation_data import (
    CONVENTION, CROP_KEYS, FEATURE_KEYS, HOOKS, LABEL_KEYS, MAX_ARCHIVE, SIZE,
    canonical_hash, crop_rgb_hash, descriptor, flow_metrics, load_dataset,
    load_npz, object_summary, read_json, require_hash, sha256,
    validate_crops, validate_features, validate_labels, visibility_metrics,
)
from .quality_adaptation_runtime import (
    LOCK, atomic_stream, cloud_guard, load_export_into_network,
    load_frozen_network, load_head, runtime_versions, tree_bytes,
    verify_sources, write_json, training_state_digest,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "gotrack_adaptation_expanded_plan_v1"
CONTRACT_SCHEMA = "gotrack_adaptation_expanded_capture_contract_v1"
CONTRACT_PATH = ROOT / "bench/runtime/adaptation-expanded-capture-v1.json"
CONFIG_PATH = ROOT / "bench/runtime/adaptation-expanded-cpu-v1.json"
HISTORICAL_LOCK_SHA256 = "44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33"
HISTORICAL_SOURCE_MANIFEST_SHA256 = "1d45840df6f3f031a031ac133d4adccf50db8a17c9ce9d206380c5a63ae54850"
HISTORICAL_SOURCE_MAP_SHA256 = "55aa368a051e7e72482b5cf921d0598bb6b4fb04be4169a6c16a982b510d33c3"
HISTORICAL_SOURCE_COUNT = 287
FROZEN_PLAN_SHA256 = "41101b22cc7435b00df7e0cab1e11aee2d5e3c8140517948e37eac189f7a7437"
FROZEN_LABEL_PLAN_SHA256 = "90d4ca0ed99b35be74cadb86f8d1fae4469512d0a246462a113543617359cebd"
RASTER_CONVENTION = "legacy_opengl_native_principal_minus_half_to_integer_v1"
CAPTURE_RECIPE = "estimated_seed_refiner_iteration0_legacy_native_K_corrected_integer_template_float_rgb_v1"
AGGREGATE_SECONDS = 1800
PRIVATE_BYTES = 1536 * 1024**2
DISK_RESERVE_BYTES = 1024**3
EXPECTED_ROWS = (
    ("train", "17", "ranker-multiview-hot3d17-v1-yaw0-pitchm25", -25),
    ("train", "17", "ranker-multiview-hot3d17-v1-yaw0-pitchp25", 25),
    ("val", "27", "ranker-multiview-hot3d27-v1-yaw0-pitchm25", -25),
)
ROW_FIELDS = {
    "id", "object_id", "split", "sequence_id", "capture_group", "frame_id",
    "yaw0_deg", "pitch_deg", "input", "mask_index", "mask", "bank",
    "generation", "timestamp_s",
}
TEST_FIELDS = {
    "id", "object_id", "sequence_id", "capture_group", "split", "status",
    "query_rgb_sha256", "failure_reason", "unavailable_provenance",
}


def verify_contract_document(document: dict, *, root: Path,
                             expected_lock_sha256: str | None = None,
                             expected_controller_path: str | None = None,
                             expected_config_path: str | None = None,
                             expected_source_manifest_sha256: str | None = None,
                             expected_source_map_sha256: str | None = None,
                             expected_source_count: int | None = None) -> dict:
    """Authenticate the new controller/config against the unchanged source lock."""
    expected_fields = {
        "schema", "historical_source_lock_sha256", "historical_source_manifest_sha256", "controller", "config",
        "historical_sources",
    }
    if not isinstance(document, dict) or set(document) != expected_fields or document["schema"] != CONTRACT_SCHEMA:
        raise ValueError("invalid expanded capture contract fields/schema")
    lock_hash = require_hash(document["historical_source_lock_sha256"], "historical source lock hash")
    if expected_lock_sha256 is not None and lock_hash != expected_lock_sha256:
        raise ValueError("expanded contract does not bind the frozen historical source lock")
    source_manifest_hash = require_hash(document["historical_source_manifest_sha256"],
                                        "historical source manifest hash")
    if expected_source_manifest_sha256 is not None and source_manifest_hash != expected_source_manifest_sha256:
        raise ValueError("expanded contract differs from the source-18 cloud manifest")
    if (not isinstance(document["historical_sources"], list)
            or (expected_source_count is not None and len(document["historical_sources"]) != expected_source_count)):
        raise ValueError("expanded contract must include the exact historical source closure")
    pins = [document["controller"], document["config"], *document["historical_sources"]]
    seen = set()
    for pin in pins:
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise ValueError("contract source pin requires path and sha256")
        path_text = pin["path"]
        if (not isinstance(path_text, str) or "\\" in path_text
                or PurePosixPath(path_text).is_absolute() or ".." in PurePosixPath(path_text).parts):
            raise ValueError("contract source path must be relative POSIX")
        digest = require_hash(pin["sha256"])
        if path_text in seen:
            raise ValueError("duplicate contract source pin")
        seen.add(path_text)
        candidate = root / path_text
        path = candidate.resolve()
        if (candidate.is_symlink() or not path.is_relative_to(root.resolve())
                or not path.is_file() or sha256(path) != digest):
            raise ValueError(f"expanded contract source mismatch: {path_text}")
    if expected_controller_path is not None and document["controller"]["path"] != expected_controller_path:
        raise ValueError("contract controller path differs from the current controller")
    if expected_config_path is not None and document["config"]["path"] != expected_config_path:
        raise ValueError("contract config path differs from the expanded config")
    historical_map = {pin["path"]: pin["sha256"] for pin in document["historical_sources"]}
    if (expected_source_map_sha256 is not None
            and canonical_hash(historical_map) != expected_source_map_sha256):
        raise ValueError("expanded contract historical source map differs from the source-18 closure")
    lock_pin = next((p for p in document["historical_sources"] if p["path"] == "bench/runtime/adaptation-sources.json"), None)
    if lock_pin is None or lock_pin["sha256"] != lock_hash:
        raise ValueError("contract historical source pins omit or disagree with the frozen lock")
    return {"schema": CONTRACT_SCHEMA, "historical_source_lock_sha256": lock_hash,
            "historical_source_manifest_sha256": source_manifest_hash,
            "controller_sha256": document["controller"]["sha256"],
            "config_sha256": document["config"]["sha256"]}


def verify_capture_contract(path: Path = CONTRACT_PATH) -> dict:
    document = read_json(path)
    return verify_contract_document(
        document, root=ROOT, expected_lock_sha256=HISTORICAL_LOCK_SHA256,
        expected_controller_path="bench/quality_adaptation_expanded.py",
        expected_config_path="bench/runtime/adaptation-expanded-cpu-v1.json",
        expected_source_manifest_sha256=HISTORICAL_SOURCE_MANIFEST_SHA256,
        expected_source_map_sha256=HISTORICAL_SOURCE_MAP_SHA256,
        expected_source_count=HISTORICAL_SOURCE_COUNT,
    )


def _physical_timestamp(frame_id: int) -> float:
    from .quality_time import FrameKey, PHYSICAL
    timestamp = frame_id / 10.0
    return FrameKey(frame_id, timestamp, PHYSICAL).timestamp_s


def manifest_frame_timestamp(manifest: dict, row: dict) -> float:
    """Bind the selected ordinal to the authenticated physical timing table."""
    from .quality_time import PHYSICAL
    timing = manifest.get("_timing")
    if not isinstance(timing, dict) or timing.get("clock_mode") != PHYSICAL:
        raise ValueError("expanded capture requires authenticated physical frame timing")
    matches = [key for key in timing.get("scored_keys", ()) if key.frame_id == row["frame_id"]]
    if len(matches) != 1 or matches[0].timestamp_s != _physical_timestamp(row["frame_id"]):
        raise ValueError("selected manifest frame timestamp differs from the frozen frame_id/10 timeline")
    return matches[0].timestamp_s


def _validate_pin(value: dict, *, verify_files: bool) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError("input pin requires path, sha256 and bytes")
    require_hash(value["sha256"])
    raw_path = value["path"]
    if (not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path
            or type(value["bytes"]) is not int or value["bytes"] <= 0):
        raise ValueError("input pin must be an absolute nonsymlink path with positive bytes")
    if verify_files:
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("input pin must be an absolute nonsymlink path with positive bytes")
    else:
        # Metadata-only protocol validation must also work on Windows when a
        # frozen cloud plan names POSIX paths that are intentionally not mounted
        # or opened on this workstation.
        posix, windows = PurePosixPath(raw_path), PureWindowsPath(raw_path)
        if (not (posix.is_absolute() or windows.is_absolute())
                or ".." in posix.parts or ".." in windows.parts):
            raise ValueError("input pin must be lexically absolute with no parent traversal")
    if verify_files and (not path.is_file() or path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]):
        raise ValueError(f"input pin mismatch: {path}")


def validate_plan(document: dict, *, verify_files: bool = True) -> dict:
    expected_top = {"schema", "rows", "sealed_test", "max_seconds", "depth_tolerance_m", "native_raster_convention"}
    if (not isinstance(document, dict) or set(document) != expected_top
            or document["schema"] != PLAN_SCHEMA or type(document["max_seconds"]) is not int
            or document["max_seconds"] != AGGREGATE_SECONDS
            or document["native_raster_convention"] != RASTER_CONVENTION
            or isinstance(document["depth_tolerance_m"], bool)
            or not isinstance(document["depth_tolerance_m"], (int, float))
            or not np.isfinite(document["depth_tolerance_m"])
            or float(document["depth_tolerance_m"]) != .002):
        raise ValueError("expanded plan must use the frozen schema, 1800-second budget and .002m tolerance")
    rows = document["rows"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("expanded adaptation requires the exact 24-frame inventory")
    ids, masks = set(), set()
    offset = 0
    for split, obj, sequence, pitch in EXPECTED_ROWS:
        for frame_id in range(8):
            row = rows[offset]
            offset += 1
            if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                raise ValueError("expanded row fields differ from the frozen 14-field plan schema")
            expected_id = f"{sequence}:frame{frame_id}:iteration0"
            if (row["id"] != expected_id or row["split"] != split or row["object_id"] != obj
                    or row["sequence_id"] != sequence or row["pitch_deg"] != pitch
                    or row["yaw0_deg"] != 0 or type(row["frame_id"]) is not int
                    or row["frame_id"] != frame_id or row["timestamp_s"] != _physical_timestamp(frame_id)
                    or isinstance(row["timestamp_s"], bool)):
                raise ValueError("row differs from the frozen sequence/object/split/frame/timestamp assignment")
            for key in ("id", "sequence_id", "capture_group"):
                if not isinstance(row[key], str) or not row[key] or len(row[key]) > 256:
                    raise ValueError(f"invalid expanded plan {key}")
            if row["id"] in ids:
                raise ValueError("duplicate expanded row id")
            ids.add(row["id"])
            _validate_pin(row["input"], verify_files=verify_files)
            _validate_pin(row["mask_index"], verify_files=verify_files)
            _validate_pin(row["mask"], verify_files=verify_files)
            _validate_pin(row["bank"], verify_files=verify_files)
            _validate_pin(row["generation"], verify_files=verify_files)
            mask_identity = (row["sequence_id"], row["mask"]["path"], row["mask"]["sha256"])
            if mask_identity in masks:
                raise ValueError("mask pin is repeated across frames")
            masks.add(mask_identity)
    test = document["sealed_test"]
    if (not isinstance(test, dict) or set(test) != TEST_FIELDS or test["id"] != "test7-unmaterialized"
            or test["object_id"] != "7" or test["sequence_id"] != "sealed-test7-unmaterialized"
            or test["split"] != "test" or test["status"] != "unavailable"
            or test["query_rgb_sha256"] is not None or not isinstance(test["failure_reason"], str)
            or not test["failure_reason"] or not isinstance(test["unavailable_provenance"], str)
            or not test["unavailable_provenance"]):
        raise ValueError("TEST7 must remain metadata-only, unavailable, and without a payload pin")
    return document


def read_frozen_plan(path: Path, *, verify_files: bool = True) -> dict:
    if sha256(path) != FROZEN_PLAN_SHA256:
        raise ValueError("expanded capture plan differs from the parent-frozen protocol-v2 bytes")
    return validate_plan(read_json(path), verify_files=verify_files)


def _persist_frozen_plan(source: Path, root: Path) -> None:
    """Preserve authenticated input bytes for downstream byte-level checks."""
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_PLAN_SHA256:
        raise ValueError("expanded capture plan changed before persistence")
    with atomic_stream(root, root / "plan.json", max_file_bytes=len(raw),
                       max_output_bytes=PRIVATE_BYTES) as stream:
        stream.write(raw)


def read_frozen_label_plan(path: Path) -> dict:
    if sha256(path) != FROZEN_LABEL_PLAN_SHA256:
        raise ValueError("expanded synthetic-label plan differs from the separately frozen bytes")
    value = read_json(path)
    _validate_label_plan_shape(value)
    return value


def _base_row(row: dict) -> dict:
    return {key: row[key] for key in ("id", "object_id", "sequence_id", "capture_group", "split")}


def _expected_ids(plan: dict) -> list[str]:
    return [row["id"] for row in plan["rows"]]


def _index_by_id(rows: list[dict], expected_ids: list[str], name: str) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} rows must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in result:
            raise ValueError(f"{name} contains a malformed or duplicate row id")
        result[row["id"]] = row
    if set(result) != set(expected_ids) or len(rows) != len(expected_ids):
        raise ValueError(f"{name} inventory must join the exact 24 requested ids")
    return result


def validate_mask_binding(row: dict, mask_index: dict, masks_root: Path, mask_path: Path) -> None:
    frames = mask_index.get("frames")
    if not isinstance(frames, list):
        raise ValueError("cached mask index lacks frame inventory")
    matched = [item for item in frames if item.get("frameId") == row["frame_id"]]
    if len(matched) != 1:
        raise ValueError("cached mask index does not have one record for the requested frame")
    record = matched[0]
    expected_path = (masks_root / record.get("path", "")).resolve()
    recorded_hash = record.get("sha256", record.get("mask_sha256"))
    if (record.get("mask_state") != "available" or expected_path != mask_path.resolve()
            or not isinstance(recorded_hash, str) or recorded_hash.lower() != row["mask"]["sha256"]):
        raise ValueError("cached mask is unavailable or bound to a different frame/path/hash")


def capture(plan_path: Path, output: Path, *, cloud_execution: bool = False) -> dict:
    cloud_guard(cloud_execution)
    verify_capture_contract()
    plan = read_frozen_plan(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    _persist_frozen_plan(plan_path, output)
    rows = [dict(_base_row(row), status="unavailable", query_rgb_sha256=None,
                 failure_reason="not_processed", unavailable_provenance="frozen 24-row plan; no crop materialized")
            for row in plan["rows"]]
    write_json(output, "capture-index.json", {"rows": rows, "test_payload_reads": 0})
    started, source, versions, fatal = time.monotonic(), None, None, None
    network = None
    try:
        source = verify_sources()
        versions = runtime_versions()
        from .quality_adaptation_smoke import (
            RetainedCapture, artifact, legacy_native_intrinsics, observe_first_crop, save_npz,
        )
        from .quality_adaptation_prepare import crops_from_retained_capture
        import torch
        from .quality_gotrack import GoTrackRefiner, TexturedRenderer
        from .quality_foundpose import FoundPoseRecovery
        from .quality_contract import Frame
        from .quality_runner import read_input, read_rgb, _physical_mask_cache
        from .vision import cv2
        from .quality_time import PHYSICAL
        network = load_frozen_network("cpu")
        for ordinal, row in enumerate(plan["rows"]):
            renderer = None
            row_started = time.monotonic()
            try:
                if time.monotonic() - started >= plan["max_seconds"]:
                    raise TimeoutError("aggregate capture time budget exhausted")
                input_path = Path(row["input"]["path"])
                input_manifest = read_input(input_path.parent)
                if (input_manifest["sequence_id"] != row["sequence_id"]
                        or input_manifest["physical_object_id"] != f"hot3d:obj_{int(row['object_id']):06d}"
                        or input_manifest.get("provenance_kind") != "synthetic"):
                    raise ValueError("input bundle sequence/object/origin differs from frozen row")
                mask_index_path = Path(row["mask_index"]["path"])
                mask_index, _ = _physical_mask_cache(input_manifest, mask_index_path.parent)
                if mask_index.get("diagnostic_control") or mask_index.get("automatic") is False:
                    raise ValueError("expanded capture requires the pinned observed automatic mask cache")
                mask_path = Path(row["mask"]["path"])
                validate_mask_binding(row, mask_index, mask_index_path.parent, mask_path)
                rgb = read_rgb(input_path.parent, input_manifest, row["frame_id"])
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                    raise ValueError("input mask is missing, empty, or differs from frame RGB size")
                original_k = np.asarray(input_manifest["intrinsics"], np.float64)
                effective_k = legacy_native_intrinsics(original_k)
                timestamp = manifest_frame_timestamp(input_manifest, row)
                frame = Frame(row["frame_id"], rgb, effective_k, timestamp_s=timestamp, clock_mode=PHYSICAL)
                renderer = TexturedRenderer(input_path.parent / input_manifest["asset"], input_manifest["object_id"],
                    unlit=True, disable_multisampling=True, coordinate_mode="integer_centers_v1")
                sink = RetainedCapture()
                refiner = GoTrackRefiner(network, renderer, input_manifest["object_id"], "cpu",
                    chronological_capture_callback=sink)
                refiner.set_chronological_capture_context({"expanded_row": row["id"], "frame_id": row["frame_id"]})
                bank = torch.load(Path(row["bank"]["path"]), map_location="cpu", weights_only=False)
                backend = FoundPoseRecovery(refiner, bank, time_policy=input_manifest["_timing"]["time_policy"])
                seeds = backend.recover(frame, mask, top_k=5)  # Existing helper returns quality-sorted seeds.
                if not seeds:
                    raise ValueError("FoundPose produced no estimated seed; reference-pose fallback is forbidden")
                pnp_failure = None
                with observe_first_crop() as sampling_checks:
                    try:
                        result = refiner.refine(frame, mask, seeds[0])
                    except (ValueError, RuntimeError, cv2.error) as exc:
                        result = None
                        pnp_failure = f"{type(exc).__name__}: {exc}"[:1024]
                if sink.arrays is None:
                    raise ValueError(sink.failure or pnp_failure or "refiner emitted no retained iteration0 crop")
                if len(sampling_checks) != 1 or not np.array_equal(sink.arrays["native_k"], effective_k):
                    raise ValueError("actual retained crop sampling map or integer-center native K failed parity")
                crops = crops_from_retained_capture(sink.arrays, preserve_float_rgb=True)
                crop = save_npz(output, f"crop-{ordinal:02d}.npz", crops)
                geometry = save_npz(output, f"capture-{ordinal:02d}.npz", {
                    "crop_from_orig": np.asarray(sink.arrays["crop_from_orig"], np.float64),
                    "actual_flow": np.asarray(sink.arrays["full_flow_crop_px"], np.float32),
                    "actual_confidence": np.asarray(sink.arrays["full_confidence"], np.float32),
                })
                rows[ordinal] = dict(_base_row(row), status="available",
                    query_rgb_sha256=crop_rgb_hash(crops["query_rgb"]), crops=crop, capture=geometry,
                    seed_selection="first_FoundPose_quality_sorted_hypothesis", pnp_available=result is not None,
                    pnp_failure=pnp_failure, sampling_map=sampling_checks[0], timestamp_s=timestamp,
                    native_calibration={"original_K": original_k.tolist(), "effective_integer_K": effective_k.tolist(),
                        "convention": RASTER_CONVENTION, "input_manifest_unchanged": True,
                        "provenance": "historical legacy TexturedRenderer raster; no pose labels"},
                    elapsed_seconds=time.monotonic() - row_started)
            except Exception as exc:
                rows[ordinal]["failure_reason"] = f"{type(exc).__name__}: {exc}"[:1024]
            finally:
                if renderer is not None:
                    renderer.close()
                write_json(output, "capture-index.json", {"rows": rows, "test_payload_reads": 0})
    except Exception as exc:
        fatal = f"{type(exc).__name__}: {exc}"[:1024]
        for row in rows:
            if row.get("failure_reason") == "not_processed":
                row["failure_reason"] = fatal
        write_json(output, "capture-index.json", {"rows": rows, "test_payload_reads": 0})
    finally:
        if network is not None:
            del network
        receipt = {
            "schema": "gotrack_adaptation_expanded_capture_receipt_v1", "recipe": CAPTURE_RECIPE,
            "plan_sha256": canonical_hash(plan), "source": source, "versions": versions,
            "rows": rows, "requested": 24, "test_payload_reads": 0, "annotation_inputs": [],
            "elapsed_seconds": time.monotonic() - started, "fatal_error": fatal,
            "solver_note": "full ordinary refiner execution; iteration0 crop from first estimated FoundPose seed; PnP outcome recorded separately",
        }
        write_json(output, "capture-receipt.json", receipt)
        available_crops = [{"id": row["id"], "crops": row["crops"]} for row in rows if row["status"] == "available"]
        write_json(output, "crops.json", {"schema": "correspondence_input_crops_v1",
            "conventions": CONVENTION, "rows": available_crops})
    if fatal:
        raise RuntimeError(f"expanded capture failed before/during neural stage; full failure inventory persisted: {fatal}")
    return {"requested": 24, "available": sum(r["status"] == "available" for r in rows),
            "unavailable": sum(r["status"] != "available" for r in rows), "test_payload_reads": 0}


def _capture_and_feature_rows(root: Path, features: Path, plan: dict):
    capture_doc = read_json(root / "capture-index.json")
    feature_index = read_json(features / "index.json")
    capture_rows = _index_by_id(capture_doc.get("rows"), _expected_ids(plan), "capture")
    feature_rows = _index_by_id(feature_index.get("rows"), _expected_ids(plan), "extractor")
    return capture_doc, feature_index, capture_rows, feature_rows


def _check_parity(actual: dict, feature: dict, template_mask: np.ndarray) -> dict:
    if not isinstance(actual, dict) or not isinstance(feature, dict):
        raise ValueError("available extraction requires paired capture and feature archives")
    required_actual = {"actual_flow", "actual_confidence"}
    if not required_actual <= set(actual):
        raise ValueError("available capture is missing actual retained flow/confidence parity arrays")
    if not {"baseline_flow", "baseline_confidence"} <= set(feature):
        raise ValueError("available extraction is missing frozen baseline flow/confidence")
    af, ac = actual["actual_flow"], actual["actual_confidence"]
    ff, fc = feature["baseline_flow"], feature["baseline_confidence"]
    if (af.shape != (SIZE, SIZE, 2) or ac.shape != (SIZE, SIZE)
            or ff.shape != (2, SIZE, SIZE) or fc.shape != (SIZE, SIZE)
            or template_mask.shape != (SIZE, SIZE)):
        raise ValueError("actual and extracted flow/confidence arrays have incompatible shapes")
    if not all(np.isfinite(x).all() for x in (af, ac, ff, fc)):
        raise ValueError("actual/extracted flow/confidence contains nonfinite values")
    flow_error = float(np.abs(ff.transpose(1, 2, 0) * template_mask[..., None] - af).max())
    confidence_error = float(np.abs(fc * template_mask - ac).max())
    if flow_error > 1e-4 or confidence_error > 1e-6:
        raise ValueError("actual retained inference and generic extractor outputs differ")
    return {"max_flow_error_px": flow_error, "max_confidence_error": confidence_error}


def validate_retained_geometry(crop_from_orig: np.ndarray) -> np.ndarray:
    """Validate the retained crop camera transform without importing GoTrack/Torch."""
    transform = np.asarray(crop_from_orig, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0., atol=1e-6)):
        raise ValueError("retained crop transform must be a finite homogeneous 4x4 matrix")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-6)
            or abs(float(np.linalg.det(rotation)) - 1.) > 1e-6):
        raise ValueError("retained crop transform rotation must be proper")
    if not np.allclose(transform[:3, 3], 0., rtol=0., atol=1e-6):
        raise ValueError("retained crop and native cameras must share the optical center")
    return transform


def _validate_extracted_crop_binding(captured: dict, extracted: dict) -> None:
    if extracted.get("crops") != captured.get("crops"):
        raise ValueError("generic feature row does not reference its exact retained capture crop")


def _validate_capture_crop_manifest(root: Path, plan: dict, capture_rows: dict,
                                    recipe: dict, crop_manifest_path: Path | None = None) -> dict:
    crop_manifest_path = crop_manifest_path or root / "crops.json"
    document = read_json(crop_manifest_path)
    if (not isinstance(document, dict) or set(document) != {"schema", "conventions", "rows"}
            or document["schema"] != "correspondence_input_crops_v1"
            or document["conventions"] != CONVENTION):
        raise ValueError("capture crop manifest must use the exact input-only schema/conventions")
    expected_ids = _expected_ids(plan)
    crop_rows = _index_by_id(document["rows"], expected_ids, "capture crop manifest")
    for row in document["rows"]:
        if set(row) != {"id", "crops"}:
            raise ValueError("capture crop manifest rows must contain only an ID and crop descriptor")
    for row in plan["rows"]:
        captured = capture_rows[row["id"]]
        entry = crop_rows[row["id"]]
        if (captured.get("status") != "available" or entry["crops"] != captured.get("crops")):
            raise ValueError("capture crop manifest does not exactly join all available retained crop descriptors")
        descriptor(root, entry["crops"], verify=True)
    if recipe.get("crop_manifest_sha256") != canonical_hash(document):
        raise ValueError("generic extraction recipe does not bind the authenticated capture crop manifest")
    return document


def _verify_capture_feature_inventory(root: Path, features: Path, plan: dict):
    capture_doc, feature_index, capture_rows, feature_rows = _capture_and_feature_rows(root, features, plan)
    capture_receipt = read_json(root / "capture-receipt.json")
    receipt_rows = _index_by_id(capture_receipt.get("rows"), _expected_ids(plan), "capture receipt")
    if (capture_receipt.get("plan_sha256") != canonical_hash(plan) or capture_receipt.get("requested") != 24
            or capture_receipt.get("test_payload_reads") != 0 or capture_receipt.get("annotation_inputs") != []):
        raise ValueError("capture receipt does not bind the exact input-only 24-row plan")
    recipe = read_json(features / "recipe.json")
    if (recipe.get("schema") != "frozen_gotrack_decoder_recipe_v1" or recipe.get("annotation_inputs") != []
            or recipe.get("hooks") != list(HOOKS) or recipe.get("source_lock_sha256") != HISTORICAL_LOCK_SHA256):
        raise ValueError("generic extractor recipe is not the frozen input-only GoTrack recipe")
    _validate_capture_crop_manifest(root, plan, capture_rows, recipe)
    if feature_index.get("recipe_sha256") != canonical_hash(recipe):
        raise ValueError("generic extractor index does not bind its recipe")
    from .quality_adaptation_smoke import validate_frozen_feature_source, artifact
    compatibility = validate_frozen_feature_source(recipe)
    parity = []
    for row in plan["rows"]:
        captured, extracted, rec = capture_rows[row["id"]], feature_rows[row["id"]], receipt_rows[row["id"]]
        if captured.get("status") not in ("available", "unavailable") or extracted.get("status") not in ("available", "unavailable"):
            raise ValueError("capture/extraction row has an invalid status")
        if rec.get("status") != captured.get("status"):
            raise ValueError("capture receipt/index status mismatch")
        if rec != captured:
            raise ValueError("capture receipt row differs from the persisted capture index")
        available_capture_fields = set(_base_row(row)) | {
            "status", "query_rgb_sha256", "crops", "capture", "seed_selection", "pnp_available",
            "pnp_failure", "sampling_map", "timestamp_s", "native_calibration", "elapsed_seconds",
        }
        unavailable_capture_fields = set(_base_row(row)) | {
            "status", "query_rgb_sha256", "failure_reason", "unavailable_provenance",
        }
        expected_capture_fields = (available_capture_fields if captured["status"] == "available"
                                   else unavailable_capture_fields)
        if set(captured) != expected_capture_fields:
            raise ValueError("capture row fields do not match its availability state")
        if (extracted["status"] == "available"
                and set(extracted) != {"id", "status", "crops", "query_rgb_sha256", "features"}):
            raise ValueError("available feature row has unexpected/missing fields")
        if (extracted["status"] == "unavailable"
                and set(extracted) != {"id", "status", "failure_reason"}):
            raise ValueError("unavailable feature row has unexpected/missing fields")
        if captured["status"] == "available":
            if captured.get("timestamp_s") != row["timestamp_s"]:
                raise ValueError("captured physical timestamp differs from the frozen frame timeline")
            if (captured.get("seed_selection") != "first_FoundPose_quality_sorted_hypothesis"
                    or type(captured.get("pnp_available")) is not bool
                    or not isinstance(captured.get("pnp_failure"), (str, type(None)))
                    or not isinstance(captured.get("sampling_map"), dict)):
                raise ValueError("capture must retain iteration0 from first estimated seed and record PnP outcome separately")
            crop_path = descriptor(root, captured.get("crops"), verify=True)
            capture_path = descriptor(root, captured.get("capture"), verify=True)
            crops = load_npz(crop_path, CROP_KEYS)
            validate_crops(crops)
            if crop_rgb_hash(crops["query_rgb"]) != captured.get("query_rgb_sha256"):
                raise ValueError("capture crop content hash differs from retained inventory")
            actual = load_npz(capture_path, {"crop_from_orig", "actual_flow", "actual_confidence"})
            validate_retained_geometry(actual["crop_from_orig"])
            if (actual["crop_from_orig"].shape != (4, 4)
                    or actual["actual_flow"].shape != (SIZE, SIZE, 2)
                    or actual["actual_confidence"].shape != (SIZE, SIZE)
                    or not all(np.isfinite(value).all() for value in actual.values())):
                raise ValueError("available capture geometry/flow/confidence arrays are malformed")
            if extracted["status"] == "available":
                _validate_extracted_crop_binding(captured, extracted)
                if extracted.get("query_rgb_sha256") != captured["query_rgb_sha256"]:
                    raise ValueError("extracted query crop differs from retained capture")
                values = load_npz(descriptor(features, extracted.get("features"), verify=True), FEATURE_KEYS)
                validate_features(values)
                parity.append({"id": row["id"], **_check_parity(actual, values, values["template_mask"])})
        elif extracted["status"] == "available":
            raise ValueError("extractor reports an available feature for a failed capture row")
        if captured["status"] == "unavailable" and extracted["status"] == "available":
            raise ValueError("unavailable capture cannot join an available extraction")
    expected_ids = set(_expected_ids(plan))
    if ({key for key, value in capture_rows.items() if value["status"] == "available"} != expected_ids
            or {key for key, value in feature_rows.items() if value["status"] == "available"} != expected_ids
            or {value["id"] for value in parity} != expected_ids):
        raise ValueError("outer prelabel seal requires all 24 captures, features and parity rows")
    return capture_doc, feature_index, capture_rows, feature_rows, compatibility, parity, artifact


def seal_features(root: Path, features: Path, *, cloud_execution: bool = False) -> dict:
    cloud_guard(cloud_execution)
    verify_capture_contract()
    verify_sources()
    plan = read_frozen_plan(root / "plan.json")
    capture_doc, index, capture_rows, feature_rows, compatibility, parity, artifact = _verify_capture_feature_inventory(root, features, plan)
    if (set(capture_doc) - {"rows", "test_payload_reads"} or capture_doc.get("test_payload_reads") != 0
            or index.get("status") not in ("completed", "incomplete_time_limit", None)):
        raise ValueError("capture/extractor top-level metadata contains unexpected fields or test access")
    receipt = {
        "schema": "gotrack_adaptation_expanded_prelabel_seal_v1",
        "capture_receipt": artifact(root, root / "capture-receipt.json"),
        "plan": artifact(root, root / "plan.json"),
        "crop_index": artifact(root, root / "capture-index.json"),
        "feature_index_sha256": sha256(features / "index.json"),
        "feature_recipe_sha256": sha256(features / "recipe.json"),
        "feature_source_compatibility": compatibility,
        "parity": parity, "available_captures": sum(r["status"] == "available" for r in capture_rows.values()),
        "available_features": sum(r["status"] == "available" for r in feature_rows.values()),
        "annotation_inputs": [], "test_payload_reads": 0,
    }
    write_json(root, "prelabel-seal.json", receipt)
    return receipt


def verify_prelabel_seal(root: Path, features: Path, plan: dict) -> dict:
    seal = read_json(root / "prelabel-seal.json")
    for key, path in (("capture_receipt", root / "capture-receipt.json"),
                      ("plan", root / "plan.json"), ("crop_index", root / "capture-index.json")):
        if seal.get(key) != {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}:
            raise ValueError(f"prelabel seal {key} hash/size changed")
    if (seal.get("feature_index_sha256") != sha256(features / "index.json")
            or seal.get("feature_recipe_sha256") != sha256(features / "recipe.json")
            or seal.get("test_payload_reads") != 0 or seal.get("annotation_inputs") != []):
        raise ValueError("prelabel feature/index/recipe seal changed")
    # Re-open and authenticate every available capture/crop/feature and repeat
    # parity before the separately pinned label recipe/split is ever read.
    _, _, _, _, _, parity, _ = _verify_capture_feature_inventory(root, features, plan)
    if parity != seal.get("parity"):
        raise ValueError("prelabel seal parity rows differ from current authenticated artifacts")
    return seal


def read_label_plan_after_seal(root: Path, features: Path, plan: dict, label_plan_path: Path) -> tuple[dict, dict]:
    seal = verify_prelabel_seal(root, features, plan)
    if sha256(label_plan_path) != FROZEN_LABEL_PLAN_SHA256:
        raise ValueError("label plan bytes differ from the frozen synthetic-label plan")
    return read_json(label_plan_path), seal


def _validate_label_plan(label_plan: dict) -> tuple[Path, Path]:
    if (not isinstance(label_plan, dict) or set(label_plan) != {"schema", "recipe", "split_manifest"}
            or label_plan["schema"] != "gotrack_adaptation_synthetic_labels_v1"):
        raise ValueError("label plan must contain only separately pinned synthetic recipe and split metadata")
    from .quality_adaptation_smoke import pinned_file
    return pinned_file(label_plan["recipe"]), pinned_file(label_plan["split_manifest"])


def labels(plan_path: Path, label_plan_path: Path, root: Path, features: Path,
           *, cloud_execution: bool = False) -> dict:
    cloud_guard(cloud_execution)
    verify_capture_contract()
    verify_sources()
    plan = read_frozen_plan(plan_path)
    if canonical_hash(read_json(root / "plan.json")) != canonical_hash(plan):
        raise ValueError("capture plan differs from the prospectively frozen label plan")
    # This call authenticates the entire pre-label closure before reading any
    # label recipe, split, pose file, or supervision bytes.
    label_plan, seal = read_label_plan_after_seal(root, features, plan, label_plan_path)
    recipe_path, split_path = _validate_label_plan(label_plan)
    capture_rows, feature_rows = _capture_and_feature_rows(root, features, plan)[2:]
    from .quality_ranker_synthetic import validate_recipe
    from .quality_adaptation_prepare import project_supervision
    from .quality_adaptation_manifest import assemble
    from .quality_gotrack import TexturedRenderer
    from .quality_runner import read_input
    from .quality_adaptation_smoke import (
        _base_row as smoke_base_row, label_pose, render_query_labels, save_npz, validate_generation_context,
    )
    inventory = []
    for row in plan["rows"]:
        source = capture_rows[row["id"]]
        feature = feature_rows[row["id"]]
        reason = source.get("failure_reason") or feature.get("failure_reason") or "label_not_processed"
        item = dict(_base_row(row), status="unavailable", query_rgb_sha256=source.get("query_rgb_sha256"),
                    failure_reason=str(reason)[:1024])
        if item["query_rgb_sha256"] is None:
            item["unavailable_provenance"] = source.get("unavailable_provenance", "frozen row; no crop materialized")
        inventory.append(item)

    def persist_inventory():
        write_json(features, "labels.json", {"schema": "correspondence_label_inventory_v1",
            "label_origin": "synthetic_renderer_training_only", "rows": inventory + [plan["sealed_test"]]})

    persist_inventory()
    try:
        from .quality_ranker_synthetic import validate_recipe
        for ordinal, row in enumerate(plan["rows"]):
            source, feature = capture_rows[row["id"]], feature_rows[row["id"]]
            renderer = None
            try:
                if source["status"] != "available" or feature["status"] != "available":
                    continue
                crops = load_npz(descriptor(root, source["crops"], verify=True), CROP_KEYS)
                validate_crops(crops)
                geometry = load_npz(descriptor(root, source["capture"], verify=True),
                                    {"crop_from_orig", "actual_flow", "actual_confidence"})
                input_path = Path(row["input"]["path"])
                manifest = read_input(input_path.parent)
                context = validate_recipe(recipe_path, split_path, input_path.parent / manifest["asset"],
                    row["sequence_id"], frames=24, width=640, height=480, fps=10.,
                    mesh_units="metres", observationally_ambiguous=True)
                generation_path = Path(row["generation"]["path"])
                generation = read_json(generation_path)
                validate_generation_context(context, generation, manifest, row, sha256(split_path))
                renderer = TexturedRenderer(input_path.parent / manifest["asset"], manifest["object_id"],
                    unlit=True, disable_multisampling=True, coordinate_mode="integer_centers_v1")
                native_pose = label_pose(renderer.vertices_m, row)
                query_pose, depth, support = render_query_labels(renderer, object_id=manifest["object_id"],
                    crop_k=crops["query_K"], crop_from_orig=geometry["crop_from_orig"], native_pose=native_pose)
                target = project_supervision(template_xyz_m=crops["template_xyz_m"],
                    template_mask=crops["template_mask"], query_K=crops["query_K"],
                    query_T_cam_from_object=query_pose, observed_depth_m=depth,
                    observed_object_mask=support, depth_tolerance_m=plan["depth_tolerance_m"])
                label_artifact = save_npz(features, f"labels-{ordinal:02d}.npz", target)
                inventory[ordinal] = dict(_base_row(row), status="available",
                    query_rgb_sha256=source["query_rgb_sha256"], labels=label_artifact)
            except Exception as exc:
                inventory[ordinal]["failure_reason"] = f"{type(exc).__name__}: {exc}"[:1024]
            finally:
                if renderer is not None:
                    renderer.close()
                persist_inventory()
        write_json(features, "label-receipt.json", {
            "schema": "gotrack_adaptation_expanded_labels_receipt_v1",
            "prelabel_seal_sha256": sha256(root / "prelabel-seal.json"),
            "label_plan_sha256": sha256(label_plan_path), "synthetic_observed_support": True,
            "test_payload_reads": 0, "labels_entered_inference": False, "quality_claim": False,
            "feature_source_compatibility": seal["feature_source_compatibility"],
        })
    finally:
        persist_inventory()
    assembled = assemble(features, features / "labels.json", features / "manifest.json")
    validate_expanded_manifest(features / "manifest.json", plan, require_complete=False)
    return assembled


def validate_expanded_manifest(manifest_path: Path, plan: dict, *, require_complete: bool) -> dict:
    ds = load_dataset(manifest_path, inspect_arrays=False)
    rows = _index_by_id(ds.document["rows"], _expected_ids(plan) + [plan["sealed_test"]["id"]], "dataset")
    for row in plan["rows"]:
        actual = rows[row["id"]]
        if any(actual.get(key) != row[key] for key in ("object_id", "sequence_id", "capture_group", "split")):
            raise ValueError("assembled manifest identity differs from the exact capture plan")
    test = rows[plan["sealed_test"]["id"]]
    expected_test = plan["sealed_test"]
    if (test["split"] != "test" or test["object_id"] != "7" or test["status"] != "unavailable"
            or test["sequence_id"] != expected_test["sequence_id"]
            or test["capture_group"] != expected_test["capture_group"]
            or test["query_rgb_sha256"] is not None):
        raise ValueError("TEST7 manifest row must remain unavailable and payload-free")
    counts = {split: (len(ds.rows(split, available=False)), len(ds.rows(split))) for split in ("train", "val", "test")}
    if {split: total for split, (total, _) in counts.items()} != {"train": 16, "val": 8, "test": 1}:
        raise ValueError("expanded dataset split inventory differs from 16 TRAIN, 8 VAL and metadata-only TEST7")
    if require_complete and counts != {"train": (16, 16), "val": (8, 8), "test": (1, 0)}:
        raise ValueError("optimization requires all 16 TRAIN and 8 VAL rows; no postselection is permitted")
    return {"dataset_sha256": ds.identity, "splits": {
        split: {"total": total, "available": available} for split, (total, available) in counts.items()},
        "test_payload_reads": 0}


def _expanded_config(config_path: Path):
    from .quality_adaptation_train import read_config
    contract = verify_capture_contract()
    if sha256(config_path) != contract["config_sha256"]:
        raise ValueError("training config bytes differ from the separately hash-bound expanded config")
    cfg = read_config(config_path)
    expected = {
        "seed": 42, "epochs": 1, "batch_size": 1, "learning_rate": 1e-5,
        "weight_decay": 1e-4, "patience": 3, "checkpoint_every_steps": 8,
        "max_steps": 16, "device": "cpu", "amp": False,
    }
    if any(getattr(cfg, key) != value for key, value in expected.items()):
        raise ValueError("expanded config differs from the frozen one-epoch 16-step CPU recipe")
    if (cfg.max_total_seconds > AGGREGATE_SECONDS or cfg.max_seconds > AGGREGATE_SECONDS
            or cfg.max_output_bytes != 1000 * 1024**2 or cfg.max_checkpoint_bytes != 320 * 1024**2):
        raise ValueError("expanded config exceeds aggregate runtime/private output budget")
    return cfg


def _training_preflight(plan_path: Path, manifest_path: Path, config_path: Path, *, require_complete=True):
    verify_capture_contract()
    verify_sources()
    plan = read_frozen_plan(plan_path)
    cfg = _expanded_config(config_path)
    data = validate_expanded_manifest(manifest_path, plan, require_complete=require_complete)
    return plan, cfg, data


def train_stage(plan_path: Path, manifest_path: Path, config_path: Path, output: Path,
                *, cloud_execution: bool, resume: bool = False, stop_after_steps: int | None = None) -> dict:
    cloud_guard(cloud_execution)
    plan, cfg, data = _training_preflight(plan_path, manifest_path, config_path, require_complete=True)
    from .quality_adaptation_train import train as generic_train
    result = generic_train(manifest_path, output, cfg, cloud_execution=True, resume=resume,
                           stop_after_steps=stop_after_steps)
    return {**result, "dataset": data, "plan_sha256": canonical_hash(plan), "config_sha256": sha256(config_path)}


def inspect_interruption(plan_path: Path, manifest_path: Path, config_path: Path,
                         run: Path, *, cloud_execution: bool) -> dict:
    cloud_guard(cloud_execution)
    _, cfg, data = _training_preflight(plan_path, manifest_path, config_path, require_complete=True)
    import torch
    from .quality_adaptation_train import epoch_batches
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    order = epoch_batches(load_dataset(manifest_path, inspect_arrays=False).rows("train"), 0, cfg)
    expected_next = [r["id"] for r in order[8]]
    if (state.get("steps") != 8 or state.get("epoch") != 0 or state.get("batch_cursor") != 8
            or state.get("completed") or not state.get("optimizer", {}).get("state")
            or state.get("last_step", {}).get("row_ids") != [r["id"] for r in order[7]]):
        raise ValueError("step-8 checkpoint lost deterministic cursor or Adam state")
    receipt = {"steps": 8, "epoch": 0, "batch_cursor": 8, "completed": False,
        "optimizer_state_entries": len(state["optimizer"]["state"]),
        "optimizer_sha256": training_state_digest(state["optimizer"]),
        "head_sha256": _state_digest(state["head"]), "checkpoint_sha256": sha256(run / "last.pt"),
        "last_step": state["last_step"], "next_row_ids": expected_next,
        "dataset_sha256": data["dataset_sha256"], "test_payload_reads": 0}
    write_json(run, "interruption.json", receipt)
    return receipt


def _state_digest(state: dict) -> str:
    from .quality_adaptation_smoke import state_digest
    return state_digest(state)


def export_stage(plan_path: Path, manifest_path: Path, config_path: Path, run: Path,
                 destination: Path, *, cloud_execution: bool) -> dict:
    cloud_guard(cloud_execution)
    _training_preflight(plan_path, manifest_path, config_path, require_complete=True)
    from .quality_adaptation_train import export as generic_export
    return generic_export(run, destination, cloud_execution=True)


def _sequence_metrics(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row["sequence_id"], {"frames": 0, "epe": [], "visibility": [], "visible_pixels": 0, "known_pixels": 0})
        group["frames"] += 1
        group["epe"].append(row["flow"]["epe_px"])
        group["visibility"].append(row["visibility"]["bce"])
        group["visible_pixels"] += row["flow"]["visible_pixels"]
        group["known_pixels"] += row["visibility"]["known_pixels"]
    return {sequence: {"frames": value["frames"], "mean_frame_epe_px": float(np.mean(value["epe"])),
                       "mean_frame_visibility_bce": float(np.mean(value["visibility"])),
                       "fixed_visible_support_pixels": value["visible_pixels"],
                       "fixed_known_support_pixels": value["known_pixels"]}
            for sequence, value in sorted(grouped.items())}


def _metrics_for_prediction(flow: np.ndarray, confidence: np.ndarray, labels: dict) -> dict:
    return {"flow": flow_metrics(flow.astype(np.float32), labels["flow"], labels["visible"]),
            "visibility": visibility_metrics(confidence.astype(np.float32), labels)}


def verify_export(plan_path: Path, manifest_path: Path, config_path: Path, run: Path,
                  export_path: Path, root: Path, *, cloud_execution: bool) -> dict:
    cloud_guard(cloud_execution)
    plan, cfg, data = _training_preflight(plan_path, manifest_path, config_path, require_complete=True)
    source = verify_sources()
    versions = runtime_versions()
    import torch
    from .quality_adaptation_smoke import artifact, state_digest
    started = time.monotonic()
    ds = load_dataset(manifest_path)
    train_state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    best_state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    status = read_json(run / "status.json")
    interrupted = read_json(run / "interruption.json")
    from .quality_adaptation_train import epoch_batches
    train_order = epoch_batches(ds.rows("train"), 0, cfg)
    restored = status.get("resumed_from") or {}
    if (not train_state.get("completed") or train_state.get("steps") != 16 or train_state.get("epoch") != 1
            or status.get("status") != "completed" or best_state.get("dataset_sha256") != ds.identity
            or interrupted.get("steps") != 8 or interrupted.get("batch_cursor") != 8
            or restored.get("steps") != 8 or restored.get("batch_cursor") != 8
            or restored.get("optimizer_state_entries") != interrupted.get("optimizer_state_entries")
            or restored.get("optimizer_sha256") != interrupted.get("optimizer_sha256")
            or training_state_digest(train_state["optimizer"]) == interrupted.get("optimizer_sha256")
            or train_state.get("last_step", {}).get("row_ids") != [r["id"] for r in train_order[15]]
            or not train_state.get("optimizer", {}).get("state")
            or any(float(slot["step"]) != 16. for slot in train_state["optimizer"]["state"].values())
            or state_digest(train_state["head"]) == interrupted.get("head_sha256")):
        raise ValueError("expanded training did not complete the fixed 16-step one-epoch run")
    cfg_hash = canonical_hash(asdict(cfg))
    value = torch.load(export_path, map_location="cpu", weights_only=True)
    if (value.get("schema") != "correspondence_head_export_v1"
            or value.get("source_lock_sha256") != sha256(LOCK)
            or value.get("base_checkpoint_sha256") != "f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086"
            or value.get("dataset_sha256") != ds.identity or value.get("config_sha256") != cfg_hash):
        raise ValueError("export header differs from the exact expanded dataset/config/source")
    standalone = load_head(initialize=False, device="cpu")
    standalone.load_state_dict(value["head"], strict=True)
    standalone.eval()
    network = load_frozen_network("cpu")
    frozen_before = {"backbone": state_digest(network.backbone.state_dict()),
                     "decoder": state_digest(network.decoder.state_dict())}
    base_head = state_digest(network.pose_head.state_dict())
    network = load_export_into_network(network, export_path, expected_sha256=sha256(export_path), cloud_execution=True)
    frozen_after = {"backbone": state_digest(network.backbone.state_dict()),
                    "decoder": state_digest(network.decoder.state_dict())}
    if frozen_before != frozen_after or state_digest(network.pose_head.state_dict()) == base_head:
        raise ValueError("fresh export load changed frozen weights or did not adapt pose head")
    capture_rows = {r["id"]: r for r in read_json(root / "capture-index.json")["rows"]}
    checks, frozen_metrics, adapted_metrics = [], [], []
    with torch.inference_mode():
        for row in ds.rows("val"):
            f, y = ds.arrays(row)
            tokens = [None] * 12
            for hook in HOOKS:
                tokens[hook] = torch.from_numpy(f[f"tokens_{hook}"][None])
            flow, confidence = standalone(tokens, (SIZE, SIZE))
            actual = network.pose_head(tokens, (SIZE, SIZE))
            head_errors = [float((a - b).abs().max()) for a, b in zip(actual, (flow, confidence))]
            if any(not torch.isfinite(a).all() or not torch.allclose(a, b, atol=1e-6, rtol=0)
                   for a, b in zip(actual, (flow, confidence))):
                raise ValueError("fresh full-network head differs from standalone export on cached validation tokens")
            crop_row = capture_rows[row["id"]]
            crops = load_npz(descriptor(root, crop_row["crops"], verify=True), CROP_KEYS)
            images = np.stack([crops["query_rgb"], crops["template_rgb"]]).transpose(0, 3, 1, 2).copy()
            if images.dtype == np.uint8:
                rgb = torch.from_numpy(images.astype(np.float32) / 255.)
            else:
                rgb = torch.from_numpy(images.astype(np.float32))
            mask = torch.from_numpy(crops["template_mask"][None])
            online = network(rgb[:1], rgb[1:], mask)
            expected_online = (flow * mask[:, None], confidence * mask)
            rgb_errors = [float((a - b).abs().max()) for a, b in zip(online, expected_online)]
            if any(not torch.isfinite(a).all() or not torch.allclose(a, b, atol=1e-4, rtol=0)
                   for a, b in zip(online, expected_online)):
                raise ValueError("fresh full RGB inference differs from cached-token export output")
            adapted = _metrics_for_prediction(flow[0].float().cpu().numpy(),
                confidence[0].float().cpu().numpy(), y)
            frozen = _metrics_for_prediction(f["baseline_flow"], f["baseline_confidence"], y)
            adapted_metrics.append({"id": row["id"], "sequence_id": row["sequence_id"], **adapted})
            frozen_metrics.append({"id": row["id"], "sequence_id": row["sequence_id"], **frozen})
            checks.append({"id": row["id"], "cached_head_max_flow_error_px": head_errors[0],
                "cached_head_max_confidence_error": head_errors[1],
                "rgb_max_flow_error_px": rgb_errors[0], "rgb_max_confidence_error": rgb_errors[1]})
    before_bad_load = state_digest(network.pose_head.state_dict())
    try:
        load_export_into_network(network, export_path, expected_sha256="0" * 64, cloud_execution=True)
    except ValueError as exc:
        if "hash mismatch" not in str(exc):
            raise
    else:
        raise ValueError("wrong export digest unexpectedly loaded")
    if before_bad_load != state_digest(network.pose_head.state_dict()):
        raise ValueError("rejected export digest mutated the loaded head")
    receipt = {
        "schema": "gotrack_adaptation_expanded_export_verify_v1", "source": source, "versions": versions,
        "plan_sha256": canonical_hash(plan), "dataset_sha256": ds.identity,
        "export": artifact(run, export_path), "trainable_parameters": sum(p.numel() for p in standalone.parameters()),
        "frozen_weights_before": frozen_before, "frozen_weights_after": frozen_after,
        "head_changed_from_base": True, "steps": train_state["steps"], "all_adam_steps": 16,
        "resume": status.get("resumed_from"), "parity": checks,
        "bad_export_digest_rejected_without_mutation": True,
        "paired_fixed_support_by_sequence": {"evaluation_split": "val",
            "frozen": _sequence_metrics(frozen_metrics), "adapted": _sequence_metrics(adapted_metrics)},
        "availability": data["splits"], "test_payload_reads": 0, "quality_claim": False,
        "mobile_ready": False, "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(), "run_bytes": tree_bytes(run),
        "private_artifact_budget_bytes": PRIVATE_BYTES,
        "uncertainty_note": "one TRAIN object and one already-inspected VAL object; frames are dependent; no population interval",
    }
    write_json(run, "expanded-verify.json", receipt)
    return receipt


def _peak_rss_bytes() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _copy_feature_tree(root: Path, features: Path, output: Path, plan: dict) -> dict:
    verify_capture_contract()
    verify_sources()
    seal_features(root, features, cloud_execution=True)
    _, index, _, rows, _, _, _ = _verify_capture_feature_inventory(root, features, plan)
    files = {"recipe.json": features / "recipe.json", "index.json": features / "index.json"}
    for row in rows.values():
        if row["status"] == "available":
            artifact_path = descriptor(features, row["features"], verify=True)
            files[row["features"]["path"]] = artifact_path
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for relative, source in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_stream(output, target, max_file_bytes=source.stat().st_size, max_output_bytes=PRIVATE_BYTES) as stream:
            stream.write(source.read_bytes())
        if sha256(target) != sha256(source):
            raise ValueError("reused generic feature artifact hash changed")
        copied.append({"path": relative, "bytes": target.stat().st_size, "sha256": sha256(target)})
    return {"files": copied, "source_recipe_sha256": canonical_hash(read_json(features / "recipe.json")),
            "inference_repeated": False, "deadline_reset": False, "test_payload_reads": 0}


def _check_aggregate(elapsed_seconds: float, output_bytes: int) -> None:
    if elapsed_seconds >= AGGREGATE_SECONDS:
        raise TimeoutError("aggregate adaptation watchdog exhausted")
    if output_bytes > PRIVATE_BYTES:
        raise ValueError("expanded adaptation output exceeded 1.5 GiB including temporary files")


def _assert_frozen_invocation(plan_path: Path, label_plan_path: Path, config_path: Path,
                              frozen: dict) -> None:
    verify_capture_contract()
    current = {"plan_sha256": sha256(plan_path), "label_plan_sha256": sha256(label_plan_path),
               "config_sha256": sha256(config_path), "contract_sha256": sha256(CONTRACT_PATH)}
    if current != frozen:
        raise ValueError("expanded workflow plan, label pins, config, or source contract changed during execution")


def _workflow_commands(plan_path: Path, label_plan_path: Path, config_path: Path,
                       output: Path) -> list[tuple[str, list[str]]]:
    capture_root, features, run = output / "capture", output / "features", output / "run"
    return [
        ("capture", [sys.executable, "-m", "bench.quality_adaptation_expanded", "capture",
            "--plan", str(plan_path), "--root", str(capture_root), "--cloud-execution"]),
        ("extract", [sys.executable, "-m", "bench.quality_adaptation_expanded", "extract",
            "--root", str(capture_root), "--output", str(features), "--cloud-execution"]),
        ("seal", [sys.executable, "-m", "bench.quality_adaptation_expanded", "seal",
            "--root", str(capture_root), "--features", str(features), "--cloud-execution"]),
        ("labels", [sys.executable, "-m", "bench.quality_adaptation_expanded", "labels",
            "--plan", str(plan_path), "--label-plan", str(label_plan_path), "--root", str(capture_root),
            "--features", str(features), "--cloud-execution"]),
        ("train_step8", [sys.executable, "-m", "bench.quality_adaptation_expanded", "train",
            "--plan", str(plan_path), "--manifest", str(features / "manifest.json"), "--config", str(config_path),
            "--run", str(run), "--stop-after-steps", "8", "--cloud-execution"]),
        ("inspect_step8", [sys.executable, "-m", "bench.quality_adaptation_expanded", "interruption",
            "--plan", str(plan_path), "--manifest", str(features / "manifest.json"), "--config", str(config_path),
            "--run", str(run), "--cloud-execution"]),
        ("resume", [sys.executable, "-m", "bench.quality_adaptation_expanded", "resume",
            "--plan", str(plan_path), "--manifest", str(features / "manifest.json"), "--config", str(config_path),
            "--run", str(run), "--cloud-execution"]),
        ("export", [sys.executable, "-m", "bench.quality_adaptation_expanded", "export",
            "--plan", str(plan_path), "--manifest", str(features / "manifest.json"), "--config", str(config_path),
            "--run", str(run), "--output", str(run / "head.pt"), "--cloud-execution"]),
        ("verify", [sys.executable, "-m", "bench.quality_adaptation_expanded", "verify",
            "--plan", str(plan_path), "--manifest", str(features / "manifest.json"), "--config", str(config_path),
            "--run", str(run), "--export", str(run / "head.pt"), "--root", str(capture_root), "--cloud-execution"]),
    ]


def _run_workflow(plan_path: Path, label_plan_path: Path, config_path: Path, output: Path,
                  *, cloud_execution: bool) -> dict:
    cloud_guard(cloud_execution)
    started = time.monotonic()  # Includes all source/input/config preflight below.
    verify_capture_contract()
    plan = read_frozen_plan(plan_path)
    label_plan = read_frozen_label_plan(label_plan_path)
    cfg = _expanded_config(config_path)
    verify_sources()
    runtime_versions()
    free = shutil.disk_usage(output.parent.resolve()).free
    if free < PRIVATE_BYTES + DISK_RESERVE_BYTES:
        raise OSError("expanded adaptation requires 1.5 GiB output headroom plus a 1 GiB free-disk reserve")
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("private expanded adaptation output must be outside the 256 MiB source packet")
    output.mkdir(parents=True, exist_ok=False)
    frozen = {"plan_sha256": sha256(plan_path), "label_plan_sha256": sha256(label_plan_path),
              "config_sha256": sha256(config_path), "contract_sha256": sha256(CONTRACT_PATH)}
    _assert_frozen_invocation(plan_path, label_plan_path, config_path, frozen)
    write_json(output, "invocation.json", {**frozen, "aggregate_max_seconds": AGGREGATE_SECONDS,
        "private_artifact_budget_bytes": PRIVATE_BYTES, "minimum_free_disk_reserve_bytes": DISK_RESERVE_BYTES,
        "test_payload_reads": 0})
    capture_root, features, run, logs = output / "capture", output / "features", output / "run", output / "logs"
    logs.mkdir()
    commands = _workflow_commands(plan_path, label_plan_path, config_path, output)
    from .quality_offline import run_stage
    stages = []
    try:
        for name, command in commands:
            _assert_frozen_invocation(plan_path, label_plan_path, config_path, frozen)
            if name == "train_step8":
                non_run_bytes = tree_bytes(output) - tree_bytes(run)
                reserve = PRIVATE_BYTES - cfg.max_output_bytes - 16 * 1024**2
                if non_run_bytes > reserve:
                    raise ValueError("pre-training capture/features/labels exceed the 320 MiB non-run budget")
            elapsed = time.monotonic() - started
            _check_aggregate(elapsed, tree_bytes(output))
            cleanup_reserve = 30
            remaining = AGGREGATE_SECONDS - elapsed - cleanup_reserve
            if remaining <= 0:
                raise TimeoutError("aggregate watchdog left no cleanup and receipt reserve")
            try:
                receipt = run_stage(name, command, logs, timeout_s=remaining,
                    storage_limit_bytes=PRIVATE_BYTES, storage_root=output, output_limit_bytes=16 * 1024**2)
            except Exception:
                failed_receipt_path = logs / f"{name}.receipt.json"
                if failed_receipt_path.is_file():
                    stages.append(read_json(failed_receipt_path))
                write_json(output, "stages.json", {"stages": stages,
                    "elapsed_seconds": time.monotonic() - started,
                    "private_artifact_bytes": tree_bytes(output), "aggregate_max_seconds": AGGREGATE_SECONDS,
                    "private_artifact_budget_bytes": PRIVATE_BYTES, "test_payload_reads": 0})
                raise
            _assert_frozen_invocation(plan_path, label_plan_path, config_path, frozen)
            stages.append(receipt)
            write_json(output, "stages.json", {"stages": stages, "elapsed_seconds": time.monotonic() - started,
                "private_artifact_bytes": tree_bytes(output), "test_payload_reads": 0})
        result = read_json(run / "expanded-verify.json")
        _assert_frozen_invocation(plan_path, label_plan_path, config_path, frozen)
        result.update(total_elapsed_seconds=time.monotonic() - started,
                      private_artifact_bytes=tree_bytes(output), stages=stages, frozen_invocation=frozen)
        _check_aggregate(result["total_elapsed_seconds"], result["private_artifact_bytes"])
        write_json(output, "result.json", result)
        return result
    finally:
        write_json(output, "stages.json", {"stages": stages, "elapsed_seconds": time.monotonic() - started,
            "private_artifact_bytes": tree_bytes(output), "aggregate_max_seconds": AGGREGATE_SECONDS,
            "private_artifact_budget_bytes": PRIVATE_BYTES, "test_payload_reads": 0})


def _validate_label_plan_shape(label_plan: dict) -> None:
    if (not isinstance(label_plan, dict) or set(label_plan) != {"schema", "recipe", "split_manifest"}
            or label_plan["schema"] != "gotrack_adaptation_synthetic_labels_v1"):
        raise ValueError("label plan must use the fixed separate recipe/split pins")
    _validate_pin(label_plan["recipe"], verify_files=False)
    _validate_pin(label_plan["split_manifest"], verify_files=False)


def _persist_extraction_failure(root: Path, features: Path, plan: dict, capture_rows: dict,
                                reason: str) -> None:
    """Keep an exact 24-ID feature failure inventory if extraction stops early."""
    index_path = features / "index.json"
    existing = {}
    if index_path.is_file():
        try:
            partial_rows = read_json(index_path).get("rows")
            if isinstance(partial_rows, list):
                expected = set(_expected_ids(plan))
                for partial in partial_rows:
                    if (not isinstance(partial, dict) or not isinstance(partial.get("id"), str)
                            or partial["id"] not in expected or partial["id"] in existing):
                        raise ValueError("partial extractor inventory has duplicate or unexpected IDs")
                    existing[partial["id"]] = partial
        except (ValueError, OSError, TypeError):
            existing = {}
    rows = []
    for row in plan["rows"]:
        partial = existing.get(row["id"])
        if (isinstance(partial, dict) and partial.get("id") == row["id"]
                and partial.get("status") == "available"):
            rows.append(partial)
        else:
            failure = partial.get("failure_reason") if isinstance(partial, dict) else None
            if not failure:
                failure = capture_rows[row["id"]].get("failure_reason") or reason
            rows.append({"id": row["id"], "status": "unavailable",
                         "failure_reason": str(failure)[:1024]})
    features.mkdir(parents=True, exist_ok=True)
    if index_path.is_file():
        try:
            metadata = read_json(index_path)
            if not isinstance(metadata, dict):
                metadata = {}
        except (ValueError, OSError, TypeError):
            metadata = {}
    else:
        metadata = {}
    metadata.update(status="failed", rows=rows,
        expanded_capture_sha256=sha256(root / "capture-index.json"),
        expanded_contract_sha256=sha256(CONTRACT_PATH), test_payload_reads=0)
    write_json(features, "index.json", metadata, max_output_bytes=512 * 1024**2)
    write_json(features, "failure-receipt.json", {
        "schema": "gotrack_adaptation_expanded_extraction_failure_v1",
        "reason": reason[:1024], "requested": 24,
        "available_features": sum(row["status"] == "available" for row in rows),
        "rows": rows, "capture_index_sha256": sha256(root / "capture-index.json"),
        "test_payload_reads": 0,
    }, max_output_bytes=512 * 1024**2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "extract", "seal", "reuse-features", "labels", "train", "resume",
                 "interruption", "export", "verify", "run"):
        p = sub.add_parser(name)
        p.add_argument("--cloud-execution", action="store_true")
        if name in ("capture", "run", "train", "resume", "interruption", "export", "verify", "labels"):
            p.add_argument("--plan", type=Path, required=True)
        if name in ("run", "labels"):
            p.add_argument("--label-plan", type=Path, required=True)
        if name in ("capture", "extract", "seal", "reuse-features", "labels", "verify"):
            p.add_argument("--root", type=Path, required=True)
        if name in ("seal", "reuse-features", "labels"):
            p.add_argument("--features", type=Path, required=True)
        if name in ("train", "resume", "interruption", "export", "verify"):
            p.add_argument("--manifest", type=Path, required=True)
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--run", type=Path, required=True)
        if name == "train":
            p.add_argument("--stop-after-steps", type=int)
        if name == "export":
            p.add_argument("--output", type=Path, required=True)
        if name == "verify":
            p.add_argument("--export", type=Path, required=True)
        if name == "run":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        if name == "extract":
            p.add_argument("--output", type=Path, required=True, dest="features")
        if name == "reuse-features":
            p.add_argument("--output", type=Path, required=True)
    return parser


def _cli(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "capture":
        result = capture(args.plan, args.root, cloud_execution=args.cloud_execution)
    elif args.command == "extract":
        result = extract_stage(args.root, args.features, cloud_execution=args.cloud_execution)
    elif args.command == "seal":
        result = seal_features(args.root, args.features, cloud_execution=args.cloud_execution)
    elif args.command == "reuse-features":
        verify_capture_contract()
        plan = read_frozen_plan(args.root / "plan.json")
        result = _copy_feature_tree(args.root, args.features, args.output, plan)
    elif args.command == "labels":
        result = labels(args.plan, args.label_plan, args.root, args.features, cloud_execution=args.cloud_execution)
    elif args.command in ("train", "resume"):
        result = train_stage(args.plan, args.manifest, args.config, args.run,
            cloud_execution=args.cloud_execution, resume=args.command == "resume",
            stop_after_steps=getattr(args, "stop_after_steps", None))
    elif args.command == "interruption":
        result = inspect_interruption(args.plan, args.manifest, args.config, args.run,
                                      cloud_execution=args.cloud_execution)
    elif args.command == "export":
        result = export_stage(args.plan, args.manifest, args.config, args.run, args.output,
                              cloud_execution=args.cloud_execution)
    elif args.command == "verify":
        result = verify_export(args.plan, args.manifest, args.config, args.run, args.export, args.root,
                               cloud_execution=args.cloud_execution)
    else:
        result = _run_workflow(args.plan, args.label_plan, args.config, args.output,
                               cloud_execution=args.cloud_execution)
    print(json.dumps(result, indent=2, allow_nan=False))


def extract_stage(root: Path, features: Path, *, cloud_execution: bool) -> dict:
    cloud_guard(cloud_execution)
    verify_capture_contract()
    source = verify_sources()
    plan = read_frozen_plan(root / "plan.json")
    capture_doc = read_json(root / "capture-index.json")
    capture_rows = _index_by_id(capture_doc.get("rows"), _expected_ids(plan), "capture")
    available = [r for r in capture_rows.values() if r["status"] == "available"]
    if not available:
        reason = "no retained capture is available for generic extraction"
        _persist_extraction_failure(root, features, plan, capture_rows, reason)
        raise ValueError("no retained crop is available for generic extraction")
    from .quality_adaptation_prepare import ExtractionConfig, extract
    cfg = ExtractionConfig(max_seconds=900, max_output_bytes=512 * 1024**2,
                           max_rows=24, device="cpu")
    if features.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("frozen features must be written outside source packet")
    try:
        status = extract(root / "crops.json", features, cfg, cloud_execution=True)
    except Exception as exc:
        _persist_extraction_failure(root, features, plan, capture_rows,
            f"{type(exc).__name__}: {exc}")
        raise
    index = read_json(features / "index.json")
    indexed = _index_by_id(index.get("rows"), [r["id"] for r in available], "generic extractor")
    merged = []
    for row in plan["rows"]:
        feature = indexed.get(row["id"])
        if feature is None:
            capture_row = capture_rows[row["id"]]
            feature = {"id": row["id"], "status": "unavailable",
                       "failure_reason": capture_row.get("failure_reason", "capture_unavailable")}
        merged.append(feature)
    index["rows"] = merged
    index["expanded_capture_sha256"] = sha256(root / "capture-index.json")
    index["expanded_contract_sha256"] = sha256(CONTRACT_PATH)
    write_json(features, "index.json", index, max_output_bytes=cfg.max_output_bytes)
    if status != "completed":
        _persist_extraction_failure(root, features, plan, capture_rows,
            f"generic extraction did not complete ({status})")
        raise RuntimeError(f"generic extraction did not complete ({status}); full capture/extraction inventory persisted")
    return {"status": status, "rows": len(merged), "available": sum(r["status"] == "available" for r in merged),
            "source": source, "test_payload_reads": 0}


if __name__ == "__main__":
    _cli()
