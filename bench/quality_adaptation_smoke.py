"""Cloud-only two-step GoTrack head experiment; no model work occurs on import.

This is a systems smoke with synthetic supervision, not an accuracy experiment.
Input-only capture, frozen extraction, label creation, training, and fresh-network
verification are separate processes. Every requested row survives failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .quality_adaptation_data import (CROP_KEYS, FEATURE_KEYS, HOOKS, MAX_ARCHIVE,
    SIZE, CONVENTION, canonical_hash, crop_rgb_hash, descriptor, load_dataset,
    load_npz, read_json, require_hash, sha256, validate_crops)
from .quality_adaptation_runtime import (LOCK, atomic_stream, cloud_guard,
    load_export_into_network, load_frozen_network, load_head, runtime_versions,
    tree_bytes, verify_sources, write_json, training_state_digest)

BUDGET = 1536 * 1024**2
PLAN_SCHEMA = "gotrack_adaptation_smoke_plan_v1"
CROP_RECIPE = "estimated_seed_refiner_iteration0_legacy_native_K_corrected_integer_template_float_rgb_v1"
RASTER_CONVENTION = "legacy_opengl_native_principal_minus_half_to_integer_v1"


def peak_rss_bytes():
    import resource  # Linux-only runtime measurement, not a Windows import dependency.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def legacy_native_intrinsics(original: np.ndarray) -> np.ndarray:
    """Represent the immutable legacy-rendered RGB at OpenCV integer centers.

    The historical renderer passed cx/cy directly to OpenGL, whose raster centers
    are x+.5/y+.5. Thus its pixel[x,y] ray uses K with principal point minus .5
    in integer coordinates. This known camera correction precedes estimated
    inference and uses no object pose or supervision.
    """
    k = np.asarray(original, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError("legacy raster correction requires zero-skew OpenCV K")
    result = k.copy()
    result[0, 2] -= .5
    result[1, 2] -= .5
    return result


def sampling_map_error(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != (SIZE * SIZE, 2) or expected.shape != actual.shape:
        raise ValueError("sampling maps require all 280x280 integer centers")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("nonfinite crop sampling map")
    maximum = float(np.linalg.norm(actual.astype(np.float64) - expected, axis=1).max())
    if maximum >= 1e-3:
        raise ValueError("returned camera and retained K/C sampling map differ by >=.001 native pixels")
    return {"integer_centers": SIZE * SIZE, "max_native_pixel_difference": maximum}


@contextmanager
def observe_first_crop():
    """Observe actual returned cameras without replacing crops or network inputs."""
    from utils import data_util, misc
    from .quality_gotrack import _capture_actual_sampling_map, _capture_native_uv
    original = data_util.compute_gotrack_inputs_from_init_poses
    checks = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        if not checks:
            _, cameras, transforms = result
            native = kwargs["input_cameras"][0]
            crop = cameras[0]
            c = transforms[0].detach().cpu().numpy().astype(np.float64)
            if not np.allclose(c[:3, 3], 0., atol=1e-6, rtol=0):
                raise ValueError("perspective crop must rotate about the native camera center")
            yy, xx = np.mgrid[:SIZE, :SIZE]
            grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
            actual = _capture_actual_sampling_map(native, crop, grid)[3]
            expected = _capture_native_uv(grid, np.asarray(misc.get_intrinsic_matrix(crop)),
                                         c[:3, :3], np.asarray(misc.get_intrinsic_matrix(native)))
            checks.append(sampling_map_error(actual, expected))
        return result
    data_util.compute_gotrack_inputs_from_init_poses = observed
    try:
        yield checks
    finally:
        data_util.compute_gotrack_inputs_from_init_poses = original


def pinned_file(value: dict) -> Path:
    """Absolute cloud paths with exact bytes; no discovery or fallback."""
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError("pin requires path, sha256 and bytes")
    require_hash(value["sha256"])
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("pin requires an existing absolute nonsymlink file")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise ValueError("pin bytes must be positive")
    if path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]:
        raise ValueError(f"input pin mismatch: {path}")
    return path


def validate_plan(doc: dict, *, verify_files=True) -> dict:
    if (set(doc) != {"schema", "rows", "sealed_test", "max_seconds", "depth_tolerance_m", "native_raster_convention"}
            or doc["schema"] != PLAN_SCHEMA or doc["max_seconds"] != 1800
            or doc["native_raster_convention"] != RASTER_CONVENTION
            or type(doc["depth_tolerance_m"]) not in (int, float)
            or not np.isfinite(doc["depth_tolerance_m"]) or not 0 < doc["depth_tolerance_m"] <= .01):
        raise ValueError("smoke requires the fixed schema, 1800s budget and predeclared depth tolerance")
    expected = [("train", "17", -25), ("train", "17", 25), ("val", "27", -25)]
    if not isinstance(doc["rows"], list) or len(doc["rows"]) != 3:
        raise ValueError("exactly the three prospectively frozen frame0 rows are required")
    ids = set()
    for row, selection in zip(doc["rows"], expected):
        fields = {"id", "object_id", "split", "sequence_id", "capture_group", "frame_id",
                  "yaw0_deg", "pitch_deg", "input", "mask_index", "mask", "bank", "generation"}
        if (set(row) != fields or (row["split"], row["object_id"], row["pitch_deg"]) != selection
                or type(row["frame_id"]) is not int or row["frame_id"] != 0
                or type(row["yaw0_deg"]) is not int or row["yaw0_deg"] != 0
                or row["id"] in ids):
            raise ValueError("row differs from frozen TRAIN17 views0/1, VAL27 view0, frame0 plan")
        for key in ("id", "sequence_id", "capture_group"):
            if not isinstance(row[key], str) or not row[key] or len(row[key]) > 256:
                raise ValueError(f"invalid plan {key}")
        ids.add(row["id"])
        if verify_files:
            for name in ("input", "mask_index", "mask", "bank", "generation"):
                pinned_file(row[name])
    test = doc["sealed_test"]
    if (set(test) != {"id", "object_id", "sequence_id", "capture_group", "split", "status",
                      "query_rgb_sha256", "failure_reason", "unavailable_provenance"}
            or test["object_id"] != "7" or test["split"] != "test"
            or test["status"] != "unavailable" or test["query_rgb_sha256"] is not None
            or not test["failure_reason"] or not test["unavailable_provenance"]):
        raise ValueError("TEST7 must remain explicitly unavailable with unknown RGB identity and no payload paths")
    return doc


def plan_from_receipt(receipt: dict) -> tuple[dict, dict]:
    """Resolve the fixed three cases from parent-authenticated input metadata.

    No cloud paths are opened and no label metadata payload is parsed here.
    The parent may serialize these plans locally before authorizing execution.
    """
    if receipt.get("kind") != "adaptation-observed-input-receipt-v1" or receipt.get("status") != "complete":
        raise ValueError("expected completed observed-input receipt")
    expected = [("train", "17", "m25", -25), ("train", "17", "p25", 25), ("val", "27", "m25", -25)]
    rows = []
    for split, obj, suffix, pitch in expected:
        sequence = f"ranker-multiview-hot3d{obj}-v1-yaw0-pitch{suffix}"
        matches = [r for r in receipt["rows"] if r["sequence_id"] == sequence and r["split"] == split]
        if len(matches) != 1:
            raise ValueError("receipt lacks a unique prospectively frozen smoke case")
        source = matches[0]
        pin = lambda value: {k: value[k] for k in ("path", "sha256", "bytes")}
        masks = source["observed_masks"]
        mask_pin = lambda name: {"path": masks["path"] + "/" + name, **masks["files"][name]}
        rows.append({"id": sequence + ":frame0:iteration0", "object_id": obj, "split": split,
            "sequence_id": sequence, "capture_group": source["input"]["scene_id"],
            "frame_id": 0, "yaw0_deg": 0, "pitch_deg": pitch,
            "input": pin(source["input_pin"]), "bank": pin(source["foundpose_bank"]),
            "mask_index": mask_pin("results.json"), "mask": mask_pin("masks/0.png"),
            "generation": pin(source["generation_recipe"])})
    plan = {"schema": PLAN_SCHEMA, "rows": rows, "max_seconds": 1800, "depth_tolerance_m": .002,
        "native_raster_convention": RASTER_CONVENTION,
        "sealed_test": {"id": "sealed-hot3d7", "object_id": "7", "sequence_id": "sealed-hot3d7",
            "capture_group": "sealed-hot3d7", "split": "test", "status": "unavailable",
            "query_rgb_sha256": None, "failure_reason": "not_materialized_or_opened",
            "unavailable_provenance": "original held-out TEST7 assignment; no payload paths or pixels accessed"}}
    validate_plan(plan, verify_files=False)
    label_plan = {"schema": "gotrack_adaptation_synthetic_labels_v1",
        "recipe": pin(receipt["label_source_pins"]["recipe.json"]),
        "split_manifest": pin(receipt["label_source_pins"]["split.json"])}
    return plan, label_plan


def artifact(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def save_npz(root: Path, name: str, values: dict) -> dict:
    with atomic_stream(root, root / name, max_file_bytes=MAX_ARCHIVE, max_output_bytes=BUDGET) as f:
        np.savez(f, **values)
    return artifact(root, root / name)


class RetainedCapture:
    """Capture only the first predetermined iteration, independent of PnP outcome."""
    def __init__(self):
        self.arrays = None
        self.metadata = None
        self.failure = None

    def should_capture(self, context, kind, metadata):
        return self.failure is None and kind == "retained" and metadata.get("iteration") == 0 and self.arrays is None

    def preflight_nbytes(self, context, kind, metadata, nbytes):
        return type(nbytes) is int and 0 < nbytes <= 32 * 1024**2

    def preflight(self, context, kind, metadata, descriptors):
        return self.should_capture(context, kind, metadata)

    def _disable_capture(self, reason):
        self.failure = str(reason)

    def __call__(self, packet):
        if not self.should_capture(packet["context"], packet["kind"], packet["metadata"]):
            raise ValueError("unexpected retained capture")
        self.arrays = packet["arrays"]
        self.metadata = packet["metadata"]


def _base_row(row: dict) -> dict:
    return {key: row[key] for key in ("id", "object_id", "sequence_id", "capture_group", "split")}


def capture(plan_path: Path, output: Path, *, cloud_execution=False) -> dict:
    cloud_guard(cloud_execution)
    plan = validate_plan(read_json(plan_path))  # Frozen before decoding any RGB.
    versions = runtime_versions()
    source = verify_sources()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output, "plan.json", plan)
    rows = [dict(_base_row(row), status="unavailable", query_rgb_sha256=None,
                 failure_reason="not_processed", unavailable_provenance="frozen plan; no crop materialized")
            for row in plan["rows"]]
    write_json(output, "capture-index.json", {"rows": rows, "test_payload_reads": 0})
    started = time.monotonic()
    network = None
    from .quality_adaptation_prepare import crops_from_retained_capture
    try:
        import torch
        from .quality_gotrack import GoTrackRefiner, TexturedRenderer
        from .quality_foundpose import FoundPoseRecovery
        from .quality_contract import Frame
        from .quality_runner import read_input, read_rgb, _physical_mask_cache
        from .vision import cv2
        network = load_frozen_network("cpu")
        for ordinal, row in enumerate(plan["rows"]):
            renderer = None
            row_started = time.monotonic()
            try:
                if time.monotonic() - started >= plan["max_seconds"]:
                    raise RuntimeError("capture time budget exhausted")
                input_path = pinned_file(row["input"])
                bundle = input_path.parent
                manifest = read_input(bundle)
                if (manifest["sequence_id"] != row["sequence_id"]
                        or manifest["physical_object_id"] != f"hot3d:obj_{int(row['object_id']):06d}"
                        or manifest.get("provenance_kind") != "synthetic"):
                    raise ValueError("plan/input sequence, object or synthetic origin mismatch")
                mask_index_path = pinned_file(row["mask_index"])
                mask_index, _ = _physical_mask_cache(manifest, mask_index_path.parent)
                if mask_index.get("diagnostic_control") or mask_index.get("automatic") is False:
                    raise ValueError("smoke requires cached automatic segmentation")
                records = [r for r in mask_index["frames"] if r["frameId"] == 0]
                mask_path = pinned_file(row["mask"])
                if (len(records) != 1 or records[0].get("mask_state") != "available"
                        or (mask_index_path.parent / records[0]["path"]).resolve() != mask_path.resolve()
                        or records[0].get("sha256", records[0].get("mask_sha256", row["mask"]["sha256"])).lower() != row["mask"]["sha256"]):
                    raise ValueError("frame0 cached mask is missing/unavailable or differs from frozen mask pin")
                rgb = read_rgb(bundle, manifest, 0)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                    raise ValueError("bad input mask shape or empty mask")
                original_k = np.asarray(manifest["intrinsics"], np.float64)
                effective_k = legacy_native_intrinsics(original_k)
                from .quality_time import PHYSICAL
                frame = Frame(0, rgb, effective_k,
                              timestamp_s=0., clock_mode=PHYSICAL)
                renderer = TexturedRenderer(bundle / manifest["asset"], manifest["object_id"],
                    unlit=True, disable_multisampling=True, coordinate_mode="integer_centers_v1")
                sink = RetainedCapture()
                refiner = GoTrackRefiner(network, renderer, manifest["object_id"], "cpu",
                                        chronological_capture_callback=sink)
                refiner.set_chronological_capture_context({"smoke_row": row["id"]})
                bank = torch.load(pinned_file(row["bank"]), map_location="cpu", weights_only=False)
                backend = FoundPoseRecovery(refiner, bank, time_policy=manifest["_timing"]["time_policy"])
                seeds = backend.recover(frame, mask, top_k=5)
                if not seeds:
                    raise ValueError("FoundPose produced no estimated seed; no GT fallback")
                pnp_failure = None
                with observe_first_crop() as sampling_checks:
                    try:
                        result = refiner.refine(frame, mask, seeds[0])
                    except (ValueError, RuntimeError, cv2.error) as exc:
                        result = None
                        pnp_failure = f"{type(exc).__name__}: {exc}"[:1024]
                if sink.arrays is None:
                    raise ValueError(sink.failure or pnp_failure or "refiner emitted no iteration0 retained crop")
                if len(sampling_checks) != 1:
                    raise ValueError("actual returned crop-camera sampling map was not checked")
                if not np.array_equal(sink.arrays["native_k"], effective_k):
                    raise ValueError("actual refiner native K differs from explicit legacy raster calibration")
                crops = crops_from_retained_capture(sink.arrays, preserve_float_rgb=True)
                crop = save_npz(output, f"crop-{ordinal}.npz", crops)
                # This file contains estimated crop geometry and the actual frozen
                # output only. It is sealed before labels are computed.
                geometry = save_npz(output, f"capture-{ordinal}.npz", {
                    "crop_from_orig": np.asarray(sink.arrays["crop_from_orig"], np.float64),
                    "actual_flow": np.asarray(sink.arrays["full_flow_crop_px"], np.float32),
                    "actual_confidence": np.asarray(sink.arrays["full_confidence"], np.float32)})
                rows[ordinal] = dict(_base_row(row), status="available", query_rgb_sha256=crop_rgb_hash(crops["query_rgb"]),
                    crops=crop, capture=geometry, seed_selection="first_FoundPose_quality_sorted_hypothesis",
                    pnp_available=result is not None, pnp_failure=pnp_failure,
                    sampling_map=sampling_checks[0],
                    native_calibration={"original_K": original_k.tolist(), "effective_integer_K": effective_k.tolist(),
                        "convention": RASTER_CONVENTION, "input_manifest_unchanged": True,
                        "provenance": "pinned historical legacy TexturedRenderer OpenGL half-center raster; no pose labels"},
                    elapsed_seconds=time.monotonic() - row_started)
            except Exception as exc:
                rows[ordinal]["failure_reason"] = f"{type(exc).__name__}: {exc}"[:1024]
            finally:
                if renderer is not None:
                    renderer.close()
                write_json(output, "capture-index.json", {"rows": rows, "test_payload_reads": 0})
    finally:
        receipt = {"schema": "gotrack_adaptation_capture_receipt_v1", "recipe": CROP_RECIPE,
            "plan_sha256": canonical_hash(plan), "source": source, "versions": versions,
            "rows": rows, "test_payload_reads": 0, "annotation_inputs": [],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "solver_note": "legacy PnP uses half centers; only pre-solver iteration0 inputs are adapted; no pose claim"}
        write_json(output, "capture-receipt.json", receipt)
        write_json(output, "crops.json", {"schema": "correspondence_input_crops_v1", "conventions": CONVENTION,
            "rows": [{"id": r["id"], "crops": r["crops"]} for r in rows if r["status"] == "available"]})
    return {"available": sum(r["status"] == "available" for r in rows), "requested": len(rows)}


def seal_features(root: Path, features: Path) -> dict:
    """Check actual refiner/extractor output parity and seal before any label read."""
    capture_index = read_json(root / "capture-index.json")
    index = read_json(features / "index.json")
    extracted = {r["id"]: r for r in index["rows"]}
    checks = []
    for row in capture_index["rows"]:
        if row["status"] != "available":
            continue
        actual = load_npz(descriptor(root, row["capture"], verify=True),
                          {"crop_from_orig", "actual_flow", "actual_confidence"})
        feature = extracted.get(row["id"], {})
        if feature.get("status") != "available":
            continue
        f = load_npz(descriptor(features, feature["features"], verify=True), FEATURE_KEYS)
        mask = f["template_mask"]
        flow_error = float(np.abs(f["baseline_flow"].transpose(1, 2, 0) * mask[..., None] - actual["actual_flow"]).max())
        confidence_error = float(np.abs(f["baseline_confidence"] * mask - actual["actual_confidence"]).max())
        if flow_error > 1e-4 or confidence_error > 1e-6:
            raise ValueError("actual retained inference and exact-float extractor outputs differ")
        checks.append({"id": row["id"], "max_flow_error_px": flow_error, "max_confidence_error": confidence_error})
    receipt = {"schema": "gotrack_adaptation_prelabel_seal_v1", "capture_receipt": artifact(root, root / "capture-receipt.json"),
        "plan": artifact(root, root / "plan.json"), "crop_index": artifact(root, root / "capture-index.json"),
        "feature_index_sha256": sha256(features / "index.json"), "feature_recipe_sha256": sha256(features / "recipe.json"),
        "parity": checks, "annotation_inputs": [], "test_payload_reads": 0}
    write_json(root, "prelabel-seal.json", receipt)
    return receipt


def label_pose(vertices: np.ndarray, row: dict) -> np.ndarray:
    """Reproduce only the predeclared broader recipe pose; no label file lookup."""
    from .quality_ranker_synthetic import _broader_geometry, broader_trajectory_pose
    center, radius, _, _ = _broader_geometry(vertices)
    return broader_trajectory_pose(row["frame_id"], yaw0_deg=row["yaw0_deg"],
                                  pitch_deg=row["pitch_deg"], center_m=center, radius_m=radius)


def render_query_labels(renderer, *, object_id, crop_k, crop_from_orig, native_pose):
    """Independent synthetic query z-buffer at the retained crop camera, in metres."""
    from utils import structs, renderer_base
    from .quality_contract import checked_pose
    # C is a rigid camera-to-camera rotation in the retained API. Its translations
    # are mm, while the independently generated query pose is metres.
    c = checked_pose(crop_from_orig)
    c[:3, 3] *= .001
    query_pose = c @ checked_pose(native_pose)
    camera_to_object = np.linalg.inv(query_pose)
    camera_to_object[:3, 3] *= 1000.
    camera = structs.PinholePlaneCameraModel(width=SIZE, height=SIZE,
        f=(crop_k[0, 0], crop_k[1, 1]), c=(crop_k[0, 2], crop_k[1, 2]), T_world_from_eye=camera_to_object)
    rendered = renderer.render_object_model(object_id, camera)
    depth = np.asarray(rendered[renderer_base.RenderType.DEPTH], np.float32) * .001
    support = np.asarray(rendered[renderer_base.RenderType.MASK], bool)
    if depth.shape != (SIZE, SIZE) or support.shape != depth.shape or not np.isfinite(depth).all():
        raise ValueError("independent synthetic query renderer returned invalid depth/support")
    return query_pose.astype(np.float64), depth, support


def validate_frozen_feature_source(recipe: dict) -> dict:
    """Admit an exact reviewed old extraction without rewriting its provenance."""
    lock = read_json(LOCK)
    current = sha256(LOCK)
    original = recipe.get("source_lock_sha256")
    if original == current:
        return {"original_source_lock_sha256": original, "current_source_lock_sha256": current,
                "compatibility_used": False}
    compatible = lock.get("frozen_feature_compatibility", {}).get(original)
    allowed = sorted(["bench/quality_adaptation_smoke.py", "bench/quality_adaptation_manifest.py",
                      "bench/quality_adaptation_train.py"])
    files = lock["gotrack"]["files"]
    if (not isinstance(compatible, dict)
            or compatible.get("schema") != "reviewed_frozen_feature_compatibility_v1"
            or compatible.get("allowed_metadata_adapters") != allowed
            or recipe.get("source_files_sha256") != compatible.get("source_files_sha256")
            or recipe.get("upstream_revision") != compatible.get("upstream_revision")
            or lock["gotrack"]["revision"] != compatible.get("upstream_revision")
            or recipe.get("producer_sha256") != compatible.get("producer_sha256")
            or sha256(Path(__file__).with_name("quality_adaptation_prepare.py")) != compatible.get("producer_sha256")
            or canonical_hash({k: v for k, v in files.items() if k not in allowed}) != compatible.get("unchanged_files_sha256")):
        raise ValueError("frozen feature source lacks exact reviewed compatibility with unchanged extraction dependencies")
    return {"schema": "frozen_feature_source_compatibility_receipt_v1",
            "original_source_lock_sha256": original, "current_source_lock_sha256": current,
            "compatibility_used": True, "unchanged_files_sha256": compatible["unchanged_files_sha256"],
            "allowed_metadata_adapters": allowed, "original_recipe_unchanged": True}


def check_prelabel_seal(root: Path, features: Path) -> dict:
    seal = read_json(root / "prelabel-seal.json")
    for key in ("capture_receipt", "plan", "crop_index"):
        descriptor(root, seal[key], verify=True)
    if (sha256(features / "index.json") != seal["feature_index_sha256"]
            or sha256(features / "recipe.json") != seal["feature_recipe_sha256"]):
        raise ValueError("sealed features/index/recipe changed")
    return seal


def reuse_features(root: Path, features: Path, output: Path, *, cloud_execution=False) -> dict:
    """Copy verified feature artifacts to a fresh retry directory, without inference.

    The original failed label inventory and all sealed bytes remain untouched.
    This helper has no timer reset; the parent must keep the original deadline.
    """
    cloud_guard(cloud_execution)
    verify_sources()
    check_prelabel_seal(root, features)
    plan = validate_plan(read_json(root / "plan.json"))
    recipe = read_json(features / "recipe.json")
    compatibility = validate_frozen_feature_source(recipe)
    index = read_json(features / "index.json")
    if index.get("recipe_sha256") != canonical_hash(recipe):
        raise ValueError("original feature index does not bind original recipe")
    capture_rows = read_json(root / "capture-index.json")["rows"]
    if ({r["id"] for r in index["rows"]} != {r["id"] for r in plan["rows"]}
            or {r["id"] for r in capture_rows} != {r["id"] for r in plan["rows"]}):
        raise ValueError("sealed feature/capture inventory differs from fixed requested rows")
    files = {"index.json": features / "index.json", "recipe.json": features / "recipe.json"}
    for row in capture_rows:
        if row["status"] != "available":
            raise ValueError("cannot continue a three-crop smoke with unavailable capture rows")
        for key in ("crops", "capture"):
            descriptor(root, row[key], verify=True)
    for row in index["rows"]:
        if row["status"] != "available":
            raise ValueError("cannot continue a three-crop smoke with unavailable frozen features")
        path = descriptor(features, row["features"], verify=True)
        files[row["features"]["path"]] = path
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for name, source in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_stream(output, target, max_file_bytes=source.stat().st_size, max_output_bytes=64 * 1024**2) as f:
            f.write(source.read_bytes())
        if sha256(target) != sha256(source):
            raise ValueError("copied frozen feature bytes differ from original seal")
        copied.append(artifact(output, target))
    receipt = {"schema": "sealed_feature_reuse_receipt_v1", "original_directory": str(features.resolve()),
        "prelabel_seal_sha256": sha256(root / "prelabel-seal.json"), "files": copied,
        "feature_source_compatibility": compatibility, "inference_repeated": False,
        "original_failure_preserved": True, "test_payload_reads": 0, "deadline_reset": False}
    write_json(output, "reuse.json", receipt)
    return receipt


def validate_generation_context(context: dict, generation: dict, manifest: dict, row: dict,
                                split_file_sha256: str) -> None:
    """Interpret the pinned historical receipt schema without changing splits."""
    from .quality_ranker_synthetic import BROADER_RECIPE_KIND
    historical_split = context["group"]["split"]
    split_aliases = {"train": "train", "validation": "val"}
    if (context["recipe"]["kind"] != BROADER_RECIPE_KIND
            or context["sequence"]["yaw0_deg"] != row["yaw0_deg"]
            or context["sequence"]["pitch_deg"] != row["pitch_deg"]
            or split_aliases.get(historical_split) != row["split"]):
        raise ValueError("label recipe differs from frozen broader-view selection")
    expected = {"schema_version": 1, "kind": "quality-ranker-synthetic-generation-v1",
        "recipe_sha256": context["recipe_sha256"], "split": historical_split,
        "sequence_id": row["sequence_id"], "physical_object_id": manifest["physical_object_id"],
        "mesh_sha256": context["mesh_sha256"], "video_sha256": manifest["source_hashes"][manifest["video"]],
        "split_file_sha256": split_file_sha256}
    for key, value in expected.items():
        if generation.get(key) != value or (key == "schema_version" and type(generation.get(key)) is not int):
            raise ValueError(f"query RGB generation receipt differs from frozen supervision {key}")
    # Early schema1 generation receipts predate this optional diagnostic field.
    # The separately authenticated recipe kind remains mandatory in all cases.
    if "recipe_kind" in generation and generation["recipe_kind"] != BROADER_RECIPE_KIND:
        raise ValueError("query RGB generation receipt has incompatible recipe_kind")


def labels(plan_path: Path, label_plan_path: Path, root: Path, features: Path, *, cloud_execution=False) -> dict:
    cloud_guard(cloud_execution)
    plan = validate_plan(read_json(plan_path))
    # Authenticate all crops and frozen features BEFORE opening the separate
    # synthetic recipe/split metadata. Never open poses.jsonl or test payloads.
    seal = check_prelabel_seal(root, features)
    compatibility = validate_frozen_feature_source(read_json(features / "recipe.json"))
    if canonical_hash(read_json(root / "plan.json")) != canonical_hash(plan):
        raise ValueError("sealed input/features changed before label creation")
    capture_rows = {r["id"]: r for r in read_json(root / "capture-index.json")["rows"]}
    feature_rows = {r["id"]: r for r in read_json(features / "index.json")["rows"]}
    inventory = []
    for row in plan["rows"]:
        source = capture_rows[row["id"]]
        pending = dict(_base_row(row), status="unavailable", query_rgb_sha256=source["query_rgb_sha256"],
                       failure_reason="label_not_processed")
        if pending["query_rgb_sha256"] is None:
            pending["unavailable_provenance"] = source["unavailable_provenance"]
        inventory.append(pending)
    def persist_inventory():
        write_json(features, "labels.json", {"schema": "correspondence_label_inventory_v1",
            "label_origin": "synthetic_renderer_training_only", "rows": inventory + [plan["sealed_test"]]})
    persist_inventory()  # All requested rows survive pin failure, timeout or interruption.
    label_plan = read_json(label_plan_path)
    if (set(label_plan) != {"schema", "recipe", "split_manifest"}
            or label_plan["schema"] != "gotrack_adaptation_synthetic_labels_v1"):
        raise ValueError("label plan must bind only frozen synthetic recipe and split metadata")
    recipe_path, split_path = pinned_file(label_plan["recipe"]), pinned_file(label_plan["split_manifest"])
    from .quality_ranker_synthetic import validate_recipe
    from .quality_adaptation_prepare import project_supervision
    from .quality_adaptation_manifest import assemble
    from .quality_gotrack import TexturedRenderer
    from .quality_runner import read_input
    for ordinal, row in enumerate(plan["rows"]):
        source = capture_rows[row["id"]]
        result = dict(_base_row(row), status="unavailable", query_rgb_sha256=source["query_rgb_sha256"],
                      failure_reason=source.get("failure_reason", "feature_or_label_unavailable"))
        if result["query_rgb_sha256"] is None:
            result["unavailable_provenance"] = source["unavailable_provenance"]
        renderer = None
        try:
            if source["status"] != "available":
                continue
            feature = feature_rows.get(row["id"], {})
            if feature.get("status") != "available":
                result["failure_reason"] = feature.get("failure_reason", "not_extracted")
                continue
            crops = load_npz(descriptor(root, source["crops"], verify=True), CROP_KEYS)
            validate_crops(crops)
            geometry = load_npz(descriptor(root, source["capture"], verify=True),
                                {"crop_from_orig", "actual_flow", "actual_confidence"})
            bundle = pinned_file(row["input"]).parent
            manifest = read_input(bundle)
            context = validate_recipe(recipe_path, split_path, bundle / manifest["asset"], row["sequence_id"],
                frames=24, width=640, height=480, fps=10., mesh_units="metres", observationally_ambiguous=True)
            generation = read_json(pinned_file(row["generation"]))
            validate_generation_context(context, generation, manifest, row, sha256(split_path))
            renderer = TexturedRenderer(bundle / manifest["asset"], manifest["object_id"],
                unlit=True, disable_multisampling=True, coordinate_mode="integer_centers_v1")
            native_pose = label_pose(renderer.vertices_m, row)
            query_pose, depth, support = render_query_labels(renderer, object_id=manifest["object_id"],
                crop_k=crops["query_K"], crop_from_orig=geometry["crop_from_orig"], native_pose=native_pose)
            y = project_supervision(template_xyz_m=crops["template_xyz_m"], template_mask=crops["template_mask"],
                query_K=crops["query_K"], query_T_cam_from_object=query_pose,
                observed_depth_m=depth, observed_object_mask=support,
                depth_tolerance_m=plan["depth_tolerance_m"])
            label_descriptor = save_npz(features, f"labels-{ordinal}.npz", y)
            result = dict(_base_row(row), status="available", query_rgb_sha256=source["query_rgb_sha256"], labels=label_descriptor)
        except Exception as exc:
            result["failure_reason"] = f"{type(exc).__name__}: {exc}"[:1024]
        finally:
            if renderer is not None:
                renderer.close()
            inventory[ordinal] = result
            persist_inventory()
    receipt = {"schema": "gotrack_adaptation_labels_receipt_v1", "prelabel_seal_sha256": sha256(root / "prelabel-seal.json"),
        "label_plan_sha256": sha256(label_plan_path), "synthetic_observed_support": True,
        "test_payload_reads": 0, "labels_entered_inference": False, "quality_claim": False,
        "feature_source_compatibility": compatibility}
    write_json(features, "label-receipt.json", receipt)
    # Failure inventory is already persisted even if no usable split exists.
    assembled = assemble(features, features / "labels.json", features / "manifest.json")
    ds = load_dataset(features / "manifest.json", inspect_arrays=False)
    if len(ds.rows("train")) != 2 or len(ds.rows("val")) != 1:
        raise ValueError("two-step smoke requires all three predeclared crops; failures remain in labels.json")
    return assembled


def state_digest(state: dict) -> str:
    """Stable model tensor identity without retaining a second backbone copy."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0" + array.dtype.str.encode() + b"\0")
        h.update(json.dumps(list(array.shape)).encode() + b"\0" + array.tobytes())
    return h.hexdigest()


def inspect_interruption(run: Path, manifest: Path, *, cloud_execution=False) -> dict:
    cloud_guard(cloud_execution)
    import torch
    from .quality_adaptation_train import epoch_batches, read_config
    cfg = read_config(run / "config.json")
    ds = load_dataset(manifest, inspect_arrays=False)
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    order = epoch_batches(ds.rows("train"), 0, cfg)
    if (state["steps"] != 1 or state["epoch"] != 0 or state["batch_cursor"] != 1
            or state["completed"] or not state["optimizer"]["state"]
            or state["last_step"]["row_ids"] != [r["id"] for r in order[0]]):
        raise ValueError("step1 checkpoint did not preserve incomplete cursor/Adam state")
    receipt = {"steps": 1, "epoch": 0, "batch_cursor": 1, "completed": False,
        "optimizer_state_entries": len(state["optimizer"]["state"]),
        "optimizer_sha256": training_state_digest(state["optimizer"]),
        "head_sha256": state_digest(state["head"]), "checkpoint_sha256": sha256(run / "last.pt"),
        "last_step": state["last_step"], "next_row_ids": [r["id"] for r in order[1]]}
    write_json(run, "interruption.json", receipt)
    return receipt


def verify_export(manifest: Path, run: Path, export: Path, root: Path, *, cloud_execution=False) -> dict:
    cloud_guard(cloud_execution)
    versions = runtime_versions()
    source = verify_sources()
    from .quality_adaptation_train import read_config
    import torch
    started = time.monotonic()
    cfg = read_config(run / "config.json")
    if cfg.epochs != 1 or cfg.batch_size != 1 or cfg.max_steps != 2 or cfg.checkpoint_every_steps != 1:
        raise ValueError("smoke config must remain one epoch, batch1, two steps, checkpoint every step")
    interrupted = read_json(run / "interruption.json")
    status = read_json(run / "status.json")
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    restored = status.get("resumed_from", {})
    final_optimizer_digest = training_state_digest(state["optimizer"])
    if (not state["completed"] or state["steps"] != 2 or state["epoch"] != 1
            or restored.get("steps") != 1 or restored.get("batch_cursor") != 1
            or restored.get("optimizer_state_entries") != interrupted["optimizer_state_entries"]
            or restored.get("optimizer_sha256") != interrupted["optimizer_sha256"]
            or not state["optimizer"]["state"] or state["last_step"]["row_ids"] != interrupted["next_row_ids"]
            or final_optimizer_digest == interrupted["optimizer_sha256"]
            or any(float(slot["step"]) != 2. for slot in state["optimizer"]["state"].values())
            or state_digest(state["head"]) == interrupted["head_sha256"]):
        raise ValueError("resume repeated/skipped data, lost optimizer state or did not change head")
    standalone = load_head(initialize=False, device="cpu")
    export_state = torch.load(export, map_location="cpu", weights_only=True)
    standalone.load_state_dict(export_state["head"], strict=True)
    standalone.eval()
    network = load_frozen_network("cpu")  # Fresh actual pinned network, not a mocked container.
    frozen_before = {"backbone": state_digest(network.backbone.state_dict()),
                     "decoder": state_digest(network.decoder.state_dict())}
    base_head = state_digest(network.pose_head.state_dict())
    network = load_export_into_network(network, export, expected_sha256=sha256(export), cloud_execution=True)
    frozen_after = {"backbone": state_digest(network.backbone.state_dict()),
                    "decoder": state_digest(network.decoder.state_dict())}
    if frozen_after != frozen_before or state_digest(network.pose_head.state_dict()) == base_head:
        raise ValueError("frozen weights changed or exported head equals base checkpoint")
    ds = load_dataset(manifest)
    capture_rows = {r["id"]: r for r in read_json(root / "capture-index.json")["rows"]}
    checks = []
    with torch.inference_mode():
        for row in ds.rows("val"):
            f, _ = ds.arrays(row)
            tokens = [None] * 12
            for hook in HOOKS:
                tokens[hook] = torch.from_numpy(f[f"tokens_{hook}"][None])
            head_started = time.monotonic()
            expected = standalone(tokens, (SIZE, SIZE))
            standalone_seconds = time.monotonic() - head_started
            head_started = time.monotonic()
            actual = network.pose_head(tokens, (SIZE, SIZE))
            attached_seconds = time.monotonic() - head_started
            errors = []
            for a, b in zip(actual, expected):
                if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a, b, atol=1e-6, rtol=0):
                    raise ValueError("fresh full-network strict export load failed cached-token head parity")
                errors.append(float((a - b).abs().max()))
            crops = load_npz(descriptor(root, capture_rows[row["id"]]["crops"], verify=True), CROP_KEYS)
            images = np.stack([crops["query_rgb"], crops["template_rgb"]]).transpose(0, 3, 1, 2).copy()
            images = torch.from_numpy(images)
            mask = torch.from_numpy(crops["template_mask"][None])
            forward_started = time.monotonic()
            online = network(images[:1], images[1:], mask)
            forward_seconds = time.monotonic() - forward_started
            online_errors = []
            for a, b in zip(online, (expected[0] * mask[:, None], expected[1] * mask)):
                if not torch.isfinite(a).all() or not torch.allclose(a, b, atol=1e-4, rtol=0):
                    raise ValueError("full online RGB inference differs from cached-token export head")
                online_errors.append(float((a - b).abs().max()))
            checks.append({"id": row["id"], "max_flow_error_px": errors[0], "max_confidence_error": errors[1],
                           "standalone_head_seconds": standalone_seconds, "attached_head_seconds": attached_seconds,
                           "online_max_flow_error_px": online_errors[0], "online_max_confidence_error": online_errors[1],
                           "online_rgb_forward_seconds": forward_seconds})
    # Exercise a load-failure path against the same real network; a rejected
    # digest must not mutate even the adapted head.
    before_bad_load = state_digest(network.pose_head.state_dict())
    try:
        load_export_into_network(network, export, expected_sha256="0" * 64, cloud_execution=True)
    except ValueError as exc:
        if "hash mismatch" not in str(exc):
            raise
    else:
        raise ValueError("bad export digest unexpectedly loaded")
    if before_bad_load != state_digest(network.pose_head.state_dict()):
        raise ValueError("failed export load mutated head")
    receipt = {"schema": "gotrack_adaptation_smoke_result_v1", "source": source, "versions": versions,
        "dataset_sha256": ds.identity, "export": artifact(run, export),
        "trainable_parameters": sum(p.numel() for p in standalone.parameters()),
        "frozen_weights_before": frozen_before, "frozen_weights_after": frozen_after,
        "head_changed_from_base": True, "head_changed_after_resume": True,
        "optimizer_state_entries": len(state["optimizer"]["state"]), "steps": state["steps"],
        "final_optimizer_sha256": final_optimizer_digest, "all_Adam_steps": 2,
        "resume": restored, "parity": checks, "bad_export_digest_rejected_without_mutation": True,
        "validation": state["history"][-1]["validation"],
        "test_payload_reads": 0, "quality_claim": False, "mobile_ready": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
        "run_bytes": tree_bytes(run), "private_artifact_budget_bytes": BUDGET,
        "performance_note": "single CPU smoke measurements, not throughput or latency percentiles"}
    write_json(run, "smoke-result.json", receipt)
    return receipt


def run_smoke(plan: Path, label_plan: Path, config: Path, output: Path, *, cloud_execution=False) -> dict:
    """Fresh process per model stage under one nonresetting wall/storage budget."""
    cloud_guard(cloud_execution)
    from .quality_adaptation_runtime import ROOT
    from .quality_adaptation_train import read_config
    validate_plan(read_json(plan))
    cfg = read_config(config)
    if (cfg.epochs != 1 or cfg.batch_size != 1 or cfg.max_steps != 2 or cfg.checkpoint_every_steps != 1
            or cfg.device != "cpu" or cfg.amp or cfg.max_output_bytes > 1408 * 1024**2):
        raise ValueError("smoke config must be CPU/batch1/epoch1/step2/checkpoint1 within private run budget")
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("private adaptation output must be outside the 256 MiB source packet directory")
    output.mkdir(parents=True, exist_ok=False)
    frozen = {"plan_sha256": sha256(plan), "label_plan_sha256": sha256(label_plan), "config_sha256": sha256(config)}
    write_json(output, "invocation.json", {**frozen, "max_seconds": 1800, "test_payload_reads": 0})
    capture_root, features, run = output / "capture", output / "features", output / "run"
    manifest = features / "manifest.json"
    common = ["--cloud-execution"]
    smoke = [sys.executable, "-m", "bench.quality_adaptation_smoke"]
    trainer = [sys.executable, "-m", "bench.quality_adaptation_train"]
    commands = [
        ("capture", smoke + ["capture", "--plan", str(plan), "--root", str(capture_root)] + common, 0),
        ("extract", [sys.executable, "-m", "bench.quality_adaptation_prepare", "--manifest", str(capture_root / "crops.json"),
            "--output", str(features), "--device", "cpu", "--max-rows", "3", "--max-output-bytes", str(64 * 1024**2)] + common, 0),
        ("seal", smoke + ["seal", "--root", str(capture_root), "--features", str(features)] + common, 0),
        ("labels", smoke + ["labels", "--plan", str(plan), "--label-plan", str(label_plan),
            "--root", str(capture_root), "--features", str(features)] + common, 0),
        ("train_one_step", trainer + ["train", "--manifest", str(manifest), "--config", str(config),
            "--output", str(run), "--stop-after-steps", "1"] + common, 3),
        ("inspect_interruption", smoke + ["interruption", "--manifest", str(manifest), "--run", str(run)] + common, 0),
        ("resume", trainer + ["resume", "--manifest", str(manifest), "--config", str(config), "--output", str(run)] + common, 0),
        ("export", trainer + ["export", "--run", str(run), "--output", str(run / "head.pt")] + common, 0),
        ("verify", smoke + ["verify", "--manifest", str(manifest), "--run", str(run), "--export", str(run / "head.pt"),
            "--root", str(capture_root)] + common, 0),
    ]
    started = time.monotonic()
    stages = []
    try:
        for name, command, expected_code in commands:
            if {"plan_sha256": sha256(plan), "label_plan_sha256": sha256(label_plan), "config_sha256": sha256(config)} != frozen:
                raise ValueError("plan/label plan/config changed during smoke invocation")
            remaining = 1800 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("nonresetting 1800s smoke budget exhausted")
            if tree_bytes(output) > BUDGET:
                raise ValueError("private adaptation directory exceeds 1.5 GiB")
            before = time.monotonic()
            process = subprocess.run(command, timeout=remaining, check=False)
            stages.append({"stage": name, "exit_code": process.returncode, "expected_exit_code": expected_code,
                           "elapsed_seconds": time.monotonic() - before})
            write_json(output, "stages.json", {"stages": stages, "elapsed_seconds": time.monotonic() - started})
            if process.returncode != expected_code:
                raise RuntimeError(f"smoke stage {name} exited {process.returncode}; expected {expected_code}; preserve all outputs")
        result = read_json(run / "smoke-result.json")
        result.update(total_elapsed_seconds=time.monotonic() - started, private_artifact_bytes=tree_bytes(output),
                      stages=stages, frozen_invocation=frozen)
        if result["private_artifact_bytes"] > BUDGET:
            raise ValueError("private adaptation directory exceeds 1.5 GiB")
        write_json(output, "result.json", result)
        return result
    finally:
        write_json(output, "stages.json", {"stages": stages, "elapsed_seconds": time.monotonic() - started,
            "private_artifact_bytes": tree_bytes(output), "max_seconds": 1800,
            "private_artifact_budget_bytes": BUDGET, "test_payload_reads": 0})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "seal", "labels", "interruption", "verify", "run", "reuse-features"):
        p = sub.add_parser(name)
        p.add_argument("--cloud-execution", action="store_true")
        if name in ("capture", "labels", "run"):
            p.add_argument("--plan", type=Path, required=True)
        if name in ("capture", "seal", "labels", "verify", "reuse-features"):
            p.add_argument("--root", type=Path, required=True)
        if name in ("seal", "labels", "reuse-features"):
            p.add_argument("--features", type=Path, required=True)
        if name in ("labels", "run"):
            p.add_argument("--label-plan", type=Path, required=True)
        if name in ("interruption", "verify"):
            p.add_argument("--manifest", type=Path, required=True)
            p.add_argument("--run", type=Path, required=True)
        if name == "verify":
            p.add_argument("--export", type=Path, required=True)
        if name == "run":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        if name == "reuse-features":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cloud_guard(args.cloud_execution)
    if args.command == "reuse-features":
        result = reuse_features(args.root, args.features, args.output, cloud_execution=True)
    elif args.command == "run":
        result = run_smoke(args.plan, args.label_plan, args.config, args.output, cloud_execution=True)
    elif args.command == "capture":
        result = capture(args.plan, args.root, cloud_execution=True)
    elif args.command == "seal":
        result = seal_features(args.root, args.features)
    elif args.command == "labels":
        result = labels(args.plan, args.label_plan, args.root, args.features, cloud_execution=True)
    elif args.command == "interruption":
        result = inspect_interruption(args.run, args.manifest, cloud_execution=True)
    else:
        result = verify_export(args.manifest, args.run, args.export, args.root, cloud_execution=True)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
