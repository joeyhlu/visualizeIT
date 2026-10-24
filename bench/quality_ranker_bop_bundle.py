"""Convert acquired BOP RGB/PLY inputs to a timed inference bundle on cloud.

Only scored RGB rows, calibration and the mesh are consumed. Onboarding and
evaluator poses are never read. Original timestamps require a separate cited
sidecar: a BOP frame number alone is not a capture clock.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .quality_ranker_data import _file_sha256, _json_load, _write_json, read_split_manifest
from .quality_ranker_synthetic import _cloud_guard
from .quality_time import PHYSICAL, parse_timing_manifest

MAX_FRAMES = 300
MAX_SOURCE_BYTES = 256 * 1024 * 1024


def checked_source(root, reference):
    if not isinstance(reference, dict) or reference.get("available") is not True:
        raise ValueError("Acquire each source file before converting the BOP bundle")
    name, sha = reference.get("path"), reference.get("sha256")
    if (not isinstance(name, str) or not name or Path(name).is_absolute() or
            not isinstance(sha, str) or len(sha) != 64 or
            any(c not in "0123456789abcdef" for c in sha.lower()) or set(sha) == {"0"}):
        raise ValueError("BOP sources require relative paths and non-placeholder SHA-256")
    root = Path(root).resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("BOP source escapes its root or exceeds the source byte limit")
    if _file_sha256(path) != sha.lower():
        raise ValueError("BOP source SHA-256 mismatch")
    return path


def plan_bundle(inference, reference_id, group, timestamps):
    """Pure metadata validation; no evaluator or heavy renderer imports."""
    if inference.get("schemaVersion") != 1 or inference.get("dataset") != "BOP YCB-Video":
        raise ValueError("Expected the existing prototype_dataset BOP YCB-Video inference manifest")
    matches = [row for row in inference.get("objects", []) if row.get("referenceId") == reference_id]
    if len(matches) != 1:
        raise ValueError("reference-id must identify exactly one inference object")
    obj = matches[0]
    number, scene = obj.get("objectId"), str(obj.get("sceneId"))
    if type(number) is not int or number <= 0:
        raise ValueError("BOP objectId must be a positive integer")
    physical = f"bop-ycbv:{number:06d}"
    sequence = f"bop-ycbv:scene-{scene}:object-{number:06d}"
    if (group.get("physical_object_id") != physical or group.get("sequence_id") != sequence or
            group.get("scene_id") != f"bop-ycbv:scene:{scene}" or
            group.get("background_id") != f"bop-ycbv:scene:{scene}" or
            group.get("background_kind") != "observed" or group.get("provenance_kind") != "real"):
        raise ValueError("BOP object, sequence and background must match the frozen split group")
    model = obj.get("model", {})
    if not group.get("mesh_sha256") or str(model.get("sha256")).lower() != group["mesh_sha256"]:
        raise ValueError("Freeze the acquired BOP mesh SHA-256 before converting")
    frames = obj.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"BOP conversion requires 1..{MAX_FRAMES} scored frames")
    ids = [row.get("frameId") for row in frames]
    if (any(type(v) is not int or v < 0 for v in ids) or
            any(b <= a for a, b in zip(ids, ids[1:]))):
        raise ValueError("BOP scored source IDs must be unique and increasing")
    if group.get("frames") != [{"frame_id": i, "source_frame_id": source} for i, source in enumerate(ids)]:
        raise ValueError("BOP frame identities must match the frozen split exactly")
    source = timestamps.get("timestamp_source")
    fps = timestamps.get("nominal_fps")
    rows = timestamps.get("frames")
    if (not isinstance(source, str) or not source.strip() or isinstance(fps, bool) or
            not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 240):
        raise ValueError("Timestamp sidecar must cite its capture source and positive nominal_fps")
    if not isinstance(rows, list) or [row.get("source_frame_id") for row in rows] != ids:
        raise ValueError("Timestamp sidecar must match every scored BOP source frame in order")
    times = [row.get("timestamp_s") for row in rows]
    if (any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Actual capture timestamps must be finite and strictly increasing")
    intrinsics = []
    for row in frames:
        k = np.asarray(row.get("calibration", {}).get("cam_K"), dtype=np.float64)
        if k.shape != (9,) or not np.isfinite(k).all():
            raise ValueError("BOP frame needs nine finite calibrated intrinsics")
        k = k.reshape(3, 3)
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] or k[1, 0]:
            raise ValueError("BOP calibration must be a pinhole matrix without skew")
        intrinsics.append(k)
    if any(not np.array_equal(k, intrinsics[0]) for k in intrinsics[1:]):
        raise ValueError("Split variable-intrinsics captures into constant-calibration bundles")
    return {"object": obj, "frames": frames, "source_ids": ids, "timestamps": times,
            "intrinsics": intrinsics[0].tolist(), "nominal_fps": float(fps), "timestamp_source": source}


def bundle_manifest(plan, group, *, resolution, video_sha256, asset_sha256):
    width, height = resolution
    alias = "bop_" + group["mesh_sha256"][:16]
    manifest = {
        "schema_version": 2, "object": alias, "object_id": alias,
        **{k: group[k] for k in ("physical_object_id", "sequence_id", "scene_id", "background_id",
                                 "background_kind", "provenance_kind", "mesh_sha256")},
        "units": "metres", "video": "recording.avi", "asset": "object.glb",
        "source_hashes": {"recording.avi": video_sha256, "object.glb": asset_sha256},
        "intrinsics": plan["intrinsics"], "native_resolution": [width, height],
        "setup_frame_id": None, "frame_ids": list(range(len(plan["frames"]))),
        "original_source_frame_ids": {str(i): value for i, value in enumerate(plan["source_ids"])},
        "clock": {"mode": PHYSICAL, "units": "seconds", "source_units": "seconds",
                  "timestamp_source": plan["timestamp_source"], "nominal_fps": plan["nominal_fps"]},
        # The decoder seeks the new lossless video's ordinals. Diagnostics use
        # original_source_frame_ids to retain the independently labeled BOP IDs.
        "timeline": [{"frame_id": i, "source_frame_id": i, "source_timestamp": t,
                      "timestamp_s": t, "role": "scored"} for i, t in enumerate(plan["timestamps"])],
        "conversion": {"source_mesh_units": "millimetres", "mesh_scale": .001,
                       "video_codec": "FFV1", "recentered": False},
    }
    parse_timing_manifest(json.dumps(manifest, allow_nan=False).encode())
    return manifest


def verify_lossless_video(cv2, video, source_paths):
    """Require every decoded frame to equal its source, including exact count."""
    reader = cv2.VideoCapture(str(video))
    try:
        if not reader.isOpened():
            raise RuntimeError("Converted BOP video cannot be reopened")
        for path in source_paths:
            valid, decoded = reader.read()
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if not valid or original is None or not np.array_equal(decoded, original):
                raise RuntimeError("BOP FFV1 round-trip differs from source RGB or lost a frame")
        if reader.read()[0]:
            raise RuntimeError("BOP output contains extra frames")
    finally:
        reader.release()


def convert(inference_path, reference_id, group, timestamps_path, data_root, output, *, cloud_execution):
    _cloud_guard(cloud_execution)
    inference, _ = _json_load(inference_path)
    timestamps, _ = _json_load(timestamps_path)
    plan = plan_bundle(inference, reference_id, group, timestamps)
    mesh_path = checked_source(data_root, plan["object"]["model"])
    rgb_paths = [checked_source(data_root, row["rgb"]) for row in plan["frames"]]
    if sum(p.stat().st_size for p in [mesh_path, *rgb_paths]) > MAX_SOURCE_BYTES:
        raise ValueError("Selected BOP sources exceed the bounded conversion byte limit")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("BOP conversion requires a fresh output directory")
    import trimesh
    from .vision import cv2
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("BOP model must contain a finite triangle mesh")
    mesh.apply_scale(.001)
    mesh.export(output / "object.glb")
    writer = None
    resolution = None
    video = output / "recording.avi"
    try:
        for path in rgb_paths:
            rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if rgb is None or rgb.ndim != 3 or max(rgb.shape[:2]) > 1920:
                raise ValueError("BOP RGB frame cannot be decoded or exceeds 1920px")
            size = (rgb.shape[1], rgb.shape[0])
            if writer is None:
                resolution = size
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), plan["nominal_fps"], size)
                if not writer.isOpened():
                    raise RuntimeError("Cloud OpenCV needs FFV1 lossless video encoding support")
            elif size != resolution:
                raise ValueError("BOP frame resolutions differ within a bundle")
            writer.write(rgb)
            if video.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("BOP output exceeds the bounded conversion byte limit")
    finally:
        if writer is not None:
            writer.release()
    if video.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("BOP output exceeds the bounded conversion byte limit")
    verify_lossless_video(cv2, video, rgb_paths)
    manifest = bundle_manifest(plan, group, resolution=resolution,
                               video_sha256=_file_sha256(video), asset_sha256=_file_sha256(output / "object.glb"))
    manifest["conversion"].update({"inference_sha256": _file_sha256(inference_path),
                                   "timestamps_sha256": _file_sha256(timestamps_path)})
    _write_json(output / "input.json", manifest)
    from .quality_runner import read_input
    read_input(output)
    return {"bundle": str(output), "frames": len(rgb_paths), "evaluation_labels_consumed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inference", "timestamps", "data-root", "split-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    split = read_split_manifest(args.split_manifest)
    groups = [row for row in split["groups"] if row["sequence_id"] == args.sequence_id]
    if len(groups) != 1:
        raise ValueError("sequence-id must resolve to one frozen split group")
    print(json.dumps(convert(args.inference, args.reference_id, groups[0], args.timestamps,
                             args.data_root, args.output, cloud_execution=args.cloud_execution)))


if __name__ == "__main__":
    main()
