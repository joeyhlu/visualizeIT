"""Cloud-only frozen-feature extraction and numerical supervision utilities.

The extractor accepts input crops only. Reference poses and annotation depth are
accepted only by the separate pure supervision function, never by the network.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .quality_adaptation_data import (CONVENTION, CROP_KEYS, HOOKS, MAX_ARCHIVE, SIZE,
                                      canonical_hash, descriptor, load_npz, crop_rgb_hash, read_json,
                                      sha256, validate_crops, validate_features, validate_labels)
from .quality_adaptation_runtime import (LOCK, atomic_stream, cloud_guard, load_frozen_network,
                                         runtime_versions, verify_sources, write_json)


def crops_from_retained_capture(arrays: dict, *, preserve_float_rgb=False) -> dict:
    """Convert the existing GoTrack ``retained`` callback arrays to input crops.

    The caller must obtain these from the estimated-seed inference branch, never
    a reference-pose replay. No GT poses are accepted. Template depth and the
    estimated template camera transform are millimetres in that existing API;
    this conversion explicitly emits object-frame metres.
    """
    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    def rgb(value):
        value = numpy(value)
        if value.shape != (1, 3, SIZE, SIZE) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("captured RGB must be [1,3,280,280] finite float in [0,1]")
        value = value[0].transpose(1, 2, 0)
        if preserve_float_rgb:
            if value.dtype != np.float32:
                raise ValueError("exact captured RGB must be float32")
            return value.copy()
        return np.floor(value * 255 + .5).astype(np.uint8)
    depth = numpy(arrays["rendered_depth_mm"]).squeeze()
    mask = numpy(arrays["rendered_mask"]).squeeze() > 0
    k = numpy(arrays["crop_k"]).astype(np.float64)
    pose = numpy(arrays["current_crop_camera_from_object_mm"]).astype(np.float64)
    if depth.shape != (SIZE, SIZE) or mask.shape != depth.shape or k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("capture geometry shapes differ from frozen crop interface")
    if (not np.isfinite(depth).all() or not np.isfinite(pose).all() or not np.isfinite(k).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-6):
        raise ValueError("capture geometry must be finite with a proper rigid estimated pose")
    mask &= depth > 0
    yy, xx = np.mgrid[:SIZE, :SIZE]
    rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(k).T
    xyz_camera_mm = rays * depth[..., None]
    xyz_object_m = ((xyz_camera_mm - pose[:3, 3]) @ pose[:3, :3]) / 1000.
    xyz_object_m[~mask] = 0
    result = {"query_rgb": rgb(arrays["query_rgb_crop"]), "template_rgb": rgb(arrays["template_rgb"]),
              "template_mask": mask, "template_xyz_m": xyz_object_m.astype(np.float32), "query_K": k}
    validate_crops(result)
    return result


def project_supervision(*, template_xyz_m: np.ndarray, template_mask: np.ndarray,
                        query_K: np.ndarray, query_T_cam_from_object: np.ndarray,
                        observed_depth_m: np.ndarray, observed_object_mask: np.ndarray,
                        depth_tolerance_m: float) -> dict:
    """Project independently annotated/synthetic object geometry into query pixels.

    Visibility requires an independently supplied observed object mask AND
    agreement with observed/rendered metric depth. A projected silhouette alone
    cannot establish observed foreground. Nearest depth sampling uses floor(x+.5)
    at integer pixel centers; unknown depth is excluded from all losses.
    """
    from .quality_adaptation_data import validate_geometry, _array
    validate_geometry({"template_xyz_m": template_xyz_m, "template_mask": template_mask, "query_K": query_K})
    _array(query_T_cam_from_object, (4, 4), "float64", "query_T_cam_from_object")
    t = query_T_cam_from_object
    if (not np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), rtol=0, atol=1e-7)
            or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-7):
        raise ValueError("label pose must be a proper rigid object-to-query-camera transform in metres")
    _array(observed_depth_m, (SIZE, SIZE), "float32", "observed_depth_m")
    _array(observed_object_mask, (SIZE, SIZE), "bool", "observed_object_mask")
    if np.any(observed_depth_m < 0):
        raise ValueError("depth is positive metres; zero denotes unknown")
    if isinstance(depth_tolerance_m, bool) or not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
        raise ValueError("depth_tolerance_m must be finite and positive")
    xyz = template_xyz_m.astype(np.float64) @ t[:3, :3].T + t[:3, 3]
    h = xyz @ query_K.T
    front = template_mask & (xyz[..., 2] > 0)
    uv = np.zeros((SIZE, SIZE, 2), dtype=np.float64)
    uv[front] = h[front, :2] / h[front, 2:3]
    inside = front & (uv[..., 0] >= 0) & (uv[..., 0] <= SIZE - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= SIZE - 1)
    known = template_mask & ~inside
    visible = np.zeros_like(template_mask)
    ids = np.where(inside)
    pix = np.floor(uv[inside] + .5).astype(np.int64)
    depth = observed_depth_m[pix[:, 1], pix[:, 0]]
    observed = depth > 0
    known[ids] = observed
    visible[ids] = (observed & observed_object_mask[pix[:, 1], pix[:, 0]] &
                    (np.abs(depth - xyz[inside, 2]) <= depth_tolerance_m))
    yy, xx = np.mgrid[:SIZE, :SIZE]
    flow = np.zeros((2, SIZE, SIZE), dtype=np.float32)
    delta = uv - np.stack([xx, yy], axis=-1)
    flow[:, front] = delta[front].T.astype(np.float32)
    result = {"flow": flow, "visible": visible, "known": known}
    validate_labels(result, template_mask)
    return result


def extract_one(network, crops: dict, *, device: str, torch_module=None) -> dict:
    """Use the same backbone/decoder/head calls and [0,1] RGB as frozen inference."""
    validate_crops(crops)
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    images = np.stack([crops["query_rgb"], crops["template_rgb"]]).transpose(0, 3, 1, 2).copy()
    with torch.inference_mode():
        rgb = torch.from_numpy(images).to(device=device, dtype=torch.float32)
        if images.dtype == np.uint8:
            rgb = rgb / 255.
        features = network.backbone(rgb)["feature_maps"]
        query, template = features.chunk(2, dim=0)
        _, reference = network.decoder(features_query=query, features_reference=template, crop_size=(SIZE, SIZE))
        flow, confidence = network.pose_head(reference, (SIZE, SIZE))
        values = {f"tokens_{hook}": reference[hook][0].detach().float().cpu().numpy().copy() for hook in HOOKS}
        values.update({"baseline_flow": flow[0].detach().float().cpu().numpy().copy(),
                       "baseline_confidence": confidence[0].detach().float().cpu().numpy().copy()})
    values.update({k: crops[k].copy() for k in ("template_mask", "template_xyz_m", "query_K")})
    validate_features(values)
    return values


@dataclass(frozen=True)
class ExtractionConfig:
    max_seconds: int = 900
    max_output_bytes: int = 512 * 1024**2
    max_rows: int = 64
    device: str = "cpu"

    def validate(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for key in ("max_seconds", "max_output_bytes", "max_rows"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_rows > 10000 or self.max_output_bytes > 8 * 1024**3 or self.max_seconds > 86400:
            raise ValueError("extraction exceeds hard row/storage/time ceiling")


def extract(manifest: Path, output: Path, cfg: ExtractionConfig, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    cfg.validate()
    runtime_versions()
    source = verify_sources()
    doc = read_json(manifest)
    if (set(doc) != {"schema", "conventions", "rows"} or doc["schema"] != "correspondence_input_crops_v1"
            or doc["conventions"] != CONVENTION or not isinstance(doc["rows"], list)
            or not 1 <= len(doc["rows"]) <= cfg.max_rows):
        raise ValueError("invalid input-only crop manifest")
    seen = set()
    for row in doc["rows"]:
        if set(row) != {"id", "crops"} or not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("crop rows require unique id and crops descriptor only; no labels/poses")
        seen.add(row["id"])
        path = descriptor(manifest.parent, row["crops"], verify=True)
        validate_crops(load_npz(path, CROP_KEYS))
    output.mkdir(parents=True, exist_ok=False)
    recipe = {"schema": "frozen_gotrack_decoder_recipe_v1", **source,
              "crop_manifest_sha256": canonical_hash(doc), "hooks": list(HOOKS),
              "annotation_inputs": [], "normalization": "upstream_DinoFeatureExtractor_RGB_0_1",
              "rgb_encoding": "uint8_div255_or_exact_float32_0_1_v1",
              "producer_sha256": sha256(Path(__file__)), "device": cfg.device}
    write_json(output, "recipe.json", recipe, max_output_bytes=cfg.max_output_bytes)
    rows = []
    start = time.monotonic()
    network = None
    status = "failed"
    try:
        network = load_frozen_network(cfg.device)
        for index, row in enumerate(doc["rows"]):
            if time.monotonic() - start >= cfg.max_seconds:
                status = "incomplete_time_limit"
                break
            try:
                path = descriptor(manifest.parent, row["crops"], verify=True)
                crops = load_npz(path, CROP_KEYS)
                validate_crops(crops)
                query_hash = crop_rgb_hash(crops["query_rgb"])
                values = extract_one(network, crops, device=cfg.device)
                name = f"features-{index:06d}.npz"
                with atomic_stream(output, output / name, max_file_bytes=MAX_ARCHIVE,
                                   max_output_bytes=cfg.max_output_bytes) as f:
                    np.savez(f, **values)
                rows.append({"id": row["id"], "status": "available", "crops": row["crops"],
                             "query_rgb_sha256": query_hash,
                             "features": {"path": name, "sha256": sha256(output / name), "bytes": (output / name).stat().st_size}})
            except (ValueError, RuntimeError, OSError) as e:
                rows.append({"id": row["id"], "status": "unavailable", "failure_reason": str(e)[:1024]})
                raise
            finally:
                write_json(output, "index.json", {"recipe_sha256": canonical_hash(recipe), "rows": rows},
                           max_output_bytes=cfg.max_output_bytes)
        else:
            status = "completed"
    finally:
        processed = {r["id"] for r in rows}
        rows += [{"id": r["id"], "status": "unavailable", "failure_reason": "not_processed"}
                 for r in doc["rows"] if r["id"] not in processed]
        write_json(output, "index.json", {"recipe_sha256": canonical_hash(recipe), "status": status,
                   "elapsed_seconds": time.monotonic() - start, "rows": rows}, max_output_bytes=cfg.max_output_bytes)
        if network is not None:
            del network
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-output-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    cfg = ExtractionConfig(args.max_seconds, args.max_output_bytes, args.max_rows, args.device)
    result = extract(args.manifest, args.output, cfg, cloud_execution=args.cloud_execution)
    print(result)
    return 0 if result == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
