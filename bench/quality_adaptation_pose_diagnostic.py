"""Cached synthetic VAL crop-pose projection diagnostic; never a tracking claim.

Prediction accepts input geometry, flow and confidence only. The CLI authenticates
cached features and exports, writes/seals EVERY branch before opening any labels,
then scores projected template XYZ against synthetic visible correspondences.
This lacks observed query masks, production pose seeds and native crop transforms;
it is not production replay, independent pose accuracy or a deployment gate.

CLI (approved Linux cloud only; existing cached weights, no downloads/training)::

    VISUALIZEIT_CLOUD=1 python -m bench.quality_adaptation_pose_diagnostic \
        --manifest cache/manifest.json --plan pose-plan.json \
        --plan-sha256 <sha256-of-exact-plan-file> --output-root fresh-pose-run \
        --cloud-execution

The strict plan JSON has exactly these fields (replace placeholders with SHA256):
{"schema":"cached_crop_pose_plan_v1", "dataset_sha256":"<evaluation-identity>",
 "feature_recipe_sha256":"<manifest-provenance-recipe>",
 "source_lock_sha256":"<current-runtime-lock>",
 "exports":[{"name":"adapted", "path":"run/head.pt", "bytes":123,
             "sha256":"<export-file>", "dataset_sha256":"<training-identity>",
             "config_sha256":"<training-config>"}]}
Export paths are relative to the plan file. Dataset identity is the existing
canonical manifest hash, not the hash of serialized manifest bytes. The saved
baseline arrays are always a separate first branch. CPU execution is fixed.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import platform
from pathlib import Path

import numpy as np

from . import quality_adaptation_data as data
from . import quality_adaptation_runtime as runtime
from .quality_pnp_mug_replay import replay_solver_inputs
from .vision import cv2


PREDICTION_SCHEMA = "cached_crop_pose_prediction_v1"
PLAN_SCHEMA = "cached_crop_pose_plan_v1"
GEOMETRY_KEYS = {"template_mask", "template_xyz_m", "query_K"}
SOURCE_FILES = ("quality_adaptation_pose_diagnostic.py", "quality_adaptation_data.py",
                "quality_adaptation_runtime.py", "quality_pnp_mug_replay.py", "vision.py")
SCOPE = "synthetic_cached_VAL_crop_projection_diagnostic_only"


def _geometry(features):
    if (not isinstance(features, dict) or not GEOMETRY_KEYS <= set(features)
            or set(features) - data.FEATURE_KEYS):
        raise ValueError("prediction geometry must contain input-only feature keys")
    data.validate_geometry(features)


def _flow_confidence(flow, confidence):
    for array, shape, name in ((flow, (2, data.SIZE, data.SIZE), "flow"),
                                (confidence, (data.SIZE, data.SIZE), "confidence")):
        if (not isinstance(array, np.ndarray) or array.dtype != np.float32
                or array.shape != shape or not np.isfinite(array).all()):
            raise ValueError(f"{name} requires finite float32{shape}")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be a probability in [0,1]")


def _targets(flow):
    yy, xx = np.mgrid[:data.SIZE, :data.SIZE]
    return np.stack((xx, yy), axis=-1) + np.moveaxis(flow, 0, -1)


def predict_pose(features, flow, confidence, seed=521):
    """Input-only frozen 3000-iteration RANSAC/LM with no extrinsic guess.

    Native float32 confidence > .3 and integer-grid+flow correspondences are
    restricted to template support and in-crop query targets. XYZ metres are
    converted to millimetres for the existing solver. At least 24 points are
    required; deterministic row-major linspace caps solver input at 10,000 points.
    """
    _geometry(features)
    _flow_confidence(flow, confidence)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be uint32")
    target = _targets(flow)
    keep = (features["template_mask"] & (confidence > .3)
            & np.isfinite(target).all(axis=-1)
            & (target >= 0).all(axis=-1) & (target <= data.SIZE - 1).all(axis=-1))
    indices = np.flatnonzero(keep)
    retained = len(indices)
    if retained > 10000:
        indices = indices[np.linspace(0, retained - 1, 10000, dtype=int)]
    result = dict(schema=PREDICTION_SCHEMA, scope=SCOPE, seed=seed,
                  retained_point_count=retained, sampled_point_count=len(indices),
                  sampled_template_flat_indices=indices.tolist(),
                  use_extrinsic_guess=False, solver_success=False,
                  crop_camera_from_object_mm=None, fit_state="insufficient_correspondences")
    if retained >= 24:
        obj = features["template_xyz_m"].reshape(-1, 3)[indices].astype(np.float64) * 1000.
        solved = replay_solver_inputs(obj, target.reshape(-1, 2)[indices].astype(np.float64),
                                      features["query_K"], np.zeros((3, 1)),
                                      np.zeros((3, 1)), seed, False)
        result.update(solved)
    data.canonical_hash(result)  # Reject nonfinite solver output before persistence.
    return result


def _pose(prediction):
    if (not isinstance(prediction, dict) or prediction.get("schema") != PREDICTION_SCHEMA
            or prediction.get("scope") != SCOPE
            or type(prediction.get("solver_success")) is not bool
            or not isinstance(prediction.get("fit_state"), str)
            or prediction.get("use_extrinsic_guess") is not False):
        raise ValueError("malformed pose prediction")
    data.canonical_hash(prediction)
    required = {"crop_camera_from_object_mm", "retained_point_count", "sampled_point_count",
                "sampled_template_flat_indices", "seed"}
    if not required <= set(prediction):
        raise ValueError("incomplete pose prediction")
    retained, sampled = prediction["retained_point_count"], prediction["sampled_point_count"]
    ids = prediction["sampled_template_flat_indices"]
    if (type(retained) is not int or not 0 <= retained <= data.SIZE**2
            or type(sampled) is not int or sampled != min(retained, 10000)
            or type(prediction["seed"]) is not int or not 0 <= prediction["seed"] <= 2**32 - 1
            or not isinstance(ids, list) or len(ids) != sampled
            or any(type(i) is not int or not 0 <= i < data.SIZE**2 for i in ids)
            or ids != sorted(set(ids))):
        raise ValueError("invalid prediction point accounting")
    value = prediction.get("crop_camera_from_object_mm")
    if value is None:
        if prediction["fit_state"] == "refined":
            raise ValueError("refined prediction is missing its pose")
        return None
    pose = np.asarray(value, dtype=np.float64)
    if (not prediction["solver_success"] or prediction["fit_state"] != "refined"
            or pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.array_equal(pose[3], [0., 0., 0., 1.])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6)):
        raise ValueError("invalid rigid crop-camera pose")
    return pose


def score_pose(features, prediction, labels):
    """Post-seal scoring only; no reference-pose/rotation accuracy is available.

    Error summaries are conditional on finite positive-depth projections. Every
    visible supported point remains in the <=2px success denominator, including
    points missing a projection or behind the camera (z <= .01 millimetres).
    """
    _geometry(features)
    data.validate_labels(labels, features["template_mask"])
    pose = _pose(prediction)
    visible = labels["visible"] & features["template_mask"]
    count = int(visible.sum())
    projected_count = behind = 0
    error = np.empty(0, dtype=np.float64)
    if pose is not None:
        obj = features["template_xyz_m"][visible].astype(np.float64) * 1000.
        camera = obj @ pose[:3, :3].T + pose[:3, 3]
        positive = np.isfinite(camera).all(axis=1) & (camera[:, 2] > .01)
        behind = int((np.isfinite(camera).all(axis=1) & (camera[:, 2] <= .01)).sum())
        homogeneous = camera @ features["query_K"].T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            projected = homogeneous[:, :2] / homogeneous[:, 2:]
            residual = np.linalg.norm(projected - _targets(labels["flow"])[visible], axis=1)
        valid = positive & np.isfinite(projected).all(axis=1) & np.isfinite(residual)
        error = residual[valid]
        projected_count = int(valid.sum())
    stats = dict(count=projected_count, mean=None, rmse=None, median=None, p95=None, maximum=None)
    if error.size:
        # hypot-based RMS avoids overflow for finite but very large diagnostics.
        stats.update(mean=float(np.mean(error)), median=float(np.median(error)),
                     rmse=float(np.linalg.norm(error / np.sqrt(error.size))),
                     p95=float(np.percentile(error, 95)), maximum=float(error.max()))
    result = dict(scope=SCOPE, visible_supported_point_count=count,
                  projected_point_count=projected_count, behind_camera_point_count=behind,
                  missing_projection_point_count=count - projected_count,
                  projection_error_px=stats,
                  within_2px_fraction_all_visible=float((error <= 2.).sum() / count),
                  conditional_error_scope="finite_positive_depth_projections_only",
                  rotation_accuracy_available=False, quality_claim=False)
    data.canonical_hash(result)
    return result


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"requires regular nonsymlink file: {path}")
    return path


def _features(ds, row):
    path = _regular(data.descriptor(ds.root, row["features"], verify=True))
    features = data.load_npz(path, data.FEATURE_KEYS)
    data.validate_features(features)
    if data.sha256(path) != row["features"]["sha256"]:
        raise ValueError("feature archive changed during loading")
    return features


def _write(root, name, value):
    # The fresh run is append-only. A failed invocation cannot be resumed over it.
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (root / name).open("xb") as stream:
        stream.write(payload)
    return data.sha256(root / name)


def _plan(path, expected_hash, ds):
    data.require_hash(expected_hash, "plan SHA256")
    path = _regular(path)
    if data.sha256(path) != expected_hash:
        raise ValueError("plan SHA256 mismatch")
    plan = data.read_json(path)
    data.canonical_hash(plan)
    if (not isinstance(plan, dict) or set(plan) != {"schema", "dataset_sha256",
            "feature_recipe_sha256", "source_lock_sha256", "exports"}
            or plan["schema"] != PLAN_SCHEMA
            or plan["dataset_sha256"] != ds.identity
            or plan["feature_recipe_sha256"] != ds.document["provenance"]["feature_recipe_sha256"]
            or plan["source_lock_sha256"] != data.sha256(runtime.LOCK)
            or ds.document["provenance"]["source_lock_sha256"] != plan["source_lock_sha256"]):
        raise ValueError("plan schema/dataset/feature/source identity mismatch")
    if not isinstance(plan["exports"], list) or not 1 <= len(plan["exports"]) <= 8:
        raise ValueError("plan requires 1..8 sequential exports")
    names, hashes = {"baseline"}, set()
    for item in plan["exports"]:
        if not isinstance(item, dict) or set(item) != {"name", "path", "sha256", "bytes",
                                                      "dataset_sha256", "config_sha256"}:
            raise ValueError("invalid export descriptor")
        name = item["name"]
        if (not isinstance(name, str) or not 1 <= len(name) <= 64
                or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
                or name in names or item["sha256"] in hashes):
            raise ValueError("invalid or duplicate export name/hash")
        names.add(name)
        hashes.add(item["sha256"])
        for key in ("sha256", "dataset_sha256", "config_sha256"):
            data.require_hash(item[key], key)
        if type(item["bytes"]) is not int or not 0 < item["bytes"] <= 384 * 1024**2:
            raise ValueError("export exceeds byte limit")
        data.local_path(path.parent, item["path"])
    return path, plan


def _export(plan_path, item, source_hash, torch):
    path = _regular(data.local_path(plan_path.parent, item["path"]))
    if path.stat().st_size != item["bytes"] or data.sha256(path) != item["sha256"]:
        raise ValueError("export bytes/hash mismatch")
    raw = path.read_bytes()
    import hashlib
    if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("export changed during authentication")
    artifact = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    required = {"schema", "head", "source_lock_sha256", "base_checkpoint_sha256",
                "dataset_sha256", "config_sha256", "validation", "quality_claim", "deployment"}
    if (not isinstance(artifact, dict) or set(artifact) != required
            or artifact["schema"] != "correspondence_head_export_v1"
            or artifact["source_lock_sha256"] != source_hash
            or artifact["base_checkpoint_sha256"] != runtime.GOTRACK_CHECKPOINT
            or artifact["dataset_sha256"] != item["dataset_sha256"]
            or artifact["config_sha256"] != item["config_sha256"]
            or artifact["quality_claim"] is not False
            or artifact["deployment"] != "explicit_cloud_research_only_pose_head_replacement"
            or not isinstance(artifact["head"], dict) or not artifact["head"]
            or not isinstance(artifact["validation"], dict)):
        raise ValueError("export schema/metadata/base/source/config/dataset mismatch")
    data.canonical_hash(artifact["validation"])
    for tensor in artifact["head"].values():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("invalid/nonfinite head state")
    return artifact


def run(manifest, plan_path, plan_sha256, output, *, cloud_execution=False, seed=521):
    """Run input-only cached heads on CPU cloud, then synthetic VAL scoring.

    Plan export dataset/config pins refer to each head's original training run;
    the top-level dataset pin refers to the evaluation manifest (e.g. union167).
    Test and train archives are never read. All declared VAL failures are counted.
    """
    runtime.cloud_guard(cloud_execution)  # Before Torch import or model operations.
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be uint32")
    manifest = _regular(manifest)
    ds = data.load_dataset(manifest, inspect_arrays=False)
    if ds.document["provenance"]["label_origin"] != "synthetic_renderer_training_only":
        raise ValueError("this diagnostic requires synthetic renderer labels")
    plan_path, plan = _plan(plan_path, plan_sha256, ds)
    source = runtime.verify_sources()
    source_pins = {name: data.sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    rows = ds.rows("val", available=False)
    if len(rows) > 24:
        raise ValueError("bounded diagnostic permits at most 24 requested VAL rows")
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output must have nonsymlink parents")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    versions = runtime.runtime_versions()
    pins = dict(dataset_sha256=ds.identity, manifest_sha256=data.sha256(manifest),
                plan_sha256=plan_sha256, feature_recipe_sha256=plan["feature_recipe_sha256"],
                source=source, source_files=source_pins, versions=versions, seed=seed,
                python_version=platform.python_version(), opencv_version=cv2.__version__,
                features={r["id"]: r["features"] for r in rows if r["status"] == "available"})
    pins_hash = _write(output, "inputs.json", pins)
    sealed = {}
    for item in [None, *plan["exports"]]:
        head = artifact = None
        name = "baseline" if item is None else item["name"]
        records = []
        try:
            if item is not None:
                artifact = _export(plan_path, item, plan["source_lock_sha256"], torch)
                head = runtime.load_head(initialize=False, device="cpu")
                head.load_state_dict(artifact["head"], strict=True)
                head.requires_grad_(False).eval()
            with torch.inference_mode():
                for row in rows:
                    entry = dict(id=row["id"], object_id=row["object_id"], status=row["status"])
                    if row["status"] != "available":
                        entry["failure_reason"] = row["failure_reason"]
                    else:
                        features = _features(ds, row)  # Never Dataset.arrays: it opens labels.
                        if head is None:
                            flow, confidence = features["baseline_flow"], features["baseline_confidence"]
                        else:
                            tokens = [None] * 12
                            for hook in data.HOOKS:
                                tokens[hook] = torch.from_numpy(features[f"tokens_{hook}"][None])
                            flow_tensor, confidence_tensor = head(tokens, (data.SIZE, data.SIZE))
                            flow = flow_tensor[0].detach().float().cpu().numpy()
                            confidence = confidence_tensor[0].detach().float().cpu().numpy()
                            del tokens, flow_tensor, confidence_tensor
                        entry["prediction"] = predict_pose(features, flow, confidence, seed)
                        del features, flow, confidence
                    records.append(entry)
            filename = f"predictions-{name}.json"
            digest = _write(output, filename, dict(schema="cached_crop_pose_branch_v1",
                name=name, inputs_sha256=pins_hash, model=item, rows=records))
            sealed[name] = dict(path=filename, sha256=digest)
        finally:
            del head, artifact
            gc.collect()
    seal_hash = _write(output, "prediction-seal.json", dict(schema="cached_crop_pose_seal_v1",
        inputs_sha256=pins_hash, branches=sealed, labels_opened=False))
    # Only this section may open labels. Re-read authenticated persisted predictions.
    scores = {}
    for name, receipt in sealed.items():
        path = output / receipt["path"]
        if data.sha256(path) != receipt["sha256"]:
            raise ValueError("sealed prediction file changed")
        branch = data.read_json(path)
        if [r["id"] for r in branch["rows"]] != [r["id"] for r in rows]:
            raise ValueError("sealed prediction inventory differs")
        scored, pose_count = [], 0
        for row, entry in zip(rows, branch["rows"]):
            if row["status"] != "available":
                scored.append(entry)
                continue
            features = _features(ds, row)
            labels = data.load_npz(_regular(data.descriptor(ds.root, row["labels"], verify=True)), data.LABEL_KEYS)
            score = score_pose(features, entry["prediction"], labels)
            pose_count += int(entry["prediction"]["crop_camera_from_object_mm"] is not None)
            scored.append(dict(id=row["id"], object_id=row["object_id"],
                               fit_state=entry["prediction"]["fit_state"], **score))
        scores[name] = dict(requested_val_rows=len(rows), available_feature_rows=len(ds.rows("val")),
                            pose_available_rows=pose_count, failed_rows=len(rows) - pose_count,
                            pose_availability=pose_count / len(rows), rows=scored)
    if (data.sha256(manifest) != pins["manifest_sha256"]
            or data.sha256(plan_path) != plan_sha256
            or data.sha256(output / "inputs.json") != pins_hash
            or data.sha256(output / "prediction-seal.json") != seal_hash
            or source_pins != {name: data.sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}
            or runtime.verify_sources() != source):
        raise ValueError("input/source/seal identity changed during diagnostic")
    report = dict(schema="cached_crop_pose_diagnostic_v1", status="complete", scope=SCOPE,
                  prediction_seal_sha256=seal_hash, inputs_sha256=pins_hash, branches=scores,
                  test_arrays_opened=False, quality_claim=False, production_replay=False,
                  rotation_accuracy_available=False, independent_accuracy_verified=False)
    _write(output, "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True,
                        help="Strict cached_crop_pose_plan_v1 with dataset/recipe/source pins and exports")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    run(args.manifest, args.plan, args.plan_sha256, args.output_root,
        cloud_execution=args.cloud_execution, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
