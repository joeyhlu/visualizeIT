"""Audited inputs for experimental correspondence-head adaptation.

This module is import-safe without Torch. Labels are separate from model inputs;
training never opens sealed test archives. All distances are crop pixels or metres.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


SCHEMA = "correspondence_head_dataset_v1"
HOOKS = (2, 5, 8, 11)
SIZE = 280
CONVENTION = {
    "flow": "template_to_query_displacement_xy_crop_pixels",
    "pixels": "integer_centers_v1",
    "object_xyz": "object_frame_metres",
    "camera": "opencv_x_right_y_down_z_forward",
    "crop_hw": [SIZE, SIZE],
}
FEATURE_KEYS = {f"tokens_{i}" for i in HOOKS} | {
    "template_mask", "template_xyz_m", "query_K", "baseline_flow", "baseline_confidence"}
CROP_KEYS = {"query_rgb", "template_rgb", "template_mask", "template_xyz_m", "query_K"}
LABEL_KEYS = {"flow", "visible", "known"}
MAX_JSON = 8 * 1024**2
MAX_ARCHIVE = 16 * 1024**2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def query_rgb_hash(rgb: np.ndarray) -> str:
    """SHA256 of validated uint8[280,280,3] query crop pixels in C order.

    This hashes decoded RGB content, not a JPEG file or the original full image.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (SIZE, SIZE, 3):
        raise ValueError("query RGB hash requires uint8[280,280,3] crop pixels")
    return hashlib.sha256(rgb.tobytes(order="C")).hexdigest()


def crop_rgb_hash(rgb: np.ndarray) -> str:
    """Versioned identity for exact inference float crops; retain byte v1 hashes."""
    if rgb.dtype == np.uint8:
        return query_rgb_hash(rgb)
    _array(rgb, (SIZE, SIZE, 3), "float32", "float RGB crop")
    if np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("float RGB crop must be in [0,1]")
    quantized = np.floor(rgb * 255 + .5).astype(np.uint8)
    if np.array_equal(rgb, quantized.astype(np.float32) / 255.):
        # The same pixels cannot cross splits merely by changing their encoding.
        return query_rgb_hash(quantized)
    return hashlib.sha256(b"gotrack_rgb_float32_0_1_v1\0" +
                          rgb.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    if Path(path).stat().st_size > MAX_JSON:
        raise ValueError(f"JSON exceeds {MAX_JSON} bytes: {path}")
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=reject)


def require_hash(value, name="sha256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def local_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact path must be a relative POSIX path")
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or ":" in relative:
        raise ValueError("artifact path escapes dataset")
    root = root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("artifact path escapes dataset")
    return target


def descriptor(root: Path, value: dict, *, verify: bool) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError("artifact descriptor requires exactly path, sha256, bytes")
    require_hash(value["sha256"])
    if type(value["bytes"]) is not int or not 0 < value["bytes"] <= MAX_ARCHIVE:
        raise ValueError("artifact bytes outside bounded archive size")
    path = local_path(root, value["path"])
    if verify and (not path.is_file() or path.stat().st_size != value["bytes"] or sha256(path) != value["sha256"]):
        raise ValueError(f"artifact identity mismatch: {value['path']}")
    return path


def load_npz(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    """Reject pickle/object arrays, archive bombs and unexpected annotation fields."""
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("archive exceeds byte limit")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if (len(infos) != len(keys) or {i.filename for i in infos} != {k + ".npy" for k in keys}
                or sum(i.file_size for i in infos) > MAX_ARCHIVE):
            raise ValueError("archive members or expanded size differ from contract")
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k] for k in keys}
    return values


def _array(a, shape, dtype, name):
    if a.shape != shape or a.dtype != np.dtype(dtype) or not np.isfinite(a).all():
        raise ValueError(f"{name} requires finite {dtype} array with shape {shape}")


def validate_geometry(v: dict) -> None:
    _array(v["template_mask"], (SIZE, SIZE), "bool", "template_mask")
    if not v["template_mask"].any():
        raise ValueError("template mask has no supported object pixels")
    _array(v["template_xyz_m"], (SIZE, SIZE, 3), "float32", "template_xyz_m")
    _array(v["query_K"], (3, 3), "float64", "query_K")
    k = v["query_K"]
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0., 0., 1.]) or k[1, 0] != 0:
        raise ValueError("query_K must be an OpenCV pinhole intrinsic matrix in crop pixels")


def validate_features(v: dict) -> None:
    if set(v) != FEATURE_KEYS:
        raise ValueError("feature keys differ from frozen decoder contract")
    validate_geometry(v)
    for hook in HOOKS:
        _array(v[f"tokens_{hook}"], (400, 768), "float32", f"tokens_{hook}")
    _array(v["baseline_flow"], (2, SIZE, SIZE), "float32", "baseline_flow")
    _array(v["baseline_confidence"], (SIZE, SIZE), "float32", "baseline_confidence")
    if np.any((v["baseline_confidence"] < 0) | (v["baseline_confidence"] > 1)):
        raise ValueError("baseline confidence outside [0,1]")


def validate_crops(v: dict) -> None:
    if set(v) != CROP_KEYS:
        raise ValueError("crop archive contains unexpected fields (labels/poses are prohibited)")
    validate_geometry(v)
    for key in ("query_rgb", "template_rgb"):
        if v[key].dtype not in (np.dtype("uint8"), np.dtype("float32")):
            raise ValueError("RGB crops require uint8 or exact float32 [0,1]")
        crop_rgb_hash(v[key])
    if v["query_rgb"].dtype != v["template_rgb"].dtype:
        raise ValueError("query/template RGB encodings must match")


def validate_labels(v: dict, template_mask: np.ndarray) -> None:
    if set(v) != LABEL_KEYS:
        raise ValueError("label archive keys must be flow, visible, known")
    _array(v["flow"], (2, SIZE, SIZE), "float32", "label flow")
    for key in ("visible", "known"):
        _array(v[key], (SIZE, SIZE), "bool", key)
    if np.any(v["visible"] & ~v["known"]) or np.any(v["known"] & ~template_mask):
        raise ValueError("visible must be a subset of known, and known of template_mask")
    if not v["visible"].any():
        raise ValueError("no visible supervised correspondence; record this row as unavailable")
    yy, xx = np.mgrid[:SIZE, :SIZE]
    target = v["flow"] + np.stack([xx, yy])
    if np.any(v["visible"] & ((target[0] < 0) | (target[0] > SIZE - 1) |
                              (target[1] < 0) | (target[1] > SIZE - 1))):
        raise ValueError("visible targets must be within the query crop integer centers")


@dataclass(frozen=True)
class Dataset:
    root: Path
    document: dict
    identity: str

    def rows(self, split: str, *, available=True) -> list[dict]:
        return [r for r in self.document["rows"] if r["split"] == split
                and (not available or r["status"] == "available")]

    def arrays(self, row: dict) -> tuple[dict, dict]:
        if row["split"] == "test":
            raise ValueError("sealed test arrays are inaccessible to adaptation training")
        f = load_npz(descriptor(self.root, row["features"], verify=True), FEATURE_KEYS)
        validate_features(f)
        y = load_npz(descriptor(self.root, row["labels"], verify=True), LABEL_KEYS)
        validate_labels(y, f["template_mask"])
        return f, y


def load_dataset(path: Path, *, inspect_arrays=True) -> Dataset:
    doc = read_json(path)
    if not isinstance(doc, dict) or set(doc) != {"schema", "conventions", "provenance", "rows"}:
        raise ValueError("dataset manifest fields differ from schema")
    if doc["schema"] != SCHEMA or doc["conventions"] != CONVENTION:
        raise ValueError("dataset schema/units/camera conventions mismatch")
    p = doc["provenance"]
    if (not isinstance(p, dict) or set(p) != {"feature_recipe_sha256", "source_lock_sha256", "annotation_inputs", "label_origin"}
            or p["annotation_inputs"] != [] or p["label_origin"] not in ("independent_annotations", "synthetic_renderer_training_only")):
        raise ValueError("explicit input-only feature provenance and label origin required")
    for key in ("feature_recipe_sha256", "source_lock_sha256"):
        require_hash(p[key], key)
    rows = doc["rows"]
    if not isinstance(rows, list) or not 3 <= len(rows) <= 10000:
        raise ValueError("manifest requires 3..10000 rows including sealed test inventory")
    ds = Dataset(path.resolve().parent, doc, canonical_hash(doc))
    # Does training contain indirect information about test samples? Reject shared
    # objects, sequence/capture families, query images, and feature content.
    # Equal labels alone are not leakage (e.g. zero flow in two independent views).
    # This validates declared identity, not the truth of an untrusted annotation.
    ownership = {k: {} for k in ("object_id", "sequence_id", "capture_group", "query_rgb_sha256", "artifact")}
    ids = set()
    for row in rows:
        base = {"id", "object_id", "sequence_id", "capture_group", "query_rgb_sha256", "split", "status"}
        unknown = row.get("query_rgb_sha256") is None
        expected = base | ({"features", "labels"} if row.get("status") == "available" else {"failure_reason"})
        if unknown:
            expected |= {"unavailable_provenance"}
        if set(row) != expected or row["split"] not in ("train", "val", "test") or row["status"] not in ("available", "unavailable"):
            raise ValueError("invalid row fields, status or split")
        if row["id"] in ids:
            raise ValueError("duplicate row id")
        ids.add(row["id"])
        for key in base - {"split", "status", "query_rgb_sha256"}:
            if not isinstance(row[key], str) or not row[key] or len(row[key]) > 256:
                raise ValueError(f"invalid row {key}")
        if unknown:
            if (row["status"] != "unavailable" or not isinstance(row["unavailable_provenance"], str)
                    or not row["unavailable_provenance"] or len(row["unavailable_provenance"]) > 1024):
                raise ValueError("unknown RGB identity requires unavailable status and explicit provenance")
        else:
            require_hash(row["query_rgb_sha256"])
        for key, seen in ownership.items():
            if key == "artifact":
                continue
            if key == "query_rgb_sha256" and unknown:
                continue
            old = seen.setdefault(row[key], row["split"])
            if old != row["split"]:
                raise ValueError(f"split leakage: shared {key} between {old} and {row['split']}")
        if row["status"] == "unavailable":
            if not isinstance(row["failure_reason"], str) or not row["failure_reason"]:
                raise ValueError("unavailable row needs a failure_reason")
            continue
        for key in ("features", "labels"):
            descriptor(ds.root, row[key], verify=False)  # Never open sealed test payloads.
            if key == "features":
                h = row[key]["sha256"]
                old = ownership["artifact"].setdefault(h, row["split"])
                if old != row["split"]:
                    raise ValueError("split leakage: duplicate archive content")
        if row["features"]["path"] == row["labels"]["path"]:
            raise ValueError("features and supervision must be separate files")
        if inspect_arrays and row["split"] != "test":
            ds.arrays(row)
    for split in ("train", "val", "test"):
        if not ds.rows(split, available=False):
            raise ValueError(f"missing {split} inventory")
        if split != "test" and not ds.rows(split):
            raise ValueError(f"no available {split} rows")
    return ds


def data_report(ds: Dataset) -> dict:
    """Metadata availability plus TRAIN EDA; held-out pixel data stays unopened."""
    report = {"dataset_sha256": ds.identity, "splits": {}, "train_eda": {}}
    for split in ("train", "val", "test"):
        rows = ds.rows(split, available=False)
        report["splits"][split] = {
            "total": len(rows), "available": sum(r["status"] == "available" for r in rows),
            "objects": len({r["object_id"] for r in rows}),
            "capture_groups": len({r["capture_group"] for r in rows}),
            "failures": dict(Counter(r["failure_reason"] for r in rows if r["status"] != "available")),
        }
    visible_fractions, flow_means = [], []
    for row in ds.rows("train"):
        _, y = ds.arrays(row)
        visible_fractions.append(float(y["visible"].sum() / y["known"].sum()))
        flow_means.append(float(np.linalg.norm(y["flow"], axis=0)[y["visible"]].mean()))
    report["train_eda"] = {
        "missing_nonfinite_values": 0,  # Rejected by validation, never imputed.
        "visible_fraction_per_frame_quantiles": np.quantile(visible_fractions, [0, .25, .5, .75, 1]).tolist(),
        "mean_flow_magnitude_px_per_frame_quantiles": np.quantile(flow_means, [0, .25, .5, .75, 1]).tolist(),
        "augmentation": "none; cached geometry and decoder features must remain aligned",
    }
    return report


def flow_metrics(prediction: np.ndarray, target: np.ndarray, visible: np.ndarray) -> dict:
    _array(prediction, (2, SIZE, SIZE), "float32", "predicted flow")
    error = np.linalg.norm(prediction.astype(np.float64) - target, axis=0)[visible]
    if not error.size:
        raise ValueError("no visible pixels for EPE")
    return {"epe_px": float(error.mean()), "within_1px": float((error < 1).mean()),
            "within_3px": float((error < 3).mean()), "visible_pixels": int(error.size)}


def visibility_metrics(confidence: np.ndarray, labels: dict) -> dict:
    """Paired fixed-support visibility BCE/Brier and historical >.3 decision."""
    _array(confidence, (SIZE, SIZE), "float32", "visibility confidence")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("visibility confidence outside [0,1]")
    known = labels["known"]
    p = confidence[known].astype(np.float64)
    truth = labels["visible"][known]
    if not len(p):
        raise ValueError("visibility metric requires known support")
    clipped = p.clip(1e-6, 1 - 1e-6)
    # Preserve the serving cutoff in confidence's native float32 dtype before
    # promoting probabilities for stable metric arithmetic.
    decision = confidence[known] > .3
    return {"known_pixels": len(p), "visible_pixels": int(truth.sum()),
            "bce": float(-(truth * np.log(clipped) + (~truth) * np.log1p(-clipped)).mean()),
            "brier": float(np.square(p - truth).mean()), "threshold": .3,
            "true_positive": int((decision & truth).sum()),
            "false_positive": int((decision & ~truth).sum()),
            "false_negative": int((~decision & truth).sum())}


def object_summary(rows: list[dict], *, seed: int) -> dict:
    grouped = {}
    for row in rows:
        if not math.isfinite(row["epe_px"]):
            raise ValueError("nonfinite frame metric")
        grouped.setdefault(row["object_id"], []).append(row["epe_px"])
    if not grouped:
        raise ValueError("no complete validation metrics")
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    result = {"object_macro_epe_px": float(means.mean()), "objects": len(means),
              "frames": len(rows), "bootstrap_95ci_px": None}
    if len(means) > 1:
        rng = np.random.default_rng(seed)
        samples = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
        result["bootstrap_95ci_px"] = np.quantile(samples, [.025, .975]).tolist()
    else:
        result["uncertainty_unavailable_reason"] = "only_one_independent_object"
    return result
