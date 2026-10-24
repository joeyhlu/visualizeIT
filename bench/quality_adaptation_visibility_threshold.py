"""Evaluate one TRAIN-fitted visibility threshold for the existing GoTrack head.

The evaluator runs only the installed DPT pose head over authenticated cached
decoder tokens. It uses baseline confidence plus known/visible labels to fit one
global float32 threshold on every available TRAIN row, persists that evidence,
then evaluates the frozen threshold on the complete available VAL inventory.
It does not load a base checkpoint or optimizer, read TEST payloads, score flow,
or promote a serving threshold.

Run in an approved Linux cloud process with a hard external watchdog, for example
``timeout --signal=TERM --kill-after=30s 930s`` around this command. The internal
monotonic budget is checked between bounded archive reads and model forwards.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile
from types import SimpleNamespace

import numpy as np

from . import quality_adaptation_data as data
from . import quality_adaptation_export_parity as parity
from . import quality_adaptation_runtime as runtime
from . import quality_gotrack_inference as installer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bench/runtime/gotrack-existing-head-20261020.json"
PINNED_MANIFEST_SHA256 = "69f2ad7e8619881a7023ea2a9543fff30f7df2ece3e764313bf9e80e18a1c190"
PINNED_EXPORT_SHA256 = "520ba2ebe2357098d878dea6ba60e7b74fc9e96ce2010d21c2d73592d5004cd1"
PINNED_EXPORT_BYTES = 80148566
PINNED_DATASET_SHA256 = "66af85861ea84de57ca60460def7021612681f0a5199c41ce96d41424e9e97af"
PINNED_ARTIFACT_ID = "existing-cached-pilot-20261020-285-updates"
SCHEMA = "gotrack_visibility_threshold_receipt_v1"
SIDECAR_SCHEMA = "gotrack_visibility_train_threshold_v1"
SIDECAR_NAME = "train-threshold.json"
RECEIPT_NAME = "receipt.json"
MAX_OUTPUT_BYTES = 2 * 1024**2
MAX_SECONDS = 900
BASELINE_THRESHOLD = np.float32(0.3)
EXPECTED_INVENTORY = {
    "train": {"total": 96, "available": 95, "unavailable": 1},
    "val": {"total": 24, "available": 20, "unavailable": 4},
    "test": {"total": 24},
}
SELECTED_FEATURE_KEYS = {f"tokens_{i}" for i in data.HOOKS} | {"baseline_confidence"}
SELECTED_LABEL_KEYS = {"known", "visible"}


class Budget:
    """A per-process monotonic work budget, independent of user deadlines."""

    def __init__(self, max_seconds: float):
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                or not math.isfinite(max_seconds) or not 0 < max_seconds <= MAX_SECONDS):
            raise ValueError(f"max_seconds must be finite and in (0,{MAX_SECONDS}]")
        self.started = time.monotonic()
        self.max_seconds = float(max_seconds)
        self.check()

    def check(self):
        if time.monotonic() - self.started >= self.max_seconds:
            raise TimeoutError("per-process visibility evaluation budget expired")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_exclusive_json(root: Path, name: str, value: dict) -> tuple[Path, str]:
    """Persist a bounded, new-only artifact and return its exact SHA256."""
    if (root.is_symlink() or not root.is_dir() or name not in (SIDECAR_NAME, RECEIPT_NAME)):
        raise ValueError("output must be a new nonsymlink directory and filename must be reserved")
    target = root / name
    if target.parent.resolve() != root.resolve() or target.is_symlink():
        raise ValueError("output artifact must remain directly inside its output directory")
    encoded = _json_bytes(value)
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("output artifact exceeds the bounded receipt limit")
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return target, hashlib.sha256(encoded).hexdigest()


def inventory(ds) -> dict:
    """Validate the immutable metadata inventory without opening any payload."""
    report = {}
    for split in ("train", "val", "test"):
        rows = ds.rows(split, available=False)
        available = sum(row["status"] == "available" for row in rows)
        report[split] = {"total": len(rows), "available": available,
                         "unavailable": len(rows) - available}
    for split, expected in EXPECTED_INVENTORY.items():
        for key, count in expected.items():
            if report[split][key] != count:
                raise ValueError(f"{split} metadata inventory {key} must be {count}, found {report[split][key]}")
    return report


def selected_rows(ds, split: str) -> list[dict]:
    """Return every available row for the one allowed split, in stable order."""
    if split not in ("train", "val"):
        raise ValueError("visibility evaluation may select only TRAIN or VAL rows")
    rows = sorted(ds.rows(split), key=lambda row: row["id"])
    expected = EXPECTED_INVENTORY[split]["available"]
    if len(rows) != expected or any(row["split"] != split or row["status"] != "available" for row in rows):
        raise ValueError(f"selected {split} rows differ from the full available inventory")
    return rows


def _load_selected_archive(path: Path, *, expected_keys: set[str], selected_keys: set[str]) -> dict[str, np.ndarray]:
    """Check the complete archive contract while materializing only needed arrays."""
    if not selected_keys <= expected_keys:
        raise ValueError("selected archive keys exceed the frozen archive contract")
    if path.stat().st_size > data.MAX_ARCHIVE:
        raise ValueError("archive exceeds byte limit")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if (len(infos) != len(expected_keys)
                or {item.filename for item in infos} != {key + ".npy" for key in expected_keys}
                or sum(item.file_size for item in infos) > data.MAX_ARCHIVE):
            raise ValueError("archive members or expanded size differ from contract")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in selected_keys}


def _validate_confidence(value: np.ndarray, name: str) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32")
            or value.shape != (data.SIZE, data.SIZE) or not np.isfinite(value).all()
            or np.any((value < np.float32(0)) | (value > np.float32(1)))):
        raise ValueError(f"{name} requires finite float32 confidence in [0,1] with shape 280x280")
    return value


def _validate_binary_support(value: np.ndarray, name: str) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("bool")
            or value.shape != (data.SIZE, data.SIZE)):
        raise ValueError(f"{name} requires bool support with shape 280x280")
    return value


def load_visibility_inputs(ds, row: dict) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Load tokens, frozen confidence, and known/visible labels for one row.

    Flow labels, reference poses, unrelated feature arrays, and TEST payloads
    are never materialized by this boundary.
    """
    if (row.get("split") not in ("train", "val") or row.get("status") != "available"
            or row not in ds.rows(row.get("split"))):
        raise ValueError("only selected available TRAIN/VAL rows may be opened")
    feature_path = data.descriptor(ds.root, row["features"], verify=True)
    features = _load_selected_archive(feature_path, expected_keys=data.FEATURE_KEYS,
                                      selected_keys=SELECTED_FEATURE_KEYS)
    tokens = []
    for hook in range(12):
        token = features.get(f"tokens_{hook}")
        if hook in data.HOOKS:
            if (not isinstance(token, np.ndarray) or token.dtype != np.dtype("float32")
                    or token.shape != (400, 768) or not np.isfinite(token).all()):
                raise ValueError(f"token hook {hook} differs from the frozen decoder contract")
            tokens.append(token)
        else:
            tokens.append(None)
    baseline = _validate_confidence(features["baseline_confidence"], "baseline confidence")
    labels_path = data.descriptor(ds.root, row["labels"], verify=True)
    labels = _load_selected_archive(labels_path, expected_keys=data.LABEL_KEYS,
                                    selected_keys=SELECTED_LABEL_KEYS)
    known = _validate_binary_support(labels["known"], "known")
    visible = _validate_binary_support(labels["visible"], "visible")
    if np.any(visible & ~known) or not visible.any():
        raise ValueError("visible labels must be nonempty and a subset of known support")
    return tokens, baseline, known, visible


def _to_model_tokens(tokens: list, device: str, torch_module) -> list:
    packed = [None] * 12
    for hook in data.HOOKS:
        packed[hook] = torch_module.from_numpy(tokens[hook][None]).to(device)
    return packed


def _checked_confidence(outputs) -> np.ndarray:
    _flow, confidence = parity.checked_outputs(outputs)
    return _validate_confidence(confidence[0].copy(), "adapted confidence")


def _counts(confidence: np.ndarray, known: np.ndarray, visible: np.ndarray,
            threshold: np.float32) -> dict:
    """Count strict float32 decisions on the shared known support only."""
    confidence = _validate_confidence(confidence, "confidence")
    known = _validate_binary_support(known, "known")
    visible = _validate_binary_support(visible, "visible")
    if np.any(visible & ~known):
        raise ValueError("visible support must be a subset of known support")
    if not isinstance(threshold, np.float32) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("decision threshold must be a finite float32 value in [0,1]")
    truth = visible[known]
    predicted = confidence[known] > threshold
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(np.count_nonzero(~predicted & ~truth))
    return {"known_pixels": int(truth.size), "visible_pixels": int(truth.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _valid_confidence_vector(values, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arrays = [values]
    else:
        arrays = list(values)
    checked = []
    for array in arrays:
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype("float32")
                or not np.isfinite(array).all()
                or np.any((array < np.float32(0)) | (array > np.float32(1)))):
            raise ValueError(f"{name} values must be finite float32 confidences in [0,1]")
        checked.append(array.reshape(-1))
    return np.concatenate(checked) if checked else np.empty(0, dtype=np.float32)


def fit_global_threshold(positive_confidences, negative_confidences, *,
                         frozen_train_tp: int, frozen_train_fp: int) -> dict:
    """Fit the largest float32 threshold preserving frozen TRAIN TP, then FP-check.

    For a positive target, the threshold is the immediate float32 predecessor
    of the target-th highest adapted confidence. Strict ``>`` means ties at that
    order statistic are all retained. At baseline TP zero, threshold 1.0 is the
    analytic maximum and predicts no positives. A zero-valued required positive
    is unattainable on the allowed [0,1] threshold domain.
    """
    for name, count in (("frozen_train_tp", frozen_train_tp), ("frozen_train_fp", frozen_train_fp)):
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    positive = _valid_confidence_vector(positive_confidences, "positive adapted confidence")
    negative = _valid_confidence_vector(negative_confidences, "negative adapted confidence")
    if frozen_train_tp > len(positive):
        raise ValueError("frozen TRAIN true positives exceed adapted positive support")
    if frozen_train_tp == 0:
        threshold = np.float32(1.0)
    else:
        positive.partition(len(positive) - frozen_train_tp)
        order_statistic = np.float32(positive[len(positive) - frozen_train_tp])
        if order_statistic == np.float32(0):
            return {"status": "rejected", "threshold": None,
                    "reason": "required TRAIN true-positive count includes zero-confidence positives, unattainable with strict > on [0,1]",
                    "positive_support": int(len(positive)), "negative_support": int(len(negative)),
                    "frozen_train_tp_target": frozen_train_tp,
                    "frozen_train_fp_limit": frozen_train_fp}
        threshold = np.float32(np.nextafter(order_statistic, np.float32(-np.inf)))
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        return {"status": "rejected", "threshold": None,
                "reason": "largest recall-feasible threshold falls outside [0,1]",
                "positive_support": int(len(positive)), "negative_support": int(len(negative)),
                "frozen_train_tp_target": frozen_train_tp,
                "frozen_train_fp_limit": frozen_train_fp}
    adapted_tp = int(np.count_nonzero(positive > threshold))
    adapted_fp = int(np.count_nonzero(negative > threshold))
    result = {"status": "passed" if adapted_tp >= frozen_train_tp and adapted_fp <= frozen_train_fp else "rejected",
              "threshold": float(threshold),
              "threshold_float32_bits": f"0x{int(threshold.view(np.uint32)):08x}",
              "positive_order_statistic": (float(order_statistic) if frozen_train_tp else None),
              "positive_support": int(len(positive)), "negative_support": int(len(negative)),
              "frozen_train_tp_target": frozen_train_tp,
              "frozen_train_fp_limit": frozen_train_fp,
              "adapted_train_tp": adapted_tp, "adapted_train_fp": adapted_fp,
              "threshold_rule": "largest_representable_float32_threshold_preserving_frozen_TRAIN_TP; strict >; ties retained",
              "false_positive_optimality": "largest recall-feasible threshold minimizes FP because predictions are monotone decreasing in threshold"}
    if result["status"] != "passed":
        result["reason"] = "adapted TRAIN false positives exceed frozen TRAIN false positives at the minimum-FP recall-feasible threshold"
    return result


def _selection_payload(rows: list[dict]) -> dict:
    return {"split": "train", "selection": "all_available_TRAIN_rows_sorted_lexically_by_id",
            "row_ids": [row["id"] for row in rows],
            "rows": [{"id": row["id"], "features": dict(row["features"]),
                      "labels": dict(row["labels"])} for row in rows]}


def _runtime_identity(record: dict) -> dict:
    versions = runtime.runtime_versions()
    source = runtime.verify_sources()
    if (runtime.ROOT.resolve() != ROOT.resolve()
            or source["source_lock_sha256"] != record["training_lock_sha256"]
            or source["upstream_revision"] != record["upstream_revision"]
            or runtime.GOTRACK_CHECKPOINT != record["base_checkpoint_sha256"]):
        raise ValueError("runtime root/source identity differs from authenticated head artifact")
    return {"versions": versions, "source": source}


def _sync(torch_module, device: str):
    if device == "cuda":
        torch_module.cuda.synchronize()


def _release(torch_module, device: str):
    gc.collect()
    if device == "cuda":
        torch_module.cuda.empty_cache()


def _run_split(rows, ds, network, torch_module, device: str, budget: Budget,
               *, split: str, receipt: dict):
    pooled = {key: 0 for key in ("known_pixels", "visible_pixels", "tp", "fp", "fn", "tn")}
    frame_rows = []
    positive_arrays, negative_arrays = [], []
    timings = []
    inference = torch_module.inference_mode() if hasattr(torch_module, "inference_mode") else nullcontext()
    with inference:
        for row in rows:
            budget.check()
            receipt["active_row_id"] = row["id"]
            tokens, baseline, known, visible = load_visibility_inputs(ds, row)
            payload_counter = "training_payload_reads" if split == "train" else "validation_payload_reads"
            receipt[payload_counter] = receipt.get(payload_counter, 0) + 1
            baseline_counts = _counts(baseline, known, visible, BASELINE_THRESHOLD)
            model_tokens = _to_model_tokens(tokens, device, torch_module)
            budget.check()
            _sync(torch_module, device)
            started = time.monotonic()
            outputs = network.pose_head(model_tokens, (data.SIZE, data.SIZE))
            _sync(torch_module, device)
            timings.append(time.monotonic() - started)
            adapted = _checked_confidence(outputs)
            del tokens, model_tokens, outputs
            adapted_fixed_counts = _counts(adapted, known, visible, BASELINE_THRESHOLD)
            for key in pooled:
                pooled[key] += baseline_counts[key]
            frame_rows.append({"id": row["id"], "object_id": row["object_id"],
                               "known_pixels": baseline_counts["known_pixels"],
                               "visible_pixels": baseline_counts["visible_pixels"],
                               "frozen_baseline_gt_0_3": baseline_counts,
                               "adapted_gt_0_3": adapted_fixed_counts})
            support = visible[known]
            confidence = adapted[known]
            positive_arrays.append(confidence[support].copy())
            negative_arrays.append(confidence[~support].copy())
            budget.check()
            del baseline, known, visible, adapted, confidence, support
    receipt.pop("active_row_id", None)
    if split == "train":
        receipt["train_inference_seconds"] = timings
    else:
        receipt["validation_inference_seconds"] = timings
    return pooled, frame_rows, positive_arrays, negative_arrays


def _train_frame_threshold_metrics(frame_rows, positive_arrays, negative_arrays, threshold):
    result = []
    for frame, positive, negative in zip(frame_rows, positive_arrays, negative_arrays):
        result.append({"id": frame["id"], "object_id": frame["object_id"],
                       "known_pixels": frame["known_pixels"], "visible_pixels": frame["visible_pixels"],
                       "frozen_baseline_gt_0_3": frame["frozen_baseline_gt_0_3"],
                       "adapted_gt_0_3": frame["adapted_gt_0_3"],
                       "adapted_fitted_threshold": {"tp": int(np.count_nonzero(positive > threshold)),
                           "fp": int(np.count_nonzero(negative > threshold))}})
    return result


def execute(record, context, dataset: Path, *, device: str, budget: Budget, receipt: dict,
            output: Path) -> dict:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    budget.check()
    runtime_info = _runtime_identity(record)
    ds = data.load_dataset(dataset, inspect_arrays=False)
    if ds.identity != record["dataset_sha256"] or ds.identity != PINNED_DATASET_SHA256:
        raise ValueError("dataset identity differs from the authenticated existing-head artifact")
    from .quality_adaptation_train import validate_feature_receipt
    validate_feature_receipt(ds)
    counts = inventory(ds)
    train_rows = selected_rows(ds, "train")
    val_rows = selected_rows(ds, "val")
    train_selection = _selection_payload(train_rows)
    train_selection_sha256 = data.canonical_hash(train_selection)
    source_code_sha256 = data.sha256(Path(__file__))
    receipt.update(runtime=runtime_info, dataset_sha256=ds.identity,
                   inventory=counts, feature_receipt_verified=True,
                   label_origin=ds.document["provenance"]["label_origin"],
                   selected_train_rows=len(train_rows), selected_validation_rows=len(val_rows),
                   selected_train_row_ids=train_selection["row_ids"],
                   selected_validation_row_ids=[row["id"] for row in val_rows],
                   train_selection_sha256=train_selection_sha256,
                   evaluator_code_sha256=source_code_sha256,
                   validation_payload_reads=0, test_payload_reads=0,
                   label_arrays_passed_to_model=0, reference_pose_reads=0,
                   flow_metrics_computed=False, serving_threshold_promoted=False)
    budget.check()
    # Authenticate the exact export before importing Torch or constructing a head.
    raw = parity.authenticated_bytes(context.head_path, record["export_bytes"],
                                     record["export_sha256"], "head export")
    del raw
    if record["export_sha256"] != PINNED_EXPORT_SHA256 or record["export_bytes"] != PINNED_EXPORT_BYTES:
        raise ValueError("export is not the one authenticated existing research head")

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    skeleton = runtime.load_head(initialize=False, device="cpu")
    network = SimpleNamespace(pose_head=skeleton)
    del skeleton
    try:
        network, provenance = installer.install_adapted_head(
            network, context, device=device, torch_module=torch)
        receipt["installation"] = provenance
        receipt["head_loaded_with_initialize_false"] = True
        receipt["model_device"] = device

        train_frozen, train_frames, train_positive, train_negative = _run_split(
            train_rows, ds, network, torch, device, budget, split="train", receipt=receipt)
        # Global fit uses only selected available TRAIN rows. The strict baseline
        # cutoff is explicitly float32 to match the existing serving comparison.
        baseline_tp = train_frozen["tp"]
        baseline_fp = train_frozen["fp"]
        fit = fit_global_threshold(train_positive, train_negative,
                                   frozen_train_tp=baseline_tp, frozen_train_fp=baseline_fp)
        train_evidence = {
            "schema": SIDECAR_SCHEMA,
            "status": "passed" if fit["status"] == "passed" else "rejected",
            "quality_claim": False,
            "serving_threshold_promoted": False,
            "validation_payloads_read": False,
            "test_payloads_read": False,
            "artifact": {"artifact_id": record["artifact_id"],
                         "manifest_sha256": context.manifest_sha256,
                         "export_sha256": record["export_sha256"],
                         "export_bytes": record["export_bytes"]},
            "dataset_sha256": ds.identity,
            "evaluator_code_sha256": source_code_sha256,
            "source_lock_sha256": record["training_lock_sha256"],
            "upstream_revision": record["upstream_revision"],
            "train_selection_sha256": train_selection_sha256,
            "train_selection": train_selection,
            "threshold_fit": fit,
            "frozen_baseline_threshold": {"intent": 0.3,
                "float32_value": float(BASELINE_THRESHOLD),
                "float32_bits": f"0x{int(BASELINE_THRESHOLD.view(np.uint32)):08x}",
                "comparison": "strict > on native float32 confidence and identical known support"},
            "frozen_train_pooled": train_frozen,
            "train_per_frame": (_train_frame_threshold_metrics(train_frames, train_positive,
                train_negative, np.float32(fit["threshold"])) if fit.get("threshold") is not None else train_frames),
            "fit_scope": "all_95_available_TRAIN_rows; one global float32 threshold",
            "threshold_usage": "experimental evaluation only; not a serving default",
        }
        if fit["status"] != "passed":
            train_evidence["rejection_reason"] = fit.get("reason", "TRAIN threshold constraint failed")
        sidecar_path, sidecar_sha256 = _write_exclusive_json(output, SIDECAR_NAME, train_evidence)
        receipt.update(train_threshold_sidecar=SIDECAR_NAME,
                       train_threshold_sidecar_sha256=sidecar_sha256,
                       train_threshold_sidecar_bytes=sidecar_path.stat().st_size,
                       frozen_train_pooled=train_frozen,
                       fitted_train_threshold=fit)
        if fit["status"] != "passed":
            receipt["status"] = "rejected_training"
            receipt["rejection_reason"] = fit.get("reason", "TRAIN threshold constraint failed")
            receipt["validation_payload_reads"] = 0
            return receipt

        # The fsynced TRAIN sidecar above is immutable in this exclusive output
        # directory and includes the threshold plus all TRAIN selection evidence.
        budget.check()
        val_frozen, val_frames, val_positive, val_negative = _run_split(
            val_rows, ds, network, torch, device, budget, split="val", receipt=receipt)
        threshold = np.float32(fit["threshold"])
        val_adapted_fitted = {key: 0 for key in ("known_pixels", "visible_pixels", "tp", "fp", "fn", "tn")}
        val_fitted_frames = []
        for frame, positive, negative in zip(val_frames, val_positive, val_negative):
            fitted_counts = {"tp": int(np.count_nonzero(positive > threshold)),
                             "fp": int(np.count_nonzero(negative > threshold)),
                             "fn": int(frame["visible_pixels"] - np.count_nonzero(positive > threshold)),
                             "tn": int(frame["known_pixels"] - frame["visible_pixels"] - np.count_nonzero(negative > threshold)),
                             "known_pixels": frame["known_pixels"],
                             "visible_pixels": frame["visible_pixels"]}
            for key in val_adapted_fitted:
                val_adapted_fitted[key] += fitted_counts[key]
            val_fitted_frames.append({"id": frame["id"], "object_id": frame["object_id"],
                                      "known_pixels": frame["known_pixels"],
                                      "visible_pixels": frame["visible_pixels"],
                                      "frozen_baseline_gt_0_3": frame["frozen_baseline_gt_0_3"],
                                      "adapted_gt_0_3": frame["adapted_gt_0_3"],
                                      "adapted_train_fitted_threshold": fitted_counts})
        accepted = (val_adapted_fitted["tp"] >= val_frozen["tp"]
                    and val_adapted_fitted["fp"] <= val_frozen["fp"])
        receipt.update(validation_payload_reads=len(val_rows),
            validation_pooled={"frozen_baseline_gt_0_3": val_frozen,
                               "adapted_gt_0_3": {key: sum(row["adapted_gt_0_3"][key] for row in val_frames)
                                                  for key in val_adapted_fitted},
                               "adapted_train_fitted_threshold": val_adapted_fitted},
            validation_per_frame=val_fitted_frames,
            validation_inventory={"available_evaluated": len(val_rows), "unavailable_accounted": 4,
                                  "total_metadata_rows": 24,
                                  "selection": "all_available_VAL_rows_sorted_lexically_by_id"},
            validation_candidate_accepted=bool(accepted),
            threshold_source="frozen_TRAIN_only",
            threshold_float32=float(threshold),
            threshold_float32_bits=f"0x{int(threshold.view(np.uint32)):08x}",
            validation_decision="pooled adapted TP >= frozen TP and pooled adapted FP <= frozen FP",
            status="passed" if accepted else "rejected_validation")
        if not accepted:
            receipt["rejection_reason"] = "frozen TRAIN threshold did not preserve or improve pooled VAL TP and FP versus frozen >0.3"
        return receipt
    finally:
        if "network" in locals():
            del network
        _release(torch, device)


def run(*, artifact_manifest=MANIFEST, artifact_manifest_sha256=PINNED_MANIFEST_SHA256,
        export: Path, dataset: Path, output: Path, max_seconds: float,
        device="cpu", cloud_execution=False) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"schema": SCHEMA, "status": "failed", "candidate_accepted": False,
               "quality_claim": False, "serving_threshold_promoted": False,
               "full_network_forward_executed": False, "pose_replay": False,
               "flow_metrics_computed": False, "label_arrays_passed_to_model": 0,
               "reference_pose_reads": 0, "test_payload_reads": 0,
               "validation_payload_reads": 0, "device": device,
               "max_seconds": (max_seconds if isinstance(max_seconds, (int, float))
                               and not isinstance(max_seconds, bool) and math.isfinite(max_seconds)
                               else str(max_seconds)),
               "watchdog": "per-process monotonic budget; external Linux timeout recommended",
               "evidence_scope": "existing_DPT_head_cached_tokens_and_visibility_support_only",
               "mocked_execution": False}
    try:
        runtime.cloud_guard(cloud_execution)  # Before Torch import or model construction.
        budget = Budget(max_seconds)
        if artifact_manifest_sha256 != PINNED_MANIFEST_SHA256:
            raise ValueError("only the pinned existing-head artifact manifest is permitted")
        record, manifest_sha256 = parity.read_manifest(Path(artifact_manifest), artifact_manifest_sha256)
        if (record["artifact_id"] != PINNED_ARTIFACT_ID
                or record["export_sha256"] != PINNED_EXPORT_SHA256
                or record["export_bytes"] != PINNED_EXPORT_BYTES
                or record["dataset_sha256"] != PINNED_DATASET_SHA256):
            raise ValueError("artifact manifest does not identify the pinned existing research head")
        receipt["artifact"] = {"artifact_id": record["artifact_id"],
                               "manifest_sha256": manifest_sha256,
                               "export_sha256": record["export_sha256"],
                               "export_bytes": record["export_bytes"],
                               "dataset_sha256": record["dataset_sha256"]}
        context = parity.context_for_export(record, manifest_sha256, Path(export))
        result = execute(record, context, Path(dataset), device=device, budget=budget,
                         receipt=receipt, output=output)
        receipt = result
        if receipt["status"] == "passed":
            receipt["candidate_accepted"] = True
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["candidate_accepted"] = False
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        try:
            _write_exclusive_json(output, RECEIPT_NAME, receipt)
        except Exception as exc:
            # The returned object still reports the write failure; never replace
            # an existing receipt or a training threshold sidecar.
            receipt["receipt_write_error"] = {"type": type(exc).__name__, "message": str(exc)[:512]}
            receipt["status"] = "failed"
            receipt["candidate_accepted"] = False
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--artifact-manifest-sha256", default=PINNED_MANIFEST_SHA256)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True,
                        help="per-process monotonic work budget, at most 900 seconds")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cloud-execution", action="store_true")
    args = vars(parser.parse_args(argv))
    result = run(**args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") == "passed" and "receipt_write_error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
