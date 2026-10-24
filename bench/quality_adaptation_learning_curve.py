"""Model-free history diagnostics; composite training loss is not validation EPE."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .quality_kaggle_preflight import read_json, require_digest, write_report


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite diagnostic metric")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _number(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def compare_matched_metrics(records):
    """Require explicit metric definitions, denominators and coverage policies."""
    if not records:
        return []
    groups = {}
    for row in records:
        _finite_tree(row)
        if row.get("split") not in ("train", "validation"):
            raise ValueError("matched metric split must be train or validation")
        step = _number(row.get("steps"), "steps", integer=True)
        for key in ("metric_definition", "units", "aggregation", "coverage_policy", "dataset_manifest_sha256"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"explicit {key} required")
        require_digest(row["dataset_manifest_sha256"])
        available = _number(row.get("available"), "available", integer=True)
        total = _number(row.get("total"), "total", integer=True)
        if total == 0 or available != total:
            raise ValueError("matched gap requires complete matched coverage; report partial availability separately")
        _number(row.get("value"), "metric value")
        pair = groups.setdefault(step, {})
        if row["split"] in pair:
            raise ValueError("duplicate matched metric split/step")
        pair[row["split"]] = row
    gaps = []
    for step, pair in sorted(groups.items()):
        if set(pair) != {"train", "validation"}:
            raise ValueError("matched metrics require both train and validation")
        train, val = pair["train"], pair["validation"]
        for key in ("metric_definition", "units", "aggregation", "coverage_policy", "dataset_manifest_sha256"):
            if train[key] != val[key]:
                raise ValueError(f"incomparable matched metrics: {key}")
        gaps.append({"steps": step, "metric_definition": train["metric_definition"],
                     "units": train["units"], "aggregation": train["aggregation"],
                     "validation_minus_train": val["value"] - train["value"],
                     "train_available": train["available"], "validation_available": val["available"]})
    return gaps


def _paired_frame_map(records, label):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} frame evidence must be a nonempty list")
    indexed = {}
    for frame in records:
        if not isinstance(frame, dict):
            raise ValueError(f"{label} frame must be an object")
        _finite_tree(frame)
        frame_id = frame.get("id")
        object_id = frame.get("object_id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError(f"{label} frame id is required")
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError(f"{label} frame object_id is required")
        if frame_id in indexed:
            raise ValueError(f"duplicate {label} frame id: {frame_id}")

        epe = _number(frame.get("epe_px"), f"{label} EPE")
        visible = _number(frame.get("visible_pixels"), f"{label} visible pixels", integer=True)
        visibility = frame.get("visibility")
        if not isinstance(visibility, dict):
            raise ValueError(f"{label} visibility counts are required")
        nested_visible = _number(visibility.get("visible_pixels"),
                                 f"{label} visibility visible pixels", integer=True)
        known = _number(visibility.get("known_pixels"), f"{label} known pixels", integer=True)
        tp = _number(visibility.get("true_positive"), f"{label} true positives", integer=True)
        fp = _number(visibility.get("false_positive"), f"{label} false positives", integer=True)
        fn = _number(visibility.get("false_negative"), f"{label} false negatives", integer=True)
        threshold = _number(visibility.get("threshold"), f"{label} visibility threshold")
        if threshold > 1:
            raise ValueError(f"{label} visibility threshold must be in [0, 1]")
        if visible != nested_visible:
            raise ValueError(f"{label} frame and visibility visible support mismatch")
        if known == 0:
            raise ValueError(f"{label} known support must be positive")
        if visible == 0:
            raise ValueError(f"{label} frames with numeric EPE require positive visible support")
        if visible > known:
            raise ValueError(f"{label} visible support exceeds known support")
        if tp + fn != visible:
            raise ValueError(f"{label} confusion counts do not sum to visible support")
        if fp > known - visible:
            raise ValueError(f"{label} false positives exceed known background support")
        indexed[frame_id] = {
            "id": frame_id,
            "object_id": object_id,
            "epe_px": epe,
            "visible_pixels": visible,
            "known_pixels": known,
            "threshold": threshold,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": known - visible - fp,
        }
    return indexed


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _pooled_visibility(frames):
    pooled = {key: sum(frame[key] for frame in frames)
              for key in ("true_positive", "false_positive", "false_negative", "true_negative")}
    pooled["precision"] = _safe_ratio(pooled["true_positive"],
                                      pooled["true_positive"] + pooled["false_positive"])
    pooled["recall"] = _safe_ratio(pooled["true_positive"],
                                   pooled["true_positive"] + pooled["false_negative"])
    return pooled


def _mean(values):
    return sum(values) / len(values) if values else None


def _object_macro_epe(frames):
    by_object = {}
    for frame in frames:
        by_object.setdefault(frame["object_id"], []).append(frame["epe_px"])
    return _mean([_mean(values) for values in by_object.values()])


def _prepare_paired_epoch(validation, epoch):
    adapted_by_id = _paired_frame_map(validation["frames"], "adapted")
    frozen_by_id = _paired_frame_map(validation["frozen_frames"], "frozen")
    available = epoch["validation_availability"]["available"]
    if len(adapted_by_id) != available or len(frozen_by_id) != available:
        raise ValueError("paired frame count must match validation availability.available")
    adapted_ids, frozen_ids = set(adapted_by_id), set(frozen_by_id)
    if adapted_ids != frozen_ids:
        raise ValueError("paired frame evidence has missing counterparts")

    thresholds = set()
    pairs = []
    for frame_id in sorted(adapted_ids):
        adapted, frozen = adapted_by_id[frame_id], frozen_by_id[frame_id]
        if adapted["object_id"] != frozen["object_id"]:
            raise ValueError(f"paired frame object mismatch for {frame_id}")
        if (adapted["visible_pixels"], adapted["known_pixels"]) != (
                frozen["visible_pixels"], frozen["known_pixels"]):
            raise ValueError(f"paired frame visible/known support mismatch for {frame_id}")
        thresholds.update((adapted["threshold"], frozen["threshold"]))
        pairs.append({"id": frame_id, "object_id": adapted["object_id"],
                      "adapted": adapted, "frozen": frozen,
                      "delta_px": adapted["epe_px"] - frozen["epe_px"]})
    if len(thresholds) != 1:
        raise ValueError("paired visibility metrics require one shared fixed threshold")
    if not pairs:
        raise ValueError("paired frame evidence must contain at least one matched frame")

    actual_adapted_macro = _object_macro_epe(adapted_by_id.values())
    actual_frozen_macro = _object_macro_epe(frozen_by_id.values())
    if not math.isclose(actual_adapted_macro,
                        epoch["validation_adapted_object_macro_epe_px"],
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("adapted history object-macro EPE disagrees with paired frame evidence")
    if not math.isclose(actual_frozen_macro,
                        epoch["validation_frozen_object_macro_epe_px"],
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("frozen history object-macro EPE disagrees with paired frame evidence")
    return pairs, next(iter(thresholds))


def _paired_frame_diagnostics(history, epochs):
    evidence_rows = []
    prepared_epochs = []
    for index, row in enumerate(history):
        validation = row.get("validation", {})
        if not isinstance(validation, dict):
            raise ValueError("validation must be an object")
        has_adapted = "frames" in validation
        has_frozen = "frozen_frames" in validation
        if has_adapted != has_frozen:
            raise ValueError("paired frame evidence requires both validation.frames and validation.frozen_frames")
        evidence_rows.append(has_adapted)
        prepared_epochs.append(_prepare_paired_epoch(validation, epochs[index]) if has_adapted else None)

    if not any(evidence_rows):
        return {"status": "unavailable", "reason": "no validation frame lists were supplied"}

    best_index = min(range(len(epochs)),
                     key=lambda index: epochs[index]["validation_adapted_object_macro_epe_px"])
    if prepared_epochs[best_index] is None:
        raise ValueError("selected minimum-EPE history epoch lacks paired frame evidence")
    pairs, selected_threshold = prepared_epochs[best_index]

    adapted_epe = [pair["adapted"]["epe_px"] for pair in pairs]
    frozen_epe = [pair["frozen"]["epe_px"] for pair in pairs]
    deltas = [pair["delta_px"] for pair in pairs]
    objects = {}
    for pair in pairs:
        objects.setdefault(pair["object_id"], []).append(pair)
    object_gains = []
    for object_id in sorted(objects):
        object_pairs = objects[object_id]
        frozen_mean = _mean([pair["frozen"]["epe_px"] for pair in object_pairs])
        adapted_mean = _mean([pair["adapted"]["epe_px"] for pair in object_pairs])
        object_gains.append({"object_id": object_id, "frames": len(object_pairs),
                             "baseline_frozen_mean_epe_px": frozen_mean,
                             "adapted_mean_epe_px": adapted_mean,
                             "gain_px": frozen_mean - adapted_mean})

    bin_specs = (("<0.1", lambda fraction: fraction < .1),
                 ("0.1<=x<0.9", lambda fraction: .1 <= fraction < .9),
                 (">=0.9", lambda fraction: fraction >= .9))
    bins = {}
    zero_known_support = 0
    for label, _ in bin_specs:
        bins[label] = []
    for pair in pairs:
        known = pair["frozen"]["known_pixels"]
        if known == 0:
            zero_known_support += 1
            continue
        fraction = pair["frozen"]["visible_pixels"] / known
        for label, belongs in bin_specs:
            if belongs(fraction):
                bins[label].append(pair)
                break
    bin_summary = {}
    for label, _ in bin_specs:
        bucket = bins[label]
        bucket_frozen = _mean([pair["frozen"]["epe_px"] for pair in bucket])
        bucket_adapted = _mean([pair["adapted"]["epe_px"] for pair in bucket])
        bin_summary[label] = {"frames": len(bucket),
                              "baseline_frozen_mean_epe_px": bucket_frozen,
                              "adapted_mean_epe_px": bucket_adapted,
                              "adapted_minus_baseline_mean_epe_px": (
                                  bucket_adapted - bucket_frozen
                                  if bucket else None)}

    visibility = {
        "threshold": selected_threshold,
        "threshold_policy": "shared persisted threshold; no cutoff tuning",
        "baseline_frozen": _pooled_visibility([pair["frozen"] for pair in pairs]),
        "adapted": _pooled_visibility([pair["adapted"] for pair in pairs]),
    }
    return {
        "status": "available",
        "evidence_label": "exploratory_paired_validation_diagnostics",
        "selected_epoch": epochs[best_index]["epoch"],
        "selection_method": (
            "minimum validation adapted object_macro_epe_px in supplied history; "
            "first occurrence on ties; this does not prove checkpoint identity"),
        "pairing": {"by": "id", "matched_frames": len(pairs),
                    "independent_objects": len(objects)},
        "all_frame_epe_px": {
            "baseline_frozen_mean": _mean(frozen_epe),
            "adapted_mean": _mean(adapted_epe),
            "adapted_minus_baseline_mean": _mean(deltas),
            "improved": sum(delta < 0 for delta in deltas),
            "tied": sum(delta == 0 for delta in deltas),
            "regressed": sum(delta > 0 for delta in deltas),
        },
        "object_macro_paired_gains": {
            "gain_definition": "baseline frozen mean EPE minus adapted mean EPE; positive favors adapted",
            "mean_gain_px": _mean([row["gain_px"] for row in object_gains]),
            "per_object": object_gains,
        },
        "pooled_visibility": visibility,
        "visible_fraction_bins": {
            "definition": "ground-truth visible_pixels / known_pixels; descriptive frame bins",
            "bins": bin_summary,
            "zero_known_support_frames": zero_known_support,
        },
        "per_frame": [{
            "id": pair["id"], "object_id": pair["object_id"],
            "epe_px": {"baseline_frozen": pair["frozen"]["epe_px"],
                       "adapted": pair["adapted"]["epe_px"],
                       "adapted_minus_baseline": pair["delta_px"]},
            "support": {"visible_pixels": pair["frozen"]["visible_pixels"],
                        "known_pixels": pair["frozen"]["known_pixels"]},
            "visibility": {
                "baseline_frozen": {key: pair["frozen"][key] for key in (
                    "threshold", "true_positive", "false_positive", "false_negative", "true_negative")},
                "adapted": {key: pair["adapted"][key] for key in (
                    "threshold", "true_positive", "false_positive", "false_negative", "true_negative")},
            },
        } for pair in pairs],
        "limitations": [
            "exploratory paired validation summaries; they do not establish causal effects or generalization",
            "one independent object is insufficient for across-object confidence intervals",
            "selected history minimum is not proof of the checkpoint used by a later inference",
        ],
    }


def summarize(history, status, *, inventory_plan=None, matched_metrics=None):
    _finite_tree(history)
    _finite_tree(status)
    if not isinstance(history, list) or not isinstance(status, dict):
        raise ValueError("existing history list and status object required")
    epochs = []
    previous_steps = -1
    previous_epoch = -1
    for row in history:
        epoch = _number(row.get("epoch"), "epoch", integer=True)
        steps = _number(row.get("steps"), "steps", integer=True)
        if steps <= previous_steps or epoch <= previous_epoch:
            raise ValueError("history epochs/steps must increase")
        previous_steps, previous_epoch = steps, epoch
        validation = row.get("validation", {})
        coverage = validation.get("availability", {})
        available = _number(coverage.get("available"), "available", integer=True)
        total = _number(coverage.get("total"), "total", integer=True)
        if available > total:
            raise ValueError("availability exceeds total")
        epochs.append({"epoch": epoch, "steps": steps,
            "train_composite_loss": _number(row.get("train_loss"), "train loss"),
            "validation_adapted_object_macro_epe_px": _number(validation.get("adapted", {}).get("object_macro_epe_px"), "validation EPE"),
            "validation_frozen_object_macro_epe_px": _number(validation.get("frozen", {}).get("object_macro_epe_px"), "frozen EPE"),
            "validation_availability": {"available": available, "total": total, "failed": total - available}})
    steps = _number(status.get("steps", 0), "status steps", integer=True)
    if steps < previous_steps:
        raise ValueError("status predates history")
    gaps = compare_matched_metrics(matched_metrics)
    paired_frames = _paired_frame_diagnostics(history, epochs)
    diversity = None
    if inventory_plan is not None:
        from .quality_adaptation_scale import validate_plan
        validate_plan(inventory_plan)
        diversity = inventory_plan["counts"]
    return {"schema": "adaptation_learning_curve_v1", "status": status.get("status", "unknown"),
        "reason": status.get("reason"), "steps": steps, "completed_epochs": len(epochs),
        "epochs": epochs, "diversity": diversity, "matched_metric_gaps": gaps,
        "paired_frame_diagnostics": paired_frames,
        "diagnostic_evidence": "matched_metrics_available_no_automatic_fit_diagnosis" if gaps else "insufficient_diagnostic_evidence",
        "loss_units": "training_composite_objective_only", "validation_units": "crop_pixels",
        "fit_diagnosis": None, "quality_claim": False,
        "limitations": ["train composite loss and validation EPE are different metrics; no gap computed",
                        "sample count alone cannot establish underfitting or generalization",
                        "independent held-out accuracy and runtime inference checks still required"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "status", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inventory-plan", type=Path)
    parser.add_argument("--matched-metrics", type=Path)
    args = parser.parse_args(argv)
    report = summarize(read_json(args.history), read_json(args.status),
        inventory_plan=read_json(args.inventory_plan) if args.inventory_plan else None,
        matched_metrics=read_json(args.matched_metrics) if args.matched_metrics else None)
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
