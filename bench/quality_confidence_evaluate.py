"""Evaluate the research-only legacy confidence veto on train/validation.

The evaluator never opens the test feature or label shard. Validation results
are development-fitting evidence from one correlated object, not an independent
accuracy or generalization estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .quality_confidence_rejection import UNSUPPORTED_CLAIMS, _read_label_policy
from .quality_pose_ranker import LegacyPoseRanker, make_ranker
from .quality_ranker_data import _cache_manifest, _label_key, audit_cache, load_cache_split
from .quality_ranker_evaluate import _rank_candidates


def _is_unprocessed(frame):
    """Recognize cache placeholders, including the bounded cloud extraction marker."""
    return frame.get("failure_reason") in (
        "not_processed", "not_processed_within_approved_compute_budget")


def _interior_thresholds(values):
    values = np.unique(np.asarray(values, dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Threshold curves require finite probabilities in [0,1]")
    edges = np.unique(np.concatenate(([0.], values, [1.])))
    return sorted(set([0., 1.] + ((edges[:-1] + edges[1:]) * .5).tolist()))


def _ratio(numerator, denominator, reason, null_reasons, name):
    if denominator == 0:
        null_reasons[name] = reason
        return None
    return numerator / denominator


def selective_metrics(targets, retained):
    y = np.asarray(targets)
    keep = np.asarray(retained)
    if y.ndim != 1 or keep.shape != y.shape or y.dtype.kind not in "biuf" or keep.dtype.kind != "b":
        raise ValueError("Selective metrics need matched binary targets and decisions")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Selective metrics targets must be binary")
    good, bad = y == 1, y == 0
    gr = int(np.count_nonzero(good & keep)); gj = int(np.count_nonzero(good & ~keep))
    ba = int(np.count_nonzero(bad & keep)); bj = int(np.count_nonzero(bad & ~keep))
    reasons = {}
    retention = _ratio(gr, gr + gj, "no_labeled_good_legacy_choices", reasons, "retention")
    bad_recall = _ratio(bj, bj + ba, "no_labeled_bad_legacy_choices", reasons, "bad_recall")
    reject_precision = _ratio(bj, bj + gj, "no_rejected_labeled_legacy_choices", reasons, "reject_precision")
    selective_error = _ratio(ba, gr + ba, "no_retained_labeled_legacy_choices", reasons, "selective_error")
    coverage = _ratio(gr + ba, len(y), "no_labeled_legacy_choices", reasons, "coverage")
    balanced = (None if retention is None or bad_recall is None
                else .5 * (retention + bad_recall))
    if balanced is None:
        reasons["balanced_accuracy"] = "retention_or_bad_recall_unavailable"
    return {
        "GR_good_retained": gr, "GJ_good_rejected": gj,
        "BA_bad_accepted": ba, "BJ_bad_rejected": bj,
        "retention": retention, "bad_recall": bad_recall,
        "reject_precision": reject_precision, "selective_error": selective_error,
        "coverage": coverage, "balanced_accuracy": balanced,
        "null_reasons": reasons,
    }


def bad_class_average_precision(probabilities, targets):
    """Average precision for bad risk (1-p), processing score ties jointly."""
    p, y = np.asarray(probabilities, dtype=float), np.asarray(targets)
    if p.shape != y.shape or p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Average precision needs matched finite probabilities")
    if np.any((p < 0.) | (p > 1.)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Average precision inputs are outside their domains")
    positive = y == 0
    total = int(np.count_nonzero(positive))
    if not total:
        return {"value": None, "null_reason": "no_bad_labels", "positive_class": "bad"}
    risk = 1. - p
    tp = seen = 0
    area = 0.
    for score in sorted(set(risk.tolist()), reverse=True):
        tied = risk == score
        group_positive = int(np.count_nonzero(positive & tied))
        group_count = int(np.count_nonzero(tied))
        tp += group_positive; seen += group_count
        area += (group_positive / total) * (tp / seen)
    return {"value": float(area), "null_reason": None,
            "positive_class": "bad", "risk_score": "1-selected_probability",
            "tie_handling": "joint_score_groups"}


def probability_metrics(probabilities, targets):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(targets, dtype=float)
    if p.shape != y.shape or p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Probability metrics need matched finite vectors")
    if np.any((p < 0.) | (p > 1.)) or not np.isin(y, [0., 1.]).all():
        raise ValueError("Probability metric values are outside their domains")
    eps = np.finfo(np.float64).eps
    clipped = np.clip(p, eps, 1. - eps)
    return {
        "bce": float(np.mean(-(y * np.log(clipped) + (1. - y) * np.log(1. - clipped)))),
        "brier": float(np.mean((p - y) ** 2)),
        "bad_class_average_precision": bad_class_average_precision(p, y),
    }


def threshold_curve(probabilities, targets):
    p = np.asarray(probabilities, dtype=float)
    return [{"threshold": float(threshold),
             **selective_metrics(targets, p >= threshold)}
            for threshold in _interior_thresholds(p)]


def _source_hashes():
    names = (
        "quality_ranker_train.py", "quality_ranker_features.py", "quality_pose_ranker.py",
        "quality_ranker_data.py", "quality_contract.py", "quality_gotrack.py",
        "quality_confidence_rejection.py", "quality_confidence_evaluate.py",
        "quality_pipeline.py", "quality_runner.py", "quality_offline.py",
    )
    result = {}
    for name in names:
        result[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return result


def _method_metrics(rows):
    if not rows:
        return {"metrics": selective_metrics(np.array([], dtype=int), np.array([], dtype=bool))}
    targets = np.asarray([row["target"] for row in rows], dtype=int)
    retained = np.asarray([row["retained"] for row in rows], dtype=bool)
    result = {"metrics": selective_metrics(targets, retained)}
    if all(row.get("probability") is not None for row in rows):
        probabilities = np.asarray([row["probability"] for row in rows], dtype=float)
        result["probability"] = probability_metrics(probabilities, targets)
        result["threshold_curve"] = threshold_curve(probabilities, targets)
    objects = sorted({row["object_id"] for row in rows})
    result["object_cluster_uncertainty"] = {
        "objects": len(objects), "ci95": None,
        "status": ("insufficient_independent_objects" if len(objects) < 2
                   else "not_computed_for_development_fitting_report"),
    }
    return result


def evaluate_confidence_cache(cache_dir, *, split, checkpoint):
    if split not in ("train", "validation"):
        raise ValueError("Confidence evaluation may read only train or validation")
    cache_dir = Path(cache_dir)
    audit = audit_cache(cache_dir, training_only=True)
    if audit.get("approved") is not True or audit.get("audit_scope") != "train_validation_only":
        raise ValueError("Confidence evaluation requires an approved training-only audit")
    ranker = make_ranker("confidence-research", checkpoint)
    artifact = ranker.to_checkpoint()
    artifact_identity = artifact["provenance"]["identity"]
    frozen_policy, frozen_policy_sha256 = _read_label_policy(cache_dir)
    if artifact_identity["data"] != audit["identity"]:
        raise ValueError("Confidence artifact data identity differs from the audited cache")
    if (artifact_identity["label_policy_sha256"] != frozen_policy_sha256 or
            artifact["provenance"]["label_policy"]["sha256"] != frozen_policy_sha256 or
            artifact["provenance"]["label_policy"]["frozen"] != frozen_policy):
        raise ValueError("Confidence artifact label policy differs from the audited cache")
    frames, labels = load_cache_split(cache_dir, split)
    label_index = {_label_key(row): row for row in labels}
    legacy_ranker = LegacyPoseRanker()
    categories = {name: 0 for name in
                  ("unprocessed", "empty", "hard_gate", "legacy_reject",
                   "unlabeled", "learned_veto", "retained")}
    learned_rows, legacy_rows, deterministic_rows = [], [], []
    diagnostics = []
    from .quality_pose_ranker import make_ranker as ranker_factory
    deterministic = ranker_factory("deterministic")
    for frame in frames:
        candidates = _rank_candidates(frame)
        legacy = legacy_ranker.rank(frame.get("context", {}), candidates)
        raw_candidates = frame.get("candidates", [])
        if _is_unprocessed(frame):
            category = "unprocessed"
        elif not raw_candidates:
            category = "empty"
        elif not candidates:
            category = "hard_gate"
        elif legacy.selected_id is None:
            category = "legacy_reject"
        else:
            category = None
        probability = None
        decision = legacy
        label = None
        if legacy.selected_id is not None:
            selected = next(row for row in candidates if row.candidate_id == legacy.selected_id)
            probability = ranker.probability(selected.features)
            decision = ranker.rank_with_selected_probability(
                frame.get("context", {}), candidates, probability)
            key = (frame["object_id"], frame["sequence_id"], frame["frame_id"], legacy.selected_id)
            label = label_index.get(key)
            if label is None or label.get("quality_target") is None:
                category = category or "unlabeled"
            else:
                target = int(label["quality_target"])
                retained = decision.selected_id is not None
                category = category or ("retained" if retained else "learned_veto")
                base = {"target": target, "object_id": frame["object_id"]}
                learned_rows.append({**base, "retained": retained, "probability": probability})
                legacy_rows.append({**base, "retained": True})
        categories[category] += 1

        deterministic_decision = deterministic.rank(frame.get("context", {}), candidates)
        if deterministic_decision.selected_id is not None:
            key = (frame["object_id"], frame["sequence_id"], frame["frame_id"],
                   deterministic_decision.selected_id)
            deterministic_label = label_index.get(key)
            if deterministic_label is not None and deterministic_label.get("quality_target") is not None:
                deterministic_rows.append({"target": int(deterministic_label["quality_target"]),
                                           "retained": True, "object_id": frame["object_id"]})
        diagnostics.append({
            "object_id": frame["object_id"], "sequence_id": frame["sequence_id"],
            "frame_id": frame["frame_id"], "category": category,
            "legacy_ordered_ids": list(legacy.ordered_ids),
            "legacy_scores": dict(legacy.scores),
            "legacy_selected_id": legacy.selected_id,
            "selected_probability": probability,
            "selected_id": decision.selected_id,
            "rejection_reason": decision.rejection_reason,
            "label_present": label is not None,
        })

    targets = [row["target"] for row in legacy_rows]
    manifest, _ = _cache_manifest(cache_dir)
    train_frames, train_labels = load_cache_split(cache_dir, "train")
    validation_frames, _ = load_cache_split(cache_dir, "validation")
    train_index = {_label_key(row): row for row in train_labels}
    selected_train_targets = []
    for frame in train_frames:
        candidates = _rank_candidates(frame)
        selected = legacy_ranker.rank(frame.get("context", {}), candidates).selected_id
        label = (train_index.get((frame["object_id"], frame["sequence_id"], frame["frame_id"], selected))
                 if selected is not None else None)
        if label is not None and label.get("quality_target") is not None:
            selected_train_targets.append(int(label["quality_target"]))
    constant = (None if not selected_train_targets else
                sum(selected_train_targets) / len(selected_train_targets))
    constant_rows = [{"target": target, "retained": bool(constant is not None and constant >= .5),
                      "probability": constant, "object_id": row["object_id"]}
                     for target, row in zip(targets, legacy_rows)]
    always_reject = [{**row, "retained": False} for row in legacy_rows]
    inventory_frames = train_frames + validation_frames
    inventory_count = len(inventory_frames)
    inventory_unprocessed = sum(
        _is_unprocessed(row) for row in inventory_frames)
    actual_traced_frames = inventory_count - inventory_unprocessed
    candidate_containing_frames = sum(
        not _is_unprocessed(row) and bool(row.get("candidates"))
        for row in inventory_frames)
    selected_inventory_frames = len(frames)
    selected_unprocessed_frames = categories["unprocessed"]
    selected_traced_frames = selected_inventory_frames - selected_unprocessed_frames
    selected_candidate_containing_frames = sum(
        not _is_unprocessed(frame) and bool(frame.get("candidates"))
        for frame in frames)
    if (inventory_count != manifest["frame_count"] or
            actual_traced_frames + inventory_unprocessed != manifest["frame_count"] or
            selected_traced_frames + selected_unprocessed_frames != selected_inventory_frames):
        raise ValueError("Training-only cache rows do not match the frozen frame inventory")
    return {
        "schema_version": 1, "kind": "confidence-rejection-research-evaluation-v1",
        "artifact_purpose": "confidence_rejection_research_v1", "policy": "legacy_veto_v1",
        "research_only": True, "split": split,
        "development_fitting": True, "test_read": False,
        "calibration_and_checkpoint_selection_used_validation": True,
        "frame_independence_ci_available": False,
        "generalization_ci_available": False,
        "unsupported_claims": list(UNSUPPORTED_CLAIMS),
        "inventory": {
            "full_frozen_frames": manifest["frame_count"],
            "actual_traced_frames": actual_traced_frames,
            "candidate_containing_trace_frames": candidate_containing_frames,
            "unprocessed_inventory_frames": inventory_unprocessed,
            "selected_split_inventory_frames": selected_inventory_frames,
            "selected_split_traced_frames": selected_traced_frames,
            "selected_split_candidate_containing_trace_frames": selected_candidate_containing_frames,
            "selected_split_unprocessed_frames": selected_unprocessed_frames,
            "interpretation": "Full frozen inventory is separate from traced pose availability; cached selection is not render availability or closed-loop tracking.",
        },
        "failure_categories": categories,
        "methods": {
            "confidence_research": _method_metrics(learned_rows),
            "always_retain_legacy": _method_metrics(legacy_rows),
            "always_reject_legacy": _method_metrics(always_reject),
            "constant_train_selected_good_fraction": {
                "constant_probability": constant, "fixed_threshold": .5,
                **_method_metrics(constant_rows),
            },
            "deterministic_same_raw_eligible_candidates": _method_metrics(deterministic_rows),
        },
        "research_diagnostics": diagnostics,
        "cache_identity": audit["identity"],
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "source_hashes": _source_hashes(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_confidence_cache(args.cache_dir, split=args.split,
                                       checkpoint=args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace evaluation report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "evaluated", "split": args.split,
                      "frames": report["inventory"]["selected_split_traced_frames"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()

