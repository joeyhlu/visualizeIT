"""Research-only data joins and diagnostics for legacy confidence rejection.

This module never imports the trainer or PyTorch.  It loads only the audited
train and validation shards and computes the legacy choice before looking up a
reference label.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .quality_pose_ranker import LegacyPoseRanker
from .quality_ranker_data import (
    _cache_manifest, _label_key, _safe_cache_file, audit_cache,
    load_cache_split, validate_split_manifest,
)
from .quality_ranker_evaluate import _rank_candidates


ARTIFACT_PURPOSE = "confidence_rejection_research_v1"
POLICY = "legacy_veto_v1"
QUALITY_TARGET_DEFINITION = "frozen_available_dimension_quality"
ANGULAR_OBSERVABILITY_INTERPRETATION = (
    "ambiguous_frozen_before_inference_rotation_excluded_unchanged"
)
CALIBRATION_SCOPE = "legacy_selected_labeled_eligible"
CALIBRATION_RULE = "fixed_grid_bce_then_balanced_accuracy_conservative_v1"
CHECKPOINT_SELECTION = "legacy_selected_validation_uncalibrated_bce"
UNSUPPORTED_CLAIMS = [
    "candidate_ordering", "independent_test_accuracy", "object_generalization",
    "physical_attachment_accuracy",
]


def _canonical_hash(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _read_label_policy(cache_dir: Path) -> tuple[dict, str]:
    manifest, _ = _cache_manifest(cache_dir)
    split_path = _safe_cache_file(cache_dir, manifest["files"]["split_manifest"])
    split_document = json.loads(split_path.read_text(encoding="utf-8"))
    split_document = validate_split_manifest(split_document)
    policy = split_document["label_policy"]
    return policy, _canonical_hash(policy)


def _prepare_split(frames: list[dict], labels: list[dict], split: str) -> dict:
    """Choose with legacy first, then join labels by the complete candidate key."""
    label_index = {_label_key(row): row for row in labels}
    groups, selected_all, selected_labeled = [], [], []
    eligible_targets = []
    legacy = LegacyPoseRanker()
    for frame in frames:
        provenance_kind = frame.get("provenance_kind", frame.get("context", {}).get("provenance_kind"))
        if provenance_kind == "fixture":
            raise ValueError("Confidence research rejects fixture-provenance training data")
        if provenance_kind not in ("real", "synthetic"):
            raise ValueError("Confidence research requires explicit real or synthetic provenance")
        ranked_candidates = _rank_candidates(frame)
        decision = legacy.rank(frame.get("context", {}), ranked_candidates)
        by_id = {row.candidate_id: row for row in ranked_candidates}
        frame_key = (frame["object_id"], frame["sequence_id"], frame["frame_id"])

        # Optimization retains every eligible labeled candidate, including
        # all-bad frames.  Cross-frame pairs are impossible because groups stay
        # intact for the existing batch_loss implementation.
        group_candidates = []
        for candidate in ranked_candidates:
            label = label_index.get((*frame_key, candidate.candidate_id))
            if label is None or label.get("quality_target") is None:
                continue
            target = label["quality_target"]
            if type(target) is not int or target not in (0, 1):
                raise ValueError("Confidence research needs frozen binary quality targets")
            eligible_targets.append(target)
            group_candidates.append({
                "candidate_id": candidate.candidate_id,
                "features": list(candidate.features),
                "quality_target": target,
                "eligible": True,
                "pose": np.asarray(candidate.pose, dtype=float).tolist(),
            })
        groups.append({
            "object_id": frame["object_id"], "sequence_id": frame["sequence_id"],
            "frame_id": frame["frame_id"], "source_frame_id": frame["source_frame_id"],
            "context": frame.get("context", {}), "candidates": group_candidates,
        })

        chosen = by_id.get(decision.selected_id)
        label = (label_index.get((*frame_key, decision.selected_id))
                 if decision.selected_id is not None else None)
        row = {
            "split": split, "frame": frame, "candidates": ranked_candidates,
            "legacy_decision": decision, "selected_candidate": chosen,
            "label": label,
        }
        selected_all.append(row)
        if chosen is not None and label is not None and label.get("quality_target") is not None:
            target = label["quality_target"]
            if type(target) is not int or target not in (0, 1):
                raise ValueError("Confidence research selected labels must be binary")
            selected_labeled.append({**row, "quality_target": target,
                                     "features": list(chosen.features)})

    selected_targets = [row["quality_target"] for row in selected_labeled]
    counts = {
        "eligible_labeled": len(eligible_targets),
        "eligible_good": eligible_targets.count(1),
        "eligible_bad": eligible_targets.count(0),
        "legacy_selected_labeled": len(selected_targets),
        "legacy_selected_good": selected_targets.count(1),
        "legacy_selected_bad": selected_targets.count(0),
    }
    return {"groups": groups, "selected_all": selected_all,
            "selected_labeled": selected_labeled, "counts": counts}


def load_research_dataset(manifest: Path) -> dict:
    """Load train/validation only through the raw cache boundary."""
    manifest = Path(manifest)
    if manifest.name != "cache_manifest.json":
        raise ValueError("--manifest must point to the frozen cache_manifest.json")
    cache_dir = manifest.parent
    audit = audit_cache(cache_dir, training_only=True)
    if audit.get("approved") is not True or audit.get("audit_scope") != "train_validation_only":
        raise ValueError("Confidence research requires an approved training-only cache audit")
    from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
    if (audit["identity"].get("feature_names") != list(FEATURE_NAMES) or
            audit["identity"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION):
        raise ValueError("Frozen cache feature schema differs from confidence inference")
    policy, policy_hash = _read_label_policy(cache_dir)
    prepared = {}
    for split in ("train", "validation"):
        frames, labels = load_cache_split(cache_dir, split)
        # The test shard is neither requested nor delegated to another loader.
        prepared[split] = _prepare_split(frames, labels, split)
    train_counts = prepared["train"]["counts"]
    validation_counts = prepared["validation"]["counts"]
    if not train_counts["eligible_good"] or not train_counts["eligible_bad"]:
        raise ValueError("Confidence research needs both classes among eligible labeled training candidates")
    if not train_counts["legacy_selected_good"] or not train_counts["legacy_selected_bad"]:
        raise ValueError("Confidence baseline needs both classes among legacy-selected labeled train rows")
    if not validation_counts["legacy_selected_good"] or not validation_counts["legacy_selected_bad"]:
        raise ValueError("Confidence calibration needs both classes among legacy-selected labeled validation rows")
    return {
        "train": prepared["train"]["groups"],
        "validation": prepared["validation"]["groups"],
        "selected_train": prepared["train"]["selected_labeled"],
        "selected_validation": prepared["validation"]["selected_labeled"],
        "selected_all_validation": prepared["validation"]["selected_all"],
        "identity": audit["identity"], "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_policy": policy, "label_policy_sha256": policy_hash,
        "supervision_counts": {"train": train_counts, "validation": validation_counts},
        "ranking_signal_available": audit.get("ranking_signal_available"),
    }


def research_eda(groups: list[dict], feature_names: list[str], *,
                 train_reference: np.ndarray | None = None) -> dict:
    """Descriptive candidate-level EDA; it makes no accuracy claim."""
    rows, targets, objects, sequences, frame_keys, candidate_counts = [], [], [], [], [], []
    for group in groups:
        eligible = [row for row in group["candidates"] if row.get("eligible", True)]
        candidate_counts.append(len(eligible))
        for row in eligible:
            vector = row["features"]
            rows.append([vector[name] for name in feature_names]
                        if isinstance(vector, dict) else list(vector))
            targets.append(row["quality_target"])
            objects.append(group["object_id"]); sequences.append(group["sequence_id"])
            frame_keys.append((group["object_id"], group["sequence_id"], group["frame_id"]))
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(feature_names) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Confidence EDA requires finite frozen-schema feature rows")
    reference = x if train_reference is None else np.asarray(train_reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != x.shape[1] or not np.isfinite(reference).all():
        raise ValueError("Confidence EDA train reference is malformed")
    q1, q3 = np.quantile(reference, [.25, .75], axis=0)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = np.sum((x < low) | (x > high), axis=0)
    centered = x - x.mean(axis=0)
    denom = np.sqrt(np.sum(centered ** 2, axis=0))
    correlations = (centered.T @ centered) / np.maximum(denom[:, None] * denom[None, :], 1e-12)
    return {
        "rows": int(len(x)), "feature_names": list(feature_names),
        "feature_quantiles": np.quantile(x, [0., .01, .25, .5, .75, .99, 1.], axis=0).tolist(),
        "missing_flags": {name: int(np.count_nonzero(x[:, index] == 0.))
                          for index, name in enumerate(feature_names) if name.startswith("has_")},
        "constant_features": [feature_names[index] for index in range(x.shape[1])
                              if np.ptp(x[:, index]) == 0.],
        "train_iqr_outlier_counts": outliers.astype(int).tolist(),
        "feature_correlations": correlations.tolist(),
        "class_counts": {"good": int(np.count_nonzero(y == 1)),
                         "bad": int(np.count_nonzero(y == 0))},
        "candidate_count_distribution": dict(sorted(Counter(candidate_counts).items())),
        "frame_count": len(set(frame_keys)), "object_count": len(set(objects)),
        "sequence_count": len(set(sequences)),
        "interpretation": "Descriptive development fitting data; correlations are not independent-test accuracy evidence.",
    }

