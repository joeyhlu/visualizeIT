"""Compare legacy, deterministic, learned, and fixed-set oracle ranking.

Evaluation labels are read only here, after the label-free candidate cache has
been frozen.  Every method receives the same recorded candidate list.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .quality_ranker_data import (
    SPLITS, _label_key, _read_jsonl, _cache_manifest, _load_split_rows,
    _safe_cache_file, _json_bytes, _sha256, audit_cache,
)


def _percentile(values, percentile):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(np.percentile(values, percentile))


def _mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(np.mean(values))


def _numeric_leaves(value, prefix=""):
    """Yield bounded scalar numeric measurements from trace timing/resource maps."""
    if isinstance(value, dict):
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(child, name)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        yield prefix or "value", float(value)


def _trace_measurement_summary(frames):
    buckets = {"frame_timings_ms": {}, "candidate_appearance_summary_ms": {},
               "resource_memory": {}}
    for frame in frames:
        for path, number in _numeric_leaves(frame.get("timings_ms") or {}):
            buckets["frame_timings_ms"].setdefault(path, []).append(number)
        for candidate in frame.get("candidates", []):
            value = candidate.get("appearance_summary_ms")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                buckets["candidate_appearance_summary_ms"].setdefault("value", []).append(float(value))
        for path, number in _numeric_leaves(frame.get("resource_memory") or {}):
            buckets["resource_memory"].setdefault(path, []).append(number)
    result = {}
    for category, values_by_name in buckets.items():
        result[category] = {
            name: {"count": len(values), "mean": float(np.mean(values)),
                   "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                   "max": max(values)}
            for name, values in sorted(values_by_name.items()) if values
        }
    result["interpretation"] = (
        "Descriptive source-trace observations only; memory/timing samples are not independent clusters "
        "and do not measure this evaluator's ranking latency.")
    return result


def _cluster_interval(rows, value_key, cluster_key, *, seed=42, draws=2000):
    """Bootstrap independent clusters, never individual adjacent frames."""
    clusters = {}
    for row in rows:
        value = row.get(value_key)
        cluster = row.get(cluster_key)
        if cluster is None or value is None or not math.isfinite(float(value)):
            continue
        clusters.setdefault(cluster, []).append(float(value))
    means = [float(np.mean(values)) for values in clusters.values() if values]
    if len(means) < 2:
        return {"clusters": len(means), "ci95": None, "status": "insufficient_independent_clusters"}
    rng = np.random.default_rng(seed)
    values = np.asarray(means, dtype=np.float64)
    draws_array = rng.integers(0, len(values), size=(draws, len(values)))
    bootstrap = values[draws_array].mean(axis=1)
    return {"clusters": len(values), "ci95": [float(np.percentile(bootstrap, 2.5)),
                                               float(np.percentile(bootstrap, 97.5))],
            "status": "cluster_bootstrap_percentile"}


def _pose_candidate(candidate):
    pose = candidate.get("pose")
    if pose is None:
        return None
    return SimpleNamespace(pose=np.asarray(pose, dtype=np.float64))


def _rank_candidates(frame):
    from .quality_pose_ranker import RankCandidate
    rows = []
    for ordinal, item in enumerate(frame["candidates"]):
        if not item.get("eligible") or item.get("pose") is None:
            continue
        rows.append(RankCandidate(
            candidate_id=item["candidate_id"],
            original_index=item.get("original_index") if type(item.get("original_index")) is int else ordinal,
            candidate=_pose_candidate(item),
            stats=item.get("validation_stats") or {},
            features=item["features"],
            ranker_features=None,
        ))
    return rows


def _method_choices(frames, learned_checkpoint=None, learned_ranker=None, *, learned_kind="learned"):
    from .quality_pose_ranker import make_ranker
    methods = {
        "deterministic": make_ranker("geometric"),
    }
    if learned_ranker is not None:
        methods["learned"] = learned_ranker
    elif learned_checkpoint is not None:
        methods["learned"] = make_ranker(learned_kind, learned_checkpoint)
    choices = {"legacy": {}}
    for frame in frames:
        key = (frame["object_id"], frame["sequence_id"], frame["frame_id"])
        choices["legacy"][key] = frame.get("legacy_choice_candidate_id")
        candidates = _rank_candidates(frame)
        for name, ranker in methods.items():
            decision = ranker.rank(frame.get("context", {}), candidates)
            choices.setdefault(name, {})[key] = decision.selected_id
    return choices


def _loss_for_oracle(label, policy):
    terms = []
    trans = label.get("translation_error_m")
    if trans is not None:
        terms.append(float(trans) / policy["max_translation_error_m"])
    rotation = label.get("rotation_error_deg")
    if rotation is not None and not label.get("observationally_ambiguous"):
        terms.append(float(rotation) / policy["max_rotation_error_deg"])
    surface = label.get("surface_error")
    if policy.get("max_surface_error_m") is not None and surface is not None:
        terms.append(float(surface["p95_m"]) / policy["max_surface_error_m"])
    projection = label.get("projection_error")
    if policy.get("max_projection_error_px") is not None and projection is not None:
        terms.append(float(projection["p95_px"]) / policy["max_projection_error_px"])
    return float(np.mean(terms)) if terms else float("inf")


def _oracle_choice(frame, label_index, policy, *, eligible_only):
    candidates = []
    for candidate in frame["candidates"]:
        if eligible_only and not candidate.get("eligible"):
            continue
        if candidate.get("pose") is None:
            continue
        key = (frame["object_id"], frame["sequence_id"], frame["frame_id"], candidate["candidate_id"])
        label = label_index.get(key)
        if label is None or label.get("quality_target") is None:
            continue
        loss = _loss_for_oracle(label, policy)
        candidates.append((int(label["quality_target"]), loss, candidate["candidate_id"]))
    if not candidates:
        return None
    # Quality success is primary; error among equally good/bad labels is a
    # deterministic secondary tie-break, then the stable candidate ID.
    return min(candidates, key=lambda row: (-row[0], row[1], row[2]))[2]


def _selection_rows(frames, labels, choices):
    label_index = {_label_key(row): row for row in labels}
    candidates_by_frame = {}
    output = []
    for frame in frames:
        frame_key = (frame["object_id"], frame["sequence_id"], frame["frame_id"])
        candidates = {row["candidate_id"]: row for row in frame["candidates"]}
        candidates_by_frame[frame_key] = candidates
        selection = choices.get(frame_key)
        candidate = candidates.get(selection) if selection is not None else None
        label = (label_index.get((*frame_key, selection)) if selection is not None else None)
        available = bool(candidate and candidate.get("eligible") and candidate.get("pose") is not None)
        if label is None:
            label = {}
        row = {
            "object_id": frame["object_id"], "sequence_id": frame["sequence_id"],
            "frame_id": frame["frame_id"], "source_frame_id": frame["source_frame_id"],
            "background_id": frame["background_id"], "provenance_kind": frame["provenance_kind"],
            "candidate_count": len(frame["candidates"]),
            "eligible_candidate_count": sum(bool(item.get("eligible")) for item in frame["candidates"]),
            "selected_candidate_id": selection,
            "selection_present": selection is not None,
            "available": available,
            "candidate_pose_present": bool(candidate and candidate.get("pose") is not None),
            "selected_ineligible": bool(candidate and not candidate.get("eligible")),
            "selected_pose_missing": bool(candidate and candidate.get("pose") is None),
            "selected_candidate_unknown": bool(selection is not None and candidate is None),
            "label_present": bool(label),
            "quality_target": label.get("quality_target"),
            "observationally_ambiguous": label.get("observationally_ambiguous"),
            "rotation_error_deg": label.get("rotation_error_deg"),
            "translation_error_m": label.get("translation_error_m"),
            "surface_error_p95_m": (label.get("surface_error") or {}).get("p95_m"),
            "projection_error_p95_px": (label.get("projection_error") or {}).get("p95_px"),
            "label_missing_reason": label.get("label_missing_reason"),
            "label_kind": label.get("label_kind"),
        }
        output.append(row)
    return output, candidates_by_frame


def _summarize_method(name, rows, *, frames, all_labels, policy, seed, draws):
    total = len(rows)
    available = sum(row["available"] for row in rows)
    selected = sum(row["selection_present"] for row in rows)
    # The before-gates oracle is a counterfactual over the same candidates.
    # Its ineligible pose may be scored for geometry but never contributes to
    # actual tracker availability or coverage.
    metric_available = ((lambda row: row["candidate_pose_present"])
                        if name == "oracle_before_gates" else (lambda row: row["available"]))
    oriented = [row for row in rows if metric_available(row) and row["rotation_error_deg"] is not None]
    translated = [row for row in rows if metric_available(row) and row["translation_error_m"] is not None]
    scored = [row for row in rows if metric_available(row) and row["quality_target"] is not None]
    eligible = [row for frame in frames for row in frame["candidates"] if row.get("eligible")]
    frame_count = max(1, total)
    ambiguous_frame_keys = {(label["object_id"], label["sequence_id"], label["frame_id"])
                            for label in all_labels if label.get("observationally_ambiguous") is True}
    method_rows = [dict(
        row,
        success_value=(None if row["quality_target"] is None or not metric_available(row)
                       else float(row["quality_target"])),
        translation_error_m=(row["translation_error_m"] if metric_available(row) else None),
        rotation_error_deg=(row["rotation_error_deg"] if metric_available(row) else None),
        surface_error_p95_m=(row["surface_error_p95_m"] if metric_available(row) else None),
        projection_error_p95_px=(row["projection_error_p95_px"] if metric_available(row) else None),
        availability_value=float(row["available"])) for row in rows]
    severe = [row for row in oriented if row["rotation_error_deg"] > policy["max_rotation_error_deg"]]
    surface_rows = [row for row in rows if metric_available(row) and row["surface_error_p95_m"] is not None]
    projection_rows = [row for row in rows if metric_available(row) and row["projection_error_p95_px"] is not None]
    object_uncertainty = {}
    sequence_uncertainty = {}
    for metric in ("availability_value", "success_value", "translation_error_m", "rotation_error_deg",
                   "surface_error_p95_m", "projection_error_p95_px"):
        object_uncertainty[metric] = _cluster_interval(method_rows, metric, "object_id", seed=seed, draws=draws)
        sequence_uncertainty[metric] = _cluster_interval(method_rows, metric, "sequence_id", seed=seed, draws=draws)
    return {
        "method": name,
        "frames": total,
        "selections": selected,
        "available_pose_count": available,
        "availability": None if not total else available / total,
        "coverage": None if not total else available / total,
        "failures": total - available,
        "metric_selection_scope": ("counterfactual_candidate_pose_including_ineligible"
                                    if name == "oracle_before_gates" else "selected_eligible_pose"),
        "failure_counts": {
            "no_selection": sum(not row["selection_present"] for row in rows),
            "selected_ineligible": sum(row["selected_ineligible"] for row in rows),
            "selected_pose_missing": sum(row["selected_pose_missing"] for row in rows),
            "selected_candidate_unknown": sum(row["selected_candidate_unknown"] for row in rows),
            "missing_reference_label": sum(row["available"] and not row["label_present"] for row in rows),
            "target_unavailable": sum(row["available"] and row["label_present"] and row["quality_target"] is None for row in rows),
        },
        "labeled_selections": len(scored),
        "quality_target_success_rate": _mean([row["quality_target"] for row in scored]),
        "translation_error_m": {"count": len(translated), "mean": _mean([row["translation_error_m"] for row in translated]),
                                "median": _percentile([row["translation_error_m"] for row in translated], 50),
                                "p95": _percentile([row["translation_error_m"] for row in translated], 95)},
        "rotation_error_deg": {"count": len(oriented), "median": _percentile([row["rotation_error_deg"] for row in oriented], 50),
                               "p95": _percentile([row["rotation_error_deg"] for row in oriented], 95),
                               "severe_count": len(severe),
                               "severe_fraction": None if not oriented else len(severe) / len(oriented),
                               "excluded_ambiguous_selected": sum(metric_available(row) and
                                                                   row["observationally_ambiguous"] is True
                                                                   for row in rows)},
        "surface_error_p95_m": {"count": len(surface_rows),
                                "median": _percentile([row["surface_error_p95_m"] for row in surface_rows], 50),
                                "p95": _percentile([row["surface_error_p95_m"] for row in surface_rows], 95)},
        "projection_error_p95_px": {"count": len(projection_rows),
                                     "median": _percentile([row["projection_error_p95_px"] for row in projection_rows], 50),
                                     "p95": _percentile([row["projection_error_p95_px"] for row in projection_rows], 95)},
        "all_bad_selected_frames": sum(row["available"] and row["quality_target"] == 0 for row in rows),
        "missing_label_frames": sum(not row["label_present"] for row in rows),
        "ambiguous_label_frames": len(ambiguous_frame_keys),
        "uncertainty": {"by_object": object_uncertainty, "by_sequence": sequence_uncertainty,
                        "unit": "cluster means; frames are not treated as independent"},
    }


def _evaluation_context(frames, labels):
    label_index = {_label_key(row): row for row in labels}
    totals = {"frames": len(frames), "candidates_before_gates": 0, "eligible_after_gates": 0,
              "rejected_candidates": 0, "empty_candidate_frames": 0,
              "all_gated_frames": 0, "legacy_choices": 0, "legacy_choice_ineligible": 0,
              "legacy_choice_unknown": 0,
              "missing_reference_frames": 0, "ambiguous_frames": 0,
              "good_candidate_frames_before_gates": 0, "good_candidate_frames_after_gates": 0,
              "oracle_before_gates_frames": 0, "oracle_after_gates_frames": 0}
    return label_index, totals


def evaluate_frames(frames, labels, label_policy, *, learned_checkpoint=None,
                    learned_kind="learned", seed=42, bootstrap_draws=2000):
    """Evaluate all input frames, including empty/rejected/null outcomes."""
    from .quality_ranker_data import _validate_label_policy
    policy = _validate_label_policy(label_policy)
    if not frames:
        raise ValueError("Evaluation requires the complete non-empty frozen frame set")
    label_index, totals = _evaluation_context(frames, labels)
    if learned_kind not in ("learned", "ordering-research"):
        raise ValueError("Unknown evaluation ranker kind")
    choices = _method_choices(frames, learned_checkpoint=learned_checkpoint,
                              learned_kind=learned_kind)
    choices["oracle_before_gates"] = {}
    choices["oracle_after_gates"] = {}
    for frame in frames:
        frame_key = (frame["object_id"], frame["sequence_id"], frame["frame_id"])
        candidates = frame["candidates"]
        eligible = [item for item in candidates if item.get("eligible") and item.get("pose") is not None]
        totals["candidates_before_gates"] += len(candidates)
        totals["eligible_after_gates"] += len(eligible)
        totals["rejected_candidates"] += sum(not item.get("eligible") for item in candidates)
        totals["empty_candidate_frames"] += not candidates
        totals["all_gated_frames"] += bool(candidates) and not eligible
        legacy_id = frame.get("legacy_choice_candidate_id")
        totals["legacy_choices"] += legacy_id is not None
        legacy_candidate = next((item for item in candidates if item["candidate_id"] == legacy_id), None)
        totals["legacy_choice_ineligible"] += bool(legacy_candidate and not legacy_candidate.get("eligible"))
        totals["legacy_choice_unknown"] += legacy_id is not None and legacy_candidate is None
        frame_labels = [label_index.get((frame["object_id"], frame["sequence_id"], frame["frame_id"],
                                         item["candidate_id"])) for item in candidates]
        totals["missing_reference_frames"] += not any(label is not None for label in frame_labels)
        totals["ambiguous_frames"] += any(label and label.get("observationally_ambiguous") is True for label in frame_labels)
        good = {item["candidate_id"] for item, label in zip(candidates, frame_labels)
                if label is not None and label.get("quality_target") == 1 and item.get("pose") is not None}
        good_eligible = {item["candidate_id"] for item in eligible
                         if (label_index.get((frame["object_id"], frame["sequence_id"], frame["frame_id"], item["candidate_id"])) or {}).get("quality_target") == 1}
        totals["good_candidate_frames_before_gates"] += bool(good)
        totals["good_candidate_frames_after_gates"] += bool(good_eligible)
        before = _oracle_choice(frame, label_index, policy, eligible_only=False)
        after = _oracle_choice(frame, label_index, policy, eligible_only=True)
        choices["oracle_before_gates"][frame_key] = before
        choices["oracle_after_gates"][frame_key] = after
        totals["oracle_before_gates_frames"] += before is not None
        totals["oracle_after_gates_frames"] += after is not None
    all_rows, raw = {}, {}
    for name, selections in choices.items():
        rows, _ = _selection_rows(frames, labels, selections)
        raw[name] = rows
        all_rows[name] = _summarize_method(name, rows, frames=frames, all_labels=labels,
                                           policy=policy, seed=seed, draws=bootstrap_draws)
    # Oracle records use the same statistics but are explicitly labelled as
    # upper-bound selection within the recorded candidate set, not search quality.
    return {
        "schema_version": 1,
        "kind": "quality-ranker-evaluation-v1",
        "learned_ranker_kind": learned_kind,
        "development_fitting": learned_kind == "ordering-research",
        "scope": "all frozen cache frames; methods share recorded candidate sets",
        "oracle_scope": "best labeled candidate inside the fixed recorded candidate set; not a search-completeness claim",
        "frame_count": len(frames),
        "provenance_counts_by_frame": {kind: sum(row["provenance_kind"] == kind for row in frames)
                                        for kind in ("real", "synthetic", "fixture")},
        "label_kinds": sorted({row.get("label_kind", "unspecified_reference") for row in labels}),
        "independent_human_labels_present": any(row.get("label_kind") == "independent_image_annotation" for row in labels),
        "candidate_audit": totals,
        "trace_measurements": _trace_measurement_summary(frames),
        "candidate_gate_keep_rate": (None if totals["candidates_before_gates"] == 0 else
                                      totals["eligible_after_gates"] / totals["candidates_before_gates"]),
        "frame_gate_coverage": (None if not len(frames) else
                                (len(frames) - totals["empty_candidate_frames"] - totals["all_gated_frames"]) / len(frames)),
        "methods": all_rows,
        "details": raw,
        "acceptance_policy": policy,
        "uncertainty": "sequence/object cluster bootstrap; no frame-level independence assumption",
        "overall_gate_passed": False,
    }


def evaluate_cache(cache_dir, *, split="test", learned_checkpoint=None,
                   learned_kind="learned", seed=42, bootstrap_draws=2000):
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    if learned_kind not in ("learned", "ordering-research"):
        raise ValueError("Unknown evaluation ranker kind")
    research = learned_kind == "ordering-research"
    if research and split not in ("train", "validation"):
        raise ValueError("Ordering research comparisons are limited to train/validation")
    audit = audit_cache(cache_dir, training_only=True) if research else audit_cache(cache_dir)
    if not audit["approved"]:
        raise ValueError("Candidate cache failed audit: " + "; ".join(audit["issues"]))
    if research:
        from .quality_pose_ranker import make_ranker
        ranker = make_ranker("ordering-research", learned_checkpoint)
        evidence = ranker.to_checkpoint()["provenance"]["identity"]["ordering_evidence"]
        if not any(source["identity"] == audit["identity"] for source in evidence["sources"]):
            raise ValueError("Ordering comparison cache is not a bound development source")
    manifest, _ = _cache_manifest(cache_dir)
    frames, labels = _load_split_rows(cache_dir, manifest, split)
    report = evaluate_frames(frames, labels, manifest["splits"].get("label_policy", None) or
                             _read_frozen_policy(cache_dir), learned_checkpoint=learned_checkpoint,
                             learned_kind=learned_kind,
                             seed=seed, bootstrap_draws=bootstrap_draws)
    report["split"] = split
    report["cache_identity"] = audit["identity"]
    report["frame_cache_sha256"] = manifest["files"][f"features_{split}"]["sha256"]
    report["label_cache_sha256"] = manifest["files"][f"labels_{split}"]["sha256"]
    report["learned_checkpoint_sha256"] = _checkpoint_sha256(learned_checkpoint)
    return report


def _read_frozen_policy(cache_dir):
    manifest, _ = _cache_manifest(cache_dir)
    split_path = _safe_cache_file(cache_dir, manifest["files"]["split_manifest"])
    with split_path.open("r", encoding="utf-8") as stream:
        frozen = json.load(stream)
    return frozen["label_policy"]


def _checkpoint_sha256(checkpoint):
    if checkpoint is None:
        return None
    if isinstance(checkpoint, (str, Path)):
        from .quality_ranker_data import _file_sha256
        return _file_sha256(checkpoint)
    return _sha256(_json_bytes(checkpoint))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-kind", choices=("learned", "ordering-research"), default="learned")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args(argv)
    report = evaluate_cache(args.cache_dir, split=args.split,
                            learned_checkpoint=args.learned_checkpoint,
                            learned_kind=args.learned_kind,
                            seed=args.seed, bootstrap_draws=args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace evaluation report: {args.output}")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": "evaluated", "split": args.split,
                      "frames": report["frame_count"], "methods": sorted(report["methods"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
