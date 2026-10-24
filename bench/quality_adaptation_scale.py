"""Metadata-only scale planning; preserves one global trainer dataset identity."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from . import quality_ranker_data as ranker
from .quality_kaggle_preflight import digest, read_json, require_digest, write_report


RAW_TOKEN_BYTES_PER_STATE = 4_915_200
ESTIMATED_RAW_BYTES_PER_STATE = 7_659_272
TARGET_REQUESTS = (1000, 2000, 4000)
MAX_STATES = 10000  # Existing lazy dataset's bound; no replacement trainer.


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _seal(value, key):
    return {**value, key: canonical_hash(value)}


def _check_seal(value, key):
    if value.get(key) != canonical_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"changed {key}")


def plan_scale(inventory, *, seed=42, shard_size=64, targets=TARGET_REQUESTS):
    """Group all shared exposure, then freeze frames using existing split guards.

    Each record is one refinement state, with stable id/refinement_state and
    inputs_sha256 supplied by the audited producer. Payload paths are never read.
    The same source frame may have several states, but stays in one split.
    """
    if type(seed) is not int or type(shard_size) is not int or not 1 <= shard_size <= 1000:
        raise ValueError("integer seed and shard_size in 1..1000 required")
    targets = sorted(set(targets))
    if not targets or any(type(n) is not int or not 1 <= n <= MAX_STATES for n in targets):
        raise ValueError("targets must be state-count requests in 1..10000")
    rows = inventory.get("records")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_STATES:
        raise ValueError("inventory requires 1..10000 metadata states")
    catalog = ranker._background_inventory(inventory)
    normalized = []
    seen = set()
    state_keys = set()
    for raw in rows:
        ident = ranker._identity(raw, known_background_ids=set(catalog))
        state_id = ranker._string(raw.get("id"), "state id")
        if state_id in seen:
            raise ValueError("duplicate state id")
        seen.add(state_id)
        state = raw.get("refinement_state")
        if type(state) is not int or state < 0:
            raise ValueError("refinement_state must be nonnegative integer")
        key = (ident["physical_object_id"], ident["sequence_id"], ident["source_frame_id"], state)
        if key in state_keys:
            raise ValueError("duplicate source-frame refinement state")
        state_keys.add(key)
        mesh = ranker._optional_mesh_digest(raw)
        if mesh is None:
            raise ValueError("exact metric mesh_sha256 required before scale planning")
        row = {**raw, **ident, "id": state_id, "refinement_state": state,
               "mesh_sha256": mesh, "inputs_sha256": require_digest(raw.get("inputs_sha256")),
               "aliases": sorted(ranker._aliases(raw))}
        if raw.get("split") is not None:
            raise ValueError("candidate inventory must be unsplit; historical manifests stay separate")
        if row["background_kind"] == "observed":
            bg = catalog[row["background_id"]]
            if bg["scene_id"] != row["scene_id"] or row["physical_object_id"] not in bg["physical_object_ids"]:
                raise ValueError("observed background does not match complete scene inventory")
        normalized.append(row)
    normalized.sort(key=lambda row: row["id"])
    parents = list(range(len(normalized)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for i, row in enumerate(normalized):
        # Does TRAIN indirectly expose TEST? Connect objects, metric geometry,
        # scenes, sequences, observed backgrounds and every co-occurring object.
        identities = {row["physical_object_id"].lower(), *row["aliases"]}
        if row["background_kind"] == "observed":
            identities.update(x.lower() for x in catalog[row["background_id"]]["physical_object_ids"])
        keys = [("object", x) for x in identities]
        keys += [(name, row[name]) for name in ("mesh_sha256", "scene_id", "sequence_id")]
        if row["background_kind"] == "observed":
            keys.append(("background", row["background_id"]))
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, row in enumerate(normalized):
        groups.setdefault(find(i), []).append(row)
    development, eligible = [], []
    for group in groups.values():
        dev = any(row["provenance_kind"] == "fixture" or
                  ranker._is_dev_only(row["physical_object_id"], row["aliases"]) or
                  (row["background_kind"] == "observed" and any(ranker._is_dev_only(x) for x in
                   catalog[row["background_id"]]["physical_object_ids"])) for row in group)
        (development if dev else eligible).append(group)
    # Keyed digest avoids dependence on input order or Python random versions.
    eligible.sort(key=lambda group: canonical_hash({"seed": seed, "ids": [r["id"] for r in group]}))
    feasible = len(eligible) >= 3
    assignments = {}
    for index, group in enumerate(eligible):
        split = ("validation" if index == 0 else "test" if index == 1 else "train") if feasible else "dev"
        for row in group:
            assignments[row["id"]] = split
    for group in development:
        assignments.update({row["id"]: "dev" for row in group})
    frame_rows = {}
    states = []
    for row in normalized:
        split = assignments[row["id"]]
        state = {**row, "split": split}
        states.append(state)
        frame_key = (row["physical_object_id"], row["sequence_id"], row["frame_id"])
        frozen = {key: state[key] for key in ("physical_object_id", "scene_id", "sequence_id", "background_id",
                  "background_kind", "provenance_kind", "frame_id", "source_frame_id", "mesh_sha256", "aliases", "split")}
        if frame_key in frame_rows and frame_rows[frame_key] != frozen:
            raise ValueError("refinement states change frame identity")
        frame_rows[frame_key] = frozen
    split_manifest = ranker.freeze_split_manifest({"records": list(frame_rows.values()),
        "background_catalog": list(catalog.values()), "label_policy": inventory.get("label_policy")})
    ranker.validate_split_manifest(split_manifest)
    global_manifest = _seal({"schema": "adaptation_scale_inventory_v1", "seed": seed,
        "split_manifest": split_manifest, "records": states}, "manifest_sha256")
    shards = []
    for split in ("train", "validation"):
        selected = [row for row in states if row["split"] == split]
        for offset in range(0, len(selected), shard_size):
            shards.append(_seal({"split": split, "manifest_sha256": global_manifest["manifest_sha256"],
                "rows": [{"id": r["id"], "inputs_sha256": r["inputs_sha256"]}
                         for r in selected[offset:offset + shard_size]]}, "shard_sha256"))
    counts = {}
    for split in ranker.SPLITS:
        selected = [r for r in states if r["split"] == split]
        counts[split] = {"refinement_states": len(selected),
            "objects": len({r["physical_object_id"] for r in selected}),
            "scenes": len({r["scene_id"] for r in selected}),
            "source_frames": len({(r["scene_id"], r["source_frame_id"]) for r in selected})}
    return _seal({"schema": "adaptation_scale_plan_v1", "global_manifest": global_manifest,
        "feasible": feasible, "blockers": [] if feasible else ["fewer_than_three_independent_nondev_components"],
        "independent_nondev_components": len(eligible), "counts": counts, "shards": shards,
        "target_requests": [{"train_states_requested": n, "train_states_in_inventory": counts["train"]["refinement_states"],
            "request_met_by_inventory": counts["train"]["refinement_states"] >= n,
            "estimated_raw_bytes": n * ESTIMATED_RAW_BYTES_PER_STATE} for n in targets],
        "raw_token_bytes_per_state": RAW_TOKEN_BYTES_PER_STATE,
        "raw_estimated_bytes_per_state": ESTIMATED_RAW_BYTES_PER_STATE,
        "estimate_excludes": "source assets, checkpoints, temporary files, evaluation and filesystem overhead",
        "shard_policy": "storage/preparation only; assemble one immutable global dataset before trainer resume",
        "acquisition_claim": False, "admitted": False}, "plan_sha256")


def validate_plan(plan):
    """Rebuild semantic invariants before any preparation output can be opened.

    A self-contained digest is not an authority: corrupt producers can reseal
    inconsistent metadata. Reusing the constructor reconciles state identities,
    deterministic split assignments, frozen frames, counts and target coverage.
    Shards may partition storage differently, but must cover TRAIN/VAL exactly.
    """
    if not isinstance(plan, dict) or plan.get("schema") != "adaptation_scale_plan_v1":
        raise ValueError("unsupported scale plan")
    _check_seal(plan, "plan_sha256")
    manifest = plan.get("global_manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != "adaptation_scale_inventory_v1":
        raise ValueError("unsupported scale inventory")
    _check_seal(manifest, "manifest_sha256")
    frozen = ranker.validate_split_manifest(manifest.get("split_manifest"))
    rows = manifest.get("records")
    targets = plan.get("target_requests")
    if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or
            not isinstance(targets, list) or not all(isinstance(row, dict) for row in targets)):
        raise ValueError("scale records and target requests must be lists of objects")
    expected = plan_scale({"records": [{k: v for k, v in row.items() if k != "split"} for row in rows],
        "background_catalog": frozen["background_catalog"], "label_policy": frozen["label_policy"]},
        seed=manifest.get("seed"), shard_size=1, targets=[row.get("train_states_requested") for row in targets])
    ignored = {"shards", "plan_sha256"}
    if canonical_hash({k: v for k, v in plan.items() if k not in ignored}) != canonical_hash(
            {k: v for k, v in expected.items() if k not in ignored}):
        raise ValueError("scale metadata disagrees with reconstructed inventory and frozen splits")
    by_id = {row["id"]: row for row in expected["global_manifest"]["records"]}
    required = {key for key, row in by_id.items() if row["split"] in ("train", "validation")}
    shards = plan.get("shards")
    if not isinstance(shards, list) or len(shards) > MAX_STATES:
        raise ValueError("bounded preparation shard list required")
    seen = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"split", "manifest_sha256", "rows", "shard_sha256"}:
            raise ValueError("invalid preparation shard fields")
        _check_seal(shard, "shard_sha256")
        if shard["split"] not in ("train", "validation") or shard["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("shard changed global identity or contains sealed split")
        if not isinstance(shard["rows"], list) or not 1 <= len(shard["rows"]) <= 1000:
            raise ValueError("preparation shard requires 1..1000 state rows")
        for row in shard["rows"]:
            if not isinstance(row, dict) or set(row) != {"id", "inputs_sha256"} or not isinstance(row.get("id"), str):
                raise ValueError("invalid preparation shard row")
            state = by_id.get(row["id"])
            if (state is None or state["split"] != shard["split"] or
                    state["inputs_sha256"] != row["inputs_sha256"]):
                raise ValueError("preparation shard row disagrees with frozen state split/input identity")
            if row["id"] in seen:
                raise ValueError("duplicate preparation state across shards")
            seen.add(row["id"])
    if seen != required:
        raise ValueError("preparation shards must cover every TRAIN/VAL state exactly once")
    return plan


def validate_preparation_receipt(plan, receipt, *, producer_sha256, config_sha256, output_root):
    """Verify only explicitly listed successful TRAIN/VAL preparation outputs.

    Failed rows remain in the receipt. Unknown/TEST rows fail before any output
    path is touched. Producer and configuration changes invalidate resume.
    """
    validate_plan(plan)
    require_digest(producer_sha256)
    require_digest(config_sha256)
    if receipt.get("producer_sha256") != producer_sha256 or receipt.get("config_sha256") != config_sha256:
        raise ValueError("preparation producer/config changed")
    if receipt.get("manifest_sha256") != plan["global_manifest"]["manifest_sha256"]:
        raise ValueError("preparation global manifest changed")
    shards = {s["shard_sha256"]: s for s in plan["shards"]}
    shard = shards.get(receipt.get("shard_sha256"))
    if shard is None:
        raise ValueError("unknown preparation shard")
    expected = {r["id"]: r["inputs_sha256"] for r in shard["rows"]}
    results = receipt.get("rows")
    if not isinstance(results, list) or len(results) != len(expected) or {r.get("id") for r in results} != set(expected):
        raise ValueError("receipt must retain every shard row exactly once")
    # Validate identities and failure metadata for ALL rows before opening files.
    for row in results:
        if row.get("inputs_sha256") != expected[row["id"]]:
            raise ValueError("preparation input identity changed")
        if row.get("status") not in ("available", "failed"):
            raise ValueError("preparation status must be available or failed")
        if row["status"] == "failed" and (not isinstance(row.get("failure_reason"), str) or not row["failure_reason"].strip()):
            raise ValueError("failed rows require failure_reason")
    elapsed = _positive(receipt.get("preparation_seconds"), "measured preparation_seconds")
    root = Path(output_root).resolve()
    total = 0
    for row in results:
        if row["status"] == "failed":
            continue
        output = row.get("output", {})
        relative = output.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
            raise ValueError("output requires relative POSIX path")
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_relative_to(root) or path == root:
            raise ValueError("output escapes preparation root")
        require_digest(output.get("sha256"))
        size = output.get("bytes")
        if type(size) is not int or size <= 0 or not path.is_file() or path.stat().st_size != size or digest(path) != output["sha256"]:
            raise ValueError("prepared output missing or changed")
        total += size
    counts = Counter(r["status"] for r in results)
    return {"shard_sha256": shard["shard_sha256"], "available": counts["available"],
            "failed": counts["failed"], "total": len(results), "output_bytes": total,
            "preparation_seconds": elapsed, "failed_rows": [r for r in results if r["status"] == "failed"]}


def assess_admission(plan, receipts=(), *, measurements=None, producer_sha256=None,
                     config_sha256=None, output_root=None):
    """Conservative resource gate; never schedules training or substitutes estimates."""
    validate_plan(plan)
    blockers = list(plan["blockers"])
    verified_receipts = []
    for receipt in receipts:
        if output_root is None:
            raise ValueError("explicit preparation output root required")
        verified_receipts.append(validate_preparation_receipt(plan, receipt,
            producer_sha256=producer_sha256, config_sha256=config_sha256, output_root=output_root))
    expected = {s["shard_sha256"] for s in plan["shards"]}
    actual = [r["shard_sha256"] for r in verified_receipts]
    if len(actual) != len(set(actual)) or set(actual) != expected or not expected:
        blockers.append("all_unique_verified_preparation_receipts_required")
    if any(r["failed"] for r in verified_receipts):
        blockers.append("failed_preparation_rows_retained")
    resource = None
    if measurements is None:
        blockers.append("measured_time_storage_and_limits_required")
    else:
        if measurements.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("resource evidence changed plan identity")
        target = measurements.get("train_states_requested")
        if type(target) is not int or target not in {r["train_states_requested"] for r in plan["target_requests"]}:
            raise ValueError("resource evidence must select a frozen train-state request")
        if plan["counts"]["train"]["refinement_states"] < target:
            blockers.append("requested_train_states_not_in_inventory")
        keys = ("measured_training_steps", "training_seconds", "planned_optimizer_steps",
                "actual_free_bytes", "storage_limit_bytes", "time_limit_seconds", "reserve_bytes")
        values = {k: _positive(measurements.get(k), k) for k in keys}
        prep = sum(r["preparation_seconds"] for r in verified_receipts)
        storage = sum(r["output_bytes"] for r in verified_receipts)
        # The verified preparation outputs already occupy the measured filesystem.
        # Current free space is observed AFTER preparation. Resident usage includes
        # all assets/outputs charged to this quota, not only these prepared rows.
        if measurements.get("storage_baseline") != "after_preparation":
            raise ValueError("storage evidence must use after_preparation baseline")
        resident = measurements.get("measured_resident_bytes")
        extra = measurements.get("measured_additional_peak_bytes")
        if type(resident) is not int or resident < storage:
            raise ValueError("measured resident bytes must include all verified prepared outputs")
        if type(extra) is not int or extra < 0:
            raise ValueError("measured additional peak bytes must be a nonnegative integer")
        seconds = prep + values["training_seconds"] / values["measured_training_steps"] * values["planned_optimizer_steps"]
        peak = resident + extra
        if extra + values["reserve_bytes"] > values["actual_free_bytes"]:
            blockers.append("additional_storage_exceeds_free_headroom_or_reserve")
        if peak + values["reserve_bytes"] > values["storage_limit_bytes"]:
            blockers.append("total_storage_exceeds_quota_or_reserve")
        if seconds > values["time_limit_seconds"]:
            blockers.append("measured_time_projection_exceeds_limit")
        resource = {"projected_preparation_training_seconds": seconds, "measured_prepared_bytes": storage,
                    "storage_baseline": "after_preparation", "measured_resident_bytes": resident,
                    "measured_additional_peak_bytes": extra,
                    "projected_peak_bytes": peak, "projection_is_not_runtime_guarantee": True}
    return {"schema": "adaptation_scale_admission_v1", "plan_sha256": plan["plan_sha256"],
            "admitted": not blockers, "blockers": blockers, "resources": resource,
            "admission_scope": "prepared_inventory_resource_fit_only_not_training_runtime_or_model_quality",
            "quality_claim": False, "trainer_manifest_assembly_still_required": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args(argv)
    plan = plan_scale(read_json(args.inventory), seed=args.seed, shard_size=args.shard_size)
    write_report(args.output, plan)
    return 0 if plan["feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
