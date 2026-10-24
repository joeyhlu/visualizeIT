"""Explicit consumer for one immutable, expanded global training dataset.

The wrapper verifies the new aggregate seal separately from the original child
RAW seals. It keeps the original extractor, numerical prelabel validator,
dataset validators, trainer and export code unchanged. No model is imported by
module import, and all cloud execution remains opt-in through a fresh request.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_scale as scale
from bench import quality_adaptation_sharded_v1 as sharded
from scripts import kaggle_prepared_v2 as prepared
from bench.quality_adaptation_portable_prepare import *  # noqa: F401,F403 - matching the reviewed label lifecycle namespace

GIB = 1024**3
MIB = 1024**2
GLOBAL_ACTIVE_STATES = 336
GLOBAL_FEATURE_ENVELOPE = GLOBAL_ACTIVE_STATES * p.PER_STATE_ENVELOPE + p.TEMP_ENVELOPE
AGGREGATE_STORAGE_LIMIT = 32 * GIB
MAX_CONTROL_WRITE = 32 * MIB
MAX_STAGE_LOG = 16 * MIB
REQUEST_SCHEMA = "kaggle_sharded_prepared_request_v1"
PILOT_SCHEMA = "visualizeit_sharded_pilot_measurement_v1"
CHILD_SUPERVISION_SCHEMA = "kaggle_sharded_supervision_v1"
CHILD_REQUEST_VERSION = "visualizeit_sharded_child_preparation_v1"
CHILD_SOURCE_BINDING_SCHEMA = "visualizeit_expanded_child_source_binding_v1"
CHILD_STAGE_NAMES = ("identity", "extract", "seal-features", "labels")
RUN_IDENTITY = "sharded-run-identity.json"
CHECKPOINT_RECEIPT = "sharded-checkpoint-receipt.json"
SOURCE_NAMES = (
    "bench/quality_adaptation_portable.py",
    "bench/quality_adaptation_portable_prepare.py",
    "bench/quality_adaptation_scale.py",
    "bench/quality_adaptation_sharded_v1.py",
    "bench/quality_adaptation_prepare.py",
    "bench/quality_adaptation_data.py",
    "bench/quality_adaptation_train.py",
    "bench/quality_adaptation_runtime.py",
    "bench/quality_adaptation_smoke.py",
    "bench/quality_adaptation_supervised_v1.py",
    "scripts/kaggle_adaptation.py",
    "scripts/kaggle_prepared_v2.py",
    "scripts/kaggle_pilot_v2.py",
    "scripts/kaggle_supervision_v1.py",
    "scripts/kaggle_sharded_prepared_v1.py",
    "scripts/kaggle_label_order_v1.py",
)


def _migration_kwargs(request):
    return {} if request.get("source_migration") is None else {"source_migration": request["source_migration"]}


def _raw_join_kwargs(request):
    values = _migration_kwargs(request)
    if request.get("raw_continuation") is not None:
        values["raw_continuation"] = request["raw_continuation"]
    return values


def _validate_sharded_limits(limits):
    if (limits["max_rss_bytes"] > 8 * GIB or limits["max_output_bytes"] > 8 * GIB
            or limits["max_log_bytes"] > MAX_STAGE_LOG or limits["reserve_bytes"] < GIB):
        raise ValueError("sharded resources exceed retained 8-GiB RSS/output, 16-MiB log and 1-GiB reserve bounds")
    return limits


def _stage_write_allocations(stage, *, working_device, feature_device=None,
                             output_device=None, cap=0, existing=0,
                             label_copy_bytes=0):
    """Return future writes per filesystem without counting resident output twice."""
    allocations = {working_device: MAX_CONTROL_WRITE}
    if stage == "extract":
        if feature_device is None or feature_device == working_device:
            raise ValueError("feature output must use its dedicated filesystem")
        allocations[feature_device] = allocations.get(feature_device, 0) + max(0, cap - existing)
    elif stage == "labels":
        if feature_device is None or feature_device == working_device:
            raise ValueError("feature control writes must use their dedicated filesystem")
        if type(label_copy_bytes) is not int or label_copy_bytes < 0:
            raise ValueError("label copy write allocation must be a nonnegative byte count")
        allocations[feature_device] = allocations.get(feature_device, 0) + MAX_CONTROL_WRITE + label_copy_bytes
    elif stage == "child-labels":
        if feature_device is None or feature_device == working_device:
            raise ValueError("child label output must use its dedicated filesystem")
        allocations[feature_device] = allocations.get(feature_device, 0) + max(0, cap - existing)
    elif stage == "assemble":
        if feature_device is None or feature_device == working_device:
            raise ValueError("assembled feature output must use its dedicated filesystem")
        allocations[feature_device] = allocations.get(feature_device, 0) + MAX_CONTROL_WRITE
    elif stage in ("train", "resume", "export", "pilot"):
        if output_device is None or output_device != working_device:
            raise ValueError("trainer output must remain on the working filesystem")
        allocations[output_device] = allocations.get(output_device, 0) + max(0, cap - existing)
    elif stage not in ("identity", "seal-features", "preflight", "aggregate"):
        raise ValueError("unknown sharded quota stage")
    return allocations


def _enforce_storage_reserve(resident, allocations, free_by_device, reserve, storage_limit):
    if resident + sum(allocations.values()) + reserve > storage_limit:
        raise ValueError("combined resident roots, stage writes and reserve exceed the prospective storage limit")
    for device, amount in allocations.items():
        if device not in free_by_device or free_by_device[device] < amount + reserve:
            raise ValueError("filesystem write allocation/reserve failed before launch on device " + str(device))


def _canonical(value):
    return p.canonical(value)


def _as_pin(value):
    if isinstance(value, (str, Path)):
        return p.pin(value)
    if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}:
        p.verify_pin(value)
        return value
    raise ValueError("file path or exact pinned descriptor required")


def _global_stage_binding_path(request, identity=None):
    identity = identity or request.get("stage_request_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("global stage binding requires its frozen preparation identity")
    working = p.no_links(Path(request["working_root"]).absolute())
    return p.no_links(working / (".visualizeit-sharded-global-stage-" + identity + ".json"))


def _sealed(value, field):
    return {**value, field: _canonical(value)}


def _check_seal(value, field, schema):
    if not isinstance(value, dict) or value.get("schema") != schema or value.get(field) != _canonical({k: v for k, v in value.items() if k != field}):
        raise ValueError("invalid " + schema + " seal")
    return value


def _root_sources(root):
    root = p.no_links(Path(root).absolute())
    return {name: p.pin(root / name) for name in SOURCE_NAMES}


def verify_wrapper_sources(request):
    root = p.no_links(Path(request["source_root"]))
    sources = request.get("source_files")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_NAMES):
        raise ValueError("exact expanded wrapper/source closure required")
    for name in SOURCE_NAMES:
        if p.descriptor_path(sources[name]) != root / name:
            raise ValueError("sharded source path differs from pinned package: " + name)
        p.verify_pin(sources[name])
    adapter = p.verify_pin(request["adapter"])
    if adapter != root / "scripts/kaggle_adaptation.py":
        raise ValueError("immutable validation adapter differs")
    return root


def request_document(*, partition, aggregate_root, execution_root, features_root,
                     feature_storage_root,
                     runtime, config, source_install, python, input_root,
                     working_root, output_root, pilot_output_root,
                     child_requests, shard_refs, ui_context, execution_evidence,
                     limits, quota_roots, neural_inputs=(), pilot_measurement=None,
                     validation_diagnostic=None, preparation_request=None, source_migration=None):
    """Build a final request while retaining the completed staged-preparation history."""
    partition = _as_pin(partition)
    migration = {} if source_migration is None else {"source_migration": _as_pin(source_migration)}
    partition_doc = sharded.verify_partition(partition, **migration)
    # The source root is obtained from the original context, not inferred from
    # the caller's CWD. Context and payload contents remain separately pinned.
    smoke_pin = partition_doc["imported_shard0"]["context"]
    smoke_context = raw.read_context(Path(smoke_pin["path"]), smoke_pin["sha256"])
    source_root = Path(smoke_context.document()["source_root"])
    features, execution = Path(features_root), Path(execution_root)
    aggregate = Path(aggregate_root)
    children = list(child_requests)
    if len(children) != len(sharded.SHARD_IDS):
        raise ValueError("four frozen child prepared requests required")
    normalized_children = []
    for child in children:
        if not isinstance(child, dict) or set(child) != {"shard_id", "request", "projection"}:
            raise ValueError("each child request requires explicit shard_id/request/projection")
        child_request = child["request"]
        if isinstance(child_request, (str, Path)):
            child_request = p.pin(child_request)
        if not isinstance(child_request, dict):
            raise ValueError("child prepared request pin/document required")
        normalized_children.append({**child, "request": child_request})
    children = normalized_children
    refs = list(shard_refs)
    if len(refs) != len(sharded.SHARD_IDS):
        raise ValueError("four frozen child RAW projections required")
    value = {
        **migration,
        "schema": REQUEST_SCHEMA, "source_root": str(source_root),
        "source_files": _root_sources(source_root), "adapter": p.pin(source_root / "scripts/kaggle_adaptation.py"),
        "source_install": str(source_install), "python": str(python), "input_root": str(input_root),
        "working_root": str(working_root), "aggregate_root": str(aggregate),
        "execution_root": str(execution), "features_root": str(features),
        "feature_storage_root": str(p.no_links(Path(feature_storage_root).absolute())),
        "output_root": str(output_root), "pilot_output_root": str(pilot_output_root),
        "runtime": _as_pin(runtime), "config": _as_pin(config), "neural_inputs": list(neural_inputs),
        "partition": partition, "shard_refs": refs,
        "child_requests": children,
        "ui_context": _as_pin(ui_context), "execution_evidence": _as_pin(execution_evidence),
        "limits": limits, "quota_roots": list(map(str, quota_roots)),
        "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
        "profile": {"kind": "expanded13", "train_states_minimum": 240,
                    "train_objects": 13, "train_states_per_object_minimum": 12,
                    "validation_states_minimum": 1},
        "paths": {"extraction_dir": str(features), "manifest_output": str(features / "manifest.json"),
                  "train_dir": str(Path(output_root) / "train"),
                  "export_output": str(Path(output_root) / "train/export.pt")},
        "artifacts": {
            "aggregate_seal": p.pin(aggregate / "raw-seal.json"),
            "execution_plan": p.pin(execution / "execution-plan.json"),
            "identity_bridge": p.pin(execution / "identity-bridge.json"),
            "prelabel_seal": p.pin(execution / "prelabel-seal.json"),
            "extraction_binding": p.pin(features / "extraction-binding.json"),
            "original_recipe": p.pin(features / "original-recipe.json"),
            "original_index": p.pin(features / "original-index.json"),
            "feature_index": p.pin(features / "index.json"),
            "labels": p.pin(features / "labels.json"),
            "label_receipt": p.pin(features / "label-receipt.json"),
            "dataset": p.pin(features / "manifest.json"),
        },
    }
    if pilot_measurement is not None:
        value["pilot_measurement"] = _as_pin(pilot_measurement)
    if validation_diagnostic is not None:
        value["validation_diagnostic"] = _as_pin(validation_diagnostic)
    preparation = _child_request(preparation_request) if preparation_request is not None else None
    if not isinstance(preparation, dict) or preparation.get("schema") != REQUEST_SCHEMA:
        raise ValueError("final aggregate request requires its validated early preparation request")
    immutable_keys = ("source_root", "source_files", "adapter", "source_install", "python", "input_root",
        "working_root", "aggregate_root", "execution_root", "features_root", "feature_storage_root",
        "output_root", "pilot_output_root", "runtime", "config", "neural_inputs", "partition",
        "shard_refs", "child_requests", "limits", "quota_roots", "storage_limit_bytes", "profile", "paths", "source_migration")
    if any(preparation.get(key) != value.get(key) for key in immutable_keys):
        raise ValueError("final request changes immutable aggregate preparation identity or resource limits")
    for key in ("initial_ui_context", "initial_execution_evidence", "global_stage_binding",
                "stage_evidence_history", "stage_request_identity_sha256"):
        value[key] = preparation[key]
    return value


def _preparation_request_value(*, partition, aggregate_root, execution_root, features_root,
                     feature_storage_root, runtime, config, source_install, python, input_root,
                     working_root, output_root, pilot_output_root, child_requests, shard_refs,
                     ui_context, execution_evidence=None, limits, quota_roots, neural_inputs=(),
                     validation_diagnostic=None, source_migration=None):
    """Normalize the immutable request fields shared by preview and publication."""
    migration = {} if source_migration is None else {"source_migration": _as_pin(source_migration)}
    partition_doc = sharded.verify_partition(partition, **migration)
    smoke_pin = partition_doc["imported_shard0"]["context"]
    smoke_context = raw.read_context(Path(smoke_pin["path"]), smoke_pin["sha256"])
    source_root = Path(smoke_context.document()["source_root"])
    aggregate = p.no_links(Path(aggregate_root).absolute())
    execution, features = p.no_links(Path(execution_root).absolute()), p.no_links(Path(features_root).absolute())
    children = []
    for value in child_requests:
        if not isinstance(value, dict) or set(value) != {"shard_id", "request", "projection"}:
            raise ValueError("each child needs frozen shard id, request and projection")
        child_request = value["request"]
        child_pin = p.pin(child_request) if isinstance(child_request, (str, Path)) else child_request
        if not isinstance(child_pin, dict):
            raise ValueError("child request path or exact metadata pin required")
        children.append({**value, "request": child_pin})
    refs = list(shard_refs)
    if len(children) != len(sharded.SHARD_IDS) or len(refs) != len(sharded.SHARD_IDS):
        raise ValueError("all four frozen shard RAW/projection references and prepared requests required")
    ref_map = {}
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"shard_id", "context", "raw_seal", "projection"}:
            raise ValueError("exact shard RAW reference required")
        sid = ref["shard_id"]
        if sid in ref_map:
            raise ValueError("duplicate child RAW reference")
        sharded.verify_shard(partition, sid, ref["context"], ref["raw_seal"], ref["projection"], payloads=False, **migration)
        ref_map[sid] = ref
    if set(ref_map) != set(sharded.SHARD_IDS):
        raise ValueError("missing or foreign child RAW reference")
    for child in children:
        sid = child["shard_id"]
        if sid not in ref_map or child["projection"] != ref_map[sid]["projection"]:
            raise ValueError("prepared child projection differs from frozen RAW reference")
        if set(child["request"]) == {"path", "bytes", "sha256"}:
            child_doc = p.metadata(child["request"])
        else:
            child_doc = child["request"]
        if child_doc.get("context") != ref_map[sid]["context"] or child_doc.get("raw_seal") != ref_map[sid]["raw_seal"]:
            raise ValueError("prepared child context/RAW seal differs from frozen reference")
    evidence_pin = None if execution_evidence is None else _as_pin(execution_evidence)
    ui_pin = _as_pin(ui_context)
    value = {
        **migration,
        "schema": REQUEST_SCHEMA, "source_root": str(source_root),
        "source_files": _root_sources(source_root), "adapter": p.pin(source_root / "scripts/kaggle_adaptation.py"),
        "source_install": str(source_install), "python": str(p.prepared_python(python,
            _as_pin(runtime), p.pin(source_root / "scripts/kaggle_adaptation.py"), working_root)),
        "input_root": str(p.no_links(Path(input_root).absolute())),
        "working_root": str(p.no_links(Path(working_root).absolute())),
        "aggregate_root": str(aggregate), "execution_root": str(execution), "features_root": str(features),
        "feature_storage_root": str(p.no_links(Path(feature_storage_root).absolute())),
        "output_root": str(p.no_links(Path(output_root).absolute())),
        "pilot_output_root": str(p.no_links(Path(pilot_output_root).absolute())),
        "runtime": _as_pin(runtime), "config": _as_pin(config),
        "neural_inputs": list(neural_inputs), "partition": partition, "shard_refs": refs,
        "child_requests": children, "ui_context": ui_pin,
        "execution_evidence": evidence_pin, "limits": limits,
        "initial_ui_context": ui_pin,
        "initial_execution_evidence": evidence_pin,
        "quota_roots": list(map(str, quota_roots)), "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
        "profile": {"kind": "expanded13", "train_states_minimum": 240,
                    "train_objects": 13, "train_states_per_object_minimum": 12,
                    "validation_states_minimum": 1},
        "paths": {"extraction_dir": str(features), "manifest_output": str(features / "manifest.json"),
                  "train_dir": str(Path(output_root) / "train"),
                  "export_output": str(Path(output_root) / "train/export.pt")},
        "artifacts": {"aggregate_seal": p.pin(aggregate / "raw-seal.json")},
    }
    if validation_diagnostic is not None:
        value["validation_diagnostic"] = _as_pin(validation_diagnostic)
    _check_layout(value)
    return value


def prospective_preparation_identity(*, partition, aggregate_root, execution_root, features_root,
                     feature_storage_root, runtime, config, source_install, python, input_root,
                     working_root, output_root, pilot_output_root, child_requests, shard_refs,
                     ui_context, limits, quota_roots, neural_inputs=(), validation_diagnostic=None, source_migration=None):
    """Compute the stable identity needed to create identity-bound evidence.

    This metadata-only preview performs the same request normalization and
    aggregate validation as publication, but writes no request or binding file.
    """
    value = _preparation_request_value(partition=partition, aggregate_root=aggregate_root,
        execution_root=execution_root, features_root=features_root,
        feature_storage_root=feature_storage_root, runtime=runtime, config=config,
        source_install=source_install, python=python, input_root=input_root,
        working_root=working_root, output_root=output_root, pilot_output_root=pilot_output_root,
        child_requests=child_requests, shard_refs=shard_refs, ui_context=ui_context,
        execution_evidence=None, limits=limits, quota_roots=quota_roots,
        neural_inputs=neural_inputs, validation_diagnostic=validation_diagnostic, source_migration=source_migration)
    raw_state = sharded.verify_aggregate_raw(partition, aggregate_root, **_migration_kwargs(value))
    return _preparation_identity(value, raw_state)


def preparation_request_document(*, partition, aggregate_root, execution_root, features_root,
                     feature_storage_root,
                     runtime, config, source_install, python, input_root,
                     working_root, output_root, pilot_output_root,
                     child_requests, shard_refs, ui_context, execution_evidence,
                     limits, quota_roots, neural_inputs=(), validation_diagnostic=None, source_migration=None):
    """Build a stage request before execution/prelabel/label outputs exist.

    This request intentionally carries only the aggregate RAW seal pin. The
    prepared-artifact pins are added later by `finalize_prepared_request`.
    """
    value = _preparation_request_value(partition=partition, aggregate_root=aggregate_root,
        execution_root=execution_root, features_root=features_root,
        feature_storage_root=feature_storage_root, runtime=runtime, config=config,
        source_install=source_install, python=python, input_root=input_root,
        working_root=working_root, output_root=output_root, pilot_output_root=pilot_output_root,
        child_requests=child_requests, shard_refs=shard_refs, ui_context=ui_context,
        execution_evidence=execution_evidence, limits=limits, quota_roots=quota_roots,
        neural_inputs=neural_inputs, validation_diagnostic=validation_diagnostic, source_migration=source_migration)
    if value["execution_evidence"] is None:
        raise ValueError("identity-bound initial execution evidence is required for request publication")
    raw_state = sharded.verify_aggregate_raw(partition, aggregate_root, **_migration_kwargs(value))
    preparation_identity = _preparation_identity(value, raw_state)
    # Keep control metadata outside the still-fresh execution root. The
    # unchanged original execution identity producer creates that directory.
    binding_path = _global_stage_binding_path(value, preparation_identity)
    if binding_path.exists():
        raise FileExistsError("fresh immutable global preparation binding required")
    binding = _sealed({"schema": "visualizeit_sharded_global_stage_binding_v1",
        "preparation_identity_sha256": preparation_identity,
        "initial_ui_context": value["initial_ui_context"],
        "initial_execution_evidence": value["initial_execution_evidence"],
        "storage_limit_bytes": value["storage_limit_bytes"],
        "limits": value["limits"], "runtime": value["runtime"], "config": value["config"],
        "source_files": value["source_files"], "partition": value["partition"],
        "aggregate_seal": value["artifacts"]["aggregate_seal"],
        "quota_roots": value["quota_roots"], "paid_resources": False}, "binding_sha256")
    p.atomic_json(binding_path, binding)
    value["global_stage_binding"] = p.pin(binding_path)
    value["stage_evidence_history"] = [{"stage": "identity", "ui_context": value["ui_context"],
        "execution_evidence": value["execution_evidence"]}]
    value["stage_request_identity_sha256"] = preparation_identity
    _verify_global_stage_history(value, raw_state)
    return value


def finalize_prepared_request(request, *, execution_evidence=None, ui_context=None, pilot_measurement=None):
    """Pin the completed global artifacts after identity/extract/seal/labels."""
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("expanded preparation request required")
    value = dict(request)
    aggregate, execution, features = (Path(value[k]) for k in ("aggregate_root", "execution_root", "features_root"))
    value["artifacts"] = {
        "aggregate_seal": p.pin(aggregate / "raw-seal.json"),
        "execution_plan": p.pin(execution / "execution-plan.json"),
        "identity_bridge": p.pin(execution / "identity-bridge.json"),
        "prelabel_seal": p.pin(execution / "prelabel-seal.json"),
        "extraction_binding": p.pin(features / "extraction-binding.json"),
        "original_recipe": p.pin(features / "original-recipe.json"),
        "original_index": p.pin(features / "original-index.json"),
        "feature_index": p.pin(features / "index.json"),
        "labels": p.pin(features / "labels.json"),
        "label_receipt": p.pin(features / "label-receipt.json"),
        "dataset": p.pin(features / "manifest.json"),
    }
    if execution_evidence is not None:
        value["execution_evidence"] = _as_pin(execution_evidence)
    if ui_context is not None:
        value["ui_context"] = _as_pin(ui_context)
    if pilot_measurement is not None:
        value["pilot_measurement"] = _as_pin(pilot_measurement)
    _check_layout(value)
    return value


def _child_request(value):
    if isinstance(value, (str, Path)):
        value = p.pin(value)
    return p.metadata(value) if isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"} else value


CHILD_BRIDGE_SOURCES = (
    "scripts/kaggle_sharded_prepared_v1.py",
    "scripts/kaggle_supervision_v1.py",
    "scripts/kaggle_label_order_v1.py",
    "bench/quality_adaptation_portable.py",
)


def _child_supervision_sources(root):
    root = p.no_links(Path(root).absolute())
    return {name: p.pin(root / name) for name in CHILD_BRIDGE_SOURCES}


def _ordered_child_label_artifacts(request, execution, inventory, artifacts):
    """Use the unchanged canonical artifact order for original child validators."""
    pin = request.get("sharded_child", {}).get("supervision_sources", {}).get(
        "scripts/kaggle_label_order_v1.py")
    if not isinstance(pin, dict):
        raise ValueError("child request lacks the authenticated original label-order helper")
    helper = p.load_source(pin, "_sharded_label_order_adapter")
    plan = p.read_json(Path(execution) / "execution-plan.json")
    return helper.ordered_artifacts(plan, inventory, artifacts)


def child_request_document(*, partition, shard_id, context, raw_seal, projection,
                           exposure_audit, label_plan, source_root, source_install,
                           python, runtime, config, adapter, input_root, working_root,
                           execution_root, features_root, feature_storage_root,
                           retained_raw_roots, control_root, ui_context,
                           execution_evidence, limits, quota_roots, neural_inputs=(),
                           defer_label_plan=False, raw_continuation=None, source_migration=None):
    """Build an early child-stage request without pins to outputs not yet made."""
    partition_doc = sharded.verify_partition(partition, **({} if source_migration is None else {"source_migration": source_migration}))
    if shard_id not in partition_doc["children"] or shard_id == "smoke0":
        raise ValueError("new 32-GiB child request is only for newly prepared shards; smoke0 remains imported")
    context_pin, raw_pin = _as_pin(context), _as_pin(raw_seal)
    exposure_pin, projection_pin = _as_pin(exposure_audit), _as_pin(projection)
    source_root = p.no_links(Path(source_root).absolute())
    if set(prepared.SOURCE_NAMES) != {
            "bench/quality_adaptation_portable_prepare.py", "scripts/kaggle_prepared_v2.py",
            "scripts/kaggle_pilot_v2.py"}:
        raise ValueError("expanded wrapper source closure incomplete")
    roots = {str(p.no_links(Path(value).absolute())) for value in quota_roots}
    value = {
        "schema": "kaggle_prepared_request_v2",
        "source_root": str(source_root),
        "source_files": {name: p.pin(source_root / name) for name in prepared.SOURCE_NAMES},
        "adapter": _as_pin(adapter), "source_install": str(p.no_links(Path(source_install).absolute())),
        "python": str(p.prepared_python(python, _as_pin(runtime), _as_pin(adapter), working_root)),
        "runtime": _as_pin(runtime), "config": _as_pin(config),
        "input_root": str(p.no_links(Path(input_root).absolute())),
        "working_root": str(p.no_links(Path(working_root).absolute())),
        "execution_root": str(p.no_links(Path(execution_root).absolute())),
        "features_root": str(p.no_links(Path(features_root).absolute())),
        "feature_storage_root": str(p.no_links(Path(feature_storage_root).absolute())),
        "retained_raw_roots": sorted({str(p.no_links(Path(value).absolute())) for value in retained_raw_roots}),
        "control_root": str(p.no_links(Path(control_root).absolute())),
        "context": context_pin, "raw_seal": raw_pin, "exposure_audit": exposure_pin,
        "neural_inputs": list(neural_inputs), "limits": limits,
        "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT, "quota_roots": sorted(roots),
        "profile": {"kind": "sharded_child", "shard_id": shard_id},
        "sharded_child": {"schema": CHILD_REQUEST_VERSION, "partition": partition,
            "shard_id": shard_id, "projection": projection_pin,
            "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
            "supervision_sources": _child_supervision_sources(source_root)},
        "ui_context": _as_pin(ui_context), "execution_evidence": _as_pin(execution_evidence),
    }
    if raw_continuation is not None:
        value["raw_continuation"] = _as_pin(raw_continuation)
    if source_migration is not None:
        value["source_migration"] = _as_pin(source_migration)
    if defer_label_plan:
        if defer_label_plan is not True or label_plan is not None:
            raise ValueError("deferred child labels require an absent initial plan")
        value["sharded_child"].update(label_binding_mode="deferred_v1",
            label_plan_path=str(Path(value["control_root"]) / "child-label-plan.json"))
    else:
        value["label_plan"] = _as_pin(label_plan)
    _check_child_layout(value)
    _validate_child_evidence_metadata(value, value["execution_evidence"], value["ui_context"])
    binding_path = Path(value["control_root"]) / "child-source-binding.json"
    if binding_path.exists():
        raise FileExistsError("fresh immutable child source-binding receipt required")
    binding = {"schema": CHILD_SOURCE_BINDING_SCHEMA,
        "shard_id": shard_id, "partition": partition, "projection": projection_pin,
        "context": context_pin, "raw_seal": raw_pin, "exposure_audit": exposure_pin,
        "source_root": str(source_root), "source_files": value["source_files"],
        "adapter": value["adapter"], "supervision_sources": value["sharded_child"]["supervision_sources"],
        "source_install": value["source_install"], "python": value["python"],
        "runtime": value["runtime"], "config": value["config"],
        "feature_storage_root": value["feature_storage_root"],
        "retained_raw_roots": value["retained_raw_roots"],
        "neural_inputs": value["neural_inputs"], "quota_roots": value["quota_roots"],
        "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
        "initial_ui_context": value["ui_context"],
        "initial_execution_evidence": value["execution_evidence"],
        "resource_schema": "visualizeit_expanded_child_quota_v1",
        "paid_resources": False}
    for key in ("raw_continuation", "source_migration"):
        if key in value:
            binding[key] = value[key]
    if defer_label_plan:
        binding.update(label_binding_mode="deferred_v1",
            label_plan_path=value["sharded_child"]["label_plan_path"])
    binding = _sealed(binding, "source_binding_sha256")
    p.atomic_json(binding_path, binding)
    value["sharded_child"]["source_binding"] = p.pin(binding_path)
    value["stage_evidence_history"] = [{"stage": "identity",
        "ui_context": value["ui_context"], "execution_evidence": value["execution_evidence"]}]
    value["stage_request_identity_sha256"] = _child_stage_request_identity(value)
    return value


def _deferred_label_slot(request):
    """Validate a prospective path without touching plan or generation files."""
    child = request.get("sharded_child", {})
    mode = child.get("label_binding_mode")
    if mode is None:
        if "label_plan_path" in child or "child_label_binding" in request:
            raise ValueError("late child labels require their explicit deferred mode")
        return None
    if mode != "deferred_v1":
        raise ValueError("unknown child label binding mode")
    slot = Path(request["control_root"]) / "child-label-plan.json"
    if not slot.is_absolute() or ".." in slot.parts or child.get("label_plan_path") != str(slot):
        raise ValueError("deferred label plan must use its frozen control-root slot")
    if ("label_plan" in request) != ("child_label_binding" in request):
        raise ValueError("deferred child plan and binding must be attached together")
    return slot


def _check_child_layout(request):
    _deferred_label_slot(request)
    work = p.no_links(Path(request["working_root"]).absolute())
    execution = p.no_links(Path(request["execution_root"]).absolute())
    features = p.no_links(Path(request["features_root"]).absolute())
    feature_storage = p.no_links(Path(request["feature_storage_root"]).absolute())
    control = p.no_links(Path(request["control_root"]).absolute())
    if len({work, execution, features, control}) != 4:
        raise ValueError("child work, execution, feature and control roots must be distinct")
    if work == Path("/") or not work.is_relative_to(Path("/kaggle/working")):
        raise ValueError("child work root must be a dedicated Kaggle working directory")
    if execution == work or not execution.is_relative_to(work):
        raise ValueError("child execution root must be a dedicated working child")
    if control == work or not control.is_relative_to(work):
        raise ValueError("child control root must be a dedicated working child")
    if (features.is_relative_to(work) or _root_device(features) == _root_device(work)
            or feature_storage in (Path("/"), Path("/tmp"))
            or not features.is_relative_to(feature_storage)
            or feature_storage.is_relative_to(work)
            or _root_device(feature_storage) != _root_device(features)):
        raise ValueError("child feature cache must use a separate filesystem, normally /tmp")
    if not features.is_absolute():
        raise ValueError("child feature cache must be absolute")
    return features


def _child_roots(request, adapter):
    context = raw.read_context(Path(request["context"]["path"]), request["context"]["sha256"])
    raw_doc = context.document()
    runtime = p.metadata(request["runtime"])
    expected = {Path(request[key]).absolute() for key in (
        "working_root", "source_root", "source_install", "execution_root", "features_root", "feature_storage_root", "control_root")}
    expected.update((Path(raw_doc["output_root"]).absolute(), Path(raw_doc["scratch_root"]).absolute(),
                     Path(raw_doc["model_cache"]).absolute(), Path(runtime["runtime_root"]).absolute(),
                     (Path(request["source_root"]) / ".cache/model-quality").absolute(),
                     (Path(request["source_install"]) / adapter.ROOT18 / ".cache/model-quality").absolute(),
                     Path(request["ui_context"]["path"]).parent.absolute(),
                     Path(request["execution_evidence"]["path"]).parent.absolute(),
                     Path(request["exposure_audit"]["path"]).parent.absolute(),
                     (_deferred_label_slot(request) or Path(request["label_plan"]["path"])).parent.absolute()))
    retained = request.get("retained_raw_roots")
    if not isinstance(retained, list) or not retained or any(not isinstance(value, str) for value in retained):
        raise ValueError("complete retained sibling RAW-root roster required")
    current_raw = {str(Path(raw_doc[k]).absolute()) for k in ("output_root", "scratch_root", "model_cache")}
    if not current_raw.issubset(set(retained)):
        raise ValueError("retained RAW-root roster omits current child roots")
    expected.update(Path(value).absolute() for value in retained)
    roots = {Path(value).absolute() for value in request["quota_roots"]}
    if roots != expected:
        raise ValueError("child quota roots must include every source/runtime/RAW/scratch/cache/feature/control/support root; missing="
                         + repr(sorted(map(str, expected - roots))) + " extra=" + repr(sorted(map(str, roots - expected))))
    for root in roots:
        p.no_links(root)
    return tuple(sorted(roots, key=str))


def child_quota_check(request, adapter, *, stage, active_states=None, live=False):
    """Versioned child quota with 32-GiB combined residency and per-device writes."""
    if (request.get("schema") != "kaggle_prepared_request_v2"
            or request.get("storage_limit_bytes") != AGGREGATE_STORAGE_LIMIT
            or request.get("sharded_child", {}).get("schema") != CHILD_REQUEST_VERSION):
        raise ValueError("explicit 32-GiB sharded child request required")
    limits = adapter.validate_limits(request["limits"])
    _validate_sharded_limits(limits)
    roots = _child_roots(request, adapter)
    runtime = p.metadata(request["runtime"])
    links = adapter.runtime_link_policy(runtime, Path(request["working_root"]))
    caches = (Path(request["source_root"]) / ".cache/model-quality",
              Path(request["source_install"]) / adapter.ROOT18 / ".cache/model-quality")
    from scripts.kaggle_supervision_v1 import stable_tree_bytes
    cache_bytes = stable_tree_bytes(adapter, caches)
    if cache_bytes > 8 * GIB:
        raise ValueError("child combined inode-aware model caches exceed 8 GiB")
    resident = stable_tree_bytes(adapter, roots, approved_links=links)
    free = adapter.free_by_filesystem(roots)
    if resident > AGGREGATE_STORAGE_LIMIT:
        raise ValueError("child resident roots exceed the prospective 32-GiB limit")
    work_device = _root_device(request["working_root"])
    feature_device = _root_device(request["features_root"])
    cap = used = 0
    output_device = None
    if stage in ("extract", "labels"):
        states = active_states
        if states is None:
            partition = sharded.verify_partition(request["sharded_child"]["partition"], **_migration_kwargs(request))
            projection_rows = partition["children"][request["sharded_child"]["shard_id"]]["projection_rows"]
            states = sum(row["split"] in ("train", "validation") for row in projection_rows)
        if type(states) is not int or states < 1 or states > 120:
            raise ValueError("child active state count must be frozen and bounded to 1..120")
        cap = states * p.PER_STATE_ENVELOPE + p.TEMP_ENVELOPE
        if live or stage == "labels" or Path(request["features_root"]).exists():
            used = stable_tree_bytes(adapter, (Path(request["features_root"]),), approved_links=links)
    elif stage in ("train", "resume", "export", "pilot"):
        output_device = _root_device(request["output_root"])
    allocation_stage = "child-labels" if stage == "labels" else stage
    allocations = _stage_write_allocations(allocation_stage, working_device=work_device,
        feature_device=feature_device, output_device=output_device, cap=cap, existing=used)
    _enforce_storage_reserve(resident, allocations, free, limits["reserve_bytes"], AGGREGATE_STORAGE_LIMIT)
    return {"schema": "visualizeit_expanded_child_quota_v1", "stage": stage,
        "resident_bytes": resident, "cache_bytes": cache_bytes,
        "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
        "writes_by_device": {str(k): v for k, v in allocations.items()},
        "free_by_device": {str(k): v for k, v in free.items()},
        "reserve_bytes": limits["reserve_bytes"], "quota_roots": list(map(str, roots)),
        "inode_aware": True, "runtime_links_authenticated": True}


def _child_stage_limits(request, stage):
    limits = dict(request["limits"])
    if stage in ("extract", "labels"):
        partition = sharded.verify_partition(request["sharded_child"]["partition"], **_migration_kwargs(request))
        rows = partition["children"][request["sharded_child"]["shard_id"]]["projection_rows"]
        active = sum(row["split"] in ("train", "validation") for row in rows)
        limits["max_output_bytes"] = min(limits["max_output_bytes"],
            active * p.PER_STATE_ENVELOPE + p.TEMP_ENVELOPE)
    elif stage == "assemble":
        from scripts.kaggle_supervision_v1 import stable_tree_bytes
        adapter = p.load_source(request["adapter"], "_sharded_child_assemble_limits_adapter",
            expected=p.ADAPTER_SHA256)
        links = adapter.runtime_link_policy(p.metadata(request["runtime"]), Path(request["working_root"]))
        existing = stable_tree_bytes(adapter, (Path(request["features_root"]),), approved_links=links)
        limits["max_output_bytes"] = min(limits["max_output_bytes"], existing + MAX_CONTROL_WRITE)
    return limits


def _child_supervision_provenance(request):
    root = Path(request["source_root"])
    sources = _child_supervision_sources(root)
    if request.get("sharded_child", {}).get("supervision_sources") != sources:
        raise ValueError("child supervision source closure changed")
    for name, pin in sources.items():
        if p.verify_pin(pin) != root / name:
            raise ValueError("child supervision source does not resolve to current source root")
    source_binding_pin = request.get("sharded_child", {}).get("source_binding")
    if not isinstance(source_binding_pin, dict):
        raise ValueError("immutable child source-binding receipt required")
    source_binding = p.metadata(source_binding_pin)
    _check_seal(source_binding, "source_binding_sha256", CHILD_SOURCE_BINDING_SCHEMA)
    slot = _deferred_label_slot(request)
    if (source_binding.get("label_binding_mode") != request["sharded_child"].get("label_binding_mode")
            or source_binding.get("label_plan_path") != (str(slot) if slot else None)):
        raise ValueError("child deferred-label source binding changed")
    fixed = {"shard_id": request["sharded_child"].get("shard_id"),
        "partition": request["sharded_child"].get("partition"),
        "projection": request["sharded_child"].get("projection"),
        "context": request.get("context"), "raw_seal": request.get("raw_seal"),
        "exposure_audit": request.get("exposure_audit"),
        "source_root": request.get("source_root"), "source_files": request.get("source_files"),
        "adapter": request.get("adapter"), "supervision_sources": sources,
        "source_install": request.get("source_install"), "python": request.get("python"),
        "runtime": request.get("runtime"), "config": request.get("config"),
        "feature_storage_root": request.get("feature_storage_root"),
        "retained_raw_roots": request.get("retained_raw_roots"),
        "neural_inputs": request.get("neural_inputs"), "quota_roots": request.get("quota_roots"),
        "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
        "resource_schema": "visualizeit_expanded_child_quota_v1", "paid_resources": False}
    for key in ("raw_continuation", "source_migration"):
        if key in request or key in source_binding:
            fixed[key] = request.get(key)
    if any(source_binding.get(key) != value for key, value in fixed.items()):
        raise ValueError("immutable child source-binding differs from request/source pins")
    return {"version": CHILD_SUPERVISION_SCHEMA, "sources": sources,
        "validation_adapter": request["adapter"],
        "supervisor": "scripts.kaggle_supervision_v1.Facade", "output_scanner": "raw._usage",
        "resource_schema": "visualizeit_expanded_child_quota_v1",
        "storage_limit_bytes": AGGREGATE_STORAGE_LIMIT,
        "source_binding": source_binding_pin,
        "source_binding_sha256": source_binding["source_binding_sha256"]}


def _child_stage_request_identity(request):
    """Hash the fixed child execution semantics, independent of fresh observations."""
    child = request.get("sharded_child", {})
    immutable = {key: request.get(key) for key in (
        "schema", "source_root", "source_files", "adapter", "source_install", "python",
        "runtime", "config", "input_root", "working_root", "execution_root", "features_root",
        "feature_storage_root", "retained_raw_roots", "control_root", "context", "raw_seal",
        "exposure_audit", "label_plan", "neural_inputs",
        "limits", "storage_limit_bytes", "quota_roots", "profile")}
    immutable["sharded_child"] = {key: child.get(key) for key in (
        "schema", "partition", "shard_id", "projection", "storage_limit_bytes",
        "supervision_sources", "source_binding")}
    for key in ("raw_continuation", "source_migration"):
        if key in request:
            immutable[key] = request[key]
    slot = _deferred_label_slot(request)
    if slot is not None:
        # The input identity freezes the slot, not labels created after sealing.
        # Legacy requests retain their original canonical material exactly.
        immutable.pop("label_plan")
        immutable["sharded_child"].update(label_binding_mode="deferred_v1",
            label_plan_path=str(slot))
    return _canonical(immutable)


def _provider_observation(ui_pin, *, require_fresh=True):
    ui = p.metadata(ui_pin)
    if (not isinstance(ui, dict) or ui.get("provider") not in (None, "kaggle")
            or ui.get("evidence_basis") != "actual Kaggle UI"
            or ui.get("paid") is not False or ui.get("gpu_enabled") is not True
            or not isinstance(ui.get("session_label"), str) or not ui["session_label"]
            or type(ui.get("gpu_remaining_seconds")) is not int
            or type(ui.get("session_remaining_seconds")) is not int
            or min(ui["gpu_remaining_seconds"], ui["session_remaining_seconds"]) <= 0):
        raise ValueError("fresh actual Kaggle UI observation required")
    try:
        observed = datetime.fromisoformat(ui["observed_at"].replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("provider UI observation timestamp is invalid") from exc
    now = datetime.now(timezone.utc).timestamp()
    if observed > now or (require_fresh and observed < now - 3600):
        raise ValueError("provider UI observation is stale or future-dated")
    return ui, observed


def _validate_child_evidence_metadata(request, evidence_pin, ui_pin=None, *, require_fresh=True):
    ui_pin = request.get("ui_context") if ui_pin is None else ui_pin
    evidence = p.metadata(evidence_pin)
    quota = evidence.get("quota", {})
    ui, ui_observed = _provider_observation(ui_pin, require_fresh=require_fresh)
    try:
        observed = datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00")).timestamp()
        expires = datetime.fromisoformat(evidence["expires_at"].replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("child preparation evidence timestamp is invalid") from exc
    now = datetime.now(timezone.utc).timestamp()
    gpu_remaining = quota.get("gpu_remaining_seconds")
    session_remaining = quota.get("session_remaining_seconds")
    if (type(gpu_remaining) is not int or type(session_remaining) is not int
            or evidence.get("schema") != "kaggle_preparation_evidence_v2"
            or evidence.get("provider") != "kaggle"
            or evidence.get("raw_seal") != request.get("raw_seal")
            or evidence.get("context") != request.get("context")
            or evidence.get("source_files") != request.get("source_files")
            or evidence.get("provider_ui_observation") != ui_pin
            or evidence.get("session_id") != ui.get("session_label")
            or observed > now or expires <= observed or expires > observed + 3600
            or (require_fresh and not now - 3600 <= observed <= now < expires)
            or observed < ui_observed or observed - ui_observed > 300
            or quota.get("paid") is not False or quota.get("gpu_enabled") is not True
            or quota.get("storage_scope") != request.get("quota_roots")
            or quota.get("storage_limit_bytes") != AGGREGATE_STORAGE_LIMIT
            or gpu_remaining > ui.get("gpu_remaining_seconds", 0)
            or session_remaining > ui.get("session_remaining_seconds", 0)
            or (require_fresh and min(gpu_remaining, session_remaining)
                < request["limits"]["max_seconds"])):
        raise ValueError("child evidence must truthfully declare original-format Kaggle preparation at 32 GiB")
    gpu = evidence.get("gpu")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("uuid"), str) or not isinstance(gpu.get("name"), str):
        raise ValueError("child preparation evidence must pin the selected GPU identity")
    return evidence


def _verify_child_stage_history(request, *, require_fresh_latest=True):
    history = request.get("stage_evidence_history")
    identity = _child_stage_request_identity(request)
    if (request.get("stage_request_identity_sha256") != identity
            or not isinstance(history, list) or not history
            or history[0].get("stage") != "identity"):
        raise ValueError("child immutable stage identity/history is missing or changed")
    source_binding = p.metadata(request["sharded_child"]["source_binding"])
    _check_seal(source_binding, "source_binding_sha256", CHILD_SOURCE_BINDING_SCHEMA)
    if (history[0].get("ui_context") != source_binding.get("initial_ui_context")
            or history[0].get("execution_evidence") != source_binding.get("initial_execution_evidence")):
        raise ValueError("child evidence history dropped or changed its immutable initial observation")
    prior = None
    latest_evidence = latest_ui = None
    allowed = {name: i for i, name in enumerate(CHILD_STAGE_NAMES)}
    stage_indexes = []
    now = datetime.now(timezone.utc).timestamp()
    for entry in history:
        if (not isinstance(entry, dict) or set(entry) != {"stage", "ui_context", "execution_evidence"}
                or entry.get("stage") not in allowed):
            raise ValueError("invalid child stage evidence history entry")
        stage_indexes.append(allowed[entry["stage"]])
        evidence = _validate_child_evidence_metadata(request, entry["execution_evidence"],
            entry["ui_context"], require_fresh=False)
        ui, ui_time = _provider_observation(entry["ui_context"], require_fresh=False)
        observed = datetime.fromisoformat(evidence["observed_at"].replace("Z", "+00:00")).timestamp()
        quota = evidence["quota"]
        gpu = evidence["gpu"]
        current = (observed, evidence["session_id"], (gpu["uuid"], gpu["name"]),
                   quota["gpu_remaining_seconds"], quota["session_remaining_seconds"])
        if observed > now or ui_time > now:
            raise ValueError("child evidence history contains a future-dated prior observation")
        if prior is not None and (current[0] < prior[0] or current[1:3] != prior[1:3]
                or current[3] > prior[3] or current[4] > prior[4]):
            raise ValueError("child stage refresh reset its clock, session, GPU or remaining quota")
        prior, latest_evidence, latest_ui = current, entry["execution_evidence"], entry["ui_context"]
    if stage_indexes != list(range(len(stage_indexes))):
        raise ValueError("child stage evidence history must complete each preparation stage in order")
    if latest_evidence != request.get("execution_evidence") or latest_ui != request.get("ui_context"):
        raise ValueError("latest child stage evidence differs from current request observations")
    if require_fresh_latest:
        _validate_child_evidence_metadata(request, latest_evidence, latest_ui, require_fresh=True)
    return history


def child_stage_request(request, *, stage, ui_context, execution_evidence):
    """Refresh only truthful UI evidence while preserving prior pins and limits."""
    value = dict(request)
    if value.get("sharded_child", {}).get("schema") != CHILD_REQUEST_VERSION:
        raise ValueError("versioned child request required")
    if stage not in CHILD_STAGE_NAMES or stage == "identity":
        raise ValueError("child evidence refresh must name a later preparation stage")
    _verify_child_stage_history(value, require_fresh_latest=False)
    previous_stage = value["stage_evidence_history"][-1]["stage"]
    if CHILD_STAGE_NAMES.index(stage) != CHILD_STAGE_NAMES.index(previous_stage) + 1:
        raise ValueError("child evidence refresh must advance to the next preparation stage")
    if stage == "labels" and _deferred_label_slot(value) is not None:
        _verify_child_label_binding(value)
    value["ui_context"] = _as_pin(ui_context)
    value["execution_evidence"] = _as_pin(execution_evidence)
    _validate_child_evidence_metadata(value, value["execution_evidence"], value["ui_context"])
    value["stage_evidence_history"] = list(value["stage_evidence_history"]) + [{
        "stage": stage, "ui_context": value["ui_context"],
        "execution_evidence": value["execution_evidence"]}]
    _verify_child_stage_history(value)
    _child_supervision_provenance(value)
    return value


def _child_label_binding_material(request):
    execution, features = Path(request["execution_root"]), Path(request["features_root"])
    child = request["sharded_child"]
    return dict(schema="visualizeit_deferred_child_labels_v1",
        stage_request_identity_sha256=request["stage_request_identity_sha256"],
        source_binding=child["source_binding"], partition=child["partition"],
        projection=child["projection"], shard_id=child["shard_id"],
        context=request["context"], raw_seal=request["raw_seal"],
        label_plan_path=child["label_plan_path"], label_plan=request["label_plan"],
        extraction_binding=p.pin(features / "extraction-binding.json"),
        prelabel_seal=p.pin(execution / "prelabel-seal.json"),
        seal_stage_observation=p.pin(execution / "seal_features-stage-observation.json"),
        labels_entered_inference=False, test_payload_reads=0)


def _verify_child_label_binding(request, *, receipt=None, verify_plan=False):
    """Metadata-first lineage checks; plan bytes are optional and post-numerical."""
    slot = _deferred_label_slot(request)
    if slot is None:
        return None
    if "label_plan" not in request:
        raise ValueError("deferred child label plan has not been bound after sealing")
    binding_pin = request["child_label_binding"]
    if (p.descriptor_path(binding_pin) != Path(request["control_root"]) / "child-label-binding.json"
            or p.descriptor_path(request["label_plan"]) != slot):
        raise ValueError("deferred plan/binding path differs from its frozen slot")
    binding = p.metadata(binding_pin)
    _check_seal(binding, "binding_sha256", "visualizeit_deferred_child_labels_v1")
    if binding != _sealed(_child_label_binding_material(request), "binding_sha256"):
        raise ValueError("deferred child plan/prelabel/source lineage changed")
    if receipt is not None and (receipt.get("label_plan") != request["label_plan"]
            or receipt.get("child_label_binding") != binding_pin):
        raise ValueError("child label receipt differs from its late plan binding")
    if verify_plan:
        p.verify_pin(request["label_plan"])
    return binding


def bind_child_label_plan(request):
    """Materialize generation descriptors only after the real numerical seal."""
    slot = _deferred_label_slot(request)
    if slot is None or "label_plan" in request:
        raise ValueError("one fresh explicit deferred child label attachment required")
    _child_supervision_provenance(request)
    history = _verify_child_stage_history(request, require_fresh_latest=False)
    if [row["stage"] for row in history] != list(CHILD_STAGE_NAMES[:3]):
        raise ValueError("child labels require completed input-stage evidence through seal-features")
    _, state = _child_raw(request, payloads=True)
    execution, features = Path(request["execution_root"]), Path(request["features_root"])
    _verify_child_supervision(request, p.read_json(features / "extraction-binding.json"),
        request["sharded_child"]["shard_id"])
    # This numerical check MUST precede even stat/hash of plan/generation paths.
    prelabel = p.verify_prelabel(state, execution, features)
    observation = p.read_json(execution / "seal_features-stage-observation.json")
    if (observation.get("schema") != "visualizeit_expanded_child_stage_observation_v1"
            or observation.get("stage") != "seal-features"
            or observation.get("stage_request_identity_sha256") != request["stage_request_identity_sha256"]
            or observation.get("stage_evidence_history") != history
            or observation.get("ui_context") != history[-1]["ui_context"]
            or observation.get("execution_evidence") != history[-1]["execution_evidence"]
            or observation.get("result") != prelabel
            or observation.get("test_payload_reads") != 0):
        raise ValueError("actual successful child numerical seal observation required")
    binding_path = Path(request["control_root"]) / "child-label-binding.json"
    if slot.exists() or binding_path.exists():
        raise FileExistsError("preserve earlier child label-plan/binding files")
    generations = []
    for row in state["physical"]:
        if row["split"] in ("train", "validation") and row["render"]["status"] == "available":
            bundle = p.descriptor_path(row["render"]["input"]).parent
            generations.append(dict(sequence_id=row["sequence_id"],
                generation=p.pin(bundle.parent / "generation.json")))
    p.verify_prelabel(state, execution, features)
    p.atomic_json(slot, dict(schema="kaggle_label_plan_v2",
        split_manifest=p.pin(state["root"] / "split.json"), generations=generations))
    value = dict(request, label_plan=p.pin(slot))
    p.atomic_json(binding_path, _sealed(_child_label_binding_material(value), "binding_sha256"))
    value["child_label_binding"] = p.pin(binding_path)
    _verify_child_label_binding(value, verify_plan=True)
    if _child_stage_request_identity(value) != request["stage_request_identity_sha256"]:
        raise ValueError("late labels changed the frozen input-stage identity")
    return value


def verify_child_request(request, *, payloads=False):
    """Verify a finalized original-format child plus its distinct 32-GiB bridge."""
    if request.get("schema") != "kaggle_prepared_request_v2":
        raise ValueError("original prepared-v2 child request required")
    prepared.verify_wrapper_sources(request)
    sources = request.get("sharded_child", {}).get("supervision_sources")
    if sources != _child_supervision_sources(request["source_root"]):
        raise ValueError("explicit sharded child/source closure required")
    for name, pin in sources.items():
        if p.verify_pin(pin) != Path(request["source_root"]) / name:
            raise ValueError("sharded child source pin differs: " + name)
    _check_child_layout(request)
    child = request["sharded_child"]
    partition = sharded.verify_partition(child["partition"], **_migration_kwargs(request))
    sid = child["shard_id"]
    sharded.verify_shard(child["partition"], sid, request["context"], request["raw_seal"], child["projection"], payloads=False,
        **_raw_join_kwargs(request))
    if child.get("supervision_sources") != _child_supervision_sources(request["source_root"]):
        raise ValueError("child supervision bridge pins changed")
    history = _verify_child_stage_history(request, require_fresh_latest=False)
    if [entry["stage"] for entry in history] != list(CHILD_STAGE_NAMES):
        raise ValueError("final child request requires identity, extraction, numerical seal and label evidence")
    joined = prepared.verify_prepared(request, payloads=payloads)
    _verify_child_label_binding(request, receipt=joined["docs"]["label_receipt"], verify_plan=payloads)
    provenance = _verify_child_supervision(request, joined["docs"]["extraction_binding"], sid)
    if provenance != _child_supervision_provenance(request):
        raise ValueError("child execution/32-GiB supervision provenance differs")
    return {**joined, "shard_id": sid, "partition": partition,
        "projection": sharded.verify_shard(child["partition"], sid, request["context"],
            request["raw_seal"], child["projection"], payloads=False, **_raw_join_kwargs(request))["projection"]}


def finalize_child_request(request):
    """Add authenticated original-format artifact pins after the child seal exists."""
    value = dict(request)
    if _deferred_label_slot(value) is not None:
        _verify_child_label_binding(value, receipt=p.read_json(Path(value["features_root"]) / "label-receipt.json"))
    features, execution = Path(value["features_root"]), Path(value["execution_root"])
    value.update({
        "execution_plan": p.pin(execution / "execution-plan.json"),
        "identity_bridge": p.pin(execution / "identity-bridge.json"),
        "prelabel_seal": p.pin(execution / "prelabel-seal.json"),
        "extraction_binding": p.pin(features / "extraction-binding.json"),
        "original_recipe": p.pin(features / "original-recipe.json"),
        "original_index": p.pin(features / "original-index.json"),
        "feature_index": p.pin(features / "index.json"),
        "labels": p.pin(features / "labels.json"),
        "label_receipt": p.pin(features / "label-receipt.json"),
        "dataset": p.pin(features / "manifest.json"),
    })
    for stage in CHILD_STAGE_NAMES:
        target = execution / (stage.replace("-", "_") + "-stage-observation.json")
        if target.is_file():
            value[stage.replace("-", "_") + "_stage_observation"] = p.pin(target)
    return value


def _child_raw(request, *, payloads):
    prepared.verify_wrapper_sources(request)
    sharded_child = request.get("sharded_child", {})
    if sharded_child.get("schema") != CHILD_REQUEST_VERSION:
        raise ValueError("versioned sharded child projection required")
    partition_pin, sid = sharded_child["partition"], sharded_child["shard_id"]
    if sharded_child.get("storage_limit_bytes") != AGGREGATE_STORAGE_LIMIT:
        raise ValueError("child-specific 32-GiB resource descriptor changed")
    proof = sharded.verify_shard(partition_pin, sid, request["context"], request["raw_seal"],
        sharded_child["projection"], payloads=False, **_raw_join_kwargs(request))
    _, state = p.prepared_raw_state(request["context"], request["raw_seal"],
        exposure_audit=request["exposure_audit"], verify_payloads=payloads, **_raw_join_kwargs(request))
    if proof["raw_state"]["seal"] != state["seal"]:
        raise ValueError("child original RAW verifier and frozen projection differ")
    return proof, state


def _child_stage_admit(request, stage):
    if (_deferred_label_slot(request) is not None and stage in CHILD_STAGE_NAMES[:3]
            and "label_plan" in request):
        raise ValueError("input-only child stages cannot consume attached labels")
    p.cloud_guard(request)
    prepared.verify_wrapper_sources(request)
    k = p.load_source(request["adapter"], "_sharded_child_preflight_adapter", expected=p.ADAPTER_SHA256)
    quota = child_quota_check(request, k, stage=stage,
        live=(stage == "labels"))
    if stage in CHILD_STAGE_NAMES:
        history = _verify_child_stage_history(request)
        if history[-1]["stage"] != stage:
            raise ValueError("child stage requires its own current provider/UI evidence refresh")
        evidence = p.preparation_evidence(request)
    else:
        evidence = p.preparation_evidence(request)
    p.source_runtime(request)
    return k, evidence, quota


def _supervision_facade(request, adapter):
    """Keep the immutable scanner under explicit predecessor source proof."""
    from scripts import kaggle_supervision_v1 as supervision
    scanner = raw
    if p.sha256(Path(raw.__file__)) != supervision.SCANNER_SHA256:
        migration_pin = request.get("source_migration")
        with p.source_migration_scope(migration_pin):
            p.verify_prepared_source_digest(request["source_root"],
                "bench/quality_adaptation_portable.py", supervision.SCANNER_SHA256)
            # The verifier above authenticates original, archive and executing
            # successor. Load the historical scanner bytes, not successor code
            # carrying an obsolete immutable-scanner claim.
            migration = p.metadata(p.active_source_migration())
            archive = migration["sources"]["bench/quality_adaptation_portable.py"]["archive"]
            scanner = p.load_source(archive, "bench._sharded_preserved_resource_scanner",
                expected=supervision.SCANNER_SHA256)
    return supervision.Facade(adapter, scanner)


def _run_child_original(request, *, stage, argv, output_root, log_path, cuda=False, env_extra=None):
    p.cloud_guard(request)
    prepared.verify_wrapper_sources(request)
    k = p.load_source(request["adapter"], "_sharded_child_run_adapter", expected=p.ADAPTER_SHA256)
    before = child_quota_check(request, k, stage=stage)
    evidence = p.preparation_evidence(request)
    k, root18, runtime = p.source_runtime(request)
    env = k.clean_environment()
    env["CUDA_VISIBLE_DEVICES"] = evidence["gpu"]["uuid"]
    if env_extra:
        env.update(env_extra)
    if cuda:
        from bench.quality_adaptation_larger import _CUDA_BOOTSTRAP
        env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",
            VISUALIZEIT_CUDA_BOOTSTRAP_SHA256=hashlib.sha256(_CUDA_BOOTSTRAP.encode()).hexdigest(),
            VISUALIZEIT_CUDA_RUNTIME_RECEIPT=str(Path(output_root) / "device-runtime.json"))
        argv = _cuda_bootstrap_argv(request, argv, _CUDA_BOOTSTRAP)
    roots = _child_roots(request, k)
    links = k.runtime_link_policy(p.metadata(request["runtime"]), Path(request["working_root"]))
    from scripts import kaggle_supervision_v1 as supervision
    facade = _supervision_facade(request, k)
    result, watch = _supervise_with_quota_watch(facade, argv=argv, cwd=root18,
        output_root=Path(output_root), working=Path(request["working_root"]),
        log_path=Path(log_path), limits=k.validate_limits(_child_stage_limits(request, stage)),
        env=env, sample=lambda: child_quota_check(request, k, stage=stage, live=True),
        initial_resident=before["resident_bytes"], thread_name="sharded-child-quota-watch")
    final_quota, final_quota_error = None, None
    try:
        final_quota = child_quota_check(request, k, stage=stage, live=True)
    except BaseException as exc:
        final_quota_error = f"{type(exc).__name__}: {exc}"
    result["expanded_child_resource_watch"] = {**watch,
        "quota": final_quota, "final_quota_error": final_quota_error}
    watch_errors = list(watch["monitor_errors"])
    if final_quota_error:
        watch_errors.append(final_quota_error)
    if watch_errors:
        result["resource_watch_failure"] = watch_errors
        if result.get("status") == "completed":
            result.update(status="incomplete", reason="child_storage_watch_failed", outputs_preserved=True)
    return result, runtime


def child_prepare_identity(request):
    _child_stage_admit(request, "identity")
    proof, state = _child_raw(request, payloads=False)
    p.bounded_recipe(state["recipe"], p.metadata(request["exposure_audit"]))
    result = p.execution_identity(state, Path(request["execution_root"]))
    _write_child_stage_observation(request, "identity", result)
    return result


def _write_child_stage_observation(request, stage, result):
    observation = {"schema": "visualizeit_expanded_child_stage_observation_v1",
        "stage": stage, "stage_request_identity_sha256": request["stage_request_identity_sha256"],
        "ui_context": request["ui_context"], "execution_evidence": request["execution_evidence"],
        "stage_evidence_history": request["stage_evidence_history"],
        "result": result, "paid_resources": False, "quality_claim": False,
        "test_payload_reads": 0}
    target = Path(request["execution_root"]) / (stage.replace("-", "_") + "-stage-observation.json")
    p.atomic_json(target, observation)
    return p.pin(target)


def child_extract(request):
    _child_stage_admit(request, "extract")
    proof, state = _child_raw(request, payloads=True)
    execution = Path(request["execution_root"])
    plan = p.verify_execution(state, execution)
    features = p.no_links(Path(request["features_root"]).absolute())
    if features.exists():
        raise FileExistsError("child extractor requires a fresh feature cache")
    manifest = p.crop_manifest(state, execution)
    active = sum(row["split"] in ("train", "validation") for row in state["captures"])
    envelope = active * p.PER_STATE_ENVELOPE + p.TEMP_ENVELOPE
    if active > 120 or envelope > min(p.FEATURE_TREE_CAP, request["limits"]["max_output_bytes"]):
        raise ValueError("child original feature/label envelope exceeds bounded cache")
    features.parent.mkdir(parents=True, exist_ok=True)
    training = {"python": request["python"], "paths": {"extraction_dir": str(features)},
        "limits": request["limits"], "extraction": {"max_seconds": int(request["limits"]["max_seconds"]),
            "max_output_bytes": p.FEATURE_TREE_CAP, "max_rows": active}, "crop_manifest": manifest}
    adapter = p.load_source(request["adapter"], "_sharded_child_adapter", expected=p.ADAPTER_SHA256)
    argv = adapter.command("extract", training)
    observed, runtime = _run_child_original(request, stage="extract", argv=argv,
        output_root=features, log_path=execution / "extract.log", cuda=True)
    p.atomic_json(execution / "extract-observation.json", observed)
    if observed.get("status") != "completed" or observed.get("returncode") != 0:
        return observed
    for name in ("recipe.json", "index.json"):
        if not (features / name).is_file():
            raise ValueError("original child extractor omitted " + name)
        shutil.copyfile(features / name, features / ("original-" + name))
    recipe = p.read_json(features / "original-recipe.json")
    original = p.read_json(features / "original-index.json")
    actual = p.indexed(original["rows"], "original child extractor")
    required = {row["id"] for row in state["captures"] if row["split"] in ("train", "validation") and row["status"] == "available"}
    if set(actual) != required:
        raise ValueError("unchanged child extractor omitted an available crop")
    final_rows = [actual.get(row["id"], {"id": row["id"], "status": "unavailable",
        "failure_reason": row.get("failure_reason") or "capture_unavailable"})
        for row in state["captures"] if row["split"] != "dev"]
    final = {**original, "rows": final_rows}
    p.atomic_json(features / ".final-index.json", final)
    os.replace(features / ".final-index.json", features / "index.json")
    binding = {"schema": "kaggle_extraction_binding_v2",
        "original_recipe": p.pin(features / "original-recipe.json"),
        "original_index": p.pin(features / "original-index.json"),
        "final_recipe": p.pin(features / "recipe.json"), "final_index": p.pin(features / "index.json"),
        "crop_manifest": manifest, "execution_plan": p.pin(execution / "execution-plan.json"),
        "raw_seal": request["raw_seal"],
        "unavailable_added": [row["id"] for row in final_rows if row["id"] not in actual],
        "dev_rows": [row for row in state["captures"] if row["split"] == "dev"],
        "successful_original_rows_unchanged": True,
        "sharded_partition_sha256": proof["projection"]["partition_sha256"],
        "sharded_projection": request["sharded_child"]["projection"],
        "source_files": request["source_files"], "runtime": runtime,
        "stage": "extract", "ui_context": request["ui_context"],
        "execution_evidence": request["execution_evidence"],
        "stage_evidence_history": request["stage_evidence_history"],
        "supervision": _child_supervision_provenance(request)}
    p.atomic_json(features / "extraction-binding.json", binding)
    observation = {"status": "completed", "requested_rows": active,
        "available_input_rows": len(required), "result": observed,
        "extraction_binding": p.pin(features / "extraction-binding.json"),
        "quality_claim": False}
    observation["stage_observation"] = _write_child_stage_observation(request, "extract", observation)
    return observation


def child_seal_features(request):
    _child_stage_admit(request, "seal-features")
    proof, state = _child_raw(request, payloads=True)
    binding = p.read_json(Path(request["features_root"]) / "extraction-binding.json")
    _verify_child_supervision(request, binding, request["sharded_child"]["shard_id"])
    if (binding.get("supervision") != _child_supervision_provenance(request)
            or binding.get("sharded_projection") != request["sharded_child"]["projection"]
            or binding.get("sharded_partition_sha256") != proof["projection"]["partition_sha256"]):
        raise ValueError("child extraction source/projection/supervision binding changed")
    if (binding.get("stage") != "extract"
            or binding.get("execution_evidence") not in [row["execution_evidence"] for row in request["stage_evidence_history"]]
            or binding.get("ui_context") not in [row["ui_context"] for row in request["stage_evidence_history"]]):
        raise ValueError("child extraction binding lacks exact evidence history")
    result = p.seal_features(state, Path(request["execution_root"]), Path(request["features_root"]))
    result["stage_observation"] = _write_child_stage_observation(request, "seal-features", result)
    return result


def _child_production_run(request, *, stage, argv, output_root, log_path):
    """Run the unchanged ROOT18 assembler under the new child quota watcher."""
    return _run_child_original(request, stage=stage, argv=argv, output_root=output_root,
        log_path=log_path, cuda=False)


def _child_labels(request, raw_state, execution):
    """Numerical child label loop copied from reviewed supervised-v1 labels()."""
    p.cloud_guard(request)
    features = Path(request["features_root"])
    prelabel = p.verify_prelabel(raw_state, execution, features)
    _verify_child_label_binding(request, verify_plan=True)
    provenance = _child_supervision_provenance(request)
    binding = p.read_json(features / "extraction-binding.json")
    _verify_child_supervision(request, binding, request["sharded_child"]["shard_id"])
    if binding.get("supervision") != provenance:
        raise ValueError("child extraction lacks current 32-GiB source/evidence binding")
    label_plan = metadata(request["label_plan"])
    if set(label_plan) != {"schema", "split_manifest", "generations"} or label_plan["schema"] != "kaggle_label_plan_v2":
        raise ValueError("invalid separate label plan")
    if verify_pin(label_plan["split_manifest"]).read_bytes() != (raw_state["root"] / "split.json").read_bytes():
        raise ValueError("label split bytes differ")
    generations = {r["sequence_id"]: r["generation"] for r in label_plan["generations"]}
    active = {r["sequence_id"] for r in raw_state["physical"] if r["split"] in ("train", "validation") and r["render"]["status"] == "available"}
    if len(generations) != len(label_plan["generations"]) or set(generations) != active:
        raise ValueError("generation metadata inventory incomplete/foreign/sealed")
    for value in generations.values():
        descriptor_path(value)
    from bench.quality_adaptation_data import CROP_KEYS, load_npz
    from bench.quality_adaptation_refinement_states import GEOMETRY_KEYS
    from bench.quality_adaptation_prepare import project_supervision
    from bench.quality_adaptation_smoke import render_query_labels
    from bench.quality_adaptation_larger import _validate_generation
    from bench.quality_adaptation_runtime import atomic_stream
    from bench.quality_gotrack import TexturedRenderer
    from bench.quality_runner import read_input
    from bench.quality_ranker_data import read_split_manifest
    import numpy as np
    split_path = verify_pin(label_plan["split_manifest"])
    split = read_split_manifest(split_path)
    k, _, _ = p.source_runtime(request)
    extracted = indexed(read_json(features / "index.json")["rows"], "features")
    physical = {r["sequence_id"]: r for r in raw_state["physical"]}
    inventory, outcomes = [], []
    started = time.monotonic()
    with p.preparation_lock(raw_state):
        from bench.quality_adaptation_larger import configure_torch, cleanup_device
        torch = configure_torch("cuda", deterministic=True)
        try:
            for ordinal, source in enumerate(raw_state["captures"]):
                if source["split"] == "dev":
                    continue
                row = dict(id=source["id"], object_id=source["object_id"], sequence_id=source["sequence_id"], capture_group=source["capture_group"],
                    split="val" if source["split"] == "validation" else source["split"], query_rgb_sha256=source.get("query_rgb_sha256"), status="unavailable",
                    failure_reason=source.get("failure_reason") or extracted[source["id"]].get("failure_reason") or "label_not_processed")
                if row["query_rgb_sha256"] is None:
                    row["unavailable_provenance"] = source.get("unavailable_provenance", "frozen state; no retained crop")
                inventory.append(row)
                renderer = None
                if source["split"] not in ("train", "validation") or source["status"] != "available" or extracted[source["id"]]["status"] != "available":
                    continue
                try:
                    if time.monotonic() - started >= request["limits"]["max_seconds"]:
                        raise TimeoutError("label deadline exhausted")
                    child_quota_check(request, k, stage="labels", live=True)
                    frame = physical[source["sequence_id"]]
                    render = frame["render"]
                    bundle = verify_pin(render["input"]).parent
                    manifest = read_input(bundle)
                    physical_projection = dict(frame, split=row["split"], **{key: render[key] for key in ("input", "video", "mesh")})
                    renderer = TexturedRenderer(bundle / manifest["asset"], manifest["object_id"], unlit=True, disable_multisampling=True, coordinate_mode="integer_centers_v1")
                    pose = _validate_generation(metadata(generations[source["sequence_id"]]), physical_projection, manifest, split, sha256(split_path), renderer.vertices_m)
                    crops = load_npz(verify_pin(source["crops"], raw_state["root"] / "raw"), CROP_KEYS)
                    geometry = load_npz(verify_pin(source["capture"], raw_state["root"] / "raw"), GEOMETRY_KEYS)
                    query_pose, depth, support = render_query_labels(renderer, object_id=manifest["object_id"], crop_k=crops["query_K"], crop_from_orig=geometry["crop_from_orig"], native_pose=pose)
                    target = project_supervision(template_xyz_m=crops["template_xyz_m"], template_mask=crops["template_mask"], query_K=crops["query_K"],
                        query_T_cam_from_object=query_pose, observed_depth_m=depth, observed_object_mask=support, depth_tolerance_m=.002)
                    name = f"labels-{ordinal:06d}.npz"
                    with atomic_stream(features, features / name, max_file_bytes=16 * MIB, max_output_bytes=FEATURE_TREE_CAP) as stream:
                        np.savez(stream, **target)
                    row.pop("failure_reason", None)
                    row.update(status="available", labels={**pin(features / name), "path": name})
                except Exception as exc:
                    row["failure_reason"] = f"label_failure: {type(exc).__name__}: {exc}"[:1024]
                finally:
                    if renderer is not None:
                        renderer.close()
                    outcomes.append(dict(id=source["id"], label_status=row["status"], failure_reason=row.get("failure_reason")))
        finally:
            cleanup_device("cuda", torch)
            seen = {r["id"] for r in inventory}
            for source in raw_state["captures"]:
                if source["split"] != "dev" and source["id"] not in seen:
                    row = dict(id=source["id"], object_id=source["object_id"], sequence_id=source["sequence_id"], capture_group=source["capture_group"],
                        split="val" if source["split"] == "validation" else source["split"], query_rgb_sha256=source.get("query_rgb_sha256"),
                        status="unavailable", failure_reason="label_stage_interrupted")
                    if row["query_rgb_sha256"] is None:
                        row["unavailable_provenance"] = "frozen state; no crop materialized"
                    inventory.append(row)
            atomic_json(features / "labels.json", dict(schema="correspondence_label_inventory_v1", label_origin="synthetic_renderer_training_only", rows=inventory))
    inventory_document = read_json(features / "labels.json")
    label_artifacts = _ordered_child_label_artifacts(request, execution, inventory_document,
        [dict(id=r["id"], **r["labels"]) for r in inventory if r["status"] == "available"])
    assembler = p.load_source(request["adapter"], "_sharded_child_assembler", expected=p.ADAPTER_SHA256)
    argv = assembler.command("assemble", {"python": request["python"], "limits": request["limits"], "paths": {
        "extraction_dir": str(features), "manifest_output": str(features / "manifest.json")},
        "labels": {"path": str(features / "labels.json")}})
    result, runtime = _child_production_run(request, stage="assemble", argv=argv,
        output_root=features, log_path=Path(execution) / "assemble.log")
    p.atomic_json(Path(execution) / "assemble-observation.json", result)
    if result.get("status") != "completed" or result.get("returncode") != 0:
        raise ValueError("unchanged child dataset assembler failed; all label outcomes were preserved")
    receipt = dict(schema="kaggle_label_receipt_v2", prelabel_seal=pin(Path(execution) / "prelabel-seal.json"),
        label_plan=request["label_plan"], inventory=pin(features / "labels.json"), dataset=pin(features / "manifest.json"),
        outcomes=outcomes, label_artifacts=label_artifacts,
        labels_entered_inference=False, test_payload_reads=0, independent_quality_claim=False,
        elapsed_seconds=time.monotonic() - started, parity_rows=len(prelabel["parity"]),
        assemble_result=result, assembler_runtime=runtime,
        supervision=provenance, sharded_child=request["sharded_child"],
        stage="labels", ui_context=request["ui_context"],
        execution_evidence=request["execution_evidence"],
        stage_evidence_history=request["stage_evidence_history"])
    if _deferred_label_slot(request) is not None:
        receipt["child_label_binding"] = request["child_label_binding"]
        _verify_child_label_binding(request, receipt=receipt, verify_plan=True)
    atomic_json(features / "label-receipt.json", receipt)
    _write_child_stage_observation(request, "labels", pin(features / "label-receipt.json"))
    return receipt


def child_labels(request):
    """Supervise the copied original label loop with the versioned 32-GiB gate."""
    _child_stage_admit(request, "labels")
    proof, state = _child_raw(request, payloads=True)
    execution, features = Path(request["execution_root"]), Path(request["features_root"])
    p.verify_prelabel(state, execution, features)
    _verify_child_label_binding(request, verify_plan=True)
    binding = p.read_json(features / "extraction-binding.json")
    if binding.get("supervision") != _child_supervision_provenance(request):
        raise ValueError("child label execution/source binding changed")
    # Only after the numerical prelabel seal passes may the separate label plan be read.
    p.metadata(request["label_plan"])
    p.cloud_guard(request)
    prepared.verify_wrapper_sources(request)
    k, _, _ = p.source_runtime(request)
    evidence = p.preparation_evidence(request)
    child_quota_check(request, k, stage="labels")
    request_path = Path(request["control_root"]) / "child-label-request.json"
    if request_path.exists():
        raise FileExistsError("fresh child label-worker request required")
    p.atomic_json(request_path, request)
    expected = p.sha256(request_path)
    env = k.clean_environment()
    env.update(CUDA_VISIBLE_DEVICES=evidence["gpu"]["uuid"], CUBLAS_WORKSPACE_CONFIG=":4096:8",
        VISUALIZEIT_SHARDED_CHILD_LABEL=expected)
    command = [request["python"], "-E", "-s", "-B", str(Path(__file__).absolute()),
        "child-label-worker", "--request", str(request_path), "--request-sha256", expected]
    result, _ = _run_child_original(request, stage="labels", argv=command,
        output_root=features, log_path=execution / "label-worker.log",
        env_extra={"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "VISUALIZEIT_SHARDED_CHILD_LABEL": expected})
    p.atomic_json(execution / "label-stage-observation.json", result)
    if result.get("status") != "completed" or result.get("returncode") != 0:
        raise ValueError("child label worker did not complete; failure artifacts remain available")
    return result


def child_label_worker(request_path, expected_sha256):
    data = Path(request_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256 or os.environ.get("VISUALIZEIT_SHARDED_CHILD_LABEL") != expected_sha256:
        raise ValueError("label worker request hash/environment authentication failed")
    request = json.loads(data)
    _child_stage_admit(request, "labels")
    proof, state = _child_raw(request, payloads=True)
    p.verify_prepared_neural_environment(request)
    return _child_labels(request, state, Path(request["execution_root"]))


def _verify_child_supervision(request, binding, shard_id=None):
    with p.source_migration_scope(request.get("source_migration")):
        return _verify_child_supervision_sources(request, binding, shard_id)


def _verify_child_supervision_sources(request, binding, shard_id=None):
    """Authenticate immutable smoke v1 or this module's explicit 32-GiB child bridge."""
    root = Path(request["source_root"])
    provenance = binding.get("supervision")
    if (isinstance(provenance, dict) and provenance.get("version") == "kaggle_feature_supervision_v1"
            and shard_id in (None, "smoke0")):
        names = {"bridge": "bench/quality_adaptation_supervised_v1.py",
            "supervisor": "scripts/kaggle_supervision_v1.py", "scanner": "bench/quality_adaptation_portable.py"}
        expected_sources = provenance.get("sources", {})
        if set(expected_sources) != set(names):
            raise ValueError("imported smoke original supervision source inventory changed")
        for role, name in names.items():
            p.verify_prepared_source(expected_sources[role], root, name)
        if (request.get("storage_limit_bytes", 18 * GIB) > 18 * GIB
                or provenance.get("sources") != expected_sources
                or provenance.get("validation_adapter") != request.get("adapter")
                or provenance.get("supervisor") != "scripts.kaggle_supervision_v1.Facade"
                or provenance.get("output_scanner") != "raw._usage"):
            raise ValueError("imported smoke original supervision/source provenance changed")
        from bench.quality_adaptation_supervised_v1 import ORIGINAL_HELPERS
        for name, digest in ORIGINAL_HELPERS.items():
            p.verify_prepared_source_digest(root, name, digest)
        return provenance
    if (not isinstance(provenance, dict) or provenance.get("version") != CHILD_SUPERVISION_SCHEMA
            or shard_id == "smoke0" or request.get("sharded_child", {}).get("schema") != CHILD_REQUEST_VERSION
            or request.get("storage_limit_bytes") != AGGREGATE_STORAGE_LIMIT):
        raise ValueError("expanded child requires explicit new 32-GiB supervision provenance")
    expected_sources = _child_supervision_sources(root)
    if (provenance.get("sources") != expected_sources
            or provenance.get("validation_adapter") != request.get("adapter")
            or provenance.get("supervisor") != "scripts.kaggle_supervision_v1.Facade"
            or provenance.get("output_scanner") != "raw._usage"
            or provenance.get("resource_schema") != "visualizeit_expanded_child_quota_v1"
            or provenance.get("storage_limit_bytes") != AGGREGATE_STORAGE_LIMIT
            or provenance.get("source_binding") != request.get("sharded_child", {}).get("source_binding")
            or provenance.get("source_binding_sha256") != p.metadata(request["sharded_child"]["source_binding"]).get("source_binding_sha256")):
        raise ValueError("expanded child supervision receipt/immutable source binding changed")
    for name, pin in expected_sources.items():
        if p.verify_pin(pin) != root / name:
            raise ValueError("expanded child bridge source pin differs: " + name)
    history = _verify_child_stage_history(request, require_fresh_latest=False)
    extract_observation = next((entry for entry in history if entry["stage"] == "extract"), None)
    bound_history = binding.get("stage_evidence_history")
    if (extract_observation is None or binding.get("stage") != "extract"
            or binding.get("execution_evidence") != extract_observation["execution_evidence"]
            or binding.get("ui_context") != extract_observation["ui_context"]
            or not isinstance(bound_history, list) or not bound_history
            or bound_history[-1].get("stage") != "extract"
            or history[:len(bound_history)] != bound_history):
        raise ValueError("child extraction binding does not retain its exact stage evidence history")
    from bench.quality_adaptation_supervised_v1 import ORIGINAL_HELPERS
    for name, digest in ORIGINAL_HELPERS.items():
        p.verify_prepared_source_digest(root, name, digest)
    return provenance


def _child_metadata(request, partition, partition_pin, shard_id, projection_pin):
    if request.get("schema") != "kaggle_prepared_request_v2":
        raise ValueError("child lineage requires original prepared-v2 request metadata")
    joined = prepared.metadata_join(request)
    _verify_child_label_binding(request, receipt=joined["docs"]["label_receipt"])
    binding = joined["docs"]["extraction_binding"]
    provenance = _verify_child_supervision(request, binding, shard_id)
    if shard_id != "smoke0":
        if provenance != _child_supervision_provenance(request):
            raise ValueError("child metadata differs from its immutable supervision source binding")
        history = _verify_child_stage_history(request, require_fresh_latest=False)
        if [entry["stage"] for entry in history] != list(CHILD_STAGE_NAMES):
            raise ValueError("expanded child metadata requires every preparation stage observation")
        labels_observation = next((entry for entry in history if entry["stage"] == "labels"), None)
        receipt = joined["docs"]["label_receipt"]
        if (labels_observation is None or receipt.get("stage") != "labels"
                or receipt.get("ui_context") != labels_observation["ui_context"]
                or receipt.get("execution_evidence") != labels_observation["execution_evidence"]
                or receipt.get("stage_evidence_history") != history):
            raise ValueError("child label receipt does not retain its exact stage evidence history")
    proof = sharded.verify_shard(partition_pin, shard_id, request["context"],
        request["raw_seal"], projection_pin, payloads=False, **_raw_join_kwargs(request))
    if proof["projection"]["shard_id"] != shard_id:
        raise ValueError("child request RAW/projection identity changed")
    if not joined["docs"]["dataset"].get("rows"):
        raise ValueError("child label dataset is empty")
    return {**joined, "projection": proof["projection"], "projection_pin": projection_pin}


def _request_children(request, partition):
    children = request.get("child_requests")
    if not isinstance(children, list) or len(children) != len(sharded.SHARD_IDS):
        raise ValueError("complete child lineage request inventory required")
    result = {}
    for value in children:
        if not isinstance(value, dict) or set(value) != {"shard_id", "request", "projection"}:
            raise ValueError("child request requires explicit shard id, request and projection pin")
        sid = value["shard_id"]
        child = _child_request(value["request"])
        if request.get("source_migration") is not None:
            if child.get("source_migration", request["source_migration"]) != request["source_migration"]:
                raise ValueError("child source migration differs from aggregate request")
            child = {**child, "source_migration": request["source_migration"]}
        if sid not in partition["children"]:
            raise ValueError("foreign child prepared request")
        if sid in result:
            raise ValueError("duplicate child prepared request")
        if not isinstance(value["projection"], dict) or set(value["projection"]) != {"path", "bytes", "sha256"}:
            raise ValueError("child shard projection pin required")
        refs = {r["shard_id"]: r for r in request["shard_refs"]}
        reference = refs.get(sid)
        if (reference is None or child.get("context") != reference["context"]
                or child.get("raw_seal") != reference["raw_seal"]
                or value["projection"] != reference["projection"]):
            raise ValueError("child prepared context/seal/projection differs from aggregate RAW references")
        result[sid] = {"request": child, "projection": value["projection"]}
    if set(result) != set(sharded.SHARD_IDS):
        raise ValueError("missing/foreign child prepared request")
    return result


def _verify_all_child_metadata(request, partition):
    """Finish every child's metadata and sealed-TEST checks before payload reads."""
    children = _request_children(request, partition)
    joins = {sid: _child_metadata(children[sid]["request"], partition,
        request["partition"], sid, children[sid]["projection"])
        for sid in sharded.SHARD_IDS}
    return children, joins


def _prospective_label_copy_allowance(request, partition, child_requests):
    """Bound cross-filesystem label bytes using only frozen row ownership."""
    target_device = _root_device(request["features_root"])
    records = {row["id"]: row for row in partition["full_plan"]["global_manifest"]["records"]}
    allowance = 0
    for sid in sharded.SHARD_IDS:
        child = child_requests[sid]["request"]
        if _root_device(child["features_root"]) == target_device:
            continue
        owned = partition["children"][sid]["owned_state_ids"]
        active = sum(records[ident]["split"] in ("train", "validation") for ident in owned)
        if active:
            allowance += active * p.PER_STATE_ENVELOPE + p.TEMP_ENVELOPE
    return allowance


def _aggregate_request_raw(request, *, payloads=False):
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("explicit sharded prepared request required")
    verify_wrapper_sources(request)
    partition = sharded.verify_partition(request["partition"], **_migration_kwargs(request))
    raw_state = sharded.verify_aggregate_raw(request["partition"], request["aggregate_root"], payloads=payloads,
        **_migration_kwargs(request))
    if (request["artifacts"]["aggregate_seal"] != p.pin(Path(request["aggregate_root"]) / "raw-seal.json")
            or raw_state["seal"]["aggregate_sha256"] != p.metadata(request["artifacts"]["aggregate_seal"])["aggregate_sha256"]):
        raise ValueError("aggregate RAW seal request pin differs")
    context_doc = raw_state["context"]
    if (Path(request["source_root"]) != Path(context_doc["source_root"])
            or request["source_files"] != _root_sources(request["source_root"])):
        raise ValueError("aggregate source root or source closure changed")
    refs = {r["shard_id"]: r for r in request["shard_refs"]}
    if set(refs) != set(sharded.SHARD_IDS):
        raise ValueError("complete four-shard RAW references required")
    for sid, ref in refs.items():
        verified = sharded.verify_shard(request["partition"], sid, ref["context"], ref["raw_seal"], ref["projection"], payloads=False,
            **_migration_kwargs(request))
        if (verified["projection"]["partition_sha256"] != partition["partition_sha256"]
                or raw_state["seal"]["child_raw_seal_sha256"][sid] != verified["raw_state"]["seal"]["seal_sha256"]):
            raise ValueError("aggregate child seal lineage differs")
    return partition, raw_state


def _check_layout(request):
    working = p.no_links(Path(request["working_root"]))
    aggregate = p.no_links(Path(request["aggregate_root"]))
    execution = p.no_links(Path(request["execution_root"]))
    features = p.no_links(Path(request["features_root"]))
    output = p.no_links(Path(request["output_root"]))
    pilot_output = p.no_links(Path(request["pilot_output_root"]))
    feature_storage = p.no_links(Path(request["feature_storage_root"]).absolute())
    if (working == Path("/") or aggregate == execution or aggregate == features
            or execution == features or output == pilot_output):
        raise ValueError("distinct dedicated aggregate/execution/features/run roots required")
    if not features.is_absolute() or not aggregate.is_absolute() or not working.is_absolute():
        raise ValueError("absolute aggregate roots required")
    if (feature_storage in (Path("/"), Path("/tmp"))
            or not features.is_relative_to(feature_storage)
            or feature_storage.is_relative_to(working)
            or _root_device(feature_storage) == _root_device(working)):
        raise ValueError("global features must stay under a dedicated shared cache root on the separate feature filesystem")
    if (Path(request["paths"]["extraction_dir"]) != features
            or Path(request["paths"]["manifest_output"]) != features / "manifest.json"
            or Path(request["paths"]["train_dir"]) != output / "train"
            or Path(request["paths"]["export_output"]) != output / "train/export.pt"):
        raise ValueError("aggregate immutable trainer/export layout differs")
    if output == working or not output.is_relative_to(working):
        raise ValueError("dedicated training output must be below Kaggle working")
    if pilot_output == working or not pilot_output.is_relative_to(working):
        raise ValueError("separate pilot output must be below Kaggle working")
    if output == pilot_output or output.is_relative_to(pilot_output) or pilot_output.is_relative_to(output):
        raise ValueError("pilot and full training outputs must be disjoint")
    for root in (aggregate, execution, features):
        if root == output or root.is_relative_to(output) or output.is_relative_to(root):
            raise ValueError("run output overlaps prepared aggregate artifacts")
    return output


def _required_roots(request, partition, child_requests, adapter):
    roots = {Path(value).absolute() for value in request["quota_roots"]}
    # input_root is a read-only boundary. Recursively sizing it could traverse
    # mounted Kaggle inputs, including sealed TEST material, and it receives no
    # writes from any stage. Keep it available to the original consumers while
    # excluding it from resident-byte scans and filesystem write reservations.
    expected = {Path(request[k]).absolute() for k in ("working_root", "source_root", "source_install",
        "aggregate_root", "execution_root", "features_root", "feature_storage_root", "output_root", "pilot_output_root")}
    expected.update(Path(request[key]["path"]).parent.absolute() for key in ("ui_context", "execution_evidence"))
    for descriptor in request.get("neural_inputs", []):
        if isinstance(descriptor, dict) and isinstance(descriptor.get("path"), str):
            expected.add(Path(descriptor["path"]).parent.absolute())
    runtime = p.metadata(request["runtime"])
    expected.add(Path(runtime["runtime_root"]).absolute())
    for sid, child_record in child_requests.items():
        value = child_record["request"]
        if sid != "smoke0":
            if value.get("feature_storage_root") != request.get("feature_storage_root"):
                raise ValueError("all new expanded child and global features must share the same frozen cache root")
            child_features = Path(value["features_root"]).absolute()
            if not child_features.is_relative_to(Path(request["feature_storage_root"]).absolute()):
                raise ValueError("child feature tree escapes the shared retained-cache root")
        for retained in value.get("retained_raw_roots", []):
            expected.add(Path(retained).absolute())
        expected.update(Path(value[k]).absolute() for k in ("source_root", "source_install", "features_root", "execution_root", "working_root"))
        if "control_root" in value:
            expected.add(Path(value["control_root"]).absolute())
        raw_context = raw.read_context(Path(value["context"]["path"]), value["context"]["sha256"])
        raw_doc = raw_context.document()
        expected.update(Path(raw_doc[k]).absolute() for k in ("output_root", "scratch_root", "model_cache"))
        for key in ("ui_context", "execution_evidence", "exposure_audit", "label_plan", "runtime", "config"):
            descriptor = value.get(key)
            if isinstance(descriptor, dict) and isinstance(descriptor.get("path"), str):
                expected.add(Path(descriptor["path"]).parent.absolute())
    imported = partition["imported_shard0"]
    smoke_context = raw.read_context(Path(imported["context"]["path"]), imported["context"]["sha256"]).document()
    expected.update(Path(smoke_context[k]).absolute() for k in ("output_root", "scratch_root", "model_cache"))
    expected.add((Path(request["source_root"]) / ".cache/model-quality").absolute())
    expected.add((Path(request["source_install"]) / adapter.ROOT18 / ".cache/model-quality").absolute())
    for key in ("config", "validation_diagnostic", "pilot_measurement", "partition"):
        descriptor = request.get(key)
        if isinstance(descriptor, dict) and isinstance(descriptor.get("path"), str):
            expected.add(Path(descriptor["path"]).parent.absolute())
    if roots != expected:
        missing, extra = sorted(map(str, expected - roots)), sorted(map(str, roots - expected))
        raise ValueError("quota root set must include all source/runtime/RAW/scratch/feature/support trees; missing="
                         + repr(missing) + " extra=" + repr(extra))
    for root in roots:
        p.no_links(root)
    return tuple(sorted(roots, key=str))


def _stage_output_cap(request, stage, limits):
    if stage == "extract":
        return GLOBAL_FEATURE_ENVELOPE
    if stage in ("train", "resume", "export", "pilot"):
        config = p.metadata(request["config"])
        cap = config.get("max_output_bytes")
        if type(cap) is not int or cap <= 0:
            raise ValueError("expanded trainer output cap must come from the pinned config")
        return min(cap, limits["max_output_bytes"])
    return 0


def _supervisor_limits(request, adapter, stage):
    limits = adapter.validate_limits(request["limits"])
    if stage == "preflight":
        limits["max_output_bytes"] = min(limits["max_output_bytes"], MAX_CONTROL_WRITE)
    elif stage == "extract":
        limits["max_output_bytes"] = min(limits["max_output_bytes"], GLOBAL_FEATURE_ENVELOPE)
    elif stage == "labels":
        # Facade scans the whole feature tree, so cap its current contents plus
        # the prospective small control-file writes.
        from scripts.kaggle_supervision_v1 import stable_tree_bytes
        links = adapter.runtime_link_policy(p.metadata(request["runtime"]), Path(request["working_root"]))
        existing = stable_tree_bytes(adapter, (Path(request["features_root"]),), approved_links=links)
        limits["max_output_bytes"] = min(limits["max_output_bytes"], existing + MAX_CONTROL_WRITE)
    elif stage in ("train", "resume", "export", "pilot"):
        limits["max_output_bytes"] = _stage_output_cap(request, stage, limits)
    return adapter.validate_limits(limits)


def quota_check(request, adapter, *, stage, child_requests=None, live=False,
                label_copy_bytes=None):
    """New 32-GiB inode-aware gate with per-filesystem write reservations."""
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("new expanded quota applies only to the sharded aggregate schema")
    limits = adapter.validate_limits(request["limits"])
    _validate_sharded_limits(limits)
    if (limits["max_output_bytes"] > 8 * GIB
            or type(request.get("storage_limit_bytes")) is not int
            or not 18 * GIB < request["storage_limit_bytes"] <= AGGREGATE_STORAGE_LIMIT):
        raise ValueError("expanded aggregate requires <=8 GiB RSS/output, <=16 MiB logs, >=1 GiB reserve and a separately declared <=32 GiB quota")
    partition = sharded.verify_partition(request["partition"], **_migration_kwargs(request))
    children = child_requests or _request_children(request, partition)
    if stage == "labels":
        copy_bound = _prospective_label_copy_allowance(request, partition, children)
        if label_copy_bytes is None:
            label_copy_bytes = copy_bound
        elif (type(label_copy_bytes) is not int or label_copy_bytes < 0
                or label_copy_bytes > copy_bound):
            raise ValueError("exact cross-filesystem label copy bytes exceed the frozen prospective allowance")
    elif label_copy_bytes is not None:
        raise ValueError("label copy allocation applies only to the labels stage")
    roots = _required_roots(request, partition, children, adapter)
    runtime = p.metadata(request["runtime"])
    links = adapter.runtime_link_policy(runtime, Path(request["working_root"]))
    caches = (Path(request["source_root"]) / ".cache/model-quality",
              Path(request["source_install"]) / adapter.ROOT18 / ".cache/model-quality")
    from scripts.kaggle_supervision_v1 import stable_tree_bytes
    cache_bytes = stable_tree_bytes(adapter, caches)
    if cache_bytes > 8 * GIB:
        raise ValueError("combined inode-aware model caches exceed 8 GiB")
    resident = stable_tree_bytes(adapter, roots, approved_links=links)
    if resident > request["storage_limit_bytes"]:
        raise ValueError("combined inode-aware resident roots already exceed the expanded quota")
    free = adapter.free_by_filesystem(roots)
    def device(path):
        path = Path(path)
        while not path.exists():
            path = path.parent
        return path.stat().st_dev
    # Write reservations are attached to the filesystem that will receive each
    # output. In particular the global 5.7-GiB feature envelope belongs on /tmp.
    work_device = device(Path(request["working_root"]))
    feature_device = device(Path(request["features_root"]))
    output_device = None
    cap = used = 0
    if stage == "extract":
        cap = GLOBAL_FEATURE_ENVELOPE
        used = stable_tree_bytes(adapter, (Path(request["features_root"]),), approved_links=links) if live else 0
    elif stage == "labels":
        pass
    elif stage in ("train", "resume", "export", "pilot"):
        output_root = Path(request["pilot_output_root"] if stage == "pilot" else request["output_root"])
        output_device = device(output_root)
        cap = _stage_output_cap(request, stage, limits)
        used = (stable_tree_bytes(adapter, (output_root,), approved_links=links)
            if live or output_root.exists() else 0)
    allocations = _stage_write_allocations(stage, working_device=work_device,
        feature_device=feature_device, output_device=output_device, cap=cap, existing=used,
        label_copy_bytes=label_copy_bytes or 0)
    extra = sum(allocations.values())
    _enforce_storage_reserve(resident, allocations, free, limits["reserve_bytes"], request["storage_limit_bytes"])
    return {"schema": "visualizeit_expanded_quota_v1", "stage": stage,
        "resident_bytes": resident, "cache_bytes": cache_bytes,
        "storage_limit_bytes": request["storage_limit_bytes"], "extra_write_bytes": extra,
        "writes_by_device": {str(k): v for k, v in allocations.items()},
        "free_by_device": {str(k): v for k, v in free.items()},
        "reserve_bytes": limits["reserve_bytes"], "quota_roots": list(map(str, roots)),
        "inode_aware": True, "runtime_links_authenticated": True}


def _training_row_identity(state, numeric, capture):
    """Keep full RAW identity checks and the original five-key dataset schema."""
    projected = sharded._portable_state_identity(state, numeric)
    if any(capture.get(key) != value for key, value in projected.items()):
        raise ValueError('global capture frame/time/state identity changed')
    identity = {key: projected[key] for key in
                ('id', 'object_id', 'sequence_id', 'capture_group', 'split')}
    identity['split'] = 'val' if projected['split'] == 'validation' else projected['split']
    return identity


def _verify_training_row_identity(state, numeric, capture, label, final):
    identity = _training_row_identity(state, numeric, capture)
    if any(final.get(key) != value or label.get(key) != value for key, value in identity.items()):
        raise ValueError('global row identity/role changed')
    return identity


def _global_metadata(request, partition, raw_state):
    execution = p.no_links(Path(request["execution_root"]))
    features = p.no_links(Path(request["features_root"]))
    artifacts = request["artifacts"]
    expected_paths = {
        "execution_plan": execution / "execution-plan.json",
        "identity_bridge": execution / "identity-bridge.json",
        "prelabel_seal": execution / "prelabel-seal.json",
        "extraction_binding": features / "extraction-binding.json",
        "original_recipe": features / "original-recipe.json",
        "original_index": features / "original-index.json",
        "feature_index": features / "index.json", "labels": features / "labels.json",
        "label_receipt": features / "label-receipt.json", "dataset": features / "manifest.json",
    }
    for key, path in expected_paths.items():
        if p.verify_pin(artifacts[key]) != path:
            raise ValueError("aggregate artifact path differs: " + key)
    plan = p.verify_execution(raw_state, execution)
    docs = {key: p.metadata(value) for key, value in artifacts.items()}
    seal = p.check_seal(docs["prelabel_seal"], "kaggle_prelabel_seal_v2")
    binding = docs["extraction_binding"]
    if (binding.get("schema") != "kaggle_extraction_binding_v2"
            or binding.get("raw_seal") != artifacts["aggregate_seal"]
            or binding.get("execution_plan") != artifacts["execution_plan"]
            or binding.get("final_index") != artifacts["feature_index"]
            or binding.get("original_recipe") != artifacts["original_recipe"]
            or binding.get("original_index") != artifacts["original_index"]):
        raise ValueError("global extractor/aggregate binding changed")
    recipe_sha = _canonical(docs["original_recipe"])
    if (docs["original_index"].get("recipe_sha256") != recipe_sha
            or docs["feature_index"].get("recipe_sha256") != recipe_sha):
        raise ValueError("global original extractor recipe/index identity differs")
    for key, role in (("raw_seal", "aggregate_seal"), ("execution_plan", "execution_plan"),
                      ("identity_bridge", "identity_bridge"), ("binding", "extraction_binding"),
                      ("index", "feature_index")):
        if seal.get(key) != artifacts[role]:
            raise ValueError("global numerical prelabel seal differs: " + key)
    if (docs["label_receipt"].get("schema") != "visualizeit_sharded_label_join_v1"
            or docs["label_receipt"].get("prelabel_seal") != artifacts["prelabel_seal"]
            or docs["label_receipt"].get("labels") != artifacts["labels"]
            or docs["label_receipt"].get("dataset") != artifacts["dataset"]
            or docs["label_receipt"].get("test_payload_reads") != 0
            or docs["label_receipt"].get("labels_entered_inference") is not False):
        raise ValueError("aggregate post-seal label receipt changed")
    dataset_doc, label_doc, index_doc = docs["dataset"], docs["labels"], docs["feature_index"]
    if (dataset_doc.get("schema") != "correspondence_head_dataset_v1"
            or label_doc.get("schema") != "correspondence_label_inventory_v1"
            or label_doc.get("label_origin") != "synthetic_renderer_training_only"):
        raise ValueError("unchanged dataset/label schema or synthetic origin changed")
    ordered = [r for r in plan["global_manifest"]["records"] if r["split"] != "dev"]
    expected_ids = [r["id"] for r in ordered]
    captures = p.indexed(raw_state["captures"], "aggregate captures")
    datasets, labels, features_by_id = dataset_doc.get("rows", []), label_doc.get("rows", []), index_doc.get("rows", [])
    if ([r.get("id") for r in datasets] != expected_ids or [r.get("id") for r in labels] != expected_ids
            or [r.get("id") for r in features_by_id] != expected_ids):
        raise ValueError("global canonical row order/inventory differs from full plan")
    crop_set, feature_set, joint_set, aliases = set(), set(), set(), []
    for state, final, label, feature in zip(ordered, datasets, labels, features_by_id):
        capture = captures[state["id"]]
        split = "val" if state["split"] == "validation" else state["split"]
        numeric = sharded._asset_maps(partition["full_recipe"])[1]
        _verify_training_row_identity(state, numeric, capture, label, final)
        if label.get("query_rgb_sha256") != capture.get("query_rgb_sha256"):
            raise ValueError("global label query RGB differs from owned capture")
        if capture.get("status") not in ("available", "unavailable") or (
                capture.get("status") == "unavailable" and not capture.get("failure_reason")):
            raise ValueError("global capture availability/failure accounting missing")
        if (feature.get("status") not in ("available", "unavailable")
                or (feature.get("status") == "unavailable" and not feature.get("failure_reason"))):
            raise ValueError("global feature availability/failure accounting missing")
        if label.get("status") not in ("available", "unavailable") or (
                label.get("status") == "unavailable" and not label.get("failure_reason")):
            raise ValueError("global label availability/failure accounting missing")
        # Reject malformed TEST descriptors from metadata alone, before even
        # resolving a path or inspecting an inode on the sealed split.
        if split == "test":
            if (feature != {"id": state["id"], "status": "unavailable",
                            "failure_reason": feature.get("failure_reason")}
                    or final.get("status") != "unavailable" or label.get("status") != "unavailable"
                    or any(k in final or k in label for k in ("features", "labels"))):
                raise ValueError("sealed TEST rows must stay unavailable without payload descriptors")
        if capture.get("status") == "available":
            crop_set.add(state["id"])
        if feature.get("status") == "available":
            if (capture.get("status") != "available"
                    or feature.get("crops") != capture.get("crops")
                    or feature.get("query_rgb_sha256") != capture.get("query_rgb_sha256")):
                raise ValueError("global feature crop/RGB identity differs from the aggregate capture")
            p.descriptor_path(feature["features"], features)
            feature_set.add(state["id"])
        elif set(feature) != {"id", "status", "failure_reason"}:
            raise ValueError("unavailable global feature row must remain descriptor-free")
        if split != "test" and capture["status"] == "unavailable" and (label.get("status") != "unavailable" or final.get("status") != "unavailable"):
            raise ValueError("global label/dataset availability exceeds original owned capture")
        if label.get("status") == "available":
            if (final.get("status") != "available" or feature.get("status") != "available"
                    or final.get("features") != feature.get("features")
                    or final.get("labels") != label.get("labels")
                    or {key: value for key, value in final.items() if key != "features"} != label):
                raise ValueError("global joint availability exceeds feature/label inputs")
            p.descriptor_path(label["labels"], features)
            joint_set.add(state["id"])
        elif final.get("status") == "available":
            raise ValueError("dataset row cannot be available without its label inventory")
        elif any(key in final for key in ("features", "labels")):
            raise ValueError("unavailable global dataset row must remain descriptor-free")
        paths, contents = [], []
        if final.get("status") == "available":
            paths = [p.descriptor_path(final[key], features) for key in ("features", "labels")]
            contents.append(final["features"]["sha256"])
        aliases.append((split, final.get("query_rgb_sha256"), paths))
        # Distinct copied files with identical feature payloads are aliases too.
        aliases.extend((split, value, []) for value in contents)
    p.reject_aliases(aliases)
    if set(row.get("id") for row in seal.get("parity", [])) != feature_set:
        raise ValueError("global numerical seal parity inventory differs from available features")
    if index_doc.get("recipe_sha256") != _canonical(docs["original_recipe"]):
        raise ValueError("global extractor index does not bind its original recipe")
    return {"partition": partition, "raw": raw_state, "plan": plan, "docs": docs,
            "seal": seal, "binding": binding, "ordered_ids": expected_ids,
            "dataset_sha256": _canonical(dataset_doc)}


def _validation_observation(request, child_joins, joined):
    """Retain smoke-reference observations alongside global availability."""
    smoke = child_joins["smoke0"]
    rows = [row for row in smoke["docs"]["dataset"]["rows"] if row.get("split") == "val"]
    rows.sort(key=lambda row: row["id"])
    by_id = {row["id"]: row for row in rows}
    global_rows = p.indexed(joined["docs"]["dataset"]["rows"], "global validation dataset")
    global_features = p.indexed(joined["docs"]["feature_index"]["rows"], "global validation features")
    diagnostic_pin = request.get("validation_diagnostic")
    if diagnostic_pin is None:
        return {"schema": "visualizeit_sharded_validation_observation_v1",
            "source": "imported smoke-reference manifest availability; global availability is reported separately",
            "diagnostic_sha256": None, "rows": [{"id": row["id"], "object_id": row["object_id"],
                "available": row.get("status") == "available", "visible_pixels": None,
                "observed_metrics": None,
                "global_feature_available": global_features[row["id"]].get("status") == "available",
                "globally_usable": global_rows[row["id"]].get("status") == "available"} for row in rows],
            "observational_ambiguity_disclosure": "All validation rows reuse one development object. Per-row availability here describes the imported smoke reference; global feature and jointly usable counts are reported separately. Accuracy support and per-view metric uncertainty are not included in this request.",
            "used_for_admission": False, "outcome_threshold_or_filter": False}
    report = p.metadata(diagnostic_pin)
    if (report.get("schema") != "visualizeit_actual_development_validation_images_v1"
            or report.get("status") != "completed" or report.get("training_executed") is not False
            or report.get("test_payload_reads") != 0 or report.get("independent_test_claim") is not False
            or report.get("development_object_reused") is not True
            or report.get("dataset_identity") != smoke["dataset_sha256"]
            or report.get("selection") != "all_available_val_rows_sorted_by_id"):
        raise ValueError("validation diagnostic is not the pinned completed reused-development report")
    reported = report.get("rows")
    failures = report.get("failures")
    if not isinstance(reported, list) or not isinstance(failures, list):
        raise ValueError("validation diagnostic must retain rows and failures")
    measured = p.indexed(reported, "validation diagnostic rows")
    unavailable = p.indexed(failures, "validation diagnostic failures")
    actual_available = {ident for ident, row in by_id.items() if row.get("status") == "available"}
    actual_unavailable = set(by_id) - actual_available
    if (set(measured) != actual_available or set(unavailable) != actual_unavailable
            or len(reported) != report.get("available") or report.get("requested") != len(rows)):
        raise ValueError("validation diagnostic availability differs from original smoke dataset")
    result_rows = []
    for ident, row in by_id.items():
        if ident in measured:
            metric = measured[ident]
            before, after = metric.get("before"), metric.get("after")
            if (not isinstance(before, dict) or not isinstance(after, dict)
                    or type(before.get("visible_pixels")) is not int
                    or before.get("visible_pixels") <= 0
                    or before.get("visible_pixels") != after.get("visible_pixels")):
                raise ValueError("validation diagnostic row lacks exact visibility support")
            result_rows.append({"id": ident, "object_id": row["object_id"], "available": True,
                "visible_pixels": before["visible_pixels"],
                "global_feature_available": global_features[ident].get("status") == "available",
                "globally_usable": global_rows[ident].get("status") == "available",
                "observed_metrics": {"before_epe_px": before.get("epe_px"), "after_epe_px": after.get("epe_px")}})
        else:
            result_rows.append({"id": ident, "object_id": row["object_id"], "available": False,
                "visible_pixels": None, "observed_metrics": None,
                "global_feature_available": global_features[ident].get("status") == "available",
                "globally_usable": global_rows[ident].get("status") == "available",
                "failure_reason": unavailable[ident].get("reason")})
    return {"schema": "visualizeit_sharded_validation_observation_v1",
        "source": "pinned actual all-available imported smoke-reference validation diagnostic",
        "diagnostic": diagnostic_pin, "diagnostic_sha256": diagnostic_pin["sha256"],
        "rows": result_rows,
        "observational_ambiguity_disclosure": "The 24 smoke-reference views reuse one development object; 20 were available in the diagnostic, four failed, support ranged from 102 to 8,347 visible pixels, and per-view errors varied substantially. Global feature and jointly usable availability are reported per row separately. These observations do not establish independent quality or generalization.",
        "used_for_admission": False, "outcome_threshold_or_filter": False}


def metadata_join(request):
    """Complete all request, partition, lineage and TEST metadata checks first."""
    _check_layout(request)
    partition, raw_state = _aggregate_request_raw(request, payloads=False)
    history = _verify_global_stage_history(request, raw_state, require_latest_matches_request=False,
        require_completed=True)
    if [entry["stage"] for entry in history] != list(GLOBAL_PREPARATION_STAGES):
        raise ValueError("final global request requires identity, extraction, numerical seal and label evidence")
    child_requests = _request_children(request, partition)
    child_joins = {sid: _child_metadata(child_requests[sid]["request"], partition,
        request["partition"], sid, child_requests[sid]["projection"]) for sid in sharded.SHARD_IDS}
    joined = _global_metadata(request, partition, raw_state)
    joined["validation_observation"] = _validation_observation(request, child_joins, joined)
    ownership = {}
    for sid, child in partition["children"].items():
        for ident in child["owned_state_ids"]:
            if ident in ownership:
                raise ValueError("duplicate owner in global prepared join")
            ownership[ident] = sid
    label_rows = p.indexed(joined["docs"]["labels"]["rows"], "global labels")
    global_features = Path(request["features_root"]).absolute()
    artifact_by_id = p.indexed(joined["docs"]["label_receipt"].get("label_artifacts", []),
        "global label artifacts")
    expected_artifact_ids = set()
    for ident, sid in ownership.items():
        if ident not in label_rows:
            raise ValueError("owned global label row missing")
        child_label = p.indexed(child_joins[sid]["docs"]["labels"]["rows"], "child labels").get(ident)
        if child_label is None:
            raise ValueError("owner child label row missing")
        row = label_rows[ident]
        if row["split"] != child_label["split"] or row["query_rgb_sha256"] != child_label["query_rgb_sha256"]:
            raise ValueError("global label lineage differs from owning child label")
        child_dataset_row = p.indexed(child_joins[sid]["docs"]["dataset"]["rows"],
            "owner child dataset").get(ident)
        if child_dataset_row is None:
            raise ValueError("owner child dataset row missing")
        lineage = joined["docs"]["label_receipt"].get("lineage", {}).get(ident)
        if (not isinstance(lineage, dict) or lineage.get("owner_shard") != sid
                or lineage.get("child_dataset_sha256") != child_joins[sid]["dataset_sha256"]
                or lineage.get("child_label_status") != child_label["status"]
                or lineage.get("child_label_sha256") != child_label.get("labels", {}).get("sha256")
                or lineage.get("imported") is not (row.get("status") == "available")):
            raise ValueError("aggregate label row lacks exact owner-child lineage")
        if row.get("status") == "available":
            child_descriptor = child_label.get("labels")
            dataset_descriptor = child_dataset_row.get("labels")
            global_descriptor = row.get("labels")
            child_root = Path(child_requests[sid]["request"]["features_root"]).absolute()
            transport = _label_import_transport(child_root, global_features)
            source_root = str(child_root)
            if (child_label.get("status") != "available"
                    or not isinstance(child_descriptor, dict) or not isinstance(global_descriptor, dict)
                    or dataset_descriptor != child_descriptor
                    or global_descriptor.get("bytes") != child_descriptor.get("bytes")
                    or global_descriptor.get("sha256") != child_descriptor.get("sha256")
                    or not isinstance(global_descriptor.get("path"), str)
                    or Path(global_descriptor["path"]).is_absolute()
                    or (global_features / global_descriptor["path"]).resolve(strict=False).parent
                        != (global_features / "labels").resolve(strict=False)
                    or lineage.get("label_transport") != transport
                    or lineage.get("source_label") != child_descriptor
                    or lineage.get("source_features_root") != source_root):
                raise ValueError("global label payload metadata differs from the owning child's original label")
            artifact = artifact_by_id.get(ident)
            expected_path = (global_features / global_descriptor["path"]).as_posix()
            if (artifact is None
                    or set(artifact) != {"id", "path", "bytes", "sha256", "transport", "source_label", "source_features_root"}
                    or artifact.get("path") != expected_path
                    or artifact.get("bytes") != global_descriptor["bytes"]
                    or artifact.get("sha256") != global_descriptor["sha256"]
                    or artifact.get("transport") != transport
                    or artifact.get("source_label") != child_descriptor
                    or artifact.get("source_features_root") != source_root):
                raise ValueError("global label receipt inventory differs from its canonical row descriptor")
            expected_artifact_ids.add(ident)
        elif (row.get("status") != "unavailable" or ident in artifact_by_id
                or lineage.get("label_transport") is not None
                or lineage.get("source_label") is not None
                or lineage.get("source_features_root") is not None):
            raise ValueError("unavailable global label must not claim imported payload bytes")
    if set(artifact_by_id) != expected_artifact_ids:
        raise ValueError("global label receipt inventory has missing or foreign payload rows")
    if set(ownership) != set(joined["ordered_ids"]):
        raise ValueError("global row ownership does not cover all nondev states")
    identity = {
        "schema": "visualizeit_sharded_prepared_identity_v1",
        "source_files": request["source_files"], "adapter": request["adapter"],
        "source_install": request["source_install"], "python": request["python"],
        "runtime": request["runtime"], "config": request["config"],
        "partition": request["partition"], "shard_refs": request["shard_refs"],
        "aggregate_seal": request["artifacts"]["aggregate_seal"],
        "execution_plan": request["artifacts"]["execution_plan"],
        "identity_bridge": request["artifacts"]["identity_bridge"],
        "prelabel_seal": request["artifacts"]["prelabel_seal"],
        "extraction_binding": request["artifacts"]["extraction_binding"],
        "feature_index": request["artifacts"]["feature_index"],
        "labels": request["artifacts"]["labels"], "label_receipt": request["artifacts"]["label_receipt"],
        "dataset": request["artifacts"]["dataset"], "child_datasets": {sid: child_joins[sid]["dataset_sha256"] for sid in sharded.SHARD_IDS},
        "validation_observation_sha256": joined["validation_observation"].get("diagnostic_sha256"),
        "dataset_canonical_sha256": joined["dataset_sha256"],
        "paths": request["paths"], "storage_limit_bytes": request["storage_limit_bytes"],
    }
    counts = _counts(joined["docs"]["dataset"], joined["docs"]["labels"],
        joined["docs"]["feature_index"], joined["raw"]["captures"])
    return {**joined, "identity": identity, "identity_sha256": _canonical(identity),
            "child_requests": child_requests, "child_joins": child_joins,
            "counts": counts, "failures": _failures(joined["docs"]["dataset"])}


def _counts(dataset, labels, features=None, captures=None):
    counts = {split: {"requested": 0, "features": 0, "jointly_usable": 0, "objects": {}}
              for split in ("train", "validation", "test")}
    label_by_id = p.indexed(labels["rows"], "labels")
    feature_by_id = p.indexed(features["rows"], "features") if features is not None else None
    capture_by_id = p.indexed(captures, "captures") if captures is not None else None
    for row in dataset["rows"]:
        split = "validation" if row["split"] == "val" else row["split"]
        item = counts[split]
        item["requested"] += 1
        feature_row = feature_by_id[row["id"]] if feature_by_id is not None else None
        capture_row = capture_by_id[row["id"]] if capture_by_id is not None else None
        item["features"] += (feature_row.get("status") == "available" if feature_row is not None
            else row["status"] == "available")
        item["crops"] = item.get("crops", 0) + (capture_row.get("status") == "available"
            if capture_row is not None else label_by_id[row["id"]].get("status") == "available")
        item["jointly_usable"] += row["status"] == "available"
        if split == "train":
            item["objects"][row["object_id"]] = item["objects"].get(row["object_id"], 0) + (row["status"] == "available")
    return counts


def _failures(dataset):
    return [dict(id=r["id"], split=r["split"], object_id=r["object_id"], status=r["status"],
                 reason=r.get("failure_reason")) for r in dataset["rows"] if r["status"] == "unavailable"]


def assess_admission(joined):
    counts = joined["counts"]
    train = counts["train"]
    per_object = train["objects"]
    blockers = []
    if train["jointly_usable"] < 240:
        blockers.append("fewer_than_240_globally_usable_train_rows")
    if len(per_object) != 13:
        blockers.append("not_all_13_frozen_train_objects_present")
    if any(n < 12 for n in per_object.values()) or len(per_object) != 13:
        blockers.append("a_train_object_has_fewer_than_12_globally_usable_rows")
    if counts["validation"]["jointly_usable"] < 1:
        blockers.append("no_globally_usable_reused_validation_row")
    return {"schema": "visualizeit_sharded_admission_v1", "admitted": not blockers,
        "blockers": blockers, "counts": counts, "availability_failures": joined["failures"],
        "fixed_gates": {"train_rows_minimum": 240, "train_objects": 13,
            "train_rows_per_object_minimum": 12, "validation_rows_minimum": 1},
        "validation_observation": joined["validation_observation"],
        "validation_outcomes_used_for_thresholds_or_filtering": False,
        "validation_is_reused_development": True, "independent_quality_claim": False,
        "test_payload_reads": 0, "identity_sha256": joined["identity_sha256"],
        "quality_claim": False}


def verify_prepared(request, *, payloads=True):
    joined = metadata_join(request)
    if payloads:
        # Metadata join, including TEST7's last row, has completed before any
        # crop, feature, label or child payload is opened.
        partition, raw_state = _aggregate_request_raw(request, payloads=True)
        p.verify_prelabel(raw_state, Path(request["execution_root"]), Path(request["features_root"]))
        from bench.quality_adaptation_data import load_dataset
        from bench.quality_adaptation_train import validate_feature_receipt
        ds = load_dataset(Path(request["artifacts"]["dataset"]["path"]), inspect_arrays=True)
        validate_feature_receipt(ds)
        if ds.identity != joined["dataset_sha256"]:
            raise ValueError("global dataset canonical identity changed")
        for sid in sharded.SHARD_IDS:
            child = joined["child_requests"][sid]["request"]
            child_join = (prepared.verify_prepared(child, payloads=True) if sid == "smoke0"
                          else verify_child_request(child, payloads=True))
            from bench.quality_adaptation_data import load_dataset as load_child_dataset
            child_ds = load_child_dataset(Path(child["dataset"]["path"]), inspect_arrays=True)
            validate_feature_receipt(child_ds)
            if sid == "smoke0":
                _verify_child_supervision(child, child_join["docs"]["extraction_binding"], sid)
        global_labels = p.indexed(joined["docs"]["labels"]["rows"], "global label payload verification")
        ownership = {}
        for sid, child in partition["children"].items():
            for ident in child["owned_state_ids"]:
                if ident in ownership:
                    raise ValueError("duplicate owner during global label payload verification")
                ownership[ident] = sid
        for ident, sid in ownership.items():
            row = global_labels[ident]
            if row.get("status") != "available":
                continue
            child = joined["child_requests"][sid]["request"]
            child_label = p.indexed(joined["child_joins"][sid]["docs"]["labels"]["rows"],
                "owner child labels")[ident]
            source_desc, aggregate_desc = child_label["labels"], row["labels"]
            source = p.verify_pin(source_desc, Path(child["features_root"]))
            target = p.no_links(Path(request["features_root"]) / aggregate_desc["path"])
            src_stat, dst_stat = source.stat(), target.stat()
            lineage = joined["docs"]["label_receipt"]["lineage"][ident]
            transport = lineage["label_transport"]
            if transport != _label_import_transport(Path(child["features_root"]), Path(request["features_root"])):
                raise ValueError("aggregate label payload transport differs from the frozen filesystem topology")
            if transport == "hardlink":
                if (src_stat.st_dev, src_stat.st_ino) != (dst_stat.st_dev, dst_stat.st_ino):
                    raise ValueError("same-filesystem aggregate label payload is not the original child hardlink")
            elif (transport != "atomic_copy" or src_stat.st_dev == dst_stat.st_dev
                    or _same_file_identity(src_stat, dst_stat)):
                raise ValueError("cross-filesystem aggregate label payload is not a distinct atomic byte copy")
            if (src_stat.st_size != dst_stat.st_size
                    or src_stat.st_size != source_desc["bytes"]
                    or p.sha256(source) != source_desc["sha256"]
                    or p.sha256(target) != aggregate_desc["sha256"]
                    or p.sha256(source) != p.sha256(target)):
                raise ValueError("aggregate label payload bytes/hash differ from the owning child original")
    return joined


def _root_device(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path.stat().st_dev


def _supervise_with_quota_watch(facade, *, argv, cwd, output_root, working,
                                log_path, limits, env, sample, initial_resident,
                                thread_name):
    """Keep a safe outer SIGTERM guard until the quota watcher has stopped."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("bounded provider supervision must run on the main thread")
    first = sample()
    stop, cancelled, facade_entered = threading.Event(), threading.Event(), threading.Event()
    monitor_errors, high = [], [max(initial_resident, first["resident_bytes"])]
    previous_handler = signal.getsignal(signal.SIGTERM)

    def outer_guard(_signum, _frame):
        cancelled.set()
        if not facade_entered.is_set():
            raise KeyboardInterrupt

    def monitor():
        # Do not cancel the parent before Facade has installed its managed
        # child handler. A fast child can finish before the first sample.
        while signal.getsignal(signal.SIGTERM) is outer_guard:
            if stop.wait(0.01):
                return
        facade_entered.set()
        while not stop.is_set():
            try:
                value = sample()
                high[0] = max(high[0], value["resident_bytes"])
            except BaseException as exc:
                monitor_errors.append(f"{type(exc).__name__}: {exc}")
                if stop.is_set():
                    return
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                except OSError as signal_error:
                    monitor_errors.append(f"watcher cancellation signal: {type(signal_error).__name__}: {signal_error}")
                return
            if stop.wait(1.0):
                return

    watcher = threading.Thread(target=monitor, name=thread_name, daemon=True)
    signal.signal(signal.SIGTERM, outer_guard)
    watcher.start()
    result = {"status": "failed", "reason": "supervision_not_started", "returncode": None,
        "outputs_preserved": True}
    try:
        result = facade.supervise(argv, cwd=cwd, output_root=Path(output_root),
            working=Path(working), log_path=Path(log_path), limits=limits, env=env)
    except KeyboardInterrupt:
        result = {"status": "incomplete", "reason": "interrupted_before_managed_launch",
            "returncode": None, "outputs_preserved": True}
    except BaseException as exc:
        result = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
            "returncode": None, "outputs_preserved": True}
    finally:
        stop.set()
        watcher.join(timeout=5)
        if watcher.is_alive():
            monitor_errors.append("quota sampler did not stop within its bounded join")
        else:
            # Facade restored outer_guard; no sampler can signal after this.
            signal.signal(signal.SIGTERM, previous_handler)
    if cancelled.is_set() and not monitor_errors:
        monitor_errors.append("outer cancellation reached the wrapper during supervised execution")
    return result, {"sample_interval_seconds": 1.0,
        "resident_high_water_sampled_bytes": high[0], "monitor_errors": monitor_errors,
        "outer_cancelled": cancelled.is_set(), "sampler_stopped": not watcher.is_alive()}


def _cuda_bootstrap_argv(request, argv, bootstrap):
    """Replace the frozen ``-m`` entry point with the CUDA bootstrap safely."""
    expected = [request["python"], "-E", "-s", "-B", "-m"]
    if not isinstance(argv, (list, tuple)) or len(argv) < 6 or list(argv[:5]) != expected:
        raise ValueError("CUDA launch requires the pinned Python -E -s -B -m command prefix")
    return [request["python"], "-E", "-s", "-B", "-c", bootstrap, *argv[5:]]


def _cli_verification_summary(joined, *, admission=None):
    """Keep terminal verification output small and free of Path-rich internals."""
    partition = joined.get("partition", {})
    summary = {"status": "verified",
        "identity_sha256": joined.get("identity_sha256"),
        "dataset_sha256": joined.get("dataset_sha256"),
        "partition_sha256": partition.get("partition_sha256"),
        "shard_id": joined.get("shard_id"),
        "counts": joined.get("counts"),
        "failure_count": len(joined.get("failures", [])),
        "test_payload_reads": 0,
        "quality_claim": False}
    if admission is not None:
        summary["admission"] = {"admitted": admission["admitted"],
            "blockers": admission["blockers"], "counts": admission["counts"],
            "validation_observation": admission["validation_observation"],
            "independent_quality_claim": admission["independent_quality_claim"],
            "test_payload_reads": admission["test_payload_reads"]}
    return summary


def _verified_pilot_status(request, measurement):
    """Verify the pinned original-trainer status used for full-run admission."""
    status_pin = measurement.get("trainer_status")
    if not isinstance(status_pin, dict):
        return None
    expected = (Path(request["pilot_output_root"]) / "train/status.json").absolute()
    try:
        if Path(status_pin.get("path", "")).absolute() != expected:
            return None
        return p.metadata(status_pin)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _full_pilot_admission_proof(request, measurement):
    """Require the requested four-step cap and its unchanged trainer stop reason."""
    status = _verified_pilot_status(request, measurement)
    return bool(status is not None
        and measurement.get("requested_stop_after_steps") == 4
        and measurement.get("verified_optimizer_steps") == 4
        and measurement.get("trainer_status_reason") == "invocation_step_limit"
        and status.get("status") == "incomplete"
        and status.get("steps") == 4
        and status.get("reason") == "invocation_step_limit"
        and measurement.get("trainer_observation", {}).get("status") == "incomplete"
        and measurement.get("trainer_observation", {}).get("returncode") == 3)


def _execution_evidence(request, joined, *, stage, pilot=False):
    value = p.metadata(request["execution_evidence"])
    now = time.time()
    ui, ui_observed = _provider_observation(request["ui_context"])
    if (value.get("schema") != "visualizeit_sharded_execution_evidence_v1"
            or value.get("provider") != "kaggle" or value.get("paid") is not False
            or value.get("gpu_enabled") is not True or not value.get("session_id")
            or value.get("prepared_identity_sha256") != joined["identity_sha256"]
            or value.get("partition_sha256") != joined["partition"]["partition_sha256"]
            or value.get("aggregate_sha256") != joined["raw"]["seal"]["aggregate_sha256"]
            or value.get("source_files") != request["source_files"]
            or value.get("ui_context") != request["ui_context"]
            or value.get("stage") != stage):
        raise ValueError("fresh free-tier aggregate evidence/source/context/stage required")
    observed = value.get("observed_unix_s")
    expires = value.get("expires_unix_s")
    if (type(observed) not in (int, float) or type(expires) not in (int, float)
            or not now - 3600 <= observed <= now < expires <= observed + 3600):
        raise ValueError("aggregate execution evidence is stale or invalid")
    quota = value.get("quota")
    if (observed < ui_observed or observed - ui_observed > 300
            or value.get("session_id") != ui.get("session_label")
            or not isinstance(quota, dict) or quota.get("paid") is not False or quota.get("gpu_enabled") is not True
            or quota.get("storage_scope") != request["quota_roots"]
            or quota.get("storage_limit_bytes") != request["storage_limit_bytes"]
            or type(quota.get("gpu_remaining_seconds")) is not int
            or type(quota.get("session_remaining_seconds")) is not int
            or quota["gpu_remaining_seconds"] > ui["gpu_remaining_seconds"]
            or quota["session_remaining_seconds"] > ui["session_remaining_seconds"]
            or min(quota.get("gpu_remaining_seconds", 0), quota.get("session_remaining_seconds", 0)) < request["limits"]["max_seconds"]):
        raise ValueError("fresh actual provider quota does not cover bounded invocation")
    if not pilot:
        measurement = p.metadata(request.get("pilot_measurement"))
        if (measurement.get("schema") != PILOT_SCHEMA or measurement.get("full_training_admitted") is not True
                or measurement.get("prepared_identity_sha256") != joined["identity_sha256"]
                or measurement.get("dataset_sha256") != joined["dataset_sha256"]
                or measurement.get("config_sha256") != request["config"]["sha256"]
                or not _full_pilot_admission_proof(request, measurement)):
            raise ValueError("separate same-dataset full-scope pilot admission required")
    selected = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    gpu = value.get("gpu")
    if not isinstance(gpu, dict) or selected.count(f"{gpu.get('uuid')}, {gpu.get('name')}") != 1:
        raise ValueError("selected actual GPU identity changed")
    prior_observations = []
    history = request.get("stage_evidence_history")
    if not isinstance(history, list) or not history:
        raise ValueError("final execution requires the authenticated global preparation evidence history")
    prior_observations.append((history[-1]["execution_evidence"], history[-1]["ui_context"]))
    if not pilot:
        measurement = p.metadata(request["pilot_measurement"])
        prior_observations.append((measurement.get("evidence"), measurement.get("ui_context")))
    if stage in ("resume", "export"):
        output = Path(request["output_root"])
        observations = [p.read_json(path) for path in output.glob("observation-*.json")
            if path.is_file() and not path.is_symlink()]
        observations.sort(key=lambda item: item.get("invocation", {}).get("started_at", ""))
        if observations:
            latest = observations[-1]
            prior_observations.append((latest.get("evidence"), latest.get("ui_context")))
    for previous_evidence_pin, previous_ui_pin in prior_observations:
        if not isinstance(previous_evidence_pin, dict) or not isinstance(previous_ui_pin, dict):
            raise ValueError("prior stage evidence/UI observation pins are incomplete")
        previous = p.metadata(previous_evidence_pin)
        previous_ui, previous_ui_observed = _provider_observation(previous_ui_pin, require_fresh=False)
        current_ui_observed = ui_observed
        previous_gpu = previous.get("gpu")
        previous_quota = previous.get("quota", {})
        if (type(previous.get("observed_unix_s")) not in (int, float)
                or not isinstance(previous_gpu, dict) or not isinstance(previous_quota, dict)
                or previous.get("session_id") != value.get("session_id")
                or (previous_gpu.get("uuid"), previous_gpu.get("name")) != (gpu.get("uuid"), gpu.get("name"))
                or observed < previous["observed_unix_s"]
                or quota.get("gpu_remaining_seconds", 0) > previous_quota.get("gpu_remaining_seconds", 0)
                or quota.get("session_remaining_seconds", 0) > previous_quota.get("session_remaining_seconds", 0)
                or current_ui_observed < previous_ui_observed
                or ui.get("gpu_remaining_seconds", 0) > previous_ui.get("gpu_remaining_seconds", 0)
                or ui.get("session_remaining_seconds", 0) > previous_ui.get("session_remaining_seconds", 0)):
            raise ValueError("stage execution refresh reset its clock, session, GPU or remaining quota")
    return value


def _preparation_identity(request, raw_state):
    return _canonical({"schema": "visualizeit_sharded_preparation_identity_v1",
        **_migration_kwargs(request),
        "partition": request["partition"], "aggregate_sha256": raw_state["seal"]["aggregate_sha256"],
        "source_files": request["source_files"], "adapter": request["adapter"],
        "source_install": request["source_install"], "python": request["python"],
        "runtime": request["runtime"], "config": request["config"],
        "initial_ui_context": request.get("initial_ui_context"),
        # Evidence contains this identity; bind the immutable evidence pin in
        # global_stage_binding/history instead of hashing it into itself.
        "neural_inputs": request["neural_inputs"], "limits": request["limits"],
        "storage_limit_bytes": request["storage_limit_bytes"],
        "quota_roots": request["quota_roots"], "input_root": request["input_root"],
        "working_root": request["working_root"], "aggregate_root": request["aggregate_root"],
        "execution_root": request["execution_root"], "features_root": request["features_root"],
        "feature_storage_root": request["feature_storage_root"],
        "output_root": request["output_root"], "pilot_output_root": request["pilot_output_root"],
        "validation_diagnostic": request.get("validation_diagnostic"),
        "child_requests": request["child_requests"], "shard_refs": request["shard_refs"]})


def _check_fresh_stage_evidence(request, raw_state, *, stage, evidence_pin=None,
                                ui_pin=None, require_fresh=True, check_selected_gpu=True):
    evidence_pin = request["execution_evidence"] if evidence_pin is None else evidence_pin
    ui_pin = request["ui_context"] if ui_pin is None else ui_pin
    value = p.metadata(evidence_pin)
    now = time.time()
    ui, ui_observed = _provider_observation(ui_pin, require_fresh=require_fresh)
    preparation_identity = _preparation_identity(request, raw_state)
    if (value.get("schema") != "visualizeit_sharded_execution_evidence_v1"
            or value.get("provider") != "kaggle" or value.get("paid") is not False
            or value.get("gpu_enabled") is not True or not value.get("session_id")
            or request.get("stage_request_identity_sha256") != preparation_identity
            or value.get("preparation_identity_sha256") != preparation_identity
            or value.get("partition_sha256") != raw_state["partition"]["partition_sha256"]
            or value.get("aggregate_sha256") != raw_state["seal"]["aggregate_sha256"]
            or value.get("source_files") != request["source_files"]
            or value.get("ui_context") != ui_pin or value.get("stage") != stage):
        raise ValueError("fresh global preparation evidence/context/source/stage required")
    observed, expires = value.get("observed_unix_s"), value.get("expires_unix_s")
    if (type(observed) not in (int, float) or type(expires) not in (int, float)
            or observed > now or expires <= observed or expires > observed + 3600
            or (require_fresh and not now - 3600 <= observed <= now < expires)):
        raise ValueError("global preparation evidence is stale")
    quota = value.get("quota")
    if (observed < ui_observed or observed - ui_observed > 300
            or value.get("session_id") != ui.get("session_label")
            or not isinstance(quota, dict) or quota.get("paid") is not False or quota.get("gpu_enabled") is not True
            or quota.get("storage_scope") != request["quota_roots"]
            or quota.get("storage_limit_bytes") != request["storage_limit_bytes"]
            or type(quota.get("gpu_remaining_seconds")) is not int
            or type(quota.get("session_remaining_seconds")) is not int
            or quota["gpu_remaining_seconds"] > ui["gpu_remaining_seconds"]
            or quota["session_remaining_seconds"] > ui["session_remaining_seconds"]
            or min(quota.get("gpu_remaining_seconds", 0), quota.get("session_remaining_seconds", 0)) < request["limits"]["max_seconds"]):
        raise ValueError("observed free Kaggle quota does not cover global preparation invocation")
    gpu = value.get("gpu")
    if (not isinstance(gpu, dict) or not isinstance(gpu.get("uuid"), str)
            or not isinstance(gpu.get("name"), str)):
        raise ValueError("selected actual preparation GPU identity is missing")
    if check_selected_gpu:
        selected = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    else:
        selected = []
    if check_selected_gpu and selected.count(f"{gpu.get('uuid')}, {gpu.get('name')}") != 1:
        raise ValueError("selected actual preparation GPU identity changed")
    return value


GLOBAL_PREPARATION_STAGES = ("identity", "extract", "seal-features", "labels")


def _verify_global_stage_history(request, raw_state, *, require_latest_matches_request=True,
                                require_completed=False):
    history = request.get("stage_evidence_history")
    identity = _preparation_identity(request, raw_state)
    binding_pin = request.get("global_stage_binding")
    if (request.get("stage_request_identity_sha256") != identity
            or not isinstance(binding_pin, dict)
            or set(binding_pin) != {"path", "bytes", "sha256"}
            or not isinstance(history, list) or not history):
        raise ValueError("immutable global stage identity/history is missing")
    binding = p.metadata(binding_pin)
    if p.verify_pin(binding_pin) != _global_stage_binding_path(request, identity):
        raise ValueError("global stage binding escaped its dedicated working-root control path")
    _check_seal(binding, "binding_sha256", "visualizeit_sharded_global_stage_binding_v1")
    expected_binding = {"schema": "visualizeit_sharded_global_stage_binding_v1",
        "preparation_identity_sha256": identity,
        "initial_ui_context": request.get("initial_ui_context"),
        "initial_execution_evidence": request.get("initial_execution_evidence"),
        "storage_limit_bytes": request.get("storage_limit_bytes"),
        "limits": request.get("limits"), "runtime": request.get("runtime"),
        "config": request.get("config"), "source_files": request.get("source_files"),
        "partition": request.get("partition"),
        "aggregate_seal": request.get("artifacts", {}).get("aggregate_seal"),
        "quota_roots": request.get("quota_roots"), "paid_resources": False}
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        raise ValueError("immutable global stage binding differs from request/source/resource pins")
    if history[0] != {"stage": "identity", "ui_context": request.get("initial_ui_context"),
                      "execution_evidence": request.get("initial_execution_evidence")}:
        raise ValueError("global preparation history lost its initial actual observation")
    allowed = {stage: index for index, stage in enumerate(GLOBAL_PREPARATION_STAGES)}
    stage_indices = []
    prior = None
    latest = None
    for index, entry in enumerate(history):
        if (not isinstance(entry, dict) or set(entry) != {"stage", "ui_context", "execution_evidence"}
                or entry.get("stage") not in allowed):
            raise ValueError("invalid global preparation stage evidence entry")
        stage_indices.append(allowed[entry["stage"]])
        evidence = _check_fresh_stage_evidence(request, raw_state, stage=entry["stage"],
            evidence_pin=entry["execution_evidence"], ui_pin=entry["ui_context"],
            require_fresh=False, check_selected_gpu=False)
        ui, ui_observed = _provider_observation(entry["ui_context"], require_fresh=False)
        observed = evidence["observed_unix_s"]
        quota = evidence["quota"]
        gpu = evidence["gpu"]
        current = (observed, evidence["session_id"], (gpu["uuid"], gpu["name"]),
            quota["gpu_remaining_seconds"], quota["session_remaining_seconds"], ui_observed,
            ui["gpu_remaining_seconds"], ui["session_remaining_seconds"])
        if prior is not None and (current[0] < prior[0] or current[1:3] != prior[1:3]
                or current[3] > prior[3] or current[4] > prior[4]
                or current[5] < prior[5] or current[6] > prior[6] or current[7] > prior[7]):
            raise ValueError("global stage refresh reset its clock, session, GPU or observed quota")
        prior, latest = current, entry
        is_completed = require_completed or index < len(history) - 1
        if is_completed:
            receipt_paths = {
                "identity": Path(request["execution_root"]) / "identity-stage-observation.json",
                "extract": Path(request["features_root"]) / "extraction-binding.json",
                "seal-features": Path(request["execution_root"]) / "seal_features-stage-observation.json",
                "labels": Path(request["features_root"]) / "label-receipt.json"}
            receipt_path = receipt_paths[entry["stage"]]
            if not receipt_path.is_file():
                raise ValueError("previous global preparation stage lacks its completion observation: " + entry["stage"])
            receipt = p.read_json(receipt_path)
            prefix = history[:index + 1]
            if (receipt.get("stage") != entry["stage"]
                    or receipt.get("stage_request_identity_sha256") != identity
                    or receipt.get("ui_context") != entry["ui_context"]
                    or receipt.get("execution_evidence") != entry["execution_evidence"]
                    or receipt.get("stage_evidence_history") != prefix):
                raise ValueError("global stage output does not bind its exact immutable request/evidence history")
    if stage_indices != list(range(len(stage_indices))):
        raise ValueError("global preparation evidence must complete each stage in fixed order")
    if require_latest_matches_request:
        if (latest["ui_context"] != request.get("ui_context")
                or latest["execution_evidence"] != request.get("execution_evidence")):
            raise ValueError("latest global stage evidence differs from current request")
        _check_fresh_stage_evidence(request, raw_state, stage=latest["stage"],
            require_fresh=True, check_selected_gpu=True)
    return history


def global_stage_request(request, *, stage, ui_context, execution_evidence):
    """Refresh truthful stage observations without changing global run semantics."""
    if stage not in GLOBAL_PREPARATION_STAGES or stage == "identity":
        raise ValueError("global preparation refresh must advance beyond identity")
    partition, raw_state = _aggregate_request_raw(request, payloads=False)
    history = _verify_global_stage_history(request, raw_state,
        require_latest_matches_request=False, require_completed=True)
    previous = history[-1]
    if GLOBAL_PREPARATION_STAGES.index(stage) != GLOBAL_PREPARATION_STAGES.index(previous["stage"]) + 1:
        raise ValueError("global stage refresh must advance to the next preparation stage")
    value = dict(request)
    value["ui_context"] = _as_pin(ui_context)
    value["execution_evidence"] = _as_pin(execution_evidence)
    _check_fresh_stage_evidence(value, raw_state, stage=stage)
    value["stage_evidence_history"] = list(history) + [{"stage": stage,
        "ui_context": value["ui_context"], "execution_evidence": value["execution_evidence"]}]
    _verify_global_stage_history(value, raw_state)
    return value


def _write_global_stage_observation(request, stage, result):
    if stage not in GLOBAL_PREPARATION_STAGES:
        raise ValueError("unknown global preparation stage")
    observation = {"schema": "visualizeit_sharded_global_stage_observation_v1",
        "stage": stage, "stage_request_identity_sha256": request["stage_request_identity_sha256"],
        "ui_context": request["ui_context"], "execution_evidence": request["execution_evidence"],
        "stage_evidence_history": request["stage_evidence_history"], "result": result,
        "paid_resources": False, "quality_claim": False, "test_payload_reads": 0}
    target = Path(request["execution_root"]) / (stage.replace("-", "_") + "-stage-observation.json")
    p.atomic_json(target, observation)
    return p.pin(target)


def _run_original(request, *, stage, argv, output_root, log_path, joined=None, raw_state=None,
                  cuda=False, label_copy_bytes=None):
    """Run an unchanged ROOT18 producer under the pinned reviewed Facade.

    The facade owns process-group, deadline, RSS, log and per-output bounds. A
    sidecar sampler adds the new versioned all-filesystem/combined-quota watch.
    """
    p.cloud_guard(request)
    verify_wrapper_sources(request)
    k = p.load_source(request["adapter"], "_sharded_run_preflight_adapter", expected=p.ADAPTER_SHA256)
    from scripts import kaggle_supervision_v1 as supervision
    if joined is not None:
        evidence = _execution_evidence(request, joined, stage=stage, pilot=stage == "pilot")
    else:
        if raw_state is None:
            raise ValueError("global raw state required for preparation execution")
        evidence = _check_fresh_stage_evidence(request, raw_state, stage=stage)
    quota_check(request, k, stage=stage,
        child_requests=joined["child_requests"] if joined else None,
        label_copy_bytes=label_copy_bytes)
    k, root18, runtime = p.source_runtime(request)
    facade = _supervision_facade(request, k)
    env = k.clean_environment()
    env["CUDA_VISIBLE_DEVICES"] = evidence["gpu"]["uuid"]
    if cuda:
        from bench.quality_adaptation_larger import _CUDA_BOOTSTRAP
        env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",
            VISUALIZEIT_CUDA_BOOTSTRAP_SHA256=hashlib.sha256(_CUDA_BOOTSTRAP.encode()).hexdigest(),
            VISUALIZEIT_CUDA_RUNTIME_RECEIPT=str(Path(output_root) / "device-runtime.json"))
        argv = _cuda_bootstrap_argv(request, argv, _CUDA_BOOTSTRAP)
    roots = tuple(_required_roots(request, joined["partition"] if joined else raw_state["partition"],
        joined["child_requests"] if joined else _request_children(request, raw_state["partition"]), k))
    links = k.runtime_link_policy(p.metadata(request["runtime"]), Path(request["working_root"]))
    initial_resident = supervision.stable_tree_bytes(k, roots, approved_links=links)
    result, watch = _supervise_with_quota_watch(facade, argv=argv, cwd=root18,
        output_root=Path(output_root), working=Path(request["working_root"]),
        log_path=Path(log_path), limits=_supervisor_limits(request, k, stage), env=env,
        sample=lambda: quota_check(request, k, stage=stage,
            child_requests=joined["child_requests"] if joined else None, live=True,
            label_copy_bytes=label_copy_bytes),
        initial_resident=initial_resident, thread_name="sharded-quota-watch")
    final_quota, final_quota_error = None, None
    try:
        final_quota = quota_check(request, k, stage=stage,
            child_requests=joined["child_requests"] if joined else None, live=True,
            label_copy_bytes=label_copy_bytes)
    except BaseException as exc:
        final_quota_error = f"{type(exc).__name__}: {exc}"
    result["aggregate_resource_watch"] = {**watch,
        "quota": final_quota, "final_quota_error": final_quota_error}
    watch_errors = list(watch["monitor_errors"])
    if final_quota_error:
        watch_errors.append(final_quota_error)
    if watch_errors:
        result["resource_watch_failure"] = watch_errors
        if result.get("status") == "completed":
            result.update(status="incomplete", reason="aggregate_storage_watch_failed", outputs_preserved=True)
    return result, runtime


def prepare_global_identity(request):
    """Create the original execution identity over the verified aggregate rows."""
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("expanded request required")
    verify_wrapper_sources(request)
    p.cloud_guard(request)
    adapter = p.load_source(request["adapter"], "_sharded_identity_preflight_adapter", expected=p.ADAPTER_SHA256)
    partition, raw_state = _aggregate_request_raw(request, payloads=False)
    history = _verify_global_stage_history(request, raw_state)
    if history[-1]["stage"] != "identity":
        raise ValueError("global identity requires its original initial-evidence stage")
    quota_check(request, adapter, stage="identity")
    _check_fresh_stage_evidence(request, raw_state, stage="identity")
    p.source_runtime(request)
    p.bounded_recipe(partition["full_recipe"], partition["exposure_audit"])
    target = Path(request["execution_root"])
    result = p.execution_identity(raw_state, target)
    if result["global_manifest"]["split_manifest"] != partition["global_split"]:
        raise ValueError("global execution identity changed the prospective split")
    _write_global_stage_observation(request, "identity", {
        "execution_plan": p.pin(target / "execution-plan.json"),
        "identity_bridge": p.pin(target / "identity-bridge.json"),
        "plan_sha256": _canonical(result)})
    return result


def extract_global(request):
    """Call the unchanged original extractor once over the global crop manifest."""
    verify_wrapper_sources(request)
    p.cloud_guard(request)
    adapter = p.load_source(request["adapter"], "_sharded_extract_preflight_adapter", expected=p.ADAPTER_SHA256)
    metadata_raw = _aggregate_request_raw(request, payloads=False)[1]
    history = _verify_global_stage_history(request, metadata_raw)
    if history[-1]["stage"] != "extract":
        raise ValueError("global extraction requires fresh extract-stage evidence")
    quota_check(request, adapter, stage="extract")
    _check_fresh_stage_evidence(request, metadata_raw, stage="extract")
    p.source_runtime(request)
    partition, raw_state = _aggregate_request_raw(request, payloads=True)
    if Path(request["execution_root"]).joinpath("execution-plan.json").exists():
        p.verify_execution(raw_state, Path(request["execution_root"]))
    else:
        raise FileNotFoundError("global identity stage must complete under its own fresh evidence before extraction")
    features = p.no_links(Path(request["features_root"]))
    if features.exists():
        raise FileExistsError("global extractor requires a fresh dedicated feature root")
    manifest = p.crop_manifest(raw_state, Path(request["execution_root"]))
    requested = sum(row["split"] in ("train", "validation") for row in raw_state["captures"])
    if requested != GLOBAL_ACTIVE_STATES:
        raise ValueError("global feature envelope must cover all 336 prospective TRAIN/VAL states")
    features.parent.mkdir(parents=True, exist_ok=True)
    limits = request["limits"]
    if limits["max_output_bytes"] < GLOBAL_FEATURE_ENVELOPE:
        raise ValueError("global feature allowance cannot cover unchanged 16-MiB/state envelope")
    training = {"python": request["python"], "paths": {"extraction_dir": str(features)},
        "limits": limits, "extraction": {"max_seconds": int(limits["max_seconds"]),
            "max_output_bytes": p.FEATURE_TREE_CAP, "max_rows": requested},
        "crop_manifest": manifest}
    argv = p.load_source(request["adapter"], "_sharded_command_adapter", expected=p.ADAPTER_SHA256).command("extract", training)
    result, runtime = _run_original(request, stage="extract", argv=argv, output_root=features,
        log_path=Path(request["execution_root"]) / "extract.log", raw_state=raw_state, cuda=True)
    p.atomic_json(Path(request["execution_root"]) / "extract-observation.json", result)
    if result.get("status") != "completed" or result.get("returncode") != 0:
        return result
    if not (features / "index.json").is_file() or not (features / "recipe.json").is_file():
        raise ValueError("original extractor exited without authentic recipe/index")
    for name in ("recipe.json", "index.json"):
        shutil.copyfile(features / name, features / ("original-" + name))
    original_recipe = p.read_json(features / "original-recipe.json")
    original_index = p.read_json(features / "original-index.json")
    actual = p.indexed(original_index["rows"], "original global extractor")
    required = {r["id"] for r in raw_state["captures"] if r["split"] in ("train", "validation") and r["status"] == "available"}
    if set(actual) != required:
        raise ValueError("unchanged global extractor omitted available owned captures")
    final_rows = []
    for row in raw_state["captures"]:
        if row["split"] == "dev":
            continue
        final_rows.append(actual.get(row["id"], {"id": row["id"], "status": "unavailable",
            "failure_reason": row.get("failure_reason") or "capture_unavailable"}))
    final = {**original_index, "rows": final_rows}
    p.atomic_json(features / ".final-index.json", final)
    os.replace(features / ".final-index.json", features / "index.json")
    binding = {"schema": "kaggle_extraction_binding_v2",
        "original_recipe": p.pin(features / "original-recipe.json"),
        "original_index": p.pin(features / "original-index.json"),
        "final_recipe": p.pin(features / "recipe.json"), "final_index": p.pin(features / "index.json"),
        "crop_manifest": manifest, "execution_plan": p.pin(Path(request["execution_root"]) / "execution-plan.json"),
        "raw_seal": p.pin(Path(request["aggregate_root"]) / "raw-seal.json"),
        "unavailable_added": [r["id"] for r in final_rows if r["id"] not in actual],
        "dev_rows": [], "execution_result": result, "successful_original_rows_unchanged": True,
        "aggregate_schema": sharded.AGGREGATE_SCHEMA, "partition_sha256": partition["partition_sha256"],
        "source_files": request["source_files"], "runtime": runtime,
        "stage": "extract", "stage_request_identity_sha256": request["stage_request_identity_sha256"],
        "ui_context": request["ui_context"], "execution_evidence": request["execution_evidence"],
        "stage_evidence_history": request["stage_evidence_history"]}
    p.atomic_json(features / "extraction-binding.json", binding)
    observation = {"status": "completed", "requested_rows": requested, "available_input_rows": len(required),
        "feature_index": p.pin(features / "index.json"), "extraction_binding": p.pin(features / "extraction-binding.json"),
        "result": result, "quality_claim": False}
    return observation


def seal_global_features(request):
    """Run the original numerical prelabel check and seal before label access."""
    p.cloud_guard(request)
    adapter = p.load_source(request["adapter"], "_sharded_seal_preflight_adapter", expected=p.ADAPTER_SHA256)
    metadata_raw = _aggregate_request_raw(request, payloads=False)[1]
    history = _verify_global_stage_history(request, metadata_raw)
    if history[-1]["stage"] != "seal-features":
        raise ValueError("global numerical seal requires fresh seal-features evidence")
    quota_check(request, adapter, stage="seal-features")
    _check_fresh_stage_evidence(request, metadata_raw, stage="seal-features")
    p.source_runtime(request)
    partition, raw_state = _aggregate_request_raw(request, payloads=True)
    features, execution = Path(request["features_root"]), Path(request["execution_root"])
    binding = p.read_json(features / "extraction-binding.json")
    if (binding.get("aggregate_schema") != sharded.AGGREGATE_SCHEMA
            or binding.get("partition_sha256") != partition["partition_sha256"]
            or binding.get("source_files") != request["source_files"]):
        raise ValueError("new aggregate extraction source binding changed")
    result = p.seal_features(raw_state, execution, features)
    _write_global_stage_observation(request, "seal-features", {
        "prelabel_seal": p.pin(execution / "prelabel-seal.json"), "result": result})
    return result


def _label_import_transport(source_root, target_root):
    """Choose hardlinks within a filesystem and byte copies across filesystems."""
    return "hardlink" if _root_device(source_root) == _root_device(target_root) else "atomic_copy"


def _same_file_identity(left_stat, right_stat):
    """Compare file identity correctly across devices, where inode numbers may repeat."""
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def _atomic_copy_label(source, source_desc, target, *, on_progress=None):
    """Copy, verify, and publish a label without exposing a partial destination."""
    temporary = None
    digest, copied = hashlib.sha256(), 0
    try:
        with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
                mode="wb", prefix="." + target.name + ".copy-", dir=target.parent,
                delete=False) as output_file:
            temporary = Path(output_file.name)
            while True:
                block = input_file.read(1024 * 1024)
                if not block:
                    break
                if copied + len(block) > source_desc["bytes"]:
                    raise ValueError("child label grew beyond its authenticated copy byte allocation")
                output_file.write(block)
                digest.update(block)
                copied += len(block)
                output_file.flush()
                if on_progress is not None:
                    on_progress(len(block))
            output_file.flush()
            os.fsync(output_file.fileno())
        if copied != source_desc["bytes"] or digest.hexdigest() != source_desc["sha256"]:
            raise ValueError("copied child label bytes/hash differ from its authenticated descriptor")
        # A hardlink within this target directory publishes only the complete,
        # verified temporary file and fails if another writer created target.
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _import_label(source_desc, source_root, target, *, on_copy_progress=None):
    source = p.verify_pin(source_desc, source_root)
    target = p.no_links(Path(target).absolute())
    if target.exists():
        raise FileExistsError("fresh aggregate label destination required")
    target.parent.mkdir(parents=True, exist_ok=True)
    p.no_links(target.parent)
    transport = _label_import_transport(source_root, target.parent)
    if transport == "hardlink":
        os.link(source, target, follow_symlinks=False)
    else:
        _atomic_copy_label(source, source_desc, target, on_progress=on_copy_progress)
    src, dst = source.stat(), target.stat()
    if transport == "hardlink" and not _same_file_identity(src, dst):
        raise ValueError("same-filesystem global label import is not the original child hardlink")
    if (transport == "atomic_copy" and (src.st_dev == dst.st_dev
            or _same_file_identity(src, dst))):
        raise ValueError("atomic label copy unexpectedly aliases its source inode")
    if p.sha256(target) != source_desc["sha256"] or dst.st_size != source_desc["bytes"]:
        raise ValueError("global label import differs from authentic child label bytes/hash")
    return {"path": target, "transport": transport, "source": source_desc,
        "source_features_root": str(Path(source_root).absolute()), "target": p.pin(target)}


def _canonical_owned_rows(partition):
    """Return non-dev global rows once each, in original full-plan order."""
    states = partition["full_plan"]["global_manifest"]["records"]
    owners = {}
    for sid, child in partition["children"].items():
        for ident in child["owned_state_ids"]:
            if ident in owners:
                raise ValueError("global owner map contains duplicate ownership")
            owners[ident] = sid
    state_ids = [row["id"] for row in states]
    if len(state_ids) != len(set(state_ids)) or set(owners) != set(state_ids):
        raise ValueError("frozen ownership does not cover each global state exactly once")
    return [(row, owners[row["id"]]) for row in states if row["split"] != "dev"]


def join_labels_after_prelabel(request):
    """Import owned child labels only after the full global numerical seal."""
    verify_wrapper_sources(request)
    p.cloud_guard(request)
    adapter = p.load_source(request["adapter"], "_sharded_label_preflight_adapter", expected=p.ADAPTER_SHA256)
    metadata_raw = _aggregate_request_raw(request, payloads=False)[1]
    history = _verify_global_stage_history(request, metadata_raw)
    if history[-1]["stage"] != "labels":
        raise ValueError("global label import requires fresh labels-stage evidence")
    quota_check(request, adapter, stage="labels")
    _check_fresh_stage_evidence(request, metadata_raw, stage="labels")
    p.source_runtime(request)
    partition, raw_state = _aggregate_request_raw(request, payloads=True)
    execution, features = Path(request["execution_root"]), Path(request["features_root"])
    prelabel = p.verify_prelabel(raw_state, execution, features)
    # The line above authenticates all global numerical/crop data before the
    # first child label inventory, descriptor or payload is touched.
    children, child_joins = _verify_all_child_metadata(request, partition)
    child_datasets = {}
    for sid in sharded.SHARD_IDS:
        child = children[sid]["request"]
        payload_join = (prepared.verify_prepared(child, payloads=True) if sid == "smoke0"
                      else verify_child_request(child, payloads=True))
        if sid == "smoke0":
            _verify_child_supervision(child, payload_join["docs"]["extraction_binding"], sid)
        if payload_join["identity_sha256"] != child_joins[sid]["identity_sha256"]:
            raise ValueError("child payload verification changed its earlier metadata identity")
        from bench.quality_adaptation_data import load_dataset
        from bench.quality_adaptation_train import validate_feature_receipt
        ds = load_dataset(Path(child["dataset"]["path"]), inspect_arrays=True)
        validate_feature_receipt(ds)
        child_datasets[sid] = ds
    feature_index = p.read_json(features / "index.json")
    global_features = p.indexed(feature_index["rows"], "global extraction index")
    aggregate_captures = p.indexed(raw_state["captures"], "global captures")
    owner_rows = _canonical_owned_rows(partition)
    # Exact label-copy bytes are known from authenticated child metadata only
    # after the global prelabel seal and payload checks above. Reserve them
    # before the first write; the earlier stage gate used a frozen upper bound.
    copy_budget = 0
    for state, sid in owner_rows:
        if state["split"] not in ("train", "validation"):
            continue
        ident = state["id"]
        capture, feature = aggregate_captures[ident], global_features[ident]
        child_label = p.indexed(child_joins[sid]["docs"]["labels"]["rows"], "owner child labels")[ident]
        child_dataset = p.indexed(child_joins[sid]["docs"]["dataset"]["rows"], "owner child dataset")[ident]
        if (capture["status"] != "available" or feature["status"] != "available"
                or child_label["status"] != "available" or child_dataset["status"] != "available"):
            continue
        source_root = Path(children[sid]["request"]["features_root"])
        if _label_import_transport(source_root, features) == "atomic_copy":
            descriptor = child_dataset["labels"]
            if type(descriptor.get("bytes")) is not int or descriptor["bytes"] < 0:
                raise ValueError("authentic child label descriptor lacks a bounded byte count")
            copy_budget += descriptor["bytes"]
    quota_check(request, adapter, stage="labels", child_requests=children,
        label_copy_bytes=copy_budget, live=True)
    copy_remaining = copy_budget

    def copy_progress(written):
        nonlocal copy_remaining
        copy_remaining -= written
        if copy_remaining < 0:
            raise ValueError("label copy exceeded its authenticated prospective byte total")
        quota_check(request, adapter, stage="labels", child_requests=children,
            label_copy_bytes=copy_remaining, live=True)

    inventory, lineage, label_artifacts, outcomes = [], {}, [], []
    for state, sid in owner_rows:
        ident = state["id"]
        capture = aggregate_captures[ident]
        feature = global_features[ident]
        split = "val" if state["split"] == "validation" else state["split"]
        numeric = sharded._asset_maps(partition["full_recipe"])[1]
        row = {**_training_row_identity(state, numeric, capture),
            "query_rgb_sha256": capture.get("query_rgb_sha256"), "status": "unavailable"}
        reason = None
        child_label = None
        child_dataset = None
        if state["split"] == "test":
            reason = "sealed_test_metadata_only"
            row["unavailable_provenance"] = "sealed TEST metadata; payload descriptors and opens forbidden"
        else:
            child_docs = child_joins[sid]["docs"]
            child_label = p.indexed(child_docs["labels"]["rows"], "owner child labels")[ident]
            child_dataset = p.indexed(child_docs["dataset"]["rows"], "owner child dataset")[ident]
            if (child_label["query_rgb_sha256"] != capture.get("query_rgb_sha256")
                    or child_label["split"] != split
                    or child_dataset["sequence_id"] != state["sequence_id"]
                    or child_dataset["capture_group"] != state["scene_id"]):
                raise ValueError("owned child label/crop RGB/geometry lineage differs")
            if capture["status"] == "unavailable":
                reason = capture.get("failure_reason") or "owned_capture_unavailable"
            elif feature["status"] != "available":
                reason = feature.get("failure_reason") or "global_feature_unavailable"
            elif child_label["status"] != "available" or child_dataset["status"] != "available":
                reason = child_label.get("failure_reason") or child_dataset.get("failure_reason") or "owned_child_label_unavailable"
            else:
                source_desc = child_dataset["labels"]
                source_root = Path(children[sid]["request"]["features_root"])
                target = features / "labels" / (hashlib.sha256(ident.encode()).hexdigest() + ".npz")
                transport = _label_import_transport(source_root, features)
                if transport == "atomic_copy":
                    quota_check(request, adapter, stage="labels", child_requests=children,
                        label_copy_bytes=copy_remaining, live=True)
                imported = _import_label(source_desc, source_root, target,
                    on_copy_progress=copy_progress if transport == "atomic_copy" else None)
                imported_path = imported["path"]
                if imported["transport"] == "atomic_copy" and transport != "atomic_copy":
                    raise ValueError("label import transport changed after stage quota admission")
                if imported["transport"] == "hardlink" and transport != "hardlink":
                    raise ValueError("label import transport changed after stage quota admission")
                row.update(status="available", labels={"path": imported_path.relative_to(features).as_posix(),
                    "bytes": source_desc["bytes"], "sha256": source_desc["sha256"]})
                label_artifacts.append({"id": ident, **imported["target"],
                    "transport": imported["transport"], "source_label": imported["source"],
                    "source_features_root": imported["source_features_root"]})
        if row["status"] != "available":
            row["failure_reason"] = reason or "owned_child_label_unavailable"
            if row["query_rgb_sha256"] is None:
                row["unavailable_provenance"] = ("sealed TEST metadata; payload descriptors and opens forbidden"
                    if state["split"] == "test" else capture.get("unavailable_provenance", "frozen state; no retained crop"))
        inventory.append(row)
        lineage[ident] = {"owner_shard": sid,
            "child_dataset_sha256": child_joins[sid]["dataset_sha256"],
            "child_label_status": child_label["status"] if child_label else "unavailable",
            "child_label_sha256": child_label.get("labels", {}).get("sha256") if child_label else None,
            "label_transport": imported["transport"] if row["status"] == "available" else None,
            "source_label": imported["source"] if row["status"] == "available" else None,
            "source_features_root": imported["source_features_root"] if row["status"] == "available" else None,
            "global_feature_status": feature["status"], "global_capture_status": capture["status"],
            "imported": row["status"] == "available"}
        outcomes.append({"id": ident, "owner_shard": sid, "status": row["status"], "reason": row.get("failure_reason")})
    if copy_remaining != 0:
        raise ValueError("actual cross-filesystem label copies differ from the preflight byte total")
    p.atomic_json(features / "labels.json", {"schema": "correspondence_label_inventory_v1",
        "label_origin": "synthetic_renderer_training_only", "rows": inventory})
    # Use the original assembler so its unchanged row/archive validators remain
    # the authority for the final global manifest.
    k = p.load_source(request["adapter"], "_sharded_assembler_adapter", expected=p.ADAPTER_SHA256)
    argv = k.command("assemble", {"python": request["python"], "limits": request["limits"], "paths": {"extraction_dir": str(features),
        "manifest_output": str(features / "manifest.json")}, "labels": {"path": str(features / "labels.json")}})
    result, runtime = _run_original(request, stage="labels", argv=argv, output_root=features,
        log_path=execution / "assemble.log", raw_state=raw_state, label_copy_bytes=0)
    p.atomic_json(execution / "assemble-observation.json", result)
    if result.get("status") != "completed" or result.get("returncode") != 0:
        raise ValueError("unchanged global dataset assembler failed; all label outcomes were preserved")
    receipt = {"schema": "visualizeit_sharded_label_join_v1",
        "prelabel_seal": p.pin(execution / "prelabel-seal.json"),
        "labels": p.pin(features / "labels.json"), "dataset": p.pin(features / "manifest.json"),
        "label_origin": "synthetic_renderer_training_only", "lineage": lineage,
        "label_artifacts": label_artifacts, "outcomes": outcomes,
        "child_requests": {sid: p.pin(children[sid]["request"]["dataset"]["path"]) for sid in sharded.SHARD_IDS},
        "labels_entered_inference": False, "test_payload_reads": 0,
        "independent_quality_claim": False, "validation_is_reused_development": True,
        "global_prelabel_sha256": prelabel["seal_sha256"], "assembler_result": result,
        "runtime": runtime, "stage": "labels",
        "stage_request_identity_sha256": request["stage_request_identity_sha256"],
        "ui_context": request["ui_context"], "execution_evidence": request["execution_evidence"],
        "stage_evidence_history": request["stage_evidence_history"]}
    p.atomic_json(features / "label-receipt.json", receipt)
    return receipt


def training_request(request, *, output_root=None):
    """Adapt only request field names for the unchanged ROOT18 trainer CLI."""
    output = Path(output_root or request["output_root"])
    features = Path(request["features_root"])
    return {"python": request["python"], "paths": {
        "extraction_dir": str(features), "manifest_output": str(features / "manifest.json"),
        "train_dir": str(output / "train"), "export_output": str(output / "train/export.pt")},
        "limits": request["limits"], "config": request["config"],
        "dataset": request["artifacts"]["dataset"]}


def _run_identity(joined):
    return {"schema": "visualizeit_sharded_run_identity_v1",
        "prepared_identity_sha256": joined["identity_sha256"],
        "prepared_identity": joined["identity"], "dataset_sha256": joined["dataset_sha256"],
        "validation_is_reused_development": True, "independent_quality_claim": False,
        "test_payload_reads": 0}


def verify_run_resume(output, identity):
    output = p.no_links(Path(output).absolute())
    if p.read_json(output / RUN_IDENTITY) != identity:
        raise ValueError("resume/export changed the immutable expanded dataset/config/source/runtime identity")
    receipt = p.read_json(output / CHECKPOINT_RECEIPT)
    if (receipt.get("schema") != "visualizeit_sharded_checkpoint_receipt_v1"
            or receipt.get("run_identity_sha256") != _canonical(identity)):
        raise ValueError("checkpoint receipt is not bound to this expanded run")
    train = output / "train"
    if not train.is_dir():
        raise ValueError("resume/export requires an existing train directory")
    current = {path.name for path in train.iterdir() if path.is_file() and not path.is_symlink()}
    recorded = {Path(value["path"]).name for value in receipt.get("files", [])}
    if current != recorded or "last.pt" not in recorded:
        raise ValueError("checkpoint/run inventory changed or last.pt is missing")
    for value in receipt["files"]:
        path = p.verify_pin(value)
        if path.parent != train:
            raise ValueError("checkpoint receipt escapes immutable expanded run")
    return receipt


def checkpoint_receipt(output, identity):
    output = p.no_links(Path(output).absolute())
    train = output / "train"
    files = []
    if train.is_dir():
        for path in sorted(train.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise ValueError("unexpected trainer artifact in expanded run")
            files.append(p.pin(path))
    receipt = {"schema": "visualizeit_sharded_checkpoint_receipt_v1",
        "run_identity_sha256": _canonical(identity), "files": files}
    temporary = output / (".checkpoint-" + uuid.uuid4().hex + ".json")
    p.atomic_json(temporary, receipt)
    os.replace(temporary, output / CHECKPOINT_RECEIPT)
    return receipt


def _validate_full_config(request, limits):
    config = p.metadata(request["config"])
    approved = {"device": "cuda", "epochs": 3, "max_steps": 936,
        "seed": 521, "batch_size": 1, "learning_rate": 1e-5,
        "weight_decay": 1e-4, "visibility_weight": 1.0, "amp": False,
        "patience": 3, "min_delta_px": 0.001, "gradient_clip": 1.0,
        "checkpoint_every_steps": 8, "max_seconds": 900.0,
        "max_total_seconds": 7200.0, "max_checkpoint_bytes": 335544320,
        "max_output_bytes": 1073741824}
    if (any(config.get(key) != value for key, value in approved.items())
            or config.get("max_steps") != 936
            or config["max_output_bytes"] > limits["max_output_bytes"]
            or config["max_seconds"] > limits["max_seconds"]
            or limits["max_seconds"] > 900):
        raise ValueError("expanded training requires the frozen FP32 seed-521, batch-one, 3-epoch/936-step config within bounded request limits")
    return config


def _attempt_identity(joined, output, stage):
    return {"roots": {"output_root": str(output)}, "paths": {"train_dir": str(output / "train")},
        "stages": [stage]}


def execute(stage, request, *, stop_after_steps=None):
    """Run one unchanged trainer/export stage for the immutable 13-object set."""
    if stage not in ("preflight", "train", "resume", "export"):
        raise ValueError("expanded consumer supports preflight/train/resume/export only")
    if stop_after_steps is not None and (stage != "train" or type(stop_after_steps) is not int
            or not 1 <= stop_after_steps <= 4):
        raise ValueError("full-run interruption seam permits only an explicit 1..4-update initial train cap")
    joined = metadata_join(request)
    output = _check_layout(request)
    p.cloud_guard(request)
    adapter = p.load_source(request["adapter"], "_sharded_execute_preflight_adapter", expected=p.ADAPTER_SHA256)
    limits = adapter.validate_limits(request["limits"])
    _validate_full_config(request, limits)
    if limits["max_rss_bytes"] > 8 * GIB or limits["max_output_bytes"] > 8 * GIB:
        raise ValueError("expanded run exceeds the retained 8 GiB RSS/output bound")
    resource_before = quota_check(request, adapter, stage=stage, child_requests=joined["child_requests"])
    evidence = _execution_evidence(request, joined, stage=stage)
    admission = assess_admission(joined)
    if not admission["admitted"]:
        raise ValueError("expanded admission failed: " + ",".join(admission["blockers"]))
    adapter, root18, runtime = p.source_runtime(request)
    payload_joined = verify_prepared(request, payloads=True)
    if payload_joined["identity_sha256"] != joined["identity_sha256"]:
        raise ValueError("expanded metadata identity changed before payload preflight")
    joined = payload_joined
    identity = _run_identity(joined)
    if stage in ("resume", "export"):
        verify_run_resume(output, identity)
    elif stage == "train":
        if output.exists():
            raise FileExistsError("fresh expanded base-init run requires a fresh output; resume is a separate stage")
        output.mkdir(parents=True)
        p.atomic_json(output / RUN_IDENTITY, identity)
    else:
        output = Path(request["execution_root"]) / ("trainer-preflight-" + uuid.uuid4().hex)
        output.mkdir(parents=True, exist_ok=False)
    command = adapter.command(stage, training_request(request))
    if stop_after_steps is not None:
        command += ["--stop-after-steps", str(stop_after_steps)]
    with adapter.run_lock(output):
        observation, run_runtime = _run_original(request, stage=stage, argv=command,
            output_root=output, log_path=output / (stage + "-" + uuid.uuid4().hex + ".log"), joined=joined)
        if stage in ("train", "resume", "export"):
            checkpoint_receipt(output, identity)
        after = quota_check(request, adapter, stage="preflight" if stage == "preflight" else stage,
            child_requests=joined["child_requests"], live=True)
        record = {"schema": "visualizeit_sharded_execution_observation_v1", "stage": stage,
            "prepared_identity_sha256": joined["identity_sha256"], "admission": admission,
            "runtime": runtime, "run_runtime": run_runtime, "evidence": request["execution_evidence"],
            "ui_context": request["ui_context"], "resource_before": resource_before,
            "resource_after": after, "result": observation, "quality_claim": False,
            "validation_is_reused_development": True, "independent_quality_claim": False,
            "test_payload_reads": 0,
            "invocation": {"stage": stage, "started_at": datetime.now(timezone.utc).isoformat(),
                "adapter_pid": os.getpid(), "trainer_source_sha256": adapter.file_sha256(root18 / "bench/quality_adaptation_train.py"),
                "stop_after_steps": stop_after_steps}}
        p.atomic_json(output / ("observation-" + uuid.uuid4().hex + ".json"), record)
    return record


def pilot_execute(request, *, stop_after_steps=4):
    """Run a separate 1..4-update diagnostic; never write into the full run."""
    if type(stop_after_steps) is not int or not 1 <= stop_after_steps <= 4:
        raise ValueError("expanded pilot permits only 1..4 original-trainer optimizer updates")
    joined = metadata_join(request)
    _check_layout(request)
    p.cloud_guard(request)
    adapter = p.load_source(request["adapter"], "_sharded_pilot_preflight_adapter", expected=p.ADAPTER_SHA256)
    limits = adapter.validate_limits(request["limits"])
    config = _validate_full_config(request, limits)
    if limits["max_seconds"] > 900:
        raise ValueError("bounded pilot invocation must be at most 900 seconds")
    before = quota_check(request, adapter, stage="pilot", child_requests=joined["child_requests"])
    evidence = _execution_evidence(request, joined, stage="pilot", pilot=True)
    admission = assess_admission(joined)
    if not admission["admitted"]:
        raise ValueError("expanded pilot requires the fixed data/resource admission gates")
    adapter, root18, runtime = p.source_runtime(request)
    payload_joined = verify_prepared(request, payloads=True)
    if payload_joined["identity_sha256"] != joined["identity_sha256"]:
        raise ValueError("expanded metadata identity changed before pilot payload preflight")
    joined = payload_joined
    output = p.no_links(Path(request["pilot_output_root"]).absolute())
    if output.exists():
        raise FileExistsError("expanded pilot needs its own fresh output; no checkpoint promotion")
    output.mkdir(parents=True)
    identity = _run_identity(joined)
    p.atomic_json(output / RUN_IDENTITY, identity)
    started = time.monotonic()
    with adapter.run_lock(output):
        command = adapter.command("train", training_request(request, output_root=output))
        command += ["--stop-after-steps", str(stop_after_steps)]
        result, pilot_runtime = _run_original(request, stage="pilot", argv=command,
            output_root=output, log_path=output / "pilot-train.log", joined=joined)
        status = adapter.read_json(output / "train/status.json") if (output / "train/status.json").is_file() else {}
        steps = status.get("steps") if type(status.get("steps")) is int else 0
        status_pin = p.pin(output / "train/status.json") if (output / "train/status.json").is_file() else None
        checkpoint = p.pin(output / "train/last.pt") if (output / "train/last.pt").is_file() else None
        errors = []
        if steps != stop_after_steps:
            errors.append("original trainer did not confirm exactly the requested optimizer update count")
        if status.get("status") != "incomplete" or status.get("reason") != "invocation_step_limit":
            errors.append("pilot must end with the unchanged trainer's invocation_step_limit status")
        if checkpoint is None or checkpoint["bytes"] > config["max_checkpoint_bytes"]:
            errors.append("bounded pilot checkpoint is absent or exceeds the frozen checkpoint cap")
        if result.get("status") != "incomplete" or result.get("returncode") != 3:
            errors.append("pilot must stop through the unchanged trainer's explicit update cap")
        if status.get("test_arrays_opened", False) is not False or status.get("quality_claim", False) is not False:
            errors.append("pilot trainer status has forbidden test/quality claim")
        after = quota_check(request, adapter, stage="pilot", child_requests=joined["child_requests"], live=True)
        record = {"schema": PILOT_SCHEMA, "prepared_identity_sha256": joined["identity_sha256"],
            "dataset_sha256": joined["dataset_sha256"], "config_sha256": request["config"]["sha256"],
            "measurement_complete": not errors, "full_training_admitted": False, "status": "failed",
            "quality_claim": False, "test_arrays_opened": False, "test_payload_reads": 0,
            "requested_stop_after_steps": stop_after_steps,
            "verified_optimizer_steps": steps, "trainer_status_reason": status.get("reason"),
            "trainer_status": status_pin, "checkpoint": checkpoint,
            "runtime": runtime, "pilot_runtime": pilot_runtime, "profile": request["profile"],
            "evidence": request["execution_evidence"], "ui_context": request["ui_context"],
            "resource_before": before, "resource_after": after, "trainer_observation": result,
            "observed": {"whole_pilot_seconds": time.monotonic() - started,
                "peak_group_rss_bytes": result.get("peak_group_rss_bytes"),
                "sampled_combined_storage_bytes": result.get("aggregate_resource_watch", {}).get("resident_high_water_sampled_bytes"),
                "peak_is_exact": False}, "failures": errors,
            "validation_is_reused_development": True, "independent_quality_claim": False}
        if not errors and stop_after_steps == 4 and _full_pilot_admission_proof(request, record):
            try:
                full_resources = quota_check(request, adapter, stage="train", child_requests=joined["child_requests"])
                if config["max_seconds"] > limits["max_seconds"]:
                    raise ValueError("full training wall bound exceeds request")
                record["full_scope_bound"] = {"dataset_sha256": joined["dataset_sha256"],
                    "config_sha256": request["config"]["sha256"], "planned_optimizer_steps": 936,
                    "full_epochs": 3, "enforced_full_run_resources": full_resources,
                    "basis": "all-data gates and complete conservative resource envelope; pilot timing is not extrapolated",
                    "completion_forecast_available": False}
                record.update(full_training_admitted=True, status="measurement_complete")
            except (ValueError, KeyError, OSError) as exc:
                record["failures"].append("full-scope admission withheld: " + str(exc))
        elif not errors:
            record["full_scope_admission_blockers"] = ["full training requires the separately measured four-update pilot"]
        p.atomic_json(output / "pilot-measurement.json", record)
        return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    stages = (
        "refresh-child", "refresh-global", "child-bind-labels", "child-identity", "child-extract",
        "child-seal-features", "child-labels", "child-label-worker",
        "child-verify", "child-finalize", "global-identity", "global-extract",
        "global-seal-features", "global-labels", "global-verify", "global-finalize",
        "inspect", "preflight", "pilot", "train", "resume", "export",
    )
    parser.add_argument("stage", choices=stages)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path,
        help="destination for a refreshed or finalized request document")
    parser.add_argument("--evidence-stage", choices=CHILD_STAGE_NAMES + GLOBAL_PREPARATION_STAGES,
        help="next completed preparation stage when refreshing truthful provider/UI evidence")
    parser.add_argument("--ui-context", type=Path,
        help="fresh provider UI-context evidence JSON used only by refresh-child/global")
    parser.add_argument("--execution-evidence", type=Path,
        help="fresh execution evidence JSON used only by refresh-child/global")
    parser.add_argument("--stop-after-steps", type=int,
        help="bounded 1..4-update interruption point for an initial full train invocation")
    args = parser.parse_args(argv)

    if args.stage == "child-label-worker":
        result = child_label_worker(args.request, args.request_sha256)
    else:
        request = p.load_request(args.request, args.request_sha256)
        if args.stage in ("refresh-child", "refresh-global"):
            if not args.output or not args.evidence_stage or not args.ui_context or not args.execution_evidence:
                parser.error("request refresh requires --output, --evidence-stage, --ui-context and --execution-evidence")
            if args.stage == "refresh-child":
                if args.evidence_stage not in CHILD_STAGE_NAMES:
                    parser.error("refresh-child requires a child preparation stage")
                refreshed = child_stage_request(request, stage=args.evidence_stage,
                    ui_context=args.ui_context, execution_evidence=args.execution_evidence)
            else:
                if args.evidence_stage not in GLOBAL_PREPARATION_STAGES:
                    parser.error("refresh-global requires a global preparation stage")
                refreshed = global_stage_request(request, stage=args.evidence_stage,
                    ui_context=args.ui_context, execution_evidence=args.execution_evidence)
            p.atomic_json(args.output, refreshed)
            result = {"status": "request_written", "stage": args.evidence_stage,
                "request": p.pin(args.output)}
        elif args.stage == "child-bind-labels":
            if not args.output:
                parser.error("child-bind-labels requires --output")
            bound = bind_child_label_plan(request)
            p.atomic_json(args.output, bound)
            result = {"status": "request_written", "request": p.pin(args.output)}
        elif args.stage == "child-verify":
            result = _cli_verification_summary(verify_child_request(request, payloads=False))
        elif args.stage == "child-finalize":
            if not args.output:
                parser.error("child-finalize requires --output")
            finalized = finalize_child_request(request)
            verify_child_request(finalized, payloads=False)
            p.atomic_json(args.output, finalized)
            result = {"status": "request_written", "request": p.pin(args.output)}
        elif args.stage == "child-identity":
            result = child_prepare_identity(request)
        elif args.stage == "child-extract":
            result = child_extract(request)
        elif args.stage == "child-seal-features":
            result = child_seal_features(request)
        elif args.stage == "child-labels":
            result = child_labels(request)
        elif args.stage == "global-identity":
            result = prepare_global_identity(request)
        elif args.stage == "global-extract":
            result = extract_global(request)
        elif args.stage == "global-seal-features":
            result = seal_global_features(request)
        elif args.stage == "global-labels":
            result = join_labels_after_prelabel(request)
        elif args.stage == "global-verify":
            result = _cli_verification_summary(verify_prepared(request, payloads=False))
        elif args.stage == "global-finalize":
            if not args.output:
                parser.error("global-finalize requires --output")
            finalized = finalize_prepared_request(request)
            verify_prepared(finalized, payloads=False)
            p.atomic_json(args.output, finalized)
            result = {"status": "request_written", "request": p.pin(args.output)}
        elif args.stage == "inspect":
            joined = verify_prepared(request, payloads=False)
            result = _cli_verification_summary(joined, admission=assess_admission(joined))
        elif args.stage == "pilot":
            result = pilot_execute(request)
        elif args.stage in ("preflight", "train", "resume", "export"):
            result = execute(args.stage, request, stop_after_steps=args.stop_after_steps)
        else:
            raise AssertionError("unhandled sharded stage: " + args.stage)
    print(json.dumps(result, allow_nan=False))
    status = result.get("status") if isinstance(result, dict) else None
    if status is None and isinstance(result, dict):
        status = result.get("result", {}).get("status")
    return 3 if status == "incomplete" else 2 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
