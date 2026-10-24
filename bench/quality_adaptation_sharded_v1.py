"""Frozen ownership and RAW aggregation for the expanded 13-object run.

This module imports no neural models. It authenticates ordinary child RAW
seals with the original metadata verifier, records a separate ownership
projection, and emits a new aggregate schema. It never rewrites a child seal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

from . import quality_adaptation_portable as raw
from . import quality_adaptation_portable_prepare as p
from . import quality_adaptation_scale as scale

VERSION = "visualizeit_sharded_training_v1"
TRAIN_GROUPS = {
    "smoke0": ("1", "6", "10"),
    "group1": ("13", "17", "18", "20"),
    "group2": ("21", "23", "29"),
    "group3": ("31", "32", "33"),
}
SHARD_IDS = tuple(TRAIN_GROUPS)
VALIDATION_OBJECT = "4"
TEST_OBJECT = "7"
EXPECTED_OBJECTS = tuple(sorted({VALIDATION_OBJECT, TEST_OBJECT, *(x for group in TRAIN_GROUPS.values() for x in group)}, key=int))
PARTITION_SCHEMA = "visualizeit_sharded_partition_v1"
PROJECTION_SCHEMA = "visualizeit_shard_projection_v1"
CONTINUATION_PROJECTION_SCHEMA = "visualizeit_shard_continuation_projection_v1"
AGGREGATE_SCHEMA = "visualizeit_sharded_raw_aggregate_v1"
_OWNED_CODE = ("bench/quality_adaptation_sharded_v1.py",)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _sealed(value, field):
    return {**value, field: _canonical(value)}


def _check_seal(value, field, schema):
    if not isinstance(value, dict):
        raise ValueError(f"invalid {schema} seal")
    body = {k: v for k, v in value.items() if k != field and not (schema == AGGREGATE_SCHEMA and k == "seal_sha256")}
    if value.get("schema") != schema or value.get(field) != _canonical(body):
        raise ValueError(f"invalid {schema} seal")
    return value


def _pin_code(root):
    root = p.no_links(Path(root).absolute())
    values = {}
    for name in _OWNED_CODE:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("required shard bridge source missing: " + name)
        values[name] = p.pin(path)
    return values


def _verify_code_pins(value, source_root):
    if not isinstance(value, dict) or set(value) != set(_OWNED_CODE):
        raise ValueError("exact sharded bridge source closure required")
    source_root = p.no_links(Path(source_root).absolute())
    for name in _OWNED_CODE:
        pin = value[name]
        path = p.verify_prepared_source(pin, source_root, name)
        if path != source_root / name:
            raise ValueError("sharded bridge source path differs: " + name)
    return value


def _asset_maps(recipe):
    assets = recipe.get("assets")
    if not isinstance(assets, list):
        raise ValueError("complete full asset list required")
    numeric, physical = {}, {}
    for item in assets:
        if not isinstance(item, dict):
            raise ValueError("invalid asset metadata")
        obj, phys = item.get("object_id"), item.get("physical_object_id")
        if not isinstance(obj, str) or obj in numeric or not isinstance(phys, str) or phys in physical:
            raise ValueError("duplicate or invalid object identity")
        numeric[obj] = item
        physical[phys] = obj
    if tuple(sorted(numeric, key=int)) != EXPECTED_OBJECTS:
        raise ValueError("expanded run requires the frozen 13 TRAIN + VAL4 + TEST7 object inventory")
    return numeric, physical


def _expected_roles():
    result = {obj: "train" for group in TRAIN_GROUPS.values() for obj in group}
    result[VALIDATION_OBJECT] = "validation"
    result[TEST_OBJECT] = "test"
    return result


def _states_by_id(plan):
    return p.indexed(plan["global_manifest"]["records"], "global plan states")


def _portable_state_identity(state, physical_to_numeric):
    """Project the frozen scale row into the unchanged portable RAW schema."""
    physical = state.get("physical_object_id")
    if physical not in physical_to_numeric:
        raise ValueError("scale state has foreign physical object identity")
    return {"id": state["id"], "object_id": physical_to_numeric[physical],
        "split": state["split"], "sequence_id": state["sequence_id"],
        "capture_group": state["scene_id"], "frame_id": state["frame_id"],
        # Existing portable RAW fixes a single captured source frame at t=0;
        # the scale planner does not carry a timestamp field.
        "timestamp_s": 0.0, "iteration": state["refinement_state"]}


def _portable_frame_identity(state, physical_to_numeric):
    projected = _portable_state_identity(state, physical_to_numeric)
    return {key: projected[key] for key in
        ("object_id", "sequence_id", "capture_group", "split", "frame_id", "timestamp_s")} | {
            "physical_object_id": state["physical_object_id"]}


def _projected_recipe(recipe, objects):
    keep = set(objects) | {VALIDATION_OBJECT, TEST_OBJECT}
    result = {**recipe, "assets": [a for a in recipe["assets"] if a["object_id"] in keep]}
    if {a["object_id"] for a in result["assets"]} != keep:
        raise ValueError("child recipe projection omitted a frozen object")
    return result


def _projected_audit(audit, recipe):
    selected = {a["physical_object_id"] for a in recipe["assets"]}
    result = {**audit, "recipe_sha256": _canonical(recipe),
              "assets": [a for a in audit["assets"] if a.get("physical_object_id") in selected]}
    if len(result["assets"]) != len(selected):
        raise ValueError("exposure audit cannot project every child object")
    return result


def _projection_rows(full_plan, local_plan, physical_to_numeric, shard_id):
    global_rows = _states_by_id(full_plan)
    local_rows = _states_by_id(local_plan)
    owned_objects = set(TRAIN_GROUPS[shard_id]) | ({VALIDATION_OBJECT, TEST_OBJECT} if shard_id == "smoke0" else set())
    expected_local_objects = set(TRAIN_GROUPS[shard_id]) | {VALIDATION_OBJECT, TEST_OBJECT}
    expected_ids = {ident for ident, row in global_rows.items() if physical_to_numeric[row["physical_object_id"]] in expected_local_objects}
    if set(local_rows) != expected_ids:
        raise ValueError("child local state inventory differs from fixed group projection")
    rows = []
    expected_role = _expected_roles()
    for ident in sorted(expected_ids):
        global_row, local_row = global_rows[ident], local_rows[ident]
        obj = physical_to_numeric[global_row["physical_object_id"]]
        if global_row["split"] != expected_role[obj] or local_row["split"] != global_row["split"]:
            raise ValueError("child split projection differs from frozen global role")
        if {k: v for k, v in local_row.items() if k != "split"} != {k: v for k, v in global_row.items() if k != "split"}:
            raise ValueError("child state identity differs from global recipe")
        owner = "smoke0" if obj in TRAIN_GROUPS["smoke0"] or obj in (VALIDATION_OBJECT, TEST_OBJECT) else next(
            key for key, group in TRAIN_GROUPS.items() if obj in group)
        rows.append({"id": ident, "object_id": obj, "split": global_row["split"],
                     "owned": owner == shard_id, "owner": owner,
                     "auxiliary_reason": None if owner == shard_id else "duplicate_anchor_not_global_owner"})
    if not owned_objects:
        raise ValueError("shard must own at least one frozen TRAIN object")
    return rows


def _load_imported_shard0(existing_shard0):
    if not isinstance(existing_shard0, dict) or set(existing_shard0) != {"context", "raw_seal"}:
        raise ValueError("imported smoke requires exact context and raw seal pins")
    context_pin, seal_pin = existing_shard0["context"], existing_shard0["raw_seal"]
    context, state = p.prepared_raw_state(context_pin, seal_pin, verify_payloads=False)
    return context, state


def freeze_partition(full_recipe, exposure_audit, existing_shard0, destination):
    """Freeze global ownership and all child role projections before payload work.

    `existing_shard0` is `{context: file_pin, raw_seal: file_pin}` for the
    completed original smoke. The return value pins the frozen partition and
    its newly issued smoke projection receipt.
    """
    if not isinstance(full_recipe, dict) or full_recipe.get("schema") != "kaggle_raw_recipe_v1":
        raise ValueError("full expanded RAW recipe required")
    if full_recipe.get("iterations") != [0] or full_recipe.get("seed") != 521:
        raise ValueError("expanded protocol freezes seed 521 and iteration [0]")
    numeric, physical = _asset_maps(full_recipe)
    full_plan = scale.validate_plan(raw.prospective(full_recipe))
    bounded = p.bounded_recipe(full_recipe, exposure_audit)
    if bounded["plan"] != full_plan:
        raise ValueError("exposure and global prospective plan differ")
    roles = _expected_roles()
    observed = {physical_id: role for physical_id, role in ((row["physical_object_id"], row["split"])
        for row in full_plan["global_manifest"]["records"])}
    if observed != {next(k for k, v in physical.items() if v == obj): role for obj, role in roles.items()}:
        raise ValueError("global plan roles differ from frozen ownership protocol")
    context0, smoke = _load_imported_shard0(existing_shard0)
    smoke_assets = {a["object_id"] for a in smoke["recipe"]["assets"]}
    expected_smoke = set(TRAIN_GROUPS["smoke0"]) | {VALIDATION_OBJECT, TEST_OBJECT}
    if smoke_assets != expected_smoke or smoke["recipe"].get("iterations") != [0] or smoke["recipe"].get("seed") != 521:
        raise ValueError("completed smoke shard does not project the frozen initial ownership")
    children = {}
    for shard_id, train_objects in TRAIN_GROUPS.items():
        local_recipe = _projected_recipe(full_recipe, train_objects)
        local_audit = _projected_audit(exposure_audit, local_recipe)
        local_plan = scale.validate_plan(raw.prospective(local_recipe))
        p.bounded_recipe(local_recipe, local_audit)
        projection = _projection_rows(full_plan, local_plan, physical, shard_id)
        children[shard_id] = {
            "train_objects": list(train_objects), "recipe": local_recipe,
            "recipe_sha256": _canonical(local_recipe), "exposure_audit": local_audit,
            "expected_plan_sha256": local_plan["plan_sha256"],
            "projection_rows": projection,
            "owned_state_ids": [r["id"] for r in projection if r["owned"]],
            "auxiliary_state_ids": [r["id"] for r in projection if not r["owned"]],
        }
    source_root = Path(context0.document()["source_root"])
    code_pins = _pin_code(source_root)
    root = p.no_links(Path(destination).absolute())
    if root.exists():
        raise FileExistsError("fresh shard partition directory required")
    root.mkdir(parents=True)
    smoke_input = {"context": existing_shard0["context"], "raw_seal": existing_shard0["raw_seal"],
                   "context_sha256": context0.sha256, "raw_seal_sha256": smoke["seal"]["seal_sha256"]}
    document = _sealed({
        "schema": PARTITION_SCHEMA, "version": VERSION,
        "full_recipe": full_recipe, "full_recipe_sha256": _canonical(full_recipe),
        "full_plan": full_plan, "full_plan_sha256": full_plan["plan_sha256"],
        "global_split": full_plan["global_manifest"]["split_manifest"],
        "global_split_sha256": _canonical(full_plan["global_manifest"]["split_manifest"]),
        "exposure_audit": exposure_audit, "exposure_audit_sha256": _canonical(exposure_audit),
        "owner_groups": {k: list(v) for k, v in TRAIN_GROUPS.items()},
        "anchor_owners": {VALIDATION_OBJECT: "smoke0", TEST_OBJECT: "smoke0"},
        "children": children, "imported_shard0": smoke_input,
        "source_files": code_pins,
        "fixed_disclosure": {"validation_is_reused_development": True,
            "validation_rows_owned_by": "smoke0", "duplicate_validation_is_ignored": True,
            "test_payload_reads": 0, "independent_quality_claim": False},
    }, "partition_sha256")
    partition_path = root / "partition.json"
    p.atomic_json(partition_path, document)
    partition_pin = p.pin(partition_path)
    smoke_projection = seal_shard(partition_pin, "smoke0", existing_shard0["context"],
        existing_shard0["raw_seal"], root / "shard-smoke0-projection.json")
    return {"partition": partition_pin, "imported_shard0_projection": smoke_projection,
            "children": children, "schema": PARTITION_SCHEMA}


def verify_partition(partition_pin, *, source_migration=None):
    with p.source_migration_scope(source_migration):
        return _verify_partition(partition_pin)


def _verify_partition(partition_pin):
    doc = p.metadata(partition_pin)
    _check_seal(doc, "partition_sha256", PARTITION_SCHEMA)
    if (doc.get("version") != VERSION or doc.get("owner_groups") != {k: list(v) for k, v in TRAIN_GROUPS.items()}
            or doc.get("anchor_owners") != {VALIDATION_OBJECT: "smoke0", TEST_OBJECT: "smoke0"}
            or doc.get("fixed_disclosure", {}).get("test_payload_reads") != 0):
        raise ValueError("partition protocol changed")
    imported_context = doc.get("imported_shard0", {}).get("context", {})
    source_root = raw.read_context(p.verify_pin(imported_context), imported_context["sha256"]).document()["source_root"]
    _verify_code_pins(doc.get("source_files"), source_root)
    recipe = doc["full_recipe"]
    if _canonical(recipe) != doc["full_recipe_sha256"]:
        raise ValueError("partition full recipe hash changed")
    plan = scale.validate_plan(doc["full_plan"])
    if (plan["plan_sha256"] != doc["full_plan_sha256"]
            or plan["global_manifest"]["split_manifest"] != doc["global_split"]
            or _canonical(doc["global_split"]) != doc["global_split_sha256"]
            or _canonical(doc["exposure_audit"]) != doc["exposure_audit_sha256"]):
        raise ValueError("partition plan, split or audit changed")
    p.bounded_recipe(recipe, doc["exposure_audit"])
    numeric, physical = _asset_maps(recipe)
    roles = _expected_roles()
    expected_by_physical = {next(k for k, v in physical.items() if v == obj): role for obj, role in roles.items()}
    if {r["physical_object_id"]: r["split"] for r in plan["global_manifest"]["records"]} != expected_by_physical:
        raise ValueError("partition global role map changed")
    if set(doc["children"]) != set(SHARD_IDS):
        raise ValueError("partition child inventory incomplete")
    for shard_id, group in TRAIN_GROUPS.items():
        child = doc["children"][shard_id]
        projected = _projected_recipe(recipe, group)
        if child.get("recipe") != projected or child.get("recipe_sha256") != _canonical(projected):
            raise ValueError("partition child recipe changed")
        p.bounded_recipe(projected, child["exposure_audit"])
        local_plan = scale.validate_plan(raw.prospective(projected))
        rows = _projection_rows(plan, local_plan, physical, shard_id)
        if (child.get("projection_rows") != rows or child.get("expected_plan_sha256") != local_plan["plan_sha256"]
                or child.get("owned_state_ids") != [r["id"] for r in rows if r["owned"]]
                or child.get("auxiliary_state_ids") != [r["id"] for r in rows if not r["owned"]]):
            raise ValueError("partition exact ownership/projection changed")
    context_pin = doc["imported_shard0"]["context"]
    seal_pin = doc["imported_shard0"]["raw_seal"]
    context, smoke = p.prepared_raw_state(context_pin, seal_pin, verify_payloads=False)
    smoke_expected = doc["children"]["smoke0"]
    if (smoke["recipe"] != smoke_expected["recipe"]
            or smoke["seal"]["seal_sha256"] != doc["imported_shard0"]["raw_seal_sha256"]
            or context.sha256 != doc["imported_shard0"]["context_sha256"]):
        raise ValueError("imported smoke identity changed")
    _verify_raw_projection(smoke, smoke_expected, doc, "smoke0")
    return doc


def _verify_raw_projection(raw_state, child, partition, shard_id):
    if raw_state["recipe"] != child["recipe"]:
        raise ValueError("child recipe differs from frozen partition")
    global_rows = _states_by_id(partition["full_plan"])
    local_rows = _states_by_id(raw_state["plan"])
    expected_projection = {r["id"]: r for r in child["projection_rows"]}
    if set(local_rows) != set(expected_projection):
        raise ValueError("child rows do not exactly cover frozen local projection")
    _, physical_to_numeric = _asset_maps(partition["full_recipe"])
    actual_captures = p.indexed(raw_state["captures"], "child captures")
    actual_physical = {r["sequence_id"]: r for r in raw_state["physical"]}
    for ident, expected in expected_projection.items():
        state = local_rows[ident]
        global_state = global_rows[ident]
        capture = actual_captures[ident]
        raw_identity = _portable_state_identity(global_state, physical_to_numeric)
        if (state["split"] != expected["split"] or state["split"] != global_state["split"]
                or any(capture.get(k) != value for k, value in raw_identity.items())):
            raise ValueError("child row split/state identity changed")
        if state["split"] == "test" and (capture.get("status") != "unavailable"
                or capture.get("query_rgb_sha256") is not None
                or any(key in capture for key in ("crops", "capture"))):
            raise ValueError("sealed TEST rows must remain payload-free metadata")
        frame = actual_physical.get(state["sequence_id"])
        if (frame is None or any(frame.get(k) != value for k, value in
                _portable_frame_identity(global_state, physical_to_numeric).items())):
            raise ValueError("child physical frame missing or reassigned")
        if not expected["owned"] and expected["split"] not in ("validation", "test"):
            raise ValueError("only fixed sealed validation/TEST anchors may be duplicated as auxiliary metadata")
    return raw_state


def seal_shard(partition_pin, shard_id, context_pin, raw_seal_pin, destination, *,
               raw_continuation=None, source_migration=None):
    with p.source_migration_scope(source_migration):
        return _seal_shard(partition_pin, shard_id, context_pin, raw_seal_pin, destination,
            raw_continuation=raw_continuation)


def _seal_shard(partition_pin, shard_id, context_pin, raw_seal_pin, destination, *, raw_continuation=None):
    """Bind a genuine completed child RAW result to its prospective projection."""
    partition = verify_partition(partition_pin)
    if shard_id not in partition["children"]:
        raise ValueError("unknown frozen shard id")
    if shard_id == "smoke0" and (context_pin != partition["imported_shard0"]["context"]
            or raw_seal_pin != partition["imported_shard0"]["raw_seal"]):
        raise ValueError("smoke0 source pins differ from imported completed smoke")
    context, raw_state = p.prepared_raw_state(context_pin, raw_seal_pin,
        raw_continuation=raw_continuation, verify_payloads=False)
    child = partition["children"][shard_id]
    _verify_raw_projection(raw_state, child, partition, shard_id)
    root = p.no_links(Path(destination).absolute())
    if root.exists():
        raise FileExistsError("fresh shard projection receipt path required")
    continuation = {} if raw_continuation is None else {"raw_continuation": raw_continuation}
    if p.active_source_migration() is not None:
        continuation["source_migration"] = p.active_source_migration()
    receipt = _sealed({
        "schema": CONTINUATION_PROJECTION_SCHEMA if continuation else PROJECTION_SCHEMA,
        **continuation, "partition_sha256": partition["partition_sha256"],
        "shard_id": shard_id, "source_files": partition["source_files"],
        "context": context_pin, "context_sha256": context.sha256,
        "raw_seal": raw_seal_pin, "raw_seal_sha256": raw_state["seal"]["seal_sha256"],
        "child_recipe_sha256": child["recipe_sha256"],
        "child_plan_sha256": raw_state["plan"]["plan_sha256"],
        "projection_rows": child["projection_rows"],
        "owned_state_ids": child["owned_state_ids"],
        "auxiliary_state_ids": child["auxiliary_state_ids"],
        "test_payload_reads": 0, "payloads_opened": False,
    }, "projection_sha256")
    p.atomic_json(root, receipt)
    return p.pin(root)


def verify_shard(partition_pin, shard_id, context_pin, raw_seal_pin, projection_pin, *, payloads=False,
                 raw_continuation=None, source_migration=None):
    projection = p.metadata(projection_pin)
    if raw_continuation is not None and projection.get("raw_continuation") != raw_continuation:
        raise ValueError("shard continuation descriptor differs from its projection")
    if source_migration is not None and projection.get("source_migration", source_migration) != source_migration:
        raise ValueError("shard source migration differs from its projection")
    with p.source_migration_scope(source_migration or projection.get("source_migration")):
        return _verify_shard(partition_pin, shard_id, context_pin, raw_seal_pin, projection_pin,
            payloads=payloads)


def _verify_shard(partition_pin, shard_id, context_pin, raw_seal_pin, projection_pin, *, payloads=False):
    partition = verify_partition(partition_pin)
    if shard_id not in partition["children"]:
        raise ValueError("unknown frozen shard id")
    projection = p.metadata(projection_pin)
    extended = "raw_continuation" in projection or "source_migration" in projection
    _check_seal(projection, "projection_sha256", CONTINUATION_PROJECTION_SCHEMA if extended else PROJECTION_SCHEMA)
    child = partition["children"][shard_id]
    if (projection.get("partition_sha256") != partition["partition_sha256"]
            or projection.get("shard_id") != shard_id or projection.get("context") != context_pin
            or projection.get("raw_seal") != raw_seal_pin
            or projection.get("source_files") != partition["source_files"]
            or projection.get("child_recipe_sha256") != child["recipe_sha256"]
            or projection.get("projection_rows") != child["projection_rows"]
            or projection.get("owned_state_ids") != child["owned_state_ids"]
            or projection.get("auxiliary_state_ids") != child["auxiliary_state_ids"]
            or projection.get("test_payload_reads") != 0 or projection.get("payloads_opened") is not False):
        raise ValueError("shard projection/partition/context binding changed")
    context = raw.read_context(p.verify_pin(context_pin), context_pin["sha256"])
    _verify_code_pins(partition["source_files"], context.document()["source_root"])
    if context.sha256 != projection["context_sha256"]:
        raise ValueError("child context differs from projection receipt")
    context, state = p.prepared_raw_state(context_pin, raw_seal_pin,
        raw_continuation=projection.get("raw_continuation"), verify_payloads=False)
    if (state["seal"]["seal_sha256"] != projection["raw_seal_sha256"]
            or state["plan"]["plan_sha256"] != projection["child_plan_sha256"]):
        raise ValueError("child raw seal/plan differs from projection receipt")
    _verify_raw_projection(state, child, partition, shard_id)
    if payloads:
        _, state = p.prepared_raw_state(context_pin, raw_seal_pin,
            raw_continuation=projection.get("raw_continuation"), verify_payloads=True)
        _verify_raw_projection(state, child, partition, shard_id)
    return {"shard_id": shard_id, "context": context, "raw_state": state,
            "projection": projection, "projection_pin": projection_pin}


def _relative_descriptor(value, root):
    return {"path": Path(value["path"]).relative_to(root).as_posix(),
            "bytes": value["bytes"], "sha256": value["sha256"]}


def _link_file(value, base, destination, *, must_be_under=None):
    source = p.descriptor_path(value, base)
    if must_be_under is not None and not source.is_relative_to(Path(must_be_under)):
        raise ValueError("child source artifact escapes authenticated RAW root")
    source = p.verify_pin({**value, "path": source.as_posix()})
    target = p.no_links(Path(destination).absolute())
    if target.exists():
        raise FileExistsError("aggregate payload destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target, follow_symlinks=False)
    except OSError as exc:
        raise OSError(f"aggregate hardlink requires one filesystem: {source} -> {target}: {exc}") from exc
    info_source, info_target = source.stat(), target.stat()
    if ((info_source.st_dev, info_source.st_ino) != (info_target.st_dev, info_target.st_ino)
            or target.is_symlink() or p.sha256(target) != value["sha256"] or target.stat().st_size != value["bytes"]):
        raise ValueError("aggregate hardlink bytes/inode differ from child source")
    return target


def aggregate_raw(partition_pin, shard_refs, fresh_root, *, source_migration=None):
    with p.source_migration_scope(source_migration):
        return _aggregate_raw(partition_pin, shard_refs, fresh_root)


def _aggregate_raw(partition_pin, shard_refs, fresh_root):
    """Import only globally owned rows into a fresh RAW aggregate via hardlinks.

    `shard_refs` contains one `{shard_id, context, raw_seal, projection}` per
    frozen child. Metadata for every child and every row is checked before the
    first payload is opened. The returned mapping is compatible with the
    original numerical execution/prelabel helpers while retaining a distinct
    aggregate seal schema.
    """
    partition = verify_partition(partition_pin)
    if not isinstance(shard_refs, list) or len(shard_refs) != len(SHARD_IDS):
        raise ValueError("all four frozen shard references required")
    refs = {}
    for ref in shard_refs:
        if not isinstance(ref, dict) or set(ref) != {"shard_id", "context", "raw_seal", "projection"}:
            raise ValueError("exact child shard reference required")
        shard_id = ref["shard_id"]
        if shard_id in refs:
            raise ValueError("duplicate shard reference")
        refs[shard_id] = ref
    if set(refs) != set(SHARD_IDS):
        raise ValueError("missing or foreign shard ownership")
    # Metadata-only child verification must complete for the entire set first.
    verified = {sid: verify_shard(partition_pin, sid, value["context"], value["raw_seal"], value["projection"], payloads=False)
                for sid, value in refs.items()}
    full_plan = partition["full_plan"]
    global_rows = full_plan["global_manifest"]["records"]
    state_ids = [r["id"] for r in global_rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("global plan has duplicate state identities")
    owner_by_id = {}
    for sid, child in partition["children"].items():
        for ident in child["owned_state_ids"]:
            if ident in owner_by_id:
                raise ValueError("duplicate global state ownership")
            owner_by_id[ident] = sid
    if set(owner_by_id) != set(state_ids):
        raise ValueError("global owned rows do not cover full plan exactly")
    # Detect split aliases from child metadata/inodes before any new link/open.
    aliases = []
    selected = {}
    for ident in state_ids:
        sid = owner_by_id[ident]
        state = verified[sid]["raw_state"]
        capture = p.indexed(state["captures"], "child capture inventory")[ident]
        selected[ident] = (sid, state, capture)
        paths = [p.descriptor_path(capture[key], state["root"] / "raw") for key in ("crops", "capture")] if capture["status"] == "available" else []
        aliases.append((capture["split"], capture.get("query_rgb_sha256"), paths))
    p.reject_aliases(aliases)
    root = p.no_links(Path(fresh_root).absolute())
    if root.exists():
        raise FileExistsError("fresh global RAW aggregation root required")
    root.parent.mkdir(parents=True, exist_ok=True)
    # Hardlinks must all originate on the output filesystem. Fail before making
    # any destination if a child has been placed on another mount.
    target_device = root.parent.stat().st_dev
    for ident in state_ids:
        sid, state, capture = selected[ident]
        if capture["status"] == "available":
            for key in ("crops", "capture"):
                source = p.descriptor_path(capture[key], state["root"] / "raw")
                if source.stat().st_dev != target_device:
                    raise OSError("all child RAW inputs and the new global RAW root must share one filesystem")
        physical = next(row for row in state["physical"] if row["sequence_id"] == capture["sequence_id"])
        for kind in ("render", "bank", "mask"):
            if physical[kind]["status"] == "available":
                for key, descriptor in physical[kind].items():
                    if key == "status":
                        continue
                    source = p.descriptor_path(descriptor)
                    if source.stat().st_dev != target_device:
                        raise OSError("all child RAW inputs and the new global RAW root must share one filesystem")
    root.mkdir()
    raw_dir = root / "raw"
    raw_dir.mkdir()
    capture_rows, physical_rows, lineage_rows, artifact_pins = [], [], [], {}
    physical_source = {}
    for ident in state_ids:
        sid, state, source_capture = selected[ident]
        state_row = next(row for row in global_rows if row["id"] == ident)
        child_physical = next(row for row in state["physical"] if row["sequence_id"] == state_row["sequence_id"])
        physical_key = state_row["sequence_id"]
        physical_source.setdefault(physical_key, (sid, state, child_physical))
        capture = dict(source_capture)
        if capture["status"] == "available":
            for key in ("crops", "capture"):
                source_desc = source_capture[key]
                target = raw_dir / "states" / hashlib.sha256(ident.encode()).hexdigest() / (key + ".npz")
                linked = _link_file(source_desc, state["root"] / "raw", target, must_be_under=state["root"] / "raw")
                capture[key] = _relative_descriptor({**source_desc, "path": linked.as_posix()}, raw_dir)
                artifact_pins[linked.relative_to(root).as_posix()] = p.pin(linked)
        capture_rows.append(capture)
        lineage_rows.append({"id": ident, "owner_shard": sid,
            "child_context": verified[sid]["projection"]["context"],
            "child_raw_seal": verified[sid]["projection"]["raw_seal"],
            "projection": verified[sid]["projection_pin"],
            "original_capture_sha256": _canonical(source_capture),
            "aggregate_capture_sha256": _canonical(capture),
            "original_physical_sha256": _canonical(child_physical),
            "aggregate_physical_sha256": None,
            "owned": True})
    for sequence_id, (sid, state, source_physical) in physical_source.items():
        physical = dict(source_physical)
        for kind in ("render", "bank", "mask"):
            item = dict(source_physical[kind])
            if item["status"] == "available":
                for key, descriptor in list(item.items()):
                    if key == "status":
                        continue
                    source = p.descriptor_path(descriptor)
                    target = root / "payload" / kind / hashlib.sha256((sequence_id + "\0" + key).encode()).hexdigest() / source.name
                    # Continuation crops live in preparation-view/raw; retained
                    # render/bank/mask inputs still belong to the authenticated
                    # predecessor context. Its output_root is verified by the
                    # shared RAW reader, never supplied as a caller override.
                    linked = _link_file(descriptor, None, target,
                        must_be_under=Path(state["context"]["output_root"]))
                    item[key] = p.pin(linked)
                    artifact_pins[linked.relative_to(root).as_posix()] = p.pin(linked)
            physical[kind] = item
        physical_rows.append(physical)
    physical_order = {r["sequence_id"]: r for r in physical_rows}
    physical_rows = [physical_order[key] for key in dict.fromkeys(r["sequence_id"] for r in global_rows)]
    physical_sha_by_sequence = {r["sequence_id"]: _canonical(r) for r in physical_rows}
    for lineage in lineage_rows:
        sequence_id = next(r["sequence_id"] for r in global_rows if r["id"] == lineage["id"])
        lineage["aggregate_physical_sha256"] = physical_sha_by_sequence[sequence_id]
    # Seal metadata for the distinct aggregate producer. No test payload pin is
    # included; TEST rows are kept only as unavailable state identities.
    contexts = {sid: verified[sid]["context"] for sid in SHARD_IDS}
    source_roots = {Path(v.document()["source_root"]) for v in contexts.values()}
    devices = {v.document()["device"] for v in contexts.values()}
    if len(source_roots) != 1 or len(devices) != 1:
        raise ValueError("child source roots/runtime device identities differ")
    context_doc = {"schema": "visualizeit_sharded_aggregate_context_v1",
                   "source_root": str(next(iter(source_roots))), "device": next(iter(devices)),
                   "partition_sha256": partition["partition_sha256"],
                   "child_context_sha256": {sid: contexts[sid].sha256 for sid in SHARD_IDS}}
    split_bytes = (json.dumps(partition["global_split"], indent=2, allow_nan=False) + "\n").encode()
    (root / "split.json").write_bytes(split_bytes)
    p.atomic_json(root / "scale-plan.json", full_plan)
    p.atomic_json(root / "aggregate-context.json", context_doc)
    p.atomic_json(root / "physical-index.json", {"rows": physical_rows})
    p.atomic_json(raw_dir / "capture-index.json", {"rows": capture_rows, "training_ready": False})
    p.atomic_json(root / "aggregate-capture-receipt.json", {
        "schema": "visualizeit_sharded_capture_receipt_v1", "partition_sha256": partition["partition_sha256"],
        "owned_state_count": len(capture_rows), "test_payload_reads": 0, "training_ready": False,
        "children": {sid: {"context_sha256": verified[sid]["context"].sha256,
            "raw_seal_sha256": verified[sid]["raw_state"]["seal"]["seal_sha256"],
            "projection_sha256": verified[sid]["projection"]["projection_sha256"]} for sid in SHARD_IDS}})
    migration = {} if p.active_source_migration() is None else {"source_migration": p.active_source_migration()}
    seal = _sealed({
        **migration,
        "schema": AGGREGATE_SCHEMA, "partition_sha256": partition["partition_sha256"],
        "full_recipe_sha256": partition["full_recipe_sha256"], "full_plan_sha256": partition["full_plan_sha256"],
        "global_split_sha256": partition["global_split_sha256"], "source_files": partition["source_files"],
        "child_context_sha256": {sid: contexts[sid].sha256 for sid in SHARD_IDS},
        "child_raw_seal_sha256": {sid: verified[sid]["raw_state"]["seal"]["seal_sha256"] for sid in SHARD_IDS},
        "physical_frames": physical_rows, "refinement_states": capture_rows,
        "lineage": lineage_rows,
        "artifacts": {**artifact_pins, "scale-plan.json": p.pin(root / "scale-plan.json")},
        "global_plan_file": p.pin(root / "scale-plan.json"),
        "test_payload_reads": 0, "training_ready": False,
        "schema_disclosure": "new aggregate receipt over original child seals; not kaggle_raw_capture_seal_v1",
    }, "aggregate_sha256")
    # The unchanged numerical prelabel/execution helpers require a generic seal
    # identity field. It aliases this new aggregate digest; the schema and
    # disclosure stay explicitly distinct from the original RAW producer.
    seal["seal_sha256"] = seal["aggregate_sha256"]
    p.atomic_json(root / "raw-seal.json", seal)
    # Compatibility mapping is deliberately internal: the top-level aggregate
    # schema remains distinct and the original RAW verifier is never called on it.
    raw_state = {"context": context_doc, "recipe": partition["full_recipe"], "plan": full_plan,
        "physical": physical_rows, "captures": capture_rows, "seal": seal, "root": root,
        "closing_stage": p.pin(root / "aggregate-capture-receipt.json"),
        "partition": partition, "verified_children": verified}
    return raw_state


def verify_aggregate_raw(partition_pin, aggregate_root, *, payloads=False, source_migration=None):
    seal = p.read_json(Path(aggregate_root) / "raw-seal.json")
    if source_migration is not None and seal.get("source_migration", source_migration) != source_migration:
        raise ValueError("aggregate source migration differs from its seal")
    with p.source_migration_scope(source_migration or seal.get("source_migration")):
        return _verify_aggregate_raw(partition_pin, aggregate_root, payloads=payloads)


def _verify_aggregate_raw(partition_pin, aggregate_root, *, payloads=False):
    partition = verify_partition(partition_pin)
    root = p.no_links(Path(aggregate_root).absolute())
    seal_path = root / "raw-seal.json"
    seal = p.read_json(seal_path)
    _check_seal(seal, "aggregate_sha256", AGGREGATE_SCHEMA)
    if (seal.get("seal_sha256") != seal.get("aggregate_sha256")
            or seal.get("partition_sha256") != partition["partition_sha256"]
            or seal.get("full_recipe_sha256") != partition["full_recipe_sha256"]
            or seal.get("full_plan_sha256") != partition["full_plan_sha256"]
            or seal.get("global_split_sha256") != partition["global_split_sha256"]
            or seal.get("source_files") != partition["source_files"]
            or seal.get("test_payload_reads") != 0 or seal.get("training_ready") is not False
            or seal.get("schema_disclosure") != "new aggregate receipt over original child seals; not kaggle_raw_capture_seal_v1"):
        raise ValueError("global RAW aggregate seal differs from partition")
    imported_context = partition["imported_shard0"]["context"]
    source_root = raw.read_context(p.verify_pin(imported_context), imported_context["sha256"]).document()["source_root"]
    _verify_code_pins(seal["source_files"], source_root)
    plan = scale.validate_plan(partition["full_plan"])
    captures = p.read_json(root / "raw/capture-index.json")["rows"]
    physical = p.read_json(root / "physical-index.json")["rows"]
    capture_index, physical_index = p.indexed(captures, "aggregate captures"), {r["sequence_id"]: r for r in physical}
    state_rows = plan["global_manifest"]["records"]
    if set(capture_index) != {r["id"] for r in state_rows} or len(captures) != len(state_rows):
        raise ValueError("aggregate rows must cover global plan exactly once")
    expected_owner = {}
    for sid, child in partition["children"].items():
        for ident in child["owned_state_ids"]:
            if ident in expected_owner:
                raise ValueError("duplicate global row owner")
            expected_owner[ident] = sid
    lineage = p.indexed(seal["lineage"], "aggregate lineage")
    if set(lineage) != set(capture_index) or len(lineage) != len(seal["lineage"]):
        raise ValueError("aggregate lineage is incomplete")
    # The aggregate seal is a new schema, so authenticate its source rows by
    # replaying the original child verifier and frozen projection for every
    # shard. Complete all child metadata checks before any optional payload pass.
    child_refs = {}
    for sid in SHARD_IDS:
        owned = partition["children"][sid]["owned_state_ids"]
        if not owned:
            raise ValueError("frozen shard owns no states")
        per_row = [lineage[ident] for ident in owned]
        reference_fields = ("child_context", "child_raw_seal", "projection")
        for field in reference_fields:
            if any(row.get(field) != per_row[0].get(field) for row in per_row):
                raise ValueError("aggregate lineage mixes child pins within an owner shard")
        child_refs[sid] = {field: per_row[0][field] for field in reference_fields}
        if any(set(child_refs[sid][field]) != {"path", "bytes", "sha256"} for field in reference_fields):
            raise ValueError("aggregate child lineage requires exact pinned source receipts")
    verified = {sid: verify_shard(partition_pin, sid, pins["child_context"], pins["child_raw_seal"],
                                   pins["projection"], payloads=False)
                for sid, pins in child_refs.items()}
    if any(seal.get("child_raw_seal_sha256", {}).get(sid) != verified[sid]["raw_state"]["seal"]["seal_sha256"]
           or seal.get("child_context_sha256", {}).get(sid) != verified[sid]["context"].sha256
           for sid in SHARD_IDS):
        raise ValueError("aggregate seal does not bind verified child RAW/source identities")
    aliases = []
    _, physical_to_numeric = _asset_maps(partition["full_recipe"])
    for state in state_rows:
        row = capture_index[state["id"]]
        item = lineage[state["id"]]
        if (item.get("owner_shard") != expected_owner[state["id"]] or item.get("owned") is not True
                or item.get("aggregate_capture_sha256") != _canonical(row)
                or item.get("aggregate_physical_sha256") != _canonical(physical_index.get(state["sequence_id"]))):
            raise ValueError("aggregate row owner/lineage binding changed")
        child = verified[item["owner_shard"]]
        source_capture = p.indexed(child["raw_state"]["captures"], "verified owner capture").get(state["id"])
        source_physical = next((value for value in child["raw_state"]["physical"]
                                if value["sequence_id"] == state["sequence_id"]), None)
        if (source_capture is None or source_physical is None
                or item.get("child_context") != child["projection"]["context"]
                or item.get("child_raw_seal") != child["projection"]["raw_seal"]
                or item.get("projection") != child["projection_pin"]
                or item.get("original_capture_sha256") != _canonical(source_capture)
                or item.get("original_physical_sha256") != _canonical(source_physical)):
            raise ValueError("aggregate row does not preserve original child capture/physical lineage")
        expected_identity = _portable_state_identity(state, physical_to_numeric)
        if any(row.get(k) != value for k, value in expected_identity.items()):
            raise ValueError("aggregate capture identity differs from global plan")
        physical_row = physical_index.get(state["sequence_id"])
        expected_frame = _portable_frame_identity(state, physical_to_numeric)
        if (physical_row is None or any(physical_row.get(k) != value for k, value in expected_frame.items())):
            raise ValueError("global physical row missing or role differs")
        source_frame = next((value for value in child["raw_state"]["physical"]
                             if value["sequence_id"] == state["sequence_id"]), None)
        if source_frame is None:
            raise ValueError("owner child physical frame is missing")
        for kind in ("render", "bank", "mask", "seed"):
            source_kind, target_kind = source_frame[kind], physical_row[kind]
            if source_kind.get("status") != target_kind.get("status"):
                raise ValueError("aggregate physical availability differs from original owner: " + kind)
            if kind == "seed" or source_kind.get("status") != "available":
                if source_kind != target_kind:
                    raise ValueError("aggregate non-rebased physical metadata differs: " + kind)
                continue
            if source_kind.get("status") == "available":
                source_desc = {k: v for k, v in source_kind.items() if k != "status"}
                target_desc = {k: v for k, v in target_kind.items() if k != "status"}
                if set(source_desc) != set(target_desc):
                    raise ValueError("aggregate physical descriptor inventory differs: " + kind)
                for key, original in source_desc.items():
                    rebased = target_desc[key]
                    if (not isinstance(original, dict) or not isinstance(rebased, dict)
                            or original.get("bytes") != rebased.get("bytes")
                            or original.get("sha256") != rebased.get("sha256")):
                        raise ValueError("aggregate physical descriptor bytes/hash differ: " + kind)
                    path = p.descriptor_path(rebased)
                    if not path.is_relative_to(root) or path.is_symlink():
                        raise ValueError("aggregate physical descriptor escapes root or is a symlink")
        if state["split"] == "test" and (row["status"] != "unavailable" or row.get("query_rgb_sha256") is not None
                or any(k in row for k in ("crops", "capture"))
                or any(physical_row[k] != {"status": "pending"} for k in ("render", "bank", "mask", "seed"))):
            raise ValueError("sealed TEST payload descriptor/open forbidden")
        if row["status"] == "available":
            for key in ("crops", "capture"):
                path = p.descriptor_path(row[key], root / "raw")
                aliases.append((row["split"], row.get("query_rgb_sha256"), [path]))
                if not path.is_relative_to(root):
                    raise ValueError("aggregate capture payload escapes its root")
        elif row["status"] != "unavailable" or not row.get("failure_reason"):
            raise ValueError("aggregate availability failure reason missing")
    p.reject_aliases(aliases)
    if set(physical_index) != {r["sequence_id"] for r in state_rows}:
        raise ValueError("aggregate physical row coverage differs")
    plan_file = seal.get("global_plan_file")
    if (not isinstance(plan_file, dict) or set(plan_file) != {"path", "bytes", "sha256"}
            or p.verify_pin(plan_file) != root / "scale-plan.json"
            or p.metadata(plan_file) != plan
            or seal.get("artifacts", {}).get("scale-plan.json") != plan_file):
        raise ValueError("aggregate scale-plan file differs from frozen global plan")
    for state in state_rows:
        item = lineage[state["id"]]
        child = verified[item["owner_shard"]]
        source_capture = p.indexed(child["raw_state"]["captures"], "verified owner capture")[state["id"]]
        aggregate_capture = capture_index[state["id"]]
        if source_capture.get("status") == "available":
            if aggregate_capture.get("status") != "available":
                raise ValueError("aggregate availability dropped an available owned source row")
            for key in ("crops", "capture"):
                if (aggregate_capture[key].get("bytes") != source_capture[key].get("bytes")
                        or aggregate_capture[key].get("sha256") != source_capture[key].get("sha256")):
                    raise ValueError("aggregate crop/geometry descriptor differs from original owner")
        elif aggregate_capture.get("status") != "unavailable":
            raise ValueError("aggregate availability exceeds its original owner")
    if payloads:
        # Optional child payload hashing comes only after all aggregate metadata,
        # sealed TEST rows, descriptors, lineage and alias checks have passed.
        verified = {sid: verify_shard(partition_pin, sid, pins["child_context"], pins["child_raw_seal"],
                                       pins["projection"], payloads=True)
                    for sid, pins in child_refs.items()}
        for relative, value in seal["artifacts"].items():
            path = p.verify_pin(value)
            if path != root / relative or not path.is_relative_to(root):
                raise ValueError("aggregate artifact pin escapes root")
        # Verify the exact two per-state descriptors after full metadata joining.
        for row in captures:
            if row["status"] == "available":
                for key in ("crops", "capture"):
                    p.verify_pin(row[key], root / "raw")
    return {"context": p.read_json(root / "aggregate-context.json"),
        "recipe": partition["full_recipe"], "plan": plan, "physical": physical,
        "captures": captures, "seal": seal, "root": root,
        "closing_stage": p.pin(root / "aggregate-capture-receipt.json"), "partition": partition}
