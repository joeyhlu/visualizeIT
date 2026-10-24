"""Join audited frozen-feature receipts without rerunning feature extraction.

The production union combines the current 95-row training inventory and the
retained 72-row TRAIN cache. Source manifests, recipes, indices, and optional
final-request records are copied byte-for-byte under ``lineage/``. Only the
selected feature/label payloads are linked or copied into the new dataset.
"""
from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time

import numpy as np

from . import quality_adaptation_data as data_module
from .quality_adaptation_data import (CONVENTION, FEATURE_KEYS, HOOKS,
                                      LABEL_KEYS, MAX_ARCHIVE, MAX_JSON, SCHEMA,
                                      canonical_hash, flow_metrics, load_dataset,
                                      read_json, require_hash, sha256,
                                      visibility_metrics)
from .quality_adaptation_runtime import LOCK


OPERATOR_SCHEMA = "frozen_gotrack_decoder_cache_union_operator_v1"
UNION_RECIPE_SCHEMA = "frozen_gotrack_decoder_cache_union_v1"
UNION_INDEX_SCHEMA = "frozen_gotrack_decoder_cache_union_index_v1"
UNION_RECEIPT_SCHEMA = "frozen_gotrack_decoder_cache_union_receipt_v1"
MIN_FREE_BYTES = 1024**3
MAX_UNION_PAYLOAD_BYTES = 2 * 1024**3
MAX_ARCHIVE_IDENTITY_BYTES = 8 * 1024**3
CHILD_NAMES = ("current", "old72")
OLD72_OBJECT_IDS = {"1", "6", "10"}

SELECTIONS = {
    "current": {"schema": "all_inventory_rows_v1",
                "splits": {"train": "all", "val": "all", "test": "all"}},
    "old72": {"schema": "train_object_subset_v1", "split": "train",
              "object_ids": ["1", "6", "10"],
              "other_splits": "metadata_only"},
}

EXPECTED_CHILD_COUNTS = {
    "current": {"train": (96, 95), "val": (24, 20), "test": (24, 0)},
    "old72": {"train": (72, 72)},
}
EXPECTED_OBJECTS = {
    "current": {"train": {"13", "17", "18", "20"},
                "val": {"4"}, "test": {"7"}},
    "old72": {"train": OLD72_OBJECT_IDS},
}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, label: str):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value in {label}: {value}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON: {label}") from exc


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _publish_validated_file(partial: Path, final: Path) -> None:
    """Publish an already-validated file, with a restricted-Windows fallback.

    Some Windows sandbox temp roots deny rename operations even within the same
    directory. The exclusive-create fallback is safe for these fresh outputs:
    validation has completed, the destination did not exist, and the final file
    remains the last artifact written. Keep the validated partial on failure.
    """
    try:
        os.replace(partial, final)
        return
    except PermissionError:
        pass
    created = False
    try:
        with partial.open("rb") as source, final.open("xb") as target:
            created = True
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        if created:
            try:
                final.unlink()
            except OSError:
                pass
        raise
    partial.unlink()


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _assert_no_links(path: Path) -> None:
    """Reject symlinks/junctions in every existing component of a path."""
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if _is_link(current):
            raise ValueError(f"symlinks and junctions are prohibited: {current}")


def _root_path(base: Path, value, label: str, *, must_exist=True) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    _assert_no_links(path)
    path = path.resolve(strict=must_exist)
    if must_exist and not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    return path


def _relative_file(root: Path, value, label: str) -> Path:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value):
        raise ValueError(f"{label} must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts) or ":" in value:
        raise ValueError(f"{label} escapes its declared root")
    target = root.joinpath(*relative.parts)
    _assert_no_links(target)
    resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"{label} escapes its declared root")
    return target


def _read_pinned_file(root: Path, pin: dict, label: str, *, path_key=None,
                      max_bytes=MAX_JSON) -> tuple[bytes, dict]:
    required = {"bytes", "sha256"} | ({"path"} if path_key else set())
    if not isinstance(pin, dict) or set(pin) != required:
        raise ValueError(f"{label} file pin fields differ from contract")
    size = pin["bytes"]
    if type(size) is not int or not 0 < size <= max_bytes:
        raise ValueError(f"{label} bytes outside bounded file size")
    digest = require_hash(pin["sha256"], f"{label} sha256")
    rel = pin["path"] if path_key else f"{label}.json"
    path = _relative_file(root, rel, label)
    if not path.is_file():
        raise ValueError(f"missing pinned {label}: {path}")
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"pinned {label} bytes/hash mismatch")
    return raw, {"path": rel, "bytes": size, "sha256": digest}


def _validate_archive_identity(value: dict, name: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"sha256", "bytes", "max_bytes"}:
        raise ValueError(f"{name} archive identity/bounds differ from contract")
    digest = require_hash(value["sha256"], f"{name} archive sha256")
    actual, maximum = value["bytes"], value["max_bytes"]
    if (type(actual) is not int or type(maximum) is not int or actual <= 0
            or maximum < actual or maximum > MAX_ARCHIVE_IDENTITY_BYTES):
        raise ValueError(f"{name} archive identity exceeds its pinned byte bounds")
    return {"sha256": digest, "bytes": actual, "max_bytes": maximum}


def _validate_transport_identity(value: dict, name: str) -> dict:
    if (not isinstance(value, dict)
            or set(value) != {"schema", "identity_sha256", "artifact_id"}
            or value.get("schema") != "existing_private_backup_transport_v1"):
        raise ValueError(f"{name} transport identity differs from retained-backup contract")
    require_hash(value["identity_sha256"], f"{name} transport identity sha256")
    if not isinstance(value["artifact_id"], str) or not value["artifact_id"]:
        raise ValueError(f"{name} transport artifact id is required")
    return copy.deepcopy(value)


def _validate_expected_child(child: dict, name: str) -> None:
    rows = child["dataset"].document["rows"]
    for split, (total, available) in EXPECTED_CHILD_COUNTS[name].items():
        split_rows = [row for row in rows if row["split"] == split]
        if (len(split_rows) != total
                or sum(row["status"] == "available" for row in split_rows) != available):
            raise ValueError(f"{name} {split} inventory count differs from frozen union contract")
        object_ids = {row["object_id"] for row in split_rows}
        if object_ids != EXPECTED_OBJECTS[name][split]:
            raise ValueError(f"{name} {split} object inventory differs from frozen union contract")
    if name == "old72":
        selected = [row for row in rows if row["split"] == "train"]
        if any(row["object_id"] not in OLD72_OBJECT_IDS for row in selected):
            raise ValueError("old72 selection includes an object outside {1,6,10}")


def _load_child(spec: dict, name: str, spec_root: Path) -> dict:
    allowed = {"name", "metadata_root", "files", "archive", "transport_identity"}
    if "payload_root" in spec:
        allowed.add("payload_root")
    if set(spec) != allowed or spec.get("name") != name:
        raise ValueError(f"{name} operator child fields differ from contract")
    metadata_root = _root_path(spec_root, spec["metadata_root"], f"{name} metadata_root")
    payload_root = _root_path(spec_root, spec.get("payload_root", spec["metadata_root"]),
                              f"{name} payload_root")
    files = spec["files"]
    if not isinstance(files, dict) or set(files) not in ({"manifest", "recipe", "index"},
                                                          {"manifest", "recipe", "index", "final_request"}):
        raise ValueError(f"{name} metadata file pins differ from contract")
    metadata = {}
    raw_metadata = {}
    for file_name in ("manifest", "recipe", "index"):
        raw, pin = _read_pinned_file(metadata_root, files[file_name], file_name)
        metadata[file_name] = pin
        raw_metadata[file_name] = raw
    if "final_request" in files:
        raw, pin = _read_pinned_file(metadata_root, files["final_request"], "final_request",
                                     path_key=True)
        metadata["final_request"] = pin
        raw_metadata["final_request"] = raw
    dataset_manifest = _decode_json(raw_metadata["manifest"], f"{name} manifest")
    source_recipe = _decode_json(raw_metadata["recipe"], f"{name} recipe")
    source_index = _decode_json(raw_metadata["index"], f"{name} index")
    archive = _validate_archive_identity(spec["archive"], name)
    transport = _validate_transport_identity(spec["transport_identity"], name)

    # load_dataset(inspect_arrays=False) and the original receipt validator bind
    # child metadata while deliberately leaving every payload unopened.
    from . import quality_adaptation_train as train_module
    source_ds = load_dataset(metadata_root / "manifest.json", inspect_arrays=False)
    train_module.validate_feature_receipt(source_ds)
    if source_ds.document != dataset_manifest:
        raise ValueError(f"{name} source manifest changed while receipt was validated")
    if source_recipe.get("annotation_inputs") != []:
        raise ValueError(f"{name} feature recipe has annotation inputs")
    if (source_index.get("recipe_sha256") != canonical_hash(source_recipe)
            or source_recipe.get("schema") != "frozen_gotrack_decoder_recipe_v1"
            or source_recipe.get("hooks") != list(HOOKS)
            or source_recipe.get("annotation_inputs") != []
            or dataset_manifest.get("provenance", {}).get("label_origin")
               != "synthetic_renderer_training_only"):
        raise ValueError(f"{name} original feature recipe/index identity mismatch")
    return {
        "name": name, "metadata_root": metadata_root, "payload_root": payload_root,
        "metadata": metadata, "raw_metadata": raw_metadata,
        "dataset": source_ds, "recipe": source_recipe, "index": source_index,
        "archive": archive, "transport_identity": transport,
    }


def _same_except_crop_manifest(recipes: list[dict]) -> dict:
    if len(recipes) != 2:
        raise ValueError("exactly two child recipes are required")
    common = []
    for recipe in recipes:
        if "crop_manifest_sha256" not in recipe:
            raise ValueError("child recipe lacks crop_manifest_sha256")
        require_hash(recipe["crop_manifest_sha256"], "crop_manifest_sha256")
        value = copy.deepcopy(recipe)
        del value["crop_manifest_sha256"]
        common.append(value)
    if common[0] != common[1]:
        raise ValueError("child recipes may differ only in crop_manifest_sha256")
    return common[0]


def _selected(name: str, row: dict) -> bool:
    if name == "current":
        return True
    return row["split"] == "train" and row["object_id"] in OLD72_OBJECT_IDS


def _payload_relative(child: str, row_id: str, kind: str) -> str:
    row_hash = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    filename = "features.npz" if kind == "features" else "labels.npz"
    return f"payloads/{child}/{row_hash}/{filename}"


def _validate_payload_descriptor(desc: dict, name: str) -> None:
    if not isinstance(desc, dict) or set(desc) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{name} payload descriptor differs from frozen dataset contract")
    require_hash(desc["sha256"], f"{name} payload sha256")
    if type(desc["bytes"]) is not int or not 0 < desc["bytes"] <= MAX_ARCHIVE:
        raise ValueError(f"{name} payload size outside bounded archive limit")


def _source_payload(child: dict, desc: dict, row_id: str, kind: str,
                    destination_root: Path, source_identities: set[tuple[int, int]]) -> dict:
    _validate_payload_descriptor(desc, f"{child['name']} {row_id} {kind}")
    source = _relative_file(child["payload_root"], desc["path"],
                            f"{child['name']} {row_id} {kind} source")
    if not source.is_file():
        raise ValueError(f"missing selected payload: {source}")
    info = source.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != desc["bytes"]:
        raise ValueError(f"selected payload byte length/type mismatch: {source}")
    inode_key = (info.st_dev, info.st_ino)
    if inode_key in source_identities:
        raise ValueError("selected payload source files alias the same inode")
    source_identities.add(inode_key)
    if sha256(source) != desc["sha256"]:
        raise ValueError(f"selected payload hash mismatch: {source}")
    target_rel = _payload_relative(child["name"], row_id, kind)
    target = _relative_file(destination_root, target_rel,
                            f"{child['name']} {row_id} {kind} destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP,
                             getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)):
            raise
        _copy_bounded(source, target, desc["bytes"], desc["sha256"])
    if target.stat().st_size != desc["bytes"] or sha256(target) != desc["sha256"]:
        raise ValueError(f"translated payload identity mismatch: {target}")
    return {"path": target_rel, "sha256": desc["sha256"], "bytes": desc["bytes"]}


def _copy_bounded(source: Path, target: Path, size: int, expected_hash: str) -> None:
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as src, target.open("xb") as out:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            copied += len(block)
            if copied > size:
                raise ValueError("selected payload grew beyond its pinned byte bound")
            digest.update(block)
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    if copied != size or digest.hexdigest() != expected_hash:
        raise ValueError("copied payload bytes/hash differ from source descriptor")


def _translate_row(child: dict, row: dict, destination_root: Path,
                   source_identities: set[tuple[int, int]], *, copy_payloads: bool) -> tuple[dict, dict]:
    selected = _selected(child["name"], row)
    index_row = {
        "original_row": copy.deepcopy(row), "selected": selected,
        "translated_features": None, "translated_labels": None,
    }
    if not selected:
        return None, index_row
    translated = copy.deepcopy(row)
    if row["status"] == "available":
        descriptors = {}
        for kind in ("features", "labels"):
            _validate_payload_descriptor(row[kind], f"{child['name']} {row['id']} {kind}")
            relative = _payload_relative(child["name"], row["id"], kind)
            descriptors[kind] = {"path": relative, "sha256": row[kind]["sha256"],
                                 "bytes": row[kind]["bytes"]}
            if copy_payloads:
                actual = _source_payload(child, row[kind], row["id"], kind,
                                         destination_root, source_identities)
                if actual != descriptors[kind]:
                    raise ValueError("translated descriptor differs from deterministic path contract")
        translated.update(descriptors)
        index_row["translated_features"] = descriptors["features"]
        index_row["translated_labels"] = descriptors["labels"]
    return translated, index_row


def _child_record(child: dict) -> dict:
    recipe = child["recipe"]
    metadata = {}
    for name, pin in child["metadata"].items():
        filename = "final-request.json" if name == "final_request" else f"{name}.json"
        metadata[name] = {"path": f"lineage/{child['name']}/{filename}",
                          "bytes": pin["bytes"], "sha256": pin["sha256"]}
    return {
        "name": child["name"],
        "metadata": metadata,
        "recipe_sha256": canonical_hash(recipe),
        "index_sha256": child["metadata"]["index"]["sha256"],
        "archive": child["archive"],
        "transport_identity": child["transport_identity"],
        "selection": SELECTIONS[child["name"]],
    }


def _assemble_documents(children: list[dict], destination_root: Path | None,
                         *, copy_payloads=False) -> tuple[dict, dict, dict, int, set[tuple[int, int]]]:
    if [child["name"] for child in children] != list(CHILD_NAMES):
        raise ValueError("children must be ordered current, old72")
    common = _same_except_crop_manifest([child["recipe"] for child in children])
    origins = {child["dataset"].document["provenance"]["label_origin"] for child in children}
    if origins != {"synthetic_renderer_training_only"}:
        raise ValueError("union requires the original synthetic_renderer_training_only label origin")
    rows, index_children = [], []
    seen_ids = set()
    source_identities = set()
    copy_bytes = 0
    for child in children:
        _validate_expected_child(child, child["name"])
        index_rows = []
        for source_row in child["dataset"].document["rows"]:
            translated, index_row = _translate_row(
                child, source_row, destination_root, source_identities,
                copy_payloads=copy_payloads)
            index_rows.append(index_row)
            if translated is None:
                continue
            if translated["id"] in seen_ids:
                raise ValueError(f"child row id collision: {translated['id']}")
            seen_ids.add(translated["id"])
            if translated["status"] == "available":
                copy_bytes += translated["features"]["bytes"] + translated["labels"]["bytes"]
                if copy_bytes > MAX_UNION_PAYLOAD_BYTES:
                    raise ValueError("selected payload bytes exceed the union cache cap")
            rows.append(translated)
        index_children.append({
            "name": child["name"], "metadata": _child_record(child)["metadata"],
            "recipe_sha256": canonical_hash(child["recipe"]),
            "index_sha256": child["metadata"]["index"]["sha256"],
            "rows": index_rows,
        })
    _validate_union_counts(rows)
    child_records = [_child_record(child) for child in children]
    recipe = {
        "schema": UNION_RECIPE_SCHEMA,
        "extraction_performed": False,
        "annotation_inputs": [],
        "label_origin": "synthetic_renderer_training_only",
        "common_extraction_recipe": common,
        "children": child_records,
    }
    recipe_sha = canonical_hash(recipe)
    index = {"schema": UNION_INDEX_SCHEMA, "recipe_sha256": recipe_sha,
             "children": index_children}
    document = {
        "schema": SCHEMA, "conventions": CONVENTION,
        "provenance": {"feature_recipe_sha256": recipe_sha,
                       "source_lock_sha256": sha256(LOCK),
                       "annotation_inputs": [],
                       "label_origin": "synthetic_renderer_training_only"},
        "rows": rows,
    }
    return document, recipe, index, copy_bytes, source_identities


def _validate_union_counts(rows: list[dict]) -> None:
    expected = {"train": (168, 167), "val": (24, 20), "test": (24, 0)}
    expected_objects = {"train": {"1", "6", "10", "13", "17", "18", "20"},
                        "val": {"4"}, "test": {"7"}}
    for split, (total, available) in expected.items():
        selected = [row for row in rows if row["split"] == split]
        if (len(selected) != total
                or sum(row["status"] == "available" for row in selected) != available
                or {row["object_id"] for row in selected} != expected_objects[split]):
            raise ValueError(f"union {split} counts/object identities differ from frozen contract")


def _validate_child_metadata(root: Path, record: dict) -> tuple[dict, dict, dict, object]:
    name = record["name"]
    if name not in CHILD_NAMES:
        raise ValueError("union receipt contains an unknown child")
    lineage = _relative_file(root, f"lineage/{name}", f"{name} lineage")
    if not lineage.is_dir():
        raise ValueError(f"missing {name} byte-identical lineage directory")
    metadata_pins = record.get("metadata")
    required = {"manifest", "recipe", "index"}
    if isinstance(metadata_pins, dict) and "final_request" in metadata_pins:
        required.add("final_request")
    if not isinstance(metadata_pins, dict) or set(metadata_pins) != required:
        raise ValueError(f"{name} lineage metadata pins differ from receipt")
    raw = {}
    for key in ("manifest", "recipe", "index"):
        source = metadata_pins[key]
        if not isinstance(source, dict) or set(source) != {"path", "bytes", "sha256"}:
            raise ValueError(f"{name} {key} lineage pin malformed")
        expected_rel = f"lineage/{name}/{key}.json"
        if source["path"] != expected_rel:
            raise ValueError(f"{name} lineage metadata path is not canonical")
        path = _relative_file(root, expected_rel, f"{name} {key}")
        if not path.is_file() or path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
            raise ValueError(f"{name} original metadata bytes changed")
        raw[key] = _decode_json(path.read_bytes(), f"{name} {key}")
    if "final_request" in metadata_pins:
        pin = metadata_pins["final_request"]
        if (not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}
                or pin["path"] != f"lineage/{name}/final-request.json"):
            raise ValueError(f"{name} final-request lineage pin malformed")
        path = _relative_file(root, pin["path"], f"{name} final request")
        if not path.is_file() or path.stat().st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
            raise ValueError(f"{name} final-request audit bytes changed")
    from . import quality_adaptation_train as train_module
    child_ds = load_dataset(lineage / "manifest.json", inspect_arrays=False)
    train_module.validate_feature_receipt(child_ds)
    if child_ds.document != raw["manifest"]:
        raise ValueError(f"{name} original metadata parse differs from child receipt")
    if (record.get("recipe_sha256") != canonical_hash(raw["recipe"])
            or record.get("index_sha256") != metadata_pins["index"]["sha256"]
            or raw["index"].get("recipe_sha256") != record.get("recipe_sha256")
            or raw["recipe"].get("annotation_inputs") != []
            or raw["manifest"].get("provenance", {}).get("label_origin")
               != "synthetic_renderer_training_only"):
        raise ValueError(f"{name} original recipe/index pin mismatch")
    return raw["manifest"], raw["recipe"], raw["index"], child_ds


def validate_union_receipt(ds, *, payload_scope=None) -> dict:
    """Validate union metadata, original child receipts, and selected bytes.

    This function hashes selected files but never opens NPZ members. In
    particular, old72 validation/test payloads remain untouched.
    """
    if not hasattr(ds, "root") or not hasattr(ds, "document"):
        raise ValueError("a loaded correspondence dataset is required")
    root = Path(ds.root).resolve()
    recipe_path = _relative_file(root, "recipe.json", "union recipe")
    index_path = _relative_file(root, "index.json", "union index")
    recipe = read_json(recipe_path)
    index = read_json(index_path)
    if (not isinstance(recipe, dict) or recipe.get("schema") != UNION_RECIPE_SCHEMA
            or recipe.get("extraction_performed") is not False
            or recipe.get("annotation_inputs") != []
            or recipe.get("label_origin") != "synthetic_renderer_training_only"):
        raise ValueError("dataset is not a frozen-feature union receipt")
    provenance = ds.document.get("provenance", {})
    if (provenance.get("source_lock_sha256") != sha256(LOCK)
            or provenance.get("annotation_inputs") != []
            or provenance.get("label_origin") != "synthetic_renderer_training_only"):
        raise ValueError("union provenance source/origin differs from the pinned source contract")
    recipe_sha = canonical_hash(recipe)
    if (ds.document.get("provenance", {}).get("feature_recipe_sha256") != recipe_sha
            or index.get("schema") != UNION_INDEX_SCHEMA
            or index.get("recipe_sha256") != recipe_sha
            or ds.identity != canonical_hash(ds.document)):
        raise ValueError("union manifest/index/recipe identity mismatch")
    records = recipe.get("children")
    index_children = index.get("children")
    if (not isinstance(records, list) or [x.get("name") for x in records] != list(CHILD_NAMES)
            or not isinstance(index_children, list)
            or [x.get("name") for x in index_children] != list(CHILD_NAMES)):
        raise ValueError("union receipt must contain current and old72 lineage in canonical order")
    original_recipes, child_rows = [], []
    if payload_scope is not None:
        allowed_scope = {("old72", "train")}
        if payload_scope != allowed_scope:
            raise ValueError("restricted union payload validation only permits old72 TRAIN")
    total_payload_bytes = 0
    payloads_verified = 0
    source_identities = set()
    for record, index_child in zip(records, index_children):
        if (not isinstance(record, dict)
                or record.get("name") not in CHILD_NAMES
                or set(record) != {"name", "metadata", "recipe_sha256", "index_sha256",
                                   "archive", "transport_identity", "selection"}
                or record["selection"] != SELECTIONS[record["name"]]):
            raise ValueError("child selection/archive/transport receipt fields differ from contract")
        _validate_archive_identity(record["archive"], record["name"])
        _validate_transport_identity(record["transport_identity"], record["name"])
        original_manifest, original_recipe, original_index, child_ds = _validate_child_metadata(root, record)
        original_recipes.append(original_recipe)
        if (not isinstance(index_child, dict)
                or set(index_child) != {"name", "metadata", "recipe_sha256", "index_sha256", "rows"}
                or index_child["metadata"] != record["metadata"]
                or index_child["recipe_sha256"] != record["recipe_sha256"]
                or index_child["index_sha256"] != record["index_sha256"]
                or len(index_child["rows"]) != len(original_manifest["rows"])):
            raise ValueError("union index does not bind original child rows and metadata")
        for source_row, index_row in zip(original_manifest["rows"], index_child["rows"]):
            expected_selected = _selected(record["name"], source_row)
            if (not isinstance(index_row, dict)
                    or set(index_row) != {"original_row", "selected", "translated_features", "translated_labels"}
                    or index_row["original_row"] != source_row
                    or index_row["selected"] is not expected_selected):
                raise ValueError("union index omitted/changed original row selection metadata")
            translated = copy.deepcopy(source_row) if expected_selected else None
            expected_feature = expected_label = None
            if expected_selected and source_row["status"] == "available":
                for kind in ("features", "labels"):
                    desc = source_row[kind]
                    _validate_payload_descriptor(desc, f"{record['name']} {source_row['id']} {kind}")
                    rel = _payload_relative(record["name"], source_row["id"], kind)
                    expected_desc = {"path": rel, "sha256": desc["sha256"], "bytes": desc["bytes"]}
                    if kind == "features":
                        expected_feature = expected_desc
                    else:
                        expected_label = expected_desc
                    verify_payload = (payload_scope is None
                                      or (record["name"], source_row["split"]) in payload_scope)
                    if verify_payload:
                        target = _relative_file(root, rel, f"union {record['name']} {source_row['id']} {kind}")
                        if not target.is_file():
                            raise ValueError(f"missing selected union payload: {rel}")
                        info = target.stat()
                        inode = (info.st_dev, info.st_ino)
                        if inode in source_identities:
                            raise ValueError("translated payloads alias the same inode")
                        source_identities.add(inode)
                        if info.st_size != desc["bytes"] or sha256(target) != desc["sha256"]:
                            raise ValueError(f"translated payload hash/size mismatch: {rel}")
                        total_payload_bytes += desc["bytes"]
                        payloads_verified += 1
                        if total_payload_bytes > MAX_UNION_PAYLOAD_BYTES:
                            raise ValueError("selected payload bytes exceed the union cache cap")
                    translated[kind] = expected_desc
            if (index_row["translated_features"] != expected_feature
                    or index_row["translated_labels"] != expected_label):
                raise ValueError("union index translated descriptor mismatch")
            if translated is not None:
                child_rows.append(translated)
        _validate_expected_child({"dataset": child_ds}, record["name"])
    if _same_except_crop_manifest(original_recipes) != recipe.get("common_extraction_recipe"):
        raise ValueError("child recipes differ beyond crop_manifest_sha256")
    if child_rows != ds.document.get("rows"):
        raise ValueError("union dataset rows differ from deterministic child selection")
    _validate_union_counts(child_rows)
    return {"schema": UNION_RECEIPT_SCHEMA, "recipe_sha256": recipe_sha,
            "dataset_sha256": ds.identity, "children": list(CHILD_NAMES),
            "selected_payload_bytes": total_payload_bytes,
            "payload_files_verified": payloads_verified,
            "payload_scope": "all_selected" if payload_scope is None else "old72_train_only",
            "selected_rows": len(child_rows),
            "train_available": sum(r["split"] == "train" and r["status"] == "available" for r in child_rows),
            "validation_available": sum(r["split"] == "val" and r["status"] == "available" for r in child_rows),
            "test_payload_reads": 0}


def assemble_union(spec_path: Path, destination: Path) -> dict:
    """Assemble the fixed current95 + old72 cache into a fresh output root."""
    spec_path = Path(spec_path)
    destination = Path(destination)
    if _is_link(spec_path) or not spec_path.is_file():
        raise ValueError("operator spec must be a regular nonsymlink file")
    spec = read_json(spec_path)
    if (not isinstance(spec, dict) or set(spec) != {"schema", "children"}
            or spec.get("schema") != OPERATOR_SCHEMA
            or not isinstance(spec.get("children"), list)
            or len(spec["children"]) != 2):
        raise ValueError("invalid frozen-feature union operator spec")
    spec_root = spec_path.resolve().parent
    children_by_name = {}
    for raw_child in spec["children"]:
        if not isinstance(raw_child, dict) or raw_child.get("name") not in CHILD_NAMES:
            raise ValueError("operator spec has an unknown child")
        name = raw_child["name"]
        if name in children_by_name:
            raise ValueError("operator spec repeats a child")
        children_by_name[name] = _load_child(raw_child, name, spec_root)
    if set(children_by_name) != set(CHILD_NAMES):
        raise ValueError("operator spec must pin exactly current and old72")
    children = [children_by_name[name] for name in CHILD_NAMES]
    for child in children:
        _validate_expected_child(child, child["name"])

    destination = Path(os.path.abspath(destination))
    if destination.exists() or _is_link(destination):
        raise ValueError("union destination must be fresh")
    parent = destination.parent
    _assert_no_links(parent)
    if not parent.is_dir():
        raise ValueError("union destination parent must already exist")
    # Build documents first to calculate the worst-case cross-filesystem copy.
    document, recipe, index, copy_bytes, _ = _assemble_documents(children, None)
    if copy_bytes > MAX_UNION_PAYLOAD_BYTES:
        raise ValueError("selected payload bytes exceed the union cache cap")
    if shutil.disk_usage(parent).free < copy_bytes + MIN_FREE_BYTES:
        raise ValueError("insufficient free storage to preserve the 1 GiB reserve")

    destination.mkdir(parents=False, exist_ok=False)
    # Preserve source metadata bytes as received; their descriptors remain bound
    # to source-relative paths, while selected output payloads get new paths.
    for child in children:
        lineage = destination / "lineage" / child["name"]
        lineage.mkdir(parents=True, exist_ok=True)
        for key in ("manifest", "recipe", "index"):
            target = lineage / f"{key}.json"
            target.write_bytes(child["raw_metadata"][key])
        if "final_request" in child["raw_metadata"]:
            (lineage / "final-request.json").write_bytes(child["raw_metadata"]["final_request"])
    # Copy/link exactly the selected descriptors and reconstruct documents with
    # destination descriptors. Failed partial outputs intentionally remain.
    document, recipe, index, _, _ = _assemble_documents(children, destination,
                                                         copy_payloads=True)
    recipe_bytes, index_bytes = _json_bytes(recipe), _json_bytes(index)
    (destination / "recipe.json").write_bytes(recipe_bytes)
    (destination / "index.json").write_bytes(index_bytes)
    # Exercise the generic loader and custom receipt on an unpublished manifest.
    temp_manifest = destination / ".manifest.partial.json"
    with temp_manifest.open("xb") as stream:
        stream.write(_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        ds = load_dataset(temp_manifest, inspect_arrays=False)
        validate_union_receipt(ds)
        _publish_validated_file(temp_manifest, destination / "manifest.json")
    except Exception:
        # Preserve the failed partial manifest and every completed payload for audit.
        raise
    final_ds = load_dataset(destination / "manifest.json", inspect_arrays=False)
    receipt = validate_union_receipt(final_ds)
    return {"manifest": str(destination / "manifest.json"),
            "manifest_sha256": sha256(destination / "manifest.json"), **receipt}


def _selected_old72_rows(ds, index: dict) -> list[dict]:
    available_ids = {row["id"] for row in ds.rows("train")}
    old_ids = [entry["original_row"]["id"] for entry in index["children"][1]["rows"]
               if entry["selected"] and entry["original_row"]["split"] == "train"
               and entry["original_row"]["status"] == "available"]
    if len(old_ids) != 72 or len(set(old_ids)) != 72:
        raise ValueError("old72 transfer diagnostic requires exactly 72 distinct TRAIN rows")
    by_id = {row["id"]: row for row in ds.rows("train", available=False)}
    if any(row_id not in available_ids for row_id in old_ids):
        raise ValueError("old72 TRAIN inventory contains unavailable or omitted rows")
    return [by_id[row_id] for row_id in old_ids]


def _write_diagnostic(output: Path, report: dict) -> None:
    output = Path(os.path.abspath(output))
    parent = output.parent
    _assert_no_links(parent)
    if not parent.is_dir() or output.exists() or _is_link(output):
        raise ValueError("diagnostic output must be a new file in an existing nonsymlink directory")
    encoded = _json_bytes(report)
    temp = parent / ("." + output.name + ".partial")
    with temp.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_validated_file(temp, output)


def transfer_diagnostic(manifest: Path, export_path: Path, expected_export_sha256: str,
                        output: Path, cloud_execution=False) -> dict:
    """Compare the pinned 95-run head against its cache baseline on old72 TRAIN."""
    from .quality_adaptation_runtime import cloud_guard
    cloud_guard(cloud_execution)
    expected = require_hash(expected_export_sha256, "expected export sha256")
    export_path = Path(export_path)
    if _is_link(export_path) or not export_path.is_file() or sha256(export_path) != expected:
        raise ValueError("transfer export is not the SHA-pinned current95 artifact")
    ds = load_dataset(Path(manifest), inspect_arrays=False)
    receipt = validate_union_receipt(ds, payload_scope={("old72", "train")})
    union_index = read_json(ds.root / "index.json")
    rows = _selected_old72_rows(ds, union_index)
    output_path = Path(output)
    if output_path.exists() or _is_link(output_path):
        raise ValueError("diagnostic output already exists")

    import torch
    artifact = torch.load(export_path, map_location="cpu", weights_only=True)
    required_artifact_keys = {"schema", "head", "source_lock_sha256", "base_checkpoint_sha256",
                              "dataset_sha256", "config_sha256", "validation", "quality_claim",
                              "deployment"}
    if (not isinstance(artifact, dict) or set(artifact) != required_artifact_keys
            or artifact.get("schema") != "correspondence_head_export_v1"
            or artifact.get("quality_claim") is not False):
        raise ValueError("export is not the ordinary strict DPT head artifact")
    current_ds = load_dataset(ds.root / "lineage/current/manifest.json",
                              inspect_arrays=False)
    if artifact.get("dataset_sha256") != current_ds.identity:
        raise ValueError("SHA-pinned transfer export is not bound to the current child receipt")
    from . import quality_adaptation_flow_terminal as flow_terminal
    head = flow_terminal.load_head(initialize=False, device="cpu")
    head.load_state_dict(artifact["head"], strict=True)
    head.eval()

    baseline_rows, adapted_rows, failures = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            row_started = time.perf_counter()
            try:
                features, labels = ds.arrays(row)
                tokens = [None] * 12
                for hook in HOOKS:
                    tokens[hook] = torch.from_numpy(features[f"tokens_{hook}"][None])
                flow, confidence = flow_terminal._model_outputs(head, tokens, torch, 1)
                prediction = flow[0].detach().cpu().numpy()
                probability = confidence[0].detach().cpu().numpy()
                adapted = {"id": row["id"], "object_id": row["object_id"],
                           **flow_metrics(prediction, labels["flow"], labels["visible"]),
                           "visibility": visibility_metrics(probability, labels)}
                baseline = {"id": row["id"], "object_id": row["object_id"],
                            **flow_metrics(features["baseline_flow"], labels["flow"], labels["visible"]),
                            "visibility": visibility_metrics(features["baseline_confidence"], labels)}
                adapted["latency_seconds"] = time.perf_counter() - row_started
                baseline_rows.append(baseline)
                adapted_rows.append(adapted)
            except Exception as exc:
                failures.append({"id": row["id"], "reason": f"{type(exc).__name__}: {exc}"})
    elapsed = time.perf_counter() - started
    report = {
        "schema": "frozen_gotrack_decoder_cache_transfer_diagnostic_v1",
        "manifest_sha256": ds.identity,
        "union_recipe_sha256": receipt["recipe_sha256"],
        "export_sha256": expected,
        "export_dataset_sha256": artifact["dataset_sha256"],
        "selection": {"child": "old72", "split": "train", "objects": ["1", "6", "10"]},
        "inventory": {"selected": 72, "available": len(rows), "evaluated": len(adapted_rows),
                      "failed": len(failures), "unavailable": 0},
        "baseline": baseline_rows, "adapted": adapted_rows, "failures": failures,
        "latency": {"elapsed_seconds": elapsed,
                    "mean_seconds_per_evaluated_row": (elapsed / len(adapted_rows)
                                                       if adapted_rows else None)},
        "validation_arrays_opened": False, "test_payload_reads": 0,
        "training_performed": False, "quality_claim": False,
    }
    _write_diagnostic(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("assemble")
    command.add_argument("--spec", type=Path, required=True)
    command.add_argument("--destination", type=Path, required=True)
    command = sub.add_parser("validate")
    command.add_argument("--manifest", type=Path, required=True)
    command = sub.add_parser("transfer-diagnostic")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--export", type=Path, required=True)
    command.add_argument("--expected-export-sha256", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--cloud-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "assemble":
        result = assemble_union(args.spec, args.destination)
    elif args.command == "validate":
        ds = load_dataset(args.manifest, inspect_arrays=False)
        result = validate_union_receipt(ds)
    else:
        result = transfer_diagnostic(args.manifest, args.export,
                                     args.expected_export_sha256, args.output,
                                     cloud_execution=args.cloud_execution)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
