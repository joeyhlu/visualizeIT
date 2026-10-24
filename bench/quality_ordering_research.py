"""Audited development ordering research; no optimizer or test-shard access.

The quality target uses only dimensions available under the frozen label policy.
Translation-only positives do not establish full rotational correctness.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .quality_ranker_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

ARTIFACT_PURPOSE = "experimental_ordering_v1"
POLICY = "eligible_candidate_ordering_v1"
QUALITY_TARGET = "frozen_available_dimension_quality"
CHECKPOINT_SELECTION = "eligible_candidate_validation_bce_plus_within_frame_pairwise"
CALIBRATION_SCOPE = "validation_eligible_labeled_available_dimensions"
MERGE_RULE = "complete_inventory_union_exact_identity_dedup_v1"
UNSUPPORTED_CLAIMS = ["heldout_ordering_improvement", "full_rotation_correctness",
                      "object_generalization", "physical_attachment_accuracy", "phone_fps"]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _union_split_documents(documents):
    """Preserve all frozen assignments and unavailable inventory, including TEST.

    Does TRAIN contain indirect information about another split? Validate the
    union using the original physical/scene/mesh/alias leakage checks as well as
    exact duplicate frame mappings. No labels or test shard contents are read.
    """
    from .quality_ranker_data import _split_digest, _sort_frame, validate_split_manifest
    result = copy.deepcopy(documents[0])
    groups, backgrounds = {}, {}
    for doc in documents:
        validate_split_manifest(doc)
        for key in set(result) - {"groups", "background_catalog", "manifest_sha256"}:
            if doc.get(key) != result[key]:
                raise ValueError("Ordering sources have incompatible split/label policies")
        for background in doc["background_catalog"]:
            key = background["background_id"]
            if key in backgrounds and backgrounds[key] != background:
                raise ValueError("Ordering sources contradict background inventory")
            backgrounds[key] = copy.deepcopy(background)
        for group in doc["groups"]:
            key = (group["physical_object_id"], group["sequence_id"])
            if key not in groups:
                groups[key] = copy.deepcopy(group)
                continue
            old = groups[key]
            if ({k: v for k, v in old.items() if k not in ("frames", "frame_ids")} !=
                    {k: v for k, v in group.items() if k not in ("frames", "frame_ids")}):
                raise ValueError("Ordering sources contradict original split/group identity")
            frames = {row["frame_id"]: row for row in old["frames"]}
            for row in group["frames"]:
                if row["frame_id"] in frames and frames[row["frame_id"]] != row:
                    raise ValueError("Ordering sources contradict frozen frame identity")
                frames[row["frame_id"]] = copy.deepcopy(row)
            old["frames"] = sorted(frames.values(), key=lambda r: _sort_frame(r["frame_id"]))
            if len({row["source_frame_id"] for row in old["frames"]}) != len(old["frames"]):
                raise ValueError("Ordering sources contradict frozen source frame identity")
            old["frame_ids"] = [r["source_frame_id"] for r in old["frames"]]
    result["groups"] = sorted(groups.values(), key=lambda g: (g["split"], g["physical_object_id"], g["sequence_id"]))
    result["background_catalog"] = sorted(backgrounds.values(), key=lambda b: b["background_id"])
    result["manifest_sha256"] = _split_digest(result)
    return validate_split_manifest(result)


def _prepare_split(frames, labels, split, policy):
    from .quality_ranker_data import _label_key
    index = {_label_key(label): label for label in labels}
    groups = []
    mixed_sequences, mixed_objects, angular_sequences, angular_objects = set(), set(), set(), set()
    counts = dict(inventory_frames=len(frames), eligible_labeled=0, eligible_good=0,
                  eligible_bad=0, orientation_observable=0, orientation_ambiguous=0,
                  mixed_frame_groups=0, dependent_pairs=0, angular_mixed_frame_groups=0,
                  angular_dependent_pairs=0, missing_labels=0)
    for frame in frames:
        provenance = frame.get("provenance_kind", frame.get("context", {}).get("provenance_kind"))
        if provenance not in ("real", "synthetic"):
            raise ValueError("Ordering research requires real/synthetic provenance; fixtures rejected")
        candidates, angular_good, angular_bad = [], 0, 0
        for candidate in frame["candidates"]:
            label = index.get((frame["object_id"], frame["sequence_id"], frame["frame_id"], candidate["candidate_id"]))
            if label is None or label.get("quality_target") is None:
                counts["missing_labels"] += 1
                continue
            target = label["quality_target"]
            if type(target) is not int or target not in (0, 1):
                raise ValueError("Ordering targets must be frozen binary labels")
            candidates.append({key: copy.deepcopy(candidate[key]) for key in
                               ("candidate_id", "features", "eligible", "pose")})
            candidates[-1]["quality_target"] = target
            if not candidate["eligible"]:
                continue
            counts["eligible_labeled"] += 1
            counts["eligible_good" if target else "eligible_bad"] += 1
            observable = label.get("orientation_supervision") is True and label.get("observationally_ambiguous") is False
            counts["orientation_observable" if observable else "orientation_ambiguous"] += 1
            rotation, translation = label.get("rotation_error_deg"), label.get("translation_error_m")
            if (observable and type(rotation) in (int, float) and math.isfinite(rotation) and
                    type(translation) in (int, float) and math.isfinite(translation) and
                    translation <= policy["max_translation_error_m"]):
                if target == 1 and rotation <= policy["max_rotation_error_deg"]:
                    angular_good += 1
                if target == 0 and rotation > policy["max_rotation_error_deg"]:
                    angular_bad += 1
        good = sum(c["eligible"] and c["quality_target"] == 1 for c in candidates)
        bad = sum(c["eligible"] and c["quality_target"] == 0 for c in candidates)
        counts["mixed_frame_groups"] += int(bool(good and bad))
        counts["dependent_pairs"] += good * bad
        counts["angular_mixed_frame_groups"] += int(bool(angular_good and angular_bad))
        counts["angular_dependent_pairs"] += angular_good * angular_bad
        if good and bad:
            mixed_sequences.add((frame["object_id"], frame["sequence_id"]))
            mixed_objects.add(frame["object_id"])
        if angular_good and angular_bad:
            angular_sequences.add((frame["object_id"], frame["sequence_id"]))
            angular_objects.add(frame["object_id"])
        groups.append({key: copy.deepcopy(frame[key]) for key in
                       ("object_id", "sequence_id", "frame_id", "source_frame_id", "context")})
        groups[-1]["candidates"] = candidates
    counts.update(mixed_sequence_units=len(mixed_sequences), mixed_object_units=len(mixed_objects),
                  angular_sequence_units=len(angular_sequences), angular_object_units=len(angular_objects))
    return groups, counts


def validate_evidence(evidence):
    fields = {"artifact_purpose", "policy", "quality_target", "merge_rule", "sources",
              "encoder_pins", "union_split_sha256", "label_policy", "label_policy_sha256",
              "ranking_signal_available", "purpose_eligible", "counts", "validation_ordering_evidence",
              "checkpoint_selection", "calibration_scope", "normalization_split", "test_read", "dev_read",
              "unsupported_claims"}
    if not isinstance(evidence, dict) or set(evidence) != fields:
        raise ValueError("Ordering evidence fields differ from the research contract")
    constants = dict(artifact_purpose=ARTIFACT_PURPOSE, policy=POLICY, quality_target=QUALITY_TARGET,
                     merge_rule=MERGE_RULE, purpose_eligible=True, checkpoint_selection=CHECKPOINT_SELECTION,
                     calibration_scope=CALIBRATION_SCOPE, normalization_split="train", test_read=False,
                     dev_read=False, unsupported_claims=UNSUPPORTED_CLAIMS)
    if any(evidence.get(k) != v for k, v in constants.items()):
        raise ValueError("Ordering evidence contradicts research purpose/scope")
    if (type(evidence["ranking_signal_available"]) is not bool or
            type(evidence["validation_ordering_evidence"]) is not bool or
            not _sha(evidence["union_split_sha256"]) or
            evidence["label_policy_sha256"] != canonical_hash(evidence["label_policy"])):
        raise ValueError("Ordering evidence policy/audit identity is malformed")
    if not isinstance(evidence["encoder_pins"], dict) or not evidence["encoder_pins"] or any(
            not _sha(v) for v in evidence["encoder_pins"].values()):
        raise ValueError("Ordering encoder pins must contain exact SHA256 identities")
    if not isinstance(evidence["sources"], list) or not evidence["sources"]:
        raise ValueError("Ordering research needs complete source identities")
    for source in evidence["sources"]:
        if (not isinstance(source, dict) or set(source) != {
                "manifest_sha256", "identity", "shards", "ranking_signal_available",
                "extraction_provenance", "extraction_provenance_sha256", "integrity_audit_sha256",
                "approved", "audit_scope"} or not _sha(source["manifest_sha256"]) or
                type(source["ranking_signal_available"]) is not bool or
                not _sha(source["integrity_audit_sha256"]) or source["approved"] is not True or
                source["audit_scope"] != "train_validation_only"):
            raise ValueError("Ordering source provenance is malformed")
        source_data = source["identity"]
        hash_fields = {"feature_cache_sha256", "label_cache_sha256", "split_sha256", "source_trace_sha256",
                       "reference_labels_sha256", "cache_manifest_sha256"}
        if (not isinstance(source_data, dict) or set(source_data) != hash_fields | {"feature_names", "feature_schema_version"} or
                source_data["feature_names"] != list(FEATURE_NAMES) or
                source_data["feature_schema_version"] != FEATURE_SCHEMA_VERSION or
                any(not _sha(source_data[name]) for name in hash_fields)):
            raise ValueError("Ordering source cache/feature identity is malformed")
        extraction = source["extraction_provenance"]
        if (not isinstance(extraction, dict) or set(extraction) != {
                "cache_manifest_sha256", "source_trace_sha256", "encoder_pins", "source_receipts",
                "inference_sealed_before_labels"} or extraction["inference_sealed_before_labels"] is not True or
                canonical_hash(extraction) != source["extraction_provenance_sha256"] or
                extraction["cache_manifest_sha256"] != source["manifest_sha256"] or
                extraction["source_trace_sha256"] != source["identity"]["source_trace_sha256"] or
                extraction["encoder_pins"] != evidence["encoder_pins"] or
                not isinstance(extraction["source_receipts"], dict) or not extraction["source_receipts"] or
                any(not _sha(value) for value in extraction["source_receipts"].values())):
            raise ValueError("Ordering extraction provenance/cache/encoder binding mismatch")
        if not isinstance(source["shards"], dict) or not all(
                key in source["shards"] for key in ("split_manifest", "features_train", "labels_train",
                                                    "features_validation", "labels_validation", "features_test", "labels_test")):
            raise ValueError("Ordering sources must bind complete shard inventories")
        if any(not _sha(row.get("sha256")) for row in source["shards"].values()):
            raise ValueError("Ordering source shard identities are malformed")
    for split in ("train", "validation"):
        counts = evidence["counts"][split]
        expected_counts = {"inventory_frames", "frozen_inventory_frames", "eligible_labeled", "eligible_good",
                           "eligible_bad", "orientation_observable", "orientation_ambiguous", "mixed_frame_groups",
                           "dependent_pairs", "angular_mixed_frame_groups", "angular_dependent_pairs", "missing_labels",
                           "mixed_sequence_units", "mixed_object_units", "angular_sequence_units", "angular_object_units"}
        if not isinstance(counts, dict) or set(counts) != expected_counts or any(
                type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("Ordering supervision counts must be nonnegative integers")
        if (counts["eligible_good"] + counts["eligible_bad"] != counts["eligible_labeled"] or
                counts["orientation_observable"] + counts["orientation_ambiguous"] != counts["eligible_labeled"] or
                not counts["angular_object_units"] <= counts["angular_sequence_units"] <= counts["angular_mixed_frame_groups"] <= counts["mixed_frame_groups"] <= counts["inventory_frames"] <= counts["frozen_inventory_frames"] or
                counts["angular_dependent_pairs"] > counts["dependent_pairs"]):
            raise ValueError("Ordering supervision count relationships are inconsistent")
    train, val = evidence["counts"]["train"], evidence["counts"]["validation"]
    if not train["angular_mixed_frame_groups"] or not train["angular_dependent_pairs"]:
        raise ValueError("Ordering research requires genuine observable angular TRAIN mixed groups")
    if not val["eligible_good"] or not val["eligible_bad"]:
        raise ValueError("Ordering calibration requires both available-dimension VAL quality classes")
    if evidence["validation_ordering_evidence"] != bool(val["mixed_frame_groups"]):
        raise ValueError("Ordering validation evidence contradicts dependent-pair counts")


def load_research_dataset(manifest):
    """Load a pinned cache or complete-cache descriptor, never TEST/DEV rows.

    Descriptor: {schema_version:1, artifact_purpose:experimental_ordering_v1,
      sources:[{manifest:path, manifest_sha256:file_digest,
                provenance:path, provenance_sha256:file_digest}, ...],
      encoder_pins:{model_or_feature_configuration:sha256, ...}}.
    All sources contribute all TRAIN/VAL rows; contradictory overlaps fail.
    """
    from .quality_ranker_data import (_cache_manifest, _safe_cache_file, audit_cache,
                                     load_cache_split, read_split_manifest)
    manifest = Path(manifest)
    descriptor = json.loads(manifest.read_text(encoding="utf-8"))
    if (set(descriptor) != {"schema_version", "artifact_purpose", "sources", "encoder_pins"} or
            descriptor["schema_version"] != 1 or descriptor["artifact_purpose"] != ARTIFACT_PURPOSE):
        raise ValueError("Ordering requires a pinned complete-cache source descriptor")
    if not isinstance(descriptor["sources"], list) or not descriptor["sources"]:
        raise ValueError("Ordering descriptor requires source caches")
    sources, docs, rows = [], [], {s: ({}, {}) for s in ("train", "validation")}
    seen_sources = set()
    for entry in descriptor["sources"]:
        if set(entry) != {"manifest", "manifest_sha256", "provenance", "provenance_sha256"}:
            raise ValueError("Ordering source descriptor fields are malformed")
        path = Path(entry["manifest"])
        if not path.is_absolute():
            path = manifest.parent / path
        if path.name != "cache_manifest.json" or path.resolve() in seen_sources:
            raise ValueError("Ordering needs unique frozen cache_manifest.json sources")
        seen_sources.add(path.resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("Ordering source manifest pin mismatch")
        audit = audit_cache(path.parent, training_only=True)
        if audit.get("approved") is not True or audit.get("audit_scope") != "train_validation_only":
            raise ValueError("Ordering research requires approved TRAIN/VAL integrity audit")
        if (audit["identity"].get("feature_names") != list(FEATURE_NAMES) or
                audit["identity"].get("feature_schema_version") != FEATURE_SCHEMA_VERSION):
            raise ValueError("Ordering frozen feature schema mismatch")
        cache, _ = _cache_manifest(path.parent)
        provenance_path = Path(entry["provenance"])
        if not provenance_path.is_absolute():
            provenance_path = manifest.parent / provenance_path
        raw_provenance = provenance_path.read_bytes()
        if hashlib.sha256(raw_provenance).hexdigest() != entry["provenance_sha256"]:
            raise ValueError("Ordering extraction provenance file pin mismatch")
        extraction = json.loads(raw_provenance)
        docs.append(read_split_manifest(_safe_cache_file(path.parent, cache["files"]["split_manifest"])))
        sources.append({"manifest_sha256": entry["manifest_sha256"], "identity": audit["identity"],
                        "shards": cache["files"], "ranking_signal_available": audit["ranking_signal_available"],
                        "extraction_provenance": extraction,
                        "extraction_provenance_sha256": canonical_hash(extraction),
                        "integrity_audit_sha256": canonical_hash(audit), "approved": True,
                        "audit_scope": "train_validation_only"})
        for split in ("train", "validation"):
            frames, labels = load_cache_split(path.parent, split)
            for values, table, key_names in ((frames, rows[split][0], ("object_id", "sequence_id", "frame_id")),
                                            (labels, rows[split][1], ("object_id", "sequence_id", "frame_id", "candidate_id"))):
                for row in values:
                    key = tuple(row[name] for name in key_names)
                    if key in table and table[key] != row:
                        raise ValueError("Ordering sources contain contradictory overlapping frame/label rows")
                    table[key] = row
    union = _union_split_documents(docs)
    prepared, counts = {}, {}
    for split in ("train", "validation"):
        prepared[split], counts[split] = _prepare_split(list(rows[split][0].values()),
                                                      list(rows[split][1].values()), split, union["label_policy"])
        counts[split]["frozen_inventory_frames"] = sum(len(group["frames"]) for group in union["groups"]
                                                       if group["split"] == split)
    evidence = dict(artifact_purpose=ARTIFACT_PURPOSE, policy=POLICY, quality_target=QUALITY_TARGET,
                    merge_rule=MERGE_RULE, sources=sources, encoder_pins=descriptor["encoder_pins"],
                    union_split_sha256=union["manifest_sha256"], label_policy=union["label_policy"],
                    label_policy_sha256=canonical_hash(union["label_policy"]),
                    ranking_signal_available=bool(counts["train"]["mixed_frame_groups"] and counts["validation"]["mixed_frame_groups"]),
                    purpose_eligible=True, counts=counts,
                    validation_ordering_evidence=bool(counts["validation"]["mixed_frame_groups"]),
                    checkpoint_selection=CHECKPOINT_SELECTION, calibration_scope=CALIBRATION_SCOPE,
                    normalization_split="train", test_read=False, dev_read=False, unsupported_claims=UNSUPPORTED_CLAIMS)
    validate_evidence(evidence)
    # Compound identities bind complete source inventories and the unmodified
    # source hashes; these are composite digests, never rewritten cache claims.
    identity = composite_data_identity(evidence)
    return {**prepared, "identity": identity, "feature_names": list(FEATURE_NAMES),
            "feature_schema_version": FEATURE_SCHEMA_VERSION, "ordering_evidence": evidence,
            "ranking_signal_available": evidence["ranking_signal_available"]}


def composite_data_identity(evidence):
    """Bind every original cache identity in the fixed source descriptor order."""
    sources = evidence["sources"]
    identity = {key: canonical_hash([source["identity"][key] for source in sources])
                for key in sources[0]["identity"] if key not in ("feature_names", "feature_schema_version")}
    identity.update(feature_names=list(FEATURE_NAMES), feature_schema_version=FEATURE_SCHEMA_VERSION)
    return identity
