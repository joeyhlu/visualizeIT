"""Analytical/mocked ordering safeguards; never import torch or train a model."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .quality_ordering_research import (
    ARTIFACT_PURPOSE, POLICY, CHECKPOINT_SELECTION, _prepare_split,
    _union_split_documents, canonical_hash, load_research_dataset, validate_evidence,
)
from .quality_pose_ranker import ExperimentalOrderingRanker, LearnedPoseRanker, make_ranker
from .quality_ranker_data import (audit_cache, freeze_split_manifest, prepare_candidate_cache,
                                 _split_digest)
from .quality_ranker_features import FEATURE_NAMES
from .quality_ranker_train import (TrainConfig, _TrainingPurpose, _flatten, _load_dataset,
                                  _require_checkpoint_purpose, _validate_resume_identity,
                                  _ordering_completion, file_hash, ordering_train, source_identity)
from .test_quality_pose_ranker import trainer_checkpoint, rank_candidate, candidate, POSE, OTHER
from .test_quality_ranker_data import POLICY as LABEL_POLICY, _candidate, _record, _rz, _trace_and_labels

HASH = "a" * 64


def cache_source(root, suffix="", *, angular=True, val_bad=True):
    """Manufactured data only for loader tests; these artifacts are never fitted."""
    records = []
    for split, count in (("train", 2), ("validation", 2), ("test", 1), ("dev", 1)):
        records.extend(_record(f"object-{split}{suffix}", f"sequence-{split}{suffix}", split,
                               frame_id=i) for i in range(count))
    split_doc = freeze_split_manifest({"records": records, "label_policy": LABEL_POLICY})
    trace, labels = _trace_and_labels(split_doc)
    for frame in trace:
        split = next(g["split"] for g in split_doc["groups"] if g["sequence_id"] == frame["sequence_id"])
        if split == "train" and frame["frame_id"] == 0:
            frame["candidates"] = [_candidate(f"good{i}", _rz(i * .1)) for i in range(4)]
            frame["candidates"].append(_candidate("bad", _rz(60.) if angular else np.eye(4),
                                                   offset=0. if angular else .1))
        elif split == "train":
            frame["candidates"] = [_candidate("allbad", np.eye(4), offset=.1)]
        elif split == "validation":
            frame["candidates"] = [_candidate("singleton", np.eye(4),
                                               offset=.1 if val_bad and frame["frame_id"] == 1 else 0.)]
        frame["legacy_choice_candidate_id"] = frame["candidates"][0]["candidate_id"] if frame["candidates"] else None
        if frame["frame_id"] == 1:
            labels.append({"object_id": frame["object_id"], "sequence_id": frame["sequence_id"],
                           "frame_id": 1, "source_frame_id": 1, "pose": np.eye(4).tolist(),
                           "observationally_ambiguous": split == "validation",
                           "label_kind": "analytic_unit_test_reference"})
    manifest = prepare_candidate_cache(root, split_doc, trace, labels)
    extraction = {"cache_manifest_sha256": file_hash(root / "cache_manifest.json"),
                  "source_trace_sha256": manifest["source_trace_sha256"],
                  "encoder_pins": {"weights": HASH, "feature_configuration": HASH},
                  "source_receipts": {"original_seal": HASH}, "inference_sealed_before_labels": True}
    provenance = root / "extraction.json"
    provenance.write_text(json.dumps(extraction), encoding="utf-8")
    return {"manifest": str(root / "cache_manifest.json"),
            "manifest_sha256": extraction["cache_manifest_sha256"],
            "provenance": str(provenance), "provenance_sha256": file_hash(provenance)}, split_doc


def descriptor(root, sources):
    path = root / "ordering_manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "artifact_purpose": ARTIFACT_PURPOSE,
                               "sources": sources, "encoder_pins": {"weights": HASH, "feature_configuration": HASH}}),
                    encoding="utf-8")
    return path


def artifact(dataset):
    document = trainer_checkpoint()
    config = asdict(TrainConfig())
    identity = document["provenance"]["identity"]
    identity.update(purpose=ARTIFACT_PURPOSE, config=config,
                    config_sha256=canonical_hash({"purpose": ARTIFACT_PURPOSE, "config": config}),
                    ordering_evidence=dataset["ordering_evidence"],
                    ordering_evidence_sha256=canonical_hash(dataset["ordering_evidence"]),
                    data=dataset["identity"], source=source_identity(_TrainingPurpose.EXPERIMENTAL_ORDERING))
    completion = {"artifact_purpose": ARTIFACT_PURPOSE, "complete": True,
                  "checkpoint_selection": CHECKPOINT_SELECTION, "identity_sha256": canonical_hash(identity),
                  "selected_checkpoint_sha256": document["provenance"]["checkpoint_sha256"],
                  "final_checkpoint_sha256": HASH, "selected_model_sha256": HASH,
                  "selected_validation_objective": .5, "epochs_completed": 40, "optimizer_steps": 2,
                  "termination_reason": "epochs_complete", "test_read": False, "dev_read": False}
    document["provenance"].update(completion=completion, completion_sha256=canonical_hash(completion))
    document.update(schema_version=4, artifact_purpose=ARTIFACT_PURPOSE, policy=POLICY, research_only=True)
    return document


class OrderingResearchTests(unittest.TestCase):
    def test_original_substantive_gate_and_research_acceptance_test_read_trap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _ = cache_source(root / "source")
            path = descriptor(root, [source])
            self.assertFalse(audit_cache(root / "source", training_only=True)["ranking_signal_available"])
            with self.assertRaisesRegex(ValueError, "no usable ranking alternatives"):
                _load_dataset(root / "source" / "cache_manifest.json")
            original_open = Path.open
            def trap(file, *args, **kwargs):
                if file.name in {"features.test.jsonl", "labels.test.jsonl", "features.dev.jsonl", "labels.dev.jsonl"}:
                    raise AssertionError("Sealed TEST/DEV shard was opened")
                return original_open(file, *args, **kwargs)
            with patch.object(Path, "open", trap):
                dataset = _load_dataset(path, purpose=_TrainingPurpose.EXPERIMENTAL_ORDERING)
            evidence = dataset["ordering_evidence"]
            self.assertFalse(dataset["ranking_signal_available"])
            self.assertFalse(evidence["sources"][0]["ranking_signal_available"])
            self.assertTrue(evidence["purpose_eligible"])
            self.assertFalse(evidence["validation_ordering_evidence"])
            # Manufactured 4 correct hypotheses versus one 60 degree error:
            # four pair constraints share ONE frame/sequence/object unit.
            self.assertEqual(evidence["counts"]["train"]["angular_dependent_pairs"], 4)
            self.assertEqual(evidence["counts"]["train"]["angular_mixed_frame_groups"], 1)
            self.assertEqual(evidence["counts"]["train"]["angular_object_units"], 1)
            _, targets, groups = _flatten(dataset["train"], list(FEATURE_NAMES), TrainConfig())
            self.assertEqual(targets.tolist(), [1., 1., 1., 1., 0., 0.])
            self.assertEqual([len(group) for group in groups], [5, 1])
            self.assertEqual(evidence["counts"]["validation"]["orientation_ambiguous"], 1)

    def test_no_angular_pairs_or_one_class_validation_refused(self):
        for opts, message in (({"angular": False}, "observable angular TRAIN"),
                              ({"val_bad": False}, "both available-dimension VAL")):
            with self.subTest(opts=opts), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, _ = cache_source(root / "source", **opts)
                with self.assertRaisesRegex(ValueError, message):
                    load_research_dataset(descriptor(root, [source]))

    def test_complete_merge_preserves_all_splits_and_original_assignments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, doca = cache_source(root / "a", "a")
            b, docb = cache_source(root / "b", "b")
            merged = _union_split_documents([doca, docb])
            self.assertEqual({(g["physical_object_id"], g["split"]) for g in merged["groups"]},
                             {(g["physical_object_id"], g["split"]) for d in (doca, docb) for g in d["groups"]})
            dataset = load_research_dataset(descriptor(root, [a, b]))
            self.assertEqual(len(dataset["train"]), 4)
            self.assertEqual(dataset["ordering_evidence"]["counts"]["train"]["frozen_inventory_frames"], 4)
            conflict = copy.deepcopy(doca)
            for group in conflict["groups"]:
                if group["split"] == "train":
                    group["split"] = "validation"
            conflict["manifest_sha256"] = _split_digest(conflict)
            with self.assertRaisesRegex(ValueError, "contradict original split"):
                _union_split_documents([doca, conflict])

    def test_overlap_dedup_conflict_and_provenance_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, _ = cache_source(root / "a")
            b, _ = cache_source(root / "b")
            dataset = load_research_dataset(descriptor(root, [a, b]))
            self.assertEqual(len(dataset["train"]), 2)  # exact duplicate frames dedup
            c, _ = cache_source(root / "c", angular=False)
            with self.assertRaisesRegex(ValueError, "contradictory overlapping"):
                load_research_dataset(descriptor(root, [a, c]))
            changed = copy.deepcopy(a)
            changed["manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "manifest pin mismatch"):
                load_research_dataset(descriptor(root, [changed]))
            provenance = Path(a["provenance"])
            extraction = json.loads(provenance.read_text())
            extraction["encoder_pins"]["weights"] = "b" * 64
            provenance.write_text(json.dumps(extraction))
            a["provenance_sha256"] = file_hash(provenance)
            with self.assertRaisesRegex(ValueError, "encoder binding mismatch"):
                load_research_dataset(descriptor(root, [a]))

    def test_runtime_explicit_optin_roundtrip_logit_and_order_invariance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _ = cache_source(root / "source")
            document = artifact(load_research_dataset(descriptor(root, [source])))
            ranker = make_ranker("ordering-research", document)
            self.assertTrue(ranker.selection_enabled)
            self.assertEqual(ranker.to_checkpoint(), document)
            with self.assertRaisesRegex(ValueError, "ordering schema v4"):
                make_ranker("learned", document)
            for factory in (LearnedPoseRanker.from_checkpoint, lambda d: make_ranker("confidence-research", d)):
                with self.assertRaises(ValueError):
                    factory(document)
            # Analytical linear map has zero weights and zero bias: logit=0,
            # p=1/2 for every finite feature vector, with ID order breaking ties.
            rows = [rank_candidate(candidate(POSE), "z", .9), rank_candidate(candidate(OTHER), "a", .1)]
            self.assertEqual(ranker.predict_logits(rows[0].features), 0.)
            self.assertEqual(ranker.probability(rows[0].features), .5)
            self.assertEqual(ranker.rank({}, rows), ranker.rank({}, list(reversed(rows))))
            decision = ranker.rank({"recovery": True}, rows)
            self.assertIsNone(decision.selected_id)
            self.assertIn("margin_below_threshold", decision.rejection_reason)

    def test_runtime_provenance_config_completion_and_purpose_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _ = cache_source(root / "source")
            document = artifact(load_research_dataset(descriptor(root, [source])))
            mutations = [lambda d: d.update(artifact_purpose="substantive_v1"),
                         lambda d: d["provenance"]["identity"]["config"].update(pairwise_weight=0.),
                         lambda d: d["provenance"]["identity"]["ordering_evidence"]["counts"]["train"].update(angular_dependent_pairs=0),
                         lambda d: d["provenance"]["completion"].update(complete=False),
                         lambda d: d["provenance"].update(test_read=True),
                         lambda d: d["provenance"]["identity"]["ordering_evidence"]["sources"][0]["extraction_provenance"]["encoder_pins"].update(weights="b" * 64)]
            for mutate in mutations:
                bad = copy.deepcopy(document)
                mutate(bad)
                with self.assertRaises(ValueError):
                    ExperimentalOrderingRanker(bad)

    def test_checkpoint_crosspurpose_rejected_and_no_local_training(self):
        config = TrainConfig()
        identity = {"purpose": ARTIFACT_PURPOSE,
                    "config_sha256": canonical_hash({"purpose": ARTIFACT_PURPOSE, "config": asdict(config)})}
        state = {"purpose": ARTIFACT_PURPOSE, "identity": identity, "config": asdict(config)}
        _require_checkpoint_purpose(state, _TrainingPurpose.EXPERIMENTAL_ORDERING, operation="export")
        with self.assertRaisesRegex(ValueError, "purpose"):
            _require_checkpoint_purpose(state, _TrainingPurpose.SUBSTANTIVE, operation="export")
        with self.assertRaisesRegex(ValueError, "Resume purpose"):
            _validate_resume_identity(state, identity, config, _TrainingPurpose.SUBSTANTIVE)
        state["config"]["pairwise_weight"] = 0.
        with self.assertRaisesRegex(ValueError, "purpose/config"):
            _require_checkpoint_purpose(state, _TrainingPurpose.EXPERIMENTAL_ORDERING, operation="export")
        with patch("platform.system", return_value="Windows"), patch("bench.quality_ranker_train._load_dataset") as loader:
            with self.assertRaisesRegex(RuntimeError, "approved Linux cloud"):
                ordering_train(Path("absent"), Path("unused"), config, cloud_execution=True)
            loader.assert_not_called()

    def test_completion_refuses_partial_wrong_purpose_and_tampered_final(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            best, last = root / "best.pt", root / "last.pt"
            best.write_bytes(b"selected")
            last.write_bytes(b"final")
            config = TrainConfig()
            identity = {"purpose": ARTIFACT_PURPOSE,
                        "config_sha256": canonical_hash({"purpose": ARTIFACT_PURPOSE, "config": asdict(config)})}
            selected = {"purpose": ARTIFACT_PURPOSE, "identity": identity, "config": asdict(config),
                        "model": {}, "initial_model_sha256": HASH,
                        "progress": {"epoch": 1, "step": 1, "best_loss": .2, "cursor": 0, "order": []}}
            final = copy.deepcopy(selected)
            final["progress"].update(epoch=40, step=40, bad_epochs=0)
            def read(torch, path):
                return selected if path == best else final
            with patch("bench.quality_ranker_train._read_checkpoint", side_effect=read), patch(
                    "bench.quality_ranker_train._torch_state_sha256", return_value=HASH):
                receipt = _ordering_completion(SimpleNamespace(), best, config=config, write=True)
                self.assertTrue(receipt["complete"])
                self.assertEqual(_ordering_completion(SimpleNamespace(), best), receipt)
                final["progress"].update(epoch=2, step=2)
                with self.assertRaisesRegex(ValueError, "completed run"):
                    _ordering_completion(SimpleNamespace(), best)
                final["progress"].update(epoch=40, step=40)
                last.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                    _ordering_completion(SimpleNamespace(), best)
                final["purpose"] = "substantive_v1"
                with self.assertRaisesRegex(ValueError, "purpose"):
                    _ordering_completion(SimpleNamespace(), best)
                with self.assertRaisesRegex(ValueError, "validation-selected best"):
                    _ordering_completion(SimpleNamespace(), last)

    def test_ineligible_or_ambiguous_rotation_cannot_supply_ordering_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _ = cache_source(root / "source")
            dataset = load_research_dataset(descriptor(root, [source]))
            evidence = dataset["ordering_evidence"]
            for field in ("angular_mixed_frame_groups", "angular_dependent_pairs"):
                bad = copy.deepcopy(evidence)
                bad["counts"]["train"][field] = 0
                with self.assertRaises(ValueError):
                    validate_evidence(bad)
            import bench.quality_ranker_data as data
            frames, labels = data.load_cache_split(root / "source", "train")
            for label in labels:
                label.update(orientation_supervision=False, observationally_ambiguous=True, rotation_error_deg=None)
            _, counts = _prepare_split(frames, labels, "train", LABEL_POLICY)
            self.assertEqual(counts["dependent_pairs"], 4)
            self.assertEqual(counts["angular_dependent_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
