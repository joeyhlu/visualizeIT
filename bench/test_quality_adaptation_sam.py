"""SAM adapter metadata/configuration tests; no model or optimizer is run."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from bench.quality_adaptation_data import sha256
from bench.quality_adaptation_runtime import ROOT
from bench.quality_adaptation_sam import (SamConfig, TEMPLATE, build_sam_config, segmentation_scores,
                                          trainer_class, validate_sam_dataset, run)


class SamConfigTests(unittest.TestCase):
    def test_actual_upstream_template_wiring(self):
        source = ROOT / ".cache/model-quality/sources/sam2" / TEMPLATE
        template = yaml.safe_load(source.read_text())
        result = build_sam_config(template, Path("/data/manifest.json"), Path("/runs/sam"), SamConfig())
        trainer = result["trainer"]
        self.assertEqual(trainer["_target_"], "training.trainer.Trainer")
        self.assertEqual(trainer["model"]["_target_"], "training.model.sam2.SAM2Train")
        self.assertEqual(trainer["model"]["num_init_cond_frames_for_train"], 1)
        self.assertEqual(trainer["distributed"]["backend"], "gloo")
        val = trainer["data"]["val"]["datasets"][0]
        self.assertEqual(val["sampler"]["_target_"], "training.dataset.vos_sampler.EvalSampler")
        self.assertEqual(Path(val["video_dataset"]["file_list_txt"]), Path("/runs/sam/val.txt"))
        self.assertTrue(val["transforms"][0]["transforms"][0]["consistent_transform"])
        self.assertFalse(trainer["data"]["train"]["datasets"][0]["dataset"]["datasets"][0]["training"])
        self.assertFalse(result["submitit"]["use_cluster"])
        self.assertEqual(template["trainer"]["accelerator"], "cuda")  # Input template not mutated.

    def test_config_rejects_invalid_bound_and_amp(self):
        SamConfig().validate()
        for cfg in (replace(SamConfig(), amp=True), replace(SamConfig(), train_frames=1),
                    replace(SamConfig(), max_frames_per_sequence=100), replace(SamConfig(), learning_rate=float("nan"))):
            with self.assertRaises(ValueError): cfg.validate()

    def test_explicit_opt_in_required_before_model_or_dataset(self):
        with patch("bench.quality_adaptation_sam.cloud_guard"):
            with self.assertRaisesRegex(ValueError, "enable-sam-adaptation"):
                run(Path("missing"), Path("missing"), SamConfig(), cloud_execution=True)

    def test_iou_dice_analytical_sets_and_empty_masks(self):
        a = np.array([1, 1, 0, 0], bool)
        b = np.array([0, 1, 1, 0], bool)
        result = segmentation_scores(a, b)
        # Set intersection1, union3; Dice=2*1/(2+2)=1/2.
        self.assertEqual(result, {"iou": 1/3, "dice": .5})
        self.assertEqual(segmentation_scores(np.zeros(4, bool), np.zeros(4, bool)), {"iou": 1., "dice": 1.})

    def test_only_mask_decoder_trainable_and_upstream_optimizer_called(self):
        params = {name: SimpleNamespace(requires_grad=True) for name in ("image_encoder.x", "sam_mask_decoder.x", "memory_encoder.x")}
        for p in params.values():
            p.requires_grad_ = lambda value, p=p: setattr(p, "requires_grad", value)
        class Base:
            def __init__(self, **kwargs):
                self.model = SimpleNamespace(named_parameters=lambda: params.items())
                self.upstream_calls = 0
            def _construct_optimizers(self):
                self.upstream_calls += 1
        child = trainer_class(Base, output=Path("unused"), cfg=SamConfig(), identity="test",
                              validation_objects={}, prior_elapsed=0, torch=None)()
        child._construct_optimizers()
        self.assertEqual(child.upstream_calls, 1)
        self.assertEqual({k for k, p in params.items() if p.requires_grad}, {"sam_mask_decoder.x"})


class SamDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({"schema": "segmentation_bottleneck_v1", "reviewed": True,
                                       "reason": "independent annotated mask errors", "baseline_iou": .6,
                                       "evaluation_protocol": "independent_masks_initial_prompt_only"}))
        self.doc = {"schema": "sam2_adaptation_dataset_v1", "label_origin": "independent_annotations",
                    "segmentation_bottleneck": {"path": evidence.name, "sha256": sha256(evidence), "bytes": evidence.stat().st_size},
                    "sequences": []}
        for index, split in enumerate(("train", "val", "test")):
            row = {"id": f"seq{index}", "object_id": f"obj{index}", "capture_group": f"group{index}",
                   "split": split, "status": "available", "frames": []}
            for frame in range(2):
                row["frames"].append({"rgb": {"path": f"images/seq{index}/{frame:05d}.jpg", "sha256": f"{index * 10 + frame + 1:064x}", "bytes": 100},
                                      "mask": {"path": f"masks/seq{index}/{frame:05d}.png", "sha256": "a" * 64, "bytes": 100}})
            self.doc["sequences"].append(row)
        self.path = self.root / "manifest.json"

    def validate(self):
        self.path.write_text(json.dumps(self.doc))
        return validate_sam_dataset(self.path, SamConfig(), inspect_images=False)

    def test_metadata_audit_preserves_sealed_test_inventory(self):
        result = self.validate()
        self.assertEqual(result["availability"]["test"], {"total": 1, "available": 1})
        self.assertFalse(result["test_arrays_opened"])

    def test_object_and_capture_leakage(self):
        self.doc["sequences"][1]["object_id"] = "obj0"
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.validate()

    def test_rgb_duplicate_leakage(self):
        self.doc["sequences"][2]["frames"][0]["rgb"]["sha256"] = self.doc["sequences"][0]["frames"][0]["rgb"]["sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate RGB"):
            self.validate()

    def test_davis_layout_and_independent_labels_required(self):
        self.doc["sequences"][0]["frames"][0]["mask"]["path"] = "masks/seq0/other.png"
        with self.assertRaisesRegex(ValueError, "DAVIS"):
            self.validate()

    def test_bottleneck_evidence_hash_required(self):
        (self.root / "evidence.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
