"""CPU fixture coverage for observed CNOS initialization; no model is run."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_ranker_initialization as initialization
from .quality_ranker_data import freeze_split_manifest
from .quality_ranker_synthetic import make_input_manifest
from .quality_time import PHYSICAL, parse_timing_manifest


def _raw_bundle(root):
    bundle = root / "raw"
    bundle.mkdir()
    mesh_bytes, video_bytes = b"analytic metric asset fixture", b"analytic video fixture"
    (bundle / "object.glb").write_bytes(mesh_bytes)
    (bundle / "synthetic.mp4").write_bytes(video_bytes)
    mesh_sha = hashlib.sha256(mesh_bytes).hexdigest()
    video_sha = hashlib.sha256(video_bytes).hexdigest()
    frozen = freeze_split_manifest({
        "records": [{
            "physical_object_id": "synthetic_fixture_object",
            "object_id": "synthetic_fixture_object",
            "scene_id": "fixture-scene", "sequence_id": "fixture-sequence",
            "background_id": "procedural:gray", "background_kind": "procedural",
            "provenance_kind": "synthetic", "mesh_sha256": mesh_sha,
            "frame_id": ordinal, "source_frame_id": ordinal, "split": "train",
        } for ordinal in range(2)],
        "label_policy": {"max_rotation_error_deg": 5., "max_translation_error_m": .05},
    })
    group = frozen["groups"][0]
    manifest = make_input_manifest(mesh_sha256=mesh_sha, video_sha256=video_sha,
        frames=2, width=64, height=64, fps=10., sequence_group=group)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    (bundle / "input.json").write_bytes(raw)
    timing = parse_timing_manifest(raw)
    detector = root / "detection.npz"
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:14, 10:14] = 255
    diagnostics = {"reason": None, "bank_sha256": "a" * 64, "score": 0.9}
    np.savez_compressed(detector,
        mask=mask, reason=np.array(""),
        diagnostics=np.array(json.dumps(diagnostics, separators=(",", ":"))),
        frameId=np.array(0, dtype=np.int64), sourceFrameId=np.array(0, dtype=np.int64),
        timestamp_s=np.array(0., dtype=np.float64), clock_mode=np.array(PHYSICAL),
        input_manifest_sha256=np.array(timing["manifest_sha256"]),
        timestamp_table_sha256=np.array(timing["timestamp_table_sha256"]))
    return bundle, detector, manifest, mask


def _review_receipt(proposal, *, decision="approve", selection=None,
                    proposal_sha256=None, reviewed_at=None):
    return {
        "schema_version": 1,
        "kind": initialization.RECEIPT_KIND,
        "proposal_sha256": proposal_sha256 or proposal["proposal_sha256"],
        "reviewed_at_utc": reviewed_at or proposal["created_at_utc"],
        "decision": decision,
        "reviewer_kind": "codex_agent",
        "reviewer_identifier": "fixture-codex-reviewer",
        "statement": initialization.REVIEW_STATEMENT,
        **{name: proposal[name] for name in (
            "source_binding", "timing_binding", "rgb_binding", "detector_binding",
            "cnos_pins", "selection", "preview_binding")},
        **({} if selection is None else {"selection": selection}),
    }


class InteriorPointTests(unittest.TestCase):
    def test_deterministic_interior_point_and_row_major_tie(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        self.assertEqual(initialization._interior_positive_point(mask), [2, 2])
        tied = np.zeros((4, 4), dtype=np.uint8)
        tied[1:3, 1:3] = 255
        self.assertEqual(initialization._interior_positive_point(tied), [1, 1])
        tiny = np.zeros((4, 4), dtype=np.uint8)
        tiny[0, 0] = 255
        self.assertEqual(initialization._interior_positive_point(tiny), [0, 0])
        self.assertIsNone(initialization._interior_positive_point(np.zeros((4, 4), dtype=np.uint8)))

    def test_point_rejects_wrong_dtype_and_nonbinary_mask(self):
        with self.assertRaisesRegex(ValueError, "uint8 binary"):
            initialization._interior_positive_point(np.ones((2, 2), dtype=np.bool_))
        with self.assertRaisesRegex(ValueError, "uint8 binary"):
            initialization._interior_positive_point(np.array([[0, 1]], dtype=np.uint8))


class InitializationProposalTests(unittest.TestCase):
    def setUp(self):
        self.preview_bytes = b"bounded RGB-observed-mask-point preview fixture"
        self.preview_binding = {
            "sha256": hashlib.sha256(self.preview_bytes).hexdigest(),
            "format": "PNG", "width": 64, "height": 64,
            "channel_order": "RGB", "overlay_rgb": [255, 32, 32],
            "overlay_mix": [0.55, 0.45], "point_rgb": [0, 255, 0],
            "point_radius_px": 3, "png_compression": 3,
        }
        self.pins = {
            "model_lock_sha256": "b" * 64,
            "checkpoints": {"FastSAM-x.pt": "c" * 64,
                            "dinov2_vitl14_pretrain.pth": "d" * 64},
            "descriptor_bank_sha256": "a" * 64,
            "descriptor_bank_asset_sha256": "e" * 64,
            "quality_cnos_source_sha256": "f" * 64,
            "quality_runner_source_sha256": "1" * 64,
        }

    def _patched_dependencies(self, rgb):
        from . import quality_runner
        return (
            patch.object(quality_runner, "read_rgb", return_value=rgb),
            patch.object(initialization, "_read_cnos_pins", return_value=self.pins),
            patch.object(initialization, "_make_preview",
                         return_value=(self.preview_bytes, self.preview_binding)),
        )

    def _create(self, root):
        bundle, detector, manifest, mask = _raw_bundle(root)
        rgb = np.full((64, 64, 3), 80, dtype=np.uint8)
        proposal_path, preview_path = root / "proposal.json", root / "preview.png"
        from . import quality_runner
        with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                patch.object(initialization, "_make_preview",
                             return_value=(self.preview_bytes, self.preview_binding)):
            result = initialization.create_proposal(bundle, detector, proposal_path, preview_path)
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        return bundle, detector, manifest, mask, rgb, proposal_path, preview_path, proposal, result

    def test_proposal_binds_six_detector_identity_fields_pixels_and_preview_without_review(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            (_bundle, detector, manifest, _mask, rgb, proposal_path,
             preview_path, proposal, result) = self._create(root)
            self.assertEqual(result["status"], "proposal_created")
            self.assertFalse(result["reviewed"])
            self.assertNotIn("reviewed", proposal)
            self.assertEqual(proposal["timing_binding"]["frame_id"], 0)
            self.assertEqual(proposal["timing_binding"]["source_frame_id"], 0)
            self.assertEqual(proposal["timing_binding"]["timestamp_s"], 0.)
            self.assertEqual(proposal["rgb_binding"]["sha256"],
                             hashlib.sha256(rgb.tobytes(order="C")).hexdigest())
            self.assertEqual(proposal["detector_binding"]["npz_sha256"],
                             hashlib.sha256(detector.read_bytes()).hexdigest())
            self.assertEqual(proposal["preview_binding"]["sha256"],
                             hashlib.sha256(preview_path.read_bytes()).hexdigest())
            self.assertEqual(proposal["source_binding"]["input_manifest_sha256"],
                             parse_timing_manifest((root / "raw/input.json").read_bytes())["manifest_sha256"])
            self.assertEqual(proposal["selection"], {"points": [[11, 11]], "labels": [1]})
            self.assertEqual(set(json.loads(proposal_path.read_text())),
                             initialization._PROPOSAL_FIELDS)

    def test_empty_detection_retains_diagnostic_npz_and_creates_no_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            bundle, detector, _manifest, _mask = _raw_bundle(root)
            with np.load(detector, allow_pickle=False) as archive:
                fields = {name: archive[name].copy() for name in archive.files}
            fields["mask"][:] = 0
            np.savez_compressed(detector, **fields)
            from . import quality_runner
            with patch.object(quality_runner, "read_rgb",
                              return_value=np.full((64, 64, 3), 80, dtype=np.uint8)):
                result = initialization.create_proposal(bundle, detector,
                    root / "proposal.json", root / "preview.png")
            self.assertEqual(result["status"], "no_detection")
            self.assertEqual(result["reason"], "empty_observed_foreground")
            self.assertTrue(detector.is_file())
            self.assertFalse((root / "proposal.json").exists())

    def test_wrong_detector_identity_dtype_and_source_hash_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            bundle, detector, _manifest, _mask = _raw_bundle(root)
            with np.load(detector, allow_pickle=False) as archive:
                fields = {name: archive[name].copy() for name in archive.files}
            fields["sourceFrameId"] = np.array(9, dtype=np.int64)
            np.savez_compressed(detector, **fields)
            from . import quality_runner
            with patch.object(quality_runner, "read_rgb",
                              return_value=np.full((64, 64, 3), 80, dtype=np.uint8)):
                with self.assertRaisesRegex(ValueError, "source_frame_id"):
                    initialization.create_proposal(bundle, detector,
                        root / "proposal.json", root / "preview.png")
            self.assertFalse((root / "proposal.json").exists())

            second_root = root / "second"
            second_root.mkdir()
            bundle, detector, _manifest, _mask = _raw_bundle(second_root)
            (bundle / "synthetic.mp4").write_bytes(b"mutated video bytes")
            with patch.object(quality_runner, "read_rgb",
                              return_value=np.full((64, 64, 3), 80, dtype=np.uint8)):
                with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                    initialization.create_proposal(bundle, detector,
                        root / "bad.json", root / "bad.png")

            third_root = root / "third"
            third_root.mkdir()
            bundle, detector, _manifest, _mask = _raw_bundle(third_root)
            with np.load(detector, allow_pickle=False) as archive:
                fields = {name: archive[name].copy() for name in archive.files}
            fields["mask"] = fields["mask"].astype(np.float32)
            np.savez_compressed(detector, **fields)
            with patch.object(quality_runner, "read_rgb",
                              return_value=np.full((64, 64, 3), 80, dtype=np.uint8)):
                with self.assertRaisesRegex(ValueError, "uint8 binary"):
                    initialization.create_proposal(bundle, detector,
                        root / "dtype.json", root / "dtype.png")
            self.assertTrue(detector.is_file())

    def test_explicit_receipt_is_required_and_fresh_bundle_passes_runner_and_sam_preflight(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            bundle, detector, _manifest, _mask, rgb, proposal_path, preview_path, proposal, _result = self._create(root)
            from . import quality_runner
            with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                    patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                    patch.object(initialization, "_make_preview",
                                 return_value=(self.preview_bytes, self.preview_binding)):
                with self.assertRaises(FileNotFoundError):
                    initialization.finalize_selection(bundle, detector, proposal_path,
                        preview_path, root / "missing-receipt.json", root / "reviewed")

                receipt_path = root / "review.json"
                receipt = _review_receipt(proposal)
                receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
                expected_review_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                result = initialization.finalize_selection(bundle, detector, proposal_path,
                    preview_path, receipt_path, root / "reviewed",
                    finalization_receipt_path=root / "finalization.json")
            self.assertEqual(result["status"], "finalized")
            self.assertEqual(result["bundle_inventory"], ["input.json", "object.glb", "synthetic.mp4"])
            self.assertEqual(sorted(path.name for path in (root / "reviewed").iterdir()),
                             ["input.json", "object.glb", "synthetic.mp4"])
            final = json.loads((root / "reviewed/input.json").read_text(encoding="utf-8"))
            original = json.loads((bundle / "input.json").read_text(encoding="utf-8"))
            expected_selection = {"reviewed": True, "frame_id": 0,
                "source_frame_id": 0, "timestamp_s": 0.0,
                "points": [[11, 11]], "labels": [1]}
            self.assertEqual(final["selection"], expected_selection)
            final.pop("selection")
            self.assertEqual(final, original)
            receipt_out = json.loads((root / "finalization.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt_out["final_input_manifest_sha256"],
                             result["final_input_manifest_sha256"])
            self.assertEqual(receipt_out["review_receipt_sha256"], expected_review_sha)
            self.assertEqual(receipt_out["bundle_inventory"], ["input.json", "object.glb", "synthetic.mp4"])

    def test_finalization_receipt_cannot_overlap_raw_or_review_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            bundle, detector, _manifest, _mask, rgb, proposal_path, preview_path, proposal, _ = self._create(root)
            receipt_path = root / "review.json"
            receipt_path.write_text(json.dumps(_review_receipt(proposal)), encoding="utf-8")
            initial_files = {path.name: path.read_bytes() for path in bundle.iterdir()}
            destinations = (
                bundle / "internal.json", bundle / "input.json",
                bundle / "synthetic.mp4", bundle / "object.glb",
                proposal_path, receipt_path, preview_path, detector,
            )
            from . import quality_runner
            with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                    patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                    patch.object(initialization, "_make_preview",
                                 return_value=(self.preview_bytes, self.preview_binding)):
                for index, destination in enumerate(destinations):
                    output = root / f"collision-{index}"
                    with self.assertRaises(ValueError):
                        initialization.finalize_selection(bundle, detector, proposal_path,
                            preview_path, receipt_path, output,
                            finalization_receipt_path=destination)
                    self.assertFalse(output.exists())
            self.assertEqual({path.name: path.read_bytes() for path in bundle.iterdir()},
                             initial_files)
            self.assertFalse((bundle / "internal.json").exists())

    def test_approval_and_observed_inputs_are_rechecked_after_preflight_before_promotion(self):
        from . import quality_runner
        # A preceding test temporarily replaces sys.modules. Import the live
        # module used by finalize_selection rather than a stale package attribute.
        quality_sam2 = importlib.import_module("bench.quality_sam2")

        for changed_name in ("proposal", "receipt", "preview", "detector"):
            with self.subTest(changed=changed_name), \
                    tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
                root = Path(temporary)
                bundle, detector, _manifest, _mask, rgb, proposal_path, preview_path, proposal, _ = self._create(root)
                receipt_path = root / "review.json"
                receipt_path.write_text(json.dumps(_review_receipt(proposal)), encoding="utf-8")
                original_prepare = quality_sam2.prepare_cache_plan

                def mutate_during_preflight(raw_input):
                    target = {
                        "proposal": proposal_path,
                        "receipt": receipt_path,
                        "preview": preview_path,
                        "detector": detector,
                    }[changed_name]
                    if changed_name == "detector":
                        target.write_bytes(target.read_bytes()[:-1])
                    else:
                        target.write_bytes(target.read_bytes() + b" ")
                    return original_prepare(raw_input)

                with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                        patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                        patch.object(initialization, "_make_preview",
                                     return_value=(self.preview_bytes, self.preview_binding)), \
                        patch.object(quality_sam2, "prepare_cache_plan",
                                     side_effect=mutate_during_preflight) as preflight:
                    output = root / "mutated-output"
                    with self.assertRaises(ValueError):
                        initialization.finalize_selection(bundle, detector, proposal_path,
                            preview_path, receipt_path, output,
                            finalization_receipt_path=root / "finalization.json")
                    preflight.assert_called_once()
                self.assertFalse(output.exists())
                incomplete = list(root.glob(".mutated-output.stage-*/incomplete.json"))
                self.assertEqual(len(incomplete), 1)
                self.assertEqual(json.loads(incomplete[0].read_text(encoding="utf-8"))["status"],
                                 "incomplete")

    def test_rejected_stale_preview_and_point_receipts_do_not_promote(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            bundle, detector, _manifest, _mask, rgb, proposal_path, preview_path, proposal, _ = self._create(root)
            from . import quality_runner
            for name, updates in (
                ("reject", {"decision": "reject"}),
                ("stale", {"proposal_sha256": "0" * 64}),
                ("point", {"selection": {"points": [[12, 12]], "labels": [1]}}),
            ):
                receipt = _review_receipt(proposal)
                receipt.update(updates)
                receipt_path = root / f"{name}.json"
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                output = root / f"out-{name}"
                with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                        patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                        patch.object(initialization, "_make_preview",
                                     return_value=(self.preview_bytes, self.preview_binding)):
                    with self.assertRaises(ValueError):
                        initialization.finalize_selection(bundle, detector, proposal_path,
                            preview_path, receipt_path, output)
                self.assertFalse(output.exists())
            preview_path.write_bytes(b"tampered preview")
            receipt_path = root / "preview-review.json"
            receipt_path.write_text(json.dumps(_review_receipt(proposal)), encoding="utf-8")
            with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                    patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                    patch.object(initialization, "_make_preview",
                                 return_value=(self.preview_bytes, self.preview_binding)):
                with self.assertRaisesRegex(ValueError, "preview bytes"):
                    initialization.finalize_selection(bundle, detector, proposal_path,
                        preview_path, receipt_path, root / "out-preview")
            self.assertFalse((root / "out-preview").exists())

    def test_preexisting_extra_raw_resource_and_preflight_failure_leave_no_successful_bundle(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            bundle, detector, _manifest, _mask, rgb, proposal_path, preview_path, proposal, _ = self._create(root)
            (bundle / "poses.jsonl").write_text("not an inference resource")
            from . import quality_runner
            quality_sam2 = importlib.import_module("bench.quality_sam2")
            with patch.object(quality_runner, "read_rgb", return_value=rgb):
                with self.assertRaisesRegex(ValueError, "only input.json"):
                    initialization.create_proposal(bundle, detector,
                        root / "extra.json", root / "extra.png")
            (bundle / "poses.jsonl").unlink()
            receipt_path = root / "approved.json"
            receipt_path.write_text(json.dumps(_review_receipt(proposal)), encoding="utf-8")
            with patch.object(quality_runner, "read_rgb", return_value=rgb), \
                    patch.object(initialization, "_read_cnos_pins", return_value=self.pins), \
                    patch.object(initialization, "_make_preview",
                                 return_value=(self.preview_bytes, self.preview_binding)), \
                    patch.object(quality_sam2, "prepare_cache_plan",
                                 side_effect=RuntimeError("fixture preflight failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture preflight"):
                    initialization.finalize_selection(bundle, detector, proposal_path,
                        preview_path, receipt_path, root / "failed-output")
            self.assertFalse((root / "failed-output").exists())
            incomplete = list(root.glob(".failed-output.stage-*/incomplete.json"))
            self.assertEqual(len(incomplete), 1)
            receipt = json.loads(incomplete[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "incomplete")
            self.assertFalse(receipt["promoted"])


if __name__ == "__main__":
    unittest.main()
