"""Manufactured geometry and mocked control flow; no Torch import or model work."""
from contextlib import nullcontext
import ast
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bench import quality_adaptation_data as data
from bench import quality_adaptation_smoke as smoke
from bench import quality_adaptation_train as training
from bench.quality_adaptation_prepare import crops_from_retained_capture, extract_one
from bench.test_quality_adaptation_contract import Tensor, dataset_fixture, geometry


def pure_source_function(filename, name):
    """Exercise a NumPy-only production helper without importing OpenCV/Torch."""
    path = Path(__file__).with_name(filename)
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def plan_fixture():
    pin = {"path": "/unused/no_payload", "bytes": 1, "sha256": "a" * 64}
    rows = []
    for index, (split, obj, pitch) in enumerate((("train", "17", -25), ("train", "17", 25), ("val", "27", -25))):
        rows.append({"id": f"crop-{index}", "object_id": obj, "split": split, "sequence_id": f"seq-{index}",
            "capture_group": f"capture-{index}", "frame_id": 0, "yaw0_deg": 0, "pitch_deg": pitch,
            **{name: dict(pin) for name in ("input", "mask_index", "mask", "bank", "generation")}})
    return {"schema": smoke.PLAN_SCHEMA, "max_seconds": 1800, "depth_tolerance_m": .002,
        "native_raster_convention": smoke.RASTER_CONVENTION,
        "rows": rows, "sealed_test": {"id": "sealed7", "object_id": "7", "sequence_id": "test7",
        "capture_group": "test7", "split": "test", "status": "unavailable", "query_rgb_sha256": None,
        "failure_reason": "not_materialized", "unavailable_provenance": "frozen TEST7 inventory; no payload opened"}}


class ContractTests(unittest.TestCase):
    def test_historical_generation_schema_and_split_alias_preserve_all_bindings(self):
        from bench.quality_ranker_synthetic import BROADER_RECIPE_KIND
        context = {"recipe": {"kind": BROADER_RECIPE_KIND}, "recipe_sha256": "a" * 64,
            "sequence": {"yaw0_deg": 0, "pitch_deg": -25}, "group": {"split": "validation"},
            "mesh_sha256": "b" * 64}
        row = {"split": "val", "yaw0_deg": 0, "pitch_deg": -25, "sequence_id": "fixed-sequence"}
        manifest = {"physical_object_id": "hot3d:obj_000027", "video": "synthetic.mp4",
            "source_hashes": {"synthetic.mp4": "c" * 64}}
        generation = {"schema_version": 1, "kind": "quality-ranker-synthetic-generation-v1",
            "recipe_sha256": "a" * 64, "split": "validation", "sequence_id": "fixed-sequence",
            "physical_object_id": "hot3d:obj_000027", "mesh_sha256": "b" * 64,
            "video_sha256": "c" * 64, "split_file_sha256": "d" * 64}
        smoke.validate_generation_context(context, generation, manifest, row, "d" * 64)
        for key in generation:
            wrong = dict(generation, **{key: "incorrect"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_generation_context(context, wrong, manifest, row, "d" * 64)
        with self.assertRaises(ValueError):
            smoke.validate_generation_context(context, dict(generation, recipe_kind="wrong"), manifest, row, "d" * 64)

    def test_exact_old_extraction_compatibility_keeps_provenance_and_rejects_changed_feature_source(self):
        lock = data.read_json(smoke.LOCK)
        prior, compatibility = next(iter(lock["frozen_feature_compatibility"].items()))
        recipe = {"source_lock_sha256": prior, "source_files_sha256": compatibility["source_files_sha256"],
                  "upstream_revision": compatibility["upstream_revision"], "producer_sha256": compatibility["producer_sha256"]}
        before = data.canonical_hash(recipe)
        self.assertTrue(smoke.validate_frozen_feature_source(recipe)["compatibility_used"])
        self.assertEqual(data.canonical_hash(recipe), before)
        wrong_lock = copy.deepcopy(lock)
        source_key = next(k for k in lock["gotrack"]["files"] if k not in compatibility["allowed_metadata_adapters"])
        wrong_lock["gotrack"]["files"][source_key] = "0" * 64
        with patch.object(smoke, "read_json", return_value=wrong_lock), self.assertRaises(ValueError):
            smoke.validate_frozen_feature_source(recipe)
        with self.assertRaises(ValueError):
            smoke.validate_frozen_feature_source(dict(recipe, source_lock_sha256="0" * 64))
        with self.assertRaises(ValueError):
            smoke.validate_frozen_feature_source(dict(recipe, producer_sha256="0" * 64))

    def test_reuse_preserves_original_failed_inventory_and_exact_recipe_bytes(self):
        lock = data.read_json(smoke.LOCK)
        prior, compatibility = next(iter(lock["frozen_feature_compatibility"].items()))
        recipe = {"source_lock_sha256": prior, "source_files_sha256": compatibility["source_files_sha256"],
                  "upstream_revision": compatibility["upstream_revision"], "producer_sha256": compatibility["producer_sha256"]}
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, features, output = base / "capture", base / "features", base / "features-v2"
            root.mkdir(); features.mkdir()
            plan = plan_fixture()
            capture_rows, index_rows = [], []
            for i, row in enumerate(plan["rows"]):
                crop = root / f"crop-{i}.npz"
                crop.write_bytes(b"manufactured opaque crop bytes for copy/identity test")
                saved = features / f"features-{i}.npz"
                saved.write_bytes(b"manufactured opaque feature bytes for copy/identity test" + bytes([i]))
                capture_rows.append({"id": row["id"], "status": "available", "crops": smoke.artifact(root, crop),
                                     "capture": smoke.artifact(root, crop)})
                index_rows.append({"id": row["id"], "status": "available", "features": smoke.artifact(features, saved)})
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "capture-index.json").write_text(json.dumps({"rows": capture_rows}))
            (root / "capture-receipt.json").write_text(json.dumps({"original_source": prior}))
            (features / "recipe.json").write_text(json.dumps(recipe, indent=3))
            (features / "index.json").write_text(json.dumps({"recipe_sha256": data.canonical_hash(recipe), "rows": index_rows}))
            (features / "labels.json").write_text('"original failed labels retained"')
            seal = {key: smoke.artifact(root, root / name) for key, name in (
                ("capture_receipt", "capture-receipt.json"), ("plan", "plan.json"), ("crop_index", "capture-index.json"))}
            seal.update(feature_index_sha256=data.sha256(features / "index.json"),
                        feature_recipe_sha256=data.sha256(features / "recipe.json"))
            (root / "prelabel-seal.json").write_text(json.dumps(seal))
            failed_hash = data.sha256(features / "labels.json")
            # This fixture copies opaque manufactured archives. Authenticate
            # the real historical checkout in cloud preflight, while retaining
            # the actual copy/hash/compatibility checks in this model-free test.
            source = dict(source_lock_sha256=data.sha256(smoke.LOCK),
                          upstream_revision=lock['gotrack']['revision'],
                          source_files_sha256=data.canonical_hash(lock['gotrack']['files']))
            with patch.object(smoke, "cloud_guard"), \
                 patch.object(smoke, "validate_plan", return_value=plan), \
                 patch.object(smoke, "verify_sources", return_value=source) as verified:
                receipt = smoke.reuse_features(root, features, output, cloud_execution=True)
                verified.assert_called_once_with()
            self.assertEqual(data.sha256(features / "labels.json"), failed_hash)
            self.assertFalse((output / "labels.json").exists())
            self.assertEqual(data.sha256(output / "recipe.json"), data.sha256(features / "recipe.json"))
            self.assertEqual(data.read_json(output / "recipe.json")["source_lock_sha256"], prior)
            self.assertFalse(receipt["inference_repeated"])
            self.assertFalse(receipt["deadline_reset"])

    def test_frozen_plan_rejects_substitution_and_test_payload_paths(self):
        plan = plan_fixture()
        smoke.validate_plan(plan, verify_files=False)
        for mutate in (lambda p: p["rows"].reverse(),
                       lambda p: p["rows"][0].update(frame_id=1),
                       lambda p: p["sealed_test"].update(features={"path": "sealed.npz"})):
            changed = copy.deepcopy(plan)
            mutate(changed)
            with self.assertRaises(ValueError):
                smoke.validate_plan(changed, verify_files=False)

    def test_unavailable_unknown_test_hash_does_not_open_or_hash_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, doc = dataset_fixture(root)
            doc["rows"][2] = plan_fixture()["sealed_test"]
            path.write_text(json.dumps(doc))
            with patch.object(data, "load_npz", wraps=data.load_npz) as loader:
                ds = data.load_dataset(path)
            self.assertEqual(len(loader.call_args_list), 4)  # TRAIN+VAL features+labels only.
            self.assertEqual(ds.rows("test", available=False)[0]["query_rgb_sha256"], None)
            for split in ("train", "val"):
                changed = copy.deepcopy(doc)
                changed["rows"][0].update(query_rgb_sha256=None, unavailable_provenance="unknown")
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    data.load_dataset(path, inspect_arrays=False)

    def test_unavailable_unknown_hash_requires_reason_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            path, doc = dataset_fixture(Path(temp))
            doc["rows"][2] = plan_fixture()["sealed_test"]
            for field in ("failure_reason", "unavailable_provenance"):
                changed = copy.deepcopy(doc)
                changed["rows"][2][field] = ""
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    data.load_dataset(path, inspect_arrays=False)

    def test_capture_ignores_pnp_and_keeps_only_fixed_iteration0(self):
        sink = smoke.RetainedCapture()
        self.assertFalse(sink.should_capture({}, "retained", {"iteration": 1}))
        sink({"kind": "retained", "context": {}, "metadata": {"iteration": 0}, "arrays": {"marker": 1}})
        self.assertEqual(sink.arrays, {"marker": 1})
        self.assertFalse(sink.should_capture({}, "retained", {"iteration": 0}))
        self.assertFalse(sink.preflight_nbytes({}, "retained", {}, 33 * 1024**2))

    def test_no_cloud_guard_prevents_model_import(self):
        with patch.object(smoke, "cloud_guard", side_effect=RuntimeError("cloud required")), patch.object(smoke, "load_frozen_network") as load:
            with self.assertRaisesRegex(RuntimeError, "cloud required"):
                smoke.capture(Path("absent"), Path("absent"))
            load.assert_not_called()


class NumericalTests(unittest.TestCase):
    def test_legacy_raster_actual_crop_map_and_integer_labels_roundtrip(self):
        actual_sampling_map = pure_source_function("quality_gotrack.py", "_capture_actual_sampling_map")
        original = np.array([[576., 0., 319.5], [0., 576., 239.5], [0., 0., 1.]])
        effective = smoke.legacy_native_intrinsics(original)
        crop_k = np.array([[420., 0., 139.5], [0., 450., 139.5], [0., 0., 1.]])
        angle = .11
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        class Camera:
            def __init__(self, k, eye_to_world): self.k, self.r = k, eye_to_world
            def window_to_eye(self, pixels): return np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(self.k).T
            def eye_to_world(self, xyz): return xyz @ self.r.T
            def world_to_eye(self, xyz): return xyz @ self.r
            def eye_to_window(self, xyz):
                h = xyz @ self.k.T
                return h[:, :2] / h[:, 2:]
        yy, xx = np.mgrid[:280, :280]
        grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        source = Camera(effective, np.eye(3))
        crop = Camera(crop_k, rotation.T)
        native_pixels = actual_sampling_map(source, crop, grid)[3].astype(np.float64)
        # Legacy raster: native pixel center uses (u+.5,v+.5) under original K.
        # Trace these physical rays through a rotated/scaled crop and compare to
        # the exact integer-center labels Kcrop @ C @ Xnative.
        native_rays = np.column_stack([native_pixels + .5, np.ones(len(grid))]) @ np.linalg.inv(original).T
        query_h = native_rays @ rotation.T @ crop_k.T
        recovered = query_h[:, :2] / query_h[:, 2:]
        np.testing.assert_allclose(recovered, grid, atol=4e-5, rtol=0)
        wrong_source = Camera(original, np.eye(3))
        wrong_native = actual_sampling_map(wrong_source, crop, grid)[3].astype(np.float64)
        wrong_rays = np.column_stack([wrong_native + .5, np.ones(len(grid))]) @ np.linalg.inv(original).T
        wrong_h = wrong_rays @ rotation.T @ crop_k.T
        wrong_crop = wrong_h[:, :2] / wrong_h[:, 2:]
        # The old uncorrected native K causes a measurable 2D label bias.
        self.assertGreater(float(np.linalg.norm(wrong_crop - grid, axis=1).max()), .5)

    def test_exact_float_rgb_preserved_and_byte_compatibility(self):
        rgb = np.full((1, 3, 280, 280), .1234567, np.float32)
        pose = np.eye(4)
        capture = {"query_rgb_crop": rgb, "template_rgb": rgb.copy(),
            "rendered_depth_mm": np.full((1, 280, 280), 1000., np.float32),
            "rendered_mask": np.ones((1, 280, 280), bool), "crop_k": np.diag([100., 100., 1.]),
            "current_crop_camera_from_object_mm": pose}
        exact = crops_from_retained_capture(capture, preserve_float_rgb=True)
        rounded = crops_from_retained_capture(capture)
        # The input value is preserved bitwise; float->byte->float changes it.
        np.testing.assert_array_equal(exact["query_rgb"], rgb[0].transpose(1, 2, 0))
        self.assertFalse(np.array_equal(exact["query_rgb"], rounded["query_rgb"].astype(np.float32) / 255))
        byte = np.full((280, 280, 3), 17, np.uint8)
        self.assertEqual(data.crop_rgb_hash(byte), data.crop_rgb_hash(byte.astype(np.float32) / 255))
        # Integer unprojection at pixel(10,20), Z=1m,f=100 gives (.1,.2,1)m.
        np.testing.assert_allclose(exact["template_xyz_m"][20, 10], [.1, .2, 1.], atol=1e-7)

    def test_float_extractor_does_not_divide_by_255(self):
        captured = []
        def backbone(rgb):
            captured.append(rgb.value.copy())
            return {"feature_maps": Tensor(np.zeros((2, 384, 20, 20), np.float32))}
        reference = [Tensor(np.zeros((1, 400, 768), np.float32)) for _ in range(12)]
        head = lambda tokens, size: (Tensor(np.zeros((1, 2, 280, 280), np.float32)),
                                    Tensor(np.full((1, 280, 280), .5, np.float32)))
        network = SimpleNamespace(backbone=backbone, decoder=lambda **kwargs: (None, reference), pose_head=head)
        crops = dict(geometry(), query_rgb=np.full((280, 280, 3), .1234567, np.float32),
                     template_rgb=np.full((280, 280, 3), .8, np.float32))
        fake = SimpleNamespace(inference_mode=nullcontext, from_numpy=Tensor, float32="float32")
        extract_one(network, crops, device="cpu", torch_module=fake)
        # Exact retained [0,1] values are the backbone inputs; no second scaling.
        np.testing.assert_array_equal(captured[0][0], crops["query_rgb"].transpose(2, 0, 1))

    def test_integer_sampling_map_known_identity_and_half_pixel_rejection(self):
        yy, xx = np.mgrid[:280, :280]
        identity = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float64)
        # Identical camera/K/C analytically maps every integer center to itself.
        self.assertEqual(smoke.sampling_map_error(identity, identity)["max_native_pixel_difference"], 0.)
        with self.assertRaisesRegex(ValueError, "sampling map"):
            smoke.sampling_map_error(identity + .5, identity)

    def test_visibility_metrics_known_support_and_historical_strict_threshold(self):
        confidence = np.zeros((280, 280), np.float32)
        known = np.zeros_like(confidence, bool)
        known[0, :2] = True
        visible = np.zeros_like(known)
        visible[0, 0] = True
        confidence[0, :2] = [.5, .25]
        result = data.visibility_metrics(confidence, {"known": known, "visible": visible})
        # Brier = ((.5-1)^2 + (.25-0)^2)/2 = .15625.
        self.assertEqual(result["brier"], .15625)
        self.assertEqual(result["known_pixels"], 2)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 0)

    def test_visibility_threshold_keeps_serving_float32_tie_semantics(self):
        confidence = np.zeros((280, 280), np.float32)
        known = np.zeros((280, 280), bool)
        known[0, 0] = True
        confidence[0, 0] = np.float32(.3)
        result = data.visibility_metrics(confidence, {"known": known, "visible": known.copy()})
        # Serving compares native float32 >.3, so an exactly stored .3 is false.
        # Promoting .3f to float64 first would incorrectly make it >.3 double.
        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["false_negative"], 1)

    def test_independent_render_camera_units_and_transform_composition(self):
        k = np.diag([100., 100., 1.])
        c = np.eye(4)
        c[0, 3] = 100.  # retained camera-to-camera translation in mm
        pose = np.eye(4)
        pose[2, 3] = 2.  # independent query pose in metres
        cameras = []
        class Camera:
            def __init__(self, **kwargs):
                cameras.append(kwargs)
        render_type = SimpleNamespace(DEPTH="depth", MASK="mask")
        renderer = SimpleNamespace(render_object_model=lambda obj, camera: {
            "depth": np.full((280, 280), 2000., np.float32), "mask": np.ones((280, 280), bool)})
        modules = {"utils": SimpleNamespace(structs=SimpleNamespace(PinholePlaneCameraModel=Camera),
                                            renderer_base=SimpleNamespace(RenderType=render_type)),
                   "bench.quality_contract": SimpleNamespace(checked_pose=pure_source_function("quality_contract.py", "checked_pose"))}
        with patch.dict(sys.modules, modules):
            query_pose, depth, support = smoke.render_query_labels(renderer, object_id="x", crop_k=k,
                crop_from_orig=c, native_pose=pose)
        # C@T has (.1,0,2)m translation; the inverse renderer camera uses (-100,0,-2000)mm.
        np.testing.assert_allclose(query_pose[:3, 3], [.1, 0., 2.])
        np.testing.assert_allclose(cameras[0]["T_world_from_eye"][:3, 3], [-100., 0., -2000.])
        self.assertTrue(np.all(depth == 2.))
        self.assertTrue(support.all())


class Scalar:
    """Minimal scalar/tensor state for loop wiring; never computes predictions."""
    def __init__(self, value): self.value = np.asarray(value, np.float32)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.value
    def backward(self): pass
    def __float__(self): return float(self.value)


class ResumeControlTests(unittest.TestCase):
    def test_actual_train_loop_interrupts_after_one_step_then_restores_without_repeat(self):
        checkpoints, optimized_rows = {}, []
        class Head:
            def __init__(self): self.weight = Scalar(0.)
            def parameters(self): return []
            def state_dict(self): return {"p": self.weight}
            def load_state_dict(self, value, strict=True): self.weight = copy.deepcopy(value["p"])
            def train(self): pass
            def __call__(self, tokens, size): return None, None
        heads = []
        def load_head(**kwargs):
            head = Head()
            heads.append(head)
            return head
        class Adam:
            def __init__(self, *args, **kwargs): self.state = {}
            def zero_grad(self, **kwargs): pass
            def step(self):
                previous = float(self.state.get(0, {"step": Scalar(0)})["step"])
                self.state = {0: {"step": Scalar(previous + 1)}}
                heads[-1].weight = Scalar(float(heads[-1].weight) + 1)
            def state_dict(self): return {"state": self.state, "param_groups": []}
            def load_state_dict(self, value): self.state = copy.deepcopy(value["state"])
        class Scaler:
            def __init__(self, *args, **kwargs): pass
            def state_dict(self): return {}
            def load_state_dict(self, value): pass
            def scale(self, loss): return loss
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        def save(root, name, state, cfg, **kwargs):
            checkpoints[name] = copy.deepcopy(state)
            (root / name).write_bytes(b"mock checkpoint control flow only")
        torch = SimpleNamespace(manual_seed=lambda seed: None, use_deterministic_algorithms=lambda value: None,
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False)), optim=SimpleNamespace(AdamW=Adam),
            amp=SimpleNamespace(GradScaler=Scaler), autocast=lambda **kwargs: nullcontext(), float16="float16",
            isfinite=lambda value: np.isfinite(float(value)), get_rng_state=lambda: Scalar(42), set_rng_state=lambda state: None,
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *args, **kwargs: Scalar(.5))),
            load=lambda *args, **kwargs: copy.deepcopy(checkpoints["last.pt"]))
        rows = [{"id": "a"}, {"id": "b"}]
        ds = SimpleNamespace(identity="dataset", rows=lambda split: rows)
        cfg = replace(training.TrainConfig(), epochs=1, max_steps=2, checkpoint_every_steps=1)
        def batch(ds, rows, *args):
            optimized_rows.extend(r["id"] for r in rows)
            return "tokens", "labels"
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"torch": torch}), \
             patch.object(training, "cloud_guard"), patch.object(training, "preflight", return_value={}), \
             patch.object(training, "runtime_versions", return_value={}), patch.object(training, "load_dataset", return_value=ds), \
             patch.object(training, "load_head", side_effect=load_head), patch.object(training, "save_checkpoint", side_effect=save), \
             patch.object(training, "tensor_batch", side_effect=batch), patch.object(training, "adaptation_loss", return_value=Scalar(2.)), \
             patch.object(training, "evaluate_validation", return_value={"adapted": {"object_macro_epe_px": 1.}}):
            root = Path(temp) / "run"
            first = training.train(Path("manifest"), root, cfg, cloud_execution=True, stop_after_steps=1)
            self.assertEqual(first, {"status": "incomplete", "reason": "invocation_step_limit"})
            self.assertEqual(checkpoints["last.pt"]["batch_cursor"], 1)
            self.assertFalse(checkpoints["last.pt"]["completed"])
            second = training.train(Path("manifest"), root, cfg, cloud_execution=True, resume=True)
            self.assertEqual(second["status"], "completed")
            self.assertEqual(checkpoints["last.pt"]["steps"], 2)
            self.assertEqual(float(checkpoints["last.pt"]["optimizer"]["state"][0]["step"]), 2.)
            self.assertEqual(optimized_rows, [r["id"] for b in training.epoch_batches(rows, 0, cfg) for r in b])
            status = data.read_json(root / "status.json")
            self.assertEqual(status["resumed_from"]["steps"], 1)
            self.assertEqual(status["resumed_from"]["optimizer_state_entries"], 1)


if __name__ == "__main__":
    unittest.main()
