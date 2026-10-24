"""Synthetic bundle metadata tests; no renderer or video generation is run."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_ranker_synthetic as synthetic_module
from .quality_ranker_data import freeze_split_manifest
from .quality_ranker_synthetic import (
    _cloud_guard, _render_bundle, broader_trajectory_pose, composite_broader_occluder,
    make_input_manifest, seal_recipe, synthetic_pose_label, trajectory_pose,
    orientation_trajectory_pose, validate_broader_trajectory,
    validate_orientation_recipe, validate_orientation_trajectory, validate_recipe,
    validate_single_view_recipe, validate_single_view_trajectory,
)


class SyntheticBundleTests(unittest.TestCase):
    def _frozen_group(self, mesh_sha):
        frozen = freeze_split_manifest({
            "records": [{
                "physical_object_id": "synthetic_metric_asset",
                "object_id": "synthetic_metric_asset",
                "scene_id": "synthetic-scene-1",
                "sequence_id": "synthetic-sequence-1",
                "background_id": "procedural:gray",
                "background_kind": "procedural",
                "provenance_kind": "synthetic",
                "mesh_sha256": mesh_sha,
                "frame_id": ordinal,
                "source_frame_id": ordinal,
                "split": "train",
            } for ordinal in range(2)],
            "label_policy": {"max_rotation_error_deg": 5., "max_translation_error_m": .05},
        })
        return frozen["groups"][0]

    def test_manifest_passes_actual_runner_reader_with_real_relative_hash_paths(self):
        from .quality_runner import read_input

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            bundle = Path(temp)
            (bundle / "object.glb").write_bytes(b"analytical fake metric mesh bytes")
            (bundle / "synthetic.mp4").write_bytes(b"analytical fake video bytes")
            mesh_sha = hashlib.sha256((bundle / "object.glb").read_bytes()).hexdigest()
            video_sha = hashlib.sha256((bundle / "synthetic.mp4").read_bytes()).hexdigest()
            manifest = make_input_manifest(mesh_sha256=mesh_sha, video_sha256=video_sha,
                                           frames=2, width=64, height=64, fps=10.,
                                           sequence_group=self._frozen_group(mesh_sha))
            self.assertEqual(set(manifest["source_hashes"]), {"synthetic.mp4", "object.glb"})
            self.assertNotIn("poses", manifest)
            self.assertNotIn("mask", manifest)
            input_path = bundle / "input.json"
            input_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            parsed = read_input(bundle)
            self.assertEqual(parsed["_timing"]["clock_mode"], "capture_seconds_v1")
            self.assertEqual(parsed["frame_ids"], [0, 1])
            self.assertEqual(parsed["video"], "synthetic.mp4")
            self.assertEqual(parsed["asset"], "object.glb")
            self.assertEqual(parsed["source_hashes"], manifest["source_hashes"])

            # This is the same physical-resource requirement enforced by the
            # runner before the SAM/tracker pipeline consumes the bundle.
            bad = dict(manifest)
            bad["video"] = "unhashed-video.mp4"
            input_path.write_text(json.dumps(bad, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must hash its declared video"):
                read_input(bundle)

    def test_trajectory_is_analytic_metric_and_labels_are_separate(self):
        first = trajectory_pose(0, 1)
        self.assertEqual(first.shape, (4, 4))
        self.assertAlmostEqual(first[2, 3], .62)
        self.assertGreater(first[2, 3], 0.)
        mesh_sha = "c" * 64
        label = synthetic_pose_label("physical-mesh", "sequence", 0, 1,
                                     observationally_ambiguous=True)
        self.assertTrue(label["observationally_ambiguous"])
        manifest = make_input_manifest(mesh_sha256=mesh_sha, video_sha256="d" * 64,
                                       frames=2, width=64, height=64, fps=10.,
                                       sequence_group=self._frozen_group(mesh_sha))
        self.assertNotIn("pose", manifest)
        self.assertNotIn("observationally_ambiguous", manifest)
        self.assertEqual(label["pose"], first.tolist())

    def test_pre_frozen_mesh_and_explicit_cloud_guard_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exact metric mesh SHA-256"):
            make_input_manifest(mesh_sha256="a" * 64, video_sha256="b" * 64,
                                frames=2, width=64, height=64, fps=10.,
                                sequence_group=self._frozen_group("c" * 64))
        with self.assertRaisesRegex(RuntimeError, "approved Linux cloud"):
            _cloud_guard(False)
        with patch("platform.system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "approved Linux cloud"):
                _cloud_guard(True)


def _off_origin_box():
    center = np.array([1.2, -0.4, 2.0], dtype=np.float64)
    half = np.array([0.08, 0.05, 0.03], dtype=np.float64)
    return np.array([[center[0] + sx * half[0], center[1] + sy * half[1],
                      center[2] + sz * half[2]]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


class BroaderSyntheticGeometryTests(unittest.TestCase):
    def test_off_origin_center_metric_depth_all_rotations_inverse_and_projection(self):
        vertices = _off_origin_box()
        validation = validate_broader_trajectory(vertices)
        center = np.array([1.2, -0.4, 2.0])
        radius = math.sqrt(0.08 ** 2 + 0.05 ** 2 + 0.03 ** 2)
        distance = 4.75 * radius
        np.testing.assert_allclose(validation["center_m"], center, atol=1e-15, rtol=0.)
        self.assertAlmostEqual(validation["radius_m"], radius, places=14)
        self.assertAlmostEqual(validation["distance_m"], distance, places=14)
        self.assertEqual(len(validation["poses"]), 8 * 24)
        for yaw0 in (0, 90, 180, 270):
            for pitch in (-25, 25):
                pose = validation["poses"][(yaw0, pitch, 0)]
                rotation = pose[:3, :3]
                self.assertAlmostEqual(np.linalg.det(rotation), 1., places=14)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14, rtol=0.)
                center_camera = rotation @ center + pose[:3, 3]
                np.testing.assert_allclose(center_camera, [0., 0., distance], atol=1e-14, rtol=0.)
                np.testing.assert_allclose(pose @ np.linalg.inv(pose), np.eye(4), atol=1e-14, rtol=0.)
        pose = broader_trajectory_pose(0, yaw0_deg=0, pitch_deg=25,
                                       center_m=center, radius_m=radius)
        angle = math.radians(25)
        expected_rotation = np.array([[1., 0., 0.],
            [0., math.cos(angle), -math.sin(angle)],
            [0., math.sin(angle), math.cos(angle)]])
        expected_translation = np.array([
            -1.2,
            -(-0.4 * math.cos(angle) - 2.0 * math.sin(angle)),
            distance - (-0.4 * math.sin(angle) + 2.0 * math.cos(angle)),
        ])
        np.testing.assert_allclose(pose[:3, :3], expected_rotation, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(pose[:3, 3], expected_translation, atol=1e-14, rtol=0.)
        pixels = validation["poses"][(90, -25, 23)]
        camera_vertices = vertices @ pixels[:3, :3].T + pixels[:3, 3]
        projected = camera_vertices @ validation["intrinsics"].T
        projected = projected[:, :2] / projected[:, 2:3]
        self.assertTrue(np.isfinite(projected).all())
        self.assertTrue(np.all((projected[:, 0] >= 0.) & (projected[:, 0] <= 639.)))
        self.assertTrue(np.all((projected[:, 1] >= 0.) & (projected[:, 1] <= 479.)))

    def test_broader_geometry_rejects_degenerate_nonfinite_clip_skew_and_bounds(self):
        with self.assertRaisesRegex(ValueError, "degenerate radius"):
            validate_broader_trajectory(np.zeros((4, 3)))
        broken = _off_origin_box()
        broken[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite Nx3"):
            validate_broader_trajectory(broken)
        with self.assertRaisesRegex(ValueError, "clip bounds"):
            validate_broader_trajectory(np.array([[-100., 0., 0.], [100., 0., 0.]]))
        skewed = [[576., 1., 319.5], [0., 576., 239.5], [0., 0., 1.]]
        with self.assertRaisesRegex(ValueError, "zero-skew"):
            validate_broader_trajectory(_off_origin_box(), skewed)
        offscreen_center = [[576., 0., 640.], [0., 576., 239.5], [0., 0., 1.]]
        with self.assertRaisesRegex(ValueError, "principal point"):
            validate_broader_trajectory(_off_origin_box(), offscreen_center)
        too_narrow = [[10000., 0., 319.5], [0., 10000., 239.5], [0., 0., 1.]]
        with self.assertRaisesRegex(ValueError, "projection leaves"):
            validate_broader_trajectory(_off_origin_box(), too_narrow)

    def test_broader_occluder_has_exact_inclusive_endpoints_and_half_up_geometry(self):
        clear = np.full((480, 640, 3), 127, dtype=np.uint8)
        self.assertEqual(composite_broader_occluder(clear.copy(), 5)[0, 0, 0], clear[0, 0, 0])
        first = composite_broader_occluder(clear.copy(), 6)
        self.assertTrue(np.all(first[72:408, 58:199] == [32, 32, 32]))
        self.assertTrue(np.all(first[71, 58:199] == [127, 127, 127]))
        self.assertTrue(np.all(first[72:408, 57] == [127, 127, 127]))
        last = composite_broader_occluder(clear.copy(), 13)
        self.assertTrue(np.all(last[72:408, 442:583] == [32, 32, 32]))
        self.assertTrue(np.all(composite_broader_occluder(clear.copy(), 14) == clear))

    def test_legacy_pose_and_default_manifest_camera_remain_unchanged(self):
        pose = trajectory_pose(0, 90)
        angle = 0.12
        self.assertEqual(pose[0].tolist(), [1., 0., 0., 0.])
        np.testing.assert_allclose(pose[1], [0., math.cos(angle), -math.sin(angle), .025], atol=1e-15)
        np.testing.assert_allclose(pose[2], [0., math.sin(angle), math.cos(angle), .62], atol=1e-15)
        mesh_sha = "e" * 64
        manifest = make_input_manifest(mesh_sha256=mesh_sha, video_sha256="f" * 64,
            frames=2, width=640, height=480, fps=10.,
            sequence_group=SyntheticBundleTests()._frozen_group(mesh_sha))
        self.assertEqual(manifest["intrinsics"], [[576., 0., 319.5], [0., 576., 239.5], [0., 0., 1.]])
        self.assertEqual([row["timestamp_s"] for row in manifest["timeline"]], [0., .1])


class OrientationSyntheticGeometryTests(unittest.TestCase):
    def test_fixed_endpoints_interior_translation_projection_inverse_and_time(self):
        vertices = _off_origin_box()
        validation = validate_orientation_trajectory(vertices)
        center = np.array([1.2, -0.4, 2.0])
        radius = math.sqrt(0.08 ** 2 + 0.05 ** 2 + 0.03 ** 2)
        distance = 4.75 * radius
        self.assertEqual(len(validation["poses"]), 24)
        np.testing.assert_allclose(validation["center_m"], center, atol=1e-15, rtol=0.)
        self.assertAlmostEqual(validation["distance_m"], distance, places=14)
        for yaw0, pitch in ((45, -15), (225, 15)):
            for frame in range(12):
                pose = validation["poses"][(yaw0, pitch, frame)]
                np.testing.assert_allclose(pose @ np.linalg.inv(pose), np.eye(4),
                                           atol=1e-10, rtol=0.)
                camera_vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
                projected = camera_vertices @ validation["intrinsics"].T
                projected = projected[:, :2] / projected[:, 2:3]
                self.assertTrue(np.all((projected[:, 0] >= 0.) & (projected[:, 0] <= 639.)))
                self.assertTrue(np.all((projected[:, 1] >= 0.) & (projected[:, 1] <= 479.)))
        interior = orientation_trajectory_pose(
            5, yaw0_deg=45, pitch_deg=-15, center_m=center, radius_m=radius)
        yaw, pitch = math.radians(45 + 100 / 11), math.radians(-15)
        expected_rotation = np.array([
            [math.cos(yaw), math.sin(yaw) * math.sin(pitch),
             math.sin(yaw) * math.cos(pitch)],
            [0., math.cos(pitch), -math.sin(pitch)],
            [-math.sin(yaw), math.cos(yaw) * math.sin(pitch),
             math.cos(yaw) * math.cos(pitch)],
        ])
        np.testing.assert_allclose(interior[:3, :3], expected_rotation, atol=1e-14, rtol=0.)
        np.testing.assert_allclose(interior[:3, 3],
                                   np.array([0., 0., distance]) - expected_rotation @ center,
                                   atol=1e-14, rtol=0.)
        endpoint = validation["poses"][(45, -15, 11)]
        expected_endpoint = np.array([
            [math.cos(math.radians(65)),
             math.sin(math.radians(65)) * math.sin(math.radians(-15)),
             math.sin(math.radians(65)) * math.cos(math.radians(-15))],
            [0., math.cos(math.radians(-15)), -math.sin(math.radians(-15))],
            [-math.sin(math.radians(65)),
             math.cos(math.radians(65)) * math.sin(math.radians(-15)),
             math.cos(math.radians(65)) * math.cos(math.radians(-15))],
        ])
        np.testing.assert_allclose(endpoint[:3, :3], expected_endpoint, atol=1e-14, rtol=0.)

        mesh_sha = "a" * 64
        frozen = freeze_split_manifest({"records": [{
            "physical_object_id": "orientation-time-object", "object_id": "orientation-time-object",
            "scene_id": "orientation-time-scene", "sequence_id": "orientation-time-sequence",
            "background_id": "procedural:gray", "background_kind": "procedural",
            "provenance_kind": "synthetic", "mesh_sha256": mesh_sha,
            "frame_id": ordinal, "source_frame_id": ordinal, "split": "train",
        } for ordinal in range(12)]}, label_policy={
            "max_rotation_error_deg": 10., "max_translation_error_m": .02})
        manifest = make_input_manifest(
            mesh_sha256=mesh_sha, video_sha256="b" * 64, frames=12,
            width=640, height=480, fps=10., sequence_group=frozen["groups"][0])
        self.assertEqual(manifest["timeline"][-1]["timestamp_s"], 1.1)

    def test_invalid_indices_pairs_numbers_geometry_projection_and_clip_fail_closed(self):
        center = np.zeros(3)
        for bad_frame in (-1, 12, True):
            with self.subTest(frame=bad_frame), self.assertRaises(ValueError):
                orientation_trajectory_pose(bad_frame, yaw0_deg=45, pitch_deg=-15,
                                            center_m=center, radius_m=.1)
        for pair in ((45, 15), (225, -15), (0, -15)):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "paired"):
                orientation_trajectory_pose(0, yaw0_deg=pair[0], pitch_deg=pair[1],
                                            center_m=center, radius_m=.1)
        for radius in (True, 0., float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                orientation_trajectory_pose(0, yaw0_deg=45, pitch_deg=-15,
                                            center_m=center, radius_m=radius)
        with self.assertRaisesRegex(ValueError, "finite Nx3"):
            validate_orientation_trajectory(np.array([[0., 0., np.nan], [1., 0., 0.]]))
        with self.assertRaisesRegex(ValueError, "degenerate radius"):
            validate_orientation_trajectory(np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "overflow|degenerate"):
            validate_orientation_trajectory(np.array([[-1e308, 0., 0.], [1e308, 0., 0.]]))
        with self.assertRaisesRegex(ValueError, "clip bounds"):
            validate_orientation_trajectory(np.array([[-100., 0., 0.], [100., 0., 0.]]))
        with self.assertRaisesRegex(ValueError, "zero-skew"):
            validate_orientation_trajectory(
                _off_origin_box(), [[576., 1., 319.5], [0., 576., 239.5], [0., 0., 1.]])
        with self.assertRaisesRegex(ValueError, "projection leaves"):
            validate_orientation_trajectory(
                _off_origin_box(), [[10000., 0., 319.5], [0., 10000., 239.5], [0., 0., 1.]])


def _orientation_preparation_module():
    path = Path(__file__).resolve().parents[1] / "scripts/cloud_ranker_prepare_orientation.py"
    spec = importlib.util.spec_from_file_location("cloud_orientation_preparation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_orientation_fixture(root, output_name="orientation-frozen"):
    module = _orientation_preparation_module()
    assets = root / "orientation-assets"
    assets.mkdir(exist_ok=True)
    specs = []
    for number, split in ((17, "train"), (27, "validation")):
        payload = f"orientation software fixture {number}".encode()
        path = assets / f"obj_{number:06d}.glb"
        if not path.exists():
            path.write_bytes(payload)
        specs.append((number, split, len(payload), hashlib.sha256(payload).hexdigest()))
    output = root / output_name
    pins = {f"hot3d:obj_{number:06d}": digest
            for number, _split, _size, digest in specs}
    with patch.object(module, "ASSETS", tuple(specs)), \
            patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins), \
            patch.object(module.platform, "system", return_value="Linux"), \
            patch.dict(module.os.environ, {"VISUALIZEIT_CLOUD": "1"}):
        module.main(["--assets", str(assets), "--output-dir", str(output)])
    return module, assets, output, tuple(specs)


class OrientationRecipeAndPreparationTests(unittest.TestCase):
    def test_preparation_is_deterministic_strict_and_does_not_claim_observability(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            _, assets, first, specs = _prepare_orientation_fixture(root, "first")
            _prepare_orientation_fixture(root, "second")
            for name in ("inventory.json", "split.json", "recipe.json", "preparation.json"):
                self.assertEqual((first / name).read_bytes(), (root / "second" / name).read_bytes())
            self.assertEqual({path.name for path in first.iterdir()},
                             {"inventory.json", "split.json", "recipe.json", "preparation.json"})
            recipe = json.loads((first / "recipe.json").read_text())
            self.assertEqual(recipe["kind"], "quality-ranker-synthetic-orientation-recipe-v1")
            self.assertEqual([(row["yaw0_deg"], row["pitch_deg"]) for row in recipe["sequences"]],
                             [(45, -15), (45, -15), (225, 15), (225, 15)])
            self.assertTrue(all(row["observationally_ambiguous"] is True
                                for row in recipe["sequences"]))
            split = json.loads((first / "split.json").read_text())
            self.assertEqual(len(split["groups"]), 4)
            self.assertEqual(sum(len(group["frames"]) for group in split["groups"]), 48)
            pins = {f"hot3d:obj_{number:06d}": digest
                    for number, _split, _size, digest in specs}
            with patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins):
                for group in split["groups"]:
                    number = 17 if group["split"] == "train" else 27
                    context = validate_recipe(
                        first / "recipe.json", first / "split.json",
                        assets / f"obj_{number:06d}.glb", group["sequence_id"],
                        frames=12, width=640, height=480, fps=10.,
                        mesh_units="metres", observationally_ambiguous=True)
                    self.assertEqual(context["group"], group)
            receipt = json.loads((first / "preparation.json").read_text())
            self.assertTrue(receipt["orientation_review_required_before_candidate_generation"])
            self.assertFalse(receipt["orientation_review_created"])
            self.assertTrue(receipt["raw_labels_observationally_ambiguous"])
            self.assertEqual(receipt["gpu_processes_started"], 0)
            self.assertEqual(receipt["new_model_weights_bytes"], 0)
            self.assertEqual(receipt["external_budget_constraints"]["credit_floor"], 3.71)

    def test_recipe_tampering_ownership_source_duplicates_counts_and_ambiguity_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            _, assets, output, specs = _prepare_orientation_fixture(root)
            pins = {f"hot3d:obj_{number:06d}": digest
                    for number, _split, _size, digest in specs}
            recipe_path, split_path = output / "recipe.json", output / "split.json"
            original = json.loads(recipe_path.read_text())
            selected = original["sequences"][0]
            mesh = assets / "obj_000017.glb"
            args = dict(frames=12, width=640, height=480, fps=10.,
                        mesh_units="metres", observationally_ambiguous=True)
            # The fake recipe, split and mesh are mutually consistent. Production
            # validation still rejects them because HOT3D identities own exact pins.
            with self.assertRaisesRegex(ValueError, "pinned HOT3D"):
                validate_orientation_recipe(recipe_path, split_path, mesh,
                                            selected["sequence_id"], **args)

            mutations = []
            unknown = json.loads(json.dumps(original)); unknown["unexpected"] = 1
            mutations.append(unknown)
            digest = json.loads(json.dumps(original)); digest["recipe_id"] = "mutated-without-resealing"
            mutations.append(digest)
            digest_field = json.loads(json.dumps(original))
            old_digest = digest_field["recipe_sha256"]
            digest_field["recipe_sha256"] = (
                ("0" if old_digest[0] != "0" else "1") + old_digest[1:])
            mutations.append(digest_field)
            ownership = json.loads(json.dumps(original)); ownership["sequences"][0]["split"] = "validation"
            mutations.append(seal_recipe(ownership))
            duplicate = json.loads(json.dumps(original)); duplicate["sequences"][1]["sequence_id"] = duplicate["sequences"][0]["sequence_id"]
            mutations.append(seal_recipe(duplicate))
            ambiguity = json.loads(json.dumps(original)); ambiguity["sequences"][0]["observationally_ambiguous"] = False
            mutations.append(seal_recipe(ambiguity))
            counts = json.loads(json.dumps(original)); counts["options"]["frames"] = 11
            mutations.append(seal_recipe(counts))
            source = json.loads(json.dumps(original))
            old_source_sha = source["source_binding"]["generator_sha256"]
            source["source_binding"]["generator_sha256"] = (
                ("0" if old_source_sha[0] != "0" else "1") + old_source_sha[1:])
            mutations.append(seal_recipe(source))
            boolean_numeric = json.loads(json.dumps(original)); boolean_numeric["options"]["fps"] = True
            mutations.append(seal_recipe(boolean_numeric))
            nonfinite = json.loads(json.dumps(original)); nonfinite["options"]["fps"] = float("inf")
            mutations.append(nonfinite)
            for index, mutation in enumerate(mutations):
                with self.subTest(mutation=index):
                    recipe_path.write_text(json.dumps(mutation, sort_keys=True), encoding="utf-8")
                    with patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins), \
                            self.assertRaises(ValueError):
                        validate_orientation_recipe(recipe_path, split_path, mesh,
                                                    selected["sequence_id"], **args)

            recipe_path.write_text(json.dumps(original), encoding="utf-8")
            split_original = split_path.read_bytes()
            split_path.write_bytes(split_original + b"\n")
            with patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins), \
                    self.assertRaisesRegex(ValueError, "split bytes"):
                validate_orientation_recipe(recipe_path, split_path, mesh,
                                            selected["sequence_id"], **args)
            split_path.write_bytes(split_original)
            mesh.write_bytes(b"mutated mesh")
            with patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins), \
                    self.assertRaisesRegex(ValueError, "mesh bytes"):
                validate_orientation_recipe(recipe_path, split_path, mesh,
                                            selected["sequence_id"], **args)
            with patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins), \
                    self.assertRaisesRegex(ValueError, "generation options"):
                validate_orientation_recipe(
                    recipe_path, split_path, assets / "obj_000027.glb",
                    original["sequences"][1]["sequence_id"],
                    **{**args, "observationally_ambiguous": False})

    def test_guard_existing_destination_and_symlink_leave_existing_bytes_unchanged(self):
        module = _orientation_preparation_module()
        with patch.object(module.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "approved Linux cloud"):
                module.main(["--assets", "absent", "--output-dir", "absent"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            module, assets, _, specs = _prepare_orientation_fixture(root, "baseline")
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "marker.bin"
            marker.write_bytes(b"preserve me")
            with patch.object(module, "ASSETS", specs), \
                    patch.object(module.platform, "system", return_value="Linux"), \
                    patch.dict(module.os.environ, {"VISUALIZEIT_CLOUD": "1"}):
                with self.assertRaises(FileExistsError):
                    module.main(["--assets", str(assets), "--output-dir", str(existing)])
            self.assertEqual(marker.read_bytes(), b"preserve me")
            target = root / "symlink-target"
            target.mkdir()
            target_marker = target / "marker.bin"
            target_marker.write_bytes(b"preserve symlink target")
            link = root / "output-link"
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")
            with patch.object(module, "ASSETS", specs), \
                    patch.object(module.platform, "system", return_value="Linux"), \
                    patch.dict(module.os.environ, {"VISUALIZEIT_CLOUD": "1"}):
                with self.assertRaises(FileExistsError):
                    module.main(["--assets", str(assets), "--output-dir", str(link)])
            self.assertEqual(target_marker.read_bytes(), b"preserve symlink target")

    def test_mock_orientation_render_keeps_raw_labels_outside_three_file_input(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            _, assets, prepared, specs = _prepare_orientation_fixture(root)
            pins = {f"hot3d:obj_{number:06d}": digest
                    for number, _split, _size, digest in specs}
            split = json.loads((prepared / "split.json").read_text())
            group = next(group for group in split["groups"] if group["split"] == "train")
            written_frames = []

            class FakeWriter:
                def __init__(self, path, *_args):
                    self.path = Path(path)
                    self.path.write_bytes(b"fake-mp4")

                def isOpened(self):
                    return True

                def write(self, image):
                    written_frames.append(np.asarray(image).copy())
                    with self.path.open("ab") as stream:
                        stream.write(b"f")

                def release(self):
                    return None

            class FakeRenderer:
                def __init__(self, *_args, **_kwargs):
                    self.vertices_m = _off_origin_box()

                def render_object_model(self, *_args):
                    return {"color": np.full((480, 640, 3), .75, dtype=np.float64),
                            "mask": np.ones((480, 640), dtype=bool)}

                def close(self):
                    return None

            fake_cv2 = types.SimpleNamespace(
                VideoWriter=lambda *args: FakeWriter(*args),
                VideoWriter_fourcc=lambda *_args: 0,
                cvtColor=lambda image, _code: image,
                COLOR_RGB2BGR=1,
            )
            fake_gotrack = types.ModuleType("bench.quality_gotrack")
            fake_gotrack.TexturedRenderer = FakeRenderer
            fake_gotrack.upstream_path = lambda: None
            fake_vision = types.ModuleType("bench.vision")
            fake_vision.cv2 = fake_cv2
            fake_utils = types.ModuleType("utils")
            fake_utils.renderer_base = types.SimpleNamespace(
                RenderType=types.SimpleNamespace(COLOR="color", MASK="mask"))
            fake_utils.structs = types.SimpleNamespace(
                PinholePlaneCameraModel=lambda **kwargs: kwargs)
            with patch.object(synthetic_module, "ORIENTATION_ASSET_PINS", pins), \
                    patch.dict(sys.modules, {
                    "bench.quality_gotrack": fake_gotrack,
                    "bench.vision": fake_vision,
                    "utils": fake_utils,
                }), patch("platform.system", return_value="Linux"):
                result = _render_bundle(
                    assets / "obj_000017.glb", root / "rendered", group,
                    frames=12, width=640, height=480, fps=10., mesh_units="metres",
                    observationally_ambiguous=True, cloud_execution=True,
                    recipe_path=prepared / "recipe.json",
                    split_manifest_path=prepared / "split.json")
            self.assertEqual(len(written_frames), 12)
            self.assertTrue(all(np.all(frame == 191) for frame in written_frames))
            bundle = root / "rendered" / "bundle"
            self.assertEqual(sorted(path.name for path in bundle.iterdir()),
                             ["input.json", "object.glb", "synthetic.mp4"])
            manifest = json.loads((bundle / "input.json").read_text())
            self.assertNotIn("pose", manifest)
            self.assertNotIn("observationally_ambiguous", manifest)
            labels = [json.loads(line) for line in
                      (root / "rendered" / "poses.jsonl").read_text().splitlines()]
            self.assertEqual(len(labels), 12)
            self.assertTrue(all(row["observationally_ambiguous"] is True for row in labels))
            recipe = json.loads((prepared / "recipe.json").read_text())
            selected = next(row for row in recipe["sequences"]
                            if row["sequence_id"] == group["sequence_id"])
            expected = orientation_trajectory_pose(
                0, yaw0_deg=selected["yaw0_deg"], pitch_deg=selected["pitch_deg"],
                center_m=np.array([1.2, -.4, 2.]),
                radius_m=math.sqrt(.08**2 + .05**2 + .03**2))
            np.testing.assert_allclose(labels[0]["pose"], expected, atol=1e-14, rtol=0.)
            generation = json.loads((root / "rendered" / "generation.json").read_text())
            self.assertTrue(generation["orientation_review_required_before_candidate_generation"])
            self.assertFalse(result["labels_in_inference_manifest"])


def _single_view_preparation_module():
    path = Path(__file__).resolve().parents[1] / "scripts/cloud_ranker_prepare_single_view.py"
    spec = importlib.util.spec_from_file_location("cloud_single_view_preparation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_single_view_fixture(root, output_name="single-view-frozen"):
    module = _single_view_preparation_module()
    assets = root / "existing-assets"
    assets.mkdir(exist_ok=True)
    payload = b"single-view software fixture hot3d17"
    mesh = assets / "obj_000017.glb"
    mesh.write_bytes(payload)
    spec = (17, "train", len(payload), hashlib.sha256(payload).hexdigest())
    output = root / output_name
    with patch.object(module, "ASSET", spec), \
            patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]), \
            patch.object(module.platform, "system", return_value="Linux"), \
            patch.dict(module.os.environ, {"VISUALIZEIT_CLOUD": "1"}):
        module.main(["--assets", str(assets), "--output-dir", str(output)])
    return module, assets, output, spec


class SingleViewRecipeAndPreparationTests(unittest.TestCase):
    def test_preparation_freezes_one_train_frame_and_current_bundle_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            _, assets, output, spec = _prepare_single_view_fixture(root)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"inventory.json", "split.json", "recipe.json", "preparation.json"})
            recipe = json.loads((output / "recipe.json").read_text())
            self.assertEqual(recipe["kind"],
                             "quality-ranker-synthetic-single-view-recipe-v1")
            self.assertEqual(recipe["recipe_id"], "ranker-single-view-synthetic-v1")
            self.assertEqual(len(recipe["sequences"]), 1)
            sequence = recipe["sequences"][0]
            self.assertEqual(sequence["sequence_id"],
                             "ranker-single-view-hot3d17-v1-yaw75-pitchm15")
            self.assertEqual(sequence["scene_id"],
                             "synthetic-scene:hot3d17:single-view-v1:yaw75-pitchm15")
            self.assertEqual((sequence["yaw0_deg"], sequence["pitch_deg"]), (75, -15))
            self.assertEqual(sequence["split"], "train")
            self.assertTrue(sequence["observationally_ambiguous"] is True)
            split = json.loads((output / "split.json").read_text())
            self.assertEqual(len(split["groups"]), 1)
            self.assertEqual(split["groups"][0]["frames"],
                             [{"frame_id": 0, "source_frame_id": 0}])
            self.assertEqual({group["split"] for group in split["groups"]}, {"train"})
            with patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]):
                context = validate_recipe(
                    output / "recipe.json", output / "split.json",
                    assets / "obj_000017.glb", sequence["sequence_id"],
                    frames=1, width=640, height=480, fps=10.,
                    mesh_units="metres", observationally_ambiguous=True)
            self.assertEqual(context["group"], split["groups"][0])
            receipt = json.loads((output / "preparation.json").read_text())
            self.assertEqual(receipt["splits"], ["train"])
            self.assertFalse(receipt["raw_labels_in_inference_bundle"])
            self.assertEqual(receipt["planned_initialization_candidates"], 5)
            self.assertFalse(receipt["candidate_generation_started"])
            self.assertFalse(receipt["witness_labels_created"])
            self.assertFalse(receipt["training_started"])
            self.assertFalse(receipt["quality_claim_created"])

    def test_single_view_pose_is_proper_visible_metric_object_to_camera_transform(self):
        vertices = _off_origin_box()
        validation = validate_single_view_trajectory(vertices)
        self.assertEqual(set(validation["poses"]), {(75, -15, 0)})
        pose = validation["poses"][(75, -15, 0)]
        center = np.array([1.2, -0.4, 2.0])
        radius = math.sqrt(0.08 ** 2 + 0.05 ** 2 + 0.03 ** 2)
        expected = orientation_trajectory_pose(
            0, yaw0_deg=75, pitch_deg=-15, center_m=center, radius_m=radius)
        # Analytical source: frame 0 makes yaw = yaw0 in the declared RyRx
        # trajectory, with t = [0,0,d] - R*mesh_center.
        np.testing.assert_allclose(pose, expected, atol=1e-14, rtol=0.)
        rotation = pose[:3, :3]
        # SO(3) definition and exact homogeneous inverse identity.
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14, rtol=0.)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=14)
        inverse = np.linalg.inv(pose)
        np.testing.assert_allclose(pose @ inverse, np.eye(4), atol=1e-14, rtol=0.)
        camera_points = vertices @ rotation.T + pose[:3, 3]
        self.assertTrue(np.all(camera_points[:, 2] > 0.01))
        pixels_h = camera_points @ validation["intrinsics"].T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        self.assertTrue(np.all((pixels[:, 0] >= 0.) & (pixels[:, 0] <= 639.)))
        self.assertTrue(np.all((pixels[:, 1] >= 0.) & (pixels[:, 1] <= 479.)))

    def test_recipe_rejects_pose_count_options_source_split_mesh_identity_and_false_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            _, assets, output, spec = _prepare_single_view_fixture(root)
            recipe_path, split_path = output / "recipe.json", output / "split.json"
            original = json.loads(recipe_path.read_text())
            sequence_id = original["sequences"][0]["sequence_id"]
            mesh = assets / "obj_000017.glb"
            args = dict(frames=1, width=640, height=480, fps=10.,
                        mesh_units="metres", observationally_ambiguous=True)
            mutations = []
            pose = json.loads(json.dumps(original))
            pose["sequences"][0]["yaw0_deg"] = 76
            mutations.append(seal_recipe(pose))
            trajectory = json.loads(json.dumps(original))
            trajectory["options"]["trajectory"]["frame_index"] = 1
            mutations.append(seal_recipe(trajectory))
            count = json.loads(json.dumps(original))
            count["options"]["frames"] = 2
            mutations.append(seal_recipe(count))
            options = json.loads(json.dumps(original))
            options["options"]["background_rgb"] = [128, 127, 127]
            mutations.append(seal_recipe(options))
            source = json.loads(json.dumps(original))
            source["source_binding"]["preparation_path"] = "scripts/wrong.py"
            mutations.append(seal_recipe(source))
            source_digest = json.loads(json.dumps(original))
            source_digest["source_binding"]["generator_sha256"] = "0" * 63 + "1"
            mutations.append(seal_recipe(source_digest))
            identity = json.loads(json.dumps(original))
            identity["sequences"][0]["physical_object_id"] = "hot3d:obj_000027"
            mutations.append(seal_recipe(identity))
            ambiguity = json.loads(json.dumps(original))
            ambiguity["sequences"][0]["observationally_ambiguous"] = False
            mutations.append(seal_recipe(ambiguity))
            for index, mutation in enumerate(mutations):
                with self.subTest(mutation=index):
                    recipe_path.write_text(json.dumps(mutation, sort_keys=True), encoding="utf-8")
                    with patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]), \
                            self.assertRaises(ValueError):
                        validate_single_view_recipe(
                            recipe_path, split_path, mesh, sequence_id, **args)

            recipe_path.write_text(json.dumps(original), encoding="utf-8")
            split_bytes = split_path.read_bytes()
            split_path.write_bytes(split_bytes + b"\n")
            with patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]), \
                    self.assertRaisesRegex(ValueError, "split bytes"):
                validate_single_view_recipe(recipe_path, split_path, mesh, sequence_id, **args)
            split_path.write_bytes(split_bytes)
            mesh.write_bytes(b"changed mesh")
            with patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]), \
                    self.assertRaisesRegex(ValueError, "mesh bytes"):
                validate_single_view_recipe(recipe_path, split_path, mesh, sequence_id, **args)
            mesh.write_bytes(b"single-view software fixture hot3d17")
            with patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]), \
                    self.assertRaisesRegex(ValueError, "generation options"):
                validate_single_view_recipe(
                    recipe_path, split_path, mesh, sequence_id,
                    **{**args, "observationally_ambiguous": False})

    def test_mock_render_uses_same_pose_for_rgb_and_raw_label_and_reads_as_physical_bundle(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            _, assets, prepared, spec = _prepare_single_view_fixture(root)
            split = json.loads((prepared / "split.json").read_text())
            group = split["groups"][0]
            camera_transforms = []

            class FakeWriter:
                def __init__(self, path, *_args):
                    self.path = Path(path)
                    self.path.write_bytes(b"fake-mp4")

                def isOpened(self):
                    return True

                def write(self, _image):
                    with self.path.open("ab") as stream:
                        stream.write(b"f")

                def release(self):
                    return None

            class FakeRenderer:
                def __init__(self, *_args, **_kwargs):
                    self.vertices_m = _off_origin_box()

                def render_object_model(self, _object_id, camera):
                    camera_transforms.append(np.asarray(camera["T_world_from_eye"]).copy())
                    return {"color": np.full((480, 640, 3), .5, dtype=np.float64),
                            "mask": np.ones((480, 640), dtype=bool)}

                def close(self):
                    return None

            fake_cv2 = types.SimpleNamespace(
                VideoWriter=lambda *args: FakeWriter(*args),
                VideoWriter_fourcc=lambda *_args: 0,
                cvtColor=lambda image, _code: image,
                COLOR_RGB2BGR=1,
            )
            fake_gotrack = types.ModuleType("bench.quality_gotrack")
            fake_gotrack.TexturedRenderer = FakeRenderer
            fake_gotrack.upstream_path = lambda: None
            fake_vision = types.ModuleType("bench.vision")
            fake_vision.cv2 = fake_cv2
            fake_utils = types.ModuleType("utils")
            fake_utils.renderer_base = types.SimpleNamespace(
                RenderType=types.SimpleNamespace(COLOR="color", MASK="mask"))
            fake_utils.structs = types.SimpleNamespace(
                PinholePlaneCameraModel=lambda **kwargs: kwargs)
            with patch.object(synthetic_module, "SINGLE_VIEW_ASSET_PIN", spec[3]), \
                    patch.dict(sys.modules, {
                        "bench.quality_gotrack": fake_gotrack,
                        "bench.vision": fake_vision,
                        "utils": fake_utils,
                    }), patch("platform.system", return_value="Linux"):
                result = _render_bundle(
                    assets / "obj_000017.glb", root / "rendered", group,
                    frames=1, width=640, height=480, fps=10., mesh_units="metres",
                    observationally_ambiguous=True, cloud_execution=True,
                    recipe_path=prepared / "recipe.json",
                    split_manifest_path=prepared / "split.json")
            labels = [json.loads(line) for line in
                      (root / "rendered" / "poses.jsonl").read_text().splitlines()]
            self.assertEqual(len(labels), 1)
            self.assertTrue(labels[0]["observationally_ambiguous"] is True)
            pose = np.asarray(labels[0]["pose"])
            expected_renderer_transform = np.linalg.inv(pose)
            expected_renderer_transform[:3, 3] *= 1000.
            # Renderer API uses the exact inverse of the metric raw label, with
            # only its documented metre-to-millimetre translation conversion.
            np.testing.assert_allclose(camera_transforms[0], expected_renderer_transform,
                                       atol=1e-10, rtol=0.)
            bundle = root / "rendered" / "bundle"
            self.assertEqual(sorted(path.name for path in bundle.iterdir()),
                             ["input.json", "object.glb", "synthetic.mp4"])
            manifest = json.loads((bundle / "input.json").read_text())
            self.assertNotIn("pose", manifest)
            self.assertNotIn("observationally_ambiguous", manifest)
            from .quality_runner import read_input
            parsed = read_input(bundle)
            self.assertEqual(parsed["frame_ids"], [0])
            self.assertEqual(parsed["_timing"]["clock_mode"], "capture_seconds_v1")
            self.assertEqual(len(parsed["_timing"]["scored_keys"]), 1)
            generation = json.loads((root / "rendered" / "generation.json").read_text())
            self.assertTrue(
                generation["orientation_review_required_before_candidate_generation"])
            self.assertFalse(result["labels_in_inference_manifest"])


class NormalizedViewTests(unittest.TestCase):
    def test_off_origin_geometry_and_all_independent_directions(self):
        vertices = _off_origin_box()
        result = synthetic_module.validate_normalized_views(vertices)
        center = (vertices.min(0) + vertices.max(0)) / 2
        radius = np.linalg.norm(vertices-center, axis=1).max()
        np.testing.assert_allclose(result['center_m'], center, atol=1e-15)
        self.assertAlmostEqual(result['radius_m'], radius)
        self.assertAlmostEqual(result['distance_m'], 4.75*radius)
        directions = []
        for i, pose in result['poses'].items():
            elevation = math.radians((-60, 0, 60)[i//8])
            azimuth = math.radians(45*(i % 8))
            expected = [-math.cos(elevation)*math.sin(azimuth), math.sin(elevation),
                        math.cos(elevation)*math.cos(azimuth)]
            rotation = pose[:3, :3]
            np.testing.assert_allclose(rotation.T @ [0., 0., 1.], expected, atol=1e-14)
            np.testing.assert_allclose(rotation @ center + pose[:3, 3],
                                       [0., 0., 4.75*radius], atol=1e-14)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            np.testing.assert_allclose(np.linalg.inv(pose) @ pose, np.eye(4), atol=1e-14)
            xyz = vertices @ rotation.T + pose[:3, 3]
            uvw = xyz @ result['intrinsics'].T
            uv = uvw[:, :2] / uvw[:, 2:]
            self.assertTrue((xyz[:, 2] > .01).all() and (xyz[:, 2] < 30.).all())
            self.assertTrue((uv >= [0, 0]).all() and (uv <= [639, 479]).all())
            directions.append(tuple(np.round(expected, 12)))
        self.assertEqual(len(set(directions)), 24)
        np.testing.assert_allclose(result['poses'][8][:3, :3], np.eye(3), atol=1e-14)
        np.testing.assert_allclose(result['poses'][10][:3, :3],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-14)

    def test_translation_equivariance_and_scale_projection_invariance(self):
        vertices = _off_origin_box()
        offset = np.array([3., -2., .5])
        initial = synthetic_module.validate_normalized_views(vertices)
        shifted = synthetic_module.validate_normalized_views(vertices+offset)
        scaled = synthetic_module.validate_normalized_views(vertices*2)
        for i, pose in initial['poses'].items():
            xyz = vertices @ pose[:3, :3].T + pose[:3, 3]
            moved = (vertices+offset) @ shifted['poses'][i][:3, :3].T + shifted['poses'][i][:3, 3]
            enlarged = (vertices*2) @ scaled['poses'][i][:3, :3].T + scaled['poses'][i][:3, 3]
            np.testing.assert_allclose(xyz, moved, atol=1e-14)
            np.testing.assert_allclose(xyz[:, :2]/xyz[:, 2:],
                                       enlarged[:, :2]/enlarged[:, 2:], atol=1e-14)

    def test_invalid_geometry_and_indices_reject(self):
        for vertices in ([], [[0, 0, 0]], [[float('nan'), 0, 0]],
                         _off_origin_box()*1000, _off_origin_box()*1e-5):
            with self.subTest(vertices=repr(vertices)[:50]), self.assertRaises(ValueError):
                synthetic_module.validate_normalized_views(vertices)
        for index in (-1, 24, True, .5, None):
            with self.subTest(index=index), self.assertRaises(ValueError):
                synthetic_module.normalized_view_pose(index, center_m=[0, 0, 0], radius_m=.1)

    def test_incompatible_options_reject_before_renderer_import(self):
        valid = dict(view_mode='normalized24-v1', view_index=0, frames=1, width=640,
                     height=480, fps=10, observationally_ambiguous=True, recipe_path=None)
        self.assertTrue(synthetic_module._validate_view_options(**valid))
        for key, value in [('view_mode', 'unknown'), ('view_index', None), ('frames', 2),
                           ('width', 320), ('height', 240), ('fps', 30),
                           ('observationally_ambiguous', False), ('recipe_path', Path('x'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                synthetic_module._validate_view_options(**{**valid, key:value})
        with self.assertRaises(ValueError):
            synthetic_module._validate_view_options(**{**valid, 'view_mode':'legacy'})

    def test_mock_render_pose_label_receipt_and_failure_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
                root = Path(temp)
                mesh = root/'mesh.glb'
                mesh.write_bytes(b'clearly identified software fixture, not a training mesh')
                digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
                records = [dict(physical_object_id='synthetic_metric_asset',
                    sequence_id='normalized-fixture-view10', scene_id='normalized-fixture-scene',
                    split='train', frame_id=0, source_frame_id=0, mesh_sha256=digest,
                    provenance_kind='synthetic', background_kind='procedural', background_id='procedural:gray')]
                frozen = freeze_split_manifest({'records':records}, label_policy={
                    'max_rotation_error_deg':10., 'max_translation_error_m':.02})
                split = root/'split.json'
                split.write_text(json.dumps(frozen))
                observed, closed = [], []
                class Writer:
                    def __init__(self, path, *_): self.path=Path(path); self.path.write_bytes(b'fake-video')
                    def isOpened(self): return True
                    def write(self, image): self.path.write_bytes(b'fake-video-frame')
                    def release(self): closed.append('writer')
                class Renderer:
                    def __init__(self, *_, **kwargs): self.vertices_m=_off_origin_box()
                    def render_object_model(self, _, camera):
                        observed.append(camera['T_world_from_eye'].copy())
                        if fail: raise RuntimeError('intentional fake render failure')
                        return {'color':np.full((480,640,3),.5),'mask':np.ones((480,640),dtype=bool)}
                    def close(self): closed.append('renderer')
                gotrack=types.ModuleType('bench.quality_gotrack')
                gotrack.TexturedRenderer=Renderer; gotrack.upstream_path=lambda:None
                vision=types.ModuleType('bench.vision')
                vision.cv2=types.SimpleNamespace(VideoWriter=Writer,VideoWriter_fourcc=lambda *_:0,
                    cvtColor=lambda image,_:image,COLOR_RGB2BGR=1)
                utils=types.ModuleType('utils')
                utils.renderer_base=types.SimpleNamespace(RenderType=types.SimpleNamespace(COLOR='color',MASK='mask'))
                utils.structs=types.SimpleNamespace(PinholePlaneCameraModel=lambda **kw:kw)
                with patch.dict(sys.modules,{'bench.quality_gotrack':gotrack,'bench.vision':vision,'utils':utils}), \
                        patch('platform.system',return_value='Linux'):
                    kwargs=dict(frames=1,width=640,height=480,fps=10,mesh_units='metres',
                        observationally_ambiguous=True,cloud_execution=True,split_manifest_path=split,
                        view_mode='normalized24-v1',view_index=10)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'intentional'):
                            _render_bundle(mesh,root/'rendered',frozen['groups'][0],**kwargs)
                    else:
                        _render_bundle(mesh,root/'rendered',frozen['groups'][0],**kwargs)
                        label=json.loads((root/'rendered/poses.jsonl').read_text())
                        expected=synthetic_module.validate_normalized_views(_off_origin_box())['poses'][10]
                        np.testing.assert_array_equal(label['pose'],expected)
                        self.assertEqual(label['timestamp_s'],0.)
                        self.assertTrue(label['observationally_ambiguous'])
                        inverse=np.linalg.inv(expected);inverse[:3,3]*=1000
                        np.testing.assert_array_equal(observed[0],inverse)
                        bundle=root/'rendered/bundle'
                        self.assertEqual(sorted(x.name for x in bundle.iterdir()),['input.json','object.glb','synthetic.mp4'])
                        manifest=json.loads((bundle/'input.json').read_text())
                        self.assertFalse(set(manifest)&{'pose','labels','observationally_ambiguous','view_index'})
                        generation=json.loads((root/'rendered/generation.json').read_text())
                        self.assertEqual(generation['view_index'],10)
                        self.assertEqual(generation['view_mode'],'normalized24-v1')
                        self.assertEqual(generation['split_file_sha256'],hashlib.sha256(split.read_bytes()).hexdigest())
                        self.assertEqual(generation['source_sha256'],hashlib.sha256(Path(synthetic_module.__file__).read_bytes()).hexdigest())
                        self.assertTrue(generation['orientation_review_required_before_candidate_generation'])
                        with self.assertRaises(FileExistsError):
                            _render_bundle(mesh,root/'rendered',frozen['groups'][0],**kwargs)
                self.assertEqual(closed.count('writer'),1)
                self.assertEqual(closed.count('renderer'),1)
                self.assertFalse((root/'rendered.claim').exists())
                if fail:
                    self.assertFalse((root/'rendered').exists())
                    incomplete=list(root.glob('.rendered.stage-*/incomplete.json'))
                    self.assertEqual(len(incomplete),1)
                    self.assertFalse(json.loads(incomplete[0].read_text())['promoted'])


def _recipe_fixture(root):
    payloads = {"train": b"fixture metric mesh train", "validation": b"fixture metric mesh validation"}
    mesh_paths = {}
    mesh_digests = {}
    for split, raw in payloads.items():
        path = root / f"{split}.glb"
        path.write_bytes(raw)
        mesh_paths[split] = path
        mesh_digests[split] = hashlib.sha256(raw).hexdigest()
    records, schedules = [], []
    for physical_id, split in (("hot3d:test-train", "train"),
                               ("hot3d:test-validation", "validation")):
        for yaw in (0, 90, 180, 270):
            for pitch in (-25, 25):
                seq = f"{physical_id}-yaw{yaw}-pitch{pitch}"
                scene = f"scene:{seq}"
                schedules.append({"sequence_id": seq, "physical_object_id": physical_id,
                    "scene_id": scene, "split": split,
                    "mesh_sha256": mesh_digests[split], "yaw0_deg": yaw,
                    "pitch_deg": pitch, "observationally_ambiguous": True})
                for ordinal in range(24):
                    records.append({"physical_object_id": physical_id, "object_id": physical_id,
                        "scene_id": scene, "sequence_id": seq,
                        "background_id": "procedural:gray", "background_kind": "procedural",
                        "provenance_kind": "synthetic", "mesh_sha256": mesh_digests[split],
                        "frame_id": ordinal, "source_frame_id": ordinal, "split": split})
    split = freeze_split_manifest({"records": records,
        "label_policy": {"max_rotation_error_deg": 10., "max_translation_error_m": .02}})
    split_path = root / "split.json"
    split_raw = json.dumps(split, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    split_path.write_bytes(split_raw)
    schedules.sort(key=lambda row: row["sequence_id"])
    recipe = seal_recipe({
        "schema_version": 1, "kind": "quality-ranker-synthetic-recipe-v1",
        "recipe_id": "ranker-broader-synthetic-v1",
        "split_binding": {"manifest_sha256": split["manifest_sha256"],
                           "file_sha256": hashlib.sha256(split_raw).hexdigest()},
        "options": {"frames": 24, "width": 640, "height": 480, "fps": 10.0,
            "seed": 0, "mesh_units": "metres", "background_rgb": [127, 127, 127],
            "intrinsics": [[576, 0, 319.5], [0, 576, 239.5], [0, 0, 1]],
            "trajectory": {"rotation_order": "RyRx", "yaw_sweep_deg": 40.0,
                "roll_deg": 0.0, "yaw_origins_deg": [0, 90, 180, 270], "pitch_deg": [-25, 25]},
            "occluder": {"frames_inclusive": [6, 13], "width_fraction": .22,
                "height_fraction": .70, "rgb": [32, 32, 32], "rounding": "half_up",
                "placement_rounding": "floor_plus_half", "center_x_fractions": [.2, .8],
                "center_y_fraction": .5},
            "limits": {"max_output_bytes": 268435456, "max_video_bytes": 67108864},
            "label_policy": split["label_policy"]},
        "sequences": schedules,
    })
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return split, split_path, recipe, recipe_path, mesh_paths


class FrozenRecipeTests(unittest.TestCase):
    def test_canonical_recipe_accepts_every_bound_group_and_rejects_option_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split, split_path, recipe, recipe_path, meshes = _recipe_fixture(root)
            self.assertEqual(seal_recipe({key: value for key, value in recipe.items()
                                          if key != "recipe_sha256"}), recipe)
            for group in split["groups"]:
                selected = "train" if group["split"] == "train" else "validation"
                context = validate_recipe(recipe_path, split_path, meshes[selected],
                    group["sequence_id"], frames=24, width=640, height=480,
                    fps=10., mesh_units="metres", observationally_ambiguous=True)
                self.assertEqual(context["group"], group)
            with self.assertRaisesRegex(ValueError, "generation options"):
                validate_recipe(recipe_path, split_path, meshes["train"],
                    split["groups"][0]["sequence_id"], frames=23, width=640,
                    height=480, fps=10., mesh_units="metres",
                    observationally_ambiguous=True)

    def test_recipe_inventory_cannot_substitute_an_unfrozen_sequence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split, split_path, recipe, recipe_path, meshes = _recipe_fixture(root)
            invalid = json.loads(json.dumps(recipe))
            invalid["sequences"][0]["sequence_id"] = "unfrozen-sequence"
            invalid["sequences"][0]["scene_id"] = "unfrozen-scene"
            recipe_path.write_text(json.dumps(seal_recipe(invalid), sort_keys=True),
                                    encoding="utf-8")
            group = split["groups"][0]
            with self.assertRaisesRegex(ValueError, "cover the 16 frozen split groups"):
                validate_recipe(recipe_path, split_path,
                    meshes["train" if group["split"] == "train" else "validation"],
                    group["sequence_id"], frames=24, width=640, height=480,
                    fps=10., mesh_units="metres", observationally_ambiguous=True)

    def test_recipe_digest_split_bytes_group_and_mesh_tampering_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split, split_path, recipe, recipe_path, meshes = _recipe_fixture(root)
            group = next(row for row in split["groups"] if row["split"] == "train")
            args = dict(frames=24, width=640, height=480, fps=10., mesh_units="metres",
                        observationally_ambiguous=True)
            tampered = dict(recipe)
            tampered["recipe_id"] = "altered"
            recipe_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "self-digest"):
                validate_recipe(recipe_path, split_path, meshes["train"], group["sequence_id"], **args)
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            split_path.write_bytes(split_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "split bytes"):
                validate_recipe(recipe_path, split_path, meshes["train"], group["sequence_id"], **args)
            split_path.write_bytes(json.dumps(split, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8"))
            next(row for row in recipe["sequences"]
                 if row["sequence_id"] == group["sequence_id"])["scene_id"] = "wrong-scene"
            recipe_path.write_text(json.dumps(seal_recipe(recipe), sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split membership"):
                validate_recipe(recipe_path, split_path, meshes["train"], group["sequence_id"], **args)
            split, split_path, recipe, recipe_path, meshes = _recipe_fixture(root)
            group = next(row for row in split["groups"] if row["split"] == "train")
            meshes["train"].write_bytes(b"changed mesh bytes")
            with self.assertRaisesRegex(ValueError, "mesh bytes"):
                validate_recipe(recipe_path, split_path, meshes["train"], group["sequence_id"], **args)


if __name__ == "__main__":
    unittest.main()
