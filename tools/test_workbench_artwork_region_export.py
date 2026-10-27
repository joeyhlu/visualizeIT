"""Synthetic, model-free acceptance of opt-in artwork-region mesh export."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_offline_export as ex
from tools import test_workbench_cylinder_oracle as oracle
from tools import test_workbench_mesh_export as mesh_fx
from tools import test_workbench_offline_export_integration as fx


BASE = "1f072492ecc03f8657ca2949fb3e2dcdbfcd2454"


def decoded(path):
    capture = fx.cv2.VideoCapture(str(path / "playback.avi"))
    frames = []
    try:
        if not capture.isOpened():
            raise AssertionError("Cannot open manufactured FFV1 export")
        while True:
            ok, bgr = capture.read()
            if not ok:
                return frames
            frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    finally:
        capture.release()


def baseline_exporter():
    """Run frozen parent implementation, retaining current dependency closure."""
    result = subprocess.run(["git", "show", BASE + ":bench/workbench_offline_export.py"],
                            cwd=fx.ROOT, capture_output=True, text=True, check=True)
    module = types.ModuleType("bench._artwork_region_parent_export")
    module.__package__ = "bench"
    module.__file__ = str(fx.ROOT / "bench/workbench_offline_export.py")
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    return module


class ArtworkRegionExportTests(unittest.TestCase):
    def setUp(self):
        (fx.ROOT / ".cache").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="artwork-region-export-", dir=fx.ROOT / ".cache")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name="plane", *, cylinder=False):
        generator = mesh_fx.cylinder_glb if cylinder else fx.plane_glb
        with patch.object(fx, "plane_glb", generator):
            return fx.make_fixture(self.root / name)

    def assert_no_output(self, f):
        self.assertFalse(f["output_dir"].exists())
        self.assertEqual(list(f["workspace"].derived_dir.glob(".offline-export-*")), [])

    def export(self, f, *, output_name, **kwargs):
        args = fx.export_args(f)
        args["output_dir"] = f["workspace"].derived_dir / output_name
        result = ex.export_saved_window(f["workspace"], **args, **kwargs)
        return result, decoded(args["output_dir"])

    def cli_command(self, f):
        command = [sys.executable, "tools/workbench_offline_export.py", "--workspace", str(f["workspace"].path)]
        for key, value in fx.export_args(f).items():
            option = {"project_path": "project", "bundle_path": "bundle", "output_dir": "output"}.get(key, key.replace("_", "-"))
            command.extend(["--" + option, str(value)])
        return command

    def test_strict_api_flag_and_planar_opt_in_refuse_before_staging(self):
        f = self.fixture()
        for flag in (None, 0, 1, "false", [], np.bool_(True)):
            with self.subTest(flag=repr(flag)), patch.object(f["workspace"], "require_budget") as budget:
                with self.assertRaisesRegex((ex.ar.Refused, ex.wg.GeometryError), "artwork_region_raster.*bool"):
                    ex.export_saved_window(f["workspace"], **fx.export_args(f), surface_mode="mesh", artwork_region_raster=flag)
                budget.assert_not_called()
            self.assert_no_output(f)
        for kwargs in ({"artwork_region_raster": True},
                       {"surface_mode": "mesh", "experimental_tracking": True, "artwork_region_raster": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ex.ar.Refused):
                ex.export_saved_window(f["workspace"], **fx.export_args(f), **kwargs)
            self.assert_no_output(f)
        vertices, faces, support, marker, art = oracle.fixture()
        args = (np.full((oracle.HEIGHT, oracle.WIDTH, 3), 40, np.uint8), art,
                np.ones((oracle.HEIGHT, oracle.WIDTH), bool), oracle.POSES[0], oracle.K,
                vertices, faces, support, marker)
        for flag in (None, 0, 1, "true", np.bool_(False)):
            with self.subTest(compose_flag=repr(flag)), self.assertRaisesRegex(
                    (ex.ar.Refused, ex.wg.GeometryError), "artwork_region_raster.*bool"):
                ex.compose_mesh(*args, artwork_region_raster=flag)

    def test_cli_planar_opt_in_refuses_and_mesh_opt_in_is_recorded(self):
        f = self.fixture()
        command = self.cli_command(f)
        invalid = subprocess.run(command + ["--artwork-region-raster"], cwd=fx.ROOT, capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2, invalid.stderr)
        self.assertIn("mesh", invalid.stderr.lower())
        self.assert_no_output(f)
        valid = subprocess.run(command + ["--surface-mode", "mesh", "--artwork-region-raster"],
                               cwd=fx.ROOT, capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        manifest = wc.loads_strict((f["output_dir"] / "manifest.json").read_bytes())
        self.assertIs(manifest["mesh"]["artwork_region_raster"], True)

    def test_default_off_exact_parent_manifest_and_decoded_compatibility(self):
        parent = baseline_exporter()
        for mode in ("planar", "mesh"):
            f = self.fixture(mode)
            args = fx.export_args(f)
            args["output_dir"] = f["workspace"].derived_dir / "parent"
            expected = parent.export_saved_window(f["workspace"], **args, surface_mode=mode)
            expected_rgb = decoded(args["output_dir"])
            for flag in ({}, {"artwork_region_raster": False}):
                with self.subTest(mode=mode, explicit=bool(flag)):
                    actual, actual_rgb = self.export(f, output_name="explicit" if flag else "default",
                                                     surface_mode=mode, **flag)
                    # Both implementations authenticate the same current on-disk
                    # sources. No manifest fields or digests are normalized away.
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(actual_rgb), len(expected_rgb))
                    for a, b in zip(actual_rgb, expected_rgb):
                        np.testing.assert_array_equal(a, b)

    def test_enabled_saved_window_exact_rgb_stats_and_identity(self):
        for cylinder in (False, True):
            f = self.fixture("cylinder" if cylinder else "plane", cylinder=cylinder)
            off, off_rgb = self.export(f, output_name="off", surface_mode="mesh")
            on, on_rgb = self.export(f, output_name="on", surface_mode="mesh", artwork_region_raster=True)
            self.assertEqual(on["rows"], off["rows"])
            self.assertEqual(on["frame_accounting"], off["frame_accounting"])
            self.assertEqual(on["frame_accounting"]["rendered"], 2)
            self.assertEqual(len(on_rgb), 6)
            for row, a, b in zip(on["rows"], on_rgb, off_rgb):
                np.testing.assert_array_equal(a, b)
                fid = row["original"]["frame_id"]
                np.testing.assert_array_equal(a[~f["masks"][fid]], f["images"][fid][~f["masks"][fid]])
                if row["render_reason"] != "rendered_mesh_artwork":
                    np.testing.assert_array_equal(a, f["images"][fid])
            self.assertIs(on["mesh"]["artwork_region_raster"], True)
            self.assertNotIn("artwork_region_raster", off["mesh"])
            self.assertEqual(on["provenance"]["mesh_composition_sha256"], wc.canonical_sha256(on["mesh"]))
            self.assertNotEqual(on["provenance"]["mesh_composition_sha256"], off["provenance"]["mesh_composition_sha256"])
            expected_identity = copy.deepcopy(off["mesh"])
            expected_identity["artwork_region_raster"] = True
            self.assertEqual(on["mesh"], expected_identity)
            sources = on["provenance"]["exporter_sources"]
            for name in ("bench/workbench_geometry.py", "bench/workbench_offline_export.py", "tools/workbench_offline_export.py"):
                self.assertEqual(sources[name], fx.sha(fx.ROOT / name))

    def test_composition_matches_independent_cylinder_oracle_with_occluders_and_masks(self):
        rgb = np.full((oracle.HEIGHT, oracle.WIDTH, 3), [39, 61, 83], np.uint8)
        for occluder, missing, masked in ((False, None, False), (True, None, False),
                                           (True, None, True), (False, 1, True)):
            vertices, faces, support, marker, art = oracle.fixture(missing, occluder)
            mask = np.ones(rgb.shape[:2], bool)
            if masked:
                mask[23:61, 31:53] = False
            pose = oracle.POSES[2]
            expected_rgba, boundary, _ = oracle.oracle(pose, mask, missing=missing, occluder=occluder)
            alpha = expected_rgba[..., 3:4].astype(np.float64) / 255
            expected = np.floor(expected_rgba[..., :3] * alpha + rgb * (1-alpha) + .5).astype(np.uint8)
            args = (rgb, art, mask, pose, oracle.K, vertices, faces, support, marker)
            off, off_reason, off_stats = ex.compose_mesh(*args)
            on, on_reason, on_stats = ex.compose_mesh(*args, artwork_region_raster=True)
            with self.subTest(occluder=occluder, missing=missing, masked=masked):
                self.assertEqual(on_reason, "rendered_mesh_artwork")
                self.assertEqual(on_reason, off_reason)
                self.assertEqual(on_stats, off_stats)
                np.testing.assert_array_equal(on, off)
                np.testing.assert_array_equal(on[~boundary], expected[~boundary])
                self.assertGreater(np.count_nonzero(np.any(on != rgb, axis=2)), 100)
                np.testing.assert_array_equal(on[~mask], rgb[~mask])

    def test_enabled_source_or_identity_mutation_refuses_publication(self):
        for fault in ("source_closure", "marker_identity"):
            f = self.fixture(fault)
            original = ex.compose_mesh
            changed = False
            def mutate(*args, **kwargs):
                nonlocal changed
                result = original(*args, **kwargs)
                if not changed:
                    changed = True
                    if fault == "marker_identity":
                        args[8]["size_object_units"] *= 1.1
                return result
            source_hashes = ex.ar.ArtworkRamReplay._source_hashes
            def changed_hashes(service, rels):
                values = source_hashes(service, rels)
                if fault == "source_closure" and changed:
                    values["bench/workbench_offline_export.py"] = "0" * 64
                return values
            with patch.object(ex, "compose_mesh", mutate), patch.object(
                    ex.ar.ArtworkRamReplay, "_source_hashes", changed_hashes), self.assertRaises(ex.ar.Refused):
                ex.export_saved_window(f["workspace"], **fx.export_args(f), surface_mode="mesh", artwork_region_raster=True)
            self.assertTrue(changed)
            self.assert_no_output(f)


if __name__ == "__main__":
    unittest.main()
