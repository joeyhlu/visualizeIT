"""Manufactured metric mesh selected-PNG exports; no learned runtime or footage."""
import copy
import builtins
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_offline_export as ex
from bench import workbench_geometry as wg
from tools import test_workbench_offline_export_integration as fx


def cylinder_glb(path):
    """Open faceted front cylinder, radius .2m, vertical y, wound toward camera."""
    angles = np.linspace(-1.2, 1.2, 17)
    vertices = np.array([[.2*np.sin(a), y, .2*(1-np.cos(a))]
                         for a in angles for y in (-.15, .15)], dtype="<f4")
    indices = np.array([[2*i, 2*i+1, 2*i+3, 2*i, 2*i+3, 2*i+2]
                        for i in range(16)], dtype="<u2").ravel()
    binary = vertices.tobytes() + indices.tobytes()
    document = dict(asset={"version": "2.0"}, buffers=[dict(byteLength=len(binary))],
        bufferViews=[dict(buffer=0, byteOffset=0, byteLength=vertices.nbytes),
                     dict(buffer=0, byteOffset=vertices.nbytes, byteLength=indices.nbytes)],
        accessors=[dict(bufferView=0, componentType=5126, count=len(vertices), type="VEC3",
                        min=vertices.min(0).tolist(), max=vertices.max(0).tolist()),
                   dict(bufferView=1, componentType=5123, count=len(indices), type="SCALAR")],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0), indices=1)])],
        nodes=[dict(mesh=0)], scenes=[dict(nodes=[0])], scene=0)
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28+len(encoded)+len(binary))
                     + struct.pack("<I4s", len(encoded), b"JSON") + encoded
                     + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


class MeshExportTests(unittest.TestCase):
    def setUp(self):
        (fx.ROOT / ".cache").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=fx.ROOT / ".cache", prefix="mesh-export-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name="plane", role="primary_neural"):
        with patch.object(fx, "plane_glb", cylinder_glb if name == "cylinder" else fx.plane_glb):
            return fx.make_fixture(self.root / name, role=role)

    def test_real_mesh_exports_plane_and_faceted_cylinder(self):
        for name in ("plane", "cylinder"):
            with self.subTest(name=name):
                f = self.fixture(name)
                original = (f["workspace"].runs_dir / f["run_id"] / "neural-artwork.json").read_bytes()
                result = ex.export_saved_window(f["workspace"], **fx.export_args(f), surface_mode="mesh")
                self.assertEqual(result["source"], "offline_saved_pose_mesh_artwork")
                self.assertEqual(result["frame_accounting"]["rendered"], 2)
                self.assertEqual(result["frame_accounting"]["suppressed"], 4)
                self.assertEqual(result["mesh"]["texture_shape"], [32, 32, 4])
                self.assertEqual(result["mesh"]["derived_marker"]["face_id"],
                                 json.loads(original)["marker"]["face_id"])
                self.assertEqual(original, (f["workspace"].runs_dir / f["run_id"] / "neural-artwork.json").read_bytes())
                capture = fx.cv2.VideoCapture(str(f["output_dir"] / "playback.avi"))
                try:
                    for row in result["rows"]:
                        ok, bgr = capture.read()
                        self.assertTrue(ok)
                        rgb = bgr[:, :, ::-1]
                        fid = row["original"]["frame_id"]
                        np.testing.assert_array_equal(rgb[~f["masks"][fid]], f["images"][fid][~f["masks"][fid]])
                        if row["render_reason"] != "rendered_mesh_artwork":
                            np.testing.assert_array_equal(rgb, f["images"][fid])
                        else:
                            self.assertGreater(row["geometry_render_stats"]["composited_pixels"], 0)
                    self.assertFalse(capture.read()[0])
                finally:
                    capture.release()

    def test_preparation_scale_rotation_aspect_and_immutable_anchor(self):
        marker = dict(face_id=0, barycentric=[1, 0, 0], tangent=[1, 0, 0],
                      bitangent=[0, -1, 0], normal=[0, 0, -1], size_object_units=.1)
        saved = copy.deepcopy(marker)
        rgba = np.full((8, 16, 4), 123, np.uint8)
        texture, derived = ex.prepare_mesh_artwork(rgba, marker,
            dict(ex.DEFAULT_PLACEMENT, scale=2, rotation_deg=90))
        self.assertEqual(marker, saved)
        np.testing.assert_array_equal(texture[4:12], rgba)
        self.assertFalse(texture[:4].any())
        self.assertFalse(texture[12:].any())
        np.testing.assert_allclose(derived["tangent"], [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(derived["bitangent"], [-1, 0, 0], atol=1e-12)
        self.assertEqual(derived["size_object_units"], .2)
        self.assertEqual(derived["artwork_sha256"], wg.array_sha256(texture))
        for height, width in ((2, 4), (4, 2)):
            tiny = np.full((height, width, 4), 255, np.uint8)
            texture, derived = ex.prepare_mesh_artwork(tiny, marker)
            self.assertEqual(derived["size_object_units"], .2)
            # Original longest selected edge still spans .1m, despite 8px minimum canvas.
            self.assertAlmostEqual(derived["size_object_units"]*max(height, width)/8, .1)
            self.assertEqual(int(np.count_nonzero(texture[:, :, 3])), height*width)

    def test_composition_halfup_support_and_transparency(self):
        from tools.test_workbench_offline_export import plane_fixture
        args = list(plane_fixture())
        args[1] = np.full((8, 8, 4), [10, 210, 60, 128], np.uint8)
        args.insert(-1, np.array([3]))
        result, reason, stats = ex.compose_mesh(*args, dict(ex.DEFAULT_PLACEMENT, opacity=.5))
        self.assertEqual(reason, "rendered_mesh_artwork")
        changed = np.any(result != args[0], axis=2)
        alpha = 128/255*.5
        expected = np.floor(np.array([10, 210, 60])*alpha + 40*(1-alpha)+.5).astype(np.uint8)
        np.testing.assert_array_equal(result[changed], np.tile(expected, (changed.sum(), 1)))
        self.assertEqual(stats["composited_pixels"], int(changed.sum()))
        for kind in ("support", "transparent", "mask", "opacity"):
            case = copy.deepcopy(args)
            placement = dict(ex.DEFAULT_PLACEMENT)
            if kind == "support":
                case[-2][:] = 0
            elif kind == "transparent":
                case[1][:, :, 3] = 0
            elif kind == "mask":
                case[2][:] = False
            else:
                placement["opacity"] = 0
            rgb, reason, _ = ex.compose_mesh(*case, placement)
            self.assertNotEqual(reason, "rendered_mesh_artwork")
            np.testing.assert_array_equal(rgb, case[0])

    def test_controls_refuse_and_default_planar_unchanged(self):
        f = self.fixture()
        for kwargs in (dict(max_chart_stretch=2), dict(surface_mode="other"),
                       dict(surface_mode="mesh", experimental_tracking=True),
                       dict(surface_mode="mesh", max_chart_stretch=float("nan")),
                       dict(surface_mode="mesh", placement=dict(ex.DEFAULT_PLACEMENT, offset_t_m=.01))):
            with self.subTest(kwargs=kwargs), self.assertRaises((ex.ar.Refused, wg.GeometryError)):
                ex.export_saved_window(f["workspace"], **fx.export_args(f), **kwargs)
            self.assertFalse(f["output_dir"].exists())
        result = ex.export_saved_window(f["workspace"], **fx.export_args(f))
        self.assertEqual(result["source"], "offline_saved_pose_planar_artwork")
        self.assertNotIn("mesh", result)
        self.assertNotIn("geometry_render_stats", result["rows"][0])

    def test_cli_actual_cylinder(self):
        f = self.fixture("cylinder")
        cmd = [sys.executable, "tools/workbench_offline_export.py", "--workspace", str(f["workspace"].path)]
        for key in ("run_id", "role", "project_path", "expected_revision", "bundle_path", "bundle_sha256", "admission_dir", "output_dir", "playback_fps"):
            option = {"project_path": "project", "bundle_path": "bundle", "output_dir": "output"}.get(key, key.replace("_", "-"))
            cmd += ["--"+option, str(f[key])]
        result = subprocess.run(cmd+["--surface-mode", "mesh", "--max-chart-stretch", "2"],
                                cwd=fx.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((f["output_dir"] / "manifest.json").read_text())["surface_mode"], "mesh")

    def test_mutated_mesh_or_selected_png_refuses_publication(self):
        for fault in ("mesh", "png", "source"):
            f = fx.make_fixture(self.root / fault)
            original = ex.compose_mesh
            def mutate(*args, **kwargs):
                value = original(*args, **kwargs)
                path = (f["bundle_path"].parent / "synthetic-plane.glb" if fault == "mesh" else
                        f["project_path"] if fault == "png" else f["bundle_path"].parent / "f2.png")
                path.write_bytes(path.read_bytes() + b"mutated")
                return value
            with patch.object(ex, "compose_mesh", mutate), self.assertRaises(Exception):
                ex.export_saved_window(f["workspace"], **fx.export_args(f), surface_mode="mesh")
            self.assertFalse(f["output_dir"].exists())

    def test_mesh_export_has_no_model_runtime_and_preserves_baseline(self):
        f = self.fixture(role="frozen_baseline")
        importer = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torchvision", "gotrack"):
                raise AssertionError("forbidden runtime import: " + name)
            return importer(name, *args, **kwargs)
        def forbidden(*args, **kwargs):
            raise AssertionError("backend/admission construction forbidden")
        with patch.object(builtins, "__import__", guarded), \
                patch.object(fx.wn, "load_admission", forbidden):
            result = ex.export_saved_window(f["workspace"], **fx.export_args(f), surface_mode="mesh")
        self.assertEqual(result["role"], "frozen_baseline")
        self.assertTrue(all(r["original"]["backend_identity"]["installed_head_sha256"] is None
                            for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
