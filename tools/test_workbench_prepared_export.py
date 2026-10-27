"""Model-free acceptance for opt-in immutable prepared mesh export state."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_geometry as wg
from bench import workbench_offline_export as ex
from tools import test_workbench_mesh_export as mesh_fx
from tools import test_workbench_offline_export as analytical_fx
from tools import test_workbench_offline_export_integration as fx


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


class PreparedMeshExportTests(unittest.TestCase):
    def setUp(self):
        (fx.ROOT / ".cache").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="prepared-mesh-export-", dir=fx.ROOT / ".cache")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def saved_window(self, name):
        with patch.object(fx, "plane_glb", mesh_fx.cylinder_glb if name == "cylinder" else fx.plane_glb):
            return fx.make_fixture(self.root / name)

    def export(self, fixture, name, **kwargs):
        args = fx.export_args(fixture)
        args["output_dir"] = fixture["workspace"].derived_dir / name
        manifest = ex.export_saved_window(fixture["workspace"], **args, surface_mode="mesh", **kwargs)
        return manifest, decoded(args["output_dir"])

    @staticmethod
    def mesh_args():
        rgb, rgba, mask, pose, K, vertices, faces, marker = analytical_fx.plane_fixture()
        support = np.array([3], dtype=np.int64)
        return [rgb, rgba, mask, pose, K, vertices, faces, support, marker]

    def compose_with_prepared(self, args, prepared, *, placement=None, max_chart_stretch=None):
        return ex.compose_mesh(*args, placement, max_chart_stretch=max_chart_stretch,
                               prepared_mesh_artwork=prepared)

    def test_prepared_export_matches_default_rgb_stats_and_ffv1_and_reuses_preparation(self):
        fixture = self.saved_window("plane")
        default, default_rgb = self.export(fixture, "default")

        prepare_calls = 0
        original_prepare = ex.prepare_mesh_artwork_state

        def count_prepare(*args, **kwargs):
            nonlocal prepare_calls
            prepare_calls += 1
            return original_prepare(*args, **kwargs)

        with patch.object(ex, "prepare_mesh_artwork_state", side_effect=count_prepare), \
             patch.object(wg, "marker_patch", wraps=wg.marker_patch) as patch_call, \
             patch.object(wg, "render_marker_rgba", wraps=wg.render_marker_rgba) as render_call:
            prepared, prepared_rgb = self.export(fixture, "prepared", prepare_mesh_once=True)

        self.assertEqual(prepare_calls, 1)
        # NeuralArtwork builds its own artwork-selection patch once. The
        # prepared selected-PNG state adds one patch for the export, reused by
        # all rendered frames.
        self.assertEqual(patch_call.call_count, 2)
        prepared_render_calls = [call for call in render_call.call_args_list
                                 if call.kwargs.get("_prepared_patch") is not None]
        render_attempts = sum(row["geometry_render_stats"] is not None for row in prepared["rows"])
        self.assertEqual(len(prepared_render_calls), render_attempts)
        texture_ids = [id(call.args[4]) for call in prepared_render_calls]
        self.assertEqual(len(set(texture_ids)), 1)
        self.assertEqual(prepared["rows"], default["rows"])
        self.assertEqual(prepared["frame_accounting"], default["frame_accounting"])
        self.assertEqual(len(prepared_rgb), len(default_rgb))
        self.assertEqual(len(prepared_rgb), 6)
        for row, a, b in zip(prepared["rows"], prepared_rgb, default_rgb):
            with self.subTest(frame=row["original"]["frame_id"]):
                np.testing.assert_array_equal(a, b)
                self.assertEqual(row["geometry_render_stats"],
                                 next(r["geometry_render_stats"] for r in default["rows"]
                                      if r["original"]["frame_id"] == row["original"]["frame_id"]))
                fid = row["original"]["frame_id"]
                if row["render_reason"] != "rendered_mesh_artwork":
                    np.testing.assert_array_equal(a, fixture["images"][fid])

    def test_compose_prepared_matches_unprepared_reference_exactly(self):
        args = self.mesh_args()
        prepared = ex.prepare_mesh_artwork_state(args[1], args[5], args[6], args[7], args[8])
        reference = ex.compose_mesh(*args)
        actual = self.compose_with_prepared(args, prepared)
        self.assertEqual(actual[1:], reference[1:])
        np.testing.assert_array_equal(actual[0], reference[0])

    def test_prepared_state_refuses_changed_artwork_geometry_support_anchor_or_policy(self):
        cases = ("artwork", "geometry", "support", "anchor", "scale", "rotation", "chart_policy")
        for change in cases:
            with self.subTest(change=change):
                args = self.mesh_args()
                prepared = ex.prepare_mesh_artwork_state(args[1], args[5], args[6], args[7], args[8])
                changed = list(args)
                placement = None
                limit = None
                if change == "artwork":
                    changed[1] = args[1].copy()
                    changed[1][0, 0, 0] ^= 1
                elif change == "geometry":
                    changed[5] = args[5].copy()
                    changed[5][0, 0] += .001
                elif change == "support":
                    changed[7] = args[7].copy()
                    changed[7][0] = 2
                elif change == "anchor":
                    changed[8] = dict(args[8], size_object_units=.11)
                elif change == "scale":
                    placement = dict(ex.DEFAULT_PLACEMENT, scale=1.1)
                elif change == "rotation":
                    placement = dict(ex.DEFAULT_PLACEMENT, rotation_deg=5)
                else:
                    limit = 2.0
                with self.assertRaises((ex.ar.Refused, wg.GeometryError)):
                    self.compose_with_prepared(changed, prepared, placement=placement,
                                               max_chart_stretch=limit)

    def test_prepared_state_is_independent_of_later_caller_array_mutation(self):
        args = self.mesh_args()
        pristine = list(args)
        pristine[1] = args[1].copy()
        pristine[5] = args[5].copy()
        pristine[6] = args[6].copy()
        pristine[7] = args[7].copy()
        pristine[8] = dict(args[8])
        prepared = ex.prepare_mesh_artwork_state(args[1], args[5], args[6], args[7], args[8])
        expected = ex.compose_mesh(*pristine)
        args[1][:] = 0
        args[5][:] = 99
        args[6][:] = 0
        args[7][:] = 0
        args[8]["size_object_units"] = .2
        actual = self.compose_with_prepared(pristine, prepared)
        self.assertEqual(actual[1:], expected[1:])
        np.testing.assert_array_equal(actual[0], expected[0])

    def test_interrupted_prepared_export_cleans_staging_and_keeps_original_rgb_for_loss(self):
        fixture = self.saved_window("interrupted")
        args = fx.export_args(fixture)
        args["output_dir"] = fixture["workspace"].derived_dir / "interrupted-output"
        original = ex.compose_mesh

        def interrupt_after_first_frame(*call_args, **kwargs):
            raise RuntimeError("manufactured export interruption")

        with patch.object(ex, "compose_mesh", side_effect=interrupt_after_first_frame):
            with self.assertRaisesRegex(RuntimeError, "manufactured export interruption"):
                ex.export_saved_window(fixture["workspace"], **args, surface_mode="mesh",
                                       prepare_mesh_once=True)
        self.assertFalse(args["output_dir"].exists())
        self.assertEqual(list(fixture["workspace"].derived_dir.glob(".offline-export-*")), [])

        # The ordinary prepared path also keeps source RGB for suppressed/lost rows.
        manifest, frames = self.export(fixture, "loss-path", prepare_mesh_once=True)
        for row, rgb in zip(manifest["rows"], frames):
            if row["render_reason"] != "rendered_mesh_artwork":
                fid = row["original"]["frame_id"]
                np.testing.assert_array_equal(rgb, fixture["images"][fid])

    def test_all_lost_prepared_export_matches_default_and_preserves_every_source_frame(self):
        fixture = self.saved_window("all-lost")
        results_path = fixture["workspace"].runs_dir / fixture["run_id"] / "results.jsonl"
        rows = [fx.wc.loads_strict(line) for line in results_path.read_bytes().splitlines()]
        for row in rows:
            row["pose_state"] = "lost"
            row["render_state"] = "suppressed"
            row["failure_reason"] = "synthetic unavailable pose"
            row["pose"] = None
            row["pose_units"] = None
        results_path.write_bytes(b"\n".join(fx.wc.canonical_json(row) for row in rows) + b"\n")

        default, default_rgb = self.export(fixture, "all-lost-default")
        prepared, prepared_rgb = self.export(fixture, "all-lost-prepared", prepare_mesh_once=True)
        self.assertEqual(prepared["frame_accounting"]["rendered"], 0)
        self.assertEqual(prepared["frame_accounting"], default["frame_accounting"])
        self.assertEqual(len(prepared_rgb), len(default_rgb))
        for row, expected, actual in zip(prepared["rows"], default_rgb, prepared_rgb):
            fid = row["original"]["frame_id"]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, fixture["images"][fid])


if __name__ == "__main__":
    unittest.main()
