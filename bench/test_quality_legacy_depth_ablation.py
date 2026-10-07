"""Independent CPU guards for the opt-in legacy bottle depth wrapper.

These tests use inert cameras/renderers and temporary metadata fixtures. They
do not exercise Torch, OpenGL, pyrender, a model, a video, or a GPU. The scratch
copy targets the candidate source; when moved beside the reviewed production
module, it targets that file instead.
"""
from __future__ import annotations

import copy
import builtins
from enum import Enum
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if Path(__file__).resolve().parent.name == "bench":
    _SOURCE = Path(__file__).resolve().with_name("quality_legacy_depth_ablation.py")
else:
    _SOURCE = _REPO_ROOT / ".cache" / "quality_legacy_depth_ablation_candidate.py"
_MODULE_NAME = "bench._legacy_depth_ablation_independent_test_target"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load depth-ablation source at {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _RenderType(Enum):
    COLOR = 1
    DEPTH = 2
    MASK = 3


class _Camera:
    def __init__(self, *, width, height, f, c, T_world_from_eye):
        self.width = width
        self.height = height
        self.f = tuple(f)
        self.c = tuple(c)
        self.T_world_from_eye = np.array(T_world_from_eye, dtype=np.float64, copy=True)


def _utils_module():
    module = types.ModuleType("utils")
    module.structs = types.SimpleNamespace(PinholePlaneCameraModel=_Camera)
    module.renderer_base = types.SimpleNamespace(RenderType=_RenderType)
    return module


def _quality_gotrack_module(renderer_factory):
    module = types.ModuleType("bench.quality_gotrack")
    module.TexturedRenderer = renderer_factory
    return module


def _camera(width=280, height=280):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    pose[:3, 3] = (37.0, -21.5, 840.25)
    return _Camera(width=width, height=height, f=(181.25, 227.5),
                   c=(47.25, 191.75), T_world_from_eye=pose)


def _payloads(*, bad=None):
    color = np.full((280, 280, 3), 0.375, dtype=np.float32)
    legacy_depth = np.zeros((280, 280), dtype=np.float32)
    zero_depth = np.zeros((280, 280), dtype=np.float32)
    legacy_depth[4, 8] = 1200.0
    legacy_depth[5, 9] = 1700.0
    legacy_depth[7, 11] = 999.0
    legacy_depth[1, 1] = 1300.0       # Legacy-only support is removed in both branches.
    zero_depth[4, 8] = 1215.0
    zero_depth[5, 9] = 1690.0
    zero_depth[7, 11] = 1001.0
    zero_depth[2, 2] = 2300.0        # Zero-sample-only support is also removed.
    if bad == "shape":
        zero_depth = zero_depth[:, :-1]
    elif bad == "nonfinite":
        zero_depth[4, 8] = np.nan
    elif bad == "negative":
        zero_depth[4, 8] = -1.0
    return color, legacy_depth, zero_depth


def _fake_renderer_type(*, bad=None, metadata=None, events=None,
                        close_errors=None, fail_constructor_at=None):
    events = [] if events is None else events
    close_errors = {} if close_errors is None else close_errors
    color, legacy_depth, zero_depth = _payloads(bad=bad)
    instances = []

    def factory(_path, obj_id, **kwargs):
        ordinal = len(instances) + 1
        if fail_constructor_at == ordinal:
            raise RuntimeError("secondary renderer constructor failed")
        role = kwargs["render_policy"]

        class FakeRenderer:
            def __init__(self):
                self.obj_id = obj_id
                self.asset_sha256 = "A" * 64
                self.vertices_m = np.asarray(((0., 0., 0.), (0.1, 0., 0.),
                                              (0., 0.1, 0.)), dtype=np.float64)
                self.role = role
                self.kwargs = dict(kwargs)
                self.calls = []
                self.closed = False

            @property
            def render_policy_metadata(self):
                return metadata

            def render_object_model(self, requested_object, camera, *,
                                    return_tensors=False, background=None):
                self.calls.append((requested_object, camera, return_tensors, background))
                depth = legacy_depth if self.role == "legacy_v1" else zero_depth
                return {
                    _RenderType.COLOR: color.copy(),
                    _RenderType.DEPTH: depth.copy(),
                    _RenderType.MASK: (depth > 0).copy(),
                }

            def close(self):
                self.closed = True
                events.append(self.role)
                if self.role in close_errors:
                    raise close_errors[self.role]

        instance = FakeRenderer()
        instances.append(instance)
        return instance

    factory.instances = instances
    factory.events = events
    return factory


class CameraGeometryTests(unittest.TestCase):
    def test_inner_camera_preserves_legacy_half_center_rays_and_copies_inputs(self):
        outer = _camera()
        original_f = outer.f
        original_c = outer.c
        original_pose = outer.T_world_from_eye.copy()
        factory = _utils_module()
        with patch.dict(sys.modules, {"utils": factory}):
            inner = _MODULE.legacy_inner_camera(outer)

        self.assertIsNot(inner, outer)
        self.assertEqual(inner.width, outer.width)
        self.assertEqual(inner.height, outer.height)
        self.assertEqual(inner.f, original_f)
        self.assertEqual(inner.c, (original_c[0] - 0.5, original_c[1] - 0.5))
        self.assertIsNot(inner.T_world_from_eye, outer.T_world_from_eye)
        np.testing.assert_array_equal(inner.T_world_from_eye, original_pose)

        pixels = np.asarray(((0., 0.), (19., 271.), (139., 143.),
                             (279., 3.), (226., 199.)), dtype=np.float64)
        outer_rays = (pixels + 0.5 - np.asarray(original_c)) / np.asarray(original_f)
        inner_rays = (pixels - np.asarray(inner.c)) / np.asarray(inner.f)
        np.testing.assert_array_equal(inner_rays, outer_rays)
        self.assertEqual(outer.f, original_f)
        self.assertEqual(outer.c, original_c)
        np.testing.assert_array_equal(outer.T_world_from_eye, original_pose)

    def test_camera_validator_rejects_non_280_and_malformed_geometry(self):
        with self.assertRaises(ValueError):
            _MODULE.LegacyDepthAblationRenderer._validate_camera(_camera(width=279))
        bad_focal = _camera()
        bad_focal.f = (181.25, float("nan"))
        with self.assertRaises(ValueError):
            _MODULE.LegacyDepthAblationRenderer._validate_camera(bad_focal)
        improper = _camera()
        improper.T_world_from_eye[3, 0] = 0.1
        with self.assertRaises(ValueError):
            _MODULE.LegacyDepthAblationRenderer._validate_camera(improper)


class DualRendererTests(unittest.TestCase):
    def _render_branch(self, branch, *, bad=None, metadata=None):
        factory = _fake_renderer_type(bad=bad, metadata=metadata)
        fake_gotrack = _quality_gotrack_module(factory)
        validator = lambda value, *, expected_dimensions, previous: value
        with patch.dict(sys.modules, {"bench.quality_gotrack": fake_gotrack}), \
             patch.object(_MODULE, "_validate_zero_metadata", validator), \
             patch.dict(sys.modules, {"utils": _utils_module()}):
            renderer = _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, branch)
            camera = _camera()
            before_f, before_c = camera.f, camera.c
            before_pose = camera.T_world_from_eye.copy()
            result = renderer.render_object_model(15, camera)
            return renderer, result, factory.instances, camera, (before_f, before_c, before_pose)

    def test_both_branches_share_legacy_rgb_and_common_template_mask(self):
        control, control_result, control_instances, control_camera, control_before = self._render_branch("control")
        candidate, candidate_result, candidate_instances, candidate_camera, candidate_before = self._render_branch("candidate")

        for result in (control_result, candidate_result):
            self.assertEqual(set(result), {_RenderType.COLOR, _RenderType.DEPTH, _RenderType.MASK})
            self.assertEqual(result[_RenderType.COLOR].shape, (280, 280, 3))
            self.assertEqual(result[_RenderType.DEPTH].shape, (280, 280))
            self.assertEqual(result[_RenderType.MASK].shape, (280, 280))
            self.assertEqual(result[_RenderType.DEPTH].dtype, np.float32)
            self.assertEqual(result[_RenderType.MASK].dtype, np.bool_)

        np.testing.assert_array_equal(control_result[_RenderType.COLOR], candidate_result[_RenderType.COLOR])
        np.testing.assert_array_equal(control_result[_RenderType.MASK], candidate_result[_RenderType.MASK])
        expected_common = np.zeros((280, 280), dtype=np.bool_)
        expected_common[4, 8] = expected_common[5, 9] = expected_common[7, 11] = True
        np.testing.assert_array_equal(control_result[_RenderType.MASK], expected_common)

        self.assertEqual(control_result[_RenderType.DEPTH][4, 8], 1200.0)
        self.assertEqual(control_result[_RenderType.DEPTH][5, 9], 1700.0)
        self.assertEqual(candidate_result[_RenderType.DEPTH][4, 8], 1215.0)
        self.assertEqual(candidate_result[_RenderType.DEPTH][5, 9], 1690.0)
        self.assertEqual(control_result[_RenderType.DEPTH][1, 1], 0.0)
        self.assertEqual(candidate_result[_RenderType.DEPTH][2, 2], 0.0)
        np.testing.assert_array_equal(control_result[_RenderType.MASK], control_result[_RenderType.DEPTH] > 0.)
        np.testing.assert_array_equal(candidate_result[_RenderType.MASK], candidate_result[_RenderType.DEPTH] > 0.)

        for wrapper, instances, camera, before in (
                (control, control_instances, control_camera, control_before),
                (candidate, candidate_instances, candidate_camera, candidate_before)):
            self.assertEqual(len(instances), 2)
            self.assertEqual(instances[0].kwargs["render_policy"], "legacy_v1")
            self.assertEqual(instances[0].kwargs["coordinate_mode"], "legacy")
            self.assertIs(instances[0].kwargs["unlit"], True)
            self.assertIs(instances[0].kwargs["disable_multisampling"], True)
            self.assertEqual(instances[1].kwargs["render_policy"], "capture_zero_sample_v1")
            self.assertEqual(instances[1].kwargs["coordinate_mode"], "integer_centers_v1")
            self.assertIs(instances[1].kwargs["unlit"], True)
            self.assertIs(instances[1].kwargs["disable_multisampling"], False)
            self.assertIs(instances[0].calls[0][1], camera)
            inner = instances[1].calls[0][1]
            self.assertIsNot(inner, camera)
            self.assertEqual(inner.c, (before[1][0] - 0.5, before[1][1] - 0.5))
            self.assertEqual(inner.f, before[0])
            np.testing.assert_array_equal(inner.T_world_from_eye, before[2])
            self.assertEqual(camera.f, before[0])
            self.assertEqual(camera.c, before[1])
            np.testing.assert_array_equal(camera.T_world_from_eye, before[2])
            record = wrapper.render_records[0]
            self.assertEqual(record["legacy_template_pixels"], 4)
            self.assertEqual(record["zero_sample_template_pixels"], 4)
            self.assertEqual(record["common_template_pixels"], 3)
            self.assertEqual(record["removed_template_pixels"], 1)

    def test_unknown_object_and_bad_depth_or_layout_fail_closed(self):
        factory = _fake_renderer_type()
        fake_gotrack = _quality_gotrack_module(factory)
        validator = lambda value, *, expected_dimensions, previous: value
        with patch.dict(sys.modules, {"bench.quality_gotrack": fake_gotrack}), \
             patch.object(_MODULE, "_validate_zero_metadata", validator), \
             patch.dict(sys.modules, {"utils": _utils_module()}):
            wrapper = _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, "candidate")
            with self.assertRaises(ValueError):
                wrapper.render_object_model(16, _camera())
            with self.assertRaises(TypeError):
                wrapper.render_object_model(
                    15, _camera(), observed_mask=np.ones((280, 280), dtype=np.uint8))

        for bad in ("shape", "nonfinite", "negative"):
            with self.subTest(bad=bad):
                factory = _fake_renderer_type(bad=bad)
                with patch.dict(sys.modules, {"bench.quality_gotrack": _quality_gotrack_module(factory)}), \
                     patch.object(_MODULE, "_validate_zero_metadata", validator), \
                     patch.dict(sys.modules, {"utils": _utils_module()}):
                    wrapper = _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, "candidate")
                    with self.assertRaises(RuntimeError):
                        wrapper.render_object_model(15, _camera())

    def test_missing_or_stale_public_metadata_aborts_before_returning_depth(self):
        factory = _fake_renderer_type(metadata=None)
        with patch.dict(sys.modules, {"bench.quality_gotrack": _quality_gotrack_module(factory)}), \
             patch.dict(sys.modules, {"utils": _utils_module()}):
            wrapper = _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, "candidate")
            with self.assertRaises(RuntimeError):
                wrapper.render_object_model(15, _camera())


class RendererOwnershipTests(unittest.TestCase):
    def test_second_constructor_failure_closes_first_and_preserves_primary_exception(self):
        events = []
        primary = RuntimeError("second constructor primary")

        class FirstRenderer:
            def __init__(self, _path, obj_id, **kwargs):
                self.obj_id = obj_id
                self.asset_sha256 = "A" * 64
                self.vertices_m = np.asarray(((0., 0., 0.),), dtype=np.float64)
                self.kwargs = kwargs

            def close(self):
                events.append("closed-first")

        calls = 0

        def factory(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return FirstRenderer(*args, **kwargs)
            raise primary

        with patch.dict(sys.modules, {"bench.quality_gotrack": _quality_gotrack_module(factory)}):
            with self.assertRaises(RuntimeError) as caught:
                _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, "control")
        self.assertIsInstance(caught.exception, _MODULE.RendererIntegrityError)
        self.assertIs(caught.exception.__cause__, primary)
        self.assertIn("second constructor primary", str(caught.exception))
        self.assertEqual(events, ["closed-first"])

    def test_constructor_primary_survives_cleanup_failure(self):
        primary = RuntimeError("second constructor primary")
        cleanup = RuntimeError("first renderer cleanup failed")
        events = []

        class FirstRenderer:
            def __init__(self, _path, obj_id, **kwargs):
                self.obj_id = obj_id
                self.asset_sha256 = "A" * 64
                self.vertices_m = np.asarray(((0., 0., 0.),), dtype=np.float64)
                self.kwargs = kwargs

            def close(self):
                events.append("closed-first")
                raise cleanup

        calls = 0

        def factory(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return FirstRenderer(*args, **kwargs)
            raise primary

        with patch.dict(sys.modules, {"bench.quality_gotrack": _quality_gotrack_module(factory)}):
            with self.assertRaises(RuntimeError) as caught:
                _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, "control")
        self.assertIs(caught.exception.__cause__, primary)
        self.assertIn("second constructor primary", str(caught.exception))
        self.assertIn("first renderer cleanup failed", str(caught.exception))
        self.assertEqual(events, ["closed-first"])

    def test_close_attempts_both_renderers_and_reports_both_failures(self):
        events = []
        failures = {"legacy_v1": RuntimeError("legacy close failed"),
                    "capture_zero_sample_v1": RuntimeError("zero close failed")}
        factory = _fake_renderer_type(events=events, close_errors=failures)
        with patch.dict(sys.modules, {"bench.quality_gotrack": _quality_gotrack_module(factory)}):
            wrapper = _MODULE.LegacyDepthAblationRenderer("unused.glb", 15, "control")
            with self.assertRaises(RuntimeError) as caught:
                wrapper.close()
        self.assertEqual(events, ["legacy_v1", "capture_zero_sample_v1"])
        self.assertIn("legacy close failed", str(caught.exception))
        self.assertIn("zero close failed", str(caught.exception))


class GeometryFixtureTests(unittest.TestCase):
    def test_geometry_bootstraps_pinned_upstream_path_before_utils_import(self):
        source = _SOURCE.read_text(encoding="utf-8")
        self.assertIn("def geometry(output):", source)
        body = source.split("def geometry(output):", 1)[1].split("\ndef build_parser", 1)[0]
        bootstrap = body.find("upstream_path()")
        renderer_api = body.find("from utils import structs, renderer_base")
        self.assertGreaterEqual(bootstrap, 0)
        self.assertGreaterEqual(renderer_api, 0)
        self.assertLess(bootstrap, renderer_api)

    def test_slanted_plane_winding_faces_camera(self):
        captured = {}

        class FakeTrimesh:
            def __init__(self, *, vertices, faces, process):
                captured["vertices"] = np.asarray(vertices, dtype=np.float64)
                captured["faces"] = np.asarray(faces, dtype=np.int64)
                captured["process"] = process

        trimesh_module = types.ModuleType("trimesh")
        trimesh_module.Trimesh = FakeTrimesh

        class Scene:
            def __init__(self):
                self.mesh_nodes = ["old-mesh"]
                self.removed = []
                self.added = []

            def remove_node(self, node):
                self.removed.append(node)
                self.mesh_nodes.remove(node)

            def add(self, mesh):
                self.added.append(mesh)
                node = "plane-mesh-node"
                self.mesh_nodes.append(node)
                return node

        scene = Scene()
        renderer = types.SimpleNamespace(
            scene=scene,
            pyrender=types.SimpleNamespace(
                Mesh=types.SimpleNamespace(
                    from_trimesh=lambda mesh, smooth=False: (mesh, smooth))))
        with patch.dict(sys.modules, {"trimesh": trimesh_module}):
            node = _MODULE._plane_fixture_scene(renderer)

        self.assertEqual(node, "plane-mesh-node")
        self.assertEqual(captured["process"], False)
        np.testing.assert_array_equal(captured["faces"], ((0, 2, 1), (0, 3, 2)))
        self.assertEqual(scene.removed, ["old-mesh"])
        self.assertEqual(scene.added[0][1], False)
        vertices = captured["vertices"]
        first_face = captured["faces"][0]
        a, b, c = vertices[first_face]
        normal = np.cross(b - a, c - a)
        toward_camera = -np.mean(vertices[first_face], axis=0)
        self.assertGreater(float(np.dot(normal, toward_camera)), 0.0)


class GeometryFailureTests(unittest.TestCase):
    def test_constructor_failure_persists_failed_report_and_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "geometry.json"
            marker = root / "geometry.json.running"
            primary = RuntimeError("synthetic renderer constructor failure")
            events = []

            gotrack = types.ModuleType("bench.quality_gotrack")

            def upstream_path():
                events.append("upstream")
                return ()

            gotrack.upstream_path = upstream_path
            runner = types.ModuleType("bench.quality_runner")
            runner.read_input = lambda _bundle: {
                "_capture": None,
                "_timing": {"clock_mode": "legacy_nominal_60_v1"},
                "object": "ranch",
                "object_id": 15,
                "asset": "ranch.glb",
            }

            with patch.dict(sys.modules, {
                    "bench.quality_gotrack": gotrack,
                    "bench.quality_runner": runner,
                    "utils": _utils_module(),
            }), patch.object(_MODULE, "_OUTPUT_ROOT", root), \
                    patch.object(_MODULE, "_REPORT_BYTE_LIMIT", 1_000_000), \
                    patch.object(_MODULE, "_claim_output", return_value=(output, marker)), \
                    patch.object(_MODULE, "LegacyDepthAblationRenderer", side_effect=primary):
                with self.assertRaises(RuntimeError) as caught:
                    _MODULE.geometry(output)

            self.assertIs(caught.exception, primary)
            report = json.loads(output.read_text(encoding="utf-8"))
            final_marker = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["complete"])
            self.assertIn("RuntimeError: synthetic renderer constructor failure",
                          report["failure_reason"])
            self.assertIsNone(report["outer_camera"])
            self.assertEqual(report["render_records"], [])
            self.assertFalse(report["is_tracking_accuracy_evidence"])
            self.assertEqual(final_marker["status"], "failed")
            self.assertEqual(final_marker["output_sha256"],
                             hashlib.sha256(output.read_bytes()).hexdigest().upper())
            self.assertEqual(events, ["upstream"])


class BoundedReportTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name).resolve()
        self.output = self.root / "candidate-120.json"
        self.context = {
            "experiment": "legacy_depth_ablation_v1",
            "branch": "candidate",
            "frame_ids": [10, 11, 12],
            "unattempted_frame_ids": [12],
            "phase": "tracking",
        }

    def _persist(self, report, *, reserve=32):
        return _MODULE._persist_report(
            self.output, report, budget_context=self.context,
            reserve_failure_bytes=reserve)

    def test_exact_persisted_bytes_and_boundary_budget(self):
        report = {"schema_version": 1, "complete": False,
                  "frames": [{"frameId": 10, "pose_state": "lost"}],
                  "message": "UTF-8 π"}
        expected = json.dumps(report, separators=(",", ":"),
                              allow_nan=False).encode("utf-8")
        self.assertEqual(_MODULE._serialize_report_bytes(report), expected)
        with patch.object(_MODULE, "_OUTPUT_ROOT", self.root), \
             patch.object(_MODULE, "_REPORT_BYTE_LIMIT", len(expected) + 32):
            written = self._persist(report, reserve=32)
            self.assertEqual(written, len(expected))
            self.assertEqual(self.output.read_bytes(), expected)
            self.assertEqual(_MODULE._report_storage_bytes(self.root), len(expected))
            self.assertFalse(self.output.with_suffix(self.output.suffix + ".tmp").exists())

    def test_budget_failure_keeps_last_report_and_writes_bounded_failure_summary(self):
        prior = {"schema_version": 1, "complete": False,
                 "frames": [{"frameId": 10, "pose_state": "lost"}]}
        next_report = {"schema_version": 1, "complete": True,
                       "frames": [{"frameId": frame_id, "pose_state": "tracking",
                                   "detail": "x" * 180} for frame_id in range(10, 120)]}
        reserve = 32
        with patch.object(_MODULE, "_OUTPUT_ROOT", self.root):
            with patch.object(_MODULE, "_REPORT_BYTE_LIMIT", 1_000_000):
                self._persist(prior, reserve=reserve)
            prior_bytes = self.output.read_bytes()
            current = _MODULE._report_storage_bytes(self.root)
            rejected_bytes = _MODULE._serialize_report_bytes(next_report)
            with patch.object(_MODULE, "_REPORT_BYTE_LIMIT", current + 1_000):
                expected_summary = _MODULE._budget_failure_summary(
                    self.output, self.context, len(rejected_bytes), current)
                expected_summary_bytes = _MODULE._serialize_report_bytes(expected_summary)
                self.assertLessEqual(current + len(expected_summary_bytes),
                                     _MODULE._REPORT_BYTE_LIMIT)
                with self.assertRaises(_MODULE.ReportBudgetError):
                    self._persist(next_report, reserve=reserve)

            sidecar = self.output.with_name(self.output.name + ".budget-failure.json")
            self.assertEqual(self.output.read_bytes(), prior_bytes)
            self.assertEqual(sidecar.read_bytes(), expected_summary_bytes)
            self.assertLessEqual(_MODULE._report_storage_bytes(self.root),
                                 _MODULE._REPORT_BYTE_LIMIT)
            self.assertFalse(self.output.with_suffix(self.output.suffix + ".tmp").exists())
            self.assertFalse(sidecar.with_suffix(sidecar.suffix + ".tmp").exists())
            self.assertIn(b"incomplete_budget_failure", sidecar.read_bytes())
            self.assertIn(b"unattempted_frame_ids", sidecar.read_bytes())

    def test_failed_atomic_replace_removes_temporary_and_preserves_prior_report(self):
        prior = {"schema_version": 1, "complete": False, "frames": []}
        replacement = {"schema_version": 1, "complete": True,
                       "frames": [{"frameId": 10, "pose_state": "tracking"}]}
        with patch.object(_MODULE, "_OUTPUT_ROOT", self.root), \
             patch.object(_MODULE, "_REPORT_BYTE_LIMIT", 1_000_000):
            self._persist(prior)
            prior_bytes = self.output.read_bytes()
            original_replace = Path.replace

            def fail_report_replace(path, target):
                if path.name.endswith(".json.tmp"):
                    raise OSError("synthetic atomic rename failure")
                return original_replace(path, target)

            with patch.object(Path, "replace", fail_report_replace):
                with self.assertRaisesRegex(OSError, "synthetic atomic rename failure"):
                    self._persist(replacement)

            self.assertEqual(self.output.read_bytes(), prior_bytes)
            self.assertFalse(self.output.with_suffix(self.output.suffix + ".tmp").exists())

    def test_nonfinite_report_fails_before_any_file_is_created(self):
        bad = {"schema_version": 1, "complete": False, "score": float("nan")}
        with patch.object(_MODULE, "_OUTPUT_ROOT", self.root), \
             patch.object(_MODULE, "_REPORT_BYTE_LIMIT", 1_000_000):
            with self.assertRaises(ValueError):
                self._persist(bad)
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.rglob("*.json.tmp")), [])

    def test_aggregate_counter_includes_nested_reports_and_atomic_temporaries(self):
        control_dir = self.root / "control"
        geometry_dir = self.root / "geometry"
        control_dir.mkdir()
        geometry_dir.mkdir()
        (control_dir / "control-240.json").write_bytes(b"control")
        (geometry_dir / "geometry.json.tmp").write_bytes(b"temp-report")
        (geometry_dir / "notes.txt").write_bytes(b"ignored")
        self.assertEqual(_MODULE._report_storage_bytes(self.root),
                         len(b"control") + len(b"temp-report"))


class PoseCleanupReportTests(unittest.TestCase):
    def test_renderer_cleanup_failure_rewrites_last_report_as_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "candidate-120.json"
            marker = root / "candidate-120.json.running"
            guarded_import = builtins.__import__
            runner = types.ModuleType("bench.quality_runner")
            runner.read_rgb = lambda _bundle, _manifest, _frame_id: np.zeros((2, 2, 3), dtype=np.uint8)
            runner._make_gotrack_refiner = lambda *_args, **_kwargs: object()
            runner._pnp_tracking_settings = lambda *_args, **_kwargs: {}
            runner.cv2 = types.SimpleNamespace(error=type("FakeCVError", (Exception,), {}))

            class Frame:
                def __init__(self, frame_id, rgb, intrinsics):
                    self.frame_id = frame_id
                    self.rgb = rgb
                    self.intrinsics = intrinsics

            class MaskPrediction:
                def __init__(self, frame_id, mask, state, reason, timings):
                    self.frame_id = frame_id
                    self.mask = mask
                    self.state = state
                    self.reason = reason
                    self.timings = timings

            class Tracker:
                def __init__(self, _backend):
                    self.accepted = None
                    self.pending = None

                def update(self, frame, segmentation):
                    return {"frameId": frame.frame_id, "cameraFromObject": None,
                            "mask_state": segmentation.state,
                            "pose_state": "lost", "render_state": "suppressed"}

            contract = types.ModuleType("bench.quality_contract")
            contract.Frame = Frame
            contract.MaskPrediction = MaskPrediction
            contract.SequentialTracker = Tracker

            class FakeRecovery:
                def __init__(self, refiner, bank, model_memory):
                    self.refiner = refiner
                    self.bank = bank
                    self.model_memory = model_memory

            foundpose = types.ModuleType("bench.quality_foundpose")
            foundpose.FoundPoseRecovery = FakeRecovery

            class FakeAppearance:
                def __init__(self, backend, _renderer, _settings):
                    self.backend = backend

            appearance = types.ModuleType("bench.quality_appearance")
            appearance.AppearanceCheckedBackend = FakeAppearance
            appearance.AppearanceSettings = lambda: types.SimpleNamespace()

            class AppearanceRenderer:
                def __init__(self, *_args, **_kwargs):
                    self.closed = False

                def close(self):
                    self.closed = True

            gotrack = types.ModuleType("bench.quality_gotrack")
            gotrack.load_network = lambda _device: object()
            gotrack.TexturedRenderer = AppearanceRenderer

            torch = types.ModuleType("torch")
            torch.load = lambda *_args, **_kwargs: {"inert": True}

            def fixture_import(name, *args, **kwargs):
                if name == "torch":
                    return torch
                return guarded_import(name, *args, **kwargs)

            class DualRenderer:
                def __init__(self, *_args, **_kwargs):
                    self.render_records = []
                    self.closed = False

                def assert_healthy(self):
                    return None

                def close(self):
                    self.closed = True
                    raise RuntimeError("synthetic paired renderer cleanup failure")

            manifest = {"asset": "asset.glb", "video": "source.mp4", "object_id": 15,
                        "intrinsics": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
            mask_records = {
                frame_id: {"frameId": frame_id, "path": f"{frame_id}.png",
                           "mask_state": "available"}
                for frame_id in (9, *range(10, 130))
            }

            def build_report(**kwargs):
                return {
                    "status": kwargs["status"],
                    "complete": kwargs["status"] == "complete",
                    "frames": list(kwargs["frames"]),
                    "unattempted_frame_ids": list(kwargs.get("unattempted_ids", ())),
                    "experiment_pins": {"error": kwargs.get("error")},
                }

            with patch.dict(sys.modules, {
                    "bench.quality_runner": runner,
                    "bench.quality_contract": contract,
                    "bench.quality_foundpose": foundpose,
                    "bench.quality_appearance": appearance,
                    "bench.quality_gotrack": gotrack,
            }), patch.object(_MODULE, "_OUTPUT_ROOT", root), \
                    patch.object(_MODULE, "_REPORT_BYTE_LIMIT", 1_000_000), \
                    patch.object(_MODULE, "_load_fixed_inputs",
                                 return_value=(runner, manifest, {}, mask_records)), \
                    patch.object(_MODULE, "_claim_output", return_value=(output, marker)), \
                    patch.object(_MODULE, "_load_mask",
                                 return_value=np.ones((2, 2), dtype=np.uint8)), \
                    patch.object(_MODULE, "_run_shared_seed_probe",
                                 return_value={"status": "passed"}), \
                    patch.object(_MODULE, "LegacyDepthAblationRenderer", DualRenderer), \
                    patch.object(builtins, "__import__", fixture_import), \
                    patch.object(_MODULE, "_build_report", side_effect=build_report), \
                    patch.object(_MODULE, "inference_provenance", return_value={}), \
                    patch.object(_MODULE, "_frame_hash", return_value="D" * 64):
                with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                    _MODULE.pose("candidate", output, device="cpu", prefix_120=True)

            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "integrity_failure")
            self.assertFalse(persisted["complete"])
            self.assertEqual(persisted["cleanup_errors"],
                             ["RuntimeError: synthetic paired renderer cleanup failure"])
            self.assertEqual(len(persisted["frames"]), 120)


class ZeroSampleMetadataTests(unittest.TestCase):
    _MODULES = {
        "quality_gotrack": "bench.quality_gotrack",
        "quality_render_stability": "bench.quality_render_stability",
        "pyrender_renderer": "pyrender.renderer",
        "pyrender_offscreen": "pyrender.offscreen",
    }

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.fake_modules = {}
        source_rows = {}
        pure_allocation_validator = importlib.import_module(
            "bench.quality_render_stability").validate_capture_allocation_pairs
        for key, module_name in self._MODULES.items():
            path = self.root / (key.replace("_", "-") + ".py")
            data = ("pinned test source for " + module_name).encode("utf-8")
            path.write_bytes(data)
            fake = types.ModuleType(module_name)
            fake.__file__ = str(path)
            if module_name == "bench.quality_render_stability":
                fake.validate_capture_allocation_pairs = pure_allocation_validator
            self.fake_modules[module_name] = fake
            source_rows[key] = {
                "module": module_name,
                "file": str(path),
                "sha256": hashlib.sha256(data).hexdigest().upper(),
            }
        self._module_patch = patch.dict(sys.modules, self.fake_modules)
        self._module_patch.start()
        self.addCleanup(self._module_patch.stop)
        color_allocation = {
            "format_name": "GL_RGBA", "target_name": "GL_RENDERBUFFER",
            "samples": 4, "width": 280, "height": 280,
            "ordinary_storage_delegated": True, "success": True,
            "renderbuffer_id": 301,
        }
        depth_allocation = {
            "format_name": "GL_DEPTH_COMPONENT24", "target_name": "GL_RENDERBUFFER",
            "samples": 4, "width": 280, "height": 280,
            "ordinary_storage_delegated": True, "success": True,
            "renderbuffer_id": 302,
        }
        allocation_calls = [color_allocation, depth_allocation]
        self.metadata = {
            "schema_version": 1,
            "render_policy": "capture_zero_sample_v1",
            "coordinate_mode": "integer_centers_v1",
            "depth_units": "millimetres",
            "dimensions": [280, 280],
            "framebuffer_complete": True,
            "framebuffer_bindings_restored": True,
            "current_context_released": True,
            "dimension_match": True,
            "gl_samples": 0,
            "gl_sample_buffers": 0,
            "color_storage_policy": "ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage",
            "depth_storage_policy": "ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage",
            "allocation_generation": 1,
            "offscreen_identity": 707,
            "framebuffer_fields": {
                "multisample_draw_fbo": 101,
                "single_sample_read_fbo": 102,
                "multisample_dimensions": [280, 280],
            },
            "allocation_pair": {"passed": True, "dimensions": [280, 280],
                                 "color": dict(color_allocation),
                                 "depth": dict(depth_allocation)},
            "allocation_calls": allocation_calls,
            "source_identities": source_rows,
        }
        self._digest_patch = patch.object(
            _MODULE, "digest",
            side_effect=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest().upper())
        self._digest_patch.start()
        self.addCleanup(self._digest_patch.stop)

    def test_valid_first_allocation_and_same_identity_reuse_pass(self):
        first = _MODULE._validate_zero_metadata(
            self.metadata, expected_dimensions=(280, 280), previous=None)
        reused = copy.deepcopy(self.metadata)
        reused["allocation_calls"] = []
        checked = _MODULE._validate_zero_metadata(
            reused, expected_dimensions=(280, 280), previous=first)
        self.assertEqual(checked["allocation_generation"], 1)
        self.assertEqual(checked["framebuffer_fields"], first["framebuffer_fields"])

    def test_policy_samples_dimensions_generation_and_allocation_fail_closed(self):
        invalid_cases = (
            ("wrong policy", lambda value: value.__setitem__("render_policy", "legacy_v1")),
            ("wrong coordinate mode", lambda value: value.__setitem__("coordinate_mode", "legacy")),
            ("wrong units", lambda value: value.__setitem__("depth_units", "metres")),
            ("GL samples", lambda value: value.__setitem__("gl_samples", 4)),
            ("sample buffers", lambda value: value.__setitem__("gl_sample_buffers", 1)),
            ("wrong dimensions", lambda value: value.__setitem__("dimensions", [279, 280])),
            ("wrong storage", lambda value: value.__setitem__("depth_storage_policy", "multisample")),
            ("bad lifecycle", lambda value: value.__setitem__("current_context_released", False)),
            ("bad generation", lambda value: value.__setitem__("allocation_generation", 0)),
            ("bad pair", lambda value: value["allocation_pair"].__setitem__("passed", False)),
            ("missing allocation", lambda value: value.__setitem__("allocation_calls", [])),
        )
        for label, mutate in invalid_cases:
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.metadata)
                mutate(candidate)
                with self.assertRaises(RuntimeError):
                    _MODULE._validate_zero_metadata(
                        candidate, expected_dimensions=(280, 280), previous=None)

    def test_allocation_pair_must_join_authentic_color_and_depth_rows(self):
        invalid_cases = (
            ("missing color row", lambda value: value["allocation_pair"].pop("color")),
            ("missing depth row", lambda value: value["allocation_pair"].pop("depth")),
            ("crossed rows", lambda value: value["allocation_pair"].update(
                color=copy.deepcopy(value["allocation_pair"]["depth"]),
                depth=copy.deepcopy(value["allocation_pair"]["color"]))),
            ("malformed row", lambda value: value["allocation_pair"]["color"].update(samples=True)),
            ("row does not match allocation log", lambda value:
                value["allocation_pair"]["color"].update(renderbuffer_id=999)),
        )
        for label, mutate in invalid_cases:
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.metadata)
                mutate(candidate)
                with self.assertRaises(RuntimeError):
                    _MODULE._validate_zero_metadata(
                        candidate, expected_dimensions=(280, 280), previous=None)

    def test_offscreen_identity_must_be_positive_non_boolean_integer(self):
        for invalid in (0, -1, True, False, None, 707.0):
            with self.subTest(offscreen_identity=invalid):
                candidate = copy.deepcopy(self.metadata)
                candidate["offscreen_identity"] = invalid
                with self.assertRaises(RuntimeError):
                    _MODULE._validate_zero_metadata(
                        candidate, expected_dimensions=(280, 280), previous=None)

    def test_missing_stale_source_and_stale_framebuffer_evidence_fail_closed(self):
        with self.assertRaises(RuntimeError):
            _MODULE._validate_zero_metadata(None, expected_dimensions=(280, 280), previous=None)

        first = _MODULE._validate_zero_metadata(
            self.metadata, expected_dimensions=(280, 280), previous=None)
        changed_generation = copy.deepcopy(self.metadata)
        changed_generation["allocation_calls"] = []
        changed_generation["allocation_generation"] = 2
        with self.assertRaises(RuntimeError):
            _MODULE._validate_zero_metadata(
                changed_generation, expected_dimensions=(280, 280), previous=first)

        changed_fbo = copy.deepcopy(self.metadata)
        changed_fbo["allocation_calls"] = []
        changed_fbo["framebuffer_fields"]["multisample_draw_fbo"] = 103
        with self.assertRaises(RuntimeError):
            _MODULE._validate_zero_metadata(
                changed_fbo, expected_dimensions=(280, 280), previous=first)

        wrong_source = copy.deepcopy(self.metadata)
        wrong_source["source_identities"]["quality_gotrack"]["module"] = "elsewhere"
        with self.assertRaises(RuntimeError):
            _MODULE._validate_zero_metadata(
                wrong_source, expected_dimensions=(280, 280), previous=None)

        changed_path = Path(self.metadata["source_identities"]["quality_gotrack"]["file"])
        changed_path.write_bytes(b"source changed after metadata was recorded")
        with self.assertRaises(RuntimeError):
            _MODULE._validate_zero_metadata(
                self.metadata, expected_dimensions=(280, 280), previous=None)


class FixedScopeAndOutputTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.bundle = self.root / "bundle"
        self.masks = self.root / "masks"
        self.bundle.mkdir()
        self.masks.mkdir()
        (self.bundle / "input.json").write_text("{}", encoding="utf-8")
        self.mask_files = self.masks / "files"
        self.mask_files.mkdir()
        self.mask_rows = []
        for frame_id in (9, *range(10, 250)):
            relative = f"files/{frame_id}.png"
            (self.masks / relative).write_bytes(b"inert placeholder; never decoded")
            self.mask_rows.append({"frameId": frame_id, "mask_state": "available",
                                   "path": relative})
        self.manifest = {
            "_capture": None,
            "_timing": {"clock_mode": "legacy_nominal_60_v1"},
            "object": "ranch", "object_id": 15, "setup_frame_id": 9,
            "frame_ids": tuple(range(10, 250)), "units": "metres",
        }
        self.mask_manifest = {
            "diagnostic_control": False, "mask_association": False,
            "provenance": {"input_manifest_sha256": "B" * 64},
            "frames": self.mask_rows,
        }
        self.mask_index = self.masks / "results.json"
        self._write_mask_manifest()

    def _write_mask_manifest(self, manifest=None):
        self.mask_index.write_text(json.dumps(self.mask_manifest if manifest is None else manifest),
                                   encoding="utf-8")

    def _call_fixed_inputs(self, manifest=None):
        runner = types.ModuleType("bench.quality_runner")
        runner.read_input = lambda _bundle: self.manifest if manifest is None else manifest
        time_module = types.ModuleType("bench.quality_time")
        time_module.LEGACY = "legacy_nominal_60_v1"
        with patch.object(_MODULE, "_BUNDLE", self.bundle), \
             patch.object(_MODULE, "_MASKS", self.masks), \
             patch.object(_MODULE, "_frame_hash", return_value="B" * 64), \
             patch.dict(sys.modules, {"bench.quality_runner": runner,
                                      "bench.quality_time": time_module}):
            return _MODULE._load_fixed_inputs()

    def test_fixed_original_window_and_original_masks_are_accepted(self):
        _runner, manifest, mask_manifest, records = self._call_fixed_inputs()
        self.assertEqual(manifest["object"], "ranch")
        self.assertEqual(manifest["setup_frame_id"], 9)
        self.assertEqual(tuple(records), tuple(sorted((9, *range(10, 250)))))
        self.assertFalse(mask_manifest["diagnostic_control"])
        self.assertFalse(mask_manifest["mask_association"])

    def test_capture_physical_wrong_object_window_and_units_are_rejected(self):
        invalid = []
        value = copy.deepcopy(self.manifest); value["_capture"] = {"enabled": True}; invalid.append(value)
        value = copy.deepcopy(self.manifest); value["_timing"]["clock_mode"] = "physical"; invalid.append(value)
        value = copy.deepcopy(self.manifest); value["object"] = "mug"; invalid.append(value)
        value = copy.deepcopy(self.manifest); value["object_id"] = 8; invalid.append(value)
        value = copy.deepcopy(self.manifest); value["setup_frame_id"] = 8; invalid.append(value)
        value = copy.deepcopy(self.manifest); value["frame_ids"] = tuple(range(11, 250)); invalid.append(value)
        value = copy.deepcopy(self.manifest); value["units"] = "millimetres"; invalid.append(value)
        for manifest in invalid:
            with self.subTest(manifest=manifest):
                with self.assertRaises(ValueError):
                    self._call_fixed_inputs(manifest)

    def test_diagnostic_associated_missing_and_escaping_masks_are_rejected(self):
        for key in ("diagnostic_control", "mask_association"):
            candidate = copy.deepcopy(self.mask_manifest)
            candidate[key] = True
            self._write_mask_manifest(candidate)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._call_fixed_inputs()

        candidate = copy.deepcopy(self.mask_manifest)
        candidate["frames"] = candidate["frames"][:-1]
        self._write_mask_manifest(candidate)
        with self.assertRaises(ValueError):
            self._call_fixed_inputs()

        candidate = copy.deepcopy(self.mask_manifest)
        candidate["frames"][0]["path"] = "../../outside.png"
        self._write_mask_manifest(candidate)
        with self.assertRaises(ValueError):
            self._call_fixed_inputs()

        candidate = copy.deepcopy(self.mask_manifest)
        candidate["frames"][0]["mask_state"] = "lost"
        self._write_mask_manifest(candidate)
        with self.assertRaises(ValueError):
            self._call_fixed_inputs()

    def test_output_claim_refuses_overwrite_and_paths_outside_fixed_root(self):
        output_root = self.root / "results"
        output_root.mkdir()
        output = output_root / "candidate-240.json"
        output.write_text("prior immutable trial", encoding="utf-8")
        with patch.object(_MODULE, "_OUTPUT_ROOT", output_root):
            with self.assertRaises(FileExistsError):
                _MODULE._claim_output(output, "candidate-240.json")
            self.assertEqual(output.read_text(encoding="utf-8"), "prior immutable trial")
            with self.assertRaises(ValueError):
                _MODULE._claim_output(self.root / "outside" / "candidate-240.json",
                                      "candidate-240.json")

    def test_cli_exposes_no_external_pose_or_reference_seed(self):
        parser = _MODULE.build_parser()
        base = ["pose", "--object", "ranch", "--branch", "candidate",
                "--output", str(self.root / "candidate-240.json")]
        for forbidden in (("--initial-pose", "pose.json"),
                          ("--reference-pose", "pose.json"),
                          ("--seed", "pose.json")):
            with self.subTest(forbidden=forbidden):
                with patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(base + list(forbidden))


if __name__ == "__main__":
    unittest.main()
