"""Independent fake-only tests for the explicit capture render policy.

The public TexturedRenderer path is exercised with a fake asset loader,
PyRender module, and OpenGL service. Only those exact optional imports are
replaced; all other imports delegate to the active guarded importer.
"""

from __future__ import annotations

import builtins
from contextlib import ExitStack
from enum import IntEnum
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import quality_gotrack as gotrack
from . import quality_render_stability as stability
from . import test_quality_capture_refiner as refiner_tests


class _RenderType(IntEnum):
    COLOR = 1
    DEPTH = 2
    MASK = 3


class _FakeGL:
    GL_RENDERBUFFER = 0x8D41
    GL_RENDERBUFFER_BINDING = 0x8CA7
    GL_RGBA = 0x1908
    GL_DEPTH_COMPONENT24 = 0x81A6
    GL_DRAW_FRAMEBUFFER = 0x8CA9
    GL_READ_FRAMEBUFFER = 0x8CA8
    GL_FRAMEBUFFER = 0x8D40
    GL_DRAW_FRAMEBUFFER_BINDING = 0x8CA6
    GL_READ_FRAMEBUFFER_BINDING = 0x8CAA
    GL_FRAMEBUFFER_BINDING = GL_DRAW_FRAMEBUFFER_BINDING
    GL_FRAMEBUFFER_COMPLETE = 0x8CD5
    GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT = 0x8CD6
    GL_SAMPLES = 0x80A9
    GL_SAMPLE_BUFFERS = 0x80A8
    GL_MULTISAMPLE = 0x809D
    GL_DITHER = 0x0BD0
    GL_VENDOR = 0x1F00
    GL_RENDERER = 0x1F01
    GL_VERSION = 0x1F02

    def __init__(self, runtime=None):
        self.runtime = runtime
        self.draw_binding = 701
        self.read_binding = 702
        self.renderbuffer_binding = 0
        self.samples = 0
        self.sample_buffers = 0
        self.forced_samples = None
        self.forced_sample_buffers = None
        self.complete = True
        self.fail_query = None
        self.fail_status = False
        self.fail_restore_target = None
        self.enabled = {self.GL_MULTISAMPLE, self.GL_DITHER}
        self.calls = []

    def _observe(self, name):
        if self.runtime is not None:
            self.runtime.observe(name)

    def glGetIntegerv(self, pname):
        self._observe("gl_integer_query")
        self.calls.append(("get_integer", pname))
        failed_queries = (self.fail_query if isinstance(self.fail_query, (tuple, list, set))
                          else (self.fail_query,))
        if pname in failed_queries:
            raise RuntimeError(f"fake mandatory GL query failed: {pname}")
        values = {
            self.GL_DRAW_FRAMEBUFFER_BINDING: self.draw_binding,
            self.GL_READ_FRAMEBUFFER_BINDING: self.read_binding,
            self.GL_RENDERBUFFER_BINDING: self.renderbuffer_binding,
            self.GL_SAMPLES: (self.samples if self.forced_samples is None
                              else self.forced_samples),
            self.GL_SAMPLE_BUFFERS: (
                self.sample_buffers if self.forced_sample_buffers is None
                else self.forced_sample_buffers),
        }
        if pname not in values:
            raise AssertionError(f"unexpected fake GL integer query: {pname}")
        return values[pname]

    def glBindFramebuffer(self, target, framebuffer):
        self._observe("gl_framebuffer_bind")
        self.calls.append(("bind_framebuffer", target, framebuffer))
        is_restore = framebuffer in (701, 702)
        if is_restore and self.fail_restore_target == target:
            raise RuntimeError("fake framebuffer binding restoration failed")
        if target in (self.GL_DRAW_FRAMEBUFFER, self.GL_FRAMEBUFFER):
            self.draw_binding = framebuffer
        if target in (self.GL_READ_FRAMEBUFFER, self.GL_FRAMEBUFFER):
            self.read_binding = framebuffer

    def glBindRenderbuffer(self, target, renderbuffer):
        self._observe("gl_renderbuffer_bind")
        if target != self.GL_RENDERBUFFER:
            raise AssertionError("unexpected fake renderbuffer target")
        self.renderbuffer_binding = renderbuffer
        self.calls.append(("bind_renderbuffer", target, renderbuffer))

    def glCheckFramebufferStatus(self, target):
        self._observe("gl_framebuffer_status")
        self.calls.append(("check_framebuffer", target))
        if self.fail_status:
            raise RuntimeError("fake framebuffer-status query failed")
        return (self.GL_FRAMEBUFFER_COMPLETE if self.complete
                else self.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)

    def glIsEnabled(self, capability):
        return capability in self.enabled

    def glEnable(self, capability):
        self._observe("gl_enable")
        self.calls.append(("enable", capability))
        self.enabled.add(capability)

    def glDisable(self, capability):
        self._observe("gl_disable")
        self.calls.append(("disable", capability))
        self.enabled.discard(capability)

    def glGetString(self, name):
        return {
            self.GL_VENDOR: b"synthetic vendor",
            self.GL_RENDERER: b"synthetic renderer",
            self.GL_VERSION: b"synthetic GL",
        }.get(name, b"synthetic")


class _Geometry:
    def __init__(self):
        self.vertices = np.array(
            [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, 0.06, 0.04]],
            dtype=np.float64)

    def copy(self):
        cloned = _Geometry()
        cloned.vertices = self.vertices.copy()
        return cloned

    def apply_transform(self, matrix):
        homogeneous = np.column_stack((self.vertices, np.ones(len(self.vertices))))
        self.vertices = (homogeneous @ matrix.T)[:, :3]


class _FakeLoadedScene:
    def __init__(self):
        self.graph = _FakeGraph()
        self.geometry = {"mesh": _Geometry()}


class _FakeGraph:
    nodes_geometry = ("mesh-node",)

    def __getitem__(self, _node):
        return np.eye(4), "mesh"


class _FakePlatform:
    def __init__(self, runtime, offscreen_id):
        self.runtime = runtime
        self.offscreen_id = offscreen_id
        self.current = False

    def supports_framebuffers(self):
        return self.runtime.framebuffer_supported

    def make_current(self):
        self.runtime.context_acquire_calls += 1
        self.runtime.observe("context_make_current")
        self.current = True
        self.runtime.events.append(("make_current", self.offscreen_id))
        if (self.runtime.fails("context_acquire") or
                self.runtime.context_acquire_calls in self.runtime.context_acquire_fail_calls):
            self.runtime.events.append(("make_current_failed_after_state_change",
                                        self.offscreen_id, self.current))
            raise RuntimeError("fake context acquisition failure")

    def make_uncurrent(self):
        self.runtime.context_release_calls += 1
        self.runtime.observe("context_make_uncurrent")
        self.runtime.events.append(("make_uncurrent", self.offscreen_id))
        self.current = False
        if (self.runtime.fails("context_release") or
                self.runtime.context_release_calls in self.runtime.context_release_fail_calls):
            self.runtime.events.append(("make_uncurrent_failed_after_state_change",
                                        self.offscreen_id, self.current))
            raise RuntimeError("fake context release failure")


class _SceneNode:
    def __init__(self, kind, payload, pose=None):
        self.kind = kind
        self.payload = payload
        self.pose = None if pose is None else np.array(pose, copy=True)


class _FakeScene:
    def __init__(self, runtime, *, bg_color, ambient_light):
        self.runtime = runtime
        self.bg_color = tuple(bg_color)
        self.ambient_light = tuple(ambient_light)
        self.nodes = []
        self.mesh_nodes = []

    def add(self, payload, pose=None):
        self.runtime.observe("scene_add")
        kind = ("camera" if isinstance(payload, _FakeIntrinsicsCamera) else
                "light" if isinstance(payload, _FakeSpotLight) else "mesh")
        node = _SceneNode(kind, payload, pose)
        self.nodes.append(node)
        if kind == "mesh":
            self.mesh_nodes.append(node)
        if kind == "camera":
            self.runtime.camera_records.append((payload, np.array(pose, copy=True)))
        self.runtime.events.append(("scene_add", kind))
        return node

    def remove_node(self, node):
        self.runtime.observe("scene_remove")
        self.runtime.events.append(("scene_remove", node.kind))
        persistent_failure = self.runtime.fails("scene_cleanup_persistent")
        one_shot_failure = (self.runtime.fails("scene_cleanup") and
                            not self.runtime.scene_cleanup_failed)
        if persistent_failure or one_shot_failure:
            self.runtime.scene_cleanup_failed = True
            raise RuntimeError("fake temporary scene-node cleanup failure")
        if node in self.nodes:
            self.nodes.remove(node)
        if node in self.mesh_nodes:
            self.mesh_nodes.remove(node)


class _FakeIntrinsicsCamera:
    def __init__(self, fx, fy, cx, cy, *, znear, zfar):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.znear, self.zfar = znear, zfar


class _FakeSpotLight:
    def __init__(self, *, color, intensity, innerConeAngle, outerConeAngle):
        self.color = np.array(color, copy=True)
        self.intensity = intensity
        self.innerConeAngle = innerConeAngle
        self.outerConeAngle = outerConeAngle


class _FakeMesh:
    @staticmethod
    def from_trimesh(mesh, *, smooth):
        return SimpleNamespace(kind="mesh", vertices=mesh.vertices.copy(), smooth=smooth)


class _FakeOffscreen:
    def __init__(self, runtime, width, height, identifier):
        self.runtime = runtime
        self.viewport_width = width
        self.viewport_height = height
        self.identifier = identifier
        self.deleted = False
        self._platform = _FakePlatform(runtime, identifier)
        metadata_width, metadata_height = runtime.metadata_dimensions or (width, height)
        draw_fbo = identifier * 10 + 1
        if runtime.fractional_fbo:
            draw_fbo += 0.5
        self._renderer = SimpleNamespace(
            _main_fb_ms=draw_fbo,
            _main_fb=identifier * 10 + 2,
            _main_fb_dims=(metadata_width, metadata_height),
        )
        runtime.offscreens.append(self)
        runtime.events.append(("create", width, height, identifier))
        self._allocate(width, height)

    def _allocate(self, width, height):
        runtime = self.runtime
        if runtime.allocation_style == "missing_pair":
            formats = (_FakeGL.GL_RGBA,)
        elif runtime.allocation_style == "same_id":
            formats = (_FakeGL.GL_RGBA, _FakeGL.GL_DEPTH_COMPONENT24)
        elif runtime.allocation_style == "extra_pair_call":
            formats = (_FakeGL.GL_RGBA, _FakeGL.GL_DEPTH_COMPONENT24,
                       _FakeGL.GL_RGBA)
        else:
            formats = (_FakeGL.GL_RGBA, _FakeGL.GL_DEPTH_COMPONENT24)
        for offset, internal_format in enumerate(formats):
            rb_id = (self.identifier * 100 + (0 if runtime.allocation_style == "same_id"
                                               else offset + 1))
            runtime.gl.glBindRenderbuffer(_FakeGL.GL_RENDERBUFFER, rb_id)
            if runtime.allocation_style == "wrong_allocation_dimensions":
                alloc_width, alloc_height = width + 1, height
            else:
                alloc_width, alloc_height = width, height
            runtime.renderer_module.glRenderbufferStorageMultisample(
                _FakeGL.GL_RENDERBUFFER, 4, internal_format,
                alloc_width, alloc_height)
        self.samples_at_creation = runtime.gl.samples
        self.sample_buffers_at_creation = runtime.gl.sample_buffers

    def render(self, scene, *, flags):
        runtime = self.runtime
        runtime.observe("offscreen_render")
        runtime.events.append(("render", self.viewport_width, self.viewport_height, flags))
        runtime.render_calls.append((self.identifier, self.viewport_width,
                                     self.viewport_height, flags))
        runtime.gl.samples = self.samples_at_creation
        runtime.gl.sample_buffers = self.sample_buffers_at_creation
        runtime.renderer_module.glEnable(_FakeGL.GL_MULTISAMPLE)
        if runtime.fails("render") or runtime.fails("readback"):
            stage = "render" if runtime.fails("render") else "readback"
            raise RuntimeError(f"fake {stage} failure")
        height, width = self.viewport_height, self.viewport_width
        color = np.full((height, width, 4), 127, dtype=np.uint8)
        depth = np.full((height, width), 1.25, dtype=np.float32)
        depth[0, 0] = 0.0
        runtime.last_raw_color = color
        runtime.last_raw_depth = depth
        if runtime.fails("depth_readback"):
            depth = _FailingArray(depth)
        return color, depth

    def delete(self):
        self.runtime.observe("offscreen_delete")
        self.runtime.events.append(("delete", self.identifier))
        self.deleted = True
        if self.runtime.fails("delete"):
            raise OSError("fake offscreen deletion failure")


class _FailingArray:
    def __init__(self, array):
        self.array = array

    def astype(self, *_args, **_kwargs):
        raise RuntimeError("fake readback conversion failure")

    def __gt__(self, value):
        return self.array > value


class _RLockSpy:
    """Single-threaded reentrant lock recorder; it never performs native locking."""

    def __init__(self):
        self.depth = 0
        self.max_depth = 0
        self.acquire_count = 0
        self.release_count = 0
        self.events = []

    def acquire(self, *_args, **_kwargs):
        self.depth += 1
        self.acquire_count += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.events.append(("acquire", self.depth))
        return True

    def release(self):
        if self.depth <= 0:
            raise RuntimeError("fake RLock release without acquire")
        self.events.append(("release", self.depth))
        self.depth -= 1
        self.release_count += 1

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False

    def observe(self, name):
        self.events.append(("operation", name, self.depth))


class _RendererRuntime:
    """Narrow fake importer and GL/PyRender services for one render test."""

    def __init__(self, *, failure=None, allocation_style="normal", fractional_fbo=False):
        self.failure = failure
        self.allocation_style = allocation_style
        self.fractional_fbo = fractional_fbo
        self.lock_spy = None
        self.context_acquire_calls = 0
        self.context_release_calls = 0
        self.context_acquire_fail_calls = set()
        self.context_release_fail_calls = set()
        self.framebuffer_supported = True
        self.metadata_dimensions = None
        self.events = []
        self.offscreens = []
        self.render_calls = []
        self.camera_records = []
        self.original_storage_calls = []
        self.ordinary_storage_calls = []
        self.trace_calls = []
        self.tensor_calls = []
        self.scene_cleanup_failed = False
        self.last_raw_color = None
        self.last_raw_depth = None
        self._next_offscreen = 10
        self.gl = _FakeGL(self)

        renderer_module = ModuleType("fake_pyrender.renderer")

        def gl_renderbuffer_storage_multisample(target, samples, internal_format, width, height):
            self.observe("multisample_storage")
            self.original_storage_calls.append(
                (target, samples, internal_format, width, height))
            self.gl.samples = samples
            self.gl.sample_buffers = 1 if samples > 0 else 0

        def gl_renderbuffer_storage(target, internal_format, width, height):
            self.observe("ordinary_storage")
            self.ordinary_storage_calls.append((target, internal_format, width, height))
            self.gl.samples = 0
            self.gl.sample_buffers = 0
            if self.fails("allocation"):
                raise OSError("fake ordinary-storage allocation failure")

        renderer_module.glRenderbufferStorageMultisample = gl_renderbuffer_storage_multisample
        renderer_module.glRenderbufferStorage = gl_renderbuffer_storage
        renderer_module.GL_MULTISAMPLE = _FakeGL.GL_MULTISAMPLE
        renderer_module.glEnable = self.gl.glEnable
        renderer_module.glDisable = self.gl.glDisable
        renderer_module.__file__ = __file__
        self.renderer_module = renderer_module

        runtime = self

        class OffscreenRenderer:
            def __init__(self, width, height):
                runtime.observe("offscreen_create")
                identifier = runtime._next_offscreen
                runtime._next_offscreen += 1
                self._fake = _FakeOffscreen(runtime, width, height, identifier)
                self.__dict__.update(self._fake.__dict__)

            def render(self, scene, *, flags):
                return self._fake.render(scene, flags=flags)

            def delete(self):
                return self._fake.delete()

        class Scene:
            def __new__(cls, *, bg_color, ambient_light):
                return _FakeScene(runtime, bg_color=bg_color,
                                  ambient_light=ambient_light)

        pyrender = ModuleType("pyrender")
        pyrender.Scene = Scene
        pyrender.Mesh = _FakeMesh
        pyrender.OffscreenRenderer = OffscreenRenderer
        pyrender.IntrinsicsCamera = _FakeIntrinsicsCamera
        pyrender.SpotLight = _FakeSpotLight
        pyrender.RenderFlags = SimpleNamespace(FLAT=1, NONE=0)
        pyrender.renderer = renderer_module
        offscreen_module = ModuleType("pyrender.offscreen")
        offscreen_module.__file__ = __file__
        pyrender.offscreen = offscreen_module
        self.fake_pyrender = pyrender
        self.fake_offscreen_module = offscreen_module

        trimesh = ModuleType("trimesh")
        trimesh.load = lambda path, **kwargs: self._load_mesh(path, kwargs)
        self.fake_trimesh = trimesh

        self.fake_gl_module = ModuleType("OpenGL.GL")
        for name in dir(self.gl):
            if name.startswith("GL_") or name.startswith("gl"):
                setattr(self.fake_gl_module, name, getattr(self.gl, name))
        opengl = ModuleType("OpenGL")
        opengl.GL = self.fake_gl_module
        self.fake_opengl = opengl

        renderer_base = ModuleType("utils.renderer_base")
        renderer_base.RenderType = _RenderType
        self.fake_renderer_base = renderer_base

        torch = ModuleType("torch")
        torch.from_numpy = self._from_numpy
        self.fake_torch = torch

    def fails(self, name):
        if isinstance(self.failure, str):
            return self.failure == name
        return name in (self.failure or ())

    def observe(self, name):
        if self.lock_spy is not None:
            self.lock_spy.observe(name)

    def _load_mesh(self, path, kwargs):
        self.events.append(("load_mesh", str(path), dict(kwargs)))
        return _FakeLoadedScene()

    def _from_numpy(self, array):
        self.observe("tensor_conversion")
        self.tensor_calls.append(np.array(array, copy=True))
        return _FakeTensor(array)

    def _trace(self, *args, **kwargs):
        self.observe("trace")
        self.trace_calls.append((args, kwargs))

    def fake_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyrender":
            return self.fake_pyrender
        if name == "pyrender.offscreen":
            # Match Python's ``import package.submodule as alias`` behavior:
            # with no fromlist, __import__ returns the package and IMPORT_FROM
            # obtains the registered submodule attribute from it.
            return self.fake_pyrender if not fromlist else self.fake_offscreen_module
        if name == "trimesh":
            return self.fake_trimesh
        if name == "OpenGL":
            return self.fake_opengl
        if name == "OpenGL.GL":
            return self.fake_gl_module
        if name == "utils.renderer_base":
            return self.fake_renderer_base
        if name == "torch":
            return self.fake_torch
        if name.split(".", 1)[0] in {"pyrender", "trimesh", "OpenGL", "torch"}:
            raise AssertionError(f"unexpected real optional import in fake renderer test: {name}")
        return self._delegate_import(name, globals, locals, fromlist, level)

    def __enter__(self):
        self._delegate_import = builtins.__import__
        self._stack = ExitStack()
        self._stack.enter_context(patch.object(builtins, "__import__", self.fake_import))
        self._stack.enter_context(patch.object(gotrack, "upstream_path", lambda: None))
        self._stack.enter_context(patch.object(gotrack, "digest", lambda _path: "A" * 64))
        self._stack.enter_context(patch.object(gotrack, "trace", self._trace))
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._stack.__exit__(exc_type, exc, traceback)


class _FakeTensor:
    def __init__(self, value):
        self.array = np.asarray(value)


def _camera(width, height, *, translation_mm=(100.0, -25.0, 2000.0)):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = translation_mm
    return SimpleNamespace(
        width=width,
        height=height,
        f=(123.0, 237.0),
        c=(17.25, 29.75),
        T_world_from_eye=pose,
    )


def _new_renderer(runtime, *, render_policy=None, disable_multisampling=False,
                  coordinate_mode="legacy", unlit=False):
    kwargs = dict(unlit=unlit, disable_multisampling=disable_multisampling,
                  coordinate_mode=coordinate_mode)
    if render_policy is not None:
        kwargs["render_policy"] = render_policy
    return gotrack.TexturedRenderer("synthetic.glb", "synthetic-object", **kwargs)


class CaptureRenderPolicyTests(unittest.TestCase):
    def test_constructor_rejects_bad_policy_before_optional_loaders(self):
        invalid = (
            {"render_policy": "unknown_v9"},
            {"render_policy": "capture_zero_sample_v1"},
            {"render_policy": "capture_zero_sample_v1",
             "coordinate_mode": "integer_centers_v1", "disable_multisampling": True},
            {"render_policy": "capture_zero_sample_v1",
             "coordinate_mode": "integer_centers_v1", "disable_multisampling": np.bool_(False)},
        )
        with patch.object(gotrack, "upstream_path",
                          side_effect=AssertionError("loader ran before policy validation")):
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises((TypeError, ValueError)):
                        gotrack.TexturedRenderer("absent.glb", "synthetic-object", **kwargs)

    def test_legacy_default_and_explicit_policy_keep_mapping_pose_flags_and_tensor_results(self):
        for disable in (False, True):
            outputs = []
            facts = []
            for explicit in (False, True):
                runtime = _RendererRuntime()
                with runtime:
                    if explicit:
                        renderer = gotrack.TexturedRenderer(
                            "synthetic.glb", "synthetic-object", True, disable,
                            "legacy", render_policy="legacy_v1")
                    else:
                        renderer = gotrack.TexturedRenderer(
                            "synthetic.glb", "synthetic-object", True, disable, "legacy")
                    self.assertEqual(renderer.render_policy, "legacy_v1")
                    with self.assertRaises(AttributeError):
                        renderer.render_policy = "capture_zero_sample_v1"
                    result = renderer.render_object_model(
                        "synthetic-object", _camera(8, 6), return_tensors=True)
                    outputs.append({key: value.array.copy() for key, value in result.items()})
                    self.assertEqual([node.kind for node in renderer.scene.nodes], ["mesh"])
                    render_node = runtime.render_calls[-1]
                    self.assertEqual(render_node[1:3], (8, 6))
                    self.assertEqual(runtime.ordinary_storage_calls, [])
                    self.assertEqual(len(runtime.original_storage_calls), 2)
                    self.assertEqual(runtime.tensor_calls[0].dtype, np.float32)
                    self.assertIs(renderer.render_policy_metadata, None)
                    renderer.close()
                    facts.append((runtime.render_calls, runtime.gl.calls,
                                  runtime.original_storage_calls))
            self.assertEqual(outputs[0].keys(), {_RenderType.COLOR, _RenderType.DEPTH, _RenderType.MASK})
            self.assertEqual(outputs[0].keys(), outputs[1].keys())
            for key in outputs[0]:
                np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
            self.assertEqual(facts[0][0], facts[1][0])
            self.assertEqual(facts[0][2], facts[1][2])
            self.assertEqual(facts[0][0][0][3], 1)  # FLAT is retained for unlit legacy renders.
            if disable:
                self.assertIn(("disable", _FakeGL.GL_MULTISAMPLE), facts[0][1])
            else:
                self.assertIn(("enable", _FakeGL.GL_MULTISAMPLE), facts[0][1])

    def test_capture_policy_preserves_camera_units_mask_and_resize_generations(self):
        runtime = _RendererRuntime()
        with runtime:
            renderer = _new_renderer(
                runtime, render_policy="capture_zero_sample_v1",
                coordinate_mode="integer_centers_v1")
            dimensions = ((280, 280), (280, 280), (720, 720),
                          (1120, 1120), (280, 280))
            expected_call_counts = (2, 0, 2, 2, 2)
            snapshots = []
            generations = []
            pair_signatures = []
            for iteration, (width, height) in enumerate(dimensions):
                storage_calls_before = len(runtime.ordinary_storage_calls)
                result = renderer.render_object_model(
                    "synthetic-object", _camera(width, height), return_tensors=False)
                self.assertEqual(
                    len(runtime.ordinary_storage_calls) - storage_calls_before,
                    expected_call_counts[iteration])
                self.assertEqual(set(result), {_RenderType.COLOR, _RenderType.DEPTH, _RenderType.MASK})
                self.assertEqual(result[_RenderType.COLOR].shape, (height, width, 4))
                self.assertEqual(result[_RenderType.COLOR].dtype, np.float32)
                self.assertAlmostEqual(float(result[_RenderType.COLOR][0, 0, 0]), 127 / 255)
                self.assertEqual(result[_RenderType.DEPTH].dtype, np.float32)
                self.assertAlmostEqual(float(result[_RenderType.DEPTH][1, 1]), 1250.0, places=3)
                self.assertEqual(result[_RenderType.MASK].dtype, np.bool_)
                self.assertFalse(result[_RenderType.MASK][0, 0])
                self.assertTrue(result[_RenderType.MASK][1, 1])

                camera, camera_pose = runtime.camera_records[-1]
                self.assertEqual((camera.fx, camera.fy), (123.0, 237.0))
                self.assertEqual((camera.cx, camera.cy), (17.75, 30.25))
                np.testing.assert_array_equal(
                    camera_pose[:3, 3], [0.1, -0.025, 2.0])
                self.assertEqual(runtime.ordinary_storage_calls[-2][2:], (width, height))

                metadata = renderer.render_policy_metadata
                self.assertIsNotNone(metadata)
                self.assertEqual(metadata["render_policy"], "capture_zero_sample_v1")
                self.assertEqual(metadata["coordinate_mode"], "integer_centers_v1")
                self.assertEqual(metadata["dimensions"], (width, height))
                self.assertIs(metadata["framebuffer_complete"], True)
                self.assertEqual(metadata["gl_samples"], 0)
                self.assertEqual(metadata["gl_sample_buffers"], 0)
                self.assertEqual(metadata["depth_units"], "millimetres")
                self.assertIs(metadata["framebuffer_bindings_restored"], True)
                self.assertIs(metadata["current_context_released"], True)
                self.assertIs(metadata["dimension_match"], True)
                self.assertEqual(
                    metadata["color_storage_policy"],
                    "ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage")
                self.assertEqual(
                    metadata["depth_storage_policy"],
                    "ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage")
                self.assertEqual(
                    set(metadata["allocation_pair"]),
                    {"passed", "dimensions", "color", "depth"})
                self.assertIs(metadata["allocation_pair"]["passed"], True)
                self.assertEqual(metadata["allocation_pair"]["dimensions"], (width, height))
                self.assertEqual(len(metadata["allocation_calls"]),
                                 expected_call_counts[iteration])
                pair = metadata["allocation_pair"]
                pair_rows = {"GL_RGBA": pair["color"],
                             "GL_DEPTH_COMPONENT24": pair["depth"]}
                self.assertEqual(
                    {row["format_name"] for row in pair_rows.values()},
                    {"GL_RGBA", "GL_DEPTH_COMPONENT24"})
                for format_name, row in pair_rows.items():
                    self.assertEqual(row["target_name"], "GL_RENDERBUFFER")
                    self.assertEqual(row["format_name"], format_name)
                    self.assertEqual(row["samples"], 4)
                    self.assertEqual((row["width"], row["height"]), (width, height))
                    self.assertGreater(row["renderbuffer_id"], 0)
                    self.assertTrue(row["success"])
                    self.assertTrue(row["ordinary_storage_delegated"])
                self.assertNotEqual(pair["color"]["renderbuffer_id"],
                                    pair["depth"]["renderbuffer_id"])
                pair_signature = (pair["dimensions"],
                                  pair["color"]["renderbuffer_id"],
                                  pair["depth"]["renderbuffer_id"])
                if expected_call_counts[iteration] == 0:
                    self.assertEqual(metadata["allocation_calls"], ())
                    self.assertEqual(pair_signature, pair_signatures[-1],
                                     "same-size reuse must retain its verified allocation pair")
                else:
                    self.assertEqual(
                        {row["format_name"] for row in metadata["allocation_calls"]},
                        {"GL_RGBA", "GL_DEPTH_COMPONENT24"})
                    self.assertTrue(all(
                        row["samples"] == 4 and row["success"] and
                        row["ordinary_storage_delegated"] and
                        (row["width"], row["height"]) == (width, height)
                        for row in metadata["allocation_calls"]))
                    allocation_ids = {
                        row["format_name"]: row["renderbuffer_id"]
                        for row in metadata["allocation_calls"]}
                    self.assertEqual(allocation_ids["GL_RGBA"],
                                     pair["color"]["renderbuffer_id"])
                    self.assertEqual(allocation_ids["GL_DEPTH_COMPONENT24"],
                                     pair["depth"]["renderbuffer_id"])
                pair_signatures.append(pair_signature)
                self.assertEqual(runtime.render_calls[-1][3], 0)
                self.assertEqual(
                    set(metadata["source_identities"]),
                    {"quality_gotrack", "quality_render_stability",
                     "pyrender_renderer", "pyrender_offscreen"})
                for identity in metadata["source_identities"].values():
                    self.assertEqual(set(identity), {"module", "file", "sha256"})
                    self.assertRegex(identity["sha256"], r"^[0-9A-F]{64}$")
                snapshots.append(metadata)
                generations.append(metadata["allocation_generation"])

            self.assertEqual(generations, [1, 1, 2, 3, 4])
            self.assertEqual(len(runtime.original_storage_calls), 0)
            self.assertEqual(len(runtime.ordinary_storage_calls), 8)
            self.assertIs(renderer.pyrender.renderer.glRenderbufferStorageMultisample,
                          runtime.renderer_module.glRenderbufferStorageMultisample)
            with self.assertRaises(TypeError):
                snapshots[-1]["dimensions"] = (1, 1)
            self.assertEqual(snapshots[-1]["dimensions"], (280, 280))
            with self.assertRaises(AttributeError):
                renderer.render_policy = "legacy_v1"

            before_close = snapshots[-1]
            self.assertIsInstance(before_close["allocation_calls"], tuple)
            with self.assertRaises(TypeError):
                before_close["source_identities"]["quality_gotrack"]["sha256"] = "0" * 64
            with self.assertRaises(TypeError):
                before_close["allocation_calls"][0]["samples"] = 0
            renderer.close()
            self.assertIsNone(renderer.render_policy_metadata)
            recreated = renderer.render_object_model(
                "synthetic-object", _camera(280, 280))
            self.assertEqual(recreated[_RenderType.DEPTH].shape, (280, 280))
            new_metadata = renderer.render_policy_metadata
            self.assertEqual(new_metadata["allocation_generation"], 5)
            self.assertEqual(before_close["dimensions"], (280, 280))

    def test_live_offscreen_and_framebuffer_drift_invalidates_metadata_immediately(self):
        for drift in ("draw_fbo", "read_fbo", "dimensions", "replace", "clear"):
            with self.subTest(drift=drift):
                runtime = _RendererRuntime()
                with runtime:
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    renderer.render_object_model(
                        "synthetic-object", _camera(80, 72), return_tensors=False)
                    self.assertIsNotNone(renderer.render_policy_metadata)
                    verified_offscreen = renderer.offscreen
                    if drift == "draw_fbo":
                        verified_offscreen._renderer._main_fb_ms += 0.5
                    elif drift == "read_fbo":
                        verified_offscreen._renderer._main_fb += 0.5
                    elif drift == "dimensions":
                        verified_offscreen._renderer._main_fb_dims = (81, 72)
                    elif drift == "replace":
                        renderer.offscreen = SimpleNamespace()
                    else:
                        renderer.offscreen = None
                    self.assertIsNone(renderer.render_policy_metadata)

    def test_capture_storage_hook_is_restored_around_an_interleaved_legacy_instance(self):
        runtime = _RendererRuntime()
        with runtime:
            capture = _new_renderer(
                runtime, render_policy="capture_zero_sample_v1",
                coordinate_mode="integer_centers_v1")
            capture.render_object_model(
                "synthetic-object", _camera(80, 72), return_tensors=False)
            capture_metadata = capture.render_policy_metadata
            self.assertIsNotNone(capture_metadata)
            ordinary_count = len(runtime.ordinary_storage_calls)

            legacy = _new_renderer(runtime)
            legacy.render_object_model(
                "synthetic-object", _camera(48, 40), return_tensors=False)
            self.assertIsNone(legacy.render_policy_metadata)
            self.assertEqual(len(runtime.original_storage_calls), 2)
            self.assertEqual(len(runtime.ordinary_storage_calls), ordinary_count)
            self.assertIsNotNone(capture.render_policy_metadata)

            capture.render_object_model(
                "synthetic-object", _camera(80, 72), return_tensors=False)
            self.assertEqual(len(runtime.ordinary_storage_calls), ordinary_count)
            self.assertEqual(len(runtime.original_storage_calls), 2)
            self.assertEqual(capture.render_policy_metadata["allocation_generation"], 1)
            legacy.close()
            capture.close()

    def test_external_offscreen_replacement_and_clear_fail_closed_on_next_render(self):
        for mutation in ("replace", "clear"):
            with self.subTest(mutation=mutation):
                runtime = _RendererRuntime()
                with runtime:
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    renderer.render_object_model(
                        "synthetic-object", _camera(64, 48), return_tensors=False)
                    previous = renderer.offscreen
                    self.assertIsNotNone(renderer.render_policy_metadata)
                    runtime.trace_calls.clear()
                    runtime.tensor_calls.clear()
                    renders_before = len(runtime.render_calls)
                    if mutation == "replace":
                        renderer.offscreen = runtime.fake_pyrender.OffscreenRenderer(64, 48)
                    else:
                        renderer.offscreen = None
                    self.assertIsNone(renderer.render_policy_metadata)
                    with self.assertRaises((RuntimeError, ValueError)):
                        renderer.render_object_model(
                            "synthetic-object", _camera(64, 48), return_tensors=True)
                    self.assertIsNone(renderer.render_policy_metadata)
                    self.assertIsNone(renderer.offscreen)
                    self.assertEqual(runtime.trace_calls, [])
                    self.assertEqual(runtime.tensor_calls, [])
                    self.assertEqual(len(runtime.render_calls), renders_before)
                    self.assertTrue(previous._fake.deleted)
                    self.assertTrue(all(offscreen.deleted for offscreen in runtime.offscreens))

    def test_invalid_capture_preflight_clears_prior_metadata_without_render_or_output(self):
        for invalid_call in ("wrong_object", "oversized", "config_drift"):
            with self.subTest(invalid_call=invalid_call):
                runtime = _RendererRuntime()
                with runtime:
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    renderer.render_object_model(
                        "synthetic-object", _camera(64, 48), return_tensors=True)
                    self.assertIsNotNone(renderer.render_policy_metadata)
                    runtime.trace_calls.clear()
                    runtime.tensor_calls.clear()
                    renders_before = len(runtime.render_calls)
                    if invalid_call == "wrong_object":
                        object_id = "other-object"
                        camera = _camera(64, 48)
                    elif invalid_call == "oversized":
                        object_id = "synthetic-object"
                        camera = _camera(1121, 48)
                    else:
                        object_id = "synthetic-object"
                        camera = _camera(64, 48)
                        renderer.disable_multisampling = True
                    with self.assertRaises((TypeError, ValueError)):
                        renderer.render_object_model(
                            object_id, camera, return_tensors=True)
                    self.assertIsNone(renderer.render_policy_metadata)
                    self.assertEqual(len(runtime.render_calls), renders_before)
                    self.assertEqual(runtime.trace_calls, [])
                    self.assertEqual(runtime.tensor_calls, [])

    def test_capture_allocation_pairs_and_zero_sample_framebuffer_are_mandatory(self):
        allocation_cases = (
            ("missing_pair", None),
            ("same_id", None),
            ("extra_pair_call", None),
            ("wrong_allocation_dimensions", None),
            ("normal", "allocation"),
        )
        for allocation_style, failure in allocation_cases:
            with self.subTest(allocation_style=allocation_style, failure=failure):
                runtime = _RendererRuntime(
                    failure=failure, allocation_style=allocation_style)
                with runtime:
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    original_storage = runtime.renderer_module.glRenderbufferStorageMultisample
                    with self.assertRaises((TypeError, ValueError, RuntimeError, OSError)):
                        renderer.render_object_model(
                            "synthetic-object", _camera(64, 48), return_tensors=True)
                    self.assertIsNone(renderer.render_policy_metadata)
                    self.assertIsNone(renderer.offscreen)
                    self.assertEqual(runtime.trace_calls, [])
                    self.assertEqual(runtime.tensor_calls, [])
                    self.assertIs(
                        runtime.renderer_module.glRenderbufferStorageMultisample,
                        original_storage)

        verification_cases = (
            ("incomplete", lambda runtime: setattr(runtime.gl, "complete", False)),
            ("gl_samples", lambda runtime: setattr(runtime.gl, "forced_samples", 4)),
            ("gl_sample_buffers", lambda runtime: setattr(
                runtime.gl, "forced_sample_buffers", 1)),
            ("wrong_internal_dimensions", lambda runtime: setattr(
                runtime, "metadata_dimensions", (65, 48))),
            ("unsupported_framebuffer", lambda runtime: setattr(
                runtime, "framebuffer_supported", False)),
            ("draw_binding_query", lambda runtime: setattr(
                runtime.gl, "fail_query", _FakeGL.GL_DRAW_FRAMEBUFFER_BINDING)),
            ("read_binding_query", lambda runtime: setattr(
                runtime.gl, "fail_query", _FakeGL.GL_READ_FRAMEBUFFER_BINDING)),
            ("samples_query", lambda runtime: setattr(
                runtime.gl, "fail_query", _FakeGL.GL_SAMPLES)),
            ("sample_buffers_query", lambda runtime: setattr(
                runtime.gl, "fail_query", _FakeGL.GL_SAMPLE_BUFFERS)),
            ("framebuffer_status_query", lambda runtime: setattr(
                runtime.gl, "fail_status", True)),
            ("context_acquire", lambda runtime: setattr(
                runtime, "failure", "context_acquire")),
            ("late_context_acquire", lambda runtime: setattr(
                runtime, "context_acquire_fail_calls", {2})),
            ("fractional_fbo", lambda runtime: None),
            ("draw_restore", lambda runtime: setattr(
                runtime.gl, "fail_restore_target", _FakeGL.GL_DRAW_FRAMEBUFFER)),
            ("read_restore", lambda runtime: setattr(
                runtime.gl, "fail_restore_target", _FakeGL.GL_READ_FRAMEBUFFER)),
            ("context_release", lambda runtime: setattr(
                runtime, "failure", "context_release")),
        )
        for name, configure in verification_cases:
            with self.subTest(verification=name):
                runtime = _RendererRuntime(fractional_fbo=(name == "fractional_fbo"))
                configure(runtime)
                with runtime:
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    with self.assertRaises((TypeError, ValueError, RuntimeError)):
                        renderer.render_object_model(
                            "synthetic-object", _camera(64, 48), return_tensors=True)
                    self.assertIsNone(renderer.render_policy_metadata)
                    self.assertIsNone(renderer.offscreen)
                    self.assertEqual(runtime.trace_calls, [])
                    self.assertEqual(runtime.tensor_calls, [])
                    self.assertTrue(any(event[0] == "make_uncurrent"
                                        for event in runtime.events))
                    if name == "draw_restore":
                        self.assertIn(("bind_framebuffer", _FakeGL.GL_READ_FRAMEBUFFER, 702),
                                      runtime.gl.calls)
                    if name == "read_restore":
                        self.assertIn(("bind_framebuffer", _FakeGL.GL_DRAW_FRAMEBUFFER, 701),
                                      runtime.gl.calls)
                    if name == "late_context_acquire":
                        offscreen_id = runtime.offscreens[0].identifier
                        self.assertEqual(runtime.context_acquire_calls, 2)
                        failed_acquire = runtime.events.index((
                            "make_current_failed_after_state_change", offscreen_id, True))
                        release_indices = [index for index, event in enumerate(runtime.events)
                                           if event == ("make_uncurrent", offscreen_id)]
                        self.assertEqual(len(release_indices), 3)
                        self.assertEqual(runtime.events[failed_acquire + 1],
                                         ("make_uncurrent", offscreen_id))
                        first_scene_cleanup = next(
                            index for index, event in enumerate(runtime.events)
                            if event[0] == "scene_remove")
                        delete_index = next(
                            index for index, event in enumerate(runtime.events)
                            if event[0] == "delete")
                        self.assertLess(release_indices[0], first_scene_cleanup)
                        self.assertLess(first_scene_cleanup, release_indices[1])
                        self.assertLess(release_indices[1], delete_index)
                        self.assertLess(delete_index, release_indices[2])
                        self.assertFalse(runtime.offscreens[0]._platform.current)

    def test_capture_readback_and_delete_failures_still_release_context_and_suppress_trace(self):
        for failure in (("render", "delete"), ("depth_readback", "delete")):
            with self.subTest(failure=failure):
                runtime = _RendererRuntime(failure=failure)
                with runtime:
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    original_storage = runtime.renderer_module.glRenderbufferStorageMultisample
                    with self.assertRaises((RuntimeError, OSError)):
                        renderer.render_object_model(
                            "synthetic-object", _camera(64, 48), return_tensors=True)
                    self.assertIsNone(renderer.render_policy_metadata)
                    self.assertIsNone(renderer.offscreen)
                    self.assertEqual(runtime.trace_calls, [])
                    self.assertEqual(runtime.tensor_calls, [])
                    self.assertTrue(runtime.offscreens[0].deleted)
                    self.assertTrue(any(event[0] == "make_uncurrent"
                                        for event in runtime.events))
                    self.assertIs(
                        runtime.renderer_module.glRenderbufferStorageMultisample,
                        original_storage)

    def test_capture_primary_failure_preserves_all_independent_cleanup_failures(self):
        cases = (
            ("render", ("render", "delete"), {1, 2}, 2,
             "fake render failure"),
            ("depth_readback", ("depth_readback", "delete"), {2, 3}, 3,
             "fake readback conversion failure"),
        )
        for stage, failures, release_fail_calls, expected_release_count, primary_text in cases:
            with self.subTest(stage=stage):
                runtime = _RendererRuntime(failure=failures)
                runtime.context_release_fail_calls = release_fail_calls
                lock_spy = _RLockSpy()
                runtime.lock_spy = lock_spy
                with runtime, patch.object(stability, "_render_lock", lock_spy):
                    renderer = _new_renderer(
                        runtime, render_policy="capture_zero_sample_v1",
                        coordinate_mode="integer_centers_v1")
                    original_storage = runtime.renderer_module.glRenderbufferStorageMultisample
                    with self.assertRaisesRegex(RuntimeError, primary_text) as caught:
                        renderer.render_object_model(
                            "synthetic-object", _camera(64, 48), return_tensors=True)

                    message = str(caught.exception)
                    for cleanup_name in (
                            "release_before_delete", "delete_offscreen", "release_after_delete"):
                        self.assertIn(cleanup_name, message)
                    self.assertIn("fake context release failure", message)
                    self.assertIn("fake offscreen deletion failure", message)
                    self.assertIsNotNone(caught.exception.__cause__)
                    self.assertIn(primary_text, str(caught.exception.__cause__))

                    offscreen_id = runtime.offscreens[0].identifier
                    disposal_events = [
                        event for event in runtime.events
                        if event[0] in ("make_uncurrent", "delete")]
                    self.assertEqual(disposal_events[-3:], [
                        ("make_uncurrent", offscreen_id),
                        ("delete", offscreen_id),
                        ("make_uncurrent", offscreen_id),
                    ])
                    self.assertEqual(runtime.context_release_calls,
                                     expected_release_count)
                    self.assertEqual(
                        sum(event[0] == "make_uncurrent_failed_after_state_change"
                            for event in runtime.events), 2)
                    self.assertFalse(runtime.offscreens[0]._platform.current)
                    self.assertTrue(runtime.offscreens[0].deleted)
                    self.assertIsNone(renderer.offscreen)
                    self.assertIsNone(renderer.render_policy_metadata)
                    self.assertEqual(runtime.trace_calls, [])
                    self.assertEqual(runtime.tensor_calls, [])
                    self.assertIs(
                        runtime.renderer_module.glRenderbufferStorageMultisample,
                        original_storage)
                    self.assertEqual(lock_spy.depth, 0)
                    self.assertEqual(lock_spy.acquire_count, lock_spy.release_count)
                    self.assertGreaterEqual(lock_spy.max_depth, 2)

    def test_failed_scene_node_cleanup_poisoning_blocks_any_later_capture_render(self):
        runtime = _RendererRuntime(failure="scene_cleanup")
        with runtime:
            renderer = _new_renderer(
                runtime, render_policy="capture_zero_sample_v1",
                coordinate_mode="integer_centers_v1")
            with self.assertRaisesRegex(RuntimeError, "scene-node cleanup"):
                renderer.render_object_model(
                    "synthetic-object", _camera(64, 48), return_tensors=True)
            self.assertIsNone(renderer.render_policy_metadata)
            self.assertIsNone(renderer.offscreen)
            self.assertEqual(runtime.trace_calls, [])
            self.assertEqual(runtime.tensor_calls, [])
            self.assertEqual([node.kind for node in renderer.scene.nodes], ["mesh"])
            self.assertTrue(any(event[0] == "delete" for event in runtime.events))

            runtime.failure = None
            renders_before = len(runtime.render_calls)
            with self.assertRaisesRegex(RuntimeError, "poison|scene integrity"):
                renderer.render_object_model(
                    "synthetic-object", _camera(64, 48), return_tensors=True)
            self.assertEqual(len(runtime.render_calls), renders_before)
            self.assertEqual(runtime.trace_calls, [])
            self.assertEqual(runtime.tensor_calls, [])

    def test_persistent_scene_cleanup_and_retry_failure_leave_nodes_and_poison_instance(self):
        runtime = _RendererRuntime(failure="scene_cleanup_persistent")
        lock_spy = _RLockSpy()
        runtime.lock_spy = lock_spy
        with runtime, patch.object(stability, "_render_lock", lock_spy):
            renderer = _new_renderer(
                runtime, render_policy="capture_zero_sample_v1",
                coordinate_mode="integer_centers_v1")
            original_storage = runtime.renderer_module.glRenderbufferStorageMultisample
            with self.assertRaisesRegex(RuntimeError, "scene-node cleanup"):
                renderer.render_object_model(
                    "synthetic-object", _camera(64, 48), return_tensors=True)

            remove_attempts = [event for event in runtime.events
                               if event[0] == "scene_remove"]
            self.assertEqual(len(remove_attempts), 4)
            self.assertCountEqual([node.kind for node in renderer.scene.nodes],
                                  ["mesh", "camera", "light"])
            self.assertTrue(renderer._capture_scene_poisoned)
            self.assertIsNone(renderer.offscreen)
            self.assertIsNone(renderer.render_policy_metadata)
            self.assertTrue(runtime.offscreens[0].deleted)
            self.assertEqual(runtime.trace_calls, [])
            self.assertEqual(runtime.tensor_calls, [])
            self.assertIs(
                runtime.renderer_module.glRenderbufferStorageMultisample,
                original_storage)
            self.assertEqual(lock_spy.depth, 0)
            self.assertEqual(lock_spy.acquire_count, lock_spy.release_count)

            runtime.failure = None
            events_before = tuple(runtime.events)
            renders_before = tuple(runtime.render_calls)
            with self.assertRaisesRegex(RuntimeError, "scene integrity"):
                renderer.render_object_model(
                    "synthetic-object", _camera(64, 48), return_tensors=True)
            self.assertEqual(tuple(runtime.events), events_before)
            self.assertEqual(tuple(runtime.render_calls), renders_before)
            self.assertEqual(runtime.trace_calls, [])
            self.assertEqual(runtime.tensor_calls, [])
            self.assertEqual(lock_spy.depth, 0)
            self.assertEqual(lock_spy.acquire_count, lock_spy.release_count)

    def test_public_render_resize_close_and_nested_hooks_hold_shared_reentrant_lock(self):
        lock_spy = _RLockSpy()
        runtime = _RendererRuntime()
        runtime.lock_spy = lock_spy

        def capture_segment(action):
            start = len(lock_spy.events)
            action()
            return lock_spy.events[start:]

        def operation_depths(events, name):
            return [event[2] for event in events
                    if event[0] == "operation" and event[1] == name]

        with runtime, patch.object(stability, "_render_lock", lock_spy):
            capture = _new_renderer(
                runtime, render_policy="capture_zero_sample_v1",
                coordinate_mode="integer_centers_v1")
            fresh = capture_segment(lambda: capture.render_object_model(
                "synthetic-object", _camera(80, 72), return_tensors=False))
            self.assertIn(2, operation_depths(fresh, "ordinary_storage"))
            self.assertIn(2, operation_depths(fresh, "gl_integer_query"))
            self.assertIn(2, operation_depths(fresh, "offscreen_render"))
            self.assertIn(1, operation_depths(fresh, "scene_remove"))
            self.assertEqual(lock_spy.depth, 0)

            resized = capture_segment(lambda: capture.render_object_model(
                "synthetic-object", _camera(720, 720), return_tensors=False))
            self.assertIn(1, operation_depths(resized, "offscreen_delete"))
            self.assertIn(2, operation_depths(resized, "ordinary_storage"))
            self.assertIn(2, operation_depths(resized, "gl_integer_query"))
            self.assertIn(2, operation_depths(resized, "offscreen_render"))
            self.assertEqual(lock_spy.depth, 0)

            capture_close = capture_segment(capture.close)
            self.assertIn(1, operation_depths(capture_close, "offscreen_delete"))
            self.assertIn(1, operation_depths(capture_close, "context_make_uncurrent"))
            self.assertEqual(lock_spy.depth, 0)

            legacy = _new_renderer(runtime, disable_multisampling=True)
            legacy_render = capture_segment(lambda: legacy.render_object_model(
                "synthetic-object", _camera(48, 40), return_tensors=False))
            self.assertIn(1, operation_depths(legacy_render, "multisample_storage"))
            self.assertIn(2, operation_depths(legacy_render, "offscreen_render"))
            self.assertIn(2, operation_depths(legacy_render, "gl_disable"))
            self.assertEqual(lock_spy.depth, 0)

            legacy_resize = capture_segment(lambda: legacy.render_object_model(
                "synthetic-object", _camera(96, 64), return_tensors=False))
            self.assertIn(1, operation_depths(legacy_resize, "offscreen_delete"))
            self.assertIn(1, operation_depths(legacy_resize, "multisample_storage"))
            self.assertIn(2, operation_depths(legacy_resize, "offscreen_render"))
            self.assertEqual(lock_spy.depth, 0)

            legacy_close = capture_segment(legacy.close)
            self.assertIn(1, operation_depths(legacy_close, "offscreen_delete"))
            self.assertEqual(lock_spy.depth, 0)

            public_segments = (
                fresh, resized, capture_close, legacy_render, legacy_resize, legacy_close)
            operation_events = [event for segment in public_segments for event in segment
                                if event[0] == "operation"]
            self.assertTrue(operation_events)
            self.assertTrue(all(event[2] >= 1 for event in operation_events))
            self.assertGreaterEqual(lock_spy.max_depth, 2)
            self.assertEqual(lock_spy.depth, 0)
            self.assertEqual(lock_spy.acquire_count, lock_spy.release_count)

    def test_capture_storage_scope_rejects_fractional_gl_integers_before_delegation(self):
        runtime = _RendererRuntime()
        original = runtime.renderer_module.glRenderbufferStorageMultisample
        good_dimensions = (64, 48)
        good_calls = []
        with stability.capture_storage_scope(
                runtime.renderer_module, runtime.fake_gl_module,
                dimensions=good_dimensions, allocation_calls=good_calls):
            for offset, internal_format in enumerate(
                    (_FakeGL.GL_RGBA, _FakeGL.GL_DEPTH_COMPONENT24)):
                runtime.gl.renderbuffer_binding = np.int64(23 + offset)
                runtime.renderer_module.glRenderbufferStorageMultisample(
                    np.int64(_FakeGL.GL_RENDERBUFFER), np.int64(4),
                    np.int64(internal_format), np.int64(64), np.int64(48))
        self.assertIs(runtime.renderer_module.glRenderbufferStorageMultisample, original)
        self.assertEqual(len(runtime.ordinary_storage_calls), 2)
        self.assertTrue(stability.validate_capture_allocation_pairs(
            good_calls, dimensions=good_dimensions)["passed"])

        for bad_samples, bad_width, bound_id in (
                (4.5, 64, 23), (4, 64.5, 23), (4, 64, 23.5)):
            with self.subTest(samples=bad_samples, width=bad_width, bound_id=bound_id):
                runtime.gl.renderbuffer_binding = bound_id
                before = len(runtime.ordinary_storage_calls)
                call_log = []
                with stability.capture_storage_scope(
                        runtime.renderer_module, runtime.fake_gl_module,
                        dimensions=good_dimensions, allocation_calls=call_log):
                    with self.assertRaises((TypeError, ValueError)):
                        runtime.renderer_module.glRenderbufferStorageMultisample(
                            _FakeGL.GL_RENDERBUFFER, bad_samples, _FakeGL.GL_RGBA,
                            bad_width, 48)
                self.assertEqual(len(runtime.ordinary_storage_calls), before)
                self.assertEqual(call_log[0]["ordinary_storage_delegated"], False)
                self.assertEqual(call_log[0]["success"], False)
                self.assertIs(runtime.renderer_module.glRenderbufferStorageMultisample, original)

    def test_capture_refiner_gate_requires_policy_and_accepts_capture_renderer(self):
        harness = refiner_tests._Harness(solve_success=False)
        harness.renderer.render_policy = "legacy_v1"
        refiner = harness.make_refiner()
        with harness.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None):
            with self.assertRaisesRegex(ValueError, "capture_zero_sample_v1"):
                refiner.refine(
                    refiner_tests._capture_frame(),
                    np.ones((720, 720), dtype=np.bool_), np.eye(4))
        self.assertEqual(harness.events, [])
        self.assertEqual(harness.torch_imports, 0)
        self.assertEqual(harness.utils_imports, 0)
        self.assertEqual(harness.network_calls, 0)
        self.assertEqual(harness.warp_calls, 0)

        accepted = refiner_tests._Harness(solve_success=False)
        accepted_refiner = accepted.make_refiner()
        with accepted.guarded_imports(), patch.object(
                gotrack, "trace", lambda *_args, **_kwargs: None), patch.object(
                gotrack.cv2, "solvePnPRefineLM", accepted.solve_lm):
            result = accepted_refiner.refine(
                refiner_tests._capture_frame(),
                np.ones((720, 720), dtype=np.bool_), np.eye(4))
        self.assertIsNone(result)
        self.assertGreater(accepted.warp_calls, 0)
        self.assertGreater(accepted.network_calls, 0)
        self.assertGreater(accepted.pnp_calls, 0)
