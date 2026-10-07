"""Renderer boundary checks without a GL context, models or image files.

These test the arguments passed to the renderer. Actual rasterization remains
the separate slanted-plane control's responsibility.
"""
import inspect
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .quality_gotrack import TexturedRenderer


class Scene:
    def __init__(self):
        self.added = []
        self.removed = []

    def add(self, item, pose):
        node = len(self.added)
        self.added.append((item, np.array(pose, copy=True)))
        return node

    def remove_node(self, node):
        self.removed.append(node)


def fixture(mode='integer_centers_v1', fail=False):
    renderer = TexturedRenderer.__new__(TexturedRenderer)
    renderer.obj_id = 8
    renderer.coordinate_mode = mode
    renderer.unlit = False
    renderer.disable_multisampling = False
    renderer.scene = Scene()
    cameras = []

    def camera(*args, **kwargs):
        item = SimpleNamespace(args=args, kwargs=kwargs)
        cameras.append(item)
        return item

    def render(scene, flags):
        if fail:
            raise RuntimeError('synthetic render failure')
        return (np.full((48, 64, 3), 128, np.uint8),
                np.full((48, 64), 1.25, np.float32))

    renderer.offscreen = SimpleNamespace(viewport_width=64, viewport_height=48,
                                         render=render)
    renderer.pyrender = SimpleNamespace(
        IntrinsicsCamera=camera, SpotLight=lambda **kw: SimpleNamespace(**kw),
        RenderFlags=SimpleNamespace(FLAT=1, NONE=0))
    transform = np.eye(4)
    transform[:3, 3] = [100., -200., 1250.]
    model = SimpleNamespace(width=64, height=48, f=(110., 130.),
                            c=(23.25, 18.75), T_world_from_eye=transform)
    module = ModuleType('utils.renderer_base')
    module.RenderType = SimpleNamespace(COLOR='color', DEPTH='depth', MASK='mask')
    return renderer, model, cameras, module


class CaptureProjectionTests(unittest.TestCase):
    def run_renderer(self, renderer, model, module):
        with patch.dict('sys.modules', {'utils.renderer_base': module}), \
                patch('bench.quality_gotrack.trace'):
            return renderer.render_object_model(8, model)

    def test_capture_boundary_shifts_both_principals_once_and_legacy_is_unchanged(self):
        for mode, offset in [('integer_centers_v1', .5), ('legacy', 0.)]:
            with self.subTest(mode=mode):
                renderer, model, cameras, module = fixture(mode)
                self.run_renderer(renderer, model, module)
                self.assertEqual(cameras[0].args,
                                 (110., 130., 23.25 + offset, 18.75 + offset))
                self.assertEqual(model.c, (23.25, 18.75))

    def test_camera_translation_is_converted_once_and_cv_axes_are_flipped(self):
        renderer, model, cameras, module = fixture()
        original = model.T_world_from_eye.copy()
        output = self.run_renderer(renderer, model, module)
        expected = np.diag([1., -1., -1., 1.])
        expected[:3, 3] = [.1, -.2, 1.25]
        for _, pose in renderer.scene.added:
            np.testing.assert_allclose(pose, expected, atol=1e-15, rtol=0.)
        np.testing.assert_array_equal(model.T_world_from_eye, original)
        np.testing.assert_array_equal(output['depth'], np.full((48, 64), 1250.))
        self.assertTrue(output['mask'].all())
        self.assertEqual(renderer.scene.removed, [0, 1])

    def test_integer_rays_land_on_gl_fragment_centers_for_asymmetric_camera(self):
        renderer, model, cameras, module = fixture()
        self.run_renderer(renderer, model, module)
        fx, fy, cx, cy = cameras[0].args
        pixels = np.array([[0., 0.], [13., 7.], [31., 22.], [63., 47.]])
        rays = np.column_stack(((pixels[:, 0] - model.c[0]) / model.f[0],
                                (pixels[:, 1] - model.c[1]) / model.f[1],
                                np.ones(len(pixels))))
        # GL viewport coordinates denote pixel edges; fragment centers are +.5.
        gl = rays * [1., -1., -1.]
        ndc = np.column_stack((2 * fx / model.width * gl[:, 0] / -gl[:, 2]
                               - (1 - 2 * cx / model.width),
                               2 * fy / model.height * gl[:, 1] / -gl[:, 2]
                               - (2 * cy / model.height - 1)))
        viewport = np.column_stack(((ndc[:, 0] + 1) * model.width / 2,
                                     (1 - ndc[:, 1]) * model.height / 2))
        np.testing.assert_allclose(viewport, pixels + .5, atol=1e-13, rtol=0.)

    def test_render_failure_still_removes_temporary_camera_and_light(self):
        renderer, model, cameras, module = fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'synthetic render failure'):
            self.run_renderer(renderer, model, module)
        self.assertEqual(renderer.scene.removed, [0, 1])

    def test_legacy_default_and_bad_mode_rejection_precede_optional_imports(self):
        self.assertEqual(inspect.signature(TexturedRenderer).parameters[
            'coordinate_mode'].default, 'legacy')
        with patch('bench.quality_gotrack.upstream_path',
                   side_effect=AssertionError('upstream touched')):
            for mode in ('invalid', None, True):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    TexturedRenderer('never-open.glb', 8, coordinate_mode=mode)


if __name__ == '__main__':
    unittest.main()
