import unittest
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
from unittest.mock import Mock, patch
from .quality_pnp_guess import without_extrinsic_guess
from .quality_gotrack import GoTrackRefiner
from . import quality_gotrack, quality_runner
from .quality_runner import _make_gotrack_refiner, _pnp_tracking_settings
from .vision import cv2


class ScopedPnPTests(unittest.TestCase):
    def test_option_is_overridden_without_changing_other_solver_parameters(self):
        with patch.object(cv2,'solvePnPRansac',return_value='solved') as solver:
            with without_extrinsic_guess():
                self.assertEqual(cv2.solvePnPRansac('xyz','uv',useExtrinsicGuess=True,reprojectionError=2.),'solved')
            self.assertIs(cv2.solvePnPRansac,solver)
            self.assertEqual(solver.call_args.kwargs,dict(useExtrinsicGuess=False,reprojectionError=2.))

    def test_solver_restores_after_failure_and_nested_context(self):
        original=cv2.solvePnPRansac
        with self.assertRaises(RuntimeError):
            with without_extrinsic_guess():
                outer=cv2.solvePnPRansac
                with without_extrinsic_guess():pass
                self.assertIs(cv2.solvePnPRansac,outer)
                raise RuntimeError('test')
        self.assertIs(cv2.solvePnPRansac,original)


class GoTrackPnPGuessTests(unittest.TestCase):
    def _assert_solver_forwarding(self, expected_guess, **constructor_kwargs):
        refiner = GoTrackRefiner(None, None, 1, 'cpu', **constructor_kwargs)
        obj_points = object(); target_points = object(); crop_k = object()
        rvec = object(); tvec = object(); result = (True, rvec, tvec, 'inliers')
        with patch.object(cv2, 'solvePnPRansac', return_value=result) as solver:
            self.assertIs(refiner._solve_pnp_ransac(obj_points, target_points, crop_k, rvec, tvec), result)
        self.assertEqual(solver.call_args.args, (obj_points, target_points, crop_k, None))
        self.assertIs(solver.call_args.kwargs['rvec'], rvec)
        self.assertIs(solver.call_args.kwargs['tvec'], tvec)
        self.assertEqual(solver.call_args.kwargs, {
            'rvec': rvec,
            'tvec': tvec,
            'useExtrinsicGuess': expected_guess,
            'iterationsCount': 3000,
            'reprojectionError': 2.,
            'confidence': .999,
            'flags': cv2.SOLVEPNP_ITERATIVE,
        })

    def test_default_setting_forwards_true_to_ransac(self):
        self._assert_solver_forwarding(True)

    def test_opt_in_setting_forwards_false_with_existing_solver_settings(self):
        self._assert_solver_forwarding(False, pnp_use_extrinsic_guess=False)

    def test_refiner_rejects_non_boolean_setting(self):
        with self.assertRaisesRegex(TypeError, 'must be a bool'):
            GoTrackRefiner(None, None, 1, 'cpu', pnp_use_extrinsic_guess=0)

    def test_option_is_mug_only_and_effective_value_is_serialized(self):
        for bundle_object in ('ranch', 'keyboard'):
            with self.subTest(bundle_object=bundle_object):
                with self.assertRaisesRegex(ValueError, 'only for the mug bundle'):
                    _pnp_tracking_settings(bundle_object, True)
        settings = _pnp_tracking_settings('mug', True)
        self.assertEqual(json.loads(json.dumps(settings)), {'pnp_use_extrinsic_guess': False})

    def test_default_tracking_metadata_records_true(self):
        self.assertEqual(_pnp_tracking_settings('keyboard', False), {'pnp_use_extrinsic_guess': True})

    def test_cli_opt_in_reaches_refiner_and_serialized_tracking_setting(self):
        cudnn = SimpleNamespace(deterministic=False, benchmark=False, allow_tf32=True)
        cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True),
                               enable_flash_sdp=lambda enabled: None,
                               enable_mem_efficient_sdp=lambda enabled: None)
        fake_torch = SimpleNamespace(manual_seed=lambda seed: None,
            set_num_threads=lambda count: None,
            use_deterministic_algorithms=lambda enabled: None,
            backends=SimpleNamespace(cudnn=cudnn, cuda=cuda))
        fake_cache = SimpleNamespace(rglob=lambda pattern: [])
        with patch.object(sys, 'argv', ['quality_runner', 'pose', '--bundle', 'mug-bundle',
                                        '--masks', 'masks', '--output', 'out.json', '--device', 'cpu',
                                        '--pnp-no-extrinsic-guess']), \
             patch.dict(sys.modules, {'torch': fake_torch}), \
             patch.object(quality_runner, 'read_input', return_value={'object': 'mug'}), \
             patch.object(quality_runner, 'device_info', return_value={}), \
             patch.object(quality_runner, 'pose_stage') as pose_stage, \
             patch.object(quality_runner, 'CACHE', fake_cache):
            quality_runner.main()

        self.assertTrue(pose_stage.call_args.args[-1])
        pnp_settings = _pnp_tracking_settings('mug', pose_stage.call_args.args[-1])
        refiner = _make_gotrack_refiner(None, None, 1, 'cpu', pnp_settings)
        self.assertFalse(refiner.pnp_use_extrinsic_guess)
        self.assertEqual(json.loads(json.dumps(pnp_settings)), {'pnp_use_extrinsic_guess': False})

    def test_cli_rejects_non_mug_before_importing_torch(self):
        import builtins
        real_import = builtins.__import__
        attempted_torch_import = []

        def guarded_import(name, *args, **kwargs):
            if name == 'torch':
                attempted_torch_import.append(name)
                raise AssertionError('Torch should not load before mug-only option validation')
            return real_import(name, *args, **kwargs)

        with patch.object(sys, 'argv', ['quality_runner', 'pose', '--bundle', 'keyboard-bundle',
                                        '--masks', 'masks', '--output', 'out.json', '--device', 'cpu',
                                        '--pnp-no-extrinsic-guess']), \
             patch.dict(sys.modules):
            sys.modules.pop('torch', None)
            with patch('builtins.__import__', side_effect=guarded_import), \
                 patch.object(quality_runner, 'read_input', return_value={'object': 'keyboard'}):
                with self.assertRaises(SystemExit):
                    quality_runner.main()

        self.assertEqual(attempted_torch_import, [])

    def test_smoke_resolves_and_constructs_default_gotrack_refiner(self):
        height, width = 280, 373
        render_type = SimpleNamespace(MASK='mask', DEPTH='depth')
        vertices = np.array([[0., 0., 0.], [.1, .1, .1]])
        mask = np.ones((height, width), dtype=bool)
        rendered = {render_type.MASK: mask, render_type.DEPTH: np.full((height, width), 1000.)}
        renderer = SimpleNamespace(vertices_m=vertices,
            render_object_model=lambda *args, **kwargs: rendered,
            close=lambda: None)
        control = SimpleNamespace(triangle=np.ones((height, width), dtype=int),
                                  depth=np.ones((height, width)))
        camera_type = lambda **kwargs: SimpleNamespace(**kwargs)
        upstream_utils = ModuleType('utils')
        upstream_utils.structs = SimpleNamespace(PinholePlaneCameraModel=camera_type)
        upstream_utils.renderer_base = SimpleNamespace(RenderType=render_type)
        manifest = {'asset': 'mug.glb', 'object': 'mug', 'object_id': 1,
                    'controlled_initial_pose': [[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 1.], [0, 0, 0, 1.]],
                    'intrinsics': [[500., 0, 160.], [0, 500., 120.], [0, 0, 1.]],
                    'native_resolution': [320, 240], 'setup_frame_id': 0}
        fake_modules = {
            'utils': upstream_utils,
            'bench.glb_model': ModuleType('bench.glb_model'),
            'bench.show3d_experiment': ModuleType('bench.show3d_experiment'),
            'bench.renderer': ModuleType('bench.renderer'),
        }
        fake_modules['bench.glb_model'].read_glb = lambda *args: SimpleNamespace(positions=vertices)
        fake_modules['bench.show3d_experiment'].camera_for = lambda *args: object()
        fake_modules['bench.renderer'].render = lambda *args: control
        network = object()
        refine = Mock(return_value=object())
        refiner = SimpleNamespace(refine=refine)
        with patch.dict(sys.modules, fake_modules), \
             patch.object(quality_runner, 'read_input', return_value=manifest), \
             patch.object(quality_runner, 'read_rgb', return_value=np.zeros((240, 320, 3), dtype=np.uint8)), \
             patch.object(quality_runner, 'device_info', return_value={}), \
             patch.object(quality_gotrack, 'load_network', return_value=network), \
             patch.object(quality_gotrack, 'TexturedRenderer', return_value=renderer), \
             patch.object(quality_gotrack, 'GoTrackRefiner', return_value=refiner) as refiner_factory, \
             patch.object(Path, 'write_text'):
            quality_runner.smoke(Path('mug-bundle'), Path('smoke.json'), 'cpu')

        refiner_factory.assert_called_once_with(network, renderer, 1, 'cpu')
        refine.assert_called_once()
