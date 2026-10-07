"""Independent inert tests for the model-smoke worker's public observation seams.

This draft targets the integrated public ``bench.quality_runner`` module.  All
renderer and network doubles below are local NumPy fixtures; these tests do not
load optional model packages or issue canonical resource receipts.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from bench import quality_runner as runner
from bench import quality_assets as assets
from bench import test_quality_capture_resource_proof as proof_fixtures


class _RenderType(Enum):
    COLOR = 'rgb'
    DEPTH = 'depth'
    MASK = 'mask'


_NATIVE_TYPES = [_RenderType.COLOR, _RenderType.DEPTH, _RenderType.MASK]
_CROP_TYPES = [_RenderType.COLOR, _RenderType.MASK, _RenderType.DEPTH]


class _Camera:
    def __init__(self, width, height, *, native=False):
        self.width = width
        self.height = height
        self.f = (360., 360.) if native else (143., 139.)
        self.c = (359.5, 359.5) if native else (139.5, 140.5)
        self.T_world_from_eye = np.eye(4, dtype=np.float64)


class _MutableRenderer:
    """One inert public-shaped renderer that reuses writable output buffers."""

    def __init__(self, *, vary_color=True, fail_close=False):
        self.vertices_m = np.asarray(
            [[-.2, -.1, 0.], [.3, -.1, 0.], [-.2, .4, .1]], dtype=np.float32)
        self.obj_id = 8
        self.render_policy = 'capture_zero_sample_v1'
        self.coordinate_mode = 'integer_centers_v1'
        self.unlit = False
        self.disable_multisampling = False
        self.vary_color = vary_color
        self.fail_close = fail_close
        self.close_calls = 0
        self.render_calls = []
        self.render_count = 0
        self.last_result = None
        self.last_arrays = None
        self._buffers = {}
        self._metadata = {'call': 0, 'nested': {'call': 0}}

    @property
    def render_policy_metadata(self):
        return self._metadata

    def _buffer_set(self, width, height):
        key = (height, width)
        if key not in self._buffers:
            self._buffers[key] = (
                np.empty((height, width, 3), dtype=np.float32),
                np.empty((height, width), dtype=np.float32),
                np.empty((height, width), dtype=np.bool_),
            )
        return self._buffers[key]

    def render_object_model(self, obj_id, camera, render_types=None,
                            return_tensors=False, background=None, **kwargs):
        self.render_count += 1
        color, depth, mask = self._buffer_set(camera.width, camera.height)
        gray_code = 128 + ((self.render_count - 1) % 80) if self.vary_color else 128
        color.fill(np.float32(gray_code / 255.))
        depth.fill(np.float32(1000. + self.render_count))
        mask[...] = depth > 0.
        self._metadata['call'] = self.render_count
        self._metadata['nested']['call'] = self.render_count
        result = {
            _RenderType.COLOR: color,
            _RenderType.DEPTH: depth,
            _RenderType.MASK: mask,
        }
        self.last_result = result
        self.last_arrays = {'color': color, 'depth': depth, 'mask': mask}
        self.render_calls.append({
            'obj_id': obj_id,
            'camera': camera,
            'render_types': render_types,
            'return_tensors': return_tensors,
            'background': background,
            'kwargs': dict(kwargs),
        })
        return result

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError('synthetic close failure')
        self._metadata = None


class _RecordingNetwork:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self.outputs = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.fail:
            raise RuntimeError('synthetic network failure')
        result = {'synthetic_call': len(self.calls)}
        self.outputs.append(result)
        return result


def _event_sink(events, observations=None):
    def sink(observation):
        if observations is not None:
            observations.append(observation)
        event = {
            'case_id': observation['case_id'],
            'ordinal': observation['ordinal'],
            'role': observation['role'],
            'iteration': observation['iteration'],
            'network_inputs': None,
        }
        events.append(event)
        return event
    return sink


def _synthetic_runtime_sources(root):
    """Create inert physical source files with runtime closure aliases."""
    root = Path(root)
    runner_file = root / 'bench' / 'quality_runner.py'
    runner_file.parent.mkdir(parents=True, exist_ok=True)
    runner_file.write_bytes(b'# inert runner path fixture\n')
    specs = {
        'quality_gotrack': (
            'bench.quality_gotrack', 'bench/quality_gotrack.py',
            root / 'bench' / 'quality_gotrack.py'),
        'quality_render_stability': (
            'bench.quality_render_stability', 'bench/quality_render_stability.py',
            root / 'bench' / 'quality_render_stability.py'),
        'pyrender_renderer': (
            'pyrender.renderer', 'runtime/pyrender/renderer.py',
            root / '.cache' / 'quality-windows' / 'Lib' / 'site-packages' /
            'pyrender' / 'renderer.py'),
        'pyrender_offscreen': (
            'pyrender.offscreen', 'runtime/pyrender/offscreen.py',
            root / '.cache' / 'quality-windows' / 'Lib' / 'site-packages' /
            'pyrender' / 'offscreen.py'),
    }
    source_closure = {}
    identities = {}
    for index, (name, (module, closure_path, physical_path)) in enumerate(specs.items()):
        physical_path.parent.mkdir(parents=True, exist_ok=True)
        source_bytes = f'inert source fixture {index}\n'.encode('ascii')
        physical_path.write_bytes(source_bytes)
        source_sha = hashlib.sha256(source_bytes).hexdigest().upper()
        source_closure[closure_path] = source_sha
        identities[name] = {
            'module': module,
            'file': str(physical_path.resolve()),
            'sha256': source_sha,
        }
    return runner_file, source_closure, {'source_identities': identities}


def _synthetic_preflight_inputs():
    """Build a fixture-only issued-shape plan, recipe, resources and provenance."""
    object_name = 'hot3d_obj_000008'
    asset_sha = 'B' * 64
    receipt_sha = 'C' * 64
    unit_sha = 'D' * 64
    proof_sha = 'E' * 64
    output_camera_sha = 'F' * 64
    resource_key = 'A' * 64
    effective_flags = {'unlit_templates': False, 'disable_multisampling': False}
    stage_flags = {**effective_flags, 'grayscale': False}
    intrinsics = [[360., 0., 359.5], [0., 360., 359.5], [0., 0., 1.]]
    source_closure = {'bench/quality_runner.py': '9' * 64}
    recipe = {
        'schema': 1,
        'resource': 'smoke',
        'resource_kind': 'quality-capture-model-smoke-v1',
        'object_id': 8,
        'object_name': object_name,
        'asset_sha256': asset_sha,
        'asset_receipt_sha256': receipt_sha,
        'unit_receipt_sha256': unit_sha,
        'renderer_proof_sha256': proof_sha,
        'output_camera_sha256': output_camera_sha,
        'coordinate_mode': 'integer_centers_v1',
        'rendering_contract': {
            'model_smoke_views': [
                {'case_id': 'a', 'euler_xyz_degrees': [9., -13., 7.],
                 'offset_extent_units': [.02, -.04, 3.]},
                {'case_id': 'b', 'euler_xyz_degrees': [-19., 24., -12.],
                 'offset_extent_units': [-.17, .11, 3.15]},
                {'case_id': 'c', 'euler_xyz_degrees': [27., 15., 21.],
                 'offset_extent_units': [.15, -.126, 3.3]},
            ],
            'model_smoke_rotation': 'Rx@Ry@Rz@diag(1,-1,-1)',
            'model_smoke_native_intrinsics': intrinsics,
            'model_smoke_observation': 'public_return_v1',
            'model_smoke_array_digest': 'quality-model-smoke-array-v1',
        },
        'render_flags': effective_flags,
        'producer_device': 'cpu',
        'settings': {
            'synthetic_pose_count': 3,
            'minimum_iou': .94,
            'maximum_median_common_depth_error_m': .002,
        },
    }
    sidecar_path = Path(runner.CAPTURE_CACHE) / resource_key / 'resource.json'
    output = SimpleNamespace(
        purpose='smoke', resource_kind='quality-capture-model-smoke-v1',
        descriptor=None, resource_key=resource_key, sidecar_path=sidecar_path,
        recipe=MappingProxyType(recipe),
        source_closure=MappingProxyType(source_closure),
    )
    asset_receipt = SimpleNamespace(
        object_id=8, object_name=object_name, asset_sha256=asset_sha,
        receipt_sha256=receipt_sha, renderer_proof_sha256=proof_sha,
    )
    plan_manifest = {
        'object': object_name, 'object_id': 8, 'units': 'metres',
        'intrinsics': intrinsics,
    }
    plan = SimpleNamespace(manifest=plan_manifest, bundle=Path('synthetic-bundle'))
    resources = SimpleNamespace(
        stage='smoke', device='cpu', required=(), output=output,
        render_flags=MappingProxyType(stage_flags), asset_receipt=asset_receipt,
    )
    provenance = {
        'capture_provenance': {
            'stage': 'smoke', 'render_flags': stage_flags, 'used_resources': [],
            'output_resource': {
                'resource_kind': output.resource_kind,
                'resource_key': output.resource_key,
                'sidecar_path': output.sidecar_path.relative_to(
                    Path(runner.CAPTURE_CACHE)).as_posix(),
                'recipe': recipe,
                'producer_source_closure': source_closure,
            },
            'asset_receipt_sha256': receipt_sha,
            'renderer_proof_sha256': proof_sha,
        },
    }
    return plan, resources, provenance, stage_flags


def _render(observer, renderer, *, case_id='a', ordinal=0, role='crop'):
    if role == 'native':
        camera = _Camera(720, 720, native=True)
        render_types = list(_NATIVE_TYPES)
        return_tensors = False
        background = None
    else:
        camera = _Camera(280, 280)
        render_types = list(_CROP_TYPES)
        return_tensors = True
        background = [.5, .5, .5]
    returned = observer.render_object_model(
        renderer.obj_id, camera, render_types=render_types,
        return_tensors=return_tensors, background=background)
    return returned, camera, render_types, background


class ModelSmokeObserverTests(unittest.TestCase):
    def test_reused_buffers_and_nested_metadata_are_snapshotted_and_delegate_result_is_exact(self):
        renderer = _MutableRenderer(vary_color=True)
        observations = []
        observer = runner.ModelSmokeRenderObserver(
            renderer, lambda value: observations.append(value) or value)

        for ordinal, case_id in enumerate(('a', 'b', 'c')):
            camera = _Camera(720, 720, native=True)
            render_types = list(_NATIVE_TYPES)
            with observer.observe_native(case_id, ordinal):
                returned = observer.render_object_model(
                    renderer.obj_id, camera, render_types=render_types,
                    return_tensors=False, background=None)
            self.assertIs(returned, renderer.last_result)
            call = renderer.render_calls[-1]
            self.assertEqual(call['obj_id'], 8)
            self.assertIs(call['camera'], camera)
            self.assertIs(call['render_types'], render_types)
            self.assertIs(call['return_tensors'], False)
            self.assertIsNone(call['background'])
            self.assertEqual(call['kwargs'], {})

        first = observations[0]
        first_color = first['arrays']['color_norm_f32']
        self.assertEqual(float(first_color[0, 0, 0]), float(np.float32(128. / 255.)))
        self.assertFalse(first_color.flags.writeable)
        with self.assertRaises(ValueError):
            first_color[0, 0, 0] = np.float32(0.)
        self.assertEqual(first['render_policy_metadata']['nested']['call'], 1)
        with self.assertRaises(TypeError):
            first['render_policy_metadata']['nested']['call'] = 99
        self.assertNotEqual(
            float(renderer.last_arrays['color'][0, 0, 0]),
            float(first_color[0, 0, 0]))
        self.assertEqual(len(renderer.render_calls), 3)

        observer.close()
        observer.close()
        self.assertEqual(renderer.close_calls, 1)
        self.assertTrue(observer.renderer_closed)

    def test_scope_and_camera_rejections_happen_before_renderer_calls(self):
        renderer = _MutableRenderer()
        observer = runner.ModelSmokeRenderObserver(renderer, lambda value: value)
        camera = _Camera(719, 720, native=True)
        with self.assertRaises(RuntimeError):
            observer.render_object_model(
                renderer.obj_id, camera, render_types=list(_NATIVE_TYPES),
                return_tensors=False, background=None)
        with self.assertRaises(ValueError):
            with observer.observe_native('a', 0):
                observer.render_object_model(
                    renderer.obj_id, camera, render_types=list(_NATIVE_TYPES),
                    return_tensors=False, background=None)
        self.assertEqual(renderer.render_calls, [])

    def test_renderer_close_failure_propagates_without_claiming_closed(self):
        renderer = _MutableRenderer(fail_close=True)
        observer = runner.ModelSmokeRenderObserver(renderer, lambda value: value)
        with self.assertRaises(RuntimeError):
            observer.close()
        self.assertEqual(renderer.close_calls, 1)
        self.assertFalse(observer.renderer_closed)


class ModelSmokeNetworkCounterTests(unittest.TestCase):
    def test_three_view_sequence_binds_fifteen_unchanged_network_calls_to_eighteen_renders(self):
        renderer = _MutableRenderer(vary_color=False)
        events = []
        observer = runner.ModelSmokeRenderObserver(renderer, _event_sink(events))
        network = _RecordingNetwork()
        counter = runner.ModelSmokeNetworkCounter(network, observer)
        supplied_inputs = []

        for ordinal, case_id in enumerate(('a', 'b', 'c')):
            with observer.observe_native(case_id, ordinal):
                returned, _, _, _ = _render(
                    observer, renderer, case_id=case_id, ordinal=ordinal, role='native')
                self.assertIs(returned, renderer.last_result)
            with observer.observe_refinement(case_id, ordinal):
                for _iteration in range(5):
                    returned, _, _, _ = _render(
                        observer, renderer, case_id=case_id,
                        ordinal=ordinal, role='crop')
                    self.assertIs(returned, renderer.last_result)
                    template = renderer.last_arrays['color'].transpose(2, 0, 1)[None].copy()
                    mask = renderer.last_arrays['mask'][None].copy()
                    query = np.full(template.shape, np.float32(.25), dtype=np.float32)
                    inputs = (query, template, mask)
                    supplied_inputs.append(inputs)
                    self.assertFalse(np.array_equal(query, template))
                    result = counter(*inputs)
                    self.assertIs(result, network.outputs[-1])

        expected = []
        for ordinal, case_id in enumerate(('a', 'b', 'c')):
            expected.append((case_id, ordinal, 'native', None))
            expected.extend((case_id, ordinal, 'crop', iteration) for iteration in range(5))
        self.assertEqual(
            [(event['case_id'], event['ordinal'], event['role'], event['iteration'])
             for event in events], expected)
        crop_events = [event for event in events if event['role'] == 'crop']
        self.assertEqual(len(events), 18)
        self.assertEqual(len(crop_events), 15)
        self.assertEqual(counter.attempted_calls, 15)
        self.assertEqual(counter.returned_calls, 15)
        self.assertEqual(len(network.calls), 15)
        for index, (event, inputs) in enumerate(zip(crop_events, supplied_inputs)):
            args, kwargs = network.calls[index]
            self.assertEqual(kwargs, {})
            self.assertEqual(len(args), 3)
            for actual, supplied in zip(args, inputs):
                self.assertIs(actual, supplied)
            descriptors = event['network_inputs']
            self.assertIs(descriptors['template_matches_observed_render'], True)
            self.assertIs(descriptors['call_returned'], True)
            self.assertEqual(descriptors['template_rgb_bchw']['shape'], [1, 3, 280, 280])
            self.assertEqual(descriptors['template_mask_bhw']['shape'], [1, 280, 280])
            self.assertNotEqual(
                descriptors['query_rgb_bchw']['sha256'],
                descriptors['template_rgb_bchw']['sha256'])
            self.assertEqual(float(inputs[1][0, 0, 0, 0]), float(np.float32(128. / 255.)))
        observer.close()

    def test_mismatched_template_is_rejected_before_network_delegate(self):
        renderer = _MutableRenderer(vary_color=False)
        events = []
        observer = runner.ModelSmokeRenderObserver(renderer, _event_sink(events))
        network = _RecordingNetwork()
        counter = runner.ModelSmokeNetworkCounter(network, observer)

        with self.assertRaises(ValueError):
            with observer.observe_refinement('a', 0):
                _render(observer, renderer)
                template = renderer.last_arrays['color'].transpose(2, 0, 1)[None].copy()
                template[0, 0, 0, 0] = np.float32(0.)
                mask = renderer.last_arrays['mask'][None].copy()
                query = np.full(template.shape, np.float32(.25), dtype=np.float32)
                counter(query, template, mask)

        self.assertEqual(renderer.render_count, 1)
        self.assertEqual(network.calls, [])
        self.assertEqual(counter.attempted_calls, 0)
        self.assertEqual(counter.returned_calls, 0)
        self.assertIs(events[0]['network_inputs']['template_matches_observed_render'], False)
        self.assertIs(events[0]['network_inputs']['call_returned'], False)

    def test_network_exception_counts_attempt_but_not_return(self):
        renderer = _MutableRenderer(vary_color=False)
        events = []
        observer = runner.ModelSmokeRenderObserver(renderer, _event_sink(events))
        network = _RecordingNetwork(fail=True)
        counter = runner.ModelSmokeNetworkCounter(network, observer)

        with self.assertRaises(RuntimeError):
            with observer.observe_refinement('a', 0):
                _render(observer, renderer)
                template = renderer.last_arrays['color'].transpose(2, 0, 1)[None].copy()
                mask = renderer.last_arrays['mask'][None].copy()
                query = np.full(template.shape, np.float32(.25), dtype=np.float32)
                counter(query, template, mask)

        self.assertEqual(counter.attempted_calls, 1)
        self.assertEqual(counter.returned_calls, 0)
        self.assertEqual(len(network.calls), 1)
        self.assertIs(events[0]['network_inputs']['template_matches_observed_render'], True)
        self.assertIs(events[0]['network_inputs']['call_returned'], False)


class ModelSmokePreflightTests(unittest.TestCase):
    def test_real_shaped_three_resource_flags_bind_two_effective_recipe_flags(self):
        plan, resources, provenance, stage_flags = _synthetic_preflight_inputs()
        manifest = {'_capture': plan}
        with patch.object(runner, 'inference_provenance', return_value=provenance) as verify:
            verified_plan, plan_manifest, recipe, flags, returned_provenance = (
                runner._model_smoke_preflight(
                    Path('synthetic-bundle'), manifest, resources, 'cpu'))

        verify.assert_called_once_with(
            Path('synthetic-bundle'), capture_plan=plan,
            capture_resources=resources)
        self.assertIs(verified_plan, plan)
        self.assertIs(plan_manifest, plan.manifest)
        self.assertEqual(flags, stage_flags)
        self.assertEqual(set(flags), {
            'unlit_templates', 'disable_multisampling', 'grayscale'})
        self.assertEqual(recipe['render_flags'], {
            'unlit_templates': False, 'disable_multisampling': False})
        self.assertIs(returned_provenance, provenance)

    def test_public_issued_smoke_resources_reach_real_inference_preflight(self):
        with tempfile.TemporaryDirectory(prefix='model-smoke-issued-preflight-') as directory:
            temporary_root = Path(directory)
            fake_root = temporary_root / 'fake-resource-root'
            with proof_fixtures._fake_resource_environment(assets, fake_root):
                bundle, _, _, plan = proof_fixtures._new_bundle(
                    temporary_root / 'synthetic-bundle')
                resources = assets.capture_stage_resources(
                    plan, stage='smoke', device='cpu')
                self.assertEqual(resources.stage, 'smoke')
                self.assertEqual(resources.device, 'cpu')
                self.assertEqual(resources.required, ())
                self.assertEqual(dict(resources.render_flags), {
                    'unlit_templates': False,
                    'disable_multisampling': False,
                    'grayscale': False,
                })
                self.assertEqual(dict(resources.output.recipe['render_flags']), {
                    'unlit_templates': False,
                    'disable_multisampling': False,
                })

                provenance = assets.inference_provenance(
                    bundle.root, capture_plan=plan, capture_resources=resources)
                self.assertEqual(provenance['capture_provenance']['render_flags'],
                                 dict(resources.render_flags))
                self.assertEqual(
                    provenance['capture_provenance']['output_resource']['recipe']['render_flags'],
                    dict(resources.output.recipe['render_flags']))

                with patch.object(runner, 'CAPTURE_CACHE', assets.CAPTURE_CACHE):
                    verified_plan, plan_manifest, recipe, flags, returned_provenance = (
                        runner._model_smoke_preflight(
                            bundle.root, {'_capture': plan}, resources, 'cpu'))

        self.assertIs(verified_plan, plan)
        self.assertIs(plan_manifest, plan.manifest)
        self.assertEqual(flags, dict(resources.render_flags))
        self.assertEqual(recipe['render_flags'], dict(resources.output.recipe['render_flags']))
        self.assertEqual(set(flags), {
            'unlit_templates', 'disable_multisampling', 'grayscale'})
        self.assertIsInstance(returned_provenance, dict)
        self.assertEqual(returned_provenance['capture_provenance']['render_flags'], flags)

    def test_preflight_rejects_provenance_prerequisites_and_wrong_output_key(self):
        for mutation in ('prerequisite', 'output-key'):
            with self.subTest(mutation=mutation):
                plan, resources, provenance, _ = _synthetic_preflight_inputs()
                capture_provenance = provenance['capture_provenance']
                if mutation == 'prerequisite':
                    capture_provenance['used_resources'] = ['synthetic-bank']
                else:
                    capture_provenance['output_resource']['resource_key'] = '0' * 64
                with patch.object(runner, 'inference_provenance', return_value=provenance):
                    with self.assertRaises(ValueError):
                        runner._model_smoke_preflight(
                            Path('synthetic-bundle'), {'_capture': plan},
                            resources, 'cpu')


class ModelSmokeArtifactBindingTests(unittest.TestCase):
    def _native_metadata(self, identities):
        color = {
            'target_name': 'GL_RENDERBUFFER', 'format_name': 'GL_RGBA',
            'samples': 4, 'width': 720, 'height': 720, 'renderbuffer_id': 11,
            'ordinary_storage_delegated': True, 'success': True,
        }
        depth = {
            'target_name': 'GL_RENDERBUFFER', 'format_name': 'GL_DEPTH_COMPONENT24',
            'samples': 4, 'width': 720, 'height': 720, 'renderbuffer_id': 12,
            'ordinary_storage_delegated': True, 'success': True,
        }
        return {
            'schema_version': 1,
            'render_policy': 'capture_zero_sample_v1',
            'coordinate_mode': 'integer_centers_v1',
            'depth_units': 'millimetres',
            'dimensions': [720, 720],
            'allocation_generation': 1,
            'offscreen_identity': 7,
            'framebuffer_fields': {
                'multisample_draw_fbo': 3,
                'single_sample_read_fbo': 4,
                'multisample_dimensions': [720, 720],
            },
            'allocation_pair': {
                'color': color, 'depth': depth,
                'dimensions': [720, 720], 'passed': True,
            },
            'allocation_calls': [color, depth],
            'framebuffer_complete': True,
            'gl_samples': 0,
            'gl_sample_buffers': 0,
            'framebuffer_bindings_restored': True,
            'current_context_released': True,
            'dimension_match': True,
            'color_storage_policy':
                'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage',
            'depth_storage_policy':
                'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage',
            'source_identities': identities,
        }

    def test_source_identity_resolves_runtime_closure_aliases_to_physical_files(self):
        with tempfile.TemporaryDirectory(prefix='model-smoke-source-alias-') as directory:
            runner_file, source_closure, metadata = _synthetic_runtime_sources(directory)
            with patch.object(runner, '__file__', str(runner_file)):
                runner._model_smoke_source_identity(metadata, source_closure)

    def test_first_native_artifact_event_uses_case_ordinal_and_runtime_alias_pins(self):
        with tempfile.TemporaryDirectory(prefix='model-smoke-first-event-') as directory:
            runner_file, source_closure, identity_metadata = _synthetic_runtime_sources(directory)
            metadata = self._native_metadata(identity_metadata['source_identities'])
            color = np.full((720, 720, 3), np.float32(128. / 255.), dtype=np.float32)
            depth = np.full((720, 720), np.float32(1000.), dtype=np.float32)
            mask = depth > 0.
            payload = {
                'case_id': 'a', 'ordinal': 0, 'role': 'native', 'iteration': None,
                'camera': {
                    'width': 720, 'height': 720, 'fx': 360., 'fy': 360.,
                    'cx': 359.5, 'cy': 359.5,
                    'T_world_from_eye_mm': np.eye(4, dtype=np.float64).tolist(),
                },
                'requested_render_types': ('rgb', 'depth', 'mask'),
                'return_tensors': False, 'requested_background': None,
                'render_policy_metadata': metadata,
                'arrays': {
                    'color_norm_f32': color, 'depth_mm': depth, 'mask': mask,
                    'model_rgb_u8': None, 'cpu_rgb_u8': None,
                    'cpu_depth_m': None, 'cpu_mask': None,
                },
                'mask_equals_depth_positive': True,
                'returned': True,
            }
            events = []
            lifecycle = {'metadata': None}
            with patch.object(runner, '__file__', str(runner_file)):
                event = runner._model_smoke_artifact_event(
                    events, payload, source_closure, lifecycle)

        self.assertEqual(events, [event])
        self.assertEqual(event['render_index'], 0)
        self.assertEqual(event['case_id'], 'a')
        self.assertEqual(event['ordinal'], 0)
        self.assertEqual(event['role'], 'native')
        self.assertIsNone(event['iteration'])
        self.assertEqual(event['arrays']['color_norm_f32']['shape'], [720, 720, 3])
        self.assertEqual(event['arrays']['depth_mm']['dtype'], '<f4')
        self.assertIsNone(event['arrays']['model_rgb_u8'])


class ModelSmokeWorkerBoundaryTests(unittest.TestCase):
    def test_missing_capture_plan_rejects_before_resources_optional_imports_or_output_effects(self):
        imported_optional = []
        original_import = builtins.__import__
        optional_roots = {
            'torch', 'cv2', 'utils', 'quality_assets', 'quality_capture',
            'quality_contract', 'quality_gotrack', 'glb_model',
            'show3d_experiment', 'renderer',
        }

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split('.', 1)[0] in optional_roots:
                imported_optional.append(name)
                raise AssertionError(f'optional import before plan rejection: {name}')
            return original_import(name, globals, locals, fromlist, level)

        class _PoisonResources:
            def __getattribute__(self, name):
                raise AssertionError(f'resources accessed before capture-plan rejection: {name}')

        class _PoisonDevice:
            def __str__(self):
                raise AssertionError('device inspected before capture-plan rejection')

        with patch('builtins.__import__', new=guarded_import):
            with self.assertRaises(ValueError):
                runner.capture_model_smoke(
                    object(), {}, _PoisonResources(), device=_PoisonDevice())
        self.assertEqual(imported_optional, [])

    def test_torch_import_failure_returns_honest_pending_and_cli_writes_only_private_file(self):
        with tempfile.TemporaryDirectory(prefix='model-smoke-pending-') as directory:
            capture_cache = Path(directory) / 'capture-resources'
            with patch.object(runner, 'CAPTURE_CACHE', capture_cache):
                plan, resources, provenance, stage_flags = _synthetic_preflight_inputs()
                trial_directory = capture_cache / 'capture-smoke-trials' / 'inert-test'
                pending_path = trial_directory / 'worker-pending.json'
                canonical_sidecar = resources.output.sidecar_path
                manifest = {'_capture': plan}
                original_import = builtins.__import__
                import_attempts = []

                def reject_optional_torch(name, globals=None, locals=None,
                                          fromlist=(), level=0):
                    if name == 'torch':
                        import_attempts.append(name)
                        raise ImportError('synthetic test blocks optional Torch import')
                    return original_import(name, globals, locals, fromlist, level)

                with (
                    patch.object(runner, 'read_input', return_value=manifest),
                    patch.object(runner, '_validate_capture_stage_options'),
                    patch.object(assets, 'capture_stage_resources', return_value=resources),
                    patch.object(
                        runner, '_model_smoke_preflight',
                        return_value=(plan, plan.manifest,
                                      dict(resources.output.recipe), stage_flags, provenance)),
                    patch.object(runner, '_model_smoke_process_peak_rss',
                                 return_value=(None, None)),
                    patch('builtins.__import__', new=reject_optional_torch),
                    patch('sys.argv', [
                        'quality_runner', 'smoke', '--bundle', str(Path(directory) / 'bundle'),
                        '--output', str(pending_path), '--device', 'cpu',
                    ]),
                ):
                    with self.assertRaises(SystemExit) as exit_result:
                        runner.main()

                self.assertEqual(exit_result.exception.code, 1)
                self.assertEqual(import_attempts, ['torch'])
                self.assertTrue(pending_path.is_file())
                pending = json.loads(pending_path.read_text(encoding='utf-8'))
                self.assertEqual(pending['status'], 'failed')
                self.assertEqual(pending['error']['type'], 'ImportError')
                self.assertIn('synthetic test blocks', pending['error']['message'])
                self.assertEqual(pending['failure_location'], {
                    'phase': 'loading', 'case_id': None, 'iteration': None,
                })
                self.assertFalse(pending['renderer_closed'])
                self.assertIsNone(pending['close_error'])
                self.assertEqual(pending['constructor_records'], [])
                self.assertEqual(pending['render_events'], [])
                self.assertEqual(pending['rows'], [])
                self.assertIsNone(pending['checkpoint_sha256'])
                self.assertIsNone(pending['build'])
                self.assertNotIn('terminal_exit_code', pending)
                self.assertNotIn('process_reaped', pending)
                self.assertNotIn('actual_unmocked_execution', pending)

                runtime = pending['runtime_evidence']
                timings = runtime['timings_ms']
                self.assertTrue(math.isfinite(timings['worker_total']))
                self.assertGreaterEqual(timings['worker_total'], 0.)
                for name in ('network_load', 'renderer_construct', 'renderer_close'):
                    self.assertIsNone(timings[name])
                for case in timings['cases']:
                    for name in ('native_render', 'cpu_geometry', 'refine', 'validation'):
                        self.assertIsNone(case[name])
                self.assertEqual(runtime['render_calls'], [])
                self.assertEqual(runtime['memory']['measurement'], 'cpu-no-cuda')
                for name in ('cuda_total_bytes', 'cuda_peak_allocated_bytes',
                             'cuda_peak_reserved_bytes'):
                    self.assertIsNone(runtime['memory'][name])

                self.assertEqual(
                    [item.name for item in trial_directory.iterdir()],
                    ['worker-pending.json'])
                self.assertFalse(canonical_sidecar.exists())
                written_bytes = pending_path.read_bytes()
                with self.assertRaises(FileExistsError):
                    runner._model_smoke_write_pending(
                        pending_path, resources.output, pending)
                self.assertEqual(pending_path.read_bytes(), written_bytes)


class ModelSmokeRuntimeEvidenceTests(unittest.TestCase):
    def test_cpu_timing_is_finite_without_cuda_and_cpu_memory_has_no_cuda_claim(self):
        class _NoCudaAccess:
            def __getattr__(self, name):
                raise AssertionError(f'CPU timing attempted CUDA access: {name}')

        no_cuda = _NoCudaAccess()
        started = runner._model_smoke_timing_start(no_cuda, 'cpu')
        elapsed_ms = runner._model_smoke_timing_finish(no_cuda, 'cpu', started)
        self.assertTrue(math.isfinite(elapsed_ms))
        self.assertGreaterEqual(elapsed_ms, 0.)

        memory = runner._model_smoke_memory_evidence('cpu')
        self.assertEqual(memory['measurement'], 'cpu-no-cuda')
        for name in ('cuda_total_bytes', 'cuda_peak_allocated_bytes',
                     'cuda_peak_reserved_bytes'):
            self.assertIsNone(memory[name])


if __name__ == '__main__':
    unittest.main()
