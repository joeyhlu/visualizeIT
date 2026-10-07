"""Independent synthetic tests for public model-smoke semantic validators.

The fixtures describe bytes only. They use the existing inert capture bundle
builder and do not create an accepted real execution or canonical smoke asset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bench import quality_assets as assets
from bench import quality_capture as capture
from bench import test_quality_capture_resource_proof as proof_fixtures


def _plain(value):
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def _sha(label):
    data = label.encode('utf-8') if isinstance(label, str) else label
    return hashlib.sha256(data).hexdigest().upper()


def _descriptor(label, shape, dtype):
    """Return a correctly framed, inert descriptor with its own synthetic SHA."""
    metadata = {'shape': list(shape), 'dtype': dtype, 'order': 'C'}
    digest = hashlib.sha256(
        b'quality-model-smoke-array-v1\0' + _canonical(metadata) + b'\0' +
        ('synthetic-array-payload|' + label).encode('utf-8')).hexdigest().upper()
    return {**metadata, 'sha256': digest}


def _matrix_multiply(left, right):
    return [[sum(left[row][index] * right[index][column]
                 for index in range(3)) for column in range(3)] for row in range(3)]


def _seed_pose(bounds, degrees, offsets):
    """Build the fixed Rx@Ry@Rz@diag pose independently from the validator."""
    x, y, z = (math.radians(value) for value in degrees)
    rx = [[1., 0., 0.], [0., math.cos(x), -math.sin(x)],
          [0., math.sin(x), math.cos(x)]]
    ry = [[math.cos(y), 0., math.sin(y)], [0., 1., 0.],
          [-math.sin(y), 0., math.cos(y)]]
    rz = [[math.cos(z), -math.sin(z), 0.], [math.sin(z), math.cos(z), 0.],
          [0., 0., 1.]]
    rotation = _matrix_multiply(_matrix_multiply(_matrix_multiply(rx, ry), rz),
                                [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    minimum, maximum = bounds['min'], bounds['max']
    center = [(minimum[index] + maximum[index]) / 2. for index in range(3)]
    extent = max(maximum[index] - minimum[index] for index in range(3))
    translation = [extent * offsets[row] - sum(
        rotation[row][column] * center[column] for column in range(3))
        for row in range(3)]
    return [rotation[row] + [translation[row]] for row in range(3)] + [
        [0., 0., 0., 1.]]


def _inverse_camera_mm(seed):
    rotation_t = [[seed[column][row] for column in range(3)] for row in range(3)]
    translation = [-sum(rotation_t[row][column] * seed[column][3]
                        for column in range(3)) * 1000. for row in range(3)]
    return [rotation_t[row] + [translation[row]] for row in range(3)] + [
        [0., 0., 0., 1.]]


@contextmanager
def _coherent_fake_resources(root):
    """Reuse the frozen fake-resource fixture with one shared inert runtime tree."""
    root = Path(root)
    with proof_fixtures._fake_resource_environment(assets, root) as tree:
        package_roots = {}
        site_packages = root / '.cache' / 'quality-windows' / 'Lib' / 'site-packages'
        for distribution_name, (package, files) in assets._CAPTURE_RENDERER_RUNTIME_FILES.items():
            package_root = site_packages / package
            package_roots[distribution_name] = package_root
            for relative in files:
                path = package_root.joinpath(*relative.split('/'))
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(
                        f'inert semantic runtime source|{distribution_name}|{relative}\n'.encode())

        previous_distribution = assets.distribution

        class _Distribution:
            def __init__(self, name):
                self.name = name
                self.version = 'synthetic-runtime-version-' + name

            def locate_file(self, package):
                expected = assets._CAPTURE_RENDERER_RUNTIME_FILES[self.name][0]
                if str(package) != expected:
                    raise AssertionError('Fixture requested an undeclared runtime package')
                return package_roots[self.name]

        def distribution(name):
            if name not in package_roots:
                raise AssertionError('Fixture requested an undeclared runtime distribution')
            return _Distribution(name)

        assets.distribution = distribution
        try:
            yield root, tree
        finally:
            assets.distribution = previous_distribution


def _source_identities(root, proof_source_closure, producer_closure):
    paths = {
        'quality_gotrack': ('bench.quality_gotrack', 'bench/quality_gotrack.py',
                            'bench/quality_gotrack.py'),
        'quality_render_stability': (
            'bench.quality_render_stability', 'bench/quality_render_stability.py',
            'bench/quality_render_stability.py'),
        'pyrender_offscreen': (
            'pyrender.offscreen',
            '.cache/quality-windows/Lib/site-packages/pyrender/offscreen.py',
            'runtime/pyrender/offscreen.py'),
        'pyrender_renderer': (
            'pyrender.renderer',
            '.cache/quality-windows/Lib/site-packages/pyrender/renderer.py',
            'runtime/pyrender/renderer.py'),
    }
    identities = {}
    for name, (module, physical, closure_path) in paths.items():
        identities[name] = {
            'module': module,
            'file': str((Path(root) / physical).resolve()),
            'sha256': proof_source_closure[physical],
        }
        if name.startswith('pyrender_'):
            # Runtime package bytes are placed at the same inert physical paths
            # used by the source slice, so the producer alias and report pin join.
            if producer_closure.get(closure_path) != proof_source_closure[physical]:
                raise AssertionError('Synthetic runtime alias differs from its pinned physical source')
    return identities


def _attachment(kind, width, height, identity):
    return {
        'target_name': 'GL_RENDERBUFFER',
        'format_name': 'GL_RGBA' if kind == 'color' else 'GL_DEPTH_COMPONENT24',
        'samples': 4, 'width': width, 'height': height,
        'renderbuffer_id': identity, 'ordinary_storage_delegated': True,
        'success': True,
    }


def _metadata(index, dimensions, generation, root, proof_sources, producer_closure):
    width, height = dimensions
    color = _attachment('color', width, height, 2000 + generation * 2)
    depth = _attachment('depth', width, height, 2001 + generation * 2)
    pair = {'color': color, 'depth': depth, 'dimensions': [width, height], 'passed': True}
    fresh = index in (0, 1, 6, 7, 12, 13)
    return {
        'schema_version': 1,
        'render_policy': 'capture_zero_sample_v1',
        'coordinate_mode': 'integer_centers_v1',
        'depth_units': 'millimetres',
        'dimensions': [width, height],
        'allocation_generation': generation,
        'offscreen_identity': 500 + generation,
        'framebuffer_fields': {
            'multisample_draw_fbo': 1000 + generation * 2,
            'single_sample_read_fbo': 1001 + generation * 2,
            'multisample_dimensions': [width, height],
        },
        'allocation_pair': pair,
        'allocation_calls': [copy.deepcopy(color), copy.deepcopy(depth)] if fresh else [],
        'framebuffer_complete': True,
        'gl_samples': 0,
        'gl_sample_buffers': 0,
        'framebuffer_bindings_restored': True,
        'current_context_released': True,
        'dimension_match': True,
        'color_storage_policy': 'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage',
        'depth_storage_policy':
            'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage',
        'source_identities': _source_identities(root, proof_sources, producer_closure),
    }


def _runtime_evidence():
    case_rows = [
        {'case_id': case, 'native_render': 2.0, 'cpu_geometry': 10.0,
         'refine': 50.0, 'validation': 3.0}
        for case in ('a', 'b', 'c')]
    render_calls = [
        {'render_index': index,
         'elapsed_ms': 2.0 if index in (0, 6, 12) else 0.2}
        for index in range(18)]
    return {
        'schema_version': 1,
        'scope': 'model-only-smoke',
        'clock': 'perf_counter_ns',
        'cuda_synchronization': 'not-applicable',
        'timings_ms': {
            'worker_total': 300.0, 'network_load': 50.0,
            'renderer_construct': 5.0, 'renderer_close': 3.0,
            'cases': case_rows,
        },
        'render_calls': render_calls,
        'memory': {
            'measurement': 'cpu-no-cuda', 'cuda_total_bytes': None,
            'cuda_peak_allocated_bytes': None, 'cuda_peak_reserved_bytes': None,
            'process_peak_rss_bytes': None, 'process_peak_rss_method': None,
        },
    }


def _evidence(plan, asset_receipt, resources, fake_root):
    recipe = _plain(resources.output.recipe)
    closure = _plain(resources.output.source_closure)
    proof_sources = _plain(asset_receipt.renderer_proof['source_closure'])
    constructor = {
        'renderer_id': 0,
        'api': 'bench.quality_gotrack.TexturedRenderer',
        'object_id': asset_receipt.object_id,
        'asset_sha256': asset_receipt.asset_sha256,
        'arguments': {
            'unlit': recipe['render_flags']['unlit_templates'],
            'disable_multisampling': False,
            'coordinate_mode': 'integer_centers_v1',
            'render_policy': 'capture_zero_sample_v1',
        },
        'observed_configuration': {
            'unlit': recipe['render_flags']['unlit_templates'],
            'disable_multisampling': False,
            'coordinate_mode': 'integer_centers_v1',
            'render_policy': 'capture_zero_sample_v1',
            'scene_bg_rgba': [0.5, 0.5, 0.5, 0.0],
            'ambient_light_rgb': [0.02, 0.02, 0.02],
            'spotlight_intensity': 2.4,
            'spotlight_inner_cone': math.pi / 16,
            'spotlight_outer_cone': math.pi / 6,
        },
    }
    build = {
        'python': 'synthetic-python', 'numpy': 'synthetic-numpy',
        'torch': 'synthetic-torch', 'opencv': 'synthetic-opencv',
        'pyrender': 'synthetic-pyrender', 'cuda_runtime': None, 'gpu_name': None,
        'dtype': 'float32', 'batch_size': 1, 'crop_size': [280, 280], 'iterations': 5,
    }
    generations = (1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6)
    view_data = (
        ('a', [9.0, -13.0, 7.0], [0.02, -0.04, 3.0]),
        ('b', [-19.0, 24.0, -12.0], [-0.17, 0.11, 3.15]),
        ('c', [27.0, 15.0, 21.0], [0.15, -0.126, 3.3]),
    )
    bounds = _plain(asset_receipt.document['post_node_bounds_m'])
    rows = []
    events = []
    for ordinal, (case_id, degrees, offsets) in enumerate(view_data):
        seed = _seed_pose(bounds, degrees, offsets)
        native_descriptors = None
        case_events = []
        for slot in range(6):
            index = ordinal * 6 + slot
            native = slot == 0
            role = 'native' if native else 'crop'
            iteration = None if native else slot - 1
            width = 720 if native else 280
            dimensions = [width, width]
            camera_pose = _inverse_camera_mm(seed) if native else [
                [1.0, 0.0, 0.0, float(ordinal)],
                [0.0, 1.0, 0.0, float(iteration) / 10.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            ]
            camera = {
                'width': width, 'height': width,
                'fx': 360.0 if native else 132.0 + ordinal + iteration / 20.0,
                'fy': 360.0 if native else 137.0 + ordinal + iteration / 30.0,
                'cx': 359.5 if native else 140.25 + iteration / 100.0,
                'cy': 359.5 if native else 139.75 + ordinal / 100.0,
                'T_world_from_eye_mm': camera_pose,
            }
            if native:
                arrays = {
                    'color_norm_f32': _descriptor(f'{case_id}-native-color',
                                                  [720, 720, 3], '<f4'),
                    'depth_mm': _descriptor(f'{case_id}-native-depth', [720, 720], '<f4'),
                    'mask': _descriptor(f'{case_id}-native-mask', [720, 720], '|b1'),
                    'model_rgb_u8': _descriptor(f'{case_id}-model-rgb',
                                                [720, 720, 3], '|u1'),
                    'cpu_rgb_u8': _descriptor(f'{case_id}-cpu-rgb',
                                              [720, 720, 3], '|u1'),
                    'cpu_depth_m': _descriptor(f'{case_id}-cpu-depth', [720, 720], '<f8'),
                    'cpu_mask': _descriptor(f'{case_id}-cpu-mask', [720, 720], '|b1'),
                }
                native_descriptors = arrays
                inputs = None
            else:
                arrays = {
                    'color_norm_f32': _descriptor(f'{case_id}-{iteration}-color',
                                                  [280, 280, 3], '<f4'),
                    'depth_mm': _descriptor(f'{case_id}-{iteration}-depth', [280, 280], '<f4'),
                    'mask': _descriptor(f'{case_id}-{iteration}-mask', [280, 280], '|b1'),
                    'model_rgb_u8': None, 'cpu_rgb_u8': None,
                    'cpu_depth_m': None, 'cpu_mask': None,
                }
                inputs = {
                    'query_rgb_bchw': _descriptor(f'{case_id}-{iteration}-query',
                                                  [1, 3, 280, 280], '<f4'),
                    'template_rgb_bchw': _descriptor(f'{case_id}-{iteration}-template',
                                                     [1, 3, 280, 280], '<f4'),
                    'template_mask_bhw': _descriptor(f'{case_id}-{iteration}-template-mask',
                                                      [1, 280, 280], '|b1'),
                    'template_matches_observed_render': True,
                    'call_returned': True,
                }
            event = {
                'render_index': index, 'renderer_id': 0, 'case_id': case_id,
                'ordinal': ordinal, 'role': role, 'iteration': iteration,
                'camera': camera,
                'requested_render_types': ['rgb', 'depth', 'mask'] if native
                else ['rgb', 'mask', 'depth'],
                'return_tensors': not native,
                'requested_background': None if native else [0.5, 0.5, 0.5],
                'render_policy_metadata': _metadata(
                    index, dimensions, generations[index], fake_root, proof_sources, closure),
                'arrays': arrays,
                'mask_equals_depth_positive': True,
                'network_inputs': inputs,
            }
            events.append(event)
            case_events.append(event)

        native = native_descriptors
        rows.append({
            'pose_id': case_id,
            'seed_camera_from_object_m': seed,
            'render': {
                'gpu_rgb_sha256': native['model_rgb_u8']['sha256'],
                'gpu_depth_mm_sha256': native['depth_mm']['sha256'],
                'gpu_mask_sha256': native['mask']['sha256'],
                'cpu_rgb_sha256': native['cpu_rgb_u8']['sha256'],
                'cpu_depth_m_sha256': native['cpu_depth_m']['sha256'],
                'cpu_mask_sha256': native['cpu_mask']['sha256'],
                'mask_iou': 0.97, 'median_common_depth_error_m': 0.001,
                'common_depth_pixel_count': 120000,
                'mask_equals_depth_positive': True,
                'native_dimensions': [720, 720],
                'coordinate_mode': 'integer_centers_v1',
                'render_policy': 'capture_zero_sample_v1',
                'unlit_templates': recipe['render_flags']['unlit_templates'],
                'disable_multisampling': False,
            },
            'refinement': {
                'candidate_present': True,
                'iterations': [
                    {
                        'index': iteration, 'solver_success': True,
                        'crop_dimensions': [280, 280], 'query_rewarp_factor': 1.0,
                        'max_sampling_map_difference_px': 0.25,
                        'sampling_map_eligible_count': 1000,
                        'render_generation': generations[ordinal * 6 + iteration + 1],
                    }
                    for iteration in range(5)
                ],
                'pose_camera_from_object_m': seed,
                'points_object_m_sha256': _sha(case_id + '-points'),
                'pixels_native_sha256': _sha(case_id + '-pixels'),
                'weights_sha256': _sha(case_id + '-weights'),
            },
            'validation': {
                'passed': True, 'reason': None,
                'stats': {
                    'correspondences': 30, 'inliers': 25,
                    'median_reprojection_720': 1.0,
                    'p95_reprojection_720': 4.0,
                    'spatial_support': 0.2, 'score': 1.2,
                    'raw_correspondences': 40, 'retained_correspondences': 30,
                    'unsupported_correspondences': 5,
                },
            },
        })

    runtime = _runtime_evidence()
    execution_id = 'synthetic-model-smoke-execution-001'
    started = '2026-10-04T10:00:00Z'
    ended = '2026-10-04T10:00:01Z'
    checkpoint = assets.MODELS['gotrack_checkpoint.pt']['sha256'].upper()
    pending = {
        'schema_version': 1,
        'kind': 'quality-capture-model-smoke-pending-v1',
        'execution_id': execution_id,
        'resource_key': resources.output.resource_key,
        'recipe': recipe,
        'source_closure': closure,
        'asset_receipt_sha256': asset_receipt.receipt_sha256,
        'renderer_proof_sha256': asset_receipt.renderer_proof_sha256,
        'started_at_utc': started,
        'worker_ended_at_utc': ended,
        'checkpoint_sha256': checkpoint,
        'device': 'cpu',
        'build': build,
        'constructor_records': [constructor],
        'render_events': events,
        'rows': rows,
        'status': 'measured_success',
        'error': None,
        'close_error': None,
        'renderer_closed': True,
        'captured_rgb_read': False,
        'evaluator_data_read': False,
        'failure_location': None,
        'runtime_evidence': runtime,
    }
    pending_bytes = _canonical(pending)
    renderer_execution = {
        'schema_version': 1,
        'kind': 'quality-capture-model-smoke-renderer-execution-v1',
        'execution_id': execution_id,
        'resource_key': resources.output.resource_key,
        'recipe': recipe,
        'source_closure': closure,
        'started_at_utc': started,
        'ended_at_utc': ended,
        'terminal_exit_code': 0,
        'actual_unmocked_execution': True,
        'constructor_records': [constructor],
        'generations': events,
        'cleanup': {'renderer_closed': True, 'process_reaped': True},
        'runtime_evidence': runtime,
    }
    renderer_execution_bytes = _canonical(renderer_execution)
    smoke = {
        'schema_version': 1,
        'resource_kind': 'quality-capture-model-smoke-v1',
        'resource_key': resources.output.resource_key,
        'recipe': recipe,
        'source_closure': closure,
        'asset_receipt_sha256': asset_receipt.receipt_sha256,
        'renderer_proof_sha256': asset_receipt.renderer_proof_sha256,
        'renderer_execution_receipt_sha256': _sha(renderer_execution_bytes),
        'execution': {
            'execution_id': execution_id,
            'started_at_utc': started,
            'ended_at_utc': ended,
            'terminal_exit_code': 0,
            'actual_unmocked_execution': True,
            'inference_executed': True,
            'captured_rgb_read': False,
            'evaluator_data_read': False,
            'checkpoint_sha256': checkpoint,
            'device': 'cpu',
            'build': build,
            'renderer_closed': True,
            'process_reaped': True,
            'runtime_evidence': runtime,
        },
        'rows': rows,
    }
    return pending, pending_bytes, smoke, _canonical(smoke), renderer_execution_bytes


class ModelSmokeSemanticValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hot3d-smoke-semantic-')
        self.addCleanup(self.temporary.cleanup)
        self.fake_root = Path(self.temporary.name) / 'fake-resource-root'
        self.environment = _coherent_fake_resources(self.fake_root)
        self.environment.__enter__()
        self.addCleanup(self.environment.__exit__, None, None, None)
        self.bundle, _, _, self.plan = proof_fixtures._new_bundle(
            Path(self.temporary.name) / 'synthetic-bundle')
        self.asset_receipt = assets.read_capture_asset_receipt(self.plan)
        self.resources = assets.capture_stage_resources(
            self.plan, stage='smoke', device='cpu')
        (self.pending, self.pending_bytes, self.smoke, self.smoke_bytes,
         self.renderer_execution_bytes) = _evidence(
            self.plan, self.asset_receipt, self.resources, self.fake_root)

    def _validate_pending(self, document=None, raw=None, *, plan=None, resources=None):
        if raw is None:
            raw = _canonical(self.pending if document is None else document)
        return assets.validate_capture_model_smoke_pending(
            self.plan if plan is None else plan,
            self.resources if resources is None else resources, raw)

    def _validate_finalized(self, smoke=None, execution=None, *, key=None,
                            recipe=None, closure=None, asset_receipt=None):
        smoke_bytes = (self.smoke_bytes if smoke is None else
                       smoke if type(smoke) is bytes else _canonical(smoke))
        execution_bytes = (self.renderer_execution_bytes if execution is None else
                           execution if type(execution) is bytes else _canonical(execution))
        return assets.validate_capture_model_smoke_finalized(
            smoke_bytes, execution_bytes,
            asset_receipt=self.asset_receipt if asset_receipt is None else asset_receipt,
            expected_recipe=self.resources.output.recipe if recipe is None else recipe,
            expected_source_closure=(self.resources.output.source_closure
                                     if closure is None else closure),
            expected_key=self.resources.output.resource_key if key is None else key)

    def _reject_pending(self, *, document=None, raw=None, plan=None, resources=None):
        with self.assertRaises((ValueError, TypeError, capture.CaptureIntegrityError)):
            self._validate_pending(document, raw, plan=plan, resources=resources)

    def _reject_finalized(self, *, smoke=None, execution=None, key=None,
                          recipe=None, closure=None, asset_receipt=None):
        with self.assertRaises((ValueError, TypeError, capture.CaptureIntegrityError)):
            self._validate_finalized(smoke, execution, key=key, recipe=recipe,
                                     closure=closure, asset_receipt=asset_receipt)

    def test_complete_inert_pending_and_finalized_evidence_validate_through_public_apis(self):
        pending_summary = self._validate_pending()
        finalized_summary = self._validate_finalized()
        self.assertEqual(len(pending_summary['render_events']), 18)
        self.assertEqual(sum(event['network_inputs'] is not None
                             for event in pending_summary['render_events']), 15)
        self.assertEqual(tuple(row['pose_id'] for row in pending_summary['rows']),
                         ('a', 'b', 'c'))
        self.assertEqual(finalized_summary['render_count'], 18)
        self.assertEqual(finalized_summary['network_call_count'], 15)
        self.assertTrue(finalized_summary['process_reaped'])
        observed = self.pending['constructor_records'][0]['observed_configuration']
        self.assertEqual(observed['scene_bg_rgba'], [0.5, 0.5, 0.5, 0.0])
        self.assertEqual(observed['ambient_light_rgb'], [0.02, 0.02, 0.02])
        with self.assertRaises(TypeError):
            pending_summary['status'] = 'failed'

    def test_pending_bindings_reject_wrong_plan_resources_recipe_source_and_checkpoint(self):
        self._reject_pending(plan=replace(self.plan, bundle=self.plan.bundle.parent / 'stale'))
        self._reject_pending(resources=replace(self.resources, device='cuda'))
        for field in ('resource_key', 'checkpoint_sha256'):
            with self.subTest(binding=field):
                document = copy.deepcopy(self.pending)
                document[field] = '0' * 64
                self._reject_pending(document=document)
        wrong_recipe = copy.deepcopy(self.pending)
        wrong_recipe['recipe']['render_flags']['unlit_templates'] = True
        self._reject_pending(document=wrong_recipe)
        wrong_closure = copy.deepcopy(self.pending)
        source = next(iter(wrong_closure['source_closure']))
        wrong_closure['source_closure'][source] = '0' * 64
        self._reject_pending(document=wrong_closure)

    def test_strict_pending_and_finalized_parsers_reject_duplicate_unknown_and_nonfinite_json(self):
        duplicates = b'{"schema_version":1,"schema_version":1}'
        self._reject_pending(raw=duplicates)
        bad_document = copy.deepcopy(self.pending)
        bad_document['unreviewed'] = True
        self._reject_pending(document=bad_document)
        nonfinite = copy.deepcopy(self.pending)
        nonfinite['rows'][0]['render']['mask_iou'] = float('inf')
        self._reject_pending(raw=json.dumps(nonfinite, separators=(',', ':')).encode('utf-8'))

        finalized_unknown = copy.deepcopy(self.smoke)
        finalized_unknown['unreviewed'] = True
        self._reject_finalized(smoke=finalized_unknown)
        self._reject_finalized(smoke=duplicates)

    def test_rgb_ambient_observation_is_three_channel_and_four_channel_claim_rejects(self):
        valid = self._validate_pending()
        self.assertEqual(len(self.pending['constructor_records'][0][
            'observed_configuration']['ambient_light_rgb']), 3)
        wrong = copy.deepcopy(self.pending)
        wrong['constructor_records'][0]['observed_configuration']['ambient_light_rgb'] = [
            0.02, 0.02, 0.02, 1.0]
        self._reject_pending(document=wrong)
        self.assertEqual(len(valid['render_events']), 18)

    def test_event_source_generation_network_and_array_joins_reject_stale_records(self):
        mutations = (
            ('missing event', lambda doc: doc['render_events'].pop()),
            ('failed network return', lambda doc: doc['render_events'][1]['network_inputs'].__setitem__(
                'call_returned', False)),
            ('wrong generation', lambda doc: doc['render_events'][2][
                'render_policy_metadata'].__setitem__('allocation_generation', 99)),
            ('stale runtime alias', lambda doc: doc['render_events'][0][
                'render_policy_metadata']['source_identities']['pyrender_renderer'].__setitem__(
                    'sha256', '0' * 64)),
            ('wrong crop array dtype', lambda doc: doc['render_events'][1]['arrays'][
                'depth_mm'].__setitem__('dtype', '<f8')),
            ('wrong query shape', lambda doc: doc['render_events'][1]['network_inputs'][
                'query_rgb_bchw'].__setitem__('shape', [1, 3, 279, 280])),
        )
        for label, mutate in mutations:
            with self.subTest(mutation=label):
                document = copy.deepcopy(self.pending)
                mutate(document)
                self._reject_pending(document=document)

    def test_pose_descriptor_joins_and_public_validation_thresholds_reject_mismatch(self):
        mutations = (
            ('wrong inverse translation units', lambda doc: doc['render_events'][0][
                'camera']['T_world_from_eye_mm'][0].__setitem__(3,
                    doc['render_events'][0]['camera']['T_world_from_eye_mm'][0][3] / 1000.0)),
            ('row digest does not join native array', lambda doc: doc['rows'][0]['render'].__setitem__(
                'gpu_depth_mm_sha256', '0' * 64)),
            ('wrong seed rotation', lambda doc: doc['rows'][0][
                'seed_camera_from_object_m'][0].__setitem__(0, 1.0)),
            ('wrong offset-row translation', lambda doc: doc['rows'][1][
                'seed_camera_from_object_m'][1].__setitem__(3,
                    doc['rows'][1]['seed_camera_from_object_m'][1][3] + 0.05)),
            ('insufficient inlier ratio', lambda doc: doc['rows'][0]['validation']['stats'].__setitem__(
                'inliers', 17)),
            ('insufficient spatial support', lambda doc: doc['rows'][1]['validation']['stats'].__setitem__(
                'spatial_support', 0.01)),
            ('retained count mismatch', lambda doc: doc['rows'][2]['validation']['stats'].__setitem__(
                'retained_correspondences', 31)),
        )
        for label, mutate in mutations:
            with self.subTest(mutation=label):
                document = copy.deepcopy(self.pending)
                mutate(document)
                self._reject_pending(document=document)

    def test_timing_phase_bounds_and_cpu_no_cuda_memory_are_enforced(self):
        too_short = copy.deepcopy(self.pending)
        too_short['runtime_evidence']['timings_ms']['worker_total'] = 1.0
        self._reject_pending(document=too_short)

        negative_mutations = [
            ('network_load', lambda timings: timings.__setitem__('network_load', -0.01)),
            ('renderer_construct', lambda timings: timings.__setitem__(
                'renderer_construct', -0.01)),
            ('renderer_close', lambda timings: timings.__setitem__('renderer_close', -0.01)),
        ]
        for case_id in ('a', 'b', 'c'):
            case_index = ('a', 'b', 'c').index(case_id)
            for phase in ('native_render', 'cpu_geometry', 'refine', 'validation'):
                negative_mutations.append((
                    f'{case_id}-{phase}',
                    lambda timings, row=case_index, field=phase:
                    timings['cases'][row].__setitem__(field, -0.01)))
        for label, mutate_timing in negative_mutations:
            with self.subTest(negative_duration=label):
                negative = copy.deepcopy(self.pending)
                mutate_timing(negative['runtime_evidence']['timings_ms'])
                self._reject_pending(document=negative)

        for render_index in (0, 1):
            with self.subTest(negative_render_duration=render_index):
                negative_render = copy.deepcopy(self.pending)
                negative_render['runtime_evidence']['render_calls'][render_index][
                    'elapsed_ms'] = -0.01
                self._reject_pending(document=negative_render)

        zero_durations = copy.deepcopy(self.pending)
        zero_timing = zero_durations['runtime_evidence']['timings_ms']
        for name in ('network_load', 'renderer_construct', 'renderer_close'):
            zero_timing[name] = 0.0
        for case in zero_timing['cases']:
            for name in ('native_render', 'cpu_geometry', 'refine', 'validation'):
                case[name] = 0.0
        for call in zero_durations['runtime_evidence']['render_calls']:
            call['elapsed_ms'] = 0.0
        self._validate_pending(document=zero_durations)

        for case_index in range(3):
            native_mismatch = copy.deepcopy(self.pending)
            native_index = case_index * 6
            native_mismatch['runtime_evidence']['render_calls'][native_index][
                'elapsed_ms'] = 3.0
            self._reject_pending(document=native_mismatch)

        for case_index in range(3):
            crops_exceed_refine = copy.deepcopy(self.pending)
            for index in range(case_index * 6 + 1, case_index * 6 + 6):
                crops_exceed_refine['runtime_evidence']['render_calls'][index][
                    'elapsed_ms'] = 11.0
            self._reject_pending(document=crops_exceed_refine)

        claims_cuda = copy.deepcopy(self.pending)
        claims_cuda['runtime_evidence']['memory']['cuda_total_bytes'] = 1024
        self._reject_pending(document=claims_cuda)

    def test_finalized_artifact_requires_exact_receipt_digest_and_cleanup_join(self):
        wrong_digest = copy.deepcopy(self.smoke)
        wrong_digest['renderer_execution_receipt_sha256'] = '0' * 64
        self._reject_finalized(smoke=wrong_digest)

        same_length_mutation = self.renderer_execution_bytes.replace(
            b'"terminal_exit_code":0', b'"terminal_exit_code":1', 1)
        self.assertEqual(len(same_length_mutation), len(self.renderer_execution_bytes))
        self._reject_finalized(execution=same_length_mutation)
        different_length_mutation = self.renderer_execution_bytes.replace(
            b'"execution_id":"synthetic-model-smoke-execution-001"',
            b'"execution_id":"synthetic-model-smoke-execution-changed"', 1)
        self.assertNotEqual(len(different_length_mutation), len(self.renderer_execution_bytes))
        self._reject_finalized(execution=different_length_mutation)

        failed_cleanup = copy.deepcopy(self.renderer_execution_bytes)
        failed_doc = json.loads(failed_cleanup.decode('utf-8'))
        failed_doc['cleanup']['process_reaped'] = False
        self._reject_finalized(execution=failed_doc)

        failed_terminal = copy.deepcopy(self.renderer_execution_bytes)
        failed_doc = json.loads(failed_terminal.decode('utf-8'))
        failed_doc['terminal_exit_code'] = 1
        self._reject_finalized(execution=failed_doc)

        self._reject_finalized(key='0' * 64)
        wrong_expected_recipe = _plain(self.resources.output.recipe)
        wrong_expected_recipe['render_flags']['unlit_templates'] = True
        self._reject_finalized(recipe=wrong_expected_recipe)
        wrong_expected_source = dict(self.resources.output.source_closure)
        source = next(iter(wrong_expected_source))
        wrong_expected_source[source] = '0' * 64
        self._reject_finalized(closure=wrong_expected_source)

    def test_public_bank_resource_path_requires_fixed_smoke_artifact_names(self):
        output = self.resources.output
        directory = output.sidecar_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'smoke.json').write_bytes(self.smoke_bytes)
        (directory / 'renderer-execution.json').write_bytes(self.renderer_execution_bytes)
        alias = directory / 'smoke-alias.json'
        alias.write_bytes(self.smoke_bytes)
        sidecar = {
            'schema_version': 1,
            'resource_kind': output.resource_kind,
            'resource_key': output.resource_key,
            'recipe': _plain(output.recipe),
            'dimensions': [3, 720, 720],
            'coordinate_mode': 'integer_centers_v1',
            'artifact': {
                'path': 'smoke.json', 'sha256': _sha(self.smoke_bytes),
                'byte_count': len(self.smoke_bytes),
            },
            'runtime': {'device': 'cpu', 'build': 'synthetic fixture'},
            'source_closure': _plain(output.source_closure),
        }
        output.sidecar_path.write_bytes(_canonical(sidecar))
        resources = assets.capture_stage_resources(
            self.plan, stage='banks', device='cpu')
        self.assertEqual(tuple(item.purpose for item in resources.required), ('smoke',))

        sidecar['artifact'] = {
            'path': alias.name, 'sha256': _sha(self.smoke_bytes),
            'byte_count': len(self.smoke_bytes),
        }
        output.sidecar_path.write_bytes(_canonical(sidecar))
        with self.assertRaises((ValueError, capture.CaptureIntegrityError)):
            assets.capture_stage_resources(self.plan, stage='banks', device='cpu')

    def test_reread_artifact_bytes_must_still_match_sidecar_hash_and_length(self):
        output = self.resources.output
        directory = output.sidecar_path.parent
        smoke_path = directory / 'smoke.json'
        execution_path = directory / 'renderer-execution.json'
        directory.mkdir(parents=True, exist_ok=True)
        execution_path.write_bytes(self.renderer_execution_bytes)
        sidecar = {
            'schema_version': 1,
            'resource_kind': output.resource_kind,
            'resource_key': output.resource_key,
            'recipe': _plain(output.recipe),
            'dimensions': [3, 720, 720],
            'coordinate_mode': 'integer_centers_v1',
            'artifact': {
                'path': 'smoke.json', 'sha256': _sha(self.smoke_bytes),
                'byte_count': len(self.smoke_bytes),
            },
            'runtime': {'device': 'cpu', 'build': 'synthetic fixture'},
            'source_closure': _plain(output.source_closure),
        }
        output.sidecar_path.write_bytes(_canonical(sidecar))
        original_verify = capture.verify_stage_resource_sidecar
        mutations = (
            ('same-length-content', self.smoke_bytes.replace(
                b'"mask_iou":0.97', b'"mask_iou":0.96', 1)),
            ('changed-length-whitespace', self.smoke_bytes + b' '),
        )
        self.assertEqual(len(mutations[0][1]), len(self.smoke_bytes))
        self.assertNotEqual(len(mutations[1][1]), len(self.smoke_bytes))
        for label, mutated_bytes in mutations:
            with self.subTest(mutation=label):
                smoke_path.write_bytes(self.smoke_bytes)

                def verify_then_mutate(*args, **kwargs):
                    descriptor = original_verify(*args, **kwargs)
                    smoke_path.write_bytes(mutated_bytes)
                    return descriptor

                with (
                    patch.object(capture, 'verify_stage_resource_sidecar',
                                 side_effect=verify_then_mutate),
                    self.assertRaises((ValueError, capture.CaptureIntegrityError)),
                ):
                    assets.capture_stage_resources(
                        self.plan, stage='banks', device='cpu')


if __name__ == '__main__':
    unittest.main()
