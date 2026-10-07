"""Isolated model stages. This module never opens evaluation annotations."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from collections.abc import Mapping
from types import MappingProxyType
import numpy as np
from .quality_assets import CACHE, CAPTURE_CACHE, BUDGET, digest, save_result, inference_provenance
from .quality_contract import Frame, MaskPrediction, SequentialTracker
from .quality_time import LEGACY, PHYSICAL, FrameKey, parse_timing_manifest
from .vision import cv2


_LEGACY_INPUTS = {
    '3CFC13197E096739698B5867383CD34808EB571FFA5A2E387C17E84819D00B80': {
        'object': 'keyboard',
        'source_hashes': {
            'source.mp4': 'dd638f8f8de4413bfbbd92e8e21e0451c13efd7fb29b1f5d5e89f12040546a02',
            'object.glb': '2e20d137a3357a30c978447be48134fecc3c294a9f908bf6f1767e5efee7313e',
        },
    },
    '4422C7373A09564112B0069153AAB40C803148F109EACEAA737BFB0398A268A3': {
        'object': 'mug',
        'source_hashes': {
            'source.mp4': 'ad9e6c4d70adfa62889b08a9f31e0d98a781619f86c5efdce48caef482be3844',
            'object.glb': '26b2884fa61a9b9870ae089de04ea9bccdf8eb8e1467bdd185e7d514ae259195',
        },
    },
    'F012F28918534F2A7AC3EFD59841A347A8BAD384D9585130DB88EAAC56377D5D': {
        'object': 'ranch',
        'source_hashes': {
            'source.mp4': '6dcd8c1dda7381d69db178e2b619447506cfcce6cfe4afce8666a43c6363283e',
            'object.glb': 'ed3ac3767202c8da56d30051d766bf7d444d3d356c30671f4bbb2a73e3a1b321',
        },
    },
}


def read_input(bundle):
    bundle = Path(bundle)
    with (bundle/'input.json').open('rb') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('Input manifest exceeds the 2 MiB timing limit')
    timing = parse_timing_manifest(raw)
    manifest = json.loads(raw.decode('utf-8'))
    rgb_source = manifest.get('rgb_source')
    if isinstance(rgb_source, dict) and rgb_source.get('kind') == 'rectified_image_sequence_v1':
        from . import quality_capture
        verified = quality_capture.load_capture_plan(bundle, raw)
        detached = _detach_capture_value(verified.manifest)
        detached['_timing'] = verified.timing
        detached['_capture'] = verified
        return detached
    if timing['clock_mode'] == LEGACY:
        pin = _LEGACY_INPUTS.get(timing['manifest_sha256'])
        if pin is None or manifest.get('object') != pin['object']:
            raise ValueError('Untimed runner input is not one of the authenticated historical manifests')
        if manifest.get('source_hashes') != pin['source_hashes']:
            raise ValueError('Historical input source hashes differ from their authenticated pins')
    source_hashes = manifest.get('source_hashes')
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError('Inference input requires source hashes')
    for name, sha in source_hashes.items():
        if not isinstance(name, str) or not name or not isinstance(sha, str) or len(sha) != 64 or any(
                c not in '0123456789abcdefABCDEF' for c in sha):
            raise ValueError('Inference source hash entries must bind relative paths to SHA-256 digests')
        resource = (bundle/name).resolve()
        if not resource.is_relative_to(bundle.resolve()) or not resource.is_file():
            raise ValueError('Inference source path is missing or escapes its input bundle')
        if digest(resource).upper() != sha.upper():
            raise ValueError('Inference source hash mismatch')
    if timing['clock_mode'] == PHYSICAL:
        for resource_name in ('video', 'asset'):
            relative = manifest.get(resource_name)
            if not isinstance(relative, str) or relative not in source_hashes:
                raise ValueError(f'Physical input must hash its declared {resource_name}')
    if manifest.get('units') != 'metres': raise ValueError('Metric asset required')
    ids = manifest['frame_ids']
    if ids != [key.frame_id for key in timing['scored_keys']]:
        raise ValueError('Input frame_ids differ from the validated timing table')
    if timing['clock_mode'] == LEGACY and ids != list(range(ids[0], ids[0]+len(ids))):
        raise ValueError('Contiguous original frames required')
    manifest['_timing'] = timing
    return manifest


def _detach_capture_value(value):
    """Copy authenticated plan metadata into ordinary containers for inference."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: _detach_capture_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_detach_capture_value(child) for child in value]
    return value


def read_decoded_frame(bundle, manifest, ordinal, *, reader=None):
    """Return the already-authorized immutable sequence record without advancing it."""
    capture_plan = manifest.get('_capture')
    if capture_plan is None:
        raise ValueError('read_decoded_frame is available only for authenticated capture sequences')
    from . import quality_capture
    if type(capture_plan) is not quality_capture.VerifiedCapturePlan:
        raise TypeError('Capture frame access requires an issued VerifiedCapturePlan')
    quality_capture._validate_verified_capture_plan_record(capture_plan)
    if type(ordinal) is not int or ordinal < 0 or ordinal >= len(capture_plan.rows):
        raise ValueError('Capture ordinal must be a non-bool planned integer')
    if reader is None:
        raise ValueError('Capture frame access requires the caller-owned CaptureReader')
    if type(reader) is not quality_capture.CaptureReader:
        raise TypeError('Capture frame access requires a CaptureReader from the reviewed module')
    if reader.plan is not capture_plan:
        raise quality_capture.CaptureIntegrityError('CaptureReader belongs to a different verified plan')
    try:
        bundle_path = Path(bundle).resolve(strict=True)
        verified_bundle = Path(capture_plan.bundle).resolve(strict=True)
        reader_bundle = Path(reader.bundle).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise quality_capture.CaptureIntegrityError('CaptureReader bundle cannot be resolved') from exc
    if bundle_path != verified_bundle or reader_bundle != verified_bundle:
        raise quality_capture.CaptureIntegrityError('CaptureReader bundle differs from the inference bundle')
    record = reader.read(ordinal)
    if type(record) is not quality_capture.DecodedCapture:
        raise quality_capture.CaptureIntegrityError('CaptureReader did not return a reviewed immutable frame record')
    return record


def read_rgb(bundle, manifest, frame_id):
    rgb_source = manifest.get('rgb_source')
    if (manifest.get('_capture') is not None or
            isinstance(rgb_source, dict) and rgb_source.get('kind') == 'rectified_image_sequence_v1'):
        raise ValueError('Capture sequences require read_decoded_frame so sampling validity is preserved')
    if manifest.get('_timing', {}).get('clock_mode') == PHYSICAL:
        try:
            source_frame_id = manifest['_timing']['source_frame_ids'][frame_id]
        except (KeyError, TypeError) as exc:
            raise ValueError('Unknown physical inference ordinal') from exc
    else:
        source_frame_id = frame_id
    cap = cv2.VideoCapture(str(bundle/manifest['video']))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_id); ok, bgr = cap.read()
        if not ok: raise ValueError('Missing source image')
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally: cap.release()


def _frame_key(manifest, frame_id):
    timing = manifest['_timing']
    if timing['clock_mode'] == LEGACY:
        return FrameKey(frame_id, None, LEGACY)
    setup_key = timing['setup_key']
    if setup_key is not None and setup_key.frame_id == frame_id:
        return setup_key
    for key in timing['scored_keys']:
        if key.frame_id == frame_id:
            return key
    raise ValueError('Unknown inference frame ordinal')


def _read_json_unique(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError(f'Duplicate JSON key: {key!r}')
            value[key] = item
        return value
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid JSON file: {path}') from exc


def _physical_mask_cache(manifest, masks_root):
    """Validate every physical setup/scored mask row before model loading."""
    capture_plan = manifest.get('_capture')
    if capture_plan is not None:
        from . import quality_capture
        if type(capture_plan) is not quality_capture.VerifiedCapturePlan:
            raise TypeError('Capture mask cache requires an issued VerifiedCapturePlan')
        quality_capture._validate_verified_capture_plan_record(capture_plan)
    timing = manifest['_timing']
    masks_root = Path(masks_root).resolve()
    record = _read_json_unique(masks_root/'results.json')
    if not isinstance(record, dict): raise ValueError('Physical mask manifest must be a JSON object')
    expected_keys = [FrameKey(row['frame_id'], row['timestamp_s'], PHYSICAL)
                     for row in manifest['timeline']]
    if type(record.get('schema_version')) is not int or record.get('schema_version') != 2 or record.get('clock_mode') != PHYSICAL:
        raise ValueError('Physical pose requires a schema2 physical-time mask cache')
    if record.get('input_manifest_sha256') != timing['manifest_sha256']:
        raise ValueError('Mask cache is bound to a different input manifest')
    if record.get('timestamp_table_sha256') != timing['timestamp_table_sha256']:
        raise ValueError('Mask cache is bound to a different timestamp table')
    if capture_plan is not None:
        from . import quality_capture
        if (record.get('source_kind') != quality_capture.SOURCE_KIND or
                record.get('capture_table_sha256') != capture_plan.capture_table_sha256 or
                record.get('coordinate_mode') != quality_capture.COORDINATE_MODE):
            raise ValueError('Capture mask cache is bound to a different source table or coordinate mode')
    if record.get('coverage_complete') is not True:
        raise ValueError('Mask cache does not cover every planned input row')
    if 'mask_association' in record:
        if type(record['mask_association']) is not bool:
            raise ValueError('Physical mask_association must be a bool when declared')
        if record['mask_association']:
            raise ValueError('Physical mask association is unsupported while prior age uses frame count')
    status = record.get('status')
    if status not in ('complete', 'diagnostic_prefix', 'failed'):
        raise ValueError('Mask cache is running or has an unsupported status')
    stage_completed = record.get('stage_completed')
    if type(stage_completed) is not bool:
        raise ValueError('Mask cache must declare stage_completed')
    if (status in ('complete', 'diagnostic_prefix')) != stage_completed:
        raise ValueError('Mask cache status conflicts with stage_completed')
    setup_count = sum(row['role'] == 'setup' for row in manifest['timeline'])
    scored_count = sum(row['role'] == 'scored' for row in manifest['timeline'])
    if (type(record.get('expected_setup_frames')) is not int or
            type(record.get('expected_scored_frames')) is not int or
            record.get('expected_setup_frames') != setup_count or
            record.get('expected_scored_frames') != scored_count):
        raise ValueError('Mask cache expected row counts differ from the input timing table')
    planned_ids = [key.frame_id for key in expected_keys]
    declared_planned = record.get('planned_frame_ids')
    if (not isinstance(declared_planned, list) or
            any(type(value) is not int for value in declared_planned) or
            declared_planned != planned_ids):
        raise ValueError('Mask cache planned frame ordinals differ from the input timing table')
    requested = record.get('requested_frame_ids')
    if (not isinstance(requested, list) or not requested or
            any(type(value) is not int for value in requested) or
            requested != planned_ids[:len(requested)]):
        raise ValueError('Mask cache requested IDs must be a nonempty planned prefix')
    if len(requested) > len(planned_ids):
        raise ValueError('Mask cache requests frame IDs outside the input timing table')
    if status == 'complete' and requested != planned_ids:
        raise ValueError('A complete mask cache must request every planned input row')
    if status == 'diagnostic_prefix' and len(requested) >= len(planned_ids):
        raise ValueError('A diagnostic_prefix cache must be a strict prefix')
    rows = record.get('frames')
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise ValueError('Mask cache must retain exactly one row for every planned input row')

    by_id = {}
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Mask cache frame rows must be objects')
        frame_id = row.get('frameId')
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id in by_id:
            raise ValueError('Mask cache frame IDs must be unique integer ordinals')
        by_id[frame_id] = row
    if set(by_id) != set(planned_ids):
        raise ValueError('Mask cache has missing or unexpected planned frame rows')

    expected_rows = {row['frame_id']: row for row in manifest['timeline']}
    capture_rows = ({} if capture_plan is None else
                    {row['frame_id']: row for row in capture_plan.rows})
    root_path = masks_root
    for key in expected_keys:
        row = by_id[key.frame_id]
        expected = expected_rows[key.frame_id]
        for field, value in (
            ('frameId', key.frame_id),
            ('sourceFrameId', expected['source_frame_id']),
            ('timestamp_s', key.timestamp_s),
            ('clock_mode', PHYSICAL),
            ('role', expected['role']),
        ):
            actual = row.get(field)
            if field in ('frameId', 'sourceFrameId') and type(actual) is not int:
                raise ValueError(f'Mask cache row {key.frame_id} has an invalid {field}')
            if field == 'timestamp_s':
                try:
                    finite_timestamp = (not isinstance(actual, bool) and
                        isinstance(actual, (int, float)) and math.isfinite(float(actual)))
                except (OverflowError, TypeError, ValueError):
                    finite_timestamp = False
                if not finite_timestamp or actual != value:
                    raise ValueError(f'Mask cache row {key.frame_id} has a timestamp mismatch')
            elif actual != value:
                raise ValueError(f'Mask cache row {key.frame_id} has a {field} mismatch')
        if capture_plan is not None:
            table_row = capture_rows.get(key.frame_id)
            if table_row is None:
                raise ValueError(f'Capture plan lacks mask cache frame {key.frame_id}')
            output_rgb = table_row.get('output_rgb')
            expected_binding = {
                'capture_table_sha256': capture_plan.capture_table_sha256,
                'capture_row_sha256': table_row['capture_row_sha256'],
                'rgb_pixel_sha256': None if output_rgb is None else output_rgb.get('pixel_sha256'),
                'sampling_valid_sha256': table_row.get('sampling_valid_sha256'),
                'warp_sha256': table_row.get('warp_sha256'),
                'calibration_sha256': table_row.get('calibration_values_sha256'),
            }
            binding = row.get('capture_binding')
            if (not isinstance(binding, dict) or set(binding) != set(expected_binding) or
                    any(binding.get(name) != expected
                        for name, expected in expected_binding.items())):
                raise ValueError(f'Mask cache row {key.frame_id} has an invalid capture binding')
            for name in ('capture_table_sha256', 'capture_row_sha256'):
                value = binding[name]
                if (type(value) is not str or len(value) != 64 or value != value.upper() or
                        any(character not in '0123456789ABCDEF' for character in value)):
                    raise ValueError(f'Mask cache row {key.frame_id} has an invalid {name}')
            for name in ('rgb_pixel_sha256', 'sampling_valid_sha256', 'warp_sha256', 'calibration_sha256'):
                value = binding[name]
                if value is not None and (
                        type(value) is not str or len(value) != 64 or value != value.upper() or
                        any(character not in '0123456789ABCDEF' for character in value)):
                    raise ValueError(f'Mask cache row {key.frame_id} has an invalid nullable {name}')
            if table_row['conversion_state'] == 'available' and any(
                    binding[name] is None for name in expected_binding):
                raise ValueError(f'Available capture row {key.frame_id} lacks six actual binding digests')
        state = row.get('mask_state')
        measurement = row.get('measurement_state')
        path_value = row.get('path')
        sha = row.get('mask_sha256')
        reason = row.get('failure_reason')
        capture_unavailable = (
            capture_plan is not None and
            capture_rows[key.frame_id]['conversion_state'] == 'unavailable'
        )
        if capture_unavailable and (
                state != 'lost' or measurement != 'unmeasured' or
                path_value is not None or sha is not None):
            raise ValueError('Unavailable capture inputs cannot claim a measured mask or artifact')
        if state == 'available':
            if measurement != 'measured' or not isinstance(path_value, str) or not path_value:
                raise ValueError('Available physical masks require measured artifact rows')
        elif state == 'lost':
            if not isinstance(reason, str) or not reason:
                raise ValueError('Lost physical mask rows require a typed reason')
            if measurement == 'unmeasured':
                if path_value is not None or sha is not None:
                    raise ValueError('Unmeasured physical rows cannot reference mask artifacts')
            elif measurement == 'measured':
                if not isinstance(path_value, str) or not path_value:
                    raise ValueError('Measured lost rows require their actual mask artifact')
            else:
                raise ValueError('Physical mask measurement_state is invalid')
        else:
            raise ValueError('Physical mask_state must be available or lost')

        if path_value is not None:
            rel = Path(path_value)
            if rel.is_absolute(): raise ValueError('Mask artifact path must be relative to its cache')
            artifact = (root_path/rel).resolve()
            if not artifact.is_relative_to(root_path) or not artifact.is_file():
                raise ValueError('Mask artifact is missing or escapes its cache directory')
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789ABCDEFabcdef' for c in sha):
                raise ValueError('Mask artifact requires a SHA-256 digest')
            if digest(artifact).upper() != sha.upper():
                raise ValueError('Mask artifact hash mismatch')
        elif measurement != 'unmeasured' or sha is not None:
            raise ValueError('Only explicit unmeasured rows may omit mask artifacts')
        if key.frame_id not in requested:
            if (state != 'lost' or measurement != 'unmeasured' or path_value is not None or
                    reason != 'outside_requested_prefix'):
                raise ValueError('Rows outside the requested prefix must remain explicit unmeasured failures')

    return record, by_id


def _load_mask_cache(manifest, masks_root):
    if manifest['_timing']['clock_mode'] == PHYSICAL:
        return _physical_mask_cache(manifest, masks_root)
    record = _read_json_unique(Path(masks_root)/'results.json')
    return record, {row['frameId']: row for row in record['frames']}


def device_info(device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; rerun with --device cpu')
    return dict(device=device, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=None if device == 'cpu' else torch.cuda.get_device_name(0),
                precision='float32', batch_size=1, iterations=5, crop_size=[280, 280],
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                math_threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                torch_cpu_threads=torch.get_num_threads())


def _pnp_tracking_settings(bundle_object, pnp_no_extrinsic_guess):
    if type(pnp_no_extrinsic_guess) is not bool:
        raise TypeError('pnp_no_extrinsic_guess must be a bool')
    if pnp_no_extrinsic_guess and bundle_object != 'mug':
        raise ValueError('--pnp-no-extrinsic-guess is supported only for the mug bundle')
    return {'pnp_use_extrinsic_guess': not pnp_no_extrinsic_guess}


def _make_gotrack_refiner(network, renderer, object_id, device, pnp_settings,
                          chronological_capture_callback=None):
    from .quality_gotrack import GoTrackRefiner
    if chronological_capture_callback is None:
        return GoTrackRefiner(network, renderer, object_id, device,
            pnp_use_extrinsic_guess=pnp_settings['pnp_use_extrinsic_guess'])
    return GoTrackRefiner(network, renderer, object_id, device,
        pnp_use_extrinsic_guess=pnp_settings['pnp_use_extrinsic_guess'],
        chronological_capture_callback=chronological_capture_callback)


def smoke(bundle, output, device, unlit_templates=False, disable_multisampling=False):
    # The raw timing table and all inference-source hashes are validated before
    # loading the network, renderer, or decoder.
    manifest = read_input(bundle)
    if not isinstance(manifest.get('controlled_initial_pose'), list):
        raise ValueError('Setup smoke requires a declared controlled_initial_pose')
    setup_ordinal = manifest.get('setup_frame_id')
    if setup_ordinal is None:
        if manifest['_timing']['clock_mode'] == PHYSICAL:
            setup_ordinal = 0
        else:
            raise ValueError('Legacy setup smoke requires its setup frame ID')
    from .quality_gotrack import load_network, TexturedRenderer, GoTrackRefiner
    from .glb_model import read_glb
    from .show3d_experiment import camera_for
    from .renderer import render
    from utils import structs, renderer_base
    network = load_network(device)
    renderer = TexturedRenderer(bundle/manifest['asset'], manifest['object_id'], unlit=unlit_templates, disable_multisampling=disable_multisampling)
    pose = np.array(manifest['controlled_initial_pose']); k = np.array(manifest['intrinsics'])
    # A supplied setup pose is legitimate for renderer validation only.
    # It never enters complete-pipeline initialization.
    h = 280; w = round(manifest['native_resolution'][0]*h/manifest['native_resolution'][1]); scale = h/manifest['native_resolution'][1]
    small_k = k.copy(); small_k[:2] *= scale
    camera = structs.PinholePlaneCameraModel(width=w, height=h, f=(small_k[0, 0], small_k[1, 1]),
        c=(small_k[0, 2], small_k[1, 2]), T_world_from_eye=np.linalg.inv(pose.copy()))
    camera.T_world_from_eye[:3, 3] *= 1000
    started = time.perf_counter()
    try:
        rendered = renderer.render_object_model(manifest['object_id'], camera)
        mesh = read_glb(bundle/manifest['asset'], manifest['object'])
        control = render(mesh, camera_for((pose[:3, :3], pose[:3, 3]), small_k, w, h))
        pred = rendered[renderer_base.RenderType.MASK]; truth = control.triangle >= 0
        iou = float(np.count_nonzero(pred & truth)/max(1, np.count_nonzero(pred | truth)))
        common = pred & truth
        error = float(np.median(np.abs(rendered[renderer_base.RenderType.DEPTH][common]*.001-control.depth[common]))) if common.any() else None
        extent_ok = bool(np.allclose(np.ptp(renderer.vertices_m, axis=0), np.ptp(mesh.positions, axis=0), atol=1e-5))
        if iou < .94 or error is None or error > .002 or not extent_ok:
            raise RuntimeError(f'Renderer/unit smoke gate failed: IoU {iou}, median depth error {error}, extent {extent_ok}')
        rgb = read_rgb(bundle, manifest, setup_ordinal)
        # Smoke mask is a labelled geometric control, never scored segmentation.
        native_camera = structs.PinholePlaneCameraModel(width=rgb.shape[1], height=rgb.shape[0], f=(k[0, 0], k[1, 1]),
            c=(k[0, 2], k[1, 2]), T_world_from_eye=camera.T_world_from_eye)
        mask = renderer.render_object_model(manifest['object_id'], native_camera)[renderer_base.RenderType.MASK]
        smoke_key = _frame_key(manifest, setup_ordinal)
        smoke_frame = Frame(smoke_key.frame_id, rgb, k, smoke_key.timestamp_s, smoke_key.clock_mode)
        candidate = GoTrackRefiner(network, renderer, manifest['object_id'], device).refine(smoke_frame, mask, pose)
        if candidate is None: raise RuntimeError('Network loaded but could not refine setup smoke frame')
        result = dict(checkpoint_loaded=True, renderer_iou=iou, renderer_median_depth_error_m=error,
            metric_extent_verified=extent_ok, inference_completed=True, smoke_passed=True,
            runtime=device_info(device), unlit_templates=unlit_templates, disable_multisampling=disable_multisampling, elapsed_ms=(time.perf_counter()-started)*1000,
            scope='Setup-only renderer and inference smoke; projected mask control is NOT foreground prediction or benchmark accuracy.')
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2))
    finally: renderer.close()


_MODEL_SMOKE_CASES = (
    ('a', (9., -13., 7.), (.02, -.04, 3.)),
    ('b', (-19., 24., -12.), (-.17, .11, 3.15)),
    ('c', (27., 15., 21.), (.15, -.126, 3.3)),
)
_MODEL_SMOKE_ROTATION = 'Rx@Ry@Rz@diag(1,-1,-1)'
_MODEL_SMOKE_NATIVE_K = (
    (360., 0., 359.5),
    (0., 360., 359.5),
    (0., 0., 1.),
)
_MODEL_SMOKE_ARRAY_PREFIX = b'quality-model-smoke-array-v1\0'
_MODEL_SMOKE_RENDER_TYPES_NATIVE = ('rgb', 'depth', 'mask')
_MODEL_SMOKE_RENDER_TYPES_CROP = ('rgb', 'mask', 'depth')
_MODEL_SMOKE_METADATA_KEYS = frozenset({
    'schema_version', 'render_policy', 'coordinate_mode', 'depth_units',
    'dimensions', 'allocation_generation', 'offscreen_identity',
    'framebuffer_fields', 'allocation_pair', 'allocation_calls',
    'framebuffer_complete', 'gl_samples', 'gl_sample_buffers',
    'framebuffer_bindings_restored', 'current_context_released',
    'dimension_match', 'color_storage_policy', 'depth_storage_policy',
    'source_identities',
})


def _model_smoke_plain(value):
    """Detach a small authenticated/provenance tree into strict JSON values."""
    if isinstance(value, Mapping):
        detached = {}
        for key, child in value.items():
            if type(key) is not str or key in detached:
                raise ValueError('Model smoke metadata keys must be unique strings')
            detached[key] = _model_smoke_plain(child)
        return detached
    if isinstance(value, (tuple, list)):
        return [_model_smoke_plain(child) for child in value]
    if isinstance(value, np.generic):
        return _model_smoke_plain(value.item())
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Model smoke metadata cannot contain nonfinite numbers')
        return value
    raise ValueError(f'Model smoke metadata contains unsupported {type(value).__name__}')


def _model_smoke_freeze(value):
    """Recursively detach public renderer metadata into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _model_smoke_freeze(child)
                                 for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_model_smoke_freeze(child) for child in value)
    if isinstance(value, np.generic):
        return _model_smoke_freeze(value.item())
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('Public render-policy metadata is not a finite detached value tree')


def _model_smoke_snapshot_array(value, *, name):
    """Return a detached C-order NumPy snapshot backed only by immutable bytes."""
    try:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f'{name} cannot use object dtype')
        copied = np.array(array, copy=True, order='C', subok=False)
        backing = copied.tobytes(order='C')
        snapshot = np.frombuffer(backing, dtype=copied.dtype).reshape(copied.shape)
    except Exception as exc:
        raise ValueError(f'{name} cannot be snapshotted as a NumPy array') from exc
    return snapshot


def _model_smoke_json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError('Model smoke descriptor metadata is not canonical JSON') from exc


def _model_smoke_array_descriptor(value):
    array = np.asarray(value)
    if (array.ndim < 1 or any(type(size) is not int or size <= 0
                               for size in array.shape) or
            array.dtype.hasobject or array.dtype.kind not in 'biuf'):
        raise ValueError('Model smoke descriptors require positive numeric/bool array dimensions')
    dtype = array.dtype.str
    if not isinstance(dtype, str) or not dtype or dtype[0] not in '<>|':
        raise ValueError('Model smoke array dtype must include an explicit byte order')
    shape = [int(size) for size in array.shape]
    metadata = {'shape': shape, 'dtype': dtype, 'order': 'C'}
    raw = np.ascontiguousarray(array).tobytes(order='C')
    digest_value = hashlib.sha256(
        _MODEL_SMOKE_ARRAY_PREFIX + _model_smoke_json_bytes(metadata) + b'\0' + raw
    ).hexdigest().upper()
    return {**metadata, 'sha256': digest_value}


def _model_smoke_camera_snapshot(camera):
    try:
        width, height = camera.width, camera.height
        fx, fy = camera.f
        cx, cy = camera.c
        pose = np.asarray(camera.T_world_from_eye, dtype=np.float64)
    except Exception as exc:
        raise ValueError('Model smoke render camera is missing public pinhole fields') from exc
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or
            pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0., 0., 0., 1.]) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
            abs(float(np.linalg.det(pose[:3, :3])) - 1.) > 1e-4 or
            not all(math.isfinite(float(item)) for item in (fx, fy, cx, cy)) or
            float(fx) <= 0. or float(fy) <= 0.):
        raise ValueError('Model smoke render camera must be finite with proper pose and positive intrinsics')
    return {
        'width': width, 'height': height,
        'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy),
        'T_world_from_eye_mm': pose.tolist(),
    }


def _model_smoke_error(exc):
    kind = type(exc).__name__[:128]
    message = str(exc).replace('\0', '')[:512]
    return {'type': kind, 'message': message}


def _model_smoke_timing_start(torch, device):
    if device == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter_ns()


def _model_smoke_timing_finish(torch, device, started_ns):
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000.
    if not math.isfinite(elapsed) or elapsed < 0.:
        raise RuntimeError('Model smoke monotonic timing produced an invalid duration')
    return elapsed


def _model_smoke_process_peak_rss():
    if os.name != 'nt':
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb)
        value = int(counters.PeakWorkingSetSize)
        if succeeded and value > 0:
            return value, 'GetProcessMemoryInfo.PeakWorkingSetSize'
    except Exception:
        pass
    return None, None


def _model_smoke_memory_evidence(device):
    return {
        'measurement': ('torch-cuda-allocator-v1' if device == 'cuda' else
                        'cpu-no-cuda'),
        'cuda_total_bytes': None,
        'cuda_peak_allocated_bytes': None,
        'cuda_peak_reserved_bytes': None,
        'process_peak_rss_bytes': None,
        'process_peak_rss_method': None,
    }


def _model_smoke_collect_memory(runtime_evidence, device, torch):
    memory = runtime_evidence['memory']
    rss_bytes, rss_method = _model_smoke_process_peak_rss()
    memory['process_peak_rss_bytes'] = rss_bytes
    memory['process_peak_rss_method'] = rss_method
    if device == 'cuda':
        torch.cuda.synchronize()
        memory['cuda_peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        memory['cuda_peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())


def _model_smoke_runtime_evidence_error(runtime_evidence, device):
    timings = runtime_evidence['timings_ms']
    if (type(runtime_evidence) is not dict or set(runtime_evidence) != {
            'schema_version', 'scope', 'clock', 'cuda_synchronization',
            'timings_ms', 'render_calls', 'memory'} or
            runtime_evidence['schema_version'] != 1 or
            runtime_evidence['scope'] != 'model-only-smoke' or
            runtime_evidence['clock'] != 'perf_counter_ns' or
            runtime_evidence['cuda_synchronization'] !=
                ('phase-boundaries' if device == 'cuda' else 'not-applicable') or
            type(timings) is not dict or set(timings) != {
                'worker_total', 'network_load', 'renderer_construct',
                'renderer_close', 'cases'}):
        return 'Model smoke runtime evidence does not match its frozen schema'
    required_times = [timings[name] for name in
                      ('worker_total', 'network_load', 'renderer_construct', 'renderer_close')]
    if (any(type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0.
            for value in required_times) or timings['worker_total'] <= 0.):
        return 'Model smoke runtime evidence has an incomplete top-level timing prefix'
    cases = timings['cases']
    if type(cases) is not list or len(cases) != 3:
        return 'Model smoke runtime evidence must contain exactly three ordered case timings'
    for index, case in enumerate(cases):
        if (type(case) is not dict or set(case) != {
                'case_id', 'native_render', 'cpu_geometry', 'refine', 'validation'} or
                case['case_id'] != _MODEL_SMOKE_CASES[index][0] or
                any(type(case[name]) not in (int, float) or
                    not math.isfinite(float(case[name])) or case[name] < 0.
                    for name in ('native_render', 'cpu_geometry', 'refine', 'validation'))):
            return 'Model smoke runtime evidence has an incomplete case timing prefix'
    calls = runtime_evidence['render_calls']
    if type(calls) is not list or len(calls) != 18:
        return 'Model smoke runtime evidence must contain exactly 18 render-call timings'
    for index, call in enumerate(calls):
        if (type(call) is not dict or set(call) != {'render_index', 'elapsed_ms'} or
                type(call['render_index']) is not int or call['render_index'] != index or
                type(call['elapsed_ms']) not in (int, float) or
                not math.isfinite(float(call['elapsed_ms'])) or call['elapsed_ms'] < 0.):
            return 'Model smoke runtime evidence has an invalid render-call timing record'
    for index, case in enumerate(cases):
        if case['native_render'] != calls[index * 6]['elapsed_ms']:
            return 'Case native_render must reuse its exact observer render-call sample'
    phase_sum = sum(required_times[1:]) + sum(
        case[name] for case in cases
        for name in ('native_render', 'cpu_geometry', 'refine', 'validation'))
    if phase_sum > timings['worker_total'] + 1.:
        return 'Model smoke non-overlapping phase timings exceed worker_total by over 1 ms'
    memory = runtime_evidence['memory']
    if type(memory) is not dict or set(memory) != {
            'measurement', 'cuda_total_bytes', 'cuda_peak_allocated_bytes',
            'cuda_peak_reserved_bytes', 'process_peak_rss_bytes',
            'process_peak_rss_method'}:
        return 'Model smoke runtime memory evidence does not match its frozen schema'
    rss_bytes = memory['process_peak_rss_bytes']
    rss_method = memory['process_peak_rss_method']
    if rss_bytes is None or rss_method is None:
        if rss_bytes is not None or rss_method is not None:
            return 'Process peak RSS bytes and method must be jointly available or null'
    elif (type(rss_bytes) is not int or rss_bytes <= 0 or
          type(rss_method) is not str or not rss_method):
        return 'Process peak RSS evidence must contain positive bytes and its method string'
    if device == 'cpu':
        if (memory['measurement'] != 'cpu-no-cuda' or
                any(memory[name] is not None for name in
                    ('cuda_total_bytes', 'cuda_peak_allocated_bytes',
                     'cuda_peak_reserved_bytes'))):
            return 'CPU model smoke must leave every CUDA memory field null'
    else:
        cuda_bytes = [memory[name] for name in
                      ('cuda_total_bytes', 'cuda_peak_allocated_bytes',
                       'cuda_peak_reserved_bytes')]
        if (memory['measurement'] != 'torch-cuda-allocator-v1' or
                any(type(value) is not int or value <= 0 for value in cuda_bytes) or
                not cuda_bytes[1] <= cuda_bytes[2] <= cuda_bytes[0]):
            return 'CUDA model smoke requires ordered positive total/allocator peak bytes'
    return None


class ModelSmokeRenderObserver:
    """Observe exactly the public renderer calls made by one model-only smoke."""

    def __init__(self, renderer, sink):
        if not callable(sink):
            raise TypeError('Model smoke render observer sink must be callable')
        self._renderer = renderer
        self._sink = sink
        self._vertices_m = _model_smoke_snapshot_array(
            renderer.vertices_m, name='renderer vertices_m')
        self._scope = None
        self._inside_render = False
        self._close_called = False
        self._timing_torch = None
        self._timing_device = 'cpu'
        self._render_call_records = []
        self.renderer_closed = False
        self.last_observation = None
        self.last_event = None

    @property
    def vertices_m(self):
        return self._vertices_m

    @property
    def obj_id(self):
        return self._renderer.obj_id

    @property
    def render_policy(self):
        return self._renderer.render_policy

    @property
    def coordinate_mode(self):
        return self._renderer.coordinate_mode

    @property
    def render_policy_metadata(self):
        metadata = self._renderer.render_policy_metadata
        return None if metadata is None else _model_smoke_freeze(metadata)

    @property
    def unlit(self):
        return self._renderer.unlit

    @property
    def disable_multisampling(self):
        return self._renderer.disable_multisampling

    @property
    def current_scope(self):
        if self._scope is None:
            return None
        scope = self._scope
        return MappingProxyType({
            'case_id': scope['case_id'], 'ordinal': scope['ordinal'],
            'role': scope['role'],
            'iteration': (scope['attempts'] - 1 if scope['role'] == 'crop' and
                          scope['attempts'] else None),
        })

    @contextmanager
    def _observe_scope(self, role, case_id, ordinal):
        if self._scope is not None:
            raise RuntimeError('Model smoke render observation scopes cannot overlap')
        if type(role) is not str or role not in ('native', 'crop'):
            raise ValueError('Model smoke render scope role must be native or crop')
        if type(case_id) is not str or type(ordinal) is not int:
            raise ValueError('Model smoke render scope requires a case ID and synthetic ordinal')
        cases = {name: index for index, (name, _, _) in enumerate(_MODEL_SMOKE_CASES)}
        if case_id not in cases or ordinal != cases[case_id]:
            raise ValueError('Model smoke case ID and ordinal must match the fixed a/b/c order')
        state = {'role': role, 'case_id': case_id, 'ordinal': ordinal,
                 'attempts': 0, 'completed': 0}
        self._scope = state
        failed = False
        try:
            yield self
        except BaseException:
            failed = True
            raise
        else:
            required = 1 if role == 'native' else 5
            if state['attempts'] != required or state['completed'] != required:
                raise ValueError(
                    f'Model smoke {role} scope requires exactly {required} completed render calls')
        finally:
            self._scope = None

    def observe_native(self, case_id, ordinal):
        return self._observe_scope('native', case_id, ordinal)

    def observe_refinement(self, case_id, ordinal):
        return self._observe_scope('crop', case_id, ordinal)

    def _event_payload(self, scope, iteration, camera, requested_types,
                       return_tensors, background, metadata, arrays, *, returned):
        payload_arrays = MappingProxyType(dict(arrays))
        payload = {
            'case_id': scope['case_id'], 'ordinal': scope['ordinal'],
            'role': scope['role'], 'iteration': iteration,
            'camera': MappingProxyType(dict(camera)),
            'requested_render_types': tuple(requested_types),
            'return_tensors': return_tensors,
            'requested_background': (None if background is None else tuple(background)),
            'render_policy_metadata': metadata,
            'arrays': payload_arrays,
            'mask_equals_depth_positive': None,
            'returned': returned,
        }
        color = arrays.get('color_norm_f32')
        depth = arrays.get('depth_mm')
        mask = arrays.get('mask')
        if (isinstance(color, np.ndarray) and isinstance(depth, np.ndarray) and
                isinstance(mask, np.ndarray) and depth.shape == mask.shape and
                mask.dtype == np.dtype(np.bool_)):
            payload['mask_equals_depth_positive'] = bool(np.array_equal(mask, depth > 0.))
        return MappingProxyType(payload)

    def _send_failed_render(self, scope, iteration, camera, requested_types,
                            return_tensors, background, cause):
        metadata = None
        try:
            value = self._renderer.render_policy_metadata
            metadata = None if value is None else _model_smoke_freeze(value)
        except BaseException:
            metadata = None
        payload = self._event_payload(
            scope, iteration, camera, requested_types, return_tensors,
            background, metadata,
            {'color_norm_f32': None, 'depth_mm': None, 'mask': None,
             'model_rgb_u8': None, 'cpu_rgb_u8': None, 'cpu_depth_m': None,
             'cpu_mask': None}, returned=False)
        try:
            self.last_event = self._sink(payload)
            self.last_observation = payload
        except BaseException as sink_error:
            raise sink_error from cause

    def render_object_model(self, obj_id, camera_model_c2w, render_types=None,
                            return_tensors=False, background=None, **kwargs):
        if self._inside_render:
            raise RuntimeError('Model smoke render sink cannot reenter the observer')
        if self._scope is None:
            raise RuntimeError('Model smoke renderer call occurred outside an observation scope')
        if self._close_called:
            raise RuntimeError('Model smoke renderer is already closed')
        if kwargs:
            raise ValueError('Model smoke renderer does not accept unexpected render keywords')
        scope = self._scope
        role = scope['role']
        if role == 'native':
            if scope['attempts'] >= 1:
                raise ValueError('Native model smoke scope permits one renderer call')
            expected_types = _MODEL_SMOKE_RENDER_TYPES_NATIVE
            expected_dimensions = (720, 720)
            expected_tensors = False
            expected_background = None
        else:
            if scope['attempts'] >= 5:
                raise ValueError('Model smoke refinement scope permits only five crop renders')
            expected_types = _MODEL_SMOKE_RENDER_TYPES_CROP
            expected_dimensions = (280, 280)
            expected_tensors = True
            expected_background = (.5, .5, .5)
        if type(obj_id) is not int or obj_id != self.obj_id:
            raise ValueError('Model smoke renderer call uses an unexpected object ID')
        if type(return_tensors) is not bool or return_tensors is not expected_tensors:
            raise ValueError('Model smoke renderer return_tensors differs from the fixed call boundary')
        if type(render_types) is not list or len(render_types) != 3:
            raise ValueError('Model smoke render_types must be the exact three-item public request')
        type_names = []
        for value in render_types:
            name = getattr(value, 'value', None)
            if type(name) is not str:
                raise ValueError('Model smoke render_types must be public RenderType values')
            type_names.append(name)
        if tuple(type_names) != expected_types:
            raise ValueError('Model smoke renderer call has an unexpected RenderType order')
        if expected_background is None:
            if background is not None:
                raise ValueError('Native model smoke render must use the renderer default background')
            normalized_background = None
        else:
            try:
                background_array = np.asarray(background, dtype=np.float64)
            except Exception as exc:
                raise ValueError('Crop model smoke render requires the upstream gray background') from exc
            if (background_array.shape != (3,) or not np.array_equal(
                    background_array, np.asarray(expected_background, dtype=np.float64))):
                raise ValueError('Crop model smoke render requires the upstream gray background')
            normalized_background = expected_background
        camera = _model_smoke_camera_snapshot(camera_model_c2w)
        if (camera['width'], camera['height']) != expected_dimensions:
            raise ValueError('Model smoke render camera dimensions differ from the active scope')
        if role == 'native' and (camera['fx'], camera['fy'], camera['cx'], camera['cy']) != (
                360., 360., 359.5, 359.5):
            raise ValueError('Native model smoke render camera differs from fixed K720')

        iteration = None if role == 'native' else scope['attempts']
        scope['attempts'] += 1
        self._inside_render = True
        returned = False
        payload = None
        try:
            call_started_ns = _model_smoke_timing_start(
                self._timing_torch, self._timing_device)
            try:
                result = self._renderer.render_object_model(
                    obj_id, camera_model_c2w, render_types, return_tensors,
                    background, **kwargs)
            finally:
                call_elapsed_ms = _model_smoke_timing_finish(
                    self._timing_torch, self._timing_device, call_started_ns)
                self._render_call_records.append({
                    'render_index': len(self._render_call_records),
                    'elapsed_ms': call_elapsed_ms,
                })
            returned = True
            scope['returned'] = scope.get('returned', 0) + 1
            metadata_value = self._renderer.render_policy_metadata
            metadata = (None if metadata_value is None else
                        _model_smoke_freeze(metadata_value))
            rendered_arrays = {'color_norm_f32': None, 'depth_mm': None, 'mask': None,
                               'model_rgb_u8': None, 'cpu_rgb_u8': None,
                               'cpu_depth_m': None, 'cpu_mask': None}
            if isinstance(result, Mapping):
                expected_names = {'COLOR': 'color_norm_f32',
                                  'DEPTH': 'depth_mm', 'MASK': 'mask'}
                for requested in render_types:
                    label = expected_names.get(getattr(requested, 'name', None))
                    if label is not None and requested in result:
                        rendered_arrays[label] = _model_smoke_snapshot_array(
                            result[requested], name=label)
            payload = self._event_payload(
                scope, iteration, camera, type_names, return_tensors,
                normalized_background, metadata, rendered_arrays, returned=True)
            self.last_event = self._sink(payload)
            self.last_observation = payload
            color = rendered_arrays['color_norm_f32']
            depth = rendered_arrays['depth_mm']
            mask = rendered_arrays['mask']
            height, width = expected_dimensions[1], expected_dimensions[0]
            if (not isinstance(result, Mapping) or set(result) != set(render_types) or
                    not isinstance(color, np.ndarray) or color.shape != (height, width, 3) or
                    color.dtype != np.dtype(np.float32) or not np.isfinite(color).all() or
                    np.any(color < 0.) or np.any(color > 1.) or
                    not isinstance(depth, np.ndarray) or depth.shape != (height, width) or
                    depth.dtype != np.dtype(np.float32) or not np.isfinite(depth).all() or
                    np.any(depth < 0.) or not isinstance(mask, np.ndarray) or
                    mask.shape != (height, width) or mask.dtype != np.dtype(np.bool_) or
                    not np.array_equal(mask, depth > 0.)):
                raise ValueError('Public model smoke renderer returned invalid raw RGB/depth/mask arrays')
            if not isinstance(metadata, Mapping):
                raise ValueError('Public model smoke renderer returned no current render-policy metadata')
            scope['completed'] += 1
            return result
        except BaseException as exc:
            if payload is None:
                self._send_failed_render(
                    scope, iteration, camera, type_names, return_tensors,
                    normalized_background, exc)
            raise
        finally:
            self._inside_render = False

    def close(self):
        if self._inside_render:
            raise RuntimeError('Model smoke renderer cannot close from its render sink')
        if self._close_called:
            return
        self._close_called = True
        self._renderer.close()
        if self._renderer.render_policy_metadata is not None:
            raise RuntimeError('Public renderer metadata remained live after close')
        self.renderer_closed = True


class ModelSmokeNetworkCounter:
    """Count actual model calls and bind their inputs to the latest crop render."""

    def __init__(self, network, observer):
        if not callable(network):
            raise TypeError('Model smoke network counter requires the loaded callable network')
        self._network = network
        self._observer = observer
        self.attempted_calls = 0
        self.returned_calls = 0

    def __call__(self, *args, **kwargs):
        if len(args) != 3 or kwargs:
            raise ValueError('Model smoke network call must use its three public positional tensors')
        scope = self._observer.current_scope
        observation = self._observer.last_observation
        event = self._observer.last_event
        if (scope is None or scope['role'] != 'crop' or observation is None or
                event is None or event.get('case_id') != scope['case_id'] or
                event.get('iteration') != scope['iteration']):
            raise ValueError('Model smoke network call has no immediately preceding crop render')
        names = ('query_rgb_bchw', 'template_rgb_bchw', 'template_mask_bhw')
        arrays = [_model_smoke_snapshot_array(value, name=name)
                  for value, name in zip(args, names)]
        expected_shapes = ((1, 3, 280, 280), (1, 3, 280, 280), (1, 280, 280))
        if any(array.shape != shape for array, shape in zip(arrays, expected_shapes)):
            raise ValueError('Model smoke network tensors have unexpected batch-one crop shapes')
        if (arrays[0].dtype != np.dtype(np.float32) or
                arrays[1].dtype != np.dtype(np.float32) or
                arrays[2].dtype != np.dtype(np.bool_) or
                not np.isfinite(arrays[0]).all() or not np.isfinite(arrays[1]).all()):
            raise ValueError('Model smoke network inputs must use finite float32 RGB and bool mask')
        observed_arrays = observation['arrays']
        raw_rgb = observed_arrays.get('color_norm_f32')
        raw_mask = observed_arrays.get('mask')
        template_matches = bool(
            isinstance(raw_rgb, np.ndarray) and isinstance(raw_mask, np.ndarray) and
            arrays[1].dtype == raw_rgb.dtype and raw_rgb.shape == (280, 280, 3) and
            np.array_equal(arrays[1][0], raw_rgb.transpose(2, 0, 1)) and
            raw_mask.shape == (280, 280) and
            np.array_equal(arrays[2][0], raw_mask))
        network_inputs = {
            'query_rgb_bchw': _model_smoke_array_descriptor(arrays[0]),
            'template_rgb_bchw': _model_smoke_array_descriptor(arrays[1]),
            'template_mask_bhw': _model_smoke_array_descriptor(arrays[2]),
            'template_matches_observed_render': template_matches,
            'call_returned': False,
        }
        event['network_inputs'] = network_inputs
        if not template_matches:
            raise ValueError('Model smoke template tensors differ from the preceding raw crop render')
        self.attempted_calls += 1
        outputs = self._network(*args, **kwargs)
        self.returned_calls += 1
        network_inputs['call_returned'] = True
        return outputs


def _model_smoke_validate_recipe(recipe, render_flags, device):
    plain = _model_smoke_plain(recipe)
    if type(plain) is not dict or set(plain) != {
            'schema', 'resource', 'resource_kind', 'object_id', 'object_name',
            'asset_sha256', 'asset_receipt_sha256', 'unit_receipt_sha256',
            'renderer_proof_sha256', 'output_camera_sha256', 'coordinate_mode',
            'rendering_contract', 'render_flags', 'producer_device', 'settings'}:
        raise ValueError('Capture smoke output recipe fields differ from the fixed resource plan')
    contract = plain['rendering_contract']
    expected_views = [
        {'case_id': case_id, 'euler_xyz_degrees': list(degrees),
         'offset_extent_units': list(offset)}
        for case_id, degrees, offset in _MODEL_SMOKE_CASES
    ]
    if (type(contract) is not dict or
            contract.get('model_smoke_views') != expected_views or
            contract.get('model_smoke_rotation') != _MODEL_SMOKE_ROTATION or
            contract.get('model_smoke_native_intrinsics') != [list(row) for row in _MODEL_SMOKE_NATIVE_K] or
            contract.get('model_smoke_observation') != 'public_return_v1' or
            contract.get('model_smoke_array_digest') != 'quality-model-smoke-array-v1'):
        raise ValueError('Capture smoke recipe lacks its issued fixed views, observation rule, or digest rule')
    recipe_flags = {
        'unlit_templates': render_flags['unlit_templates'],
        'disable_multisampling': render_flags['disable_multisampling'],
    }
    if (plain.get('resource') != 'smoke' or
            plain.get('resource_kind') != 'quality-capture-model-smoke-v1' or
            plain.get('coordinate_mode') != 'integer_centers_v1' or
            plain.get('producer_device') != device or
            plain.get('render_flags') != recipe_flags):
        raise ValueError('Capture smoke recipe does not match its issued device and render flags')
    if (plain.get('settings', {}).get('synthetic_pose_count') != 3 or
            plain.get('settings', {}).get('minimum_iou') != .94 or
            plain.get('settings', {}).get('maximum_median_common_depth_error_m') != .002):
        raise ValueError('Capture smoke recipe thresholds differ from the fixed public smoke gates')
    return plain


def _model_smoke_preflight(bundle, manifest, resources, device):
    """Re-authenticate the exact smoke plan before model or renderer imports."""
    if not isinstance(manifest, Mapping) or manifest.get('_capture') is None:
        raise ValueError('Model-only capture smoke requires an authenticated capture plan')
    plan = manifest['_capture']
    provenance = inference_provenance(
        bundle, capture_plan=plan, capture_resources=resources)
    if (getattr(resources, 'stage', None) != 'smoke' or
            getattr(resources, 'device', None) != device or
            type(getattr(resources, 'required', None)) is not tuple or
            resources.required or resources.output is None):
        raise ValueError('Model-only capture smoke requires exact smoke resources without prerequisites')
    expected_flags = {
        'unlit_templates': None,
        'disable_multisampling': False,
        'grayscale': False,
    }
    flags = _model_smoke_plain(resources.render_flags)
    if (type(flags) is not dict or set(flags) != set(expected_flags) or
            type(flags.get('unlit_templates')) is not bool or
            type(flags.get('disable_multisampling')) is not bool or
            type(flags.get('grayscale')) is not bool or
            flags.get('disable_multisampling') is not False or
            flags.get('grayscale') is not False or
            flags != {**expected_flags, 'unlit_templates': flags.get('unlit_templates')}):
        raise ValueError('Model-only capture smoke resource flags differ from the exact issued three-key policy')
    output = resources.output
    if (output.purpose != 'smoke' or
            output.resource_kind != 'quality-capture-model-smoke-v1' or
            output.descriptor is not None or
            type(output.resource_key) is not str or len(output.resource_key) != 64 or
            any(char not in '0123456789ABCDEF' for char in output.resource_key)):
        raise ValueError('Model-only capture smoke output is not the exact unissued smoke resource plan')
    capture_provenance = provenance.get('capture_provenance')
    if (not isinstance(capture_provenance, Mapping) or
            capture_provenance.get('stage') != 'smoke' or
            capture_provenance.get('render_flags') != flags or
            capture_provenance.get('used_resources') != []):
        raise ValueError('Authenticated smoke provenance has prerequisites or a different stage/flag set')
    expected_output = {
        'resource_kind': output.resource_kind,
        'resource_key': output.resource_key,
        'sidecar_path': output.sidecar_path.relative_to(CAPTURE_CACHE).as_posix(),
        'recipe': _model_smoke_plain(output.recipe),
        'producer_source_closure': _model_smoke_plain(output.source_closure),
    }
    actual_output = _model_smoke_plain(capture_provenance.get('output_resource'))
    if actual_output != expected_output:
        raise ValueError('Authenticated smoke provenance does not bind the exact output plan')
    recipe = _model_smoke_validate_recipe(output.recipe, resources.render_flags, device)
    asset = resources.asset_receipt
    plan_manifest = plan.manifest
    if (plan_manifest.get('object') != asset.object_name or
            plan_manifest.get('object_id') != asset.object_id or
            plan_manifest.get('units') != 'metres' or
            recipe['object_id'] != asset.object_id or
            recipe['object_name'] != asset.object_name or
            recipe['asset_sha256'].upper() != asset.asset_sha256.upper() or
            recipe['asset_receipt_sha256'].upper() != asset.receipt_sha256.upper() or
            recipe['renderer_proof_sha256'].upper() != asset.renderer_proof_sha256.upper()):
        raise ValueError('Model smoke plan asset/object fields differ from the authenticated receipt')
    if _model_smoke_plain(plan_manifest.get('intrinsics')) != [
            list(row) for row in _MODEL_SMOKE_NATIVE_K]:
        raise ValueError('Model smoke plan camera differs from its authenticated fixed K720')
    if (capture_provenance.get('asset_receipt_sha256') != asset.receipt_sha256 or
            capture_provenance.get('renderer_proof_sha256') != asset.renderer_proof_sha256):
        raise ValueError('Model smoke provenance asset/proof digests differ from the authenticated receipt')
    return plan, plan_manifest, recipe, flags, provenance


def _model_smoke_source_identity(metadata, source_closure):
    expected = {
        'quality_gotrack': (
            'bench.quality_gotrack', 'bench/quality_gotrack.py', 'bench/quality_gotrack.py'),
        'quality_render_stability': (
            'bench.quality_render_stability', 'bench/quality_render_stability.py',
            'bench/quality_render_stability.py'),
        'pyrender_renderer': (
            'pyrender.renderer',
            '.cache/quality-windows/Lib/site-packages/pyrender/renderer.py',
            'runtime/pyrender/renderer.py'),
        'pyrender_offscreen': (
            'pyrender.offscreen',
            '.cache/quality-windows/Lib/site-packages/pyrender/offscreen.py',
            'runtime/pyrender/offscreen.py'),
    }
    identities = metadata.get('source_identities')
    if type(identities) is not dict or set(identities) != set(expected):
        raise ValueError('Renderer source identity keys differ from the reviewed public renderer closure')
    root = Path(__file__).resolve().parents[1]
    for name, (module_name, physical_relative, closure_key) in expected.items():
        identity = identities[name]
        if type(identity) is not dict or set(identity) != {'module', 'file', 'sha256'}:
            raise ValueError('Renderer source identity record fields differ from the fixed schema')
        expected_path = (root / physical_relative).resolve(strict=True)
        expected_sha = source_closure.get(closure_key)
        if (identity['module'] != module_name or
                identity['file'] != str(expected_path) or
                type(identity['sha256']) is not str or
                len(identity['sha256']) != 64 or
                any(char not in '0123456789ABCDEF' for char in identity['sha256']) or
                expected_sha != identity['sha256']):
            raise ValueError(f'Renderer source identity {name!r} differs from its issued source closure')
        if digest(expected_path).upper() != identity['sha256']:
            raise ValueError(f'Renderer source identity {name!r} differs from current source bytes')


def _model_smoke_validate_attachment(row, *, kind, width, height):
    fields = {'target_name', 'format_name', 'samples', 'width', 'height',
              'renderbuffer_id', 'ordinary_storage_delegated', 'success'}
    if type(row) is not dict or set(row) != fields:
        raise ValueError('Renderer allocation attachment row differs from the fixed schema')
    expected_format = 'GL_RGBA' if kind == 'color' else 'GL_DEPTH_COMPONENT24'
    if (row['target_name'] != 'GL_RENDERBUFFER' or row['format_name'] != expected_format or
            type(row['samples']) is not int or row['samples'] != 4 or
            type(row['width']) is not int or row['width'] != width or
            type(row['height']) is not int or row['height'] != height or
            type(row['renderbuffer_id']) is not int or row['renderbuffer_id'] <= 0 or
            row['ordinary_storage_delegated'] is not True or row['success'] is not True):
        raise ValueError(f'Renderer {kind} attachment does not match ordinary storage at this size')


def _model_smoke_validate_metadata(metadata, dimensions, source_closure, lifecycle):
    plain = _model_smoke_plain(metadata)
    if type(plain) is not dict or set(plain) != _MODEL_SMOKE_METADATA_KEYS:
        raise ValueError('Public render-policy metadata does not match the exact 19-field schema')
    width, height = dimensions
    if (type(plain['schema_version']) is not int or plain['schema_version'] != 1 or
            plain['render_policy'] != 'capture_zero_sample_v1' or
            plain['coordinate_mode'] != 'integer_centers_v1' or
            plain['depth_units'] != 'millimetres' or
            plain['dimensions'] != [width, height]):
        raise ValueError('Public render-policy metadata has a stale schema, mode, policy, units, or dimensions')
    generation = plain['allocation_generation']
    identity = plain['offscreen_identity']
    if (type(generation) is not int or generation <= 0 or
            type(identity) is not int or identity <= 0):
        raise ValueError('Public renderer generation and offscreen identity must be positive integers')
    fields = plain['framebuffer_fields']
    if type(fields) is not dict or set(fields) != {
            'multisample_draw_fbo', 'single_sample_read_fbo', 'multisample_dimensions'}:
        raise ValueError('Renderer framebuffer field snapshot differs from the fixed schema')
    draw_fbo = fields['multisample_draw_fbo']
    read_fbo = fields['single_sample_read_fbo']
    if (type(draw_fbo) is not int or draw_fbo <= 0 or
            type(read_fbo) is not int or read_fbo <= 0 or draw_fbo == read_fbo or
            fields['multisample_dimensions'] != [width, height]):
        raise ValueError('Renderer framebuffer identities/dimensions are invalid')
    pair = plain['allocation_pair']
    if type(pair) is not dict or set(pair) != {'color', 'depth', 'dimensions', 'passed'}:
        raise ValueError('Renderer allocation pair differs from the fixed schema')
    if pair['dimensions'] != [width, height] or pair['passed'] is not True:
        raise ValueError('Renderer allocation pair did not pass at the observed dimensions')
    _model_smoke_validate_attachment(pair['color'], kind='color', width=width, height=height)
    _model_smoke_validate_attachment(pair['depth'], kind='depth', width=width, height=height)
    if pair['color']['renderbuffer_id'] == pair['depth']['renderbuffer_id']:
        raise ValueError('Renderer color and depth attachments must have distinct identities')
    calls = plain['allocation_calls']
    if type(calls) is not list:
        raise ValueError('Renderer allocation_calls must be an ordered list')
    if len(calls) == 2:
        _model_smoke_validate_attachment(calls[0], kind='color', width=width, height=height)
        _model_smoke_validate_attachment(calls[1], kind='depth', width=width, height=height)
        if calls != [pair['color'], pair['depth']]:
            raise ValueError('Fresh renderer allocation calls differ from the verified pair')
    elif calls:
        raise ValueError('Renderer allocation_calls must be empty on reuse or contain the exact pair')
    if (plain['framebuffer_complete'] is not True or
            type(plain['gl_samples']) is not int or plain['gl_samples'] != 0 or
            type(plain['gl_sample_buffers']) is not int or plain['gl_sample_buffers'] != 0 or
            plain['framebuffer_bindings_restored'] is not True or
            plain['current_context_released'] is not True or
            plain['dimension_match'] is not True or
            plain['color_storage_policy'] !=
                'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage' or
            plain['depth_storage_policy'] !=
                'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage'):
        raise ValueError('Renderer attachment, sample, context, or storage evidence is incomplete')
    _model_smoke_source_identity(plain, source_closure)

    previous = lifecycle.get('metadata')
    if previous is None:
        if generation != 1 or len(calls) != 2:
            raise ValueError('First smoke render must show a fresh generation-1 allocation pair')
    elif previous['dimensions'] == [width, height]:
        if (generation != previous['allocation_generation'] or calls or
                identity != previous['offscreen_identity'] or
                pair != previous['allocation_pair'] or fields != previous['framebuffer_fields']):
            raise ValueError('Same-size renderer reuse changed its generation or public allocations')
    else:
        if generation != previous['allocation_generation'] + 1 or len(calls) != 2:
            raise ValueError('Renderer resize must create exactly one new generation and allocation pair')
    lifecycle['metadata'] = plain
    return plain


def _model_smoke_artifact_event(events, payload, source_closure, lifecycle):
    index = len(events)
    if index >= 18:
        raise ValueError('Model smoke emitted more than the fixed 18 render events')
    case_index, within_case = divmod(index, 6)
    case_id, _, _ = _MODEL_SMOKE_CASES[case_index]
    ordinal = case_index
    role = 'native' if within_case == 0 else 'crop'
    iteration = None if role == 'native' else within_case - 1
    if (payload['case_id'] != case_id or payload['ordinal'] != ordinal or
            payload['role'] != role or payload['iteration'] != iteration):
        raise ValueError('Model smoke render-event order differs from native-plus-five-crop recipe')
    arrays = payload['arrays']
    descriptors = {}
    for name in ('color_norm_f32', 'depth_mm', 'mask', 'model_rgb_u8',
                 'cpu_rgb_u8', 'cpu_depth_m', 'cpu_mask'):
        value = arrays.get(name)
        if value is None:
            descriptors[name] = None
        else:
            try:
                descriptors[name] = _model_smoke_array_descriptor(value)
            except ValueError:
                descriptors[name] = None
    metadata = payload['render_policy_metadata']
    try:
        metadata_plain = None if metadata is None else _model_smoke_plain(metadata)
    except ValueError:
        metadata_plain = None
    event = {
        'render_index': index, 'renderer_id': 0,
        'case_id': case_id, 'ordinal': ordinal, 'role': role,
        'iteration': iteration,
        'camera': _model_smoke_plain(payload['camera']),
        'requested_render_types': list(payload['requested_render_types']),
        'return_tensors': payload['return_tensors'],
        'requested_background': (None if payload['requested_background'] is None else
                                 list(payload['requested_background'])),
        'render_policy_metadata': metadata_plain,
        'arrays': descriptors,
        'mask_equals_depth_positive': payload['mask_equals_depth_positive'],
        'network_inputs': None,
    }
    events.append(event)
    if not payload['returned']:
        return event
    width, height = ((720, 720) if role == 'native' else (280, 280))
    if (event['requested_render_types'] != list(
            _MODEL_SMOKE_RENDER_TYPES_NATIVE if role == 'native' else
            _MODEL_SMOKE_RENDER_TYPES_CROP) or
            type(event['return_tensors']) is not bool or
            event['return_tensors'] is not (role == 'crop') or
            event['requested_background'] != (None if role == 'native' else [.5, .5, .5]) or
            event['camera']['width'] != width or event['camera']['height'] != height or
            event['arrays']['color_norm_f32'] is None or
            event['arrays']['depth_mm'] is None or event['arrays']['mask'] is None or
            event['mask_equals_depth_positive'] is not True):
        raise ValueError('Model smoke render event is missing a required raw public observation')
    expected_shapes = {
        'color_norm_f32': ([height, width, 3], '<f4'),
        'depth_mm': ([height, width], '<f4'),
        'mask': ([height, width], '|b1'),
    }
    for name, (shape, dtype) in expected_shapes.items():
        descriptor = event['arrays'][name]
        if descriptor['shape'] != shape or descriptor['dtype'] != dtype or descriptor['order'] != 'C':
            raise ValueError(f'Model smoke {name} descriptor differs from its raw renderer type/shape')
    if role == 'native':
        if any(event['arrays'][name] is not None for name in
               ('model_rgb_u8', 'cpu_rgb_u8', 'cpu_depth_m', 'cpu_mask')):
            raise ValueError('Native comparison descriptors must be added only after the CPU render')
    elif any(event['arrays'][name] is not None for name in
             ('model_rgb_u8', 'cpu_rgb_u8', 'cpu_depth_m', 'cpu_mask')):
        raise ValueError('Crop events cannot carry native or CPU comparison arrays')
    if not isinstance(metadata_plain, dict):
        raise ValueError('Model smoke render event has no complete public metadata snapshot')
    _model_smoke_validate_metadata(metadata_plain, (width, height), source_closure, lifecycle)
    return event


def _model_smoke_seed_pose(vertices_m, euler_degrees, offset_units, checked_pose):
    angles = np.deg2rad(np.asarray(euler_degrees, dtype=np.float64))
    ax, ay, az = angles
    rx = np.asarray([[1., 0., 0.],
                     [0., math.cos(ax), -math.sin(ax)],
                     [0., math.sin(ax), math.cos(ax)]], dtype=np.float64)
    ry = np.asarray([[math.cos(ay), 0., math.sin(ay)],
                     [0., 1., 0.],
                     [-math.sin(ay), 0., math.cos(ay)]], dtype=np.float64)
    rz = np.asarray([[math.cos(az), -math.sin(az), 0.],
                     [math.sin(az), math.cos(az), 0.],
                     [0., 0., 1.]], dtype=np.float64)
    rotation = rx @ ry @ rz @ np.diag([1., -1., -1.])
    low = np.min(vertices_m, axis=0)
    high = np.max(vertices_m, axis=0)
    extents = high - low
    extent = float(np.max(extents))
    if not math.isfinite(extent) or extent <= 0.:
        raise ValueError('Model smoke GLB has no positive finite maximum extent')
    center = (low + high) * .5
    translation = extent * np.asarray(offset_units, dtype=np.float64) - rotation @ center
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return checked_pose(pose), low, high, extents, extent, center


def _model_smoke_verify_bounds(mesh, renderer_vertices_m, asset_receipt):
    if (type(asset_receipt.source_units) is not str or asset_receipt.source_units != 'metres' or
            float(asset_receipt.conversion_to_metres) != 1.):
        raise ValueError('Model smoke requires an already-normalized metric asset receipt')
    vertices = np.asarray(mesh.positions, dtype=np.float64)
    rendered_vertices = np.asarray(renderer_vertices_m, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or
            not np.isfinite(vertices).all() or rendered_vertices.ndim != 2 or
            rendered_vertices.shape[1] != 3 or not len(rendered_vertices) or
            not np.isfinite(rendered_vertices).all()):
        raise ValueError('Independent CPU GLB and public renderer vertices must be finite Nx3 metres')
    bounds = _model_smoke_plain(asset_receipt.document).get('post_node_bounds_m')
    if type(bounds) is not dict or set(bounds) != {'min', 'max', 'extents'}:
        raise ValueError('Authenticated asset receipt has no exact post-node metric bounds')
    low = np.min(vertices, axis=0)
    high = np.max(vertices, axis=0)
    extents = high - low
    receipt_values = (
        (np.asarray(bounds['min'], dtype=np.float64), low),
        (np.asarray(bounds['max'], dtype=np.float64), high),
        (np.asarray(bounds['extents'], dtype=np.float64), extents),
    )
    if any(expected.shape != (3,) or not np.isfinite(expected).all() or
           not np.allclose(actual, expected, rtol=0., atol=1e-6)
           for expected, actual in receipt_values):
        raise ValueError('Independent CPU GLB bounds differ from the authenticated asset receipt')
    if (not np.allclose(np.min(rendered_vertices, axis=0), low, rtol=0., atol=1e-6) or
            not np.allclose(np.max(rendered_vertices, axis=0), high, rtol=0., atol=1e-6)):
        raise ValueError('Public renderer and independent CPU GLB use different original-origin bounds')
    return low, high, extents


def _model_smoke_check_native_comparison(event, gpu_mask, cpu_mask, gpu_depth_mm,
                                         cpu_depth_m, model_rgb, cpu_rgb):
    expected_descriptors = {
        'model_rgb_u8': model_rgb,
        'cpu_rgb_u8': cpu_rgb,
        'cpu_depth_m': cpu_depth_m,
        'cpu_mask': cpu_mask,
    }
    for name, value in expected_descriptors.items():
        event['arrays'][name] = _model_smoke_array_descriptor(value)
    if (event['arrays']['model_rgb_u8']['shape'] != [720, 720, 3] or
            event['arrays']['model_rgb_u8']['dtype'] != '|u1' or
            event['arrays']['cpu_rgb_u8']['shape'] != [720, 720, 3] or
            event['arrays']['cpu_rgb_u8']['dtype'] != '|u1' or
            event['arrays']['cpu_depth_m']['shape'] != [720, 720] or
            event['arrays']['cpu_depth_m']['dtype'] != '<f8' or
            event['arrays']['cpu_mask']['shape'] != [720, 720] or
            event['arrays']['cpu_mask']['dtype'] != '|b1'):
        raise ValueError('Native model/CPU comparison descriptors differ from their source raster dtypes')
    if not np.any(gpu_mask) or not np.any(cpu_mask):
        raise ValueError('Native model/CPU geometry comparison requires nonempty foreground masks')
    union = int(np.count_nonzero(gpu_mask | cpu_mask))
    if union <= 0:
        raise ValueError('Native model/CPU geometry comparison has an empty union')
    iou = float(np.count_nonzero(gpu_mask & cpu_mask) / union)
    common = gpu_mask & cpu_mask
    common_count = int(np.count_nonzero(common))
    if common_count <= 0:
        raise ValueError('Native model/CPU geometry comparison has no common foreground pixels')
    if (not np.isfinite(gpu_depth_mm[common]).all() or
            not np.isfinite(cpu_depth_m[common]).all() or
            np.any(gpu_depth_mm[common] <= 0.) or np.any(cpu_depth_m[common] <= 0.)):
        raise ValueError('Native common foreground depth must be positive and finite in its original units')
    median_error = float(np.median(np.abs(
        gpu_depth_mm[common].astype(np.float64) * .001 - cpu_depth_m[common])))
    if not math.isfinite(iou) or not math.isfinite(median_error):
        raise ValueError('Native geometry comparison produced a nonfinite metric')
    if iou < .94 or median_error > .002:
        raise RuntimeError(
            f'Model smoke geometry gate failed: silhouette IoU {iou:.9g}, '
            f'median common depth error {median_error:.9g} m')
    return iou, median_error, common_count


def _model_smoke_build(device, torch, cv2):
    from importlib.metadata import version

    if device == 'cuda':
        cuda_runtime = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        if type(cuda_runtime) is not str or not cuda_runtime:
            raise ValueError('CUDA model smoke requires an actual nonempty CUDA runtime version')
    else:
        cuda_runtime = None
        gpu_name = None
    values = {
        'python': sys.version.split()[0],
        'numpy': str(np.__version__),
        'torch': str(torch.__version__),
        'opencv': str(cv2.__version__),
        'pyrender': str(version('pyrender')),
        'cuda_runtime': cuda_runtime,
        'gpu_name': gpu_name,
    }
    string_values = {key: value for key, value in values.items()
                     if key not in ('cuda_runtime', 'gpu_name')}
    if (any(type(value) is not str or not value for value in string_values.values()) or
            (device == 'cuda' and any(type(values[key]) is not str or not values[key]
                                      for key in ('cuda_runtime', 'gpu_name'))) or
            (device == 'cpu' and any(values[key] is not None
                                     for key in ('cuda_runtime', 'gpu_name')))):
        raise ValueError('Model smoke runtime build fields must be actual nonempty strings')
    return {**values, 'dtype': 'float32', 'batch_size': 1,
            'crop_size': [280, 280], 'iterations': 5}


def _model_smoke_constructor_record(renderer, asset_receipt, flags):
    arguments = {
        'unlit': flags['unlit_templates'],
        'disable_multisampling': flags['disable_multisampling'],
        'coordinate_mode': 'integer_centers_v1',
        'render_policy': 'capture_zero_sample_v1',
    }
    observed = {
        'unlit': renderer.unlit,
        'disable_multisampling': renderer.disable_multisampling,
        'coordinate_mode': renderer.coordinate_mode,
        'render_policy': renderer.render_policy,
    }
    if (any(type(observed[name]) is not type(arguments[name]) or
            observed[name] != arguments[name] for name in arguments) or
            renderer.obj_id != asset_receipt.object_id):
        raise ValueError('Public TexturedRenderer constructor configuration differs from the issued recipe')
    try:
        background = np.asarray(renderer.scene.bg_color, dtype=np.float64)
        ambient = np.asarray(renderer.scene.ambient_light, dtype=np.float64)
    except Exception as exc:
        raise ValueError('Public renderer scene does not expose its actual lighting configuration') from exc
    if (background.shape != (4,) or ambient.shape != (3,) or
            not np.isfinite(background).all() or not np.isfinite(ambient).all()):
        raise ValueError('Public renderer scene lighting metadata is malformed')
    observed_configuration = {
        **observed,
        'scene_bg_rgba': background.tolist(),
        'ambient_light_rgb': ambient.tolist(),
        'spotlight_intensity': 2.4,
        'spotlight_inner_cone': math.pi / 16.,
        'spotlight_outer_cone': math.pi / 6.,
    }
    return {
        'renderer_id': 0,
        'api': 'bench.quality_gotrack.TexturedRenderer',
        'object_id': asset_receipt.object_id,
        'asset_sha256': asset_receipt.asset_sha256,
        'arguments': arguments,
        'observed_configuration': observed_configuration,
    }


def _model_smoke_diagnostic_rows(refinement, case_id, events):
    diagnostics = getattr(refinement, 'iterations', None)
    if type(diagnostics) is not tuple:
        raise ValueError('Model smoke refinement must return its public diagnostic tuple')
    rows = []
    expected_fields = {
        'index', 'solver_success', 'crop_dimensions', 'query_rewarp_factor',
        'max_sampling_map_difference_px', 'sampling_map_eligible_count',
        'render_generation',
    }
    ordinal = {'a': 0, 'b': 1, 'c': 2}[case_id]
    for index, diagnostic in enumerate(diagnostics):
        row = _model_smoke_plain(diagnostic.as_dict())
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError('Model smoke iteration diagnostic fields differ from the public result schema')
        if (type(row['index']) is not int or row['index'] != index or
                type(row['solver_success']) is not bool or
                row['crop_dimensions'] != [280, 280] or
                type(row['query_rewarp_factor']) is not int or row['query_rewarp_factor'] != 1 or
                type(row['sampling_map_eligible_count']) is not int or
                type(row['render_generation']) is not int or row['render_generation'] <= 0):
            raise ValueError('Model smoke iteration diagnostic has invalid dimensions, factor, or count')
        difference = row['max_sampling_map_difference_px']
        if (difference is not None and
                (type(difference) not in (int, float) or not math.isfinite(float(difference)))):
            raise ValueError('Model smoke sampling map diagnostic must be finite or explicitly absent')
        event_index = ordinal * 6 + index + 1
        if event_index < len(events):
            event_meta = events[event_index]['render_policy_metadata']
            if (events[event_index]['role'] != 'crop' or
                    events[event_index]['iteration'] != index or
                    event_meta is None or
                    event_meta['allocation_generation'] != row['render_generation']):
                raise ValueError('Model smoke iteration generation differs from its crop render event')
        rows.append(row)
    return rows


def _model_smoke_validation_record(result, *, required_success):
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError('Public model smoke validator returned an unexpected result')
    passed, reason, raw_stats = result
    stats = _model_smoke_plain(raw_stats)
    fields = {
        'correspondences', 'inliers', 'median_reprojection_720',
        'p95_reprojection_720', 'spatial_support', 'score',
        'raw_correspondences', 'retained_correspondences',
        'unsupported_correspondences',
    }
    if type(stats) is not dict or set(stats) != fields:
        raise ValueError('Public model smoke validation stats differ from the supported field schema')
    result_record = {'passed': passed, 'reason': reason, 'stats': stats}
    if type(passed) is not bool or (passed and reason is not None) or (
            not passed and (type(reason) is not str or not reason)):
        raise ValueError('Public model smoke validation result has inconsistent pass/reason fields')
    counts = ('correspondences', 'inliers', 'raw_correspondences',
              'retained_correspondences', 'unsupported_correspondences')
    if any(type(stats[name]) is not int or stats[name] < 0 for name in counts):
        raise ValueError('Public model smoke validation correspondence counts must be nonnegative integers')
    if (stats['inliers'] > stats['correspondences'] or
            stats['retained_correspondences'] != stats['correspondences'] or
            stats['raw_correspondences'] <
            stats['retained_correspondences'] + stats['unsupported_correspondences']):
        raise ValueError('Public model smoke validation count accounting is inconsistent')
    if passed:
        numbers = ('median_reprojection_720', 'p95_reprojection_720',
                   'spatial_support', 'score')
        if any(type(stats[name]) not in (int, float) or
               not math.isfinite(float(stats[name])) for name in numbers):
            raise ValueError('Passed model smoke validation stats must be finite')
        if (stats['correspondences'] < 24 or stats['inliers'] < 20 or
                stats['inliers'] / max(stats['correspondences'], 1) < .6 or
                stats['median_reprojection_720'] > 3. or
                stats['p95_reprojection_720'] > 8. or
                stats['spatial_support'] < .12):
            raise ValueError('Passed public model smoke validation does not meet frozen thresholds')
    if required_success and not passed:
        raise RuntimeError(f'Public model smoke validation failed: {reason}')
    return result_record


def _model_smoke_utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _model_smoke_trial_output_path(output, output_use):
    path = Path(output)
    if path.name != 'worker-pending.json':
        raise ValueError('Capture model smoke output must be named worker-pending.json')
    if path.exists() or path.is_symlink():
        raise FileExistsError('Capture model smoke pending output must be a new file')
    trial_root = (CAPTURE_CACHE / 'capture-smoke-trials').resolve()
    parent = path.parent.resolve()
    try:
        relative = parent.relative_to(trial_root)
    except ValueError as exc:
        raise ValueError('Capture model smoke pending output must be inside a private trial directory') from exc
    if not relative.parts:
        raise ValueError('Capture model smoke output requires a distinct private trial directory')
    canonical_root = output_use.sidecar_path.parent.resolve()
    if parent == canonical_root or canonical_root in parent.parents:
        raise ValueError('Capture model smoke pending output cannot be inside the canonical resource key')
    if parent.exists() and not parent.is_dir():
        raise ValueError('Capture model smoke pending output parent must be a directory')
    return path


def _model_smoke_write_pending(output, output_use, pending):
    expected_fields = {
        'schema_version', 'kind', 'execution_id', 'resource_key', 'recipe',
        'source_closure', 'asset_receipt_sha256', 'renderer_proof_sha256',
        'started_at_utc', 'worker_ended_at_utc', 'checkpoint_sha256', 'device',
        'build', 'runtime_evidence', 'constructor_records', 'render_events', 'rows', 'status', 'error',
        'close_error', 'failure_location', 'renderer_closed', 'captured_rgb_read',
        'evaluator_data_read',
    }
    if type(pending) is not dict or set(pending) != expected_fields:
        raise ValueError('Capture model smoke pending report differs from its exact top-level schema')
    target = _model_smoke_trial_output_path(output, output_use)
    try:
        encoded = json.dumps(pending, ensure_ascii=False, allow_nan=False,
                             indent=2).encode('utf-8')
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError('Capture model smoke pending report is not finite JSON') from exc
    if len(encoded) > 2 * 1024 * 1024:
        raise ValueError('Capture model smoke pending report exceeds the 2 MiB bound')
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _model_smoke_trial_output_path(target, output_use)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(str(target), flags, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(encoded)
        stream.flush()


def capture_model_smoke(bundle, manifest, resources, *, device):
    """Run one bounded model-generated smoke and return a parent-finalized pending report."""
    plan, plan_manifest, recipe, flags, provenance = _model_smoke_preflight(
        bundle, manifest, resources, device)
    worker_started_ns = time.perf_counter_ns()
    asset_receipt = resources.asset_receipt
    output_use = resources.output
    source_closure = _model_smoke_plain(output_use.source_closure)
    output_recipe = _model_smoke_plain(output_use.recipe)
    runtime_evidence = {
        'schema_version': 1,
        'scope': 'model-only-smoke',
        'clock': 'perf_counter_ns',
        'cuda_synchronization': ('phase-boundaries' if device == 'cuda' else
                                 'not-applicable'),
        'timings_ms': {
            'worker_total': None,
            'network_load': None,
            'renderer_construct': None,
            'renderer_close': None,
            'cases': [
                {'case_id': case_id, 'native_render': None,
                 'cpu_geometry': None, 'refine': None, 'validation': None}
                for case_id, _, _ in _MODEL_SMOKE_CASES
            ],
        },
        'render_calls': [],
        'memory': _model_smoke_memory_evidence(device),
    }
    lifecycle = {'metadata': None}
    active = {'phase': 'loading', 'case_id': None, 'iteration': None}
    renderer = None
    observer = None
    torch = None
    network = None
    network_counter = None
    refiner = None
    worker_succeeded = False
    pending = {
        'schema_version': 1,
        'kind': 'quality-capture-model-smoke-pending-v1',
        'execution_id': uuid.uuid4().hex.upper(),
        'resource_key': output_use.resource_key,
        'recipe': output_recipe,
        'source_closure': source_closure,
        'asset_receipt_sha256': asset_receipt.receipt_sha256,
        'renderer_proof_sha256': asset_receipt.renderer_proof_sha256,
        'started_at_utc': _model_smoke_utc_now(),
        'worker_ended_at_utc': None,
        'checkpoint_sha256': None,
        'device': device,
        'build': None,
        'runtime_evidence': runtime_evidence,
        'constructor_records': [],
        'render_events': [],
        'rows': [],
        'status': 'failed',
        'error': None,
        'close_error': None,
        'failure_location': None,
        'renderer_closed': False,
        'captured_rgb_read': False,
        'evaluator_data_read': False,
    }

    try:
        active['phase'] = 'loading'
        import torch
        from .quality_assets import MODELS
        from .quality_gotrack import load_network, upstream_path

        torch.use_deterministic_algorithms(True)
        if device == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if device == 'cuda':
            device_info(device)
            torch.cuda.reset_peak_memory_stats(device)
            total_memory = int(torch.cuda.get_device_properties(
                torch.cuda.current_device()).total_memory)
            if total_memory <= 0:
                raise RuntimeError('CUDA device reported a nonpositive total memory size')
            runtime_evidence['memory']['cuda_total_bytes'] = total_memory
        network_load_started = _model_smoke_timing_start(torch, device)
        try:
            checkpoint_path = CACHE / 'checkpoints' / 'gotrack_checkpoint.pt'
            checkpoint_sha = digest(checkpoint_path).upper()
            expected_checkpoint = MODELS['gotrack_checkpoint.pt']['sha256'].upper()
            if checkpoint_sha != expected_checkpoint:
                raise ValueError('Pinned GoTrack checkpoint bytes differ from their current lock')
            pending['checkpoint_sha256'] = checkpoint_sha
            network = load_network(device)
        finally:
            runtime_evidence['timings_ms']['network_load'] = _model_smoke_timing_finish(
                torch, device, network_load_started)
        pending['build'] = _model_smoke_build(device, torch, cv2)

        active['phase'] = 'constructor'
        from .quality_gotrack import TexturedRenderer, GoTrackRefiner
        from .quality_contract import (ValidationSettings, checked_pose,
                                       make_model_smoke_frame, validate_model_smoke)
        from . import quality_contract
        from .glb_model import read_glb
        from .show3d_experiment import camera_for
        from .renderer import render as render_cpu_geometry
        upstream_path()
        from utils import renderer_base, structs

        asset_path = (Path(plan.bundle) / plan_manifest['asset']).resolve(strict=True)
        try:
            asset_path.relative_to(Path(plan.bundle).resolve(strict=True))
        except ValueError as exc:
            raise ValueError('Authenticated metric asset path escapes its capture bundle') from exc
        if digest(asset_path).upper() != asset_receipt.asset_sha256.upper():
            raise ValueError('Current metric asset bytes differ from the authenticated asset receipt')
        mesh = read_glb(asset_path, asset_receipt.object_name)
        renderer_construct_started = _model_smoke_timing_start(torch, device)
        try:
            renderer = TexturedRenderer(
                asset_path, asset_receipt.object_id,
                unlit=flags['unlit_templates'],
                disable_multisampling=flags['disable_multisampling'],
                coordinate_mode='integer_centers_v1',
                render_policy='capture_zero_sample_v1')
        finally:
            runtime_evidence['timings_ms']['renderer_construct'] = _model_smoke_timing_finish(
                torch, device, renderer_construct_started)
        _model_smoke_verify_bounds(mesh, renderer.vertices_m, asset_receipt)
        pending['constructor_records'].append(
            _model_smoke_constructor_record(renderer, asset_receipt, flags))

        def render_sink(payload):
            active['case_id'] = payload['case_id']
            active['iteration'] = payload['iteration']
            event = _model_smoke_artifact_event(
                pending['render_events'], payload, source_closure, lifecycle)
            return event

        observer = ModelSmokeRenderObserver(renderer, render_sink)
        observer._timing_torch = torch
        observer._timing_device = device
        network_counter = ModelSmokeNetworkCounter(network, observer)
        refiner = GoTrackRefiner(
            network_counter, observer, asset_receipt.object_id, device,
            pnp_use_extrinsic_guess=True)
        intrinsics = np.asarray(_MODEL_SMOKE_NATIVE_K, dtype=np.float64)
        cpu_intrinsics = intrinsics.copy()
        cpu_intrinsics[0, 2] += .5
        cpu_intrinsics[1, 2] += .5

        for case_id, euler_degrees, offset_units in _MODEL_SMOKE_CASES:
            case_timing = runtime_evidence['timings_ms']['cases'][
                {'a': 0, 'b': 1, 'c': 2}[case_id]]
            active['case_id'] = case_id
            active['iteration'] = None
            active['phase'] = 'native'
            seed, low, high, extents, extent, center = _model_smoke_seed_pose(
                observer.vertices_m, euler_degrees, offset_units, checked_pose)
            camera_world_from_eye = np.linalg.inv(seed)
            camera_world_from_eye[:3, 3] *= 1000.
            native_camera = structs.PinholePlaneCameraModel(
                width=720, height=720, f=(360., 360.), c=(359.5, 359.5),
                T_world_from_eye=camera_world_from_eye)
            render_types = [renderer_base.RenderType.COLOR,
                            renderer_base.RenderType.DEPTH,
                            renderer_base.RenderType.MASK]
            with observer.observe_native(case_id, {'a': 0, 'b': 1, 'c': 2}[case_id]):
                observer.render_object_model(
                    asset_receipt.object_id, native_camera,
                    render_types=render_types, return_tensors=False,
                    background=None)
            if not observer._render_call_records:
                raise RuntimeError('Native model smoke renderer timing sample was not recorded')
            case_timing['native_render'] = observer._render_call_records[-1]['elapsed_ms']
            native_event = pending['render_events'][-1]
            observation = observer.last_observation
            if (observation is None or observer.last_event is not native_event or
                    native_event['role'] != 'native'):
                raise RuntimeError('Native model smoke render lacks its synchronous public observation')
            raw = observation['arrays']
            color_norm = raw['color_norm_f32']
            gpu_depth_mm = raw['depth_mm']
            gpu_mask = raw['mask']
            model_rgb = np.floor(
                np.clip(color_norm, 0., 1.) * 255. + .5).astype(np.uint8)
            sampling_valid = np.ones((720, 720), dtype=np.bool_)

            active['phase'] = 'cpu_comparison'
            cpu_geometry_started = time.perf_counter_ns()
            cpu_camera = camera_for(
                (seed[:3, :3], seed[:3, 3]), cpu_intrinsics, 720, 720)
            cpu_frame = render_cpu_geometry(mesh, cpu_camera)
            cpu_rgb = np.asarray(cpu_frame.color)
            cpu_depth_m = np.asarray(cpu_frame.depth)
            cpu_mask = np.asarray(cpu_frame.triangle >= 0, dtype=np.bool_)
            if (cpu_rgb.shape != (720, 720, 3) or cpu_rgb.dtype != np.dtype(np.uint8) or
                    cpu_depth_m.shape != (720, 720) or cpu_depth_m.dtype != np.dtype(np.float64) or
                    cpu_mask.shape != (720, 720) or cpu_mask.dtype != np.dtype(np.bool_)):
                raise ValueError('Independent CPU geometry render returned an unexpected native raster')
            iou, median_depth_error, common_count = _model_smoke_check_native_comparison(
                native_event, gpu_mask, cpu_mask, gpu_depth_mm,
                cpu_depth_m, model_rgb, cpu_rgb)
            if (native_event['arrays']['model_rgb_u8'] is None or
                    native_event['arrays']['cpu_rgb_u8'] is None or
                    native_event['arrays']['cpu_depth_m'] is None or
                    native_event['arrays']['cpu_mask'] is None):
                raise ValueError('Native model/CPU comparison descriptors were not retained')
            case_timing['cpu_geometry'] = (
                time.perf_counter_ns() - cpu_geometry_started) / 1_000_000.

            frame = make_model_smoke_frame(
                case_id=case_id,
                ordinal={'a': 0, 'b': 1, 'c': 2}[case_id],
                rgb=model_rgb, intrinsics=intrinsics,
                sampling_valid=sampling_valid)
            active['phase'] = 'refinement'
            before_returned = network_counter.returned_calls
            refine_started = _model_smoke_timing_start(torch, device)
            with observer.observe_refinement(
                    case_id, {'a': 0, 'b': 1, 'c': 2}[case_id]):
                refinement = refiner.refine_model_smoke(frame, gpu_mask, seed)
                if refinement.candidate is None:
                    partial_iterations = _model_smoke_diagnostic_rows(
                        refinement, case_id, pending['render_events'])
                    native_render = {
                        'gpu_rgb_sha256': native_event['arrays']['model_rgb_u8']['sha256'],
                        'gpu_depth_mm_sha256': native_event['arrays']['depth_mm']['sha256'],
                        'gpu_mask_sha256': native_event['arrays']['mask']['sha256'],
                        'cpu_rgb_sha256': native_event['arrays']['cpu_rgb_u8']['sha256'],
                        'cpu_depth_m_sha256': native_event['arrays']['cpu_depth_m']['sha256'],
                        'cpu_mask_sha256': native_event['arrays']['cpu_mask']['sha256'],
                        'mask_iou': iou,
                        'median_common_depth_error_m': median_depth_error,
                        'common_depth_pixel_count': common_count,
                        'mask_equals_depth_positive': True,
                        'native_dimensions': [720, 720],
                        'coordinate_mode': 'integer_centers_v1',
                        'render_policy': 'capture_zero_sample_v1',
                        'unlit_templates': flags['unlit_templates'],
                        'disable_multisampling': flags['disable_multisampling'],
                    }
                    pending['rows'].append({
                        'pose_id': case_id,
                        'seed_camera_from_object_m': seed.tolist(),
                        'render': native_render,
                        'refinement': {
                            'candidate_present': False,
                            'iterations': partial_iterations,
                            'pose_camera_from_object_m': None,
                            'points_object_m_sha256': None,
                            'pixels_native_sha256': None,
                            'weights_sha256': None,
                        },
                        'validation': None,
                    })
                    active['iteration'] = None
                    raise RuntimeError('Model smoke refiner did not return a candidate pose')
                if len(refinement.iterations) != 5:
                    raise ValueError('Successful model smoke refinement must contain five actual iterations')
            case_timing['refine'] = _model_smoke_timing_finish(
                torch, device, refine_started)

            active['iteration'] = None
            iteration_rows = _model_smoke_diagnostic_rows(
                refinement, case_id, pending['render_events'])
            if (any(row['solver_success'] is not True or
                    row['sampling_map_eligible_count'] <= 0 or
                    row['max_sampling_map_difference_px'] is None or
                    row['max_sampling_map_difference_px'] >= 1.
                    for row in iteration_rows) or
                    network_counter.returned_calls - before_returned != 5):
                raise RuntimeError('Model smoke refinement did not complete five valid map/PnP/network iterations')
            candidate = refinement.candidate
            candidate_pose = checked_pose(candidate.pose)
            candidate_points = np.asarray(candidate.points_object_m)
            candidate_pixels = np.asarray(candidate.pixels_image)
            candidate_weights = np.asarray(candidate.weights)
            if (not np.isfinite(candidate_pose).all() or
                    candidate_points.ndim != 2 or candidate_points.shape[1] != 3 or
                    candidate_pixels.ndim != 2 or candidate_pixels.shape[1] != 2 or
                    candidate_weights.ndim != 1 or
                    not (len(candidate_points) == len(candidate_pixels) == len(candidate_weights)) or
                    not np.isfinite(candidate_points).all() or
                    not np.isfinite(candidate_pixels).all() or
                    not np.isfinite(candidate_weights).all()):
                raise ValueError('Model smoke candidate pose and correspondence arrays must be finite')
            candidate_points_descriptor = _model_smoke_array_descriptor(candidate_points)
            candidate_pixels_descriptor = _model_smoke_array_descriptor(candidate_pixels)
            candidate_weights_descriptor = _model_smoke_array_descriptor(candidate_weights)

            active['phase'] = 'validation'
            validation_started = time.perf_counter_ns()
            validation_result = quality_contract.validate_model_smoke(
                candidate, frame, gpu_mask, settings=ValidationSettings())
            case_timing['validation'] = (
                time.perf_counter_ns() - validation_started) / 1_000_000.
            validation = _model_smoke_validation_record(
                validation_result, required_success=False)
            refinement_record = {
                'candidate_present': True,
                'iterations': iteration_rows,
                'pose_camera_from_object_m': candidate_pose.tolist(),
                'points_object_m_sha256': candidate_points_descriptor['sha256'],
                'pixels_native_sha256': candidate_pixels_descriptor['sha256'],
                'weights_sha256': candidate_weights_descriptor['sha256'],
            }
            native_render = {
                'gpu_rgb_sha256': native_event['arrays']['model_rgb_u8']['sha256'],
                'gpu_depth_mm_sha256': native_event['arrays']['depth_mm']['sha256'],
                'gpu_mask_sha256': native_event['arrays']['mask']['sha256'],
                'cpu_rgb_sha256': native_event['arrays']['cpu_rgb_u8']['sha256'],
                'cpu_depth_m_sha256': native_event['arrays']['cpu_depth_m']['sha256'],
                'cpu_mask_sha256': native_event['arrays']['cpu_mask']['sha256'],
                'mask_iou': iou,
                'median_common_depth_error_m': median_depth_error,
                'common_depth_pixel_count': common_count,
                'mask_equals_depth_positive': True,
                'native_dimensions': [720, 720],
                'coordinate_mode': 'integer_centers_v1',
                'render_policy': 'capture_zero_sample_v1',
                'unlit_templates': flags['unlit_templates'],
                'disable_multisampling': flags['disable_multisampling'],
            }
            pending['rows'].append({
                'pose_id': case_id,
                'seed_camera_from_object_m': seed.tolist(),
                'render': native_render,
                'refinement': refinement_record,
                'validation': validation,
            })
            if validation['passed'] is not True:
                raise RuntimeError(
                    f'Public model smoke validation failed for case {case_id}: '
                    f"{validation['reason']}")

        active['case_id'] = 'c'
        active['iteration'] = None
        active['phase'] = 'validation'
        if (len(pending['constructor_records']) != 1 or
                len(pending['render_events']) != 18 or len(pending['rows']) != 3 or
                network_counter.attempted_calls != 15 or
                network_counter.returned_calls != 15 or
                any(event['network_inputs'] is None or
                    event['network_inputs']['template_matches_observed_render'] is not True or
                    event['network_inputs']['call_returned'] is not True
                    for event in pending['render_events'] if event['role'] == 'crop')):
            raise RuntimeError('Model smoke did not produce the complete three-view/18-render/15-call prefix')
        worker_succeeded = True
    except Exception as exc:
        pending['status'] = 'failed'
        pending['error'] = _model_smoke_error(exc)
        failure_iteration = active['iteration']
        if failure_iteration is None and observer is not None:
            current_scope = observer.current_scope
            if current_scope is not None and current_scope['role'] == 'crop':
                failure_iteration = current_scope['iteration']
        pending['failure_location'] = {
            'phase': active['phase'],
            'case_id': active['case_id'],
            'iteration': failure_iteration,
        }
    finally:
        if observer is not None or renderer is not None:
            active['phase'] = 'close'
            close_started = None
            try:
                if device == 'cuda':
                    torch.cuda.synchronize()
                close_started = time.perf_counter_ns()
            except Exception as exc:
                pending['close_error'] = _model_smoke_error(exc)
            try:
                if observer is not None:
                    observer.close()
                    pending['renderer_closed'] = observer.renderer_closed
                else:
                    renderer.close()
                    if renderer.render_policy_metadata is not None:
                        raise RuntimeError('Public renderer metadata remained live after close')
                    pending['renderer_closed'] = True
            except Exception as exc:
                if pending['close_error'] is None:
                    pending['close_error'] = _model_smoke_error(exc)
                if pending['error'] is None:
                    pending['failure_location'] = {
                        'phase': 'close', 'case_id': active['case_id'],
                        'iteration': active['iteration'],
                    }
            finally:
                if close_started is not None:
                    try:
                        runtime_evidence['timings_ms']['renderer_close'] = (
                            _model_smoke_timing_finish(torch, device, close_started))
                    except Exception as exc:
                        if pending['close_error'] is None:
                            pending['close_error'] = _model_smoke_error(exc)
                        if pending['error'] is None:
                            pending['failure_location'] = {
                                'phase': 'close', 'case_id': active['case_id'],
                                'iteration': active['iteration'],
                            }
        if observer is not None:
            runtime_evidence['render_calls'] = [
                dict(record) for record in observer._render_call_records]
        try:
            _model_smoke_collect_memory(runtime_evidence, device, torch)
        except Exception as exc:
            if pending['error'] is None:
                pending['error'] = _model_smoke_error(exc)
                pending['failure_location'] = {
                    'phase': 'validation', 'case_id': active['case_id'],
                    'iteration': active['iteration'],
                }
        pending['worker_ended_at_utc'] = _model_smoke_utc_now()
        runtime_evidence['timings_ms']['worker_total'] = (
            time.perf_counter_ns() - worker_started_ns) / 1_000_000.
        if (worker_succeeded and pending['error'] is None and
                pending['close_error'] is None and pending['renderer_closed'] is True):
            evidence_error = _model_smoke_runtime_evidence_error(runtime_evidence, device)
            if evidence_error is None:
                pending['status'] = 'measured_success'
                pending['failure_location'] = None
            else:
                pending['status'] = 'failed'
                pending['error'] = {'type': 'RuntimeError', 'message': evidence_error}
                pending['failure_location'] = {
                    'phase': 'validation', 'case_id': 'c', 'iteration': None,
                }
        else:
            pending['status'] = 'failed'
    return pending


def _physical_unmeasured_row(key, source_frame_id, reason, *, mask_record=None,
                             rgb_state='unmeasured'):
    row = dict(frameId=key.frame_id, sourceFrameId=source_frame_id,
               timestamp_s=key.timestamp_s, clock_mode=PHYSICAL,
               cameraFromObject=None, mask_state='lost', pose_state='lost',
               render_state='suppressed', failure_reason=reason, timings_ms={},
               validation=None, rgb_state=rgb_state, unmeasured_failure=True)
    if mask_record is not None:
        row['mask_state'] = mask_record.get('mask_state', 'lost')
        row['mask_measurement_state'] = mask_record.get('measurement_state')
        row['mask_path'] = mask_record.get('path')
        row['mask_sha256'] = mask_record.get('mask_sha256')
    else:
        row.update(mask_measurement_state='unmeasured', mask_path=None, mask_sha256=None)
    return row


def _pose_stage_physical(bundle, masks_root, output, device, mode, limit,
                         forced_occlusion, model_memory, unlit_templates,
                         appearance_check, disable_multisampling,
                         pnp_no_extrinsic_guess, manifest, mask_manifest,
                         mask_records):
    """Execute a fully joined physical-time pose prefix without dropping rows."""
    from dataclasses import asdict

    timing = manifest['_timing']
    policy = timing['time_policy']
    timeline = manifest['timeline']
    scored_keys = list(timing['scored_keys'])
    if mode not in ('controlled', 'complete'):
        raise ValueError('Pose mode must be controlled or complete')
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit > len(scored_keys):
            raise ValueError('Physical pose limit must select a nonempty scored prefix')
        requested_keys = scored_keys[:limit]
    else:
        requested_keys = scored_keys
    requested_set = {key.frame_id for key in requested_keys}
    setup_key = timing['setup_key']
    if mode == 'controlled':
        if setup_key is None or setup_key.frame_id != 0:
            raise ValueError('Physical controlled mode requires an ordinal-0 setup row')
        if not isinstance(manifest.get('controlled_initial_pose'), list):
            raise ValueError('Physical controlled mode requires its declared setup pose')
    if forced_occlusion is not None:
        if isinstance(forced_occlusion, bool) or not isinstance(forced_occlusion, int):
            raise ValueError('Physical forced-occlusion start must be an inference ordinal')
        if forced_occlusion not in {key.frame_id for key in scored_keys}:
            raise ValueError('Physical forced-occlusion start must name a scored inference ordinal')

    diagnostic = bool(mask_manifest.get('diagnostic_control'))
    if diagnostic and Path(output).name in ('complete.json', 'controlled.json'):
        raise ValueError('Diagnostic mask controls must not overwrite automatic benchmark results')

    required_mask_ids = ([setup_key.frame_id] if mode == 'complete' and setup_key is not None else []) + [
        key.frame_id for key in requested_keys]
    mask_requested = set(mask_manifest['requested_frame_ids'])
    if any(frame_id not in mask_requested for frame_id in required_mask_ids):
        raise ValueError('Physical mask cache requested prefix does not cover this pose prefix')

    # All timing, option and full-cache joins above complete before these model
    # imports or any decoder/image operations.
    import torch
    from .quality_gotrack import load_network, TexturedRenderer
    from .quality_foundpose import FoundPoseRecovery
    from .quality_appearance import AppearanceCheckedBackend, AppearanceSettings

    pnp_settings = _pnp_tracking_settings(manifest['object'], pnp_no_extrinsic_guess)
    smoke_path = CACHE/'smoke'/(f'{manifest["object"]}-{device}'+
        ('-unlit' if unlit_templates else '')+('-no-msaa' if disable_multisampling else '')+'.json')
    if not smoke_path.exists() or not _read_json_unique(smoke_path).get('smoke_passed'):
        raise RuntimeError('Run and pass quality_runner smoke before video evaluation')
    network = load_network(device)
    renderer = TexturedRenderer(bundle/manifest['asset'], manifest['object_id'],
        unlit=unlit_templates, disable_multisampling=disable_multisampling)
    appearance_renderer = None
    try:
        bank = torch.load(CACHE/'banks'/f'{manifest["object"]}-foundpose.pt',
                          map_location='cpu', weights_only=False)
        refiner = _make_gotrack_refiner(network, renderer, manifest['object_id'], device, pnp_settings)
        backend = FoundPoseRecovery(refiner, bank, model_memory=model_memory, time_policy=policy)
        appearance_settings = None
        if appearance_check:
            appearance_renderer = TexturedRenderer(bundle/manifest['asset'], manifest['object_id'],
                unlit=True, disable_multisampling=disable_multisampling)
            appearance_settings = AppearanceSettings()
            backend = AppearanceCheckedBackend(backend, appearance_renderer,
                appearance_settings, time_policy=policy)

        initial = manifest.get('controlled_initial_pose') if mode == 'controlled' else None
        tracker = SequentialTracker(backend, initial_pose=initial, time_policy=policy,
            initial_frame_id=0 if initial is not None else None,
            initial_timestamp_s=setup_key.timestamp_s if initial is not None else None)
        provenance = inference_provenance(bundle)
        provenance['foundpose_bank_sha256'] = digest(CACHE/'banks'/f'{manifest["object"]}-foundpose.pt')
        provenance['input_manifest_sha256'] = timing['manifest_sha256']
        provenance['timestamp_table_sha256'] = timing['timestamp_table_sha256']
        k = np.array(manifest['intrinsics'])
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        timeline_rows = {row['frame_id']: row for row in timeline}
        all_source_ids = timing['source_frame_ids']
        scored_id_list = [key.frame_id for key in scored_keys]
        result_rows = []
        initialization = None
        runtime = device_info(device)

        if mode == 'controlled':
            initialization = dict(frameId=setup_key.frame_id,
                sourceFrameId=all_source_ids[setup_key.frame_id],
                timestamp_s=setup_key.timestamp_s, clock_mode=PHYSICAL,
                pose_state='controlled_seed', render_state='suppressed')
        elif setup_key is not None:
            setup_record = mask_records[setup_key.frame_id]
            try:
                setup_rgb = read_rgb(bundle, manifest, setup_key.frame_id)
            except Exception as exc:
                initialization = tracker.update_unavailable(setup_key,
                    reason='rgb_decode_failed')
                initialization.update(sourceFrameId=all_source_ids[setup_key.frame_id],
                    rgb_state='decode_failed', mask_path=setup_record.get('path'),
                    mask_sha256=setup_record.get('mask_sha256'))
            else:
                setup_frame = Frame(setup_key.frame_id, setup_rgb, k,
                                    setup_key.timestamp_s, PHYSICAL)
                setup_mask = None
                setup_reason = setup_record.get('failure_reason')
                if setup_record.get('mask_state') == 'available':
                    try:
                        setup_mask = cv2.imread(str(Path(masks_root)/setup_record['path']),
                                                cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        setup_mask = None
                    if setup_mask is None:
                        setup_reason = 'mask_decode_failed'
                setup_state = setup_record.get('mask_state')
                if setup_mask is None and setup_state == 'available':
                    setup_state = 'lost'
                setup_segmentation = MaskPrediction(setup_key.frame_id, setup_mask,
                    setup_state, setup_reason or ('mask_unmeasured' if setup_mask is None else None),
                    setup_record.get('timings_ms', {}), setup_key.timestamp_s, PHYSICAL)
                initialization = tracker.update(setup_frame, setup_segmentation)
                initialization.update(sourceFrameId=all_source_ids[setup_key.frame_id],
                    rgb_state='decoded', mask_measurement_state=setup_record.get('measurement_state'),
                    mask_path=setup_record.get('path'), mask_sha256=setup_record.get('mask_sha256'))

        def save_physical_result(stage_completed):
            full_window = len(requested_keys) == len(scored_keys) and stage_completed
            segmentation_complete = (mask_manifest.get('status') == 'complete' and
                                     mask_manifest.get('stage_completed') is True)
            pipeline_complete = full_window and segmentation_complete
            result = dict(schema_version=2, clock_mode=PHYSICAL,
                input_manifest_sha256=timing['manifest_sha256'],
                timestamp_table_sha256=timing['timestamp_table_sha256'],
                mode=mode, object=manifest['object'], runtime=runtime,
                source_frame_ids={str(k): v for k, v in all_source_ids.items()},
                planned_frame_ids=scored_id_list,
                frame_ids=scored_id_list,
                requested_frame_ids=[key.frame_id for key in requested_keys],
                expected_setup_frames=sum(row['role'] == 'setup' for row in timeline),
                expected_scored_frames=len(scored_keys), coverage_complete=len(result_rows) == len(scored_keys),
                stage_completed=bool(stage_completed), complete=bool(pipeline_complete),
                mask_cache_status=mask_manifest.get('status'),
                mask_cache_stage_completed=mask_manifest.get('stage_completed'),
                segmentation_complete=segmentation_complete,
                status=('failed' if not stage_completed or mask_manifest.get('status') == 'failed' else
                        'complete' if pipeline_complete else 'diagnostic_prefix'),
                frames=result_rows, initialization=initialization,
                provenance=provenance,
                tracking_settings={**asdict(tracker.settings), **pnp_settings,
                    'time_policy': asdict(policy), 'model_memory': model_memory,
                    'motion_bounds_clock': 'elapsed_seconds',
                    'max_angular_rate_deg_s': policy.max_angular_rate_deg_s,
                    'max_translation_rate_m_s': policy.max_translation_rate_m_s,
                    'private_pose_memory_s': policy.private_pose_memory_s,
                    'inactive_legacy_fields': ['max_rotation_degrees_per_frame',
                        'max_translation_m_per_frame', 'private_pose_memory_frames'],
                    'unlit_templates': unlit_templates, 'appearance_check': appearance_check,
                    'disable_multisampling': disable_multisampling, 'mask_association': False,
                    'appearance_render_mode': 'unlit' if appearance_check else None,
                    'appearance_settings': None if appearance_settings is None else asdict(appearance_settings)},
                automatic=not diagnostic, diagnostic_control=diagnostic,
                stress_test=None if forced_occlusion is None else
                    {'occlusion_start': forced_occlusion, 'source_frames': 15})
            save_result(output, result)

        # Retain setup failures in initialization and every scored row in frames.
        if initialization is not None and initialization.get('terminal'):
            # The current physical cleanup state is uncertain.  Do not decode any
            # scored image or mask after it; preserve them as explicit failures.
            for key in scored_keys:
                record = mask_records[key.frame_id]
                reason = ('motion_cleanup_unconfirmed' if key.frame_id in requested_set
                          else 'outside_requested_prefix')
                result_rows.append(_physical_unmeasured_row(key,
                    all_source_ids[key.frame_id], reason,
                    mask_record=record))
            save_physical_result(False)
            return

        completed_requested = 0
        for index, key in enumerate(scored_keys):
            row_info = timeline_rows[key.frame_id]
            record = mask_records[key.frame_id]
            source_id = all_source_ids[key.frame_id]
            if key.frame_id not in requested_set:
                result_rows.append(_physical_unmeasured_row(key, source_id,
                    'outside_requested_prefix', mask_record=record))
                continue
            if tracker.terminal:
                result_rows.append(_physical_unmeasured_row(key, source_id,
                    'motion_cleanup_unconfirmed', mask_record=record))
                continue
            try:
                rgb = read_rgb(bundle, manifest, key.frame_id)
            except Exception:
                result = tracker.update_unavailable(key, reason='rgb_decode_failed')
                result.update(sourceFrameId=source_id, timestamp_s=key.timestamp_s,
                    clock_mode=PHYSICAL, rgb_state='decode_failed',
                    mask_measurement_state=record.get('measurement_state'),
                    mask_path=record.get('path'), mask_sha256=record.get('mask_sha256'))
            else:
                if forced_occlusion is not None and forced_occlusion <= key.frame_id < forced_occlusion+15:
                    rgb[:] = 0
                frame = Frame(key.frame_id, rgb, k, key.timestamp_s, PHYSICAL)
                mask = None
                reason = record.get('failure_reason')
                if record.get('mask_state') == 'available':
                    try:
                        mask = cv2.imread(str(Path(masks_root)/record['path']), cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        mask = None
                    if mask is None:
                        reason = 'mask_decode_failed'
                mask_state = record.get('mask_state')
                if mask is None and mask_state == 'available':
                    mask_state = 'lost'
                if mask_state == 'lost' and reason is None:
                    reason = ('mask_unmeasured' if record.get('measurement_state') == 'unmeasured'
                              else 'mask_unavailable')
                segmentation = MaskPrediction(key.frame_id, mask, mask_state, reason,
                    record.get('timings_ms', {}), key.timestamp_s, PHYSICAL)
                result = tracker.update(frame, segmentation)
                result.update(sourceFrameId=source_id, timestamp_s=key.timestamp_s,
                    clock_mode=PHYSICAL, rgb_state='decoded',
                    mask_measurement_state=record.get('measurement_state'))
            result['mask_path'] = record.get('path')
            result['mask_sha256'] = record.get('mask_sha256')
            result_rows.append(result)
            completed_requested += 1
            if tracker.terminal:
                # Remaining requested rows are kept below as suppressed failures.
                pass
            if completed_requested % 40 == 1:
                print(manifest['object'], mode, completed_requested, '/', len(requested_keys),
                      result.get('pose_state'), flush=True)
            save_physical_result(completed_requested == len(requested_keys) and not tracker.terminal)
        stage_completed = completed_requested == len(requested_keys) and not tracker.terminal
        # Write only after the full planned score map includes prefix and terminal
        # suffix rows, so the final receipt always preserves every scored key.
        save_physical_result(stage_completed)
    finally:
        renderer.close()
        if appearance_renderer is not None:
            appearance_renderer.close()


def pose_stage(bundle, masks_root, output, device, mode, limit=None, forced_occlusion=None, model_memory=False,
               unlit_templates=False, appearance_check=False, disable_multisampling=False,
               pnp_no_extrinsic_guess=False, chronological_mug_capture_root=None,
               chronological_mug_capture_branch=None):
    manifest = read_input(bundle)
    if manifest.get('_capture') is not None:
        if (mode != 'complete' or forced_occlusion is not None or model_memory or
                appearance_check or pnp_no_extrinsic_guess or
                chronological_mug_capture_root is not None or chronological_mug_capture_branch is not None):
            raise ValueError('Capture pose currently supports only the bounded complete-mode option set')
        scored_count = len(manifest['_timing']['scored_keys'])
        if limit is not None and (type(limit) is not int or limit <= 0 or limit > scored_count):
            raise ValueError('Capture pose limit must select a nonempty scored prefix')
        from .quality_assets import capture_stage_resources
        capture_stage_resources(
            manifest['_capture'], stage='pose',
            render_flags={'unlit_templates': unlit_templates,
                          'disable_multisampling': disable_multisampling},
            device=device,
        )
        raise ValueError('Capture pose stage body is unavailable until Wave3 integration')
    if manifest['_timing']['clock_mode'] == PHYSICAL:
        if chronological_mug_capture_root is not None or chronological_mug_capture_branch is not None:
            raise ValueError('Chronological capture does not support physical time')
        pnp_settings = _pnp_tracking_settings(manifest['object'], pnp_no_extrinsic_guess)
        mask_manifest, mask_records = _physical_mask_cache(manifest, masks_root)
        return _pose_stage_physical(bundle, masks_root, output, device, mode, limit,
            forced_occlusion, model_memory, unlit_templates, appearance_check,
            disable_multisampling, pnp_no_extrinsic_guess, manifest,
            mask_manifest, mask_records)
    manifest = manifest
    pnp_settings = _pnp_tracking_settings(manifest['object'], pnp_no_extrinsic_guess)
    capture_requested = chronological_mug_capture_root is not None or chronological_mug_capture_branch is not None
    capture_output = None
    capture_trace = None
    if capture_requested:
        if chronological_mug_capture_root is None or chronological_mug_capture_branch not in ('control', 'candidate'):
            raise ValueError('Chronological mug capture requires both capture root and known branch')
        if (manifest['object'] != 'mug' or mode != 'complete' or limit != 201 or forced_occlusion is not None or
                model_memory or appearance_check or not unlit_templates or not disable_multisampling):
            raise ValueError('Chronological capture requires the fixed complete 201-frame unlit/no-MSAA mug run')
        expected_guess = chronological_mug_capture_branch == 'control'
        if pnp_settings['pnp_use_extrinsic_guess'] is not expected_guess:
            raise ValueError('Chronological branch PnP setting differs from the prepared control/candidate design')
        capture_root = Path(chronological_mug_capture_root).resolve()
        capture_output = capture_root / f'{chronological_mug_capture_branch}-201.json'
        capture_trace = capture_root / f'{chronological_mug_capture_branch}-201.trace.jsonl'
        if Path(output).resolve() != capture_output:
            raise ValueError('Chronological pose output must match the prepared branch path')
        configured_trace = os.environ.get('VISUALIZEIT_QUALITY_TRACE')
        if configured_trace is not None and Path(configured_trace).resolve() != capture_trace:
            raise ValueError('Chronological trace environment path differs from the prepared branch path')
        os.environ['VISUALIZEIT_QUALITY_TRACE'] = str(capture_trace)
    import torch
    from .quality_gotrack import load_network, TexturedRenderer
    from .quality_foundpose import FoundPoseRecovery
    smoke_path = CACHE/'smoke'/(f'{manifest["object"]}-{device}'+('-unlit' if unlit_templates else '')+('-no-msaa' if disable_multisampling else '')+'.json')
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get('smoke_passed'):
        raise RuntimeError('Run and pass quality_runner smoke before video evaluation')
    mask_manifest = json.loads((masks_root/'results.json').read_text())
    diagnostic = bool(mask_manifest.get('diagnostic_control'))
    if diagnostic and output.name in ('complete.json', 'controlled.json'):
        raise ValueError('Diagnostic mask controls must not overwrite automatic benchmark results')
    mask_records = {r['frameId']: r for r in mask_manifest['frames']}
    network = load_network(device); renderer = TexturedRenderer(bundle/manifest['asset'], manifest['object_id'], unlit=unlit_templates, disable_multisampling=disable_multisampling)
    bank = torch.load(CACHE/'banks'/f'{manifest["object"]}-foundpose.pt', map_location='cpu', weights_only=False)
    capture = None
    if capture_requested:
        from .quality_mug_chronological_capture import ChronologicalCaptureBackend, MugChronologicalCapture
        capture = MugChronologicalCapture(chronological_mug_capture_root,
            chronological_mug_capture_branch, capture_trace, bundle=bundle, masks_root=masks_root)
    refiner = _make_gotrack_refiner(network, renderer, manifest['object_id'], device, pnp_settings,
        chronological_capture_callback=capture)
    backend = FoundPoseRecovery(refiner, bank, model_memory=model_memory)
    if capture is not None:
        backend = ChronologicalCaptureBackend(backend, refiner, capture)
    appearance_renderer = None; appearance_settings = None
    if appearance_check:
        from .quality_appearance import AppearanceCheckedBackend, AppearanceSettings
        appearance_renderer = TexturedRenderer(bundle/manifest['asset'], manifest['object_id'], unlit=True, disable_multisampling=disable_multisampling)
        appearance_settings = AppearanceSettings()
        backend = AppearanceCheckedBackend(backend, appearance_renderer, appearance_settings)
    initial = manifest['controlled_initial_pose'] if mode == 'controlled' else None
    tracker = SequentialTracker(backend, initial_pose=initial)
    from dataclasses import asdict
    provenance = inference_provenance(bundle)
    provenance['foundpose_bank_sha256'] = digest(CACHE/'banks'/f'{manifest["object"]}-foundpose.pt')
    k = np.array(manifest['intrinsics']); output.parent.mkdir(parents=True, exist_ok=True)
    initialization = None
    if mode == 'complete':
        setup_id = manifest['setup_frame_id']; record = mask_records[setup_id]
        setup_mask = cv2.imread(str(masks_root/record['path']), cv2.IMREAD_GRAYSCALE)
        setup_frame = Frame(setup_id, read_rgb(bundle, manifest, setup_id), k)
        setup_segmentation = MaskPrediction(setup_id, setup_mask, record['mask_state'],
            record.get('failure_reason'), record.get('timings_ms', {}))
        if capture is not None:
            capture.begin_frame(setup_frame, setup_mask, tracker.accepted is not None,
                record.get('path'), digest(masks_root/record['path']), record['mask_state'], record.get('failure_reason'))
        initialization = tracker.update(setup_frame, setup_segmentation)
        if capture is not None:
            capture.end_frame(initialization)
    ids = manifest['frame_ids'][:limit]; frames = []
    cap = cv2.VideoCapture(str(bundle/manifest['video'])); cap.set(cv2.CAP_PROP_POS_FRAMES, ids[0])
    try:
        for i, fid in enumerate(ids):
            ok, bgr = cap.read()
            if not ok: raise ValueError('Incomplete source video')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if forced_occlusion is not None and forced_occlusion <= fid < forced_occlusion+15: rgb[:] = 0
            record = mask_records.get(fid)
            if record is None: raise ValueError('Missing cached mask frame; cannot silently drop frame')
            mask = cv2.imread(str(masks_root/record['path']), cv2.IMREAD_GRAYSCALE)
            segmentation = MaskPrediction(fid, mask, record['mask_state'], record.get('failure_reason'), record.get('timings_ms', {}))
            frame = Frame(fid, rgb, k)
            mask_hash = digest(masks_root/record['path']) if capture is not None else None
            if capture is not None:
                capture.begin_frame(frame, mask, tracker.accepted is not None, record['path'], mask_hash,
                    segmentation.state, segmentation.reason)
            try: result = tracker.update(frame, segmentation)
            except (RuntimeError, ValueError, cv2.error) as e:
                if device == 'cuda' and ('out of memory' in str(e).lower() or 'cuda' in str(e).lower()):
                    raise RuntimeError('GPU execution failed; rerun this same stage on CPU: '+str(e)) from e
                tracker.accepted = None; tracker.pending = None
                result = dict(frameId=fid, cameraFromObject=None, mask_state=segmentation.state, pose_state='lost',
                    render_state='suppressed', failure_reason='pose_runtime_failure: '+str(e), timings_ms=segmentation.timings_ms)
            if capture is not None:
                capture.end_frame(result)
            result['mask_path'] = record['path']; result['mask_sha256'] = mask_hash if capture is not None else digest(masks_root/record['path']); frames.append(result)
            save_result(output, dict(schema_version=1, mode=mode, object=manifest['object'], runtime=device_info(device),
                frame_ids=ids, complete=len(frames) == len(manifest['frame_ids']), frames=frames,
                provenance=provenance, tracking_settings={**asdict(tracker.settings), **pnp_settings,
                    'model_memory': model_memory,
                    'unlit_templates': unlit_templates, 'appearance_check': appearance_check,
                    'disable_multisampling': disable_multisampling,
                    'mask_association': mask_manifest.get('mask_association',False),
                    'appearance_render_mode': 'unlit' if appearance_check else None,
                    'appearance_settings': None if appearance_settings is None else asdict(appearance_settings)},
                independent_accuracy_scored=False, automatic=not diagnostic, diagnostic_control=diagnostic, initialization=initialization,
                stress_test=None if forced_occlusion is None else {'occlusion_start': forced_occlusion, 'source_frames': 15}))
            if i % 40 == 0: print(manifest['object'], mode, i+1, '/', len(ids), result['pose_state'], flush=True)
    finally:
        cap.release(); renderer.close()
        if appearance_renderer is not None: appearance_renderer.close()
        if capture is not None:
            try:
                capture.finish_branch(output)
            except Exception as e:
                try:
                    refiner._disable_chronological_capture(capture,
                        f'branch finalization failed: {type(e).__name__}: {e}')
                except Exception:
                    pass


def _validate_capture_stage_options(args, manifest):
    """Reject capture options outside the frozen Wave1 policy before stage effects."""
    planned_count = len(manifest['_capture'].rows)
    if args.mode != 'complete':
        raise ValueError('Capture pilot stages require complete mode; controlled capture is unsupported')
    if (args.force_occlusion is not None or args.model_memory or args.appearance_check or
            args.pnp_no_extrinsic_guess or args.mask_association or
            args.detection_prior is not None or args.prior_frame is not None or
            args.chronological_mug_capture_root is not None or
            args.chronological_mug_capture_branch is not None):
        raise ValueError('Capture stage options include an unsupported physical or diagnostic mode')
    if args.stage in ('segment', 'pose'):
        if args.limit is not None and (type(args.limit) is not int or args.limit <= 0 or args.limit > planned_count):
            raise ValueError('Capture stage limit must be a nonempty planned prefix')
    elif args.limit is not None:
        raise ValueError('Capture limit is available only for segment or pose')
    if args.stage == 'detect':
        if type(args.frame) is not int or args.frame < 0 or args.frame >= planned_count:
            raise ValueError('--frame must name a planned capture ordinal')
    elif args.frame is not None:
        raise ValueError('--frame is available only for detect')
    if args.stage == 'pose' and args.masks is None:
        raise ValueError('--masks is required for pose')
    if args.stage != 'pose' and args.masks is not None:
        raise ValueError('--masks is available only for pose')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'banks', 'cnos-bank', 'segment', 'detect', 'pose'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--mode', choices=['controlled', 'complete'], default='complete')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--force-occlusion', type=int)
    parser.add_argument('--model-memory', action='store_true', help='Use private fixed-model motion evidence; never display optical-flow-only poses')
    parser.add_argument('--unlit-templates', action='store_true', help='Experimental stable texture-only tracking renders; does not change the mapped design viewer')
    parser.add_argument('--appearance-check', action='store_true', help='Experimental current-image texture validation of GoTrack hypotheses')
    parser.add_argument('--disable-multisampling', action='store_true', help='Experimental repeatable fixed-model rendering; separate smoke and prefix gates required')
    parser.add_argument('--pnp-no-extrinsic-guess', action='store_true', help='Experimental mug-only GoTrack RANSAC initialization ablation')
    parser.add_argument('--mask-association',action='store_true',help='Opt-in current-proposal association with a recent observed object mask')
    parser.add_argument('--detection-prior',type=Path)
    parser.add_argument('--prior-frame',type=int)
    parser.add_argument('--chronological-mug-capture-root', type=Path,
        help='Opt-in private V5 chronological observer root; prepared with quality_mug_chronological_capture prepare')
    parser.add_argument('--chronological-mug-capture-branch', choices=['control', 'candidate'])
    args = parser.parse_args()
    if args.stage == 'pose' and args.masks is None:
        parser.error('--masks is required for pose')
    manifest = read_input(args.bundle)
    if manifest.get('_capture') is not None:
        try:
            _validate_capture_stage_options(args, manifest)
            from .quality_assets import capture_stage_resources
            capture_resources = capture_stage_resources(
                manifest['_capture'], stage=args.stage,
                render_flags={'unlit_templates': args.unlit_templates,
                              'disable_multisampling': args.disable_multisampling},
                device=args.device,
            )
            if args.stage == 'smoke':
                _model_smoke_trial_output_path(args.output, capture_resources.output)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.stage == 'smoke':
            try:
                pending = capture_model_smoke(
                    args.bundle, manifest, capture_resources, device=args.device)
                _model_smoke_write_pending(
                    args.output, capture_resources.output, pending)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                parser.error(str(exc))
            if pending['status'] != 'measured_success':
                raise SystemExit(1)
            return
        parser.error('Capture stage body is unavailable until Wave3 integration')
    is_physical = manifest['_timing']['clock_mode'] == PHYSICAL
    if args.pnp_no_extrinsic_guess:
        if args.stage != 'pose': parser.error('--pnp-no-extrinsic-guess is available only for the pose stage')
        if args.masks is None: parser.error('--masks is required for pose')
        try:
            _pnp_tracking_settings(manifest['object'], True)
        except ValueError as e:
            parser.error(str(e))
    if args.chronological_mug_capture_root is not None or args.chronological_mug_capture_branch is not None:
        if args.stage != 'pose': parser.error('Chronological mug capture is available only for the pose stage')
        if args.chronological_mug_capture_root is None or args.chronological_mug_capture_branch is None:
            parser.error('Both chronological mug capture flags are required')
    if args.chronological_mug_capture_root is not None or args.chronological_mug_capture_branch is not None:
        if is_physical:
            parser.error('Chronological capture does not support physical time')
    if args.mask_association and args.stage != 'segment':
        parser.error('--mask-association is available only for the segment stage')
    if is_physical and args.mask_association:
        parser.error('Physical mask association is unsupported until its prior-age policy uses elapsed seconds')
    if args.stage == 'detect':
        if args.frame is None:
            parser.error('--frame is required for detect')
        if args.frame not in manifest['_timing']['source_frame_ids']:
            parser.error('--frame must name a planned inference frame (physical ordinal in physical mode)')
        if (args.detection_prior is None) != (args.prior_frame is None):
            parser.error('--prior-frame and --detection-prior must be supplied together')
        if is_physical and (args.detection_prior is not None or args.prior_frame is not None):
            parser.error('Physical detector priors are unsupported while association uses frame age')
    elif args.detection_prior is not None or args.prior_frame is not None:
        parser.error('--detection-prior and --prior-frame are available only for detect')
    if is_physical and args.stage == 'segment':
        parser.error('Physical mask generation requires the pending schema2 cache producer')
    physical_mask_cache = None
    if is_physical and args.stage == 'pose':
        try:
            _pnp_tracking_settings(manifest['object'], args.pnp_no_extrinsic_guess)
            physical_mask_cache = _physical_mask_cache(manifest, args.masks)
            if args.mode not in ('controlled', 'complete'):
                raise ValueError('Pose mode must be controlled or complete')
            scored = manifest['_timing']['scored_keys']
            if args.limit is not None and (args.limit <= 0 or args.limit > len(scored)):
                raise ValueError('Physical pose limit must select a nonempty scored prefix')
            requested_pose = scored if args.limit is None else scored[:args.limit]
            if args.mode == 'controlled':
                if manifest['_timing']['setup_key'] is None:
                    raise ValueError('Physical controlled mode requires an ordinal-0 setup row')
                if not isinstance(manifest.get('controlled_initial_pose'), list):
                    raise ValueError('Physical controlled mode requires its declared setup pose')
            required_mask_ids = ([manifest['_timing']['setup_key'].frame_id]
                if args.mode == 'complete' and manifest['_timing']['setup_key'] is not None else []) + [
                    key.frame_id for key in requested_pose]
            cache_requested = set(physical_mask_cache[0]['requested_frame_ids'])
            if any(frame_id not in cache_requested for frame_id in required_mask_ids):
                raise ValueError('Physical mask cache requested prefix does not cover this pose prefix')
            if (physical_mask_cache[0].get('diagnostic_control') and
                    args.output.name in ('complete.json', 'controlled.json')):
                raise ValueError('Diagnostic mask controls must not overwrite automatic benchmark results')
            if args.force_occlusion is not None and args.force_occlusion not in manifest['frame_ids']:
                raise ValueError('Physical forced-occlusion start must name a scored inference ordinal')
        except ValueError as e:
            parser.error(str(e))
    if args.stage == 'smoke' and not isinstance(manifest.get('controlled_initial_pose'), list):
        parser.error('Setup smoke requires a declared controlled_initial_pose')
    cv2.setNumThreads(1)
    # Pyrender selects its hidden pyglet context when the variable is absent.
    # "pyglet" is not a supported PYOPENGL_PLATFORM value.
    if os.name != 'nt': os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import random
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device_info(args.device)
    if args.stage == 'smoke':
        from .quality_gotrack import upstream_path
        upstream_path(); smoke(args.bundle, args.output, args.device, args.unlit_templates,
            args.disable_multisampling)
    elif args.stage == 'banks':
        from .quality_gotrack import load_network, TexturedRenderer
        from .quality_foundpose import build_bank
        network = load_network(args.device)
        renderer = TexturedRenderer(args.bundle/manifest['asset'], manifest['object_id'], unlit=args.unlit_templates)
        try: build_bank(renderer, network.backbone, args.device, args.output)
        finally: renderer.close()
    elif args.stage == 'cnos-bank':
        from .quality_cnos import build_descriptors
        build_descriptors(args.bundle, args.device)
    elif args.stage == 'segment':
        from .quality_sam2 import run
        interval = None if args.force_occlusion is None else (args.force_occlusion, args.force_occlusion+15)
        run(args.bundle, args.output, args.device, args.limit, interval,args.mask_association)
    elif args.stage == 'detect':
        rgb = read_rgb(args.bundle, manifest, args.frame)
        prior=None;age=None
        if args.detection_prior is not None:
            prior=cv2.imread(str(args.detection_prior),cv2.IMREAD_GRAYSCALE)
            if prior is None:raise ValueError('Observed detection prior missing')
            age=args.frame-args.prior_frame
        from .quality_cnos import detect
        mask, info = detect(args.bundle, rgb, args.device,prior,age)
        if mask is None: mask = np.zeros(rgb.shape[:2], np.uint8)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = dict(mask=mask, reason=info.get('reason') or '', diagnostics=json.dumps(info))
        if is_physical:
            key = _frame_key(manifest, args.frame)
            fields.update(frameId=key.frame_id,
                sourceFrameId=manifest['_timing']['source_frame_ids'][key.frame_id],
                timestamp_s=key.timestamp_s, clock_mode=PHYSICAL,
                input_manifest_sha256=manifest['_timing']['manifest_sha256'],
                timestamp_table_sha256=manifest['_timing']['timestamp_table_sha256'])
        np.savez_compressed(args.output, **fields)
    elif args.stage == 'pose':
        pose_stage(args.bundle, args.masks, args.output, args.device, args.mode, args.limit, args.force_occlusion,
                   args.model_memory, args.unlit_templates, args.appearance_check, args.disable_multisampling,
                   args.pnp_no_extrinsic_guess,
                   chronological_mug_capture_root=args.chronological_mug_capture_root,
                   chronological_mug_capture_branch=args.chronological_mug_capture_branch)
    if sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()) > BUDGET: raise ValueError('8 GiB cache budget exceeded')


if __name__ == '__main__': main()
