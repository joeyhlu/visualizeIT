"""Opt-in, bounded GoTrack onset capture and V2 trajectory gate for the mug.

This module stores actual chronological refiner inputs/results only. It never
loads evaluation references or annotations, and it does not score surface
identity. Normal pose runs do not import or instantiate the observer.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tempfile

import numpy as np

from .quality_assets import BUDGET, CACHE, ROOT, digest, inference_provenance, save_result


SETUP_FRAME_ID = 826
RUN_FRAME_IDS = tuple(range(827, 1028))
CAPTURE_FRAME_IDS = tuple(range(1016, 1028))
CAPTURE_ITERATION = 4
CAPTURE_EXPERIMENT = 'mug-chronological-identity-resource-v2'
CAPTURE_SCHEMA_VERSION = 2
CAPTURE_BUDGET_BYTES = 320 * 1024**2
OLD_V1_CAPTURE_BUDGET_BYTES = 64 * 1024**2
METADATA_ARTIFACT_BUDGET_BYTES = 64 * 1024**2
WORKING_COPY_BUDGET_BYTES = 128 * 1024**2
CACHE_PREPARE_RESERVATION_BYTES = CAPTURE_BUDGET_BYTES + METADATA_ARTIFACT_BUDGET_BYTES
MODEL_CACHE_BUDGET_BYTES = BUDGET
CAPACITY_SPEC_PATH = 'docs/vision-next-experiment.md'
CAPACITY_SPEC_SHA256 = '7bb62f26f1f36bc2be518781a21abf3ca607dba304202d94053a13ff039aded5'
MAX_FULL_CORRESPONDENCES = 280 * 280
MAX_SAMPLED_CORRESPONDENCES = 10_000
MAX_RETAINED_PACKET_BYTES = 9_066_224
MAX_RESULT_PACKET_BYTES = 160_304
MAX_PACKET_PAIR_BYTES = 9_226_528
MAX_NATIVE_CONTEXT_BYTES = 5_242_952
MAX_PACKET_PAIRS = 24
MAX_NATIVE_CONTEXTS = 12
MAX_PLANNED_CAPTURE_BYTES = 284_352_096
MAX_PLANNED_CAPTURE_HEADROOM_BYTES = CAPTURE_BUDGET_BYTES - MAX_PLANNED_CAPTURE_BYTES
VALIDATION_ATOL = 1e-10
VALIDATION_RTOL = 1e-9
BRANCHES = ('control', 'candidate')
BRANCH_GUESS = {'control': True, 'candidate': False}
V2_STAGE_IDS = tuple(f'{branch}-{length}' for branch in BRANCHES for length in (30, 120, 240))
INSTRUMENTED_SOURCE_PATHS = ('bench/quality_gotrack.py', 'bench/quality_runner.py')
V2_INSTRUMENTATION_BASELINES = {
    'bench/quality_gotrack.py': 'd38f4a94523f0ffebc2d5c24af7470a54a08eda03a8f467c8181f25ae9ef5e1f',
    'bench/quality_runner.py': '4cf5a4f69fd82d7eaf62c1a97adb49a5d27d2fdc760adf13a208d0d23fb0d0dc',
}
V2_AUXILIARY_SOURCE_PATH = 'bench/quality_annotations.py'
V2_AUXILIARY_SOURCE_BASELINE = '48b8e67f6eced0cf8b28ab4b23ad2f96673d5defc76a38aa8f73ae47ca0183ba'
CURRENT_AUXILIARY_SOURCE = 'dd2ad2a9cf0807e7c3e2745ce733155a960bee0abe7bea934406f73c524a881c'
LEGACY_SAM2_PRODUCER_PATH = '.cache/model-quality/experiments/appearance-v2-code/quality_sam2.py'
LEGACY_SAM2_PRODUCER_SHA256 = '9cf8bf46920dfa43b75eb6f5e873c7350890b5efac54bebd9b5f9eaa9cfcd363'
COMPATIBILITY_REVIEW_SCOPE = 'mug-chronological-inference-source-and-legacy-mask-review'
INSTRUMENTATION_DELTA_CLASSIFICATION = 'reviewed opt-in chronological observer instrumentation only'
AUXILIARY_DELTA_CLASSIFICATION = 'separately reviewed quality_annotations.py delta; outside the V5 pose-stage runtime path'
LEGACY_MISSING_MODE_FLAGS = ('automatic', 'diagnostic_control', 'stress_test')
LEGACY_REVIEW_KEYS = frozenset({'manifest_path', 'manifest_sha256', 'input_manifest_sha256',
    'producer_path', 'producer_sha256', 'provenance_sha256', 'missing_flags', 'mask_rows'})
LEGACY_MASK_ROW_KEYS = frozenset({'frame_id', 'path', 'sha256', 'width', 'height', 'state'})
REQUIRED_V4_SETTINGS = {
    'model_memory': False,
    'unlit_templates': True,
    'appearance_check': False,
    'disable_multisampling': True,
}


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return _sha_bytes(raw)


def _array_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _array_nbytes(value):
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return int(value.numel()) * int(value.element_size())
    array = np.asarray(value)
    return int(array.nbytes)


def _safe_relative_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('A nonempty relative path is required')
    parsed = PurePosixPath(relative.replace('\\', '/'))
    if parsed.is_absolute() or '..' in parsed.parts or not parsed.parts:
        raise ValueError(f'Unsafe repository-relative path: {relative!r}')
    base = Path(root).resolve()
    target = (base / Path(*parsed.parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f'Path escapes its bound root: {relative!r}') from exc
    return target


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_exclusive(path, value, artifact_root=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if artifact_root is not None:
        _check_metadata_artifact_budget(artifact_root, len(encoded.encode('utf-8')))
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def _atomic_json(path, value, artifact_root=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if artifact_root is not None:
        _check_metadata_artifact_budget(artifact_root, len(encoded.encode('utf-8')))
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(encoded, encoding='utf-8')
    temporary.replace(path)


def _existing_cache_bytes(cache_root=CACHE):
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return 0
    total = 0
    for path in cache_root.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError as exc:
            raise ValueError(f'Cannot account for existing model-quality cache file {path}: {exc}') from exc
    return total


def _preflight_model_cache_budget(cache_root=CACHE,
                                  reservation_bytes=CACHE_PREPARE_RESERVATION_BYTES,
                                  budget_bytes=MODEL_CACHE_BUDGET_BYTES):
    if type(reservation_bytes) is not int or reservation_bytes < 0 or \
            type(budget_bytes) is not int or budget_bytes <= 0:
        raise ValueError('Cache budget preflight requires nonnegative integer byte limits')
    current = _existing_cache_bytes(cache_root)
    projected = current + reservation_bytes
    if projected > budget_bytes:
        raise ValueError(f'Existing model-quality cache plus resource-v2 reservation exceeds the 8 GiB budget: '
                         f'{current} + {reservation_bytes} > {budget_bytes}')
    return dict(current_cache_bytes=current, reservation_bytes=reservation_bytes,
                projected_cache_bytes=projected, cache_budget_bytes=budget_bytes)


def _metadata_artifact_bytes(root):
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file() and not set(path.relative_to(root).parts).intersection(('packets', 'contexts')):
                total += path.stat().st_size
        except OSError as exc:
            raise ValueError(f'Cannot account for capture metadata artifact {path}: {exc}') from exc
    return total


def _check_metadata_artifact_budget(root, additional_bytes=0):
    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ValueError('Metadata artifact reservation must be a nonnegative integer')
    current = _metadata_artifact_bytes(root)
    projected = current + additional_bytes
    if projected > METADATA_ARTIFACT_BUDGET_BYTES:
        raise ValueError(f'Capture metadata/trace/pose/log allowance exceeded: '
                         f'{projected} > {METADATA_ARTIFACT_BUDGET_BYTES} bytes')
    return projected


def _verify_capacity_spec(source_root=ROOT):
    spec_path = _safe_relative_path(source_root, CAPACITY_SPEC_PATH)
    if not spec_path.is_file() or digest(spec_path) != CAPACITY_SPEC_SHA256:
        raise ValueError('Frozen resource-v2 capacity/comparator specification changed')
    return dict(path=CAPACITY_SPEC_PATH, sha256=CAPACITY_SPEC_SHA256)


def _descriptor_details(name, descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {'shape', 'dtype', 'nbytes'}:
        raise ValueError(f'{name} array descriptor schema is incomplete or unknown')
    shape = descriptor.get('shape')
    dtype_name = descriptor.get('dtype')
    nbytes = descriptor.get('nbytes')
    if not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape):
        raise ValueError(f'{name} array descriptor has an invalid shape')
    if not isinstance(dtype_name, str) or not dtype_name:
        raise ValueError(f'{name} array descriptor has an invalid dtype')
    normalized_dtype = dtype_name.rsplit('.', 1)[-1] if dtype_name.startswith('torch.') else dtype_name
    try:
        dtype = np.dtype(normalized_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} array descriptor has an unsupported dtype {dtype_name!r}') from exc
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError(f'{name} array descriptor has an invalid byte count')
    expected = math.prod(shape) * dtype.itemsize
    if nbytes != expected:
        raise ValueError(f'{name} array descriptor byte count disagrees with shape/dtype')
    return tuple(shape), dtype, nbytes


def _require_descriptor(descriptors, name, shape, kinds, max_itemsize):
    if name not in descriptors:
        raise ValueError(f'{name} array descriptor is missing')
    actual_shape, dtype, nbytes = _descriptor_details(name, descriptors[name])
    if callable(shape):
        if not shape(actual_shape):
            raise ValueError(f'{name} array descriptor exceeds the frozen shape/count bound')
    elif actual_shape != tuple(shape):
        raise ValueError(f'{name} array descriptor has an unexpected shape')
    if dtype.kind not in kinds or dtype.itemsize > max_itemsize:
        raise ValueError(f'{name} array descriptor has an unsupported or oversized dtype')
    return actual_shape, dtype, nbytes


RETAINED_ARRAY_NAMES = frozenset({
    'query_rgb_crop', 'template_rgb', 'rendered_depth_mm', 'rendered_mask',
    'observed_crop_mask', 'full_flow_crop_px', 'full_confidence',
    'full_obj_points_mm', 'full_target_crop_px', 'source_flat_indices',
    'source_crop_pixel_centers', 'sample_ids', 'sample_obj_points_mm',
    'sample_target_crop_px', 'sample_weights', 'crop_k', 'crop_from_orig',
    'native_k', 'seed_camera_from_object_m',
    'current_crop_camera_from_object_mm', 'initial_rvec', 'initial_tvec_mm',
})
RESULT_BASE_ARRAY_NAMES = frozenset({'ransac_inlier_ids', 'retained_inlier_ids'})
RESULT_REFINED_ARRAY_NAMES = frozenset({
    'refined_rvec', 'refined_tvec_mm', 'crop_camera_from_object_mm',
    'native_camera_from_object_m',
})


def _packet_descriptor_bytes(kind, metadata, descriptors):
    """Validate the frozen packet inventory and return raw bytes without copying arrays."""
    if not isinstance(descriptors, dict):
        raise ValueError('Chronological packet descriptors must be a mapping')
    if kind == 'retained':
        if set(descriptors) != RETAINED_ARRAY_NAMES:
            raise ValueError('Retained packet array inventory differs from the frozen resource-v2 schema')
        fixed = {
            'query_rgb_crop': ((1, 3, 280, 280), 'f', 4),
            'template_rgb': ((1, 3, 280, 280), 'f', 4),
            'rendered_depth_mm': ((1, 280, 280), 'f', 4),
            'rendered_mask': ((1, 280, 280), 'bu', 1),
            'observed_crop_mask': ((1, 280, 280), 'biuf', 4),
            'full_flow_crop_px': ((280, 280, 2), 'f', 4),
            'full_confidence': ((280, 280), 'f', 4),
            'crop_k': ((3, 3), 'f', 8),
            'crop_from_orig': ((4, 4), 'f', 8),
            'native_k': ((3, 3), 'f', 8),
            'seed_camera_from_object_m': ((4, 4), 'f', 8),
            'current_crop_camera_from_object_mm': ((4, 4), 'f', 8),
            'initial_rvec': ((3, 1), 'f', 8),
            'initial_tvec_mm': ((3,), 'f', 8),
        }
        sizes = {}
        for name, (shape, kinds, max_itemsize) in fixed.items():
            _, _, sizes[name] = _require_descriptor(descriptors, name, shape, kinds, max_itemsize)
        full_obj_shape, _, sizes['full_obj_points_mm'] = _require_descriptor(
            descriptors, 'full_obj_points_mm', lambda value: len(value) == 2 and value[1] == 3
            and 0 <= value[0] <= MAX_FULL_CORRESPONDENCES, 'f', 8)
        full_count = full_obj_shape[0]
        full_rows = {
            'full_target_crop_px': ((full_count, 2), 'f', 8),
            'source_flat_indices': ((full_count,), 'iu', 8),
            'source_crop_pixel_centers': ((full_count, 2), 'f', 8),
        }
        for name, (shape, kinds, max_itemsize) in full_rows.items():
            _, _, sizes[name] = _require_descriptor(descriptors, name, shape, kinds, max_itemsize)
        sample_shape, _, sizes['sample_ids'] = _require_descriptor(
            descriptors, 'sample_ids', lambda value: len(value) == 1
            and 0 <= value[0] <= MAX_SAMPLED_CORRESPONDENCES, 'iu', 8)
        sample_count = sample_shape[0]
        sample_rows = {
            'sample_obj_points_mm': ((sample_count, 3), 'f', 8),
            'sample_target_crop_px': ((sample_count, 2), 'f', 8),
            'sample_weights': ((sample_count,), 'f', 4),
        }
        for name, (shape, kinds, max_itemsize) in sample_rows.items():
            _, _, sizes[name] = _require_descriptor(descriptors, name, shape, kinds, max_itemsize)
        total = sum(sizes.values())
        if total > MAX_RETAINED_PACKET_BYTES:
            raise ValueError(f'Retained packet raw bytes exceed frozen bound: {total} > {MAX_RETAINED_PACKET_BYTES}')
        return total

    if kind != 'result' or not isinstance(metadata, dict):
        raise ValueError('Chronological packet kind/metadata is outside the frozen schema')
    fit_state = metadata.get('fit_state')
    expected = set(RESULT_BASE_ARRAY_NAMES)
    if fit_state == 'refined':
        expected.update(RESULT_REFINED_ARRAY_NAMES)
    elif fit_state not in ('ransac_unavailable', 'insufficient_retained_correspondences'):
        raise ValueError('Result packet has an unknown solver terminal state')
    if set(descriptors) != expected:
        raise ValueError('Result packet array inventory differs from the frozen solver-state schema')
    sizes = {}
    id_shapes = []
    for name in sorted(RESULT_BASE_ARRAY_NAMES):
        shape, _, sizes[name] = _require_descriptor(descriptors, name,
            lambda value: len(value) == 1 and 0 <= value[0] <= MAX_SAMPLED_CORRESPONDENCES, 'iu', 8)
        id_shapes.append(shape)
    if id_shapes[0] != id_shapes[1]:
        raise ValueError('Result inlier-ID arrays have different counts')
    if fit_state == 'refined':
        refined = {
            'refined_rvec': ((3,), 'f', 8),
            'refined_tvec_mm': ((3,), 'f', 8),
            'crop_camera_from_object_mm': ((4, 4), 'f', 8),
            'native_camera_from_object_m': ((4, 4), 'f', 8),
        }
        for name, (shape, kinds, max_itemsize) in refined.items():
            _, _, sizes[name] = _require_descriptor(descriptors, name, shape, kinds, max_itemsize)
    total = sum(sizes.values())
    if total > MAX_RESULT_PACKET_BYTES:
        raise ValueError(f'Result packet raw bytes exceed frozen bound: {total} > {MAX_RESULT_PACKET_BYTES}')
    return total


def _native_context_descriptor_bytes(rgb_shape, rgb_dtype, mask_shape, mask_dtype,
                                     intrinsics_shape, intrinsics_dtype):
    rgb_shape, mask_shape, intrinsics_shape = tuple(rgb_shape), tuple(mask_shape), tuple(intrinsics_shape)
    rgb_dtype, mask_dtype, intrinsics_dtype = map(np.dtype, (rgb_dtype, mask_dtype, intrinsics_dtype))
    if rgb_dtype != np.dtype(np.uint8) or len(rgb_shape) != 3 or rgb_shape[2] != 3 or \
            min(rgb_shape[:2]) <= 0 or max(rgb_shape[:2]) > 1280 or min(rgb_shape[:2]) > 1024:
        raise ValueError('Native source RGB exceeds frozen uint8 1280x1024 RGB dimensions')
    if mask_dtype != np.dtype(np.uint8) or mask_shape != rgb_shape[:2]:
        raise ValueError('Native observed mask exceeds frozen uint8 source dimensions')
    if intrinsics_shape != (3, 3) or intrinsics_dtype.kind != 'f' or intrinsics_dtype.itemsize > 8:
        raise ValueError('Native camera matrix exceeds the frozen float64 3x3 descriptor')
    total = int(math.prod(rgb_shape) * rgb_dtype.itemsize + math.prod(mask_shape) * mask_dtype.itemsize
                + math.prod(intrinsics_shape) * intrinsics_dtype.itemsize)
    if total > MAX_NATIVE_CONTEXT_BYTES:
        raise ValueError(f'Native image/mask context exceeds frozen raw-byte bound: {total}')
    return total


def _native_context_bytes(rgb, observed_mask, intrinsics):
    rgb, observed_mask, intrinsics = map(np.asarray, (rgb, observed_mask, intrinsics))
    return _native_context_descriptor_bytes(rgb.shape, rgb.dtype, observed_mask.shape,
        observed_mask.dtype, intrinsics.shape, intrinsics.dtype)


def _working_copy_peak_bytes(packet_bytes, context_bytes=0):
    if type(packet_bytes) is not int or packet_bytes < 0 or \
            type(context_bytes) is not int or context_bytes < 0:
        raise ValueError('Working-copy byte accounting requires nonnegative integer sizes')
    # The producer's private event copy and the observer's serialization copy
    # coexist briefly. Context copies are made and released before packet copy.
    return 2 * packet_bytes + 2 * context_bytes


def _trace_records(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b'\n'):
        raise ValueError('Closed trace has an unterminated final record')
    try:
        return [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Closed trace contains an invalid record: {exc}') from exc


def _sha256_map(source_files, source_root):
    """Hash every pinned V2 source path, including unchanged dependencies."""
    result = {}
    for relative in sorted(source_files):
        path = _safe_relative_path(source_root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = digest(path)
    return result


def _verify_full_current_source_map(source_root, expected_source_hashes):
    if not isinstance(expected_source_hashes, dict) or not expected_source_hashes:
        raise ValueError('The full reviewed current source map is missing')
    current = _sha256_map(expected_source_hashes, source_root)
    if current != expected_source_hashes:
        changed = next((relative for relative in expected_source_hashes
                        if current.get(relative) != expected_source_hashes.get(relative)), '<unknown>')
        raise ValueError(f'Full reviewed source freeze changed at {changed}')
    return current


def _verify_manifest_source_freeze(bindings, source_root=None):
    source_root = Path(source_root or bindings.get('source_root', ROOT)).resolve()
    _verify_full_current_source_map(source_root, bindings.get('current_source_hashes'))
    for relative, expected in bindings.get('frozen_files', {}).items():
        path = _safe_relative_path(source_root, relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Frozen V2 input/checkpoint/mask/smoke binding changed: {relative}')
    record_rel = bindings.get('legacy_review_record_path')
    record_path = _safe_relative_path(source_root, record_rel)
    if not record_path.is_file() or digest(record_path) != bindings.get('legacy_review_record_sha256'):
        raise ValueError('Sol compatibility review record bytes changed after preparation')
    v2_root = Path(bindings.get('v2_root', '')).resolve()
    for name, key in (('experiment.json', 'v2_experiment_sha256'),
                      ('stage-evidence.json', 'v2_stage_evidence_sha256'),
                      ('report.json', 'v2_report_sha256')):
        path = v2_root / name
        if not path.is_file() or digest(path) != bindings.get(key):
            raise ValueError(f'Closed V2 {name} bytes changed after preparation')
    capture_module = source_root / 'bench/quality_mug_chronological_capture.py'
    if not capture_module.is_file() or digest(capture_module) != bindings.get('new_capture_module_sha256'):
        raise ValueError('Chronological capture module source hash changed after preparation')
    capacity_spec = bindings.get('capacity_spec')
    if capacity_spec is not None and capacity_spec != _verify_capacity_spec(source_root):
        raise ValueError('Resource-v2 capacity/comparator specification binding changed after preparation')
    return True


def _classified_source_delta(source_files, current_source_hashes):
    delta = {}
    for relative in INSTRUMENTED_SOURCE_PATHS:
        delta[relative] = dict(v2_sha256=source_files.get(relative),
            current_sha256=current_source_hashes.get(relative),
            classification=INSTRUMENTATION_DELTA_CLASSIFICATION)
    delta[V2_AUXILIARY_SOURCE_PATH] = dict(
        v2_sha256=source_files.get(V2_AUXILIARY_SOURCE_PATH),
        current_sha256=current_source_hashes.get(V2_AUXILIARY_SOURCE_PATH),
        classification=AUXILIARY_DELTA_CLASSIFICATION)
    return delta


def _review_record_path(path, source_root):
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(source_root) / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(Path(source_root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError('Compatibility review record must be inside the pinned source root') from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate, relative


def _validate_compatibility_review_record(record, record_path, v2_root, experiment,
        evidence_path, report_path, source_files, current_source_hashes):
    """Validate the independently accepted exact legacy/source compatibility record."""
    expected_keys = {
        'schema_version', 'review_id', 'reviewer', 'decision', 'scope',
        'runtime_path_equivalence', 'v2_bindings', 'captured_source_hashes',
        'current_source_hashes', 'instrumentation_source_delta',
        'auxiliary_source_delta', 'approved_delta_sha256', 'legacy_segmentation',
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError('Sol compatibility review record has an unknown or incomplete schema')
    legacy = record.get('legacy_segmentation')
    if not isinstance(legacy, dict) or set(legacy) != LEGACY_REVIEW_KEYS:
        raise ValueError('Sol compatibility review record has an unknown or incomplete legacy mask schema')
    legacy_rows = legacy.get('mask_rows')
    if not isinstance(legacy_rows, list) or [row.get('frame_id') for row in legacy_rows
            if isinstance(row, dict)] != list(range(SETUP_FRAME_ID, 1067)):
        raise ValueError('Sol compatibility review record must bind all ordered 241 native mask rows')
    if any(not isinstance(row, dict) or set(row) != LEGACY_MASK_ROW_KEYS or
            not isinstance(row.get('path'), str) or not row['path'] or
            not isinstance(row.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) or
            type(row.get('width')) is not int or row['width'] <= 0 or
            type(row.get('height')) is not int or row['height'] <= 0 or
            row.get('state') != 'available' for row in legacy_rows):
        raise ValueError('Sol compatibility review record contains a malformed native mask row')
    for key in ('manifest_path',):
        if not isinstance(legacy.get(key), str) or not legacy[key]:
            raise ValueError('Sol compatibility review record has a malformed legacy manifest path')
    for key in ('manifest_sha256', 'input_manifest_sha256', 'provenance_sha256'):
        if not isinstance(legacy.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', legacy[key]):
            raise ValueError(f'Sol compatibility review record has a malformed {key}')
    if legacy.get('producer_path') != LEGACY_SAM2_PRODUCER_PATH or \
            legacy.get('producer_sha256') != LEGACY_SAM2_PRODUCER_SHA256:
        raise ValueError('Sol compatibility review record does not bind the exact legacy SAM producer')
    if legacy.get('missing_flags') != list(LEGACY_MISSING_MODE_FLAGS):
        raise ValueError('Sol compatibility review record has an invalid legacy missing-flags list')
    if record.get('schema_version') != 1 or not isinstance(record.get('review_id'), str) or not record['review_id']:
        raise ValueError('Compatibility review record lacks its stable review identity')
    if record.get('reviewer') != 'Sol' or record.get('decision') != 'accepted' or \
            record.get('scope') != COMPATIBILITY_REVIEW_SCOPE or record.get('runtime_path_equivalence') is not True:
        raise ValueError('Compatibility record does not accept the exact Sol-reviewed runtime path and legacy-mask scope')
    v2_root = Path(v2_root).resolve()
    expected_v2 = dict(root=str(v2_root),
        experiment_sha256=digest(v2_root / 'experiment.json'),
        stage_evidence_sha256=digest(evidence_path), report_sha256=digest(report_path))
    if record.get('v2_bindings') != expected_v2:
        raise ValueError('Compatibility review record is bound to different V2 experiment/evidence/report bytes')
    if record.get('captured_source_hashes') != source_files:
        raise ValueError('Compatibility review record does not preserve the full V2 source map')
    if record.get('current_source_hashes') != current_source_hashes:
        raise ValueError('Compatibility review record current source map differs from exact current bytes')

    changed = {relative for relative, old in source_files.items()
               if current_source_hashes.get(relative) != old}
    expected_changed = set(INSTRUMENTED_SOURCE_PATHS) | {V2_AUXILIARY_SOURCE_PATH}
    if changed != expected_changed:
        raise ValueError('Current V2 source delta contains a missing or unreviewed path')
    if source_files.get(V2_AUXILIARY_SOURCE_PATH) != V2_AUXILIARY_SOURCE_BASELINE or \
            current_source_hashes.get(V2_AUXILIARY_SOURCE_PATH) != CURRENT_AUXILIARY_SOURCE:
        raise ValueError('Auxiliary annotation source delta differs from the exact reviewed old/current hashes')
    for relative, baseline in V2_INSTRUMENTATION_BASELINES.items():
        if source_files.get(relative) != baseline:
            raise ValueError(f'Unexpected frozen V2 instrumentation baseline for {relative}')
        current = current_source_hashes.get(relative)
        if not isinstance(current, str) or not re.fullmatch(r'[0-9a-f]{64}', current):
            raise ValueError(f'Current instrumentation source hash is malformed for {relative}')
    classified = _classified_source_delta(source_files, current_source_hashes)
    expected_instrumentation = {relative: classified[relative] for relative in INSTRUMENTED_SOURCE_PATHS}
    expected_auxiliary = {V2_AUXILIARY_SOURCE_PATH: classified[V2_AUXILIARY_SOURCE_PATH]}
    if record.get('instrumentation_source_delta') != expected_instrumentation:
        raise ValueError('Reviewed observer source delta classification differs from the exact two-file delta')
    if record.get('auxiliary_source_delta') != expected_auxiliary:
        raise ValueError('Reviewed auxiliary annotation delta classification differs from its exact hash pair')
    if record.get('approved_delta_sha256') != _json_sha(classified):
        raise ValueError('Canonical classified V2 source delta hash does not match the Sol review record')
    return dict(path=str(record_path), sha256=digest(record_path), record=record,
                classified_delta=classified)


def _validate_legacy_segmentation_record(record, segmentation, segmentation_path, bundle,
        masks_root, v2_experiment, source_root):
    """Require exact producer/input/manifest/native-mask bindings for omitted legacy flags."""
    review = record.get('legacy_segmentation') if isinstance(record, dict) else None
    if not isinstance(review, dict) or set(review) != LEGACY_REVIEW_KEYS:
        raise ValueError('Compatibility record lacks the exact legacy segmentation provenance group')
    expected_flags = {'automatic': True, 'diagnostic_control': False, 'stress_test': None}
    missing = [key for key in LEGACY_MISSING_MODE_FLAGS if key not in segmentation]
    for key, expected in expected_flags.items():
        if key in segmentation and segmentation[key] != expected:
            raise ValueError(f'Legacy segmentation has contradictory {key} mode metadata')
    if missing != list(LEGACY_MISSING_MODE_FLAGS) or review.get('missing_flags') != list(LEGACY_MISSING_MODE_FLAGS):
        raise ValueError('Legacy flag exception applies only when all three exact mode fields are absent')

    source_root = Path(source_root).resolve()
    segmentation_path = Path(segmentation_path).resolve()
    manifest_relative = segmentation_path.relative_to(source_root).as_posix()
    input_path = Path(bundle).resolve() / 'input.json'
    provenance = segmentation.get('provenance')
    expected_provenance = v2_experiment.get('segmentation_provenance')
    if not isinstance(provenance, dict) or provenance != expected_provenance:
        raise ValueError('Legacy segmentation provenance differs from the V2 frozen segmentation record')
    input_sha = digest(input_path)
    if provenance.get('input_manifest_sha256') != input_sha:
        raise ValueError('Legacy segmentation producer input manifest differs from the original mug bundle')
    producer_path = _safe_relative_path(source_root, LEGACY_SAM2_PRODUCER_PATH)
    if not producer_path.is_file() or digest(producer_path) != LEGACY_SAM2_PRODUCER_SHA256:
        raise ValueError('Exact legacy SAM producer source bytes are unavailable or changed')
    if provenance.get('adapter_sha256', {}).get('quality_sam2') != LEGACY_SAM2_PRODUCER_SHA256:
        raise ValueError('Legacy manifest provenance does not bind the exact SAM producer')

    from .vision import cv2
    rows = segmentation.get('frames')
    mask_hashes = v2_experiment.get('mask_sha256_by_frame', {})
    if not isinstance(rows, list) or [row.get('frameId') for row in rows] != list(range(SETUP_FRAME_ID, 1067)):
        raise ValueError('Legacy mask chronology differs from the exact ordered 241-frame V2 input window')
    native_rows = []
    for row in rows:
        frame_id = int(row['frameId'])
        relative_mask_path = row.get('path')
        mask_path = _safe_relative_path(masks_root, relative_mask_path)
        sha = digest(mask_path) if mask_path.is_file() else None
        if sha != mask_hashes.get(str(frame_id)):
            raise ValueError(f'Legacy observed mask hash differs from V2 at frame {frame_id}')
        if row.get('mask_state') != 'available' or row.get('failure_reason') is not None or \
                row.get('automatic_detection_required') is not False:
            raise ValueError(f'Legacy row contradicts no-fallback available-mask evidence at frame {frame_id}')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.ndim != 2 or not np.any(mask):
            raise ValueError(f'Legacy mask is missing, malformed, or blank at frame {frame_id}')
        native_rows.append(dict(frame_id=frame_id, path=relative_mask_path, sha256=sha,
            width=int(mask.shape[1]), height=int(mask.shape[0]), state=row['mask_state']))
    if review.get('manifest_path') != manifest_relative or \
            review.get('manifest_sha256') != digest(segmentation_path) or \
            review.get('input_manifest_sha256') != input_sha or \
            review.get('producer_path') != LEGACY_SAM2_PRODUCER_PATH or \
            review.get('producer_sha256') != LEGACY_SAM2_PRODUCER_SHA256 or \
            review.get('provenance_sha256') != _json_sha(provenance) or \
            review.get('mask_rows') != native_rows:
        raise ValueError('Legacy segmentation review record does not bind exact provenance and all native mask bytes')
    return dict(manifest_path=manifest_relative, manifest_sha256=digest(segmentation_path),
        input_manifest_sha256=input_sha, producer_path=LEGACY_SAM2_PRODUCER_PATH,
        producer_sha256=LEGACY_SAM2_PRODUCER_SHA256, provenance_sha256=_json_sha(provenance),
        missing_flags=list(missing), mask_rows=native_rows,
        interpretation='No effective forced blackout or same-frame CNOS fallback is present in this exact window; original CLI arguments remain unknown and no mask-accuracy claim is made.')


class TraceOrdinalReader:
    """Read-only cursor over the already-written inference trace JSONL."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.records = []

    def sync(self):
        if not self.path.is_file():
            return []
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            raw = stream.read()
        if not raw:
            return []
        final_newline = raw.rfind(b'\n')
        if final_newline < 0:
            return []
        complete = raw[:final_newline + 1]
        rows = [json.loads(line) for line in complete.decode('utf-8').splitlines()]
        first = len(self.records)
        self.records.extend(rows)
        self.offset += len(complete)
        return [(first + index, row) for index, row in enumerate(rows)]


def _validate_v2_terminal(v2_root, source_root=ROOT, compatibility_record_path=None):
    """Validate the immutable V2 lifecycle without using its quality labels."""
    root = Path(v2_root).resolve()
    experiment = _read_json(root / 'experiment.json')
    evidence = _read_json(root / 'stage-evidence.json')
    report = _read_json(root / 'report.json')
    if experiment.get('object') != 'mug' or experiment.get('setup_frame_id') != SETUP_FRAME_ID:
        raise ValueError('V2 mug snapshot does not bind setup frame 826')
    if experiment.get('expected_frame_ids') != list(range(827, 1067)):
        raise ValueError('V2 mug snapshot does not bind the exact original 240-frame window')
    if experiment.get('run_mode') != 'complete' or experiment.get('reference_or_annotation_inputs') is not False:
        raise ValueError('V2 snapshot is not the frozen complete-mode, reference-free run')
    if evidence.get('status') != 'complete' or evidence.get('attempted_stage_ids') != list(V2_STAGE_IDS):
        raise ValueError('V2 six-stage lifecycle is not terminal and complete')
    stages = evidence.get('stages')
    if not isinstance(stages, list) or [row.get('stage_id') for row in stages if isinstance(row, dict)] != list(V2_STAGE_IDS):
        raise ValueError('V2 terminal stages do not match the exact six chronological prefixes')
    if report.get('full_comparison_complete') is not True:
        raise ValueError('V2 evaluator did not close the complete comparison')
    if report.get('frozen_snapshot_intact') is not True:
        raise ValueError('V2 report says its immutable snapshot changed')
    lifecycle = report.get('terminal_stage_evidence')
    if not isinstance(lifecycle, dict) or lifecycle.get('passed') is not True:
        raise ValueError('V2 report did not accept terminal stage bindings')
    prefixes = report.get('prefix_checks')
    if not isinstance(prefixes, dict) or not prefixes or any(
            not isinstance(row, dict) or row.get('passed') is not True for row in prefixes.values()):
        raise ValueError('V2 chronological prefixes did not all pass')
    if not isinstance(report.get('paired_source_settings_checks'), dict) or report['paired_source_settings_checks'].get('passed') is not True:
        raise ValueError('V2 paired source/settings checks did not pass')
    source_files = experiment.get('source_files')
    frozen_files = experiment.get('frozen_files')
    provenance = experiment.get('inference_provenance')
    if not isinstance(source_files, dict) or not isinstance(frozen_files, dict) or not isinstance(provenance, dict):
        raise ValueError('V2 snapshot lacks source, immutable-file, or inference bindings')
    if experiment.get('required_settings') != REQUIRED_V4_SETTINGS:
        raise ValueError('V2 frozen settings do not match the chronological control')
    for relative, artifact in experiment.get('source_snapshot_files', {}).items():
        if not isinstance(artifact, dict) or artifact.get('sha256') != source_files.get(relative):
            raise ValueError(f'V2 content-addressed source binding is malformed: {relative}')
        artifact_path = root / artifact.get('artifact_path', '')
        if not artifact_path.is_file() or digest(artifact_path) != artifact['sha256']:
            raise ValueError(f'V2 content-addressed source bytes are missing or changed: {relative}')
    current_source_hashes = _sha256_map(source_files, source_root)
    review_path_info = _review_record_path(compatibility_record_path, source_root)
    if review_path_info is None:
        raise ValueError('A Sol-accepted exact source/legacy-mask compatibility record is required')
    review_path, review_relative = review_path_info
    compatibility_record = _read_json(review_path)
    reviewed = _validate_compatibility_review_record(compatibility_record, review_path,
        root, experiment, root / 'stage-evidence.json', root / 'report.json',
        source_files, current_source_hashes)
    reviewed['relative_path'] = review_relative
    actual_source_delta = {relative: reviewed['classified_delta'][relative]
                           for relative in INSTRUMENTED_SOURCE_PATHS}
    auxiliary_source_delta = {V2_AUXILIARY_SOURCE_PATH:
        reviewed['classified_delta'][V2_AUXILIARY_SOURCE_PATH]}
    for relative, expected in frozen_files.items():
        path = _safe_relative_path(source_root, relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'V2 immutable input/checkpoint/mask/smoke binding changed: {relative}')
    expected_adapters = provenance.get('adapter_sha256', {})
    if not isinstance(expected_adapters, dict):
        raise ValueError('V2 inference provenance lacks adapter pins')
    input_name = next((name for name in frozen_files if name.endswith('/input.json')), None)
    if input_name is None:
        raise ValueError('V2 immutable files do not identify the input manifest')
    bundle_root = _safe_relative_path(source_root, input_name[:-len('input.json')].rstrip('/'))
    current_inference = inference_provenance(bundle_root)
    # The per-run bank pin is added outside inference_provenance by the runner.
    bank_key = 'foundpose_bank_sha256'
    expected_bank = provenance.get(bank_key)
    if expected_bank is None:
        raise ValueError('V2 inference provenance is missing its FoundPose bank hash')
    bank_name = next((name for name in frozen_files if name.endswith('mug-foundpose.pt')), None)
    if bank_name is None or frozen_files[bank_name] != expected_bank:
        raise ValueError('V2 FoundPose bank provenance does not match its immutable file binding')
    current_inference[bank_key] = expected_bank
    current_adapters = current_inference['adapter_sha256']
    for name, expected in expected_adapters.items():
        if name in ('quality_gotrack', 'quality_runner'):
            relative = f'bench/{name}.py'
            if current_adapters.get(name) != actual_source_delta[relative]['current_sha256']:
                raise ValueError(f'Current observer source hash does not bind inference adapter {name}')
        elif current_adapters.get(name) != expected:
            raise ValueError(f'V2 inference adapter changed outside the instrumentation allowlist: {name}')
    for key in ('input_manifest_sha256', 'source_revisions', 'submodule_revisions', 'checkpoint_sha256'):
        if current_inference.get(key) != provenance.get(key):
            raise ValueError(f'V2 inference pin changed: {key}')
    if current_inference.get('checkpoint_sha256') != provenance.get('checkpoint_sha256'):
        raise ValueError('V2 checkpoint binding changed')
    baseline_runtime = None
    for stage in stages:
        if stage.get('status') != 'succeeded' or stage.get('exit_code') != 0 or stage.get('terminal_exit_recorded') is not True:
            raise ValueError(f'{stage.get("stage_id")} lacks terminal exit-zero evidence')
        stage_id = stage['stage_id']
        branch, length_text = stage_id.rsplit('-', 1)
        length = int(length_text)
        expected_ids = list(range(827, 827 + length))
        if stage.get('requested_frame_ids') != expected_ids or stage.get('result_frame_ids') != expected_ids:
            raise ValueError(f'{stage_id} terminal evidence does not bind the exact chronological prefix IDs')
        if stage.get('expected_complete') is not (length == 240):
            raise ValueError(f'{stage_id} terminal evidence has the wrong completion expectation')
        output_path = root / stage.get('output', '')
        trace_path = root / stage.get('trace', '')
        if stage.get('output') != f'{stage_id}.json' or not output_path.is_file():
            raise ValueError(f'{stage_id} output path is not the immutable expected path')
        if stage.get('trace') != f'{stage_id}.trace.jsonl' or not trace_path.is_file():
            raise ValueError(f'{stage_id} trace path is not the immutable expected path')
        if digest(output_path) != stage.get('output_sha256') or digest(trace_path) != stage.get('trace_sha256'):
            raise ValueError(f'{stage_id} immutable output or trace hash differs from terminal evidence')
        trace_rows = _trace_records(trace_path)
        if type(stage.get('trace_record_count')) is not int or len(trace_rows) != stage['trace_record_count'] or not trace_rows:
            raise ValueError(f'{stage_id} terminal trace row count is not completely bound')
        if any(not isinstance(item, dict) or not isinstance(item.get('stage'), str) or
                not isinstance(item.get('arrays'), dict) or not item['arrays'] for item in trace_rows):
            raise ValueError(f'{stage_id} trace contains a malformed inference record')
        rows = _read_json(output_path)
        if rows.get('object') != 'mug' or rows.get('mode') != 'complete' or rows.get('frame_ids') != expected_ids:
            raise ValueError(f'{stage_id} output IDs/mode do not match the frozen prefix')
        if 'stress_test' not in rows or rows['stress_test'] is not None:
            raise ValueError(f'{stage_id} lacks explicit no-stress output evidence or records forced occlusion')
        if rows.get('provenance') != provenance or rows.get('complete') is not (length == 240):
            raise ValueError(f'{stage_id} output provenance/completion differs from V2 snapshot')
        if rows.get('tracking_settings', {}).get('pnp_use_extrinsic_guess') is not BRANCH_GUESS[branch]:
            raise ValueError(f'{stage_id} output has the wrong immutable branch setting')
        if any(type(rows.get('tracking_settings', {}).get(key)) is not bool or
               rows['tracking_settings'][key] is not value for key, value in REQUIRED_V4_SETTINGS.items()):
            raise ValueError(f'{stage_id} output changed a fixed V2 setting')
        if [item.get('frameId') for item in rows.get('frames', [])] != expected_ids:
            raise ValueError(f'{stage_id} output omits one or more requested frame rows')
        expected_masks = experiment.get('mask_sha256_by_frame', {})
        if any(item.get('mask_sha256') != expected_masks.get(str(item.get('frameId')))
               for item in rows['frames']):
            raise ValueError(f'{stage_id} output has an unfrozen observed-mask binding')
        if stage.get('runtime') != rows.get('runtime') or stage.get('tracking_settings') != rows.get('tracking_settings'):
            raise ValueError(f'{stage_id} terminal runtime/settings differ from saved output')
        if baseline_runtime is None:
            baseline_runtime = rows.get('runtime')
        elif rows.get('runtime') != baseline_runtime:
            raise ValueError(f'{stage_id} runtime differs between V2 chronological stages')
        if stage.get('source_files') != source_files or stage.get('frozen_files') != frozen_files:
            raise ValueError(f'{stage_id} terminal source/input bindings differ from V2 snapshot')
        if stage.get('inference_provenance') != provenance:
            raise ValueError(f'{stage_id} terminal inference bindings differ from V2 snapshot')
    return dict(root=str(root), experiment=experiment, evidence=evidence, report=report,
                provenance=provenance, source_delta=actual_source_delta,
                auxiliary_source_delta=auxiliary_source_delta,
                current_source_sha256={name: current_source_hashes[name]
                                       for name in INSTRUMENTED_SOURCE_PATHS},
                current_source_hashes=current_source_hashes,
                compatibility_review=reviewed,
                source_root=str(Path(source_root).resolve()))


def _read_v2_trace(root, branch):
    path = Path(root) / f'{branch}-240.trace.jsonl'
    rows = _trace_records(path)
    if not rows:
        raise ValueError(f'{branch}-240 trace is empty or malformed')
    return rows


def _trace_frame_id(row):
    stage = row.get('stage', '')
    match = re.fullmatch(r'(?:refine/(\d+)/\d+|recover/(\d+))', stage)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def _v2_trace_prefix(rows, last_frame_id):
    last = -1
    for index, row in enumerate(rows):
        frame_id = _trace_frame_id(row)
        if frame_id is not None and frame_id <= last_frame_id:
            last = index
        elif frame_id is not None and frame_id > last_frame_id:
            break
    if last < 0:
        raise ValueError('V2 trace has no inference record through the requested prefix')
    return rows[:last + 1]


def _compare_trace_records(short, long):
    if len(short) != len(long):
        return dict(reason='different_record_count', short_count=len(short), long_count=len(long))
    for index, (left, right) in enumerate(zip(short, long)):
        if left != right:
            return dict(index=index, reason='trace_record_mismatch', short=left, long=right)
    return None


def prepare_capture_root(bundle, masks_root, v2_root, capture_root, source_root=ROOT,
                         compatibility_record_path=None):
    """Create a fresh, pinned two-branch 201-frame capture plan; run nothing."""
    source_root = Path(source_root).resolve()
    bundle = Path(bundle).resolve()
    masks_root = Path(masks_root).resolve()
    target = Path(capture_root).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError('Choose a fresh private capture root; existing trials are preserved')
    cache_preflight = _preflight_model_cache_budget()
    capacity_spec = _verify_capacity_spec(source_root)
    v2 = _validate_v2_terminal(v2_root, source_root, compatibility_record_path)
    input_manifest = _read_json(bundle / 'input.json')
    if input_manifest.get('object') != 'mug' or input_manifest.get('units') != 'metres':
        raise ValueError('Chronological capture requires the original metric mug bundle')
    if input_manifest.get('frame_ids') != list(range(827, 1067)) or input_manifest.get('setup_frame_id') != SETUP_FRAME_ID:
        raise ValueError('Chronological capture requires setup 826 followed by original IDs 827–1066')
    input_relative = (bundle / 'input.json').resolve().relative_to(source_root).as_posix()
    if v2['experiment'].get('frozen_files', {}).get(input_relative) != digest(bundle / 'input.json'):
        raise ValueError('Chronological bundle input manifest is not the V2 frozen input')
    segmentation_path = masks_root / 'results.json'
    segmentation = _read_json(segmentation_path)
    legacy_segmentation = _validate_legacy_segmentation_record(
        v2['compatibility_review']['record'], segmentation, segmentation_path,
        bundle, masks_root, v2['experiment'], source_root)
    mask_hashes = v2['experiment'].get('mask_sha256_by_frame', {})
    smoke_relative = next((name for name in v2['experiment']['frozen_files'] if '/smoke/' in name and name.endswith('.json')), None)
    if smoke_relative is None:
        raise ValueError('V2 snapshot does not bind its existing unlit/no-MSAA smoke record')
    smoke = _read_json(_safe_relative_path(source_root, smoke_relative))
    if smoke.get('smoke_passed') is not True or smoke.get('unlit_templates') is not True or smoke.get('disable_multisampling') is not True:
        raise ValueError('Existing renderer smoke does not match the frozen V2 settings')
    required = v2['experiment'].get('required_settings')
    if required != REQUIRED_V4_SETTINGS:
        raise ValueError('V2 frozen settings differ from the approved chronological control')

    current_provenance = inference_provenance(bundle)
    current_provenance['foundpose_bank_sha256'] = v2['provenance']['foundpose_bank_sha256']
    for key in ('input_manifest_sha256', 'source_revisions', 'submodule_revisions', 'checkpoint_sha256'):
        if current_provenance.get(key) != v2['provenance'].get(key):
            raise ValueError(f'Current chronological run does not preserve V2 binding {key}')
    for adapter, expected in v2['provenance']['adapter_sha256'].items():
        if adapter in ('quality_gotrack', 'quality_runner'):
            continue
        if current_provenance['adapter_sha256'].get(adapter) != expected:
            raise ValueError(f'Current chronological run changes pinned adapter {adapter}')
    source_manifest = {
        'v2_experiment_sha256': digest(Path(v2_root) / 'experiment.json'),
        'v2_stage_evidence_sha256': digest(Path(v2_root) / 'stage-evidence.json'),
        'v2_report_sha256': digest(Path(v2_root) / 'report.json'),
        'prior_source_files': v2['experiment']['source_files'],
        'captured_source_hashes': v2['experiment']['source_files'],
        'current_source_hashes': v2['current_source_hashes'],
        'instrumentation_only_source_delta': v2['source_delta'],
        'auxiliary_source_delta': v2['auxiliary_source_delta'],
        'approved_classified_delta_sha256': v2['compatibility_review']['record']['approved_delta_sha256'],
        'current_source_sha256': v2['current_source_sha256'],
        'new_capture_module_sha256': digest(source_root / 'bench/quality_mug_chronological_capture.py'),
        'current_inference_provenance': current_provenance,
        'prior_inference_provenance_sha256': _json_sha(v2['provenance']),
        'frozen_files': v2['experiment']['frozen_files'],
        'mask_sha256_by_frame': mask_hashes,
        'legacy_review_record_path': v2['compatibility_review']['relative_path'],
        'legacy_review_record_sha256': v2['compatibility_review']['sha256'],
        'legacy_segmentation_review': legacy_segmentation,
        'capacity_spec': capacity_spec,
        'cache_preflight': cache_preflight,
        'source_root': str(source_root),
        'bundle_root': str(bundle),
        'masks_root': str(masks_root),
        'v2_root': v2['root'],
        'unchanged_binding_policy': 'All V2 source hashes must match the independently reviewed full map except the exact two-file observer delta and the separately classified quality_annotations.py auxiliary delta; all frozen files, input/checkpoint/bank/upstream/submodule/smoke/native-mask hashes and fixed settings remain byte-identical.',
        'legacy_mode_limitations': 'The legacy SAM manifest omitted automatic/diagnostic_control/stress_test mode fields. Bound producer, provenance, input and all native masks plus closed V2 stress_test:null records support no effective forced blackout or same-frame CNOS fallback in this exact window; original SAM CLI arguments and mask accuracy remain unknown.',
    }
    _verify_manifest_source_freeze(source_manifest, source_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=False, exist_ok=False)
    manifest = dict(schema_version=CAPTURE_SCHEMA_VERSION, experiment=CAPTURE_EXPERIMENT, status='prepared',
        setup_frame_id=SETUP_FRAME_ID, frame_ids=list(RUN_FRAME_IDS), capture_frame_ids=list(CAPTURE_FRAME_IDS),
        capture_iteration=CAPTURE_ITERATION, branch_pnp_use_extrinsic_guess=dict(BRANCH_GUESS),
        settings={**REQUIRED_V4_SETTINGS, 'model_memory': False, 'appearance_check': False,
                  'mode': 'complete', 'limit': len(RUN_FRAME_IDS), 'recovery_top_k': 5, 'iterations': 5,
                  'crop_size': [280, 280], 'crop_rel_pad': .1, 'cropping_type': 'perspective_2d_box',
                  'ssaa_factor': 1.0},
        capture_budget_bytes=CAPTURE_BUDGET_BYTES, copied_array_bytes=0, capture_stopped=False,
        metadata_artifact_budget_bytes=METADATA_ARTIFACT_BUDGET_BYTES,
        working_copy_budget_bytes=WORKING_COPY_BUDGET_BYTES,
        planned_max_capture_bytes=MAX_PLANNED_CAPTURE_BYTES,
        planned_capture_headroom_bytes=MAX_PLANNED_CAPTURE_HEADROOM_BYTES,
        max_packet_pairs=MAX_PACKET_PAIRS, max_native_contexts=MAX_NATIVE_CONTEXTS,
        bindings=source_manifest,
        branches={branch: dict(status='prepared', frame_records=[], packets={}, output=None, trace=None,
                               exit_code=None) for branch in BRANCHES},
        reference_or_annotation_inputs=False,
        scope='Opt-in actual chronological retained-correspondence observer. No surface identity labels or accuracy claim.')
    _write_exclusive(target / 'capture.json', manifest, artifact_root=target)
    commands = {}
    for branch in BRANCHES:
        output = target / f'{branch}-201.json'
        trace_path = target / f'{branch}-201.trace.jsonl'
        command = [sys.executable, '-m', 'bench.quality_runner', 'pose',
            '--bundle', str(bundle), '--masks', str(masks_root), '--output', str(output),
            '--device', str(v2['experiment']['device']), '--mode', 'complete', '--limit', '201',
            '--unlit-templates', '--disable-multisampling', '--chronological-mug-capture-root', str(target),
            '--chronological-mug-capture-branch', branch]
        if branch == 'candidate':
            command.append('--pnp-no-extrinsic-guess')
        commands[branch] = dict(command=command, trace_environment={'VISUALIZEIT_QUALITY_TRACE': str(trace_path)},
            expected_ids=list(RUN_FRAME_IDS), expected_complete=False,
            required_after_exit=['python -m bench.quality_mug_chronological_capture terminal --root '
                + shlex.quote(str(target)) + ' --branch ' + branch + ' --exit-code 0'])
    _write_exclusive(target / 'commands.json', dict(run_order=list(BRANCHES), branches=commands,
        note='Execute control to terminal exit 0 and record it before starting candidate. Each process is a separate complete-mode chronological run.'),
        artifact_root=target)
    return manifest


class MugChronologicalCapture:
    """Bounded opt-in observer. Callback failures never enter the tracker path."""

    def __init__(self, root, branch, trace_path, source_root=ROOT, bundle=None, masks_root=None):
        if branch not in BRANCHES:
            raise ValueError('Capture branch must be control or candidate')
        self.root = Path(root).resolve()
        self.branch = branch
        self.trace_path = Path(trace_path).resolve()
        self.source_root = Path(source_root).resolve()
        self.bundle = Path(bundle).resolve() if bundle is not None else None
        self.masks_root = Path(masks_root).resolve() if masks_root is not None else None
        self.manifest_path = self.root / 'capture.json'
        self.manifest = _read_json(self.manifest_path)
        if self.manifest.get('schema_version') != CAPTURE_SCHEMA_VERSION or \
                self.manifest.get('experiment') != CAPTURE_EXPERIMENT:
            raise ValueError('Wrong resource-v2 chronological mug capture manifest')
        expected_trace = self.root / f'{branch}-201.trace.jsonl'
        if self.trace_path != expected_trace:
            raise ValueError('Trace path must be the fresh branch trace recorded by prepare')
        self.branch_row = self.manifest['branches'][branch]
        if self.branch_row.get('status') != 'prepared':
            raise ValueError(f'{branch} branch is not fresh and prepared')
        if branch == 'candidate' and self.manifest['branches']['control'].get('exit_code') != 0:
            raise ValueError('Candidate cannot start until control has terminal exit code 0')
        if (self.root / f'{branch}-201.json').exists() or expected_trace.exists():
            raise FileExistsError('Existing chronological branch data must be preserved')
        if self.bundle is not None or self.masks_root is not None:
            self._validate_runtime_bindings(self.bundle, self.masks_root)
        self.trace_reader = TraceOrdinalReader(self.trace_path)
        self.trace_reader.sync()
        self.frame = None
        self.entry_tracking = False
        self.invocation_ordinal = 0
        self.current_invocation = None
        self.current_row = None
        self.failure = None
        self.disabled = bool(self.manifest.get('capture_stopped'))
        self.expected_next_frame_id = SETUP_FRAME_ID
        self._set_branch_status('running')
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open('xb'):
            pass

    def _validate_runtime_bindings(self, bundle, masks_root):
        bindings = self.manifest.get('bindings', {})
        bundle = Path(bundle).resolve() if bundle is not None else None
        masks_root = Path(masks_root).resolve() if masks_root is not None else None
        if bundle is None or masks_root is None or str(bundle) != bindings.get('bundle_root') or str(masks_root) != bindings.get('masks_root'):
            raise ValueError('Runtime bundle/mask roots differ from the prepared immutable capture')
        _verify_manifest_source_freeze(bindings, self.source_root)
        for relative, expected in bindings.get('current_source_sha256', {}).items():
            path = _safe_relative_path(self.source_root, relative)
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f'Reviewed instrumentation source changed before branch: {relative}')
        current = inference_provenance(bundle)
        bank_name = next((name for name in bindings['frozen_files'] if name.endswith('mug-foundpose.pt')), None)
        if bank_name is None:
            raise ValueError('Frozen capture manifest is missing the FoundPose bank')
        current['foundpose_bank_sha256'] = digest(_safe_relative_path(self.source_root, bank_name))
        if current != bindings.get('current_inference_provenance'):
            raise ValueError('Current inference provenance differs from the prepared instrumentation pins')
        segmentation_path = masks_root / 'results.json'
        relative_manifest = segmentation_path.resolve().relative_to(self.source_root).as_posix()
        if digest(segmentation_path) != bindings['frozen_files'].get(relative_manifest):
            raise ValueError('Cached segmentation manifest changed after preflight')
        segmentation = _read_json(segmentation_path)
        review_path = _safe_relative_path(self.source_root, bindings.get('legacy_review_record_path'))
        review_record = _read_json(review_path)
        v2_experiment = _read_json(Path(bindings['v2_root']) / 'experiment.json')
        legacy_review = _validate_legacy_segmentation_record(review_record, segmentation,
            segmentation_path, bundle, masks_root, v2_experiment, self.source_root)
        if legacy_review != bindings.get('legacy_segmentation_review'):
            raise ValueError('Legacy segmentation review summary changed after preparation')
        for row in segmentation['frames']:
            path = _safe_relative_path(masks_root, row.get('path'))
            if digest(path) != bindings['mask_sha256_by_frame'].get(str(row['frameId'])):
                raise ValueError(f'Observed mask changed after preflight at frame {row["frameId"]}')

    def _set_branch_status(self, status):
        self.branch_row['status'] = status
        _atomic_json(self.manifest_path, self.manifest, artifact_root=self.root)

    def enter_refinement(self, frame_id):
        self.invocation_ordinal += 1
        phase = 'primary_tracking' if self.entry_tracking and self.invocation_ordinal == 1 else 'recovery'
        invocation = dict(invocation_ordinal=self.invocation_ordinal, frame_id=int(frame_id), phase=phase,
                          trace_ordinals=[], iteration_trace_ordinals={}, retained_iteration_seen=False,
                          result_iteration_seen=False)
        self.current_invocation = invocation
        if self.current_row is not None:
            self.current_row['invocations'].append(invocation)
        return dict(frame_id=int(frame_id), phase=phase, invocation_ordinal=self.invocation_ordinal)

    def clear_invocation(self):
        self.current_invocation = None

    def _sync_trace(self, expected_stage=None):
        rows = self.trace_reader.sync()
        if expected_stage is not None:
            matching = [(ordinal, row) for ordinal, row in rows if row.get('stage') == expected_stage]
            if not matching:
                raise ValueError(f'Expected just-appended trace stage {expected_stage!r} is absent')
            return matching[-1][0], rows
        return (rows[-1][0] if rows else None), rows

    def after_recovery(self, frame_id):
        try:
            ordinal, trace_rows = self._sync_trace(f'recover/{int(frame_id)}')
        except Exception:
            # Recovery may legitimately stop before emitting its feature trace.
            # Record that absence; never let the observer affect tracker control.
            ordinal, trace_rows = None, []
        if self.current_row is not None and self.current_row['frame_id'] == int(frame_id):
            self.current_row['recoveries'].append(dict(trace_ordinal=ordinal, frame_id=int(frame_id),
                trace_missing=ordinal is None,
                trace_records=[dict(trace_ordinal=index, stage=record.get('stage'))
                               for index, record in trace_rows]))
            try:
                self._persist_manifest()
            except Exception as exc:
                self._disable_capture(f'recovery ordinal persistence failed: {type(exc).__name__}: {exc}')
        return ordinal

    def should_capture(self, context, kind, metadata):
        if kind == 'trace':
            return True
        return (not self.disabled and isinstance(context, dict) and context.get('phase') == 'primary_tracking'
            and context.get('frame_id') in CAPTURE_FRAME_IDS and metadata.get('iteration') == CAPTURE_ITERATION
            and kind in ('retained', 'result'))

    def preflight(self, context, kind, metadata, arrays):
        if not self.should_capture(context, kind, metadata):
            return False
        # GoTrack provides descriptors here so no solver-owned array crosses
        # the callback boundary. The exact packet inventory/counts are checked
        # before its private event copy is made.
        try:
            packet_bytes = _packet_descriptor_bytes(kind, metadata, arrays)
        except (TypeError, ValueError) as exc:
            self._stop_capture(f'packet descriptor rejected before copy: {type(exc).__name__}: {exc}')
            return False
        return self.preflight_nbytes(context, kind, metadata, packet_bytes)

    def preflight_nbytes(self, context, kind, metadata, packet_bytes):
        if not self.should_capture(context, kind, metadata):
            return False
        if type(packet_bytes) is not int or packet_bytes < 0:
            self._stop_capture('packet byte preflight received an invalid raw byte count')
            return False
        max_packet_bytes = MAX_RETAINED_PACKET_BYTES if kind == 'retained' else MAX_RESULT_PACKET_BYTES
        if packet_bytes > max_packet_bytes:
            self._stop_capture(f'{kind} packet raw bytes exceed frozen resource-v2 bound')
            return False
        context_bytes = 0
        if self.current_row is None or self.current_row.get('context_key') is None:
            try:
                context_bytes = _native_context_bytes(self.frame.rgb, self._mask_value,
                    self.frame.intrinsics)
            except Exception as exc:
                self._stop_capture(f'native context descriptor rejected before copy: {type(exc).__name__}: {exc}')
                return False
            key = self._frame_context_key() if self.current_row is not None else None
            if key in self.manifest.get('contexts', {}):
                context_bytes = 0
        if _working_copy_peak_bytes(packet_bytes, context_bytes) > WORKING_COPY_BUDGET_BYTES:
            self._stop_capture('simultaneous private instrumentation copies exceed the 128 MiB working-copy budget')
            return False
        if self.manifest['copied_array_bytes'] + packet_bytes + context_bytes > CAPTURE_BUDGET_BYTES:
            self._stop_capture('320 MiB aggregate raw copied-array budget reached; prior packets preserved without thinning')
            return False
        return True

    @property
    def _mask_value(self):
        # Set by begin_frame without copying; copied only when the first retained pair is accepted.
        return self._observed_mask

    def begin_frame(self, frame, observed_mask, entry_tracking, mask_path=None, mask_sha256=None,
                    mask_state=None, failure_reason=None):
        if frame.frame_id != self.expected_next_frame_id:
            try:
                self._disable_capture(f'non-chronological frame entry: expected {self.expected_next_frame_id}, received {frame.frame_id}')
            except Exception:
                self.disabled = True
        self.expected_next_frame_id = int(frame.frame_id) + 1
        self._observed_mask = observed_mask
        self.frame = frame
        self.entry_tracking = bool(entry_tracking)
        self.invocation_ordinal = 0
        self.current_invocation = None
        self.current_row = None
        try:
            self.trace_reader.sync()
        except Exception as exc:
            self._disable_capture(f'pre-frame trace cursor failed: {type(exc).__name__}: {exc}')
        if frame.frame_id in CAPTURE_FRAME_IDS:
            mask_array = None if observed_mask is None else np.asarray(observed_mask)
            usable = (mask_state == 'available' and mask_array is not None and
                mask_array.shape == frame.rgb.shape[:2] and bool(np.any(mask_array)))
            self.current_row = dict(frame_id=int(frame.frame_id), entry_tracking=self.entry_tracking,
                mask_path=mask_path, mask_sha256=mask_sha256, mask_state=mask_state,
                mask_usable_at_entry=usable,
                entry_failure_reason=failure_reason, invocations=[], recoveries=[],
                retained_packet=None, result_packet=None, context_key=None,
                availability='pending', unavailable_reason=None, public_decision=None)

    def _frame_context_key(self):
        rgb = np.asarray(self.frame.rgb)
        mask = np.asarray(self._observed_mask)
        intrinsic = np.asarray(self.frame.intrinsics)
        descriptor = (str(self.frame.frame_id) + '|' + str(rgb.dtype) + '|' + str(rgb.shape)
            + '|' + str(mask.dtype) + '|' + str(mask.shape)).encode('ascii')
        return _sha_bytes(descriptor + rgb.tobytes() + mask.tobytes() + intrinsic.tobytes())

    def __call__(self, packet):
        try:
            kind = packet.get('kind')
            context = packet.get('context') or {}
            metadata = packet.get('metadata') or {}
            if kind == 'trace':
                ordinal, trace_rows = self._sync_trace(metadata.get('expected_stage'))
                if self.current_invocation is not None:
                    self.current_invocation['trace_ordinals'].append(ordinal)
                    self.current_invocation.setdefault('trace_records', []).extend(
                        dict(trace_ordinal=index, stage=row.get('stage')) for index, row in trace_rows)
                    iteration = metadata.get('iteration')
                    if iteration is not None:
                        self.current_invocation['iteration_trace_ordinals'][str(iteration)] = ordinal
                self._check_current_metadata_artifacts()
                return
            if not self.should_capture(context, kind, metadata):
                return
            arrays = packet.get('arrays', {})
            _packet_descriptor_bytes(kind, metadata, {
                name: dict(shape=[int(size) for size in _array_numpy(value).shape],
                           dtype=str(_array_numpy(value).dtype), nbytes=_array_nbytes(value))
                for name, value in arrays.items() if value is not None
            })
            if kind == 'retained':
                if self.current_row is None or self.current_row['frame_id'] != context['frame_id']:
                    return
                self._ensure_context()
                info = self._write_arrays(context['frame_id'], 'retained', arrays, metadata)
                info['event_metadata'] = dict(metadata)
                info['invocation'] = dict(context)
                if self.current_invocation is not None:
                    info['trace_ordinal'] = self.current_invocation['iteration_trace_ordinals'].get(str(metadata.get('iteration')))
                self.current_row['retained_packet'] = info
                self.current_row['context_key'] = self._context_key
                if self.current_invocation is not None:
                    self.current_invocation['retained_iteration_seen'] = True
                    self.current_invocation['retained_packet'] = info
                self._persist_manifest()
            elif kind == 'result':
                if self.current_row is None or self.current_row['frame_id'] != context['frame_id']:
                    return
                info = self._write_arrays(context['frame_id'], 'result', arrays, metadata)
                info['event_metadata'] = dict(metadata)
                info['invocation'] = dict(context)
                if self.current_invocation is not None:
                    info['trace_ordinal'] = self.current_invocation['iteration_trace_ordinals'].get(str(metadata.get('iteration')))
                self.current_row['result_packet'] = info
                self.current_row['context_key'] = self._context_key
                if self.current_invocation is not None:
                    self.current_invocation['result_iteration_seen'] = True
                    self.current_invocation['result_packet'] = info
                self._persist_manifest()
        except Exception as exc:
            self._disable_capture(f'capture I/O or binding failure: {type(exc).__name__}: {exc}')

    def _ensure_context(self):
        if self.current_row is None:
            raise ValueError('No selected frame context is active')
        rgb_view = np.asarray(self.frame.rgb)
        mask_view = np.asarray(self._observed_mask)
        intrinsic_view = np.asarray(self.frame.intrinsics)
        size = _native_context_bytes(rgb_view, mask_view, intrinsic_view)
        key = self._frame_context_key()
        relative = f'contexts/{key}.npz'
        path = self.root / relative
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                if not (np.array_equal(existing['native_rgb'], rgb_view) and
                        np.array_equal(existing['observed_native_mask'], mask_view) and
                        np.array_equal(existing['native_k'], intrinsic_view)):
                    raise ValueError('Content-addressed native context collision')
            self._context_key = key
            self.current_row['context_key'] = key
            self.manifest.setdefault('contexts', {})[key] = dict(path=relative, sha256=digest(path),
                array_bytes=int(rgb_view.nbytes + mask_view.nbytes + intrinsic_view.nbytes),
                frame_id=int(self.frame.frame_id), rgb_sha256=_sha_bytes(rgb_view.tobytes()),
                observed_mask_sha256=_sha_bytes(mask_view.tobytes()))
            self._persist_manifest()
            return
        if len(self.manifest.get('contexts', {})) >= MAX_NATIVE_CONTEXTS:
            self._stop_capture('native context count exceeds the frozen twelve-context cohort')
            raise RuntimeError('Native context count exhausted')
        if self.manifest['copied_array_bytes'] + size > CAPTURE_BUDGET_BYTES:
            self._stop_capture('320 MiB aggregate raw copied-array budget reached before native context; prior packets preserved without thinning')
            raise RuntimeError('Capture budget exhausted before native context')
        if _working_copy_peak_bytes(0, size) > WORKING_COPY_BUDGET_BYTES:
            self._stop_capture('native context copies exceed the 128 MiB working-copy budget')
            raise RuntimeError('Working-copy budget exhausted before native context')
        rgb = np.array(rgb_view, copy=True, order='C')
        mask = np.array(mask_view, copy=True, order='C')
        intrinsic = np.array(intrinsic_view, copy=True, order='C')
        arrays = dict(native_rgb=rgb, observed_native_mask=mask, native_k=intrinsic)
        copied_size = sum(value.nbytes for value in arrays.values())
        if copied_size != size:
            raise ValueError('Native context raw bytes changed after preflight')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.part')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        self.manifest['copied_array_bytes'] += size
        self._context_key = key
        self.manifest.setdefault('contexts', {})[key] = dict(path=relative, sha256=digest(path), array_bytes=size,
            frame_id=int(self.frame.frame_id), rgb_sha256=_sha_bytes(rgb.tobytes()),
            observed_mask_sha256=_sha_bytes(mask.tobytes()))
        self.current_row['context_key'] = key
        self._persist_manifest()

    def _write_arrays(self, frame_id, kind, arrays, metadata):
        if self.disabled:
            raise RuntimeError('Capture has been stopped')
        if self.current_row is None or self.current_row['frame_id'] != int(frame_id):
            raise ValueError('Packet frame does not match the active chronological frame')
        source_arrays = {name: _array_numpy(value) for name, value in arrays.items() if value is not None}
        descriptors = {name: dict(shape=[int(size) for size in value.shape],
                                  dtype=str(value.dtype), nbytes=int(value.nbytes))
                       for name, value in source_arrays.items()}
        size = _packet_descriptor_bytes(kind, metadata, descriptors)
        if _working_copy_peak_bytes(size) > WORKING_COPY_BUDGET_BYTES:
            self._stop_capture('packet copies exceed the 128 MiB working-copy budget')
            raise RuntimeError('Working-copy budget exhausted')
        if self.manifest['copied_array_bytes'] + size > CAPTURE_BUDGET_BYTES:
            self._stop_capture('320 MiB aggregate raw copied-array budget reached; prior packets preserved without thinning')
            raise RuntimeError('Capture budget exhausted')
        payload = {name: np.array(value, copy=True, order='C') for name, value in source_arrays.items()}
        if sum(value.nbytes for value in payload.values()) != size:
            raise ValueError('Packet raw bytes changed after descriptor preflight')
        relative = f'packets/{self.branch}/{frame_id}-{kind}.npz'
        path = self.root / relative
        if path.exists():
            raise FileExistsError(f'Packet already exists and is preserved: {path.name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.part')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)
        self.manifest['copied_array_bytes'] += size
        array_bindings = {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                                     sha256=_sha_bytes(value.tobytes()))
                          for name, value in payload.items()}
        del payload
        return dict(path=relative, sha256=digest(path), array_bytes=size, arrays=array_bindings)

    def end_frame(self, decision):
        if self.current_row is None:
            self.frame = None
            return
        row = self.current_row
        row['public_decision'] = dict(pose_state=decision.get('pose_state'),
            render_state=decision.get('render_state'), mask_state=decision.get('mask_state'),
            failure_reason=decision.get('failure_reason'), cameraFromObject=decision.get('cameraFromObject'))
        captured_primary = any(inv.get('retained_iteration_seen') for inv in row['invocations']
            if inv.get('phase') == 'primary_tracking')
        if row.get('retained_packet') and row.get('result_packet'):
            row['availability'] = 'captured'
            row['unavailable_reason'] = None
        elif self.disabled:
            row['availability'] = 'unavailable'
            row['unavailable_reason'] = self.failure or 'capture_stopped'
        elif not row.get('mask_usable_at_entry'):
            row['availability'] = 'unavailable'
            row['unavailable_reason'] = 'observed_mask_unavailable_at_frame_entry'
        elif not row['entry_tracking']:
            row['availability'] = 'unavailable'
            row['unavailable_reason'] = 'recovery_only_at_frame_entry'
        elif not captured_primary:
            row['availability'] = 'unavailable'
            row['unavailable_reason'] = 'primary_refinement_ended_before_iteration_4'
        else:
            row['availability'] = 'unavailable'
            row['unavailable_reason'] = 'iteration_4_packet_pair_incomplete'
        self.branch_row['frame_records'].append(row)
        try:
            self._persist_manifest()
        except Exception as exc:
            self._disable_capture(f'frame decision persistence failed: {type(exc).__name__}: {exc}')
        self.current_row = None
        self.current_invocation = None
        self.frame = None

    def _persist_manifest(self):
        self.manifest['metadata_artifact_budget_bytes'] = METADATA_ARTIFACT_BUDGET_BYTES
        self.manifest['working_copy_budget_bytes'] = WORKING_COPY_BUDGET_BYTES
        _atomic_json(self.manifest_path, self.manifest, artifact_root=self.root)

    def _check_current_metadata_artifacts(self):
        total = _metadata_artifact_bytes(self.root)
        if total > METADATA_ARTIFACT_BUDGET_BYTES:
            self._stop_capture(f'metadata/trace/pose/log bytes exceed 64 MiB: {total}')
            return False
        self.manifest['metadata_artifact_bytes_observed'] = total
        return True

    def _stop_capture(self, reason):
        self.disabled = True
        self.manifest['capture_stopped'] = True
        self.failure = reason
        self.manifest['capture_stop_reason'] = reason
        self._persist_manifest()

    def _disable_capture(self, reason):
        self.disabled = True
        self.manifest['capture_stopped'] = True
        self.failure = reason
        self.manifest['capture_stop_reason'] = reason
        self.branch_row['status'] = 'partial'
        self._persist_manifest()

    def finish_branch(self, output_path):
        output = Path(output_path).resolve()
        expected = self.root / f'{self.branch}-201.json'
        if output != expected:
            self._disable_capture('pose output path does not match prepared branch output')
        try:
            if self.bundle is not None and self.masks_root is not None:
                self._validate_runtime_bindings(self.bundle, self.masks_root)
            else:
                _verify_manifest_source_freeze(self.manifest.get('bindings', {}), self.source_root)
        except Exception as exc:
            self._disable_capture(f'post-branch immutable binding check failed: {type(exc).__name__}: {exc}')
        if self.current_row is not None:
            self.end_frame(dict(pose_state='lost', render_state='suppressed', mask_state=None,
                                failure_reason='stage_ended_before_frame_decision', cameraFromObject=None))
        output_info = None
        if output.is_file():
            output_info = dict(path=output.name, sha256=digest(output))
        trace_info = None
        if self.trace_path.is_file():
            trace_info = dict(path=self.trace_path.name, sha256=digest(self.trace_path),
                              record_count=len(_trace_records(self.trace_path)))
        if not self._check_current_metadata_artifacts():
            self.branch_row['capture_failure'] = self.failure
        expected_rows = [row['frame_id'] for row in self.branch_row['frame_records']]
        all_rows = expected_rows == list(CAPTURE_FRAME_IDS)
        result_ok = output_info is not None and trace_info is not None
        status = 'stage_returned' if all_rows and result_ok and not self.disabled else 'partial'
        self.branch_row.update(status=status, output=output_info, trace=trace_info,
            capture_stopped=self.disabled, capture_failure=self.failure,
            captured_pairs=sum(row['availability'] == 'captured' for row in self.branch_row['frame_records']))
        self._persist_manifest()


class ChronologicalCaptureBackend:
    """Tag the first refiner call according to frame-entry tracker state."""

    def __init__(self, backend, refiner, observer):
        self.backend = backend
        self.refiner = refiner
        self.observer = observer

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def refine(self, frame, mask, seed):
        context = self.observer.enter_refinement(frame.frame_id)
        self.refiner.set_chronological_capture_context(context)
        try:
            return self.backend.refine(frame, mask, seed)
        finally:
            self.refiner.set_chronological_capture_context(None)
            self.observer.clear_invocation()

    def recover(self, frame, mask, top_k=5):
        result = self.backend.recover(frame, mask, top_k=top_k)
        self.observer.after_recovery(frame.frame_id)
        return result


def _nested_json_mismatch(left, right, path, float_tolerance=False):
    """Compare JSON structures without conflating bool/int/null or array order."""
    if type(left) is not type(right):
        return f'{path} value type mismatch ({type(left).__name__} != {type(right).__name__})'
    if isinstance(left, dict):
        if set(left) != set(right):
            missing = sorted(set(left) - set(right))
            extra = sorted(set(right) - set(left))
            return f'{path} key set mismatch (missing={missing}, extra={extra})'
        for key in sorted(left):
            mismatch = _nested_json_mismatch(left[key], right[key], f'{path}.{key}', float_tolerance)
            if mismatch:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f'{path} list length mismatch ({len(left)} != {len(right)})'
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = _nested_json_mismatch(a, b, f'{path}[{index}]', float_tolerance)
            if mismatch:
                return mismatch
        return None
    if type(left) is float:
        if float_tolerance:
            if not math.isfinite(left) or not math.isfinite(right):
                return f'{path} contains a nonfinite float'
            if not math.isclose(left, right, rel_tol=VALIDATION_RTOL, abs_tol=VALIDATION_ATOL):
                return f'{path} float differs beyond atol={VALIDATION_ATOL}, rtol={VALIDATION_RTOL}'
        elif left != right:
            return f'{path} value mismatch ({left!r} != {right!r})'
        return None
    if left != right:
        return f'{path} value mismatch ({left!r} != {right!r})'
    return None


def _frame_record_mismatch(left, right, label, require_mask_hash):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return f'{label} frame record is not an object'
    left_values = {key: value for key, value in left.items() if key != 'timings_ms'}
    right_values = {key: value for key, value in right.items() if key != 'timings_ms'}
    if set(left_values) != set(right_values):
        missing = sorted(set(left_values) - set(right_values))
        extra = sorted(set(right_values) - set(left_values))
        return f'{label} frame schema mismatch (missing={missing}, extra={extra})'
    required = {'frameId', 'cameraFromObject', 'mask_state', 'pose_state', 'render_state',
                'failure_reason', 'validation', 'rejection_reasons'}
    if require_mask_hash:
        required.add('mask_sha256')
    if not required.issubset(left_values):
        return f'{label} frame schema is missing required keys {sorted(required-set(left_values))}'
    if require_mask_hash and (not isinstance(left_values.get('mask_sha256'), str)
            or not left_values['mask_sha256']):
        return f'{label} mask hash is missing or malformed'
    for values, side in ((left_values, 'left'), (right_values, 'right')):
        if values.get('validation') is not None and not isinstance(values.get('validation'), dict):
            return f'{label} {side} validation is neither an object nor explicit null'
    if not isinstance(left_values.get('rejection_reasons'), list) or \
            not isinstance(right_values.get('rejection_reasons'), list):
        return f'{label} rejection_reasons must be an ordered list'
    for key in sorted(left_values):
        left_value, right_value = left_values[key], right_values[key]
        path = f'{label}.{key}'
        if key == 'cameraFromObject':
            if (left_value is None) != (right_value is None):
                return f'{label} null-pose mismatch'
            if left_value is not None:
                try:
                    pose_a, pose_b = np.asarray(left_value, dtype=np.float64), np.asarray(right_value, dtype=np.float64)
                except (TypeError, ValueError):
                    return f'{label} pose is malformed'
                if pose_a.shape != pose_b.shape or not np.isfinite(pose_a).all() or not np.isfinite(pose_b).all():
                    return f'{label} pose shape/nonfinite mismatch'
                if not np.allclose(pose_a, pose_b, atol=1e-6, rtol=0):
                    return f'{label} pose mismatch beyond atol=1e-6, rtol=0'
        elif key == 'validation' and left_value is None and right_value is None:
            continue
        elif key == 'validation' and (left_value is None or right_value is None):
            return f'{label} validation null/object mismatch'
        elif key == 'validation':
            mismatch = _nested_json_mismatch(left_value, right_value, path, float_tolerance=True)
            if mismatch:
                return mismatch
        else:
            mismatch = _nested_json_mismatch(left_value, right_value, path, float_tolerance=False)
            if mismatch:
                if key in ('frameId', 'mask_state', 'pose_state', 'render_state', 'failure_reason', 'mask_sha256'):
                    return f'{label} {key} mismatch: {mismatch}'
                return mismatch
    return None


def _frame_equal_prefix(short, long):
    a_frames = short.get('frames') if isinstance(short, dict) else None
    b_frames = long.get('frames') if isinstance(long, dict) else None
    if not isinstance(a_frames, list) or not isinstance(b_frames, list) or len(a_frames) != len(RUN_FRAME_IDS):
        return False, 'chronological output must contain all 201 frame rows'
    if len(b_frames) < len(a_frames):
        return False, 'V2 full output is shorter than the requested V5 prefix'
    for index, (left, right) in enumerate(zip(a_frames, b_frames)):
        mismatch = _frame_record_mismatch(left, right, f'frame prefix index {index}', require_mask_hash=True)
        if mismatch:
            return False, mismatch
    has_initialization_a = 'initialization' in short
    has_initialization_b = 'initialization' in long
    if has_initialization_a != has_initialization_b:
        return False, 'setup initialization decision presence mismatch'
    if has_initialization_a:
        init_a, init_b = short['initialization'], long['initialization']
        if (init_a is None) != (init_b is None):
            return False, 'setup initialization null-decision mismatch'
        if init_a is not None:
            mismatch = _frame_record_mismatch(init_a, init_b, 'setup initialization', require_mask_hash=False)
            if mismatch:
                return False, mismatch
    return True, None


def _load_bound_packet(root, info):
    if not isinstance(info, dict) or not isinstance(info.get('path'), str):
        raise ValueError('Capture packet path/hash binding is missing')
    path = (Path(root) / info['path']).resolve()
    path.relative_to(Path(root).resolve())
    if not path.is_file() or digest(path) != info.get('sha256'):
        raise ValueError(f'Capture packet bytes do not match their immutable binding: {path.name}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = info.get('arrays')
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError(f'Capture packet array inventory differs: {path.name}')
    for name, value in arrays.items():
        binding = expected[name]
        if (binding.get('shape') != list(value.shape) or binding.get('dtype') != str(value.dtype) or
                binding.get('sha256') != _sha_bytes(value.tobytes())):
            raise ValueError(f'Capture packet array binding differs: {path.name}:{name}')
    if sum(value.nbytes for value in arrays.values()) != info.get('array_bytes'):
        raise ValueError(f'Capture packet byte count differs: {path.name}')
    return arrays


def _validate_capture_branch(root, branch, manifest, v2_root, v2, source_root=ROOT):
    row = manifest.get('branches', {}).get(branch, {})
    issues = []
    binding_record = manifest.get('bindings', {})
    if binding_record.get('v2_experiment_sha256') != digest(Path(v2_root) / 'experiment.json') or \
       binding_record.get('v2_stage_evidence_sha256') != digest(Path(v2_root) / 'stage-evidence.json') or \
       binding_record.get('v2_report_sha256') != digest(Path(v2_root) / 'report.json'):
        issues.append('V2 experiment/evidence/report bytes differ from the prepared immutable bindings')
    if binding_record.get('prior_inference_provenance_sha256') != _json_sha(v2['provenance']):
        issues.append('prior V2 inference provenance hash differs from the prepared binding')
    if binding_record.get('instrumentation_only_source_delta') != v2['source_delta']:
        issues.append('V2 to observer source delta differs from its explicit reviewed classification')
    if binding_record.get('auxiliary_source_delta') != v2['auxiliary_source_delta'] or \
            binding_record.get('approved_classified_delta_sha256') != v2['compatibility_review']['record']['approved_delta_sha256']:
        issues.append('Auxiliary source delta differs from the independently accepted canonical classification')
    if binding_record.get('captured_source_hashes') != v2['experiment'].get('source_files') or \
            binding_record.get('current_source_hashes') != v2['current_source_hashes']:
        issues.append('Full V2/current source maps differ from the accepted compatibility record')
    if binding_record.get('legacy_review_record_sha256') != v2['compatibility_review']['sha256'] or \
            binding_record.get('legacy_review_record_path') != v2['compatibility_review']['relative_path']:
        issues.append('Legacy source/mask compatibility evidence differs from its immutable record')
    if binding_record.get('frozen_files') != v2['experiment'].get('frozen_files') or \
       binding_record.get('prior_source_files') != v2['experiment'].get('source_files'):
        issues.append('V2 frozen/source file manifests differ from the prepared bindings')
    records = row.get('frame_records')
    if not isinstance(records, list) or [item.get('frame_id') for item in records] != list(CAPTURE_FRAME_IDS):
        return ['capture rows do not contain the exact twelve fixed source IDs']
    if row.get('capture_stopped') is True or manifest.get('capture_stopped') is True:
        issues.append('resource-v2 raw-copy overflow or capture failure stopped evidence; only preserved partial data is available')
    for entry in records:
        frame_id = entry['frame_id']
        state = entry.get('availability')
        invocation_ids = [item.get('invocation_ordinal') for item in entry.get('invocations', [])]
        if invocation_ids != list(range(1, len(invocation_ids) + 1)):
            issues.append(f'{branch} frame {frame_id} invocation ordinals are missing or out of order')
        for invocation in entry.get('invocations', []):
            ordinals = invocation.get('trace_ordinals', [])
            if ordinals != sorted(ordinals) or any(type(value) is not int for value in ordinals):
                issues.append(f'{branch} frame {frame_id} inference trace ordinals are malformed')
        for recovery in entry.get('recoveries', []):
            if recovery.get('trace_ordinal') is None and recovery.get('trace_missing') is not True:
                issues.append(f'{branch} frame {frame_id} recovery invocation lacks explicit trace accounting')
        if state == 'captured':
            if not entry.get('entry_tracking') or not entry.get('mask_usable_at_entry'):
                issues.append(f'{branch} frame {frame_id} captured outside a usable primary tracking state')
                continue
            retained_info, result_info = entry.get('retained_packet'), entry.get('result_packet')
            try:
                retained = _load_bound_packet(root, retained_info)
                result = _load_bound_packet(root, result_info)
                context_key = entry.get('context_key')
                context_info = manifest.get('contexts', {}).get(context_key)
                if not isinstance(context_info, dict):
                    raise ValueError('deduplicated native RGB/mask context binding is missing')
                context_path = (Path(root) / context_info['path']).resolve()
                context_path.relative_to(Path(root).resolve())
                if not context_path.is_file() or digest(context_path) != context_info.get('sha256'):
                    raise ValueError('deduplicated native context bytes do not match their binding')
                with np.load(context_path, allow_pickle=False) as archive:
                    native_rgb = np.asarray(archive['native_rgb'])
                    observed_mask = np.asarray(archive['observed_native_mask'])
                    native_k = np.asarray(archive['native_k'])
                _native_context_bytes(native_rgb, observed_mask, native_k)
                context_descriptor = (str(frame_id) + '|' + str(native_rgb.dtype) + '|' + str(native_rgb.shape)
                    + '|' + str(observed_mask.dtype) + '|' + str(observed_mask.shape)).encode('ascii')
                context_digest = _sha_bytes(context_descriptor + native_rgb.tobytes() + observed_mask.tobytes() + native_k.tobytes())
                if context_key != context_digest:
                    raise ValueError('deduplicated native RGB/mask context content hash differs')
                if (context_info.get('rgb_sha256') != _sha_bytes(native_rgb.tobytes()) or
                        context_info.get('observed_mask_sha256') != _sha_bytes(observed_mask.tobytes())):
                    raise ValueError('native source RGB/mask hashes differ from their context manifest')
                required_retained = {'query_rgb_crop', 'template_rgb', 'rendered_depth_mm', 'rendered_mask',
                    'observed_crop_mask', 'full_flow_crop_px', 'full_confidence', 'full_obj_points_mm',
                    'full_target_crop_px', 'source_flat_indices', 'source_crop_pixel_centers', 'sample_ids',
                    'sample_obj_points_mm', 'sample_target_crop_px', 'sample_weights', 'crop_k',
                    'crop_from_orig', 'native_k', 'seed_camera_from_object_m',
                    'current_crop_camera_from_object_mm', 'initial_rvec', 'initial_tvec_mm'}
                if not required_retained.issubset(retained):
                    raise ValueError(f'retained array set is missing {sorted(required_retained-set(retained))}')
                retained_descriptors = {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                    nbytes=int(value.nbytes)) for name, value in retained.items()}
                result_descriptors = {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                    nbytes=int(value.nbytes)) for name, value in result.items()}
                retained_bytes = _packet_descriptor_bytes('retained',
                    retained_info.get('event_metadata', {}), retained_descriptors)
                result_bytes = _packet_descriptor_bytes('result',
                    result_info.get('event_metadata', {}), result_descriptors)
                if retained_bytes != retained_info.get('array_bytes') or result_bytes != result_info.get('array_bytes'):
                    raise ValueError('packet byte counters differ from validated resource-v2 descriptors')
                if native_rgb.dtype != np.uint8 or native_rgb.ndim != 3 or native_rgb.shape[2] != 3:
                    raise ValueError('native source image is not lossless uint8 RGB')
                if observed_mask.shape != native_rgb.shape[:2] or native_k.shape != (3, 3):
                    raise ValueError('native observed mask or camera binding has an invalid shape')
                np.testing.assert_array_equal(retained['native_k'], native_k)
                source_ids = retained['source_flat_indices'].astype(np.int64).reshape(-1)
                source_centers = retained['source_crop_pixel_centers'].reshape(-1, 2)
                full_obj = retained['full_obj_points_mm'].reshape(-1, 3)
                full_target = retained['full_target_crop_px'].reshape(-1, 2)
                sample_ids = retained['sample_ids'].astype(np.int64).reshape(-1)
                if not (len(source_ids) == len(source_centers) == len(full_obj) == len(full_target)):
                    raise ValueError('retained source indices, pixel centers, and point arrays differ in length')
                if len(source_ids) and (source_ids.min() < 0 or source_ids.max() >= 280*280):
                    raise ValueError('retained source flat index is outside the 280x280 crop')
                expected_centers = np.column_stack((source_ids % 280 + .5, source_ids // 280 + .5))
                if not np.allclose(source_centers, expected_centers, atol=1e-7, rtol=0):
                    raise ValueError('retained source coordinates do not preserve production pixel centers')
                if len(sample_ids) and (sample_ids.min() < 0 or sample_ids.max() >= len(source_ids)):
                    raise ValueError('sample IDs do not index the full retained correspondence set')
                np.testing.assert_array_equal(retained['sample_obj_points_mm'], full_obj[sample_ids])
                np.testing.assert_array_equal(retained['sample_target_crop_px'], full_target[sample_ids])
                flat_flow = retained['full_flow_crop_px'].reshape(-1, 2)
                np.testing.assert_allclose(full_target, source_centers + flat_flow[source_ids], atol=1e-7, rtol=0)
                confidence = retained['full_confidence'].reshape(-1)
                np.testing.assert_allclose(retained['sample_weights'], confidence[source_ids][sample_ids], atol=0, rtol=0)
                if retained_info.get('event_metadata', {}).get('iteration') != CAPTURE_ITERATION or \
                   result_info.get('event_metadata', {}).get('iteration') != CAPTURE_ITERATION:
                    raise ValueError('captured pair is not the fixed zero-based iteration 4')
                for info in (retained_info, result_info):
                    invocation = info.get('invocation', {})
                    if invocation.get('phase') != 'primary_tracking' or invocation.get('frame_id') != frame_id:
                        raise ValueError('captured pair is not from the first primary tracking invocation')
                    if type(info.get('trace_ordinal')) is not int:
                        raise ValueError('captured packet has no bound inference trace ordinal')
                    matching = [inv for inv in entry['invocations']
                        if inv.get('invocation_ordinal') == invocation.get('invocation_ordinal')]
                    if len(matching) != 1 or matching[0].get('iteration_trace_ordinals', {}).get('4') != info.get('trace_ordinal'):
                        raise ValueError('captured packet trace ordinal differs from its invocation/iteration')
                ransac_ids = result.get('ransac_inlier_ids', np.empty(0, dtype=int)).astype(np.int64).reshape(-1)
                retained_ids = result.get('retained_inlier_ids', np.empty(0, dtype=int)).astype(np.int64).reshape(-1)
                if len(ransac_ids) and (ransac_ids.min() < 0 or ransac_ids.max() >= len(sample_ids)):
                    raise ValueError('RANSAC inlier ID is outside the deterministic sample')
                np.testing.assert_array_equal(retained_ids, sample_ids[ransac_ids])
                if result_info.get('event_metadata', {}).get('fit_state') == 'refined':
                    crop_pose = result['crop_camera_from_object_mm']
                    native_pose = result['native_camera_from_object_m']
                    expected_native = np.linalg.inv(retained['crop_from_orig']) @ crop_pose
                    expected_native[:3, 3] *= .001
                    np.testing.assert_allclose(native_pose, expected_native, atol=1e-10, rtol=1e-9)
                elif result_info.get('event_metadata', {}).get('fit_state') not in (
                        'ransac_unavailable', 'insufficient_retained_correspondences'):
                    raise ValueError('captured result has an unknown solver terminal state')
            except Exception as exc:
                issues.append(f'{branch} frame {frame_id} packet validation failed: {type(exc).__name__}: {exc}')
        elif state == 'unavailable':
            reason = entry.get('unavailable_reason')
            if reason == 'recovery_only_at_frame_entry':
                if entry.get('entry_tracking') is not False:
                    issues.append(f'{branch} frame {frame_id} recovery-only status contradicts frame entry state')
                if not entry.get('invocations') and not entry.get('recoveries'):
                    issues.append(f'{branch} frame {frame_id} lacks invocation/trace ordinal accounting')
            elif reason == 'primary_refinement_ended_before_iteration_4':
                if entry.get('entry_tracking') is not True or not entry.get('invocations'):
                    issues.append(f'{branch} frame {frame_id} early-stop status lacks primary invocation evidence')
                if entry.get('mask_usable_at_entry') is not True:
                    issues.append(f'{branch} frame {frame_id} early-stop contradicts observed-mask availability')
                elif not any(inv.get('phase') == 'primary_tracking' and inv.get('trace_ordinals')
                             for inv in entry['invocations']):
                    issues.append(f'{branch} frame {frame_id} early-stop has no trace ordinal evidence')
            elif reason == 'observed_mask_unavailable_at_frame_entry':
                if entry.get('mask_usable_at_entry') is not False:
                    issues.append(f'{branch} frame {frame_id} mask-unavailable reason contradicts mask state')
            else:
                issues.append(f'{branch} frame {frame_id} has an unrecognized unavailable reason {reason!r}')
        else:
            issues.append(f'{branch} frame {frame_id} lacks a terminal captured/unavailable decision')
    if row.get('captured_pairs') != sum(item.get('availability') == 'captured' for item in records):
        issues.append(f'{branch} captured-pair count differs from its frame accounting')
    if row.get('captured_pairs', 0) > len(CAPTURE_FRAME_IDS):
        issues.append(f'{branch} exceeds the twelve-pair branch limit')
    if manifest.get('copied_array_bytes', CAPTURE_BUDGET_BYTES+1) > CAPTURE_BUDGET_BYTES:
        issues.append('aggregate copied packet/image/mask data exceeds 320 MiB')
    contexts = manifest.get('contexts', {})
    if not isinstance(contexts, dict) or len(contexts) > MAX_NATIVE_CONTEXTS:
        issues.append('native context inventory exceeds the frozen twelve-context cohort')
    elif any(not isinstance(info, dict) or type(info.get('array_bytes')) is not int or
             info['array_bytes'] > MAX_NATIVE_CONTEXT_BYTES for info in contexts.values()):
        issues.append('native context descriptors exceed the frozen per-context byte bound')
    return issues


def verify_trajectory(v2_root, capture_root, source_root=ROOT):
    """Require exact branch output/state and trace-prefix equivalence to V2."""
    root = Path(capture_root).resolve()
    manifest = _read_json(root / 'capture.json')
    source_root = Path(source_root).resolve()
    if manifest.get('schema_version') != CAPTURE_SCHEMA_VERSION or manifest.get('experiment') != CAPTURE_EXPERIMENT:
        raise ValueError('Trajectory verifier requires a fresh resource-v2 capture root')
    bindings = manifest.get('bindings', {})
    _verify_manifest_source_freeze(bindings, source_root)
    review_path = _safe_relative_path(source_root, bindings.get('legacy_review_record_path'))
    v2 = _validate_v2_terminal(v2_root, source_root, review_path)
    if Path(v2_root).resolve() != Path(manifest.get('bindings', {}).get('v2_root', '')).resolve():
        raise ValueError('Trajectory verifier V2 root differs from the immutable prepared binding')
    segmentation_path = Path(bindings.get('masks_root', '')) / 'results.json'
    segmentation = _read_json(segmentation_path)
    legacy_review = _validate_legacy_segmentation_record(v2['compatibility_review']['record'],
        segmentation, segmentation_path, bindings.get('bundle_root'), bindings.get('masks_root'),
        v2['experiment'], source_root)
    if legacy_review != bindings.get('legacy_segmentation_review'):
        raise ValueError('Legacy segmentation compatibility evidence differs from its prepared binding')
    report = dict(schema_version=2, experiment=CAPTURE_EXPERIMENT,
        capacity_spec=bindings.get('capacity_spec'), v2_terminal_bindings_passed=True, branches={},
        source_instrumentation_delta=v2['source_delta'], independent_accuracy_scored=False,
        auxiliary_source_delta=v2['auxiliary_source_delta'],
        compatibility_review_id=v2['compatibility_review']['record']['review_id'],
        compatibility_review_sha256=v2['compatibility_review']['sha256'],
        legacy_mode_limitations=bindings.get('legacy_mode_limitations'),
        surface_identity_interpreted=False, passed=False)
    issues = []
    if manifest.get('status') != 'complete':
        issues.append('both chronological branch processes are not terminal exit-zero')
    if manifest.get('capture_stopped') is True:
        issues.append('capture overflow/failure preserved only partial evidence')
    if manifest.get('frame_ids') != list(RUN_FRAME_IDS) or manifest.get('capture_frame_ids') != list(CAPTURE_FRAME_IDS):
        issues.append('chronological run/capture frame IDs differ from the fixed specification')
    if manifest.get('capture_budget_bytes') != CAPTURE_BUDGET_BYTES or manifest.get('copied_array_bytes', CAPTURE_BUDGET_BYTES+1) > CAPTURE_BUDGET_BYTES:
        issues.append('resource-v2 raw copied-array budget is missing or exceeded')
    if bindings.get('capacity_spec') != dict(path=CAPACITY_SPEC_PATH, sha256=CAPACITY_SPEC_SHA256):
        issues.append('resource-v2 capacity/comparator specification pin is missing or changed')
    if manifest.get('metadata_artifact_budget_bytes') != METADATA_ARTIFACT_BUDGET_BYTES or \
            manifest.get('working_copy_budget_bytes') != WORKING_COPY_BUDGET_BYTES:
        issues.append('resource-v2 metadata or working-copy budget is missing or changed')
    if manifest.get('planned_max_capture_bytes') != MAX_PLANNED_CAPTURE_BYTES or \
            manifest.get('planned_capture_headroom_bytes') != MAX_PLANNED_CAPTURE_HEADROOM_BYTES or \
            MAX_PLANNED_CAPTURE_BYTES != (MAX_PACKET_PAIRS * MAX_PACKET_PAIR_BYTES
                + MAX_NATIVE_CONTEXTS * MAX_NATIVE_CONTEXT_BYTES):
        issues.append('resource-v2 maximum-cohort capacity proof is missing or inconsistent')
    if manifest.get('max_packet_pairs') != MAX_PACKET_PAIRS or \
            manifest.get('max_native_contexts') != MAX_NATIVE_CONTEXTS:
        issues.append('resource-v2 packet/context cohort limits are missing or changed')
    if bindings.get('cache_preflight', {}).get('reservation_bytes') != CACHE_PREPARE_RESERVATION_BYTES or \
            bindings.get('cache_preflight', {}).get('projected_cache_bytes', MODEL_CACHE_BUDGET_BYTES+1) > MODEL_CACHE_BUDGET_BYTES:
        issues.append('resource-v2 8 GiB model-cache preflight is missing or over budget')
    try:
        metadata_bytes = _metadata_artifact_bytes(root)
    except Exception as exc:
        metadata_bytes = METADATA_ARTIFACT_BUDGET_BYTES + 1
        issues.append(f'metadata/trace/pose/log allowance could not be measured: {type(exc).__name__}: {exc}')
    if metadata_bytes > METADATA_ARTIFACT_BUDGET_BYTES:
        issues.append(f'metadata/trace/pose/log bytes exceed 64 MiB: {metadata_bytes}')
    report['metadata_artifact_bytes'] = metadata_bytes
    report['metadata_artifact_budget_bytes'] = METADATA_ARTIFACT_BUDGET_BYTES
    accounted_bytes = sum(item.get('array_bytes', 0) for item in manifest.get('contexts', {}).values())
    for branch_row in manifest.get('branches', {}).values():
        for frame_row in branch_row.get('frame_records', []):
            for packet_key in ('retained_packet', 'result_packet'):
                info = frame_row.get(packet_key)
                if isinstance(info, dict):
                    accounted_bytes += info.get('array_bytes', 0)
    if accounted_bytes != manifest.get('copied_array_bytes'):
        issues.append('manifest copied-array total does not equal deduplicated contexts plus packet arrays')
    current_provenance = manifest.get('bindings', {}).get('current_inference_provenance')
    if not isinstance(current_provenance, dict):
        issues.append('current instrumentation provenance is missing')
    else:
        for key in ('input_manifest_sha256', 'source_revisions', 'submodule_revisions', 'checkpoint_sha256', 'foundpose_bank_sha256'):
            if current_provenance.get(key) != v2['provenance'].get(key):
                issues.append(f'V2 immutable inference binding differs: {key}')
        for name, expected in v2['provenance']['adapter_sha256'].items():
            if name in ('quality_gotrack', 'quality_runner'):
                current = manifest['bindings']['current_source_sha256'].get(f'bench/{name}.py')
                if current != current_provenance['adapter_sha256'].get(name):
                    issues.append(f'{name} current reviewed instrumentation hash does not match run provenance')
            elif current_provenance['adapter_sha256'].get(name) != expected:
                issues.append(f'Non-instrumented adapter changed: {name}')
    for relative, expected in manifest.get('bindings', {}).get('current_source_sha256', {}).items():
        path = _safe_relative_path(source_root, relative)
        if not path.is_file() or digest(path) != expected:
            issues.append(f'current reviewed source changed after capture: {relative}')
    capture_module = Path(source_root).resolve() / 'bench/quality_mug_chronological_capture.py'
    if not capture_module.is_file() or digest(capture_module) != manifest.get('bindings', {}).get('new_capture_module_sha256'):
        issues.append('new capture module source hash changed after capture')
    for branch in BRANCHES:
        row = manifest.get('branches', {}).get(branch, {})
        branch_report = dict(output_passed=False, settings_passed=False, provenance_passed=False,
                             trace_passed=False, capture_accounting_passed=False, complete=False, issues=[])
        if row.get('status') != 'terminal' or row.get('exit_code') != 0:
            branch_report['issues'].append('branch lacks terminal exit-zero record')
        output_info, trace_info = row.get('output'), row.get('trace')
        if not isinstance(output_info, dict) or not isinstance(trace_info, dict):
            branch_report['issues'].append('branch output/trace bindings are missing')
            report['branches'][branch] = branch_report
            issues.extend(f'{branch}: {item}' for item in branch_report['issues'])
            continue
        output_path = root / output_info.get('path', '')
        trace_path = root / trace_info.get('path', '')
        if not output_path.is_file() or digest(output_path) != output_info.get('sha256'):
            branch_report['issues'].append('branch output bytes do not match terminal SHA256')
        if not trace_path.is_file() or digest(trace_path) != trace_info.get('sha256'):
            branch_report['issues'].append('branch trace bytes do not match terminal SHA256')
        if branch_report['issues']:
            report['branches'][branch] = branch_report
            issues.extend(f'{branch}: {item}' for item in branch_report['issues'])
            continue
        short = _read_json(output_path)
        capture_issues = _validate_capture_branch(root, branch, manifest, v2_root, v2, source_root)
        branch_report['capture_accounting_passed'] = not capture_issues
        branch_report['issues'].extend(capture_issues)
        full_stage = next(stage for stage in v2['evidence']['stages'] if stage['stage_id'] == f'{branch}-240')
        long_path = Path(v2_root) / full_stage['output']
        long = _read_json(long_path)
        pose_equal, mismatch = _frame_equal_prefix(short, long)
        branch_report['output_passed'] = pose_equal
        if mismatch:
            branch_report['issues'].append(mismatch)
        branch_report['settings_passed'] = short.get('tracking_settings', {}).get('pnp_use_extrinsic_guess') is BRANCH_GUESS[branch]
        for key, expected in REQUIRED_V4_SETTINGS.items():
            branch_report['settings_passed'] &= type(short.get('tracking_settings', {}).get(key)) is bool and short['tracking_settings'][key] is expected
        branch_report['provenance_passed'] = short.get('provenance') == current_provenance
        if short.get('runtime') != long.get('runtime'):
            branch_report['issues'].append('runtime differs from immutable V2 branch')
        if not branch_report['settings_passed']:
            branch_report['issues'].append('chronological branch settings differ from the frozen V2 comparison')
        if not branch_report['provenance_passed']:
            branch_report['issues'].append('run provenance differs from reviewed instrumentation and immutable V2 pins')
        try:
            new_trace = _trace_records(trace_path)
            old_trace = _read_v2_trace(v2_root, branch)
            expected_prefix = _v2_trace_prefix(old_trace, RUN_FRAME_IDS[-1])
            trace_mismatch = _compare_trace_records(new_trace, expected_prefix)
            if trace_info.get('record_count') != len(new_trace):
                trace_mismatch = dict(reason='terminal_trace_record_count_mismatch',
                    declared=trace_info.get('record_count'), actual=len(new_trace))
            branch_report['trace_passed'] = trace_mismatch is None
            if trace_mismatch:
                branch_report['issues'].append('trace ' + str(trace_mismatch))
        except Exception as exc:
            branch_report['issues'].append(f'trace prefix validation failed: {type(exc).__name__}: {exc}')
        branch_report['complete'] = (short.get('mode') == 'complete' and short.get('frame_ids') == list(RUN_FRAME_IDS)
            and short.get('complete') is False and [frame.get('frameId') for frame in short.get('frames', [])] == list(RUN_FRAME_IDS))
        if not branch_report['complete']:
            branch_report['issues'].append('chronological output is not the exact incomplete 201-frame prefix')
        if not all(branch_report[key] for key in ('output_passed', 'settings_passed', 'provenance_passed',
                'trace_passed', 'capture_accounting_passed', 'complete')):
            issues.extend(f'{branch}: {item}' for item in branch_report['issues'])
        report['branches'][branch] = branch_report
    report['issues'] = issues
    report['passed'] = not issues and all(all(row[key] for key in ('output_passed', 'settings_passed',
        'provenance_passed', 'trace_passed', 'capture_accounting_passed', 'complete'))
        for row in report['branches'].values())
    try:
        _check_metadata_artifact_budget(root, len((json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8')))
    except Exception as exc:
        report['passed'] = False
        report.setdefault('issues', []).append(f'verification report exceeds metadata artifact allowance: {exc}')
    _atomic_json(root / 'trajectory-verification.json', report, artifact_root=root)
    return report


def record_terminal_exit(root, branch, exit_code):
    if branch not in BRANCHES or type(exit_code) is not int:
        raise ValueError('Terminal record requires a known branch and integer process exit code')
    target = Path(root).resolve()
    manifest_path = target / 'capture.json'
    manifest = _read_json(manifest_path)
    row = manifest['branches'][branch]
    if exit_code != 0:
        previous_status = row.get('status')
        if previous_status not in ('prepared', 'running', 'stage_returned', 'partial'):
            raise ValueError(f'Cannot close failed process from branch state {previous_status!r}')
        artifacts = {}
        for key, name in (('output', f'{branch}-201.json'),
                          ('trace', f'{branch}-201.trace.jsonl')):
            path = target / name
            if path.is_file():
                artifacts[key] = dict(path=name, sha256=digest(path))
        row['status'] = 'failed'
        row['exit_code'] = exit_code
        row['failure_previous_status'] = previous_status
        row['failure_artifacts'] = artifacts
        manifest['status'] = 'failed'
    else:
        _verify_manifest_source_freeze(manifest.get('bindings', {}))
        if row.get('status') != 'stage_returned':
            raise ValueError('Exit code 0 requires a completed runner stage')
        if row.get('captured_pairs', 0) > len(CAPTURE_FRAME_IDS):
            raise ValueError('Capture packet pair count exceeds the fixed frame window')
        for info in (row.get('output'), row.get('trace')):
            if not isinstance(info, dict) or not (target / info.get('path', '')).is_file() or digest(target / info['path']) != info.get('sha256'):
                raise ValueError('Stage output or trace changed before terminal exit binding')
        row['status'] = 'terminal'
        row['exit_code'] = 0
    if all(manifest['branches'][name].get('status') == 'terminal' and
           manifest['branches'][name].get('exit_code') == 0 for name in BRANCHES):
        manifest['status'] = 'complete'
    _atomic_json(manifest_path, manifest)
    return manifest


def _prepare_command(args):
    result = prepare_capture_root(args.bundle, args.masks, args.v2_root, args.capture_root,
        args.source_root, args.legacy_review_record)
    print(json.dumps(dict(root=str(Path(args.capture_root).resolve()), status=result['status'],
        run_order=list(BRANCHES), capture_frames=list(CAPTURE_FRAME_IDS), budget_bytes=CAPTURE_BUDGET_BYTES), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='validate immutable V2 pins and write fresh sequential branch commands')
    prepare.add_argument('--bundle', type=Path, required=True)
    prepare.add_argument('--masks', type=Path, required=True)
    prepare.add_argument('--v2-root', type=Path, required=True)
    prepare.add_argument('--capture-root', type=Path, required=True)
    prepare.add_argument('--source-root', type=Path, default=ROOT)
    prepare.add_argument('--legacy-review-record', type=Path,
        help='Sol-accepted exact V2 source/legacy-mask compatibility record; required for omitted V4 mode flags')
    terminal = sub.add_parser('terminal', help='bind the actual process exit after one prepared pose stage returns')
    terminal.add_argument('--root', type=Path, required=True)
    terminal.add_argument('--branch', choices=BRANCHES, required=True)
    terminal.add_argument('--exit-code', type=int, required=True)
    verify = sub.add_parser('verify-trajectory', help='compare completed chronological outputs/traces to exact V2 prefixes')
    verify.add_argument('--v2-root', type=Path, required=True)
    verify.add_argument('--capture-root', type=Path, required=True)
    verify.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'prepare':
        _prepare_command(args)
    elif args.command == 'terminal':
        result = record_terminal_exit(args.root, args.branch, args.exit_code)
        print(json.dumps(dict(branch=args.branch, status=result['branches'][args.branch]['status'],
                              exit_code=args.exit_code), indent=2))
    else:
        result = verify_trajectory(args.v2_root, args.capture_root, args.source_root)
        print(json.dumps(dict(passed=result['passed'], issues=result['issues']), indent=2))
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
