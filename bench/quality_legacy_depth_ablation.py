"""Opt-in paired legacy-depth experiment for the original ranch bottle window.

This file is a source-only candidate.  Its public renderer wrapper is kept
separate from the normal benchmark runner; root owns validation and execution.
The fixed color/template crop and historical GoTrack/contract policy are shared
by both branches.  Only crop-template depth changes, under a common synthetic
template support mask.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np

from .quality_assets import CACHE, digest, inference_provenance


_WIDTH = 280
_HEIGHT = 280
_SETUP_ID = 9
_FIRST_ID = 10
_LAST_ID = 249
_FULL_IDS = tuple(range(_FIRST_ID, _LAST_ID + 1))
_PREFIX_LENGTHS = (120, 240)
_OUTPUT_ROOT = CACHE / 'diagnostics' / 'legacy-depth-v1'
_BUNDLE = CACHE / 'inputs' / 'ranch'
_MASKS = CACHE / 'results' / 'ranch' / 'segmentation'
_BANK = CACHE / 'banks' / 'ranch-foundpose.pt'
_REPORT_BYTE_LIMIT = 32 * 1024**2
_REPORT_FAILURE_RESERVE_BYTES = 16 * 1024

_LEGACY_RENDER_POLICY = 'legacy_v1'
_CAPTURE_RENDER_POLICY = 'capture_zero_sample_v1'
_LEGACY_COORDINATE_MODE = 'legacy'
_CAPTURE_COORDINATE_MODE = 'integer_centers_v1'


class RendererIntegrityError(RuntimeError):
    """Raised when an actual paired renderer call cannot be trusted."""


class ReportBudgetError(RuntimeError):
    """Raised after preserving the last bounded report and writing a short summary."""


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialize_report_bytes(value):
    """Return the exact compact UTF-8 bytes used by this module's writer."""
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _report_storage_bytes(root=None):
    root = Path(_OUTPUT_ROOT if root is None else root).resolve()
    if not root.exists():
        return 0
    total = 0
    for pattern in ('*.json', '*.json.tmp'):
        for path in root.rglob(pattern):
            if path.is_file():
                total += path.stat().st_size
    return total


def _write_report_payload(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists():
        raise FileExistsError(f'refusing to overwrite report temporary file: {temporary}')
    created = False
    try:
        with temporary.open('xb') as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _budget_failure_summary(path, context, rejected_bytes, current_bytes):
    context = dict(context) if isinstance(context, Mapping) else {}
    frame_ids = context.get('frame_ids', [])
    unattempted = context.get('unattempted_frame_ids', [])
    if not isinstance(frame_ids, list):
        frame_ids = []
    if not isinstance(unattempted, list):
        unattempted = []
    last_digest = _frame_hash(path) if Path(path).is_file() else None
    return {
        'schema_version': 1,
        'experiment': context.get('experiment', 'legacy_depth_ablation_v1'),
        'status': 'incomplete_budget_failure',
        'complete': False,
        'branch': context.get('branch'),
        'phase': context.get('phase', 'report_write'),
        'failure_reason': 'aggregate diagnostic report budget exceeded; prior bounded report retained',
        'attempted_frame_count': len(frame_ids),
        'attempted_frame_ids': frame_ids,
        'unattempted_frame_ids': unattempted,
        'geometry_ray_count': context.get('geometry_ray_count'),
        'last_bounded_report_sha256': last_digest,
        'rejected_report_bytes': int(rejected_bytes),
        'aggregate_bytes_before_rejected_write': int(current_bytes),
        'aggregate_report_budget_bytes': _REPORT_BYTE_LIMIT,
        'is_tracking_accuracy_evidence': False,
    }


def _persist_report(path, value, *, budget_context, reserve_failure_bytes=_REPORT_FAILURE_RESERVE_BYTES):
    """Atomically persist the exact module-owned JSON bytes inside the aggregate budget."""
    path = Path(path).resolve()
    root = Path(_OUTPUT_ROOT).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError('report output must remain inside the fixed legacy-depth-v1 directory') from exc
    payload = _serialize_report_bytes(value)
    current_bytes = _report_storage_bytes(root)
    reserve = max(0, int(reserve_failure_bytes))
    if current_bytes + len(payload) + reserve > _REPORT_BYTE_LIMIT:
        summary = _budget_failure_summary(path, budget_context, len(payload), current_bytes)
        summary_path = path.with_name(path.name + '.budget-failure.json')
        if summary_path.exists():
            raise ReportBudgetError('report budget exceeded; prior bounded report and failure summary retained')
        summary_payload = _serialize_report_bytes(summary)
        if current_bytes + len(summary_payload) > _REPORT_BYTE_LIMIT:
            raise ReportBudgetError('report budget exceeded and reserved failure summary could not fit')
        _write_report_payload(summary_path, summary_payload)
        raise ReportBudgetError('aggregate legacy-depth report budget exceeded; bounded failure summary written')
    _write_report_payload(path, payload)
    return len(payload)


def _array_digest(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _camera_pin(camera):
    pose = np.asarray(camera.T_world_from_eye, dtype=np.float64)
    return {
        'dimensions': [int(camera.width), int(camera.height)],
        'f': [float(camera.f[0]), float(camera.f[1])],
        'c': [float(camera.c[0]), float(camera.c[1])],
        'T_world_from_eye': pose.tolist(),
        'T_world_from_eye_sha256': _array_digest(pose.astype('<f8', copy=False)),
    }


def legacy_inner_camera(camera):
    """Return a fresh integer-center camera with legacy half-center GL rays."""
    from utils import structs

    return structs.PinholePlaneCameraModel(
        width=int(camera.width), height=int(camera.height),
        f=(float(camera.f[0]), float(camera.f[1])),
        c=(float(camera.c[0]) - .5, float(camera.c[1]) - .5),
        T_world_from_eye=np.array(camera.T_world_from_eye, dtype=np.float64, copy=True))


def _validate_zero_metadata(metadata, *, expected_dimensions, previous):
    if not isinstance(metadata, Mapping):
        raise RuntimeError('zero-sample renderer returned no public render_policy_metadata')
    plain = _plain(metadata)
    if (plain.get('schema_version') != 1 or
            plain.get('render_policy') != _CAPTURE_RENDER_POLICY or
            plain.get('coordinate_mode') != _CAPTURE_COORDINATE_MODE or
            plain.get('depth_units') != 'millimetres'):
        raise RuntimeError('zero-sample renderer metadata has the wrong public policy contract')
    if tuple(plain.get('dimensions', ())) != tuple(expected_dimensions):
        raise RuntimeError('zero-sample renderer metadata dimensions differ from the actual crop')
    for field in ('framebuffer_complete', 'framebuffer_bindings_restored',
                  'current_context_released', 'dimension_match'):
        if plain.get(field) is not True:
            raise RuntimeError(f'zero-sample renderer did not prove {field}')
    for field in ('gl_samples', 'gl_sample_buffers'):
        value = plain.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise RuntimeError(f'zero-sample renderer did not prove {field}=0')
    if (plain.get('color_storage_policy') !=
            'ordinary_GL_RENDERBUFFER_GL_RGBA_via_glRenderbufferStorage' or
            plain.get('depth_storage_policy') !=
            'ordinary_GL_RENDERBUFFER_GL_DEPTH_COMPONENT24_via_glRenderbufferStorage'):
        raise RuntimeError('zero-sample renderer attachment storage policy is not the pinned ordinary policy')
    generation = plain.get('allocation_generation')
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise RuntimeError('zero-sample renderer allocation generation is missing or invalid')
    offscreen_identity = plain.get('offscreen_identity')
    if type(offscreen_identity) is not int or offscreen_identity <= 0:
        raise RendererIntegrityError('zero-sample renderer offscreen identity is missing or invalid')
    fields = plain.get('framebuffer_fields')
    if not isinstance(fields, dict):
        raise RuntimeError('zero-sample renderer framebuffer identity is missing')
    framebuffers = (fields.get('multisample_draw_fbo'), fields.get('single_sample_read_fbo'))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in framebuffers) or framebuffers[0] == framebuffers[1]:
        raise RuntimeError('zero-sample renderer framebuffer identities are invalid')
    if fields.get('multisample_dimensions') != list(expected_dimensions):
        raise RuntimeError('zero-sample renderer framebuffer viewport differs from the crop')
    allocation = plain.get('allocation_pair')
    if not isinstance(allocation, dict) or allocation.get('passed') is not True:
        raise RuntimeError('zero-sample renderer has no validated ordinary attachment pair')
    if allocation.get('dimensions') != list(expected_dimensions):
        raise RuntimeError('zero-sample attachment pair dimensions differ from the crop')
    allocation_calls = plain.get('allocation_calls')
    if not isinstance(allocation_calls, list):
        raise RuntimeError('zero-sample renderer allocation lifecycle is missing')
    if previous is None:
        if generation != 1 or len(allocation_calls) != 2:
            raise RuntimeError('fresh zero-sample renderer did not report one verified allocation pair')
        from .quality_render_stability import validate_capture_allocation_pairs

        try:
            validated_pair = validate_capture_allocation_pairs(
                allocation_calls, dimensions=expected_dimensions)
        except (TypeError, ValueError) as exc:
            raise RendererIntegrityError(
                f'fresh zero-sample allocation rows fail the pinned validator: {exc}') from exc
        if allocation != validated_pair:
            raise RendererIntegrityError(
                'zero-sample allocation_pair does not exactly join the validated color/depth rows')
    else:
        if (generation != previous['allocation_generation'] or allocation_calls or
                offscreen_identity != previous['offscreen_identity'] or
                fields != previous['framebuffer_fields'] or
                allocation != previous['allocation_pair']):
            raise RuntimeError('zero-sample renderer reuse changed its verified allocation lifecycle')
    source_identities = plain.get('source_identities')
    expected_modules = {
        'quality_gotrack': 'bench.quality_gotrack',
        'quality_render_stability': 'bench.quality_render_stability',
        'pyrender_renderer': 'pyrender.renderer',
        'pyrender_offscreen': 'pyrender.offscreen',
    }
    if not isinstance(source_identities, dict) or set(source_identities) != set(expected_modules):
        raise RuntimeError('zero-sample renderer source identity inventory is incomplete')
    for name, expected_module in expected_modules.items():
        row = source_identities.get(name)
        module = sys.modules.get(expected_module)
        if not isinstance(row, dict) or module is None:
            raise RuntimeError(f'zero-sample renderer source identity is missing: {name}')
        module_path = getattr(module, '__file__', None)
        if (row.get('module') != expected_module or not isinstance(module_path, str) or
                Path(module_path).resolve() != Path(str(row.get('file'))).resolve()):
            raise RuntimeError(f'zero-sample renderer source path does not match {name}')
        source_digest = row.get('sha256')
        if (not isinstance(source_digest, str) or len(source_digest) != 64 or
                source_digest.upper() != source_digest or
                any(character not in '0123456789ABCDEF' for character in source_digest)):
            raise RuntimeError(f'zero-sample renderer source digest is invalid: {name}')
        source_path = Path(str(row['file']))
        if not source_path.is_file() or digest(source_path).upper() != source_digest:
            raise RuntimeError(f'zero-sample renderer source digest is stale: {name}')
    if previous is not None and source_identities != previous['source_identities']:
        raise RendererIntegrityError('zero-sample renderer source identity changed during the branch')
    return plain


class LegacyDepthAblationRenderer:
    """Dual public renderer with common support and legacy outer coordinates."""

    coordinate_mode = _LEGACY_COORDINATE_MODE
    render_policy = 'legacy_depth_ablation_v1'

    def __init__(self, path, obj_id, branch):
        if branch not in ('control', 'candidate'):
            raise ValueError('branch must be control or candidate')
        from .quality_gotrack import TexturedRenderer

        self.branch = branch
        self.legacy = None
        self.zero = None
        self._last_zero_metadata = None
        self._previous_zero_metadata = None
        self._integrity_error = None
        self.render_records = []
        self._closed = False
        primary = None
        try:
            self.legacy = TexturedRenderer(
                path, obj_id, unlit=True, disable_multisampling=True,
                coordinate_mode=_LEGACY_COORDINATE_MODE,
                render_policy=_LEGACY_RENDER_POLICY)
            self.zero = TexturedRenderer(
                path, obj_id, unlit=True, disable_multisampling=False,
                coordinate_mode=_CAPTURE_COORDINATE_MODE,
                render_policy=_CAPTURE_RENDER_POLICY)
            if (self.legacy.obj_id != self.zero.obj_id or
                    self.legacy.asset_sha256 != self.zero.asset_sha256 or
                    not np.array_equal(self.legacy.vertices_m, self.zero.vertices_m)):
                raise RuntimeError('paired renderers do not own the same object asset and metric vertices')
            self.obj_id = self.legacy.obj_id
            self.vertices_m = self.legacy.vertices_m
            self.asset_sha256 = self.legacy.asset_sha256
        except BaseException as exc:
            primary = exc
        if primary is not None:
            failures = self._close_created()
            if failures:
                detail = '; '.join(f'{type(error).__name__}: {error}' for error in failures)
                raise RendererIntegrityError(
                    f'paired renderer construction failed ({primary}); cleanup failed ({detail})') from primary
            raise RendererIntegrityError(
                f'paired renderer construction failed: {type(primary).__name__}: {primary}') from primary

    @property
    def render_policy_metadata(self):
        return self._last_zero_metadata

    def assert_healthy(self):
        if self._integrity_error is not None:
            raise RendererIntegrityError(
                f'paired renderer integrity failure is latched: {self._integrity_error}')
        if self._closed:
            raise RendererIntegrityError('paired renderer was closed before the branch completed')

    def _latch_integrity_error(self, message):
        self._integrity_error = str(message)
        self._last_zero_metadata = None
        raise RendererIntegrityError(self._integrity_error)

    @staticmethod
    def _cpu_array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _validate_camera(camera):
        width, height = camera.width, camera.height
        if (type(width) is not int or type(height) is not int or
                (width, height) != (_WIDTH, _HEIGHT)):
            raise ValueError('depth ablation renderer accepts only an actual 280x280 crop')
        f = np.asarray(camera.f, dtype=np.float64)
        c = np.asarray(camera.c, dtype=np.float64)
        pose = np.asarray(camera.T_world_from_eye, dtype=np.float64)
        if (f.shape != (2,) or c.shape != (2,) or pose.shape != (4, 4) or
                not np.isfinite(f).all() or not np.isfinite(c).all() or
                not np.isfinite(pose).all() or np.any(f <= 0.)):
            raise ValueError('legacy camera must have finite 280x280 millimetre pose and positive focal values')
        if not np.allclose(pose[3], (0., 0., 0., 1.), atol=1e-10, rtol=0.):
            raise ValueError('legacy camera pose must be a finite rigid-transform matrix')
        return f.copy(), c.copy(), pose.copy()

    @staticmethod
    def _camera_with_principal(camera, principal):
        if not np.array_equal(np.asarray(principal, dtype=np.float64),
                              np.asarray(camera.c, dtype=np.float64) - .5):
            raise ValueError('inner principal must preserve the legacy half-center ray')
        return legacy_inner_camera(camera)

    def render_object_model(self, obj_id, camera_model_c2w, render_types=None,
                            return_tensors=False, background=None, **kwargs):
        self.assert_healthy()
        if kwargs:
            raise TypeError(f'unexpected renderer arguments: {sorted(kwargs)}')
        if obj_id != self.obj_id:
            raise ValueError('wrong object asset for paired renderer')
        if type(return_tensors) is not bool:
            raise TypeError('return_tensors must be bool')
        from utils import renderer_base

        try:
            f, c, pose = self._validate_camera(camera_model_c2w)
        except BaseException as exc:
            self._latch_integrity_error(
                f'legacy crop camera failed size/finite/unit preflight: {type(exc).__name__}: {exc}')
        if render_types is not None and not isinstance(render_types, (tuple, list, set, frozenset)):
            raise TypeError('render_types must be a renderer type collection or None')
        before = _camera_pin(camera_model_c2w)
        inner_camera = self._camera_with_principal(camera_model_c2w, c - .5)
        if not np.array_equal(np.asarray(inner_camera.T_world_from_eye), pose):
            raise RuntimeError('fresh zero-sample camera changed the legacy outer pose')

        # Run both public APIs at every seed.  The legacy pass owns RGB in both
        # branches; the verified pass contributes only its candidate depth.
        try:
            legacy_render = self.legacy.render_object_model(
                obj_id, camera_model_c2w, return_tensors=False, background=background)
            zero_render = self.zero.render_object_model(
                obj_id, inner_camera, return_tensors=False, background=background)
            metadata = _validate_zero_metadata(
                self.zero.render_policy_metadata,
                expected_dimensions=(_WIDTH, _HEIGHT),
                previous=self._previous_zero_metadata)
        except RendererIntegrityError as exc:
            self._integrity_error = str(exc)
            self._last_zero_metadata = None
            raise
        except BaseException as exc:
            self._integrity_error = (
                f'paired native renderer call or zero-sample proof failed: {type(exc).__name__}: {exc}')
            self._last_zero_metadata = None
            raise RendererIntegrityError(self._integrity_error) from exc
        self._previous_zero_metadata = metadata
        self._last_zero_metadata = metadata

        color_key = renderer_base.RenderType.COLOR
        depth_key = renderer_base.RenderType.DEPTH
        mask_key = renderer_base.RenderType.MASK
        try:
            color = self._cpu_array(legacy_render[color_key])
            legacy_depth = self._cpu_array(legacy_render[depth_key])
            zero_depth = self._cpu_array(zero_render[depth_key])
            legacy_mask = self._cpu_array(legacy_render[mask_key]).astype(bool, copy=False)
            zero_mask = self._cpu_array(zero_render[mask_key]).astype(bool, copy=False)
        except BaseException as exc:
            self._integrity_error = f'paired renderer result layout is unavailable: {type(exc).__name__}: {exc}'
            self._last_zero_metadata = None
            raise RendererIntegrityError(self._integrity_error) from exc
        expected_color = (_HEIGHT, _WIDTH, 3)
        expected_map = (_HEIGHT, _WIDTH)
        if color.shape != expected_color or color.dtype.kind not in 'fiu':
            self._latch_integrity_error('legacy renderer RGB has the wrong public layout')
        if (legacy_depth.shape != expected_map or zero_depth.shape != expected_map or
                legacy_depth.dtype.kind not in 'fiu' or zero_depth.dtype.kind not in 'fiu' or
                legacy_mask.shape != expected_map or zero_mask.shape != expected_map):
            self._latch_integrity_error('paired renderer depth/mask has the wrong public layout')
        if (not np.isfinite(color).all() or not np.isfinite(legacy_depth).all() or
                not np.isfinite(zero_depth).all() or np.any(legacy_depth < 0.) or
                np.any(zero_depth < 0.) or not np.array_equal(legacy_mask, legacy_depth > 0.) or
                not np.array_equal(zero_mask, zero_depth > 0.)):
            self._latch_integrity_error('paired renderer returned invalid millimetre depth or mask values')
        common = legacy_mask & zero_mask
        selected_control = np.where(common, legacy_depth, 0.).astype(np.float32, copy=False)
        selected_candidate = np.where(common, zero_depth, 0.).astype(np.float32, copy=False)
        selected_depth = selected_control if self.branch == 'control' else selected_candidate
        selected_color = np.asarray(color, dtype=np.float32)
        if (not np.isfinite(selected_color).all() or
                float(selected_color.min()) < 0. or float(selected_color.max()) > 1.):
            self._latch_integrity_error('legacy renderer RGB is outside its normalized public range')
        if not np.array_equal(_camera_pin(camera_model_c2w), before):
            self._latch_integrity_error('outer GoTrack camera was mutated by the dual render')

        record = {
            'render_ordinal': len(self.render_records),
            'branch': self.branch,
            'outer_camera': before,
            'inner_camera': _camera_pin(inner_camera),
            'inner_principal_rule': 'c_inner=c_legacy-(0.5,0.5)',
            'zero_renderer_metadata': metadata,
            'legacy_template_pixels': int(np.count_nonzero(legacy_mask)),
            'zero_sample_template_pixels': int(np.count_nonzero(zero_mask)),
            'common_template_pixels': int(np.count_nonzero(common)),
            'removed_template_pixels': int(np.count_nonzero(legacy_mask & ~common)),
            'removed_fraction_of_legacy_template': float(
                np.count_nonzero(legacy_mask & ~common) / max(1, np.count_nonzero(legacy_mask))),
            'legacy_rgb_sha256': _array_digest(selected_color.astype('<f4', copy=False)),
            'common_mask_sha256': _array_digest(common.astype(np.uint8, copy=False)),
        }
        self.render_records.append(record)
        # The pinned renderer returns all three RenderType entries even when
        # data_util asks for a subset. Keep that adapter layout.
        result = {color_key: selected_color.copy(),
                  depth_key: selected_depth.copy(),
                  mask_key: common.copy()}
        if return_tensors:
            try:
                import torch
                result = {key: torch.from_numpy(np.array(value, copy=True))
                          for key, value in result.items()}
            except BaseException as exc:
                self._integrity_error = f'paired renderer tensor conversion failed: {type(exc).__name__}: {exc}'
                self._last_zero_metadata = None
                raise RendererIntegrityError(self._integrity_error) from exc
        return result

    def _close_created(self):
        failures = []
        for renderer in (self.legacy, self.zero):
            if renderer is None:
                continue
            try:
                renderer.close()
            except BaseException as exc:
                failures.append(exc)
        return failures

    def close(self):
        if self._closed:
            return
        self._closed = True
        failures = self._close_created()
        if failures:
            detail = '; '.join(f'{type(error).__name__}: {error}' for error in failures)
            self._integrity_error = f'paired renderer cleanup failed: {detail}'
            raise RendererIntegrityError(self._integrity_error) from failures[0]


class _NetworkRecorder:
    """Transparent one-call recorder for the shared-seed first-iteration gate."""

    def __init__(self, network):
        self.network = network
        self.backbone = network.backbone
        self.calls = []

    def __call__(self, query_rgb, template_rgb, template_mask):
        flow, confidence = self.network(query_rgb, template_rgb, template_mask)
        self.calls.append({
            'query_rgb': _array_digest(_to_numpy(query_rgb).astype('<f4', copy=False)),
            'template_rgb': _array_digest(_to_numpy(template_rgb).astype('<f4', copy=False)),
            'template_mask': _array_digest(_to_numpy(template_mask)),
            'flow': _array_digest(_to_numpy(flow).astype('<f4', copy=False)),
            'confidence': _array_digest(_to_numpy(confidence).astype('<f4', copy=False)),
            'query_shape': list(getattr(query_rgb, 'shape', ())),
            'template_shape': list(getattr(template_rgb, 'shape', ())),
            'mask_shape': list(getattr(template_mask, 'shape', ())),
        })
        return flow, confidence


def _to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _diagnostic_recorder():
    return []


class _StopAfterFirstRetained(Exception):
    pass


def _capture_retained_probe(rows):
    def callback(packet):
        if packet.get('event', {}).get('stage') != 'retained':
            return
        arrays = packet.get('arrays', {})
        rows.append({
            'event': _plain(packet.get('event', {})),
            'sample_ids_sha256': _array_digest(arrays.get('sample_ids', np.empty(0, dtype=np.int64))),
            'sample_obj_points_sha256': _array_digest(arrays.get('sample_obj_points_mm', np.empty((0, 3)))),
            'sample_target_sha256': _array_digest(arrays.get('sample_target_crop_px', np.empty((0, 2)))),
            'crop_k_sha256': _array_digest(arrays.get('crop_k', np.empty((0, 0)))),
            'crop_from_orig_sha256': _array_digest(arrays.get('crop_from_orig', np.empty((0, 0)))),
            'sample_count': int(len(arrays.get('sample_ids', ()))),
        })
        # The legacy adapter calls this diagnostic hook immediately before
        # RANSAC. Stop here so the probe compares the first shared iteration;
        # branch-specific poses can then evolve only in the actual trajectories.
        raise _StopAfterFirstRetained()
    return callback


def _run_shared_seed_probe(setup_frame, setup_mask, network, asset_path, object_id,
                           device, pnp_settings, bank):
    """Compare both branches for one FoundPose-generated setup seed only."""
    from .quality_foundpose import FoundPoseRecovery
    from .quality_runner import _make_gotrack_refiner

    control_renderer = None
    candidate_renderer = None
    control_network = _NetworkRecorder(network)
    candidate_network = _NetworkRecorder(network)
    control_retained = _diagnostic_recorder()
    candidate_retained = _diagnostic_recorder()
    failures = []
    probe = None
    try:
        control_renderer = LegacyDepthAblationRenderer(asset_path, object_id, 'control')
        candidate_renderer = LegacyDepthAblationRenderer(asset_path, object_id, 'candidate')
        control_refiner = _make_gotrack_refiner(
            control_network, control_renderer, object_id, device, pnp_settings)
        candidate_refiner = _make_gotrack_refiner(
            candidate_network, candidate_renderer, object_id, device, pnp_settings)
        seed_backend = FoundPoseRecovery(control_refiner, bank, model_memory=False)
        proposals = seed_backend.recover(setup_frame, setup_mask, top_k=5)
        if not proposals:
            raise RuntimeError('automatic setup-mask FoundPose produced no shared-seed probe candidate')
        seed = np.asarray(proposals[0], dtype=np.float64).copy()
        control_refiner.diagnostic_capture_callback = _capture_retained_probe(control_retained)
        candidate_refiner.diagnostic_capture_callback = _capture_retained_probe(candidate_retained)
        for renderer, refiner in ((control_renderer, control_refiner),
                                  (candidate_renderer, candidate_refiner)):
            renderer.assert_healthy()
            try:
                refiner.refine(setup_frame, setup_mask, seed.copy())
            except _StopAfterFirstRetained:
                pass
            renderer.assert_healthy()
        if len(control_network.calls) != 1 or len(candidate_network.calls) != 1:
            raise RuntimeError('shared-seed probe did not stop after exactly one GoTrack iteration')
        if len(control_retained) != 1 or len(candidate_retained) != 1:
            raise RuntimeError('shared-seed probe did not capture the pre-PnP retained records')
        differences = []
        for index, (left, right) in enumerate(zip(control_network.calls, candidate_network.calls)):
            for field in ('query_rgb', 'template_rgb', 'template_mask', 'flow', 'confidence',
                          'query_shape', 'template_shape', 'mask_shape'):
                if left[field] != right[field]:
                    differences.append(f'iteration {index}: {field}')
        for index, (left, right) in enumerate(zip(control_retained, candidate_retained)):
            for field in ('sample_ids_sha256', 'sample_target_sha256', 'crop_k_sha256',
                          'crop_from_orig_sha256', 'sample_count'):
                if left[field] != right[field]:
                    differences.append(f'iteration {index}: retained {field}')
            # The sampled 3D template points are the intended depth-mediated
            # difference, so they are retained as paired evidence only.
        control_render = control_renderer.render_records[0]
        candidate_render = candidate_renderer.render_records[0]
        for field in ('outer_camera', 'inner_camera', 'legacy_rgb_sha256',
                      'common_mask_sha256', 'common_template_pixels'):
            if control_render[field] != candidate_render[field]:
                differences.append(f'paired render: {field}')
        if differences:
            raise RendererIntegrityError(
                'shared-seed first-iteration parity gate failed: ' + ', '.join(differences[:12]))
        probe = {
            'status': 'passed',
            'seed_source': 'automatic setup-mask FoundPose top-ranked proposal; probe-only',
            'seed_sha256': _array_digest(seed.astype('<f8', copy=False)),
            'frame_id': int(setup_frame.frame_id),
            'iterations_compared': 1,
            'control_network_calls': control_network.calls,
            'candidate_network_calls': candidate_network.calls,
            'control_retained': control_retained,
            'candidate_retained': candidate_retained,
            'depth_point_inputs_differ': [
                left['sample_obj_points_sha256'] != right['sample_obj_points_sha256']
                for left, right in zip(control_retained, candidate_retained)],
            'control_render_records': control_renderer.render_records,
            'candidate_render_records': candidate_renderer.render_records,
        }
    finally:
        for renderer in (control_renderer, candidate_renderer):
            if renderer is None:
                continue
            try:
                renderer.close()
            except BaseException as exc:
                failures.append(exc)
    if failures:
        if probe is None:
            raise RuntimeError('shared-seed renderer cleanup failed: ' + '; '.join(str(e) for e in failures)) from failures[0]
        probe['status'] = 'failed_cleanup'
        probe['cleanup_errors'] = [f'{type(error).__name__}: {error}' for error in failures]
    return probe


def _claim_output(path, expected_name):
    path = Path(path).resolve()
    root = _OUTPUT_ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError('output must remain inside the fixed legacy-depth-v1 diagnostic directory') from exc
    if path.name != expected_name:
        raise ValueError(f'this command requires output filename {expected_name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = path.with_name(path.name + '.budget-failure.json')
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    summary_temporary_path = summary_path.with_suffix(summary_path.suffix + '.tmp')
    if summary_path.exists() or temporary_path.exists() or summary_temporary_path.exists():
        raise FileExistsError('refusing to overwrite a prior budget summary or report temporary file')
    if path.exists():
        raise FileExistsError(f'refusing to overwrite previous trial output: {path}')
    marker = path.with_name(path.name + '.running')
    payload = json.dumps({'run_id': uuid.uuid4().hex, 'created_utc': time.time()}, separators=(',', ':'))
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f'output has an existing run marker: {marker}') from exc
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        marker.unlink(missing_ok=True)
        raise FileExistsError(f'refusing to overwrite previous trial output: {path}')
    return path, marker


def _tracking_budget_context(branch, limit, frames, phase):
    return {
        'experiment': 'legacy_depth_ablation_v1',
        'branch': branch,
        'phase': phase,
        'frame_ids': [row['frameId'] for row in frames],
        'unattempted_frame_ids': list(_FULL_IDS[:limit][len(frames):]),
    }


def _frame_hash(path):
    value = digest(path).upper()
    if len(value) != 64:
        raise RuntimeError(f'bad SHA-256 while pinning {Path(path).name}')
    return value


def _load_fixed_inputs():
    from . import quality_runner
    from .quality_time import LEGACY

    manifest = quality_runner.read_input(_BUNDLE)
    if manifest.get('_capture') is not None:
        raise ValueError('capture inputs are forbidden for this legacy-only experiment')
    timing = manifest.get('_timing', {})
    if timing.get('clock_mode') != LEGACY:
        raise ValueError('physical or unknown clock mode is forbidden')
    if (manifest.get('object') != 'ranch' or manifest.get('object_id') != 15 or
            manifest.get('setup_frame_id') != _SETUP_ID or
            tuple(manifest.get('frame_ids', ())) != _FULL_IDS):
        raise ValueError('only original ranch setup frame 9 and scored source frames 10–249 are allowed')
    if manifest.get('units') != 'metres':
        raise ValueError('the pinned metric ranch asset is required')
    mask_manifest_path = _MASKS / 'results.json'
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding='utf-8'))
    if (mask_manifest.get('diagnostic_control') is True or
            mask_manifest.get('mask_association') is True):
        raise ValueError('diagnostic or associated segmentation masks are forbidden')
    if mask_manifest.get('provenance', {}).get('input_manifest_sha256', '').upper() != _frame_hash(_BUNDLE / 'input.json'):
        raise ValueError('original ranch segmentation cache is not bound to this input manifest')
    frame_rows = mask_manifest.get('frames', [])
    if not isinstance(frame_rows, list):
        raise ValueError('original ranch segmentation frame index is malformed')
    records = {row.get('frameId'): row for row in frame_rows}
    expected_mask_ids = {_SETUP_ID, *_FULL_IDS}
    if (set(records) != expected_mask_ids or len(records) != len(expected_mask_ids) or
            len(frame_rows) != len(expected_mask_ids)):
        raise ValueError('original ranch segmentation cache must include setup9 and every frame10–249')
    for frame_id in sorted(expected_mask_ids):
        row = records[frame_id]
        if row.get('mask_state') != 'available' or not isinstance(row.get('path'), str):
            raise ValueError(f'original segmentation mask {frame_id} is not available')
        mask_path = (_MASKS / row['path']).resolve()
        try:
            mask_path.relative_to(_MASKS.resolve())
        except ValueError as exc:
            raise ValueError('segmentation mask path escapes the pinned ranch cache') from exc
        if not mask_path.is_file():
            raise ValueError(f'original segmentation mask is missing for frame {frame_id}')
    return quality_runner, manifest, mask_manifest, records


def _collect_pins(manifest, mask_manifest, mask_records, limit):
    relevant_ids = sorted({_SETUP_ID, *_FULL_IDS[:limit]})
    adapter_names = ('quality_assets', 'quality_contract', 'quality_time', 'quality_runner',
                     'quality_gotrack', 'quality_foundpose', 'quality_appearance',
                     'quality_render_stability', 'vision')
    return {
        'input_manifest': str(_BUNDLE / 'input.json'),
        'input_manifest_sha256': _frame_hash(_BUNDLE / 'input.json'),
        'video_sha256': _frame_hash(_BUNDLE / manifest['video']),
        'asset_sha256': _frame_hash(_BUNDLE / manifest['asset']),
        'foundpose_bank_path': str(_BANK),
        'foundpose_bank_sha256': _frame_hash(_BANK),
        'mask_index_sha256': _frame_hash(_MASKS / 'results.json'),
        'masks': {
            str(frame_id): {
                'path': mask_records[frame_id]['path'],
                'sha256': _frame_hash(_MASKS / mask_records[frame_id]['path']),
            }
            for frame_id in relevant_ids
        },
        'adapter_sha256': {
            name: _frame_hash(Path(__file__).resolve().parents[1] / 'bench' / f'{name}.py')
            for name in adapter_names
        },
        'experiment_module_sha256': _frame_hash(Path(__file__).resolve()),
        'legacy_clock_mode': 'legacy_nominal_60_v1',
        'nominal_fps': 60,
        'mask_cache_provenance': {
            'model': mask_manifest.get('model'),
            'device': mask_manifest.get('device'),
            'propagation': mask_manifest.get('propagation'),
            'diagnostic_control': bool(mask_manifest.get('diagnostic_control', False)),
            'mask_association': bool(mask_manifest.get('mask_association', False)),
        },
    }


def _load_mask(record):
    from .vision import cv2
    mask = cv2.imread(str(_MASKS / record['path']), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2 or not np.any(mask):
        raise ValueError(f'original segmentation mask cannot be decoded for frame {record.get("frameId")}')
    return mask


def _state_summary(frames, limit):
    from collections import Counter

    state_counts = Counter(row.get('pose_state', 'missing') for row in frames)
    mask_counts = Counter(row.get('mask_state', 'missing') for row in frames)
    render_counts = Counter(row.get('render_state', 'missing') for row in frames)
    final_reasons = Counter(row.get('failure_reason') for row in frames if row.get('failure_reason'))
    candidate_rejections = Counter(
        reason for row in frames for reason in row.get('rejection_reasons', ()))
    intervals = []
    active = None
    for row in frames:
        frame_id = row.get('frameId')
        failed = row.get('pose_state') != 'tracking'
        if failed and active is None:
            active = {'start': frame_id, 'end': frame_id, 'length': 1}
        elif failed:
            active['end'] = frame_id
            active['length'] += 1
        elif active is not None:
            intervals.append(active)
            active = None
    if active is not None:
        intervals.append(active)
    recovering = [row for row in frames if row.get('pose_state') == 'recovering']
    recovery_groups = []
    active_recovery = None
    for row in frames:
        frame_id = row.get('frameId')
        if row.get('pose_state') == 'recovering':
            if active_recovery is None:
                active_recovery = {'start_frame_id': frame_id, 'last_recovering_frame_id': frame_id}
            else:
                active_recovery['last_recovering_frame_id'] = frame_id
        elif active_recovery is not None:
            active_recovery['confirmed_frame_id'] = frame_id if row.get('pose_state') == 'tracking' else None
            active_recovery['confirmation_delay_frames'] = (
                None if active_recovery['confirmed_frame_id'] is None else
                active_recovery['confirmed_frame_id'] - active_recovery['start_frame_id'])
            recovery_groups.append(active_recovery)
            active_recovery = None
    if active_recovery is not None:
        active_recovery.update(confirmed_frame_id=None, confirmation_delay_frames=None)
        recovery_groups.append(active_recovery)
    disagreement_ids = {209, 211, 217, 218, 221, 222, 224, 225, 226, 227}
    tail_rows = [row for row in frames if 186 <= row.get('frameId', -1) <= 249]
    disagreement_rows = [row for row in frames if row.get('frameId') in disagreement_ids]
    return {
        'denominator_requested_frames': int(limit),
        'rows_written': len(frames),
        'tracking_lost': int(state_counts.get('lost', 0)),
        'recovering': int(state_counts.get('recovering', 0)),
        'tracking': int(state_counts.get('tracking', 0)),
        'pose_state_counts': dict(state_counts),
        'mask_state_counts': dict(mask_counts),
        'render_state_counts': dict(render_counts),
        'candidate_rejection_counts': dict(candidate_rejections),
        'final_failure_reason_counts': dict(final_reasons),
        'nontracking_intervals_in_written_prefix': intervals,
        'recovering_source_ids': [int(row['frameId']) for row in recovering],
        'recovery_confirmation_delays': recovery_groups,
        'late_186_249': {
            'rows_written': len(tail_rows),
            'tracking': sum(row.get('pose_state') == 'tracking' for row in tail_rows),
            'recovering': sum(row.get('pose_state') == 'recovering' for row in tail_rows),
            'lost': sum(row.get('pose_state') == 'lost' for row in tail_rows),
            'large_disagreement_ids_state_rows': [
                {'frameId': int(row['frameId']), 'pose_state': row.get('pose_state'),
                 'failure_reason': row.get('failure_reason')}
                for row in disagreement_rows],
        },
    }


def _build_report(*, branch, manifest, mask_manifest, mask_records, pins,
                  output, device, limit, probe, provenance, tracker, initialization,
                  frames, renderer, appearance_settings, status, error=None,
                  unattempted_ids=()):
    from .quality_runner import _pnp_tracking_settings

    report = {
        'schema_version': 1,
        'experiment': 'legacy_depth_ablation_v1',
        'branch': branch,
        'object': 'ranch',
        'source_setup_frame_id': _SETUP_ID,
        'frame_ids': list(_FULL_IDS[:limit]),
        'complete': bool(status == 'complete' and len(frames) == limit),
        'status': status,
        'frames': frames,
        'unattempted_frame_ids': list(unattempted_ids),
        'state_summary': _state_summary(frames, limit),
        'candidate_rejections': [] if tracker is None else list(tracker.rejections),
        'initialization': initialization,
        'shared_seed_probe': probe,
        'provenance': provenance,
        'experiment_pins': {
            **pins,
            'legacy_clock_mode': 'legacy_nominal_60_v1',
            'render_policy': 'legacy RGB and common-support paired model-template depth',
            'branch_depth': 'legacy MSAA depth' if branch == 'control' else 'verified zero-sample depth under legacy half-center unprojection',
            'observed_foreground_policy': 'unchanged original current-image segmentation masks',
            'pnp_settings': _pnp_tracking_settings('ranch', False),
            'model_memory': False,
            'appearance_check': True,
            'appearance_settings': asdict(appearance_settings),
            'unlit_templates': True,
            'disable_multisampling_for_legacy_color': True,
            'new_report_budget_bytes': _REPORT_BYTE_LIMIT,
            'neural_load': {'batch_size': 1, 'concurrent_gpu_stages': 1,
                            'segmentation_resident': False},
            'runtime': _plain(provenance),
            'render_counts': {
                'paired_calls': 0 if renderer is None else len(renderer.render_records),
                'legacy_template_pixels': [] if renderer is None else [row['legacy_template_pixels'] for row in renderer.render_records],
                'zero_sample_template_pixels': [] if renderer is None else [row['zero_sample_template_pixels'] for row in renderer.render_records],
                'common_template_pixels': [] if renderer is None else [row['common_template_pixels'] for row in renderer.render_records],
                'removed_template_pixels': [] if renderer is None else [row['removed_template_pixels'] for row in renderer.render_records],
                'render_records': [] if renderer is None else renderer.render_records,
            },
            'error': error,
        },
        'independent_accuracy_scored': False,
        'independent_labels_reviewed': 0,
        'secondary_reference_accuracy': None,
    }
    return report


def pose(branch, output, *, device='cuda', prefix_120=False):
    """Run a fresh automatic legacy chronology, fixed to ranch and its mask bank."""
    if branch not in ('control', 'candidate'):
        raise ValueError('branch must be control or candidate')
    limit = 120 if prefix_120 else 240
    if limit not in _PREFIX_LENGTHS:
        raise ValueError('only the exact 120-frame prefix or full 240-frame window is supported')
    expected_name = f'{branch}-{limit}.json'
    quality_runner, manifest, mask_manifest, mask_records = _load_fixed_inputs()
    pins = _collect_pins(manifest, mask_manifest, mask_records, limit)
    output, marker = _claim_output(output, expected_name)
    from .quality_contract import Frame, MaskPrediction, SequentialTracker
    from .quality_runner import _make_gotrack_refiner, _pnp_tracking_settings
    from .quality_gotrack import load_network, TexturedRenderer
    from .quality_foundpose import FoundPoseRecovery
    from .quality_appearance import AppearanceCheckedBackend, AppearanceSettings
    import torch

    pnp_settings = _pnp_tracking_settings('ranch', False)
    network = None
    renderer = None
    appearance_renderer = None
    frames = []
    initialization = None
    tracker = None
    provenance = None
    error = None
    status = 'incomplete'
    probe = None
    last_report = None
    appearance_settings = AppearanceSettings()
    try:
        network = load_network(device)
        bank = torch.load(_BANK, map_location='cpu', weights_only=False)
        renderer = LegacyDepthAblationRenderer(_BUNDLE / manifest['asset'], manifest['object_id'], branch)
        appearance_renderer = TexturedRenderer(
            _BUNDLE / manifest['asset'], manifest['object_id'],
            unlit=True, disable_multisampling=True)
        refiner = _make_gotrack_refiner(network, renderer, manifest['object_id'], device, pnp_settings)
        backend = FoundPoseRecovery(refiner, bank, model_memory=False)
        backend = AppearanceCheckedBackend(backend, appearance_renderer, appearance_settings)
        tracker = SequentialTracker(backend)
        provenance = inference_provenance(_BUNDLE)
        provenance['foundpose_bank_sha256'] = _frame_hash(_BANK)

        k = np.asarray(manifest['intrinsics'], dtype=np.float64)
        setup_record = mask_records[_SETUP_ID]
        setup_mask = _load_mask(setup_record)
        setup_frame = Frame(_SETUP_ID, quality_runner.read_rgb(_BUNDLE, manifest, _SETUP_ID), k)
        setup_segmentation = MaskPrediction(
            _SETUP_ID, setup_mask, setup_record['mask_state'],
            setup_record.get('failure_reason'), setup_record.get('timings_ms', {}))

        # The seed is generated automatically from the current setup mask and
        # existing FoundPose bank, and used only for this shared-seed parity gate.
        probe = _run_shared_seed_probe(
            setup_frame, setup_mask, network, _BUNDLE / manifest['asset'],
            manifest['object_id'], device, pnp_settings, bank)
        if probe is None or probe.get('status') != 'passed':
            raise RendererIntegrityError('shared-seed first-iteration mechanism gate did not pass')

        renderer.assert_healthy()
        initialization = tracker.update(setup_frame, setup_segmentation)
        renderer.assert_healthy()
        ids = list(_FULL_IDS[:limit])
        for ordinal, frame_id in enumerate(ids):
            record = mask_records[frame_id]
            rgb = quality_runner.read_rgb(_BUNDLE, manifest, frame_id)
            mask = _load_mask(record)
            segmentation = MaskPrediction(
                frame_id, mask, record['mask_state'], record.get('failure_reason'),
                record.get('timings_ms', {}))
            frame = Frame(frame_id, rgb, k)
            try:
                renderer.assert_healthy()
                row = tracker.update(frame, segmentation)
                renderer.assert_healthy()
            except RendererIntegrityError:
                raise
            except (RuntimeError, ValueError, quality_runner.cv2.error) as exc:
                if device == 'cuda' and ('out of memory' in str(exc).lower() or 'cuda' in str(exc).lower()):
                    raise RuntimeError(f'GPU execution failed during this fixed stage: {exc}') from exc
                tracker.accepted = None
                tracker.pending = None
                row = {
                    'frameId': frame_id, 'cameraFromObject': None,
                    'mask_state': segmentation.state, 'pose_state': 'lost',
                    'render_state': 'suppressed',
                    'failure_reason': f'pose_runtime_failure: {type(exc).__name__}: {exc}',
                    'timings_ms': dict(segmentation.timings_ms),
                }
            row['mask_path'] = record['path']
            row['mask_sha256'] = _frame_hash(_MASKS / record['path'])
            frames.append(row)
            report = _build_report(
                branch=branch, manifest=manifest, mask_manifest=mask_manifest,
                mask_records=mask_records, pins=pins,
                output=output, device=device, limit=limit, probe=probe,
                provenance=provenance, tracker=tracker, initialization=initialization,
                frames=frames, renderer=renderer, appearance_settings=appearance_settings,
                status='incomplete',
                unattempted_ids=_FULL_IDS[:limit][len(frames):])
            _persist_report(
                output, report,
                budget_context=_tracking_budget_context(branch, limit, frames, 'progress'))
            last_report = report
            if ordinal % 40 == 0:
                print('ranch', branch, ordinal + 1, '/', limit, row.get('pose_state'), flush=True)
        status = 'complete' if len(frames) == limit else 'incomplete'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        status = ('budget_failure' if isinstance(exc, ReportBudgetError) else
                  'integrity_failure' if isinstance(exc, RendererIntegrityError) else 'incomplete')
        if tracker is not None:
            tracker.accepted = None
            tracker.pending = None
        if not isinstance(exc, ReportBudgetError):
            report = _build_report(
                branch=branch, manifest=manifest, mask_manifest=mask_manifest,
                mask_records=mask_records, pins=pins,
                output=output, device=device, limit=limit, probe=probe,
                provenance=provenance, tracker=tracker, initialization=initialization,
                frames=frames, renderer=renderer, appearance_settings=appearance_settings,
                status=status, error=error,
                unattempted_ids=_FULL_IDS[:limit][len(frames):])
            _persist_report(
                output, report,
                budget_context=_tracking_budget_context(branch, limit, frames, 'exception'))
            last_report = report
        raise
    finally:
        close_failures = []
        for resource in (renderer, appearance_renderer):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                close_failures.append(exc)
        if close_failures:
            status = 'integrity_failure'
            if last_report is None:
                last_report = _build_report(
                    branch=branch, manifest=manifest, mask_manifest=mask_manifest,
                    mask_records=mask_records, pins=pins,
                    output=output, device=device, limit=limit, probe=probe,
                    provenance=provenance, tracker=tracker, initialization=initialization,
                    frames=frames, renderer=renderer, appearance_settings=appearance_settings,
                    status=status, error='renderer cleanup failed',
                    unattempted_ids=_FULL_IDS[:limit][len(frames):])
            last_report['status'] = status
            last_report['complete'] = False
            last_report['cleanup_errors'] = [f'{type(exc).__name__}: {exc}' for exc in close_failures]
            existing_error = last_report.get('experiment_pins', {}).get('error')
            last_report.setdefault('experiment_pins', {})['error'] = (
                existing_error or 'renderer cleanup failed')
            try:
                _persist_report(
                    output, last_report,
                    budget_context=_tracking_budget_context(branch, limit, frames, 'cleanup_failure'))
            except ReportBudgetError:
                pass
        try:
            marker.write_text(json.dumps({
                'status': 'cleanup_pending' if status == 'complete' and not close_failures else status,
                'output_sha256': _frame_hash(output) if output.exists() else None,
                'cleanup_errors': [f'{type(exc).__name__}: {exc}' for exc in close_failures],
            }, indent=2), encoding='utf-8')
        except BaseException:
            pass
        if close_failures:
            raise RuntimeError('legacy-depth renderer cleanup failed: ' +
                               '; '.join(f'{type(exc).__name__}: {exc}' for exc in close_failures)) from close_failures[0]
    if status == 'complete':
        try:
            final_report = _build_report(
                branch=branch, manifest=manifest, mask_manifest=mask_manifest,
                mask_records=mask_records, pins=pins,
                output=output, device=device, limit=limit, probe=probe,
                provenance=provenance, tracker=tracker, initialization=initialization,
                frames=frames, renderer=renderer, appearance_settings=appearance_settings,
                status='complete', unattempted_ids=())
            _persist_report(
                output, final_report,
                budget_context=_tracking_budget_context(branch, limit, frames, 'final_success'))
        except BaseException:
            status = 'budget_failure'
            try:
                marker.write_text(json.dumps({
                    'status': status,
                    'output_sha256': _frame_hash(output) if output.exists() else None,
                    'cleanup_errors': [],
                }, indent=2), encoding='utf-8')
            except BaseException:
                pass
            raise
        try:
            marker.write_text(json.dumps({
                'status': 'complete',
                'output_sha256': _frame_hash(output) if output.exists() else None,
                'cleanup_errors': [],
            }, indent=2), encoding='utf-8')
        except BaseException:
            pass


def _plane_vertices():
    return np.asarray((
        (-2.0, -2.0, 1.0 + .25 * -2.0 - .15 * -2.0),
        ( 2.0, -2.0, 1.0 + .25 *  2.0 - .15 * -2.0),
        ( 2.0,  2.0, 1.0 + .25 *  2.0 - .15 *  2.0),
        (-2.0,  2.0, 1.0 + .25 * -2.0 - .15 *  2.0),
    ), dtype=np.float64)


def _plane_fixture_scene(renderer):
    """Replace scene geometry with one native slanted plane for the exact shim."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=_plane_vertices(),
                           faces=np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.int64),
                           process=False)
    for node in list(renderer.scene.mesh_nodes):
        renderer.scene.remove_node(node)
    node = renderer.scene.add(renderer.pyrender.Mesh.from_trimesh(mesh, smooth=False))
    return node


def geometry(output):
    """Run eight analytic legacy-half-center rays through the real dual wrapper."""
    from .quality_gotrack import upstream_path

    upstream_path()
    from . import quality_runner
    from .quality_runner import read_input
    from utils import structs, renderer_base

    manifest = read_input(_BUNDLE)
    if (manifest.get('_capture') is not None or
            manifest.get('_timing', {}).get('clock_mode') != 'legacy_nominal_60_v1' or
            manifest.get('object') != 'ranch' or manifest.get('object_id') != 15):
        raise ValueError('geometry command is fixed to the authenticated original legacy ranch asset')
    output, marker = _claim_output(output, 'geometry.json')
    renderer = None
    report = None
    passed = False
    try:
        renderer = LegacyDepthAblationRenderer(_BUNDLE / manifest['asset'], manifest['object_id'], 'candidate')
        _plane_fixture_scene(renderer.legacy)
        _plane_fixture_scene(renderer.zero)
        camera = structs.PinholePlaneCameraModel(
            width=_WIDTH, height=_HEIGHT,
            f=(139.5, 139.5), c=(139.5, 139.5), T_world_from_eye=np.eye(4))
        rendered = renderer.render_object_model(manifest['object_id'], camera)
        candidate_depth_m = rendered[renderer_base.RenderType.DEPTH].astype(np.float64) / 1000.
        common = rendered[renderer_base.RenderType.MASK]
        pixels = ((60, 60), (139, 60), (219, 60), (60, 139),
                  (219, 139), (60, 219), (139, 219), (219, 219))
        rows = []
        errors = []
        parity_errors = []
        fx, fy = camera.f
        cx, cy = camera.c
        for u, v in pixels:
            if not common[v, u] or not np.isfinite(candidate_depth_m[v, u]) or candidate_depth_m[v, u] <= 0.:
                raise RuntimeError(f'fixed analytic ray {(u, v)} is outside common rendered plane support')
            ray_x = (u + .5 - cx) / fx
            ray_y = (v + .5 - cy) / fy
            denominator = 1. - .25 * ray_x + .15 * ray_y
            if denominator <= 0.:
                raise RuntimeError(f'fixed analytic ray {(u, v)} meets the plane behind the camera')
            expected_z = 1. / denominator
            actual_z = float(candidate_depth_m[v, u])
            point = np.asarray((ray_x * actual_z, ray_y * actual_z, actual_z))
            projected = (point[0] / point[2] * fx + cx,
                         point[1] / point[2] * fy + cy)
            error = abs(actual_z - expected_z)
            parity = max(abs(projected[0] - (u + .5)), abs(projected[1] - (v + .5)))
            errors.append(error)
            parity_errors.append(parity)
            rows.append({
                'pixel': [u, v],
                'legacy_half_center_ray_xy': [float(ray_x), float(ray_y)],
                'expected_depth_m': float(expected_z),
                'actual_candidate_depth_m': actual_z,
                'absolute_depth_error_m': float(error),
                'reprojected_pixel_center': [float(projected[0]), float(projected[1])],
                'projection_unprojection_error_px': float(parity),
            })
        max_depth_error = max(errors)
        max_parity_error = max(parity_errors)
        metadata = renderer.render_records[-1]['zero_renderer_metadata']
        passed = (max_depth_error <= 0.00005 and max_parity_error <= 0.01 and
                  metadata.get('gl_samples') == 0 and metadata.get('gl_sample_buffers') == 0 and
                  metadata.get('framebuffer_bindings_restored') is True and
                  metadata.get('current_context_released') is True)
        report = {
            'schema_version': 1,
            'experiment': 'legacy_depth_ablation_v1_geometry_gate',
            'status': 'cleanup_pending' if passed else 'failed',
            'complete': False,
            'geometry_gate_passed': bool(passed),
            'native_renderer_calls': 2,
            'plane_fixture': {'equation': 'z - 0.25*x + 0.15*y = 1m',
                              'vertices_world_m': _plane_vertices().tolist(),
                              'mesh_vertices_sha256': _array_digest(_plane_vertices().astype('<f8', copy=False))},
            'outer_camera': _camera_pin(camera),
            'inner_camera': renderer.render_records[-1]['inner_camera'],
            'rays': rows,
            'maximum_depth_error_m': float(max_depth_error),
            'maximum_projection_unprojection_error_px': float(max_parity_error),
            'zero_sample_metadata': metadata,
            'render_record': renderer.render_records[-1],
            'current_source_provenance': inference_provenance(_BUNDLE),
            'is_tracking_accuracy_evidence': False,
        }
        if not passed:
            report['failure_reason'] = 'legacy half-center analytic depth or projection gate failed'
        _persist_report(
            output, report,
            budget_context={
                'experiment': 'legacy_depth_ablation_v1_geometry_gate',
                'phase': 'native_geometry_gate',
                'frame_ids': [], 'unattempted_frame_ids': [],
                'geometry_ray_count': len(rows),
            })
        if not passed:
            raise RuntimeError('legacy half-center analytic depth or projection gate failed')
    except BaseException as exc:
        if report is None and not isinstance(exc, ReportBudgetError):
            report = {
                'schema_version': 1,
                'experiment': 'legacy_depth_ablation_v1_geometry_gate',
                'status': 'failed',
                'complete': False,
                'failure_reason': f'{type(exc).__name__}: {exc}',
                'outer_camera': None,
                'zero_sample_metadata': None if renderer is None or not renderer.render_records else
                    renderer.render_records[-1]['zero_renderer_metadata'],
                'render_records': [] if renderer is None else renderer.render_records,
                'is_tracking_accuracy_evidence': False,
            }
            try:
                _persist_report(
                    output, report,
                    budget_context={
                        'experiment': 'legacy_depth_ablation_v1_geometry_gate',
                        'phase': 'geometry_failure',
                        'frame_ids': [], 'unattempted_frame_ids': [],
                        'geometry_ray_count': 0,
                    })
            except ReportBudgetError:
                pass
        raise
    finally:
        close_error = None
        if renderer is not None:
            try:
                renderer.close()
            except BaseException as exc:
                close_error = exc
                if report is not None:
                    report['status'] = 'failed'
                    report['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                    report['complete'] = False
                    try:
                        _persist_report(
                            output, report,
                            budget_context={
                                'experiment': 'legacy_depth_ablation_v1_geometry_gate',
                                'phase': 'geometry_cleanup_failure',
                                'frame_ids': [], 'unattempted_frame_ids': [],
                                'geometry_ray_count': len(report.get('rays', [])),
                            })
                    except ReportBudgetError:
                        pass
        marker.write_text(json.dumps({
            'status': 'cleanup_pending' if close_error is None and passed and report is not None else 'failed',
            'output_sha256': _frame_hash(output) if output.exists() else None,
            'cleanup_error': None if close_error is None else f'{type(close_error).__name__}: {close_error}',
        }, indent=2), encoding='utf-8')
        if close_error is not None:
            raise RuntimeError(f'geometry renderer cleanup failed: {close_error}') from close_error
    if passed and report is not None:
        report['status'] = 'passed'
        report['complete'] = True
        try:
            _persist_report(
                output, report,
                budget_context={
                    'experiment': 'legacy_depth_ablation_v1_geometry_gate',
                    'phase': 'final_geometry_gate',
                    'frame_ids': [], 'unattempted_frame_ids': [],
                    'geometry_ray_count': len(report.get('rays', [])),
                })
        except BaseException:
            marker.write_text(json.dumps({
                'status': 'failed',
                'output_sha256': _frame_hash(output) if output.exists() else None,
                'cleanup_error': None,
            }, indent=2), encoding='utf-8')
            raise
        marker.write_text(json.dumps({
            'status': 'complete',
            'output_sha256': _frame_hash(output) if output.exists() else None,
            'cleanup_error': None,
        }, indent=2), encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    geometry_parser = subparsers.add_parser('geometry')
    geometry_parser.add_argument('--output', required=True)
    pose_parser = subparsers.add_parser('pose')
    pose_parser.add_argument('--object', choices=('ranch',), required=True)
    pose_parser.add_argument('--branch', choices=('control', 'candidate'), required=True)
    pose_parser.add_argument('--output', required=True)
    pose_parser.add_argument('--prefix-120', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'geometry':
        return geometry(args.output)
    if args.object != 'ranch':
        raise ValueError('only the original ranch bottle tracking window is supported')
    return pose(args.branch, args.output, prefix_120=args.prefix_120)


if __name__ == '__main__':
    main()
