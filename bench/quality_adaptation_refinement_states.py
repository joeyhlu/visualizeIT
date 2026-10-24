"""Prospective integer-refinement crops for the unchanged source-18 trainer.

Run separate cloud processes in order (all paths and plan hashes are explicit)::

  python -m bench.quality_adaptation_refinement_states capture --plan PLAN --plan-sha256 SHA --output CAPTURE --cloud-execution
  python -m bench.quality_adaptation_refinement_states extract --root CAPTURE --features FEATURES --training-source-root SOURCE18 --training-source-manifest MANIFEST --cloud-execution
  python -m bench.quality_adaptation_refinement_states seal --root CAPTURE --features FEATURES --cloud-execution
  python -m bench.quality_adaptation_refinement_states labels --root CAPTURE --features FEATURES --label-plan LABELPLAN --label-plan-sha256 SHA --training-source-root SOURCE18 --training-source-manifest MANIFEST --cloud-execution
  python -m bench.quality_adaptation_refinement_states train --root CAPTURE --features FEATURES --config CONFIG --config-sha256 SHA --output RUN --training-source-root SOURCE18 --training-source-manifest MANIFEST --cloud-execution

Captures contain estimated geometry and frozen predictions only. Labels open
after every available extraction has passed masked prediction parity and the
entire pre-label closure is sealed. TEST7 has metadata only. No quality claim is
made; the frozen released head remains the serving default. Import is Torch-free.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

from . import quality_gotrack_inference as inference
from .quality_adaptation_data import (CONVENTION, CROP_KEYS, FEATURE_KEYS, HOOKS,
    SIZE, canonical_hash, crop_rgb_hash, descriptor, load_dataset, load_npz,
    read_json, require_hash, sha256, validate_crops, validate_features)
from .quality_adaptation_runtime import atomic_stream, cloud_guard, tree_bytes, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'gotrack_refinement_state_input_plan_v1'
SOURCE_SCHEMA = 'gotrack_refinement_state_source_contract_v1'
LABEL_SCHEMA = 'gotrack_refinement_state_label_plan_v1'
HISTORICAL_MANIFEST_SHA256 = '1d45840df6f3f031a031ac133d4adccf50db8a17c9ce9d206380c5a63ae54850'
HISTORICAL_LOCK_SHA256 = inference.TRAINING_LOCK_SHA256
TRAIN_ITERATIONS = (0, 1, 0, 3, 0, 1, 0, 3)
SEQUENCES = (
    ('train', '17', 'ranker-multiview-hot3d17-v1-yaw0-pitchm25', -25),
    ('train', '17', 'ranker-multiview-hot3d17-v1-yaw0-pitchp25', 25),
    ('val', '27', 'ranker-multiview-hot3d27-v1-yaw0-pitchm25', -25),
)
LIMITS = dict(max_capture_seconds=900, max_extract_seconds=900,
    max_train_seconds=1200, max_aggregate_seconds=3600,
    max_feature_bytes=512 * 1024**2, max_run_bytes=1024**3,
    max_private_bytes=2 * 1024**3, disk_reserve_bytes=1024**3,
    model_cache_bytes=8 * 1024**3)
EXTRA_SOURCES = frozenset('bench/' + name + '.py' for name in (
    'quality_adaptation_refinement_states', 'quality_gotrack_inference',
    'quality_gotrack_probe', 'quality_adaptation_prepare', 'quality_adaptation_data',
    'quality_adaptation_runtime', 'quality_adaptation_smoke',
    'quality_adaptation_manifest', 'quality_adaptation_expanded',
    'quality_ranker_synthetic', 'quality_adaptation_train'))
GEOMETRY_KEYS = {'crop_from_orig', 'crop_k', 'native_k', 'current_crop_camera_from_object_mm',
                 'actual_flow', 'actual_confidence', 'rendered_depth_mm', 'rendered_mask'}
_ISSUER = object()


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{name} has missing or unexpected fields')


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _relative(root, raw):
    if (not isinstance(raw, str) or not raw or '\\' in raw or ':' in raw
            or PurePosixPath(raw).is_absolute() or '..' in PurePosixPath(raw).parts):
        raise ValueError('source path must be normalized relative POSIX')
    candidate = root / raw
    if inference._has_symlink_component(candidate):
        raise ValueError('source path cannot contain symlinks')
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('source path escapes source root or is not regular')
    return path


def pinned_file(pin, *, source=False):
    _keys(pin, ('path', 'bytes', 'sha256'), 'file pin')
    require_hash(pin['sha256'])
    if type(pin['bytes']) is not int or not 0 < pin['bytes'] <= 8 * 1024**3:
        raise ValueError('pin bytes must be a positive bounded integer')
    raw = pin['path']
    if not isinstance(raw, str) or '\\' in raw or '..' in PurePosixPath(raw).parts:
        raise ValueError('pin path must be normalized POSIX')
    if source:
        path = _relative(ROOT, raw)
    else:
        candidate = Path(raw)
        if not candidate.is_absolute() or not raw.startswith('/teamspace/studios/this_studio/'):
            raise ValueError('data pin must be absolute within the reviewed cloud studio')
        if inference._has_symlink_component(candidate):
            raise ValueError('data pin cannot contain symlinks')
        path = candidate.resolve(strict=True)
    if not path.is_file() or path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
        raise ValueError(f'file pin mismatch: {raw}')
    return path


def validate_plan(plan, *, verify_files=False):
    _keys(plan, ('schema', 'runtime_contract', 'source_contract', 'rows', 'sealed_test', 'limits'), 'input plan')
    if plan['schema'] != SCHEMA or plan['limits'] != LIMITS:
        raise ValueError('input plan schema or prospective resource limits differ')
    for key in ('runtime_contract', 'source_contract'):
        pin = plan[key]
        _keys(pin, ('path', 'bytes', 'sha256'), key)
        require_hash(pin['sha256'])
        if type(pin['bytes']) is not int or pin['bytes'] <= 0:
            raise ValueError('source contract bytes must be positive')
        if verify_files:
            pinned_file(pin, source=True)
    if not isinstance(plan['rows'], list) or len(plan['rows']) != 24:
        raise ValueError('input plan requires exactly 24 prospectively selected physical frames')
    fields = {'id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id',
              'yaw0_deg', 'pitch_deg', 'timestamp_s', 'input', 'video', 'mesh',
              'mask_index', 'mask', 'bank', 'iterations'}
    for ordinal, row in enumerate(plan['rows']):
        _keys(row, fields, 'physical input row')
        split, obj, seq, pitch = SEQUENCES[ordinal // 8]
        frame = ordinal % 8
        selected = [TRAIN_ITERATIONS[frame]] if split == 'train' else [0, 1, 3]
        if (type(row['frame_id']) is not int or row['frame_id'] != frame
                or (row['split'], row['object_id'], row['sequence_id'], row['pitch_deg']) != (split, obj, seq, pitch)
                or type(row['yaw0_deg']) is not int or row['yaw0_deg'] != 0
                or type(row['timestamp_s']) not in (int, float) or row['timestamp_s'] != frame / 10.
                or row['iterations'] != selected or any(type(i) is not int for i in row['iterations'])
                or row['id'] != f'{seq}:frame{frame}'):
            raise ValueError('physical row differs from the fixed frame/iteration schedule')
        if not isinstance(row['capture_group'], str) or not 0 < len(row['capture_group']) <= 256:
            raise ValueError('invalid capture group')
        for name, suffix in (('input', 'input.json'), ('video', 'synthetic.mp4'),
                ('mesh', 'object.glb'), ('mask_index', 'results.json'), ('mask', f'{frame}.png'), ('bank', '.pt')):
            pin = row[name]
            _keys(pin, ('path', 'bytes', 'sha256'), 'input pin')
            require_hash(pin['sha256'])
            raw = pin['path']
            if (not isinstance(raw, str) or not raw.startswith('/teamspace/studios/this_studio/')
                    or '\\' in raw or '..' in PurePosixPath(raw).parts
                    or not raw.endswith(suffix) or type(pin['bytes']) is not int or pin['bytes'] <= 0
                    or any(token in raw.lower() for token in ('reference', 'evaluation', 'generation', 'poses.json', 'obj_000007', 'hot3d7'))):
                raise ValueError('annotation, generation, TEST or malformed inference input is forbidden')
            if verify_files:
                pinned_file(pin)
    test = plan['sealed_test']
    _keys(test, ('id', 'object_id', 'sequence_id', 'capture_group', 'split', 'status',
        'query_rgb_sha256', 'failure_reason', 'unavailable_provenance'), 'sealed TEST metadata')
    if (test['object_id'] != '7' or test['split'] != 'test' or test['status'] != 'unavailable'
            or test['query_rgb_sha256'] is not None or not test['failure_reason'] or not test['unavailable_provenance']):
        raise ValueError('TEST7 must remain unavailable with no payload paths')
    # The original object split is preserved. VAL27 has already been used for
    # development; no independent generalization claim follows from this split.
    if set(r['capture_group'] for r in plan['rows'][:16]) & set(r['capture_group'] for r in plan['rows'][16:]):
        raise ValueError('TRAIN and used VAL capture groups must be disjoint')
    return plan


def verify_source_contract(plan):
    contract = read_json(pinned_file(plan['source_contract'], source=True))
    _keys(contract, ('schema', 'sources', 'historical_training_manifest_sha256'), 'source contract')
    if (contract['schema'] != SOURCE_SCHEMA
            or contract['historical_training_manifest_sha256'] != HISTORICAL_MANIFEST_SHA256
            or not isinstance(contract['sources'], list)):
        raise ValueError('source contract differs from the historical trainer closure')
    pins = {}
    for pin in contract['sources']:
        _keys(pin, ('path', 'sha256'), 'source pin')
        require_hash(pin['sha256'])
        if pin['path'] in pins:
            raise ValueError('duplicate source pin')
        path = _relative(ROOT, pin['path'])
        if sha256(path) != pin['sha256']:
            raise ValueError(f'capture source mismatch: {pin["path"]}')
        pins[pin['path']] = pin['sha256']
    if set(pins) != set(inference.EXPECTED_RUNTIME_FILES) | EXTRA_SOURCES:
        raise ValueError('capture source contract has missing or foreign dependencies')
    runtime = read_json(pinned_file(plan['runtime_contract'], source=True))
    inference._validate_contract(runtime)
    inference.verify_source_compatibility(runtime)
    for path, expected in runtime['additional_runtime_files'].items():
        if pins[path] != expected:
            raise ValueError('runtime/source contracts disagree')
    return contract


@dataclass(frozen=True, slots=True)
class _VerifiedPreflight:
    issuer: object
    plan_bytes: bytes


@dataclass(frozen=True, slots=True)
class RefStateObserverCapability:
    """Exact observer/profile binding issued after input-only authentication."""
    _issuer: object
    _plan_bytes: bytes
    _plan_sha256: str
    _sink: object

    def validate(self, sink, geometry):
        if (self._issuer is not _ISSUER or sink is not self._sink
                or type(sink) is not RefStateCapture or geometry != inference.PROFILE
                or hashlib.sha256(self._plan_bytes).hexdigest() != self._plan_sha256):
            raise ValueError('invalid refinement-state observer capability/sink/profile')
        verify_source_contract(validate_plan(inference._loads_unique(self._plan_bytes, 'input plan')))


def _issue(plan_bytes, plan_sha256, sink, verified):
    if (type(verified) is not _VerifiedPreflight or verified.issuer is not _ISSUER
            or verified.plan_bytes != plan_bytes or type(sink) is not RefStateCapture
            or hashlib.sha256(plan_bytes).hexdigest() != plan_sha256):
        raise ValueError('capability requires exact completed input-only preflight')
    capability = RefStateObserverCapability(_ISSUER, plan_bytes, plan_sha256, sink)
    capability.validate(sink, inference.PROFILE)
    return capability


def state_rows(plan):
    rows = []
    for physical in plan['rows']:
        for iteration in physical['iterations']:
            rows.append({**{key: physical[key] for key in ('object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s')},
                'id': physical['id'] + f':iteration{iteration}', 'iteration': iteration,
                'status': 'unavailable', 'query_rgb_sha256': None,
                'failure_reason': 'not_processed', 'unavailable_provenance': 'prospective refinement state; no crop materialized'})
    return rows


def _base(row):
    return {key: row[key] for key in ('id', 'object_id', 'split', 'sequence_id', 'capture_group')}


def _save_npz(root, name, arrays):
    with atomic_stream(root, root / name, max_file_bytes=16 * 1024**2,
                       max_output_bytes=LIMITS['max_private_bytes']) as stream:
        np.savez(stream, **arrays)
    return _artifact(root, root / name)


def _artifact(root, path):
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path))


@contextmanager
def _stage_lock():
    """One capture/extraction/rendering/training stage on the shared model cache."""
    path = ROOT / '.cache/model-quality/.refinement-state-stage.lock'
    # A lock left by a killed process requires operator investigation. Never
    # steal it using a stale PID guess or automatically launch another model.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_unix_s=time.time())))
    try:
        yield
    finally:
        path.unlink()


def _exclusive_stage(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        cloud_guard(kwargs.get('cloud_execution', False))
        with _stage_lock():
            return function(*args, **kwargs)
    return guarded


@contextmanager
def observe_selected_crops(selected):
    """Check the returned sampling map at EACH selected ordinary iteration."""
    from utils import data_util, misc
    from .quality_gotrack import _capture_actual_sampling_map, _capture_native_uv
    from .quality_adaptation_smoke import sampling_map_error
    original = data_util.compute_gotrack_inputs_from_init_poses
    checks, counter = {}, 0

    def observed(*args, **kwargs):
        nonlocal counter
        iteration = counter
        counter += 1
        result = original(*args, **kwargs)
        if iteration in selected:
            _, cameras, transforms = result
            c = transforms[0]
            c = c.detach().cpu().numpy() if hasattr(c, 'detach') else np.asarray(c)
            from .quality_adaptation_expanded import validate_retained_geometry
            c = validate_retained_geometry(c)
            native, crop = kwargs['input_cameras'][0], cameras[0]
            yy, xx = np.mgrid[:SIZE, :SIZE]
            grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
            actual = _capture_actual_sampling_map(native, crop, grid)[3]
            expected = _capture_native_uv(grid, np.asarray(misc.get_intrinsic_matrix(crop)),
                c[:3, :3], np.asarray(misc.get_intrinsic_matrix(native)))
            checks[iteration] = sampling_map_error(actual, expected)
        return result
    data_util.compute_gotrack_inputs_from_init_poses = observed
    try:
        yield checks
    finally:
        data_util.compute_gotrack_inputs_from_init_poses = original


class RefStateCapture:
    """Persist only prescribed retained states; PnP availability is independent."""
    def __init__(self, output, physical, rows, sampling_checks, native_k, *,
                 max_output_bytes=None, reserve_bytes=1024**3, resource_check=None):
        self.output, self.physical, self.rows = output, physical, rows
        self.sampling_checks, self.native_k = sampling_checks, np.array(native_k, copy=True)
        self.failure = None
        self.seen = set()
        self.solver = {}
        if max_output_bytes is not None and (type(max_output_bytes) is not int or
                not 0 < max_output_bytes <= 100 * 1024**3):
            raise ValueError('bounded capture output bytes required')
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError('nonnegative capture reserve required')
        self.max_output_bytes, self.reserve_bytes = max_output_bytes, reserve_bytes
        self.resource_check = resource_check

    def _write(self, name, arrays):
        if self.resource_check is not None:
            self.resource_check()
        if self.max_output_bytes is None:
            return _save_npz(self.output, name, arrays)
        with atomic_stream(self.output, self.output / name, max_file_bytes=16 * 1024**2,
                max_output_bytes=self.max_output_bytes, reserve_bytes=self.reserve_bytes) as stream:
            np.savez(stream, **arrays)
        return _artifact(self.output, self.output / name)

    def _disable_capture(self, reason):
        self.failure = str(reason)[:1024]

    def should_capture(self, context, kind, metadata):
        return (self.failure is None and metadata.get('iteration') in self.physical['iterations']
                and kind in ('retained', 'result'))

    def preflight_nbytes(self, context, kind, metadata, nbytes):
        return type(nbytes) is int and 0 <= nbytes <= 32 * 1024**2

    def preflight(self, context, kind, metadata, descriptors):
        return self.should_capture(context, kind, metadata) and self.preflight_nbytes(
            context, kind, metadata, sum(v['nbytes'] for v in descriptors.values()))

    def __call__(self, packet):
        meta, arrays = packet['metadata'], packet['arrays']
        iteration, kind = meta['iteration'], packet['kind']
        if not self.should_capture(packet['context'], kind, meta) or (iteration, kind) in self.seen:
            raise ValueError('unexpected/duplicate refinement-state packet')
        if (meta.get('frame_id') != self.physical['frame_id']
                or packet['context'] != dict(physical_id=self.physical['id'])):
            raise ValueError('retained packet frame/context differs from selected physical row')
        self.seen.add((iteration, kind))
        row = next(r for r in self.rows if r['id'] == self.physical['id'] + f':iteration{iteration}')
        if kind == 'result':
            self.solver[iteration] = meta.get('fit_state', 'solver_unavailable')
            if row['status'] == 'available':
                row['pnp_state'] = self.solver[iteration]
            return
        if iteration not in self.sampling_checks:
            raise ValueError('selected iteration lacks its own returned-camera sampling check')
        if not np.array_equal(np.asarray(arrays['native_k']), self.native_k):
            raise ValueError('retained native calibration differs from authenticated integer profile')
        from .quality_adaptation_prepare import crops_from_retained_capture
        from .quality_adaptation_expanded import validate_retained_geometry
        crops = crops_from_retained_capture(arrays, preserve_float_rgb=True)
        c = validate_retained_geometry(arrays['crop_from_orig'])
        ordinal = self.rows.index(row)
        crop = self._write(f'crop-{ordinal:02d}.npz', crops)
        geometry = self._write(f'capture-{ordinal:02d}.npz', {
            'crop_from_orig': c, 'crop_k': np.asarray(arrays['crop_k'], np.float64),
            'native_k': self.native_k, 'current_crop_camera_from_object_mm': np.asarray(arrays['current_crop_camera_from_object_mm'], np.float64),
            'rendered_depth_mm': np.asarray(arrays['rendered_depth_mm'], np.float32).squeeze(),
            'rendered_mask': np.asarray(arrays['rendered_mask']).squeeze().astype(bool),
            'actual_flow': np.asarray(arrays['full_flow_crop_px'], np.float32),
            'actual_confidence': np.asarray(arrays['full_confidence'], np.float32)})
        row.pop('failure_reason', None)
        row.pop('unavailable_provenance', None)
        row.update(status='available', query_rgb_sha256=crop_rgb_hash(crops['query_rgb']),
            crops=crop, capture=geometry, sampling_map=self.sampling_checks[iteration],
            seed_selection='first_FoundPose_quality_sorted_hypothesis',
            pnp_state='solver_result_not_observed', native_calibration=self.native_k.tolist())

    def finish(self, reason=None):
        for row in self.rows:
            if not row['id'].startswith(self.physical['id'] + ':iteration'):
                continue
            if row['status'] == 'unavailable':
                row['failure_reason'] = self.failure or reason or 'unrun_iteration_after_solver_termination'
            elif row['pnp_state'] == 'solver_result_not_observed':
                row['pnp_state'] = 'solver_unavailable_after_retained_crop'
                row['pnp_failure'] = reason


@dataclass(frozen=True, slots=True)
class _InputContext:
    profile: str
    contract: object
    artifact: object
    accepted_input: object
    effective_intrinsics: np.ndarray


def _allowlisted_input(plan, row):
    """Reject foreign payload pins using authenticated source metadata only."""
    runtime = read_json(pinned_file(plan['runtime_contract'], source=True))
    accepted = [entry for entry in runtime['accepted_inputs'] if entry['sequence_id'] == row['sequence_id']]
    if len(accepted) != 1:
        raise ValueError('input sequence lacks a unique authenticated allowlist entry')
    entry = accepted[0]
    if any(row[name]['sha256'] != entry[field] for name, field in
            (('input', 'input_sha256'), ('video', 'video_sha256'), ('mesh', 'mesh_sha256'))):
        raise ValueError('observed input pins differ from the serving allowlist')
    return runtime, entry


def _input_context(plan, row, manifest):
    """Admit observed input bytes without opening the generation receipt.

    The existing serving validator also validates generation metadata. This
    prospective data path deliberately defers that extra file to the sealed
    label phase, while retaining its exact input allowlist, geometry and masks.
    """
    runtime, entry = _allowlisted_input(plan, row)
    if (manifest.get('schema_version') != 2 or manifest.get('provenance_kind') != 'synthetic'
            or manifest.get('units') != 'metres' or manifest.get('asset') != 'object.glb'
            or manifest.get('video') != 'synthetic.mp4' or manifest.get('_capture') is not None
            or manifest.get('_timing', {}).get('clock_mode') != 'capture_seconds_v1'
            or manifest.get('physical_object_id') != entry['physical_object_id']
            or manifest.get('native_resolution') != entry['native_resolution']
            or manifest.get('source_hashes') != {entry['mesh_path']: entry['mesh_sha256'], entry['video_path']: entry['video_sha256']}):
        raise ValueError('observed input metadata/bytes differ from the serving allowlist')
    original = inference._validated_native_intrinsics(manifest, entry)
    effective = np.array(original, dtype=np.float64, copy=True)
    effective[0, 2] += inference.NATIVE_CORRECTION[0]
    effective[1, 2] += inference.NATIVE_CORRECTION[1]
    effective.setflags(write=False)
    # Only fields used by the existing physical-mask producer validator survive.
    accepted_input = {k: entry[k] for k in ('input_sha256', 'mesh_path', 'mesh_sha256', 'video_path', 'video_sha256')}
    return _InputContext(inference.PROFILE, inference._freeze(runtime),
        inference._freeze(dict(inference._artifact_for_contract(runtime))),
        inference._freeze(accepted_input), effective)


def _preflight(plan):
    validate_plan(plan, verify_files=False)
    verify_source_contract(plan)
    # Admit every row before hashing or parsing any prospective data payload.
    # read_input hashes resources named inside a manifest, so it must follow
    # authentication against the exact serving allowlist.
    for row in plan['rows']:
        _allowlisted_input(plan, row)
    validate_plan(plan, verify_files=True)
    from .quality_runner import read_input, _physical_mask_cache
    from .quality_adaptation_expanded import validate_mask_binding, manifest_frame_timestamp
    inputs = {}
    for row in plan['rows']:
        bundle = pinned_file(row['input']).parent
        if pinned_file(row['video']) != bundle / 'synthetic.mp4' or pinned_file(row['mesh']) != bundle / 'object.glb':
            raise ValueError('input/video/mesh must belong to the same exact bundle')
        manifest = read_input(bundle)
        if (manifest['sequence_id'] != row['sequence_id']
                or manifest['physical_object_id'] != f"hot3d:obj_{int(row['object_id']):06d}"
                or manifest.get('provenance_kind') != 'synthetic'):
            raise ValueError('input sequence/object/synthetic provenance differs')
        manifest_frame_timestamp(manifest, row)
        context = _input_context(plan, row, manifest)
        index_path = pinned_file(row['mask_index'])
        masks, _ = _physical_mask_cache(manifest, index_path.parent)
        inference.validate_physical_mask_producer(context, masks)
        if masks.get('diagnostic_control') or masks.get('automatic') is False:
            raise ValueError('capture requires cached observed automatic masks')
        validate_mask_binding(row, masks, index_path.parent, pinned_file(row['mask']))
        inputs[row['id']] = (bundle, manifest, context)
    return inputs


def _disk_preflight(output):
    if inference._has_symlink_component(output):
        raise ValueError('output cannot contain symlinks')
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < LIMITS['max_private_bytes'] + LIMITS['disk_reserve_bytes']:
        raise ValueError('requires 2 GiB private budget plus 1 GiB free reserve')
    if tree_bytes(ROOT / '.cache/model-quality') > LIMITS['model_cache_bytes']:
        raise ValueError('model cache exceeds 8 GiB; automatic eviction is forbidden')


@_exclusive_stage
def capture(plan_path, plan_sha256, output, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    require_hash(plan_sha256)
    raw = Path(plan_path).read_bytes()
    if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != plan_sha256:
        raise ValueError('input plan byte size/hash mismatch')
    plan = validate_plan(inference._loads_unique(raw, 'input plan'))
    output = Path(output)
    _disk_preflight(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(raw)
    rows = state_rows(plan)  # All forty states exist BEFORE input decoding/model work.
    write_json(output, 'capture-index.json', dict(rows=rows, test_payload_reads=0))
    started, wall_started, network, fatal = time.monotonic(), float(os.environ.get(
        'VISUALIZEIT_REFINEMENT_WORKFLOW_STARTED', time.time())), None, None
    versions = None
    try:
        inputs = _preflight(plan)
        verified = _VerifiedPreflight(_ISSUER, raw)
        from .quality_adaptation_runtime import runtime_versions
        versions = runtime_versions()
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        from .quality_gotrack import GoTrackRefiner, TexturedRenderer, load_network
        from .quality_foundpose import FoundPoseRecovery
        from .quality_contract import Frame
        from .quality_runner import read_rgb
        from .quality_time import PHYSICAL
        from .vision import cv2
        network = load_network('cpu')  # Released frozen network; no head installation.
        network.eval()
        for physical in plan['rows']:
            renderer = sink = None
            bank = backend = seed_refiner = refiner = None
            reason = None
            try:
                if time.monotonic() - started >= LIMITS['max_capture_seconds']:
                    raise TimeoutError('capture budget exhausted before physical frame')
                bundle, manifest, context = inputs[physical['id']]
                rgb = read_rgb(bundle, manifest, physical['frame_id'])
                mask = cv2.imread(str(pinned_file(physical['mask'])), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                    raise ValueError('decoded observed mask is missing/empty/wrong shape')
                frame = Frame(physical['frame_id'], rgb, context.effective_intrinsics,
                    timestamp_s=physical['timestamp_s'], clock_mode=PHYSICAL)
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                    unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                # Seed estimation never sees an observer or labels and never refines.
                seed_refiner = GoTrackRefiner(network, renderer, manifest['object_id'], 'cpu', inference_geometry=inference.PROFILE)
                bank = torch.load(pinned_file(physical['bank']), map_location='cpu', weights_only=False)
                backend = FoundPoseRecovery(seed_refiner, bank, time_policy=manifest['_timing']['time_policy'])
                seeds = backend.recover(frame, mask, top_k=5)
                del bank, backend, seed_refiner
                if not seeds:
                    raise ValueError('unavailable_seed: FoundPose produced no estimated hypothesis')
                with observe_selected_crops(physical['iterations']) as checks:
                    sink = RefStateCapture(output, physical, rows, checks, context.effective_intrinsics)
                    capability = _issue(raw, plan_sha256, sink, verified)
                    refiner = GoTrackRefiner(network, renderer, manifest['object_id'], 'cpu',
                        inference_geometry=inference.PROFILE, chronological_capture_callback=sink,
                        inference_observer_capability=capability)
                    refiner.set_chronological_capture_context(dict(physical_id=physical['id']))
                    refiner.refine(frame, mask, seeds[0])  # Each physical frame refined exactly once.
                    del refiner
            except Exception as exc:
                reason = f'{type(exc).__name__}: {exc}'[:1024]
            finally:
                if sink is not None:
                    sink.finish(reason)
                else:
                    for row in rows:
                        if row['id'].startswith(physical['id'] + ':iteration'):
                            row['failure_reason'] = reason or 'capture_unavailable'
                if renderer is not None:
                    try:
                        renderer.close()
                    except Exception as exc:
                        fatal = f'renderer cleanup failed: {exc}'[:1024]
                bank = backend = seed_refiner = refiner = renderer = None
                write_json(output, 'capture-index.json', dict(rows=rows, test_payload_reads=0))
                if fatal:
                    break
    except Exception as exc:
        fatal = f'{type(exc).__name__}: {exc}'[:1024]
    finally:
        if network is not None:
            try:
                network.cpu()
            except Exception as exc:
                fatal = f'network cleanup failed: {exc}'[:1024]
            finally:
                network = None
        for row in rows:
            if row.get('failure_reason') == 'not_processed':
                row['failure_reason'] = fatal or 'unrun_after_capture_budget_or_termination'
        write_json(output, 'capture-index.json', dict(rows=rows, test_payload_reads=0))
        write_json(output, 'capture-receipt.json', dict(schema='gotrack_refinement_state_capture_receipt_v1',
            plan_sha256=plan_sha256, rows=rows, versions=versions, annotation_inputs=[],
            test_payload_reads=0, elapsed_seconds=time.monotonic() - started, fatal_error=fatal,
            started_unix_s=wall_started,
            input_only=True, physical_frames_requested=24, refinement_states_requested=40))
        write_json(output, 'crops.json', dict(schema='correspondence_input_crops_v1', conventions=CONVENTION,
            rows=[dict(id=r['id'], crops=r['crops']) for r in rows if r['status'] == 'available']))
    if fatal:
        raise RuntimeError(f'capture failed; full inventory retained: {fatal}')
    return dict(requested=40, available=sum(r['status'] == 'available' for r in rows), test_payload_reads=0)


def verify_training_root(root, manifest_path):
    """Authenticate the original source-18 closure in its separate checkout."""
    root, manifest_path = Path(root).resolve(strict=True), Path(manifest_path)
    if root == ROOT.resolve() or sha256(manifest_path) != HISTORICAL_MANIFEST_SHA256:
        raise ValueError('training requires separate historical source root and exact source-18 manifest')
    manifest = read_json(manifest_path)
    if manifest.get('kind') != 'verified-gotrack-adaptation-source-v18' or not isinstance(manifest.get('files'), dict):
        raise ValueError('invalid historical source manifest')
    for relative, expected in manifest['files'].items():
        if sha256(_relative(root, relative)) != require_hash(expected):
            raise ValueError(f'historical source-18 mismatch: {relative}')
    lock_path = _relative(root, 'bench/runtime/adaptation-sources.json')
    if sha256(lock_path) != HISTORICAL_LOCK_SHA256:
        raise ValueError('historical adaptation source lock changed')
    lock = read_json(lock_path)
    for relative, expected in lock['gotrack']['files'].items():
        candidate = root / relative
        # A shared model cache alias is permitted ONLY to the existing reviewed
        # mount; ordinary source files still reject all symlink components.
        if relative.startswith('.cache/model-quality/'):
            cache = (root / '.cache/model-quality').resolve(strict=True)
            if cache.as_posix() != inference.APPROVED_CACHE_MOUNT:
                raise ValueError('historical model cache differs from reviewed mount')
            path = candidate.resolve(strict=True)
            if not path.is_relative_to(cache) or not path.is_file():
                raise ValueError('historical upstream path escapes model cache')
        else:
            path = _relative(root, relative)
        if sha256(path) != expected:
            raise ValueError(f'historical training dependency mismatch: {relative}')
    return manifest


def _run_historical(root, manifest_path, module, args, *, timeout, accepted_exit_codes=(0,)):
    verify_training_root(root, manifest_path)
    # -m in the authenticated cwd deliberately avoids this checkout's imports.
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, '-m', module, *map(str, args)],
        cwd=Path(root).resolve(), env=env, timeout=timeout, check=False)
    verify_training_root(root, manifest_path)
    if completed.returncode not in accepted_exit_codes:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return completed.returncode


def _remaining(root, *others):
    """Share one capture-start deadline/storage allowance across all processes."""
    receipt = read_json(root / 'capture-receipt.json')
    started = receipt.get('started_unix_s')
    if type(started) not in (int, float) or not np.isfinite(started) or started > time.time() + 1:
        raise ValueError('capture receipt lacks a valid shared aggregate clock')
    remaining = LIMITS['max_aggregate_seconds'] - (time.time() - started)
    if remaining <= 0:
        raise TimeoutError('shared 3600-second aggregate budget exhausted; no automatic reset')
    roots = []
    monitors = list(root.parent.glob(root.name + '-monitor-*'))
    for candidate in (root, *others, *monitors):
        candidate = candidate.resolve()
        if candidate.exists() and not any(candidate.is_relative_to(parent) for parent in roots):
            roots = [parent for parent in roots if not parent.is_relative_to(candidate)] + [candidate]
    if sum(tree_bytes(path) for path in roots) > LIMITS['max_private_bytes']:
        raise ValueError('shared 2 GiB private artifact budget exceeded')
    if shutil.disk_usage(root).free < LIMITS['disk_reserve_bytes']:
        raise ValueError('private artifact disk reserve fell below 1 GiB')
    return remaining


def _read_capture(root):
    plan = validate_plan(read_json(root / 'plan.json'))
    verify_source_contract(plan)
    index, receipt = read_json(root / 'capture-index.json'), read_json(root / 'capture-receipt.json')
    expected = state_rows(plan)
    rows = index.get('rows')
    if (not isinstance(rows, list) or len(rows) != 40
            or [r.get('id') for r in rows] != [r['id'] for r in expected]
            or receipt.get('rows') != rows or index.get('test_payload_reads') != 0
            or receipt.get('test_payload_reads') != 0 or receipt.get('annotation_inputs') != []
            or receipt.get('plan_sha256') != sha256(root / 'plan.json') or receipt.get('fatal_error')):
        raise ValueError('capture inventory/receipt does not bind all forty prospective states')
    for actual, prospect in zip(rows, expected):
        if any(actual.get(k) != prospect[k] for k in ('id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s', 'iteration')):
            raise ValueError('capture row identity differs from prospective schedule')
        if actual.get('status') not in ('available', 'unavailable'):
            raise ValueError('invalid capture availability')
    return plan, rows


@_exclusive_stage
def extract_stage(root, features, training_source_root, training_source_manifest, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    root, features = Path(root), Path(features)
    _, rows = _read_capture(root)
    remaining = _remaining(root, features)
    if features.exists():
        raise FileExistsError('extraction requires a new directory')
    try:
        return _run_historical(training_source_root, training_source_manifest,
            'bench.quality_adaptation_prepare', ['--manifest', (root / 'crops.json').resolve(),
                '--output', features.resolve(), '--max-seconds', LIMITS['max_extract_seconds'],
                '--max-output-bytes', LIMITS['max_feature_bytes'], '--max-rows', 40, '--cloud-execution'],
            timeout=min(LIMITS['max_extract_seconds'], remaining))
    finally:
        features.mkdir(parents=True, exist_ok=True)
        path = features / 'index.json'
        index = read_json(path) if path.is_file() else {}
        available = {r['id']: r for r in index.get('rows', [])}
        if set(available) - {r['id'] for r in rows}:
            raise ValueError('extractor contains foreign row identity')
        index['rows'] = [available.get(row['id'], dict(id=row['id'], status='unavailable',
            failure_reason=row.get('failure_reason') or 'extraction_failure_or_not_processed')) for row in rows]
        # Adding explicit failed-capture rows does not alter any feature receipt,
        # recipe, payload or historical extractor implementation.
        write_json(features, 'index.json', index, max_output_bytes=LIMITS['max_feature_bytes'])
        _remaining(root, features)


def _seal_material(root, features):
    plan, rows = _read_capture(root)
    crops_doc, recipe, index = (read_json(root / 'crops.json'), read_json(features / 'recipe.json'), read_json(features / 'index.json'))
    expected_crops = dict(schema='correspondence_input_crops_v1', conventions=CONVENTION,
        rows=[dict(id=r['id'], crops=r['crops']) for r in rows if r['status'] == 'available'])
    if crops_doc != expected_crops:
        raise ValueError('input-only crop manifest differs from capture inventory')
    from .quality_adaptation_runtime import LOCK
    lock = read_json(LOCK)
    if (sha256(LOCK) != HISTORICAL_LOCK_SHA256
            or recipe.get('schema') != 'frozen_gotrack_decoder_recipe_v1'
            or recipe.get('source_lock_sha256') != HISTORICAL_LOCK_SHA256
            or recipe.get('source_files_sha256') != canonical_hash(lock['gotrack']['files'])
            or recipe.get('upstream_revision') != lock['gotrack']['revision']
            or recipe.get('producer_sha256') != lock['gotrack']['files']['bench/quality_adaptation_prepare.py']
            or recipe.get('hooks') != list(HOOKS) or recipe.get('annotation_inputs') != []
            or recipe.get('crop_manifest_sha256') != canonical_hash(crops_doc)
            or recipe.get('rgb_encoding') != 'uint8_div255_or_exact_float32_0_1_v1'
            or index.get('recipe_sha256') != canonical_hash(recipe)
            or [r.get('id') for r in index.get('rows', [])] != [r['id'] for r in rows]):
        raise ValueError('historical extractor recipe/index/source binding differs')
    material = dict(schema='gotrack_refinement_state_prelabel_seal_v1',
        capture_files=[_artifact(root, root / p) for p in ('plan.json', 'capture-index.json', 'capture-receipt.json', 'crops.json')],
        feature_files=[_artifact(features, features / p) for p in ('recipe.json', 'index.json')],
        geometry=[], crops=[], features=[], identities=state_rows(plan), parity=[],
        annotation_inputs=[], test_payload_reads=0)
    from .quality_adaptation_expanded import _check_parity, validate_retained_geometry
    for captured, extracted in zip(rows, index['rows']):
        if extracted.get('status') not in ('available', 'unavailable'):
            raise ValueError('invalid extraction availability')
        if captured['status'] != 'available':
            if extracted['status'] == 'available':
                raise ValueError('foreign extraction for an unavailable captured state')
            continue
        crop_path = descriptor(root, captured['crops'], verify=True)
        actual_path = descriptor(root, captured['capture'], verify=True)
        crops = load_npz(crop_path, CROP_KEYS)
        validate_crops(crops)
        if crops['query_rgb'].dtype != np.float32 or crops['template_rgb'].dtype != np.float32:
            raise ValueError('refinement states require exact float32 RGB crops')
        if captured['query_rgb_sha256'] != crop_rgb_hash(crops['query_rgb']):
            raise ValueError('capture query RGB identity changed')
        actual = load_npz(actual_path, GEOMETRY_KEYS)
        validate_retained_geometry(actual['crop_from_orig'])
        from .quality_adaptation_prepare import crops_from_retained_capture
        reconstructed = crops_from_retained_capture(dict(query_rgb_crop=crops['query_rgb'].transpose(2, 0, 1)[None],
            template_rgb=crops['template_rgb'].transpose(2, 0, 1)[None], rendered_depth_mm=actual['rendered_depth_mm'],
            rendered_mask=actual['rendered_mask'], crop_k=actual['crop_k'],
            current_crop_camera_from_object_mm=actual['current_crop_camera_from_object_mm']), preserve_float_rgb=True)
        if any(not np.array_equal(reconstructed[k], crops[k]) for k in CROP_KEYS):
            raise ValueError('derived crop geometry differs from retained depth/pose in millimetres')
        if not np.array_equal(crops['query_K'], actual['crop_k']) or not np.array_equal(actual['native_k'], captured['native_calibration']):
            raise ValueError('crop/native calibration differs from retained geometry')
        check = captured.get('sampling_map', {})
        if check.get('integer_centers') != SIZE**2 or not 0 <= check.get('max_native_pixel_difference', -1) < .001:
            raise ValueError('selected iteration lacks a valid sampling map receipt')
        material['crops'].append(dict(id=captured['id'], **captured['crops']))
        material['geometry'].append(dict(id=captured['id'], **captured['capture']))
        if extracted['status'] == 'available':
            _keys(extracted, ('id', 'status', 'crops', 'query_rgb_sha256', 'features'), 'extraction row')
            if extracted['crops'] != captured['crops'] or extracted['query_rgb_sha256'] != captured['query_rgb_sha256']:
                raise ValueError('extractor crop receipt differs from retained input')
            values = load_npz(descriptor(features, extracted['features'], verify=True), FEATURE_KEYS)
            validate_features(values)
            if any(not np.array_equal(values[k], crops[k]) for k in ('template_mask', 'template_xyz_m', 'query_K')):
                raise ValueError('extracted geometry differs from captured crop')
            material['features'].append(dict(id=captured['id'], **extracted['features']))
            material['parity'].append(dict(id=captured['id'], **_check_parity(actual, values, crops['template_mask'])))
    material['available'] = len(material['features'])
    if material['available'] == 0:
        raise ValueError('no extracted crop with frozen prediction parity; cannot seal')
    return material


@_exclusive_stage
def seal_features(root, features, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    root, features = Path(root), Path(features)
    _remaining(root, features)
    material = _seal_material(root, features)
    if (root / 'prelabel-seal.json').exists():
        raise FileExistsError('prelabel seal is immutable')
    write_json(root, 'prelabel-seal.json', material)
    return material


def verify_prelabel_seal(root, features):
    seal = read_json(root / 'prelabel-seal.json')
    if seal != _seal_material(root, features):
        raise ValueError('prelabel seal changed: input/capture/crops/geometry/extraction/parity differ')
    return seal


def _read_label_plan(path, expected_sha256):
    if sha256(path) != require_hash(expected_sha256):
        raise ValueError('separate label plan SHA-256 mismatch')
    doc = read_json(path)
    _keys(doc, ('schema', 'recipe', 'split_manifest', 'generations'), 'label plan')
    if doc['schema'] != LABEL_SCHEMA or not isinstance(doc['generations'], list) or len(doc['generations']) != 3:
        raise ValueError('label plan requires exact three separately pinned synthetic generations')
    generations = {}
    for row, expected in zip(doc['generations'], SEQUENCES):
        _keys(row, ('sequence_id', 'generation'), 'generation row')
        if row['sequence_id'] != expected[2]:
            raise ValueError('generation sequence differs from input plan')
        generations[row['sequence_id']] = pinned_file(row['generation'])
    return doc, pinned_file(doc['recipe']), pinned_file(doc['split_manifest']), generations


@_exclusive_stage
def labels(root, features, label_plan_path, label_plan_sha256, training_source_root,
           training_source_manifest, *, cloud_execution=False):
    cloud_guard(cloud_execution)
    root, features = Path(root), Path(features)
    _remaining(root, features)
    seal = verify_prelabel_seal(root, features)  # BEFORE even opening label plan.
    plan, captures = _read_capture(root)
    _, recipe_path, split_path, generations = _read_label_plan(Path(label_plan_path), label_plan_sha256)
    verify_training_root(training_source_root, training_source_manifest)
    from .quality_ranker_synthetic import validate_recipe
    from .quality_adaptation_prepare import project_supervision
    from .quality_adaptation_smoke import label_pose, render_query_labels, validate_generation_context
    from .quality_gotrack import TexturedRenderer
    from .quality_runner import read_input
    extracted = {r['id']: r for r in read_json(features / 'index.json')['rows']}
    inventory = []
    for source in captures:
        item = dict(_base(source), status='unavailable', query_rgb_sha256=source['query_rgb_sha256'],
            failure_reason=source.get('failure_reason') or extracted[source['id']].get('failure_reason') or 'label_not_processed')
        if item['query_rgb_sha256'] is None:
            item['unavailable_provenance'] = source['unavailable_provenance']
        inventory.append(item)

    def persist():
        write_json(features, 'labels.json', dict(schema='correspondence_label_inventory_v1',
            label_origin='synthetic_renderer_training_only', rows=inventory + [plan['sealed_test']]))

    persist()
    try:
        for ordinal, source in enumerate(captures):
            _remaining(root, features)
            if source['status'] != 'available' or extracted[source['id']]['status'] != 'available':
                continue
            renderer = None
            try:
                physical = next(r for r in plan['rows'] if source['id'].startswith(r['id'] + ':iteration'))
                bundle = pinned_file(physical['input']).parent
                manifest = read_input(bundle)
                context = validate_recipe(recipe_path, split_path, pinned_file(physical['mesh']),
                    physical['sequence_id'], frames=24, width=640, height=480, fps=10.,
                    mesh_units='metres', observationally_ambiguous=True)
                validate_generation_context(context, read_json(generations[physical['sequence_id']]),
                    manifest, physical, sha256(split_path))
                crops = load_npz(descriptor(root, source['crops'], verify=True), CROP_KEYS)
                geometry = load_npz(descriptor(root, source['capture'], verify=True), GEOMETRY_KEYS)
                renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                    unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                native_pose_m = label_pose(renderer.vertices_m, physical)
                query_pose_m, depth_m, support = render_query_labels(renderer, object_id=manifest['object_id'],
                    crop_k=crops['query_K'], crop_from_orig=geometry['crop_from_orig'], native_pose=native_pose_m)
                target = project_supervision(template_xyz_m=crops['template_xyz_m'], template_mask=crops['template_mask'],
                    query_K=crops['query_K'], query_T_cam_from_object=query_pose_m,
                    observed_depth_m=depth_m, observed_object_mask=support, depth_tolerance_m=.002)
                inventory[ordinal] = dict(_base(source), status='available', query_rgb_sha256=source['query_rgb_sha256'],
                    labels=_save_npz(features, f'labels-{ordinal:02d}.npz', target))
            except Exception as exc:
                inventory[ordinal]['failure_reason'] = f'label_failure: {type(exc).__name__}: {exc}'[:1024]
            finally:
                try:
                    if renderer is not None:
                        renderer.close()
                finally:
                    persist()
    finally:
        persist()
        write_json(features, 'label-stage.json', dict(status='labels_persisted_assembly_pending',
            test_payload_reads=0, labels_entered_inference=False))
    _run_historical(training_source_root, training_source_manifest, 'bench.quality_adaptation_manifest',
        ['--extraction', features.resolve(), '--labels', (features / 'labels.json').resolve(),
         '--output', (features / 'manifest.json').resolve()], timeout=min(60, _remaining(root, features)))
    ds = load_dataset(features / 'manifest.json')
    write_json(features, 'label-receipt.json', dict(schema='gotrack_refinement_state_labels_receipt_v1',
        prelabel_seal_sha256=sha256(root / 'prelabel-seal.json'), label_plan_sha256=label_plan_sha256,
        label_inventory_sha256=sha256(features / 'labels.json'), manifest_sha256=sha256(features / 'manifest.json'),
        label_artifacts=[dict(id=r['id'], **r['labels']) for r in inventory if r['status'] == 'available'],
        test_payload_reads=0, labels_entered_inference=False, independent_quality_claim=False,
        available=sum(r['status'] == 'available' for r in inventory), parity_rows=len(seal['parity'])))
    _remaining(root, features)
    return dict(manifest_sha256=sha256(features / 'manifest.json'), rows=len(ds.document['rows']))


@_exclusive_stage
def train_stage(root, features, config, config_sha256, output, training_source_root,
                training_source_manifest, *, cloud_execution=False, resume=False,
                stop_after_steps=None, export_output=None):
    cloud_guard(cloud_execution)
    root, features, config, output = map(Path, (root, features, config, output))
    remaining = _remaining(root, features, output)
    verify_prelabel_seal(root, features)
    if sha256(config) != require_hash(config_sha256):
        raise ValueError('training configuration SHA-256 mismatch')
    from .quality_adaptation_train import TrainConfig, read_config
    from dataclasses import asdict
    cfg = read_config(config)
    # Numerical behavior and optimizer/loss match the existing trainer exactly.
    expected = asdict(TrainConfig())
    allowed = {'epochs': 1, 'max_steps': 16, 'max_seconds': 1200., 'max_total_seconds': 3600.,
        'max_output_bytes': LIMITS['max_run_bytes'], 'max_checkpoint_bytes': 320 * 1024**2,
        'checkpoint_every_steps': 8}
    expected.update(allowed)
    if asdict(cfg) != expected:
        raise ValueError('training config differs from the predeclared released-init 16-update recipe')
    receipt = read_json(features / 'label-receipt.json')
    if (receipt.get('prelabel_seal_sha256') != sha256(root / 'prelabel-seal.json')
            or receipt.get('label_inventory_sha256') != sha256(features / 'labels.json')
            or receipt.get('manifest_sha256') != sha256(features / 'manifest.json')
            or receipt.get('test_payload_reads') != 0 or receipt.get('labels_entered_inference') is not False):
        raise ValueError('label receipt differs from sealed inventory/manifest')
    ds = load_dataset(features / 'manifest.json')
    plan, _ = _read_capture(root)
    expected_rows = state_rows(plan)
    if ([r['id'] for r in ds.document['rows']] != [r['id'] for r in expected_rows] + [plan['sealed_test']['id']]
            or ds.document['rows'][-1] != plan['sealed_test']):
        raise ValueError('training requires exact full 41-row inventory without substitution')
    inventory = read_json(features / 'labels.json')
    extracted = {r['id']: r for r in read_json(features / 'index.json')['rows']}
    if inventory.get('label_origin') != 'synthetic_renderer_training_only' or len(inventory.get('rows', [])) != 41:
        raise ValueError('label inventory schema/origin differs')
    for row, label_row, expected_row in zip(ds.document['rows'][:-1], inventory['rows'][:-1], expected_rows):
        if any(row.get(k) != expected_row[k] for k in ('id', 'object_id', 'split', 'sequence_id', 'capture_group')):
            raise ValueError('training row identity differs from prospective split/frame binding')
        if {k: v for k, v in row.items() if k != 'features'} != label_row:
            raise ValueError('assembled training row differs from authenticated label inventory')
        if row['status'] == 'available' and row['features'] != extracted[row['id']].get('features'):
            raise ValueError('training feature descriptor differs from actual extraction receipt')
    artifacts = [dict(id=r['id'], **r['labels']) for r in inventory['rows'] if r['status'] == 'available']
    if receipt.get('label_artifacts') != artifacts or inventory['rows'][-1] != plan['sealed_test']:
        raise ValueError('label payload descriptors or TEST metadata differ from label receipt')
    if type(resume) is not bool or (stop_after_steps is not None and
            (type(stop_after_steps) is not int or not 1 <= stop_after_steps <= 16)):
        raise ValueError('resume must be boolean; stop-after-steps must be 1..16')
    if resume or export_output is not None:
        run_receipt = read_json(output / 'refinement-state-run-receipt.json')
        if (run_receipt.get('dataset_sha256') != ds.identity
                or run_receipt.get('config_sha256') != canonical_hash(read_json(config))
                or run_receipt.get('label_receipt_sha256') != sha256(features / 'label-receipt.json')):
            raise ValueError('resume/export run differs from authenticated dataset/config/labels')
        for pin in run_receipt['files']:
            path = _relative(output, pin['path'])
            if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
                raise ValueError('resume/export checkpoint or run metadata changed')
    if export_output is not None:
        destination = Path(export_output)
        if destination.resolve().parent != output.resolve() or destination.exists():
            raise ValueError('export must be a new file inside the authenticated run')
        metadata = read_json(output / 'preflight.json')
        if metadata.get('config_sha256') != canonical_hash(read_json(config)) or metadata.get('data', {}).get('dataset_sha256') != ds.identity:
            raise ValueError('export run dataset/config metadata differs')
        command = ['export', '--run', output.resolve(), '--output', destination.resolve(), '--cloud-execution']
    else:
        command = ['resume' if resume else 'train', '--manifest', (features / 'manifest.json').resolve(),
            '--config', config.resolve(), '--output', output.resolve(), '--cloud-execution']
        if stop_after_steps is not None:
            command += ['--stop-after-steps', stop_after_steps]
    result = _run_historical(training_source_root, training_source_manifest, 'bench.quality_adaptation_train',
        command, timeout=min(LIMITS['max_train_seconds'], remaining),
        accepted_exit_codes=(0,) if export_output is not None else (0, 3))
    if export_output is None:
        status = read_json(output / 'status.json')
        write_json(output, 'refinement-state-run-receipt.json', dict(
            dataset_sha256=ds.identity, config_sha256=canonical_hash(read_json(config)),
            label_receipt_sha256=sha256(features / 'label-receipt.json'),
            actual_optimizer_steps=status['steps'], status=status['status'],
            available_train_rows=len(ds.rows('train')), available_used_val_rows=len(ds.rows('val')),
            controlled_quality_claim=False,
            files=[_artifact(output, path) for path in sorted(output.iterdir())
                if path.is_file() and path.name != 'refinement-state-run-receipt.json']))
    _remaining(root, features, output)
    return result


def _supervise(args, argv):
    """Mandatory CLI watchdog; native calls execute only in a bounded child.

    A stop request is sampled every 250ms and process-tree cleanup may require
    additional time. This bounds execution, not provider startup or billing.
    """
    cloud_guard(args.cloud_execution)
    import psutil
    stage = args.command
    target = Path(args.output if stage == 'capture' else args.root).resolve()
    monitor = target.parent / (target.name + '-monitor-' + uuid.uuid4().hex)
    monitor.mkdir(parents=True, exist_ok=False)
    began = time.time()
    if stage == 'capture':
        raw = args.plan.read_bytes()
        if hashlib.sha256(raw).hexdigest() != require_hash(args.plan_sha256):
            raise ValueError('capture supervisor input plan hash mismatch')
        plan = validate_plan(inference._loads_unique(raw, 'input plan'))
        timeout = LIMITS['max_capture_seconds']
    else:
        plan = validate_plan(read_json(target / 'plan.json'))
        timeout = _remaining(target, Path(args.features), *([Path(args.output)] if stage in ('train', 'export') else []))
        if stage in ('extract', 'train', 'export'):
            timeout = min(timeout, LIMITS['max_extract_seconds' if stage == 'extract' else 'max_train_seconds'])
    requested = state_rows(plan)
    write_json(monitor, 'requested-inventory.json', dict(stage=stage, rows=requested, test_payload_reads=0))
    roots = [target, *target.parent.glob(target.name + '-monitor-*')]
    if stage != 'capture':
        roots.append(Path(args.features).resolve())
    if stage in ('train', 'export'):
        roots.append(Path(args.output).resolve())
    roots = [path for path in roots if not any(path != other and path.is_relative_to(other) for other in roots)]
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    if stage == 'capture':
        env['VISUALIZEIT_REFINEMENT_WORKFLOW_STARTED'] = str(began)
    process, descendants, reason = None, [], None
    peak_rss = 0
    try:
        with (monitor / 'stage.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-B', '-m',
                'bench.quality_adaptation_refinement_states', *argv, '--_worker'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    descendants = parent.children(recursive=True)
                    peak_rss = max(peak_rss, sum(p.memory_info().rss for p in [parent, *descendants]))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                used = sum(tree_bytes(path) for path in roots if path.exists())
                if time.time() - began >= timeout:
                    reason = 'stage_or_aggregate_runtime_limit'
                elif used > LIMITS['max_private_bytes']:
                    reason = 'private_aggregate_storage_limit'
                elif tree_bytes(ROOT / '.cache/model-quality') > LIMITS['model_cache_bytes']:
                    reason = 'model_cache_storage_limit'
                elif shutil.disk_usage(monitor).free < LIMITS['disk_reserve_bytes']:
                    reason = 'free_disk_reserve_limit'
                elif stage != 'capture' and tree_bytes(Path(args.features)) > LIMITS['max_feature_bytes']:
                    reason = 'feature_storage_limit'
                elif stage in ('train', 'export') and Path(args.output).exists() and tree_bytes(Path(args.output)) > LIMITS['max_run_bytes']:
                    reason = 'training_storage_limit'
                if reason:
                    break
                time.sleep(.25)
            if reason is None and process.wait() != 0:
                reason = 'stage_exit_nonzero'
    except BaseException as exc:
        reason = f'supervision_exception: {type(exc).__name__}: {exc}'[:1024]
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            for child in descendants:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            _, alive = psutil.wait_procs(descendants, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=2)
            if alive:
                reason = 'process_tree_cleanup_incomplete'
        if descendants:
            _, alive = psutil.wait_procs(descendants, timeout=0)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=2)
            if alive:
                reason = 'process_tree_cleanup_incomplete'
        lock = ROOT / '.cache/model-quality/.refinement-state-stage.lock'
        if (process is not None and reason != 'process_tree_cleanup_incomplete'
                and lock.is_file() and read_json(lock).get('pid') == process.pid):
            lock.unlink()  # Only after this supervisor's own worker has exited.
        # A hard stop may prevent the worker's finally block. Preserve every
        # prescribed state in the supervisor, joining only exact saved IDs.
        inventory = [dict(row) for row in requested]
        saved_path = (target / 'capture-index.json' if stage == 'capture' else
            Path(args.features) / ('index.json' if stage == 'extract' else 'labels.json'))
        saved = {}
        if saved_path.is_file() and not saved_path.is_symlink():
            try:
                saved_rows = read_json(saved_path).get('rows', [])
                saved = {row['id']: row for row in saved_rows if isinstance(row, dict) and row.get('split') != 'test'}
            except (ValueError, OSError, KeyError, TypeError):
                saved = {}
        for row in inventory:
            persisted = saved.get(row['id'])
            if persisted is not None and persisted.get('status') in ('available', 'unavailable'):
                row.update(status=persisted['status'], failure_reason=persisted.get('failure_reason'),
                    evidence='worker_persisted_row; stage status is separate')
            else:
                row['failure_reason'] = reason or 'stage_has_no_persisted_per_state_evidence'
        write_json(monitor, 'state-inventory.json', dict(stage=stage, rows=inventory, test_payload_reads=0))
        write_json(monitor, 'supervision.json', dict(stage=stage, complete=reason is None,
            failure_reason=reason, elapsed_seconds=time.time() - began,
            sampled_peak_process_tree_rss_bytes=peak_rss or None, billing_stopped=None))
    if reason:
        raise RuntimeError(f'{stage} stopped: {reason}; inspect {monitor}')
    if stage != 'capture':
        _remaining(target, Path(args.features), *([Path(args.output)] if stage in ('train', 'export') else []))
    return dict(stage=stage, complete=True, monitor=str(monitor))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('capture', 'extract', 'seal', 'labels', 'train', 'export'):
        p = sub.add_parser(command)
        p.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
        p.add_argument('--cloud-execution', action='store_true')
        if command == 'capture':
            p.add_argument('--plan', type=Path, required=True)
            p.add_argument('--plan-sha256', required=True)
        else:
            p.add_argument('--root', type=Path, required=True)
            p.add_argument('--features', type=Path, required=True)
        if command in ('extract', 'labels', 'train', 'export'):
            p.add_argument('--training-source-root', type=Path, required=True)
            p.add_argument('--training-source-manifest', type=Path, required=True)
        if command in ('capture', 'train', 'export'):
            p.add_argument('--output', type=Path, required=True)
        if command == 'labels':
            p.add_argument('--label-plan', type=Path, required=True)
            p.add_argument('--label-plan-sha256', required=True)
        if command in ('train', 'export'):
            p.add_argument('--config', type=Path, required=True)
            p.add_argument('--config-sha256', required=True)
            if command == 'train':
                p.add_argument('--resume', action='store_true')
                p.add_argument('--stop-after-steps', type=int)
            else:
                p.add_argument('--export-output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args._worker:
        print(json.dumps(_supervise(args, argv), allow_nan=False))
        return
    common = dict(cloud_execution=args.cloud_execution)
    if args.command == 'capture':
        result = capture(args.plan, args.plan_sha256, args.output, **common)
    elif args.command == 'extract':
        result = extract_stage(args.root, args.features, args.training_source_root, args.training_source_manifest, **common)
    elif args.command == 'seal':
        result = seal_features(args.root, args.features, **common)
    elif args.command == 'labels':
        result = labels(args.root, args.features, args.label_plan, args.label_plan_sha256,
            args.training_source_root, args.training_source_manifest, **common)
    else:
        result = train_stage(args.root, args.features, args.config, args.config_sha256,
            args.output, args.training_source_root, args.training_source_manifest,
            resume=getattr(args, 'resume', False), stop_after_steps=getattr(args, 'stop_after_steps', None),
            export_output=getattr(args, 'export_output', None), **common)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    # Preserve exact capability/sink class identity under python -m.
    from .quality_adaptation_refinement_states import main as imported_main
    imported_main()
