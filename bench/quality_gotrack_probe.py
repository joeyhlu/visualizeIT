"""Authenticated same-input ordinary GoTrack observer; no neural imports at import.

This is development diagnosis with adapted-trajectory seed replay and a frozen
same-input counterfactual. It measures no independent pose accuracy. OpenCV's
internal RANSAC samples and LM steps are unavailable; only returned boundaries
are observed. Existing inference validation authenticates generation metadata,
but neither reference pose nor evaluation sidecars are opened here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shutil
import stat
import sys
import time

import numpy as np

from . import quality_gotrack_inference as inference

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'gotrack_same_input_probe_plan_v1'
INVENTORY_SHA256 = 'c0e09a741a839680f45a993fdbec102725ee7fb3a4e4db00ff3ade1681e0a52f'
DESIGN_SHA256 = 'dd341fc00821e3cf4ba5923ab2f1f16ca67abb6f135625c81800c1226adc8e98'
LIMITS = dict(max_seconds=1800, max_raw_capture_bytes=256 * 1024**2,
              max_summary_bytes=8 * 1024**2, model_cache_bytes=8 * 1024**3)
CONDITIONS = ('frozen_base', 'expanded_head')
MODES = ('disabled_control', 'enabled_diagnostic')
_ISSUER = object()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    return inference._loads_unique(data, 'probe JSON')


def _encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode('utf-8') + b'\n'


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name} has unexpected or missing fields')


def pinned_file(pin, *, source=False):
    _keys(pin, ('path', 'bytes', 'sha256'), 'file pin')
    if type(pin['bytes']) is not int or not 0 < pin['bytes'] <= 8 * 1024**3:
        raise ValueError('invalid pinned byte count')
    inference._require_sha256(pin['sha256'], 'file pin')
    raw = pin['path']
    if not isinstance(raw, str) or '\\' in raw or '..' in PurePosixPath(raw).parts:
        raise ValueError('pin path must be normalized')
    candidate = Path(raw)
    if source:
        if candidate.is_absolute():
            raise ValueError('source pins must be repository relative')
        candidate = ROOT / candidate
    elif not candidate.is_absolute():
        candidate = ROOT / candidate
    elif not raw.startswith('/teamspace/studios/this_studio/'):
        raise ValueError('data pin is outside the reviewed cloud root')
    if inference._has_symlink_component(candidate):
        raise ValueError('pin path cannot contain symlinks')
    path = candidate.resolve(strict=True)
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size != pin['bytes']:
        raise ValueError('pin file type/size mismatch')
    if inference._sha256_file(path) != pin['sha256'].lower():
        raise ValueError('pin SHA-256 mismatch')
    return path


def validate_plan(plan):
    _keys(plan, ('schema', 'runtime_contract', 'probe_sources', 'seed_inventory',
                 'design', 'head', 'cases', 'limits'), 'probe plan')
    if plan['schema'] != SCHEMA or plan['limits'] != LIMITS:
        raise ValueError('probe schema/limits differ from the reviewed design')
    if plan['seed_inventory']['sha256'] != INVENTORY_SHA256 or plan['design']['sha256'] != DESIGN_SHA256:
        raise ValueError('probe inventory/design identity differs from the reviewed pins')
    if plan['head']['sha256'] != inference.EXPANDED_EXPORT_SHA256 or plan['head']['bytes'] != inference.EXPANDED_EXPORT_BYTES:
        raise ValueError('probe head is not the expanded artifact')
    sources = plan['probe_sources']
    if not isinstance(sources, list) or len(sources) != 1 or sources[0]['path'] != 'bench/quality_gotrack_probe.py':
        raise ValueError('probe source closure must contain exactly the driver')
    if plan['runtime_contract']['path'] != 'bench/runtime/gotrack-inference-probe-v1.json':
        raise ValueError('probe requires its separate runtime contract')
    inventory = _json(pinned_file(plan['seed_inventory']).read_bytes())
    pinned_file(plan['design'])
    expected = inventory['cases']
    cases = plan['cases']
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError('probe requires exactly six fixed cases')
    for case, seed in zip(cases, expected):
        _keys(case, ('case_id', 'sequence_id', 'frame_id', 'timestamp_s', 'input',
                     'video', 'mesh', 'mask_index', 'mask', 'seed_camera_from_object_m'), 'probe case')
        for key in ('case_id', 'sequence_id', 'frame_id', 'timestamp_s', 'seed_camera_from_object_m'):
            if case[key] != seed[key]:
                raise ValueError(f'probe case differs from authenticated inventory: {key}')
        if type(case['frame_id']) is not int or type(case['timestamp_s']) not in (int, float):
            raise ValueError('case frame/timestamp types must be numeric and nonboolean')
        from .quality_contract import checked_pose
        checked_pose(np.asarray(case['seed_camera_from_object_m'], np.float64))
        for key, suffix in (('input', 'input.json'), ('video', 'synthetic.mp4'),
                            ('mesh', 'object.glb'), ('mask_index', 'results.json'),
                            ('mask', f"{case['frame_id']}.png")):
            pin = case[key]
            _keys(pin, ('path', 'bytes', 'sha256'), 'case input pin')
            path = PurePosixPath(pin['path'])
            if path.name != suffix or any(token in str(path).lower() for token in
                    ('reference', 'evaluation', 'generation', 'obj_000007', 'hot3d7', 'poses.json')):
                raise ValueError('annotation/test/generation input is forbidden')
    return plan


@dataclass(frozen=True, slots=True)
class InferenceObserverCapability:
    """Issued only after plan/source/data preflight; exact sink and runtime bound."""
    _issuer: object
    _plan_bytes: bytes
    _plan_sha256: str
    _source_pins: tuple
    _sink: object

    def validate(self, sink, geometry):
        if (self._issuer is not _ISSUER or sink is not self._sink or
                type(sink) is not ProbeCapture or geometry != inference.PROFILE or
                _sha(self._plan_bytes) != self._plan_sha256):
            raise ValueError('invalid/mixed inference observer capability')
        for text in self._source_pins:
            pinned_file(_json(text), source=True)
        inference.verify_source_compatibility(_json(pinned_file(
            _json(self._source_pins[0]), source=True).read_bytes()))


@dataclass(frozen=True, slots=True)
class _VerifiedPreflight:
    issuer: object
    plan_bytes: bytes


def _issue(plan_bytes, plan_sha256, sink, verified):
    if type(verified) is not _VerifiedPreflight or verified.issuer is not _ISSUER or verified.plan_bytes != plan_bytes:
        raise ValueError('observer issuance requires completed exact inference/data preflight')
    if _sha(plan_bytes) != plan_sha256:
        raise ValueError('plan SHA-256 mismatch')
    plan = validate_plan(_json(plan_bytes))
    pins = [plan['runtime_contract'], *plan['probe_sources']]
    for pin in pins:
        pinned_file(pin, source=True)
    return InferenceObserverCapability(_ISSUER, plan_bytes, plan_sha256,
                                      tuple(_encoded(pin) for pin in pins), sink)


def candidate_identity(candidate):
    if candidate is None:
        return None
    return {name: array_identity(getattr(candidate, attribute)) for name, attribute in
            (('pose', 'pose'), ('points', 'points_object_m'), ('pixels', 'pixels_image'), ('weights', 'weights'))}


def array_identity(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=_sha(array.tobytes()))


def conditioning_summary(points):
    """Centered 3D SVD in mm: geometric spread only, not PnP uncertainty."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('conditioning points must be finite Nx3 millimetres')
    if len(points) < 3:
        return dict(count=len(points), singular_values_mm=None, rank=None)
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    tolerance = singular[0] * max(points.shape) * np.finfo(np.float64).eps
    return dict(count=len(points), singular_values_mm=singular.tolist(),
                rank=int(np.count_nonzero(singular > tolerance)))


def reprojection_summary(points, targets, camera, rvec, tvec):
    """Analytical pinhole residual (crop pixels), without solver calls."""
    from .vision import cv2
    points, targets = np.asarray(points, np.float64), np.asarray(targets, np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or targets.shape != (len(points), 2):
        raise ValueError('reprojection shape mismatch')
    xyz = points @ cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0].T + np.asarray(tvec).reshape(3)
    projected = xyz @ np.asarray(camera).T
    valid = np.isfinite(projected).all(1) & np.isfinite(targets).all(1) & (xyz[:, 2] > 0)
    residual = np.linalg.norm(projected[valid, :2] / projected[valid, 2:] - targets[valid], axis=1)
    return dict(count=len(points), front_finite=int(valid.sum()), rms_px=(float(np.sqrt(np.mean(residual**2))) if len(residual) else None))


class OutputBudget:
    def __init__(self, root, limits):
        self.root = Path(root)
        self.raw_remaining = limits['max_raw_capture_bytes']
        self.raw_disk_remaining = limits['max_raw_capture_bytes']
        self.summary_remaining = limits['max_summary_bytes']

    def write(self, name, data, *, raw=False, raw_bytes=None):
        if Path(name).name != name:
            raise ValueError('output must be a leaf name')
        field = 'raw_remaining' if raw else 'summary_remaining'
        charge = len(data) if not raw or raw_bytes is None else raw_bytes
        if type(charge) is not int or charge < 0 or charge > getattr(self, field) or (raw and len(data) > self.raw_disk_remaining):
            raise ValueError('probe output budget exceeded')
        path = self.root / name
        with path.open('xb') as stream:
            stream.write(data)
        setattr(self, field, getattr(self, field) - charge)
        if raw:
            self.raw_disk_remaining -= len(data)
        return dict(path=name, bytes=len(data), sha256=_sha(data))


class ProbeCapture:
    """Release packet payloads after bounded compressed persistence.

    Uses the RetainedCapture hook protocol, extended to all five iterations and
    solver boundaries. Keep only sampled geometry until its paired solver
    result, then release it. No experiment-specific sink/training lock is used.
    """
    def __init__(self, output, prefix):
        self.output, self.prefix = output, prefix
        self.failure = None
        self.events = {}
        self.rows = [dict(iteration=i, status='unrun', reason='refinement_not_reached') for i in range(5)]
        self._geometry = {}

    def _disable_capture(self, reason):
        if self.failure is None:
            self.failure = str(reason)[:1024]

    def should_capture(self, context, kind, metadata):
        if self.failure is not None:
            return False
        iteration = metadata.get('iteration')
        if type(iteration) is not int or iteration not in range(5) or kind not in ('trace', 'retained', 'pre_lm', 'result'):
            raise ValueError('unexpected probe event')
        return True

    def preflight_nbytes(self, context, kind, metadata, nbytes):
        if type(nbytes) is not int or not 0 < nbytes <= min(32 * 1024**2, self.output.raw_remaining):
            self._disable_capture('capture byte preflight budget refused')
            return False
        return True

    def preflight(self, context, kind, metadata, descriptors):
        if not self.should_capture(context, kind, metadata):
            return False
        total = 0
        for descriptor in descriptors.values():
            _keys(descriptor, ('shape', 'dtype', 'nbytes'), 'array descriptor')
            nbytes = descriptor['nbytes']
            if type(nbytes) is not int or nbytes < 0:
                raise ValueError('malformed capture descriptor')
            if 'object' in descriptor['dtype']:
                raise ValueError('object capture arrays are forbidden')
            total += nbytes
        if total == 0 and kind == 'result':
            return True  # Unavailable solvers legitimately return empty index arrays.
        return self.preflight_nbytes(context, kind, metadata, total)

    def __call__(self, packet):
        kind, meta, arrays = packet['kind'], packet['metadata'], packet['arrays']
        if not self.should_capture(packet['context'], kind, meta):
            raise ValueError('unexpected disabled packet')
        iteration = meta['iteration']
        key = (iteration, kind)
        if key in self.events:
            raise ValueError('duplicate probe event')
        if kind == 'trace':
            self.events[key] = dict(metadata=meta)
            self.rows[iteration].update(status='started', reason=None)
            return
        if kind == 'retained':
            required = ('sample_obj_points_mm', 'sample_target_crop_px', 'crop_k', 'sample_ids')
            if any(name not in arrays for name in required):
                raise ValueError('malformed retained packet')
            self._geometry[iteration] = {name: arrays[name] for name in required}
            summary = dict(conditioning=conditioning_summary(arrays['sample_obj_points_mm']))
        else:
            geometry = self._geometry.get(iteration)
            if geometry is None:
                raise ValueError('solver event has no retained packet')
            summary = {}
            ids = np.asarray(arrays['ransac_inlier_ids'])
            sample_ids = geometry['sample_ids']
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(sample_ids)):
                raise ValueError('invalid RANSAC sample indices')
            if not np.array_equal(arrays['retained_inlier_ids'], sample_ids[ids]):
                raise ValueError('RANSAC indices do not map into retained correspondences')
            if kind == 'pre_lm':
                summary['reprojection'] = reprojection_summary(geometry['sample_obj_points_mm'][ids],
                    geometry['sample_target_crop_px'][ids], geometry['crop_k'], arrays['ransac_rvec'], arrays['ransac_tvec_mm'])
            if kind == 'result':
                if meta.get('fit_state') == 'refined':
                    if (iteration, 'pre_lm') not in self.events:
                        raise ValueError('successful solver result has no pre-LM snapshot')
                    summary['reprojection'] = reprojection_summary(geometry['sample_obj_points_mm'][ids],
                        geometry['sample_target_crop_px'][ids], geometry['crop_k'], arrays['refined_rvec'], arrays['refined_tvec_mm'])
                self.rows[iteration].update(status=meta['fit_state'], reason=None)
                self._geometry.pop(iteration, None)
        # A bounded in-memory ZIP avoids a partially written artifact on refusal.
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        raw_bytes = sum(value.nbytes for value in arrays.values())
        artifact = self.output.write(f'{self.prefix}-i{iteration}-{kind}.npz', buffer.getvalue(), raw=True, raw_bytes=raw_bytes)
        artifact['uncompressed_array_bytes'] = raw_bytes
        self.events[key] = dict(metadata=meta, artifact=artifact, summary=summary)

    def finish(self, failure=None):
        if not failure:
            for i in range(5):
                result = self.events.get((i, 'result'))
                if result is None:
                    if i == 0 or (i - 1, 'result') in self.events and self.events[i - 1, 'result']['metadata']['fit_state'] == 'refined':
                        self._disable_capture('missing required iteration/result events')
                    break
                if (i, 'retained') not in self.events or (i, 'trace') not in self.events:
                    self._disable_capture('result missing retained/trace event')
        for row in self.rows:
            if row['status'] == 'started':
                row.update(status='failed', reason=failure or self.failure or 'missing_result_event')
                self._disable_capture(row['reason'])
            elif row['status'] == 'unrun':
                row['reason'] = failure or self.failure or 'prior_iteration_terminated_refinement'
        self._geometry.clear()
        return dict(iterations=self.rows, capture_failure=self.failure,
                    events=[dict(iteration=i, kind=kind, **value) for (i, kind), value in sorted(self.events.items())])


def _cloud_guard(explicit):
    if not explicit or os.environ.get('VISUALIZEIT_CLOUD') != '1' or platform.system() != 'Linux':
        raise ValueError('probe requires --cloud-execution, VISUALIZEIT_CLOUD=1 and Linux')


def _preflight(plan):
    contexts, inputs = {}, {}
    contract_path = pinned_file(plan['runtime_contract'], source=True)
    contract = _json(contract_path.read_bytes())
    inference._validate_contract(contract)
    inference.verify_source_compatibility(contract)
    from .quality_runner import read_input, _physical_mask_cache
    from .quality_offline import tree_bytes
    head_path = pinned_file(plan['head'])
    for pin in plan['probe_sources']:
        pinned_file(pin, source=True)
    if tree_bytes(ROOT / '.cache/model-quality') > plan['limits']['model_cache_bytes']:
        raise ValueError('model cache exceeds reviewed budget')
    for case in plan['cases']:
        accepted = [row for row in contract['accepted_inputs'] if row['sequence_id'] == case['sequence_id']]
        if len(accepted) != 1 or any(case[name]['sha256'] != accepted[0][field] for name, field in
                (('input', 'input_sha256'), ('video', 'video_sha256'), ('mesh', 'mesh_sha256'))):
            raise ValueError('case source pins differ from exact accepted-input allowlist')
        paths = {name: pinned_file(case[name]) for name in ('input', 'video', 'mesh', 'mask_index', 'mask')}
        bundle = paths['input'].parent
        if paths['video'] != bundle / 'synthetic.mp4' or paths['mesh'] != bundle / 'object.glb':
            raise ValueError('case video/mesh must belong to the exact inference bundle')
        manifest = read_input(bundle)
        if manifest['sequence_id'] != case['sequence_id']:
            raise ValueError('case sequence differs from inference input')
        timeline = [row for row in manifest['timeline'] if row['frame_id'] == case['frame_id']]
        if len(timeline) != 1 or timeline[0]['timestamp_s'] != case['timestamp_s']:
            raise ValueError('case timestamp differs from pinned input')
        masks, by_id = _physical_mask_cache(manifest, paths['mask_index'].parent)
        mask_row = by_id[case['frame_id']]
        if mask_row.get('mask_state') != 'available' or (paths['mask_index'].parent / mask_row['path']).resolve() != paths['mask']:
            raise ValueError('case mask differs from cached automatic observation')
        if masks.get('diagnostic_control') or masks.get('automatic') is False:
            raise ValueError('probe requires observed automatic masks')
        for condition in CONDITIONS:
            options = inference.options_from_values(config_path=contract_path,
                config_sha256=plan['runtime_contract']['sha256'], cloud_execution=True,
                head_path=head_path if condition == 'expanded_head' else None,
                head_sha256=plan['head']['sha256'] if condition == 'expanded_head' else None)
            context = inference.prepare_inference(options, bundle, manifest)
            inference.validate_physical_mask_producer(context, masks)
            contexts[case['case_id'], condition] = context
        inputs[case['case_id']] = (bundle, manifest, paths['mask'])
    return contexts, inputs


def _inventory(plan):
    return [dict(case_id=case['case_id'], condition=condition, capture_mode=mode,
                 status='unrun', failure_reason='not_reached', cleanup='not_created',
                 iterations=[dict(iteration=i, status='unrun', reason='trajectory_not_reached') for i in range(5)])
            for condition in CONDITIONS for case in plan['cases'] for mode in MODES]


def run(plan_path, plan_sha256, output, *, cloud_execution=False):
    _cloud_guard(cloud_execution)
    started = time.monotonic()
    plan_bytes = Path(plan_path).read_bytes()
    if len(plan_bytes) > 2 * 1024**2 or _sha(plan_bytes) != plan_sha256:
        raise ValueError('probe plan SHA-256/size mismatch')
    plan = validate_plan(_json(plan_bytes))
    output = Path(output)
    if output.exists():
        raise FileExistsError('probe output must be a new directory')
    if inference._has_symlink_component(output):
        raise ValueError('probe output cannot contain symlinks')
    disk_root = output.parent
    while not disk_root.exists():
        disk_root = disk_root.parent
    if shutil.disk_usage(disk_root).free < LIMITS['max_raw_capture_bytes'] + LIMITS['max_summary_bytes'] + 1024**3:
        raise ValueError('probe requires output budget plus 1 GiB free disk reserve')
    contexts, inputs = _preflight(plan)  # All exact input/source/head/mask pins before Torch.
    verified = _VerifiedPreflight(_ISSUER, plan_bytes)
    output.mkdir(parents=True, exist_ok=False)
    budget = OutputBudget(output, {**plan['limits'], 'max_summary_bytes': plan['limits']['max_summary_bytes'] - 1024**2})
    budget.write('plan.json', plan_bytes)
    rows = _inventory(plan)
    complete = False
    fatal = None
    model_cleanup = []
    runtime_flags = None
    try:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        runtime_flags = dict(torch_threads_requested=1, torch_threads_observed=torch.get_num_threads(),
                            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                            python_numpy_torch_seed=0, inference_mode=True, network_eval=True, device='cpu',
                            environment={key: os.environ.get(key) for key in ('PYOPENGL_PLATFORM', 'XFORMERS_DISABLED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VISUALIZEIT_CLOUD', 'SAM2_BUILD_CUDA')})
        if runtime_flags['torch_threads_observed'] != 1 or runtime_flags['deterministic_algorithms'] is not True:
            raise ValueError('runtime did not apply the reviewed deterministic CPU settings')
        from .quality_gotrack import load_network, GoTrackRefiner, TexturedRenderer
        from .quality_runner import read_rgb
        from .quality_contract import Frame
        from .quality_time import PHYSICAL
        from .vision import cv2
        for condition in CONDITIONS:
            network = None
            try:
                network = load_network('cpu')
                if condition == 'expanded_head':
                    inference.install_adapted_head(network, contexts[plan['cases'][0]['case_id'], condition], device='cpu')
                for case in plan['cases']:
                    bundle, manifest, mask_path = inputs[case['case_id']]
                    try:
                        if time.monotonic() - started >= plan['limits']['max_seconds']:
                            raise RuntimeError('probe time budget exhausted')
                        rgb = read_rgb(bundle, manifest, case['frame_id'])
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                            raise ValueError('decoded observed mask missing/empty/wrong shape')
                        context = contexts[case['case_id'], condition]
                        seed = np.asarray(case['seed_camera_from_object_m'], np.float64)
                        frame = Frame(case['frame_id'], rgb, context.effective_intrinsics,
                                      timestamp_s=case['timestamp_s'], clock_mode=PHYSICAL)
                    except Exception as exc:
                        for row in rows:
                            if row['case_id'] == case['case_id'] and row['condition'] == condition:
                                row.update(status='failed', failure_reason=f'case input decode: {type(exc).__name__}: {exc}'[:1024])
                                for iteration in row['iterations']:
                                    iteration['reason'] = row['failure_reason']
                                budget.write(f"row-{rows.index(row):02d}.json", _encoded(row))
                        continue
                    for mode in MODES:
                        row = next(r for r in rows if (r['case_id'], r['condition'], r['capture_mode']) == (case['case_id'], condition, mode))
                        renderer = sink = refiner = None
                        try:
                            if time.monotonic() - started >= plan['limits']['max_seconds']:
                                raise RuntimeError('probe time budget exhausted')
                            random.seed(0); np.random.seed(0); torch.manual_seed(0)
                            renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                                unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                            row['cleanup'] = 'pending'
                            if mode == 'enabled_diagnostic':
                                sink = ProbeCapture(budget, f"{condition}-{case['case_id']}")
                                capability = _issue(plan_bytes, plan_sha256, sink, verified)
                            else:
                                capability = None
                            refiner = GoTrackRefiner(network, renderer, manifest['object_id'], 'cpu',
                                inference_geometry=inference.PROFILE, chronological_capture_callback=sink,
                                inference_observer_capability=capability)
                            refiner.set_chronological_capture_context(dict(case_id=case['case_id'], condition=condition))
                            identity = dict(rgb=array_identity(rgb), mask=array_identity(mask), seed=array_identity(seed), intrinsics=array_identity(context.effective_intrinsics))
                            candidate = refiner.refine(frame, mask, seed.copy())
                            row.update(status='available' if candidate is not None else 'solver_unavailable',
                                       failure_reason=None, input_identity=identity, candidate_identity=candidate_identity(candidate))
                            if sink is None:
                                row['iterations'] = [dict(iteration=i, status=('completed_unobserved' if candidate is not None else 'unobserved'),
                                    reason=('observer_disabled_control' if candidate is not None else 'observer_disabled_solver_termination_iteration_unknown')) for i in range(5)]
                        except Exception as exc:
                            row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}'[:1024])
                        finally:
                            if sink is not None:
                                row['capture'] = sink.finish(row['failure_reason'])
                                row['iterations'] = row['capture']['iterations']
                                if sink.failure:
                                    row.update(status='failed', failure_reason=sink.failure)
                            if renderer is not None:
                                try:
                                    renderer.close(); row['cleanup'] = 'closed'
                                except Exception as exc:
                                    row.update(status='failed', cleanup='failed', failure_reason=f'renderer cleanup: {exc}'[:1024])
                            refiner = renderer = sink = None
                            budget.write(f"row-{rows.index(row):02d}.json", _encoded(row))
            finally:
                try:
                    if network is not None:
                        network.cpu()
                    model_cleanup.append(dict(condition=condition, status='released'))
                except Exception as exc:
                    model_cleanup.append(dict(condition=condition, status='failed', failure=f'{type(exc).__name__}: {exc}'[:1024]))
                    raise
                finally:
                    network = None
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
        parity = []
        for case in plan['cases']:
            identities = []
            for condition in CONDITIONS:
                pair = [r for r in rows if r['case_id'] == case['case_id'] and r['condition'] == condition]
                equal = (all(r['status'] in ('available', 'solver_unavailable') for r in pair)
                         and pair[0].get('input_identity') == pair[1].get('input_identity')
                         and pair[0].get('candidate_identity') == pair[1].get('candidate_identity'))
                parity.append(dict(case_id=case['case_id'], condition=condition, observer_parity=equal))
                identities.extend(r.get('input_identity') for r in pair)
            parity.append(dict(case_id=case['case_id'], cross_head_input_seed_equality=(None not in identities and all(value == identities[0] for value in identities))))
        complete = all(r['status'] in ('available', 'solver_unavailable') and r['cleanup'] == 'closed' for r in rows) and all(r.get('observer_parity', r.get('cross_head_input_seed_equality')) for r in parity)
    except Exception as exc:
        fatal = f'{type(exc).__name__}: {exc}'[:1024]
        parity = []
    finally:
        for row in rows:
            if row['status'] == 'unrun':
                row['failure_reason'] = fatal or 'execution_terminated_before_trajectory'
                for iteration in row['iterations']:
                    iteration['reason'] = row['failure_reason']
        receipt = dict(schema='gotrack_same_input_probe_receipt_v1', plan_sha256=plan_sha256,
            complete=complete, fatal_failure=fatal, rows=rows, parity=parity,
            elapsed_seconds=time.monotonic() - started, reference_pose_reads=0, evaluation_reads=0,
            model_cleanup=model_cleanup, device='cpu',
            runtime_flags=runtime_flags,
            time_budget_enforcement='checked at trajectory boundaries including preflight time; external watchdog required for blocking native calls',
            implicit_input_authentication='existing prepare_inference reads generation metadata only; no label sidecar',
            interpretation='adapted-trajectory seed replay / frozen same-input counterfactual; no independent accuracy claim',
            unavailable_internal_observations=['OpenCV internal RANSAC samples', 'OpenCV internal LM steps'])
        budget.write('receipt.json', _encoded(receipt))
        if complete:
            budget.write('completion.json', _encoded(dict(receipt_sha256=inference._sha256_file(output / 'receipt.json'))))
    return receipt


def supervise(plan_path, plan_sha256, output, *, cloud_execution=False):
    """Bound the child, including blocking native calls, using the existing runner.

    The monitor samples every 250 ms; process termination can take additional
    cleanup time. Its timeout is a stop request, not a billing-stop claim.
    """
    _cloud_guard(cloud_execution)
    started = time.monotonic()
    data = Path(plan_path).read_bytes()
    if len(data) > 2 * 1024**2 or _sha(data) != plan_sha256:
        raise ValueError('probe plan SHA-256/size mismatch')
    plan = validate_plan(_json(data))
    for pin in (plan['runtime_contract'], *plan['probe_sources']):
        pinned_file(pin, source=True)
    contract = _json(pinned_file(plan['runtime_contract'], source=True).read_bytes())
    inference._validate_contract(contract)
    inference.verify_source_compatibility(contract)
    from .quality_offline import run_stage
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = _inventory(plan)
    (output / 'requested-inventory.json').write_bytes(_encoded(dict(rows=rows)))
    child = output / 'capture'
    command = [sys.executable, '-B', '-m', 'bench.quality_gotrack_probe', 'run',
               '--plan', str(Path(plan_path).resolve()), '--plan-sha256', plan_sha256,
               '--output', str(child), '--cloud-execution']
    stage, failure = None, None
    try:
        remaining = LIMITS['max_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError('probe time budget exhausted in supervisor preflight')
        stage = run_stage('probe', command, output, timeout_s=remaining,
            storage_limit_bytes=LIMITS['model_cache_bytes'], storage_root=ROOT / '.cache/model-quality',
            output_limit_bytes=LIMITS['max_raw_capture_bytes'] + LIMITS['max_summary_bytes'])
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'[:1024]
    finally:
        stage_path = output / 'probe.receipt.json'
        if stage is None and stage_path.is_file() and stage_path.stat().st_size <= 256 * 1024:
            try:
                stage = _json(stage_path.read_bytes())
            except (ValueError, OSError):
                pass
        for index, row in enumerate(rows):
            path = child / f'row-{index:02d}.json'
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024:
                try:
                    saved = _json(path.read_bytes())
                    if all(saved.get(key) == row[key] for key in ('case_id', 'condition', 'capture_mode')):
                        rows[index] = saved
                except (ValueError, OSError):
                    pass
            if rows[index]['status'] == 'unrun':
                rows[index]['failure_reason'] = failure or 'child_terminated_before_trajectory'
                for iteration in rows[index]['iterations']:
                    iteration['reason'] = rows[index]['failure_reason']
        complete = False
        if stage and stage['status'] == 'complete':
            try:
                seal_path, receipt_path = child / 'completion.json', child / 'receipt.json'
                if any(path.is_symlink() or not path.is_file() for path in (seal_path, receipt_path)):
                    raise ValueError('missing or symlinked child completion/receipt')
                if seal_path.stat().st_size > 4096 or receipt_path.stat().st_size > LIMITS['max_summary_bytes']:
                    raise ValueError('child completion/receipt exceeds bounded size')
                seal = _json(seal_path.read_bytes())
                _keys(seal, ('receipt_sha256',), 'child completion')
                receipt_bytes = receipt_path.read_bytes()
                if seal['receipt_sha256'] != _sha(receipt_bytes):
                    raise ValueError('child completion does not authenticate receipt')
                receipt = _json(receipt_bytes)
                if receipt.get('schema') != 'gotrack_same_input_probe_receipt_v1' or receipt.get('plan_sha256') != plan_sha256 or receipt.get('complete') is not True:
                    raise ValueError('child receipt plan/schema/completion differs')
                actual_rows = receipt.get('rows')
                expected_rows = _inventory(plan)
                if not isinstance(actual_rows, list) or len(actual_rows) != 24:
                    raise ValueError('child receipt lacks requested trajectories')
                for actual, expected in zip(actual_rows, expected_rows):
                    if any(actual.get(key) != expected[key] for key in ('case_id', 'condition', 'capture_mode')) or actual.get('status') not in ('available', 'solver_unavailable') or actual.get('cleanup') != 'closed':
                        raise ValueError('child receipt row failed or differs from requested trajectory')
                    if [row.get('iteration') for row in actual.get('iterations', [])] != list(range(5)):
                        raise ValueError('child receipt does not account for all iterations')
                if len(receipt.get('parity', [])) != 18 or not all(row.get('observer_parity', row.get('cross_head_input_seed_equality')) is True for row in receipt['parity']):
                    raise ValueError('child receipt parity is missing or failed')
                rows = actual_rows
                complete = True
            except (ValueError, OSError, TypeError, KeyError) as exc:
                failure = f'child completion rejected: {type(exc).__name__}: {exc}'[:1024]
        value = dict(schema='gotrack_same_input_probe_supervision_v1', complete=complete,
                     failure=failure, stage=stage, rows=rows, billing_stopped=None)
        encoded = _encoded(value)
        if len(encoded) > LIMITS['max_summary_bytes']:
            raise ValueError('supervision inventory exceeds summary budget')
        with (output / 'supervision.json').open('xb') as stream:
            stream.write(encoded)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'supervise'):
        command = sub.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--plan-sha256', required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--cloud-execution', action='store_true')
    args = parser.parse_args(argv)
    runner = run if args.command == 'run' else supervise
    receipt = runner(args.plan, args.plan_sha256, args.output, cloud_execution=args.cloud_execution)
    print(json.dumps(dict(complete=receipt['complete'], output=str(args.output))))
    return 0 if receipt['complete'] else 1


if __name__ == '__main__':
    # Keep the capability's exact type stable when invoked with python -m.
    from .quality_gotrack_probe import main as imported_main
    raise SystemExit(imported_main())
