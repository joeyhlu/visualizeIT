"""Capture full-retained GoTrack control candidates for native validation replay.

This diagnostic runs the unchanged automatic mug control branch on the same ten
frames as the preliminary sampled-input PnP capture. Each attempted iteration
stores every retained crop-space correspondence, its exact deterministic sample
IDs, the crop/native transforms and intrinsics, the copied observed mask, and
the actual native candidate and validation result. It does not read evaluation
references or annotations and makes no independent-accuracy claim.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .quality_assets import CACHE, ROOT, digest, inference_provenance, save_result
from .quality_contract import Frame, PoseCandidate, validate
from .quality_gotrack import GoTrackRefiner, TexturedRenderer, load_network
from .quality_pnp_mug_replay import (
    FRAME_IDS, PACKET_BUDGET_BYTES as SAMPLED_PACKET_BUDGET_BYTES,
    _load_capture as _load_sampled_capture,
    _load_packet as _load_sampled_packet,
)
from .quality_runner import read_input, read_rgb
from .vision import cv2


OBJECT = 'mug'
ITERATIONS = 5
PACKET_BUDGET_BYTES = 64 * 1024**2
VALIDATION_STATS_ATOL = 1e-10
VALIDATION_STATS_RTOL = 1e-9
CAPTURE_ARRAYS = (
    'full_obj_points_mm', 'full_target_crop_px', 'full_weights', 'sample_ids',
    'sample_obj_points_mm', 'sample_target_crop_px', 'crop_k', 'crop_from_orig',
    'native_k', 'current_crop_pose_mm', 'initial_rvec', 'initial_tvec_mm',
)
PACKET_INPUT_FIELDS = (
    'full_obj_points_mm', 'full_target_crop_px', 'full_weights', 'sample_ids',
    'crop_k', 'crop_from_orig', 'native_k', 'current_crop_pose_mm',
    'initial_rvec', 'initial_tvec_mm',
)
PACKET_RESULT_FIELDS = (
    'ransac_inlier_ids', 'retained_inlier_ids', 'refined_rvec',
    'refined_tvec_mm', 'crop_camera_from_object_mm',
    'native_camera_from_object_m',
)
SOURCE_FILES = (
    'bench/quality_pnp_mug_native_probe.py',
    'bench/quality_pnp_mug_native_replay.py',
    'bench/quality_pnp_mug_probe.py',
    'bench/quality_pnp_mug_replay.py',
    'bench/quality_gotrack.py',
    'bench/quality_runner.py',
    'bench/quality_assets.py',
    'bench/quality_contract.py',
    'bench/quality_render_stability.py',
    'bench/quality_trace.py',
    'bench/vision.py',
)


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    return _bytes_sha256(array.tobytes(order='C'))


def _array_info(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=_array_sha256(array))


def _snapshot_sources(output_root):
    copied = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f'Missing fixed native diagnostic source: {relative}')
        target = output_root / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_sha = digest(source)
        copied[relative] = dict(path=str(target.relative_to(output_root)), sha256=source_sha)
        if digest(target) != source_sha:
            raise IOError(f'Source snapshot hash mismatch: {relative}')
        try:
            target.chmod(0o444)
        except OSError:
            pass
    return copied


def _create_output_root(path):
    output_root = Path(path).expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f'Output root already exists; preserving it: {output_root}')
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root.resolve()


def _runtime_info(torch):
    return dict(
        device='cuda', batch_size=1, python=sys.version,
        platform=platform.platform(), numpy=np.__version__, opencv=cv2.__version__,
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        math_threads={name: os.environ.get(name) for name in
                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        torch_cpu_threads=torch.get_num_threads(),
    )


def _set_deterministic_runtime(torch):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    cv2.setNumThreads(1)


def _validated_mask_path(mask_root, mask_path):
    relative = Path(mask_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe cached mask path: {mask_path}')
    mask_root = mask_root.resolve()
    result = (mask_root / relative).resolve()
    if not result.is_relative_to(mask_root):
        raise ValueError(f'Cached mask path escaped its root: {mask_path}')
    return result


def _load_sampled_boundary(manifest_path):
    path, capture, entries, packet_bytes = _load_sampled_capture(manifest_path)
    packets = {}
    for entry in entries:
        frame_id, iteration, rng_seed, arrays = _load_sampled_packet(path, entry)
        key = (frame_id, iteration)
        packets[key] = dict(rng_seed=rng_seed, arrays=arrays, entry=entry)
    expected = {(frame_id, iteration) for frame_id in FRAME_IDS for iteration in range(ITERATIONS)}
    if set(packets) != expected:
        raise ValueError('Preliminary sampled capture must contain every requested frame/iteration')
    return path, capture, packets, packet_bytes


def _verify_sampled_boundary(full_arrays, event, sampled, context):
    expected_arrays = sampled['arrays']
    mappings = (
        ('sample_obj_points_mm', 'obj_points'),
        ('sample_target_crop_px', 'target_points'),
        ('crop_k', 'crop_k'),
        ('initial_rvec', 'rvec'),
        ('initial_tvec_mm', 'tvec'),
    )
    for retained_name, sampled_name in mappings:
        if not np.array_equal(full_arrays[retained_name], expected_arrays[sampled_name]):
            raise ValueError(f'Sampled input mismatch at {retained_name}/{sampled_name}')
    ids = full_arrays['sample_ids']
    if not np.array_equal(full_arrays['full_obj_points_mm'][ids], expected_arrays['obj_points']):
        raise ValueError('full_obj_points_mm[sample_ids] differs from the preliminary packet')
    if not np.array_equal(full_arrays['full_target_crop_px'][ids], expected_arrays['target_points']):
        raise ValueError('full_target_crop_px[sample_ids] differs from the preliminary packet')
    if int(event['rng_seed']) != int(sampled['rng_seed']):
        raise ValueError('Frame/iteration RNG seed differs from the preliminary packet')
    prior_hashes = sampled['entry']['input_hashes']
    current_hashes = context['input_hashes']
    for name in ('rgb_sha256', 'mask_sha256', 'seed_pose_sha256'):
        if current_hashes.get(name) != prior_hashes.get(name):
            raise ValueError(f'Current {name} differs from the preliminary sampled capture')
    prior_frame_hashes = sampled['frame_input_hashes']
    if current_hashes.get('mask_artifact_sha256') != prior_frame_hashes.get('mask_artifact_sha256'):
        raise ValueError('Observed mask file hash differs from the preliminary sampled capture')


def _copy_observed_mask(output_root, record, frame_id, mask_path, mask):
    relative = Path('observed_masks') / f'frame-{frame_id:04d}{mask_path.suffix.lower() or ".png"}'
    final_path = output_root / relative
    mask_path = Path(mask_path)
    source_bytes = mask_path.stat().st_size
    source_sha = digest(mask_path)
    total = int(record['aggregate_capture_bytes']) + int(source_bytes)
    if total > PACKET_BUDGET_BYTES:
        record['packet_failures'].append(dict(
            frame_id=int(frame_id), iteration=None, reason='aggregate_64_mib_budget_exceeded_by_mask',
            attempted_bytes=int(source_bytes), aggregate_bytes=total,
        ))
        save_result(output_root / 'capture.json', record)
        raise ValueError(f'Observed masks and retained packets exceed {PACKET_BUDGET_BYTES} byte aggregate budget')
    if final_path.exists():
        raise FileExistsError(f'Refusing to replace copied observed mask: {final_path}')
    final_path.parent.mkdir(parents=True, exist_ok=True)
    copied_bytes = 0
    copied_hash = hashlib.sha256()
    try:
        with mask_path.open('rb') as source, final_path.open('xb') as destination:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                attempted_total = int(record['aggregate_capture_bytes']) + copied_bytes + len(chunk)
                if attempted_total > PACKET_BUDGET_BYTES:
                    record['packet_failures'].append(dict(
                        frame_id=int(frame_id), iteration=None,
                        reason='aggregate_64_mib_budget_exceeded_by_mask',
                        attempted_bytes=copied_bytes + len(chunk), aggregate_bytes=attempted_total,
                    ))
                    save_result(output_root / 'capture.json', record)
                    raise ValueError(
                        f'Observed masks and retained packets exceed {PACKET_BUDGET_BYTES} byte aggregate budget')
                destination.write(chunk)
                copied_hash.update(chunk)
                copied_bytes += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        actual_bytes = final_path.stat().st_size
        actual_sha = digest(final_path)
        if actual_bytes != source_bytes or copied_bytes != source_bytes or actual_sha != source_sha or copied_hash.hexdigest() != source_sha:
            raise IOError(f'Copied observed mask size/hash changed for frame {frame_id}')
        copied = cv2.imread(str(final_path), cv2.IMREAD_GRAYSCALE)
        if copied is None or not np.array_equal(copied, mask):
            raise IOError(f'Copied observed mask does not decode identically for frame {frame_id}')
    except Exception:
        try:
            final_path.unlink()
        except FileNotFoundError:
            pass
        raise
    mask_bytes = int(actual_bytes)
    try:
        final_path.chmod(0o444)
    except OSError:
        pass
    info = dict(
        path=str(relative.as_posix()), bytes=int(mask_bytes), file_sha256=digest(final_path),
        array=dict(shape=list(mask.shape), dtype=str(mask.dtype), sha256=_array_sha256(mask)),
    )
    record['observed_mask_bytes'] = int(record['observed_mask_bytes']) + int(mask_bytes)
    record['aggregate_capture_bytes'] = total
    return info


class NativePacketCapture:
    """Join the adapter's isolated retained/result events into immutable packets."""

    def __init__(self, output_root, record, sampled_packets):
        self.output_root = output_root
        self.record = record
        self.sampled_packets = sampled_packets
        self.context = None
        self.pending = {}

    def bind_frame(self, context):
        if self.pending:
            raise RuntimeError('Previous frame has an unfinished native capture event')
        self.context = context

    def __call__(self, packet):
        event = packet['event']
        frame_id = int(event['frame_id'])
        iteration = int(event['iteration'])
        key = (frame_id, iteration)
        if self.context is None or self.context['frame_id'] != frame_id:
            raise ValueError('Native capture callback has no matching current frame')
        if event['stage'] == 'retained':
            if key in self.pending:
                raise ValueError(f'Duplicate retained-set callback for {key}')
            if list(self.context['frame'].rgb.shape[:2]) != list(event['native_shape']):
                raise ValueError(f'Native frame shape changed during capture for frame {frame_id}')
            arrays = packet['arrays']
            if tuple(arrays) != CAPTURE_ARRAYS:
                if set(arrays) != set(CAPTURE_ARRAYS):
                    raise ValueError(f'Unexpected retained callback arrays: {sorted(arrays)}')
            sampled = self.sampled_packets.get(key)
            if sampled is None:
                raise ValueError(f'No matching preliminary packet for frame/iteration {key}')
            _verify_sampled_boundary(arrays, event, sampled, self.context)
            ids = arrays['sample_ids']
            n = len(arrays['full_obj_points_mm'])
            if (arrays['full_obj_points_mm'].shape != (n, 3) or
                    arrays['full_target_crop_px'].shape != (n, 2) or
                    arrays['full_weights'].shape != (n,) or
                    ids.ndim != 1 or len(ids) != min(n, 10000) or
                    not np.array_equal(ids, np.linspace(0, n-1, min(n, 10000), dtype=int))):
                raise ValueError(f'Invalid full-retained arrays or deterministic sample IDs for {key}')
            self.pending[key] = dict(event=event, arrays=arrays, context=self.context,
                                     sampled_entry=sampled['entry'])
            return
        if event['stage'] != 'result':
            raise ValueError(f'Unknown GoTrack diagnostic callback stage: {event["stage"]}')
        if key not in self.pending:
            raise ValueError(f'Result callback preceded retained packet for {key}')
        pending = self.pending.pop(key)
        self._write_packet(key, pending, event, packet['arrays'])

    def _write_packet(self, key, pending, event, result_arrays):
        frame_id, iteration = key
        frame_context = pending['context']
        retained = pending['arrays']
        arrays = dict(retained)
        arrays.update(result_arrays)
        frame_id_arr = np.asarray(frame_id, dtype=np.int64)
        iteration_arr = np.asarray(iteration, dtype=np.int32)
        rng_seed_arr = np.asarray(int(event['rng_seed']), dtype=np.int64)

        expected_candidate = dict(
            fit_state=event['fit_state'], solver_success=bool(event['solver_success']),
            rng_seed=int(event['rng_seed']),
            retained_count=len(retained['full_obj_points_mm']),
            sampled_count=len(retained['sample_ids']),
            ransac_inlier_count=int(event.get('ransac_inlier_count', 0)),
            ransac_inlier_ids=(result_arrays.get('ransac_inlier_ids', np.empty(0, dtype=np.int32)).astype(int).tolist()),
            retained_inlier_ids=(result_arrays.get('retained_inlier_ids', np.empty(0, dtype=np.int32)).astype(int).tolist()),
            validation_state='not_run', validation_reason=f'candidate_unavailable_{event["fit_state"]}',
            validation_stats=None, pose_m=None, candidate_arrays={},
        )
        if event['fit_state'] == 'refined':
            candidate_arrays = dict(
                pose_m=result_arrays['native_camera_from_object_m'],
                points_object_m=result_arrays['candidate_points_object_m'],
                pixels_native=result_arrays['candidate_pixels_native'],
                weights=result_arrays['candidate_weights'],
            )
            candidate = PoseCandidate(
                candidate_arrays['pose_m'], candidate_arrays['points_object_m'],
                candidate_arrays['pixels_native'], candidate_arrays['weights'],
            )
            valid, reason, stats = validate(candidate, frame_context['frame'], frame_context['mask'])
            expected_candidate.update(
                validation_state='accepted' if valid else 'rejected',
                validation_reason=reason,
                validation_stats=stats,
                pose_m=candidate.pose.tolist(),
                candidate_arrays={name: _array_info(value) for name, value in candidate_arrays.items()},
            )

        # Persist only the full retained inputs and compact solver outputs.
        # Sampled arrays and native candidate correspondences are hashed in the
        # manifest and rederived during replay; duplicating them here can exceed
        # the frozen aggregate packet budget.
        arrays = {name: retained[name] for name in PACKET_INPUT_FIELDS}
        arrays.update({name: result_arrays[name] for name in PACKET_RESULT_FIELDS
                       if name in result_arrays})
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, **arrays, frame_id=frame_id_arr, iteration=iteration_arr,
            rng_seed=rng_seed_arr,
        )
        payload = buffer.getvalue()
        total = int(self.record['aggregate_capture_bytes']) + len(payload)
        if total > PACKET_BUDGET_BYTES:
            failure = dict(
                frame_id=frame_id, iteration=iteration,
                reason='aggregate_64_mib_budget_exceeded_by_full_packet',
                attempted_bytes=len(payload), aggregate_bytes=total,
                retained_count=len(retained['full_obj_points_mm']),
            )
            self.record['packet_failures'].append(failure)
            save_result(self.output_root / 'capture.json', self.record)
            raise ValueError(f'Full retained packets exceed {PACKET_BUDGET_BYTES} byte aggregate budget')

        packet_dir = self.output_root / 'packets'
        packet_dir.mkdir(parents=True, exist_ok=True)
        relative = Path('packets') / f'frame-{frame_id:04d}-iteration-{iteration}.npz'
        final_path = self.output_root / relative
        if final_path.exists():
            raise FileExistsError(f'Refusing to replace immutable native packet: {final_path}')
        temporary = final_path.with_suffix(final_path.suffix + '.tmp')
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final_path)
        try:
            final_path.chmod(0o444)
        except OSError:
            pass

        packet_entry = dict(
            frame_id=frame_id, iteration=iteration, rng_seed=int(event['rng_seed']),
            path=relative.as_posix(), bytes=len(payload), sha256=_bytes_sha256(payload),
            arrays={name: _array_info(value) for name, value in arrays.items()},
            retained_count=len(retained['full_obj_points_mm']),
            sampled_count=len(retained['sample_ids']),
            sample_boundary=dict(
                path=pending['sampled_entry']['path'],
                packet_sha256=pending['sampled_entry']['sha256'],
                exact_array_match=True,
                arrays={
                    'obj_points': _array_info(retained['sample_obj_points_mm']),
                    'target_points': _array_info(retained['sample_target_crop_px']),
                    'crop_k': _array_info(retained['crop_k']),
                    'rvec': _array_info(retained['initial_rvec']),
                    'tvec': _array_info(retained['initial_tvec_mm']),
                },
            ),
            input_hashes=frame_context['input_hashes'],
            observed_mask=frame_context['observed_mask'],
            native_shape=list(frame_context['frame'].rgb.shape[:2]),
            transform_direction='crop_from_orig maps native camera coordinates to crop camera coordinates',
            units=dict(
                full_obj_points_mm='GoTrack object/model coordinates in millimetres',
                full_target_crop_px='crop image pixel coordinates', full_weights='GoTrack confidence weights',
                crop_k='crop-camera intrinsics in pixels', crop_from_orig='native camera to crop camera C',
                native_k='native-camera intrinsics in pixels', current_crop_pose_mm='crop camera from object, mm',
                initial_rvec='initial crop-camera rotation vector, radians',
                initial_tvec_mm='initial crop-camera translation, millimetres',
                candidate_pose_m='native camera from object, metres',
            ),
            fit_state=event['fit_state'], expected_candidate=expected_candidate,
        )
        self.record['packets'].append(packet_entry)
        self.record['captured_packet_bytes'] = int(self.record['captured_packet_bytes']) + len(payload)
        self.record['aggregate_capture_bytes'] = total
        save_result(self.output_root / 'capture.json', self.record)

    def flush_pending(self, reason):
        for frame_id, iteration in sorted(self.pending):
            pending = self.pending[(frame_id, iteration)]
            self.record['packet_failures'].append(dict(
                frame_id=frame_id, iteration=iteration, reason=reason,
                retained_count=len(pending['arrays']['full_obj_points_mm']),
            ))
        self.pending.clear()

    def account_iterations(self, frame_result, no_seed=False, failure_reason=None):
        by_iteration = {
            int(entry['iteration']): entry for entry in self.record['packets']
            if int(entry['frame_id']) == int(frame_result['frame_id'])
        }
        failed_attempts = [i for i, entry in by_iteration.items() if entry['fit_state'] != 'refined']
        stopped_at = min(failed_attempts) if failed_attempts else None
        accounting = []
        for iteration in range(ITERATIONS):
            entry = by_iteration.get(iteration)
            if entry is not None:
                accounting.append(dict(
                    iteration=iteration, attempted=True, fit_state=entry['fit_state'],
                    packet_path=entry['path'], reason=None if entry['fit_state'] == 'refined'
                    else entry['expected_candidate']['validation_reason'],
                ))
            else:
                if no_seed:
                    reason = 'automatic_seed_unavailable'
                elif failure_reason:
                    reason = failure_reason
                elif stopped_at is not None and iteration > stopped_at:
                    reason = f'refiner_stopped_after_iteration_{stopped_at}'
                else:
                    reason = 'iteration_not_recorded_in_incomplete_capture'
                accounting.append(dict(iteration=iteration, attempted=False, fit_state='not_attempted', reason=reason))
        frame_result['iteration_accounting'] = accounting


def _save_neural_compatibility_record(output_root, record):
    value = dict(
        schema_version=1,
        scope=('Single-control mug capture for independent native replay; this record contains no paired '
               'candidate branch and is not an E2 paired-comparison result.'),
        object=OBJECT,
        status=record['status'],
        complete=bool(record.get('complete', False)),
        frames=record.get('frames', []),
        requested_frame_ids=FRAME_IDS,
        provenance=record.get('inference_provenance'),
        seed_result_sha256=record.get('seed_result_sha256'),
        probe_code_sha256=record.get('native_probe_source_sha256'),
        fitting_patch_sha256=record.get('inference_provenance', {}).get(
            'adapter_sha256', {}).get('quality_gotrack'),
        settings=dict(unlit=True, disable_multisampling=True, iterations=ITERATIONS,
                      thresholds_changed=False, control_only=True,
                      pnp_use_extrinsic_guess=True, paired_branch_comparison=False),
        selected_after_prior_benchmark=True,
        full_window_benchmark=False,
        sequential_recovery_test=False,
        reference_or_annotation_inputs=False,
        rendering_success_not_claimed=True,
        packet_manifest='capture.json',
        packet_count=len(record.get('packets', [])),
        packet_bytes=record.get('captured_packet_bytes', 0),
        packet_budget_bytes=record.get('packet_budget_bytes'),
        packet_scope=record.get('packet_semantics'),
        replay_scope=record.get('replay_scope'),
        independent_accuracy_verified=False,
        overall_quality_gate_passed=False,
    )
    if 'failure' in record:
        value['failure'] = record['failure']
    save_result(output_root / 'neural.json', value)


def _branch_result(refiner, frame, mask, seed, torch):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    candidate = refiner.refine(frame, mask, np.asarray(seed))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - begin) * 1000
    if candidate is None:
        valid, reason, stats = False, 'refinement_unavailable', None
    else:
        valid, reason, stats = validate(candidate, frame, mask)
    return candidate, dict(
        pnp_use_extrinsic_guess=True,
        current_image_validated=bool(valid),
        pose=None if candidate is None else candidate.pose.tolist(),
        reason=reason, validation=stats, elapsed_ms=elapsed_ms,
        timing_scope='Diagnostic capture timing includes immutable full-retained packet work; not a latency comparison.',
        paired_latency_comparable=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory for immutable full-retained packets and progressive records')
    parser.add_argument('--sampled-capture-manifest', type=Path,
                        default=CACHE / 'diagnostics/mug-local-pnp-v1/capture.json',
                        help='Preserved preliminary sampled-input capture used as an exact solver-input boundary')
    args = parser.parse_args(argv)
    output_root = _create_output_root(args.output_root)
    record = dict(
        schema_version=1,
        scope=__doc__, status='initializing', object=OBJECT,
        requested_frame_ids=FRAME_IDS, requested_iterations=ITERATIONS,
        frames=[], packets=[], packet_failures=[], captured_packet_bytes=0,
        observed_mask_bytes=0, aggregate_capture_bytes=0,
        packet_budget_bytes=PACKET_BUDGET_BYTES,
        packet_semantics=('Every retained correspondence before deterministic subsampling, plus exact sample IDs, '
                          'crop/native geometry and the captured control candidate for each attempted iteration.'),
        replay_scope=('CPU replay re-runs unchanged RANSAC+LM on exact sampled IDs, reconstructs native candidates '
                      'from all retained points, and calls unchanged quality_contract.validate against copied masks.'),
        validation_stats_tolerance=dict(atol=VALIDATION_STATS_ATOL, rtol=VALIDATION_STATS_RTOL,
                                        frozen_before_capture=True),
        reference_or_annotation_inputs=False,
        output_root=str(output_root), started_unix=time.time(),
    )
    save_result(output_root / 'capture.json', record)
    _save_neural_compatibility_record(output_root, record)
    renderer = None
    capturer = None
    active_frame_result = None
    try:
        record['status'] = 'running'
        record['source_snapshot'] = _snapshot_sources(output_root)
        sample_path, sample_capture, sampled_packets, sample_bytes = _load_sampled_boundary(
            args.sampled_capture_manifest)
        sampled_frame_hashes = {
            int(frame['frame_id']): frame['input_hashes']
            for frame in sample_capture.get('frames', [])
        }
        for key, sampled in sampled_packets.items():
            sampled['frame_input_hashes'] = sampled_frame_hashes.get(key[0], {})
        if set(sampled_frame_hashes) != set(FRAME_IDS):
            raise ValueError('Preliminary sampled capture does not have frame provenance for all ten inputs')
        record['sampled_boundary'] = dict(
            manifest_path=str(sample_path), manifest_sha256=digest(sample_path),
            capture_source_snapshot=sample_capture.get('source_snapshot'),
            packet_count=len(sampled_packets), packet_bytes=sample_bytes,
            source_schema=sample_capture.get('packet_semantics'),
        )

        bundle = CACHE / 'inputs' / OBJECT
        manifest = read_input(bundle)
        if manifest.get('object') != OBJECT:
            raise ValueError('Mug input bundle required')
        if not set(FRAME_IDS).issubset(set(manifest.get('frame_ids', []))):
            raise ValueError('Input bundle is missing one or more selected mug frames')
        seed_path = CACHE / 'results' / OBJECT / 'complete.json'
        seed_result = json.loads(seed_path.read_text(encoding='utf-8'))
        if (seed_result.get('object') != OBJECT or seed_result.get('mode') != 'complete' or
                not seed_result.get('complete') or not seed_result.get('automatic') or
                seed_result.get('diagnostic_control')):
            raise ValueError('Expected the preserved complete automatic mug seed result')
        predicted = {int(frame['frameId']): frame for frame in seed_result['frames']}
        missing = [frame_id for frame_id in FRAME_IDS if frame_id not in predicted]
        if missing:
            raise ValueError(f'Automatic mug seed result is missing selected frames: {missing}')
        mask_root = CACHE / 'results' / OBJECT / 'segmentation'
        record.update(
            seed_result_sha256=digest(seed_path),
            seed_result_provenance=seed_result.get('provenance'),
            seed_tracking_settings=seed_result.get('tracking_settings'),
            inference_provenance=inference_provenance(bundle),
            input_manifest_sha256=digest(bundle / 'input.json'),
            input_source_hashes=manifest.get('source_hashes', {}),
            input_intrinsics=np.asarray(manifest['intrinsics'], dtype=np.float64).tolist(),
            seed_pose_origin='preserved automatic complete run; no reference or evaluation pose',
            native_probe_source_sha256=digest(ROOT / 'bench/quality_pnp_mug_native_probe.py'),
            native_replay_source_sha256=digest(ROOT / 'bench/quality_pnp_mug_native_replay.py'),
        )

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        _set_deterministic_runtime(torch)
        record['runtime'] = _runtime_info(torch)
        record['gotrack_checkpoint_sha256'] = digest(CACHE / 'checkpoints' / 'gotrack_checkpoint.pt')
        capturer = NativePacketCapture(output_root, record, sampled_packets)
        save_result(output_root / 'capture.json', record)
        _save_neural_compatibility_record(output_root, record)

        network = load_network('cuda')
        renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                                    unlit=True, disable_multisampling=True)
        for frame_id in FRAME_IDS:
            automatic = predicted[frame_id]
            seed = automatic.get('cameraFromObject')
            seed_array = None if seed is None else np.asarray(seed, dtype=np.float64)
            seed_hash = None if seed_array is None else _array_sha256(seed_array)
            rgb = read_rgb(bundle, manifest, frame_id)
            mask_path = automatic.get('mask_path')
            if automatic.get('mask_state') != 'available' or not mask_path:
                raise ValueError(f'Actual automatic mask unavailable for frame {frame_id}')
            source_mask_path = _validated_mask_path(mask_root, mask_path)
            mask = cv2.imread(str(source_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                raise ValueError(f'Invalid actual automatic mask for frame {frame_id}')
            mask_info = _copy_observed_mask(output_root, record, frame_id, source_mask_path, mask)
            frame = Frame(frame_id, rgb, np.asarray(manifest['intrinsics']))
            frame_result = dict(
                frame_id=frame_id,
                input_hashes=dict(rgb_sha256=_array_sha256(rgb),
                                  mask_sha256=_array_sha256(mask),
                                  mask_artifact_sha256=digest(source_mask_path),
                                  seed_pose_sha256=seed_hash),
                mask_artifact_path=str(source_mask_path.relative_to(mask_root.resolve())),
                observed_mask=mask_info,
                branches={},
            )
            record['frames'].append(frame_result)
            active_frame_result = frame_result
            save_result(output_root / 'capture.json', record)
            _save_neural_compatibility_record(output_root, record)

            if seed_array is None:
                frame_result['branches']['control'] = dict(
                    pnp_use_extrinsic_guess=True, current_image_validated=False,
                    pose=None, reason='automatic_seed_unavailable', validation=None,
                    timing_scope='No model execution: automatic seed unavailable.',
                    paired_latency_comparable=False)
                capturer.account_iterations(frame_result, no_seed=True)
                save_result(output_root / 'capture.json', record)
                _save_neural_compatibility_record(output_root, record)
                continue

            capturer.bind_frame(dict(
                frame_id=frame_id, frame=frame, mask=mask,
                input_hashes=frame_result['input_hashes'], observed_mask=mask_info,
            ))
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            refiner = GoTrackRefiner(
                network, renderer, manifest['object_id'], 'cuda',
                pnp_use_extrinsic_guess=True,
                diagnostic_capture_callback=capturer,
            )
            try:
                candidate, branch = _branch_result(refiner, frame, mask, seed_array, torch)
            except BaseException:
                capturer.flush_pending('refiner_raised_before_iteration_result')
                capturer.account_iterations(frame_result, failure_reason='refiner_raised_before_iteration_result')
                raise
            frame_result['branches']['control'] = branch
            capturer.account_iterations(frame_result)
            save_result(output_root / 'capture.json', record)
            _save_neural_compatibility_record(output_root, record)
            print(OBJECT, frame_id, branch['current_image_validated'],
                  len([p for p in record['packets'] if p['frame_id'] == frame_id]), flush=True)
            active_frame_result = None

        if [item['frame_id'] for item in record['frames']] != FRAME_IDS:
            raise ValueError('Native probe did not account for all ten selected mug frames')
        all_iteration_rows = [
            row for frame_result in record['frames']
            for row in frame_result.get('iteration_accounting', [])
        ]
        if len(all_iteration_rows) != len(FRAME_IDS) * ITERATIONS:
            raise ValueError('Native probe is missing explicit iteration accounting')
        record['status'] = 'complete'
        record['complete'] = True
        record['completed_unix'] = time.time()
        record['independent_accuracy_verified'] = False
        record['full_window_benchmark'] = False
        record['sequential_recovery_test'] = False
        record['overall_quality_gate_passed'] = False
        save_result(output_root / 'capture.json', record)
        _save_neural_compatibility_record(output_root, record)
    except BaseException as error:
        if capturer is not None and capturer.pending:
            capturer.flush_pending('capture_failed_before_iteration_result')
        if active_frame_result is not None:
            capturer.account_iterations(
                active_frame_result, failure_reason='capture_failed_before_iteration_result')
        record['status'] = 'failed'
        record['complete'] = False
        record['failure'] = dict(type=type(error).__name__, message=str(error))
        record['failed_unix'] = time.time()
        try:
            save_result(output_root / 'capture.json', record)
            _save_neural_compatibility_record(output_root, record)
        except Exception:
            pass
        raise
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == '__main__':
    main()
