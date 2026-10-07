"""Capture and compare a bounded mug-only GoTrack PnP initialization diagnostic.

This is an exploratory paired ten-frame probe. It uses saved automatic mug
poses and the exact cached observed masks from the earlier diagnostic. It
captures the control branch's crop-space solver inputs before RANSAC, then runs
the paired local True/False solver option on the same frame and seed. It does
not read evaluation references or annotations, run chronology, or establish
independent accuracy.
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
from .quality_contract import Frame, validate
from .quality_gotrack import GoTrackRefiner, TexturedRenderer, load_network
from .quality_runner import read_rgb
from .vision import cv2


OBJECT = 'mug'
FRAME_IDS = [827, 853, 880, 906, 933, 959, 986, 1012, 1039, 1066]
PACKET_BUDGET_BYTES = 64 * 1024**2
PACKET_FIELDS = ('obj_points', 'target_points', 'crop_k', 'rvec', 'tvec')
SOURCE_FILES = (
    'bench/quality_pnp_mug_probe.py',
    'bench/quality_pnp_mug_replay.py',
    'bench/quality_pnp_guess_probe.py',
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
            raise FileNotFoundError(f'Missing fixed diagnostic source: {relative}')
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


class PacketCapturingRefiner(GoTrackRefiner):
    """Capture exact RANSAC arguments at the existing refiner method boundary."""

    def __init__(self, *args, packet_writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._packet_writer = packet_writer
        self._capture_context = None
        self._capture_iteration = 0

    def refine(self, frame, mask, seed):
        if self._packet_writer is not None:
            self._capture_iteration = 0
            self._capture_context = dict(
                frame_id=int(frame.frame_id),
                rgb_sha256=_array_sha256(frame.rgb),
                mask_sha256=_array_sha256(mask),
                seed_pose_sha256=_array_sha256(np.asarray(seed)),
            )
        try:
            return super().refine(frame, mask, seed)
        finally:
            self._capture_context = None

    def _solve_pnp_ransac(self, obj_points, target_points, crop_k, rvec, tvec):
        if self._packet_writer is not None and self._capture_context is not None:
            iteration = self._capture_iteration
            self._capture_iteration += 1
            copied = {name: np.array(value, copy=True, order='C') for name, value in zip(
                PACKET_FIELDS, (obj_points, target_points, crop_k, rvec, tvec))}
            self._packet_writer(
                context=dict(self._capture_context), iteration=iteration,
                rng_seed=(int(self._capture_context['frame_id']) * 17 + iteration) & 0x7fffffff,
                arrays=copied,
            )
        return super()._solve_pnp_ransac(obj_points, target_points, crop_k, rvec, tvec)


def _packet_writer(output_root, record):
    packet_dir = output_root / 'packets'
    packet_dir.mkdir(parents=True, exist_ok=True)

    def write(context, iteration, rng_seed, arrays):
        frame_id = context['frame_id']
        filename = f'frame-{frame_id:04d}-iteration-{iteration}.npz'
        final_path = packet_dir / filename
        if final_path.exists():
            raise FileExistsError(f'Refusing to replace immutable packet: {final_path}')
        point_indices = np.arange(len(arrays['obj_points']), dtype=np.int32)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            **arrays,
            point_indices=point_indices,
            frame_id=np.asarray(frame_id, dtype=np.int64),
            iteration=np.asarray(iteration, dtype=np.int32),
            rng_seed=np.asarray(rng_seed, dtype=np.int64),
        )
        payload = buffer.getvalue()
        new_total = int(record['captured_packet_bytes']) + len(payload)
        if new_total > PACKET_BUDGET_BYTES:
            raise ValueError(f'Immutable PnP packets exceed {PACKET_BUDGET_BYTES} byte budget')
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

        entry = dict(
            frame_id=frame_id, iteration=iteration, rng_seed=rng_seed,
            path=str(final_path.relative_to(output_root)),
            bytes=len(payload), sha256=_bytes_sha256(payload),
            arrays={name: _array_info(arrays[name]) for name in PACKET_FIELDS},
            point_indices=dict(count=len(point_indices), first=0,
                               last=int(point_indices[-1]) if len(point_indices) else None,
                               meaning='indices in the exact sampled RANSAC input order'),
            input_hashes={key: context[key] for key in
                          ('rgb_sha256', 'mask_sha256', 'seed_pose_sha256')},
            units=dict(object_points='GoTrack object/model coordinates in millimetres',
                       target_points='GoTrack crop pixels',
                       crop_k='crop-camera intrinsics in pixels',
                       rvec_tvec='RANSAC initial crop-camera pose; translation millimetres'),
        )
        record['packets'].append(entry)
        record['captured_packet_bytes'] = new_total
        save_result(output_root / 'capture.json', record)

    return write


def _save_neural_compatibility_record(output_root, record):
    """Write the established diagnostic branch schema beside packet metadata."""
    value = dict(
        schema_version=1,
        scope=('Paired mug-only five-iteration GoTrack diagnostic on ten selected frames; '
               'branch validations use the existing current-mask quality_contract.validate.'),
        object=OBJECT,
        status=record['status'],
        complete=bool(record.get('complete', False)),
        frames=record.get('frames', []),
        requested_frame_ids=FRAME_IDS,
        provenance=record.get('inference_provenance'),
        seed_result_sha256=record.get('seed_result_sha256'),
        probe_code_sha256=record.get('packet_writer_source_sha256'),
        fitting_patch_sha256=record.get('inference_provenance', {}).get(
            'adapter_sha256', {}).get('quality_gotrack'),
        settings=dict(unlit=True, disable_multisampling=True, iterations=5,
                      thresholds_changed=False, pnp_options=dict(control=True, no_guess=False)),
        selected_after_prior_benchmark=True,
        full_window_benchmark=False,
        sequential_recovery_test=False,
        reference_or_annotation_inputs=False,
        rendering_success_not_claimed=True,
        packet_manifest='capture.json',
        packet_count=len(record.get('packets', [])),
        packet_bytes=record.get('captured_packet_bytes', 0),
        packet_scope=record.get('packet_semantics'),
        replay_scope=record.get('replay_scope'),
        independent_accuracy_verified=False,
        overall_quality_gate_passed=False,
    )
    if 'failure' in record:
        value['failure'] = record['failure']
    save_result(output_root / 'neural.json', value)


def _validated_mask_path(mask_root, mask_path):
    relative = Path(mask_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe cached mask path: {mask_path}')
    mask_root = mask_root.resolve()
    result = (mask_root / relative).resolve()
    if not result.is_relative_to(mask_root):
        raise ValueError(f'Cached mask path escaped its root: {mask_path}')
    return result


def _branch_result(refiner, frame, mask, seed, torch, use_guess):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    candidate = refiner.refine(frame, mask, np.asarray(seed))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - begin) * 1000
    if candidate is None:
        valid, reason, stats = False, 'refinement_unavailable', None
    else:
        valid, reason, stats = validate(candidate, frame, mask)
    return dict(
        pnp_use_extrinsic_guess=bool(use_guess),
        current_image_validated=bool(valid),
        pose=None if candidate is None else candidate.pose.tolist(),
        reason=reason, validation=stats, elapsed_ms=elapsed_ms,
        timing_scope=('Diagnostic wall time. The control includes baseline packet '
                      'compression, hashing, fsync and progressive record writes; '
                      'paired latency comparison is invalid.'),
        paired_latency_comparable=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory for immutable packets and progressive diagnostic records')
    args = parser.parse_args(argv)
    output_root = _create_output_root(args.output_root)
    record = dict(
        schema_version=1, scope=__doc__, status='initializing', object=OBJECT,
        requested_frame_ids=FRAME_IDS, frames=[], packets=[], captured_packet_bytes=0,
        packet_budget_bytes=PACKET_BUDGET_BYTES,
        packet_semantics=('Exact arrays passed to solvePnPRansac at the baseline refiner hook; '
                          'these are the deterministic <=10,000 sampled crop-space solver inputs.'),
        replay_scope=('CPU replay reports crop-space residuals over every packet solver input. '
                      'It does not run quality_contract.validate on all retained/native-mask '
                      'correspondences and cannot independently verify correspondence identity.'),
        reference_or_annotation_inputs=False,
        settings=dict(
            paired_frames=FRAME_IDS, iterations=5, crop_size=[280, 280],
            render_unlit=True, disable_multisampling=True,
            automatic_seed_source='.cache/model-quality/results/mug/complete.json',
            mask_source='.cache/model-quality/results/mug/segmentation',
            branches=dict(control=True, no_guess=False),
            ransac=dict(iterations=3000, reprojection_error_crop_px=2.0,
                        confidence=0.999, flags='SOLVEPNP_ITERATIVE'),
            same_frame_iteration_rng_seed='(frame_id*17+iteration)&0x7fffffff',
        ),
        output_root=str(output_root), started_unix=time.time(),
    )
    save_result(output_root / 'capture.json', record)
    _save_neural_compatibility_record(output_root, record)
    renderer = None
    try:
        record['status'] = 'running'
        record['source_snapshot'] = _snapshot_sources(output_root)
        bundle = CACHE / 'inputs' / OBJECT
        manifest_path = bundle / 'input.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
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
            input_manifest_sha256=digest(manifest_path),
            input_source_hashes=manifest.get('source_hashes', {}),
            seed_pose_origin='preserved automatic complete run; no reference or evaluation pose',
            packet_writer_source_sha256=digest(ROOT / 'bench/quality_pnp_mug_probe.py'),
            replay_source_sha256=digest(ROOT / 'bench/quality_pnp_mug_replay.py'),
        )

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        _set_deterministic_runtime(torch)
        record['runtime'] = _runtime_info(torch)
        record['gotrack_checkpoint_sha256'] = digest(CACHE / 'checkpoints' / 'gotrack_checkpoint.pt')
        save_result(output_root / 'capture.json', record)
        _save_neural_compatibility_record(output_root, record)

        network = load_network('cuda')
        renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                                    unlit=True, disable_multisampling=True)
        for frame_id in FRAME_IDS:
            automatic = predicted[frame_id]
            seed = automatic.get('cameraFromObject')
            if seed is None:
                seed_array = None
                seed_hash = None
            else:
                seed_array = np.asarray(seed, dtype=np.float64)
                seed_hash = _array_sha256(seed_array)
            rgb = read_rgb(bundle, manifest, frame_id)
            mask_path = automatic.get('mask_path')
            if automatic.get('mask_state') != 'available' or not mask_path:
                raise ValueError(f'Actual automatic mask unavailable for frame {frame_id}')
            mask_file = _validated_mask_path(mask_root, mask_path)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != rgb.shape[:2] or not np.any(mask):
                raise ValueError(f'Invalid actual automatic mask for frame {frame_id}')
            frame = Frame(frame_id, rgb, np.asarray(manifest['intrinsics']))
            frame_result = dict(
                frame_id=frame_id,
                input_hashes=dict(rgb_sha256=_array_sha256(rgb),
                                  mask_sha256=_array_sha256(mask),
                                  mask_artifact_sha256=digest(mask_file),
                                  seed_pose_sha256=seed_hash),
                mask_artifact_path=str(mask_file.relative_to(mask_root.resolve())),
                branches={},
            )
            record['frames'].append(frame_result)
            save_result(output_root / 'capture.json', record)
            _save_neural_compatibility_record(output_root, record)

            if seed_array is None:
                frame_result['branches']['control'] = dict(
                    pnp_use_extrinsic_guess=True, current_image_validated=False,
                    pose=None, reason='automatic_seed_unavailable', validation=None,
                    timing_scope='No model execution: automatic seed unavailable.',
                    paired_latency_comparable=False)
                save_result(output_root / 'capture.json', record)
                _save_neural_compatibility_record(output_root, record)
                frame_result['branches']['no_guess'] = dict(
                    pnp_use_extrinsic_guess=False, current_image_validated=False,
                    pose=None, reason='automatic_seed_unavailable', validation=None,
                    timing_scope='No model execution: automatic seed unavailable.',
                    paired_latency_comparable=False)
                save_result(output_root / 'capture.json', record)
                _save_neural_compatibility_record(output_root, record)
                continue

            for branch_name, use_guess in (('control', True), ('no_guess', False)):
                # Each branch starts from the same automatic pose and deterministic
                # model RNG state. OpenCV RANSAC is reset inside each refiner iteration.
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                writer = _packet_writer(output_root, record) if branch_name == 'control' else None
                refiner = PacketCapturingRefiner(
                    network, renderer, manifest['object_id'], 'cuda',
                    pnp_use_extrinsic_guess=use_guess, packet_writer=writer,
                )
                frame_result['branches'][branch_name] = _branch_result(
                    refiner, frame, mask, seed_array, torch, use_guess)
                save_result(output_root / 'capture.json', record)
                _save_neural_compatibility_record(output_root, record)
                print(OBJECT, frame_id, branch_name,
                      frame_result['branches'][branch_name]['current_image_validated'], flush=True)

        if [item['frame_id'] for item in record['frames']] != FRAME_IDS:
            raise ValueError('Probe did not produce every requested mug frame in order')
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
