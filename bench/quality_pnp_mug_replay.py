"""Replay captured mug PnP solver inputs on CPU with both guess options.

The report computes crop-camera reprojection residuals over every point in each
captured RANSAC input packet. These are the deterministic <=10,000 sampled
solver inputs. This replay does not have the refiner's full retained/native-mask
correspondence set, run quality_contract.validate, load a model, read references
or annotations, or establish correspondence identity or independent accuracy.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .quality_assets import ROOT, digest, save_result
from .vision import cv2


FRAME_IDS = [827, 853, 880, 906, 933, 959, 986, 1012, 1039, 1066]
PACKET_BUDGET_BYTES = 64 * 1024**2
SOLVER_FIELDS = ('obj_points', 'target_points', 'crop_k', 'rvec', 'tvec')
SOURCE_FILES = (
    'bench/quality_pnp_mug_replay.py',
    'bench/quality_gotrack.py',
    'bench/quality_assets.py',
    'bench/vision.py',
)


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    return _bytes_sha256(array.tobytes(order='C'))


def _snapshot_sources(output_root):
    copied = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f'Missing fixed replay source: {relative}')
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


def replay_solver_inputs(obj_points, target_points, crop_k, rvec, tvec,
                         rng_seed, use_extrinsic_guess, cv2_module=None):
    """Run the frozen RANSAC+LM calls, returning pose and full packet residuals."""
    cv = cv2 if cv2_module is None else cv2_module
    if type(use_extrinsic_guess) is not bool:
        raise TypeError('use_extrinsic_guess must be a bool')
    obj = np.array(obj_points, copy=True, order='C')
    target = np.array(target_points, copy=True, order='C')
    camera = np.array(crop_k, copy=True, order='C')
    initial_rvec = np.array(rvec, copy=True, order='C')
    initial_tvec = np.array(tvec, copy=True, order='C')
    captured_initial_rvec = initial_rvec.copy()
    captured_initial_tvec = initial_tvec.copy()
    n = len(obj)
    if (obj.shape != (n, 3) or target.shape != (n, 2) or n < 6 or
            camera.shape != (3, 3) or initial_rvec.size != 3 or initial_tvec.size != 3 or
            not all(np.isfinite(array).all() for array in
                    (obj, target, camera, initial_rvec, initial_tvec)) or
            camera[0, 0] <= 0 or camera[1, 1] <= 0):
        raise ValueError('Invalid finite crop-space RANSAC packet arrays')

    # The option and solver kwargs match GoTrackRefiner._solve_pnp_ransac.
    # Reset immediately before the call for each branch so both options see the
    # same OpenCV RNG state and byte-identical copied inputs.
    cv.setRNGSeed(int(rng_seed) & 0x7fffffff)
    try:
        success, solved_rvec, solved_tvec, inliers = cv.solvePnPRansac(
            obj, target, camera, None,
            rvec=initial_rvec, tvec=initial_tvec,
            useExtrinsicGuess=use_extrinsic_guess,
            iterationsCount=3000, reprojectionError=2., confidence=.999,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
    except cv.error as error:
        return dict(
            use_extrinsic_guess=use_extrinsic_guess,
            solver_success=False, fit_state='ransac_exception',
            exception=f'{type(error).__name__}: {error}',
            input_point_count=n, ransac_inlier_count=0,
            ransac_inlier_ids=[], crop_camera_from_object_mm=None,
            full_solver_input_residual_px=None,
            full_solver_input_projected_inlier_count_at_2px=None,
        )

    success = bool(success)
    inlier_ids = [] if inliers is None else np.asarray(inliers, dtype=int).reshape(-1)
    result = dict(
        use_extrinsic_guess=use_extrinsic_guess,
        solver_success=success, input_point_count=int(n),
        ransac_inlier_count=int(len(inlier_ids)),
        ransac_inlier_ids=[int(value) for value in inlier_ids],
        initial_rvec=captured_initial_rvec.reshape(3).tolist(),
        initial_tvec_mm=captured_initial_tvec.reshape(3).tolist(),
        crop_camera_from_object_mm=None,
        full_solver_input_residual_px=None,
        full_solver_input_projected_inlier_count_at_2px=None,
    )
    if not success or inliers is None or len(inlier_ids) < 6:
        result['fit_state'] = 'ransac_unavailable'
        return result

    chosen = inlier_ids.copy()
    try:
        solved_rvec, solved_tvec = cv.solvePnPRefineLM(
            obj[chosen].copy(), target[chosen].copy(), camera.copy(), None,
            np.array(solved_rvec, copy=True), np.array(solved_tvec, copy=True),
        )
    except cv.error as error:
        result['fit_state'] = 'lm_exception'
        result['exception'] = f'{type(error).__name__}: {error}'
        return result

    solved_rvec = np.asarray(solved_rvec, dtype=np.float64).reshape(3, 1)
    solved_tvec = np.asarray(solved_tvec, dtype=np.float64).reshape(3, 1)
    rotation, _ = cv.Rodrigues(solved_rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = solved_tvec[:, 0]
    camera_points = obj @ rotation.T + solved_tvec[:, 0]
    projected, _ = cv.projectPoints(obj, solved_rvec, solved_tvec, camera, None)
    projected = np.asarray(projected).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    finite = np.isfinite(errors) & np.isfinite(camera_points).all(axis=1)
    positive = camera_points[:, 2] > .01
    full_inliers = finite & positive & (errors <= 2.)
    if not finite.all() or not np.isfinite(pose).all():
        result['fit_state'] = 'nonfinite_refined_pose_or_residual'
        return result

    def summary(values):
        values = np.asarray(values, dtype=np.float64)
        if not values.size:
            return dict(count=0, mean=None, rmse=None, median=None, p95=None,
                        p99=None, maximum=None)
        return dict(
            count=int(values.size), mean=float(np.mean(values)),
            rmse=float(np.sqrt(np.mean(values**2))),
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            maximum=float(np.max(values)),
        )

    result.update(
        fit_state='refined',
        refined_rvec=solved_rvec[:, 0].tolist(),
        refined_tvec_mm=solved_tvec[:, 0].tolist(),
        crop_camera_from_object_mm=pose.tolist(),
        full_solver_input_residual_px=summary(errors),
        ransac_inlier_residual_px=summary(errors[chosen]),
        full_solver_input_projected_inlier_count_at_2px=int(full_inliers.sum()),
        full_solver_input_projected_inlier_ids_at_2px=[int(i) for i in np.flatnonzero(full_inliers)],
        residual_scope=('post-LM crop-pixel residuals over all points stored in this exact '
                        'sampled RANSAC input packet; not the refiner full-retained/native set'),
        pose_scope='crop-camera from object with translation in millimetres; no native pose transform was captured',
    )
    return result


def _safe_packet_path(manifest_path, relative):
    relative_path = Path(relative)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        raise ValueError(f'Unsafe packet path in capture manifest: {relative}')
    root = manifest_path.parent.resolve()
    packet_path = (root / relative_path).resolve()
    if not packet_path.is_relative_to(root):
        raise ValueError(f'Packet path escaped capture root: {relative}')
    return packet_path


def _load_packet(manifest_path, entry):
    packet_path = _safe_packet_path(manifest_path, entry['path'])
    if not packet_path.is_file():
        raise FileNotFoundError(f'Missing captured packet: {entry["path"]}')
    if packet_path.stat().st_size != int(entry['bytes']) or digest(packet_path) != entry['sha256']:
        raise ValueError(f'Captured packet size/hash mismatch: {entry["path"]}')
    with np.load(packet_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in SOLVER_FIELDS}
        point_indices = np.array(archive['point_indices'], copy=True)
        frame_id = int(archive['frame_id'])
        iteration = int(archive['iteration'])
        rng_seed = int(archive['rng_seed'])
    if frame_id != int(entry['frame_id']) or iteration != int(entry['iteration']) or rng_seed != int(entry['rng_seed']):
        raise ValueError(f'Packet metadata mismatch: {entry["path"]}')
    if (point_indices.ndim != 1 or not np.array_equal(point_indices,
            np.arange(len(arrays['obj_points']), dtype=np.int32))):
        raise ValueError(f'Packet solver input point order mismatch: {entry["path"]}')
    for name, array in arrays.items():
        if _array_sha256(array) != entry['arrays'][name]['sha256']:
            raise ValueError(f'Packet array hash mismatch for {name}: {entry["path"]}')
        if list(array.shape) != entry['arrays'][name]['shape'] or str(array.dtype) != entry['arrays'][name]['dtype']:
            raise ValueError(f'Packet array shape/type mismatch for {name}: {entry["path"]}')
    return frame_id, iteration, rng_seed, arrays


def _runtime_info():
    return dict(device='cpu', python=sys.version, platform=platform.platform(),
                numpy=np.__version__, opencv=cv2.__version__, torch_loaded=False,
                ransac_rng='reset to captured frame/iteration seed before each option')


def _load_capture(capture_manifest_path):
    path = Path(capture_manifest_path).expanduser().resolve()
    capture = json.loads(path.read_text(encoding='utf-8'))
    if capture.get('object') != 'mug' or capture.get('requested_frame_ids') != FRAME_IDS:
        raise ValueError('Expected the fixed ten-frame mug capture manifest')
    if capture.get('status') != 'complete' or not capture.get('complete'):
        raise ValueError('Capture manifest must be complete before replay')
    frame_ids = [frame['frame_id'] for frame in capture.get('frames', [])]
    if frame_ids != FRAME_IDS:
        raise ValueError('Capture must contain all ten selected frames in requested order')
    packet_entries = capture.get('packets', [])
    if not packet_entries:
        raise ValueError('Capture contains no immutable RANSAC packets')
    packet_bytes = sum(int(entry['bytes']) for entry in packet_entries)
    if packet_bytes != int(capture.get('captured_packet_bytes', -1)) or packet_bytes > PACKET_BUDGET_BYTES:
        raise ValueError('Capture packet budget accounting is invalid')
    packet_keys = [(int(entry['frame_id']), int(entry['iteration'])) for entry in packet_entries]
    if len(packet_keys) != len(set(packet_keys)):
        raise ValueError('Capture contains duplicate frame/iteration packets')
    return path, capture, packet_entries, packet_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-manifest', type=Path, required=True,
                        help='Complete capture.json from quality_pnp_mug_probe')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory for the CPU replay report')
    args = parser.parse_args(argv)
    output_root = _create_output_root(args.output_root)
    report = dict(
        schema_version=1, scope=__doc__, status='initializing', object='mug',
        capture_manifest=str(args.capture_manifest.expanduser().resolve()),
        branches=dict(control_use_extrinsic_guess=True, candidate_use_extrinsic_guess=False),
        solver=dict(iterations=3000, reprojection_error_crop_px=2.0,
                    confidence=0.999, flags='SOLVEPNP_ITERATIVE', lm='solvePnPRefineLM'),
        residual_scope=('Every solver-input point in each captured packet is projected after LM. '
                        'Metrics are crop-pixel diagnostics only; they are not full-retained-point '
                        'contract validation or independent accuracy.'),
        reference_or_annotation_inputs=False,
        output_root=str(output_root), started_unix=time.time(), packets=[],
    )
    save_result(output_root / 'replay.json', report)
    try:
        report['status'] = 'running'
        report['source_snapshot'] = _snapshot_sources(output_root)
        capture_path, capture, entries, packet_bytes = _load_capture(args.capture_manifest)
        report.update(
            runtime=_runtime_info(),
            capture_manifest_sha256=digest(capture_path),
            capture_packet_bytes=packet_bytes,
            capture_source_snapshot=capture.get('source_snapshot'),
            inference_provenance=capture.get('inference_provenance'),
            capture_runtime=capture.get('runtime'),
            replay_source_sha256=digest(ROOT / 'bench/quality_pnp_mug_replay.py'),
        )
        save_result(output_root / 'replay.json', report)
        for entry in entries:
            frame_id, iteration, rng_seed, arrays = _load_packet(capture_path, entry)
            pair = dict(
                frame_id=frame_id, iteration=iteration, rng_seed=rng_seed,
                packet_path=entry['path'], packet_sha256=entry['sha256'],
                packet_input_hashes=entry['input_hashes'],
                arrays={name: dict(shape=list(arrays[name].shape), dtype=str(arrays[name].dtype),
                                   sha256=_array_sha256(arrays[name])) for name in SOLVER_FIELDS},
                branches={},
            )
            for mode_name, use_guess in (
                ('control_use_extrinsic_guess', True),
                ('candidate_use_extrinsic_guess', False),
            ):
                pair['branches'][mode_name] = replay_solver_inputs(
                    arrays['obj_points'], arrays['target_points'], arrays['crop_k'],
                    arrays['rvec'], arrays['tvec'], rng_seed, use_guess,
                )
            report['packets'].append(pair)
            save_result(output_root / 'replay.json', report)
            print(frame_id, iteration,
                  {name: branch['fit_state'] for name, branch in pair['branches'].items()},
                  flush=True)
        report['status'] = 'complete'
        report['complete'] = True
        report['completed_unix'] = time.time()
        report['independent_accuracy_verified'] = False
        report['full_window_benchmark'] = False
        report['overall_quality_gate_passed'] = False
        save_result(output_root / 'replay.json', report)
    except BaseException as error:
        report['status'] = 'failed'
        report['complete'] = False
        report['failure'] = dict(type=type(error).__name__, message=str(error))
        report['failed_unix'] = time.time()
        try:
            save_result(output_root / 'replay.json', report)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
