"""CPU replay of full-retained mug PnP packets in native camera coordinates.

For every captured iteration, replay True and False from fresh copies of the
full retained set and exact sampled IDs. It converts the crop-camera fit through
the captured native-to-crop transform, reconstructs every retained native pixel
correspondence, and calls the unchanged current-mask quality contract. A True
replay is checked against the actual captured control candidate. No model,
reference pose, evaluation annotation, or RGB appearance check is used.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .quality_assets import CACHE, ROOT, digest, save_result
from .quality_contract import Frame, PoseCandidate, canonical_pose, pose_units, validate
from .quality_pnp_mug_native_probe import (
    FRAME_IDS, ITERATIONS, PACKET_BUDGET_BYTES,
    PACKET_INPUT_FIELDS, PACKET_RESULT_FIELDS,
    VALIDATION_STATS_ATOL, VALIDATION_STATS_RTOL,
    _array_info, _array_sha256, _load_sampled_boundary,
)
from .vision import cv2


REPLAY_ARRAYS = set(PACKET_INPUT_FIELDS) | set(PACKET_RESULT_FIELDS)
SOURCE_FILES = (
    'bench/quality_pnp_mug_native_replay.py',
    'bench/quality_pnp_mug_native_probe.py',
    'bench/quality_pnp_mug_replay.py',
    'bench/quality_gotrack.py',
    'bench/quality_assets.py',
    'bench/quality_contract.py',
    'bench/vision.py',
)
NUMERIC_VALIDATION_STATS = (
    'median_reprojection_720', 'p95_reprojection_720', 'spatial_support', 'score',
)
INTEGER_VALIDATION_STATS = ('correspondences', 'inliers')


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _snapshot_sources(output_root):
    copied = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f'Missing fixed native replay source: {relative}')
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


def _safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe relative path in capture manifest: {relative}')
    root = root.resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path escaped capture root: {relative}')
    return result


def _validate_stats_tolerance(capture):
    tolerance = capture.get('validation_stats_tolerance')
    expected_keys = {'atol', 'rtol', 'frozen_before_capture'}
    if not isinstance(tolerance, dict) or set(tolerance) != expected_keys:
        raise ValueError('Native capture must record the reviewed frozen validation-stat tolerances')
    if (type(tolerance['atol']) is not float or tolerance['atol'] != VALIDATION_STATS_ATOL or
            type(tolerance['rtol']) is not float or tolerance['rtol'] != VALIDATION_STATS_RTOL or
            type(tolerance['frozen_before_capture']) is not bool or
            tolerance['frozen_before_capture'] is not True):
        raise ValueError('Native capture validation-stat tolerances differ from the reviewed frozen values')


def _load_capture(path):
    manifest_path = Path(path).expanduser().resolve()
    capture = json.loads(manifest_path.read_text(encoding='utf-8'))
    _validate_stats_tolerance(capture)
    if capture.get('object') != 'mug' or capture.get('requested_frame_ids') != FRAME_IDS:
        raise ValueError('Expected the fixed ten-frame mug native capture')
    if capture.get('requested_iterations') != ITERATIONS:
        raise ValueError('Expected the fixed five-iteration GoTrack refinement')
    if capture.get('status') != 'complete' or not capture.get('complete'):
        raise ValueError('Native capture manifest must be complete before replay')
    frames = capture.get('frames', [])
    if [int(frame['frame_id']) for frame in frames] != FRAME_IDS:
        raise ValueError('Native capture must account for every selected frame in requested order')
    for frame in frames:
        rows = frame.get('iteration_accounting', [])
        if [int(row['iteration']) for row in rows] != list(range(ITERATIONS)):
            raise ValueError(f'Frame {frame["frame_id"]} lacks explicit five-iteration accounting')
    total = int(capture.get('aggregate_capture_bytes', -1))
    expected = int(capture.get('captured_packet_bytes', -1)) + int(capture.get('observed_mask_bytes', -1))
    if total != expected or total < 0 or total > PACKET_BUDGET_BYTES:
        raise ValueError('Native capture aggregate packet budget accounting is invalid')
    packets = capture.get('packets', [])
    keys = [(int(item['frame_id']), int(item['iteration'])) for item in packets]
    if len(keys) != len(set(keys)):
        raise ValueError('Native capture has duplicate frame/iteration packets')
    expected_keys = {
        (int(frame['frame_id']), int(row['iteration']))
        for frame in frames for row in frame['iteration_accounting'] if row['attempted']
    }
    if set(keys) != expected_keys:
        raise ValueError('Attempted iteration accounting differs from captured packet set')
    if sum(int(item['bytes']) for item in packets) != int(capture.get('captured_packet_bytes', -1)):
        raise ValueError('Native packet byte total differs from the manifest')
    return manifest_path, capture


def _load_packet(manifest_path, entry):
    packet_path = _safe_path(manifest_path.parent, entry['path'])
    if (not packet_path.is_file() or packet_path.stat().st_size != int(entry['bytes']) or
            digest(packet_path) != entry['sha256']):
        raise ValueError(f'Native packet size/hash mismatch: {entry["path"]}')
    with np.load(packet_path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = set(PACKET_INPUT_FIELDS) | {'frame_id', 'iteration', 'rng_seed'}
        if not required.issubset(names) or not names.issubset(REPLAY_ARRAYS | {'frame_id', 'iteration', 'rng_seed'}):
            raise ValueError(f'Unexpected arrays in native packet: {entry["path"]}')
        arrays = {name: np.array(archive[name], copy=True) for name in names - {'frame_id', 'iteration', 'rng_seed'}}
        frame_id, iteration, rng_seed = int(archive['frame_id']), int(archive['iteration']), int(archive['rng_seed'])
    if (frame_id != int(entry['frame_id']) or iteration != int(entry['iteration']) or
            rng_seed != int(entry['rng_seed'])):
        raise ValueError(f'Native packet metadata mismatch: {entry["path"]}')
    for name, array in arrays.items():
        info = entry['arrays'].get(name)
        if (info is None or list(array.shape) != info['shape'] or str(array.dtype) != info['dtype'] or
                _array_sha256(array) != info['sha256']):
            raise ValueError(f'Native packet array metadata/hash mismatch for {name}: {entry["path"]}')
    n = len(arrays['full_obj_points_mm'])
    ids = arrays['sample_ids']
    expected_ids = np.linspace(0, n-1, min(n, 10000), dtype=int)
    if (arrays['full_obj_points_mm'].shape != (n, 3) or
            arrays['full_target_crop_px'].shape != (n, 2) or
            arrays['full_weights'].shape != (n,) or
            ids.ndim != 1 or not np.array_equal(ids, expected_ids) or
            len(ids) != int(entry['sampled_count']) or n != int(entry['retained_count'])):
        raise ValueError(f'Invalid full retained arrays or sample mapping: {entry["path"]}')
    for name in ('full_obj_points_mm', 'full_target_crop_px', 'full_weights', 'crop_k',
                 'crop_from_orig', 'native_k', 'current_crop_pose_mm', 'initial_rvec', 'initial_tvec_mm'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'Nonfinite native packet input {name}: {entry["path"]}')
    return frame_id, iteration, rng_seed, arrays


def verify_sampled_boundary(arrays, entry, sampled):
    sample = sampled['arrays']
    ids = arrays['sample_ids']
    pairs = (
        (arrays['full_obj_points_mm'][ids], sample['obj_points'], 'full sampled object points'),
        (arrays['full_target_crop_px'][ids], sample['target_points'], 'full sampled target points'),
        (arrays['crop_k'], sample['crop_k'], 'crop intrinsics'),
        (arrays['initial_rvec'], sample['rvec'], 'initial crop rotation'),
        (arrays['initial_tvec_mm'], sample['tvec'], 'initial crop translation'),
    )
    verified_hashes = {}
    for actual, expected, label in pairs:
        if not np.array_equal(actual, expected):
            raise ValueError(f'Sampled arrays differ from the preliminary boundary: {label}')
        verified_hashes[label] = _array_sha256(actual)
    if entry['sample_boundary'].get('exact_array_match') is not True:
        raise ValueError('Native capture did not record a matched preliminary packet boundary')
    for name, value in entry['sample_boundary']['arrays'].items():
        corresponding = {
            'obj_points': arrays['full_obj_points_mm'][ids],
            'target_points': arrays['full_target_crop_px'][ids],
            'crop_k': arrays['crop_k'], 'rvec': arrays['initial_rvec'],
            'tvec': arrays['initial_tvec_mm'],
        }[name]
        if _array_sha256(corresponding) != value['sha256']:
            raise ValueError(f'Captured sampled-boundary hash mismatch for {name}')
    return verified_hashes


def _summary(values):
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        return dict(count=0, mean=None, rmse=None, median=None, p95=None, p99=None, maximum=None)
    return dict(count=int(data.size), mean=float(np.mean(data)),
                rmse=float(np.sqrt(np.mean(data**2))), median=float(np.median(data)),
                p95=float(np.percentile(data, 95)), p99=float(np.percentile(data, 99)),
                maximum=float(np.max(data)))


def _residual_accounting(points, projected, observed, depths, scale=1., depth_threshold=.01,
                         depth_units='metres'):
    points = np.asarray(points)
    projected = np.asarray(projected)
    observed = np.asarray(observed)
    depths = np.asarray(depths)
    finite_points = np.isfinite(points).all(axis=1)
    finite_projected = np.isfinite(projected).all(axis=1)
    finite_observed = np.isfinite(observed).all(axis=1)
    finite_depth = np.isfinite(depths)
    finite_residual = finite_projected & finite_observed & np.isfinite(depths)
    errors = np.full(len(points), np.nan, dtype=np.float64)
    errors[finite_residual] = np.linalg.norm(projected[finite_residual] - observed[finite_residual], axis=1) * scale
    finite_residual &= np.isfinite(errors)
    positive_depth = finite_depth & (depths > depth_threshold)
    negative_or_near_depth = finite_depth & ~positive_depth
    return dict(
        retained_count=int(len(points)), finite_object_points=int(finite_points.sum()),
        nonfinite_object_points=int((~finite_points).sum()),
        finite_projected_points=int(finite_projected.sum()),
        nonfinite_projected_points=int((~finite_projected).sum()),
        finite_observed_points=int(finite_observed.sum()),
        nonfinite_observed_points=int((~finite_observed).sum()),
        finite_depth_points=int(finite_depth.sum()), nonfinite_depth_points=int((~finite_depth).sum()),
        positive_depth_points=int(positive_depth.sum()),
        behind_or_near_depth_points=int(negative_or_near_depth.sum()),
        depth_units=depth_units, positive_depth_threshold=float(depth_threshold),
        finite_residual_points=int(finite_residual.sum()),
        nonfinite_residual_points=int((~finite_residual).sum()),
        residual_all_finite=_summary(errors[finite_residual]),
        residual_positive_depth=_summary(errors[finite_residual & positive_depth]),
    )


def _replay_native_packet(arrays, rng_seed, use_extrinsic_guess, native_height=720, cv2_module=None):
    cv = cv2 if cv2_module is None else cv2_module
    if type(use_extrinsic_guess) is not bool:
        raise TypeError('use_extrinsic_guess must be a bool')
    full_obj = np.array(arrays['full_obj_points_mm'], copy=True, order='C')
    full_target = np.array(arrays['full_target_crop_px'], copy=True, order='C')
    ids = np.array(arrays['sample_ids'], copy=True, order='C')
    crop_k = np.array(arrays['crop_k'], copy=True, order='C')
    initial_rvec = np.array(arrays['initial_rvec'], copy=True, order='C')
    initial_tvec = np.array(arrays['initial_tvec_mm'], copy=True, order='C')
    crop_from_orig = np.array(arrays['crop_from_orig'], copy=True, order='C')
    native_k = np.array(arrays['native_k'], copy=True, order='C')
    weights = np.array(arrays['full_weights'], copy=True, order='C')
    n = len(full_obj)
    if (full_obj.shape != (n, 3) or full_target.shape != (n, 2) or weights.shape != (n,) or
            ids.ndim != 1 or crop_k.shape != (3, 3) or native_k.shape != (3, 3) or
            crop_from_orig.shape != (4, 4) or initial_rvec.size != 3 or initial_tvec.size != 3 or
            len(ids) != min(n, 10000) or not np.isfinite(np.r_[full_obj.ravel(), full_target.ravel(),
                weights, crop_k.ravel(), native_k.ravel(), crop_from_orig.ravel(),
                initial_rvec.ravel(), initial_tvec.ravel()]).all()):
        raise ValueError('Invalid finite full-retained native replay packet')
    sample_obj = np.array(full_obj[ids], copy=True, order='C')
    sample_target = np.array(full_target[ids], copy=True, order='C')
    initial_rvec_captured = initial_rvec.copy()
    initial_tvec_captured = initial_tvec.copy()
    cv.setRNGSeed(int(rng_seed) & 0x7fffffff)
    try:
        success, solved_rvec, solved_tvec, inliers = cv.solvePnPRansac(
            sample_obj, sample_target, crop_k, None,
            rvec=initial_rvec, tvec=initial_tvec,
            useExtrinsicGuess=use_extrinsic_guess,
            iterationsCount=3000, reprojectionError=2., confidence=.999,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
    except cv.error as error:
        return dict(
            use_extrinsic_guess=use_extrinsic_guess, solver_success=False,
            rng_seed=int(rng_seed),
            fit_state='ransac_exception', exception=f'{type(error).__name__}: {error}',
            initial_rvec=initial_rvec_captured.reshape(3).tolist(),
            initial_tvec_mm=initial_tvec_captured.reshape(3).tolist(),
            ransac_inlier_ids=[], retained_inlier_ids=[], candidate=None,
        )
    inlier_ids = (np.empty(0, dtype=np.int32) if inliers is None
                  else np.asarray(inliers, dtype=int).reshape(-1))
    chosen = ids[inlier_ids] if inliers is not None else np.empty(0, dtype=int)
    base = dict(
        use_extrinsic_guess=use_extrinsic_guess, solver_success=bool(success),
        rng_seed=int(rng_seed),
        initial_rvec=initial_rvec_captured.reshape(3).tolist(),
        initial_tvec_mm=initial_tvec_captured.reshape(3).tolist(),
        input_retained_count=int(n), sampled_input_count=int(len(ids)),
        ransac_inlier_count=int(len(inlier_ids)), ransac_inlier_ids=inlier_ids.tolist(),
        retained_ransac_inlier_ids=chosen.astype(int).tolist(), candidate=None,
    )
    if not success or inliers is None or len(inlier_ids) < 6:
        base['fit_state'] = 'ransac_unavailable'
        base['full_retained_crop_residuals'] = None
        base['full_retained_native_residuals_720'] = None
        return base
    try:
        solved_rvec, solved_tvec = cv.solvePnPRefineLM(
            full_obj[chosen], full_target[chosen], crop_k, None, solved_rvec, solved_tvec,
        )
    except cv.error as error:
        base.update(fit_state='lm_exception', exception=f'{type(error).__name__}: {error}')
        return base
    solved_rvec = np.asarray(solved_rvec, dtype=np.float64).reshape(3, 1)
    solved_tvec = np.asarray(solved_tvec, dtype=np.float64).reshape(3, 1)
    crop_pose = np.eye(4, dtype=np.float64)
    crop_pose[:3, :3] = cv.Rodrigues(solved_rvec)[0]
    crop_pose[:3, 3] = solved_tvec[:, 0]
    native_pose_mm = canonical_pose(np.linalg.inv(crop_from_orig) @ crop_pose)
    native_pose_m = pose_units(native_pose_mm, .001)

    rays = np.c_[full_target, np.ones(n)] @ np.linalg.inv(crop_k).T
    rays_native = rays @ crop_from_orig[:3, :3]
    projected_h = rays_native @ native_k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        native_pixels = projected_h[:, :2] / projected_h[:, 2:3]
    candidate = PoseCandidate(native_pose_m, full_obj * .001, native_pixels, weights)

    crop_camera_points = full_obj @ crop_pose[:3, :3].T + crop_pose[:3, 3]
    crop_projected, _ = cv.projectPoints(full_obj, solved_rvec, solved_tvec, crop_k, None)
    crop_projected = np.asarray(crop_projected).reshape(-1, 2)
    crop_diag = _residual_accounting(
        crop_camera_points, crop_projected, full_target, crop_camera_points[:, 2],
        scale=1., depth_threshold=10., depth_units='millimetres')

    native_camera_points = candidate.points_object_m @ native_pose_m[:3, :3].T + native_pose_m[:3, 3]
    native_projected_h = native_camera_points @ native_k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        native_projected = native_projected_h[:, :2] / native_projected_h[:, 2:3]
    native_diag = _residual_accounting(
        native_camera_points, native_projected, native_pixels, native_camera_points[:, 2],
        scale=720. / int(native_height), depth_threshold=.01, depth_units='metres')
    base.update(
        fit_state='refined',
        refined_rvec=solved_rvec[:, 0].tolist(), refined_tvec_mm=solved_tvec[:, 0].tolist(),
        crop_camera_from_object_mm=crop_pose.tolist(), native_camera_from_object_m=native_pose_m.tolist(),
        crop_pose_translation_units='millimetres', native_pose_translation_units='metres',
        full_retained_crop_residuals=crop_diag,
        full_retained_native_residuals_720=native_diag,
        candidate=dict(
            pose_m=native_pose_m,
            points_object_m=candidate.points_object_m,
            pixels_native=candidate.pixels_image,
            weights=candidate.weights,
        ),
    )
    return base


def replay_native_packet(arrays, rng_seed, use_extrinsic_guess, frame, mask, cv2_module=None):
    result = _replay_native_packet(
        arrays, rng_seed, use_extrinsic_guess, frame.rgb.shape[0], cv2_module)
    if result['candidate'] is None:
        result['validation_state'] = 'not_run'
        result['validation_reason'] = f'candidate_unavailable_{result["fit_state"]}'
        result['validation_stats'] = None
        result['candidate_arrays'] = {}
        result['candidate'] = None
        return result
    candidate_arrays = result['candidate']
    candidate = PoseCandidate(
        candidate_arrays['pose_m'], candidate_arrays['points_object_m'],
        candidate_arrays['pixels_native'], candidate_arrays['weights'],
    )
    valid, reason, stats = validate(candidate, frame, mask)
    result.update(
        validation_state='accepted' if valid else 'rejected',
        validation_reason=reason,
        validation_stats=stats,
        candidate_arrays={name: _array_info(value) for name, value in candidate_arrays.items()},
        candidate={name: (value.tolist() if isinstance(value, np.ndarray) else value)
                   for name, value in candidate_arrays.items()},
    )
    return result


def compare_control_capture(replayed, expected, pose_atol=1e-6,
                            stats_atol=VALIDATION_STATS_ATOL,
                            stats_rtol=VALIDATION_STATS_RTOL):
    """Compare replay True with the actual per-iteration control capture."""
    mismatches = []
    for field in ('fit_state', 'solver_success', 'rng_seed', 'input_retained_count',
                  'sampled_input_count', 'ransac_inlier_count',
                  'ransac_inlier_ids', 'retained_inlier_ids'):
        got = replayed.get(field)
        want = expected.get(field)
        if field == 'input_retained_count':
            got = replayed.get('input_retained_count')
            want = expected.get('retained_count')
        elif field == 'sampled_input_count':
            got = replayed.get('sampled_input_count')
            want = expected.get('sampled_count')
        if field == 'retained_inlier_ids':
            got = replayed.get('retained_ransac_inlier_ids')
            want = expected.get('retained_inlier_ids')
        if got != want:
            mismatches.append(field)
    if expected.get('fit_state') == 'refined' and replayed.get('candidate') is not None:
        actual_pose = np.asarray(replayed['candidate']['pose_m'], dtype=np.float64)
        expected_pose = np.asarray(expected['pose_m'], dtype=np.float64)
        if not np.allclose(actual_pose, expected_pose, atol=pose_atol, rtol=0):
            mismatches.append('native_pose_atol_1e-6_rtol_0')
        replay_arrays = replayed.get('candidate_arrays') or {}
        expected_arrays = expected.get('candidate_arrays') or {}
        for name in ('points_object_m', 'pixels_native', 'weights'):
            if replay_arrays.get(name) != expected_arrays.get(name):
                mismatches.append(f'candidate_correspondence_hashes.{name}')
        replay_pose_info = replay_arrays.get('pose_m') or {}
        expected_pose_info = expected_arrays.get('pose_m') or {}
        if (replay_pose_info.get('shape') != expected_pose_info.get('shape') or
                replay_pose_info.get('dtype') != expected_pose_info.get('dtype')):
            mismatches.append('candidate_pose_shape_dtype')
        for field in ('validation_state', 'validation_reason'):
            if replayed.get(field) != expected.get(field):
                mismatches.append(field)
        actual_stats = replayed.get('validation_stats')
        expected_stats = expected.get('validation_stats')
        if (actual_stats is None) != (expected_stats is None):
            mismatches.append('validation_stats_presence')
        elif actual_stats is not None:
            for name in INTEGER_VALIDATION_STATS:
                if actual_stats.get(name) != expected_stats.get(name):
                    mismatches.append(f'validation_stats.{name}')
            for name in NUMERIC_VALIDATION_STATS:
                left, right = actual_stats.get(name), expected_stats.get(name)
                if left is None or right is None:
                    if left != right:
                        mismatches.append(f'validation_stats.{name}')
                elif not np.isclose(float(left), float(right), atol=stats_atol, rtol=stats_rtol):
                    mismatches.append(f'validation_stats.{name}')
    elif expected.get('fit_state') != 'refined':
        for field in ('validation_state', 'validation_reason'):
            if replayed.get(field) != expected.get(field):
                mismatches.append(field)
    return dict(
        matched=not mismatches, mismatches=mismatches,
        tolerances=dict(pose_atol=pose_atol, pose_rtol=0,
                        validation_stats_atol=stats_atol, validation_stats_rtol=stats_rtol),
    )


def _compact_branch_record(branch):
    """Keep candidate pose and hashes in JSON; full correspondences stay in NPZ."""
    compact = dict(branch)
    candidate = branch.get('candidate')
    if isinstance(candidate, dict):
        compact['candidate'] = ({'pose_m': candidate['pose_m']}
                                if 'pose_m' in candidate else {})
    return compact


def _compact_replay_pair(pair):
    """Return detailed compact frame record and a small packet-index row."""
    detail = dict(pair)
    branches = pair.get('branches', {})
    detail['branches'] = {
        name: _compact_branch_record(branch) for name, branch in branches.items()
    }
    packet_index = {
        name: pair[name] for name in
        ('frame_id', 'iteration', 'attempted', 'packet_path', 'packet_sha256', 'control_equivalence')
        if name in pair
    }
    packet_index['branches'] = {
        name: {field: branch.get(field) for field in
               ('fit_state', 'solver_success', 'validation_state', 'validation_reason')}
        for name, branch in branches.items()
    }
    return detail, packet_index


def _load_observed_mask(capture_root, frame):
    info = frame.get('observed_mask')
    if not info:
        raise ValueError(f'No copied observed mask metadata for frame {frame["frame_id"]}')
    path = _safe_path(capture_root, info['path'])
    if not path.is_file() or path.stat().st_size != int(info['bytes']) or digest(path) != info['file_sha256']:
        raise ValueError(f'Observed mask file hash mismatch for frame {frame["frame_id"]}')
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or list(mask.shape) != info['array']['shape'] or str(mask.dtype) != info['array']['dtype']:
        raise ValueError(f'Observed mask shape/type mismatch for frame {frame["frame_id"]}')
    if _array_sha256(mask) != info['array']['sha256']:
        raise ValueError(f'Observed mask array hash mismatch for frame {frame["frame_id"]}')
    return mask, info


def _runtime_info():
    return dict(device='cpu', python=sys.version, platform=platform.platform(),
                numpy=np.__version__, opencv=cv2.__version__, torch_loaded=False,
                ransac_rng='reset to captured frame/iteration seed before each solver option')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-manifest', type=Path, required=True,
                        help='Complete full-retained native capture.json')
    parser.add_argument('--sampled-capture-manifest', type=Path,
                        default=CACHE / 'diagnostics/mug-local-pnp-v1/capture.json',
                        help='Preserved preliminary sampled-input boundary to reverify')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory for the CPU native replay report')
    args = parser.parse_args(argv)
    output_root = _create_output_root(args.output_root)
    report = dict(
        schema_version=1, scope=__doc__, status='initializing', object='mug',
        capture_manifest=str(args.capture_manifest.expanduser().resolve()),
        branches=dict(control_use_extrinsic_guess=True, candidate_use_extrinsic_guess=False),
        solver=dict(iterations=3000, reprojection_error_crop_px=2.0,
                    confidence=0.999, flags='SOLVEPNP_ITERATIVE', lm='solvePnPRefineLM'),
        pose_comparison_tolerance=dict(atol=1e-6, rtol=0),
        validation_stats_tolerance=dict(atol=VALIDATION_STATS_ATOL, rtol=VALIDATION_STATS_RTOL,
                                        frozen_before_capture=True),
        residual_scope=('Crop and native residual/depth/finite summaries cover the entire retained set. '
                        'Native candidate validation uses all retained points and the copied observed mask.'),
        reference_or_annotation_inputs=False,
        output_root=str(output_root), started_unix=time.time(), frames=[], packets=[],
    )
    save_result(output_root / 'replay.json', report)
    try:
        report['status'] = 'running'
        report['source_snapshot'] = _snapshot_sources(output_root)
        capture_path, capture = _load_capture(args.capture_manifest)
        sampled_path, sampled_capture, sampled_packets, sampled_bytes = _load_sampled_boundary(
            args.sampled_capture_manifest)
        report.update(
            runtime=_runtime_info(), capture_manifest_sha256=digest(capture_path),
            capture_source_snapshot=capture.get('source_snapshot'),
            capture_inference_provenance=capture.get('inference_provenance'),
            capture_packet_bytes=int(capture['captured_packet_bytes']),
            capture_observed_mask_bytes=int(capture['observed_mask_bytes']),
            capture_aggregate_bytes=int(capture['aggregate_capture_bytes']),
            sampled_boundary=dict(path=str(sampled_path), sha256=digest(sampled_path),
                                  packet_count=len(sampled_packets), packet_bytes=sampled_bytes),
            replay_source_sha256=digest(ROOT / 'bench/quality_pnp_mug_native_replay.py'),
        )
        frames_by_id = {int(frame['frame_id']): frame for frame in capture['frames']}
        sampled_frames = {
            int(frame['frame_id']): frame for frame in sampled_capture.get('frames', [])
        }
        if set(sampled_frames) != set(FRAME_IDS):
            raise ValueError('Preliminary sampled capture lacks all ten frame provenance records')
        packet_entries = {
            (int(entry['frame_id']), int(entry['iteration'])): entry for entry in capture['packets']
        }
        for frame_id in FRAME_IDS:
            frame_capture = frames_by_id[frame_id]
            mask, mask_info = _load_observed_mask(capture_path.parent, frame_capture)
            prior_hashes = sampled_frames[frame_id].get('input_hashes', {})
            current_hashes = frame_capture.get('input_hashes', {})
            for name in ('rgb_sha256', 'mask_sha256', 'seed_pose_sha256', 'mask_artifact_sha256'):
                if current_hashes.get(name) != prior_hashes.get(name):
                    raise ValueError(f'Frame {frame_id} {name} differs from preliminary sampled capture')
            if mask_info.get('file_sha256') != current_hashes.get('mask_artifact_sha256'):
                raise ValueError(f'Copied observed mask file is not the captured frame mask for {frame_id}')
            shape = tuple(mask.shape)
            dummy_rgb = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
            # RGB content is not used by validate; only its native dimensions are required.
            frame_row = dict(frame_id=frame_id, observed_mask=mask_info, iterations=[])
            report['frames'].append(frame_row)
            for iteration in range(ITERATIONS):
                key = (frame_id, iteration)
                entry = packet_entries.get(key)
                accounting = frame_capture['iteration_accounting'][iteration]
                pair = dict(frame_id=frame_id, iteration=iteration,
                            attempted=bool(accounting['attempted']),
                            capture_accounting=accounting, branches={})
                if entry is None:
                    reason = accounting.get('reason') or 'iteration_not_captured'
                    for mode_name, use_guess in (
                        ('control_use_extrinsic_guess', True),
                        ('candidate_use_extrinsic_guess', False),
                    ):
                        pair['branches'][mode_name] = dict(
                            use_extrinsic_guess=use_guess, fit_state='not_attempted',
                            solver_success=False, candidate=None, validation_state='not_run',
                            validation_reason=reason, validation_stats=None,
                            full_retained_crop_residuals=None,
                            full_retained_native_residuals_720=None,
                        )
                    pair['control_equivalence'] = dict(matched=None, reason='no captured attempted packet')
                    detail, packet_index = _compact_replay_pair(pair)
                    frame_row['iterations'].append(detail)
                    report['packets'].append(packet_index)
                    continue

                frame_id_loaded, iteration_loaded, rng_seed, arrays = _load_packet(capture_path, entry)
                sampled = sampled_packets.get(key)
                if sampled is None:
                    raise ValueError(f'Missing preliminary sampled packet for {key}')
                sample_hashes = verify_sampled_boundary(arrays, entry, sampled)
                if not np.array_equal(arrays['native_k'], np.asarray(capture['input_intrinsics'], dtype=arrays['native_k'].dtype)):
                    raise ValueError(f'Native intrinsics differ from input manifest for frame {frame_id}')
                native_frame = Frame(frame_id, dummy_rgb, arrays['native_k'])
                pair.update(
                    rng_seed=rng_seed, packet_path=entry['path'], packet_sha256=entry['sha256'],
                    packet_array_hashes={name: _array_info(value) for name, value in arrays.items()},
                    sampled_boundary_verified=True, sampled_boundary_hashes=sample_hashes,
                    observed_mask_hash=mask_info['array']['sha256'],
                )
                control = replay_native_packet(arrays, rng_seed, True, native_frame, mask)
                candidate = replay_native_packet(arrays, rng_seed, False, native_frame, mask)
                expected = entry['expected_candidate']
                equivalence = compare_control_capture(control, expected)
                pair['branches']['control_use_extrinsic_guess'] = control
                pair['branches']['candidate_use_extrinsic_guess'] = candidate
                pair['control_equivalence'] = equivalence
                pair['control_expected_capture'] = dict(
                    fit_state=expected['fit_state'], solver_success=expected['solver_success'],
                    ransac_inlier_count=expected['ransac_inlier_count'],
                    ransac_inlier_ids=expected['ransac_inlier_ids'],
                    retained_inlier_ids=expected['retained_inlier_ids'],
                    pose_m=expected['pose_m'],
                    validation_state=expected['validation_state'],
                    validation_reason=expected['validation_reason'],
                    validation_stats=expected['validation_stats'],
                    candidate_arrays=expected['candidate_arrays'],
                )
                detail, packet_index = _compact_replay_pair(pair)
                frame_row['iterations'].append(detail)
                report['packets'].append(packet_index)
                print(frame_id, iteration, control['fit_state'], candidate['fit_state'],
                      'control_match='+str(equivalence['matched']), flush=True)
                save_result(output_root / 'replay.json', report)

        total_rows = sum(len(frame['iterations']) for frame in report['frames'])
        if total_rows != len(FRAME_IDS) * ITERATIONS:
            raise ValueError('Replay omitted one or more requested frame/iteration records')
        report['status'] = 'complete'
        report['complete'] = True
        report['completed_unix'] = time.time()
        report['control_equivalence_all_matched'] = all(
            row['control_equivalence']['matched'] is True
            for frame in report['frames'] for row in frame['iterations']
            if row['attempted']
        )
        report['all_ten_frames_accounted'] = len(report['frames']) == 10
        report['all_fifty_iteration_slots_accounted'] = total_rows == 50
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
