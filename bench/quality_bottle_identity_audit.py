"""CPU-only audit for frozen bottle surface-identity correspondence packets.

Synthetic records contain model-geometry oracle labels. Real records contain
only current-image patch support and learned correspondences; no real pose
accuracy score or evaluation annotation is read here.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .quality_assets import CACHE, ROOT, digest, save_result
from .quality_contract import Frame, PoseCandidate, canonical_pose, validate
from .vision import cv2


FRAME_IDS = (10, 50, 100, 150, 185, 209, 217, 225, 230, 249)
SYNTHETIC_CARRIERS = (10, 50, 100, 209)
QUERY_OFFSETS = (8, 188)
TEMPLATE_OFFSETS = (0, 180)
APPEARANCES = ('rgb', 'gray')
FORWARD_CAP = 108
PACKET_BUDGET_BYTES = 128 * 1024**2
PATCH_SIZE = 11
PATCH_RADIUS = PATCH_SIZE // 2
PATCH_STD_MIN = .005
NCC_MIN = .75
NCC_MARGIN_MIN = .10
IDENTITY_DISTANCE_FRACTION = .1
GRAY_CHANGE_MIN = .05
GRAY_RECOVERY_IMPROVEMENT_MIN = .15
VISIBILITY_WEIGHT = .3


def array_sha256(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def array_info(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=array_sha256(array))


def _float_rgb(rgb):
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Expected HxWx3 RGB image')
    value = value.astype(np.float32, copy=False)
    if value.size and float(np.nanmax(value)) > 1.5:
        value = value / 255.
    return np.clip(value, 0., 1.)


def gray_rgb(rgb, foreground_mask):
    """Replicate luminance on foreground only, preserving the exact background."""
    image = _float_rgb(rgb)
    mask = np.asarray(foreground_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError('Appearance mask shape differs from RGB image')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result = image.copy()
    result[mask] = gray[mask][:, None]
    delta = np.max(np.abs(result - image), axis=2)
    changed = mask & (delta >= GRAY_CHANGE_MIN)
    chroma = image.max(axis=2) - image.min(axis=2)
    foreground_count = int(mask.sum())
    changed_count = int(changed.sum())
    return result, dict(
        foreground_pixels=foreground_count,
        changed_pixels_ge_005=changed_count,
        changed_pixel_fraction=changed_count / max(1, foreground_count),
        chroma_fraction_ge_005=int((mask & (chroma >= GRAY_CHANGE_MIN)).sum()) / max(1, foreground_count),
        state='informative' if changed_count / max(1, foreground_count) >= .05 else 'appearance_ablation_uninformative',
    )


def object_points_from_depth(depth_mm, intrinsics, crop_camera_from_object_mm, mask=None):
    """Unproject upstream pixel centres and invert a crop-camera object pose."""
    depth = np.asarray(depth_mm, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(crop_camera_from_object_mm, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError('Invalid depth, intrinsics or crop pose shape')
    keep = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != depth.shape:
            raise ValueError('Template mask and depth shapes differ')
        keep &= supplied
    flat = np.flatnonzero(keep.reshape(-1))
    yy, xx = np.unravel_index(flat, depth.shape)
    pixels = np.column_stack((xx + .5, yy + .5)).astype(np.float64)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
    camera_points_mm = rays * depth.reshape(-1)[flat, None]
    object_points_m = (camera_points_mm - pose[:3, 3]) @ pose[:3, :3] * .001
    good = np.isfinite(object_points_m).all(axis=1)
    return flat[good].astype(np.int64), pixels[good], object_points_m[good]


def project_object_points(points_object_m, camera_from_object_m, intrinsics):
    points = np.asarray(points_object_m, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(camera_from_object_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    projected = camera_points @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = projected[:, :2] / projected[:, 2:3]
    return pixels, camera_points[:, 2]


def native_to_crop_pose_mm(native_pose_m, crop_from_native):
    pose_mm = np.array(native_pose_m, dtype=np.float64, copy=True)
    pose_mm[:3, 3] *= 1000.
    return np.asarray(crop_from_native, dtype=np.float64) @ pose_mm


def crop_to_native_pose_m(crop_pose_mm, crop_from_native):
    native_mm = canonical_pose(
        np.linalg.inv(np.asarray(crop_from_native, dtype=np.float64)) @
        np.asarray(crop_pose_mm, dtype=np.float64))
    native_mm = native_mm.copy()
    native_mm[:3, 3] *= .001
    return native_mm


def projection_parity(source_points_m, source_pixels, camera_from_object_m, intrinsics):
    projected, depth = project_object_points(source_points_m, camera_from_object_m, intrinsics)
    errors = np.linalg.norm(projected - np.asarray(source_pixels, dtype=np.float64), axis=1)
    finite = (depth > 0) & np.isfinite(errors)
    summary = dict(
        points=int(len(errors)), finite_positive_depth=int(finite.sum()),
        max_error_px=float(errors[finite].max()) if finite.any() else None,
        median_error_px=float(np.median(errors[finite])) if finite.any() else None,
        threshold_px=.01,
    )
    summary['passed'] = bool(finite.all() and finite.any() and summary['max_error_px'] <= .01)
    return summary


def _floor_indices(points_xy, width, height):
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(xy).all(axis=1)
    ix = np.zeros(len(xy), dtype=np.int64)
    iy = np.zeros(len(xy), dtype=np.int64)
    ix[finite] = np.floor(xy[finite, 0]).astype(np.int64)
    iy[finite] = np.floor(xy[finite, 1]).astype(np.int64)
    inside = finite & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    return ix, iy, inside


def _spatial_anchor_rows(points_xy, eligible, confidence, grid=(4, 4)):
    xy = np.asarray(points_xy, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float64)
    rows = np.flatnonzero(eligible & np.isfinite(xy).all(axis=1) & np.isfinite(confidence))
    if not len(rows):
        return np.empty(0, dtype=np.int64)
    lo = xy[rows].min(axis=0)
    hi = xy[rows].max(axis=0)
    span = np.maximum(hi - lo + 1., 1.)
    gx, gy = grid
    cell_x = np.clip(((xy[rows, 0] - lo[0]) / span[0] * gx).astype(int), 0, gx - 1)
    cell_y = np.clip(((xy[rows, 1] - lo[1]) / span[1] * gy).astype(int), 0, gy - 1)
    selected = []
    for cell in sorted(set(zip(cell_y.tolist(), cell_x.tolist()))):
        members = rows[(cell_y == cell[0]) & (cell_x == cell[1])]
        # Stable highest-confidence representative; ties preserve row order.
        selected.append(int(members[np.argmax(confidence[members])]))
    return np.asarray(selected, dtype=np.int64)


def synthetic_identity_metrics(source_points_m, source_pixels, source_indices,
                               predicted_endpoints, confidence, query_pose_m,
                               query_depth_mm, query_mask, intrinsics, diagonal_m):
    """Label 3D identity separately from endpoint reprojection fit."""
    points = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    source_xy = np.asarray(source_pixels, dtype=np.float64).reshape(-1, 2)
    endpoints = np.asarray(predicted_endpoints, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    ids = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    depth = np.asarray(query_depth_mm, dtype=np.float64)
    mask = np.asarray(query_mask, dtype=bool)
    k = np.asarray(intrinsics, dtype=np.float64)
    if not (len(points) == len(source_xy) == len(endpoints) == len(conf) == len(ids)):
        raise ValueError('Synthetic identity correspondence arrays have different lengths')
    if depth.shape != mask.shape or depth.ndim != 2:
        raise ValueError('Synthetic query depth and mask shapes differ')
    h, w = depth.shape
    true_xy, true_z = project_object_points(points, query_pose_m, k)
    true_ix, true_iy, true_inside = _floor_indices(true_xy, w, h)
    true_visible = np.zeros(len(points), dtype=bool)
    true_depth_mm = np.zeros(len(points), dtype=np.float64)
    true_valid = true_inside & (true_z > 0)
    rows = np.flatnonzero(true_valid)
    if len(rows):
        true_masked = mask[true_iy[rows], true_ix[rows]]
        true_depth_mm[rows] = depth[true_iy[rows], true_ix[rows]]
        tolerance_m = np.maximum(.001, .005 * true_z[rows])
        true_visible[rows] = (true_masked & np.isfinite(true_depth_mm[rows]) &
                              (true_depth_mm[rows] > 0) &
                              (np.abs(true_depth_mm[rows] * .001 - true_z[rows]) <= tolerance_m))

    target_ix, target_iy, target_inside = _floor_indices(endpoints, w, h)
    target_valid = target_inside.copy()
    rows = np.flatnonzero(target_inside)
    if len(rows):
        target_valid[rows] &= mask[target_iy[rows], target_ix[rows]]
    target_depth_mm = np.zeros(len(points), dtype=np.float64)
    rows = np.flatnonzero(target_valid)
    target_depth_mm[rows] = depth[target_iy[rows], target_ix[rows]]
    target_valid &= np.isfinite(target_depth_mm) & (target_depth_mm > 0)

    query_pose = np.asarray(query_pose_m, dtype=np.float64)
    rays = np.column_stack((endpoints, np.ones(len(endpoints)))) @ np.linalg.inv(k).T
    target_camera_m = rays * (target_depth_mm * .001)[:, None]
    target_object_m = (target_camera_m - query_pose[:3, 3]) @ query_pose[:3, :3]
    distances = np.linalg.norm(target_object_m - points, axis=1) / max(float(diagonal_m), 1e-12)
    endpoint_error = np.linalg.norm(endpoints - true_xy, axis=1)
    correct = (true_visible & target_valid & np.isfinite(endpoint_error) &
               (endpoint_error <= 3.) & np.isfinite(distances) & (distances <= .02))
    confident = np.isfinite(conf) & (conf > VISIBILITY_WEIGHT)
    source_flat = np.asarray(query_mask, dtype=bool).reshape(-1)
    source_visible_in_query = np.zeros(len(points), dtype=bool)
    source_valid_ids = (ids >= 0) & (ids < len(source_flat))
    source_visible_in_query[source_valid_ids] = source_flat[ids[source_valid_ids]]
    masked_confident = confident & source_visible_in_query & target_valid
    invisible_confident = masked_confident & ~true_visible
    confident_visible = confident & true_visible
    spatial_rows = _spatial_anchor_rows(source_xy, confident, conf)
    spatial_visible = true_visible[spatial_rows]
    spatial_correct = correct[spatial_rows]
    invisible_distances = distances[invisible_confident & np.isfinite(distances)]
    summary = dict(
        eligible_sources=int(len(points)), confidence_gt_03=int(confident.sum()),
        truly_visible_sources=int(true_visible.sum()),
        confidence_gt_03_visible=int(confident_visible.sum()),
        confidence_availability_on_visible=(float(confident_visible.sum() / max(1, true_visible.sum()))),
        correct_confident_visible=int(correct[confident_visible].sum()),
        correct_fraction_confident_visible=(float(correct[confident_visible].mean()) if confident_visible.any() else None),
        correct_all_eligible=int(correct.sum()),
        predicted_endpoint_foreground=int(target_valid.sum()),
        confident_masked_correspondences=int(masked_confident.sum()),
        confident_invisible_source_endpoints=int(invisible_confident.sum()),
        confident_invisible_fraction=(float(invisible_confident.sum() / max(1, masked_confident.sum()))),
        confident_invisible_median_identity_distance_fraction=(
            float(np.median(invisible_distances)) if len(invisible_distances) else None),
        incorrect_confident_visible=int((confident_visible & ~correct).sum()),
        correct_all_eligible_rate=(float(correct.mean()) if len(correct) else None),
        correct_confident_visible_availability=(float(correct[confident_visible].sum() /
                                                       max(1, true_visible.sum()))),
        spatial_anchors=int(len(spatial_rows)),
        spatial_anchor_visible=int(spatial_visible.sum()),
        spatial_anchor_correct=int(spatial_correct.sum()),
        spatial_anchor_correct_rate=(float(spatial_correct[spatial_visible].mean())
                                     if spatial_visible.any() else None),
        thresholds=dict(confidence_gt=.3, endpoint_error_crop_px=3., identity_distance_fraction=.02,
                        true_visibility_depth_tolerance='max(1 mm, 0.005*z)'),
    )
    arrays = dict(
        true_endpoints_px=true_xy, true_depth_m=true_z, true_visible=true_visible,
        predicted_target_valid=target_valid, target_object_m=target_object_m,
        identity_distance_fraction=distances, endpoint_error_px=endpoint_error,
        correct_identity=correct, confident=confident, masked_confident=masked_confident,
        confident_invisible=invisible_confident, spatial_rows=spatial_rows,
    )
    return summary, arrays


def select_spatially_distributed(points_xy, eligible, max_points=1024, grid=(32, 32)):
    xy = np.asarray(points_xy, dtype=np.float64)
    keep = np.asarray(eligible, dtype=bool) & np.isfinite(xy).all(axis=1)
    rows = np.flatnonzero(keep)
    if not len(rows):
        return np.empty(0, dtype=np.int64)
    gx, gy = grid
    lo, hi = xy[rows].min(axis=0), xy[rows].max(axis=0)
    span = np.maximum(hi - lo + 1., 1.)
    cell_x = np.clip(((xy[rows, 0] - lo[0]) / span[0] * gx).astype(int), 0, gx - 1)
    cell_y = np.clip(((xy[rows, 1] - lo[1]) / span[1] * gy).astype(int), 0, gy - 1)
    selected = []
    for cell in sorted(set(zip(cell_y.tolist(), cell_x.tolist()))):
        members = rows[(cell_y == cell[0]) & (cell_x == cell[1])]
        center = lo + (np.array([cell[1] + .5, cell[0] + .5]) / np.array([gx, gy])) * span
        selected.append(int(members[np.argmin(np.sum((xy[members] - center) ** 2, axis=1))]))
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) > max_points:
        selected = selected[np.linspace(0, len(selected) - 1, max_points, dtype=int)]
    return selected


def oracle_pnp(source_points_m, true_endpoints, visible, source_pixels,
               near_template_pose_m, query_pose_m, intrinsics, diagonal_m,
               rng_seed, cv2_module=None):
    """Run the unchanged RANSAC+LM solver on exact synthetic visible pairs."""
    cv = cv2 if cv2_module is None else cv2_module
    points = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    endpoints = np.asarray(true_endpoints, dtype=np.float64).reshape(-1, 2)
    visible = np.asarray(visible, dtype=bool).reshape(-1)
    xy = np.asarray(source_pixels, dtype=np.float64).reshape(-1, 2)
    chosen = select_spatially_distributed(xy, visible, 1024)
    base = dict(
        state='unavailable' if len(chosen) < 128 else 'pending',
        available_points=int(len(chosen)), selected_source_rows=chosen.tolist(),
        rng_seed=int(rng_seed), solver=dict(iterations=3000, reprojection_error_crop_px=2.,
                                             confidence=.999, flags='SOLVEPNP_ITERATIVE', lm='solvePnPRefineLM',
                                             use_extrinsic_guess=True),
    )
    if len(chosen) < 128:
        base['reason'] = 'fewer_than_128_spatially_distributed_visible_points'
        return base
    initial = np.asarray(near_template_pose_m, dtype=np.float64).copy()
    initial[:3, 3] *= 1000.
    rvec = cv.Rodrigues(initial[:3, :3])[0]
    tvec = initial[:3, 3].reshape(3, 1)
    cv.setRNGSeed(int(rng_seed) & 0x7fffffff)
    try:
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            points[chosen] * 1000., endpoints[chosen], np.asarray(intrinsics, dtype=np.float64), None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True, iterationsCount=3000,
            reprojectionError=2., confidence=.999, flags=cv.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < 6:
            base.update(state='failed', reason='oracle_ransac_unavailable', solver_success=bool(success))
            return base
        rvec, tvec = cv.solvePnPRefineLM(
            points[chosen][inliers.reshape(-1)] * 1000., endpoints[chosen][inliers.reshape(-1)],
            np.asarray(intrinsics, dtype=np.float64), None, rvec, tvec)
    except cv.error as error:
        base.update(state='failed', reason='oracle_solver_exception', exception=f'{type(error).__name__}: {error}')
        return base
    recovered = np.eye(4, dtype=np.float64)
    recovered[:3, :3] = cv.Rodrigues(rvec)[0]
    recovered[:3, 3] = np.asarray(tvec).reshape(3) * .001
    truth = np.asarray(query_pose_m, dtype=np.float64)
    rotation = recovered[:3, :3] @ truth[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.) / 2., -1., 1.)))
    translation = float(np.linalg.norm(recovered[:3, 3] - truth[:3, 3]))
    projected, _ = project_object_points(points[chosen], recovered, intrinsics)
    error = np.linalg.norm(projected - endpoints[chosen], axis=1)
    median = float(np.median(error[np.isfinite(error)])) if np.isfinite(error).any() else None
    limits = dict(rotation_degrees=.1, translation_m=.001 * float(diagonal_m), median_reprojection_crop_px=.01)
    passed = bool(angle <= limits['rotation_degrees'] and translation <= limits['translation_m'] and
                  median is not None and median <= limits['median_reprojection_crop_px'])
    base.update(
        state='passed' if passed else 'failed', reason=None if passed else 'oracle_pose_or_projection_gate_failed',
        solver_success=True, inlier_count=int(len(inliers)), recovered_pose_m=recovered.tolist(),
        rotation_error_degrees=float(angle), translation_error_m=translation,
        median_reprojection_crop_px=median, limits=limits,
    )
    return base


def _gray_image(rgb):
    return cv2.cvtColor(_float_rgb(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _highpass(gray, sigma=2.):
    value = np.asarray(gray, dtype=np.float32)
    return value - cv2.GaussianBlur(value, (0, 0), float(sigma))


def _eroded_mask(mask, margin=6):
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError('Expected 2D foreground mask')
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    return distance >= float(margin)


def _normalize_patch(patch):
    value = np.asarray(patch, dtype=np.float32)
    centered = value - float(value.mean())
    norm = float(np.linalg.norm(centered))
    std = float(value.std())
    if not np.isfinite(std) or std < PATCH_STD_MIN or norm <= 1e-12:
        return None, std
    return centered / norm, std


def build_template_patch_bank(template_rgb, template_mask, source_indices,
                              source_points_m, grid_size=16, max_anchors=64):
    """Select deterministic 11x11 high-pass texture anchors before inference."""
    rgb = _float_rgb(template_rgb)
    mask = np.asarray(template_mask, dtype=bool)
    ids = np.asarray(source_indices, dtype=np.int64)
    points = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    if rgb.shape[:2] != mask.shape or len(ids) != len(points):
        raise ValueError('Template image, mask and object-point arrays differ')
    h, w = mask.shape
    eroded = _eroded_mask(mask, 6)
    high = _highpass(_gray_image(rgb), 2.)
    xs = np.clip(np.rint((np.arange(grid_size) + .5) * w / grid_size).astype(int),
                 PATCH_RADIUS, max(PATCH_RADIUS, w - PATCH_RADIUS - 1))
    ys = np.clip(np.rint((np.arange(grid_size) + .5) * h / grid_size).astype(int),
                 PATCH_RADIUS, max(PATCH_RADIUS, h - PATCH_RADIUS - 1))
    id_to_row = {int(value): i for i, value in enumerate(ids.tolist())}
    candidates = []
    for grid_y, y in enumerate(ys):
        for grid_x, x in enumerate(xs):
            flat = int(y * w + x)
            if flat not in id_to_row:
                continue
            patch_mask = eroded[y-PATCH_RADIUS:y+PATCH_RADIUS+1, x-PATCH_RADIUS:x+PATCH_RADIUS+1]
            if patch_mask.shape != (PATCH_SIZE, PATCH_SIZE) or not patch_mask.all():
                continue
            patch = high[y-PATCH_RADIUS:y+PATCH_RADIUS+1, x-PATCH_RADIUS:x+PATCH_RADIUS+1]
            descriptor, std = _normalize_patch(patch)
            if descriptor is None:
                continue
            row = id_to_row[flat]
            candidates.append(dict(
                grid_index=int(grid_y * grid_size + grid_x), source_index=flat,
                source_xy_crop=[float(x + .5), float(y + .5)],
                object_xyz_m=points[row].tolist(), highpass_std=std,
                descriptor=descriptor, patch_rgb=rgb[y-PATCH_RADIUS:y+PATCH_RADIUS+1,
                                                      x-PATCH_RADIUS:x+PATCH_RADIUS+1].copy(),
            ))
    eligible_count = len(candidates)
    if len(candidates) > max_anchors:
        selected = np.linspace(0, len(candidates) - 1, max_anchors, dtype=int)
        candidates = [candidates[i] for i in selected]
    public = [{key: value for key, value in row.items() if key not in ('descriptor', 'patch_rgb')}
              for row in candidates]
    return dict(
        state='eligible' if candidates else 'no_distinctive_source_patches',
        candidate_grid_count=grid_size * grid_size,
        eligible_source_patch_count=eligible_count,
        selected_anchor_count=len(candidates), anchors=candidates,
        serialized_anchors=public,
    )


_PATCH_BANK_INPUTS = (
    'template_rgb', 'template_mask', 'source_indices', 'source_pixels_xy',
    'source_points_object_m',
)


def fixed_patch_bank_manifest(bank, template_arrays):
    """Serialize the already-selected fixed bank before any model forward."""
    return dict(
        schema_version=1,
        selector=dict(grid_size=16, max_anchors=64, patch_size=PATCH_SIZE,
                      patch_std_min=PATCH_STD_MIN),
        template_input_sha256={name: array_sha256(template_arrays[name])
                               for name in _PATCH_BANK_INPUTS},
        candidate_grid_count=int(bank['candidate_grid_count']),
        eligible_source_patch_count=int(bank['eligible_source_patch_count']),
        selected_anchor_count=int(bank['selected_anchor_count']),
        anchors=bank['serialized_anchors'],
    )


def verify_fixed_patch_bank(context_entry, template_arrays):
    """Verify and use the pre-forward anchor selection bound to its inputs."""
    frozen = context_entry.get('fixed_patch_bank')
    if not isinstance(frozen, dict):
        raise ValueError(f"Template context lacks its pre-forward fixed patch bank: {context_entry.get('context_id')}")
    computed = build_template_patch_bank(
        template_arrays['template_rgb'], template_arrays['template_mask'],
        template_arrays['source_indices'], template_arrays['source_points_object_m'])
    if frozen != fixed_patch_bank_manifest(computed, template_arrays):
        raise ValueError(f"Frozen template patch bank or input binding changed: {context_entry.get('context_id')}")
    # Descriptors and RGB patches are reconstructed from the hash-verified inputs;
    # ordering and source identity are taken from the frozen pre-forward manifest.
    by_source = {int(anchor['source_index']): anchor for anchor in computed['anchors']}
    ordered = []
    for row in frozen['anchors']:
        anchor = by_source.get(int(row['source_index']))
        if anchor is None:
            raise ValueError(f"Frozen patch-bank source is absent: {context_entry.get('context_id')}")
        ordered.append(anchor)
    computed['anchors'] = ordered
    computed['serialized_anchors'] = frozen['anchors']
    return computed


def _bilinear_sample(image, x, y):
    value = np.asarray(image)
    h, w = value.shape[:2]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    xc0, xc1 = np.clip(x0, 0, w-1), np.clip(x1, 0, w-1)
    yc0, yc1 = np.clip(y0, 0, h-1), np.clip(y1, 0, h-1)
    wx, wy = x - x0, y - y0
    extra = (None,) * (value.ndim - 2)
    wx, wy = wx[(...,) + extra], wy[(...,) + extra]
    sampled = (value[yc0, xc0] * (1-wx) * (1-wy) + value[yc0, xc1] * wx * (1-wy) +
               value[yc1, xc0] * (1-wx) * wy + value[yc1, xc1] * wx * wy)
    return sampled, valid


def _bilinear_patch(image, endpoint_xy, eroded_mask):
    endpoint = np.asarray(endpoint_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(endpoint).all():
        return None, None
    offsets = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=np.float64)
    # Network coordinates use upstream pixel centres; OpenCV array indices
    # place those centres at integer coordinates after subtracting 0.5.
    xx, yy = np.meshgrid(endpoint[0] - .5 + offsets, endpoint[1] - .5 + offsets)
    samples, in_bounds = _bilinear_sample(image, xx, yy)
    mask_value = np.asarray(eroded_mask, dtype=bool)
    h, w = mask_value.shape
    x0, y0 = np.floor(xx).astype(int), np.floor(yy).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    inside = in_bounds & mask_value[np.clip(y0, 0, h-1), np.clip(x0, 0, w-1)]
    inside &= mask_value[np.clip(y0, 0, h-1), np.clip(x1, 0, w-1)]
    inside &= mask_value[np.clip(y1, 0, h-1), np.clip(x0, 0, w-1)]
    inside &= mask_value[np.clip(y1, 0, h-1), np.clip(x1, 0, w-1)]
    if not inside.all():
        return None, None
    return np.asarray(samples, dtype=np.float32), None


def _corr(a, b):
    left, _ = _normalize_patch(a)
    right, _ = _normalize_patch(b)
    if left is None or right is None:
        return None
    return float(np.sum(left * right))


def audit_patch_identity(query_rgb, observed_mask, flow, confidence,
                         source_bank, all_banks, diagonal_m):
    """Compare bilinear current-image patches with distant 3D identity anchors."""
    query = _float_rgb(query_rgb)
    mask = np.asarray(observed_mask, dtype=bool)
    flow = np.asarray(flow, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    if query.shape[:2] != mask.shape or flow.shape != (*mask.shape, 2) or conf.shape != mask.shape:
        raise ValueError('Patch audit query, mask, flow or confidence shape mismatch')
    h, w = mask.shape
    query_high = _highpass(_gray_image(query), 2.)
    eroded = _eroded_mask(mask, 6)
    source_anchors = source_bank['anchors']
    competitor_anchors = [a for bank in all_banks for a in bank['anchors']]
    rows, patches = [], []
    bank_has_distant_competitor = any(
        np.linalg.norm(np.asarray(other['object_xyz_m']) - np.asarray(source['object_xyz_m'])) >
        .1 * diagonal_m
        for source in source_anchors for other in competitor_anchors)
    for source in source_anchors:
        source_index = int(source['source_index'])
        sy, sx = divmod(source_index, w)
        endpoint = np.array(source['source_xy_crop'], dtype=np.float64) + flow[sy, sx]
        source_confidence = float(conf[sy, sx])
        finite_endpoint = bool(np.isfinite(endpoint).all())
        record = dict(
            source_index=source_index, source_xy_crop=source['source_xy_crop'],
            source_object_xyz_m=source['object_xyz_m'],
            target_xy_crop=(endpoint.tolist() if finite_endpoint else [None, None]),
            confidence=(source_confidence if np.isfinite(source_confidence) else None),
            state='rejected', reason=None, ncc=None, best_distant_ncc=None,
            distant_margin=None, competitor_source_xy_crop=None,
            competitor_object_xyz_m=None,
        )
        if not finite_endpoint:
            record['reason'] = 'target_endpoint_nonfinite'
            rows.append(record)
            continue
        target_patch, _ = _bilinear_patch(query_high, endpoint, eroded)
        target_rgb_patch, _ = _bilinear_patch(query, endpoint, eroded)
        if target_patch is None:
            record['reason'] = 'target_patch_outside_eroded_observed_mask_or_image'
            rows.append(record)
            continue
        target_descriptor, target_std = _normalize_patch(target_patch)
        record['target_highpass_std'] = target_std
        if target_descriptor is None:
            record['reason'] = 'target_patch_low_texture'
            rows.append(record)
            continue
        record['target_eligible'] = True
        source_xyz = np.asarray(source['object_xyz_m'], dtype=np.float64)
        competitors = [a for a in competitor_anchors
                       if np.linalg.norm(np.asarray(a['object_xyz_m']) - source_xyz) > .1 * diagonal_m]
        if not competitors:
            record['state'] = 'identity_bank_uninformative'
            record['reason'] = 'no_distant_texture_competitor'
            rows.append(record)
            continue
        ncc = _corr(source['descriptor'], target_patch)
        scores = [(_corr(other['descriptor'], target_patch), other) for other in competitors]
        scores = [(score, other) for score, other in scores if score is not None]
        if not scores:
            record['state'] = 'identity_bank_uninformative'
            record['reason'] = 'distant_texture_competitors_uninformative'
            rows.append(record)
            continue
        best_score, best_other = max(scores, key=lambda item: item[0])
        margin = None if ncc is None else float(ncc - best_score)
        record.update(
            ncc=ncc, best_distant_ncc=float(best_score), distant_margin=margin,
            competitor_source_xy_crop=best_other['source_xy_crop'],
            competitor_object_xyz_m=best_other['object_xyz_m'],
            state='supported' if ncc is not None and ncc >= NCC_MIN and margin is not None and margin >= NCC_MARGIN_MIN else 'rejected',
            reason=None if ncc is not None and ncc >= NCC_MIN and margin is not None and margin >= NCC_MARGIN_MIN else 'ncc_or_distant_identity_margin_below_gate',
        )
        rows.append(record)
        patches.append(dict(
            source_index=source_index, source=source['patch_rgb'], target=target_rgb_patch,
            competitor=best_other['patch_rgb'], state=record['state'],
        ))
    eligible = len(source_anchors)
    target_eligible = sum(bool(row.get('target_eligible')) for row in rows)
    supported_rows = [row for row in rows if row['state'] == 'supported']
    ys, xs = np.nonzero(mask)
    if len(xs) and supported_rows:
        bbox = dict(x0=int(xs.min()), x1=int(xs.max()), y0=int(ys.min()), y1=int(ys.max()))
        span_x, span_y = max(1, bbox['x1'] - bbox['x0'] + 1), max(1, bbox['y1'] - bbox['y0'] + 1)
        cells = set()
        support_xy = []
        for row in supported_rows:
            x, y = row['target_xy_crop']
            cx = int(np.clip((x - bbox['x0']) / span_x * 4, 0, 3))
            cy = int(np.clip((y - bbox['y0']) / span_y * 4, 0, 3))
            cells.add((cx, cy)); support_xy.append([x, y])
        hull_area = 0.
        if len(support_xy) >= 3:
            hull = cv2.convexHull(np.asarray(support_xy, dtype=np.float32))
            hull_area = float(cv2.contourArea(hull))
        hull_fraction = hull_area / float(span_x * span_y)
    else:
        bbox, cells, hull_fraction = None, set(), 0.
    gate = (len(supported_rows) >= 8 and len(cells) >= 3 and hull_fraction >= .12)
    if not source_anchors or not bank_has_distant_competitor:
        state = 'identity_bank_uninformative'
    else:
        state = 'distinctive_current_image_support' if gate else 'unobservable_or_ambiguous'
    summary = dict(
        state=state, source_eligible=int(eligible),
        source_eligible_before_cap=int(source_bank.get('eligible_source_patch_count', eligible)),
        source_anchors_selected=int(eligible), target_eligible=int(target_eligible),
        rejected=int(eligible - len(supported_rows)), supported=int(len(supported_rows)),
        supported_cells_4x4=int(len(cells)), supported_hull_fraction=float(hull_fraction),
        observed_mask_bbox=bbox, thresholds=dict(source_std_min=PATCH_STD_MIN,
            target_std_min=PATCH_STD_MIN, ncc_min=NCC_MIN, distant_margin_min=NCC_MARGIN_MIN,
            distant_identity_distance_fraction=IDENTITY_DISTANCE_FRACTION,
            min_supported_anchors=8, min_supported_cells=3, min_hull_fraction=.12),
    )
    return dict(summary=summary, rows=rows, mosaic_patches=patches)


def _write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'Refusing to replace immutable audit artifact: {path}')
    np.savez_compressed(temporary, **arrays)
    # numpy appends .npz when the supplied temporary suffix does not end in it.
    generated = temporary if temporary.exists() else Path(str(temporary) + '.npz')
    generated.replace(path)


def _load_npz(root, entry):
    root = Path(root).resolve()
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe identity packet path: {entry["path"]}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Identity packet escaped capture root: {entry["path"]}')
    if not path.is_file() or path.stat().st_size != int(entry['bytes']) or digest(path) != entry['sha256']:
        raise ValueError(f'Identity packet size/hash mismatch: {entry["path"]}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = entry.get('arrays', {})
    if set(arrays) != set(expected):
        raise ValueError(f'Identity packet array set mismatch: {entry["path"]}')
    for name, value in arrays.items():
        if array_info(value) != expected[name]:
            raise ValueError(f'Identity packet array metadata mismatch: {entry["path"]}:{name}')
    return arrays


def _rotation_distance_degrees(a, b):
    relative = np.asarray(a) @ np.asarray(b).T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))


def _appearance_context_check(rgb, gray_rgb_value, foreground_mask):
    rgb = _float_rgb(rgb)
    gray = _float_rgb(gray_rgb_value)
    mask = np.asarray(foreground_mask, dtype=bool)
    if rgb.shape != gray.shape or mask.shape != rgb.shape[:2]:
        raise ValueError('Captured appearance arrays have inconsistent shapes')
    background_equal = bool(np.array_equal(rgb[~mask], gray[~mask]))
    gray_replicated = bool(not mask.any() or np.allclose(
        gray[mask, 0], gray[mask, 1], atol=1e-6, rtol=0) and
        np.allclose(gray[mask, 0], gray[mask, 2], atol=1e-6, rtol=0))
    delta = np.max(np.abs(gray - rgb), axis=2)
    changed = mask & (delta >= GRAY_CHANGE_MIN)
    fraction = float(changed.sum() / max(1, int(mask.sum())))
    return dict(
        background_unchanged=background_equal, foreground_is_achromatic=gray_replicated,
        foreground_pixels=int(mask.sum()), changed_pixels_ge_005=int(changed.sum()),
        changed_pixel_fraction=fraction,
        state='informative' if fraction >= .05 else 'appearance_ablation_uninformative',
    )


def fit_learned_packet(source_points_m, endpoints_crop, confidence, source_indices,
                       source_query_mask, target_query_mask, template_pose_crop_m,
                       crop_k, crop_from_native, native_k, native_frame, native_mask,
                       rng_seed, cv2_module=None):
    """Fit one captured forward packet once, then apply the current contract."""
    cv = cv2 if cv2_module is None else cv2_module
    xyz = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(endpoints_crop, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    ids = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    source_mask = np.asarray(source_query_mask, dtype=bool).reshape(-1)
    target_mask = np.asarray(target_query_mask, dtype=bool)
    if not (len(xyz) == len(uv) == len(conf) == len(ids)):
        raise ValueError('Learned PnP correspondence arrays differ in length')
    h, w = target_mask.shape
    ix, iy, inside = _floor_indices(uv, w, h)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1) & np.isfinite(conf)
    valid &= conf > VISIBILITY_WEIGHT
    valid &= (ids >= 0) & (ids < len(source_mask))
    valid &= source_mask[np.clip(ids, 0, max(0, len(source_mask)-1))]
    valid &= inside
    rows = np.flatnonzero(inside)
    endpoint_keep = np.zeros(len(xyz), dtype=bool)
    endpoint_keep[rows] = target_mask[iy[rows], ix[rows]]
    valid &= endpoint_keep
    retained = np.flatnonzero(valid)
    result = dict(
        state='unavailable', reason=None, retained_correspondences=int(len(retained)),
        sampled_correspondences=min(int(len(retained)), 10000), rng_seed=int(rng_seed),
        ransac_inlier_ids=[], retained_inlier_ids=[], pose_crop_m=None, pose_native_m=None,
        validation_state='not_run', validation_reason=None, validation_stats=None,
    )
    if len(retained) < 24:
        result['reason'] = 'fewer_than_24_confident_masked_correspondences'
        result['validation_reason'] = result['reason']
        return result
    sample_local = np.linspace(0, len(retained)-1, min(len(retained), 10000), dtype=int)
    sampled_rows = retained[sample_local]
    initial = np.asarray(template_pose_crop_m, dtype=np.float64).copy()
    initial[:3, 3] *= 1000.
    rvec = cv.Rodrigues(initial[:3, :3])[0]
    tvec = initial[:3, 3].reshape(3, 1)
    cv.setRNGSeed(int(rng_seed) & 0x7fffffff)
    try:
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            xyz[sampled_rows] * 1000., uv[sampled_rows], np.asarray(crop_k, dtype=np.float64), None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True, iterationsCount=3000,
            reprojectionError=2., confidence=.999, flags=cv.SOLVEPNP_ITERATIVE,
        )
    except cv.error as error:
        result.update(state='ransac_exception', reason=f'{type(error).__name__}: {error}',
                      validation_reason='candidate_unavailable_ransac_exception')
        return result
    if not success or inliers is None or len(inliers) < 6:
        result.update(state='ransac_unavailable', reason='ransac_inliers_unavailable',
                      solver_success=bool(success), validation_reason='candidate_unavailable_ransac_unavailable')
        return result
    inlier_ids = np.asarray(inliers, dtype=np.int64).reshape(-1)
    chosen_rows = sampled_rows[inlier_ids]
    try:
        rvec, tvec = cv.solvePnPRefineLM(
            xyz[chosen_rows] * 1000., uv[chosen_rows], np.asarray(crop_k, dtype=np.float64), None, rvec, tvec)
    except cv.error as error:
        result.update(state='lm_exception', reason=f'{type(error).__name__}: {error}',
                      solver_success=True, ransac_inlier_ids=inlier_ids.tolist(),
                      retained_inlier_ids=chosen_rows.tolist(), validation_reason='candidate_unavailable_lm_exception')
        return result
    pose_crop_mm = np.eye(4, dtype=np.float64)
    pose_crop_mm[:3, :3] = cv.Rodrigues(rvec)[0]
    pose_crop_mm[:3, 3] = np.asarray(tvec).reshape(3)
    pose_crop_m = pose_crop_mm.copy(); pose_crop_m[:3, 3] *= .001
    pose_native_m = crop_to_native_pose_m(pose_crop_mm, crop_from_native)
    rays_crop = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(np.asarray(crop_k)).T
    rays_native = rays_crop @ np.asarray(crop_from_native)[:3, :3]
    projected = rays_native @ np.asarray(native_k).T
    with np.errstate(divide='ignore', invalid='ignore'):
        native_uv = projected[:, :2] / projected[:, 2:3]
    candidate = PoseCandidate(pose_native_m, xyz[retained], native_uv[retained], conf[retained])
    valid_candidate, validation_reason, stats = validate(candidate, native_frame, native_mask)
    result.update(
        state='refined', reason=None, solver_success=True,
        ransac_inlier_ids=inlier_ids.tolist(), retained_inlier_ids=chosen_rows.tolist(),
        pose_crop_m=pose_crop_m.tolist(), pose_native_m=pose_native_m.tolist(),
        candidate_arrays={name: array_info(value) for name, value in dict(
            pose_m=candidate.pose, points_object_m=candidate.points_object_m,
            pixels_native=candidate.pixels_image, weights=candidate.weights).items()},
        validation_state='accepted' if valid_candidate else 'rejected',
        validation_reason=validation_reason, validation_stats=stats,
    )
    return result


def _validate_capture_manifest(capture):
    if capture.get('schema_version') != 1 or capture.get('object') != 'ranch':
        raise ValueError('Expected schema-1 ranch bottle-identity capture')
    if capture.get('requested_real_frame_ids') != list(FRAME_IDS):
        raise ValueError('Real identity frame set differs from the frozen R1 list')
    if capture.get('requested_synthetic_carriers') != list(SYNTHETIC_CARRIERS):
        raise ValueError('Synthetic carriers differ from the frozen R1 list')
    if capture.get('planned_forward_cap') != FORWARD_CAP or len(capture.get('conditions', [])) != FORWARD_CAP:
        raise ValueError('Capture does not enumerate the frozen 108-forward plan')
    if capture.get('packet_budget_bytes') != PACKET_BUDGET_BYTES:
        raise ValueError('Capture packet budget differs from frozen 128 MiB cap')
    if capture.get('reference_or_annotations_loaded') is not False:
        raise ValueError('Identity capture must declare that no reference or annotation inputs were loaded')
    if not isinstance(capture.get('contexts'), list) or not isinstance(capture.get('forwards'), list):
        raise ValueError('Capture packet index is missing')
    condition_ids = [row.get('condition_id') for row in capture.get('conditions', [])]
    if len(set(condition_ids)) != FORWARD_CAP or None in condition_ids:
        raise ValueError('Capture must enumerate 108 unique frozen conditions')
    conditions = capture['conditions']
    synthetic = [row for row in conditions if row.get('kind') == 'synthetic']
    real = [row for row in conditions if row.get('kind') == 'real']
    repeats = [row for row in conditions if row.get('kind') == 'repeat']
    expected_synthetic = {
        (frame_id, qoffset, qappearance, toffset, tappearance)
        for frame_id in SYNTHETIC_CARRIERS for qoffset in QUERY_OFFSETS
        for qappearance in APPEARANCES for toffset in TEMPLATE_OFFSETS
        for tappearance in APPEARANCES
    }
    actual_synthetic = {
        (row.get('frame_id'), row.get('query_offset_deg'), row.get('query_appearance'),
         row.get('template_offset_deg'), row.get('template_appearance'))
        for row in synthetic
    }
    expected_real = {(frame_id, toffset, tappearance) for frame_id in FRAME_IDS
                     for toffset in TEMPLATE_OFFSETS for tappearance in APPEARANCES}
    actual_real = {(row.get('frame_id'), row.get('template_offset_deg'), row.get('template_appearance'))
                   for row in real}
    if (len(synthetic) != 64 or actual_synthetic != expected_synthetic or
            len(real) != 40 or actual_real != expected_real or len(repeats) != 4 or
            any(row.get('frame_id') != 10 for row in repeats)):
        raise ValueError('Capture condition matrix differs from the frozen 64+40+4 plan')
    by_id = {row['condition_id']: row for row in conditions}
    for index, row in enumerate(conditions):
        if row.get('forward_index') != index or row.get('state') not in ('pending', 'captured', 'unavailable', 'failed'):
            raise ValueError('Capture condition order/state differs from the frozen plan')
        expected_seed = (int(row['frame_id']) * 17 +
                         (int(row.get('query_offset_deg', 0)) // 180) * 4 +
                         (int(row['template_offset_deg']) // 180) * 2 +
                         APPEARANCES.index(row['template_appearance'])) & 0x7fffffff
        if row.get('kind') == 'repeat':
            original = by_id.get(row.get('repeat_of'))
            if (original is None or original.get('kind') != 'real' or
                    original.get('frame_id') != 10 or row.get('rng_seed') != original.get('rng_seed')):
                raise ValueError('Repeat condition is not an exact frame-10 replay')
        elif row.get('rng_seed') != expected_seed:
            raise ValueError('Condition RNG seed differs from frozen frame-local formula')
    expected_repeats = {row['condition_id'] for row in real if row.get('frame_id') == 10}
    if {row.get('repeat_of') for row in repeats} != expected_repeats:
        raise ValueError('Four frame-10 repeat conditions do not cover their exact originals')
    if capture.get('complete') is True:
        if any(row.get('state') not in ('captured', 'unavailable') for row in conditions):
            raise ValueError('Complete capture contains pending or failed conditions')
    if int(capture.get('packet_bytes', -1)) < 0 or int(capture['packet_bytes']) > PACKET_BUDGET_BYTES:
        raise ValueError('Capture packet byte counter is outside the frozen budget')
    context_ids = [row.get('context_id') for row in capture['contexts']]
    forward_ids = [row.get('condition_id') for row in capture['forwards']]
    if len(set(context_ids)) != len(context_ids) or None in context_ids:
        raise ValueError('Capture context packet ids are not unique')
    if len(set(forward_ids)) != len(forward_ids) or not set(forward_ids).issubset(set(condition_ids)):
        raise ValueError('Capture forward packet ids are duplicated or outside the frozen plan')
    captured_ids = {row['condition_id'] for row in conditions if row.get('state') == 'captured'}
    if set(forward_ids) != captured_ids:
        raise ValueError('Capture forward index does not exactly match captured condition states')
    diagonal = capture.get('object_bbox_diagonal_m')
    if diagonal is not None and (type(diagonal) not in (int, float) or
                                 not np.isfinite(diagonal) or diagonal <= 0):
        raise ValueError('Capture asset bbox diagonal is invalid')


def _load_capture_manifest(capture_root):
    root = Path(capture_root).expanduser().resolve()
    path = root / 'capture.json'
    capture = json.loads(path.read_text(encoding='utf-8'))
    _validate_capture_manifest(capture)
    return root, capture


_SYNTHETIC_ONLY_FIELDS = {
    'oracle', 'known_query_pose', 'known_query_pose_m', 'query_pose_native_m',
    'query_depth_mm', 'synthetic_query_mask', 'synthetic_native_mask',
    'synthetic_geometry_oracle', 'annotation', 'reference_pose',
}


def _real_branch_is_leak_free(condition_entries, context_entries):
    """Reject synthetic labels in real/repeat inputs while allowing seed/template poses."""
    for condition_id, condition in condition_entries.items():
        if condition.get('kind') not in ('real', 'repeat'):
            continue
        if _SYNTHETIC_ONLY_FIELDS.intersection(condition):
            return False
        refs = condition.get('context_refs', {})
        for context_id in refs.values():
            entry = context_entries.get(context_id)
            if entry is None:
                raise ValueError(f'Real condition references missing context: {condition_id}')
            role = entry.get('role')
            if role not in ('real_frame', 'template'):
                return False
            if _SYNTHETIC_ONLY_FIELDS.intersection(entry):
                return False
            if _SYNTHETIC_ONLY_FIELDS.intersection(entry.get('arrays', {})):
                return False
    return True


def _source_rows(template):
    return (np.asarray(template['source_indices'], dtype=np.int64),
            np.asarray(template['source_pixels_xy'], dtype=np.float64),
            np.asarray(template['source_points_object_m'], dtype=np.float64))


def _forward_mapping(template, forward):
    ids, pixels, points = _source_rows(template)
    h, w = forward['flow'].shape[:2]
    if forward['flow'].shape != (h, w, 2) or forward['confidence'].shape != (h, w):
        raise ValueError('Captured flow/confidence shape mismatch')
    yy, xx = np.divmod(ids, w)
    endpoints = pixels + forward['flow'][yy, xx]
    tx, ty, in_bounds = _floor_indices(endpoints, w, h)
    target_indices = np.full((len(ids), 2), -1, dtype=np.int32)
    target_indices[in_bounds] = np.column_stack((tx[in_bounds], ty[in_bounds])).astype(np.int32)
    return ids, pixels, points, endpoints, forward['confidence'][yy, xx], target_indices


def _array_hashes_for_forward(template, forward):
    ids, pixels, points, endpoints, confidence, target_indices = _forward_mapping(template, forward)
    return {
        'source_indices': array_info(ids), 'source_pixels_xy': array_info(pixels),
        'source_points_object_m': array_info(points), 'target_endpoints_px': array_info(endpoints),
        'target_indices_xy': array_info(target_indices), 'source_confidence': array_info(confidence),
    }


def _write_mosaic(path, entries):
    if not entries:
        return None
    columns = 16
    cell_w, cell_h = PATCH_SIZE * 6, PATCH_SIZE * 2 + 18
    rows = (len(entries) + columns - 1) // columns
    canvas = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    for index, entry in enumerate(entries):
        y, x = divmod(index, columns)
        source = np.rint(np.clip(entry['source'], 0, 1) * 255).astype(np.uint8)
        target = np.rint(np.clip(entry['target'], 0, 1) * 255).astype(np.uint8)
        competitor = np.rint(np.clip(entry['competitor'], 0, 1) * 255).astype(np.uint8)
        # Three grayscale panels expose the exact source/current/competitor patches.
        source_g = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        target_g = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        competitor_g = cv2.cvtColor(competitor, cv2.COLOR_RGB2GRAY)
        panel = np.concatenate((source_g, target_g, competitor_g), axis=1)
        panel = np.repeat(np.repeat(panel, 2, axis=0), 2, axis=1)
        panel = np.repeat(panel[:, :, None], 3, axis=2)
        row_slice = slice(y * cell_h, y * cell_h + PATCH_SIZE * 2)
        col_slice = slice(x * cell_w, x * cell_w + PATCH_SIZE * 6)
        canvas[row_slice, col_slice] = panel
        color = (0, 255, 0) if entry['state'] == 'supported' else (255, 120, 0)
        cv2.putText(canvas, str(entry['source_index']), (x * cell_w, y * cell_h + cell_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target_path), canvas):
        raise IOError(f'Failed to write patch mosaic: {target_path}')
    return str(target_path)


def audit_capture(capture_root, output_root):
    """Verify packet invariants, then run CPU identity, NCC and PnP audits."""
    root, capture = _load_capture_manifest(capture_root)
    output_root = Path(output_root).expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f'Output root already exists; preserving it: {output_root}')
    output_root.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema_version=1, scope=__doc__, status='running', capture_root=str(root),
        capture_sha256=digest(root / 'capture.json'), capture_packet_bytes=capture.get('packet_bytes'),
        packet_budget_bytes=PACKET_BUDGET_BYTES, input_conditions=len(capture['conditions']),
        capture_status=capture.get('status'), capture_declared_complete=capture.get('complete'),
        cpu_runtime=dict(numpy=np.__version__, opencv=cv2.__version__),
        reference_or_annotations_loaded=False, synthetic_geometry_is_not_real_pose_truth=True,
        invariants=dict(repeat_hashes_match=None, shared_camera_depth_masks_match=None,
                        projection_parity_passed=None, real_records_leak_free=None,
                        context_arrays_consistent=None, appearance_ablation_consistent=None),
        synthetic=[], real=[], forwards=[], decisions={},
        output_root=str(output_root), started_unix=time.time(),
    )
    save_result(output_root / 'audit.json', report)
    try:
        context_entries = {entry['context_id']: entry for entry in capture.get('contexts', [])}
        condition_entries = {entry['condition_id']: entry for entry in capture['conditions']}
        forward_entries = {entry['condition_id']: entry for entry in capture.get('forwards', [])}
        contexts = {key: _load_npz(root, entry) for key, entry in context_entries.items()}
        forwards = {key: _load_npz(root, entry) for key, entry in forward_entries.items()}
        real_leak_free = _real_branch_is_leak_free(condition_entries, context_entries)
        if not real_leak_free:
            raise ValueError('Real-input packet manifest contains synthetic oracle/annotation fields')

        context_roles = {key: entry.get('role') for key, entry in context_entries.items()}
        for condition_id, condition in condition_entries.items():
            if condition.get('state') != 'captured':
                continue
            refs = condition.get('context_refs', {})
            expected_query_role = 'synthetic_query' if condition.get('kind') == 'synthetic' else 'real_frame'
            if (context_roles.get(refs.get('query')) != expected_query_role or
                    context_roles.get(refs.get('template')) != 'template' or
                    context_roles.get(refs.get('observed_frame')) != 'real_frame'):
                raise ValueError(f'Condition context roles differ from frozen capture plan: {condition_id}')

        context_checks = []
        appearance_checks = []
        real_masks_by_frame = {
            int(context_entries[context_id]['frame_id']): np.asarray(arrays['native_mask'], dtype=bool)
            for context_id, arrays in contexts.items()
            if context_entries[context_id].get('role') == 'real_frame'
        }
        for context_id, arrays in contexts.items():
            entry = context_entries[context_id]
            role = entry.get('role')
            if role == 'real_frame':
                rgb = np.asarray(arrays['native_rgb'])
                mask = np.asarray(arrays['native_mask'], dtype=np.uint8)
                if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or
                        mask.shape != rgb.shape[:2]):
                    raise ValueError(f'Native input context dimensions/dtype are invalid: {context_id}')
                if (entry.get('native_rgb_sha256') != array_sha256(rgb) or
                        entry.get('native_mask_sha256') != array_sha256(mask > 0)):
                    raise ValueError(f'Captured native RGB/mask hashes differ from context metadata: {context_id}')
                context_checks.append(dict(context_id=context_id, native_rgb=True, observed_mask=True))
            elif role == 'template':
                depth = np.asarray(arrays['template_depth_mm'])
                mask = np.asarray(arrays['template_mask'], dtype=bool)
                if (depth.ndim != 2 or mask.shape != depth.shape or
                        np.asarray(arrays['template_rgb']).shape != (*depth.shape, 3) or
                        not np.array_equal(mask, np.isfinite(depth) & (depth > 0))):
                    raise ValueError(f'Template depth/mask geometry is inconsistent: {context_id}')
                hashes = entry.get('geometry_hashes', {})
                if (hashes.get('depth_mm') != array_sha256(depth) or
                        hashes.get('mask') != array_sha256(mask)):
                    raise ValueError(f'Template geometry hashes differ from context metadata: {context_id}')
                appearance = _appearance_context_check(
                    arrays['template_rgb'], arrays['template_gray_rgb'], mask)
                if not appearance['background_unchanged'] or not appearance['foreground_is_achromatic']:
                    raise ValueError(f'Template appearance ablation changed background or was not gray: {context_id}')
                appearance_checks.append(dict(context_id=context_id, role=role, **appearance))
                context_checks.append(dict(context_id=context_id, depth_mask=True,
                                           template_pose=True, crop_camera=True))
            elif role == 'synthetic_query':
                observed = np.asarray(arrays['observed_crop_mask'], dtype=bool)
                rendered = np.asarray(arrays['query_render_mask'], dtype=bool)
                visible = np.asarray(arrays['synthetic_query_mask'], dtype=bool)
                depth = np.asarray(arrays['query_depth_mm'])
                if (observed.shape != visible.shape or rendered.shape != visible.shape or
                        depth.shape != visible.shape or np.any(visible & ~observed) or
                        np.any(visible & ~rendered) or np.any(visible & ~(np.isfinite(depth) & (depth > 0)))):
                    raise ValueError(f'Synthetic query visibility/depth intersection is inconsistent: {context_id}')
                native_mask = np.asarray(arrays['synthetic_native_mask'], dtype=bool)
                observed_native = real_masks_by_frame.get(int(entry['frame_id']))
                if observed_native is None or native_mask.shape != observed_native.shape or np.any(native_mask & ~observed_native):
                    raise ValueError(f'Synthetic native mask is not a subset of the captured observed mask: {context_id}')
                hashes = entry.get('geometry_hashes', {})
                if (hashes.get('depth_mm') != array_sha256(depth) or
                        hashes.get('render_mask') != array_sha256(rendered) or
                        hashes.get('synthetic_mask') != array_sha256(visible) or
                        entry.get('synthetic_native_mask_sha256') != array_sha256(native_mask)):
                    raise ValueError(f'Synthetic query geometry hashes differ from context metadata: {context_id}')
                appearance = _appearance_context_check(
                    arrays['query_rgb_rgb'], arrays['query_rgb_gray'], visible)
                if not appearance['background_unchanged'] or not appearance['foreground_is_achromatic']:
                    raise ValueError(f'Synthetic query appearance ablation changed background or was not gray: {context_id}')
                appearance_checks.append(dict(context_id=context_id, role=role, **appearance))
                context_checks.append(dict(context_id=context_id, visible_mask_intersection=True,
                                           native_mask_intersection=True, query_depth=True))

        # Shared geometry must hash identically across every appearance condition.
        camera_groups = {}
        for context_id, arrays in contexts.items():
            entry = context_entries[context_id]
            role = entry.get('role')
            group = entry.get('geometry_group')
            if group is None:
                continue
            camera_groups.setdefault(group, []).append((role, arrays))
        geometry_checks = []
        for group, members in camera_groups.items():
            frame_id = next((int(context_entries[key]['frame_id']) for key in context_entries
                             if context_entries[key].get('geometry_group') == group and
                             context_entries[key].get('role') == 'real_frame'), None)
            roles_ok = frame_id is not None
            if roles_ok:
                template_offsets = {int(context_entries[key]['template_offset_deg']) for key in context_entries
                                    if context_entries[key].get('geometry_group') == group and
                                    context_entries[key].get('role') == 'template'}
                query_offsets = {int(context_entries[key]['query_offset_deg']) for key in context_entries
                                 if context_entries[key].get('geometry_group') == group and
                                 context_entries[key].get('role') == 'synthetic_query'}
                roles_ok = template_offsets == set(TEMPLATE_OFFSETS)
                expected_queries = set(QUERY_OFFSETS) if frame_id in SYNTHETIC_CARRIERS else set()
                roles_ok &= query_offsets == expected_queries
                roles_ok &= sum(role == 'real_frame' for role, _ in members) == 1
            geometry_checks.append(dict(group=group, field='required_context_roles', matched=bool(roles_ok)))
            for field in ('crop_k', 'crop_from_native', 'native_k', 'observed_crop_mask'):
                hashes = {array_sha256(item[field]) for role, item in members if field in item}
                field_present_everywhere = all(field in item for role, item in members)
                if len(hashes) > 1 or not field_present_everywhere:
                    geometry_checks.append(dict(group=group, field=field, matched=False))
                elif hashes:
                    geometry_checks.append(dict(group=group, field=field, matched=True,
                                                sha256=next(iter(hashes))))
        required_groups = {entry.get('geometry_group') for entry in context_entries.values()
                           if entry.get('role') == 'real_frame'}
        checked_groups = {row['group'] for row in geometry_checks}
        shared_match = bool(geometry_checks) and required_groups.issubset(checked_groups) and all(
            row['matched'] for row in geometry_checks)

        templates_by_frame_hyp = {}
        for context_id, arrays in contexts.items():
            entry = context_entries[context_id]
            if entry.get('role') == 'template':
                templates_by_frame_hyp[(int(entry['frame_id']), int(entry['template_offset_deg']))] = (entry, arrays)

        patch_rows = {}
        real_query_contexts = {}
        synthetic_query_contexts = {}
        for context_id, arrays in contexts.items():
            entry = context_entries[context_id]
            if entry.get('role') == 'real_frame':
                real_query_contexts[int(entry['frame_id'])] = (entry, arrays)
            elif entry.get('role') == 'synthetic_query':
                synthetic_query_contexts[context_id] = (entry, arrays)

        # Verify crop-depth projection parity and prepare the two fixed analytic banks per frame.
        banks_by_frame = {}
        parity_rows = []
        for frame_id in FRAME_IDS:
            banks = []
            for offset in TEMPLATE_OFFSETS:
                item = templates_by_frame_hyp.get((frame_id, offset))
                if item is None:
                    continue
                entry, arrays = item
                ids, pixels, points = _source_rows(arrays)
                # Convert translation explicitly at the native/crop boundary.
                template_pose_m = np.asarray(arrays['template_pose_m'], dtype=np.float64).copy()
                pose_crop_mm = native_to_crop_pose_mm(template_pose_m, arrays['crop_from_native'])
                pose_crop_m = pose_crop_mm.copy(); pose_crop_m[:3, 3] *= .001
                parity = projection_parity(points, pixels, pose_crop_m, arrays['crop_k'])
                parity_rows.append(dict(frame_id=frame_id, template_offset_deg=offset, **parity))
                bank = verify_fixed_patch_bank(entry, arrays)
                bank['frame_id'] = frame_id
                bank['template_offset_deg'] = offset
                banks.append(bank)
            banks_by_frame[frame_id] = banks
        # Oracle PnP is a CPU control per unique synthetic query, independent of
        # query/template appearance forwards. Each of the eight controls is run once.
        oracle_controls = []
        for frame_id in SYNTHETIC_CARRIERS:
            for query_offset in QUERY_OFFSETS:
                control_id = f'oracle-{frame_id}-q{query_offset}'
                template_offset = 0 if query_offset == 8 else 180
                template_item = templates_by_frame_hyp.get((frame_id, template_offset))
                query_id = f'synthetic-query-{frame_id:04d}-{query_offset:03d}'
                query_arrays = contexts.get(query_id)
                control = dict(
                    control_id=control_id, frame_id=frame_id, query_offset_deg=query_offset,
                    near_template_offset_deg=template_offset, state='unavailable', reason=None,
                    oracle_pnp=None, projection_parity=None,
                )
                if template_item is None or query_arrays is None:
                    control['reason'] = 'synthetic_query_or_near_template_context_unavailable'
                else:
                    _, template_arrays = template_item
                    ids, source_xy, points = _source_rows(template_arrays)
                    query_pose = np.asarray(query_arrays['known_query_pose_m'], dtype=np.float64)
                    query_mask = np.asarray(query_arrays['synthetic_query_mask'], dtype=bool)
                    near_pose = crop_pose_from_context(template_arrays)
                    identity, identity_arrays = synthetic_identity_metrics(
                        points, source_xy, ids, identity_arrays_endpoints := project_object_points(
                            points, query_pose, template_arrays['crop_k'])[0],
                        np.ones(len(points), dtype=np.float32), query_pose,
                        query_arrays['query_depth_mm'], query_mask, template_arrays['crop_k'],
                        float(capture.get('object_bbox_diagonal_m', 0.)))
                    parity = projection_parity(points, source_xy, near_pose, template_arrays['crop_k'])
                    parity_rows.append(dict(control_id=control_id, frame_id=frame_id,
                                            template_offset_deg=template_offset, **parity))
                    rng_seed = (frame_id * 17 + (query_offset // 180) * 4 +
                                (template_offset // 180) * 2) & 0x7fffffff
                    oracle = oracle_pnp(
                        points, identity_arrays_endpoints, identity_arrays['true_visible'], source_xy,
                        near_pose, query_pose, template_arrays['crop_k'],
                        float(capture.get('object_bbox_diagonal_m', 0.)), rng_seed)
                    control.update(
                        state=oracle['state'], reason=oracle.get('reason'), oracle_pnp=oracle,
                        projection_parity=parity,
                        eligible_sources=int(len(points)),
                        truly_visible_sources=identity['truly_visible_sources'],
                        source_indices=array_info(ids),
                        query_context_id=query_id,
                    )
                oracle_controls.append(control)
        report['oracle_pnp_controls'] = oracle_controls

        repeats = {}
        for condition_id, entry in forward_entries.items():
            condition = condition_entries[condition_id]
            if condition.get('repeat_of'):
                repeats.setdefault(condition['repeat_of'], []).append(entry)
        repeat_checks = []
        for original_id, entries in repeats.items():
            original = forward_entries.get(original_id)
            same = original is not None and len(entries) == 1 and entries[0].get('output_hashes') == original.get('output_hashes')
            repeat_checks.append(dict(original=original_id, repeat=entries[0]['condition_id'] if entries else None,
                                      matched=bool(same), original_hashes=None if original is None else original.get('output_hashes'),
                                      repeat_hashes=entries[0].get('output_hashes') if entries else None))
        repeat_match = len(repeat_checks) == 4 and all(row['matched'] for row in repeat_checks)

        for condition_id, condition in condition_entries.items():
            result_row = dict(
                condition_id=condition_id, kind=condition['kind'], frame_id=condition['frame_id'],
                state=condition.get('state'), reason=condition.get('reason'),
                query_offset_deg=condition.get('query_offset_deg'),
                query_appearance=condition.get('query_appearance'),
                template_offset_deg=condition.get('template_offset_deg'),
                template_appearance=condition.get('template_appearance'),
                repeat_of=condition.get('repeat_of'),
            )
            fentry = forward_entries.get(condition_id)
            if not fentry:
                report['forwards'].append(result_row)
                if condition['kind'] == 'synthetic':
                    report['synthetic'].append(result_row)
                elif condition['kind'] == 'real':
                    report['real'].append(result_row)
                continue
            if fentry.get('condition_id') != condition_id or fentry.get('context_refs') != condition.get('context_refs'):
                raise ValueError(f'Forward packet provenance differs from frozen condition: {condition_id}')
            forward = forwards[condition_id]
            refs = condition.get('context_refs', {})
            if set(refs) != {'query', 'template', 'observed_frame'} or any(key not in context_entries for key in refs.values()):
                raise ValueError(f'Forward packet references missing or invalid context: {condition_id}')
            template_context = contexts[refs['template']]
            query_context = contexts[refs['query']]
            ids, source_xy, points, endpoints, confidence, target_indices = _forward_mapping(template_context, forward)
            if fentry.get('derived_array_hashes') != _array_hashes_for_forward(template_context, forward):
                raise ValueError(f'Derived correspondence hashes differ from capture manifest: {condition_id}')
            expected_output_hashes = dict(flow=array_sha256(forward['flow']),
                                          confidence=array_sha256(forward['confidence']))
            if fentry.get('output_hashes') != expected_output_hashes or condition.get('output_hashes') != expected_output_hashes:
                raise ValueError(f'Forward output hashes differ from capture manifest: {condition_id}')
            result_row.update(
                state='captured', elapsed_ms=fentry.get('elapsed_ms'),
                packet_path=fentry['path'], packet_sha256=fentry['sha256'],
                output_hashes=fentry['output_hashes'],
                derived_array_hashes=_array_hashes_for_forward(template_context, forward),
                retained_sources=int(len(ids)), confidence_gt_03=int((confidence > .3).sum()),
            )
            if condition['kind'] == 'repeat':
                report['forwards'].append(result_row)
                continue
            if condition['kind'] == 'synthetic':
                qpose = np.asarray(query_context['known_query_pose_m'], dtype=np.float64)
                qmask = np.asarray(query_context['synthetic_query_mask'], dtype=bool)
                identity, identity_arrays = synthetic_identity_metrics(
                    points, source_xy, ids, endpoints, confidence, qpose,
                    query_context['query_depth_mm'], qmask, template_context['crop_k'],
                    float(capture['object_bbox_diagonal_m']))
                learned = fit_learned_packet(
                    points, endpoints, confidence, ids, qmask, qmask,
                    crop_pose_from_context(template_context), template_context['crop_k'],
                    template_context['crop_from_native'], template_context['native_k'],
                    _frame_from_context(condition['frame_id'], real_query_contexts, query_context),
                    query_context['synthetic_native_mask'], int(fentry['rng_seed']))
                if learned['retained_correspondences'] != identity['confident_masked_correspondences']:
                    raise ValueError(f'Synthetic identity/PnP retained correspondence counts differ: {condition_id}')
                qoffset = int(condition['query_offset_deg'])
                oracle_control_id = f'oracle-{condition["frame_id"]}-q{qoffset}'
                result_row['oracle_pnp_control_id'] = oracle_control_id
                identity_summary = identity
                negative = dict(
                    state='not_a_negative',
                    contract_validated=learned.get('validation_state') == 'accepted',
                    confident_masked_endpoints=identity_summary['confident_masked_correspondences'],
                    confident_invisible_endpoints=identity_summary['confident_invisible_source_endpoints'],
                    confident_invisible_fraction=identity_summary['confident_invisible_fraction'],
                    median_invisible_identity_distance_fraction=(
                        identity_summary['confident_invisible_median_identity_distance_fraction']),
                    thresholds=dict(min_confident_invisible_endpoints=24,
                                    min_confident_invisible_fraction=.05,
                                    min_median_identity_distance_fraction=.1),
                )
                negative['thresholds_passed'] = bool(
                    negative['confident_invisible_endpoints'] >= 24 and
                    negative['confident_invisible_fraction'] >= .05 and
                    negative['median_invisible_identity_distance_fraction'] is not None and
                    negative['median_invisible_identity_distance_fraction'] > .1)
                negative['state'] = ('validated_wrong_surface_false_acceptance'
                                     if negative['contract_validated'] and negative['thresholds_passed']
                                     else 'below_frozen_negative_gate')
                learned['synthetic_identity_negative'] = negative
                result_row['learned_pnp_contract'] = learned
                result_row['validated_wrong_surface_false_acceptance'] = negative
                result_row['synthetic_identity'] = identity
                patch = audit_patch_identity(
                    query_context['query_rgb_' + condition['query_appearance']], qmask,
                    forward['flow'], forward['confidence'],
                    banks_by_frame[int(condition['frame_id'])][
                        0 if condition['template_offset_deg'] == 0 else 1],
                    banks_by_frame[int(condition['frame_id'])], float(capture['object_bbox_diagonal_m']))
                result_row['patch_identity'] = patch['summary']
                result_row['patch_identity_anchors'] = patch['rows']
                patch_rows[condition_id] = patch
            else:
                context_entry, _ = real_query_contexts[int(condition['frame_id'])]
                real_mask = np.asarray(query_context['observed_crop_mask'], dtype=bool)
                patch = audit_patch_identity(
                    query_context['crop_rgb'], real_mask, forward['flow'], forward['confidence'],
                    banks_by_frame[int(condition['frame_id'])][
                        0 if condition['template_offset_deg'] == 0 else 1],
                    banks_by_frame[int(condition['frame_id'])], float(capture['object_bbox_diagonal_m']))
                # Validation uses only captured correspondences and the current observed native mask.
                learned = fit_learned_packet(
                    points, endpoints, confidence, ids, real_mask, real_mask,
                    crop_pose_from_context(template_context), template_context['crop_k'],
                    template_context['crop_from_native'], template_context['native_k'],
                    _frame_from_context(int(condition['frame_id']), real_query_contexts, None),
                    query_context['native_mask'], int(fentry['rng_seed']))
                result_row['learned_pnp_contract'] = learned
                result_row['patch_identity'] = patch['summary']
                result_row['patch_identity_anchors'] = patch['rows']
                patch_rows[condition_id] = patch
            report['forwards'].append(result_row)
            if condition['kind'] == 'synthetic':
                report['synthetic'].append(result_row)
            elif condition.get('repeat_of') is None:
                report['real'].append(result_row)
            if condition['kind'] != 'repeat':
                mosaic_path = _write_mosaic(
                    output_root / 'patch_mosaics' / f'{condition_id.replace(":", "_")}.png',
                    patch['mosaic_patches'])
                result_row['patch_mosaic'] = mosaic_path

        projection_parity_ok = bool(parity_rows) and all(row['passed'] for row in parity_rows)
        report['invariants'] = dict(
            repeat_hashes_match=repeat_match,
            repeat_checks=repeat_checks,
            shared_camera_depth_masks_match=shared_match,
            shared_geometry_checks=geometry_checks,
            projection_parity_passed=projection_parity_ok,
            projection_parity=parity_rows,
            real_records_leak_free=real_leak_free,
            context_arrays_consistent=True,
            context_checks=context_checks,
            appearance_ablation_consistent=True,
            appearance_checks=appearance_checks,
        )
        report['decisions'] = _decision_summary(report, capture, banks_by_frame)
        report['status'] = 'complete' if (capture.get('complete') is True and repeat_match and shared_match and
                                          projection_parity_ok and real_leak_free) else 'invariants_failed'
        report['complete'] = True
        report['completed_unix'] = time.time()
        report['independent_real_accuracy_verified'] = False
        report['real_pose_accuracy_claimed'] = False
        save_result(output_root / 'audit.json', report)
    except BaseException as error:
        report['status'] = 'failed'
        report['complete'] = False
        report['failure'] = dict(type=type(error).__name__, message=str(error))
        report['failed_unix'] = time.time()
        try:
            save_result(output_root / 'audit.json', report)
        except Exception:
            pass
        raise
    return report


def crop_pose_from_context(template_context):
    pose_mm = native_to_crop_pose_mm(template_context['template_pose_m'], template_context['crop_from_native'])
    pose_m = pose_mm.copy()
    pose_m[:3, 3] *= .001
    return pose_m


def _frame_from_context(frame_id, real_query_contexts, synthetic_query_context):
    if synthetic_query_context is not None:
        # The contract reads only image dimensions/calibration and the explicit mask.
        # Synthetic validation uses a dummy image so no real appearance data enters it.
        rgb = np.zeros(tuple(synthetic_query_context['synthetic_native_mask'].shape) + (3,), dtype=np.uint8)
        return Frame(int(frame_id), rgb, synthetic_query_context['native_k'])
    entry, arrays = real_query_contexts[int(frame_id)]
    return Frame(int(frame_id), arrays['native_rgb'], arrays['native_k'])


def _decision_summary(report, capture, banks_by_frame):
    synthetic_rows = report['synthetic']
    controls = report.get('oracle_pnp_controls', [])
    oracle_states = [row.get('state') for row in controls]
    query_ablation = {}
    template_ablation = {}
    for entry in capture.get('contexts', []):
        if entry.get('role') == 'synthetic_query':
            query_ablation[(int(entry['frame_id']), int(entry['query_offset_deg']))] = (
                entry.get('query_appearance_ablation') or {}).get('state')
        elif entry.get('role') == 'template':
            template_ablation[(int(entry['frame_id']), int(entry['template_offset_deg']))] = (
                entry.get('appearance_ablation') or {}).get('state')

    def _state_counts(rows):
        return dict(
            captured=sum(row.get('state') == 'captured' for row in rows),
            unavailable=sum(row.get('state') == 'unavailable' for row in rows),
            failed=sum(row.get('state') == 'failed' for row in rows),
            other=sum(row.get('state') not in ('captured', 'unavailable', 'failed') for row in rows),
            rows=len(rows),
        )

    def _ablation_informativeness(query_offset, template_offset):
        query_states = {frame_id: query_ablation.get((frame_id, query_offset))
                        for frame_id in SYNTHETIC_CARRIERS}
        template_states = {frame_id: template_ablation.get((frame_id, template_offset))
                           for frame_id in SYNTHETIC_CARRIERS}
        return dict(
            query_by_frame={str(key): value for key, value in query_states.items()},
            template_by_frame={str(key): value for key, value in template_states.items()},
            informative=all(value == 'informative' for value in
                            (*query_states.values(), *template_states.values())),
        )

    def _near_rows(query_offset, query_appearance='rgb', template_appearance='rgb'):
        near_offset = 0 if query_offset == 8 else 180
        return [row for row in synthetic_rows
                if row.get('query_offset_deg') == query_offset and
                row.get('query_appearance') == query_appearance and
                row.get('template_offset_deg') == near_offset and
                row.get('template_appearance') == template_appearance]

    def _near_frame_gate(row):
        identity = row.get('synthetic_identity') or {}
        visible = int(identity.get('truly_visible_sources', 0))
        confident_visible = int(identity.get('confidence_gt_03_visible', 0))
        confident_correct = int(identity.get('correct_confident_visible', 0))
        accuracy = confident_correct / confident_visible if confident_visible else None
        availability = confident_visible / visible if visible else None
        passed = bool(row.get('state') == 'captured' and accuracy is not None and availability is not None and
                      accuracy >= .90 and availability >= .50)
        return dict(
            condition_id=row['condition_id'], frame_id=row['frame_id'], state=row.get('state'),
            visible_sources=visible, confidence_gt_03_visible=confident_visible,
            correct_confident_visible=confident_correct,
            correct_fraction_confident_visible=accuracy,
            confidence_availability_on_visible=availability, passed=passed,
        )

    native_positive_controls = {}
    patch_calibration = {}
    for query_offset in QUERY_OFFSETS:
        near = [_near_frame_gate(row) for row in _near_rows(query_offset, 'rgb', 'rgb')]
        frame_rows = {int(row['frame_id']): row for row in near}
        for frame_id in SYNTHETIC_CARRIERS:
            frame_rows.setdefault(frame_id, dict(frame_id=frame_id, state='unavailable',
                                                 visible_sources=0, confidence_gt_03_visible=0,
                                                 correct_confident_visible=0,
                                                 correct_fraction_confident_visible=None,
                                                 confidence_availability_on_visible=None, passed=False))
        passed_frames = sum(row['passed'] for row in frame_rows.values())
        native_positive_controls[str(query_offset)] = dict(
            frames=[frame_rows[frame_id] for frame_id in SYNTHETIC_CARRIERS],
            passed_carriers=int(passed_frames), required_carriers=3,
            side_passed=passed_frames >= 3,
            thresholds=dict(correct_fraction_confident_visible=.90,
                            confidence_availability_on_visible=.50),
        )
        calibration_rows = []
        for row in _near_rows(query_offset, 'rgb', 'rgb'):
            patch = row.get('patch_identity') or {}
            bank = banks_by_frame.get(int(row['frame_id']), [])
            eligible = int(patch.get('source_eligible', 0))
            enough_texture = eligible >= 16
            calibration_rows.append(dict(
                frame_id=row['frame_id'], state=patch.get('state'),
                source_eligible=eligible, supported=int(patch.get('supported', 0)),
                enough_distinctive_texture=enough_texture,
                spatial_gate_passed=patch.get('state') == 'distinctive_current_image_support',
            ))
        eligible_carriers = [row for row in calibration_rows if row['enough_distinctive_texture']]
        calibration_passed = sum(row['spatial_gate_passed'] for row in eligible_carriers)
        patch_calibration[str(query_offset)] = dict(
            frames=calibration_rows, carriers_with_at_least_16_eligible=len(eligible_carriers),
            carriers_passing_spatial_gate=int(calibration_passed), required_carriers=3,
            side_passed=(len(eligible_carriers) >= 3 and calibration_passed >= 3),
            state=('calibrated' if len(eligible_carriers) >= 3 and calibration_passed >= 3
                   else 'uninformative' if len(eligible_carriers) < 3 else 'failed'),
        )

    gray_query_comparison = {}
    for query_offset in QUERY_OFFSETS:
        appearance_rows = {}
        near_offset = 0 if query_offset == 8 else 180
        ablation = _ablation_informativeness(query_offset, near_offset)
        for template_appearance in APPEARANCES:
            rows = _near_rows(query_offset, 'gray', template_appearance)
            visible = sum(int((row.get('synthetic_identity') or {}).get('truly_visible_sources', 0)) for row in rows)
            correct = sum(int((row.get('synthetic_identity') or {}).get('correct_confident_visible', 0)) for row in rows)
            invisible = sum(int((row.get('synthetic_identity') or {}).get('confident_invisible_source_endpoints', 0)) for row in rows)
            masked = sum(int((row.get('synthetic_identity') or {}).get('confident_masked_correspondences', 0)) for row in rows)
            false_accepted = sum(
                bool((row.get('validated_wrong_surface_false_acceptance') or {}).get('state') ==
                     'validated_wrong_surface_false_acceptance') for row in rows)
            states = _state_counts(rows)
            appearance_rows[template_appearance] = dict(
                captured_conditions=states['captured'],
                unavailable_conditions=states['unavailable'], failed_conditions=states['failed'],
                other_conditions=states['other'], expected_conditions=len(SYNTHETIC_CARRIERS),
                state_counts=states,
                visible_sources=visible, correct_confident_visible=correct,
                correct_identity_availability=correct / visible if visible else None,
                confident_invisible_endpoints=invisible,
                confident_masked_endpoints=masked,
                confident_invisible_fraction=invisible / masked if masked else None,
                validated_false_acceptance_conditions=false_accepted,
                appearance_informativeness=ablation,
                appearance_ablation_uninformative=not ablation['informative'],
            )
        native_value = appearance_rows['rgb']['correct_identity_availability']
        gray_value = appearance_rows['gray']['correct_identity_availability']
        improvement = None if native_value is None or gray_value is None else gray_value - native_value
        native_false = appearance_rows['rgb']['confident_invisible_fraction']
        gray_false = appearance_rows['gray']['confident_invisible_fraction']
        false_acceptance_hypotheses = []
        query_negative_totals = {}
        expected_per_hypothesis = len(SYNTHETIC_CARRIERS)
        for query_appearance in APPEARANCES:
            accepted_total = 0
            fully_evaluated = True
            state_total = dict(captured=0, unavailable=0, failed=0, other=0, rows=0)
            evaluated_total = 0
            for template_offset in TEMPLATE_OFFSETS:
                for template_appearance in APPEARANCES:
                    hypothesis_rows = [
                        row for row in synthetic_rows
                        if row.get('query_offset_deg') == query_offset and
                        row.get('query_appearance') == query_appearance and
                        row.get('template_offset_deg') == template_offset and
                        row.get('template_appearance') == template_appearance
                    ]
                    counts = _state_counts(hypothesis_rows)
                    for key in state_total:
                        state_total[key] += counts[key]
                    evaluable = [
                        row for row in hypothesis_rows
                        if row.get('state') == 'captured' and
                        (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') in
                        ('validated_wrong_surface_false_acceptance', 'below_frozen_negative_gate')
                    ]
                    accepted = sum(
                        (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') ==
                        'validated_wrong_surface_false_acceptance' for row in evaluable)
                    accepted_total += accepted
                    evaluated_total += len(evaluable)
                    complete_hypothesis = (
                        len(hypothesis_rows) == expected_per_hypothesis and
                        counts['captured'] == expected_per_hypothesis and
                        counts['unavailable'] == 0 and counts['failed'] == 0 and
                        counts['other'] == 0 and len(evaluable) == expected_per_hypothesis
                    )
                    fully_evaluated &= complete_hypothesis
                    false_acceptance_hypotheses.append(dict(
                        query_appearance=query_appearance,
                        template_offset_deg=template_offset,
                        template_appearance=template_appearance,
                        expected_conditions=expected_per_hypothesis,
                        state_counts=counts, negative_cases_evaluated=len(evaluable),
                        validated_false_acceptances=int(accepted),
                        complete=bool(complete_hypothesis),
                    ))
            expected_total = expected_per_hypothesis * len(TEMPLATE_OFFSETS) * len(APPEARANCES)
            complete_query = (fully_evaluated and state_total['rows'] == expected_total and
                              evaluated_total == expected_total)
            query_negative_totals[query_appearance] = dict(
                expected_conditions=expected_total, state_counts=state_total,
                negative_cases_evaluated=evaluated_total,
                validated_false_acceptances=int(accepted_total),
                complete=bool(complete_query),
            )
        rgb_negatives = query_negative_totals['rgb']
        gray_negatives = query_negative_totals['gray']
        gray_query_template_totals = {}
        expected_per_template_appearance = expected_per_hypothesis * len(TEMPLATE_OFFSETS)
        all_gray_query_template_hypotheses_complete = True
        for template_appearance in APPEARANCES:
            hypothesis_rows = [
                row for row in synthetic_rows
                if row.get('query_offset_deg') == query_offset and
                row.get('query_appearance') == 'gray' and
                row.get('template_appearance') == template_appearance
            ]
            counts = _state_counts(hypothesis_rows)
            evaluable = [
                row for row in hypothesis_rows
                if row.get('state') == 'captured' and
                (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') in
                ('validated_wrong_surface_false_acceptance', 'below_frozen_negative_gate')
            ]
            accepted = sum(
                (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') ==
                'validated_wrong_surface_false_acceptance' for row in evaluable)
            by_surface_hypothesis = {}
            hypotheses_complete = True
            for template_offset in TEMPLATE_OFFSETS:
                rows = [row for row in hypothesis_rows
                        if row.get('template_offset_deg') == template_offset]
                surface_counts = _state_counts(rows)
                surface_evaluable = [
                    row for row in rows
                    if row.get('state') == 'captured' and
                    (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') in
                    ('validated_wrong_surface_false_acceptance', 'below_frozen_negative_gate')
                ]
                surface_accepted = sum(
                    (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') ==
                    'validated_wrong_surface_false_acceptance' for row in surface_evaluable)
                surface_complete = (
                    len(rows) == expected_per_hypothesis and
                    surface_counts['captured'] == expected_per_hypothesis and
                    surface_counts['unavailable'] == 0 and surface_counts['failed'] == 0 and
                    surface_counts['other'] == 0 and len(surface_evaluable) == expected_per_hypothesis
                )
                hypotheses_complete &= surface_complete
                by_surface_hypothesis[str(template_offset)] = dict(
                    expected_conditions=expected_per_hypothesis,
                    state_counts=surface_counts,
                    negative_cases_evaluated=len(surface_evaluable),
                    validated_false_acceptances=int(surface_accepted),
                    complete=bool(surface_complete),
                )
            complete_template = (
                len(hypothesis_rows) == expected_per_template_appearance and
                counts['captured'] == expected_per_template_appearance and
                counts['unavailable'] == 0 and counts['failed'] == 0 and
                counts['other'] == 0 and len(evaluable) == expected_per_template_appearance and
                hypotheses_complete
            )
            all_gray_query_template_hypotheses_complete &= complete_template
            gray_query_template_totals[template_appearance] = dict(
                query_appearance='gray', expected_conditions=expected_per_template_appearance,
                state_counts=counts, negative_cases_evaluated=len(evaluable),
                validated_false_acceptances=int(accepted),
                by_source_surface_hypothesis=by_surface_hypothesis,
                complete=bool(complete_template),
            )
        all_32_conditions_complete = bool(
            rgb_negatives['expected_conditions'] + gray_negatives['expected_conditions'] == 32 and
            rgb_negatives['complete'] and gray_negatives['complete'])
        native_template_negatives = gray_query_template_totals['rgb']
        gray_template_negatives = gray_query_template_totals['gray']
        no_increase_false_acceptance = bool(
            all_32_conditions_complete and all_gray_query_template_hypotheses_complete and
            native_template_negatives['complete'] and gray_template_negatives['complete'] and
            gray_template_negatives['validated_false_acceptances'] <=
            native_template_negatives['validated_false_acceptances'])
        near_complete = all(
            appearance_rows[appearance]['state_counts']['rows'] == len(SYNTHETIC_CARRIERS) and
            appearance_rows[appearance]['state_counts']['captured'] == len(SYNTHETIC_CARRIERS) and
            appearance_rows[appearance]['state_counts']['unavailable'] == 0 and
            appearance_rows[appearance]['state_counts']['failed'] == 0 and
            appearance_rows[appearance]['state_counts']['other'] == 0
            for appearance in APPEARANCES)
        appearance_informative = all(row['appearance_informativeness']['informative']
                                     for row in appearance_rows.values())
        controlled_gap = bool(
            improvement is not None and improvement >= GRAY_RECOVERY_IMPROVEMENT_MIN and
            no_increase_false_acceptance and near_complete and appearance_informative)
        gray_query_comparison[str(query_offset)] = dict(
            templates=appearance_rows,
            correct_identity_availability_improvement=improvement,
            near_view_conditions_complete=near_complete,
            appearance_informative=appearance_informative,
            appearance_ablation_uninformative=not appearance_informative,
            negative_false_acceptance_by_template_hypothesis=false_acceptance_hypotheses,
            negative_false_acceptance_by_query_appearance=query_negative_totals,
            negative_false_acceptance_by_template_appearance_on_gray_query=gray_query_template_totals,
            negative_false_acceptance_comparison_axis='same_gray_query_native_vs_gray_template',
            all_32_negative_conditions_accounted=all_32_conditions_complete,
            no_increase_in_negative_false_acceptance=no_increase_false_acceptance,
            false_acceptance_fraction_native_template=native_false,
            false_acceptance_fraction_gray_template=gray_false,
            controlled_color_domain_gap=controlled_gap,
            required_improvement=GRAY_RECOVERY_IMPROVEMENT_MIN,
        )

    wrong_surface_negatives = [
        dict(condition_id=row['condition_id'], frame_id=row['frame_id'],
             query_offset_deg=row.get('query_offset_deg'), query_appearance=row.get('query_appearance'),
             template_offset_deg=row.get('template_offset_deg'), template_appearance=row.get('template_appearance'),
             **row['validated_wrong_surface_false_acceptance'])
        for row in synthetic_rows
        if (row.get('validated_wrong_surface_false_acceptance') or {}).get('state') ==
        'validated_wrong_surface_false_acceptance'
    ]

    real_counts = {}
    real_by_frame_hyp_appearance = {}
    for row in report['real']:
        patch = row.get('patch_identity') or {}
        key = f"{row['frame_id']}:{row['template_offset_deg']}:{row['template_appearance']}"
        details = dict(
            state=row.get('state'), patch_state=patch.get('state'),
            source_eligible=int(patch.get('source_eligible', 0)),
            source_eligible_before_cap=int(patch.get('source_eligible_before_cap', 0)),
            target_eligible=int(patch.get('target_eligible', 0)),
            rejected=int(patch.get('rejected', 0)), supported=int(patch.get('supported', 0)),
            supported_cells_4x4=patch.get('supported_cells_4x4'),
            supported_hull_fraction=patch.get('supported_hull_fraction'),
        )
        real_counts[key] = details
        real_by_frame_hyp_appearance[(int(row['frame_id']), int(row['template_offset_deg']),
                                      row['template_appearance'])] = details
    real_appearance_comparison = []
    for frame_id in FRAME_IDS:
        for template_offset in TEMPLATE_OFFSETS:
            native = real_by_frame_hyp_appearance.get((frame_id, template_offset, 'rgb'))
            gray = real_by_frame_hyp_appearance.get((frame_id, template_offset, 'gray'))
            if native is None or gray is None:
                continue
            real_appearance_comparison.append(dict(
                frame_id=frame_id, template_offset_deg=template_offset,
                native_rgb=native, gray_template=gray,
                supported_anchor_gain=gray['supported'] - native['supported'],
                template_appearance_ablation_state=template_ablation.get((frame_id, template_offset)),
                appearance_ablation_informative=(
                    template_ablation.get((frame_id, template_offset)) == 'informative'),
                formerly_supported_gate_lost=(
                    native['patch_state'] == 'distinctive_current_image_support' and
                    gray['patch_state'] != 'distinctive_current_image_support'),
                labeled_texture_support=(native['source_eligible'] >= 16 and gray['source_eligible'] >= 16 and
                                         native['target_eligible'] >= 8),
            ))
    real_score_by_frame_hyp_appearance = {}
    for row in report['real']:
        validation = row.get('learned_pnp_contract') or {}
        stats = validation.get('validation_stats') or {}
        real_score_by_frame_hyp_appearance[(int(row['frame_id']), int(row['template_offset_deg']),
                                            row['template_appearance'])] = dict(
            state=validation.get('validation_state'), score=stats.get('score'),
            validation_reason=validation.get('validation_reason'))
    support_vs_score = []
    for frame_id in FRAME_IDS:
        for appearance in APPEARANCES:
            left = real_by_frame_hyp_appearance.get((frame_id, 0, appearance))
            right = real_by_frame_hyp_appearance.get((frame_id, 180, appearance))
            left_score = real_score_by_frame_hyp_appearance.get((frame_id, 0, appearance))
            right_score = real_score_by_frame_hyp_appearance.get((frame_id, 180, appearance))
            if left is None or right is None or left_score is None or right_score is None:
                continue
            left_supported = left['patch_state'] == 'distinctive_current_image_support'
            right_supported = right['patch_state'] == 'distinctive_current_image_support'
            left_value, right_value = left_score.get('score'), right_score.get('score')
            score_preferred = None
            if left_value is not None and right_value is not None and left_value != right_value:
                score_preferred = 0 if left_value > right_value else 180
            support_hypothesis = 0 if left_supported and not right_supported else (
                180 if right_supported and not left_supported else None)
            support_vs_score.append(dict(
                frame_id=frame_id, template_appearance=appearance,
                support_state_by_hypothesis=dict(
                    **{'0': left['patch_state'], '180': right['patch_state']}),
                contract_score_by_hypothesis=dict(
                    **{'0': left_score, '180': right_score}),
                unique_supported_hypothesis=support_hypothesis,
                contract_score_preferred_hypothesis=score_preferred,
                current_image_support_contradicts_contract_score=(
                    support_hypothesis is not None and score_preferred is not None and
                    support_hypothesis != score_preferred),
            ))
    expected_real_appearance_comparisons = len(FRAME_IDS) * len(TEMPLATE_OFFSETS)
    real_appearance_informative = (
        len(real_appearance_comparison) == expected_real_appearance_comparisons and
        all(row['appearance_ablation_informative'] for row in real_appearance_comparison))
    gain_frames = {row['frame_id'] for row in real_appearance_comparison
                   if row['appearance_ablation_informative'] and row['labeled_texture_support'] and
                   row['supported_anchor_gain'] >= 4}
    formerly_supported_losses = [row for row in real_appearance_comparison
                                 if row['appearance_ablation_informative'] and
                                 row['formerly_supported_gate_lost']]
    color_real_support_gate = bool(real_appearance_informative and len(gain_frames) >= 3 and
                                   not formerly_supported_losses)

    oracle_pass = len(controls) == 8 and all(state == 'passed' for state in oracle_states)
    projection_pass = report['invariants']['projection_parity_passed'] is True
    base_invariants = all(report['invariants'].get(key) is True for key in (
        'repeat_hashes_match', 'shared_camera_depth_masks_match',
        'projection_parity_passed', 'real_records_leak_free'))
    native_positive_pass = all(row['side_passed'] for row in native_positive_controls.values())
    patch_calibration_pass = all(row['side_passed'] for row in patch_calibration.values())
    gray_gap_pass = any(row['controlled_color_domain_gap'] for row in gray_query_comparison.values())
    if not base_invariants:
        permitted_conclusion = 'implementation_invariant_failure'
    elif not oracle_pass or not projection_pass:
        permitted_conclusion = 'geometry_crop_or_solver_control_unresolved'
    elif not native_positive_pass or not patch_calibration_pass:
        permitted_conclusion = 'synthetic_near_view_controls_failed_or_uninformative'
    elif wrong_surface_negatives:
        permitted_conclusion = 'validated_correspondences_can_accept_wrong_synthetic_surface_identity'
    elif gray_gap_pass and color_real_support_gate:
        permitted_conclusion = 'controlled_color_domain_gap_demonstrated_for_further_review'
    elif gray_gap_pass:
        permitted_conclusion = 'synthetic_color_domain_gap_without_required_real_support_gain'
    elif any(row['current_image_support_contradicts_contract_score'] for row in support_vs_score):
        permitted_conclusion = 'real_current_image_support_contradicts_contract_score_for_tested_identity'
    elif any(row.get('patch_state') == 'distinctive_current_image_support'
             for row in real_counts.values()):
        permitted_conclusion = 'real_current_image_texture_support_exists_for_tested_identity_only'
    else:
        permitted_conclusion = 'real_identity_unobservable_or_ambiguous_under_tested_patch_evidence'

    return dict(
        oracle_pnp_controls=dict(available=sum(state in ('passed', 'failed') for state in oracle_states),
                                passed=sum(state == 'passed' for state in oracle_states),
                                failed=sum(state == 'failed' for state in oracle_states),
                                unavailable=sum(state == 'unavailable' for state in oracle_states),
                                required=8, all_passed=oracle_pass, rows=controls),
        native_rgb_near_view_controls=native_positive_controls,
        synthetic_near_view_patch_calibration=patch_calibration,
        gray_query_native_vs_gray_template=gray_query_comparison,
        validated_wrong_surface_false_acceptances=wrong_surface_negatives,
        real_patch_support=real_counts,
        real_template_appearance_comparison=real_appearance_comparison,
        real_support_vs_contract_score=support_vs_score,
        gray_template_real_support_gain_gate=dict(
            frames_with_at_least_four_anchor_gain=sorted(gain_frames),
            formerly_supported_gate_losses=formerly_supported_losses,
            expected_comparisons=expected_real_appearance_comparisons,
            comparisons=len(real_appearance_comparison),
            appearance_informative=real_appearance_informative,
            uninformative_comparisons=[
                dict(frame_id=row['frame_id'], template_offset_deg=row['template_offset_deg'],
                     state=row['template_appearance_ablation_state'])
                for row in real_appearance_comparison if not row['appearance_ablation_informative']],
            required_frames=3, passed=color_real_support_gate),
        appearance_informativeness={
            entry.get('context_id'): entry.get('appearance_ablation') or entry.get('query_appearance_ablation')
            for entry in capture.get('contexts', [])
            if entry.get('appearance_ablation') is not None or entry.get('query_appearance_ablation') is not None
        },
        synthetic_conditions=len(report['synthetic']), real_conditions=len(report['real']),
        frozen_gates=dict(base_invariants=base_invariants, projection_parity=projection_pass,
                          oracle_pnp=oracle_pass, native_rgb_near_view=native_positive_pass,
                          analytic_near_view_patch=patch_calibration_pass,
                          gray_query_template_improvement=gray_gap_pass,
                          real_template_support_gain=color_real_support_gate),
        permitted_conclusion=permitted_conclusion,
        conclusions=('Synthetic geometry labels describe only the fixed model and never establish real pose accuracy. '
                     'The real branch reports current-image texture support only.'),
        next_step='Require Sol review of this frozen diagnostic before any neural capture interpretation.',
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True,
                        help='Immutable full bottle-identity-v1 capture root')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Fresh CPU-only audit output directory')
    args = parser.parse_args(argv)
    report = audit_capture(args.capture_root, args.output_root)
    print(json.dumps(dict(status=report['status'], complete=report['complete'],
                          repeat_hashes_match=report['invariants']['repeat_hashes_match'],
                          projection_parity_passed=report['invariants']['projection_parity_passed'],
                          oracle_pnp_controls=report['decisions']['oracle_pnp_controls']), indent=2), flush=True)


if __name__ == '__main__':
    main()
