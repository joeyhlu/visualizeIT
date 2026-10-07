"""Capture the frozen 108-forward bottle surface-identity diagnostic.

This module is the only neural stage in R1. It makes one forward per listed
condition, stores shared immutable query/template contexts once, and preserves
every flow/confidence packet under a 128 MiB private packet budget. It reads no
evaluation references or annotations and does not change a tracking seed.
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
from .quality_bottle_identity_audit import (
    APPEARANCES, FORWARD_CAP, FRAME_IDS, PACKET_BUDGET_BYTES,
    QUERY_OFFSETS, SYNTHETIC_CARRIERS, TEMPLATE_OFFSETS,
    array_info, array_sha256, build_template_patch_bank,
    fixed_patch_bank_manifest, gray_rgb, native_to_crop_pose_mm,
    object_points_from_depth,
)
from .quality_contract import checked_pose
from .quality_runner import read_input, read_rgb
from .quality_texture_detail import axial_pose
from .vision import cv2


OBJECT = 'ranch'
OUTPUT_ROOT = CACHE / 'diagnostics' / 'bottle-identity-v1'
SOURCE_FILES = (
    'bench/quality_bottle_identity_probe.py',
    'bench/quality_bottle_identity_audit.py',
    'bench/quality_gotrack.py',
    'bench/quality_contract.py',
    'bench/quality_assets.py',
    'bench/quality_runner.py',
    'bench/quality_texture_detail.py',
    'bench/quality_render_stability.py',
    'bench/vision.py',
    '.cache/model-quality/sources/gotrack/utils/data_util.py',
    '.cache/model-quality/sources/gotrack/utils/crop_generation.py',
    '.cache/model-quality/sources/gotrack/utils/transform3d.py',
    '.cache/model-quality/sources/gotrack/utils/poser_util.py',
)


def _create_output_root(path):
    output_root = Path(path).expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f'Output root already exists; preserving it: {output_root}')
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root.resolve()


def _runtime_info(torch):
    return dict(
        device='cuda', batch_size=1, network_forwards_max=FORWARD_CAP,
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        opencv=cv2.__version__, torch=torch.__version__, cuda_runtime=torch.version.cuda,
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


def _snapshot_sources(output_root):
    sources = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f'Missing pinned bottle identity source: {relative}')
        target = output_root / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_sha = digest(source)
        if digest(target) != source_sha:
            raise IOError(f'Source snapshot hash mismatch: {relative}')
        try:
            target.chmod(0o444)
        except OSError:
            pass
        sources[relative] = dict(path=str(target.relative_to(output_root)), sha256=source_sha)
    return sources


def _capture_manifest_fields(manifest):
    """Copy the input fields needed by downstream preparation into the capture."""
    return dict(
        object_id=int(manifest['object_id']),
        input_intrinsics=np.asarray(manifest['intrinsics'], dtype=np.float64).tolist(),
        native_resolution=list(manifest['native_resolution']),
    )


def _write_packet(output_root, record, relative_path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    total = int(record['packet_bytes']) + len(payload)
    if total > PACKET_BUDGET_BYTES:
        failure = dict(path=str(relative_path.as_posix()), attempted_bytes=len(payload),
                       aggregate_bytes=total, reason='aggregate_128_mib_packet_budget_exceeded')
        record['packet_failures'].append(failure)
        save_result(output_root / 'capture.json', record)
        raise ValueError(f'Bottle identity packet aggregate exceeds {PACKET_BUDGET_BYTES} bytes')
    final_path = output_root / relative_path
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        raise FileExistsError(f'Refusing to replace immutable bottle identity packet: {final_path}')
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
        path=relative_path.as_posix(), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        arrays={name: array_info(value) for name, value in arrays.items()},
    )
    record['packet_bytes'] = total
    save_result(output_root / 'capture.json', record)
    return entry


def _write_context(output_root, record, context_id, role, frame_id, geometry_group,
                   arrays, **metadata):
    relative = Path('packets') / 'contexts' / f'{context_id}.npz'
    entry = _write_packet(output_root, record, relative, arrays)
    entry.update(context_id=context_id, role=role, frame_id=int(frame_id),
                 geometry_group=geometry_group, **metadata)
    record['contexts'].append(entry)
    save_result(output_root / 'capture.json', record)
    return entry


def _pose_crop_m(seed_pose_m, crop_from_native):
    pose_mm = native_to_crop_pose_mm(seed_pose_m, crop_from_native)
    pose_m = pose_mm.copy()
    pose_m[:3, 3] *= .001
    return pose_m


def _object_axis_and_center(vertices_m):
    vertices = np.asarray(vertices_m, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * .5
    _, _, vt = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
    axis = vt[0]
    axis *= 1. if axis[np.argmax(np.abs(axis))] >= 0 else -1.
    return axis, center


def _make_camera(structs, width, height, intrinsics, camera_to_world):
    k = np.asarray(intrinsics, dtype=np.float64)
    return structs.PinholePlaneCameraModel(
        width=int(width), height=int(height), f=(float(k[0, 0]), float(k[1, 1])),
        c=(float(k[0, 2]), float(k[1, 2])),
        T_world_from_eye=np.asarray(camera_to_world, dtype=np.float64),
    )


def _render_pose(renderer, structs, renderer_base, object_id, pose_camera_from_object_mm,
                 intrinsics, width, height):
    camera_from_object_mm = np.asarray(pose_camera_from_object_mm, dtype=np.float64)
    camera_to_object_mm = np.linalg.inv(camera_from_object_mm)
    camera = _make_camera(structs, width, height, intrinsics, camera_to_object_mm)
    rendered = renderer.render_object_model(object_id, camera)
    color = np.asarray(rendered[renderer_base.RenderType.COLOR], dtype=np.float32)
    depth = np.asarray(rendered[renderer_base.RenderType.DEPTH], dtype=np.float32)
    mask = (np.asarray(rendered[renderer_base.RenderType.MASK]) > 0) & np.isfinite(depth) & (depth > 0)
    return color, depth, mask


def _torch_rgb(torch, rgb, device):
    value = np.asarray(rgb, dtype=np.float32)
    if value.max(initial=0.) > 1.5:
        value = value / 255.
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))[None]).to(device)


def _torch_mask(torch, mask, device):
    value = np.asarray(mask, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(value[None])).to(device)


def _planned_conditions():
    conditions = []
    for frame_id in SYNTHETIC_CARRIERS:
        for query_offset in QUERY_OFFSETS:
            for query_appearance in APPEARANCES:
                for template_offset in TEMPLATE_OFFSETS:
                    for template_appearance in APPEARANCES:
                        seed = frame_id * 17 + (query_offset // 180) * 4 + (template_offset // 180) * 2 + APPEARANCES.index(template_appearance)
                        conditions.append(dict(
                            condition_id=f'syn-{frame_id}-q{query_offset}-{query_appearance}-t{template_offset}-{template_appearance}',
                            kind='synthetic', frame_id=frame_id, query_offset_deg=query_offset,
                            query_appearance=query_appearance, template_offset_deg=template_offset,
                            template_appearance=template_appearance, rng_seed=int(seed & 0x7fffffff),
                            state='pending',
                        ))
    real_for_frame = {}
    for frame_id in FRAME_IDS:
        real_for_frame[frame_id] = []
        for template_offset in TEMPLATE_OFFSETS:
            for template_appearance in APPEARANCES:
                seed = frame_id * 17 + (template_offset // 180) * 2 + APPEARANCES.index(template_appearance)
                condition = dict(
                    condition_id=f'real-{frame_id}-t{template_offset}-{template_appearance}',
                    kind='real', frame_id=frame_id, query_appearance='rgb',
                    template_offset_deg=template_offset, template_appearance=template_appearance,
                    rng_seed=int(seed & 0x7fffffff), state='pending',
                )
                real_for_frame[frame_id].append(condition)
    conditions.extend(real_for_frame[10])
    for original in real_for_frame[10]:
        conditions.append(dict(
            condition_id=f'repeat-{original["condition_id"]}', kind='repeat', frame_id=10,
            query_appearance='rgb', template_offset_deg=original['template_offset_deg'],
            template_appearance=original['template_appearance'], rng_seed=original['rng_seed'],
            repeat_of=original['condition_id'], state='pending',
        ))
    for frame_id in FRAME_IDS:
        if frame_id != 10:
            conditions.extend(real_for_frame[frame_id])
    if len(conditions) != FORWARD_CAP:
        raise AssertionError(f'Frozen R1 condition plan changed: {len(conditions)} != {FORWARD_CAP}')
    for index, condition in enumerate(conditions):
        condition['forward_index'] = index
    return conditions


def _safe_mask_path(mask_root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe cached observed mask path: {relative}')
    root = mask_root.resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Cached observed mask escaped its root: {relative}')
    return result


def _prepare_frame(frame_id, seed_record, bundle, manifest, mask_root,
                   renderer, data_util, misc, structs, renderer_base, torch,
                   output_root, capture, axis, center, object_id):
    seed_value = seed_record.get('cameraFromObject')
    if seed_value is None:
        return None, None, 'automatic_seed_unavailable'
    try:
        rgb = read_rgb(bundle, manifest, frame_id)
    except Exception as error:
        return None, None, f'source_image_unavailable: {type(error).__name__}: {error}'
    width, height = map(int, manifest['native_resolution'])
    if rgb.shape[:2] != (height, width):
        return None, None, f'source_image_shape_mismatch: got {list(rgb.shape[:2])}, expected {[height, width]}'
    relative = seed_record.get('mask_path')
    if not relative:
        return None, None, 'observed_mask_path_unavailable'
    mask_path = _safe_mask_path(mask_root, relative)
    if not mask_path.is_file():
        return None, None, 'observed_mask_file_unavailable'
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != rgb.shape[:2]:
        return None, None, 'observed_mask_decode_or_shape_unavailable'
    mask_sha = digest(mask_path)
    expected_mask_sha = seed_record.get('mask_sha256')
    if expected_mask_sha and mask_sha != expected_mask_sha:
        raise ValueError(f'Cached observed mask hash changed for frame {frame_id}')
    seed_pose_m = checked_pose(seed_value)
    native_k = np.asarray(manifest['intrinsics'], dtype=np.float64)
    native_camera = structs.PinholePlaneCameraModel(
        width=width, height=height, f=(native_k[0, 0], native_k[1, 1]),
        c=(native_k[0, 2], native_k[1, 2]), T_world_from_eye=np.eye(4),
    )
    seed_pose_mm = seed_pose_m.copy(); seed_pose_mm[:3, 3] *= 1000.
    image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None] / 255.
    mask_tensor = torch.from_numpy((mask > 0).astype(np.float32))[None]
    with torch.inference_mode():
        data, crop_cameras, crop_from_orig = data_util.compute_gotrack_inputs_from_init_poses(
            input_rgbs=image_tensor, input_cameras=[native_camera],
            init_poses_cam_from_model=torch.from_numpy(seed_pose_mm.astype(np.float32))[None],
            renderer=renderer, obj_ids=[object_id], object_vertices=[renderer.vertices_m * 1000.],
            crop_size=(280, 280), crop_rel_pad=.1, cropping_type='perspective_2d_box',
            ssaa_factor=1., background_type='gray', input_masks=mask_tensor,
        )
    crop_rgb = data['crop_rgbs'][0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    crop_mask = data['crop_masks'][0].detach().cpu().numpy() > .5
    crop_k = misc.get_intrinsic_matrix(crop_cameras[0])
    if hasattr(crop_k, 'detach'):
        crop_k = crop_k.detach().cpu().numpy()
    crop_k = np.asarray(crop_k, dtype=np.float64)
    crop_from_native = crop_from_orig[0].detach().cpu().numpy().astype(np.float64)
    templates = data['templates']
    seed_color = templates.rgbs[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    seed_depth = templates.depths[0].detach().cpu().numpy().astype(np.float32)
    seed_template_mask = templates.masks[0].detach().cpu().numpy() > 0

    context_id = f'real-frame-{frame_id:04d}'
    arrays = dict(
        native_rgb=rgb.astype(np.uint8), native_mask=(mask > 0).astype(np.uint8),
        crop_rgb=crop_rgb, observed_crop_mask=crop_mask.astype(np.uint8),
        native_k=native_k, crop_k=crop_k, crop_from_native=crop_from_native,
        seed_pose_m=seed_pose_m,
    )
    entry = _write_context(
        output_root, capture, context_id, 'real_frame', frame_id, f'frame-{frame_id}', arrays,
        native_rgb_sha256=array_sha256(rgb), native_mask_sha256=array_sha256(mask > 0),
        observed_mask_file_sha256=mask_sha, seed_pose_sha256=array_sha256(seed_pose_m),
        input_rgb_sha256=array_sha256(rgb), reference_or_annotation_inputs=False,
    )
    real = dict(entry=entry, arrays=arrays, mask_path=str(mask_path), mask_sha256=mask_sha)

    for offset in TEMPLATE_OFFSETS:
        template_pose_m = axial_pose(seed_pose_m, axis, center, offset)
        if offset == 0:
            color, depth, template_mask = seed_color, seed_depth, seed_template_mask
        else:
            pose_crop_mm = native_to_crop_pose_mm(template_pose_m, crop_from_native)
            color, depth, template_mask = _render_pose(
                renderer, structs, renderer_base, object_id, pose_crop_mm,
                crop_k, 280, 280)
        template_mask = np.asarray(template_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
        gray_color, appearance = gray_rgb(color, template_mask)
        pose_crop_mm = native_to_crop_pose_mm(template_pose_m, crop_from_native)
        source_indices, source_pixels, source_points_m = object_points_from_depth(
            depth, crop_k, pose_crop_mm, template_mask)
        template_id = f'template-{frame_id:04d}-{offset:03d}'
        template_arrays = dict(
            template_rgb=np.asarray(color, dtype=np.float32), template_gray_rgb=gray_color.astype(np.float32),
            template_depth_mm=np.asarray(depth, dtype=np.float32), template_mask=template_mask.astype(np.uint8),
            source_indices=source_indices, source_pixels_xy=source_pixels,
            source_points_object_m=source_points_m,
            observed_crop_mask=crop_mask.astype(np.uint8), crop_k=crop_k,
            crop_from_native=crop_from_native, native_k=native_k,
            seed_pose_m=seed_pose_m, template_pose_m=template_pose_m,
        )
        template_entry = _write_context(
            output_root, capture, template_id, 'template', frame_id, f'frame-{frame_id}', template_arrays,
            template_offset_deg=offset, appearance_ablation=appearance,
            geometry_hashes=dict(depth_mm=array_sha256(depth), mask=array_sha256(template_mask)),
            object_source_points=int(len(source_indices)), reference_or_annotation_inputs=False,
        )
        fixed_bank = build_template_patch_bank(
            template_arrays['template_rgb'], template_arrays['template_mask'],
            template_arrays['source_indices'], template_arrays['source_points_object_m'])
        template_entry['fixed_patch_bank'] = fixed_patch_bank_manifest(fixed_bank, template_arrays)
        save_result(output_root / 'capture.json', capture)
        real.setdefault('templates', {})[offset] = dict(entry=template_entry, arrays=template_arrays)

    return real, seed_pose_m, None


def _prepare_synthetic_query(frame_id, query_offset, frame_context, seed_pose_m,
                             axis, center, renderer, renderer_base, structs, output_root,
                             capture, observed_mask_native):
    arrays = frame_context['arrays']
    native_k = arrays['native_k']; crop_k = arrays['crop_k']
    crop_from_native = arrays['crop_from_native']
    crop_mask = arrays['observed_crop_mask'].astype(bool)
    query_pose_native_m = axial_pose(seed_pose_m, axis, center, query_offset)
    query_pose_crop_mm = native_to_crop_pose_mm(query_pose_native_m, crop_from_native)
    query_pose_crop_m = query_pose_crop_mm.copy(); query_pose_crop_m[:3, 3] *= .001
    crop_rgb, crop_depth_mm, crop_render_mask = _render_pose(
        renderer, structs, renderer_base, int(capture['object_id']), query_pose_crop_mm,
        crop_k, 280, 280)
    synthetic_crop_mask = crop_mask & crop_render_mask
    crop_rgb = np.asarray(crop_rgb, dtype=np.float32).copy()
    crop_rgb[~synthetic_crop_mask] = .5
    crop_gray, query_appearance_info = gray_rgb(crop_rgb, synthetic_crop_mask)
    native_pose_mm = query_pose_native_m.copy(); native_pose_mm[:3, 3] *= 1000.
    native_h, native_w = observed_mask_native.shape
    _, _, native_render_mask = _render_pose(
        renderer, structs, renderer_base, int(capture['object_id']), native_pose_mm,
        native_k, native_w, native_h)
    synthetic_native_mask = (np.asarray(native_render_mask, dtype=bool) &
                             np.asarray(observed_mask_native, dtype=bool))
    context_id = f'synthetic-query-{frame_id:04d}-{query_offset:03d}'
    context_arrays = dict(
        query_rgb_rgb=crop_rgb.astype(np.float32), query_rgb_gray=crop_gray.astype(np.float32),
        query_depth_mm=np.asarray(crop_depth_mm, dtype=np.float32),
        query_render_mask=np.asarray(crop_render_mask, dtype=np.uint8),
        synthetic_query_mask=synthetic_crop_mask.astype(np.uint8),
        synthetic_native_mask=synthetic_native_mask.astype(np.uint8),
        observed_crop_mask=crop_mask.astype(np.uint8), crop_k=crop_k,
        crop_from_native=crop_from_native, native_k=native_k,
        query_pose_native_m=query_pose_native_m, known_query_pose_m=query_pose_crop_m,
    )
    entry = _write_context(
        output_root, capture, context_id, 'synthetic_query', frame_id, f'frame-{frame_id}', context_arrays,
        query_offset_deg=int(query_offset), query_appearance_ablation=query_appearance_info,
        geometry_hashes=dict(depth_mm=array_sha256(crop_depth_mm),
                             render_mask=array_sha256(crop_render_mask),
                             synthetic_mask=array_sha256(synthetic_crop_mask)),
        object_camera_carrier_only=True, synthetic_geometry_oracle=True,
        real_query_values_used_only_as_camera_crop_and_occlusion=False,
    )
    # The native synthetic mask is separately hashed for contract validation.
    entry['synthetic_native_mask_sha256'] = array_sha256(synthetic_native_mask)
    save_result(output_root / 'capture.json', capture)
    return dict(entry=entry, arrays=context_arrays)


def _forward_one(condition, query_context, template_context, network, torch,
                 output_root, capture, device):
    if int(capture['forward_calls']) >= FORWARD_CAP:
        raise ValueError('Frozen 108-forward budget reached before this condition')
    qarrays = query_context['arrays']; tarr = template_context['arrays']
    query_rgb = qarrays['crop_rgb'] if condition['kind'] in ('real', 'repeat') else qarrays[
        'query_rgb_' + condition['query_appearance']]
    template_rgb = tarr['template_' + ('gray_rgb' if condition['template_appearance'] == 'gray' else 'rgb')]
    template_mask = tarr['template_mask'] > 0
    query_tensor = _torch_rgb(torch, query_rgb, device)
    template_tensor = _torch_rgb(torch, template_rgb, device)
    mask_tensor = _torch_mask(torch, template_mask, device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        flow_tensor, confidence_tensor = network(query_tensor, template_tensor, mask_tensor)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.
    capture['forward_calls'] = int(capture['forward_calls']) + 1
    flow = flow_tensor[0].detach().float().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    confidence = confidence_tensor[0].detach().float().cpu().numpy().astype(np.float32)
    if flow.shape != (280, 280, 2) or confidence.shape != (280, 280):
        raise ValueError(f'Unexpected GoTrack output shapes: {flow.shape}, {confidence.shape}')
    arrays = dict(flow=flow, confidence=confidence)
    relative = Path('packets') / 'forwards' / f'{condition["condition_id"]}.npz'
    entry = _write_packet(output_root, capture, relative, arrays)
    ids = np.asarray(tarr['source_indices'], dtype=np.int64)
    source_xy = np.asarray(tarr['source_pixels_xy'], dtype=np.float64)
    source_points = np.asarray(tarr['source_points_object_m'], dtype=np.float64)
    yy, xx = np.divmod(ids, flow.shape[1])
    endpoints = source_xy + flow[yy, xx]
    target_x = np.full(len(ids), -1, dtype=np.int32)
    target_y = np.full(len(ids), -1, dtype=np.int32)
    finite = np.isfinite(endpoints).all(axis=1)
    target_x[finite] = np.floor(endpoints[finite, 0]).astype(np.int32)
    target_y[finite] = np.floor(endpoints[finite, 1]).astype(np.int32)
    inside = finite & (target_x >= 0) & (target_x < flow.shape[1]) & (target_y >= 0) & (target_y < flow.shape[0])
    target_indices = np.full((len(ids), 2), -1, dtype=np.int32)
    target_indices[inside] = np.column_stack((target_x[inside], target_y[inside]))
    derived = dict(
        source_indices=array_info(ids), source_pixels_xy=array_info(source_xy),
        source_points_object_m=array_info(source_points), target_endpoints_px=array_info(endpoints),
        target_indices_xy=array_info(target_indices), source_confidence=array_info(confidence[yy, xx]),
    )
    output_hashes = dict(flow=array_sha256(flow), confidence=array_sha256(confidence))
    entry.update(
        condition_id=condition['condition_id'], rng_seed=int(condition['rng_seed']),
        output_hashes=output_hashes, derived_array_hashes=derived,
        elapsed_ms=float(elapsed_ms), context_refs=condition['context_refs'],
        forward_index=int(condition['forward_index']),
    )
    capture['forwards'].append(entry)
    condition.update(state='captured', packet_path=entry['path'], packet_sha256=entry['sha256'],
                     output_hashes=output_hashes, elapsed_ms=float(elapsed_ms))
    save_result(output_root / 'capture.json', capture)
    return entry


def _unavailable_frame_conditions(conditions, frame_id, reason):
    for condition in conditions:
        if int(condition['frame_id']) == int(frame_id) and condition['state'] == 'pending':
            condition.update(state='unavailable', reason=reason)


def run_capture(output_root=OUTPUT_ROOT):
    """Run one frozen GPU capture; called only by the parent-scheduled stage."""
    output_root = _create_output_root(output_root)
    conditions = _planned_conditions()
    record = dict(
        schema_version=1, object=OBJECT, scope=__doc__, status='initializing', complete=False,
        requested_real_frame_ids=list(FRAME_IDS), requested_synthetic_carriers=list(SYNTHETIC_CARRIERS),
        requested_query_offsets_deg=list(QUERY_OFFSETS), requested_template_offsets_deg=list(TEMPLATE_OFFSETS),
        requested_template_appearances=list(APPEARANCES), synthetic_query_appearances=list(APPEARANCES),
        crop=dict(size=[280, 280], relative_padding=.1, type='perspective_2d_box', ssaa=1.,
                  camera_and_transform_fixed_from_saved_seed=True),
        renderer=dict(unlit=True, disable_multisampling=True),
        inference=dict(direction='template-to-query forward', batch_size=1,
                       no_tracking_iterations=True, no_seed_selection=True),
        planned_forward_cap=FORWARD_CAP, forward_calls=0,
        packet_budget_bytes=PACKET_BUDGET_BYTES, packet_bytes=0,
        reference_or_annotations_loaded=False, source_snapshot={}, contexts=[], forwards=[],
        packet_failures=[], failures=[], conditions=conditions,
        output_root=str(output_root), started_unix=time.time(),
    )
    save_result(output_root / 'capture.json', record)
    renderer = None
    capture_error = None
    try:
        bundle = CACHE / 'inputs' / OBJECT
        manifest = read_input(bundle)
        seed_path = CACHE / 'results' / OBJECT / 'render-stability-appearance' / 'complete.json'
        seed_result = json.loads(seed_path.read_text(encoding='utf-8'))
        if (seed_result.get('object') != OBJECT or seed_result.get('mode') != 'complete' or
                not seed_result.get('complete') or not seed_result.get('automatic') or
                seed_result.get('diagnostic_control')):
            raise ValueError('Expected the preserved automatic render-stability-appearance bottle seed result')
        seed_records = {int(row['frameId']): row for row in seed_result.get('frames', [])}
        if not set(FRAME_IDS).issubset(seed_records):
            raise ValueError('Saved automatic bottle result does not contain all fixed R1 frames')
        record.update(
            source_snapshot=_snapshot_sources(output_root),
            input_manifest_sha256=digest(bundle / 'input.json'),
            input_source_hashes=manifest.get('source_hashes', {}),
            asset_sha256=digest(bundle / manifest['asset']),
            source_video_sha256=digest(bundle / manifest['video']),
            seed_result_path=str(seed_path), seed_result_sha256=digest(seed_path),
            seed_result_provenance=seed_result.get('provenance'),
            seed_tracking_settings=seed_result.get('tracking_settings'),
            inference_provenance=inference_provenance(bundle),
            model_checkpoint_sha256=digest(CACHE / 'checkpoints' / 'gotrack_checkpoint.pt'),
            foundpose_bank_sha256=(digest(CACHE / 'banks' / 'ranch-foundpose.pt')
                                   if (CACHE / 'banks' / 'ranch-foundpose.pt').is_file() else None),
            foundpose_bank_used=False,
            **_capture_manifest_fields(manifest),
            selection='Fixed exploratory frame list from R1 appendix; not held out and not a benchmark.',
            seed_scope='Saved automatic poses are camera carriers only, not independent truth.',
        )
        save_result(output_root / 'capture.json', record)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        from .quality_gotrack import TexturedRenderer, load_network, upstream_path
        upstream_path()
        from utils import data_util, misc, renderer_base, structs
        _set_deterministic_runtime(torch)
        network = load_network('cuda')
        renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'],
                                    unlit=True, disable_multisampling=True)
        record['runtime'] = _runtime_info(torch)
        vertices = renderer.vertices_m
        axis, center = _object_axis_and_center(vertices)
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        record.update(
            object_bbox_diagonal_m=diagonal, object_axis_longest_svd=axis.tolist(),
            object_bbox_center_m=center.tolist(), status='running',
        )
        save_result(output_root / 'capture.json', record)
        mask_root = CACHE / 'results' / OBJECT / 'segmentation'
        frame_contexts = {}
        frame_states = {}
        for frame_id in FRAME_IDS:
            try:
                frame_context, seed_pose, reason = _prepare_frame(
                    frame_id, seed_records[frame_id], bundle, manifest, mask_root,
                    renderer, data_util, misc, structs, renderer_base, torch,
                    output_root, record, axis, center, manifest['object_id'])
            except Exception as error:
                raise RuntimeError(f'Failed fixed crop/context preparation for frame {frame_id}: {error}') from error
            if reason:
                frame_states[frame_id] = dict(state='unavailable', reason=reason)
                _unavailable_frame_conditions(conditions, frame_id, reason)
                continue
            frame_states[frame_id] = dict(state='available', reason=None)
            frame_contexts[frame_id] = frame_context

        # Build synthetic query contexts from the same crop/K/C fixed by each saved seed.
        synthetic_contexts = {}
        for frame_id in SYNTHETIC_CARRIERS:
            if frame_id not in frame_contexts:
                continue
            context = frame_contexts[frame_id]
            for query_offset in QUERY_OFFSETS:
                synthetic_contexts[(frame_id, query_offset)] = _prepare_synthetic_query(
                    frame_id, query_offset, context, context['arrays']['seed_pose_m'],
                    axis, center, renderer, renderer_base, structs, output_root,
                    record, context['arrays']['native_mask'])

        context_ids = {entry['context_id'] for entry in record['contexts']}
        forward_by_id = {entry['condition_id']: entry for entry in record['forwards']}
        for condition in conditions:
            frame_id = int(condition['frame_id'])
            if frame_id not in frame_contexts:
                if condition['state'] == 'pending':
                    condition.update(state='unavailable', reason=frame_states[frame_id]['reason'])
                continue
            frame_context = frame_contexts[frame_id]
            template_offset = int(condition['template_offset_deg'])
            template_context = frame_context['templates'][template_offset]
            if condition['kind'] == 'synthetic':
                query_context = synthetic_contexts[(frame_id, int(condition['query_offset_deg']))]
            else:
                query_context = frame_context
            condition['context_refs'] = dict(
                query=query_context['entry']['context_id'],
                template=template_context['entry']['context_id'],
                observed_frame=frame_context['entry']['context_id'],
            )
            missing_ref = set(condition['context_refs'].values()) - context_ids
            if missing_ref:
                raise ValueError(f'Missing immutable context references for {condition["condition_id"]}: {sorted(missing_ref)}')
            if condition['kind'] == 'repeat':
                original = forward_by_id.get(condition['repeat_of'])
                if original is None:
                    condition.update(state='unavailable', reason='frame_10_original_condition_unavailable')
                    continue
            try:
                entry = _forward_one(
                    condition, query_context, template_context, network, torch,
                    output_root, record, 'cuda')
                if condition['kind'] == 'repeat':
                    original = forward_by_id.get(condition['repeat_of'])
                    matched = original is not None and entry['output_hashes'] == original['output_hashes']
                    condition['repeat_hash_match'] = bool(matched)
                    capture_check = dict(
                        condition_id=condition['condition_id'], repeat_of=condition['repeat_of'],
                        matched=bool(matched), original_hashes=None if original is None else original['output_hashes'],
                        repeat_hashes=entry['output_hashes'],
                    )
                    record.setdefault('repeat_checks', []).append(capture_check)
                    save_result(output_root / 'capture.json', record)
                forward_by_id[condition['condition_id']] = entry
            except Exception as error:
                condition.update(state='failed', reason=f'{type(error).__name__}: {error}')
                record['failures'].append(dict(condition_id=condition['condition_id'],
                                               type=type(error).__name__, message=str(error)))
                save_result(output_root / 'capture.json', record)
                raise
        pending = [row['condition_id'] for row in conditions if row['state'] == 'pending']
        if pending:
            raise ValueError(f'Capture did not explicitly account for all frozen conditions: {pending[:5]}')
        record['conditions'] = conditions
        record['frame_availability'] = [dict(frame_id=key, **value) for key, value in frame_states.items()]
        record['all_108_conditions_accounted'] = len(conditions) == FORWARD_CAP
        record['all_available_forwards_captured'] = all(
            row['state'] in ('captured', 'unavailable') for row in conditions)
        record['repeat_hashes_match'] = bool(len(record.get('repeat_checks', [])) == 4 and
                                             all(row['matched'] for row in record['repeat_checks']))
        record['complete'] = bool(record['all_108_conditions_accounted'] and
                                  record['all_available_forwards_captured'] and
                                  record['forward_calls'] <= FORWARD_CAP)
        record['status'] = 'complete' if record['complete'] else 'incomplete'
        record['completed_unix'] = time.time()
        record['independent_real_accuracy_verified'] = False
        record['real_pose_accuracy_claimed'] = False
        save_result(output_root / 'capture.json', record)
    except BaseException as error:
        capture_error = error
        record['status'] = 'failed'
        record['complete'] = False
        record['failure'] = dict(type=type(error).__name__, message=str(error))
        record['failed_unix'] = time.time()
        try:
            save_result(output_root / 'capture.json', record)
        except Exception:
            pass
    finally:
        if renderer is not None:
            renderer.close()
    if capture_error is not None:
        raise capture_error
    print(json.dumps(dict(
        status=record['status'], complete=record['complete'],
        planned_forwards=FORWARD_CAP, actual_forwards=record['forward_calls'],
        packet_bytes=record['packet_bytes'], packet_budget_bytes=PACKET_BUDGET_BYTES,
        repeat_hashes_match=record.get('repeat_hashes_match'),
        output_root=str(output_root),
    ), indent=2), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT,
                        help='Fresh immutable private capture directory; defaults to bottle-identity-v1')
    args = parser.parse_args(argv)
    run_capture(args.output_root)


if __name__ == '__main__':
    main()
