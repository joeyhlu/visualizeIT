"""Capture the frozen six-forward bottle zero-view calibration.

The probe reuses immutable R1 v2 crop/template packets. Its only new neural
inputs are self/full and self/clipped RGB queries for carriers 10, 50 and 100;
known synthetic pose/depth are read only after inference for CPU scoring.
"""
import argparse
import gc
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_identity_probe as capture_probe
from .quality_assets import CACHE, MODELS, ROOT, digest, save_result
from .vision import cv2


CAPTURE_ROOT = CACHE / 'diagnostics' / 'bottle-identity-v2'
OUTPUT_ROOT = CACHE / 'diagnostics' / 'bottle-zero-view-calibration-v1'
EXPECTED_CAPTURE_SHA256 = '33EFA32778C4094A5DF0690D49658CB3BFB0BE522204B3DC4E1279D9F0D03063'
CARRIERS = (10, 50, 100)
CONDITIONS = ('full', 'clipped')
FORWARD_CAP = 6
PACKET_BUDGET_BYTES = 16 * 1024**2


def _create_output_root(path):
    root = Path(path).expanduser()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'Output root already exists; preserving it: {root}')
    root.mkdir(parents=True, exist_ok=False)
    return root.resolve()


def _planned_conditions():
    rows = []
    for frame_id in CARRIERS:
        for condition in CONDITIONS:
            rows.append(dict(
                condition_id=f'zero-{frame_id}-{condition}',
                frame_id=int(frame_id), template_offset_deg=0,
                condition=condition, appearance='rgb', state='pending',
                forward_index=len(rows),
            ))
    if len(rows) != FORWARD_CAP:
        raise AssertionError('Frozen zero-view plan must contain exactly six forwards')
    return rows


def _safe_relative(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe linked capture path: {relative}')
    target = (Path(root).resolve() / path).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Linked capture path escaped its root: {relative}')
    return target


def _canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _source_pins(capture_root, capture, review_record=None):
    saved = capture.get('source_snapshot')
    if not isinstance(saved, dict):
        raise ValueError('R1 capture has no frozen source snapshot')
    for relative, entry in saved.items():
        if not isinstance(entry, dict) or not entry.get('path') or not entry.get('sha256'):
            raise ValueError(f'Invalid source snapshot entry: {relative}')
        if digest(_safe_relative(capture_root, entry['path'])) != entry['sha256']:
            raise ValueError(f'Linked R1 source snapshot hash changed: {relative}')
    missing = set(capture_probe.SOURCE_FILES) - set(saved)
    if missing:
        raise ValueError(f'R1 capture source snapshot is missing runtime files: {sorted(missing)}')
    captured_runtime = {relative: saved[relative]['sha256'] for relative in capture_probe.SOURCE_FILES}
    current_runtime = {}
    for relative in capture_probe.SOURCE_FILES:
        current = ROOT / relative
        if not current.is_file():
            raise FileNotFoundError(f'Current frozen inference source is missing: {relative}')
        current_runtime[relative] = digest(current)
    changed = {relative: dict(captured_sha256=captured_runtime[relative],
                              current_sha256=current_runtime[relative])
               for relative in capture_probe.SOURCE_FILES
               if captured_runtime[relative] != current_runtime[relative]}
    review_info = None
    if changed and review_record is None:
        raise ValueError(
            'Current inference sources differ from R1 (' + ', '.join(sorted(changed)) +
            '); pass an independent accepted Sol source-review record')
    if review_record is not None:
        review_path = Path(review_record).expanduser().resolve()
        review = json.loads(review_path.read_text(encoding='utf-8'))
        expected_delta_sha = _canonical_json_sha256(changed)
        if (review.get('schema_version') != 1 or review.get('reviewer') != 'Sol' or
                review.get('decision') != 'accepted' or
                review.get('scope') != 'bottle-zero-view-inference-source-delta' or
                not review.get('review_id') or
                review.get('capture_sha256') != digest(capture_root / 'capture.json') or
                review.get('captured_source_hashes') != captured_runtime or
                review.get('current_source_hashes') != current_runtime or
                review.get('inference_path_equivalence') is not True or
                review.get('approved_delta_sha256') != expected_delta_sha):
            raise ValueError('Independent Sol source-review record does not bind this exact source delta')
        review_info = dict(
            path=str(review_path), sha256=digest(review_path), reviewer=review['reviewer'],
            decision=review['decision'], review_id=review['review_id'],
            approved_delta_sha256=expected_delta_sha, changed_sources=changed,
        )
    expected_checkpoint = MODELS['gotrack_checkpoint.pt']['sha256']
    checkpoint_sha = capture.get('model_checkpoint_sha256')
    if checkpoint_sha != expected_checkpoint:
        raise ValueError('R1 capture checkpoint pin differs from the reviewed GoTrack checkpoint')
    current_files = (
        'bench/quality_bottle_identity_probe.py',
        'bench/quality_bottle_identity_audit.py',
        'bench/quality_gotrack.py',
        'bench/quality_assets.py',
        'bench/quality_bottle_zero_view_probe.py',
    )
    return dict(
        capture_source_snapshot={key: value['sha256'] for key, value in saved.items()},
        current_sources={key: digest(ROOT / key) for key in current_files},
        runtime_source_snapshot_sha256=_canonical_json_sha256(captured_runtime),
        runtime_current_source_sha256=_canonical_json_sha256(current_runtime),
        reviewed_source_deltas=review_info,
        checkpoint=dict(
            sha256=checkpoint_sha,
            upstream_revision='68f76055755f2a4a8967e13ece834f975f008bdf',
            actual_file_checked=False,
        ),
    )


def _entry_by_role(entries, role, frame_id, offset=None):
    matches = [entry for entry in entries
               if entry.get('role') == role and int(entry.get('frame_id', -1)) == int(frame_id)
               and (offset is None or int(entry.get('template_offset_deg', -1)) == int(offset))]
    if len(matches) != 1:
        raise ValueError(f'Expected one {role} context for frame {frame_id}, offset {offset}; got {len(matches)}')
    return matches[0]


def _validate_template(entry, arrays):
    needed = {
        'template_rgb', 'template_depth_mm', 'template_mask', 'source_indices',
        'source_pixels_xy', 'source_points_object_m', 'observed_crop_mask',
        'crop_k', 'crop_from_native', 'native_k', 'template_pose_m',
    }
    if not needed.issubset(arrays):
        raise ValueError(f"Template context lacks fields: {sorted(needed - set(arrays))}")
    rgb = np.asarray(arrays['template_rgb'])
    depth = np.asarray(arrays['template_depth_mm'])
    mask = np.asarray(arrays['template_mask'], dtype=bool)
    observed = np.asarray(arrays['observed_crop_mask'], dtype=bool)
    if rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Invalid captured RGB template: {entry['context_id']}")
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape or mask.shape != depth.shape or observed.shape != depth.shape:
        raise ValueError(f"Template/depth/observed-mask shapes differ: {entry['context_id']}")
    if not np.array_equal(mask, np.isfinite(depth) & (depth > 0)):
        raise ValueError(f"Template depth/mask disagreement: {entry['context_id']}")
    if not np.isfinite(rgb).all() or np.any((rgb < 0.) | (rgb > 1.)):
        raise ValueError(f"Captured GoTrack RGB is outside float [0,1]: {entry['context_id']}")
    if (np.asarray(arrays['crop_k']).shape != (3, 3) or
            np.asarray(arrays['crop_from_native']).shape != (4, 4) or
            np.asarray(arrays['native_k']).shape != (3, 3)):
        raise ValueError(f"Invalid camera calibration in {entry['context_id']}")
    ids = np.asarray(arrays['source_indices']).reshape(-1)
    pixels = np.asarray(arrays['source_pixels_xy']).reshape(-1, 2)
    points = np.asarray(arrays['source_points_object_m']).reshape(-1, 3)
    if not (len(ids) == len(pixels) == len(points)):
        raise ValueError(f"Source correspondence dimensions differ: {entry['context_id']}")
    geometry_hashes = entry.get('geometry_hashes', {})
    if (geometry_hashes.get('depth_mm') != audit.array_sha256(depth) or
            geometry_hashes.get('mask') != audit.array_sha256(mask)):
        raise ValueError(f"Template geometry hash differs from immutable context: {entry['context_id']}")


def _validate_plus8_context(query_entry, query_arrays, template_arrays):
    needed = {'query_rgb_rgb', 'query_depth_mm', 'synthetic_query_mask',
              'known_query_pose_m', 'crop_k', 'crop_from_native', 'native_k',
              'observed_crop_mask', 'query_render_mask'}
    if query_entry.get('role') != 'synthetic_query' or not needed.issubset(query_arrays):
        raise ValueError('Stored +8 RGB query lacks frozen synthetic scoring fields')
    rgb = np.asarray(query_arrays['query_rgb_rgb'])
    depth = np.asarray(query_arrays['query_depth_mm'])
    mask = np.asarray(query_arrays['synthetic_query_mask'], dtype=bool)
    observed = np.asarray(query_arrays['observed_crop_mask'], dtype=bool)
    rendered = np.asarray(query_arrays['query_render_mask'], dtype=bool)
    if (rgb.dtype != np.float32 or rgb.shape != np.asarray(template_arrays['template_rgb']).shape or
            depth.shape != mask.shape or depth.shape != rgb.shape[:2] or
            observed.shape != mask.shape or rendered.shape != mask.shape or
            np.any(mask & ~observed) or np.any(mask & ~rendered) or
            np.any(mask & ~(np.isfinite(depth) & (depth > 0))) or
            np.asarray(query_arrays['known_query_pose_m']).shape != (4, 4) or
            not np.array_equal(query_arrays['crop_k'], template_arrays['crop_k']) or
            not np.array_equal(query_arrays['crop_from_native'], template_arrays['crop_from_native']) or
            not np.array_equal(query_arrays['native_k'], template_arrays['native_k']) or
            not np.array_equal(observed, template_arrays['observed_crop_mask'])):
        raise ValueError('Stored +8 RGB query dimensions/calibration differ from its frozen template')
    hashes = query_entry.get('geometry_hashes', {})
    if (hashes.get('depth_mm') != audit.array_sha256(depth) or
            hashes.get('render_mask') != audit.array_sha256(rendered) or
            hashes.get('synthetic_mask') != audit.array_sha256(mask)):
        raise ValueError('Stored +8 RGB query geometry hashes differ from immutable context metadata')


def _load_inputs(capture_root, capture):
    entries = {entry['context_id']: entry for entry in capture['contexts']}
    forwards = {entry['condition_id']: entry for entry in capture['forwards']}
    condition_entries = {entry['condition_id']: entry for entry in capture['conditions']}
    prepared = {}
    for frame_id in CARRIERS:
        frame_entry = _entry_by_role(capture['contexts'], 'real_frame', frame_id)
        source_entry = _entry_by_role(capture['contexts'], 'template', frame_id, 0)
        competitor_entry = _entry_by_role(capture['contexts'], 'template', frame_id, 180)
        frame_arrays = audit._load_npz(capture_root, frame_entry)
        source_arrays = audit._load_npz(capture_root, source_entry)
        competitor_arrays = audit._load_npz(capture_root, competitor_entry)
        if frame_arrays.get('observed_crop_mask') is None:
            raise ValueError(f"Real frame lacks the captured crop mask: {frame_entry['context_id']}")
        if not np.array_equal(frame_arrays['observed_crop_mask'], source_arrays['observed_crop_mask']):
            raise ValueError(f"Real and template observed crop masks differ for frame {frame_id}")
        for field in ('crop_k', 'crop_from_native', 'native_k', 'observed_crop_mask'):
            if (field not in frame_arrays or field not in source_arrays or field not in competitor_arrays or
                    not np.array_equal(frame_arrays[field], source_arrays[field]) or
                    not np.array_equal(source_arrays[field], competitor_arrays[field])):
                raise ValueError(f'Fixed camera/mask field differs across frame/template contexts: {field}')
        _validate_template(source_entry, source_arrays)
        _validate_template(competitor_entry, competitor_arrays)
        if not np.allclose(source_arrays['template_pose_m'], frame_arrays['seed_pose_m'],
                           atol=1e-12, rtol=0):
            raise ValueError(f'Zero-offset template pose differs from saved seed for frame {frame_id}')
        source_bank = audit.verify_fixed_patch_bank(source_entry, source_arrays)
        competitor_bank = audit.verify_fixed_patch_bank(competitor_entry, competitor_arrays)
        if not source_entry.get('fixed_patch_bank') or not competitor_entry.get('fixed_patch_bank'):
            raise ValueError(f'Both frozen 0/180 patch banks are required for frame {frame_id}')
        observed = np.asarray(source_arrays['observed_crop_mask'], dtype=bool)
        template_mask = np.asarray(source_arrays['template_mask'], dtype=bool)
        common_mask = template_mask & observed
        if not common_mask.any():
            raise ValueError(f'Empty common self-query mask for frame {frame_id}')
        full_rgb = source_arrays['template_rgb']
        clipped_rgb = np.array(full_rgb, dtype=np.float32, copy=True)
        clipped_rgb[~common_mask] = np.float32(.5)
        if not np.array_equal(full_rgb[common_mask], clipped_rgb[common_mask]):
            raise AssertionError('Clipped query changed pixels inside the frozen common mask')
        prepared[frame_id] = dict(
            frame_entry=frame_entry, frame_arrays=frame_arrays,
            source_entry=source_entry, source_arrays=source_arrays,
            competitor_entry=competitor_entry, competitor_arrays=competitor_arrays,
            source_bank=source_bank, competitor_bank=competitor_bank,
            common_mask=common_mask, full_rgb=full_rgb, clipped_rgb=clipped_rgb,
        )

        plus8_id = f'syn-{frame_id}-q8-rgb-t0-rgb'
        condition = condition_entries.get(plus8_id)
        forward_entry = forwards.get(plus8_id)
        if condition is None or condition.get('state') != 'captured' or forward_entry is None:
            raise ValueError(f'Original +8 RGB condition is unavailable for carrier {frame_id}')
        refs = condition.get('context_refs', {})
        query_entry = entries.get(refs.get('query'))
        if (query_entry is None or refs.get('template') != source_entry['context_id'] or
                refs.get('observed_frame') != frame_entry['context_id']):
            raise ValueError(f'Original +8 RGB context provenance differs for carrier {frame_id}')
        query_arrays = audit._load_npz(capture_root, query_entry)
        _validate_plus8_context(query_entry, query_arrays, source_arrays)
        if (forward_entry.get('context_refs') != refs or
                forward_entry.get('forward_index') != condition.get('forward_index')):
            raise ValueError(f'Original +8 RGB forward provenance differs for carrier {frame_id}')
        old_packet = audit._load_npz(capture_root, forward_entry)
        if (forward_entry.get('output_hashes') != dict(
                flow=audit.array_sha256(old_packet['flow']),
                confidence=audit.array_sha256(old_packet['confidence']))):
            raise ValueError(f'Original +8 RGB output hashes changed for carrier {frame_id}')
        expected_derived = audit._array_hashes_for_forward(source_arrays, old_packet)
        if forward_entry.get('derived_array_hashes') != expected_derived:
            raise ValueError(f'Original +8 RGB correspondence hashes changed for carrier {frame_id}')
        prepared[frame_id]['plus8'] = dict(
            condition=condition, query_entry=query_entry, query_arrays=query_arrays,
            forward_entry=forward_entry, forward_arrays=old_packet,
        )
    return prepared


def _zero_flow_controls(prepared, diagonal_m):
    rows = []
    for frame_id in CARRIERS:
        item = prepared[frame_id]
        h, w = item['common_mask'].shape
        zero_flow = np.zeros((h, w, 2), dtype=np.float32)
        confidence = np.ones((h, w), dtype=np.float32)
        patch = audit.audit_patch_identity(
            item['full_rgb'], item['common_mask'], zero_flow, confidence,
            item['source_bank'], [item['source_bank'], item['competitor_bank']], diagonal_m)
        summary = patch['summary']
        rows.append(dict(
            frame_id=frame_id, state=summary['state'],
            supported=summary['supported'], supported_cells_4x4=summary['supported_cells_4x4'],
            supported_hull_fraction=summary['supported_hull_fraction'],
            common_mask_sha256=audit.array_sha256(item['common_mask']),
            source_bank_sha256=item['source_entry']['fixed_patch_bank']['template_input_sha256'],
            competitor_bank_sha256=item['competitor_entry']['fixed_patch_bank']['template_input_sha256'],
            patch_summary=summary,
        ))
    return rows


def _rgb_tensor_array(rgb):
    value = np.asarray(rgb, dtype=np.float32)
    if value.max(initial=0.) > 1.5:
        value = value / 255.
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None])


def _torch_output_array(tensor):
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach()
    if hasattr(tensor, 'float'):
        tensor = tensor.float()
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    return np.asarray(tensor.numpy())


def _sync(torch, device):
    if device == 'cuda' and getattr(torch, 'cuda', None) is not None:
        torch.cuda.synchronize()


def _write_packet(output_root, record, condition, arrays, budget_bytes=PACKET_BUDGET_BYTES):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    total = int(record['packet_bytes']) + len(payload)
    if total > int(budget_bytes):
        record['packet_failures'].append(dict(
            condition_id=condition['condition_id'], attempted_bytes=len(payload),
            aggregate_bytes=total, budget_bytes=int(budget_bytes),
            reason='aggregate_16_mib_packet_budget_exceeded'))
        save_result(output_root / 'zero_view.json', record)
        raise ValueError(f'Zero-view packet aggregate exceeds {budget_bytes} bytes')
    relative = Path('packets') / f"{condition['condition_id']}.npz"
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(target.suffix + '.tmp').exists():
        raise FileExistsError(f'Refusing to replace immutable zero-view packet: {target}')
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    try:
        target.chmod(0o444)
    except OSError:
        pass
    entry = dict(
        path=relative.as_posix(), bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        arrays={name: audit.array_info(value) for name, value in arrays.items()},
    )
    record['packet_bytes'] = total
    return entry


def _load_runtime():
    import torch
    from .quality_gotrack import load_network, upstream_path

    upstream_path()
    capture_probe._set_deterministic_runtime(torch)
    return torch, load_network('cuda'), 'cuda'


def _run_one(condition, item, network, torch, device, output_root, record, budget_bytes):
    if int(record['forward_calls']) >= FORWARD_CAP:
        raise ValueError('Frozen six-forward budget reached before this condition')
    query_rgb = item['full_rgb'] if condition['condition'] == 'full' else item['clipped_rgb']
    template_rgb = item['source_arrays']['template_rgb']
    template_mask = np.asarray(item['source_arrays']['template_mask'], dtype=bool)
    if condition['condition'] == 'full' and query_rgb is not template_rgb:
        raise AssertionError('Full self input must be the captured template RGB array')
    query_tensor = capture_probe._torch_rgb(torch, query_rgb, device)
    template_tensor = capture_probe._torch_rgb(torch, template_rgb, device)
    mask_tensor = capture_probe._torch_mask(torch, template_mask, device)
    condition['state'] = 'running'
    record['forward_calls'] = int(record['forward_calls']) + 1
    condition['input_hashes'] = dict(
        query_rgb=audit.array_info(query_rgb),
        template_rgb=audit.array_info(template_rgb),
        template_mask=audit.array_info(template_mask),
        query_tensor_nchw=audit.array_info(_rgb_tensor_array(query_rgb)),
        template_tensor_nchw=audit.array_info(_rgb_tensor_array(template_rgb)),
        template_mask_tensor=audit.array_info(np.asarray(mask_tensor.detach().cpu().numpy())),
    )
    condition['context_refs'] = dict(
        source_template=item['source_entry']['context_id'],
        competitor_template=item['competitor_entry']['context_id'],
        observed_frame=item['frame_entry']['context_id'],
    )
    save_result(output_root / 'zero_view.json', record)
    _sync(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        flow_tensor, confidence_tensor = network(query_tensor, template_tensor, mask_tensor)
    _sync(torch, device)
    elapsed_ms = (time.perf_counter() - started) * 1000.
    flow_batch = _torch_output_array(flow_tensor)
    confidence_batch = _torch_output_array(confidence_tensor)
    if flow_batch.shape != (1, 2, 280, 280) or confidence_batch.shape != (1, 280, 280):
        raise ValueError(f'Unexpected GoTrack output shapes: {flow_batch.shape}, {confidence_batch.shape}')
    arrays = dict(
        flow=np.asarray(flow_batch[0].transpose(1, 2, 0), dtype=np.float32),
        confidence=np.asarray(confidence_batch[0], dtype=np.float32),
    )
    entry = _write_packet(output_root, record, condition, arrays, budget_bytes)
    entry.update(
        condition_id=condition['condition_id'], forward_index=condition['forward_index'],
        output_hashes={name: audit.array_sha256(value) for name, value in arrays.items()},
        elapsed_ms=float(elapsed_ms), context_refs=condition['context_refs'],
    )
    record['forwards'].append(entry)
    condition.update(
        state='captured', packet_path=entry['path'], packet_sha256=entry['sha256'],
        output_hashes=entry['output_hashes'], elapsed_ms=float(elapsed_ms),
    )
    save_result(output_root / 'zero_view.json', record)


def _mapping(template_arrays, forward_arrays):
    ids = np.asarray(template_arrays['source_indices'], dtype=np.int64)
    pixels = np.asarray(template_arrays['source_pixels_xy'], dtype=np.float64)
    h, w = np.asarray(forward_arrays['confidence']).shape
    flow = np.asarray(forward_arrays['flow'], dtype=np.float32)
    confidence = np.asarray(forward_arrays['confidence'], dtype=np.float32)
    if flow.shape != (h, w, 2) or len(ids) != len(pixels):
        raise ValueError('Zero-view packet dimensions do not match the frozen template')
    yy, xx = np.divmod(ids, w)
    return pixels + flow[yy, xx], confidence[yy, xx]


def _identity_record(item, forward_arrays, query_rgb, query_depth, query_mask,
                     query_pose_m, frame_id, condition_id, diagonal_m, common_denominator=None):
    template = item['source_arrays']
    endpoints, source_confidence = _mapping(template, forward_arrays)
    metrics, metric_arrays = audit.synthetic_identity_metrics(
        template['source_points_object_m'], template['source_pixels_xy'],
        template['source_indices'], endpoints, source_confidence, query_pose_m,
        query_depth, query_mask, template['crop_k'], diagonal_m)
    visible_ids = np.asarray(template['source_indices'], dtype=np.int64)[metric_arrays['true_visible']]
    bank_result = audit.audit_patch_identity(
        query_rgb, query_mask, forward_arrays['flow'], forward_arrays['confidence'],
        item['source_bank'], [item['source_bank'], item['competitor_bank']], diagonal_m)
    confidence_ok = metrics['confidence_availability_on_visible'] >= .50
    identity_ok = (metrics['correct_fraction_confident_visible'] is not None and
                   metrics['correct_fraction_confident_visible'] >= .90)
    patch_ok = bank_result['summary']['state'] == 'distinctive_current_image_support'
    return dict(
        condition_id=condition_id, frame_id=int(frame_id), state='scored',
        visible_denominator=dict(
            count=int(metrics['truly_visible_sources']),
            source_indices_sha256=audit.array_sha256(visible_ids),
            common_mask_sha256=audit.array_sha256(query_mask),
            source_count=int(len(template['source_indices'])),
            shared_with_zero_conditions=common_denominator,
        ),
        identity_metrics=metrics,
        patch_audit=dict(summary=bank_result['summary'], rows=bank_result['rows']),
        gates=dict(
            confidence_availability_passed=bool(confidence_ok),
            confident_visible_identity_passed=bool(identity_ok),
            fixed_patch_identity_passed=bool(patch_ok),
            qualified=bool(confidence_ok and identity_ok and patch_ok),
            thresholds=dict(confidence_gt=audit.VISIBILITY_WEIGHT,
                            min_availability=.50, min_correct_confident_visible=.90,
                            endpoint_error_crop_px=3., identity_distance_fraction=.02,
                            patch_supported=8, patch_cells=3, patch_hull_fraction=.12),
        ),
    )


def _score_outputs(capture_root, capture, prepared, output_root, record):
    new_rows = []
    for condition in record['conditions']:
        if condition['state'] != 'captured':
            continue
        item = prepared[condition['frame_id']]
        entry = next(row for row in record['forwards']
                     if row['condition_id'] == condition['condition_id'])
        forward_arrays = audit._load_npz(output_root, entry)
        query_rgb = item['full_rgb'] if condition['condition'] == 'full' else item['clipped_rgb']
        common_mask = item['common_mask']
        result = _identity_record(
            item, forward_arrays, query_rgb, item['source_arrays']['template_depth_mm'],
            common_mask, audit.crop_pose_from_context(item['source_arrays']),
            condition['frame_id'], condition['condition_id'],
            float(capture['object_bbox_diagonal_m']),
            common_denominator=True,
        )
        condition['score'] = result
        new_rows.append(result)
    by_frame = {}
    for row in new_rows:
        by_frame.setdefault((row['frame_id'], row['condition_id'].rsplit('-', 1)[-1]), []).append(row)
    full_qualified = [next((row['gates']['qualified'] for row in new_rows
                            if row['frame_id'] == frame_id and row['condition_id'].endswith('-full')), False)
                      for frame_id in CARRIERS]
    clipped_qualified = [next((row['gates']['qualified'] for row in new_rows
                               if row['frame_id'] == frame_id and row['condition_id'].endswith('-clipped')), False)
                         for frame_id in CARRIERS]
    record['zero_view_scores'] = new_rows
    record['zero_view_decision'] = dict(
        full_all_carriers_qualified=bool(len(full_qualified) == 3 and all(full_qualified)),
        clipped_all_carriers_qualified=bool(len(clipped_qualified) == 3 and all(clipped_qualified)),
        per_carrier={str(frame_id): {
            'full': next((row['gates']['qualified'] for row in new_rows
                          if row['frame_id'] == frame_id and row['condition_id'].endswith('-full')), None),
            'clipped': next((row['gates']['qualified'] for row in new_rows
                             if row['frame_id'] == frame_id and row['condition_id'].endswith('-clipped')), None),
        } for frame_id in CARRIERS},
        interpretation_scope='Controlled synthetic self-input calibration only; no real pose accuracy claim.',
    )

    plus8_rows = []
    for frame_id in CARRIERS:
        item = prepared[frame_id]
        old = item['plus8']
        arrays = old['query_arrays']
        result = _identity_record(
            item, old['forward_arrays'], arrays['query_rgb_rgb'], arrays['query_depth_mm'],
            np.asarray(arrays['synthetic_query_mask'], dtype=bool),
            np.asarray(arrays['known_query_pose_m'], dtype=np.float64),
            frame_id, old['condition']['condition_id'],
            float(capture['object_bbox_diagonal_m']), common_denominator=False,
        )
        result.update(
            source_capture_condition=old['condition']['condition_id'],
            source_capture_packet_sha256=old['forward_entry']['sha256'],
            query_context_id=old['query_entry']['context_id'],
        )
        plus8_rows.append(result)
    record['stored_plus8_rgb_rows'] = plus8_rows
    record['stored_plus8_denominators_differ_from_zero_view'] = {
        str(frame_id): dict(
            zero_view_count=next((row['visible_denominator']['count'] for row in new_rows
                                  if row['frame_id'] == frame_id), None),
            plus8_count=next((row['visible_denominator']['count'] for row in plus8_rows
                              if row['frame_id'] == frame_id), None),
        ) for frame_id in CARRIERS
    }
    record['plus8_recomputed_from_existing_packets'] = True


def _mark_remaining_unavailable(record, reason):
    for condition in record['conditions']:
        if condition['state'] == 'pending':
            condition.update(state='unavailable', reason=reason)


def _release_runtime(torch, device):
    gc.collect()
    if device == 'cuda':
        _sync(torch, device)
    if device == 'cuda' and getattr(torch, 'cuda', None) is not None:
        torch.cuda.empty_cache()


def _exception_details(error):
    """Copy structured exception evidence without retaining traceback frames."""
    chain = []
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(dict(type=type(current).__name__, message=str(current)))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        nested = getattr(current, 'exceptions', ())
        pending.extend(item for item in nested if isinstance(item, BaseException))
    details = dict(chain[0])
    if len(chain) > 1:
        details['chain'] = chain[1:]
    return details


def _detach_exception_tracebacks(error):
    """Break traceback/cause links that can keep inference locals alive."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(item for item in (current.__cause__, current.__context__)
                       if item is not None)
        pending.extend(item for item in getattr(current, 'exceptions', ())
                       if isinstance(item, BaseException))
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def run_capture(capture_root=CAPTURE_ROOT, output_root=OUTPUT_ROOT, *,
                _runtime_loader=None, _review_record=None):
    """Run the frozen zero-view stage; only the parent schedules its GPU use."""
    output_root = _create_output_root(output_root)
    record = dict(
        schema_version=1, scope=__doc__, status='initializing', complete=False,
        capture_root=str(Path(capture_root).expanduser().resolve()),
        planned_forward_cap=FORWARD_CAP, forward_calls=0,
        packet_budget_bytes=PACKET_BUDGET_BYTES, packet_bytes=0,
        reference_or_annotations_loaded=False,
        known_synthetic_geometry_used_only_for_cpu_scoring=True,
        selected_frame_ids=list(CARRIERS), template_offset_deg=0,
        query_template_appearance='rgb', clipping_value_float32=.5,
        conditions=_planned_conditions(), forwards=[], packet_failures=[],
        output_root=str(output_root), started_unix=time.time(),
        runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                     opencv=cv2.__version__),
    )
    save_result(output_root / 'zero_view.json', record)
    prepared = None
    torch = network = None
    device = None
    failure = None
    failure_details = None
    try:
        capture_root, capture = audit._load_capture_manifest(capture_root)
        if capture.get('complete') is not True or capture.get('status') != 'complete':
            raise ValueError('Zero-view calibration requires the completed immutable R1 v2 capture')
        actual_capture_sha = digest(capture_root / 'capture.json')
        if actual_capture_sha.casefold() != EXPECTED_CAPTURE_SHA256.casefold():
            raise ValueError('Input capture.json is not the frozen R1 v2 packet set')
        if capture.get('reference_or_annotations_loaded') is not False:
            raise ValueError('Source R1 capture must declare that no references/annotations were loaded')
        record.update(
            capture_sha256=actual_capture_sha,
            capture_packet_bytes=int(capture['packet_bytes']),
            source_pins=_source_pins(capture_root, capture, _review_record),
            object_bbox_diagonal_m=float(capture['object_bbox_diagonal_m']),
            model_checkpoint_sha256=capture['model_checkpoint_sha256'],
        )
        prepared = _load_inputs(capture_root, capture)
        controls = _zero_flow_controls(prepared, float(capture['object_bbox_diagonal_m']))
        record['zero_flow_controls'] = controls
        record['zero_flow_control_gate'] = dict(
            thresholds=dict(min_supported_anchors=8, min_supported_cells=3,
                            min_hull_fraction=.12, ncc_min=audit.NCC_MIN,
                            distant_margin_min=audit.NCC_MARGIN_MIN),
            passed=bool(len(controls) == 3 and all(
                row['state'] == 'distinctive_current_image_support' and
                row['supported'] >= 8 and row['supported_cells_4x4'] >= 3 and
                row['supported_hull_fraction'] >= .12 for row in controls)),
        )
        if not record['zero_flow_control_gate']['passed']:
            record['status'] = 'unavailable'
            record['unavailable_reason'] = 'zero_flow_patch_controls_do_not_pass_all_three_carriers'
            record['calibration_available'] = False
            for condition in record['conditions']:
                condition.update(state='unavailable', reason=record['unavailable_reason'])
            record['completed_unix'] = time.time()
            save_result(output_root / 'zero_view.json', record)
        else:
            record.update(status='preflight_passed', calibration_available=True,
                          planned_conditions_verified=True)
            save_result(output_root / 'zero_view.json', record)

            if _runtime_loader is None:
                checkpoint_path = CACHE / 'checkpoints' / 'gotrack_checkpoint.pt'
                if not checkpoint_path.is_file() or digest(checkpoint_path) != capture['model_checkpoint_sha256']:
                    raise ValueError('Current GoTrack checkpoint does not match the frozen R1 capture pin')
                record['source_pins']['checkpoint']['actual_file_checked'] = True
                record['source_pins']['checkpoint']['actual_file_sha256'] = digest(checkpoint_path)
                torch, network, device = _load_runtime()
            else:
                record['runtime']['mocked_runtime_boundary'] = True
                save_result(output_root / 'zero_view.json', record)
                torch, network, device = _runtime_loader()
            record['runtime']['device'] = device
            record['runtime']['network_loaded_after_zero_flow_controls'] = True
            if device == 'cuda':
                record['runtime'].update(capture_probe._runtime_info(torch))
                record['runtime']['network_forwards_max'] = FORWARD_CAP
            save_result(output_root / 'zero_view.json', record)
            for condition in record['conditions']:
                _run_one(condition, prepared[condition['frame_id']], network,
                         torch, device, output_root, record, PACKET_BUDGET_BYTES)
    except BaseException as error:
        failure = error
        failure_details = _exception_details(error)
        for condition in record['conditions']:
            if condition['state'] == 'running':
                condition.update(state='failed', reason=f'{type(error).__name__}: {error}')
        _mark_remaining_unavailable(record, 'prior_stage_failed')
        record.setdefault('failures', []).append(dict(failure_details))
        _detach_exception_tracebacks(error)
    finally:
        runtime_released = False
        try:
            network = None
            _release_runtime(torch, device)
            record['network_released_before_cpu_scoring'] = True
            runtime_released = True
        except BaseException as release_error:
            release_details = _exception_details(release_error)
            if failure is None:
                failure = release_error
                failure_details = dict(release_details, stage='network_release')
            record.setdefault('failures', []).append(dict(release_details, stage='network_release'))
            _detach_exception_tracebacks(release_error)
        if prepared is not None and record.get('calibration_available') and runtime_released:
            try:
                _score_outputs(capture_root, capture, prepared, output_root, record)
                record['cpu_scoring_completed_after_network_release'] = True
            except BaseException as score_error:
                score_details = _exception_details(score_error)
                if failure is None:
                    failure = score_error
                    failure_details = score_details
                record.setdefault('failures', []).append(dict(score_details, stage='cpu_scoring'))
                _detach_exception_tracebacks(score_error)
        elif prepared is not None and record.get('calibration_available'):
            record['cpu_scoring_skipped_reason'] = 'runtime_release_failed'
        record['captured_condition_count'] = sum(row['state'] == 'captured' for row in record['conditions'])
        record['failed_condition_count'] = sum(row['state'] == 'failed' for row in record['conditions'])
        record['unavailable_condition_count'] = sum(row['state'] == 'unavailable' for row in record['conditions'])
        record['all_six_conditions_accounted'] = all(
            row['state'] in ('captured', 'failed', 'unavailable') for row in record['conditions'])
        record['complete'] = bool(failure is None and record['captured_condition_count'] == FORWARD_CAP and
                                  record.get('cpu_scoring_completed_after_network_release'))
        if failure is not None:
            record['status'] = 'failed'
            record['failure'] = dict(failure_details or _exception_details(failure))
        elif record['complete']:
            record['status'] = 'complete'
        record['completed_unix'] = time.time()
        try:
            save_result(output_root / 'zero_view.json', record)
        except Exception as persistence_error:
            persistence_details = _exception_details(persistence_error)
            _detach_exception_tracebacks(persistence_error)
            record['status'] = 'failed'
            record['complete'] = False
            record['failure'] = dict(persistence_details, stage='terminal_persistence')
            record.setdefault('failures', []).append(dict(persistence_details, stage='terminal_persistence'))
            if failure is not None:
                raise persistence_error from failure
            raise persistence_error
    if failure is not None:
        raise failure
    print(json.dumps(dict(
        status=record['status'], complete=record['complete'],
        planned_forwards=FORWARD_CAP, actual_forwards=record['forward_calls'],
        packet_bytes=record['packet_bytes'], packet_budget_bytes=PACKET_BUDGET_BYTES,
        output_root=str(output_root),
    ), indent=2), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=CAPTURE_ROOT,
                        help='Immutable completed R1 v2 capture to link and score')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT,
                        help='Fresh private output directory; existing roots are preserved')
    parser.add_argument('--review-record', type=Path, default=None,
                        help='Independent accepted Sol record for explicit inference-source deltas')
    args = parser.parse_args(argv)
    run_capture(args.capture_root, args.output_root, _review_record=args.review_record)


if __name__ == '__main__':
    main()
