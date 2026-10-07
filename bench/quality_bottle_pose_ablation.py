"""CPU-only source-observability pose ablation over frozen bottle packets.

The source-only texture selection is frozen before any forward field or
synthetic evaluator geometry is loaded. Fits call the existing
quality_bottle_identity_audit.fit_learned_packet helper unchanged.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_zero_view_probe as zero_probe
from .quality_assets import CACHE, ROOT, digest, save_result
from .quality_contract import Frame, PoseCandidate
from .vision import cv2


CAPTURE_ROOT = CACHE / 'diagnostics' / 'bottle-identity-v2'
ZERO_ROOT = CACHE / 'diagnostics' / 'bottle-zero-view-calibration-v3'
OUTPUT_ROOT = CACHE / 'diagnostics' / 'bottle-source-observability-pose-ablation-v1'
CEILING_PATH = CACHE / 'diagnostics' / 'bottle-endpoint-eligibility-ceiling-v1.json'
SPEC_PATH = ROOT / 'docs' / 'bottle-identity-next-experiment.md'
EXPECTED_SPEC_SHA256 = '4bb91eb3782ef689b97b14f0cae68860768b39348526ced21af25ec5cfb78340'
EXPECTED_CAPTURE_SHA256 = '33efa32778c4094a5df0690d49658cb3bfb0be522204b3dc4e1279d9f0d03063'
EXPECTED_ZERO_SHA256 = 'dccbff28e6ea84693fe2a64d40006304f318532daefcff955646827b6129819d'
EXPECTED_CEILING_SHA256 = 'a0c21af577a0cd14c454899a2ab9509b4e52f0a78e948f93be318c7ae1ec7169'
EXPECTED_CHECKPOINT_SHA256 = 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'
CARRIERS = (10, 50, 100)
ARMS = ('all', 'source_observable')
MAX_WORKING_BYTES = 128 * 1024**2
MAX_OUTPUT_BYTES = 8 * 1024**2
MAX_EVIDENCE_IDS = 192
R1_CAPTURE_SCOPE = 'immutable R1 bottle identity capture; source/checkpoint pins inherited'
R3_REPORT_SCOPE = 'closed six-forward R3 zero-view calibration; no new inference'


class WorkingSetLimitError(ValueError):
    pass


class OutputLimitError(ValueError):
    pass


def _canonical_json_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def _array_ref(value):
    return audit.array_info(value)


def _expected_condition_plan():
    rows = []
    for frame_id in CARRIERS:
        rows.append(dict(
            condition_id=f'syn-{frame_id}-q8-rgb-t0-rgb', frame_id=frame_id,
            kind='synthetic_positive', template_offset_deg=0, appearance='rgb'))
        for condition in ('full', 'clipped'):
            rows.append(dict(
                condition_id=f'zero-{frame_id}-{condition}', frame_id=frame_id,
                kind='self_control', template_offset_deg=0, appearance='rgb',
                self_condition=condition))
        rows.append(dict(
            condition_id=f'syn-{frame_id}-q8-rgb-t180-rgb', frame_id=frame_id,
            kind='synthetic_negative', template_offset_deg=180, appearance='rgb'))
    if len(rows) != 12 or len({row['condition_id'] for row in rows}) != 12:
        raise AssertionError('Frozen R5 condition plan must contain exactly twelve unique rows')
    return rows


def _safe_entry_path(root, entry):
    base = Path(root).expanduser().resolve()
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe frozen packet path: {entry["path"]}')
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f'Frozen packet escaped its root: {entry["path"]}')
    return target


def _load_bound_manifests(capture_root=CAPTURE_ROOT, zero_root=ZERO_ROOT):
    capture_root = Path(capture_root).expanduser().resolve()
    zero_root = Path(zero_root).expanduser().resolve()
    if digest(SPEC_PATH).casefold() != EXPECTED_SPEC_SHA256:
        raise ValueError('Frozen R5 appendix hash changed')
    capture_root, capture = audit._load_capture_manifest(capture_root)
    capture_path = capture_root / 'capture.json'
    capture_sha = digest(capture_path)
    if capture_sha.casefold() != EXPECTED_CAPTURE_SHA256:
        raise ValueError('R1 capture is not the frozen source packet set')
    if capture.get('complete') is not True or capture.get('status') != 'complete':
        raise ValueError('R5 requires the complete immutable R1 capture')
    if capture.get('reference_or_annotations_loaded') is not False:
        raise ValueError('R1 capture declares reference or annotation access')
    if str(capture.get('model_checkpoint_sha256', '')).casefold() != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError('R1 checkpoint pin differs from the frozen GoTrack checkpoint')

    zero_path = zero_root / 'zero_view.json'
    zero_sha = digest(zero_path)
    if zero_sha.casefold() != EXPECTED_ZERO_SHA256:
        raise ValueError('R3 report is not the frozen closed six-forward calibration')
    zero_report = json.loads(zero_path.read_text(encoding='utf-8'))
    if (zero_report.get('complete') is not True or zero_report.get('status') != 'complete' or
            zero_report.get('all_six_conditions_accounted') is not True or
            zero_report.get('captured_condition_count') != 6 or
            zero_report.get('capture_sha256', '').casefold() != EXPECTED_CAPTURE_SHA256 or
            zero_report.get('reference_or_annotations_loaded') is not False or
            zero_report.get('model_checkpoint_sha256', '').casefold() != EXPECTED_CHECKPOINT_SHA256):
        raise ValueError('R3 closed report does not bind the expected R1 source and checkpoint')
    zero_rows = {row.get('condition_id'): row for row in zero_report.get('conditions', [])}
    zero_entries = {row.get('condition_id'): row for row in zero_report.get('forwards', [])}
    expected_zero = {f'zero-{frame_id}-{kind}' for frame_id in CARRIERS
                     for kind in ('full', 'clipped')}
    if (set(zero_rows) != expected_zero or set(zero_entries) != expected_zero or
            any(zero_rows[key].get('state') != 'captured' for key in expected_zero)):
        raise ValueError('R3 report does not contain all six frozen self-control packets')

    r1_conditions = {row.get('condition_id'): row for row in capture.get('conditions', [])}
    r1_forwards = {row.get('condition_id'): row for row in capture.get('forwards', [])}
    r1_contexts = {row.get('context_id'): row for row in capture.get('contexts', [])}
    required_r1_ids = {f'syn-{frame_id}-q8-rgb-t{offset}-rgb'
                       for frame_id in CARRIERS for offset in (0, 180)}
    if (not required_r1_ids.issubset(r1_conditions) or
            not required_r1_ids.issubset(r1_forwards) or
            any(r1_conditions[key].get('state') != 'captured' for key in required_r1_ids)):
        raise ValueError('R1 capture is missing one of the six frozen +8 RGB conditions')
    if any(entry.get('role') not in ('real_frame', 'template', 'synthetic_query')
           for entry in r1_contexts.values()):
        raise ValueError('Unexpected context role in the frozen R1 capture')

    ceiling_sha = digest(CEILING_PATH)
    if ceiling_sha.casefold() != EXPECTED_CEILING_SHA256:
        raise ValueError('Closed R4 dense eligibility ceiling hash changed')
    ceiling = json.loads(CEILING_PATH.read_text(encoding='utf-8'))
    if (ceiling.get('all_three_potentially_reachable') is not False or
            ceiling.get('oracle_used_only_for_ceiling_evaluation') is not True or
            ceiling.get('source_selection_before_scoring') is not True):
        raise ValueError('R4 closed record no longer preserves the frozen dense-gate failure')

    source_pins = zero_report.get('source_pins') or {}
    current_sources = source_pins.get('current_sources') or {}
    current_audit_sha = digest(ROOT / 'bench' / 'quality_bottle_identity_audit.py')
    if current_sources.get('bench/quality_bottle_identity_audit.py') != current_audit_sha:
        raise ValueError('Current shared audit source differs from the accepted R3 source snapshot')
    review = source_pins.get('reviewed_source_deltas') or {}
    if (review.get('reviewer') != 'Sol' or review.get('decision') != 'accepted' or
            review.get('review_id') != 'final_correctness_review-R3-neural-boundary-2026-10-02'):
        raise ValueError('R3 source-delta review is absent or no longer accepted')

    return dict(
        capture_root=capture_root, capture=capture, capture_path=capture_path,
        capture_sha256=capture_sha, zero_root=zero_root, zero_report=zero_report,
        zero_path=zero_path, zero_sha256=zero_sha, r1_conditions=r1_conditions,
        r1_forwards=r1_forwards, r1_contexts=r1_contexts, zero_rows=zero_rows,
        zero_entries=zero_entries, ceiling=ceiling, ceiling_sha256=ceiling_sha,
        input_ledger={
            str(capture_path): dict(bytes=capture_path.stat().st_size, sha256=capture_sha),
            str(zero_path): dict(bytes=zero_path.stat().st_size, sha256=zero_sha),
            str(CEILING_PATH.resolve()): dict(bytes=CEILING_PATH.stat().st_size, sha256=ceiling_sha),
        },
        source_pins=dict(
            capture_snapshot={key: item.get('sha256') for key, item in
                              (capture.get('source_snapshot') or {}).items()},
            zero_view=source_pins, checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            current_audit_sha256=current_audit_sha),
    )


def _context_entry(bundle, context_id):
    entry = bundle['r1_contexts'].get(context_id)
    if entry is None:
        raise ValueError(f'Frozen R1 condition references missing context {context_id!r}')
    return entry


def _source_template_refs(frame_id, template_offset_deg):
    return f'template-{frame_id:04d}-{template_offset_deg:03d}'


def _source_frame_ref(frame_id):
    return f'real-frame-{frame_id:04d}'


def freeze_source_selection(source_arrays):
    """Freeze per-source texture eligibility from the source template alone."""
    required = {
        'template_rgb', 'template_mask', 'observed_crop_mask', 'source_indices',
        'source_pixels_xy', 'source_points_object_m',
    }
    missing = required - set(source_arrays)
    if missing:
        raise ValueError(f'Source template is missing frozen selection fields: {sorted(missing)}')
    rgb = np.asarray(source_arrays['template_rgb'])
    template_mask = np.asarray(source_arrays['template_mask'], dtype=bool)
    observed = np.asarray(source_arrays['observed_crop_mask'], dtype=bool)
    ids = np.asarray(source_arrays['source_indices'], dtype=np.int64).reshape(-1)
    pixels = np.asarray(source_arrays['source_pixels_xy'], dtype=np.float64).reshape(-1, 2)
    points = np.asarray(source_arrays['source_points_object_m'], dtype=np.float64).reshape(-1, 3)
    if (rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3 or
            template_mask.shape != rgb.shape[:2] or observed.shape != rgb.shape[:2] or
            not (len(ids) == len(pixels) == len(points))):
        raise ValueError('Frozen source selection inputs have inconsistent dimensions or dtype')
    source_hashes = {name: _array_ref(source_arrays[name]) for name in sorted(required)}
    started = time.perf_counter()
    highpass = audit._highpass(audit._gray_image(rgb), 2.)
    eroded = audit._eroded_mask(template_mask & observed, 6)
    eligible = np.zeros(len(ids), dtype=bool)
    boundary_count = 0
    low_std_count = 0
    for row, pixel in enumerate(pixels):
        patch, _ = audit._bilinear_patch(highpass, pixel, eroded)
        if patch is None:
            boundary_count += 1
            continue
        descriptor, _ = audit._normalize_patch(patch)
        if descriptor is None:
            low_std_count += 1
            continue
        eligible[row] = True
    eligible.setflags(write=False)
    if int(eligible.sum()) + boundary_count + low_std_count != len(ids):
        raise AssertionError('Source eligibility accounting lost one or more frozen source rows')
    return dict(
        eligible=eligible, eligible_count=int(eligible.sum()), total_sources=int(len(ids)),
        boundary_excluded=int(boundary_count), low_std_excluded=int(low_std_count),
        eligible_array=_array_ref(eligible), source_array_hashes=source_hashes,
        source_indices_sha256=_array_digest(ids), source_pixels_sha256=_array_digest(pixels),
        elapsed_ms=float((time.perf_counter() - started) * 1000.),
        selector=dict(gray='RGB luminance', highpass_sigma=2., patch_size=11,
                      minimum_highpass_std=.005, common_mask='template_mask & observed_crop_mask',
                      erosion_pixels=6, pixel_coordinates='upstream pixel centres'))


def _condition_source_only_inputs(bundle, plan):
    frame_id = int(plan['frame_id'])
    template_id = _source_template_refs(frame_id, int(plan['template_offset_deg']))
    template_entry = _context_entry(bundle, template_id)
    if template_entry.get('role') != 'template' or int(template_entry.get('frame_id', -1)) != frame_id:
        raise ValueError(f'Unexpected source template binding for {plan["condition_id"]}')
    arrays = audit._load_npz(bundle['capture_root'], template_entry)
    zero_probe._validate_template(template_entry, arrays)
    return template_entry, arrays


def freeze_all_source_selections(bundle, plans=None):
    """Hash selectors before any R1/R3 flow packet or evaluator truth is read."""
    plans = _expected_condition_plan() if plans is None else plans
    template_specs = {}
    for plan in plans:
        template_id = _source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        template_specs.setdefault(template_id, dict(
            frame_id=plan['frame_id'], template_offset_deg=plan['template_offset_deg']))
    if len(template_specs) != 6:
        raise AssertionError('R5 requires six independently frozen source templates')
    selections = {}
    for template_id, spec in template_specs.items():
        plan = dict(condition_id=template_id, **spec)
        entry, arrays = _condition_source_only_inputs(bundle, plan)
        selection = freeze_source_selection(arrays)
        selection.update(context_id=template_id, packet_path=entry['path'],
                         packet_sha256=entry['sha256'], packet_bytes=int(entry['bytes']))
        selections[template_id] = selection
        path = _safe_entry_path(bundle['capture_root'], entry)
        bundle['input_ledger'][str(path)] = dict(bytes=entry['bytes'], sha256=entry['sha256'])
        del arrays
    return selections


def _query_truth_for_plan(bundle, plan, source_arrays):
    """Load evaluator-only truth after the source selector has already frozen."""
    if plan['kind'] == 'self_control':
        truth_pose_crop_m = audit.crop_pose_from_context(source_arrays)
        common_mask = (np.asarray(source_arrays['template_mask'], dtype=bool) &
                       np.asarray(source_arrays['observed_crop_mask'], dtype=bool))
        return dict(
            truth_pose_crop_m=truth_pose_crop_m,
            query_depth_mm=np.asarray(source_arrays['template_depth_mm']),
            query_mask=common_mask,
            query_context_id=source_arrays.get('_context_id', plan['condition_id'] + ':self-template'),
            truth_origin='same captured source template pose/depth with R3 common mask',
            evaluator_mask='R3 frozen common_mask (template_mask & observed_crop_mask)')
    condition = bundle['r1_conditions'][plan['condition_id']]
    refs = condition.get('context_refs') or {}
    if refs.get('template') != _source_template_refs(plan['frame_id'], plan['template_offset_deg']) or \
            refs.get('observed_frame') != _source_frame_ref(plan['frame_id']):
        raise ValueError(f'R1 evaluator refs differ from the frozen plan: {plan["condition_id"]}')
    query_entry = _context_entry(bundle, refs.get('query'))
    if query_entry.get('role') != 'synthetic_query':
        raise ValueError(f'R1 truth context is not a synthetic evaluator record: {plan["condition_id"]}')
    query_arrays = audit._load_npz(bundle['capture_root'], query_entry)
    zero_probe._validate_plus8_context(query_entry, query_arrays, source_arrays)
    truth_pose = np.asarray(query_arrays['known_query_pose_m'], dtype=np.float64)
    if truth_pose.shape != (4, 4):
        raise ValueError(f'Invalid evaluator pose in {plan["condition_id"]}')
    entry_path = _safe_entry_path(bundle['capture_root'], query_entry)
    bundle['input_ledger'][str(entry_path)] = dict(bytes=query_entry['bytes'], sha256=query_entry['sha256'])
    return dict(
        truth_pose_crop_m=truth_pose,
        query_depth_mm=np.asarray(query_arrays['query_depth_mm']),
        query_mask=np.asarray(query_arrays['synthetic_query_mask'], dtype=bool),
        query_context_id=query_entry['context_id'], truth_origin='captured synthetic evaluator context',
        evaluator_mask='captured synthetic_query_mask (historical dense evaluation only)')


def freeze_visible_evaluation_sets(bundle, selections, plans=None):
    """Freeze all-surface V sets from full source maps and evaluator truth only."""
    plans = _expected_condition_plan() if plans is None else plans
    frozen = {}
    diagonal_m = float(bundle['capture']['object_bbox_diagonal_m'])
    for plan in plans:
        template_id = _source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        if template_id not in selections:
            raise ValueError(f'Missing frozen source-only selector for {template_id}')
        entry, source = _condition_source_only_inputs(bundle, plan)
        source['_context_id'] = entry['context_id']
        truth = _query_truth_for_plan(bundle, plan, source)
        points = np.asarray(source['source_points_object_m'], dtype=np.float64)
        pixels = np.asarray(source['source_pixels_xy'], dtype=np.float64)
        ids = np.asarray(source['source_indices'], dtype=np.int64)
        # Endpoints/confidence are evaluator-irrelevant placeholders. The only
        # value retained from this helper is its full-map true_visible vector.
        _, metric_arrays = audit.synthetic_identity_metrics(
            points, pixels, ids, np.zeros((len(points), 2), dtype=np.float64),
            np.ones(len(points), dtype=np.float32), truth['truth_pose_crop_m'],
            truth['query_depth_mm'], truth['query_mask'], source['crop_k'], diagonal_m)
        visible_rows = np.flatnonzero(metric_arrays['true_visible']).astype(np.int64)
        visible_rows.setflags(write=False)
        visible_ids = ids[visible_rows].copy()
        visible_ids.setflags(write=False)
        template_entry = entry
        frozen[plan['condition_id']] = dict(
            visible_rows=visible_rows, visible_source_ids=visible_ids,
            count=int(len(visible_rows)), source_id_sha256=_array_digest(visible_ids),
            ordered_source_id_sha256=_array_digest(visible_ids),
            truth_pose_crop_m=np.array(truth['truth_pose_crop_m'], dtype=np.float64, copy=True),
            truth_context_id=truth['query_context_id'], truth_origin=truth['truth_origin'],
            evaluator_mask=truth['evaluator_mask'],
            source_template_id=template_id,
            source_indices_sha256=_array_digest(ids),
            source_map_hashes={name: _array_ref(source[name]) for name in
                               ('source_indices', 'source_pixels_xy', 'source_points_object_m')},
            query_depth_sha256=_array_digest(truth['query_depth_mm']),
            query_mask_sha256=_array_digest(truth['query_mask']),
            source_packet_sha256=template_entry['sha256'],
            truth_valid=True if len(visible_rows) else False,
            truth_unavailable_reason=None if len(visible_rows) else 'no_full_map_true_visible_surface_sources')
        if len(visible_rows):
            crop_pose_mm = np.array(truth['truth_pose_crop_m'], dtype=np.float64, copy=True)
            crop_pose_mm[:3, 3] *= 1000.
            truth_native = audit.crop_to_native_pose_m(crop_pose_mm, source['crop_from_native'])
            camera_points = (points[visible_rows] @ truth_native[:3, :3].T +
                             truth_native[:3, 3])
            z = camera_points[:, 2]
            truth_valid = bool(np.isfinite(camera_points).all() and np.all(z > .01))
            frozen[plan['condition_id']]['truth_valid'] = truth_valid
            if not truth_valid:
                frozen[plan['condition_id']]['truth_unavailable_reason'] = \
                    'nonfinite_or_nonpositive_10mm_native_truth_projection'
            else:
                native_xy_h = camera_points @ np.asarray(source['native_k'], dtype=np.float64).T
                native_xy = native_xy_h[:, :2] / native_xy_h[:, 2:3]
                frozen[plan['condition_id']]['truth_native_projection_sha256'] = _array_digest(native_xy)
        template_path = _safe_entry_path(bundle['capture_root'], entry)
        bundle['input_ledger'][str(template_path)] = dict(bytes=entry['bytes'], sha256=entry['sha256'])
        del source
    return frozen


def _load_condition_payload(bundle, plan):
    """Load and verify one closed row; no fitting or output mutation occurs here."""
    frame_id = int(plan['frame_id'])
    template_id = _source_template_refs(frame_id, int(plan['template_offset_deg']))
    frame_id_ref = _source_frame_ref(frame_id)
    template_entry = _context_entry(bundle, template_id)
    frame_entry = _context_entry(bundle, frame_id_ref)
    source = audit._load_npz(bundle['capture_root'], template_entry)
    frame = audit._load_npz(bundle['capture_root'], frame_entry)
    zero_probe._validate_template(template_entry, source)
    if frame_entry.get('role') != 'real_frame' or int(frame_entry.get('frame_id', -1)) != frame_id:
        raise ValueError(f'Unexpected observed frame binding for {plan["condition_id"]}')
    if (not np.array_equal(frame.get('observed_crop_mask'), source['observed_crop_mask']) or
            not np.array_equal(frame['native_k'], source['native_k']) or
            not np.array_equal(frame['crop_k'], source['crop_k']) or
            not np.array_equal(frame['crop_from_native'], source['crop_from_native'])):
        raise ValueError(f'Observed frame/template camera or mask provenance differs: {plan["condition_id"]}')
    if (frame['native_rgb'].dtype != np.uint8 or frame['native_rgb'].ndim != 3 or
            frame['native_rgb'].shape[2] != 3 or
            np.asarray(frame['native_mask']).shape != frame['native_rgb'].shape[:2]):
        raise ValueError(f'Invalid observed native frame: {plan["condition_id"]}')
    if (frame_entry.get('native_rgb_sha256') != audit.array_sha256(frame['native_rgb']) or
            frame_entry.get('native_mask_sha256') != audit.array_sha256(frame['native_mask'] > 0)):
        raise ValueError(f'Observed native frame hash differs from R1 context metadata: {plan["condition_id"]}')

    query_entry = None
    query = None
    if plan['kind'] in ('synthetic_positive', 'synthetic_negative'):
        condition = bundle['r1_conditions'][plan['condition_id']]
        refs = condition.get('context_refs') or {}
        if (refs.get('template') != template_id or refs.get('observed_frame') != frame_id_ref or
                condition.get('kind') != 'synthetic' or condition.get('state') != 'captured'):
            raise ValueError(f'R1 condition refs differ from frozen R5 plan: {plan["condition_id"]}')
        query_entry = _context_entry(bundle, refs.get('query'))
        query = audit._load_npz(bundle['capture_root'], query_entry)
        zero_probe._validate_plus8_context(query_entry, query, source)
        forward_entry = bundle['r1_forwards'][plan['condition_id']]
        if (forward_entry.get('condition_id') != plan['condition_id'] or
                forward_entry.get('context_refs') != refs or
                forward_entry.get('forward_index') != condition.get('forward_index') or
                forward_entry.get('rng_seed') != condition.get('rng_seed')):
            raise ValueError(f'R1 forward provenance differs from frozen condition: {plan["condition_id"]}')
        forward_root = bundle['capture_root']
        seed = int(condition['rng_seed'])
        condition_provenance = dict(
            condition_id=plan['condition_id'], kind=condition['kind'],
            forward_index=int(condition['forward_index']), rng_seed=seed,
            context_refs=refs)
    else:
        condition = bundle['zero_rows'][plan['condition_id']]
        forward_entry = bundle['zero_entries'][plan['condition_id']]
        expected_refs = dict(source_template=template_id,
                             competitor_template=_source_template_refs(frame_id, 180),
                             observed_frame=frame_id_ref)
        if (condition.get('state') != 'captured' or
                forward_entry.get('condition_id') != plan['condition_id'] or
                forward_entry.get('context_refs') != expected_refs or
                condition.get('frame_id') != frame_id or
                condition.get('condition') != plan['self_condition']):
            raise ValueError(f'R3 self-control provenance differs from frozen row: {plan["condition_id"]}')
        forward_root = bundle['zero_root']
        seed = frame_id * 17
        condition_provenance = dict(
            condition_id=plan['condition_id'], kind='self_control',
            forward_index=int(condition.get('forward_index', -1)), rng_seed=seed,
            context_refs=expected_refs, self_condition=plan['self_condition'])

    forward = audit._load_npz(forward_root, forward_entry)
    if (forward_entry.get('output_hashes') != dict(
            flow=audit.array_sha256(forward['flow']),
            confidence=audit.array_sha256(forward['confidence']))):
        raise ValueError(f'Frozen forward array hashes differ: {plan["condition_id"]}')
    if forward['flow'].shape != (280, 280, 2) or forward['confidence'].shape != (280, 280):
        raise ValueError(f'Frozen forward shape differs: {plan["condition_id"]}')
    if plan['kind'] in ('synthetic_positive', 'synthetic_negative'):
        expected_derived = audit._array_hashes_for_forward(source, forward)
        if forward_entry.get('derived_array_hashes') != expected_derived:
            raise ValueError(f'Frozen R1 source map/endpoint hashes differ: {plan["condition_id"]}')
        condition_provenance['derived_array_hashes'] = expected_derived
    else:
        condition_provenance['derived_array_hashes'] = audit._array_hashes_for_forward(source, forward)

    endpoint, confidence = zero_probe._mapping(source, forward)
    array_sources = (('template', bundle['capture_root'], template_entry),
                     ('observed_frame', bundle['capture_root'], frame_entry),
                     ('forward', forward_root, forward_entry))
    if query_entry is not None:
        array_sources += (('synthetic_evaluator', bundle['capture_root'], query_entry),)
    provenance = {}
    for role, root, entry in array_sources:
        path = _safe_entry_path(root, entry)
        provenance[role] = dict(
            context_id=entry.get('context_id'), condition_id=entry.get('condition_id'),
            path=entry['path'], bytes=int(entry['bytes']), sha256=entry['sha256'],
            array_hashes={name: metadata['sha256'] for name, metadata in
                          entry.get('arrays', {}).items()})
        bundle['input_ledger'][str(path)] = dict(bytes=int(entry['bytes']), sha256=entry['sha256'])
    return dict(
        plan=plan, source_entry=template_entry, source=source,
        frame_entry=frame_entry, frame=frame, query_entry=query_entry, query=query,
        forward_entry=forward_entry, forward=forward, endpoints=endpoint,
        confidence=confidence, seed=seed, condition_provenance=condition_provenance,
        provenance=provenance, template_id=template_id, frame_id_ref=frame_id_ref)


def _payload_array_bytes(payload):
    total = 0
    for name in ('source', 'frame', 'query', 'forward'):
        value = payload.get(name)
        if isinstance(value, dict):
            total += sum(int(np.asarray(array).nbytes) for array in value.values())
    total += int(np.asarray(payload['endpoints']).nbytes)
    total += int(np.asarray(payload['confidence']).nbytes)
    return total


def _check_working_budget(payload, extra_bytes=0):
    estimate = _payload_array_bytes(payload) + int(extra_bytes)
    if estimate > MAX_WORKING_BYTES:
        raise WorkingSetLimitError(
            f'Estimated R5 working arrays {estimate} exceed {MAX_WORKING_BYTES} bytes')
    return estimate


def preflight_all_conditions(bundle, plans, selections, evaluation_sets):
    """Load/hash every actual frozen row before any pose fit starts."""
    loaded = []
    for plan in plans:
        template_id = _source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        if template_id not in selections or plan['condition_id'] not in evaluation_sets:
            raise ValueError(f'R5 row lacks source-only selector or frozen V: {plan["condition_id"]}')
        payload = _load_condition_payload(bundle, plan)
        estimate = _check_working_budget(payload)
        all_inputs = _arm_inputs(payload, selections[template_id], 'all')
        observable_inputs = _arm_inputs(payload, selections[template_id], 'source_observable')
        full_endpoint_hash = _array_digest(payload['endpoints'])
        full_confidence_hash = _array_digest(payload['confidence'])
        if (_array_digest(all_inputs['endpoints_crop']) != full_endpoint_hash or
                _array_digest(all_inputs['confidence']) != full_confidence_hash or
                _array_digest(payload['endpoints']) != full_endpoint_hash or
                _array_digest(payload['confidence']) != full_confidence_hash):
            raise AssertionError(f'Closed R5 arm loader modified a frozen packet: {plan["condition_id"]}')
        loaded.append(dict(condition_id=plan['condition_id'],
                           template_id=template_id, state='loaded',
                           estimated_working_bytes=estimate,
                           source_packet_sha256=payload['source_entry']['sha256'],
                           forward_packet_sha256=payload['forward_entry']['sha256'],
                           endpoint_dtype=str(payload['endpoints'].dtype),
                           confidence_dtype=str(payload['confidence'].dtype),
                           all_endpoint_sha256=full_endpoint_hash,
                           all_confidence_sha256=full_confidence_hash,
                           observable_row_count=int(len(observable_inputs['rows'])),
                           observable_endpoint_sha256=_array_digest(
                               observable_inputs['endpoints_crop']),
                           observable_confidence_sha256=_array_digest(
                               observable_inputs['confidence'])))
        del payload
    if len(loaded) != 12 or {row['condition_id'] for row in loaded} != {
            plan['condition_id'] for plan in plans}:
        raise AssertionError('Closed R5 preflight did not load all twelve conditions')
    return loaded


def _truth_native_pose(truth_pose_crop_m, crop_from_native):
    pose_mm = np.asarray(truth_pose_crop_m, dtype=np.float64).copy()
    if pose_mm.shape != (4, 4) or not np.isfinite(pose_mm).all():
        raise ValueError('Frozen evaluator crop pose is not a finite 4x4 matrix')
    pose_mm[:3, 3] *= 1000.
    return audit.crop_to_native_pose_m(pose_mm, crop_from_native)


def surface_projection_score(source_points_m, visible_rows, truth_pose_crop_m,
                             candidate_pose_native_m, crop_from_native, native_k,
                             native_frame, visible_source_ids=None, native_rgb_height=None):
    """Score a pose on the immutable full-map truth-visible set V in 720 px units."""
    points = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    rows = np.asarray(visible_rows, dtype=np.int64).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= len(points)):
        raise ValueError('Frozen all-surface visible rows are outside the source map')
    count = int(len(rows))
    denominator = dict(count=count, ordered_source_rows_sha256=_array_digest(rows),
                       ordered_source_ids_sha256=(None if visible_source_ids is None
                                                  else _array_digest(visible_source_ids)))
    if count == 0:
        return dict(state='unavailable_empty_frozen_truth_visible_set', denominator=denominator,
                    invalid_candidate_projection_count=0, median_720=None, p95_720=None,
                    max_720=None, finite_only=None)
    truth_native = _truth_native_pose(truth_pose_crop_m, crop_from_native)
    truth_camera = points[rows] @ truth_native[:3, :3].T + truth_native[:3, 3]
    truth_z = truth_camera[:, 2]
    if not np.isfinite(truth_camera).all() or np.any(truth_z <= .01):
        return dict(state='invalid_truth_projection', denominator=denominator,
                    invalid_candidate_projection_count=0, median_720=None, p95_720=None,
                    max_720=None, finite_only=None,
                    invalid_truth_count=int(np.count_nonzero(~np.isfinite(truth_z) | (truth_z <= .01))))
    if candidate_pose_native_m is None:
        return dict(state='unavailable_candidate_pose', denominator=denominator,
                    invalid_candidate_projection_count=0, median_720=None, p95_720=None,
                    max_720=None, finite_only=None)
    candidate = np.asarray(candidate_pose_native_m, dtype=np.float64)
    if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
        return dict(state='invalid_candidate_pose', denominator=denominator,
                    invalid_candidate_projection_count=count, median_720=None, p95_720=None,
                    max_720=None, finite_only=None)
    k = np.asarray(native_k, dtype=np.float64)
    truth_h = truth_camera @ k.T
    candidate_camera = points[rows] @ candidate[:3, :3].T + candidate[:3, 3]
    candidate_h = candidate_camera @ k.T
    truth_xy = truth_h[:, :2] / truth_h[:, 2:3]
    candidate_xy = candidate_h[:, :2] / candidate_h[:, 2:3]
    frame_height = int(native_rgb_height if native_rgb_height is not None
                       else np.asarray(native_frame.rgb).shape[0])
    if frame_height <= 0:
        raise ValueError('Native frame height must be positive for 720-pixel normalization')
    invalid = (~np.isfinite(candidate_camera).all(axis=1) |
               (candidate_camera[:, 2] <= .01) | ~np.isfinite(candidate_xy).all(axis=1))
    errors = np.linalg.norm(candidate_xy - truth_xy, axis=1) * (720. / frame_height)
    invalid |= ~np.isfinite(errors)
    invalid_count = int(invalid.sum())
    finite = errors[~invalid]
    if invalid_count:
        return dict(state='invalid_projection', denominator=denominator,
                    invalid_candidate_projection_count=invalid_count,
                    median_720=None, p95_720=None, max_720=None,
                    finite_only=(dict(count=int(len(finite)), median_720=float(np.median(finite)),
                                     p95_720=float(np.percentile(finite, 95)),
                                     max_720=float(np.max(finite))) if len(finite) else None))
    return dict(state='scored', denominator=denominator,
                invalid_candidate_projection_count=0,
                median_720=float(np.median(errors)),
                p95_720=float(np.percentile(errors, 95)), max_720=float(np.max(errors)),
                finite_only=None)


def _pose_truth_errors(pose_crop_m, truth_pose_crop_m, diagonal_m):
    if pose_crop_m is None:
        return dict(rotation_error_degrees=None, translation_error_m=None,
                    translation_error_fraction_D=None)
    fitted = np.asarray(pose_crop_m, dtype=np.float64)
    truth = np.asarray(truth_pose_crop_m, dtype=np.float64)
    if fitted.shape != (4, 4) or not np.isfinite(fitted).all():
        return dict(rotation_error_degrees=None, translation_error_m=None,
                    translation_error_fraction_D=None)
    relative = fitted[:3, :3] @ truth[:3, :3].T
    angle = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))
    translation = float(np.linalg.norm(fitted[:3, 3] - truth[:3, 3]))
    return dict(rotation_error_degrees=angle, translation_error_m=translation,
                translation_error_fraction_D=translation / float(diagonal_m))


def _filtered_observed_rows(source, endpoints, confidence, observed_mask):
    points = np.asarray(source['source_points_object_m'], dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(source['source_indices'], dtype=np.int64).reshape(-1)
    uv = np.asarray(endpoints, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    mask = np.asarray(observed_mask, dtype=bool)
    if not (len(points) == len(ids) == len(uv) == len(conf)):
        raise ValueError('Observed residual input arrays differ in length')
    h, w = mask.shape
    ix, iy, inside = audit._floor_indices(uv, w, h)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(uv).all(axis=1) & np.isfinite(conf)
    valid &= conf > audit.VISIBILITY_WEIGHT
    valid &= (ids >= 0) & (ids < mask.size)
    valid &= inside
    source_mask = mask.reshape(-1)
    valid &= source_mask[np.clip(ids, 0, max(0, len(source_mask) - 1))]
    rows = np.flatnonzero(inside)
    target_keep = np.zeros(len(points), dtype=bool)
    target_keep[rows] = mask[iy[rows], ix[rows]]
    return np.flatnonzero(valid & target_keep)


def _residual_summary(points_m, endpoints_crop, source_ids, pose_native_m,
                      crop_k, crop_from_native, native_k, native_height):
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    endpoints = np.asarray(endpoints_crop, dtype=np.float64).reshape(-1, 2)
    if pose_native_m is None:
        return dict(state='unavailable_candidate_pose', denominator=int(len(points)),
                    source_ids=dict(count=int(len(source_ids)), sha256=_array_digest(source_ids)),
                    median_720=None, p95_720=None, max_720=None, invalid_projection_count=0)
    crop_rays = np.column_stack((endpoints, np.ones(len(endpoints)))) @ np.linalg.inv(
        np.asarray(crop_k, dtype=np.float64)).T
    native_rays = crop_rays @ np.asarray(crop_from_native, dtype=np.float64)[:3, :3]
    measured_h = native_rays @ np.asarray(native_k, dtype=np.float64).T
    with np.errstate(divide='ignore', invalid='ignore'):
        measured = measured_h[:, :2] / measured_h[:, 2:3]
    fit_pose = np.asarray(pose_native_m, dtype=np.float64)
    camera = points @ fit_pose[:3, :3].T + fit_pose[:3, 3]
    projected_h = camera @ np.asarray(native_k, dtype=np.float64).T
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = projected_h[:, :2] / projected_h[:, 2:3]
    errors = np.linalg.norm(projected - measured, axis=1) * 720. / float(native_height)
    invalid = (~np.isfinite(camera).all(axis=1) | (camera[:, 2] <= .01) |
               ~np.isfinite(measured).all(axis=1) | ~np.isfinite(errors))
    invalid_count = int(invalid.sum())
    finite = errors[~invalid]
    if invalid_count:
        return dict(state='invalid_projection', denominator=int(len(points)),
                    source_ids=dict(count=int(len(source_ids)), sha256=_array_digest(source_ids)),
                    median_720=None, p95_720=None, max_720=None,
                    invalid_projection_count=invalid_count,
                    finite_only=(dict(count=int(len(finite)), median_720=float(np.median(finite)),
                                     p95_720=float(np.percentile(finite, 95)),
                                     max_720=float(np.max(finite))) if len(finite) else None))
    return dict(state='scored', denominator=int(len(points)),
                source_ids=dict(count=int(len(source_ids)), sha256=_array_digest(source_ids)),
                median_720=float(np.median(errors)) if len(errors) else None,
                p95_720=float(np.percentile(errors, 95)) if len(errors) else None,
                max_720=float(np.max(errors)) if len(errors) else None,
                invalid_projection_count=0)


def _native_xy_from_crop_endpoints(endpoints_crop, crop_k, crop_from_native, native_k):
    endpoints = np.asarray(endpoints_crop, dtype=np.float64).reshape(-1, 2)
    crop_rays = np.column_stack((endpoints, np.ones(len(endpoints)))) @ np.linalg.inv(
        np.asarray(crop_k, dtype=np.float64)).T
    native_rays = crop_rays @ np.asarray(crop_from_native, dtype=np.float64)[:3, :3]
    projected = native_rays @ np.asarray(native_k, dtype=np.float64).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return projected[:, :2] / projected[:, 2:3]


def _all_original_candidate_validation(payload, original_rows, pose_native_m):
    """Report the unchanged current-image contract on every original observed row."""
    source = payload['source']
    frame = payload['frame']
    rows = np.asarray(original_rows, dtype=np.int64).reshape(-1)
    source_ids = np.asarray(source['source_indices'], dtype=np.int64)[rows]
    common = dict(
        report_only=True,
        changes_fit_or_acceptance=False,
        candidate_pose_native_m=pose_native_m,
        input_correspondences=int(len(rows)),
        source_ids=dict(count=int(len(source_ids)), sha256=_array_digest(source_ids)),
        contract='quality_contract.validate with fitted pose, native RGB/K/mask and all original observed-filtered rows')
    if pose_native_m is None:
        return dict(state='unavailable_candidate_pose', reason='candidate_pose_unavailable',
                    validation_stats=None, **common)
    points = np.asarray(source['source_points_object_m'], dtype=np.float64)[rows]
    endpoints_crop = np.asarray(payload['endpoints'], dtype=np.float64)[rows]
    confidence = np.asarray(payload['confidence'], dtype=np.float64)[rows]
    native_xy = _native_xy_from_crop_endpoints(
        endpoints_crop, source['crop_k'], source['crop_from_native'], frame['native_k'])
    candidate = PoseCandidate(np.asarray(pose_native_m, dtype=np.float64),
                              points, native_xy, confidence)
    native_frame = Frame(int(payload['plan']['frame_id']), frame['native_rgb'], frame['native_k'])
    accepted, reason, stats = audit.validate(candidate, native_frame, frame['native_mask'])
    return dict(state='accepted' if accepted else 'rejected',
                reason=reason, validation_stats=stats,
                candidate_native_pixels_sha256=_array_digest(native_xy), **common)


def _arm_inputs(payload, selection, arm):
    source = payload['source']
    points = np.asarray(source['source_points_object_m'], dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(source['source_indices'], dtype=np.int64).reshape(-1)
    pixels = np.asarray(source['source_pixels_xy'], dtype=np.float64).reshape(-1, 2)
    raw_endpoints = np.asarray(payload['endpoints']).reshape(-1, 2)
    raw_confidence = np.asarray(payload['confidence']).reshape(-1)
    endpoints = np.asarray(raw_endpoints, dtype=np.float64)
    confidence = np.asarray(raw_confidence, dtype=np.float64)
    if not (len(points) == len(ids) == len(pixels) == len(endpoints) == len(confidence) ==
            selection['total_sources']):
        raise ValueError(f'Frozen source map and endpoint arrays differ for {payload["plan"]["condition_id"]}')
    chosen = np.arange(len(ids), dtype=np.int64) if arm == 'all' else np.flatnonzero(selection['eligible'])
    if arm not in ARMS:
        raise ValueError(f'Unknown R5 arm: {arm}')
    before = dict(endpoints=_array_digest(raw_endpoints), confidence=_array_digest(raw_confidence),
                  endpoint_dtype=str(raw_endpoints.dtype), confidence_dtype=str(raw_confidence.dtype))
    result = dict(
        rows=chosen.copy(), source_points_m=points[chosen].copy(), source_pixels_xy=pixels[chosen].copy(),
        source_indices=ids[chosen].copy(), endpoints_crop=raw_endpoints[chosen].copy(),
        confidence=raw_confidence[chosen].copy(),
        source_mask=np.asarray(source['observed_crop_mask'], dtype=bool),
        target_mask=np.asarray(source['observed_crop_mask'], dtype=bool),
        endpoint_confidence_hashes_before=before)
    if (_array_digest(raw_endpoints) != before['endpoints'] or
            _array_digest(raw_confidence) != before['confidence']):
        raise AssertionError('Arm construction modified frozen endpoint or confidence inputs')
    return result


def _observed_filtered_rows_for_arm(arm_inputs):
    source = dict(source_points_object_m=arm_inputs['source_points_m'],
                  source_indices=arm_inputs['source_indices'])
    return _filtered_observed_rows(source, arm_inputs['endpoints_crop'],
                                   arm_inputs['confidence'], arm_inputs['target_mask'])


def _spatial_support(points_xy, selected_rows, observed_mask):
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    mask = np.asarray(observed_mask, dtype=bool)
    observed_y, observed_x = np.nonzero(mask)
    if not len(rows) or not len(observed_x):
        return dict(source_count=int(len(rows)), grid_cells_4x4=0,
                    convex_hull_fraction_of_observed_bbox=0., bbox_crop_xy=None)
    bbox = [int(observed_x.min()), int(observed_y.min()), int(observed_x.max()), int(observed_y.max())]
    span_x, span_y = max(1, bbox[2] - bbox[0] + 1), max(1, bbox[3] - bbox[1] + 1)
    points = xy[rows]
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if not len(points):
        return dict(source_count=int(len(rows)), grid_cells_4x4=0,
                    convex_hull_fraction_of_observed_bbox=0., bbox_crop_xy=bbox)
    cx = np.clip(((points[:, 0] - bbox[0]) / span_x * 4).astype(int), 0, 3)
    cy = np.clip(((points[:, 1] - bbox[1]) / span_y * 4).astype(int), 0, 3)
    cells = sorted({(int(y), int(x)) for y, x in zip(cy, cx)})
    area = 0.
    if len(points) >= 3:
        area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
    return dict(source_count=int(len(points)), grid_cells_4x4=len(cells),
                convex_hull_fraction_of_observed_bbox=float(area / (span_x * span_y)),
                bbox_crop_xy=bbox)


def _compact_ids(values, max_ids=MAX_EVIDENCE_IDS):
    ids = np.asarray(values, dtype=np.int64).reshape(-1)
    if len(ids) <= max_ids:
        sample = ids.tolist()
    else:
        head = max_ids // 2
        tail = max_ids - head
        sample = ids[:head].tolist() + ids[-tail:].tolist()
    return dict(count=int(len(ids)), sha256=_array_digest(ids),
                sample_source_ids=sample, truncated=len(ids) > max_ids,
                sample_policy=(f'first {max_ids // 2} and last {max_ids - max_ids // 2} ordered IDs'
                               if len(ids) > max_ids else 'all ordered IDs'))


def _shared_fit_call(arm_inputs, payload):
    source = payload['source']
    frame_arrays = payload['frame']
    return audit.fit_learned_packet(
        arm_inputs['source_points_m'], arm_inputs['endpoints_crop'], arm_inputs['confidence'],
        arm_inputs['source_indices'], arm_inputs['source_mask'], arm_inputs['target_mask'],
        audit.crop_pose_from_context(source), source['crop_k'], source['crop_from_native'],
        frame_arrays['native_k'], Frame(int(payload['plan']['frame_id']), frame_arrays['native_rgb'],
                                        frame_arrays['native_k']),
        frame_arrays['native_mask'], int(payload['seed']))


def _fit_repeat_signature(result):
    keys = ('state', 'reason', 'retained_correspondences', 'sampled_correspondences',
            'rng_seed', 'ransac_inlier_ids', 'retained_inlier_ids', 'pose_crop_m',
            'pose_native_m', 'validation_state', 'validation_reason', 'validation_stats')
    return {key: result.get(key) for key in keys}


def _fit_source_id_lists(arm_inputs, fit_result, retained_rows):
    arm_ids = np.asarray(arm_inputs['source_indices'], dtype=np.int64)
    retained_rows = np.asarray(retained_rows, dtype=np.int64)
    sample_local = np.linspace(0, len(retained_rows) - 1,
                               min(len(retained_rows), 10000), dtype=np.int64) if len(retained_rows) else np.empty(0, dtype=np.int64)
    sampled_rows = retained_rows[sample_local]
    ransac_local = np.asarray(fit_result.get('ransac_inlier_ids') or [], dtype=np.int64)
    retained_inlier_local = np.asarray(fit_result.get('retained_inlier_ids') or [], dtype=np.int64)
    ransac_source_ids = arm_ids[sampled_rows[ransac_local]] if len(ransac_local) else np.empty(0, dtype=np.int64)
    retained_inlier_source_ids = (arm_ids[retained_inlier_local]
                                  if len(retained_inlier_local) else np.empty(0, dtype=np.int64))
    return dict(
        retained_source_ids=_compact_ids(arm_ids[retained_rows]),
        sampled_source_ids=_compact_ids(arm_ids[sampled_rows]),
        ransac_inlier_source_ids=_compact_ids(ransac_source_ids),
        retained_inlier_source_ids=_compact_ids(retained_inlier_source_ids),
        retained_count=int(len(retained_rows)), sampled_count=int(len(sampled_rows)),
        ransac_inlier_count=int(len(ransac_source_ids)),
        retained_inlier_count=int(len(retained_inlier_source_ids)))


def _unit_pose_parity(fit_result, crop_from_native):
    crop_value = fit_result.get('pose_crop_m')
    native_value = fit_result.get('pose_native_m')
    if crop_value is None or native_value is None:
        return dict(state='unavailable')
    crop_m = np.asarray(crop_value, dtype=np.float64)
    native_m = np.asarray(native_value, dtype=np.float64)
    crop_mm = crop_m.copy()
    crop_mm[:3, 3] *= 1000.
    native_rebuilt = audit.crop_to_native_pose_m(crop_mm, crop_from_native)
    crop_rebuilt_mm = audit.native_to_crop_pose_mm(native_m, crop_from_native)
    crop_rebuilt_m = crop_rebuilt_mm.copy()
    crop_rebuilt_m[:3, 3] *= .001
    return dict(
        state='checked',
        native_translation_roundtrip_error_m=float(np.linalg.norm(native_rebuilt[:3, 3] - native_m[:3, 3])),
        crop_translation_roundtrip_error_m=float(np.linalg.norm(crop_rebuilt_m[:3, 3] - crop_m[:3, 3])),
        native_rotation_roundtrip_max_abs=float(np.max(np.abs(native_rebuilt[:3, :3] - native_m[:3, :3]))),
        crop_rotation_roundtrip_max_abs=float(np.max(np.abs(crop_rebuilt_m[:3, :3] - crop_m[:3, :3]))),
        metres_to_millimetres_factor=1000., millimetres_to_metres_factor=.001)


def _arm_result(payload, selection, visible_set, arm_name, diagonal_m):
    arm_inputs = _arm_inputs(payload, selection, arm_name)
    filtered_rows = _observed_filtered_rows_for_arm(arm_inputs)
    expected_retained_count = int(len(filtered_rows))
    input_endpoint_hash = arm_inputs['endpoint_confidence_hashes_before']['endpoints']
    input_confidence_hash = arm_inputs['endpoint_confidence_hashes_before']['confidence']
    extra = int(arm_inputs['source_points_m'].nbytes + arm_inputs['endpoints_crop'].nbytes +
                arm_inputs['confidence'].nbytes + filtered_rows.nbytes +
                arm_inputs['source_indices'].nbytes + 3 * payload['source']['template_rgb'].nbytes +
                len(payload['source']['source_indices']) * 320 + 2 * 1024 * 1024)
    working_bytes = _check_working_budget(payload, extra)
    started = time.perf_counter()
    primary = _shared_fit_call(arm_inputs, payload)
    primary_ms = (time.perf_counter() - started) * 1000.
    repeat_started = time.perf_counter()
    repeated = _shared_fit_call(arm_inputs, payload)
    repeat_ms = (time.perf_counter() - repeat_started) * 1000.
    if (_array_digest(payload['endpoints']) != input_endpoint_hash or
            _array_digest(payload['confidence']) != input_confidence_hash):
        raise AssertionError(f'Fit modified frozen endpoint/confidence arrays: {payload["plan"]["condition_id"]}')
    repeat_signature = _fit_repeat_signature(repeated)
    primary_signature = _fit_repeat_signature(primary)
    deterministic = _canonical_json_sha256(repeat_signature) == _canonical_json_sha256(primary_signature)
    proposal_crop = primary.get('pose_crop_m')
    proposal_native = primary.get('pose_native_m')
    proposal_finite = bool(proposal_crop is not None and proposal_native is not None and
                           np.isfinite(np.asarray(proposal_crop, dtype=np.float64)).all() and
                           np.isfinite(np.asarray(proposal_native, dtype=np.float64)).all())
    accepted = bool(proposal_finite and primary.get('validation_state') == 'accepted')
    fit_state = str(primary.get('state', 'unavailable'))
    accepted_state = ('accepted' if accepted else
                      'rejected' if proposal_finite else 'unavailable')
    source = payload['source']
    frame = payload['frame']
    retained_info = _fit_source_id_lists(arm_inputs, primary, filtered_rows)
    selected_support = _spatial_support(arm_inputs['source_pixels_xy'], np.arange(
        len(arm_inputs['source_pixels_xy']), dtype=np.int64), arm_inputs['source_mask'])
    retained_support = _spatial_support(arm_inputs['source_pixels_xy'], filtered_rows,
                                        arm_inputs['source_mask'])
    truth_pose = visible_set['truth_pose_crop_m']
    proposal_errors = _pose_truth_errors(proposal_crop, truth_pose, diagonal_m)
    proposal_surface = surface_projection_score(
        source['source_points_object_m'], visible_set['visible_rows'], truth_pose,
        proposal_native if proposal_finite else None, source['crop_from_native'],
        frame['native_k'], Frame(int(payload['plan']['frame_id']), frame['native_rgb'], frame['native_k']),
        visible_source_ids=visible_set['visible_source_ids'])
    accepted_surface = (surface_projection_score(
        source['source_points_object_m'], visible_set['visible_rows'], truth_pose,
        proposal_native, source['crop_from_native'], frame['native_k'],
        Frame(int(payload['plan']['frame_id']), frame['native_rgb'], frame['native_k']),
        visible_source_ids=visible_set['visible_source_ids']) if accepted else
        dict(state='no_accepted_pose', denominator=dict(
            count=visible_set['count'],
            ordered_source_rows_sha256=_array_digest(visible_set['visible_rows']),
            ordered_source_ids_sha256=visible_set['source_id_sha256']),
             invalid_candidate_projection_count=0, median_720=None, p95_720=None,
             max_720=None, finite_only=None))
    original_rows = _filtered_observed_rows(
        source, payload['endpoints'], payload['confidence'], source['observed_crop_mask'])
    original_points = np.asarray(source['source_points_object_m'], dtype=np.float64)[original_rows]
    original_endpoints = np.asarray(payload['endpoints'], dtype=np.float64)[original_rows]
    original_ids = np.asarray(source['source_indices'], dtype=np.int64)[original_rows]
    original_validation = _all_original_candidate_validation(
        payload, original_rows, proposal_native if proposal_finite else None)
    residuals = _residual_summary(
        original_points, original_endpoints, original_ids,
        proposal_native if proposal_finite else None,
        source['crop_k'], source['crop_from_native'], frame['native_k'], frame['native_rgb'].shape[0])
    selected_rows = np.asarray(selection['eligible'], dtype=bool)
    selection_fraction = float(selected_rows.mean()) if len(selected_rows) else 0.
    result = dict(
        arm=arm_name, fit_state=fit_state, fit_reason=primary.get('reason'),
        accepted_pose_state=accepted_state,
        accepted_pose=proposal_native if accepted else None,
        proposal_pose=dict(crop_m=proposal_crop, native_m=proposal_native) if proposal_finite else None,
        proposal_rotation_error=proposal_errors,
        proposal_surface_projection_720=proposal_surface,
        accepted_pose_surface_projection_720=accepted_surface,
        validation_state=primary.get('validation_state'),
        validation_reason=primary.get('validation_reason'),
        validation_stats=primary.get('validation_stats'),
        retained=retained_info,
        arm_input=dict(source_count=int(len(arm_inputs['rows'])),
                       original_source_count=int(selection['total_sources']),
                       original_source_fraction_selected=(float(len(arm_inputs['rows']) /
                                                                max(1, selection['total_sources']))),
                       source_rows_sha256=_array_digest(arm_inputs['rows']),
                       source_indices_sha256=_array_digest(arm_inputs['source_indices']),
                       measured_endpoint_sha256=_array_digest(arm_inputs['endpoints_crop']),
                       original_confidence_sha256=_array_digest(arm_inputs['confidence']),
                       source_selection_coverage_fraction=selection_fraction,
                       selected_source_count=int(selection['eligible_count'])),
        selected_spatial_support=selected_support,
        retained_spatial_support=retained_support,
        current_image_validation_all_observed_sources=original_validation,
        current_image_residuals_all_observed_sources=residuals,
        crop_native_unit_parity=_unit_pose_parity(primary, source['crop_from_native']),
        costs_ms=dict(primary_fit=primary_ms, deterministic_repeat=repeat_ms),
        deterministic_repeat=dict(matched=bool(deterministic),
                                  primary_sha256=_canonical_json_sha256(primary_signature),
                                  repeated_sha256=_canonical_json_sha256(repeat_signature)),
        expected_retained_correspondences=expected_retained_count,
        working_arrays_estimated_bytes=int(working_bytes),
        shared_solver=dict(
            function='quality_bottle_identity_audit.fit_learned_packet',
            seed=int(payload['seed']), source_and_target_mask='captured observed_crop_mask',
            crop_pose_initialization='captured template crop pose',
            crop_intrinsics_sha256=_array_digest(source['crop_k']),
            crop_from_native_sha256=_array_digest(source['crop_from_native']),
            native_intrinsics_sha256=_array_digest(frame['native_k']),
            native_observed_mask_sha256=_array_digest(frame['native_mask'] > 0),
            native_frame_rgb_sha256=_array_digest(frame['native_rgb'])))
    if int(primary.get('retained_correspondences', -1)) != expected_retained_count:
        result['retained_count_matches_shared_filter'] = False
        result['retained_count_mismatch'] = dict(shared_solver=primary.get('retained_correspondences'),
                                                  independent_filter=expected_retained_count)
    else:
        result['retained_count_matches_shared_filter'] = True
    return result


def _legacy_dense_baseline(bundle, payload):
    plan = payload['plan']
    if plan['kind'] == 'self_control':
        closed_row = bundle['zero_rows'].get(plan['condition_id'])
        closed = None if closed_row is None else closed_row.get('score')
        if not isinstance(closed, dict) or closed.get('condition_id') != plan['condition_id']:
            raise ValueError(f'R3 stored dense score is missing: {plan["condition_id"]}')
        return dict(
            source='unchanged closed R3 CPU score',
            evaluator_mask='R3 frozen common_mask (template_mask & observed_crop_mask)',
            visible_denominator=closed.get('visible_denominator'),
            identity_metrics=closed.get('identity_metrics'),
            patch_audit=closed.get('patch_audit', {}).get('summary'),
            gates=closed.get('gates'),
            dense_qualification=bool((closed.get('gates') or {}).get('qualified')))

    source = payload['source']
    query = payload['query']
    truth_pose = np.asarray(query['known_query_pose_m'], dtype=np.float64)
    identity, identity_arrays = audit.synthetic_identity_metrics(
        source['source_points_object_m'], source['source_pixels_xy'], source['source_indices'],
        payload['endpoints'], payload['confidence'], truth_pose,
        query['query_depth_mm'], query['synthetic_query_mask'], source['crop_k'],
        float(bundle['capture']['object_bbox_diagonal_m']))
    source_bank = audit.verify_fixed_patch_bank(payload['source_entry'], source)
    competitor_offset = 180 if int(plan['template_offset_deg']) == 0 else 0
    competitor_id = _source_template_refs(plan['frame_id'], competitor_offset)
    competitor_entry = _context_entry(bundle, competitor_id)
    competitor = audit._load_npz(bundle['capture_root'], competitor_entry)
    competitor_bank = audit.verify_fixed_patch_bank(competitor_entry, competitor)
    competitor_path = _safe_entry_path(bundle['capture_root'], competitor_entry)
    bundle['input_ledger'][str(competitor_path)] = dict(
        bytes=int(competitor_entry['bytes']), sha256=competitor_entry['sha256'])
    banks = [source_bank, competitor_bank]
    patch = audit.audit_patch_identity(
        query['query_rgb_rgb'], query['synthetic_query_mask'], payload['forward']['flow'],
        payload['forward']['confidence'], source_bank, banks,
        float(bundle['capture']['object_bbox_diagonal_m']))
    availability_passed = identity['confidence_availability_on_visible'] >= .50
    identity_passed = (identity['correct_fraction_confident_visible'] is not None and
                       identity['correct_fraction_confident_visible'] >= .90)
    patch_passed = patch['summary']['state'] == 'distinctive_current_image_support'
    dense_gate = bool(availability_passed and identity_passed and patch_passed)
    negative_thresholds = dict(min_confident_invisible_endpoints=24,
                               min_confident_invisible_fraction=.05,
                               min_median_identity_distance_fraction=.1)
    evidence_passed = bool(
        identity['confident_invisible_source_endpoints'] >=
        negative_thresholds['min_confident_invisible_endpoints'] and
        identity['confident_invisible_fraction'] >=
        negative_thresholds['min_confident_invisible_fraction'] and
        identity['confident_invisible_median_identity_distance_fraction'] is not None and
        identity['confident_invisible_median_identity_distance_fraction'] >
        negative_thresholds['min_median_identity_distance_fraction'])
    result = dict(
        source='recomputed unchanged R1 full-field evaluator metrics',
        evaluator_mask='synthetic_query_mask (historical dense evaluation only)',
        visible_denominator=dict(count=identity['truly_visible_sources'],
                                 source_indices_sha256=_array_digest(
                                     np.asarray(source['source_indices'])[identity_arrays['true_visible']]),
                                 source_count=len(source['source_indices'])),
        identity_metrics=identity,
        patch_audit=patch['summary'],
        gates=dict(confidence_availability_passed=bool(availability_passed),
                   confident_visible_identity_passed=bool(identity_passed),
                   fixed_patch_identity_passed=bool(patch_passed),
                   qualified=dense_gate,
                   thresholds=dict(min_availability=.50, min_correct_confident_visible=.90,
                                   min_patch_support=8, min_patch_cells=3,
                                   min_patch_hull_fraction=.12)),
        dense_qualification=dense_gate,
        original_wrong_surface_evidence=dict(
            thresholds=negative_thresholds, thresholds_passed=evidence_passed,
            confident_masked_endpoints=identity['confident_masked_correspondences'],
            confident_invisible_endpoints=identity['confident_invisible_source_endpoints'],
            confident_invisible_fraction=identity['confident_invisible_fraction'],
            median_invisible_identity_distance_fraction=(
                identity['confident_invisible_median_identity_distance_fraction']),
            source_field_sha256=payload['forward_entry']['sha256'],
            source_indices_sha256=_array_digest(source['source_indices'])))
    if plan['kind'] != 'synthetic_negative':
        result.pop('original_wrong_surface_evidence', None)
    return result


def _load_original_dense_baselines(bundle, plans):
    """Recompute/cache-read every frozen dense evaluator before any fit starts."""
    baselines = {}
    for plan in plans:
        payload = _load_condition_payload(bundle, plan)
        baseline = _legacy_dense_baseline(bundle, payload)
        baselines[plan['condition_id']] = baseline
        del payload
    if set(baselines) != {plan['condition_id'] for plan in plans}:
        raise AssertionError('R5 loader regression did not account for all twelve original dense baselines')
    return baselines


def _frozen_input_revalidation(bundle):
    results = []
    for path_text, expected in sorted(bundle['input_ledger'].items()):
        path = Path(path_text)
        if not path.is_file():
            results.append(dict(path=str(path), state='missing', passed=False))
            continue
        actual_size = int(path.stat().st_size)
        actual_hash = digest(path)
        matched = actual_size == int(expected['bytes']) and actual_hash == expected['sha256']
        results.append(dict(path=str(path), expected_bytes=int(expected['bytes']),
                            actual_bytes=actual_size, expected_sha256=expected['sha256'],
                            actual_sha256=actual_hash, state='matched' if matched else 'changed',
                            passed=bool(matched)))
    return dict(state='matched' if results and all(row['passed'] for row in results) else 'failed',
                verified_files=len(results), changed_or_missing=sum(not row['passed'] for row in results),
                files=results)


def _arm_by_condition(record):
    return {row['condition_id']: {arm['arm']: arm for arm in row.get('arms', [])}
            for row in record.get('conditions', [])}


def _exploratory_pose_prerequisite(record, visible_sets, baselines, diagonal_m):
    arms_by_id = _arm_by_condition(record)
    positive_rows = []
    positive_ok = True
    improvement_rows = []
    for frame_id in CARRIERS:
        condition_id = f'syn-{frame_id}-q8-rgb-t0-rgb'
        arms = arms_by_id.get(condition_id, {})
        all_arm = arms.get('all')
        observable = arms.get('source_observable')
        all_error = (all_arm or {}).get('proposal_rotation_error') or {}
        obs_error = (observable or {}).get('proposal_rotation_error') or {}
        truth_valid = bool(visible_sets.get(condition_id, {}).get('truth_valid'))
        accepted = ((observable or {}).get('accepted_pose_state') == 'accepted')
        errors_valid = all(
            value is not None for value in
            (all_error.get('rotation_error_degrees'), all_error.get('translation_error_m'),
             all_error.get('translation_error_fraction_D'), obs_error.get('rotation_error_degrees'),
             obs_error.get('translation_error_m'), obs_error.get('translation_error_fraction_D')))
        no_worse = bool(errors_valid and
                        obs_error['rotation_error_degrees'] <= all_error['rotation_error_degrees'] + 1e-6 and
                        obs_error['translation_error_m'] <= all_error['translation_error_m'] + 1e-9)
        thresholds = bool(errors_valid and obs_error['rotation_error_degrees'] <= 3. and
                          obs_error['translation_error_fraction_D'] <= .02)
        rotation_gain = (all_error.get('rotation_error_degrees') -
                         obs_error.get('rotation_error_degrees')) if errors_valid else None
        translation_gain_D = (all_error.get('translation_error_fraction_D') -
                              obs_error.get('translation_error_fraction_D')) if errors_valid else None
        improved = bool((rotation_gain is not None and rotation_gain >= .1) or
                        (translation_gain_D is not None and translation_gain_D >= .001))
        improvement_rows.append(dict(condition_id=condition_id,
                                     rotation_improvement_degrees=rotation_gain,
                                     translation_improvement_fraction_D=translation_gain_D,
                                     meets_frozen_improvement=improved))
        row_ok = bool(truth_valid and accepted and no_worse and thresholds)
        positive_ok &= row_ok
        positive_rows.append(dict(condition_id=condition_id, truth_set_valid=truth_valid,
                                  source_observable_accepted=accepted, errors_no_worse_than_all=no_worse,
                                  source_observable_errors_within_3deg_002D=thresholds, passed=row_ok))
    self_rows = []
    self_ok = True
    for frame_id in CARRIERS:
        for condition in ('full', 'clipped'):
            condition_id = f'zero-{frame_id}-{condition}'
            arms = arms_by_id.get(condition_id, {})
            all_arm, observable = arms.get('all'), arms.get('source_observable')
            all_error = (all_arm or {}).get('proposal_rotation_error') or {}
            obs_error = (observable or {}).get('proposal_rotation_error') or {}
            truth_valid = bool(visible_sets.get(condition_id, {}).get('truth_valid'))
            accepted = ((all_arm or {}).get('accepted_pose_state') == 'accepted' and
                        (observable or {}).get('accepted_pose_state') == 'accepted')
            errors_valid = all(value is not None for value in (
                all_error.get('rotation_error_degrees'), all_error.get('translation_error_m'),
                all_error.get('translation_error_fraction_D'), obs_error.get('rotation_error_degrees'),
                obs_error.get('translation_error_m'), obs_error.get('translation_error_fraction_D')))
            no_worse = bool(errors_valid and
                            obs_error['rotation_error_degrees'] <= all_error['rotation_error_degrees'] + 1e-6 and
                            obs_error['translation_error_m'] <= all_error['translation_error_m'] + 1e-9)
            within = bool(errors_valid and all_error['rotation_error_degrees'] <= 1. and
                          obs_error['rotation_error_degrees'] <= 1. and
                          all_error['translation_error_fraction_D'] <= .01 and
                          obs_error['translation_error_fraction_D'] <= .01)
            row_ok = bool(truth_valid and accepted and no_worse and within)
            self_ok &= row_ok
            self_rows.append(dict(condition_id=condition_id, truth_set_valid=truth_valid,
                                  both_arms_accepted=accepted, errors_no_worse_than_all=no_worse,
                                  both_arms_within_1deg_001D=within, passed=row_ok))
    negative_rows = []
    no_new_negative_false_acceptance = True
    for frame_id in CARRIERS:
        condition_id = f'syn-{frame_id}-q8-rgb-t180-rgb'
        arms = arms_by_id.get(condition_id, {})
        visible = visible_sets.get(condition_id) or {}
        evidence = (baselines.get(condition_id) or {}).get('original_wrong_surface_evidence') or {}
        evidence_present = bool(evidence.get('thresholds_passed'))
        all_false = bool(evidence_present and (arms.get('all') or {}).get('accepted_pose_state') == 'accepted')
        observable_false = bool(evidence_present and
                                (arms.get('source_observable') or {}).get('accepted_pose_state') == 'accepted')
        new_false = bool(observable_false and not all_false)
        no_new_negative_false_acceptance &= not new_false
        negative_rows.append(dict(
            condition_id=condition_id, full_original_wrong_surface_evidence=evidence_present,
            truth_visible_count=int(visible.get('count', 0)),
            truth_visible_set_valid=bool(visible.get('truth_valid')),
            projection_role='report_only',
            projection_state=('unobservable_empty_truth_visible_set'
                              if int(visible.get('count', 0)) == 0 else
                              'report_only_projection_available'),
            all_arm_false_acceptance=all_false,
            source_observable_false_acceptance=observable_false,
            new_negative_false_acceptance=new_false))
    all_deterministic = bool(len(record.get('conditions', [])) == 12 and all(
        len(row.get('arms', [])) == 2 and
        all(arm.get('deterministic_repeat', {}).get('matched') is True for arm in row['arms'])
        for row in record.get('conditions', [])))
    improvement_pass = any(row['meets_frozen_improvement'] for row in improvement_rows)
    prerequisite_truth_ids = {
        f'syn-{frame_id}-q8-rgb-t0-rgb' for frame_id in CARRIERS
    } | {
        f'zero-{frame_id}-{condition}' for frame_id in CARRIERS
        for condition in ('full', 'clipped')
    }
    prerequisite_truth_sets = [visible_sets.get(condition_id)
                               for condition_id in prerequisite_truth_ids]
    complete_truth = len(prerequisite_truth_sets) == 9 and all(
        item is not None and item.get('truth_valid') is True and item.get('count', 0) > 0
        for item in prerequisite_truth_sets)
    return dict(
        name='exploratory_source_observable_pose_prerequisite',
        thresholds=dict(positive_rotation_degrees_max=3., positive_translation_fraction_D_max=.02,
                        positive_no_worse_rotation_tolerance_degrees=1e-6,
                        positive_no_worse_translation_tolerance_m=1e-9,
                        required_positive_improvement_rotation_degrees=.1,
                        required_positive_improvement_translation_fraction_D=.001,
                        self_rotation_degrees_max=1., self_translation_fraction_D_max=.01,
                        self_no_worse_rotation_tolerance_degrees=1e-6,
                        self_no_worse_translation_tolerance_m=1e-9,
                        negative_new_false_acceptances_allowed=0),
        positive_and_self_truth_sets_valid=bool(complete_truth),
        negative_truth_visible_sets_report_only=True,
        all_three_positive_source_observable_passed=bool(positive_ok),
        positive_rows=positive_rows, positive_improvements=improvement_rows,
        at_least_one_positive_improves=bool(improvement_pass),
        all_six_self_controls_passed=bool(self_ok), self_rows=self_rows,
        no_new_negative_false_acceptance=bool(no_new_negative_false_acceptance),
        negative_rows=negative_rows, deterministic_repeats_match=all_deterministic,
        passed=bool(complete_truth and positive_ok and improvement_pass and self_ok and
                    no_new_negative_false_acceptance and all_deterministic),
        original_dense_identity_and_R4_eligibility_gates_remain_failed=True,
        runtime_integration_authorized=False,
        interpretation='Exploratory pose prerequisite only; no dense identity, real surface identity, mask, attachment, or independent-accuracy claim.')


def _selection_report(selections):
    return {template_id: {key: value for key, value in selection.items() if key != 'eligible'}
            for template_id, selection in selections.items()}


def _summary_cost(values):
    clean = np.asarray([float(value) for value in values if value is not None and
                        np.isfinite(value)], dtype=np.float64)
    if not len(clean):
        return dict(count=0, median_ms=None, p95_ms=None)
    return dict(count=int(len(clean)), median_ms=float(np.median(clean)),
                p95_ms=float(np.percentile(clean, 95)))


def _initial_record(output_root, plans):
    rows = []
    for plan in plans:
        row = dict(plan)
        row.update(state='pending', arms=[])
        rows.append(row)
    return dict(
        schema_version=1,
        scope='Frozen R5 CPU-only source-observability pose ablation; no forwards or renders',
        status='running', complete=False, output_root=str(Path(output_root).resolve()),
        spec_path=str(SPEC_PATH), spec_sha256=EXPECTED_SPEC_SHA256,
        expected_r1_capture_sha256=EXPECTED_CAPTURE_SHA256,
        expected_r3_closed_report_sha256=EXPECTED_ZERO_SHA256,
        expected_r4_ceiling_sha256=EXPECTED_CEILING_SHA256,
        planned_conditions=12, planned_fit_arms=24, forward_calls=0, render_calls=0,
        reference_or_annotations_loaded=False, neural_or_model_calls=0,
        source_selection_frozen_before_flow_and_evaluator_truth=False,
        all_twelve_closed_conditions_loaded_before_cpu_fitting=False,
        working_buffer_budget_bytes=MAX_WORKING_BYTES,
        output_budget_bytes=MAX_OUTPUT_BYTES,
        conditions=rows, selection={}, visible_truth_sets={}, preflight=[],
        started_unix=time.time())


def _serialized_record(value):
    return json.dumps(value, indent=2, allow_nan=False).encode('utf-8')


def _write_report(output_root, record, limit_bytes=MAX_OUTPUT_BYTES):
    root = Path(output_root).resolve()
    target = root / 'pose_ablation.json'
    payload = _serialized_record(record)
    current_size = target.stat().st_size if target.exists() else 0
    other_bytes = sum(path.stat().st_size for path in root.rglob('*')
                      if path.is_file() and path.resolve() != target.resolve())
    projected_total = other_bytes + len(payload)
    if projected_total > int(limit_bytes):
        raise OutputLimitError(
            f'R5 output would use {projected_total} bytes, over {int(limit_bytes)}')
    temp = target.with_suffix(target.suffix + '.tmp')
    if temp.exists():
        raise FileExistsError(f'Refusing to replace unfinished terminal file {temp}')
    with temp.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    return dict(path=str(target), bytes=len(payload), sha256=digest(target),
                total_output_bytes=other_bytes + len(payload), budget_bytes=int(limit_bytes))


def _unavailable_arm(plan, arm, reason):
    return dict(
        arm=arm, fit_state='unavailable', fit_reason=str(reason),
        accepted_pose_state='unavailable', accepted_pose=None, proposal_pose=None,
        proposal_rotation_error=dict(rotation_error_degrees=None,
                                     translation_error_m=None,
                                     translation_error_fraction_D=None),
        proposal_surface_projection_720=dict(
            state='unavailable_candidate_pose', denominator=dict(count=0,
                ordered_source_rows_sha256=None, ordered_source_ids_sha256=None),
            invalid_candidate_projection_count=0, median_720=None, p95_720=None,
            max_720=None, finite_only=None),
        accepted_pose_surface_projection_720=dict(state='no_accepted_pose',
            denominator=dict(count=0, ordered_source_rows_sha256=None,
                             ordered_source_ids_sha256=None),
            invalid_candidate_projection_count=0, median_720=None, p95_720=None,
            max_720=None, finite_only=None),
        validation_state='not_run', validation_reason=str(reason), validation_stats=None,
        retained=dict(retained_source_ids=dict(count=0, sha256=_array_digest(np.empty(0, dtype=np.int64)),
                                               sample_source_ids=[], truncated=False),
                      sampled_source_ids=dict(count=0, sha256=_array_digest(np.empty(0, dtype=np.int64)),
                                              sample_source_ids=[], truncated=False),
                      ransac_inlier_source_ids=dict(count=0, sha256=_array_digest(np.empty(0, dtype=np.int64)),
                                                    sample_source_ids=[], truncated=False),
                      retained_inlier_source_ids=dict(count=0, sha256=_array_digest(np.empty(0, dtype=np.int64)),
                                                      sample_source_ids=[], truncated=False),
                      retained_count=0, sampled_count=0, ransac_inlier_count=0,
                      retained_inlier_count=0),
        costs_ms=dict(primary_fit=None, deterministic_repeat=None),
        deterministic_repeat=dict(matched=False, primary_sha256=None, repeated_sha256=None))


def _account_unfinished_rows(record, reason):
    for row in record['conditions']:
        existing = {arm.get('arm') for arm in row.get('arms', [])}
        for arm in ARMS:
            if arm not in existing:
                row.setdefault('arms', []).append(_unavailable_arm(row, arm, reason))
        if row.get('state') != 'complete':
            row['state'] = 'failed_or_unavailable'


def _report_failure(output_root, record, error):
    record['status'] = 'failed'
    record['complete'] = False
    record['failure'] = dict(type=type(error).__name__, message=str(error))
    _account_unfinished_rows(record, f'{type(error).__name__}: {error}')
    record['all_twelve_conditions_accounted'] = len(record.get('conditions', [])) == 12 and all(
        len(row.get('arms', [])) == 2 for row in record['conditions'])
    record['all_24_arms_accounted'] = sum(len(row.get('arms', [])) for row in record['conditions']) == 24
    record['failed_unix'] = time.time()
    return _write_report(output_root, record)


def _gate_baseline_preservation(bundle):
    rows = bundle['ceiling'].get('rows', [])
    return dict(
        r1_capture_sha256=bundle['capture_sha256'], r3_closed_report_sha256=bundle['zero_sha256'],
        r4_closed_ceiling_sha256=bundle['ceiling_sha256'],
        r4_candidate_reachable=bool(bundle['ceiling'].get('all_three_potentially_reachable')),
        r4_all_three_dense_identity_gate_remains_failed=not bool(
            bundle['ceiling'].get('all_three_potentially_reachable')),
        r4_rows=[{key: row.get(key) for key in (
            'frame_id', 'baseline_correct', 'confident_visible_denominator',
            'optimistic_correct_fraction_ceiling', 'reachable_under_loose_upper_bound')}
                  for row in rows],
        r1_original_dense_and_patch_scores_preserved=True,
        no_historical_pnp_fit_replayed=True)


def load_closed_conditions_only(capture_root=CAPTURE_ROOT, zero_root=ZERO_ROOT):
    """Regression entry point: load and hash all twelve rows, never fitting."""
    plans = _expected_condition_plan()
    bundle = _load_bound_manifests(capture_root, zero_root)
    selections = freeze_all_source_selections(bundle, plans)
    visible_sets = freeze_visible_evaluation_sets(bundle, selections, plans)
    preflight = preflight_all_conditions(bundle, plans, selections, visible_sets)
    baselines = _load_original_dense_baselines(bundle, plans)
    for plan in plans:
        condition_id = plan['condition_id']
        denominator = (baselines[condition_id].get('visible_denominator') or {})
        visible = visible_sets[condition_id]
        if (int(denominator.get('count', -1)) != int(visible['count']) or
                denominator.get('source_indices_sha256') not in
                (visible['source_id_sha256'], None)):
            raise ValueError(f'R5 frozen V differs from the original dense denominator: {condition_id}')
    revalidation = _frozen_input_revalidation(bundle)
    if revalidation['state'] != 'matched':
        raise ValueError('Closed packet inputs changed during loader-only regression')
    return dict(
        condition_ids=[row['condition_id'] for row in plans],
        source_selection={key: {name: value for name, value in data.items() if name != 'eligible'}
                          for key, data in selections.items()},
        visible_truth_sets={key: {name: value for name, value in data.items()
                                  if name not in ('visible_rows', 'visible_source_ids', 'truth_pose_crop_m')}
                            for key, data in visible_sets.items()},
        original_dense_baselines=baselines,
        preflight=preflight, input_revalidation=revalidation,
        fit_calls=0, forward_calls=0, render_calls=0,
        all_twelve_loaded_before_any_fit=len(preflight) == 12,
        source_selection_frozen_before_forward_loading=True)


def run_ablation(capture_root=CAPTURE_ROOT, zero_root=ZERO_ROOT, output_root=OUTPUT_ROOT):
    plans = _expected_condition_plan()
    output_root = Path(output_root).expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f'R5 output root already exists; preserving it: {output_root}')
    diagnostics_root = (CACHE / 'diagnostics').resolve()
    resolved_output_root = output_root.resolve()
    if not resolved_output_root.is_relative_to(diagnostics_root):
        raise ValueError(f'R5 output must be a fresh private directory under {diagnostics_root}')
    output_root.mkdir(parents=True, exist_ok=False)
    output_root = output_root.resolve()
    record = _initial_record(output_root, plans)
    try:
        _write_report(output_root, record)
        bundle = _load_bound_manifests(capture_root, zero_root)
        record.update(
            capture_root=str(bundle['capture_root']), zero_root=str(bundle['zero_root']),
            capture_sha256=bundle['capture_sha256'], zero_view_report_sha256=bundle['zero_sha256'],
            object_bbox_diagonal_m=float(bundle['capture']['object_bbox_diagonal_m']),
            model_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            source_pins=bundle['source_pins'],
            provenance=dict(r1_capture_scope=R1_CAPTURE_SCOPE, r3_scope=R3_REPORT_SCOPE,
                            captured_conditions=12, original_capture_packet_bytes=int(
                                bundle['capture']['packet_bytes']),
                            r3_packet_bytes=int(bundle['zero_report']['packet_bytes'])),
            original_dense_gates=_gate_baseline_preservation(bundle))
        _write_report(output_root, record)

        selections = freeze_all_source_selections(bundle, plans)
        record['source_selection_frozen_before_flow_and_evaluator_truth'] = True
        record['selection'] = _selection_report(selections)
        record['selection_cost_ms'] = _summary_cost(
            selection['elapsed_ms'] for selection in selections.values())
        record['selection_template_count'] = len(selections)
        _write_report(output_root, record)

        visible_sets = freeze_visible_evaluation_sets(bundle, selections, plans)
        record['visible_truth_sets'] = {
            condition_id: {key: value for key, value in data.items()
                           if key not in ('visible_rows', 'visible_source_ids', 'truth_pose_crop_m')}
            | {'truth_pose_crop_m': data['truth_pose_crop_m'].tolist()}
            for condition_id, data in visible_sets.items()}
        if len(visible_sets) != 12:
            raise AssertionError('Frozen all-surface evaluation V does not cover the twelve rows')
        _write_report(output_root, record)

        preflight = preflight_all_conditions(bundle, plans, selections, visible_sets)
        record['preflight'] = preflight
        record['all_twelve_closed_conditions_loaded_before_cpu_fitting'] = bool(len(preflight) == 12)
        record['preflight_max_estimated_working_bytes'] = max(
            row['estimated_working_bytes'] for row in preflight)
        _write_report(output_root, record)

        diagonal_m = float(bundle['capture']['object_bbox_diagonal_m'])
        row_by_id = {row['condition_id']: row for row in record['conditions']}
        baselines = {}
        run_error = None
        for index, plan in enumerate(plans):
            row = row_by_id[plan['condition_id']]
            try:
                payload = _load_condition_payload(bundle, plan)
                selection = selections[_source_template_refs(
                    plan['frame_id'], plan['template_offset_deg'])]
                visible_set = visible_sets[plan['condition_id']]
                baseline = _legacy_dense_baseline(bundle, payload)
                baselines[plan['condition_id']] = baseline
                row['original_dense_baseline'] = baseline
                row['frozen_truth_set_sha256'] = visible_set['source_id_sha256']
                row['frozen_truth_set_count'] = visible_set['count']
                row['loaded_provenance'] = payload['provenance']
                row['state'] = 'fitting'
                _check_working_budget(payload)
                for arm_name in ARMS:
                    try:
                        arm_result = _arm_result(payload, selection, visible_set, arm_name, diagonal_m)
                    except BaseException as fit_error:
                        arm_result = _unavailable_arm(
                            plan, arm_name, f'{type(fit_error).__name__}: {fit_error}')
                        arm_result['fit_state'] = 'execution_error'
                        arm_result['accepted_pose_state'] = 'unavailable'
                        arm_result['accepted_pose'] = None
                        arm_result['proposal_pose'] = None
                        record.setdefault('fit_failures', []).append(dict(
                            condition_id=plan['condition_id'], arm=arm_name,
                            type=type(fit_error).__name__, message=str(fit_error)))
                    row['arms'].append(arm_result)
                    _write_report(output_root, record)
                row['state'] = 'complete' if len(row['arms']) == 2 else 'failed_or_unavailable'
                del payload
            except BaseException as row_error:
                run_error = run_error or row_error
                row['state'] = 'failed_or_unavailable'
                row['failure'] = dict(type=type(row_error).__name__, message=str(row_error))
                for arm_name in ARMS:
                    if arm_name not in {item.get('arm') for item in row.get('arms', [])}:
                        row['arms'].append(_unavailable_arm(
                            plan, arm_name, f'{type(row_error).__name__}: {row_error}'))
                for pending in plans[index + 1:]:
                    pending_row = row_by_id[pending['condition_id']]
                    pending_row['state'] = 'unavailable'
                    pending_row['unavailable_reason'] = 'prior_row_failed; no stale/seed pose fallback'
                    if not pending_row['arms']:
                        pending_row['arms'] = [
                            _unavailable_arm(pending, arm, 'prior_row_failed') for arm in ARMS]
                break

        record['original_dense_baselines'] = baselines
        record['costs_ms'] = dict(
            primary_fit=_summary_cost(arm.get('costs_ms', {}).get('primary_fit')
                                      for row in record['conditions'] for arm in row.get('arms', [])),
            deterministic_repeat=_summary_cost(arm.get('costs_ms', {}).get('deterministic_repeat')
                                               for row in record['conditions'] for arm in row.get('arms', [])),
            source_selection=record.get('selection_cost_ms'))
        record['fit_arm_outcome_count'] = sum(len(row.get('arms', [])) for row in record['conditions'])
        record['all_twelve_conditions_accounted'] = len(record['conditions']) == 12 and all(
            len(row.get('arms', [])) == 2 for row in record['conditions'])
        record['all_24_arms_accounted'] = record['fit_arm_outcome_count'] == 24
        record['immutable_input_revalidation'] = _frozen_input_revalidation(bundle)
        record['exploratory_pose_prerequisite'] = _exploratory_pose_prerequisite(
            record, visible_sets, baselines, diagonal_m)
        record['independent_real_accuracy_verified'] = False
        record['real_pose_accuracy_claimed'] = False
        record['default_or_runtime_promotion'] = False
        if record['immutable_input_revalidation']['state'] != 'matched':
            run_error = run_error or ValueError('One or more immutable closed inputs changed during R5')
        record['complete'] = bool(run_error is None and record['all_twelve_conditions_accounted'] and
                                  record['all_24_arms_accounted'] and
                                  record['all_twelve_closed_conditions_loaded_before_cpu_fitting'] and
                                  record['immutable_input_revalidation']['state'] == 'matched')
        record['status'] = 'complete' if record['complete'] else 'failed'
        record['completed_unix'] = time.time()
        if run_error is not None:
            record['failure'] = dict(type=type(run_error).__name__, message=str(run_error))
        record['output'] = _write_report(output_root, record)
        if run_error is not None:
            raise run_error
        print(json.dumps(dict(status=record['status'], complete=record['complete'],
                              all_24_arms_accounted=record['all_24_arms_accounted'],
                              exploratory_pose_prerequisite=record['exploratory_pose_prerequisite']['passed'],
                              output_root=str(output_root), output_bytes=record['output']['bytes']),
                         indent=2), flush=True)
        return record
    except BaseException as error:
        if record.get('status') == 'complete':
            raise
        try:
            record['terminal_failure_preserved'] = True
            record['output'] = _report_failure(output_root, record, error)
        except OutputLimitError:
            compact = dict(
                schema_version=1, status='failed', complete=False,
                scope='R5 terminal failure accounting; oversized report replaced by compact terminal record',
                failure=dict(type=type(error).__name__, message=str(error)),
                planned_conditions=12, planned_fit_arms=24,
                conditions=[dict(condition_id=row['condition_id'], state=row.get('state'),
                                 arms=[dict(arm=arm['arm'], fit_state=arm.get('fit_state'),
                                            accepted_pose_state=arm.get('accepted_pose_state'))
                                       for arm in row.get('arms', [])])
                            for row in record.get('conditions', [])],
                all_twelve_conditions_accounted=all(len(row.get('arms', [])) == 2
                                                    for row in record.get('conditions', [])),
                terminal_failure_preserved=True)
            _write_report(output_root, compact)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=CAPTURE_ROOT,
                        help='Immutable completed R1 bottle packet root')
    parser.add_argument('--zero-root', type=Path, default=ZERO_ROOT,
                        help='Closed six-forward R3 zero-view report and packet root')
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT,
                        help='Fresh private output root; existing roots are preserved')
    args = parser.parse_args(argv)
    run_ablation(args.capture_root, args.zero_root, args.output_root)


if __name__ == '__main__':
    main()
