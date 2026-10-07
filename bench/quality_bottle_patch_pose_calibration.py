"""R6 CPU diagnostic: calibrate current-image patch endpoints before pose fitting.

The source selector is inherited byte-for-byte from the closed R5 diagnostic.
All source selectors are frozen and hash-checked before this module loads a
forward packet or evaluator truth. Matching accepts captured RGB and the
observed crop mask only. A single failed required calibration gate stops every
new fit and leaves all twelve fixed condition rows accounted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import quality_bottle_identity_audit as audit
from . import quality_bottle_pose_ablation as r5
from . import quality_bottle_zero_view_probe as zero_probe
from .quality_assets import CACHE, ROOT, digest
from .quality_contract import Frame
from .vision import cv2


CAPTURE_ROOT = CACHE / 'diagnostics' / 'bottle-identity-v2'
ZERO_ROOT = CACHE / 'diagnostics' / 'bottle-zero-view-calibration-v3'
R5_ROOT = CACHE / 'diagnostics' / 'bottle-source-observability-pose-ablation-v1'
R5_REPORT_PATH = R5_ROOT / 'pose_ablation.json'
R5_TERMINAL_PATH = CACHE / 'diagnostics' / 'bottle-r5-parent-v1' / 'terminal.json'
OUTPUT_ROOT = CACHE / 'diagnostics' / 'bottle-patch-pose-calibration-v1'
SPEC_PATH = ROOT / 'docs' / 'bottle-patch-pose-calibration-spec.md'

EXPECTED_SPEC_SHA256 = '761c9a4400548a4f9be1e3129e7b932bfb513960b6d1341d918c13f2b6621aaa'
EXPECTED_R1_CAPTURE_SHA256 = '33efa32778c4094a5df0690d49658cb3bfb0be522204b3dc4e1279d9f0d03063'
EXPECTED_R3_REPORT_SHA256 = 'dccbff28e6ea84693fe2a64d40006304f318532daefcff955646827b6129819d'
EXPECTED_R5_REPORT_SHA256 = 'a394bf54f51ddf527549ee5b3b6c7e8f623546b2c277a49051df2ce1c4d95259'
EXPECTED_R5_TERMINAL_SHA256 = '7d193b09d6d9dda241e039fce757b0a19d5c82e0fc6bc84b287d73e8915392ae'
EXPECTED_R5_SOURCE_SHA256 = '54bdd2f4ca4fb82551492e5397ab42cea2e464c5dad1f83360f7b23d33926542'
EXPECTED_R5_TEST_SHA256 = '4faf00469e154fe56e9352cca375674a848d6e21104340d52cd51d7ac685cc91'
EXPECTED_SHARED_AUDIT_SHA256 = '954d4b4e3a460187963894988cce53a85a077abaf07e085a0833131d14418f6a'
EXPECTED_R5_SPEC_SHA256 = '4bb91eb3782ef689b97b14f0cae68860768b39348526ced21af25ec5cfb78340'
EXPECTED_R4_CEILING_SHA256 = 'a0c21af577a0cd14c454899a2ab9509b4e52f0a78e948f93be318c7ae1ec7169'
EXPECTED_CHECKPOINT_SHA256 = 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'

CARRIERS = (10, 50, 100)
PATCH_SIZE = 11
PATCH_RADIUS = 5
HIGH_PASS_SIGMA = 2.0
PATCH_STD_MIN = 0.005
CONFIDENCE_MIN_EXCLUSIVE = 0.3
NCC_MIN = 0.75
MARGIN_MIN = 0.10
DISTANT_IDENTITY_FRACTION = 0.1
SPATIAL_COMPETITOR_DISTANCE_PX = 3.0
SEARCH_RADIUS = 8
SEARCH_OFFSETS = np.asarray(
    [(dx, dy) for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
     for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)], dtype=np.int64)
SEARCH_OFFSET_COUNT = (2 * SEARCH_RADIUS + 1) ** 2
MAX_MATCH_BATCH = 8
MAX_WORKING_BYTES = 128 * 1024**2
MAX_OUTPUT_BYTES = 8 * 1024**2
MAX_IDS_PER_SAMPLE = 16
FIXED_BANK_RESERVATION_BYTES = 8 * 1024**2
FROZEN_METADATA_RESERVATION_BYTES = 4 * 1024**2
# Each target pixel can be live in several coordinate/index/sample arrays in
# the vectorized bilinear sampler. 128 bytes per scalar patch pixel covers the
# known NumPy temporaries plus the score descriptor scratch conservatively.
MAX_SEARCH_SCRATCH_BYTES = MAX_MATCH_BATCH * SEARCH_OFFSET_COUNT * PATCH_SIZE**2 * 128
STATUS_NAMES = (
    'source_ineligible', 'nonfinite_endpoint_or_confidence', 'low_confidence',
    'footprint_rejection', 'low_target_texture', 'missing_distant_competitor',
    'spatial_ambiguity', 'ncc_or_distant_margin_rejection',
    'verified_unchanged', 'verified_corrected',
)
STATUS_CODES = {name: index for index, name in enumerate(STATUS_NAMES)}

SOURCE_SELECTOR_HASHES_T0 = {
    'template-0010-000': 'bb7bc76794761c656eabe8c4859d4f39c3c3fda203c19aa59eaa07cc34715f28',
    'template-0050-000': 'f400a1a65d4fbb99a002b93c127b834546e65d0487af02ef3ad4ffa7494342d3',
    'template-0100-000': '139200954c4a458ff122113dcb7c26c7252b3824b4864e89642f6f27f636a903',
}


class WorkingSetLimitError(ValueError):
    """The frozen row and worst-case live buffers exceed the working cap."""


class OutputLimitError(ValueError):
    """The private result would exceed the frozen output cap."""


class CalibrationInputError(ValueError):
    """A frozen cache, source, hash, or shape did not match its pin."""


def _array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order='C')).hexdigest()


def _canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_r6_bundle(capture_root=CAPTURE_ROOT, zero_root=ZERO_ROOT,
                    r5_report_path=R5_REPORT_PATH,
                    r5_terminal_path=R5_TERMINAL_PATH):
    """Bind R1/R3/R5 closed artifacts and the exact reviewed R5 sources."""
    if digest(SPEC_PATH).casefold() != EXPECTED_SPEC_SHA256:
        raise CalibrationInputError('R6 frozen specification hash changed')
    source_paths = {
        'bench/quality_bottle_pose_ablation.py': ROOT / 'bench' / 'quality_bottle_pose_ablation.py',
        'bench/test_quality_bottle_pose_ablation.py': ROOT / 'bench' / 'test_quality_bottle_pose_ablation.py',
        'bench/quality_bottle_identity_audit.py': ROOT / 'bench' / 'quality_bottle_identity_audit.py',
        'docs/bottle-identity-next-experiment.md': ROOT / 'docs' / 'bottle-identity-next-experiment.md',
        'bench/quality_bottle_patch_pose_calibration.py': ROOT / 'bench' / 'quality_bottle_patch_pose_calibration.py',
        'bench/test_quality_bottle_patch_pose_calibration.py': ROOT / 'bench' / 'test_quality_bottle_patch_pose_calibration.py',
    }
    expected_sources = {
        'bench/quality_bottle_pose_ablation.py': EXPECTED_R5_SOURCE_SHA256,
        'bench/test_quality_bottle_pose_ablation.py': EXPECTED_R5_TEST_SHA256,
        'bench/quality_bottle_identity_audit.py': EXPECTED_SHARED_AUDIT_SHA256,
        'docs/bottle-identity-next-experiment.md': EXPECTED_R5_SPEC_SHA256,
    }
    actual_sources = {name: digest(path).casefold() for name, path in source_paths.items()}
    for name, expected in expected_sources.items():
        if actual_sources[name] != expected:
            raise CalibrationInputError(f'Pinned R5 source changed: {name}')

    bundle = r5._load_bound_manifests(capture_root, zero_root)
    if bundle['capture_sha256'].casefold() != EXPECTED_R1_CAPTURE_SHA256:
        raise CalibrationInputError('R1 capture does not match the R6 pin')
    if bundle['zero_sha256'].casefold() != EXPECTED_R3_REPORT_SHA256:
        raise CalibrationInputError('R3 report does not match the R6 pin')

    r5_report_path = Path(r5_report_path).expanduser().resolve()
    r5_terminal_path = Path(r5_terminal_path).expanduser().resolve()
    if not r5_report_path.is_file() or digest(r5_report_path).casefold() != EXPECTED_R5_REPORT_SHA256:
        raise CalibrationInputError('Closed R5 pose ablation report hash changed')
    if not r5_terminal_path.is_file() or digest(r5_terminal_path).casefold() != EXPECTED_R5_TERMINAL_SHA256:
        raise CalibrationInputError('Closed R5 parent terminal receipt hash changed')
    report = json.loads(r5_report_path.read_text(encoding='utf-8'))
    terminal = json.loads(r5_terminal_path.read_text(encoding='utf-8'))
    expected_r5_pins = {
        'bench/quality_bottle_pose_ablation.py': EXPECTED_R5_SOURCE_SHA256,
        'bench/test_quality_bottle_pose_ablation.py': EXPECTED_R5_TEST_SHA256,
        'bench/quality_bottle_identity_audit.py': EXPECTED_SHARED_AUDIT_SHA256,
        'docs/bottle-identity-next-experiment.md': EXPECTED_R5_SPEC_SHA256,
    }
    if (report.get('complete') is not True or report.get('status') != 'complete' or
            report.get('all_twelve_conditions_accounted') is not True or
            report.get('all_24_arms_accounted') is not True or
            report.get('spec_sha256', '').casefold() != EXPECTED_R5_SPEC_SHA256 or
            report.get('capture_sha256', '').casefold() != EXPECTED_R1_CAPTURE_SHA256 or
            report.get('zero_view_report_sha256', '').casefold() != EXPECTED_R3_REPORT_SHA256):
        raise CalibrationInputError('Closed R5 report does not satisfy its pinned terminal state')
    if (terminal.get('state') != 'terminal' or terminal.get('exit_code') != 0 or
            terminal.get('source_freeze_matched') is not True or
            terminal.get('source_pins') != terminal.get('source_pins_after')):
        raise CalibrationInputError('R5 parent receipt is not a clean terminal source freeze')
    for name, expected in expected_r5_pins.items():
        if terminal.get('source_pins', {}).get(name, '').casefold() != expected:
            raise CalibrationInputError(f'R5 receipt does not bind expected source: {name}')
    if len(report.get('conditions', [])) != 12 or len(report.get('selection', {})) != 6:
        raise CalibrationInputError('Closed R5 report has the wrong condition or selector count')
    expected_ids = {row['condition_id'] for row in r5._expected_condition_plan()}
    condition_ids = {row.get('condition_id') for row in report['conditions']}
    if condition_ids != expected_ids:
        raise CalibrationInputError('Closed R5 report condition IDs differ from the frozen twelve')
    for row in report['conditions']:
        arms = {arm.get('arm'): arm for arm in row.get('arms', [])}
        if set(arms) != {'all', 'source_observable'}:
            raise CalibrationInputError(f'R5 controls are incomplete for {row["condition_id"]}')

    bundle['r5_report'] = report
    bundle['r5_report_path'] = r5_report_path
    bundle['r5_terminal_path'] = r5_terminal_path
    bundle['r5_terminal'] = terminal
    bundle['r5_report_sha256'] = EXPECTED_R5_REPORT_SHA256
    bundle['r5_terminal_sha256'] = EXPECTED_R5_TERMINAL_SHA256
    bundle['r5_source_pins'] = expected_r5_pins
    bundle['r6_source_pins'] = {
        **actual_sources,
        'docs/bottle-patch-pose-calibration-spec.md': EXPECTED_SPEC_SHA256,
        'R1 capture.json': EXPECTED_R1_CAPTURE_SHA256,
        'R3 zero_view.json': EXPECTED_R3_REPORT_SHA256,
        'R5 pose_ablation.json': EXPECTED_R5_REPORT_SHA256,
        'R5 terminal.json': EXPECTED_R5_TERMINAL_SHA256,
    }
    bundle['input_ledger'][str(SPEC_PATH.resolve())] = dict(
        bytes=SPEC_PATH.stat().st_size, sha256=EXPECTED_SPEC_SHA256)
    for name, path in source_paths.items():
        bundle['input_ledger'][str(path.resolve())] = dict(
            bytes=path.stat().st_size,
            sha256=expected_sources.get(name, actual_sources[name]))
    bundle['input_ledger'][str(r5_report_path)] = dict(
        bytes=r5_report_path.stat().st_size, sha256=EXPECTED_R5_REPORT_SHA256)
    bundle['input_ledger'][str(r5_terminal_path)] = dict(
        bytes=r5_terminal_path.stat().st_size, sha256=EXPECTED_R5_TERMINAL_SHA256)
    return bundle


def _freeze_and_bind_source_selections(bundle, plans=None):
    """Run exactly R5's source-only selector before any flow/truth read."""
    plans = r5._expected_condition_plan() if plans is None else plans
    selections = r5.freeze_all_source_selections(bundle, plans)
    expected = bundle['r5_report'].get('selection') or {}
    if set(selections) != set(expected) or len(selections) != 6:
        raise CalibrationInputError('R6 did not recompute exactly the six closed R5 selectors')
    summary = {}
    for template_id in sorted(selections):
        selection = selections[template_id]
        closed = expected[template_id]
        actual_hash = selection['eligible_array']['sha256'].casefold()
        closed_hash = closed['eligible_array']['sha256'].casefold()
        if actual_hash != closed_hash:
            raise CalibrationInputError(f'R6 source selector differs from closed R5: {template_id}')
        if int(selection['eligible_count']) != int(closed['eligible_count']):
            raise CalibrationInputError(f'R6 source selector count differs from R5: {template_id}')
        if template_id in SOURCE_SELECTOR_HASHES_T0 and actual_hash != SOURCE_SELECTOR_HASHES_T0[template_id]:
            raise CalibrationInputError(f'R6 0-degree selector differs from the frozen design: {template_id}')
        summary[template_id] = {
            'eligible_count': int(selection['eligible_count']),
            'total_sources': int(selection['total_sources']),
            'eligible_array_sha256': actual_hash,
            'source_indices_sha256': selection['source_indices_sha256'],
            'source_pixels_sha256': selection['source_pixels_sha256'],
            'source_array_hashes': selection['source_array_hashes'],
            'packet_path': selection['packet_path'],
            'packet_sha256': selection['packet_sha256'],
            'packet_bytes': int(selection['packet_bytes']),
        }
    bundle['source_selection_frozen_before_flow_or_evaluator_truth'] = True
    bundle['source_selections'] = selections
    bundle['source_selection_summary'] = summary
    return selections


def _load_fixed_banks(bundle):
    """Rebuild the original max-64 banks from their pinned source packets."""
    banks = {}
    for frame_id in CARRIERS:
        for offset in (0, 180):
            plan = dict(condition_id=f'template-{frame_id:04d}-{offset:03d}',
                        frame_id=frame_id, template_offset_deg=offset)
            entry, arrays = r5._condition_source_only_inputs(bundle, plan)
            bank = audit.verify_fixed_patch_bank(entry, arrays)
            if int(bank.get('selected_anchor_count', -1)) > 64:
                raise CalibrationInputError(f'Frozen R5 patch bank exceeds 64 anchors: {entry["context_id"]}')
            path = r5._safe_entry_path(bundle['capture_root'], entry)
            bundle['input_ledger'][str(path)] = dict(bytes=int(entry['bytes']), sha256=entry['sha256'])
            banks[entry['context_id']] = dict(entry=entry, anchors=bank['anchors'],
                                              manifest_sha256=_canonical_json_sha256(
                                                  entry['fixed_patch_bank']),
                                              selected_anchor_count=int(bank['selected_anchor_count']))
            del arrays
    if len(banks) != 6:
        raise CalibrationInputError('R6 failed to bind all six original 0/180 patch banks')
    return banks


def _entry_array_nbytes(entry):
    total = 0
    for name, meta in (entry.get('arrays') or {}).items():
        shape = tuple(int(value) for value in meta.get('shape', ()))
        if not shape or any(value < 0 for value in shape):
            raise CalibrationInputError(f'Malformed shape for {entry.get("path")}:{name}')
        try:
            itemsize = np.dtype(meta['dtype']).itemsize
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationInputError(f'Malformed dtype for {entry.get("path")}:{name}') from error
        total += int(math.prod(shape)) * int(itemsize)
    return total


def _condition_entries(bundle, plan):
    """Resolve packet metadata without reading flow or evaluator values."""
    frame_id = int(plan['frame_id'])
    template_id = r5._source_template_refs(frame_id, int(plan['template_offset_deg']))
    template_entry = r5._context_entry(bundle, template_id)
    frame_entry = r5._context_entry(bundle, r5._source_frame_ref(frame_id))
    if plan['kind'] == 'self_control':
        condition = bundle['zero_rows'].get(plan['condition_id'])
        forward = bundle['zero_entries'].get(plan['condition_id'])
        query_entry = None
        forward_root = bundle['zero_root']
    else:
        condition = bundle['r1_conditions'].get(plan['condition_id'])
        forward = bundle['r1_forwards'].get(plan['condition_id'])
        refs = (condition or {}).get('context_refs') or {}
        query_entry = r5._context_entry(bundle, refs.get('query'))
        forward_root = bundle['capture_root']
    if condition is None or forward is None:
        raise CalibrationInputError(f'Frozen R6 condition is missing: {plan["condition_id"]}')
    entries = [(bundle['capture_root'], template_entry), (bundle['capture_root'], frame_entry)]
    if query_entry is not None:
        entries.append((bundle['capture_root'], query_entry))
    entries.append((forward_root, forward))
    return entries


def _unique_array_owner_bytes(*values):
    """Count each live NumPy owning buffer once across nested containers."""
    seen_containers, seen_owners = set(), set()
    total = 0

    def visit(value):
        nonlocal total
        if isinstance(value, np.ndarray):
            owner = value
            while isinstance(getattr(owner, 'base', None), np.ndarray):
                owner = owner.base
            key = id(owner)
            if key not in seen_owners:
                seen_owners.add(key)
                total += int(owner.nbytes)
            return
        if isinstance(value, dict):
            key = id(value)
            if key in seen_containers:
                return
            seen_containers.add(key)
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            key = id(value)
            if key in seen_containers:
                return
            seen_containers.add(key)
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    return int(total)


def _retained_fit_input_upper_bound(bundle, plans, selections):
    """Reserve worst-case verified fit arrays for all twelve rows up front."""
    total = 0
    rows = []
    for plan in plans:
        template_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        entries = _condition_entries(bundle, plan)
        template_arrays = (entries[0][1].get('arrays') or {})
        forward_arrays = (entries[-1][1].get('arrays') or {})
        try:
            source_count = int(template_arrays['source_indices']['shape'][0])
            point_shape = tuple(int(x) for x in template_arrays['source_points_object_m']['shape'])
            pixel_shape = tuple(int(x) for x in template_arrays['source_pixels_xy']['shape'])
            mask_shape = tuple(int(x) for x in template_arrays['observed_crop_mask']['shape'])
            conf_shape = tuple(int(x) for x in forward_arrays['confidence']['shape'])
            conf_dtype = np.dtype(forward_arrays['confidence']['dtype'])
            eligible_count = int(selections[template_id]['eligible_count'])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise CalibrationInputError(f'Malformed retained fit-input descriptors for {plan["condition_id"]}') from error
        if (source_count < 0 or point_shape != (source_count, 3) or
                pixel_shape != (source_count, 2) or len(mask_shape) != 2 or
                conf_shape != (280, 280) or conf_dtype.kind != 'f' or conf_dtype.itemsize > 8 or
                eligible_count < 0 or eligible_count > source_count):
            raise CalibrationInputError(f'Invalid retained fit-input shapes/dtypes for {plan["condition_id"]}')
        # X(3*f64), uv(2*f64), confidence(f64 upper bound), source IDs and
        # source rows (i64 each); source and target observed masks are copied.
        mask_bytes = int(math.prod(mask_shape))
        candidate_bytes = eligible_count * 64 + 2 * mask_bytes
        total += candidate_bytes
        rows.append(dict(condition_id=plan['condition_id'], source_count=source_count,
                         verified_rows_upper_bound=eligible_count,
                         candidate_input_upper_bound_bytes=int(candidate_bytes)))
    return dict(total_bytes=int(total), rows=rows,
                derivation='sum over all twelve rows with K<=frozen eligible_count; 64 bytes/K plus two observed-mask copies')


def _visible_set_upper_bound(bundle, plans):
    total = 0
    for plan in plans:
        entries = _condition_entries(bundle, plan)
        try:
            count = int(entries[0][1]['arrays']['source_indices']['shape'][0])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise CalibrationInputError(f'Malformed frozen V source descriptor for {plan["condition_id"]}') from error
        total += count * 16 + 512  # int64 visible rows + copied int64 source IDs + small pose/meta
    return int(total)


def _estimated_working_bytes(entries, selector_count, resident_state_bytes=0,
                             retained_fit_input_bytes=0, reserved_state_bytes=0):
    """Conservative bound for one active row plus all live/future resident buffers."""
    base = sum(_entry_array_nbytes(entry) for _, entry in entries)
    template_entry = entries[0][1]
    try:
        source_count = int(template_entry['arrays']['source_indices']['shape'][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise CalibrationInputError('Source index descriptor is missing or malformed') from error
    descriptor_bytes = int(selector_count) * PATCH_SIZE**2 * 8
    # Upper bound covers float64 bilinear coordinates, four footprint indices,
    # target patches/descriptors, validity, query high-pass and all candidate
    # pose/evaluator arrays live while one row is measured.
    # Covers converted source X/pixels/IDs, original and candidate endpoints,
    # confidence/status buffers, verified output arrays and hypothetical field.
    endpoint_field_bytes = source_count * 160
    # Both evaluator array sets can coexist with full-map pose-Jacobian
    # temporary matrices and their SVD inputs before the row is released.
    evaluator_bytes = source_count * 700
    query_scratch_bytes = 12 * 280 * 280 * 8
    additional = (descriptor_bytes + endpoint_field_bytes + evaluator_bytes +
                  query_scratch_bytes + MAX_SEARCH_SCRATCH_BYTES + 4 * 1024**2)
    total = int(base + additional + resident_state_bytes + retained_fit_input_bytes +
                reserved_state_bytes)
    if total > MAX_WORKING_BYTES:
        raise WorkingSetLimitError(
            f'One frozen R6 condition needs at most {total} bytes, over {MAX_WORKING_BYTES}')
    return dict(input_array_bytes=int(base), source_descriptor_bytes=int(descriptor_bytes),
                endpoint_field_bytes=int(endpoint_field_bytes), evaluator_bytes=int(evaluator_bytes),
                query_scratch_bytes=int(query_scratch_bytes),
                candidate_scratch_bytes=int(MAX_SEARCH_SCRATCH_BYTES),
                resident_unique_owner_bytes=int(resident_state_bytes),
                all_twelve_retained_fit_inputs_bytes=int(retained_fit_input_bytes),
                reserved_future_state_bytes=int(reserved_state_bytes),
                fixed_safety_reserve_bytes=4 * 1024**2,
                estimated_peak_bytes=total, budget_bytes=MAX_WORKING_BYTES)


def _preflight_working_sets(bundle, plans, selections, resident_state_bytes=0,
                            reserved_state_bytes=0):
    retained = _retained_fit_input_upper_bound(bundle, plans, selections)
    result = []
    for plan in plans:
        template_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        budget = _estimated_working_bytes(
            _condition_entries(bundle, plan), selections[template_id]['eligible_count'],
            resident_state_bytes=resident_state_bytes,
            retained_fit_input_bytes=retained['total_bytes'],
            reserved_state_bytes=reserved_state_bytes)
        result.append(dict(condition_id=plan['condition_id'],
                           source_template_id=template_id, working_set=budget))
    return dict(rows=result, retained_fit_inputs=retained,
                resident_unique_owner_bytes=int(resident_state_bytes),
                reserved_future_state_bytes=int(reserved_state_bytes),
                max_estimated_peak_bytes=max(row['working_set']['estimated_peak_bytes']
                                             for row in result))


def _source_selector_working_preflight(bundle, plans):
    """Bound R5 selector live arrays before allocating any of six selectors."""
    templates = {}
    for plan in plans:
        template_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        templates.setdefault(template_id, _condition_entries(bundle, plan)[0][1])
    retained_boolean_bytes = 0
    largest_active = 0
    for entry in templates.values():
        arrays = entry.get('arrays') or {}
        try:
            source_count = int(arrays['source_indices']['shape'][0])
            active = sum(_entry_array_nbytes(dict(arrays={name: meta}))
                         for name, meta in arrays.items())
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise CalibrationInputError('Malformed source selector descriptors') from error
        retained_boolean_bytes += source_count
        # RGB luminance/high-pass, blur, distance transform, common/eroded
        # masks and loop temporaries. The source packet arrays are counted once.
        selector_scratch = 4 * 280 * 280 * 4 + 4 * 280 * 280
        largest_active = max(largest_active, active + selector_scratch + retained_boolean_bytes)
    peak = int(largest_active + FROZEN_METADATA_RESERVATION_BYTES)
    if peak > MAX_WORKING_BYTES:
        raise WorkingSetLimitError(
            f'Source-only selector needs at most {peak} bytes, over {MAX_WORKING_BYTES}')
    return dict(max_estimated_peak_bytes=peak,
                selector_boolean_reservation_bytes=int(retained_boolean_bytes),
                source_packet_and_scratch_peak_bytes=int(largest_active),
                metadata_reservation_bytes=FROZEN_METADATA_RESERVATION_BYTES,
                budget_bytes=MAX_WORKING_BYTES)


def _condition_query_rgb(bundle, payload):
    """Return only the captured network-input RGB array and its pinned hash."""
    plan = payload['plan']
    if plan['kind'] == 'self_control':
        source = payload['source']
        common_mask = (np.asarray(source['template_mask'], dtype=bool) &
                       np.asarray(source['observed_crop_mask'], dtype=bool))
        if plan['self_condition'] == 'full':
            query_rgb = source['template_rgb']
        else:
            query_rgb = np.array(source['template_rgb'], dtype=np.float32, copy=True)
            query_rgb[~common_mask] = np.float32(0.5)
        condition = bundle['zero_rows'][plan['condition_id']]
        query_hash = (condition.get('input_hashes') or {}).get('query_rgb')
        template_hash = (condition.get('input_hashes') or {}).get('template_rgb')
        if template_hash != audit.array_info(source['template_rgb']):
            raise CalibrationInputError(f'R3 template network input hash changed: {plan["condition_id"]}')
        if query_hash != audit.array_info(query_rgb):
            raise CalibrationInputError(f'R3 self network input RGB hash changed: {plan["condition_id"]}')
        return query_rgb, query_hash['sha256'], 'captured R3 full/clipped self network input'
    query = payload['query']
    if not isinstance(query, dict) or 'query_rgb_rgb' not in query:
        raise CalibrationInputError(f'R1 synthetic RGB input is missing: {plan["condition_id"]}')
    query_rgb = np.asarray(query['query_rgb_rgb'])
    if (query_rgb.dtype != np.float32 or query_rgb.ndim != 3 or query_rgb.shape[2] != 3 or
            not np.isfinite(query_rgb).all()):
        raise CalibrationInputError(f'R1 synthetic RGB input is malformed: {plan["condition_id"]}')
    if query_rgb.shape[:2] != np.asarray(payload['source']['observed_crop_mask']).shape:
        raise CalibrationInputError(f'R1 synthetic RGB dimensions differ from crop: {plan["condition_id"]}')
    return query_rgb, audit.array_sha256(query_rgb), 'captured R1 synthetic query RGB'


def preflight_actual_cache(capture_root=CAPTURE_ROOT, zero_root=ZERO_ROOT,
                           r5_report_path=R5_REPORT_PATH,
                           r5_terminal_path=R5_TERMINAL_PATH):
    """Load all twelve bound rows through the matcher boundary; no matching or fits."""
    plans = r5._expected_condition_plan()
    bundle = _load_r6_bundle(capture_root, zero_root, r5_report_path, r5_terminal_path)
    selector_working = _source_selector_working_preflight(bundle, plans)
    selections = _freeze_and_bind_source_selections(bundle, plans)
    selector_bytes = _unique_array_owner_bytes(selections)
    early_working = _preflight_working_sets(
        bundle, plans, selections, resident_state_bytes=selector_bytes,
        reserved_state_bytes=(_visible_set_upper_bound(bundle, plans) +
                              FIXED_BANK_RESERVATION_BYTES + FROZEN_METADATA_RESERVATION_BYTES))
    banks = _load_fixed_banks(bundle)
    bank_owner_bytes = _unique_array_owner_bytes(banks)
    if bank_owner_bytes > FIXED_BANK_RESERVATION_BYTES:
        raise WorkingSetLimitError(
            f'Frozen fixed banks use {bank_owner_bytes} bytes, over their {FIXED_BANK_RESERVATION_BYTES}-byte reservation')
    preflight_rows = []
    working_by_id = {row['condition_id']: row for row in early_working['rows']}
    for plan in plans:
        source_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
        budget = working_by_id[plan['condition_id']]['working_set']
        payload = r5._load_condition_payload(bundle, plan)
        query_rgb, query_rgb_sha, query_role = _condition_query_rgb(bundle, payload)
        source = payload['source']
        observed = np.asarray(source['observed_crop_mask'], dtype=bool)
        if not np.array_equal(observed, payload['frame']['observed_crop_mask']):
            raise CalibrationInputError(f'Observed crop mask differs inside R6 row {plan["condition_id"]}')
        if np.asarray(payload['forward']['flow']).dtype != np.float32 or \
                np.asarray(payload['forward']['confidence']).dtype != np.float32:
            raise CalibrationInputError(f'R6 forward arrays have unexpected dtypes: {plan["condition_id"]}')
        endpoint_sha = _array_sha256(payload['endpoints'])
        confidence_sha = _array_sha256(payload['confidence'])
        camera_hashes = _camera_hashes(payload)
        preflight_rows.append(dict(
            condition_id=plan['condition_id'], state='loaded_before_matcher',
            template_id=source_id, eligible_source_count=int(selections[source_id]['eligible_count']),
            query_rgb_sha256=query_rgb_sha, query_rgb_role=query_role,
            observed_crop_mask_sha256=audit.array_sha256(observed),
            flow_sha256=audit.array_sha256(payload['forward']['flow']),
            confidence_field_sha256=audit.array_sha256(payload['forward']['confidence']),
            original_endpoint_sha256=endpoint_sha, original_confidence_sha256=confidence_sha,
            seed=int(payload['seed']), seed_sha256=_canonical_json_sha256(int(payload['seed'])),
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            source_provenance=payload['provenance'],
            camera_hashes=camera_hashes,
            working_set=budget,
            source_fixed_bank_sha256=banks[source_id]['manifest_sha256'],
            opposite_fixed_bank_sha256=banks[
                r5._source_template_refs(plan['frame_id'], 180 - int(plan['template_offset_deg']))
            ]['manifest_sha256'],
        ))
        if (_array_sha256(payload['endpoints']) != endpoint_sha or
                _array_sha256(payload['confidence']) != confidence_sha):
            raise AssertionError(f'R6 loader modified original endpoint evidence: {plan["condition_id"]}')
        del query_rgb, payload
    if len(preflight_rows) != 12 or {row['condition_id'] for row in preflight_rows} != {
            plan['condition_id'] for plan in plans}:
        raise CalibrationInputError('R6 actual-cache loader did not account for all twelve rows')
    revalidation = r5._frozen_input_revalidation(bundle)
    if revalidation['state'] != 'matched':
        raise CalibrationInputError('R6 input pins changed during loader-only preflight')
    return dict(
        condition_ids=[plan['condition_id'] for plan in plans],
        r6_source_pins=bundle['r6_source_pins'],
        source_selection=bundle['source_selection_summary'],
        fixed_bank_count=len(banks), preflight=preflight_rows,
        source_selector_working_preflight=selector_working,
        all_row_working_preflight=early_working,
        resident_selection_owner_bytes=selector_bytes,
        resident_fixed_bank_owner_bytes=bank_owner_bytes,
        max_estimated_working_bytes=max(row['working_set']['estimated_peak_bytes']
                                         for row in preflight_rows),
        input_revalidation=revalidation, matcher_calls=0, fitter_calls=0,
        source_selection_frozen_before_flow_or_evaluator_truth=True,
        matching_started=False, fitting_started=False)


def _normalize_patch_float64(patch):
    value = np.asarray(patch, dtype=np.float64)
    if value.shape != (PATCH_SIZE, PATCH_SIZE) or not np.isfinite(value).all():
        return None, None
    mean = float(value.mean())
    centered = value - mean
    std = float(np.sqrt(np.mean(centered * centered)))
    norm = float(np.sqrt(np.sum(centered * centered, dtype=np.float64)))
    if not np.isfinite(std) or std < PATCH_STD_MIN or not np.isfinite(norm) or norm <= 1e-12:
        return None, std
    return centered / norm, std


def build_source_descriptors(source_arrays, selection):
    """Compute high-pass source descriptors after R5 eligibility is frozen."""
    required = ('template_rgb', 'template_mask', 'observed_crop_mask', 'source_indices',
                'source_pixels_xy', 'source_points_object_m')
    if any(name not in source_arrays for name in required):
        raise CalibrationInputError('R6 source descriptor packet lacks a required source array')
    ids = np.asarray(source_arrays['source_indices'], dtype=np.int64).reshape(-1)
    pixels = np.asarray(source_arrays['source_pixels_xy'], dtype=np.float64).reshape(-1, 2)
    points = np.asarray(source_arrays['source_points_object_m'], dtype=np.float64).reshape(-1, 3)
    eligible = np.asarray(selection['eligible'], dtype=bool).reshape(-1)
    rgb = np.asarray(source_arrays['template_rgb'])
    common = (np.asarray(source_arrays['template_mask'], dtype=bool) &
              np.asarray(source_arrays['observed_crop_mask'], dtype=bool))
    if not (len(ids) == len(pixels) == len(points) == len(eligible)):
        raise CalibrationInputError('R6 source selector and source map lengths differ')
    if rgb.dtype != np.float32 or rgb.shape[:2] != common.shape or rgb.shape[2:] != (3,):
        raise CalibrationInputError('R6 source RGB or common source mask has invalid dimensions')
    order = np.argsort(ids, kind='stable')
    if not np.array_equal(ids[order], ids) or len(np.unique(ids)) != len(ids):
        raise CalibrationInputError('R6 source indices must be unique and in ascending order')
    high = audit._highpass(audit._gray_image(rgb), HIGH_PASS_SIGMA)
    eroded = audit._eroded_mask(common, 6)
    rows = np.flatnonzero(eligible)
    descriptors = np.empty((len(rows), PATCH_SIZE**2), dtype=np.float64)
    descriptor_rows = np.full(len(ids), -1, dtype=np.int32)
    stds = np.empty(len(rows), dtype=np.float64)
    for descriptor_index, row in enumerate(rows.tolist()):
        patch, _ = audit._bilinear_patch(high, pixels[row], eroded)
        descriptor, std = _normalize_patch_float64(patch) if patch is not None else (None, None)
        if descriptor is None:
            raise CalibrationInputError(
                f'R5-eligible source patch failed R6 descriptor construction at source {int(ids[row])}')
        descriptors[descriptor_index] = descriptor.reshape(-1)
        descriptor_rows[row] = descriptor_index
        stds[descriptor_index] = std
    if len(rows) != int(selection['eligible_count']):
        raise AssertionError('R6 source descriptor ledger differs from frozen selector count')
    return dict(rows=rows.astype(np.int64, copy=False), source_ids=ids[rows].copy(),
                descriptors=descriptors, row_to_descriptor=descriptor_rows,
                source_std=stds, highpass_sha256=_array_sha256(high),
                descriptor_sha256=_array_sha256(descriptors),
                source_array_hashes={name: audit.array_sha256(source_arrays[name])
                                     for name in required})


def _query_highpass(query_rgb):
    value = np.asarray(query_rgb)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Current-image matcher requires captured float32 RGB')
    if not np.isfinite(value).all():
        raise ValueError('Current-image RGB contains nonfinite values')
    return audit._highpass(audit._gray_image(value), HIGH_PASS_SIGMA)


def _bilinear_patch_batch(image, endpoints_xy, eroded_mask):
    """Original four-tap sampler applied in batches of at most eight sources."""
    value = np.asarray(image)
    mask = np.asarray(eroded_mask, dtype=bool)
    endpoints = np.asarray(endpoints_xy, dtype=np.float64).reshape(-1, 2)
    if value.ndim != 2 or mask.shape != value.shape:
        raise ValueError('Batched patch image and eroded mask dimensions differ')
    if not np.isfinite(endpoints).all():
        raise ValueError('Batched patch coordinates must be finite')
    if len(endpoints) > MAX_MATCH_BATCH:
        raise ValueError('Batched current-image patch search exceeds eight sources')
    batch = len(endpoints)
    offsets = SEARCH_OFFSETS.astype(np.float64)
    centers = endpoints[:, None, :] + offsets[None, :, :]
    if np.any(np.abs(centers) > 1e9):
        return (np.zeros((batch, SEARCH_OFFSET_COUNT, PATCH_SIZE**2), dtype=np.float32),
                np.zeros((batch, SEARCH_OFFSET_COUNT), dtype=bool))
    patch_offsets = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=np.float64)
    patch_x, patch_y = np.meshgrid(patch_offsets, patch_offsets)
    xx = centers[:, :, 0, None, None] - 0.5 + patch_x[None, None, :, :]
    yy = centers[:, :, 1, None, None] - 0.5 + patch_y[None, None, :, :]
    x0 = np.floor(xx).astype(np.int64)
    y0 = np.floor(yy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    h, w = value.shape
    in_bounds = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    xc0, xc1 = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    yc0, yc1 = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    wx, wy = xx - x0, yy - y0
    samples = (value[yc0, xc0] * (1.0 - wx) * (1.0 - wy) +
               value[yc0, xc1] * wx * (1.0 - wy) +
               value[yc1, xc0] * (1.0 - wx) * wy +
               value[yc1, xc1] * wx * wy)
    inside = in_bounds & mask[yc0, xc0] & mask[yc0, xc1] & mask[yc1, xc0] & mask[yc1, xc1]
    full = np.all(inside, axis=(2, 3))
    return np.asarray(samples, dtype=np.float32).reshape(batch, SEARCH_OFFSET_COUNT, -1), full


def _score_candidate_batch(query_highpass, target_eroded_mask, q0_batch, source_desc_batch):
    patches, footprints = _bilinear_patch_batch(query_highpass, q0_batch, target_eroded_mask)
    batch, count, width = patches.shape
    flat = patches.reshape(batch * count, width).astype(np.float64)
    centered = flat - flat.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(centered * centered, axis=1, dtype=np.float64))
    norm = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float64))
    target_ok = np.isfinite(std) & (std >= PATCH_STD_MIN) & np.isfinite(norm) & (norm > 1e-12)
    descriptors = np.zeros_like(centered, dtype=np.float64)
    descriptors[target_ok] = centered[target_ok] / norm[target_ok, None]
    scores = np.full(batch * count, -np.inf, dtype=np.float64)
    src = np.asarray(source_desc_batch, dtype=np.float64).reshape(batch, width)
    products = descriptors.reshape(batch, count, width) * src[:, None, :]
    scores[target_ok] = np.sum(products.reshape(batch * count, width)[target_ok],
                               axis=1, dtype=np.float64)
    valid = footprints.reshape(-1) & target_ok
    scores[~valid] = -np.inf
    return dict(scores=scores.reshape(batch, count), target_std=std.reshape(batch, count),
                descriptors=descriptors.reshape(batch, count, width),
                footprint_valid=footprints, target_valid=target_ok.reshape(batch, count))


def _score_candidate_scalar(query_highpass, target_eroded_mask, q0, source_descriptor):
    valid = np.zeros(SEARCH_OFFSET_COUNT, dtype=bool)
    std = np.full(SEARCH_OFFSET_COUNT, np.nan, dtype=np.float64)
    scores = np.full(SEARCH_OFFSET_COUNT, -np.inf, dtype=np.float64)
    descriptors = np.zeros((SEARCH_OFFSET_COUNT, PATCH_SIZE**2), dtype=np.float64)
    footprints = np.zeros(SEARCH_OFFSET_COUNT, dtype=bool)
    for index, (dx, dy) in enumerate(SEARCH_OFFSETS.tolist()):
        endpoint = np.asarray(q0, dtype=np.float64) + np.asarray((dx, dy), dtype=np.float64)
        patch, _ = audit._bilinear_patch(query_highpass, endpoint, target_eroded_mask)
        if patch is None:
            continue
        footprints[index] = True
        descriptor, patch_std = _normalize_patch_float64(patch)
        if descriptor is None:
            std[index] = np.nan if patch_std is None else patch_std
            continue
        valid[index] = True
        std[index] = patch_std
        descriptors[index] = descriptor.reshape(-1)
        scores[index] = float(np.sum(
            np.asarray(source_descriptor, dtype=np.float64).reshape(-1) *
            descriptors[index], dtype=np.float64))
    return dict(scores=scores, target_std=std, descriptors=descriptors,
                footprint_valid=footprints,
                target_valid=valid)


def _bank_descriptor_arrays(banks):
    anchors = []
    for bank in banks:
        anchors.extend(bank.get('anchors', []))
    if not anchors:
        return dict(anchors=[], descriptors=np.empty((0, PATCH_SIZE**2), dtype=np.float64),
                    object_points_m=np.empty((0, 3), dtype=np.float64), source_ids=np.empty(0, dtype=np.int64))
    descriptors, object_points, source_ids = [], [], []
    for anchor in anchors:
        descriptor = np.asarray(anchor.get('descriptor'), dtype=np.float64).reshape(-1)
        point = np.asarray(anchor.get('object_xyz_m'), dtype=np.float64).reshape(-1)
        if (descriptor.shape != (PATCH_SIZE**2,) or point.shape != (3,) or
                not np.isfinite(descriptor).all() or not np.isfinite(point).all()):
            raise CalibrationInputError('Frozen fixed-bank anchor descriptor or point is malformed')
        descriptors.append(descriptor)
        object_points.append(point)
        source_ids.append(int(anchor['source_index']))
    return dict(anchors=anchors, descriptors=np.asarray(descriptors, dtype=np.float64),
                object_points_m=np.asarray(object_points, dtype=np.float64),
                source_ids=np.asarray(source_ids, dtype=np.int64))


def _source_ledger(status_codes, source_ids):
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    codes = np.asarray(status_codes, dtype=np.uint8).reshape(-1)
    if len(ids) != len(codes) or (len(ids) and not np.all(ids[:-1] < ids[1:])):
        raise ValueError('Source outcome ledger must cover unique ascending source IDs')
    dtype = np.dtype([('source_id', '<i8'), ('status_code', 'u1')], align=False)
    ledger = np.empty(len(ids), dtype=dtype)
    ledger['source_id'] = ids
    ledger['status_code'] = codes
    samples = {}
    for name, code in STATUS_CODES.items():
        chosen = ids[codes == code]
        samples[name] = dict(count=int(len(chosen)),
                             first_source_ids=chosen[:MAX_IDS_PER_SAMPLE].tolist(),
                             last_source_ids=(chosen[-MAX_IDS_PER_SAMPLE:].tolist()
                                              if len(chosen) > MAX_IDS_PER_SAMPLE else []))
    return dict(source_count=int(len(ids)), ordered_source_ids_sha256=_array_sha256(ids),
                ordered_source_outcome_sha256=hashlib.sha256(ledger.tobytes(order='C')).hexdigest(),
                status_counts={name: int(np.count_nonzero(codes == code))
                               for name, code in STATUS_CODES.items()},
                bounded_status_samples=samples)


def _resolve_candidate(scores, target_valid, footprint_valid, target_std,
                        descriptors, bank_data, source_xyz_m, diagonal_m):
    valid_indices = np.flatnonzero(target_valid & np.isfinite(scores))
    if not np.any(footprint_valid):
        return dict(status='footprint_rejection')
    if not len(valid_indices):
        return dict(status='low_target_texture')
    maximum = float(np.max(scores[valid_indices]))
    maxima = valid_indices[scores[valid_indices] == maximum]
    winner_index = min(maxima.tolist(), key=lambda index: (
        int(SEARCH_OFFSETS[index, 0] ** 2 + SEARCH_OFFSETS[index, 1] ** 2),
        int(SEARCH_OFFSETS[index, 1]), int(SEARCH_OFFSETS[index, 0])))
    dx, dy = (int(value) for value in SEARCH_OFFSETS[winner_index])
    candidate_position = np.asarray((dx, dy), dtype=np.float64)
    separated = np.linalg.norm(SEARCH_OFFSETS[valid_indices].astype(np.float64) -
                               candidate_position[None, :], axis=1) >= SPATIAL_COMPETITOR_DISTANCE_PX
    spatial_indices = valid_indices[separated]
    if len(spatial_indices):
        best_spatial = float(np.max(scores[spatial_indices]))
        spatial_margin = float(maximum - best_spatial)
    else:
        best_spatial, spatial_margin = None, None

    xyz = np.asarray(source_xyz_m, dtype=np.float64).reshape(3)
    distant = (np.linalg.norm(bank_data['object_points_m'] - xyz[None, :], axis=1) >
               DISTANT_IDENTITY_FRACTION * float(diagonal_m))
    distant_rows = np.flatnonzero(distant)
    winner_descriptor = descriptors[winner_index]
    if not len(distant_rows):
        distant_score, distant_margin = None, None
    else:
        distant_scores = bank_data['descriptors'][distant_rows] @ winner_descriptor
        distant_score = float(np.max(distant_scores))
        distant_margin = float(maximum - distant_score)

    winner_tied = len(maxima) > 1
    if not len(distant_rows):
        state = 'missing_distant_competitor'
    elif winner_tied or spatial_margin is None or spatial_margin < MARGIN_MIN:
        state = 'spatial_ambiguity'
    elif maximum < NCC_MIN or distant_margin is None or distant_margin < MARGIN_MIN:
        state = 'ncc_or_distant_margin_rejection'
    else:
        state = 'verified'
    return dict(status=state, winning_offset_xy=[dx, dy], winning_endpoint_offset_xy=[dx, dy],
                winner_candidate_index=int(winner_index), winner_ncc=maximum,
                winner_tie_count=int(len(maxima)), spatial_competitor_ncc=best_spatial,
                spatial_margin=spatial_margin, distant_competitor_ncc=distant_score,
                distant_margin=distant_margin, valid_candidate_count=int(len(valid_indices)),
                target_std=float(target_std[winner_index]))


def match_current_image_patches(query_rgb, observed_crop_mask, original_endpoints_crop,
                                original_confidence, source_indices, source_points_m,
                                eligible_source, source_descriptors, fixed_banks, diagonal_m,
                                batch_size=MAX_MATCH_BATCH):
    """Verify/refine every R5-eligible endpoint using observed RGB and mask only.

    Query pose, depth, synthetic masks, annotations and projections are absent
    from this interface by construction. Endpoints retain their fractional phase.
    """
    rgb = np.asarray(query_rgb)
    observed = np.asarray(observed_crop_mask, dtype=bool)
    endpoints = np.asarray(original_endpoints_crop, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(original_confidence).reshape(-1)
    ids = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    points = np.asarray(source_points_m, dtype=np.float64).reshape(-1, 3)
    eligible = np.asarray(eligible_source, dtype=bool).reshape(-1)
    batch_size = int(batch_size)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.float32 or
            observed.shape != rgb.shape[:2] or not np.isfinite(rgb).all()):
        raise ValueError('Current-image matcher received malformed captured RGB or observed mask')
    if (not (len(ids) == len(points) == len(endpoints) == len(confidence) == len(eligible)) or
            len(ids) != len(source_descriptors['row_to_descriptor'])):
        raise ValueError('Current-image matcher arrays do not align with the frozen source map')
    if batch_size < 1 or batch_size > MAX_MATCH_BATCH:
        raise ValueError('Current-image matcher batch size must be in [1, 8]')
    if not np.isfinite(float(diagonal_m)) or float(diagonal_m) <= 0:
        raise ValueError('Current-image matcher requires a finite positive model diagonal')
    order = np.argsort(ids, kind='stable')
    if not np.array_equal(ids[order], ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Current-image matcher source IDs must be unique and ascending')
    if not np.array_equal(source_descriptors['rows'], np.flatnonzero(eligible)):
        raise ValueError('Source descriptor rows differ from the frozen eligible source selector')

    query_high = _query_highpass(rgb)
    target_eroded = audit._eroded_mask(observed, 6)
    bank_data = _bank_descriptor_arrays(fixed_banks)
    status_codes = np.full(len(ids), STATUS_CODES['source_ineligible'], dtype=np.uint8)
    selected_rows = np.flatnonzero(eligible)
    verified_rows_buffer = np.empty(len(selected_rows), dtype=np.int64)
    verified_endpoints_buffer = np.empty((len(selected_rows), 2), dtype=np.float64)
    verified_offsets_buffer = np.empty((len(selected_rows), 2), dtype=np.int16)
    verified_count = 0
    for start in range(0, len(selected_rows), batch_size):
        rows = selected_rows[start:start + batch_size]
        q0 = endpoints[rows]
        conf = np.asarray(confidence[rows], dtype=np.float64)
        desc_indices = source_descriptors['row_to_descriptor'][rows]
        if np.any(desc_indices < 0):
            raise CalibrationInputError('Frozen eligible source has no high-pass descriptor')
        desc = source_descriptors['descriptors'][desc_indices]
        for local_index, row in enumerate(rows.tolist()):
            if (not np.isfinite(q0[local_index]).all() or not np.isfinite(conf[local_index]) or
                    not np.isfinite(points[row]).all()):
                status_codes[row] = STATUS_CODES['nonfinite_endpoint_or_confidence']
            elif conf[local_index] <= CONFIDENCE_MIN_EXCLUSIVE:
                status_codes[row] = STATUS_CODES['low_confidence']

        searchable = np.asarray([
            local for local, row in enumerate(rows.tolist())
            if status_codes[row] == STATUS_CODES['source_ineligible']
        ], dtype=np.int64)
        if not len(searchable):
            continue
        batch_rows = rows[searchable]
        batch_q0 = q0[searchable]
        batch_desc = desc[searchable]
        scored = _score_candidate_batch(query_high, target_eroded, batch_q0, batch_desc)
        for batch_row, row in enumerate(batch_rows.tolist()):
            source_xyz = points[row]
            selected = _resolve_candidate(
                scored['scores'][batch_row], scored['target_valid'][batch_row],
                scored['footprint_valid'][batch_row], scored['target_std'][batch_row],
                scored['descriptors'][batch_row], bank_data, source_xyz, diagonal_m)
            state = selected.pop('status')
            if state == 'verified':
                offset = np.asarray(selected['winning_offset_xy'], dtype=np.int64)
                winner = q0[searchable[batch_row]] + offset.astype(np.float64)
                if not np.isfinite(winner).all():
                    status_codes[row] = STATUS_CODES['nonfinite_endpoint_or_confidence']
                    continue
                verified_rows_buffer[verified_count] = int(row)
                verified_endpoints_buffer[verified_count] = winner
                verified_offsets_buffer[verified_count] = offset.astype(np.int16)
                verified_count += 1
                status_codes[row] = STATUS_CODES[
                    'verified_unchanged' if np.array_equal(offset, np.zeros(2, dtype=np.int64))
                    else 'verified_corrected']
            elif state == 'footprint_rejection':
                status_codes[row] = STATUS_CODES[state]
            elif state == 'low_target_texture':
                status_codes[row] = STATUS_CODES[state]
            elif state == 'missing_distant_competitor':
                status_codes[row] = STATUS_CODES[state]
            elif state == 'spatial_ambiguity':
                status_codes[row] = STATUS_CODES[state]
            else:
                status_codes[row] = STATUS_CODES[state]
    verified_rows_array = verified_rows_buffer[:verified_count].copy()
    if verified_count:
        verified_order = np.argsort(ids[verified_rows_array], kind='stable')
        verified_rows_array = verified_rows_array[verified_order]
        winner_array = verified_endpoints_buffer[:verified_count][verified_order].copy()
        offset_array = verified_offsets_buffer[:verified_count][verified_order].copy()
    else:
        winner_array = np.empty((0, 2), dtype=np.float64)
        offset_array = np.empty((0, 2), dtype=np.int16)
    ledger = _source_ledger(status_codes, ids)
    correction = np.linalg.norm(winner_array - endpoints[verified_rows_array], axis=1)
    offset_histogram = np.zeros(SEARCH_OFFSET_COUNT, dtype=np.int64)
    for dx, dy in offset_array.tolist():
        offset_histogram[(int(dy) + SEARCH_RADIUS) * (2 * SEARCH_RADIUS + 1) +
                         int(dx) + SEARCH_RADIUS] += 1
    correction_summary = dict(
        count=int(len(correction)), unchanged=int(np.count_nonzero(correction == 0.0)),
        corrected=int(np.count_nonzero(correction > 0.0)),
        median_px=(float(np.median(correction)) if len(correction) else None),
        p95_px=(float(np.percentile(correction, 95)) if len(correction) else None),
        max_px=(float(np.max(correction)) if len(correction) else None),
        search_boundary_count=int(np.count_nonzero(np.any(np.abs(offset_array) == SEARCH_RADIUS, axis=1))),
        boundary_x_count=int(np.count_nonzero(np.abs(offset_array[:, 0]) == SEARCH_RADIUS)),
        boundary_y_count=int(np.count_nonzero(np.abs(offset_array[:, 1]) == SEARCH_RADIUS)),
        nonzero_offset_histogram=[dict(dx=int(dx), dy=int(dy), count=int(count))
                                  for (dx, dy), count in zip(SEARCH_OFFSETS.tolist(),
                                                             offset_histogram.tolist()) if count],
    )
    return dict(
        verified_rows=verified_rows_array, verified_endpoints=winner_array,
        verified_offsets_xy=offset_array, verified_confidence=np.asarray(
            confidence[verified_rows_array]).copy(), status_codes=status_codes,
        source_outcome_ledger=ledger,
        correction_summary=correction_summary,
        query_highpass_sha256=_array_sha256(query_high),
        query_observed_eroded_mask_sha256=_array_sha256(target_eroded),
        search=dict(radius_px=SEARCH_RADIUS, candidate_count=SEARCH_OFFSET_COUNT,
                    batch_size=int(batch_size), coordinate_rule='q0 plus integer dx/dy; q0 phase preserved',
                    tie_order=['displacement_squared', 'dy', 'dx'],
                    scalar_arithmetic='float64 coordinates, normalization and NCC'),
        matcher_input_hashes=dict(query_rgb_sha256=audit.array_sha256(rgb),
                                  observed_crop_mask_sha256=audit.array_sha256(observed),
                                  original_endpoints_sha256=_array_sha256(endpoints),
                                  original_confidence_sha256=_array_sha256(confidence),
                                  source_indices_sha256=_array_sha256(ids),
                                  source_points_m_sha256=_array_sha256(points),
                                  eligible_array_sha256=_array_sha256(eligible),
                                  source_descriptors_sha256=source_descriptors['descriptor_sha256']),
    )


def _matched_field(match_result, original_endpoints, original_confidence):
    endpoints = np.asarray(original_endpoints, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(original_confidence).reshape(-1)
    endpoint_hash, confidence_hash = _array_sha256(endpoints), _array_sha256(confidence)
    hypothetical = endpoints.copy()
    rows = np.asarray(match_result['verified_rows'], dtype=np.int64)
    if len(rows):
        hypothetical[rows] = np.asarray(match_result['verified_endpoints'], dtype=np.float64)
    if _array_sha256(endpoints) != endpoint_hash or _array_sha256(confidence) != confidence_hash:
        raise AssertionError('R6 hypothetical field construction modified original endpoint/confidence inputs')
    return hypothetical, endpoint_hash, confidence_hash


def _evaluator_inputs(payload):
    plan, source = payload['plan'], payload['source']
    if plan['kind'] == 'self_control':
        return dict(
            pose_crop_m=audit.crop_pose_from_context(source),
            depth_mm=np.asarray(source['template_depth_mm']),
            mask=np.asarray(source['template_mask'], dtype=bool) &
                 np.asarray(source['observed_crop_mask'], dtype=bool),
            mask_role='R3 common mask for evaluator only',
        )
    query = payload['query']
    return dict(
        pose_crop_m=np.asarray(query['known_query_pose_m'], dtype=np.float64),
        depth_mm=np.asarray(query['query_depth_mm']),
        mask=np.asarray(query['synthetic_query_mask'], dtype=bool),
        mask_role='captured synthetic foreground for evaluator only',
    )


def _visible_set_hash(source_ids, true_visible):
    visible_ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)[np.asarray(true_visible, dtype=bool)]
    return dict(count=int(len(visible_ids)), ordered_source_ids_sha256=_array_sha256(visible_ids))


def _conditional_visible_summary(source_ids, eligible, true_visible, verified_rows,
                                 correct_identity):
    ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    visible = np.asarray(true_visible, dtype=bool).reshape(-1)
    correct = np.asarray(correct_identity, dtype=bool).reshape(-1)
    if not (len(ids) == len(eligible) == len(visible) == len(correct)):
        raise ValueError('Conditional calibration denominator arrays differ in length')
    verified = np.zeros(len(ids), dtype=bool)
    rows = np.asarray(verified_rows, dtype=np.int64).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= len(ids)) or len(np.unique(rows)) != len(rows):
        raise ValueError('Verified source rows are outside the frozen source map')
    verified[rows] = True
    denominator = eligible & visible
    verified_visible = verified & visible
    denominator_count = int(denominator.sum())
    verified_count = int((verified_visible & eligible).sum())
    coverage = (float(verified_count / denominator_count) if denominator_count else None)
    correct_count = int((correct & verified_visible & eligible).sum())
    fraction = (float(correct_count / verified_count) if verified_count else None)
    summary = dict(
        eligible_true_visible_denominator=denominator_count,
        eligible_true_visible_source_ids_sha256=_array_sha256(ids[denominator]),
        verified_confident_true_visible_count=verified_count,
        verified_confident_true_visible_source_ids_sha256=_array_sha256(
            ids[verified_visible & eligible]),
        verified_confident_coverage=coverage,
        correct_verified_confident_true_visible=correct_count,
        correct_fraction_among_verified_confident_true_visible=fraction,
        thresholds=dict(min_verified_coverage=.50, min_correct_fraction=.90,
                        endpoint_error_crop_px=3., identity_distance_fraction=.02,
                        confidence_gt=.3),
        coverage_passed=bool(coverage is not None and coverage >= .50),
        correctness_passed=bool(fraction is not None and fraction >= .90),
        empty_eligible_V_unavailable=denominator_count == 0)
    return summary, denominator, verified_visible & eligible


def _repeat_unavailable_precondition_decision(primary, recompute):
    """Hash a repeated deterministic no-fit decision without entering the solver."""
    repeated = recompute()
    primary_sha = _canonical_json_sha256(primary)
    repeated_sha = _canonical_json_sha256(repeated)
    return dict(matched=primary_sha == repeated_sha,
                state=('matched_no_fit_unavailable_decision'
                       if primary_sha == repeated_sha else 'mismatched_no_fit_decision'),
                primary_sha256=primary_sha, repeated_sha256=repeated_sha,
                primary_decision=primary, repeated_decision=repeated)


def _camera_hashes(payload):
    source, frame = payload['source'], payload['frame']
    return dict(
        crop_k_sha256=_array_sha256(source['crop_k']),
        crop_from_native_sha256=_array_sha256(source['crop_from_native']),
        native_k_sha256=_array_sha256(frame['native_k']),
        observed_crop_mask_sha256=audit.array_sha256(source['observed_crop_mask']),
        observed_native_mask_sha256=audit.array_sha256(frame['native_mask'] > 0),
        native_frame_rgb_sha256=audit.array_sha256(frame['native_rgb']))


def _geometry_for_condition(source, observed_mask, eligible, match_result, diagonal_m):
    ids = np.asarray(source['source_indices'], dtype=np.int64).reshape(-1)
    pixels = np.asarray(source['source_pixels_xy'], dtype=np.float64).reshape(-1, 2)
    ix, iy, inside = audit._floor_indices(pixels, observed_mask.shape[1], observed_mask.shape[0])
    in_observed = np.zeros(len(ids), dtype=bool)
    rows = np.flatnonzero(inside)
    in_observed[rows] = observed_mask[iy[rows], ix[rows]]
    all_rows = np.flatnonzero(in_observed)
    eligible_rows = np.flatnonzero(np.asarray(eligible, dtype=bool))
    verified_rows = np.asarray(match_result['verified_rows'], dtype=np.int64)
    verified_endpoints = np.asarray(match_result['verified_endpoints'], dtype=np.float64)
    template_pose = audit.crop_pose_from_context(source)
    groups = {}
    for name, group_rows, current in (
            ('all_original', all_rows, None),
            ('source_observable', eligible_rows, None),
            ('verified', verified_rows, verified_endpoints)):
        groups[name] = _pose_jacobian_metrics(source, group_rows, observed_mask, current, diagonal_m)
    return dict(source_all_observed_count=int(len(all_rows)),
                source_observable_count=int(len(eligible_rows)), verified_count=int(len(verified_rows)),
                initial_template_pose_sha256=_array_sha256(template_pose), groups=groups)


def _pose_jacobian_metrics(source, rows, observed_mask, current_endpoints, diagonal_m):
    """R6 source-centered pose Jacobian and spatial support for a fixed set."""
    points = np.asarray(source['source_points_object_m'], dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(source['source_pixels_xy'], dtype=np.float64).reshape(-1, 2)
    chosen = np.asarray(rows, dtype=np.int64).reshape(-1)
    observed = np.asarray(observed_mask, dtype=bool)
    source_support = r5._spatial_support(pixels, chosen, observed)
    if current_endpoints is None:
        current_support = None
    else:
        endpoints = np.asarray(current_endpoints, dtype=np.float64).reshape(-1, 2)
        if len(endpoints) != len(chosen):
            raise ValueError('R6 measured endpoint set differs from verified source rows')
        current_support = r5._spatial_support(
            endpoints, np.arange(len(endpoints), dtype=np.int64), observed)
    if not len(chosen):
        return dict(point_count=0, source_support=source_support,
                    current_support=current_support, geometric_rank=0,
                    pose_jacobian_rank=0, pose_jacobian_singular_values=[],
                    pose_jacobian_condition=None, thickness_ratio=None,
                    weak_direction=None, status='empty_source_set')
    selected = points[chosen]
    if not np.isfinite(selected).all():
        raise CalibrationInputError('R6 geometry contains nonfinite object points')
    center = points.mean(axis=0)
    pose = audit.crop_pose_from_context(source)
    r = np.asarray(pose[:3, :3], dtype=np.float64)
    t = np.asarray(pose[:3, 3], dtype=np.float64)
    k = np.asarray(source['crop_k'], dtype=np.float64)
    camera = selected @ r.T + t
    z = camera[:, 2]
    if not np.isfinite(camera).all() or np.any(z <= 0) or not np.isfinite(k).all():
        return dict(point_count=int(len(chosen)), source_support=source_support,
                    current_support=current_support, geometric_rank=None,
                    pose_jacobian_rank=None, pose_jacobian_singular_values=[],
                    pose_jacobian_condition=None, thickness_ratio=None,
                    weak_direction=None, status='invalid_initial_template_projection')
    centered = selected - selected.mean(axis=0)
    point_singular = np.linalg.svd(centered / math.sqrt(len(selected)), compute_uv=False)
    if len(point_singular) < 3:
        point_singular = np.pad(point_singular, (0, 3 - len(point_singular)))
    geometric_rank = int(np.linalg.matrix_rank(centered))
    thickness = (float(point_singular[-1] / point_singular[0])
                 if len(point_singular) and point_singular[0] > 0 else None)
    h = np.zeros((len(chosen), 2, 3), dtype=np.float64)
    h[:, 0, 0] = float(k[0, 0]) / z
    h[:, 0, 2] = -float(k[0, 0]) * camera[:, 0] / (z * z)
    h[:, 1, 1] = float(k[1, 1]) / z
    h[:, 1, 2] = -float(k[1, 1]) * camera[:, 1] / (z * z)
    centered_source = selected - center
    skew = np.zeros((len(chosen), 3, 3), dtype=np.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -centered_source[:, 2], centered_source[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = centered_source[:, 2], -centered_source[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -centered_source[:, 1], centered_source[:, 0]
    rotation_block = np.einsum('ij,njk->nik', r, -skew)
    j_rotation = np.einsum('nij,njk->nik', h, rotation_block)
    j_translation = h * float(diagonal_m)
    jacobian = np.concatenate((j_rotation, j_translation), axis=2).reshape(-1, 6)
    jacobian /= math.sqrt(len(chosen))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    weak = (np.linalg.svd(jacobian, full_matrices=False)[2][-1].tolist() if rank < 6 else None)
    if len(singular) < 6:
        singular = np.pad(singular, (0, 6 - len(singular)))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return dict(point_count=int(len(chosen)), source_support=source_support,
                current_support=current_support, geometric_rank=geometric_rank,
                point_singular_values=point_singular.tolist(), pose_jacobian_rank=rank,
                pose_jacobian_singular_values=singular.tolist(),
                pose_jacobian_condition=condition, thickness_ratio=thickness,
                weak_direction=weak, status='measured')


def _condition_measurement(bundle, plan, payload, selection, banks, visible_set, r5_row,
                           diagonal_m):
    source = payload['source']
    observed = np.asarray(source['observed_crop_mask'], dtype=bool)
    query_rgb, query_rgb_sha, query_rgb_role = _condition_query_rgb(bundle, payload)
    source_bank_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
    competitor_offset = 180 - int(plan['template_offset_deg'])
    competitor_bank_id = r5._source_template_refs(plan['frame_id'], competitor_offset)
    source_bank, opposite_bank = banks[source_bank_id], banks[competitor_bank_id]
    descriptors = build_source_descriptors(source, selection)
    original_endpoint_hash = _array_sha256(payload['endpoints'])
    original_confidence_hash = _array_sha256(payload['confidence'])
    original_flow_hash = _array_sha256(payload['forward']['flow'])
    original_conf_field_hash = _array_sha256(payload['forward']['confidence'])
    match = match_current_image_patches(
        query_rgb, observed, payload['endpoints'], payload['confidence'],
        source['source_indices'], source['source_points_object_m'], selection['eligible'],
        descriptors, [source_bank, opposite_bank], diagonal_m)
    if (_array_sha256(payload['endpoints']) != original_endpoint_hash or
            _array_sha256(payload['confidence']) != original_confidence_hash or
            _array_sha256(payload['forward']['flow']) != original_flow_hash or
            _array_sha256(payload['forward']['confidence']) != original_conf_field_hash):
        raise AssertionError(f'R6 matcher mutated original forward evidence: {plan["condition_id"]}')
    hypothetical, original_endpoint_hash, original_confidence_hash = _matched_field(
        match, payload['endpoints'], payload['confidence'])
    truth = _evaluator_inputs(payload)
    original_metrics, original_metric_arrays = audit.synthetic_identity_metrics(
        source['source_points_object_m'], source['source_pixels_xy'], source['source_indices'],
        payload['endpoints'], payload['confidence'], truth['pose_crop_m'], truth['depth_mm'],
        truth['mask'], source['crop_k'], diagonal_m)
    hypothetical_metrics, hypothetical_arrays = audit.synthetic_identity_metrics(
        source['source_points_object_m'], source['source_pixels_xy'], source['source_indices'],
        hypothetical, payload['confidence'], truth['pose_crop_m'], truth['depth_mm'],
        truth['mask'], source['crop_k'], diagonal_m)
    actual_v = _visible_set_hash(source['source_indices'], original_metric_arrays['true_visible'])
    frozen_v = dict(count=int(visible_set['count']),
                    ordered_source_ids_sha256=visible_set['source_id_sha256'])
    if actual_v != frozen_v:
        raise CalibrationInputError(f'R6 evaluator V differs from R5 frozen V: {plan["condition_id"]}')
    if _visible_set_hash(source['source_indices'], hypothetical_arrays['true_visible']) != frozen_v:
        raise AssertionError('R6 hypothetical endpoint field changed the immutable V denominator')

    eligible = np.asarray(selection['eligible'], dtype=bool)
    visible = np.asarray(original_metric_arrays['true_visible'], dtype=bool)
    verified = np.zeros(len(eligible), dtype=bool)
    verified[np.asarray(match['verified_rows'], dtype=np.int64)] = True
    conditional_metrics, conditional_denominator, conditional_verified_visible = \
        _conditional_visible_summary(
            source['source_indices'], eligible, visible, match['verified_rows'],
            hypothetical_arrays['correct_identity'])
    conditional_count = int(conditional_denominator.sum())
    verified_visible_count = int(conditional_verified_visible.sum())

    hypothetical_flow = np.array(payload['forward']['flow'], dtype=np.float32, copy=True)
    for row, offset in zip(match['verified_rows'].tolist(), match['verified_offsets_xy'].tolist()):
        source_id = int(source['source_indices'][row])
        sy, sx = divmod(source_id, hypothetical_flow.shape[1])
        hypothetical_flow[sy, sx] = (payload['forward']['flow'][sy, sx] +
                                      np.asarray(offset, dtype=np.float32))
    fixed_patch = audit.audit_patch_identity(
        query_rgb, observed, hypothetical_flow, payload['forward']['confidence'],
        source_bank['anchors'] and {'anchors': source_bank['anchors'],
                                    'eligible_source_patch_count': source_bank['selected_anchor_count']}
        or {'anchors': [], 'eligible_source_patch_count': 0},
        [dict(anchors=source_bank['anchors']), dict(anchors=opposite_bank['anchors'])], diagonal_m)
    fixed_patch_summary = fixed_patch['summary']
    fixed_patch_passed = fixed_patch_summary['state'] == 'distinctive_current_image_support'
    legacy_fixed_patch = audit.audit_patch_identity(
        query_rgb, truth['mask'], hypothetical_flow, payload['forward']['confidence'],
        source_bank['anchors'] and {'anchors': source_bank['anchors'],
                                    'eligible_source_patch_count': source_bank['selected_anchor_count']}
        or {'anchors': [], 'eligible_source_patch_count': 0},
        [dict(anchors=source_bank['anchors']), dict(anchors=opposite_bank['anchors'])], diagonal_m)
    legacy_fixed_patch_summary = legacy_fixed_patch['summary']
    legacy_fixed_patch_passed = (
        legacy_fixed_patch_summary['state'] == 'distinctive_current_image_support')
    dense_thresholds = hypothetical_metrics['thresholds']
    availability_passed = hypothetical_metrics['confidence_availability_on_visible'] >= .50
    identity_passed = (hypothetical_metrics['correct_fraction_confident_visible'] is not None and
                       hypothetical_metrics['correct_fraction_confident_visible'] >= .90)
    hypothetical_dense_qualified = bool(
        availability_passed and identity_passed and legacy_fixed_patch_passed)
    original_dense = r5_row.get('original_dense_baseline') or {}
    original_dense_qualified = bool(original_dense.get('dense_qualification'))
    self_dense_retained = (hypothetical_dense_qualified if plan['kind'] == 'self_control' else None)

    geometry = _geometry_for_condition(source, observed, eligible, match, diagonal_m)
    status_code_counts = match['source_outcome_ledger']['status_counts']
    invisible_verified = verified & ~visible
    invisible_distance = hypothetical_arrays['identity_distance_fraction'][invisible_verified]
    invisible_distance = invisible_distance[np.isfinite(invisible_distance)]
    conditional_passed = bool(conditional_metrics['coverage_passed'] and
                              conditional_metrics['correctness_passed'])
    row_gates = dict(
        required_condition=plan['kind'] != 'synthetic_negative',
        conditional_coverage_passed=conditional_metrics['coverage_passed'],
        conditional_identity_passed=conditional_metrics['correctness_passed'],
        fixed_anchor_spatial_calibration_passed=fixed_patch_passed,
        self_full_V_dense_qualification_retained=(self_dense_retained if plan['kind'] == 'self_control'
                                                  else None),
        passed=(conditional_passed and fixed_patch_passed and
                (self_dense_retained if plan['kind'] == 'self_control' else True)),
    )
    row = dict(
        condition_id=plan['condition_id'], kind=plan['kind'], frame_id=int(plan['frame_id']),
        template_offset_deg=int(plan['template_offset_deg']),
        self_condition=plan.get('self_condition'), state='measured',
        seed=int(payload['seed']), seed_sha256=_canonical_json_sha256(int(payload['seed'])),
        source_template_id=source_bank_id,
        query_rgb=dict(role=query_rgb_role, shape=list(query_rgb.shape), dtype=str(query_rgb.dtype),
                       sha256=query_rgb_sha),
        observed_mask_role='captured observed_crop_mask only',
        observed_crop_mask_sha256=audit.array_sha256(observed),
        source_selector_sha256=selection['eligible_array']['sha256'],
        source_descriptor_sha256=descriptors['descriptor_sha256'],
        fixed_banks=dict(source_template_id=source_bank_id,
                         source_packet_sha256=source_bank['entry']['sha256'],
                         source_manifest_sha256=source_bank['manifest_sha256'],
                         source_anchor_count=source_bank['selected_anchor_count'],
                         opposite_template_id=competitor_bank_id,
                         opposite_packet_sha256=opposite_bank['entry']['sha256'],
                         opposite_manifest_sha256=opposite_bank['manifest_sha256'],
                         opposite_anchor_count=opposite_bank['selected_anchor_count']),
        endpoint_evidence=dict(original_endpoint_sha256=original_endpoint_hash,
                               original_confidence_sha256=original_confidence_hash,
                               original_flow_sha256=original_flow_hash,
                               original_confidence_field_sha256=original_conf_field_hash,
                               hypothetical_endpoint_sha256=_array_sha256(hypothetical),
                               hypothetical_confidence_sha256=_array_sha256(payload['confidence']),
                               unverified_endpoint_and_confidence_bytes_unchanged=True),
        source_outcome_ledger=match['source_outcome_ledger'],
        correction_summary=match['correction_summary'],
        matcher=dict(search=match['search'], input_hashes=match['matcher_input_hashes'],
                     query_highpass_sha256=match['query_highpass_sha256'],
                     eroded_observed_mask_sha256=match['query_observed_eroded_mask_sha256']),
        original_dense_measurements=copy.deepcopy(original_dense),
        hypothetical_full_field=dict(
            identity_metrics=hypothetical_metrics,
            visible_denominator=dict(count=actual_v['count'],
                                     ordered_source_ids_sha256=actual_v['ordered_source_ids_sha256']),
            dense_qualification=hypothetical_dense_qualified,
            dense_gates=dict(confidence_availability_passed=bool(availability_passed),
                             confident_visible_identity_passed=bool(identity_passed),
                             fixed_patch_identity_passed=bool(legacy_fixed_patch_passed),
                             qualified=hypothetical_dense_qualified)),
        conditional_observable_measurement=conditional_metrics,
        fixed_anchor_spatial_calibration=dict(
            summary=fixed_patch_summary, rows=fixed_patch['rows'],
            qualified=bool(fixed_patch_passed),
            mask_role='captured observed_crop_mask; spatial bbox and bilinear support',
            original_r3_qualification=original_dense_qualified),
        legacy_evaluator_mask_fixed_anchor_audit=dict(
            summary=legacy_fixed_patch_summary, rows=legacy_fixed_patch['rows'],
            qualified=bool(legacy_fixed_patch_passed), mask_role=truth['mask_role']),
        evaluator_only_evidence=dict(
            verified_confident_invisible_sources=int(invisible_verified.sum()),
            verified_invisible_median_identity_distance_fraction=(
                float(np.median(invisible_distance)) if len(invisible_distance) else None),
            verified_invisible_wrong_surface_count=int(np.count_nonzero(
                invisible_distance > .1)),
            visible_count=actual_v['count'],
            original_visible_source_ids_sha256=actual_v['ordered_source_ids_sha256']),
        geometry=geometry,
        camera_hashes=_camera_hashes(payload),
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        gates=row_gates,
        source_packet_provenance=payload['provenance'],
    )
    if plan['kind'] == 'synthetic_negative':
        expected_v = {10: 8, 50: 0, 100: 0}[int(plan['frame_id'])]
        row['evaluator_only_evidence']['expected_negative_V_count'] = expected_v
        row['evaluator_only_evidence']['negative_V_count_matches_frozen_cohort'] = (
            actual_v['count'] == expected_v)
        wrong = original_dense.get('original_wrong_surface_evidence') or {}
        row['evaluator_only_evidence']['original_wrong_surface_evidence'] = copy.deepcopy(wrong)
        row_gates['negative_V_count_matches_frozen_cohort'] = actual_v['count'] == expected_v
    if plan['kind'] == 'self_control':
        row_gates['original_r3_dense_gate_was_qualified'] = original_dense_qualified
    fit_inputs = dict(
        source_points_m=np.asarray(source['source_points_object_m'], dtype=np.float64)[match['verified_rows']].copy(),
        endpoints_crop=np.asarray(match['verified_endpoints'], dtype=np.float64).copy(),
        confidence=np.asarray(match['verified_confidence']).copy(),
        source_indices=np.asarray(source['source_indices'], dtype=np.int64)[match['verified_rows']].copy(),
        source_rows=np.asarray(match['verified_rows'], dtype=np.int64).copy(),
        source_mask=np.asarray(source['observed_crop_mask'], dtype=bool),
        target_mask=np.asarray(source['observed_crop_mask'], dtype=bool),
        input_endpoint_sha256=original_endpoint_hash,
        input_confidence_sha256=original_confidence_hash,
    )
    return row, fit_inputs


def _retained_candidate_rows(fit_inputs):
    source = dict(source_points_object_m=fit_inputs['source_points_m'],
                  source_indices=fit_inputs['source_indices'])
    return r5._filtered_observed_rows(source, fit_inputs['endpoints_crop'],
                                     fit_inputs['confidence'], fit_inputs['target_mask'])


def _fit_candidate(payload, fit_inputs, visible_set, diagonal_m):
    """Run exactly one refined verified fit and exact deterministic repeat."""
    source = payload['source']
    frame = payload['frame']
    original_endpoint_hash = _array_sha256(payload['endpoints'])
    original_conf_hash = _array_sha256(payload['confidence'])
    retained = _retained_candidate_rows(fit_inputs)
    geometry = _pose_jacobian_metrics(
        source, fit_inputs['source_rows'], source['observed_crop_mask'],
        fit_inputs['endpoints_crop'], diagonal_m)
    if len(retained) < 24 or geometry.get('pose_jacobian_rank') != 6:
        unavailable_reason = (
            'fewer_than_24_verified_masked_correspondences' if len(retained) < 24 else
            'verified_set_pose_jacobian_rank_deficient')

        def precondition_decision():
            repeated_rows = _retained_candidate_rows(fit_inputs)
            repeated_geometry = _pose_jacobian_metrics(
                source, fit_inputs['source_rows'], source['observed_crop_mask'],
                fit_inputs['endpoints_crop'], diagonal_m)
            if len(repeated_rows) < 24:
                reason = 'fewer_than_24_verified_masked_correspondences'
            elif repeated_geometry.get('pose_jacobian_rank') != 6:
                reason = 'verified_set_pose_jacobian_rank_deficient'
            else:
                reason = None
            return dict(decision=('unavailable' if reason else 'fit_preconditions_passed'),
                        reason=reason, retained_correspondences=int(len(repeated_rows)),
                        source_rows_sha256=_array_sha256(fit_inputs['source_rows']),
                        endpoint_sha256=_array_sha256(fit_inputs['endpoints_crop']),
                        confidence_sha256=_array_sha256(fit_inputs['confidence']),
                        pose_jacobian_rank=repeated_geometry.get('pose_jacobian_rank'),
                        pose_jacobian_singular_values=repeated_geometry.get(
                            'pose_jacobian_singular_values'))

        primary_decision = dict(
            decision='unavailable', reason=unavailable_reason,
            retained_correspondences=int(len(retained)),
            source_rows_sha256=_array_sha256(fit_inputs['source_rows']),
            endpoint_sha256=_array_sha256(fit_inputs['endpoints_crop']),
            confidence_sha256=_array_sha256(fit_inputs['confidence']),
            pose_jacobian_rank=geometry.get('pose_jacobian_rank'),
            pose_jacobian_singular_values=geometry.get('pose_jacobian_singular_values'))
        unavailable_repeat = _repeat_unavailable_precondition_decision(
            primary_decision, precondition_decision)
        if not unavailable_repeat['matched']:
            raise CalibrationInputError('R6 no-fit precondition decision changed on deterministic repeat')
        return dict(
            arm='verified_refined', fit_state='unavailable',
            accepted_pose_state='unavailable', accepted_pose=None, proposal_pose=None,
            fit_attempted=False, unavailable_reason=unavailable_reason,
            retained_correspondences=int(len(retained)),
            deterministic_repeat=unavailable_repeat,
            geometry=geometry)
    arm_inputs = {key: fit_inputs[key] for key in (
        'source_points_m', 'endpoints_crop', 'confidence', 'source_indices',
        'source_mask', 'target_mask')}
    started = time.perf_counter()
    primary = r5._shared_fit_call(arm_inputs, payload)
    primary_ms = (time.perf_counter() - started) * 1000.0
    repeat_started = time.perf_counter()
    repeated = r5._shared_fit_call(arm_inputs, payload)
    repeat_ms = (time.perf_counter() - repeat_started) * 1000.0
    if (_array_sha256(payload['endpoints']) != original_endpoint_hash or
            _array_sha256(payload['confidence']) != original_conf_hash or
            fit_inputs['input_endpoint_sha256'] != original_endpoint_hash or
            fit_inputs['input_confidence_sha256'] != original_conf_hash):
        raise AssertionError(f'R6 rigid fit modified original or candidate input arrays: {payload["plan"]["condition_id"]}')
    signature, repeated_signature = r5._fit_repeat_signature(primary), r5._fit_repeat_signature(repeated)
    repeat_sha = _canonical_json_sha256(signature)
    repeated_sha = _canonical_json_sha256(repeated_signature)
    deterministic = repeat_sha == repeated_sha
    pose_crop, pose_native = primary.get('pose_crop_m'), primary.get('pose_native_m')
    finite = bool(pose_crop is not None and pose_native is not None and
                  np.isfinite(np.asarray(pose_crop, dtype=np.float64)).all() and
                  np.isfinite(np.asarray(pose_native, dtype=np.float64)).all())
    accepted = bool(finite and primary.get('validation_state') == 'accepted')
    accepted_state = 'accepted' if accepted else ('rejected' if finite else 'unavailable')
    original_rows = r5._filtered_observed_rows(
        source, payload['endpoints'], payload['confidence'], source['observed_crop_mask'])
    full_validation = r5._all_original_candidate_validation(
        payload, original_rows, pose_native if finite else None)
    residuals = r5._residual_summary(
        np.asarray(source['source_points_object_m'], dtype=np.float64)[original_rows],
        np.asarray(payload['endpoints'], dtype=np.float64)[original_rows],
        np.asarray(source['source_indices'], dtype=np.int64)[original_rows],
        pose_native if finite else None, source['crop_k'], source['crop_from_native'],
        frame['native_k'], frame['native_rgb'].shape[0])
    errors = r5._pose_truth_errors(pose_crop, visible_set['truth_pose_crop_m'], diagonal_m)
    surface = r5.surface_projection_score(
        source['source_points_object_m'], visible_set['visible_rows'],
        visible_set['truth_pose_crop_m'], pose_native if finite else None,
        source['crop_from_native'], frame['native_k'],
        Frame(int(payload['plan']['frame_id']), frame['native_rgb'], frame['native_k']),
        visible_source_ids=visible_set['visible_source_ids'])
    return dict(
        arm='verified_refined', fit_state=str(primary.get('state', 'unavailable')),
        fit_attempted=True,
        fit_reason=primary.get('reason'), accepted_pose_state=accepted_state,
        accepted_pose=pose_native if accepted else None,
        proposal_pose=(dict(crop_m=pose_crop, native_m=pose_native) if finite else None),
        proposal_rotation_error=errors,
        proposal_surface_projection_720=surface,
        validation_state=primary.get('validation_state'),
        validation_reason=primary.get('validation_reason'),
        validation_stats=primary.get('validation_stats'),
        current_image_validation_all_original_observed=full_validation,
        current_image_residuals_all_original_observed=residuals,
        retained_correspondences=int(primary.get('retained_correspondences', -1)),
        expected_retained_correspondences=int(len(retained)),
        retained_count_matches_shared_filter=int(primary.get('retained_correspondences', -1)) == len(retained),
        retained_source_ids=r5._compact_ids(fit_inputs['source_indices'][retained]),
        candidate_input=dict(
            verified_correspondence_count=int(len(fit_inputs['source_indices'])),
            verified_source_rows_sha256=_array_sha256(fit_inputs['source_rows']),
            verified_source_ids_sha256=_array_sha256(fit_inputs['source_indices']),
            measured_endpoint_sha256=_array_sha256(fit_inputs['endpoints_crop']),
            original_confidence_sha256=_array_sha256(fit_inputs['confidence']),
            source_points_metres_sha256=_array_sha256(fit_inputs['source_points_m']),
            source_mask_sha256=audit.array_sha256(fit_inputs['source_mask']),
            target_mask_sha256=audit.array_sha256(fit_inputs['target_mask'])),
        geometry=geometry,
        deterministic_repeat=dict(matched=bool(deterministic), primary_sha256=repeat_sha,
                                  repeated_sha256=repeated_sha),
        costs_ms=dict(primary_fit=float(primary_ms), deterministic_repeat=float(repeat_ms)),
        solver=dict(
            function='quality_bottle_identity_audit.fit_learned_packet (unchanged via R5 boundary)',
            seed=int(payload['seed']), source_points_boundary='metres',
            helper_pnp_unit='millimetres', source_and_target_mask='captured observed_crop_mask',
            initialization='captured template crop pose',
            sample_cap=10000, solver='iterative solvePnPRansac with extrinsic guess',
            ransac_iterations=3000, reprojection_error_px=2., ransac_confidence=.999,
            final_refinement='solvePnPRefineLM',
            crop_k_sha256=_array_sha256(source['crop_k']),
            crop_from_native_sha256=_array_sha256(source['crop_from_native']),
            native_k_sha256=_array_sha256(frame['native_k']),
            native_observed_mask_sha256=audit.array_sha256(frame['native_mask'] > 0),
            native_frame_rgb_sha256=audit.array_sha256(frame['native_rgb'])),
    )


def _pose_candidate_gates(condition_rows, r5_report, candidate_arms):
    r5_rows = {row['condition_id']: row for row in r5_report['conditions']}
    positive = []
    positive_ok = True
    improvement = []
    for frame_id in CARRIERS:
        condition_id = f'syn-{frame_id}-q8-rgb-t0-rgb'
        candidate = candidate_arms.get(condition_id, {})
        control = {arm['arm']: arm for arm in r5_rows[condition_id]['arms']}.get('all', {})
        c_error = candidate.get('proposal_rotation_error') or {}
        a_error = control.get('proposal_rotation_error') or {}
        enough = all(c_error.get(name) is not None and a_error.get(name) is not None for name in (
            'rotation_error_degrees', 'translation_error_m', 'translation_error_fraction_D'))
        accepted = candidate.get('accepted_pose_state') == 'accepted'
        no_worse = bool(enough and
                        c_error['rotation_error_degrees'] <= a_error['rotation_error_degrees'] + 1e-6 and
                        c_error['translation_error_m'] <= a_error['translation_error_m'] + 1e-9)
        within = bool(enough and c_error['rotation_error_degrees'] <= 3. and
                      c_error['translation_error_fraction_D'] <= .02)
        rotation_gain = (a_error.get('rotation_error_degrees') - c_error.get('rotation_error_degrees')
                         if enough else None)
        translation_gain = (a_error.get('translation_error_fraction_D') -
                             c_error.get('translation_error_fraction_D') if enough else None)
        improved = bool((rotation_gain is not None and rotation_gain >= .1) or
                        (translation_gain is not None and translation_gain >= .001))
        improvement.append(dict(condition_id=condition_id,
                                rotation_improvement_degrees=rotation_gain,
                                translation_improvement_fraction_D=translation_gain,
                                meets_improvement_gate=improved))
        passed = bool(accepted and no_worse and within)
        positive_ok &= passed
        positive.append(dict(condition_id=condition_id, accepted=accepted,
                             no_worse_than_r5_all=no_worse, within_3deg_002D=within,
                             passed=passed))
    self_rows = []
    self_ok = True
    for frame_id in CARRIERS:
        for variant in ('full', 'clipped'):
            condition_id = f'zero-{frame_id}-{variant}'
            candidate = candidate_arms.get(condition_id, {})
            control = {arm['arm']: arm for arm in r5_rows[condition_id]['arms']}.get('all', {})
            c_error = candidate.get('proposal_rotation_error') or {}
            a_error = control.get('proposal_rotation_error') or {}
            enough = all(c_error.get(name) is not None and a_error.get(name) is not None for name in (
                'rotation_error_degrees', 'translation_error_m', 'translation_error_fraction_D'))
            accepted = candidate.get('accepted_pose_state') == 'accepted'
            no_worse = bool(enough and
                            c_error['rotation_error_degrees'] <= a_error['rotation_error_degrees'] + 1e-6 and
                            c_error['translation_error_m'] <= a_error['translation_error_m'] + 1e-9)
            within = bool(enough and c_error['rotation_error_degrees'] <= 1. and
                          c_error['translation_error_fraction_D'] <= .01)
            passed = bool(accepted and no_worse and within)
            self_ok &= passed
            self_rows.append(dict(condition_id=condition_id, accepted=accepted,
                                  no_worse_than_r5_all=no_worse, within_1deg_001D=within,
                                  passed=passed))
    negative_rows = []
    negative_ok = True
    for frame_id in CARRIERS:
        condition_id = f'syn-{frame_id}-q8-rgb-t180-rgb'
        candidate = candidate_arms.get(condition_id, {})
        accepted = candidate.get('accepted_pose_state') == 'accepted'
        original_wrong = ((r5_rows[condition_id].get('original_dense_baseline') or {})
                          .get('original_wrong_surface_evidence') or {})
        negative_ok &= not accepted
        negative_rows.append(dict(
            condition_id=condition_id, accepted_candidate=accepted,
            accepted_wrong_surface_forbidden=accepted,
            original_wrong_surface_evidence=copy.deepcopy(original_wrong),
            finite_candidate_proposal=bool(candidate.get('proposal_pose') is not None),
            candidate_state=candidate.get('accepted_pose_state', 'candidate_not_run')))
    deterministic = (len(candidate_arms) == 12 and all(
        arm.get('deterministic_repeat', {}).get('matched') is True
        for arm in candidate_arms.values()))
    all_positive_improvement = any(row['meets_improvement_gate'] for row in improvement)
    return dict(
        positive_rows=positive, positive_all_accepted_and_within_gate=bool(positive_ok),
        positive_improvement_rows=improvement,
        at_least_one_positive_improved=bool(all_positive_improvement),
        self_rows=self_rows, all_six_self_accepted_within_gate=bool(self_ok),
        negative_rows=negative_rows, zero_accepted_opposite_template_candidates=bool(negative_ok),
        every_candidate_deterministic=bool(deterministic),
        passed=bool(positive_ok and all_positive_improvement and self_ok and negative_ok and deterministic),
    )


def _initial_record(output_root, plans):
    conditions = []
    for plan in plans:
        conditions.append(dict(
            **plan, state='pending', candidate=dict(state='candidate_not_run', reason='calibration pending')))
    return dict(
        schema_version=1,
        scope='R6 CPU-only current-image patch calibration followed by a conditional pose candidate',
        status='running', complete=False, calibration_passed=False,
        output_root=str(Path(output_root).resolve()),
        spec_path=str(SPEC_PATH.resolve()), spec_sha256=EXPECTED_SPEC_SHA256,
        expected_r1_capture_sha256=EXPECTED_R1_CAPTURE_SHA256,
        expected_r3_report_sha256=EXPECTED_R3_REPORT_SHA256,
        expected_r5_report_sha256=EXPECTED_R5_REPORT_SHA256,
        expected_r5_terminal_sha256=EXPECTED_R5_TERMINAL_SHA256,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        planned_conditions=12, forward_calls=0, render_calls=0, model_calls=0,
        reference_or_annotations_loaded=False,
        source_selection_frozen_before_flow_or_evaluator_truth=False,
        matcher_rule=dict(patch_size=PATCH_SIZE, sigma=HIGH_PASS_SIGMA,
                          minimum_std=PATCH_STD_MIN, erosion_px=6,
                          confidence_strictly_greater_than=CONFIDENCE_MIN_EXCLUSIVE,
                          ncc_min=NCC_MIN, margins_min=MARGIN_MIN,
                          distant_identity_fraction=DISTANT_IDENTITY_FRACTION,
                          search_radius_px=SEARCH_RADIUS, candidate_count=SEARCH_OFFSET_COUNT,
                          max_batch=MAX_MATCH_BATCH),
        working_buffer_budget_bytes=MAX_WORKING_BYTES,
        output_budget_bytes=MAX_OUTPUT_BYTES,
        r5_controls={}, source_selection={}, preflight=[], conditions=conditions,
        toy_calibration={}, started_unix=time.time())


def _serialize_record(record):
    return json.dumps(_json_value(record), indent=2, allow_nan=False).encode('utf-8')


def _atomic_json(path, value, max_bytes=None, exclusive=False):
    path = Path(path)
    encoded = _serialize_record(value)
    if max_bytes is not None and len(encoded) > int(max_bytes):
        raise OutputLimitError(f'Atomic JSON needs {len(encoded)} bytes, over {int(max_bytes)}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists() or (exclusive and path.exists()):
        raise FileExistsError(f'Preserving existing atomic output: {path}')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return dict(path=str(path), bytes=len(encoded), sha256=digest(path))


def _write_terminal(output_root, terminal):
    root = Path(output_root).resolve()
    target = root / 'terminal.json'
    payload = _serialize_record(terminal)
    other_bytes = sum(path.stat().st_size for path in root.rglob('*')
                      if path.is_file() and path.resolve() != target.resolve())
    if other_bytes + len(payload) > MAX_OUTPUT_BYTES:
        raise OutputLimitError(
            f'R6 output would use {other_bytes + len(payload)} bytes, over {MAX_OUTPUT_BYTES}')
    result = _atomic_json(target, terminal, exclusive=True)
    result['total_output_bytes'] = other_bytes + result['bytes']
    result['budget_bytes'] = MAX_OUTPUT_BYTES
    return result


def _write_private_report(output_root, record):
    root = Path(output_root).resolve()
    target = root / 'calibration.json'
    payload = _serialize_record(record)
    other_bytes = sum(path.stat().st_size for path in root.rglob('*')
                      if path.is_file() and path.resolve() != target.resolve())
    if other_bytes + len(payload) > MAX_OUTPUT_BYTES:
        raise OutputLimitError(
            f'R6 output would use {other_bytes + len(payload)} bytes, over {MAX_OUTPUT_BYTES}')
    result = _atomic_json(target, record)
    result['total_output_bytes'] = other_bytes + result['bytes']
    result['budget_bytes'] = MAX_OUTPUT_BYTES
    return result


def _fill_candidate_not_run(record, reason):
    for row in record['conditions']:
        if row.get('candidate', {}).get('state') == 'candidate_not_run':
            row['candidate'] = dict(state='candidate_not_run', reason=str(reason))


def _account_unmeasured_rows(record, reason):
    for row in record['conditions']:
        if row.get('state') == 'pending':
            row['state'] = 'not_run_after_failure'
            row['measurement_unavailable_reason'] = str(reason)


def _compact_failure_record(record, error):
    compact = dict(
        schema_version=1, scope=record.get('scope'), status='failed', complete=False,
        calibration_passed=False, failure=dict(type=type(error).__name__, message=str(error)),
        planned_conditions=12, conditions=[dict(
            condition_id=row['condition_id'], state=row.get('state'),
            candidate=row.get('candidate', dict(state='candidate_not_run')))
            for row in record.get('conditions', [])],
        terminal_failure_preserved=True)
    return compact


def _terminal_record(record, status, complete, failure=None):
    return dict(
        schema_version=1, state='terminal', status=status, complete=bool(complete),
        calibration_passed=bool(record.get('calibration_passed')),
        all_twelve_conditions_accounted=bool(record.get('all_twelve_conditions_accounted')),
        all_twelve_candidates_accounted=bool(record.get('all_twelve_candidates_accounted')),
        forward_calls=0, render_calls=0, model_calls=0,
        input_revalidation_state=(record.get('input_revalidation') or {}).get('state'),
        failure=(None if failure is None else
                 dict(type=type(failure).__name__, message=str(failure))),
        source_pins=record.get('source_pins'), completed_unix=time.time())


def _record_r5_controls(record, r5_report):
    controls = {}
    for row in r5_report['conditions']:
        arms = {arm['arm']: arm for arm in row['arms']}
        controls[row['condition_id']] = dict(
            all=copy.deepcopy(arms['all']),
            source_observable=copy.deepcopy(arms['source_observable']),
            original_dense_baseline=copy.deepcopy(row.get('original_dense_baseline')),
            frozen_truth_set_count=row.get('frozen_truth_set_count'),
            frozen_truth_set_sha256=row.get('frozen_truth_set_sha256'))
    record['r5_controls'] = controls


def _run_calibration_toys():
    """Execute the frozen CPU-only geometry and false-support controls."""
    from .test_quality_bottle_patch_pose_calibration import run_toy_fixture_checks
    return run_toy_fixture_checks()


def run_calibration(capture_root=CAPTURE_ROOT, zero_root=ZERO_ROOT,
                    output_root=OUTPUT_ROOT):
    plans = r5._expected_condition_plan()
    output_root = Path(output_root).expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f'R6 output root already exists; preserving it: {output_root}')
    diagnostics_root = (CACHE / 'diagnostics').resolve()
    resolved_output = output_root.resolve()
    if not resolved_output.is_relative_to(diagnostics_root):
        raise ValueError(f'R6 output must be a fresh private directory under {diagnostics_root}')
    output_root.mkdir(parents=True, exist_ok=False)
    output_root = output_root.resolve()
    record = _initial_record(output_root, plans)
    failure = None
    bundle = None
    try:
        _write_private_report(output_root, record)
        toy = _run_calibration_toys()
        record['toy_calibration'] = toy
        if toy.get('passed') is not True:
            raise CalibrationInputError('Frozen CPU toy calibration controls did not pass')
        bundle = _load_r6_bundle(capture_root, zero_root)
        record.update(
            capture_root=str(bundle['capture_root']), zero_root=str(bundle['zero_root']),
            capture_sha256=bundle['capture_sha256'], r3_report_sha256=bundle['zero_sha256'],
            r5_report_sha256=bundle['r5_report_sha256'],
            r5_terminal_sha256=bundle['r5_terminal_sha256'],
            object_bbox_diagonal_m=float(bundle['capture']['object_bbox_diagonal_m']),
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            r4_r5_gates_preserved=dict(r4_dense_gate_remains_failed=True,
                                       r5_pose_gate_remains_failed=True),
            source_pins=bundle['r6_source_pins'],
            source_snapshot=bundle['source_pins']['capture_snapshot'],
        )
        _record_r5_controls(record, bundle['r5_report'])
        selector_working = _source_selector_working_preflight(bundle, plans)
        record['source_selector_working_preflight'] = selector_working
        selections = _freeze_and_bind_source_selections(bundle, plans)
        record['source_selection_frozen_before_flow_or_evaluator_truth'] = True
        record['source_selection'] = bundle['source_selection_summary']
        selector_owner_bytes = _unique_array_owner_bytes(selections)
        preliminary_working = _preflight_working_sets(
            bundle, plans, selections, resident_state_bytes=selector_owner_bytes,
            reserved_state_bytes=(_visible_set_upper_bound(bundle, plans) +
                                  FIXED_BANK_RESERVATION_BYTES +
                                  FROZEN_METADATA_RESERVATION_BYTES))
        record['working_preflight_before_visible_sets_or_banks'] = preliminary_working
        visible_sets = r5.freeze_visible_evaluation_sets(bundle, selections, plans)
        if len(visible_sets) != 12:
            raise CalibrationInputError('R6 evaluator V does not cover all twelve fixed rows')
        banks = _load_fixed_banks(bundle)
        bank_owner_bytes = _unique_array_owner_bytes(banks)
        if bank_owner_bytes > FIXED_BANK_RESERVATION_BYTES:
            raise WorkingSetLimitError(
                f'Frozen fixed banks use {bank_owner_bytes} bytes, over their {FIXED_BANK_RESERVATION_BYTES}-byte reservation')
        resident_state_bytes = _unique_array_owner_bytes(selections, visible_sets, banks)
        actual_working = _preflight_working_sets(
            bundle, plans, selections, resident_state_bytes=resident_state_bytes,
            reserved_state_bytes=(max(0, FIXED_BANK_RESERVATION_BYTES - bank_owner_bytes) +
                                  FROZEN_METADATA_RESERVATION_BYTES))
        record['working_preflight_with_frozen_resident_buffers'] = actual_working
        record['resident_buffer_accounting'] = dict(
            unique_owner_bytes=resident_state_bytes,
            source_selection_owner_bytes=selector_owner_bytes,
            frozen_visible_set_owner_bytes=_unique_array_owner_bytes(visible_sets),
            fixed_bank_owner_bytes=bank_owner_bytes,
            fixed_bank_object_overhead_reservation_bytes=FIXED_BANK_RESERVATION_BYTES,
            all_twelve_candidate_inputs_reserved_before_matching=True,
            last_loop_fit_input_alias_owner_bytes=0,
            metadata_safety_reservation_bytes=FROZEN_METADATA_RESERVATION_BYTES)
        record['fixed_patch_banks'] = {
            template_id: dict(packet_sha256=entry['entry']['sha256'],
                              manifest_sha256=entry['manifest_sha256'],
                              selected_anchor_count=entry['selected_anchor_count'])
            for template_id, entry in banks.items()}
        preflight_rows = actual_working['rows']
        record['preflight'] = preflight_rows
        if len(preflight_rows) != 12:
            raise CalibrationInputError('R6 resource preflight omitted a fixed row')
        before_measurement = r5._frozen_input_revalidation(bundle)
        if before_measurement['state'] != 'matched':
            raise CalibrationInputError('R6 frozen inputs changed before matching')
        record['input_revalidation_before_measurement'] = before_measurement

        report_rows = {row['condition_id']: row for row in record['conditions']}
        control_rows = {row['condition_id']: row for row in bundle['r5_report']['conditions']}
        candidate_inputs = {}
        for plan in plans:
            condition_id = plan['condition_id']
            row = report_rows[condition_id]
            source_id = r5._source_template_refs(plan['frame_id'], plan['template_offset_deg'])
            payload = r5._load_condition_payload(bundle, plan)
            measured, fit_input = _condition_measurement(
                bundle, plan, payload, selections[source_id], banks,
                visible_sets[condition_id], control_rows[condition_id],
                float(bundle['capture']['object_bbox_diagonal_m']))
            row.update(measured)
            row['state'] = 'measured'
            candidate_inputs[condition_id] = fit_input
            del payload
            _write_private_report(output_root, record)

        if len(candidate_inputs) != 12:
            raise CalibrationInputError('R6 matching did not produce a candidate-input outcome for every fixed row')
        after_measurement = r5._frozen_input_revalidation(bundle)
        record['input_revalidation_after_measurement'] = after_measurement
        if after_measurement['state'] != 'matched':
            raise CalibrationInputError('R6 frozen inputs changed during matching')
        measured_rows = {row['condition_id']: row for row in record['conditions']}
        self_ids = [f'zero-{frame}-{variant}' for frame in CARRIERS for variant in ('full', 'clipped')]
        positive_ids = [f'syn-{frame}-q8-rgb-t0-rgb' for frame in CARRIERS]
        gate_ids = self_ids + positive_ids
        calibration_rows = []
        for condition_id in gate_ids:
            row = measured_rows[condition_id]
            gates = row.get('gates') or {}
            calibration_rows.append(dict(
                condition_id=condition_id,
                conditional_coverage_passed=gates.get('conditional_coverage_passed') is True,
                conditional_identity_passed=gates.get('conditional_identity_passed') is True,
                fixed_anchor_spatial_calibration_passed=(
                    gates.get('fixed_anchor_spatial_calibration_passed') is True),
                self_full_V_dense_qualification_retained=(
                    gates.get('self_full_V_dense_qualification_retained')
                    if condition_id.startswith('zero-') else None),
                passed=gates.get('passed') is True))
        negative_rows = [measured_rows[f'syn-{frame}-q8-rgb-t180-rgb'] for frame in CARRIERS]
        negative_counts_ok = all(
            row['evaluator_only_evidence']['negative_V_count_matches_frozen_cohort']
            for row in negative_rows)
        calibration_passed = bool(
            toy['passed'] and len(calibration_rows) == 9 and
            all(row['passed'] for row in calibration_rows) and negative_counts_ok and
            after_measurement['state'] == 'matched')
        record['calibration_gate'] = dict(
            passed=calibration_passed, required_condition_count=9,
            required_rows=calibration_rows,
            negative_rows=[dict(condition_id=row['condition_id'],
                                visible_count=row['evaluator_only_evidence']['visible_count'],
                                expected_count=row['evaluator_only_evidence']['expected_negative_V_count'],
                                wrong_surface=copy.deepcopy(
                                    row['evaluator_only_evidence']['original_wrong_surface_evidence']))
                           for row in negative_rows],
            negative_cohort_denominators_match=bool(negative_counts_ok),
            all_toy_false_support_controls_passed=bool(toy['passed']),
            input_revalidation_matched=after_measurement['state'] == 'matched')
        record['calibration_passed'] = calibration_passed
        if not calibration_passed:
            _fill_candidate_not_run(record, 'one or more frozen R6 calibration gates failed')
            record['all_twelve_candidates_accounted'] = all(
                row['candidate'].get('state') == 'candidate_not_run' for row in record['conditions'])
            record['candidate_fit_calls'] = 0
            record['input_revalidation'] = after_measurement
            record['all_twelve_conditions_accounted'] = all(
                row.get('state') == 'measured' for row in record['conditions'])
            record['complete'] = bool(record['all_twelve_conditions_accounted'] and
                                      record['all_twelve_candidates_accounted'])
            record['status'] = 'calibration_failed' if record['complete'] else 'failed'
            record['completed_unix'] = time.time()
            record['output'] = _write_private_report(output_root, record)
            terminal_output = _write_terminal(output_root, _terminal_record(
                record, record['status'], record['complete']))
            record['terminal_output'] = {key: terminal_output[key]
                                         for key in ('path', 'bytes', 'sha256', 'budget_bytes')}
            _write_private_report(output_root, record)
            return record

        record['candidate_fit_calls'] = 0
        for plan in plans:
            condition_id = plan['condition_id']
            payload = r5._load_condition_payload(bundle, plan)
            fit_input = candidate_inputs.pop(condition_id)
            fit_result = _fit_candidate(
                payload, fit_input, visible_sets[condition_id],
                float(bundle['capture']['object_bbox_diagonal_m']))
            record['candidate_fit_calls'] += int(fit_result.get('fit_attempted') is True) * 2
            row = report_rows[condition_id]
            row['candidate'] = fit_result
            row['candidate']['state'] = fit_result.get('accepted_pose_state', 'unavailable')
            del fit_input, payload
            _write_private_report(output_root, record)
        after_fits = r5._frozen_input_revalidation(bundle)
        record['input_revalidation'] = after_fits
        if after_fits['state'] != 'matched':
            raise CalibrationInputError('R6 frozen inputs changed during candidate fitting')
        candidate_arms = {row['condition_id']: row['candidate'] for row in record['conditions']}
        record['pose_candidate_gate'] = _pose_candidate_gates(
            record['conditions'], bundle['r5_report'], candidate_arms)
        record['all_twelve_conditions_accounted'] = all(
            row.get('state') == 'measured' for row in record['conditions'])
        record['all_twelve_candidates_accounted'] = all(
            row.get('candidate', {}).get('state') in ('accepted', 'rejected', 'unavailable')
            for row in record['conditions'])
        record['complete'] = bool(record['all_twelve_conditions_accounted'] and
                                  record['all_twelve_candidates_accounted'] and
                                  record['input_revalidation']['state'] == 'matched')
        record['status'] = 'complete' if record['complete'] else 'failed'
        record['independent_real_accuracy_verified'] = False
        record['dense_endpoint_replacement_claimed'] = False
        record['runtime_or_phone_promotion'] = False
        record['completed_unix'] = time.time()
        record['output'] = _write_private_report(output_root, record)
        terminal_output = _write_terminal(output_root, _terminal_record(
            record, record['status'], record['complete']))
        record['terminal_output'] = {key: terminal_output[key]
                                     for key in ('path', 'bytes', 'sha256', 'budget_bytes')}
        record['output'] = _write_private_report(output_root, record)
        return record
    except BaseException as error:
        failure = error
        _account_unmeasured_rows(record, f'{type(error).__name__}: {error}')
        _fill_candidate_not_run(record, f'not run after {type(error).__name__}: {error}')
        record['failure'] = dict(type=type(error).__name__, message=str(error))
        try:
            record['input_revalidation'] = (r5._frozen_input_revalidation(bundle)
                                            if bundle is not None else dict(state='not_run'))
        except BaseException as pin_error:
            record['input_revalidation'] = dict(
                state='failed', error_type=type(pin_error).__name__, error=str(pin_error))
        record['all_twelve_conditions_accounted'] = len(record['conditions']) == 12 and all(
            row.get('state') in ('measured', 'not_run_after_failure')
            for row in record['conditions'])
        record['all_twelve_candidates_accounted'] = len(record['conditions']) == 12 and all(
            row.get('candidate', {}).get('state') in
            ('candidate_not_run', 'accepted', 'rejected', 'unavailable')
            for row in record['conditions'])
        if 'calibration_gate' not in record:
            record['calibration_passed'] = False
        record['status'] = 'failed'
        record['complete'] = False
        record['terminal_failure_preserved'] = True
        record['completed_unix'] = time.time()
        try:
            record['output'] = _write_private_report(output_root, record)
        except OutputLimitError:
            record['output'] = _write_private_report(
                output_root, _compact_failure_record(record, error))
        try:
            terminal_output = _write_terminal(
                output_root, _terminal_record(record, 'failed', False, error))
            record['terminal_output'] = {key: terminal_output[key]
                                         for key in ('path', 'bytes', 'sha256', 'budget_bytes')}
            record['output'] = _write_private_report(output_root, record)
        except (OutputLimitError, FileExistsError):
            # The first terminal is immutable; the primary report retains the
            # complete fixed-row outcome when a terminal path already exists.
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=CAPTURE_ROOT)
    parser.add_argument('--zero-root', type=Path, default=ZERO_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--preflight-only', action='store_true',
                        help='Load all twelve pinned rows and stop before matching or fitting')
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight_actual_cache(args.capture_root, args.zero_root), indent=2,
                         allow_nan=False))
    else:
        record = run_calibration(args.capture_root, args.zero_root, args.output_root)
        print(json.dumps(dict(status=record['status'], complete=record['complete'],
                              calibration_passed=record['calibration_passed'],
                              candidate_fit_calls=record.get('candidate_fit_calls', 0),
                              output=record.get('output'), output_root=str(args.output_root)),
                         indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
