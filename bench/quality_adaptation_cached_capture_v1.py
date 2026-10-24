"""Authenticated reuse of frozen RAW inputs in a separately budgeted phase.

This adapter imports completed render/bank inputs and observed masks from one
terminally failed capture attempt into a fresh RAW context. It does not import
old crops or seeds and does not run neural inference, training, or downloads.
The default cached-input import remains separate from the explicit continuation
join below. Source release approval must bind the successor adapter bytes.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time

from . import quality_adaptation_mask_recovery_v1 as recovery
from . import quality_adaptation_portable as raw
from . import quality_adaptation_portable_prepare as p

SOURCE_NAME = 'bench/quality_adaptation_cached_capture_v1.py'
RECOVERY_NAME = 'bench/quality_adaptation_mask_recovery_v1.py'
REVIEW_SCHEMA = 'visualizeit_raw_cached_capture_phase_review_v1'
REVIEW_SCOPE = 'authenticated_cached_input_import_v1'
TERMINAL_SCHEMA = 'visualizeit_raw_parent_observed_capture_terminal_v2'
MODE = 'authenticated_cached_input_import_v1'
POLICY_FILE = 'cached-capture-phase-policy-v1.json'
PHASE_FILE = 'cached-capture-phase-validation-v1.json'
ROLE_FILES = {
    'render': 'cached-input-render-v1.json',
    'banks': 'cached-input-banks-v1.json',
    'masks': 'cached-input-masks-v1.json',
}
GIB = 1024**3
MIB = 1024**2


def _seconds(value, name='elapsed seconds'):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative ' + name + ' required')
    return float(value)


def continuation_metadata(context, successor, terminal_pin, receipt_pin, *, adapter_pin,
                          predecessor_adapter_archive=None, materialize=False):
    """Join a completed continuation to the existing preparation functions.

    This separate view has its own seal schema and source lineage. Pass its
    returned state to ``p.execution_identity(state, execution_root)`` and then
    ``p.extract(request, state, execution_root)``. Legacy CLI seal admission is
    unchanged. No original clean stage or RAW seal is fabricated.

    Explicit materialization makes byte-identical hardlinks and input-only
    metadata. An interrupted view is preserved for operator review.
    """
    began = time.monotonic()
    admission = raw.authenticate_capture_continuation(context, successor, terminal_pin)
    root = Path(context.document()['output_root'])
    if raw.check_pin(adapter_pin) != Path(__file__).resolve():
        raise ValueError('exact successor continuation adapter pin required')
    policy_pin = admission['metadata'].get(POLICY_FILE)
    if policy_pin is not None:
        old_adapter = raw.read_json(raw.check_pin(policy_pin, under=root))['adapter']
        if (predecessor_adapter_archive is None or any(predecessor_adapter_archive.get(k) != old_adapter[k]
                for k in ('bytes', 'sha256'))):
            raise ValueError('exact archived predecessor cached-capture adapter required')
        raw.check_pin(predecessor_adapter_archive)
    elif predecessor_adapter_archive is not None:
        raise ValueError('predecessor adapter archive lacks frozen phase policy binding')
    receipt_path = raw.check_pin(receipt_pin, under=root, max_bytes=raw.MAX_METADATA)
    receipt = raw.read_json(receipt_path)
    allowance = raw._continuation_allowance(context, successor, terminal_pin, receipt['allowance'])
    attempt = raw._absolute((root / 'capture-continuations' / allowance['attempt_id']).as_posix(), 'continuation attempt')
    if receipt_path != attempt / 'receipt.json':
        raise ValueError('continuation receipt is not at its exact attempt path')
    expected_ids = admission['unattempted_physical_ids']
    elapsed = _seconds(receipt.get('elapsed_seconds'), 'continuation elapsed')
    if (receipt.get('schema') != 'visualizeit_raw_capture_continuation_receipt_v1'
            or receipt.get('status') != 'completed' or receipt.get('fatal_error') is not None
            or receipt.get('preservation_error') is not None
            or receipt.get('predecessor_terminal') != terminal_pin
            or receipt.get('predecessor_context_sha256') != context.sha256
            or receipt.get('successor_context_sha256') != successor.sha256
            or receipt.get('predecessor_metadata') != admission['metadata']
            or receipt.get('source_archive') != admission['source_archive']
            or receipt.get('attempted_physical_ids') != expected_ids
            or receipt.get('completed_physical_ids') != expected_ids
            or receipt.get('remaining_unattempted_physical_ids') != []
            or receipt.get('predecessor_phase_elapsed_seconds') != admission['predecessor_phase_elapsed_seconds']
            or receipt.get('prior_history_elapsed_seconds') != admission['prior_history_elapsed_seconds']
            or receipt.get('cumulative_elapsed_seconds') != admission['cumulative_elapsed_seconds'] + elapsed
            or receipt.get('original_phase_budget_seconds') != admission['original_phase_budget_seconds']
            or receipt.get('additional_allowance_seconds') != allowance['max_seconds']
            or elapsed > allowance['max_seconds'] or receipt.get('annotation_inputs') != []
            or receipt.get('test_payload_reads') != 0 or receipt.get('training_ready') is not False
            or receipt.get('legacy_raw_seal') is not False):
        raise ValueError('continuation is incomplete, changed or lacks exact provenance/budget accounting')
    progress = raw.read_json(raw.check_pin(receipt['progress'], under=attempt, max_bytes=raw.MAX_METADATA))
    if progress != dict(attempted=expected_ids, completed=expected_ids, in_progress=None):
        raise ValueError('continuation progress lacks completed whole-unit chronology')
    if (raw.check_pin(receipt['physical_index'], under=attempt, max_bytes=raw.MAX_METADATA) != attempt / 'physical-index.json'
            or raw.check_pin(receipt['capture_index'], under=attempt, max_bytes=raw.MAX_METADATA) != attempt / 'capture-index.json'):
        raise ValueError('continuation indexes are not at their exact attempt paths')
    physical = raw.read_json(attempt / 'physical-index.json')['rows']
    index = raw.read_json(attempt / 'capture-index.json')
    if set(index) != {'rows', 'artifact_roots', 'training_ready'} or index['training_ready'] is not False:
        raise ValueError('continuation capture index contract differs')
    captures = index['rows']; eligible = set(expected_ids)
    expected_roots = {r['id']: (attempt / 'raw' if r['id'] in eligible else root / 'raw').as_posix()
                      for r in admission['physicals']}
    if receipt.get('artifact_roots') != expected_roots or index['artifact_roots'] != expected_roots:
        raise ValueError('continuation artifact origins differ')
    identities = ('id', 'object_id', 'split', 'sequence_id', 'capture_group', 'frame_id', 'timestamp_s', 'iteration')
    if len(physical) != len(admission['physicals']) or len(captures) != len(admission['states']):
        raise ValueError('continuation physical/state inventory incomplete')
    for actual, old in zip(physical, admission['physicals']):
        if (set(actual) != set(old) or any(actual[k] != old[k] for k in old if k != 'seed')
                or (old['id'] not in eligible and actual != old)):
            raise ValueError('continuation changed preserved physical identity/input/seed')
        if old['id'] in eligible and actual['seed'].get('status') not in ('available', 'unavailable'):
            raise ValueError('continuation lacks terminal seed outcome')
    origin_by_state = {}; descriptors = []
    for actual, old in zip(captures, admission['states']):
        physical_id = actual.get('id', '').rsplit(':iteration', 1)[0]
        if any(actual.get(k) != old[k] for k in identities):
            raise ValueError('continuation state identities missing/duplicate/reordered')
        if physical_id not in eligible and actual != old:
            raise ValueError('continuation changed preserved chronological capture')
        if old['split'] not in ('train', 'validation'):
            continue
        if actual.get('status') == 'available':
            raw.require_digest(actual.get('query_rgb_sha256'))
            origin = Path(expected_roots[physical_id]); origin_by_state[actual['id']] = origin
            for key in ('crops', 'capture'):
                descriptor = actual[key]; relative = Path(descriptor['path'])
                if relative.is_absolute() or len(relative.parts) != 1 or relative.name in ('.', '..'):
                    raise ValueError('continuation crop path must be one relative filename')
                descriptors.append((actual['id'], key, descriptor, origin))
        elif (actual.get('status') != 'unavailable' or not actual.get('failure_reason')
                or actual['failure_reason'] == 'not_processed'
                or actual.get('crops') is not None or actual.get('capture') is not None):
            raise ValueError('continuation state lacks an explicit terminal failure')
    p.reject_aliases([(r['split'], r.get('query_rgb_sha256'),
        [origin_by_state[r['id']] / r[k]['path'] for k in ('crops', 'capture')]
        if r['status'] == 'available' else []) for r in captures])
    for ident, key, descriptor, origin in descriptors:
        raw.check_pin(dict(descriptor, path=(origin / descriptor['path']).as_posix()), under=origin,
            max_bytes=16 * 1024**2)
    with raw._successor_sources(context, successor, attempt / 'raw'):
        raw.preflight_inputs(raw._read_metadata_bytes(root / 'input-plan.json'),
            admission['metadata']['input-plan.json']['sha256'], context)
    view = attempt / 'preparation-view'; seal_path = view / 'raw-seal.json'
    if materialize:
        if view.exists() or view.is_symlink():
            raise FileExistsError('continuation preparation view already attempted; preserve it')
        raw._absolute(view.as_posix(), 'continuation preparation view')
        (view / 'raw').mkdir(parents=True, exist_ok=False)
        for name in ('recipe.json', 'scale-plan.json', 'split.json'):
            raw.write_bytes(context, view / name, raw._read_metadata_bytes(root / name))
        raw.write_bytes(context, view / 'successor-source-context.json', successor.raw)
        seen = set()
        for ident, key, descriptor, origin in descriptors:
            destination = view / 'raw' / descriptor['path']
            if destination in seen:
                raise ValueError('duplicate continuation view crop filename')
            seen.add(destination); raw.resource_check(context)
            os.link(origin / descriptor['path'], destination)
        artifacts = {name: raw.pin(view / name) for name in (
            'recipe.json', 'scale-plan.json', 'split.json', 'successor-source-context.json')}
        artifacts.update({('raw/' + d['path']): raw.pin(view / 'raw' / d['path'])
                          for _, _, d, _ in descriptors})
        join_elapsed = time.monotonic() - began
        if elapsed + join_elapsed > allowance['max_seconds']:
            raise raw.ResourceError('continuation allowance exhausted while joining preparation view')
        seal = dict(schema='visualizeit_raw_capture_continuation_seal_v1',
            continuation_receipt=receipt_pin, predecessor_terminal=terminal_pin,
            adapter=adapter_pin, predecessor_adapter_archive=predecessor_adapter_archive,
            predecessor_context_sha256=context.sha256, successor_context_sha256=successor.sha256,
            artifact_roots=expected_roots, physical_frames=physical, refinement_states=captures,
            artifacts=artifacts, join_elapsed_seconds=join_elapsed,
            cumulative_elapsed_seconds=receipt['cumulative_elapsed_seconds'] + join_elapsed,
            original_cumulative_budget_compliance=False, legacy_raw_seal=False,
            annotation_inputs=[], test_payload_reads=0, training_ready=False)
        seal['seal_sha256'] = p.canonical(seal)
        raw.write_json(context, seal_path, seal)
    seal = p.check_seal(raw.read_json(seal_path), 'visualizeit_raw_capture_continuation_seal_v1')
    if (seal.get('continuation_receipt') != receipt_pin or seal.get('predecessor_terminal') != terminal_pin
            or seal.get('adapter') != adapter_pin
            or seal.get('predecessor_adapter_archive') != predecessor_adapter_archive
            or seal.get('predecessor_context_sha256') != context.sha256
            or seal.get('successor_context_sha256') != successor.sha256
            or seal.get('physical_frames') != physical or seal.get('refinement_states') != captures
            or seal.get('artifact_roots') != expected_roots or seal.get('legacy_raw_seal') is not False
            or seal.get('annotation_inputs') != [] or seal.get('test_payload_reads') != 0
            or seal.get('training_ready') is not False):
        raise ValueError('continuation preparation seal lineage differs')
    expected_files = {'recipe.json', 'scale-plan.json', 'split.json', 'successor-source-context.json'}
    expected_files |= {'raw/' + d['path'] for _, _, d, _ in descriptors}
    if set(seal['artifacts']) != expected_files:
        raise ValueError('continuation preparation seal artifact inventory differs')
    for name, descriptor in seal['artifacts'].items():
        if raw.check_pin(descriptor, under=view) != view / name:
            raise ValueError('continuation preparation artifact path differs')
    for name in ('recipe.json', 'scale-plan.json', 'split.json'):
        if raw.sha256(view / name) != admission['metadata'][name]['sha256']:
            raise ValueError('continuation preparation recipe/plan/split changed')
    if (view / 'successor-source-context.json').read_bytes() != successor.raw:
        raise ValueError('continuation preparation executing source context differs')
    join_elapsed = _seconds(seal.get('join_elapsed_seconds'), 'preparation join elapsed')
    if (elapsed + join_elapsed > allowance['max_seconds'] or seal.get('cumulative_elapsed_seconds') !=
            receipt['cumulative_elapsed_seconds'] + join_elapsed):
        raise ValueError('continuation preparation cumulative clock changed')
    with raw._successor_sources(context, successor, attempt / 'raw'):
        _, spec, plan, _ = raw.frozen(context)
    return dict(context=successor.document(), recipe=spec, plan=plan, physical=physical,
        captures=captures, seal=seal, root=view, closing_stage=receipt_pin)


def _pin_equal(expected, actual):
    return isinstance(expected, dict) and expected == actual


def approve(context, review_pin):
    """Require an independent decision for the exact adapter and helper bytes."""
    doc = raw.verify_sources(context)
    review = p.metadata(review_pin)
    files = review.get('files')
    required = {
        SOURCE_NAME: p.sha256(Path(__file__)),
        RECOVERY_NAME: p.sha256(Path(recovery.__file__)),
    }
    if (review.get('schema') != REVIEW_SCHEMA
            or review.get('decision') != 'APPROVE'
            or review.get('scope') != REVIEW_SCOPE
            or review.get('original_raw_source_files') != doc['source_files']
            or not isinstance(files, dict)
            or any(files.get(name) != digest for name, digest in required.items())):
        raise ValueError('independent exact cached-capture source approval required')
    return review


def _same_phase_context(old_doc, doc):
    if ({k: v for k, v in old_doc.items() if k not in ('output_root', 'scratch_root')}
            != {k: v for k, v in doc.items() if k not in ('output_root', 'scratch_root')}):
        raise ValueError('capture phase may change only fresh output/scratch roots')
    expected = dict(wall_seconds=7200, rss_bytes=8 * GIB,
                    storage_bytes=32 * GIB, reserve_bytes=GIB)
    if doc['limits'] != expected or old_doc['limits'] != expected:
        raise ValueError('capture phase requires frozen7200s/8GiB/32GiB/1GiB limits')


def _validate_terminal(context, terminal_evidence_pin):
    """Validate the parent's actual terminal observation and exact source pins."""
    doc = context.document()
    root = Path(doc['output_root'])
    evidence = p.metadata(terminal_evidence_pin)
    if set(evidence) != {'schema', 'source_context', 'source_context_sha256', 'job', 'gpu_guard',
                         'outer_observation', 'absence_observation', 'watcher_elapsed_seconds',
                         'source_pins'}:
        raise ValueError('unknown/missing parent terminal evidence fields')
    source_context_pin = evidence['source_context']
    if (evidence['schema'] != TERMINAL_SCHEMA
            or not isinstance(source_context_pin, dict)
            or set(source_context_pin) != {'path', 'bytes', 'sha256'}
            or evidence['source_context_sha256'] != context.sha256
            or source_context_pin['sha256'] != context.sha256):
        raise ValueError('parent terminal evidence is bound to another RAW context')
    # The cloud operator may pin its immutable control-context alias instead
    # of output_root/context.json. Accept only byte-identical context metadata.
    context_path = p.verify_pin(source_context_pin)
    if context_path.read_bytes() != context.raw:
        raise ValueError('parent terminal context alias differs from source bytes')
    job = evidence['job']
    if set(job) != {'pid', 'returncode', 'poll_observed_unix_s', 'pid_absent',
                    'process_start_token', 'process_start_token_at_poll',
                    'timeout_seconds', 'kill_grace_seconds'}:
        raise ValueError('unknown/missing terminal job observation fields')
    if (type(job['pid']) is not int or job['pid'] <= 1
            or type(job['returncode']) is not int or job['returncode'] == 0
            or type(job['pid_absent']) is not bool or not job['pid_absent']):
        raise ValueError('actual failed Popen terminal state and absent PID required')
    observed = job['poll_observed_unix_s']
    if type(observed) not in (int, float) or not math.isfinite(observed) or observed <= 0:
        raise ValueError('actual terminal poll timestamp required')
    token, token_at_poll = job['process_start_token'], job['process_start_token_at_poll']
    if token is not None and (not isinstance(token, str) or not token):
        raise ValueError('invalid recorded process start identity')
    if token_at_poll is not None and (not isinstance(token_at_poll, str) or not token_at_poll):
        raise ValueError('invalid terminal process start identity')
    if token != token_at_poll:
        raise ValueError('PID reuse or process start identity mismatch')
    if (type(job['timeout_seconds']) is not int or job['timeout_seconds'] <= 0
            or type(job['kill_grace_seconds']) is not int or job['kill_grace_seconds'] < 0
            or job['returncode'] != 124):
        raise ValueError('outer timeout wrapper completion and deadline evidence required')
    gpu = evidence['gpu_guard']
    if (set(gpu) != {'global_lock_available', 'compute_pids'}
            or gpu['global_lock_available'] is not True
            or gpu['compute_pids'] != []):
        raise ValueError('actual available global GPU lock and no compute processes required')
    expected_names = {
        'freeze': 'freeze.json',
        'physical_index': 'physical-index.json',
        'stage_render': 'stage-render.json',
        'stage_banks': 'stage-banks.json',
        'stage_masks': 'stage-masks.json',
        'failed_capture_stage': None,
        'capture_receipt': None,
    }
    pins = evidence['source_pins']
    if not isinstance(pins, dict) or set(pins) != set(expected_names):
        raise ValueError('complete exact parent source evidence pins required')
    for key, name in expected_names.items():
        if name is None:
            if pins[key] is not None:
                raise ValueError('missing RAW capture artifacts must remain explicitly absent')
            continue
        if not _pin_equal(pins[key], p.pin(root / name)):
            raise ValueError('parent terminal source pin differs: ' + key)
    absence = evidence['absence_observation']
    if (set(absence) != {'observed_unix_s', 'stage_capture_absent', 'capture_receipt_absent'}
            or type(absence['observed_unix_s']) not in (int, float)
            or not math.isfinite(absence['observed_unix_s'])
            or absence['observed_unix_s'] < observed
            or absence['stage_capture_absent'] is not True
            or absence['capture_receipt_absent'] is not True
            or (root / 'stage-capture.json').exists() or (root / 'stage-capture.json').is_symlink()
            or (root / 'raw' / 'capture-receipt.json').exists()
            or (root / 'raw' / 'capture-receipt.json').is_symlink()):
        raise ValueError('explicit terminal observation of missing capture receipt/stage required')
    observation = p.metadata(evidence['outer_observation'])
    expected_context = source_context_pin
    if (observation.get('schema') != 'visualizeit_expanded_mask_recovery_observation_v1'
            or observation.get('stage') != 'capture'
            or observation.get('shard_id') != 'group1'
            or observation.get('status') != 'incomplete'
            or observation.get('reason') != 'interrupted'
            or observation.get('returncode') != -15
            or observation.get('outputs_preserved') is not True
            or observation.get('cleanup_errors') != []
            or observation.get('log_errors') != []
            or observation.get('training_executed') is not False
            or observation.get('test_payload_reads') != 0
            or observation.get('context') != expected_context):
        raise ValueError('pinned real outer capture observation is incomplete or unsafe')
    outer_elapsed = _seconds(observation.get('elapsed_seconds'), 'measured outer capture elapsed')
    if outer_elapsed > job['timeout_seconds'] + job['kill_grace_seconds']:
        raise ValueError('outer measured capture duration exceeds timeout upper bound')
    watch = observation.get('combined_raw_storage_watch')
    if (not isinstance(watch, dict) or watch.get('errors') != []
            or watch.get('sampler_stopped') is not True or watch.get('outer_cancelled') is not False):
        raise ValueError('outer capture storage watcher did not stop cleanly')
    watcher_elapsed = evidence['watcher_elapsed_seconds']
    if watcher_elapsed is not None:
        watcher_elapsed = _seconds(watcher_elapsed, 'separate watcher elapsed')
    return dict(document=evidence, observation=observation,
                observation_pin=evidence['outer_observation'], outer_elapsed_seconds=outer_elapsed,
                outer_upper_bound_seconds=job['timeout_seconds'] + job['kill_grace_seconds'],
                watcher_elapsed_seconds=watcher_elapsed)


def _capture_history(context, terminal):
    """Authenticate metadata receipts before any active payload is traversed."""
    doc, spec, plan, rows = raw.frozen(context)
    root = Path(doc['output_root'])
    # Complete the identity/TEST metadata barrier before the recovery helper
    # authenticates any hardlinked payload bytes.
    _verify_active_identity(rows, raw.physical_rows(spec, plan))
    expected_stages = {'stage-render.json', 'stage-banks.json', 'stage-masks.json',
                       recovery.MARKER, recovery.IMPORT_MARKER}
    if {path.name for path in root.glob('stage-*.json')} != expected_stages:
        raise ValueError('exact authenticated retry and completed render/banks/masks inventory required')
    receipts = {}
    for name in ('render', 'banks', 'masks'):
        receipt = p.read_json(root / ('stage-' + name + '.json'))
        recovery._clean_receipt(context, name, receipt)
        receipts[name] = receipt
    source_marker_path = root / recovery.MARKER
    import_receipt_path = root / recovery.IMPORT_MARKER
    source_marker = p.read_json(source_marker_path)
    source_marker_pin_before = p.pin(source_marker_path)
    recovery_review_pin = source_marker.get('independent_review')
    if (source_marker.get('schema') != 'visualizeit_raw_mask_resource_source_v1'
            or source_marker.get('mode') != 'authenticated_retry'
            or source_marker.get('context_sha256') != context.sha256
            or source_marker.get('adapter') != p.pin(Path(recovery.__file__))
            or not isinstance(recovery_review_pin, dict)
            or source_marker.get('elapsed_seconds') != 0.
            or source_marker.get('status') != 'completed'
            or source_marker.get('original_sources_changed') is not False
            or source_marker.get('training_ready') is not False
            or source_marker.get('test_payload_reads') != 0):
        raise ValueError('authenticated retry source marker changed')
    recovery_receipt = recovery.validate_extension(context, recovery_review_pin)
    if (p.read_json(import_receipt_path) != recovery_receipt
            or p.pin(source_marker_path) != source_marker_pin_before):
        raise ValueError('recovery extension receipts changed during authentication')
    if (recovery_receipt.get('schema') != 'visualizeit_raw_authenticated_recovery_import_v1'
            or recovery_receipt.get('context_sha256') != context.sha256
            or recovery_receipt.get('status') != 'completed'
            or recovery_receipt.get('source_marker') != p.pin(source_marker_path)
            or recovery_receipt.get('test_payload_reads') != 0
            or recovery_receipt.get('generation_payload_reads') != 0
            or recovery_receipt.get('training_ready') is not False
            or recovery_receipt.get('original_stage_seconds_charged_in_imported_receipts') is not True
            or recovery_receipt.get('prior_attempts_preserved') is not True):
        raise ValueError('authenticated recovery import history is incomplete')
    old_context_pin = recovery_receipt.get('old_context')
    old_context_path = p.verify_pin(old_context_pin)
    old_context = raw.read_context(old_context_path, old_context_pin['sha256'])
    _same_phase_context(old_context.document(), doc)
    old_root = Path(old_context.document()['output_root'])
    for key, name in (('old_physical_index', 'physical-index.json'),
                      ('old_freeze', 'freeze.json'),
                      ('old_failed_masks', 'stage-masks.json')):
        pin = recovery_receipt.get(key)
        if pin != p.pin(old_root / name):
            raise ValueError('recovery prior-attempt pin differs: ' + key)
    prior_elapsed = _seconds(recovery_receipt.get('charged_prior_seconds'),
                             'prior failed attempt charge')
    transport_elapsed = _seconds(recovery_receipt.get('transport_elapsed_seconds'),
                                 'recovery transport elapsed')
    recovery_elapsed = _seconds(recovery_receipt.get('elapsed_seconds'),
                                'authenticated recovery elapsed')
    if recovery_elapsed != prior_elapsed + transport_elapsed:
        raise ValueError('recovery import must retain prior failure plus measured transport exactly once')
    recovery_pin = p.pin(import_receipt_path)
    marker_pin = p.pin(source_marker_path)
    recovery_review = p.metadata(recovery_review_pin)
    if (recovery_review.get('schema') != recovery.REVIEW_SCHEMA
            or recovery_review.get('decision') != 'APPROVE'
            or recovery_review.get('original_raw_source_files') != doc['source_files']
            or recovery_review.get('files', {}).get(recovery.SOURCE_NAME) != p.sha256(Path(recovery.__file__))):
        raise ValueError('original recovery extension review did not authenticate its adapter')
    original_context_sha = recovery_receipt.get('old_context', {}).get('sha256')
    for name in ('render', 'banks'):
        receipt = receipts[name]
        if (receipt.get('mode') != 'authenticated_import'
                or receipt.get('neural_execution') is not False
                or receipt.get('original_context_sha256') != original_context_sha
                or not isinstance(receipt.get('original_receipt'), dict)):
            raise ValueError('cached render/bank stages lack authenticated recovery lineage')
    if (receipts['masks'].get('mode') is not None
            or receipts['masks'].get('neural_execution') is True):
        raise ValueError('completed observed-mask stage must remain a genuine source receipt')
    freeze = p.read_json(root / 'freeze.json')
    _seconds(freeze['elapsed_seconds'], 'source freeze elapsed')
    # This explicit v2 case is admitted only because the parent pinned the
    # actual interrupted outer observation and separately observed both RAW
    # capture artifacts absent. No missing RAW receipt is synthesized.
    mask_counts = _status_counts(rows, 'mask')
    if mask_counts['pending']:
        raise ValueError('completed source mask stage has pending TRAIN/VAL outcomes')
    source_pins = {
        'context': p.pin(root / 'context.json'),
        'recipe': p.pin(root / 'recipe.json'),
        'scale_plan': p.pin(root / 'scale-plan.json'),
        'split': p.pin(root / 'split.json'),
        'freeze': p.pin(root / 'freeze.json'),
        'physical_index': p.pin(root / 'physical-index.json'),
        'stage_receipts': {name: p.pin(root / ('stage-' + name + '.json'))
                           for name in ('render', 'banks', 'masks')},
    }
    source_pins['failed_capture_stage'] = None
    source_pins['capture_receipt'] = None
    source_pins['outer_capture_observation'] = terminal['observation_pin']
    source_pins['recovery_lineage'] = dict(source_marker=marker_pin,
        import_receipt=recovery_pin, independent_review=recovery_review_pin,
        original_context=recovery_receipt['old_context'],
        original_freeze=recovery_receipt['old_freeze'],
        original_physical_index=recovery_receipt['old_physical_index'],
        original_failed_masks=recovery_receipt['old_failed_masks'])
    evidence = terminal['document']
    expected_terminal = {
        'freeze': source_pins['freeze'],
        'physical_index': source_pins['physical_index'],
        'stage_render': source_pins['stage_receipts']['render'],
        'stage_banks': source_pins['stage_receipts']['banks'],
        'stage_masks': source_pins['stage_receipts']['masks'],
        'failed_capture_stage': None,
        'capture_receipt': None,
    }
    if evidence['source_pins'] != expected_terminal:
        raise ValueError('terminal snapshot physical inventory pin changed')
    current_raw_elapsed = freeze['elapsed_seconds'] + sum(r['elapsed_seconds'] for r in receipts.values())
    raw_history = current_raw_elapsed + recovery_elapsed
    watcher = terminal['watcher_elapsed_seconds'] or 0.
    historical_measured = raw_history + terminal['outer_elapsed_seconds'] + watcher
    historical_upper_bound = raw_history + terminal['outer_upper_bound_seconds'] + watcher
    return dict(doc=doc, spec=spec, plan=plan, rows=rows, receipts=receipts,
                freeze=freeze, source_pins=source_pins,
                raw_history_elapsed_seconds=raw_history,
                source_raw_stage_elapsed_seconds=current_raw_elapsed,
                source_recovery_import_elapsed_seconds=recovery_elapsed,
                outer_elapsed_seconds=terminal['outer_elapsed_seconds'],
                outer_upper_bound_seconds=terminal['outer_upper_bound_seconds'],
                watcher_elapsed_seconds=watcher,
                measured_historical_elapsed_seconds=historical_measured,
                historical_upper_bound_seconds=historical_upper_bound,
                mask_counts=mask_counts, seed_counts=_status_counts(rows, 'seed'))


def _status_counts(rows, kind):
    counts = {'available': 0, 'unavailable': 0, 'pending': 0}
    for row in rows:
        if row['split'] in ('train', 'validation'):
            status = row[kind]['status']
            if status not in counts:
                raise ValueError('unknown active ' + kind + ' status')
            counts[status] += 1
    return counts


def _check_storage(context, sample_combined):
    """Retain both the frozen RAW guards and the combined retained-root check."""
    raw.resource_check(context, p.TEMP_ENVELOPE)
    recovery.proposal_reserve(context, 0, sample_combined=sample_combined,
                              output_future=p.TEMP_ENVELOPE)


def _preflight(context, *, sample_combined=None):
    doc = context.document()
    path = Path(doc['output_root']) / 'input-plan.json'
    if path.is_file():
        data = raw._read_metadata_bytes(path)
    else:
        _, _, _, rows = raw.frozen(context)
        data = raw.encoded(raw.capture_plan(context, rows))
    if sample_combined is not None:
        _check_storage(context, sample_combined)
    digest = hashlib.sha256(data).hexdigest()
    return raw.preflight_inputs(data, digest, context)


def _verify_active_identity(source, fresh_template):
    # This metadata-only barrier checks LAST TEST/dev rows and every active
    # identity before any source artifact path is inspected or opened.
    return recovery.import_rows(fresh_template, source)


def _rebased(root, old_root, descriptor):
    path = recovery._contained(descriptor['path'], old_root)
    return recovery._contained(root / path.relative_to(old_root), root)


def _verify_render_transport(old_rows, new_rows, old_root, new_root):
    old_by_id = p.indexed(old_rows, 'original source rows')
    new_by_id = p.indexed(new_rows, 'phase rows')
    lineage, opaque = [], []
    for ident, old in old_by_id.items():
        new = new_by_id[ident]
        if old['split'] not in ('train', 'validation'):
            continue
        if old['render']['status'] == 'unavailable':
            if new['render'] != old['render']:
                raise ValueError('unavailable render failure changed during import')
            continue
        if old['render']['status'] != 'available' or new['render'].get('status') != 'available':
            raise ValueError('completed render source state is not importable')
        for key in ('input', 'video', 'mesh'):
            original = old['render'][key]
            target = _rebased(new_root, old_root, original)
            imported = {**original, 'path': target.as_posix()}
            if new['render'].get(key) != imported or not os.path.samefile(original['path'], target):
                raise ValueError('render cached-input hardlink lineage changed')
            lineage.append(dict(id=ident, key=key, original=original, imported=imported,
                                transport='same_inode_hardlink'))
        source_generation = recovery._contained(
            Path(old['render']['input']['path']).parent.parent / 'generation.json', old_root)
        target_generation = recovery._contained(
            Path(new['render']['input']['path']).parent.parent / 'generation.json', new_root)
        source_info = source_generation.stat(follow_symlinks=False)
        target_info = target_generation.stat(follow_symlinks=False)
        if (not source_generation.is_file() or not target_generation.is_file()
                or not os.path.samefile(source_generation, target_generation)
                or source_info.st_size != target_info.st_size
                or source_info.st_mtime_ns != target_info.st_mtime_ns):
            raise ValueError('opaque generation sidecar hardlink identity changed')
        opaque.append(dict(id=ident, source=source_generation.as_posix(), target=target_generation.as_posix(),
                           device=source_info.st_dev, inode=source_info.st_ino, bytes=source_info.st_size,
                           mtime_ns=source_info.st_mtime_ns, transport='opaque_hardlink_not_opened',
                           payload_authentication='deferred_until_label_phase'))
    return lineage, opaque


def _verify_bank_transport(old_rows, new_rows, old_root, new_root):
    old_by_id = p.indexed(old_rows, 'original source rows')
    new_by_id = p.indexed(new_rows, 'phase rows')
    lineage, pointers = [], {}
    for ident, old in old_by_id.items():
        new = new_by_id[ident]
        if old['split'] not in ('train', 'validation'):
            continue
        if old['bank']['status'] == 'unavailable':
            if new['bank'] != old['bank']:
                raise ValueError('unavailable bank failure changed during import')
            continue
        if old['bank']['status'] != 'available' or new['bank'].get('status') != 'available':
            raise ValueError('completed bank source state is not importable')
        for key in ('bank', 'cnos_bank'):
            original = old['bank'][key]
            target = _rebased(new_root, old_root, original)
            imported = {**original, 'path': target.as_posix()}
            if new['bank'].get(key) != imported or not os.path.samefile(original['path'], target):
                raise ValueError('bank cached-input hardlink lineage changed')
            lineage.append(dict(id=ident, key=key, original=original, imported=imported,
                                transport='same_inode_hardlink'))
        obj = old['object_id']
        original_pointer = old_root / f'cnos-{int(obj):06d}.json'
        imported_pointer = new_root / original_pointer.name
        expected = {'bank': new['bank']['cnos_bank']}
        if (p.read_json(original_pointer) != {'bank': old['bank']['cnos_bank']}
                or p.read_json(imported_pointer) != expected):
            raise ValueError('canonical CNOS pointer rebasing changed')
        pointers.setdefault(obj, dict(original=p.pin(original_pointer), imported=p.pin(imported_pointer),
                                      value=expected,
                                      transport='canonical_pointer_rebased_to_imported_bank'))
    return lineage, list(pointers.values())


def _source_mask_index(old, old_root, source_context_sha256, *, verify_bytes=True):
    mask = old['mask']
    # The unchanged RAW mask producer includes failure_reason=None on success.
    # Older imported states omit it. Neither form may carry a failure value.
    if (mask.get('status') != 'available'
            or set(mask) not in ({'status', 'mask', 'mask_index'},
                                 {'status', 'mask', 'mask_index', 'failure_reason'})
            or mask.get('failure_reason') is not None):
        raise ValueError('exact completed observed mask state required')
    mask_path = recovery._contained(mask['mask']['path'], old_root)
    index_path = recovery._contained(mask['mask_index']['path'], old_root)
    if verify_bytes:
        p.verify_pin(mask['mask']); p.verify_pin(mask['mask_index'])
    index = p.read_json(index_path)
    expected = dict(schema='kaggle_raw_observed_mask_v1', automatic=True,
        mask_origin='observed_CNOS_box_then_SAM2_image_predictor',
        context_sha256=None, physical_id=old['id'],
        input=old['render']['input'], video=old['render']['video'], mesh=old['render']['mesh'],
        cnos_bank=old['bank']['cnos_bank'], mask=old['mask']['mask'],
        annotation_inputs=[], test_payload_reads=0)
    expected['context_sha256'] = source_context_sha256
    if set(index) != set(expected) | {'detector', 'prompt_box_xyxy', 'proposal_intermediate_sha256'}:
        raise ValueError('observed mask producer fields changed')
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError('observed mask source identity/render/bank pins changed')
    p.digest(index['proposal_intermediate_sha256'])
    box = index['prompt_box_xyxy']
    if (not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box)
            or not 0 <= box[0] <= box[2] < 640 or not 0 <= box[1] <= box[3] < 480
            or not isinstance(index['detector'], dict)
            or index['detector'].get('bank_sha256') != old['bank']['cnos_bank']['sha256']):
        raise ValueError('observed mask detector/proposal association changed')
    return mask_path, index_path, index


def _import_masks(context, source_rows, *, source_context, sample_combined):
    old_doc = source_context.document(); doc = context.document()
    old_root, root = Path(old_doc['output_root']), Path(doc['output_root'])
    fresh_rows = raw.frozen(context)[3]
    _verify_active_identity(source_rows, raw.physical_rows(*raw.frozen(context)[1:3]))
    by_id = p.indexed(fresh_rows, 'phase physical rows')
    old_by_id = p.indexed(source_rows, 'source physical rows')
    lineage, failures = [], []
    started = time.monotonic()
    for ident, old in old_by_id.items():
        new = by_id[ident]
        if old['split'] not in ('train', 'validation'):
            if any(new[key] != {'status': 'pending'} for key in ('render', 'bank', 'mask', 'seed')):
                raise ValueError('sealed TEST/dev metadata changed before mask import')
            continue
        if old['mask']['status'] == 'unavailable':
            if set(old['mask']) != {'status', 'failure_reason'} or not old['mask']['failure_reason']:
                raise ValueError('explicit original mask failure required')
            new['mask'] = dict(old['mask'])
            failures.append(dict(id=ident, source_mask=old['mask'], imported_mask=new['mask'],
                                 source_mask_index=None, transport='preserved_observed_failure'))
            continue
        if old['mask']['status'] != 'available':
            raise ValueError('source mask outcome is unfinished')
        if old['render']['status'] != 'available' or old['bank']['status'] != 'available':
            raise ValueError('observed mask lacks completed source render and bank')
        _check_storage(context, sample_combined)
        source_mask, source_index, index = _source_mask_index(old, old_root, source_context.sha256)
        from .vision import cv2
        mask_pixels = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if mask_pixels is None or mask_pixels.shape != (480, 640) or not mask_pixels.any():
            raise ValueError('original observed mask PNG has invalid dimensions or is empty')
        target_mask = recovery._contained(root / source_mask.relative_to(old_root), root)
        target_index = recovery._contained(root / source_index.relative_to(old_root), root)
        recovery._link(source_mask, target_mask, set())
        imported_mask = {**old['mask']['mask'], 'path': target_mask.as_posix()}
        p.verify_pin(imported_mask)
        if not os.path.samefile(source_mask, target_mask):
            raise ValueError('observed mask transport did not preserve its inode')
        rewritten = dict(index,
            context_sha256=context.sha256,
            input=new['render']['input'], video=new['render']['video'], mesh=new['render']['mesh'],
            cnos_bank=new['bank']['cnos_bank'], mask=imported_mask)
        _check_storage(context, sample_combined)
        p.atomic_json(target_index, rewritten)
        imported_index = p.pin(target_index)
        new['mask'] = dict(status='available', mask=imported_mask, mask_index=imported_index)
        lineage.append(dict(id=ident, original_mask=old['mask']['mask'], imported_mask=imported_mask,
            original_mask_index=old['mask']['mask_index'], imported_mask_index=imported_index,
            preserved_detector=index['detector'], preserved_prompt_box_xyxy=index['prompt_box_xyxy'],
            preserved_proposal_intermediate_sha256=index['proposal_intermediate_sha256'],
            transport='same_inode_hardlink_with_context_rebound_index'))
    raw._save_rows(context, fresh_rows)
    # The exact unchanged preflight is the admission check for rebased masks.
    preflight_inputs = _preflight(context, sample_combined=sample_combined)
    return dict(schema='visualizeit_raw_cached_capture_role_import_v1', role='masks', mode=MODE,
        context_sha256=context.sha256, source_context_sha256=source_context.sha256,
        source_context=p.pin(old_root / 'context.json'), phase_context=p.pin(root / 'context.json'),
        adapter=p.pin(Path(__file__)), recovery_adapter=p.pin(Path(recovery.__file__)),
        independent_review=None, original_receipt=None, policy=None,
        neural_execution=False, test_payload_reads=0, generation_payload_reads=0,
        measured_operation_seconds=time.monotonic() - started,
        preflight_available_inputs=sorted(preflight_inputs),
        source_mask_counts=_status_counts(source_rows, 'mask'),
        imported_mask_counts=_status_counts(fresh_rows, 'mask'),
        lineage=lineage, preserved_failures=failures)


def _source_auth(context, terminal_evidence_pin, sample_combined):
    terminal = _validate_terminal(context, terminal_evidence_pin)
    history = _capture_history(context, terminal)
    _verify_active_identity(history['rows'], raw.physical_rows(history['spec'], history['plan']))
    # recovery.link_inputs will authenticate actual render/bank bytes before
    # linking. Authenticate the source's exact mask indexes and PNG dimensions
    # here, after the full metadata/TEST barrier and before importing anything.
    old_root = Path(history['doc']['output_root'])
    from .vision import cv2
    for row in history['rows']:
        if row['split'] not in ('train', 'validation') or row['mask']['status'] != 'available':
            continue
        _check_storage(context, sample_combined)
        mask_path, _, _ = _source_mask_index(row, old_root, context.sha256)
        pixels = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if pixels is None or pixels.shape != (480, 640) or not pixels.any():
            raise ValueError('original observed mask PNG has invalid dimensions or is empty')
    return terminal, history


def _role_common(context, source_context, review_pin, policy_pin, old_stage_pin, role):
    doc = context.document(); old = source_context.document()
    return dict(schema='visualizeit_raw_cached_capture_role_import_v1', role=role, mode=MODE,
        context_sha256=context.sha256, source_context_sha256=source_context.sha256,
        source_context=p.pin(Path(old['output_root']) / 'context.json'),
        phase_context=p.pin(Path(doc['output_root']) / 'context.json'),
        adapter=p.pin(Path(__file__)), recovery_adapter=p.pin(Path(recovery.__file__)),
        independent_review=review_pin, original_receipt=old_stage_pin, policy=policy_pin,
        neural_execution=False, test_payload_reads=0, generation_payload_reads=0)


def _charge_nonstage_elapsed(context, field, elapsed):
    """Charge measured adapter metadata/validation time to the current RAW clock."""
    if field not in ('phase_metadata_elapsed_seconds', 'phase_validation_elapsed_seconds'):
        raise ValueError('unknown non-stage phase clock component')
    root = Path(context.document()['output_root'])
    freeze = raw.read_json(root / 'freeze.json')
    prior = _seconds(freeze.get(field, 0.), field)
    freeze[field] = prior + _seconds(elapsed, field)
    freeze['elapsed_seconds'] = sum(_seconds(freeze.get(key, 0.), key) for key in (
        'phase_setup_elapsed_seconds', 'phase_metadata_elapsed_seconds',
        'phase_validation_elapsed_seconds'))
    raw.write_json(context, root / 'freeze.json', freeze, replace=True)


def _augment_stage(context, name, original_receipt, review_pin, policy_pin, role_pin):
    started = time.monotonic()
    root = Path(context.document()['output_root'])
    path = root / ('stage-' + name + '.json')
    value = raw.read_json(path)
    if (value.get('stage') != name or value.get('context_sha256') != context.sha256
            or value.get('status') != 'completed' or value.get('error') is not None
            or value.get('test_payload_reads') != 0 or value.get('training_ready') is not False):
        raise ValueError('guarded cached-input import stage did not complete cleanly')
    _seconds(value.get('elapsed_seconds'), 'current phase stage')
    value.update(mode=MODE, neural_execution=False, original_receipt=original_receipt,
        independent_review=review_pin, adapter=p.pin(Path(__file__)),
        cache_provenance=dict(schema='visualizeit_raw_cached_capture_stage_provenance_v1',
            mode=MODE, policy=policy_pin, role_receipt=role_pin, adapter=p.pin(Path(__file__)),
            recovery_adapter=p.pin(Path(recovery.__file__)), independent_review=review_pin,
            original_receipt=original_receipt))
    raw.write_json(context, path, value, replace=True)
    _charge_nonstage_elapsed(context, 'phase_metadata_elapsed_seconds', time.monotonic() - started)
    return p.pin(path)


def prepare_phase(source_context, context, review_pin, terminal_evidence_pin, *, sample_combined):
    """Create a fresh phase and import authenticated render/bank/mask inputs.

    The current phase receives its own 7200-second limit. Source elapsed time
    remains pinned and reported separately; it is never reset or counted as a
    claim that the original cumulative limit was met.
    """
    if type(source_context) is not raw.PortableContext or type(context) is not raw.PortableContext:
        raise TypeError('exact immutable portable source and phase contexts required')
    began = time.monotonic()
    review = approve(context, review_pin)
    source_doc = source_context.document(); doc = context.document()
    _same_phase_context(source_doc, doc)
    old_root, root, scratch = (Path(source_doc['output_root']), Path(doc['output_root']),
                               Path(doc['scratch_root']))
    if root.exists() or scratch.exists():
        raise FileExistsError('fresh capture-phase output and scratch roots required')
    evidence, history = _source_auth(source_context, terminal_evidence_pin, sample_combined)
    # Recipe and all role/identity metadata are checked before link_inputs can
    # stat/hash any active payload or inspect any generation sidecar.
    if history['spec'] != raw.read_json(old_root / 'recipe.json'):
        raise ValueError('source recipe changed during terminal admission')
    fresh_template = raw.physical_rows(history['spec'], history['plan'])
    _verify_active_identity(history['rows'], fresh_template)
    _check_storage(context, sample_combined)
    raw.freeze(old_root / 'recipe.json', p.sha256(old_root / 'recipe.json'), context)
    scratch.mkdir(parents=True, exist_ok=False)

    original_freeze = raw.read_json(root / 'freeze.json')
    setup_seconds = time.monotonic() - began
    history_pins = history['source_pins']
    policy = dict(schema='visualizeit_raw_cached_capture_phase_policy_v1', mode=MODE,
        context_sha256=context.sha256, source_context_sha256=source_context.sha256,
        source_context=p.pin(old_root / 'context.json'), phase_context=p.pin(root / 'context.json'),
        original_receipt_pins=history_pins,
        terminal_evidence=terminal_evidence_pin,
        independent_review=review_pin, adapter=p.pin(Path(__file__)),
        recovery_adapter=p.pin(Path(recovery.__file__)),
        phase_budget=dict(scope='separate_capture_phase_budget', max_seconds=7200,
            phase_started_unix_s=time.time(), setup_elapsed_seconds=setup_seconds,
            source_raw_stage_elapsed_seconds=history['source_raw_stage_elapsed_seconds'],
            source_recovery_import_elapsed_seconds=history['source_recovery_import_elapsed_seconds'],
            source_capture_actual_elapsed_seconds=history['outer_elapsed_seconds'],
            source_capture_timeout_upper_bound_seconds=history['outer_upper_bound_seconds'],
            source_watcher_elapsed_seconds=history['watcher_elapsed_seconds'],
            source_historical_measured_elapsed_seconds=history['measured_historical_elapsed_seconds'],
            source_historical_upper_bound_seconds=history['historical_upper_bound_seconds'],
            combined_historical_plus_phase_time_reported=True,
            original_cumulative_budget_compliance=False),
        original_failure_counts=dict(mask=history['mask_counts'], seed=history['seed_counts']),
        imported_masks_start_from_observed_source=True, old_partial_crops_imported=False,
        old_seed_states_imported=False, old_source_modified=False, test_payload_reads=0,
        generation_payload_reads=0, training_ready=False, neural_execution=False)
    policy_path = root / POLICY_FILE
    p.atomic_json(policy_path, policy)
    policy_pin = p.pin(policy_path)
    # Make the ordinary RAW clock include all actual preparation time before
    # the first import stage, while preserving the separately measured freeze.
    freeze = original_freeze
    freeze['raw_freeze_elapsed_seconds'] = original_freeze['elapsed_seconds']
    freeze['phase_setup_elapsed_seconds'] = time.monotonic() - began
    freeze['phase_metadata_elapsed_seconds'] = 0.
    freeze['phase_validation_elapsed_seconds'] = 0.
    freeze['elapsed_seconds'] = freeze['phase_setup_elapsed_seconds']
    freeze['phase_mode'] = MODE
    freeze['phase_policy_sha256'] = policy_pin['sha256']
    raw.write_json(context, root / 'freeze.json', freeze, replace=True)

    old_stage_pins = history_pins['stage_receipts']
    check_storage = lambda: _check_storage(context, sample_combined)
    # First stage transports render, banks and opaque generation hardlinks by
    # the original metadata-first authenticated recovery helper.
    with raw.stage(context, 'render', cloud_execution=True):
        started = time.monotonic()
        fresh = raw.frozen(context)[3]
        imported = recovery.link_inputs(history['rows'], fresh, old_root=old_root,
            new_root=root, check_storage=check_storage)
        raw._save_rows(context, imported['rows'])
        current = raw.frozen(context)[3]
        render_lineage, opaque = _verify_render_transport(history['rows'], current, old_root, root)
        bank_lineage, pointers = _verify_bank_transport(history['rows'], current, old_root, root)
        role = _role_common(context, source_context, review_pin, policy_pin,
                            old_stage_pins['render'], 'render')
        role.update(measured_operation_seconds=time.monotonic() - started,
            render_lineage=render_lineage, opaque_generation=opaque,
            imported_bank_lineage=bank_lineage, imported_cnos_pointers=pointers,
            transport='same_inode_hardlinks_and_canonical_pointer_rebase',
            prior_attempts_preserved=True)
        p.atomic_json(root / ROLE_FILES['render'], role)
    render_role_pin = p.pin(root / ROLE_FILES['render'])
    render_stage_pin = _augment_stage(context, 'render', old_stage_pins['render'],
                                      review_pin, policy_pin, render_role_pin)

    with raw.stage(context, 'banks', cloud_execution=True):
        started = time.monotonic()
        current = raw.frozen(context)[3]
        bank_lineage, pointers = _verify_bank_transport(history['rows'], current, old_root, root)
        role = _role_common(context, source_context, review_pin, policy_pin,
                            old_stage_pins['banks'], 'banks')
        role.update(measured_operation_seconds=time.monotonic() - started,
            bank_lineage=bank_lineage, cnos_pointers=pointers,
            validation='authenticated_imported_bank_and_rebased_pointer',
            prior_attempts_preserved=True)
        p.atomic_json(root / ROLE_FILES['banks'], role)
    banks_role_pin = p.pin(root / ROLE_FILES['banks'])
    banks_stage_pin = _augment_stage(context, 'banks', old_stage_pins['banks'],
                                     review_pin, policy_pin, banks_role_pin)

    with raw.stage(context, 'masks', cloud_execution=True):
        role = _import_masks(context, history['rows'], source_context=source_context,
                             sample_combined=sample_combined)
        role.update(independent_review=review_pin,
            original_receipt=old_stage_pins['masks'], policy=policy_pin)
        p.atomic_json(root / ROLE_FILES['masks'], role)
    masks_role_pin = p.pin(root / ROLE_FILES['masks'])
    masks_stage_pin = _augment_stage(context, 'masks', old_stage_pins['masks'],
                                     review_pin, policy_pin, masks_role_pin)

    # The initial import admission is the only point where this adapter may
    # charge phase-validation time or publish its immutable receipt. Every
    # later operator check is read-only so RAW capture/seal artifacts stay
    # byte-stable once the producer has pinned them.
    result = _validate_phase(context, review_pin, finalize=True)
    return dict(context=p.pin(root / 'context.json'), phase_context=result['phase_context'],
                source_context=result['source_context'],
                frozen_recipe_identity=result['frozen_recipe_identity'],
                source_identity_pins=result['source_identity_pins'],
                phase_identity_pins=result['phase_identity_pins'],
                lineage_pins=result['lineage_pins'], policy=policy_pin,
                phase_receipt=p.pin(root / PHASE_FILE),
                stage_receipts=dict(render=render_stage_pin, banks=banks_stage_pin, masks=masks_stage_pin),
                review=review_pin, terminal_evidence=terminal_evidence_pin,
                current_phase_elapsed_seconds=result['current_phase_elapsed_seconds'],
                combined_historical_plus_phase_seconds=result['combined_historical_plus_phase_seconds'],
                original_cumulative_budget_compliance=False)


def _validate_source_lineage(policy, context, review_pin):
    source_pin = policy.get('source_context')
    source_path = p.verify_pin(source_pin)
    source_context = raw.read_context(source_path, policy['source_context_sha256'])
    old_doc = source_context.document(); doc = context.document()
    _same_phase_context(old_doc, doc)
    evidence_pin = policy.get('terminal_evidence')
    evidence = _validate_terminal(source_context, evidence_pin)
    history = _capture_history(source_context, evidence)
    if policy.get('original_receipt_pins') != history['source_pins']:
        raise ValueError('pinned original capture history changed')
    phase_budget = policy.get('phase_budget', {})
    if (phase_budget.get('source_raw_stage_elapsed_seconds') != history['source_raw_stage_elapsed_seconds']
            or phase_budget.get('source_recovery_import_elapsed_seconds') != history['source_recovery_import_elapsed_seconds']
            or phase_budget.get('source_capture_actual_elapsed_seconds') != history['outer_elapsed_seconds']
            or phase_budget.get('source_capture_timeout_upper_bound_seconds') != history['outer_upper_bound_seconds']
            or phase_budget.get('source_watcher_elapsed_seconds') != history['watcher_elapsed_seconds']
            or phase_budget.get('source_historical_measured_elapsed_seconds') != history['measured_historical_elapsed_seconds']
            or phase_budget.get('source_historical_upper_bound_seconds') != history['historical_upper_bound_seconds']):
        raise ValueError('source historical elapsed time changed')
    if policy.get('original_failure_counts') != dict(mask=history['mask_counts'], seed=history['seed_counts']):
        raise ValueError('source mask/capture failure counts changed')
    _verify_active_identity(history['rows'], raw.physical_rows(history['spec'], history['plan']))
    # Reauthenticate original mask bytes/indexes after the sealed-role barrier.
    # Shape/nonempty validation is performed once during import and again by
    # the unchanged destination preflight over the same-inode hardlinks.
    old_root = Path(old_doc['output_root'])
    for row in history['rows']:
        if row['split'] not in ('train', 'validation') or row['mask']['status'] != 'available':
            continue
        _source_mask_index(row, old_root, source_context.sha256)
    return source_context, history


def validate_phase(context, review_pin):
    """Read-only recheck, including after clean RAW capture/seal stages.

    The cached-capture admission receipt and freeze clock are immutable after
    :func:`prepare_phase`. Later capture/seal receipts are included in the
    returned clock report but never cause either admission artifact to be
    rewritten.
    """
    return _validate_phase(context, review_pin, finalize=False)


def _validate_phase(context, review_pin, *, finalize):
    """Validate imported inputs and, once only, finalize initial admission."""
    validation_started = time.monotonic()
    review = approve(context, review_pin)
    doc = context.document(); root = Path(doc['output_root'])
    phase_path = root / PHASE_FILE
    if finalize and phase_path.exists():
        raise FileExistsError('cached-capture phase admission was already finalized')
    if not finalize and not phase_path.is_file():
        raise ValueError('read-only phase validation requires a finalized admission receipt')
    policy_path = root / POLICY_FILE
    policy = p.read_json(policy_path)
    policy_pin = p.pin(policy_path)
    if (policy.get('schema') != 'visualizeit_raw_cached_capture_phase_policy_v1'
            or policy.get('mode') != MODE or policy.get('context_sha256') != context.sha256
            or policy.get('phase_context') != p.pin(root / 'context.json')
            or policy.get('adapter') != p.pin(Path(__file__))
            or policy.get('recovery_adapter') != p.pin(Path(recovery.__file__))
            or policy.get('independent_review') != review_pin
            or policy.get('terminal_evidence') is None
            or policy.get('phase_budget', {}).get('scope') != 'separate_capture_phase_budget'
            or policy.get('phase_budget', {}).get('max_seconds') != 7200
            or policy.get('phase_budget', {}).get('original_cumulative_budget_compliance') is not False
            or policy.get('old_source_modified') is not False
            or policy.get('old_partial_crops_imported') is not False
            or policy.get('old_seed_states_imported') is not False
            or policy.get('neural_execution') is not False
            or policy.get('test_payload_reads') != 0
            or policy.get('generation_payload_reads') != 0
            or policy.get('training_ready') is not False):
        raise ValueError('cached-capture phase policy or separate budget binding changed')
    source_context, history = _validate_source_lineage(policy, context, review_pin)
    _, spec, plan, rows = raw.frozen(context)
    old_doc = source_context.document()
    if spec != history['spec'] or plan != history['plan']:
        raise ValueError('fresh capture-phase recipe or split changed')
    _verify_active_identity(history['rows'], raw.physical_rows(spec, plan))
    old_root = Path(old_doc['output_root'])
    render_lineage, opaque = _verify_render_transport(history['rows'], rows, old_root, root)
    bank_lineage, pointers = _verify_bank_transport(history['rows'], rows, old_root, root)
    masks = p.read_json(root / ROLE_FILES['masks'])
    active_source = p.indexed(history['rows'], 'source physical rows')
    active_dest = p.indexed(rows, 'phase physical rows')
    mask_lineage, preserved_failures = [], []
    for ident, old in active_source.items():
        new = active_dest[ident]
        if old['split'] not in ('train', 'validation'):
            if any(new[key] != {'status': 'pending'} for key in ('render', 'bank', 'mask', 'seed')):
                raise ValueError('sealed TEST/dev payload state changed')
            continue
        if old['mask']['status'] == 'unavailable':
            if new['mask'] != old['mask']:
                raise ValueError('explicit observed-mask failure count/reason changed')
            preserved_failures.append(dict(id=ident, source_mask=old['mask'], imported_mask=new['mask'],
                                           source_mask_index=None,
                                           transport='preserved_observed_failure'))
            continue
        source_mask, source_index, index = _source_mask_index(old, old_root,
                                                              source_context.sha256)
        imported = new['mask']
        if imported.get('status') != 'available':
            raise ValueError('authenticated source observed mask was lost')
        dest_mask = recovery._contained(imported['mask']['path'], root)
        dest_index = recovery._contained(imported['mask_index']['path'], root)
        p.verify_pin(old['mask']['mask']); p.verify_pin(imported['mask'])
        p.verify_pin(imported['mask_index'])
        expected_mask = {**old['mask']['mask'], 'path': dest_mask.as_posix()}
        expected_rewrite = dict(index, context_sha256=context.sha256,
            input=new['render']['input'], video=new['render']['video'], mesh=new['render']['mesh'],
            cnos_bank=new['bank']['cnos_bank'], mask=expected_mask)
        if (imported['mask'] != expected_mask or not os.path.samefile(source_mask, dest_mask)
                or p.read_json(dest_index) != expected_rewrite):
            raise ValueError('observed mask hardlink or rebased producer index changed')
        mask_lineage.append(dict(id=ident, original_mask=old['mask']['mask'], imported_mask=imported['mask'],
            original_mask_index=old['mask']['mask_index'], imported_mask_index=imported['mask_index'],
            preserved_detector=index['detector'], preserved_prompt_box_xyxy=index['prompt_box_xyxy'],
            preserved_proposal_intermediate_sha256=index['proposal_intermediate_sha256'],
            transport='same_inode_hardlink_with_context_rebound_index'))
    counts = _status_counts(rows, 'mask')
    source_counts = history['mask_counts']
    if counts != source_counts:
        raise ValueError('source and imported observed-mask success/failure counts differ')
    if (masks.get('role') != 'masks' or masks.get('mode') != MODE
            or masks.get('context_sha256') != context.sha256
            or masks.get('source_context_sha256') != source_context.sha256
            or masks.get('source_mask_counts') != source_counts
            or masks.get('imported_mask_counts') != counts
            or masks.get('preflight_available_inputs') != sorted(
                ident for ident, row in active_dest.items()
                if row['split'] in ('train', 'validation')
                and all(row[key]['status'] == 'available' for key in ('render', 'bank', 'mask')))
            or masks.get('lineage') != mask_lineage
            or masks.get('preserved_failures') != preserved_failures):
        raise ValueError('cached mask role receipt differs from authenticated inventory')

    phase_names = ('render', 'banks', 'masks')
    expected_stage_names = {'stage-' + name + '.json' for name in phase_names}
    actual_stage_names = {path.name for path in root.glob('stage-*.json')}
    later_stage_names = {'stage-capture.json', 'stage-seal-capture.json'}
    if (not expected_stage_names.issubset(actual_stage_names)
            or actual_stage_names - expected_stage_names - later_stage_names
            or ('stage-seal-capture.json' in actual_stage_names
                and 'stage-capture.json' not in actual_stage_names)
            or (finalize and (actual_stage_names != expected_stage_names
                              or (root / PHASE_FILE).exists()))):
        raise ValueError('unexpected RAW phase stages or repeated admission finalization')
    stage_pins, role_pins, stage_elapsed = {}, {}, {}
    for name in phase_names:
        role_path = root / ROLE_FILES[name]
        role = p.read_json(role_path)
        role_pin = p.pin(role_path)
        stage_path = root / ('stage-' + name + '.json')
        stage = raw.read_json(stage_path)
        stage_pin = p.pin(stage_path)
        old_stage_pin = history['source_pins']['stage_receipts'][name]
        if (role.get('role') != name or role.get('mode') != MODE
                or role.get('context_sha256') != context.sha256
                or role.get('source_context_sha256') != source_context.sha256
                or role.get('source_context') != p.pin(old_root / 'context.json')
                or role.get('phase_context') != p.pin(root / 'context.json')
                or role.get('adapter') != p.pin(Path(__file__))
                or role.get('recovery_adapter') != p.pin(Path(recovery.__file__))
                or role.get('independent_review') != review_pin
                or role.get('original_receipt') != old_stage_pin
                or role.get('policy') != policy_pin
                or role.get('neural_execution') is not False
                or role.get('test_payload_reads') != 0
                or role.get('generation_payload_reads') != 0):
            raise ValueError('per-role import receipt provenance changed: ' + name)
        role_elapsed = _seconds(role.get('measured_operation_seconds'), 'role import elapsed')
        if (stage.get('stage') != name or stage.get('context_sha256') != context.sha256
                or stage.get('status') != 'completed' or stage.get('error') is not None
                or stage.get('test_payload_reads') != 0 or stage.get('training_ready') is not False
                or stage.get('mode') != MODE or stage.get('neural_execution') is not False
                or stage.get('original_receipt') != old_stage_pin
                or stage.get('independent_review') != review_pin
                or stage.get('adapter') != p.pin(Path(__file__))
                or stage.get('cache_provenance') != dict(
                    schema='visualizeit_raw_cached_capture_stage_provenance_v1', mode=MODE,
                    policy=policy_pin, role_receipt=role_pin, adapter=p.pin(Path(__file__)),
                    recovery_adapter=p.pin(Path(recovery.__file__)), independent_review=review_pin,
                    original_receipt=old_stage_pin)):
            raise ValueError('guarded import-stage provenance changed: ' + name)
        stage_seconds = _seconds(stage.get('elapsed_seconds'), 'current phase stage')
        if role_elapsed > stage_seconds:
            raise ValueError('role operation elapsed exceeds its guarded stage measurement')
        stage_pins[name] = stage_pin
        role_pins[name] = role_pin
        stage_elapsed[name] = stage_seconds
    # Later neural execution and sealing use the unchanged RAW producer. Their
    # own clean receipts remain ordinary RAW receipts and are counted once in
    # the phase clock, without adding cache-provenance fields or rewriting the
    # immutable cached-input admission.
    for name in ('capture', 'seal-capture'):
        stage_path = root / ('stage-' + name + '.json')
        if not stage_path.exists():
            continue
        stage = raw.read_json(stage_path)
        recovery._clean_receipt(context, name, stage)
        stage_elapsed[name] = _seconds(stage.get('elapsed_seconds'), 'later RAW stage')
        stage_pins[name] = p.pin(stage_path)
    if 'capture' in stage_elapsed:
        capture_receipt = p.read_json(root / 'raw' / 'capture-receipt.json')
        from .quality_adaptation_refinement_states import state_rows
        expected_states = state_rows(dict(rows=rows))
        if (capture_receipt.get('schema') != 'kaggle_raw_capture_receipt_v1'
                or capture_receipt.get('context_sha256') != context.sha256
                or capture_receipt.get('input_plan_sha256') != p.sha256(root / 'input-plan.json')
                or capture_receipt.get('fatal_error') is not None
                or capture_receipt.get('cleanup_errors') != []
                or capture_receipt.get('physical_frames_requested') != len(rows)
                or capture_receipt.get('refinement_states_requested') != len(expected_states)
                or capture_receipt.get('test_payload_reads') != 0
                or capture_receipt.get('training_ready') is not False):
            raise ValueError('later RAW capture receipt is incomplete or unsafe')
    if 'seal-capture' in stage_elapsed:
        seal = p.read_json(root / 'raw-seal.json')
        p.check_seal(seal, 'kaggle_raw_capture_seal_v1')
        if (seal.get('context_sha256') != context.sha256
                or seal.get('training_ready') is not False
                or seal.get('test_payload_reads') != 0
                or seal.get('capture_fatal_error') is not None
                or seal.get('cleanup_errors') != []):
            raise ValueError('later RAW seal is incomplete or unsafe')
        sealed_artifacts = seal.get('artifacts', {})
        for name in ('render', 'banks', 'masks', 'capture'):
            if sealed_artifacts.get('stage-' + name + '.json') != p.pin(
                    root / ('stage-' + name + '.json')):
                raise ValueError('RAW seal does not retain the clean phase stage: ' + name)
    if (render_lineage != p.read_json(root / ROLE_FILES['render']).get('render_lineage')
            or opaque != p.read_json(root / ROLE_FILES['render']).get('opaque_generation')
            or bank_lineage != p.read_json(root / ROLE_FILES['banks']).get('bank_lineage')
            or pointers != p.read_json(root / ROLE_FILES['banks']).get('cnos_pointers')):
        raise ValueError('render/bank per-role transport lineage changed')
    render_role = p.read_json(root / ROLE_FILES['render'])
    if (render_role.get('imported_bank_lineage') != bank_lineage
            or render_role.get('imported_cnos_pointers') != pointers):
        raise ValueError('render import receipt lost bank-transport history')

    # This exact unchanged producer preflight checks all rebased render/bank
    # pins, new mask indexes and mask dimensions without opening generation data.
    _preflight(context)
    if finalize:
        _charge_nonstage_elapsed(context, 'phase_validation_elapsed_seconds',
                                 time.monotonic() - validation_started)
    freeze = p.read_json(root / 'freeze.json')
    setup_elapsed = _seconds(freeze.get('phase_setup_elapsed_seconds'), 'capture phase setup')
    if (freeze.get('phase_mode') != MODE or freeze.get('phase_policy_sha256') != policy_pin['sha256']
            or freeze.get('elapsed_seconds') != setup_elapsed + _seconds(
                freeze.get('phase_metadata_elapsed_seconds', 0.), 'phase metadata time') + _seconds(
                freeze.get('phase_validation_elapsed_seconds', 0.), 'phase validation time')
            or freeze.get('raw_freeze_elapsed_seconds') is None
            or setup_elapsed > _seconds(policy['phase_budget']['setup_elapsed_seconds']) + 1.0):
        raise ValueError('phase setup elapsed was reset or changed')
    _seconds(freeze['raw_freeze_elapsed_seconds'], 'raw freeze operation')
    current_elapsed = freeze['elapsed_seconds'] + sum(stage_elapsed.values())
    if current_elapsed > 7200:
        raise ValueError('new capture phase cumulative clock exhausted')
    historical = history['historical_upper_bound_seconds']
    source_root = Path(old_doc['output_root'])
    phase_freeze = p.read_json(root / 'freeze.json')
    source_freeze = history['freeze']
    source_identity_pins = history['source_pins']
    phase_identity_pins = dict(context=p.pin(root / 'context.json'),
        recipe=p.pin(root / 'recipe.json'), scale_plan=p.pin(root / 'scale-plan.json'),
        split=p.pin(root / 'split.json'), freeze=p.pin(root / 'freeze.json'),
        physical_index=p.pin(root / 'physical-index.json'))
    frozen_recipe_identity = dict(recipe_sha256=source_freeze['recipe_sha256'],
        recipe_file_sha256=source_freeze['recipe_file_sha256'],
        phase_recipe_sha256=phase_freeze['recipe_sha256'],
        phase_recipe_file_sha256=phase_freeze['recipe_file_sha256'],
        source_scale_plan_sha256=source_freeze['scale_plan_sha256'],
        phase_scale_plan_sha256=phase_freeze['scale_plan_sha256'],
        source_split_sha256=p.sha256(source_root / 'split.json'),
        phase_split_sha256=p.sha256(root / 'split.json'),
        physical_frames=source_freeze['physical_frames'],
        refinement_states=source_freeze['refinement_states'])
    if (frozen_recipe_identity['recipe_sha256'] != frozen_recipe_identity['phase_recipe_sha256']
            or frozen_recipe_identity['recipe_file_sha256'] != frozen_recipe_identity['phase_recipe_file_sha256']
            or frozen_recipe_identity['source_scale_plan_sha256'] != frozen_recipe_identity['phase_scale_plan_sha256']
            or frozen_recipe_identity['source_split_sha256'] != frozen_recipe_identity['phase_split_sha256']):
        raise ValueError('phase frozen recipe or split identity differs from original')
    lineage_pins = dict(policy=policy_pin, terminal_evidence=policy['terminal_evidence'],
        independent_review=review_pin, adapter=p.pin(Path(__file__)),
        recovery_adapter=p.pin(Path(recovery.__file__)), source_original_receipts=history['source_pins'],
        phase_stage_receipts=stage_pins, role_receipts=role_pins)
    final = dict(schema='visualizeit_raw_cached_capture_phase_validation_v1', mode=MODE,
        context_sha256=context.sha256, source_context_sha256=source_context.sha256,
        phase_context=phase_identity_pins['context'],
        source_context=source_identity_pins['context'],
        frozen_recipe_identity=frozen_recipe_identity,
        source_identity_pins=source_identity_pins,
        phase_identity_pins=phase_identity_pins,
        lineage_pins=lineage_pins,
        policy=policy_pin, terminal_evidence=policy['terminal_evidence'],
        independent_review=review_pin, adapter=p.pin(Path(__file__)),
        source_historical_elapsed_seconds=historical,
        source_historical_measured_elapsed_seconds=history['measured_historical_elapsed_seconds'],
        source_raw_stage_elapsed_seconds=history['source_raw_stage_elapsed_seconds'],
        source_recovery_import_elapsed_seconds=history['source_recovery_import_elapsed_seconds'],
        source_capture_actual_elapsed_seconds=history['outer_elapsed_seconds'],
        source_capture_timeout_upper_bound_seconds=history['outer_upper_bound_seconds'],
        source_watcher_elapsed_seconds=history['watcher_elapsed_seconds'],
        current_phase_elapsed_seconds=current_elapsed,
        combined_historical_plus_phase_seconds=historical + current_elapsed,
        original_cumulative_budget_compliance=False,
        phase_budget_seconds=7200, phase_stages=stage_elapsed,
        stage_receipts=stage_pins, role_receipts=role_pins,
        source_mask_counts=source_counts, imported_mask_counts=counts,
        source_seed_history_counts=history['seed_counts'],
        source_capture_receipt=history['source_pins']['capture_receipt'],
        source_outer_capture_observation=history['source_pins']['outer_capture_observation'],
        test_payload_reads=0, generation_payload_reads=0,
        old_partial_crops_imported=False, old_seed_states_imported=False,
        neural_execution=False, training_ready=False)
    phase_path = root / PHASE_FILE
    if finalize:
        if phase_path.exists():
            raise FileExistsError('cached-capture phase admission was already finalized')
        raw.write_json(context, phase_path, final)
    else:
        admission = p.read_json(phase_path)
        if (admission.get('schema') != final['schema']
                or admission.get('context_sha256') != final['context_sha256']
                or admission.get('policy') != policy_pin
                or admission.get('independent_review') != review_pin
                or admission.get('phase_context') != final['phase_context']
                or admission.get('phase_identity_pins', {}).get('freeze') != p.pin(root / 'freeze.json')
                or admission.get('stage_receipts') != {name: stage_pins[name] for name in phase_names}
                or admission.get('role_receipts') != {name: role_pins[name] for name in phase_names}
                or admission.get('source_identity_pins') != source_identity_pins
                or admission.get('source_context') != final['source_context']):
            raise ValueError('immutable cached-input admission receipt changed')
        final['admission_receipt'] = p.pin(phase_path)
    return final

