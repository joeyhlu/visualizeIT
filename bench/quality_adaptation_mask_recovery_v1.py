"""Explicit RAW recovery transport and destination-aware mask resource checks.

Original RAW/model implementations remain unchanged. A separately reviewed
operator must authenticate this module, supervise the complete retained tree,
and bind its use into stage receipts before inference or child admission.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import shutil
import stat
import time

from . import quality_adaptation_portable as raw
from . import quality_adaptation_portable_prepare as p

SOURCE_NAME = 'bench/quality_adaptation_mask_recovery_v1.py'
MARKER = 'stage-mask-resource-source-v1.json'
IMPORT_MARKER = 'stage-authenticated-recovery-import-v1.json'
REVIEW_SCHEMA = 'visualizeit_raw_mask_recovery_source_review_v1'
GIB = 1024**3


def approve(context, review_pin):
    review = p.metadata(review_pin)
    doc = raw.verify_sources(context)
    if (review.get('schema') != REVIEW_SCHEMA or review.get('decision') != 'APPROVE'
            or review.get('original_raw_source_files') != doc['source_files']
            or review.get('files', {}).get(SOURCE_NAME) != p.sha256(Path(__file__))):
        raise ValueError('independent exact recovery-source approval required')
    return review


def allocation_check(*, resident, limit, reserve, future_bytes, scratch_free, other_free,
                     output_future=0, output_free=None, same_device=False):
    """Reserve future proposals only on their actual destination filesystem."""
    values = (resident, limit, reserve, future_bytes, scratch_free, output_future, *other_free)
    if output_free is not None:
        values += (output_free,)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('nonnegative integer storage measurements required')
    if reserve < GIB:
        raise ValueError('at least1GiB physical reserve required')
    if not 0 < limit <= 32 * GIB:
        raise ValueError('at most32GiB combined storage required')
    if resident + future_bytes + output_future + reserve > limit:
        raise raw.ResourceError('combined retained storage cannot fit proposal envelope')
    scratch_needed = future_bytes + (output_future if same_device else 0) + reserve
    if (scratch_free < scratch_needed or any(free < reserve for free in other_free)
            or output_future and (output_free is None or output_free < output_future + reserve)):
        raise raw.ResourceError('destination or retained filesystem reserve exhausted')
    return dict(resident_bytes=resident, future_bytes=future_bytes,
                output_future_bytes=output_future,
                storage_limit_bytes=limit, reserve_bytes=reserve,
                scratch_free_bytes=scratch_free, other_free_bytes=list(other_free))


def proposal_reserve(context, future_bytes, *, sample_combined, output_future=0):
    # Retain the original live RAW/cache/physical-base guards. The separately
    # bounded outer operator owns the complete runtime/old/new retained union.
    raw.resource_check(context)
    doc = context.document()
    roots = [Path(doc[key]) for key in ('output_root', 'scratch_root', 'model_cache')]
    scratch = roots[1]
    while not scratch.exists():
        scratch = scratch.parent
    other_free, seen = [], {scratch.stat().st_dev}
    for path in roots:
        while not path.exists():
            path = path.parent
        device = path.stat().st_dev
        if device not in seen:
            other_free.append(shutil.disk_usage(path).free)
            seen.add(device)
    measurement = sample_combined()
    output = roots[0]
    while not output.exists():
        output = output.parent
    return allocation_check(resident=measurement['resident_bytes'],
        limit=doc['limits']['storage_bytes'], reserve=doc['limits']['reserve_bytes'],
        future_bytes=future_bytes, scratch_free=shutil.disk_usage(scratch).free,
        other_free=other_free, output_future=output_future,
        output_free=shutil.disk_usage(output).free,
        same_device=output.stat().st_dev == scratch.stat().st_dev)


def import_rows(expected, old_rows):
    """Check every role/identity before any active payload path is traversed."""
    wanted = p.indexed(expected, 'fresh physical inventory')
    actual = p.indexed(old_rows, 'previous physical inventory')
    if list(wanted) != list(actual):
        raise ValueError('complete canonical physical inventory required')
    for ident, fresh in wanted.items():
        previous = actual[ident]
        identity = {k: v for k, v in fresh.items() if k not in ('render', 'bank', 'mask', 'seed')}
        if set(previous) != set(fresh) or any(previous[k] != v for k, v in identity.items()):
            raise ValueError('imported object/frame/mesh/split/timestamp identity differs')
        if fresh['split'] in ('test', 'dev'):
            if previous != fresh:
                raise ValueError('sealed TEST/dev metadata changed before import')
            continue
        if fresh['split'] not in ('train', 'validation'):
            raise ValueError('unknown physical role')
        if any(fresh[k] != {'status': 'pending'} for k in ('render', 'bank', 'mask', 'seed')):
            raise ValueError('fresh recovery states must all be pending')
    # No path traversal (including symlink inspection) before ALL sealed rows
    # and fresh destination states have passed the metadata-only barrier.
    for ident, fresh in wanted.items():
        if fresh['split'] not in ('train', 'validation'):
            continue
        previous = actual[ident]
        for kind, keys in (('render', ('input', 'video', 'mesh')), ('bank', ('bank', 'cnos_bank'))):
            value = previous[kind]
            if value.get('status') == 'available':
                if set(value) != {'status', *keys}:
                    raise ValueError('unexpected imported observed artifact fields')
                for key in keys:
                    pin = value[key]
                    if (not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256'}
                            or type(pin['bytes']) is not int or not 0 < pin['bytes'] <= raw.MAX_ARTIFACT
                            or not isinstance(pin['path'], str)):
                        raise ValueError('bounded exact imported descriptor required')
                    p.digest(pin['sha256'])
            elif value.get('status') != 'unavailable' or not value.get('failure_reason'):
                raise ValueError('unfinished render/bank cannot be imported')
    return wanted, actual


def _contained(path, root):
    path, root = Path(path), Path(root)
    if (not path.is_absolute() or '..' in path.parts or path != Path(os.path.abspath(path))
            or not path.is_relative_to(root) or path == root):
        raise ValueError('artifact escapes its declared RAW root')
    p.no_links(root)
    return p.no_links(path)


def _link(source, target, created):
    if target in created:
        if not os.path.samefile(source, target):
            raise ValueError('repeated import target has a different inode')
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError('preserve existing import target')
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError('RAW artifact import requires same-filesystem hardlinks')
    os.link(source, target)
    created.add(target)
    if not os.path.samefile(source, target):
        raise ValueError('RAW import did not preserve inode identity')


def link_inputs(old_rows, fresh_rows, *, old_root, new_root, check_storage):
    """Import render/bank artifacts and opaque generation sidecars, no labels read."""
    wanted, actual = import_rows(fresh_rows, old_rows)
    created, lineage, opaque, pointers = set(), [], [], {}
    result = copy.deepcopy(fresh_rows)
    for fresh in result:
        if fresh['split'] not in ('train', 'validation'):
            continue
        previous = actual[fresh['id']]
        for kind, keys in (('render', ('input', 'video', 'mesh')), ('bank', ('bank', 'cnos_bank'))):
            original = previous[kind]
            if original['status'] != 'available':
                fresh[kind] = copy.deepcopy(original)
                continue
            imported = {'status': 'available'}
            for key in keys:
                source = _contained(original[key]['path'], old_root)
                p.verify_pin(original[key])
                target = _contained(Path(new_root) / source.relative_to(old_root), new_root)
                check_storage()
                _link(source, target, created)
                imported[key] = {**original[key], 'path': target.as_posix()}
                p.verify_pin(imported[key])
                lineage.append(dict(id=fresh['id'], kind=kind, key=key,
                    original=original[key], imported=imported[key], transport='same_inode_hardlink'))
            fresh[kind] = imported
        if fresh['bank']['status'] == 'available':
            obj = fresh['object_id']
            pointer = _contained(Path(old_root) / f'cnos-{int(obj):06d}.json', old_root)
            source_value = p.read_json(pointer)
            if source_value != {'bank': previous['bank']['cnos_bank']}:
                raise ValueError('original CNOS pointer differs from imported bank')
            target = _contained(Path(new_root) / pointer.relative_to(old_root), new_root)
            new_value = {'bank': fresh['bank']['cnos_bank']}
            if obj in pointers:
                if pointers[obj]['value'] != new_value:
                    raise ValueError('inconsistent per-object CNOS pointer')
            else:
                check_storage()
                p.atomic_json(target, new_value)
                pointers[obj] = dict(original=p.pin(pointer), imported=p.pin(target),
                    value=new_value, transport='canonical_pointer_rebased_to_imported_bank')
        if fresh['render']['status'] == 'available':
            source = _contained(Path(previous['render']['input']['path']).parent.parent / 'generation.json', old_root)
            target = _contained(Path(fresh['render']['input']['path']).parent.parent / 'generation.json', new_root)
            info = source.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= p.MAX_JSON:
                raise ValueError('bounded regular opaque generation sidecar required')
            check_storage()
            _link(source, target, created)  # Intentionally no open/read/hash/parse.
            opaque.append(dict(id=fresh['id'], source=source.as_posix(), target=target.as_posix(),
                device=info.st_dev, inode=info.st_ino, bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                transport='opaque_hardlink_not_opened', payload_authentication='deferred_until_label_phase'))
        if fresh['mask'] != {'status': 'pending'} or fresh['seed'] != {'status': 'pending'}:
            raise ValueError('fresh retry mask/seed states must remain pending')
    return dict(rows=result, lineage=lineage, opaque_generation=opaque,
                cnos_pointers=list(pointers.values()),
                test_payload_reads=0, generation_payload_reads=0)


def _seconds(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative elapsed charge required')
    return value


def _same_context(old_doc, doc):
    if ({k: v for k, v in old_doc.items() if k not in ('output_root', 'scratch_root')}
            != {k: v for k, v in doc.items() if k not in ('output_root', 'scratch_root')}
            or doc['limits']['storage_bytes'] != 32 * GIB):
        raise ValueError('retry may change only output/scratch roots under32GiB cap')


def _clean_receipt(context, name, receipt, *, failed=False):
    if (receipt.get('context_sha256') != context.sha256 or receipt.get('stage') != name
            or receipt.get('status') != ('failed' if failed else 'completed')
            or receipt.get('test_payload_reads') != 0 or receipt.get('training_ready') is not False
            or ((receipt.get('error') is None) == failed)):
        raise ValueError('original stage attempt identity/status/safety differs')
    _seconds(receipt['elapsed_seconds'])


def _prior(context):
    doc, spec, plan, rows = raw.frozen(context)
    root = Path(doc['output_root'])
    paths = sorted(path.name for path in root.glob('stage-*.json'))
    if paths != ['stage-banks.json', 'stage-masks.json', 'stage-render.json']:
        raise ValueError('only completed render/banks followed by failed masks can be imported')
    receipts = {name: p.read_json(root / ('stage-' + name + '.json'))
                for name in ('render', 'banks', 'masks')}
    for name, receipt in receipts.items():
        _clean_receipt(context, name, receipt, failed=name == 'masks')
    freeze = p.read_json(root / 'freeze.json')
    _seconds(freeze['elapsed_seconds'])
    return doc, spec, plan, rows, receipts, freeze


def prepare_retry(old_context, context, review_pin, *, sample_combined):
    """Fresh retry with authenticated import; never reruns successful neural work."""
    began = time.monotonic()
    approve(context, review_pin)
    old_doc, spec, plan, old_rows, receipts, freeze = _prior(old_context)
    doc = context.document()
    _same_context(old_doc, doc)
    old_root, root = Path(old_doc['output_root']), Path(doc['output_root'])
    scratch = Path(doc['scratch_root'])
    if root.exists() or scratch.exists():
        raise FileExistsError('fresh retry output and scratch roots required')
    import_rows(raw.physical_rows(spec, plan), old_rows)
    # The trusted operator has already admitted both roots into its live union.
    check = lambda: proposal_reserve(context, 0, sample_combined=sample_combined,
                                    output_future=p.TEMP_ENVELOPE)
    check()
    raw.freeze(old_root / 'recipe.json', p.sha256(old_root / 'recipe.json'), context)
    scratch.mkdir(parents=True, exist_ok=False)
    fresh = raw.frozen(context)[3]
    imported = link_inputs(old_rows, fresh, old_root=old_root, new_root=root, check_storage=check)
    raw._save_rows(context, imported.pop('rows'))
    source = dict(schema='visualizeit_raw_mask_resource_source_v1', mode='authenticated_retry',
        context_sha256=context.sha256, elapsed_seconds=0., status='completed',
        adapter=p.pin(Path(__file__)), independent_review=review_pin,
        reservation='full_retained_union_and_actual_destination_filesystem',
        original_sources_changed=False, test_payload_reads=0, training_ready=False)
    raw.write_json(context, root / MARKER, source)
    for name in ('render', 'banks'):
        raw.write_json(context, root / ('stage-' + name + '.json'),
            {**receipts[name], 'context_sha256': context.sha256,
             'mode': 'authenticated_import', 'neural_execution': False,
             'original_context_sha256': old_context.sha256,
             'original_receipt': p.pin(old_root / ('stage-' + name + '.json'))})
    transport = time.monotonic() - began
    prior = freeze['elapsed_seconds'] + receipts['masks']['elapsed_seconds']
    marker = dict(schema='visualizeit_raw_authenticated_recovery_import_v1',
        context_sha256=context.sha256, status='completed',
        old_context=p.pin(old_root / 'context.json'),
        old_physical_index=p.pin(old_root / 'physical-index.json'),
        old_freeze=p.pin(old_root / 'freeze.json'),
        old_failed_masks=p.pin(old_root / 'stage-masks.json'),
        source_marker=p.pin(root / MARKER),
        charged_prior_seconds=prior, transport_elapsed_seconds=transport,
        elapsed_seconds=prior + transport, original_stage_seconds_charged_in_imported_receipts=True,
        prior_attempts_preserved=True, training_ready=False, **imported)
    raw.write_json(context, root / IMPORT_MARKER, marker)
    validate_extension(context, review_pin)
    return dict(context=p.pin(root / 'context.json'), import_receipt=p.pin(root / IMPORT_MARKER),
                source_marker=p.pin(root / MARKER))


def bind_fresh(context, review_pin):
    """Attribute destination-aware masks in a shard with no prior mask attempt."""
    approve(context, review_pin)
    doc, _, _, rows = raw.frozen(context)
    root = Path(doc['output_root'])
    if (doc['limits']['storage_bytes'] != 32 * GIB or (root / IMPORT_MARKER).exists()
            or (root / 'stage-masks.json').exists()
            or any(row['mask'] != {'status': 'pending'} or row['seed'] != {'status': 'pending'}
                   for row in raw._active(rows))):
        raise ValueError('fresh unattempted32GiB mask stage required')
    stages = {}
    for name in ('render', 'banks'):
        receipt = p.read_json(root / ('stage-' + name + '.json'))
        _clean_receipt(context, name, receipt)
        stages[name] = p.pin(root / ('stage-' + name + '.json'))
    value = dict(schema='visualizeit_raw_mask_resource_source_v1', mode='fresh',
        context_sha256=context.sha256, elapsed_seconds=0., status='completed',
        adapter=p.pin(Path(__file__)), independent_review=review_pin,
        reservation='full_retained_union_and_actual_destination_filesystem',
        original_sources_changed=False, test_payload_reads=0, training_ready=False,
        original_completed_stages=stages)
    raw.write_json(context, root / MARKER, value)
    return p.pin(root / MARKER)


def validate_extension(context, review_pin):
    """Operator admission gate before masks/capture/seal and child publication."""
    approve(context, review_pin)
    doc, spec, plan, rows = raw.frozen(context)  # complete sealed metadata barrier
    if doc['limits']['storage_bytes'] != 32 * GIB:
        raise ValueError('recovery requires frozen32GiB shared limit')
    root = Path(doc['output_root'])
    source = p.read_json(root / MARKER)
    if (source.get('schema') != 'visualizeit_raw_mask_resource_source_v1'
            or source.get('context_sha256') != context.sha256
            or source.get('adapter') != p.pin(Path(__file__))
            or source.get('independent_review') != review_pin
            or source.get('elapsed_seconds') != 0. or source.get('status') != 'completed'):
        raise ValueError('explicit mask-source attribution changed')
    if source.get('mode') == 'fresh':
        if (root / IMPORT_MARKER).exists():
            raise ValueError('fresh source cannot hide recovery import')
        stages = {}
        for name in ('render', 'banks'):
            receipt = p.read_json(root / ('stage-' + name + '.json'))
            _clean_receipt(context, name, receipt)
            stages[name] = p.pin(root / ('stage-' + name + '.json'))
        if source.get('original_completed_stages') != stages:
            raise ValueError('original fresh-stage receipt changed')
        return source
    if source.get('mode') != 'authenticated_retry':
        raise ValueError('explicit fresh/retry mask-source mode required')
    marker = p.read_json(root / IMPORT_MARKER)
    if (marker.get('schema') != 'visualizeit_raw_authenticated_recovery_import_v1'
            or marker.get('context_sha256') != context.sha256 or marker.get('status') != 'completed'
            or marker.get('source_marker') != p.pin(root / MARKER)
            or marker.get('test_payload_reads') != 0 or marker.get('generation_payload_reads') != 0):
        raise ValueError('authenticated recovery lineage changed')
    old_pin = marker['old_context']
    old_context = raw.read_context(p.verify_pin(old_pin), old_pin['sha256'])
    old_doc, old_spec, old_plan, old_rows, receipts, freeze = _prior(old_context)
    _same_context(old_doc, doc)
    old_root = Path(old_doc['output_root'])
    if spec != old_spec or plan != old_plan:
        raise ValueError('recovery recipe/partition differs')
    import_rows(raw.physical_rows(spec, plan), old_rows)
    for key, name in (('old_physical_index', 'physical-index.json'),
                      ('old_freeze', 'freeze.json'), ('old_failed_masks', 'stage-masks.json')):
        if marker[key] != p.pin(old_root / name):
            raise ValueError('original failed-attempt evidence changed')
    prior = freeze['elapsed_seconds'] + receipts['masks']['elapsed_seconds']
    if (marker['charged_prior_seconds'] != prior
            or marker['elapsed_seconds'] != prior + _seconds(marker['transport_elapsed_seconds'])):
        raise ValueError('prior freeze/failure and transport must be charged exactly once')
    for name in ('render', 'banks'):
        expected = {**receipts[name], 'context_sha256': context.sha256,
            'mode': 'authenticated_import', 'neural_execution': False,
            'original_context_sha256': old_context.sha256,
            'original_receipt': p.pin(old_root / ('stage-' + name + '.json'))}
        if p.read_json(root / ('stage-' + name + '.json')) != expected:
            raise ValueError('imported successful-stage provenance/elapsed changed')
    actual, previous = p.indexed(rows, 'retry rows'), p.indexed(old_rows, 'original rows')
    expected_lineage, expected_pointers, expected_opaque = [], {}, []
    checked = set()
    for row in rows:
        if row['split'] not in ('train', 'validation'):
            continue
        old = previous[row['id']]
        for kind, keys in (('render', ('input', 'video', 'mesh')), ('bank', ('bank', 'cnos_bank'))):
            if old[kind]['status'] != 'available':
                if row[kind] != old[kind]:
                    raise ValueError('unavailable import state changed')
                continue
            if row[kind].get('status') != 'available':
                raise ValueError('available import state lost')
            for key in keys:
                pin = old[kind][key]
                source_path = _contained(pin['path'], old_root)
                target = _contained(root / source_path.relative_to(old_root), root)
                wanted_pin = {**pin, 'path': target.as_posix()}
                if row[kind].get(key) != wanted_pin or not os.path.samefile(source_path, target):
                    raise ValueError('import path/byte/inode lineage differs')
                for descriptor in (pin, wanted_pin):
                    if descriptor['path'] not in checked:
                        p.verify_pin(descriptor); checked.add(descriptor['path'])
                expected_lineage.append(dict(id=row['id'], kind=kind, key=key,
                    original=pin, imported=wanted_pin, transport='same_inode_hardlink'))
        if row['bank']['status'] == 'available':
            obj = row['object_id']
            pointer = old_root / f'cnos-{int(obj):06d}.json'
            target = root / pointer.name
            value = {'bank': row['bank']['cnos_bank']}
            if p.read_json(pointer) != {'bank': old['bank']['cnos_bank']} or p.read_json(target) != value:
                raise ValueError('rebased CNOS pointer changed')
            expected_pointers.setdefault(obj, dict(original=p.pin(pointer), imported=p.pin(target),
                value=value, transport='canonical_pointer_rebased_to_imported_bank'))
        if row['render']['status'] == 'available':
            path = _contained(Path(old['render']['input']['path']).parent.parent / 'generation.json', old_root)
            target = _contained(root / path.relative_to(old_root), root)
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or not os.path.samefile(path, target):
                raise ValueError('opaque generation link changed')
            expected_opaque.append(dict(id=row['id'], source=path.as_posix(), target=target.as_posix(),
                device=info.st_dev, inode=info.st_ino, bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                transport='opaque_hardlink_not_opened', payload_authentication='deferred_until_label_phase'))
    if (marker['lineage'] != expected_lineage or marker['cnos_pointers'] != list(expected_pointers.values())
            or marker['opaque_generation'] != expected_opaque):
        raise ValueError('complete import lineage differs')
    return marker


def masks(context, *, sample_combined):
    """Original mask producer mechanics with a destination-aware reserve only."""
    raw._require_stage(context, 'masks')
    from . import quality_adaptation_larger as larger
    from .quality_adaptation_larger_optimized import NeuralMaskBackend, run_mask_phases, PROPOSAL_BYTES_PER_VIEW
    doc, spec, plan, rows = raw.frozen(context)
    root = Path(doc['output_root'])
    torch = larger.configure_torch(doc['device'], deterministic=True)
    try:
        for obj in dict.fromkeys(r['object_id'] for r in raw._active(rows)):
            selected = [r for r in raw._active(rows) if r['object_id'] == obj]
            views = [r for r in selected if r['render']['status'] == r['bank']['status'] == 'available']
            for row in selected:
                if row not in views:
                    row['mask'] = dict(status='unavailable', failure_reason='render_or_bank_unavailable')
            raw._save_rows(context, rows)
            if not views:
                continue
            by_sequence = {v['sequence_id']: v for v in views}
            backend_doc = dict(assets=spec['assets'])
            backend = NeuralMaskBackend(root, backend_doc, obj, doc['device'], torch,
                Path(doc['model_cache']), pin_resolver=lambda value: raw.check_pin(value, under=root),
                input_reader=lambda view: raw.read_observed_input(context, by_sequence[view['sequence_id']]),
                mask_writer=lambda view, selected_mask, diagnostic: raw._write_mask(context,
                    by_sequence[view['sequence_id']], selected_mask, diagnostic))
            def persist(results, summary):
                for result in results:
                    row = by_sequence[result['view']['sequence_id']]
                    row['mask'] = dict(status=result['status'], failure_reason=result['failure_reason'], **result.get('output', {}))
                raw._save_rows(context, rows)
                raw.write_json(context, root / f'mask-phases-{int(obj):06d}.json', summary)
            scratch = Path(doc['scratch_root']) / ('proposals-' + obj)
            cap = len(views) * PROPOSAL_BYTES_PER_VIEW
            def reserve(amount):
                if type(amount) is not int or amount not in (0, cap):
                    raise ValueError('mask producer proposal cap differs')
                current = raw._usage([scratch])
                if current > cap:
                    raise raw.ResourceError('proposal tree exceeds frozen per-object cap')
                return proposal_reserve(context, max(0, cap - current),
                    sample_combined=sample_combined,
                    output_future=len(views) * 2 * 1024**2 + p.TEMP_ENVELOPE)
            run_mask_phases(views, scratch, backend, reserve=reserve, persist=persist)
    finally:
        try:
            larger.cleanup_device(doc['device'], torch)
        finally:
            for row in raw._active(rows):
                if row['mask']['status'] == 'pending':
                    row['mask'] = dict(status='unavailable', failure_reason='unrun_after_termination')
            raw._save_rows(context, rows)
