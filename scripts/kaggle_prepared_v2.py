"""Prepared-only v2 join/admission and unchanged ROOT18 trainer control.

The request's external SHA256 is the authority for this new source closure and
all metadata pins. No v1 metadata/admission function accepts v2 artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import quality_adaptation_portable_prepare as p

GIB = p.GIB
SOURCE_NAMES = ('bench/quality_adaptation_portable_prepare.py', 'scripts/kaggle_prepared_v2.py', 'scripts/kaggle_pilot_v2.py')
METADATA_ROLES = ('context', 'raw_seal', 'execution_plan', 'identity_bridge', 'prelabel_seal', 'extraction_binding',
                  'original_recipe', 'original_index', 'feature_index', 'label_receipt', 'labels', 'dataset', 'runtime', 'config', 'exposure_audit')


def verify_wrapper_sources(request):
    root = p.no_links(Path(request['source_root']))
    files = request.get('source_files')
    if not isinstance(files, dict) or set(files) != set(SOURCE_NAMES):
        raise ValueError('exact v2 wrapper source closure required')
    with p.source_migration_scope(request.get('source_migration')):
        for name, value in files.items():
            p.verify_prepared_source(value, root, name)
    return root


def metadata_join(request):
    """Metadata first, including LAST sealed state, before any payload hash."""
    if request.get('schema') != 'kaggle_prepared_request_v2':
        raise ValueError('prepared v2 request required')
    verify_wrapper_sources(request)
    docs = {key: p.metadata(request[key]) for key in METADATA_ROLES}
    context, raw = p.prepared_raw_state(request['context'], request['raw_seal'],
        raw_continuation=request.get('raw_continuation'), source_migration=request.get('source_migration'),
        exposure_audit=request['exposure_audit'], verify_payloads=False)
    exposure = p.bounded_recipe(raw['recipe'], docs['exposure_audit'])
    if Path(request['source_root']) != Path(raw['context']['source_root']):
        raise ValueError('v2 source root differs from authenticated raw closure')
    execution = p.no_links(Path(request['execution_root']))
    features = p.no_links(Path(request['features_root']))
    if (Path(request['execution_plan']['path']) != execution / 'execution-plan.json'
            or Path(request['identity_bridge']['path']) != execution / 'identity-bridge.json'
            or Path(request['prelabel_seal']['path']) != execution / 'prelabel-seal.json'
            or Path(request['dataset']['path']) != features / 'manifest.json'):
        raise ValueError('prepared artifact layout differs')
    plan = p.verify_execution(raw, execution)
    if plan != docs['execution_plan']:
        raise ValueError('execution plan request differs')
    seal = p.check_seal(docs['prelabel_seal'], 'kaggle_prelabel_seal_v2')
    binding = docs['extraction_binding']
    for name, path in (('original_recipe', 'original-recipe.json'), ('original_index', 'original-index.json'),
                       ('feature_index', 'index.json'), ('labels', 'labels.json'), ('label_receipt', 'label-receipt.json'),
                       ('extraction_binding', 'extraction-binding.json')):
        if Path(request[name]['path']) != features / path:
            raise ValueError('prepared metadata path mismatch: ' + name)
    for key, role in (('raw_seal', 'raw_seal'), ('execution_plan', 'execution_plan'), ('identity_bridge', 'identity_bridge'),
                      ('binding', 'extraction_binding'), ('index', 'feature_index')):
        if seal.get(key) != request[role]:
            raise ValueError('prelabel metadata differs: ' + key)
    for key, role in (('original_recipe', 'original_recipe'), ('original_index', 'original_index'), ('final_index', 'feature_index')):
        if binding.get(key) != request[role]:
            raise ValueError('original extraction binding differs: ' + key)
    receipt = docs['label_receipt']
    if (receipt.get('schema') != 'kaggle_label_receipt_v2' or receipt.get('prelabel_seal') != request['prelabel_seal']
            or receipt.get('inventory') != request['labels'] or receipt.get('dataset') != request['dataset']
            or receipt.get('labels_entered_inference') is not False or receipt.get('test_payload_reads') != 0
            or receipt.get('independent_quality_claim') is not False):
        raise ValueError('label receipt/prelabel/inventory/dataset binding changed')
    captures = p.indexed(raw['captures'], 'capture')
    states = p.indexed(plan['global_manifest']['records'], 'execution inventory')
    labels = p.indexed(docs['labels']['rows'], 'labels')
    dataset = p.indexed(docs['dataset']['rows'], 'dataset')
    features_by_id = p.indexed(docs['feature_index']['rows'], 'features')
    expected = {key for key, row in states.items() if row['split'] != 'dev'}
    if (set(labels) != expected or set(dataset) != expected or set(features_by_id) != expected
            or docs['labels'].get('schema') != 'correspondence_label_inventory_v1'
            or docs['labels'].get('label_origin') != 'synthetic_renderer_training_only'
            or docs['dataset'].get('schema') != 'correspondence_head_dataset_v1'):
        raise ValueError('complete frozen nondev dataset/label/feature accounting required')
    aliases, crop_set, feature_set, joint_set = [], set(), set(), set()
    expected_artifacts = []
    for ident, state in states.items():
        capture = captures[ident]
        if state['split'] == 'dev':
            continue
        label, final, feature = labels[ident], dataset[ident], features_by_id[ident]
        split = 'val' if state['split'] == 'validation' else state['split']
        if split == 'test' and (feature.get('status') != 'unavailable'
                or set(feature) != {'id', 'status', 'failure_reason'}):
            raise ValueError('sealed TEST feature inventory must remain unavailable metadata only')
        base = dict(id=ident, split=split, object_id=capture['object_id'], sequence_id=state['sequence_id'], capture_group=state['scene_id'])
        if any(label.get(k) != v or final.get(k) != v for k, v in base.items()):
            raise ValueError('dataset split/state/object identity differs')
        if label.get('query_rgb_sha256') != capture.get('query_rgb_sha256'):
            raise ValueError('label RGB identity differs from actual capture')
        if {k: v for k, v in final.items() if k != 'features'} != label:
            raise ValueError('assembler row differs from label inventory')
        for row in (capture, feature, label):
            if row.get('status') not in ('available', 'unavailable') or (row['status'] == 'unavailable' and not row.get('failure_reason')):
                raise ValueError('availability failure accounting missing')
        if capture['status'] == 'available':
            crop_set.add(ident)
        if feature['status'] == 'available':
            if capture['status'] != 'available' or feature.get('crops') != capture.get('crops') or feature.get('query_rgb_sha256') != capture.get('query_rgb_sha256'):
                raise ValueError('feature availability/input binding differs')
            feature_set.add(ident)
        paths, contents = [], []
        if final['status'] == 'available':
            if split == 'test' or ident not in feature_set or final.get('features') != feature.get('features'):
                raise ValueError('joint availability cannot exceed feature/crop availability or open TEST')
            joint_set.add(ident)
            for key in ('features', 'labels'):
                paths.append(p.descriptor_path(final[key], features))
                # Equal labels alone (e.g. identity zero flow) are valid across
                # independent objects, as in the unchanged dataset validator.
                if key == 'features':
                    contents.append(final[key]['sha256'])
            expected_artifacts.append(dict(id=ident, **label['labels']))
        elif any(k in final for k in ('features', 'labels')):
            raise ValueError('unavailable row must contain no payload descriptors')
        aliases.append((split, final.get('query_rgb_sha256'), paths))
        # Hash equality detects copied aliases, including distinct paths/inodes.
        for value in contents:
            aliases.append((split, value, []))
    p.reject_aliases(aliases)
    if receipt.get('label_artifacts') != expected_artifacts:
        raise ValueError('label artifact receipt differs from actual inventory')
    if set(r['id'] for r in seal['parity']) != feature_set:
        raise ValueError('all available features require numerical prelabel parity')
    counts = {}
    for split in ('train', 'validation', 'test', 'dev'):
        wanted = {key for key, row in states.items() if row['split'] == split}
        counts[split] = dict(requested=len(wanted), crops=len(wanted & crop_set), features=len(wanted & feature_set), jointly_usable=len(wanted & joint_set))
    identity = dict(schema='kaggle_prepared_identity_v2', source_files=request['source_files'], adapter=request['adapter'],
        source_install=request['source_install'], python=request['python'], runtime=request['runtime'], config=request['config'],
        artifacts={k: request[k] for k in METADATA_ROLES if k not in ('runtime', 'config')},
        dataset_canonical_sha256=p.canonical(docs['dataset']), execution_plan_sha256=plan['plan_sha256'],
        preparation_profile=request['profile'])
    for name in ('raw_continuation', 'source_migration'):
        if name in request:
            identity[name] = request[name]
    return dict(raw=raw, context=context, docs=docs, counts=counts, exposure_disclosure=exposure['exposure_disclosure'], available_crops=sorted(crop_set), available_features=sorted(feature_set),
        jointly_usable=sorted(joint_set), identity=identity, identity_sha256=p.canonical(identity), dataset_sha256=p.canonical(docs['dataset']),
        failures=[dict(id=key, crop=captures[key]['status'], feature=features_by_id[key]['status'], label=labels[key]['status'],
            reason=labels[key].get('failure_reason')) for key in sorted(expected) if key not in joint_set])


def verify_prepared(request, *, payloads=True):
    joined = metadata_join(request)
    if payloads:
        # No payload is hashed before the COMPLETE metadata join above.
        p.prepared_raw_state(request['context'], request['raw_seal'],
            raw_continuation=request.get('raw_continuation'), source_migration=request.get('source_migration'),
            exposure_audit=request['exposure_audit'])
        p.verify_prelabel(joined['raw'], Path(request['execution_root']), Path(request['features_root']))
        from bench.quality_adaptation_data import load_dataset
        ds = load_dataset(Path(request['dataset']['path']), inspect_arrays=True)
        if ds.identity != joined['dataset_sha256']:
            raise ValueError('dataset canonical identity changed')
    return joined


def assess_admission(joined, profile, *, resource_admitted):
    """Ordinary observed failures stay counted; structural errors already raise."""
    if not isinstance(profile, dict) or set(profile) != {'kind', 'train_states_requested'}:
        raise ValueError('explicit bounded_smoke or target profile required')
    requested = profile['train_states_requested']
    if type(requested) is not int or requested < 1:
        raise ValueError('positive TRAIN request required')
    if profile['kind'] == 'target':
        if requested not in (1000, 2000, 4000):
            raise ValueError('target must be 1000/2000/4000 actual usable TRAIN states')
    elif profile['kind'] == 'bounded_smoke':
        # A first real integration smoke may be tiny. Its actual count is always
        # shown; a larger-than-prior experiment explicitly requests >=17.
        pass
    else:
        raise ValueError('unknown admission profile')
    actual = joined['counts']['train']['jointly_usable']
    blockers = []
    if actual < requested:
        blockers.append('insufficient_actual_jointly_usable_train')
    if joined['counts']['validation']['jointly_usable'] < 1:
        blockers.append('no_actual_jointly_usable_validation')
    if not resource_admitted:
        blockers.append('resource_admission_missing')
    return dict(schema='adaptation_scale_admission_v2', admitted=not blockers, blockers=blockers,
        counts=joined['counts'], availability_failures=joined['failures'], profile=profile,
        identity_sha256=joined['identity_sha256'], targets=[dict(train_states_requested=n, actual_usable_train=actual, met=actual >= n) for n in (1000, 2000, 4000)],
        exposure_disclosure=joined.get('exposure_disclosure', dict(independent_validation_claim=False, independent_test_claim=False)), quality_claim=False)


def fresh_evidence(request, identity, limits, *, pilot=False):
    """Current observed provider quota and selected physical GPU, never guesses."""
    p.cloud_guard(request)
    evidence = p.metadata(request['execution_evidence'])
    now = datetime.now(timezone.utc).timestamp()
    observed = datetime.fromisoformat(evidence['observed_at'].replace('Z', '+00:00')).timestamp()
    expires = datetime.fromisoformat(evidence['expires_at'].replace('Z', '+00:00')).timestamp()
    if (evidence.get('schema') != 'kaggle_execution_evidence_v2' or evidence.get('provider') != 'kaggle'
            or evidence.get('prepared_identity_sha256') != identity or not evidence.get('session_id')
            or not now - 3600 <= observed <= now < expires <= observed + 3600):
        raise ValueError('fresh actual Kaggle identity/session evidence required')
    quota = evidence['quota']
    if (quota.get('paid') is not False or quota.get('gpu_enabled') is not True
            or quota.get('storage_scope') != request['quota_roots'] or quota.get('storage_limit_bytes') != request['storage_limit_bytes']
            or min(quota.get('gpu_remaining_seconds', 0), quota.get('session_remaining_seconds', 0)) < limits['max_seconds']):
        raise ValueError('observed free provider quota cannot cover this bounded invocation')
    selected = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], check=True, capture_output=True,
        text=True, timeout=15).stdout.splitlines()
    if selected.count(f"{evidence['gpu']['uuid']}, {evidence['gpu']['name']}") != 1:
        raise ValueError('selected actual GPU identity changed')
    if not pilot:
        measurement = p.metadata(request['pilot_measurement'])
        if (measurement.get('schema') != 'kaggle_pilot_measurement_v2' or measurement.get('prepared_identity_sha256') != identity
                or measurement.get('verified_optimizer_steps', 0) < 1 or measurement.get('full_training_admitted') is not True
                or measurement.get('full_scope_bound', {}).get('dataset_sha256') != p.canonical(p.metadata(request['dataset']))
                or measurement.get('full_scope_bound', {}).get('config_file_sha256') != request['config']['sha256']):
            raise ValueError('same immutable dataset/config real pilot and conservative full-scope bound required')
    return evidence


def training_request(request):
    """Only map argument names; never call v1 metadata/admission functions."""
    return dict(python=request['python'], paths=request['paths'], limits=request['limits'],
        config=request['config'], dataset=request['dataset'])


def output_layout(request):
    working = p.no_links(Path(request['working_root']))
    output = p.no_links(Path(request['output_root']))
    if output == working or not output.is_relative_to(working):
        raise ValueError('dedicated training output below working required')
    for key in ('source_install', 'features_root', 'execution_root'):
        other = p.no_links(Path(request[key]))
        if output == other or output.is_relative_to(other) or other.is_relative_to(output):
            raise ValueError('training output overlaps preparation/source artifacts')
    source_root = Path(request['source_root'])
    # Local source trees may contain ignored run fixtures under .cache. Protect
    # executable trees and fixed model caches rather than forbidding all data
    # below the repository. The immutable ROOT18 installation stays disjoint.
    if source_root.is_relative_to(output) or any(output.is_relative_to(source_root / part) for part in ('bench', 'scripts', '.cache/model-quality')):
        raise ValueError('training output overlaps executing source/model cache')
    paths = request['paths']
    if (Path(paths['extraction_dir']) != Path(request['features_root']) or Path(paths['train_dir']) != output / 'train'
            or Path(paths['manifest_output']) != Path(request['dataset']['path']) or Path(paths['export_output']) != output / 'train/export.pt'):
        raise ValueError('immutable trainer/export layout differs')
    return output


def verify_resume(output, identity):
    receipt = p.read_json(output / 'v2-run-identity.json')
    if receipt != identity:
        raise ValueError('resume/export changed dataset/config/source/runtime/global identity')
    checkpoint = p.read_json(output / 'v2-checkpoint-receipt.json')
    if checkpoint.get('prepared_identity_sha256') != p.canonical(identity):
        raise ValueError('checkpoint receipt changed prepared identity')
    expected = {path.name for path in (output / 'train').iterdir() if path.is_file()}
    actual = {Path(value['path']).name for value in checkpoint['files']}
    if expected != actual or 'last.pt' not in actual:
        raise ValueError('checkpoint/run metadata inventory changed')
    for value in checkpoint['files']:
        path = p.verify_pin(value)
        if path.parent != output / 'train':
            raise ValueError('checkpoint receipt escapes immutable run')


def checkpoint_receipt(output, identity):
    train = output / 'train'
    files = []
    if train.is_dir():
        for path in sorted(train.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise ValueError('unexpected trainer artifact')
            files.append(p.pin(path))
    receipt = dict(schema='kaggle_checkpoint_receipt_v2', prepared_identity_sha256=p.canonical(identity), files=files)
    destination = output / 'v2-checkpoint-receipt.json'
    temporary = output / ('.checkpoint-' + uuid.uuid4().hex + '.json')
    p.atomic_json(temporary, receipt)
    os.replace(temporary, destination)
    return receipt


def monitor(request):
    joined = verify_prepared(request, payloads=False)
    output = output_layout(request)
    verify_resume(output, joined['identity'])
    k = p.load_source(request['adapter'], '_kaggle_v1_monitor_v2', expected=p.ADAPTER_SHA256)
    observations = [p.read_json(path) for path in output.glob('observation-*.json')]
    attempts = [value['invocation'] for value in observations if value.get('stage') in ('train', 'resume')
                and value.get('prepared_identity_sha256') == joined['identity_sha256']]
    attempts.sort(key=lambda value: value['started_at'])
    state = dict(identity=dict(roots=dict(output_root=str(output)), paths=dict(train_dir=str(output / 'train'))), stages=attempts)
    return k.trainer_progress(output, state)


def execute(stage, request, *, stop_after_steps=None, pilot=False):
    if stage not in ('preflight', 'train', 'resume', 'export'):
        raise ValueError('prepared consumer supports trainer stages only')
    joined = verify_prepared(request)
    output = output_layout(request)
    p.cloud_guard(request)
    k, root18, runtime = p.source_runtime(request)
    limits = k.validate_limits(request['limits'])
    if limits['max_rss_bytes'] > 8 * GIB or limits['max_output_bytes'] > 2 * GIB:
        raise ValueError('v2 run RSS/output exceeds 8 GiB/2 GiB bound')
    quota = p.quota_check(request, k, extra=limits['max_output_bytes'])
    evidence = fresh_evidence(request, joined['identity_sha256'], limits, pilot=pilot)
    admission = assess_admission(joined, request['profile'], resource_admitted=True)
    if not admission['admitted']:
        raise ValueError('actual v2 admission failed: ' + ','.join(admission['blockers']))
    cfg = p.metadata(request['config'])
    if cfg.get('device') != 'cuda' or cfg.get('max_output_bytes', 2 * GIB) > limits['max_output_bytes']:
        raise ValueError('explicit CUDA trainer config must fit enforced output bound')
    if stage in ('resume', 'export'):
        verify_resume(output, joined['identity'])
    elif stage == 'train':
        if output.exists() and not (pilot and not (output / 'train').exists() and p.read_json(output / 'v2-run-identity.json') == joined['identity']):
            raise FileExistsError('new train run requires fresh output; use resume for an existing run')
        if not output.exists():
            output.mkdir(parents=True)
            p.atomic_json(output / 'v2-run-identity.json', joined['identity'])
    else:
        # A preflight must not consume the fresh future training directory.
        output = Path(request['execution_root']) / ('trainer-preflight-' + uuid.uuid4().hex)
        output.mkdir()
    argv = k.command(stage, training_request(request))
    if stop_after_steps is not None:
        if not pilot or stage != 'train' or type(stop_after_steps) is not int or not 1 <= stop_after_steps <= 4:
            raise ValueError('only explicit pilot mode permits 1..4 update interruption')
        argv += ['--stop-after-steps', str(stop_after_steps)]
    if pilot and (limits['max_seconds'] > 900 or stop_after_steps is None):
        raise ValueError('pilot requires <=900 seconds and explicit real-update cap')
    with k.run_lock(output):
        started_at = datetime.now(timezone.utc).isoformat()
        env = k.clean_environment()
        env['CUDA_VISIBLE_DEVICES'] = evidence['gpu']['uuid']
        result = k.supervise(argv, cwd=root18, output_root=output, working=Path(request['working_root']),
            log_path=output / (stage + '-' + uuid.uuid4().hex + '.log'), limits=limits, env=env)
        if stage in ('train', 'resume', 'export'):
            checkpoint_receipt(output, joined['identity'])
        observation = dict(schema='kaggle_execution_observation_v2', stage=stage, prepared_identity_sha256=joined['identity_sha256'],
            admission=admission, runtime=runtime, evidence=request['execution_evidence'], resource_before=quota,
            resource_after=p.quota_check(request, k), result=result, quality_claim=False, outputs_preserved=True,
            invocation=dict(stage=stage, started_at=started_at, adapter_pid=os.getpid(),
                trainer_source_sha256=k.file_sha256(root18 / 'bench/quality_adaptation_train.py'), **result))
        p.atomic_json(output / ('observation-' + uuid.uuid4().hex + '.json'), observation)
    return observation


def request_document(*, preparation, runtime, config, source_install, python, input_root, working_root,
                     output_root, profile, limits, quota_roots, storage_limit_bytes, execution_evidence,
                     pilot_measurement=None, neural_inputs=(), raw_continuation=None, source_migration=None):
    """Model-free builder from actual artifacts, never asserted success receipts."""
    root = Path(preparation['source_root'])
    execution, features = Path(preparation['execution_root']), Path(preparation['features_root'])
    raw_root = Path(preparation['raw_root'])
    continuation = None if raw_continuation is None else p.metadata(raw_continuation)
    context_pin = p.pin(raw_root / 'context.json') if continuation is None else continuation['successor_context']
    raw_pin = p.pin(raw_root / 'raw-seal.json') if continuation is None else continuation['view_seal']
    value = dict(schema='kaggle_prepared_request_v2', source_root=str(root), source_files={name: p.pin(root / name) for name in SOURCE_NAMES},
        adapter=p.pin(root / 'scripts/kaggle_adaptation.py'), source_install=str(source_install), python=str(python), input_root=str(input_root),
        working_root=str(working_root), output_root=str(output_root), execution_root=str(execution), features_root=str(features),
        runtime=p.pin(runtime), config=p.pin(config), profile=profile, limits=limits, quota_roots=list(map(str, quota_roots)), neural_inputs=list(neural_inputs),
        storage_limit_bytes=storage_limit_bytes, execution_evidence=p.pin(execution_evidence), exposure_audit=preparation['exposure_audit'],
        context=context_pin, raw_seal=raw_pin, execution_plan=p.pin(execution / 'execution-plan.json'),
        identity_bridge=p.pin(execution / 'identity-bridge.json'), prelabel_seal=p.pin(execution / 'prelabel-seal.json'),
        extraction_binding=p.pin(features / 'extraction-binding.json'), original_recipe=p.pin(features / 'original-recipe.json'),
        original_index=p.pin(features / 'original-index.json'), feature_index=p.pin(features / 'index.json'), label_receipt=p.pin(features / 'label-receipt.json'),
        labels=p.pin(features / 'labels.json'), dataset=p.pin(features / 'manifest.json'),
        paths=dict(extraction_dir=str(features), manifest_output=str(features / 'manifest.json'), train_dir=str(Path(output_root) / 'train'), export_output=str(Path(output_root) / 'train/export.pt')))
    if pilot_measurement is not None:
        value['pilot_measurement'] = p.pin(pilot_measurement)
    if raw_continuation is not None:
        value['raw_continuation'] = raw_continuation
    if source_migration is not None:
        value['source_migration'] = source_migration
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('inspect', 'preflight', 'train', 'resume', 'export', 'monitor'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    request = p.load_request(args.request, args.request_sha256)
    if args.stage == 'inspect':
        joined = verify_prepared(request, payloads=False)
        result = assess_admission(joined, request['profile'], resource_admitted=False)
    elif args.stage == 'monitor':
        result = monitor(request)
    else:
        result = execute(args.stage, request)
    print(json.dumps(result, allow_nan=False))
    status = result.get('result', {}).get('status')
    return 3 if status == 'incomplete' else 2 if status == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
