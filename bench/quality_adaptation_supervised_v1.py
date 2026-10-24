"""Explicit v1 supervision bridge over immutable v2 validators and producers.

New files may live alongside the frozen RAW source closure. The request retains
its original source_root/source_files/adapter; supervision_sources pins the exact
new execution code and the original no-follow scanner. No runtime monkeypatches.
"""
from __future__ import annotations
import sys
import json
import hashlib
from pathlib import Path
# Absolute-source-file CLI also works when the deployment root is separate.
if __package__ in (None, ''):
    if '--request' in sys.argv:
        _request_file = Path(sys.argv[sys.argv.index('--request') + 1])
        _request_bytes = _request_file.read_bytes()
        if ('--request-sha256' not in sys.argv or hashlib.sha256(_request_bytes).hexdigest()
                != sys.argv[sys.argv.index('--request-sha256') + 1]):
            raise ValueError('externally authenticated request hash differs before imports')
        _bootstrap = json.loads(_request_bytes)
        _source = _bootstrap.get('training', _bootstrap)['source_root']
        sys.path.insert(0, _source)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'bench'
from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_portable as raw
from scripts import kaggle_prepared_v2 as prepared
from scripts import kaggle_pilot_v2 as pilot_v2
from scripts import kaggle_supervision_v1 as supervision
from bench.quality_adaptation_portable_prepare import *
from scripts.kaggle_prepared_v2 import (output_layout, fresh_evidence,
    assess_admission, training_request, verify_resume, checkpoint_receipt)
import threading

VERSION = 'kaggle_feature_supervision_v1'
ORIGINAL_HELPERS = {
 'bench/quality_adaptation_portable_prepare.py': '6513d31edfcc0460fd86e0e783b1f51634b69203b05b2b7a1b294c5decfbe76e',
 'scripts/kaggle_prepared_v2.py': 'a993aba7acee237da4b0d7f82610c12e0873b1a70e795c9a6e8c6a3806915e35',
 'scripts/kaggle_pilot_v2.py': '28d071c38243809504e7f9884d0e2f839eb2226f117bcf42abbb4de5f4ae03e0',
}


def verify_supervision(request):
    sources = request.get('supervision_sources')
    actual = {'bridge': Path(__file__).absolute(),
              'supervisor': Path(supervision.__file__).absolute(),
              'scanner': Path(raw.__file__).absolute()}
    if not isinstance(sources, dict) or set(sources) != set(actual):
        raise ValueError('exact bridge/supervisor/scanner supervision_sources required')
    for name, path in actual.items():
        if p.verify_pin(sources[name]) != path:
            raise ValueError('supervision source is not the actually loaded file: ' + name)
    if sources['scanner']['sha256'] != supervision.SCANNER_SHA256:
        raise ValueError('fixed original no-follow scanner pin differs')
    limits = request['limits']
    for name, ceiling in (('max_seconds', 1200), ('max_rss_bytes', 8 * GIB),
                          ('max_output_bytes', 8 * GIB)):
        value = limits[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= ceiling:
            raise ValueError('versioned execution bound exceeded: ' + name)
    if limits['reserve_bytes'] < GIB or request['storage_limit_bytes'] > 18 * GIB:
        raise ValueError('versioned execution requires 1 GiB reserve and <=18 GiB combined storage')
    root = Path(request['source_root'])
    for name, expected in ORIGINAL_HELPERS.items():
        if p.sha256(root / name) != expected:
            raise ValueError('immutable orchestration helper changed: ' + name)
    for module, name in ((p, 'bench/quality_adaptation_portable_prepare.py'),
                         (prepared, 'scripts/kaggle_prepared_v2.py'),
                         (pilot_v2, 'scripts/kaggle_pilot_v2.py')):
        if Path(module.__file__).absolute() != root / name:
            raise ValueError('original helper imported from foreign source root')
    return dict(version=VERSION, sources=sources, validation_adapter=request['adapter'],
        supervisor='scripts.kaggle_supervision_v1.Facade', output_scanner='raw._usage',
        runtime_sampler='three complete original unique_tree_bytes attempts; FileNotFoundError only')


def source_runtime(request):
    verify_supervision(request)
    adapter, root18, observed = p.source_runtime(request)
    return supervision.Facade(adapter, raw), root18, observed


def verify_binding(request, features):
    provenance = verify_supervision(request)
    if p.read_json(Path(features) / 'extraction-binding.json').get('supervision') != provenance:
        raise ValueError('extraction supervision provenance differs from actual execution closure')
    return provenance


def verify_prepared(request, *, payloads=True):
    verify_binding(request, request['features_root'])
    return prepared.verify_prepared(request, payloads=payloads)


def full_scope_bound(request, joined, k, before, *, elapsed, steps):
    verify_binding(request['full_run'], request['full_run']['features_root'])
    return pilot_v2.full_scope_bound(request, joined, k, before, elapsed=elapsed, steps=steps)


def production_run(request, module, args, *, output_root, log, cuda=False, additional_bytes=None):
    """One unchanged producer per fresh process with the authenticated bootstrap."""
    cloud_guard(request)
    k, root18, _ = source_runtime(request)
    evidence = preparation_evidence(request)
    from . import quality_adaptation_larger as larger
    argv = [request['python'], '-E', '-s', '-B', '-m', module, *map(str, args)]
    env = k.clean_environment()
    env['CUDA_VISIBLE_DEVICES'] = evidence['gpu']['uuid']
    if cuda:
        bootstrap = larger._CUDA_BOOTSTRAP
        env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', VISUALIZEIT_CUDA_BOOTSTRAP_SHA256=hashlib.sha256(bootstrap.encode()).hexdigest(),
            VISUALIZEIT_CUDA_RUNTIME_RECEIPT=str(Path(output_root) / 'device-runtime.json'))
        argv = [request['python'], '-E', '-s', '-B', '-c', bootstrap, module, *map(str, args)]
    extra = request['limits']['max_output_bytes'] if additional_bytes is None else additional_bytes
    quota_check(request, k, extra=extra)
    result = k.supervise(argv, cwd=root18, output_root=Path(output_root), working=Path(request['working_root']),
        log_path=Path(log), limits=request['limits'], env=env)
    try:
        result['observed_quota_after'] = quota_check(request, k)
    except (Exception, raw.ResourceError) as error:
        if result['status'] == 'completed':
            raise
        result['quota_after_error'] = f'{type(error).__name__}: {error}'
    return result


def run_labels(request, request_path, expected):
    """The label renderer is a fresh process under the same RSS/quota watchdog."""
    cloud_guard(request)
    from scripts.kaggle_prepared_v2 import verify_wrapper_sources
    verify_wrapper_sources(request)
    k, _, _ = source_runtime(request)
    evidence = preparation_evidence(request)
    output = Path(request['features_root'])
    quota_check(request, k, extra=TEMP_ENVELOPE)
    env = k.clean_environment()
    env.update(CUDA_VISIBLE_DEVICES=evidence['gpu']['uuid'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
        VISUALIZEIT_V2_SUPERVISED_LABEL=expected)
    result = k.supervise([request['python'], '-E', '-s', '-B', str(Path(__file__).absolute()),
        'label-worker', '--request', str(request_path), '--request-sha256', expected],
        cwd=Path(request['source_root']), output_root=output, working=Path(request['working_root']),
        log_path=Path(request['execution_root']) / 'label-worker.log', limits=request['limits'], env=env)
    atomic_json(Path(request['execution_root']) / 'label-stage-observation.json', result)
    return result


def extract(request, raw_state, execution):
    verify_execution(raw_state, execution)
    features = Path(request['features_root'])
    if features.exists():
        raise FileExistsError('fresh global extraction directory required')
    manifest_pin = crop_manifest(raw_state, execution)
    active_count = sum(r['split'] in ('train', 'validation') for r in raw_state['captures'])
    envelope = active_count * PER_STATE_ENVELOPE + TEMP_ENVELOPE
    if envelope > min(FEATURE_TREE_CAP, request['limits'].get('max_output_bytes', FEATURE_TREE_CAP)):
        raise ValueError('global feature/label/temp envelope exceeds unchanged extractor ceiling')
    with preparation_lock(raw_state):
        result = production_run(request, 'bench.quality_adaptation_prepare', ['--manifest', manifest_pin['path'], '--output', features,
            '--device', 'cuda', '--max-seconds', int(request['limits']['max_seconds']), '--max-output-bytes', FEATURE_TREE_CAP,
            '--max-rows', active_count, '--cloud-execution'], output_root=features, log=Path(execution) / 'extract.log', cuda=True,
            additional_bytes=envelope)
    if result.get('status') != 'completed' or result.get('returncode') != 0:
        return result
    if features.is_dir() and (features / 'index.json').is_file():
        # Exact original receipts survive. Only the reviewed final index adds
        # captured-unavailable and sealed TEST accounting; dev stays separate.
        for name in ('recipe.json', 'index.json'):
            shutil.copyfile(features / name, features / ('original-' + name))
        original = read_json(features / 'index.json')
        actual = indexed(original['rows'], 'original extractor')
        required = {r['id'] for r in raw_state['captures'] if r['status'] == 'available' and r['split'] in ('train', 'validation')}
        if set(actual) != required:
            raise ValueError('unchanged extractor omitted requested accounting')
        final_rows = []
        for row in raw_state['captures']:
            if row['split'] == 'dev':
                continue
            final_rows.append(actual.get(row['id'], dict(id=row['id'], status='unavailable', failure_reason=row.get('failure_reason') or 'capture_unavailable')))
        final = {**original, 'rows': final_rows}
        # The ONLY permitted replacement is the final explicitly audited index.
        temporary = features / '.final-index.json'
        atomic_json(temporary, final)
        os.replace(temporary, features / 'index.json')
        atomic_json(features / 'extraction-binding.json', dict(schema='kaggle_extraction_binding_v2',
            original_recipe=pin(features / 'original-recipe.json'), original_index=pin(features / 'original-index.json'),
            final_recipe=pin(features / 'recipe.json'), final_index=pin(features / 'index.json'), crop_manifest=manifest_pin,
            execution_plan=pin(Path(execution) / 'execution-plan.json'), raw_seal=pin(raw_state['root'] / 'raw-seal.json'),
            unavailable_added=[r['id'] for r in final_rows if r['id'] not in actual], dev_rows=[r for r in raw_state['captures'] if r['split'] == 'dev'],
            execution_result=result, successful_original_rows_unchanged=True, supervision=verify_supervision(request)))
    return result


def labels(request, raw_state, execution):
    """Separate renderer supervision stage; no GT data enters input identities."""
    cloud_guard(request)
    features = Path(request['features_root'])
    # This statement MUST precede even reading the label plan descriptor bytes.
    prelabel = p.verify_prelabel(raw_state, execution, features)
    verify_binding(request, features)
    label_plan = metadata(request['label_plan'])
    if set(label_plan) != {'schema', 'split_manifest', 'generations'} or label_plan['schema'] != 'kaggle_label_plan_v2':
        raise ValueError('invalid separate label plan')
    if verify_pin(label_plan['split_manifest']).read_bytes() != (raw_state['root'] / 'split.json').read_bytes():
        raise ValueError('label split bytes differ')
    generations = {r['sequence_id']: r['generation'] for r in label_plan['generations']}
    active = {r['sequence_id'] for r in raw_state['physical'] if r['split'] in ('train', 'validation') and r['render']['status'] == 'available'}
    if len(generations) != len(label_plan['generations']) or set(generations) != active:
        raise ValueError('generation metadata inventory incomplete/foreign/sealed')
    for value in generations.values():
        descriptor_path(value)  # complete metadata join before first generation hash
    from .quality_adaptation_data import CROP_KEYS, load_npz
    from .quality_adaptation_refinement_states import GEOMETRY_KEYS
    from .quality_adaptation_prepare import project_supervision
    from .quality_adaptation_smoke import render_query_labels
    from .quality_adaptation_larger import _validate_generation
    from .quality_adaptation_runtime import atomic_stream
    from .quality_gotrack import TexturedRenderer
    from .quality_runner import read_input
    from .quality_ranker_data import read_split_manifest
    import numpy as np
    split_path = verify_pin(label_plan['split_manifest'])
    split = read_split_manifest(split_path)
    k, _, _ = source_runtime(request)
    extracted = indexed(read_json(features / 'index.json')['rows'], 'features')
    physical = {r['sequence_id']: r for r in raw_state['physical']}
    inventory, outcomes = [], []
    started = time.monotonic()
    with preparation_lock(raw_state):
        from .quality_adaptation_larger import configure_torch, cleanup_device
        torch = configure_torch('cuda', deterministic=True)
        try:
            for ordinal, source in enumerate(raw_state['captures']):
                if source['split'] == 'dev':
                    continue
                row = dict(id=source['id'], object_id=source['object_id'], sequence_id=source['sequence_id'], capture_group=source['capture_group'],
                    split='val' if source['split'] == 'validation' else source['split'], query_rgb_sha256=source.get('query_rgb_sha256'), status='unavailable',
                    failure_reason=source.get('failure_reason') or extracted[source['id']].get('failure_reason') or 'label_not_processed')
                if row['query_rgb_sha256'] is None:
                    row['unavailable_provenance'] = source.get('unavailable_provenance', 'frozen state; no retained crop')
                inventory.append(row)
                renderer = None
                if source['split'] not in ('train', 'validation') or source['status'] != 'available' or extracted[source['id']]['status'] != 'available':
                    continue
                try:
                    if time.monotonic() - started >= request['limits']['max_seconds']:
                        raise TimeoutError('label deadline exhausted')
                    quota_check(request, k, extra=TEMP_ENVELOPE)
                    frame = physical[source['sequence_id']]
                    render = frame['render']
                    bundle = verify_pin(render['input']).parent
                    manifest = read_input(bundle)
                    # Original validator expects legacy val spelling only.
                    physical_projection = dict(frame, split=row['split'], **{key: render[key] for key in ('input', 'video', 'mesh')})
                    renderer = TexturedRenderer(bundle / manifest['asset'], manifest['object_id'], unlit=True, disable_multisampling=True, coordinate_mode='integer_centers_v1')
                    pose = _validate_generation(metadata(generations[source['sequence_id']]), physical_projection, manifest, split, sha256(split_path), renderer.vertices_m)
                    crops = load_npz(verify_pin(source['crops'], raw_state['root'] / 'raw'), CROP_KEYS)
                    geometry = load_npz(verify_pin(source['capture'], raw_state['root'] / 'raw'), GEOMETRY_KEYS)
                    query_pose, depth, support = render_query_labels(renderer, object_id=manifest['object_id'], crop_k=crops['query_K'], crop_from_orig=geometry['crop_from_orig'], native_pose=pose)
                    target = project_supervision(template_xyz_m=crops['template_xyz_m'], template_mask=crops['template_mask'], query_K=crops['query_K'],
                        query_T_cam_from_object=query_pose, observed_depth_m=depth, observed_object_mask=support, depth_tolerance_m=.002)
                    name = f'labels-{ordinal:06d}.npz'
                    with atomic_stream(features, features / name, max_file_bytes=16 * MIB, max_output_bytes=FEATURE_TREE_CAP) as stream:
                        np.savez(stream, **target)
                    row.pop('failure_reason', None)
                    row.update(status='available', labels={**pin(features / name), 'path': name})
                except Exception as exc:
                    row['failure_reason'] = f'label_failure: {type(exc).__name__}: {exc}'[:1024]
                finally:
                    if renderer is not None:
                        renderer.close()
                    outcomes.append(dict(id=source['id'], label_status=row['status'], failure_reason=row.get('failure_reason')))
        finally:
            cleanup_device('cuda', torch)
            # All requested IDs survive even an interrupted label stage.
            seen = {r['id'] for r in inventory}
            for source in raw_state['captures']:
                if source['split'] != 'dev' and source['id'] not in seen:
                    row = dict(id=source['id'], object_id=source['object_id'], sequence_id=source['sequence_id'], capture_group=source['capture_group'],
                        split='val' if source['split'] == 'validation' else source['split'], query_rgb_sha256=source.get('query_rgb_sha256'),
                        status='unavailable', failure_reason='label_stage_interrupted')
                    if row['query_rgb_sha256'] is None:
                        row['unavailable_provenance'] = 'frozen state; no crop materialized'
                    inventory.append(row)
            atomic_json(features / 'labels.json', dict(schema='correspondence_label_inventory_v1', label_origin='synthetic_renderer_training_only', rows=inventory))
    result = production_run(request, 'bench.quality_adaptation_manifest', ['--extraction', features, '--labels', features / 'labels.json', '--output', features / 'manifest.json'],
        output_root=features, log=Path(execution) / 'assemble.log', additional_bytes=TEMP_ENVELOPE)
    atomic_json(Path(execution) / 'assemble-observation.json', result)
    if result.get('status') != 'completed' or result.get('returncode') != 0:
        raise ValueError('global assembler did not complete; all labels/failures and supervision observation retained')
    receipt = dict(schema='kaggle_label_receipt_v2', prelabel_seal=pin(Path(execution) / 'prelabel-seal.json'), label_plan=request['label_plan'],
        inventory=pin(features / 'labels.json'), dataset=pin(features / 'manifest.json'), outcomes=outcomes,
        label_artifacts=[dict(id=r['id'], **r['labels']) for r in inventory if r['status'] == 'available'],
        labels_entered_inference=False, test_payload_reads=0, independent_quality_claim=False, elapsed_seconds=time.monotonic() - started,
        parity_rows=len(prelabel['parity']), assemble_result=result)
    atomic_json(features / 'label-receipt.json', receipt)
    return receipt


def prepare_stage(stage, request):
    verify_supervision(request)
    from scripts.kaggle_prepared_v2 import verify_wrapper_sources
    verify_wrapper_sources(request)
    from .quality_adaptation_portable import read_context, frozen
    context = read_context(Path(request['context']['path']), request['context']['sha256'])
    # Exposure checks precede even the raw registry's first active asset hash.
    _, recipe, _, _ = frozen(context)
    bounded_recipe(recipe, metadata(request['exposure_audit']))
    state = raw_metadata(context, request['raw_seal'])
    execution = Path(request['execution_root'])
    if stage == 'identity':
        # Exposure evidence is metadata; verify it before selecting/opening any
        # source asset. Raw stages independently retain their frozen inventory.
        bounded_recipe(state['recipe'], metadata(request['exposure_audit']))
        return execution_identity(state, execution)
    if stage == 'extract':
        return extract(request, state, execution)
    if stage == 'seal-features':
        verify_binding(request, request['features_root'])
        return seal_features(state, execution, Path(request['features_root']))
    if stage == 'label-worker':
        from .quality_adaptation_portable import verify_neural_environment
        verify_neural_environment(context)
        return labels(request, state, execution)
    raise ValueError('unknown preparation stage')


def execute(stage, request, *, stop_after_steps=None, pilot=False):
    if stage not in ('preflight', 'train', 'resume', 'export'):
        raise ValueError('prepared consumer supports trainer stages only')
    joined = verify_prepared(request)
    output = output_layout(request)
    p.cloud_guard(request)
    k, root18, runtime = source_runtime(request)
    limits = k.validate_limits(request['limits'])
    if limits['max_rss_bytes'] > 8 * GIB or limits['max_output_bytes'] > 2 * GIB:
        raise ValueError('v2 run RSS/output exceeds 8 GiB/2 GiB bound')
    quota = p.quota_check(request, k, extra=limits['max_output_bytes'])
    evidence = fresh_evidence(request, joined['identity_sha256'], limits, pilot=pilot)
    admission = assess_admission(joined, request['profile'], resource_admitted=True)
    if not admission['admitted']:
        raise ValueError('actual v2 admission failed: ' + ','.join(admission['blockers']))
    cfg = p.metadata(request['config'])
    if cfg.get('max_seconds', 900) > 900:
        raise ValueError('unchanged trainer invocation must be <=900 seconds')
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
            admission=admission, runtime=runtime, evidence=request['execution_evidence'], supervision=verify_supervision(request), resource_before=quota,
            resource_after=p.quota_check(request, k), result=result, quality_claim=False, outputs_preserved=True,
            invocation=dict(stage=stage, started_at=started_at, adapter_pid=os.getpid(),
                trainer_source_sha256=k.file_sha256(root18 / 'bench/quality_adaptation_train.py'), **result))
        p.atomic_json(output / ('observation-' + uuid.uuid4().hex + '.json'), observation)
    return observation


def pilot_execute(request):
    steps_cap = pilot_v2.validate_caps(request)
    training = request['training']
    started = time.monotonic()
    deadline = started + training['limits']['max_seconds']
    joined = verify_prepared(training)
    p.cloud_guard(training)
    k, root18, runtime = source_runtime(training)
    old = p.load_source(request['pilot_helper'], '_kaggle_v1_pilot_for_v2', expected=p.PILOT_SHA256)
    output = prepared.output_layout(training)
    if output.exists():
        raise FileExistsError('pilot must use fresh standalone output; no checkpoint promotion')
    prepared.fresh_evidence(training, joined['identity_sha256'], training['limits'], pilot=True)
    admission = prepared.assess_admission(joined, training['profile'], resource_admitted=True)
    if not admission['admitted']:
        raise ValueError('prepared pilot fails actual crop/feature/joint availability admission')
    quota_before = p.quota_check(training, k, extra=training['limits']['max_output_bytes'])
    output.mkdir(parents=True)
    p.atomic_json(output / 'v2-run-identity.json', joined['identity'])
    legacy_shape = dict(training={**prepared.training_request(training), 'working_root': training['working_root']}, limits=training['limits'],
        pilot_output_root=str(output), stop_after_steps=steps_cap)
    record = dict(schema='kaggle_pilot_measurement_v2', prepared_identity_sha256=joined['identity_sha256'],
        measurement_complete=False, full_training_admitted=False, quality_claim=False, test_arrays_opened=False,
        verified_optimizer_steps=0, status='failed', outputs_preserved=True, enforced_bounds=training['limits'],
        source_helper=request['pilot_helper'], supervision=verify_supervision(training), runtime=runtime, profile=training['profile'], quota_before=quota_before)
    samples = [quota_before['resident_bytes']]
    errors = []
    stop = threading.Event()
    links = k.runtime_link_policy(p.metadata(training['runtime']), Path(training['working_root']))
    roots = tuple(Path(x) for x in training['quota_roots'])
    def sample():
        while not stop.is_set():
            try:
                samples[0] = max(samples[0], k.unique_tree_bytes(roots, approved_links=links))
            except (Exception, raw.ResourceError) as exc:
                errors.append(str(exc))
                return
            stop.wait(.1)
    observer = threading.Thread(target=sample, daemon=True)
    try:
        # The immutable inspection helper owns its own non-nestable timer.
        # V2-owned segments arm that same absolute deadline between inspections.
        before = old.inspect_training(k, legacy_shape, root18, output / 'train', log_path=output / 'inspect-before.json', deadline=deadline)
        with old.execution_deadline(deadline):
            remaining = max(0, deadline - time.monotonic())
            bounded = {**training, 'limits': {**training['limits'], 'max_seconds': remaining}}
            observer.start()
            try:
                observation = execute('train', bounded, stop_after_steps=steps_cap, pilot=True)
            finally:
                stop.set()
                observer.join(timeout=5)
            if errors or observer.is_alive():
                raise ValueError('storage sampler failed: ' + str(errors))
            result = observation['result']
            status = k.read_json(output / 'train/status.json')
            state = dict(identity=dict(roots=dict(output_root=str(output)), paths=dict(train_dir=str(output / 'train'))), stages=[observation['invocation']])
            progress = k.trainer_progress(output, state, observe_invocation=True)
        after = old.inspect_training(k, legacy_shape, root18, output / 'train', log_path=output / 'inspect-after.json', deadline=deadline)
        with old.execution_deadline(deadline):
            checkpoint = p.pin(output / 'train/last.pt')
            if after.get('checkpoint_sha256') != checkpoint['sha256'] or checkpoint['bytes'] > before['config']['max_checkpoint_bytes']:
                raise ValueError('checkpoint changed across original weights_only inspection or exceeds cap')
            steps = old.reconcile(k, legacy_shape, result, before, after, status, progress)
            after_resources = p.quota_check(training, k)
            samples[0] = max(samples[0], after_resources['resident_bytes'])
            record.update(status='measurement_complete', measurement_complete=True, trainer_exit_code=3,
                verified_optimizer_steps=steps, checkpoint=checkpoint, checkpoint_metadata=after,
                optimizer_progress=progress, trainer_observation=observation,
                observed=dict(trainer_wall_seconds=result['elapsed_seconds'], whole_pilot_seconds=time.monotonic() - started,
                    peak_sampled_group_rss_bytes=result['peak_group_rss_bytes'], combined_storage_high_water_bytes=samples[0],
                    peak_is_exact=False, sampling='100ms snapshots; native transient peak may be larger', resources_after=after_resources),
                forecast=dict(linear_training_seconds=result['elapsed_seconds'] / steps * before['config']['max_steps'], full_scope_representativeness_established=False))
            try:
                record['full_scope_bound'] = full_scope_bound(request, joined, k, before, elapsed=result['elapsed_seconds'], steps=steps)
                record['full_training_admitted'] = True
            except old.PilotDeadline:
                raise
            except (ValueError, KeyError, OSError) as exc:
                record['admission_records_withheld_reason'] = str(exc)
    except (Exception, KeyboardInterrupt, raw.ResourceError) as exc:
        record.update(status='failed', measurement_complete=False, full_training_admitted=False,
            reason=f'{type(exc).__name__}: {exc}')
    finally:
        stop.set()
        cleanup_errors = []
        if observer.ident is not None:
            try:
                observer.join(timeout=5)
                if observer.is_alive():
                    cleanup_errors.append('storage sampler did not stop within cleanup allowance')
            except (Exception, raw.ResourceError) as exc:
                cleanup_errors.append(f'{type(exc).__name__}: {exc}')
        if cleanup_errors:
            record.update(status='failed', measurement_complete=False, full_training_admitted=False,
                cleanup_errors=cleanup_errors)
        record['whole_pilot_observed_seconds'] = time.monotonic() - started
        p.atomic_json(output / 'pilot-measurement.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('recipe-preflight', 'identity', 'extract', 'seal-features', 'labels', 'label-worker', 'inspect', 'preflight', 'train', 'resume', 'export', 'monitor', 'pilot'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    request = p.load_request(args.request, args.request_sha256)
    if args.stage == 'pilot':
        result = pilot_execute(request)
    elif args.stage == 'labels':
        result = run_labels(request, args.request, args.request_sha256)
    elif args.stage in ('preflight', 'train', 'resume', 'export'):
        result = execute(args.stage, request)
    elif args.stage == 'inspect':
        joined = verify_prepared(request, payloads=False)
        result = assess_admission(joined, request['profile'], resource_admitted=False)
    elif args.stage == 'monitor':
        verify_binding(request, request['features_root'])
        result = prepared.monitor(request)
    elif args.stage == 'recipe-preflight':
        verify_supervision(request)
        result = p.bounded_recipe(p.metadata(request['recipe']), p.metadata(request['exposure_audit']))
    else:
        if args.stage == 'label-worker' and os.environ.get('VISUALIZEIT_V2_SUPERVISED_LABEL') != args.request_sha256:
            raise ValueError('label worker requires authenticated supervised invocation')
        result = prepare_stage(args.stage, request)
    print(json.dumps(result, allow_nan=False))
    status = result.get('result', result).get('status')
    return 3 if status in ('incomplete', 'measurement_complete') else 2 if status == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())

