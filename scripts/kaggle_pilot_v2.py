"""Measure 1..4 genuine unchanged-trainer updates on one prepared v2 identity.

Pilot mode admits acquisition of the first timing observation without inventing
prior timing. A clean trainer exit 3 is retained as incomplete, never model quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import quality_adaptation_portable_prepare as p
from scripts import kaggle_prepared_v2 as prepared


def validate_caps(request):
    if request.get('schema') != 'kaggle_pilot_request_v2':
        raise ValueError('explicit v2 pilot request required')
    steps = request.get('stop_after_steps')
    if type(steps) is not int or not 1 <= steps <= 4:
        raise ValueError('pilot optimizer update cap must be 1..4')
    limits = request['training']['limits']
    for key, ceiling in (('max_seconds', 900), ('max_rss_bytes', 8 * p.GIB), ('max_output_bytes', 2 * p.GIB)):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= ceiling:
            raise ValueError('invalid pilot cap: ' + key)
    if limits.get('reserve_bytes', 0) < p.GIB:
        raise ValueError('pilot requires >=1 GiB reserve')
    return steps


def full_scope_bound(request, joined, k, before, *, elapsed, steps):
    """Bounded smoke uses enforced caps; a prefix is no full-cost forecast."""
    full = request.get('full_run')
    if not isinstance(full, dict):
        raise ValueError('complete eventual full-run request missing')
    full_join = prepared.verify_prepared(full, payloads=False)
    if full_join['identity_sha256'] != joined['identity_sha256']:
        raise ValueError('eventual full run changed immutable prepared identity')
    if full['profile']['kind'] != 'bounded_smoke':
        raise ValueError('prefix pilot cannot establish completion fit for large target runs')
    limits = k.validate_limits(full['limits'])
    cfg = before['config']
    if cfg['max_seconds'] > limits['max_seconds'] or cfg['max_total_seconds'] > 86400:
        raise ValueError('unchanged trainer wall allowance exceeds enforced invocation envelope')
    envelope = cfg['max_output_bytes'] + limits['max_log_bytes'] + 2 * p.MAX_JSON
    if cfg['max_output_bytes'] < 3 * cfg['max_checkpoint_bytes'] + 8 * p.MIB or envelope > limits['max_output_bytes']:
        raise ValueError('full checkpoint/atomic/log/metadata envelope does not fit')
    resources = p.quota_check(full, k, extra=limits['max_output_bytes'])
    return dict(dataset_sha256=before['dataset_sha256'], config_file_sha256=full['config']['sha256'],
        source_lock_sha256=before['source_lock_sha256'], planned_optimizer_steps=cfg['max_steps'],
        enforced_invocation_wall_seconds=limits['max_seconds'], unchanged_cumulative_trainer_seconds=cfg['max_total_seconds'],
        enforced_output_allowance_bytes=limits['max_output_bytes'], resources=resources,
        basis='enforced complete global-dataset wall/RSS/output bounds; bounded execution admission only',
        completion_forecast_available=False, full_epoch_validation_time_measured=False,
        prefix_linear_forecast_seconds=elapsed / steps * cfg['max_steps'], prefix_forecast_representative=False,
        is_measured_peak=False)


def execute(request):
    steps_cap = validate_caps(request)
    training = request['training']
    started = time.monotonic()
    deadline = started + training['limits']['max_seconds']
    joined = prepared.verify_prepared(training)
    p.cloud_guard(training)
    k, root18, runtime = p.source_runtime(training)
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
        source_helper=request['pilot_helper'], runtime=runtime, profile=training['profile'], quota_before=quota_before)
    samples = [quota_before['resident_bytes']]
    errors = []
    stop = threading.Event()
    links = k.runtime_link_policy(p.metadata(training['runtime']), Path(training['working_root']))
    roots = tuple(Path(x) for x in training['quota_roots'])
    def sample():
        while not stop.is_set():
            try:
                samples[0] = max(samples[0], k.unique_tree_bytes(roots, approved_links=links))
            except Exception as exc:
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
                observation = prepared.execute('train', bounded, stop_after_steps=steps_cap, pilot=True)
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
    except (Exception, KeyboardInterrupt) as exc:
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
            except Exception as exc:
                cleanup_errors.append(f'{type(exc).__name__}: {exc}')
        if cleanup_errors:
            record.update(status='failed', measurement_complete=False, full_training_admitted=False,
                cleanup_errors=cleanup_errors)
        record['whole_pilot_observed_seconds'] = time.monotonic() - started
        p.atomic_json(output / 'pilot-measurement.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    result = execute(p.load_request(args.request, args.request_sha256))
    print(json.dumps(result, allow_nan=False))
    return 3 if result['measurement_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
