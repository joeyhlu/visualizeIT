"""Explicit MODEL-FREE SOFTWARE FIXTURES; no capture/model-quality evidence.

These small analytical arrays and controlled per-physical callbacks exercise
metadata, chronological transaction and exception contracts. No ML dependency,
checkpoint, asset payload, neural inference or download is used.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import copy
import hashlib
from pathlib import Path
import tempfile
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_cached_capture_v1 as cached
from bench import quality_adaptation_portable_prepare as preparation
from bench.quality_adaptation_refinement_states import state_rows

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.cache/cloud-prepare/test-tmp'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw.encoded(value))
    return raw.pin(path)


def detached(document):
    data = raw.encoded(document)
    return raw.PortableContext(data, hashlib.sha256(data).hexdigest())


def materialize(output, physical, states):
    """Controlled fake producer: analytical identity calibration, no model."""
    physical['seed'] = dict(status='available', poses=[np.eye(4).tolist()],
        selection='SOFTWARE_FIXTURE_IDENTITY_SEED')
    for ordinal, row in enumerate(states):
        if not row['id'].startswith(physical['id'] + ':iteration'):
            continue
        descriptors = {}
        for kind, prefix in (('crops', 'crop'), ('capture', 'capture')):
            path = output / f'{prefix}-{ordinal:02d}.npz'
            # The identity matrix is an explicit software fixture, not camera
            # geometry derived from any image or reference annotation.
            np.savez(path, SOFTWARE_FIXTURE=np.eye(3))
            descriptors[kind] = dict(raw.pin(path), path=path.name)
        row.pop('failure_reason', None)
        row.pop('unavailable_provenance', None)
        row.update(status='available', pnp_state='SOFTWARE_FIXTURE_TERMINAL_SOLVER_OUTCOME', query_rgb_sha256=hashlib.sha256(
            ('SOFTWARE_FIXTURE_RGB_' + row['id']).encode()).hexdigest(), **descriptors)


class Fixture:
    def __init__(self, base, *, iterations=(0,)):
        self.base = Path(base)
        parity = write(self.base / 'parity.json', dict(MODEL_FREE_SOFTWARE_FIXTURE=True))
        self.root = self.base / 'predecessor'
        doc = raw.context_document(output_root=self.root, scratch_root=self.base / 'scratch',
            parity=dict(reference=parity, candidate=parity), device='cpu')
        self.successor = detached(doc)
        spec = dict(schema='kaggle_raw_recipe_v1', iterations=list(iterations), seed=42,
            targets=[1000, 2000, 4000], assets=[dict(object_id=str(n),
                physical_object_id='SOFTWARE_FIXTURE_' + str(n), aliases=[],
                mesh=dict(path=(self.base / f'MUST_NOT_OPEN-{n}.glb').as_posix(),
                    bytes=100, sha256=hashlib.sha256(str(n).encode()).hexdigest())) for n in (101, 102, 103)])
        recipe = write(self.base / 'recipe.json', spec)
        raw.freeze(recipe['path'], recipe['sha256'], self.successor)
        _, _, _, self.physicals = raw.frozen(self.successor)
        # Exercise real source migration: the original producer is archived and
        # its digest stays a historical pin after successor source changes.
        archive = ROOT / '.cache/capture-resume-v1/frozen-originals/bench/quality_adaptation_portable.py'
        self.archive = raw.pin(archive)
        old_doc = copy.deepcopy(doc)
        old_doc['source_files']['bench/quality_adaptation_portable.py'] = self.archive['sha256']
        self.context = detached(old_doc)
        (self.root / 'context.json').write_bytes(self.context.raw)
        freeze = raw.read_json(self.root / 'freeze.json')
        freeze.update(context_sha256=self.context.sha256, elapsed_seconds=2.)
        write(self.root / 'freeze.json', freeze)
        self.output = self.root / 'raw'; self.output.mkdir()
        self.states = state_rows(dict(rows=self.physicals))
        active = raw._active(self.physicals)
        self.completed, self.failed, self.pending = active[0], active[1], active[2:]
        materialize(self.output, self.completed, self.states)
        self.failed['seed'] = dict(status='unavailable', failure_reason='SOFTWARE_FIXTURE_OBSERVED_FAILURE')
        for row in self.states:
            if row['id'].startswith(self.failed['id'] + ':iteration'):
                row['failure_reason'] = 'SOFTWARE_FIXTURE_OBSERVED_FAILURE'
        for name in ('render', 'banks', 'masks'):
            write(self.root / ('stage-' + name + '.json'), dict(stage=name,
                context_sha256=self.context.sha256, elapsed_seconds=3., status='completed',
                test_payload_reads=0, training_ready=False))
        self.units = [dict(id=p['id'], outcome=('sealed' if p['split'] not in ('train', 'validation')
            else 'completed' if p is self.completed else 'failed' if p is self.failed else 'unattempted'))
            for p in self.physicals]
        self.terminal_path = self.base / 'terminal.json'
        self.terminal = dict(schema='visualizeit_raw_capture_terminal_resume_v1',
            context_sha256=self.context.sha256, metadata={}, source_archive={
                'bench/quality_adaptation_portable.py': self.archive},
            job=dict(pid=5611, returncode=1, process_start_token='SOFTWARE_FIXTURE_PROCESS_START',
                process_start_token_at_poll='SOFTWARE_FIXTURE_PROCESS_START',
                terminal_observed_unix_s=1000., absence_observed_unix_s=1001.,
                root_and_descendants_absent=True, gpu_compute_pids=[], gpu_used_bytes=0),
            units=self.units, capture_elapsed_seconds=5., prior_history=None)
        self.flush()

    def flush(self):
        write(self.root / 'physical-index.json', dict(rows=self.physicals))
        write(self.root / 'input-plan.json', raw.capture_plan(self.context, self.physicals))
        write(self.output / 'capture-index.json', dict(rows=self.states, training_ready=False))
        pins = {p.name: raw.pin(p) for p in self.root.glob('*.json')}
        pins['raw/capture-index.json'] = raw.pin(self.output / 'capture-index.json')
        pins['raw/capture-receipt.json'] = (raw.pin(self.output / 'capture-receipt.json')
            if (self.output / 'capture-receipt.json').exists() else None)
        pins.setdefault('stage-capture.json', None)
        self.terminal['metadata'] = pins
        self.terminal_pin = write(self.terminal_path, self.terminal)

    def allowance(self, **changes):
        value = dict(schema='visualizeit_raw_capture_continuation_allowance_v1',
            operation='capture_continuation', predecessor_context_sha256=self.context.sha256,
            successor_context_sha256=self.successor.sha256, terminal=self.terminal_pin,
            additional_allowance=True, max_seconds=60., attempt_id='software-fixture')
        value.update(changes)
        return write(self.base / 'allowance.json', value)

    def authenticate(self):
        return raw.authenticate_capture_continuation(self.context, self.successor, self.terminal_pin)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        TEMP.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(raw, 'resource_check', return_value={}).start()
        self.f = Fixture(temporary.name)

    @contextmanager
    def fake_guard(self, context, successor, output, seconds, *, cloud_execution):
        self.assertGreater(seconds, 0)
        with raw._successor_sources(context, successor, output):
            yield

    def run_fake(self, *, fail_at=None, cleanup_failure=False):
        calls = []; cleaned = []
        @contextmanager
        def backend(context, data, digest, inputs, states, output):
            try:
                def execute(physical):
                    calls.append(physical['id'])
                    if physical['id'] == fail_at:
                        raise raw.ResourceError('SOFTWARE_FIXTURE_INTERRUPTION')
                    materialize(output, physical, states)
                yield execute
            finally:
                cleaned.append(True)
                if cleanup_failure:
                    raise RuntimeError('SOFTWARE_FIXTURE_CLEANUP_FAILURE')
        with patch.object(raw, '_capture_continuation_guard', self.fake_guard), \
                patch.object(raw, 'preflight_inputs', return_value={}) as preflight, \
                patch.object(raw, '_capture_backend', backend):
            try:
                result = raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin,
                    self.f.allowance(), cloud_execution=True)
            finally:
                self.calls, self.cleaned = calls, cleaned
        self.assertEqual(preflight.call_count, 1)
        return result

    def test_authentication_reuses_bytes_and_preserves_observed_failures(self):
        before = {p: p.read_bytes() for p in self.f.output.iterdir()}
        result = self.f.authenticate()
        self.assertEqual(result['unattempted_physical_ids'], [p['id'] for p in self.f.pending])
        self.assertIn(self.f.failed['id'], result['preserved_physical_ids'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.f.output.iterdir()})
        # Independent arithmetic: 2 freeze + 3 stages * 3 + 5 capture = 16.
        self.assertEqual(result['cumulative_elapsed_seconds'], 16.)
        self.assertNotEqual(result['predecessor_context_sha256'], result['successor_context_sha256'])
        with self.assertRaisesRegex(ValueError, 'executing source changed'):
            raw.verify_sources(self.f.context)

    def test_continuation_only_executes_unattempted_and_retains_original_files(self):
        original = {p: p.read_bytes() for p in self.f.root.rglob('*') if p.is_file()}
        result = self.run_fake()
        self.assertEqual(self.calls, [p['id'] for p in self.f.pending])
        self.assertEqual(self.cleaned, [True])
        self.assertEqual(result['status'], 'completed')
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertTrue(result['artifact_roots'][self.f.completed['id']].endswith('/raw'))
        self.assertIn('/capture-continuations/software-fixture/', result['artifact_roots'][self.f.pending[0]['id']])
        self.assertEqual(result['cumulative_elapsed_seconds'], 16. + result['elapsed_seconds'])
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['legacy_raw_seal'])

    def test_changed_attempt_id_cannot_replay_a_completed_continuation(self):
        self.run_fake()
        next_document = raw.read_json(self.f.base / 'allowance.json')
        next_document['attempt_id'] = 'second-reviewed-attempt'
        next_allowance = write(self.f.base / 'second-allowance.json', next_document)
        with patch.object(raw, '_capture_backend') as backend:
            with self.assertRaisesRegex(FileExistsError, 'prior continuation exists'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin,
                    next_allowance, cloud_execution=True)
        backend.assert_not_called()
        self.assertFalse((self.f.root / 'capture-continuations/second-reviewed-attempt').exists())

    def test_interrupted_attempt_without_receipt_blocks_another_attempt(self):
        interrupted = self.f.root / 'capture-continuations/interrupted-before-receipt'
        interrupted.mkdir(parents=True)
        allowance = self.f.allowance(attempt_id='second-reviewed-attempt')
        with patch.object(raw, '_capture_backend') as backend:
            with self.assertRaisesRegex(FileExistsError, 'prior continuation exists'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin,
                    allowance, cloud_execution=True)
        backend.assert_not_called()
        self.assertTrue(interrupted.is_dir())

    def test_predecessor_attempt_fence_is_rechecked_under_stage_lock(self):
        @contextmanager
        def raced_guard(context, successor, output, seconds, *, cloud_execution):
            (self.f.root / 'capture-continuations/racing-attempt').mkdir(parents=True)
            with self.fake_guard(context, successor, output, seconds, cloud_execution=cloud_execution):
                yield
        with patch.object(raw, '_capture_continuation_guard', raced_guard), \
                patch.object(raw, '_capture_backend') as backend:
            with self.assertRaisesRegex(FileExistsError, 'prior continuation exists'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin,
                    self.f.allowance(), cloud_execution=True)
        backend.assert_not_called()
        self.assertFalse((self.f.root / 'capture-continuations/software-fixture').exists())

    @unittest.skipIf(os.name == 'nt', 'Windows sandbox denies physical hardlinks; mandatory Linux CPU integration before cloud use')
    def test_real_preparation_identity_and_crop_manifest_consume_explicit_join(self):
        self.run_fake()
        receipt = raw.pin(self.f.root / 'capture-continuations/software-fixture/receipt.json')
        kwargs = dict(adapter_pin=raw.pin(Path(cached.__file__)))
        with patch.object(raw, 'preflight_inputs', return_value={}):
            state = cached.continuation_metadata(self.f.context, self.f.successor, self.f.terminal_pin,
                receipt, materialize=True, **kwargs)
            reread = cached.continuation_metadata(self.f.context, self.f.successor, self.f.terminal_pin,
                receipt, **kwargs)
        self.assertEqual(state, reread)
        self.assertEqual(state['seal']['schema'], 'visualizeit_raw_capture_continuation_seal_v1')
        self.assertFalse(state['seal']['legacy_raw_seal'])
        for row in state['captures']:
            if row['status'] == 'available':
                physical_id = row['id'].rsplit(':iteration', 1)[0]
                origin = Path(state['seal']['artifact_roots'][physical_id])
                for key in ('crops', 'capture'):
                    self.assertTrue(os.path.samefile(origin / row[key]['path'], state['root'] / 'raw' / row[key]['path']))
        destination = self.f.base / 'genuine-existing-preparation-identity'
        execution = preparation.execution_identity(state, destination)
        self.assertEqual(preparation.verify_execution(state, destination), execution)
        crop_pin = preparation.crop_manifest(state, destination)
        crops = raw.read_json(crop_pin['path'])
        self.assertEqual(len(crops['rows']), len(raw._active(state['physical'])) - 1)
        self.assertEqual((destination / 'split.json').read_bytes(), (self.f.root / 'split.json').read_bytes())
        # The new seal is deliberately incompatible with legacy clean stages.
        with self.assertRaisesRegex(ValueError, 'changed kaggle_raw_capture_seal_v1'):
            preparation.check_seal(state['seal'], 'kaggle_raw_capture_seal_v1')

    def test_preparation_join_rejects_mutated_successor_index_before_view_creation(self):
        self.run_fake()
        attempt = self.f.root / 'capture-continuations/software-fixture'
        receipt = raw.pin(attempt / 'receipt.json')
        index = raw.read_json(attempt / 'capture-index.json')
        index['rows'].reverse()
        write(attempt / 'capture-index.json', index)
        with self.assertRaisesRegex(ValueError, 'altered pinned artifact'):
            cached.continuation_metadata(self.f.context, self.f.successor, self.f.terminal_pin,
                receipt, adapter_pin=raw.pin(Path(cached.__file__)), materialize=True)
        self.assertFalse((attempt / 'preparation-view').exists())

    def test_existing_preparation_identity_contract_with_independent_software_fixture(self):
        # This standalone fixture checks the existing consumer contract only;
        # it does not claim to execute the Linux hardlink join above.
        seal = preparation.sealed(dict(schema='SOFTWARE_FIXTURE_CONTINUATION_SEAL', training_ready=False))
        write(self.f.root / 'raw-seal.json', seal)
        state = dict(root=self.f.root, context=self.f.successor.document(),
            recipe=raw.read_json(self.f.root / 'recipe.json'),
            plan=raw.read_json(self.f.root / 'scale-plan.json'), physical=self.f.physicals,
            captures=self.f.states, seal=seal, closing_stage=self.f.terminal_pin)
        destination = self.f.base / 'standalone-consumer-contract'
        execution = preparation.execution_identity(state, destination)
        self.assertEqual(preparation.verify_execution(state, destination), execution)
        self.assertEqual(len(raw.read_json(preparation.crop_manifest(state, destination)['path'])['rows']), 1)

    def test_owned_original_popen_witness_needs_exact_request_launch_and_observation_chain(self):
        context_pin = raw.pin(self.f.root / 'context.json')
        request = write(self.f.base / 'original-request.json', dict(
            schema='visualizeit_expanded_cached_capture_stage_request_v1', stage='capture', shard_id='group1',
            python='/SOFTWARE_FIXTURE/python', context=context_pin))
        operator_path = self.f.base / 'original-operator.py'
        operator_path.write_bytes(b'# SOFTWARE FIXTURE: NEVER EXECUTED\n')
        operator = raw.pin(operator_path)
        launch = write(self.f.base / 'original-launch.json', dict(pid=5611, stage='capture',
            request=request, training_executed=False))
        observation = write(self.f.base / 'original-observation.json', dict(
            schema='visualizeit_expanded_cached_capture_stage_observation_v1', stage='capture', shard_id='group1',
            context=context_pin, status='incomplete', reason='deadline_exceeded', returncode=-15,
            training_executed=False, test_payload_reads=0, elapsed_seconds=5.))
        args = ['timeout', '--signal=TERM', '--kill-after=15s', '7230s', '/SOFTWARE_FIXTURE/python',
            '-E', '-s', '-B', operator['path'], request['path'], request['sha256']]
        owner = dict(schema='visualizeit_raw_owned_popen_terminal_v1', handle_name='expanded_group1_cached_phase_job',
            pid=5611, returncode=1, args=args, args_sha256=hashlib.sha256(raw.encoded(args)).hexdigest(),
            launch=launch, request=request, operator=operator, observation=observation)
        self.f.terminal['job'].update(process_start_token=None, process_start_token_at_poll=None, popen_owner=owner)
        self.f.flush()
        self.assertEqual(self.f.authenticate()['unattempted_physical_ids'], [p['id'] for p in self.f.pending])
        owner['args'][-1] = '0' * 64
        owner['args_sha256'] = hashlib.sha256(raw.encoded(owner['args'])).hexdigest()
        self.f.flush()
        with self.assertRaisesRegex(ValueError, 'owned Popen original launch/request'):
            self.f.authenticate()
        self.f.terminal['job'].pop('popen_owner')
        self.f.flush()
        with self.assertRaisesRegex(ValueError, 'owned Popen terminal witness'):
            self.f.authenticate()

    def test_original_physical_backend_success_failure_and_cleanup_with_fake_dependencies(self):
        from bench import quality_adaptation_larger as larger
        from bench import quality_adaptation_refinement_states as shared
        physical = copy.deepcopy(self.f.pending[0]); physical['mask']['mask'] = {'path': 'SOFTWARE_FIXTURE_MASK'}
        physical['bank']['bank'] = {'path': 'SOFTWARE_FIXTURE_BANK'}
        states = state_rows(dict(rows=[physical])); output = self.f.base / 'fake-backend'; output.mkdir()
        events = []; fail = [False]
        network = SimpleNamespace(eval=lambda: network, cpu=lambda: events.append('network_cpu'))
        torch = SimpleNamespace(manual_seed=lambda value: events.append(('seed', value)))
        class Renderer:
            def __init__(self, *args, **kwargs): events.append('renderer_created')
            def close(self): events.append('renderer_closed')
        class Search:
            def __init__(self, **kwargs): events.append('search_created')
            def close(self): events.append('search_closed')
            def bind(self, *args, **kwargs):
                return SimpleNamespace(recover=lambda *args, **kwargs: [] if fail[0] else [np.eye(4)])
        class Sink:
            def __init__(self, destination, frame, rows, *args, **kwargs):
                self.destination, self.frame, self.rows = destination, frame, rows
            def finish(self, reason): events.append(('sink_finished', reason))
        class Refiner:
            def __init__(self, *args, **kwargs): self.sink = kwargs.get('chronological_capture_callback')
            def set_chronological_capture_context(self, value): pass
            def refine(self, *args): materialize(self.sink.destination, self.sink.frame, self.sink.rows)
        @contextmanager
        def observed(selected): yield {}
        modules = {'bench.quality_adaptation_larger_optimized': SimpleNamespace(
            PreparedFoundPoseSearch=Search, Timings=lambda device: SimpleNamespace(torch=None)),
            'bench.quality_gotrack': SimpleNamespace(load_network=lambda device: network,
                TexturedRenderer=Renderer, GoTrackRefiner=Refiner),
            'bench.quality_contract': SimpleNamespace(Frame=lambda *args, **kwargs: object()),
            'bench.quality_runner': SimpleNamespace(read_rgb=lambda *args: np.zeros((1, 1, 3))),
            'bench.vision': SimpleNamespace(cv2=SimpleNamespace(IMREAD_GRAYSCALE=0, imread=lambda *args: np.ones((1, 1))))}
        inputs = {physical['id']: (self.f.base, dict(object_id=physical['object_id'],
            _timing=dict(time_policy='SOFTWARE_FIXTURE')), np.eye(3))}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(larger, 'configure_torch', return_value=torch))
            stack.enter_context(patch.object(larger, 'cleanup_device', side_effect=lambda *args: events.append('device_cleanup')))
            stack.enter_context(patch.object(larger, 'issue_portable_observer', return_value=object()))
            stack.enter_context(patch.object(shared, 'RefStateCapture', Sink))
            stack.enter_context(patch.object(shared, 'observe_selected_crops', observed))
            stack.enter_context(patch.object(raw, 'check_pin', return_value=self.f.base / 'unused-mask'))
            with raw._capture_backend(self.f.successor, b'SOFTWARE_FIXTURE', '0' * 64, inputs, states, output) as execute:
                execute(physical)
            self.assertEqual(states[0]['status'], 'available')
            self.assertEqual(events.count('renderer_closed'), 1)
            self.assertEqual(events.count('search_closed'), 1)
            self.assertEqual(events.count('device_cleanup'), 1)
            fail[0] = True; physical['seed'] = {'status': 'pending'}
            failed_states = state_rows(dict(rows=[physical]))
            with raw._capture_backend(self.f.successor, b'SOFTWARE_FIXTURE', '0' * 64, inputs, failed_states, output) as execute:
                execute(physical)
            self.assertEqual(physical['seed']['failure_reason'], 'no_estimated_FoundPose_hypothesis')
            self.assertEqual(failed_states[0]['status'], 'unavailable')
            self.assertIn('unavailable estimated seed', failed_states[0]['failure_reason'])
            self.assertEqual(events.count('renderer_closed'), 2)
            self.assertEqual(events.count('device_cleanup'), 2)

    def test_interruption_records_started_unit_and_does_not_retry_it(self):
        target = self.f.pending[1]['id']
        with self.assertRaisesRegex(raw.ResourceError, 'SOFTWARE_FIXTURE_INTERRUPTION'):
            self.run_fake(fail_at=target)
        attempt = self.f.root / 'capture-continuations/software-fixture'
        progress = raw.read_json(attempt / 'progress.json')
        receipt = raw.read_json(attempt / 'receipt.json')
        self.assertEqual(progress['in_progress'], target)
        self.assertEqual(receipt['attempted_physical_ids'], [p['id'] for p in self.f.pending[:2]])
        self.assertNotIn(target, receipt['remaining_unattempted_physical_ids'])
        self.assertEqual(receipt['status'], 'failed')
        self.assertEqual(self.cleaned, [True])
        with patch.object(raw, '_capture_backend', side_effect=AssertionError('MODEL MUST NOT LOAD')):
            with self.assertRaisesRegex(FileExistsError, 'prior continuation exists'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin, self.f.allowance())

    def test_cleanup_failure_is_observable_and_failed_receipt_retains_history(self):
        with self.assertRaisesRegex(RuntimeError, 'SOFTWARE_FIXTURE_CLEANUP_FAILURE'):
            self.run_fake(cleanup_failure=True)
        receipt = raw.read_json(self.f.root / 'capture-continuations/software-fixture/receipt.json')
        self.assertEqual(receipt['status'], 'failed')
        self.assertIn('SOFTWARE_FIXTURE_CLEANUP_FAILURE', receipt['fatal_error'])
        self.assertEqual(receipt['cumulative_elapsed_seconds'], 16. + receipt['elapsed_seconds'])

    def test_no_allowance_or_default_local_call_can_execute_backend(self):
        with patch.object(raw, '_capture_backend', side_effect=AssertionError('MODEL MUST NOT LOAD')):
            with self.assertRaisesRegex(ValueError, 'allowance'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin,
                    self.f.allowance(additional_allowance=False))
            with self.assertRaisesRegex(RuntimeError, 'Kaggle Linux'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin, self.f.allowance())
        self.assertFalse((self.f.root / 'capture-continuations').exists())

    def test_input_preflight_failure_happens_before_backend_and_is_retained(self):
        with patch.object(raw, '_capture_continuation_guard', self.fake_guard), \
                patch.object(raw, 'preflight_inputs', side_effect=ValueError('changed input payload')), \
                patch.object(raw, '_capture_backend', side_effect=AssertionError('MODEL MUST NOT LOAD')):
            with self.assertRaisesRegex(ValueError, 'changed input payload'):
                raw.continue_capture(self.f.context, self.f.successor, self.f.terminal_pin, self.f.allowance())
        receipt = raw.read_json(self.f.root / 'capture-continuations/software-fixture/receipt.json')
        self.assertEqual(receipt['attempted_physical_ids'], [])
        self.assertEqual(receipt['status'], 'failed')

    def test_mutated_payload_or_orphan_geometry_rejected(self):
        crop = next(self.f.output.glob('crop-*.npz'))
        data = crop.read_bytes()
        crop.write_bytes(data + b'altered')
        with self.assertRaisesRegex(ValueError, 'altered pinned artifact'):
            self.f.authenticate()
        crop.write_bytes(data)
        (self.f.output / 'capture-orphan.npz').write_bytes(b'SOFTWARE_FIXTURE_PARTIAL')
        with self.assertRaisesRegex(ValueError, 'orphan/partial'):
            self.f.authenticate()

    def test_source_archive_and_successor_source_mutations_rejected(self):
        self.f.terminal['source_archive']['bench/quality_adaptation_portable.py'] = {
            **self.f.archive, 'sha256': '0' * 64}
        self.f.flush()
        with self.assertRaisesRegex(ValueError, 'source archive identity'):
            self.f.authenticate()
        doc = self.f.successor.document()
        doc['source_files']['bench/quality_adaptation_portable.py'] = '1' * 64
        self.f.successor = detached(doc)
        with self.assertRaisesRegex(ValueError, 'executing source changed'):
            self.f.authenticate()

    def test_context_and_input_plan_changes_rejected(self):
        doc = self.f.successor.document()
        doc['device'] = 'cuda'
        with self.assertRaisesRegex(ValueError, 'context may change only'):
            raw.authenticate_capture_continuation(self.f.context, detached(doc), self.f.terminal_pin)
        plan = raw.read_json(self.f.root / 'input-plan.json')
        plan['rows'][0]['mesh_recipe']['sha256'] = 'f' * 64
        write(self.f.root / 'input-plan.json', plan)
        self.f.terminal['metadata']['input-plan.json'] = raw.pin(self.f.root / 'input-plan.json')
        self.f.terminal_pin = write(self.f.terminal_path, self.f.terminal)
        with self.assertRaisesRegex(ValueError, 'frozen physical input'):
            self.f.authenticate()

    def test_duplicate_missing_reordered_state_and_unit_identities_rejected(self):
        original_states, original_units = copy.deepcopy(self.f.states), copy.deepcopy(self.f.units)
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                       lambda rows: rows.reverse()):
            for kind in ('states', 'units'):
                with self.subTest(kind=kind, mutation=mutate):
                    self.f.states, self.f.units = copy.deepcopy(original_states), copy.deepcopy(original_units)
                    self.f.terminal['units'] = self.f.units
                    mutate(getattr(self.f, kind))
                    self.f.flush()
                    with self.assertRaisesRegex(ValueError, 'identity|inventory'):
                        self.f.authenticate()

    def test_terminal_must_attest_absence_identity_idle_gpu_and_execution_suffix(self):
        original = copy.deepcopy(self.f.terminal)
        for change in (dict(returncode=0), dict(root_and_descendants_absent=False),
                dict(process_start_token_at_poll='OTHER_PROCESS'), dict(gpu_compute_pids=[123]),
                dict(gpu_used_bytes=1), dict(absence_observed_unix_s=999.)):
            with self.subTest(change=change):
                self.f.terminal = copy.deepcopy(original)
                self.f.terminal['job'].update(change)
                self.f.flush()
                with self.assertRaisesRegex(ValueError, 'actual terminal'):
                    self.f.authenticate()
        self.f.terminal = copy.deepcopy(original)
        self.f.units = self.f.terminal['units']
        self.f.units[next(i for i, u in enumerate(self.f.units) if u['id'] == self.f.pending[0]['id'])]['outcome'] = 'failed'
        self.f.flush()
        with self.assertRaisesRegex(ValueError, 'observed failure'):
            self.f.authenticate()

    def test_test_reference_payload_rejected_before_any_capture_payload_open(self):
        test = next(p for p in self.f.physicals if p['split'] == 'test')
        test['mask'] = dict(status='available', mask=dict(path='MUST_NOT_OPEN_REFERENCE', bytes=1, sha256='a' * 64))
        self.f.flush()
        with patch.object(raw, 'preflight_inputs', side_effect=AssertionError('payload opened')):
            with self.assertRaisesRegex(ValueError, 'TEST/dev'):
                self.f.authenticate()

    def test_partial_chronology_is_preserved_and_replay_rejected(self):
        temporary = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(temporary.cleanup)
        f = Fixture(temporary.name, iterations=(0, 1))
        row = next(r for r in f.states if r['id'].startswith(f.completed['id'] + ':iteration'))
        row['pnp_state'] = 'solver_result_not_observed'
        f.flush()
        with self.assertRaisesRegex(ValueError, 'partial chronological solver'):
            f.authenticate()
        row['pnp_state'] = 'SOFTWARE_FIXTURE_TERMINAL_SOLVER_OUTCOME'
        for key in ('crops', 'capture'):
            (f.output / row.pop(key)['path']).unlink()
        row.update(status='unavailable', failure_reason='SOFTWARE_FIXTURE_INTERRUPT')
        f.flush()
        before = {p: p.read_bytes() for p in f.output.iterdir()}
        with self.assertRaisesRegex(ValueError, 'partial chronological'):
            f.authenticate()
        self.assertEqual(before, {p: p.read_bytes() for p in f.output.iterdir()})

    def test_stage_capture_elapsed_counted_once_and_prior_clock_bound_to_snapshot(self):
        write(self.f.root / 'stage-capture.json', dict(context_sha256=self.f.context.sha256,
            elapsed_seconds=5., status='failed'))
        history = write(self.f.root / 'cached-capture-phase-validation-v1.json', dict(
            source_historical_elapsed_seconds=17.))
        self.f.terminal['prior_history'] = dict(receipt=history, field='source_historical_elapsed_seconds')
        self.f.flush()
        # Independent arithmetic: 2 + 3*3 + 5 + 17 = 33, never 38.
        self.assertEqual(self.f.authenticate()['cumulative_elapsed_seconds'], 33.)
        self.f.terminal['capture_elapsed_seconds'] = 4.
        self.f.flush()
        with self.assertRaisesRegex(ValueError, 'count only once'):
            self.f.authenticate()

    def test_default_capture_still_uses_fresh_inventory_and_rejects_prior_output(self):
        # Fresh capture's real entry point still refuses an existing RAW tree.
        token = raw._EXECUTION.set((self.f.successor.sha256, 'capture'))
        try:
            with patch.object(raw, 'frozen', return_value=(self.f.successor.document(), {}, {}, self.f.physicals)), \
                    patch.object(raw, '_capture_backend', side_effect=AssertionError('MODEL MUST NOT LOAD')):
                with self.assertRaises(FileExistsError):
                    raw.capture(self.f.successor)
        finally:
            raw._EXECUTION.reset(token)

    def test_default_capture_success_failure_and_sealed_rows_via_controlled_callback(self):
        output = self.f.base / 'fresh-run'; output.mkdir()
        doc = self.f.successor.document(); doc['output_root'] = output.as_posix()
        context = detached(doc)
        physicals = copy.deepcopy(self.f.physicals)
        for p in physicals:
            p['seed'] = {'status': 'pending'}
        calls = []
        @contextmanager
        def backend(context, data, digest, inputs, states, destination):
            def execute(physical):
                calls.append(physical['id'])
                selected = [r for r in states if r['id'].startswith(physical['id'] + ':iteration')]
                if physical['split'] not in ('train', 'validation'):
                    for r in selected:
                        r['failure_reason'] = 'sealed_metadata_only'
                elif physical['id'] == self.f.failed['id']:
                    physical['seed'] = dict(status='unavailable', failure_reason='SOFTWARE_FIXTURE_OBSERVED_FAILURE')
                    for r in selected:
                        r['failure_reason'] = 'SOFTWARE_FIXTURE_OBSERVED_FAILURE'
                else:
                    materialize(destination, physical, states)
            yield execute
        token = raw._EXECUTION.set((context.sha256, 'capture'))
        try:
            with patch.object(raw, 'frozen', return_value=(doc, {}, {}, physicals)), \
                    patch.object(raw, 'preflight_inputs', return_value={}), \
                    patch.object(raw, '_capture_backend', backend):
                result = raw.capture(context)
        finally:
            raw._EXECUTION.reset(token)
        self.assertEqual(calls, [p['id'] for p in physicals])
        self.assertEqual(result['available'], len(raw._active(physicals)) - 1)
        receipt = raw.read_json(output / 'raw/capture-receipt.json')
        self.assertEqual(receipt['cleanup_errors'], [])
        self.assertIsNone(receipt['fatal_error'])
        self.assertFalse(receipt['training_ready'])


if __name__ == '__main__':
    unittest.main()
