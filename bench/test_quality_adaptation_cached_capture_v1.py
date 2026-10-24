"""Model-free tests for authenticated cached RAW input-phase admission."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from bench import quality_adaptation_cached_capture_v1 as cached
from bench import quality_adaptation_mask_recovery_v1 as recovery
from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench.test_quality_adaptation_portable_prepare import known_role_fixture

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def _context(base, name):
    evidence = base / 'fixture-parity.json'
    if not evidence.exists():
        p.atomic_json(evidence, {'MODEL_FREE_TEST_FIXTURE': True})
    document = raw.context_document(output_root=base / (name + '-root'),
        scratch_root=base / (name + '-scratch'),
        parity={key: p.pin(evidence) for key in ('reference', 'candidate')},
        device='cpu', wall_seconds=7200, rss_bytes=8 * GIB,
        storage_bytes=32 * GIB, reserve_bytes=GIB)
    data = raw.encoded(document)
    return raw.PortableContext(data, hashlib.sha256(data).hexdigest())


class TerminalFixture:
    def __init__(self, base, context_name='source'):
        self.base = Path(base)
        self.source = _context(self.base, context_name)
        self.root = Path(self.source.document()['output_root'])
        self.root.mkdir()
        (self.root / 'context.json').write_bytes(self.source.raw)
        for name, value in (('freeze.json', {'elapsed_seconds': 4900.}),
                ('physical-index.json', {'rows': []}), ('stage-render.json', {'stage': 'render'}),
                ('stage-banks.json', {'stage': 'banks'}), ('stage-masks.json', {'stage': 'masks'})):
            p.atomic_json(self.root / name, value)
        self.alias = self.base / 'control-context.json'
        self.alias.write_bytes(self.source.raw)
        self.observation_path = self.base / 'outer-observation.json'
        self.observation = dict(schema='visualizeit_expanded_mask_recovery_observation_v1',
            stage='capture', shard_id='group1', status='incomplete', reason='interrupted',
            returncode=-15, elapsed_seconds=2211.046315514999, outputs_preserved=True,
            cleanup_errors=[], log_errors=[], training_executed=False, test_payload_reads=0,
            context=p.pin(self.alias), combined_raw_storage_watch=dict(errors=[], sampler_stopped=True,
                outer_cancelled=False))
        p.atomic_json(self.observation_path, self.observation)
        self.terminal_path = self.base / 'terminal.json'
        self.document = dict(schema=cached.TERMINAL_SCHEMA,
            source_context=p.pin(self.alias), source_context_sha256=self.source.sha256,
            job=dict(pid=4923, returncode=124, poll_observed_unix_s=1000., pid_absent=True,
                process_start_token=None, process_start_token_at_poll=None,
                timeout_seconds=2223, kill_grace_seconds=15),
            gpu_guard=dict(global_lock_available=True, compute_pids=[]),
            outer_observation=p.pin(self.observation_path),
            absence_observation=dict(observed_unix_s=1001., stage_capture_absent=True,
                capture_receipt_absent=True), watcher_elapsed_seconds=None,
            source_pins=dict(freeze=p.pin(self.root / 'freeze.json'),
                physical_index=p.pin(self.root / 'physical-index.json'),
                stage_render=p.pin(self.root / 'stage-render.json'),
                stage_banks=p.pin(self.root / 'stage-banks.json'),
                stage_masks=p.pin(self.root / 'stage-masks.json'),
                failed_capture_stage=None, capture_receipt=None))
        p.atomic_json(self.terminal_path, self.document)
        self.terminal_pin = p.pin(self.terminal_path)


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.fixture = TerminalFixture(self.temp.name)

    def test_actual_interrupted_outer_observation_authenticates_explicit_absence(self):
        result = cached._validate_terminal(self.fixture.source, self.fixture.terminal_pin)
        self.assertAlmostEqual(result['outer_elapsed_seconds'], 2211.046315514999)
        self.assertEqual(result['outer_upper_bound_seconds'], 2238.)
        self.assertIsNone(result['watcher_elapsed_seconds'])
        self.assertEqual(result['observation_pin'], self.fixture.document['outer_observation'])

    def test_running_pid_cleanup_failure_or_unobserved_absence_is_rejected(self):
        for mutate, message in (
                (lambda value: value['job'].update(pid_absent=False), 'absent PID'),
                (lambda value: value['absence_observation'].update(capture_receipt_absent=False), 'capture receipt'),
                (lambda value: value['gpu_guard'].update(compute_pids=[88]), 'GPU'),
                (lambda value: value['source_pins'].update(capture_receipt={'path': 'forged'}), 'missing RAW')):
            with self.subTest(message=message):
                changed = copy.deepcopy(self.fixture.document)
                mutate(changed)
                path = Path(self.temp.name) / ('changed-' + message.replace(' ', '-') + '.json')
                p.atomic_json(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    cached._validate_terminal(self.fixture.source, p.pin(path))

    def test_terminal_source_freeze_clock_pin_cannot_be_rewritten(self):
        (self.fixture.root / 'freeze.json').write_bytes(raw.encoded({'elapsed_seconds': 0.}))
        with self.assertRaisesRegex(ValueError, 'source pin differs'):
            cached._validate_terminal(self.fixture.source, self.fixture.terminal_pin)

    def test_review_must_bind_exact_adapter_and_recovery_bytes(self):
        changed = copy.deepcopy(self.fixture.document)
        changed['files'] = {cached.SOURCE_NAME: '0' * 64,
            cached.RECOVERY_NAME: p.sha256(Path(recovery.__file__))}
        review_path = Path(self.temp.name) / 'review.json'
        p.atomic_json(review_path, dict(schema=cached.REVIEW_SCHEMA, decision='APPROVE',
            scope=cached.REVIEW_SCOPE, original_raw_source_files=self.fixture.source.document()['source_files'],
            files=changed['files']))
        with self.assertRaisesRegex(ValueError, 'exact cached-capture source approval'):
            cached.approve(self.fixture.source, p.pin(review_path))


class MetadataBarrierTests(unittest.TestCase):
    def test_available_mask_cannot_carry_failure_or_unknown_fields(self):
        for extra in ({'failure_reason':'observed_failure'}, {'unknown':None}):
            row={'mask':{'status':'available','mask':{},'mask_index':{},**extra}}
            with self.subTest(extra=extra), patch.object(p,'verify_pin',side_effect=AssertionError('payload opened')):
                with self.assertRaisesRegex(ValueError,'exact completed observed mask'):
                    cached._source_mask_index(row,ROOT,'0'*64)

    def test_last_sealed_test_row_fails_before_any_payload_path_or_hash(self):
        fresh = [dict(id='train:first', object_id='1', physical_object_id='train',
            sequence_id='train', capture_group='g', split='train', frame_id=0,
            timestamp_s=0., view_index=0, iterations=[0], mesh_recipe={'sha256': 'a' * 64},
            render={'status': 'pending'}, bank={'status': 'pending'}, mask={'status': 'pending'},
            seed={'status': 'pending'}),
            dict(id='test:last', object_id='9', physical_object_id='test',
            sequence_id='test', capture_group='g', split='test', frame_id=0,
            timestamp_s=0., view_index=0, iterations=[0], mesh_recipe={'sha256': 'b' * 64},
            render={'status': 'pending'}, bank={'status': 'pending'}, mask={'status': 'pending'},
            seed={'status': 'pending'})]
        malformed = copy.deepcopy(fresh)
        malformed[-1]['mask'] = dict(status='available', mask={'path': 'must-not-open'})
        with patch.object(recovery, '_contained', side_effect=AssertionError('path traversed')), \
                patch.object(p, 'no_links', side_effect=AssertionError('symlink/path inspected')), \
                patch.object(p, 'verify_pin', side_effect=AssertionError('payload hashed')):
            with self.assertRaisesRegex(ValueError, 'sealed TEST'):
                cached._verify_active_identity(malformed, fresh)

    def test_context_identity_only_allows_fresh_roots_and_exact_budgets(self):
        old = dict(output_root='old', scratch_root='old-scratch', frozen='same',
            limits=dict(wall_seconds=7200, rss_bytes=8 * GIB,
                storage_bytes=32 * GIB, reserve_bytes=GIB))
        fresh = {**old, 'output_root': 'new', 'scratch_root': 'new-scratch'}
        cached._same_phase_context(old, fresh)
        for change in ({'frozen': 'changed'},
                {'limits': {**old['limits'], 'wall_seconds': 7201}},
                {'limits': {**old['limits'], 'storage_bytes': 31 * GIB}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cached._same_phase_context(old, {**fresh, **change})


@unittest.skipIf(os.name == 'nt', 'same-inode import is verified in the Linux cloud runtime')
class ImportPhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).absolute()
        self.source = _context(self.base, 'source')
        self.new = _context(self.base, 'new')
        self.old_root = Path(self.source.document()['output_root'])
        self.new_root = Path(self.new.document()['output_root'])
        recipe, _ = known_role_fixture(self.base)
        recipe_path = self.base / 'recipe.json'
        p.atomic_json(recipe_path, recipe)
        raw.freeze(recipe_path, p.sha256(recipe_path), self.source)
        _, spec, plan, rows = raw.frozen(self.source)
        for row in rows:
            if row['split'] in ('train', 'validation'):
                for kind in ('render', 'bank', 'mask'):
                    row[kind] = dict(status='unavailable', failure_reason='fixture_observed_failure')
                row['seed'] = {'status': 'pending'}
        selected = [next(row for row in rows if row['split'] == split)
                    for split in ('train', 'validation')]
        self.mask_paths = []
        for ordinal, row in enumerate(selected):
            self._add_available_input(row, ordinal)

        # The source context is an authenticated retry, so reproduce its
        # preserved failed-mask attempt and the exact recovery-import charge.
        # validate_extension is mocked below only at its public trust boundary;
        # this fixture still binds every marker and prior-attempt pin.
        self.prior = _context(self.base, 'prior-source')
        raw.freeze(recipe_path, p.sha256(recipe_path), self.prior)
        prior_root = Path(self.prior.document()['output_root'])
        prior_freeze = p.read_json(prior_root / 'freeze.json')
        prior_freeze['elapsed_seconds'] = 8.
        raw.write_json(self.prior, prior_root / 'freeze.json', prior_freeze, replace=True)
        for name, status, error in (('render', 'completed', None), ('banks', 'completed', None),
                                    ('masks', 'failed', 'fixture_prior_mask_failure')):
            p.atomic_json(prior_root / ('stage-' + name + '.json'), dict(stage=name,
                context_sha256=self.prior.sha256, status=status, error=error,
                elapsed_seconds=5. if name == 'masks' else 1., test_payload_reads=0,
                training_ready=False))
        freeze = p.read_json(self.old_root / 'freeze.json')
        freeze['elapsed_seconds'] = 100.
        raw.write_json(self.source, self.old_root / 'freeze.json', freeze, replace=True)
        raw._save_rows(self.source, rows)
        raw.write_json(self.source, self.old_root / 'input-plan.json', raw.capture_plan(self.source, rows))
        for name, elapsed in (('render', 10.), ('banks', 20.), ('masks', 40.)):
            receipt = dict(stage=name, context_sha256=self.source.sha256, status='completed', error=None,
                elapsed_seconds=elapsed, test_payload_reads=0, training_ready=False)
            if name in ('render', 'banks'):
                receipt.update(mode='authenticated_import', neural_execution=False,
                    original_context_sha256=self.prior.sha256,
                    original_receipt=p.pin(prior_root / ('stage-' + name + '.json')))
            p.atomic_json(self.old_root / ('stage-' + name + '.json'), receipt)
        recovery_review_path = self.base / 'recovery-review.json'
        p.atomic_json(recovery_review_path, dict(schema=recovery.REVIEW_SCHEMA,
            decision='APPROVE', original_raw_source_files=self.source.document()['source_files'],
            files={recovery.SOURCE_NAME: p.sha256(Path(recovery.__file__))}))
        self.recovery_review_pin = p.pin(recovery_review_path)
        marker = dict(schema='visualizeit_raw_mask_resource_source_v1', mode='authenticated_retry',
            context_sha256=self.source.sha256, elapsed_seconds=0., status='completed',
            adapter=p.pin(Path(recovery.__file__)), independent_review=self.recovery_review_pin,
            reservation='full_retained_union_and_actual_destination_filesystem',
            original_sources_changed=False, test_payload_reads=0, training_ready=False)
        p.atomic_json(self.old_root / recovery.MARKER, marker)
        import_marker = dict(schema='visualizeit_raw_authenticated_recovery_import_v1',
            context_sha256=self.source.sha256, status='completed',
            old_context=p.pin(prior_root / 'context.json'),
            old_physical_index=p.pin(prior_root / 'physical-index.json'),
            old_freeze=p.pin(prior_root / 'freeze.json'),
            old_failed_masks=p.pin(prior_root / 'stage-masks.json'),
            source_marker=p.pin(self.old_root / recovery.MARKER),
            charged_prior_seconds=13., transport_elapsed_seconds=7.5,
            elapsed_seconds=20.5, original_stage_seconds_charged_in_imported_receipts=True,
            prior_attempts_preserved=True, test_payload_reads=0, generation_payload_reads=0,
            training_ready=False)
        p.atomic_json(self.old_root / recovery.IMPORT_MARKER, import_marker)
        self.terminal = TerminalFixture(self.base, 'terminal-dummy')
        # Use the actual frozen source context and source-root pins for this
        # integration fixture; the terminal control alias remains byte-identical.
        self.terminal.source = self.source
        self.terminal.root = self.old_root
        self.terminal.alias.write_bytes(self.source.raw)
        self._rewrite_terminal_for_source()
        review_path = self.base / 'independent-review.json'
        p.atomic_json(review_path, dict(schema=cached.REVIEW_SCHEMA, decision='APPROVE',
            scope=cached.REVIEW_SCOPE, original_raw_source_files=self.new.document()['source_files'],
            files={cached.SOURCE_NAME: p.sha256(Path(cached.__file__)),
                   cached.RECOVERY_NAME: p.sha256(Path(recovery.__file__))}))
        self.review_pin = p.pin(review_path)

    def _add_available_input(self, row, ordinal):
        root = self.old_root
        view = raw._view_directory(root, row)
        bundle = view / 'bundle'
        bundle.mkdir(parents=True)
        render = {}
        for key, name in (('input', 'input.json'), ('video', 'synthetic.mp4'), ('mesh', 'object.glb')):
            path = bundle / name
            path.write_bytes(('MODEL_FREE_' + key).encode())
            render[key] = p.pin(path)
        (view / 'generation.json').write_bytes(b'POISON_GENERATION_MUST_NEVER_OPEN')
        bank_dir = root / 'banks'
        bank_dir.mkdir(exist_ok=True)
        bank = {}
        for key in ('bank', 'cnos_bank'):
            path = bank_dir / (row['object_id'] + '-' + key + '.npz')
            path.write_bytes(('MODEL_FREE_' + key).encode())
            bank[key] = p.pin(path)
        p.atomic_json(root / f"cnos-{int(row['object_id']):06d}.json", {'bank': bank['cnos_bank']})
        mask_path = root / 'masks' / f'mask-{ordinal}.png'
        mask_path.parent.mkdir(exist_ok=True)
        mask_path.write_bytes(b'MODEL_FREE_MASK_BYTES')
        mask = p.pin(mask_path)
        index = dict(schema='kaggle_raw_observed_mask_v1', automatic=True,
            mask_origin='observed_CNOS_box_then_SAM2_image_predictor', context_sha256=self.source.sha256,
            physical_id=row['id'], input=render['input'], video=render['video'], mesh=render['mesh'],
            cnos_bank=bank['cnos_bank'], mask=mask, annotation_inputs=[], test_payload_reads=0,
            detector={'bank_sha256': bank['cnos_bank']['sha256']}, prompt_box_xyxy=[1, 2, 10, 12],
            proposal_intermediate_sha256='c' * 64)
        index_path = root / 'masks' / f'index-{ordinal}.json'
        p.atomic_json(index_path, index)
        row.update(render=dict(status='available', **render), bank=dict(status='available', **bank),
            mask=dict(status='available', failure_reason=None, mask=mask, mask_index=p.pin(index_path)))

    def _rewrite_terminal_for_source(self):
        # Replace only fixture-specific context/artifact pins with the real
        # source context and extant cached-input receipt pins.
        source_alias = self.base / 'control-source-context.json'
        source_alias.write_bytes(self.source.raw)
        observation_path = self.base / 'outer-observation-actual.json'
        observation = dict(self.terminal.observation, context=p.pin(source_alias))
        p.atomic_json(observation_path, observation)
        terminal_path = self.base / 'terminal-actual.json'
        document = dict(self.terminal.document, source_context=p.pin(source_alias),
            source_context_sha256=self.source.sha256, outer_observation=p.pin(observation_path),
            source_pins=dict(freeze=p.pin(self.old_root / 'freeze.json'),
                physical_index=p.pin(self.old_root / 'physical-index.json'),
                stage_render=p.pin(self.old_root / 'stage-render.json'),
                stage_banks=p.pin(self.old_root / 'stage-banks.json'),
                stage_masks=p.pin(self.old_root / 'stage-masks.json'),
                failed_capture_stage=None, capture_receipt=None))
        p.atomic_json(terminal_path, document)
        self.terminal_pin = p.pin(terminal_path)

    def test_phase_imports_only_authenticated_inputs_and_reports_both_clocks(self):
        from contextlib import contextmanager
        import sys

        stage_names, env_checks, storage_samples, preflights = [], [], [], []
        @contextmanager
        def guarded_import_stage(context, name, *, cloud_execution=False):
            self.assertTrue(cloud_execution)
            began = time.monotonic()
            stage_names.append(name)
            raw.resource_check(context)
            raw.frozen(context)
            raw.verify_neural_environment(context)
            try:
                yield
            except BaseException:
                raise
            else:
                p.atomic_json(Path(context.document()['output_root']) / f'stage-{name}.json',
                    dict(stage=name, context_sha256=context.sha256, status='completed', error=None,
                        elapsed_seconds=time.monotonic() - began,
                        test_payload_reads=0, training_ready=False))

        def fake_preflight(data, expected, context):
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
            plan = raw._loads(data, 'fixture preflight')
            self.assertEqual(plan, raw.capture_plan(context, raw.frozen(context)[3]))
            rows = plan['rows']
            sealed_rows = [r for r in rows if r['split'] in ('test', 'dev')]
            self.assertTrue(sealed_rows)
            self.assertTrue(all(all(r[key] == {'status': 'pending'}
                for key in ('render', 'bank', 'mask')) for r in sealed_rows))
            preflights.append(context.sha256)
            return {row['id']: None for row in rows if row['split'] in ('train', 'validation')
                and all(row[key]['status'] == 'available' for key in ('render', 'bank', 'mask'))}

        class Pixels:
            shape = (480, 640)
            def any(self): return True
        class FakeCV2:
            IMREAD_GRAYSCALE = 0
            @staticmethod
            def imread(path, flags): return Pixels()

        original_open = Path.open
        def guard_generation(path, *args, **kwargs):
            if path.name == 'generation.json':
                raise AssertionError('opaque generation sidecar opened')
            return original_open(path, *args, **kwargs)
        import types
        fake_vision = types.ModuleType('bench.vision')
        fake_vision.cv2 = FakeCV2
        source_before = (self.old_root / 'physical-index.json').read_bytes()
        source_receipt_names = ('render', 'banks', 'masks', recovery.MARKER, recovery.IMPORT_MARKER)
        source_receipt_paths = {name: self.old_root / (
            name if name.endswith('.json') else 'stage-' + name + '.json')
            for name in source_receipt_names}
        old_stage_bytes = {name: path.read_bytes() for name, path in source_receipt_paths.items()}
        with patch.object(recovery, 'validate_extension',
                          side_effect=lambda context, review: p.read_json(
                              Path(context.document()['output_root']) / recovery.IMPORT_MARKER)), \
                patch.object(raw, 'stage', guarded_import_stage), \
                patch.object(raw, 'resource_check', side_effect=lambda *args, **kw: None), \
                patch.object(raw, 'verify_neural_environment', side_effect=lambda context: env_checks.append(context.sha256)), \
                patch.object(raw, 'preflight_inputs', side_effect=fake_preflight), \
                patch.object(Path, 'open', guard_generation), \
                patch.dict(sys.modules, {'bench.vision': fake_vision}):
            result = cached.prepare_phase(self.source, self.new, self.review_pin, self.terminal_pin,
                sample_combined=lambda: storage_samples.append({'resident_bytes': 64}) or {'resident_bytes': 64})
            admission_bytes = (self.new_root / cached.PHASE_FILE).read_bytes()
            freeze_bytes = (self.new_root / 'freeze.json').read_bytes()
            before_capture = cached.validate_phase(self.new, self.review_pin)
            self.assertEqual(admission_bytes, (self.new_root / cached.PHASE_FILE).read_bytes())
            self.assertEqual(freeze_bytes, (self.new_root / 'freeze.json').read_bytes())
            self.assertEqual(set(before_capture['stage_receipts']), {'render', 'banks', 'masks'})
            self.assertFalse((self.new_root / 'raw').exists())

            phase_rows = raw.frozen(self.new)[3]
            from bench.quality_adaptation_refinement_states import state_rows
            capture_states = state_rows(dict(rows=phase_rows))
            raw.write_json(self.new, self.new_root / 'input-plan.json',
                raw.capture_plan(self.new, phase_rows))
            (self.new_root / 'raw').mkdir()
            p.atomic_json(self.new_root / 'raw' / 'capture-receipt.json', dict(
                schema='kaggle_raw_capture_receipt_v1', context_sha256=self.new.sha256,
                input_plan_sha256=p.sha256(self.new_root / 'input-plan.json'), fatal_error=None,
                cleanup_errors=[], physical_frames_requested=len(phase_rows),
                refinement_states_requested=len(capture_states), annotation_inputs=[], test_payload_reads=0,
                training_ready=False))
            p.atomic_json(self.new_root / 'stage-capture.json', dict(stage='capture',
                context_sha256=self.new.sha256, status='completed', error=None,
                elapsed_seconds=120., test_payload_reads=0, training_ready=False))
            after_capture = cached.validate_phase(self.new, self.review_pin)
            self.assertEqual(admission_bytes, (self.new_root / cached.PHASE_FILE).read_bytes())
            self.assertEqual(freeze_bytes, (self.new_root / 'freeze.json').read_bytes())
            self.assertEqual(set(after_capture['stage_receipts']), {'render', 'banks', 'masks', 'capture'})
            self.assertAlmostEqual(after_capture['current_phase_elapsed_seconds'],
                before_capture['current_phase_elapsed_seconds'] + 120.)

            sealed_artifacts = {name: p.pin(self.new_root / ('stage-' + name + '.json'))
                for name in ('render', 'banks', 'masks', 'capture')}
            seal = dict(schema='kaggle_raw_capture_seal_v1', context_sha256=self.new.sha256,
                artifacts={'stage-' + name + '.json': pin for name, pin in sealed_artifacts.items()},
                training_ready=False, test_payload_reads=0, capture_fatal_error=None,
                cleanup_errors=[])
            p.atomic_json(self.new_root / 'raw-seal.json', p.sealed(seal))
            p.atomic_json(self.new_root / 'stage-seal-capture.json', dict(stage='seal-capture',
                context_sha256=self.new.sha256, status='completed', error=None,
                elapsed_seconds=1., test_payload_reads=0, training_ready=False))
            after_seal = cached.validate_phase(self.new, self.review_pin)
            self.assertEqual(admission_bytes, (self.new_root / cached.PHASE_FILE).read_bytes())
            self.assertEqual(freeze_bytes, (self.new_root / 'freeze.json').read_bytes())
            self.assertEqual(set(after_seal['stage_receipts']),
                {'render', 'banks', 'masks', 'capture', 'seal-capture'})
            self.assertAlmostEqual(after_seal['current_phase_elapsed_seconds'],
                after_capture['current_phase_elapsed_seconds'] + 1.)

        self.assertEqual(stage_names, ['render', 'banks', 'masks'])
        self.assertEqual(len(env_checks), 3)
        self.assertGreaterEqual(len(storage_samples), 5)
        self.assertEqual(len(preflights), 5)
        self.assertEqual(result['original_cumulative_budget_compliance'], False)
        self.assertAlmostEqual(result['combined_historical_plus_phase_seconds'],
            100. + 10. + 20. + 40. + 20.5 + 2238. + result['current_phase_elapsed_seconds'])
        self.assertEqual(result['frozen_recipe_identity']['recipe_sha256'],
            result['frozen_recipe_identity']['phase_recipe_sha256'])
        self.assertEqual(result['source_identity_pins']['capture_receipt'], None)
        self.assertEqual(result['source_identity_pins']['failed_capture_stage'], None)
        self.assertEqual(source_before, (self.old_root / 'physical-index.json').read_bytes())
        self.assertEqual(old_stage_bytes, {name: path.read_bytes()
            for name, path in source_receipt_paths.items()})
        phase_rows = raw.frozen(self.new)[3]
        self.assertEqual(cached._status_counts(phase_rows, 'mask'),
                         {'available': 2, 'unavailable': 334, 'pending': 0})
        for row in phase_rows:
            if row['split'] in ('test', 'dev'):
                self.assertTrue(all(row[key] == {'status': 'pending'}
                    for key in ('render', 'bank', 'mask', 'seed')))
            if row['split'] in ('train', 'validation'):
                self.assertEqual(row['seed'], {'status': 'pending'})

        phase_index_before = (self.new_root / 'physical-index.json').read_bytes()
        with self.assertRaises(FileExistsError):
            cached.prepare_phase(self.source, self.new, self.review_pin, self.terminal_pin,
                sample_combined=lambda: {'resident_bytes': 64})
        self.assertEqual(phase_index_before, (self.new_root / 'physical-index.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
