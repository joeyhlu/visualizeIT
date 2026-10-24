"""MODEL-FREE integration tests with declared neural/render supervision doubles.

These fixtures are software evidence only. No model/training/data acquisition.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench.quality_adaptation_manifest import assemble
from bench.quality_adaptation_prepare import project_supervision
from bench.test_quality_adaptation_portable_prepare import Fixture, ROOT, write, known_role_fixture, forbid_payload_opens
from scripts import kaggle_prepared_v2 as prepared
from scripts import kaggle_pilot_v2 as pilot


def prepared_fixture(fixture):
    fixture.prepare_features()
    inventory = []
    for ordinal, source in enumerate(fixture.captures):
        if source['split'] == 'dev':
            continue
        row = dict(id=source['id'], object_id=source['object_id'], sequence_id=source['sequence_id'], capture_group=source['capture_group'],
            split='val' if source['split'] == 'validation' else source['split'], status='unavailable', query_rgb_sha256=source.get('query_rgb_sha256'),
            failure_reason=source.get('failure_reason', 'fixture_label_failure'))
        if source['id'] in fixture.arrays:
            crops, _ = fixture.arrays[source['id']]
            # Analytic renderer double: unit frontoparallel plane, identity pose,
            # all observed foreground. The real pure projection function runs.
            target = project_supervision(template_xyz_m=crops['template_xyz_m'], template_mask=crops['template_mask'], query_K=crops['query_K'],
                query_T_cam_from_object=np.eye(4), observed_depth_m=np.ones((280, 280), np.float32),
                observed_object_mask=np.ones((280, 280), bool), depth_tolerance_m=.002)
            path = fixture.features / f'labels-{ordinal}.npz'
            np.savez(path, **target)
            row.pop('failure_reason')
            row.update(status='available', labels={**p.pin(path), 'path': path.name})
        if row['query_rgb_sha256'] is None:
            row['unavailable_provenance'] = 'MODEL_FREE_TEST_FIXTURE; no crop materialized'
        inventory.append(row)
    write(fixture.features / 'labels.json', dict(schema='correspondence_label_inventory_v1', label_origin='synthetic_renderer_training_only', rows=inventory))
    assemble(fixture.features, fixture.features / 'labels.json', fixture.features / 'manifest.json')
    write(fixture.features / 'label-receipt.json', dict(schema='kaggle_label_receipt_v2', prelabel_seal=p.pin(fixture.execution / 'prelabel-seal.json'),
        inventory=p.pin(fixture.features / 'labels.json'), dataset=p.pin(fixture.features / 'manifest.json'), labels_entered_inference=False,
        test_payload_reads=0, independent_quality_claim=False, label_artifacts=[dict(id=r['id'], **r['labels']) for r in inventory if r['status'] == 'available']))
    runtime = write(fixture.base / 'runtime.json', dict(MODEL_FREE_TEST_FIXTURE=True))
    config = write(fixture.base / 'config.json', dict(device='cuda', epochs=1, max_steps=24, max_seconds=300., max_total_seconds=600.,
        max_checkpoint_bytes=16 * p.MIB, max_output_bytes=96 * p.MIB))
    evidence = write(fixture.base / 'execution-evidence.json', dict(MODEL_FREE_TEST_FIXTURE=True))
    return prepared.request_document(preparation=dict(source_root=str(ROOT), execution_root=str(fixture.execution), features_root=str(fixture.features),
        raw_root=str(fixture.root), exposure_audit=fixture.audit_pin), runtime=runtime['path'], config=config['path'], source_install=fixture.base / 'source-install',
        python=Path(sys.executable), input_root=fixture.base / 'input', working_root=fixture.base, output_root=fixture.base / 'run',
        profile=dict(kind='bounded_smoke', train_states_requested=1), limits=dict(max_seconds=600, max_rss_bytes=8 * p.GIB,
            max_output_bytes=2 * p.GIB, max_log_bytes=8 * p.MIB, reserve_bytes=p.GIB), quota_roots=[fixture.base], storage_limit_bytes=20 * p.GIB,
        execution_evidence=evidence['path'])


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)
        self.request = prepared_fixture(self.fixture)
        self.frozen = patch.object(raw, 'frozen', side_effect=self.fixture.frozen)
        self.frozen.start()
        self.addCleanup(self.frozen.stop)

    def test_genuine_assembler_and_existing_dataset_loader(self):
        joined = prepared.verify_prepared(self.request)
        self.assertEqual(joined['counts']['train'], dict(requested=24, crops=1, features=1, jointly_usable=1))
        self.assertEqual(joined['counts']['test']['requested'], 24)
        self.assertEqual(joined['counts']['dev']['requested'], 24)
        self.assertEqual(len(joined['failures']), 70)

    def test_smoke_allows_tiny_without_claiming_larger_training_data(self):
        joined = prepared.verify_prepared(self.request, payloads=False)
        admission = prepared.assess_admission(joined, self.request['profile'], resource_admitted=True)
        self.assertTrue(admission['admitted'])
        self.assertEqual(admission['counts']['train']['jointly_usable'], 1)
        self.assertFalse(any(r['met'] for r in admission['targets']))

    def test_target_needs_actual_joint_train_not_prospective_count(self):
        joined = prepared.verify_prepared(self.request, payloads=False)
        for target in (17, 1000, 2000, 4000):
            profile = dict(kind='target' if target != 17 else 'bounded_smoke', train_states_requested=target)
            admission = prepared.assess_admission(joined, profile, resource_admitted=True)
            self.assertFalse(admission['admitted'])
            self.assertIn('insufficient_actual_jointly_usable_train', admission['blockers'])

    def test_no_resource_admission_no_training(self):
        joined = prepared.verify_prepared(self.request, payloads=False)
        self.assertFalse(prepared.assess_admission(joined, self.request['profile'], resource_admitted=False)['admitted'])

    def test_last_sealed_dataset_change_rejected_before_payload_hash(self):
        dataset = p.metadata(self.request['dataset'])
        dataset['rows'][-1]['split'] = 'train'
        self.request['dataset'] = write(Path(self.request['dataset']['path']), dataset)
        with patch.object(p, 'verify_prelabel') as numerical:
            with self.assertRaises(ValueError):
                prepared.verify_prepared(self.request)
            numerical.assert_not_called()

    def test_final_index_mutation_rejected(self):
        value = p.metadata(self.request['feature_index'])
        value['rows'][0]['status'] = 'unavailable'
        value['rows'][0]['failure_reason'] = 'mutated_test_stimulus'
        self.request['feature_index'] = write(Path(self.request['feature_index']['path']), value)
        with self.assertRaises(ValueError):
            prepared.verify_prepared(self.request, payloads=False)

    def test_wrapper_pin_forgery_rejected(self):
        self.request['source_files']['scripts/kaggle_pilot_v2.py']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'artifact bytes changed'):
            prepared.verify_prepared(self.request, payloads=False)

    def test_physical_hardlink_alias_rejected_without_opening_bytes(self):
        one, two = self.fixture.base / 'one', self.fixture.base / 'two'
        one.write_bytes(b'fixture')
        try:
            os.link(one, two)
        except OSError:
            self.skipTest('host sandbox does not permit hardlinks')
        with self.assertRaisesRegex(ValueError, 'cross-split payload alias'):
            p.reject_aliases([('train', None, [one]), ('test', None, [two])])

    def test_copied_crosssplit_content_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cross-split RGB content'):
            p.reject_aliases([('train', 'a' * 64, []), ('test', 'a' * 64, [])])

    def test_resume_uses_same_dataset_config_and_checkpoint(self):
        joined = prepared.verify_prepared(self.request, payloads=False)
        output = prepared.output_layout(self.request)
        output.mkdir()
        p.atomic_json(output / 'v2-run-identity.json', joined['identity'])
        (output / 'train').mkdir()
        (output / 'train/last.pt').write_bytes(b'MODEL_FREE_CHECKPOINT_DOUBLE')
        prepared.checkpoint_receipt(output, joined['identity'])
        prepared.verify_resume(output, joined['identity'])
        changed = copy.deepcopy(joined['identity'])
        changed['config']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'resume/export changed'):
            prepared.verify_resume(output, changed)
        (output / 'train/last.pt').write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'artifact bytes changed'):
            prepared.verify_resume(output, joined['identity'])

    def test_v1_does_not_accept_v2_request(self):
        k = p.load_source(self.request['adapter'], '_test_frozen_adapter_v2', expected=p.ADAPTER_SHA256)
        with self.assertRaises(k.ContractError):
            k.validate_request(self.request, 'train')

    def test_pilot_caps_and_hard_bound(self):
        request = dict(schema='kaggle_pilot_request_v2', stop_after_steps=4, training=self.request)
        self.assertEqual(pilot.validate_caps(request), 4)
        for key, value in (('stop_after_steps', 5), ('stop_after_steps', 0), ('stop_after_steps', True)):
            changed = copy.deepcopy(request)
            changed[key] = value
            with self.assertRaises(ValueError):
                pilot.validate_caps(changed)
        changed = copy.deepcopy(request)
        changed['training']['limits']['max_seconds'] = 901
        with self.assertRaises(ValueError):
            pilot.validate_caps(changed)

    def test_bounded_smoke_full_scope_is_enforced_not_measured_forecast(self):
        joined = prepared.verify_prepared(self.request, payloads=False)
        before = dict(config=p.metadata(self.request['config']), dataset_sha256=joined['dataset_sha256'], source_lock_sha256='0' * 64)
        class AdapterDouble:
            def validate_limits(self, value):
                return value
        with patch.object(p, 'quota_check', return_value=dict(observed_fixture=True)):
            bound = pilot.full_scope_bound(dict(full_run=self.request), joined, AdapterDouble(), before, elapsed=2., steps=1)
        self.assertFalse(bound['completion_forecast_available'])
        self.assertFalse(bound['full_epoch_validation_time_measured'])
        self.assertEqual(bound['enforced_invocation_wall_seconds'], 600)

    def test_neural_execution_never_starts_on_local_fixture(self):
        with patch.object(p, 'source_runtime') as neural:
            with self.assertRaisesRegex(ValueError, 'Kaggle'):
                prepared.execute('train', self.request)
            neural.assert_not_called()

    def test_outputs_cannot_overlap_fixed_source_or_preparation(self):
        for key in ('features_root', 'execution_root', 'source_install'):
            request = copy.deepcopy(self.request)
            request['output_root'] = str(Path(request[key]) / 'run')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'overlaps'):
                prepared.output_layout(request)
        request = copy.deepcopy(self.request)
        request['working_root'] = str(ROOT)
        request['output_root'] = str(ROOT / 'bench/run')
        with self.assertRaisesRegex(ValueError, 'executing source'):
            prepared.output_layout(request)

    def test_real_consumer_orchestration_preflight_train_resume_export_doubles(self):
        # The complete actual metadata/numerical/dataset path runs. The only
        # neural producer is a declared file-writing supervised-process double.
        original = p.load_source(self.request['adapter'], '_test_commands_v2', expected=p.ADAPTER_SHA256)
        commands = []
        class AdapterDouble:
            validate_limits = staticmethod(original.validate_limits)
            command = staticmethod(original.command)
            clean_environment = staticmethod(original.clean_environment)
            file_sha256 = staticmethod(original.file_sha256)
            run_lock = staticmethod(original.run_lock)
            def supervise(self, argv, **kwargs):
                commands.append(argv)
                train = Path(self_request['paths']['train_dir'])
                stage = argv[argv.index('bench.quality_adaptation_train') + 1]
                if stage != 'preflight':
                    train.mkdir(exist_ok=True)
                    if stage == 'export':
                        Path(self_request['paths']['export_output']).write_bytes(b'MODEL_FREE_EXPORT_DOUBLE')
                    else:
                        (train / 'last.pt').write_bytes(('MODEL_FREE_CHECKPOINT_' + stage).encode())
                        write(train / 'status.json', dict(status='incomplete' if stage == 'train' else 'completed', reason='fixture', steps=1, epoch=0, elapsed_seconds=1., quality_claim=False, test_arrays_opened=False))
                return dict(status='incomplete' if stage == 'train' else 'completed', returncode=3 if stage == 'train' else 0,
                    reason='child_reported_incomplete' if stage == 'train' else None, elapsed_seconds=.1, peak_group_rss_bytes=1024)
        self_request = self.request
        with patch.object(p, 'cloud_guard'), patch.object(p, 'source_runtime', return_value=(AdapterDouble(), ROOT, {})), patch.object(p, 'quota_check', return_value={}), patch.object(prepared, 'fresh_evidence', return_value=dict(gpu=dict(uuid='GPU-DOUBLE'))):
            prepared.execute('preflight', self.request)
            self.assertFalse(Path(self.request['output_root']).exists())
            first = prepared.execute('train', self.request)
            self.assertEqual(first['result']['returncode'], 3)
            prepared.execute('resume', self.request)
            prepared.execute('export', self.request)
        self.assertEqual([argv[argv.index('bench.quality_adaptation_train') + 1] for argv in commands], ['preflight', 'train', 'resume', 'export'])
        self.assertTrue(Path(self.request['paths']['export_output']).is_file())
        joined = prepared.verify_prepared(self.request, payloads=False)
        prepared.verify_resume(Path(self.request['output_root']), joined['identity'])


class KnownRoleJoinTests(unittest.TestCase):
    """Two analytic MODEL_FREE states in the actual known15 metadata inventory.

    Existing raw.frozen source verification alone is doubled, as in the original
    fixture suite. Genuine raw metadata joins, identity bridge, numerical seal,
    projection, assembler and dataset metadata execute. No real asset is opened.
    """
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).absolute()
        recipe, audit = known_role_fixture(base)
        self.fixture = Fixture(base, recipe=recipe)
        self.fixture.audit = audit
        self.fixture.audit_pin = write(base / 'exposure-audit.json', audit)
        self.request = prepared_fixture(self.fixture)
        self.frozen = patch.object(raw, 'frozen', side_effect=self.fixture.frozen)
        self.frozen.start()
        self.addCleanup(self.frozen.stop)

    def test_known15_identity_split_bytes_sealed_metadata_and_disclosed_full_join(self):
        with forbid_payload_opens():
            joined = prepared.verify_prepared(self.request, payloads=False)
            plan = p.verify_execution(self.fixture.state, self.fixture.execution)
            self.assertEqual((self.fixture.root / 'split.json').read_bytes(), (self.fixture.execution / 'split.json').read_bytes())
            for old, new in zip(self.fixture.plan['global_manifest']['records'], plan['global_manifest']['records']):
                self.assertEqual({k: v for k, v in old.items() if k != 'inputs_sha256'}, {k: v for k, v in new.items() if k != 'inputs_sha256'})
            self.assertEqual(joined['counts']['train'], dict(requested=312, crops=1, features=1, jointly_usable=1))
            self.assertEqual(joined['counts']['validation'], dict(requested=24, crops=1, features=1, jointly_usable=1))
            self.assertEqual(joined['counts']['test'], dict(requested=24, crops=0, features=0, jointly_usable=0))
            admission = prepared.assess_admission(joined, dict(kind='bounded_smoke', train_states_requested=17), resource_admitted=True)
            self.assertFalse(admission['admitted'])
            self.assertEqual(admission['exposure_disclosure']['reused_development_validation_objects'], ['hot3d:obj_000004'])
            self.assertFalse(admission['exposure_disclosure']['independent_validation_claim'])
            self.assertFalse(admission['quality_claim'])
            self.assertTrue(all(not row['met'] for row in admission['targets']))
            test_rows = [r for r in joined['docs']['dataset']['rows'] if r['split'] == 'test']
            self.assertEqual(len(test_rows), 24)
            for row in test_rows:
                self.assertEqual(row['object_id'], '7')
                self.assertEqual(row['status'], 'unavailable')
                self.assertFalse(set(row) & {'crops', 'capture', 'features', 'labels'})
            bridge = p.read_json(self.fixture.execution / 'identity-bridge.json')
            for item in bridge['rows']:
                if item['split'] == 'test':
                    identity = p.metadata(item['provenance'])
                    self.assertEqual(identity['input_availability'], 'sealed_metadata_only')
                    self.assertEqual(identity['crop_availability'], 'unavailable')
                    self.assertFalse(set(identity) & {'source_inputs', 'capture_inputs', 'generation', 'labels'})

    def test_last_raw_sealed_test_rejects_all_payload_descriptors_without_open(self):
        original_rows = copy.deepcopy(self.fixture.captures)
        for key in ('crops', 'capture', 'features', 'labels'):
            rows = copy.deepcopy(original_rows)
            row = next(r for r in rows if r['split'] == 'test')
            rows.remove(row)
            rows.append(row)  # Poison is deliberately the final metadata row.
            row[key] = dict(path='POISON_SEALED_TEST.npz', bytes=1, sha256='0' * 64)
            index_pin = write(self.fixture.root / 'raw/capture-index.json', dict(rows=rows, training_ready=False))
            seal = copy.deepcopy(self.fixture.seal)
            seal['refinement_states'] = rows
            seal['artifacts']['raw/capture-index.json'] = index_pin
            seal = p.sealed({k: v for k, v in seal.items() if k != 'seal_sha256'})
            seal_pin = write(self.fixture.root / 'raw-seal.json', seal)
            with self.subTest(key=key), forbid_payload_opens(), self.assertRaisesRegex(ValueError, 'sealed TEST/dev'):
                p.raw_metadata(self.fixture.context, seal_pin)

    def test_last_dataset_test_feature_or_label_poison_rejects_before_any_payload(self):
        original_dataset = p.metadata(self.request['dataset'])
        original_inventory = p.metadata(self.request['labels'])
        original_receipt = p.metadata(self.request['label_receipt'])
        for key in ('features', 'labels'):
            dataset, inventory = copy.deepcopy(original_dataset), copy.deepcopy(original_inventory)
            target = next(row for row in dataset['rows'] if row['split'] == 'test')
            label = next(row for row in inventory['rows'] if row['id'] == target['id'])
            dataset['rows'].remove(target)
            dataset['rows'].append(target)
            inventory['rows'].remove(label)
            inventory['rows'].append(label)
            poison = dict(path='POISON_SEALED_TEST.npz', bytes=1, sha256='0' * 64)
            target[key] = poison
            if key == 'labels':
                label[key] = poison
            request = copy.deepcopy(self.request)
            request['dataset'] = write(Path(request['dataset']['path']), dataset)
            request['labels'] = write(Path(request['labels']['path']), inventory)
            receipt = copy.deepcopy(original_receipt)
            receipt.update(dataset=request['dataset'], inventory=request['labels'])
            request['label_receipt'] = write(Path(request['label_receipt']['path']), receipt)
            with self.subTest(key=key), forbid_payload_opens(), self.assertRaisesRegex(ValueError, 'unavailable row'):
                prepared.verify_prepared(request)

    def test_sealed_feature_index_descriptor_rejects_before_any_payload(self):
        index = p.metadata(self.request['feature_index'])
        row = next(r for r in index['rows'] if r['id'].startswith('portable-raw-obj7-'))
        index['rows'].remove(row)
        index['rows'].append(row)
        row['features'] = dict(path='POISON_SEALED_TEST.npz', bytes=1, sha256='0' * 64)
        request = copy.deepcopy(self.request)
        request['feature_index'] = write(Path(request['feature_index']['path']), index)
        binding = p.metadata(request['extraction_binding'])
        binding['final_index'] = request['feature_index']
        request['extraction_binding'] = write(Path(request['extraction_binding']['path']), binding)
        seal = p.metadata(request['prelabel_seal'])
        seal.update(index=request['feature_index'], binding=request['extraction_binding'])
        seal = p.sealed({k: v for k, v in seal.items() if k != 'seal_sha256'})
        request['prelabel_seal'] = write(Path(request['prelabel_seal']['path']), seal)
        receipt = p.metadata(request['label_receipt'])
        receipt['prelabel_seal'] = request['prelabel_seal']
        request['label_receipt'] = write(Path(request['label_receipt']['path']), receipt)
        with forbid_payload_opens(), self.assertRaisesRegex(ValueError, 'sealed TEST feature inventory'):
            prepared.verify_prepared(request)


class ImportTests(unittest.TestCase):
    def test_sealed_payload_symlink_rejected_without_opening_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
            base = Path(directory).absolute()
            original, alias = base / 'fixture-original.npz', base / 'fixture-test-alias.npz'
            original.write_bytes(b'MODEL_FREE_NONEXECUTABLE_PAYLOAD_TRAP')
            try:
                os.symlink(original, alias)
            except OSError:
                self.skipTest('host sandbox does not permit symlink creation; Linux behavior unverified')
            with forbid_payload_opens(), self.assertRaisesRegex(ValueError, 'symlink'):
                p.reject_aliases([('train', None, [original]), ('test', None, [alias])])

    def test_cli_preserves_incomplete_and_failed_exit_codes(self):
        for status, expected in (('incomplete', 3), ('failed', 2), ('completed', 0)):
            with self.subTest(status=status), patch.object(p, 'load_request', return_value={}), patch.object(p, 'prepare_stage', return_value=dict(status=status)), redirect_stdout(io.StringIO()):
                self.assertEqual(p.main(['extract', '--request', 'fixture.json', '--request-sha256', '0' * 64]), expected)
            with patch.object(p, 'load_request', return_value={}), patch.object(prepared, 'execute', return_value=dict(result=dict(status=status))), redirect_stdout(io.StringIO()):
                self.assertEqual(prepared.main(['train', '--request', 'fixture.json', '--request-sha256', '0' * 64]), expected)

    def test_wrapper_bytecode_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
            root = Path(directory).absolute()
            files = {}
            for name in prepared.SOURCE_NAMES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# MODEL_FREE_SOURCE_DESCRIPTOR_FIXTURE\n')
                files[name] = p.pin(path)
            (root / 'bench/stale.pyc').write_bytes(b'MODEL_FREE_NONEXECUTABLE_POISON')
            with self.assertRaisesRegex(ValueError, 'wrapper bytecode'):
                p.verify_no_bytecode(root)

    def test_quota_must_cover_raw_scratch_even_on_separate_device(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
            base = Path(directory).absolute()
            runtime = base / 'runtime'
            unaccounted = ROOT.parent / 'MODEL_FREE_UNOPENED_SCRATCH'
            request = dict(working_root=str(base), source_root=str(base / 'source'), source_install=str(base / 'install'),
                features_root=str(base / 'features'), execution_root=str(base / 'execution'), quota_roots=[str(base), str(runtime)], storage_limit_bytes=20 * p.GIB,
                limits=dict(max_rss_bytes=8 * p.GIB, max_output_bytes=2 * p.GIB, reserve_bytes=p.GIB),
                runtime=write(base / 'runtime.json', dict(runtime_root=str(runtime))),
                context=write(base / 'context.json', dict(output_root=str(base / 'raw'), scratch_root=str(unaccounted), model_cache=str(base / 'source/.cache/model-quality'))))
            class AccountingDouble:
                ROOT18 = 'visualizeit-cloud-v18'
                validate_limits = staticmethod(lambda value: value)
                runtime_link_policy = staticmethod(lambda receipt, working: {})
                unique_tree_bytes = staticmethod(lambda roots, **kwargs: 0)
                free_by_filesystem = staticmethod(lambda roots: {1: 40 * p.GIB})
            with self.assertRaisesRegex(ValueError, 'combined quota omits'):
                p.quota_check(request, AccountingDouble())
            request['quota_roots'].append(str(unaccounted))
            result = p.quota_check(request, AccountingDouble())
            self.assertEqual(result['quota_scope'], request['quota_roots'])

    def test_help_and_import_do_not_import_torch(self):
        for module in ('bench.quality_adaptation_portable_prepare', 'scripts.kaggle_prepared_v2', 'scripts.kaggle_pilot_v2'):
            code = f"import importlib,sys; importlib.import_module('{module}'); assert 'torch' not in sys.modules"
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, '-B', '-m', module, '--help'], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


class PilotExecuteDeadlineTests(unittest.TestCase):
    """Actual v2 execute + immutable old inspection/deadline/reconcile helpers.

    A faithful signal-state double supplies Linux ITIMER_REAL semantics on
    Windows; no OS signal, Torch, provider call or neural child executes.
    Supervised-child doubles write explicit MODEL_FREE inspection/checkpoint
    responses. Admission and neural launch are declared doubles only.
    """
    def run_pilot(self, mode='complete'):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
            base = Path(directory).absolute()
            old = p.load_source(p.pin(ROOT / 'scripts/kaggle_pilot.py'),
                '_kaggle_frozen_pilot_deadline_regression', expected=p.PILOT_SHA256)
            clock = SimpleNamespace(value=100.)
            clock.monotonic = lambda: clock.value
            class SignalStateDouble:
                ITIMER_REAL, SIGALRM = 0, 14
                def __init__(self):
                    self.timer = (5., 0.) if mode == 'preexisting_timer' else (0., 0.)
                    self.handler = 'original_handler'
                    self.arms = []
                def getitimer(self, which):
                    return self.timer
                def getsignal(self, which):
                    return self.handler
                def signal(self, which, handler):
                    previous, self.handler = self.handler, handler
                    return previous
                def setitimer(self, which, seconds):
                    previous, self.timer = self.timer, (seconds, 0.)
                    if seconds:
                        self.arms.append(clock.value + seconds)
                    return previous
                def expire(self):
                    clock.value = 161.
                    self.handler(self.SIGALRM, None)
            signals = SignalStateDouble()
            cfg = dict(device='cuda', max_steps=24, max_seconds=30., max_total_seconds=120., epochs=1,
                max_checkpoint_bytes=16 * p.MIB, max_output_bytes=96 * p.MIB)
            before = dict(config=cfg, config_sha256=p.canonical(cfg), dataset_sha256='a' * 64,
                source_lock_sha256='b' * 64, versions=dict(MODEL_FREE_FIXTURE=True))
            joined = dict(identity=dict(MODEL_FREE_IDENTITY=True), identity_sha256='c' * 64,
                dataset_sha256=before['dataset_sha256'], counts=dict(train=dict(jointly_usable=1), validation=dict(jointly_usable=1)))
            limits = dict(max_seconds=60, max_rss_bytes=8 * p.GIB, max_output_bytes=2 * p.GIB,
                max_log_bytes=8 * p.MIB, reserve_bytes=p.GIB)
            output = base / 'pilot'
            runtime = write(base / 'runtime.json', dict(MODEL_FREE_FIXTURE=True))
            training = dict(python=sys.executable, output_root=str(output), working_root=str(base), input_root=str(base),
                config=write(base / 'config.json', cfg), dataset=write(base / 'dataset.json', dict(MODEL_FREE_FIXTURE=True)),
                runtime=runtime, quota_roots=[str(base)], limits=limits, profile=dict(kind='bounded_smoke', train_states_requested=1),
                paths=dict(train_dir=str(output / 'train')))
            full = {**training, 'output_root': str(base / 'full')}
            request = dict(schema='kaggle_pilot_request_v2', stop_after_steps=1, training=training,
                full_run=full, pilot_helper=p.pin(ROOT / 'scripts/kaggle_pilot.py'))
            observations = dict(inspections=0, inspection_cleanup=0, train_started=False, train_cleanup=False, quota_calls=0)
            class AdapterDouble:
                read_json = staticmethod(p.read_json)
                canonical = staticmethod(p.canonical)
                validate_limits = staticmethod(lambda value: value)
                clean_environment = staticmethod(lambda: {})
                runtime_link_policy = staticmethod(lambda receipt, working: {})
                unique_tree_bytes = staticmethod(lambda roots, **kwargs: 1024)
                def supervise(self, argv, **kwargs):
                    # This is called by the AUTHENTIC old inspect_training.
                    # Its own timer must already cover this harmless child.
                    if signals.timer[0] <= 0:
                        raise AssertionError('inspection child lacks interruptible deadline')
                    observations['inspections'] += 1
                    ordinal = observations['inspections']
                    try:
                        if mode == 'initial_timeout' and ordinal == 1:
                            signals.expire()
                        if mode == 'later_timeout' and ordinal == 2:
                            signals.expire()
                        clock.value += 1.
                        value = copy.deepcopy(before)
                        if ordinal == 2:
                            value.update(checkpoint_sha256=p.sha256(output / 'train/last.pt'),
                                checkpoint=dict(schema='correspondence_head_checkpoint_v1', config_sha256=before['config_sha256'],
                                    dataset_sha256=before['dataset_sha256'], source_lock_sha256=before['source_lock_sha256'],
                                    versions=before['versions'], steps=1, epoch=0, batch_cursor=1, completed=False, elapsed_seconds=1.),
                                optimizer_steps=[1.], resume_state_present=True)
                        write(kwargs['log_path'], value)
                        if mode == 'inspection_cleanup_failure':
                            return dict(status='failed', returncode=1, reason='declared_child_cleanup_failure')
                        return dict(status='completed', returncode=0, reason=None)
                    finally:
                        observations['inspection_cleanup'] += 1
                def trainer_progress(self, *args, **kwargs):
                    if signals.timer[0] <= 0:
                        raise AssertionError('v2 work lacks interruptible deadline')
                    return dict(freshness='fresh', last_confirmed_steps=1)
            adapter = AdapterDouble()
            def quota(*args, **kwargs):
                observations['quota_calls'] += 1
                if mode == 'exhausted_before_inspection' and observations['quota_calls'] == 1:
                    clock.value = 161.
                if mode == 'full_scope_timeout' and observations['quota_calls'] == 3:
                    signals.expire()
                return dict(resident_bytes=1024)
            def neural_child_double(*args, **kwargs):
                observations['train_started'] = True
                if signals.timer[0] <= 0:
                    raise AssertionError('training segment lacks interruptible deadline')
                try:
                    if mode == 'training_timeout':
                        signals.expire()
                    clock.value += 1.
                    (output / 'train').mkdir()
                    (output / 'train/last.pt').write_bytes(b'MODEL_FREE_CHECKPOINT_DOUBLE')
                    write(output / 'train/status.json', dict(status='incomplete', reason='invocation_step_limit',
                        resumed_from=None, test_arrays_opened=False, quality_claim=False, steps=1, epoch=0, elapsed_seconds=1.))
                    result = dict(status='incomplete', returncode=3, reason='child_reported_incomplete',
                        elapsed_seconds=1., peak_group_rss_bytes=1024)
                    return dict(result=result, invocation=dict(stage='train', **result))
                finally:
                    observations['train_cleanup'] = True
            real_threads = []
            class FailedJoinDouble:
                def __init__(self, **kwargs):
                    self.thread = threading.Thread(**kwargs)
                    real_threads.append(self.thread)
                @property
                def ident(self):
                    return self.thread.ident
                def start(self):
                    self.thread.start()
                def join(self, **kwargs):
                    raise RuntimeError('declared sampler cleanup failure')
                def is_alive(self):
                    return self.thread.is_alive()
            thread_api = SimpleNamespace(Event=threading.Event,
                Thread=FailedJoinDouble if mode == 'sampler_cleanup_failure' else threading.Thread)
            with patch.object(pilot, 'time', clock), patch.object(old, 'time', clock), patch.object(old, 'sys', SimpleNamespace(platform='linux')), patch.object(old, 'signal', signals), patch.object(pilot, 'threading', thread_api), patch.object(prepared, 'verify_prepared', return_value=joined), patch.object(prepared, 'output_layout', return_value=output), patch.object(prepared, 'fresh_evidence'), patch.object(prepared, 'assess_admission', return_value=dict(admitted=True)), patch.object(prepared, 'execute', side_effect=neural_child_double), patch.object(p, 'cloud_guard'), patch.object(p, 'source_runtime', return_value=(adapter, ROOT, {})), patch.object(p, 'load_source', return_value=old), patch.object(p, 'quota_check', side_effect=quota):
                result = pilot.execute(request)
            for thread in real_threads:
                thread.join(timeout=1)  # Only fixture cleanup, no product join override.
                self.assertFalse(thread.is_alive())
            self.assertEqual(p.read_json(output / 'pilot-measurement.json'), result)
            self.assertEqual(signals.handler, 'original_handler')
            self.assertEqual(signals.timer, (5., 0.) if mode == 'preexisting_timer' else (0., 0.))
            # Every actual helper/v2 arm ends at the ORIGINAL absolute deadline.
            self.assertTrue(all(deadline == 160. for deadline in signals.arms))
            return result, observations, signals.arms

    def test_actual_execute_reaches_both_authentic_inspections(self):
        result, observed, arms = self.run_pilot()
        self.assertTrue(result['measurement_complete'])
        self.assertTrue(result['full_training_admitted'])
        self.assertEqual(observed['inspections'], 2)
        self.assertEqual(observed['inspection_cleanup'], 2)
        self.assertTrue(observed['train_cleanup'])
        self.assertEqual(result['verified_optimizer_steps'], 1)
        self.assertEqual(len(arms), 4)  # inspect, v2 train, inspect, v2 seal/bounds

    def test_initial_and_later_inspection_timeout_restore_timer(self):
        for mode, count in (('initial_timeout', 1), ('later_timeout', 2)):
            with self.subTest(mode=mode):
                result, observed, _ = self.run_pilot(mode)
                self.assertFalse(result['measurement_complete'])
                self.assertIn('PilotDeadline', result['reason'])
                self.assertEqual(observed['inspections'], count)
                self.assertEqual(observed['inspection_cleanup'], count)

    def test_v2_training_and_postinspection_timeouts_restore_timer(self):
        for mode in ('training_timeout', 'full_scope_timeout'):
            with self.subTest(mode=mode):
                result, observed, _ = self.run_pilot(mode)
                self.assertFalse(result['measurement_complete'])
                self.assertFalse(result['full_training_admitted'])
                self.assertIn('PilotDeadline', result['reason'])
                self.assertTrue(observed['train_cleanup'])

    def test_exhaustion_and_preexisting_timer_start_no_child(self):
        for mode in ('exhausted_before_inspection', 'preexisting_timer'):
            with self.subTest(mode=mode):
                result, observed, arms = self.run_pilot(mode)
                self.assertFalse(result['measurement_complete'])
                self.assertEqual(observed['inspections'], 0)
                self.assertFalse(observed['train_started'])
                self.assertEqual(arms, [])

    def test_failed_child_and_sampler_cleanup_do_not_admit_measurement(self):
        for mode in ('inspection_cleanup_failure', 'sampler_cleanup_failure'):
            with self.subTest(mode=mode):
                result, observed, _ = self.run_pilot(mode)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['measurement_complete'])
                self.assertFalse(result['full_training_admitted'])
                if mode == 'sampler_cleanup_failure':
                    self.assertIn('cleanup_errors', result)
                    self.assertTrue(observed['train_cleanup'])


if __name__ == '__main__':
    unittest.main()
