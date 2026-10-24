"""Software-only deferred supervision tests; no model/training quality claims."""
from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import kaggle_sharded_prepared_v1 as s
from bench.test_quality_adaptation_portable_prepare import Fixture, ROOT, write


class DeferredChildLabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / '.cache', prefix='dl-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).absolute()

    def request(self, *, deferred=True, fixture=None):
        control = self.base / 'control'
        control.mkdir()
        metadata = write(self.base / 'metadata.json', {'MODEL_FREE_TEST_FIXTURE': True})
        plan = None if deferred else write(control / 'legacy-plan.json', {'generations': []})
        original_pin = s.p.pin
        def no_early_plan(path, *args, **kwargs):
            if Path(path).name in ('child-label-plan.json', 'generation.json'):
                raise AssertionError('early label materialization: ' + str(path))
            return original_pin(path, *args, **kwargs)
        # Explicit platform/provider/partition doubles; the real constructor,
        # source binding, file pins and semantic identity run unchanged.
        with patch.object(s.sharded, 'verify_partition', return_value={'children': {'group1': {}}}), \
                patch.object(s, '_check_child_layout'), \
                patch.object(s, '_validate_child_evidence_metadata'), \
                patch.object(s.p, 'pin', side_effect=no_early_plan):
            return s.child_request_document(partition=metadata, shard_id='group1',
                context=s.p.pin(fixture.root / 'context.json') if fixture else metadata,
                raw_seal=fixture.seal_pin if fixture else metadata, projection=metadata,
                exposure_audit=fixture.audit_pin if fixture else metadata, label_plan=plan, source_root=ROOT,
                source_install=self.base / 'source-install', python=Path(sys.executable),
                runtime=metadata, config=metadata, adapter=ROOT / 'scripts/kaggle_adaptation.py',
                input_root=self.base / 'input', working_root=self.base,
                execution_root=fixture.execution if fixture else self.base / 'execution',
                features_root=fixture.features if fixture else self.base / 'features',
                feature_storage_root=self.base / 'feature-parent', retained_raw_roots=[str(self.base)],
                control_root=control, ui_context=metadata, execution_evidence=metadata,
                limits={'max_seconds': 600, 'max_output_bytes': 2 * s.GIB},
                quota_roots=[str(self.base)], defer_label_plan=deferred)

    def sealed_request(self):
        request = self.request()
        fixture_base = self.base / 'numerical'
        fixture_base.mkdir()
        fixture = Fixture(fixture_base)
        # Real original execution/crop/array/parity seal, analytical zero flow.
        seal = fixture.prepare_features()
        request.update(execution_root=str(fixture.execution), features_root=str(fixture.features))
        request['stage_request_identity_sha256'] = s._child_stage_request_identity(request)
        history = [{'stage': stage, 'ui_context': request['ui_context'],
                    'execution_evidence': request['execution_evidence']}
                   for stage in s.CHILD_STAGE_NAMES[:3]]
        request['stage_evidence_history'] = history
        s._write_child_stage_observation(request, 'seal-features', seal)
        # A test-only opaque sidecar is first materialized after numerical seal.
        write(fixture.base / 'generation.json', {'MODEL_FREE_TEST_FIXTURE': True})
        return request, fixture, history

    def binder_doubles(self, fixture, history):
        stack = ExitStack()
        # Only lineage/provider fixtures are doubled here; numerical verification
        # and all late plan/binding operations remain real.
        stack.enter_context(patch.object(s, '_child_supervision_provenance', return_value={}))
        stack.enter_context(patch.object(s, '_verify_child_stage_history', return_value=history))
        stack.enter_context(patch.object(s, '_child_raw', return_value=({}, fixture.state)))
        stack.enter_context(patch.object(s, '_verify_child_supervision', return_value={}))
        return stack

    def test_constructor_never_opens_prospective_plan_and_legacy_hash_is_unchanged(self):
        request = self.request()
        self.assertNotIn('label_plan', request)
        self.assertNotIn('child_label_binding', request)
        self.assertEqual(s._deferred_label_slot(request), self.base / 'control/child-label-plan.json')
        s._child_supervision_provenance(request)
        # Analytical legacy canonicalization: the unchanged previous field list.
        legacy = copy.deepcopy(request)
        legacy['sharded_child'].pop('label_binding_mode')
        legacy['sharded_child'].pop('label_plan_path')
        legacy['label_plan'] = request['context']
        keys = ('schema', 'source_root', 'source_files', 'adapter', 'source_install', 'python',
            'runtime', 'config', 'input_root', 'working_root', 'execution_root', 'features_root',
            'feature_storage_root', 'retained_raw_roots', 'control_root', 'context', 'raw_seal',
            'exposure_audit', 'label_plan', 'neural_inputs', 'limits', 'storage_limit_bytes', 'quota_roots', 'profile')
        material = {key: legacy.get(key) for key in keys}
        material['sharded_child'] = {key: legacy['sharded_child'].get(key) for key in
            ('schema', 'partition', 'shard_id', 'projection', 'storage_limit_bytes', 'supervision_sources', 'source_binding')}
        self.assertEqual(s._child_stage_request_identity(legacy), s._canonical(material))
        with self.assertRaisesRegex(ValueError, 'source binding changed'):
            s._child_supervision_provenance(legacy)

    def test_missing_numerical_seal_fails_before_generation_pin(self):
        request, fixture, history = self.sealed_request()
        (fixture.execution / 'prelabel-seal.json').unlink()
        original_pin = s.p.pin
        def guard(path, *args, **kwargs):
            self.assertNotIn(Path(path).name, ('generation.json', 'child-label-plan.json'))
            return original_pin(path, *args, **kwargs)
        with self.binder_doubles(fixture, history), patch.object(s.p, 'pin', side_effect=guard):
            with self.assertRaises(FileNotFoundError):
                s.bind_child_label_plan(request)
        self.assertFalse((self.base / 'control/child-label-plan.json').exists())

    def test_real_numerical_seal_precedes_plan_and_binding_preserves_input_identity(self):
        request, fixture, history = self.sealed_request()
        events = []
        verify, pin = s.p.verify_prelabel, s.p.pin
        def numerical(*args, **kwargs):
            result = verify(*args, **kwargs)
            events.append('numerical-seal')
            return result
        def observed_pin(path, *args, **kwargs):
            if Path(path).name in ('generation.json', 'child-label-plan.json'):
                self.assertIn('numerical-seal', events)
                events.append(Path(path).name)
            return pin(path, *args, **kwargs)
        with self.binder_doubles(fixture, history), \
                patch.object(s.p, 'verify_prelabel', side_effect=numerical), \
                patch.object(s.p, 'pin', side_effect=observed_pin):
            bound = s.bind_child_label_plan(request)
        self.assertEqual(s._child_stage_request_identity(bound), request['stage_request_identity_sha256'])
        self.assertNotIn('label_plan', request)
        self.assertEqual(events[0], 'numerical-seal')
        receipt = {'label_plan': bound['label_plan'], 'child_label_binding': bound['child_label_binding']}
        s._verify_child_label_binding(bound, receipt=receipt, verify_plan=True)
        for field in receipt:
            bad = dict(receipt); bad[field] = request['context']
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'receipt differs'):
                s._verify_child_label_binding(bound, receipt=bad)
        for field in ('context', 'raw_seal'):
            bad = copy.deepcopy(bound); bad[field] = {'path': '/changed', 'bytes': 1, 'sha256': '0' * 64}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'lineage changed'):
                s._verify_child_label_binding(bad)
        Path(bound['label_plan']['path']).write_text('{"changed":true}')
        # Metadata-only aggregate admission must not open the plan itself.
        s._verify_child_label_binding(bound, receipt=receipt)
        with self.assertRaises(ValueError):
            s._verify_child_label_binding(bound, receipt=receipt, verify_plan=True)
        with self.binder_doubles(fixture, history), self.assertRaisesRegex(ValueError, 'fresh explicit'):
            s.bind_child_label_plan(bound)

    def test_history_without_actual_seal_observation_rejects_before_plan(self):
        request, fixture, history = self.sealed_request()
        (fixture.execution / 'seal_features-stage-observation.json').unlink()
        with self.binder_doubles(fixture, history), self.assertRaises(FileNotFoundError):
            s.bind_child_label_plan(request)
        self.assertFalse((self.base / 'control/child-label-plan.json').exists())

    def test_unknown_mode_mixed_fields_and_relocated_slot_rejected(self):
        request = self.request()
        for mode in ('unknown', None):
            bad = copy.deepcopy(request); bad['sharded_child']['label_binding_mode'] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError): s._deferred_label_slot(bad)
        bad = copy.deepcopy(request); bad['label_plan'] = request['context']
        with self.assertRaisesRegex(ValueError, 'attached together'): s._deferred_label_slot(bad)
        bad = copy.deepcopy(request); bad['sharded_child']['label_plan_path'] = str(self.base / 'outside.json')
        with self.assertRaisesRegex(ValueError, 'frozen control-root slot'): s._deferred_label_slot(bad)

    def test_aggregate_metadata_rejects_stripping_deferred_mode_and_binding(self):
        request = self.request()
        request['sharded_child'].pop('label_binding_mode')
        request['sharded_child'].pop('label_plan_path')
        request['label_plan'] = request['context']
        joined = {'docs': {'extraction_binding': {}, 'label_receipt': {}}}
        with patch.object(s.prepared, 'metadata_join', return_value=joined), \
                patch.object(s, '_verify_child_supervision', return_value={}):
            with self.assertRaisesRegex(ValueError, 'deferred-label source binding changed'):
                s._child_metadata(request, {}, request['context'], 'group1', request['context'])

    def test_fresh_child_identity_extract_seal_then_bind_with_no_early_labels(self):
        base = self.base / 'fresh'
        base.mkdir()
        fixture = Fixture(base)
        fixture.features.rmdir()  # Empty generated fixture directory, no real data.
        request = self.request(fixture=fixture)
        self.assertFalse(fixture.execution.exists())
        self.assertFalse(fixture.features.exists())
        self.assertFalse(s._deferred_label_slot(request).exists())
        proof = {'projection': {'partition_sha256': 'MODEL_FREE_TEST_PARTITION'}}
        adapter = s.p.load_source(request['adapter'], '_deferred_test_adapter')

        def extract_double(current, **kwargs):
            self.assertEqual(kwargs['stage'], 'extract')
            self.assertNotIn('label_plan', current)
            self.assertFalse(s._deferred_label_slot(current).exists())
            self.assertFalse((fixture.execution / 'prelabel-seal.json').exists())
            fixture.features.mkdir()
            # Reuse only the existing fixture's analytical extractor arrays.
            # Execution/crop manifest already came from real child wrappers;
            # numerical sealing must remain a later, separate real stage.
            plan = s.p.verify_execution(fixture.state, fixture.execution)
            manifest = s.p.pin(fixture.root / 'raw/v2-crops.json')
            with patch.object(s.p, 'execution_identity', return_value=plan), \
                    patch.object(s.p, 'crop_manifest', return_value=manifest), \
                    patch.object(s.p, 'seal_features', return_value={}):
                fixture.prepare_features()
            original_index = s.p.read_json(fixture.features / 'original-index.json')
            for name in ('original-index.json', 'original-recipe.json', 'extraction-binding.json'):
                (fixture.features / name).unlink()
            write(fixture.features / 'index.json', original_index)
            return {'status': 'completed', 'returncode': 0}, {'MODEL_FREE_TEST_FIXTURE': True}

        def history(current, **kwargs):
            return current['stage_evidence_history']

        with patch.object(s.p, 'cloud_guard'), \
                patch.object(s, 'child_quota_check', return_value={'resident_bytes': 0}), \
                patch.object(s.p, 'source_runtime', return_value=(adapter, base, {})), \
                patch.object(s.p, 'preparation_evidence', return_value={}), \
                patch.object(s, '_validate_child_evidence_metadata'), \
                patch.object(s, '_verify_child_stage_history', side_effect=history), \
                patch.object(s, '_child_raw', return_value=(proof, fixture.state)), \
                patch.object(s, '_run_child_original', side_effect=extract_double):
            s.child_prepare_identity(request)
            request = s.child_stage_request(request, stage='extract',
                ui_context=request['ui_context'], execution_evidence=request['execution_evidence'])
            s.child_extract(request)
            self.assertNotIn('label_plan', request)
            self.assertFalse((fixture.execution / 'prelabel-seal.json').exists())
            request = s.child_stage_request(request, stage='seal-features',
                ui_context=request['ui_context'], execution_evidence=request['execution_evidence'])
            sealed = s.child_seal_features(request)
            # Known analytical flow is zero and confidence is constant .5.
            self.assertEqual(sealed['parity'][0]['max_flow_error_px'], 0.)
            self.assertFalse(s._deferred_label_slot(request).exists())
            write(fixture.base / 'generation.json', {'MODEL_FREE_TEST_FIXTURE': True})
            bound = s.bind_child_label_plan(request)
            labels_request = s.child_stage_request(bound, stage='labels',
                ui_context=bound['ui_context'], execution_evidence=bound['execution_evidence'])
            self.assertEqual(labels_request['stage_request_identity_sha256'], request['stage_request_identity_sha256'])
            self.assertEqual([row['stage'] for row in labels_request['stage_evidence_history']], list(s.CHILD_STAGE_NAMES))


if __name__ == '__main__':
    unittest.main()
