"""MODEL-FREE software fixtures, never captured/model-quality/training evidence.

Two synthetic array archives exercise genuine existing numerical validators.
The raw frozen() source verifier alone is doubled: no mesh/render/model executes.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext, contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import quality_adaptation_portable as raw
from bench import quality_adaptation_portable_prepare as p
from bench import quality_adaptation_scale as scale
from bench.quality_adaptation_data import CONVENTION, HOOKS, crop_rgb_hash
from bench.quality_adaptation_prepare import crops_from_retained_capture, project_supervision
from bench.quality_adaptation_refinement_states import state_rows

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return p.pin(path)


class Fixture:
    """Explicit test-only raw/extractor/render doubles; two real array states."""
    def __init__(self, directory, *, recipe=None):
        self.base = Path(directory).absolute()
        self.root = self.base / 'raw-root'
        self.root.mkdir()
        self.execution = self.base / 'execution'
        self.features = self.base / 'features'
        self.features.mkdir()
        evidence = write(self.base / 'fixture-evidence.json', {'MODEL_FREE_TEST_FIXTURE': True})
        self.spec = dict(schema='kaggle_raw_recipe_v1', iterations=[0], seed=42, targets=[1000, 2000, 4000], assets=[
            dict(object_id=str(n), physical_object_id='test-object:' + str(n), aliases=[],
                 mesh=dict(path=(self.base / f'poison-unopened-{n}.glb').as_posix(), bytes=100, sha256=hashlib.sha256(str(n).encode()).hexdigest())) for n in (101, 102, 103)])
        self.spec['assets'].append(dict(object_id='104', physical_object_id='keyboard', aliases=['keyboard'],
            mesh=dict(path=(self.base / 'poison-dev.glb').as_posix(), bytes=100, sha256='d' * 64)))
        if recipe is not None:
            self.spec = copy.deepcopy(recipe)
        self.plan = raw.prospective(self.spec)
        context_doc = raw.context_document(output_root=self.root, scratch_root=self.base / 'scratch',
            parity=dict(reference=evidence, candidate=evidence), device='cpu')
        context_raw = (json.dumps(context_doc) + '\n').encode()
        self.context = raw.PortableContext(context_raw, hashlib.sha256(context_raw).hexdigest())
        (self.root / 'context.json').write_bytes(context_raw)
        write(self.root / 'recipe.json', self.spec)
        write(self.root / 'scale-plan.json', self.plan)
        # Noncanonical whitespace is intentional: byte-for-byte split preservation.
        (self.root / 'split.json').write_text(json.dumps(self.plan['global_manifest']['split_manifest'], indent=3) + '\n\n')
        write(self.root / 'freeze.json', {'MODEL_FREE_TEST_FIXTURE': True})
        self.physical = raw.physical_rows(self.spec, self.plan)
        for frame in self.physical:
            if frame['split'] in ('train', 'validation'):
                for kind in ('render', 'bank', 'mask', 'seed'):
                    frame[kind] = dict(status='unavailable', failure_reason='fixture_observed_failure')
        self.captures = state_rows(dict(rows=self.physical))
        for row in self.captures:
            row['failure_reason'] = 'fixture_observed_failure' if row['split'] in ('train', 'validation') else 'sealed_' + row['split']
        self.arrays = {}
        for ordinal, split in enumerate(('train', 'validation')):
            row = next(r for r in self.captures if r['split'] == split)
            frame = next(r for r in self.physical if r['sequence_id'] == row['sequence_id'])
            render = {'status': 'available'}
            for kind in ('input', 'video', 'mesh'):
                path = self.root / ('fixture-' + split + '-' + kind + '.json')
                render[kind] = write(path, {'MODEL_FREE_TEST_FIXTURE': True, 'split': split, 'kind': kind})
            frame['render'] = render
            k = np.array([[280., 0., 139.5], [0., 280., 139.5], [0., 0., 1.]])
            captured = dict(query_rgb_crop=np.full((1, 3, 280, 280), .25 + ordinal * .25, np.float32),
                template_rgb=np.full((1, 3, 280, 280), .1, np.float32), rendered_depth_mm=np.full((280, 280), 1000., np.float32),
                rendered_mask=np.ones((280, 280), bool), crop_k=k, current_crop_camera_from_object_mm=np.eye(4))
            crops = crops_from_retained_capture(captured, preserve_float_rgb=True)
            geometry = dict(crop_from_orig=np.eye(4), crop_k=k, native_k=k,
                current_crop_camera_from_object_mm=np.eye(4), rendered_depth_mm=captured['rendered_depth_mm'], rendered_mask=captured['rendered_mask'],
                actual_flow=np.zeros((280, 280, 2), np.float32), actual_confidence=np.full((280, 280), .5, np.float32))
            self.root.joinpath('raw').mkdir(exist_ok=True)
            crop_path, capture_path = self.root / f'raw/crop-{ordinal}.npz', self.root / f'raw/capture-{ordinal}.npz'
            np.savez(crop_path, **crops)
            np.savez(capture_path, **geometry)
            row.pop('failure_reason')
            row.pop('unavailable_provenance')
            row.update(status='available', crops={**p.pin(crop_path), 'path': crop_path.name}, capture={**p.pin(capture_path), 'path': capture_path.name},
                query_rgb_sha256=crop_rgb_hash(crops['query_rgb']), native_calibration=k.tolist(),
                sampling_map=dict(integer_centers=280**2, max_native_pixel_difference=0.), pnp_state='fixture_unavailable_after_good_crop')
            self.arrays[row['id']] = (crops, geometry)
        write(self.root / 'physical-index.json', dict(rows=self.physical))
        write(self.root / 'input-plan.json', raw.capture_plan(self.context, self.physical))
        write(self.root / 'raw/capture-index.json', dict(rows=self.captures, training_ready=False))
        write(self.root / 'raw/capture-receipt.json', dict(schema='kaggle_raw_capture_receipt_v1', context_sha256=self.context.sha256,
            input_plan_sha256=p.sha256(self.root / 'input-plan.json'), fatal_error=None, cleanup_errors=[], physical_frames_requested=len(self.physical),
            refinement_states_requested=len(self.captures), annotation_inputs=[], test_payload_reads=0, training_ready=False))
        artifacts = {}
        for frame in self.physical:
            if frame['render']['status'] == 'available':
                for key in ('input', 'video', 'mesh'):
                    value = frame['render'][key]
                    artifacts[value['path']] = value
        for row in self.captures:
            if row['status'] == 'available':
                for key in ('crops', 'capture'):
                    value = p.pin(self.root / 'raw' / row[key]['path'])
                    artifacts[value['path']] = value
        for name in ('context.json', 'recipe.json', 'scale-plan.json', 'split.json', 'freeze.json', 'physical-index.json', 'input-plan.json', 'raw/capture-index.json', 'raw/capture-receipt.json'):
            artifacts[name] = p.pin(self.root / name)
        for name in ('render', 'banks', 'masks', 'capture', 'seal-capture'):
            value = write(self.root / ('stage-' + name + '.json'), dict(stage=name, context_sha256=self.context.sha256,
                status='completed', error=None, elapsed_seconds=.01, process_tree_peak_rss_bytes=1024,
                quota_scope=context_doc['quota_scope'], test_payload_reads=0, training_ready=False))
            if name != 'seal-capture':
                artifacts['stage-' + name + '.json'] = value
        self.seal = p.sealed(dict(schema='kaggle_raw_capture_seal_v1', context_sha256=self.context.sha256,
            prospective_plan_sha256=p.sha256(self.root / 'scale-plan.json'), artifacts=artifacts, physical_frames=self.physical,
            refinement_states=self.captures, physical_frames_requested=len(self.physical), refinement_states_requested=len(self.captures),
            available_crops=2, training_ready=False, final_execution_inventory=False, prospective_inputs_are_recipe_identity_only=True,
            annotation_inputs=[], test_payload_reads=0, capture_fatal_error=None, cleanup_errors=[]))
        self.seal_pin = write(self.root / 'raw-seal.json', self.seal)
        self.state = dict(context=context_doc, recipe=self.spec, plan=self.plan, physical=self.physical,
            captures=self.captures, seal=self.seal, root=self.root, closing_stage=p.pin(self.root / 'stage-seal-capture.json'))
        self.audit = dict(schema='kaggle_exposure_audit_v2', complete=True, scope='visualizeit_experimental_exposure_only',
            recipe_sha256=p.canonical(self.spec), evidence_descriptors=[evidence], assets=[dict(physical_object_id=a['physical_object_id'],
                mesh_sha256=a['mesh']['sha256'], aliases=a['aliases'], eligible=True, historical_roles=[]) for a in self.spec['assets']])
        self.audit_pin = write(self.base / 'exposure-audit.json', self.audit)

    def frozen(self, context):
        return self.context.document(), self.spec, self.plan, self.physical

    def prepare_features(self):
        p.execution_identity(self.state, self.execution)
        crop_pin = p.crop_manifest(self.state, self.execution)
        lock = p.read_json(ROOT / 'bench/runtime/adaptation-sources.json')
        recipe = dict(schema='frozen_gotrack_decoder_recipe_v1', source_lock_sha256=raw.SOURCE_LOCK_SHA256,
            source_files_sha256=p.canonical(lock['gotrack']['files']), upstream_revision=lock['gotrack']['revision'],
            crop_manifest_sha256=p.canonical(p.metadata(crop_pin)), hooks=list(HOOKS), annotation_inputs=[],
            normalization='upstream_DinoFeatureExtractor_RGB_0_1', rgb_encoding='uint8_div255_or_exact_float32_0_1_v1',
            producer_sha256=lock['gotrack']['files']['bench/quality_adaptation_prepare.py'], device='cpu')
        write(self.features / 'recipe.json', recipe)
        write(self.features / 'original-recipe.json', recipe)
        original = []
        for ordinal, row in enumerate(self.captures):
            if row['id'] not in self.arrays:
                continue
            crops, geometry = self.arrays[row['id']]
            # Analytical identity baseline: zero flow and constant .5 confidence.
            # Existing parity validator checks retained HWC vs extracted CHW.
            values = {f'tokens_{hook}': np.full((400, 768), ordinal + 1, np.float32) for hook in HOOKS}
            values.update({k: crops[k] for k in ('template_mask', 'template_xyz_m', 'query_K')})
            values.update(baseline_flow=np.zeros((2, 280, 280), np.float32), baseline_confidence=np.full((280, 280), .5, np.float32))
            path = self.features / f'feature-{ordinal}.npz'
            np.savez(path, **values)
            original.append(dict(id=row['id'], status='available', crops=row['crops'], query_rgb_sha256=row['query_rgb_sha256'], features={**p.pin(path), 'path': path.name}))
        index = dict(recipe_sha256=p.canonical(recipe), status='completed', elapsed_seconds=.1, rows=original)
        write(self.features / 'original-index.json', index)
        by_id = {r['id']: r for r in original}
        final = {**index, 'rows': [by_id.get(r['id'], dict(id=r['id'], status='unavailable', failure_reason=r['failure_reason'])) if r['id'] not in by_id else by_id[r['id']] for r in self.captures if r['split'] != 'dev']}
        write(self.features / 'index.json', final)
        write(self.features / 'extraction-binding.json', dict(schema='kaggle_extraction_binding_v2', original_recipe=p.pin(self.features / 'original-recipe.json'),
            original_index=p.pin(self.features / 'original-index.json'), final_recipe=p.pin(self.features / 'recipe.json'), final_index=p.pin(self.features / 'index.json'),
            crop_manifest=crop_pin, execution_plan=p.pin(self.execution / 'execution-plan.json'), raw_seal=self.seal_pin,
            unavailable_added=[r['id'] for r in final['rows'] if r['id'] not in by_id], dev_rows=[r for r in self.captures if r['split'] == 'dev'],
            successful_original_rows_unchanged=True))
        return p.seal_features(self.state, self.execution, self.features)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)

    def test_exposure_preflight_never_opens_assets(self):
        result = p.bounded_recipe(self.fixture.spec, self.fixture.audit)
        self.assertEqual(result['active_states'], 48)
        self.assertTrue(all(not r['request_met_by_inventory'] for r in result['targets']))

    def test_incomplete_audit_rejected(self):
        self.fixture.audit['complete'] = False
        with self.assertRaisesRegex(ValueError, 'exposure audit'):
            p.bounded_recipe(self.fixture.spec, self.fixture.audit)

    def test_historical_roles_reassignment_rejected(self):
        for role in ('validation', 'test', 'dev', 'inspected_development'):
            self.fixture.audit['assets'][0]['historical_roles'] = [role]
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'historically exposed'):
                p.bounded_recipe(self.fixture.spec, self.fixture.audit)

    def test_no_three_iteration_multiplication(self):
        self.fixture.spec['iterations'] = [0, 1, 3]
        self.fixture.audit['recipe_sha256'] = p.canonical(self.fixture.spec)
        with self.assertRaisesRegex(ValueError, 'iterations'):
            p.bounded_recipe(self.fixture.spec, self.fixture.audit)

    def test_raw_metadata_preserves_failures_and_no_test_dev_payload(self):
        with patch.object(raw, 'frozen', side_effect=self.fixture.frozen):
            result = p.raw_metadata(self.fixture.context, self.fixture.seal_pin)
        self.assertEqual(len(result['captures']), 96)
        self.assertEqual(sum(r['status'] == 'available' for r in result['captures']), 2)

    def test_last_sealed_row_rejected_before_payload_hash(self):
        self.fixture.captures[-1]['crops'] = dict(path='POISON', bytes=1, sha256='0' * 64)
        self.fixture.seal['refinement_states'] = self.fixture.captures
        self.fixture.seal = p.sealed({k: v for k, v in self.fixture.seal.items() if k != 'seal_sha256'})
        self.fixture.seal_pin = write(self.fixture.root / 'raw-seal.json', self.fixture.seal)
        write(self.fixture.root / 'raw/capture-index.json', dict(rows=self.fixture.captures))
        with patch.object(raw, 'frozen', side_effect=self.fixture.frozen), patch.object(p, 'verify_pin', wraps=p.verify_pin) as verified:
            with self.assertRaisesRegex(ValueError, 'sealed TEST/dev'):
                p.raw_metadata(self.fixture.context, self.fixture.seal_pin)
            self.assertEqual(len(verified.call_args_list), 1)  # raw seal metadata only

    def test_actual_raw_seal_producer_roundtrip_with_postseal_receipt(self):
        # Only the cloud stage guard/resource policy are doubled. The genuine raw
        # seal producer writes its actual receipt schema and artifact registry.
        (self.fixture.root / 'raw-seal.json').unlink()
        closing = self.fixture.root / 'stage-seal-capture.json'
        closing_bytes = closing.read_bytes()
        closing.unlink()
        with patch.object(raw, 'frozen', side_effect=self.fixture.frozen), patch.object(raw, '_require_stage'), patch.object(raw, 'resource_check'):
            produced = raw.seal_capture(self.fixture.context)
        closing.write_bytes(closing_bytes)  # genuine stage finally happens later
        self.assertNotIn('stage-seal-capture.json', produced['artifacts'])
        self.assertNotIn('rows', p.read_json(self.fixture.root / 'raw/capture-receipt.json'))
        with patch.object(raw, 'frozen', side_effect=self.fixture.frozen):
            result = p.raw_metadata(self.fixture.context, p.pin(self.fixture.root / 'raw-seal.json'))
        self.assertEqual(result['seal'], produced)

    def test_execution_changes_only_input_digest_and_preserves_split_bytes(self):
        plan = p.execution_identity(self.fixture.state, self.fixture.execution)
        self.assertEqual((self.fixture.root / 'split.json').read_bytes(), (self.fixture.execution / 'split.json').read_bytes())
        self.assertEqual(p.verify_execution(self.fixture.state, self.fixture.execution), plan)
        for before, after in zip(self.fixture.plan['global_manifest']['records'], plan['global_manifest']['records']):
            self.assertNotEqual(before['inputs_sha256'], after['inputs_sha256'])
            self.assertEqual({k: v for k, v in before.items() if k != 'inputs_sha256'}, {k: v for k, v in after.items() if k != 'inputs_sha256'})

    def test_forged_provenance_resealed_plan_rejected(self):
        p.execution_identity(self.fixture.state, self.fixture.execution)
        bridge = p.read_json(self.fixture.execution / 'identity-bridge.json')
        value = bridge['rows'][0]['provenance']
        doc = p.metadata(value)
        doc['query_rgb_sha256'] = 'f' * 64
        write(Path(value['path']), doc)
        with self.assertRaises(ValueError):
            p.verify_execution(self.fixture.state, self.fixture.execution)

    def test_real_numerical_prelabel_seal(self):
        seal = self.fixture.prepare_features()
        self.assertEqual(len(seal['parity']), 2)
        # Analytically identical zero flows and constant confidence have zero error.
        self.assertEqual(seal['parity'][0]['max_flow_error_px'], 0.)
        self.assertEqual(p.verify_prelabel(self.fixture.state, self.fixture.execution, self.fixture.features), seal)

    def test_parity_mutation_rejected_even_with_updated_feature_descriptor(self):
        self.fixture.prepare_features()
        from bench.quality_adaptation_expanded import _check_parity
        mask = np.ones((280, 280), bool)
        actual = dict(actual_flow=np.zeros((280, 280, 2), np.float32), actual_confidence=np.full((280, 280), .5, np.float32))
        feature = dict(baseline_flow=np.full((2, 280, 280), .01, np.float32), baseline_confidence=np.full((280, 280), .5, np.float32))
        with self.assertRaisesRegex(ValueError, 'outputs differ'):
            _check_parity(actual, feature, mask)

    def test_label_plan_inaccessible_before_seal(self):
        with patch.object(p, 'cloud_guard'), patch.object(p, 'verify_prelabel', side_effect=ValueError('numerical seal absent')), patch.object(p, 'metadata') as opened:
            with self.assertRaisesRegex(ValueError, 'numerical seal absent'):
                p.labels(dict(features_root=str(self.fixture.features), label_plan={'path': 'POISON'}), self.fixture.state, self.fixture.execution)
            opened.assert_not_called()

    def test_atomic_publication_does_not_clobber(self):
        path = self.fixture.base / 'receipt.json'
        p.atomic_json(path, dict(first=True))
        with self.assertRaises(FileExistsError):
            p.atomic_json(path, dict(second=True))
        self.assertEqual(p.read_json(path), dict(first=True))
        self.assertFalse(list(path.parent.glob('.receipt.json.*.tmp')))

    def test_production_extract_uses_original_cuda_bootstrap_and_supervisor(self):
        from bench.quality_adaptation_larger import _CUDA_BOOTSTRAP
        class AdapterDouble:
            def clean_environment(self):
                return {}
            def supervise(self, argv, **kwargs):
                self.argv, self.kwargs = argv, kwargs
                return dict(status='incomplete', returncode=3)
        adapter = AdapterDouble()
        request = dict(python='PINNED_PYTHON', working_root=str(self.fixture.base), limits=dict(max_output_bytes=123))
        with patch.object(p, 'cloud_guard'), patch.object(p, 'source_runtime', return_value=(adapter, ROOT, {})), patch.object(p, 'quota_check', return_value={}), patch.object(p, 'preparation_evidence', return_value=dict(gpu=dict(uuid='GPU-TEST-DOUBLE'))):
            p.production_run(request, 'bench.quality_adaptation_prepare', ['--manifest', 'ACTUAL_MANIFEST'],
                output_root=self.fixture.features, log=self.fixture.base / 'fixture.log', cuda=True)
        self.assertIn(_CUDA_BOOTSTRAP, adapter.argv)
        self.assertEqual(adapter.kwargs['cwd'], ROOT)
        self.assertEqual(adapter.kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(adapter.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-TEST-DOUBLE')

    def test_global_extract_augments_accounting_preserving_original_receipts(self):
        self.fixture.prepare_features()
        original_recipe = (self.fixture.features / 'original-recipe.json').read_bytes()
        original_index = (self.fixture.features / 'original-index.json').read_bytes()
        destination = self.fixture.base / 'new-features'
        (self.fixture.root / 'raw/v2-crops.json').unlink()
        calls = []
        def original_extractor_double(request, module, args, **kwargs):
            calls.append((module, args))
            destination.mkdir()
            (destination / 'recipe.json').write_bytes(original_recipe)
            (destination / 'index.json').write_bytes(original_index)
            return dict(status='completed', returncode=0)
        request = dict(features_root=str(destination), limits=dict(max_seconds=30))
        with patch.object(p, 'production_run', side_effect=original_extractor_double), patch.object(p, 'preparation_lock', return_value=nullcontext()):
            p.extract(request, self.fixture.state, self.fixture.execution)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'bench.quality_adaptation_prepare')
        self.assertEqual((destination / 'original-recipe.json').read_bytes(), original_recipe)
        self.assertEqual((destination / 'original-index.json').read_bytes(), original_index)
        final = p.read_json(destination / 'index.json')
        self.assertEqual(len(final['rows']), 72)
        self.assertEqual(sum(r['status'] == 'available' for r in final['rows']), 2)
        self.assertEqual(len(p.read_json(destination / 'extraction-binding.json')['dev_rows']), 24)

    def test_separate_label_orchestration_with_declared_renderer_double(self):
        from bench import quality_adaptation_larger as larger
        from bench import quality_adaptation_smoke as smoke
        from bench import quality_gotrack, quality_runner
        from bench.quality_adaptation_manifest import assemble
        self.fixture.prepare_features()
        generations = []
        for frame in self.fixture.physical:
            if frame['render']['status'] == 'available':
                generations.append(dict(sequence_id=frame['sequence_id'], generation=write(self.fixture.base / (frame['sequence_id'] + '.json'), {'MODEL_FREE_RENDER_DOUBLE': True})))
        label_plan = write(self.fixture.base / 'label-plan.json', dict(schema='kaggle_label_plan_v2', split_manifest=p.pin(self.fixture.root / 'split.json'), generations=generations))
        calls = []
        class RendererDouble:
            vertices_m = np.zeros((8, 3))
            def __init__(self, *args, **kwargs):
                calls.append('renderer')
            def close(self):
                calls.append('close')
        def render_double(*args, **kwargs):
            return np.eye(4), np.ones((280, 280), np.float32), np.ones((280, 280), bool)
        def assembler_double(request, module, args, **kwargs):
            self.assertEqual(module, 'bench.quality_adaptation_manifest')
            assemble(self.fixture.features, self.fixture.features / 'labels.json', self.fixture.features / 'manifest.json')
            return dict(returncode=0, status='completed')
        request = dict(features_root=str(self.fixture.features), label_plan=label_plan, limits=dict(max_seconds=60))
        with patch.object(p, 'cloud_guard'), patch.object(p, 'source_runtime', return_value=(object(), ROOT, {})), patch.object(p, 'quota_check'), patch.object(p, 'preparation_lock', return_value=nullcontext()), patch.object(larger, 'configure_torch'), patch.object(larger, 'cleanup_device'), patch.object(larger, '_validate_generation', return_value=np.eye(4)), patch.object(quality_gotrack, 'TexturedRenderer', RendererDouble), patch.object(quality_runner, 'read_input', return_value=dict(asset='FIXTURE_MESH', object_id='101')), patch.object(smoke, 'render_query_labels', side_effect=render_double), patch.object(p, 'production_run', side_effect=assembler_double):
            receipt = p.labels(request, self.fixture.state, self.fixture.execution)
        inventory = p.read_json(self.fixture.features / 'labels.json')['rows']
        self.assertEqual(len(inventory), 72)
        self.assertEqual(sum(r['status'] == 'available' for r in inventory), 2)
        self.assertEqual(calls, ['renderer', 'close', 'renderer', 'close'])
        self.assertFalse(receipt['labels_entered_inference'])

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw_json in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                p.loads(raw_json)


# MODEL-FREE descriptor fixtures copied from scientific metadata audit147,
# SHA6848c9201f938aa435cbe813dd48a1d7904c7e2b933868b6707f7099d1e1b297.
# No mesh content accompanies these descriptors. Expected roles/counts come
# from that audit's scientific_contract_judgment, using the unchanged planner.
KNOWN_ROLE_METADATA = (
    (1, 1270036, '605ae0f1ccb3529b0d362d9637447ca0a54b9b33cc2e93d019a43687e5093711', '232501673989606', 'holder_black', 'train'),
    (4, 1655024, '03554445ec87ee6dc62ae789015c51fa7a4f4f6802f3b87bc9ce090775620854', '225397651484143', 'spoon_wooden', 'validation'),
    (6, 1708408, 'bc8d0eb8f1525ee9ab39b00a197980f14ffce1b7d83531373e62696f28c18852', '270231216246839', 'spatula_red', 'train'),
    (7, 4725172, 'a0f8ee0675b1fb06521af00a8e384666f2c2a124797a5ff2228f71a160a9f6ac', '228358276546933', 'coffee_pot', 'test'),
    (10, 6647852, '321cb27bcaa12c7abf521208c8cd8e0df3f21790ad3379ec36b2166476737aed', '98604936546412', 'can_soup', 'train'),
    (13, 3060680, 'd7812064269510c8b762e5ee6f8202402061f9088bc514b35317c5cb7a3006dd', '261746112525368', 'bottle_mustard', 'train'),
    (17, 5975664, 'e1ce5b89e7021b9c9cc6f5a63e895c6c5fc1519fdaedb213d3199b74d48daacc', '125863066770940', 'carton_milk', 'train'),
    (18, 8215676, '7c6d742b80b6df24b8de08b5f866522a452ddd2151da53073da577d5b4f6fefc', '204462113746498', 'carton_oj', 'train'),
    (20, 4972976, 'bf0e99e2763e34bf46f0ec3cf27b5c31a62d21ded54190f4faacaae1797e3540', '253405647833885', 'food_waffles', 'train'),
    (21, 3179560, 'b998ffc8932f1cacac5259ff8d9bbe8b64eb2f112d29ec8263ebb5f187cf0065', '96945373046044', 'food_vegetables', 'train'),
    (23, 741364, 'eb6ee406897f946d14da9357d94cbf533e9e7ad663233023df750af4b6b37510', '111305855142671', 'aria_small', 'train'),
    (29, 1602952, 'f5dbb2634e430b8e7b963b5f82d1b9e6057bcdefd442185ce9ec5c59b76bc312', '258906041248094', 'whiteboard_eraser', 'train'),
    (31, 807492, '946516d7b944a2b729006e367185b971b80b0bdd2f1136c0cb7dab5f2d114a87', '265826671143948', 'mouse', 'train'),
    (32, 3754072, '6fccaab7762c735141a443e16c7b332e3fd6d5e494987a07258e38bbf2571387', '238686662724712', 'whiteboard_marker', 'train'),
    (33, 6243472, 'a4fcc74e8a6f74302f08ecfe2a8228e9101edf263f38f88e31fbd9388a885b85', '249541253457812', 'dvd_remote', 'train'),
)


def known_role_fixture(base, count=15):
    selected = [row for row in KNOWN_ROLE_METADATA if count == 15 or row[0] not in (17, 31, 33)]
    recipe = dict(schema='kaggle_raw_recipe_v1', iterations=[0], seed=521, targets=[1000, 2000, 4000], assets=[])
    audit = dict(schema='kaggle_exposure_audit_v2', complete=True, scope='visualizeit_experimental_exposure_only',
        completeness_basis='selected_known_roles_over_pinned_evidence', frozen_seed=521, assets=[],
        evidence_descriptors=[write(Path(base) / 'known-role-fixture-evidence.json', dict(MODEL_FREE_CONTRACT_FIXTURE=True,
            source_audit_sha256='6848c9201f938aa435cbe813dd48a1d7904c7e2b933868b6707f7099d1e1b297'))])
    for number, size, mesh, original, name, role in selected:
        obj, aliases = f'hot3d:obj_{number:06d}', [name, original]
        recipe['assets'].append(dict(object_id=str(number), physical_object_id=obj, aliases=aliases,
            mesh=dict(path=f'/kaggle/working/visualizeit-known-development-inputs-v1/assets/obj_{number:06d}.glb', bytes=size, sha256=mesh)))
        item = dict(physical_object_id=obj, mesh_sha256=mesh, aliases=aliases, eligible=True, historical_roles=[role])
        item[dict(train='allow_prior_train', validation='allow_prior_validation_reuse', test='retain_prior_test_metadata_only')[role]] = True
        audit['assets'].append(item)
    audit['recipe_sha256'] = p.canonical(recipe)
    return recipe, audit


@contextmanager
def forbid_payload_opens():
    original_open = Path.open
    def trapped(path, *args, **kwargs):
        if path.suffix.lower() in ('.glb', '.npz', '.pt', '.pth', '.mp4', '.png', '.jpg', '.jpeg') or 'POISON' in path.name:
            raise AssertionError('MODEL_FREE payload open trap: ' + str(path))
        return original_open(path, *args, **kwargs)
    with patch.object(Path, 'open', trapped):
        yield


class KnownRoleRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).absolute()
        self.recipe, self.audit = known_role_fixture(self.base)

    def test_known12_and_known15_use_actual_unchanged_planner_without_payloads(self):
        for count, train, active, envelope in ((12, 240, 264, 4496293888), (15, 312, 336, 5704253440)):
            recipe, audit = known_role_fixture(self.base, count)
            with self.subTest(count=count), forbid_payload_opens():
                result = p.bounded_recipe(recipe, audit)
                self.assertEqual(result['plan'], raw.prospective(recipe))
                self.assertEqual(result['plan']['counts']['train']['refinement_states'], train)
                self.assertEqual(result['plan']['counts']['validation']['refinement_states'], 24)
                self.assertEqual(result['plan']['counts']['test']['refinement_states'], 24)
                self.assertEqual(result['active_states'], active)
                self.assertEqual(result['feature_tree_worst_case_bytes'], envelope)
                states = result['plan']['global_manifest']['records']
                self.assertTrue(all(r['split'] == 'validation' for r in states if r['physical_object_id'] == 'hot3d:obj_000004'))
                self.assertTrue(all(r['split'] == 'test' for r in states if r['physical_object_id'] == 'hot3d:obj_000007'))
                self.assertFalse(result['exposure_disclosure']['independent_validation_claim'])
                self.assertFalse(result['exposure_disclosure']['independent_test_claim'])
                self.assertFalse(result['exposure_disclosure']['execution_admission'])
                self.assertTrue(all(not x['request_met_by_inventory'] for x in result['targets']))

    def test_validation_synonyms_normalize_to_same_disclosed_role(self):
        self.audit['assets'][1]['historical_roles'] = ['val', 'VALIDATION']
        with forbid_payload_opens():
            result = p.bounded_recipe(self.recipe, self.audit)
        self.assertEqual(result['exposure_disclosure']['reused_development_validation_objects'], ['hot3d:obj_000004'])

    def test_role_moves_reject_before_payload_access(self):
        for number, role in ((4, 'train'), (4, 'test'), (7, 'train'), (7, 'validation'), (1, 'validation'), (1, 'test')):
            audit = copy.deepcopy(self.audit)
            item = next(r for r in audit['assets'] if r['physical_object_id'] == f'hot3d:obj_{number:06d}')
            item.update(historical_roles=[role], allow_prior_train=True, allow_prior_validation_reuse=True, retain_prior_test_metadata_only=True)
            with self.subTest(number=number, role=role), forbid_payload_opens(), self.assertRaisesRegex(ValueError, 'cannot be reassigned'):
                p.bounded_recipe(self.recipe, audit)

    def test_all_explicit_role_flags_required_including_last_train_row(self):
        for number, flag in ((33, 'allow_prior_train'), (4, 'allow_prior_validation_reuse'), (7, 'retain_prior_test_metadata_only')):
            audit = copy.deepcopy(self.audit)
            item = next(r for r in audit['assets'] if r['physical_object_id'] == f'hot3d:obj_{number:06d}')
            item.pop(flag)
            with self.subTest(number=number), forbid_payload_opens(), self.assertRaisesRegex(ValueError, 'requires explicit'):
                p.bounded_recipe(self.recipe, audit)

    def test_mixed_unknown_development_and_missing_known_roles_reject(self):
        for roles in (['train', 'validation'], ['test', 'train'], ['dev'], ['inspected_development'], [], ['unknown']):
            audit = copy.deepcopy(self.audit)
            audit['assets'][-1]['historical_roles'] = roles
            with self.subTest(roles=roles), forbid_payload_opens(), self.assertRaises(ValueError):
                p.bounded_recipe(self.recipe, audit)

    def test_incomplete_schema_seed_iteration_recipe_and_identity_mismatch_reject(self):
        for field, value in (('complete', False), ('schema', 'foreign'), ('recipe_sha256', '0' * 64),
                             ('completeness_basis', 'unknown_history_is_complete'), ('frozen_seed', 522)):
            audit = copy.deepcopy(self.audit)
            audit[field] = value
            with self.subTest(field=field), forbid_payload_opens(), self.assertRaises(ValueError):
                p.bounded_recipe(self.recipe, audit)
        for field, value in (('seed', 522), ('iterations', [0, 1, 3])):
            recipe, audit = copy.deepcopy(self.recipe), copy.deepcopy(self.audit)
            recipe[field] = value
            audit['recipe_sha256'] = p.canonical(recipe)
            with self.subTest(field=field), forbid_payload_opens(), self.assertRaises(ValueError):
                p.bounded_recipe(recipe, audit)
        for field, value in (('aliases', ['wrong_alias']), ('mesh_sha256', '0' * 64), ('eligible', False)):
            audit = copy.deepcopy(self.audit)
            audit['assets'][-1][field] = value
            with self.subTest(field=field), forbid_payload_opens(), self.assertRaises(ValueError):
                p.bounded_recipe(self.recipe, audit)

    def test_conflicting_alias_and_duplicate_mesh_declarations_reject(self):
        for kind in ('alias', 'mesh', 'physical'):
            recipe, audit = copy.deepcopy(self.recipe), copy.deepcopy(self.audit)
            if kind == 'alias':
                # A TRAIN declaration may not alias the known preserved VAL4.
                recipe['assets'][-1]['aliases'].append('hot3d:obj_000004')
                audit['assets'][-1]['aliases'] = recipe['assets'][-1]['aliases'][:]
            elif kind == 'mesh':
                recipe['assets'][-1]['mesh']['sha256'] = recipe['assets'][1]['mesh']['sha256']
                audit['assets'][-1]['mesh_sha256'] = recipe['assets'][1]['mesh']['sha256']
            else:
                recipe['assets'][-1]['physical_object_id'] = recipe['assets'][1]['physical_object_id']
            audit['recipe_sha256'] = p.canonical(recipe)
            with self.subTest(kind=kind), forbid_payload_opens(), self.assertRaises(ValueError):
                p.bounded_recipe(recipe, audit)


if __name__ == '__main__':
    unittest.main()
